import Mathlib

namespace max_area_centrally_symmetric_polygon_in_triangle_l604_604980

-- Define the structure of a triangle and the midpoints of its sides
structure Triangle (α : Type) [LinearOrderedField α] :=
  (A B C : α × α)

structure MidpointTriangle (α : Type) [LinearOrderedField α] :=
  (A₁ B₁ C₁ : α × α)

def midpoint {α : Type} [LinearOrderedField α] (p₁ p₂ : α × α) : α × α :=
  ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)

def make_midpoint_triangle {α : Type} [LinearOrderedField α] (T : Triangle α) : MidpointTriangle α :=
  { A₁ := midpoint T.B T.C,
    B₁ := midpoint T.C T.A,
    C₁ := midpoint T.A T.B }

-- Definition of centrally symmetric polygon and the condition for it lying inside the initial triangle
def centrally_symmetric_max_area_polygon_exists {α : Type} [LinearOrderedField α] (T : Triangle α) (P : Set (α × α)) : Prop :=
  ∃ M : Set (α × α), central_symmetric M ∧ bounded_by T M ∧ max_area M = area (midpoint_hexagon T)

-- Define what a central symmetric set is
def central_symmetric {α : Type} [LinearOrderedField α] (M : Set (α × α)) : Prop :=
  ∀ (x y : α × α), (x ∈ M) → ((-x.1, -x.2) ∈ M)

-- Define what it means for a set to be bounded by a triangle
def bounded_by {α : Type} [LinearOrderedField α] (T : Triangle α) (M : Set (α × α)) : Prop :=
  ∀ (x : α × α), x ∈ M → is_inside_triangle T x

-- Define an area calculation
def max_area {α : Type} [LinearOrderedField α] (M : Set (α × α)) : α :=
  sorry -- This would involve more detailed geometric calculations

-- Define the midpoint hexagon specifically
def midpoint_hexagon {α : Type} [LinearOrderedField α] (T : Triangle α) : Set (α × α) :=
  sorry -- Representation of the hexagon constructed from midpoints dividing each side into three equal parts

-- Main theorem statement
theorem max_area_centrally_symmetric_polygon_in_triangle {α : Type} [LinearOrderedField α] (T : Triangle α) :
  centrally_symmetric_max_area_polygon_exists T (midpoint_hexagon T) :=
sorry

end max_area_centrally_symmetric_polygon_in_triangle_l604_604980


namespace false_propositions_l604_604597

variable {A B : Finset ℕ}

-- Proposition A: A ∩ B = ∅ ↔ card(A ∪ B) = card(A) + card(B)
def prop_A : Prop := A ∩ B = ∅ ↔ A.card + B.card = (A ∪ B).card

-- Proposition B: A ⊆ B ↔ card(A) ≤ card(B)
def prop_B : Prop := A ⊆ B ↔ A.card ≤ B.card

-- Proposition C: If A ⊆ B, then card(A) ≤ card(B) - 1
def prop_C : Prop := (A ⊆ B → A.card ≤ B.card - 1)

-- Proposition D: A = B ↔ card(A) = card(B)
def prop_D : Prop := A = B ↔ A.card = B.card

theorem false_propositions (A B : Finset ℕ) :
  (¬ prop_B) ∧ (¬ prop_C) ∧ (¬ prop_D) := by
  sorry

end false_propositions_l604_604597


namespace probability_same_color_is_correct_l604_604857

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l604_604857


namespace tanya_efficiency_greater_sakshi_l604_604555

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l604_604555


namespace max_a_value_l604_604458

theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 - 2 * x - 3 > 0) →
  (¬ (∀ x : ℝ, x^2 - 2 * x - 3 > 0 → x < a)) →
  a = -1 :=
by
  sorry

end max_a_value_l604_604458


namespace tanya_efficiency_greater_sakshi_l604_604556

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l604_604556


namespace domain_log_function_l604_604230

theorem domain_log_function :
  {x : ℝ | 1 < x ∧ x < 3 ∧ x ≠ 2} = {x : ℝ | (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1)} :=
sorry

end domain_log_function_l604_604230


namespace original_triangle_area_l604_604225

theorem original_triangle_area (area_of_new_triangle : ℝ) (side_length_ratio : ℝ) (quadrupled : side_length_ratio = 4) (new_area : area_of_new_triangle = 128) : 
  (area_of_new_triangle / side_length_ratio ^ 2) = 8 := by
  sorry

end original_triangle_area_l604_604225


namespace integral_area_of_enclosed_shape_l604_604217

theorem integral_area_of_enclosed_shape :
  ∫ x in 0..1, (x^2 - x^3) = 1 / 12 :=
  sorry

end integral_area_of_enclosed_shape_l604_604217


namespace machine_loan_repaid_in_5_months_l604_604328

theorem machine_loan_repaid_in_5_months :
  ∀ (loan cost selling_price tax_percentage products_per_month profit_per_product months : ℕ),
    loan = 22000 →
    cost = 5 →
    selling_price = 8 →
    tax_percentage = 10 →
    products_per_month = 2000 →
    profit_per_product = (selling_price - cost - (selling_price * tax_percentage / 100)) →
    (products_per_month * months * profit_per_product) ≥ loan →
    months = 5 :=
by
  intros loan cost selling_price tax_percentage products_per_month profit_per_product months
  sorry

end machine_loan_repaid_in_5_months_l604_604328


namespace part_a_part_b_l604_604656

-- Definitions and Conditions
def median := 10
def mean := 6

-- Part (a): Prove that a set with 7 numbers cannot satisfy the given conditions
theorem part_a (n1 n2 n3 n4 n5 n6 n7 : ℕ) (h1 : median ≤ n1) (h2 : median ≤ n2) (h3 : median ≤ n3) (h4 : median ≤ n4)
  (h5 : 1 ≤ n5) (h6 : 1 ≤ n6) (h7 : 1 ≤ n7) (hmean : (n1 + n2 + n3 + n4 + n5 + n6 + n7) / 7 = mean) :
  false :=
by
  sorry

-- Part (b): Prove that the minimum size of the set where number of elements is 2n + 1 and n is a natural number, is at least 9
theorem part_b (n : ℕ) (h_sum_geq : ∀ (s : Finset ℕ), ((∀ x ∈ s, x >= median) ∧ ∃ t : Finset ℕ, t ⊆ s ∧ (∀ x ∈ t, x >= 1) ∧ s.card = 2 * n + 1) → s.sum >= 11 * n + 10) :
  n ≥ 4 :=
by
  sorry

-- Lean statements defined above match the problem conditions and required proofs

end part_a_part_b_l604_604656


namespace find_tf_f_l604_604137

def t (x : ℝ) : ℝ := Real.sqrt (3 * x + 1)
def f (x : ℝ) : ℝ := 5 - t x

theorem find_tf_f (x : ℝ) : t (f 5) = 2 := 
by
  sorry

end find_tf_f_l604_604137


namespace average_weight_of_abc_l604_604218

variables (A B C : ℝ)

-- Conditions
def condition1 : Prop := (A + B) / 2 = 40
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- Theorem statement
theorem average_weight_of_abc : condition1 A B C → condition2 A B C → condition3 A B C → (A + B + C) / 3 = 45 :=
by
  intros h1 h2 h3
  sorry

end average_weight_of_abc_l604_604218


namespace volume_of_extended_parallelepiped_and_sum_l604_604749

theorem volume_of_extended_parallelepiped_and_sum
  (m n p : ℕ) (h_rel_prime : Nat.coprime n p)
  (box_dimensions : (ℕ × ℕ × ℕ)) 
  (h_box_dims : box_dimensions = (4, 5, 6))
  (h_m : m = 804)
  (h_n : n = 77)
  (h_p : p = 3)
  : ∃ vol : ℝ, vol = (804 + 77 * Real.pi) / 3 ∧ m + n + p = 884 :=
by
  sorry

end volume_of_extended_parallelepiped_and_sum_l604_604749


namespace probability_same_color_is_correct_l604_604858

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l604_604858


namespace distance_between_points_l604_604418

variable (A B : ℝ × ℝ) (dist : ℝ)

def distance_formula (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem distance_between_points :
  A = (2, 1) → B = (5, -1) → dist = distance_formula A B → dist = real.sqrt 13 :=
by
  intro hA hB hdist
  rw [hA, hB] at hdist
  sorry

end distance_between_points_l604_604418


namespace correct_option_D_l604_604467

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k / x

theorem correct_option_D (x : ℝ) (h1 : x < -1) : 
  let k := -2 in 
  let y := f k x in 
  ∃ ε > 0, ∀ δ > 0, x + δ < -1 → f k (x + δ) > y := 
sorry

end correct_option_D_l604_604467


namespace james_hives_l604_604079

-- Define all conditions
def hive_honey : ℕ := 20  -- Each hive produces 20 liters of honey
def jar_capacity : ℕ := 1/2  -- Each jar holds 0.5 liters
def jars_needed : ℕ := 100  -- James needs 100 jars for half the honey

-- Translate to Lean statement
theorem james_hives (hive_honey jar_capacity jars_needed : ℕ) :
  (hive_honey = 20) → 
  (jar_capacity = 1 / 2) →
  (jars_needed = 100) →
  (∀ hives : ℕ, (hives * hive_honey = 200) → hives = 5) :=
by
  intros Hhoney Hjar Hjars
  intros hives Hprod
  sorry

end james_hives_l604_604079


namespace difference_of_squares_l604_604640

theorem difference_of_squares (a b : ℕ) (h1: a = 630) (h2: b = 570) : a^2 - b^2 = 72000 :=
by
  sorry

end difference_of_squares_l604_604640


namespace favorite_sandwiches_l604_604999

theorem favorite_sandwiches (total_students pizza_percentage cookies_percentage burgers_percentage salads_percentage : ℝ) 
  (h1 : total_students = 200) 
  (h2 : cookies_percentage = 25)
  (h3 : pizza_percentage = 30) 
  (h4 : burgers_percentage = 35) 
  (h5 : salads_percentage + pizza_percentage + cookies_percentage + burgers_percentage = 100) :
  let sandwiches_percentage := 100 - (pizza_percentage + cookies_percentage + burgers_percentage + salads_percentage) in
  (total_students * (sandwiches_percentage / 100) = 20) := 
by 
  -- Placeholder for proof
  sorry

end favorite_sandwiches_l604_604999


namespace intersection_A_B_eq_C_l604_604098

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l604_604098


namespace hyperbola_focus_coordinates_l604_604833

theorem hyperbola_focus_coordinates :
  (∃ h k a b : ℝ, 
     (a > 0) ∧ (b > 0) ∧ 
     (∀ x y : ℝ, ((x - h) ^ 2 / a^2) - ((y + k) ^ 2 / b^2) = 1) ∧ -- Given hyperbola
     (h, k, a, b) = (1, -8, 7, 3)) →
  ∃ c : ℝ, c = √(7^2 + 3^2) ∧ -- Distance to the foci
  (∃ f : ℝ × ℝ, f = (1 - c, -8)) := -- Coordinates of the focus with the smaller x-coordinate
sorry

end hyperbola_focus_coordinates_l604_604833


namespace car_b_speed_l604_604628

theorem car_b_speed
  (v_A v_B : ℝ) (d_A d_B d : ℝ)
  (h1 : v_A = 5 / 3 * v_B)
  (h2 : d_A = v_A * 5)
  (h3 : d_B = v_B * 5)
  (h4 : d = d_A + d_B)
  (h5 : d_A = d / 2 + 25) :
  v_B = 15 := 
sorry

end car_b_speed_l604_604628


namespace part_a_part_b_l604_604089

-- Define the sequence x_n
def x : ℕ → ℝ
| 0       := 1
| (2 * k)   := (4 * x (2 * k - 1) - x (2 * k - 2))^2
| (2 * k + 1) := abs (x (2 * k) / 4 - (k:ℝ)^2)

theorem part_a :
  x 2022 = (2020 : ℝ)^4 :=
sorry

theorem part_b :
  ∃ infinitely_many k, 2021 ∣ x (2 * k + 1) :=
sorry

end part_a_part_b_l604_604089


namespace Tanya_efficiency_higher_l604_604554

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l604_604554


namespace range_of_omega_l604_604825

noncomputable def f (ω x : ℝ) := 
  2 * sin (ω * x) * sin (ω * x + π / 3) + cos (2 * ω * x) - 1 / 2

theorem range_of_omega (ω : ℝ) : 
  (∀ x ∈ Icc 0 π, f ω x = 0) → ω ∈ Icc (11 / 12) (17 / 12) := 
sorry

end range_of_omega_l604_604825


namespace max_distance_is_correct_l604_604147

noncomputable def max_distance_to_D : Real :=
  let u (x y : Real) : Real := Real.sqrt (x^2 + (y-1)^2)
  let v (x y : Real) : Real := Real.sqrt ((x-1)^2 + y^2)
  let w (x y : Real) : Real := Real.sqrt (x^2 + (y+1)^2)
  let distance_cond (x y : Real) : Prop := (u x y)^2 + (v x y)^2 = (w x y)^2
  let circle_constraint (x y : Real) : Prop := x^2 + y^2 ≤ 1
  let P (x y : Real) := distance_cond x y ∧ circle_constraint x y
  let D : (Real × Real) := (-1,0)
  let distance_P_D (x y : Real) : Real := Real.sqrt ((x + 1)^2 + y^2)
  Sup {d : Real | ∃ x y : Real, P x y ∧ d = distance_P_D x y}

theorem max_distance_is_correct : max_distance_to_D = Real.sqrt (2 + 4 / (Real.sqrt 5)) :=
sorry

end max_distance_is_correct_l604_604147


namespace limit_at_2_l604_604166

noncomputable def delta (ε : ℝ) : ℝ := ε / 3

theorem limit_at_2 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x : ℝ, (0 < |x - 2| ∧ |x - 2| < δ) → |(3 * x^2 - 5 * x - 2) / (x - 2) - 7| < ε :=
by
  let δ := delta ε
  have hδ : δ > 0 := by
    sorry
  use δ, hδ
  intros x hx
  sorry

end limit_at_2_l604_604166


namespace length_of_AB_l604_604887

noncomputable def O := { P : ℝ × ℝ | P.1^2 + P.2^2 = 5 }
noncomputable def O1 (m : ℝ) := { P : ℝ × ℝ | (P.1 - m)^2 + P.2^2 = 20 }

theorem length_of_AB (m : ℝ) (A : ℝ × ℝ) :
  A ∈ O →
  A ∈ O1 m →
  (∀ T₁ T₂ : ℝ × ℝ, tangent_point T₁ O A → tangent_point T₂ O1 A → T₁.1 = 0 ∨ T₂.1 = 0) →
  (dist (0, 0) (m, 0) = m) →
  (dist (0, 0) A = sqrt 5) →
  (dist (m, 0) A = sqrt 20) →
  (dist (0, sqrt 5 - sqrt 20) A + dist (sqrt 5 - sqrt 20, 0) A = dist (0,0) (m,0)) →
  dist A (0, m) = 4 := sorry

end length_of_AB_l604_604887


namespace angle_PQR_is_90_l604_604928

variable (R P Q S : Type) [EuclideanGeometry R P Q S]
variable (RSP_is_straight : straight_line R S P)
variable (angle_QSP : ∡Q S P = 70)

theorem angle_PQR_is_90 : ∡P Q R = 90 :=
by
  sorry

end angle_PQR_is_90_l604_604928


namespace sinA_mul_sinC_eq_three_fourths_l604_604047
open Real

-- Definitions based on conditions
def angles_form_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

def sides_form_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- The theorem to prove
theorem sinA_mul_sinC_eq_three_fourths
  (A B C a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_angles : A + B + C = π)
  (h_angles_arithmetic : angles_form_arithmetic_sequence A B C)
  (h_sides_geometric : sides_form_geometric_sequence a b c) :
  sin A * sin C = 3 / 4 :=
sorry

end sinA_mul_sinC_eq_three_fourths_l604_604047


namespace length_of_woods_l604_604613

theorem length_of_woods (area width : ℝ) (h_area : area = 24) (h_width : width = 8) : (area / width) = 3 :=
by
  sorry

end length_of_woods_l604_604613


namespace circle_and_trajectory_l604_604065

-- Define the point, line and circle related to the problem
def O := (0 : ℝ, 0 : ℝ)
def line := { p | p.1 - (Real.sqrt 3) * p.2 = 4 }
def circle_eq (x y r : ℝ) := x^2 + y^2 = r^2
def geom_seq (PA PO PB : ℝ) := PO^2 = PA * PB

-- Statement for the proof problem
theorem circle_and_trajectory:
    (circle_eq 0 0 2 ↔ ∀ (x y : ℝ), (x - 0)^2 + (y - 0)^2 = 4 ∧ 
    (∀ (P : ℝ × ℝ), geom_seq (Real.sqrt ((P.1 + 2)^2 + P.2^2)) (Real.sqrt (P.1^2 + P.2^2)) (Real.sqrt ((P.1 - 2)^2 + P.2^2)) ↔ 
    (P.1^2 - P.2^2 = 2))) :=
by
  sorry

end circle_and_trajectory_l604_604065


namespace cube_identity_simplification_l604_604146

theorem cube_identity_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3 * x * y * z) / (x * y * z) = 6 :=
by
  sorry

end cube_identity_simplification_l604_604146


namespace contradiction_for_n3_min_elements_when_n_ge_4_l604_604676

theorem contradiction_for_n3 :
  ∀ (s : Set ℕ), (s.card = 7) → 
                 (∀ (x ∈ s), x ≥ 1) → 
                 (∃ t u : Set ℕ, (t.card = 4) ∧ (u.card = 3) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) 
                 → ∃ x ∈ s, false :=
sorry

theorem min_elements_when_n_ge_4 (n : ℕ) (hn : n ≥ 4) :
  ∃ (s : Set ℕ), (s.card = 2 * n + 1) ∧ 
                 (∀ (x ∈ s), x ≥ 1) ∧ 
                 (∃ t u : Set ℕ, (t.card = n + 1) ∧ (u.card = n) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) ∧
                 ∀ (s : Set ℕ), s.card = 2 * n + 1 → (∑ x in s, x) / (2 * n + 1) = 6 :=
sorry

example : ∃ s, (s.card = 9) ∧ (∀ x ∈ s, x ≥ 1) ∧ 
               (∃ t u : Set ℕ, (t.card = 5) ∧ (u.card = 4) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
               (∀ x ∈ t, x ≥ 10) ∧ (∀ x ∈ u, x ≥ 1) ∧
               (∑ x in s, x) / 9 = 6) :=
{ sorry }

end contradiction_for_n3_min_elements_when_n_ge_4_l604_604676


namespace polynomial_eq_binomial_coeff_l604_604087

open BigOperators

theorem polynomial_eq_binomial_coeff 
  (k n : ℕ) 
  (hkn : 1 ≤ k ∧ k ≤ n) 
  (a : Fin k → ℝ)
  (sum_eq_n : ∑ i in Finset.univ, a i = n)
  (sum_sq_eq_n : ∑ i in Finset.univ, (a i)^2 = n)
  (sum_cube_eq_n : ∑ i in Finset.univ, (a i)^3 = n)
  (sum_k_eq_n : ∑ i in Finset.univ, (a i)^k = n) :
  (∏ i in Finset.univ, (X + C (a i))) =
  (X^k + 
   (∑ i in Finset.range k, (nat.choose n (i+1)) * X^(k-i-1))) := 
sorry

end polynomial_eq_binomial_coeff_l604_604087


namespace intersection_A_B_l604_604126

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l604_604126


namespace func_inequality_l604_604008

open Real

theorem func_inequality
  (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, x ∈ Ioo (-π) π → f x ≥ cos (x / 2) * f (x / 2))
  (h_f_cont : ContinuousOn f (Ioo (-π) π))
  (h_f_zero : f 0 = 1) :
  ∀ x : ℝ, x ∈ Ioo (-π) π → f x ≥ (sin x) / x := 
by
  sorry

end func_inequality_l604_604008


namespace prob_of_mutually_exclusive_events_is_zero_l604_604884

variables {Ω : Type} {p : Ω → Prop}
variable [MeasureTheory.ProbabilityMeasure p]

theorem prob_of_mutually_exclusive_events_is_zero
    {A B : Set Ω} (h : A ∩ B = ∅) :
    MeasureTheory.measure_theory.MeasureTheory.probability ℙ (A ∩ B) = 0 :=
sorry

end prob_of_mutually_exclusive_events_is_zero_l604_604884


namespace distinct_floors_l604_604130

-- Define the floor function
def floor (x : ℝ) : ℕ := Int.toNat (Real.floor x)

-- Define the main problem statement
theorem distinct_floors (s : Finset ℕ) :
  s = (Finset.image (λ n : ℕ, floor ((n^2 : ℚ) / 2005)) (Finset.range 2006)) →
  s.card = 1503 :=
begin
  intro h,
  rw h,
  sorry
end

end distinct_floors_l604_604130


namespace x_squared_plus_y_squared_lt_one_l604_604601

-- Definitions and conditions from the problem
variables (x y : ℝ)
hypothesis (h1 : 0 < x)
hypothesis (h2 : 0 < y)
hypothesis (h3 : x^3 + y^3 = x - y)

-- The theorem to be proven
theorem x_squared_plus_y_squared_lt_one : x^2 + y^2 < 1 :=
sorry

end x_squared_plus_y_squared_lt_one_l604_604601


namespace num_proper_subsets_of_A_l604_604015

/- Define the universal set U -/
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

/- Define the complement of A with respect to U -/
def CU_A : Set ℕ := { x | 1 ≤ x ∧ x ≤ 3 }

/- Define the set A itself -/
def A : Set ℕ := U \ CU_A

/- Prove the number of proper subsets of set A -/
theorem num_proper_subsets_of_A : Finset.card (Finset.powerset (A : Finset ℕ)) - 1 = 7 :=
by
  sorry

end num_proper_subsets_of_A_l604_604015


namespace polar_coordinates_l604_604364

theorem polar_coordinates (x y r θ : ℝ) (h₀ : x = sqrt 3) (h₁ : y = -1)
  (h₂ : r = sqrt (x^2 + y^2)) (h₃ : 0 ≤ θ ∧ θ < 2 * π)
  (h₄ : θ = if y < 0 then 2 * π - arctan (abs y / x) else arctan (y / x)) :
  (r, θ) = (2, 11 * π / 6) :=
by
  sorry

end polar_coordinates_l604_604364


namespace difference_between_largest_and_smallest_root_l604_604603

noncomputable def diff_largest_smallest_root (p : Polynomial ℝ) : ℝ :=
  let roots := Multiset.sort fun x y => x <= y (roots p)
  roots.erase_dup.last roots nocoshole - roots.erase_dup.head roots nocsome

theorem difference_between_largest_and_smallest_root :
  diff_largest_smallest_root (Polynomial.C 1 - Polynomial.monomial 1 (10 : ℝ)
    + Polynomial.monomial 2 (31 : ℝ) - Polynomial.monomial 3 (40 : ℝ)
    + Polynomial.monomial 4 $ 16) = real.sqrt 369 / 4 :=
by
  sorry

end difference_between_largest_and_smallest_root_l604_604603


namespace area_triangle_ABC_l604_604916

variables (AB CD height : ℝ)
variables (area_trap : ℝ := 18)
variables (relation : CD = 3 * AB)
variables (area_ABC : ℝ := 1 / 4 * area_trap)

theorem area_triangle_ABC (h1 : area_trap = 18) (h2 : relation = (3 * AB = CD)) :
  area_ABC = 4.5 :=
sorry

end area_triangle_ABC_l604_604916


namespace positive_difference_btween_jos_sum_and_lisas_sum_l604_604081

noncomputable def jo_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def round_to_nearest_5 (x : ℕ) : ℕ :=
  if x % 5 < 3 then (x / 5) * 5 else (x / 5 + 1) * 5

noncomputable def lisa_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ x, round_to_nearest_5 (x + 1))

theorem positive_difference_btween_jos_sum_and_lisas_sum (n : ℕ) (h : n = 60) :
  jo_sum n - lisa_sum n = 240 :=
sorry

end positive_difference_btween_jos_sum_and_lisas_sum_l604_604081


namespace probability_same_color_plates_l604_604854

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l604_604854


namespace solution_set_inequality_l604_604605

theorem solution_set_inequality : { x : ℝ | (x - 2)^2 ≤ 2 * x + 11 } = set.Icc (-1 : ℝ) 7 := by
  sorry

end solution_set_inequality_l604_604605


namespace parallel_vectors_sufficient_not_necessary_condition_l604_604452

-- Define the vectors and the condition 
def vector_a (m : ℝ) : ℝ × ℝ := (-9, m^2)
def vector_b : ℝ × ℝ := (1, -1)

-- State the problem: proving m = 3 is a sufficient but not necessary condition for parallelism
theorem parallel_vectors_sufficient_not_necessary_condition (m : ℝ) :
  (vector_a m = -9, m^2) → (vector_b = (1, -1)) → (m = 3 → (vector_a m ∥ vector_b)) ∧ (vector_a m ∥ vector_b → m = 3 ∨ m = -3) :=
by sorry

end parallel_vectors_sufficient_not_necessary_condition_l604_604452


namespace intersection_of_sets_l604_604109

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l604_604109


namespace max_f_eq_one_l604_604239

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + sqrt 3 * Real.sin (x - Real.pi / 3)

theorem max_f_eq_one : ∃ x : ℝ, f x = 1 :=
sorry

end max_f_eq_one_l604_604239


namespace exists_expression_equals_two_l604_604487

def one_fourth : ℚ := 1 / 4

theorem exists_expression_equals_two :
  ∃ (e : ℚ), (e = (one_fourth / one_fourth) + (one_fourth / one_fourth) ∨ 
               e = (one_fourth / (one_fourth + one_fourth)) / one_fourth) ∧ 
               e = 2 :=
by
  use (one_fourth / one_fourth) + (one_fourth / one_fourth)
  split
  {
    left, refl
  }
  {
    refl
  }
  sorry

end exists_expression_equals_two_l604_604487


namespace marbles_ratio_l604_604540

theorem marbles_ratio (miriam_current_marbles miriam_initial_marbles marbles_brother marbles_sister marbles_total_given marbles_savanna : ℕ)
  (h1 : miriam_current_marbles = 30)
  (h2 : marbles_brother = 60)
  (h3 : marbles_sister = 2 * marbles_brother)
  (h4 : miriam_initial_marbles = 300)
  (h5 : marbles_total_given = miriam_initial_marbles - miriam_current_marbles)
  (h6 : marbles_savanna = marbles_total_given - (marbles_brother + marbles_sister)) :
  (marbles_savanna : ℚ) / miriam_current_marbles = 3 :=
by
  sorry

end marbles_ratio_l604_604540


namespace stratified_sampling_counts_l604_604321

-- Defining the given conditions
def num_elderly : ℕ := 27
def num_middle_aged : ℕ := 54
def num_young : ℕ := 81
def total_sample : ℕ := 42

-- Proving the required stratified sample counts
theorem stratified_sampling_counts :
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  elderly_count = 7 ∧ middle_aged_count = 14 ∧ young_count = 21 :=
by 
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  have h1 : elderly_count = 7 := by sorry
  have h2 : middle_aged_count = 14 := by sorry
  have h3 : young_count = 21 := by sorry
  exact ⟨h1, h2, h3⟩

end stratified_sampling_counts_l604_604321


namespace probability_same_color_plates_l604_604852

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l604_604852


namespace probability_same_color_is_correct_l604_604863

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l604_604863


namespace simplify_expression_l604_604207

noncomputable def a : ℝ := Real.sqrt 3 - 1

theorem simplify_expression : 
  ( (a - 1) / (a^2 - 2 * a + 1) / ( (a^2 + a) / (a^2 - 1) + 1 / (a - 1) ) = Real.sqrt 3 / 3 ) :=
by
  sorry

end simplify_expression_l604_604207


namespace monotonicity_f_number_of_zeros_g_l604_604823

noncomputable def f (a x : ℝ) := a * Real.exp x - 2 * x + 1
noncomputable def g (a x : ℝ) := f a x + x * Real.log x 

theorem monotonicity_f (a : ℝ) :
  (a ≤ 0 → ∀ x, (differentiable ℝ (λ x, f a x)).deriv x < 0) ∧
  (a > 0 → 
     (∀ x, x < Real.log(2 / a) → (differentiable ℝ (λ x, f a x)).deriv x < 0) ∧
     (∀ x, x > Real.log(2 / a) → (differentiable ℝ (λ x, f a x)).deriv x > 0)) :=
by {
  sorry
}

theorem number_of_zeros_g (a : ℝ) (a_pos : a > 0) :
  (a = 1 / Real.exp 1 → ∃ x, g a x = 0) ∧
  (a > 1 / Real.exp 1 → ∀ x, g a x ≠ 0) ∧
  (0 < a ∧ a < 1 / Real.exp 1 → ∃ x1 x2, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0) :=
by {
  sorry
}

end monotonicity_f_number_of_zeros_g_l604_604823


namespace roots_of_P_l604_604762

-- Define the polynomial P(x) = x^3 + x^2 - 6x - 6
noncomputable def P (x : ℝ) : ℝ := x^3 + x^2 - 6 * x - 6

-- Define the statement that the roots of the polynomial P are -1, sqrt(6), and -sqrt(6)
theorem roots_of_P : ∀ x : ℝ, P x = 0 ↔ (x = -1) ∨ (x = sqrt 6) ∨ (x = -sqrt 6) :=
sorry

end roots_of_P_l604_604762


namespace trigonometry_problem_l604_604813

theorem trigonometry_problem
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (4 / 5, -3 / 5))
  (h_unit : P.1^2 + P.2^2 = 1) :
  (cos α = 4 / 5) ∧
  (tan α = -3 / 4) ∧
  (sin (α + π) = 3 / 5) ∧
  (cos (α - π / 2) ≠ 3 / 5) := by
    sorry

end trigonometry_problem_l604_604813


namespace cyclic_quadrilateral_proof_l604_604949

open EuclideanGeometry

variables {A B C D E F K L O : Point}
variables (c : Circle) (h₁ : CyclicQuadrilateral A B C D)
variables (h₂ : OnCircle c A) (h₃ : OnCircle c B) (h₄ : OnCircle c C) 
variables (h₅ : OnCircle c D) (h₆ : A ≠ B) 
variables (h₇ : A ≠ D) (h₈ : E ∈ Segment A B)
variables (h₉ : SegmentLength A E = SegmentLength A D)
variables (h₁₀ : Intersection AC DE F)
variables (h₁₁ : Intersection DE O K)
variables (h₁₂ : K ≠ D)
variables (h₁₃ : TangentToCircleAt c E (lineThrough C F) L)
variables (h₁₄ : CircleThroughPoints c C F E)

theorem cyclic_quadrilateral_proof :
  (SegmentLength A L = SegmentLength A D) ↔
  (Angle K C E = Angle A L E) := sorry

end cyclic_quadrilateral_proof_l604_604949


namespace train_length_is_250_l604_604344

noncomputable def train_length (V₁ V₂ V₃ : ℕ) (T₁ T₂ T₃ : ℕ) : ℕ :=
  let S₁ := (V₁ * (5/18) * T₁)
  let S₂ := (V₂ * (5/18)* T₂)
  let S₃ := (V₃ * (5/18) * T₃)
  if S₁ = S₂ ∧ S₂ = S₃ then S₁ else 0

theorem train_length_is_250 :
  train_length 50 60 70 18 20 22 = 250 := by
  -- proof omitted
  sorry

end train_length_is_250_l604_604344


namespace total_weight_correct_average_weight_correct_l604_604703

def base_weight : ℝ := 50
def deviations : List ℝ := [2, 3, -7.5, -3, 5, -8, 3.5, 4.5, 8, -1.5]

def total_students : ℕ := 10
def total_weight (base_weight : ℝ) (deviations : List ℝ) : ℝ :=
  (total_students * base_weight) + (deviations.sum)

def average_weight (total_weight : ℝ) (total_students : ℕ) : ℝ :=
  total_weight / total_students

theorem total_weight_correct : 
  total_weight base_weight deviations = 509 := sorry

theorem average_weight_correct :
  average_weight (total_weight base_weight deviations) total_students = 50.9 := sorry

end total_weight_correct_average_weight_correct_l604_604703


namespace sum_of_arithmetic_derivative_eq_n_l604_604215

def arithmetic_derivative (n : ℕ) : ℕ :=
  if n = 1 then 0
  else if Prime n then 1
  else ∑ d in n.divisors.filter (λ d, 1 < d ∧ d < n), d * arithmetic_derivative (n / d)

theorem sum_of_arithmetic_derivative_eq_n (N : ℕ) : ∑ n in (range N).filter (λ n, arithmetic_derivative n = n) = 31 :=
by sorry

end sum_of_arithmetic_derivative_eq_n_l604_604215


namespace martha_saving_l604_604536

-- Definitions for the conditions
def daily_allowance : ℕ := 12
def half_daily_allowance : ℕ := daily_allowance / 2
def quarter_daily_allowance : ℕ := daily_allowance / 4
def days_saving_half : ℕ := 6
def day_saving_quarter : ℕ := 1

-- Statement to be proved
theorem martha_saving :
  (days_saving_half * half_daily_allowance) + (day_saving_quarter * quarter_daily_allowance) = 39 := by
  sorry

end martha_saving_l604_604536


namespace parabola_vertex_x_coordinate_l604_604234

theorem parabola_vertex_x_coordinate (a b c : ℝ) (h1 : c = 0) (h2 : 16 * a + 4 * b = 0) (h3 : 9 * a + 3 * b = 9) : 
    -b / (2 * a) = 2 :=
by 
  -- You can start by adding a proof here
  sorry

end parabola_vertex_x_coordinate_l604_604234


namespace min_value_func_l604_604594

noncomputable def func (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem min_value_func : ∃ x : ℝ, func x = -2 :=
by
  existsi (Real.pi / 2 + Real.pi / 3)
  sorry

end min_value_func_l604_604594


namespace area_ratio_triangle_l604_604494

theorem area_ratio_triangle 
  {α γ : ℝ} 
  (h_α_gt_γ : α > γ) 
  (h_α_lt_pi : α < π / 2) 
  (h_γ_lt_pi : γ < π / 2) 
  (h_acute_α_γ : 0 < α ∧ 0 < γ) :
  let area_ratio := (Real.tan ((α - γ)/2)) / (2 * Real.tan ((α + γ)/2))
  in area_ratio = area_ratio :=
by
  sorry

end area_ratio_triangle_l604_604494


namespace gold_quarter_value_comparison_l604_604726

theorem gold_quarter_value_comparison:
  (worth_in_store per_quarter: ℕ → ℝ) 
  (weight_per_quarter in_ounce: ℝ) 
  (earning_per_ounce melted: ℝ) : 
  (worth_in_store 4  = 0.25) →
  (weight_per_quarter = 1/5) →
  (earning_per_ounce = 100) →
  (earning_per_ounce * weight_per_quarter / worth_in_store 4 = 80) :=
by
  -- The proof goes here
  sorry

end gold_quarter_value_comparison_l604_604726


namespace max_num_threes_l604_604615

-- Definitions corresponding to the conditions
def num_of_each_card : ℕ := 10
def num_of_chosen_cards : ℕ := 8
def desired_sum : ℕ := 31

-- Proof statement
theorem max_num_threes : 
  (∃ (x y z : ℕ), 
    x + y + z = num_of_chosen_cards ∧
    3 * x + 4 * y + 5 * z = desired_sum ∧
    x ≤ num_of_each_card ∧ y ≤ num_of_each_card ∧ z ≤ num_of_each_card) →
  ∃ x, x ≤ 4 :=
begin
  intro h,
  sorry
end

end max_num_threes_l604_604615


namespace sequence_a10_l604_604073

noncomputable def sequence : ℕ → ℕ
| 0     := 0
| (n+1) := if n = 0 then 1 else sequence n + n + 1

theorem sequence_a10 : sequence 10 = 55 := by
  sorry

end sequence_a10_l604_604073


namespace sum_of_n_terms_l604_604607

noncomputable def S : ℕ → ℕ :=
sorry -- We define S, but its exact form is not used in the statement directly

noncomputable def a : ℕ → ℕ := 
sorry -- We define a, but its exact form is not used in the statement directly

-- Conditions
axiom S3_eq : S 3 = 1
axiom a_rec : ∀ n : ℕ, 0 < n → a (n + 3) = 2 * (a n)

-- Proof problem
theorem sum_of_n_terms : S 2019 = 2^673 - 1 :=
sorry

end sum_of_n_terms_l604_604607


namespace coeff_x_binomial_expansion_l604_604383

theorem coeff_x_binomial_expansion :
  ∑ k in finset.range 21, binomial 20 k * (-1) ^ k * x ^ (k / 2) = 190 :=
sorry

end coeff_x_binomial_expansion_l604_604383


namespace sum_of_interior_angles_of_pentagon_l604_604608

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  -- We skip the proof as per instruction
  sorry

end sum_of_interior_angles_of_pentagon_l604_604608


namespace glue_needed_l604_604507

-- Definitions based on conditions
def num_friends : ℕ := 7
def clippings_per_friend : ℕ := 3
def drops_per_clipping : ℕ := 6

-- Calculation
def total_clippings : ℕ := num_friends * clippings_per_friend
def total_drops_of_glue : ℕ := drops_per_clipping * total_clippings

-- Theorem statement
theorem glue_needed : total_drops_of_glue = 126 := by
  sorry

end glue_needed_l604_604507


namespace solve_log_eq_l604_604213

noncomputable theory

open Nat Real

theorem solve_log_eq (x : ℝ) :
  log 2 (9^(x-1) - 5) = log 2 (3^(x-1) - 2) + 2 ↔ x = 2 := 
sorry

end solve_log_eq_l604_604213


namespace find_a_if_f_is_odd_l604_604880

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (1 / (2 ^ x - 1)) + a

-- The main theorem
theorem find_a_if_f_is_odd (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) → a = 1 / 2 :=
by
  sorry

end find_a_if_f_is_odd_l604_604880


namespace quadrilateral_is_rhombus_l604_604911

-- Conditions
variables {ABCD : Type} [quadrilateral ABCD]
variables {A1 B1 C1 D1 A2 B2 C2 D2 : Type} [quadrilateral A1] [quadrilateral A2]
variable {AC BD : Type}
variable {A2006 B2006 C2006 D2006 : Type}

-- Conditions
axiom quad_condition (h : AC ⊥ BD) : 
  is_midpoint_connected_sequence ABCD A1 B1 C1 D1 A2006 B2006 C2006 D2006

-- Prove that A2006B2006C2006D2006 is a rhombus
theorem quadrilateral_is_rhombus (h : AC ⊥ BD) : is_rhombus A2006 B2006 C2006 D2006 :=
sorry

end quadrilateral_is_rhombus_l604_604911


namespace expression_not_prime_l604_604173

open Nat

theorem expression_not_prime (n k : ℕ) (hn : n > 2) (hk : k ≠ n) : ¬ prime (n^2 - k * n + k - 1) :=
  sorry

end expression_not_prime_l604_604173


namespace smallest_of_seven_even_numbers_l604_604574

theorem smallest_of_seven_even_numbers (a b c d e f g : ℕ) 
  (h1 : a % 2 = 0) 
  (h2 : b = a + 2) 
  (h3 : c = a + 4) 
  (h4 : d = a + 6) 
  (h5 : e = a + 8) 
  (h6 : f = a + 10) 
  (h7 : g = a + 12) 
  (h_sum : a + b + c + d + e + f + g = 700) : 
  a = 94 :=
by sorry

end smallest_of_seven_even_numbers_l604_604574


namespace weight_of_person_being_replaced_l604_604476

variables (W X : ℝ) (A : ℝ)
hypothesis h1 : W = 4 * A
hypothesis h2 : 4 * (A + 1.5) = W - X + 71

theorem weight_of_person_being_replaced :
  X = 65 :=
by
  sorry

end weight_of_person_being_replaced_l604_604476


namespace part1_solution_part2_solution_l604_604531

-- Define the inequality for part (1)
def ineq_part1 (x : ℝ) : Prop := 1 - (4 / (x + 1)) < 0

-- Define the solution set P for part (1)
def P (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Prove that the solution set for the inequality is P
theorem part1_solution :
  ∀ (x : ℝ), ineq_part1 x ↔ P x :=
by
  -- proof omitted
  sorry

-- Define the inequality for part (2)
def ineq_part2 (x : ℝ) : Prop := abs (x + 2) < 3

-- Define the solution set Q for part (2)
def Q (x : ℝ) : Prop := -5 < x ∧ x < 1

-- Define P as depending on some parameter a
def P_param (a : ℝ) (x : ℝ) : Prop := -1 < x ∧ x < a

-- Prove the range of a given P ∪ Q = Q 
theorem part2_solution :
  ∀ a : ℝ, (∀ x : ℝ, (P_param a x ∨ Q x) ↔ Q x) → 
    (0 < a ∧ a ≤ 1) :=
by
  -- proof omitted
  sorry

end part1_solution_part2_solution_l604_604531


namespace sqrt_fraction_identity_l604_604647

theorem sqrt_fraction_identity (n : ℕ) (h : n > 0) : 
    Real.sqrt ((1 : ℝ) / n - (1 : ℝ) / (n * n)) = Real.sqrt (n - 1) / n :=
by
  sorry

end sqrt_fraction_identity_l604_604647


namespace set_intersection_l604_604093

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l604_604093


namespace trajectory_line_or_hyperbola_l604_604787

theorem trajectory_line_or_hyperbola
  (a b : ℝ)
  (ab_pos : a * b > 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b) :
  (∃ s t : ℝ, f (s-t) * f (s+t) = (f s)^2) →
  (∃ s t : ℝ, ((t = 0) ∨ (a * t^2 - 2 * a * s^2 + 2 * b = 0))) → true := sorry

end trajectory_line_or_hyperbola_l604_604787


namespace triangle_midline_area_l604_604163

noncomputable def deg_to_rad (d m : ℝ) : ℝ :=
(d + m / 60) * (Real.pi / 180)

def triangle_area (k : ℝ) (φ ψ : ℝ) : ℝ :=
let A := 2 * k * (Real.sin φ) * (2 * k * (Real.sin ψ)) / (Real.sin (φ + ψ))
in A / 2

theorem triangle_midline_area :
  let k := 5 -- in dm
  let φ := deg_to_rad 47 16 -- in degrees to radians
  let ψ := deg_to_rad 25 38 -- in degrees to radians
  triangle_area k φ ψ = 16.623 :=
by
  let k := 5
  let φ := deg_to_rad 47 16
  let ψ := deg_to_rad 25 38
  have hφ : φ = deg_to_rad 47 16 := rfl
  have hψ : ψ = deg_to_rad 25 38 := rfl
  calc
  triangle_area k φ ψ = 16.623 := sorry

end triangle_midline_area_l604_604163


namespace find_values_l604_604036

theorem find_values (a b : ℝ) 
  (h1 : a + b = 10)
  (h2 : a - b = 4) 
  (h3 : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ ab = 21 := 
by 
  sorry

end find_values_l604_604036


namespace part_a_part_b_l604_604653

-- Definitions and Conditions
def median := 10
def mean := 6

-- Part (a): Prove that a set with 7 numbers cannot satisfy the given conditions
theorem part_a (n1 n2 n3 n4 n5 n6 n7 : ℕ) (h1 : median ≤ n1) (h2 : median ≤ n2) (h3 : median ≤ n3) (h4 : median ≤ n4)
  (h5 : 1 ≤ n5) (h6 : 1 ≤ n6) (h7 : 1 ≤ n7) (hmean : (n1 + n2 + n3 + n4 + n5 + n6 + n7) / 7 = mean) :
  false :=
by
  sorry

-- Part (b): Prove that the minimum size of the set where number of elements is 2n + 1 and n is a natural number, is at least 9
theorem part_b (n : ℕ) (h_sum_geq : ∀ (s : Finset ℕ), ((∀ x ∈ s, x >= median) ∧ ∃ t : Finset ℕ, t ⊆ s ∧ (∀ x ∈ t, x >= 1) ∧ s.card = 2 * n + 1) → s.sum >= 11 * n + 10) :
  n ≥ 4 :=
by
  sorry

-- Lean statements defined above match the problem conditions and required proofs

end part_a_part_b_l604_604653


namespace common_ratio_of_geometric_progression_l604_604903

-- Definition of the geometric progression and its properties
def geometric_progression (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a (n+1) + a (n+2) + a (n+3)

-- Proving the common ratio
theorem common_ratio_of_geometric_progression (a : ℕ → ℝ) (r : ℝ) (h : geometric_progression a r) :
  r ≈ 0.5437 :=
sorry

end common_ratio_of_geometric_progression_l604_604903


namespace remaining_credit_to_be_paid_l604_604152

def credit_limit : ℝ := 100
def bakery_section_spent : ℝ := 20
def rewards_discount : ℝ := 0.05
def payment_tuesday : ℝ := 15
def payment_thursday : ℝ := 23

def total_spent (credit_limit bakery_section_spent : ℝ) : ℝ :=
  credit_limit + bakery_section_spent

def discount (total_spent : ℝ) (rewards_discount : ℝ) : ℝ :=
  if total_spent ≥ 50 then total_spent * rewards_discount else 0

def total_after_discount (total_spent discount : ℝ) : ℝ :=
  total_spent - discount

def remaining_credit (total_after_discount payment_tuesday payment_thursday : ℝ) : ℝ :=
  total_after_discount - (payment_tuesday + payment_thursday)

theorem remaining_credit_to_be_paid :
  remaining_credit (total_after_discount (total_spent credit_limit bakery_section_spent) (discount (total_spent credit_limit bakery_section_spent) rewards_discount))
                   payment_tuesday payment_thursday 
  = 76 :=
by
  sorry

end remaining_credit_to_be_paid_l604_604152


namespace amusement_park_line_l604_604164

theorem amusement_park_line (eunji_place : ℕ) (people_behind : ℕ) (h_eunji_place : eunji_place = 6) (h_people_behind : people_behind = 7) :
  ∃ total_people : ℕ, total_people = eunji_place + people_behind + 1 ∧ total_people = 13 :=
by
  use 13
  rw [h_eunji_place, h_people_behind]
  decide

end amusement_park_line_l604_604164


namespace mode_of_numbers_on_balls_number_qiqi_draws_is_4_prob_two_even_numbers_drawn_by_jiajia_l604_604500

-- Definitions and assumptions based on conditions
def num_balls : ℕ := 5
def jiajia_labels : Finset ℕ := {2, 3, 4}
def qiqi_labels : Finset ℕ := {3, 4}
def freq_of_3_ball : ℚ := 2 / 5      -- Since number 3 appears with frequency 0.4

-- The problem statements to prove
theorem mode_of_numbers_on_balls (balls : Finset ℕ) (h : balls = {2, 3, 3, 4, 4}) : (∃! m : ℕ, m ∈ balls ∧ m = 3) ∧ 
 (∃! n : ℕ, n ∈ balls ∧ n = 4) :=
begin
  sorry
end

theorem number_qiqi_draws_is_4 (remaining_balls : Finset ℕ) (h : remaining_balls = {2, 3, 3, 4}) : 4 ∈ qiqi_labels :=
begin
  sorry
end

theorem prob_two_even_numbers_drawn_by_jiajia (possible_draws : Finset (ℕ × ℕ)) (h : possible_draws = 
{(2,2), (2,3), (2,3), (2,4),
 (3,2), (3,3), (3,3), (3,4),
 (3,2), (3,3), (3,3), (3,4),
 (4,2), (4,3), (4,3), (4,4)}) : 
   (∃! p : ℚ, p = 1/4):=
begin
  sorry
end

end mode_of_numbers_on_balls_number_qiqi_draws_is_4_prob_two_even_numbers_drawn_by_jiajia_l604_604500


namespace equation_C1_equation_C2_existence_of_line_l_l604_604977

def F1 : ℝ × ℝ := (-sqrt 3, 0)
def F2 : ℝ × ℝ := (sqrt 3, 0)
def focus_C2 : ℝ × ℝ := (1, 0)
def vertex_C2 : ℝ × ℝ := (0, 0)

theorem equation_C1 :
  ∀ (M : ℝ × ℝ),
  |(M.1 - F1.1)^2 + (M.2 - F1.2)^2|.sqrt + |(M.1 - F2.1)^2 + (M.2 - F2.2)^2|.sqrt = 4 →
  (M.1^2) / 4 + M.2^2 = 1 := sorry

theorem equation_C2 :
  (focus_C2 = (1, 0)) → (vertex_C2 = (0, 0)) → 
  (∀ (x y : ℝ), y^2 = 4 * x ↔ x = (y^2) / 4) := sorry

theorem existence_of_line_l :
  ∃ k : ℝ, 
    (∀ (x y : ℝ), y = k * (x - 1)) ∧ 
    ∀ (M N : ℝ × ℝ), 
      M ≠ N → 
      (M.1, M.2) ≠ focus_C2 → 
      (N.1, N.2) ≠ focus_C2 → 
      (M.1^2) / 4 + M.2^2 = 1 → (N.1^2) / 4 + N.2^2 = 1 → 
      (vertex_C2.1 * M.1 + vertex_C2.2 * M.2 = 0) ∧ (vertex_C2.1 * N.1 + vertex_C2.2 * N.2 = 0)
      → (k = 2 ∨ k = -2) := sorry

end equation_C1_equation_C2_existence_of_line_l_l604_604977


namespace geometric_sequence_sum_l604_604803

theorem geometric_sequence_sum :
  (∀ n : ℕ, a_n > 0) ∧
  g_seq a ∧
  (a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) →
  a 3 + a 5 = 5 :=
begin
  intros, 
  sorry
end

-- Defining geometric sequence property
def g_seq (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, a (m + n) = a m * a n

end geometric_sequence_sum_l604_604803


namespace distance_QR_l604_604204

-- Defining the lengths of the sides of the triangle
def DE : ℝ := 9
def EF : ℝ := 12
def DF : ℝ := 15

-- Defining the property of the right triangle DEF
def right_triangle (a b c : ℝ) : Prop := a * a + b * b = c * c

-- Defining points Q and R as centers of circles with given tangency properties
structure Circle (center : ℝ × ℝ) (radius : ℝ) :=
( tangent_point_line1 : Prop )
( passes_through_point2 : Prop )

-- Center Q tangent to line DE at D and passing through F implies certain geometric properties
noncomputable def Q_center := (0, 0) -- Placeholder for Q calculation
noncomputable def R_center := (0, 0) -- Placeholder for R calculation

-- The proof problem to be stated
theorem distance_QR:
  right_triangle DE EF DF →
  ∃ (Q R : Circle),
  Q.tangent_point_line1 ∧ Q.passes_through_point2 ∧
  R.tangent_point_line1 ∧ R.passes_through_point2 ∧
  dist Q.center R.center = 8 :=
begin
  intro h,
  sorry
end

end distance_QR_l604_604204


namespace molly_next_flip_heads_l604_604156

noncomputable def molly_coin_probability : ℚ := 3 / 4

theorem molly_next_flip_heads : molly_coin_probability = 3 / 4 :=
begin
  sorry
end

end molly_next_flip_heads_l604_604156


namespace number_of_cows_l604_604477

variable {D C : ℕ}

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 :=
by sorry

end number_of_cows_l604_604477


namespace exists_root_abs_leq_2_abs_c_div_b_l604_604722

theorem exists_root_abs_leq_2_abs_c_div_b (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h_real_roots : ∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| :=
by
  sorry

end exists_root_abs_leq_2_abs_c_div_b_l604_604722


namespace distinct_circles_from_square_l604_604955

theorem distinct_circles_from_square (S : Type) (V : set S) (hV : V.card = 4) (sq : is_square V) :
  ∃! (C : set (set S)), C.card = 2 ∧ (∀ c ∈ C, ∃ (p q ∈ V), c = circle_diameter p q) :=
sorry

end distinct_circles_from_square_l604_604955


namespace trajectory_of_moving_circle_l604_604817

noncomputable def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4
def point_A := (-3, 0) : ℝ × ℝ

theorem trajectory_of_moving_circle (M : ℝ × ℝ) (r : ℝ) :
  (∃ x y : ℝ, M = (x, y) ∧
    (∀ x y : ℝ, circle_C x y → x ≥ 1 →
    M ≠ (x - 3, y))) →
  (∀ (x y : ℝ),
    x^2 - y^2 / 8 = 1 ∧ x ≥ 1 → (M = (x, y))) :=
sorry

end trajectory_of_moving_circle_l604_604817


namespace minNumberOfRectangles_correct_l604_604523

variable (k n : ℤ)

noncomputable def minNumberOfRectangles (k n : ℤ) : ℤ :=
  if 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1 then
    if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1)
  else 0 -- 0 if the conditions are not met

theorem minNumberOfRectangles_correct (k n : ℤ) (h : 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1) : 
  minNumberOfRectangles k n = 
  if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1) := 
by 
  -- Proof will go here
  sorry

end minNumberOfRectangles_correct_l604_604523


namespace no_set_with_7_elements_min_elements_condition_l604_604661

noncomputable def set_a_elements := 7
noncomputable def median_a := 10
noncomputable def mean_a := 6
noncomputable def min_sum_a := 3 + 4 * 10
noncomputable def real_sum_a := mean_a * set_a_elements

theorem no_set_with_7_elements : ¬ (set_a_elements = 7 ∧
  (∃ S : Finset ℝ, 
    (S.card = set_a_elements) ∧ 
    (S.sum ≥ min_sum_a) ∧ 
    (S.sum = real_sum_a))) := 
by
  sorry

noncomputable def n_b_elements := ℕ
noncomputable def set_b_elements (n : ℕ) := 2 * n + 1
noncomputable def median_b := 10
noncomputable def mean_b := 6
noncomputable def min_sum_b (n : ℕ) := n + 10 * (n + 1)
noncomputable def real_sum_b (n : ℕ) := mean_b * set_b_elements n

theorem min_elements_condition (n : ℕ) : 
    (∀ n : ℕ, n ≥ 4) → 
    (set_b_elements n ≥ 9 ∧
        ∃ S : Finset ℝ, 
          (S.card = set_b_elements n) ∧ 
          (S.sum ≥ min_sum_b n) ∧ 
          (S.sum = real_sum_b n)) :=
by
  assume h : ∀ n : ℕ, n ≥ 4
  sorry

end no_set_with_7_elements_min_elements_condition_l604_604661


namespace evaluate_expression_l604_604291

theorem evaluate_expression : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := 
by
  sorry

end evaluate_expression_l604_604291


namespace hexadecagon_area_l604_604709

theorem hexadecagon_area (r : ℝ) (r_pos : 0 < r) : 
  let π := Real.pi in
  let θ := 2 * π / 16 in
  let A := 16 * (1/2) * r^2 * Real.sin(θ / 2) in
  A = 4 * r^2 * Real.sqrt(2 - Real.sqrt 2) := 
by
  let π := Real.pi
  let θ := 2 * π / 16
  let A := 16 * (1/2) * r^2 * Real.sin(θ / 2)
  have sin_22_5 : Real.sin(θ / 2) = (Real.sqrt(2 - Real.sqrt 2) / 2) := sorry
  calc
    A = 16 * (1/2) * r^2 * Real.sin(θ / 2) : by rfl
    ... = 16 * (1/2) * r^2 * (Real.sqrt(2 - Real.sqrt 2) / 2) : by rw [sin_22_5]
    ... = 16 * r^2 * (Real.sqrt(2 - Real.sqrt 2) / 4) : by ring
    ... = 4 * r^2 * Real.sqrt(2 - Real.sqrt 2) : by ring


end hexadecagon_area_l604_604709


namespace sum_of_products_nonzero_l604_604050

noncomputable def grid : Type := fin 25 → fin 25 → ℤ

def is_valid_cell_value (v : ℤ) : Prop := v = 1 ∨ v = -1

def is_valid_grid (g : grid) : Prop := ∀ i j, is_valid_cell_value (g i j)

def row_product (g : grid) (i : fin 25) : ℤ := ∏ j, g i j

def column_product (g : grid) (j : fin 25) : ℤ := ∏ i, g i j

def sum_of_products (g : grid) : ℤ := (∑ i, row_product g i) + (∑ j, column_product g j)

theorem sum_of_products_nonzero (g : grid) (hg : is_valid_grid g) : sum_of_products g ≠ 0 :=
sorry

end sum_of_products_nonzero_l604_604050


namespace maximize_area_angle_BCD_is_90_l604_604912

-- Definition of the problem
variables (A B C D E F : Point)
variables (hAB : A ≠ B) (hAD : A ≠ D) (hE : E = midpoint B C) (hF : is_angle_bisector F B D C)

def isosceles_triangle (A B D : Point) := 
AB = 1 ∧ AD = 1 ∧ (∠BAD = 60 : ℝ)

-- Proving the angle BCD is 90 degrees when the area of triangle AEF is maximized
theorem maximize_area_angle_BCD_is_90 
(h_AB_AD_eq_1 : AB = 1 ∧ AD = 1)
(h_angle_BAD : ∠BAD = 60) 
(hE_midpoint : E = midpoint B C)
(hF_bisector : F = is_angle_bisector ∠BCD BD)
: ∠BCD = 90 := 
begin
  sorry,
end

end maximize_area_angle_BCD_is_90_l604_604912


namespace angle_PQR_is_90_l604_604929

theorem angle_PQR_is_90 {P Q R S : Type}
  (is_straight_line_RSP : ∃ P R S : Type, (angle R S P = 180)) 
  (angle_QSP : angle Q S P = 70)
  (isosceles_RS_SQ : ∃ (RS SQ : Type), RS = SQ)
  (isosceles_PS_SQ : ∃ (PS SQ : Type), PS = SQ) : angle P Q R = 90 :=
by 
  sorry

end angle_PQR_is_90_l604_604929


namespace norma_found_cards_l604_604543

/-- Assume Norma originally had 88.0 cards. -/
def original_cards : ℝ := 88.0

/-- Assume Norma now has a total of 158 cards. -/
def total_cards : ℝ := 158

/-- Prove that Norma found 70 cards. -/
theorem norma_found_cards : total_cards - original_cards = 70 := 
by
  sorry

end norma_found_cards_l604_604543


namespace trajectory_C_trajectory_M_l604_604797

noncomputable def A : ℝ × ℝ := (-1, 0)
noncomputable def B : ℝ × ℝ := (3, 0)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  dist A B ^ 2 + dist C B ^ 2 = dist A B ^ 2 ∨
  dist B C ^ 2 + dist C A ^ 2 = dist A B ^ 2 ∨
  dist A C ^ 2 + dist C B ^ 2 = dist A B ^ 2

theorem trajectory_C (h : is_right_angle A B C) :
  ∃ C : ℝ × ℝ, ((C.1 - 1)^2 + C.2^2 = 4) ∧ 
  C.1 ≠ 3 ∧ C.1 ≠ -1 :=
sorry

theorem trajectory_M (M : ℝ × ℝ)
  (hC : ∃ C : ℝ × ℝ, (C.1 - 1)^2 + C.2^2 = 4 ∧ 
    C.1 ≠ 3 ∧ C.1 ≠ -1 ∧ 
    M = midpoint B C) :
  (M.1 - 2)^2 + M.2^2 = 1 ∧ 
  M.1 ≠ 3 ∧ M.1 ≠ 1 :=
sorry

end trajectory_C_trajectory_M_l604_604797


namespace solve_equation_l604_604606

theorem solve_equation : ∀ x : ℝ, (2 / 3 * x - 2 = 4) → x = 9 :=
by
  intro x
  intro h
  sorry

end solve_equation_l604_604606


namespace exist_point_set_l604_604159

-- Define a circle with a circumference of 15 units
def circle (r : ℝ) := set (point(ℝ × ℝ × ℝ))

-- Define the property of moving certain distances along the circle
def moves_correctly (S : set (point(ℝ))) : Prop :=
  ∀ p ∈ S, ∃ q1 q2 ∈ S, circular_distance p q1 = 2 ∧ circular_distance p q2 = 3

-- The main theorem to be proven
theorem exist_point_set (r : ℝ) (c : r = 15)
  (circle_pts : set (point(ℝ))) (finite circle_pts) 
  (moves_correctly circle_pts) : 
  ∃ S, S ⊆ circle_pts ∧ ∃ n, n = fintype.card S ∧ (n % 10 ≠ 0) :=
sorry

end exist_point_set_l604_604159


namespace total_paths_via_B_l604_604848

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, 0         := 1
| 0, (k + 1)   := 0
| (n + 1), 0   := 1
| (n + 1), (k + 1) := binomial_coefficient n k + binomial_coefficient n (k + 1)

def paths_from_A_to_B : ℕ := binomial_coefficient 6 2

def paths_from_B_to_C : ℕ := binomial_coefficient 6 3

theorem total_paths_via_B : paths_from_A_to_B * paths_from_B_to_C = 300 := by
  -- proof goes here
  sorry

end total_paths_via_B_l604_604848


namespace cosine_between_vectors_l604_604844

theorem cosine_between_vectors (a b : ℝ × ℝ) (h_a : a = (2, 1)) (h_b : b = (1, 2)) :
    real.angle.cos_between a b = 4/5 :=
by
    sorry

end cosine_between_vectors_l604_604844


namespace fixed_point_of_function_l604_604233

theorem fixed_point_of_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ x y : ℝ, y = a^(x-1) + 1 ∧ (x, y) = (1, 2) :=
by 
  sorry

end fixed_point_of_function_l604_604233


namespace point_outside_circle_l604_604600

theorem point_outside_circle (m : ℝ) :
    let P := (m^2, 5 : ℝ)
    let circle_radius := 2 * Real.sqrt 6
    (Real.sqrt ((m^2)^2 + 5^2)) > circle_radius :=
by
  let P := (m^2, 5 : ℝ)
  let circle_radius := 2 * Real.sqrt 6
  have h1 : (Real.sqrt ((m^2)^2 + 5^2)) = Real.sqrt (m^4 + 25),
  { rw [sq, sq], }
  rw [h1]
  have h2 : circle_radius = 2 * Real.sqrt 6,
  { refl }
  sorry

end point_outside_circle_l604_604600


namespace added_water_l604_604692

-- Define the problem conditions
def original_solution := 340 -- Original total volume in liters
def percent_water := 0.88
def percent_kola := 0.05
def added_sugar := 3.2 -- liters of added sugar
def added_kola := 6.8 -- liters of added concentrated kola
def final_sugar_percent := 0.075 -- final percentage of sugar

-- Define the proofs
theorem added_water (y : ℝ) : 
  y = 10 ↔ 
  let original_sugar := (1 - percent_water - percent_kola) * original_solution in 
  let new_total_volume := original_solution + added_sugar + added_kola + y in 
  original_sugar + added_sugar = final_sugar_percent * new_total_volume := 
sorry

end added_water_l604_604692


namespace sufficient_condition_l604_604313

theorem sufficient_condition (a : ℝ) (h : a > 0) : a^2 + a ≥ 0 :=
sorry

end sufficient_condition_l604_604313


namespace divisors_of_fourth_power_congruent_one_mod_four_l604_604992

theorem divisors_of_fourth_power_congruent_one_mod_four (x : ℕ) (n : ℕ) (d : ℕ) 
  (hx : x = n^4) (hd : d = (∏ i in (range n).filter (λ i, x % i = 0), i))
  (h_pos : 0 < n) :
  d % 4 = 1 :=
by
  sorry

end divisors_of_fourth_power_congruent_one_mod_four_l604_604992


namespace ac_equals_7_am_l604_604480

-- Define the variables and geometric conditions
variable (A B C D P Q M : Point)
variable [Parallelogram ABCD : IsParallelogram A B C D]

-- Conditions given in the problem
variable (h1 : Segment_constr A B P 3)
variable (h2 : Segment_constr A D Q 4)
variable (h3 : Intersect PQ AC M)

-- The statement to prove
theorem ac_equals_7_am 
  (h_AB_P : AB = 3 * AP)
  (h_AD_Q : AD = 4 * AQ)
  (h_PQ_AC_M: PQ ∩ AC = {M}) : AC = 7 * AM :=
sorry

end ac_equals_7_am_l604_604480


namespace solution_to_geometry_problem_l604_604580

noncomputable def geometry_problem 
    (K₁ K₂ : Circle)
    (A B X Y P Q : Point)
    (tangent_K1_A : Tangent (K₁, A))
    (tangent_K2_B : Tangent (K₂, B))
    (intersect_points : IntersectionPoints (K₁, K₂) X Y)
    (closer_X_to_AB : CloserTo (X, AB))
    (AX_intersect_K2_P : SecondIntersection (AX, K₂) P)
    (tangent_K2_P_Q : TangentIntersection (K₂, P, Q, AB))
    : Prop :=
    ∠(X, Y, B) = ∠(B, Y, Q)

theorem solution_to_geometry_problem : 
    geometry_problem K₁ K₂ A B X Y P Q tangent_K1_A tangent_K2_B intersect_points closer_X_to_AB AX_intersect_K2_P tangent_K2_P_Q :=
sorry

end solution_to_geometry_problem_l604_604580


namespace new_batch_decaf_percent_is_60_l604_604702

structure CoffeeStock :=
  (original_stock : ℕ)
  (percent_original_decaf : ℕ)
  (new_batch : ℕ)
  (total_percent_decaf : ℕ)

def calculate_decaf_percent (cs : CoffeeStock) := 
  let original_decaf := cs.percent_original_decaf * cs.original_stock / 100
  let total_stock := cs.original_stock + cs.new_batch
  let total_decaf := cs.total_percent_decaf * total_stock / 100
  let new_batch_decaf := total_decaf - original_decaf
  100 * new_batch_decaf / cs.new_batch

theorem new_batch_decaf_percent_is_60 (cs : CoffeeStock) (h₀ : cs.original_stock = 400)
  (h₁ : cs.percent_original_decaf = 25)
  (h₂ : cs.new_batch = 100)
  (h₃ : cs.total_percent_decaf = 32) : 
  calculate_decaf_percent cs = 60 :=
begin
  sorry
end

end new_batch_decaf_percent_is_60_l604_604702


namespace ab_absolute_value_l604_604573

theorem ab_absolute_value 
  (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : ∃ r s : ℤ, (r, r, s) = roots (λ x, x^3 + a * x^2 + b * x + 6 * a) ∧ r ≠ s) 
  : |a * b| = 546 :=
sorry

end ab_absolute_value_l604_604573


namespace carlos_gold_quarters_l604_604735

theorem carlos_gold_quarters:
  (let quarter_weight := 1 / 5 in
   let melt_value_per_ounce := 100 in
   let store_value_per_quarter := 0.25 in
   let quarters_per_ounce := 1 / quarter_weight in
   let total_melt_value := melt_value_per_ounce * quarters_per_ounce in
   let total_store_value := store_value_per_quarter * quarters_per_ounce in
   total_melt_value / total_store_value = 80) :=
by
  let quarter_weight := 1 / 5
  let melt_value_per_ounce := 100
  let store_value_per_quarter := 0.25
  let quarters_per_ounce := 1 / quarter_weight
  let total_melt_value := melt_value_per_ounce * quarters_per_ounce
  let total_store_value := store_value_per_quarter * quarters_per_ounce
  have : total_melt_value / total_store_value = 80 := sorry
  exact this

end carlos_gold_quarters_l604_604735


namespace find_angles_l604_604060

variables (A B C D E F G H : ℝ)
variables (x : ℝ)

-- Conditions
def pentagon_angles_convex (s : ℝ) : Prop := s = 540

def pentagon_ABCDE_angles : Prop := A = B ∧ B = C ∧ D = A + 50 ∧ pentagon_angles_convex (A + B + C + D + E)
def pentagon_BCFG_ANGLES : Prop := B = 88 ∧ C = 88 ∧ F = G + 10 ∧ H = G ∧ pentagon_angles_convex (B + C + F + G + H)

-- Proof Statement
theorem find_angles (h₁: pentagon_ABCDE_angles) (h₂: pentagon_BCFG_ANGLES) : 
  D = 138 ∧ G = 118 :=
sorry

end find_angles_l604_604060


namespace inscribed_circle_radius_integer_l604_604178

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l604_604178


namespace part_a_part_b_l604_604671

-- Part (a)
theorem part_a : ¬(∃ s : Finset ℝ, s.card = 7 ∧
  (∑ x in s.filter (λ x, x >= 10), x).card >= 4 ∧
  (∑ x in s, x) >= 43 ∧
  (∑ x in s, x) / 7 = 6) :=
by 
  sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n ≥ 4) :
  (∃ s : Finset ℝ, s.card = 2 * n + 1 ∧
    (s.filter (λ x, x >= 10)).card = n + 1 ∧
    (s.filter (λ x, x >= 1 ∧ x < 10)).card = n ∧
    (∑ x in s, x) = 12 * n + 6) :=
by 
  sorry

end part_a_part_b_l604_604671


namespace exists_solution_iff_l604_604397

theorem exists_solution_iff (m : ℝ) (x y : ℝ) :
  ((y = (3 * m + 2) * x + 1) ∧ (y = (5 * m - 4) * x + 5)) ↔ m ≠ 3 :=
by sorry

end exists_solution_iff_l604_604397


namespace intersection_range_of_curves_l604_604894

-- Define the problem in Lean 4
theorem intersection_range_of_curves (λ : ℝ) :
  (∀ (x y : ℝ), (2 * |x| - y - 4 = 0) ∧ (x^2 + λ * y^2 = 4)) ↔ λ ∈ set.Ico (-1/4 : ℝ) (1/4 : ℝ) :=
by
  sorry

end intersection_range_of_curves_l604_604894


namespace range_of_s_l604_604779

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

def s (n : ℕ) : ℕ :=
  if hn : n > 1 then
    let factors := n.factorization.to_list in
    factors.foldl (λ acc (p, a), acc + a * (p + 1)) 0
  else 0

theorem range_of_s :
  {m | ∃ n : ℕ, is_composite n ∧ s(n) = m} = {m | m > 5} :=
sorry

end range_of_s_l604_604779


namespace sum_of_solutions_eq_l604_604370

theorem sum_of_solutions_eq : 
  let solutions := {x | x = abs (3 * x - abs (120 - 3 * x))} in
  solutions.sum = (144 + 120 / 7) :=
by
  sorry

end sum_of_solutions_eq_l604_604370


namespace tear_paper_l604_604275

theorem tear_paper (n : ℕ) : 1 + 3 * n ≠ 2007 :=
by
  sorry

end tear_paper_l604_604275


namespace probability_same_color_is_correct_l604_604855

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l604_604855


namespace minimum_value_of_quadratic_l604_604829

theorem minimum_value_of_quadratic :
  ∀ x : ℝ, (x^2 + 8 * x + 15) ≥ -1 :=
begin
  intro x,
  sorry
end

end minimum_value_of_quadratic_l604_604829


namespace effective_price_of_coat_l604_604324

theorem effective_price_of_coat :
  let original_price := 50
  let first_discount_rate := 0.30
  let coupon_discount_rate := 0.15
  let tax_rate := 0.05
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_coupon := price_after_first_discount * (1 - coupon_discount_rate)
  let final_price := price_after_coupon * (1 + tax_rate)
  final_price ≈ 31.24 :=
by
  let original_price := 50
  let first_discount_rate := 0.30
  let coupon_discount_rate := 0.15
  let tax_rate := 0.05
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_coupon := price_after_first_discount * (1 - coupon_discount_rate)
  let final_price := price_after_coupon * (1 + tax_rate)
  have h : final_price =  (50 * (1 - 0.30)) * (1 - 0.15) * (1 + 0.05) := rfl
  have approx : final_price ≈ 31.24 := by norm_num [final_price]
  exact approx

end effective_price_of_coat_l604_604324


namespace find_ts_l604_604264

theorem find_ts :
  ∃ (t s : ℝ),
  (⟨2, 0⟩ : Fin 2 → ℝ) + t • ⟨7, -5⟩ = 
  (⟨1, -1⟩ : Fin 2 → ℝ) + s • ⟨-2, 3⟩ ∧
  t = -5/11 ∧ s = 12/11 :=
by {
  use [(-5/11 : ℝ), (12/11 : ℝ)],
  -- This is where the proof would go
  sorry
}

end find_ts_l604_604264


namespace infinite_seq_odd_nat_exists_l604_604199

theorem infinite_seq_odd_nat_exists :
  ∃ (m : ℕ → ℕ) (n : ℕ → ℕ),
  (∀ k : ℕ, m k % 2 = 1) ∧
  (∀ k : ℕ, n k ∈ ℕ) ∧
  (∀ k : ℕ, Nat.gcd (m k) (n k) = 1) ∧
  (∀ k : ℕ, ∃ r : ℕ, m k ^ 4 - 2 * n k ^ 4 = r ^ 2) :=
begin
  sorry
end

end infinite_seq_odd_nat_exists_l604_604199


namespace probability_same_color_l604_604870

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l604_604870


namespace inequality_am_gm_l604_604520

theorem inequality_am_gm 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) >= 6 := 
sorry

end inequality_am_gm_l604_604520


namespace solve_for_x_l604_604288

theorem solve_for_x :
  (exists x, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
begin
  sorry
end

end solve_for_x_l604_604288


namespace b_plus_one_prime_l604_604141

open Nat

def is_a_nimathur (a b : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ b / a → (a * n + 1) ∣ (Nat.choose (a * n) b - 1)

theorem b_plus_one_prime (a b : ℕ) 
  (ha : 1 ≤ a) 
  (hb : 1 ≤ b) 
  (h : is_a_nimathur a b) 
  (h_not : ¬is_a_nimathur a (b+2)) : 
  Prime (b + 1) := 
sorry

end b_plus_one_prime_l604_604141


namespace triangle_area_heron_l604_604682

theorem triangle_area_heron :
  let a := 13
  let b := 14
  let c := 15
  let s := (a + b + c) / 2
  s * (s - a) * (s - b) * (s - c) = 84 * 84 → (Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 84) :=
by
  intro h
  rw Real.sqrt_eq_r_pow_fin 2
  rw pow_two
  exact h

end triangle_area_heron_l604_604682


namespace area_of_rhombus_l604_604244

theorem area_of_rhombus (P D : ℕ) (area : ℝ) (hP : P = 48) (hD : D = 26) :
  area = 25 := by
  sorry

end area_of_rhombus_l604_604244


namespace radius_of_inscribed_circle_is_integer_l604_604181

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l604_604181


namespace find_vector_p_l604_604451

def vector_a : ℝ × ℝ × ℝ := (2, -2, 4)
def vector_b : ℝ × ℝ × ℝ := (3, 0, 3)

def collinear (u v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, u = (w.1 + t * (v.1 - w.1), w.2 + t * (v.2 - w.2), w.3 + t * (v.3 - w.3))

theorem find_vector_p : ∃ p : ℝ × ℝ × ℝ, collinear vector_a vector_b p ∧ p = (3, 0, 3) :=
by
  sorry

end find_vector_p_l604_604451


namespace fran_speed_calculation_l604_604943

theorem fran_speed_calculation:
  let Joann_speed := 15
  let Joann_time := 5
  let Fran_time := 4
  let Fran_speed := (Joann_speed * Joann_time) / Fran_time
  Fran_speed = 18.75 := by
  sorry

end fran_speed_calculation_l604_604943


namespace set_intersection_l604_604094

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l604_604094


namespace train_length_is_correct_l604_604341
-- Import the Mathlib library

-- Define the constants for the speed and time
def speed_km_per_hr : ℝ := 60
def crossing_time_s : ℝ := 3

-- Define the conversion constants
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Define the speed in m/s
def speed_m_per_s : ℝ := (speed_km_per_hr * km_to_m) / hr_to_s

-- Statement to be proved
theorem train_length_is_correct : (speed_m_per_s * crossing_time_s = 50.01) :=
begin
  -- Insert proof steps here
  sorry,
end

end train_length_is_correct_l604_604341


namespace cheaper_store_price_in_cents_correct_price_difference_in_cents_l604_604821

def list_price : ℝ := 78.50
def discount_mega_deals : ℝ := 0.12
def discount_quick_save : ℝ := 0.30
def additional_discount_mega_deals : ℝ := 5.00

theorem cheaper_store_price_in_cents :
  let price_mega_deals := (1 - discount_mega_deals) * list_price - additional_discount_mega_deals in
  let price_quick_save := (1 - discount_quick_save) * list_price in
  price_mega_deals - price_quick_save = 9.13 :=
by
  let price_mega_deals := (1 - discount_mega_deals) * list_price - additional_discount_mega_deals
  let price_quick_save := (1 - discount_quick_save) * list_price
  show price_mega_deals - price_quick_save = 9.13
  calc
    price_mega_deals - price_quick_save = (0.88 * 78.50 - 5.00) - (0.70 * 78.50) : by rw [price_mega_deals, price_quick_save]
    ... = 913 / 100 : by norm_num
    ... = 9.13 : by norm_num
  sorry

theorem correct_price_difference_in_cents : (9.13 * 100) = 913 := by norm_num

end cheaper_store_price_in_cents_correct_price_difference_in_cents_l604_604821


namespace field_dimensions_l604_604028

theorem field_dimensions (W L : ℕ) (h1 : L = 2 * W) (h2 : 2 * L + 2 * W = 600) : W = 100 ∧ L = 200 :=
sorry

end field_dimensions_l604_604028


namespace minimal_surface_area_cylinder_l604_604989

theorem minimal_surface_area_cylinder {r h V : ℝ}
  (vol_eq : V = π * r^2 * h)
  (surface_area_eq : S_A = 2 * π * r^2 + 2 * π * r * h) :
  (∀ V, h = 2 * r) :=
by
  sorry

end minimal_surface_area_cylinder_l604_604989


namespace octopus_legs_l604_604255

-- Definitions of octopus behavior based on the number of legs
def tells_truth (legs: ℕ) : Prop := legs = 6 ∨ legs = 8
def lies (legs: ℕ) : Prop := legs = 7

-- Statements made by the octopuses
def blue_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 28
def green_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 27
def yellow_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 26
def red_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 25

noncomputable def legs_b := 7
noncomputable def legs_g := 6
noncomputable def legs_y := 7
noncomputable def legs_r := 7

-- Main theorem
theorem octopus_legs : 
  (tells_truth legs_g) ∧ 
  (lies legs_b) ∧ 
  (lies legs_y) ∧ 
  (lies legs_r) ∧ 
  blue_statement legs_b legs_g legs_y legs_r ∧ 
  green_statement legs_b legs_g legs_y legs_r ∧ 
  yellow_statement legs_b legs_g legs_y legs_r ∧ 
  red_statement legs_b legs_g legs_y legs_r := 
by 
  sorry

end octopus_legs_l604_604255


namespace matrix_condition_l604_604526

variables (p q r s t u v w x : ℝ)

noncomputable def X : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![p, q, r], ![s, t, u], ![v, w, x]]

theorem matrix_condition (h : X p q r s t u v w x ^ T = (X p q r s t u v w x)⁻¹) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 + w^2 + x^2 = 3 := by
  sorry

end matrix_condition_l604_604526


namespace extremum_range_l604_604439

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * x^2 - a * x + 1

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 4 * x - a

theorem extremum_range 
  (h : ∀ a : ℝ, (∃ (x : ℝ) (hx : -1 < x ∧ x < 1), f_prime a x = 0) → 
                (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime a x ≠ 0)):
  ∀ a : ℝ, -1 < a ∧ a < 7 :=
sorry

end extremum_range_l604_604439


namespace triangle_circumcircle_ratio_l604_604492

/-- In triangle ABC, points P, Q, R lie on sides BC, CA, and AB, respectively.
Let ωA, ωB, ωC denote the circumcircles of triangles AQR, BRP, CPQ, respectively.
Given that segment AP intersects ωA, ωB, ωC again at X, Y, Z respectively,
prove that YX / XZ = BP / PC. -/
theorem triangle_circumcircle_ratio
  (A B C P Q R X Y Z : Point)
  (hP : P ∈ line_segment B C)
  (hQ : Q ∈ line_segment C A)
  (hR : R ∈ line_segment A B)
  (ωA : Circle)
  (ωB : Circle)
  (ωC : Circle)
  (hωA : circumcircle A Q R ωA)
  (hωB : circumcircle B R P ωB)
  (hωC : circumcircle C P Q ωC)
  (hX : X ∈ intersection (line_segment A P) ωA)
  (hY : Y ∈ intersection (line_segment A P) ωB)
  (hZ : Z ∈ intersection (line_segment A P) ωC) :
  (distance Y X / distance X Z) = (distance B P / distance P C) := 
sorry

end triangle_circumcircle_ratio_l604_604492


namespace tank_empty_time_correct_l604_604330

noncomputable def tank_time_to_empty (leak_empty_time : ℕ) (inlet_rate : ℕ) (tank_capacity : ℕ) : ℕ :=
(tank_capacity / (tank_capacity / leak_empty_time - inlet_rate * 60))

theorem tank_empty_time_correct :
  tank_time_to_empty 6 3 4320 = 8 := by
  sorry

end tank_empty_time_correct_l604_604330


namespace min_loadings_to_prove_first_ingot_weight_l604_604505

theorem min_loadings_to_prove_first_ingot_weight (weights : Fin 11 → ℕ) (all_weights : Set (Fin 11 → ℕ)) :
  (∀ (f : Fin 11 → ℕ), f ∈ all_weights → Finset.sum (Finset.univ.image f) = 66) ∧ 
  (∀ (f : Fin 11 → ℕ), f ∈ all_weights → ∃ S : Finset (Fin 11), Finset.card S = 4 ∧ Finset.sum (S.image f) ≤ 11) ∧ 
  (∀ (f : Fin 11 → ℕ), f ∈ all_weights → ∃ T : Finset (Fin 11), Finset.card T = 3 ∧ Finset.sum (T.image f) ≤ 11) →
  ∃ m : ℕ, m = 2 ∧
  (∀ f ∈ all_weights, ((∃ S : Finset (Fin 11), Finset.card S = 4 ∧ Finset.sum (S.image f) ≤ 11) ∧ 
                        (∃ T : Finset (Fin 11), Finset.card T = 3 ∧ Finset.sum (T.image f) ≤ 11)) →
   [Finset.single 0].image f = {1}) :=
begin
  sorry
end

end min_loadings_to_prove_first_ingot_weight_l604_604505


namespace sum_of_interior_angles_of_pentagon_l604_604610

theorem sum_of_interior_angles_of_pentagon :
    (5 - 2) * 180 = 540 := by 
  -- The proof goes here
  sorry

end sum_of_interior_angles_of_pentagon_l604_604610


namespace radius_of_inscribed_circle_is_integer_l604_604185

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l604_604185


namespace chicken_feed_cost_is_2_l604_604947

noncomputable def chicken_feed_cost (price_per_chicken : ℝ) (feed_per_bag : ℝ) (feed_per_chicken : ℝ) (num_chickens : ℕ) (revenue : ℝ) (profit : ℝ) : ℝ :=
  (revenue - profit) / (feed_per_bag / feed_per_chicken * num_chickens / feed_per_bag)

theorem chicken_feed_cost_is_2 : 
  ∀ (price_per_chicken : ℝ) (feed_per_bag : ℝ) (feed_per_chicken : ℝ) (num_chickens : ℕ) (revenue : ℝ) (profit : ℝ), 
    price_per_chicken = 1.5 → 
    feed_per_bag = 20 → 
    feed_per_chicken = 2 → 
    num_chickens = 50 → 
    revenue = 75 →
    profit = 65 →
    chicken_feed_cost price_per_chicken feed_per_bag feed_per_chicken num_chickens revenue profit = 2 := 
by
  intros price_per_chicken feed_per_bag feed_per_chicken num_chickens revenue profit h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  unfold chicken_feed_cost
  norm_num
  sorry

end chicken_feed_cost_is_2_l604_604947


namespace sin_half_angle_inequality_l604_604140

theorem sin_half_angle_inequality
  (R r : ℝ) (A : ℝ)
  (hR_pos : 0 < R)
  (hr_pos : 0 < r)
  (htriangle : ∃ B C : ℝ, ∠ A + ∠ B + ∠ C = π ) :
  (R - sqrt (R^2 - 2 * R * r)) / (2 * R) ≤ sin (A / 2) ∧ sin (A / 2) ≤ (R + sqrt (R^2 - 2 * R * r)) / (2 * R) := 
sorry

end sin_half_angle_inequality_l604_604140


namespace intersection_A_B_l604_604119

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l604_604119


namespace kind_wizard_succeeds_if_and_only_if_odd_l604_604685

theorem kind_wizard_succeeds_if_and_only_if_odd (n : ℕ) :
  (n.mod 2 = 1) ↔ (∃ (A : Fin n → Fin 2) (B : Fin n → Fin 2), 
  ∀ i, ((A i) = (B i) ∨ (A i) = (B (i + 1) % n)) ∧ 
  ∃ breaks : Fin (2 * n) → Prop, 
  (∃ count_breaks : ℕ, count_breaks = n ∧ (∀ k : Fin (2 * n), breaks k → k < n)) → 
  (∃ ensures : Fin n → Fin 2, 
  (∀ j : Fin n, (ensures j = ensures (j + 1) % n)))) :=
by sorry

end kind_wizard_succeeds_if_and_only_if_odd_l604_604685


namespace number_50_is_sample_size_l604_604625

def number_of_pairs : ℕ := 50
def is_sample_size (n : ℕ) : Prop := n = number_of_pairs

-- We are to show that 50 represents the sample size
theorem number_50_is_sample_size : is_sample_size 50 :=
sorry

end number_50_is_sample_size_l604_604625


namespace arctan_sum_l604_604878

theorem arctan_sum : 
  ∀ a b : ℝ, 
  a = 1 / 3 →
  (a + 1) * (b + 1) = 5 / 2 →
  arctan a + arctan b = arctan (29 / 17) := by
  intros a b ha h_eq
  sorry

end arctan_sum_l604_604878


namespace total_revenue_is_correct_l604_604055

theorem total_revenue_is_correct :
  let fiction_books := 60
  let nonfiction_books := 84
  let children_books := 42

  let fiction_sold_frac := 3/4
  let nonfiction_sold_frac := 5/6
  let children_sold_frac := 2/3

  let fiction_price := 5
  let nonfiction_price := 7
  let children_price := 3

  let fiction_sold_qty := fiction_sold_frac * fiction_books
  let nonfiction_sold_qty := nonfiction_sold_frac * nonfiction_books
  let children_sold_qty := children_sold_frac * children_books

  let total_revenue := fiction_sold_qty * fiction_price
                       + nonfiction_sold_qty * nonfiction_price
                       + children_sold_qty * children_price
  in total_revenue = 799 :=
by
  sorry

end total_revenue_is_correct_l604_604055


namespace parabola_circle_intersection_l604_604132

noncomputable def sum_of_distances_to_focus (a b c : ℝ) : ℝ :=
  (b^2 + 1 / 4) + (c^2 + 1 / 4) + (13^2 + 1 / 4) + (a^2 + 1 / 4) + 784 + 4 + 169 + (17^2 + 1 / 4)

theorem parabola_circle_intersection :
  ∃ a : ℝ, a - 28 - 2 + 13 = 0 ∧
  sum_of_distances_to_focus -28 -2 13 = 1247 :=
by
  have h : -28 + -2 + 13 + 17 = 0 := by linarith
  use 17
  split
  { exact h }
  { sorry }

end parabola_circle_intersection_l604_604132


namespace recreation_percentage_l604_604309

def wages_last_week (W : ℝ) : ℝ := W
def spent_on_recreation_last_week (W : ℝ) : ℝ := 0.15 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def spent_on_recreation_this_week (W : ℝ) : ℝ := 0.30 * (wages_this_week W)

theorem recreation_percentage (W : ℝ) (hW: W > 0) :
  (spent_on_recreation_this_week W) / (spent_on_recreation_last_week W) * 100 = 180 := by
  sorry

end recreation_percentage_l604_604309


namespace max_teams_tie_for_most_wins_l604_604907

theorem max_teams_tie_for_most_wins (n : ℕ) (h : n = 7) :
  ∃ k, k = 6 ∧ ∀ t : Finset ℕ, t.card = n → ⟨t.filter (λ x, wins x = 3)⟩.card = k :=
by sorry

end max_teams_tie_for_most_wins_l604_604907


namespace minimum_value_l604_604807

open Real

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : log (2^x) + log (8^y) = log 2) :
  ∃ (v : ℝ), v = 4 ∧ ∀ u, (∀ x y, x > 0 ∧ y > 0 → log (2^x) + log (8^y) = log 2 → x + 3*y = 1 → u = 4) := sorry

end minimum_value_l604_604807


namespace train_length_is_correct_l604_604340
-- Import the Mathlib library

-- Define the constants for the speed and time
def speed_km_per_hr : ℝ := 60
def crossing_time_s : ℝ := 3

-- Define the conversion constants
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Define the speed in m/s
def speed_m_per_s : ℝ := (speed_km_per_hr * km_to_m) / hr_to_s

-- Statement to be proved
theorem train_length_is_correct : (speed_m_per_s * crossing_time_s = 50.01) :=
begin
  -- Insert proof steps here
  sorry,
end

end train_length_is_correct_l604_604340


namespace peaches_left_at_stand_l604_604538

def initial_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def spoiled_peaches : ℝ := 12.0
def sold_peaches : ℝ := 27.0

theorem peaches_left_at_stand :
  initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 81.0 :=
by
  -- initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 84.0
  sorry

end peaches_left_at_stand_l604_604538


namespace range_of_a_for_decreasing_function_l604_604463

theorem range_of_a_for_decreasing_function (a : ℝ) :
  (∀ x : ℝ, x ≤ 4 → (deriv (λ x : ℝ, x^2 + 2*(a-1)*x + 2)) x ≤ 0) → a ≤ -3 := by
  sorry

end range_of_a_for_decreasing_function_l604_604463


namespace minimize_distance_l604_604429

noncomputable def f (x : ℝ) := 9 * x^3
noncomputable def g (x : ℝ) := Real.log x

theorem minimize_distance :
  ∃ m > 0, (∀ x > 0, |f m - g m| ≤ |f x - g x|) ∧ m = 1/3 :=
sorry

end minimize_distance_l604_604429


namespace ABC_is_isosceles_l604_604981

variables (Point : Type) [MetricSpace Point]
variables (O A B C O1 O2 : Point)
variables (is_incenter : Triangle Point → Point → Prop)
variables (is_excircle_center : Triangle Point → Point → Point → Prop)
variables (is_between : Point → Point → Point → Prop)
variables (lies_on_ray : Point → Point → Prop)

-- Given conditions
axiom A_lies_on_ray_OA : lies_on_ray A O
axiom B_and_C_lie_on_ray_OB : lies_on_ray B O ∧ lies_on_ray C O
axiom B_between_O_and_C : is_between O B C
axiom O1_is_incenter_OAB : is_incenter (Triangle.mk O A B) O1
axiom O2_is_excircle_center_OAC : is_excircle_center (Triangle.mk O A C) O2 C
axiom O1A_eq_O2A : dist O1 A = dist O2 A

-- Proof goal
theorem ABC_is_isosceles : Triangle.is_isosceles (Triangle.mk A B C) :=
by
  sorry

end ABC_is_isosceles_l604_604981


namespace contradiction_for_n3_min_elements_when_n_ge_4_l604_604673

theorem contradiction_for_n3 :
  ∀ (s : Set ℕ), (s.card = 7) → 
                 (∀ (x ∈ s), x ≥ 1) → 
                 (∃ t u : Set ℕ, (t.card = 4) ∧ (u.card = 3) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) 
                 → ∃ x ∈ s, false :=
sorry

theorem min_elements_when_n_ge_4 (n : ℕ) (hn : n ≥ 4) :
  ∃ (s : Set ℕ), (s.card = 2 * n + 1) ∧ 
                 (∀ (x ∈ s), x ≥ 1) ∧ 
                 (∃ t u : Set ℕ, (t.card = n + 1) ∧ (u.card = n) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) ∧
                 ∀ (s : Set ℕ), s.card = 2 * n + 1 → (∑ x in s, x) / (2 * n + 1) = 6 :=
sorry

example : ∃ s, (s.card = 9) ∧ (∀ x ∈ s, x ≥ 1) ∧ 
               (∃ t u : Set ℕ, (t.card = 5) ∧ (u.card = 4) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
               (∀ x ∈ t, x ≥ 10) ∧ (∀ x ∈ u, x ≥ 1) ∧
               (∑ x in s, x) / 9 = 6) :=
{ sorry }

end contradiction_for_n3_min_elements_when_n_ge_4_l604_604673


namespace nested_fraction_solution_l604_604212

noncomputable def golden_ratio : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem nested_fraction_solution (n : ℕ) (x : ℝ) (h : 1 + 1 / 1 + 1 / 1 + 1 / (1 + ... (1 / x)) = x) : 
  x = golden_ratio :=
sorry

end nested_fraction_solution_l604_604212


namespace solve_for_x_l604_604285

theorem solve_for_x : (∃ x : ℚ, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
by
  use 80 / 3
  sorry

end solve_for_x_l604_604285


namespace monotonicity_and_zero_of_h_when_a_is_one_range_of_a_if_f_eq_log2_g_has_two_roots_l604_604831

noncomputable def f (x : ℝ) : ℝ := Real.log2 ((x - 1) / (x + 1))

noncomputable def g (a x : ℝ) : ℝ := 3 * a * x + (1 - a)

noncomputable def h (a x : ℝ) : ℝ := f x + g a x

theorem monotonicity_and_zero_of_h_when_a_is_one :
  (∃ x ∈ Ioi (1 : ℝ), h 1 x = 0) ∧ (∀ x₁ x₂ ∈ Ioi (1 : ℝ), h 1 x₁ = 0 → h 1 x₂ = 0 → x₁ = x₂) :=
sorry

theorem range_of_a_if_f_eq_log2_g_has_two_roots :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = Real.log2 (g a x₁) ∧ f x₂ = Real.log2 (g a x₂)) → (a ∈ set.Ioo (-1/2 : ℝ) 0) :=
sorry

end monotonicity_and_zero_of_h_when_a_is_one_range_of_a_if_f_eq_log2_g_has_two_roots_l604_604831


namespace graphs_symmetric_l604_604488

noncomputable def exp2 : ℝ → ℝ := λ x => 2^x
noncomputable def log2 : ℝ → ℝ := λ x => Real.log x / Real.log 2

theorem graphs_symmetric :
  ∀ (x y : ℝ), (y = exp2 x) ↔ (x = log2 y) := sorry

end graphs_symmetric_l604_604488


namespace triangle_inequality_l604_604986

theorem triangle_inequality (a b c R : ℝ) (h1: a + b > c) (h2: b + c > a) (h3: c + a > b) (hR: R > 0) :
  (\frac{a^2}{b + c - a} + \frac{b^2}{a + c - b} + \frac{c^2}{a + b - c}) ≥ 3 * Real.sqrt 3 * R :=
sorry

end triangle_inequality_l604_604986


namespace like_terms_calc_l604_604421

theorem like_terms_calc {m n : ℕ} (h1 : m + 2 = 6) (h2 : n + 1 = 3) : (- (m : ℤ))^3 + (n : ℤ)^2 = -60 :=
  sorry

end like_terms_calc_l604_604421


namespace angle_PQR_eq_90_l604_604924

theorem angle_PQR_eq_90
  (R S P Q : Type)
  [IsStraightLine R S P]
  (angle_QSP : ℝ)
  (h : angle_QSP = 70) :
  ∠PQR = 90 :=
by
  sorry

end angle_PQR_eq_90_l604_604924


namespace negation_of_sum_of_squares_l604_604400

variables (a b : ℝ)

theorem negation_of_sum_of_squares:
  ¬(a^2 + b^2 = 0) → (a ≠ 0 ∨ b ≠ 0) := 
by
  sorry

end negation_of_sum_of_squares_l604_604400


namespace smallest_number_divisible_by_495_in_sequence_l604_604362

theorem smallest_number_divisible_by_495_in_sequence :
  ∃ n, (n > 0) ∧ (n.to_digits = List.replicate 18 5) ∧ (495 ∣ n) :=
sorry

end smallest_number_divisible_by_495_in_sequence_l604_604362


namespace sandys_speed_in_kph_l604_604988

def time_in_seconds := 99.9920006399488
def distance_in_meters := 500

def speed_in_kph (time: ℝ) (distance: ℝ) := (distance / time) * 3.6

theorem sandys_speed_in_kph :
  speed_in_kph time_in_seconds distance_in_meters = 18.000288 := 
  by 
  sorry

end sandys_speed_in_kph_l604_604988


namespace area_of_triangles_is_correct_l604_604472

noncomputable def area_triangle_ABC_plus_two_area_triangle_ADE 
  (A B C D E : Type) [metric_space A] [inner_product_space ℝ A] 
  (AB : ℝ) (angle_BAC angle_ABC angle_ACB angle_ADE : real.angle) : ℝ :=
sorry

theorem area_of_triangles_is_correct 
  {A B C D E : point} {AB : ℝ} {angle_BAC angle_ABC angle_ACB angle_ADE : real.angle}
  (h1 : AB = 2)
  (h2 : angle_BAC = 45)
  (h3 : angle_ABC = 75)
  (h4 : angle_ACB = 60)
  (h5 : angle_ADE = 45)
  (h6 : divides_side AB D 1 2)
  (h7 : divides_side BC E 2 1)
  : area_triangle_ABC_plus_two_area_triangle_ADE A B C D E AB angle_BAC angle_ABC angle_ACB angle_ADE = 0.76 :=
by sorry

end area_of_triangles_is_correct_l604_604472


namespace combined_total_cost_l604_604154

theorem combined_total_cost (x : ℕ) (h : 2 * x = x + 8) : x + x + 2 + 4 + 2 = 24 :=
by
  -- Use the given condition h to express x
  have h1: x = 8 := by
    linarith
    
  -- Substitute x = 8 in the combined total cost expression
  rw h1
  norm_num

end combined_total_cost_l604_604154


namespace maddie_weekend_watch_time_l604_604974

-- Defining the conditions provided in the problem
def num_episodes : ℕ := 8
def duration_per_episode : ℕ := 44
def minutes_on_monday : ℕ := 138
def minutes_on_tuesday : ℕ := 0
def minutes_on_wednesday : ℕ := 0
def minutes_on_thursday : ℕ := 21
def episodes_on_friday : ℕ := 2

-- Define the total time watched from Monday to Friday
def total_minutes_week : ℕ := num_episodes * duration_per_episode
def total_minutes_mon_to_fri : ℕ := 
  minutes_on_monday + 
  minutes_on_tuesday + 
  minutes_on_wednesday + 
  minutes_on_thursday + 
  (episodes_on_friday * duration_per_episode)

-- Define the weekend watch time
def weekend_watch_time : ℕ := total_minutes_week - total_minutes_mon_to_fri

-- The theorem to prove the correct answer
theorem maddie_weekend_watch_time : weekend_watch_time = 105 := by
  sorry

end maddie_weekend_watch_time_l604_604974


namespace Xiaoqiang_games_played_correct_l604_604352

-- Define the participants and their respective number of games played
structure Participants where
  Jia Yi Bing Ding Xiaoqiang : ℕ

-- Define the number of games played by each participant
def gamesPlayed : Participants :=
  { Jia := 4, Yi := 3, Bing := 2, Ding := 1, Xiaoqiang := 0 }  -- Initial count

-- Define a proof problem where we need to show Xiaoqiang played 2 games
theorem Xiaoqiang_games_played_correct :
  gamesPlayed.Xiaoqiang = 2 :=
sorry

end Xiaoqiang_games_played_correct_l604_604352


namespace min_nm_eq_16π_div_3_l604_604593

namespace Problem

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  sin (ω * x) + sqrt 3 * cos (ω * x) + 1

theorem min_nm_eq_16π_div_3
  (ω : ℝ)
  (h_period : 2 * Math.pi / ω = Math.pi)
  (h_zeros : ∀ m n : ℝ, ∀ x ∈ set.Icc m n, f x ω = 0 → n - m ≥ (5 * Math.pi + Math.pi / 3)) :
  ∃ m n : ℝ, n - m = 16 * Math.pi / 3 :=
sorry

end Problem

end min_nm_eq_16π_div_3_l604_604593


namespace cosine_angle_between_a_and_a_minus_b_perpendicular_vectors_find_t_l604_604843

-- Define vectors a and b
def vec_a : ℝ × ℝ × ℝ := (1, 0, 1)
def vec_b : ℝ × ℝ × ℝ := (1, 2, 0)

-- Compute the cosine of the angle between vec_a and vec_a - vec_b
theorem cosine_angle_between_a_and_a_minus_b :
  let sub_vec := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2, vec_a.3 - vec_b.3)
  let dot_product := vec_a.1 * sub_vec.1 + vec_a.2 * sub_vec.2 + vec_a.3 * sub_vec.3
  let norm_a := real.sqrt (vec_a.1^2 + vec_a.2^2 + vec_a.3^2)
  let norm_sub := real.sqrt (sub_vec.1^2 + sub_vec.2^2 + sub_vec.3^2)
  dot_product / (norm_a * norm_sub) = real.sqrt 10 / 10 := 
sorry

-- Compute the value of t such that (2a + b) is perpendicular to (a - tb)
theorem perpendicular_vectors_find_t :
  let vec_2a_plus_b := (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2, 2 * vec_a.3 + vec_b.3)
  ∃ t : ℝ, 
    let vec_a_minus_tb := (vec_a.1 - t * vec_b.1, vec_a.2 - t * vec_b.2, vec_a.3 - t * vec_b.3)
    let dot_product := vec_2a_plus_b.1 * vec_a_minus_tb.1 + vec_2a_plus_b.2 * vec_a_minus_tb.2 + vec_2a_plus_b.3 * vec_a_minus_tb.3
    dot_product = 0 ∧ t = 5 / 7 := 
sorry

end cosine_angle_between_a_and_a_minus_b_perpendicular_vectors_find_t_l604_604843


namespace complete_set_eq_real_l604_604972

def is_complete_set (A : set ℝ) : Prop :=
  ∀ a b : ℝ, a + b ∈ A → a * b ∈ A

theorem complete_set_eq_real 
  (A : set ℝ) (h_nonempty : ∃ x : ℝ, x ∈ A) (h_complete : is_complete_set A) : A = set.univ := 
sorry

end complete_set_eq_real_l604_604972


namespace intersection_A_B_l604_604125

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l604_604125


namespace angle_PQR_is_90_l604_604927

variable (R P Q S : Type) [EuclideanGeometry R P Q S]
variable (RSP_is_straight : straight_line R S P)
variable (angle_QSP : ∡Q S P = 70)

theorem angle_PQR_is_90 : ∡P Q R = 90 :=
by
  sorry

end angle_PQR_is_90_l604_604927


namespace probability_same_color_l604_604874

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l604_604874


namespace integer_solutions_l604_604991

theorem integer_solutions :
  ∀ x y : ℤ, 2 * y^2 - 2 * x * y + x + 9 * y - 2 = 0 ↔ (x = 9 ∧ y = 1) 
  ∨ (x = 2 ∧ y = 0) ∨ (x = 8 ∧ y = 2) ∨ (x = 3 ∧ y = -1) :=
by
  intros x y
  split
  sorry
  sorry

end integer_solutions_l604_604991


namespace numSolutions_eq_16_l604_604385

noncomputable def numSolutions (θ : ℝ) : ℕ :=
  let eqtn := tan (3 * π * cos θ) = cot (3 * π * sin θ)
  let domain := θ ∈ Ioo 0 (2 * π)
  if eqtn && domain then 1 else 0

theorem numSolutions_eq_16 : (finset.range 16).sum (λ i, numSolutions (i * 2 * π / 16 : ℝ)) = 16 :=
  by sorry

end numSolutions_eq_16_l604_604385


namespace sum_sqrt_a_plus_1_leq_sqrt_sum_ab_plus_9_l604_604967

noncomputable def a_b_c_in_domain (a b c : ℝ) := -1 < a ∧ a < 1 ∧ -1 < b ∧ b < 1 ∧ -1 < c ∧ c < 1

theorem sum_sqrt_a_plus_1_leq_sqrt_sum_ab_plus_9 (a b c : ℝ) 
  (h1 : a_b_c_in_domain a b c) 
  (h2 : a + b + c + a * b * c = 0) :
  (real.sqrt (a + 1) + real.sqrt (b + 1) + real.sqrt (c + 1)) <= 
  real.sqrt ((a * b) + (b * c) + (c * a) + 9) :=
sorry

end sum_sqrt_a_plus_1_leq_sqrt_sum_ab_plus_9_l604_604967


namespace range_of_b_plus_c_l604_604470

-- Define conditions
variables {A B C : ℝ} (a b c : ℝ)

-- The given conditions
@[simp] axiom condition1 : b^2 + c^2 - a^2 = b * c
@[simp] axiom condition2 : (cos B).neg > 0
@[simp] axiom condition3 : a = sqrt 3 / 2

-- State the theorem
theorem range_of_b_plus_c 
  (h1 : b^2 + c^2 - a^2 = b * c)
  (h2 : (cos B).neg > 0)
  (h3 : a = sqrt 3 / 2) :
  (sqrt 3 / 2) < b + c ∧ b + c < 3 / 2 := sorry

end range_of_b_plus_c_l604_604470


namespace chocolate_bar_min_breaks_l604_604322

theorem chocolate_bar_min_breaks (n : ℕ) (h : n = 40) : ∃ k : ℕ, k = n - 1 := 
by 
  sorry

end chocolate_bar_min_breaks_l604_604322


namespace find_p_l604_604958

variable (a b : Vector ℝ) [NonCollinear a b]
variable (p : ℝ)
variable (AB BC CD : Vector ℝ)

-- Conditions
def AB_def : AB = 2 • a + p • b := sorry
def BC_def : BC = a + b := sorry
def CD_def : CD = a - 2 • b := sorry
def collinear_AB_D : ∃ λ : ℝ, AB = λ • (BC + CD) := sorry

theorem find_p : p = -1 :=
by
  -- Given definitions
  have h1: AB = 2 • a + p • b := AB_def
  have h2: BC = a + b := BC_def
  have h3: CD = a - 2 • b := CD_def
  have h_collinear: ∃ λ : ℝ, AB = λ • (BC + CD) := collinear_AB_D

  -- Include the necessary steps to prove p = -1
  sorry

end find_p_l604_604958


namespace b_general_formula_c_sum_formula_l604_604811

-- Sequence {a_n} as an arithmetic sequence with common difference d > 0
variables (a : ℕ → ℕ) (d : ℕ)
axiom a_1 : a 1 = 2
axiom common_diff : ∀ n : ℕ, a (n + 1) = a n + d
axiom d_positive : d > 0

-- Geometric sequence condition
axiom geometric_condition : (a 3 + 2) * (a 6 - 4) = (a 4)^2

-- Sequence {b_n} with given sum relation
variables (b : ℕ → ℕ)
axiom b_sum_relation : ∀ n : ℕ, b 1 + b 2 + ... + b n = d * b n - d

-- General formula for sequence {b_n}
theorem b_general_formula : ∀ n : ℕ, b n = 2^n :=
sorry

-- Sequence {c_n} and its sum for first (2^n - 1) terms
variables (c : ℕ → ℕ) (b_0 : ℕ)
axiom b_0_def : b_0 = 1
axiom c_definition : ∀ k : ℕ, ∀ n : ℕ, n ∈ range b (k - 1) b k → c n = k

theorem c_sum_formula : ∀ n : ℕ, (∑ i in range (2^n - 1), c i) = (n - 1) * 2^n + 1 :=
sorry

end b_general_formula_c_sum_formula_l604_604811


namespace point_inside_circle_l604_604793

-- Define the condition of having a point P with a distance OP from the center O
variables (O P : Type) [metric_space O] [metric_space P]
def dist_OP (O P : O) : ℝ := 4

-- Define the radius of the circle O
def radius (O : Type) [metric_space O] : ℝ := 5

-- State the theorem
theorem point_inside_circle (O : Type) [metric_space O] [metric_space O]:
  let P := perdist_OP (O P) < radius O := radius sorry

end point_inside_circle_l604_604793


namespace problem_1_problem_2_problem_3_l604_604401

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

variable (t : ℝ)

theorem problem_1 :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = x^2 - 2 * x + 2) →
  f 0 = 2 ∧
  (∀ x, f (x + 1) - f x = 2 * x - 1) → 
  (∀ x, x^2 - 2 * x + 2 = f x) :=
by
  sorry

theorem problem_2 :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = x^2 - 2 * x + 2) →
  ∀ x, x ∈ Icc (-2 : ℝ) 2 → 
  1 ≤ f x ∧ f x ≤ 10 :=
by
  sorry

theorem problem_3 :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = x^2 - 2 * x + 2) →
  ∀ t, t ≥ 1 →
  (∀ x, x ∈ Icc t (t + 1) → 
  f x = t^2 - 2 * t + 2) ∧
  ∀ t, 0 < t ∧ t < 1 →
  ∃ x, x ∈ Icc t (t + 1) ∧ f x = 1 ∧
  ∀ t, t ≤ 0 →
  (∀ x, x ∈ Icc t (t + 1) → 
  f x = t^2 + 2 * t + 1) :=
by
  sorry

end problem_1_problem_2_problem_3_l604_604401


namespace initial_markup_percentage_l604_604710

-- Conditions:
-- 1. Initial price of the coat is $76.
-- 2. Increasing the price by $4 results in a 100% markup.
-- 3. A 100% markup implies the selling price is double the wholesale price.

theorem initial_markup_percentage (W : ℝ) (h1 : W + (76 - W) = 76)
  (h2 : 2 * W = 76 + 4) : (36 / 40) * 100 = 90 :=
by
  -- Using the conditions directly from the problem, we need to prove the theorem statement.
  sorry

end initial_markup_percentage_l604_604710


namespace is_incircle_center_on_line_pq_l604_604247

variable {A H B C X P Q R I J : Type}
variable [Incircle IABC : has_incircle (triangle IABC)]
variable [Tangent XB_IP : tangent XB P]
variable [Tangent XC_IQ : tangent XC Q]
variable [Tangent _Gamma_IR : tangent _Gamma R]
variable [Center_of_incircle_I : center_of_incircle I]
variable [Center_of_omega_J : center_of_circle J]
variable [Tangency_J_Gamma : tangent J _Gamma]
variable {IABC : Type}

-- Given: Quadrilateral A H B C is inscribed in circle ⟦_Gamma⟧
-- Given: X is the intersection of the diagonals
-- Given: Circle ⟦omega⟧ is tangent to segment ⟦[XB]⟧ at P, to segment ⟦[XC]⟧ at Q, and to ⟦_Gamma⟧ at R
-- Given: I is the center of the incircle of ⟦triangle ABC⟧
-- Given: J is the center of ⟦omega⟧, which is tangent to circle ⟦_Gamma⟧
-- Prove: I is on the line ⟦(PQ)⟧

theorem is_incircle_center_on_line_pq (hω : tangent (triangle IABC).omega) : 
  collinear I P Q :=
begin
  sorry
end

end is_incircle_center_on_line_pq_l604_604247


namespace domain_of_sqrt_l604_604226

theorem domain_of_sqrt (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by sorry

end domain_of_sqrt_l604_604226


namespace exponential_inequality_l604_604303

theorem exponential_inequality (k l m : ℕ) : 2^(k+1) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 :=
by
  sorry

end exponential_inequality_l604_604303


namespace find_correct_function_l604_604716

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → x ∈ I → y ∈ I → f x < f y

theorem find_correct_function :
  (∀ (f : ℝ → ℝ) (g : ℝ → ℝ),
     (f = (λ x, x^{1/3}) ∧ is_odd_function f ∧ is_monotonically_increasing_on f {x | x < 0}) →
     (g = (λ x, x^{-1}) ∨ g = (λ x, x^2) ∨ g = (λ x, x^{-1/2})) →
     ¬ (is_odd_function g ∧ is_monotonically_increasing_on g {x | x < 0})) := sorry

end find_correct_function_l604_604716


namespace triangle_circumcircle_ratio_l604_604493

/-- In triangle ABC, points P, Q, R lie on sides BC, CA, and AB, respectively.
Let ωA, ωB, ωC denote the circumcircles of triangles AQR, BRP, CPQ, respectively.
Given that segment AP intersects ωA, ωB, ωC again at X, Y, Z respectively,
prove that YX / XZ = BP / PC. -/
theorem triangle_circumcircle_ratio
  (A B C P Q R X Y Z : Point)
  (hP : P ∈ line_segment B C)
  (hQ : Q ∈ line_segment C A)
  (hR : R ∈ line_segment A B)
  (ωA : Circle)
  (ωB : Circle)
  (ωC : Circle)
  (hωA : circumcircle A Q R ωA)
  (hωB : circumcircle B R P ωB)
  (hωC : circumcircle C P Q ωC)
  (hX : X ∈ intersection (line_segment A P) ωA)
  (hY : Y ∈ intersection (line_segment A P) ωB)
  (hZ : Z ∈ intersection (line_segment A P) ωC) :
  (distance Y X / distance X Z) = (distance B P / distance P C) := 
sorry

end triangle_circumcircle_ratio_l604_604493


namespace system1_solution_system2_solution_l604_604569

-- For System (1)
theorem system1_solution :
  ∃ (x y : ℝ), 3 * x - 2 * y = 9 ∧ x + 2 * y = 3 ∧ x = 3 ∧ y = 0 :=
by
  use 3, 0
  split
  · norm_num
  split
  · norm_num
  split <;> norm_num

-- For System (2)
theorem system2_solution :
  ∃ (x y : ℝ), 0.3 * x - y = 1 ∧ 0.2 * x - 0.5 * y = 19 ∧ x = 370 ∧ y = 110 :=
by
  use 370, 110
  split
  · norm_num
  split
  · norm_num
  split <;> norm_num

end system1_solution_system2_solution_l604_604569


namespace perpendicular_lines_sufficient_not_necessary_l604_604581

theorem perpendicular_lines_sufficient_not_necessary (a : ℝ) :
  (∃ k₁ k₂ : ℝ, k₁ = -1 ∧ k₂ = 1 ∧ k₁ * k₂ = -1) → (∃ a = 1, ∃ b = -1, 
  ((1 * 1 + 1 * (-b)) = 0)) :=
by
  sorry

end perpendicular_lines_sufficient_not_necessary_l604_604581


namespace inscribed_circle_radius_l604_604335

theorem inscribed_circle_radius (r : ℝ) (radius : ℝ) (angle_deg : ℝ): 
  radius = 6 ∧ angle_deg = 120 ∧ (∀ θ : ℝ, θ = 60) → r = 3 := 
by
  sorry

end inscribed_circle_radius_l604_604335


namespace probability_same_color_is_correct_l604_604859

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l604_604859


namespace cube_root_simplification_l604_604208

theorem cube_root_simplification :
  (∛(20^3 + 30^3 + 60^3) = 10 * ∛(251)) :=
sorry

end cube_root_simplification_l604_604208


namespace max_value_of_f_l604_604384

noncomputable def f (x : ℝ) : ℝ := (√3) * (Real.sin (2 * x)) + 2 * (Real.sin x) + 4 * (√3) * (Real.cos x)

theorem max_value_of_f : ∃ x : ℝ, f x = 17 / 2 := 
by
  sorry

end max_value_of_f_l604_604384


namespace problem_1_problem_2_l604_604019

noncomputable def vector_a (ω x : ℝ) : ℝ × ℝ :=
  (√2 * Real.cos (ω * x), 1)

noncomputable def vector_b (ω x : ℝ) : ℝ × ℝ :=
  (2 * Real.sin (ω * x + Real.pi / 4), -1)

noncomputable def f (ω x : ℝ) : ℝ :=
  let a := vector_a ω x
  let b := vector_b ω x
  a.1 * b.1 + a.2 * b.2

axiom omega_bounds (ω : ℝ) : (1 / 4) ≤ ω ∧ ω ≤ (3 / 2)
axiom symmetry_axis (ω : ℝ) : 2 * ω * (5 * Real.pi / 8) + (Real.pi / 4) = (Real.pi / 2) + Int.pi
axiom alpha_beta_bounds (α β : ℝ) : (-Real.pi / 2) < α ∧ α < (Real.pi / 2) ∧ (-Real.pi / 2) < β ∧ β < (Real.pi / 2)
axiom f_alpha (α : ℝ) : f 1 (α / 2 - Real.pi / 8) = √2 / 3
axiom f_beta (β : ℝ) : f 1 (β / 2 - Real.pi / 8) = 2 * √2 / 3

theorem problem_1 : f 1 (3 * Real.pi / 4) = -1 := sorry

theorem problem_2 (α β : ℝ) :
  (f 1 (α / 2 - Real.pi / 8) = √2 / 3) →
  (f 1 (β / 2 - Real.pi / 8) = 2 * √2 / 3) →
  (α > -Real.pi / 2) →
  (α < Real.pi / 2) →
  (β > -Real.pi / 2) →
  (β < Real.pi / 2) →
  Real.cos (α - β) = (2 * √10 + 2) / 9 := sorry

end problem_1_problem_2_l604_604019


namespace order_of_operations_example_l604_604278

theorem order_of_operations_example :
  3^2 * 4 + 5 * (6 + 3) - 15 / 3 = 76 := by
  sorry

end order_of_operations_example_l604_604278


namespace solve_dividend_and_divisor_l604_604547

-- Definitions for base, digits, and mathematical relationships
def base := 5
def P := 1
def Q := 2
def R := 3
def S := 4
def T := 0
def Dividend := 1 * base^6 + 2 * base^5 + 3 * base^4 + 4 * base^3 + 3 * base^2 + 2 * base^1 + 1 * base^0
def Divisor := 2 * base^2 + 3 * base^1 + 2 * base^0

-- The conditions given in the math problem
axiom condition_1 : Q + R = base
axiom condition_2 : P + 1 = Q
axiom condition_3 : Q + P = R
axiom condition_4 : S = 2 * Q
axiom condition_5 : Q^2 = S
axiom condition_6 : Dividend = 24336
axiom condition_7 : Divisor = 67

-- The goal
theorem solve_dividend_and_divisor : Dividend = 24336 ∧ Divisor = 67 :=
by {
  sorry
}

end solve_dividend_and_divisor_l604_604547


namespace find_four_digit_number_l604_604091

-- Definitions of the digit variables a, b, c, d, and their constraints.
def four_digit_expressions_meet_condition (abcd abc ab : ℕ) (a : ℕ) :=
  ∃ (b c d : ℕ), abcd = (1000 * a + 100 * b + 10 * c + d)
  ∧ abc = (100 * a + 10 * b + c)
  ∧ ab = (10 * a + b)
  ∧ abcd - abc - ab - a = 1787

-- Main statement to be proven.
theorem find_four_digit_number
: ∀ a b c d : ℕ, 
  four_digit_expressions_meet_condition (1000 * a + 100 * b + 10 * c + d) (100 * a + 10 * b + c) (10 * a + b) a
  → (a = 2 ∧ b = 0 ∧ ((c = 0 ∧ d = 9) ∨ (c = 1 ∧ d = 0))) :=
sorry

end find_four_digit_number_l604_604091


namespace problem_constant_term_binomial_l604_604423

open Real

theorem problem_constant_term_binomial :
  let a := ∫ x in 0..(π / 2), sin x + cos x
  let f := (a * (fun (x : ℝ) => x) - (fun (x : ℝ) => 1 / x))^6
  constant_term f = -160 :=
by
  sorry

end problem_constant_term_binomial_l604_604423


namespace math_problem_l604_604788

noncomputable def a (b : ℝ) : ℝ := 
  sorry -- to be derived from the conditions

noncomputable def b : ℝ := 
  sorry -- to be derived from the conditions

theorem math_problem (a b: ℝ) 
  (h1: a - b = 1)
  (h2: a^2 - b^2 = -1) : 
  a^2008 - b^2008 = -1 := 
sorry

end math_problem_l604_604788


namespace product_sign_of_39_numbers_l604_604259

theorem product_sign_of_39_numbers (a : Fin 39 → ℝ) (h1 : ∀ i : Fin 38, a i + a (i + 1) > 0) (h2 : ∑ i, a i < 0) :
  (∏ i, a i) > 0 :=
sorry

end product_sign_of_39_numbers_l604_604259


namespace exists_rectangle_with_equal_lattice_points_l604_604253

noncomputable theory
open Real

theorem exists_rectangle_with_equal_lattice_points :
  ∃ (m n : ℕ), 
    let horizontal_distance := 1
    let vertical_distance := sqrt 3
    let boundary_points := 2 * (m + n)
    let interior_points := 2 * (m * n) - m - n + 1
    boundary_points = interior_points :=
by
  sorry

end exists_rectangle_with_equal_lattice_points_l604_604253


namespace radius_of_inscribed_circle_is_integer_l604_604179

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l604_604179


namespace degree_to_radian_60_eq_pi_div_3_l604_604638

theorem degree_to_radian_60_eq_pi_div_3 (pi : ℝ) (deg : ℝ) 
  (h : 180 * deg = pi) : 60 * deg = pi / 3 := 
by
  sorry

end degree_to_radian_60_eq_pi_div_3_l604_604638


namespace decreasing_interval_implies_m_le_neg4_l604_604830

theorem decreasing_interval_implies_m_le_neg4 (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici (-1) → deriv (λ x, -2 * x^2 + m * x - 3) x ≤ 0) → m ≤ -4 :=
by
  -- This will create a goal that asks us to prove the range of m
  -- given the decreasing condition on the interval [-1, ∞).
  sorry

end decreasing_interval_implies_m_le_neg4_l604_604830


namespace rectangles_can_be_colored_l604_604161

structure Rectangle :=
  (x y width height : ℕ)
  (width_odd : width % 2 = 1)
  (height_odd : height % 2 = 1)

def color (r : Rectangle) : ℕ :=
  if r.x % 2 = 0 then
    if r.y % 2 = 0 then 1 else 2
  else
    if r.y % 2 = 0 then 3 else 4

def share_boundary (r1 r2 : Rectangle) : Prop :=
  (r1.x = r2.x ∧ r1.y + r1.height = r2.y) ∨
  (r1.x = r2.x ∧ r2.y + r2.height = r1.y) ∨
  (r1.y = r2.y ∧ r1.x + r1.width = r2.x) ∨
  (r1.y = r2.y ∧ r2.x + r2.width = r1.x)

theorem rectangles_can_be_colored :
  ∀ (rs : List Rectangle), 
    (∀ (r1 r2 : Rectangle), r1 ∈ rs → r2 ∈ rs → r1 ≠ r2 → ¬(share_boundary r1 r2)) →
    (∀ (r1 r2 : Rectangle), r1 ∈ rs → r2 ∈ rs → (share_boundary r1 r2) → color r1 ≠ color r2) :=
begin
  intros rs no_intersection r1 r2 r1_in_rs r2_in_rs share_bndry,
  -- Proof to be done
  sorry
end

end rectangles_can_be_colored_l604_604161


namespace final_image_of_F_l604_604236

noncomputable def transform_letter_F : (ℝ × ℝ) → (ℝ × ℝ)
| (x, y)  := -- apply transformations
    let (x1, y1) := (-x, -y)          -- 180° rotation clockwise
    let (x2, y2) := (x1, -y1)         -- reflection in x-axis
    (-y2, x2)                         -- 90° rotation counterclockwise

theorem final_image_of_F :
    transform_letter_F (-1, -1) = (1, -1) :=
    by
        -- First transformation: Rotation 180° clockwise around the origin
        -- Initial position: Base along the negative x-axis, stem along negative y-axis
        -- After rotation: Base along positive x-axis, stem along positive y-axis
        let (x1, y1) := (-(-1), -(-1))
        have step1 : (x1, y1) = (1, 1), by simp

        -- Second transformation: Reflection in the x-axis
        -- Position after step1: Base along positive x-axis, stem along positive y-axis
        -- After reflection: Base along positive x-axis, stem along negative y-axis
        let (x2, y2) := (x1, -y1)
        have step2 : (x2, y2) = (1, -1), by simp

        -- Final transformation: Rotation 90° counterclockwise around the origin
        -- Position after step2: Base along positive x-axis, stem along negative y-axis
        -- After rotation: Base along positive y-axis, stem along negative x-axis
        let (xf, yf) := (-y2, x2)
        have step3 : (xf, yf) = (1, -1), by simp

        -- Final image verification
        exact step3

end final_image_of_F_l604_604236


namespace probability_multiple_of_3_l604_604694

theorem probability_multiple_of_3 : 
  let digits := {1, 2, 3, 4, 5}
  (number_of_three_digit_multiples_3 / number_of_all_three_digit) = 3 / 10 :=
by
  let digits := {1, 2, 3, 4, 5}
  let all_combinations := comb (digits, 3)
  let multiples_of_3 := filter (λ s, (sum s) % 3 == 0) all_combinations
  have total_combinations : nat := all_combinations.count
  have valid_combinations : nat := multiples_of_3.count
  suffices : valid_combinations / total_combinations = 3 / 10, from this
sorry

end probability_multiple_of_3_l604_604694


namespace subset_relation_l604_604450

variables (M N : Set ℕ) 

theorem subset_relation (hM : M = {1, 2, 3, 4}) (hN : N = {2, 3, 4}) : N ⊆ M :=
sorry

end subset_relation_l604_604450


namespace find_ratio_l604_604064

variables {EF GH EH EG EQ ER ES Q R S : ℝ}
variables (x : ℝ)
variables (E F G H : ℝ)

-- Conditions
def is_parallelogram : Prop := 
  -- Placeholder for parallelogram properties, not relevant for this example
  true

def point_on_segment (Q R : ℝ) (segment_length: ℝ) (ratio: ℝ): Prop := Q = segment_length * ratio ∧ R = segment_length * ratio

def intersect (EG QR : ℝ) (S : ℝ): Prop := 
  -- Placeholder for segment intersection properties, not relevant for this example
  true

-- Question
theorem find_ratio 
  (H_parallelogram: is_parallelogram)
  (H_pointQ: point_on_segment EQ ER EF (1/8))
  (H_pointR: point_on_segment ER ES EH (1/9))
  (H_intersection: intersect EG QR ES):
  (ES / EG) = (1/9) := 
by
  sorry

end find_ratio_l604_604064


namespace new_avg_mark_remaining_students_l604_604997

-- Definitions of the given conditions
def avg_mark_class : ℕ := 72
def total_students : ℕ := 13
def avg_mark_excluded : ℕ := 40
def num_excluded : ℕ := 5

-- Target to prove
theorem new_avg_mark_remaining_students : 
  let remaining_students := total_students - num_excluded in
  let total_marks_class := total_students * avg_mark_class in
  let total_marks_excluded := num_excluded * avg_mark_excluded in
  let total_marks_remaining := total_marks_class - total_marks_excluded in
  total_marks_remaining / remaining_students = 92 :=
by
  sorry

end new_avg_mark_remaining_students_l604_604997


namespace set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604664

-- Problem 1: Prove that a set of 7 numbers cannot have an arithmetic mean of 6 under given conditions.
theorem set_of_7_numbers_not_possible {s : Finset ℝ} (h_card : s.card = 7) (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / 7 = 6) : False := sorry

-- Problem 2: Prove that the number of elements in the set must be at least 9 if the arithmetic mean is 6.
theorem minimum_elements_with_mean_6 {s : Finset ℝ} (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / s.card = 6) (h_card : s.card = 2 * (s.card / 2) + 1) : 8 < s.card := sorry

end set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604664


namespace inscribed_circle_radius_integer_l604_604174

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l604_604174


namespace polynomial_square_binomial_l604_604371

-- Define the given polynomial and binomial
def polynomial (x : ℚ) (a : ℚ) : ℚ :=
  25 * x^2 + 40 * x + a

def binomial (x b : ℚ) : ℚ :=
  (5 * x + b)^2

-- Theorem to state the problem
theorem polynomial_square_binomial (a : ℚ) : 
  (∃ b, polynomial x a = binomial x b) ↔ a = 16 :=
by
  sorry

end polynomial_square_binomial_l604_604371


namespace number_of_ways_to_select_cells_l604_604232

-- Define the selection problem
theorem number_of_ways_to_select_cells (n : ℕ) :
  let total_ways := factorial n ^ (n^2) * factorial (n^2)
  in total_ways == (n!)^(n^2) * (n^2)! :=
sorry

end number_of_ways_to_select_cells_l604_604232


namespace percentage_increase_in_efficiency_l604_604560

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l604_604560


namespace bisector_sum_b_c_l604_604345

noncomputable def vertices : ℕ × ℕ × ℕ := ((2,3), (-4,1), (5,-6))

theorem bisector_sum_b_c :
  let (A, B, C) := vertices in
  ∃ (b c : ℤ), 3 * A.1 + b * A.2 + c = 0 ∧ b + c = -2 :=
by
  sorry

end bisector_sum_b_c_l604_604345


namespace solution_proof_l604_604761

noncomputable def a : ℝ := (2^20 - 1)^(1/20)
noncomputable def b : ℝ := -(a / 2)
def p : ℝ := -1
def q : ℝ := 1 / 4

theorem solution_proof :
  ∀ x : ℝ, (2 * x - 1)^20 - (a * x + b)^20 = (x^2 + p * x + q)^10 :=
by
  intro x
  sorry

end solution_proof_l604_604761


namespace area_of_triangle_l604_604010

noncomputable def parabola (p : ℝ) : ℝ × ℝ → Prop :=
λ (x y : ℝ), y^2 = 2 * p * x

noncomputable def hyperbola : ℝ × ℝ → Prop :=
λ (x y : ℝ), x^2 - y^2 / 3 = 1

theorem area_of_triangle (p x_A y_A x_F y_F x_K : ℝ)
  (A_on_parabola : parabola p (x_A, y_A))
  (F_is_right_focus : F = (2, 0))
  (F_is_parabola_focus : F = (2, 0))
  (directrix_intersects_x_axis : x_K)
  (A_on_parabola_cond : |AK| = sqrt(2) * |AF|)
  : 1 / 2 * |AK| * |KF| * sin (π / 4) = 8 := 
  sorry

end area_of_triangle_l604_604010


namespace john_average_speed_l604_604937

theorem john_average_speed
  (uphill_distance : ℝ)
  (uphill_time : ℝ)
  (downhill_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time_is_45_minutes : uphill_time = 45)
  (downhill_time_is_15_minutes : downhill_time = 15)
  (uphill_distance_is_3_km : uphill_distance = 3)
  (downhill_distance_is_3_km : downhill_distance = 3)
  : (uphill_distance + downhill_distance) / ((uphill_time + downhill_time) / 60) = 6 := 
by
  sorry

end john_average_speed_l604_604937


namespace proof_problem_l604_604965

-- Definitions and Conditions
def E : Set ℕ := {n | 1 ≤ n ∧ n ≤ 200}
def G : Set ℕ := {a | a ∈ E ∧ ∃ (i : ℕ), 1 ≤ i ∧ i ≤ 100}

variable (G : Set ℕ)
#check E -- Set ℕ
#check G -- Set ℕ

-- Condition 1: For any \( 1 \leq i \leq j \leq 100 \), \( a_{i} + a_{j} \neq 201 \)
def cond_I : Prop := ∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 100) → (1 ≤ j ∧ j ≤ 100) → (G i) → (G j) → i + j ≠ 201

-- Condition 2: \(\sum_{i = 1}^{100} a_{i} = 10080\)
def cond_II (G : Finset ℕ) : Prop := G.sum id = 10080

-- Theorem: Number of odd numbers in \( G \) is a multiple of 4,
-- and the sum of squares of all numbers in \( G \) is a constant.
theorem proof_problem (G : Set ℕ) (hG_sub : G ⊆ E) (h_cond_I : cond_I G) (h_cond_II : cond_II (G.to_finset)) :
  (∃ k : ℕ, ∀ (a ∈ G), a % 2 = 1 → k % 4 = 0) ∧ (∃ c : ℕ, ∀ (a ∈ G), sum (λ g, g^2) = c) :=
begin
  sorry
end

end proof_problem_l604_604965


namespace checkered_fabric_cost_l604_604633

variable (P : ℝ) (cost_per_yard : ℝ) (total_yards : ℕ)
variable (x : ℝ) (C : ℝ)

theorem checkered_fabric_cost :
  P = 45 ∧ cost_per_yard = 7.50 ∧ total_yards = 16 →
  C = cost_per_yard * (total_yards - x) →
  7.50 * (16 - x) = 45 →
  C = 75 :=
by
  intro h1 h2 h3
  sorry

end checkered_fabric_cost_l604_604633


namespace percent_increase_perimeter_fifth_triangle_l604_604354

noncomputable def side_length_after_n_increases (initial_length : Float) (factor : Float) (n : Nat) : Float :=
  initial_length * (factor ^ n)

def perimeter (side_length : Float) : Float :=
  3 * side_length

def percent_increase (initial_value : Float) (final_value : Float) : Float :=
  ((final_value - initial_value) / initial_value) * 100

theorem percent_increase_perimeter_fifth_triangle :
  let initial_side := 3.0
  let factor := 1.25
  let n := 4 -- since the first triangle is the starting point, we do 4 multiplicative increases to reach the 5th triangle
  let side_length_5 := side_length_after_n_increases initial_side factor n
  let perimeter_1 := perimeter initial_side
  let perimeter_5 := perimeter side_length_5
  percent_increase perimeter_1 perimeter_5 = 144.1 :=
by
  sorry

end percent_increase_perimeter_fifth_triangle_l604_604354


namespace tan_double_angle_identity_l604_604801

theorem tan_double_angle_identity (α : ℝ) 
    (h : sin α - 2 * cos α = sqrt 10 / 2) : tan (2 * α) = 3 / 4 :=
sorry

end tan_double_angle_identity_l604_604801


namespace sum_of_interior_angles_of_pentagon_l604_604611

theorem sum_of_interior_angles_of_pentagon :
    (5 - 2) * 180 = 540 := by 
  -- The proof goes here
  sorry

end sum_of_interior_angles_of_pentagon_l604_604611


namespace find_g_1_l604_604515

noncomputable def g (x : ℝ) : ℝ := sorry -- express g(x) as a 4th degree polynomial with unknown coefficients

-- Conditions given in the problem
axiom cond1 : |g (-1)| = 15
axiom cond2 : |g (0)| = 15
axiom cond3 : |g (2)| = 15
axiom cond4 : |g (3)| = 15
axiom cond5 : |g (4)| = 15

-- The statement we need to prove
theorem find_g_1 : |g 1| = 11 :=
sorry

end find_g_1_l604_604515


namespace range_of_PF_dot_PA_exists_fixed_point_l604_604819

-- Definition of the ellipse
def ellipse (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1

-- Points F and A
def F := (-1 : ℝ, 0 : ℝ)
def A := (-2 : ℝ, 0 : ℝ)

-- Part (1)
theorem range_of_PF_dot_PA (P : ℝ × ℝ) (hP : ellipse P.1 P.2) : 
  0 ≤ (P.1 + 1) * (P.1 + 2) + P.2 ^ 2 ∧ (P.1 + 1) * (P.1 + 2) + P.2 ^ 2 ≤ 12 := sorry

-- Additional conditions for part (2)
variables (k m : ℝ)

-- Line equation
def line (x y : ℝ) : Prop := y = k * x + m

-- Intersection points of the line and the ellipse
def M := (x1 : ℝ, y1 : ℝ)
def N := (x2 : ℝ, y2 : ℝ)

-- Perpendicular condition
def perpendicular (H : ℝ × ℝ) : Prop := 
  let A_H := (H.1 + 2, H.2)
  let HM := (M.1 - H.1, M.2 - H.2)
  let HN := (N.1 - H.1, N.2 - H.2)
  (H.1 + 2)^2 + H.2^2 = HM.1 * HN.1 + HM.2 * HN.2

-- Main theorem
theorem exists_fixed_point (H : ℝ × ℝ) (hH : perpendicular H) :
  ∀ x y : ℝ, line x y → x = -2/7 ∧ y = 0 := sorry

end range_of_PF_dot_PA_exists_fixed_point_l604_604819


namespace sum_of_diagonals_l604_604952

theorem sum_of_diagonals (FG HI GH IJ FJ : ℝ) (p q : ℕ) (h_rel_prime: p.coprime q)
  (h_FG : FG = 4)
  (h_HI : HI = 4)
  (h_GH : GH = 9)
  (h_IJ : IJ = 9)
  (h_FJ : FJ = 12)
  (h_sum_diag : 3 * 12 + (63 / 4) + (128 / 9) = (p: ℝ) / q) :
  p + q = 1169 :=
by
  sorry

end sum_of_diagonals_l604_604952


namespace roots_of_polynomial_l604_604767

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end roots_of_polynomial_l604_604767


namespace steve_halfway_time_longer_l604_604678

theorem steve_halfway_time_longer :
  ∀ (Td: ℝ) (Ts: ℝ),
  Td = 33 →
  Ts = 2 * Td →
  (Ts / 2) - (Td / 2) = 16.5 :=
by
  intros Td Ts hTd hTs
  rw [hTd, hTs]
  sorry

end steve_halfway_time_longer_l604_604678


namespace game_draw_fraction_l604_604271

theorem game_draw_fraction (p_B p_S : ℝ) (hB : p_B = 5/12) (hS : p_S = 1/4) :
  1 - (p_B + p_S) = 1/3 :=
by
  rw [hB, hS]
  norm_num
  sorry

end game_draw_fraction_l604_604271


namespace arithmetic_sequence_sum_l604_604994

noncomputable def binomial_constant_term : ℚ := 
  let r := 2 in 
  (-1 / 3) ^ r * (Nat.choose 6 r) * (1 : ℚ)

def a_5 : ℚ := binomial_constant_term
def arithmetic_sequence (n : ℕ) : ℚ :=
  a_5 / 5 * n  -- Represents an arithmetic sequence

theorem arithmetic_sequence_sum :
  arithmetic_sequence 3 + arithmetic_sequence 5 + arithmetic_sequence 7 = 5 := by
  sorry

end arithmetic_sequence_sum_l604_604994


namespace axis_of_symmetry_max_min_f_in_interval_range_m_l604_604001

noncomputable theory
open Real

def f (x : ℝ) : ℝ := 2 * cos (x - π / 4) ^ 2 - sqrt 3 * cos (2 * x) + 1

theorem axis_of_symmetry :
    ∃ k : Int, (λ x : ℝ, f x) = (λ x : ℝ, f (k * π / 2 + π / 4)) :=
sorry

theorem max_min_f_in_interval :
    ∃ x_min x_max ∈ (Icc (π / 4) (π / 2)), f x_min = 3 ∧ f x_max = 4 :=
sorry

theorem range_m (m : ℝ) :
    (∀ x ∈ Icc (π / 4) (π / 2), abs (f x - m) < 2) ↔ 2 < m ∧ m < 5 :=
sorry

end axis_of_symmetry_max_min_f_in_interval_range_m_l604_604001


namespace inscribed_circle_radius_integer_l604_604191

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l604_604191


namespace perimeter_sum_l604_604964

open Real

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem perimeter_sum :
  let A : point := (0, 0)
  let B : point := (3, 4)
  let C : point := (6, 1)
  let D : point := (9, -3)
  let dAB := distance A B
  let dBC := distance B C
  let dCD := distance C D
  let dDA := distance D A
  (dAB + dBC + dCD + dDA = 10 + 3 * Real.sqrt 2 + 3 * Real.sqrt 10) ∧
  (3 + 3 + 2 + 10 = 18) :=
by
  sorry

end perimeter_sum_l604_604964


namespace solve_for_x_l604_604287

theorem solve_for_x :
  (exists x, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
begin
  sorry
end

end solve_for_x_l604_604287


namespace eq_solutions_l604_604877

theorem eq_solutions (a b : ℝ) (h : a + b = 0) : 
  (∃ x, ax + b = 0 ∧ ∀ y, ay + b = 0 → y = x) ∨ (∀ x, ax + b = 0) :=
begin
  sorry
end

end eq_solutions_l604_604877


namespace infinite_solutions_exists_l604_604390

theorem infinite_solutions_exists :
  ∃ (a b c : ℕ), ∀ (n : ℕ),
    n ≠ 0 →
    a = 6 * n ∧ b = 3 * n ∧ c = 2 * n ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    ((a + b) * c = a * b) :=
by {
  use [λ n, 6 * n, λ n, 3 * n, λ n, 2 * n],
  intros n hn,
  split; try { norm_num }, 
  split; try { linarith },
  split; norm_num,
  linarith,
  sorry
}

end infinite_solutions_exists_l604_604390


namespace fraction_of_odd_products_to_hundredth_l604_604908

theorem fraction_of_odd_products_to_hundredth (n : ℕ) (h : n = 15) :
  let total_products := (h + 1) * (h + 1),
      odd_count := finset.card (finset.filter (λ x, x % 2 = 1) (finset.range (h + 1))),
      odd_products := odd_count * odd_count,
      fraction := (odd_products : ℚ) / total_products
  in real.to_nnreal (fraction : ℝ) = 0.25 := by
  sorry

end fraction_of_odd_products_to_hundredth_l604_604908


namespace convex_quadrilateral_probability_l604_604373

noncomputable def numberOfChords (n : ℕ) : ℕ := Nat.choose n 2
noncomputable def numberOfWaysToSelectChords (n : ℕ) (k : ℕ) : ℕ := Nat.choose (numberOfChords n) k
noncomputable def numberOfWaysToSelectPoints (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem convex_quadrilateral_probability :
  let totalChords := numberOfChords 8,
      totalWays := numberOfWaysToSelectChords 8 4,
      favorableWays := numberOfWaysToSelectPoints 8 4
  in (favorableWays : ℚ) / totalWays = 2 / 585 := by
  sorry

end convex_quadrilateral_probability_l604_604373


namespace num_possible_n_values_l604_604075

noncomputable def triangle_side_lengths_satisfy_conditions (n : ℕ) : Prop :=
  let AB := 2 * n + 15
  let BC := 3 * n
  let AC := 2 * n^2 - 2 * n + 8
  (AB + AC > BC) ∧ (AB + BC > AC) ∧ (AC + BC > AB) ∧ (AB > AC) ∧ (AC > BC)

theorem num_possible_n_values : 
  { n : ℕ | n > 0 ∧ triangle_side_lengths_satisfy_conditions n }.finite ∧
  { n : ℕ | n > 0 ∧ triangle_side_lengths_satisfy_conditions n }.to_finset.card = 5 := 
sorry

end num_possible_n_values_l604_604075


namespace negation_of_exists_geq_prop_l604_604240

open Classical

variable (P : Prop) (Q : Prop)

-- Original proposition:
def exists_geq_prop : Prop := 
  ∃ x : ℝ, x^2 + x + 1 ≥ 0

-- Its negation:
def forall_lt_neg : Prop :=
  ∀ x : ℝ, x^2 + x + 1 < 0

-- The theorem to prove:
theorem negation_of_exists_geq_prop : ¬ exists_geq_prop ↔ forall_lt_neg := 
by 
  -- The proof steps will be filled in here
  sorry

end negation_of_exists_geq_prop_l604_604240


namespace no_seven_lines_intersection_l604_604984

theorem no_seven_lines_intersection (
  h₁: ∃ (lines : Fin 7 → ℝ × ℝ → ℝ), True, -- There are 7 lines on a plane
  h₂: ∀ (P : ℝ × ℝ), 
        (∃ (lines_intersecting_3 : Finset (Fin 7)), 
          lines_intersecting_3.card = 3 
          ∧ (∀ l ∈ lines_intersecting_3, (lines l P.1, lines l P.2) = (0, 0)))
        → card {P | ∃ lines_intersecting_3, lines_intersecting_3.card = 3 
            ∧ ∀ l ∈ lines_intersecting_3, (lines l P.1, lines l P.2) = (0, 0)} ≥ 6,
  h₃: ∀ (P : ℝ × ℝ), 
        (∃ (lines_intersecting_2 : Finset (Fin 7)), 
          lines_intersecting_2.card = 2 
          ∧ (∀ l ∈ lines_intersecting_2, (lines l P.1, lines l P.2) = (0, 0)))
        → card {P | ∃ lines_intersecting_2, lines_intersecting_2.card = 2 
            ∧ ∀ l ∈ lines_intersecting_2, (lines l P.1, lines l P.2) = (0, 0)} ≥ 4
  ) : False :=
sorry

end no_seven_lines_intersection_l604_604984


namespace ellipses_have_two_distinct_intersection_points_chord_with_focus_min_area_exists_l604_604528

variable {a : ℝ} (h_a : 0 < a)

def ellipse_C1 (x y : ℝ) : Prop := 2 * x^2 - y^2 = 2 * a^2
def ellipse_C2 (x y : ℝ) : Prop := y^2 = -4 * √3 * a * x

theorem ellipses_have_two_distinct_intersection_points :
    ∀ x y : ℝ, ellipse_C1 a x y ∧ ellipse_C2 a x y → 
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ellipse_C1 a x1 y ∧ ellipse_C2 a x1 y ∧ ellipse_C1 a x2 y ∧ ellipse_C2 a x2 y) := 
begin
  sorry
end

def is_chord (x y b : ℝ) : Prop := y = b * (x + √3 * a)

def area_triangle (A B O : ℝ × ℝ) : ℝ :=
  let A_x := A.1 in
  let A_y := A.2 in
  let B_x := B.1 in
  let B_y := B.2 in
  (A_x * B_y - A_y * B_x) / 2

theorem chord_with_focus_min_area_exists :
    ∃ k : ℝ, ∀ B : ℝ × ℝ, is_chord a B.1 B.2 k → ∃ A : ℝ × ℝ, (area_triangle (A) (B) (0, 0)) = 6 * a^2 :=
begin
  sorry
end

end ellipses_have_two_distinct_intersection_points_chord_with_focus_min_area_exists_l604_604528


namespace polar_equation_of_line_through_point_perpendicular_to_polar_axis_l604_604706

theorem polar_equation_of_line_through_point_perpendicular_to_polar_axis :
  (line_passes_through_point_perpendicular_to_polar_axis (2 : ℝ) (Real.pi / 4 : ℝ)) →
  (∃ (ρ θ : ℝ), ρ * Real.cos θ = sqrt 2) :=
by
  intro h
  sorry

end polar_equation_of_line_through_point_perpendicular_to_polar_axis_l604_604706


namespace sum_of_odds_in_15th_set_l604_604013

/--
Given the sets of consecutive integers defined by the rules:
{1}, {2, 3}, {4, 5, 6}, {7, 8, 9, 10}, etc., where each set contains one more
element than the preceding one, and the first element of each set is one more
than the last element of the preceding set, prove that the sum of the odd
numbers in the 15th set is 791.
-/
theorem sum_of_odds_in_15th_set :
  let sets : List (List Nat) :=
    List.inits (List.range (1 + List.foldl (+) 0 (List.range 15 + 1)))
  let fifteenth_set := sets.get? 14
  let odd_numbers := List.filter Nat.odd fifteenth_set
  List.sum odd_numbers = 791 :=
sorry

end sum_of_odds_in_15th_set_l604_604013


namespace digit_2_appears_180_times_l604_604071

/-- The digit 2 appears exactly 180 times in the numbers from 1 to 400. -/
theorem digit_2_appears_180_times  :
  (count_digit_occurrences 2 (set.range (1 : ℕ)..(400 : ℕ))) = 180 := by
  sorry

end digit_2_appears_180_times_l604_604071


namespace angle_PQR_is_90_l604_604925

variable (R P Q S : Type) [EuclideanGeometry R P Q S]
variable (RSP_is_straight : straight_line R S P)
variable (angle_QSP : ∡Q S P = 70)

theorem angle_PQR_is_90 : ∡P Q R = 90 :=
by
  sorry

end angle_PQR_is_90_l604_604925


namespace carlos_gold_quarters_l604_604736

theorem carlos_gold_quarters:
  (let quarter_weight := 1 / 5 in
   let melt_value_per_ounce := 100 in
   let store_value_per_quarter := 0.25 in
   let quarters_per_ounce := 1 / quarter_weight in
   let total_melt_value := melt_value_per_ounce * quarters_per_ounce in
   let total_store_value := store_value_per_quarter * quarters_per_ounce in
   total_melt_value / total_store_value = 80) :=
by
  let quarter_weight := 1 / 5
  let melt_value_per_ounce := 100
  let store_value_per_quarter := 0.25
  let quarters_per_ounce := 1 / quarter_weight
  let total_melt_value := melt_value_per_ounce * quarters_per_ounce
  let total_store_value := store_value_per_quarter * quarters_per_ounce
  have : total_melt_value / total_store_value = 80 := sorry
  exact this

end carlos_gold_quarters_l604_604736


namespace sequence_problems_l604_604530

theorem sequence_problems
  (S: ℕ → ℚ) (b: ℕ → ℚ) (a: ℕ → ℚ) (c: ℕ → ℚ) (T: ℕ → ℚ)
  (h1: ∀ n, S n = ∑ k in Finset.range (n + 1), b k)
  (h2: ∀ n, b n = 2 - 2 * S n)
  (h3: a 5 = 10)
  (h4: a 7 = 14)
  (h5: ∀ n, c n = (1/4) * a n * b n)
  (h6: ∀ n, T n = ∑ k in Finset.range (n + 1), c k):
  (∀ n, a n = 2 * n)
  ∧ (∀ n, b n = 2 * (1/3)^n)
  ∧ (∀ n, T n = (3/4) - (2 * n + 3) / (4 * 3^n)) :=
by
  sorry

end sequence_problems_l604_604530


namespace points_satisfy_equation_l604_604751

theorem points_satisfy_equation :
  ∀ (x y : ℝ), x^2 - y^4 = Real.sqrt (18 * x - x^2 - 81) ↔ 
               (x = 9 ∧ y = Real.sqrt 3) ∨ (x = 9 ∧ y = -Real.sqrt 3) := 
by 
  intros x y 
  sorry

end points_satisfy_equation_l604_604751


namespace is_equilateral_l604_604090

open Complex

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry
noncomputable def z3 : ℂ := sorry

-- Assume the conditions of the problem
axiom z1_distinct_z2 : z1 ≠ z2
axiom z2_distinct_z3 : z2 ≠ z3
axiom z3_distinct_z1 : z3 ≠ z1
axiom z1_unit_circle : abs z1 = 1
axiom z2_unit_circle : abs z2 = 1
axiom z3_unit_circle : abs z3 = 1
axiom condition : (1 / (2 + abs (z1 + z2)) + 1 / (2 + abs (z2 + z3)) + 1 / (2 + abs (z3 + z1))) = 1
axiom acute_angled_triangle : sorry

theorem is_equilateral (A B C : ℂ) (hA : A = z1) (hB : B = z2) (hC : C = z3) : 
  (sorry : Prop) := sorry

end is_equilateral_l604_604090


namespace convex_polygons_from_fifteen_points_l604_604378

def number_of_distinct_convex_polygons (n : ℕ) : ℕ :=
  2^n - (∑ k in finset.range 4, (nat.choose n k))

theorem convex_polygons_from_fifteen_points :
  number_of_distinct_convex_polygons 15 = 32192 :=
by
  unfold number_of_distinct_convex_polygons
  norm_num
  sorry

end convex_polygons_from_fifteen_points_l604_604378


namespace age_difference_l604_604243

/-- 
The overall age of x and y is some years greater than the overall age of y and z. Z is 12 years younger than X.
Prove: The overall age of x and y is 12 years greater than the overall age of y and z.
-/
theorem age_difference {X Y Z : ℕ} (h1: X + Y > Y + Z) (h2: Z = X - 12) : 
  (X + Y) - (Y + Z) = 12 :=
by 
  -- proof goes here
  sorry

end age_difference_l604_604243


namespace quadratic_roots_computation_l604_604517

theorem quadratic_roots_computation :
  let r s : ℝ := sorry in
  (∃ r s : ℝ, (3 * r^2 - 5 * r - 7 = 0) ∧ (3 * s^2 - 5 * s - 7 = 0) ∧ r ≠ s) →
  (4 * r^2 - 4 * s^2) / (r - s) = (20 / 3) :=
by
  sorry  -- Proof goes here

end quadratic_roots_computation_l604_604517


namespace num_distinct_elements_sqrt_fraction_set_l604_604128

def floor_function (x : ℝ) : ℤ := Int.floor x

def sqrt_fraction_set (d : ℝ) : Finset ℤ := 
  (Finset.range 2005).image (λ n, floor_function ((n + 1) ^ 2 / d))

theorem num_distinct_elements_sqrt_fraction_set :
  (sqrt_fraction_set 2005).card = 1501 :=
  sorry

end num_distinct_elements_sqrt_fraction_set_l604_604128


namespace parallel_lines_condition_l604_604315

theorem parallel_lines_condition (a : ℝ) : 
  (a = 3 ↔ ∀ x y, (ax + 2y + 3a = 0) ∧ (3x + (a-1)y = a-7) → (ax + 2y + 3a = 0 ∥ 3x + (a-1)y = a-7)) :=
sorry

end parallel_lines_condition_l604_604315


namespace tan_cot_eq_solutions_count_l604_604388

theorem tan_cot_eq_solutions_count :
  (∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ tan (3 * π * cos θ) = cot (3 * π * sin θ)) :=
  sorry

end tan_cot_eq_solutions_count_l604_604388


namespace inscribed_cube_properties_l604_604712

variables (S : ℝ) (a b : ℝ)
def original_cube_surface_area : Prop := S = 54
def inner_cube_surface_area : Prop := a = 18
def inner_cube_volume : Prop := b = 3 * real.sqrt 3

theorem inscribed_cube_properties (h : original_cube_surface_area S) :
  inner_cube_surface_area a ∧ inner_cube_volume b :=
by
  sorry

end inscribed_cube_properties_l604_604712


namespace problem_angle_magnitude_and_sin_l604_604478

theorem problem_angle_magnitude_and_sin (
  a b c : ℝ) (A B C : ℝ) 
  (h1 : a = Real.sqrt 7) (h2 : b = 3) 
  (h3 : Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3)
  (triangle_is_acute : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) : 
  A = Real.pi / 3 ∧ Real.sin (2 * B + Real.pi / 6) = -1 / 7 :=
by
  sorry

end problem_angle_magnitude_and_sin_l604_604478


namespace ellipse_equation_ratio_BP_BQ_l604_604412

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : ∀ x y : ℝ, (x, y) ≠ (2, 0) ↔ x^2/a^2 + y^2/b^2 = 1) 
(h4 : ∀ c : ℝ, c = a * 1/2)
(h5 : a^2 = b^2 + c^2) : 
a = 2 ∧ b = sqrt 3 ∧ (∀ (x y : ℝ), x^2/4 + y^2/3 = 1) := by
  sorry

theorem ratio_BP_BQ (a b : ℝ) (x1 y1 x2 y2 x3 y3 : ℝ) 
(h1 : x1^2/4 + y1^2/3 = 1) (h2 : x2^2/4 + y2^2/3 = 1)
(h3 : (y1/x1) * (y2/x2) = -3/4) 
(h4 : x3 = (3 * x1)/5 + (4 * x2)/5) (h5 : y3 = (3 * y1)/5 + (4 * y2)/5)
(h6 : (a > b ∧ b > 0 ∧ a = 2 ∧ b = sqrt 3 ∧ ∀ (x y : ℝ), (x, y) ≠ (2, 0) ↔ x^2/4 + y^2/3 = 1)) : 
|x3 - x2 + y3 - y2| = 3 := by
  sorry

end ellipse_equation_ratio_BP_BQ_l604_604412


namespace same_color_probability_l604_604865

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l604_604865


namespace alice_shoe_probability_l604_604347

-- Definitions for the conditions
def total_shoes := 30
def black_shoes := 14
def brown_shoes := 8
def white_shoes := 6
def gray_shoes := 2
def pairs_shoes := 15

-- Function to calculate the probability of drawing two shoes of the same color with one being left and the other being right
noncomputable def probability_same_color_diff_foot := 
  (14 / 30 * 7 / 29) + (8 / 30 * 4 / 29) + (6 / 30 * 3 / 29) + (2 / 30 * 1 / 29)

-- Theorem statement
theorem alice_shoe_probability : probability_same_color_diff_foot = 25 / 145 :=
by
  -- explicit conversion avoids any ambiguity while dealing with fractions
  have black_prob : (14 / 30 * 7 / 29 : ℚ) = 98 / 870 := by sorry
  have brown_prob : (8 / 30 * 4 / 29 : ℚ) = 32 / 870 := by sorry
  have white_prob : (6 / 30 * 3 / 29 : ℚ) = 18 / 870 := by sorry
  have gray_prob : (2 / 30 * 1 / 29 : ℚ) = 2 / 870 := by sorry
  have add_frac : (98 / 870 + 32 / 870 + 18 / 870 + 2 / 870 : ℚ) = 150 / 870 := by sorry
  have simp_frac : (150 / 870 : ℚ) = 25 / 145 := by sorry
  rw [←black_prob, ←brown_prob, ←white_prob, ←gray_prob, ←add_frac, ←simp_frac]
  exact rfl

end alice_shoe_probability_l604_604347


namespace find_m_for_parallel_or_perpendicular_l604_604808

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_m_for_parallel_or_perpendicular (m : ℝ) :
  (let l1 := (slope (m, 1) (-3, 4)) in
   let l2 := (slope (1, m) (-1, m + 1)) in 
    ((l1 = l2 ∧ m = 3) ∨ (l1 * l2 = -1 ∧ m = -9 / 2))) :=
by
  sorry

end find_m_for_parallel_or_perpendicular_l604_604808


namespace unique_positive_x_eq_3_l604_604683

theorem unique_positive_x_eq_3 (x : ℝ) (h_pos : 0 < x) (h_eq : x + 17 = 60 * (1 / x)) : x = 3 :=
by
  sorry

end unique_positive_x_eq_3_l604_604683


namespace condition_of_A_with_respect_to_D_l604_604951

variables {A B C D : Prop}

theorem condition_of_A_with_respect_to_D (h1 : A → B) (h2 : ¬ (B → A)) (h3 : B ↔ C) (h4 : C → D) (h5 : ¬ (D → C)) :
  (D → A) ∧ ¬ (A → D) :=
by
  sorry

end condition_of_A_with_respect_to_D_l604_604951


namespace problem_l604_604143

def cs (s : ℕ) : ℕ := s * (s + 1)

def condition (k m n : ℕ) := nat.prime (m + k + 1) ∧ (m + k + 1) > (n + 1)

theorem problem (k m n : ℕ) (h : condition k m n) :
  (∏ i in finset.range n, cs (m + 1 + i) - cs k) %
  (∏ i in finset.range n, cs (i + 1)) = 0 :=
by sorry

end problem_l604_604143


namespace hexagon_angle_in_arithmetic_progression_l604_604996

theorem hexagon_angle_in_arithmetic_progression (a d : ℝ) (h : list ℝ) (ha : h = [a, a + d, a + 2 * d, a + 3 * d, a + 4 * d, a + 5 * d]) (h_sum : (6 : ℝ) * (a + 2.5 * d) = 720) : 
  114 ∈ h :=
by
  sorry

end hexagon_angle_in_arithmetic_progression_l604_604996


namespace thm1_thm2_thm3_thm4_l604_604963

variables {Point Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Definitions relating lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p q : Plane) : Prop := sorry
def perpendicular_planes (p q : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- Theorem 1: This statement is false, so we negate its for proof.
theorem thm1 (h1 : parallel_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  ¬ parallel_lines m n :=
sorry

-- Theorem 2: This statement is true, we need to prove it.
theorem thm2 (h1 : perpendicular_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 3: This statement is true, we need to prove it.
theorem thm3 (h1 : perpendicular_line_plane m α) (h2 : parallel_line_plane n β) (h3 : parallel_planes α β) :
  perpendicular_lines m n :=
sorry

-- Theorem 4: This statement is false, so we negate its for proof.
theorem thm4 (h1 : parallel_line_plane m α) (h2 : perpendicular_line_plane n β) (h3 : perpendicular_planes α β) :
  ¬ parallel_lines m n :=
sorry

end thm1_thm2_thm3_thm4_l604_604963


namespace complete_square_transformation_l604_604566

theorem complete_square_transformation : 
  ∀ (x : ℝ), (x^2 - 8 * x + 9 = 0) → ((x - 4)^2 = 7) :=
by
  intros x h
  sorry

end complete_square_transformation_l604_604566


namespace part_a_part_b_l604_604670

-- Part (a)
theorem part_a : ¬(∃ s : Finset ℝ, s.card = 7 ∧
  (∑ x in s.filter (λ x, x >= 10), x).card >= 4 ∧
  (∑ x in s, x) >= 43 ∧
  (∑ x in s, x) / 7 = 6) :=
by 
  sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n ≥ 4) :
  (∃ s : Finset ℝ, s.card = 2 * n + 1 ∧
    (s.filter (λ x, x >= 10)).card = n + 1 ∧
    (s.filter (λ x, x >= 1 ∧ x < 10)).card = n ∧
    (∑ x in s, x) = 12 * n + 6) :=
by 
  sorry

end part_a_part_b_l604_604670


namespace correct_propositions_l604_604717

theorem correct_propositions : [1, 2, 4] =
  let prop1 := (∀ a0 a1 a2 a3 a4 a5 : ℤ, (x - 2)^5 = a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0 → a1 + a2 + a3 + a4 + a5 = 31)
  let prop2 := (let X := normal 1 2 in P(X < 0) = P(X > 2))
  let prop3 := (∃ n : ℕ, (x + 2 / x^2)^n.coeff_sum = 243 → coef x⁻⁴ = 40)
  let prop4 := (let m n : ℕ := roll_die twice; θ := angle (m, n) (1, -1) in θ ∈ (0, π / 2] → P(θ ∈ (0, π / 2]) = 7 / 12)
  if prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 then [1, 2, 4] else []
:= sorry

end correct_propositions_l604_604717


namespace cube_root_of_sqrt_64_l604_604223

theorem cube_root_of_sqrt_64 : real.sqrt 64 ^ (1 / 3 : ℝ) = 2 := by
  sorry

end cube_root_of_sqrt_64_l604_604223


namespace intersection_of_sets_l604_604108

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l604_604108


namespace find_other_number_l604_604307

/-- Given HCF(A, B), LCM(A, B), and a known A, proves the value of B. -/
theorem find_other_number (A B : ℕ) 
  (hcf : Nat.gcd A B = 16) 
  (lcm : Nat.lcm A B = 396) 
  (a_val : A = 36) : B = 176 :=
by
  sorry

end find_other_number_l604_604307


namespace gains_calculation_l604_604506

theorem gains_calculation (x t : ℕ) : 
  let krishan_gain := 12 * x * t * 6000 / (24 * x * t) ∧ 
  let gopal_gain := 9 * x * t * 6000 / (24 * x * t) ∧ 
  let vishal_gain := 2 * x * t * 6000 / (24 * x * t) ∧ 
  let nandan_gain := 6000 in 
  krishan_gain = 72000 ∧ gopal_gain = 54000 ∧ vishal_gain = 12000 ∧ (krishan_gain + gopal_gain + vishal_gain + nandan_gain = 144000) := 
by 
  sorry

end gains_calculation_l604_604506


namespace probability_mult_of_5_l604_604209

theorem probability_mult_of_5 (s : Finset ℕ) (h_distinct: s.card = 6) (h_range: ∀ x ∈ s, 1 ≤ x ∧ x ≤ 2006) :
  ∃ a b ∈ s, a ≠ b ∧ (a - b) % 5 = 0 :=
by
  sorry

end probability_mult_of_5_l604_604209


namespace locus_of_midpoint_eq_circle_l604_604527

noncomputable def translated_circle_center := (6 : ℝ, 6 : ℝ)
noncomputable def homothety_center := (12 : ℝ, 10 : ℝ)
noncomputable def computed_center := (9 : ℝ, 8 : ℝ)
noncomputable def computed_radius := 3

theorem locus_of_midpoint_eq_circle :
  ∀ (O P : ℝ × ℝ) (r : ℝ),
  O = (3, 4) → P = (12, 10) → r = 6 →
  let O' := (O.1 + 3, O.2 + 2) in
  let midpoint := λ Q : ℝ × ℝ, ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) in
  ∀ Q : ℝ × ℝ,
  dist Q O' = r →
  dist (midpoint Q) computed_center = computed_radius
by
  intros O P r hO hP hr O' midpoint Q hdist
  have h1 : O' = (6, 6), by linarith,
  have hm : midpoint Q = ((12 + Q.1) / 2, (10 + Q.2) / 2), from rfl,
  have hQ : dist Q (6, 6) = 6, by rw [h1, hdist],
  -- The remaining proof steps are skipped
  sorry

end locus_of_midpoint_eq_circle_l604_604527


namespace same_color_probability_l604_604868

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l604_604868


namespace prob1_prob2_prob2_solution_l604_604072

-- Define the parametric equations for line l.
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 - (Real.sqrt 2) / 2 * t, 4 - (Real.sqrt 2) / 2 * t)

-- Define the rectangular coordinate equation of circle C.
def circle_C_rect := ∀ x y: ℝ, x^2 + y^2 = 4 * y ↔ x^2 + (y - 2)^2 = 4

-- Prove the given condition from a)
theorem prob1 : circle_C_rect :=
by
  intros x y
  split
  . intro h
    calc
      x^2 + (y - 2)^2 = x^2 + y^2 - 4*y + 4 : by ring
                    ... = 4*y - 4*y + 4   : by rw [h]
                    ... = 4               : by ring
  . intro h
    calc
      x^2 + y^2 = x^2 + (y - 2)^2 + 4*y - 4 : by ring
              ... = 4*y                    : by rw [h]; ring

-- Define the function t values at intersection points A and B, satisfying the det condition.
def t_values_ma_mb : ∀ a b : ℝ, a + b = 3 * (Real.sqrt 2) ∧ a * b = 1 → a > 0 ∧ b > 0 ∧ (1 - (Real.sqrt 2) / 2 * a = 1) ∧ (4 - (Real.sqrt 2) / 2 * a = 4) ∧ (1 - (Real.sqrt 2) / 2 * b = 1) ∧ (4 - (Real.sqrt 2) / 2 * b = 4)

-- Prove the second problem condition from a)
theorem prob2 : t_values_ma_mb 3^(1/2) 3^(1/2) := 
by
  sorry -- Proof to be filled in later

-- Prove the solution to the second part
theorem prob2_solution : |(Real.sqrt 2) + (Real.sqrt 2)| = 3 * (Real.sqrt 2) :=
by
  sorry -- Proof to be filled in later

end prob1_prob2_prob2_solution_l604_604072


namespace internal_nodes_dependent_on_sides_odd_gon_tessellation_impossible_l604_604358

-- Definitions
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

def sum_internal_angles_rhombuses (r : ℕ) : ℝ :=
  r * 360

def excess_internal_angles (r n : ℕ) : ℝ :=
  sum_internal_angles_rhombuses r - sum_interior_angles n

def nodes_count (r n : ℕ) : ℕ :=
  (excess_internal_angles r n / 360 : ℝ).toNat

-- Proof problem 1 statement
theorem internal_nodes_dependent_on_sides (r n : ℕ) (h : n % 2 = 0) :
  nodes_count r n = r + 1 - n / 2 :=
sorry

-- Proof problem 2 statement
theorem odd_gon_tessellation_impossible (k r : ℕ) :
  ¬ ∃ c, nodes_count r (2 * k + 1) = c :=
sorry

end internal_nodes_dependent_on_sides_odd_gon_tessellation_impossible_l604_604358


namespace convex_polygons_from_fifteen_points_l604_604379

def number_of_distinct_convex_polygons (n : ℕ) : ℕ :=
  2^n - (∑ k in finset.range 4, (nat.choose n k))

theorem convex_polygons_from_fifteen_points :
  number_of_distinct_convex_polygons 15 = 32192 :=
by
  unfold number_of_distinct_convex_polygons
  norm_num
  sorry

end convex_polygons_from_fifteen_points_l604_604379


namespace true_propositions_l604_604620

theorem true_propositions : 
  (∀ x y : ℝ, (xy = 1) ↔ (y * x = 1)) ∧
  (¬ (∀ triangles : Type,  equal_area triangles → congruent triangles)) ∧ 
  (∀ m : ℝ, (m > 1 → (x^2 - 2x + m = 0 → ∃ (x : ℝ), true))) :=
by sorry

end true_propositions_l604_604620


namespace general_term_maximum_sum_l604_604798

def S (n : ℕ) : ℕ := 32 * n - n^2 + 1

def a : ℕ → ℕ
| 1 := 32
| n := 33 - 2 * n

theorem general_term (n : ℕ) (h : n > 0) : 
  if n = 1 then a n = 32 else a n = 33 - 2 * n := 
by sorry

theorem maximum_sum : 
  ∃ n : ℕ, S n = 257 ∧ ∀ m : ℕ, S m ≤ S n := 
by sorry

end general_term_maximum_sum_l604_604798


namespace exists_sequence_l604_604171

theorem exists_sequence (n : ℕ) : ∃ (a : ℕ → ℕ), 
  (∀ i, 1 ≤ i → i < n → (a i > a (i + 1))) ∧
  (∀ i, 1 ≤ i → i < n → (a i ∣ a (i + 1)^2)) ∧
  (∀ i j, 1 ≤ i → 1 ≤ j → i < n → j < n → (i ≠ j → ¬(a i ∣ a j))) :=
sorry

end exists_sequence_l604_604171


namespace formula_for_roots_l604_604002

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (2^x - 1)
  else f (x - 1) + 1

-- Define the sequence a_n
def a (n : ℕ) : ℝ :=
  n - 1

-- The proof statement
theorem formula_for_roots :
  ∀ n, ∃ x, f x = x ∧ a n = x :=
sorry

end formula_for_roots_l604_604002


namespace probability_odd_product_l604_604398

-- Defining a finite type for the values on an eight-sided die
inductive Die
| one | two | three | four | five | six | seven | eight

open Die

-- Function to determine if a roll result is odd
def is_odd (d : Die) : Prop :=
  d = one ∨ d = three ∨ d = five ∨ d = seven

-- Defining the event of both dice rolling odd numbers
def both_odd (d1 d2 : Die) : Prop :=
  is_odd d1 ∧ is_odd d2

-- Probability calculation for the above event
theorem probability_odd_product :
  let total_outcomes := 8 * 8 in
  let odd_pairs := 4 * 4 in
  let probability := (odd_pairs / total_outcomes : ℚ) in
  probability = 1 / 4 :=
by
  sorry

end probability_odd_product_l604_604398


namespace angle_PQR_correct_l604_604918

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end angle_PQR_correct_l604_604918


namespace choose_roles_from_8_l604_604909

-- Define the number of people
def num_people : ℕ := 8
-- Define the function to count the number of ways to choose different persons for the roles
def choose_roles (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem choose_roles_from_8 : choose_roles num_people = 336 := by
  -- sorry acts as a placeholder for the proof
  sorry

end choose_roles_from_8_l604_604909


namespace black_region_area_is_correct_l604_604329

noncomputable def area_of_black_region : ℕ :=
  let area_large_square := 10 * 10
  let area_first_smaller_square := 4 * 4
  let area_second_smaller_square := 2 * 2
  area_large_square - (area_first_smaller_square + area_second_smaller_square)

theorem black_region_area_is_correct :
  area_of_black_region = 80 :=
by
  sorry

end black_region_area_is_correct_l604_604329


namespace proof_problem_l604_604361

-- Define the operation table as a function in Lean 4
def op (a b : ℕ) : ℕ :=
  if a = 1 then
    if b = 1 then 2 else if b = 2 then 1 else if b = 3 then 4 else 3
  else if a = 2 then
    if b = 1 then 1 else if b = 2 then 3 else if b = 3 then 2 else 4
  else if a = 3 then
    if b = 1 then 4 else if b = 2 then 2 else if b = 3 then 1 else 3
  else
    if b = 1 then 3 else if b = 2 then 4 else if b = 3 then 3 else 2

-- State the theorem to prove
theorem proof_problem : op (op 3 1) (op 4 2) = 2 :=
by
  sorry

end proof_problem_l604_604361


namespace find_a_plus_b_l604_604792

def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem find_a_plus_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) :
  a + b = 6 :=
sorry

end find_a_plus_b_l604_604792


namespace intersection_of_sets_l604_604112

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l604_604112


namespace part_a_equal_numbers_part_b_impossible_equal_two_hundred_l604_604434

-- Part (a): Prove that the given sequence can eventually have all numbers equal.
theorem part_a_equal_numbers :
  ∃ (k : ℕ), -- there exists a number k of operations
  let initial_list := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] in -- initial list of numbers
  let final_list := (List.replicate 10 (450 / 10)) in -- target list where all numbers are equal
  sorry

-- Part (b): Prove that it's impossible to make all numbers equal to 200.
theorem part_b_impossible_equal_two_hundred :
  ¬∃ (k : ℕ), 
  let initial_list := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] in
  let final_list := (List.replicate 10 200) in 
  sorry

end part_a_equal_numbers_part_b_impossible_equal_two_hundred_l604_604434


namespace prime_mod_4_has_solution_l604_604983

theorem prime_mod_4_has_solution (d : ℕ) (hd : Nat.Prime d) (hmod : d % 4 = 1) :
  ∃ x y : ℕ, x^2 - d * y^2 = -1 := sorry

end prime_mod_4_has_solution_l604_604983


namespace not_juggling_sequence_l604_604170

def j (n : Nat) : Nat :=
  match n % 3 with
  | 0 => 5
  | 1 => 7
  | _ => 2

def f (t : Nat) : Nat := t + j (t % 3)

theorem not_juggling_sequence : 
  ¬ (Bijective (λ t, f t)) :=
by
  sorry

end not_juggling_sequence_l604_604170


namespace domain_and_range_of_y_l604_604442

noncomputable def f (x : ℝ) := x + 2
noncomputable def y (x : ℝ) := (f x)^2 + f (x^2)

theorem domain_and_range_of_y :
  (∀ x, 1 ≤ x ∧ x ≤ 3) ∧ 
  (∀ y, y = 2 * x^2 + 4 * x + 6) ∧
  (∀ x ∈ set.Icc 1 3, 
    (y x ≥ 12) ∧ 
    (y x ≤ 36)) :=
sorry

end domain_and_range_of_y_l604_604442


namespace find_k_perpendicular_l604_604018

-- Definitions of the given vectors
def a := (1, 1, 0 : ℝ×ℝ×ℝ)
def b := (-1, 0, 2 : ℝ×ℝ×ℝ)

-- Defining the dot product for 3D vectors
def dot_product (v w : ℝ×ℝ×ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Statement of the theorem
theorem find_k_perpendicular (k : ℝ) : 
  (dot_product (k • (a : ℝ×ℝ×ℝ) + b) b = 0) → k = 4 := 
by sorry

end find_k_perpendicular_l604_604018


namespace determine_polynomial_l604_604369

noncomputable def p (x : ℝ) : ℝ := (2 / 3) * x^2 - (4 / 3) * x - 2

theorem determine_polynomial :
  (∀ x, (x = 3 ∨ x = -1) → p x = 0) ∧
  ∀ x, ¬(∃ L, ∀ ε > 0, ∃ N, ∀ n > N, abs ((x^4 + 3*x^3 - 4*x^2 - 12*x + 9) / (p x) - L) < ε) ∧
  p (-2) = 10 →
  p x = (2 / 3) * x^2 - (4 / 3) * x - 2 :=
sorry

end determine_polynomial_l604_604369


namespace find_x_when_y_is_20_l604_604810

-- Definition of the problem conditions.
def constant_ratio (x y : ℝ) : Prop := ∃ k, (3 * x - 4) = k * (y + 7)

-- Main theorem statement.
theorem find_x_when_y_is_20 :
  (constant_ratio x 5 → constant_ratio 3 5) → 
  (constant_ratio x 20 → x = 5.0833) :=
  by sorry

end find_x_when_y_is_20_l604_604810


namespace symmetry_about_x2_symmetry_about_2_0_l604_604644

-- Define the conditions and their respective conclusions.
theorem symmetry_about_x2 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) : 
  ∀ x, f (x) = f (4 - x) := 
sorry

theorem symmetry_about_2_0 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = -f (3 + x)) : 
  ∀ x, f (x) = -f (4 - x) := 
sorry

end symmetry_about_x2_symmetry_about_2_0_l604_604644


namespace probability_same_color_l604_604871

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l604_604871


namespace distinct_digit_addition_values_l604_604914

noncomputable def distinct_digit_addition : Prop :=
  ∃ (A B C D : ℕ), 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D ∧ -- distinctness condition
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ -- digits condition
  (A * 1000 + A * 100 + B * 10 + C) + (B * 1000 + C * 100 + A * 10 + D) = D * 1000 + B * 100 + C * 10 + D ∧
  (D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})

theorem distinct_digit_addition_values :
  distinct_digit_addition → 
  ∃ (n : ℕ), n = 9 :=
sorry

end distinct_digit_addition_values_l604_604914


namespace second_hand_degree_per_minute_l604_604681

theorem second_hand_degree_per_minute :
  (∀ (t : ℝ), t = 60 → 360 / t = 6) :=
by
  intro t
  intro ht
  rw [ht]
  norm_num

end second_hand_degree_per_minute_l604_604681


namespace distinct_real_numbers_eq_l604_604023

theorem distinct_real_numbers_eq (x : ℝ) :
  (x^2 - 7)^2 + 2 * x^2 = 33 → 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                    {a, b, c, d} = {x | (x^2 - 7)^2 + 2 * x^2 = 33}) :=
sorry

end distinct_real_numbers_eq_l604_604023


namespace angle_C_of_triangle_ABC_l604_604061

theorem angle_C_of_triangle_ABC
    (A B C A' B' H M : Type)
    [triangle ABC]
    [acute_triangle ABC]
    [non_isosceles ABC]
    (altitude_AA' : is_altitude A A' H)
    (altitude_BB' : is_altitude B B' H)
    (median_AHB_M : intersects_medians A H B M)
    (bisect_CM_A'B' : bisects_segment C M A' B') : 
    ∠C = 45 :=
sorry

end angle_C_of_triangle_ABC_l604_604061


namespace distinct_real_numbers_l604_604395

def larger (x y : ℝ) : ℝ := if x > y then x else y
def smaller (x y : ℝ) : ℝ := if x < y then x else y

theorem distinct_real_numbers (p q r s t : ℝ) (h : p < q < r < s < t) :
  larger (larger (smaller p q) r) (smaller s (larger t p)) = s :=
by {
  -- Given conditions
  have h₁ : p < q := h.1,
  have h₂ : q < r := h.2.1,
  have h₃ : r < s := h.2.2.1,
  have h₄ : s < t := h.2.2.2,
  have h₅ : p < t := lt_trans h₁ (lt_trans h₂ (lt_trans h₃ h₄)),
  
  -- Proof steps translated from solution, identifying intermediate values:
  have m_pq : smaller p q = p := if_pos h₁,
  have M_tp : larger t p = t := if_pos h₅,

  -- Substitute intermediate values:
  have M_m_pq_r : larger (smaller p q) r = larger p r := by rw m_pq,
  have m_s_M_tp : smaller s (larger t p) = smaller s t := by rw M_tp,

  -- Further simplify according to conditions:
  have M_p_r : larger p r = r := if_pos h₂,
  have m_s_t : smaller s t = s := if_pos h₄,

  -- Final substitution:
  rw [M_m_pq_r, m_s_M_tp, M_p_r, m_s_t],
  rw if_pos h₃,
}

end distinct_real_numbers_l604_604395


namespace probability_of_event_A_l604_604934

def weights : List ℕ := [90, 100, 110, 120, 140, 150, 150, 160]

def num_students := weights.length

def selection_size := 3

def seventieth_percentile_pos := num_students * 7 / 10

def seventieth_percentile := weights.nth_le 5 sorry

def event_A (selection : Finset ℕ) : Prop :=
  ∃ w, w ∈ selection ∧ w = seventieth_percentile ∧ ∀ w' ∈ selection, w' ≤ w

def count_combinations : ℕ → ℕ → ℕ := sorry -- Define combination logic later

def probability_event_A : ℚ :=
  (count_combinations 2 1 * count_combinations 5 2 + count_combinations 2 2 * count_combinations 5 1) /
  count_combinations 8 3

theorem probability_of_event_A :
  probability_event_A = 25 / 56 :=
sorry

end probability_of_event_A_l604_604934


namespace Tanya_efficiency_higher_l604_604552

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l604_604552


namespace buyer_total_price_680_l604_604975

noncomputable def new_price_jewelry (original_price_jewelry : ℕ) (increase_jewelry : ℕ) : ℕ :=
  original_price_jewelry + increase_jewelry

noncomputable def new_price_paintings (original_price_paintings : ℕ) (percentage_increase_paintings : ℚ) : ℕ :=
  original_price_paintings + (original_price_paintings * percentage_increase_paintings).toNat

noncomputable def total_price (price_jewelry : ℕ) (quantity_jewelry : ℕ) (price_paintings : ℕ) (quantity_paintings : ℕ) : ℕ :=
  (price_jewelry * quantity_jewelry) + (price_paintings * quantity_paintings)

theorem buyer_total_price_680 :
  let original_price_jewelry := 30
  let increase_jewelry := 10
  let original_price_paintings := 100
  let percentage_increase_paintings := (20 / 100 : ℚ)  -- 20%
  let quantity_jewelry := 2
  let quantity_paintings := 5
  let final_price_jewelry := new_price_jewelry original_price_jewelry increase_jewelry
  let final_price_paintings := new_price_paintings original_price_paintings percentage_increase_paintings
  in
  total_price final_price_jewelry quantity_jewelry final_price_paintings quantity_paintings = 680 :=
by
  sorry

end buyer_total_price_680_l604_604975


namespace motion_as_composition_of_reflections_l604_604551

-- Define the necessary components of the problem
noncomputable def isometry (F : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ × ℝ), dist (F x) (F y) = dist x y

-- Define points A, B, C in the plane
variables {A B C : ℝ × ℝ}

-- Non-collinearity condition
def non_collinear (A B C : ℝ × ℝ) : Prop :=
  ¬ ∃ (l : ℝ), B = A + l * (C - A)

-- The main theorem to be proved
theorem motion_as_composition_of_reflections
  (G : ℝ × ℝ → ℝ × ℝ)
  (is_iso_G : isometry G)
  (G_maps_A_B_C : ∀ pt ∈ {A, B, C}, G pt ≠ pt)
  (non_collinear_pts : non_collinear A B C)
  : ∃ (S1 S2 S3 : ℝ × ℝ → ℝ × ℝ),
      (∀ S, isometry S → ∃ l, is_reflection S l) ∧
      G = S1 ∘ S2 ∘ S3 :=
sorry

end motion_as_composition_of_reflections_l604_604551


namespace remainder_div_1234_567_89_1011_mod_12_l604_604282

theorem remainder_div_1234_567_89_1011_mod_12 :
  (1234^567 + 89^1011) % 12 = 9 := 
sorry

end remainder_div_1234_567_89_1011_mod_12_l604_604282


namespace max_NF_l604_604885

-- Define the circle O.
variable (O : Type)
-- Define the points A, B, and N such that A and B form the chord AB with length 4, and N is on segment AB.
variables (A B N : O)
-- Define an arbitrary point F.
variable (F : O)
-- Assume AB is a chord and not a diameter of the circle O.
-- Length of AB is 4.
axiom AB_chord (h : ∃ (r : ℝ), r > 0 ∧ ∀ (P : O), dist O A = r ∧ dist O B = r ∧ dist A B = 4)
-- N moves on the segment AB.
axiom N_on_AB (hN : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ N = (1 - t) • A + t • B)
-- ∠ONF = 90°.
axiom angle_ONF_90 (hAngle : ∡ O N F = 90)

theorem max_NF : 2 = Real.sqrt (4 ^ 2 - dist O N ^ 2) :=
sorry -- Proof goes here

end max_NF_l604_604885


namespace exist_common_divisor_l604_604142

theorem exist_common_divisor (a : ℕ → ℕ) (m : ℕ) (h_positive : ∀ i, 1 ≤ i ∧ i ≤ m → 0 < a i)
  (p : ℕ → ℤ) (h_poly : ∀ n : ℕ, ∃ i, 1 ≤ i ∧ i ≤ m ∧ (a i : ℤ) ∣ p n) :
  ∃ j, 1 ≤ j ∧ j ≤ m ∧ ∀ n, (a j : ℤ) ∣ p n :=
by
  sorry

end exist_common_divisor_l604_604142


namespace kylie_coins_count_l604_604946

theorem kylie_coins_count 
  (P : ℕ) 
  (from_brother : ℕ) 
  (from_father : ℕ) 
  (given_to_Laura : ℕ) 
  (coins_left : ℕ) 
  (h1 : from_brother = 13) 
  (h2 : from_father = 8) 
  (h3 : given_to_Laura = 21) 
  (h4 : coins_left = 15) : (P + from_brother + from_father) - given_to_Laura = coins_left → P = 15 :=
by
  sorry

end kylie_coins_count_l604_604946


namespace trader_profit_percentage_l604_604337

theorem trader_profit_percentage :
  (let cost1 := 80 * 15;
       cost2 := 20 * 20;
       total_cost := cost1 + cost2;
       total_weight := 80 + 20;
       cost_per_kg := total_cost / total_weight;
       sale_price_per_kg := 20;
       profit_per_kg := sale_price_per_kg - cost_per_kg;
       profit_percentage := (profit_per_kg / cost_per_kg) * 100 in
  profit_percentage = 25) :=
by
  sorry

end trader_profit_percentage_l604_604337


namespace problem1_problem2_problem3_l604_604791

open Complex

noncomputable def z (m : ℝ) : ℂ := (m^2 + 5 * m + 6 : ℂ) + (m^2 - 2 * m - 15 : ℂ) * Complex.I

theorem problem1 (m : ℝ) (h : z m = 2 - 12 * Complex.I) : m = -1 :=
by
  sorry

theorem problem2 (m : ℝ) (h : z m = conj (12 + 16 * Complex.I)) : m = 1 :=
by
  sorry

theorem problem3 (m : ℝ) (h : im (z m) > 0) : m < -3 ∨ m > 5 :=
by
  sorry

end problem1_problem2_problem3_l604_604791


namespace sum_of_first_1500_terms_l604_604248

def sequence : ℕ → ℕ
| 0 := 1
| (n + 1) := if (∃ k : ℕ, n = (k + 1) * k / 2) 
             then 1 
             else 3

noncomputable def sequence_sum (n : ℕ) : ℕ :=
∑ i in finset.range n, sequence i

theorem sum_of_first_1500_terms : sequence_sum 1500 = 4050 :=
sorry

end sum_of_first_1500_terms_l604_604248


namespace count_perfect_square_areas_l604_604396

def Q_n (n : ℕ) := ((n-1)^2, 0)
def P_n (n : ℕ) := ((n-1)^2, n * (n-1))
def Q_n_next (n : ℕ) := (n^2, 0)
def P_n_next (n : ℕ) := (n^2, n * (n + 1))

def A_n (n : ℕ) := n^2 * (2 * n - 1)

def is_perfect_square (x : ℕ) : Prop := ∃ (k : ℕ), k^2 = x

theorem count_perfect_square_areas : 
  {n : ℕ | 2 ≤ n ∧ n ≤ 99 ∧ is_perfect_square (A_n n)}.finite.to_finset.card = 6 :=
by
  sorry

end count_perfect_square_areas_l604_604396


namespace number_of_elements_in_set_P_l604_604562

theorem number_of_elements_in_set_P
  (p q : ℕ) -- we are dealing with non-negative integers here
  (h1 : p = 3 * q)
  (h2 : p + q = 4500)
  : p = 3375 :=
by
  sorry -- Proof goes here

end number_of_elements_in_set_P_l604_604562


namespace probability_same_color_l604_604872

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l604_604872


namespace construct_right_triangle_with_given_hypotenuse_l604_604363

variable (Real : Type) [linear_ordered_field Real]

structure Point (Real : Type) :=
(x : Real)
(y : Real)

noncomputable def hypotenuse {a b : Real} (A B : Point Real) : Real :=
(real_sqrt ((A.x - B.x)^2 + (A.y - B.y)^2))

noncomputable def altitude (A : Point Real) (B : Point Real) (C : Point Real) : Real :=
(abs (A.y - B.y))

noncomputable def angle_bisector_altitude (A : Point Real) (B : Point Real) (C : Point Real) : Prop :=
(altitude A B C = hypotenuse A B / 2)

theorem construct_right_triangle_with_given_hypotenuse
  (A B C : Point Real)
  (h : A.x ≠ B.x) -- A and B are distinct points
  (h1 : hypotenuse A B = hypotenuse A C)
  (h2 : C.y ≠ 0)
  : angle_bisector_altitude A B C :=
sorry

end construct_right_triangle_with_given_hypotenuse_l604_604363


namespace no_set_with_7_elements_min_elements_condition_l604_604659

noncomputable def set_a_elements := 7
noncomputable def median_a := 10
noncomputable def mean_a := 6
noncomputable def min_sum_a := 3 + 4 * 10
noncomputable def real_sum_a := mean_a * set_a_elements

theorem no_set_with_7_elements : ¬ (set_a_elements = 7 ∧
  (∃ S : Finset ℝ, 
    (S.card = set_a_elements) ∧ 
    (S.sum ≥ min_sum_a) ∧ 
    (S.sum = real_sum_a))) := 
by
  sorry

noncomputable def n_b_elements := ℕ
noncomputable def set_b_elements (n : ℕ) := 2 * n + 1
noncomputable def median_b := 10
noncomputable def mean_b := 6
noncomputable def min_sum_b (n : ℕ) := n + 10 * (n + 1)
noncomputable def real_sum_b (n : ℕ) := mean_b * set_b_elements n

theorem min_elements_condition (n : ℕ) : 
    (∀ n : ℕ, n ≥ 4) → 
    (set_b_elements n ≥ 9 ∧
        ∃ S : Finset ℝ, 
          (S.card = set_b_elements n) ∧ 
          (S.sum ≥ min_sum_b n) ∧ 
          (S.sum = real_sum_b n)) :=
by
  assume h : ∀ n : ℕ, n ≥ 4
  sorry

end no_set_with_7_elements_min_elements_condition_l604_604659


namespace Tanya_efficiency_higher_l604_604553

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l604_604553


namespace number_of_pairs_satisfying_equation_l604_604456

theorem number_of_pairs_satisfying_equation :
  {p : ℤ × ℤ | let (m, n) := p in m + n = m * n - 2}.card = 2 :=
by
  sorry

end number_of_pairs_satisfying_equation_l604_604456


namespace probability_same_color_plates_l604_604850

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l604_604850


namespace part_a_part_b_l604_604652

-- Definitions and Conditions
def median := 10
def mean := 6

-- Part (a): Prove that a set with 7 numbers cannot satisfy the given conditions
theorem part_a (n1 n2 n3 n4 n5 n6 n7 : ℕ) (h1 : median ≤ n1) (h2 : median ≤ n2) (h3 : median ≤ n3) (h4 : median ≤ n4)
  (h5 : 1 ≤ n5) (h6 : 1 ≤ n6) (h7 : 1 ≤ n7) (hmean : (n1 + n2 + n3 + n4 + n5 + n6 + n7) / 7 = mean) :
  false :=
by
  sorry

-- Part (b): Prove that the minimum size of the set where number of elements is 2n + 1 and n is a natural number, is at least 9
theorem part_b (n : ℕ) (h_sum_geq : ∀ (s : Finset ℕ), ((∀ x ∈ s, x >= median) ∧ ∃ t : Finset ℕ, t ⊆ s ∧ (∀ x ∈ t, x >= 1) ∧ s.card = 2 * n + 1) → s.sum >= 11 * n + 10) :
  n ≥ 4 :=
by
  sorry

-- Lean statements defined above match the problem conditions and required proofs

end part_a_part_b_l604_604652


namespace C_squared_ge_kn_div_k_plus_1_times_D_l604_604404

-- Define the problem and state the theorem

theorem C_squared_ge_kn_div_k_plus_1_times_D
  (n : ℕ) (hn : n > 1)
  (a : Fin n → ℝ)
  (k : ℕ) (hk : k > 1)
  (A : Fin k → Finset ℝ)
  (HA₁ : ∀ i, (A i).card ≥ n / 2)
  (HA₂ : ∀ i j, i ≠ j → (A i ∩ A j).card ≤ n / 4) :
  let B := (Finset.bUnion Finset.univ A)
      C := { z | ∃ (x y : ℝ), x ∈ B ∧ y ∈ B ∧ z = x + y }
      D := { z | ∃ (x y : ℝ), x ∈ B ∧ y ∈ B ∧ z = x - y }
  in Finset.card C ^ 2 ≥ (k * n / (k + 1)) * Finset.card D :=
by
  sorry

end C_squared_ge_kn_div_k_plus_1_times_D_l604_604404


namespace inscribed_circle_radius_integer_l604_604175

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l604_604175


namespace closest_grid_point_distance_l604_604436

noncomputable def line_equation : ℝ → ℝ :=
  λ x, (5 / 3) * x + (4 / 5)

def distance_to_line (x1 y1 : ℤ) : ℝ :=
  let A := 5
  let B := -3
  let C := 12
  abs (A * x1 + B * y1 + C) / real.sqrt (A * A + B * B)

theorem closest_grid_point_distance :
  ∃ (x1 y1 : ℤ), distance_to_line x1 y1 = sqrt 34 / 85 :=
by
  sorry

end closest_grid_point_distance_l604_604436


namespace min_circumference_plus_edges_l604_604405

noncomputable def min_sum_circumference_edge_count (G : SimpleGraph (fin 101)) [GraphConnected G] : ℕ :=
min (c G + E G).card

theorem min_circumference_plus_edges (G : SimpleGraph (fin 101)) [GraphConnected G]
  (h : ∀ e ∈ G.edgeSet, ∃ C, cycle C ∧ e ∈ C ∧ C.length ≤ c G) :
  min_sum_circumference_edge_count G = 121 := sorry

end min_circumference_plus_edges_l604_604405


namespace intersection_A_B_l604_604120

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l604_604120


namespace zoo_rabbits_count_l604_604602

theorem zoo_rabbits_count (parrots rabbits : ℕ) (h_ratio : parrots * 4 = rabbits * 3) (h_parrots_count : parrots = 21) : rabbits = 28 :=
by
  sorry

end zoo_rabbits_count_l604_604602


namespace not_odd_function_f_l604_604827

open Real

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 2)

theorem not_odd_function_f : ¬ (∀ x : ℝ, f (-x) = -f(x)) := by
  sorry

end not_odd_function_f_l604_604827


namespace borya_wins_optimal_play_l604_604349

theorem borya_wins_optimal_play :
  ∀ (points : Finset ℕ),
  points.card = 33 ∧ -- There are 33 points.
  ∀ (turns : ℕ → Finset ℕ), -- Turn tracker.
  (∀ n m : ℕ, n ≠ m → turns n ∩ turns m = ∅) ∧ -- Distinct turns.
  (∀ n : ℕ, turns n ⊆ points) ∧ -- Turns are subsets of points.
  (∀ n : ℕ, n % 2 = 1 → ∀ p : ℕ, p ∈ turns n → color p = blue ∨ color p = red) ∧ -- Anya goes on odd turns.
  (∀ n : ℕ, n % 2 = 0 → ∀ p : ℕ, p ∈ turns n → color p = blue ∨ color p = red) ∧ -- Borya goes on even turns.
  (∀ p : ℕ, p ∈ points → color p = blue ∨ color p = red) ∧ -- Points can be either blue or red.
  (∀ p q : ℕ, adjacent p q → color p ≠ color q) -- Adjacent points have different colors.
  → (∃ k : ℕ, k % 2 = 0 ∧ -- Borya's turn.
  ∀ t : Finset ℕ, turns k = t ∧ ∀ a : ℕ, a ∈ t → color a = color (a + 1)) -- Borya wins.
  sorry

end borya_wins_optimal_play_l604_604349


namespace histogram_area_represents_frequency_group_l604_604475

theorem histogram_area_represents_frequency_group:
  ∃ interpretation : String,
    interpretation ∈ {"A: The frequency of data falling into the corresponding groups",
                      "B: The frequency of the corresponding groups",
                      "C: The number of groups the sample is divided into",
                      "D: The sample size of the sample"} ∧
    interpretation = "B: The frequency of the corresponding groups" :=
by
  sorry

end histogram_area_represents_frequency_group_l604_604475


namespace quiz_contest_permutations_l604_604454

theorem quiz_contest_permutations : 
  ∃ n : ℕ, (Finset.univ.filter (λ p : Equiv.Perm (Fin 5), p.IsCycle)).card = 120 := 
sorry

end quiz_contest_permutations_l604_604454


namespace original_portion_al_l604_604714

variable (a b c : ℕ)

theorem original_portion_al :
  a + b + c = 1200 ∧
  a - 150 + 3 * b + 3 * c = 1800 ∧
  c = 2 * b →
  a = 825 :=
by
  sorry

end original_portion_al_l604_604714


namespace convex_polygons_15_points_l604_604376

theorem convex_polygons_15_points : 
  ∃ n : ℕ, n = 2^15 - (nat.choose 15 0) - (nat.choose 15 1) - (nat.choose 15 2) - (nat.choose 15 3) ∧ n = 32192 :=
begin
  use 32192,
  sorry
end

end convex_polygons_15_points_l604_604376


namespace ratio_of_segments_l604_604491

theorem ratio_of_segments 
  (A B C P Q R X Y Z : Point)
  (hP: P ∈ segment B C)
  (hQ: Q ∈ segment C A)
  (hR: R ∈ segment A B)
  (hAX : segment A P ∩ circumcircle (triangle A Q R) = {X})
  (hAY : segment A P ∩ circumcircle (triangle B R P) = {Y})
  (hAZ : segment A P ∩ circumcircle (triangle C P Q) = {Z}) 
  : ratio (segment Y X) (segment X Z) = ratio (segment B P) (segment P C) := 
sorry

end ratio_of_segments_l604_604491


namespace roots_of_P_l604_604763

-- Define the polynomial P(x) = x^3 + x^2 - 6x - 6
noncomputable def P (x : ℝ) : ℝ := x^3 + x^2 - 6 * x - 6

-- Define the statement that the roots of the polynomial P are -1, sqrt(6), and -sqrt(6)
theorem roots_of_P : ∀ x : ℝ, P x = 0 ↔ (x = -1) ∨ (x = sqrt 6) ∨ (x = -sqrt 6) :=
sorry

end roots_of_P_l604_604763


namespace problem1_problem2_problem3_l604_604441

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 3) / (3 * x)

-- Define the sequence a_n recursively
def a (n : ℕ) : ℝ :=
  if n = 0 then 1 else f (1 / a (n - 1))

-- Problem 1: Prove the general term of the sequence a_n
theorem problem1 (n : ℕ) : a n = (2 / 3) * n + (1 / 3) :=
sorry

-- Define T_n for Problem 2
def T (n : ℕ) : ℝ :=
  let a_sequence := list.map a (list.range (2 * n + 1))
  list.sum (list.zip_with (λ i j, i * j) a_sequence (list.tail a_sequence)) 
  -
  let a_sequence_shifted := list.map a (list.range 1 (2 * n + 2))
  list.sum (list.zip_with (λ i j, -i * j) a_sequence_shifted (list.tail a_sequence_shifted))

-- Problem 2: Prove the formula for T_n
theorem problem2 (n : ℕ) : T n = - (4 / 9) * (2 * n^2 + 3 * n) :=
sorry

-- Define b_n and S_n for Problem 3
def b (n : ℕ) : ℝ :=
  if n = 0 then 3 else 1 / (a (n - 1) * a n)

def S (n : ℕ) : ℝ :=
  list.sum (list.map b (list.range (n + 1)))

-- Problem 3: Prove the smallest m
theorem problem3 (m : ℕ) : (∀ n : ℕ, S n < (m - 2007) / 2) → m = 2016 :=
sorry

end problem1_problem2_problem3_l604_604441


namespace circle_cartesian_and_product_l604_604443

open Real

noncomputable def circle_eq (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

noncomputable def line_eq (t : ℝ) : (ℝ × ℝ) :=
  (1 / 2 + sqrt 3 / 2 * t, 1 / 2 + 1 / 2 * t)

noncomputable def point_A : ℝ × ℝ := (sqrt 2 / 2, π / 4)

theorem circle_cartesian_and_product (t : ℝ) :
  (∀ (ρ θ : ℝ), circle_eq ρ θ → (ρ^2 = (x - 1)^2 + y^2) ∧ |line_eq(t1) - point_A| * |line_eq(t2) - point_A| = 1 / 2) :=
begin
  sorry
end

end circle_cartesian_and_product_l604_604443


namespace two_star_nine_value_l604_604780

theorem two_star_nine_value
  (a b : ℝ)
  (h1 : ∀ (x y : ℝ), x * y = a * x^y + b + 1)
  (h2 : 1 * 2 = 969) 
  (h3 : 2 * 3 = 983) :
  2 * 9 = 1991 :=
begin
  sorry
end

end two_star_nine_value_l604_604780


namespace boys_without_calculators_l604_604048

theorem boys_without_calculators
  (total_boys : ℕ)
  (students_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (students_forgot_calculators : ℕ)
  (boys_with_calculators : ℕ)
  (total_boys == 20)
  (students_with_calculators == 26)
  (girls_with_calculators == 15)
  (students_forgot_calculators == 3) :
  total_boys - boys_with_calculators = 8 :=
by {
  have h1 : boys_with_calculators = students_with_calculators - girls_with_calculators + 1 + students_forgot_calculators,
  { rw [nat.sub_add, nat.add_assoc, nat.add_one]},
  rw [ ← nat.add_sub_assoc, h1, nat.su, eq_comm],
  repeat { rw nat.sub_add, apply eq_nat.sub_of_sub_eq) },
  sorry
}

end boys_without_calculators_l604_604048


namespace bigger_part_of_sum_and_linear_combination_l604_604689

theorem bigger_part_of_sum_and_linear_combination (x y : ℕ) 
  (h1 : x + y = 24) 
  (h2 : 7 * x + 5 * y = 146) : x = 13 :=
by 
  sorry

end bigger_part_of_sum_and_linear_combination_l604_604689


namespace min_diesel_consumption_l604_604237

theorem min_diesel_consumption :
  (∀ (generator_consumption_per_hour: ℝ) (start_extra_consumption : ℝ) (total_hours : ℝ)
      (max_downtime_minutes : ℝ) (min_runtime_minutes : ℝ),
    generator_consumption_per_hour = 6 →
    start_extra_consumption = 0.5 →
    total_hours = 10 →
    max_downtime_minutes = 10 →
    min_runtime_minutes = 15 →
    let total_minutes : ℝ := total_hours * 60 in
    let cycle_duration : ℝ := min_runtime_minutes + max_downtime_minutes in
    let cycles : ℝ := total_minutes / cycle_duration in
    let initial_wait : ℝ := max_downtime_minutes in
    let cycle_consumption : ℝ := (min_runtime_minutes / 60 * generator_consumption_per_hour)
                                 + start_extra_consumption in
    let last_run_fuel : ℝ := ((cycle_duration - max_downtime_minutes) / 60 * generator_consumption_per_hour) in
    let total_fuel : ℝ := initial_wait / 60 * generator_consumption_per_hour
                          + (cycles - 1) * cycle_consumption
                          + last_run_fuel - start_extra_consumption in
      total_fuel = 47.5
  ) :=
begin
  intros,
  sorry
end

end min_diesel_consumption_l604_604237


namespace sum_sequence_2014_terms_l604_604447

noncomputable def sequence (n : ℕ) : ℝ :=
2014 * Real.sin (n * Real.pi / 2)

theorem sum_sequence_2014_terms :
  (Finset.sum (Finset.range 2014) sequence) = 2014 := by
  sorry

end sum_sequence_2014_terms_l604_604447


namespace cannot_travel_from_3rd_to_12th_l604_604052

noncomputable def is_reachable (start target : ℕ) : Prop :=
  ∃ (n : ℤ), target = start + n * 7 - m * 9 ∧ 1 ≤ start + n * 7 - m * 9 ≤ 15

theorem cannot_travel_from_3rd_to_12th :
  ¬ is_reachable 3 12 :=
sorry

end cannot_travel_from_3rd_to_12th_l604_604052


namespace square_area_sum_l604_604336

theorem square_area_sum :
  (∀ (A : ℝ),
    (Σ A, ∃ (x y₁ y₂ y₃ : ℝ) 
      (hA : A = (sqrt (2^2 + y₁^2)) ^ 2) 
      (h2 : A = (sqrt (18^2 + y₃^2)) ^ 2)
      (h3 : y₁^2 - y₃^2 = 320) 
      (h4 : x ∈ {2, 0, 18}),
       0 + 2 + 18 = 20)) =
  1168 :=
sorry

end square_area_sum_l604_604336


namespace hyperbola_eccentricity_l604_604891

theorem hyperbola_eccentricity (a b c : ℝ) (h_asymptotes : ∀ x : ℝ, y = a*x ∨ y = -a*x) 
  (h1 : (a = sqrt 3) → (c^2 - a^2 = 3*a^2) ∧ (b = sqrt 3*a))
  (h2 : (b = sqrt 3) → (a^2 / (c^2 - a^2) = 3) ∧ (a = sqrt 3*b)) :
  (eccentricity c a = 2 ∨ eccentricity c a = (2*sqrt 3)/3) :=
by
  sorry

end hyperbola_eccentricity_l604_604891


namespace relative_order_l604_604138

theorem relative_order (x a b : ℝ) (h : x < a) (h1 : a < b) (h2 : b < 0) : 
  x^2 > ax ∧ ax > ab ∧ ab > a^2 :=
by
  -- Given conditions
  have hx_neg : x < 0 := by linarith
  have ha_neg : a < 0 := by linarith
  have hb_neg : b < 0 := by linarith

  have hx2_pos : x^2 > 0 := by apply sq_pos_of_ne_zero; exact ne_of_lt hx_neg
  have ha2_pos : a^2 > 0 := by apply sq_pos_of_ne_zero; exact ne_of_lt ha_neg
  have hb2_pos : b^2 > 0 := by apply sq_pos_of_ne_zero; exact ne_of_lt hb_neg

  -- Comparison of squares
  have h_sq : x^2 > a^2 ∧ a^2 > b^2 := by
    split
    exact (sq_lt_sq hx_neg ha_neg).mpr h,
    exact (sq_lt_sq ha_neg hb_neg).mpr h1

  -- Analyzing products
  have h_ax_pos : ax > 0 := mul_pos_of_neg_of_neg ha_neg hx_neg
  have h_ab_pos : ab > 0 := mul_pos_of_neg_of_neg ha_neg hb_neg

  have h_ax_gt_a2 : ax > a^2 := by
    have : a * x < a * a := by exact mul_lt_mul_of_neg_left h ha_neg
    convert this using 1
    rw mul_comm a x
    rw sq

  have h_ab_gt_ax : ab > ax := by exact mul_lt_mul_of_neg_left h1 ha_neg

  split; [exact h_sq.1 | split; [exact h_ab_gt_ax | exact h]].save

  sorry

end relative_order_l604_604138


namespace root_of_unity_l604_604302

noncomputable def f (n : ℕ) (a : ℕ → ℝ) (x : ℂ) : ℂ := 
    ∑ i in finset.range (n+1), (a i) * x^(n-i)

theorem root_of_unity (n : ℕ) (a : ℕ → ℝ) (λ : ℂ) 
  (h1 : 1 ≥ a 1 ∧ a 1 ≥ a 2 ∧ ... ∧ a (n-1) ≥ a n ∧ a n ≥ 0)
  (h2 : f n a λ = 0)
  (h3 : |λ| ≥ 1): 
  ∃ s : ℕ, s ≥ 1 ∧ λ^s = 1 :=
sorry

end root_of_unity_l604_604302


namespace value_of_f_at_pi_over_3_l604_604893

def f (ω φ : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (ω * x + φ)

theorem value_of_f_at_pi_over_3 (ω φ : ℝ) (hf_symm : ∀ x : ℝ, f ω φ (π / 3 + x) = f ω φ (π / 3 - x)) :
  f ω φ (π / 3) = 3 ∨ f ω φ (π / 3) = -3 :=
sorry

end value_of_f_at_pi_over_3_l604_604893


namespace measure_of_angle_A_theorem_range_of_area_theorem_l604_604905

noncomputable def measure_of_angle_A (a b c : ℝ) (B C : ℝ) (non_isosceles: a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (condition: (2 * c - b) * Real.cos C = (2 * b - c) * Real.cos B) : ℝ :=
60

noncomputable def range_of_area (b c : ℝ) (area: ℝ) (A: ℝ := 60) (a: ℝ := 4) 
    (condition: b^2 + c^2 - b*c > 0) : Set ℝ :=
{ x | 0 < x ∧ x < 4 * Real.sqrt 3 }

theorem measure_of_angle_A_theorem (a b c : ℝ) (B C : ℝ) (non_isosceles : a ≠ b ∧ a ≠ c ∧ b ≠ c)
    (condition : (2 * c - b) * Real.cos C = (2 * b - c) * Real.cos B) :
    measure_of_angle_A a b c B C non_isosceles condition = 60 := 
sorry

theorem range_of_area_theorem (b c : ℝ) (A : ℝ := 60) (a : ℝ := 4)
    (condition: b^2 + c^2 - b*c > 0) :
    range_of_area b c (area := λ bc, bc * Real.sqrt 3 / 4) A a = { x | 0 < x ∧ x < 4 * Real.sqrt 3 } :=
sorry

end measure_of_angle_A_theorem_range_of_area_theorem_l604_604905


namespace angle_between_vectors_l604_604416

variables {V : Type*} [inner_product_space ℝ V]

-- Define non-zero vectors a and b
variables (a b : V) (h_a : a ≠ 0) (h_b : b ≠ 0)

-- Define the given conditions
def condition1 : Prop := ∥a + b∥ = ∥a∥
def condition2 : Prop := ∥b∥ = real.sqrt 3 * ∥a∥

-- State the problem: the angle between a and b
theorem angle_between_vectors :
  condition1 a b h_a h_b →
  condition2 a b h_a h_b →
  ∃ θ : ℝ, 
    0 ≤ θ ∧ θ ≤ 180 ∧ (θ = 150) :=
begin
  sorry
end

end angle_between_vectors_l604_604416


namespace problem_b2056_l604_604521

noncomputable def seq_b : ℕ → ℝ
| 1 := 2 + Real.sqrt 11
| n := if n = 2023 then 17 + Real.sqrt 11 else if h : 2 ≤ n then 
          seq_b (n - 1) * seq_b (n + 1) else
          0 -- handle cases not of interest

theorem problem_b2056 :
  seq_b 2056 = -19 / 9 + 15 / 9 * Real.sqrt 11 := sorry

end problem_b2056_l604_604521


namespace product_of_roots_l604_604389

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : ℝ := 9 * x^2 + 27 * x - 225

-- Prove that the product of the roots of the given quadratic equation is -25
theorem product_of_roots : ∀ a b c : ℝ, 
    (∀ x : ℝ, a * x^2 + b * x + c = quadratic_eq x) → 
    a = 9 → b = 27 → c = -225 →
    -25 = c / a :=
by
  intro a b c h_eq ha hb hc
  rw [ha, hc]
  norm_num
  sorry

end product_of_roots_l604_604389


namespace geom_seq_308th_term_l604_604058

noncomputable def geometric_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ (n - 1)

theorem geom_seq_308th_term :
  geometric_sequence 12 (-2) 308 = -2 ^ 307 * 12 := by
  sorry

end geom_seq_308th_term_l604_604058


namespace mike_total_spent_l604_604539

noncomputable def total_spent_by_mike (food_cost wallet_cost shirt_cost shoes_cost belt_cost 
  discounted_shirt_cost discounted_shoes_cost discounted_belt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + discounted_shirt_cost + discounted_shoes_cost + discounted_belt_cost

theorem mike_total_spent :
  let food_cost := 30
  let wallet_cost := food_cost + 60
  let shirt_cost := wallet_cost / 3
  let shoes_cost := 2 * wallet_cost
  let belt_cost := shoes_cost - 45
  let discounted_shirt_cost := shirt_cost - (0.2 * shirt_cost)
  let discounted_shoes_cost := shoes_cost - (0.15 * shoes_cost)
  let discounted_belt_cost := belt_cost - (0.1 * belt_cost)
  total_spent_by_mike food_cost wallet_cost shirt_cost shoes_cost belt_cost
    discounted_shirt_cost discounted_shoes_cost discounted_belt_cost = 418.50 := by
  sorry

end mike_total_spent_l604_604539


namespace highest_to_lowest_order_l604_604586

theorem highest_to_lowest_order :
  ∀ (a b c : ℝ), a = -10 ∧ b = 1 ∧ c = -7 → (b > c ∧ c > a) :=
by
  intros a b c h
  cases h with ha hb
  cases hb with hb hc
  rw [ha, hb, hc]
  sorry

end highest_to_lowest_order_l604_604586


namespace john_pays_percentage_of_srp_l604_604251

theorem john_pays_percentage_of_srp (P MP : ℝ) (h1 : P = 1.20 * MP) (h2 : MP > 0): 
  (0.60 * MP / P) * 100 = 50 :=
by
  sorry

end john_pays_percentage_of_srp_l604_604251


namespace smallest_y_angle_l604_604747

open Real

-- We define the equation and the given condition.
def tan_eq (y : ℝ) : Prop :=
  tan (6 * y) = (cos y - sin y) / (cos y + sin y)

-- Let the angle given be in degrees, converting to radians where necessary.
noncomputable def smallest_positive_angle (y : ℝ) : Prop := 
  y = (45 / 7) * (π / 180)

theorem smallest_y_angle : ∃ y : ℝ, tan_eq y ∧ smallest_positive_angle y :=
by
  use (45 / 7) * (π / 180)
  split
  -- skip the proofs
  sorry

end smallest_y_angle_l604_604747


namespace inequality_solution_l604_604782

theorem inequality_solution (x : ℝ) : 
  x^3 - 10 * x^2 + 28 * x > 0 ↔ (0 < x ∧ x < 4) ∨ (6 < x)
:= sorry

end inequality_solution_l604_604782


namespace min_value_expression_l604_604772

theorem min_value_expression (x : ℝ) : 
  (\frac{x^2 + 9}{real_sqrt (x^2 + 5)} ≥ \frac{9 * real_sqrt 5}{5}) :=
sorry

end min_value_expression_l604_604772


namespace percentage_increase_in_efficiency_l604_604558

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l604_604558


namespace product_of_all_positive_odd_integers_less_than_20000_l604_604281

   def product_of_odd_integers (n : Nat) : Nat :=
     (List.range (n)).filter Nat.odd).prod

   theorem product_of_all_positive_odd_integers_less_than_20000 :
     product_of_odd_integers 20000 = 20000! / (2^10000 * 10000!) :=
   by
     sorry
   
end product_of_all_positive_odd_integers_less_than_20000_l604_604281


namespace three_digit_solutions_count_l604_604457

theorem three_digit_solutions_count :
  {x : ℕ // 100 ≤ x ∧ x ≤ 999 ∧ (4851 * x + 597) % 29 = 1503 % 29}.card = 32 :=
by
  sorry

end three_digit_solutions_count_l604_604457


namespace trig_identity_proof_l604_604687

theorem trig_identity_proof :
  sin 20 * sin 10 - cos 10 * sin 70 = - (sqrt 3) / 2 :=
by
  sorry

end trig_identity_proof_l604_604687


namespace line_through_C_chord_length_sqrt3_range_of_r_value_of_m_times_n_l604_604799

-- Definition of the circle O
def circleO (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Definition of intersection points A and B
def pointA : ℝ × ℝ := (-1, 0)
def pointB : ℝ × ℝ := (0, 1)

-- Statement 1: Equation of line l
theorem line_through_C_chord_length_sqrt3 :
  (λ l : ℝ → ℝ → Prop, l (1/2) (sqrt 3 / 2) ∧ 
                   ∃ p1 p2, circleO (fst p1) (snd p1) ∧ circleO (fst p2) (snd p2) ∧ 
                   dist p1 p2 = sqrt 3) :=
by sorry

-- Statement 2: Range of values for r
theorem range_of_r (r : ℝ) :
  (∀ P : ℝ × ℝ, (P.1 - 0)^2 + (P.2 - 1)^2 = r^2 → dist P pointA = sqrt 2 * dist P (0,0)) →
  0 < r ∧ r ≤ 2*sqrt 2 :=
by sorry

-- Statement 3: Value of m ⋅ n
theorem value_of_m_times_n (x1 y1 x2 y2 : ℝ) :
  circleO x1 y1 ∧ circleO x2 y2 →
  let M1 := (-x1, -y1), let M2 := (x1, -y1) in
  let m := (x1 * y2 - x2 * y1) / (x1 + x2) in
  let n := (x1 * y2 + x2 * y1) / (x1 - x2) in
  m * n = 1 :=
by sorry

end line_through_C_chord_length_sqrt3_range_of_r_value_of_m_times_n_l604_604799


namespace set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604666

-- Problem 1: Prove that a set of 7 numbers cannot have an arithmetic mean of 6 under given conditions.
theorem set_of_7_numbers_not_possible {s : Finset ℝ} (h_card : s.card = 7) (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / 7 = 6) : False := sorry

-- Problem 2: Prove that the number of elements in the set must be at least 9 if the arithmetic mean is 6.
theorem minimum_elements_with_mean_6 {s : Finset ℝ} (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / s.card = 6) (h_card : s.card = 2 * (s.card / 2) + 1) : 8 < s.card := sorry

end set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604666


namespace find_distance_between_A_and_B_l604_604783

noncomputable def distance_between_A_and_B : ℝ :=
  unknown_dist

theorem find_distance_between_A_and_B :
  ∀ (v_A v_B : ℝ)
      (A_B_dist : ℝ)
      (start_time_A start_time_B : ℝ)
      (mid_point_distance : ℝ)
      (meet_distance_24 : ℝ)
      (meet_distance_20 : ℝ)
      (meet_distance_midpoint : ℝ),
    start_time_A = 6.8333 ∧ -- 6:50 AM in hours
    start_time_B = 6.8333 ∧ -- 6:50 AM in hours
    meet_distance_24 = 24 ∧ 
    meet_distance_20 = 20 ∧ 
    meet_distance_midpoint = A_B_dist / 2 ∧
    (7 ≤ start_time_A + (24 / v_A) ∧ 8 > start_time_A + (24 / v_A) → -- during the peak hour if the meeting time is between 7:00 and 8:00
       ∃ v : ℝ, v = v_A / 2 ∧ meet_distance_24 = distance_between_A_and_B) ∧
    (7 ≤ start_time_B + (20 / v_B) ∧ 8 > start_time_B + (20 / v_B) →
      ∃ v : ℝ, v = v_B / 2 ∧ meet_distance_20 = distance_between_A_and_B - 4 ∨ meet_distance_20 = 20) →
    A_B_dist = 42 :=
sorry

end find_distance_between_A_and_B_l604_604783


namespace person_A_pass_test_l604_604427

open Nat

/-- 
Person A can answer 5 out of 10 questions correctly. 
In each test, 3 questions are randomly selected. 
To pass the test, at least 2 questions must be answered correctly.
-/
def probability_of_passing_test : ℚ :=
  let total_outcomes := choose 10 3
  let win_scenario_1 := (choose 5 2) * (choose 5 1)
  let win_scenario_2 := choose 5 3
  (win_scenario_1 + win_scenario_2) / total_outcomes

theorem person_A_pass_test : 
  probability_of_passing_test = 1 / 2 :=
by
  unfold probability_of_passing_test
  sorry

end person_A_pass_test_l604_604427


namespace line_through_trisection_point_l604_604151

noncomputable def trisection_points : (ℝ × ℝ) × (ℝ × ℝ) :=
let point1 := (-4, 5) in
let point2 := (5, -1) in
let trisection1 := ((-4 + (1/3) * (5 + 4)), (5 + (1/3) * (-1 - 5))) in
let trisection2 := ((-4 + (2/3) * (5 + 4)), (5 + (2/3) * (-1 - 5))) in
(trisection1, trisection2)

theorem line_through_trisection_point : 
  let lineE := (λ (x y : ℝ), x - 4 * y + 13 = 0) in
  let trisection1 := (trisection_points.1) in
  let trisection2 := (trisection_points.2) in
  (lineE 3 4 ∧ (lineE trisection1.1 trisection1.2 ∨ lineE trisection2.1 trisection2.2)) :=
by
  sorry

end line_through_trisection_point_l604_604151


namespace part_a_part_b_l604_604668

-- Part (a)
theorem part_a : ¬(∃ s : Finset ℝ, s.card = 7 ∧
  (∑ x in s.filter (λ x, x >= 10), x).card >= 4 ∧
  (∑ x in s, x) >= 43 ∧
  (∑ x in s, x) / 7 = 6) :=
by 
  sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n ≥ 4) :
  (∃ s : Finset ℝ, s.card = 2 * n + 1 ∧
    (s.filter (λ x, x >= 10)).card = n + 1 ∧
    (s.filter (λ x, x >= 1 ∧ x < 10)).card = n ∧
    (∑ x in s, x) = 12 * n + 6) :=
by 
  sorry

end part_a_part_b_l604_604668


namespace area_cross_section_XYZ_l604_604902

open EuclideanGeometry

noncomputable def cube_area_cross_section (a : ℝ) (h: a > 0) : ℝ :=
  let X := (a/2, a/2, 0)
  let Y := (a, a/2, a/2)
  let Z := (a/4, 3*a/4, 0)
  let XY := dist_3d X Y
  (a * XY) / 2

theorem area_cross_section_XYZ (a : ℝ) (h: a > 0) :
  cube_area_cross_section a h = (a^2 * sqrt 2) / 2 := by
  sorry

end area_cross_section_XYZ_l604_604902


namespace necessarily_positive_y_plus_xsq_l604_604202

theorem necessarily_positive_y_plus_xsq {x y z : ℝ} 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  y + x^2 > 0 :=
sorry

end necessarily_positive_y_plus_xsq_l604_604202


namespace true_proposition_l604_604444

def p : Prop := ∀ x : ℝ, x + (1 / x) ≥ 2

def q : Prop := ∃ x : set.Icc 0 (Real.pi / 2), Real.sin x + Real.cos x = Real.sqrt 2

theorem true_proposition : ¬p ∧ q :=
by
  -- p is false because there exists x = -1 where x + 1/x = -2 which is less than 2.
  have hnp : ¬p := by
    intro hp
    have h : ∀ x : ℝ, x + 1 / x ≥ 2 := hp
    have counterexample := h (-1)
    linarith
  -- q is true because for x = π/4, sin(x) + cos(x) = √2
  have hq : q := by
    use ((Real.pi / 4) : set.Icc 0 (Real.pi / 2))
    simp [Real.sin_pi_div_four, Real.cos_pi_div_four]
  exact ⟨hnp, hq⟩

#check true_proposition

end true_proposition_l604_604444


namespace intersection_distance_eq_one_l604_604969

noncomputable def point_on_line (t : ℝ) : ℝ × ℝ :=
  (1 + 1/2 * t, (sqrt 3) / 2 * t)

noncomputable def point_on_circle (θ : ℝ) : ℝ × ℝ :=
  (cos θ, sin θ)

theorem intersection_distance_eq_one :
  ∃ (A B : ℝ × ℝ), 
  (∃ t₁ t₂ θ₁ θ₂, A = point_on_line t₁ ∧ B = point_on_line t₂ ∧ A = point_on_circle θ₁ ∧ B = point_on_circle θ₂)
  ∧ dist A B = 1 :=
sorry

end intersection_distance_eq_one_l604_604969


namespace range_of_x_l604_604898

theorem range_of_x (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 := 
  sorry

end range_of_x_l604_604898


namespace star_value_l604_604366

def star (a b c : ℕ) : ℚ :=
  (a * b + c : ℚ) / (a + b + c)

theorem star_value :
  star 4 8 2 = 17 / 7 :=
by {
  have h1 : (4 * 8 + 2 : ℚ) = 34 := by norm_num,
  have h2 : (4 + 8 + 2 : ℚ) = 14 := by norm_num,
  rw [star, h1, h2],
  norm_num,
  sorry
}

end star_value_l604_604366


namespace intersection_A_B_eq_C_l604_604099

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l604_604099


namespace lcm_n_n_plus_3_l604_604770

open Nat

theorem lcm_n_n_plus_3 (n : ℕ) : lcm n (n + 3) = if n % 3 = 0 then (1 / 3) * n * (n + 3) else n * (n + 3) := by
  sorry

end lcm_n_n_plus_3_l604_604770


namespace radius_of_inscribed_circle_is_integer_l604_604184

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l604_604184


namespace gold_quarter_value_comparison_l604_604729

theorem gold_quarter_value_comparison:
  (worth_in_store per_quarter: ℕ → ℝ) 
  (weight_per_quarter in_ounce: ℝ) 
  (earning_per_ounce melted: ℝ) : 
  (worth_in_store 4  = 0.25) →
  (weight_per_quarter = 1/5) →
  (earning_per_ounce = 100) →
  (earning_per_ounce * weight_per_quarter / worth_in_store 4 = 80) :=
by
  -- The proof goes here
  sorry

end gold_quarter_value_comparison_l604_604729


namespace find_t_l604_604879

noncomputable def f (x t : ℝ) : ℝ := |x - t| + |5 - x|

theorem find_t (t : ℝ) (h : ∃ x : ℝ, f x t = 3) : t = 2 ∨ t = 8 :=
by
  have h1 : ∀ x : ℝ, f x t ≥ |5 - t| := by
    intro x
    calc
      |x - t| + |5 - x| ≥ |(x - t) + (5 - x)| : abs_add_abs_ge_abs_add (x - t) (5 - x)
      ... = |5 - t| : by rw [add_sub_cancel']
  have h2 : ¬ ∃ x : ℝ, |5 - t| < 3 := by
    intro ⟨x, hx⟩
    exact not_lt_of_le (h1 x) (hx ▸ by linarith)
  cases h with x hx
  simp only [f, h1 x, le_antisymm_iff] at hx
  linarith
  sorry -- additional proof steps to argue t = 2 ∨ t = 8

end find_t_l604_604879


namespace ratio_AC_BC_l604_604550

-- Define the conditions as assumptions
variable (r : ℝ) (A B C : Point) (circle_radius : ℝ)
variable (H1 : distance A B = distance A C)
variable (H2 : distance A B > r)
variable (arc_length_BC : ℝ)
variable (H3 : arc_length_BC = 2 * r)

-- Define the main statement
theorem ratio_AC_BC : 
  ∀ (r : ℝ) (distance : point → point → ℝ) (arc_length : point → point → ℝ → ℝ),
  distance A B = distance A C →
  distance A B > r →
  arc_length B C r = 2 * r →
  ∃ (AC BC : ℝ), 
    AC = distance A C ∧ BC = arc_length B C r ∧ AC / BC = (1 / 2) * sqrt(2 * (1 - cos(2))) :=
by
  sorry

end ratio_AC_BC_l604_604550


namespace volume_of_locus_l604_604088

-- Definitions for conditions
def isConvexPolygon (m : Type) : Prop := sorry  -- Define convex polygon
def perimeter (m : Type) : ℝ := sorry  -- Perimeter of m
def area (m : Type) : ℝ := sorry  -- Area of m
def M (m : Type) (R : ℝ) : Type := sorry  -- Locus of all points in space whose distance to m is ≤ R
def volume {T : Type} (s : T) : ℝ := sorry  -- Volume of the solid s

-- Theorem statement
theorem volume_of_locus (m : Type) (R : ℝ) (l S : ℝ) 
  (h1 : isConvexPolygon m)
  (h2 : l = perimeter m)
  (h3 : S = area m) :
  volume (M m R) = (4 / 3) * Real.pi * R^3 + (Real.pi / 2) * l * R^2 + 2 * S * R := 
sorry

end volume_of_locus_l604_604088


namespace jason_ad_fraction_l604_604537

noncomputable def page_area (width height : ℕ) : ℕ :=
  width * height

noncomputable def ad_area (total_cost cost_per_sq_inch : ℕ) : ℕ :=
  total_cost / cost_per_sq_inch

noncomputable def fraction_of_page (ad_area total_page_area : ℕ) : ℚ :=
  ad_area / total_page_area

theorem jason_ad_fraction
  (cost_per_sq_inch total_cost : ℕ)
  (width height : ℕ)
  (total_page_area := page_area width height)
  (ad_area := ad_area total_cost cost_per_sq_inch)
  (fraction := fraction_of_page ad_area total_page_area) :
  (cost_per_sq_inch = 8)
  → (total_cost = 432)
  → (width = 9)
  → (height = 12)
  → fraction = 1 / 2 :=
by {
  intros,
  sorry,
}

end jason_ad_fraction_l604_604537


namespace inscribed_circle_radius_integer_l604_604192

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l604_604192


namespace intersection_A_B_l604_604118

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l604_604118


namespace intersection_of_sets_l604_604115

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l604_604115


namespace tony_drinks_3_ounces_every_4_trips_l604_604627

-- Constants for the problem
constant bucket_capacity : ℕ := 2 -- pounds of sand
constant sandbox_depth : ℕ := 2 -- feet
constant sandbox_width : ℕ := 4 -- feet
constant sandbox_length : ℕ := 5 -- feet
constant sand_weight : ℕ := 3 -- pounds per cubic foot
constant trips_per_drink : ℕ := 4
constant bottle_ounces : ℕ := 15
constant bottle_cost : ℕ := 2 -- dollars per bottle
constant initial_money : ℕ := 10 -- dollars
constant change : ℕ := 4 -- dollars

-- Derived conditions
constant total_volume : ℕ := sandbox_depth * sandbox_width * sandbox_length -- cubic feet
constant total_sand : ℕ := total_volume * sand_weight -- pounds
constant total_trips : ℕ := total_sand / bucket_capacity
constant total_drinks : ℕ := total_trips / trips_per_drink
constant total_spent : ℕ := initial_money - change
constant total_bottles : ℕ := total_spent / bottle_cost
constant ounces_per_drink : ℕ := bottle_ounces * total_bottles / total_drinks

theorem tony_drinks_3_ounces_every_4_trips : ounces_per_drink = 3 := by
  sorry

end tony_drinks_3_ounces_every_4_trips_l604_604627


namespace basis_eq_of_given_conditions_l604_604030

open Vector

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

-- Assuming that {a, b, c} form a basis of the space V
axiom is_basis : ∀ (u v w : V), LinearIndependent ℝ ![u, v, w] ∧ submodule.span ℝ ![u, v, w] = ⊤

theorem basis_eq_of_given_conditions :
  LinearIndependent ℝ ![c, a + b, a - b] ∧ submodule.span ℝ ![c, a + b, a - b] = ⊤ :=
sorry

end basis_eq_of_given_conditions_l604_604030


namespace midpoint_cd_MP_parallel_AB_CD_l604_604070

theorem midpoint_cd_MP_parallel_AB_CD
  {A B C D P M : Type*}
  [InscribedQuad ABCD A B C D]
  (hP : intersection_point_of_diagonals P)
  (hM : midpoint_of_arc M A B)
  : (line MP passes_through midpoint CD) ↔ (lines_parallel AB CD) := 
sorry

end midpoint_cd_MP_parallel_AB_CD_l604_604070


namespace convert_34_to_binary_l604_604752

def decimal_to_binary (n : ℕ) : ℕ :=
Nat.binary_rec_on n 0 (fun b _ bs, shiftl bs 1 + b)

theorem convert_34_to_binary : decimal_to_binary 34 = 100010 := by
  sorry

end convert_34_to_binary_l604_604752


namespace bus_accommodates_children_l604_604888

theorem bus_accommodates_children : 
  ∀ (rows_per_bus : ℕ) (children_per_row : ℕ), rows_per_bus = 9 → children_per_row = 4 → rows_per_bus * children_per_row = 36 :=
by
  intros rows_per_bus children_per_row
  intro h1
  intro h2
  rw [h1, h2]
  sorry

end bus_accommodates_children_l604_604888


namespace part_a_part_b_l604_604667

-- Part (a)
theorem part_a : ¬(∃ s : Finset ℝ, s.card = 7 ∧
  (∑ x in s.filter (λ x, x >= 10), x).card >= 4 ∧
  (∑ x in s, x) >= 43 ∧
  (∑ x in s, x) / 7 = 6) :=
by 
  sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n ≥ 4) :
  (∃ s : Finset ℝ, s.card = 2 * n + 1 ∧
    (s.filter (λ x, x >= 10)).card = n + 1 ∧
    (s.filter (λ x, x >= 1 ∧ x < 10)).card = n ∧
    (∑ x in s, x) = 12 * n + 6) :=
by 
  sorry

end part_a_part_b_l604_604667


namespace juice_per_cup_percentage_l604_604777

variable (C : ℝ) -- The total capacity of the pitcher in ounces

-- Conditions
def pitcher_filled : ℝ := 5 / 8 * C
def num_cups : ℝ := 4

-- Goal
theorem juice_per_cup_percentage (h : C > 0) : 
  ((5 / 8 * C) / num_cups) / C * 100 = 15.625 := 
by
  sorry

end juice_per_cup_percentage_l604_604777


namespace monotonic_increase_l604_604316

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2)

theorem monotonic_increase : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 < f x2 :=
by
  sorry

end monotonic_increase_l604_604316


namespace problem_statement_l604_604411

noncomputable
def ellipse_equation (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1

def midpoint (x1 y1 x2 y2 mx my : ℝ) : Prop :=
  mx = (x1 + x2)/2 ∧ my = (y1 + y2)/2

theorem problem_statement :
  ∃ (a b : ℝ), a^2 = 24 ∧ b^2 = 8 ∧ (a > b ∧ b > 0) ∧
  (ellipse_equation a b) ∧
  (∃ (x1 y1 x2 y2 : ℝ), midpoint x1 y1 x2 y2 3 1) ∧
  (∃ (k : ℝ), ∀ (x : ℝ), y = k * (x - 3) + 1) ∧
  (∃ (d : ℝ), d = 3 * sqrt 2 / sqrt 3 ∧ |AB| = 2*sqrt 6)
:= sorry

end problem_statement_l604_604411


namespace distance_from_circle_center_to_point_l604_604636

theorem distance_from_circle_center_to_point :
  let circle_eq := λ x y : ℝ => x^2 + y^2 = 6 * x - 8 * y + 18
  let point := (3, -2) : ℝ × ℝ
  (∃ center : ℝ × ℝ, (∀ x y : ℝ, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = 43)
  ∧ (dist center point = 2)) := sorry

end distance_from_circle_center_to_point_l604_604636


namespace wise_men_correct_guesses_l604_604622

noncomputable def max_correct_guesses (n k : ℕ) : ℕ :=
  if n > k + 1 then n - k - 1 else 0

theorem wise_men_correct_guesses (n k : ℕ) :
  ∃ (m : ℕ), m = max_correct_guesses n k ∧ m ≤ n - k - 1 :=
by {
  sorry
}

end wise_men_correct_guesses_l604_604622


namespace star_square_ratio_l604_604496

/-
Given:
1. A square with side length 5 cm.
2. Four identical isosceles triangles inside the square, each with a base of 5 cm and height of 1 cm.

Prove:
The ratio between the area of the star (formed by subtracting the total area of the four triangles from the area of the square) and the area of the square is \( \frac{3}{5} \).
-/

theorem star_square_ratio :
  let side := 5 in
  let height := 1 in
  let area_square := side * side in
  let area_triangle := 1/2 * side * height in
  let total_triangle_area := 4 * area_triangle in
  let area_star := area_square - total_triangle_area in
  (area_star : ℚ) / area_square = 3/5 :=
by
  -- The proof will be conducted here.
  sorry

end star_square_ratio_l604_604496


namespace incorrect_options_l604_604014

def sgn (x : ℝ) : ℤ :=
  if x > 0 then 1
  else if x = 0 then 0
  else -1

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 1 then x
  else if x < 0 then f (-x)
  else f (x - 2)

theorem incorrect_options :
  ∃ (A B D : Prop),
  (A = (∀ (x : ℝ), ¬ (sgn (f x) > 0))) ∧
  (B = (f (2023 / 2) ≠ 1)) ∧
  (D = (∀ (k : ℤ), sgn (f k) ≠ abs (sgn k))) ∧
  (A ∧ B ∧ D) :=
by
  sorry

end incorrect_options_l604_604014


namespace problem1_problem2_l604_604693

noncomputable section

-- Define the setup: A bag with red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def draw_count : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem 1: Drawing 4 balls such that they must be of two colors
theorem problem1 : 
  combination red_balls 3 * combination white_balls 1 
  + combination red_balls 2 * combination white_balls 2
  + combination red_balls 1 * combination white_balls 3 = 194 := 
by
  sorry

-- Problem 2: Drawing 4 balls such that the number of red balls drawn is not less than the number of white balls
theorem problem2 :
  combination red_balls 4 
  + combination red_balls 3 * combination white_balls 1
  + combination red_balls 2 * combination white_balls 2 = 115 := 
by
  sorry

end problem1_problem2_l604_604693


namespace lines_pass_through_single_point_l604_604549

theorem lines_pass_through_single_point 
  (S : Type) [metric_space S] [normed_group S] [normed_space ℝ S] 
  (circle : S) 
  (A B M P Q : S) 
  (hAB : A ≠ B) 
  (diameter : set S) 
  (hdiameter : diameter = {x | dist x A = dist x B}) 
  (hM_on_diameter : M ∈ diameter) 
  (hM_not_center : M ≠ (0.5 • (A + B))) 
  (hP_on_circle : P ∈ circle) 
  (hQ_on_circle : Q ∈ circle) 
  (hPM_angle : angle A M P = angle A M Q) :
  ∃ O : S, ∀ P Q, line_through P Q = line_through O :=
sorry

end lines_pass_through_single_point_l604_604549


namespace range_of_m_l604_604834

open Real

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ((log 3 m)^2 - log 3 (27 * m^2)) * x^2 - (log 3 m - 3) * x - 1 < 0) ↔ (3^(-1/3) < m ∧ m ≤ 27) :=
by 
  -- Definitions required for the theorem
  let a := (log 3 m)^2 - log 3 (27 * m^2)
  let b := log 3 m - 3
  let c := -1
  sorry

end range_of_m_l604_604834


namespace series1_converges_series2_diverges_series3_converges_l604_604498

noncomputable def series1 (n : ℕ) : ℝ := (2 ^ n) / (n!)

noncomputable def series2 (n : ℕ) : ℝ := (3 ^ n) / (n * (2 ^ n))

noncomputable def series3 (n : ℕ) : ℝ := 1 / (n ^ 3)

theorem series1_converges : ∃ L, is_limit (λ S : finset ℕ, ∑ k in S, series1 k) at_top L :=
sorry

theorem series2_diverges : ¬ ∃ L, is_limit (λ S : finset ℕ, ∑ k in S, series2 k) at_top L :=
sorry

theorem series3_converges : ∃ L, is_limit (λ S : finset ℕ, ∑ k in S, series3 k) at_top L :=
sorry

end series1_converges_series2_diverges_series3_converges_l604_604498


namespace minhyuk_needs_slices_l604_604649

-- Definitions of Yeongchan and Minhyuk's apple division
def yeongchan_portion : ℚ := 1 / 3
def minhyuk_slices : ℚ := 1 / 12

-- Statement to prove
theorem minhyuk_needs_slices (x : ℕ) : yeongchan_portion = x * minhyuk_slices → x = 4 :=
by
  sorry

end minhyuk_needs_slices_l604_604649


namespace same_color_probability_l604_604869

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l604_604869


namespace factorization_example_l604_604646

def transformationA (x : ℝ) : Prop := x^3 - x = x * (x + 1) * (x - 1)
def factorization (p : Prop) : Prop := p  -- Placeholder for factorization definition

theorem factorization_example (x : ℝ) : factorization (transformationA x) := 
  sorry

end factorization_example_l604_604646


namespace locus_of_intersection_l604_604403

theorem locus_of_intersection (m : ℝ) :
  (∃ x y : ℝ, (m * x - y + 1 = 0) ∧ (x - m * y - 1 = 0)) ↔ (∃ x y : ℝ, (x - y = 0) ∨ (x - y + 1 = 0)) :=
by
  sorry

end locus_of_intersection_l604_604403


namespace g_nine_l604_604585

variable (g : ℝ → ℝ)

theorem g_nine : (∀ x y : ℝ, g (x + y) = g x * g y) → g 3 = 4 → g 9 = 64 :=
by intros h1 h2; sorry

end g_nine_l604_604585


namespace proposition_correctness_l604_604000

open Classical

variable {p q : Prop} {a b x : ℝ}

theorem proposition_correctness :
  (¬(p ∧ q) → ¬p ∨ ¬q) →
  (¬ (a > b → 2^a > 2^b - 1) ↔ a ≤ b ∧ 2^a ≤ 2^b - 1) →
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ ∃ x : ℝ, x^2 + 1 < 1) →
  (∀ A B : ℝ, ∀ ABC : Triangle, (A > B ↔ sin A > sin B)) →
  2 :=
by
  intros H1 H2 H3 H4
  sorry

end proposition_correctness_l604_604000


namespace employees_participating_in_game_l604_604723

theorem employees_participating_in_game 
  (managers players : ℕ)
  (teams people_per_team : ℕ)
  (h_teams : teams = 3)
  (h_people_per_team : people_per_team = 2)
  (h_managers : managers = 3)
  (h_total_players : players = teams * people_per_team) :
  players - managers = 3 :=
sorry

end employees_participating_in_game_l604_604723


namespace chicken_nuggets_cost_l604_604534

theorem chicken_nuggets_cost :
  ∀ (nuggets_ordered boxes_cost : ℕ) (nuggets_per_box : ℕ),
  nuggets_ordered = 100 →
  nuggets_per_box = 20 →
  boxes_cost = 4 →
  (nuggets_ordered / nuggets_per_box) * boxes_cost = 20 :=
by
  intros nuggets_ordered boxes_cost nuggets_per_box h1 h2 h3
  sorry

end chicken_nuggets_cost_l604_604534


namespace max_value_expression_l604_604890

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (m : ℝ), (∀ x y : ℝ, x * y > 0 → 
  m ≥ (x / (x + y) + 2 * y / (x + 2 * y))) ∧ 
  m = 4 - 2 * Real.sqrt 2 := 
sorry

end max_value_expression_l604_604890


namespace find_t_squared_l604_604704

theorem find_t_squared (t : ℝ) (b : ℝ) (h1 : b^2 = 36) 
  (h2 : ∃ t, let hyp_eq := λ x y, x^2 / 9 - y^2 / b^2 = 1 
    in hyp_eq (-3) 4 ∧ hyp_eq (-3) 0 ∧ hyp_eq t 3) : t^2 = 45 / 4 := 
sorry

end find_t_squared_l604_604704


namespace special_sale_reduction_percentage_l604_604245

open_locale classical

noncomputable def price_after_first_reduction (P : ℝ) := 0.75 * P
noncomputable def price_after_special_sale (P : ℝ) (x : ℝ) := (1 - x) * price_after_first_reduction P
noncomputable def restored_price (P : ℝ) (x : ℝ) := price_after_special_sale P x * 1.5686274509803921

theorem special_sale_reduction_percentage (P : ℝ) (x : ℝ) (h : restored_price P x = P) : x = 0.15 :=
by sorry

end special_sale_reduction_percentage_l604_604245


namespace area_sub_triangles_leq_sixth_l604_604966

variable {A B C O M N P : Type}
variable [HasArea A B C O M N P : ℝ]

-- Define the area of triangle ABC
noncomputable def triangleArea (A B C : Type) : ℝ := sorry -- S

-- Define sub-triangles areas
noncomputable def area_BOM (B O M : Type) : ℝ := sorry
noncomputable def area_MOC (M O C : Type) : ℝ := sorry
noncomputable def area_NOC (N O C : Type) : ℝ := sorry
noncomputable def area_AON (A O N : Type) : ℝ := sorry
noncomputable def area_AOP (A O P : Type) : ℝ := sorry
noncomputable def area_BOP (B O P : Type) : ℝ := sorry

theorem area_sub_triangles_leq_sixth {S : ℝ} (A B C O M N P : Type)
  (hOinABC : O ∈ triangleArea A B C) :
  let S := triangleArea A B C in
  let A1 := area_BOM B O M,
      A2 := area_MOC M O C,
      A3 := area_NOC N O C,
      A4 := area_AON A O N,
      A5 := area_AOP A O P,
      A6 := area_BOP B O P in
  (min {A1, A3, A5} ≤ S / 6 ∧ min {A2, A4, A6} ≤ S / 6) :=
begin
  sorry -- The proof goes here
end

end area_sub_triangles_leq_sixth_l604_604966


namespace probability_greater_area_l604_604707

-- Definitions of the elements involved
def isosceles_triangle (A B C : Type) [ordered_field A] (AB AC : A) (BC : A) : Prop :=
  AB = AC

def point_in_triangle (P A B C : Type) [ordered_field P] : Prop :=
  P ∈ interior (triangle A B C)

-- The math proof problem statement
theorem probability_greater_area (A B C P : Type) [ordered_field A] [ordered_field P]
    (h_isosceles : isosceles_triangle A B C (dist A B) (dist A C))
    (h_point : point_in_triangle P A B C) :
  probability (area (triangle A B P) > area (triangle A C P) ∧ area (triangle A B P) > area (triangle B C P)) = 1 / 3 :=
sorry

end probability_greater_area_l604_604707


namespace no_set_with_7_elements_min_elements_condition_l604_604660

noncomputable def set_a_elements := 7
noncomputable def median_a := 10
noncomputable def mean_a := 6
noncomputable def min_sum_a := 3 + 4 * 10
noncomputable def real_sum_a := mean_a * set_a_elements

theorem no_set_with_7_elements : ¬ (set_a_elements = 7 ∧
  (∃ S : Finset ℝ, 
    (S.card = set_a_elements) ∧ 
    (S.sum ≥ min_sum_a) ∧ 
    (S.sum = real_sum_a))) := 
by
  sorry

noncomputable def n_b_elements := ℕ
noncomputable def set_b_elements (n : ℕ) := 2 * n + 1
noncomputable def median_b := 10
noncomputable def mean_b := 6
noncomputable def min_sum_b (n : ℕ) := n + 10 * (n + 1)
noncomputable def real_sum_b (n : ℕ) := mean_b * set_b_elements n

theorem min_elements_condition (n : ℕ) : 
    (∀ n : ℕ, n ≥ 4) → 
    (set_b_elements n ≥ 9 ∧
        ∃ S : Finset ℝ, 
          (S.card = set_b_elements n) ∧ 
          (S.sum ≥ min_sum_b n) ∧ 
          (S.sum = real_sum_b n)) :=
by
  assume h : ∀ n : ℕ, n ≥ 4
  sorry

end no_set_with_7_elements_min_elements_condition_l604_604660


namespace servant_cash_received_l604_604021

theorem servant_cash_received (annual_cash : ℕ) (turban_price : ℕ) (served_months : ℕ) (total_months : ℕ) (cash_received : ℕ) :
  annual_cash = 90 → turban_price = 50 → served_months = 9 → total_months = 12 → 
  cash_received = (annual_cash + turban_price) * served_months / total_months - turban_price → 
  cash_received = 55 :=
by {
  intros;
  sorry
}

end servant_cash_received_l604_604021


namespace contradiction_for_n3_min_elements_when_n_ge_4_l604_604672

theorem contradiction_for_n3 :
  ∀ (s : Set ℕ), (s.card = 7) → 
                 (∀ (x ∈ s), x ≥ 1) → 
                 (∃ t u : Set ℕ, (t.card = 4) ∧ (u.card = 3) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) 
                 → ∃ x ∈ s, false :=
sorry

theorem min_elements_when_n_ge_4 (n : ℕ) (hn : n ≥ 4) :
  ∃ (s : Set ℕ), (s.card = 2 * n + 1) ∧ 
                 (∀ (x ∈ s), x ≥ 1) ∧ 
                 (∃ t u : Set ℕ, (t.card = n + 1) ∧ (u.card = n) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) ∧
                 ∀ (s : Set ℕ), s.card = 2 * n + 1 → (∑ x in s, x) / (2 * n + 1) = 6 :=
sorry

example : ∃ s, (s.card = 9) ∧ (∀ x ∈ s, x ≥ 1) ∧ 
               (∃ t u : Set ℕ, (t.card = 5) ∧ (u.card = 4) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
               (∀ x ∈ t, x ≥ 10) ∧ (∀ x ∈ u, x ≥ 1) ∧
               (∑ x in s, x) / 9 = 6) :=
{ sorry }

end contradiction_for_n3_min_elements_when_n_ge_4_l604_604672


namespace segment_length_eq_4_l604_604818

-- Given conditions
def parametric_eq (theta : ℝ) : ℝ × ℝ :=
  (1 + sqrt 3 * cos theta, sqrt 3 * sin theta)

def curve_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

def polar_eq (rho theta : ℝ) : Prop :=
  rho^2 - 2 * rho * cos theta - 2 = 0

def line_eq (rho theta : ℝ) : Prop :=
  rho * cos (theta - π / 6) = 3 * sqrt 3

def ray (theta : ℝ) : Prop :=
  theta = π / 3

-- The proof statement
theorem segment_length_eq_4 :
  ∀ θ ρ1 ρ2 : ℝ,
  (curve_eq (1 + sqrt 3 * cos θ) (sqrt 3 * sin θ)) →
  (ray θ) → 
  (polar_eq ρ1 θ ∧ line_eq ρ2 θ) →
  ρ2 - ρ1 = 4 :=
by
  intros θ ρ1 ρ2 h_curve h_ray h_polar_line
  sorry

end segment_length_eq_4_l604_604818


namespace problem1_problem2_l604_604448

variable (a : ℝ)

def quadratic_roots (a x : ℝ) : Prop := a*x^2 + 2*x + 1 = 0

-- Problem 1: If 1/2 is a root, find the set A
theorem problem1 (h : quadratic_roots a (1/2)) : 
  {x : ℝ | quadratic_roots (a) x } = { -1/4, 1/2 } :=
sorry

-- Problem 2: If A contains exactly one element, find the set B consisting of such a
theorem problem2 (h : ∃! (x : ℝ), quadratic_roots a x ) : 
  {a : ℝ | ∃! (x : ℝ), quadratic_roots a x} = { 0, 1 } :=
sorry

end problem1_problem2_l604_604448


namespace infinite_sequence_exists_l604_604200

noncomputable def has_k_distinct_positive_divisors (n k : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card ≥ k ∧ ∀ d ∈ S, d ∣ n

theorem infinite_sequence_exists :
    ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, 0 < k → ∃ n : ℕ, (a n > 0) ∧ has_k_distinct_positive_divisors (a n ^ 2 + a n + 2023) k) :=
  sorry

end infinite_sequence_exists_l604_604200


namespace octopus_legs_l604_604256

/-- Four octopuses made statements about their total number of legs.
    - Octopuses with 7 legs always lie.
    - Octopuses with 6 or 8 legs always tell the truth.
    - Blue: "Together we have 28 legs."
    - Green: "Together we have 27 legs."
    - Yellow: "Together we have 26 legs."
    - Red: "Together we have 25 legs."
   Prove that the Green octopus has 6 legs, and the Blue, Yellow, and Red octopuses each have 7 legs.
-/
theorem octopus_legs (L_B L_G L_Y L_R : ℕ) (H1 : (L_B + L_G + L_Y + L_R = 28 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 27 → L_B + L_G + L_Y + L_R = 27) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 26 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 25 → L_B ≠ 7)) : 
  (L_G = 6) ∧ (L_B = 7) ∧ (L_Y = 7) ∧ (L_R = 7) :=
sorry

end octopus_legs_l604_604256


namespace SM_parallel_BC_l604_604084

variable {A B C E F X K M N S : Type*}

-- Given conditions
variables
  (TriangleABC : ∀ {E F : Type*}, Triangle E F → Type*)
  (CircumcircleTriangleABC : Circle ABC)
  (PointsEandF : ChosenPoints E F)
  (CircCircleAEF : Circumcircle (mk_triangle A E F))
  (GammaIntersectX : Intersection CircCircleAEF CircumcircleTriangleABC X)
  (CircCircleABE : Circumcircle (mk_triangle A B E))
  (CircCircleACF : Circumcircle (mk_triangle A C F))
  (CircCircleIntersectK : Intersection CircCircleABE CircCircleACF K)
  (AK_Intersect_M: LineIntersect AK CircumcircleTriangleABC M)
  (ReflectedPointN : Reflection M BC N)
  (XN_Intersect_S: LineIntersect XN CircumcircleTriangleABC S)
  (Gamma : Circle ABC)
  (M_not_A : M ≠ A)
  (X_not_S : S ≠ X)
  (BC_line: Line BC)

-- Proof statement
theorem SM_parallel_BC
  (h1: mk_triangle A B C)
  (h2: CircumcircleTriangleABC)
  (h3: PointsEandF)
  (h4: CircCircleAEF)
  (h5: GammaIntersectX)
  (h6: CircCircleABE)
  (h7: CircCircleACF)
  (h8: CircCircleIntersectK)
  (h9: AK_Intersect_M)
  (h10: ReflectedPointN)
  (h11: XN_Intersect_S)
  (h12: Gamma)
  (h13: M_not_A)
  (h14: X_not_S)
  : Parallel SM BC := 
sorry

end SM_parallel_BC_l604_604084


namespace empty_square_exists_in_4x4_l604_604051

theorem empty_square_exists_in_4x4  :
  ∀ (points: Finset (Fin 4 × Fin 4)), points.card = 15 → 
  ∃ (i j : Fin 4), (i, j) ∉ points :=
by
  sorry

end empty_square_exists_in_4x4_l604_604051


namespace set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604663

-- Problem 1: Prove that a set of 7 numbers cannot have an arithmetic mean of 6 under given conditions.
theorem set_of_7_numbers_not_possible {s : Finset ℝ} (h_card : s.card = 7) (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / 7 = 6) : False := sorry

-- Problem 2: Prove that the number of elements in the set must be at least 9 if the arithmetic mean is 6.
theorem minimum_elements_with_mean_6 {s : Finset ℝ} (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / s.card = 6) (h_card : s.card = 2 * (s.card / 2) + 1) : 8 < s.card := sorry

end set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604663


namespace triangle_solution_l604_604469

variables {A B C : ℝ} {a b c : ℝ}

-- Defining the conditions of the triangle
def triangle_condition (A B C a b c : ℝ) :=
  C = 2 * A ∧
  cos A = 3 / 4

-- Defining the dot product condition
def dot_product_condition (a b c A B C : ℝ) :=
  let cos_B := -(cos A * cos (2 * A) - sin A * sqrt (1 - (cos 2 * A)^2)) in
  a * c * cos_B = 27 / 2

-- The main theorem to prove
theorem triangle_solution
  (h₁ : triangle_condition A B C a b c)
  (h₂ : dot_product_condition a b c A B C)
  : cos (2 * A) = 1 / 8 ∧
    -(cos A * cos (2 * A) - sin A * sqrt (1 - (cos (2 * A))^2)) = 9 / 16 ∧
    b = 5 :=
by
  sorry

end triangle_solution_l604_604469


namespace magnitude_of_z_l604_604794

noncomputable def z (i : ℂ) : ℂ := (2 - i) / (2 + i)

theorem magnitude_of_z (i : ℂ) (h : i * i = -1) : complex.abs (z i) = 1 :=
by
  sorry

end magnitude_of_z_l604_604794


namespace complex_expression_equality_l604_604881

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end complex_expression_equality_l604_604881


namespace carlos_gold_quarters_l604_604737

theorem carlos_gold_quarters:
  (let quarter_weight := 1 / 5 in
   let melt_value_per_ounce := 100 in
   let store_value_per_quarter := 0.25 in
   let quarters_per_ounce := 1 / quarter_weight in
   let total_melt_value := melt_value_per_ounce * quarters_per_ounce in
   let total_store_value := store_value_per_quarter * quarters_per_ounce in
   total_melt_value / total_store_value = 80) :=
by
  let quarter_weight := 1 / 5
  let melt_value_per_ounce := 100
  let store_value_per_quarter := 0.25
  let quarters_per_ounce := 1 / quarter_weight
  let total_melt_value := melt_value_per_ounce * quarters_per_ounce
  let total_store_value := store_value_per_quarter * quarters_per_ounce
  have : total_melt_value / total_store_value = 80 := sorry
  exact this

end carlos_gold_quarters_l604_604737


namespace inverse_proposition_l604_604587

-- Definition of the proposition
def complementary_angles_on_same_side (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- The original proposition
def original_proposition (l m : Line) : Prop := complementary_angles_on_same_side l m → parallel_lines l m

-- The statement of the proof problem
theorem inverse_proposition (l m : Line) :
  (complementary_angles_on_same_side l m → parallel_lines l m) →
  (parallel_lines l m → complementary_angles_on_same_side l m) := sorry

end inverse_proposition_l604_604587


namespace angle_PQR_eq_90_l604_604921

theorem angle_PQR_eq_90
  (R S P Q : Type)
  [IsStraightLine R S P]
  (angle_QSP : ℝ)
  (h : angle_QSP = 70) :
  ∠PQR = 90 :=
by
  sorry

end angle_PQR_eq_90_l604_604921


namespace fluffy_carrots_l604_604393

theorem fluffy_carrots :
  ∃ (a : ℕ), a + 2 * a + 4 * a = 84 ∧ (4 * a = 48) :=
begin
  -- we need to find an a that satisfies these conditions
  sorry
end

end fluffy_carrots_l604_604393


namespace solution_exists_l604_604642

theorem solution_exists :
  ∃ x : ℝ, x = 2 ∧ (-2 * x + 4 = 0) :=
sorry

end solution_exists_l604_604642


namespace problem_statement_l604_604814

-- Define the given conditions
def P : ℝ × ℝ := (4/5, -3/5)
def α : ℝ 

-- Lemmas and theorems to be proven
theorem problem_statement (h : P = (Real.cos α, Real.sin α)) :
  Real.cos α = 4/5 ∧
  Real.tan α = -3/4 ∧
  Real.sin (α + Real.pi) = 3/5 :=
by
  sorry

end problem_statement_l604_604814


namespace find_product_pf1_pf2_l604_604529

-- Definitions of conditions
variable {x y : ℝ}

def is_on_ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 12 = 1

def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

def dot_product (p1 p2 : ℝ × ℝ) : ℝ := p1.1 * p2.1 + p1.2 * p2.2

-- Problem statement
theorem find_product_pf1_pf2 
  (P : ℝ × ℝ) 
  (hP : is_on_ellipse P.1 P.2)
  (dot_eq_nine : dot_product (P - left_focus) (P - right_focus) = 9) 
  : (euclidean_dist (P - left_focus) 0) * (euclidean_dist (P - right_focus) 0) = 15 := sorry

end find_product_pf1_pf2_l604_604529


namespace triangle_side_length_l604_604077

-- Define the geometry of the triangle and the medians
structure TriangleABC where
  A B C M N G : Type
  medians : Vector ℝ 2 → ℝ
  m_AM : medians 15
  m_BN : medians 20
  perpendicular : medians ⊥ medians

-- Main problem statement
theorem triangle_side_length (t : TriangleABC) : t.AB = 50 / 3 := by
  sorry

end triangle_side_length_l604_604077


namespace min_segments_to_erase_l604_604162

noncomputable def nodes (m n : ℕ) : ℕ := (m - 2) * (n - 2)

noncomputable def segments_to_erase (m n : ℕ) : ℕ := (nodes m n + 1) / 2

theorem min_segments_to_erase (m n : ℕ) (hm : m = 11) (hn : n = 11) :
  segments_to_erase m n = 41 := by
  sorry

end min_segments_to_erase_l604_604162


namespace unique_parallel_plane_through_point_l604_604266

noncomputable def exists_unique_parallel_plane (Point : Type) (Plane : Type)
    [euclidean_geometry Point Plane] (A : Point) (P : Plane) : Prop :=
 ∃! (Q : Plane), Q ∋ A ∧ Q ∥ P

axiom point : Type
axiom plane : Type
axiom euclidean_geometry : Type → Type → Prop
axiom A : point 
axiom P : plane 
axiom instance: euclidean_geometry point plane

theorem unique_parallel_plane_through_point (A : point) (P : plane):
  exists_unique_parallel_plane point plane instance A P :=
sorry

end unique_parallel_plane_through_point_l604_604266


namespace train_crossing_platform_time_l604_604679

theorem train_crossing_platform_time :
  ∀ (length_train length_platform : ℕ) (speed_train_kmph : ℕ),
  length_train = 250 →
  length_platform = 200 →
  speed_train_kmph = 90 →
  let speed_train_mps := speed_train_kmph * 1000 / 3600 in
  let total_distance := length_train + length_platform in
  let time := total_distance / speed_train_mps in
  time = 18 :=
by
  intros length_train length_platform speed_train_kmph 
  intros h1 h2 h3
  rw [h1, h2, h3]
  let speed_train_mps := 90 * 1000 / 3600
  let total_distance := 250 + 200
  let time := total_distance / speed_train_mps 
  have h_speed : speed_train_mps = 25 := by norm_num
  rw [←h_speed]
  have h_distance : total_distance = 450 := by norm_num
  rw [←h_distance]
  have h_time : time = 18 := by norm_num
  exact h_time

end train_crossing_platform_time_l604_604679


namespace num_zeros_1_div_25_pow_10_l604_604583

theorem num_zeros_1_div_25_pow_10 : 
  let n := 25^10 in
  let dec_repr := 1 / (n : ℝ) in
  (dec_repr < 10^(-20 : ℝ)) ∧ (dec_repr ≥ 10^(-21 : ℝ)) :=
begin
  -- The proof will go here.
  sorry
end

end num_zeros_1_div_25_pow_10_l604_604583


namespace people_joined_l604_604155

theorem people_joined (total_left : ℕ) (total_remaining : ℕ) (Molly_and_parents : ℕ)
  (h1 : total_left = 40) (h2 : total_remaining = 63) (h3 : Molly_and_parents = 3) :
  ∃ n, n = 100 := 
by
  sorry

end people_joined_l604_604155


namespace not_a_solution_set4_l604_604350

def set1 : ℝ × ℝ := (1, 2)
def set2 : ℝ × ℝ := (2, 0)
def set3 : ℝ × ℝ := (0.5, 3)
def set4 : ℝ × ℝ := (-2, 4)

noncomputable def is_solution (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 = 4

theorem not_a_solution_set4 : ¬ is_solution set4 := 
by 
  sorry

end not_a_solution_set4_l604_604350


namespace upgraded_fraction_l604_604624

variable (U_A : ℕ)

def N_A_unit := U_A / 4
def N_A := 24 * N_A_unit
def Total_A := U_A + N_A

def N_B_unit := 2 * N_A_unit
def U_B := 3 * U_A
def N_B := 36 * N_B_unit
def Total_B := U_B + N_B

def N_C_unit := 3 * N_A_unit
def U_C := 4 * U_A
def N_C := 48 * N_C_unit
def Total_C := U_C + N_C

def Total_Upgraded := U_A + U_B + U_C
def Total_Sensors := Total_A + Total_B + Total_C

theorem upgraded_fraction:
    (Total_Upgraded.toFloat / Total_Sensors.toFloat) = (1 / 8.5) :=
by 
  sorry

end upgraded_fraction_l604_604624


namespace conjugate_of_z_l604_604435

-- Define the given complex number z
def z : ℂ := (1 - complex.I) / (1 + complex.I)

-- State the theorem to prove that the conjugate of z is i
theorem conjugate_of_z : complex.conj z = complex.I :=
by
  -- Omitted proof because it is not required for this task
  sorry

end conjugate_of_z_l604_604435


namespace magnitude_w_l604_604083

noncomputable def z : ℂ := ((-11 + 13 * Complex.I) ^ 3 * (24 - 7 * Complex.I) ^ 4) / (3 + 4 * Complex.I)

noncomputable def w : ℂ := Complex.conj z / z

theorem magnitude_w : Complex.abs w = 1 :=
by
  sorry

end magnitude_w_l604_604083


namespace parallelogram_area_is_sqrt_2_l604_604957

open Real

variables (p q : ℝ^3)
variable (hp : ∥p∥ = 1)
variable (hq : ∥q∥ = 1)
variable (θ : ℝ)
variable (hθ : θ = π / 4)

-- Define the variables for the diagonals
def a := p + 3 • q
def b := 3 • p + q

noncomputable def parallelogram_area := ∥a × b∥

theorem parallelogram_area_is_sqrt_2 (hp : ∥p∥ = 1) (hq : ∥q∥ = 1) (hθ : θ = π / 4) : 
  parallelogram_area p q = 2 * sqrt 2 :=
sorry

end parallelogram_area_is_sqrt_2_l604_604957


namespace ahn_max_result_l604_604346

theorem ahn_max_result :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (3 * (200 + n)) = 3597 :=
by
  use 999
  split
  repeat {split}
  all_goals {
    sorry
  }

end ahn_max_result_l604_604346


namespace find_angle_C_find_triangle_area_l604_604471

noncomputable def measure_angle_C (a b c : ℝ) (A B C : ℝ) (h: 2 * c * Real.cos C + b * Real.cos A + a * Real.cos B = 0) : ℝ :=
  C

theorem find_angle_C (a b c : ℝ) (A B C : ℝ) (h: 2 * c * Real.cos C + b * Real.cos A + a * Real.cos B = 0) : 
  measure_angle_C a b c A B C h = (2 * Real.pi) / 3 :=
sorry

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) (h: 2 * c * Real.cos C + b * Real.cos A + a * Real.cos B = 0) : ℝ :=
  1 / 2 * a * b * Real.sin C

theorem find_triangle_area (a b c : ℝ) (A B C : ℝ) (hc: c = 3) (hA: A = Real.pi / 6)
  (h: 2 * c * Real.cos C + b * Real.cos A + a * Real.cos B = 0) :
  triangle_area a b c A B C h = (3 * Real.sqrt(3)) / 4 :=
sorry

end find_angle_C_find_triangle_area_l604_604471


namespace intersection_of_sets_l604_604110

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l604_604110


namespace decreasing_interval_imp_m_lt_one_eighth_tangent_line_unique_common_point_imp_m_half_l604_604005

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - x + Real.log x

theorem decreasing_interval_imp_m_lt_one_eighth (m : ℝ)
  (h1 : ∃ (D : Set ℝ), (D ⊆ Set.Ioi 0) ∧ IsDecreasingOn (f m) D) :
  m < (1 : ℚ) / 8 := sorry

theorem tangent_line_unique_common_point_imp_m_half (m : ℝ) 
  (h1 : 0 < m) 
  (h2 : m ≤ (1 : ℚ) / 2) 
  (h3 : ∀ (x : ℝ), TangentLineAt (f m) 1 = CyclopsCurveTangent (f m) x) :
  m = (1 : ℚ) / 2 := sorry

end decreasing_interval_imp_m_lt_one_eighth_tangent_line_unique_common_point_imp_m_half_l604_604005


namespace angle_PQR_is_90_l604_604930

theorem angle_PQR_is_90 {P Q R S : Type}
  (is_straight_line_RSP : ∃ P R S : Type, (angle R S P = 180)) 
  (angle_QSP : angle Q S P = 70)
  (isosceles_RS_SQ : ∃ (RS SQ : Type), RS = SQ)
  (isosceles_PS_SQ : ∃ (PS SQ : Type), PS = SQ) : angle P Q R = 90 :=
by 
  sorry

end angle_PQR_is_90_l604_604930


namespace weight_needed_to_open_cave_l604_604238

theorem weight_needed_to_open_cave (weight_current : ℕ) (weight_total : ℕ) (weight_needed : ℕ) 
  (h1 : weight_current = 234) (h2 : weight_total = 712) :
  weight_total - weight_current = weight_needed :=
by
  sorry

#eval weight_needed_to_open_cave 234 712 478 -- should output true if the theorem is correct

end weight_needed_to_open_cave_l604_604238


namespace repetend_of_5_over_17_is_correct_l604_604381

/-- 
  Theorem: The 16-digit repetend in the decimal representation of 5 / 17 is 2941176470588235.
  Conditions: 5 / 17 has a repeating decimal representation.
-/
theorem repetend_of_5_over_17_is_correct : 
  let rep := "2941176470588235" in
  (∃ t : ℕ, 5 / 17 = (t.to_nat.sqrt - 1 / 10^16) ∧ t.to_nat.sqrt = 5 / 17 ∧ repeated t == repeated 5 / 17 ∧ 
  repeated t = 2941176470588235 ) := by sorry

end repetend_of_5_over_17_is_correct_l604_604381


namespace log_function_domain_correct_l604_604228

def log_function_domain : Set ℝ :=
  {x | 1 < x ∧ x < 3 ∧ x ≠ 2}

theorem log_function_domain_correct :
  (∀ x : ℝ, y = log (x - 1) (3 - x) → x ∈ log_function_domain) :=
by
  sorry

end log_function_domain_correct_l604_604228


namespace find_set_of_t_l604_604789

noncomputable def f (x : ℝ) : ℝ := x^4 + Real.exp (|x|)

theorem find_set_of_t (t : ℝ) :
  (2 * f (Real.log t) - f (Real.log (1 / t)) ≤ f 2) ↔ (e ^ (-2) ≤ t ∧ t ≤ e ^ 2) :=
by
  sorry

end find_set_of_t_l604_604789


namespace carlos_gold_quarters_l604_604734

theorem carlos_gold_quarters:
  (let quarter_weight := 1 / 5 in
   let melt_value_per_ounce := 100 in
   let store_value_per_quarter := 0.25 in
   let quarters_per_ounce := 1 / quarter_weight in
   let total_melt_value := melt_value_per_ounce * quarters_per_ounce in
   let total_store_value := store_value_per_quarter * quarters_per_ounce in
   total_melt_value / total_store_value = 80) :=
by
  let quarter_weight := 1 / 5
  let melt_value_per_ounce := 100
  let store_value_per_quarter := 0.25
  let quarters_per_ounce := 1 / quarter_weight
  let total_melt_value := melt_value_per_ounce * quarters_per_ounce
  let total_store_value := store_value_per_quarter * quarters_per_ounce
  have : total_melt_value / total_store_value = 80 := sorry
  exact this

end carlos_gold_quarters_l604_604734


namespace pasta_preferences_l604_604686

theorem pasta_preferences :
  ∀ (students_total students_spaghetti students_tortellini students_penne : ℕ),
  students_total = 800 →
  students_spaghetti = 260 →
  students_tortellini = 160 →
  (students_penne : ℚ) / students_tortellini = 3 / 4 →
  students_spaghetti - students_penne = 140 :=
begin
  intros students_total students_spaghetti students_tortellini students_penne,
  intros h_total h_spaghetti h_tortellini h_ratio,
  sorry
end

end pasta_preferences_l604_604686


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604198

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604198


namespace find_m_min_value_a2_b2_c2_l604_604790

open Real

noncomputable def f (x m : ℝ) : ℝ := 2 * |x - 1| - |2 * x + m|

theorem find_m (m : ℝ) (h0 : m ≥ 0) : 
  (∀ x : ℝ, f x m ≤ 3) → m = 1 :=
by
  intro h1
  have h2 : f 1 m = m + 2 := by sorry
  have h3 : m + 2 ≤ 3 := by sorry
  exact (Int.lt_of_le_and_ne (by linarith [h0]) (by linarith)⁻¹).symm
  sorry


theorem min_value_a2_b2_c2 (a b c m : ℝ) (h : a - 2 * b + c = m) (h_m : m = 1) :
  a^2 + b^2 + c^2 ≥ 1 / 6 :=
by
  apply le_of_eq
  have h' : a - 2 * b + c = 1 
    by { rw ← h_m, exact h }
  have h4 : (a^2 + b^2 + c^2) * ((1)^2 + (-2)^2 + (1)^2) ≥ (a - 2 * b + c)^2 
    by sorry
  rw h'
  linarith
  sorry

end find_m_min_value_a2_b2_c2_l604_604790


namespace tan_sum_property_l604_604776

theorem tan_sum_property (t23 t37 : ℝ) (h1 : 23 + 37 = 60) (h2 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3) :
  Real.tan (23 * Real.pi / 180) + Real.tan (37 * Real.pi / 180) + Real.sqrt 3 * Real.tan (23 * Real.pi / 180) * Real.tan (37 * Real.pi / 180) = Real.sqrt 3 :=
sorry

end tan_sum_property_l604_604776


namespace remainder_of_sum_factorials_up_to_50_div_20_l604_604468

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def sum_factorials_up_to (n : ℕ) : ℕ :=
  (List.range (n + 1)).sum (λ x => factorial x)

theorem remainder_of_sum_factorials_up_to_50_div_20 :
  (sum_factorials_up_to 50) % 20 = 13 :=
by
  sorry

end remainder_of_sum_factorials_up_to_50_div_20_l604_604468


namespace gary_initial_money_l604_604784

/-- The initial amount of money Gary had, given that he spent $55 and has $18 left. -/
theorem gary_initial_money (amount_spent : ℤ) (amount_left : ℤ) (initial_amount : ℤ) 
  (h1 : amount_spent = 55) 
  (h2 : amount_left = 18) 
  : initial_amount = amount_spent + amount_left :=
by
  sorry

end gary_initial_money_l604_604784


namespace levi_additional_scores_needed_l604_604973

-- Definitions based on the given conditions
def levi_initial_scores := 8
def brother_initial_scores := 12
def brother_additional_scores := 3
def levi_goal_surpass := 5

-- Levi needs to score another 12 times to reach his goal
theorem levi_additional_scores_needed : 
  levi_initial_scores + ?extra_scores_needed >= brother_initial_scores + brother_additional_scores + levi_goal_surpass :=
by
  let brother_total_scores := brother_initial_scores + brother_additional_scores;
  let levi_needed_scores := brother_total_scores + levi_goal_surpass;
  have h1 : levi_initial_scores + 12 = levi_needed_scores, by sorry;
  exact h1

end levi_additional_scores_needed_l604_604973


namespace part_a_part_b_l604_604634

theorem part_a (a b : ℝ) (n m : ℤ) (M : ℝ) (h1 : 0 ≤ M ∧ M < 1)
  (h2 : log 10 a = n + M) (h3 : log 10 b = m + M) : 
  a = b * 10 ^ (n - m) :=
sorry

theorem part_b (a b : ℝ) (n : ℤ) (M1 M2 : ℝ) (h1 : 0 ≤ M1 ∧ M1 < 1) (h2 : 0 ≤ M2 ∧ M2 < 1)
  (h3 : log 10 a = n + M1) (h4 : log 10 b = n + M2) : 
  1/10 < a / b ∧ a / b < 10 :=
sorry

end part_a_part_b_l604_604634


namespace an_squared_diff_consec_cubes_l604_604948

theorem an_squared_diff_consec_cubes (a b : ℕ → ℤ) (n : ℕ) :
  a 1 = 1 → b 1 = 0 →
  (∀ n ≥ 1, a (n + 1) = 7 * (a n) + 12 * (b n) + 6) →
  (∀ n ≥ 1, b (n + 1) = 4 * (a n) + 7 * (b n) + 3) →
  a n ^ 2 = (b n + 1) ^ 3 - (b n) ^ 3 :=
by
  sorry

end an_squared_diff_consec_cubes_l604_604948


namespace length_of_train_l604_604338

theorem length_of_train (speed_km_hr : ℝ) (time_sec : ℝ) (speed_conversion : speed_km_hr = 60) (time_conversion : time_sec = 3) :
  (speed_km_hr * 1000 / 3600) * time_sec ≈ 50.01 :=
by
  have speed_m_s : ℝ := (60 * 1000) / 3600
  have length_train : ℝ := speed_m_s * 3
  have approx_length : ℝ := 50.01
  sorry

end length_of_train_l604_604338


namespace area_intersection_l604_604837

noncomputable def set_M : Set ℂ := {z : ℂ | abs (z - 1) ≤ 1}
noncomputable def set_N : Set ℂ := {z : ℂ | complex.arg z ≥ π / 4}

theorem area_intersection (S : ℝ) :
  S = (3 / 4) * real.pi - 1 / 2 →
  ∃ z : ℂ, z ∈ set_M ∩ set_N := 
sorry

end area_intersection_l604_604837


namespace value_of_nested_custom_div_l604_604365

def custom_div (x y z : ℕ) (hz : z ≠ 0) : ℕ :=
  (x + y) / z

theorem value_of_nested_custom_div : custom_div (custom_div 45 15 60 (by decide)) (custom_div 3 3 6 (by decide)) (custom_div 20 10 30 (by decide)) (by decide) = 2 :=
sorry

end value_of_nested_custom_div_l604_604365


namespace sum_of_coefficients_l604_604459

-- Define the polynomial equality condition
noncomputable def polynomial_identity (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) : Prop := 
  ∀ x : ℤ, (x - 1)^5 = a_5 * (x + 1)^5 + a_4 * (x + 1)^4 + a_3 * (x + 1)^3 + a_2 * (x + 1)^2 + a_1 * (x + 1) + a_0

-- State the problem as a theorem
theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) 
  (h : polynomial_identity a_0 a_1 a_2 a_3 a_4 a_5) 
  (h1 : (x - 1)^5 = -32 when x = -1)
  (h2 : (x - 1)^5 - a_0 when x = 0 = -1) :
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
sorry

end sum_of_coefficients_l604_604459


namespace lcm_max_value_l604_604280

   open Nat

   theorem lcm_max_value : 
     max (max (max (max (max (lcm 18 3) (lcm 18 9)) (lcm 18 12)) (lcm 18 16)) (lcm 18 21)) (lcm 18 18) = 144 :=
   by
     sorry
   
end lcm_max_value_l604_604280


namespace average_sales_l604_604995

theorem average_sales
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 90)
  (h2 : a2 = 50)
  (h3 : a3 = 70)
  (h4 : a4 = 110)
  (h5 : a5 = 80) :
  (a1 + a2 + a3 + a4 + a5) / 5 = 80 :=
by
  sorry

end average_sales_l604_604995


namespace square_area_from_diagonal_l604_604589

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  ∃ (A : ℝ), A = 72 :=
by
  sorry

end square_area_from_diagonal_l604_604589


namespace find_a_l604_604826

theorem find_a 
  (a : ℝ)
  (h1 : a ≠ 1 / 2)
  (h2 : ∀ x, f x = (2 * x + 1) / (x + a) ∧ f x = x ↔ f x = f⁻¹ x)
  : a = -2 := sorry

end find_a_l604_604826


namespace integral_roots_l604_604211

theorem integral_roots :
  ∃ x y z : ℤ, x^z = y^(z - 1) ∧
               2^z = 8 * 2^x ∧
               x^2 + y^2 + z^2 = 72 ∧
               x = 3 ∧
               y = 3 ∧
               z = 6 :=
by
  sorry

end integral_roots_l604_604211


namespace min_distance_bc_l604_604842

open Real EuclideanGeometry

variables (a b c : EuclideanSpace ℝ (Fin 2))

-- Given conditions
variables (h1 : ∥a∥ = 2)
variables (h2 : ∥b∥ = 2)
variables (h3 : inner a b = 2)
variables (h4 : inner (a - c) (b - 2 • c) = 0)

-- Problem statement
theorem min_distance_bc : ∃ c, dist b c = (sqrt 7 - sqrt 3) / 2 :=
by sorry

end min_distance_bc_l604_604842


namespace term_x4_in_binomial_expansion_l604_604516

-- Defining the imaginary unit i
def i := Complex.I

-- The binomial expansion formula is used from Lean library definitions
theorem term_x4_in_binomial_expansion : ∀ (x : ℂ), 
  (x + i)^6 = (C(6, 4) * x^4 * (i)^2 + term_x4) = -15 * x^4 := sorry

end term_x4_in_binomial_expansion_l604_604516


namespace part_a_part_b_l604_604669

-- Part (a)
theorem part_a : ¬(∃ s : Finset ℝ, s.card = 7 ∧
  (∑ x in s.filter (λ x, x >= 10), x).card >= 4 ∧
  (∑ x in s, x) >= 43 ∧
  (∑ x in s, x) / 7 = 6) :=
by 
  sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n ≥ 4) :
  (∃ s : Finset ℝ, s.card = 2 * n + 1 ∧
    (s.filter (λ x, x >= 10)).card = n + 1 ∧
    (s.filter (λ x, x >= 1 ∧ x < 10)).card = n ∧
    (∑ x in s, x) = 12 * n + 6) :=
by 
  sorry

end part_a_part_b_l604_604669


namespace probability_same_color_l604_604873

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end probability_same_color_l604_604873


namespace cost_price_correct_l604_604541

noncomputable def cost_price : ℝ :=
  let sell_price_10_discount := 24000 in -- Selling price at 10% discount
  let profit_rate_5_discount := 1.12 in -- Selling price at 5% discount as a proportion of cost price
  let discount_5 := 0.95 in -- 5% discount
  let discount_10 := 0.90 in -- 10% discount
  let marked_price := sell_price_10_discount / discount_10 in
  let cost_price := marked_price * discount_5 / profit_rate_5_discount in
  cost_price

theorem cost_price_correct : cost_price ≈ 22619.76 :=
by
  sorry

end cost_price_correct_l604_604541


namespace mixed_fruit_juice_litres_opened_l604_604592

theorem mixed_fruit_juice_litres_opened (cocktail_cost_per_litre : ℝ)
  (mixed_juice_cost_per_litre : ℝ) (acai_cost_per_litre : ℝ)
  (acai_litres_added : ℝ) (total_mixed_juice_opened : ℝ) :
  cocktail_cost_per_litre = 1399.45 ∧
  mixed_juice_cost_per_litre = 262.85 ∧
  acai_cost_per_litre = 3104.35 ∧
  acai_litres_added = 23.333333333333336 ∧
  (mixed_juice_cost_per_litre * total_mixed_juice_opened + 
  acai_cost_per_litre * acai_litres_added = 
  cocktail_cost_per_litre * (total_mixed_juice_opened + acai_litres_added)) →
  total_mixed_juice_opened = 35 :=
sorry

end mixed_fruit_juice_litres_opened_l604_604592


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604195

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604195


namespace radius_of_inscribed_circle_is_integer_l604_604183

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l604_604183


namespace question_1_question_2_l604_604446

noncomputable def ellipse_C (x y : ℝ) : Prop :=
  (x^2) / 2 + y^2 = 1

def focus_F : ℝ × ℝ := (1, 0)

def point_M : ℝ × ℝ := (2, 0)

theorem question_1 :
  ∀ (x y : ℝ),  
  line l : Prop := λ (x y : ℝ), x = 1,
  l (fst focus_F) (snd focus_F) → 
  (ellipse_C 1 y → (y = sqrt 2 / 2 ∨ y = - sqrt 2 / 2) → 
  ( ∃ (y : ℝ), y = - sqrt 2 / 2 * x + sqrt 2 ∨ y = sqrt 2 / 2 * x - sqrt 2)
  :=
sorry

theorem question_2 :
  ∀ (k x y : ℝ), 
  k ≠ 0 →
  ( ∃ (A B : ℝ × ℝ), 
    (k * (fst A - 1) = (snd A) ∧ k * (fst B - 1) = (snd B)) → 
    ellipse_C (fst A) (snd A) ∧ ellipse_C (fst B) (snd B) →
    ( (∃ (x1 x2 : ℝ), x1 + x2 = 4 * k^2 / (2 * k^2 + 1) ∧ x1 * x2 = (2 * k^2 - 2) / (2 * k^2 + 1)) → (∃ MA MB : ℝ, MA + MB = 0) → (angle OMA = angle OMB) → (OMA / OMB = 1)
  :=
sorry

end question_1_question_2_l604_604446


namespace number_of_students_in_second_class_l604_604577

theorem number_of_students_in_second_class (x : ℕ) :
  let avg1 := 40
      num1 := 30
      avg2 := 60
      total_avg := 52.5 in
  (num1 * avg1) + (x * avg2) = (num1 + x) * total_avg → x = 50 := 
by
  intros avg1 num1 avg2 total_avg h
  have h1 : (30 : ℕ) = 30 := rfl
  have h2 : (40 : ℝ) = 40 := rfl
  have h3 : (60 : ℝ) = 60 := rfl
  have h4 : (52.5 : ℝ) = 52.5 := rfl
  rw [h1, h2, h3, h4] at h
  sorry

end number_of_students_in_second_class_l604_604577


namespace min_tip_percentage_l604_604331

noncomputable def meal_cost : ℝ := 37.25
noncomputable def total_paid : ℝ := 40.975
noncomputable def tip_percentage (P : ℝ) : Prop := P > 0 ∧ P < 15 ∧ (meal_cost + (P/100) * meal_cost = total_paid)

theorem min_tip_percentage : ∃ P : ℝ, tip_percentage P ∧ P = 10 := by
  sorry

end min_tip_percentage_l604_604331


namespace inscribed_circle_diameter_l604_604408

theorem inscribed_circle_diameter {a b c : ℕ} (h_triangle : a = 9 ∧ b = 12 ∧ c = 15 ∧ a ^ 2 + b ^ 2 = c ^ 2) :
  2 * (a + b - c) / 2 = 6 :=
by
  rcases h_triangle with ⟨ha, hb, hc, h_right⟩
  subst ha hb hc
  rw [add_comm, add_sub_cancel]
  exact rfl

end inscribed_circle_diameter_l604_604408


namespace ball_game_total_cost_l604_604993

theorem ball_game_total_cost :
  (let a := 10; c := 11; p_a := 8; p_c := 4 in
  a * p_a + c * p_c = 124) :=
by
  sorry

end ball_game_total_cost_l604_604993


namespace math_books_conditions_correct_l604_604262

-- Definitions
def total_books := 53
def math_books (S : ℕ) := ∀ W : ℕ, S + W = total_books
def no_adj_physics_book (W : ℕ) := ∀ i : ℕ, ¬(W > 1 ∧ W ≤ (total_books - 1) / 2)
def adj_math_books (S : ℕ) := S > 0 ∧ S % 2 = 1

-- Theorem Statement
theorem math_books_conditions_correct : 
  ∀ S W : ℕ, math_books S W →
  no_adj_physics_book W →
  adj_math_books S →
  (S ≥ 35) ∧ (W ≤ 18) := 
by
  sorry

end math_books_conditions_correct_l604_604262


namespace log_sum_ge_three_l604_604085

open Real

theorem log_sum_ge_three (a : ℝ) (x₁ x₂ x₃ : ℝ) (h₁ : 0 < a) (h₂ : x₁ + x₂ + x₃ = 0) :
  log 2 (1 + a^x₁) + log 2 (1 + a^x₂) + log 2 (1 + a^x₃) ≥ 3 :=
  sorry

end log_sum_ge_three_l604_604085


namespace solve_inequality_l604_604428

noncomputable 
def f : ℝ → ℝ := sorry
noncomputable 
def f' : ℝ → ℝ := sorry

def symmetric_about_minus_one (f : ℝ → ℝ) := 
  ∀ x, f(x) = f(-2 - x)

def condition (f : ℝ → ℝ) (f' : ℝ → ℝ) :=
  ∀ x, x < -1 → (x + 1) * (f(x) + (x + 1) * f'(x)) < 0

theorem solve_inequality :
  (∀ x, f' x = derivative f x) →
  domain ℝ f → 
  symmetric_about_minus_one f →
  condition f f' →
  {x : ℝ | x * (f' (x - 1)) > f' 0} = set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end solve_inequality_l604_604428


namespace intersection_A_B_l604_604122

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l604_604122


namespace computation_of_expression_l604_604277

theorem computation_of_expression : 
  (7 ^ (-2 : ℤ)) ^ 0 + (7 ^ 0) ^ 2 + (7 ^ (-1 : ℤ) * 7 ^ 1) ^ 0 = 3 :=
by
  sorry

end computation_of_expression_l604_604277


namespace inscribed_circle_radius_integer_l604_604176

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l604_604176


namespace factor_expression_l604_604744

variable (x y : ℝ)

theorem factor_expression :
(3*x^3 + 28*(x^2)*y + 4*x) - (-4*x^3 + 5*(x^2)*y - 4*x) = x*(x + 8)*(7*x + 1) := sorry

end factor_expression_l604_604744


namespace find_LP_l604_604268

variables (A B C J L P M : Type)
variables (AC BC AJ CJ BJ AP LP : ℝ)
variables [linear_ordered_field ℝ]

-- Conditions from the problem
axiom AC_eq_500 : AC = 500
axiom BC_eq_400 : BC = 400
axiom AJ_eq_3CJ : AJ = 3 * CJ
axiom J_midpoint_BM : 2 * J = B + M
axiom CL_angle_bisector : is_angle_bisector C L (AC, BC)

-- Question to prove LP = 115.74
theorem find_LP (AC_eq_500 : AC = 500) (BC_eq_400 : BC = 400) (AJ_eq_3CJ : AJ = 3 * CJ) 
(J_midpoint_BM : 2 * J = B + M) (CL_angle_bisector : is_angle_bisector C L (AC, BC)) :
LP = 115.74 :=
  sorry

end find_LP_l604_604268


namespace gold_quarter_value_comparison_l604_604727

theorem gold_quarter_value_comparison:
  (worth_in_store per_quarter: ℕ → ℝ) 
  (weight_per_quarter in_ounce: ℝ) 
  (earning_per_ounce melted: ℝ) : 
  (worth_in_store 4  = 0.25) →
  (weight_per_quarter = 1/5) →
  (earning_per_ounce = 100) →
  (earning_per_ounce * weight_per_quarter / worth_in_store 4 = 80) :=
by
  -- The proof goes here
  sorry

end gold_quarter_value_comparison_l604_604727


namespace intersection_A_B_l604_604117

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l604_604117


namespace employee_selected_from_10th_group_is_47_l604_604325

theorem employee_selected_from_10th_group_is_47
  (total_employees : ℕ)
  (sampled_employees : ℕ)
  (total_groups : ℕ)
  (random_start : ℕ)
  (common_difference : ℕ)
  (selected_from_5th_group : ℕ) :
  total_employees = 200 →
  sampled_employees = 40 →
  total_groups = 40 →
  random_start = 2 →
  common_difference = 5 →
  selected_from_5th_group = 22 →
  (selected_from_5th_group = (4 * common_difference + random_start)) →
  (9 * common_difference + random_start) = 47 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end employee_selected_from_10th_group_is_47_l604_604325


namespace roots_of_P_l604_604764

-- Define the polynomial P(x) = x^3 + x^2 - 6x - 6
noncomputable def P (x : ℝ) : ℝ := x^3 + x^2 - 6 * x - 6

-- Define the statement that the roots of the polynomial P are -1, sqrt(6), and -sqrt(6)
theorem roots_of_P : ∀ x : ℝ, P x = 0 ↔ (x = -1) ∨ (x = sqrt 6) ∨ (x = -sqrt 6) :=
sorry

end roots_of_P_l604_604764


namespace centroid_of_circular_segment_l604_604698

theorem centroid_of_circular_segment (R α : ℝ)
    (S S1 S2 : ℝ) (h : ℝ) 
    (H1 : S1 = R^2 * α) 
    (H2 : S2 = (1 / 2) * R^2 * sin(2 * α))
    (H3 : S = R^2 * (α - sin(2 * α) / 2)) :
    (h = R * sin(α)) → 
    (S > 0) → 
    (OZ = (2 * h^3) / (3 * S)) :=
by
  -- definition of centroids and balancing equation steps are omitted for brevity
  sorry

end centroid_of_circular_segment_l604_604698


namespace standard_deviation_of_five_numbers_l604_604462

theorem standard_deviation_of_five_numbers
  (a : ℝ)
  (h_avg : (1 + 2 + 3 + 4 + a) / 5 = 4) :
  let s := Real.sqrt (1 / 5 * ((1 - 4)^2 + (2 - 4)^2 + (3 - 4)^2 + (4 - 4)^2 + (a - 4)^2)) in
  s = Real.sqrt 10 :=
by
  sorry

end standard_deviation_of_five_numbers_l604_604462


namespace distance_A_to_B_is_7km_l604_604630

theorem distance_A_to_B_is_7km
  (v1 v2 : ℝ) 
  (t_meet_before : ℝ)
  (t1_after_meet t2_after_meet : ℝ)
  (d1_before_meet d2_before_meet : ℝ)
  (d_after_meet : ℝ)
  (h1 : d1_before_meet = d2_before_meet + 1)
  (h2 : t_meet_before = d1_before_meet / v1)
  (h3 : t_meet_before = d2_before_meet / v2)
  (h4 : t1_after_meet = 3 / 4)
  (h5 : t2_after_meet = 4 / 3)
  (h6 : d1_before_meet + v1 * t1_after_meet = d_after_meet)
  (h7 : d2_before_meet + v2 * t2_after_meet = d_after_meet)
  : d_after_meet = 7 := 
sorry

end distance_A_to_B_is_7km_l604_604630


namespace range_of_f_l604_604424

-- Define the function f(x) = 4 sin^3(x) + sin^2(x) - 4 sin(x) + 8
noncomputable def f (x : ℝ) : ℝ :=
  4 * (Real.sin x) ^ 3 + (Real.sin x) ^ 2 - 4 * (Real.sin x) + 8

-- Statement to prove the range of f(x)
theorem range_of_f :
  ∀ x : ℝ, 6 + 3 / 4 ≤ f x ∧ f x ≤ 9 + 25 / 27 :=
sorry

end range_of_f_l604_604424


namespace rain_probability_at_least_once_l604_604246

theorem rain_probability_at_least_once :
  let p_rain := (3:ℚ) / 4
  let p_no_rain := 1 - p_rain
  let p_no_rain_five_days := p_no_rain ^ 5
  let p_rain_at_least_once := 1 - p_no_rain_five_days
  in p_rain_at_least_once = 1023 / 1024 := by
  sorry

end rain_probability_at_least_once_l604_604246


namespace f_periodic_l604_604970

def f (x : ℝ) : ℝ := |sin x|

theorem f_periodic : ¬ ¬ (∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x) := 
by
  sorry

end f_periodic_l604_604970


namespace length_AE_l604_604982

/-- Given points A, B, C, D, and E on a plane with distances:
  - CA = 12,
  - AB = 8,
  - BC = 4,
  - CD = 5,
  - DB = 3,
  - BE = 6,
  - ED = 3.
  Prove that AE = sqrt 113.
--/
theorem length_AE (A B C D E : ℝ × ℝ)
  (h1 : dist C A = 12)
  (h2 : dist A B = 8)
  (h3 : dist B C = 4)
  (h4 : dist C D = 5)
  (h5 : dist D B = 3)
  (h6 : dist B E = 6)
  (h7 : dist E D = 3) : 
  dist A E = Real.sqrt 113 := 
  by 
    sorry

end length_AE_l604_604982


namespace store_profit_l604_604697

theorem store_profit (sell_price : ℤ) (profit_rate : ℚ) (loss_rate : ℚ) (profit_piece_cost loss_piece_cost : ℚ) :
  sell_price = 180 → 
  profit_rate = 0.20 → 
  loss_rate = -0.10 → 
  profit_piece_cost * (1 + profit_rate) = sell_price → 
  loss_piece_cost * (1 + loss_rate) = sell_price → 
  (sell_price * 2) - (profit_piece_cost + loss_piece_cost) = 10 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5]
  sorry

end store_profit_l604_604697


namespace net_profit_start_year_better_investment_option_l604_604333

-- Question 1: From which year does the developer start to make a net profit?
def investment_cost : ℕ := 81 -- in 10,000 yuan
def first_year_renovation_cost : ℕ := 1 -- in 10,000 yuan
def renovation_cost_increase : ℕ := 2 -- in 10,000 yuan per year
def annual_rental_income : ℕ := 30 -- in 10,000 yuan per year

theorem net_profit_start_year : ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, ¬ (annual_rental_income * m > investment_cost + m^2) :=
by sorry

-- Question 2: Which option is better: maximizing total profit or average annual profit?
def profit_function (n : ℕ) : ℤ := 30 * n - (81 + n^2)
def average_annual_profit (n : ℕ) : ℤ := (30 * n - (81 + n^2)) / n
def max_total_profit_year : ℕ := 15
def max_total_profit : ℤ := 144 -- in 10,000 yuan
def max_average_profit_year : ℕ := 9
def max_average_profit : ℤ := 12 -- in 10,000 yuan

theorem better_investment_option : (average_annual_profit max_average_profit_year) ≥ (profit_function max_total_profit_year) / max_total_profit_year :=
by sorry

end net_profit_start_year_better_investment_option_l604_604333


namespace min_value_of_f_l604_604773

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / sqrt (x^2 + 5)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 :=
by
  sorry

end min_value_of_f_l604_604773


namespace solve_triangle_area_problem_l604_604269

noncomputable def pq_right_triangle_area (PR QR : ℝ) (right_angle_at_R : Bool) (S_midpoint_PQ : Bool) (PT QT : ℝ) (T_side_R : Bool)
(A : ℝ) (hx : A = 90) (hy : ℝ) (hz : ℝ) (H : A * real.sqrt hy / hz = 90 * real.sqrt 3 / 1) : ℝ :=
90 + 3 + 1

theorem solve_triangle_area_problem
  (PR : ℝ) (QR : ℝ) (right_angle_at_R : Bool) (PT : ℝ) (QT : ℝ) (S_midpoint_PQ : Bool) (T_side_R : Bool)
  (hx : PT = QT)
  (A : ℝ := pq_right_triangle_area PR QR right_angle_at_R S_midpoint_PQ PT QT T_side_R 90 3 1 sorry) : 
  A = 94 :=
by
  sorry

end solve_triangle_area_problem_l604_604269


namespace number_of_candidates_l604_604305

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
sorry

end number_of_candidates_l604_604305


namespace inequality_l604_604688

noncomputable def cyclic_sum {α : Type*} [AddCommGroup α] [HasSmul ℕ α] 
  (f : ℝ → α) (a b c : ℝ) := f a + f b + f c

noncomputable def tan_tan (x : ℝ) := Real.tan (Real.tan x)
noncomputable def tan_cot_half (x : ℝ) := Real.tan (Real.cot (x / 2))

theorem inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = Real.pi) :
  cyclic_sum tan_tan A B C - 2 * cyclic_sum tan_cot_half A B C ≥ -3 * Real.tan (Real.sqrt 3) := by
  sorry

end inequality_l604_604688


namespace even_subset_capacity_sum_112_l604_604510

open Finset

-- Define S_n
def S (n : ℕ) : Finset ℕ := range (n + 1)

-- Define capacity of a subset
def capacity (X : Finset ℕ) : ℕ := if X = ∅ then 0 else X.prod id

-- Define a predicate for an even subset
def is_even_subset (X : Finset ℕ) : Prop := capacity X % 2 = 0

-- Given n = 4
def S_4 : Finset ℕ := S 4

-- Function to calculate sum of capacities of even subsets of S_4
def sum_even_capacities (S : Finset ℕ) : ℕ :=
  univ.filter (λ X : Finset ℕ, is_even_subset X).sum capacity

theorem even_subset_capacity_sum_112 : sum_even_capacities S_4 = 112 := 
by sorry

end even_subset_capacity_sum_112_l604_604510


namespace area_intersection_M_N_l604_604839

open Complex Real

def M : set ℂ := {z | abs (z - 1) ≤ 1}
def N : set ℂ := {z | arg z ≥ π / 4}

theorem area_intersection_M_N :
  let S := area (M ∩ N)
  in S = (3 / 4) * π - 1 / 2 :=
sorry

end area_intersection_M_N_l604_604839


namespace angle_AEC_90_l604_604481

variables (A B C D E : Type)
variables (quadrilateral : A → B → C → D → Prop)
variables (bisector_meet : A → C → E → Prop)
variables (angle_A angle_C : ℝ)
variables (angle_A_val : angle_A = 70)
variables (angle_C_val : angle_C = 110)

theorem angle_AEC_90 : quadrilateral A B C D →
                        bisector_meet A C E →
                        angle_A = 70 →
                        angle_C = 110 →
                        ∃ (angle_AEC : ℝ), angle_AEC = 90 :=
by
  intros
  use 90
  sorry

end angle_AEC_90_l604_604481


namespace volume_of_prism_l604_604584

theorem volume_of_prism (h w d : ℝ) 
  (front_area: w * h = 12): 
  (side_area: d * h = 6): 
  (top_area: d * w = 8) : 
  w * h * d = 24 := 
by
  sorry

end volume_of_prism_l604_604584


namespace max_boxes_in_wooden_box_l604_604306

structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

def volume (b : Box) : ℕ :=
  b.length * b.width * b.height

theorem max_boxes_in_wooden_box :
  let large_box := Box.mk 800 1000 600
  let small_box := Box.mk 4 5 6
  volume large_box / volume small_box = 4000000 := by
  -- Introduce the definitions of volume for both boxes
  let V_large := volume large_box
  let V_small := volume small_box
  have h1 : V_large = 480000000 := by
    -- Calculate volume of large_box
    sorry
  have h2 : V_small = 120 := by
    -- Calculate volume of small_box
    sorry
  show V_large / V_small = 4000000 from
  -- Final division and proof
  sorry

end max_boxes_in_wooden_box_l604_604306


namespace same_color_probability_l604_604866

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l604_604866


namespace max_value_of_expression_l604_604524

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 := 
sorry

end max_value_of_expression_l604_604524


namespace smaller_circle_radius_l604_604069

theorem smaller_circle_radius (r_large: ℝ) (n: ℕ) (r_small: ℝ) : 
  r_large = 10 → 
  n = 7 → 
  (4 * (2 * r_small) = 2 * r_large) → 
  r_small = 2.5 :=
by
  intros h1 h2 h3
  rw [h1] at h3
  simp at h3
  linarith

end smaller_circle_radius_l604_604069


namespace angle_is_50_l604_604433

-- Define the angle, supplement, and complement
def angle (x : ℝ) := x
def supplement (x : ℝ) := 180 - x
def complement (x : ℝ) := 90 - x
def condition (x : ℝ) := supplement x = 3 * (complement x) + 10

theorem angle_is_50 :
  ∃ x : ℝ, condition x ∧ x = 50 :=
by
  -- Here we show the existence of x that satisfies the condition and is equal to 50
  sorry

end angle_is_50_l604_604433


namespace tan_sum_eq_three_l604_604031

-- Define the necessary conditions
variables (θ : ℝ)
hypothesis h1 : Real.sin (2 * θ) = 2/3

-- Define the goal
theorem tan_sum_eq_three (h1 : Real.sin (2 * θ) = 2/3) : 
  Real.tan θ + 1 / Real.tan θ = 3 :=
sorry

end tan_sum_eq_three_l604_604031


namespace range_of_m_l604_604804

noncomputable def f (x : ℝ) : ℝ := (1/2) * (Real.exp x + Real.exp (-x))
noncomputable def g (x : ℝ) : ℝ := (1/2) * (Real.exp x - Real.exp (-x))

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Icc 0 1, ∃ x2 ∈ Icc 0 1, f x1 = m * g x2) ↔ m ∈ Icc ((Real.exp 2 + 1) / (Real.exp 2 - 1)) ∞ :=
sorry

end range_of_m_l604_604804


namespace solve_for_x_l604_604289

theorem solve_for_x :
  (exists x, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
begin
  sorry
end

end solve_for_x_l604_604289


namespace ratio_of_segments_l604_604490

theorem ratio_of_segments 
  (A B C P Q R X Y Z : Point)
  (hP: P ∈ segment B C)
  (hQ: Q ∈ segment C A)
  (hR: R ∈ segment A B)
  (hAX : segment A P ∩ circumcircle (triangle A Q R) = {X})
  (hAY : segment A P ∩ circumcircle (triangle B R P) = {Y})
  (hAZ : segment A P ∩ circumcircle (triangle C P Q) = {Z}) 
  : ratio (segment Y X) (segment X Z) = ratio (segment B P) (segment P C) := 
sorry

end ratio_of_segments_l604_604490


namespace inscribed_circle_radius_integer_l604_604190

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l604_604190


namespace max_value_of_sqrt_expr_l604_604544

theorem max_value_of_sqrt_expr (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1) ≤ 3 * sqrt 3 :=
sorry

end max_value_of_sqrt_expr_l604_604544


namespace triangle_inequality_l604_604294

theorem triangle_inequality (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 := sorry

-- Given specific inputs
example : ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a + b > c ∧ a + c > b ∧ b + c > a :=
begin
  use [3, 4, 5],
  repeat {split},
  repeat {norm_num}
end

end triangle_inequality_l604_604294


namespace gloria_initial_cash_l604_604847

noncomputable def initial_cash (cypress_trees pine_trees maple_trees leftover cash_per_cypress cash_per_pine cash_per_maple cabin_cost) :=
  let total_tree_sales := (cypress_trees * cash_per_cypress) + (pine_trees * cash_per_pine) + (maple_trees * cash_per_maple)
  let total_before_buying := cabin_cost + leftover
  total_before_buying - total_tree_sales

theorem gloria_initial_cash :
  initial_cash 20 600 24 350 100 200 300 129000 = 150 :=
by
  unfold initial_cash
  sorry

end gloria_initial_cash_l604_604847


namespace roots_of_polynomial_l604_604765

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end roots_of_polynomial_l604_604765


namespace democrats_ratio_l604_604618

variable (F M D_F D_M TotalParticipants : ℕ)

-- Assume the following conditions
variables (H1 : F + M = 660)
variables (H2 : D_F = 1 / 2 * F)
variables (H3 : D_F = 110)
variables (H4 : D_M = 1 / 4 * M)
variables (H5 : TotalParticipants = 660)

theorem democrats_ratio 
  (H1 : F + M = 660)
  (H2 : D_F = 1 / 2 * F)
  (H3 : D_F = 110)
  (H4 : D_M = 1 / 4 * M)
  (H5 : TotalParticipants = 660) :
  (D_F + D_M) / TotalParticipants = 1 / 3
:= 
  sorry

end democrats_ratio_l604_604618


namespace quadratic_root_c_l604_604372

theorem quadratic_root_c (c : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + c = (x + (3/2))^2 - 7/4) → c = 1/2 :=
by
  sorry

end quadratic_root_c_l604_604372


namespace numbers_satisfy_conditions_l604_604391

noncomputable def find_numbers : List (ℝ × ℝ × ℝ) :=
  let solutions := [(8, 4, 2), (-6.4, 11.2, -19.6)] in
  solutions

theorem numbers_satisfy_conditions :
  ∀ (x y z : ℝ), 
  (x, y, z) ∈ find_numbers →
  (x / y = y / z) ∧
  (x - (y + z) = 2) ∧
  (x + (y - z) / 2 = 9) :=
by
  intros x y z h
  cases' h with
  | inl h =>
    rw [h]
    simp
  | inr h =>
    rw [h]
    simp
  sorry

end numbers_satisfy_conditions_l604_604391


namespace prob1_l604_604453

variable {V : Type _} [InnerProductSpace ℝ V]

variables (a b : V)
variable aba : a ≠ 0
variable abb : b ≠ 0

def a_mag : ℝ := 2
def b_mag : ℝ := Real.sqrt 3

axiom dot_eq : ((a + 2 • b) ⬝ (b - 3 • a)) = 9

def ab_dot : ℝ := -3
def bc_sq : ℝ := 13
def proj_ab_onto_ac : ℝ := -Real.sqrt 3

theorem prob1 : a • b = ab_dot ∧ norm (b - a) = Real.sqrt bc_sq ∧ (a • b) / (norm b) = proj_ab_onto_ac :=
by
  -- proofs go here
  sorry

end prob1_l604_604453


namespace _l604_604231

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

noncomputable theorem hyperbola_eccentricity (p a b : ℝ) (h1 : 0 < p) (h2 : 0 < a) (h3 : 0 < b)
  (h4 : a * b = 0 → False) (h5 : ∀ M : ℝ × ℝ, (M ∈ {M : ℝ × ℝ | (M.2) ^ 2 = 2 * p * (M.1)}) → 
    (M ∈ {M : ℝ × ℝ | ((M.1) ^ 2 / a ^ 2) - ((M.2) ^ 2 / b ^ 2) = 1}) → dist M (parabola_focus p) = p) :
  sqrt (1 + (b * b) / (a * a)) = 1 + sqrt 2 :=
sorry

end _l604_604231


namespace multiply_neg_reverse_inequality_l604_604742

theorem multiply_neg_reverse_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end multiply_neg_reverse_inequality_l604_604742


namespace determine_b_from_quadratic_l604_604022

theorem determine_b_from_quadratic (b n : ℝ) (h1 : b > 0) 
  (h2 : ∀ x, x^2 + b*x + 36 = (x + n)^2 + 20) : b = 8 := 
by 
  sorry

end determine_b_from_quadratic_l604_604022


namespace intersection_A_B_l604_604124

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l604_604124


namespace coin_toss_probability_l604_604641

theorem coin_toss_probability :
  let p : ℝ := 0.5 in
  let n : ℕ := 3 in
  let k : ℕ := 2 in
  (@nat.choose n k * p^k * (1 - p)^(n - k) = 0.375) :=
by
  sorry

end coin_toss_probability_l604_604641


namespace train_crosses_platform_in_39_seconds_l604_604691

-- Definitions of the given conditions
def length_of_train : ℝ := 300
def time_to_cross_signal_pole : ℝ := 20
def length_of_platform : ℝ := 285

-- Proof statement
theorem train_crosses_platform_in_39_seconds :
  let speed_of_train := length_of_train / time_to_cross_signal_pole in
  let total_distance_to_cover := length_of_train + length_of_platform in
  let time_to_cross_platform := total_distance_to_cover / speed_of_train in
  time_to_cross_platform = 39 := by {
  sorry
}

end train_crosses_platform_in_39_seconds_l604_604691


namespace min_max_F_l604_604824

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

def F (x y : ℝ) : ℝ := x^2 + y^2

theorem min_max_F (x y : ℝ) (h1 : f (y^2 - 6 * y + 11) + f (x^2 - 8 * x + 10) ≤ 0) (h2 : y ≥ 3) :
  ∃ (min_val max_val : ℝ), min_val = 13 ∧ max_val = 49 ∧
    min_val ≤ F x y ∧ F x y ≤ max_val :=
sorry

end min_max_F_l604_604824


namespace sum_length_AB_and_major_arc_l604_604160

def circle_radius : ℝ := 4
def circle_center : ℝ × ℝ := (0, 0)
def line_eq (x : ℝ) : ℝ := sqrt 3 * x - 4

def point_A : ℝ × ℝ := (0, -4)
def point_B : ℝ × ℝ := (2 * sqrt 3, 2)

noncomputable def length_segment_AB : ℝ := real.sqrt((2 * sqrt 3 - 0)^2 + (2 - (-4))^2)
noncomputable def angle_AOB : ℝ := 2 * real.pi / 3
noncomputable def circumference_circle : ℝ := 2 * real.pi * circle_radius
noncomputable def length_major_arc_AB : ℝ := circumference_circle - (circle_radius * angle_AOB)

theorem sum_length_AB_and_major_arc :
  (length_segment_AB + length_major_arc_AB) = (4 * sqrt 3 + 16 * real.pi / 3) :=
begin
  sorry
end

end sum_length_AB_and_major_arc_l604_604160


namespace intersection_of_sets_l604_604113

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l604_604113


namespace angle_PQR_eq_90_l604_604922

theorem angle_PQR_eq_90
  (R S P Q : Type)
  [IsStraightLine R S P]
  (angle_QSP : ℝ)
  (h : angle_QSP = 70) :
  ∠PQR = 90 :=
by
  sorry

end angle_PQR_eq_90_l604_604922


namespace common_point_collinear_l604_604139

theorem common_point_collinear 
  (ω1 ω2 ω3 : Circle)
  (O : Point)
  (x1 x2 x3 y1 y2 y3 : Real)
  (P : Point)
  (hω1 : passes_through ω1 O)
  (hω2 : passes_through ω2 O)
  (hω3 : passes_through ω3 O)
  (hx1 : intersects_at ω1 (x1, 0))
  (hx2 : intersects_at ω2 (x2, 0))
  (hx3 : intersects_at ω3 (x3, 0))
  (hy1 : intersects_at ω1 (0, y1))
  (hy2 : intersects_at ω2 (0, y2))
  (hy3 : intersects_at ω3 (0, y3))
  (not_tangent : ¬tangent ω1 ω2)
  (not_tangent : ¬tangent ω2 ω3)
  (not_tangent : ¬tangent ω1 ω3)
  (not_tangent_axis1 : ¬tangent_axis ω1)
  (not_tangent_axis2 : ¬tangent_axis ω2)
  (not_tangent_axis3 : ¬tangent_axis ω3) :
  (∃ P ≠ O, passes_through ω1 P ∧ passes_through ω2 P ∧ passes_through ω3 P) ↔ collinear ((x1, y1) :: (x2, y2) :: (x3, y3) :: []). 
  :=
sorry

end common_point_collinear_l604_604139


namespace f_continuous_l604_604317

open Set Filter TopologicalSpace

variable (K : Set (ℝ × ℝ × ℝ))
variable [CompactSpace K]

def f (p : ℝ × ℝ × ℝ) : ℝ := inf { dist p k | k ∈ K }

theorem f_continuous (K : Set (ℝ × ℝ × ℝ)) [CompactSpace K] : Continuous (f K) :=
by
  sorry

end f_continuous_l604_604317


namespace simplify_pow_prod_eq_l604_604990

noncomputable def simplify_pow_prod : ℝ :=
  (256:ℝ)^(1/4) * (625:ℝ)^(1/2)

theorem simplify_pow_prod_eq :
  simplify_pow_prod = 100 := by
  sorry

end simplify_pow_prod_eq_l604_604990


namespace set_intersection_l604_604092

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l604_604092


namespace maria_total_payment_l604_604300

theorem maria_total_payment
  (original_price discount : ℝ)
  (discount_percentage : ℝ := 0.25)
  (discount_amount : ℝ := 40) 
  (total_paid : ℝ := 120) :
  discount_amount = discount_percentage * original_price →
  total_paid = original_price - discount_amount → 
  total_paid = 120 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end maria_total_payment_l604_604300


namespace distance_traveled_l604_604695

theorem distance_traveled :
  ∫ t in (3:ℝ)..(5:ℝ), (2 * t + 3 : ℝ) = 22 :=
by
  sorry

end distance_traveled_l604_604695


namespace proof_statements_l604_604796

variable {a b c : ℝ}
variable (h1 : a ≠ 0)

-- Condition ①
def cond1 (h : a - b + c = 0) : Prop :=
  b^2 - 4*a*c ≥ 0

-- Condition ②
def cond2 (h : a*(1:ℝ)*1 + b*(1:ℝ) + c = 0 ∧ a*(2:ℝ)*2 + b*(2:ℝ) + c = 0) : Prop :=
  2*a - c = 0

-- Condition ③
def cond3 (h : -4*a*c > 0) : Prop :=
  b^2 - 4*a*c ≥ 0

-- Condition ④
def cond4 (h : b = 2*a + c) : Prop :=
  b^2 - 4*a*c > 0

theorem proof_statements : 
  (cond1 (a - b + c) ∧
  cond2 (a*(1:ℝ)*1 + b*(1:ℝ) + c = 0 ∧ a*(2:ℝ)*2 + b*(2:ℝ) + c = 0) ∧
  cond3 (a ≠ 0 ∧ -4*a*c > 0) ∧ 
  cond4 (b = 2*a + c)) := sorry

end proof_statements_l604_604796


namespace point_M_construction_l604_604525

-- Define points and their properties
variables {Point : Type} [euclidean_geometry Point]
variables (A B C D E F G M : Point)

-- Conditions
axiom squares : square A B C D ∧ square E C G F
axiom collinear : collinear [B, C, G]
axiom same_side : ∃ l : line, ¬A ∈ l ∧ ¬D ∈ l ∧ ¬E ∈ l ∧ ¬F ∈ l ∧ A ≠ D ∧ E ≠ F ∧ l.has_point B ∧ l.has_point C

-- Question (to be proved): M is the intersection of lines (AF) and (BE)
theorem point_M_construction :
  is_intersection (line_through A F) (line_through B E) M := sorry

end point_M_construction_l604_604525


namespace problem_I_problem_II_l604_604699

def houses := ({4, 5, 5, 6, 6} : finset ℕ)

def families (A B : ℕ) : Prop :=
  A ∈ houses ∧ B ∈ houses ∧ A ≠ B

noncomputable def same_floor_probability : ℚ :=
  if h : ∃ (A B : ℕ), families A B ∧ (A = B) then 1 / 5 else 0

noncomputable def adjacent_floor_probability : ℚ :=
  if h : ∃ (A B : ℕ), families A B ∧ (A = B + 1 ∨ A + 1 = B) then 3 / 5 else 0

theorem problem_I : same_floor_probability = 1 / 5 := by
  sorry

theorem problem_II : adjacent_floor_probability = 3 / 5 := by
  sorry

end problem_I_problem_II_l604_604699


namespace same_color_probability_l604_604867

open Nat

theorem same_color_probability :
  let total_plates := 11
  let red_plates := 6
  let blue_plates := 5
  let chosen_plates := 3
  let total_ways := choose total_plates chosen_plates
  let red_ways := choose red_plates chosen_plates
  let blue_ways := choose blue_plates chosen_plates
  let same_color_ways := red_ways + blue_ways
  let probability := (same_color_ways : ℚ) / (total_ways : ℚ)
  probability = 2 / 11 := by sorry

end same_color_probability_l604_604867


namespace eval_expression_l604_604961

def f (x : ℝ) : ℝ := x - 1
def g (x : ℝ) : ℝ := 2 * x
def f_inv (x : ℝ) : ℝ := x + 1
def g_inv (x : ℝ) : ℝ := x / 2

theorem eval_expression : f (g_inv (f_inv (f_inv (g (f 10))))) = 9 := 
by sorry

end eval_expression_l604_604961


namespace minimum_value_of_a_l604_604437

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x < 1 then -x + a else x^2

theorem minimum_value_of_a (a : ℝ) : f a (f a (-2)) = 16 ↔ a = 2 :=
sorry

end minimum_value_of_a_l604_604437


namespace trigonometric_functions_unchanged_l604_604896

theorem trigonometric_functions_unchanged
  (o a h : ℝ) (h_positive : o > 0 ∧ a > 0 ∧ h > 0)
  (right_triangle : o^2 + a^2 = h^2) :
  let sin_A := o / h
  let cos_A := a / h
  let tan_A := o / a
  let sin_A' := (3 * o) / (3 * h)
  let cos_A' := (3 * a) / (3 * h)
  let tan_A' := (3 * o) / (3 * a)
  in
  sin_A = sin_A' ∧ cos_A = cos_A' ∧ tan_A = tan_A' :=
by sorry

end trigonometric_functions_unchanged_l604_604896


namespace triangle_area_solutions_count_l604_604026

theorem triangle_area_solutions_count :
  (∃ θ₁ θ₂ θ₃ θ₄ ∈ ℝ, 
     θ₁ ≠ θ₂ ∧ θ₁ ≠ θ₃ ∧ θ₁ ≠ θ₄ ∧ θ₂ ≠ θ₃ ∧ θ₂ ≠ θ₄ ∧ θ₃ ≠ θ₄ ∧ 
     (5 * sin θ₁ = 2 ∧ 5 * sin θ₂ = 2 ∧ 5 * sin θ₃ = 2 ∧ 5 * sin θ₄ = 2)) := 
begin
  sorry
end

end triangle_area_solutions_count_l604_604026


namespace vec_parallel_l604_604017

variable {R : Type*} [LinearOrderedField R]

def is_parallel (a b : R × R) : Prop :=
  ∃ k : R, a = (k * b.1, k * b.2)

theorem vec_parallel {x : R} : 
  is_parallel (1, x) (-3, 4) ↔ x = -4/3 := by
  sorry

end vec_parallel_l604_604017


namespace angle_PQR_correct_l604_604919

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end angle_PQR_correct_l604_604919


namespace tan_2017pi_minus_2a_eq_neg2_l604_604020

theorem tan_2017pi_minus_2a_eq_neg2
  (a : ℝ)
  (ha : (sin a, 1) = (2 * cos a * cos a - 1, cos a)) :
  tan (2017 * π - 2 * a) = -2 := sorry

end tan_2017pi_minus_2a_eq_neg2_l604_604020


namespace intersection_A_B_l604_604116

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l604_604116


namespace distance_between_city_A_and_city_B_l604_604757

noncomputable def eddyTravelTime : ℝ := 3  -- hours
noncomputable def freddyTravelTime : ℝ := 4  -- hours
noncomputable def constantDistance : ℝ := 300  -- km
noncomputable def speedRatio : ℝ := 2  -- Eddy:Freddy

theorem distance_between_city_A_and_city_B (D_B D_C : ℝ) (h1 : D_B = (3 / 2) * D_C) (h2 : D_C = 300) :
  D_B = 450 :=
by
  sorry

end distance_between_city_A_and_city_B_l604_604757


namespace eight_people_permutations_l604_604910

theorem eight_people_permutations : (finset.univ : finset (fin 8)).card.factorial = 40320 := sorry

end eight_people_permutations_l604_604910


namespace volume_PQRS_is_48_39_cm3_l604_604482

noncomputable def area_of_triangle (a h : ℝ) : ℝ := 0.5 * a * h

noncomputable def volume_of_tetrahedron (base_area height : ℝ) : ℝ := (1/3) * base_area * height

noncomputable def height_from_area (area base : ℝ) : ℝ := (2 * area) / base

noncomputable def volume_of_tetrahedron_PQRS : ℝ :=
  let PQ := 5
  let area_PQR := 18
  let area_PQS := 16
  let angle_PQ := 45
  let h_PQR := height_from_area area_PQR PQ
  let h_PQS := height_from_area area_PQS PQ
  let h := h_PQS * (Real.sin (angle_PQ * Real.pi / 180))
  volume_of_tetrahedron area_PQR h

theorem volume_PQRS_is_48_39_cm3 : volume_of_tetrahedron_PQRS = 48.39 := by
  sorry

end volume_PQRS_is_48_39_cm3_l604_604482


namespace limit_proof_l604_604168

theorem limit_proof :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → | (3 * x^2 - 5 * x - 2) / (x - 2) - 7 | < ε) :=
begin
  assume ε ε_pos,
  existsi ε / 3,
  use div_pos ε_pos (by norm_num),
  assume x hx,
  have h_denom : x ≠ 2, from λ h, hx.1 (by rwa h),
  calc
  |(3 * x^2 - 5 * x - 2) / (x - 2) - 7|
      = |((3 * x + 1) * (x - 2) / (x - 2)) - 7| : by {
        ring_exp,
        exact (by norm_num : (3 : ℝ) ≠ 0 ),
      }
  ... = |3 * x + 1 - 7| : by rw [(by_ring_exp * (x - 2)).symm], (ne_of_apply_ne (3 * x + 1) (by norm_num)).elim]
  ... = |3 * (x - 2)| : by ring_exp
  ... = 3 * |x - 2| : abs_mul,
  exact (mul_lt_iff_lt_one_left (by norm_num)).2 (calc
      |x - 2| < ε / 3 : hx.2
      ... = ε / 3
    )
end

end limit_proof_l604_604168


namespace find_k_l604_604308

-- Definitions
variable (m n k : ℝ)

-- Given conditions
def on_line_1 : Prop := m = 2 * n + 5
def on_line_2 : Prop := (m + 5) = 2 * (n + k) + 5

-- Desired conclusion
theorem find_k (h1 : on_line_1 m n) (h2 : on_line_2 m n k) : k = 2.5 :=
sorry

end find_k_l604_604308


namespace probability_same_color_is_correct_l604_604864

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l604_604864


namespace find_noon_temperature_l604_604357

theorem find_noon_temperature (T T₄₀₀ T₈₀₀ : ℝ) 
  (h1 : T₄₀₀ = T + 8)
  (h2 : T₈₀₀ = T₄₀₀ - 11)
  (h3 : T₈₀₀ = T + 1) : 
  T = 4 :=
by
  sorry

end find_noon_temperature_l604_604357


namespace total_amount_paid_l604_604532

-- Define the conditions
def chicken_nuggets_ordered : ℕ := 100
def nuggets_per_box : ℕ := 20
def cost_per_box : ℕ := 4

-- Define the hypothesis on the amount of money paid for the chicken nuggets
theorem total_amount_paid :
  (chicken_nuggets_ordered / nuggets_per_box) * cost_per_box = 20 :=
by
  sorry

end total_amount_paid_l604_604532


namespace cube_root_of_sqrt_64_l604_604224

theorem cube_root_of_sqrt_64 : real.sqrt 64 ^ (1 / 3 : ℝ) = 2 := by
  sorry

end cube_root_of_sqrt_64_l604_604224


namespace tom_and_eva_children_count_l604_604503

theorem tom_and_eva_children_count (karen_donald_children : ℕ)
  (total_legs_in_pool : ℕ) (people_not_in_pool : ℕ) 
  (total_legs_each_person : ℕ) (karen_donald : ℕ) (tom_eva : ℕ) 
  (total_people_in_pool : ℕ) (total_people : ℕ) :
  karen_donald_children = 6 ∧ total_legs_in_pool = 16 ∧ people_not_in_pool = 6 ∧ total_legs_each_person = 2 ∧
  karen_donald = 2 ∧ tom_eva = 2 ∧ total_people_in_pool = total_legs_in_pool / total_legs_each_person ∧ 
  total_people = total_people_in_pool + people_not_in_pool ∧ 
  total_people - (karen_donald + karen_donald_children + tom_eva) = 4 :=
by
  intros
  sorry

end tom_and_eva_children_count_l604_604503


namespace probability_same_color_is_correct_l604_604856

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l604_604856


namespace cos_greater_than_sin_l604_604743

theorem cos_greater_than_sin :
  cos (3 * Real.pi / 14) > sin (-15 * Real.pi / 8) :=
by
  sorry

end cos_greater_than_sin_l604_604743


namespace polygon_sides_l604_604042

theorem polygon_sides (h : ∑ interior_sum = 1440) : nat :=
  let n := (1440 / 180) + 2
  in if n = 10 
    then valid sides
    else invalid

end polygon_sides_l604_604042


namespace radius_of_inscribed_circle_is_integer_l604_604182

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l604_604182


namespace range_of_f_l604_604775

-- Given condition: a + b = 2
def condition (a b : ℝ) : Prop := a + b = 2

-- Function to analyze
def f (a b : ℝ) : ℝ := (1 / a) + (4 / b)

-- The range of our function
def range_f : Set ℝ := Set.Iic (1/2) ∪ Set.Ici (9/2)

-- Statement of the problem to be proved
theorem range_of_f (a b : ℝ) (h : condition a b) : f a b ∈ range_f :=
sorry

end range_of_f_l604_604775


namespace distinct_floors_l604_604131

-- Define the floor function
def floor (x : ℝ) : ℕ := Int.toNat (Real.floor x)

-- Define the main problem statement
theorem distinct_floors (s : Finset ℕ) :
  s = (Finset.image (λ n : ℕ, floor ((n^2 : ℚ) / 2005)) (Finset.range 2006)) →
  s.card = 1503 :=
begin
  intro h,
  rw h,
  sorry
end

end distinct_floors_l604_604131


namespace sin_half_angles_iff_sides_equal_l604_604936

theorem sin_half_angles_iff_sides_equal
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + B + C = π)
  (h2 : sin (A / 2) * sin (B / 2) * sin (C / 2) = 1 / 8) 
  (ha : a = 2 * sin (A / 2) / sin (B / 2) * sin (C / 2))
  (hb : b = 2 * sin (B / 2) / sin (A / 2) * sin (C / 2))
  (hc : c = 2 * sin (C / 2) / sin (A / 2) * sin (B / 2)) :
  a = b = c ↔ A = B = C :=
sorry

end sin_half_angles_iff_sides_equal_l604_604936


namespace angle_PQR_correct_l604_604920

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end angle_PQR_correct_l604_604920


namespace paper_stars_per_bottle_l604_604320

theorem paper_stars_per_bottle (a b total_bottles : ℕ) (h1 : a = 33) (h2 : b = 307) (h3 : total_bottles = 4) :
  (a + b) / total_bottles = 85 :=
by
  sorry

end paper_stars_per_bottle_l604_604320


namespace fraction_equivalent_to_repeating_decimal_l604_604635

noncomputable def repeating_decimal := 0.3 + 25 / 990

theorem fraction_equivalent_to_repeating_decimal : 
  repeating_decimal = (161 : ℚ) / 495 := sorry

end fraction_equivalent_to_repeating_decimal_l604_604635


namespace solve_for_x_l604_604290

theorem solve_for_x :
  (exists x, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
begin
  sorry
end

end solve_for_x_l604_604290


namespace car_wash_cost_l604_604499

-- Definitions based on the conditions
def washes_per_bottle : ℕ := 4
def bottle_cost : ℕ := 4   -- Assuming cost is recorded in dollars
def total_weeks : ℕ := 20

-- Stating the problem
theorem car_wash_cost : (total_weeks / washes_per_bottle) * bottle_cost = 20 := 
by
  -- Placeholder for the proof
  sorry

end car_wash_cost_l604_604499


namespace find_a_l604_604522

def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 2 else 3 * x - 50

theorem find_a (a : ℝ) (ha : a < 0) : a = -5 ↔ g(g(g(15))) = g(g(g(a))) :=
by
  have h1 : g 15 = 3 * 15 - 50 := by norm_num
  have h2 : g (3 * 15 - 50) = 7 := by norm_num
  have h3 : g 7 = -29 := by norm_num
  have h4 : g(g(g(15))) = -29 := by rw [h1, h2, h3]
  split
  case mp =>
    intro h
    rw [h]
    rw [h4]
    sorry
  case mpr =>
    intro eq
    have := eq.symm
    sorry

end find_a_l604_604522


namespace james_sells_over_market_value_l604_604080

theorem james_sells_over_market_value 
    (house_market_value : ℝ) 
    (people : ℤ) 
    (amount_each_after_taxes : ℝ) 
    (tax_rate : ℝ) 
    (split : ℝ) 
    (num_people : people = 4) 
    (market_value : house_market_value = 500000) 
    (revenue_after_taxes : amount_each_after_taxes * people = 540000) 
    (tax_cut : tax_rate = 0.1) 
    (each_amount : amount_each_after_taxes = 135000) 
    : split = 20 := 
by
  have value_before_taxes := 540000 / (1 - tax_rate)
  have split_calc := (value_before_taxes - house_market_value) / house_market_value * 100
  exact calc 
    split = split_calc : sorry
    ... = 20 : sorry

end james_sells_over_market_value_l604_604080


namespace cos_graph_shift_l604_604267

theorem cos_graph_shift (x : ℝ) :
    cos (2 * (x - π / 8)) = cos (2 * x - π / 4) :=
by
  sorry

end cos_graph_shift_l604_604267


namespace bianca_picked_roses_l604_604725

def number_of_tulips : ℕ := 39
def flowers_used : ℕ := 81
def extra_flowers : ℕ := 7
def total_flowers : ℕ := flowers_used + extra_flowers
def number_of_roses : ℕ := total_flowers - number_of_tulips

theorem bianca_picked_roses (t u e f r : ℕ) 
  (ht : t = number_of_tulips)
  (hu : u = flowers_used)
  (he : e = extra_flowers)
  (hf : f = total_flowers)
  (hr : r = number_of_roses) : 
  r = 49 :=
by {
  rw [ht, hu, he, hf, hr],
  unfold total_flowers number_of_tulips number_of_roses,
  simp,
  exact nat.sub_eq_iff_eq_add.mpr rfl,
  sorry
}

end bianca_picked_roses_l604_604725


namespace calculate_f_5_l604_604440

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem calculate_f_5 : f 5 = 4485 := 
by {
  -- The proof of the theorem will go here, using the Horner's method as described.
  sorry
}

end calculate_f_5_l604_604440


namespace find_a_squared_div_b_l604_604519

noncomputable def isosceles_triangle (z1 z2 : ℂ) : Prop :=
  ∥z1∥ = ∥2 * z2∥

theorem find_a_squared_div_b (z1 z2 a b : ℂ) (h1 : isosceles_triangle z1 z2) (h2 : z1 + z2 = -a) (h3 : z1 * z2 = b) : 
    a^2 / b = 4.5 := 
  sorry

end find_a_squared_div_b_l604_604519


namespace measure_angle_ABD_regular_pentagon_l604_604165

noncomputable def interior_angle (n : ℕ) : ℝ := ((n - 2) * 180) / n

theorem measure_angle_ABD_regular_pentagon
  (ABCDE : Type)
  [pentagon : is_regular_pentagon ABCDE]
  : angle ABD = 36 := by
  sorry

end measure_angle_ABD_regular_pentagon_l604_604165


namespace radius_of_inscribed_circle_is_integer_l604_604187

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l604_604187


namespace left_handed_rock_lovers_l604_604904

theorem left_handed_rock_lovers (total_people left_handed rock_music right_dislike_rock x : ℕ) :
  total_people = 30 →
  left_handed = 14 →
  rock_music = 20 →
  right_dislike_rock = 5 →
  (x + (left_handed - x) + (rock_music - x) + right_dislike_rock = total_people) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end left_handed_rock_lovers_l604_604904


namespace option_D_is_mapping_l604_604293

definition is_mapping (A B : Set ℝ) (f : ℝ → ℝ) := ∀ (x : ℝ), x ∈ A → f x ∈ B

theorem option_D_is_mapping : is_mapping (Set.univ) (Set.univ) (λ x, -x + 1) := sorry

end option_D_is_mapping_l604_604293


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604196

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604196


namespace area_intersection_M_N_l604_604838

open Complex Real

def M : set ℂ := {z | abs (z - 1) ≤ 1}
def N : set ℂ := {z | arg z ≥ π / 4}

theorem area_intersection_M_N :
  let S := area (M ∩ N)
  in S = (3 / 4) * π - 1 / 2 :=
sorry

end area_intersection_M_N_l604_604838


namespace part1_part2_l604_604004

-- Define the function f and its properties
def f (x : ℝ) (a : ℝ) : ℝ := a * real.log x

-- The tangent line condition in Part (1)
theorem part1 (a : ℝ) (h : 0 < a) : 
  (exists x0 : ℝ, x0 > 0 ∧
   f x0 a = (x0 / real.exp a) + real.exp 2 ∧
   (f' (x0) a = (1 / real.exp a))) 
   ↔ (a = real.exp 1) :=
sorry

-- The inequality condition in Part (2)
theorem part2 (a : ℝ) (h : 0 < a) :
  (∀ m n : ℝ, m ≠ n → 0 < m → 0 < n →
   (real.sqrt (m * n) + (m + n) / 2 > (m - n) / (f m a - f n a))) 
   ↔ (1 / 2 ≤ a) :=
sorry

end part1_part2_l604_604004


namespace maximize_possible_value_l604_604241

open Finset

noncomputable def max_value {a b c d : ℕ} (h : {a, b, c, d} = {1, 3, 5, 7}) : ℕ :=
  max ((a + b) * (c + d) + (a + 1) * (d + 1))
      ((a + c) * (b + d) + (a + 1) * (d + 1))
-- Add all possible permutations of the terms for the max_value function,
-- this line would theoretically cover all combinations,
-- omitted here for conciseness.

theorem maximize_possible_value : ∃ (a b c d : ℕ), 
  {a, b, c, d} = {1, 3, 5, 7} ∧ max_value ({a, b, c, d} = {1, 3, 5, 7}) = 112 :=
begin
  sorry
end

end maximize_possible_value_l604_604241


namespace number_of_periods_dividing_hour_l604_604755

theorem number_of_periods_dividing_hour :
  ∃ (n m : ℕ), n * m = 3600 ∧ (45 = (∑ d in (finset.divisors 3600).to_list, if d = 60 then 1 else 1 / 2)) :=
by
  sorry

end number_of_periods_dividing_hour_l604_604755


namespace inscribed_circle_radius_integer_l604_604189

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l604_604189


namespace cos_theta_geom_prog_l604_604959

theorem cos_theta_geom_prog (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2)
  (hp : ∃ (r : ℝ), (sin θ = r * sin (2 * θ)) ∨ (sin (2 * θ) = r * sin (4 * θ)) ∨ (sin θ = r * sin (4 * θ))) :
  cos θ = 3 / 4 :=
by
  sorry

end cos_theta_geom_prog_l604_604959


namespace cube_root_of_sqrt_64_l604_604222

theorem cube_root_of_sqrt_64 : real.sqrt 64 ^ (1 / 3 : ℝ) = 2 := by
  sorry

end cube_root_of_sqrt_64_l604_604222


namespace contradiction_method_example_l604_604273

theorem contradiction_method_example (x y : ℝ) (h : x > y) : x^3 > y^3 :=
by
  intro h1 : x^3 <= y^3
  sorry

end contradiction_method_example_l604_604273


namespace total_seashells_l604_604153

theorem total_seashells (m : ℕ) (j : ℕ) (h_m : m = 18) (h_j : j = 41) : m + j = 59 := 
by
  rw [h_m, h_j]
  exact eq.refl 59

end total_seashells_l604_604153


namespace common_area_of_congruent_triangles_l604_604629

def area_common (hypotenuse : ℝ) : ℝ :=
  let short_leg := hypotenuse / 2
  let long_leg := short_leg * (Real.sqrt 3)
  let area := (1 / 2) * short_leg * long_leg
  area

theorem common_area_of_congruent_triangles {hypotenuse : ℝ} (h : hypotenuse = 10) :
  area_common hypotenuse = (25 * Real.sqrt 3) / 2 :=
by
  sorry

end common_area_of_congruent_triangles_l604_604629


namespace ratio_is_47_to_10_l604_604158

variable (time_taken x : ℕ)

def total_time_wasted (time_taken x : ℕ) : ℕ := time_taken + time_taken * x + 14

theorem ratio_is_47_to_10
  (h1 : time_taken = 20)
  (h2 : total_time_wasted time_taken x = 114)
  : (time_taken * x + 14) = 94  ∧ (94 / 20) = 47 / 10 :=
by
  have h3 : 20 + 20 * 4 + 14 = 114 := by
    calc
      20 + 20 * 4 + 14 = 20 + 80 + 14 := by rfl
      ... = 114 := by rfl
  have h4 : 80 + 14 = 94 := by
    calc
      80 + 14 = 94 := by rfl
  have h5 : 94 / 20 = 47 / 10 := by
    calc
      94 / 20 = 4.7 := by rfl
      ... = 47 / 10 := by rfl
  sorry

end ratio_is_47_to_10_l604_604158


namespace solve_for_n_remainder_of_n4_l604_604875

theorem solve_for_n (n : ℤ) (h : 4 * n + 3 ≡ 0 [MOD 11]) : n ≡ 2 [MOD 11] :=
sorry

theorem remainder_of_n4 (n : ℤ) (h : n ≡ 2 [MOD 11]) : n^4 ≡ 5 [MOD 11] :=
sorry

end solve_for_n_remainder_of_n4_l604_604875


namespace limit_s_eq_one_third_l604_604954

noncomputable def Q (m : ℝ) : ℝ :=
  (root_of (λ x => x^3 - 6 * x - m, leftmost))

noncomputable def Q_neg (m : ℝ) : ℝ :=
  (root_of (λ x => x^3 - 6 * x + m, leftmost))

theorem limit_s_eq_one_third (Q Q_neg : ℝ → ℝ)
  (hQ : ∀ m, Q m = (classical.some (root_of (λ x => x^3 - 6 * x - m, leftmost))))
  (hQ_neg : ∀ m, Q_neg m = (classical.some (root_of (λ x => x^3 - 6 * x + m, leftmost)))) :
  tendsto (λ m => (Q_neg(-m) - Q(m)) / m) (nhds 0) (nhds (1/3)) := 
sorry

end limit_s_eq_one_third_l604_604954


namespace college_student_ticket_cost_l604_604063

theorem college_student_ticket_cost 
    (total_visitors : ℕ)
    (nyc_residents: ℕ)
    (college_students_nyc: ℕ)
    (total_money_received : ℕ) :
    total_visitors = 200 →
    nyc_residents = total_visitors / 2 →
    college_students_nyc = (nyc_residents * 30) / 100 →
    total_money_received = 120 →
    (total_money_received / college_students_nyc) = 4 := 
sorry

end college_student_ticket_cost_l604_604063


namespace train_speed_kmh_l604_604343

variables (train_length bridge_length : ℕ) (time_to_pass_bridge : ℝ)

theorem train_speed_kmh : 
  train_length = 360 ∧ bridge_length = 160 ∧ time_to_pass_bridge = 41.6 → 
  (train_length + bridge_length) / time_to_pass_bridge * 3.6 = 45 := 
by
  intros
  cases a with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  sorry

end train_speed_kmh_l604_604343


namespace problem1_problem2_l604_604314

-- Problem 1
theorem problem1 :
  (1 / 3) ^ (-2 : ℤ) - (π - real.sqrt 5) ^ (0 : ℤ) + real.abs (real.sqrt 3 - 2) + 4 * real.sin (real.pi / 3) = 9 + 2 * real.sqrt 3 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : x ≠ -3) :
  (x^2 - 6*x + 9) / (x + 3) / (1 - 6 / (x + 3)) = x - 3 :=
sorry

end problem1_problem2_l604_604314


namespace remainder_of_3_pow_90_plus_5_mod_7_l604_604639

theorem remainder_of_3_pow_90_plus_5_mod_7 : (3^90 + 5) % 7 = 6 := by
  have h1 : 3 % 7 = 3 := rfl
  have h2 : (3^2) % 7 = 2 := by norm_num
  have h3 : (3^3) % 7 = 6 := by norm_num
  have h4 : (3^4) % 7 = 4 := by norm_num
  have h5 : (3^5) % 7 = 5 := by norm_num
  have h6 : (3^6) % 7 = 1 := by norm_num
  sorry

end remainder_of_3_pow_90_plus_5_mod_7_l604_604639


namespace points_ABC_cannot_form_triangle_k_collinear_l604_604575

noncomputable theory

-- Defining the vectors and their conditions
variables {ℝ : Type} [OrderedRing ℝ] 
variables (a b : ℝ) 
variables (A B C D : Type) 

-- Conditions: a and b are non-zero and not collinear
variables (ha : a ≠ 0) (hb : b ≠ 0) (h_not_collinear : ∀ k : ℝ, k ≠ 0 → a ≠ k * b)

-- Defining vector operations for directions
variables {AB BC CD BD : ℝ} 
variables (h_AB : AB = a + b) 
variables (h_BC : BC = 2 * a + 8 * b) 
variables (h_CD : CD = 3 * (a - b)) 

-- Problem statement
theorem points_ABC_cannot_form_triangle (A B D : Type) : ¬ collinear {A, B, D} :=
sorry

theorem k_collinear (k : ℝ) (a b : ℝ) : k * a + b = (a + k * b) ↔ k = 1 ∨ k = -1 :=
sorry

end points_ABC_cannot_form_triangle_k_collinear_l604_604575


namespace tabs_in_all_browsers_l604_604938

-- Definitions based on conditions
def windows_per_browser := 3
def tabs_per_window := 10
def number_of_browsers := 2

-- Total tabs calculation
def total_tabs := number_of_browsers * (windows_per_browser * tabs_per_window)

-- Proving the total number of tabs is 60
theorem tabs_in_all_browsers : total_tabs = 60 := by
  sorry

end tabs_in_all_browsers_l604_604938


namespace problem_solution_1_problem_solution_2_problem_solution_3_problem_solution_4_l604_604545

open Nat

def nth_row_last_number (n : ℕ) : ℕ :=
  2^n - 1

def nth_row_sum (n : ℕ) : ℕ :=
  3 * 2^(2*n - 3) - 2^(n - 2)

def find_row_position (num : ℕ) : ℕ × ℕ :=
  let n := Nat.find (λ n, 2^(n + 1) - 1 ≥ num)
  let last_num_prev_row := 2^n - 1
  (n + 1, num - last_num_prev_row)

noncomputable def row_sum_in_range (n : ℕ) : ℕ :=
  let S := λ k, 3 * 2^(2*k - 3) - 2^(k - 2)
  (Finset.range (10 : ℕ)).sum (λ i, S (n + i))

theorem problem_solution_1 (n : ℕ) : nth_row_last_number n = 2^n - 1 := sorry

theorem problem_solution_2 (n : ℕ) : nth_row_sum n = 3 * 2^(2*n - 3) - 2^(n - 2) := sorry

theorem problem_solution_3 : find_row_position 2010 = (12, 987) := sorry

theorem problem_solution_4 : ∃ n : ℕ, row_sum_in_range n = 2^27 - 2^13 - 120 := ⟨5, sorry⟩

end problem_solution_1_problem_solution_2_problem_solution_3_problem_solution_4_l604_604545


namespace paperback_count_l604_604617

theorem paperback_count (H P : ℕ) (total_books : Finset ℕ) (hb : Finset.card total_books = 6) 
  (h_hardbacks : H = 4) 
  (h_possible_selections : ∃ (s : Finset (Finset ℕ)), 
    Finset.card s = 14 ∧ ∀ t ∈ s, Finset.card t = 4 ∧ (t ∩ (Finset.range P)).nonempty) : 
  P = 2 :=
by
  sorry

end paperback_count_l604_604617


namespace find_e_l604_604945

-- Define values for a, b, c, d
def a := 2
def b := 3
def c := 4
def d := 5

-- State the problem
theorem find_e (e : ℚ) : a + b + c + d + e = a + (b + (c - (d * e))) → e = -5/6 :=
by
  sorry

end find_e_l604_604945


namespace solve_for_x_l604_604284

theorem solve_for_x : (∃ x : ℚ, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
by
  use 80 / 3
  sorry

end solve_for_x_l604_604284


namespace general_term_formula_and_sum_of_sequence_sequence_b_not_geometric_l604_604432

-- Definitions based on conditions
def sum_a (a : ℕ → ℕ) (n : ℕ) : ℕ := a 5 + a 6 = 24
def sum_Sn_11 (S : ℕ → ℕ) : Prop := S 11 = 143
def sum_Tn (T : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → 2^(a n - 1) = T n - a 1

-- Main problems
theorem general_term_formula_and_sum_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) :
  sum_Sn_11 S →
  sum_a a 11 →
  sum_Tn T a →
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, (∑ i in finset.range n, 1 / (a i * a (i + 1))) = n / (6 * n + 9)) := sorry

theorem sequence_b_not_geometric (a : ℕ → ℕ) (T : ℕ → ℕ) (b : ℕ → ℕ) :
  sum_Tn T a →
  (∀ n : ℕ, (n > 0) → 2^(a n - 1) = T n - a 1) →
  ¬ ∀ n : ℕ, b (n + 1) = 4 * b n := sorry

end general_term_formula_and_sum_of_sequence_sequence_b_not_geometric_l604_604432


namespace transformation_result_l604_604272

noncomputable def rotate_and_dilate (z : ℂ) : ℂ :=
  let rotation := (1/2 : ℝ) + ((Real.sqrt 3 / 2) : ℝ) * Complex.I
  let dilation := 2 : ℂ
  z * (rotation * dilation)

theorem transformation_result : rotate_and_dilate (1 + 3 * Complex.I) = 1 - 3 * Real.sqrt 3 + (3 + Real.sqrt 3) * Complex.I :=
by 
  sorry

end transformation_result_l604_604272


namespace min_max_of_f_l604_604007

def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem min_max_of_f :
  (∀ x, 3 ≤ x ∧ x ≤ 4 → f 4 ≤ f x) ∧ (∀ x, 3 ≤ x ∧ x ≤ 4 → f x ≤ f 3) :=
by {
  sorry
}

end min_max_of_f_l604_604007


namespace find_M_N_sum_l604_604029

theorem find_M_N_sum
  (M N : ℕ)
  (h1 : 3 * 75 = 5 * M)
  (h2 : 3 * N = 5 * 90) :
  M + N = 195 := 
sorry

end find_M_N_sum_l604_604029


namespace parabola_coefficients_l604_604576

theorem parabola_coefficients (a b c : ℝ) 
  (h_vertex : ∀ x, a * (x - 4) * (x - 4) + 3 = a * x * x + b * x + c) 
  (h_pass_point : 1 = a * (2 - 4) * (2 - 4) + 3) :
  (a = -1/2) ∧ (b = 4) ∧ (c = -5) :=
by
  sorry

end parabola_coefficients_l604_604576


namespace Amos_finishes_book_on_Friday_l604_604353

theorem Amos_finishes_book_on_Friday :
  ∃ (n : ℕ), n = 5 ∧
  (∑ i in finset.range n, (40 + 20 * i)) = 400 := by
sorry

end Amos_finishes_book_on_Friday_l604_604353


namespace find_fraction_result_l604_604145

open Complex

theorem find_fraction_result (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
    (h1 : x + y + z = 30)
    (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 33 := 
    sorry

end find_fraction_result_l604_604145


namespace dice_sum_divisible_by_3_l604_604392

theorem dice_sum_divisible_by_3 (d1 d2 d3 d4 d5 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6)
  (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6)
  (h5 : 1 ≤ d5 ∧ d5 ≤ 6) (heven : (d1 * d2 * d3 * d4 * d5) % 2 = 0) :
  let outcomes := 6 ^ 5 - 3 ^ 5 in
  let favorable := (6 ^ 5 - 3 ^ 5) / 3 in
  favorable / outcomes = 1 / 3 := sorry

end dice_sum_divisible_by_3_l604_604392


namespace intersection_A_B_eq_C_l604_604102

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l604_604102


namespace sheets_in_stack_l604_604332

theorem sheets_in_stack (n_sheets : ℕ) (thickness_pack : ℕ) (h_stack : ℕ) (thickness_one_sheet : ℝ) :
  n_sheets = 1200 :=
by
  have pack_thickness := 4 -- pack thickness in cm
  have total_sheets := 800 -- number of sheets in a pack
  have stack_height := 6 -- stack height in cm
  have thickness_sheet := (pack_thickness : ℝ) / (total_sheets : ℝ)
  have result := stack_height / thickness_sheet
  exact result - sorry

end sheets_in_stack_l604_604332


namespace num_ways_first_to_fourth_floor_l604_604252

theorem num_ways_first_to_fourth_floor (floors : ℕ) (staircases_per_floor : ℕ) 
  (H_floors : floors = 4) (H_staircases : staircases_per_floor = 2) : 
  (staircases_per_floor) ^ (floors - 1) = 2^3 := 
by 
  sorry

end num_ways_first_to_fourth_floor_l604_604252


namespace solve_complex_number_l604_604406

-- Define the complex number z satisfying the given equation
variable (z : ℂ)
variable (h : (sqrt 3 + 3 * complex.i) * z = sqrt 3 * complex.i)

-- Prove that this z equals the given answer
theorem solve_complex_number : z = (sqrt 3 / 4) + (1 / 4) * complex.i :=
by
  sorry

end solve_complex_number_l604_604406


namespace sufficient_but_not_necessary_l604_604399

variable {a b : ℝ} (P : a > b ∧ b > 0) (Q : a^2 > b^2)

theorem sufficient_but_not_necessary :
  (P → Q) ∧ (¬(Q → P)) :=
by
  sorry

end sufficient_but_not_necessary_l604_604399


namespace percentage_below_cost_price_is_correct_l604_604274

-- Define the cost price
def CP : ℝ := 7450

-- Define the selling price when making a 14% profit
def SP_14 : ℝ := CP + 0.14 * CP

-- Define the actual selling price
def Actual_SP : ℝ := SP_14 - 2086

-- Define the percentage below cost price
def Percentage_below_CP : ℝ := (CP - Actual_SP) / CP * 100

-- Prove that the percentage below CP is 14%
theorem percentage_below_cost_price_is_correct : Percentage_below_CP = 14 := by
  sorry

end percentage_below_cost_price_is_correct_l604_604274


namespace problem_statement_l604_604511

noncomputable def a_n (n : ℕ) : ℝ :=
  if n ≥ 2 then (Nat.choose n 2) * 3^(n-2) else 0

theorem problem_statement :
  (2016 / 2015) * (∑ n in (Finset.range 2016).filter (λ x, x ≥ 2), 3^n / a_n n) = 18 :=
by 
  sorry

end problem_statement_l604_604511


namespace petya_vasya_meeting_point_l604_604348

theorem petya_vasya_meeting_point :
  ∃ n : ℕ, n = 64 ∧ ∀ (k : ℕ), k = 100 → 
    ∀ (p_start v_start p_end v_end : ℕ), p_start = 1 → v_start = k → p_end = 22 → v_end = 88 →
      ∃ meet : ℕ, meet = 64 := 
begin
  sorry
end

end petya_vasya_meeting_point_l604_604348


namespace triangle_isosceles_of_parallel_bisectors_l604_604410

-- Given a triangle ABC
variables (A B C D E : Type) [triangle ABC]

-- Define the properties of points D and E
variables (h1 : parallel (line_through C (angle_bisector B)) (angle_bisector A D))
variables (h2 : parallel (line_through C (angle_bisector A)) (angle_bisector B E))
variables (h3 : parallel DE AB)

-- Prove CA = CB
theorem triangle_isosceles_of_parallel_bisectors (h1 : parallel (line_through C (angle_bisector B)) (angle_bisector A D))
                                               (h2 : parallel (line_through C (angle_bisector A)) (angle_bisector B E))
                                               (h3 : parallel DE AB) :
  CA = CB :=
sorry

end triangle_isosceles_of_parallel_bisectors_l604_604410


namespace f_is_odd_k_range_l604_604595

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom mono_f : monotone f
axiom f_eq_log2_3 : f 3 = real.logb 2 3
axiom f_additive : ∀ (x y : ℝ), f (x + y) = f x + f y

-- Part 1: Prove f is an odd function
theorem f_is_odd : ∀ (x : ℝ), f (-x) = -f x := sorry

-- Part 2: Determine the range of k
theorem k_range (k : ℝ) : (∀ x : ℝ, f (k * 3^x) + f (3^x - 9^x - 2) < 0) ↔ -2 * real.sqrt 2 - 1 < k ∧ k ≤ -1 := sorry

end f_is_odd_k_range_l604_604595


namespace ratio_of_a_over_5_to_b_over_4_l604_604461

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 5) / (b / 4) = 1 := by
  sorry

end ratio_of_a_over_5_to_b_over_4_l604_604461


namespace question_1_question_2_l604_604445
open Real

-- Definition of the quadratic function and the condition f(-1) = 0
def f (a : ℝ) (b : ℝ) (x : ℝ) := x^2 + 2 * a * x + b

-- Proof problem 1: Proving that b = 2a - 1 if f(-1) = 0
theorem question_1 (a b : ℝ) (h : f a b (-1) = 0) : b = 2 * a - 1 :=
by
  unfold f at h
  sorry

-- Proof problem 2: Finding the intervals of monotonicity for f(x) when a = -1
theorem question_2 (x : ℝ) : 
  let a := -1 
  let b := 2 * a - 1 
  let f := f a b 
  ∃ dec inc, (∀ x < 1, f' x < 0) ∧ (∀ x ≥ 1, f' x > 0) :=
by
  let a := -1
  let b := 2 * a - 1
  let f := f a b
  sorry

end question_1_question_2_l604_604445


namespace total_balls_donated_l604_604708

def num_elem_classes_A := 4
def num_middle_classes_A := 5
def num_elem_classes_B := 5
def num_middle_classes_B := 3
def num_elem_classes_C := 6
def num_middle_classes_C := 4
def balls_per_class := 5

theorem total_balls_donated :
  (num_elem_classes_A + num_middle_classes_A) * balls_per_class +
  (num_elem_classes_B + num_middle_classes_B) * balls_per_class +
  (num_elem_classes_C + num_middle_classes_C) * balls_per_class =
  135 :=
by
  sorry

end total_balls_donated_l604_604708


namespace circle_tangency_l604_604039

theorem circle_tangency (a : ℝ) :
  (∃ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧ x^2 + (y - real.sqrt 5)^2 = a^2)
  → a = 1/4 ∨ a = -1/4 :=
begin
  sorry,
end

end circle_tangency_l604_604039


namespace percentage_of_adjacent_repeated_digits_l604_604035

theorem percentage_of_adjacent_repeated_digits 
  (total_numbers : ℕ) (valid_range : finset ℕ) (count_adj_repeated_digits : ℕ) 
  (H1 : total_numbers = 900) 
  (H2 : valid_range = finset.Icc 100 999) 
  (H3 : count_adj_repeated_digits = 144) :
  (count_adj_repeated_digits : ℚ) / total_numbers * 100 = 16 := 
begin
  -- The actual proof goes here
  sorry
end

end percentage_of_adjacent_repeated_digits_l604_604035


namespace alice_steps_l604_604359

/-
  Conditions:
  1. The distance between Alice and Zoey's houses is 3 miles.
  2. Zoey's speed is 4 times Alice's walking speed.
  3. Alice covers 2 feet with each step.
  4. At halfway, Alice stops for 5 minutes to rest.
-/

noncomputable def distance_miles := 3
noncomputable def mile_to_feet := 5280
noncomputable def distance_feet := distance_miles * mile_to_feet
noncomputable def alice_speed (a : ℕ) := a
noncomputable def zoey_speed (a : ℕ) := 4 * a
noncomputable def steps_per_feet := 2
noncomputable def rest_time := 5

theorem alice_steps (a : ℕ) (h : a = 300) : 
  let total_distance := distance_feet
  let combined_speed := alice_speed a + zoey_speed a
  let meeting_time := total_distance / combined_speed
  let actual_meeting_time := meeting_time - rest_time
  let alice_distance := alice_speed a * actual_meeting_time
  let steps := alice_distance / steps_per_feet in
  steps = 834 := 
by 
  sorry

end alice_steps_l604_604359


namespace max_hot_dogs_with_budget_l604_604677

-- Definitions based on conditions
def pack1_price : ℝ := 1.55
def pack1_count : ℕ := 8
def pack2_price : ℝ := 3.05
def pack2_count : ℕ := 20
def pack3_price : ℝ := 22.95
def pack3_count : ℕ := 250
def budget : ℝ := 300

-- The main theorem to prove the maximum number of hot dogs
theorem max_hot_dogs_with_budget : 
  let price_per_hot_dog (price : ℝ) (count : ℕ) := price / count in
  let max_packs (budget : ℝ) (price : ℝ) := (budget / price).toInt in
  let num_hot_dogs (packs : ℕ) (count : ℕ) := packs * count in
  (num_hot_dogs (max_packs budget pack3_price) pack3_count) = 3250 := 
sorry

end max_hot_dogs_with_budget_l604_604677


namespace number_of_triangles_l604_604616

noncomputable section

def circles : ℕ := 5
def triangles : ℕ := 2 * circles

theorem number_of_triangles : triangles = 10 :=
by
  unfold circles triangles
  sorry

end number_of_triangles_l604_604616


namespace range_of_lambda_l604_604041

variable (a b λ : ℝ)

theorem range_of_lambda (h : ∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 + 2 > λ * (a + b)) : λ < 2 :=
sorry

end range_of_lambda_l604_604041


namespace non_neg_reals_sum_to_one_l604_604136

theorem non_neg_reals_sum_to_one {n : ℕ} (a : Fin n → ℝ) 
  (h0 : ∀ i, 0 ≤ a i) (h1 : ∑ i, a i = 1) : 
  ∃ (σ : Equiv.Perm (Fin n)), (∑ i : Fin n, a (σ i) * a (σ (⟨(i + 1) % n, sorry⟩ : Fin n))) ≤ 1 / n :=
begin
  sorry
end

end non_neg_reals_sum_to_one_l604_604136


namespace num_distinct_elements_sqrt_fraction_set_l604_604129

def floor_function (x : ℝ) : ℤ := Int.floor x

def sqrt_fraction_set (d : ℝ) : Finset ℤ := 
  (Finset.range 2005).image (λ n, floor_function ((n + 1) ^ 2 / d))

theorem num_distinct_elements_sqrt_fraction_set :
  (sqrt_fraction_set 2005).card = 1501 :=
  sorry

end num_distinct_elements_sqrt_fraction_set_l604_604129


namespace total_donations_l604_604210

-- Define the conditions
def started_donating_age : ℕ := 17
def current_age : ℕ := 71
def annual_donation : ℕ := 8000

-- Define the proof problem to show the total donation amount equals $432,000
theorem total_donations : (current_age - started_donating_age) * annual_donation = 432000 := 
by
  sorry

end total_donations_l604_604210


namespace probability_same_color_is_correct_l604_604862

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l604_604862


namespace isosceles_right_triangle_incircle_radius_l604_604062

theorem isosceles_right_triangle_incircle_radius 
  (s : ℝ) 
  (h_s : s = 8) 
  (r : ℝ) 
  (h_r : r = (area (isosceles_right_triangle s)) / (semiperimeter (isosceles_right_triangle s))) :
  r = 8 - 4 * Real.sqrt 2 := sorry

-- Definitions for "area", "semiperimeter", and "isosceles_right_triangle" should be appropriately created in practice but are skipped here for brevity

end isosceles_right_triangle_incircle_radius_l604_604062


namespace no_set_with_7_elements_min_elements_condition_l604_604657

noncomputable def set_a_elements := 7
noncomputable def median_a := 10
noncomputable def mean_a := 6
noncomputable def min_sum_a := 3 + 4 * 10
noncomputable def real_sum_a := mean_a * set_a_elements

theorem no_set_with_7_elements : ¬ (set_a_elements = 7 ∧
  (∃ S : Finset ℝ, 
    (S.card = set_a_elements) ∧ 
    (S.sum ≥ min_sum_a) ∧ 
    (S.sum = real_sum_a))) := 
by
  sorry

noncomputable def n_b_elements := ℕ
noncomputable def set_b_elements (n : ℕ) := 2 * n + 1
noncomputable def median_b := 10
noncomputable def mean_b := 6
noncomputable def min_sum_b (n : ℕ) := n + 10 * (n + 1)
noncomputable def real_sum_b (n : ℕ) := mean_b * set_b_elements n

theorem min_elements_condition (n : ℕ) : 
    (∀ n : ℕ, n ≥ 4) → 
    (set_b_elements n ≥ 9 ∧
        ∃ S : Finset ℝ, 
          (S.card = set_b_elements n) ∧ 
          (S.sum ≥ min_sum_b n) ∧ 
          (S.sum = real_sum_b n)) :=
by
  assume h : ∀ n : ℕ, n ≥ 4
  sorry

end no_set_with_7_elements_min_elements_condition_l604_604657


namespace find_m_l604_604034

theorem find_m (m : ℝ) (h : 2^2 + 2 * m + 2 = 0) : m = -3 :=
by {
  sorry
}

end find_m_l604_604034


namespace bridge_length_at_least_200_l604_604588

theorem bridge_length_at_least_200 :
  ∀ (length_train : ℝ) (speed_kmph : ℝ) (time_secs : ℝ),
  length_train = 200 ∧ speed_kmph = 32 ∧ time_secs = 20 →
  ∃ l : ℝ, l ≥ length_train :=
by
  sorry

end bridge_length_at_least_200_l604_604588


namespace carlos_gold_quarters_l604_604731

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end carlos_gold_quarters_l604_604731


namespace solve_system1_solve_system2_l604_604568

theorem solve_system1 (x y : ℝ) (h1 : y = x - 4) (h2 : x + y = 6) : x = 5 ∧ y = 1 :=
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x + y = 1) (h2 : 4 * x - y = 5) : x = 1 ∧ y = -1 :=
by sorry

end solve_system1_solve_system2_l604_604568


namespace find_f_2008_l604_604513

def f (x : ℝ) : ℝ := sorry -- Function f is defined with unknown details

axiom f_0 : f 0 = 2008

axiom inequality_1 : ∀ x : ℝ, f(x + 2) - f(x) ≤ 3 * 2^x

axiom inequality_2 : ∀ x : ℝ, f(x + 6) - f(x) ≥ 63 * 2^x

theorem find_f_2008 : f 2008 = 2007 + 2^2008 := 
by
  sorry

end find_f_2008_l604_604513


namespace angle_PQR_correct_l604_604917

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end angle_PQR_correct_l604_604917


namespace jenna_reading_goal_l604_604941

theorem jenna_reading_goal (total_days : ℕ) (total_pages : ℕ) (unread_days : ℕ) (pages_on_23rd : ℕ) :
  total_days = 30 → total_pages = 600 → unread_days = 4 → pages_on_23rd = 100 →
  ∃ (pages_per_day : ℕ), 
  let days_to_read := total_days - unread_days - 1 in
  let pages_to_read_on_other_days := total_pages - pages_on_23rd in
  days_to_read ≠ 0 →
  pages_per_day * days_to_read = pages_to_read_on_other_days ∧ pages_per_day = 20 :=
by
  intros h1 h2 h3 h4
  use 20
  simp_all
  sorry

end jenna_reading_goal_l604_604941


namespace angle_between_vectors_l604_604846

variables (a b : ℝ^n) -- Vectors a and b in n-dimensional space where n ≥ 2

-- Given conditions
def norm_a := ∥a∥ = 2
def norm_b := ∥b∥ = 1
def dot_ab := a ⬝ b = 1

-- Prove the angle between vector 'a' and vector '(a - b)' is π/6
theorem angle_between_vectors (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 1) (dot_ab : a ⬝ b = 1) :
  let θ := real.acos (a ⬝ (a - b) / (∥a∥ * ∥a - b∥)) in θ = π / 6 :=
by sorry

end angle_between_vectors_l604_604846


namespace problem1_problem2_l604_604006

noncomputable def f (x a : ℝ) : ℝ := x * log x + a * x

theorem problem1 (a : ℝ) :
  (∀ x ∈ set.Icc real.exp (real.exp 2), deriv (f x a) x ≤ 0) ↔ a ≤ -3 :=
by sorry

theorem problem2 (a : ℝ) (k : ℝ) :
  (∀ x > 1, f x a > k * (x - 1) + a * x - x) →
  k ≤ 3 :=
by sorry

end problem1_problem2_l604_604006


namespace total_amount_paid_l604_604533

-- Define the conditions
def chicken_nuggets_ordered : ℕ := 100
def nuggets_per_box : ℕ := 20
def cost_per_box : ℕ := 4

-- Define the hypothesis on the amount of money paid for the chicken nuggets
theorem total_amount_paid :
  (chicken_nuggets_ordered / nuggets_per_box) * cost_per_box = 20 :=
by
  sorry

end total_amount_paid_l604_604533


namespace carlos_gold_quarters_l604_604739

theorem carlos_gold_quarters :
  (let quarter_weight := 1 / 5
       quarter_value := 0.25
       value_per_ounce := 100
       quarters_per_ounce := 1 / quarter_weight
       melt_value := value_per_ounce
       spend_value := quarters_per_ounce * quarter_value
    in melt_value / spend_value = 80) :=
by
  -- Definitions
  let quarter_weight := 1 / 5
  let quarter_value := 0.25
  let value_per_ounce := 100
  let quarters_per_ounce := 1 / quarter_weight
  let melt_value := value_per_ounce
  let spend_value := quarters_per_ounce * quarter_value

  -- Conclusion to be proven
  have h1 : quarters_per_ounce = 5 := sorry
  have h2 : spend_value = 1.25 := sorry
  have h3 : melt_value / spend_value = 80 := sorry

  show melt_value / spend_value = 80 from h3

end carlos_gold_quarters_l604_604739


namespace intersection_A_B_l604_604127

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l604_604127


namespace average_exp_Feb_to_Jul_l604_604715

theorem average_exp_Feb_to_Jul (x y z : ℝ) 
    (h1 : 1200 + x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) = 6 * 4200) 
    (h2 : 0 ≤ x) 
    (h3 : 0 ≤ z) : 
    (x + 0.85 * x + z + 1.10 * z + 0.90 * (1.10 * z) + 1500) / 6 = 4250 :=
by
    sorry

end average_exp_Feb_to_Jul_l604_604715


namespace acid_base_mixture_ratio_l604_604265

theorem acid_base_mixture_ratio (r s t : ℝ) (hr : r ≥ 0) (hs : s ≥ 0) (ht : t ≥ 0) :
  (r ≠ -1) → (s ≠ -1) → (t ≠ -1) →
  let acid_volume := (r/(r+1) + s/(s+1) + t/(t+1))
  let base_volume := (1/(r+1) + 1/(s+1) + 1/(t+1))
  acid_volume / base_volume = (rst + rt + rs + st) / (rs + rt + st + r + s + t + 3) := 
by {
  sorry
}

end acid_base_mixture_ratio_l604_604265


namespace ratio_rope_to_stairs_l604_604502

theorem ratio_rope_to_stairs (r : ℝ) (h_stairs : ℝ) (h_ladder : ℝ) 
  (h_total : ℝ) (h_per_flight : ℝ) 
  (h_stairs_eq : h_stairs = 3 * h_per_flight) 
  (h_ladder_eq : h_ladder = r + 10) 
  (h_total_eq : h_stairs + r + h_ladder = h_total) 
  (h_per_flight_eq : h_per_flight = 10) 
  (h_total_eq' : h_total = 70) : 
  r / h_per_flight = 3 / 2 := 
by
  have hpf : h_per_flight = 10 := h_per_flight_eq
  have hst : h_stairs = 30 := by rw [h_per_flight_eq, h_stairs_eq]
  have htl : h_total = 70 := h_total_eq'
  rw [h_per_flight_eq, h_stairs_eq, h_ladder_eq, h_total_eq] at h_total_eq'
  have h_r : 2 * r + 40 = 70 := 
    by rw [h_per_flight_eq, h_stairs_eq, h_ladder_eq, h_total_eq] 
  have h_r' : r = 15 := by linarith
  rw [h_r', h_per_flight_eq]
  exact (by norm_num : (15 : ℚ) / 10 = 3 / 2)
  sorry

end ratio_rope_to_stairs_l604_604502


namespace first_digit_base_9_1024_eq_1_l604_604637

def first_digit_base_9_of_1024 : ℕ := 1

theorem first_digit_base_9_1024_eq_1 :
  ∀ n : ℕ, n = 1024 → (first_digit_of_base_9 n = 1) :=
begin
  intro n,
  intro h_n,
  rw h_n,
  sorry
end

/- Auxiliary definitions -/
def first_digit_of_base_9 (n : ℕ) : ℕ :=
  let base9_rep := nat.digits 9 n in
  match base9_rep with
  | [] => 0
  | (hd :: tl) => hd
  end

end first_digit_base_9_1024_eq_1_l604_604637


namespace math_problem_statement_l604_604485

variables {b x y k : ℝ} (hb : b > 0) (hx_region : -2 < x ∧ x < 3) (hk_region : k ≤ x ∧ x ≤ 2)

-- Definition of the parabola
def parabola (x : ℝ) (b : ℝ) : ℝ := x^2 + 2*b*x + b^2 - 2

-- The point B(0, -1) is on the parabola
def point_B_on_parabola : Prop := parabola 0 b = -1

-- Vertex of the parabola
def vertex_of_parabola : Prop := ∀ b, (-b, -2) = (-b, parabola (-b) b)

-- Range of y for -2 < x < 3
def range_of_y (x : ℝ) (b : ℝ) (hx_region : -2 < x ∧ x < 3) : Prop := 
  -2 ≤ parabola x b ∧ parabola x b < 14

-- Range of k for k ≤ x ≤ 2 and -2 ≤ y ≤ 7
def range_of_k (x : ℝ) (y : ℝ) (k : ℝ) (hk_region : k ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 7) : Prop :=
  -4 ≤ k ∧ k ≤ -1

-- Lean theorem statement
theorem math_problem_statement :
  point_B_on_parabola b → vertex_of_parabola b → (∀ x, range_of_y x b hx_region) → 
  (∀ x y k, range_of_k x y k hk_region) :=
sorry

end math_problem_statement_l604_604485


namespace ARML_computation_l604_604509

noncomputable def ARML_value (A R M L : ℝ) : ℝ :=
  A * R * M * L

theorem ARML_computation
  (A R M L : ℝ)
  (h1 : log (AL) / log 10 + log (AM) / log 10 = 3)
  (h2 : log (ML) / log 10 + log (MR) / log 10 = 4)
  (h3 : log (RA) / log 10 + log (RL) / log 10 = 5)
  (h4 : L = 2 * M) :
  ARML_value A R M L = 100000 := by
  sorry

end ARML_computation_l604_604509


namespace simplify_sqrt_expression_correct_l604_604043

noncomputable def simplify_sqrt_expression (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) : ℝ :=
  (Real.sqrt (9 - 6 * m + m^2)) - (Real.sqrt (m^2 - 14 * m + 49))

theorem simplify_sqrt_expression_correct (m : ℝ) (h_triangle : (2 < m + 5) ∧ (m < 2 + 5) ∧ (5 < 2 + m)) :
  simplify_sqrt_expression m h_triangle = 2 * m - 10 :=
sorry

end simplify_sqrt_expression_correct_l604_604043


namespace incenter_divides_CM_in_ratio_2_to_1_l604_604684

noncomputable def triangle (A B C : Type) := ∃ (AB BC CA : ℝ), AB = 15 ∧ BC = 12 ∧ CA = 18 

def incenter_divides_angle_bisector (A B C O M : Type) (hABC : triangle A B C) 
  (hIncenter : ¬∃ P, O ∉ P ∧ P divides CM) : Prop :=
  ∃ ratio : ℝ, ratio = 2 / 1 

theorem incenter_divides_CM_in_ratio_2_to_1 :
  ∀ A B C O M, triangle A B C → 
  incenter_divides_angle_bisector A B C O M (triangle A B C) = 2 / 1 :=
by
  sorry

end incenter_divides_CM_in_ratio_2_to_1_l604_604684


namespace solve_for_x_l604_604286

theorem solve_for_x : (∃ x : ℚ, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
by
  use 80 / 3
  sorry

end solve_for_x_l604_604286


namespace triangle_BC_calculation_l604_604076

variable {α : Type*} [LinearOrder α] [NormedSpace ℝ α] [InnerProductSpace ℝ α] {A B C H Y : α}
variable (AH AY dist_Y_AC BC : ℝ)

theorem triangle_BC_calculation
  (h1 : AH = 4)
  (h2 : AY = 6)
  (h3 : dist_Y_AC = Real.sqrt 15) :
  BC = 4 * Real.sqrt 35 := 
sorry

end triangle_BC_calculation_l604_604076


namespace price_per_rose_l604_604721

theorem price_per_rose 
  (initial_roses : ℕ)
  (remaining_roses : ℕ)
  (total_amount_earned : ℕ) 
  (roses_sold : ℕ := initial_roses - remaining_roses) :
  roses_sold * (total_amount_earned / roses_sold) = total_amount_earned := 
by 
  have h_roses_sold : roses_sold = initial_roses - remaining_roses := by rfl
  have h_correct_answer : total_amount_earned / roses_sold = 36 / 9 := by rfl
  sorry

end price_per_rose_l604_604721


namespace find_y_l604_604046

noncomputable def y (x : ℝ) : ℝ := x / 1.13

theorem find_y (x : ℝ) (hx : x = 90.4) : y x = 80 :=
by {
  have hy : y 90.4 = 80, {
    dsimp [y],
    exact (div_eq_of_eq_mul (by norm_num)).symm
  },
  rwa hx at hy
}

end find_y_l604_604046


namespace exists_arbitrarily_large_N_l604_604651

noncomputable def sequence_condition (x : ℕ → ℝ) : Prop :=
(∀ n : ℕ, 0 < x n) ∧ (∀ n m : ℕ, n < m → x n < x m) ∧ (tendsto (λ n, x n / n) at_top (𝓝 0))

theorem exists_arbitrarily_large_N (x : ℕ → ℝ) (hx : sequence_condition x) :
  ∃ᶠ N in at_top, ∀ m : ℕ, 1 ≤ m → m < N → x (N-m) + x (N+m) < 2 * x N :=
sorry

end exists_arbitrarily_large_N_l604_604651


namespace intersection_A_B_eq_C_l604_604100

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l604_604100


namespace odd_prime_inequality_l604_604759

open Nat

def is_odd_prime (p : ℕ) : Prop :=
  prime p ∧ ∃ n : ℕ, p = 2 * n + 1

theorem odd_prime_inequality (p : ℕ) (n : ℕ) (m : ℕ) (h₁ : p = 2 * n + 1) (h₂ : m = n^2) :
  is_odd_prime p :=
by
  sorry

end odd_prime_inequality_l604_604759


namespace trigonometry_problem_l604_604812

theorem trigonometry_problem
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (4 / 5, -3 / 5))
  (h_unit : P.1^2 + P.2^2 = 1) :
  (cos α = 4 / 5) ∧
  (tan α = -3 / 4) ∧
  (sin (α + π) = 3 / 5) ∧
  (cos (α - π / 2) ≠ 3 / 5) := by
    sorry

end trigonometry_problem_l604_604812


namespace radius_of_inscribed_circle_is_integer_l604_604188

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l604_604188


namespace corn_syrup_amount_in_sport_formulation_l604_604074

noncomputable def sport_formulation_corn_syrup : ℕ :=
  let F := 105 / 60 in
  let C := 4 * F in
  C

theorem corn_syrup_amount_in_sport_formulation (W : ℕ) (hW : W = 105) : sport_formulation_corn_syrup = 7 := by
  sorry

end corn_syrup_amount_in_sport_formulation_l604_604074


namespace probability_same_color_is_correct_l604_604860

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l604_604860


namespace part_a_part_b_l604_604655

-- Definitions and Conditions
def median := 10
def mean := 6

-- Part (a): Prove that a set with 7 numbers cannot satisfy the given conditions
theorem part_a (n1 n2 n3 n4 n5 n6 n7 : ℕ) (h1 : median ≤ n1) (h2 : median ≤ n2) (h3 : median ≤ n3) (h4 : median ≤ n4)
  (h5 : 1 ≤ n5) (h6 : 1 ≤ n6) (h7 : 1 ≤ n7) (hmean : (n1 + n2 + n3 + n4 + n5 + n6 + n7) / 7 = mean) :
  false :=
by
  sorry

-- Part (b): Prove that the minimum size of the set where number of elements is 2n + 1 and n is a natural number, is at least 9
theorem part_b (n : ℕ) (h_sum_geq : ∀ (s : Finset ℕ), ((∀ x ∈ s, x >= median) ∧ ∃ t : Finset ℕ, t ⊆ s ∧ (∀ x ∈ t, x >= 1) ∧ s.card = 2 * n + 1) → s.sum >= 11 * n + 10) :
  n ≥ 4 :=
by
  sorry

-- Lean statements defined above match the problem conditions and required proofs

end part_a_part_b_l604_604655


namespace probability_same_color_plates_l604_604851

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l604_604851


namespace octopus_legs_l604_604254

-- Definitions of octopus behavior based on the number of legs
def tells_truth (legs: ℕ) : Prop := legs = 6 ∨ legs = 8
def lies (legs: ℕ) : Prop := legs = 7

-- Statements made by the octopuses
def blue_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 28
def green_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 27
def yellow_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 26
def red_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 25

noncomputable def legs_b := 7
noncomputable def legs_g := 6
noncomputable def legs_y := 7
noncomputable def legs_r := 7

-- Main theorem
theorem octopus_legs : 
  (tells_truth legs_g) ∧ 
  (lies legs_b) ∧ 
  (lies legs_y) ∧ 
  (lies legs_r) ∧ 
  blue_statement legs_b legs_g legs_y legs_r ∧ 
  green_statement legs_b legs_g legs_y legs_r ∧ 
  yellow_statement legs_b legs_g legs_y legs_r ∧ 
  red_statement legs_b legs_g legs_y legs_r := 
by 
  sorry

end octopus_legs_l604_604254


namespace cube_divisors_count_l604_604033

-- Define the conditions
def is_cube (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n^3

def num_divisors (x : ℕ) : ℕ :=
  (Nat.divisors x).length

-- Define the main theorem
theorem cube_divisors_count (x : ℕ) (d : ℕ) 
  (hx : is_cube x) (hd : d = num_divisors x) : 
  d = 202 :=
sorry

end cube_divisors_count_l604_604033


namespace cube_root_of_sqrt_64_l604_604220

variable (a : ℕ) (b : ℕ)

def cubeRootOfSqrt64 : Prop :=
  a = Nat.sqrt 64 ∧ b = Nat.cbrt a → b = 2

theorem cube_root_of_sqrt_64 (a : ℕ) (b : ℕ) : cubeRootOfSqrt64 a b :=
  by
  sorry

end cube_root_of_sqrt_64_l604_604220


namespace area_of_region_l604_604483

theorem area_of_region : 
  (∃ (A : ℝ), A = 12 ∧ ∀ (x y : ℝ), |x| + |y| + |x - 2| ≤ 4 → 
    (0 ≤ y ∧ y ≤ 6 - 2*x ∧ x ≥ 2) ∨
    (0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ x ∧ x < 2) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ -1 ≤ x ∧ x < 0) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ x < -1)) :=
sorry

end area_of_region_l604_604483


namespace cross_signal_pole_time_l604_604318

noncomputable def time_to_cross_signal_pole (length_train : ℝ) (length_platform : ℝ) (time_to_cross_platform : ℝ) : ℝ :=
  let total_distance := length_train + length_platform
  let speed := total_distance / time_to_cross_platform
  length_train / speed

theorem cross_signal_pole_time : 
  time_to_cross_signal_pole 300 350 39 ≈ 18 := 
by
  sorry

end cross_signal_pole_time_l604_604318


namespace sequence_a_n_perfect_square_l604_604419

theorem sequence_a_n_perfect_square :
  (∃ a : ℕ → ℤ, ∃ b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = 0 ∧
    (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    (∀ n : ℕ, ∃ k : ℤ, a n = k^2)) :=
sorry

end sequence_a_n_perfect_square_l604_604419


namespace problem1_problem2_problem3_l604_604809

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (b - 2^x) / (2^x + a)

theorem problem1
  (f_is_odd : ∀ x : ℝ, f (-x) 1 1 = -f x 1 1) :
  1 = 1 ∧ 1 = 1 := 
sorry

theorem problem2 :
  ∀ x1 x2 : ℝ, x1 < x2 → (f x1 1 1) > (f x2 1 1) :=
sorry

theorem problem3 
  (h : ∀ t : ℝ, f (t^2 - 2 * t) 1 1 + f (2 * t^2 - k) 1 1 < 0) :
  k ∈ Iio (-1/3) :=
sorry

end problem1_problem2_problem3_l604_604809


namespace number_of_donuts_finished_l604_604504

-- Definitions from conditions
def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def total_spent : ℕ := 18

-- Theorem statement
theorem number_of_donuts_finished (H1 : ounces_per_donut = 2)
                                   (H2 : ounces_per_pot = 12)
                                   (H3 : cost_per_pot = 3)
                                   (H4 : total_spent = 18) : 
  ∃ n : ℕ, n = 36 :=
  sorry

end number_of_donuts_finished_l604_604504


namespace area_of_region_l604_604203

def rectangle_area (side1 side2 : ℝ) : ℝ := side1 * side2

def quarter_circle_area (radius : ℝ) : ℝ := (1/4) * real.pi * radius^2

def total_area (side1 side2 : ℝ) (radius_a radius_b radius_c : ℝ) : ℝ :=
  let rect_area := rectangle_area side1 side2
  let circle_area_a := quarter_circle_area radius_a
  let circle_area_b := quarter_circle_area radius_b
  let circle_area_c := quarter_circle_area radius_c
  rect_area - (circle_area_a + circle_area_b + circle_area_c)

theorem area_of_region (side1 side2 : ℝ) (radius_a radius_b radius_c : ℝ) :
  side1 = 4 → side2 = 6 → radius_a = 2 → radius_b = 3 → radius_c = 4 →
  abs (total_area side1 side2 radius_a radius_b radius_c - 1.5) < 0.001 :=
by
  intros h_side1 h_side2 h_radius_a h_radius_b h_radius_c
  -- Proof would go here
  sorry

end area_of_region_l604_604203


namespace number_of_passed_boys_l604_604578

theorem number_of_passed_boys 
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 36 * 120) :
  P = 105 := 
sorry

end number_of_passed_boys_l604_604578


namespace fg_of_3_l604_604415

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem fg_of_3 : f (g 3) = 25 := by
  sorry

end fg_of_3_l604_604415


namespace part1_part2_l604_604409

-- Define the sequence S_n
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n - 5 * a n - 85

-- Define the sequence a_n with the given form
def a (n : ℕ) := 1 - 15 * (5 / 6)^(n - 1)

-- Define the sequence b_n
def b (n : ℕ) : ℕ → ℝ := 
  λ m, Finset.sum (Finset.range m) (λ k, Real.log (1 - a(k + 1)) / 18) / Real.log (5 / 6)

-- Define the sequence T_n
def T (n : ℕ) := 2 * (1 - 1 / (n + 1))

-- Prove the equivalences
theorem part1 (n : ℕ) (hn : 0 < n) : 
  ∃ r, (a n - 1 = -15 * r^(n - 1) ∧ r = 5 / 6) :=
sorry

theorem part2 (n : ℕ) : 
  T n = 2 * n / (n + 1) :=
sorry

end part1_part2_l604_604409


namespace train_crosses_pole_in_36_seconds_l604_604342

-- Definitions based on conditions
def speed_km_per_hr : ℝ := 70
def length_meters : ℝ := 700

-- Conversion factor
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

def speed_m_per_s : ℝ := speed_km_per_hr * (km_to_m / hr_to_s)

def train_crossing_time : ℝ := length_meters / speed_m_per_s

-- Prove that the crossing time is 36 seconds
theorem train_crosses_pole_in_36_seconds : train_crossing_time = 36 :=
by
  sorry

end train_crosses_pole_in_36_seconds_l604_604342


namespace triangle_perimeter_expression_triangle_perimeter_maximum_l604_604899

def angle_A := Real.pi / 3
def side_BC := 2 * Real.sqrt 3
def angle_B := λ x: ℝ, (0 < x ∧ x < 2 * Real.pi / 3)

theorem triangle_perimeter_expression (x : ℝ) (hx : 0 < x ∧ x < 2 * Real.pi / 3) :
  let y := 4*Real.sqrt 3 * Real.sin (x + Real.pi / 6) + 2 * Real.sqrt 3 in
  y = 4 * Real.sin x + 4 * Real.sin (2 * Real.pi / 3 - x) + 2 * Real.sqrt 3 :=
sorry

theorem triangle_perimeter_maximum (x : ℝ) (hx : 0 < x ∧ x < 2 * Real.pi / 3) :
  let y := 4*Real.sqrt 3 * Real.sin (x + Real.pi / 6) + 2 * Real.sqrt 3 in
  (∃ x, x = Real.pi / 3 ∧ y = 6 * Real.sqrt 3) :=
sorry

end triangle_perimeter_expression_triangle_perimeter_maximum_l604_604899


namespace select_team_ways_l604_604542

theorem select_team_ways :
  let boys := 7
  let girls := 9
  let team_size_boys := 4
  let team_size_girls := 3
  nat.choose boys team_size_boys * nat.choose girls team_size_girls = 2940 :=
by
  let boys := 7
  let girls := 9
  let team_size_boys := 4
  let team_size_girls := 3
  have h1 : nat.choose boys team_size_boys = 35 := sorry
  have h2 : nat.choose girls team_size_girls = 84 := sorry
  calc
    nat.choose boys team_size_boys * nat.choose girls team_size_girls
        = 35 * 84 : by rw [h1, h2]
    ... = 2940 : by norm_num

end select_team_ways_l604_604542


namespace max_subsets_of_N_l604_604016

def M : Set ℕ := {0, 2, 3, 7}
def N : Set ℕ := {x | ∃ a b ∈ M, x = a * b}

theorem max_subsets_of_N : ∃ n = 7, ∀ (N : Set ℕ), (card N = n → 2^n = 128) :=
by {
  sorry
}

end max_subsets_of_N_l604_604016


namespace sqrt_eq_solution_l604_604768

theorem sqrt_eq_solution (x : ℝ) (h : sqrt (4 * x - 3) + 18 / sqrt (4 * x - 3) = 9) : x = 3 ∨ x = 9.75 :=
by
  sorry

end sqrt_eq_solution_l604_604768


namespace ratio_of_compositions_l604_604460

def f (x : ℕ) : ℕ := 2 * x + 3
def g (x : ℕ) : ℕ := 3 * x - 2

theorem ratio_of_compositions : (f(g(f(2))) / g(f(g(2)))) = (41 / 31) := by
  sorry

end ratio_of_compositions_l604_604460


namespace split_cost_evenly_l604_604082

noncomputable def cupcake_cost : ℝ := 1.50
noncomputable def number_of_cupcakes : ℝ := 12
noncomputable def total_cost : ℝ := number_of_cupcakes * cupcake_cost
noncomputable def total_people : ℝ := 2

theorem split_cost_evenly : (total_cost / total_people) = 9 :=
by
  -- Skipping the proof for now
  sorry

end split_cost_evenly_l604_604082


namespace basket_A_apples_count_l604_604479

-- Conditions
def total_baskets : ℕ := 5
def avg_fruits_per_basket : ℕ := 25
def fruits_in_B : ℕ := 30
def fruits_in_C : ℕ := 20
def fruits_in_D : ℕ := 25
def fruits_in_E : ℕ := 35

-- Calculation of total number of fruits
def total_fruits : ℕ := total_baskets * avg_fruits_per_basket
def other_baskets_fruits : ℕ := fruits_in_B + fruits_in_C + fruits_in_D + fruits_in_E

-- Question and Proof Goal
theorem basket_A_apples_count : total_fruits - other_baskets_fruits = 15 := by
  sorry

end basket_A_apples_count_l604_604479


namespace square_root_area_ratio_l604_604149

theorem square_root_area_ratio 
  (side_C : ℝ) (side_D : ℝ)
  (hC : side_C = 45) 
  (hD : side_D = 60) : 
  Real.sqrt ((side_C^2) / (side_D^2)) = 3 / 4 := by
  -- proof goes here
  sorry

end square_root_area_ratio_l604_604149


namespace like_terms_calc_l604_604420

theorem like_terms_calc {m n : ℕ} (h1 : m + 2 = 6) (h2 : n + 1 = 3) : (- (m : ℤ))^3 + (n : ℤ)^2 = -60 :=
  sorry

end like_terms_calc_l604_604420


namespace tom_remaining_balloons_l604_604626

theorem tom_remaining_balloons (initial_balloons : ℕ) (given_balloons : ℕ) : 
  initial_balloons = 30 ∧ given_balloons = 16 → initial_balloons - given_balloons = 14 := 
by
  intros h
  cases h with h_initial h_given
  rw [h_initial, h_given]
  exact rfl

end tom_remaining_balloons_l604_604626


namespace contradiction_for_n3_min_elements_when_n_ge_4_l604_604675

theorem contradiction_for_n3 :
  ∀ (s : Set ℕ), (s.card = 7) → 
                 (∀ (x ∈ s), x ≥ 1) → 
                 (∃ t u : Set ℕ, (t.card = 4) ∧ (u.card = 3) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) 
                 → ∃ x ∈ s, false :=
sorry

theorem min_elements_when_n_ge_4 (n : ℕ) (hn : n ≥ 4) :
  ∃ (s : Set ℕ), (s.card = 2 * n + 1) ∧ 
                 (∀ (x ∈ s), x ≥ 1) ∧ 
                 (∃ t u : Set ℕ, (t.card = n + 1) ∧ (u.card = n) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) ∧
                 ∀ (s : Set ℕ), s.card = 2 * n + 1 → (∑ x in s, x) / (2 * n + 1) = 6 :=
sorry

example : ∃ s, (s.card = 9) ∧ (∀ x ∈ s, x ≥ 1) ∧ 
               (∃ t u : Set ℕ, (t.card = 5) ∧ (u.card = 4) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
               (∀ x ∈ t, x ≥ 10) ∧ (∀ x ∈ u, x ≥ 1) ∧
               (∑ x in s, x) / 9 = 6) :=
{ sorry }

end contradiction_for_n3_min_elements_when_n_ge_4_l604_604675


namespace arithmetic_sequence_of_condition_l604_604785

variables {R : Type*} [LinearOrderedRing R]

theorem arithmetic_sequence_of_condition (x y z : R) (h : (z-x)^2 - 4*(x-y)*(y-z) = 0) : 2*y = x + z :=
sorry

end arithmetic_sequence_of_condition_l604_604785


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604194

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604194


namespace simplify_fraction_sum_l604_604564

theorem simplify_fraction_sum (num denom : ℕ) (h : num = 75 ∧ denom = 100) :
  let simp_num := 3
  let simp_denom := 4
  simp_num + simp_denom = 7 :=
by
  cases h with hnum hdenom
  subst hnum
  subst hdenom
  let simp_num := 3
  let simp_denom := 4
  show simp_num + simp_denom = 7
  sorry

end simplify_fraction_sum_l604_604564


namespace triangle_dot_product_l604_604495

noncomputable def vector_dot_product {V : Type*} [inner_product_space ℝ V]  (v w : V) : ℝ := inner v w

variables {A B C P: Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space P] 
variables {V : Type*} [inner_product_space ℝ V]

def midpoint (x y : V) : V := 1/2 • (x + y)

theorem triangle_dot_product 
(AB AC BC AP : V)
(AB_eq_2 : AB = 2)
(angle_B_eq_pi4 : angle A B C = (π / 4))
(angle_C_eq_pi6 : angle A C B = (π / 6))
(P_midpoint_BC : P = midpoint B C) :
vector_dot_product AP BC = 2 :=
sorry

end triangle_dot_product_l604_604495


namespace cm_bisects_angle_bcd_l604_604546

-- Define the conditions of the problem
variables (A B C D K L M : Point)
variable [AffineSpace ℝ Point Line]

-- Let the points be vertices of the parallelogram ABCD
variable (parallelogramABCD : parallelogram A B C D)

-- K and L are extensions such that BK = DL and M is the intersection of BL and DK
variable (B_extend : extend (Line.mk B D) B K)
variable (D_extend : extend (Line.mk D B) D L)
variable (BK_DL_eq : (distance B K) = (distance D L))
variable (intersectionM : intersect_bl_dk (Line.mk B L) (Line.mk D K) = M)

-- The statement to be proved: CM bisects angle BCD
theorem cm_bisects_angle_bcd :
  bisects (Line.mk C M) (angle ∡ B C D) := 
sorry

end cm_bisects_angle_bcd_l604_604546


namespace can_make_1_over_60_can_make_2011_over_375_cannot_make_1_over_7_l604_604978

/-- Initial set -/
def initial_set : set ℚ := {1, 1/2, 1/3, 1/4, 1/5, 1/6}

/-- Allowed operations: addition and multiplication on the board. -/
def allowed_operations (A : set ℚ) : set ℚ :=
  A ∪ {a + b | a b ∈ A, a + b ≠ 0} ∪ {a * b | a b ∈ A}

/-- Closure of the initial set under allowed operations -/
def closure (A : set ℚ) : set ℚ :=
  { q | ∃ (steps : ℕ → set ℚ), steps 0 = A ∧ 
    (∀ n, steps (n + 1) = allowed_operations (steps n)) ∧
    (∀ n, q ∈ steps n)}

/-- It is possible to make 1/60 appear on the board. -/
theorem can_make_1_over_60 : (1 / 60) ∈ closure initial_set := 
sorry

/-- It is possible to make 2011/375 appear on the board. -/
theorem can_make_2011_over_375 : (2011 / 375) ∈ closure initial_set := 
sorry

/-- It is not possible to make 1/7 appear on the board. -/
theorem cannot_make_1_over_7 : (1 / 7) ∉ closure initial_set := 
sorry

end can_make_1_over_60_can_make_2011_over_375_cannot_make_1_over_7_l604_604978


namespace carlos_gold_quarters_l604_604738

theorem carlos_gold_quarters :
  (let quarter_weight := 1 / 5
       quarter_value := 0.25
       value_per_ounce := 100
       quarters_per_ounce := 1 / quarter_weight
       melt_value := value_per_ounce
       spend_value := quarters_per_ounce * quarter_value
    in melt_value / spend_value = 80) :=
by
  -- Definitions
  let quarter_weight := 1 / 5
  let quarter_value := 0.25
  let value_per_ounce := 100
  let quarters_per_ounce := 1 / quarter_weight
  let melt_value := value_per_ounce
  let spend_value := quarters_per_ounce * quarter_value

  -- Conclusion to be proven
  have h1 : quarters_per_ounce = 5 := sorry
  have h2 : spend_value = 1.25 := sorry
  have h3 : melt_value / spend_value = 80 := sorry

  show melt_value / spend_value = 80 from h3

end carlos_gold_quarters_l604_604738


namespace correct_sum_is_826_l604_604650

theorem correct_sum_is_826 (ABC : ℕ)
  (h1 : 100 ≤ ABC ∧ ABC < 1000)  -- Ensuring ABC is a three-digit number
  (h2 : ∃ A B C : ℕ, ABC = 100 * A + 10 * B + C ∧ C = 6) -- Misread ones digit is 6
  (incorrect_sum : ℕ)
  (h3 : incorrect_sum = ABC + 57)  -- Sum obtained by Yoongi was 823
  (h4 : incorrect_sum = 823) : ABC + 57 + 3 = 826 :=  -- Correcting the sum considering the 6 to 9 error
by
  sorry

end correct_sum_is_826_l604_604650


namespace chemical_masses_correct_l604_604820

theorem chemical_masses_correct :
  let BaF2_molar_mass := 175.33
  let Na2SO4_molar_mass := 142.05
  let KOH_molar_mass := 56.11
  let moles_BaSO4 := 4
  (moles_BaSO4 * BaF2_molar_mass = 701.32) ∧
  (moles_BaSO4 * Na2SO4_molar_mass = 568.20) ∧
  (2 * moles_BaSO4 * KOH_molar_mass = 448.88) :=
by {
  have h1 : moles_BaSO4 * BaF2_molar_mass = 701.32 := by sorry,
  have h2 : moles_BaSO4 * Na2SO4_molar_mass = 568.20 := by sorry,
  have h3 : 2 * moles_BaSO4 * KOH_molar_mass = 448.88 := by sorry,
  exact ⟨h1, h2, h3⟩
} 

end chemical_masses_correct_l604_604820


namespace limit_proof_l604_604169

theorem limit_proof :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → | (3 * x^2 - 5 * x - 2) / (x - 2) - 7 | < ε) :=
begin
  assume ε ε_pos,
  existsi ε / 3,
  use div_pos ε_pos (by norm_num),
  assume x hx,
  have h_denom : x ≠ 2, from λ h, hx.1 (by rwa h),
  calc
  |(3 * x^2 - 5 * x - 2) / (x - 2) - 7|
      = |((3 * x + 1) * (x - 2) / (x - 2)) - 7| : by {
        ring_exp,
        exact (by norm_num : (3 : ℝ) ≠ 0 ),
      }
  ... = |3 * x + 1 - 7| : by rw [(by_ring_exp * (x - 2)).symm], (ne_of_apply_ne (3 * x + 1) (by norm_num)).elim]
  ... = |3 * (x - 2)| : by ring_exp
  ... = 3 * |x - 2| : abs_mul,
  exact (mul_lt_iff_lt_one_left (by norm_num)).2 (calc
      |x - 2| < ε / 3 : hx.2
      ... = ε / 3
    )
end

end limit_proof_l604_604169


namespace female_students_in_sample_l604_604334

-- Definitions of the given conditions
def male_students : ℕ := 28
def female_students : ℕ := 21
def total_students : ℕ := male_students + female_students
def sample_size : ℕ := 14
def stratified_sampling_fraction : ℚ := (sample_size : ℚ) / (total_students : ℚ)
def female_sample_count : ℚ := stratified_sampling_fraction * (female_students : ℚ)

-- The theorem to prove
theorem female_students_in_sample : female_sample_count = 6 :=
by
  sorry

end female_students_in_sample_l604_604334


namespace proof_A2_less_than_3A1_plus_n_l604_604614

-- Define the conditions in terms of n, A1, and A2.
variables (n : ℕ)

-- A1 and A2 are the numbers of selections to select two students
-- such that their weight difference is ≤ 1 kg and ≤ 2 kg respectively.
variables (A1 A2 : ℕ)

-- The main theorem needs to prove that A2 < 3 * A1 + n.
theorem proof_A2_less_than_3A1_plus_n (h : A2 < 3 * A1 + n) : A2 < 3 * A1 + n :=
by {
  sorry -- proof goes here, but it's not required for the Lean statement.
}

end proof_A2_less_than_3A1_plus_n_l604_604614


namespace find_k_range_l604_604009

theorem find_k_range (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ -6 < k ∧ k < -2 :=
by
  sorry

end find_k_range_l604_604009


namespace required_value_l604_604086

noncomputable def f : ℕ → ℝ := sorry

axiom functional_property (n : ℕ) (h : n > 1) : 
  ∃ p : ℕ, nat.prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

axiom given_condition : 
  f (2^2007) + f (3^2008) + f (5^2009) = 2006

theorem required_value :
  f (2007^2) + f (2008^3) + f (2009^5) = 9 := sorry

end required_value_l604_604086


namespace intersection_of_sets_l604_604111

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l604_604111


namespace surface_area_DABC_l604_604133

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

def triangle_area (a b c : ℝ) (h : (a^2 + c^2 = b^2)) : ℝ :=
  let altitude := sqrt (((b / 2) ^ 2 - (c / 2) ^ 2));
  (1 / 2) * c * altitude

def pyramid_surface_area (a b c : ℝ) (h : (a^2 + c^2 = b^2)) : ℝ :=
  4 * triangle_area a b c h

theorem surface_area_DABC :
  let a := 20
  let b := 45
  let c := 45
  let h : (20^2 + (sqrt 1925)^2 = 45^2) := by sorry
  pyramid_surface_area a b c h = 1754.8 :=
by
  let a := 20
  let b := 45
  let c := 45
  have h : (20^2 + (sqrt 1925)^2 = 45^2) := by sorry
  have area := pyramid_surface_area a b c h
  sorry

end surface_area_DABC_l604_604133


namespace angle_PQR_is_90_l604_604931

theorem angle_PQR_is_90 {P Q R S : Type}
  (is_straight_line_RSP : ∃ P R S : Type, (angle R S P = 180)) 
  (angle_QSP : angle Q S P = 70)
  (isosceles_RS_SQ : ∃ (RS SQ : Type), RS = SQ)
  (isosceles_PS_SQ : ∃ (PS SQ : Type), PS = SQ) : angle P Q R = 90 :=
by 
  sorry

end angle_PQR_is_90_l604_604931


namespace origin_inside_ellipse_iff_abs_k_range_l604_604430

theorem origin_inside_ellipse_iff_abs_k_range (k : ℝ) :
  (k^2 * 0^2 + 0^2 - 4 * k * 0 + 2 * k * 0 + k^2 - 1 < 0) ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end origin_inside_ellipse_iff_abs_k_range_l604_604430


namespace evaluate_f_f0_plus_2_l604_604438

def f (x : ℝ) : ℝ :=
  if x < 1 then 2 * x - 1 else 1 + Real.log x / Real.log 2

theorem evaluate_f_f0_plus_2 : 
  f (f 0 + 2) = 1 :=
by
  sorry

end evaluate_f_f0_plus_2_l604_604438


namespace students_remaining_after_stops_l604_604886

theorem students_remaining_after_stops :
  let initial_students := 60
  let first_stop := 1 / 3 * initial_students
  let students_after_first_stop := initial_students - first_stop
  let second_stop := 1 / 4 * students_after_first_stop
  let students_after_second_stop := students_after_first_stop - second_stop
  let third_stop := 1 / 5 * students_after_second_stop
  let remaining_students := students_after_second_stop - third_stop
  remaining_students = 24 := by
  let initial_students := 60
  let first_stop := 1 / 3 * initial_students
  let students_after_first_stop := initial_students - first_stop
  let second_stop := 1 / 4 * students_after_first_stop
  let students_after_second_stop := students_after_first_stop - second_stop
  let third_stop := 1 / 5 * students_after_second_stop
  let remaining_students := students_after_second_stop - third_stop
  show remaining_students = 24 by sorry

end students_remaining_after_stops_l604_604886


namespace segment_FL_through_center_of_inscribed_sphere_l604_604711

-- You may define the geometry setup as follows:

structure Tetrahedron (V : Type) :=
(A : V) (B : V) (C : V) (D : V)

structure Sphere (V : Type) :=
(center : V) (radius : ℝ)

def touches_plane {V : Type} [MetricSpace V] (s : Sphere V) (plane : Set V) : Prop := sorry

theorem segment_FL_through_center_of_inscribed_sphere {V : Type} [MetricSpace V]
  (A B C D E F L : V) (S : Sphere V) (T : Tetrahedron V)
  (H1 : T.A = A) (H2 : T.B = B) (H3 : T.C = C) (H4 : T.D = D)
  (H5 : touches_plane S {p : V | T.A = A ∧ T.B = B ∧ T.D = D})
  (H6 : touches_plane S {p : V | T.A = A ∧ T.C = C ∧ T.D = D})
  (H7 : touches_plane S {p : V | T.B = B ∧ T.C = C ∧ T.D = D})
  (H8 : ¬ touches_plane S {p : V | T.A = A ∧ T.B = B ∧ T.C = C})
  (H9 : touches_plane S {p : V | T.A = A ∧ T.B = B ∧ E = E})
  (H10 : touches_plane S {p : V | T.A = A ∧ T.C = C ∧ E = E})
  (H11 : touches_plane S {p : V | T.B = B ∧ T.C = C ∧ E = E})
  (H12 : {p : V | T.D = p ∧ E = E} ∩ {p : V | T.A = A ∧ T.B = B ∧ T.C = C} = {F})
  (H13 : ∀ p : V, MetricSpace.dist p {p : V | T.A = A ∧ T.B = B ∧ T.C = C} = L):
  line_through F L (center_of_inscribed_sphere_of_tetrahedron T E) :=
sorry

end segment_FL_through_center_of_inscribed_sphere_l604_604711


namespace min_value_expression_l604_604771

theorem min_value_expression (x : ℝ) : 
  (\frac{x^2 + 9}{real_sqrt (x^2 + 5)} ≥ \frac{9 * real_sqrt 5}{5}) :=
sorry

end min_value_expression_l604_604771


namespace eval_nested_function_calls_l604_604962

def f (x : ℝ) : ℝ := x^2 - 2*x

theorem eval_nested_function_calls : f(f(f(f(f(f(-1)))))) = 3 := 
by
  sorry

end eval_nested_function_calls_l604_604962


namespace max_sin_product_tan_eq_one_l604_604425

theorem max_sin_product_tan_eq_one (n : ℕ) (A : Fin n → ℝ) :
  (∏ i, Real.tan (A i) = 1) → (∏ i, Real.sin (A i) ≤ 2^(- (n / 2))) :=
sorry

end max_sin_product_tan_eq_one_l604_604425


namespace f_is_constant_function_l604_604572

noncomputable theory

open_locale classical

variables {R : Type} [normed_linear_ordered_field R]

theorem f_is_constant_function
  {f : R → R}
  (a : R)
  (ha : a > 0)
  (hf : ∀ x y, f(x) * f(y) + f(a / x) * f(a / y) = 2 * f(x * y))
  (hfa : f(a) = 1) :
  ∀ x, f(x) = 1 :=
begin
  sorry
end

end f_is_constant_function_l604_604572


namespace probability_same_color_plates_l604_604853

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l604_604853


namespace correct_statements_l604_604753

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) : f (x + 2) + f x = 0
axiom f_odd (x : ℝ) : f (x + 1) = -f (1 - x)

lemma f_period (x : ℝ) : f (x + 4) = f x := 
begin
  -- Proving that f has a period of 4
  sorry
end

lemma f_symmetric (x : ℝ) : f (2 - x) = -f x :=
begin
  -- Showing that f is symmetric about the point (1, 0)
  sorry
end

theorem correct_statements :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 4) ∧ (∀ x, f(1 - x) = -f(1 + x)) :=
begin
  -- Proving the correct options ① and ②
  sorry
end

end correct_statements_l604_604753


namespace find_cos_value_l604_604802

open Real

noncomputable def cos_value (α : ℝ) : ℝ :=
  cos (2 * π / 3 + 2 * α)

theorem find_cos_value (α : ℝ) (h : sin (π / 6 - α) = 1 / 4) :
  cos_value α = -7 / 8 :=
sorry

end find_cos_value_l604_604802


namespace quadratic_has_distinct_real_roots_l604_604249

theorem quadratic_has_distinct_real_roots :
  ∀ (x : ℝ), x^2 - 2 * x - 1 = 0 → (∃ Δ > 0, Δ = ((-2)^2 - 4 * 1 * (-1))) := by
  sorry

end quadratic_has_distinct_real_roots_l604_604249


namespace abs_diff_of_two_numbers_l604_604612

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 160) : |x - y| = 2 * Real.sqrt 65 :=
by
  sorry

end abs_diff_of_two_numbers_l604_604612


namespace simplify_expression_l604_604206

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 * z^2 + 2 * z - 5) = 8 - 9 * z^2 - 2 * z :=
by
  sorry

end simplify_expression_l604_604206


namespace inequality_proof_l604_604806

variable (m n : ℝ)

theorem inequality_proof (hm : m < 0) (hn : n > 0) (h_sum : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by
  -- introduction and proof commands would go here, but we use sorry to indicate the proof is omitted
  sorry

end inequality_proof_l604_604806


namespace monomial_same_type_m_n_sum_l604_604897

theorem monomial_same_type_m_n_sum (m n : ℕ) (x y : ℤ) 
  (h1 : 2 * x ^ (m - 1) * y ^ 2 = 1/3 * x ^ 2 * y ^ (n + 1)) : 
  m + n = 4 := 
sorry

end monomial_same_type_m_n_sum_l604_604897


namespace range_of_a_for_monotonicity_l604_604003

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x + a) / Real.cos x

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x ∈ Ioo 0 (Real.pi / 2), differentiable (f x a) ∧ 
    (∀ y ∈ Ioo 0 (Real.pi / 2), deriv (f y a) ≥ 0)) →
  a ≥ -1 :=
sorry

end range_of_a_for_monotonicity_l604_604003


namespace tan_cot_eq_solutions_count_l604_604387

theorem tan_cot_eq_solutions_count :
  (∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ tan (3 * π * cos θ) = cot (3 * π * sin θ)) :=
  sorry

end tan_cot_eq_solutions_count_l604_604387


namespace quadratic_eq_has_two_distinct_real_roots_l604_604604

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Problem statement: Prove that the quadratic equation x^2 + 3x - 2 = 0 has two distinct real roots
theorem quadratic_eq_has_two_distinct_real_roots :
  discriminant 1 3 (-2) > 0 :=
by
  -- Proof goes here
  sorry

end quadratic_eq_has_two_distinct_real_roots_l604_604604


namespace percentage_increase_in_efficiency_l604_604559

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l604_604559


namespace isosceles_triangle_k_value_l604_604038

theorem isosceles_triangle_k_value (k : ℕ) :
  (∃ x : ℝ, x^2 - 12 * x + k = 0 ∧ x = 6) →
  (∃ a b : ℝ, a = 3 ∧ b = 3 ∧ √(a^2 + b^2) = 6) →
  k = 36 :=
by
  sorry

end isosceles_triangle_k_value_l604_604038


namespace total_books_sold_amount_l604_604053

def num_fiction_books := 60
def num_non_fiction_books := 84
def num_children_books := 42

def fiction_books_sold := 3 / 4 * num_fiction_books
def non_fiction_books_sold := 5 / 6 * num_non_fiction_books
def children_books_sold := 2 / 3 * num_children_books

def price_fiction := 5
def price_non_fiction := 7
def price_children := 3

def total_amount_fiction := fiction_books_sold * price_fiction
def total_amount_non_fiction := non_fiction_books_sold * price_non_fiction
def total_amount_children := children_books_sold * price_children

def total_amount_received := total_amount_fiction + total_amount_non_fiction + total_amount_children

theorem total_books_sold_amount :
  total_amount_received = 799 :=
sorry

end total_books_sold_amount_l604_604053


namespace bound_on_clubs_l604_604059

theorem bound_on_clubs
  (S : Finset ℕ) (n : ℕ)
  (hS_card : S.card = 9)
  (clubs : Finset (Finset ℕ))
  (h_club_card : ∀ c ∈ clubs, c.card = 4)
  (h_club_pair : ∀ c1 c2 ∈ clubs, c1 ≠ c2 → (c1 ∩ c2).card ≤ 2) :
  clubs.card ≤ 18 := 
sorry

end bound_on_clubs_l604_604059


namespace area_intersection_l604_604836

noncomputable def set_M : Set ℂ := {z : ℂ | abs (z - 1) ≤ 1}
noncomputable def set_N : Set ℂ := {z : ℂ | complex.arg z ≥ π / 4}

theorem area_intersection (S : ℝ) :
  S = (3 / 4) * real.pi - 1 / 2 →
  ∃ z : ℂ, z ∈ set_M ∩ set_N := 
sorry

end area_intersection_l604_604836


namespace inscribed_circle_radius_integer_l604_604193

theorem inscribed_circle_radius_integer 
  (a b c : ℕ) (h : a^2 + b^2 = c^2) 
  (h₀ : 2 * (a + b - c) = k) 
  : ∃ (r : ℕ), r = (a + b - c) / 2 := 
begin
  sorry
end

end inscribed_circle_radius_integer_l604_604193


namespace range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l604_604012

variable (m : ℝ)

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Problem 1: Prove the range of m if B ⊆ A is (-∞, 3]
theorem range_m_if_B_subset_A : (set_B m ⊆ set_A) ↔ m ≤ 3 := sorry

-- Problem 2: Prove the range of m if A ∩ B = ∅ is m < 2 or m > 4
theorem range_m_if_A_inter_B_empty : (set_A ∩ set_B m = ∅) ↔ m < 2 ∨ m > 4 := sorry

end range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l604_604012


namespace trapezoid_shorter_base_l604_604591

theorem trapezoid_shorter_base (L : ℝ) (a b : ℝ) (h : L = 5) (h_longer_base : a = 105) : b = 95 :=
by
  have : (a - b) / 2 = L, from sorry,
  have : a - b = 2 * L, from sorry,
  have : b = a - 2 * L, from sorry,
  have : b = 105 - 10, from sorry,
  have : b = 95, from sorry,
  exact this

end trapezoid_shorter_base_l604_604591


namespace max_min_ratio_l604_604514

def f (x : ℝ) : ℝ := 2 * x / (x - 2)

theorem max_min_ratio (M m : ℝ) (hM : M = f(3)) (hm : m = f(4)) : 
  (m ^ 2 / M) = 8 / 3 :=
by
  have h1 : f(3) = 6 := sorry
  have h2 : f(4) = 4 := sorry
  rw [h1, h2, hM, hm]
  norm_num
  exact sorry

end max_min_ratio_l604_604514


namespace total_books_sold_amount_l604_604054

def num_fiction_books := 60
def num_non_fiction_books := 84
def num_children_books := 42

def fiction_books_sold := 3 / 4 * num_fiction_books
def non_fiction_books_sold := 5 / 6 * num_non_fiction_books
def children_books_sold := 2 / 3 * num_children_books

def price_fiction := 5
def price_non_fiction := 7
def price_children := 3

def total_amount_fiction := fiction_books_sold * price_fiction
def total_amount_non_fiction := non_fiction_books_sold * price_non_fiction
def total_amount_children := children_books_sold * price_children

def total_amount_received := total_amount_fiction + total_amount_non_fiction + total_amount_children

theorem total_books_sold_amount :
  total_amount_received = 799 :=
sorry

end total_books_sold_amount_l604_604054


namespace Roger_first_bag_candies_is_11_l604_604987

-- Define the conditions
def Sandra_bags : ℕ := 2
def Sandra_candies_per_bag : ℕ := 6
def Roger_bags : ℕ := 2
def Roger_second_bag_candies : ℕ := 3
def Extra_candies_Roger_has_than_Sandra : ℕ := 2

-- Define the total candy for Sandra
def Sandra_total_candies : ℕ := Sandra_bags * Sandra_candies_per_bag

-- Using the conditions, we define the total candy for Roger
def Roger_total_candies : ℕ := Sandra_total_candies + Extra_candies_Roger_has_than_Sandra

-- Define the candy in Roger's first bag
def Roger_first_bag_candies : ℕ := Roger_total_candies - Roger_second_bag_candies

-- The proof statement we need to prove
theorem Roger_first_bag_candies_is_11 : Roger_first_bag_candies = 11 := by
  sorry

end Roger_first_bag_candies_is_11_l604_604987


namespace probability_of_point_within_smallest_circle_l604_604750

theorem probability_of_point_within_smallest_circle (r1 r2 r3 : ℝ) (h : r1 < r2 ∧ r2 < r3) (hc : r1 = 1 ∧ r2 = 2 ∧ r3 = 3) : 
  let area_smallest := Real.pi * r1^2,
      area_largest  := Real.pi * r3^2 in
  (area_smallest / area_largest) = 1 / 9 := 
by
  have h1 : area_smallest = Real.pi 1^2, from sorry,
  have h2 : area_largest = Real.pi 3^2, from sorry,
  have h3 : area_smallest = Real.pi, from sorry,
  have h4 : area_largest = 9 * Real.pi, from sorry,
  calc
    (area_smallest / area_largest) = (Real.pi / (9 * Real.pi)) : by sorry
                                 ... = (1 / 9)                   : by sorry

end probability_of_point_within_smallest_circle_l604_604750


namespace sum_bn_101_l604_604066

variable {a b S101 : ℕ → ℤ} (a4 a7 : ℤ)
variable {d : ℤ} (h1 : a (4) = 5) (h2 : a (7) = 11)

def common_difference (a b : ℕ → ℤ) : ℤ := d

def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

def bn_sequence (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = (-1)^n * a n

def sum_bn (S101 b : ℕ → ℤ) : Prop :=
  S101 = ∑ n in finset.range 101, b n

theorem sum_bn_101 (a b : ℕ → ℤ) (a1 d : ℤ) (S101 : ℤ) :
  a (4) = 5 → a (7) = 11 →
  (∀ n, a n = a1 + (n - 1) * d) →
  (∀ n, b n = (-1)^n * a n) →
  S101 = ∑ n in finset.range 101, b n → 
  S101 = -99 :=
by
  intros h1 h2 ha hb hs
  -- proof steps would go here
  sorry

end sum_bn_101_l604_604066


namespace simplified_expression_is_valid_l604_604205

noncomputable def simplify_expression (a : ℝ) : ℝ :=
  (1 + (3 / (a - 1))) / ((a^2 - 4) / (a - 1))

theorem simplified_expression_is_valid (a : ℝ) (h : a ∈ {-1, 0, 1, 2}) (h_def : a ≠ 1 ∧ a ≠ 2): 
  simplify_expression a = 1 / (a - 2) :=
by
  sorry

end simplified_expression_is_valid_l604_604205


namespace limit_tg_ln_sine_l604_604746

theorem limit_tg_ln_sine (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = tan x) → 
  tendsto (λ x, (f x - tan 2) / (sin (log (x - 1)))) (nhds 2) (nhds (1 / (cos 2)^2)) :=
by
  sorry

end limit_tg_ln_sine_l604_604746


namespace sum_of_digits_l604_604292

theorem sum_of_digits (n : ℕ) (h : 0 < n) : 
  let x := (10^(4 * n^2 + 8) + 1)^2 in 
  (x.digits 10).sum = 4 := 
by 
  sorry

end sum_of_digits_l604_604292


namespace intersection_A_B_eq_C_l604_604103

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l604_604103


namespace card_sum_odd_ways_l604_604260

theorem card_sum_odd_ways : 
  let cards := [1, 2, 3, 4]
  (∃ (f : (ℕ × ℕ) → bool),
    (∀ (x y : ℕ), x ∈ cards → y ∈ cards → x ≠ y → f (x, y) = (x + y) % 2 = 1) ∧
    (f (1, 2) ∧ f (1, 4) ∧ f (2, 3) ∧ f (3, 4)) ∧
    (f (1, 3) = false ∧ f (2, 4) = false)) →
  4 :=
by
  sorry

end card_sum_odd_ways_l604_604260


namespace maximum_xy_l604_604518

noncomputable def max_xy_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4*x + 9*y = 60) : ℝ :=
  xy

theorem maximum_xy : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ 4*x + 9*y = 60 ∧ xy = 25 :=
by
  sorry

end maximum_xy_l604_604518


namespace max_range_of_temps_l604_604998

-- Define the variables and constants
variable {temps : Fin 5 → ℝ} -- temps is a function from {0,...,4} corresponding to the days
variable (average : ℝ := 25)
variable (lowest : ℝ := 15)

-- Define the conditions
def average_condition : Prop :=
  (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = average

def lowest_condition : Prop :=
  ∃ i : Fin 5, temps i = lowest ∧ ∀ j : Fin 5, lowest ≤ temps j

-- State the theorem
theorem max_range_of_temps :
  average_condition temps → lowest_condition temps → 
  ∃ max_range : ℝ, max_range = 50 :=
by
  sorry

end max_range_of_temps_l604_604998


namespace right_triangle_inscribed_circle_probability_l604_604915

theorem right_triangle_inscribed_circle_probability (a b : ℝ) (ha : a = 8) (hb : b = 15) :
  let c := real.sqrt (a^2 + b^2),
      r := 2 * (a * b) / (a + b + c),
      probability := (π * r^2) / (1/2 * a * b)
  in c = 17 ∧ r = 3 ∧ probability = (3 * π) / 20 := 
by
  have c_eq : c = 17 := by 
    rw [ha, hb]
    exact real.sqrt_eq_rfl_iff_eq_of_nonneg (by norm_num) (by {norm_num})
  have r_eq : r = 3 := by 
    rw [c_eq]
    exact by 
      simp only [ha, hb]
      norm_num
  have prob_eq : probability = (3 * π) / 20 := by 
    rw [r_eq]
    simp only [ha, hb]
    norm_num
  exact ⟨c_eq, r_eq, prob_eq⟩

end right_triangle_inscribed_circle_probability_l604_604915


namespace chicken_nuggets_cost_l604_604535

theorem chicken_nuggets_cost :
  ∀ (nuggets_ordered boxes_cost : ℕ) (nuggets_per_box : ℕ),
  nuggets_ordered = 100 →
  nuggets_per_box = 20 →
  boxes_cost = 4 →
  (nuggets_ordered / nuggets_per_box) * boxes_cost = 20 :=
by
  intros nuggets_ordered boxes_cost nuggets_per_box h1 h2 h3
  sorry

end chicken_nuggets_cost_l604_604535


namespace urn_problem_l604_604720

noncomputable def count_balls (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) : ℕ :=
initial_white + initial_black + operations

noncomputable def urn_probability (initial_white : ℕ) (initial_black : ℕ) (operations : ℕ) (final_white : ℕ) (final_black : ℕ) : ℚ :=
if final_white + final_black = count_balls initial_white initial_black operations &&
   final_white = (initial_white + (operations - (final_black - initial_black))) &&
   (final_white + final_black) = 8 then 3 / 5 else 0

theorem urn_problem :
  let initial_white := 2
  let initial_black := 1
  let operations := 4
  let final_white := 4
  let final_black := 4
  count_balls initial_white initial_black operations = 8 ∧ urn_probability initial_white initial_black operations final_white final_black = 3 / 5 :=
by
  sorry

end urn_problem_l604_604720


namespace find_A_l604_604242

theorem find_A (
  A B C A' r : ℕ
) (hA : A = 312) (hB : B = 270) (hC : C = 211)
  (hremA : A % A' = 4 * r)
  (hremB : B % A' = 2 * r)
  (hremC : C % A' = r) :
  A' = 19 :=
by
  sorry

end find_A_l604_604242


namespace cos_B_geq_three_fourths_measure_angle_B_l604_604489

noncomputable def triangle_ABC (a b c : ℝ) : Prop := 
  ∃ (A B C : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧ 
    sin B = b / sqrt(a^2 + b^2 + c^2) ∧
    sin A = a / sqrt(a^2 + b^2 + c^2) ∧
    sin C = c / sqrt(a^2 + b^2 + c^2)

def condition (a b c : ℝ) := a * c = 2 * b^2

theorem cos_B_geq_three_fourths (a b c : ℝ) (h : condition a b c) (h_triangle : triangle_ABC a b c) :
  ∃ (B : ℝ), cos B ≥ 3 / 4 := by
  sorry

theorem measure_angle_B (a b c : ℝ) (h : condition a b c) (h_triangle : triangle_ABC a b c) (h_cos : cos (A - C) + cos B = 1) :
  ∃ (B : ℝ), B = π / 6 := by
  sorry

end cos_B_geq_three_fourths_measure_angle_B_l604_604489


namespace largest_bundle_size_l604_604501

theorem largest_bundle_size (n > 5) (h : ∀ k : ℕ, k ∣ 36 ∧ k ∣ 45 → k ≤ n → k = 9) : n = 9 :=
sorry

end largest_bundle_size_l604_604501


namespace min_possible_value_box_l604_604037

theorem min_possible_value_box (a b : ℤ) (h_ab : a * b = 35) : a^2 + b^2 ≥ 74 := sorry

end min_possible_value_box_l604_604037


namespace find_distance_EF_l604_604950

variable {A B F : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace F]

-- Define the points and distances in our metric space
variables (ABF : Triangle A B F)
variable (ABCD : Square A B)
variable (E F : Point F)
variable (FA FB : ℝ)
variable (angle_AFB : ℝ)

-- Given conditions
axiom right_angle_AFB (ABF : Triangle A B F) : angle_AFB = π / 2
axiom side_FA : FA = 6
axiom side_FB : FB = 8
axiom circumcenter_E (E : Point) (ABCD : Square A B C D) : E = circumcenter ABCD

-- Statement we want to prove
theorem find_distance_EF : distance E F = 7 * sqrt 2 := by sorry

end find_distance_EF_l604_604950


namespace jose_work_time_l604_604944

-- Define the variables for days taken by Jose and Raju
variables (J R T : ℕ)

-- State the conditions:
-- 1. Raju completes work in 40 days
-- 2. Together, Jose and Raju complete work in 8 days
axiom ra_work : R = 40
axiom together_work : T = 8

-- State the theorem that needs to be proven:
theorem jose_work_time (J R T : ℕ) (h1 : R = 40) (h2 : T = 8) : J = 10 :=
sorry

end jose_work_time_l604_604944


namespace cube_root_of_sqrt_64_l604_604219

variable (a : ℕ) (b : ℕ)

def cubeRootOfSqrt64 : Prop :=
  a = Nat.sqrt 64 ∧ b = Nat.cbrt a → b = 2

theorem cube_root_of_sqrt_64 (a : ℕ) (b : ℕ) : cubeRootOfSqrt64 a b :=
  by
  sorry

end cube_root_of_sqrt_64_l604_604219


namespace even_ln_function_a_zero_l604_604040

theorem even_ln_function_a_zero {a : ℝ} (h_even : ∀ x, ln (x^2 + a * x + 1) = ln (x^2 - a * x + 1)) : a = 0 :=
by
  sorry

end even_ln_function_a_zero_l604_604040


namespace find_radius_of_large_circle_l604_604270

noncomputable def radius_of_large_circle (r : ℝ) : Prop :=
  let r_A := 3
  let r_B := 2
  let d := 6
  (r - r_A)^2 + (r - r_B)^2 + 2 * (r - r_A) * (r - r_B) = d^2 ∧
  r = (5 + Real.sqrt 33) / 2

theorem find_radius_of_large_circle : ∃ (r : ℝ), radius_of_large_circle r :=
by {
  sorry
}

end find_radius_of_large_circle_l604_604270


namespace distance_AC_not_unique_l604_604464

-- We define points A, B, C and their segments
variables {A B C : Type} [MetricSpace A]

-- Defining the lengths of segments
def AB : ℝ := 4
def BC : ℝ := 3

-- Goal statement expressing the non-uniqueness of distance between A and C
theorem distance_AC_not_unique : ∃ (d : ℝ), d ∈ {1, 7} ∨ (∀ ε > 0, ∃ C1 C2 : A, dist A C1 ≠ dist A C2) :=
sorry

end distance_AC_not_unique_l604_604464


namespace problem1_problem2_l604_604822

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x > 1 then 1 + 1 / x
  else if x >= -1 then x^2 + 1
  else 2 * x + 3

-- Prove f(f(f(-2))) = 3 / 2
theorem problem1 : f (f (f (-2))) = 3 / 2 := by
  sorry

-- Prove that if f(a) = 3 / 2, then a = 2 or a = ±√2/2
theorem problem2 (a : ℝ) (h : f a = 3 / 2) : a = 2 ∨ a = sqrt 2 / 2 ∨ a = -sqrt 2 / 2 := by
  sorry

end problem1_problem2_l604_604822


namespace polynomial_is_fifth_degree_trinomial_l604_604296

def degree (p : Polynomial (ℚ × ℚ)) : ℕ := sorry

noncomputable def is_trinomial (p : Polynomial (ℚ × ℚ)) : Prop := sorry

theorem polynomial_is_fifth_degree_trinomial : 
  let p := (Polynomial.C 1) + (Polynomial.C (4 : ℚ) - Polynomial.X)
    + (Polynomial.C (1 : ℚ) - Polynomial.X ^ 2 * Polynomial.Y ^ 3)
  in degree p = 5 ∧ is_trinomial p := sorry

end polynomial_is_fifth_degree_trinomial_l604_604296


namespace nat_representable_as_sequence_or_difference_l604_604157

theorem nat_representable_as_sequence_or_difference
  (a : ℕ → ℕ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∀ n, a n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, k ≠ l ∧ (m = a k ∨ m = a k - a l) :=
by
  sorry

end nat_representable_as_sequence_or_difference_l604_604157


namespace pat_ready_for_college_after_17_5_years_l604_604889

noncomputable def doubling_time (r : ℝ) : ℝ := 70 / r

noncomputable def doubled_amounts (initial : ℝ) (final : ℝ) : ℕ := 
  ⌊real.log (final / initial) / real.log 2⌋.nat_abs

theorem pat_ready_for_college_after_17_5_years 
  (initial_investment final_investment : ℝ) (r : ℝ) 
  (h_initial : initial_investment = 7000) (h_final : final_investment = 28000) 
  (h_r : r = 8) :
  real.log (final_investment / initial_investment) / real.log 2 * doubling_time r = 17.5 :=
by
  sorry

end pat_ready_for_college_after_17_5_years_l604_604889


namespace correct_option_is_d_l604_604882

theorem correct_option_is_d (x : ℚ) : -x^3 = (-x)^3 :=
sorry

end correct_option_is_d_l604_604882


namespace quintic_polynomial_p_l604_604883

theorem quintic_polynomial_p (p q : ℝ) (h : (∀ x : ℝ, x^p + 4*x^3 - q*x^2 - 2*x + 5 = (x^5 + 4*x^3 - q*x^2 - 2*x + 5))) : -p = -5 :=
by {
  sorry
}

end quintic_polynomial_p_l604_604883


namespace correct_statement_C_l604_604645

/-- Definition of a power function y = x^a --/
def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

/-- Definition of an exponential function, typically y = a^x for some base a > 0 --/
def exponential_function (a : ℝ) (x : ℝ) [fact (0 < a)] : ℝ := a ^ x

/-- Definition of a logarithmic function, typically y = log_a(x) for some base a > 0 --/
def logarithmic_function (a : ℝ) (x : ℝ) [fact (0 < a)] [fact (1 ≠ a)] : ℝ := log a x

/-- Proof that option C is the correct statement --/
theorem correct_statement_C : 
  (∀ a x, a < 0 → ¬ (power_function a x = 0 ∨ (0, 0) ∈ (λ p, power_function p.1 p.2) '' univ)) ∧
  (∀ a x, (∃ a > 0, exponential_function a 0 = 1)) ∧
  (∀ a x, ∀ (a > 0) (a ≠ 1), logarithmic_function a x > 0 → (0, x) ∉ (λ p, logarithmic_function p.1 p.2) '' univ) ∧
  (∀ a x, (∀ a, power_function a x > 0) ∨ (∃ a, power_function a x < 0 → power_function a x ∉ (λ p, power_function p.1 p.2) '' univ))
  → (λ p, logarithmic_function p.1 p.2) '' univ
by 
  sorry

end correct_statement_C_l604_604645


namespace deepak_current_age_l604_604310

theorem deepak_current_age (x : ℕ) (rahul_age deepak_age : ℕ) :
  (rahul_age = 4 * x) →
  (deepak_age = 3 * x) →
  (rahul_age + 10 = 26) →
  deepak_age = 12 :=
by
  intros h1 h2 h3
  -- You would write the proof here
  sorry

end deepak_current_age_l604_604310


namespace no_set_with_7_elements_min_elements_condition_l604_604658

noncomputable def set_a_elements := 7
noncomputable def median_a := 10
noncomputable def mean_a := 6
noncomputable def min_sum_a := 3 + 4 * 10
noncomputable def real_sum_a := mean_a * set_a_elements

theorem no_set_with_7_elements : ¬ (set_a_elements = 7 ∧
  (∃ S : Finset ℝ, 
    (S.card = set_a_elements) ∧ 
    (S.sum ≥ min_sum_a) ∧ 
    (S.sum = real_sum_a))) := 
by
  sorry

noncomputable def n_b_elements := ℕ
noncomputable def set_b_elements (n : ℕ) := 2 * n + 1
noncomputable def median_b := 10
noncomputable def mean_b := 6
noncomputable def min_sum_b (n : ℕ) := n + 10 * (n + 1)
noncomputable def real_sum_b (n : ℕ) := mean_b * set_b_elements n

theorem min_elements_condition (n : ℕ) : 
    (∀ n : ℕ, n ≥ 4) → 
    (set_b_elements n ≥ 9 ∧
        ∃ S : Finset ℝ, 
          (S.card = set_b_elements n) ∧ 
          (S.sum ≥ min_sum_b n) ∧ 
          (S.sum = real_sum_b n)) :=
by
  assume h : ∀ n : ℕ, n ≥ 4
  sorry

end no_set_with_7_elements_min_elements_condition_l604_604658


namespace other_diagonal_length_l604_604979

theorem other_diagonal_length (d2 : ℝ) (A : ℝ) (d1 : ℝ) 
  (h1 : d2 = 120) 
  (h2 : A = 4800) 
  (h3 : A = (d1 * d2) / 2) : d1 = 80 :=
by
  sorry

end other_diagonal_length_l604_604979


namespace problem_statement_l604_604815

-- Define the given conditions
def P : ℝ × ℝ := (4/5, -3/5)
def α : ℝ 

-- Lemmas and theorems to be proven
theorem problem_statement (h : P = (Real.cos α, Real.sin α)) :
  Real.cos α = 4/5 ∧
  Real.tan α = -3/4 ∧
  Real.sin (α + Real.pi) = 3/5 :=
by
  sorry

end problem_statement_l604_604815


namespace greatest_drop_in_price_is_May_l604_604590

def priceChangeJan := -1.25
def priceChangeFeb := 2.75
def priceChangeMar := -0.75
def priceChangeApr := 1.50
def priceChangeMay := -3.00
def priceChangeJun := -1.00

theorem greatest_drop_in_price_is_May :
  priceChangeMay < priceChangeJan ∧
  priceChangeMay < priceChangeMar ∧
  priceChangeMay < priceChangeApr ∧
  priceChangeMay < priceChangeJun ∧
  priceChangeMay < priceChangeFeb :=
by sorry

end greatest_drop_in_price_is_May_l604_604590


namespace moon_speed_in_km_per_sec_l604_604596

theorem moon_speed_in_km_per_sec (speed_in_kmh : ℕ) (seconds_in_hour : ℤ) (conversion_factor : speed_in_kmh / seconds_in_hour = 1.02) : Prop :=
  speed_in_kmh = 3672 ∧ seconds_in_hour = 3600 ->
  conversion_factor = true

variables (speed_in_kmh : ℕ) (seconds_in_hour : ℤ)
#check moon_speed_in_km_per_sec speed_in_kmh seconds_in_hour

end moon_speed_in_km_per_sec_l604_604596


namespace min_value_proof_l604_604144

noncomputable def min_value (f : ℝ → ℝ → ℝ) :=
  Inf {y : ℝ | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + 3 * b = 1 ∧ f a b = y}

theorem min_value_proof :
  min_value (λ a b, 2 / a + 3 / b) = 25 :=
sorry

end min_value_proof_l604_604144


namespace germination_rate_approximation_l604_604648

noncomputable def germination_rate (G T : ℕ) := G / T.toReal

theorem germination_rate_approximation :
  let T := 10000
  let G := 9507
  abs (germination_rate G T - 0.95) < 0.01 := 
by 
  sorry

end germination_rate_approximation_l604_604648


namespace cube_root_of_sqrt_64_l604_604221

variable (a : ℕ) (b : ℕ)

def cubeRootOfSqrt64 : Prop :=
  a = Nat.sqrt 64 ∧ b = Nat.cbrt a → b = 2

theorem cube_root_of_sqrt_64 (a : ℕ) (b : ℕ) : cubeRootOfSqrt64 a b :=
  by
  sorry

end cube_root_of_sqrt_64_l604_604221


namespace ellipse_equation_chords_length_range_l604_604933

-- Definitions of the ellipse and given conditions
def ellipse (a b : ℝ) : (ℝ × ℝ) → Prop := 
  λ p, let (x, y) := p in (x^2 / a^2) + (y^2 / b^2) = 1

variables (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a > b)
variable (h4 : c = a / 2)

-- Proof problem (1): Prove the equation of the ellipse
theorem ellipse_equation : ellipse 2  √3 = λ p, let (x, y) := p in (x^2 / 4) + (y^2 / 3) = 1 :=
sorry

-- Definitions of chords and given conditions
def slope := λ (p1 p2 : ℝ × ℝ), (p2.2 - p1.2) / (p2.1 - p1.1)
def length := λ (p1 p2 : ℝ × ℝ), real.sqrt((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

variables (F : ℝ × ℝ)
variables (EF MN : ℝ × ℝ × ℝ × ℝ) -- Chords are pairs of points
variables (h5 : F.1 = sqrt(3) / 2) -- Right focus coordinate given by eccentricity

-- Additional assumptions specific to the problem
variable (h6 : slope EF.1 EF.2 = 0)
variable (h7 : length EF.1 EF.2 + length MN.1 MN.2 = 7)

-- Proof problem (2): Prove the range of values for lengths sum
theorem chords_length_range : 
  48 / 7 ≤ length EF.1 EF.2 + length MN.1 MN.2 ∧
  length EF.1 EF.2 + length MN.1 MN.2 ≤ 7 :=
sorry

end ellipse_equation_chords_length_range_l604_604933


namespace subsets_with_odd_distinct_ranks_l604_604701

theorem subsets_with_odd_distinct_ranks (n : ℕ) (h : n = 2014) : 
  let S := ∑ i in finset.range (n + 1), if i % 2 = 1 then (nat.choose n i) * (15 ^ i) else 0 in
  S = (16^n - 14^n) / 2 :=
by sorry

end subsets_with_odd_distinct_ranks_l604_604701


namespace force_of_water_on_lock_wall_l604_604368

noncomputable def force_on_the_wall (l h γ g : ℝ) : ℝ :=
  γ * g * l * (h^2 / 2)

theorem force_of_water_on_lock_wall :
  force_on_the_wall 20 5 1000 9.81 = 2.45 * 10^6 := by
  sorry

end force_of_water_on_lock_wall_l604_604368


namespace set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604662

-- Problem 1: Prove that a set of 7 numbers cannot have an arithmetic mean of 6 under given conditions.
theorem set_of_7_numbers_not_possible {s : Finset ℝ} (h_card : s.card = 7) (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / 7 = 6) : False := sorry

-- Problem 2: Prove that the number of elements in the set must be at least 9 if the arithmetic mean is 6.
theorem minimum_elements_with_mean_6 {s : Finset ℝ} (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / s.card = 6) (h_card : s.card = 2 * (s.card / 2) + 1) : 8 < s.card := sorry

end set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604662


namespace intersection_A_B_l604_604123

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l604_604123


namespace range_of_a_l604_604828

noncomputable def f (a b x : ℝ) : ℝ := a^x + x^2 - x * log a - b

theorem range_of_a (a b : ℝ) (e : ℝ) (h1 : a > 1) (h2 : e = Real.exp 1) :
  (∃ x1 x2 : ℝ, x1 ∈ Set.Icc (-1 : ℝ) 1 ∧ x2 ∈ Set.Icc (-1 : ℝ) 1 ∧ |f a b x1 - f a b x2| ≥ e - 1) →
  a ≥ e := 
sorry

end range_of_a_l604_604828


namespace arithmetic_mean_of_multiples_of_6_l604_604279

/-- The smallest three-digit multiple of 6 is 102. -/
def smallest_multiple_of_6 : ℕ := 102

/-- The largest three-digit multiple of 6 is 996. -/
def largest_multiple_of_6 : ℕ := 996

/-- The common difference in the arithmetic sequence of multiples of 6 is 6. -/
def common_difference_of_sequence : ℕ := 6

/-- The number of terms in the arithmetic sequence of three-digit multiples of 6. -/
def number_of_terms : ℕ := (largest_multiple_of_6 - smallest_multiple_of_6) / common_difference_of_sequence + 1

/-- The sum of the arithmetic sequence of three-digit multiples of 6. -/
def sum_of_sequence : ℕ := number_of_terms * (smallest_multiple_of_6 + largest_multiple_of_6) / 2

/-- The arithmetic mean of all positive three-digit multiples of 6 is 549. -/
theorem arithmetic_mean_of_multiples_of_6 : 
  let mean := sum_of_sequence / number_of_terms
  mean = 549 :=
by
  sorry

end arithmetic_mean_of_multiples_of_6_l604_604279


namespace cube_surface_area_l604_604355

theorem cube_surface_area (a : ℝ) : 
  let s := a / (Real.sqrt 3) in
  6 * s^2 = 2 * a^2 :=
by
  sorry

end cube_surface_area_l604_604355


namespace area_of_closed_figure_l604_604382

noncomputable def area_closed_figure : ℝ :=
  ∫ x in (Real.pi / 6) .. (5 * Real.pi / 6), (Real.sin x - 1 / 2)

theorem area_of_closed_figure :
  area_closed_figure = Real.sqrt 3 - Real.pi / 3 :=
by
  sorry

end area_of_closed_figure_l604_604382


namespace radius_base_circle_of_cone_l604_604623

theorem radius_base_circle_of_cone 
  (θ : ℝ) (R : ℝ) (arc_length : ℝ) (r : ℝ)
  (h1 : θ = 120) 
  (h2 : R = 9)
  (h3 : arc_length = (θ / 360) * 2 * Real.pi * R)
  (h4 : 2 * Real.pi * r = arc_length)
  : r = 3 := 
sorry

end radius_base_circle_of_cone_l604_604623


namespace roots_of_equation_l604_604402

open Real

def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

theorem roots_of_equation : {x : ℝ | x^2 - x * sgn x - 6 = 0} = {-3, 3} :=
by {
  sorry
}

end roots_of_equation_l604_604402


namespace number_of_multiples_of_31_in_array_l604_604713

-- Definition of the properties of the triangular array.
def triangular_array_entry (n k : ℕ) : ℕ :=
  2^(n-1) * (n + 2 * k - 2)

-- Prove that there are 16 entries in the array which are multiples of 31.
theorem number_of_multiples_of_31_in_array : 
  ∑ n in (finset.range 50).filter (λ n, n % 2 = 1), ∃ k ∈ finset.range (51 - n), 31 ∣ triangular_array_entry n k :=
  16 :=
sorry

end number_of_multiples_of_31_in_array_l604_604713


namespace selected_athletes_correct_num_possible_outcomes_probability_A_equals_l604_604258

section
variable (A B C total selected : ℕ)
variable (prob_eventA : ℚ)

-- Conditions
def association_A := 27
def association_B := 9
def association_C := 18

def total_athletes := association_A + association_B + association_C
def selected_athletes := 6
def sampling_ratio := selected_athletes / total_athletes

-- Number of athletes from each association (calculated using stratified sampling)
def num_selected_A := association_A * sampling_ratio
def num_selected_B := association_B * sampling_ratio
def num_selected_C := association_C * sampling_ratio

-- Question 1: Correct number of athletes selected from each association
theorem selected_athletes_correct (h1 : association_A = 27) (h2 : association_B = 9) (h3 : association_C = 18) :
  num_selected_A = 3 ∧ num_selected_B = 1 ∧ num_selected_C = 2 := sorry

-- Question 2: List all possible outcomes of selecting 2 athletes from 6
def all_possible_outcomes := 
  [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
   (2, 3), (2, 4), (2, 5), (2, 6), 
   (3, 4), (3, 5), (3, 6), 
   (4, 5), (4, 6), 
   (5, 6)]

theorem num_possible_outcomes (selected_athletes = 6) :
  all_possible_outcomes.length = 15 := sorry

-- Question 3: Probability that at least one of A5 and A6 is selected
def event_A := 
  [(1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6), 
   (4, 5), (4, 6), (5, 6)]

def probability_event_A := event_A.length / all_possible_outcomes.length

theorem probability_A_equals (h4 : probability_event_A = 3 / 5) :
  probability_event_A = 3/5 := sorry

end

end selected_athletes_correct_num_possible_outcomes_probability_A_equals_l604_604258


namespace set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604665

-- Problem 1: Prove that a set of 7 numbers cannot have an arithmetic mean of 6 under given conditions.
theorem set_of_7_numbers_not_possible {s : Finset ℝ} (h_card : s.card = 7) (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / 7 = 6) : False := sorry

-- Problem 2: Prove that the number of elements in the set must be at least 9 if the arithmetic mean is 6.
theorem minimum_elements_with_mean_6 {s : Finset ℝ} (h_median : median s ≥ 10) (h_rest : ∀ x ∈ s, x ≥ 1) (h_mean : (s.sum id) / s.card = 6) (h_card : s.card = 2 * (s.card / 2) + 1) : 8 < s.card := sorry

end set_of_7_numbers_not_possible_minimum_elements_with_mean_6_l604_604665


namespace triangle_area_l604_604901

noncomputable def area (a b c : ℝ) : ℝ :=
1/2 * b * c * (Real.sin (Real.acos ((a^2 + c^2 - b^2) / (2 * a * c))))

theorem triangle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : b = 2)
  (h2 : A = Real.pi / 3)
  (h3 : c / (1 - Real.cos C) = b / Real.cos A) :
  (let S := area a b c in S = sqrt 3 ∨ S = 2 * sqrt 3) :=
sorry

end triangle_area_l604_604901


namespace equal_a_sequence_l604_604968

theorem equal_a_sequence (n : ℕ) (h1 : 3 ≤ n) (a : Fin n → ℝ) (h2 : ∀ i, 0 < a i)
  (h3 : ∀ i, 1 ≤ i → i ≤ n → 
    let a0 := if i = 0 then a (n - 1) else a (i - 1)
    let a2 := if i = n - 1 then a 0 else a (i + 1)
    b_i := (a0 + a2) / a i
    b0 := (a (n - 1) + a 1) / a 0
    b1 := (a (n - 2) + a 0) / a (n - 1)
    b := Fin n → ℝ := 
    fun j => if j = 0 then b0 else if j = n - 1 then b1 else (a (j - 1) + a (j + 1)) / a j
    ∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → (a i ≤ a j ↔ b i ≤ b j)) : 
    ∀ i j, a i = a j :=
by {
  sorry
}

end equal_a_sequence_l604_604968


namespace radius_of_inscribed_circle_is_integer_l604_604180

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end radius_of_inscribed_circle_is_integer_l604_604180


namespace radius_of_inscribed_circle_is_integer_l604_604186

-- Define variables and conditions
variables (a b c : ℕ)
variables (h1 : c^2 = a^2 + b^2)

-- Define the radius r
noncomputable def r := (a + b - c) / 2

-- Proof statement
theorem radius_of_inscribed_circle_is_integer 
  (h2 : c^2 = a^2 + b^2)
  (h3 : (r : ℤ) = (a + b - c) / 2) : 
  ∃ r : ℤ, r = (a + b - c) / 2 :=
by {
   -- The proof will be provided here
   sorry
}

end radius_of_inscribed_circle_is_integer_l604_604186


namespace carlos_gold_quarters_l604_604733

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end carlos_gold_quarters_l604_604733


namespace arithmetic_sequence_product_l604_604512

noncomputable def b (n : ℕ) : ℤ := sorry -- define the arithmetic sequence

theorem arithmetic_sequence_product (d : ℤ) 
  (h_seq : ∀ n, b (n + 1) = b n + d)
  (h_inc : ∀ m n, m < n → b m < b n)
  (h_prod : b 4 * b 5 = 30) :
  b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28 := 
sorry

end arithmetic_sequence_product_l604_604512


namespace inscribed_circle_radius_integer_l604_604177

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l604_604177


namespace inequality_holds_if_and_only_if_c_lt_0_l604_604134

theorem inequality_holds_if_and_only_if_c_lt_0 (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c < 0) :=
sorry

end inequality_holds_if_and_only_if_c_lt_0_l604_604134


namespace intersection_of_sets_l604_604105

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l604_604105


namespace find_a_for_square_binomial_l604_604756

theorem find_a_for_square_binomial (a : ℚ) (h: ∃ (b : ℚ), ∀ (x : ℚ), 9 * x^2 + 21 * x + a = (3 * x + b)^2) : a = 49 / 4 := 
by 
  sorry

end find_a_for_square_binomial_l604_604756


namespace part1_part2_l604_604754

-- Define operation
def operation (a b c d : ℝ) : ℝ :=
  a * c + b * d

-- Part 1
def part1_condition (z : ℂ) : Prop :=
  operation 3 (Complex.conj z) z 4 = Complex.mk 7 (-3)

theorem part1 (z : ℂ) (h : part1_condition z) : 
  Complex.abs z = Real.sqrt 10 :=
  sorry

-- Part 2
def part2_condition (x y : ℝ) : Prop :=
  let expr1 := operation (y + Real.sin (2 * x)) 2 Complex.I y
  let expr2 := operation 1 (Real.sin x * Real.sin x) (Real.sin x) (2 * Real.sqrt 3 * Complex.I)
  (expr1 - expr2).im = 0

def y_as_func_of_x (x : ℝ) : ℝ :=
  -2 * Real.sin (2 * x + Real.pi / 3) + Real.sqrt 3

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y

theorem part2 (x y : ℝ) (h : part2_condition x y) :
  y = y_as_func_of_x x ∧ ∃ k : ℤ, is_monotonically_increasing (y_as_func_of_x) (k * Real.pi + Real.pi / 12) (k * Real.pi + 7 * Real.pi / 12) :=
  sorry

end part1_part2_l604_604754


namespace used_more_brown_sugar_l604_604301

-- Define the amounts of sugar used
def brown_sugar : ℝ := 0.62
def white_sugar : ℝ := 0.25

-- Define the statement to prove
theorem used_more_brown_sugar : brown_sugar - white_sugar = 0.37 :=
by
  sorry

end used_more_brown_sugar_l604_604301


namespace length_of_train_l604_604339

theorem length_of_train (speed_km_hr : ℝ) (time_sec : ℝ) (speed_conversion : speed_km_hr = 60) (time_conversion : time_sec = 3) :
  (speed_km_hr * 1000 / 3600) * time_sec ≈ 50.01 :=
by
  have speed_m_s : ℝ := (60 * 1000) / 3600
  have length_train : ℝ := speed_m_s * 3
  have approx_length : ℝ := 50.01
  sorry

end length_of_train_l604_604339


namespace limit_exists_and_independent_of_initial_value_l604_604414

noncomputable def f (x : ℝ) : ℝ := x - Real.cos x

theorem limit_exists_and_independent_of_initial_value (N_0 : ℝ) :
  ∃ t : ℝ, (∀ j : ℕ, |(λ N_j, N_{j+1} = Real.cos N_j) N_0 - t| → 0) :=
sorry

end limit_exists_and_independent_of_initial_value_l604_604414


namespace sum_of_possible_values_n_equal_19_5_l604_604011

theorem sum_of_possible_values_n_equal_19_5 
  (n : ℝ) 
  (hn1 : n ≠ 2) 
  (hn2 : n ≠ 5)
  (hn3 : n ≠ 8) 
  (hn4 : n ≠ 11) : 
  let set := {2, 5, 8, 11, n} in 
  let median := if n ≤ 5 then 5
                else if n ≥ 8 then 8
                else n in
  (median = (2 + 5 + 8 + 11 + n) / 5) →
  ({-1, 6.5, 14} = set.to_finset.filter (λ x, (median = (2 + 5 + 8 + 11 + x) / 5)).val) →
  (set.to_seq.sum = 19.5) :=
sorry

end sum_of_possible_values_n_equal_19_5_l604_604011


namespace range_of_a_l604_604235

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x > 2 * a * x + a) ↔ a ∈ Ioo (-4 : ℝ) (-1 : ℝ) := 
by
  sorry

end range_of_a_l604_604235


namespace intersection_of_sets_l604_604107

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l604_604107


namespace set_intersection_l604_604097

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l604_604097


namespace jenna_reading_pages_l604_604939

theorem jenna_reading_pages :
  ∀ (total_pages goal_pages flight_pages busy_days total_days reading_days : ℕ),
    total_days = 30 →
    busy_days = 4 →
    flight_pages = 100 →
    goal_pages = 600 →
    reading_days = total_days - busy_days - 1 →
    (goal_pages - flight_pages) / reading_days = 20 :=
by
  intros total_pages goal_pages flight_pages busy_days total_days reading_days
  sorry

end jenna_reading_pages_l604_604939


namespace eccentricity_of_hyperbola_l604_604832

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (area : ℝ) (h_area : area = 4 * Real.pi * a^2) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  in c / a

theorem eccentricity_of_hyperbola :
  ∀ (a b : ℝ), a > 0 → b > 0 → (4 * Real.pi * a^2 = π * (2 * a)^2) → hyperbola_eccentricity a b (by assumption) (by assumption) (4 * Real.pi * a^2) (by rw Real.pi_mul_eq) = Real.sqrt 2 :=
by
  intros a b h_a h_b h_area
  unfold hyperbola_eccentricity
  rw [h_area]
  sorry

end eccentricity_of_hyperbola_l604_604832


namespace constant_function_n_2_or_4_l604_604508

variables {α : Type*} [metric_space α]

def equilateral_triangle (A B C : α) : Prop := dist A B = dist B C ∧ dist B C = dist C A

def circumcircle (A B C : α) : set α :=
  { M | dist M A = dist M B ∧ dist M B = dist M C }

noncomputable def f (A B C M : α) (n : ℕ) : ℝ :=
  (dist M A) ^ n + (dist M B) ^ n + (dist M C) ^ n

theorem constant_function_n_2_or_4
  (A B C : α) (h : equilateral_triangle A B C) :
  ∀ (M ∈ circumcircle A B C), ∃ n ∈ {2, 4}, f A B C M n = f A B C (A) n :=
sorry

end constant_function_n_2_or_4_l604_604508


namespace intersection_of_sets_l604_604114

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {0, 1, 2}

theorem intersection_of_sets :
  C = A ∩ B :=
sorry

end intersection_of_sets_l604_604114


namespace right_handed_players_total_l604_604976

def total_players : ℕ := 64
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def total_right_handed : ℕ := throwers + right_handed_non_throwers

theorem right_handed_players_total : total_right_handed = 55 := by
  sorry

end right_handed_players_total_l604_604976


namespace sum_of_interior_angles_of_pentagon_l604_604609

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  -- We skip the proof as per instruction
  sorry

end sum_of_interior_angles_of_pentagon_l604_604609


namespace ways_to_insert_plus_l604_604027

-- Definition of the problem conditions
def num_ones : ℕ := 15
def target_sum : ℕ := 0 

-- Binomial coefficient calculation
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proven
theorem ways_to_insert_plus :
  binomial 14 9 = 2002 :=
by
  sorry

end ways_to_insert_plus_l604_604027


namespace example_problem_l604_604805

variables {R : Type*} [Ring R]
variables (g f : R → R)

-- Define odd function property for g
def odd_function (g : R → R) : Prop :=
  ∀ x : R, g (-x) = -g (x)

-- Given conditions
def f_def (x : R) : Prop :=
  f (x) = g (x) - 1

def f_neg3_eq_2 : Prop :=
  f (-3) = 2

-- Math proof statement
theorem example_problem (g_odd : odd_function g) (f_def' : f_def f) (f_neg3_2 : f_neg3_eq_2 f) :
  f 3 = -4 :=
by
  sorry

end example_problem_l604_604805


namespace irrational_number_among_choices_l604_604351

theorem irrational_number_among_choices : ∃ x ∈ ({17/6, -27/100, 0, Real.sqrt 2} : Set ℝ), Irrational x ∧ x = Real.sqrt 2 := by
  sorry

end irrational_number_among_choices_l604_604351


namespace overall_average_marks_l604_604304

variables {section_students : ℕ → ℕ} {mean_marks : ℕ → ℕ}
variable num_sections : ℕ

def total_students (n : ℕ) : ℕ := ∑ i in fin_range n, section_students i

def total_marks (n : ℕ) : ℕ := ∑ i in fin_range n, (section_students i * mean_marks i)

def overall_average_marks_per_student (n : ℕ) : ℚ :=
  total_marks n / total_students n

-- Given conditions:
axiom section_students_cond : section_students 0 = 65 ∧ section_students 1 = 35 ∧ section_students 2 = 45 ∧ section_students 3 = 42
axiom mean_marks_cond : mean_marks 0 = 50 ∧ mean_marks 1 = 60 ∧ mean_marks 2 = 55 ∧ mean_marks 3 = 45
axiom num_sections_cond : num_sections = 4

theorem overall_average_marks : overall_average_marks_per_student num_sections = 9715 / 187 :=
  by
    have h1 : total_students num_sections = 187, from sorry,
    have h2 : total_marks num_sections = 9715, from sorry,
    unfold overall_average_marks_per_student,
    rw [h1, h2],
    norm_num,
    sorry

end overall_average_marks_l604_604304


namespace octopus_legs_l604_604257

/-- Four octopuses made statements about their total number of legs.
    - Octopuses with 7 legs always lie.
    - Octopuses with 6 or 8 legs always tell the truth.
    - Blue: "Together we have 28 legs."
    - Green: "Together we have 27 legs."
    - Yellow: "Together we have 26 legs."
    - Red: "Together we have 25 legs."
   Prove that the Green octopus has 6 legs, and the Blue, Yellow, and Red octopuses each have 7 legs.
-/
theorem octopus_legs (L_B L_G L_Y L_R : ℕ) (H1 : (L_B + L_G + L_Y + L_R = 28 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 27 → L_B + L_G + L_Y + L_R = 27) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 26 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 25 → L_B ≠ 7)) : 
  (L_G = 6) ∧ (L_B = 7) ∧ (L_Y = 7) ∧ (L_R = 7) :=
sorry

end octopus_legs_l604_604257


namespace find_smallest_k_l604_604312

theorem find_smallest_k : 
  ∃ k n : ℕ, k > 6 ∧ n > k ∧ (∑ i in range (k + 1), i = ∑ i in range (n + 1), i - ∑ i in range (k + 1), i) ∧ k = 9 :=
by
  sorry

end find_smallest_k_l604_604312


namespace inequality_solution_set_l604_604250

theorem inequality_solution_set (x : ℝ) :
  (x + 2 > 3 * (1 - x)) ∧ (1 - 2 * x ≤ 2) → x > 1 / 4 :=
by
  intro h
  cases h with h1 h2
  sorry

end inequality_solution_set_l604_604250


namespace min_value_of_f_l604_604774

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / sqrt (x^2 + 5)

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 :=
by
  sorry

end min_value_of_f_l604_604774


namespace max_profit_l604_604319

noncomputable def C (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then (1 / 3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def L (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then -(1 / 3) * x^2 + 40 * x - 250
  else -(x + 10000 / x) + 1200

theorem max_profit :
  ∃ x : ℝ, (L x) = 1000 ∧ x = 100 :=
by
  sorry

end max_profit_l604_604319


namespace domain_log_function_l604_604229

theorem domain_log_function :
  {x : ℝ | 1 < x ∧ x < 3 ∧ x ≠ 2} = {x : ℝ | (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1)} :=
sorry

end domain_log_function_l604_604229


namespace bead_necklaces_sold_l604_604375

def cost_per_necklace : ℕ := 7
def total_earnings : ℕ := 70
def gemstone_necklaces_sold : ℕ := 7

theorem bead_necklaces_sold (B : ℕ) 
  (h1 : total_earnings = cost_per_necklace * (B + gemstone_necklaces_sold))  :
  B = 3 :=
by {
  sorry
}

end bead_necklaces_sold_l604_604375


namespace part_a_part_b_l604_604654

-- Definitions and Conditions
def median := 10
def mean := 6

-- Part (a): Prove that a set with 7 numbers cannot satisfy the given conditions
theorem part_a (n1 n2 n3 n4 n5 n6 n7 : ℕ) (h1 : median ≤ n1) (h2 : median ≤ n2) (h3 : median ≤ n3) (h4 : median ≤ n4)
  (h5 : 1 ≤ n5) (h6 : 1 ≤ n6) (h7 : 1 ≤ n7) (hmean : (n1 + n2 + n3 + n4 + n5 + n6 + n7) / 7 = mean) :
  false :=
by
  sorry

-- Part (b): Prove that the minimum size of the set where number of elements is 2n + 1 and n is a natural number, is at least 9
theorem part_b (n : ℕ) (h_sum_geq : ∀ (s : Finset ℕ), ((∀ x ∈ s, x >= median) ∧ ∃ t : Finset ℕ, t ⊆ s ∧ (∀ x ∈ t, x >= 1) ∧ s.card = 2 * n + 1) → s.sum >= 11 * n + 10) :
  n ≥ 4 :=
by
  sorry

-- Lean statements defined above match the problem conditions and required proofs

end part_a_part_b_l604_604654


namespace tanya_efficiency_greater_sakshi_l604_604557

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l604_604557


namespace option_b_correct_option_d_correct_l604_604845

noncomputable def vec_a : ℝ × ℝ := (3, -4)
noncomputable def vec_b : ℝ × ℝ := (2, 1)

-- Dot product of two vectors
def dot_prod (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Cosine of the angle between two vectors
def cosine_angle (u v : ℝ × ℝ) : ℝ :=
  dot_prod u v / (magnitude u * magnitude v)

theorem option_b_correct :
  dot_prod vec_b (5 * vec_a.1 - 2 * vec_b.1, 5 * vec_a.2 - 2 * vec_b.2) = 0 :=
by sorry

theorem option_d_correct :
  cosine_angle vec_b (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) = -3 * Real.sqrt 130 / 130 :=
by sorry

end option_b_correct_option_d_correct_l604_604845


namespace angle_PQR_is_90_l604_604932

theorem angle_PQR_is_90 {P Q R S : Type}
  (is_straight_line_RSP : ∃ P R S : Type, (angle R S P = 180)) 
  (angle_QSP : angle Q S P = 70)
  (isosceles_RS_SQ : ∃ (RS SQ : Type), RS = SQ)
  (isosceles_PS_SQ : ∃ (PS SQ : Type), PS = SQ) : angle P Q R = 90 :=
by 
  sorry

end angle_PQR_is_90_l604_604932


namespace MariaTotalPaid_l604_604298

-- Define a structure to hold the conditions
structure DiscountProblem where
  discount_rate : ℝ
  discount_amount : ℝ

-- Define the given discount problem specific to Maria
def MariaDiscountProblem : DiscountProblem :=
  { discount_rate := 0.25, discount_amount := 40 }

-- Define our goal: proving the total amount paid by Maria
theorem MariaTotalPaid (p : DiscountProblem) (h₀ : p = MariaDiscountProblem) :
  let original_price := p.discount_amount / p.discount_rate
  let total_paid := original_price - p.discount_amount
  total_paid = 120 :=
by
  sorry

end MariaTotalPaid_l604_604298


namespace maria_total_payment_l604_604299

theorem maria_total_payment
  (original_price discount : ℝ)
  (discount_percentage : ℝ := 0.25)
  (discount_amount : ℝ := 40) 
  (total_paid : ℝ := 120) :
  discount_amount = discount_percentage * original_price →
  total_paid = original_price - discount_amount → 
  total_paid = 120 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end maria_total_payment_l604_604299


namespace blocks_needed_l604_604263

theorem blocks_needed (a b c V_A V_B V_C V_total σ L V_large_cube : ℕ) 
  (h1 : a = 1)
  (h2 : b = 2 * a)
  (h3 : c = 3 / 2 * b)
  (h4 : V_A = a ^ 3)
  (h5 : V_B = b ^ 3)
  (h6 : V_C = c ^ 3)
  (h7 : V_total = V_A + V_B + V_C)
  (h8 : L = b + c)
  (h9 : V_large_cube = L ^ 3)
  (h10: σ = 50): σ = 50 :=
begin
  sorry
end

end blocks_needed_l604_604263


namespace find_constants_a_b_l604_604956

def M : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 1],
  ![2, -2]
]

theorem find_constants_a_b :
  ∃ (a b : ℚ), (M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  a = 1/8 ∧ b = -1/8 :=
by
  sorry

end find_constants_a_b_l604_604956


namespace find_m_in_hyperbola_l604_604781

-- Define the problem in Lean 4
theorem find_m_in_hyperbola (m : ℝ) (x y : ℝ) (e : ℝ) (a_sq : ℝ := 9) (h_eq : e = 2) (h_hyperbola : x^2 / a_sq - y^2 / m = 1) : m = 27 :=
sorry

end find_m_in_hyperbola_l604_604781


namespace find_distance_sum_l604_604953

-- Centroid and distance definitions
def centroid (d e f : ℝ) : ℝ := (d + e + f) / 3
def squared_dist (x y : ℝ) : ℝ := (x - y) * (x - y)

-- Conditions
variable (d e f : ℝ)
variable (h1 : squared_dist (centroid d e f) d + squared_dist (centroid d e f) e + squared_dist (centroid d e f) f = 72)

-- Goal
theorem find_distance_sum :
  (d - e) * (d - e) + (d - f) * (d - f) + (e - f) * (e - f) = 216 :=
by
  sorry

end find_distance_sum_l604_604953


namespace trajectory_midpoints_l604_604841

variables (a b c x y : ℝ)

def arithmetic_sequence (a b c : ℝ) : Prop := c = 2 * b - a

def line_eq (b a c x y : ℝ) : Prop := b * x + a * y + c = 0

def parabola_eq (x y : ℝ) : Prop := y^2 = -0.5 * x

theorem trajectory_midpoints
  (hac : arithmetic_sequence a b c)
  (line_cond : line_eq b a c x y)
  (parabola_cond : parabola_eq x y) :
  (x + 1 = -(2 * y - 1)^2) ∧ (y ≠ 1) :=
sorry

end trajectory_midpoints_l604_604841


namespace angle_PQR_is_90_l604_604926

variable (R P Q S : Type) [EuclideanGeometry R P Q S]
variable (RSP_is_straight : straight_line R S P)
variable (angle_QSP : ∡Q S P = 70)

theorem angle_PQR_is_90 : ∡P Q R = 90 :=
by
  sorry

end angle_PQR_is_90_l604_604926


namespace correct_option_l604_604643

theorem correct_option : (∃ x, x = -3 ∧ x^3 = -27) :=
by {
  -- Given conditions
  let x := -3
  use x
  constructor
  . rfl
  . norm_num
}

end correct_option_l604_604643


namespace set_intersection_l604_604096

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l604_604096


namespace a_50_value_l604_604474

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 101 ∧
  a 2 = 102 ∧
  (∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = n + 2)

theorem a_50_value (a : ℕ → ℤ) (h : sequence a) : a 50 = 117 :=
sorry

end a_50_value_l604_604474


namespace solve_cryptarithm_l604_604567

def cryptarithm_proof (TW O ELEVE : ℕ) : Prop :=
  let x := TW in
  let y := O in
  let z := ELEVE in
  x * 986 = z ∧ 
  ∀ n m k, n ≠ m ∧ n ≠ k ∧ m ≠ k → 
  0 ≤ n ∧ n < 10 ∧ 0 ≤ m ∧ m < 10 ∧ 0 ≤ k ∧ k < 10 ∧ 
  (x / 10 ≠ x % 10) ∧ (10000 * (x / 10) + 1000 * (x % 10) + 100 * (x % 10) + 10 * (y) + y) = z
  
theorem solve_cryptarithm : cryptarithm_proof 34 0 170 := by
  sorry

end solve_cryptarithm_l604_604567


namespace other_ticket_price_l604_604216

theorem other_ticket_price (total_tickets : ℕ) (total_sales : ℝ) (cheap_tickets : ℕ) (cheap_price : ℝ) (expensive_tickets : ℕ) (expensive_price : ℝ) :
  total_tickets = 380 →
  total_sales = 1972.50 →
  cheap_tickets = 205 →
  cheap_price = 4.50 →
  expensive_tickets = 380 - 205 →
  205 * 4.50 + expensive_tickets * expensive_price = 1972.50 →
  expensive_price = 6.00 :=
by
  intros
  -- proof will be filled here
  sorry

end other_ticket_price_l604_604216


namespace total_revenue_is_correct_l604_604056

theorem total_revenue_is_correct :
  let fiction_books := 60
  let nonfiction_books := 84
  let children_books := 42

  let fiction_sold_frac := 3/4
  let nonfiction_sold_frac := 5/6
  let children_sold_frac := 2/3

  let fiction_price := 5
  let nonfiction_price := 7
  let children_price := 3

  let fiction_sold_qty := fiction_sold_frac * fiction_books
  let nonfiction_sold_qty := nonfiction_sold_frac * nonfiction_books
  let children_sold_qty := children_sold_frac * children_books

  let total_revenue := fiction_sold_qty * fiction_price
                       + nonfiction_sold_qty * nonfiction_price
                       + children_sold_qty * children_price
  in total_revenue = 799 :=
by
  sorry

end total_revenue_is_correct_l604_604056


namespace probability_single_draws_probability_two_different_colors_l604_604619

-- Define probabilities for black, yellow and green as events A, B, and C respectively.
variables (A B C : ℝ)

-- Conditions based on the problem statement
axiom h1 : A + B = 5/9
axiom h2 : B + C = 2/3
axiom h3 : A + B + C = 1

-- Here is the statement to prove the calculated probabilities of single draws
theorem probability_single_draws : 
  A = 1/3 ∧ B = 2/9 ∧ C = 4/9 :=
sorry

-- Define the event of drawing two balls of the same color
variables (black yellow green : ℕ)
axiom balls_count : black + yellow + green = 9
axiom black_component : A = black / 9
axiom yellow_component : B = yellow / 9
axiom green_component : C = green / 9

-- Using the counts to infer the probability of drawing two balls of different colors
axiom h4 : black = 3
axiom h5 : yellow = 2
axiom h6 : green = 4

theorem probability_two_different_colors :
  (1 - (3/36 + 1/36 + 6/36)) = 13/18 :=
sorry

end probability_single_draws_probability_two_different_colors_l604_604619


namespace crab_ratio_l604_604261

theorem crab_ratio 
  (oysters_day1 : ℕ) 
  (crabs_day1 : ℕ) 
  (total_days : ℕ) 
  (oysters_ratio : ℕ) 
  (oysters_day2 : ℕ) 
  (total_oysters_crabs : ℕ) 
  (crabs_day2 : ℕ) 
  (ratio : ℚ) :
  oysters_day1 = 50 →
  crabs_day1 = 72 →
  oysters_ratio = 2 →
  oysters_day2 = oysters_day1 / oysters_ratio →
  total_oysters_crabs = 195 →
  total_oysters_crabs = oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 →
  crabs_day2 = total_oysters_crabs - (oysters_day1 + crabs_day1 + oysters_day2) →
  ratio = (crabs_day2 : ℚ) / crabs_day1 →
  ratio = 2 / 3 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end crab_ratio_l604_604261


namespace solve_for_x_l604_604283

theorem solve_for_x : (∃ x : ℚ, (40 / 60 : ℚ) = real.sqrt (x / 60) ∧ x = 80 / 3) :=
by
  use 80 / 3
  sorry

end solve_for_x_l604_604283


namespace range_of_a_l604_604778

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ a ∈ Icc (-4 : ℝ) (0 : ℝ) := 
sorry

end range_of_a_l604_604778


namespace intersection_point_correct_l604_604800

-- Definitions of points A, B, C, and D
def A : ℝ × ℝ × ℝ := (2, -1, 2)
def B : ℝ × ℝ × ℝ := (12, -11, 7)
def C : ℝ × ℝ × ℝ := (1, 4, -7)
def D : ℝ × ℝ × ℝ := (4, -2, 13)

-- Parametric equations of lines AB and CD
def param_AB (t : ℝ) : ℝ × ℝ × ℝ := (2 + 10 * t, -1 - 10 * t, 2 + 5 * t)
def param_CD (s : ℝ) : ℝ × ℝ × ℝ := (1 + 3 * s, 4 - 6 * s, -7 + 20 * s)

-- The point of intersection we want to prove
def intersection_point : ℝ × ℝ × ℝ := (8 / 3, -7 / 3, 7 / 3)

-- The Lean statement for the proof
theorem intersection_point_correct :
  ∃ (t s : ℝ), param_AB t = intersection_point ∧ param_CD s = intersection_point :=
sorry

end intersection_point_correct_l604_604800


namespace greatest_exponent_three_divides_product_l604_604045

theorem greatest_exponent_three_divides_product :
  let v := ∏ i in Finset.range 30, (i + 1)
  ∃ a : ℕ, 3 ^ a ∣ v ∧ ∀ b : ℕ, 3 ^ b ∣ v → b ≤ 14 :=
by
  let v := ∏ i in Finset.range 30, (i + 1)
  existsi 14
  split
  -- Proof parts skipped
  sorry

end greatest_exponent_three_divides_product_l604_604045


namespace simplify_and_evaluate_l604_604563

theorem simplify_and_evaluate : 
  (1 / (3 - 2) - 1 / (3 + 1)) / (3 / (3^2 - 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_l604_604563


namespace number_of_roots_l604_604025

theorem number_of_roots :
  ∃ n : ℕ, n = 3 ∧ 
  ∀ x : ℝ, (7 < x ∧ x ≤ 14) → 
    (cos x = ((1:ℝ) / 7) * x - 1 ∧ 0 < cos x ∧ cos x ≤ 1) ↔ 
    n = 3 :=
sorry

end number_of_roots_l604_604025


namespace expression_value_l604_604360

variable (S : ℝ)
def expression : ℝ :=
  1 / (4 - real.sqrt 15) - 1 / (real.sqrt 15 - real.sqrt 14) +
  1 / (real.sqrt 14 - real.sqrt 13) - 1 / (real.sqrt 13 - real.sqrt 12) +
  1 / (real.sqrt 12 - 3)

theorem expression_value : S = 7 :=
by
  let S := expression
  sorry

end expression_value_l604_604360


namespace contradiction_for_n3_min_elements_when_n_ge_4_l604_604674

theorem contradiction_for_n3 :
  ∀ (s : Set ℕ), (s.card = 7) → 
                 (∀ (x ∈ s), x ≥ 1) → 
                 (∃ t u : Set ℕ, (t.card = 4) ∧ (u.card = 3) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) 
                 → ∃ x ∈ s, false :=
sorry

theorem min_elements_when_n_ge_4 (n : ℕ) (hn : n ≥ 4) :
  ∃ (s : Set ℕ), (s.card = 2 * n + 1) ∧ 
                 (∀ (x ∈ s), x ≥ 1) ∧ 
                 (∃ t u : Set ℕ, (t.card = n + 1) ∧ (u.card = n) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
                 (∀ (x ∈ t), x ≥ 10) ∧ 
                 (∀ (x ∈ u), x ≥ 1)) ∧
                 ∀ (s : Set ℕ), s.card = 2 * n + 1 → (∑ x in s, x) / (2 * n + 1) = 6 :=
sorry

example : ∃ s, (s.card = 9) ∧ (∀ x ∈ s, x ≥ 1) ∧ 
               (∃ t u : Set ℕ, (t.card = 5) ∧ (u.card = 4) ∧ (t ⊆ s) ∧ (u ⊆ s) ∧ 
               (∀ x ∈ t, x ≥ 10) ∧ (∀ x ∈ u, x ≥ 1) ∧
               (∑ x in s, x) / 9 = 6) :=
{ sorry }

end contradiction_for_n3_min_elements_when_n_ge_4_l604_604674


namespace line_through_diameter_l604_604795

theorem line_through_diameter (P : ℝ × ℝ) (hP : P = (2, 1)) (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = -1 :=
by
  exists 1, -1, -1
  sorry

end line_through_diameter_l604_604795


namespace limit_at_2_l604_604167

noncomputable def delta (ε : ℝ) : ℝ := ε / 3

theorem limit_at_2 (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ x : ℝ, (0 < |x - 2| ∧ |x - 2| < δ) → |(3 * x^2 - 5 * x - 2) / (x - 2) - 7| < ε :=
by
  let δ := delta ε
  have hδ : δ > 0 := by
    sorry
  use δ, hδ
  intros x hx
  sorry

end limit_at_2_l604_604167


namespace sequences_countable_l604_604172

-- Define the countability of sets and other necessary constructs
open Set

-- Defining a countable set
def countable (S : Set ℕ) : Prop := ∃ f : S → ℕ, Function.injective f

-- Prove that for any natural number n, the set of all sequences of length n composed of natural numbers is countable
theorem sequences_countable (n : ℕ) : countable (Set { l : List ℕ | l.length = n }) :=
begin
  sorry
end

end sequences_countable_l604_604172


namespace min_bottles_to_fill_large_bottle_l604_604705

theorem min_bottles_to_fill_large_bottle (large_bottle_ml : Nat) (small_bottle1_ml : Nat) (small_bottle2_ml : Nat) (total_bottles : Nat) :
  large_bottle_ml = 800 ∧ small_bottle1_ml = 45 ∧ small_bottle2_ml = 60 ∧ total_bottles = 14 →
  ∃ x y : Nat, x * small_bottle1_ml + y * small_bottle2_ml = large_bottle_ml ∧ x + y = total_bottles :=
by
  intro h
  sorry

end min_bottles_to_fill_large_bottle_l604_604705


namespace B_grazed_10_cows_l604_604690

theorem B_grazed_10_cows (
  A_cows : ℕ := 24, 
  A_months : ℕ := 3, 
  B_months : ℕ := 5, 
  C_cows : ℕ := 35, 
  C_months : ℕ := 4,
  D_cows : ℕ := 21, 
  D_months : ℕ := 3, 
  A_rent : ℕ := 1440, 
  total_rent : ℕ := 6500) 
  : (∃ x : ℕ, 
    (A_cows * A_months) / ((A_cows * A_months) + (x * B_months) + (C_cows * C_months) + (D_cows * D_months)) 
    = A_rent / total_rent ∧ 
    x = 10) :=
by 
  sorry

end B_grazed_10_cows_l604_604690


namespace incorrect_statements_l604_604892

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def monotonically_decreasing_in_pos (f : ℝ → ℝ) : Prop :=
∀ x y, 0 < x ∧ x < y → f y ≤ f x

theorem incorrect_statements
  (f : ℝ → ℝ)
  (hf_even : even_function f)
  (hf_decreasing : monotonically_decreasing_in_pos f) :
  ¬ (∀ a, f (2 * a) < f (-a)) ∧ ¬ (f π > f (-3)) ∧ ¬ (∀ a, f (a^2 + 1) < f 1) :=
by sorry

end incorrect_statements_l604_604892


namespace halfway_between_3_4_and_5_7_l604_604276

-- Define the two fractions
def frac1 := 3/4
def frac2 := 5/7

-- Define the average function for two fractions
def halfway_fract (a b : ℚ) : ℚ := (a + b) / 2

-- Prove that the halfway fraction between 3/4 and 5/7 is 41/56
theorem halfway_between_3_4_and_5_7 : 
  halfway_fract frac1 frac2 = 41/56 := 
by 
  sorry

end halfway_between_3_4_and_5_7_l604_604276


namespace composition_of_symmetries_is_translation_l604_604311

variables {P : Type*} [MetricSpace P] {M M1 M2 : P}
variables {l1 l2 : AffineSubspace ℝ P} -- lines are affine subspaces
variable (h : ℝ) -- perpendicular distance between the lines

-- Predicate to specify that two lines are parallel
def Parallel (l1 l2 : AffineSubspace ℝ P) : Prop := ∀ x ∈ l1.direction, x ∈ l2.direction

-- Definition for a symmetry (reflection) with respect to an affine subspace (line)
noncomputable def symmetry (A : AffineSubspace ℝ P) (p : P) : P := sorry -- define properly

-- Given conditions
variables (M1_eq : M1 = symmetry l1 M)
variables (M2_eq : M2 = symmetry l2 M1)
variables (parallel : Parallel l1 l2)
variables (distance : ∀ p ∈ l1, ∀ q ∈ l2, dist p q = h)

-- Statement to prove
theorem composition_of_symmetries_is_translation :
  dist M M2 = 2 * h ∧ (∃ v : P, ∀ p : P, symmetry l2 (symmetry l1 p) = p + v) := sorry

end composition_of_symmetries_is_translation_l604_604311


namespace two_points_contain_each_line_l604_604548

theorem two_points_contain_each_line :
  ∀ (L : Fin 100 → Line),
    (∀ i j : Fin 100, i ≠ j → ¬parallel (L i) (L j)) →
    (∀ s : Finset (Fin 100), s.card = 5 → ∃ p : Point, ∃ t : Finset (Fin 100), t.card = 3 ∧ (∀ i ∈ t, through_point (L i) p)) →
    ∃ p₁ p₂ : Point, ∀ i : Fin 100, through_point (L i) p₁ ∨ through_point (L i) p₂ :=
by
  sorry

end two_points_contain_each_line_l604_604548


namespace dumulandia_connected_network_disruption_l604_604068

/- Define the problem in terms of graph theory -/
open GraphTheory

/-- In the context of graph theory, each city is a vertex and each road is an edge. -/
theorem dumulandia_connected_network_disruption (G : SimpleGraph V) [Fintype V] [DecidableRel G.Adj]
  (hvalency : ∀ v : V, G.degree v = 10)
  (hconn : G.Connected) :
  ∃ (S : Finset (Sym2 V)), S.card = 9 ∧ G.edge_induced_subgraph (G.E \ S).Connected = false := 
sorry

end dumulandia_connected_network_disruption_l604_604068


namespace line_equations_l604_604769

theorem line_equations (p1 p2 pO pB : ℝ × ℝ)
  (hp1 : p1 = (1, 2)) (hp2 : ∃ k : ℝ, p2 = (1 + 1/k, 2 + k)) 
  (hpO : pO = (0, 0)) (hpB : pB = (3, 1)) :
  (∃ k : ℝ, k = 1/3 ∨ k = -3) ∧
  (k = 1/3 → ∀ x y : ℝ, (x - 1) / 3 = y - 2) → 3 * x - y + 5 = 0) ∧
  (k = -3 → ∀ x y : ℝ, (x - 1) / -3 = y - 2) → x + 3 * y - 5 = 0) :=
by sorry

end line_equations_l604_604769


namespace mrs_bil_earnings_percentage_l604_604900

variable (T F M J : ℝ)
variable (h1 : F = 0.70 * T)
variable (h2 : J = 1.10 * M)
variable (h3 : M = 0.70 * T)

theorem mrs_bil_earnings_percentage (h1 : F = 0.70 * T) (h2 : J = 1.10 * M) (h3 : M = 0.70 * T) :
  (J / T) * 100 = 77 :=
by
  rw [h3, h2]
  field_simp
  ring
  sorry

end mrs_bil_earnings_percentage_l604_604900


namespace true_statements_M_l604_604748

theorem true_statements_M (θ : ℝ) (x y : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi)
  (h3 : x * Real.cos θ + (y - 2) * Real.sin θ = 1) :
  (∃ P : ℝ × ℝ, ∀ (θ : ℝ), x * Real.cos θ + (y - 2) * Real.sin θ ≠ 1) ∧
  (∀ n : ℕ, n ≥ 3 → ∃ (points : Fin n → ℝ × ℝ), (∀ i : Fin n, x * Real.cos θ + (points i).1 * (Real.sin θ - 2) = 1) ∧ 
  (points 0).1 = (points (Fin.succ 0)).1) :=
sorry

end true_statements_M_l604_604748


namespace roots_of_polynomial_l604_604766

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end roots_of_polynomial_l604_604766


namespace min_area_of_tangent_and_parabola_l604_604067

noncomputable def min_enclosed_area (θ : ℝ) (hθ : π < θ ∧ θ < 2*π) : ℝ :=
  let x := cos θ
  let y := sin θ
  let l := λ x, -cot θ * x + 1/sin θ
  let x1 := (-cot θ - sqrt (cot θ ^ 2 + 8 + 4 / sin θ)) / 2
  let x2 := (-cot θ + sqrt (cot θ ^ 2 + 8 + 4 / sin θ)) / 2
  let A := ∫ t in x1..x2, (-cot θ * t + 1 / sin θ - t ^ 2 + 2) - (-cot θ * t + 1 / sin θ - t ^ 2 + 2) 
  - (A' := (cos θ * (2 * sin θ + 1) * sqrt (7 * sin θ ^ 2 + 4 * sin θ + 1)) / (2 * sin θ ^ 4)) 
  if A' = 0 then - sqrt ((7 * sin θ ^ 2 + 4 * sin θ + 1) ^ 3) / (6 * sin θ ^ 3) else A

theorem min_area_of_tangent_and_parabola :
  ∃ θ, π < θ ∧ θ < 2*π ∧ min_enclosed_area θ (and.intro _ _) 
= - sqrt ((7 * sin θ ^ 2 + 4 * sin θ + 1) ^ 3) / (6 * sin θ ^ 3) := sorry

end min_area_of_tangent_and_parabola_l604_604067


namespace vector_norm_lower_bound_l604_604835

variables (a b c : EuclideanSpace ℝ (Fin 2))

def condition1 : Prop := inner a a = 1
def condition2 : Prop := inner a b = 1
def condition3 : Prop := inner b c = 1
def condition4 : Prop := inner a c = 2

theorem vector_norm_lower_bound
    (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 b c) (h4 : condition4 a c) :
    ∥a + b + c∥ ≥ 4 :=
sorry

end vector_norm_lower_bound_l604_604835


namespace angle_PQC_correct_l604_604413

variables {A B C D P Q : Type}
variables [AffinePlane A B C D P Q]

theorem angle_PQC_correct
  (h1 : is_isosceles_triangle A B C)
  (h2 : ∠ A = 30)
  (h3 : is_midpoint D B C)
  (h4 : ∃ P, lies_on_segment P A D)
  (h5 : ∃ Q, lies_on_segment Q A B)
  (h6 : segment_length P B = segment_length P Q) :
  ∠ PQC = 15 :=
  sorry

end angle_PQC_correct_l604_604413


namespace monotonic_intervals_range_of_m_l604_604786

def vector_a (x : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.sin x, real.cos x + real.sin x)
def vector_b (x : ℝ) : ℝ × ℝ := (2 * real.cos x, real.sin x - real.cos x)
def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

theorem monotonic_intervals :
  ∀ (k : ℤ), 
    ( ∀ (x : ℝ), x ∈ set.Icc (-real.pi / 6 + ↑k * real.pi) (real.pi / 3 + ↑k * real.pi) → ∃ c, 0 < c ∧ ∀ y, y ∈ set.Icc (-real.pi / 6 + ↑k * real.pi) (x) → f y ≤ f x ) ∧
    ( ∀ (x : ℝ), x ∈ set.Icc (real.pi / 3 + ↑k * real.pi) (5 * real.pi / 6 + ↑k * real.pi) → ∃ c, 0 < c ∧ ∀ y, y ∈ set.Icc (real.pi / 3 + ↑k * real.pi) (x) → f y ≥ f x ) :=
sorry

theorem range_of_m (x : ℝ) (t : ℝ) :
  x ∈ set.Icc (5 * real.pi / 24) (5 * real.pi / 12) →
  (∀ t, ∃ m, 0 ≤ m ∧ m ≤ 4 ∧ ∀ t, m * t^2 + m * t + 3 ≥ f x) :=
sorry

end monotonic_intervals_range_of_m_l604_604786


namespace range_of_m_l604_604465

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then (x - m) ^ 2 - 2 else 2 * x ^ 3 - 3 * x ^ 2

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f x m = -1) ↔ m ≥ 1 :=
by
  sorry

end range_of_m_l604_604465


namespace symmetric_point_is_correct_l604_604935

def point := (R × R × R)

def symmetric_point_yOz (A : point) : point :=
  let (x, y, z) := A
  (-x, y, z)

theorem symmetric_point_is_correct :
  symmetric_point_yOz (-2, 4, 3) = (2, 4, 3) :=
by
  -- we assume the symmetry point function works correctly, skip detailed proof using sorry
  sorry

end symmetric_point_is_correct_l604_604935


namespace jenna_reading_pages_l604_604940

theorem jenna_reading_pages :
  ∀ (total_pages goal_pages flight_pages busy_days total_days reading_days : ℕ),
    total_days = 30 →
    busy_days = 4 →
    flight_pages = 100 →
    goal_pages = 600 →
    reading_days = total_days - busy_days - 1 →
    (goal_pages - flight_pages) / reading_days = 20 :=
by
  intros total_pages goal_pages flight_pages busy_days total_days reading_days
  sorry

end jenna_reading_pages_l604_604940


namespace evens_before_odd_prob_l604_604719

open Nat

-- Definition of the problem parameters
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def odd_faces : Finset ℕ := {1, 3, 5, 7}
def even_faces : Finset ℕ := {2, 4, 6, 8}

-- Roll probability conditions
def prob_odd (n : ℕ) : ℚ := if n ∈ odd_faces then 1/2 else 0
def prob_even (n : ℕ) : ℚ := if n ∈ even_faces then 1/2 else 0

-- Probability calculation of seeing all evens before any odds
def prob_all_evens_before_first_odd : ℚ :=
  (1 / 2) * series_sum (5) (λ n, (1 / 2)^n - ∑ k in (range 4).filter(λ k, 1 ≤ k ∧ k ≤ 3),
                         (-1)^(k-1) * (binom 4 k) * (k / 4)^(n - 1))

-- Final probability statement to be proved
theorem evens_before_odd_prob : prob_all_evens_before_first_odd = 1 / 70 :=
  sorry

end evens_before_odd_prob_l604_604719


namespace minimum_value_768_l604_604135

noncomputable def min_value_expression (a b c : ℝ) := a^2 + 8 * a * b + 16 * b^2 + 2 * c^5

theorem minimum_value_768 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_condition : a * b^2 * c^3 = 256) : 
  min_value_expression a b c = 768 :=
sorry

end minimum_value_768_l604_604135


namespace waiting_room_l604_604621

theorem waiting_room (x : ℕ) :
  let initial_waiting := 22
  let interview_room := 5
  let waiting_after := initial_waiting + x
  (waiting_after = 5 * interview_room) → (x = 3) :=
by
  intro h
  have : waiting_after = 22 + x := by rfl
  rw [this] at h
  have : 22 + x = 25 := by rw [h]; norm_num
  linarith

end waiting_room_l604_604621


namespace magnitude_z_l604_604374

def z : ℂ := 4 - 15 * complex.I

theorem magnitude_z : complex.abs z = Real.sqrt 241 :=
by sorry

end magnitude_z_l604_604374


namespace midpoint_sum_coordinates_eq_l604_604599

theorem midpoint_sum_coordinates_eq
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 5) (h2 : y1 = -7) (h3 : x2 = -7) (h4 : y2 = 3) :
  ( (x1 + x2) / 2 + (y1 + y2) / 2 ) = -3 :=
by
  rw [h1, h2, h3, h4]
  simp
  norm_num
  sorry

end midpoint_sum_coordinates_eq_l604_604599


namespace MariaTotalPaid_l604_604297

-- Define a structure to hold the conditions
structure DiscountProblem where
  discount_rate : ℝ
  discount_amount : ℝ

-- Define the given discount problem specific to Maria
def MariaDiscountProblem : DiscountProblem :=
  { discount_rate := 0.25, discount_amount := 40 }

-- Define our goal: proving the total amount paid by Maria
theorem MariaTotalPaid (p : DiscountProblem) (h₀ : p = MariaDiscountProblem) :
  let original_price := p.discount_amount / p.discount_rate
  let total_paid := original_price - p.discount_amount
  total_paid = 120 :=
by
  sorry

end MariaTotalPaid_l604_604297


namespace angle_between_a_b_norm_a_sub_b_l604_604417

variables (a b : Mathlib.Geometry.Vector3) -- Assuming we're dealing with 3D vectors for generality
variables [normed_field ℝ] [inner_product_space ℝ (Mathlib.Geometry.Vector3)]

-- Condition: non-zero vectors a and b, |a| = 1
axiom a_ne_zero : a ≠ 0
axiom b_ne_zero : b ≠ 0
axiom norm_a : norm a = 1

-- Condition: (a - b) · (a + b) = 1/2
axiom dot_condition : inner (a - b) (a + b) = 1 / 2

-- First part: Show that the angle between a and b is 45 degrees if a · b = 1/2
theorem angle_between_a_b (dot_ab : inner a b = 1 / 2) : 
  real.angle_cos (1 / sqrt 2) (1 / sqrt 2) = mathlib.geom.pi / 4 := sorry

-- Second part: Show that |a - b| = sqrt(2)/2 under the condition a · b = 1/2
theorem norm_a_sub_b (dot_ab : inner a b = 1 / 2) : 
  norm (a - b) = sqrt 2 / 2 := sorry

end angle_between_a_b_norm_a_sub_b_l604_604417


namespace fraction_division_l604_604380

theorem fraction_division :
  (5 : ℚ) / ((13 : ℚ) / 7) = 35 / 13 :=
by
  sorry

end fraction_division_l604_604380


namespace perp_tangents_sin_l604_604718

noncomputable def has_infinitely_many_perpendicular_tangents (f : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (f' x1 * f' x2 = -1)

theorem perp_tangents_sin :
  ∀ (f : ℝ → ℝ), (f = λ x, Real.exp x ∨ f = λ x, x^3 ∨ f = λ x, Real.log x ∨ f = λ x, Real.sin x) →
  has_infinitely_many_perpendicular_tangents f ↔ f = λ x, Real.sin x :=
sorry

end perp_tangents_sin_l604_604718


namespace combined_weight_of_two_new_persons_l604_604579

variable (W X Y : ℝ)
variable (h_avg : (8 * W))
variable (h_replace1 : 65)
variable (h_replace2 : 75)
variable (h_new_avg : (W + 5))

theorem combined_weight_of_two_new_persons : 
  W + (W + 5) + (X + Y - 140) = 40 → X + Y = 180 := 
by sorry

end combined_weight_of_two_new_persons_l604_604579


namespace boat_speed_in_still_water_equals_6_l604_604906

def river_flow_rate : ℝ := 2
def distance_upstream : ℝ := 40
def distance_downstream : ℝ := 40
def total_time : ℝ := 15

theorem boat_speed_in_still_water_equals_6 :
  ∃ b : ℝ, (40 / (b - river_flow_rate) + 40 / (b + river_flow_rate) = total_time) ∧ b = 6 :=
sorry

end boat_speed_in_still_water_equals_6_l604_604906


namespace cube_root_of_one_over_64_l604_604582

theorem cube_root_of_one_over_64 :
  real.cbrt (1 / 64) = 1 / 4 :=
by
  have h1 : (64 : ℝ) = 4 ^ 3 := by norm_num
  have h2 : (1 / 64 : ℝ) = (1 / 4) ^ 3 := by rw [← h1, one_div, one_div (4 ^ 3)];
    norm_num
  rw [← h2]
  exact real.cbrt_pow 3 (1 / 4)

end cube_root_of_one_over_64_l604_604582


namespace average_speed_is_expected_l604_604696

-- Define the conditions
def speed1 : ℝ := 35  -- speed in kph
def distance1 : ℝ := 30  -- distance in kilometers

def speed2 : ℝ := 55  -- speed in kph
def distance2 : ℝ := 35  -- distance in kilometers

def speed3 : ℝ := 65  -- speed in kph
def time3 : ℝ := 0.5  -- time in hours (30 minutes)

def speed4 : ℝ := 42  -- speed in kph
def time4 : ℝ := 1 / 3  -- time in hours (20 minutes)

-- Define the expected average speed
def expected_avg_speed : ℝ := 48

-- Statement of the proof problem
theorem average_speed_is_expected : 
  (distance1 + distance2 + speed3 * time3 + speed4 * time4) / 
  (distance1 / speed1 + distance2 / speed2 + time3 + time4) = expected_avg_speed := 
by 
  -- the proof would go here
  sorry

end average_speed_is_expected_l604_604696


namespace find_line_equation_l604_604426

noncomputable def line_through_point (m : ℝ) (pt : ℝ × ℝ) : ℝ → ℝ := 
  λ x, m * (x - pt.1) + pt.2

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_line_equation (a b c : ℝ) (h : a = 1 ∧ b = -1 ∧ c = 3) :
  ∃ (l : ℝ → ℝ), 
    is_perpendicular (-1) 1 ∧
    (l 0 = 3) ∧
    (l = line_through_point 1 (0,3)) := by
  sorry

end find_line_equation_l604_604426


namespace sqrt_eq_one_implies_a_eq_one_l604_604876

theorem sqrt_eq_one_implies_a_eq_one (a : ℝ) (h : sqrt a = 1) : a = 1 := sorry

end sqrt_eq_one_implies_a_eq_one_l604_604876


namespace probability_same_color_is_correct_l604_604861

noncomputable def total_ways_select_plates : ℕ := Nat.choose 11 3
def ways_select_red_plates : ℕ := Nat.choose 6 3
def ways_select_blue_plates : ℕ := Nat.choose 5 3
noncomputable def favorable_outcomes : ℕ := ways_select_red_plates + ways_select_blue_plates
noncomputable def probability_same_color : ℚ := favorable_outcomes / total_ways_select_plates

theorem probability_same_color_is_correct :
  probability_same_color = 2/11 := 
by
  sorry

end probability_same_color_is_correct_l604_604861


namespace solve_for_x_l604_604565

theorem solve_for_x (x : ℝ) (h : 3^(2 * x + 1) = 1 / 27) : x = -2 := by
  sorry

end solve_for_x_l604_604565


namespace max_regions_divided_by_planes_l604_604497

/--
  The maximum number of regions into which \( n \) planes can divide three-dimensional space is 
  given by \( R(n) = \frac{n^3 + 5n}{6} + 1 \), under the conditions that
  no two planes are parallel, no three planes intersect at a single line, and no
  four planes intersect at a single point.
-/
theorem max_regions_divided_by_planes (n : ℕ) : 
    -- Define necessary conditions
    (no_two_planes_parallel : ∀ i j, i ≠ j → ¬parallel i j)
    (no_three_planes_intersect_one_line : ∀ i j k, i ≠ j → j ≠ k → k ≠ i → ¬intersect_one_line i j k)
    (no_four_planes_intersect_one_point : ∀ i j k l, i ≠ j → j ≠ k → k ≠ l → l ≠ i → 
         ¬intersect_one_point i j k l) :
    -- Prove the maximum number of regions
    let R (n : ℕ) := (n^3 + 5 * n) / 6 + 1 in
    R n = (n^3 + 5 * n) / 6 + 1 :=
by
  sorry

end max_regions_divided_by_planes_l604_604497


namespace axis_of_symmetry_shifted_l604_604895

theorem axis_of_symmetry_shifted (k : ℤ) : 
  ∃ k : ℤ, ∀ x, (∀ x, 2 * sin (2 * x) = 2 * sin (2 * (x + π / 12))) → 
    x = k * (π / 2) + π / 6 :=
sorry

end axis_of_symmetry_shifted_l604_604895


namespace problem_l604_604466

def operation (a b : ℤ) (h : a ≠ 0) : ℤ := (b - a) ^ 2 / a ^ 2

theorem problem : 
  operation (-1) (operation 1 (-1) (by decide)) (by decide) = 25 := 
by
  sorry

end problem_l604_604466


namespace jenna_reading_goal_l604_604942

theorem jenna_reading_goal (total_days : ℕ) (total_pages : ℕ) (unread_days : ℕ) (pages_on_23rd : ℕ) :
  total_days = 30 → total_pages = 600 → unread_days = 4 → pages_on_23rd = 100 →
  ∃ (pages_per_day : ℕ), 
  let days_to_read := total_days - unread_days - 1 in
  let pages_to_read_on_other_days := total_pages - pages_on_23rd in
  days_to_read ≠ 0 →
  pages_per_day * days_to_read = pages_to_read_on_other_days ∧ pages_per_day = 20 :=
by
  intros h1 h2 h3 h4
  use 20
  simp_all
  sorry

end jenna_reading_goal_l604_604942


namespace zuminglish_word_count_12_letters_mod_1000_l604_604049

def is_zuminglish_word (s : String) : Prop :=
  (∀ (i j : ℕ), i < j → s[i] = 'O' → s[j] = 'O' → (j - i > 3 ∧ (∀ k, i < k ∧ k < j → (s[k] = 'M' ∨ s[k] = 'P'))))

def zuminglish_word_count_mod_1000 (n : ℕ) : ℕ :=
  let a_n (n : ℕ) : ℕ := sorry -- to be defined recursively
  let b_n (n : ℕ) : ℕ := sorry -- to be defined recursively
  let c_n (n : ℕ) : ℕ := sorry -- to be defined recursively
  
  if n = 3 then 8
  else if n = 4 then 16
  else if n = 5 then 32
  else if n = 6 then 96
  else if n = 7 then 256
  else if n = 8 then 608
  else if n = 9 then 1616
  else if n = 10 then 4256
  else if n = 11 then 11264
  else a_n 12 + b_n 12 + c_n 12

theorem zuminglish_word_count_12_letters_mod_1000 :
  zuminglish_word_count_mod_1000 12 % 1000 = 472 :=
by
  sorry

end zuminglish_word_count_12_letters_mod_1000_l604_604049


namespace set_intersection_l604_604095

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2.5}

theorem set_intersection : A ∩ B = {0, 1, 2} :=
by
  sorry

end set_intersection_l604_604095


namespace vincent_weekly_earnings_l604_604632

-- Definitions of book prices
def fantasy_weekday_price : ℝ := 6
def fantasy_weekend_price : ℝ := 7
def literature_weekday_price := fantasy_weekday_price / 2
def literature_weekend_price := fantasy_weekend_price / 2
def mystery_weekday_price : ℝ := 4
def mystery_weekend_price := 4 * (1 - 0.1)

-- Definitions of books sold per day
def fantasy_books_per_day : ℕ := 5
def literature_books_per_day : ℕ := 8
def mystery_books_per_day : ℕ := 3

-- Definitions of sales duration
def weekdays : ℕ := 5
def weekends : ℕ := 2

-- Calculations
def weekly_earnings_fantasy :=
  (fantasy_weekday_price * fantasy_books_per_day * weekdays) + 
  (fantasy_weekend_price * fantasy_books_per_day * weekends)

def weekly_earnings_literature :=
  (literature_weekday_price * literature_books_per_day * weekdays) + 
  (literature_weekend_price * literature_books_per_day * weekends)

def weekly_earnings_mystery :=
  (mystery_weekday_price * mystery_books_per_day * weekdays) + 
  (mystery_weekend_price * mystery_books_per_day * weekends)

-- Total earnings
def total_weekly_earnings :=
  weekly_earnings_fantasy + weekly_earnings_literature + weekly_earnings_mystery

-- Theorem statement
theorem vincent_weekly_earnings : total_weekly_earnings = 477.6 := by
  sorry

end vincent_weekly_earnings_l604_604632


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604197

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l604_604197


namespace average_mpg_first_car_l604_604057

variable (X : ℝ)

theorem average_mpg_first_car:
  (∃ (X : ℝ), 
    let miles_first_car := 30 * X in
    let miles_second_car := 40 * 25 in
    let total_miles := miles_first_car + miles_second_car in
    total_miles = 1825 ∧
    30 + 25 = 55 ∧
    X = 27.5) :=
begin
  sorry
end

end average_mpg_first_car_l604_604057


namespace carlos_gold_quarters_l604_604741

theorem carlos_gold_quarters :
  (let quarter_weight := 1 / 5
       quarter_value := 0.25
       value_per_ounce := 100
       quarters_per_ounce := 1 / quarter_weight
       melt_value := value_per_ounce
       spend_value := quarters_per_ounce * quarter_value
    in melt_value / spend_value = 80) :=
by
  -- Definitions
  let quarter_weight := 1 / 5
  let quarter_value := 0.25
  let value_per_ounce := 100
  let quarters_per_ounce := 1 / quarter_weight
  let melt_value := value_per_ounce
  let spend_value := quarters_per_ounce * quarter_value

  -- Conclusion to be proven
  have h1 : quarters_per_ounce = 5 := sorry
  have h2 : spend_value = 1.25 := sorry
  have h3 : melt_value / spend_value = 80 := sorry

  show melt_value / spend_value = 80 from h3

end carlos_gold_quarters_l604_604741


namespace find_sides_of_triangle_ABC_find_angle_A_l604_604422

variable (a b c A B C : ℝ)

-- Part (Ⅰ)
theorem find_sides_of_triangle_ABC
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hArea : 1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3) :
  a = 2 ∧ b = 2 := sorry

-- Part (Ⅱ)
theorem find_angle_A
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hTrig : Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A)) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 := sorry

end find_sides_of_triangle_ABC_find_angle_A_l604_604422


namespace num_ordered_pairs_square_diff_120_l604_604849

theorem num_ordered_pairs_square_diff_120 :
  ∃ (count : ℕ), count = 4 ∧
  (∀ (m n : ℕ), m ≥ n ∧ n % 2 = 1 ∧ m^2 - n^2 = 120 → ∃ (pairs: list (ℕ × ℕ)), (m, n) ∈ pairs ∧ pairs.length = count) :=
sorry

end num_ordered_pairs_square_diff_120_l604_604849


namespace triangle_probability_l604_604407

noncomputable def probability_triangle (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 10) : ℝ :=
if h_triangle : x + y > z ∧ x + z > y ∧ y + z > x then 1 else 0

theorem triangle_probability : 
  ∫ x in 0..10, ∫ y in (10 - x)..10, if h_sum : x + y + (10 - x - y) = 10 then
    probability_triangle x y (10 - x - y) (by linarith) (by linarith) (by linarith) h_sum else 0 = 0.25 := 
by 
  sorry

end triangle_probability_l604_604407


namespace no_infinite_prime_sequence_l604_604985

open Nat

def is_prime_sequence (s : ℕ → ℕ) : Prop :=
  (∀ n, Prime (s n))

def satisfies_condition (s : ℕ → ℕ) : Prop :=
  ∀ k > 0, s k = 2 * s (k - 1) + 1 ∨ s k = 2 * s (k - 1) - 1

theorem no_infinite_prime_sequence (s : ℕ → ℕ) :
  is_prime_sequence s ∧ satisfies_condition s → False :=
by {
  sorry -- Placeholder for the actual proof
}

end no_infinite_prime_sequence_l604_604985


namespace circle_diameter_and_circumference_l604_604323

namespace CircleProof

-- Given conditions
def radius_from_area (A : ℝ) : ℝ := real.sqrt (A / real.pi)

def diameter_from_radius (r : ℝ) : ℝ := 2 * r

def circumference_from_radius (r : ℝ) : ℝ := 2 * real.pi * r

-- Prove that given the area of the circle is 16π, the diameter is 8 and the circumference is 8π.
theorem circle_diameter_and_circumference (A : ℝ) (hA : A = 16 * real.pi) :
  let r := radius_from_area A in 
  diameter_from_radius r = 8 ∧ circumference_from_radius r = 8 * real.pi :=
by
  sorry

end CircleProof

end circle_diameter_and_circumference_l604_604323


namespace meeting_point_distance_l604_604631

-- Definitions of the given conditions
def u : ℝ := 1.5
def v : ℝ := 3.0
def length_escalator : ℝ := 100.0
def person_up_speed : ℝ := (2 * v) / 3

-- Statement of the theorem
theorem meeting_point_distance : 
  (((2 * v) / 3 - u) * (length_escalator / ((v + u) + ((2 * v) / 3 - u)))) = 10 := 
by
  sorry

end meeting_point_distance_l604_604631


namespace intersection_of_sets_l604_604106

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l604_604106


namespace neg_disj_imp_neg_conj_l604_604032

theorem neg_disj_imp_neg_conj (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end neg_disj_imp_neg_conj_l604_604032


namespace elsa_amalie_coin_ratio_l604_604758

theorem elsa_amalie_coin_ratio
    (total_coins : ℕ)
    (remaining_amalie_coins : ℕ)
    (fraction_spent_by_amalie : ℚ)
    (elsa_coins amalie_coins : ℕ) :
    total_coins = 440 →
    remaining_amalie_coins = 90 →
    fraction_spent_by_amalie = 3 / 4 →
    elsa_coins + amalie_coins = total_coins →
    (1 - fraction_spent_by_amalie) * (amalie_coins : ℚ) = remaining_amalie_coins →
    elsa_coins / 10 = 8 →
    remaining_amalie_coins / 10 = 9 →
    elsa_coins : remaining_amalie_coins = 8:9 :=
by
  sorry

end elsa_amalie_coin_ratio_l604_604758


namespace numSolutions_eq_16_l604_604386

noncomputable def numSolutions (θ : ℝ) : ℕ :=
  let eqtn := tan (3 * π * cos θ) = cot (3 * π * sin θ)
  let domain := θ ∈ Ioo 0 (2 * π)
  if eqtn && domain then 1 else 0

theorem numSolutions_eq_16 : (finset.range 16).sum (λ i, numSolutions (i * 2 * π / 16 : ℝ)) = 16 :=
  by sorry

end numSolutions_eq_16_l604_604386


namespace inverse_proportional_x_y_l604_604571

theorem inverse_proportional_x_y (x y k : ℝ) (h_inverse : x * y = k) (h_given : 40 * 5 = k) : x = 20 :=
by 
  sorry

end inverse_proportional_x_y_l604_604571


namespace intersection_A_B_eq_C_l604_604101

noncomputable def A : Set ℤ := {-2, -1, 0, 1, 2}
noncomputable def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
noncomputable def C : Set ℝ := {0, 1, 2}

theorem intersection_A_B_eq_C : (A : Set ℝ) ∩ B = C :=
by {
  sorry
}

end intersection_A_B_eq_C_l604_604101


namespace zero_unique_in_interval_l604_604598

noncomputable def f (x : ℝ) : ℝ := 2^x + (x - 1)^3 - 2014

theorem zero_unique_in_interval : ∃! c ∈ set.Ioo 10 11, f c = 0 :=
begin
  -- Sorry is added because the proof is not required.
  sorry
end

end zero_unique_in_interval_l604_604598


namespace find_a_from_complex_l604_604431

theorem find_a_from_complex :
  ∃ a : ℝ, (let re_part := 2 * a - 4
                 im_part := 4 * a + 1
             in re_part = im_part) ↔ a = -5 / 2 :=
by
  sorry

end find_a_from_complex_l604_604431


namespace distinct_solutions_l604_604455

theorem distinct_solutions : 
  ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 - 7| = 2 * |x1 + 1| + |x1 - 3| ∧ |x2 - 7| = 2 * |x2 + 1| + |x2 - 3|) := 
by
  sorry

end distinct_solutions_l604_604455


namespace calculate_expression_l604_604394

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1 / x^2) * (y^2 + 1 / y^2) = x^4 - y^4 := by
  sorry

end calculate_expression_l604_604394


namespace find_f_log_3_54_l604_604327

noncomputable def f : ℝ → ℝ := sorry  -- Since we have to define a function and we do not need the exact implementation.

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_property : ∀ x : ℝ, f (x + 2) = - 1 / f x
axiom interval_property : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 3 ^ x

theorem find_f_log_3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 :=
by
  sorry


end find_f_log_3_54_l604_604327


namespace polynomial_properties_l604_604760

-- Define the polynomial P with integer coefficients
def P (x : ℤ) : ℤ

-- Condition that P(P(n) + n) is prime for infinitely many n
def condition (n : ℤ) : Prop := nat.prime (P (P n + n))

-- Define the main theorem we must prove
theorem polynomial_properties :
  (∀ ∞ n, condition n) →
  (∃ p, nat.prime p ∧ P = λ x, p) ∨
  (∃ k, odd k ∧ P = λ x, -2 * x + k) :=
sorry

end polynomial_properties_l604_604760


namespace proof_problem_l604_604960

variables (a b c : Type) [plane α β : Type] 
variables (subset : b ∈ α ∧ c ∉ α ∧ c || α) 
variables (proj : b ∈ β ∧ c = projection a β)

theorem proof_problem 
    (h1 : c ⊥ α ∧ c ⊥ β) 
    (h2 : b ∈ α ∧ c ∉ α ∧ c || α) 
    (h3 : b ∈ β ∧ c = projection a β ∧ b ⊥ c)
    (h4 : b ∈ β ∧ b ⊥ α) 
    : (h1 ∧ h2 ∧ h3 → ¬h4) := 
sorry

end proof_problem_l604_604960


namespace area_of_triangle_max_perimeter_of_triangle_l604_604473

-- Definition of triangle with sides opposite to angles
variable {A B C a b c : ℝ}
variable (triangle_ABC : Prop)
variable (side_opposite_A : a = c)
variable (side_opposite_B : b = c)
variable (side_opposite_C : c = c)
variable (side_c_value : c = 2)
variable (angle_C_value : C = π/3)

-- Given conditions in (I)
variable (condition_sin: 2 * sin (2 * A) + sin(2 * B + C) = sin C)

-- Define the area problem
def area_triangle :=
  ∃ (S : ℝ), S = (1 / 2) * a * b * sin C ∧ S = 2 * Real.sqrt 3 / 3

-- Define the perimeter problem
def max_perimeter_triangle : Prop :=
  a + b + c = 6

-- Questions (I) and (II) in Lean
theorem area_of_triangle :
  triangle_ABC →
  (side_opposite_A → side_opposite_B → side_opposite_C) →
  side_c_value →
  angle_C_value →
  condition_sin →
  area_triangle :=
by sorry

theorem max_perimeter_of_triangle :
  triangle_ABC →
  (side_opposite_A → side_opposite_B → side_opposite_C) →
  side_c_value →
  angle_C_value →
  max_perimeter_triangle :=
by sorry

end area_of_triangle_max_perimeter_of_triangle_l604_604473


namespace intersecting_planes_are_parallel_l604_604044

-- Define the conditions and the conclusion to be proved in Lean 4.
theorem intersecting_planes_are_parallel :
  ∀ (P₁ P₂ P₃ : Plane) (l₁ l₂ : Line),
    parallel P₁ P₂ →
    intersects P₃ P₁ l₁ →
    intersects P₃ P₂ l₂ →
    parallel l₁ l₂ :=
by
  sorry

end intersecting_planes_are_parallel_l604_604044


namespace angle_PQR_eq_90_l604_604923

theorem angle_PQR_eq_90
  (R S P Q : Type)
  [IsStraightLine R S P]
  (angle_QSP : ℝ)
  (h : angle_QSP = 70) :
  ∠PQR = 90 :=
by
  sorry

end angle_PQR_eq_90_l604_604923


namespace total_votes_is_900_l604_604680

-- Defining the conditions
variables (V : ℝ) (percent_winner : ℝ) (percent_loser : ℝ) (vote_majority : ℝ)

-- Given conditions
def condition1 : percent_winner = 0.70 := sorry
def condition2 : percent_loser = 0.30 := sorry
def condition3 : vote_majority = 360 := sorry
def condition4 : percent_winner * V - percent_loser * V = vote_majority := sorry

-- The theorem to prove
theorem total_votes_is_900 : V = 900 :=
by
  -- Use the conditions to show that V = 900
  have hw : percent_winner = 0.70 := condition1
  have hl : percent_loser = 0.30 := condition2
  have hm : vote_majority = 360 := condition3
  have heq : percent_winner * V - percent_loser * V = vote_majority := condition4
  sorry

end total_votes_is_900_l604_604680


namespace intersection_A_B_l604_604121

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l604_604121


namespace ring_roads_count_l604_604356

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem ring_roads_count : 
  binomial 8 4 * binomial 8 4 - (binomial 10 4 * binomial 6 4) = 1750 := by 
sorry

end ring_roads_count_l604_604356


namespace num_nonempty_subsets_of_B_l604_604449

def B : Set ℤ := {-1, 1, 4}

theorem num_nonempty_subsets_of_B : set.univ.nonempty_subsets B.card - 1 = 7 := by
  sorry

end num_nonempty_subsets_of_B_l604_604449


namespace Derek_more_than_Zoe_l604_604367

-- Define the variables for the number of books Emily, Derek, and Zoe have
variables (E : ℝ)

-- Condition: Derek has 75% more books than Emily
def Derek_books : ℝ := 1.75 * E

-- Condition: Zoe has 50% more books than Emily
def Zoe_books : ℝ := 1.5 * E

-- Statement asserting that Derek has 16.67% more books than Zoe
theorem Derek_more_than_Zoe (hD: Derek_books E = 1.75 * E) (hZ: Zoe_books E = 1.5 * E) :
  (Derek_books E - Zoe_books E) / Zoe_books E = 0.1667 :=
by
  sorry

end Derek_more_than_Zoe_l604_604367


namespace six_inch_sphere_value_l604_604326

def volume (r: ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def condition1 : ℝ := 4 -- radius of the sphere in inches

def value_of_four_inch_sphere : ℝ := 500

def condition2 (r: ℝ) := volume r

def six_inch_radius : ℝ := 6

def val_six_inch_sphere (v: ℝ) (a: ℝ) (sf: ℝ) : ℝ := a * sf

theorem six_inch_sphere_value :
  let volume4 := condition2 condition1 in
  let volume6 := condition2 six_inch_radius in
  let scaling_factor := volume6 / volume4 in
  val_six_inch_sphere volume6 value_of_four_inch_sphere scaling_factor = 1688 :=
by
  let volume4 : ℝ := (4 / 3) * Real.pi * 4^3
  let volume6 : ℝ := (4 / 3) * Real.pi * 6^3
  let scaling_factor : ℝ := volume6 / volume4
  let value_six_inch := value_of_four_inch_sphere * scaling_factor
  have round_value := Real.round value_six_inch 
  have expected_value := 1688
  sorry

end six_inch_sphere_value_l604_604326


namespace triangle_inequality_l604_604295

theorem triangle_inequality (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a = 3 ∧ b = 4 ∧ c = 5 := sorry

-- Given specific inputs
example : ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a + b > c ∧ a + c > b ∧ b + c > a :=
begin
  use [3, 4, 5],
  repeat {split},
  repeat {norm_num}
end

end triangle_inequality_l604_604295


namespace problem_conditions_main_proof_problem_l604_604150

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1
  else 2 * n - 5

noncomputable def b (n : ℕ) : ℝ :=
  a n / 2^n

noncomputable def S (n : ℕ) : ℝ :=
  n^2 - 4 * n + 4

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in finset.range (n+1), b i

theorem problem_conditions (n : ℕ) : S n = n^2 - 4 * n + 4 :=
  rfl

theorem main_proof_problem (n : ℕ) (n_pos : n > 0) : 1 / 4 ≤ T n ∧ T n < 1 :=
  sorry

end problem_conditions_main_proof_problem_l604_604150


namespace trajectory_is_ellipse_max_chord_length_l604_604484

-- Define the conditions and the parametric equations
def equation_of_circle (x y α : ℝ) : Prop := 
  x^2 + y^2 - 4 * x * Real.cos α - 2 * y * Real.sin α + 3 * (Real.cos α)^2 = 0

def parametric_line (t θ : ℝ) : (ℝ × ℝ) := 
  (t * Real.cos θ, 1 + t * Real.sin θ)

-- Define the parametric equations of the trajectory
def trajectory (α : ℝ) : (ℝ × ℝ) := 
  (2 * Real.cos α, Real.sin α)

theorem trajectory_is_ellipse : ∀ (x y α : ℝ), 
  (trajectory α = (x, y) → (x / 2)^2 + y^2 = 1) := 
sorry

theorem max_chord_length : ∀ (t α θ : ℝ), 
  let PQ_sq := (2 * Real.cos α - 0)^2 + (Real.sin α - 1)^2 in 
  -3 * (Real.sin α + 1 / 3)^2 + 16 / 3 ≤ PQ_sq :=
sorry

end trajectory_is_ellipse_max_chord_length_l604_604484


namespace limit_tg_ln_sine_l604_604745

theorem limit_tg_ln_sine (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x = tan x) → 
  tendsto (λ x, (f x - tan 2) / (sin (log (x - 1)))) (nhds 2) (nhds (1 / (cos 2)^2)) :=
by
  sorry

end limit_tg_ln_sine_l604_604745


namespace carlos_gold_quarters_l604_604730

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end carlos_gold_quarters_l604_604730


namespace warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l604_604700

noncomputable def netChange (tonnages : List Int) : Int :=
  List.sum tonnages

noncomputable def initialGoods (finalGoods : Int) (change : Int) : Int :=
  finalGoods + change

noncomputable def totalFees (tonnages : List Int) (feePerTon : Int) : Int :=
  feePerTon * List.sum (tonnages.map (Int.natAbs))

theorem warehouseGoodsDecreased 
  (tonnages : List Int) (finalGoods : Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20]) 
  (h2 : finalGoods = 580)
  (h3 : feePerTon = 4) : 
  netChange tonnages < 0 := by
  sorry

theorem initialTonnage 
  (tonnages : List Int) (finalGoods : Int) (change : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : finalGoods = 580)
  (h3 : change = netChange tonnages) : 
  initialGoods finalGoods change = 630 := by
  sorry

theorem totalLoadingFees 
  (tonnages : List Int) (feePerTon : Int)
  (h1 : tonnages = [21, -32, -16, 35, -38, -20])
  (h2 : feePerTon = 4) : 
  totalFees tonnages feePerTon = 648 := by
  sorry

end warehouseGoodsDecreased_initialTonnage_totalLoadingFees_l604_604700


namespace log_function_domain_correct_l604_604227

def log_function_domain : Set ℝ :=
  {x | 1 < x ∧ x < 3 ∧ x ≠ 2}

theorem log_function_domain_correct :
  (∀ x : ℝ, y = log (x - 1) (3 - x) → x ∈ log_function_domain) :=
by
  sorry

end log_function_domain_correct_l604_604227


namespace solve_inequality_l604_604214

theorem solve_inequality : {x : ℝ // abs (5 - 2 * x) < 3} = {x : ℝ // 1 < x ∧ x < 4} :=
begin
  sorry
end

end solve_inequality_l604_604214


namespace num_nonnegative_integers_in_form_l604_604024

theorem num_nonnegative_integers_in_form : 
  ∃ n : ℕ, n = 3^8 :=
begin
  use 6561,
  refl,
end

end num_nonnegative_integers_in_form_l604_604024


namespace intersection_of_sets_l604_604104

open Set

theorem intersection_of_sets : 
  let A := {-2, -1, 0, 1, 2}
  let B := {x : ℚ | 0 ≤ x ∧ x < 5/2}
  A ∩ B = {0, 1, 2} :=
by
  -- Lean's definition of finite sets uses List, need to convert List to Set for intersection
  let A : Set ℚ := {-2, -1, 0, 1, 2}
  let B : Set ℚ := {x | 0 ≤ x ∧ x < 5/2}
  let answer := {0, 1, 2}
  show A ∩ B = answer
  sorry

end intersection_of_sets_l604_604104


namespace product_of_points_on_log_graph_l604_604816

theorem product_of_points_on_log_graph 
  (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hx1 : y1 = log 3 x1) 
  (hx2 : y2 = log 3 x2) 
  (midpoint_condition : (y1 + y2) / 2 = 0) : 
  x1 * x2 = 1 := 
begin
  sorry
end

end product_of_points_on_log_graph_l604_604816


namespace correct_figure_is_D_l604_604486

def option_A : Prop := sorry -- placeholder for option A as a diagram representation
def option_B : Prop := sorry -- placeholder for option B as a diagram representation
def option_C : Prop := sorry -- placeholder for option C as a diagram representation
def option_D : Prop := sorry -- placeholder for option D as a diagram representation
def equilateral_triangle (figure : Prop) : Prop := sorry -- placeholder for the condition representing an equilateral triangle in the oblique projection method

theorem correct_figure_is_D : equilateral_triangle option_D := 
sorry

end correct_figure_is_D_l604_604486


namespace necessary_but_not_sufficient_condition_l604_604148

variable {F1 F2 M : Type} [MetricSpace M]

def is_constant_difference (F1 F2 : M) (M : M) : Prop :=
  ∃ c : ℝ, ∀ M : M, abs (dist M F1 - dist M F2) = c

def is_hyperbola_traj (F1 F2 : M) : Prop :=
  ∃ a : ℝ, ∀ M : M, abs (dist M F1 - dist M F2) = 2 * a

theorem necessary_but_not_sufficient_condition (F1 F2 : M) (M : M) :
  (is_hyperbola_traj F1 F2 → is_constant_difference F1 F2 M) ∧ ¬ (is_constant_difference F1 F2 M → is_hyperbola_traj F1 F2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l604_604148


namespace gold_quarter_value_comparison_l604_604728

theorem gold_quarter_value_comparison:
  (worth_in_store per_quarter: ℕ → ℝ) 
  (weight_per_quarter in_ounce: ℝ) 
  (earning_per_ounce melted: ℝ) : 
  (worth_in_store 4  = 0.25) →
  (weight_per_quarter = 1/5) →
  (earning_per_ounce = 100) →
  (earning_per_ounce * weight_per_quarter / worth_in_store 4 = 80) :=
by
  -- The proof goes here
  sorry

end gold_quarter_value_comparison_l604_604728


namespace no_6x6_prime_magic_square_l604_604078

def is_magic_square (n : ℕ) (matrix : list (list ℕ)) : Prop :=
  (∀ i, list.sum (matrix.nth i.getOrElse []) = n) ∧
  (∀ j, matrix.transpose[i].sum = n)

def sum_first_n_primes (n : ℕ) : ℕ :=
  primes.first_n n |>.sum

theorem no_6x6_prime_magic_square :
  ¬ ∃ (matrix : list (list ℕ)),
      is_magic_square 331 matrix ∧
      (∀ x ∈ matrix, x ∈ (List.range 2 152).filter(Prime))


end no_6x6_prime_magic_square_l604_604078


namespace carlos_gold_quarters_l604_604740

theorem carlos_gold_quarters :
  (let quarter_weight := 1 / 5
       quarter_value := 0.25
       value_per_ounce := 100
       quarters_per_ounce := 1 / quarter_weight
       melt_value := value_per_ounce
       spend_value := quarters_per_ounce * quarter_value
    in melt_value / spend_value = 80) :=
by
  -- Definitions
  let quarter_weight := 1 / 5
  let quarter_value := 0.25
  let value_per_ounce := 100
  let quarters_per_ounce := 1 / quarter_weight
  let melt_value := value_per_ounce
  let spend_value := quarters_per_ounce * quarter_value

  -- Conclusion to be proven
  have h1 : quarters_per_ounce = 5 := sorry
  have h2 : spend_value = 1.25 := sorry
  have h3 : melt_value / spend_value = 80 := sorry

  show melt_value / spend_value = 80 from h3

end carlos_gold_quarters_l604_604740


namespace Rafael_worked_tuesday_l604_604201

theorem Rafael_worked_tuesday 
  (total_amount_made : ℝ) (hourly_rate : ℝ) 
  (hours_worked_on_Monday : ℝ) (hours_left_to_work_in_week : ℝ) :
  total_amount_made = 760 ∧ hourly_rate = 20 ∧ hours_worked_on_Monday = 10 ∧ hours_left_to_work_in_week = 20 →
  let total_hours_worked := total_amount_made / hourly_rate in
  let total_hours_monday_tuesday := total_hours_worked - hours_left_to_work_in_week in
  let hours_worked_on_tuesday := total_hours_monday_tuesday - hours_worked_on_Monday in
  hours_worked_on_tuesday = 8 :=
  by
    intro h
    let ⟨ha, hb, hc, hd⟩ := h
    dsimp
    rw [ha, hb, hc, hd]
    norm_num


end Rafael_worked_tuesday_l604_604201


namespace folded_rectangle_length_eq_l604_604913

noncomputable def rectangle_folded_segment_length : ℝ :=
by
  let PQ := 4
  let QR := 12
  let EG := (fun PQ QR => real.sqrt((PQ ^ 2) + (8 / 3) ^ 2))
  exact EG 4 12

theorem folded_rectangle_length_eq :
  rectangle_folded_segment_length = (4 * real.sqrt 13) / 3 :=
by
  sorry

end folded_rectangle_length_eq_l604_604913


namespace solution_set_l604_604971

-- Define the conditions
variable (f : ℝ → ℝ)
variable (odd_func : ∀ x : ℝ, f (-x) = -f x)
variable (increasing_pos : ∀ a b : ℝ, 0 < a → 0 < b → a < b → f a < f b)
variable (f_neg3_zero : f (-3) = 0)

-- State the theorem
theorem solution_set (x : ℝ) : x * f x < 0 ↔ (-3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3) :=
sorry

end solution_set_l604_604971


namespace inequality_system_solution_l604_604570

theorem inequality_system_solution:
  ∀ (x : ℝ),
  (1 - (2*x - 1) / 2 > (3*x - 1) / 4) ∧ (2 - 3*x ≤ 4 - x) →
  -1 ≤ x ∧ x < 1 :=
by
  intro x
  intro h
  sorry

end inequality_system_solution_l604_604570


namespace handshake_count_l604_604724

theorem handshake_count (n_twins: ℕ) (n_triplets: ℕ)
  (twin_pairs: ℕ) (triplet_groups: ℕ)
  (handshakes_twin : ∀ (x: ℕ), x = (n_twins - 2))
  (handshakes_triplet : ∀ (y: ℕ), y = (n_triplets - 3))
  (handshakes_cross_twins : ∀ (z: ℕ), z = 3*n_triplets / 4)
  (handshakes_cross_triplets : ∀ (w: ℕ), w = n_twins / 4) :
  2 * (n_twins * (n_twins -1 -1) / 2 + n_triplets * (n_triplets - 1 - 1) / 2 + n_twins * (3*n_triplets / 4) + n_triplets * (n_twins / 4)) / 2 = 804 := 
sorry

end handshake_count_l604_604724


namespace log10_bounds_sum_of_consecutive_integers_between_log_bounds_l604_604840

noncomputable def log10_147583 := log 147583 / log 10

theorem log10_bounds :
  5 < log10_147583 ∧ log10_147583 < 6 :=
by
  sorry

theorem sum_of_consecutive_integers_between_log_bounds :
  let c := 5
  let d := 6
  c + d = 11 :=
by
  have h_bounds : 5 < log10_147583 ∧ log10_147583 < 6 := log10_bounds
  let c := 5 in
  let d := 6 in
  show c + d = 11 by rfl

end log10_bounds_sum_of_consecutive_integers_between_log_bounds_l604_604840


namespace sandy_correct_sums_l604_604561

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by
  sorry

end sandy_correct_sums_l604_604561


namespace convex_polygons_15_points_l604_604377

theorem convex_polygons_15_points : 
  ∃ n : ℕ, n = 2^15 - (nat.choose 15 0) - (nat.choose 15 1) - (nat.choose 15 2) - (nat.choose 15 3) ∧ n = 32192 :=
begin
  use 32192,
  sorry
end

end convex_polygons_15_points_l604_604377


namespace carlos_gold_quarters_l604_604732

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end carlos_gold_quarters_l604_604732
