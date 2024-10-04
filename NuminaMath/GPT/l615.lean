import Mathlib

namespace inequality_a_c_b_l615_615069

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615069


namespace group_can_cross_river_l615_615661

-- Definitions for the problem conditions
structure Person where
  is_bedouin : Bool
  is_wife : Bool

-- Define the persons
def B1 : Person := { is_bedouin := true, is_wife := false }
def B2 : Person := { is_bedouin := true, is_wife := false }
def B3 : Person := { is_bedouin := true, is_wife := false }
def W1 : Person := { is_bedouin := false, is_wife := true }
def W2 : Person := { is_bedouin := false, is_wife := true }
def W3 : Person := { is_bedouin := false, is_wife := true }

-- Define the initial state of the banks
def initial_left_bank : List Person := [B1, B2, B3, W1, W2, W3]
def initial_right_bank : List Person := []

-- Define the boat capacity
def boat_capacity : Nat := 2

-- Define the rule for the Bedouin and wives condition
def safe_state (left_bank right_bank : List Person) : Bool :=
  let count_bedouins (l : List Person) := l.countp (λ p => p.is_bedouin)
  let count_wives (l : List Person) := l.countp (λ p => p.is_wife)
  let unsafe_left := left_bank.any (λ p => p.is_wife && (count_bedouins left_bank > 0 && (count_bedouins left_bank < count_wives left_bank)))
  let unsafe_right := right_bank.any (λ p => p.is_wife && (count_bedouins right_bank > 0 && (count_bedouins right_bank < count_wives right_bank)))
  !(unsafe_left || unsafe_right)

-- Theorem statement
theorem group_can_cross_river : 
  ∃ (sequence_of_moves : List (List Person × List Person)), 
  (sequence_of_moves.head? = some (initial_left_bank, initial_right_bank)) ∧
  (sequence_of_moves.last? = some ([], initial_left_bank)) ∧
  (∀ (move : List Person × List Person) in sequence_of_moves, safe_state move.fst move.snd) :=
by
  sorry

end group_can_cross_river_l615_615661


namespace number_of_balanced_subsets_l615_615898

def is_balanced_subset (M B : Finset ℕ) : Prop :=
  B.nonempty ∧ (B.sum id) * M.card = (M.sum id) * B.card

theorem number_of_balanced_subsets :
  let M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  in (M.powerset.filter (is_balanced_subset M)).card = 51 :=
by
  -- Let M be the set {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Provide the conditions for B to be a balanced subset of M
  have H : M.sum id = 45 := rfl
  have n : M.card = 9 := rfl
  -- Define the balanced subset condition
  have balanced_cond (B : Finset ℕ) : Prop :=
    B.nonempty ∧ (B.sum id) * 9 = 45 * B.card
  -- Calculate the number of balanced subsets
  let balanced_subsets := M.powerset.filter balanced_cond
  have balanced_count : balanced_subsets.card = 51 := sorry
  exact balanced_count

end number_of_balanced_subsets_l615_615898


namespace union_complement_eq_l615_615498

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615498


namespace John_spent_15_dollars_on_soap_l615_615887

theorem John_spent_15_dollars_on_soap (number_of_bars : ℕ) (weight_per_bar : ℝ) (cost_per_pound : ℝ)
  (h1 : number_of_bars = 20) (h2 : weight_per_bar = 1.5) (h3 : cost_per_pound = 0.5) :
  (number_of_bars * weight_per_bar * cost_per_pound) = 15 :=
by
  sorry

end John_spent_15_dollars_on_soap_l615_615887


namespace max_height_l615_615668

def height (t : ℝ) : ℝ := -12 * t^2 + 48 * t + 25

theorem max_height : ∃ t, height t ≤ height 2 ∧ height 2 = 73 :=
by 
  use 2
  split
  sorry
  sorry

end max_height_l615_615668


namespace solution_for_x_l615_615749

theorem solution_for_x (x : ℝ) : 
  (∀ (y : ℝ), 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by 
  -- Proof should go here
  sorry

end solution_for_x_l615_615749


namespace union_complement_eq_target_l615_615487

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615487


namespace ordered_pair_solves_system_l615_615713

variables {x y : ℝ}

theorem ordered_pair_solves_system :
  (∃ x y : ℝ, 
    x + y = (7 - x) + (7 - y) ∧ 
    x - y = (x - 2) + (y - 2) ∧ 
    (x, y) = (3, 4)) :=
by
  use [3, 4]
  split
  sorry
  sorry

end ordered_pair_solves_system_l615_615713


namespace geometric_sequence_S4_over_a4_l615_615767

noncomputable theory
open_locale classical

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ)

-- Conditions
def is_geometric (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n+1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) :=
  S n = a 0 * (1 - (1/2)^n) / (1 - 1/2)

-- Question to prove
theorem geometric_sequence_S4_over_a4 :
  is_geometric a (1/2) →
  sum_first_n_terms a 4 S →
  a 0 = a1 →
  S 4 / a 4 = 15 :=
by
  assume h_geom h_sum ha1,
  -- The detailed proof would go here
  sorry

end geometric_sequence_S4_over_a4_l615_615767


namespace product_plus_sum_positive_probability_l615_615616

noncomputable def probability_positive_sum_product : ℝ :=
  let interval : set ℝ := {x : ℝ | -15 ≤ x ∧ x ≤ 15} in
  let measure_interval := λ s, ∫⁻ x in s, 1 in
  let measure_space := ⨅ (s : set ℝ), measure_interval s in
  let μ := measure_theory.measure_space.measureof interval measure_space in
  let prob {s t : set ℝ} (a b : s) : Prop := (a * b + a + b) > 0 in
  (∫⁻ a in interval, ∫⁻ b in interval, if prob a b then 1 else 0) / (μ interval * μ interval)

theorem product_plus_sum_positive_probability :
  probability_positive_sum_product = 3 / 8 :=
sorry

end product_plus_sum_positive_probability_l615_615616


namespace length_of_PF_l615_615871

theorem length_of_PF (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R]
  {PQ PR PL RM F: ℝ}
  (right_angle_P : angle P Q R = π / 2)
  (PQ_eq : PQ = 3)
  (PR_eq : PR = 3 * √3)
  (PL_intersects_RM_at_F : ∃ L M, altitude P L ∧ median R M ∧ intersect_at PL RM F) :
  length PF = 3 * √3 / 4 :=
begin
  sorry
end

end length_of_PF_l615_615871


namespace problem_l615_615532

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615532


namespace problem_l615_615524

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615524


namespace abs_five_minus_two_e_l615_615725

noncomputable def e : ℝ := 2.718

theorem abs_five_minus_two_e : |5 - 2 * e| = 0.436 := by
  sorry

end abs_five_minus_two_e_l615_615725


namespace problem_1_problem_2_problem_3_l615_615736

noncomputable def derivative_1 (x : ℝ) : ℝ := (x^3 + 1) * (x - 1)
noncomputable def expected_derivative_1 (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + x - 2

theorem problem_1 : ∀ x : ℝ, (derivative_1)' x = expected_derivative_1 x := 
by sorry

noncomputable def derivative_2 (x : ℝ) : ℝ := (real.cos x) / x
noncomputable def expected_derivative_2 (x : ℝ) : ℝ := ((-x * real.sin x) - real.cos x) / (x^2)

theorem problem_2 : ∀ x : ℝ, (derivative_2)' x = expected_derivative_2 x :=
by sorry

noncomputable def derivative_3 (x : ℝ) : ℝ := real.log (2 * x^2 + 1)
noncomputable def expected_derivative_3 (x : ℝ) : ℝ := 4 * x / (2 * x^2 + 1)

theorem problem_3 : ∀ x : ℝ, (derivative_3)' x = expected_derivative_3 x :=
by sorry

end problem_1_problem_2_problem_3_l615_615736


namespace probability_of_winning_l615_615989

theorem probability_of_winning (P_lose : ℚ) (h1 : P_lose = 5 / 8) : 1 - P_lose = 3 / 8 :=
by
  rw [h1]
  norm_num
  sorry -- This sorry avoids using the solution steps directly

#print probability_of_winning -- This can help in verifying that the statement is properly defined

end probability_of_winning_l615_615989


namespace exists_other_convex_polyhedron_l615_615882

/-- Definition: A triangular pyramid with opposite edges pairwise equal --/
structure TriangularPyramid := 
  (vertices : fin 4 → ℝ × ℝ × ℝ)
  (edges_equal : ∀ (i j : fin 4), i ≠ j → (dist (vertices i) (vertices j) = dist (vertices i.succ) (vertices j.succ)))

/-- Definition: A convex polyhedron --/
structure ConvexPolyhedron := 
  (faces : fin n → ℕ)

/-- Theorem: There exists another convex polyhedron that can be cut along some of its edges 
     and unfolded so that its net is a triangle without internal cuts. --/
theorem exists_other_convex_polyhedron (P : TriangularPyramid) : 
  ∃ (Q : ConvexPolyhedron), 
  (∃ (cut_edges : list (Q.faces)), 
  Q.faces.to_list.card = 3 ∧ 
  (∀ e ∈ cut_edges, e ∈ Q.faces.to_list)) :=
  sorry

end exists_other_convex_polyhedron_l615_615882


namespace hyperbola_center_l615_615253

theorem hyperbola_center (F1 F2 : ℝ × ℝ)
  (hF1 : F1 = (3, -2))
  (hF2 : F2 = (11, 6)) :
  let center := ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2)
  in center = (7, 2) := 
by 
  dsimp at *; -- dsimp simplifies the definitions
  rw [hF1, hF2];  -- using the given conditions
  dsimp;
  -- Applying the calculations step-by-step as described in the solution
  change ((3 + 11) / 2, (-2 + 6) / 2) with (7, 2);
  -- Justify the final change, even though it's clear mathematically
  exact rfl

end hyperbola_center_l615_615253


namespace donuts_cost_per_dozen_l615_615430

theorem donuts_cost_per_dozen
  (n_dozen : ℕ) (n_donuts_per_dozen : ℕ) (price_per_donut : ℝ) (total_profit : ℝ) :
  n_dozen = 10 →
  n_donuts_per_dozen = 12 →
  price_per_donut = 1 →
  total_profit = 96 →
  let total_donuts := n_dozen * n_donuts_per_dozen in
  let total_sales := total_donuts * price_per_donut in
  let cost_of_donuts := total_sales - total_profit in
  let cost_per_dozen := cost_of_donuts / n_dozen in
  cost_per_dozen = 2.4 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end donuts_cost_per_dozen_l615_615430


namespace students_get_same_result_l615_615311

def radius1 : ℝ := 1
def radius2 : ℝ := 10

def tangent_intersections := ∃ A B C : ℝ × ℝ, 
  dist A (0, 0) = radius2 ∧ 
  dist B (0, 0) = radius2 ∧ 
  dist C (0, 0) = radius2

def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  0.5 * abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2)))

def curvilinear_area (r angle : ℝ) : ℝ := 
  0.5 * r^2 * (angle - sin angle)

theorem students_get_same_result :
  ∀ A B C : ℝ × ℝ,
  dist A (0, 0) = radius2 → 
  dist B (0, 0) = radius2 → 
  dist C (0, 0) = radius2 →
  let S := triangle_area A B C,
      S_A := curvilinear_area radius2 (∠A_OB),
      S_B := curvilinear_area radius2 (∠B_OC),
      S_C := curvilinear_area radius2 (∠C_OA) 
  in
  S_A + S_B + S_C - S = 27 * Real.pi :=
by
  sorry

end students_get_same_result_l615_615311


namespace arithmetic_sequence_75th_term_l615_615201

theorem arithmetic_sequence_75th_term (a1 d : ℤ) (n : ℤ) (h1 : a1 = 3) (h2 : d = 5) (h3 : n = 75) :
  a1 + (n - 1) * d = 373 :=
by
  rw [h1, h2, h3]
  -- Here, we arrive at the explicitly stated elements and evaluate:
  -- 3 + (75 - 1) * 5 = 373
  sorry

end arithmetic_sequence_75th_term_l615_615201


namespace maximum_profit_l615_615864

def fixed_cost : ℝ := 2.5
def R : ℝ → ℝ
| x := if (0 < x) ∧ (x < 40) then 10 * x^2 + 100 * x else 701 * x + 10000 / x - 9450
def selling_price_per_phone : ℝ := 0.7
def revenue (x : ℝ) : ℝ := selling_price_per_phone * 1000 * x
def total_cost (x : ℝ) : ℝ := fixed_cost + R x

def profit (x : ℝ) : ℝ :=
  revenue x - total_cost x

def W (x : ℝ) : ℝ :=
  if (0 < x) ∧ (x < 40) then -10 * x^2 + 600 * x - 250
  else -(x + 10000 / x) + 9200

theorem maximum_profit : ∀ (x : ℝ), 0 < x → (W x ≤ 9000) :=
begin
  sorry
end

end maximum_profit_l615_615864


namespace fraction_of_time_at_15_mph_l615_615223

theorem fraction_of_time_at_15_mph
  (t1 t2 : ℝ)
  (h : (5 * t1 + 15 * t2) / (t1 + t2) = 10) :
  t2 / (t1 + t2) = 1 / 2 :=
by
  sorry

end fraction_of_time_at_15_mph_l615_615223


namespace spring_pizza_sales_l615_615260

theorem spring_pizza_sales 
  (winter_sales : ℝ := 4) 
  (winter_percent : ℝ := 0.20) 
  (summer_percent : ℝ := 0.30) 
  (fall_percent : ℝ := 0.25) 
  (total_percent : ℝ := 1) : 
  (spring_sales : ℝ := 0.25 * (winter_sales / winter_percent)) = 5 :=
by
  sorry

end spring_pizza_sales_l615_615260


namespace problem_l615_615526

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615526


namespace point_slope_form_l615_615362

theorem point_slope_form (k : ℝ) (p : ℝ × ℝ) (h_slope : k = 2) (h_point : p = (2, -3)) :
  (∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ y = 2 * (x - 2) + (-3)) := 
sorry

end point_slope_form_l615_615362


namespace angles_of_triangle_ABC_l615_615104

-- Definitions and conditions
variables (A M C B H : Type)
variables [Points : Point A, Point M, Point C, Point B, Point H]

-- Define that the triangle AMC is isosceles with M as the vertex of the equal sides and 
-- Angle AMC is acute.
def isosceles_triangle_AMC_isosceles_at_M := IsoscelesTriangle A M C ∧ Angle(A, M, C) < 90

-- Define that B is the symmetric point of A with respect to M (M is the midpoint of AB)
def symmetric_point_B := Midpoint M A B

-- Define H as the foot of the altitude from C in triangle ABC and that AH = HM
def foot_of_altitude_H := Foot_of_Altitude C (A, B)
def equal_heights_AH_HM := AH = HM

-- The final proof problem
theorem angles_of_triangle_ABC :
  isosceles_triangle_AMC_isosceles_at_M →
  symmetric_point_B →
  foot_of_altitude_H →
  equal_heights_AH_HM →
   (Angle(A, B, C) = 30 * degrees ∧ 
    Angle(B, C, A) = 90 * degrees ∧ 
    Angle(C, A, B) = 60 * degrees) :=
by
  intros
  sorry

end angles_of_triangle_ABC_l615_615104


namespace part1_part2_1_part2_2_l615_615610

-- Definitions for Part 1
def P_A_given_B := 1 / 1
def P_A_given_not_B := 1 / 4
def P_B := 1 / 2
def P_not_B := 1 / 2

theorem part1 : (P_B * P_A_given_B) / (P_B * P_A_given_B + P_not_B * P_A_given_not_B) = 4 / 5 := by
  sorry

-- Definitions for Part 2
def P_A1 := 1 / 2
def P_A2 := 1 / 3
def P_A3 := 1 / 6
def P_X_0 := 25 / 36
def P_X_2 := 1 / 4
def P_X_5 := 1 / 18
def expected_X := 7 / 9

theorem part2_1 : P_X_0 = 25 / 36 := by
  sorry

theorem part2_2 :
  (∀ x, x = 0 ∨ x = 2 ∨ x = 5 → if x = 0 then P_X_0 else if x = 2 then P_X_2 else P_X_5) ∧ (P_X_0 * 0 + P_X_2 * 2 + P_X_5 * 5 = expected_X) := by
  sorry

end part1_part2_1_part2_2_l615_615610


namespace pass_each_other_l615_615126

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def angular_speed (v : ℝ) (r : ℝ) : ℝ := v / circumference r * 2 * Real.pi

theorem pass_each_other (t : ℝ) (v_O : ℝ) (r_O : ℝ) (v_K : ℝ) (r_K : ℝ) (n : ℕ) :
  (t = 45) →
  (v_O = 200) →
  (r_O = 40) →
  (v_K = 240) →
  (r_K = 55) →
  (n = 67) →
  let ω_O := angular_speed v_O r_O,
      ω_K := angular_speed v_K r_K,
      ω_relative := ω_O + ω_K,
      k := 2 * Real.pi / ω_relative
  in floor (t / k) = n :=
begin
  sorry
end

end pass_each_other_l615_615126


namespace probability_of_exactly_one_defective_l615_615683

def total_number_ways_to_select_two (total_items : ℕ) : ℕ :=
  Nat.choose total_items 2

def number_favorable_outcomes (qualified : ℕ) (defective : ℕ) : ℕ :=
  Nat.choose qualified 1 * Nat.choose defective 1

theorem probability_of_exactly_one_defective
  (total_items : ℕ)
  (defective_items : ℕ)
  (qualified_items : ℕ)
  (h_total : total_items = defective_items + qualified_items)
  (h_total_items : total_items = 5)
  (h_defective : defective_items = 2)
  (h_qualified : qualified_items = 3) :
  (number_favorable_outcomes qualified_items defective_items) / (total_number_ways_to_select_two total_items) = 0.6 :=
by
  sorry

end probability_of_exactly_one_defective_l615_615683


namespace xiao_ming_school_time_l615_615634

noncomputable theory
open_locale classical

def arrives_at_morning : ℕ := 7 * 60 + 50 -- Xiao Ming arrives at 7:50 AM (converted to minutes)
def leaves_morning : ℕ := 11 * 60 + 50 -- Xiao Ming leaves at 11:50 AM (converted to minutes)
def arrives_afternoon : ℕ := 14 * 60 + 10 -- Xiao Ming arrives at 2:10 PM (converted to minutes)
def leaves_afternoon : ℕ := 17 * 60 -- Xiao Ming leaves at 5:00 PM (converted to minutes)

def time_spent_morning : ℕ := leaves_morning - arrives_at_morning
def time_spent_afternoon : ℕ := leaves_afternoon - arrives_afternoon
def total_time_spent : ℕ := time_spent_morning + time_spent_afternoon

theorem xiao_ming_school_time : total_time_spent = 410 := by
  sorry

end xiao_ming_school_time_l615_615634


namespace problem1_problem2_l615_615791

noncomputable def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2

-- Problem 1: Prove the general term form of the sequence
theorem problem1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2) :
  ∀ n, a n = 2^n := sorry

-- Problem 2: Prove the sum of the sequence na_n
noncomputable def T (n : ℕ) : ℕ := ∑ i in range n, (i + 1) * (2^(i + 1))

theorem problem2 (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2)
  (ha : ∀ n, a n = 2^n) :
  ∀ n, T n = ((n - 1) * 2^(n + 1)) + 2 := sorry

end problem1_problem2_l615_615791


namespace both_players_same_score_probability_l615_615356

theorem both_players_same_score_probability :
  let p_A_score := 0.6
  let p_B_score := 0.8
  let p_A_miss := 1 - p_A_score
  let p_B_miss := 1 - p_B_score
  (p_A_score * p_B_score + p_A_miss * p_B_miss = 0.56) :=
by
  sorry

end both_players_same_score_probability_l615_615356


namespace sum_series_l615_615297

theorem sum_series : (∑ k in (Finset.range 51), (if k % 2 = 0 then 2 * k + 1 else - (2 * (k - 1) + 1))) = 1 := by
  sorry

end sum_series_l615_615297


namespace greatest_number_of_Sundays_in_53_days_l615_615620

theorem greatest_number_of_Sundays_in_53_days (days_first_year : ℕ) (weeks_in_a_year : ℕ) 
    (days_in_a_week : ℕ) (sundays_in_a_week : ℕ) (days_in_year : ℕ):
    days_first_year = 53 →
    weeks_in_a_year = 52 →
    days_in_a_week = 7 →
    sundays_in_a_week = 1 →
    ∃ (n : ℕ), n = 7 := 
by
  intros h1 h2 h3 h4
  have weeks := (days_first_year / days_in_a_week)
  have remaining_days := (days_first_year % days_in_a_week)
  have total_sundays := (weeks * sundays_in_a_week)
  have max_sundays := if remaining_days > 0 then total_sundays else total_sundays
  use 7
  sorry

end greatest_number_of_Sundays_in_53_days_l615_615620


namespace probability_of_stopping_with_one_as_last_l615_615815

/-- 
  Consider the set A = {1, 2, 3, 4}. 
  Each second, Anders picks a number randomly from A (with replacement). 
  He stops picking when the sum of the last two picked numbers is a prime number. 
  We need to prove that the probability the last number picked is "1" is 15/44.
-/
theorem probability_of_stopping_with_one_as_last (A : set ℕ) (H : A = {1, 2, 3, 4}) :
  (∃ (p : ℚ), p = 15 / 44 ∧ 
  ∀ n1 n2 ∈ A, prime (n1 + n2) → n2 = 1) :=
begin
  -- Difficulty: The actual proof
  sorry
end

end probability_of_stopping_with_one_as_last_l615_615815


namespace average_games_per_month_l615_615190

theorem average_games_per_month (total_games : ℕ) (months : ℕ) 
  (h_total_games : total_games = 323) (h_months : months = 19) : 
  total_games / months = 17 := by
  -- Total games played: 323
  -- Months in the season: 19
  -- Calculation: 323 / 19 = 17
  rw [h_total_games, h_months]
  norm_num
  done -- ⊢ 323 / 19 = 17
  
  sorry

end average_games_per_month_l615_615190


namespace largest_of_7_consecutive_numbers_with_average_20_l615_615964

variable (n : ℤ) 

theorem largest_of_7_consecutive_numbers_with_average_20
  (h_avg : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6))/7 = 20) : 
  (n + 6) = 23 :=
by
  -- Placeholder for the actual proof
  sorry

end largest_of_7_consecutive_numbers_with_average_20_l615_615964


namespace smallest_five_digit_divisible_by_53_and_3_l615_615204

/-- The smallest five-digit positive integer divisible by 53 and 3 is 10062 -/
theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 ∧ n % 3 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0 → n ≤ m ∧ n = 10062 :=
by
  sorry

end smallest_five_digit_divisible_by_53_and_3_l615_615204


namespace sqrt_fraction_sum_as_common_fraction_l615_615214

theorem sqrt_fraction_sum_as_common_fraction (a b c d : ℚ) (ha : a = 25) (hb : b = 36) (hc : c = 16) (hd : d = 9) :
  Real.sqrt ((a / b) + (c / d)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_sum_as_common_fraction_l615_615214


namespace abc_geometric_progression_l615_615788

noncomputable def discriminant_eq_zero (k a b c : ℝ) : Prop :=
  (2 * k * b)^2 - 4 * k * a * (k * c) = 0

theorem abc_geometric_progression (k a b c : ℝ) (hk : k ≠ 0) (h_discriminant : discriminant_eq_zero k a b c) :
  b^2 = a * c :=
by
  have h1 : (2 * k * b)^2 = 4 * k^2 * b^2 := by ring
  have h2 : 4 * k * a * (k * c) = 4 * k^2 * a * c := by ring
  rw [discriminant_eq_zero] at h_discriminant
  simp only [h1, h2] at h_discriminant
  exact (mul_eq_zero.mp h_discriminant).resolve_left (pow_ne_zero 2 hk)

end abc_geometric_progression_l615_615788


namespace inequality_a_c_b_l615_615070

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615070


namespace union_complement_set_l615_615463

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615463


namespace solve_for_x_l615_615573

def expression (x : ℝ) : ℝ := sqrt (9 + sqrt (18 + 9 * x)) + sqrt (3 + sqrt (3 + x))

theorem solve_for_x : 
  (expression x = 3 + 3 * sqrt 3) ↔ x = 3 :=
begin
  sorry
end

end solve_for_x_l615_615573


namespace compare_abc_l615_615030

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615030


namespace range_of_a_l615_615909

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x + a / x + 7 else x + a / x - 7

theorem range_of_a (a : ℝ) (ha : 0 < a)
  (hodd : ∀ x : ℝ, f (-x) a = -f x a)
  (hcond : ∀ x : ℝ, 0 ≤ x → f x a ≥ 1 - a) :
  4 ≤ a := sorry

end range_of_a_l615_615909


namespace continuity_f_at_1_l615_615558

theorem continuity_f_at_1 (f : ℝ → ℝ) (x0 : ℝ)
  (h1 : f x0 = -12)
  (h2 : ∀ x : ℝ, f x = -5 * x^2 - 7)
  (h3 : x0 = 1) :
  ∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end continuity_f_at_1_l615_615558


namespace system1_solution_system2_solution_l615_615576

theorem system1_solution (p q : ℝ) 
  (h1 : p + q = 4)
  (h2 : 2 * p - q = 5) : 
  p = 3 ∧ q = 1 := 
sorry

theorem system2_solution (v t : ℝ)
  (h3 : 2 * v + t = 3)
  (h4 : 3 * v - 2 * t = 3) :
  v = 9 / 7 ∧ t = 3 / 7 :=
sorry

end system1_solution_system2_solution_l615_615576


namespace compare_abc_l615_615032

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615032


namespace compare_abc_l615_615046

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615046


namespace sets_of_products_not_identical_l615_615198

noncomputable def could_the_sets_be_identical : Prop :=
  let numbers := {i | 109 ≤ i ∧ i ≤ 208}
  let products_of_rows := {prod_r | ∃ row : Fin 10, prod_r = ∏ i in (Set.range 10), (row * 10 + i : Nat)}
  let products_of_columns := {prod_c | ∃ col : Fin 10, prod_c = ∏ i in (Set.range 10), (i * 10 + col : Nat)}
  ∀ (prod_r ∈ products_of_rows) (prod_c ∈ products_of_columns), prod_r ≠ prod_c

theorem sets_of_products_not_identical : could_the_sets_be_identical :=
  sorry

end sets_of_products_not_identical_l615_615198


namespace quartic_polynomial_irreducible_l615_615729

theorem quartic_polynomial_irreducible :
  ∃ (p : Polynomial ℝ), p.degree = 4 ∧ p.leadingCoeff = 1 ∧
  p.is_irreducible ∧
  (∀ (k : ℤ), 0 ≤ k ∧ k < 12 → p.eval (Complex.exp (2 * Real.pi * Complex.I * k / 12)) = 0) ∧ 
  roots p ⊆ {Complex.exp (2 * Real.pi * Complex.I * k / 12) | k : ℤ, 0 ≤ k ∧ k < 12} :=
begin
  use Polynomial.C 1 + Polynomial.X ^ 4 - Polynomial.X ^ 2,
  have h_deg : (Polynomial.C 1 + Polynomial.X ^ 4 - Polynomial.X ^ 2).degree = 4,
  { sorry }, -- Proof of the degree being 4
  have h_lc : (Polynomial.C 1 + Polynomial.X ^ 4 - Polynomial.X ^ 2).leadingCoeff = 1,
  { sorry }, -- Proof of the leading coefficient being 1
  have h_irr : (Polynomial.C 1 + Polynomial.X ^ 4 - Polynomial.X ^ 2).is_irreducible,
  { sorry }, -- Proof of irreducibility
  have h_roots : ∀ (k : ℤ), 0 ≤ k ∧ k < 12 → 
    (Polynomial.C 1 + Polynomial.X ^ 4 - Polynomial.X ^ 2).eval (Complex.exp (2 * Real.pi * Complex.I * k / 12)) = 0,
  { sorry }, -- Proof that the roots are twelfth roots of unity
  have h_sub : roots (Polynomial.C 1 + Polynomial.X ^ 4 - Polynomial.X ^ 2) ⊆ 
    {Complex.exp (2 * Real.pi * Complex.I * k / 12) | k : ℤ, 0 ≤ k ∧ k < 12},
  { sorry }, -- Proof that roots are subset of twelfth roots of unity
  exact ⟨Polynomial.C 1 + Polynomial.X ^ 4 - Polynomial.X ^ 2, h_deg, h_lc, h_irr, h_roots, h_sub⟩,
end

end quartic_polynomial_irreducible_l615_615729


namespace complex_conjugate_of_fraction_l615_615324

variable (a b : ℂ)

-- Define the complex number z
def z : ℂ := 2 / (1 - complex.i)

-- Define the complex conjugate function
def conj (x : ℂ) : ℂ := complex.conj x

-- State the theorem
theorem complex_conjugate_of_fraction :
  conj z = 1 - complex.i :=
by
  sorry

end complex_conjugate_of_fraction_l615_615324


namespace new_average_age_l615_615156

theorem new_average_age (n : ℕ) (n_val : n = 10) (avg_age : ℕ) (avg_age_val : avg_age = 15) 
(new_person_age : ℕ) (new_person_age_val : new_person_age = 37): 
  let total_age := avg_age * n in
  let new_total_age := total_age + new_person_age in
  let new_number_of_people := n + 1 in
  let new_avg_age := new_total_age / new_number_of_people in
  new_avg_age = 17 :=
by
  sorry

end new_average_age_l615_615156


namespace simplify_expression_l615_615951

variables (c b x a y z : ℝ)

theorem simplify_expression :
  (cx(bx^2 + 3a^2y^2 + c^2z^2) + bz(bx^2 + 3c^2x^2 + a^2y^2)) / (cx + bz) = bx^2 + a^2y^2 + c^2z^2 :=
by
  sorry

end simplify_expression_l615_615951


namespace food_last_days_l615_615692

theorem food_last_days
    (food_per_meal_d1 : ℕ := 250)
    (food_per_meal_d2 : ℕ := 350)
    (food_per_meal_d3 : ℕ := 450)
    (food_per_meal_d4 : ℕ := 550)
    (meals_per_day : ℕ := 2)
    (total_food_grams : ℕ := 100000) : ℕ :=
let total_food_per_meal := food_per_meal_d1 + food_per_meal_d2 + food_per_meal_d3 + food_per_meal_d4 in
let total_food_per_day := total_food_per_meal * meals_per_day in
total_food_grams / total_food_per_day

end food_last_days_l615_615692


namespace exists_natural_number_starting_and_ending_with_pattern_l615_615941

theorem exists_natural_number_starting_and_ending_with_pattern (n : ℕ) : 
  ∃ (m : ℕ), 
  (m % 10 = 1) ∧ 
  (∃ t : ℕ, 
    m^2 / 10^t = 10^(n - 1) * (10^n - 1) / 9) ∧ 
  (m^2 % 10^n = 1 ∨ m^2 % 10^n = 2) :=
sorry

end exists_natural_number_starting_and_ending_with_pattern_l615_615941


namespace min_max_values_l615_615455

theorem min_max_values (x1 x2 x3 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 ≥ 0) (h3 : x3 ≥ 0) (h_sum : x1 + x2 + x3 = 1) :
  1 ≤ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ∧ (x1 + 3*x2 + 5*x3) * (x1 + x2 / 3 + x3 / 5) ≤ 9/5 :=
by sorry

end min_max_values_l615_615455


namespace comparison_abc_l615_615087

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615087


namespace exists_non_intersecting_curve_proof_l615_615160

noncomputable def exists_non_intersecting_curve : Prop :=
  ∃ (curve : Set (ℝ × ℝ)), 
    (∀ p q ∈ curve, p ≠ q → LineSegment p q ∩ curve = {p, q}) ∧
    (curve ⊆ bounded_set ∧ curve ≠ circle) ∧
    (∃ tri : triangle, 
      is_equilateral tri ∧ 
      ∀ v ∈ vertices tri, 
        move_triangle_on_curve tri curve v)

-- Definitions of terms used in the theorem
def bounded_set : Set (ℝ × ℝ) := {p | ∃ r, r > 0 ∧ dist p (0, 0) < r}
def circle (radius : ℝ) : Set (ℝ × ℝ) := {p | dist p (0, 0) = radius}
def is_equilateral (tri : triangle) : Prop := ∀ (v₁ v₂ : ℝ × ℝ), v₁ ∈ vertices tri → v₂ ∈ vertices tri → dist v₁ v₂ = side_length tri

-- Presumed auxiliary definitions
def LineSegment (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry
def move_triangle_on_curve (tri : triangle) (curve : Set (ℝ × ℝ)) (vertex : ℝ × ℝ) : Prop := sorry
def vertices (tri : triangle) : Set (ℝ × ℝ) := sorry
def side_length (tri : triangle) : ℝ := sorry

theorem exists_non_intersecting_curve_proof : exists_non_intersecting_curve :=
  sorry

end exists_non_intersecting_curve_proof_l615_615160


namespace compare_a_b_c_l615_615039

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615039


namespace parabola_focus_distance_l615_615174

noncomputable def parabola := { p : ℝ // p = 2 }
noncomputable def line := { a : ℝ // a = 2 }
def A := (1 : ℝ, 2 : ℝ)
def F := (1 / 2 : ℝ, 0 : ℝ) -- Focus of the parabola y^2 = 2px is at (p/2, 0).

theorem parabola_focus_distance (p a : ℝ) (h_p : p = 2) (h_a : a = 2) (A : ℝ × ℝ) (F : ℝ × ℝ) : 
  -- Given conditions
  A = (1, 2) →
  F = (p / 2, 0) →
  let B := (4, 4 - 2 * 4) in
  -- The result
  real.dist F A + real.dist F B = 7 :=
by
  intros hA hF
  -- Further proofs would be written here
  sorry

end parabola_focus_distance_l615_615174


namespace exists_b_sequence_l615_615764

theorem exists_b_sequence (n : ℕ) (h1 : n ≥ 1) (a : Fin n → ℝ) :
  ∃ b : Fin n → ℝ, 
    (∀ i, ∃ k : ℤ, a i - b i = k ∧ k > 0) ∧ 
    (∑ i j in Finset.offDiag (Finset.univ : Finset (Fin n)), (b i - b j)^2 ≤ (n^2 - 1) / 12) :=
sorry

end exists_b_sequence_l615_615764


namespace trigonometric_identity_l615_615945

def tg (x : ℝ) : ℝ := Real.tan x
def ctg (x : ℝ) : ℝ := Real.cot x

theorem trigonometric_identity (x : ℝ) :
  tg x + 2 * tg (2 * x) + 4 * tg (4 * x) + 8 * ctg (8 * x) = ctg x := 
by sorry

end trigonometric_identity_l615_615945


namespace triangle_at_most_one_obtuse_l615_615628

theorem triangle_at_most_one_obtuse (T : Type) (triangle : T) : 
  (exists (A B C : T), is_triangle A B C triangle) -> 
  at_most_one_obtuse_triangle triangle :=
begin
  -- we assume for contradiction that there are at least two obtuse angles
  assume h : (exists (A B C : T), is_triangle A B C triangle),
  -- assuming there are at least two obtuse angles
  assume h_neg : at_least_two_obtuse triangle,
  
  sorry -- proof goes here
end

end triangle_at_most_one_obtuse_l615_615628


namespace total_length_is_correct_l615_615562

-- Define the conditions
def original_length : ℝ := 12
def additional_ratio : ℝ := 3 / 4

-- Define the additional length
def additional_length : ℝ := additional_ratio * original_length

-- Define the total length
def total_length : ℝ := original_length + additional_length

-- The statement to be proven
theorem total_length_is_correct : total_length = 21 := by
  sorry

end total_length_is_correct_l615_615562


namespace hyperbola_constant_ellipse_constant_l615_615382

variables {a b : ℝ} (a_pos_b_gt_a : 0 < a ∧ a < b)
variables {A B : ℝ × ℝ} (on_hyperbola_A : A.1^2 / a^2 - A.2^2 / b^2 = 1)
variables (on_hyperbola_B : B.1^2 / a^2 - B.2^2 / b^2 = 1) (perp_OA_OB : A.1 * B.1 + A.2 * B.2 = 0)

-- Hyperbola statement
theorem hyperbola_constant :
  (1 / (A.1^2 + A.2^2)) + (1 / (B.1^2 + B.2^2)) = 1 / a^2 - 1 / b^2 :=
sorry

variables {C D : ℝ × ℝ} (on_ellipse_C : C.1^2 / a^2 + C.2^2 / b^2 = 1)
variables (on_ellipse_D : D.1^2 / a^2 + D.2^2 / b^2 = 1) (perp_OC_OD : C.1 * D.1 + C.2 * D.2 = 0)

-- Ellipse statement
theorem ellipse_constant :
  (1 / (C.1^2 + C.2^2)) + (1 / (D.1^2 + D.2^2)) = 1 / a^2 + 1 / b^2 :=
sorry

end hyperbola_constant_ellipse_constant_l615_615382


namespace bill_toilet_paper_duration_l615_615694

variables (rolls : ℕ) (squares_per_roll : ℕ) (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ)

def total_squares (rolls squares_per_roll : ℕ) : ℕ := rolls * squares_per_roll

def squares_per_day (bathroom_visits_per_day squares_per_visit : ℕ) : ℕ := bathroom_visits_per_day * squares_per_visit

def days_supply_last (total_squares squares_per_day : ℕ) : ℕ := total_squares / squares_per_day

theorem bill_toilet_paper_duration
  (h1 : rolls = 1000)
  (h2 : squares_per_roll = 300)
  (h3 : bathroom_visits_per_day = 3)
  (h4 : squares_per_visit = 5)
  :
  days_supply_last (total_squares rolls squares_per_roll) (squares_per_day bathroom_visits_per_day squares_per_visit) = 20000 := sorry

end bill_toilet_paper_duration_l615_615694


namespace proof_problem_l615_615347

-- Definitions of the conditions
def y_condition (x y : ℝ) : Prop := y = Real.sqrt (x - 9) + Real.sqrt (9 - x) + 3
def star_operation (a b : ℝ) : ℝ := Real.sqrt a * Real.sqrt b - Real.sqrt a / Real.sqrt b

theorem proof_problem (x y : ℝ) (hx : y_condition x y) :
  x = 9 ∧ y = 3 ∧ star_operation x y = 2 * Real.sqrt 3 :=
by
  sorry

end proof_problem_l615_615347


namespace maximum_number_of_workers_l615_615271

theorem maximum_number_of_workers :
  ∀ (n : ℕ), n ≤ 5 → 2 * n + 6 ≤ 16 :=
by
  intro n h
  have hn : n ≤ 5 := h
  linarith

end maximum_number_of_workers_l615_615271


namespace drug_price_reduction_eq_l615_615193

variable (x : ℝ)
variable (initial_price : ℝ := 144)
variable (final_price : ℝ := 81)

theorem drug_price_reduction_eq :
  initial_price * (1 - x)^2 = final_price :=
by
  sorry

end drug_price_reduction_eq_l615_615193


namespace min_dist_A1_B1_l615_615769

theorem min_dist_A1_B1 (A B C A1 B1 C1 : ℝ)
  (h_tri : is_right_angle_isosceles_triangle A B C)
  (h_legs : A - B = 1 ∧ B - C = 1)
  (h_A1_on_AB : 0 ≤ A1 ∧ A1 ≤ 1)
  (h_B1_on_BC : 0 ≤ B1 ∧ B1 ≤ 1)
  (h_C1_on_AC : C1 = A + C)
  (h_similar : ∀ P Q R S T U, similar_triangles P Q R S T U ↔ is_right_angle_isosceles_triangle S T U) :
  (min_dist (A1, B1)) = (real.sqrt 5 / 5) := by
  sorry

end min_dist_A1_B1_l615_615769


namespace FirstCandidatePercentage_l615_615239

noncomputable def percentage_of_first_candidate_marks (PassingMarks TotalMarks MarksFirstCandidate : ℝ) :=
  (MarksFirstCandidate / TotalMarks) * 100

theorem FirstCandidatePercentage 
  (PassingMarks TotalMarks MarksFirstCandidate : ℝ)
  (h1 : PassingMarks = 200)
  (h2 : 0.45 * TotalMarks = PassingMarks + 25)
  (h3 : MarksFirstCandidate = PassingMarks - 50)
  : percentage_of_first_candidate_marks PassingMarks TotalMarks MarksFirstCandidate = 30 :=
sorry

end FirstCandidatePercentage_l615_615239


namespace scientific_notation_of_0_000000022_l615_615648

theorem scientific_notation_of_0_000000022 :
  (0.000000022 : ℝ) = 2.2 * 10 ^ (-8) :=
by
  sorry

end scientific_notation_of_0_000000022_l615_615648


namespace inequality_l615_615133

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c) ≤ 1 / (a * b * c) :=
sorry

end inequality_l615_615133


namespace melanie_bread_pieces_l615_615122

theorem melanie_bread_pieces :
  let first_slice := 1/2 * 3 + 1/2 * 4,
      second_slice := 1/3 * 2 + 2/3 * 5
  in
  first_slice + second_slice = 19 := by
  sorry

end melanie_bread_pieces_l615_615122


namespace sum_of_first_n_terms_l615_615768

-- Definition and conditions of the geometric sequence
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variable (n : ℕ)

-- The geometric sequence is positive
def positive_geometric_sequence : Prop :=
  ∀ n, a n > 0

-- Given conditions
def a2_eq_2 : Prop := 
  a 2 = 2

def S3_eq_2a3_minus_1 : Prop :=
  S 3 = 2 * a 3 - 1

-- Target: Prove that the sum of the first n terms S_n equals 2^n - 1
theorem sum_of_first_n_terms (n : ℕ) (h1 : positive_geometric_sequence) (h2 : a2_eq_2) (h3 : S3_eq_2a3_minus_1) :
  S n = 2^n - 1 :=
sorry

end sum_of_first_n_terms_l615_615768


namespace geometry_theorem_l615_615107

theorem geometry_theorem :
  ∀ (A B C T T* M A₁ B₁ : Point)
    (k : Circle)
    (hijk : IsoscelesTriangle A B C)
    (AC_eq_BC : distance A C = distance B C)
    (k_center : Center k C)
    (tangent_AT : Tangent A T k)
    (tangent_BT* : Tangent B T* k)
    (intersection_M : Intersect (Line A T) (Line B T*) M)
    (k₁ : Circle)
    (k₁_center : Center k₁ C)
    (k₁_radius : Radius k₁ (distance C M))
    (intersection_A1 : Intersect (Ray C A) (Circle k₁) A₁)
    (intersection_B1 : Intersect (Ray C B) (Circle k₁) B₁), 
  distance A₁ B₁ = distance M T + distance M T* :=
sorry

end geometry_theorem_l615_615107


namespace coeff_x_term_l615_615966

noncomputable def binomial_coeff (n k : ℕ) : ℕ := (nat.choose n k)

theorem coeff_x_term :
  let f := (1 - 2 / x^2) * (2 + sqrt x)^6 in
  coeff (λ x, f) 1 = 238 :=
by
  sorry

end coeff_x_term_l615_615966


namespace constant_term_in_binomial_expansion_l615_615418

theorem constant_term_in_binomial_expansion:
  let n := 6 in
  let x := (λ x : ℝ, (sqrt x - (1 / (2 * x)))^n) in
  (is_constant_term ((sqrt _) - (1 / (2 * _)))^n (15 / 4))

end constant_term_in_binomial_expansion_l615_615418


namespace ratio_of_perimeters_l615_615203

theorem ratio_of_perimeters (d : ℝ) (s1 s2 P1 P2 : ℝ) (h1 : d^2 = 2 * s1^2)
  (h2 : (3 * d)^2 = 2 * s2^2) (h3 : P1 = 4 * s1) (h4 : P2 = 4 * s2) :
  P2 / P1 = 3 := 
by sorry

end ratio_of_perimeters_l615_615203


namespace compare_abc_l615_615003

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615003


namespace rect_area_is_eight_l615_615159

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + 4*y^2 + 8*x - 16*y + 32 = 0

-- Define the height condition for the rectangle being twice the diameter of the circle
def rect_height (h : ℝ) : Prop :=
  h = 4

-- Define the width condition for the rectangle being the diameter of the circle
def rect_width (w : ℝ) : Prop :=
  w = 2

-- Define the area condition of the rectangle
def rect_area (A : ℝ) (h w : ℝ) : Prop :=
  A = h * w

-- The theorem statement to be proved
theorem rect_area_is_eight : ∃ h w A : ℝ, rect_height h ∧ rect_width w ∧ rect_area A h w ∧ A = 8 :=
begin
  -- Provide a proof outline, replace with actual proof as needed
  sorry
end

end rect_area_is_eight_l615_615159


namespace postal_cost_correct_l615_615756

def postal_cost (W : ℝ) : ℝ :=
  5 * min (Real.ceil W) 3 + 10 * max (Real.ceil W - 3) 0

theorem postal_cost_correct (W : ℝ) : postal_cost W = 5 * min (Real.ceil W) 3 + 10 * max (Real.ceil W - 3) 0 :=
by
  sorry

end postal_cost_correct_l615_615756


namespace probability_each_player_has_1_after_2023_rings_l615_615335

theorem probability_each_player_has_1_after_2023_rings :
  ∀ (rings : ℕ), rings = 2023 → 
  (initial_money : fin 4 → ℕ), (∀ i : fin 4, initial_money i = 1) → 
  (final_probability : ℚ), final_probability = 1/13 →
  true :=
by
  sorry

end probability_each_player_has_1_after_2023_rings_l615_615335


namespace graph_translation_equivalence_l615_615981

-- Definitions of the functions involved
def f1 (x : ℝ) : ℝ := sqrt 3 * cos (2 * x) - sin (2 * x)
def f2 (x : ℝ) : ℝ := 2 * sin (2 * x)

-- The transformation function
def translated (g : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := g (x + a)

-- Proof statement (translation to the left by π/3)
theorem graph_translation_equivalence : ∀ x : ℝ, f1 x = translated f2 -(π / 3) x := by
  sorry

end graph_translation_equivalence_l615_615981


namespace chrysler_building_floors_l615_615153

theorem chrysler_building_floors (L : ℕ) (Chrysler : ℕ) (h1 : Chrysler = L + 11)
  (h2 : L + Chrysler = 35) : Chrysler = 23 :=
by {
  --- proof would be here
  sorry
}

end chrysler_building_floors_l615_615153


namespace union_complement_eq_l615_615497

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615497


namespace fixed_point_of_f_l615_615794

noncomputable def f (a x : ℝ) : ℝ := log a (x - 1) + 1

theorem fixed_point_of_f (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : f a 2 = 1 :=
by sorry

end fixed_point_of_f_l615_615794


namespace compare_abc_l615_615059

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615059


namespace number_of_songs_in_album_l615_615926

/-- Lucy's jumping rope problem -/
theorem number_of_songs_in_album :
  let time_per_jump := 1 -- second per jump
  let total_jumps := 2100
  let song_length := 3.5 * 60 -- seconds per song
  total_jumps * time_per_jump / song_length = 10 := 
by
  -- Definitions as per the problem statement
  let time_per_jump := 1 -- second per jump
  let total_jumps := 2100
  let song_length := 3.5 * 60 -- seconds per song
  -- Perform the calculation 
  calculate
    calc
      total_jumps * time_per_jump / song_length = 2100 / 210 : by sorry_on_precalculation
      ... = 10 : by sorry

where
  sorry := sorry

end number_of_songs_in_album_l615_615926


namespace inclination_angle_range_l615_615807

theorem inclination_angle_range (θ : ℝ) : 
  ∃ α ∈ (Set.Icc 0 (Real.pi / 6) ∪ Set.Icc (5 * Real.pi / 6) Real.pi), 
    ∀ x y : ℝ, x * Real.cos θ + sqrt 3 * y - 1 = 0 ↔ y = x * tan α :=
sorry

end inclination_angle_range_l615_615807


namespace count_pairs_sets_condition_l615_615449

open Finset

/-! 
  We want to find the number of pairs of non-empty subsets A and B of {1, 2, ..., 10} such that 
  the smallest element of A is not less than the largest element of B. 
  We need to prove that the total number of such pairs is 9217.
-/

theorem count_pairs_sets_condition :
  let S := (range 10).map (λ n, n + 1) in
  let valid_pairs := 
    (S.powerset.filter (λ A, A ≠ ∅)).bind (λ A,
      (S.powerset.filter (λ B, B ≠ ∅ ∧ B.max ≤ A.min)).image (λ B, (A, B))) in
  valid_pairs.card = 9217 := sorry

end count_pairs_sets_condition_l615_615449


namespace total_possible_secret_codes_l615_615849

theorem total_possible_secret_codes (num_colors num_slots : ℕ) (h1 : num_colors = 5) (h2 : num_slots = 5) : num_colors ^ num_slots = 3125 :=
by
  rw [h1, h2]
  norm_num
  
-- Proof omitted
sorry

end total_possible_secret_codes_l615_615849


namespace police_catch_thief_l615_615411

-- Definitions corresponding to the conditions
def house_numbers : Set ℕ := {n | n ≥ 2}
def neighboring_houses (n m : ℕ) : Prop := (n = m + 1) ∨ (n + 1 = m)
def highest_prime_divisor (n : ℕ) : ℕ :=
  (list.filter nat.prime (nat.divisors n)).maximum'.get_or_else 0

-- Main theorem stating the police can catch the thief in finite time
theorem police_catch_thief (moves : ℕ → ℕ) :
  (∀ t, (moves (t + 1) = moves t + 1 ∨ moves (t + 1) + 1 = moves t) ∧
       ∀ t, highest_prime_divisor (moves (t + 1)) = highest_prime_divisor (moves t)) →
  ∃ t, moves t = moves (t + 7 * t) :=
sorry

end police_catch_thief_l615_615411


namespace relationship_among_a_b_c_l615_615786

variable {R : Type} [linearOrderedField R] [expRing R] [cosRing R]

def f (x : R) (m : R) : R :=
  if x ≥ 0 then -exp x + 1 + m * cos x else - (-exp (-x) + 1 + m * cos (-x))

def a (m : R) : R := -2 * f (-2) m
def b (m : R) : R := -f (-1) m
def c (m : R) : R := 3 * f 3 m

theorem relationship_among_a_b_c (m : R) (hm : m = 0) : c m < a m < b m :=
by {
  sorry
}

end relationship_among_a_b_c_l615_615786


namespace table_sortable_in_99_moves_l615_615398

structure Table (α : Type) :=
(n : ℕ)
(matrix : fin n → fin n → α)

def Table.rotated_by_180 {α : Type} [DecidableEq α] (t : Table α) (r1 c1 r2 c2 : ℕ) : Table α :=
{ matrix := λ i j, if r1 ≤ i ∧ i ≤ r2 ∧ c1 ≤ j ∧ j <= c2
                   then t.matrix (r1 + r2 - i) (c1 + c2 - j)
                   else t.matrix i j,
  .. t }

def Table.is_sorted (t : Table ℕ) : Prop :=
∀ i j i' j', i <= i' → j <= j' → t.matrix i j <= t.matrix i' j'

theorem table_sortable_in_99_moves :
  ∃ (moves : ℕ → Table ℕ → Table ℕ), ∀ (init_table : Table ℕ), 
    moves 99 init_table = Table.rotated_by_180 Table.initial
      → Table.is_sorted (moves 99 init_table) :=
by
  haveI := classical.dec_eq ℕ
  sorry

end table_sortable_in_99_moves_l615_615398


namespace ratio_eq_one_l615_615829

theorem ratio_eq_one {a b : ℝ} (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0 ∧ b ≠ 0) : (a^2 / 5) / (b^3 / 4) = 1 :=
by
  sorry

end ratio_eq_one_l615_615829


namespace min_abs_sum_of_extreme_points_of_symmetric_sin_cos_func_l615_615795

theorem min_abs_sum_of_extreme_points_of_symmetric_sin_cos_func :
  ∀ (a : ℝ) (x1 x2 : ℝ), 
  (∀ x, f x = sin x - a * cos x) →
  (∀ x, (f x = sin x - a * cos x) → x = (3/4) * π) →
  (∀ (x1 x2 : ℝ), x1 != x2 → (f x1 < f x2 ∨ f x2 < f x1)) →
  (|x1 + x2| = π / 2) :=
by
  sorry

end min_abs_sum_of_extreme_points_of_symmetric_sin_cos_func_l615_615795


namespace compare_abc_l615_615008

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615008


namespace relation_among_abc_l615_615092

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615092


namespace arithmetic_sequence_a2_value_l615_615361

open Nat

theorem arithmetic_sequence_a2_value (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 + a 3 = 12) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) : 
  a 2 = 5 :=
  sorry

end arithmetic_sequence_a2_value_l615_615361


namespace positive_area_triangles_count_l615_615421

namespace PositiveTriangles

-- Define the set of points in the Cartesian coordinate system where i, j = 1, 2, ..., 5
def points : list (ℕ × ℕ) := [(i, j) | i ← [1, 2, 3, 4, 5], j ← [1, 2, 3, 4, 5]]

-- Function to determine if three points are collinear
def is_collinear (p1 p2 p3 : ℕ × ℕ) : Prop :=
  let (x1, y1) := p1;
      (x2, y2) := p2;
      (x3, y3) := p3 in
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

-- Function to count non-collinear triangles
def non_collinear_triangles_count : ℕ :=
  points.to_finset.subsets_of_len 3.count_lens (λ t,
    (is_collinear t[0] t[1] t[2]) = false to_nat )

-- The theorem statement we want to prove
theorem positive_area_triangles_count : non_collinear_triangles_count points = 2148 :=
by
  sorry

end PositiveTriangles

end positive_area_triangles_count_l615_615421


namespace increasing_function_in_4_5_l615_615595

-- Define the derivative function and the interval
variables {f : ℝ → ℝ}
noncomputable def f' := derivative f

-- Condition that f' is positive in the interval (4,5)
axiom f'_positive_in_4_5 : ∀ x, 4 < x ∧ x < 5 → 0 < f' x

-- The proof statement
theorem increasing_function_in_4_5 : ∀ x y, 4 < x ∧ x < 5 ∧ 4 < y ∧ y < 5 ∧ x < y → f x < f y :=
sorry

end increasing_function_in_4_5_l615_615595


namespace root_in_interval_l615_615173

-- Definitions for the given problem conditions
variables (a b : ℝ)
variables (x1 x2 : ℝ)
hypothesis hb_pos : b > 0
hypothesis h_distinct : x1 ≠ x2
hypothesis h_roots : ∀ x, x^2 + a * x + b = 0 ↔ x = x1 ∨ x = x2
hypothesis h_interval : (x1 > -1 ∧ x1 < 1 ∨ x2 > -1 ∧ x2 < 1) ∧ (x1 ≤ -1 ∨ x1 ≥ 1 ∨ x2 ≤ -1 ∨ x2 ≥ 1)

-- The main statement to be proved
theorem root_in_interval : (x1 > -b ∧ x1 < b ∧ (x2 ≤ -b ∨ x2 ≥ b)) ∨ (x2 > -b ∧ x2 < b ∧ (x1 ≤ -b ∨ x1 ≥ b)) :=
sorry

end root_in_interval_l615_615173


namespace complex_number_in_third_quadrant_l615_615869

theorem complex_number_in_third_quadrant (z : ℂ) (h : z = (1-3*complex.i) / (1+2*complex.i)) : 
    z.re < 0 ∧ z.im < 0 :=
by sorry

end complex_number_in_third_quadrant_l615_615869


namespace ana_bonita_age_gap_l615_615688

theorem ana_bonita_age_gap (A B n : ℚ) (h1 : A = 2 * B + 3) (h2 : A - 2 = 6 * (B - 2)) (h3 : A = B + n) : n = 6.25 :=
by
  sorry

end ana_bonita_age_gap_l615_615688


namespace no_zero_sum_possible_between_1_and_10_l615_615988

-- Definitions reflecting the conditions
def sum_1_to_10 : ℕ := 55

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def possible_to_make_sum_zero (l : List Int) : Prop :=
  ∃ (signs : List Int) (h_sign_length : signs.length = l.length - 1),
    (∑ (i : ℕ) in Finset.range l.length.pred, (l[i]*signs[i]) = 0)

-- The Lean statement articulating the problem
theorem no_zero_sum_possible_between_1_and_10 (l : List Int) (h_list: l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) :
  ¬ possible_to_make_sum_zero l :=
by {
  have h_sum := sum_1_to_10,
  have h_odd := is_odd h_sum,
  sorry
}

end no_zero_sum_possible_between_1_and_10_l615_615988


namespace smallest_positive_period_center_of_symmetry_range_on_interval_l615_615364

-- Definition of the function f(x) as given
def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * (Real.sin x)^2 - 3

-- Simplify the function
def f_simplified (x : ℝ) : ℝ := 4 * (Real.sin (2 * x))^2 + 1

-- Question 1: Proving the smallest positive period is π
theorem smallest_positive_period : 
  (∀ (x : ℝ), f_simplified (x + Real.pi) = f_simplified x) ∧
  (∀ (T : ℝ), T > 0 → (∀ x : ℝ, f_simplified (x + T) = f_simplified x) → T ≥ Real.pi) :=
sorry

-- Question 2: Proving the center of symmetry
theorem center_of_symmetry (k : ℤ) : 
  Real.sin (2 * (k * Real.pi / 2)) = 0 ∧
  f_simplified (k * Real.pi / 2) = 1 :=
sorry

-- Question 3: Proving the range on the interval [ -π/4, π/4 ]
theorem range_on_interval : 
  ∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
  3 ≤ f_simplified x ∧ f_simplified x ≤ 5 :=
sorry

end smallest_positive_period_center_of_symmetry_range_on_interval_l615_615364


namespace compare_a_b_c_l615_615035

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615035


namespace p_necessary_but_not_sufficient_for_q_l615_615390

variable {a b : ℝ}

def p : Prop := -2 < a ∧ a < 0 ∧ 0 < b ∧ b < 1

def has_two_distinct_positive_roots (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ f x1 = 0 ∧ f x2 = 0

def q : Prop := has_two_distinct_positive_roots (λ x => x^2 + a*x + b)

theorem p_necessary_but_not_sufficient_for_q : (p → q) ∧ ¬ (q → p) :=
by
  sorry

end p_necessary_but_not_sufficient_for_q_l615_615390


namespace minimum_length_AP_correct_l615_615343

noncomputable def minimum_length_AP 
  (AD AB AA1 : ℝ) 
  (hAD : AD = 1) 
  (hAB : AB = 2) 
  (hAA1 : AA1 = 3)
  (P_in_plane : ∃ P, ∃ α β γ : ℝ, P = α • A1 + β • B + γ • D) : 
  ℝ :=
sorry

-- The theorem stating the minimum length AP given the conditions
theorem minimum_length_AP_correct :
  minimum_length_AP 1 2 3 
  (by rfl) (by rfl) (by rfl) 
  (by use [α, β, γ, h]) = 6 / 7 :=
sorry

end minimum_length_AP_correct_l615_615343


namespace Jackson_to_Williams_Ratio_l615_615428

-- Define the amounts of money Jackson and Williams have, given the conditions.
def JacksonMoney : ℤ := 125
def TotalMoney : ℤ := 150
-- Define Williams' money based on the given conditions.
def WilliamsMoney : ℤ := TotalMoney - JacksonMoney

-- State the theorem that the ratio of Jackson's money to Williams' money is 5:1
theorem Jackson_to_Williams_Ratio : JacksonMoney / WilliamsMoney = 5 := 
by
  -- Proof steps are omitted as per the instruction.
  sorry

end Jackson_to_Williams_Ratio_l615_615428


namespace compute_matrix_vector_l615_615907

open Matrix

-- Aliases for vector and matrix types
def Vector2 := Fin 2 → ℝ
def Matrix2 := Matrix (Fin 2) (Fin 2) ℝ

-- Given assumptions
variables (N : Matrix2)
variables (p q : Vector2)
variables (hp : N.mulVec p = ![3, -4])
variables (hq : N.mulVec q = ![-2, 6])

-- Goal
theorem compute_matrix_vector : N.mulVec (3 • p - 2 • q) = ![13, -24] :=
by 
  sorry

end compute_matrix_vector_l615_615907


namespace number_of_circles_l615_615199

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Define the side length of the square
def square_side_length : ℝ := 8

-- Define the diameter of a circle
def circle_diameter := 2 * circle_radius

-- Define the number of circles that can fit along one side of the square
def circles_per_side := square_side_length / circle_diameter

-- Define the total number of circles that can fit inside the square
def total_circles := circles_per_side * circles_per_side

-- The theorem to prove
theorem number_of_circles : total_circles = 4 := by
  -- Place holder for the proof
  sorry

end number_of_circles_l615_615199


namespace parkertown_working_at_home_trend_l615_615840

theorem parkertown_working_at_home_trend:
  (∃ f : ℕ → ℕ, f 1990 = 12 ∧ f 1995 = 15 ∧ f 2000 = 14 ∧ f 2005 = 28 ∧ 
  ∀ g ∈ {A, B, C, D}, 
  (g = C ↔ (f 1995 > f 1990 ∧ f 2000 < f 1995 ∧ f 2005 > f 2000))) :=
  sorry

end parkertown_working_at_home_trend_l615_615840


namespace number_of_ways_to_form_team_l615_615680

-- Defining the conditions
def total_employees : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def team_size : ℕ := 6
def men_in_team : ℕ := 4
def women_in_team : ℕ := 2

-- Using binomial coefficient to represent combinations
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proved
theorem number_of_ways_to_form_team :
  (choose num_men men_in_team) * (choose num_women women_in_team) = 
  choose 10 4 * choose 5 2 :=
by
  sorry

end number_of_ways_to_form_team_l615_615680


namespace smallest_integer_in_consecutive_odds_l615_615985

theorem smallest_integer_in_consecutive_odds (median greates_int smallest_int : ℤ) 
  (h_median : median = 155) 
  (h_greates_int : greates_int = 167) 
  (h_condition : set_of_consecutive_odds median greates_int) :
  smallest_int = 143 := 
  sorry

end smallest_integer_in_consecutive_odds_l615_615985


namespace determine_xyz_l615_615717

theorem determine_xyz (x y z : ℝ) 
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 12) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) : 
  x * y * z = -4 / 3 := 
sorry

end determine_xyz_l615_615717


namespace range_of_m_l615_615379

open Set Real

-- Define over the real numbers ℝ
noncomputable def A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 ≤ 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0 }
noncomputable def CRB (m : ℝ) : Set ℝ := { x : ℝ | x < m - 2 ∨ x > m + 2 }

-- Main theorem statement
theorem range_of_m (m : ℝ) (h : A ⊆ CRB m) : m < -3 ∨ m > 5 :=
sorry

end range_of_m_l615_615379


namespace seven_power_expression_l615_615447

theorem seven_power_expression (x y z : ℝ) (h₀ : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) (h₂ : xy + xz + yz ≠ 0) :
  (x^7 + y^7 + z^7) / (xyz * (x^2 + y^2 + z^2)) = 14 :=
by
  sorry

end seven_power_expression_l615_615447


namespace binary_to_decimal_1110011_l615_615707

theorem binary_to_decimal_1110011 : 
  let b := [1, 1, 1, 0, 0, 1, 1],
      n := 2 in
  (∑ i in List.range (b.length), b[i] * n^i) = 115 :=
by
  let b := [1, 1, 1, 0, 0, 1, 1]
  let n := 2
  have h : (∑ i in List.range (b.length), b[i] * n^i) = 115 := sorry
  exact h

end binary_to_decimal_1110011_l615_615707


namespace magnitude_of_b_parallel_to_a_l615_615383

-- Define the vectors a and b
def a (m : ℝ) : ℝ × ℝ := (1, m + 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

-- Introduction of conditions
theorem magnitude_of_b_parallel_to_a (m : ℝ) (h_para : a m.1 * b m.2 = b m.1 * a m.2): 
  |((m=-1) -> sqrt(((b m.1)^2) + ((b m.2)^2)) = sqrt(2)) := sorry

end magnitude_of_b_parallel_to_a_l615_615383


namespace new_volume_is_270_l615_615602

-- Define the original volume condition
def original_volume (r h : ℝ) : Prop := (Real.pi * r^2 * h = 15)

-- Define the transformation to the new dimensions
def new_dimensions (r h : ℝ) (r' h' : ℝ) : Prop := 
  (r' = 3 * r) ∧ (h' = 2 * h)

-- Define the volume function
def volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Define the new volume
def new_volume (r h : ℝ) (r' h' : ℝ) : ℝ := volume r' h'

-- The theorem to prove
theorem new_volume_is_270 (r h r' h' : ℝ) 
  (hv : original_volume r h)
  (hd : new_dimensions r h r' h') :
  new_volume r h r' h' = 270 := by
  sorry

end new_volume_is_270_l615_615602


namespace min_intersection_size_l615_615755

open Set

-- Define the given conditions
def n (S : Set α) : ℕ := 2 ^ S.size

-- Given sets A, B, and C
variables (A B C : Set α)

-- Given conditions
axiom h1 : n A + n B + n C = n (A ∪ B ∪ C)
axiom h2 : A.size = 100
axiom h3 : B.size = 100

-- Conclusion that needs to be proved
theorem min_intersection_size : (A ∩ B ∩ C).size = 97 :=
sorry

end min_intersection_size_l615_615755


namespace A_lt_one_tenth_l615_615555

noncomputable def A : ℝ :=
  (finset.Ico 1 50).prod (λ n, (2*n - 1 : ℝ) / (2*n : ℝ))

theorem A_lt_one_tenth : A < 1 / 10 := 
by
  have hA : A = (finset.Ico 1 50).prod (λ n, (2*n - 1 : ℝ) / (2*n : ℝ)) := rfl
  sorry

end A_lt_one_tenth_l615_615555


namespace pencils_per_person_l615_615955

theorem pencils_per_person (x : ℕ) (h : 3 * x = 24) : x = 8 :=
by
  -- sorry we are skipping the actual proof
  sorry

end pencils_per_person_l615_615955


namespace compute_factorial_expression_l615_615700

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Problem statement
theorem compute_factorial_expression : (factorial 12 / factorial 11) - factorial 7 = -5028 := by
  sorry

end compute_factorial_expression_l615_615700


namespace min_sum_a_b_c_l615_615777

open Nat

theorem min_sum_a_b_c (a b c : ℕ) (h_lcm : lcm (lcm a b) c = 48)
  (h_gcd_ab : gcd a b = 4) (h_gcd_bc : gcd b c = 3) : a + b + c = 31 :=
  sorry

end min_sum_a_b_c_l615_615777


namespace count_negative_numbers_l615_615978

theorem count_negative_numbers :
  let a := -|(-1 : ℤ)|
  let b := -(3 : ℤ) ^ 2
  let c := (-(1 / 2 : ℚ)) ^ 3
  let d := -((2 / 3 : ℚ) ^ 2)
  let e := -((-1 : ℤ) ^ 2021)
  (∃ l : List ℤ, l = [a, b, c, d, e] ∧ l.filter (λ x, x < 0) = [a, b, c, d]) :=
by
  sorry

end count_negative_numbers_l615_615978


namespace sequence_problem_l615_615771

theorem sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1)) (h_eq : a 100 = a 96) :
  a 2018 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_problem_l615_615771


namespace different_graphs_l615_615213

theorem different_graphs :
  ¬((∀ x, (y = x - 1) ↔ (y = (x^2 - 1) / (x + 1))) ∧
    (∀ x, (y = x - 1) ↔ ((x + 1) * y = x^2 - 1)) ∧
    (∀ x, (y = (x^2 - 1) / (x + 1)) ↔ ((x + 1) * y = x^2 - 1))) :=
begin
  sorry
end

end different_graphs_l615_615213


namespace acute_angle_cosine_sum_inequality_l615_615753

theorem acute_angle_cosine_sum_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  cos (α + β) < cos α + cos β :=
sorry

end acute_angle_cosine_sum_inequality_l615_615753


namespace solve_for_x_l615_615747

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) → x = 3 / 2 := by
  sorry

end solve_for_x_l615_615747


namespace jimmy_income_l615_615137

theorem jimmy_income (r_income : ℕ) (r_increase : ℕ) (combined_percent : ℚ) (j_income : ℕ) : 
  r_income = 15000 → 
  r_increase = 7000 → 
  combined_percent = 0.55 → 
  (combined_percent * (r_income + r_increase + j_income) = r_income + r_increase) → 
  j_income = 18000 := 
by
  intros h1 h2 h3 h4
  sorry

end jimmy_income_l615_615137


namespace det_matrixE_l615_615905

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end det_matrixE_l615_615905


namespace compare_abc_l615_615002

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615002


namespace prime_triple_l615_615732

theorem prime_triple (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (p * r - 1)) (h3 : r ∣ (p * q - 1)) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
sorry

end prime_triple_l615_615732


namespace sum_of_geometric_sequence_l615_615125

theorem sum_of_geometric_sequence :
  ∀ (a_1 q : ℤ), (a_1 * q^5 - a_1 * q^3 = 24) → (a_1 * q^2 * a_1 * q^4 = 64) →
  let S_8 := a_1 * (1 - q^8) / (1 - q) in
    S_8 = 255 ∨ S_8 = 85 :=
by
  intros a_1 q h1 h2
  let S_8 := a_1 * (1 - q^8) / (1 - q)
  sorry

end sum_of_geometric_sequence_l615_615125


namespace train_length_is_approx_605_l615_615678

noncomputable def length_of_train (train_speed man_speed time : ℝ) : ℝ :=
  let relative_speed := (train_speed + man_speed) * (5 / 18) -- converting km/hr to m/s
  relative_speed * time

theorem train_length_is_approx_605 :
  length_of_train 60 6 32.99736021118311 ≈ 605 :=
by
  -- skipping the proof
  sorry

end train_length_is_approx_605_l615_615678


namespace subset_coloring_l615_615899

theorem subset_coloring (S : Finset (Fin 2002)) (N : ℕ) (hN : 0 ≤ N ∧ N ≤ 2^2002) :
  ∃ (coloring : Finset (Finset (Fin 2002)) → Bool),
    (∀ A B : Finset (Finset (Fin 2002)), (coloring A = true ∧ coloring B = true) → coloring (A ∪ B) = true) ∧
    (∀ A B : Finset (Finset (Fin 2002)), (coloring A = false ∧ coloring B = false) → coloring (A ∪ B) = false) ∧
    ((Finset.filter (λ x, coloring x = true) (Finset.powerset S)).card = N) := by
sorry

end subset_coloring_l615_615899


namespace geometric_series_sum_l615_615944

theorem geometric_series_sum (n : ℕ) (u1 q : ℝ) (hq : q ≠ 1) : 
  let S_n := ∑ i in Finset.range n, u1 * q^i in
  S_n = u1 * (q^n - 1) / (q - 1) :=
sorry

end geometric_series_sum_l615_615944


namespace largest_product_of_three_l615_615685

-- Definitions of the numbers in the set
def numbers : List Int := [-5, 1, -3, 5, -2, 2]

-- Define a function to calculate the product of a list of three integers
def product_of_three (a b c : Int) : Int := a * b * c

-- Define a predicate to state that 75 is the largest product of any three numbers from the given list
theorem largest_product_of_three :
  ∃ (a b c : Int), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ product_of_three a b c = 75 :=
sorry

end largest_product_of_three_l615_615685


namespace side_length_DF_l615_615403

variable (DEF : Type) [MetricSpace DEF] [HAS_NORM DEF]
variable (E D F : DEF)
variable (right_angle_E : ∠ DEF D E = π / 2)
variable (length_EF : (∥ E - F ∥ : ℝ) = Real.sqrt 145)
variable (cosine_D : cos (angle D E F) = 16 * Real.sqrt 145 / 145)

theorem side_length_DF : (∥ D - F ∥ : ℝ) = 16 := by
  sorry

end side_length_DF_l615_615403


namespace problem_propositions_l615_615348

-- Definitions for the conditions
def p : Prop := ∀ (T₁ T₂ : Triangle), (area T₁ = area T₂) → (T₁ ≅ T₂)
def q : Prop := ∀ (m : ℝ), (m ≤ 1) → ∃ (x : ℝ), x^2 - 2*x + m = 0

-- Stating the problem
theorem problem_propositions :
  (¬p ∧ q) →
  (p ∨ q) ∧ (¬ (p ∧ q)) ∧ (¬(p ∨ ¬q)) :=
by
  sorry

end problem_propositions_l615_615348


namespace chrysler_building_floors_l615_615152

theorem chrysler_building_floors (L : ℕ) (Chrysler : ℕ) (h1 : Chrysler = L + 11)
  (h2 : L + Chrysler = 35) : Chrysler = 23 :=
by {
  --- proof would be here
  sorry
}

end chrysler_building_floors_l615_615152


namespace every_algorithm_must_have_sequential_structure_l615_615609

def is_sequential_structure (alg : Type) : Prop := sorry -- This defines what a sequential structure is

def must_have_sequential_structure (alg : Type) : Prop :=
∀ alg, is_sequential_structure alg

theorem every_algorithm_must_have_sequential_structure :
  must_have_sequential_structure nat := sorry

end every_algorithm_must_have_sequential_structure_l615_615609


namespace problem_solution_l615_615208

/-- Define the repeating decimal 0.\overline{49} as a rational number. --/
def rep49 := 7 / 9

/-- Define the repeating decimal 0.\overline{4} as a rational number. --/
def rep4 := 4 / 9

/-- The main theorem stating that 99 times the difference between 
    the repeating decimals 0.\overline{49} and 0.\overline{4} equals 5. --/
theorem problem_solution : 99 * (rep49 - rep4) = 5 := by
  sorry

end problem_solution_l615_615208


namespace polynomial_property_l615_615175

theorem polynomial_property (P : ℕ → ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :
  (∀ x y, P x y = P (x + 1) (y + 1)) →
  (∀ x y, P x y = ∑ k in Finset.range (n + 1), a k * (x - y) ^ k) :=
sorry

end polynomial_property_l615_615175


namespace proof_problem_l615_615780

-- Define the moving point A on the circle Gamma
def is_on_circle (A : ℝ × ℝ) : Prop :=
  let (x, y) := A in (x - 4)^2 + y^2 = 36

-- Define the point B
def B : ℝ × ℝ := (-2, 0)

-- Define the point P on line segment AB satisfying |BP| / |AP| = 1 / 2
def is_on_line_segment_AB (A P B : ℝ × ℝ) : Prop :=
  let (Ax, Ay) := A in
  let (Px, Py) := P in
  let BP_dist : ℝ := real.sqrt ((Px - (-2))^2 + (Py - 0)^2) in
  let AP_dist : ℝ := real.sqrt ((Px - Ax)^2 + (Py - Ay)^2) in
  2 * BP_dist = AP_dist

-- Define the equation of the trajectory C of point P
def trajectory_C (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 + y^2 = 4

-- Define the line l passing through (-1, 3)
def line_l (l : ℝ → ℝ) : Prop := l (-1) = 3

-- Define if l intersects C at M and N with |MN| = 2sqrt(3)
def intersects_C_with_MN (l : ℝ → ℝ) (M N : ℝ × ℝ) : Prop :=
  let (Mx, My) := M in
  let (Nx, Ny) := N in
  trajectory_C M ∧ trajectory_C N ∧ real.sqrt ((Mx - Nx)^2 + (My - Ny)^2) = 2 * real.sqrt 3

-- The main statement to prove the conditions
theorem proof_problem :
  ∀ (A P M N : ℝ × ℝ) (l : ℝ → ℝ),
    is_on_circle A →
    is_on_line_segment_AB A P B →
    trajectory_C P →
    line_l l →
    intersects_C_with_MN l M N →
    (∀ x, l x = (-4 / 3) * x + 1 / 3) ∨ (∀ x, x = -1) :=
by sorry

end proof_problem_l615_615780


namespace union_complement_eq_target_l615_615485

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615485


namespace product_of_functions_l615_615371

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := -(3 * x - 1) / x

theorem product_of_functions (x : ℝ) (h : x ≠ 0) : f x * g x = -6 * x + 2 := by
  sorry

end product_of_functions_l615_615371


namespace exists_even_common_enemies_l615_615673

def is_friend_or_enemy (a b : Fin 100) : Prop := sorry

def common_enemies (a b c : Fin 100) : Prop := sorry

theorem exists_even_common_enemies :
  (∀ (a b : Fin 100), is_friend_or_enemy a b) →
  ∃ (a b : Fin 100), ∃ n : ℕ, even n ∧ common_enemies a b n :=
sorry

end exists_even_common_enemies_l615_615673


namespace part1_proof_part2_proof_l615_615459

-- Definition of the conditions
variables {A B : set ℝ} {α : ℝ}

-- Conditions for question 1
def condition1 (A B : set ℝ) (α : ℝ) : Prop :=
∀ (α > 0), A ⊆ {y + n * α | (y ∈ B) ∧ (n ∈ ℤ)} ∧ B ⊆ {x + m * α | (x ∈ A) ∧ (m ∈ ℤ)}

-- Proof that A ≠ B under the conditions for question 1
theorem part1_proof (h : condition1 A B α) : A ≠ B := 
sorry

-- Condition for question 2 where B is a bounded set
def bounded_set (B : set ℝ) : Prop :=
∃ (M : ℝ), ∀ (y ∈ B), abs y < M

-- Proof that A = B given the above conditions and that B is bounded
theorem part2_proof (h1 : condition1 A B α) (h2 : bounded_set B) : A = B :=
sorry

end part1_proof_part2_proof_l615_615459


namespace area_of_Q1Q3Q5Q7_l615_615670

def regular_octagon_apothem : ℝ := 3

def area_of_quadrilateral (a : ℝ) : Prop :=
  let s := 6 * (1 - Real.sqrt 2)
  let side_length := s * Real.sqrt 2
  let area := side_length ^ 2
  area = 72 * (3 - 2 * Real.sqrt 2)

theorem area_of_Q1Q3Q5Q7 : area_of_quadrilateral regular_octagon_apothem :=
  sorry

end area_of_Q1Q3Q5Q7_l615_615670


namespace length_of_AC_l615_615414

/-- 
Given AB = 15 cm, DC = 24 cm, and AD = 7 cm, 
prove that the length of AC to the nearest tenth of a centimeter is 30.3 cm.
-/
theorem length_of_AC {A B C D : Point} 
  (hAB : dist A B = 15) (hDC : dist D C = 24) (hAD : dist A D = 7) : 
  dist A C ≈ 30.3 :=
sorry

end length_of_AC_l615_615414


namespace exists_subset_S_l615_615896

def T := {p : ℤ × ℤ × ℤ // true}

def neighbors (p q : ℤ × ℤ × ℤ) : Prop := 
  |p.1 - q.1| + |p.2 - q.2| + |p.3 - q.3| = 1

def S (p : ℤ × ℤ × ℤ) : Prop := 
  (p.1 + 2 * p.2 + 3 * p.3) % 7 = 0

theorem exists_subset_S : 
  ∃ S : ℤ × ℤ × ℤ → Prop, 
  (∀ p : ℤ × ℤ × ℤ, 
    (∃ q : ℤ × ℤ × ℤ, neighbors p q ∧ S q) ∧
    (∀ q1 q2 : ℤ × ℤ × ℤ, neighbors p q1 ∧ neighbors p q2 ∧ S q1 ∧ S q2 → q1 = q2)) :=
begin
  use S,
  sorry
end

end exists_subset_S_l615_615896


namespace union_complement_set_l615_615464

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615464


namespace probability_shaded_l615_615687

noncomputable def isosceles_triangle := sorry -- Definition of an isosceles triangle with base angles of 45 degrees.

def total_regions : ℕ := 6
def shaded_regions : ℕ := 3

theorem probability_shaded : (shaded_regions : ℚ) / total_regions = 1 / 2 := by
  -- We assert the given conditions:
  have cond1 : total_regions = 6 := by sorry
  have cond2 : shaded_regions = 3 := by sorry
  sorry

end probability_shaded_l615_615687


namespace sequence_bound_for_difference_l615_615874

theorem sequence_bound_for_difference (a : ℕ → ℕ)
  (H_injective : function.injective a)
  (H_cover : ∀ n, ∃ m, a m = n)
  (H_inequality : ∀ n m, n ≠ m → (1/1998 : ℝ) < abs (a n - a m) / abs (n - m) ∧ abs (a n - a m) / abs (n - m) < 1998) :
  ∀ n, abs (a n - n) < 2000000 := 
by 
  sorry

end sequence_bound_for_difference_l615_615874


namespace sqrt_fraction_sum_as_common_fraction_l615_615215

theorem sqrt_fraction_sum_as_common_fraction (a b c d : ℚ) (ha : a = 25) (hb : b = 36) (hc : c = 16) (hd : d = 9) :
  Real.sqrt ((a / b) + (c / d)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_sum_as_common_fraction_l615_615215


namespace positive_integer_solutions_of_inequality_l615_615601

theorem positive_integer_solutions_of_inequality :
  {x : ℕ | 2 * (x - 1) < 7 - x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_l615_615601


namespace movie_ticket_cost_l615_615127

variable (x : ℝ)
variable (h1 : x * 2 + 1.59 + 13.95 = 36.78)

theorem movie_ticket_cost : x = 10.62 :=
by
  sorry

end movie_ticket_cost_l615_615127


namespace det_E_eq_25_l615_615901

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end det_E_eq_25_l615_615901


namespace compare_abc_l615_615054

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615054


namespace max_fraction_value_l615_615102

variable {x y a : ℝ}

theorem max_fraction_value (hx : x > 0) (hy : y > 0) (ha : a > 0) : 
  ∃ x y a, (x > 0) ∧ (y > 0) ∧ (a > 0) ∧ (x = y = a) → 
  (∀ x y a, (x > 0) ∧ (y > 0) ∧ (a > 0) → 
  ((x + y + a)^2 / (x^2 + y^2 + a^2) ≤ 3) ∧ 
  ((x = y = a) → ((x + y + a)^2 / (x^2 + y^2 + a^2) = 3))) :=
by
  sorry

end max_fraction_value_l615_615102


namespace polynomial_distinct_mod_p3_l615_615893

noncomputable def is_prime (p : ℕ) : Prop := nat.prime p

noncomputable def polynomial_with_integer_coeffs (h : ℤ[X]) : Prop := true

def distinct_mod (h : ℤ[X]) (n : ℕ) (m : ℕ) (a b : ℕ) : Prop :=
  a ≠ b ∧ 0 ≤ a ∧ a < n → 0 ≤ b ∧ b < n → (h.eval a) % m ≠ (h.eval b) % m

theorem polynomial_distinct_mod_p3 (p : ℕ) (h : ℤ[X])
  (hp : is_prime p)
  (h_poly : polynomial_with_integer_coeffs h)
  (H : ∀ i j, i ≠ j ∧ 0 ≤ i ∧ i < p^2 → 0 ≤ j ∧ j < p^2 → (h.eval i) % p^2 ≠ (h.eval j) % p^2) :
  ∀ k l, k ≠ l ∧ 0 ≤ k ∧ k < p^3 → 0 ≤ l ∧ l < p^3 → (h.eval k) % p^3 ≠ (h.eval l) % p^3 :=
sorry

end polynomial_distinct_mod_p3_l615_615893


namespace marcus_scored_50_percent_l615_615535

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l615_615535


namespace union_complement_eq_l615_615470

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615470


namespace first_negative_term_at_14_l615_615410

-- Define the n-th term of the arithmetic sequence
def a_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Given values
def a₁ := 51
def d := -4

-- Proof statement
theorem first_negative_term_at_14 : ∃ n : ℕ, a_n a₁ d n < 0 ∧ ∀ m < n, a_n a₁ d m ≥ 0 :=
  by sorry

end first_negative_term_at_14_l615_615410


namespace union_complement_eq_l615_615475

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615475


namespace compare_abc_l615_615007

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615007


namespace wall_width_l615_615242

noncomputable def brick_dim : ℝ × ℝ × ℝ := (0.21, 0.10, 0.08) -- dimensions of one brick in meters
noncomputable def wall_dim : ℝ × ℝ := (9, 18.5) -- dimensions of the wall in length and height
noncomputable def num_bricks : ℝ := 4955.357142857142 -- total number of bricks

theorem wall_width :
  let (bl, bw, bh) := brick_dim in
  let (wl, wh) := wall_dim in
  let brick_volume := bl * bw * bh in
  let total_volume_bricks := brick_volume * num_bricks in
  let wall_volume w := wl * w * wh in
  ∃ w : ℝ, wall_volume w = total_volume_bricks ∧ w = 0.05 :=
by
  simplify_bricks sorry

end wall_width_l615_615242


namespace real_part_of_z_given_condition_l615_615359

open Complex

noncomputable def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_of_z_given_condition :
  ∀ (z : ℂ), (i * (z + 1) = -3 + 2 * i) → real_part_of_z z = 1 :=
by
  intro z h
  sorry

end real_part_of_z_given_condition_l615_615359


namespace recipe_flour_cups_l615_615929

theorem recipe_flour_cups (F : ℕ) : 
  (exists (sugar : ℕ) (flourAdded : ℕ) (sugarExtra : ℕ), sugar = 11 ∧ flourAdded = 4 ∧ sugarExtra = 6 ∧ ((F - flourAdded) + sugarExtra = sugar)) →
  F = 9 :=
sorry

end recipe_flour_cups_l615_615929


namespace average_apples_per_guest_l615_615884

noncomputable def Redicious_to_apple_pieces : ℝ := 1
noncomputable def GrannySmith_to_apple_pieces : ℝ := 1.25
noncomputable def Redicious_conversion_factor : ℝ := 0.7
noncomputable def GrannySmith_conversion_factor : ℝ := 0.8

theorem average_apples_per_guest 
    (number_of_guests : ℝ)
    (number_of_pies : ℝ)
    (servings_per_pie : ℝ)
    (apple_pieces_per_serving : ℝ)
    (proportion_redicious : ℝ)
    (proportion_granny_smith : ℝ)
    (total_cups_redicious : ℝ)
    (total_cups_granny_smith : ℝ)
    (rounded_total_redicious_apples : ℝ)
    (rounded_total_granny_smith_apples : ℝ)
    (total_apples : ℝ)
    (average_apples_per_guest : ℝ) 
    : average_apples_per_guest = 2.25 :=
by
  have h1 : number_of_pies * servings_per_pie = 24 := by sorry
  have h2 : h1 * apple_pieces_per_serving = 36 := by sorry
  have h3 : (proportion_redicious / (proportion_redicious + proportion_granny_smith)) * 36 = 24 := by sorry
  have h4 : (proportion_granny_smith / (proportion_redicious + proportion_granny_smith)) * 36 = 12 := by sorry
  have h5 : 24 * Redicious_conversion_factor = 16.8 := by sorry
  have h6 : 12 * GrannySmith_conversion_factor = 9.6 := by sorry
  let rounded_total_redicious_apples := 17
  let rounded_total_granny_smith_apples := 10
  have total_apples := rounded_total_redicious_apples + rounded_total_granny_smith_apples := by sorry
  have total_apples = 27 := by sorry
  have average_apples_per_guest := total_apples / number_of_guests := by sorry
  exact (average_apples_per_guest : ℝ) ≈ 2.25

end average_apples_per_guest_l615_615884


namespace number_of_valid_pairs_l615_615598

theorem number_of_valid_pairs :
  (∃ (count : ℕ), count = 2466 ∧
    (∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2887 → 7^n < 3^m → 3^m < 3^(m + 3) → 3^(m + 3) < 7^(n + 1) → (m, n) ∈ set.range pair)) :=
sorry

end number_of_valid_pairs_l615_615598


namespace inequality_a_c_b_l615_615073

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615073


namespace inequality_a_c_b_l615_615076

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615076


namespace no_integer_n_for_fractions_l615_615141

theorem no_integer_n_for_fractions (n : ℤ) : ¬ (∃ n : ℤ, (n - 6) % 15 = 0 ∧ (n - 5) % 24 = 0) :=
by sorry

end no_integer_n_for_fractions_l615_615141


namespace number_of_diagonals_l615_615854

-- Define the regular pentagonal prism and its properties
def regular_pentagonal_prism : Type := sorry

-- Define what constitutes a diagonal in this context
def is_diagonal (p : regular_pentagonal_prism) (v1 v2 : Nat) : Prop :=
  sorry -- We need to detail what counts as a diagonal based on the conditions

-- Hypothesis on the structure specifying that there are 5 vertices on the top and 5 on the bottom
axiom vertices_on_top_and_bottom (p : regular_pentagonal_prism) : sorry -- We need the precise formalization

-- The main theorem
theorem number_of_diagonals (p : regular_pentagonal_prism) : ∃ n, n = 10 :=
  sorry

end number_of_diagonals_l615_615854


namespace union_with_complement_l615_615519

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615519


namespace syllogism_arrangement_l615_615631

theorem syllogism_arrangement : 
  (∀ n : ℕ, Odd n → ¬ (n % 2 = 0)) → 
  Odd 2013 → 
  (¬ (2013 % 2 = 0)) :=
by
  intros h1 h2
  exact h1 2013 h2

end syllogism_arrangement_l615_615631


namespace cyclist_speed_solution_l615_615197

noncomputable def cyclist_speed_problem : Prop :=
  ∃ (c d : ℝ), 
    -- Cyclists start simultaneously from Newport to Kingston, 80 miles apart
    (d = c + 6) ∧                    -- Cyclist C is 6 mph slower than cyclist D
    (80 - 20 = 60) ∧                 -- Distance C travels until meeting point
    (80 + 20 = 100) ∧                -- Distance D travels until meeting point
    (60 / c = 100 / d) ∧             -- Travel times are equal
    (c = 9)                          -- Speed of cyclist C is 9 mph, the solution to the equation

theorem cyclist_speed_solution : cyclist_speed_problem :=
by {
  use 9, 15,                         -- use c = 9, d = 9 + 6 = 15
  repeat { split },
  -- Following are the given distances and conditions
  exact (by norm_num : 80 - 20 = 60),
  exact (by norm_num : 80 + 20 = 100),
  -- Verifying the travel times
  have hc : (60 : ℝ) / (9 : ℝ) = (100 : ℝ) / (15 : ℝ),
  { exact (by norm_num : 60 / 9 = 100 / 15) },
  exact hc,
  -- Given the solution
  exact (by norm_num : (9 : ℝ) = 9),
}

-- Adding the sorry placeholder since the proof steps are not required based on the user's instructions.

end cyclist_speed_solution_l615_615197


namespace min_value_3x2_plus_1_div_x2_l615_615741

theorem min_value_3x2_plus_1_div_x2 (x : ℝ) (hx : x > 0) : 
  ∃ C : ℝ, (∀ x > 0, 3 * x^2 + 1 / x^2 ≥ C) ∧ (∃ x : ℝ, x > 0 ∧ 3 * x^2 + 1 / x^2 = C) ∧ C = 2 * sqrt 3 := 
by
  sorry

end min_value_3x2_plus_1_div_x2_l615_615741


namespace inequality_a_c_b_l615_615066

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615066


namespace bucket_weight_one_third_l615_615627

theorem bucket_weight_one_third 
    (x y c b : ℝ) 
    (h1 : x + 3/4 * y = c)
    (h2 : x + 1/2 * y = b) :
    x + 1/3 * y = 5/3 * b - 2/3 * c :=
by
  sorry

end bucket_weight_one_third_l615_615627


namespace coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6_l615_615210

theorem coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6 : 
  (∑ k in finset.range 7, (nat.choose 6 k) * (x ^ k) * (y ^ (6 - k)))[3] = 20 := 
by sorry

end coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6_l615_615210


namespace probability_at_least_one_woman_l615_615852

theorem probability_at_least_one_woman (men women total selected : ℕ) (H_men : men = 10) (H_women : women = 5)
    (H_total : total = men + women) (H_selected : selected = 4) :
    (selected ≤ total) →
    ∃ p : ℚ, p = 77 / 91 :=
by
    have H_total : total = 15 := by rw [H_men, H_women]
    have H_selected : (selected = 4) := rfl
    sorry

end probability_at_least_one_woman_l615_615852


namespace minimum_roads_l615_615284

open Finset

variables (V : Type) [Fintype V] [DecidableEq V] (G : SimpleGraph V)
variables [Fintype G.V]

def is_9_cycle (H : SimpleGraph V) : Prop :=
  ∃ (C : Finset V), C.card = 9 ∧ SimpleGraph.induced_subgraph G C = H ∧ is_cycle H

def graph_condition := ∀ v ∈ G.verts, ∃ (H : SimpleGraph (G.verts \ {v})), is_9_cycle H

theorem minimum_roads (G : SimpleGraph V) [graph_condition G] : G.edge_count ≥ 15 :=
sorry

end minimum_roads_l615_615284


namespace compare_abc_l615_615023

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615023


namespace num_valid_ns_l615_615100

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m * m ≤ n → (m ∣ n → m = n)

def is_valid_n (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 20 ∧ is_prime n ∧ ¬ (∃ k : ℕ, 180 * (n - 2) = k * n)

theorem num_valid_ns : (finset.filter is_valid_n (finset.range 21)).card = 4 :=
by
  sorry

end num_valid_ns_l615_615100


namespace right_triangle_AB_l615_615838

theorem right_triangle_AB (A B C : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (hypotenuse_AC : Real) (BC_ratio : Real) (AB_ratio : Real)
  (hypotenuse_AC_eq_39 : hypotenuse_AC = 39)
  (tan_B : BC_ratio / AB_ratio = 5 / 12) :
  let k := hypotenuse_AC / 13 in
  AB_ratio * k = 36 := 
by
  sorry

end right_triangle_AB_l615_615838


namespace evaluate_sum_l615_615726

theorem evaluate_sum : 
  (\sum k in Finset.range 2006, 2007 / ((k + 1) * (k + 2))) = 2006 :=
by
  sorry

end evaluate_sum_l615_615726


namespace number_of_solutions_l615_615386

def floor (x : ℝ) : ℤ := Int.floor x

theorem number_of_solutions :
  (finset.univ.filter (λ (x : ℤ), floor ((x : ℝ) / 10) = floor ((x : ℝ) / 11) + 1)).card = 110 :=
sorry

end number_of_solutions_l615_615386


namespace compare_abc_l615_615064

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615064


namespace min_value_frac_expr_l615_615910

theorem min_value_frac_expr (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : a < 1) (h₃ : 0 ≤ b) (h₄ : b < 1) (h₅ : 0 ≤ c) (h₆ : c < 1) :
  (1 / ((2 - a) * (2 - b) * (2 - c)) + 1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1 / 8 :=
sorry

end min_value_frac_expr_l615_615910


namespace valid_reasonings_l615_615630

-- Define the conditions as hypotheses
def analogical_reasoning (R1 : Prop) : Prop := R1
def inductive_reasoning (R2 R4 : Prop) : Prop := R2 ∧ R4
def invalid_generalization (R3 : Prop) : Prop := ¬R3

-- Given the conditions, prove that the valid reasonings are (1), (2), and (4)
theorem valid_reasonings
  (R1 : Prop) (R2 : Prop) (R3 : Prop) (R4 : Prop)
  (h1 : analogical_reasoning R1) 
  (h2 : inductive_reasoning R2 R4) 
  (h3 : invalid_generalization R3) : 
  R1 ∧ R2 ∧ R4 :=
by 
  sorry

end valid_reasonings_l615_615630


namespace find_A_angle_and_area_l615_615877

-- Given conditions
variables {A B C : ℝ} {a b c : ℝ}
hypothesis1 : a * cos C + c * cos A = 2 * b * cos A
hypothesis2 : a = 2
hypothesis3 : b + c = sqrt 10

-- Conclusion
theorem find_A_angle_and_area : A = π / 3 ∧ (1 / 2) * b * c * (sin (π / 3)) = 7 * sqrt 3 / 6 :=
by
  sorry

end find_A_angle_and_area_l615_615877


namespace equidistant_line_l615_615975

theorem equidistant_line (x y : ℝ) : (2 * x - 7 * y + 1 = 0) ↔
  (abs_eval (2 * x - 7 * y + 8) = abs_eval (2 * x - 7 * y - 6)) :=
by
  sorry

end equidistant_line_l615_615975


namespace John_spent_15_dollars_on_soap_l615_615886

theorem John_spent_15_dollars_on_soap (number_of_bars : ℕ) (weight_per_bar : ℝ) (cost_per_pound : ℝ)
  (h1 : number_of_bars = 20) (h2 : weight_per_bar = 1.5) (h3 : cost_per_pound = 0.5) :
  (number_of_bars * weight_per_bar * cost_per_pound) = 15 :=
by
  sorry

end John_spent_15_dollars_on_soap_l615_615886


namespace number_of_children_l615_615546

-- Define the number of adults and their ticket price
def num_adults := 9
def adult_ticket_price := 11

-- Define the children's ticket price and the total cost difference
def child_ticket_price := 7
def cost_difference := 50

-- Define the total cost for adult tickets
def total_adult_cost := num_adults * adult_ticket_price

-- Given the conditions, prove that the number of children is 7
theorem number_of_children : ∃ c : ℕ, total_adult_cost = c * child_ticket_price + cost_difference ∧ c = 7 :=
by
  sorry

end number_of_children_l615_615546


namespace square_side_length_in_right_triangle_l615_615561

theorem square_side_length_in_right_triangle (PQ PR : ℝ) (hPQ : PQ = 9) (hPR : PR = 12) :
  ∃ (s : ℝ), s = 3 ∧ ∃ (hQR : QR = real.sqrt (PQ ^ 2 + PR ^ 2)),
  ∃ (x y : ℝ), x = PR - s ∧ y = PQ - s ∧ PQ - s + PR - s = QR - s - s :=
by
  -- Definition of QR using Pythagorean theorem
  let QR := real.sqrt (PQ^2 + PR^2)
  have hypQR : QR = 15 := by
    rw [hPQ, hPR]
    norm_num
    exact rfl

  -- Sum of segments along hypotenuse
  use 3
  use hypQR
  exists 12 - 3, 9 - 3
  split
  · norm_num
    change (9 - 3) + (12 - 3) = 15
    norm_num
    sorry

  sorry

end square_side_length_in_right_triangle_l615_615561


namespace triangle_area_ratio_l615_615878

-- Definitions and conditions from the given problem
variables (A B C X : Point)
variable [euclidean_space Point]
variables (AB BC AC : ℝ)
variables (is_right_angle_ABC : ∠ABC = π / 2)
variables (CX_bisects_BCA : bisector C X ∠BCA)
variables (AB_eq : AB = 20)
variables (BC_eq : BC = 15)
variables (AC_eq : AC = 25)

-- The proof problem in Lean 4 statement
theorem triangle_area_ratio :
  let area_BCX := triangle_area B C X in
  let area_ABX := triangle_area A B X in
  area_BCX / area_ABX = 3 / 4 :=
begin
  sorry
end

end triangle_area_ratio_l615_615878


namespace maries_trip_distance_l615_615540

theorem maries_trip_distance (x : ℚ)
  (h1 : x = x / 4 + 15 + x / 6) :
  x = 180 / 7 :=
by
  sorry

end maries_trip_distance_l615_615540


namespace compare_a_b_c_l615_615041

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615041


namespace value_of_a_plus_b_l615_615442

open Real

def S (x y z : ℝ) : Prop :=
  log x y 2 + y = z ∧ log(x^2 y^2 2 = 2 * z + 1

theorem value_of_a_plus_b :
  ∀ (x y z : ℝ), S x y z → x^3 y^3 = (5/2) * 2^(3 * z) + 0 * 2^z := 
by
  intros x y z h
  sorry

end value_of_a_plus_b_l615_615442


namespace jane_wins_prob_correct_l615_615269

-- Definitions based on conditions
def sectors : Finset ℕ := {1, 2, 3, 4, 5, 6}
def possible_outcomes := sectors.product sectors
def losing_conditions : Finset (ℕ × ℕ) := {(1, 5), (1, 6), (2, 6), (5, 1), (6, 1), (6, 2)}

-- noncomputable and proof skeleton
noncomputable def jane_wins_probability : ℚ :=
  1 - (losing_conditions.card / possible_outcomes.card : ℚ)

theorem jane_wins_prob_correct : jane_wins_probability = 5 / 6 :=
by
  -- proof steps will be filled here
  sorry

end jane_wins_prob_correct_l615_615269


namespace alternating_sum_total_7_l615_615331

/-- Sum of all alternating sums for n = 7 --/
theorem alternating_sum_total_7 :
  let S := {1, 2, 3, 4, 5, 6, 7}
  let subsets := (S.powerset \\ {∅})
  let alternating_sum := λ (s : Finset ℕ), s.sort (· ≥ ·).foldl (λ acc x => if acc.next_odd then acc - x else acc + x) 0
  subsets.sum alternating_sum = 448 := by sorry

end alternating_sum_total_7_l615_615331


namespace parabola_intersection_points_l615_615702

def parabola_focus := (0, 0)

def directrices (a : ℤ) (b : ℤ) : Prop :=
  a ∈ {-3, -2, -1, 0, 1, 2, 3} ∧ b ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4}

theorem parabola_intersection_points :
  (∃ (p : finset (ℤ × ℤ)), p.card = 50 ∧ 
  (∀ (f : (ℤ × ℤ)), f ∈ p → directrices f.1 f.2) ∧
  (∀ (x1 x2 x3 : ℤ × ℤ), 
    (x1 ∈ p ∧ x2 ∈ p ∧ x3 ∈ p ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
    ¬(parabola_focus_intersection x1 x2 x3)) ∧
  (number_of_intersection_points p = 2170)) :=
sorry

-- Auxiliary function to model the concept of intersections
noncomputable def parabola_focus_intersection (x1 x2 x3 : ℤ × ℤ) : Prop :=
  -- Placeholder function to handle intersection logic
  sorry

noncomputable def number_of_intersection_points (p : finset (ℤ × ℤ)) : ℕ :=
  -- Placeholder function to handle number computation logic
  sorry

end parabola_intersection_points_l615_615702


namespace multiplier_is_3_l615_615971

theorem multiplier_is_3 (x : ℝ) (num : ℝ) (difference : ℝ) (h1 : num = 15.0) (h2 : difference = 40) (h3 : x * num - 5 = difference) : x = 3 := 
by 
  sorry

end multiplier_is_3_l615_615971


namespace trigonometric_function_range_l615_615834

theorem trigonometric_function_range :
  let f (x : ℝ) := 2 * sin (2 * x - π / 2)
  ∀ x ∈ Set.Icc (π / 6) (2 * π / 3), f x ∈ Set.Icc (-1 : ℝ) (2 : ℝ) := by
  let f (x : ℝ) := 2 * sin (2 * x - π / 2)
  sorry

end trigonometric_function_range_l615_615834


namespace Sandy_pumpkins_l615_615948

-- Definitions from the conditions
def Mike_pumpkins : ℕ := 23
def Total_pumpkins : ℕ := 74

-- Theorem to prove the number of pumpkins Sandy grew
theorem Sandy_pumpkins : ∃ (n : ℕ), n + Mike_pumpkins = Total_pumpkins :=
by
  existsi 51
  sorry

end Sandy_pumpkins_l615_615948


namespace sum_first_1990_even_l615_615426

theorem sum_first_1990_even : (finset.range 1990).sum + 1 % 2 = 0 := 
sorry

end sum_first_1990_even_l615_615426


namespace length_of_AC_l615_615416

theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) : AC = 30.1 :=
sorry

end length_of_AC_l615_615416


namespace sequence_conjecture_l615_615344

noncomputable def a_sequence (n : ℕ) : ℚ :=
if n = 1 then -2 / 3 else 
  let s_n := sorry, s_n_minus_1 := sorry in 
  s_n + s_n_minus_1

noncomputable def s_sequence (n : ℕ) : ℚ :=
if n = 1 then -2 / 3 else 
  let a_n := a_sequence n in 
  sorry

theorem sequence_conjecture (n : ℕ) (h : n > 0) :
  s_sequence n = -((n + 1) / (n + 2)) :=
by sorry

end sequence_conjecture_l615_615344


namespace inequality_a_c_b_l615_615068

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615068


namespace classroom_gpa_l615_615980

theorem classroom_gpa (n : ℕ) (x : ℝ)
  (h1 : n > 0)
  (h2 : (1/3 : ℝ) * n * 45 + (2/3 : ℝ) * n * x = n * 55) : x = 60 :=
by
  sorry

end classroom_gpa_l615_615980


namespace subset_S_T_not_subset_T_S_S_proper_subset_T_l615_615105

def S (x y : ℝ) : Prop := ∃ n : ℤ, x^2 - y^2 = 2 * n + 1
def T (x y : ℝ) : Prop := sin (2 * π * x^2) - sin (2 * π * y^2) = cos (2 * π * x^2) - cos (2 * π * y^2)

theorem subset_S_T : ∀ x y : ℝ, S x y → T x y :=
sorry

theorem not_subset_T_S : ∃ x y : ℝ, T x y ∧ ¬S x y :=
sorry

theorem S_proper_subset_T : ∀ x y : ℝ, S x y → T x y ∧ (∃ x y : ℝ, T x y ∧ ¬S x y) :=
sorry

end subset_S_T_not_subset_T_S_S_proper_subset_T_l615_615105


namespace determine_z_l615_615968

variable (z : ℂ)

-- Define the given condition
def condition : Prop := conj z * (1 + 2 * Complex.I) = 4 + 3 * Complex.I

-- Define the goal to prove 
def goal : Prop := z = 2 + Complex.I

-- State the theorem which ties the condition to the goal
theorem determine_z (hz : condition z) : goal z :=
by
  sorry

end determine_z_l615_615968


namespace octahedron_probability_l615_615599

-- Definitions for the conditions
def isValidConfiguration (faces: List ℕ) : Prop := 
  ∀ i ∈ List.range 8, 
    let cur := faces.getD i 0
    let next := faces.getD ((i + 1) % 8) 0
    let prev := faces.getD ((i + 7) % 8) 0
    cur ≠ 0 ∧ next ≠ 0 ∧ prev ≠ 0 ∧ 
    (abs (cur - next) ≠ 1 ∨ (cur + next = 9 ∨ cur = 1 ∨ next = 1)) ∧
    abs (cur - prev) ≠ 1

def countValidConfigurations : ℕ :=
  List.permutations [1, 2, 3, 4, 5, 6, 7, 8] |>.filter isValidConfiguration |>.length

theorem octahedron_probability : ∃ m n : ℕ, 
  (Nat.gcd m n = 1) ∧ (countValidConfigurations = m * 84) → m + n = 85 :=
sorry

end octahedron_probability_l615_615599


namespace largest_of_consecutive_numbers_l615_615962

theorem largest_of_consecutive_numbers (avg : ℕ) (n : ℕ) (h1 : n = 7) (h2 : avg = 20) :
  let sum := n * avg in
  let middle := sum / n in
  let largest := middle + 3 in
  largest = 23 :=
by
  -- Introduce locals to use 
  let sum := n * avg
  let middle := sum / n
  let largest := middle + 3
  -- Add the proof placeholder
  sorry

end largest_of_consecutive_numbers_l615_615962


namespace reading_times_l615_615932

-- Define the total material in the book
def B : ℝ := 1 -- Using 1 as a unit material for simplicity

-- Define my reading time (in minutes)
def my_reading_time : ℝ := 180

-- Arthur's reading speed is three times my reading speed
def arthur_speed : ℝ := 3 * (B / my_reading_time)

-- Ben's reading speed is four times my reading speed
def ben_speed : ℝ := 4 * (B / my_reading_time)

-- Define Arthur's reading time
def arthur_reading_time : ℝ := B / arthur_speed

-- Define Ben's reading time
def ben_reading_time : ℝ := B / ben_speed

-- Main theorem
theorem reading_times :
  arthur_reading_time = 60 ∧ ben_reading_time = 45 :=
by
  sorry

end reading_times_l615_615932


namespace distance_between_parallel_lines_l615_615191

-- Definitions of the given conditions
variable {r d : ℝ}
variable (OC OD OE OF : ℝ)
variable {CD EF DE : ℝ}
variable {d : ℝ} [decidable_eq ℝ]

-- Assumptions based on the problem statement
noncomputable def conditions : Prop :=
  CD = 42 ∧ EF = 42 ∧ DE = 36

-- The main statement proving the distance d between two adjacent parallel lines
theorem distance_between_parallel_lines (h : conditions) :
  d = 7.65 :=
sorry

end distance_between_parallel_lines_l615_615191


namespace compare_a_b_c_l615_615037

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615037


namespace pictures_in_each_album_l615_615139

/-
Problem Statement:
Robin uploaded 35 pictures from her phone and 5 from her camera, which makes 35 + 5 = 40 pictures in total.
She sorted these 40 pictures into 5 different albums with the same amount of pictures in each album.
Prove that there are 8 pictures in each album.
-/

theorem pictures_in_each_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (total_albums : ℕ)
  (total_pics : phone_pics + camera_pics = 40)
  (same_amount : ∀ x, x < total_albums → (phone_pics + camera_pics) / total_albums = 8) : 
  8 = (phone_pics + camera_pics) / total_albums :=
by
  -- given conditions
  have phone_pics := 35
  have camera_pics := 5
  have total_albums := 5
  -- show the result
  show 8 = (phone_pics + camera_pics) / total_albums
  sorry

end pictures_in_each_album_l615_615139


namespace check_amount_l615_615541

theorem check_amount:
  let tip_percent := 0.20 in
  let friend_contribution := 10 in
  let mark_contribution := 30 in
    0.20 * (C : ℝ) = 40 → C = 200 :=
by {
  intro h,
  have hC : C = 40 / 0.20 := sorry,
  exact hC,
}

end check_amount_l615_615541


namespace hunting_trips_per_month_l615_615883

variable (quarter : ℕ) (deer_per_hunt : ℕ) (weight_per_deer : ℕ)
variable (kept_weight_per_year : ℕ) (kept_weight : ℕ)

axiom (hunting_season : quarter = 3)
axiom (deer_per_hunting_trip : deer_per_hunt = 2)
axiom (weight_of_each_deer : weight_per_deer = 600)
axiom (half_weight_kept : 2 * kept_weight_per_year = kept_weight)
axiom (jack_keeps : kept_weight_per_year = 10800)

theorem hunting_trips_per_month :
  let total_weight_per_year := 2 * kept_weight_per_year
  let weight_per_trip := deer_per_hunt * weight_per_deer
  let total_trips_per_year := total_weight_per_year / weight_per_trip
  let trips_per_month := total_trips_per_year / quarter
  trips_per_month = 6 := sorry

end hunting_trips_per_month_l615_615883


namespace standard_equation_of_hyperbola_l615_615605

noncomputable def hyperbola_equation (a c b : ℝ) (h1 : 2 * a = 2) (h2 : c / a = Real.sqrt 2) : Prop :=
  let eq1 := a = 1
  let eq2 := c = Real.sqrt 2
  let eq3 := b^2 = c^2 - a^2
  (eq3) ∧ (b = 1) ∧ ((x^2 - y^2 = 1) ∨ (y^2 - x^2 = 1))

theorem standard_equation_of_hyperbola : ∃ (a c b : ℝ), hyperbola_equation a c b (2 * a = 2) (c / a = Real.sqrt 2) :=
by
  have h1 : 2 * 1 = 2 := by norm_num
  have h2 : Real.sqrt 2 / 1 = Real.sqrt 2 := by norm_num
  exists (1, Real.sqrt 2, 1)
  split
  { exact h1 }
  split
  { exact h2 }
  split
  { calc (1:ℝ) ^ 2 = 1 : by norm_num
         ... = (Real.sqrt 2 ^ 2 - 1 ^ 2) : by norm_num }
  { exact Or.inl (rfl) }

end standard_equation_of_hyperbola_l615_615605


namespace real_part_of_complex_i_mul_1_plus_i_l615_615990

theorem real_part_of_complex_i_mul_1_plus_i : 
  complex.re (complex.I * (1 + complex.I)) = -1 :=
by
  sorry

end real_part_of_complex_i_mul_1_plus_i_l615_615990


namespace count_teams_of_6_l615_615614

theorem count_teams_of_6 
  (students : Fin 12 → Type)
  (played_together_once : ∀ (s : Finset (Fin 12)) (h : s.card = 5), ∃! t : Finset (Fin 12), t.card = 6 ∧ s ⊆ t) :
  (∃ team_count : ℕ, team_count = 132) :=
by
  -- Proof omitted
  sorry

end count_teams_of_6_l615_615614


namespace union_complement_eq_target_l615_615486

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615486


namespace count_valid_arrangements_l615_615282

def people : Type := { A, B, C, D, E }

def valid_arrangement (arr : list people) : Prop :=
  ∀ (i : ℕ), (i < arr.length - 1) → 
  (arr.nth i ≠ some A ∨ arr.nth (i+1) ≠ some C) ∧
  (arr.nth i ≠ some B ∨ arr.nth (i+1) ≠ some C) ∧
  (arr.nth i ≠ some C ∨ arr.nth (i+1) ≠ some A) ∧
  (arr.nth i ≠ some C ∨ arr.nth (i+1) ≠ some B)
 
theorem count_valid_arrangements : 
  ∃ (arrangements : finset (list people)), arrangements.card = 36 ∧ 
  ∀ (arr : list people), arr ∈ arrangements ↔ valid_arrangement arr :=
begin
  sorry
end

end count_valid_arrangements_l615_615282


namespace product_equals_A_29_minus_x_10_l615_615101

-- Let x be an element of the positive natural numbers.
def x : ℕ := sorry

-- Assume the condition that x is less than 10.
lemma x_lt_10 : x < 10 := sorry

-- Define the product from (20-x) to (29-x).
def product_20_to_29_minus_x : ℕ :=
  (20 - x) * (21 - x) * (22 - x) * (23 - x) * (24 - x) * (25 - x) * 
  (26 - x) * (27 - x) * (28 - x) * (29 - x)

-- Define the permutation notation A_{29-x}^{10}.
def A_29_minus_x_10 : ℕ := (29 - x).choose(10)

-- The theorem to prove.
theorem product_equals_A_29_minus_x_10 : product_20_to_29_minus_x = A_29_minus_x_10 :=
by
  sorry

end product_equals_A_29_minus_x_10_l615_615101


namespace simplify_expression_l615_615143

variable (x : ℝ)

theorem simplify_expression :
  (3 * x - 6) * (2 * x + 8) - (x + 6) * (3 * x + 1) = 3 * x^2 - 7 * x - 54 :=
by
  sorry

end simplify_expression_l615_615143


namespace solve_equation_l615_615953

theorem solve_equation : ∃ x : ℚ, (x^2 + 4 * x + 7) / (x + 5) = x + 6 ∧ x = - 23 / 7 :=
by
  use -23 / 7
  split
  sorry

end solve_equation_l615_615953


namespace inequality_proof_l615_615911

-- Define the conditions: a and b are sequences of reals, n is a positive integer
variables {n : ℕ} (a b : ℕ → ℝ)

-- The main theorem statement
theorem inequality_proof (hn : 0 < n) : 
  (∑ i in finset.range n, a i * b i) + 
  real.sqrt ((∑ i in finset.range n, (a i)^2) * (∑ i in finset.range n, (b i)^2)) 
  ≥ (2/n) * (∑ i in finset.range n, a i) * (∑ i in finset.range n, b i) := 
sorry

end inequality_proof_l615_615911


namespace probability_S1987_l615_615250

-- Define the recursive probability function
noncomputable def P (k : ℕ) : ℝ :=
if k = 0 then 1 else ∑ i in finset.range k, P (k - i) * (1 / (2:ℝ) ^ i)

-- The main theorem stating the final probability
theorem probability_S1987 : P 1987 = 1 / 2 ^ 1987 := by
  sorry

end probability_S1987_l615_615250


namespace hyperbola_focus_vertex_and_eccentricity_l615_615355

def focus_of_parabola : ℝ × ℝ := (1, 0)

def parabola_equation (x y : ℝ) := y^2 = 4 * x

def hyperbola_equation (x y a b : ℝ) := (x^2) / (a^2) - (y^2) / (b^2) = 1

def eccentricity (c a : ℝ) := c / a

theorem hyperbola_focus_vertex_and_eccentricity :
  ∃ (x y a b : ℝ), a = 1 ∧ focus_of_parabola = (1, 0) ∧ eccentricity (sqrt 5) a = sqrt 5 ∧ 
  hyperbola_equation x y 1 2 := 
sorry

end hyperbola_focus_vertex_and_eccentricity_l615_615355


namespace c_minus_3_eq_neg3_l615_615292

variable (g : ℝ → ℝ)
variable (c : ℝ)

-- defining conditions
axiom invertible_g : Function.Injective g
axiom g_c_eq_3 : g c = 3
axiom g_3_eq_5 : g 3 = 5

-- The goal is to prove that c - 3 = -3
theorem c_minus_3_eq_neg3 : c - 3 = -3 :=
by
  sorry

end c_minus_3_eq_neg3_l615_615292


namespace coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6_l615_615211

theorem coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6 : 
  (∑ k in finset.range 7, (nat.choose 6 k) * (x ^ k) * (y ^ (6 - k)))[3] = 20 := 
by sorry

end coefficient_x3y3_term_in_expansion_of_x_plus_y_pow_6_l615_615211


namespace value_of_a_l615_615761

theorem value_of_a (a b : ℝ) (A B : Set ℝ) 
  (hA : A = {a, a^2}) (hB : B = {1, b}) (hAB : A = B) : a = -1 := 
by 
  sorry

end value_of_a_l615_615761


namespace union_complement_eq_l615_615502

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615502


namespace relation_among_abc_l615_615088

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615088


namespace symmetry_even_reflection_double_reflection_l615_615637

-- Definition for Problem (a)
theorem symmetry_even_reflection (n : ℕ) (h_even : n % 2 = 0)
  (O : ℕ → ℝ × ℝ) (A B : ℝ × ℝ) :
  let AB := (A, B)
  let A_n := (reflect_n_times n O AB).1
  let B_n := (reflect_n_times n O AB).2
  in dist A A_n = dist B B_n :=
sorry

-- Definition for Problem (b)
theorem double_reflection (n : ℕ)
  (O : ℕ → ℝ × ℝ) (A : ℝ × ℝ) :
  let A_2n := reflect_n_times (2 * n) O (A, A)
  in (A_2n.1 = A_2n.2) ∧ (A_2n.1 = A) :=
sorry

end symmetry_even_reflection_double_reflection_l615_615637


namespace y_ne_z_l615_615633

theorem y_ne_z (a b c x : ℝ) (h1: y = a * (cos x) ^ 2 + 2 * b * sin x * cos x + c * (sin x) ^ 2)
              (h2: z = a * (sin x) ^ 2 - 2 * b * sin x * cos x + c * (cos x) ^ 2)
              (h3: tan x = 2 * b / (a - c))
              (ha: a ≠ c)
              (hx: ∀ k : ℤ, x ≠ (2 * k + 1) * (π / 2)) 
              : y ≠ z := 
sorry

end y_ne_z_l615_615633


namespace symmetry_of_F_l615_615833

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
|f x| + f (|x|)

theorem symmetry_of_F (f : ℝ → ℝ) (h : is_odd_function f) :
    ∀ x : ℝ, F f x = F f (-x) :=
by
  sorry

end symmetry_of_F_l615_615833


namespace compare_a_b_c_l615_615014

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615014


namespace shaded_area_semicircles_l615_615949

theorem shaded_area_semicircles :
  let foot_length := 12
      diameter := 3
      radius := diameter / 2
      semicircle_area := (π * (radius * radius)) / 2
      number_of_semicircles := foot_length / diameter
      total_shaded_area := number_of_semicircles * semicircle_area
  in total_shaded_area = (9 / 2) * π :=
by {
  let foot_length := 12
  let diameter := 3
  let radius := diameter / 2
  let semicircle_area := (π * (radius * radius)) / 2
  let number_of_semicircles := foot_length / diameter
  let total_shaded_area := number_of_semicircles * semicircle_area
  have h1 : total_shaded_area = (12 / 3) * (π * (3 / 2 * 3 / 2)) / 2, by sorry,
  have h2 : (12 / 3) * (π * (3 / 2 * 3 / 2)) / 2 = (9 / 2) * π, by sorry,
  exact eq.trans h1 h2
}

end shaded_area_semicircles_l615_615949


namespace chord_length_of_intersection_l615_615258

-- Define the parametric equations of the line
def line_param_x (t : ℝ) : ℝ := 2 + (sqrt 2 / 2) * t
def line_param_y (t : ℝ) : ℝ := -1 + (sqrt 2 / 2) * t

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- The theorem to prove
theorem chord_length_of_intersection : 
  let t1 t2 : ℝ := sorry, sorry in -- Using sorry for roots for now
  let sum_roots := -sqrt 2 in
  let prod_roots := -4 in
  abs (t1 - t2) = 3 * sqrt 2 :=
by
  sorry

end chord_length_of_intersection_l615_615258


namespace length_extension_l615_615935

open Real

-- Conditions
variables (n : ℕ) (a : ℝ) (α : ℝ)
-- Assertions
hypothesis : n ≥ 5
hypothesis : 0 < a
hypothesis : 0 < α ∧ α < π

-- The goal is to prove that the length x from A_{k+1} to B_k is as follows:
theorem length_extension (n : ℕ) (a α : ℝ) (h1 : n ≥ 5) (h2 : 0 < a) (h3 : 0 < α ∧ α < π):
  let x := a * (cos α) / (1 - cos α) in
  ∀ k : ℕ, 0 ≤ k < n → (cos α) ^ n ≠ 1 → 
  x = a * (cos α) / (1 - cos α) := by
  sorry

end length_extension_l615_615935


namespace problem1_problem2_l615_615299

-- Definitions for imaginary unit properties
def i := Complex.I
axiom i_squared : i^2 = -1
axiom i_fourth_power : i^4 = 1

-- Problem 1
theorem problem1 : (4 - i^5) * (6 + 2 * i^7) + (7 + i^{11}) * (4 - 3 * i) = 57 - 39 * i :=
by sorry

-- Problem 2
theorem problem2 : (5 * (4 + i)^2) / (i * (2 + i)) = -47 - 98 * i :=
by sorry

end problem1_problem2_l615_615299


namespace marcus_percentage_of_team_points_l615_615537

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l615_615537


namespace number_of_boxwoods_l615_615545

variables (x : ℕ)
def charge_per_trim := 5
def charge_per_shape := 15
def number_of_shaped_boxwoods := 4
def total_charge := 210
def total_shaping_charge := number_of_shaped_boxwoods * charge_per_shape

theorem number_of_boxwoods (h : charge_per_trim * x + total_shaping_charge = total_charge) : x = 30 :=
by
  sorry

end number_of_boxwoods_l615_615545


namespace maximum_value_f_value_of_a_l615_615804

noncomputable def f (x : Real) : Real :=
  sin x * cos (x - π / 6) + 1 / 2 * cos (2 * x)

theorem maximum_value_f :
  ∃ x : Real, ∀ y : Real, f y ≤ 3 / 4 :=
by
  sorry

variables {A B C a b c : Real}
hypothesis (H1 : sin (2 * A + π / 6) = 1/2)
hypothesis (H2 : b + c = 10)
hypothesis (H3 : 4 * sqrt 3 = 1 / 2 * b * c * sin (π / 3))

theorem value_of_a :
  a = 2 * sqrt 13 :=
by
  sorry

end maximum_value_f_value_of_a_l615_615804


namespace nested_function_limit_l615_615112

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p * x + q

theorem nested_function_limit (p q : ℝ) (h : ∀ x ∈ set.Icc 4 6, |f x p q| ≤ 1/2) :
  let x0 := (9 - real.sqrt 19) / 2 in
  filter.tendsto (λ n, (λ x, f x p q)^n x0) filter.at_top (𝓝 ((9 + real.sqrt 19) / 2)) :=
sorry

end nested_function_limit_l615_615112


namespace bucket_volume_l615_615431

theorem bucket_volume (secs_per_bucket : ℕ) (pool_gallons : ℕ) (mins_to_fill_pool : ℕ) (secs_per_min : ℕ) (gallons_per_bucket : ℕ) :
  secs_per_bucket = 20 →
  pool_gallons = 84 →
  mins_to_fill_pool = 14 →
  secs_per_min = 60 →
  (mins_to_fill_pool * secs_per_min) / secs_per_bucket = pool_gallons / gallons_per_bucket →
  gallons_per_bucket = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end bucket_volume_l615_615431


namespace binomial_identity_l615_615708

theorem binomial_identity (n k : ℕ) : 
  (Nat.choose (2 * n + 1) (2 * k + 1)) = ∑ m in finset.range (2 * n - 2 * k + 1), Nat.choose (k + m) k * Nat.choose (2 * n - k - m) k := 
by sorry

end binomial_identity_l615_615708


namespace triangle_ratio_l615_615876

/-- In triangle ABC, AB = 15, AC = 8. The angle bisector of ∠A intersects BC at point D, 
    and the incenter I of triangle ABC is on the segment AD. The midpoint of AD is M. 
    We need to prove that IP = PD where P is the intersection of AI and BM. -/
theorem triangle_ratio (A B C D I P M : Point) 
  (h₁ : IsTriangle A B C) 
  (h₂ : AB = 15) 
  (h₃ : AC = 8) 
  (h₄ : IsAngleBisector A B C D) 
  (h₅ : IsIncenter A B C I) 
  (h₆ : OnSegment I A D) 
  (h₇ : IsMidpoint M A D) 
  (h₈ : IsIntersection P AI BM) :
  Distance I P = Distance P D := 
sorry

end triangle_ratio_l615_615876


namespace find_angle_A_range_of_perimeter_l615_615860

variables {A B C : ℝ} (a b c : ℝ)

-- Given conditions
def acute_triangle : Prop := 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

def vectors_parallel : Prop := ∃ (k : ℝ), (a, sqrt 3 * b) = (k * cos A, k * sin B)

def given_a : Prop := a = 2

-- Proof problem statements
theorem find_angle_A (h_acute : acute_triangle A B C) (h_parallel : vectors_parallel a b A B) : 
  A = π / 3 :=
by sorry

theorem range_of_perimeter (h_acute : acute_triangle A B C) (h_parallel : vectors_parallel a b A B) 
  (h_a : given_a a) :
  2 * sqrt 3 + 2 < a + b + c ∧ a + b + c ≤ 6 :=
by sorry

end find_angle_A_range_of_perimeter_l615_615860


namespace second_athlete_long_jump_distance_l615_615195

noncomputable def first_long_jump : ℕ := 26
noncomputable def first_triple_jump : ℕ := 30
noncomputable def first_high_jump : ℕ := 7

noncomputable def second_triple_jump : ℕ := 34
noncomputable def second_high_jump : ℕ := 8
noncomputable def winner_avg_jump : ℕ := 22

theorem second_athlete_long_jump_distance :
  ∃ (L : ℕ), (L + second_triple_jump + second_high_jump) / 3 = winner_avg_jump
    ∧ L = 24 :=
by
  use 24
  split
  sorry
  sorry

end second_athlete_long_jump_distance_l615_615195


namespace find_ruv_l615_615715

theorem find_ruv (u v : ℝ) : 
  (∃ u v : ℝ, 
    (3 + 8 * u + 5, 1 - 4 * u + 2) = (4 + -3 * v + 5, 2 + 4 * v + 2)) →
  (u = -1/2 ∧ v = -1) :=
by
  intros H
  sorry

end find_ruv_l615_615715


namespace jason_pokemon_cards_l615_615429

theorem jason_pokemon_cards :
  let initial_cards := 9
  let cards_gave_away := 4
  let packs_bought := 3
  let cards_per_pack := 5
  let total_cards_after_buying := initial_cards - cards_gave_away + packs_bought * cards_per_pack
  let final_cards := total_cards_after_buying - total_cards_after_buying / 2
  final_cards = 10 :=
by
  let initial_cards := 9
  let cards_gave_away := 4
  let packs_bought := 3
  let cards_per_pack := 5
  let total_cards_after_buying := initial_cards - cards_gave_away + packs_bought * cards_per_pack
  let final_cards := total_cards_after_buying - total_cards_after_buying / 2
  show final_cards = 10, from sorry

end jason_pokemon_cards_l615_615429


namespace exist_rectangle_l615_615933

open Real

/-- Definition of the problem -/
theorem exist_rectangle (n p : ℕ) :
    ∃ (rectangles : List (Real × Real × Real × Real)), 
      rectangles.length = (Real.toNat ((Real.sqrt (Real.sqrt 3)) / 3 * n)) ∧ 
      (∀ r ∈ rectangles, ∃ p_rectangles ⊆ rectangles, p_rectangles.length ≥ p ∧ ∀ pr ∈ p_rectangles, intersects r pr) ∧ 
      (∃ r ∈ rectangles, ∀ r' ∈ rectangles, intersects r r') := 
sorry

/-- Define the intersection of rectangles -/
def intersects (r1 r2 : Real × Real × Real × Real) : Prop :=
    let (x1, y1, x1', y1') := r1 in
    let (x2, y2, x2', y2') := r2 in
    ¬ (x1' < x2 ∨ x2' < x1 ∨ y1' < y2 ∨ y2' < y1)

end exist_rectangle_l615_615933


namespace solve_for_x_l615_615574

theorem solve_for_x (x : ℝ) (h : (x^2 - 36) / 3 = (x^2 + 3 * x + 9) / 6) : x = 9 ∨ x = -9 := 
by 
  sorry

end solve_for_x_l615_615574


namespace Anne_carrying_four_cats_weight_l615_615285

theorem Anne_carrying_four_cats_weight : 
  let w1 := 2
  let w2 := 1.5 * w1
  let m1 := 2 * w1
  let m2 := w1 + w2
  w1 + w2 + m1 + m2 = 14 :=
by
  sorry

end Anne_carrying_four_cats_weight_l615_615285


namespace calculate_triply_nested_sigma_8_l615_615443

def divisor_sum (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ d, d > 0 ∧ n % d = 0).sum id

theorem calculate_triply_nested_sigma_8 : divisor_sum (divisor_sum (divisor_sum 8)) = 0 := 
  sorry

end calculate_triply_nested_sigma_8_l615_615443


namespace mr_smith_total_cost_l615_615931

noncomputable def total_cost : ℝ :=
  let adult_price := 30
  let child_price := 15
  let teen_price := 25
  let senior_discount := 0.10
  let college_discount := 0.05
  let senior_price := adult_price * (1 - senior_discount)
  let college_price := adult_price * (1 - college_discount)
  let soda_price := 2
  let iced_tea_price := 3
  let coffee_price := 4
  let juice_price := 1.50
  let wine_price := 6
  let buffet_cost := 2 * adult_price + 2 * senior_price + 3 * child_price + teen_price + 2 * college_price
  let drinks_cost := 3 * soda_price + 2 * iced_tea_price + coffee_price + juice_price + 2 * wine_price
  buffet_cost + drinks_cost

theorem mr_smith_total_cost : total_cost = 270.50 :=
by
  sorry

end mr_smith_total_cost_l615_615931


namespace variance_scaled_translated_sample_l615_615363

-- Assuming necessary definitions for variance calculation
namespace Variance

variable {α : Type*} [field α] [fintype α]

noncomputable def variance (s : finset α) (f : α → ℝ) : ℝ := sorry

variables (s : finset α) (x : α → ℝ)

-- x_1, x_2, ..., x_n sample
variable (x_var : variance s x = 2)

-- Define the scaled and translated sample
def scaled_translated_sample (i : α) : ℝ := 3 * x i + 2

-- The proof goal will be
theorem variance_scaled_translated_sample : variance s (scaled_translated_sample x) = 18 := by
  sorry  -- Proof of the theorem goes here

end Variance

end variance_scaled_translated_sample_l615_615363


namespace distance_between_points_l615_615296

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance 1 2 5 6 = 4 * Real.sqrt 2 := 
sorry

end distance_between_points_l615_615296


namespace area_bounded_arcsin_cos_l615_615734

noncomputable def area_arcsin_cos (a b : ℝ) : ℝ :=
  ∫ x in a .. b, Real.arcsin (Real.cos x)

theorem area_bounded_arcsin_cos :
  area_arcsin_cos 0 (3 * Real.pi) = (3 * Real.pi^2) / 4 :=
by
  sorry

end area_bounded_arcsin_cos_l615_615734


namespace sum_of_z_values_l615_615445

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem sum_of_z_values (z1 z2 : ℝ) (hz1 : f (3 * z1) = 11) (hz2 : f (3 * z2) = 11) :
  z1 + z2 = - (2 / 9) :=
sorry

end sum_of_z_values_l615_615445


namespace least_five_digit_prime_mod_20_l615_615621

theorem least_five_digit_prime_mod_20 :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 20 = 7 ∧ Prime n ∧
  ∀ m : ℕ, 10000 ≤ m ∧ m < n → ¬ (m % 20 = 7 ∧ Prime m) :=
begin
  use 10127,
  split,
  { exact dec_trivial }, -- 10000 ≤ 10127
  split,
  { exact dec_trivial }, -- 10127 < 100000
  split,
  { exact dec_trivial }, -- 10127 % 20 = 7
  split,
  { exact dec_trivial }, -- Prime 10127
  { intros m h1 h2,
    -- m cannot satisfy both conditions for lower values than 10127
    sorry },
end

end least_five_digit_prime_mod_20_l615_615621


namespace union_complement_l615_615491

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615491


namespace sequence_periodic_l615_615405

/-- Given a sequence u defined by an initial value u₀ and a recurrence relation where uₙ₊₁ is half of uₙ if uₙ
is even, and uₙ₊₁ is a + uₙ if uₙ is odd, and a is a fixed odd positive integer, then the sequence is periodic 
beginning from a certain step. -/
theorem sequence_periodic (a : ℕ) (h_odd_a : a % 2 = 1) (u₀ : ℕ) :
  ∃ N T : ℕ, ∀ n ≥ N, u (n + T) = u n :=
sorry

noncomputable def u : ℕ → ℕ
| 0       := u₀
| (n + 1) := if u n % 2 = 0 then u n / 2 else u n + a

end sequence_periodic_l615_615405


namespace distance_from_K_to_AC_l615_615420

theorem distance_from_K_to_AC (A B C K : Point) (ABC_isosceles : is_isosceles A B C) 
  (angle_A : angle BAC = 75) (K_on_bisector : bisects K angle_A) (BK : distance B K = 10) :
  ∃ d, d = 5 /\ distance_from_point_to_line K (line AC) = d :=
sorry

end distance_from_K_to_AC_l615_615420


namespace union_complement_eq_l615_615505

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615505


namespace quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l615_615341

-- Prove the range of k for distinct real roots
theorem quadratic_distinct_real_roots (k: ℝ) (h: k ≠ 0) : 
  (40 * k + 16 > 0) ↔ (k > -2/5) := 
by sorry

-- Prove the solutions for the quadratic equation when k = 1
theorem quadratic_solutions_k_eq_1 (x: ℝ) : 
  (x^2 - 6*x - 5 = 0) ↔ 
  (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14) := 
by sorry

end quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l615_615341


namespace monotonic_increasing_intervals_l615_615803

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 3))

theorem monotonic_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, (k : ℝ) * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 5 * Real.pi / 12 → f'(x) > 0 :=
by sorry

end monotonic_increasing_intervals_l615_615803


namespace deltoid_diagonals_right_angles_l615_615939

-- Define the quadrilateral and its properties
def is_deltoid (ACE O₁ : Type) [quadrilateral ACE O₁] : Prop := 
  -- A deltoid (kite) has two pairs of adjacent sides of equal length and diagonals intersect at right angles
  (∃ (A E C O₁ : Type), 
    pair_adjacents_equal A E ∧ 
    pair_adjacents_equal C O₁ ∧ 
    diagonals_intersect_right_angles A C E O₁)

-- Given that ACE O₁ is a deltoid, we need to prove the angles are right angles
theorem deltoid_diagonals_right_angles 
  (ACE O₁ : Type) [quadrilateral ACE O₁]
  (h : is_deltoid ACE O₁) : 
  angle_at_diagonal C N E = 90 ∧ angle_at_diagonal C K E = 90 := 
begin 
  sorry
end

end deltoid_diagonals_right_angles_l615_615939


namespace orange_balls_count_l615_615275

-- Define the constants
constant total_balls : ℕ := 50
constant red_balls : ℕ := 20
constant blue_balls : ℕ := 10

-- Define the conditions
axiom total_parts : total_balls = red_balls + blue_balls + (total_balls - red_balls - blue_balls)
axiom pink_or_orange_balls : total_balls - red_balls - blue_balls = 20
axiom pink_is_three_times_orange {O P : ℕ} : P = 3 * O
axiom sum_pink_orange {O P : ℕ} : P + O = 20

-- Main statement to prove
theorem orange_balls_count : ∃ O : ℕ, ∀ P : ℕ, P = 3 * O → P + O = 20 → O = 5 :=
by
  sorry

end orange_balls_count_l615_615275


namespace comparison_abc_l615_615081

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615081


namespace convex_pentagon_largest_angle_l615_615244

theorem convex_pentagon_largest_angle 
  (x : ℝ)
  (h1 : (x + 2) + (2 * x + 3) + (3 * x + 6) + (4 * x + 5) + (5 * x + 4) = 540) :
  5 * x + 4 = 532 / 3 :=
by
  sorry

end convex_pentagon_largest_angle_l615_615244


namespace problem_l615_615525

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615525


namespace value_range_of_function_l615_615998

-- Statement of the problem, proving the range of the function equals to the interval [-2, 0]
theorem value_range_of_function :
  (∀ x : ℝ, y = sin x - |sin x| → y ∈ Set.Icc (-2 : ℝ) (0 : ℝ)) :=
sorry

end value_range_of_function_l615_615998


namespace range_of_f_monotone_decreasing_interval_l615_615367

-- Definition of the function f(x)
def f (x : Real) : Real := cos(x)^4 - 2 * sin(x) * cos(x) - sin(x)^4

-- Problem 1: Prove the range of f(x) is [-sqrt(2), 1]
theorem range_of_f : ∀ x ∈ Icc 0 (Real.pi / 2), f x ∈ Icc (-Real.sqrt 2) 1 :=
by
  sorry

-- Problem 2: Prove the interval where f(x) is monotonically decreasing
theorem monotone_decreasing_interval : ∀ x ∈ Icc 0 (Real.pi / 2), (0 < x ∧ x < 3 * Real.pi / 8) → deriv f x < 0 :=
by
  sorry

end range_of_f_monotone_decreasing_interval_l615_615367


namespace partition_into_perfect_squares_l615_615135

noncomputable def p (n : ℕ) : ℕ :=
∑ k in Finset.range n, 3^k

theorem partition_into_perfect_squares :
  ∀ n : ℕ, ∃ (pn : ℕ), (pn = p n) ∧ ∃ (segments : Filter (Finset (Fin (pn + 1))) ),
    ∀ seg in segments, ∃ k : ℕ, ∑ m in seg, m = k^2 := 
by
  sorry

end partition_into_perfect_squares_l615_615135


namespace octagon_triang_area_ratio_l615_615560

theorem octagon_triang_area_ratio (K : ℝ) (h1 : ∀ (a b c : ℕ), regular_octagon ABCDEFGH ∧ small_eq_triangle a ∧ small_eq_triangle b ∧ small_eq_triangle c → area (triangle ABJ) = 2 * K) 
                                   (h2 : ∀ (d e f : ℕ), large_eq_triangle d ∧ large_eq_triangle e ∧ large_eq_triangle f ∧ regular_octagon ABCDEFGH → area (triangle ACE) = 4 * K) :
  (area (triangle ABJ)) / (area (triangle ACE)) = 1 / 2 :=
by sorry

end octagon_triang_area_ratio_l615_615560


namespace angle_CED_is_90_degrees_l615_615166

-- Definition of points and distances, and the result we want
theorem angle_CED_is_90_degrees (O C D P E : Type)
    (h1 : dist O C = dist O D)
    (h2 : dist O P = dist O E)
    (h3 : is_midpoint P C D)
    (h4 : dist P C = dist P D ∧ dist P C = dist P E) :
  angle C E D = 90 :=
sorry

-- Definitions for distance and midpoint conditions would need to be established within Lean's geometry framework

end angle_CED_is_90_degrees_l615_615166


namespace relation_among_abc_l615_615094

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615094


namespace right_triangle_division_l615_615940

theorem right_triangle_division (A B C D E F : Type) 
  (AB AC BC BD DF CE EF : ℝ)
  (h_right : ∠ABC = 90)
  (h_iso1 : AB = AD)
  (h_iso2 : BF = FD)
  (h_iso3 : CE = EF) :
  ∃ D E F, is_isosceles (triangle A B D) ∧ is_isosceles (triangle B F D) ∧ is_isosceles (triangle C E F) := 
sorry

end right_triangle_division_l615_615940


namespace P_I_plus_F_eq_one_l615_615982

theorem P_I_plus_F_eq_one {n : ℕ} (I F : ℝ) 
    (h1 : (√10 + 3)^(2 * n + 1) = I + F)
    (h2 : (√10 - 3)^(2 * n + 1) * (√10 + 3)^(2 * n + 1) = 1) :
  (F * (I + F)) = 1 :=
sorry

end P_I_plus_F_eq_one_l615_615982


namespace compare_a_b_c_l615_615013

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615013


namespace subsets_neither_A_nor_B_l615_615380

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {4, 5, 6, 7, 8}

theorem subsets_neither_A_nor_B : 
  (U.powerset.card - A.powerset.card - B.powerset.card + (A ∩ B).powerset.card) = 196 := by 
  sorry

end subsets_neither_A_nor_B_l615_615380


namespace odd_function_value_l615_615345

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - Real.sin x + b + 2

theorem odd_function_value (a b : ℝ) (h1 : ∀ x, f x b = -f (-x) b) (h2 : a - 4 + 2 * a - 2 = 0) : f a b + f (2 * -a) b = 0 := by
  sorry

end odd_function_value_l615_615345


namespace addition_pyramid_max_value_l615_615318

theorem addition_pyramid_max_value (a b c d e f g : ℕ) 
  (h1 : {a, b, c, d, e, f, g} = {1, 1, 2, 2, 3, 3, 4}) :
  a + 5*b + 10*c + 10*d + 5*e + f + g ≤ 65 :=
sorry

end addition_pyramid_max_value_l615_615318


namespace tangent_line_equation_l615_615976

open Set Filter

-- Define the function representing the curve y = x^2
def curve (x : ℝ) : ℝ := x^2

-- Define the point of tangency (1, 1)
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Define the tangent line equation in standard form 2x - y - 1 = 0
def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

-- The statement of our theorem: proving that the tangent line to the curve at the
-- point (1,1) has the equation 2x - y - 1 = 0
theorem tangent_line_equation :
  let x0 := 1
  let y0 := curve x0
  let m := deriv curve x0
  ∀ y, tangent_line x0 y :=
by
  sorry

end tangent_line_equation_l615_615976


namespace total_number_of_workers_l615_615586

theorem total_number_of_workers 
    (W : ℕ) 
    (average_salary_all : ℕ := 8000) 
    (average_salary_technicians : ℕ := 12000) 
    (average_salary_rest : ℕ := 6000) 
    (total_salary_all : ℕ := average_salary_all * W) 
    (salary_technicians : ℕ := 6 * average_salary_technicians) 
    (N : ℕ := W - 6) 
    (salary_rest : ℕ := average_salary_rest * N) 
    (salary_equation : total_salary_all = salary_technicians + salary_rest) 
  : W = 18 := 
sorry

end total_number_of_workers_l615_615586


namespace candy_cost_l615_615632

theorem candy_cost (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) 
  (total_tickets : ℕ) (candies : ℕ) (cost_per_candy : ℕ) 
  (h1 : tickets_whack_a_mole = 8) (h2 : tickets_skee_ball = 7)
  (h3 : total_tickets = tickets_whack_a_mole + tickets_skee_ball)
  (h4 : candies = 3) (h5 : total_tickets = candies * cost_per_candy) :
  cost_per_candy = 5 :=
by
  sorry

end candy_cost_l615_615632


namespace john_spent_money_on_soap_l615_615889

def number_of_bars : ℕ := 20
def weight_per_bar : ℝ := 1.5
def cost_per_pound : ℝ := 0.5

theorem john_spent_money_on_soap :
  let total_weight := number_of_bars * weight_per_bar in
  let total_cost := total_weight * cost_per_pound in
  total_cost = 15 :=
by
  sorry

end john_spent_money_on_soap_l615_615889


namespace max_variance_of_nonneg_triple_l615_615830

def variance (a b c : ℝ) : ℝ :=
  let μ := (a + b + c) / 3
  in (1 / 3) * ((a - μ)^2 + (b - μ)^2 + (c - μ)^2)

theorem max_variance_of_nonneg_triple (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a > 0) (h₄ : a + b + c = 6) :
  variance a b c ≤ 8 :=
sorry

end max_variance_of_nonneg_triple_l615_615830


namespace not_divisible_2_n_minus_1_l615_615916

theorem not_divisible_2_n_minus_1 (n : ℤ) (h1 : n > 1) : ¬(n ∣ (2 ^ n - 1)) := sorry

end not_divisible_2_n_minus_1_l615_615916


namespace tangent_line_circle_l615_615810

theorem tangent_line_circle (a : ℤ) : (a = 8 ∨ a = -18) ↔ (∀ x y : ℝ, 5 * x + 12 * y + a = 0 → x^2 + y^2 - 2 * x = 0) :=
begin
  sorry
end

end tangent_line_circle_l615_615810


namespace graduation_messages_count_l615_615660

theorem graduation_messages_count : 
  let n := 45 in
  let k := 2 in
  n.choose k = 1980 := 
sorry

end graduation_messages_count_l615_615660


namespace union_complement_eq_l615_615509

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615509


namespace total_school_supplies_l615_615316

theorem total_school_supplies (rows : ℕ) (crayons_per_row : ℕ) (colored_pencils_per_row : ℕ) (graphite_pencils_per_row : ℕ) :
  rows = 28 →
  crayons_per_row = 12 →
  colored_pencils_per_row = 15 →
  graphite_pencils_per_row = 18 →
  crayons_per_row * rows + colored_pencils_per_row * rows + graphite_pencils_per_row * rows = 1260 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end total_school_supplies_l615_615316


namespace union_complement_eq_l615_615472

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615472


namespace union_complement_l615_615488

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615488


namespace total_area_of_combined_figure_l615_615584

noncomputable def combined_area (A_triangle : ℕ) (b : ℕ) : ℕ :=
  let h := (2 * A_triangle) / b
  let A_square := b * b
  A_square + A_triangle

theorem total_area_of_combined_figure :
  combined_area 720 40 = 2320 := by
  sorry

end total_area_of_combined_figure_l615_615584


namespace union_complement_eq_l615_615513

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615513


namespace find_tuesday_temperature_l615_615158

variable (T W Th F : ℝ)

def average_temperature_1 : Prop := (T + W + Th) / 3 = 52
def average_temperature_2 : Prop := (W + Th + F) / 3 = 54
def friday_temperature : Prop := F = 53

theorem find_tuesday_temperature (h1 : average_temperature_1 T W Th) (h2 : average_temperature_2 W Th F) (h3 : friday_temperature F) :
  T = 47 :=
by
  sorry

end find_tuesday_temperature_l615_615158


namespace cost_of_each_soccer_ball_l615_615960

theorem cost_of_each_soccer_ball (total_amount_paid : ℕ) (change_received : ℕ) (number_of_balls : ℕ)
  (amount_spent := total_amount_paid - change_received)
  (unit_price := amount_spent / number_of_balls) :
  total_amount_paid = 100 →
  change_received = 20 →
  number_of_balls = 2 →
  unit_price = 40 := by
  sorry

end cost_of_each_soccer_ball_l615_615960


namespace compare_abc_l615_615004

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615004


namespace det_of_matrix_l615_615438

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)

theorem det_of_matrix (h1 : 1 ≤ n)
  (h2 : A ^ 7 + A ^ 5 + A ^ 3 + A - 1 = 0) :
  0 < Matrix.det A :=
sorry

end det_of_matrix_l615_615438


namespace gcd_x1995_x1996_l615_615706

def lcm (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

-- Define the sequence
def x : ℕ → ℕ
| 0     := 0
| 1     := 19
| 2     := 95
| (n+3) := lcm (x (n+2)) (x (n+1)) + x (n+1)

-- Prove that the gcd of x 1995 and x 1996 is 19
theorem gcd_x1995_x1996 : Nat.gcd (x 1995) (x 1996) = 19 :=
sorry

end gcd_x1995_x1996_l615_615706


namespace comparison_abc_l615_615085

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615085


namespace range_of_values_for_k_l615_615353

theorem range_of_values_for_k (k : ℝ) (h : k ≠ 0) :
  (1 : ℝ) ∈ { x : ℝ | k^2 * x^2 - 6 * k * x + 8 ≥ 0 } ↔ (k ≥ 4 ∨ k ≤ 2) := 
by
  -- proof 
  sorry

end range_of_values_for_k_l615_615353


namespace union_with_complement_l615_615517

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615517


namespace graph_C2_function_form_l615_615168

-- Definitions for the transformations and the final result.
def symm_C_with_respect_to_line_x_eq_1 (f : ℝ → ℝ) (x : ℝ) := f(2 - x)

def shift_left_by_2 (f : ℝ → ℝ) (x : ℝ) := f(x + 2)

theorem graph_C2_function_form (f : ℝ → ℝ) :
  ∀ x : ℝ, shift_left_by_2 (symm_C_with_respect_to_line_x_eq_1 f) x = f(1 - x) :=
by
  sorry

end graph_C2_function_form_l615_615168


namespace length_of_AC_l615_615413

/-- 
Given AB = 15 cm, DC = 24 cm, and AD = 7 cm, 
prove that the length of AC to the nearest tenth of a centimeter is 30.3 cm.
-/
theorem length_of_AC {A B C D : Point} 
  (hAB : dist A B = 15) (hDC : dist D C = 24) (hAD : dist A D = 7) : 
  dist A C ≈ 30.3 :=
sorry

end length_of_AC_l615_615413


namespace BD_eq_DE_l615_615582

-- Given data definitions
def is_isosceles_triangle (ABC : Triangle) : Prop := 
  ABC.is_isosceles ∧ ABC.angle_ABC = 108

def angle_bisector (A B C D : Point) (ABC : Triangle) : Prop := 
  angle_eq (line A D).angle_with (line B C) (Bachelor_720.B.part_cube \C) 54

def perpendicular_from_D (D : Point) (AD : Line) (E : Point) (AC : Line) : Prop := 
  D.is_foot_of_perpendicular AD ∧ E ∈ intersection_of_perpendicular AD AC ∧ 
  AC.angle_between AD = 90

-- Prove the stated geometric relationships
theorem BD_eq_DE (A B C D E : Point) (ABC : Triangle) :
  is_isosceles_triangle ABC → angle_bisector A B C D ABC → 
  perpendicular_from_D D (angle_bisector_line A D) E (line A C) → 
  dist (B, D) = dist (D, E) :=
by
  intros
  sorry

end BD_eq_DE_l615_615582


namespace incorrect_geometric_solid_description_l615_615684

theorem incorrect_geometric_solid_description :
  ¬ (
    ∀ (sphere : Type) (semicircle_rotation : sphere → Prop),
    (∀ (cone : Type) (triangle_rotation : cone → Prop),
      (∀ (frustum : Type) (cone_cut : frustum → Prop),
        (∀ (cylinder : Type) (rectangle_rotation : cylinder → Prop),
          semicircle_rotation sphere →
          triangle_rotation cone →
          (∃ plane : frustum, cone_cut plane) →
          rectangle_rotation cylinder
        )
      )
    )
  ):
  ∃ incorrect_option : Option, Option.get orElse incorrect_option = 3 :=
by
  sorry

end incorrect_geometric_solid_description_l615_615684


namespace girls_distance_in_miles_l615_615222

-- Definitions based on conditions
def boys_laps : ℕ := 27
def additional_laps : ℕ := 9
def lap_distance : ℚ := 3 / 4

-- Assertion to prove
theorem girls_distance_in_miles : 
  let girls_laps := boys_laps + additional_laps in
  let distance := girls_laps * lap_distance in
  distance = 27 := 
  by
  sorry

end girls_distance_in_miles_l615_615222


namespace hospital_allocation_l615_615662

theorem hospital_allocation :
    (Nat.choose 3 1) * (Nat.choose 6 2) * (Nat.choose 2 1) * (Nat.choose 4 2) = 540 := by
  sorry

end hospital_allocation_l615_615662


namespace smallest_five_digit_perfect_square_and_cube_l615_615623

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ n = 15625 :=
by
  sorry

end smallest_five_digit_perfect_square_and_cube_l615_615623


namespace rates_of_interest_l615_615257

theorem rates_of_interest (P_B P_C T_B T_C SI_B SI_C : ℝ) (R_B R_C : ℝ)
  (hB1 : P_B = 5000) (hB2: T_B = 5) (hB3: SI_B = 2200)
  (hC1 : P_C = 3000) (hC2 : T_C = 7) (hC3 : SI_C = 2730)
  (simple_interest : ∀ {P R T SI : ℝ}, SI = (P * R * T) / 100)
  : R_B = 8.8 ∧ R_C = 13 := by
  sorry

end rates_of_interest_l615_615257


namespace distance_between_centers_l615_615196

noncomputable theory
open_locale classical

-- Given conditions
variables {A B C D O1 O2 : Type}
variables {a b : ℝ}  -- BC = a and BD = b
variables (h1 : intersect A B O1 C O2 D)
variables (h2 : diameter_through A C O1) 
variables (h3 : diameter_through A D O2)
variables (h4 : distance C B = a) 
variables (h5 : distance D B = b)

-- Question: Prove the distance between centers O1 and O2
theorem distance_between_centers :
  distance O1 O2 = (1 / 2) * (a + b) ∨ 
  distance O1 O2 = (1 / 2) * |a - b| :=
sorry

end distance_between_centers_l615_615196


namespace find_S5_l615_615792

theorem find_S5 :
  ∃ (S : ℕ → ℕ) (a : ℕ → ℕ),
  a 1 = 2 ∧
  (∀ n : ℕ, n > 0 → a (n + 1) = (∑ i in Finset.range n, a (i + 1)) + 1) ∧
  S 5 = (∑ i in Finset.range 5, a (i + 1)) ∧
  S 5 = 47 :=
sorry

end find_S5_l615_615792


namespace compare_abc_l615_615051

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615051


namespace white_area_correct_l615_615548

def board_width : ℕ := 18
def board_height : ℕ := 6
def total_area : ℝ := board_width * board_height

def area_P : ℝ := (6 * 1) + (1 * 4)
def diameter_O : ℝ := 5
def area_O : ℝ := Real.pi * (diameter_O / 2) ^ 2
def area_S : ℝ := (3 * 1) * 3 + (1 * 1) * 2
def area_T : ℝ := (1 * 18) + (6 * 1)

def total_black_area : ℝ := area_P + area_O + area_S + area_T

noncomputable def white_area : ℝ := total_area - total_black_area

theorem white_area_correct : white_area = 43.365 := 
by
  sorry -- The proof is omitted

end white_area_correct_l615_615548


namespace part1_a_n_part1_b_n_part2_T_n_l615_615457

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1 else 3 * a (n - 1)

def S (n : ℕ) : ℕ -> ℕ
| 0 := 0
| (n + 1) := S n + n

def b (n : ℕ) : ℕ :=
if n = 1 then 1 else 2 * b (n - 1)

def c (n : ℕ) : ℕ :=
b n * (n - 1)

def T (n : ℕ) : ℕ :=
(0..(n-1)).sum (λ k, c k)

theorem part1_a_n (n : ℕ) : a n = 3^(n - 1) := sorry
theorem part1_b_n (n : ℕ) : b n = 2^(n - 1) := sorry
theorem part2_T_n (n : ℕ) : T n = (n - 2) * 2^n + 2 := sorry

end part1_a_n_part1_b_n_part2_T_n_l615_615457


namespace intervals_of_monotonicity_f_range_of_m_log_inequality_l615_615372

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (m / x) - (3 / (x^2)) - 1

theorem intervals_of_monotonicity_f :
  (∀ x : ℝ, 0 < x ∧ x < exp 1 → 0 < (1 - log x) / (x ^ 2)) ∧
  (∀ x : ℝ, exp 1 < x → 0 > (1 - log x) / (x ^ 2)) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * f x ≥ g x m) → m ≤ 4 :=
sorry

theorem log_inequality :
  ∀ x : ℝ, 0 < x → log x < (2 * x / exp 1) - (x^2 / exp x) :=
sorry

end intervals_of_monotonicity_f_range_of_m_log_inequality_l615_615372


namespace compare_a_b_c_l615_615042

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615042


namespace determine_phi_l615_615798

-- Define the function f
def f (x ϕ : ℝ) := 2 * Real.sin (x + ϕ)

-- Define the function g, which is a left shift of f by π/3
def g (x ϕ : ℝ) := f (x + π / 3) ϕ

-- Define the conditions for ϕ being between 0 and π/2
def phi_condition (ϕ : ℝ) := 0 < ϕ ∧ ϕ < π / 2

-- Define the property of an even function
def is_even (h : ℝ → ℝ) := ∀ x, h x = h (-x)

-- Main theorem to prove that ϕ = π / 6
theorem determine_phi (ϕ : ℝ) (h_phi_condition : phi_condition ϕ) (h_even : is_even (g · ϕ)) : ϕ = π / 6 :=
by
  sorry

end determine_phi_l615_615798


namespace compare_abc_l615_615029

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615029


namespace chair_cost_l615_615304

theorem chair_cost (T P n : ℕ) (hT : T = 135) (hP : P = 55) (hn : n = 4) : 
  (T - P) / n = 20 := by
  sorry

end chair_cost_l615_615304


namespace difference_in_ages_l615_615972

variables (J B : ℕ)

-- The conditions: Jack's age is twice Bill's age, and in eight years, Jack will be three times Bill's age then.
axiom condition1 : J = 2 * B
axiom condition2 : J + 8 = 3 * (B + 8)

-- The theorem statement we are proving: The difference in their current ages is 16.
theorem difference_in_ages : J - B = 16 :=
by
  sorry

end difference_in_ages_l615_615972


namespace percent_transplanted_seeds_eq_l615_615752

-- Define the number of seeds, germination rates, and transplant success rates for each plot.
def plot1 := (300, 0.25, 0.90)
def plot2 := (200, 0.35, 0.85)
def plot3 := (400, 0.45, 0.80)
def plot4 := (350, 0.15, 0.95)
def plot5 := (150, 0.50, 0.70)

-- Function to calculate the number of successfully transplanted seeds for each plot.
def transplanted_seeds (plot : ℕ × ℝ × ℝ) : ℝ :=
  plot.1 * plot.2 * plot.3

-- List of all the plots.
def plots := [plot1, plot2, plot3, plot4, plot5]

-- Total seeds planted.
def total_seeds_planted : ℕ := (plots.map (λ plot => plot.1)).sum

-- Total successfully transplanted seeds.
def total_transplanted_seeds : ℝ := (plots.map transplanted_seeds).sum

-- The percentage of successfully transplanted seeds.
def percent_successfully_transplanted : ℝ :=
  (total_transplanted_seeds / total_seeds_planted.to_nat) * 100

-- The theorem to be proved.
theorem percent_transplanted_seeds_eq :
  percent_successfully_transplanted ≈ 26.67 :=
sorry

end percent_transplanted_seeds_eq_l615_615752


namespace increasing_difference_implies_inequality_l615_615360

theorem increasing_difference_implies_inequality
  (f g : ℝ → ℝ)
  (f' g' : ℝ → ℝ)
  (h_der_f : ∀ x, has_deriv_at f (f' x) x)
  (h_der_g : ∀ x, has_deriv_at g (g' x) x)
  (h_diff_pos : ∀ x, f' x > g' x) :
  ∀ a x b, a < x → x < b → f(x) + g(a) > g(x) + f(a) :=
by
  sorry

end increasing_difference_implies_inequality_l615_615360


namespace sequence_first_five_terms_l615_615814

noncomputable def a_n (n : ℕ) : ℤ := (-1) ^ n + (n : ℤ)

theorem sequence_first_five_terms :
  a_n 1 = 0 ∧
  a_n 2 = 3 ∧
  a_n 3 = 2 ∧
  a_n 4 = 5 ∧
  a_n 5 = 4 :=
by
  sorry

end sequence_first_five_terms_l615_615814


namespace number_of_factors_l615_615698

theorem number_of_factors (a b c d : ℕ) (h₁ : a = 6) (h₂ : b = 6) (h₃ : c = 5) (h₄ : d = 1) :
  ((a + 1) * (b + 1) * (c + 1) * (d + 1) = 588) :=
by {
  -- This is a placeholder for the actual proof
  sorry
}

end number_of_factors_l615_615698


namespace relation_among_abc_l615_615091

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615091


namespace intersection_M_N_l615_615460

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | 1 - |x| > 0}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_M_N_l615_615460


namespace inverse_function_of_ln_l615_615170

theorem inverse_function_of_ln (x : ℝ) (hx : x > 1) :
  ∃ y : ℝ, y = ln (x - 1) + 1 ∧ (∀ t : ℝ, t = ln (y - 1) + 1 → t = e^(y-1 + 1 - t)) :=
sorry

end inverse_function_of_ln_l615_615170


namespace union_complement_eq_l615_615508

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615508


namespace smallest_denominator_between_l615_615281

theorem smallest_denominator_between :
  ∃ (a b : ℕ), b > 0 ∧ a < b ∧ 6 / 17 < (a : ℚ) / b ∧ (a : ℚ) / b < 9 / 25 ∧ (∀ (c d : ℕ), d > 0 → c < d → 6 / 17 < (c : ℚ) / d → (c : ℚ) / d < 9 / 25 → b ≤ d) ∧ a = 5 ∧ b = 14 :=
by
  existsi 5
  existsi 14
  sorry

end smallest_denominator_between_l615_615281


namespace first_day_is_sunday_l615_615581

-- Define the days of the week
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open Day

-- Function to determine the day of the week for a given day number
def day_of_month (n : ℕ) (start_day : Day) : Day :=
  match n % 7 with
  | 0 => start_day
  | 1 => match start_day with
          | Sunday    => Monday
          | Monday    => Tuesday
          | Tuesday   => Wednesday
          | Wednesday => Thursday
          | Thursday  => Friday
          | Friday    => Saturday
          | Saturday  => Sunday
  | 2 => match start_day with
          | Sunday    => Tuesday
          | Monday    => Wednesday
          | Tuesday   => Thursday
          | Wednesday => Friday
          | Thursday  => Saturday
          | Friday    => Sunday
          | Saturday  => Monday
-- ... and so on for the rest of the days of the week.
  | _ => start_day -- Assuming the pattern continues accordingly.

-- Prove that the first day of the month is a Sunday given that the 18th day of the month is a Wednesday.
theorem first_day_is_sunday (h : day_of_month 18 Wednesday = Wednesday) : day_of_month 1 Wednesday = Sunday :=
  sorry

end first_day_is_sunday_l615_615581


namespace compare_a_b_c_l615_615036

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615036


namespace expression_factorization_l615_615232

variables (a b c : ℝ)

theorem expression_factorization :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3)
  = (a - b) * (b - c) * (c - a) * (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
sorry

end expression_factorization_l615_615232


namespace union_complement_set_l615_615462

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615462


namespace area_under_curve_l615_615307

noncomputable def f : ℝ → ℝ
| x := if h : 0 ≤ x ∧ x ≤ 4 then x
       else if h : 4 ≤ x ∧ x <= 7 then 3*x - 8
       else if h : 7 ≤ x ∧ x <= 10 then x^2 - 14*x + 49
       else 0

theorem area_under_curve :
  let K := ( ∫ x in (0:ℝ)..(4:ℝ), f x dx) + ( ∫ x in (4:ℝ)..(7:ℝ), f x dx) + ( ∫ x in (7:ℝ)..(10:ℝ), f x dx) 
  in K = 42.5 :=
begin
  sorry

end area_under_curve_l615_615307


namespace sunland_more_plates_than_moonland_l615_615925

theorem sunland_more_plates_than_moonland : 
  let sunland_plates := 26^4 * 10^2
  let moonland_plates := 26^3 * 10^3
  (sunland_plates - moonland_plates) = 7321600 := 
by
  sorry

end sunland_more_plates_than_moonland_l615_615925


namespace compare_abc_l615_615045

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615045


namespace marcus_percentage_of_team_points_l615_615538

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l615_615538


namespace alloy_mixture_l615_615655

theorem alloy_mixture (m n p q : ℚ) (hmn : m ≠ n) (hpq : p ≠ q) : 
  let x := (1/2 : ℚ) + ((m * p - n * q) / (2 * (n * p - m * q))),
      y := (1/2 : ℚ) - ((m * p - n * q) / (2 * (n * p - m * q)))
  in x + y = 1 :=
by sorry

end alloy_mixture_l615_615655


namespace compare_abc_l615_615060

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615060


namespace gcd_floor_eq_lhs_rhs_l615_615647

open Int

theorem gcd_floor_eq_lhs_rhs (n : ℕ) (hn : 2 ≤ n)
  (a : ℕ → ℕ) (ha1 : ∀ i ≠ j, Nat.gcd (a i) (a j) = 1)
  (ha2 : ∀ i, a i ≥ 1402) :
  (∑ i in (Finset.range n), Int.floor ((a i) / (a ((i + 1) % n))))
    =
  (∑ i in (Finset.range n), Int.floor ((a ((i + 1) % n)) / (a i))) :=
  sorry

end gcd_floor_eq_lhs_rhs_l615_615647


namespace eccentric_leq_unsociable_l615_615619

def person : Type := sorry

def is_unsociable (p : person) : Prop :=
  sorry -- fewer than 10 acquaintances

def is_eccentric (p : person) : Prop :=
  ∀ q : person, q ∈ acquaintances_of p → is_unsociable q

def num_unsociable : ℕ := sorry -- number of unsociable people

def num_eccentric : ℕ := sorry -- number of eccentrics

theorem eccentric_leq_unsociable : num_eccentric ≤ num_unsociable := sorry

end eccentric_leq_unsociable_l615_615619


namespace alpha_is_2_l615_615832

theorem alpha_is_2 (α : ℝ) (h_curve : ∀ x, y x = x^α + 1) 
                    (h_point : y 1 = 2) 
                    (h_tangent : ∀ x, tangent_of y 1 = y' 0) : 
                    α = 2 := by
  have h_slope : y' = α * (1^(α-1)) := sorry
  have h_tangent_line : 2 = tangent_slope (0,0) (1,2) := sorry
  exact sorry

end alpha_is_2_l615_615832


namespace initial_amount_in_account_l615_615578

theorem initial_amount_in_account 
  (h₀ : ∀ x : ℝ, x * 0.1 = x / 10)
  (h₁ : ∀ P : ℝ, (1.10 * P + 10) * 1.10 + 10 = 142) :
  100 = (λ P : ℝ, P) :=
sorry

end initial_amount_in_account_l615_615578


namespace three_digit_numbers_count_l615_615760

-- Definitions based on the conditions
def digits := {1, 2, 3, 4, 5} : set ℕ

def valid_combination (nums : list ℕ) : Prop :=
  nums.length = 3 ∧ (2 ∉ nums ∨ 3 ∉ nums ∨ nums.index_of 2 < nums.index_of 3)

-- The proof goal
theorem three_digit_numbers_count : 
  (finset.univ.filter (λ n, valid_combination (nat.digits 10 n))).card = 51 := 
sorry

end three_digit_numbers_count_l615_615760


namespace problem_proof_l615_615817

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
real.sqrt (v.1^2 + v.2^2)

noncomputable def vector_dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem problem_proof (a b : ℝ × ℝ) (θ : ℝ) (h₁ : vector_magnitude (a.1 + b.1, a.2 + b.2) = 3) (h₂ : vector_magnitude (a.1 - b.1, a.2 - b.2) = 1) :
  (vector_magnitude a) / ((vector_magnitude b) * real.cos θ) + (vector_magnitude b) / ((vector_magnitude a) * real.cos θ) = 5 / 2 :=
sorry

end problem_proof_l615_615817


namespace spiral_length_100_l615_615286

-- Definitions based on conditions
def segment_length (n : ℕ) : ℕ :=
  if n ≤ 4 then 1
  else if n ≤ 6 then 2
  else 
    let k := (n - 5) / 3 in
    k + 2

-- Summing up the length of first 100 segments
def total_spiral_length (m : ℕ) : ℕ :=
  (Finset.range m).sum (fun n => segment_length (n + 1))

-- Lean statement to prove the problem
theorem spiral_length_100 : total_spiral_length 100 = 1156 :=
by 
  sorry

end spiral_length_100_l615_615286


namespace find_f3_l615_615580

theorem find_f3 (f : ℚ → ℚ)
  (h : ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x) / x = x^3) :
  f 3 = 7753 / 729 :=
sorry

end find_f3_l615_615580


namespace chromic_acid_mass_percentage_O_l615_615327

-- Definitions of molar masses
def molar_mass_H : Float := 1.01
def molar_mass_Cr : Float := 51.99
def molar_mass_O : Float := 16.00

-- Definition of the chemical formula H2CrO4
def molar_mass_H2CrO4 : Float := (2 * molar_mass_H) + molar_mass_Cr + (4 * molar_mass_O)

-- Mass of oxygen in one mole of H2CrO4
def mass_of_O : Float := 4 * molar_mass_O

-- Function to calculate mass percentage
def mass_percentage_O : Float := (mass_of_O / molar_mass_H2CrO4) * 100

-- Theorem proving the mass percentage of O in Chromic acid equals 54.23%
theorem chromic_acid_mass_percentage_O : mass_percentage_O ≈ 54.23 := 
by 
  sorry

end chromic_acid_mass_percentage_O_l615_615327


namespace alexis_initial_budget_l615_615682

-- Define all the given conditions
def cost_shirt : Int := 30
def cost_pants : Int := 46
def cost_coat : Int := 38
def cost_socks : Int := 11
def cost_belt : Int := 18
def cost_shoes : Int := 41
def amount_left : Int := 16

-- Define the total expenses
def total_expenses : Int := cost_shirt + cost_pants + cost_coat + cost_socks + cost_belt + cost_shoes

-- Define the initial budget
def initial_budget : Int := total_expenses + amount_left

-- The proof statement
theorem alexis_initial_budget : initial_budget = 200 := by
  sorry

end alexis_initial_budget_l615_615682


namespace sequence_congruence_l615_615301

-- Defining the sequence {a_n}
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := ∑ k in Finset.range (n + 1), Nat.choose (n + 1) k * a k

-- Defining the relationship to prove
theorem sequence_congruence (m : ℕ) (p q r : ℕ) [hp : Fact (Nat.Prime p)] (hm : m > 0) (hq : q ≥ 0) (hr : r ≥ 0) :
  a (p^m * q + r) ≡ a (p^(m-1) * q + r) [MOD p^m] :=
sorry

end sequence_congruence_l615_615301


namespace length_of_second_sheet_l615_615967

-- Conditions
def area_first_sheet : ℝ := 2 * (11 * 19)
def width_second_sheet : ℝ := 9.5
def area_diff : ℝ := 100

-- Goal: find the length L of the second sheet
theorem length_of_second_sheet : 
  ∃ L : ℝ, (2 * (width_second_sheet * L)) = area_first_sheet - area_diff ∧ L = 16.7368 :=
by
  let L := 318 / 19
  use L
  split
  · calc 2 * (width_second_sheet * L)
         = 2 * (9.5 * L) : by rfl
     ... = 19 * L : by simp [width_second_sheet]
     ... = 318 : by norm_num
     ... = area_first_sheet - area_diff : by norm_num [area_first_sheet, area_diff]
  · calc L = 318 / 19 : by rfl
         ... = 16.7368 : by norm_num

end length_of_second_sheet_l615_615967


namespace bill_toilet_paper_duration_l615_615693

variables (rolls : ℕ) (squares_per_roll : ℕ) (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ)

def total_squares (rolls squares_per_roll : ℕ) : ℕ := rolls * squares_per_roll

def squares_per_day (bathroom_visits_per_day squares_per_visit : ℕ) : ℕ := bathroom_visits_per_day * squares_per_visit

def days_supply_last (total_squares squares_per_day : ℕ) : ℕ := total_squares / squares_per_day

theorem bill_toilet_paper_duration
  (h1 : rolls = 1000)
  (h2 : squares_per_roll = 300)
  (h3 : bathroom_visits_per_day = 3)
  (h4 : squares_per_visit = 5)
  :
  days_supply_last (total_squares rolls squares_per_roll) (squares_per_day bathroom_visits_per_day squares_per_visit) = 20000 := sorry

end bill_toilet_paper_duration_l615_615693


namespace g_equals_zero_for_all_x_l615_615915

def g (x: ℝ) : ℝ := sqrt (2 * sin x ^ 4 + 3 * cos x ^ 2) - sqrt (2 * cos x ^ 4 + 3 * sin x ^ 2)

theorem g_equals_zero_for_all_x : ∀ x : ℝ, g x = 0 :=
by
  intro x
  -- Placeholder to indicate where the proof would go
  sorry

end g_equals_zero_for_all_x_l615_615915


namespace compare_a_b_c_l615_615016

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615016


namespace cubic_increasing_on_positive_real_l615_615233

theorem cubic_increasing_on_positive_real (a b : ℝ) (h : a > b) : a^3 > b^3 :=
by
  have h' : a^3 ≤ b^3 := sorry
  contradiction

end cubic_increasing_on_positive_real_l615_615233


namespace roy_is_6_years_older_than_julia_l615_615563

theorem roy_is_6_years_older_than_julia :
  ∀ (R J K : ℕ) (x : ℕ), 
    R = J + x →
    R = K + x / 2 →
    R + 4 = 2 * (J + 4) →
    (R + 4) * (K + 4) = 108 →
    x = 6 :=
by
  intros R J K x h1 h2 h3 h4
  -- Proof goes here (using sorry to skip the proof)
  sorry

end roy_is_6_years_older_than_julia_l615_615563


namespace exists_plane_equal_angles_with_skew_lines_l615_615825

-- Definitions of skew lines and plane making equal angles
variable {α : Type*} [EuclideanSpace α]
variables (a b : Line α)
variables (α : Plane α)

-- Skew lines definition
def are_skew (a b : Line α) : Prop :=
  ¬∃ (P : Plane α), a ∈ P ∧ b ∈ P

-- Plane making equal angles with skew lines a and b
def plane_makes_equal_angles (α : Plane α) (a b : Line α) : Prop :=
  ∃ (c : Line α), c ⊥ α ∧ (angle_between_lines a c = angle_between_lines b c)

-- Statement of the theorem
theorem exists_plane_equal_angles_with_skew_lines (a b : Line α) :
  are_skew a b → ∃ (α : Plane α), plane_makes_equal_angles α a b :=
sorry

end exists_plane_equal_angles_with_skew_lines_l615_615825


namespace find_a_subtract_two_l615_615389

theorem find_a_subtract_two (a b : ℤ) 
    (h1 : 2 + a = 5 - b) 
    (h2 : 5 + b = 8 + a) : 
    2 - a = 2 := 
by
  sorry

end find_a_subtract_two_l615_615389


namespace converse_proposition_inverse_proposition_contrapositive_proposition_l615_615219

theorem converse_proposition (x y : ℝ) : (xy = 0 → x^2 + y^2 = 0) = false :=
sorry

theorem inverse_proposition (x y : ℝ) : (x^2 + y^2 ≠ 0 → xy ≠ 0) = false :=
sorry

theorem contrapositive_proposition (x y : ℝ) : (xy ≠ 0 → x^2 + y^2 ≠ 0) = true :=
sorry

end converse_proposition_inverse_proposition_contrapositive_proposition_l615_615219


namespace faye_earnings_l615_615317

theorem faye_earnings
  (bead_necklaces : ℕ)
  (gem_stone_necklaces : ℕ)
  (cost_per_necklace : ℕ) :
  bead_necklaces = 3 →
  gem_stone_necklaces = 7 →
  cost_per_necklace = 7 →
  (bead_necklaces + gem_stone_necklaces) * cost_per_necklace = 70 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end faye_earnings_l615_615317


namespace astra_paths_l615_615288

-- Define a tetrahedron structure with vertices
structure Tetrahedron :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))
  (is_edge : ℕ → ℕ → Prop)

-- Instantiate the tetrahedron with 4 vertices (0, 1, 2, 3) and edges between them
def tetrahedron : Tetrahedron :=
  { vertices := {0, 1, 2, 3},
    edges := {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)},
    is_edge := λ v1 v2, (v1, v2) ∈ {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)} ∨ (v2, v1) ∈ {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)} }

-- Define a function that counts the number of valid paths
noncomputable def count_valid_paths : ℕ :=
  6  -- This is the correct answer derived from the solution.

-- Prove the main theorem
theorem astra_paths : count_valid_paths = 6 := 
by sorry

end astra_paths_l615_615288


namespace chair_cost_l615_615302

namespace ChairCost

-- Conditions
def total_cost : ℕ := 135
def table_cost : ℕ := 55
def chairs_count : ℕ := 4

-- Problem Statement
theorem chair_cost : (total_cost - table_cost) / chairs_count = 20 :=
by
  sorry

end ChairCost

end chair_cost_l615_615302


namespace compare_abc_l615_615028

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615028


namespace simplify_expression_l615_615569

theorem simplify_expression (y : ℝ) : 
  3 * y - 5 * y ^ 2 + 12 - (7 - 3 * y + 5 * y ^ 2) = -10 * y ^ 2 + 6 * y + 5 :=
by 
  sorry

end simplify_expression_l615_615569


namespace chord_length_eq_l615_615373

def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem chord_length_eq : 
  ∀ (x y : ℝ), 
  (line_eq x y) ∧ (circle_eq x y) → 
  ∃ l, l = 2 * Real.sqrt 3 :=
sorry

end chord_length_eq_l615_615373


namespace f_leq_2x_l615_615448

-- Definitions
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

variables (f : ℝ → ℝ) (x y : ℝ)

-- Conditions
axiom f_mapped : ∀ x, x ∈ A → (f x) ∈ set.univ ℝ
axiom f_1 : f 1 = 1
axiom f_nonneg : ∀ x, x ∈ A → f x ≥ 0
axiom f_additive : ∀ x y, x ∈ A ∧ y ∈ A ∧ x + y ∈ A → f (x + y) ≥ f x + f y

-- To Prove
theorem f_leq_2x : ∀ x, x ∈ A → f x ≤ 2 * x := by
  sorry

end f_leq_2x_l615_615448


namespace union_complement_eq_l615_615477

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615477


namespace solve_vol_pyramid_l615_615262

noncomputable def volume_of_pyramid (AB BC : ℝ) (h_AB : AB = 15 * Real.sqrt 2) (h_BC : BC = 17 * Real.sqrt 2) : ℝ :=
  let P := (15 * Real.sqrt 2 / 2, 17 * Real.sqrt 2 / 2) in
  let AP := Real.sqrt ((15 * Real.sqrt 2 / 2) ^ 2 + (17 * Real.sqrt 2 / 2) ^ 2) in
  let AC := Real.sqrt ((15 * Real.sqrt 2) ^ 2 + (17 * Real.sqrt 2) ^ 2) in
  let area_CD := 0.5 * 15 * Real.sqrt 2 * 17 * Real.sqrt 2 in
  let height_from_P := AC / 2 in
  (1 / 3) * area_CD * height_from_P

theorem solve_vol_pyramid : volume_of_pyramid (15 * Real.sqrt 2) (17 * Real.sqrt 2) (by simp) (by simp) = 1445 := by
  sorry

end solve_vol_pyramid_l615_615262


namespace union_complement_eq_l615_615474

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615474


namespace average_of_seven_consecutive_l615_615567

variable (a : ℕ) 

def average_of_consecutive_integers (x : ℕ) : ℕ :=
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) / 7

theorem average_of_seven_consecutive (a : ℕ) :
  average_of_consecutive_integers (average_of_consecutive_integers a) = a + 6 :=
by
  sorry

end average_of_seven_consecutive_l615_615567


namespace proof_conclusions_l615_615381

variables (Cem Ben Den : Type) (is_Ben : Cem → Prop) (is_Den : Cem → Prop)

-- Hypotheses from the problem
hypothesis H1 : ∀ (c : Cem), ¬ is_Ben c
hypothesis H2 : ∀ (d : Den), is_Ben d
hypothesis H3 : ∃ (c : Cem), ¬ is_Den c

-- Conclusions to be proved
def conclusion_B : Prop := ∃ (d : Den), ∀ (c : Cem), ¬ (d = c)
def conclusion_C : Prop := ∀ (d : Den), ∀ (c : Cem), ¬ (d = c)

theorem proof_conclusions (H1 : ∀ (c : Cem), ¬ is_Ben c) 
                          (H2 : ∀ (d : Den), is_Ben d) 
                          (H3 : ∃ (c : Cem), ¬ is_Den c) : conclusion_B Cem Den ∧ conclusion_C Cem Den :=
by 
sory  -- Proof is not required.

end proof_conclusions_l615_615381


namespace distance_from_center_to_plane_of_triangle_l615_615993

def radius : ℝ := 8

def a : ℝ := 13
def b : ℝ := 13
def c : ℝ := 10

def correct_distance : ℝ := 2 * Real.sqrt 119 / 3

theorem distance_from_center_to_plane_of_triangle (O : Point) (triangle_plane : Plane) :
  (radius = 8) ∧ (a = 13) ∧ (b = 13) ∧ (c = 10) ∧
  (inradius (⟨a, b, c⟩) (radius = 8) = inradius_value) ∧
  (distance_from_center_to_plane (O, triangle_plane) = correct_distance) :=
sorry

end distance_from_center_to_plane_of_triangle_l615_615993


namespace max_objective_value_l615_615809

-- Define the region D as a set of points within the boundaries
def region_D (x y : ℝ) : Prop := 
  (x = 3 ∧ -3 ≤ y ∧ y ≤ 3)

-- Define the objective function z
def objective_function (x y : ℝ) : ℝ :=
  x + 4 * y

-- State the theorem
theorem max_objective_value :
  ∃ x y ∈ { p : ℝ × ℝ | region_D p.1 p.2 }, objective_function x y = 15 :=
sorry

end max_objective_value_l615_615809


namespace circumcircle_eq_l615_615765

-- Definitions and conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_P : (ℝ × ℝ) := (4, 2)
def is_tangent_point (x y : ℝ) : Prop := sorry -- You need a proper definition for tangency

theorem circumcircle_eq :
  ∃ (hA : is_tangent_point 0 2) (hB : ∃ x y, is_tangent_point x y),
  ∃ (x y : ℝ), (circle_eq 0 2 ∧ circle_eq x y) ∧ (x-2)^2 + (y-1)^2 = 5 :=
  sorry

end circumcircle_eq_l615_615765


namespace average_age_nine_students_l615_615155

theorem average_age_nine_students (total_age_15_students : ℕ)
                                (total_age_5_students : ℕ)
                                (age_15th_student : ℕ)
                                (h1 : total_age_15_students = 225)
                                (h2 : total_age_5_students = 65)
                                (h3 : age_15th_student = 16) :
                                (total_age_15_students - total_age_5_students - age_15th_student) / 9 = 16 := by
  sorry

end average_age_nine_students_l615_615155


namespace smallest_k_proof_l615_615329

noncomputable def find_smallest_k : Nat :=
  7

theorem smallest_k_proof :
  ∀ (f : Polynomial ℤ), (f.degree > find_smallest_k) →
    let n := f.natDegree in
    let m := Nat.floor (n / 2) + 1 in
    (∃ (x : Fin m → ℤ), ∀ i, abs (Polynomial.eval (x i) f) = 1) →
    ¬ ∃ (g h : Polynomial ℤ), g.degree > 0 ∧ h.degree > 0 ∧ f = g * h :=
by
  intros f f_deg_gt_k n m exists_x abs_eval_x_f_eq_1 exists_factorization
  sorry

#eval find_smallest_k  -- Should output 7

end smallest_k_proof_l615_615329


namespace hyperbola_real_axis_length_proof_l615_615169

noncomputable def hyperbola_real_axis_length : ℝ :=
  let a := sqrt 15 / 2 in
  2 * a

theorem hyperbola_real_axis_length_proof :
  ∃ C O P T (C_center : point) (O_center : point) (C_asymptote : line)
  (C_equation : equation) (O_equation : equation) (T_equation : equation), 
  C_center = (0, 0) ∧
  O_center = (0, 0) ∧
  O_equation = λ (x y: ℝ), x^2 + y^2 = 5 ∧
  P = (2, -1) ∧
  T_equation = λ (x y: ℝ), 2 * x - y = 5 ∧
  parallel T_equation C_asymptote ∧
  length_real_axis C_equation = sqrt 15 := 
  sorry

end hyperbola_real_axis_length_proof_l615_615169


namespace function_extreme_points_range_l615_615352

noncomputable def has_two_extreme_points (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ ∀ x, ((x = a ∨ x = b) → f' x = 0)

theorem function_extreme_points_range (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x > 0, f x = x * (Real.log x - 2 * a * x)) ∧ has_two_extreme_points f)
  ↔ 0 < a ∧ a < 1/4 := 
by
  sorry

end function_extreme_points_range_l615_615352


namespace number_of_5s_more_than_9s_l615_615722

-- Define conditions
def total_pages : ℕ := 698

-- Main theorem statement
theorem number_of_5s_more_than_9s : 
  ∀ (n : ℕ), n = total_pages → 
    (let count_digit (d : ℕ) (n : ℕ) : ℕ :=
        (List.range n).map (λ num, (num.digits 10).count d).sum in
    count_digit 5 total_pages = count_digit 9 total_pages + 1) :=
sorry

end number_of_5s_more_than_9s_l615_615722


namespace AE_ratio_l615_615704

noncomputable def AE_div_AB (s : ℝ) : ℝ :=
  s + (s * real.sqrt 3 / 2) / s 

theorem AE_ratio (s : ℝ) : AE_div_AB s = 1 + real.sqrt 3 / 2 := 
begin
  sorry
end

end AE_ratio_l615_615704


namespace arithmetic_sequence_sum_l615_615995

theorem arithmetic_sequence_sum (a_3 a_4 a_5 : ℤ) (h₃ : a_3 = 8) (h₄ : a_4 = 14) (h₅ : a_5 = 20) :
  ∃ a_1 a_2 : ℤ, let d := a_4 - a_3 in
  let a_1 := a_3 - 2 * d in
  let a_2 := a_1 + d in
  a_1 + a_2 + a_3 + a_4 = 20 :=
by
  sorry

end arithmetic_sequence_sum_l615_615995


namespace right_triangle_with_perimeter_l615_615718

open Real

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_triangle_with_perimeter (a b c : ℝ) :
  is_valid_triangle a b c ∧ is_right_triangle a b c →
  a + b + c = 60 :=
by
  sorry

def sides_of_triangle := (10 : ℝ, 24 : ℝ, 26 : ℝ)

#eval sides_of_triangle.1 + sides_of_triangle.2 + sides_of_triangle.3 -- should equal 60

end right_triangle_with_perimeter_l615_615718


namespace trains_meet_in_time_l615_615617

noncomputable def time_to_meet (length1 length2 distance_between speed1 speed2 : ℝ) : ℝ :=
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := distance_between + length1 + length2
  total_distance / relative_speed

theorem trains_meet_in_time :
  time_to_meet 150 250 850 110 130 = 18.75 :=
by 
  -- here would go the proof steps, but since we are not required,
  sorry

end trains_meet_in_time_l615_615617


namespace final_grey_cats_l615_615400

def initially_total_cats : Nat := 16
def initial_white_cats : Nat := 2
def percent_black_cats : Nat := 25
def black_cats_left_fraction : Nat := 2
def new_white_cats : Nat := 2
def new_grey_cats : Nat := 1

/- We will calculate the number of grey cats after all specified events -/
theorem final_grey_cats :
  let total_cats := initially_total_cats
  let white_cats := initial_white_cats + new_white_cats
  let black_cats := (percent_black_cats * total_cats / 100) / black_cats_left_fraction
  let initial_grey_cats := total_cats - white_cats - black_cats
  let final_grey_cats := initial_grey_cats + new_grey_cats
  final_grey_cats = 11 := by
  sorry

end final_grey_cats_l615_615400


namespace ratio_of_areas_l615_615641

-- Define the original hexagon and cut-out shape areas and the conditions.
variable (H : Type) [RegularHexagon H] (cut_out_shape : H → ℝ)

-- Given conditions: points inside and on the perimeter divide the segments into quarters.
axiom divide_into_quarters (x : H) : divides_segments_into_quarters x

-- Goal: Prove the ratio of the area of the original hexagon to the cut-out shape is 4.
theorem ratio_of_areas (A_H : ℝ) (A_cut : ℝ) (hH : A_H = regular_hexagon_area H) 
  (h_cut : A_cut = cut_out_shape H) : A_H / A_cut = 4 := 
sorry

end ratio_of_areas_l615_615641


namespace det_E_eq_25_l615_615902

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end det_E_eq_25_l615_615902


namespace union_complement_eq_l615_615476

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615476


namespace exists_integer_n_tangent_l615_615737
open Real

noncomputable def degree_to_radian (d : ℝ) : ℝ :=
  d * (π / 180)

theorem exists_integer_n_tangent :
  ∃ (n : ℤ), -90 < (n : ℝ) ∧ (n : ℝ) < 90 ∧ tan (degree_to_radian (n : ℝ)) = tan (degree_to_radian 345) ∧ n = -15 :=
by
  sorry

end exists_integer_n_tangent_l615_615737


namespace min_value_frac_2_over_x_1_over_y_l615_615358

open Real

theorem min_value_frac_2_over_x_1_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) :
  ∃ l, (∀ (x y : ℝ), 0 < x → 0 < y → x + 2 * y = 2 → l ≤ (2 / x + 1 / y)) ∧ l = 4 :=
by
  use 4
  intros x y hx hy h
  sorry

end min_value_frac_2_over_x_1_over_y_l615_615358


namespace equivalent_resistance_is_15_l615_615759

-- Definitions based on conditions
def R : ℝ := 5 -- Resistance of each resistor in Ohms
def num_resistors : ℕ := 4

-- The equivalent resistance due to the short-circuit path removing one resistor
def simplified_circuit_resistance : ℝ := (num_resistors - 1) * R

-- The statement to prove
theorem equivalent_resistance_is_15 :
  simplified_circuit_resistance = 15 :=
by
  sorry

end equivalent_resistance_is_15_l615_615759


namespace prob_sin_ge_half_l615_615664

theorem prob_sin_ge_half : 
  let a := -Real.pi / 6
  let b := Real.pi / 2
  let p := (Real.pi / 2 - Real.pi / 6) / (Real.pi / 2 + Real.pi / 6)
  a ≤ b ∧ a = -Real.pi / 6 ∧ b = Real.pi / 2 → p = 1 / 2 :=
by
  sorry

end prob_sin_ge_half_l615_615664


namespace side_length_square_l615_615154

theorem side_length_square (A : ℝ) (s : ℝ) (h1 : A = 30) (h2 : A = s^2) : 5 < s ∧ s < 6 :=
by
  -- the proof would go here
  sorry

end side_length_square_l615_615154


namespace compare_abc_l615_615006

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615006


namespace sequence_probability_sum_l615_615672

theorem sequence_probability_sum :
  let a_n := fun (n : ℕ) ↦ match n with
  | 0     => 1
  | 1     => 2
  | (n+2) => a_n (n+1) + a_n n in
  let m := a_n 12,
      n := 4096 in
  Nat.gcd m n = 1 →
  m + n = 4473 := by
  sorry

end sequence_probability_sum_l615_615672


namespace compare_abc_l615_615044

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615044


namespace custom_op_example_l615_615391

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_example : (custom_op 7 4) - (custom_op 4 7) = -9 :=
by
  sorry

end custom_op_example_l615_615391


namespace polynomial_solution_count_l615_615705

theorem polynomial_solution_count :
  let Q := λ (a b c : ℕ), a * (1 : ℕ)^2 + b * (1 : ℕ) + c
  in ∃ (a b c : ℕ), Q a b c = 17 ∧
      a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
      (finset.card ((finset.filter (λ abc : ℕ × ℕ × ℕ, Q abc.1 abc.2.1 abc.2.2 = 17)
                                   (finset.product (finset.range 10)
                                                   (finset.product (finset.range 10) (finset.range 10))) : finset (ℕ × ℕ × ℕ))) = 63) :=
begin
  let Q := λ (a b c : ℕ), a * (1 : ℕ)^2 + b * (1 : ℕ) + c,
  use [0, 0, 0], -- placeholder for existential quantifiers
  split,
  { sorry }, -- proof that Q 0 0 0 = 17
  repeat { split },
  { sorry }, -- proof that 0 ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} 
  { sorry }, -- proof that 0 ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} 
  { sorry }, -- proof that 0 ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  { sorry }  -- proof that the number of valid solutions is 63
end

end polynomial_solution_count_l615_615705


namespace area_ratio_triangle_l615_615938

variable {V : Type _} [InnerProductSpace ℝ V]

open Classical

theorem area_ratio_triangle (A B C O : V) 
  (h1 : O ∈ triangle ABC)
  (h2 : (OA : V) + 2 • (OB : V) + 3 • (OC : V) = 0) :
  (area (triangle ABC)) / (area (triangle AOC)) = 3 := 
sorry

end area_ratio_triangle_l615_615938


namespace find_c_l615_615446

def f (x c : ℕ) : ℕ := x^2 + x + c

theorem find_c
  (h : ∀ c : ℤ, (f 1 c) * (f 3 c) = (f 2 c)^2) :
  c = 6 := 
sorry

end find_c_l615_615446


namespace find_subsets_not_equal_union_third_l615_615332

theorem find_subsets_not_equal_union_third (t : ℕ) (A : Fin t → Set α) :
  ∃ (S : Fin (⌊Real.sqrt (t : ℝ)⌋ : ℕ) → Set α), 
    ∀ (i j k : Fin (⌊Real.sqrt (t : ℝ)⌋ : ℕ)), 
    i ≠ j → S i ∪ S j ≠ S k :=
sorry

end find_subsets_not_equal_union_third_l615_615332


namespace num_sets_M_eq_4_l615_615172

theorem num_sets_M_eq_4 : 
  ({M : Set ℕ | {1, 2} ∪ M = {1, 2, 3}}).card = 4 :=
sorry

end num_sets_M_eq_4_l615_615172


namespace is_incenter_of_triangle_I_CEF_l615_615225

variable {P : Type*} [EuclideanGeometry P]

open EuclideanGeometry

-- Definitions of given conditions
variable (O A B C D I E F : P)
variable (circleO : Circle O)

axiom h1 : Diameter circleO B C

axiom h2 : OnCircle circleO A ∧ 0 < Angle A O B < 120

axiom h3 : Midpoint (minorArc circleO A B) D

axiom h4 : Parallel (LineThrough O parallelTo DA) (LineThrough I AC)

axiom h5 : Intersect (PerpendicularBisector OA) circleO = {E, F}

-- The proper theorem to prove the assertion
theorem is_incenter_of_triangle_I_CEF : Incenter (triangle C E F) I := by
  sorry

end is_incenter_of_triangle_I_CEF_l615_615225


namespace reciprocal_of_neg_one_third_l615_615991

theorem reciprocal_of_neg_one_third : 
  ∃ x : ℚ, (-1 / 3) * x = 1 :=
begin
  use -3,
  sorry
end

end reciprocal_of_neg_one_third_l615_615991


namespace circle_parallels_l615_615699

noncomputable theory
open_locale classical

variables {k : Type*} [field k] [char_zero k]

/-- Let ω₁ and ω₂ be circles with centers O₁ and O₂ respectively. 
Points C₁ and C₂ on these circles lie on the same side of the line O₁O₂. 
Ray O₁C₁ intersects ω₂ at points A₂ and B₂. 
Ray O₂C₂ intersects ω₁ at points A₁ and B₁. 
Prove that ∠A₁O₁B₁ = ∠A₂B₂C₂ if and only if C₁C₂ ∥ O₁O₂. -/
theorem circle_parallels
  (ω₁ ω₂ : set (k × k))
  (O₁ O₂ C₁ C₂ A₁ B₁ A₂ B₂ : k × k)
  (hO₁ : O₁ ∈ ω₁) (hO₂ : O₂ ∈ ω₂)
  (hC₁ : C₁ ∈ ω₁) (hC₂ : C₂ ∈ ω₂)
  (hA₁ : A₁ ∈ ω₁) (hB₁ : B₁ ∈ ω₁)
  (hA₂ : A₂ ∈ ω₂) (hB₂ : B₂ ∈ ω₂)
  (h_same_side : same_side O₁ O₂ C₁ C₂)
  (h_ray1 : ∃ (r₁ : k), O₁ + r₁ • (C₁ - O₁) = A₂ ∧ O₁ + r₁ • (C₁ - O₁) = B₂)
  (h_ray2 : ∃ (r₂ : k), O₂ + r₂ • (C₂ - O₂) = A₁ ∧ O₂ + r₂ • (C₂ - O₂) = B₁) :
  ∠(A₁ - O₁) (B₁ - O₁) = ∠(A₂ - B₂) (C₂ - B₂) ↔ (C₁ - C₂) ∥ (O₁ - O₂) :=
sorry

end circle_parallels_l615_615699


namespace problem1_problem2_l615_615234

-- Problem 1: relating the roots and coefficients
theorem problem1 (a b c d x1 x2 x3 : ℝ) 
(h : a ≠ 0) 
(h_eq : a * (x - x1) * (x - x2) * (x - x3) = a * x^3 + b * x^2 + c * x + d) : 
  x1 + x2 + x3 = -b / a ∧ x1 * x2 + x1 * x3 + x2 * x3 = c / a ∧ x1 * x2 * x3 = -d / a :=
sorry

-- Problem 2: finding the specific roots of the given cubic equation
theorem problem2 (a b c d : ℝ)
(h_eq : 8 * x^3 - 20 * x^2 - 10 * x + 33 = 0) :
  ∀ x1 x2 x3 : ℝ, 
  (x1 = 1.5 ∧ x2 = 0.5 + sqrt 3 ∧ x3 = 0.5 - sqrt 3) :=
sorry

end problem1_problem2_l615_615234


namespace zero_positive_integers_prime_polynomial_l615_615308

noncomputable def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem zero_positive_integers_prime_polynomial :
  ∀ (n : ℕ), ¬ is_prime (n^3 - 7 * n^2 + 16 * n - 12) :=
by
  sorry

end zero_positive_integers_prime_polynomial_l615_615308


namespace dodecagon_product_value_l615_615264

noncomputable def regular_dodecagon_product : ℂ := 
  let Q : Fin 12 → ℂ := λ i, 2 + exp (2 * i * π * I / 12) -- Roots of (z - 2)^12 = 1
  ∏ i, Q i
  
theorem dodecagon_product_value : regular_dodecagon_product = 4095 := 
  sorry

end dodecagon_product_value_l615_615264


namespace days_of_supply_l615_615696

-- Define the conditions as Lean definitions
def visits_per_day : ℕ := 3
def squares_per_visit : ℕ := 5
def total_rolls : ℕ := 1000
def squares_per_roll : ℕ := 300

-- Define the daily usage calculation
def daily_usage : ℕ := squares_per_visit * visits_per_day

-- Define the total squares calculation
def total_squares : ℕ := total_rolls * squares_per_roll

-- Define the proof statement for the number of days Bill's supply will last
theorem days_of_supply : (total_squares / daily_usage) = 20000 :=
by
  -- Placeholder for the actual proof, which is not required per instructions
  sorry

end days_of_supply_l615_615696


namespace percentage_of_hawks_is_30_l615_615401

-- Definitions from conditions
variables (total_birds : ℝ) (H P K : ℝ)

-- 40% of non-hawks are paddyfield-warblers
def percentage_paddyfield_warblers := 0.4 * (total_birds - H)

-- 25% as many kingfishers as paddyfield-warblers
def percentage_kingfishers := 0.25 * percentage_paddyfield_warblers

-- 35% of the birds are not hawks, paddyfield-warblers, or kingfishers
def non_hawks_paddyfield_kingfishers := 0.35 * total_birds

-- Total percentage equation
def total_percentage_hawks_paddyfield_kingfishers :=
  H + percentage_paddyfield_warblers + percentage_kingfishers

-- The proof goal
theorem percentage_of_hawks_is_30 :
  total_percentage_hawks_paddyfield_kingfishers = 0.65 * total_birds → 
  H = 0.30 * total_birds :=
by sorry

end percentage_of_hawks_is_30_l615_615401


namespace hamburgers_left_over_l615_615266

theorem hamburgers_left_over (made served : ℕ) (h_made : made = 9) (h_served: served = 3) : made - served = 6 :=
by
  rw [h_made, h_served]
  rfl

end hamburgers_left_over_l615_615266


namespace union_with_complement_l615_615520

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615520


namespace union_complement_eq_l615_615499

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615499


namespace binom_n_plus_one_n_l615_615200

theorem binom_n_plus_one_n (n : ℕ) (h : 0 < n) : Nat.choose (n + 1) n = n + 1 := 
sorry

end binom_n_plus_one_n_l615_615200


namespace union_with_complement_l615_615515

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615515


namespace combined_wave_amplitude_l615_615863

theorem combined_wave_amplitude :
  ∀ (t : ℝ), 
  let y1 := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t),
      y2 := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4) in
  let y := y1 + y2 in
  ∃ A, A = 3 * Real.sqrt 5 ∧ ∀ t, y = A * Real.sin (100 * Real.pi * t - Real.pi / 4) :=
by sorry

end combined_wave_amplitude_l615_615863


namespace committee_members_greater_than_60_impossible_committees_more_than_30_l615_615221

-- Part (a)
theorem committee_members_greater_than_60 (N : ℕ) :
  ∀ (m : ℕ), (m = 40) →
  ∀ (p : ℕ), (p = 10) →
  (∃ (S : set (ℕ × ℕ)), (∀ (i j : ℕ), i ≠ j → (S i).pairwise_disjoint ⋂ (S j).pairwise_disjoint)
  → N > 60) :=
sorry

-- Part (b)
theorem impossible_committees_more_than_30 :
  ∀ (pe : ℕ), (pe = 25) →
  ∀ (k : ℕ), (∀ (comms : set (finset ℕ))),
  (∀ (c1 c2 : finset ℕ), (c1 ≠ c2 → (c1 ∩ c2).card ≤ 1)) →
  k ≤ 30 :=
sorry

end committee_members_greater_than_60_impossible_committees_more_than_30_l615_615221


namespace graphs_intersection_l615_615937

theorem graphs_intersection 
  (a b c d x y : ℝ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) 
  (h1: y = ax^2 + bx + c) 
  (h2: y = ax^2 - bx + c + d) 
  : x = d / (2 * b) ∧ y = (a * d^2) / (4 * b^2) + d / 2 + c := 
sorry

end graphs_intersection_l615_615937


namespace cos_angle_BAD_l615_615423

theorem cos_angle_BAD (A B C D : Type) [euclidean_space] 
  (hAB : dist A B = 3) 
  (hAC : dist A C = 6) 
  (hBC : dist B C = 8) 
  (hAD_bisects_BAC : bisects A D B C) :
  cos (angle A D B) = sqrt 34 / 12 :=
by sorry

end cos_angle_BAD_l615_615423


namespace example_of_centrally_symmetric_shapes_l615_615283

-- Define the concept of central symmetry
def centrally_symmetric (s : Type) [Shape s] : Prop := 
  ∃ O : Point, ∀ P : Point, P ∈ s → (2 * O - P) ∈ s

-- Declare the circle and parallelogram as examples of learned shapes
axiom Circle : Type
axiom Parallelogram : Type
class Shape (s : Type)

-- Assume shapes learned in junior high school to be centrally symmetric
instance : Shape Circle := sorry
instance : Shape Parallelogram := sorry

-- Define that Circle and Parallelogram are centrally symmetric
axiom CentralSymmetryCircle : centrally_symmetric Circle
axiom CentralSymmetryParallelogram : centrally_symmetric Parallelogram

-- Prove that Circle and Parallelogram are centrally symmetric
theorem example_of_centrally_symmetric_shapes :
  ∀ s, (s = Circle ∨ s = Parallelogram) → centrally_symmetric s :=
by 
  intro s 
  intro hs
  cases hs with hc hp
  · exact CentralSymmetryCircle
  · exact CentralSymmetryParallelogram

end example_of_centrally_symmetric_shapes_l615_615283


namespace corrected_mean_is_99_375_l615_615984

-- Define the initial conditions
def mean := 100
def num_observations := 40
def incorrect_observation := 75
def correct_observation := 50

-- Define the total sum based on the incorrect observation
def incorrect_total_sum := mean * num_observations

-- Correct the total sum by removing the wrong observation and adding the correct one
def corrected_total_sum := incorrect_total_sum - incorrect_observation + correct_observation

-- Prove that the corrected mean is 99.375
theorem corrected_mean_is_99_375 : corrected_total_sum / num_observations = 99.375 := by
  sorry

end corrected_mean_is_99_375_l615_615984


namespace coeff_x4_in_P_l615_615710

def P : Polynomial ℝ := 2 * (Polynomial.X ^ 4 - 2 * Polynomial.X ^ 3 + 3 * Polynomial.X ^ 2) 
                     + 4 * (2 * Polynomial.X ^ 4 + Polynomial.X ^ 3 - Polynomial.X ^ 2 + 2 * Polynomial.X ^ 5) 
                     - 7 * (3 + 2 * Polynomial.X ^ 2 - 5 * Polynomial.X ^ 4)

theorem coeff_x4_in_P : Polynomial.coeff P 4 = 45 := sorry

end coeff_x4_in_P_l615_615710


namespace marcus_scored_50_percent_l615_615536

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l615_615536


namespace find_coefficient_a3_l615_615228

/-- Given the expansion of (1 + 2 * sqrt(x))^n and that the sum of coefficients is 243,
    and the expansion of x^(2n) in powers of (x+1), prove a_3 is -120. -/
theorem find_coefficient_a3 (x : ℝ) (n : ℕ) 
  (h1 : (1 + 2 * real.sqrt x)^n = 243)
  (h2 : x^(2*n) = ∑ i in finset.range (2 * n + 1), (λ i, a i * (x + 1)^i)) :
  a 3 = -120 :=
sorry

end find_coefficient_a3_l615_615228


namespace sample_size_l615_615241

theorem sample_size (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : 8 = n * 2 / 10) : n = 40 :=
by
  sorry

end sample_size_l615_615241


namespace compare_a_b_c_l615_615038

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615038


namespace polynomial_q_evaluation_at_1_l615_615176

theorem polynomial_q_evaluation_at_1 :
  ∀ (r s : ℝ),
    r + s = 3 →
    r * s = 1 →
    let q := λ x, x^2 - 18*x + 1 in
    q 1 = -16 :=
by
  intros r s h1 h2
  let q := λ x, x^2 - 18*x + 1
  show q 1 = -16
  sorry

end polynomial_q_evaluation_at_1_l615_615176


namespace optimal_play_winner_l615_615853

theorem optimal_play_winner (n : ℕ) (h : n > 1) :
  (∀ strategies : (Σ (turn : ℕ), (ℕ → ℝ × ℝ) → (ℝ × ℝ)), -- assuming all possible strategies
    (Σ (sum_vectors : ℝ × ℝ),
      (∃ player_1_move, player_1_move.1 = turn ∧ player_1_move.2 (turn + 1) = sum_vectors ∧ sum_vectors ≠ (0, 0)))
    ∨
    (Σ (sum_vectors : ℝ × ℝ),
      (∃ player_2_move, player_2_move.1 = turn + 1 ∧ player_2_move.2 (turn + 1) = sum_vectors ∧ sum_vectors = (0, 0))))
    → 
  ∃ strategy_for_player_1 : (ℕ → ℝ × ℝ) → (ℝ × ℝ), 
    (∀ moves_so_far : ℕ,
      let sum_vectors := ∑ i in finset.range moves_so_far, strategy_for_player_1 i in
        sum_vectors ≠ (0, 0)) :=
begin
  sorry
end

end optimal_play_winner_l615_615853


namespace max_good_permutations_correct_l615_615897

-- Define the concept of a "good" sequence of points in the plane.
def good_sequence (P : ℕ → ℂ) (n : ℕ) : Prop :=
  (∀ i j k : ℕ, i < j ∧ j < k → (P i ≠ P j ∧ P j ≠ P k ∧ P i ≠ P k)) ∧
  (∀ i : ℕ, i < n - 2 → counter_clockwise (P i) (P (i+1)) (P (i+2))) ∧
  (non_self_intersecting (λ i, P i) n)

-- Define the problem of finding the maximum number of permutations σ such that 
-- (P_σ(1), ..., P_σ(n)) is good.
def max_good_permutations (n : ℕ) (h : n ≥ 3) : ℕ :=
  (n^2 - 4*n + 6)

-- The main theorem statement.
theorem max_good_permutations_correct (n : ℕ) (h : n ≥ 3) : 
  ∃ (P : ℕ → ℂ), (∀ σ : ℕ → ℕ, σ_permutation σ n → good_sequence (λ i, P (σ i)) n) ↔ max_good_permutations n h :=
sorry

end max_good_permutations_correct_l615_615897


namespace relation_among_abc_l615_615096

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615096


namespace limit_Tn_cn_l615_615781

-- Definitions for sequences a_n and b_n
def a_seq (n : ℕ) : ℕ := n + 1
def b_seq (n : ℕ) : ℝ := (3:ℝ)^(n + 1) / 2

-- Definition for c_n and T_n
def c_seq (n : ℕ) : ℝ := a_seq n * b_seq n
def T_seq (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), c_seq i

-- Theorem statement
theorem limit_Tn_cn : 
  (real.lim (λ n, T_seq n / c_seq n) = 3 / 2) :=
begin
  sorry,
end

end limit_Tn_cn_l615_615781


namespace DM_parallel_BC_l615_615407

variable {Point : Type}
variable [affine_space Point]

structure Quadrilateral := 
(A B C D: Point)

def parallel (p1 p2: Point) : Prop := ∃ (v: Vec), v ≠ 0 ∧ p2 -ᵥ p1 = v

def Midpoint (a b pt: Point) : Prop := pt = (a +ᵥ b) / 2

def Area_eq (ABM ACD: ℝ) : Prop := ABM = ACD

def ABM (AB h2: ℝ) : ℝ := (1/4) * AB * h2

def ACD (CD h1: ℝ) : ℝ := (1/2) * CD * h1

noncomputable def h2 (h1: ℝ) : ℝ := h1 / 2

def equal_areas (AB CD h1: ℝ) : Prop := ABM AB (h2 h1) = ACD CD h1

theorem DM_parallel_BC (A B C D M: Point) (h1: ℝ) (AB CD: ℝ)
  (h_parallel: parallel A B C D)
  (h_midpoint: Midpoint A C M)
  (h_areas: equal_areas AB CD h1) :
  parallel D M B C :=
by sorry

end DM_parallel_BC_l615_615407


namespace quadratic_poly_unique_l615_615714

noncomputable def q (x : ℝ) : ℝ := (9 / 4) * x ^ 2 - (27 / 4) * x - 40.5

theorem quadratic_poly_unique :
  (q (-3) = 0) ∧ (q (6) = 0) ∧ (q (2) = -45) :=
by
  have h1 : q (-3) = 0,
  { sorry },
  have h2 : q (6) = 0,
  { sorry },
  have h3 : q (2) = -45,
  { sorry },
  exact ⟨h1, h2, h3⟩

end quadratic_poly_unique_l615_615714


namespace comparison_abc_l615_615080

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615080


namespace compare_a_b_c_l615_615011

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615011


namespace sum_of_integers_in_range_l615_615824

noncomputable def range_irrational_bound : Set ℤ := {n : ℤ | -Real.sqrt 3 < n ∧ n < 2.236}

theorem sum_of_integers_in_range :
  (∑ k in range_irrational_bound.to_finset, k) = 2 :=
by 
  sorry

end sum_of_integers_in_range_l615_615824


namespace mark_bananas_equals_mike_matt_fruits_l615_615690

theorem mark_bananas_equals_mike_matt_fruits :
  (∃ (bananas_mike matt_apples mark_bananas : ℕ),
    bananas_mike = 3 ∧
    matt_apples = 2 * bananas_mike ∧
    mark_bananas = 18 - (bananas_mike + matt_apples) ∧
    mark_bananas = (bananas_mike + matt_apples)) :=
sorry

end mark_bananas_equals_mike_matt_fruits_l615_615690


namespace number_of_satisfying_functions_l615_615106

def X := {f : Fin (2017 + 1) → Fin (2017 + 1) | true}

def d (f g : Fin (2017 + 1) → Fin (2017 + 1)) : ℤ :=
  Int.ofNat (Finset.min' (Finset.Icc 0 2016) (Finset.finite_Icc).some (λ i, max (f i) (g i))) -
  Int.ofNat (Finset.max' (Finset.Icc 0 2016) (Finset.finite_Icc).some (λ i, min (f i) (g i)))

def condition_met (f ∈ X) : Prop :=
  ∀ g ∈ X, d f g ≤ 2015 ∧ ∃ g ∈ X, d f g = 2015

theorem number_of_satisfying_functions : ∃ (n : ℕ), 
  n = 2 * (3 ^ (2017 + 1) - 2 ^ (2017 + 1)) ∧
  cardinality {f ∈ X | condition_met f} = n := sorry

end number_of_satisfying_functions_l615_615106


namespace find_f_neg2_l615_615099

-- Condition (1): f is an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Condition (2): f(x) = x^2 + 1 for x > 0
def function_defined_for_positive_x {f : ℝ → ℝ} (h_even : even_function f): Prop :=
  ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Proof problem: prove that given the conditions, f(-2) = 5
theorem find_f_neg2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_pos : function_defined_for_positive_x h_even) : 
  f (-2) = 5 := 
sorry

end find_f_neg2_l615_615099


namespace hours_worked_Sep_to_Feb_l615_615564

-- Define the conditions
variables (earnings_Mar_to_Aug : ℕ) (hours_Mar_to_Aug : ℕ) (cost_console : ℕ) 
          (spent_on_car : ℕ) (hours_needed_more : ℕ)

-- Conditions given in the problem
axioms 
  (h1 : earnings_Mar_to_Aug = 460) 
  (h2 : hours_Mar_to_Aug = 23)
  (h3 : cost_console = 600)
  (h4 : spent_on_car = 340)
  (h5 : hours_needed_more = 16)

-- Define the hourly rate based on conditions
noncomputable def hourly_rate : ℕ := earnings_Mar_to_Aug / hours_Mar_to_Aug

-- Define the remaining amount after spending on the car
noncomputable def remaining_amount : ℕ := earnings_Mar_to_Aug - spent_on_car

-- Define the amount needed to reach the cost of the console
noncomputable def amount_needed : ℕ := cost_console - remaining_amount

-- Define the total hours needed to earn the remaining amount needed for the console
noncomputable def total_hours_needed : ℕ := amount_needed / hourly_rate

-- Prove that the hours worked from September to February is 8
theorem hours_worked_Sep_to_Feb : total_hours_needed - hours_needed_more = 8 :=
by
  sorry

end hours_worked_Sep_to_Feb_l615_615564


namespace expected_value_correct_l615_615123

noncomputable def expected_value : ℝ :=
  let prob := (1:ℝ) / 8
  let values := [2, 3, 5, 7, 5, 0, -4, -4]
  (list.sum values) * prob + (-4) * prob

theorem expected_value_correct : expected_value = 1.75 := by
  sorry

end expected_value_correct_l615_615123


namespace solve_age_multiplier_l615_615543

def age_multiplier_proof : Prop :=
  ∃ x : ℕ, 
    let m := 41 in
    let j := 11 in
    m + j = 52 ∧
    m = x * j - 3 ∧
    x = 4

theorem solve_age_multiplier : age_multiplier_proof :=
by
  sorry

end solve_age_multiplier_l615_615543


namespace price_after_9_years_l615_615970

theorem price_after_9_years (initial_price : ℝ) (years : ℝ) (reduction_factor : ℝ) (current_price : ℝ) : 
  initial_price = current_price → 
  reduction_factor = (2/3) → 
  years = 9 → 
  current_price = 8100 → 
  increasing_factor_type (years/3) : 
  (initial_price * reduction_factor ^ (years / 3)) = 2400 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end price_after_9_years_l615_615970


namespace exist_points_C_D_l615_615346

variables {K : Type*} [Field K]
variables (S : Circle K)
variables (A B : Point K)
variables (α : RealAngle)

-- The statement
theorem exist_points_C_D (S : Circle K) (A B : Point K) (α : RealAngle) :
  ∃ (C D : Point K), C ∈ S.points ∧ D ∈ S.points ∧ LineThrough A C ∥ LineThrough B D ∧ AngleSubtended S C D = α := 
sorry

end exist_points_C_D_l615_615346


namespace det_E_eq_25_l615_615903

def E : Matrix (Fin 2) (Fin 2) ℝ := ![![5, 0], ![0, 5]]

theorem det_E_eq_25 : E.det = 25 := by
  sorry

end det_E_eq_25_l615_615903


namespace union_complement_eq_l615_615501

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615501


namespace tony_drive_time_l615_615194

theorem tony_drive_time (d1 d2 t1 t2 : ℝ) (h1 : d1 = 120) (h2 : t1 = 3) (h3 : d2 = 200) (h4 : t2 = d2 * (t1 / d1)) : t2 = 5 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4.symm

#check @tony_drive_time

end tony_drive_time_l615_615194


namespace trailing_zeros_150_factorials_l615_615441

def trailing_zeros_product_factorials (n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), ∑ i in ([1, 2, 3].to_finset), k / (5 ^ i)

theorem trailing_zeros_150_factorials :
  trailing_zeros_product_factorials 150 % 1000 = 703 :=
by sorry

end trailing_zeros_150_factorials_l615_615441


namespace sum_common_divisors_50_15_l615_615745

theorem sum_common_divisors_50_15 : 
  let divisors_50 := {d ∈ finset.range (50 + 1) | 50 % d = 0}
  let divisors_15 := {d ∈ finset.range (15 + 1) | 15 % d = 0}
  let common_divisors := divisors_50 ∩ divisors_15
  finset.sum common_divisors id = 6 := 
by
  sorry

end sum_common_divisors_50_15_l615_615745


namespace inequality_a_c_b_l615_615067

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615067


namespace obtuse_triangle_third_vertex_l615_615618

theorem obtuse_triangle_third_vertex (y : ℝ) (h : y > 0):
  let A := (2 : ℝ, 3 : ℝ)
  let B := (0 : ℝ, 0 : ℝ)
  let C := (0 : ℝ, y)
  let area := (1 / 2) * abs (2 * (3 - y))
  area = 36 →
  y = 39 :=
by
  intros A B C area h_area
  dunfold area
  sorry

end obtuse_triangle_third_vertex_l615_615618


namespace vector_field_lines_l615_615750

noncomputable def vector_lines : Prop :=
  ∃ (C_1 C_2 : ℝ), ∀ (x y z : ℝ), (9 * z^2 + 4 * y^2 = C_1) ∧ (x = C_2)

-- We state the proof goal as follows:
theorem vector_field_lines :
  ∀ (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ), 
    (∀ (x y z : ℝ), a (x, y, z) = (0, 9 * z, -4 * y)) →
    vector_lines :=
by
  intro a ha
  sorry

end vector_field_lines_l615_615750


namespace recommend_player_B_l615_615587

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l in
  (l.map (fun x => (x - m)^2)).sum / l.length

def shots_A : List ℝ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def shots_B : List ℝ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def mean_A : ℝ := mean shots_A
def mean_B : ℝ := mean shots_B

def variance_A : ℝ := variance shots_A
def variance_B : ℝ := variance shots_B

theorem recommend_player_B : mean_A = mean_B ∧ variance_A > variance_B → recommendation = "B" :=
by
  unfold mean variance
  sorry

end recommend_player_B_l615_615587


namespace sqrt_representable_l615_615325

open Real

def is_perfect_square (z : ℤ) : Prop :=
∃ n : ℤ, n * n = z

theorem sqrt_representable (A B : ℝ) :
  (∃ (x y : ℚ), sqrt (A + sqrt B) = sqrt x + sqrt y ∧
                 sqrt (A - sqrt B) = sqrt x - sqrt y) ↔
  is_perfect_square (A^2 - B) :=
sorry

end sqrt_representable_l615_615325


namespace perpendicular_bisector_of_AB_l615_615776

theorem perpendicular_bisector_of_AB (A B C D : Point)
  (hCA_CB : dist C A = dist C B)
  (hDA_DB : dist D A = dist D B) :
  is_perpendicular_bisector C D A B :=
begin
  sorry
end

end perpendicular_bisector_of_AB_l615_615776


namespace bicycle_final_price_l615_615651

theorem bicycle_final_price (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  original_price = 200 → first_discount = 0.40 → second_discount = 0.25 →
  let first_reduced_price := original_price * (1 - first_discount) in
  let final_price := first_reduced_price * (1 - second_discount) in
  final_price = 90 :=
by
  intros h_orig h_first_disc h_second_disc
  let first_reduced_price := original_price * (1 - first_discount)
  let final_price := first_reduced_price * (1 - second_discount)
  have h_first_price : first_reduced_price = 120 := by
    rw [h_orig, h_first_disc]
    norm_num
  have h_final_price : final_price = 90 := by
    rw [h_first_price, h_second_disc]
    norm_num
  exact h_final_price

end bicycle_final_price_l615_615651


namespace problem_l615_615531

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615531


namespace find_f_of_x_result_f_x_l615_615802

noncomputable def f : ℝ → ℝ :=
  λ x, x^2 - 2*x + 2

theorem find_f_of_x (x : ℝ) : f (x - 1) = x^2 - 2 * x + 2 := sorry

theorem result_f_x (x : ℝ) : f x = x^2 + 1 := sorry

end find_f_of_x_result_f_x_l615_615802


namespace union_with_complement_l615_615516

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615516


namespace part1_part2_l615_615368

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x

theorem part1 (a b : ℝ) :
  (differentiable_at ℝ (f a) 1)
  ∧ (deriv (f a) 1 = 2 * a - 1)
  ∧ (tangent_line (f a) 1 = fun x => x + b) →
  (a = 1) ∧ (b = 0) :=
sorry

theorem part2 (a : ℝ) :
  (differentiable_on ℝ (f a) (Set.Icc (1 / Real.exp 1) Real.exp 1))
  ∧ (deriv (f a) 2 = 0)
  ∧ (a = 1 / 8) →
  (∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, 
    f a x ≤ f a (1 / Real.exp 1)) :=
sorry

end part1_part2_l615_615368


namespace min_value_l615_615762

theorem min_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 10) :
  ∃ (c : ℝ), (c = (1 / x + 2 / y)) ∧ c = (3 + 2 * Real.sqrt 2) / 10 :=
begin
  sorry
end

end min_value_l615_615762


namespace find_c_if_lines_parallel_l615_615207

theorem find_c_if_lines_parallel (c : ℝ) : 
  (∀ x : ℝ, 5 * x - 3 = (3 * c) * x + 1) → 
  c = 5 / 3 :=
by
  intro h
  sorry

end find_c_if_lines_parallel_l615_615207


namespace angle_P_is_90_degrees_l615_615424

open EuclideanGeometry

noncomputable def problem_triangle (P Q R S: Point) :=
  IsIsoscelesTriangle P Q R ∧
  (S ∈ Segment R P) ∧
  (IsAngleBisector Q S P R) ∧
  (distance Q S = distance Q R)

theorem angle_P_is_90_degrees (P Q R S : Point) 
  (h1 : IsIsoscelesTriangle P Q R)
  (h2 : S ∈ Segment R P)
  (h3 : IsAngleBisector Q S P R)
  (h4 : distance Q S = distance Q R) : 
  angle P ≤ \pi/2 := 
sorry

end angle_P_is_90_degrees_l615_615424


namespace compare_abc_l615_615026

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615026


namespace matrix_unique_solution_l615_615323

def matrix_mul (a b : Matrix (Fin 2) (Fin 1) ℝ) := 
  λ i j, (∑ k, a i k * b k j)

theorem matrix_unique_solution (N : Matrix (Fin 2) (Fin 2) ℝ) :
  (matrix_mul N ![![2], ![3]] = ![![1], ![5]]) ∧ 
  (matrix_mul N ![![4], ![-2]] = ![![10], ![-6]]) →
  N = ![![2, -1], ![-(1/3), 8/3]] :=
begin
  sorry,
end

end matrix_unique_solution_l615_615323


namespace find_C_share_l615_615140

-- Definitions
variable (A B C : ℝ)
variable (H1 : A + B + C = 585)
variable (H2 : 4 * A = 6 * B)
variable (H3 : 6 * B = 3 * C)

-- Problem statement
theorem find_C_share (A B C : ℝ) (H1 : A + B + C = 585) (H2 : 4 * A = 6 * B) (H3 : 6 * B = 3 * C) : C = 260 :=
by
  sorry

end find_C_share_l615_615140


namespace ak_cd_eq_kc_ad_l615_615119

variables (A B C D K : Type)
variables [convex_quadrilateral A B C D] [point_on_diagonal K B D]
variables [AB_eq_BC : A = B] [angle_sum_eq : angle AK B + angle B KC = angle A + angle C]

theorem ak_cd_eq_kc_ad : AK * CD = KC * AD :=
sorry

end ak_cd_eq_kc_ad_l615_615119


namespace martha_cards_l615_615121

theorem martha_cards :
  let initial_cards := 3
  let emily_cards := 25
  let alex_cards := 43
  let jenny_cards := 58
  let sam_cards := 14
  initial_cards + emily_cards + alex_cards + jenny_cards - sam_cards = 115 := 
by
  sorry

end martha_cards_l615_615121


namespace tan_alpha_value_l615_615354

theorem tan_alpha_value (α : ℝ) (h : sin α + 2 * cos α = sqrt 10 / 2) : tan α = -1 / 3 :=
sorry

end tan_alpha_value_l615_615354


namespace simplify_polynomial_l615_615146

noncomputable def P : ℕ → ℝ := 
  fun x => 2 * x^6 + 3 * x^5 + 2 * x^4 + 5 * x^2 + 16

noncomputable def Q : ℕ → ℝ := 
  fun x => x^6 + 4 * x^5 - 2 * x^3 + 3 * x^2 + 18

theorem simplify_polynomial :
  (fun x => P x - Q x) = (fun x => x^6 - x^5 + 2 * x^4 + 2 * x^3 + 2 * x^2 - 2) :=
by
  sorry

end simplify_polynomial_l615_615146


namespace ideal_pair_1_ideal_pair_2_not_ideal_pair_3_final_answer_l615_615779

variable (S T : Set ℝ)
variable (S_even T_int : Set ℤ)
variable (S_real T_complex : Set ℂ)

theorem ideal_pair_1 (S = {0}) (T = Set.univ : Set ℝ) :
  (∀ a b ∈ S, a - b ∈ S ∧ a * b ∈ S) →
  (∀ r ∈ S, ∀ n ∈ T, r * n ∈ S) →
  S ⊂ T := sorry

theorem ideal_pair_2 (S_even = {n | ∃ k, n = 2 * k}) (T_int = Set.univ : Set ℤ) :
  (∀ a b ∈ S_even, a - b ∈ S_even ∧ a * b ∈ S_even) →
  (∀ r ∈ S_even, ∀ n ∈ T_int, r * n ∈ S_even) →
  S_even ⊂ T_int := sorry

theorem not_ideal_pair_3 (S_real = Set.univ : Set ℝ) (T_complex = Set.univ : Set ℂ) :
  ¬ ((∀ a b ∈ S_real, a - b ∈ S_real ∧ a * b ∈ S_real) →
    (∀ r ∈ S_real, ∀ n ∈ T_complex, r * n ∈ S_real)) := sorry

theorem final_answer : "①②" := sorry

end ideal_pair_1_ideal_pair_2_not_ideal_pair_3_final_answer_l615_615779


namespace nested_function_stability_l615_615111

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ :=
  x^2 + p * x + q

theorem nested_function_stability (p q : ℝ)
  (h : ∀ x ∈ set.Icc (4 : ℝ) (6 : ℝ), abs (f x p q) ≤ 1 / 2) :
  ∀ n : ℕ, n ≥ 2017 →
  (Nat.iterate (λ x, f x p q) n ((9 - Real.sqrt 19) / 2)) = ((9 + Real.sqrt 19) / 2) :=
sorry

end nested_function_stability_l615_615111


namespace left_handed_rock_music_members_l615_615607

-- definitions based on the given conditions
variables 
  (num_people : ℕ)        -- total number of club members
  (num_left_handed : ℕ)   -- number of left-handed people
  (num_rock_music : ℕ)    -- number of people who like rock music
  (num_right_no_rock : ℕ) -- number of right-handed people who do not like rock music

-- given conditions
def conditions := num_people = 25 ∧ 
                  num_left_handed = 10 ∧ 
                  num_rock_music = 18 ∧ 
                  num_right_no_rock = 3 ∧
                  (∀ p, (p = "left" ∨ p = "right") ∧ ¬(p = "left" ∧ p = "right")) -- People are either left-handed or right-handed, but not both

-- target statement
def left_handed_rock_music_lovers (x : ℕ) : Prop :=
  num_people = (x + (num_left_handed - x) + (num_rock_music - x) + num_right_no_rock) ∧ x = 6

-- The theorem we need to prove
theorem left_handed_rock_music_members : ∃ x, conditions → left_handed_rock_music_lovers x :=
begin
  -- we only need the statement, no proof required
  sorry
end

end left_handed_rock_music_members_l615_615607


namespace solve_fisherman_problem_l615_615952

def fisherman_problem : Prop :=
  ∃ (x y z : ℕ), x + y + z = 16 ∧ 13 * x + 5 * y + 4 * z = 113 ∧ x = 5 ∧ y = 4 ∧ z = 7

theorem solve_fisherman_problem : fisherman_problem :=
sorry

end solve_fisherman_problem_l615_615952


namespace compare_abc_l615_615009

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615009


namespace alarm_prob_l615_615613

theorem alarm_prob (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.90) : 
  (1 - (1 - pA) * (1 - pB)) = 0.98 :=
by 
  sorry

end alarm_prob_l615_615613


namespace tetrahedron_volume_is_zero_l615_615842

noncomputable def volume_of_tetrahedron (p q r : ℝ) : ℝ :=
  (1 / 6) * p * q * r

theorem tetrahedron_volume_is_zero (p q r : ℝ)
  (hpq : p^2 + q^2 = 36)
  (hqr : q^2 + r^2 = 64)
  (hrp : r^2 + p^2 = 100) :
  volume_of_tetrahedron p q r = 0 := by
  sorry

end tetrahedron_volume_is_zero_l615_615842


namespace constant_term_is_40_l615_615182

noncomputable def expansion_sum (a : ℝ) : ℝ :=
  let x := 1
  in (x + a / x) * (2 * x - 1 / x)^5

theorem constant_term_is_40 (a : ℝ) (h : expansion_sum a = 2) :
  let x := 1
  in (x + a / x) * (2 * x - 1 / x)^5 = 1 + a :=
by
  have a_eq_1 : a = 1 :=
    by sorry
  have const_term : (x + a / x) * (2 * x - 1 / x)^5 = 40 :=
    by sorry
  exact const_term

end constant_term_is_40_l615_615182


namespace product_of_series_l615_615697

theorem product_of_series :
  (∏ n in finset.range(11).map (nat.add 3), (1 - 1 / (n^2 : ℝ))) = (4 / 13 : ℝ) :=
by
  sorry

end product_of_series_l615_615697


namespace snow_probability_l615_615847

theorem snow_probability :
  let p₁ := 1/2
  let p₂ := 2/3
  let p₃ := 3/4
  let p₄ := 4/5
  let p₅ := 5/6
  let p₆ := 7/8
  let p₇ := 7/8
  1 - (p₁ * p₂ * p₃ * p₄ * p₅ * p₆ * p₇) = 139 / 384 := by
sorry

end snow_probability_l615_615847


namespace sunflower_count_l615_615819

theorem sunflower_count (r l d : ℕ) (t : ℕ) (h1 : r + l + d = 40) (h2 : t = 160) : 
  t - (r + l + d) = 120 := by
  sorry

end sunflower_count_l615_615819


namespace compare_abc_l615_615052

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615052


namespace range_of_a_l615_615801

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x : ℝ, f x = x^2 + a * x + 1) :
  (∃ x₀ : ℝ, |f x₀| ≤ 1/4 ∧ |f (x₀ + 1)| ≤ 1/4) →
  a ∈ set.Icc (-real.sqrt 6) (-2) ∪ set.Icc 2 (real.sqrt 6) :=
sorry

end range_of_a_l615_615801


namespace cube_properties_l615_615870

theorem cube_properties
  (A B C D A1 B1 C1 D1 P : Type)
  [has_edge_length: ∀ (X Y : Type), Prop] -- Simplified type to represent edge lengths
  (h_edge_length : has_edge_length A B ∧ has_edge_length B C ∧ has_edge_length C D ∧ has_edge_length A1 B1 ∧ has_edge_length B1 C1 ∧ has_edge_length C1 D1)
  (edge_length_one: ∀ (X Y : Type), has_edge_length X Y → dist X Y = 1)
  (P_on_B1D1 : P ∈ segment B1 D1)
  
  : (sphere_surface_area (circumscribed_sphere (tetrahedron A1 B D C1)) = 3 * π)
∧ (volume (tetrahedron P A1 B D) = 1 / 6)
∧ (area (polygon_intersection (plane_through_point_parallel_to_plane P (plane A1 B D)) (cube A B C D A1 B1 C1 D1)) = sqrt 3 / 2)
∧ (range (sine_angle_between_line_and_plane (line P A1) (plane A1 B D)) = [√3/3, √6/3]) :=
begin
  sorry
end

end cube_properties_l615_615870


namespace richard_walked_first_day_l615_615947

theorem richard_walked_first_day (x : ℝ) (h70 : 70 = 34 + 36)
  (h1 : (70 - 36) = 34)
  (h2 : 34 = x + (1 / 2) * x - 6 + 10) :
  x = 20 :=
begin
  sorry
end

end richard_walked_first_day_l615_615947


namespace eighth_son_receives_184_jin_l615_615862

variable (a : ℕ) -- a_1, the amount received by the eldest son

theorem eighth_son_receives_184_jin
  (h : ∑ i in Finset.range 8, (a + 17 * i) = 996) :
  a + 17 * 7 = 184 := 
sorry

end eighth_son_receives_184_jin_l615_615862


namespace compare_a_b_c_l615_615017

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615017


namespace coeff_x60_eq_zero_l615_615735

theorem coeff_x60_eq_zero :
  let p := (List.range 10).map (λ n => X^(n + 1) - (n + 1))
  polynomial.degree (List.foldl (*) 1 p) = 55 →
  polynomial.coeff (List.foldl (*) 1 p) 60 = 0 :=
by
  intros p hp
  sorry

end coeff_x60_eq_zero_l615_615735


namespace morning_registration_count_l615_615129

variable (M : ℕ) -- Number of students registered for the morning session
variable (MorningAbsentees : ℕ := 3) -- Absentees in the morning session
variable (AfternoonRegistered : ℕ := 24) -- Students registered for the afternoon session
variable (AfternoonAbsentees : ℕ := 4) -- Absentees in the afternoon session

theorem morning_registration_count :
  (M - MorningAbsentees) + (AfternoonRegistered - AfternoonAbsentees) = 42 → M = 25 :=
by
  sorry

end morning_registration_count_l615_615129


namespace b_5_eq_neg_1054_l615_615813

-- Define the sequence a_n recursively
def a : ℕ+ → ℤ
| 1 := 2
| (n+1) := 3 - 2 * a n

-- Define b_n based on the given conditions
def b (n : ℕ+) : ℤ :=
2 * (a n) * (a (n+1))

-- Prove that b_5 equals -1054
theorem b_5_eq_neg_1054 : b 5 = -1054 := 
sorry

end b_5_eq_neg_1054_l615_615813


namespace sum_of_roots_l615_615994

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x)

theorem sum_of_roots : 
  (∑ x in (Icc (-2010 : ℝ) 2012).to_finset, if f x = g x then x else 0) = 4020 := 
sorry

end sum_of_roots_l615_615994


namespace contrapositive_example_l615_615590

theorem contrapositive_example 
  (x y : ℝ) (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_example_l615_615590


namespace find_ellipse_equation_find_max_area_ABCD_l615_615873

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  (∃ (x y : ℝ), x + y - 2 * real.sqrt 2 = 0 ∧ 
                real.sqrt ((x - 0)^2 + (y - 2 * real.sqrt 2)^2) = b ∧
                ((x^2) / (b^2) + (y^2) / (a^2) = 1) ∧
                b < a ∧ a > 0 ∧ b > 0) 

noncomputable def slope_of_og (x0 y0 : ℝ) (G : ℝ × ℝ) (a b : ℝ) : Prop :=
  (∃ x1 y1 x2 y2 : ℝ,
    G = (x0, y0) ∧ 
    (y2 - y1) / (x2 - x1) = -1 ∧ 
    x1 + x2 = 2*x0 ∧
    y1 + y2 = 2*y0 ∧
    (y0 / x0 = 9) ∧
    (y1^2/a^2 + x1^2/b^2 = 1) ∧ 
    (y2^2/a^2 + x2^2/b^2 = 1))

theorem find_ellipse_equation :
  (∃ (a b : ℝ), ellipse_equation a b ∧ (a^2 = 9 ∧ b^2 = 1)) ∧ slope_of_og 0 (2 * real.sqrt 2) (0, 2 * real.sqrt 2) 3 1 :=
sorry

noncomputable def perpendicular_AC_BD (xA yA xB yB xC yC xD yD : ℝ) : Prop :=
  (yC - yA) * (xD - xB) = -(xC - xA) * (yD - yB)

noncomputable def max_area_ABCD (a b : ℝ) : ℝ :=
  (λ k : ℝ, 9 * 2 / (1 + 9 * k^2))⁻¹ / (9 * k^2 + 64)

theorem find_max_area_ABCD :
  ∃ (a b xA yA xB yB xC yC xD yD : ℝ),
  ellipse_equation a b ∧ slope_of_og 0 (2 * real.sqrt 2) (0, 2 * real.sqrt 2) 3 1 ∧
  perpendicular_AC_BD xA yA xB yB xC yC xD yD ∧ 
  (a^2 = 9 ∧ b^2 = 1) ∧ 
  max_area_ABCD a b = 27 / 8 :=
sorry

end find_ellipse_equation_find_max_area_ABCD_l615_615873


namespace find_a_l615_615116

def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

theorem find_a (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = -1 := by
  sorry

end find_a_l615_615116


namespace sequence_a_n_property_l615_615456

theorem sequence_a_n_property : ∀ {a : ℕ → ℕ} {S : ℕ → ℕ}, 
  (∀ n, (n > 0 → ∑ i in finset.range n, 1 / (S i) = n / (n+1)) ∧ 
        (∀ n, S n = (n + 1) * n) ∧ 
        (∀ n, n > 0 → a n = S n - S (n - 1)))
  → (∀ n, a n = 2 * n) :=
by
  intros a S h
  sorry

end sequence_a_n_property_l615_615456


namespace union_with_complement_l615_615518

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615518


namespace _l615_615177

noncomputable def sequence_relation (x : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → x (i - 1) + 2 / x (i - 1) = 2 * x i + 1 / x i

noncomputable theorem max_x0_of_sequence :
  ∀ (x : ℕ → ℝ), 0 < x 0 ∧ (∀ i, 0 < x i) ∧ x 0 = x 1995 ∧ sequence_relation x 1995 → x 0 ≤ 2 ^ 997 :=
by
  have h : ∃ (y : ℝ), y = 2 ^ 997
  -- proof would go here
  sorry

end _l615_615177


namespace quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l615_615342

-- Prove the range of k for distinct real roots
theorem quadratic_distinct_real_roots (k: ℝ) (h: k ≠ 0) : 
  (40 * k + 16 > 0) ↔ (k > -2/5) := 
by sorry

-- Prove the solutions for the quadratic equation when k = 1
theorem quadratic_solutions_k_eq_1 (x: ℝ) : 
  (x^2 - 6*x - 5 = 0) ↔ 
  (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14) := 
by sorry

end quadratic_distinct_real_roots_quadratic_solutions_k_eq_1_l615_615342


namespace photo_counts_correct_l615_615120

open Real

-- Definitions based on the conditions from step a)
def animal_photos : ℕ := 20
def flower_photos : ℕ := 30 -- 1.5 * 20
def total_animal_flower_photos : ℕ := animal_photos + flower_photos
def scenery_abstract_photos_combined : ℕ := (4 / 10) * total_animal_flower_photos -- 40% of total_animal_flower_photos

def x : ℕ := scenery_abstract_photos_combined / 5
def scenery_photos : ℕ := 3 * x
def abstract_photos : ℕ := 2 * x
def total_photos : ℕ := animal_photos + flower_photos + scenery_photos + abstract_photos

-- The statement to prove
theorem photo_counts_correct :
  animal_photos = 20 ∧
  flower_photos = 30 ∧
  total_animal_flower_photos = 50 ∧
  scenery_abstract_photos_combined = 20 ∧
  scenery_photos = 12 ∧
  abstract_photos = 8 ∧
  total_photos = 70 :=
by
  sorry

end photo_counts_correct_l615_615120


namespace sum_of_smallest_and_largest_is_correct_l615_615330

-- Define the conditions
def digits : Set ℕ := {0, 3, 4, 8}

-- Define the smallest and largest valid four-digit number using the digits
def smallest_number : ℕ := 3048
def largest_number : ℕ := 8430

-- Define the sum of the smallest and largest numbers
def sum_of_numbers : ℕ := smallest_number + largest_number

-- The theorem to be proven
theorem sum_of_smallest_and_largest_is_correct : 
  sum_of_numbers = 11478 := 
by
  -- Proof omitted
  sorry

end sum_of_smallest_and_largest_is_correct_l615_615330


namespace largest_very_prime_is_373_l615_615923

def very_prime (n : ℕ) : Prop :=
  ∀ (d : ℕ), (d > 0 ∧ d ≤ Nat.digits 10 n).count (λ i, Nat.Prime (Nat.digit 10 n (i - 1))) > 0

theorem largest_very_prime_is_373 : ∀ n : ℕ, (very_prime n) → n ≤ 373 :=
sorry

end largest_very_prime_is_373_l615_615923


namespace inequality_a_c_b_l615_615074

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615074


namespace problem_statement_l615_615958

noncomputable def angle_BDC (α β γ θ : ℝ) : ℝ := 15

theorem problem_statement
  (A B C D : Type)
  [metric_space A] [point A B] [point A C] [point A D] [segment A B] [segment A C] [segment A D]
  (h_congruent : congruent A B C D)
  (h_AB_eq_AC : AB = AC)
  (h_AC_eq_AD : AC = AD)
  (h_angle_BAC : ∠ B A C = 30°):
  ∠ B D C = angle_BDC 30 75 150 15 :=
by
  sorry

end problem_statement_l615_615958


namespace smallest_number_meeting_both_conditions_l615_615785

theorem smallest_number_meeting_both_conditions :
  ∃ n, (n = 2019) ∧
    (∃ a b c d e f : ℕ,
      n = a^4 + b^4 + c^4 + d^4 + e^4 ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
      c ≠ d ∧ c ≠ e ∧
      d ≠ e ∧
      a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
    (∃ x y z u v w : ℕ,
      y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
      n = x + y + z + u + v + w) ∧
    (¬ ∃ m, m < 2019 ∧
      (∃ a b c d e f : ℕ,
        m = a^4 + b^4 + c^4 + d^4 + e^4 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
        c ≠ d ∧ c ≠ e ∧
        d ≠ e ∧
        a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) ∧
      (∃ x y z u v w : ℕ,
        y = x + 1 ∧ z = x + 2 ∧ u = x + 3 ∧ v = x + 4 ∧ w = x + 5 ∧
        m = x + y + z + u + v + w)) :=
by
  sorry

end smallest_number_meeting_both_conditions_l615_615785


namespace tank_capacity_l615_615256

variable (C : ℝ)

def leak_rate (C : ℝ) : ℝ :=
  C / 6

def inlet_rate : ℝ :=
  2.5 * 60

def net_emptying_rate (C : ℝ) : ℝ :=
  C / 8

theorem tank_capacity :
  inlet_rate - leak_rate C = net_emptying_rate C →
  C = 3600 / 7 :=
sorry

end tank_capacity_l615_615256


namespace compare_a_b_c_l615_615019

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615019


namespace area_of_polygon_formed_by_circle_and_hyperbolas_l615_615589

theorem area_of_polygon_formed_by_circle_and_hyperbolas (R : ℝ) (hR_pos : 0 < R) :
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = R^2 }
  let hyperbola1 := { p : ℝ × ℝ | p.1 * p.2 = 1 }
  let hyperbola2 := { p : ℝ × ℝ | p.1 * p.2 = -1 }
  ∃ polygon : set (ℝ × ℝ), 
    (∀ p ∈ polygon, p ∈ circle ∧ (p ∈ hyperbola1 ∨ p ∈ hyperbola2)) ∧ 
    is_regular_polygon polygon ∧ 
    area polygon = R^4 :=
sorry

end area_of_polygon_formed_by_circle_and_hyperbolas_l615_615589


namespace polygon_side_intersections_l615_615946

theorem polygon_side_intersections :
  let m6 := 6
  let m7 := 7
  let m8 := 8
  let m9 := 9
  let pairs := [(m6, m7), (m6, m8), (m6, m9), (m7, m8), (m7, m9), (m8, m9)]
  let count_intersections (m n : ℕ) : ℕ := 2 * min m n
  let total_intersections := pairs.foldl (fun total pair => total + count_intersections pair.1 pair.2) 0
  total_intersections = 80 :=
by
  sorry

end polygon_side_intersections_l615_615946


namespace KLMN_is_parallelogram_l615_615440

-- Definitions based on the given conditions
variables {A B C D E F K L M N : Type}

-- Conditions that E and F are the midpoints of AB and CD respectively
def midpoint_AB (A B E : Point) : Prop := dist A E = dist E B
def midpoint_CD (C D F : Point) : Prop := dist C F = dist F D

-- Conditions that K,L,M,N are the midpoints of AF, CE, BF, DE respectively
def midpoint_AF (A F K : Point) : Prop := dist A K = dist K F
def midpoint_CE (C E L : Point) : Prop := dist C L = dist L E
def midpoint_BF (B F M : Point) : Prop := dist B M = dist M F
def midpoint_DE (D E N : Point) : Prop := dist D N = dist N E

-- Prove that KLMN is a parallelogram.
theorem KLMN_is_parallelogram (A B C D E F K L M N : Point)
  (h1 : midpoint_AB A B E)
  (h2 : midpoint_CD C D F)
  (h3 : midpoint_AF A F K)
  (h4 : midpoint_CE C E L)
  (h5 : midpoint_BF B F M)
  (h6 : midpoint_DE D E N) :
  parallelogram K L M N := sorry

end KLMN_is_parallelogram_l615_615440


namespace sqrt_of_sum_of_fractions_l615_615217

theorem sqrt_of_sum_of_fractions:
  sqrt ((25 / 36) + (16 / 9)) = sqrt 89 / 6 := by
    sorry 

end sqrt_of_sum_of_fractions_l615_615217


namespace tshirt_cost_is_100_l615_615294

def total_cost (num_tshirts num_pants tshirt_cost pant_cost) : Nat :=
  num_tshirts * tshirt_cost + num_pants * pant_cost

theorem tshirt_cost_is_100 :
  ∀ (tshirt_cost : Nat),
  (∃ (cost : Nat), total_cost 5 4 tshirt_cost 250 = 1500) →
  tshirt_cost = 100 :=
by
  intro tshirt_cost h
  cases h with cost h1
  have : 5 * tshirt_cost + 4 * 250 = 1500 := by assumption
  have : 5 * tshirt_cost + 1000 = 1500 := by rw [this, 4 * 250]
  have : 5 * tshirt_cost = 500 := by linarith
  have : tshirt_cost = 500 / 5 := by exact this
  have : tshirt_cost = 100 := by norm_num at this; exact this
  exact this

end tshirt_cost_is_100_l615_615294


namespace tax_revenue_collected_optimal_tax_rate_tax_revenue_target_l615_615640

noncomputable def market_supply_function (P: ℝ) : ℝ := 6 * P - 312

theorem tax_revenue_collected (P_s: ℝ) (t: ℝ) (Q_d: ℝ) : 
  P_s = 64 → t = 90 → Q_d = 688 - 4 * (P_s + t) → 
  Q_d * t = 6480 :=
by
  intros hP_s ht hQ_d
  have hP_d : ℝ := P_s + t
  have hQ_d_ : ℝ := 688 - 4 * hP_d
  rw [hP_s, ht] at hP_d hQ_d hQ_d_
  rw [hQ_d]
  have hT : 72 * 90 = 6480
  exact hT

theorem optimal_tax_rate (Q_d: ℝ) (t: ℝ): 
  (Q_d = 688 - 4 * (100 + 0.6 * t)) → 
  t = 60 :=
by
  intros hQ_d
  sorry -- solve the equation to show t = 60

theorem tax_revenue_target (t: ℝ) : 
  t = 60 → 
  288 * t - 2.4 * t * t = 8640 :=
by
  intros ht
  rw [ht]
  have hT_goal: 288 * 60 - 2.4 * 60 * 60 = 8640
  exact hT_goal

end tax_revenue_collected_optimal_tax_rate_tax_revenue_target_l615_615640


namespace point_in_fourth_quadrant_l615_615867

def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant :
  in_fourth_quadrant (1, -2) ∧
  ¬ in_fourth_quadrant (2, 1) ∧
  ¬ in_fourth_quadrant (-2, 1) ∧
  ¬ in_fourth_quadrant (-1, -3) :=
by
  sorry

end point_in_fourth_quadrant_l615_615867


namespace prove_f_decreasing_and_odd_l615_615231

noncomputable def decreasing_and_odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x1 x2 : ℝ, x1 > x2 → f(x1) < f(x2)) ∧
  (∀ x : ℝ, f(-x) = -f(x))

axiom f_property (f : ℝ → ℝ) :
  (∀ a b : ℝ, f(a + b) = f(a) + f(b)) ∧
  (f(0) = 0) ∧
  (∀ x1 x2 : ℝ, x1 > x2 → (x1 - x2 > 0 ∧ f(x1 - x2) < 0))

theorem prove_f_decreasing_and_odd (f : ℝ → ℝ) (h : f_property f) : decreasing_and_odd_function f :=
  sorry

end prove_f_decreasing_and_odd_l615_615231


namespace vector_proof_l615_615914

-- Define the points A, B, C, D, E and the vector operations
variables (A B C D E : Type) [AddGroup A] [Module ℝ A] (f : A → A)

-- Given that D is the midpoint of B and C
axiom midpoint_condition : f(D) = (f(B) + f(C)) / 2

-- Given that AB + AC = 4AE
axiom vector_condition : f(B) - f(A) + (f(C) - f(A)) = 4 * (f(E) - f(A))

-- Prove that AD = 2AE
theorem vector_proof : f(D) - f(A) = 2 * (f(E) - f(A)) :=
sorry

end vector_proof_l615_615914


namespace problem_a_problem_b_l615_615773

noncomputable def tetrahedron_conditions := ∀ (A B C D X Y Z T : Point)(triangleABC : triangle A B C)(triangleBCD : triangle B C D)(triangleCDA : triangle C D A)(triangleDAB : triangle D A B), 
(triangleABC.acute) ∧ 
(triangleBCD.acute) ∧
(triangleCDA.acute) ∧
(triangleDAB.acute) ∧
(on_segment X A B) ∧
(on_segment Y B C) ∧
(on_segment Z C D) ∧
(on_segment T D A)

theorem problem_a (A B C D X Y Z T : Point) (triangleABC : triangle A B C) (triangleBCD : triangle B C D) (triangleCDA : triangle C D A) (triangleDAB : triangle D A B) : 
  tetrahedron_conditions A B C D X Y Z T triangleABC triangleBCD triangleCDA triangleDAB →
  let σ := (angle D A B + angle B C D - angle A B C - angle C D A) in
  σ ≠ 0 → ¬ (∃ (min_length_path : broken_line XYZT), length minimal min_length_path) :=
begin
  sorry
end

theorem problem_b (A B C D X Y Z T : Point) (triangleABC : triangle A B C) (triangleBCD : triangle B C D) (triangleCDA : triangle C D A) (triangleDAB : triangle D A B) : 
  tetrahedron_conditions A B C D X Y Z T triangleABC triangleBCD triangleCDA triangleDAB →
  let σ := (angle D A B + angle B C D - angle A B C - angle C D A) in
  σ = 0 → (∃ (infinitely_many_paths : set (broken_line XYZT)), ∀ p ∈ infinitely_many_paths, length p = 2 * AC * (sin (angle B A C + angle C A D + angle D A B) / 2)) :=
begin
  sorry
end

end problem_a_problem_b_l615_615773


namespace find_x_l615_615417

-- Define the conditions
variables {AB DE : Type*} [HasZero AB] [HasZero DE]
variables (A B C : AB) (D E : DE)
variables (angle_ACE : ℝ) (x : ℝ)
variables (parallel : ∀ {l1 l2 : AB}, l1 = AB → l2 = DE → l1 ∥ l2)
variables (point_on_line : ∀ {l : AB} (P : l), P = C → P ∈ AB)
variables (transversal : ∀ {CE : Type*}, CE = line C E → CE ∩ DE ≠ ∅)

-- Define the given condition angle_ACE = 64°
def angle_ACE_eq_64 : Prop := angle_ACE = 64

-- Define the condition that alternate interior angles are equal
def alternate_interior_angles_equal : Prop :=
  ∀ {l1 l2 l3 : Type*}, l1 ∥ l2 → exists_transversal l1 l3 l2 → angle_ACE = x

-- State the theorem
theorem find_x (A B C D E : Type*) [HasZero A] [HasZero B]
  (parallel : ∀ {l1 l2 : A}, l1 = AB → l2 = DE → l1 ∥ l2)
  (point_on_line : ∀ {l : A} (P : l), P = C → P ∈ AB)
  (transversal : ∀ {CE : Type*}, CE = line C E → CE ∩ DE ≠ ∅)
  (angle_ACE_eq_64 : angle_ACE = 64) :
  x = 64 :=
by
  sorry

end find_x_l615_615417


namespace compare_abc_l615_615058

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615058


namespace megatech_astrophysics_degrees_l615_615635

theorem megatech_astrophysics_degrees :
  let microphotonics := 10
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let astrophysics_percentage := 100 - total_percentage
  let total_degrees := 360
  let astrophysics_degrees := (astrophysics_percentage / 100) * total_degrees
  astrophysics_degrees = 50.4 :=
by
  sorry

end megatech_astrophysics_degrees_l615_615635


namespace sum_of_odd_coeffs_eq_32_implies_a_eq_3_l615_615183

theorem sum_of_odd_coeffs_eq_32_implies_a_eq_3 (a : ℝ) :
  (∀ x : ℝ, (x = -1 ∨ x = 1) →
     ((a + x) * (1 + x)^4).coefficients.filter (λ n, n % 2 = 1) .sum = 32) →
  a = 3 :=
by
  sorry

end sum_of_odd_coeffs_eq_32_implies_a_eq_3_l615_615183


namespace range_of_a_l615_615922

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

def g (a : ℝ) (x : ℝ) : ℝ := (1/3)*x^3 - ((a+1)/2)*x^2 + a*x - (1/3)

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Icc (0:ℝ) 4, ∃ x₂ ∈ Icc (0:ℝ) 4, f x₁ = g a x₂) ↔ (1 < a ∧ a ≤ 2.25) ∨ (9 ≤ a) :=
sorry

end range_of_a_l615_615922


namespace original_number_is_24_l615_615259

def number_parts (x y original_number : ℝ) : Prop :=
  7 * x + 5 * y = 146 ∧ x = 13 ∧ original_number = x + y

theorem original_number_is_24 :
  ∃ (x y original_number : ℝ), number_parts x y original_number ∧ original_number = 24 :=
by
  sorry

end original_number_is_24_l615_615259


namespace solve_for_x_l615_615570

theorem solve_for_x :
  (sqrt (9 + sqrt (18 + 9 * 34)) + sqrt (3 + sqrt (3 + 34)) = 3 + 3 * sqrt 3) :=
  by {
    rw sqrt_add_sqrt_eq_3_mulsqrt_sqrt_3_34,
    sorry,
  }

-- The statement sqrt_add_sqrt_eq_3_mulsqrt_sqrt_3_34 should be built for the equation equivalence.
-- We do not require a proof for this theorem here, so it's kept with 'sorry'.
-- Note: actual function names and details would depend on the definitions provided in the actual Lean environment.

end solve_for_x_l615_615570


namespace problem_solved_l615_615716

theorem problem_solved : 
  1005 + (1 / 2) * (1004 + (1 / 2) * (1003 + ... + (1 / 2) * (3 + (1 / 2) * 2))) = 2008 := 
by
  sorry

end problem_solved_l615_615716


namespace maria_savings_percentage_l615_615691

theorem maria_savings_percentage
  (p : ℕ) (d_2 : ℚ) (d_a : ℚ) (n : ℕ)
  (hp : p = 60)
  (hd2 : d_2 = 0.30)
  (hda : d_a = 0.60)
  (hn : n = 5) : 
  let original_cost := n * p in
  let discounted_cost := p + (p * (1 - d_2)) + 3 * (p * (1 - d_a)) in
  let savings := original_cost - discounted_cost in
  (savings * 100) / original_cost = 42 :=
by
  sorry

end maria_savings_percentage_l615_615691


namespace union_complement_set_l615_615466

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615466


namespace max_g_sum_l615_615300

-- Define the conditions
def isNonIncreasing (f : ℕ → ℕ) : Prop :=
  ∀ ⦃n m : ℕ⦄, n ≤ m → f n ≥ f m

def sumLeq300 (f : ℕ → ℕ) : Prop :=
  (∑ i in Finset.range 301, f i) ≤ 300

def gLeqF (f g : ℕ → ℕ) : Prop :=
  ∀ (n : Fin 21 → ℕ), g (Finset.univ.sum n) ≤ (Finset.univ.sum (λ i, f (n i)))

-- Define the main problem
theorem max_g_sum
  (f g : ℕ → ℕ)
  (hf₁ : isNonIncreasing f)
  (hf₂ : sumLeq300 f)
  (hg : gLeqF f g) :
  (∑ i in Finset.range 6001, g i) ≤ 115440 :=
by
  sorry

end max_g_sum_l615_615300


namespace sandy_spent_correct_amount_l615_615565

-- Definitions
def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def shoes_price : ℝ := 8.50
def accessories_price : ℝ := 10.75
def discount_rate : ℝ := 0.10
def coupon_amount : ℝ := 5.00
def tax_rate : ℝ := 0.075

-- Sum of all items before discounts and coupons
def total_before_discount : ℝ :=
  shorts_price + shirt_price + jacket_price + shoes_price + accessories_price

-- Total after applying the discount
def total_after_discount : ℝ :=
  total_before_discount * (1 - discount_rate)

-- Total after applying the coupon
def total_after_coupon : ℝ :=
  total_after_discount - coupon_amount

-- Total after applying the tax
def total_after_tax : ℝ :=
  total_after_coupon * (1 + tax_rate)

-- Theorem assertion that total amount spent is equal to $45.72
theorem sandy_spent_correct_amount : total_after_tax = 45.72 := by
  sorry

end sandy_spent_correct_amount_l615_615565


namespace M_equals_N_l615_615859

variable (n : ℕ)
variable (a : Fin n → ℕ) -- Scores of n students
variable (b : Fin 100 → ℕ) -- Number of students who scored at least k points

-- Define M = ∑ a_i
def M : ℕ := ∑ i in Finset.range n, a ⟨i, sorry⟩ -- A valid uw housing range of i's definition
-- Define N = ∑ b_k
def N : ℕ := ∑ k in Finset.Icc 1 100, b ⟨k, sorry⟩ -- A valid Lean Icc range definition

theorem M_equals_N : M a = N b :=
sorry -- Proof is omitted

end M_equals_N_l615_615859


namespace operations_result_l615_615579

-- Define initial conditions: the starting number and the effective subtraction per operation
def initial_number : ℕ := 2100
def effective_subtraction : ℕ := 30

-- Define the total number of operations required to reach 0
def operations_to_zero (initial_number effective_subtraction : ℕ) : ℕ :=
  initial_number / effective_subtraction

-- Statement to prove that the result after the computed number of operations is indeed 0
theorem operations_result (initial_number effective_subtraction : ℕ) :
  initial_number % effective_subtraction = 0 → initial_number - operations_to_zero initial_number effective_subtraction * effective_subtraction = 0 :=
by
  intros h
  rw operations_to_zero
  exact Nat.div_mul_cancel h

-- Given the problem's specific conditions
example : operations_result initial_number effective_subtraction := by
  have h : initial_number % effective_subtraction = 0 := by
    norm_num [initial_number, effective_subtraction]
  exact operations_result initial_number effective_subtraction h

end operations_result_l615_615579


namespace correct_multiplication_l615_615934

noncomputable def a := 668
def approximate_result := 102325
def correct_result := 102357
def multiplier := 153

theorem correct_multiplication :
  ∃ a : ℕ, a * multiplier = correct_result :=
by
  use a
  have h : approximate_result - 2 * 10000 - 2 * 100 + approximate_result % 100 = 102195 := by sorry -- (illustrative computation simplification)
  have k : (∃ a : ℕ, a * multiplier = 102204 ∨ a * multiplier = 102357) := by sorry -- (from the mathematical derivation given in solution)
  exact k.2 -- Picking the correct result from the previously stated approximately equivalent options

end correct_multiplication_l615_615934


namespace union_complement_eq_l615_615503

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615503


namespace union_complement_l615_615494

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615494


namespace number_of_correct_propositions_is_zero_l615_615865

variable {α β l : Type}

def proposition1 (P Q : Type) [IncidencePlane α] : Prop := 
  ∀ P Q ∉ α, ¬ (∃! σ, σ ⊥ α ∧ P ∈ σ ∧ Q ∈ σ)

def proposition2 (P Q R : α) [IncidencePlane α] [IncidencePlane β] : Prop := 
  three_non_collinear_points P Q R →
  equal_distances_to_plane α [P, Q, R] →
  α ∥ β

def proposition3 (l : Type) [IncidencePlane α] : Prop := 
  (∀ p ∈ α, p ⊥ l) → l ⊥ α

def proposition4 (P Q : Type) [IncidencePlane α] : Prop := 
  skew_lines P Q →
  projection_in_same_plane P Q α →
  ¬ intersecting_lines (proj P α) (proj Q α)

theorem number_of_correct_propositions_is_zero :
  (¬ proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 ∧ ¬ proposition4) →
  number_of_correct_propositions = 0 :=
  by
  sorry

end number_of_correct_propositions_is_zero_l615_615865


namespace suzie_store_revenue_l615_615959

theorem suzie_store_revenue 
  (S B : ℝ) 
  (h1 : B = S + 15) 
  (h2 : 22 * S + 16 * B = 460) : 
  8 * S + 32 * B = 711.60 :=
by
  sorry

end suzie_store_revenue_l615_615959


namespace compare_abc_l615_615025

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615025


namespace factorial_div_sub_factorial_equality_l615_615624

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => (n + 1) * factorial n

theorem factorial_div_sub_factorial_equality :
  (factorial 12 - factorial 11) / factorial 10 = 121 :=
by
  sorry

end factorial_div_sub_factorial_equality_l615_615624


namespace average_of_second_set_of_two_numbers_l615_615585

theorem average_of_second_set_of_two_numbers
  (S : ℝ)
  (avg1 avg2 avg3 : ℝ)
  (h1 : S = 6 * 3.95)
  (h2 : avg1 = 3.4)
  (h3 : avg3 = 4.6) :
  (S - (2 * avg1) - (2 * avg3)) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_of_two_numbers_l615_615585


namespace number_of_valid_seven_digit_numbers_l615_615454

def seven_digit_numbers_divisible_by_7 (n : ℕ) (q r : ℕ) : Prop :=
  (1000000 ≤ n) ∧ (n < 10000000) ∧ (n = 50 * q + r) ∧ (10000 ≤ q) ∧ (q < 100000) ∧ (0 ≤ r) ∧ (r < 50) ∧ (q + r) % 7 = 0

theorem number_of_valid_seven_digit_numbers : ∃ n, (700000 : ℕ) = 720000 :=
by
  sorry

end number_of_valid_seven_digit_numbers_l615_615454


namespace geometric_seq_and_ineq_l615_615772

noncomputable theory

-- Define the sequence and its sum relation
def seq_and_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → 2 * S n = 3 * a n - 2 * n

-- Define the specific sequence a_n = 3^n - 1
def specific_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 3 ^ n - 1

-- Define the b sequence
def b_seq (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = a n + 2 ^ n + 1

-- Main theorem statement for question (Ⅰ) and (Ⅱ)
theorem geometric_seq_and_ineq (a S b : ℕ → ℕ)
  (h1: seq_and_sum a S) (h2: specific_seq a) (h3: b_seq a b) :
  (∀ n : ℕ, 0 < n → a n + 1 = 3 ^ (n - 1 + 1)) ∧
  (∀ n : ℕ, 0 < n → (∑ i in finset.range (n + 1), (1 : ℝ) / b (i + 1)) 
    < (1 : ℝ) / 2 - (1 : ℝ) / 2 ^ (n + 2)) :=
begin
  sorry -- proof part is omitted
end

end geometric_seq_and_ineq_l615_615772


namespace students_unassigned_l615_615187

theorem students_unassigned 
  (h_total : 37 + 31 + 25 + 35 = 128)
  (h_size_class_a : 37)
  (h_size_class_b : 31)
  (h_size_class_c : 25)
  (h_size_class_d : 35)
  (h_groups : 9) :
  128 % 9 = 2 :=
by
  have h : 128 = 37 + 31 + 25 + 35 := by rw [add_comm 37 31, add_comm 25 35, add_assoc, add_assoc, add_comm 31 25]
  simp [h]
  sorry

end students_unassigned_l615_615187


namespace Chrysler_Building_floors_l615_615150

variable (C L : ℕ)

theorem Chrysler_Building_floors :
  (C = L + 11) → (C + L = 35) → (C = 23) :=
by
  intro h1 h2
  sorry

end Chrysler_Building_floors_l615_615150


namespace point_not_similar_inflection_point_ln_l615_615394

noncomputable def similar_inflection_point (C : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
∃ (m : ℝ → ℝ), (∀ x, m x = (deriv C P.1) * (x - P.1) + P.2) ∧
  ∃ ε > 0, ∀ h : ℝ, |h| < ε → (C (P.1 + h) > m (P.1 + h) ∧ C (P.1 - h) < m (P.1 - h)) ∨ 
                     (C (P.1 + h) < m (P.1 + h) ∧ C (P.1 - h) > m (P.1 - h))

theorem point_not_similar_inflection_point_ln :
  ¬ similar_inflection_point (fun x => Real.log x) (1, 0) :=
sorry

end point_not_similar_inflection_point_ln_l615_615394


namespace compare_a_b_c_l615_615040

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615040


namespace compare_abc_l615_615024

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615024


namespace domain_f_l615_615711

def f (x : ℝ) : ℝ := sqrt (1 - x) / (x + 1)

theorem domain_f :
  { x : ℝ | 1 - x ≥ 0 } ∩ { x : ℝ | x ≠ -1 } = { x : ℝ | x < -1 } ∪ { x : ℝ | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end domain_f_l615_615711


namespace solution_for_x_l615_615748

theorem solution_for_x (x : ℝ) : 
  (∀ (y : ℝ), 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by 
  -- Proof should go here
  sorry

end solution_for_x_l615_615748


namespace probability_correct_l615_615235

noncomputable def probability_at_most_3_sixes : ℝ :=
  let p : ℝ := 1/6 in
  let q : ℝ := 5/6 in
  (Nat.choose 10 0) * q^10 +
  (Nat.choose 10 1) * p * q^9 +
  (Nat.choose 10 2) * p^2 * q^8 +
  (Nat.choose 10 3) * p^3 * q^7

-- Now we need to state the theorem to prove
theorem probability_correct : probability_at_most_3_sixes =
  (Nat.choose 10 0) * (5 / 6)^10 +
  (Nat.choose 10 1) * (1 / 6) * (5 / 6)^9 +
  (Nat.choose 10 2) * (1 / 6)^2 * (5 / 6)^8 +
  (Nat.choose 10 3) * (1 / 6)^3 * (5 / 6)^7 := by
  sorry

end probability_correct_l615_615235


namespace union_complement_eq_l615_615478

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615478


namespace minimum_sum_sets_l615_615437

noncomputable def S := ℕ
variables {n : ℕ} (A : fin n → S)
variable (h_n : n ≥ 2)
variable (h_symmetric_diff : ∀ i j : fin n, |A i ∆ A j| = |i - j|)

theorem minimum_sum_sets :
  n.even ∨ n.odd → ∑ i : fin n, |A i| = if n % 2 = 0 then n * n / 4 else (n * n - 1) / 4 :=
sorry

end minimum_sum_sets_l615_615437


namespace minimal_max_difference_l615_615554

noncomputable def min_max_adjacent_difference (n : ℕ) : ℕ :=
  n^2 + n + 1

theorem minimal_max_difference (n : ℕ) (cube : Finₓ(n) → Finₓ(n) → Finₓ(n) → ℕ) :
  (∀ i j k l m o, (i ≠ l ∨ j ≠ m ∨ k ≠ o) → cube i j k ≠ cube l m o) → -- All numbers are unique
  (∀ x y z, 1 ≤ cube x y z ∧ cube x y z ≤ n^3) → -- All numbers are within correct range
  ∃ path : list (Finₓ n × Finₓ n × Finₓ n),
    (∀ p ∈ path, cube (p.1) (p.2) (p.3) ∈ (1 : ℕ) .. n^3) ∧ -- All cube values along the path are in the given range
    (∀ p q ∈ path, (abs (cube (p.1) (p.2) (p.3) - cube (q.1) (q.2) (q.3)) ≤ n^2 + n + 1) ∧ -- Max difference condition
      ((p.1 = q.1 ∧ p.2 = q.2 ∧ abs (p.3 - q.3) = 1) ∨ -- Adjacent cell in z direction
       (p.1 = q.1 ∧ abs (p.2 - q.2) = 1 ∧ p.3 = q.3) ∨ -- Adjacent cell in y direction
       (abs (p.1 - q.1) = 1 ∧ p.2 = q.2 ∧ p.3 = q.3)) → -- Adjacent cell in x direction
    minimal_max_difference = n^2 + n + 1 :=
sorry -- Proof to be supplied

end minimal_max_difference_l615_615554


namespace find_point_C_coordinates_l615_615999

/-- Given vertices A and B of a triangle, and the centroid G of the triangle, 
prove the coordinates of the third vertex C. 
-/
theorem find_point_C_coordinates : 
  ∀ (x y : ℝ),
  let A := (2, 3)
  let B := (-4, -2)
  let G := (2, -1)
  (2 + -4 + x) / 3 = 2 →
  (3 + -2 + y) / 3 = -1 →
  (x, y) = (8, -4) :=
by
  intro x y A B G h1 h2
  sorry

end find_point_C_coordinates_l615_615999


namespace tetrahedron_side_length_for_tangent_spheres_l615_615336

theorem tetrahedron_side_length_for_tangent_spheres :
  ∃ S : ℝ, (S = 2 + 2 * sqrt 6) ∧
  (∀ (r : ℝ), r = 1 → 
   ∀ (T : Type) (triangle : T → T → T → Prop), 
     let tangent_spheres := ∀ A B C D,
       triangle A B C ∧ triangle A B D ∧
       triangle A C D ∧ triangle B C D ∧
       dist A B = 2 * r ∧ dist B C = 2 * r ∧
       dist C D = 2 * r ∧ dist D A = 2 * r ∧
       dist A C = 2 * r ∧ dist B D = 2 * r in
   tangent_spheres T) :=
begin
  sorry,
end

end tetrahedron_side_length_for_tangent_spheres_l615_615336


namespace find_a_l615_615378

noncomputable def A := { x | x^2 - 5 * x + 6 = 0 }
noncomputable def C := { x | x^2 + 2 * x - 8 = 0 }
noncomputable def B (a : ℝ) := { x | x^2 - a * x + 18 = 0 }

theorem find_a (a : ℝ) (A_inter_B_ne_empty : (A ∩ B a).Nonempty) (B_inter_C_empty : (B a ∩ C).Empty) :
  a = 9 :=
by
  sorry

end find_a_l615_615378


namespace major_premise_false_l615_615820

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Define the interval
def a : ℝ := -1
def b : ℝ := 1

theorem major_premise_false :
  (∀ x : ℝ, a < x ∧ x < b → (f x) ≠ (f x → f' x = 0)) := 
sorry

end major_premise_false_l615_615820


namespace weighted_mean_calculation_l615_615209

/-- Prove the weighted mean of the numbers 16, 28, and 45 with weights 2, 3, and 5 is 34.1 -/
theorem weighted_mean_calculation :
  let numbers := [16, 28, 45]
  let weights := [2, 3, 5]
  let total_weight := (2 + 3 + 5 : ℝ)
  let weighted_sum := ((16 * 2 + 28 * 3 + 45 * 5) : ℝ)
  (weighted_sum / total_weight) = 34.1 :=
by
  -- We only state the theorem without providing the proof
  sorry

end weighted_mean_calculation_l615_615209


namespace number_of_valid_sequences_l615_615385

theorem number_of_valid_sequences : 
  let letters := ['E', 'Q', 'U', 'A', 'T', 'I', 'O', 'N'],
      first_letter := 'M'
  ∃ (count : ℕ), count = 42 ∧ 
    (∀ seq : List Char, 
      seq.head = first_letter ∧ 
      ¬ seq.last = 'B' ∧ 
      seq.nodup) → count = 42 := 
by
  existsi 42
  sorry

end number_of_valid_sequences_l615_615385


namespace no_integer_solutions_l615_615917

def p (x : ℤ) : ℤ := x^4 + 6 * x^3 + 11 * x^2 + 13 * x + 37

theorem no_integer_solutions : ∀ x : ℤ, ¬ ∃ a : ℤ, p(x) = a^2 := by
  intro x
  sorry

end no_integer_solutions_l615_615917


namespace sum_of_cooler_right_triangle_areas_l615_615671

def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

def is_cooler (a b : ℕ) : Prop := (a * b) / 2 = 3 * (a + b)

def all_possible_cooler_areas (areas : Set ℕ) : ℕ :=
  areas.fold 0 (λ x sum, x + sum)

theorem sum_of_cooler_right_triangle_areas (areas : Set ℕ) :
  (∀ a b : ℕ, is_right_triangle a b (a * a + b * b).sqrt → is_cooler a b → (a * b) / 2 ∈ areas) →
  all_possible_cooler_areas areas = 471 :=
by
  sorry

end sum_of_cooler_right_triangle_areas_l615_615671


namespace bp_ge_br_bp_eq_br_equality_points_l615_615895

open_locale classical
open set

structure Square (A B C D P Q R: Type*) :=
  (is_square : (∀ x ∈ {A, B, C, D} → ∀ y ∈ {A, B, C, D}, x ≠ y → dist x y = dist A B)
               ∧ (angle A B P ≥ 60))
  (Q_description : ∃ (P inside: ∀ A B C D P: Type*, Line A D ∩ Perpendicular BP at P))
  (R_description :  ∃ (Q on: ∀ A B C D P Q: Type*, Line BQ ∩ Perpendicular BP from C))

theorem bp_ge_br (A B C D P Q R : Type*) (sq : Square A B C D P Q R) : 
   dist B P ≥ dist B R := sorry

theorem bp_eq_br_equality_points (A B C D P Q R : Type*) (sq : Square A B C D P Q R) :
   (dist B P = dist B R) ↔ (∃ (α : angle A B P = 60) (β : angle A B P = 60), is_equilateral_triangle A B P) := sorry

end bp_ge_br_bp_eq_br_equality_points_l615_615895


namespace marcus_percentage_of_team_points_l615_615539

theorem marcus_percentage_of_team_points 
  (marcus_3_point_goals : ℕ)
  (marcus_2_point_goals : ℕ)
  (team_total_points : ℕ)
  (h1 : marcus_3_point_goals = 5)
  (h2 : marcus_2_point_goals = 10)
  (h3 : team_total_points = 70) :
  (marcus_3_point_goals * 3 + marcus_2_point_goals * 2) / team_total_points * 100 = 50 := 
by
  sorry

end marcus_percentage_of_team_points_l615_615539


namespace value_of_a_l615_615458

noncomputable def a : ℕ := 4

def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a*a}
def C : Set ℕ := {0, 1, 2, 4, 16}

theorem value_of_a : A ∪ B = C → a = 4 := by
  intro h
  sorry

end value_of_a_l615_615458


namespace algebraic_expression_value_l615_615409

theorem algebraic_expression_value {m n : ℝ} 
  (h1 : n = m - 2022) 
  (h2 : m * n = -2022) : 
  (2022 / m) + ((m^2 - 2022 * m) / n) = 2022 := 
by sorry

end algebraic_expression_value_l615_615409


namespace sale_book_cost_l615_615928

variable (x : ℝ)

def fiveSaleBooksCost (x : ℝ) : ℝ :=
  5 * x

def onlineBooksCost : ℝ :=
  40

def bookstoreBooksCost : ℝ :=
  3 * 40

def totalCost (x : ℝ) : ℝ :=
  fiveSaleBooksCost x + onlineBooksCost + bookstoreBooksCost

theorem sale_book_cost :
  totalCost x = 210 → x = 10 := by
  sorry

end sale_book_cost_l615_615928


namespace union_complement_l615_615495

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615495


namespace Chrysler_Building_floors_l615_615151

variable (C L : ℕ)

theorem Chrysler_Building_floors :
  (C = L + 11) → (C + L = 35) → (C = 23) :=
by
  intro h1 h2
  sorry

end Chrysler_Building_floors_l615_615151


namespace exists_real_solution_real_solution_specific_values_l615_615950

theorem exists_real_solution (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

theorem real_solution_specific_values  (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

end exists_real_solution_real_solution_specific_values_l615_615950


namespace compare_abc_l615_615022

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615022


namespace approx_pi_using_right_triangle_l615_615642

theorem approx_pi_using_right_triangle (d : ℝ) : d * Real.sqrt 10 ≈ π * d :=
by
  sorry

end approx_pi_using_right_triangle_l615_615642


namespace sum_Tn_l615_615868

/-- Given sequences a_n and b_n and c_n derived from them, 
    prove that T_n equals the given formula for the sum of the first n terms of c_n. -/
open Nat

def a_n : ℕ → ℕ
| 0     => 0
| 1     => 1
| n + 2 => a_n n + 1

def S_n (n : ℕ) : ℕ := ∑ i in range (n + 1), a_n i

def b_n : ℕ → ℕ
| 0     => 0
| 1     => 1
| n + 2 => 2 * b_n (n + 1) + 1

def c_n (n : ℕ) : ℝ := (a_n n : ℝ) / (b_n n + 1 : ℝ)

def T_n (n : ℕ) : ℝ := ∑ i in range (n + 1), c_n i

theorem sum_Tn (n : ℕ) :
  T_n n = 2 - (1 / 2^(n - 1)) - (n / 2^n) := by sorry

end sum_Tn_l615_615868


namespace marcus_scored_50_percent_l615_615534

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end marcus_scored_50_percent_l615_615534


namespace pentagon_diagonals_l615_615439

-- Define the conditions
def AB : ℝ := 4
def CD : ℝ := 4
def BC : ℝ := 12
def DE : ℝ := 12
def AE : ℝ := 18

theorem pentagon_diagonals:
  let p := 1247 in
  let q := 9 in
  p + q = 128 := 
by
  -- assuming the computations are correct and skipping actual proof
  sorry

end pentagon_diagonals_l615_615439


namespace speed_of_train_is_correct_l615_615272

-- Define the lengths and time
def length_of_train : ℝ := 175
def length_of_platform : ℝ := 225.03
def time_to_cross_platform : ℝ := 40

-- The total distance covered by the train when crossing the platform
def total_distance : ℝ := length_of_train + length_of_platform

-- Speed in m/s
def speed_m_s : ℝ := total_distance / time_to_cross_platform

-- Conversion factor from m/s to km/h
def m_s_to_km_h : ℝ := 3.6

-- Speed in km/h
def speed_km_h : ℝ := speed_m_s * m_s_to_km_h

-- Theorem stating the speed of the train
theorem speed_of_train_is_correct : speed_km_h = 36.0027 := by
  sorry

end speed_of_train_is_correct_l615_615272


namespace union_with_complement_l615_615522

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615522


namespace simplify_expression_l615_615144

theorem simplify_expression :
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by sorry

end simplify_expression_l615_615144


namespace union_complement_eq_target_l615_615479

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615479


namespace compare_abc_l615_615047

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615047


namespace solve_z_6_eq_neg_8_l615_615743

open Complex

theorem solve_z_6_eq_neg_8 (x y : ℝ) (z : ℂ) (hx : z = x + Complex.I * y) :
  (z^6 = -8) → (z = Complex.I * Real.cbrt 2 ∨ z = -Complex.I * Real.cbrt 2) :=
by
  sorry

end solve_z_6_eq_neg_8_l615_615743


namespace intersection_M_N_l615_615395

def M : Set ℤ := {-1, 0, 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = Real.sin x}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end intersection_M_N_l615_615395


namespace octagon_area_l615_615270

theorem octagon_area (perimeter : ℝ) (h_perimeter : perimeter = 72) : 
  let side_length := perimeter / 4 in
  let segment_length := side_length / 3 in
  let area_square := side_length ^ 2 in
  let area_triangle := (1 / 2) * segment_length * segment_length in
  let total_area_removed := 4 * area_triangle in
  let area_octagon := area_square - total_area_removed in
  area_octagon = 252 := 
begin
  dsimp only [side_length, segment_length, area_square, area_triangle, total_area_removed, area_octagon],
  have h1 : side_length = 18 := by rw [h_perimeter, div_eq_mul_one_div, mul_one_div, mul_comm, mul_div_cancel_left, inv_eq_one_div],
  rw [h1],
  have h2 : segment_length = 6 := by rw [h1, div_eq_mul_one_div, mul_one_div, mul_comm, mul_div_cancel_left, inv_eq_one_div],
  rw [h2],
  have h3 : area_square = 18 ^ 2 := by rw [h1],
  rw [pow_two],
  have h4 : area_triangle = (1 / 2) * 6 * 6 := by rw [h2],
  rw [mul_assoc, mul_comm (1 / 2)],
  have h5 : 4 * ((1 / 2) * 6 * 6) = 72 := by ring,
  simp only [h5],
  simp only [h3, pow_two],
  linarith,
end

end octagon_area_l615_615270


namespace union_complement_eq_l615_615510

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615510


namespace ratio_of_square_side_lengths_l615_615677

-- Definitions based on problem conditions
def side_length_first_square := 2 * a
def side_length_second_square := 2 * b
def circle_radius := (Math.sqrt 2) * a
def distance_from_center := Math.sqrt ((a + 2 * b) ^ 2 + b ^ 2)

-- Theorem stating the ratio of the squares' side lengths
theorem ratio_of_square_side_lengths (a b : ℝ) (h : Math.sqrt ((a + 2 * b) ^ 2 + b ^ 2) = (Math.sqrt 2) * a) :
  a = 5 * b :=
sorry

end ratio_of_square_side_lengths_l615_615677


namespace john_spent_money_on_soap_l615_615888

def number_of_bars : ℕ := 20
def weight_per_bar : ℝ := 1.5
def cost_per_pound : ℝ := 0.5

theorem john_spent_money_on_soap :
  let total_weight := number_of_bars * weight_per_bar in
  let total_cost := total_weight * cost_per_pound in
  total_cost = 15 :=
by
  sorry

end john_spent_money_on_soap_l615_615888


namespace frog_final_position_1995_jumps_l615_615550

-- Define the points and the jumping rules
inductive Point : Type
| p1 | p2 | p3 | p4 | p5

open Point

-- Define the transition function for the frog’s jumps based on odd/even position
def jump (p : Point) : Point :=
  match p with
  | p1 => p2
  | p2 => p4
  | p3 => p4
  | p4 => p1
  | p5 => p1

-- Define the frog's journey starting from point 5
def frog_position_after_n_jumps (n : ℕ) : Point :=
  let jump_cycle := [p1, p2, p4]
  match List.nth jump_cycle (n % 3) with
  | some p => p
  | none   => p1

theorem frog_final_position_1995_jumps : frog_position_after_n_jumps 1995 = p4 :=
by
  sorry

end frog_final_position_1995_jumps_l615_615550


namespace min_b_minus_a_l615_615806

variables {a b : ℝ}

def f (x : ℝ) : ℝ := real.exp (2 * x)
def g (x : ℝ) : ℝ := real.log x + 0.5

theorem min_b_minus_a :
  (∃ (a : ℝ) (b ∈ set.Ioi 0), f a = g b ∧ b - a = 1 + (real.log 2) / 2) :=
sorry

end min_b_minus_a_l615_615806


namespace sequence_division_l615_615920

noncomputable def a : ℕ → ℕ
| 1 := 1
| 2 := 2009
| n := if h : n > 2 then
    let m := n - 1 in
    (a (m + 1) * a m + a m * a (m - 1)) / a (m - 1)
  else
    0

theorem sequence_division :
  (a 993) / (100 * a 991) = 89970 :=
sorry

end sequence_division_l615_615920


namespace phantom_additional_money_needed_l615_615553

theorem phantom_additional_money_needed
  (given_money : ℕ)
  (black_inks_cost : ℕ)
  (red_inks_cost : ℕ)
  (yellow_inks_cost : ℕ)
  (blue_inks_cost : ℕ)
  (total_money_needed : ℕ)
  (additional_money_needed : ℕ) :
  given_money = 50 →
  black_inks_cost = 3 * 12 →
  red_inks_cost = 4 * 16 →
  yellow_inks_cost = 3 * 14 →
  blue_inks_cost = 2 * 17 →
  total_money_needed = black_inks_cost + red_inks_cost + yellow_inks_cost + blue_inks_cost →
  additional_money_needed = total_money_needed - given_money →
  additional_money_needed = 126 :=
by
  intros h_given_money h_black h_red h_yellow h_blue h_total h_additional
  sorry

end phantom_additional_money_needed_l615_615553


namespace bahs_equals_500_yahs_l615_615828

theorem bahs_equals_500_yahs :
  (∀ (bah rah yah : ℝ),
  (20 * bah = 30 * rah) ∧
  (12 * rah = 20 * yah) →
  (500 * yah = 200 * bah)) :=
by
  intros bah rah yah h
  cases h with h1 h2
  sorry

end bahs_equals_500_yahs_l615_615828


namespace trigonometric_identity_sum_sum_constants_l615_615712

noncomputable def trigonometric_identity (x : ℝ) : Prop :=
  (cos x + cos (5 * x) + cos (11 * x) + cos (15 * x) = 4 * cos (8 * x) * cos (5 * x) * cos (2 * x))

theorem trigonometric_identity_sum : ∀ x : ℝ, trigonometric_identity x :=
by sorry

theorem sum_constants (a b c d : ℕ) (h₁ : a = 4) (h₂ : b = 8) (h₃ : c = 5) (h₄ : d = 2) : a + b + c + d = 19 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end trigonometric_identity_sum_sum_constants_l615_615712


namespace comparison_abc_l615_615079

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615079


namespace count_diff_squares_l615_615822

/-- 
  The number of integers between 1 and 1500 that can be expressed as the difference
  of the squares of two positive integers is 1125.
-/
theorem count_diff_squares (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1500) :
  (finset.filter (λ k, ∃ a b : ℕ, k = (a+1)^2 - a^2 ∨ k = (b+1)^2 - (b-1)^2) (finset.range 1501)).card = 1125 :=
sorry

end count_diff_squares_l615_615822


namespace complex_b_squared_eq_9975_l615_615248

theorem complex_b_squared_eq_9975 
  (a b : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_mod : complex.abs (a + b * complex.I) = 10)
  (h_eqdist : ∀ (z : ℂ), complex.abs (((a + b * complex.I) * z) - z) = complex.abs (((a + b * complex.I) * z))) :
  b^2 = 99.75 :=
sorry

end complex_b_squared_eq_9975_l615_615248


namespace inequality_holds_l615_615556

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) :=
sorry

end inequality_holds_l615_615556


namespace measure_of_angle_EBC_l615_615872

-- Definitions for important points and angles
variables {A B C D E : Type} -- Points
variables (angle_AED angle_DEC : ℝ)
variables (isosceles_AED equilateral_BEC : Prop)

-- Assumptions based on conditions
axiom point_E_on_segment_AB : E ∈ [A,B]
axiom isosceles_AED_defn : isosceles_AED := (triangle.isosceles A E D)
axiom equilateral_BEC_defn : equilateral_BEC := (triangle.equilateral B E C)
axiom angle_DEC_three_ADE : angle_DEC = 3 * (180 - 2 * angle_AED)
axiom angle_AED_value : angle_AED = 80

-- The statement we need to prove
theorem measure_of_angle_EBC : 
    ∀ (E ∈ [A,B]) (triangle.isosceles A E D) (triangle.equilateral B E C) 
    (angle_DEC = 3 * (180 - 2 * 80)) (angle_AED = 80), 
    ∠ EBC = 60 :=
by sorry

end measure_of_angle_EBC_l615_615872


namespace union_complement_eq_l615_615514

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615514


namespace dissimilar_terms_expansion_l615_615384

theorem dissimilar_terms_expansion : 
  let expression := (a + b + c + d) ^ 12
  let terms := λ (i j k l : ℕ), i + j + k + l = 12 ∧ 0 ≤ i ∧ 0 ≤ j ∧ 0 ≤ k ∧ 0 ≤ l
  let num_distinct_terms := (Nat.choose 15 3)
  num_distinct_terms = 455 :=
by 
  sorry

end dissimilar_terms_expansion_l615_615384


namespace equivalence_conditions_l615_615114

noncomputable theory
open MeasureTheory Classical Sequence

variables {α : Type*} {m : MeasurableSpace α} {μ : Measure α}
variables (ξ ξ_n : ℕ → α → ℝ) (p : ℝ)
variables [IsFiniteMeasure μ]

-- Conditions
def E_p_finite (n : ℕ) := ∫ (a : α), ∥ξ_n n a∥^p ∂μ < ∞
def conv_prob (pn : MeasureTheory.ProbabilitySpace α) := 
  ∀ᵐ a ∂μ, ∀ ε > 0, ∃ N, ∀ n ≥ N, ∥ξ_n n a - ξ a∥ < ε

-- Statements of equivalence
theorem equivalence_conditions :
  (1 : (tendsto (λ n, ∫ (a : α), ∥ξ_n n a∥^p ∂μ) at_top (𝓝 (∫ (a : α), ∥ξ a∥^p ∂μ)) < ∞) ↔ 
   (2 : (∃ l, tendsto (λ n, ∫ (a : α), ∥ξ_n n a∥^p ∂μ) at_top (𝓝 l) ∧ l ≤ ∫ (a : α), ∥ξ a∥^p ∂μ) < ∞) ↔ 
   (3 : UniformIntegrable (λ n, (λ a, ∥ξ_n n a∥^p)) at_top μ) ↔ 
   (4 : tendsto (λ n, ∫ (a : α), ∥ξ_n n a - ξ a∥^p ∂μ) at_top (𝓝 0))) :=
begin
  sorry
end

end equivalence_conditions_l615_615114


namespace max_value_expression_l615_615908

theorem max_value_expression (a b: ℝ) (ha: 0 < a) (hb: 0 < b) : 
  ∃ x: ℝ, ∀ y: ℝ, y = 2 * (a - x) * (x + real.sqrt (x^2 + b^2)) → y ≤ a^2 + b^2 :=
sorry

end max_value_expression_l615_615908


namespace union_complement_l615_615496

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615496


namespace boxes_count_l615_615720

theorem boxes_count (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) : (total_notebooks / notebooks_per_box) = 3 :=
by
  sorry

end boxes_count_l615_615720


namespace count_diff_squares_l615_615821

/-- 
  The number of integers between 1 and 1500 that can be expressed as the difference
  of the squares of two positive integers is 1125.
-/
theorem count_diff_squares (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1500) :
  (finset.filter (λ k, ∃ a b : ℕ, k = (a+1)^2 - a^2 ∨ k = (b+1)^2 - (b-1)^2) (finset.range 1501)).card = 1125 :=
sorry

end count_diff_squares_l615_615821


namespace cake_flour_amount_l615_615337

theorem cake_flour_amount (total_flour cakes_flour cupcakes_flour flour_per_cupcake cakes_price cupcakes_price revenue : ℝ) 
    (h1 : total_flour = 6)
    (h2 : cakes_flour = 4)
    (h3 : cupcakes_flour = 2)
    (h4 : flour_per_cupcake = 1/5)
    (h5 : cakes_price = 2.5)
    (h6 : cupcakes_price = 1)
    (h7 : revenue = 30) :
    let x := cakes_flour / ((revenue - cupcakes_price * (cupcakes_flour / flour_per_cupcake)) / cakes_price) in
    x = 1/2 :=
by {
  sorry,
}

end cake_flour_amount_l615_615337


namespace right_triangle_properties_l615_615856

-- Define the legs of the triangle
def leg1 : ℝ := 30
def leg2 : ℝ := 45

-- Define the area of the right triangle
def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

-- Define the Pythagorean theorem
def hypotenuse_length (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

-- Main theorem to be proved
theorem right_triangle_properties :
  triangle_area leg1 leg2 = 675 ∧ hypotenuse_length leg1 leg2 = 54 :=
by
  -- Skipping the proof
  sorry

end right_triangle_properties_l615_615856


namespace sum_of_all_positive_integers_nu_lcm_eq_45_l615_615206

theorem sum_of_all_positive_integers_nu_lcm_eq_45 :
  let νs := {ν | ν > 0 ∧ Nat.lcm ν 15 = 45} in
  (∑ ν in νs, ν) = 72 :=
by
  let νs := {ν | ν > 0 ∧ Nat.lcm ν 15 = 45}
  have : List.sum (νs.toList) = 72 := sorry
  exact this

end sum_of_all_positive_integers_nu_lcm_eq_45_l615_615206


namespace solve_D_l615_615412

-- Define the digits represented by each letter
variable (P M T D E : ℕ)

-- Each letter represents a different digit (0-9) and should be distinct
axiom distinct_digits : (P ≠ M) ∧ (P ≠ T) ∧ (P ≠ D) ∧ (P ≠ E) ∧ 
                        (M ≠ T) ∧ (M ≠ D) ∧ (M ≠ E) ∧ 
                        (T ≠ D) ∧ (T ≠ E) ∧ 
                        (D ≠ E)

-- Each letter is a digit from 0 to 9
axiom digit_range : 0 ≤ P ∧ P ≤ 9 ∧ 0 ≤ M ∧ M ≤ 9 ∧ 
                    0 ≤ T ∧ T ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 
                    0 ≤ E ∧ E ≤ 9

-- Each column sums to the digit below it, considering carry overs from right to left
axiom column1 : T + T + E = E ∨ T + T + E = 10 + E
axiom column2 : E + D + T + (if T + T + E = 10 + E then 1 else 0) = P
axiom column3 : P + M + (if E + D + T + (if T + T + E = 10 + E then 1 else 0) = 10 + P then 1 else 0) = M

-- Prove that D = 4 given the above conditions
theorem solve_D : D = 4 :=
by sorry

end solve_D_l615_615412


namespace problem1_problem2_problem3_problem4_l615_615315

-- Problem 1: Solutions to the equation x^2 - 4 = 0.
theorem problem1 : {x : ℝ | x^2 - 4 = 0} = {-2, 2} :=
sorry

-- Problem 2: Prime numbers that satisfy 0 < 2x < 18.
def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem problem2 : {x : ℕ | is_prime x ∧ 0 < 2 * x ∧ 2 * x < 18} = {2, 3, 5, 7} :=
sorry

-- Problem 3: The set of all even numbers.
theorem problem3 : {x : ℤ | ∃ n : ℤ, x = 2 * n} = {x | ∃ n : ℤ, x = 2 * n} :=
begin
  intros x,
  split;
  intro h;
  exact h,
end

-- Problem 4: The set of points in the fourth quadrant of the Cartesian coordinate plane.
theorem problem4 : {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0} = {(x, y) | x > 0 ∧ y < 0} :=
begin
  intros p,
  split;
  intro h;
  exact h,
end

end problem1_problem2_problem3_problem4_l615_615315


namespace right_triangle_AB_l615_615839

theorem right_triangle_AB (A B C : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (hypotenuse_AC : Real) (BC_ratio : Real) (AB_ratio : Real)
  (hypotenuse_AC_eq_39 : hypotenuse_AC = 39)
  (tan_B : BC_ratio / AB_ratio = 5 / 12) :
  let k := hypotenuse_AC / 13 in
  AB_ratio * k = 36 := 
by
  sorry

end right_triangle_AB_l615_615839


namespace equivalent_oranges_l615_615957

-- Given condition
def bananas_to_oranges : Prop :=
  (2/3) * 15 = 10 ∧ 10 = 12

-- The theorem statement
theorem equivalent_oranges : bananas_to_oranges → (1/4) * 20 * (12/10) = 6 :=
begin
  intro h,
  rw mul_assoc,
  have h1 : (1/4) * 20 = 5 := by norm_num,
  rw h1,
  have h2 : 5 * (12/10) = 6 := by norm_num,
  assumption h2
end

end equivalent_oranges_l615_615957


namespace chessboard_l_shaped_l_necessary_l615_615549

def is_l_shaped (square : list (list bool)) : Prop :=
  match square with
  | [[false, false], [false, true]] => true
  | [[false, false], [true, false]] => true
  | [[false, true], [false, false]] => true
  | [[true, false], [false, false]] => true
  | _ => false
  end

def has_l_shaped (board : list (list bool)) : Prop :=
  ∃ i j, i < 7 ∧ j < 7 ∧ is_l_shaped ([
    [board.nth i >>= list.nth j, board.nth i >>= list.nth (j + 1)],
    [board.nth (i + 1) >>= list.nth j, board.nth (i + 1) >>= list.nth (j + 1)]
  ])

theorem chessboard_l_shaped_l_necessary :
  ∀ (board : list (list bool)),
    (board.length = 8) →
    (∀ row : list bool, row ∈ board → row.length = 8) →
    (sum (list.map (fun row => sum (list.map (fun b => if b then 1 else 0) row)) board) = 31) →
    has_l_shaped board :=
by
  intros board board_len row_len shaded_squares
  sorry

end chessboard_l_shaped_l_necessary_l615_615549


namespace tangent_line_at_1_l615_615593

open Function

def f (x : ℝ) : ℝ := x^2

theorem tangent_line_at_1 :
  let f' := deriv f,
      f1 := f 1,
      f1_deriv := f' 1
  in f1 = 1 ∧ f1_deriv = 2 → (∀ x y : ℝ, y = 2 * (x - 1) + 1 ↔ 2 * x - y - 1 = 0) :=
by
  intros f' f1 f1_deriv h
  sorry

end tangent_line_at_1_l615_615593


namespace compare_abc_l615_615001

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615001


namespace max_modulus_of_z_l615_615827

open Complex

theorem max_modulus_of_z (z : ℂ) (h : |z + 3 + 4 * I| = 2) : |z| ≤ 7 :=
by
  sorry

end max_modulus_of_z_l615_615827


namespace hat_value_in_rice_l615_615850

variables (f l r h : ℚ)

theorem hat_value_in_rice :
  (4 * f = 3 * l) →
  (l = 5 * r) →
  (5 * f = 7 * h) →
  h = (75 / 28) * r :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end hat_value_in_rice_l615_615850


namespace distance_before_pie_l615_615881

variable (x : ℝ) -- distance driven before buying the pie

theorem distance_before_pie
  (total_distance : ℝ)
  (distance_after_pie : ℝ)
  (distance_left : ℝ)
  (h1 : total_distance = 78)
  (h2 : distance_after_pie = 18)
  (h3 : distance_left = 25)
  : x = 35 :=
by
  have h_total_driven_after_pie := h3 + h2
  have h_total_distance := h1
  have h_total_distance_correct := h_total_distance - h_total_driven_after_pie
  exact h_total_distance_correct
  sorry   -- proof steps are skipped

end distance_before_pie_l615_615881


namespace B_can_finish_in_15_days_l615_615653

variable (W : ℝ) (A B : ℝ)

-- Conditions from the problem
def A_work_rate := W / 18
def B_work_days := 10
def A_remaining_work_days := 6

-- Expected result
def B_completion_days := 15

theorem B_can_finish_in_15_days 
  (hB' : B = W / 15)
  (hA_rate : A = A_work_rate) 
  (hconditions : B_work_days * B + A_remaining_work_days * A_work_rate = W) : 
  B_completion_days = 15 := 
  by 
    -- Proof omitted
    sorry

end B_can_finish_in_15_days_l615_615653


namespace solve_for_x_l615_615572

def expression (x : ℝ) : ℝ := sqrt (9 + sqrt (18 + 9 * x)) + sqrt (3 + sqrt (3 + x))

theorem solve_for_x : 
  (expression x = 3 + 3 * sqrt 3) ↔ x = 3 :=
begin
  sorry
end

end solve_for_x_l615_615572


namespace alice_has_ball_after_three_turns_l615_615280

-- Given conditions
def p_Alice_to_Bob := 1 / 3
def p_Alice_keeps := 2 / 3
def p_Bob_to_Alice := 1 / 3
def p_Bob_keeps := 2 / 3

-- We need a definition for the initial state
def initial_state_Alice := 1 -- Alice starts with the ball

-- Main theorem to be proved
theorem alice_has_ball_after_three_turns :
  ∑ p in 
    { (p_Alice_keeps * p_Alice_keeps * p_Alice_keeps),
      (p_Alice_keeps * p_Alice_to_Bob * p_Bob_to_Alice),
      (p_Alice_to_Bob * p_Bob_to_Alice * p_Alice_keeps),
      (p_Alice_to_Bob * p_Bob_keeps * p_Alice_to_Bob) 
    }.toFinset, p  = 14 / 27 :=
by
  sorry

end alice_has_ball_after_three_turns_l615_615280


namespace area_of_PQRSUV_l615_615295

theorem area_of_PQRSUV (PQ QR UV SU TU RS : ℝ)
  (hPQ : PQ = 8) (hQR : QR = 10)
  (hUV : UV = 6) (hSU : SU = 3)
  (hTuUV : PQ = TU + UV) (hRSuSU : QR = RS + SU)
  (h_parallelogram_PQRT : true) (h_parallelogram_SUVU : true) :
  let area_PQRT := PQ * QR,
      area_SUVU := SU * UV
  in area_PQRSUV = area_PQRT - area_SUVU :=
begin
  sorry
end

end area_of_PQRSUV_l615_615295


namespace comparison_abc_l615_615083

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615083


namespace prime_p_p_plus_15_l615_615987

theorem prime_p_p_plus_15 (p : ℕ) (hp : Nat.Prime p) (hp15 : Nat.Prime (p + 15)) : p = 2 :=
sorry

end prime_p_p_plus_15_l615_615987


namespace parabola_problem_l615_615375

theorem parabola_problem (P F H : ℝ × ℝ) (x y : ℝ) 
    (on_parabola : y^2 = 4 * x)
    (focus_at : F = (1, 0))
    (directrix_intersects_x : H = (0, 0))
    (P_coords : P = (x, y))
    (distance_condition : dist P H = sqrt 2 * dist P F)
  : x = 1 := 
sorry

end parabola_problem_l615_615375


namespace total_yards_thrown_l615_615130

-- Definitions for the conditions
def distance_50_degrees : ℕ := 20
def distance_80_degrees : ℕ := distance_50_degrees * 2

def throws_on_saturday : ℕ := 20
def throws_on_sunday : ℕ := 30

def headwind_penalty : ℕ := 5
def tailwind_bonus : ℕ := 10

-- Theorem for the total yards thrown in two days
theorem total_yards_thrown :
  ((distance_50_degrees - headwind_penalty) * throws_on_saturday) + 
  ((distance_80_degrees + tailwind_bonus) * throws_on_sunday) = 1800 :=
by
  sorry

end total_yards_thrown_l615_615130


namespace problem_l615_615527

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615527


namespace sixteen_k_plus_eight_not_perfect_square_l615_615754

theorem sixteen_k_plus_eight_not_perfect_square (k : ℕ) (hk : 0 < k) : ¬ ∃ m : ℕ, (16 * k + 8) = m * m := sorry

end sixteen_k_plus_eight_not_perfect_square_l615_615754


namespace sum_bn_l615_615979

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc (0 : ℝ) (1 / 2) then 2 * x^2 
  else real.log x / real.log (1 / 4)

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℝ := n / 2014

-- Define the sequence {b_n}
def b (n : ℕ) : ℝ := f (a (n + 1)) - f (a n)

-- Prove that the sum of the first 2014 terms of the sequence {b_n} is 0
theorem sum_bn : ∑ i in finset.range 2014, b i = 0 :=
by
  sorry

end sum_bn_l615_615979


namespace collinear_lambda_l615_615793

noncomputable def collinear_vectors {α : Type*} [LinearOrderedField α] {v w : EuclideanSpace α (Fin 2)} : Prop :=
∃ (k : α) (h : k ≠ 0), k • v = w

theorem collinear_lambda (e₁ e₂ : EuclideanSpace ℝ (Fin 2)) (hne : e₁ ≠ e₂)
  (hne_clr : ¬ collinear_vectors e₁ e₂)
  (a := 2 • e₁ - e₂)
  (b := e₁ + λ_ • e₂) :
  collinear_vectors a b → λ_ = -1/2 := 
by
  sorry

end collinear_lambda_l615_615793


namespace pure_imaginary_m_zero_determine_z_l615_615787

theorem pure_imaginary_m_zero (m : ℝ) : 
  ∃ m, (m * (m - 1) = 0) ∧ (m - 1 ≠ 0) → m = 0 := 
by
  sorry

theorem determine_z (z : ℂ) :
  let z₁ : ℂ := - complex.I in
  ((3 + z₁) * z = 4 + 2 * complex.I) → z = 1 + complex.I :=
by
  sorry

end pure_imaginary_m_zero_determine_z_l615_615787


namespace smallest_number_divisible_by_18_70_100_84_increased_by_3_l615_615205

theorem smallest_number_divisible_by_18_70_100_84_increased_by_3 :
  ∃ n : ℕ, (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 84 = 0 ∧ n = 6297 :=
by
  sorry

end smallest_number_divisible_by_18_70_100_84_increased_by_3_l615_615205


namespace det_matrixE_l615_615904

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end det_matrixE_l615_615904


namespace compare_a_b_c_l615_615020

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615020


namespace problem_l615_615530

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615530


namespace cell_X_is_Red_l615_615568

-- Define the colours as an enumeration type
inductive Colour
| Red
| Blue
| Yellow
| Green

open Colour

-- Define a function representing the grid colouring constraint
def vertex_colour_constraint (grid : ℕ → ℕ → Colour) : Prop :=
  ∀ x y, (∀ dx dy, ((dx ≠ 0) ∨ (dy ≠ 0)) → grid x y ≠ grid (x + dx) (y + dy))

-- Assume a predefined partially coloured grid (not specified in the problem)
-- For example purpose, let's assume: 
-- grid 0 0 = Red, grid 0 1 = Blue, grid 1 0 = Green,
-- The implementation detail of the grid is abstracted away for simplicity here.

noncomputable def partially_coloured_grid : ℕ → ℕ → Colour := sorry

-- The final goal is to determine the colour of cell marked X
def cell_X : ℕ × ℕ := (2, 2)  -- Marking cell X at coordinates (2, 2) for example

-- The hypothesized grid function should satisfy the constraints
axiom correct_grid : vertex_colour_constraint partially_coloured_grid

-- Prove that the colour of cell X is Red
theorem cell_X_is_Red : partially_coloured_grid cell_X.fst cell_X.snd = Red :=
by
  -- The proof steps would be here, but they are omitted since we only need the statement
  sorry

end cell_X_is_Red_l615_615568


namespace AIMN_cyclic_l615_615919

open EuclideanGeometry

variables (A B C I D E F M N : Point)

-- Assume we have a triangle ABC with incentre I
variables [IsIncenter I ABC]

-- Assume the bisectors of ABC meet BC, CA, AB at D, E, F respectively
variables [IsAngleBisector AI BC D]
variable [IsAngleBisector BI CA E]
variable [IsAngleBisector CI AB F]

-- Perpendicular bisector of AD intersects BI and CI at M and N respectively
variables [IsPerpendicularBisectorOf (Segment A D) M BI]
variables [IsPerpendicularBisectorOf (Segment A D) N CI]

theorem AIMN_cyclic : Cyclic (quadrilateral A I M N) :=
sorry

end AIMN_cyclic_l615_615919


namespace sculpture_cost_in_chinese_yuan_l615_615551

-- Definitions from conditions
def namibian_dollars_to_usd (n_dollars : ℝ) : ℝ := n_dollars / 5
def usd_to_chinese_yuan (usd : ℝ) : ℝ := usd * 8
def sculpture_cost_in_namibian_dollars : ℝ := 200

-- Lean theorem to prove the sculpture cost in Chinese Yuan
theorem sculpture_cost_in_chinese_yuan : 
  usd_to_chinese_yuan (namibian_dollars_to_usd sculpture_cost_in_namibian_dollars) = 320 :=
by
  sorry

end sculpture_cost_in_chinese_yuan_l615_615551


namespace probability_sum_squares_l615_615615

noncomputable def prob_sum_of_squares_greater_than (l : ℝ) : ℝ :=
  if 0 ≤ l then 
    1 - (π / 4)
  else 
    0

theorem probability_sum_squares (l : ℝ) (hl_ge_zero : 0 ≤ l) :
  prob_sum_of_squares_greater_than l = 1 - (π / 4) :=
begin
  unfold prob_sum_of_squares_greater_than,
  split_ifs,
  { simp },
  { exfalso, linarith },
end

end probability_sum_squares_l615_615615


namespace range_of_a_l615_615812

theorem range_of_a 
  (f : ℝ → ℝ) (α : ℝ)
  (h1 : f = λ x, x^α)
  (h2 : f (1/2) = 4)
  (h3 : f (a + 1) < f 3) : a ∈ -∞, -4) ∪ (2, ∞) :=
by
  -- Solution to be filled #
sorry

end range_of_a_l615_615812


namespace closed_not_S_has_four_elements_l615_615247

def is_closed (H : set ℝ) : Prop :=
  ∀ x y ∈ H, x + y ∈ H ∨ |x - y| ∈ H

def is_S (H : set ℝ) (n : ℕ) (α : ℝ) : Prop :=
  ∃ (α > 0), ∃ (n : ℕ), H = {k * α | k ∈ fin (n + 1)}

theorem closed_not_S_has_four_elements (H : set ℝ) 
  (H_finite : H.finite)
  (H_closed : is_closed H)
  (H_not_S : ∀ α > 0, ∀ n : ℕ, H ≠ {k * α | k ∈ fin (n + 1)}):
  H.finite.to_finset.card = 4 :=
  sorry

end closed_not_S_has_four_elements_l615_615247


namespace reflection_fixed_points_l615_615229

/-- Given a convex quadrilateral ABCD, determine the number of points that return to their original
positions after being reflected successively across the lines AB, BC, CD, and DA. The result is 1 
if ABCD is not a cyclic quadrilateral, and 0 if ABCD is a cyclic quadrilateral. -/
theorem reflection_fixed_points (A B C D : Point) (h1 : ConvexQuadrilateral A B C D) :
  (isCyclicQuadrilateral A B C D → fixedPoints (reflect successively [AB, BC, CD, DA]) = 0) ∧
  (¬ isCyclicQuadrilateral A B C D → fixedPoints (reflect successively [AB, BC, CD, DA]) = 1) := 
sorry

end reflection_fixed_points_l615_615229


namespace Jung_age_is_26_l615_615220

-- Define the ages of Li, Zhang, and Jung
def Li : ℕ := 12
def Zhang : ℕ := 2 * Li
def Jung : ℕ := Zhang + 2

-- The goal is to prove Jung's age is 26 years
theorem Jung_age_is_26 : Jung = 26 :=
by
  -- Placeholder for the proof
  sorry

end Jung_age_is_26_l615_615220


namespace tan_2α_value_l615_615351

theorem tan_2α_value
  (α : ℝ)
  (h₁ : α ∈ Set.Ioo (π / 4) (π / 2))
  (h₂ : cos (2 * α) / sin (α + π / 4) = -((2: ℝ) * Real.sqrt 5) / 5) :
  tan (2 * α) = -3 / 4 :=
sorry

end tan_2α_value_l615_615351


namespace compare_abc_l615_615055

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615055


namespace compare_abc_l615_615062

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615062


namespace range_of_a_l615_615826

noncomputable def f (a x : ℝ) := -x^2 + 2 * a * x
noncomputable def g (a x : ℝ) := (a + 1)^(1 - x)

theorem range_of_a (a : ℝ) (h₁ : ∀ x ∈ set.Icc (1 : ℝ) 2, f a x' < 0) (h₂ : ∀ x ∈ set.Icc (1 : ℝ) 2, g a x' < 0) : 
0 < a ∧ a ≤ 1 :=
begin
  sorry
end

end range_of_a_l615_615826


namespace luke_clothing_distribution_l615_615927

theorem luke_clothing_distribution (total_clothing: ℕ) (first_load: ℕ) (num_loads: ℕ) 
  (remaining_clothing : total_clothing - first_load = 30)
  (equal_load_per_small_load: (total_clothing - first_load) / num_loads = 6) : 
  total_clothing = 47 ∧ first_load = 17 ∧ num_loads = 5 :=
by
  have h1 : total_clothing - first_load = 30 := remaining_clothing
  have h2 : (total_clothing - first_load) / num_loads = 6 := equal_load_per_small_load
  sorry

end luke_clothing_distribution_l615_615927


namespace union_complement_eq_target_l615_615480

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615480


namespace smallest_positive_integer_to_make_perfect_square_l615_615255

noncomputable def y : ℕ := 2^10 * 3^15 * 4^20 * 5^25 * 6^30 * 7^35 * 8^40 * 9^45

theorem smallest_positive_integer_to_make_perfect_square : ∃ k : ℕ, k = 105 ∧ (∃ n : ℕ, (k * y) = n^2) :=
by
  use 105
  split
  · refl
  · sorry

end smallest_positive_integer_to_make_perfect_square_l615_615255


namespace inequality_a_c_b_l615_615072

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615072


namespace minimum_value_of_f_l615_615108

noncomputable def f (a b : ℝ) : ℝ :=
  3 * real.sqrt (1 + 2 * a^2) + 2 * real.sqrt (40 + 9 * b^2)

theorem minimum_value_of_f : 
  ∃ (a b : ℝ), a + b = 1 ∧ f a b = 5 * real.sqrt (11) :=
sorry

end minimum_value_of_f_l615_615108


namespace clock_equiv_l615_615547

theorem clock_equiv (h : ℕ) (h_gt_6 : h > 6) : h ≡ h^2 [MOD 12] ∧ h ≡ h^3 [MOD 12] → h = 9 :=
by
  sorry

end clock_equiv_l615_615547


namespace measure_angle_B_l615_615784

variables (a b c : ℝ) (A B C : ℝ)
noncomputable def m : ℝ × ℝ := (real.sqrt 3, -1)
noncomputable def n : ℝ × ℝ := (real.cos A, real.sin A)

-- orthogonality condition
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

-- statement of the problem
theorem measure_angle_B (h₁ : orthogonal m n) 
                        (h₂ : a = c * real.sin B) 
                        (h₃ : b = c * real.cos A) : B = 30 :=
                      sorry

end measure_angle_B_l615_615784


namespace cost_of_20_pencils_14_notebooks_l615_615267

-- Define the cost of a pencil and a notebook
variables (p n : ℝ)

-- Given conditions as hypotheses
def condition1 : Prop := 6 * p + 6 * n = 3.90
def condition2 : Prop := 8 * p + 4 * n = 3.28

-- The question: the cost of 20 pencils and 14 notebooks
def total_cost : ℝ := 20 * p + 14 * n

-- The proof problem: prove that the total cost is equal to 10.12 given the conditions
theorem cost_of_20_pencils_14_notebooks (h1 : condition1) (h2 : condition2) : total_cost p n = 10.12 :=
by
  sorry

end cost_of_20_pencils_14_notebooks_l615_615267


namespace midpoint_coordinates_l615_615450

theorem midpoint_coordinates (x0 y0 x1 y1 x2 y2 : ℝ) :
  (x0 = (x1 + x2) / 2) ∧ (y0 = (y1 + y2) / 2) ↔ 
  M (x0, y0) is_midpoint_of_segment (A (x1, y1)) (B (x2, y2)) := 
sorry

end midpoint_coordinates_l615_615450


namespace nested_function_stability_l615_615110

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ :=
  x^2 + p * x + q

theorem nested_function_stability (p q : ℝ)
  (h : ∀ x ∈ set.Icc (4 : ℝ) (6 : ℝ), abs (f x p q) ≤ 1 / 2) :
  ∀ n : ℕ, n ≥ 2017 →
  (Nat.iterate (λ x, f x p q) n ((9 - Real.sqrt 19) / 2)) = ((9 + Real.sqrt 19) / 2) :=
sorry

end nested_function_stability_l615_615110


namespace community_center_chairs_l615_615657

def chairs_needed (people : ℕ) (people_per_chair : ℕ) : ℝ :=
  (people : ℝ) / (people_per_chair : ℝ)  -- Lean's default Euclidean division gives float value

theorem community_center_chairs (h : ℕ) (h = 231) : chairs_needed h 3 = 30.33 := by
  sorry

end community_center_chairs_l615_615657


namespace problem_solution_l615_615770

noncomputable def sequence_a (n : ℕ) : ℕ :=
  n

noncomputable def sequence_b (n : ℕ) : ℕ :=
  2^(n-1)

noncomputable def sequence_c (n : ℕ) : ℕ :=
  (-1 : ℤ)^(n : ℤ) * n + 2^(n-1)

noncomputable def T (n : ℕ) : ℤ :=
  finset.range n.sum (λ i, (-1 : ℤ)^(i+1) * (i+1) + 2^((i+1)-1))

theorem problem_solution : T 100 = 2^100 + 49 :=
sorry

end problem_solution_l615_615770


namespace farthest_vertex_is_H_l615_615675

-- Define the center and area of the square
def center : ℝ × ℝ := (4, -6)
def area : ℝ := 16

-- Define the side length based on the area
def side_length : ℝ := real.sqrt area

-- Coordinates of vertices based on the center
def vertex_E : ℝ × ℝ := (center.1 - side_length / 2, center.2 - side_length / 2)
def vertex_F : ℝ × ℝ := (center.1 - side_length / 2, center.2 + side_length / 2)
def vertex_G : ℝ × ℝ := (center.1 + side_length / 2, center.2 + side_length / 2)
def vertex_H : ℝ × ℝ := (center.1 + side_length / 2, center.2 - side_length / 2)

-- Define the center of dilation and scale factor
def dilation_center : ℝ × ℝ := (2, -2)
def scale_factor : ℝ := 3

-- Function to perform dilation on a point
def dilation (center : ℝ × ℝ) (factor : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + factor * (point.1 - center.1), center.2 + factor * (point.2 - center.2))

-- Coordinates of dilated vertices
def dilated_E : ℝ × ℝ := dilation dilation_center scale_factor vertex_E
def dilated_F : ℝ × ℝ := dilation dilation_center scale_factor vertex_F
def dilated_G : ℝ × ℝ := dilation dilation_center scale_factor vertex_G
def dilated_H : ℝ × ℝ := dilation dilation_center scale_factor vertex_H

-- The proof problem: Prove that the vertex farthest from the center of dilation is (14, -20)
theorem farthest_vertex_is_H : dilated_H = (14, -20) :=
  sorry

end farthest_vertex_is_H_l615_615675


namespace relation_among_abc_l615_615097

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615097


namespace angle_negative_390_is_fourth_quadrant_l615_615583

theorem angle_negative_390_is_fourth_quadrant :
  let alpha := -390 % 360 in
  0 <= alpha + 360 ∧ alpha + 360 < 360 ∧ α + 360 > 270 ∧ α + 360 < 360 → is_in_fourth_quadrant α :=
by
  sorry

end angle_negative_390_is_fourth_quadrant_l615_615583


namespace relation_among_abc_l615_615098

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615098


namespace isosceles_right_triangle_incenter_l615_615996

theorem isosceles_right_triangle_incenter (A B C I : Type) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space I] 
  (h₀ : ∠B = 90°)
  (h₁ : dist A B = 6 * sqrt 2)
  (h₂ : dist B C = 6 * sqrt 2) 
  (h₃ : dist A C = 6 * sqrt 2)
  (h_incenter : ∀ X Y Z : Type, incenter X Y Z = I) : 
  dist B I = 4 := 
sorry

end isosceles_right_triangle_incenter_l615_615996


namespace convex_polytope_exists_l615_615436

noncomputable def convex_polytope_cond (d : ℕ) (K L : Set (ℝ^d)) (ε : ℝ) : Prop :=
  (1 - ε) • K ⊆ L ∧ L ⊆ K ∧
  ∃ X ⊆ K, finset.card X ≤ C(d) * ε^(1 - d) ∧ L = convexHull ℝ X

theorem convex_polytope_exists (d : ℕ) (hd : d ≥ 2) (K : Set (ℝ^d)) (hK : isConvex K ∧ isSymmetric K 0)
  (ε : ℝ) (hε : 0 < ε ∧ ε < 1) :
  ∃ C : ℕ → ℝ, ∃ L : Set (ℝ^d), convex_polytope_cond d K L ε :=
sorry

end convex_polytope_exists_l615_615436


namespace least_grapes_in_heap_l615_615252

theorem least_grapes_in_heap :
  ∃ n : ℕ, (n % 19 = 1) ∧ (n % 23 = 1) ∧ (n % 29 = 1) ∧ n = 12209 :=
by
  sorry

end least_grapes_in_heap_l615_615252


namespace greyson_spent_on_fuel_l615_615818

theorem greyson_spent_on_fuel : ∀ (cost_per_refill times_refilled total_cost : ℕ), 
  cost_per_refill = 10 → 
  times_refilled = 4 → 
  total_cost = cost_per_refill * times_refilled → 
  total_cost = 40 :=
by
  intro cost_per_refill times_refilled total_cost
  intro h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end greyson_spent_on_fuel_l615_615818


namespace train_length_is_360_l615_615679

-- Conditions from the problem
variable (speed_kmph : ℕ) (time_sec : ℕ) (platform_length_m : ℕ)

-- Definitions to be used for the conditions
def speed_ms (speed_kmph : ℕ) : ℤ := (speed_kmph * 1000) / 3600 -- Speed in m/s
def total_distance (speed_ms : ℤ) (time_sec : ℕ) : ℤ := speed_ms * (time_sec : ℤ) -- Total distance covered
def train_length (total_distance : ℤ) (platform_length : ℤ) : ℤ := total_distance - platform_length -- Length of the train

-- Assertion statement
theorem train_length_is_360 : train_length (total_distance (speed_ms speed_kmph) time_sec) platform_length_m = 360 := 
  by sorry

end train_length_is_360_l615_615679


namespace nail_polish_count_l615_615892

-- Definitions from conditions
def K : ℕ := 25
def H : ℕ := K + 8
def Ka : ℕ := K - 6
def L : ℕ := 2 * K
def S : ℕ := 13 + 10  -- Since 25 / 2 = 12.5, rounded to 13 for practical purposes

-- Statement to prove
def T : ℕ := H + Ka + L + S

theorem nail_polish_count : T = 125 := by
  sorry

end nail_polish_count_l615_615892


namespace new_arithmetic_mean_after_removal_l615_615686

theorem new_arithmetic_mean_after_removal (A : List ℤ) (h_len : A.length = 40) 
    (h_mean : (A.sum : ℚ) / 40 = 45) :
    (A.sum - 60 - 70) / 38 = 43.95 := 
by
    sorry

end new_arithmetic_mean_after_removal_l615_615686


namespace compare_a_b_c_l615_615018

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615018


namespace glucose_mass_in_container_l615_615659

noncomputable def concentration (x : ℝ) : ℝ := 5 * Real.sin(x) + 10

theorem glucose_mass_in_container : 
  let mass_per_cc := 1.2
  let volume_in_cc := 45
  ∃ total_mass : ℝ, 
   (total_mass = mass_per_cc * volume_in_cc) := sorry

end glucose_mass_in_container_l615_615659


namespace largestPerfectSquareFactor_and_LCM_proof_l615_615202

-- Define the given conditions
def givenNumber1 := 4410
def givenNumber2 := 18

-- Define the prime factorization of the givenNumber1 and the resulting perfect square factor
def primeFactorization1 := [2^1, 3^2, 5^1, 7^2]
def largestPerfectSquareFactor := 3^2 * 7^2

-- Define the prime factorization of the givenNumber2
def primeFactorization2 := [2^1, 3^2]

-- LCM calculation based on the highest powers of shared primes
def lcm := 2^1 * 3^2 * 7^2

-- Prove that the largest perfect square factor of givenNumber1 is 441, and the LCM of this factor with givenNumber2 is 882
theorem largestPerfectSquareFactor_and_LCM_proof : 
    largestPerfectSquareFactor = 441 ∧ lcm = 882 :=
begin
    -- Since the proof is not required, we use sorry to fill in the proof
    sorry
end

end largestPerfectSquareFactor_and_LCM_proof_l615_615202


namespace compare_abc_l615_615063

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615063


namespace problem_l615_615528

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615528


namespace cone_volume_l615_615790

noncomputable def volume_of_cone (slant_height lateral_area : ℝ) : ℝ :=
  let r := lateral_area / (π * slant_height)
  let h := Math.sqrt (slant_height^2 - r^2)
  (1 / 3) * π * r^2 * h

theorem cone_volume (slant_height : ℝ) (lateral_area : ℝ) (h := 3) :
  slant_height = 5 →
  lateral_area = 20 * π →
  volume_of_cone slant_height lateral_area = 16 * π :=
by
  intros
  simp [volume_of_cone]
  sorry

end cone_volume_l615_615790


namespace max_integer_solutions_poly_eq_l615_615667

theorem max_integer_solutions_poly_eq (p : ℤ[X]) (h_coeffs : ∀ n : ℕ, p.coeff n ∈ ℤ) (h100: p.eval 100 = 100) : 
  ∃ k : ℕ, (∀ x : ℤ, (p.eval x = x^4 -> is_int_root p x) ∧ k ≤ 12) :=
sorry

end max_integer_solutions_poly_eq_l615_615667


namespace compare_a_b_c_l615_615012

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615012


namespace proof_problem_l615_615879

-- Define the angles A, B, C of the triangle in radians 
variables {A B C : ℝ}

-- Since we are given a trigonometric equation, we'll assume x, y, z to be real numbers 
variables {x y z : ℝ}

-- Define the sine and cosine terms
variables (sin cos : ℝ → ℝ)

-- Lean can deduce that terms like sin A and cos A pertain to the angles A, B, and C

-- The condition given in the problem
def condition : Prop := x * sin A + y * sin B + z * sin C = 0

-- The theorem to prove
theorem proof_problem (h : condition) : 
  (y + z * cos A) * (z + x * cos B) * (x + y * cos C) + 
  (y * cos A + z) * (z * cos B + x) * (x * cos C + y) = 0 := by
  sorry

end proof_problem_l615_615879


namespace find_remainder_l615_615851

noncomputable def dividend : ℝ := 82247.3
noncomputable def divisor : ℝ := 5127.54
noncomputable def quotient : ℝ := 16.041

theorem find_remainder :
  (dividend - divisor * quotient).round = 24.53 :=
by
  sorry

end find_remainder_l615_615851


namespace two_pow_2001_mod_127_l615_615622

theorem two_pow_2001_mod_127 : (2^2001) % 127 = 64 := 
by
  sorry

end two_pow_2001_mod_127_l615_615622


namespace min_value_proof_l615_615103

noncomputable def minimum_value (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x = 2 * y) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_proof :
  ∀ (x y z : ℝ), (0 < x) → (0 < y) → (0 < z) → (x + y + z = 3) → (x = 2 * y) → 
  minimum_value x y z (x + y + z = 3) (x = 2 * y) = 4 / 3 :=
by
  intros 
  sorry

end min_value_proof_l615_615103


namespace max_sum_unique_digits_expression_equivalent_l615_615728

theorem max_sum_unique_digits_expression_equivalent :
  ∃ (a b c d e : ℕ), (2 * 19 * 53 = 2014) ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (2 * (b + c) * (d + e) = 2014) ∧
    (a + b + c + d + e = 35) ∧ 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) :=
by
  sorry

end max_sum_unique_digits_expression_equivalent_l615_615728


namespace function_neither_even_nor_odd_l615_615719

def f (x : ℝ) := ⌊x⌋ + 1 / 3

theorem function_neither_even_nor_odd : ¬ (∀ x : ℝ, f x = f (-x)) ∧ ¬ (∀ x : ℝ, f x = -f (-x)) := by
  -- Proof corresponding to the given conditions and solution
  sorry

end function_neither_even_nor_odd_l615_615719


namespace probability_regions_equal_l615_615652

theorem probability_regions_equal :
  let P : String → ℚ := λ s, if s = "A" then 1/4 else if s = "B" then 1/3 else if s = "C" then 5/24 else 5/24
  in ∀ (region : String), region ∈ ["C", "D"] → P region = 5/24 :=
by
  let P : String → ℚ := λ s, if s = "A" then 1/4 else if s = "B" then 1/3 else if s = "C" then 5/24 else 5/24
  intro region h
  have h_total : P "A" + P "B" + P "C" + P "D" = 1 := by
    simp [P]
    norm_num
  cases h
  repeat {
    simp [P, h]
  }
  sorry

end probability_regions_equal_l615_615652


namespace centroid_distance_in_triangle_l615_615783

theorem centroid_distance_in_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (O : Type) 
  (cosB : ℝ) 
  (OA OB OC : ℝ → ℝ) 
  (cosA : ℝ)
  (sinA : ℝ) 
  (h1 : b = 6)
  (h2 : accosB = a^2 - b^2 + (sqrt 7 / 4) * b * c)
  (h3 : (OA + OB + OC = 0) : Prop)
  (h4 : ∠ BAO = 30°) :
  |OA| = 3 := 
sorry

end centroid_distance_in_triangle_l615_615783


namespace lincoln_summer_camp_l615_615290

theorem lincoln_summer_camp :
    (∀ (N : ℝ), N > 0 →
       let basketball_players := 0.7 * N in
       let swimmers := 0.4 * N in
       let basketball_swimmers := 0.3 * basketball_players in
       let non_swimming_basketball_players := basketball_players - basketball_swimmers in
       let non_swimmers := N - swimmers in
       let percent_non_swimmers_in_basketball := (non_swimming_basketball_players / non_swimmers) * 100 in
       100 - percent_non_swimmers_in_basketball = 18) :=
  sorry

end lincoln_summer_camp_l615_615290


namespace number_of_planes_with_distance_ratio_l615_615823

variable {A B C D : Point}

-- Definition of a regular tetrahedron.
def regular_tetrahedron (A B C D : Point) : Prop :=
  equidistant A B C D

-- Definition of the distance ratio condition.
def distance_ratio (plane : Plane) (A B C D : Point) : Prop :=
  dist_to_plane plane A = dist_to_plane plane B ∧
  dist_to_plane plane B = dist_to_plane plane C ∧
  dist_to_plane plane C = dist_to_plane plane A ∧
  dist_to_plane plane D = sqrt 2 * dist_to_plane plane A

-- Problem statement in Lean 4.
theorem number_of_planes_with_distance_ratio (A B C D : Point) (h : regular_tetrahedron A B C D) :
  ∃ planes : ℕ, planes = 32 ∧ ∀ plane, distance_ratio plane A B C D ↔ plane_in_counted_planes plane :=
sorry

end number_of_planes_with_distance_ratio_l615_615823


namespace max_green_socks_l615_615658

theorem max_green_socks (g y : ℕ) (h_t : g + y ≤ 2000) (h_prob : (g * (g - 1) + y * (y - 1) = (g + y) * (g + y - 1) / 3)) :
  g ≤ 19 := by
  sorry

end max_green_socks_l615_615658


namespace inequality_a_c_b_l615_615071

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615071


namespace stickers_initial_count_l615_615434

variable (initial : ℕ) (lost : ℕ)

theorem stickers_initial_count (lost_stickers : lost = 6) (remaining_stickers : initial - lost = 87) : initial = 93 :=
by {
  sorry
}

end stickers_initial_count_l615_615434


namespace liam_number_of_nickels_l615_615924

theorem liam_number_of_nickels :
  ∃ n : ℤ, 120 < n ∧ n < 400 ∧
  (n % 4 = 2) ∧
  (n % 5 = 3) ∧
  (n % 6 = 4) ∧
  (n = 374) :=
by
  use 374
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  split
  { norm_num }
  { refl }

end liam_number_of_nickels_l615_615924


namespace prob_at_least_4_yuan_prob_less_than_20_yuan_l615_615312

theorem prob_at_least_4_yuan :
  let black := 3
  let red := 2
  let white := 1
  num_total_draws := 15
  events_4_yuan_draws := 4
  events_5_yuan_draws := 2

  (events_4_yuan_draws + events_5_yuan_draws) / num_total_draws = (2 / 5) := by
  sorry

theorem prob_less_than_20_yuan :
  let black := 3
  let red := 2
  let white := 1
  num_total_draws := 36
  events_20_yuan_draws := 4

  (1 - events_20_yuan_draws / num_total_draws) = (8 / 9) := by
  sorry

end prob_at_least_4_yuan_prob_less_than_20_yuan_l615_615312


namespace solve_system_l615_615644

theorem solve_system :
  (∀ x y : ℝ, log 4 x - log 2 y = 0 ∧ x^2 - 5 * y^2 + 4 = 0 → 
    (x, y) = (1, 1) ∨ (x, y) = (4, 2)) :=
by
  intros x y h
  cases h with Hlog Heq
  sorry

end solve_system_l615_615644


namespace union_complement_eq_l615_615507

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615507


namespace find_100m_l615_615956

-- Definition of the problem with conditions
def square_side_length : ℝ := 4
def segment_length : ℝ := 4

-- Let T be the set of all line segments of length 4 with endpoints on adjacent sides of the square.
def endpoint_relation (x y : ℝ) : Prop := x^2 + y^2 = segment_length^2

-- Midpoint contained area
def enclosed_area_by_midpoints (side_length : ℝ) : ℝ := 
  let quarter_circle_area := (Real.pi * (side_length / 2) ^ 2) / 2
  (side_length * side_length) - 4 * quarter_circle_area

-- The statement to be proved
theorem find_100m : 100 * enclosed_area_by_midpoints square_side_length = 343 := 
sorry

end find_100m_l615_615956


namespace inequality_a_c_b_l615_615075

theorem inequality_a_c_b :
  let a := 2 * Real.log 1.01
  let b := Real.log 1.02
  let c := Real.sqrt 1.04 - 1
  a > c ∧ c > b :=
by
  let a : ℝ := 2 * Real.log 1.01
  let b : ℝ := Real.log 1.02
  let c : ℝ := Real.sqrt 1.04 - 1
  split
  sorry
  sorry

end inequality_a_c_b_l615_615075


namespace proof_correct_answer_l615_615298

noncomputable def compute_expression : ℝ :=
  ((1 / 27) ^ (-1 / 3)) + ((Mathlib.Math.pi - 1) ^ 0) + 2 * (Real.logBase 3 1) - (Real.log 10 2) - (Real.log 10 5)

theorem proof_correct_answer : compute_expression = 3 :=
by
  sorry

end proof_correct_answer_l615_615298


namespace compare_abc_l615_615065

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615065


namespace solve_for_x_l615_615571

theorem solve_for_x :
  (sqrt (9 + sqrt (18 + 9 * 34)) + sqrt (3 + sqrt (3 + 34)) = 3 + 3 * sqrt 3) :=
  by {
    rw sqrt_add_sqrt_eq_3_mulsqrt_sqrt_3_34,
    sorry,
  }

-- The statement sqrt_add_sqrt_eq_3_mulsqrt_sqrt_3_34 should be built for the equation equivalence.
-- We do not require a proof for this theorem here, so it's kept with 'sorry'.
-- Note: actual function names and details would depend on the definitions provided in the actual Lean environment.

end solve_for_x_l615_615571


namespace pictures_in_each_album_l615_615138

/-
Problem Statement:
Robin uploaded 35 pictures from her phone and 5 from her camera, which makes 35 + 5 = 40 pictures in total.
She sorted these 40 pictures into 5 different albums with the same amount of pictures in each album.
Prove that there are 8 pictures in each album.
-/

theorem pictures_in_each_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (total_albums : ℕ)
  (total_pics : phone_pics + camera_pics = 40)
  (same_amount : ∀ x, x < total_albums → (phone_pics + camera_pics) / total_albums = 8) : 
  8 = (phone_pics + camera_pics) / total_albums :=
by
  -- given conditions
  have phone_pics := 35
  have camera_pics := 5
  have total_albums := 5
  -- show the result
  show 8 = (phone_pics + camera_pics) / total_albums
  sorry

end pictures_in_each_album_l615_615138


namespace find_c_value_l615_615600

def polynomial_factorization (c q a : ℤ) : Prop :=
  3 + q = 8 ∧ a = 9 ∧ 3 * q + a = c

theorem find_c_value : ∃ c, polynomial_factorization c 5 9 :=
by
  use 24
  unfold polynomial_factorization
  simp
  sorry

end find_c_value_l615_615600


namespace min_sum_abs_l615_615986

theorem min_sum_abs (x : ℝ) : ∃ m, m = 4 ∧ ∀ x : ℝ, |x + 2| + |x - 2| + |x - 1| ≥ m := 
sorry

end min_sum_abs_l615_615986


namespace intersection_A_B_eq_l615_615376

open Set

namespace LeanProof

def A : Set ℕ := { x | 0 ≤ x ∧ x ≤ 5 }
def B : Set ℕ := { x | x ∈ {1, 2, 3} }

theorem intersection_A_B_eq : A ∩ B = { 1, 2, 3 } :=
by
  sorry

end LeanProof

end intersection_A_B_eq_l615_615376


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l615_615227

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 / 18) → (∀ x > 0, x + a / (2 * x) ≥ 1 / 3) :=
by
  intros ha x hx
  sorry

theorem not_necessary_condition :
  ¬ ∀ a x > 0, x + a / (2 * x) ≥ 1 / 3 :=
by
  intro h
  have ha : 2 + 2 / (2 * 1) = 3 := by norm_num
  specialize h 2 1 zero_lt_one
  linarith

end sufficient_but_not_necessary_condition_not_necessary_condition_l615_615227


namespace union_complement_eq_l615_615511

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615511


namespace integral_solution_l615_615226

noncomputable def integral_func (x : ℝ) : ℝ :=
  (4 * real.sqrt (2 - x) - real.sqrt (3 * x + 2)) / ((real.sqrt (3 * x + 2) + 4 * real.sqrt (2 - x)) * (3 * x + 2) ^ 2)

theorem integral_solution :
  ∫ x in 0..2, integral_func x = 1 / 32 * real.log 5 :=
by
  sorry

end integral_solution_l615_615226


namespace range_for_k_solutions_when_k_eq_1_l615_615339

noncomputable section

-- Part (1): Range for k
theorem range_for_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - (2 * k + 4) * x + k - 6 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2)) ↔ (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Part (2): Completing the square for k = 1
theorem solutions_when_k_eq_1 :
  (∀ x : ℝ, x^2 - 6 * x - 5 = 0 → (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14)) :=
sorry

end range_for_k_solutions_when_k_eq_1_l615_615339


namespace geometric_sequence_first_term_and_ratio_l615_615603

theorem geometric_sequence_first_term_and_ratio (b : ℕ → ℚ) 
  (hb2 : b 2 = 37 + 1/3) 
  (hb6 : b 6 = 2 + 1/3) : 
  ∃ (b1 q : ℚ), b 1 = b1 ∧ (∀ n, b n = b1 * q^(n-1)) ∧ b1 = 224 / 3 ∧ q = 1 / 2 :=
by 
  sorry

end geometric_sequence_first_term_and_ratio_l615_615603


namespace smallest_square_area_l615_615649

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 4) (h4 : d = 5) :
  ∃ s, s^2 = 81 ∧ (a ≤ s ∧ b ≤ s ∧ c ≤ s ∧ d ≤ s ∧ (a + c) ≤ s ∧ (b + d) ≤ s) :=
sorry

end smallest_square_area_l615_615649


namespace isabel_initial_amount_l615_615427

theorem isabel_initial_amount (X : ℝ) (h : X / 2 - X / 4 = 51) : X = 204 :=
sorry

end isabel_initial_amount_l615_615427


namespace average_comparisons_sequential_search_l615_615974

theorem average_comparisons_sequential_search 
  (n : ℕ)
  (arr : Array ℕ)
  (not_found : n = 100) 
  (avg_comparisons : nat.average_comparisons(arr) = 100) 
  : ∀ order : list ℕ, nat.average_comparisons(order) = 100 := 
by sorry

end average_comparisons_sequential_search_l615_615974


namespace number_of_selection_methods_l615_615608

theorem number_of_selection_methods (students lectures : ℕ) (h_stu : students = 6) (h_lect : lectures = 5) :
  lectures ^ students = 15625 :=
by
  -- Substitute the values based on the conditions
  have h1 : lectures = 5 := h_lect,
  have h2 : students = 6 := h_stu,
  rw [h1, h2],
  -- Compute 5^6
  exact Nat.pow_succ_succ 5 5 (5 ^ 5) sorry -- 5^6 = 15625

end number_of_selection_methods_l615_615608


namespace days_of_supply_l615_615695

-- Define the conditions as Lean definitions
def visits_per_day : ℕ := 3
def squares_per_visit : ℕ := 5
def total_rolls : ℕ := 1000
def squares_per_roll : ℕ := 300

-- Define the daily usage calculation
def daily_usage : ℕ := squares_per_visit * visits_per_day

-- Define the total squares calculation
def total_squares : ℕ := total_rolls * squares_per_roll

-- Define the proof statement for the number of days Bill's supply will last
theorem days_of_supply : (total_squares / daily_usage) = 20000 :=
by
  -- Placeholder for the actual proof, which is not required per instructions
  sorry

end days_of_supply_l615_615695


namespace parabola_focus_distance_circle_parabola_intersection_l615_615374

noncomputable def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x
noncomputable def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
noncomputable def circle (a x y : ℝ) := (x - a)^2 + y^2 = 1

theorem parabola_focus_distance (y : ℝ) (hp : 0 < p) (hd : distance 1 y p 0 = 17 / 16) :
  p = 1 / 8 :=
sorry

theorem circle_parabola_intersection (p : ℝ) (y : ℝ) (hp : 0 < p) (a : ℝ) :
  (∀ x y, parabola p x y → circle a x y → x^2 + y^2 = 1 → -1 ≤ a ∧ a ≤ 65 / 16) :=
sorry

end parabola_focus_distance_circle_parabola_intersection_l615_615374


namespace contrapositive_true_count_is_two_l615_615189

def proposition_1 (OA OA' OB OB' : Prop) : Prop :=
  (OA ∧ OA' ∧ OB ∧ OB') → (angle AOB = angle A'O'B' ∨ ∠AOB + ∠A'O'B' = 180)

def proposition_2 : Prop :=
  (right_angled_trapezoid → planar_figure)

def proposition_3 : Prop :=
  (∃ (X : Type), regular_quadrilateral_prism X ∧ right_parallelepiped X ∧ rectangular_prism X)

def proposition_4 (P A B C : Type) : Prop :=
  (tetrahedron P A B C ∧ (PA ⟂ BC) ∧ (PB ⟂ AC)) → is_orthocenter (project A (plane P B C)) (triangle P B C)

theorem contrapositive_true_count_is_two : 
  (number_of_true_contrapositives [proposition_1, proposition_2, proposition_3, proposition_4]) = 2 := 
by
  sorry

end contrapositive_true_count_is_two_l615_615189


namespace pizza_slices_with_both_toppings_l615_615237

theorem pizza_slices_with_both_toppings :
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  n = 6 :=
by
  let T := 16
  let c := 10
  let p := 12
  let n := c + p - T
  show n = 6
  sorry

end pizza_slices_with_both_toppings_l615_615237


namespace problem_l615_615529

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l615_615529


namespace gcd_seq_finitely_many_values_l615_615894

def gcd_seq_finite_vals (A B : ℕ) (x : ℕ → ℕ) : Prop :=
  (∀ n ≥ 2, x (n + 1) = A * Nat.gcd (x n) (x (n-1)) + B) →
  ∃ N : ℕ, ∀ m n, m ≥ N → n ≥ N → x m = x n

theorem gcd_seq_finitely_many_values (A B : ℕ) (x : ℕ → ℕ) :
  gcd_seq_finite_vals A B x :=
by
  intros h
  sorry

end gcd_seq_finitely_many_values_l615_615894


namespace EF_over_EH_equal_2_l615_615997

theorem EF_over_EH_equal_2
  (A B C D E F H G : Point)
  (EA EB AD BC : ℝ)
  (P : Plane)
  (h1 : A ≠ B)
  (h2 : A ≠ C)
  (h3 : D ≠ A)
  (h4 : D ≠ B)
  (h5 : A ≠ E)
  (h6 : B ≠ E)
  (h7 : C ≠ F)
  (h8 : ∀ P Q : Point, parallel P Q AD BC)
  (h9 : on_edge E A B)
  (h10 : on_edge F A C)
  (h11 : EA / EB = 2)
  (h12 : AD = BC)
  : EF / EH = 2 :=
sorry

end EF_over_EH_equal_2_l615_615997


namespace masha_talk_time_l615_615542

-- condition: setting the battery drain times for talking and standby
constant talk_time : ℝ := 5
constant standby_time : ℝ := 150
constant total_time : ℝ := 24

-- The main theorem we need to prove
theorem masha_talk_time :
  ∃ x : ℝ, (x / talk_time + (total_time - x) / standby_time = 1) ∧ x = 126 / 29 :=
by
  sorry

end masha_talk_time_l615_615542


namespace union_complement_l615_615493

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615493


namespace fraction_of_smaller_part_l615_615566

theorem fraction_of_smaller_part (A B : ℕ) (x : ℚ) (h1 : A + B = 66) (h2 : A = 50) (h3 : 0.40 * A = x * B + 10) : x = 5 / 8 :=
by
  sorry

end fraction_of_smaller_part_l615_615566


namespace arithmetic_sequence_sum_l615_615184

theorem arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (c : ℤ) :
  (∀ n : ℕ, 0 < n → S_n n = n^2 + c) →
  a_n 1 = 1 + c →
  (∀ n, 1 < n → a_n n = S_n n - S_n (n - 1)) →
  (∀ n : ℕ, 0 < n → a_n n = 1 + (n - 1) * 2) →
  c = 0 ∧ (∀ n : ℕ, 0 < n → a_n n = 2 * n - 1) :=
by
  sorry

end arithmetic_sequence_sum_l615_615184


namespace dice_probability_sum_15_l615_615192

def is_valid_combination (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 15

def count_outcomes : ℕ :=
  6 * 6 * 6

def count_valid_combinations : ℕ :=
  10  -- From the list of valid combinations

def probability (valid_count total_count : ℕ) : ℚ :=
  valid_count / total_count

theorem dice_probability_sum_15 : probability count_valid_combinations count_outcomes = 5 / 108 :=
by
  sorry

end dice_probability_sum_15_l615_615192


namespace alice_bob_probability_l615_615983

noncomputable def probability_of_exactly_two_sunny_days : ℚ :=
  let p_sunny := 3 / 5
  let p_rain := 2 / 5
  3 * (p_sunny^2 * p_rain)

theorem alice_bob_probability :
  probability_of_exactly_two_sunny_days = 54 / 125 := 
sorry

end alice_bob_probability_l615_615983


namespace determine_phi_l615_615799

-- Define the function f
def f (x ϕ : ℝ) := 2 * Real.sin (x + ϕ)

-- Define the function g, which is a left shift of f by π/3
def g (x ϕ : ℝ) := f (x + π / 3) ϕ

-- Define the conditions for ϕ being between 0 and π/2
def phi_condition (ϕ : ℝ) := 0 < ϕ ∧ ϕ < π / 2

-- Define the property of an even function
def is_even (h : ℝ → ℝ) := ∀ x, h x = h (-x)

-- Main theorem to prove that ϕ = π / 6
theorem determine_phi (ϕ : ℝ) (h_phi_condition : phi_condition ϕ) (h_even : is_even (g · ϕ)) : ϕ = π / 6 :=
by
  sorry

end determine_phi_l615_615799


namespace original_useful_item_is_pencil_l615_615845

def code_language (x : String) : String :=
  if x = "item" then "pencil"
  else if x = "pencil" then "mirror"
  else if x = "mirror" then "board"
  else x

theorem original_useful_item_is_pencil : 
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") ∧
  (code_language "item" = "pencil") ∧
  (code_language "pencil" = "mirror") ∧
  (code_language "mirror" = "board") 
  → "mirror" = "pencil" :=
by sorry

end original_useful_item_is_pencil_l615_615845


namespace min_value_reciprocal_sum_l615_615782

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

end min_value_reciprocal_sum_l615_615782


namespace biased_coin_probability_l615_615650

theorem biased_coin_probability : ∃ x : ℝ, x = (1 / 2 - (real.sqrt (16 - 8 * real.cbrt 5)) / 16) ∧ x < 1 / 2 ∧ (20 * x^3 * (1 - x)^3 = 5 / 32) :=
by
  sorry

end biased_coin_probability_l615_615650


namespace value_of_f_f_neg1_l615_615167

def f (x: ℝ) : ℝ :=
if x >= 0 then x^2 + 2 else -x + 1

theorem value_of_f_f_neg1 :
  f (f (-1)) = 6 :=
by
  -- sorry is a placeholder for the proof
  sorry

end value_of_f_f_neg1_l615_615167


namespace compare_abc_l615_615000

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615000


namespace number_zeros_in_quotient_l615_615900

def S (k : ℕ) : ℕ := (10^k - 1) / 9

theorem number_zeros_in_quotient :
  let Q := S 30 / S 5
  (∃ n : ℕ, number_of_zeros Q = 59049)
:= sorry

end number_zeros_in_quotient_l615_615900


namespace orange_balls_count_l615_615278

theorem orange_balls_count :
  ∀ (total red blue orange pink : ℕ), 
  total = 50 → red = 20 → blue = 10 → 
  total = red + blue + orange + pink → 3 * orange = pink → 
  orange = 5 :=
by
  intros total red blue orange pink h_total h_red h_blue h_total_eq h_ratio
  sorry

end orange_balls_count_l615_615278


namespace calculate_AB_l615_615836

theorem calculate_AB (A B C : Point) (α β γ : ℝ) (h_triangle : right_triangle A B C)
  (h_angle_a : angle A = 90)
  (h_tan_b : tan B = 5 / 12)
  (h_ac : dist A C = 39) : dist A B = 15 :=
by 
  sorry

end calculate_AB_l615_615836


namespace number_of_charms_l615_615611

-- Let x be the number of charms used to make each necklace
variable (x : ℕ)

-- Each charm costs $15
variable (cost_per_charm : ℕ)
axiom cost_per_charm_is_15 : cost_per_charm = 15

-- Tim sells each necklace for $200
variable (selling_price : ℕ)
axiom selling_price_is_200 : selling_price = 200

-- Tim makes a profit of $1500 if he sells 30 necklaces
variable (total_profit : ℕ)
axiom total_profit_is_1500 : total_profit = 1500

theorem number_of_charms (h : 30 * (selling_price - cost_per_charm * x) = total_profit) : x = 10 :=
sorry

end number_of_charms_l615_615611


namespace min_distance_l615_615866

def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x + 2 * y + 1 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y + 9 = 0

def is_tangent (P : ℝ × ℝ) (circle : ℝ × ℝ → Prop) : Prop := sorry

def equal_tangents (P : ℝ × ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ), is_tangent P (circle1 A.1 A.2) → is_tangent P (circle2 B.1 B.2) → dist P A = dist P B

def locus_P (x y : ℝ) : Prop :=
  3 * x + 4 * y - 4 = 0

def min_OP_distance :=
  4 / 5

theorem min_distance (x y : ℝ) (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  equal_tangents P →
  locus_P P.1 P.2 →
  ∀ O : ℝ × ℝ, O = (0, 0) → 
  dist O P = min_OP_distance := 
sorry

end min_distance_l615_615866


namespace impossible_twice_as_many_knights_as_liars_l615_615128

variables {Inhabitant : Type} (is_knight is_liar : Inhabitant → Prop)
variables (friends : Inhabitant → Finset Inhabitant)

theorem impossible_twice_as_many_knights_as_liars (x y D : ℕ) :
  (∀ i : Inhabitant,
     friends i.card = 10 ∧
     (is_knight i → (∀ f ∈ friends i, is_liar f) → 6 * x ≤ D) ∧
     (is_liar i → (∀ f ∈ friends i, is_knight f) → D ≤ 10 * y)) →
  ¬ (x = 2 * y) :=
by
  sorry

end impossible_twice_as_many_knights_as_liars_l615_615128


namespace smallest_three_digit_number_l615_615392

theorem smallest_three_digit_number (x : ℤ) (h1 : x - 7 % 7 = 0) (h2 : x - 8 % 8 = 0) (h3 : x - 9 % 9 = 0) : x = 504 := 
sorry

end smallest_three_digit_number_l615_615392


namespace sum_row_50_l615_615703

def sum_in_row : ℕ → ℕ
| 1 := 1
| n := 2 * sum_in_row (n - 1) + 2 * n

theorem sum_row_50 : sum_in_row 50 = 2^50 * 50 :=
by
  sorry

end sum_row_50_l615_615703


namespace solve_system_l615_615643

theorem solve_system :
  (∀ x y : ℝ, log 4 x - log 2 y = 0 ∧ x^2 - 5 * y^2 + 4 = 0 → 
    (x, y) = (1, 1) ∨ (x, y) = (4, 2)) :=
by
  intros x y h
  cases h with Hlog Heq
  sorry

end solve_system_l615_615643


namespace intersection_of_A_and_complement_of_B_l615_615118

open Set

variable (α : Type) [Fintype α] [DecidableEq α]

def universalSet : Set ℕ := {1, 2, 3, 4, 5}
def setA : Set ℕ := {1, 2}
def setB : Set ℕ := {2, 3}

theorem intersection_of_A_and_complement_of_B :
  setA \cap (universalSet \setB) = {1} :=
sorry

end intersection_of_A_and_complement_of_B_l615_615118


namespace union_complement_eq_l615_615473

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615473


namespace monochromatic_triangle_probability_l615_615313
open Classical

theorem monochromatic_triangle_probability :
  let vertices := Finset.univ : Finset (Fin 6),
      edges := vertices.powerset.filter (λ s, s.card = 2),
      edge_coloring : List (edge → Prop) := [edge → (rand.bool)]
  in
  ∃ triangle ∈ vertices.powerset, triangle.card = 3 ∧
    (∃ color : Prop, triangle.all (λ edge, color edge)) :=
begin
  sorry
end

end monochromatic_triangle_probability_l615_615313


namespace greatest_special_nat_exists_l615_615326

theorem greatest_special_nat_exists (d : ℕ → ℕ) (n : ℕ) (h1 : d 1 = 9) (hn : d n = 1)
  (h2 : ∀ i, 2 ≤ i ∧ i ≤ n - 1 → d i < (d (i - 1) + d (i + 1)) / 2) : 
  ( ∃ m, nat.to_digits m = [9, 8, 6, 4, 2, 1] ) ∧ ( ∀ k, nat.to_digits k = [9, 8, 6, 4, 2, 1] → m ≤ k ):=
by
  sorry

end greatest_special_nat_exists_l615_615326


namespace fence_length_40_l615_615263

noncomputable def total_fence_length (x : ℝ) : ℝ := 2 * x + 2 * x

theorem fence_length_40 (x : ℝ) (h1 : 2 * x^2 = 200) : total_fence_length (Real.sqrt 100) = 40 :=
by
  have hx : x = 10 := by sorry
  rw [←hx, Real.sqrt_sq]
  simp [total_fence_length]
  exact Real.sqrt_eq rfl

end fence_length_40_l615_615263


namespace parabola_distance_l615_615357

theorem parabola_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (h_distance_focus : (P.1 - 1)^2 + P.2^2 = 9) : 
  Real.sqrt (P.1^2 + P.2^2) = 2 * Real.sqrt 3 :=
by
  sorry

end parabola_distance_l615_615357


namespace investment_duration_l615_615334

theorem investment_duration 
  (P SI R : ℕ) (T : ℕ) 
  (hP : P = 800) 
  (hSI : SI = 128) 
  (hR : R = 4) 
  (h : SI = P * R * T / 100) 
  : T = 4 :=
by 
  rw [hP, hSI, hR] at h
  sorry

end investment_duration_l615_615334


namespace find_X_value_l615_615399

variable (country_value_for_fifths : ℝ)
variable (multiplier : ℝ)
variable (X : ℝ)

def specific_condition_one : Prop :=
  (1 / 5) * 8 = country_value_for_fifths

def specific_condition_two : Prop :=
  country_value_for_fifths = 4

def calculate_X : Prop :=
  (1 / 4) * X * multiplier = 10

theorem find_X_value (h1 : specific_condition_one country_value_for_fifths) (h2 : specific_condition_two) : X = 16 :=
by
  sorry

end find_X_value_l615_615399


namespace unique_and_double_solutions_l615_615669

theorem unique_and_double_solutions (a : ℝ) :
  (∃ (x : ℝ), 5 + |x - 2| = a ∧ ∀ y, 5 + |y - 2| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 7 - |2*x1 + 6| = a ∧ 7 - |2*x2 + 6| = a)) ∨
  (∃ (x : ℝ), 7 - |2*x + 6| = a ∧ ∀ y, 7 - |2*y + 6| = a → y = x ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 5 + |x1 - 2| = a ∧ 5 + |x2 - 2| = a)) ↔ a = 5 ∨ a = 7 :=
by
  sorry

end unique_and_double_solutions_l615_615669


namespace john_horizontal_distance_l615_615433

-- Definition of the given conditions
def vertical_to_horizontal_ratio := 5 / 2
def total_vertical_distance := 5100

-- Statement to prove
theorem john_horizontal_distance :
  let x := (2 * total_vertical_distance) / 5 in
  x = 2040 :=
by
  -- The proof is omitted
  sorry

end john_horizontal_distance_l615_615433


namespace number_div_mult_l615_615212

theorem number_div_mult (n : ℕ) (h : n = 4) : (n / 6) * 12 = 8 :=
by
  sorry

end number_div_mult_l615_615212


namespace unique_female_ages_l615_615844

noncomputable theory
open Classical

-- Define the types for citizens, ages, and the relationship "knows"
universe u
variable (Citizen : Type u)
variable (Age : Type)
variable [Inhabited Citizen] [Inhabited Age] [Nonempty Age]

-- Define the age function and the knows relationship
variable (age : Citizen → Age)
variable (knows : Citizen → Citizen → Prop)

-- Define the conditions given in the problem
variable (male : Citizen → Prop)
variable (female : Citizen → Prop)

axiom age_real : ∀ c : Citizen, age c ∈ ℝ
axiom knows_symm : ∀ {c1 c2 : Citizen}, knows c1 c2 → knows c2 c1
axiom graph_connected : ∀ c1 c2 : Citizen, c1 ≠ c2 → ∃ chain : List Citizen,
  chain.head = some c1 ∧ chain.reverse.head = some c2 ∧ ∀ᵢ (i : ℕ) (h : i < chain.length - 1), knows (chain.nth_le i h) (chain.nth_le (i+1) h)

axiom male_age_declared : ∀ c : Citizen, male c → ∃ a : Age, age c = a
axiom at_least_one_male : ∃ c : Citizen, male c
axiom female_age_average : ∀ {c : Citizen}, female c → age c = (Σ (c' : Citizen), (if knows c c' then age c' else 0)) / (Σ (c' : Citizen), if knows c c' then 1 else 0)

-- State the theorem to be proved
theorem unique_female_ages : 
  ∀ {c1 c2 : Citizen}, female c1 → female c2 → age c1 = age c2 :=
sorry

end unique_female_ages_l615_615844


namespace comparison_abc_l615_615086

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615086


namespace solve_inequality_l615_615575

theorem solve_inequality (x : ℝ) : 2 * x + 6 > 5 * x - 3 → x < 3 :=
by
  -- Proof steps would go here
  sorry

end solve_inequality_l615_615575


namespace find_a_b_find_m_l615_615800

noncomputable def f : ℝ → ℝ :=
λ x, x^2 - 4 * x + 1

theorem find_a_b (a b : ℝ) (h_pos : a > 0)
  (h_max : f 0 = 1) (h_min : f 1 = -2) :
  a = 1 ∧ b = a :=
by
  sorry

theorem find_m (m : ℝ)
  (h_ineq : ∀ x ∈ set.Icc (-1 : ℝ) 1, f x > -x + m) :
  m < -1 :=
by
  sorry

end find_a_b_find_m_l615_615800


namespace cubic_stone_weight_l615_615245

theorem cubic_stone_weight :
  ∀ (edge_length weight_per_dm : ℝ),  
  edge_length = 8 → weight_per_dm = 3.5 → 
  (edge_length ^ 3 * weight_per_dm = 1792) :=
begin
  sorry
end

end cubic_stone_weight_l615_615245


namespace union_complement_set_l615_615467

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615467


namespace find_a_l615_615766

noncomputable def constant_inverse_square_distance (a : ℝ) : Prop :=
  ∀ (P Q : ℝ × ℝ),
  (∃ t1 t2 : ℝ, (P.1 = a + t1 * Math.cos α) ∧ (P.2 = t1 * Math.sin α) ∧
                 ((P.2)^2 = 4 * P.1) ∧ (Q.1 = a + t2 * Math.cos α) ∧ (Q.2 = t2 * Math.sin α) ∧
                 ((Q.2)^2 = 4 * Q.1)) →
  (1 / ((P.1 - a)^2 + P.2^2) + 1 / ((Q.1 - a)^2 + Q.2^2)) = k

theorem find_a : ∃ a : ℝ, constant_inverse_square_distance a :=
by
  use 2
  sorry

end find_a_l615_615766


namespace monotonicity_extreme_values_tangent_lines_parallel_l615_615370

-- Problem Conditions
noncomputable def f (a x: ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

-- Theorem 1: f(x) is monotonically decreasing in (0, 1/a) and increasing in (1/a, 1) for a > 1.
theorem monotonicity (a : ℝ) (h : a > 1) : 
  (∀ x, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x, 1/a < x ∧ x < 1 → f a x > f a (1/a)) := 
  sorry

-- Theorem 2: Minimum and maximum values of f(x)
theorem extreme_values (a : ℝ) (h_nonneg : a > 0) (h_gt_one : a > 1) : 
  let x_min := 1/a
  let x_max := a 
  min f a x_min ∧ max f a x_max := 
  sorry

-- Theorem 3: There exist distinct points where the tangent lines at these points are parallel and x₁ + x₂ > 6/5 for a ≥ 3.
theorem tangent_lines_parallel (a : ℝ) (h : a ≥ 3) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (∂ x, f a x = 0 at x₁) ∧ (∂ x, f a x = 0 at x₂) ∧ x₁ + x₂ > 6/5 := 
  sorry

end monotonicity_extreme_values_tangent_lines_parallel_l615_615370


namespace log_base_decreasing_major_premise_incorrect_l615_615969

theorem log_base_decreasing (a : ℝ) (x : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a < 1) : 
  ∀ x, 0 < x → log a x < log a (x * 2) :=
begin
  sorry
end

theorem major_premise_incorrect : ¬(∀ a : ℝ, 0 < a ∧ a ≠ 1 → ∀ x : ℝ, 0 < x → log a x < log a (x * 2)) :=
begin
  -- Provide a specific counterexample when a = 1/2
  intro h,
  have h1 : 0 < (1/2) ∧ (1/2) ≠ 1 := by norm_num,
  specialize h (1/2) h1,
  have h_x: ∀ x, 0 < x → log (1/2) x < log (1/2) (x * 2) := h,
  -- Utilize a known fact about logarithm monotonicity for base 1/2
  have h_counterexample: ∃ x, 0 < x ∧ ¬(log (1/2) x < log (1/2) (x * 2)),
  { use 1,
    split,
    { norm_num },
    { norm_num,
      apply not_lt,
      calc log (1/2) 1 = 0 : by norm_num
                      ... ≥ -1 : by norm_num (at most)
      }, 
   },
  cases h_counterexample with x hx,
  exact hx.2 (h_x x hx.1),
end

end log_base_decreasing_major_premise_incorrect_l615_615969


namespace correct_option_is_A_l615_615629

theorem correct_option_is_A : 
    (√2 * √3 = √6) ∧ 
    ¬(2 + √2 = 2 * √2) ∧ 
    ¬(2 * √3 - 2 = √3) ∧ 
    ¬(√2 + √3 = √5) := 
by 
    sorry

end correct_option_is_A_l615_615629


namespace distance_S_from_origin_l615_615117

noncomputable def max_distance (z : ℂ) (hz : abs z = 1) : ℝ :=
  let w := (1 + Complex.i) * z + 3 * Complex.conj z
  sqrt (Complex.abs2 w)

theorem distance_S_from_origin (z : ℂ) (hz : abs z = 1) :
  max_distance z hz = sqrt 17 := by
  sorry

end distance_S_from_origin_l615_615117


namespace new_device_significant_improvement_l615_615240

def oldDeviceData := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def newDeviceData := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (lst : List ℝ) : ℝ := (lst.sum) / (lst.length)

noncomputable def variance (lst : List ℝ) : ℝ :=
  let m := mean lst
  (lst.map (λ x => (x - m) ^ 2)).sum / (lst.length)

def sampleMeanOld := mean oldDeviceData
def sampleMeanNew := mean newDeviceData

def sampleVarianceOld := variance oldDeviceData
def sampleVarianceNew := variance newDeviceData

noncomputable def significantImprovement : Prop :=
  (sampleMeanNew - sampleMeanOld) ≥ 2 * Real.sqrt ((sampleVarianceOld + sampleVarianceNew) / 10)

theorem new_device_significant_improvement : significantImprovement :=
  sorry

end new_device_significant_improvement_l615_615240


namespace angle_at_intersection_is_90_l615_615142

-- Define the setup: 
-- A regular dodecagon and points where sides extended intersect at P
structure RegularDodecagon :=
  (vertices : Fin 12 → ℝ × ℝ)
  (regular : ∀ i : Fin 12, angle (vertices i) (vertices (i + 1) % 12) = 150)

variable {d : RegularDodecagon}

/-- Given a regular dodecagon with points A, M and C, D extended to meet at P, the 
    measure of angle P is 90 degrees. -/
theorem angle_at_intersection_is_90 : 
  let A := d.vertices 0,
      M := d.vertices 5,
      C := d.vertices 2,
      D := d.vertices 3,
      P := extension_point A M C D in
  angle A M P = 90 :=
sorry

end angle_at_intersection_is_90_l615_615142


namespace unique_intersection_of_A_and_B_l615_615109

theorem unique_intersection_of_A_and_B (a : ℝ) (A := { x | 2^(1 + x) + 2^(1 - x) = a }) (B := {y | ∃ θ ∈ ℝ, y = Real.sin θ}) :
  (∃! x, A ∩ B x) ↔ a = 4 :=
by sorry

end unique_intersection_of_A_and_B_l615_615109


namespace sum_g_equals_half_l615_615751

noncomputable def g (n : ℕ) : ℝ :=
  ∑' k : ℕ, if k ≥ 3 then 1 / (k : ℝ) ^ n else 0

theorem sum_g_equals_half :
  ∑' n : ℕ, if n ≥ 2 then g n else 0 = 1 / 2 :=
sorry

end sum_g_equals_half_l615_615751


namespace solve_for_A_l615_615913

variable (A C : ℝ)
variable (C_ne_zero : C ≠ 0)
variable f g : ℝ → ℝ

def f (x : ℝ) : ℝ := A * x - 3 * C^2
def g (x : ℝ) : ℝ := C * x + 1

theorem solve_for_A : f (g 2) = 0 → A = 3 * C^2 / (2 * C + 1) :=
by
  intro h
  sorry

end solve_for_A_l615_615913


namespace total_population_l615_615857

variable (b g t : ℕ)

-- Conditions: 
axiom boys_to_girls (h1 : b = 4 * g) : Prop
axiom girls_to_teachers (h2 : g = 8 * t) : Prop

theorem total_population (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * b / 32 :=
sorry

end total_population_l615_615857


namespace intersection_sum_l615_615249

noncomputable def f (x : ℝ) : ℝ := 5 - (x - 1) ^ 2 / 3

theorem intersection_sum :
  ∃ a b : ℝ, f a = f (a - 4) ∧ b = f a ∧ a + b = 16 / 3 :=
sorry

end intersection_sum_l615_615249


namespace find_phi_even_function_l615_615797

theorem find_phi_even_function :
  ∀ (ϕ : ℝ), (0 < ϕ ∧ ϕ < π/2) →
    (∀ x : ℝ, 2 * sin (x + π/3 + ϕ) = 2 * sin (-x + π/3 + ϕ)) →
    ϕ = π/6 := by
  intro ϕ hϕ h_even
  sorry

end find_phi_even_function_l615_615797


namespace coffee_price_increase_l615_615965

theorem coffee_price_increase : 
  ∀ (P_high P_low : ℝ), P_high = 8 → P_low = 5 → ((P_high - P_low) / P_low) * 100 = 60 :=
begin
  intros P_high P_low H_high H_low,
  rw [H_high, H_low],
  norm_num,
  sorry
end

end coffee_price_increase_l615_615965


namespace compare_polynomials_l615_615681

theorem compare_polynomials (x : ℝ) : 2 * x^2 - 2 * x + 1 > x^2 - 2 * x := 
by
  sorry

end compare_polynomials_l615_615681


namespace ratio_of_pretzels_l615_615689

-- Definitions based on the conditions
def Barry_pretzels : ℕ := 12
def Shelly_pretzels : ℕ := Barry_pretzels / 2
def Angie_pretzels : ℕ := 18

-- Theorem statement proving the ratio
theorem ratio_of_pretzels : Angie_pretzels / Shelly_pretzels = 3 := by
  simp [Barry_pretzels, Shelly_pretzels, Angie_pretzels]
  sorry

end ratio_of_pretzels_l615_615689


namespace correct_b_values_l615_615321

-- Definitions and conditions
def does_not_divide (a b : ℕ) : Prop := ¬ (a ∣ b)
def divides (a b : ℕ) : Prop := a ∣ b

-- Statement of the problem
theorem correct_b_values (b : ℕ) :
  (∀ a : ℕ, a > 0 → does_not_divide a b → divides (a ^ a) (b ^ b)) ↔
  (¬ (is_prime_power b) ∧ ¬ (∃ (p q : ℕ), prime p ∧ prime q ∧ 2 * p > q ∧ q > p ∧ b = p * q)) :=
sorry

end correct_b_values_l615_615321


namespace distance_from_point_to_asymptote_l615_615973

-- Definitions for conditions
def point_P := (0 : ℝ, 1 : ℝ)

def hyperbola_eq (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

def asymptote1_eq (x y : ℝ) : Prop := 2 * x - y = 0
def asymptote2_eq (x y : ℝ) : Prop := 2 * x + y = 0

-- Function to calculate distance from a point to a line
def distance_to_line (a b c x0 y0 : ℝ) : ℝ := (a * x0 + b * y0 + c).abs / (real.sqrt (a^2 + b^2))

-- The theorem to prove
theorem distance_from_point_to_asymptote :
  distance_to_line 2 (-1) 0 0 1 = (real.sqrt 5) / 5 :=
sorry

end distance_from_point_to_asymptote_l615_615973


namespace solve_equation_l615_615147

theorem solve_equation (x y : ℝ) (k : ℤ) :
  x^2 - 2 * x * Real.sin (x * y) + 1 = 0 ↔ (x = 1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) ∨ (x = -1 ∧ y = (Real.pi / 2) + 2 * Real.pi * k) :=
by
  -- Logical content will be filled here, sorry is used because proof steps are not required.
  sorry

end solve_equation_l615_615147


namespace union_complement_set_l615_615461

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615461


namespace tangent_line_at_1_l615_615594

open Function

def f (x : ℝ) : ℝ := x^2

theorem tangent_line_at_1 :
  let f' := deriv f,
      f1 := f 1,
      f1_deriv := f' 1
  in f1 = 1 ∧ f1_deriv = 2 → (∀ x y : ℝ, y = 2 * (x - 1) + 1 ↔ 2 * x - y - 1 = 0) :=
by
  intros f' f1 f1_deriv h
  sorry

end tangent_line_at_1_l615_615594


namespace trajectory_of_M_max_area_OPQ_l615_615778

-- Definition of points A and B
def A : (ℝ × ℝ) := (-1, 0)
def B : (ℝ × ℝ) := (1, 0)

-- Condition k_MA * k_MB = -2
def k_MA (M : ℝ × ℝ) : ℝ := M.2 / (M.1 + 1)
def k_MB (M : ℝ × ℝ) : ℝ := M.2 / (M.1 - 1)
def k_condition (M : ℝ × ℝ) : Prop := k_MA M * k_MB M = -2

-- Problem (1)
theorem trajectory_of_M (M : ℝ × ℝ) (h : k_condition M) : M.1 ^ 2 + M.2 ^ 2 / 2 = 1 := 
sorry

-- Problem (2)
def F : (ℝ × ℝ) := (0, 1)
def triangle_area (k : ℝ) : ℝ := sqrt 2 * sqrt (k ^ 2 + 1) / (k ^ 2 + 2)
theorem max_area_OPQ : ∃ k : ℝ, triangle_area k = sqrt 2 / 2 := 
sorry

end trajectory_of_M_max_area_OPQ_l615_615778


namespace necessary_but_not_sufficient_l615_615349

-- Definitions of the lines
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, (a^2 - 2 * a) * x + y = 0
def line2 : ℝ → ℝ → Prop := λ x y, 3 * x + y + 1 = 0

-- Condition of the problem
def a_condition : ℝ := 3

-- Main theorem statement
theorem necessary_but_not_sufficient (a : ℝ) : 
  (a = a_condition → ∀ x y, line1 a x y → line2 x y) ∧
  (∀ x y, line1 a x y → line2 x y → a = a_condition) → (∃ a' : ℝ, ∀ x y, line1 a' x y → line2 x y ∧ ¬(a = a')) :=
begin
  sorry
end

end necessary_but_not_sufficient_l615_615349


namespace mass_percentage_Cl_in_HClO2_is_51_78_l615_615738

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_Cl : ℝ := 35.45
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_HClO2 : ℝ :=
  molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

noncomputable def mass_percentage_Cl_in_HClO2 : ℝ :=
  (molar_mass_Cl / molar_mass_HClO2) * 100

theorem mass_percentage_Cl_in_HClO2_is_51_78 :
  mass_percentage_Cl_in_HClO2 = 51.78 := 
sorry

end mass_percentage_Cl_in_HClO2_is_51_78_l615_615738


namespace sum_of_first_16_terms_l615_615992

noncomputable def sequence_sum := 
  let a : ℕ → ℚ := sorry -- placeholder for the sequence definition
  let condition : ∀ n ≥ 2, (a n + a (n-1) + 2) / 3 = ((-1 : ℚ)^n + 1) / 3 * a (n-1) + n := sorry
  S : ℕ → ℚ := λ n, (Finset.range n).sum (λ i, a i)
  let S_16 := S 16
  S_16 = 224

-- Mathematical statement:
theorem sum_of_first_16_terms :
  let a : ℕ → ℚ := sorry
  let condition : ∀ n ≥ 2, (a n + a (n-1) + 2) / 3 = ((-1)^n + 1) / 3 * a (n-1) + n := sorry
  let S : ℕ → ℚ := λ n, (Finset.range n).sum (λ i, a i)
  S 16 = 224 := 
sorry

end sum_of_first_16_terms_l615_615992


namespace blueberry_basket_count_l615_615188

noncomputable def number_of_blueberry_baskets 
    (plums_in_basket : ℕ) 
    (plum_baskets : ℕ) 
    (blueberries_in_basket : ℕ) 
    (total_fruits : ℕ) : ℕ := 
  let total_plums := plum_baskets * plums_in_basket
  let total_blueberries := total_fruits - total_plums
  total_blueberries / blueberries_in_basket

theorem blueberry_basket_count
  (plums_in_basket : ℕ) 
  (plum_baskets : ℕ) 
  (blueberries_in_basket : ℕ) 
  (total_fruits : ℕ)
  (h1 : plums_in_basket = 46)
  (h2 : plum_baskets = 19)
  (h3 : blueberries_in_basket = 170)
  (h4 : total_fruits = 1894) : 
  number_of_blueberry_baskets plums_in_basket plum_baskets blueberries_in_basket total_fruits = 6 := by
  sorry

end blueberry_basket_count_l615_615188


namespace truncated_pyramid_volume_l615_615274

theorem truncated_pyramid_volume :
  let unit_cube_vol := 1
  let tetrahedron_base_area := 1 / 2
  let tetrahedron_height := 1 / 2
  let tetrahedron_vol := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let two_tetrahedra_vol := 2 * tetrahedron_vol
  let truncated_pyramid_vol := unit_cube_vol - two_tetrahedra_vol
  truncated_pyramid_vol = 5 / 6 :=
by
  sorry

end truncated_pyramid_volume_l615_615274


namespace percentage_chess_swimming_l615_615186

def total_students : Nat := 1000
def chess_percentage : Real := 0.25
def swimming_students : Nat := 125

noncomputable def chess_students : Nat := (chess_percentage * total_students).toNat

theorem percentage_chess_swimming :
  (swimming_students.toReal / chess_students.toReal) * 100 = 50 :=
by
  sorry

end percentage_chess_swimming_l615_615186


namespace problem1_problem2_l615_615230

-- Problem 1: Prove that if \( x^{ \frac {1}{2}} + x^{- \frac {1}{2}} = 3 \), then \( x + x^{-1} = 7 \).
theorem problem1 (x : ℝ) (h: x^(1/2) + x^(-1/2) = 3) : x + x⁻¹ = 7 :=
by sorry

-- Problem 2: Prove that \( (\frac {1}{8})^{- \frac {1}{3}} - 3^{\log_{3}2}(\log_{3}4)\cdot(\log_{8}27) + 2\log_{ \frac {1}{6}} \sqrt {3} - \log_{6}2 = -3 \).
theorem problem2 : 
  ( (1/8)^( - (1/3) ) - 3^(real.log 2 / real.log 3) * (real.log 4 / real.log 3) * (real.log 27 / real.log 8) + 
    2 * (real.log (real.sqrt 3)/ real.log (1/6)) - real.log 2 / real.log 6 ) = -3 :=
by sorry

end problem1_problem2_l615_615230


namespace compare_abc_l615_615057

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615057


namespace find_solutions_l615_615320

theorem find_solutions : 
  { (a, b, c) // 0 < a ∧ 0 < b ∧ 0 < c ∧ 2^a + 2^b + 2^c = 2336 } = 
  { (11, 8, 5), (11, 5, 8), (8, 11, 5), (8, 5, 11), (5, 11, 8), (5, 8, 11) } :=
sorry

end find_solutions_l615_615320


namespace a_is_perfect_square_l615_615757

theorem a_is_perfect_square (a b : ℤ) (h : a = a^2 + b^2 - 8b - 2 * a * b + 16) : ∃ k : ℤ, a = k^2 :=
by
  sorry

end a_is_perfect_square_l615_615757


namespace union_complement_eq_target_l615_615483

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615483


namespace coplanar_vectors_l615_615645

variables (a b c : ℝ × ℝ × ℝ)
variable (x : ℝ)

def vectors := (1, -1, 3, -1, 4, -2, 1, 5, x)

theorem coplanar_vectors (a_eq : a = (1, -1, 3)) (b_eq : b = (-1, 4, -2)) (c_eq : c = (1, 5, x)) :
  are_coplanar a b c → x = 5 :=
by {
  sorry
}

end coplanar_vectors_l615_615645


namespace total_herd_l615_615246

theorem total_herd (n : ℕ) (h : n > 0) (h1 : (1 / 3 : ℚ) * n ∈ ℤ) (h2: (1 / 6 : ℚ) * n ∈ ℤ) (h3: (1 / 9 : ℚ) * n ∈ ℤ) (h4 : (2 / 9 : ℚ) * n = 11) :
  n = 54 :=
by
  sorry

end total_herd_l615_615246


namespace imaginary_part_of_z_l615_615763

noncomputable def i_unit : ℂ := complex.I

def z : ℂ := i_unit / (1 + i_unit) - 1 / (2 * i_unit)

theorem imaginary_part_of_z : complex.im z = 1 := 
by 
  sorry

end imaginary_part_of_z_l615_615763


namespace triangle_expression_l615_615422
noncomputable theory

def is_triangle (A B C : ℝ) (a b c : ℝ) (area : ℝ) : Prop :=
  A = 60 ∧ b = 1 ∧ (1/2 * b * c * real.sin A = area)

theorem triangle_expression (A B C a b c : ℝ) (area : ℝ) :
  is_triangle A B C a b c area → 
  ∀ (sinA sinB sinC : ℝ),
  sinA = real.sin A ∧
  sinB = real.sin B ∧
  sinC = real.sin C →
  (a + 2 * b - 3 * c)/(sinA + 2 * sinB - 3 * sinC) = 2 * real.sqrt (39) / 3 :=
begin
  sorry,
end

end triangle_expression_l615_615422


namespace find_initial_red_balloons_l615_615891

-- Define the initial state of balloons and the assumption.
def initial_blue_balloons : ℕ := 4
def red_balloons_after_inflation (R : ℕ) : ℕ := R + 2
def blue_balloons_after_inflation : ℕ := initial_blue_balloons + 2
def total_balloons (R : ℕ) : ℕ := red_balloons_after_inflation R + blue_balloons_after_inflation

-- Define the likelihood condition.
def likelihood_red (R : ℕ) : Prop := (red_balloons_after_inflation R : ℚ) / (total_balloons R : ℚ) = 0.4

-- Statement of the problem.
theorem find_initial_red_balloons (R : ℕ) (h : likelihood_red R) : R = 2 := by
  sorry

end find_initial_red_balloons_l615_615891


namespace lambda_value_l615_615831

section
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C P : V)

theorem lambda_value (hP_on_plane: P = (1/4 : ℝ) • (A - O) + λ • (B - O) + (1/8 : ℝ) • (C - O)) : λ = 5 / 8 :=
sorry
end

end lambda_value_l615_615831


namespace sqrt_inequality_abc_l615_615942

theorem sqrt_inequality_abc (a b c : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) (h3 : 1 ≤ c) (h4 : a + b + c = 9) : 
  real.sqrt (a * b + b * c + c * a) ≤ real.sqrt a + real.sqrt b + real.sqrt c :=
sorry

end sqrt_inequality_abc_l615_615942


namespace compare_abc_l615_615048

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615048


namespace nested_function_limit_l615_615113

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p * x + q

theorem nested_function_limit (p q : ℝ) (h : ∀ x ∈ set.Icc 4 6, |f x p q| ≤ 1/2) :
  let x0 := (9 - real.sqrt 19) / 2 in
  filter.tendsto (λ n, (λ x, f x p q)^n x0) filter.at_top (𝓝 ((9 + real.sqrt 19) / 2)) :=
sorry

end nested_function_limit_l615_615113


namespace angle_in_fourth_quadrant_l615_615646

theorem angle_in_fourth_quadrant (α : ℝ) (h : α = -75) : (∃ q, q = 4) :=
by
  use 4
  exact sorry

end angle_in_fourth_quadrant_l615_615646


namespace volume_of_rotated_solid_l615_615701

-- Definitions corresponding to the conditions in the problem
def vertical_strip_vol : ℝ := 7 * π
def horizontal_strip_vol : ℝ := 12 * π

-- The total volume calculation
def total_volume : ℝ := vertical_strip_vol + horizontal_strip_vol

-- The theorem to prove
theorem volume_of_rotated_solid : total_volume = 19 * π :=
by
  -- Definitions used here directly reflect the conditions; the proof step is omitted according to instructions.
  sorry

end volume_of_rotated_solid_l615_615701


namespace new_cooks_waiters_ratio_l615_615291

-- Definitions based on the conditions
variables (cooks waiters new_waiters : ℕ)

-- Given conditions
def ratio := 3
def initial_waiters := (ratio * cooks) / 3 -- Derived from 3 cooks / 11 waiters = 9 cooks / x waiters
def hired_waiters := 12
def total_waiters := initial_waiters + hired_waiters

-- The restaurant has 9 cooks
def restaurant_cooks := 9

-- Conclusion to prove
theorem new_cooks_waiters_ratio :
  (ratio = 3) →
  (restaurant_cooks = 9) →
  (initial_waiters = (ratio * restaurant_cooks) / 3) →
  (cooks = restaurant_cooks) →
  (waiters = initial_waiters) →
  (new_waiters = waiters + hired_waiters) →
  (new_waiters = 45) →
  (cooks / new_waiters = 1 / 5) :=
by
  intros
  sorry

end new_cooks_waiters_ratio_l615_615291


namespace probability_of_divisibility_by_5_is_zero_l615_615674

-- Define the spinner and its possible outcomes
def spinner := {1, 2, 3, 4}

-- Define what it means for the outcome to be a three-digit number
def three_digit_number := set (ℕ × ℕ × ℕ)

-- Define the total number of possible outcomes
def total_outcomes := 4 ^ 3

-- Define the favorable outcomes for divisibility by 5
def favorable_outcomes := 0

-- Define the probability as the ratio of favorable to total outcomes
def probability_divisible_by_5 := favorable_outcomes / total_outcomes

-- Prove that this probability is 0
theorem probability_of_divisibility_by_5_is_zero :
  probability_divisible_by_5 = 0 := by
  -- The probability is directly defined as 0 / 64 = 0
  rfl

end probability_of_divisibility_by_5_is_zero_l615_615674


namespace union_complement_set_l615_615468

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615468


namespace paint_cost_l615_615596

theorem paint_cost (l : ℝ) (b : ℝ) (rate : ℝ) (area : ℝ) (cost : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : l = 18.9999683334125) 
  (h3 : rate = 3.00001) 
  (h4 : area = l * b) 
  (h5 : cost = area * rate) : 
  cost = 361.00 :=
by
  sorry

end paint_cost_l615_615596


namespace union_complement_eq_l615_615500

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615500


namespace relation_among_abc_l615_615090

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615090


namespace problem_proof_l615_615774

noncomputable def problem_statement : Prop :=
∀ (ABC : Triangle) (γ : Circle) (H : Point) (M_A M_B M_C : Point),
  Orthocenter ABC H →
  Circumcircle ABC γ →
  Midpoint (BC ABC) M_A →
  Midpoint (CA ABC) M_B →
  Midpoint (AB ABC) M_C →
  OnCircle (reflect H (line BC ABC)) γ ∧
  OnCircle (reflect H (line CA ABC)) γ ∧
  OnCircle (reflect H (line AB ABC)) γ ∧
  OnCircle (reflect H M_A) γ ∧
  OnCircle (reflect H M_B) γ ∧
  OnCircle (reflect H M_C) γ

theorem problem_proof : problem_statement :=
by
  sorry

end problem_proof_l615_615774


namespace resulting_figure_has_25_sides_l615_615149

/-- Consider a sequential construction starting with an isosceles triangle, adding a rectangle 
    on one side, then a regular hexagon on a non-adjacent side of the rectangle, followed by a
    regular heptagon, another regular hexagon, and finally, a regular nonagon. -/
def sides_sequence : List ℕ := [3, 4, 6, 7, 6, 9]

/-- The number of sides exposed to the outside in the resulting figure. -/
def exposed_sides (sides : List ℕ) : ℕ :=
  let total_sides := sides.sum
  let adjacent_count := 2 + 2 + 2 + 2 + 1
  total_sides - adjacent_count

theorem resulting_figure_has_25_sides :
  exposed_sides sides_sequence = 25 := 
by
  sorry

end resulting_figure_has_25_sides_l615_615149


namespace union_complement_set_l615_615469

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615469


namespace number_of_people_in_group_l615_615251

theorem number_of_people_in_group (n : ℕ) :
  (∃ k : ℕ, k * n * (n - 1) = 440) → (n = 2 ∨ n = 5 ∨ n = 11) := 
begin
  sorry
end

end number_of_people_in_group_l615_615251


namespace prob_le_45_l615_615846

-- Define the probability conditions
def prob_between_1_and_45 : ℚ := 7 / 15
def prob_ge_1 : ℚ := 14 / 15

-- State the theorem to prove
theorem prob_le_45 : prob_between_1_and_45 = 7 / 15 := by
  sorry

end prob_le_45_l615_615846


namespace number_of_assignment_methods_l615_615243

theorem number_of_assignment_methods :
  let C (n k : ℕ) := Nat.choose n k
  let A (n k : ℕ) := Nat.perm n k
  C 5 3 * C 4 2 * A 3 3 = 360 :=
by {
  sorry
}

end number_of_assignment_methods_l615_615243


namespace compare_a_b_c_l615_615015

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615015


namespace find_numbers_l615_615591

-- Definitions for the conditions
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0
def difference_is_three (x y : ℕ) : Prop := x - y = 3

-- Statement of the proof problem
theorem find_numbers (x y : ℕ) (h1 : is_three_digit x) (h2 : is_even_two_digit y) (h3 : difference_is_three x y) :
  x = 101 ∧ y = 98 :=
sorry

end find_numbers_l615_615591


namespace T_sum_correct_l615_615724

-- Defining the sequence T_n
def T (n : ℕ) : ℤ := 
(-1)^n * 2 * n + (-1)^(n + 1) * n

-- Values to compute
def n1 : ℕ := 27
def n2 : ℕ := 43
def n3 : ℕ := 60

-- Sum of particular values
def T_sum : ℤ := T n1 + T n2 + T n3

-- Placeholder value until actual calculation
def expected_sum : ℤ := -42 -- Replace with the correct calculated result

theorem T_sum_correct : T_sum = expected_sum := sorry

end T_sum_correct_l615_615724


namespace aku_mother_packages_l615_615333

theorem aku_mother_packages
  (friends : Nat)
  (cookies_per_package : Nat)
  (cookies_per_child : Nat)
  (total_children : Nat)
  (birthday : Nat)
  (H_friends : friends = 4)
  (H_cookies_per_package : cookies_per_package = 25)
  (H_cookies_per_child : cookies_per_child = 15)
  (H_total_children : total_children = friends + 1)
  (H_birthday : birthday = 10) :
  (total_children * cookies_per_child) / cookies_per_package = 3 :=
by
  sorry

end aku_mother_packages_l615_615333


namespace sum_of_digits_of_n_equals_36_l615_615452

def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 20 }
def is_factor (n d : ℕ) := d ∣ n

noncomputable def N : ℕ := 2^3 * 3^2 * 5 * 7 * 11 * 13 * 19

theorem sum_of_digits_of_n_equals_36 :
  ( ∃ s : Set ℕ, s = S ) ∧
  ( ∃ n : ℕ, n = N ) ∧
  ( ∃ c : ℕ, c ∈ S ∧ is_factor N c ) ∧
  ( ∀ (d : ℕ), d ∉ S → let a := d in is_factor N a ) → 
    ( ∃ f1 f2 : ℕ, (f1 ∈ S ∧ f2 ∈ S ∧ f1 ≠ f2 ∧ abs (f1 - f2) = 1) ∧
            ¬ is_factor N f1 ∧ ¬ is_factor N f2 ) →
    ( Nat.digits 10 N ).sum = 36 := by
  sorry

end sum_of_digits_of_n_equals_36_l615_615452


namespace first_discount_l615_615279

theorem first_discount (P F : ℕ) (D₂ : ℝ) (D₁ : ℝ) 
  (hP : P = 150) 
  (hF : F = 105)
  (hD₂ : D₂ = 12.5)
  (hF_eq : F = P * (1 - D₁ / 100) * (1 - D₂ / 100)) : 
  D₁ = 20 :=
by
  sorry

end first_discount_l615_615279


namespace counterexample_statement1_counterexample_statement2_statement3_is_true_counterexample_statement4_counterexample_statement6_l615_615880

theorem counterexample_statement1 : ∃ student : Type, ∃ textbook : student, has_corrections_and_markings textbook :=
by
  sorry

theorem counterexample_statement2 : ∃ n : ℕ, n % 5 = 0 ∧ n % 10 ≠ 5 :=
by
  sorry

theorem statement3_is_true : ∀ n : ℕ, even (n * (n + 1)) :=
by
  sorry

theorem counterexample_statement4 : ∃ x : ℝ, x^2 ≤ x :=
by
  sorry

theorem counterexample_statement6 : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 :=
by
  sorry

end counterexample_statement1_counterexample_statement2_statement3_is_true_counterexample_statement4_counterexample_statement6_l615_615880


namespace line_through_PQ_eq_yx_circle_C_eq_l615_615338

open_locale real

-- Definition of points P and Q
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (0, 1)

-- Given line intersecting circle at P and Q
theorem line_through_PQ_eq_yx :
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, (l x = y) ↔ (x + y = 1) :=
sorry

-- Given radius of circle is 1
def radius_C_eq_1 : ℝ := 1

-- Definition of Circle C passing through points P and Q with radius 1
theorem circle_C_eq :
  (∃ a b : ℝ, ∀ x y : ℝ, ((x-a)^2 + (y-b)^2 = radius_C_eq_1^2) ∧ ((1-a)^2 + b^2 = radius_C_eq_1^2) ∧ (a^2 + (b-1)^2 = radius_C_eq_1^2)) ↔
  (∀ x y : ℝ, ((x^2 + y^2 = 1) ∨ (x^2 + y^2 - 2*x - 2*y + 1 = 0))) :=
sorry

end line_through_PQ_eq_yx_circle_C_eq_l615_615338


namespace union_complement_eq_target_l615_615484

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615484


namespace intersection_complement_l615_615816

open Set

def U : Set ℤ := univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_complement :
  P ∩ (U \ M) = {-2, -1, 0} :=
by
  sorry

end intersection_complement_l615_615816


namespace longest_side_range_of_obtuse_triangle_l615_615406

theorem longest_side_range_of_obtuse_triangle (a b c : ℝ) (h₁ : a = 1) (h₂ : b = 2) :
  a^2 + b^2 < c^2 → (Real.sqrt 5 < c ∧ c < 3) ∨ c = 2 :=
by
  sorry

end longest_side_range_of_obtuse_triangle_l615_615406


namespace compare_abc_l615_615031

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615031


namespace intersection_A_B_l615_615377

def A : Set Int := {-1, 0, 1, 5, 8}
def B : Set Int := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} :=
by
  sorry

end intersection_A_B_l615_615377


namespace regular_polygon_sides_l615_615265

theorem regular_polygon_sides (P s : ℕ) (hP : P = 150) (hs : s = 15) :
  P / s = 10 :=
by
  sorry

end regular_polygon_sides_l615_615265


namespace evaluate_expression_l615_615727

theorem evaluate_expression :
  - (20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 :=
by
  sorry

end evaluate_expression_l615_615727


namespace union_complement_eq_l615_615504

open Set -- Opening the Set namespace for easier access to set operations

variable (U M N : Set ℕ) -- Introducing the sets as variables

-- Defining the universal set U, set M, and set N in terms of natural numbers
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- Statement to prove
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end union_complement_eq_l615_615504


namespace largest_of_7_consecutive_numbers_with_average_20_l615_615963

variable (n : ℤ) 

theorem largest_of_7_consecutive_numbers_with_average_20
  (h_avg : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6))/7 = 20) : 
  (n + 6) = 23 :=
by
  -- Placeholder for the actual proof
  sorry

end largest_of_7_consecutive_numbers_with_average_20_l615_615963


namespace algebraic_expressions_same_terms_l615_615396

theorem algebraic_expressions_same_terms (a b : ℝ) (x y : ℤ) 
  (H1 : 3 * a ^ (x + 7) * b ^ 4 = -a ^ 4 * b ^ (2 * y)) : x^y = 9 :=
begin
  sorry
end

end algebraic_expressions_same_terms_l615_615396


namespace intervals_of_monotonicity_f_range_of_k_l615_615805

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log (x + 1)
def g (x : ℝ) : ℝ := Real.exp x - x - 1

-- Given condition: The tangents at the origin are the same
axiom tangent_at_origin_eq (f g : ℝ → ℝ) (a : ℝ) : 
  deriv (f a) 0 = deriv g 0

-- (Ⅰ) Prove the intervals of monotonicity for f(x)
theorem intervals_of_monotonicity_f (a x : ℝ) (h : tangent_at_origin_eq f g a) : 
  (a = 1 → 
   ((∀ x, x ∈ Ioo (-1 : ℝ) 0 → deriv (f a) x < 0) ∧ 
    (∀ x, x ∈ Ioo 0 Real.Infty → deriv (f a) x > 0))) := 
  sorry

-- (Ⅱ) Prove the range of k such that ∀ x ≥ 0, g(x) ≥ k * f(x)
theorem range_of_k (a k : ℝ) (h : tangent_at_origin_eq f g a) : 
  (g 0 ≥ k * (f a) 0) → k ≤ 1 :=
  sorry

end intervals_of_monotonicity_f_range_of_k_l615_615805


namespace factorial_div_sub_factorial_equality_l615_615625

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => (n + 1) * factorial n

theorem factorial_div_sub_factorial_equality :
  (factorial 12 - factorial 11) / factorial 10 = 121 :=
by
  sorry

end factorial_div_sub_factorial_equality_l615_615625


namespace find_value_f1_plus_2_fprime1_l615_615789

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h : ∀ x, f' x = 1/2)

noncomputable def tangent_line_at_1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

theorem find_value_f1_plus_2_fprime1
  (hf : f 1 = 1)
  (hf' : f' 1 = 1/2)
  (htangent : tangent_line_at_1 (1 : ℝ) (f 1)) :
  f 1 + 2 * f' 1 = 2 :=
by {
  rw [hf, hf'],
  norm_num,
}

end find_value_f1_plus_2_fprime1_l615_615789


namespace inequality_sqrt_l615_615134

theorem inequality_sqrt (a b c : ℝ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  sqrt (1 / a - a) + sqrt (1 / b - b) + sqrt (1 / c - c) ≥ sqrt (2 * a) + sqrt (2 * b) + sqrt (2 * c) :=
begin
  sorry
end

end inequality_sqrt_l615_615134


namespace ratio_female_to_male_l615_615289

variable {f m c : ℕ}

/-- 
  The following conditions are given:
  - The average age of female members is 35 years.
  - The average age of male members is 30 years.
  - The average age of children members is 10 years.
  - The average age of the entire membership is 25 years.
  - The number of children members is equal to the number of male members.
  We need to show that the ratio of female to male members is 1.
-/
theorem ratio_female_to_male (h1 : c = m)
  (h2 : 35 * f + 40 * m = 25 * (f + 2 * m)) :
  f = m :=
by sorry

end ratio_female_to_male_l615_615289


namespace find_arithmetic_sequence_find_t_range_l615_615775

variables {a b c : ℕ → ℝ}

def arithmetic_sequence (a d : ℕ → ℝ) (a_n : ℕ → ℝ) :=
  ∀ n : ℕ, a_n n = a 1 + ((n - 1) * d 0)

def sum (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :=
  ∀ n : ℕ, S_n n = (n * (a_n n + a_n 1)) / 2

theorem find_arithmetic_sequence 
  (S_9 : ℕ → ℝ)
  (S_15 : ℕ → ℝ) 
  (h1 : S_9 9 = 90)
  (h2 : S_15 15 = 240) :
  ∃ a_n S_n, ∀ n, (S_n n = n * (n + 1)) ∧ (a_n n = 2 * n) :=
  sorry

theorem find_t_range 
  (a_n b_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ)
  (h3 : ∀ n, a_n n * b_n n = 1 / (n + 1))
  (h4 : ∀ n, S_n n = (1 / 2) * (1 - 1 / (n + 1)))
  (h5 : ∀ n, S_n n < t) :
  ∃ t : ℝ, t ≥ (1 / 2) :=
  sorry

end find_arithmetic_sequence_find_t_range_l615_615775


namespace shortest_distance_correct_l615_615180

noncomputable def shortest_distance : ℝ :=
  let curve (x : ℝ) := real.log (2*x - 1)
  let dist (x1 y1 : ℝ) (a b c : ℝ) := abs (a*x1 + b*y1 + c) / real.sqrt (a^2 + b^2)
  let x_tangent := 1 -- from solution step
  let point := (x_tangent, curve x_tangent)
  dist (fst point) (snd point) 2 (-1) 3

theorem shortest_distance_correct :
  shortest_distance = real.sqrt 5 :=
begin
  sorry
end

end shortest_distance_correct_l615_615180


namespace find_function_form_l615_615709

-- Defining the set of nonzero integers
def ℤ_star := {z : ℤ // z ≠ 0}

-- Defining the type of functions from ℤ* to ℕ₀
def func := ℤ_star → ℕ

-- Definition of the main conditions
variable (f : func)

axiom cond1 : ∀ (a b : ℤ_star), ((a.val + b.val) ≠ 0) → f ⟨a.val + b.val, by linarith [a.property, b.property]⟩ ≥ min (f a) (f b)
axiom cond2 : ∀ (a b : ℤ_star), f ⟨a.val * b.val, by linarith [a.property, b.property]⟩ = f a + f b

-- Definition of v_p function
def v_p (p : ℕ) (n : ℤ_star) : ℕ :=
  if h : p ∣ n.val
  then Nat.find $ (exists_pow_dvd $ p) n.val h
  else 0

-- The theorem statement
theorem find_function_form :
  ∃ (p : ℕ) (k : ℕ), ∀ (n : ℤ_star), f n = k * v_p p n :=
sorry

end find_function_form_l615_615709


namespace mary_investment_l615_615930

noncomputable def compoundInterest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem mary_investment :
  ∀ (P : ℝ) (r : ℝ) (n t : ℕ),
    r = 0.05 → n = 12 → t = 10 →
    (compoundInterest P r n t) = 80000 → 
    P = 48_563 :=
by 
  intros P r n t hr hn ht hA
  sorry

end mary_investment_l615_615930


namespace functional_equation_solution_l615_615730

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (f (xy - x)) + f (x + y) = y * f (x) + f (y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end functional_equation_solution_l615_615730


namespace range_of_k_l615_615306

noncomputable def H (b : ℕ → ℝ) (n : ℕ) : ℝ :=
(b 1 + ∑ i in range (n - 1), 2^i * b (i + 2)) / n

theorem range_of_k
  (H : (ℕ → ℝ) → ℕ → ℝ)
  (H_n_def : ∀ (b : ℕ → ℝ) (n : ℕ), H b n = 2^(n + 1))
  (S : (ℕ → ℝ) → ℝ → ℕ → ℝ)
  (b : ℕ → ℝ)
  (k : ℝ)
  (S_n_leq_S3 : ∀ (n : ℕ), S b k n ≤ S b k 3) :
  7 / 3 ≤ k ∧ k ≤ 12 / 5 :=
sorry

end range_of_k_l615_615306


namespace relation_among_abc_l615_615095

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615095


namespace inserted_15_middle_is_perfect_square_l615_615132

theorem inserted_15_middle_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, (\underbrace{33 \ldots 34}_{n \text{ digits}}) = k^2 :=
begin
  -- Define T_n
  let T_n := (10^n - 1) / 9,
  -- Define the number pattern
  let num := 3 * T_n + 4,
  -- Assert that num is a perfect square
  use 10^n + 1,
  sorry
end

end inserted_15_middle_is_perfect_square_l615_615132


namespace na_h_moles_required_l615_615733

theorem na_h_moles_required :
  ∀ (H2O moles : ℕ),
  (∀ n, (1 * n = 1 * n) ∧ n = 2) → (H2O = 2) → 
  H2O = 2 :=
begin
  sorry
end

end na_h_moles_required_l615_615733


namespace pyramid_volume_l615_615855

-- Defining the setup for the pyramid and its properties
structure Pyramid :=
  (S A B C E F L M : ℝ → ℝ → ℝ)
  (side_length_base : ℝ)
  (length_lateral_edges : ℝ)
  (midpoint_E : S E = 0.5 * (S + E))
  (condition_L : |AL| = (1 / 10) * |AC|)
  (iso_trapezoid : is_isosceles_trapezoid E F L M)
  (EF_length : |EF| = √7)

-- The theorem to prove the volume of the pyramid
theorem pyramid_volume (p : Pyramid)
  (h1 : p.E = (p.S + p.B + p.C) / 2)
  (h2 : |p.F - p.A| = (1/10) * |p.C - p.A|)
  (h3 : is_isosceles_trapezoid p.E p.F p.L p.M)
  (h4 : |p.E - p.F| = √7) :
  calculate_pyramid_volume p.S p.A p.B p.C = (16/3) * √13 :=
sorry

end pyramid_volume_l615_615855


namespace initial_percentage_proof_l615_615654

noncomputable def initialPercentageAntifreeze (P : ℝ) : Prop :=
  let initial_fluid : ℝ := 4
  let drained_fluid : ℝ := 2.2857
  let added_antifreeze_fluid : ℝ := 2.2857 * 0.8
  let final_percentage : ℝ := 0.5
  let final_fluid : ℝ := 4
  
  let initial_antifreeze : ℝ := initial_fluid * P
  let drained_antifreeze : ℝ := drained_fluid * P
  let total_antifreeze_after_replacement : ℝ := initial_antifreeze - drained_antifreeze + added_antifreeze_fluid
  
  total_antifreeze_after_replacement = final_fluid * final_percentage

-- Prove that the initial percentage is 0.1
theorem initial_percentage_proof : initialPercentageAntifreeze 0.1 :=
by
  dsimp [initialPercentageAntifreeze]
  simp
  exact sorry

end initial_percentage_proof_l615_615654


namespace compare_a_b_c_l615_615021

noncomputable def a : ℝ := 2 * Real.ln 1.01
noncomputable def b : ℝ := Real.ln 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615021


namespace meaningful_expression_range_l615_615397

theorem meaningful_expression_range (x : ℝ) : (¬ (x - 1 = 0)) ↔ (x ≠ 1) := 
by
  sorry

end meaningful_expression_range_l615_615397


namespace choose_7_starters_with_at_least_one_quadruplet_l615_615552

-- Given conditions
variable (n : ℕ := 18) -- total players
variable (k : ℕ := 7)  -- number of starters
variable (q : ℕ := 4)  -- number of quadruplets

-- Lean statement
theorem choose_7_starters_with_at_least_one_quadruplet 
  (h : n = 18) 
  (h1 : k = 7) 
  (h2 : q = 4) :
  (Nat.choose 18 7 - Nat.choose 14 7) = 28392 :=
by
  sorry

end choose_7_starters_with_at_least_one_quadruplet_l615_615552


namespace solve_problem_l615_615419

theorem solve_problem
    (product_trailing_zeroes : ∃ (x y z w v u p q r : ℕ), (10 ∣ (x * y * z * w * v * u * p * q * r)) ∧ B = 0)
    (digit_sequences : (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) % 10 = 8 ∧
                       (11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19) % 10 = 4 ∧
                       (21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29) % 10 = 4 ∧
                       (31 * 32 * 33 * 34 * 35) % 10 = 4 ∧
                       A = 2 ∧ B = 0)
    (divisibility_rule_11 : ∀ C D, (71 + C) - (68 + D) = 11 → C - D = -3 ∨ C - D = 8)
    (divisibility_rule_9 : ∀ C D, (139 + C + D) % 9 = 0 → C + D = 5 ∨ C + D = 14)
    (system_of_equations : ∀ C D, (C - D = -3 ∧ C + D = 5) → (C = 1 ∧ D = 4)) :
  A = 2 ∧ B = 0 ∧ C = 1 ∧ D = 4 :=
by
  sorry

end solve_problem_l615_615419


namespace compare_abc_l615_615053

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615053


namespace probability_of_drawing_2_black_and_2_white_l615_615238

def total_balls : ℕ := 17
def black_balls : ℕ := 9
def white_balls : ℕ := 8
def balls_drawn : ℕ := 4
def favorable_outcomes := (Nat.choose 9 2) * (Nat.choose 8 2)
def total_outcomes := Nat.choose 17 4
def probability_draw : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_drawing_2_black_and_2_white :
  probability_draw = 168 / 397 :=
by
  sorry

end probability_of_drawing_2_black_and_2_white_l615_615238


namespace union_complement_eq_l615_615506

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615506


namespace find_original_sum_of_money_l615_615636

theorem find_original_sum_of_money
  (R : ℝ)
  (P : ℝ)
  (h1 : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 63) :
  P = 2100 :=
sorry

end find_original_sum_of_money_l615_615636


namespace relation_among_abc_l615_615093

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615093


namespace tailwind_speed_l615_615666

-- Define the given conditions
def plane_speed_with_wind (P W : ℝ) : Prop := P + W = 460
def plane_speed_against_wind (P W : ℝ) : Prop := P - W = 310

-- Theorem stating the proof problem
theorem tailwind_speed (P W : ℝ) 
  (h1 : plane_speed_with_wind P W) 
  (h2 : plane_speed_against_wind P W) : 
  W = 75 :=
sorry

end tailwind_speed_l615_615666


namespace union_complement_eq_l615_615471

open Set  -- Open the Set namespace

-- Define the universal set U
def U := {0, 1, 2, 4, 6, 8}

-- Define the set M
def M := {0, 4, 6}

-- Define the set N
def N := {0, 1, 6}

-- Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8}
theorem union_complement_eq :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_complement_eq_l615_615471


namespace compare_a_b_c_l615_615034

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615034


namespace unique_v_star_l615_615638

theorem unique_v_star {v : ℝ} (h1 : v * = v - v / 3) (h2 : (v * - v / 3) * = 12) : v = 27 :=
sorry

end unique_v_star_l615_615638


namespace arrangement_count_l615_615663

-- Definitions corresponding to the conditions in a)
def num_students : ℕ := 8
def max_per_activity : ℕ := 5

-- Lean statement reflecting the target theorem in c)
theorem arrangement_count (n : ℕ) (max : ℕ) 
  (h1 : n = num_students)
  (h2 : max = max_per_activity) :
  ∃ total : ℕ, total = 182 :=
sorry

end arrangement_count_l615_615663


namespace min_P_inf_P_l615_615451

noncomputable def P (X1 X2 : ℝ) : ℝ :=
  X1^2 + (1 - X1 * X2)^2

theorem min_P (X1 X2 : ℝ) : ∃ ε > 0, ∀ x ∈ metric.ball (0 : ℝ) ε, P x.1 x.2 > 0 := sorry

theorem inf_P : real.Inf (set.image (λ (x : ℝ × ℝ), P x.1 x.2) set.univ) = 0 := sorry

end min_P_inf_P_l615_615451


namespace solution_set_of_inequality_l615_615604

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l615_615604


namespace TU_squared_1090_l615_615577

noncomputable def proof_TU_squared : Prop :=
  ∀ (P Q R S T U : ℝ) (a b : ℂ),
    (a = 14) ∧ (b = 15) ∧ (P = 7) ∧ (Q = 14) ∧ (R = 7) ∧ (S = 14) ∧
    (T = Real.cos(β)) ∧ (U = Real.sin(β))
    
theorem TU_squared_1090 
  (PQRS_is_square : ∀ (PQ RS QR PS : ℝ),
    PQ = RS ∧ RS = QR ∧ QR = PS ∧ PS = PQ)
  (side_length_PQRS : PQRS_is_square 15 15 15 15)
  (PT_distance : ∀ (P T : ℝ), ∀ (dist_PT : T = 7), dist_PT )
  (RU_distance : ∀ (R U : ℝ), ∀ (dist_RU : U = 7), dist_RU)
  (QT_distance : ∀ (Q T : ℝ), ∀ (dist_QT : T = 14), dist_QT)
  (SU_distance : ∀ (S U : ℝ), ∀ (dist_SU : U = 14), dist_SU): 
  TU^2 = 1090 :=
by
  sorry

end TU_squared_1090_l615_615577


namespace compare_abc_l615_615050

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615050


namespace stretched_curve_l615_615164

noncomputable def transformed_curve (x : ℝ) : ℝ :=
  2 * Real.sin (x / 3 + Real.pi / 3)

theorem stretched_curve (y x : ℝ) :
  y = 2 * Real.sin (x + Real.pi / 3) → y = transformed_curve x := by
  intro h
  sorry

end stretched_curve_l615_615164


namespace functional_relationship_l615_615612

def electricity_bill (x : ℝ) : ℝ :=
  if x ≤ 200 then 0.6 * x
  else 1.1 * x - 100

theorem functional_relationship (x : ℝ) (y : ℝ) :
  (0 ≤ x ∧ x ≤ 200 ∧ y = 0.6 * x) ∨ (200 < x ∧ y = 1.1 * x - 100) :=
sorry

end functional_relationship_l615_615612


namespace largest_rectangle_in_triangle_proof_l615_615665

-- Define the vertices and sides of the triangle
variables {A B C D E F G H : Type*}
          [AffineSpace ℝ A]
          [AffineSpace ℝ B]
          [Nonempty ℝ C]

-- Define a predicate indicating the midpoint of a line segment
def is_midpoint (P Q R : AffineSpace ℝ) : Prop :=
  ∃ M, is_midpoint_points P M Q ∧ is_midpoint_points Q M R

-- Define a predicate indicating perpendicularity
def is_perpendicular (P Q R S : AffineSpace ℝ) : Prop :=
  ∃ QP, ∃ QR, QP ≠ QR ∧ orthogonal QP QR

-- Axiom indicating the sides of the lawn triangle are proportional to the reference triangle's sides
axiom sides_proportional_to_reference_triangle : Prop

-- Axiom indicating the largest rectangle inside a triangle
axiom largest_rectangle_in_triangle (A B C D E F G H : AffineSpace ℝ) :
  is_midpoint A B D ∧ is_midpoint B C E ∧ is_midpoint C A F ∧
  is_perpendicular D G A ∧ is_perpendicular E H B →
  largest_rectangle_vertices A B G H

-- The proof statement
theorem largest_rectangle_in_triangle_proof :
  is_midpoint A B D ∧ is_midpoint B C E ∧ is_midpoint C A F ∧
  is_perpendicular D G A ∧ is_perpendicular E H B →
  largest_rectangle_vertices A B G H :=
begin
  sorry
end

end largest_rectangle_in_triangle_proof_l615_615665


namespace problem_1_problem_2_problem_3_l615_615954

-- Problem 1: Proving the solution for (x^2 + 2) * |2x - 5| = 0
theorem problem_1 (x : ℝ) : (x^2 + 2) * |2x - 5| = 0 ↔ x = 5 / 2 :=
begin
  sorry
end

-- Problem 2: Proving the solutions for (x - 3)^3 * x = 0
theorem problem_2 (x : ℝ) : (x - 3)^3 * x = 0 ↔ x = 0 ∨ x = 3 :=
begin
  sorry
end

-- Problem 3: Proving the solution for |x^4 + 1| = x^4 + x
theorem problem_3 (x : ℝ) : |x^4 + 1| = x^4 + x ↔ x = 1 :=
begin
  sorry
end

end problem_1_problem_2_problem_3_l615_615954


namespace sum_of_solutions_eq_one_l615_615744

theorem sum_of_solutions_eq_one : 
  (∀ x : ℝ, (x ≠ 2 ∧ x ≠ -2) → (frac (-12 * x) (x^2 - 4) = (frac (3 * x) (x + 2) - frac 9 (x - 2)))
  → x = 3 ∨ x = -2) 
  → (3 + (-2) = 1) :=
by 
  sorry

end sum_of_solutions_eq_one_l615_615744


namespace johann_mail_delivery_l615_615432

-- Definitions of given conditions
def total_pieces : ℕ := 250
def friend1 : ℕ := 35
def friend2 : ℕ := 42
def friend3 : ℕ := 38
def friend4 : ℕ := 45
def total_delivered_by_friends : ℕ := friend1 + friend2 + friend3 + friend4 -- Sum delivered by friends

-- Problem: Calculate the number of pieces Johann needs to deliver
theorem johann_mail_delivery : (250 - (35 + 42 + 38 + 45)) = 90 :=
by
  unfold total_pieces friend1 friend2 friend3 friend4 total_delivered_by_friends
  norm_num

-- Add sorry to indicate the proof step is omitted

end johann_mail_delivery_l615_615432


namespace right_triangle_side_length_l615_615444

theorem right_triangle_side_length
  (A B C P Q : ℝ)
  (AP BQ : ℝ)
  (h1 : ∠BAC = 90)
  (h2 : P = (A + B) / 2)
  (h3 : Q = (A + C) / 2)
  (h4 : BQ = 20)
  (h5 : AP = 15) :
  BC = 10 * sqrt 5 := 
sorry

end right_triangle_side_length_l615_615444


namespace boatworks_total_canoes_l615_615293

theorem boatworks_total_canoes : 
  let jan := 2 in
  let feb := 3 * jan in
  let mar := 3 * feb in
  let apr := 3 * mar in
  jan + feb + mar + apr = 80 := 
by
  sorry

end boatworks_total_canoes_l615_615293


namespace min_value_expression_l615_615740

noncomputable def minValue : ℝ :=
  (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_expression (x : ℝ) : 
  ∃ x : ℝ, minValue x = -784 :=
sorry

end min_value_expression_l615_615740


namespace minimum_value_m_l615_615387

theorem minimum_value_m (x0 : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 1| ≤ m) → m = 2 :=
by
  sorry

end minimum_value_m_l615_615387


namespace compare_abc_l615_615049

def a : ℝ := 2 * Real.log 1.01
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b :=
by
  /- Proof goes here -/
  sorry

end compare_abc_l615_615049


namespace hyperbola_solution_l615_615808

noncomputable def hyperbola_equation (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (4 * b^2 = 3 * a^2) ∧ (a^2 + b^2 = 7) 

theorem hyperbola_solution : 
  ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ hyperbola_equation a b ha hb → 
  (a^2 = 4 ∧ b^2 = 3) ∧ (a^2 = 4 ∧ b^2 = 3) ∧ (by exact (frac (x^2) (4) - frac (y^2) (3)) = 1) :=
begin
 sorry
end

end hyperbola_solution_l615_615808


namespace varphi_value_l615_615366

theorem varphi_value (f : ℝ → ℝ) (g : ℝ → ℝ) (φ : ℝ) (h₁ : -π ≤ φ ∧ φ < π) 
  (h₂ : ∀ x, f (x - π / 2) = g x) 
  (h₃ : ∀ x, g x = sin x * cos x + (sqrt 3 / 2) * cos x) : 
  |φ| = 5 * π / 6 := 
sorry

end varphi_value_l615_615366


namespace four_positive_reals_inequality_l615_615557

theorem four_positive_reals_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + b^2 * c + c^2 * d + d^2 * a :=
sorry

end four_positive_reals_inequality_l615_615557


namespace max_sum_arithmetic_sequence_l615_615165

theorem max_sum_arithmetic_sequence :
  (∃ (n : ℕ), ∀ (S_n : ℕ → ℤ),
    (S_n := λ n, 23 * n + (n * (n-1) / 2) * (-2)) →
    (∃ (n : ℕ), S_n n = 144)) :=
by
  sorry

end max_sum_arithmetic_sequence_l615_615165


namespace cos_monotonic_decreasing_interval_l615_615597

theorem cos_monotonic_decreasing_interval : 
  ∃ (a b : ℝ), a = 0 ∧ b = π ∧ ∀ x, 0 ≤ x ∧ x ≤ π → ∀ y, 0 ≤ y ∧ y ≤ π → x ≤ y → cos x ≥ cos y :=
sorry

end cos_monotonic_decreasing_interval_l615_615597


namespace solve_for_x_l615_615746

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) → x = 3 / 2 := by
  sorry

end solve_for_x_l615_615746


namespace remainder_of_S_divided_by_500_l615_615453

noncomputable def is_perfect_square (x : Int) : Prop := ∃ k : Int, k^2 = x

theorem remainder_of_S_divided_by_500 :
  let S : Int := 1005 + 39 in
  let remainder := S % 500 in
  remainder = 44 :=
by
  let S := 1005 + 39
  let remainder := S % 500
  have : S = 1044 := by sorry
  have : remainder = 44 := by sorry
  exact this

end remainder_of_S_divided_by_500_l615_615453


namespace irrational_product_rational_l615_615218

noncomputable def is_rational (r : ℚ) : Prop := true  -- Every rational is rational, trivial

noncomputable def is_irrational (a : ℝ) : Prop := ¬∃ (r : ℚ), a = r

theorem irrational_product_rational :
  ∃ (a b : ℝ), is_irrational a ∧ is_irrational b ∧ a ≠ b ∧ is_rational (a * b) :=
by
  use (Real.sqrt 3 + Real.sqrt 2)
  use (Real.sqrt 3 - Real.sqrt 2)
  split
  · sorry -- proof of a being irrational
  · split
    · sorry -- proof of b being irrational
    · split
      · sorry -- proof of inequality a ≠ b
      · sorry -- proof of (a * b) being rational

end irrational_product_rational_l615_615218


namespace compare_abc_l615_615061

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615061


namespace largest_of_consecutive_numbers_l615_615961

theorem largest_of_consecutive_numbers (avg : ℕ) (n : ℕ) (h1 : n = 7) (h2 : avg = 20) :
  let sum := n * avg in
  let middle := sum / n in
  let largest := middle + 3 in
  largest = 23 :=
by
  -- Introduce locals to use 
  let sum := n * avg
  let middle := sum / n
  let largest := middle + 3
  -- Add the proof placeholder
  sorry

end largest_of_consecutive_numbers_l615_615961


namespace folklore_functional_equation_l615_615319

theorem folklore_functional_equation (f : ℝ+ → ℝ+) 
  (h : ∀ x y : ℝ+, f (f x + y) = x + f y) : 
  ∀ x : ℝ+, f x = x :=
by
  sorry

end folklore_functional_equation_l615_615319


namespace max_distance_ellipse_line_l615_615328

theorem max_distance_ellipse_line :
  let P (α : ℝ) := (4 * Real.cos α, 2 * Real.sin α)
  let d (α : ℝ) := abs((4 * Real.cos α + 4 * Real.sin α - Real.sqrt 2) / Real.sqrt 5)
  ∃ α₀ : ℝ, d α₀ = Real.sqrt 10 :=
begin
  sorry
end

end max_distance_ellipse_line_l615_615328


namespace range_of_a_l615_615369

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - (x^2 / 2) - a * x - 1

theorem range_of_a (x : ℝ) (a : ℝ) (h : 1 ≤ x) : (0 ≤ f a x) → (a ≤ Real.exp 1 - 3 / 2) :=
by
  sorry

end range_of_a_l615_615369


namespace increment_in_displacement_l615_615163

variable (d : ℝ)

def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

theorem increment_in_displacement:
  let t1 := 2
  let t2 := 2 + d
  let s1 := equation_of_motion t1
  let s2 := equation_of_motion t2
  s2 - s1 = 8 * d + 2 * d^2 := by
  sorry

end increment_in_displacement_l615_615163


namespace parabola_vertex_l615_615162

theorem parabola_vertex (x y : ℝ) : y^2 + 6*y + 2*x + 5 = 0 → (x, y) = (2, -3) :=
sorry

end parabola_vertex_l615_615162


namespace fractions_non_integer_l615_615393

theorem fractions_non_integer (a b c d : ℤ) : 
  ∃ (a b c d : ℤ), 
    ¬((a-b) % 2 = 0 ∧ 
      (b-c) % 2 = 0 ∧ 
      (c-d) % 2 = 0 ∧ 
      (d-a) % 2 = 0) :=
sorry

end fractions_non_integer_l615_615393


namespace compare_abc_l615_615005

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615005


namespace fifth_rectangle_is_square_l615_615676

-- Define the conditions
variables (s : ℝ) (a b : ℝ)
variables (R1 R2 R3 R4 : Set (ℝ × ℝ))
variables (R5 : Set (ℝ × ℝ))

-- Assume the areas of the corner rectangles are equal
def equal_area (R : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (a b : ℝ), R = {p | p.1 < a ∧ p.2 < b} ∧ a * b = k

-- State the conditions
axiom h1 : equal_area R1 a
axiom h2 : equal_area R2 a
axiom h3 : equal_area R3 a
axiom h4 : equal_area R4 a

axiom h5 : ∀ (p : ℝ × ℝ), p ∈ R5 → p.1 ≠ 0 → p.2 ≠ 0

-- Prove that the fifth rectangle is a square
theorem fifth_rectangle_is_square : ∃ c : ℝ, ∀ r1 r2, r1 ∈ R5 → r2 ∈ R5 → r1.1 - r2.1 = c ∧ r1.2 - r2.2 = c :=
by sorry

end fifth_rectangle_is_square_l615_615676


namespace symmetric_point_A_in_xOz_l615_615875

-- Definition of a point in three-dimensional space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition of the symmetry with respect to the plane xOz
def symmetric_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

-- Define the given point A
def A : Point3D := { x := -3, y := 2, z := -4 }

-- The proof statement
theorem symmetric_point_A_in_xOz :
  symmetric_xOz A = { x := -3, y := -2, z := -4 } :=
by
  -- Proof is omitted as "sorry"
  sorry

end symmetric_point_A_in_xOz_l615_615875


namespace compare_abc_l615_615010

noncomputable theory

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l615_615010


namespace simplify_and_rationalize_l615_615145

theorem simplify_and_rationalize :
  (1 : ℚ) / (1 + (1 / (real.sqrt 2 + 2))) = (4 + real.sqrt 2) / 7 :=
by
  sorry

end simplify_and_rationalize_l615_615145


namespace smallest_positive_period_monotonic_interval_correct_minimum_value_of_a_l615_615365

section Problem

-- Define the function f(x)
def f (x : ℝ) : ℝ := (sqrt 3 / 2) * sin x * cos x + (1 + cos (2 * x)) / 4

-- Define the period of the function
def period : ℝ := π

-- Define the monotonic interval
def monotonic_interval (k : ℤ) : Set ℝ := { x | k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 }

-- Triangle ABC, where a, b, c are sides opposite to angles A, B, C respectively
variable (A B C a b c : ℝ)
variable (hA : f A = 1 / 2)
variable (hBC : b + c = 3)

-- Prove the smallest positive period of f(x) is π
theorem smallest_positive_period :  ∀ x : ℝ, f (x + period) = f x := sorry

-- Prove the monotonically increasing interval of f(x)
theorem monotonic_interval_correct (k : ℤ) : ∀ x ∈ monotonic_interval k, f x = f x := sorry

-- Prove the minimum value of a
theorem minimum_value_of_a : a ≥ 3 / 2 := sorry

end Problem

end smallest_positive_period_monotonic_interval_correct_minimum_value_of_a_l615_615365


namespace average_salary_increase_l615_615157

theorem average_salary_increase:
  (A1 S_manager Total1 Total2 A2 Increase: ℝ)
  (h1: A1 = 1500)
  (h2: S_manager = 14100)
  (h3: Total1 = 20 * A1)
  (h4: Total2 = Total1 + S_manager)
  (h5: A2 = Total2 / 21)
  (h6: Increase = A2 - A1):
  Increase = 600 := 
by
  sorry

end average_salary_increase_l615_615157


namespace comparison_abc_l615_615077

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615077


namespace comparison_abc_l615_615082

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615082


namespace proof_problem_l615_615848

variable (a b k : ℕ)
variable (ratings : ℕ → ℕ → Bool) -- ratings : participant_index → judge_index → rating ("pass" or "fail")

-- conditions
def condition1 := b ≥ 3
def condition2 := b % 2 = 1
def condition3 := ∀ i j, i ≠ j → (∑ x in Finset.range a, if ratings x i = ratings x j then 1 else 0) ≤ k

theorem proof_problem (h1 : condition1 b) (h2 : condition2 b) (h3 : condition3 a b k ratings) :
  (k:ℚ) / a ≥ (b - 1) / (2 * b) :=
sorry

end proof_problem_l615_615848


namespace lisa_needs_additional_marbles_l615_615533

-- Lean 4 statement
theorem lisa_needs_additional_marbles (friends marbles : ℕ) (h_friends : friends = 15) (h_marbles : marbles = 60) :
  let required_marbles := (friends * (friends + 1)) / 2 in
  let min_additional_marbles := required_marbles - marbles in
  min_additional_marbles = 60 :=
by
  sorry

end lisa_needs_additional_marbles_l615_615533


namespace find_x_l615_615758

theorem find_x : ∃ x : ℚ, 5^(3*x^2 - 4*x + 3) = 5^(3*x^2 + 6*x - 5) ∧ x = 4 / 5 :=
by
  use 4 / 5
  sorry

end find_x_l615_615758


namespace union_complement_eq_target_l615_615482

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615482


namespace valid_words_count_l615_615977

noncomputable def count_valid_words : Nat :=
  let total_possible_words : Nat := ((25^1) + (25^2) + (25^3) + (25^4) + (25^5))
  let total_possible_words_without_B : Nat := ((24^1) + (24^2) + (24^3) + (24^4) + (24^5))
  total_possible_words - total_possible_words_without_B

theorem valid_words_count : count_valid_words = 1864701 :=
by
  let total_1_letter_words := 25^1
  let total_2_letter_words := 25^2
  let total_3_letter_words := 25^3
  let total_4_letter_words := 25^4
  let total_5_letter_words := 25^5

  let total_words_without_B_1_letter := 24^1
  let total_words_without_B_2_letter := 24^2
  let total_words_without_B_3_letter := 24^3
  let total_words_without_B_4_letter := 24^4
  let total_words_without_B_5_letter := 24^5

  let valid_1_letter_words := total_1_letter_words - total_words_without_B_1_letter
  let valid_2_letter_words := total_2_letter_words - total_words_without_B_2_letter
  let valid_3_letter_words := total_3_letter_words - total_words_without_B_3_letter
  let valid_4_letter_words := total_4_letter_words - total_words_without_B_4_letter
  let valid_5_letter_words := total_5_letter_words - total_words_without_B_5_letter

  let valid_words := valid_1_letter_words + valid_2_letter_words + valid_3_letter_words + valid_4_letter_words + valid_5_letter_words
  sorry

end valid_words_count_l615_615977


namespace problem_statement_l615_615435

-- Definitions and assumptions
variables
  {A B T : Point}
  (Ω : SemiCircle A B)
  (ω : Circle)
  {P C D K : Point}

-- Given conditions as definitions
def conditions : Prop :=
  T ∈ Ω ∧
  ω.contains A ∧ ω.contains T ∧ ω.center ∈ triangle A B T ∧
  P ∈ arc TB ∧
  C ∈ segment A P ∧
  D ∈ ω ∧
  C.side_ne D.side (line A B) ∧
  perp CD (line A B) ∧
  K = circ_center (triangle C D P)

-- Questions rephrased as propositions
def part_i (h : conditions) : Prop :=
  K ∈ circumcircle (triangle T D P)

def part_ii (h : conditions) : Prop :=
  is_fixed K

-- Lean 4 proof problem statement
theorem problem_statement (h : conditions) : part_i h ∧ part_ii h :=
  by sorry

end problem_statement_l615_615435


namespace find_side_c_find_angle_B_l615_615835

variables (A B C : ℝ) (a b c : ℝ)
  (h₁ : a * real.cos B = 3)
  (h₂ : b * real.cos A = 1)
  (h₃ : A - B = real.pi / 6)

theorem find_side_c : c = 4 :=
sorry

theorem find_angle_B : B = real.pi / 6 :=
sorry

end find_side_c_find_angle_B_l615_615835


namespace compare_abc_l615_615056

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l615_615056


namespace sum_of_distinct_products_l615_615171

def number := "538G5073H6"

-- Conditions
def isDivisibleBy72 (n : String) : Prop :=
  let num := 538 * 10^7 + natDigit G * 10^6 + 507 * 10^3 + natDigit H * 10 + 6
  (num % 8 = 0) ∧ (num % 9 = 0)

def natDigit (c : Char) : Nat :=
  if '0' ≤ c ∧ c ≤ '9' then c.toNat - '0'.toNat else 0

-- Main statement
theorem sum_of_distinct_products (G H : Nat) (hG : G < 10) (hH : H < 10)
  (hdiv : isDivisibleBy72 number) :
  (G, H) ∈ [(4, 4), (1, 7), (0, 8)] →
  (GH = 16 ∨ GH = 7 ∨ GH = 0) ∧ (16 + 7 + 0 = 23) :=
sorry

end sum_of_distinct_products_l615_615171


namespace complex_number_z_solution_l615_615388

theorem complex_number_z_solution :
  ∃ z : ℂ, (1 + complex.sqrt 3 * complex.i = z * (1 - complex.sqrt 3 * complex.i))
  ∧ (z = -1/2 + (complex.sqrt 3)/2 * complex.i) :=
begin
  sorry
end

end complex_number_z_solution_l615_615388


namespace prove_bound_on_c_l615_615179

theorem prove_bound_on_c (n : ℕ) (a : ℕ → ℝ) (c : ℝ)
  (h0 : a 0 = 0) (hn : a n = 0) 
  (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n - 1 → a k = c + ∑ i in finset.range (n - k), a i * (a (i + k) + a (i + k + 1))) :
  c ≤ 1 / (4 * n) :=
by 
  sorry

end prove_bound_on_c_l615_615179


namespace product_of_translated_roots_l615_615912

noncomputable def roots (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_translated_roots
  {d e : ℝ}
  (h_d : roots 3 4 (-7) d)
  (h_e : roots 3 4 (-7) e)
  (sum_roots : d + e = -4 / 3)
  (product_roots : d * e = -7 / 3) :
  (d - 1) * (e - 1) = 1 :=
by
  sorry

end product_of_translated_roots_l615_615912


namespace find_phi_even_function_l615_615796

theorem find_phi_even_function :
  ∀ (ϕ : ℝ), (0 < ϕ ∧ ϕ < π/2) →
    (∀ x : ℝ, 2 * sin (x + π/3 + ϕ) = 2 * sin (-x + π/3 + ϕ)) →
    ϕ = π/6 := by
  intro ϕ hϕ h_even
  sorry

end find_phi_even_function_l615_615796


namespace bananas_to_oranges_l615_615858

theorem bananas_to_oranges (B A O : ℕ) 
    (h1 : 4 * B = 3 * A) 
    (h2 : 7 * A = 5 * O) : 
    28 * B = 15 * O :=
by
  sorry

end bananas_to_oranges_l615_615858


namespace water_flow_rates_l615_615185

-- Given conditions
variables (V t : ℝ) (small_pipe_flow_rate large_pipe_flow_rate : ℝ)

-- Definitions based on the conditions
def volume : ℝ := V
def total_time : ℝ := t
def small_water_pipe_flow_rate : ℝ := small_pipe_flow_rate
def large_water_pipe_flow_rate : ℝ := large_pipe_flow_rate

-- The assertion
theorem water_flow_rates :
  small_water_pipe_flow_rate = (5 * V) / (8 * t) ∧
  large_water_pipe_flow_rate = (5 * V) / (2 * t) :=
by
  sorry

end water_flow_rates_l615_615185


namespace union_complement_eq_target_l615_615481

namespace Proof

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U := U \ N -- defining the complement of N in U

-- Stating the theorem: M ∪ complement_U = {0,2,4,6,8}
theorem union_complement_eq_target : M ∪ complement_U = {0, 2, 4, 6, 8} :=
by sorry

end Proof

end union_complement_eq_target_l615_615481


namespace range_for_k_solutions_when_k_eq_1_l615_615340

noncomputable section

-- Part (1): Range for k
theorem range_for_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - (2 * k + 4) * x + k - 6 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2)) ↔ (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Part (2): Completing the square for k = 1
theorem solutions_when_k_eq_1 :
  (∀ x : ℝ, x^2 - 6 * x - 5 = 0 → (x = 3 + Real.sqrt 14 ∨ x = 3 - Real.sqrt 14)) :=
sorry

end range_for_k_solutions_when_k_eq_1_l615_615340


namespace proof_part_a_proof_part_b_l615_615918

-- Defining the sides, semi-perimeter, area, inradius, and exradii of the triangle
variables (a b c : ℝ)
noncomputable def p := (a + b + c) / 2
noncomputable def S := sqrt(p * (p - a) * (p - b) * (p - c))
noncomputable def r := S / p
noncomputable def r_a := S / (p - a)
noncomputable def r_b := S / (p - b)
noncomputable def r_c := S / (p - c)

-- Proof statements
theorem proof_part_a : (1 / r) = (1 / r_a) + (1 / r_b) + (1 / r_c) := sorry

theorem proof_part_b : S = sqrt(r * r_a * r_b * r_c) := sorry

end proof_part_a_proof_part_b_l615_615918


namespace proof_sum_difference_l615_615639

-- Define the sums a and b based on the conditions
def a : ℕ := (List.range' 2 20).filter (λ n, n % 2 = 0).sum

def b : ℕ := (List.range' 1 19).filter (λ n, n % 2 = 1).sum

-- The proof problem based on the translated problem
theorem proof_sum_difference : b - a = -10 := 
by
  -- Defined sums from conditions
  have a_def : a = (List.range' 2 20).filter (λ n, n % 2 = 0).sum := rfl
  have b_def : b = (List.range' 1 19).filter (λ n, n % 2 = 1).sum := rfl
  -- Calculate the exact sums for a and b
  have a_val : a = 110 := by
    sorry  -- Calculation of sum can be added here
  have b_val : b = 100 := by
    sorry  -- Calculation of sum can be added here
  -- Conclude the theorem using the calculated values
  calc
    b - a = 100 - 110 : by rw [b_val, a_val]
        ... = -10     : by norm_num

end proof_sum_difference_l615_615639


namespace find_triplets_l615_615322

def satisfy_equation (x y z : ℕ) : Prop :=
  x^2 + y^2 = 3 * 2016^z + 77

def solution_set : set (ℕ × ℕ × ℕ) := 
  {(8, 4, 0), (4, 8, 0), (77, 14, 1), (14, 77, 1), (70, 35, 1), (35, 70, 1)}

theorem find_triplets (x y z : ℕ) : satisfy_equation x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end find_triplets_l615_615322


namespace sum_of_solutions_equation_l615_615309

theorem sum_of_solutions_equation (x : ℝ) :
  let eqn := (4 * x + 7) * (3 * x - 8) = -12 in
  (eqn → root_sum (12 * x^2 - 11 * x - 68) = 11 / 12) :=
by
  sorry

end sum_of_solutions_equation_l615_615309


namespace chair_cost_l615_615303

namespace ChairCost

-- Conditions
def total_cost : ℕ := 135
def table_cost : ℕ := 55
def chairs_count : ℕ := 4

-- Problem Statement
theorem chair_cost : (total_cost - table_cost) / chairs_count = 20 :=
by
  sorry

end ChairCost

end chair_cost_l615_615303


namespace coeff_xcubed_expansion_l615_615588

theorem coeff_xcubed_expansion : 
  coefficient (expand (2*x - 3)^5) x^3 = 720 := 
sorry

end coeff_xcubed_expansion_l615_615588


namespace total_work_completion_time_l615_615236

def work_completion_time (a b c d e f W : ℝ) (h : a + b + c = W / 6) : ℝ :=
  2 + (2 / (1 + (6 * (d + e + f) / W)))

theorem total_work_completion_time
  (a b c d e f W : ℝ)
  (h : a + b + c = W / 6) :
  work_completion_time a b c d e f W h = 2 + (2 / (1 + (6 * (d + e + f) / W))) := 
sorry

end total_work_completion_time_l615_615236


namespace collinear_points_l615_615115

theorem collinear_points {A B C D P Q R E : Type}
  [defined_as_cyclic_quad ABCD]
  [point_on_extension_of_line P A C]
  [tangent_to_circle PB ω]
  [tangent_to_circle PD ω]
  [tangent_intersection_at C PD Q]
  [line_intersection_at AD Q R]
  [second_intersection_point AQ ω E] : 
  collinear B E R :=
sorry

end collinear_points_l615_615115


namespace length_of_AC_l615_615415

theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) : AC = 30.1 :=
sorry

end length_of_AC_l615_615415


namespace midpoint_on_midline_l615_615936

variables {A B C M N P Q : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] [MetricSpace P] [MetricSpace Q]

/-- The theorem states that given an isosceles triangle ABC, with points M on AB and N on BC such that BM = CN, the midpoint of MN lies on the midline of triangle ABC. -/
theorem midpoint_on_midline
  (h_iso : is_isosceles_triangle A B C)
  (hM : M ∈ line_segment A B)
  (hN : N ∈ line_segment B C)
  (h_eq : distance (B, M) = distance (C, N)) :
  ∃ F, F = midpoint M N ∧ F ∈ midline A B C :=
sorry

end midpoint_on_midline_l615_615936


namespace union_with_complement_l615_615523

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615523


namespace compare_abc_l615_615027

variables (a b c : ℝ)

-- Given conditions
def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

-- Goal to prove
theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l615_615027


namespace product_of_sums_of_squares_l615_615943

theorem product_of_sums_of_squares (p q r s : ℤ) (a b : ℤ) (h1 : a = p^2 + q^2) (h2 : b = r^2 + s^2) : 
  ∃ x y : ℤ, a * b = x^2 + y^2 := 
by 
  use [p * r + q * s, q * r - p * s]
  sorry

end product_of_sums_of_squares_l615_615943


namespace moving_circle_on_hyperbola_l615_615656

-- Define the given circles C1 and C2
structure Circle (center : ℝ × ℝ) (radius : ℝ)

-- Define the circles C1 and C2
def C1 : Circle := { center := (0, -1), radius := 1 }
def C2 : Circle := { center := (0, 4), radius := 2 }

-- Define the geometric properties and the hypothesis
def moving_circle (M : ℝ × ℝ) (r : ℝ) :=
  (dist M C1.center = r + C1.radius) ∧ (dist M C2.center = r + C2.radius)

-- The statement to be proved, with M on a branch of a hyperbola
theorem moving_circle_on_hyperbola (M : ℝ × ℝ) (r : ℝ) :
  moving_circle M r → abs (dist M C2.center - dist M C1.center) = 1 :=
by
  sorry

end moving_circle_on_hyperbola_l615_615656


namespace evaluate_expression_l615_615723

theorem evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) :
  (a^4 + b^4) / (a^2 - a * b + b^2) = 97 / 7 := by
  sorry

example : (3^4 + 2^4) / (3^2 - 3 * 2 + 2^2) = 97 / 7 := evaluate_expression 3 2 rfl rfl

end evaluate_expression_l615_615723


namespace compare_a_b_c_l615_615033

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615033


namespace sufficient_not_necessary_condition_l615_615131

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, abs (x - 1) < 3 → (x + 2) * (x + a) < 0) ∧ 
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ ¬(abs (x - 1) < 3)) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l615_615131


namespace orange_balls_count_l615_615276

-- Define the constants
constant total_balls : ℕ := 50
constant red_balls : ℕ := 20
constant blue_balls : ℕ := 10

-- Define the conditions
axiom total_parts : total_balls = red_balls + blue_balls + (total_balls - red_balls - blue_balls)
axiom pink_or_orange_balls : total_balls - red_balls - blue_balls = 20
axiom pink_is_three_times_orange {O P : ℕ} : P = 3 * O
axiom sum_pink_orange {O P : ℕ} : P + O = 20

-- Main statement to prove
theorem orange_balls_count : ∃ O : ℕ, ∀ P : ℕ, P = 3 * O → P + O = 20 → O = 5 :=
by
  sorry

end orange_balls_count_l615_615276


namespace det_matrixE_l615_615906

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end det_matrixE_l615_615906


namespace cube_root_simplification_l615_615626

theorem cube_root_simplification (a b : ℕ) (h : a > 0 ∧ b > 0 ∧ b = 1) (h_eq : ∛8000 = a * ∛b) : a + b = 21 :=
sorry

end cube_root_simplification_l615_615626


namespace edward_initial_money_l615_615314

variable (spent_books : ℕ) (spent_pens : ℕ) (money_left : ℕ)

theorem edward_initial_money (h_books : spent_books = 6) 
                             (h_pens : spent_pens = 16)
                             (h_left : money_left = 19) : 
                             spent_books + spent_pens + money_left = 41 := by
  sorry

end edward_initial_money_l615_615314


namespace comparison_abc_l615_615078

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615078


namespace range_of_k_points_making_isosceles_triangle_l615_615811

variables (k : ℝ) (x y : ℝ) (P : ℝ × ℝ)
def intersection_point := (x = k + 4) ∧ (y = k - 1)
def fourth_quadrant := (k + 4 > 0) ∧ (k - 1 < 0)
def line_eq1 (k x y : ℝ) := (x - 2 * y = -k + 6)
def line_eq2 (k x y : ℝ) := (x + 3 * y = 4k + 1)
def point_A := (2, 0)

theorem range_of_k (k : ℝ) : 
  (∃ x y, line_eq1 k x y ∧ line_eq2 k x y ∧ fourth_quadrant k x y) ↔ (-4 < k ∧ k < 1) := sorry

theorem points_making_isosceles_triangle (k : ℕ) (k_range : 0 ≤ k ∧ k < 1) :
  ∃ (P : ℝ × ℝ), P ∈ [{((1, -5/2) : ℝ × ℝ)}, ((2, -2) : ℝ × ℝ), ((18/5, -6/5) : ℝ × ℝ)] ∧
                  P ∈ (λ P, line_eq1 k P.1 P.2) ∧ (P, 0) := sorry

end range_of_k_points_making_isosceles_triangle_l615_615811


namespace tan_product_range_l615_615287

-- Definitions using the given conditions
variables {α β γ : ℝ}

def mutually_perpendicular_edges (α β γ : ℝ) : Prop :=
  (cos α)^2 + (cos β)^2 + (cos γ)^2 = 1

-- The goal statement in Lean
theorem tan_product_range (α β γ : ℝ) (h : mutually_perpendicular_edges α β γ) :
  ∃ (k : ℝ), k = 2 * sqrt 2 ∧ tan α * tan β * tan γ ≥ k :=
by
  intros
  use 2 * sqrt 2
  sorry

end tan_product_range_l615_615287


namespace smallest_N_exists_l615_615402

theorem smallest_N_exists 
  (N c_1 c_2 c_3 c_4: ℕ) 
  (h1: c_1 = 4 * c_2 - 3)
  (h2: N + c_2 = 4 * c_4) 
  (h3: N = (3 * c_3 - 1) / 2)
  (h4: 3 * N + c_4 = 4 * c_1 - 3) 
  : N = 1 :=
begin
  sorry
end

end smallest_N_exists_l615_615402


namespace distribution_count_l615_615310

theorem distribution_count :
  let students := 4
  let towns := 3
  (nat.choose 4 2) * (nat.factorial 3) = 36 :=
by
  sorry

end distribution_count_l615_615310


namespace comparison_abc_l615_615084

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l615_615084


namespace sum_of_digits_10_pow_30_minus_54_l615_615224

theorem sum_of_digits_10_pow_30_minus_54 :
  let k := 10^30 - 54 in
  (Nat.digits 10 k).sum = 271 :=
by
  let k := 10^30 - 54
  sorry

end sum_of_digits_10_pow_30_minus_54_l615_615224


namespace moles_of_KI_formed_l615_615742

-- Definitions for initial moles of the reactants
def mole_KOH : ℝ := 1
def mole_NH4I : ℝ := 1

-- The balanced chemical reaction
def reaction (k : ℝ) (n : ℝ) : ℝ :=
  if k = 1 ∧ n = 1 then 1 else 0

-- Theorem: Number of moles of KI formed
theorem moles_of_KI_formed (k : ℝ) (n : ℝ) (h1 : k = 1) (h2 : n = 1) : reaction k n = 1 :=
by
  rw [h1, h2]
  simp
  sorry

end moles_of_KI_formed_l615_615742


namespace distance_between_lines_l615_615592

noncomputable def distance_between_parallel_lines_eq : Prop :=
  let line1 (x y : ℝ) := (2 * x - y = 0)
  let line2 (x y : ℝ) := (2 * x - y + 5 = 0)

  let distance : ℝ := |0 - 5| / real.sqrt (2^2 + (-1)^2)

  distance = real.sqrt 5

theorem distance_between_lines :
  distance_between_parallel_lines_eq := by
  sorry

end distance_between_lines_l615_615592


namespace union_complement_l615_615489

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615489


namespace probability_black_second_draw_l615_615843

theorem probability_black_second_draw
  (total_balls : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)
  (draws : ℕ)
  (first_draw_white : draws > 0) :
  total_balls = 5 ∧ white_balls = 3 ∧ black_balls = 2 ∧ (draws = 2) →
  (probability_of_black_second_draw : ℚ) : 
  probability_of_black_second_draw = 1/2 :=
begin
  sorry
end

end probability_black_second_draw_l615_615843


namespace probability_even_product_l615_615721

def is_even (n : ℕ) : Prop := n % 2 = 0

def chips_box_A := {1, 2, 4}
def chips_box_B := {1, 3, 5}

def total_outcomes : ℕ := chips_box_A.card * chips_box_B.card

def favorable_outcomes : ℕ :=
  (chips_box_A.filter is_even).card * chips_box_B.card

theorem probability_even_product : 
  (favorable_outcomes.to_rat / total_outcomes.to_rat) = (2 : ℚ / 3 : ℚ) :=
by
  sorry

end probability_even_product_l615_615721


namespace RachelMathHomeworkPages_l615_615136

variable (reading_pages : ℕ) (math_pages : ℕ)

def RachelHomework (reading_pages : ℕ) (math_pages : ℕ) : Prop :=
  reading_pages = 2 ∧ math_pages = reading_pages + 2

theorem RachelMathHomeworkPages : RachelHomework 2 4 :=
by
  unfold RachelHomework
  simp
  sorry

end RachelMathHomeworkPages_l615_615136


namespace hyperbola_asymptotes_l615_615254

theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 8 * b = 4 * (sqrt (((b ^ 2) * (c ^ 2) / (a ^ 2)) + c ^ 2 ))) : 
  ∃ c : ℝ, (a = b) → (y = x ∨ y = -x) :=
by
  sorry

end hyperbola_asymptotes_l615_615254


namespace range_of_f_l615_615731

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f : set.range f = {Real.pi / 4} :=
sorry

end range_of_f_l615_615731


namespace solve_abs_inequality_l615_615148

theorem solve_abs_inequality (x : ℝ) : (|x + 3| + |x - 4| < 8) ↔ (4 ≤ x ∧ x < 4.5) := sorry

end solve_abs_inequality_l615_615148


namespace problem1_minimum_problem1_maximum_problem2_minimum_problem2_maximum_l615_615739

noncomputable def f1 (x : ℝ) : ℝ := x^3 + 2 * x
noncomputable def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem problem1_minimum : ∀ x ∈ set.Icc (-1 : ℝ) 1, f1 x >= -3 :=
sorry

theorem problem1_maximum : ∀ x ∈ set.Icc (-1 : ℝ) 1, f1 x <= 3 :=
sorry

theorem problem2_minimum : ∀ x ∈ set.Icc (0 : ℝ) 3, f2 x >= -4 :=
sorry

theorem problem2_maximum : ∀ x ∈ set.Icc (0 : ℝ) 3, f2 x <= 2 :=
sorry

end problem1_minimum_problem1_maximum_problem2_minimum_problem2_maximum_l615_615739


namespace union_complement_eq_l615_615512

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l615_615512


namespace natasha_time_reach_top_l615_615124

variable (t : ℝ) (d_up d_total T : ℝ)

def time_to_reach_top (T d_up d_total t : ℝ) : Prop :=
  d_total = 2 * d_up ∧
  d_up = 1.5 * t ∧
  T = t + 2 ∧
  2 = d_total / T

theorem natasha_time_reach_top (T : ℝ) (h : time_to_reach_top T (1.5 * 4) (3 * 4) 4) : T = 4 :=
by
  sorry

end natasha_time_reach_top_l615_615124


namespace school_class_student_count_l615_615404

theorem school_class_student_count
  (num_classes : ℕ) (num_students : ℕ)
  (h_classes : num_classes = 30)
  (h_students : num_students = 1000)
  (h_max_students_per_class : ∀(n : ℕ), n < 30 → ∀(s : ℕ), s ≤ 33 → s ≤ 1000 / 30) :
  ∃ c, c ≤ num_classes ∧ ∃s, s ≥ 34 :=
by
  sorry

end school_class_student_count_l615_615404


namespace seating_arrangements_l615_615268

noncomputable def child := {a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3}
noncomputable def row := {0, 1, 2}
noncomputable def seat := {0, 1, 2}

structure seating :=
(position: child → (row × seat))
(row_constraint: ∀ (f: set child), (∀ x ∈ f, ∃ y ∈ f, y ≠ x → position x.1 = position y.1 → false) → f ⊆ {a_1, a_2, a_3} ∨ f ⊆ {b_1, b_2, b_3} ∨ f ⊆ {c_1, c_2, c_3})
(front_constraint: ∀ x ∈ child, ∀ y ∈ child, x ≠ y → position x = (r, s) → position y = (r + 1, s) → false)

theorem seating_arrangements: {s: seating // s satisfies constraints}.to_finset.card = 864 :=
sorry

end seating_arrangements_l615_615268


namespace michael_total_cost_l615_615544

def peach_pies : ℕ := 5
def apple_pies : ℕ := 4
def blueberry_pies : ℕ := 3

def pounds_per_pie : ℕ := 3

def price_per_pound_peaches : ℝ := 2.0
def price_per_pound_apples : ℝ := 1.0
def price_per_pound_blueberries : ℝ := 1.0

def total_peach_pounds : ℕ := peach_pies * pounds_per_pie
def total_apple_pounds : ℕ := apple_pies * pounds_per_pie
def total_blueberry_pounds : ℕ := blueberry_pies * pounds_per_pie

def cost_peaches : ℝ := total_peach_pounds * price_per_pound_peaches
def cost_apples : ℝ := total_apple_pounds * price_per_pound_apples
def cost_blueberries : ℝ := total_blueberry_pounds * price_per_pound_blueberries

def total_cost : ℝ := cost_peaches + cost_apples + cost_blueberries

theorem michael_total_cost :
  total_cost = 51.0 := by
  sorry

end michael_total_cost_l615_615544


namespace quadratic_inequality_solution_set_l615_615181

theorem quadratic_inequality_solution_set
  (a b c : ℝ)
  (h_solution : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ x ∈ Ioo (-1 / 2) 2) :
  b > 0 ∧ c > 0 ∧ a + b + c > 0 :=
sorry

end quadratic_inequality_solution_set_l615_615181


namespace minimum_value_ineq_l615_615921

open Real

theorem minimum_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
    (x + 1 / (y * y)) * (x + 1 / (y * y) - 500) + (y + 1 / (x * x)) * (y + 1 / (x * x) - 500) ≥ -125000 :=
by 
  sorry

end minimum_value_ineq_l615_615921


namespace probability_sum_odd_l615_615841

noncomputable def boxA : finset ℕ := {1, 2}
noncomputable def boxB : finset ℕ := {1, 2, 3}

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem probability_sum_odd : 
  (finset.card (boxA ×ˢ boxB).filter (λ (ab : ℕ × ℕ), is_odd (ab.1 + ab.2))) = 3 → 
  (finset.card (boxA ×ˢ boxB) = 6) → 
  (3 : ℚ) / (6 : ℚ) = (1 : ℚ) / (2 : ℚ) := 
by sorry

end probability_sum_odd_l615_615841


namespace calculate_AB_l615_615837

theorem calculate_AB (A B C : Point) (α β γ : ℝ) (h_triangle : right_triangle A B C)
  (h_angle_a : angle A = 90)
  (h_tan_b : tan B = 5 / 12)
  (h_ac : dist A C = 39) : dist A B = 15 :=
by 
  sorry

end calculate_AB_l615_615837


namespace union_complement_l615_615490

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615490


namespace container_unoccupied_volume_is_628_l615_615885

def rectangular_prism_volume (length width height : ℕ) : ℕ :=
  length * width * height

def water_volume (total_volume : ℕ) : ℕ :=
  total_volume / 3

def ice_cubes_volume (number_of_cubes volume_per_cube : ℕ) : ℕ :=
  number_of_cubes * volume_per_cube

def unoccupied_volume (total_volume occupied_volume : ℕ) : ℕ :=
  total_volume - occupied_volume

theorem container_unoccupied_volume_is_628 :
  let length := 12
  let width := 10
  let height := 8
  let number_of_ice_cubes := 12
  let volume_per_ice_cube := 1
  let V := rectangular_prism_volume length width height
  let V_water := water_volume V
  let V_ice := ice_cubes_volume number_of_ice_cubes volume_per_ice_cube
  let V_occupied := V_water + V_ice
  unoccupied_volume V V_occupied = 628 :=
by
  sorry

end container_unoccupied_volume_is_628_l615_615885


namespace hyperbola_asymptotes_l615_615161

variable {a b x y : ℝ}
variable (a_pos : a > 0) (b_pos : b > 0) (ecc : 2 * a = sqrt (a^2 + b^2))

theorem hyperbola_asymptotes :
  (∀ x y, ((x^2 / a^2) - (y^2 / b^2) = 1) → (y = sqrt 3 * x ∨ y = -sqrt 3 * x)) :=
by
  sorry

end hyperbola_asymptotes_l615_615161


namespace double_acute_angle_lt_180_l615_615350

theorem double_acute_angle_lt_180
  (α : ℝ) (h : 0 < α ∧ α < 90) : 2 * α < 180 := 
sorry

end double_acute_angle_lt_180_l615_615350


namespace relation_among_abc_l615_615089

def a := 2 * Real.log 1.01
def b := Real.log 1.02
def c := Real.sqrt 1.04 - 1

theorem relation_among_abc : a > c ∧ c > b := 
by {
    -- proof steps go here, but we use sorry for now
    sorry
}

end relation_among_abc_l615_615089


namespace minimum_number_of_pipes_l615_615261

theorem minimum_number_of_pipes (h : ℝ) :
  let r1 := 6 -- radius of 12-inch pipe
      v1 := π * r1^2 * h -- volume of the 12-inch pipe
      r2 := 1.5 -- radius of each 3-inch pipe
      v2 := π * r2^2 * h -- volume of each 3-inch pipe
  in v1 / v2 = 16 :=
by {
  let r1 := 6,
  let r2 := 1.5,
  let v1 := π * r1^2 * h,
  let v2 := π * r2^2 * h,
  have : v1 / v2 = (π * r1^2 * h) / (π * r2^2 * h),
  { rw [mul_div_cancel_left, mul_div_cancel_left] },
  norm_num at this,
  exact this,
  all_goals { norm_num },
}

end minimum_number_of_pipes_l615_615261


namespace log_floor_probability_l615_615559

-- We define the conditions: x and y are real numbers in the interval (0, 4)
def uniform_interval (a b : ℝ) : set ℝ := {x | a < x ∧ x < b}

-- Define the probability of the event
noncomputable def probability_event : ℝ :=
  let distribution : measure ℝ := volume.restrict (uniform_interval 0 4) in
  probability_space ℝ distribution,
  ∫⁺ x, ∫⁴ y, indicator (λ (z : ℝ × ℝ), ⌊log 4 (fst z)⌋ = ⌊log 4 (snd z)⌋) (λ _, 1) (x, y) ∂distribution

-- The claim to be proved that the probability is 5 / 8
theorem log_floor_probability: probability_event = 5 / 8 := 
by sorry

end log_floor_probability_l615_615559


namespace union_complement_set_l615_615465

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def complement_in_U (U N : Set ℕ) : Set ℕ :=
  U \ N

theorem union_complement_set :
  U = {0, 1, 2, 4, 6, 8} →
  M = {0, 4, 6} →
  N = {0, 1, 6} →
  M ∪ (complement_in_U U N) = {0, 2, 4, 6, 8} :=
by
  intros
  rw [complement_in_U, union_comm]
  sorry

end union_complement_set_l615_615465


namespace flight_time_is_approximately_50_hours_l615_615178

noncomputable def flightTime (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  circumference / speed

theorem flight_time_is_approximately_50_hours :
  let radius := 4200
  let speed := 525
  abs (flightTime radius speed - 50) < 1 :=
by
  sorry

end flight_time_is_approximately_50_hours_l615_615178


namespace orange_balls_count_l615_615277

theorem orange_balls_count :
  ∀ (total red blue orange pink : ℕ), 
  total = 50 → red = 20 → blue = 10 → 
  total = red + blue + orange + pink → 3 * orange = pink → 
  orange = 5 :=
by
  intros total red blue orange pink h_total h_red h_blue h_total_eq h_ratio
  sorry

end orange_balls_count_l615_615277


namespace union_with_complement_l615_615521

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l615_615521


namespace net_population_change_l615_615606

theorem net_population_change :
  let initial_population := 1
  let first_year := initial_population * (7 / 5)
  let second_year := first_year * (17 / 20)
  let final_population := second_year * (17 / 20)
  let net_change := ((final_population / initial_population) - 1) * 100
  abs (net_change - 1) < 0.5 :=
by
  -- Definitions of initial and yearly population changes
  let initial_population := 1
  let first_year := initial_population * (7 / 5)
  let second_year := first_year * (17 / 20)
  let final_population := second_year * (17 / 20)

  -- Calculating the net change
  let net_change := ((final_population / initial_population) - 1) * 100

  -- We need to show: abs (net_change - 1) < 0.5
  sorry

end net_population_change_l615_615606


namespace sin_X_value_l615_615425

variables (a b X : ℝ)

-- Conditions
def conditions :=
  (1/2 * a * b * Real.sin X = 100) ∧ (Real.sqrt (a * b) = 15)

theorem sin_X_value (h : conditions a b X) : Real.sin X = 8 / 9 := by
  sorry

end sin_X_value_l615_615425


namespace find_two_digit_number_l615_615273

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b

theorem find_two_digit_number (a b : ℕ) (h1 : a = 2 * b) (h2 : original_number b a = original_number a b - 36) : original_number a b = 84 :=
by
  sorry

end find_two_digit_number_l615_615273


namespace sqrt_of_sum_of_fractions_l615_615216

theorem sqrt_of_sum_of_fractions:
  sqrt ((25 / 36) + (16 / 9)) = sqrt 89 / 6 := by
    sorry 

end sqrt_of_sum_of_fractions_l615_615216


namespace karen_avg_speed_l615_615890

theorem karen_avg_speed : 
  let start_time := (8, 30) -- 8:30 a.m as (hours, minutes)
  let end_time := (14, 15) -- 2:15 p.m as (hours, minutes)
  let distance := 210 -- distance in miles
  let total_time_in_hours := (5:ℚ) + (45:ℚ)/60 -- total time from 8:30 a.m. to 2:15 p.m.
  let average_speed := distance / total_time_in_hours -- definition of average speed
  in average_speed = 840 / 23 :=
by
  sorry

end karen_avg_speed_l615_615890


namespace union_complement_l615_615492

namespace Example

def U := {0, 1, 2, 4, 6, 8}
def M := {0, 4, 6}
def N := {0, 1, 6}
def complement_U_N := U \ N

theorem union_complement : M ∪ complement_U_N = {0, 2, 4, 6, 8} :=
by
  sorry

end Example

end union_complement_l615_615492


namespace election_votes_l615_615861

theorem election_votes
(total_votes : ℕ)
(invalid_percentage : ℕ)
(valid_percentage : ℕ)
(candidate_A_percentage : ℕ)
(candidate_B_percentage : ℕ)
(candidate_C_percentage : ℕ)
(h_total_votes : total_votes = 800000)
(h_invalid_perc : invalid_percentage = 20)
(h_valid_perc : valid_percentage = 80)
(h_candidate_A_perc : candidate_A_percentage = 45)
(h_candidate_B_perc : candidate_B_percentage = 30)
(h_candidate_C_perc : candidate_C_percentage = 25) :
let valid_votes := (valid_percentage * total_votes) / 100 in
let votes_A := (candidate_A_percentage * valid_votes) / 100 in
let votes_B := (candidate_B_percentage * valid_votes) / 100 in
let votes_C := (candidate_C_percentage * valid_votes) / 100 in
votes_A = 288000 ∧ votes_B = 192000 ∧ votes_C = 160000 :=
by {
  sorry
}

end election_votes_l615_615861


namespace compare_a_b_c_l615_615043

noncomputable def a := 2 * Real.log 1.01
noncomputable def b := Real.log 1.02
noncomputable def c := Real.sqrt 1.04 - 1

theorem compare_a_b_c : a > c ∧ c > b :=
by
  sorry

end compare_a_b_c_l615_615043


namespace distance_AB_l615_615408

def parametric_curve_C1 (t : ℝ) : ℝ × ℝ :=
  (6 + (√3 / 2) * t, (1 / 2) * t)

def cartesian_curve_C2 (x y : ℝ) : Prop :=
  x^2 + y^2 = 10 * x

theorem distance_AB : 
  (∀ A B : ℝ × ℝ, 
    (∃ t₁ t₂ : ℝ, 
      parametric_curve_C1 t₁ = A ∧ parametric_curve_C1 t₂ = B ∧ 
      cartesian_curve_C2 A.1 A.2 ∧ cartesian_curve_C2 B.1 B.2) →
      (dist A B = 3 * √11)) :=
sorry

end distance_AB_l615_615408


namespace chair_cost_l615_615305

theorem chair_cost (T P n : ℕ) (hT : T = 135) (hP : P = 55) (hn : n = 4) : 
  (T - P) / n = 20 := by
  sorry

end chair_cost_l615_615305
