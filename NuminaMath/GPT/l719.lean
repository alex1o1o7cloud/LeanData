import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Parity.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Trigonometry.Bounds
import Mathlib.Combinatorics
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Determinant
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Int.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import data.nat.choose.basic

namespace age_hence_l719_719047

theorem age_hence (A x : ℕ) (h1 : A = 50)
  (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 :=
by sorry

end age_hence_l719_719047


namespace floor_sqrt_48_squared_l719_719830

theorem floor_sqrt_48_squared :
  (let sqrt48 := real.sqrt 48 in
  (floor sqrt48)^2 = 36) :=
by
  -- define conditions
  have h1 : real.sqrt 36 = 6 := by norm_num
  have h2 : real.sqrt 49 = 7 := by norm_num
  have h3 : 36 < 48 ∧ 48 < 49 := by norm_num
  -- skip the proof
  sorry

end floor_sqrt_48_squared_l719_719830


namespace first_valve_time_l719_719698

noncomputable def first_valve_filling_time (V1 V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ) : ℝ :=
  pool_capacity / V1

theorem first_valve_time :
  ∀ (V1 : ℝ) (V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ),
    V2 = V1 + 50 →
    V1 + V2 = pool_capacity / combined_time →
    combined_time = 48 →
    pool_capacity = 12000 →
    first_valve_filling_time V1 V2 pool_capacity combined_time / 60 = 2 :=
  
by
  intros V1 V2 pool_capacity combined_time h1 h2 h3 h4
  sorry

end first_valve_time_l719_719698


namespace melanie_gave_3_plums_to_sam_l719_719967

theorem melanie_gave_3_plums_to_sam 
  (initial_plums : ℕ) 
  (plums_left : ℕ) 
  (plums_given : ℕ) 
  (h1 : initial_plums = 7) 
  (h2 : plums_left = 4) 
  (h3 : plums_left + plums_given = initial_plums) : 
  plums_given = 3 :=
by 
  sorry

end melanie_gave_3_plums_to_sam_l719_719967


namespace candy_distribution_l719_719998

theorem candy_distribution :
  let n := 10;
  let bags := 3;
  (∀ a b c : ℕ, a + b + c = n → a ≥ 1 ∧ b ≥ 1 ∧ c = 0 ∨ a ≥ 1 ∧ c ≥ 1 ∧ b = 0 ∨ b ≥ 1 ∧ c ≥ 1 ∧ a = 0) →
  ∑ (a : ℕ) in finset.range (n + 1), ∑ (b : ℕ) in finset.range (n + 1 - a),
    (if (1 ≤ a ∧ 1 ≤ b) then (nat.choose n a) * (nat.choose (n - a) b) else 0) = 128 :=
by 
  sorry

end candy_distribution_l719_719998


namespace circles_disjoint_l719_719895

-- Define the circles and the condition m > 3
def C1 (x y m : ℝ) : ℝ := x^2 + y^2 - 2 * m * x + m^2 - 4
def C2 (x y m : ℝ) : ℝ := x^2 + y^2 + 2 * x - 2 * m * y - (8 - m^2)

noncomputable def distance_centers (m : ℝ) : ℝ := real.sqrt ((m + 1)^2 + m^2)
noncomputable def sum_radii : ℝ := 2 + 3

theorem circles_disjoint (m : ℝ) (h : m > 3) : distance_centers m > sum_radii :=
by
  -- the statement of inequality we're supposed to prove
  sorry

end circles_disjoint_l719_719895


namespace cleaner_solution_needed_l719_719963

-- Definitions based on problem conditions
def solution_oz_per_dog_stain := 6
def solution_oz_per_cat_stain := 4
def solution_oz_per_bird_stain := 3
def solution_oz_per_rabbit_stain := 1
def solution_oz_per_fish_stain := 2

def num_dog_stains := 10
def num_cat_stains := 8
def num_bird_stains := 5
def num_rabbit_stains := 1
def num_fish_stains := 3

def solution_oz_available := 64

-- Proof statement
theorem cleaner_solution_needed :
  let 
    total_solution_needed := 
      (num_dog_stains * solution_oz_per_dog_stain) + 
      (num_cat_stains * solution_oz_per_cat_stain) + 
      (num_bird_stains * solution_oz_per_bird_stain) + 
      (num_rabbit_stains * solution_oz_per_rabbit_stain) + 
      (num_fish_stains * solution_oz_per_fish_stain)
    additional_solution_needed := 
      if total_solution_needed > solution_oz_available then 
        total_solution_needed - solution_oz_available 
      else 
        0 
  in
  total_solution_needed = 114 ∧ additional_solution_needed = 50 := by
sorry

end cleaner_solution_needed_l719_719963


namespace probability_of_rolling_five_l719_719053

-- Define a cube with the given face numbers
def cube_faces : List ℕ := [1, 1, 2, 4, 5, 5]

-- Prove the probability of rolling a "5" is 1/3
theorem probability_of_rolling_five :
  (cube_faces.count 5 : ℚ) / cube_faces.length = 1 / 3 := by
  sorry

end probability_of_rolling_five_l719_719053


namespace sqrt_domain_condition_l719_719917

theorem sqrt_domain_condition (x : ℝ) : (x + 1 ≥ 0) ↔ (∃ y : ℝ, y = sqrt (x + 1)) :=
by
  sorry

end sqrt_domain_condition_l719_719917


namespace line_equation_minimized_area_l719_719457

theorem line_equation_minimized_area :
  ∀ (l_1 l_2 l_3 : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop),
    (∀ x y : ℝ, l_1 (x, y) ↔ 3 * x + 2 * y - 1 = 0) ∧
    (∀ x y : ℝ, l_2 (x, y) ↔ 5 * x + 2 * y + 1 = 0) ∧
    (∀ x y : ℝ, l_3 (x, y) ↔ 3 * x - 5 * y + 6 = 0) →
    (∃ c : ℝ, ∀ x y : ℝ, l (x, y) ↔ 3 * x - 5 * y + c = 0) →
    (∃ x y : ℝ, l_1 (x, y) ∧ l_2 (x, y) ∧ l (x, y)) →
    (∀ a : ℝ, ∀ x y : ℝ, l (x, y) ↔ x + y = a) →
    (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, l (x, y) ↔ 2 * x - y + 4 = 0) → 
    sorry :=
sorry

end line_equation_minimized_area_l719_719457


namespace sum_inequality_l719_719149

theorem sum_inequality (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2) :
  (x + y + z) * (x⁻¹ + y⁻¹ + z⁻¹) ≥ 6 * (x / (y + z) + y / (z + x) + z / (x + y)) := sorry

end sum_inequality_l719_719149


namespace cyclic_points_PQBC_l719_719579

theorem cyclic_points_PQBC
  (A B C D K L P Q : Point)
  (h_trapezoid : is_trapezoid A B C D)
  (h_parallel : parallel (line_through A B) (line_through C D))
  (h_greater : length (line_through A B) > length (line_through C D))
  (h_ratio_KL : length (segment A K) / length (segment K B) = length (segment D L) / length (segment L C))
  (h_P_on_KL : on_line_segment P K L)
  (h_Q_on_KL : on_line_segment Q K L)
  (h_angle_APB_BCD : angle A P B = angle B C D)
  (h_angle_CQD_ABC : angle C Q D = angle A B C) :
  cyclic P Q B C :=
sorry

end cyclic_points_PQBC_l719_719579


namespace polynomial_symmetry_l719_719384

noncomputable def has_center_of_symmetry (P : ℝ → ℝ) : Prop :=
∃ a, ∀ x, P(a - x) = -P(a + x)

theorem polynomial_symmetry (P : ℝ → ℝ) (h : ∀ᶠ m in Filter.at_top, ∃ n : ℤ, P m + P n = 0) : 
  has_center_of_symmetry P :=
sorry

end polynomial_symmetry_l719_719384


namespace first_valve_fill_time_l719_719697

theorem first_valve_fill_time (V1 V2: ℕ) (capacity: ℕ) (t_combined t1: ℕ) 
  (h1: t_combined = 48)
  (h2: V2 = V1 + 50)
  (h3: capacity = 12000)
  (h4: V1 + V2 = capacity / t_combined)
  : t1 = 2 * 60 :=
by
  -- The proof would come here
  sorry

end first_valve_fill_time_l719_719697


namespace angle_relation_l719_719975

-- Define the convex quadrilateral and points P, Q, E
variables (A B C D P Q E : Type)
variables [infinite A] [infinite B] [infinite C] [infinite D] [infinite P] [infinite Q] [infinite E]

-- Define the relevant angles and intersections
variables (angle_1 angle_2 angle_3 angle_4 : ℝ)
variables (AQ_E : E = AQ.intPt)
variables (CP_E : E = CP.intPt)
variables (ABP_CBQ : angle_1 = angle_2)

-- Define the actual theorem
theorem angle_relation (H : convex_quadrilateral A B C D) (H1 : P ∈ segment A D)
  (H2 : Q ∈ segment D C) (H3 : angle A B P = angle C B Q)
  (H4 : E = AQ.intPt) (H5 : E = CP.intPt)
  : angle A B E = angle C B D :=
sorry

end angle_relation_l719_719975


namespace part1_part2_l719_719890

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem part1 (hx : f (-x) = 2 * f x) : f x ^ 2 = 2 / 5 := 
  sorry

theorem part2 : 
  ∀ k : ℤ, ∃ a b : ℝ, [a, b] = [2 * π * k + (5 * π / 6), 2 * π * k + (11 * π / 6)] ∧ 
  ∀ x : ℝ, x ∈ Set.Icc a b → ∀ y : ℝ, y = f (π / 12 - x) → 
  ∃ δ > 0, ∀ ε > 0, 0 < |x - y| ∧ |x - y| < δ → y < x := 
  sorry

end part1_part2_l719_719890


namespace sales_overlap_once_l719_719719

def bookstore_sales (n : ℕ) : ℕ → List ℕ
| 31 => []

| _ => List.range' (n - n % 5 + 5) 5

def bakery_sales (days_in_july : ℕ) : List ℕ :=
  List.range' 2 7 (days_in_july - 2 + 7)

def common_sales_days (days_in_july : ℕ) : List ℕ :=
(bookstore_sales days_in_july).intersect (bakery_sales days_in_july)

theorem sales_overlap_once :
  ∃ days_in_july, (1 ≤ days_in_july ∧ days_in_july ≤ 31) ∧ (common_sales_days days_in_july).length = 1 :=
by
  sorry

end sales_overlap_once_l719_719719


namespace piecewise_equivalent_l719_719354

def piecewise_function (x : ℝ) : ℝ :=
if x < -2 then 4 + x else if x < 0 then -x else if x < 2 then x else 4 - x

theorem piecewise_equivalent (x : ℝ) : 
  (if -2 ≤ x ∧ x < 2 then |x| else 4 - |x|) = 
  (if x < -2 then 4 + x else if x < 0 then -x else if x < 2 then x else 4 - x) :=
by
  sorry

end piecewise_equivalent_l719_719354


namespace problem_statement_l719_719561

theorem problem_statement 
    (C₁ : ℝ → ℝ → Prop) 
    (C₂ : ℝ → ℝ → Prop)
    (M : ℝ × ℝ)
    (N : ℝ × ℝ) 
    (l₁ : ℝ → ℝ) 
    (l₂ : ℝ → ℝ) 
    (A : ℝ × ℝ) 
    (B : ℝ × ℝ) 
    (P : ℝ × ℝ)
    (hC₁ : ∀ x y, C₁ x y ↔ (x + 1)^2 + y^2 = 16)
    (hC₂ : ∀ x y, C₂ x y ↔ x^2 / 4 + y^2 / 3 = 1)
    (hM : M = (-1, 0))
    (hN : N = (1, 0))
    (hIntersects : ∃ A B, A ≠ B ∧ C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ 
                         (∃ m : ℝ, ∀ t : ℝ, (A.1, A.2 + t * (B.2 - A.2)) ∧ (B.1, B.2 + t * (A.2 - B.2))))
    (hBisectors : ∀ A B N, let mid := λ p₁ p₂ : ℝ × ℝ, ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2) in 
                           let perpbis := λ p₁ p₂ : ℝ × ℝ, ∃ (slope : ℝ), 
                             (λ x : ℝ, slope * (x - mid p₁ p₂).1 + mid p₁ p₂).2 in 
                           (l₁ = perpbis A N) ∧ (l₂ = perpbis B N))
    (hP : ∃ x y, P = (x, y) ∧ (l₁ x = y) ∧ (l₂ x = y)) :
  (∀ x y, (l₁ x = y ∧ C₂ x y) → x = y) ∧
  ∃ x, P = (-4, x) :=
by sorry

end problem_statement_l719_719561


namespace largest_fraction_l719_719352

theorem largest_fraction (A B C D E : ℚ)
    (hA: A = 5 / 11)
    (hB: B = 7 / 16)
    (hC: C = 23 / 50)
    (hD: D = 99 / 200)
    (hE: E = 202 / 403) : 
    E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_fraction_l719_719352


namespace find_x_coordinate_l719_719934

variable {ℝ : Type} [linear_ordered_field ℝ]

-- Define the coordinates of the points
def point1 : ℝ × ℝ := (10, 3)
def point2 : ℝ × ℝ := (4, 0)
def target_y : ℝ := -4

-- Finding the x-coordinate for which y = -4 on the line
theorem find_x_coordinate (x : ℝ) : 
  ∃ x, (point1.snd - point2.snd) / (point1.fst - point2.fst) = (target_y - 0) / (x - 4) := sorry

end find_x_coordinate_l719_719934


namespace trajectory_center_of_moving_circle_max_area_triangle_l719_719573

theorem trajectory_center_of_moving_circle (P : ℝ × ℝ) (A : ℝ × ℝ) (chord_length : ℝ) :
  A = (2, 0) ∧ chord_length = 4 ∧ (P ∈ set_of (λ P, ∃ r, distance P A = r ∧ distance P (⟨0, sqrt(4 * r)⟩) = r)) →
  ∃ C, ∀ (P : ℝ × ℝ), P ∈ C ↔ P.2^2 = 4 * P.1 :=
by sorry

theorem max_area_triangle (x1 x2 y1 y2 k m : ℝ) :
  y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 4 →
  m = (2 / k) - 2 * k →
  k^2 > 1 / 2 →
  let t := sqrt (2 - 1 / k^2) in
  let area := 12 * t - 4 * t^3 in
  area ≤ 8 :=
by sorry

end trajectory_center_of_moving_circle_max_area_triangle_l719_719573


namespace find_M_l719_719668

variable (p q r M : ℝ)
variable (h1 : p + q + r = 100)
variable (h2 : p + 10 = M)
variable (h3 : q - 5 = M)
variable (h4 : r / 5 = M)

theorem find_M : M = 15 := by
  sorry

end find_M_l719_719668


namespace solve_system_eqns_l719_719294

theorem solve_system_eqns (a x y : ℝ)
  (h1 : x - 4 = a * (y^3 - 2))
  (h2 : 2 * x / (|y^3| + y^3) = real.sqrt x) :
  (a ∈ Ioo 2 4 → x = (a - 2)^2 ∧ y = real.cbrt (a - 2)) ∨ 
  (a ∈ Set.Iic 2 → x = 4 ∧ y = real.cbrt 2) ∨
  (a ∈ Set.union (Iic 0) (Set.singleton 4) → x = 0 ∧ y = real.cbrt ((2 * a - 4) / a)) := 
begin
  sorry
end

end solve_system_eqns_l719_719294


namespace polygon_interior_exterior_relation_l719_719918

theorem polygon_interior_exterior_relation (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n ≥ 3) :
  n = 8 :=
by
  sorry

end polygon_interior_exterior_relation_l719_719918


namespace find_number_l719_719128

theorem find_number (x : ℝ) (hx : (50 + 20 / x) * x = 4520) : x = 90 :=
sorry

end find_number_l719_719128


namespace tables_count_is_correct_l719_719041

-- Definitions based on conditions
def invited_people : ℕ := 18
def people_didnt_show_up : ℕ := 12
def people_per_table : ℕ := 3

-- Calculation based on definitions
def people_attended : ℕ := invited_people - people_didnt_show_up
def tables_needed : ℕ := people_attended / people_per_table

-- The main theorem statement
theorem tables_count_is_correct : tables_needed = 2 := by
  unfold tables_needed
  unfold people_attended
  unfold invited_people
  unfold people_didnt_show_up
  unfold people_per_table
  sorry

end tables_count_is_correct_l719_719041


namespace first_valve_time_l719_719699

noncomputable def first_valve_filling_time (V1 V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ) : ℝ :=
  pool_capacity / V1

theorem first_valve_time :
  ∀ (V1 : ℝ) (V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ),
    V2 = V1 + 50 →
    V1 + V2 = pool_capacity / combined_time →
    combined_time = 48 →
    pool_capacity = 12000 →
    first_valve_filling_time V1 V2 pool_capacity combined_time / 60 = 2 :=
  
by
  intros V1 V2 pool_capacity combined_time h1 h2 h3 h4
  sorry

end first_valve_time_l719_719699


namespace trip_savings_l719_719336

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end trip_savings_l719_719336


namespace slope_of_dividing_line_l719_719816

theorem slope_of_dividing_line (P1 P2 P3 P4 : (ℝ × ℝ))
  (hP1 : P1 = (5, 30))
  (hP2 : P2 = (5, 90))
  (hP3 : P3 = (20, 120))
  (hP4 : P4 = (20, 60))
  (O : ℝ × ℝ) (hO : O = (0, 0))
  (line : ℝ × ℝ → Prop)
  (hline : ∀ p, line p ↔ ∃ k, p = (k * O.1, k * O.2))
  (h_congruent : ∃ P5 P6 : ℝ × ℝ, P5 ∈ line ∧ P6 ∈ line ∧
                   (P5 = (5, 30)) ∧ (P6 = (20, 120))) :
  let slope := (P6.2 - P5.2) / (P6.1 - P5.1) in
    slope = 6 := by
{
  sorry
}

end slope_of_dividing_line_l719_719816


namespace find_x_l719_719269

-- Define the functions δ (delta) and φ (phi)
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- State the theorem with conditions and question, and assert the answer
theorem find_x :
  (delta ∘ phi) x = 11 → x = -5/6 := by
  intros
  sorry

end find_x_l719_719269


namespace maximize_probability_C_l719_719040

variable (p1 p2 p3 : ℝ)
variable (h1 : 0 < p1)
variable (h2 : p1 < p2)
variable (h3 : p2 < p3)

noncomputable def probability_A := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
noncomputable def probability_B := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
noncomputable def probability_C := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem maximize_probability_C :  
  probability_C p1 p2 p3 h1 h2 h3 > probability_A p1 p2 p3 h1 h2 h3 ∧
  probability_C p1 p2 p3 h1 h2 h3 > probability_B p1 p3 p2 h1 h2 h3 := 
by {
  sorry
}

end maximize_probability_C_l719_719040


namespace marcus_mileage_l719_719405

theorem marcus_mileage (initial_mileage : ℕ) (miles_per_gallon : ℕ) (gallons_per_tank : ℕ) (tanks_used : ℕ) :
  initial_mileage = 1728 →
  miles_per_gallon = 30 →
  gallons_per_tank = 20 →
  tanks_used = 2 →
  initial_mileage + (miles_per_gallon * gallons_per_tank * tanks_used) = 2928 :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  exact rfl,
}

end marcus_mileage_l719_719405


namespace max_value_of_k_l719_719707

open Nat

-- Define the given conditions and the problem statement in Lean.
def A (i k : ℕ) : set ℕ := sorry -- Placeholder definition for the subsets

noncomputable def max_k (n : ℕ) : ℕ :=
  let k := 2 * n - 1 in
  -- Conditions
  let cond1 := ∀ (i j : ℕ), i ≠ j → ((A i k) ∩ (A j k)).to_finset.card % 2 = 1 in
  let cond2 := ∀ (i : ℕ), i ∈ range k → i ∉ A i k in
  let cond3 := ∀ (i j : ℕ), i ∈ range k → j ∈ A i k → i ∈ A j k in
  if cond1 ∧ cond2 ∧ cond3 then k else 0

-- The theorem that proves the maximum value of k.
theorem max_value_of_k (n : ℕ) : ∃ k : ℕ, k = 2 * n - 1 ∧ 
  (∀ (i j : ℕ), i ≠ j → ((A i k) ∩ (A j k)).to_finset.card % 2 = 1) ∧
  (∀ (i : ℕ), i ∈ range k → i ∉ A i k) ∧
  (∀ (i j : ℕ), i ∈ range k → j ∈ A i k → i ∈ A j k) :=
begin
  use 2 * n - 1,
  split,
  { refl },
  split,
  { intros i j h,
    exact sorry, -- Proof for cond1
  },
  split,
  { intros i h,
    exact sorry, -- Proof for cond2
  },
  { intros i j h1 h2,
    exact sorry -- Proof for cond3
  }
end

end max_value_of_k_l719_719707


namespace taxi_fare_for_8_2_km_l719_719734

-- Define the base fare, the incremental fare from 3km to 7km, and the incremental fare beyond 7km
def base_fare : ℝ := 6
def fare_3_to_7 (km: ℝ) : ℝ := 1 * km
def fare_above_7 (km: ℝ) : ℝ := 0.8 * km

-- Calculate the total fare for 8.2km travel
def total_fare (distance: ℝ) : ℝ :=
  if distance ≤ 3 then 
    base_fare
  else if distance ≤ 7 then 
    base_fare + fare_3_to_7 (distance - 3)
  else 
    base_fare + fare_3_to_7 4 + fare_above_7 (distance - 7)

-- Given problem: calculate fare for 8.2km
theorem taxi_fare_for_8_2_km : total_fare 8.2 = 11.6 := 
  by 
  -- Provide proof here if necessary
  sorry

end taxi_fare_for_8_2_km_l719_719734


namespace smallest_positive_integer_l719_719346

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end smallest_positive_integer_l719_719346


namespace exponentiation_property_l719_719641

variable (a : ℝ)

theorem exponentiation_property : a^2 * a^3 = a^5 := by
  sorry

end exponentiation_property_l719_719641


namespace shortest_distance_dasha_vasya_l719_719566

variables (dasha galia asya borya vasya : Type)
variables (dist : ∀ (a b : Type), ℕ)
variables (dist_dasha_galia : dist dasha galia = 15)
variables (dist_vasya_galia : dist vasya galia = 17)
variables (dist_asya_galia : dist asya galia = 12)
variables (dist_galia_borya : dist galia borya = 10)
variables (dist_asya_borya : dist asya borya = 8)

theorem shortest_distance_dasha_vasya : dist dasha vasya = 18 :=
by sorry

end shortest_distance_dasha_vasya_l719_719566


namespace range_of_x_l719_719481

theorem range_of_x (a : ℝ) (h : a ∈ Icc (-1 : ℝ) 1) : 
  ∀ x : ℝ, x^2 + (a-4)*x + 4 - 2*a > 0 ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end range_of_x_l719_719481


namespace permutations_no_youngest_first_last_l719_719232

-- Define the permutations function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the total number of permutations of five people
def total_permutations : ℕ := factorial 5

-- Define the number of permutations where the youngest is in the first or last position
def restricted_permutations : ℕ := 2 * factorial 4

-- Prove that the number of valid permutations is as required
theorem permutations_no_youngest_first_last : 
  total_permutations - restricted_permutations = 72 := by
  -- Expand the expression with specific calculated values
  have h1 : factorial 5 = 120 := by sorry
  have h2 : factorial 4 = 24 := by sorry
  have h3 : restricted_permutations = 2 * 24 := by rw h2; refl
  rw [total_permutations, restricted_permutations, h1, h3]
  show 120 - 48 = 72
  sorry

end permutations_no_youngest_first_last_l719_719232


namespace solution_of_inequality_l719_719665

theorem solution_of_inequality (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := 
sorry

end solution_of_inequality_l719_719665


namespace find_z_value_l719_719150

noncomputable def z_value (z : ℂ) : Prop :=
  arg (z^2 - 4) = 5 * real.pi / 6 ∧ arg (z^2 + 4) = real.pi / 3

theorem find_z_value (z : ℂ) :
  z_value z → z = 1 + real.sqrt 3 * complex.i ∨ z = -1 - real.sqrt 3 * complex.i :=
by
  sorry

end find_z_value_l719_719150


namespace rational_number_25_units_away_l719_719662

theorem rational_number_25_units_away (x : ℚ) (h : |x| = 2.5) : x = 2.5 ∨ x = -2.5 := 
by
  sorry

end rational_number_25_units_away_l719_719662


namespace find_other_integer_l719_719322

theorem find_other_integer (x y : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 7 * x + y = 68) : y = 5 :=
by
  sorry

end find_other_integer_l719_719322


namespace _l719_719877

noncomputable def geom_theorem (A B C U V W T : Point) (circumcircle : ∀ (ΔABC : Triangle), Circle ΔABC)
  (perp_bisector_AB : Line) (perp_bisector_AC : Line) (intersect_AU_V : AU ∩ perp_bisector_AB = V) 
  (intersect_AU_W : AU ∩ perp_bisector_AC = W) (intersect_BV_CW : BV ∩ CW = T)
  (smallest_angle : ( ∀ (ΔABC : Triangle), angle A < interior_angles ΔABC - {angle A})) 
  : Prop :=
  AU.length = TB.length + TC.length

end _l719_719877


namespace radius_ratio_in_right_triangle_l719_719571

theorem radius_ratio_in_right_triangle (PQ QR PR PS SR : ℝ)
  (h₁ : PQ = 5) (h₂ : QR = 12) (h₃ : PR = 13)
  (h₄ : PS + SR = PR) (h₅ : PS / SR = 5 / 8)
  (r_p r_q : ℝ)
  (hr_p : r_p = (1 / 2 * PQ * PS / 3) / ((PQ + PS / 3 + PS) / 3))
  (hr_q : r_q = (1 / 2 * QR * SR) / ((PS / 3 + QR + SR) / 3)) :
  r_p / r_q = 175 / 576 :=
sorry

end radius_ratio_in_right_triangle_l719_719571


namespace f_neg_example_l719_719946

-- Definitions and conditions given in the problem
def f : ℚ → ℚ := sorry

axiom condition1 (a b : ℚ) (ha : a > 0) (hb : b > 0) : f (a * b) = f a + f b
axiom condition2 (p : ℕ) (hp : nat.prime p) : f (p) = p

-- This is the statement that corresponds to the problem's question and conclusion.
theorem f_neg_example : f (25 / 11) < 0 :=
sorry

end f_neg_example_l719_719946


namespace shark_stingray_ratio_l719_719777

theorem shark_stingray_ratio (S R : ℕ) (hR : R = 28) (hTotal : S + R = 84) :
  S / R = 2 :=
by {
  -- Substitute R in hTotal
  have hTotal' : S + 28 = 84 := by rwa hR at hTotal,
  -- Solve for S
  have hS : S = 56 := by linarith,
  -- Compute the ratio S/R
  rw [hS, hR],
  -- Simplify the ratio
  exact nat.div_eq_of_eq_mul_right (by norm_num : 28 > 0) (by norm_num : 56 = 28 * 2),
}

end shark_stingray_ratio_l719_719777


namespace angle_between_hands_at_arrival_l719_719971

def angle_between_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (minute * 6 : ℝ)
  let hour_angle := (30 * hour + minute / 2 : ℝ)
  abs (minute_angle - hour_angle)

def travel_time (distance speed : ℝ) (ride_period rest_period : ℝ) : ℝ :=
  let base_time := (distance / speed)
  let intervals := base_time / ride_period
  base_time + floor intervals * (rest_period / 60)

noncomputable def arrival_time (start_hour start_minute travel_hours : ℝ) : (ℕ × ℕ) :=
  let total_minutes := start_hour * 60 + start_minute + travel_hours * 60
  let minutes := total_minutes % 60
  let hours := total_minutes / 60 % 12
  (hours.toNat, minutes.toNat)

def problem_data : Prop :=
  let start_hour := 6
  let start_minute := 0
  let speed := 12
  let distance := 16.8
  let ride_period := 0.5
  let rest_period := 6
  let travel_hours := travel_time distance speed ride_period rest_period
  let (arr_hour, arr_minute) := arrival_time start_hour start_minute travel_hours
  angle_between_hands arr_hour arr_minute = 12

theorem angle_between_hands_at_arrival : problem_data :=
by
  sorry

end angle_between_hands_at_arrival_l719_719971


namespace CD_length_of_trapezoid_l719_719109

-- Define the trapezoid ABCD with given conditions
noncomputable def trapezoid (A B C D : Type) :=
∃ (AB DC: Type), AB = 6 ∧ DC = 12 ∧ ∃ (BC: Type), BC = 4 * Real.sqrt 2 ∧
∃ (angle_BCD: Type), angle_BCD = 45 ∧ ∃ (angle_CDA: Type), angle_CDA = 30

-- Prove that the length of CD is 12
theorem CD_length_of_trapezoid (A B C D : Type) (h: trapezoid A B C D) : DC = 12 :=
by { simp [trapezoid], exact 12 } -- hypothesis and conditions simplify to the result

end CD_length_of_trapezoid_l719_719109


namespace one_of_a_b_c_is_zero_l719_719956

theorem one_of_a_b_c_is_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 :=
by
  sorry

end one_of_a_b_c_is_zero_l719_719956


namespace α_sufficient_not_necessary_for_β_l719_719851

variables {a b : ℕ}

/-- Proposition α states that a^2 + b^2 is divisible by 8 --/
def α : Prop := ((a^2 + b^2) % 8 = 0)

/-- Proposition β states that a^3 + b^3 is divisible by 16 --/
def β : Prop := ((a^3 + b^3) % 16 = 0)

/-- α is a sufficient condition for β, but not a necessary condition --/
theorem α_sufficient_not_necessary_for_β (a b : ℕ) : α → β ∧ ¬(β → α) :=
sorry

end α_sufficient_not_necessary_for_β_l719_719851


namespace max_cookies_eaten_l719_719774

theorem max_cookies_eaten (total_cookies : ℕ) (ratio : ℕ) (h_total : total_cookies = 30) (h_ratio : ratio = 3) : 
  ∃ (a : ℕ), a ≤ (total_cookies / (1 + ratio)) ∧ (a * ratio + a ≤ total_cookies) :=
by
  have h1 : 1 + ratio = 4 := by rw [h_ratio]; norm_num
  have h2 : total_cookies / (1 + ratio) = 7 := by rw [h_total, h1]; norm_num
  use 7
  split
  · exact le_refl 7
  · linarith

end max_cookies_eaten_l719_719774


namespace camp_problem_l719_719325

variable (x : ℕ) -- number of girls
variable (y : ℕ) -- number of boys
variable (total_children : ℕ) -- total number of children
variable (girls_cannot_swim : ℕ) -- number of girls who cannot swim
variable (boys_cannot_swim : ℕ) -- number of boys who cannot swim
variable (children_can_swim : ℕ) -- total number of children who can swim
variable (children_cannot_swim : ℕ) -- total number of children who cannot swim
variable (o_six_girls : ℕ) -- one-sixth of the total number of girls
variable (o_eight_boys : ℕ) -- one-eighth of the total number of boys

theorem camp_problem 
    (hc1 : total_children = 50)
    (hc2 : girls_cannot_swim = x / 6)
    (hc3 : boys_cannot_swim = y / 8)
    (hc4 : children_can_swim = 43)
    (hc5 : children_cannot_swim = total_children - children_can_swim)
    (h_total : x + y = total_children)
    (h_swim : children_cannot_swim = girls_cannot_swim + boys_cannot_swim) :
    x = 18 :=
  by
    have hc6 : children_cannot_swim = 7 := by sorry -- from hc4 and hc5
    have h_eq : x / 6 + (50 - x) / 8 = 7 := by sorry -- from hc2, hc3, hc6
    -- solving for x
    sorry

end camp_problem_l719_719325


namespace min_difference_of_composite_square_sum_to_96_l719_719366

def is_composite (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ n = p * q

def is_square (n : ℕ) : Prop := 
  ∃ (k : ℕ), n = k * k

theorem min_difference_of_composite_square_sum_to_96 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ is_square a ∧ a + b = 96 ∧ (∀ (a' b' : ℕ), (is_composite a' ∧ is_composite b' ∧ is_square a' ∧ a' + b' = 96) → abs (a' - b') ≥ abs (a - b)) ∧ abs (a - b) = 6 :=
by
  sorry

end min_difference_of_composite_square_sum_to_96_l719_719366


namespace eccentricity_is_correct_projection_coordinates_l719_719492

-- Define the ellipse, its parameters, and points
def ellipse (a : ℝ) (b : ℝ) := {p : ℝ × ℝ // (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

theorem eccentricity_is_correct (a b : ℝ) (ha : a > 2) (sum_distances : ∀ p : ellipse a b, 6 = 2 * a) :
    a = 3 → b = 2 → (let c := sqrt (a ^ 2 - b ^ 2) in c / a = sqrt 5 / 3) :=
by
  intros
  sorry

theorem projection_coordinates (a b : ℝ) (ha : a = 3) (hb : b = 2) :
    ∀ (p : ellipse a b), p.val.1 = sqrt 5 → 
    (let q := (0, p.val.2) in q.2 = 4 / 3 ∨ q.2 = -4 / 3) :=
by
  intros
  sorry

end eccentricity_is_correct_projection_coordinates_l719_719492


namespace families_seating_arrangements_l719_719710

/-- In a row of 9 seats, there are 3 families of three sitting together. 
    If each family sits together, the number of different seating arrangements is (3!)^4. -/
theorem families_seating_arrangements :
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1) in
  ∃ n : ℕ, n = 9 ∧
           (∃ families : ℕ, families = 3 ∧ 
           (∀ family_size : ℕ, family_size = 3 ∧
            (∃ total_arrangements : ℕ, total_arrangements = (factorial 3) ^ 4))
  )
:= sorry

end families_seating_arrangements_l719_719710


namespace correct_propositions_l719_719896

-- Definitions based on conditions
def Prop1 (n : Line) (α β : Plane) : Prop :=
  n ⊥ α ∧ n ⊥ β → α || β

def Prop2 (α β : Plane) : Prop :=
  (∃ p1 p2 p3 : Point, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 ∈ α ∧ p2 ∈ α ∧ p3 ∈ α ∧ equidistant β p1 ∧ equidistant β p2 ∧ equidistant β p3) → α || β

def Prop3 (n m : Line) (α β : Plane) : Prop :=
  skew n m ∧ n ⊆ α ∧ n || β ∧ m ⊆ β ∧ m || α → α || β

-- Final theorem to be proven
theorem correct_propositions (n m : Line) (α β : Plane) :
  (Prop1 n α β) ∧ (Prop2 α β) ∧ (Prop3 n m α β) → number_of_correct_propositions = 3 :=
sorry

end correct_propositions_l719_719896


namespace problem_solution_l719_719484

noncomputable def circles_intersect (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), (A ∈ { p | p.1^2 + p.2^2 = 1 }) ∧ (B ∈ { p | p.1^2 + p.2^2 = 1 }) ∧
  (A ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ (B ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ 
  (dist A B = (4 * Real.sqrt 5) / 5)

theorem problem_solution (m : ℝ) : circles_intersect m ↔ (m = 1 ∨ m = -3) := by
  sorry

end problem_solution_l719_719484


namespace printing_time_l719_719745

theorem printing_time (rate pages : ℕ) (h_rate : rate = 20) (h_pages : pages = 250) :
  (Int.ofNat ((pages : ℝ) / (rate : ℝ)).ceil) = 13 :=
begin
  rw [h_rate, h_pages],
  norm_num,
end

end printing_time_l719_719745


namespace find_n_l719_719350

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end find_n_l719_719350


namespace actual_area_of_park_l719_719748

-- Definitions of given conditions
def map_scale : ℕ := 250 -- scale: 1 inch = 250 miles
def map_length : ℕ := 6 -- length on map in inches
def map_width : ℕ := 4 -- width on map in inches

-- Definition of actual lengths
def actual_length : ℕ := map_length * map_scale -- actual length in miles
def actual_width : ℕ := map_width * map_scale -- actual width in miles

-- Theorem to prove the actual area
theorem actual_area_of_park : actual_length * actual_width = 1500000 := by
  -- By the conditions provided, the actual length and width in miles can be calculated directly:
  -- actual_length = 6 * 250 = 1500
  -- actual_width = 4 * 250 = 1000
  -- actual_area = 1500 * 1000 = 1500000
  sorry

end actual_area_of_park_l719_719748


namespace complement_union_P_Q_l719_719178

open Set

variable (U : Set ℝ := univ) (P Q : Set ℝ)
variable (U_eq : U = univ)
variable (P_def : P = {x : ℝ | 1 < x})
variable (Q_def : Q = {x : ℝ | x * (x - 2) < 0})

theorem complement_union_P_Q :
  ∁ (P ∪ Q) = {x : ℝ | x ≤ 0} :=
by
  rw [compl_union, P_def, Q_def]
  sorry

end complement_union_P_Q_l719_719178


namespace rope_length_eqn_l719_719235

theorem rope_length_eqn (x : ℝ) : 8^2 + (x - 3)^2 = x^2 := 
by 
  sorry

end rope_length_eqn_l719_719235


namespace PQ_length_maximal_iff_AM_diameter_l719_719477

open EuclideanGeometry

noncomputable def length_PQ_maximal (A B C M P Q : Point) (circumcircle : Circle) 
  (hA : A ∈ circumcircle) (hB : B ∈ circumcircle) (hC : C ∈ circumcircle) (hM : M ∈ circumcircle) 
  (h_perp_P : IsPerpendicular (M, P) (A, B)) (h_perp_Q : IsPerpendicular (M, Q) (A, C)) : Prop :=
  let diameter := Segment A (M) in
  ∃ PQ_maximal : Segment P Q, Maximal_length PQ_maximal (M, P) (M, Q) circumcircle ↔ IsDiameter diameter circumcircle

theorem PQ_length_maximal_iff_AM_diameter (A B C M P Q : Point) (circumcircle : Circle) 
  (hA : A ∈ circumcircle) (hB : B ∈ circumcircle) (hC : C ∈ circumcircle) (hM : M ∈ circumcircle) 
  (h_perp_P : IsPerpendicular (M, P) (A, B)) (h_perp_Q : IsPerpendicular (M, Q) (A, C)) :
  length_PQ_maximal A B C M P Q circumcircle hA hB hC hM h_perp_P h_perp_Q :=
begin
  sorry
end

end PQ_length_maximal_iff_AM_diameter_l719_719477


namespace rightmost_ten_digits_l719_719652

-- Definition of the number N
def N : ℕ := sorry -- placeholder for the actual number

-- N consists of 1999 digits
def has_1999_digits (n : ℕ) : Prop := n >= 10^1998 ∧ n < 10^1999

-- The sum of the digits of N is 9599
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Each pair of consecutive digits in N forms a two-digit number that is either a multiple of 17 or 23
def valid_pairs (n : ℕ) : Prop :=
  ∀ i < (n.digits 10).length - 1, 
  let x := (n.digits 10).nth_le i (by sorry) * 10 + 
           (n.digits 10).nth_le (i+1) (by sorry) in
  x % 17 = 0 ∨ x % 23 = 0

-- Statement of the theorem
theorem rightmost_ten_digits :
  has_1999_digits N ∧ sum_of_digits N = 9599 ∧ valid_pairs N →
  N % 10^10 = 3469234685 :=
sorry

end rightmost_ten_digits_l719_719652


namespace volume_of_oil_is_correct_l719_719045

noncomputable def volume_of_oil (h_full : ℝ) (d : ℝ) (ratio_ow : ℝ) : ℝ :=
  let r := d / 2
  let V_mixture := (1 / 3) * π * r^2 * h_full
  let V_oil := (1 / (1 + ratio_ow)) * V_mixture
  V_oil

theorem volume_of_oil_is_correct :
  volume_of_oil 9 3 5 ≈ 3.53 :=
begin
  sorry
end

end volume_of_oil_is_correct_l719_719045


namespace squirrel_travel_time_l719_719757

theorem squirrel_travel_time :
  ∀ (d r : ℝ), d = 2 → r = 5 → (d / r) * 60 = 24 :=
by
  intros d r hd hr
  rw [hd, hr]
  norm_num
  sorry

end squirrel_travel_time_l719_719757


namespace minimum_OP_l719_719199

-- Definitions for the conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def midpoint (A B P : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def distance (O P : ℝ × ℝ) : ℝ :=
  real.sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2)

-- Main theorem
theorem minimum_OP :
  ∃ P : ℝ × ℝ, 
    parabola P.1 P.2 ∧
    (∀ Q : ℝ × ℝ, 
      (∃ A B : ℝ × ℝ, A = (1, 2) ∧ parabola B.1 B.2 ∧ midpoint A B Q ∧ (Q.2 = (B.2 + 2) / 2 ∧ parabola (Q.1 + (B.2 + 2)^2 / 16) (B.2 + 2) / 2)) ∧
        (Q = ((B.2) / 2, (B.2 + 2) / 2) ∧ midpoint Q P Q ∧ distance (0, 0) P = real.sqrt (1/2))) ∧
    distance (0, 0) P = real.sqrt (1/2) :=
sorry

end minimum_OP_l719_719199


namespace chord_length_of_parabola_l719_719742

theorem chord_length_of_parabola 
  (k : ℝ) 
  (C : ∀ x y : ℝ, x^2 = 4 * y → Prop)
  (focus : (0 : ℝ, 1 : ℝ))
  (line : ∀ x y : ℝ, y = k * x + 1 → Prop)
  (midpoint_distance_to_x_axis : ℝ)
  (h1 : C (0 : ℝ) (1 : ℝ))
  (h2 : line (0 : ℝ) (1 : ℝ))
  (h3 : midpoint_distance_to_x_axis = 5) :
  ∃ AB_length : ℝ, AB_length = 6 := sorry

end chord_length_of_parabola_l719_719742


namespace sin_sum_alpha_pi_over_3_l719_719868

theorem sin_sum_alpha_pi_over_3 (α : ℝ) (h1 : cos (π / 2 + α) = sqrt 3 / 3) 
(h2 : -π / 2 < α) (h3 : α < π / 2) : 
sin (α + π / 3) = (3 * sqrt 2 - sqrt 3) / 6 := 
sorry

end sin_sum_alpha_pi_over_3_l719_719868


namespace monotonic_decreasing_interval_l719_719651

def f (x : ℝ) : ℝ := (x^2 + x + 1) * Real.exp x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -2 ∧ b = -1 ∧ ∀ (x : ℝ), a < x ∧ x < b → f' x < 0 :=
by
  sorry

end monotonic_decreasing_interval_l719_719651


namespace min_Q_l719_719061

noncomputable def F (m: ℕ) : ℤ := 
  let a := (m / 10) 
  let b := (m % 10) 
  9 * a - 9 * b

def Q (s t: ℕ) : ℚ := 
  (↑t - ↑s) / ↑s

theorem min_Q (s t : ℕ) (a b x y : ℕ)
  (h1 : s = 10 * a + b) 
  (h2 : t = 10 * x + y)
  (h3 : 1 ≤ b ∧ b < a ∧ a ≤ 7) 
  (h4 : 1 ≤ x ∧ x ≤ 8) 
  (h5 : 1 ≤ y ∧ y ≤ 8) 
  (h6 : F(s) % 5 = 1) 
  (h7 : F(t) - F(s) + 18 * ↑x = 36) : 
  Q(s, t) = -42 / 73 :=
sorry

end min_Q_l719_719061


namespace lois_books_count_l719_719958

variable (initial_books : ℕ)
variable (given_books_ratio : ℚ)
variable (donated_books_ratio : ℚ)
variable (purchased_books : ℕ)

theorem lois_books_count (h_initial : initial_books = 40)
  (h_given_ratio : given_books_ratio = 1 / 4)
  (h_donated_ratio : donated_books_ratio = 1 / 3)
  (h_purchased : purchased_books = 3) :
  let remaining_after_given := initial_books - initial_books * given_books_ratio in
  let remaining_after_donated := remaining_after_given - remaining_after_given * donated_books_ratio in
    remaining_after_donated + purchased_books = 23 := by
  sorry

end lois_books_count_l719_719958


namespace unattainable_y_value_l719_719132

theorem unattainable_y_value (x : ℚ) : x ≠ -4/3 → ¬ ∃ y : ℚ, y = (2 - x) / (3 * x + 4) ∧ y = -1/3 :=
by
  intro h
  apply not_exists_of_forall_not
  intro y
  intro hy
  have : 1 + 3 * (-1/3) = 0 := by norm_num
  have hy' : y ≠ -1/3
  { intro h'
    rw [h'] at hy
    have : 3 * x + 4 = 0 := by linarith
    linarith }
  contradiction

end unattainable_y_value_l719_719132


namespace wheel_rpm_l719_719027

noncomputable def radius : ℝ := 175
noncomputable def speed_kmh : ℝ := 66
noncomputable def speed_cmm := speed_kmh * 100000 / 60 -- convert from km/h to cm/min
noncomputable def circumference := 2 * Real.pi * radius -- circumference of the wheel
noncomputable def rpm := speed_cmm / circumference -- revolutions per minute

theorem wheel_rpm : rpm = 1000 := by
  sorry

end wheel_rpm_l719_719027


namespace final_mileage_after_trip_l719_719408

-- Define the conditions as constants
def initial_miles : ℕ := 1728
def fill_ups : ℕ := 2
def miles_per_gallon : ℕ := 30
def gallon_capacity : ℕ := 20

-- Define the final calculation as the proof problem.
theorem final_mileage_after_trip : initial_miles + (fill_ups * gallon_capacity * miles_per_gallon) = 2928 := 
by 
  exact congrArg (· + initial_miles) (congrArg (· * miles_per_gallon) (congArg (· * gallon_capacity) fill_ups)).trans sorry

end final_mileage_after_trip_l719_719408


namespace expression_equals_7_l719_719782
noncomputable def calculate_expression : ℝ :=
  0.25 * (-0.5)⁻⁴ + Real.log 8 / Real.log 10 + 3 * (Real.log 5 / Real.log 10)

theorem expression_equals_7 : calculate_expression = 7 :=
  sorry

end expression_equals_7_l719_719782


namespace simplify_expression_l719_719441

theorem simplify_expression :
  let a := 7
  let b := 11
  let c := 19
  (49 * (1 / 11 - 1 / 19) + 121 * (1 / 19 - 1 / 7) + 361 * (1 / 7 - 1 / 11)) /
  (7 * (1 / 11 - 1 / 19) + 11 * (1 / 19 - 1 / 7) + 19 * (1 / 7 - 1 / 11)) = 37 := by
  sorry

end simplify_expression_l719_719441


namespace ratio_PA_AB_l719_719923

theorem ratio_PA_AB (A B C P : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup P]
  (h1 : ∃ AC CB : ℕ, AC = 2 * CB)
  (h2 : ∃ PA AB : ℕ, PA = 4 * (AB / 5)) :
  PA / AB = 4 / 5 := sorry

end ratio_PA_AB_l719_719923


namespace equilateral_triangle_perimeter_equilateral_triangle_height_l719_719224

theorem equilateral_triangle_perimeter (a : ℝ) (h1 : a = 10) :
  3 * a = 30 :=
by {
  rw h1,
  norm_num,
}

theorem equilateral_triangle_height (a : ℝ) (h1 : a = 10) :
  sqrt (a^2 - (a / 2)^2) = 5 * sqrt 3 :=
by {
  rw h1,
  norm_num,
  rw sqrt_eq_rpow,
  norm_num,
  rw pow_two,
  norm_num,
  rw mul_rpow,
  norm_num,
}

end equilateral_triangle_perimeter_equilateral_triangle_height_l719_719224


namespace sum_of_integers_is_34_l719_719556

theorem sum_of_integers_is_34 (a b : ℕ) (h1 : a - b = 6) (h2 : a * b = 272) (h3a : a > 0) (h3b : b > 0) : a + b = 34 :=
  sorry

end sum_of_integers_is_34_l719_719556


namespace determine_centers_l719_719382

noncomputable def is_circumcenter (A B C O : Point) : Prop :=
∀ (P : Point), (dist P A = dist P B ∧ dist P B = dist P C) → (dist O A = dist O B ∧ dist O B = dist O C)

noncomputable def is_incenter (A B C O : Point) : Prop :=
∀ (P : Point), (dist_to_sides P A B C = dist_to_sides P B C A ∧ dist_to_sides P B C A = dist_to_sides P C A B) 
  → (dist_to_sides O A B C = dist_to_sides O B C A ∧ dist_to_sides O B C A = dist_to_sides O C A B)

noncomputable def is_orthocenter (A B C O : Point) : Prop :=
∀ (P : Point), (right_angle (segment A P) (segment B P) ∧ right_angle (segment B P) (segment C P)) 
  → orthogonal_projection P A B C = O

theorem determine_centers (A B C P O : Point) :
  (∀ (P : Point), dist P A = dist P B ∧ dist P B = dist P C) 
  ∧ (∀ (P : Point), dist_to_sides P A B C = dist_to_sides P B C A ∧ dist_to_sides P B C A = dist_to_sides P C A B)
  ∧ (∀ (P : Point), right_angle (segment A P) (segment B P) ∧ right_angle (segment B P) (segment C P)) 
  →
  (is_circumcenter A B C O) ∧ (is_incenter A B C O) ∧ (is_orthocenter A B C O) :=
by
  sorry

end determine_centers_l719_719382


namespace translated_vector_ab_l719_719170

-- Define points A and B, and vector a
def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (5, 2)
def a : ℝ × ℝ := (1, 2)

-- Define the vector AB
def vectorAB : ℝ × ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  (Bx - Ax, By - Ay)

-- Prove that after translating vector AB by vector a, the result remains (2, -5)
theorem translated_vector_ab :
  vectorAB = (2, -5) := by
  sorry

end translated_vector_ab_l719_719170


namespace graph_no_k_clique_bound_l719_719940

variable {V : Type} [Fintype V] [DecidableEq V]

-- Define conditions of the graph and absence of k-clique
structure Graph (V : Type) :=
(edges : V → V → Prop)
(symm : ∀ {a b}, edges a b → edges b a)
(loopless : ∀ {a}, ¬ edges a a)

def edge_count (G : Graph V) : Nat :=
  Fintype.card {e : V × V // G.edges e.1 e.2}

def is_k_clique (G : Graph V) (k : Nat) (s : Finset V) : Prop :=
  s.card = k ∧ ∀ a ∈ s, ∀ b ∈ s, a ≠ b → G.edges a b

noncomputable def max_edges_without_k_clique (n k : Nat) : ℚ :=
  (k - 2) / (k - 1) * (n^2 / 2)

-- The theorem statement
theorem graph_no_k_clique_bound (n k : ℕ) (G : Graph V) (hG : Fintype.card V = n)
  (no_k_clique : ∀ s : Finset V, ¬is_k_clique G k s) :
  edge_count G ≤ (max_edges_without_k_clique n k).to_nat := 
sorry

end graph_no_k_clique_bound_l719_719940


namespace log_eq_solution_l719_719016

theorem log_eq_solution (x : ℝ) (hx : 0 < x) : 
  log 5 x + log 25 x = log (1/5) (sqrt 3) ↔ x = 1 / (3^(1/3)) :=
by
  sorry

end log_eq_solution_l719_719016


namespace find_room_height_l719_719037

-- Define the conditions
def room_dimensions : ℝ := 6 
def radius_small_sphere : ℝ := 1 
def radius_large_sphere : ℝ := 3 

-- Assume the centers of the small spheres
def small_sphere_centers : list (ℝ × ℝ × ℝ) := [(1, 1, 1), (1, 5, 1), (5, 1, 1), (5, 5, 1)]

-- Assume the center of the large sphere, to be determined
def center_large_sphere (z : ℝ) : ℝ × ℝ × ℝ := (3, 3, z)

-- Tangency condition
def tangency_condition (z : ℝ) : Prop :=
  ∀ p ∈ small_sphere_centers, Real.dist (3, 3, z) p = 4

-- Height of the room
def room_height (z : ℝ) : ℝ := 2 * z

-- The final proof statement
theorem find_room_height :
  ∃ z, tangency_condition z ∧ room_height(z) = 2 + 4 * Real.sqrt 2 :=
sorry

end find_room_height_l719_719037


namespace min_value_of_sum_of_squares_l719_719954

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x * y + y * z + x * z = 4) :
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_value_of_sum_of_squares_l719_719954


namespace value_of_p_l719_719098

theorem value_of_p (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 3 * x2 ∧ x^2 - (3 * p - 2) * x + p^2 - 1 = 0) →
  (p = 2 ∨ p = 14 / 11) :=
by
  sorry

end value_of_p_l719_719098


namespace part1_solution_part2_solution_l719_719195

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |x - 1|

theorem part1_solution :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x} :=
by
  sorry

theorem part2_solution (x0 : ℝ) :
  (∃ x0 : ℝ, ∀ t : ℝ, f x0 < |(x0 + t)| + |(t - x0)|) →
  ∀ m : ℝ, (f x0 < |m + t| + |t - m|) ↔ m ≠ 0 ∧ (|m| > 5 / 4) :=
by
  sorry

end part1_solution_part2_solution_l719_719195


namespace area_triangle_MDA_zero_l719_719925

noncomputable def circle_config (O A B M D : Type) : Prop :=
  ∃ (r : ℝ), 
  ∃ (h_circle_radius : distance O A = r ∧ distance O B = r),
  ∃ (h_chord_length : distance A B = 2 * r),
  ∃ (h_midpoint_M : midpoint A B M),
  ∃ (h_perpendicular_OM : linear_independent ℝ (![subvector.singleton (O - M) (1:ℝ), subvector.singleton (M - A) (1:ℝ)])) ∧
  ∃ (h_perpendicular_MO : linear_independent ℝ (![subvector.singleton (M - A) (1:ℝ), subvector.singleton (A - D) (1:ℝ)])) 

theorem area_triangle_MDA_zero (O A B M D : Type) (h : circle_config O A B M D) : 
  triangle_area M D A = 0 :=
sorry

end area_triangle_MDA_zero_l719_719925


namespace limit_of_2a_n_over_n_square_plus_1_l719_719635

noncomputable def a_n (n : ℕ) : ℝ := (1 / 2 : ℝ) * n * (n - 1)

theorem limit_of_2a_n_over_n_square_plus_1 :
  tendsto (λ n, (2 * a_n n) / (n^2 + 1 : ℝ)) atTop (𝓝 1) :=
by
  sorry

end limit_of_2a_n_over_n_square_plus_1_l719_719635


namespace david_total_hours_on_course_l719_719424

def hours_per_week_class := 2 * 3 + 4 -- hours per week in class
def hours_per_week_homework := 4 -- hours per week in homework
def total_hours_per_week := hours_per_week_class + hours_per_week_homework -- total hours per week

theorem david_total_hours_on_course :
  let total_weeks := 24
  in total_weeks * total_hours_per_week = 336 := by
  sorry

end david_total_hours_on_course_l719_719424


namespace compute_abs_diff_l719_719259

noncomputable def fractional_part (z : ℝ) : ℝ := z - z.floor

theorem compute_abs_diff (x y : ℝ) 
  (h1 : x.floor - fractional_part y = 1.3)
  (h2 : fractional_part x + y.floor = 3.7) : 
  |x - y| = 1 :=
by
  sorry

end compute_abs_diff_l719_719259


namespace sin_dihedral_angle_l719_719226

variable (α : ℝ)
variable (A B C A1 B1 C1 P : ℝ × ℝ × ℝ)
variable (length_eq : ∀ x y, (x ∈ {A, B, C, A1, B1, C1}) → (y ∈ {A, B, C, A1, B1, C1}) → x ≠ y → dist x y = 2)
variable (P_midpoint : P = ((C.1 + C1.1) / 2, (C.2 + C1.2) / 2, (C.3 + C1.3) / 2))

theorem sin_dihedral_angle :
  sin α = √10 / 4 :=
sorry

end sin_dihedral_angle_l719_719226


namespace statement_incorrect_A_statement_incorrect_D_l719_719010

theorem statement_incorrect_A : 
  let m := -1 / 2
  in m ≠ -Real.sqrt 3 := by
  have m := -1 / 2
  show m ≠ -Real.sqrt 3
  sorry

theorem statement_incorrect_D :
  let a := 5
  let b := 4
  let short_axis_length := 2 * b
  in short_axis_length ≠ 4 := by
  have a := 5
  have b := 4
  let short_axis_length := 2 * b
  show short_axis_length ≠ 4
  sorry

end statement_incorrect_A_statement_incorrect_D_l719_719010


namespace complex_z_value_l719_719042

noncomputable def complex_z : ℂ :=
  sqrt 2 * (complex.cos (π / 4) - complex.sin (π / 4) * complex.I)

theorem complex_z_value :
  ∃ z : ℂ, z * (1 + complex.I) = complex.abs (1 - sqrt 3 * complex.I) ∧
           z = complex_z := by
  sorry

end complex_z_value_l719_719042


namespace integer_roots_quadratic_21_real_values_l719_719211

theorem integer_roots_quadratic_21_real_values (a : ℝ) :
  (∃ r s : ℤ, r + s = -a ∧ r * s = 24 * a) ↔ a ∈ {x ∈ ℝ | ∃ d : ℕ, d ∈ finset.Icc 1 576 ∧ a = (-d.val.to_real + d.val.inv.to_real) - 24} :=
by
  sorry

end integer_roots_quadratic_21_real_values_l719_719211


namespace sum_of_first_9_terms_l719_719214

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- a_n is the nth term of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of first n terms of the arithmetic sequence
def sum_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Hypotheses
axiom h1 : 2 * a 8 = 6 + a 11
axiom h2 : arithmetic_seq a
axiom h3 : sum_seq S a

-- The theorem we want to prove
theorem sum_of_first_9_terms : S 9 = 54 :=
sorry

end sum_of_first_9_terms_l719_719214


namespace root_and_value_of_a_equation_has_real_roots_l719_719512

theorem root_and_value_of_a (a : ℝ) (other_root : ℝ) :
  (∃ x : ℝ, x^2 + a * x + a - 1 = 0 ∧ x = 2) → a = -1 ∧ other_root = -1 :=
by sorry

theorem equation_has_real_roots (a : ℝ) :
  ∃ x : ℝ, x^2 + a * x + a - 1 = 0 :=
by sorry

end root_and_value_of_a_equation_has_real_roots_l719_719512


namespace math_problem_l719_719124

-- Definition to check the condition for interior angle being an integer 
def is_valid_n (n : ℕ) : Prop :=
  n ≥ 3 ∧ (360 % n = 0)

-- Definition for the sum of valid n's
def sum_of_valid_ns : ℕ :=
  Finset.sum (Finset.filter is_valid_n (Finset.range 361)) id

theorem math_problem : sum_of_valid_ns = 1167 :=
by
  -- Skipping the proof
  sorry

end math_problem_l719_719124


namespace constant_term_binomial_expansion_l719_719300

theorem constant_term_binomial_expansion :
  let binom := (λ x : ℝ, (1 / sqrt x) - (x / 2)) in
  constant_term (binom x)^9 = -21 / 2 :=
by
  sorry

end constant_term_binomial_expansion_l719_719300


namespace distance_to_y_axis_parabola_midpoint_l719_719174

noncomputable def distance_from_midpoint_to_y_axis (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_to_y_axis_parabola_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), y1^2 = x1 → y2^2 = x2 → 
  abs (x1 + 1 / 4) + abs (x2 + 1 / 4) = 3 →
  abs (distance_from_midpoint_to_y_axis x1 x2) = 5 / 4 :=
by
  intros x1 y1 x2 y2 h1 h2 h3
  sorry

end distance_to_y_axis_parabola_midpoint_l719_719174


namespace solution_set_eq_l719_719181

variable (f : ℝ → ℝ)
variable (h1 : ∀ x : ℝ, f x + x * (deriv f x) < 0)
variable (h2 : f 2 = 1)

theorem solution_set_eq : {x : ℝ | (x + 1) * f (x + 1) > 2} = set.Iio 1 :=
by sorry

end solution_set_eq_l719_719181


namespace largest_power_of_prime_divides_factorial_l719_719474

theorem largest_power_of_prime_divides_factorial (p : ℕ) (hp : p.Prime) : 
  ∃ n : ℕ, (∀ m : ℕ, m < p + 1 → (∃ k : ℕ, k = m ∧ (p!)^k ∣ (p^2)!)) ∧ 
           (∀ k : ℕ, (p!)^(p + 1) ∣ (p^2)!) :=
begin
  sorry
end

end largest_power_of_prime_divides_factorial_l719_719474


namespace infinite_series_converges_to_3_l719_719786

noncomputable def sum_of_series := ∑' k in (Finset.range ∞).filter (λ k, k > 0), 
  (8 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))

theorem infinite_series_converges_to_3 : sum_of_series = 3 := 
  sorry

end infinite_series_converges_to_3_l719_719786


namespace area_transformed_function_l719_719317

noncomputable def area_g : ℝ := 15

noncomputable def area_4g_shifted : ℝ :=
  4 * area_g

theorem area_transformed_function :
  area_4g_shifted = 60 := by
  sorry

end area_transformed_function_l719_719317


namespace evaluate_f_l719_719183

def f (x : ℝ) : ℝ := sorry  -- Placeholder function definition

theorem evaluate_f :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 5/2) = -1 / f x) ∧
  (∀ x : ℝ, x ∈ [-5/2, 0] → f x = x * (x + 5/2))
  → f 2016 = 3/2 :=
by
  sorry

end evaluate_f_l719_719183


namespace angle_B_is_30_l719_719570

noncomputable def find_angle_B (A a b : ℝ) (hA : A = 120) (ha : a = 2) (hb : b = 2 * real.sqrt 3 / 3) : ℝ :=
  let sin120 := real.sin (120 * real.pi / 180) in
  let sinB := b * sin120 / a in
  real.arcsin (sinB) * 180 / real.pi

theorem angle_B_is_30 : 
  find_angle_B 120 2 (2 * real.sqrt 3 / 3) 120 rfl rfl rfl = 30 :=
sorry

end angle_B_is_30_l719_719570


namespace find_general_formula_maximize_terms_sum_l719_719158

noncomputable def sequence (a1 : ℝ) (λ : ℝ) (Sn : ℕ → ℝ) (an : ℕ → ℝ) :=
  (λ > 0) ∧ (a1 ≠ 0) ∧ (∀ n, λ * a1 * an n = Sn 0 + Sn n)

theorem find_general_formula
  (a1 : ℝ) (λ : ℝ) (Sn : ℕ → ℝ) (an : ℕ → ℝ)
  (h : sequence a1 λ Sn an) :
  ∀ n, an n = (2 : ℝ)^n / λ := 
sorry

theorem maximize_terms_sum 
  (a1 : ℝ) (λ : ℝ) (Sn : ℕ → ℝ) (an : ℕ → ℝ)
  (h : sequence a1 λ Sn an)
  (h_a1_pos : a1 > 0) 
  (h_λ_100 : λ = 100) :
  ∃ n, n = 6 ∧ 
       (∀ m, m < 7 → log (1 / an m) ≤ log (1 / an 6)) := 
sorry

end find_general_formula_maximize_terms_sum_l719_719158


namespace problem1_problem2_l719_719780

-- Define each part of the problem's expressions as separate terms
def expr1 : ℕ := (2 + 3 / 5)^0 + 2^(-2) * (2 + 1 / 4)^(-1 / 2) + (25 / 36)^0.5 + Real.sqrt ((-2)^2)
def expr2 : ℚ := 1 / 2 * Real.Log.log (32 / 49) - 4 / 3 * Real.Log.log (√8) + Real.Log.log (√245) + 2^(1 + Real.Log.log 3 / Real.Log.log 2)

theorem problem1 : expr1 = 4 := by
  sorry

theorem problem2 : expr2 = 13 / 2 := by
  sorry

end problem1_problem2_l719_719780


namespace sequence_convergence_l719_719643

noncomputable theory

open Classical

variable {a : ℕ → ℝ}

-- The sequence has positive terms
axiom pos_terms (n : ℕ) : a n > 0

-- The inequality condition for the sequence
axiom inequality_condition (i j : ℕ) : (i + j) * a (i + j) ≤ j * a i + i * a j

-- The main theorem to prove
theorem sequence_convergence :
  ∃ A, (∀ n, a n ≤ a 1) ∧ (∀ ε > 0, ∃ N, ∀ n ≥ N, a n < A + ε) :=
by
  sorry

end sequence_convergence_l719_719643


namespace wire_length_is_approx_84_029_l719_719365

noncomputable def length_of_wire (diameter_mm : Float) (volume_cm3 : Float) : Float :=
  let diameter_m := diameter_mm * 0.001
  let radius_m := diameter_m / 2
  let volume_m3 := volume_cm3 * 1e-6
  volume_m3 / (Float.pi * radius_m^2)

theorem wire_length_is_approx_84_029 :
  length_of_wire 1 66 ≈ 84.029 :=
  sorry

end wire_length_is_approx_84_029_l719_719365


namespace lego_airplane_models_l719_719578

theorem lego_airplane_models (L : ℕ) (N : ℕ) (M : ℕ) (A : ℕ) :
  L = 400 → N = 80 → M = 2 →
  A = (L + N) / M →
  A = 240 :=
by
  intros hL hN hM hA
  rw [hL, hN, hM] at hA
  exact hA
  sorry

end lego_airplane_models_l719_719578


namespace father_20_bills_count_l719_719277

-- Defining the conditions from the problem.
variables (mother50 mother20 mother10 father50 father10 : ℕ)
def mother_total := mother50 * 50 + mother20 * 20 + mother10 * 10
def father_total (x : ℕ) := father50 * 50 + x * 20 + father10 * 10

-- Given conditions
axiom mother_given : mother50 = 1 ∧ mother20 = 2 ∧ mother10 = 3
axiom father_given : father50 = 4 ∧ father10 = 1
axiom school_fee : 350 = 350

-- Theorem to prove
theorem father_20_bills_count (x : ℕ) :
  mother_total 1 2 3 + father_total 4 x 1 = 350 → x = 1 :=
by sorry

end father_20_bills_count_l719_719277


namespace parallelogram_construction_l719_719091

-- Define the variables for vertices and point P
variables {A B C D P : ℝ×ℝ} -- assuming coordinates in the plane

-- Define the distance function between two points
def dist (X Y : ℝ×ℝ) : ℝ := real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)

-- Define conditions: distances from P to A, B, C, D and the diagonal AC
variables {PA PB PC PD AC : ℝ}

-- State the main theorem
theorem parallelogram_construction (hAC : dist A C = AC)
    (hPA : dist P A = PA)
    (hPB : dist P B = PB)
    (hPC : dist P C = PC)
    (hPD : dist P D = PD):
    -- The construction yields exactly 4 distinct parallelograms
    ∃ ABCD : (ℝ×ℝ) × (ℝ×ℝ) × (ℝ×ℝ) × (ℝ×ℝ), 
    4 = card {parallelogram | let ⟨A, B, C, D⟩ := parallelogram in 
        dist P A = PA ∧
        dist P B = PB ∧
        dist P C = PC ∧
        dist P D = PD ∧
        dist A C = AC ∧
        (dist A B = dist C D ∧ dist B C = dist D A ∧ dist A D = dist B C)} :=
sorry

end parallelogram_construction_l719_719091


namespace largest_possible_x_l719_719926

theorem largest_possible_x
  (x y : ℕ)
  (h1 : x > y)
  (h2 : x + y ≤ 2008)
  (h3 : 2 * (x * (x - 1) / 2 + y * (y - 1) / 2) = (x + y) * (x + y - 1)) :
  x ≤ 990 :=
sorry

end largest_possible_x_l719_719926


namespace binomial_coeff_sum_and_constant_term_l719_719509

theorem binomial_coeff_sum_and_constant_term (x : ℝ) (a : ℝ) (n : ℕ)
  (h1 : 2^n = 32)
  (h2 : (finset.range (n + 1)).sum (λ k, nat.choose n k * ((real.sqrt x)^(n - k)) * 
        ((a / (real.cbrt x))^k)) = 80) :
  a = 2 :=
sorry

end binomial_coeff_sum_and_constant_term_l719_719509


namespace sum_series_eq_two_l719_719798

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l719_719798


namespace max_possible_score_l719_719813

-- Define the conditions of the disks
structure Disk (n : ℕ) :=
  (center : Nat)
  (radius : Nat)

-- Predicate to express the condition that a disk properly contains another disk
def properly_contains (C_i C_j : Disk) : Prop :=
  C_i.radius > C_j.radius

-- Define the score of an arrangement of disks 
def score (disks : List (Disk n)) : Nat :=
  List.sum (List.map (λ i, List.length (List.filter (λ j, properly_contains (disks.get i) (disks.get j)) disks)) (List.range (List.length disks)))

-- Theorem to state the maximum possible score
theorem max_possible_score (disks : List (Disk n)) : score disks ≤ (n-1)*(n-2)/2 := 
  sorry

end max_possible_score_l719_719813


namespace not_all_squares_congruent_to_each_other_l719_719011

def Square (a : Type) [OrderedRing a] := { x // x ∈ setOf  (x : a × a × a × a) (x.1 = a ∧ x.2 = a ∧ x.3 = a ∧ x.4 = a)}

variables (a : Type) [OrderedRing a]

theorem not_all_squares_congruent_to_each_other : 
  ∃ (sq1 sq2 : Square a), sq1 ≠ sq2 ∧ ¬(sq1 ≅ sq2) :=
sorry

end not_all_squares_congruent_to_each_other_l719_719011


namespace monotonous_count_between_1_and_9999_l719_719429

def is_monotonous (n : Nat) : Prop :=
  if n < 10 then True
  else 
    let digits := List.digits n
    List.strict_sorted (· < ·) digits ∨
    List.strict_sorted (· > ·) digits ∨
    List.all_eq digits

def monotonous_count := Finset.card (Finset.filter is_monotonous (Finset.range 10000))

theorem monotonous_count_between_1_and_9999 : monotonous_count = 556 := by
  sorry

end monotonous_count_between_1_and_9999_l719_719429


namespace number_count_correct_l719_719901

def number_count : ℕ :=
  let digit_choices := Nat.choose 9 1 * Nat.choose 8 6 in
  let permutations := Fact.mk 7 -+ Nat.choose 6 1 * (Fact.mk 7 / Fact.mk 2)
  digit_choices * permutations

theorem number_count_correct : number_count = 5080320 := by
  sorry

end number_count_correct_l719_719901


namespace angle_between_vectors_l719_719912

variables (a b : ℝ^3)
variables (hab : 0 < ∥a∥ ∧ 0 < ∥b∥)
variables (h1 : ∥a∥ = 3 * ∥b∥)
variables (h2 : dot (2 • a + 3 • b) b = 0)
  
theorem angle_between_vectors :
  real.angle a b = real.angle (2/3 * real.pi) :=
sorry

end angle_between_vectors_l719_719912


namespace intersection_proof_l719_719205

open Set

def M : Set ℝ := {x | x^2 - 3 * x - 28 ≤ 0 }
def N : Set ℝ := {x | x^2 - x - 6 > 0 }
def correct_answer : Set ℝ := {x | (-4 ≤ x ∧ x < -2) ∨ (3 < x ∧ x ≤ 7)}

theorem intersection_proof : M ∩ N = correct_answer := 
by
  sorry

end intersection_proof_l719_719205


namespace valid_triplets_l719_719833

theorem valid_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_leq1 : a ≤ b) (h_leq2 : b ≤ c)
  (h_div1 : a ∣ (b + c)) (h_div2 : b ∣ (a + c)) (h_div3 : c ∣ (a + b)) :
  (a = b ∧ b = c) ∨ (a = b ∧ c = 2 * a) ∨ (b = 2 * a ∧ c = 3 * a) :=
sorry

end valid_triplets_l719_719833


namespace range_of_a_l719_719897

variables {a x : ℝ}

def P (a : ℝ) : Prop := ∀ x, ¬ (x^2 - (a + 1) * x + 1 ≤ 0)

def Q (a : ℝ) : Prop := ∀ x, |x - 1| ≥ a + 2

theorem range_of_a (a : ℝ) : 
  (¬ P a ∧ ¬ Q a) → a ≥ 1 :=
by
  sorry

end range_of_a_l719_719897


namespace seventh_common_multiple_of_4_and_6_l719_719770

theorem seventh_common_multiple_of_4_and_6 : (∃ n : ℕ, n > 0 ∧ n < 100 ∧ (n % 4 = 0) ∧ (n % 6 = 0) ∧ list.nth_le (list.filter (λ n, (n % 4 = 0 ∧ n % 6 = 0)) (list.range 100) (by linarith)) 6 sorry = 84) :=
sorry

end seventh_common_multiple_of_4_and_6_l719_719770


namespace midpoints_collinear_l719_719705

noncomputable def midpoint (A B : Point) : Point := sorry

variables {A B C D E F : Point}

-- Given conditions
variable (h1 : is_diameter A B (semicircle A B C D)) -- C and D are distinct points on a semicircle with diameter [AB]
variable (h2 : distinct C D)
variable (h3 : intersect_lines (line_through A C) (line_through B D) = F)
variable (h4 : intersect_lines (line_through A D) (line_through B C) = E)

-- Definitions of midpoints
def midpoint_AB := midpoint A B
def midpoint_CD := midpoint C D
def midpoint_EF := midpoint E F

-- Goal
theorem midpoints_collinear : collinear (midpoint_AB h1 h2) (midpoint_CD h1 h2) (midpoint_EF h1 h2 h3 h4) := sorry

end midpoints_collinear_l719_719705


namespace proof_problem_l719_719182

noncomputable theory

variables {ℝ : Type} [LinearOrderedField ℝ]

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = f (2 * a - x)

def periodic (g : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, g (x + p) = g x

theorem proof_problem 
  (f g : ℝ → ℝ)
  (hf_dom : ∀ x, f x ≠ ⊥)
  (hg_dom : ∀ x, g x ≠ ⊥)
  (H1 : ∀ x, f x + g (x + 2) = 1)
  (H2 : ∀ x, f (x - 4) - g x = 3)
  (H3 : symmetric_about f 1)
  (H4 : f (-1) = 0) :
  (g 1 = 1) ∧ 
  (periodic g 4) ∧ 
  (symmetric_about g 4 ∧ g 4 = -1) ∧ 
  (g 2 = -1) := 
by 
  sorry

end proof_problem_l719_719182


namespace smallest_N_gt_1395_l719_719296

/-- Define the sequence satisfying the conditions. --/
def seq (a : ℤ) : ℕ → ℤ
| 1       := a
| (2*k+1) := 2*(seq a (2*k)) - 1
| (2*k+2) := 2*seq a (2*k+1)

theorem smallest_N_gt_1395 (a : ℤ) (n > 10) :
  ∃ a : ℤ, ∃ n > 10, seq a n = 1409 ∧ ∀ m > 10, seq a m ≠ 1409 → 1409 > 1395 :=
sorry -- Proof omitted

end smallest_N_gt_1395_l719_719296


namespace range_of_m_l719_719949

variable {R : Type*} [LinearOrder R] [HasSub R] [Add R] [Neg R]

def odd_function (f : R → R) :=
  ∀ x : R, f (-x) = -f x

def monotonically_decreasing (f : R → R) (I : Set R) :=
  ∀ ⦃x y : R⦄, x ∈ I → y ∈ I → x < y → f x > f y

theorem range_of_m (f : R → R) (m : R) :
  odd_function f →
  monotonically_decreasing f (Set.Icc (0 : R) (2 : R)) →
  f m + f (m - 1) > 0 →
  -1 ≤ m ∧ m < (1 / 2 : R) := by
  sorry

end range_of_m_l719_719949


namespace count_paths_no_consecutive_restriction_l719_719902

theorem count_paths_no_consecutive_restriction : 
  let total_steps := 13,
      steps_up := 6,
      steps_right := 7
  in binomial total_steps steps_up = 1716 :=
by
  let total_steps := 13
  let steps_up := 6
  let steps_right := 7
  show binomial total_steps steps_up = 1716
  sorry

end count_paths_no_consecutive_restriction_l719_719902


namespace least_value_in_S_l719_719262

def S := {n : ℕ | n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}

theorem least_value_in_S {S : set ℕ} (hS1 : S ⊆ {n | n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}) 
    (hS2 : S.card = 8) 
    (hS3 : ∀ a b, a ∈ S → b ∈ S → a < b → ¬ (b % a = 0)):  ∃ x, x ∈ S ∧ ∀ y ∈ S, y ≥ x ∧ x = 4 :=
by {
    sorry
}

end least_value_in_S_l719_719262


namespace exterior_angle_of_coplanar_triangle_and_nonagon_l719_719387

noncomputable def exterior_angle_BAC (n : ℕ) (triangle_angle : ℝ) : ℝ :=
  let nonagon_angle := (180 * (n - 2)) / n
  360 - nonagon_angle - triangle_angle

theorem exterior_angle_of_coplanar_triangle_and_nonagon
        (n : ℕ) (h_n : n = 9) (triangle_angle : ℝ) (h_triangle : triangle_angle = 60) :
  exterior_angle_BAC n triangle_angle = 160 :=
by
  have h_nonagon_angle : (180 * (n - 2)) / n = 140 := by
    simp [h_n]
    norm_num
  rw [exterior_angle_BAC, h_nonagon_angle, h_triangle]
  norm_num
  sorry

end exterior_angle_of_coplanar_triangle_and_nonagon_l719_719387


namespace find_a7_l719_719848

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def Sn_for_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a7 (h_arith : arithmetic_sequence a)
  (h_sum_property : Sn_for_arithmetic_sequence a S)
  (h1 : a 2 + a 5 = 4)
  (h2 : S 7 = 21) :
  a 7 = 9 :=
sorry

end find_a7_l719_719848


namespace six_times_number_eq_132_l719_719436

theorem six_times_number_eq_132 (x : ℕ) (h : x / 11 = 2) : 6 * x = 132 :=
sorry

end six_times_number_eq_132_l719_719436


namespace numberOfBaseballBoxes_l719_719409

-- Given conditions as Lean definitions and assumptions
def numberOfBasketballBoxes : ℕ := 4
def basketballCardsPerBox : ℕ := 10
def baseballCardsPerBox : ℕ := 8
def cardsGivenToClassmates : ℕ := 58
def cardsLeftAfterGiving : ℕ := 22

def totalBasketballCards : ℕ := numberOfBasketballBoxes * basketballCardsPerBox
def totalCardsBeforeGiving : ℕ := cardsLeftAfterGiving + cardsGivenToClassmates

-- Target number of baseball cards
def totalBaseballCards : ℕ := totalCardsBeforeGiving - totalBasketballCards

-- Prove that the number of baseball boxes is 5
theorem numberOfBaseballBoxes :
  totalBaseballCards / baseballCardsPerBox = 5 :=
sorry

end numberOfBaseballBoxes_l719_719409


namespace probability_three_red_balls_l719_719552

open scoped BigOperators

noncomputable def hypergeometric_prob (r : ℕ) (b : ℕ) (k : ℕ) (d : ℕ) : ℝ :=
  (Nat.choose r d * Nat.choose b (k - d) : ℝ) / Nat.choose (r + b) k

theorem probability_three_red_balls :
  hypergeometric_prob 10 5 5 3 = 1200 / 3003 :=
by sorry

end probability_three_red_balls_l719_719552


namespace sum_numerator_denominator_l719_719375

theorem sum_numerator_denominator (a b : ℕ) (h_coprime : Nat.gcd a b = 1)
    (h_condition : ((2/3:ℚ) * a^2) / (0.20 * b^2) = 2 * (a / (b:ℚ))) :
    a + b = 8 :=
begin
  sorry
end

end sum_numerator_denominator_l719_719375


namespace infinite_sum_problem_l719_719806

theorem infinite_sum_problem : 
  (∑ k in (set.Ici 1), (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 1 :=
by
  sorry

end infinite_sum_problem_l719_719806


namespace approx_root_of_quadratic_l719_719644

noncomputable def quadratic (x : ℝ) : ℝ := x^2 + 3 * x - 5

def table_values : List (ℝ × ℝ) := 
  [(1, -1), (1.1, -0.49), (1.2, 0.04), (1.3, 0.59), (1.4, 1.16)]

theorem approx_root_of_quadratic : ∃ x : ℝ, quadractic x = 0 :=
by
  have transition : (∃ x : ℝ in {1.1, 1.2}, quadratic x = 0) := sorry
  exact transition

end approx_root_of_quadratic_l719_719644


namespace dietitian_lunch_fraction_l719_719741

theorem dietitian_lunch_fraction
  (total_calories : ℕ)
  (recommended_calories : ℕ)
  (extra_calories : ℕ)
  (h1 : total_calories = 40)
  (h2 : recommended_calories = 25)
  (h3 : extra_calories = 5)
  : (recommended_calories + extra_calories) / total_calories = 3 / 4 :=
by
  sorry

end dietitian_lunch_fraction_l719_719741


namespace donuts_left_for_coworkers_l719_719413

theorem donuts_left_for_coworkers :
  let t := 2.5 * 12 in
  let g := 12 in
  let r := t - g in
  let gluten_free_eaten := ⌊0.10 * g⌋ in
  let regular_eaten := ⌊0.05 * r⌋ in
  let total_gluten_free_eaten := gluten_free_eaten + 2 in
  let total_regular_eaten := regular_eaten + 4 in
  let remaining_gluten_free := g - total_gluten_free_eaten in
  let remaining_regular := r - total_regular_eaten in
  remaining_gluten_free + remaining_regular = 23 := by
  sorry

end donuts_left_for_coworkers_l719_719413


namespace a_2_eq_3_a_3_eq_4_a_4_eq_5_a_n_formula_a_1_formula_l719_719608

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 2
| (n+1) => a n ^ 2 - n * a n + 1

-- Part (Ⅰ): Verify specific terms in the sequence
theorem a_2_eq_3 : a 2 = 3 :=
by
  rw [a, a]
  -- additional detailed steps would be necessary to finish this proof
  sorry

theorem a_3_eq_4 : a 3 = 4 :=
by
  rw [a, a]
  -- additional detailed steps would be necessary to finish this proof
  sorry

theorem a_4_eq_5 : a 4 = 5 :=
by
  rw [a, a]
  -- additional detailed steps would be necessary to finish this proof
  sorry

-- Part (Ⅱ): Conjecture the general formula and prove via induction
theorem a_n_formula (n : ℕ) : 1 < n → a n = n + 1 := 
by
  intro h
  induction n with
  | zero => sorry
  | succ n IH => sorry

-- Ensure that the initial term aligns with the conjectured formula
theorem a_1_formula : a 1 = 1 + 1 := rfl

end a_2_eq_3_a_3_eq_4_a_4_eq_5_a_n_formula_a_1_formula_l719_719608


namespace probability_floor_condition_zero_l719_719941

def P (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 20

theorem probability_floor_condition_zero :
  (∃ a b c d e : ℕ, a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0 ∧ 
   (P x ∈ set.Icc 3 10) → (floor (P x) = floor (P (floor x)) → 0)) := by {
  sorry
}

end probability_floor_condition_zero_l719_719941


namespace find_a_l719_719495

-- Define the real numbers x, y, and a
variables (x y a : ℝ)

-- Define the conditions as premises
axiom cond1 : x + 3 * y + 5 ≥ 0
axiom cond2 : x + y - 1 ≤ 0
axiom cond3 : x + a ≥ 0

-- Define z as x + 2y and state its minimum value is -4
def z : ℝ := x + 2 * y
axiom min_z : z = -4

-- The theorem to prove the value of a given the above conditions
theorem find_a : a = 2 :=
sorry

end find_a_l719_719495


namespace sarah_milk_amount_l719_719829

theorem sarah_milk_amount
  (initial_milk : ℚ := 1/4)
  (rachel_fraction : ℚ := 2/3)
  (sarah_fraction : ℚ := 1/2) :
  let rachel_milk := rachel_fraction * initial_milk,
      remaining_milk := initial_milk - rachel_milk,
      sarah_milk := sarah_fraction * remaining_milk
  in sarah_milk = 1/24 :=
by
  sorry

end sarah_milk_amount_l719_719829


namespace final_mileage_after_trip_l719_719407

-- Define the conditions as constants
def initial_miles : ℕ := 1728
def fill_ups : ℕ := 2
def miles_per_gallon : ℕ := 30
def gallon_capacity : ℕ := 20

-- Define the final calculation as the proof problem.
theorem final_mileage_after_trip : initial_miles + (fill_ups * gallon_capacity * miles_per_gallon) = 2928 := 
by 
  exact congrArg (· + initial_miles) (congrArg (· * miles_per_gallon) (congArg (· * gallon_capacity) fill_ups)).trans sorry

end final_mileage_after_trip_l719_719407


namespace can_color_all_pebbles_l719_719682

-- Define the size of the chessboard
def chessboard_size : Nat := 8

-- Define the number of pebbles
def pebbles_count : Nat := 12

-- Define a function that takes a set of coordinates of pebbles and returns True if we can find 4 rows and 4 columns covering all pebbles
def can_color_rows_and_columns (pebbles : Finset (Fin chessboard_size × Fin chessboard_size)) : Prop :=
  ∃ (rows columns : Finset (Fin chessboard_size)), 
    rows.card = 4 ∧ 
    columns.card = 4 ∧ 
    ∀ (pebble ∈ pebbles), ∃ (r ∈ rows) (c ∈ columns), (r, c) = pebble

-- Statement: Prove that it is possible to color 4 rows and 4 columns such that all pebbles are covered
theorem can_color_all_pebbles (pebbles : Finset (Fin chessboard_size × Fin chessboard_size)) 
  (h_count : pebbles.card = pebbles_count) :
  can_color_rows_and_columns pebbles :=
by
  sorry

end can_color_all_pebbles_l719_719682


namespace pb5_l719_719928

noncomputable def midpoint (Q R M : Point) : Prop := (M.x = (Q.x + R.x) / 2) ∧ (M.y = (Q.y + R.y) / 2)

noncomputable def incircle (C : Circle) (P Q R : Point) : Prop := C.isTangentTriangle P Q R

theorem pb5 (C : Circle) (L : Line) (M : Point) (P : Point) :
  (L.isTangent C) ∧ (M ∈ L) → 
  (∃ Q R : Point, (Q ∈ L) ∧ (R ∈ L) ∧ midpoint Q R M ∧ incircle C P Q R)
  → P ∈ ((line_through D E) \ {D}) :=
by sorry

end pb5_l719_719928


namespace angleQ_is_80_l719_719276

-- Definitions of angles and lines being parallel
variables (m n : Line) (P Q R : Point)
variables (angleP angleR angleQ : ℝ)

-- Definitions based on given conditions
def m_parallel_n : Prop := m.parallel n
def angleP_is_100 : Prop := angleP = 100
def angleR_is_50 : Prop := angleR = 50

-- Proof statement
theorem angleQ_is_80 (h1 : m_parallel_n) (h2 : angleP_is_100) (h3 : angleR_is_50) : angleQ = 80 := 
sorry

end angleQ_is_80_l719_719276


namespace chris_raisins_nuts_l719_719085

theorem chris_raisins_nuts (R N x : ℝ) 
  (hN : N = 4 * R) 
  (hxR : x * R = 0.15789473684210525 * (x * R + 4 * N)) :
  x = 3 :=
by
  sorry

end chris_raisins_nuts_l719_719085


namespace max_value_expr_l719_719871

-- Define the conditions given in the problem
variables {a b : ℝ}
variables (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1)

-- Define the expression to be maximized
def expr := 2 * real.sqrt (a * b) - 4 * a^2 - b^2

-- Statement of the problem as a Lean theorem
theorem max_value_expr : 
  (∀ a b, a > 0 → b > 0 → 2 * a + b = 1 → expr a b ≤ (real.sqrt 2 - 1) / 2) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ expr a b = (real.sqrt 2 - 1) / 2) :=
by
  sorry

end max_value_expr_l719_719871


namespace cos_of_angle_in_third_quadrant_l719_719216

theorem cos_of_angle_in_third_quadrant (A : ℝ) (hA : π < A ∧ A < 3 * π / 2) (h_sin : Real.sin A = -1 / 3) :
  Real.cos A = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_of_angle_in_third_quadrant_l719_719216


namespace sum_of_possible_values_M_l719_719473

theorem sum_of_possible_values_M {α : Type} [LinearOrderedField α]
  (lines : Finset (α × α × α)) (h : lines.card = 3)
  (M : ℕ → Prop)
  (h_possible_values : ∀ n, M n ↔ n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3) :
  (Finset.sum (Finset.filter M {n | n ≤ 3}) id = 6) :=
by sorry

end sum_of_possible_values_M_l719_719473


namespace cos_alpha_minus_beta_l719_719142

theorem cos_alpha_minus_beta : 
  ∀ (α β : ℝ), 
  2 * Real.cos α - Real.cos β = 3 / 2 →
  2 * Real.sin α - Real.sin β = 2 →
  Real.cos (α - β) = -5 / 16 :=
by
  intros α β h1 h2
  sorry

end cos_alpha_minus_beta_l719_719142


namespace infinite_sum_problem_l719_719808

theorem infinite_sum_problem : 
  (∑ k in (set.Ici 1), (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 1 :=
by
  sorry

end infinite_sum_problem_l719_719808


namespace infinite_series_sum_l719_719445

theorem infinite_series_sum :
  (∑ k in Nat, (k^2 : ℝ)/(3^k : ℝ)) = 6 := sorry

end infinite_series_sum_l719_719445


namespace sockPairsCount_l719_719223

noncomputable def countSockPairs : ℕ :=
  let whitePairs := Nat.choose 6 2 -- 15
  let brownPairs := Nat.choose 7 2 -- 21
  let bluePairs := Nat.choose 3 2 -- 3
  let oneRedOneWhite := 4 * 6 -- 24
  let oneRedOneBrown := 4 * 7 -- 28
  let oneRedOneBlue := 4 * 3 -- 12
  let bothRed := Nat.choose 4 2 -- 6
  whitePairs + brownPairs + bluePairs + oneRedOneWhite + oneRedOneBrown + oneRedOneBlue + bothRed

theorem sockPairsCount : countSockPairs = 109 := by
  sorry

end sockPairsCount_l719_719223


namespace additional_amount_deductibles_next_year_l719_719974

theorem additional_amount_deductibles_next_year :
  let avg_deductible : ℝ := 3000
  let inflation_rate : ℝ := 0.03
  let plan_a_rate : ℝ := 2 / 3
  let plan_b_rate : ℝ := 1 / 2
  let plan_c_rate : ℝ := 3 / 5
  let plan_a_percent : ℝ := 0.40
  let plan_b_percent : ℝ := 0.30
  let plan_c_percent : ℝ := 0.30
  let additional_a : ℝ := avg_deductible * plan_a_rate
  let additional_b : ℝ := avg_deductible * plan_b_rate
  let additional_c : ℝ := avg_deductible * plan_c_rate
  let weighted_additional : ℝ := (additional_a * plan_a_percent) + (additional_b * plan_b_percent) + (additional_c * plan_c_percent)
  let inflation_increase : ℝ := weighted_additional * inflation_rate
  let total_additional_amount : ℝ := weighted_additional + inflation_increase
  total_additional_amount = 1843.70 :=
sorry

end additional_amount_deductibles_next_year_l719_719974


namespace longest_segment_cylinder_l719_719044

theorem longest_segment_cylinder (r h : ℤ) (c : ℝ) (hr : r = 4) (hh : h = 9) : 
  c = Real.sqrt (2 * r * r + h * h) ↔ c = Real.sqrt 145 :=
by
  sorry

end longest_segment_cylinder_l719_719044


namespace smallest_phi_l719_719915

def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

def shifted_f (phi : ℝ) (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3 - 2 * phi)

theorem smallest_phi (phi : ℝ) : 
    (∀ x, shifted_f phi x = shifted_f phi (-x)) → 
    0 < phi → 
    phi = π * 5 / 12 :=
by
  intros hphi hpos
  sorry

end smallest_phi_l719_719915


namespace convex_polygon_sides_l719_719549

theorem convex_polygon_sides (sum_angles_except_two : ℕ) (h_sum : sum_angles_except_two = 3420) : ∃ n : ℕ, n = 22 :=
by
  -- convert the condition into an equation based on the sum of interior angles formula
  let eq_sum := 180 * (22 - 2)
  -- verify the interior angle sum including two unspecific angles equals to the expected value
  have h_eq: 3420 + (100/: ℕ) = eq_sum,
  sorry

end convex_polygon_sides_l719_719549


namespace cos_F_of_right_triangle_l719_719227

theorem cos_F_of_right_triangle (D E F : Type*) [TopologicalSpace D] [TopologicalSpace E] [TopologicalSpace F]
  (DE : Real) (DF : Real) (DEF_hypotenuse : ∀ (EF : Real), EF = Real.sqrt (DE^2 + DF^2))
  (h_angle_D : DE = 8) (h_DF : DF = 15) :
  ∃ EF : Real, EF = 17 ∧ (DE = 8 ∧ DF = 15 → cos (arccos ((15 : Real) / 17)) = 15 / 17) :=
by
  sorry

end cos_F_of_right_triangle_l719_719227


namespace meadow_total_revenue_correct_l719_719966

-- Define the given quantities and conditions as Lean definitions
def total_diapers : ℕ := 192000
def price_per_diaper : ℝ := 4.0
def bundle_discount : ℝ := 0.05
def purchase_discount : ℝ := 0.05
def tax_rate : ℝ := 0.10

-- Define a function that calculates the revenue from selling all the diapers
def calculate_revenue (total_diapers : ℕ) (price_per_diaper : ℝ) (bundle_discount : ℝ) 
    (purchase_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let gross_revenue := total_diapers * price_per_diaper
  let bundle_discounted_revenue := gross_revenue * (1 - bundle_discount)
  let purchase_discounted_revenue := bundle_discounted_revenue * (1 - purchase_discount)
  let taxed_revenue := purchase_discounted_revenue * (1 + tax_rate)
  taxed_revenue

-- The main theorem to prove that the calculated revenue matches the expected value
theorem meadow_total_revenue_correct : 
  calculate_revenue total_diapers price_per_diaper bundle_discount purchase_discount tax_rate = 762432 := 
by
  sorry

end meadow_total_revenue_correct_l719_719966


namespace find_function_eq_identity_l719_719454

theorem find_function_eq_identity {f : ℕ → ℕ} (h1 : ∀ (x y : ℕ), f(f(x) + y) ∣ x + f(y)) : 
  ∀ x, f(x) = x :=
by
  sorry

end find_function_eq_identity_l719_719454


namespace sum_of_possible_values_l719_719266

theorem sum_of_possible_values (x y : ℝ)
  (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 5) :
  ∃ s : ℝ, s = (x - 2) * (y - 2) ∧ (s = -3 ∨ s = 9) :=
sorry

end sum_of_possible_values_l719_719266


namespace scientific_notation_of_area_l719_719638

theorem scientific_notation_of_area : 2720000 = 2.72 * 10^6 :=
by
  sorry

end scientific_notation_of_area_l719_719638


namespace exists_zero_in_interval_l719_719299

noncomputable def f (x : ℝ) : ℝ := Real.log x - (2 / x)

theorem exists_zero_in_interval :
  (∀ x y : ℝ, 0 < x → x < y → y < ⊤ → f x < f y) →
  f 2 < 0 →
  f Real.exp 1 > 0 →
  ∃ c : ℝ, 2 < c ∧ c < Real.exp 1 ∧ f c = 0 :=
by
  intros h_mono h_f2 h_fe
  sorry

end exists_zero_in_interval_l719_719299


namespace number_of_subsets_of_M_is_4_l719_719654

/-- Define the set M as {1, 2} --/
def M : Set ℕ := {1, 2}

/-- Prove that the number of subsets of M is 4 --/
theorem number_of_subsets_of_M_is_4 : (Set.powerset M).toFinset.card = 4 := 
sorry

end number_of_subsets_of_M_is_4_l719_719654


namespace probability_pass_only_two_levels_l719_719229

theorem probability_pass_only_two_levels : 
  let p1 := 0.8
  let p2 := 0.7
  let p3 := 0.6
  p1 * p2 * (1 - p3) = 0.224 :=
by 
  let p1 := 0.8
  let p2 := 0.7
  let p3 := 0.6
  calc 
    p1 * p2 * (1 - p3) = 0.8 * 0.7 * 0.4 : by rfl
                  ... = 0.224 : by norm_num

end probability_pass_only_two_levels_l719_719229


namespace fraction_difference_l719_719147

variable x y : ℝ
hypothesis hx : x = Real.sqrt 5 - 1
hypothesis hy : y = Real.sqrt 5 + 1

theorem fraction_difference : (1 / x - 1 / y = 1 / 2) :=
by 
  sorry

end fraction_difference_l719_719147


namespace ratio_green_to_yellow_l719_719086

noncomputable def ratio_of_areas (r1 r2 : ℝ) (h1 : r1 = 1) (h2 : r2 = 2) : ℝ :=
  let area_small := π * r1^2
  let area_large := π * r2^2
  let area_ring := area_large - area_small
  area_ring / area_small

theorem ratio_green_to_yellow :
  ratio_of_areas 1 2 rfl rfl = 3 := 
by
  sorry

end ratio_green_to_yellow_l719_719086


namespace abs_sum_zero_implies_diff_eq_five_l719_719539

theorem abs_sum_zero_implies_diff_eq_five (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 :=
  sorry

end abs_sum_zero_implies_diff_eq_five_l719_719539


namespace find_opposite_pair_l719_719768

-- Define the pairs of numbers as given conditions
def pair_A_lhs := -((-2) ^ 3)
def pair_A_rhs := | -2 | ^ 3
def pair_B_lhs := (-2) ^ 3
def pair_B_rhs := - (2 ^3)
def pair_C_lhs := -(2 ^ 2)
def pair_C_rhs := + ((-2) ^ 2)
def pair_D_lhs := -(-2)
def pair_D_rhs := | -2 |

-- Define the properties to check if two numbers are opposite
def are_opposites (a b : ℤ) : Prop := a = -b

-- The theorem to prove that only the pair C are opposites
theorem find_opposite_pair :
  (are_opposites pair_A_lhs pair_A_rhs = False) ∧
  (are_opposites pair_B_lhs pair_B_rhs = False) ∧
  (are_opposites pair_C_lhs pair_C_rhs = True) ∧
  (are_opposites pair_D_lhs pair_D_rhs = False) :=
by
  sorry

end find_opposite_pair_l719_719768


namespace inequality_proof_l719_719613

theorem inequality_proof (n : ℕ) (hn : 0 < n) :
  (finset.range (2^n - 1)).sum (λ k, 1 / (k + 1 : ℝ)) > n / 2 :=
sorry

end inequality_proof_l719_719613


namespace only_integer_solution_is_trivial_l719_719110

theorem only_integer_solution_is_trivial (a b c : ℤ) (h : 5 * a^2 + 9 * b^2 = 13 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end only_integer_solution_is_trivial_l719_719110


namespace parabola_conditions_l719_719458

open Real

theorem parabola_conditions (A B C : Point) (L : Line) :
    A = (1, -2) → B = (9, -6) → C = (4, 4) → L = (1, -2, 4) →
    ∃ λ : ℝ, 
      (L.1 + 2 * L.2 + 3) * (L.1 - 2 * L.2 + 4) + λ * (2 * L.1 + L.2 - 12) * (2 * L.1 - L.2 - 4) = 0 :=
by 
    intros hA hB hC hL
    have h_eq1 : (∃ λ : ℝ, (1 + 2 * (-2) + 3) * (1 - 2 * (-2) + 4) + λ * (2 * 1 + (-2) - 12) * (2 * 1 - (-2) - 4) = 0) :=
    sorry
    use λ
    exact h_eq1

end parabola_conditions_l719_719458


namespace pay_docked_per_lateness_l719_719900

variable (hourly_rate : ℤ) (work_hours : ℤ) (times_late : ℕ) (actual_pay : ℤ) 

theorem pay_docked_per_lateness (h_rate : hourly_rate = 30) 
                                (w_hours : work_hours = 18) 
                                (t_late : times_late = 3) 
                                (a_pay : actual_pay = 525) :
                                (hourly_rate * work_hours - actual_pay) / times_late = 5 :=
by
  sorry

end pay_docked_per_lateness_l719_719900


namespace emails_in_inbox_l719_719632

theorem emails_in_inbox :
  let total_emails := 400
  let trash_emails := total_emails / 2
  let work_emails := 0.4 * (total_emails - trash_emails)
  total_emails - trash_emails - work_emails = 120 :=
by
  sorry

end emails_in_inbox_l719_719632


namespace smallest_w_value_l719_719910

theorem smallest_w_value (x y z w : ℝ) 
    (hx : -2 ≤ x ∧ x ≤ 5) 
    (hy : -3 ≤ y ∧ y ≤ 7) 
    (hz : 4 ≤ z ∧ z ≤ 8) 
    (hw : w = x * y - z) : 
    w ≥ -23 :=
sorry

end smallest_w_value_l719_719910


namespace greater_number_is_64_l719_719307

-- Proof statement: The greater number (y) is 64 given the conditions
theorem greater_number_is_64 (x y : ℕ) 
    (h1 : y = 2 * x) 
    (h2 : x + y = 96) : 
    y = 64 := 
sorry

end greater_number_is_64_l719_719307


namespace quad_inequality_l719_719050

-- Define the quadrilateral properties for Lean
theorem quad_inequality 
  (AB CD AD BC : ℝ) (S : ℝ)
  (inscribed : AD + BC = AB + CD) 
  (parallel : true) -- AD ∥ BC condition as true (in comprehensive proofs this would be more specific)
  (area: S = ((AD + BC) / 2) * (abs(BC - AD)))  -- approximate height as |BC - AD| or similar:
  : AB + CD ≥ 2 * sqrt S := 
by
  sorry

end quad_inequality_l719_719050


namespace part_a_part_b_l719_719362

theorem part_a (x y : ℝ) (h : y - x = 50) :
  let BAE := x in
  let CAF := y in
  let EAF := y - x in
  EAF = 50 := 
by
  sorry

theorem part_b (CAB EAB x y : ℝ) 
  (h1 : CAB + EAB = 120) 
  (h2 : CAB - EAB = 80) 
  (h3 : 2 * x = CAB) 
  (h4 : y = EAB) : 
  let DAE := x - y in 
  DAE = 30 := 
by
  sorry

end part_a_part_b_l719_719362


namespace min_value_of_k_l719_719952

open Finset

variable {α : Type*} [DecidableEq α]

def symmetric_difference (A B : Finset α) := (A \ B) ∪ (B \ A)

theorem min_value_of_k (S T : Finset α) (hS : S.nonempty) (hT : T.nonempty) (h : (symmetric_difference S T).card = 1) : 
  (S.card + T.card) = 3 :=
sorry

end min_value_of_k_l719_719952


namespace focus_of_parabola_l719_719459

noncomputable def parabola_focus (a h k : ℝ) : ℝ × ℝ :=
  (h, k + 1 / (4 * a))

theorem focus_of_parabola :
  parabola_focus 9 (-1/3) (-3) = (-1/3, -107/36) := 
  sorry

end focus_of_parabola_l719_719459


namespace sequence_period_2016_l719_719202

theorem sequence_period_2016 : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 1 / (1 - a n)) → 
  a 1 = 1 / 2 → 
  a 2016 = -1 :=
by
  sorry

end sequence_period_2016_l719_719202


namespace find_speeds_l719_719140

-- Definitions of the given conditions
def distance : ℝ := 70
def meet_time : ℝ := 7 / 5  -- 1 hour 24 minutes converted into hours
def overtake_time : ℝ := 49 / 12  -- 4 hours 5 minutes
def bus_stop_time : ℝ := 1 / 3  -- 20 minutes converted into hours

-- Variables for the speeds of the bus and cyclist
variables (v1 v2 : ℝ)

-- Conditions derived from the problem
def condition1 : Prop := meet_time * v1 + meet_time * v2 = distance
def condition2 : Prop := 
  let t_bus := overtake_time - bus_stop_time in
  t_bus * v1 - overtake_time * v2 = distance

-- Final statement to be proved
theorem find_speeds (h1 : condition1) (h2 : condition2) : v1 = 35 ∧ v2 = 15 :=
  sorry

end find_speeds_l719_719140


namespace part1_part2_l719_719888

def f (x : ℝ) := (x^2 - 2*x + 3) / (x + 1)

theorem part1 (x : ℝ) : f x > 1 ↔ (-1 < x ∧ x < 1) ∨ (2 < x) := sorry

theorem part2 (hx : 1 < x ∧ x < 3) : 2 * real.sqrt 6 - 4 ≤ f x ∧ f x < 3 / 2 := sorry

end part1_part2_l719_719888


namespace symmetric_point_origin_l719_719931

theorem symmetric_point_origin (m : ℤ) : 
  (symmetry_condition : (3, m - 2) = (-(-3), -5)) → m = -3 :=
by
  sorry

end symmetric_point_origin_l719_719931


namespace term_with_base_25_l719_719191

variables (x y : ℝ)

-- Given conditions
axiom condition1 : 5^(x + 1) * 4^(y - 1) = 25^x * 64^y
axiom condition2 : x + y = 0.5

-- The main theorem we need to prove:
theorem term_with_base_25 : 5^(2 * x) = 25^x :=
 by sorry

end term_with_base_25_l719_719191


namespace unique_positive_b_for_one_solution_l719_719839

theorem unique_positive_b_for_one_solution
  (a : ℝ) (c : ℝ) :
  a = 3 →
  (∃! (b : ℝ), b > 0 ∧ (3 * (b + (1 / b)))^2 - 4 * c = 0 ) →
  c = 9 :=
by
  intros ha h
  -- Proceed to show that c must be 9
  sorry

end unique_positive_b_for_one_solution_l719_719839


namespace number_of_solutions_l719_719604

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then -x + 3 else 2 * x - 5

theorem number_of_solutions : 
  set.count { x : ℝ | f (f x) = 4 } = 3 := by
  sorry

end number_of_solutions_l719_719604


namespace fraction_given_to_emma_is_7_over_36_l719_719071

-- Define initial quantities
def stickers_noah : ℕ := sorry
def stickers_emma : ℕ := 3 * stickers_noah
def stickers_liam : ℕ := 12 * stickers_noah

-- Define required number of stickers for equal distribution
def total_stickers := stickers_noah + stickers_emma + stickers_liam
def equal_stickers := total_stickers / 3

-- Define the number of stickers to be given to Emma and the fraction of Liam's stickers he should give to Emma
def stickers_given_to_emma := equal_stickers - stickers_emma
def fraction_liams_stickers_given_to_emma := stickers_given_to_emma / stickers_liam

-- Theorem statement
theorem fraction_given_to_emma_is_7_over_36 :
  fraction_liams_stickers_given_to_emma = 7 / 36 :=
sorry

end fraction_given_to_emma_is_7_over_36_l719_719071


namespace inequality_proof_l719_719270

variable {a b c A B C : ℝ}

-- Lengths of the sides of a triangle and their opposite angles
axiom sides_of_triangle : a > 0 ∧ b > 0 ∧ c > 0
axiom angles_of_triangle : A > 0 ∧ B > 0 ∧ C > 0

theorem inequality_proof 
  (h : sides_of_triangle ∧ angles_of_triangle) 
  : A * a + B * b + C * c ≥ (1/2) * (A * b + B * a + A * c + C * a + B * c + C * b) :=
sorry

end inequality_proof_l719_719270


namespace sum_of_real_roots_l719_719272

def f (x : ℝ) : ℝ :=
if x > 0 then x - (3 / x)
else x^2 - (1 / 4)

theorem sum_of_real_roots :
  ∑ r in {x | f x = 2}.toFinset, r = 3 / 2 :=
by
  -- The proof is omitted
  sorry

end sum_of_real_roots_l719_719272


namespace arithmetic_sequence_sum_l719_719497

theorem arithmetic_sequence_sum (a d : ℚ) (a1 : a = 1 / 2) 
(S : ℕ → ℚ) (Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) 
(S2_eq_a3 : S 2 = a + 2 * d) :
  ∀ n, S n = (1 / 4 : ℚ) * n^2 + (1 / 4 : ℚ) * n :=
by
  intros n
  sorry

end arithmetic_sequence_sum_l719_719497


namespace evaluate_f_l719_719859

def f (x : ℝ) : ℝ :=
  if x < 2 then x^2 - 4 * x + 2 else 2 * x - 3
  
theorem evaluate_f : f (1 / f 2) = -1 :=
by sorry

end evaluate_f_l719_719859


namespace cube_intersection_volume_l719_719584

theorem cube_intersection_volume :
  let prism_vol := sqrt 2 - 1
  let pyramid_vol := 1 / 6
  let total_volume := 2 * pyramid_vol + prism_vol
  total_volume = sqrt 2 - 2 / 3 :=
by
  sorry

end cube_intersection_volume_l719_719584


namespace surface_area_of_solid_l719_719198

noncomputable def solid_surface_area (r : ℝ) (h : ℝ) : ℝ :=
  2 * Real.pi * r * h

theorem surface_area_of_solid : solid_surface_area 1 3 = 6 * Real.pi := by
  sorry

end surface_area_of_solid_l719_719198


namespace divisor_1000_l719_719417

theorem divisor_1000 {a : Fin 10 → ℤ} : ∃ b : Fin 10 → ℤ, (∀ i, b i ∈ {-1, 0, 1}) ∧ (¬ ∀ i, b i = 0) ∧ (∑ i, b i * a i) % 1000 = 0 :=
by
  sorry

end divisor_1000_l719_719417


namespace number_of_sequences_l719_719664

theorem number_of_sequences : 
  ∃ n : ℕ, n = 491 ∧ 
  ∀ a : ℕ → ℤ, 
  (a 1 = 1 ∧ a 9 = 1 ∧ 
   (∀ i : ℕ, 1 ≤ i ∧ i ≤ 8 → 
     ∃ k : ℚ, k ∈ {2, 1, -1/2} ∧ a (i+1) = a i * k)) 
  → ∃ seq_count : ℕ, seq_count = n ∧ 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 8 → a (i+1) = a i * id)
  → seq_count = 491 := 
sorry

end number_of_sequences_l719_719664


namespace f_25_over_11_neg_l719_719944

variable (f : ℚ → ℚ)
axiom f_mul : ∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b
axiom f_prime : ∀ p : ℕ, Prime p → f p = p

theorem f_25_over_11_neg : f (25 / 11) < 0 :=
by
  -- You can prove the necessary steps during interactive proof:
  -- Using primes 25 = 5^2 and 11 itself,
  -- f (25/11) = f 25 - f 11. Thus, f (25) = 2f(5) = 2 * 5 = 10 and f(11) = 11
  -- Therefore, f (25/11) = 10 - 11 = -1 < 0.
  sorry

end f_25_over_11_neg_l719_719944


namespace rem_value_l719_719412

def rem (x y : ℝ) : ℝ := x - y * Real.floor (x / y)

theorem rem_value :
  rem (5 / 7) (-3 / 4) = -1 / 28 :=
by
  sorry

end rem_value_l719_719412


namespace infinite_non_prime_l719_719031

theorem infinite_non_prime: ∃ᶠ k in at_top, ∀ n : ℕ, ¬ prime (n^4 + 4 * k^4) := 
by
  sorry

end infinite_non_prime_l719_719031


namespace first_valve_fill_time_l719_719696

theorem first_valve_fill_time (V1 V2: ℕ) (capacity: ℕ) (t_combined t1: ℕ) 
  (h1: t_combined = 48)
  (h2: V2 = V1 + 50)
  (h3: capacity = 12000)
  (h4: V1 + V2 = capacity / t_combined)
  : t1 = 2 * 60 :=
by
  -- The proof would come here
  sorry

end first_valve_fill_time_l719_719696


namespace total_calculators_sold_l719_719720

theorem total_calculators_sold 
    (x y : ℕ)
    (h₁ : y = 35)
    (h₂ : 15 * x + 67 * y = 3875) :
    x + y = 137 :=
by 
  -- We will insert the proof here
  sorry

end total_calculators_sold_l719_719720


namespace centers_form_equilateral_triangle_l719_719074

noncomputable theory

open Real EuclideanGeometry 

variables {A B C P Q R : EuclideanGeometry.Point}
variables (b c : ℝ)

-- Definition of Points
def A : EuclideanGeometry.Point := (0, 0)
def B : EuclideanGeometry.Point := (b, 0)
def C : EuclideanGeometry.Point := (c, 0)

-- Midpoints and Heights Calculation
def midpoint_AB : EuclideanGeometry.Point := ((b / 2), 0)
def midpoint_BC : EuclideanGeometry.Point := ((b + c) / 2, 0)
def midpoint_CA : EuclideanGeometry.Point := (c / 2, 0)

def height_AB : ℝ := (b * sqrt 3) / 2
def height_BC : ℝ := (abs (c - b) * sqrt 3) / 2
def height_CA : ℝ := (c * sqrt 3) / 2

-- Centers of the equilateral triangles
def P : EuclideanGeometry.Point := (midpoint_AB.1, height_AB)
def Q : EuclideanGeometry.Point := (midpoint_BC.1, height_BC)
def R : EuclideanGeometry.Point := (midpoint_CA.1, -height_CA)

-- Centroid of triangle PQR
def centroid_PQR : EuclideanGeometry.Point := 
  ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Main theorem
theorem centers_form_equilateral_triangle :
  (P, Q, R ∈ EuclideanGeometry.equilateral A B C) →
  centroid_PQR ∈ segment ℝ A C := sorry

end centers_form_equilateral_triangle_l719_719074


namespace sequence_arithmetic_sum_of_reciprocals_l719_719185

theorem sequence_arithmetic (a : ℕ → ℕ) (n : ℕ) (h1 : a 2 = 6) (h2 : a 3 - 3 * a 1 = 6) (h_arith : ∀ i, (a (i+1) / (i+1) - a i / i) = (a (i+2) / (i+2) - a (i+1) / (i+1))) :
  (a n = n * (n + 1)) :=
sorry

theorem sum_of_reciprocals (a : ℕ → ℕ) (n : ℕ) (S : ℕ → ℚ) (h1 : a 2 = 6) (h2 : a 3 - 3 * a 1 = 6) (h_arith : ∀ i, (a (i+1) / (i+1) - a i / i) = (a (i+2) / (i+2) - a (i+1) / (i+1))) (h_a : ∀ k, a k = k * (k + 1)) :
  (S n = ∑ i in range (n+1), 1 / a i = (n)/(n + 1)) :=
sorry

end sequence_arithmetic_sum_of_reciprocals_l719_719185


namespace number_of_triangles_fitting_in_square_l719_719529

-- Define the conditions for the right triangle and the square
def right_triangle_height := 2
def right_triangle_width := 2
def square_side := 2

-- Define the areas
def area_triangle := (1 / 2) * right_triangle_height * right_triangle_width
def area_square := square_side * square_side

-- Define the proof statement to show the number of right triangles fitting in the square is 2
theorem number_of_triangles_fitting_in_square : (area_square / area_triangle) = 2 := by
  sorry

end number_of_triangles_fitting_in_square_l719_719529


namespace molecular_weight_is_approx_294_18_l719_719685

def molecular_weight (p_avg cr_avg o_avg : ℝ) : ℝ := 
  2 * p_avg + 2 * cr_avg + 7 * o_avg

theorem molecular_weight_is_approx_294_18 : 
  let k_avg := (0.932581 * 38.9637 + 0.067302 * 40.9618)
  let cr_avg := (0.83789 * 51.9405 + 0.09501 * 52.9407)
  let o_avg := 15.999
  molecular_weight k_avg cr_avg o_avg ≈ 294.18 :=
by {
  let k_avg := (0.932581 * 38.9637 + 0.067302 * 40.9618)
  let cr_avg := (0.83789 * 51.9405 + 0.09501 * 52.9407)
  let o_avg := 15.999
  show molecular_weight k_avg cr_avg o_avg ≈ 294.18,
  -- we skip the actual proof details
  sorry
}

end molecular_weight_is_approx_294_18_l719_719685


namespace barbara_weekly_allowance_l719_719778

theorem barbara_weekly_allowance (W C S : ℕ) (H : W = 100) (A : S = 20) (N : C = 16) :
  (W - S) / C = 5 :=
by
  -- definitions to match conditions
  have W_def : W = 100 := H
  have S_def : S = 20 := A
  have C_def : C = 16 := N
  sorry

end barbara_weekly_allowance_l719_719778


namespace integer_side_lengths_triangle_l719_719095

theorem integer_side_lengths_triangle (a b : ℕ) (ha : a = 8) (hb : b = 5) : 
  { x : ℕ | 3 < x ∧ x < 13 }.finite.card = 9 := by
  sorry

end integer_side_lengths_triangle_l719_719095


namespace sport_formulation_water_l719_719704

theorem sport_formulation_water
  (f : ℝ) (c : ℝ) (w : ℝ) 
  (f_s : ℝ) (c_s : ℝ) (w_s : ℝ)
  (standard_ratio : f / c = 1 / 12 ∧ f / w = 1 / 30)
  (sport_ratio_corn_syrup : f_s / c_s = 3 * (f / c))
  (sport_ratio_water : f_s / w_s = (1 / 2) * (f / w))
  (corn_syrup_amount : c_s = 3) :
  w_s = 45 :=
by
  sorry

end sport_formulation_water_l719_719704


namespace selection_methods_count_l719_719972

-- Definitions based on the conditions from the problem
def students := {A, B, C, D, E, F}       -- Define the set of students
def rep_count := 4                       -- Number of representatives to be selected
def forbidden_positions_B := {1}          -- Student B cannot run the 1st leg

-- Define the constraints for student A
def possible_positions_A := {1, 4}       -- Student A can run only 1st or 4th leg

-- Theorem statement: number of selection methods is 108 given the constraints
theorem selection_methods_count : 
    let possible_teams : finset (finset students) := 
        { s | s.card = rep_count ∧ (A ∈ s ∧ (1 ∈ possible_positions_A ∨ 4 ∈ possible_positions_A)) 
        ∧ (B ∈ s → 1 ∉ forbidden_positions_B) 
        ∧ students.card = 6 
        ∧ (∀ t ∈ s, t ∈ students)} in
    possible_teams.card = 108 :=
sorry

end selection_methods_count_l719_719972


namespace find_sum_of_inverses_l719_719590

def f (x : ℝ) : ℝ :=
if x >= 0 then x^3 else -x^3

theorem find_sum_of_inverses :
  (f 2 = 8) ∧ (f (-3) = -27) ∧ (f (-3) + (f 2) = -1) :=
by
  sorry

end find_sum_of_inverses_l719_719590


namespace count_valid_integers_correct_l719_719528

def is_odd (n : ℕ) : Prop := n % 2 = 1
def has_three_different_digits (n : ℕ) : Prop := 
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  (digits.nodup) && (digits.length = 4)

noncomputable def count_valid_integers : ℕ :=
  (finset.range 1000).sum (λ x, 
    let n := x + 2000 in 
    if n < 3000 ∧ is_odd (n % 10) ∧ has_three_different_digits n then 1 else 0)

theorem count_valid_integers_correct : count_valid_integers = 280 := sorry

end count_valid_integers_correct_l719_719528


namespace parallel_lines_perpendicular_lines_l719_719522

-- Define the lines
def l₁ (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- The first proof statement: lines l₁ and l₂ are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → (a * (a - 1) - 2 = 0)) → (a = 2 ∨ a = -1) :=
by
  sorry

-- The second proof statement: lines l₁ and l₂ are perpendicular
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → ((a - 1) * 1 + 2 * a = 0)) → (a = -1 / 3) :=
by
  sorry

end parallel_lines_perpendicular_lines_l719_719522


namespace line_equation_min_intercepts_l719_719380

theorem line_equation_min_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : 1 / a + 4 / b = 1) : 2 * 1 + 4 - 6 = 0 ↔ (a = 3 ∧ b = 6) :=
by
  sorry

end line_equation_min_intercepts_l719_719380


namespace horse_grazing_area_l719_719019

def rope_length : ℝ := 17
def field_length : ℝ := 46
def field_width : ℝ := 20
def grazing_area (r : ℝ) : ℝ := (π * r^2) / 4

theorem horse_grazing_area :
  grazing_area rope_length = 226.1945 := sorry

end horse_grazing_area_l719_719019


namespace statement_III_must_be_true_l719_719542

universe u

variable (Dragons MagicalCreatures SpookySpirits : Type u)
variable (isDragon : Dragons → Prop)
variable (isMagicalCreature : MagicalCreatures → Prop)
variable (isSpookySpirit : SpookySpirits → Prop)

axiom all_dragons_magical :
  ∀ d : Dragons, isDragon d → isMagicalCreature d

axiom no_spooky_spirits_dragons :
  ∀ s : SpookySpirits, isSpookySpirit s → ¬ isDragon s

theorem statement_III_must_be_true :
  (∃ m : MagicalCreatures, isMagicalCreature m ∧ ¬ isSpookySpirit m) := by
  sorry

end statement_III_must_be_true_l719_719542


namespace remainder_when_6n_divided_by_4_l719_719009

theorem remainder_when_6n_divided_by_4 (n : ℤ) (h : n % 4 = 1) : 6 * n % 4 = 2 := by
  sorry

end remainder_when_6n_divided_by_4_l719_719009


namespace dany_farm_bushels_l719_719419

theorem dany_farm_bushels :
  let cows := 5
  let cows_bushels_per_day := 3
  let sheep := 4
  let sheep_bushels_per_day := 2
  let chickens := 8
  let chickens_bushels_per_day := 1
  let pigs := 6
  let pigs_bushels_per_day := 4
  let horses := 2
  let horses_bushels_per_day := 5
  cows * cows_bushels_per_day +
  sheep * sheep_bushels_per_day +
  chickens * chickens_bushels_per_day +
  pigs * pigs_bushels_per_day +
  horses * horses_bushels_per_day = 65 := by
  sorry

end dany_farm_bushels_l719_719419


namespace valid_program_count_l719_719759

theorem valid_program_count : 
  let courses := ["English", "Algebra", "Geometry", "History", "Art", "Latin", "Calculus"]
  let math_courses := ["Algebra", "Geometry", "Calculus"]
  in (∃ program : Finset String, 
        "English" ∈ program ∧ 
        (math_courses.toFinset.filter (λ x, x ∈ program)).card ≥ 2 ∧ 
        program.card = 4
      ) ∧
      (Finset.powerset courses.toFinset).filter 
        (λ program, 
          "English" ∈ program ∧ 
          (math_courses.toFinset.filter (λ x, x ∈ program)).card ≥ 2 ∧ 
          program.card = 4
        ).toFinset.card = 10 :=
by sorry

end valid_program_count_l719_719759


namespace line_through_origin_intersecting_AB_and_CD_l719_719172

noncomputable def A : Vector3 ℝ := ⟨1, 0, 1⟩
noncomputable def B : Vector3 ℝ := ⟨-2, 2, 1⟩
noncomputable def C : Vector3 ℝ := ⟨2, 0, 3⟩
noncomputable def D : Vector3 ℝ := ⟨0, 4, -2⟩

theorem line_through_origin_intersecting_AB_and_CD :
  ∃ t : ℝ, (8 * t, 2 * t, 11 * t) ∈ LineThrough(A, B) ∧ (8 * t, 2 * t, 11 * t) ∈ LineThrough(C, D) :=
sorry

end line_through_origin_intersecting_AB_and_CD_l719_719172


namespace constant_term_expansion_l719_719241

theorem constant_term_expansion (x : ℝ) : 
  (x - 2 / x) ^ 6 =
  (sum (λ k, (nat.choose 6 k) * (-2) ^ k * x ^ (6 - 2 * k))) →
  real.constant_term (x - 2 / x) ^ 6 = -160 := 
sorry

end constant_term_expansion_l719_719241


namespace circumference_of_wheel_l719_719763

def distance : ℝ := 1056
def num_revolutions : ℝ := 3.002729754322111
def circumference : ℝ := distance / num_revolutions

theorem circumference_of_wheel :
  circumference ≈ 351.855 := sorry

end circumference_of_wheel_l719_719763


namespace triangle_probability_is_one_fourth_l719_719102

-- Define the lengths of the sticks
def lengths : List ℕ := [1, 3, 4, 6, 8, 10, 12, 15]

-- Predicate to check if three lengths can form a triangle
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Number of valid triangles that can be formed
def num_valid_triangles : ℕ :=
  (lengths.combinations 3).count (λ l, is_triangle l.head! l.get! 1 l.get! 2)

-- Total number of combinations of 3 sticks
def num_combinations : ℕ := (lengths.length.choose 3)

-- Probability of forming a triangle
def triangle_probability : ℚ :=
  num_valid_triangles / num_combinations

-- The theorem to prove
theorem triangle_probability_is_one_fourth : triangle_probability = 1 / 4 := by
  sorry

end triangle_probability_is_one_fourth_l719_719102


namespace correct_mean_is_correct_mean_l719_719358

variable (mean_incorrect : ℝ) (number_of_values : ℕ)
variable (incorrect_value correct_value : ℝ)

theorem correct_mean_is_correct_mean:
  let S_incorrect := mean_incorrect * number_of_values in
  let Difference := correct_value - incorrect_value in
  let S_correct := S_incorrect + Difference in
  let mean_correct := S_correct / number_of_values in
  number_of_values = 30 →
  mean_incorrect = 180 →
  incorrect_value = 135 →
  correct_value = 155 →
  mean_correct = 180.67 :=
by
  intros
  sorry

end correct_mean_is_correct_mean_l719_719358


namespace triangular_pyramid_proof_l719_719162

noncomputable def triangular_pyramid_sum_of_angles (A B C D : Type*) 
  (a b : ℝ) (CD AD DB : ℝ) 
  (h1 : CD = AD + DB)
  (h2 : CD = a + b)
  (α β γ : ℝ) : Prop :=
α + β + γ = 90

theorem triangular_pyramid_proof (A B C D : Type*) 
  (a b : ℝ) (CD AD DB : ℝ)
  (h1 : CD = AD + DB)
  (h2 : CD = a + b)
  (α β γ : ℝ)
  (h_angles : α + β + γ = 90): triangular_pyramid_sum_of_angles A B C D a b CD AD DB h1 h2 α β γ :=
by
  rw [h_angles]
  exact 90

#check triangular_pyramid_sum_of_angles
#check triangular_pyramid_proof

end triangular_pyramid_proof_l719_719162


namespace sum_k_squared_div_3_k_l719_719452

theorem sum_k_squared_div_3_k : ∑ k in (Finset.range n).map (λ x, x+1), (k^2 / (3^k : ℝ)) = 4 :=
by
  sorry

end sum_k_squared_div_3_k_l719_719452


namespace infinite_series_converges_to_3_l719_719787

noncomputable def sum_of_series := ∑' k in (Finset.range ∞).filter (λ k, k > 0), 
  (8 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))

theorem infinite_series_converges_to_3 : sum_of_series = 3 := 
  sorry

end infinite_series_converges_to_3_l719_719787


namespace intersection_points_define_parallelogram_l719_719043

def cyclic_quadrilateral (K L M N : Point) : Prop :=
  ∃ O : Point, is_circumscribed K L M N O

def intersections_form_parallelogram (K L M N K' L' M' N' P Q R S: Point) : Prop :=
  is_intersection_point P K K' L L' ∧
  is_intersection_point Q L L' M M' ∧
  is_intersection_point R M M' N N' ∧
  is_intersection_point S N N' K K' ∧
  is_parallelogram P Q R S

theorem intersection_points_define_parallelogram
  (K L M N : Point)
  (angle phi : ℝ) 
  (h1 : cyclic_quadrilateral K L M N)
  (h2 : 0 < phi ∧ phi < 180) :
  ∃ K' L' M' N' P Q R S : Point, intersections_form_parallelogram K L M N K' L' M' N' P Q R S :=
sorry

end intersection_points_define_parallelogram_l719_719043


namespace infinite_series_sum_eq_two_l719_719794

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l719_719794


namespace sin_cubed_cos_cubed_tan_plus_cot_l719_719496

def trig_roots_equiv (a : ℝ) (θ : ℝ) : Prop :=
  ∃ θ : ℝ, (sin θ + cos θ = a) ∧ ((sin θ) * (cos θ) = a)

theorem sin_cubed_cos_cubed (θ a : ℝ) (h : trig_roots_equiv a θ) :
  sin θ + cos θ = 1 - sqrt 2 ∧ (sin θ) * (cos θ) = 1 - sqrt 2 →
  (sin θ) ^ 3 + (cos θ) ^ 3 = sqrt 2 - 2 := sorry

theorem tan_plus_cot (θ a : ℝ) (h : trig_roots_equiv a θ) :
  sin θ + cos θ = 1 - sqrt 2 ∧ (sin θ) * (cos θ) = 1 - sqrt 2 →
  tan θ + (1 / tan θ) = -1 - sqrt 2 := sorry

end sin_cubed_cos_cubed_tan_plus_cot_l719_719496


namespace angle_range_condition_l719_719876

variable (a b : ℝ)
variable (θ : ℝ)

def f (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * (2 * b) * x^2 + (2 * b^2 * Real.cos θ) * x

theorem angle_range_condition (hab : |a| = 2 * |b|) (hb_ne_zero : |b| ≠ 0) 
  (extremum_exists : ∃ x : ℝ, differentiation f x = 0) : 
  θ ∈ (Set.Ioc (π / 3) π) := by
  sorry

end angle_range_condition_l719_719876


namespace no_integer_solutions_l719_719617

theorem no_integer_solutions (x y : ℤ) : 2 * x^2 - 5 * y^2 ≠ 7 :=
  sorry

end no_integer_solutions_l719_719617


namespace tom_makes_money_l719_719677

def flour_pounds : ℕ := 500
def flour_cost_per_bag : ℕ := 20
def flour_bag_weight : ℕ := 50

def salt_pounds : ℕ := 10
def salt_cost_per_pound : ℝ := 0.2

def sugar_pounds : ℕ := 20
def sugar_cost_per_pound : ℝ := 0.5

def butter_pounds : ℕ := 50
def butter_cost_per_pound : ℕ := 2
def butter_discount : ℝ := 0.1

def chefA_cost : ℕ := 200
def chefB_cost : ℕ := 250
def chefC_cost : ℕ := 300
def chef_tax : ℝ := 0.05

def promotion_cost : ℕ := 1000
def ticket_price : ℕ := 20
def tickets_sold : ℕ := 1200

def revenue := (tickets_sold * ticket_price : ℝ)

def total_ingredient_cost := 
  ((flour_pounds / flour_bag_weight * flour_cost_per_bag) +
  (salt_pounds * salt_cost_per_pound) +
  (sugar_pounds * sugar_cost_per_pound) +
  ((butter_pounds * butter_cost_per_pound) * (1 - butter_discount)) : ℝ)

def total_chef_cost := 
  ((chefA_cost + chefB_cost + chefC_cost) * (1 + chef_tax) : ℝ)

def total_cost := total_ingredient_cost + total_chef_cost + promotion_cost

def tom_profit := revenue - total_cost

theorem tom_makes_money : tom_profit = 21910.50 := 
by
  sorry

end tom_makes_money_l719_719677


namespace count_of_tri_connected_collections_l719_719360

def tri_connected_collection_exists (n : ℕ) : Prop :=
  ∃ (squares : Finset (set (ℝ × ℝ))), 
    squares.card = n ∧
    (∀ s ∈ squares, ∃ exactly_three (touches s)) ∧
    (∀ s₁ s₂ ∈ squares, congruent s₁ s₂ ∧ common_vertex_if_touch s₁ s₂)

def touches (s1 s2 : set (ℝ × ℝ)) : Prop := sorry -- Define the exact relation for touching

def congruent (s1 s2 : set (ℝ × ℝ)) : Prop := sorry -- Define the exact relation for congruence 

def common_vertex_if_touch (s1 s2 : set (ℝ × ℝ)) : Prop :=
  (∃ (P : ℝ × ℝ), P ∈ s1 ∧ P ∈ s2) → 
  (∀ (P : ℝ × ℝ), P ∈ s1 ∧ P ∈ s2 → is_vertex P s1 ∧ is_vertex P s2)

def is_vertex (P : ℝ × ℝ) (s : set (ℝ × ℝ)) : Prop := sorry -- Determine if P is a vertex of square s

theorem count_of_tri_connected_collections :
  (finset.range 3019).filter (λ n, 
    2018 ≤ n ∧ tri_connected_collection_exists n ∧ n % 2 = 0 
  ).card = 501 := by
    sorry

end count_of_tri_connected_collections_l719_719360


namespace combined_share_is_50000_l719_719310

def profit : ℝ := 80000

def majority_owner_share : ℝ := 0.25 * profit

def remaining_profit : ℝ := profit - majority_owner_share

def partner_share : ℝ := 0.25 * remaining_profit

def combined_share_majority_two_owners : ℝ := majority_owner_share + 2 * partner_share

theorem combined_share_is_50000 :
  combined_share_majority_two_owners = 50000 := 
by 
  sorry

end combined_share_is_50000_l719_719310


namespace probability_four_friends_same_group_l719_719999

variable (Ω : Type) [Fintype Ω] (students : Finset Ω) (groups : Finset (Finset Ω))
variable [ProbTheory groups]
noncomputable def four_friends_probability (Dave Eve Frank Grace : Ω) (h : ∀ group ∈ groups, size group = 200):

  def probability_same_group : ℝ :=
  (1 / 4) * (1 / 4) * (1 / 4)

theorem probability_four_friends_same_group (Dave Eve Frank Grace : Ω) (h : ∀ group ∈ groups, size group = 200) :
  four_friends_probability Dave Eve Frank Grace h = 1 / 64
:= sorry

end probability_four_friends_same_group_l719_719999


namespace coordinates_of_P_l719_719237

noncomputable def point_coordinates (x y : ℝ) : Prop :=
  -3 = x ∧ 4 = y

theorem coordinates_of_P (x y : ℤ) 
  (h1 : x < 0) 
  (h2 : y > 0)
  (h3 : (y: ℤ) = 4)
  (h4 : (x: ℤ) = -3) : point_coordinates x y :=
by
  dsimp [point_coordinates]
  split
  · exact h4
  · exact h3

end coordinates_of_P_l719_719237


namespace remainder_when_divided_by_x_minus_2_l719_719001

def f (x : ℝ) : ℝ := x^5 - 4 * x^4 + 6 * x^3 + 25 * x^2 - 20 * x - 24

theorem remainder_when_divided_by_x_minus_2 : f 2 = 52 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l719_719001


namespace infinite_sum_problem_l719_719809

theorem infinite_sum_problem : 
  (∑ k in (set.Ici 1), (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 1 :=
by
  sorry

end infinite_sum_problem_l719_719809


namespace emails_left_are_correct_l719_719630

-- Define the initial conditions for the problem
def initial_emails : ℕ := 400
def trash_emails : ℕ := initial_emails / 2
def remaining_after_trash : ℕ := initial_emails - trash_emails
def work_emails : ℕ := (remaining_after_trash * 40) / 100

-- Define the final number of emails left in the inbox
def emails_left_in_inbox : ℕ := remaining_after_trash - work_emails

-- The proof goal
theorem emails_left_are_correct : emails_left_in_inbox = 120 :=
by 
    -- The computations are correct based on the conditions provided
    have h_trash : trash_emails = 200 := by rfl
    have h_remaining : remaining_after_trash = 200 := by rw [← h_trash, Nat.sub_eq_iff_eq_add (Nat.le_refl 200)]
    have h_work : work_emails = 80 := by 
        rw [← h_remaining, Nat.mul_div_cancel (Nat.le_refl 8000) (Nat.lt_of_sub_one_eq_zero (by refl), 4000)]
    show emails_left_in_inbox = 120 := by
        rw [emails_left_in_inbox, h_remaining, h_work, Nat.sub_eq_iff_eq_add (Nat.le_refl 80)]
        exact rfl

end emails_left_are_correct_l719_719630


namespace probability_ace_then_diamond_l719_719338

/-- The probability of drawing an Ace first and a diamond second from a standard
 deck of 52 cards when drawn sequentially without replacement is 1/30. -/
theorem probability_ace_then_diamond :
  let p := (4 / 52) * (13 / 51) + (1 / 52) * (12 / 51) in p = 1 / 30 :=
by
  -- Conditions: 
  have h1 : p = (4 / 52) * (13 / 51) + (1 / 52) * (12 / 51), by sorry
  show p = 1 / 30, by sorry

end probability_ace_then_diamond_l719_719338


namespace largest_c_l719_719461

theorem largest_c (c : ℝ) : (∃ x : ℝ, x^2 + 4 * x + c = -3) → c ≤ 1 :=
by
  sorry

end largest_c_l719_719461


namespace three_digit_primes_ending_in_one_and_gt_200_eq_23_l719_719530

theorem three_digit_primes_ending_in_one_and_gt_200_eq_23 :
  {p : ℕ | nat.prime p ∧ p % 10 = 1 ∧ 200 < p ∧ p < 1000}.card = 23 :=
by
  sorry

end three_digit_primes_ending_in_one_and_gt_200_eq_23_l719_719530


namespace age_of_50th_student_l719_719639

theorem age_of_50th_student (avg_50_students : ℝ) (total_students : ℕ)
                           (avg_15_students : ℝ) (group_1_count : ℕ)
                           (avg_15_students_2 : ℝ) (group_2_count : ℕ)
                           (avg_10_students : ℝ) (group_3_count : ℕ)
                           (avg_9_students : ℝ) (group_4_count : ℕ) :
                           avg_50_students = 20 → total_students = 50 →
                           avg_15_students = 18 → group_1_count = 15 →
                           avg_15_students_2 = 22 → group_2_count = 15 →
                           avg_10_students = 25 → group_3_count = 10 →
                           avg_9_students = 24 → group_4_count = 9 →
                           ∃ (age_50th_student : ℝ), age_50th_student = 66 := by
                           sorry

end age_of_50th_student_l719_719639


namespace coopers_daily_pie_count_l719_719819

-- Definitions of conditions
def total_pies_made_per_day (x : ℕ) : ℕ := x
def days := 12
def pies_eaten_by_ashley := 50
def remaining_pies := 34

-- Lean 4 statement of the problem to prove
theorem coopers_daily_pie_count (x : ℕ) : 
  12 * total_pies_made_per_day x - pies_eaten_by_ashley = remaining_pies → 
  x = 7 := 
by
  intro h
  -- Solution steps (not included in the theorem)
  -- Given proof follows from the Lean 4 statement
  sorry

end coopers_daily_pie_count_l719_719819


namespace triangle_third_side_length_l719_719976

-- Define a triangle with sides (AB = 10), (AC = 20), and an angle condition
variables (A B C: Type) [EuclideanGeometry A B C]
variables (AB AC BC: Real)
variables (angleB angleC : Real)

-- Given conditions
def condition1 : Prop := (angleB = 3 * angleC)
def condition2 : Prop := (AB = 10) ∧ (AC = 20)

-- Goal: Prove that BC equals 10 * sqrt(3)
theorem triangle_third_side_length :
  (condition1 ∧ condition2) → (BC = 10 * Real.sqrt 3) :=
by
  sorry

end triangle_third_side_length_l719_719976


namespace find_c_l719_719297

theorem find_c (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
(h_asc : a < b) (h_asc2 : b < c)
(h_sum : a + b + c = 11)
(h_eq : 1 / a + 1 / b + 1 / c = 1) : c = 6 := 
sorry

end find_c_l719_719297


namespace range_log_function_l719_719891

theorem range_log_function {x k : ℝ} (y : ℝ) 
  (h1 : y = log 2 (x ^ 2 - 2 * k * x + k)) 
  (hrange : ∀ y : ℝ, ∃ x : ℝ, y = log 2 (x ^ 2 - 2 * k * x + k)) :
  k ≤ 0 ∨ k ≥ 1 :=
sorry

end range_log_function_l719_719891


namespace triangle_area_l719_719587

variables (a b : ℝ × ℝ)

def reflect_x (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.1, -v.2)

def parallelogram_area (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.2) - (v1.2 * v2.1)

theorem triangle_area
  (h₁ : a = (4, -1))
  (h₂ : b = (3, 4)) :
  let a' := reflect_x a in
  0.5 * (parallelogram_area a' b) = 13 / 2 :=
by
  -- Since no proof steps are required, we just use sorry to skip the proof.
  sorry

end triangle_area_l719_719587


namespace Alyssa_bought_cards_l719_719576

/-- Jason originally had 676 Pokemon cards. -/
def original_cards := 676

/-- Jason now has 452 Pokemon cards after Alyssa bought some. -/
def remaining_cards := 452

/-- The number of Pokemon cards Alyssa bought. -/
def cards_bought := original_cards - remaining_cards

theorem Alyssa_bought_cards : cards_bought = 224 := by
  rw [cards_bought, original_cards, remaining_cards]
  norm_num
  sorry

end Alyssa_bought_cards_l719_719576


namespace carbon_copies_after_folding_l719_719021

def initial_sheets : ℕ := 6
def initial_carbons (sheets : ℕ) : ℕ := sheets - 1
def final_copies (sheets : ℕ) : ℕ := sheets - 1

theorem carbon_copies_after_folding :
  (final_copies initial_sheets) =
  initial_carbons initial_sheets :=
by {
    -- sorry is a placeholder for the proof
    sorry
}

end carbon_copies_after_folding_l719_719021


namespace math_problem_l719_719516

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 + 3 * x * (f' a)
noncomputable def f' (a : ℝ) : ℝ := - (a^2) / 2

theorem math_problem (x a : ℝ) (hx : f'(-1) = -1/2) (h : f(a) = 7/6) :
  (f(x a) = (1 / 3) * x ^ 3 - (3 / 2) * x) ∧ ((3:ℝ) * x + (6:ℝ) * f(x a) - 4 = 0) := 
sorry

end math_problem_l719_719516


namespace carol_maximizes_at_0_75_l719_719765

def winning_probability (a b c : ℝ) : Prop :=
(0 ≤ a ∧ a ≤ 1) ∧ (0.25 ≤ b ∧ b ≤ 0.75) ∧ (a < c ∧ c < b ∨ b < c ∧ c < a)

theorem carol_maximizes_at_0_75 :
  ∀ (a b : ℝ), (0 ≤ a ∧ a ≤ 1) → (0.25 ≤ b ∧ b ≤ 0.75) → (∃ c : ℝ, 0 ≤ c ∧ c ≤ 1 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → winning_probability a b x ≤ winning_probability a b 0.75)) :=
sorry

end carol_maximizes_at_0_75_l719_719765


namespace remove_point_partition_smaller_diameter_l719_719616

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

def diameter (M : set Point) : ℝ :=
  Sup { distance P Q | P Q : Point, P ∈ M, Q ∈ M }

theorem remove_point_partition_smaller_diameter
  (M : set Point) (hM : set.finite M) (hM_card : M.card > 1) :
  ∃ P ∈ M, ∃ M' ⊆ M, M' = M \ {P} ∧ 
  ∃ A B, A ∈ M' ∧ B ∈ M' ∧ 
  diameter M' < diameter M ∧
  (∀ x y ∈ M', distance x y < diameter M) :=
sorry

end remove_point_partition_smaller_diameter_l719_719616


namespace monotonically_increasing_f_iff_a_gt_1_l719_719853

theorem monotonically_increasing_f_iff_a_gt_1
  (a : ℝ) (h_positive : a ≥ 0) (h_monotonic : ∀ x > 0, (λ x, a * x - a / x - 2 * Real.log x)' x ≥ 0) :
  a > 1 :=
sorry

end monotonically_increasing_f_iff_a_gt_1_l719_719853


namespace polynomial_division_remainder_l719_719002

theorem polynomial_division_remainder :
  let p := 3 * X ^ 2 - 22 * X + 64
  let d := X - 3
  (p % d).coeff 0 = 25 :=
by
  sorry

end polynomial_division_remainder_l719_719002


namespace solve_equation_l719_719292

theorem solve_equation (a : ℝ) : 
  {x : ℝ | x * (x + a)^3 * (5 - x) = 0} = {0, -a, 5} :=
sorry

end solve_equation_l719_719292


namespace total_ice_cream_l719_719396

theorem total_ice_cream (pints_friday : ℝ) (pints_saturday : ℝ) 
  (h_friday : pints_friday = 3.25)
  (h_saturday : pints_saturday = 0.25) :
  pints_friday + pints_saturday = 3.50 :=
by {
  rw [h_friday, h_saturday],
  norm_num,
}

end total_ice_cream_l719_719396


namespace infinite_series_equals_two_l719_719802

noncomputable def sum_series : ℕ → ℝ := λ k, (8^k : ℝ) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_equals_two :
  (∑' k : ℕ, if k > 0 then sum_series k else 0) = 2 :=
by 
  sorry

end infinite_series_equals_two_l719_719802


namespace problem_one_problem_two_l719_719176

theorem problem_one (α : ℝ) (h : Real.tan α = 2) : (3 * Real.sin α - 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

theorem problem_two (α : ℝ) (h : Real.tan α = 2) (h_quadrant : α > π ∧ α < 3 * π / 2) : Real.cos α = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_one_problem_two_l719_719176


namespace area_of_shaded_region_l719_719776

theorem area_of_shaded_region (s : ℝ) (h_s : s = 3) (r : ℝ) (h_r : r = s / 2) :
  let A_octagon := 2 * (1 + Real.sqrt 2) * s^2,
      A_semicircle := (π * r^2) / 2,
      A_total_semicircles := 8 * A_semicircle,
      A_shaded_region := A_octagon - A_total_semicircles
  in A_shaded_region = 54 + 54 * Real.sqrt 2 - 18 * π := 
  by
  sorry

end area_of_shaded_region_l719_719776


namespace gift_wrapping_combinations_l719_719377

theorem gift_wrapping_combinations :
    (10 * 3 * 4 * 5 = 600) :=
by
    sorry

end gift_wrapping_combinations_l719_719377


namespace largest_sphere_radius_in_prism_l719_719161

noncomputable def largestInscribedSphereRadius (m : ℝ) : ℝ :=
  (Real.sqrt 6 - Real.sqrt 2) / 4 * m

theorem largest_sphere_radius_in_prism (m : ℝ) (h : 0 < m) :
  ∃ r, r = largestInscribedSphereRadius m ∧ r < m/2 :=
sorry

end largest_sphere_radius_in_prism_l719_719161


namespace line_parallel_plane_non_intersect_l719_719545

-- Definitions: line, plane
structure Line := 
  (point1 : ℝ × ℝ × ℝ)
  (point2 : ℝ × ℝ × ℝ)

structure Plane :=
  (normal_vector : ℝ × ℝ × ℝ)
  (point_on_plane : ℝ × ℝ × ℝ)

def is_parallel_plane (l : Line) (p : Plane) : Prop := 
  let (a, b, c) := p.normal_vector in
  let (p1, q1, r1) := l.point1 in
  let (p2, q2, r2) := l.point2 in
  -- We need the direction vector of the line to be orthogonal to the normal vector of the plane
  (a * (p2 - p1) + b * (q2 - q1) + c * (r2 - r1) = 0)

def is_in_plane (l : Line) (p : Plane) : Prop :=
  -- A line is in the plane if both of its points satisfy the plane equation
  let (a, b, c) := p.normal_vector in
  let (px, py, pz) := p.point_on_plane in
  let (p1, q1, r1) := l.point1 in
  let (p2, q2, r2) := l.point2 in
  a * (p1 - px) + b * (q1 - py) + c * (r1 - pz) = 0 ∧
  a * (p2 - px) + b * (q2 - py) + c * (r2 - pz) = 0

theorem line_parallel_plane_non_intersect (a : Line) (α : Plane) (l : Line) 
  (h_parallel : is_parallel_plane a α) (h_in_plane : is_in_plane l α) :
  ¬ ∃ p, (p = a.point1 ∨ p = a.point2) ∧ (p = l.point1 ∨ p = l.point2) :=
by
  sorry

end line_parallel_plane_non_intersect_l719_719545


namespace max_negatives_in_product_l719_719218

noncomputable def integers := {i : ℤ // i > 1}

theorem max_negatives_in_product (s : Finset integers)
  (h_card : s.card = 18)
  (h_distinct : s.to_list.nodup)
  (h_prod_neg : (s.to_list.map Subtype.val).prod < 0) :
  ∃ k, k ≤ 17 ∧ (s.to_list.filter (λ x => x.1 < 0)).length = k := sorry

end max_negatives_in_product_l719_719218


namespace win_sector_area_l719_719732

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 12) (h_p : p = 1 / 3) :
  ∃ A : ℝ, A = 48 * π :=
by {
  sorry
}

end win_sector_area_l719_719732


namespace problem1_problem2_min_value_problem2_max_value_l719_719034

-- Equivalent Proof Problem for Problem (1)
theorem problem1 : ( (9/4)^(1/2) - 1 - (27/8)^(-2/3) + (3/2)^(-2) ) = 1/2 := 
by
  sorry

-- Equivalent Proof Problem for Problem (2)
def f (x : Real) := log (1 / 4) x ^ 2 - log (1 / 4) x + 5

theorem problem2_min_value : ∀ x, 2 ≤ x ∧ x ≤ 4 → (∃ x, f x = 23 / 4) :=
by
  sorry

theorem problem2_max_value : ∀ x, 2 ≤ x ∧ x ≤ 4 → (∃ x, f x = 7) :=
by
  sorry

end problem1_problem2_min_value_problem2_max_value_l719_719034


namespace first_year_students_count_number_of_students_l719_719735

variables (x : ℕ) (non_first_years first_years total_students : ℕ)

-- Definitions corresponding to the conditions.
definition non_first_year_students (x : ℕ) : ℕ := x
definition first_year_students (x : ℕ) : ℕ := x + 12

-- Main statement to prove
theorem first_year_students_count {x : ℕ} (h : 2 * x + 12 = 128) : first_year_students x = 70 :=
by sorry

-- Definitions for total students and the actual theorem we are proving for clarity.
definition total_students (x : ℕ) : ℕ := non_first_year_students x + first_year_students x

-- Ensure total students number condition holds
theorem number_of_students (h : 2 * x + 12 = 128) : total_students x = 128 :=
by rw [total_students, non_first_year_students, first_year_students]; exact h

end first_year_students_count_number_of_students_l719_719735


namespace complex_number_solution_l719_719510

noncomputable def z : ℂ := 1 + 2i

theorem complex_number_solution (h : z * (1 - I) = 3 + I) : z = 1 + 2i :=
by
  sorry

end complex_number_solution_l719_719510


namespace series_sum_l719_719447

-- Define the infinite series
def series := ∑' k : ℕ, (k^2 : ℝ) / 3^k

-- State the theorem to prove
theorem series_sum : series = 1 := 
sorry

end series_sum_l719_719447


namespace translated_vector_coordinates_l719_719169

variables {A B a : Point}

def Point := (ℝ × ℝ)

def vector (P Q : Point) : Point := (Q.1 - P.1, Q.2 - P.2)
def translate (v a : Point) : Point := (v.1 + a.1, v.2 + a.2)

theorem translated_vector_coordinates : 
  let A := (3, 7)
  let B := (5, 2)
  let a := (1, 2)
  translate (vector A B) a = (2, -5)  :=
by
  sorry

end translated_vector_coordinates_l719_719169


namespace number_of_functions_l719_719117

theorem number_of_functions : 
  ∃ (f : ℝ → ℝ) (a b c d : ℝ), 
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) ∧ 
  (∀ x, (f x) * (f (-x)) = f (x^3)) 
  ∧ (nat.card {f : ℝ → ℝ | ∃ a b c d : ℝ, 
                (∀ x, f x = a * x^3 + b * x^2 + c * x + d) 
                ∧ (∀ x, (f x) * (f (-x)) = f (x^3))}) = 6 := 
by sorry

end number_of_functions_l719_719117


namespace expectation_of_attempts_is_correct_probability_of_license_within_three_years_is_correct_l719_719395

noncomputable def probability_distribution : list ℝ :=
  [0.5, 0.3, 0.14, 0.048, 0.012]

theorem expectation_of_attempts_is_correct :
  ∑ i in [1, 2, 3, 4, 5], i * (probability_distribution.nth (i - 1)).get_or_else 0 = 1.772 := 
sorry

noncomputable def probability_failing_all_attempts : ℝ :=
  (1 - 0.5) * (1 - 0.6) * (1 - 0.7) * (1 - 0.8) * (1 - 0.9)

theorem probability_of_license_within_three_years_is_correct :
  1 - probability_failing_all_attempts = 0.9988 := 
sorry

end expectation_of_attempts_is_correct_probability_of_license_within_three_years_is_correct_l719_719395


namespace problem1_problem2_l719_719518

variables {a b c x y m : ℝ}

/-- (1) Prove that the equation of the ellipse C is x^2/4 + y^2/3 = 1 --/
noncomputable def ellipse_C_eq : Prop :=
  ∃ (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c : ℝ), 
  a > b ∧ b = sqrt 3 ∧ c = 1 ∧ a^2 = b^2 + c^2 ∧ 
  ∀ x y : ℝ, (L_c := line "x = my + 1") ∧ passes_through_right_focus (x / a^2 + y^2 / b^2 = 1) _
  -- Ellipse equation:
  ellipse_C_eq ∧ 
  (L_c := passes_through_right_focus) = ellipse_C_eq

/-- (2) Prove that λ1 + λ2 = -8/3 given the conditions and ellipse in (1) --/
noncomputable def lambda_sum_eq : Prop :=
  ∃ (m : ℝ) {y_1 y_2 x_1 x_2 : ℝ} (M : ℝ × ℝ), (L_c := line "x = my + 1") ∧ (\overrightarrow{MA} = λ_1 \overrightarrow{AF}) ∧ (\overrightarrow{MB} = λ_2 \overrightarrow{BF}) ∧ 
  (λ_1 + λ_2 = -8/3)

theorem problem1 : ellipse_C_eq := 
  sorry

theorem problem2 : ellipse_C_eq -> lambda_sum_eq := 
  sorry

end problem1_problem2_l719_719518


namespace ellipse_equation_max_area_triangle_l719_719165

theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + c = 3) (h4 : a^2 = b^2 + c^2) (h5 : b = sqrt 3) :
  (a = 2) ∧ (c = 1) → ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) :=
by
  intros h6 x y
  cases h6 with ha hc
  sorry

theorem max_area_triangle (x1 y1 x2 y2 k : ℝ) (h1 : y1 + y2 = 6 * k / (3 * k^2 + 4)) (h2 : y1 * y2 = -9 / (3 * k^2 + 4))
    (h3 : ∀ k : ℝ, S ≤ 3) : S = 3 :=
by
  intros h
  sorry

end ellipse_equation_max_area_triangle_l719_719165


namespace number_of_correct_factorizations_is_one_l719_719767

theorem number_of_correct_factorizations_is_one :
  (¬ (x^3 + 2 * x * y + x = x * (x^2 + 2 * y))) ∧ 
  (x^2 + 4 * x + 4 = (x + 2) ^ 2) ∧ 
  (¬ (-x^2 + y^2 = (x + y) * (x - y))) → 
  1 := 
by sorry

end number_of_correct_factorizations_is_one_l719_719767


namespace magnitude_of_a_correct_l719_719881

open Real

noncomputable def vec_mag {α : Type} [inner_product_space ℝ α] (v : α) : ℝ :=
  sqrt (inner_product_space.norm_sq v)

variables {V : Type} [inner_product_space ℝ V]

def magnitude_of_a (e1 e2 : V) (h_angle : inner e1 e2 = - (1 / 2)) : ℝ :=
  let a := e1 + 2 • e2 in
  vec_mag a

theorem magnitude_of_a_correct (e1 e2 : V)
  (h_norm1 : ∥e1∥ = 1)
  (h_norm2 : ∥e2∥ = 1)
  (h_angle : inner e1 e2 = - (1 / 2)) :
  magnitude_of_a e1 e2 h_angle = sqrt 3 := 
sorry

end magnitude_of_a_correct_l719_719881


namespace base3_problem_l719_719582

def is_valid_base3_rep (n : ℕ) : Prop :=
  let digits := n.digits 3
  let count_0 := digits.count 0
  let count_1 := digits.count 1
  let count_2 := digits.count 2
  count_2 > count_0 + count_1

def count_valid_numbers (limit : ℕ) : ℕ :=
  (List.range' 1 limit).count is_valid_base3_rep

theorem base3_problem :
  (count_valid_numbers 1001) % 500 = 233 :=
sorry

end base3_problem_l719_719582


namespace max_dist_ge_two_min_dist_l719_719855

noncomputable def max_dist {α : Type} [MetricSpace α] (s : Finset α) : ℝ :=
  s.sup' (Finset.nonempty_mk (by decide))
  (λ x, s.sup' (Finset.nonempty_mk (by decide)) (dist x))

noncomputable def min_dist {α : Type} [MetricSpace α] (s : Finset α) : ℝ :=
  s.inf' (Finset.nonempty_mk (by decide))
  (λ x, s.inf' (Finset.nonempty_mk (by decide)) (dist x))

theorem max_dist_ge_two_min_dist (points : Fin (10) → ℝ × ℝ) :
  max_dist (Finset.univ.image points)
  ≥ 2 * min_dist (Finset.univ.image points) :=
by 
  sorry

end max_dist_ge_two_min_dist_l719_719855


namespace sandra_remaining_money_l719_719986

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def num_candies : ℝ := 14
def num_jelly_beans : ℝ := 20

theorem sandra_remaining_money : (sandra_savings + mother_contribution + father_contribution) - (num_candies * candy_cost + num_jelly_beans * jelly_bean_cost) = 11 :=
by
  sorry

end sandra_remaining_money_l719_719986


namespace vector_problem_l719_719898

open Real

noncomputable def a : ℝ × ℝ × ℝ := (2, 4, x)
noncomputable def b : ℝ × ℝ × ℝ := (2, y, 2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem vector_problem (x y : ℝ)
  (h1 : magnitude (2, 4, x) = 6)
  (h2 : dot_product (2, 4, x) (2, y, 2) = 0) :
  x + y = 1 ∨ x + y = -3 :=
sorry

end vector_problem_l719_719898


namespace correct_statements_count_l719_719511

def z := 1 + complex.i
def statement1 := complex.abs z = real.sqrt 2
def statement2 := complex.conj z = 1 - complex.i
def statement3 := complex.im z = complex.i
def statement4 := (complex.re z > 0 ∧ complex.im z > 0)

theorem correct_statements_count : 
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4) ↔ 3 = 3 := 
by 
  sorry

end correct_statements_count_l719_719511


namespace trajectory_midpoint_l719_719155

variable (P Q : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (x y : ℝ)
variable a b : ℝ

-- Conditions
axiom point_on_circle : P.1^2 + P.2^2 = 5
axiom point_Q : Q = (0, -1)

-- Definitions
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def P_coordinates : P = (a, b) := by sorry
def M_coordinates : M = midpoint P Q := by sorry

-- Question Rewritten as a Proof
theorem trajectory_midpoint :
  (M.1 = x) → (M.2 = y) → x^2 + y^2 - y - 1 = 0 := by sorry

end trajectory_midpoint_l719_719155


namespace correct_statements_count_l719_719432

theorem correct_statements_count :
  (¬(∀ r : ℝ, abs r = 0) ∧
   (∀ x : ℝ, x^2 + x - 1 < 0) ↔ (∃ x : ℝ, x^2 + x - 1 ≥ 0) ∧
   (∀ p q : Prop, (p ∨ q ↔ true) ∧ (¬p → false)) ∧
   ((∃ m b : ℝ, m = 1.23 ∧ b = 5 - 1.23 * 4 ∧
     ∀ x y : ℝ, y = m * x + b) ↔ (∀ x y : ℝ, y = 1.23 * x + 0.08)) → 2) :=
by
  sorry

end correct_statements_count_l719_719432


namespace prob_diff_colors_correct_l719_719220

def total_chips := 6 + 5 + 4 + 3

def prob_diff_colors : ℚ :=
  (6 / total_chips * (12 / total_chips) +
  5 / total_chips * (13 / total_chips) +
  4 / total_chips * (14 / total_chips) +
  3 / total_chips * (15 / total_chips))

theorem prob_diff_colors_correct :
  prob_diff_colors = 119 / 162 := by
  sorry

end prob_diff_colors_correct_l719_719220


namespace polar_to_rectangular_line_and_circle_non_intersect_l719_719933

theorem polar_to_rectangular {x y : ℝ} (h₁ : ∀ (θ : ℝ), (x = ρ * cos θ) ∧ (y = ρ * sin θ))
(h₂ : ∀ (θ : ℝ), ρ = 2 * sqrt 2 * cos (θ + π / 4)) : 
x^2 + y^2 - 2*x + 2*y = 0 :=
begin
  sorry
end

theorem line_and_circle_non_intersect (A B C : ℝ) (h₃ : A = 1) (h₄ : B = -1) (h₅ : C = 2)
(center : ℝ × ℝ) (radius : ℝ) (h₆ : center = (1, -1)) (h₇ : radius = sqrt 2) :
  ∀ (x y : ℝ), (x - y + 2 = 0) → (x, y) ∈ metric.sphere (1, -1) (sqrt 2) → false :=
begin
  sorry
end

end polar_to_rectangular_line_and_circle_non_intersect_l719_719933


namespace equal_distances_l719_719863

-- Definitions of points and properties as per conditions
variables {A B C D K L X Y M : Type}
variables (dist : M → M → ℝ) (midpoint : M → M → M)

-- Given conditions as assumptions
variables (is_tangent_ωA_BCD : Prop)
variables (is_tangent_ωA_plane_outside_faces : Prop)
variables (is_tangent_ωB_ACD : Prop)
variables (is_tangent_ωB_plane_outside_faces : Prop)
variables (K_tangent_ACD : Prop)
variables (L_tangent_BCD : Prop)
variables (X_on_AK : Prop)
variables (Y_on_BL : Prop)
variables (angle_condition_1 : ∀ (C D X K : M), C ≠ D → K ≠ D → angle_data (angle C K D = angle C X D + angle C B D))
variables (angle_condition_2 : ∀ (C D Y L : M), C ≠ D → L ≠ D → angle_data (angle C L D = angle CY D + angle CAD))

-- Goal to prove equal distances
theorem equal_distances (h1 : is_tangent_ωA_BCD)
                        (h2 : is_tangent_ωA_plane_outside_faces)
                        (h3 : is_tangent_ωB_ACD)
                        (h4 : is_tangent_ωB_plane_outside_faces)
                        (h5 : K_tangent_ACD)
                        (h6 : L_tangent_BCD)
                        (h7 : X_on_AK)
                        (h8 : Y_on_BL)
                        (a1 : angle_condition_1)
                        (a2 : angle_condition_2) :
  dist X (midpoint C D) = dist Y (midpoint C D) :=
sorry

end equal_distances_l719_719863


namespace intervals_of_monotonicity_range_of_a_l719_719711

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + a * x + b

theorem intervals_of_monotonicity (b : ℝ) :
  (∀ x : ℝ, -x^3 + x^2 + 3 * x + b = f 3 b x) →
  (∀ x : ℝ, derivative (f 3 b) x = -3 * x^2 + 2 * x + 3) →
  (∀ x : ℝ, derivative (f 3 b) x > 0 ↔ -1 < x ∧ x < 3) ∧
  (∀ x : ℝ, derivative (f 3 b) x < 0 ↔ x < -1 ∨ x > 3) :=
sorry

theorem range_of_a (a b : ℝ) :
  (∀ x : ℝ, slope_of_tangent (f a b) x < 2 * a^2) →
  (∀ x : ℝ, -3 * x^2 + 2 * x + a < 2 * a^2) →
  (∀ x : ℝ, x^2 - 2 * x > a - 2 * a^2) →
  (∀ x : ℝ, (x - 1)^2 - 1 > a - 2 * a^2) →
  (a > 1 ∨ a < -1 / 2) :=
sorry

end intervals_of_monotonicity_range_of_a_l719_719711


namespace square_area_l719_719388

theorem square_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  ∃ (s : ℝ), (s * Real.sqrt 2 = d) ∧ (s^2 = 144) := by
  sorry

end square_area_l719_719388


namespace inbox_emails_after_movements_l719_719625

def initial_emails := 400
def trash_emails := initial_emails / 2
def remaining_emails := initial_emails - trash_emails
def work_emails := 0.4 * remaining_emails
def final_inbox_emails := remaining_emails - work_emails

theorem inbox_emails_after_movements : final_inbox_emails = 120 :=
by
  sorry

end inbox_emails_after_movements_l719_719625


namespace area_of_triangle_CBG_l719_719937

theorem area_of_triangle_CBG :
  (∀ A B C D E F G : Point,
  isosceles_triangle A B C ∧ radius_of_circle_by_triangle A B C = 3 ∧
  on_extension B D AB ∧ on_extension C E AC ∧
  line_through_parallel_to D A E AE ∧ line_through_parallel_to E A D AD ∧
  point_intercepts F l1 l2 ∧ point_collinear_with G A F on_circle_of_triangle ABC ∧
  length BC = 6 ∧ length AD = 17 ∧ length AE = 15)
  →
  ∃ (p q r : ℕ), relatively_prime p r ∧ not_divisible_by_square q ∧
  area_of_triangle C B G = ((p : ℚ) * real.sqrt(q)) / (r : ℚ) ∧ p + q + r = 4882 :=
sorry

end area_of_triangle_CBG_l719_719937


namespace carmen_sugar_cookies_l719_719784

variables
  (boxes_samoas : ℕ)
  (price_per_samoa : ℝ)
  (boxes_thin_mints : ℕ)
  (price_per_thin_mint : ℝ)
  (box_fudge_delights : ℕ)
  (price_per_fudge_delight : ℝ)
  (boxes_sugar_cookies : ℕ)
  (price_per_sugar_cookie : ℝ)
  (total_earnings : ℝ)

-- Given conditions
def condition_1 : boxes_samoas = 3 := by sorry
def condition_2 : price_per_samoa = 4 := by sorry
def condition_3 : boxes_thin_mints = 2 := by sorry
def condition_4 : price_per_thin_mint = 3.5 := by sorry
def condition_5 : box_fudge_delights = 1 := by sorry
def condition_6 : price_per_fudge_delight = 5 := by sorry
def condition_7 : price_per_sugar_cookie = 2 := by sorry
def condition_8 : total_earnings = 42 := by sorry

-- Expected answer
def expected_answer : boxes_sugar_cookies = 9 := by sorry

-- Proof statement
theorem carmen_sugar_cookies:
  boxes_samoas * price_per_samoa +
  boxes_thin_mints * price_per_thin_mint +
  box_fudge_delights * price_per_fudge_delight +
  boxes_sugar_cookies * price_per_sugar_cookie = total_earnings →
  boxes_sugar_cookies = expected_answer := by
  intro h
  sorry

end carmen_sugar_cookies_l719_719784


namespace larger_jar_half_full_l719_719438

-- Defining the capacities of the jars
variables (S L W : ℚ)

-- Conditions
def equal_amount_water (S L W : ℚ) : Prop :=
  W = (1/5 : ℚ) * S ∧ W = (1/4 : ℚ) * L

-- Question: What fraction will the larger jar be filled if the water from the smaller jar is added to it?
theorem larger_jar_half_full (S L W : ℚ) (h : equal_amount_water S L W) :
  (2 * W) / L = (1 / 2 : ℚ) :=
sorry

end larger_jar_half_full_l719_719438


namespace center_of_symmetry_ratio_bc_l719_719515

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos (2 * x + Real.pi / 3) - 2 * Real.cos (2 * x) + 1

theorem center_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f x = 1 → x = -Real.pi / 12 + k * Real.pi / 2 :=
sorry

variable {a b c : ℝ}
variable {A B C : ℝ}
variable {f_A_zero : f A = 0}
variable {acute_triangle : A + B + C = Real.pi ∧ A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2}

theorem ratio_bc :
  acute_triangle → (∃ b c : ℝ, b / c ∈ Ioo (1 / 2) 2) :=
sorry

end center_of_symmetry_ratio_bc_l719_719515


namespace arithmetic_sequence_sum_eq_48_l719_719230

variable (a : ℕ → ℝ) (d : ℝ)
-- Define the arithmetic properties
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence terms for a_4 and a_5 
def a4 := a 4
def a5 := a 5

def S8 (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 8, a i  -- Sum of the first 8 terms of the sequence

theorem arithmetic_sequence_sum_eq_48
  (h_seq : arithmetic_sequence a)
  (h_cond : a4 + a5 = 12) :
  S8 a = 48 :=
  sorry

end arithmetic_sequence_sum_eq_48_l719_719230


namespace find_m_value_l719_719501

open Nat

theorem find_m_value {m : ℕ} (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 24 :=
  sorry

end find_m_value_l719_719501


namespace find_F_2_l719_719543

noncomputable def f (x : ℝ) := -- assume some definition of f here
noncomputable def g (x : ℝ) := -- assume some definition of g here

def odd_function (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h (x)

variable (a b : ℝ) (F : ℝ → ℝ)
hypothesis h1 : odd_function f
hypothesis h2 : odd_function g
hypothesis h3 : F = λ x, a * f x + b * g x + 2
hypothesis h4 : F (-2) = 5

theorem find_F_2 : F 2 = -1 := by
  sorry

end find_F_2_l719_719543


namespace triangle_is_isosceles_not_equilateral_l719_719167

variables {V : Type*} [inner_product_space ℝ V] {A B C : V}

theorem triangle_is_isosceles_not_equilateral
  (h1 : (\frac {(B-A)}∥B-A∥+ \frac {(C-A)}∥C-A∥) • (C-B) = 0)
  (h2 : (\frac {(B-A)}∥B-A∥) • (\frac {(C-A)}∥C-A∥) = -1/2) :
  ∃ (ABC : triangle), is_isosceles_triangle ABC ∧ ¬ is_equilateral_triangle ABC :=
begin
  sorry,
end

end triangle_is_isosceles_not_equilateral_l719_719167


namespace find_counterfeit_coin_in_6_weighings_l719_719100

theorem find_counterfeit_coin_in_6_weighings :
  ∀ (coins : Finset ℕ) (n : ℕ),
  coins.card = 64 ∧
  (∀ coin ∈ coins, 0 < coin) ∧
  (∃ counterfeit ∈ coins, ∀ genuine ∈ coins, genuine ≠ counterfeit → coin < genuine) →
  ∃ k ≤ 6, ∀ F : Finset ℕ → ℕ, 
  (∀ coins, F coins = if (Finset.card coins ≤ 1) then 0 else 1 + F (if (F coins % 2 = 0) then Finset.take (Finset.card coins / 2) coins else Finset.drop (Finset.card coins / 2) coins)) ∧
  F coins = k :=
sorry

end find_counterfeit_coin_in_6_weighings_l719_719100


namespace triangle_pairs_bound_l719_719482

theorem triangle_pairs_bound (n l T : ℕ)
  (h1 : n ≥ 4)
  (h2 : l ≥ (n^2 / 4) + 1)
  (h3 : ∀ (points : Fin n → Point), no three points are collinear)
  (h4 : connected_by_segments points l)
  (h5 : T = triangle_pairs_formed points l) :
  T ≥ l * (4 * l - n^2) * (4 * l - n^2 - n) / (2 * n^2) :=
sorry

end triangle_pairs_bound_l719_719482


namespace sum_of_squares_greater_than_cubics_l719_719180

theorem sum_of_squares_greater_than_cubics (a b c : ℝ)
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a)
  : 
  (2 * (a + b + c) * (a^2 + b^2 + c^2)) / 3 > a^3 + b^3 + c^3 + a * b * c := 
by 
  sorry

end sum_of_squares_greater_than_cubics_l719_719180


namespace sharpened_off_length_l719_719250

-- Define the conditions
def original_length : ℤ := 31
def length_after_sharpening : ℤ := 14

-- Define the theorem to prove the length sharpened off is 17 inches
theorem sharpened_off_length : original_length - length_after_sharpening = 17 := sorry

end sharpened_off_length_l719_719250


namespace win_sector_area_l719_719725

-- Defining the conditions
def radius : ℝ := 12
def total_area : ℝ := π * radius^2
def win_probability : ℝ := 1 / 3

-- Theorem to prove the area of the WIN sector
theorem win_sector_area : total_area * win_probability = 48 * π := by
  sorry

end win_sector_area_l719_719725


namespace smallest_and_largest_unique_digit_sum_seventeen_l719_719667

theorem smallest_and_largest_unique_digit_sum_seventeen :
  (∃ (n : ℕ), ( ∀ d ∈ (Nat.digits 10 n), d ≠ 0 → ∃ (m : ℕ), m ≠ d ∧ m ∉ Nat.digits 10 n) ∧ (Nat.digits 10 n).sum = 17 ∧ n = 89) ∧
  (∃ (n : ℕ), ( ∀ d ∈ (Nat.digits 10 n), d ≠ 0 → ∃ (m : ℕ), m ≠ d ∧ m ∉ Nat.digits 10 n) ∧ (Nat.digits 10 n).sum = 17 ∧ n = 743210) :=
by
  sorry

end smallest_and_largest_unique_digit_sum_seventeen_l719_719667


namespace cylindrical_surface_area_increase_l719_719372

theorem cylindrical_surface_area_increase (x : ℝ) :
  (2 * Real.pi * (10 + x)^2 + 2 * Real.pi * (10 + x) * (5 + x) = 
   2 * Real.pi * 10^2 + 2 * Real.pi * 10 * (5 + x)) →
   (x = -10 + 5 * Real.sqrt 6 ∨ x = -10 - 5 * Real.sqrt 6) :=
by
  intro h
  sorry

end cylindrical_surface_area_increase_l719_719372


namespace original_seat_cost_l719_719750

theorem original_seat_cost (x : ℝ) : 
  (∀ (rows seats : ℕ), rows = 5 ∧ seats = 8) → 
  (discount_per_group : ℝ) (total_seats groups : ℕ) (rows seats x final_cost : ℝ) 
  (discount : ℝ) (original_cost : ℝ),
  total_seats = rows * seats ∧ 
  groups = total_seats / 10 ∧ 
  discount_per_group = 0.10 * 10 * x ∧ 
  discount = groups * discount_per_group ∧ 
  final_cost = 1080 ∧ 
  original_cost = total_seats * x ∧ 
  original_cost - discount = final_cost -> 
  x = 30 :=
by
  sorry

end original_seat_cost_l719_719750


namespace theater_performance_repetition_l719_719054

-- Defining the conditions as Lean definitions
def total_seats := 400
def fill_percentage := 0.8
def ticket_price := 30
def total_revenue := 28800

-- The number of seats filled per performance
def seats_filled := total_seats * fill_percentage

-- The revenue per performance
def revenue_per_performance := seats_filled * ticket_price

-- Number of performances
def number_of_performances := total_revenue / revenue_per_performance

-- The number of other days they repeated the performance
def initial_performance := 1
def other_days_performed := number_of_performances - initial_performance

theorem theater_performance_repetition :
  other_days_performed = 2 := by
  -- Proof skipped with sorry
  sorry

end theater_performance_repetition_l719_719054


namespace sum_G_eq_10098_l719_719135

def G (n : ℕ) : ℕ := 2 * n

theorem sum_G_eq_10098 : ∑ n in Finset.range 99 \ Finset.singleton 0, G (n + 2) = 10098 :=
by
  -- The proof is not required.
  sorry

end sum_G_eq_10098_l719_719135


namespace distance_GH_l719_719260

-- Given conditions of the trapezoid and points
variables (A B C D G H : ℝ → ℝ → Prop)
variable (AD BC : ℝ)
variable (angleBAD angleCDA : ℝ)
variable (diagonalLength : ℝ)
variable (distanceGA distanceGD : ℝ)
variable (footOfAltitudeBH : Bool)

-- Assume the given conditions as axioms
axiom trapezoid_properties : 
  is_isosceles_trapezoid ABCD ∧ 
  AD ∥ BC ∧ 
  angleBAD = π / 4 ∧ 
  angleCDA = π / 4 ∧ 
  diagonalLength = 20 * sqrt 5 ∧ 
  distanceGA = 20 ∧ 
  distanceGD = 40 ∧ 
  footOfAltitudeBH ∧ 
  ||A - D|| = 40

-- The goal is to prove the distance GH
theorem distance_GH : 
  trapezoid_properties A B C D G H AD BC angleBAD angleCDA diagonalLength distanceGA distanceGD footOfAltitudeBH → 
  ||G - H|| = 40 :=
sorry

end distance_GH_l719_719260


namespace simplify_tan_expression_l719_719990

theorem simplify_tan_expression :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have tan_add : Real.tan (10 * Real.pi / 180 + 35 * Real.pi / 180) = (Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180)) / (1 - Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180)) := by sorry
  have eq : Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180) = 1 - Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180) := by
    rw [← tan_add, tan_45]
    field_simp
    ring
  have res : (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 1 + (Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180)) + Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180) := by ring
  rw [eq, res]
  ring
  sorry

end simplify_tan_expression_l719_719990


namespace minimize_M_values_l719_719304
noncomputable def F (A B x : ℝ) : ℝ := abs ((Real.cos x)^2 + 2 * Real.sin x * Real.cos x - (Real.sin x)^2 + A * x + B)

theorem minimize_M_values (A B : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ (3 / 2) * Real.pi → F A B x ≤ sqrt 2) →
  (A = 0 ∧ B = 0) :=
by
  sorry

end minimize_M_values_l719_719304


namespace most_likely_sum_exceeding_twelve_l719_719740

-- Define a die with faces 0, 1, 2, 3, 4, 5
def die_faces : List ℕ := [0, 1, 2, 3, 4, 5]

-- Define a function to get the sum of rolled results exceeding 12
noncomputable def sum_exceeds_twelve (rolls : List ℕ) : ℕ :=
  let sum := rolls.foldl (· + ·) 0
  if sum > 12 then sum else 0

-- Define a function to simulate the die roll until the sum exceeds 12
noncomputable def roll_die_until_exceeds_twelve : ℕ :=
  sorry -- This would contain the logic to simulate the rolling process

-- The theorem statement that the most likely value of the sum exceeding 12 is 13
theorem most_likely_sum_exceeding_twelve : roll_die_until_exceeds_twelve = 13 :=
  sorry

end most_likely_sum_exceeding_twelve_l719_719740


namespace find_x_l719_719875

theorem find_x (x : ℝ) (h1 : Vector := (3, 2)) (h2 : Vector := (x, 4)) 
  (h_parallel : h1 ∥ h2) : x = 6 :=
sorry

end find_x_l719_719875


namespace sin_A_in_triangle_dot_product_BC_CA_l719_719921

noncomputable def triangle_ABCs (A B C : ℝ → ℝ → Prop) (AB BC : ℝ) (cosC : ℝ) :=
  ∃ (a b c : ℝ), 
    AB = sqrt 2 ∧ 
    BC = 1 ∧ 
    cosC = 3 / 4

theorem sin_A_in_triangle (A B C : ℝ → ℝ → Prop) : 
  triangle_ABCs A B C (sqrt 2) 1 (3 / 4) → 
  let sinA := sqrt 14 / 8 in sinA = sqrt 14 / 8 := 
by
  intros hABC
  let sinA := sqrt 14 / 8
  exact eq.refl sinA
  sorry

theorem dot_product_BC_CA (A B C : ℝ → ℝ → Prop) : 
  triangle_ABCs A B C (sqrt 2) 1 (3 / 4) → 
  let dotProduct := -3 / 2 in dotProduct = -3 / 2 := 
by
  intros hABC
  let dotProduct := -3 / 2
  exact eq.refl dotProduct
  sorry

end sin_A_in_triangle_dot_product_BC_CA_l719_719921


namespace find_prices_maximize_profit_l719_719369

-- Definition of conditions
def sales_eq1 (m n : ℝ) : Prop := 150 * m + 100 * n = 1450
def sales_eq2 (m n : ℝ) : Prop := 200 * m + 50 * n = 1100

def profit_function (x : ℕ) : ℝ := -2 * x + 1500
def range_x (x : ℕ) : Prop := 375 ≤ x ∧ x ≤ 500

-- Theorem to prove the unit prices
theorem find_prices : ∃ m n : ℝ, sales_eq1 m n ∧ sales_eq2 m n ∧ m = 3 ∧ n = 10 := 
sorry

-- Theorem to prove the profit function and maximum profit
theorem maximize_profit : ∃ (x : ℕ) (W : ℝ), range_x x ∧ W = profit_function x ∧ W = 750 :=
sorry

end find_prices_maximize_profit_l719_719369


namespace sum_of_squares_of_distances_is_constant_l719_719920

variable {r1 r2 : ℝ}
variable {x y : ℝ}

theorem sum_of_squares_of_distances_is_constant
  (h1 : r1 < r2)
  (h2 : x^2 + y^2 = r1^2) :
  let PA := (x - r2)^2 + y^2
  let PB := (x + r2)^2 + y^2
  PA + PB = 2 * r1^2 + 2 * r2^2 :=
by
  sorry

end sum_of_squares_of_distances_is_constant_l719_719920


namespace paint_cube_l719_719559

theorem paint_cube (colors : Fin 7 → ℕ) (faces : Fin 6 → Fin 7) :
  (∃ h : ∀ (i j : Fin 6), i ≠ j → faces i ≠ faces j, 
    ∃ g : Function.Bijective (colors ∘ faces),
    210) :=
sorry

end paint_cube_l719_719559


namespace uma_fraction_part_l719_719983

theorem uma_fraction_part (r s t u : ℕ) 
  (hr : r = 6) 
  (hs : s = 5) 
  (ht : t = 7) 
  (hu : u = 8) 
  (shared_amount: ℕ)
  (hr_amount: shared_amount = r / 6)
  (hs_amount: shared_amount = s / 5)
  (ht_amount: shared_amount = t / 7)
  (hu_amount: shared_amount = u / 8) :
  ∃ total : ℕ, ∃ uma_total : ℕ, uma_total * 13 = 2 * total :=
sorry

end uma_fraction_part_l719_719983


namespace determine_k_m_l719_719435

theorem determine_k_m :
  ∃ k m : ℝ, (2 * k * 3 - m * (-2) = 4) ∧ (m = 2 * k) ∧ (k = 2 / 5) ∧ (m = 4 / 5) :=
by
  existsi (2 / 5)
  existsi (4 / 5)
  split
  { norm_num }  -- This verifies the equation 2 * k * 3 - m * (-2) = 4
  split
  { norm_num }  -- This verifies the condition m = 2 * k
  split
  { norm_num }  -- This verifies k = 2 / 5
  { norm_num }  -- This verifies m = 4 / 5

end determine_k_m_l719_719435


namespace domain_of_f_l719_719302

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (3 - x)) + Real.log (x + 2)

theorem domain_of_f : {x : ℝ | -2 < x ∧ x < 3} = {x : ℝ | f x ∈ ℝ} :=
by
  sorry

end domain_of_f_l719_719302


namespace largest_power_of_3_dividing_Q_l719_719583

-- Define the product of the first 150 positive odd integers
def productOdd150 : ℕ := (List.range 150).map (λ i => 2 * (i + 1) - 1) |> List.prod

-- Define the function to count the largest power of a prime p dividing n!
def countPrimePowerInFactorial (n p : ℕ) : ℕ :=
  if p = 0 ∨ p = 1 then 0 else List.range (nat.log (p, n) + 1) |> List.map (λ k => n / p^k) |> List.sum

-- Define the problem statement
theorem largest_power_of_3_dividing_Q : 
  ∃ k : ℕ, 
  (productOdd150 % 3^k = 0) 
  ∧ (∀ m : ℕ, m > k → productOdd150 % 3^m ≠ 0) 
  ∧ k = 75 := 
by
  sorry

end largest_power_of_3_dividing_Q_l719_719583


namespace num_possible_values_l719_719586

variable {x y z : ℝ}

def floor_x := floor x = 5
def floor_y := floor y = -3
def floor_z := floor z = -2

-- State the theorem
theorem num_possible_values (h1 : floor_x) (h2 : floor_y) (h3 : floor_z) : 
  ∃ n, (5 ≤ n ∧ n ≤ 7) ∧ n ∈ {floor (x - y + z)} := 
sorry  -- Proof goes here

end num_possible_values_l719_719586


namespace win_sector_area_l719_719729

-- Given Conditions
def radius := 12  -- radius of the circle in cm
def probability_of_winning := 1 / 3  -- probability of winning on one spin

-- Calculate the total area of the circle
def total_area_circle : ℝ := real.pi * radius^2

-- Calculate the area of the WIN sector
def area_of_win_sector : ℝ := probability_of_winning * total_area_circle

-- Proof Statement
theorem win_sector_area : area_of_win_sector = 48 * real.pi :=
by
  -- proof steps will go here
  sorry

end win_sector_area_l719_719729


namespace bus_seat_capacity_l719_719221

theorem bus_seat_capacity (x : ℕ) :
  (15 * x) + (12 * x) + 12 = 93 → x = 3 :=
by
  intros h,
  sorry

end bus_seat_capacity_l719_719221


namespace find_point_B_l719_719746

-- The conditions are translated to definitions:
def A : ℝ × ℝ × ℝ := (-4, 10, 12)
def C : ℝ × ℝ × ℝ := (4, 4, 8)
def plane (x y z : ℝ) : Prop := x + y + z = 15

-- The target is to prove B is the point of reflection satisfying given conditions:
theorem find_point_B (B : ℝ × ℝ × ℝ) : 
  (∃ x y z : ℝ, B = (x, y, z) ∧ plane x y z) ∧ 
  module (A, B) = module (B, C) :=
  (∃ (x y z : ℝ), B = (x, y, z) ∧ plane x y z) ∧ 
   B = ( 1/3, 30/11, 80/11 ) :=
by
  sorry

end find_point_B_l719_719746


namespace problem_l719_719472

def f0 (y : ℝ) : ℝ := 0

def f (b : ℝ) (f_prev : ℝ → ℝ) (y : ℝ) (a : ℝ) : ℝ :=
  let m := int.floor (y / a)
  list.maximum (list.map (λ x : ℕ, b * x + f_prev (y - a*x))
    (list.range (nat.succ m.to_nat)))

theorem problem (P : list (ℝ × ℝ)) (a1 a2 : ℝ) (b1 b2 : ℝ) (p : ℝ) (h1 : P = [(2, 3), (3, 4), (3, p)]) :
  (f 3 f0 7 2 = 9) ∧ (f 4 (f 3 f0 7 2) 7 3 = 10) ∧ (4 < p ∧ p ≤ 4.5) := 
by
  sorry

end problem_l719_719472


namespace clock_angle_at_3_30_l719_719082

theorem clock_angle_at_3_30 :
  let hour_angle : ℝ := 90 + (30 / 60) * 30
  let minute_angle : ℝ := 30 * 6
  abs (minute_angle - hour_angle) = 75
:=
by
  let hour_angle : ℝ := 90 + (30 / 60) * 30
  let minute_angle : ℝ := 30 * 6
  have h1 : hour_angle = 105 := by sorry
  have h2 : minute_angle = 180 := by sorry
  show abs (minute_angle - hour_angle) = 75 from sorry

end clock_angle_at_3_30_l719_719082


namespace infinite_series_converges_to_3_l719_719789

noncomputable def sum_of_series := ∑' k in (Finset.range ∞).filter (λ k, k > 0), 
  (8 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))

theorem infinite_series_converges_to_3 : sum_of_series = 3 := 
  sorry

end infinite_series_converges_to_3_l719_719789


namespace sum_k_squared_div_3_k_l719_719453

theorem sum_k_squared_div_3_k : ∑ k in (Finset.range n).map (λ x, x+1), (k^2 / (3^k : ℝ)) = 4 :=
by
  sorry

end sum_k_squared_div_3_k_l719_719453


namespace tangent_line_at_pi_l719_719824

theorem tangent_line_at_pi :
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.sin x) → 
  ∀ x, x = Real.pi →
  ∀ y, (y = -x + Real.pi) ↔
        (∀ x, y = -x + Real.pi) := 
  sorry

end tangent_line_at_pi_l719_719824


namespace calc_expression_l719_719033

theorem calc_expression : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end calc_expression_l719_719033


namespace arithmetic_sequence_first_term_l719_719190

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_terms_int : ∀ n, ∃ k : ℤ, a n = k) 
  (ha20 : a 20 = 205) : a 1 = 91 :=
sorry

end arithmetic_sequence_first_term_l719_719190


namespace bisects_AP_CD_l719_719268

noncomputable section

variables (A B C D E P : Type) [convex (pentagon A B C D E)] 
  (angleBAC : angle A B C) (angleCAD : angle C A D) (angleDAE : angle D A E)
  (angleCBA : angle C B A) (angleDCA : angle D C A) (angleEDA : angle E D A)
  (intersectionBDCE : ∃ P, is_inter (line B D) (line C E) P)

theorem bisects_AP_CD :
  (angleBAC = angleCAD ∧ angleCAD = angleDAE) →
  (angleCBA = angleDCA ∧ angleDCA = angleEDA) →
  ∃ M, is_midpoint (line A P) (segment C D) M :=
by
  intros h_angles1 h_angles2 h_intersection
  sorry

end bisects_AP_CD_l719_719268


namespace sequence_is_constant_l719_719832

noncomputable def sequence_condition (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ 2 * m * n / (m ^ 2 + n ^ 2)

theorem sequence_is_constant (a : ℕ → ℝ) 
  (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 :=
by
  sorry

end sequence_is_constant_l719_719832


namespace sum_of_coordinates_of_B_l719_719285

noncomputable def point := ℝ × ℝ

def midpoint (A B : point) : point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem sum_of_coordinates_of_B (A M : point) (B : point) 
(hA : A = (10, 6)) (hM : M = (4, 4)) (h_mid : M = midpoint A B) :
  B.1 + B.2 = 0 :=
sorry

end sum_of_coordinates_of_B_l719_719285


namespace base_conversion_l719_719541

theorem base_conversion (b : ℕ) 
  (h1 : 35_b = 3 * b + 5)
  (h2 : 1225_b = 1 * b^3 + 2 * b^2 + 2 * b + 5) :
  (3 * b + 5)^2 = 1 * b^3 + 2 * b^2 + 2 * b + 5 → b = 10 :=
by
  sorry

end base_conversion_l719_719541


namespace area_m1_n1_area_m2_n2_combined_area_max_value_area_relation_l719_719236

-- Define vectors and areas
structure Vec2 :=
  (x : ℝ)
  (y : ℝ)

def area (u v : Vec2) : ℝ :=
  |u.x * v.y - v.x * u.y|

-- Specific vectors for the problems
def m1 : Vec2 := ⟨2, 1⟩
def n1 : Vec2 := ⟨-1, 2⟩

def m2 : Vec2 := ⟨1, 2⟩
def n2 : Vec2 := ⟨2, 4⟩

-- Prove the areas for specific vectors
theorem area_m1_n1 : area m1 n1 = 5 := 
by sorry

theorem area_m2_n2 : area m2 n2 = 0 := 
by sorry

-- Define combinations and areas with given conditions
def combined_area (a b c : Vec2) : ℝ :=
  (area a b) + (area b c) + (area c a)

-- Prove the combined area is 3 * sqrt(3) / 2
theorem combined_area_max_value (a b c : Vec2) (h1 : |a.x| = 1) (h2 : |a.y| = 1) 
  (h3 : |b.x| = 1) (h4 : |b.y| = 1) (h5 : |c.x| = 1) (h6 : |c.y| = 1) :
  combined_area a b c = 3 * real.sqrt 3 / 2 :=
by sorry

-- Vector and scalar definitions for Part II
def λ : ℝ := sorry
def μ : ℝ := sorry
variable (λ_nonzero : (λ^2 + μ^2 ≠ 0))

def p (λ μ : ℝ) (m n : Vec2) : Vec2 :=
  ⟨λ * m.x + μ * n.x, λ * m.y + μ * n.y⟩

-- Prove the relationship for the defined p
theorem area_relation (λ μ : ℝ) (m n : Vec2) :
  area (p λ μ m n) m + area (p λ μ m n) n = (|λ| + |μ|) * area m n :=
by sorry

end area_m1_n1_area_m2_n2_combined_area_max_value_area_relation_l719_719236


namespace num_intersection_points_is_zero_l719_719872

variable {a b c : ℝ}

theorem num_intersection_points_is_zero (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
    let Δ := (2 * c)^2 - 4 * (a + b) * (a + b)
    Δ < 0 → roots (a + b) (2 * c) (a + b) = 0 :=
by
  let Δ := (2 * c)^2 - 4 * (a + b) * (a + b)
  sorry

end num_intersection_points_is_zero_l719_719872


namespace extremum_at_one_l719_719194

theorem extremum_at_one (a b : ℝ) (h1 : polynomial.eval 1 (polynomial.derivative (polynomial.C a^2 + polynomial.X * polynomial.C b + polynomial.X^2 * polynomial.C a + polynomial.X^3) ) = 0)
    (h2 : polynomial.eval 1 (polynomial.C a^2 + polynomial.X * polynomial.C b + polynomial.X^2 * polynomial.C a + polynomial.X^3) = 10) :
    a = 4 ∧ b = -11 := by
sorry

end extremum_at_one_l719_719194


namespace ratio_green_to_yellow_l719_719973

theorem ratio_green_to_yellow (yellow fish blue fish green fish total fish : ℕ) 
  (h_yellow : yellow = 12)
  (h_blue : blue = yellow / 2)
  (h_total : total = yellow + blue + green)
  (h_aquarium_total : total = 42) : 
  green / yellow = 2 := 
sorry

end ratio_green_to_yellow_l719_719973


namespace matrix_power_conditions_l719_719156

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ :=
  let A := Matrix.ofVects [[0, 1], [1, 2]]
  let B := Matrix.ofVects [[1, 2], [0, 1]]
  have h : M ⬝ A = B := sorry
  Matrix.mulMatrix (Matrix.inv A) B

theorem matrix_power_conditions (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (hM : M ⬝ Matrix.ofVects [[0, 1], [1, 2]] = Matrix.ofVects [[1, 2], [0, 1]]) :
  (M ^ 2 = Matrix.diag ![1, 1]) ∧ ((M ^ 2014) ⬝ Vector.ofVec [2, -4] = Vector.ofVec [2, -4]) :=
by {
  sorry
}

end matrix_power_conditions_l719_719156


namespace largest_d_for_1_in_range_l719_719838

-- Define the function g
def g (x : ℝ) (d : ℝ) : ℝ := 2 * x^2 - 8 * x + d

-- Define the condition that 1 should be in the range of g
def is_in_range_of_g (d : ℝ) : Prop :=
  ∃ x : ℝ, g x d = 1

-- Prove that the largest value of d for which 1 is in the range of g is 9
theorem largest_d_for_1_in_range : ∃ d : ℝ, (d = 9 ∧ is_in_range_of_g(d)) ∧
  ∀ d' : ℝ, is_in_range_of_g(d') → d' ≤ 9 :=
by
  sorry

end largest_d_for_1_in_range_l719_719838


namespace area_triangle_BEM_l719_719242

theorem area_triangle_BEM (A B C M D E : Point)
  (h_isosceles : isosceles_right_triangle A B C)
  (h_midpoint : midpoint A C M)
  (h_perpendicular_MD : perpendicular M D A C)
  (h_perpendicular_BE : perpendicular B E A C)
  (h_area_ABC : area_triangle A B C = 50) :
  area_triangle B E M = 25 :=
sorry

end area_triangle_BEM_l719_719242


namespace median_sum_inequality_l719_719709

variable (a b c : ℝ) -- sides of the triangle
variable (r : ℝ) -- radius of the incircle
variable (right_triangle : ∠C = 90°) -- right angle at C
variable (m_a m_b : ℝ) -- medians to sides a and b

-- Definitions for required variables and properties
def median_to_side (x y z : ℝ) : ℝ := (1 / 2) * math.sqrt(2 * (y * y + z * z) - x * x)
def incircle_radius (a b c : ℝ) : ℝ := (a + b - c) / 2

-- Constraints based on the problem
axiom median_def_a : m_a = median_to_side a b c
axiom median_def_b : m_b = median_to_side b a c
axiom incircle_def : r = incircle_radius a b c
axiom right_triangle_hypotenuse : c * c = a * a + b * b  -- Pythagorean theorem for right triangle

-- Theorem to be proved
theorem median_sum_inequality : m_a^2 + m_b^2 > 29 * r^2 :=
sorry

end median_sum_inequality_l719_719709


namespace isosceles_triangle_legs_length_l719_719217

-- Define the given conditions in Lean
def perimeter (L B: ℕ) : ℕ := 2 * L + B
def base_length : ℕ := 8
def given_perimeter : ℕ := 20

-- State the theorem to be proven
theorem isosceles_triangle_legs_length :
  ∃ (L : ℕ), perimeter L base_length = given_perimeter ∧ L = 6 :=
by
  sorry

end isosceles_triangle_legs_length_l719_719217


namespace median_of_36_consecutive_integers_l719_719666

theorem median_of_36_consecutive_integers (sum_of_integers : ℕ) (h : sum_of_integers = 6^5) :
  let n := 36 in
  let median := sum_of_integers / n in
  median = 216 := 
by
  -- proof will be here
  sorry

end median_of_36_consecutive_integers_l719_719666


namespace primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l719_719035

-- Define Part (a)
theorem primitive_root_coprime_distinct_residues (m k : ℕ) (h: Nat.gcd m k = 1) :
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∀ i j s t, (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k) :=
sorry

-- Define Part (b)
theorem noncoprime_non_distinct_residues (m k : ℕ) (h: Nat.gcd m k > 1) :
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∃ i j x t, (i ≠ x ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a x * b t) % (m * k) :=
sorry

end primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l719_719035


namespace possible_value_m_l719_719916

theorem possible_value_m (x m : ℝ) (h : ∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) : m ≥ -25 / 8 := sorry

end possible_value_m_l719_719916


namespace pump_no_leak_fill_time_l719_719386

noncomputable def pump_fill_time (P t l : ℝ) :=
  1 / P - 1 / l = 1 / t

theorem pump_no_leak_fill_time :
  ∃ P : ℝ, pump_fill_time P (13 / 6) 26 ∧ P = 2 :=
by
  sorry

end pump_no_leak_fill_time_l719_719386


namespace expected_value_xi_l719_719688

-- Define the conditions: 5 fair coins, each trial, and \( \xi \)
def coin : Type := bool -- Represent each coin with a boolean value, heads or tails
def trial : Type := vector coin 5 -- A trial consists of tossing 5 coins

-- Probability of exactly one head in a single trial
def prob_one_head : ℝ := 5 * (1 / 2) ^ 5

-- Number of trials: 160
def n_trials : ℕ := 160

-- Expected value of \(\xi\)
def expected_value : ℝ := n_trials * prob_one_head

-- Main theorem statement
theorem expected_value_xi :
  expected_value = 25 := by
  sorry

end expected_value_xi_l719_719688


namespace find_tan_sum_l719_719208

-- Define the vectors a and b and the condition for parallelism
variables (α : ℝ)
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (Real.sin α, Real.cos α)
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- Main theorem to prove
theorem find_tan_sum :
  3 * Real.cos α - 4 * Real.sin α = 0 →
  Real.tan (α + Real.pi / 4) = 7 :=
by
  sorry

end find_tan_sum_l719_719208


namespace parallel_lines_of_conditions_l719_719874

variables (Line Plane : Type)
variables (m n : Line) (α β : Plane)

-- Definitions and conditions
def different_lines (m n : Line) : Prop := m ≠ n
def different_planes (α β : Plane) : Prop := α ≠ β
def parallel_line_plane (m : Line) (β : Plane) : Prop := sorry -- define parallel relation
def line_in_plane (m : Line) (α : Plane) : Prop := sorry -- define inclusion
def intersection_plane (α β : Plane) : Line := sorry -- define intersection
def parallel_lines (m n : Line) : Prop := sorry -- define parallel relation

-- Conditions
axiom h1 : different_lines m n
axiom h2 : different_planes α β
axiom h3 : parallel_line_plane m β
axiom h4 : line_in_plane m α
axiom h5 : intersection_plane α β = n

-- Goal
theorem parallel_lines_of_conditions : parallel_lines m n :=
sorry

end parallel_lines_of_conditions_l719_719874


namespace imaginary_part_of_z_is_neg_one_l719_719308

-- Define the complex number z
def z : ℂ := 2 - complex.i

-- The theorem stating that the imaginary part of z is -1
theorem imaginary_part_of_z_is_neg_one : complex.im z = -1 := by
  -- To be proved
  sorry

end imaginary_part_of_z_is_neg_one_l719_719308


namespace range_of_angle_of_inclination_of_tangent_line_l719_719314

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 2

theorem range_of_angle_of_inclination_of_tangent_line :
  let α (x : ℝ) := atan (fderiv ℝ f x) in
  ∀ x : ℝ, α(x) ∈ set.union (set.Ico 0 (π/2)) (set.Ico (3*π/4) π) :=
sorry

end range_of_angle_of_inclination_of_tangent_line_l719_719314


namespace win_sector_area_l719_719728

-- Given Conditions
def radius := 12  -- radius of the circle in cm
def probability_of_winning := 1 / 3  -- probability of winning on one spin

-- Calculate the total area of the circle
def total_area_circle : ℝ := real.pi * radius^2

-- Calculate the area of the WIN sector
def area_of_win_sector : ℝ := probability_of_winning * total_area_circle

-- Proof Statement
theorem win_sector_area : area_of_win_sector = 48 * real.pi :=
by
  -- proof steps will go here
  sorry

end win_sector_area_l719_719728


namespace greatest_integer_of_PS_is_282_l719_719234

noncomputable def sqrt_approx : ℝ := 1.414

def PQ : ℝ := 150
def QU : ℝ := 50
def QR : ℝ := PQ
def UR : ℝ := 100
def PS : ℝ := 200 * Real.sqrt 2
def greatest_integer_less_than_PS : ℕ := ⌊PS⌋

theorem greatest_integer_of_PS_is_282 :
  greatest_integer_less_than_PS = 282 :=
by
  have hPS : PS = 200 * Real.sqrt 2 := by sorry
  have h_approx : Real.sqrt 2 ≈ sqrt_approx := by sorry
  -- Proof that the floor of 200 * sqrt_approx equals 282
  sorry

end greatest_integer_of_PS_is_282_l719_719234


namespace non_zero_poly_has_non_zero_multiple_with_exponents_div_by_3_l719_719291

theorem non_zero_poly_has_non_zero_multiple_with_exponents_div_by_3 
  (P : Polynomial ℤ) (hP : P ≠ 0) : 
  ∃ Q : Polynomial ℤ, Q ≠ 0 ∧ (∀ n : ℕ, (coeff (P * Q) n ≠ 0 → n % 3 = 0)) :=
  sorry

end non_zero_poly_has_non_zero_multiple_with_exponents_div_by_3_l719_719291


namespace no_real_roots_l719_719702

-- Define the polynomial P(X) = X^5
def P (X : ℝ) : ℝ := X^5

-- Prove that for every α ∈ ℝ*, the polynomial P(X + α) - P(X) has no real roots
theorem no_real_roots (α : ℝ) (hα : α ≠ 0) : ∀ (X : ℝ), P (X + α) ≠ P X :=
by sorry

end no_real_roots_l719_719702


namespace corners_sum_eq_k_l719_719993

structure MagicSquare (n : ℕ) where
  (a : Fin n → Fin n → ℝ)
  (row_sum_eq_k : ∀ i, (Finset.univ.sum (λ j => a i j)) = k)
  (col_sum_eq_k : ∀ j, (Finset.univ.sum (λ i => a i j)) = k)
  (diag1_sum_eq_k : (Finset.univ.sum (λ i => a i i)) = k)
  (diag2_sum_eq_k : (Finset.univ.sum (λ i => a i (n - 1 - i))) = k)

variables {k : ℝ} {a : Fin 4 → Fin 4 → ℝ}

theorem corners_sum_eq_k (ms : MagicSquare 4 a k) :
  a (Fin.ofNat 0) (Fin.ofNat 0) + a (Fin.ofNat 0) (Fin.ofNat 3) +
  a (Fin.ofNat 3) (Fin.ofNat 0) + a (Fin.ofNat 3) (Fin.ofNat 3) = k :=
sorry

end corners_sum_eq_k_l719_719993


namespace problem_1_problem_2_l719_719187

variables {a : ℕ → ℕ} {S : ℕ → ℚ}

-- Condition: The sequence {a_n / n} is an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, n ≥ 1 → m ≥ 1 → (a n / n) - (a m / m) = k * (n - m)
-- Note: This condition is simplified for the purpose of statement. k is an arbitrary constant in ℕ.

-- Given conditions
def conditions (a : ℕ → ℕ) : Prop :=
a 3 - 3 * a 1 = 6 ∧ a 2 = 6

-- Question 1: Find a_n
def find_a_n (a : ℕ → ℕ) : Prop :=
a n = n * (n + 1)

-- Mathematical equivalence: Show that a_n = n(n + 1)
theorem problem_1 : conditions a → find_a_n a := sorry

-- Question 2: Find S_n given a_n
def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℚ) : Prop :=
S n = Σ (k : ℕ) in finset.range (n + 1), 1 / (a k)

-- Mathematical equivalence: Show that S_n = n / (n + 1)
theorem problem_2 (h : ∀ n, a n = n * (n + 1)) : sum_sequence a S → S n = n / (n + 1) := sorry

end problem_1_problem_2_l719_719187


namespace number_of_odd_handshakes_is_even_l719_719404

-- Define a participant who shakes hands
def Participant : Type := ℕ

-- Define what it means for a person to have an even or odd number of handshakes
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define a function that returns the number of handshakes for a participant (not explicitly needed for the proof but conceptual)
def handshake_count (p : Participant) : ℕ := sorry

-- Define the invariant that the number of people with odd handshakes is always even
theorem number_of_odd_handshakes_is_even (participants : List Participant)
  (handshakes : List (Participant × Participant)) :
  let odd_shake_count := list.countp (λ p => is_odd (handshake_count p)) participants
  in is_even odd_shake_count :=
by
  -- Assume that initially no handshakes imply no odd handshakes
  let initial_odd_shake_count := 0
  have h_initial : is_even initial_odd_shake_count := by simp [is_even]
  -- Apply the invariant through handshake events
  sorry

end number_of_odd_handshakes_is_even_l719_719404


namespace goldfish_count_l719_719676

noncomputable def goldfish_below_surface (total_surface_goldfish total_goldfish : ℕ) : ℕ :=
  total_goldfish - total_surface_goldfish

theorem goldfish_count {total_surface_goldfish total_goldfish : ℕ} 
  (h1 : total_surface_goldfish = 15)
  (h2 : total_surface_goldfish = (0.25 * total_goldfish).to_nat)
  : goldfish_below_surface total_surface_goldfish total_goldfish = 45 :=
by
  sorry

end goldfish_count_l719_719676


namespace projection_question_l719_719870

noncomputable def vector_projection (a b : ℝ^3) : ℝ^3 := 
  let dot_product (u v : ℝ^3) := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  (dot_product a b / dot_product b b) • b

theorem projection_question (a b : ℝ^3) (h₀ : b ≠ 0)
  (h₁ : dot_product b (a + b) = 0) : 
  vector_projection (a - b) b = -2 • b := sorry

end projection_question_l719_719870


namespace steve_berry_picking_strategy_l719_719995

def berry_picking_goal_reached (monday_earnings tuesday_earnings total_goal: ℕ) : Prop :=
  monday_earnings + tuesday_earnings >= total_goal

def optimal_thursday_strategy (remaining_goal payment_per_pound total_capacity : ℕ) : ℕ :=
  if remaining_goal = 0 then 0 else total_capacity

theorem steve_berry_picking_strategy :
  let monday_lingonberries := 8
  let monday_cloudberries := 10
  let monday_blueberries := 30 - monday_lingonberries - monday_cloudberries
  let tuesday_lingonberries := 3 * monday_lingonberries
  let tuesday_cloudberries := 2 * monday_cloudberries
  let tuesday_blueberries := 5
  let lingonberry_rate := 2
  let cloudberry_rate := 3
  let blueberry_rate := 5
  let max_capacity := 30
  let total_goal := 150

  let monday_earnings := (monday_lingonberries * lingonberry_rate) + 
                         (monday_cloudberries * cloudberry_rate) + 
                         (monday_blueberries * blueberry_rate)
                         
  let tuesday_earnings := (tuesday_lingonberries * lingonberry_rate) + 
                          (tuesday_cloudberries * cloudberry_rate) +
                          (tuesday_blueberries * blueberry_rate)

  let total_earnings := monday_earnings + tuesday_earnings

  berry_picking_goal_reached monday_earnings tuesday_earnings total_goal ∧
  optimal_thursday_strategy (total_goal - total_earnings) blueberry_rate max_capacity = 30 
:= by {
  sorry
}

end steve_berry_picking_strategy_l719_719995


namespace infinite_series_sum_l719_719442

theorem infinite_series_sum :
  (∑ k in Nat, (k^2 : ℝ)/(3^k : ℝ)) = 6 := sorry

end infinite_series_sum_l719_719442


namespace brownie_pan_dimensions_l719_719524

def brownie_dimensions (m n : ℕ) : Prop :=
  let numSectionsLength := m - 1
  let numSectionsWidth := n - 1
  let totalPieces := (numSectionsLength + 1) * (numSectionsWidth + 1)
  let interiorPieces := (numSectionsLength - 1) * (numSectionsWidth - 1)
  let perimeterPieces := totalPieces - interiorPieces
  (numSectionsLength = 3) ∧ (numSectionsWidth = 5) ∧ (interiorPieces = 2 * perimeterPieces)

theorem brownie_pan_dimensions :
  ∃ (m n : ℕ), brownie_dimensions m n ∧ m = 6 ∧ n = 12 :=
by
  existsi 6
  existsi 12
  unfold brownie_dimensions
  simp
  exact sorry

end brownie_pan_dimensions_l719_719524


namespace trip_movie_savings_l719_719332

def evening_ticket_cost : ℕ := 10
def combo_cost : ℕ := 10
def ticket_discount_percentage : ℕ := 20
def combo_discount_percentage : ℕ := 50

theorem trip_movie_savings :
  let ticket_saving := evening_ticket_cost * ticket_discount_percentage / 100,
      combo_saving := combo_cost * combo_discount_percentage / 100
  in ticket_saving + combo_saving = 7 :=
by
  sorry

end trip_movie_savings_l719_719332


namespace curve_eccentricity_l719_719363

  theorem curve_eccentricity (x y θ : ℝ)
    (h1 : x = 3 * real.sec θ)
    (h2 : y = 4 * real.tan θ) :
    ∃ e : ℝ, e = 5 / 3 :=
  sorry
  
end curve_eccentricity_l719_719363


namespace fg_neg1_eq_3_l719_719948

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := 2 * x^2 + 3

theorem fg_neg1_eq_3 : f(g(-1)) = 3 :=
by
  sorry

end fg_neg1_eq_3_l719_719948


namespace lucky_larry_l719_719610

theorem lucky_larry (a b c d e k : ℤ) 
    (h1 : a = 2) 
    (h2 : b = 3) 
    (h3 : c = 4) 
    (h4 : d = 5)
    (h5 : a - b - c - d + e = 2 - (b - (c - (d + e)))) 
    (h6 : k * 2 = e) : 
    k = 2 := by
  sorry

end lucky_larry_l719_719610


namespace emails_in_inbox_l719_719633

theorem emails_in_inbox :
  let total_emails := 400
  let trash_emails := total_emails / 2
  let work_emails := 0.4 * (total_emails - trash_emails)
  total_emails - trash_emails - work_emails = 120 :=
by
  sorry

end emails_in_inbox_l719_719633


namespace seq_div_by_11_start_num_l719_719674

theorem seq_div_by_11_start_num (a : ℕ → ℕ) : 
  (∀ n, (a n) % 11 = 0) → 
  a 7 = 79 →
  (∀ n, n ∈ finset.range 7) → 
  a 1 = 11 :=
  by 
    sorry

end seq_div_by_11_start_num_l719_719674


namespace binary_addition_to_decimal_l719_719342

theorem binary_addition_to_decimal : (2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^3 + 2^2 + 2^1 + 2^0)
                                     + (2^5 + 2^4 + 2^3 + 2^2) = 571 := by
  sorry

end binary_addition_to_decimal_l719_719342


namespace square_of_triangle_side_in_circle_l719_719370

theorem square_of_triangle_side_in_circle :
  ∃ (l : ℝ), (∀ (x y : ℝ), x^2 + y^2 = 16 ∧ (0, 4) ∧ altitude_on_y_axis → l^2 = 49) :=
sorry

end square_of_triangle_side_in_circle_l719_719370


namespace win_sector_area_l719_719733

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 12) (h_p : p = 1 / 3) :
  ∃ A : ℝ, A = 48 * π :=
by {
  sorry
}

end win_sector_area_l719_719733


namespace top_card_is_11_l719_719714

-- Define the initial configuration of cards
def initial_array : List (List Nat) := [
  [1, 2, 3, 4, 5, 6],
  [7, 8, 9, 10, 11, 12],
  [13, 14, 15, 16, 17, 18]
]

-- Perform the described sequence of folds
def fold1 (arr : List (List Nat)) : List (List Nat) := [
  [3, 4, 5, 6],
  [9, 10, 11, 12],
  [15, 16, 17, 18],
  [1, 2],
  [7, 8],
  [13, 14]
]

def fold2 (arr : List (List Nat)) : List (List Nat) := [
  [5, 6],
  [11, 12],
  [17, 18],
  [3, 4, 1, 2],
  [9, 10, 7, 8],
  [15, 16, 13, 14]
]

def fold3 (arr : List (List Nat)) : List (List Nat) := [
  [11, 12, 7, 8],
  [17, 18, 13, 14],
  [5, 6, 1, 2],
  [9, 10, 3, 4],
  [15, 16, 9, 10]
]

-- Define the final array after all the folds
def final_array := fold3 (fold2 (fold1 initial_array))

-- Statement to be proven
theorem top_card_is_11 : (final_array.head!.head!) = 11 := 
  by
    sorry -- Proof to be filled in

end top_card_is_11_l719_719714


namespace trip_movie_savings_l719_719334

def evening_ticket_cost : ℕ := 10
def combo_cost : ℕ := 10
def ticket_discount_percentage : ℕ := 20
def combo_discount_percentage : ℕ := 50

theorem trip_movie_savings :
  let ticket_saving := evening_ticket_cost * ticket_discount_percentage / 100,
      combo_saving := combo_cost * combo_discount_percentage / 100
  in ticket_saving + combo_saving = 7 :=
by
  sorry

end trip_movie_savings_l719_719334


namespace parity_f_range_m_range_n_l719_719246

-- a) Definitions from conditions
def f (x a : ℝ) : ℝ := (x + a) / (x - a)
def g (x m : ℝ) : ℝ := (x - m + 1) / (x - 2 * m)
def h (x : ℝ) : ℝ := 4 - 1 / x

-- b) Lean 4 Statements for the proof problems

-- 1. Proving the parity of function f(x)
theorem parity_f : 
  ∀ (a : ℝ), 
  (a = 0 → ∀ x, f x a = f (-x) a) ∧ 
  (a ≠ 0 → ¬ (∀ x, f x a = f (-x) a) ∧ ¬ (∀ x, f x a = -f (-x) a)) :=
by 
  intros
  split
  case left =>
    intros h x
    rw [h]
    sorry
  case right =>
    intros h
    split
    sorry
    sorry

-- 2. Finding range of m such that g(x) < 0 for a given interval
theorem range_m (m : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) → g x m < 0) → m ∈ set.Icc (1/4 : ℝ) (4/3 : ℝ) :=
by 
  intros
  sorry

-- 3. Finding range of n such that h maps an interval to another interval
theorem range_n (n : ℝ) : 
  (∃ λ μ, (λ > 1/3) ∧ (μ > 1/3) ∧ (∀ x, λ ≤ x ∧ x ≤ μ → h x ∈ set.Icc (n * λ) (n * μ))) →
  n ∈ set.Ioo (3 : ℝ) (4 : ℝ) :=
by 
  intros
  sorry

end parity_f_range_m_range_n_l719_719246


namespace mt_product_l719_719950

noncomputable def g (x : ℝ) : ℝ := sorry

theorem mt_product
  (hg : ∀ (x y : ℝ), g (g x + y) = g x + g (g y + g (-x)) - x) : 
  ∃ m t : ℝ, m = 1 ∧ t = -5 ∧ m * t = -5 := 
by
  sorry

end mt_product_l719_719950


namespace find_n_l719_719348

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end find_n_l719_719348


namespace david_course_hours_l719_719427

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end david_course_hours_l719_719427


namespace shaded_to_largest_ratio_l719_719240

theorem shaded_to_largest_ratio :
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let r4 := 4
  let area (r : ℝ) := π * r^2
  let largest_circle_area := area r4
  let innermost_shaded_area := area r1
  let outermost_shaded_area := area r3 - area r2
  let shaded_area := innermost_shaded_area + outermost_shaded_area
  shaded_area / largest_circle_area = 3 / 8 :=
by
  sorry

end shaded_to_largest_ratio_l719_719240


namespace translated_vector_coordinates_l719_719168

variables {A B a : Point}

def Point := (ℝ × ℝ)

def vector (P Q : Point) : Point := (Q.1 - P.1, Q.2 - P.2)
def translate (v a : Point) : Point := (v.1 + a.1, v.2 + a.2)

theorem translated_vector_coordinates : 
  let A := (3, 7)
  let B := (5, 2)
  let a := (1, 2)
  translate (vector A B) a = (2, -5)  :=
by
  sorry

end translated_vector_coordinates_l719_719168


namespace slope_of_line_l719_719536

theorem slope_of_line (θ : ℝ) (sin_cos_sum : sin θ + cos θ = sqrt 5 / 5) : 
  tan θ = -2 := by
  sorry

end slope_of_line_l719_719536


namespace sum_k_squared_div_3_k_l719_719450

theorem sum_k_squared_div_3_k : ∑ k in (Finset.range n).map (λ x, x+1), (k^2 / (3^k : ℝ)) = 4 :=
by
  sorry

end sum_k_squared_div_3_k_l719_719450


namespace find_m_l719_719504
open Nat

theorem find_m (m : ℕ) (hm : m > 0) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 24 := 
by
  sorry

end find_m_l719_719504


namespace number_of_sequences_l719_719736

-- Define the number of possible outcomes for a single coin flip
def coinFlipOutcomes : ℕ := 2

-- Define the number of flips
def numberOfFlips : ℕ := 8

-- Theorem statement: The number of distinct sequences when flipping a coin eight times is 256
theorem number_of_sequences (n : ℕ) (outcomes : ℕ) (h : outcomes = 2) (hn : n = 8) : outcomes ^ n = 256 := by
  sorry

end number_of_sequences_l719_719736


namespace no_p_q_for_quadratic_8_distinct_real_roots_arith_seq_l719_719828

theorem no_p_q_for_quadratic_8_distinct_real_roots_arith_seq :
  ∀ (f : ℝ → ℝ), (∃ (p q : ℝ), f = λ x, x^2 + p * x + q) →
  ¬ (∃ (roots : Fin 8 → ℝ), (∀ i j : Fin 8, i ≠ j → roots i ≠ roots j) ∧
      ∃ (d : ℝ), ∀ i : Fin 7, roots (i + 1) = roots i + d ∧
      ∀ x : ℝ, f (f (f x)) = 0 ↔ ∃ i : Fin 8, x = roots i) := 
sorry

end no_p_q_for_quadratic_8_distinct_real_roots_arith_seq_l719_719828


namespace infinite_sum_problem_l719_719810

theorem infinite_sum_problem : 
  (∑ k in (set.Ici 1), (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 1 :=
by
  sorry

end infinite_sum_problem_l719_719810


namespace tangent_circle_equations_l719_719012

theorem tangent_circle_equations
  (a b r : ℝ)
  (h1 : (a - 1)^2 + (b - sqrt 3)^2 = r^2 ∨ (a + 1)^2 + (b + sqrt 3)^2 = r^2 ∨ 
        (a - (2 * sqrt 3 + 3))^2 + (b + 2 + sqrt 3)^2 = 21 + 12 * sqrt 3 ∨ 
        (a + (2 * sqrt 3 + 3))^2 + (b - 2 - sqrt 3)^2 = 21 + 12 * sqrt 3)
  (h2 : ∃ b, ∃ r, ∃ a, ((x - a)^2 + (y - b)^2 = r^2 ∧
         sqrt (a^2 + b^2) = 1 + r ∧
         (b = sqrt 3 * a ∨ b = - (sqrt 3 / 3) * a))) :
  r = 1 ∨ r = 3 + 2 * sqrt 3 :=
sorry

end tangent_circle_equations_l719_719012


namespace find_f6_l719_719636

theorem find_f6 (f : ℝ → ℝ) (hf_lin : ∀ x y, f (x + y) = f x + f y)
  (hf_eq : ∀ x, f x = 3 * (classical.some (function.has_inverse f) x) + 9)
  (hf_val : f 3 = 6) : f 6 = 10.5 * (real.sqrt 3) - 4.5 :=
sorry

end find_f6_l719_719636


namespace john_books_sold_on_monday_l719_719577

variable (M : ℤ)
variable (books_stock : ℤ := 1300)
variable (sold_tuesday : ℤ := 50)
variable (sold_wednesday : ℤ := 64)
variable (sold_thursday : ℤ := 78)
variable (sold_friday : ℤ := 135)
variable (unsold_percentage : ℝ := 69.07692307692308)

theorem john_books_sold_on_monday :
  ∃ M, (M : ℝ) = 402 - (sold_tuesday + sold_wednesday + sold_thursday + sold_friday) ∧ 
  M = 75 :=
by
  let total_books_sold_tuesday_to_friday := sold_tuesday + sold_wednesday + sold_thursday + sold_friday
  let sold_percentage := 100 - unsold_percentage
  let total_books_sold := (sold_percentage / 100) * (books_stock : ℝ)
  use M
  calc
    M = total_books_sold - (total_books_sold_tuesday_to_friday : ℝ) : sorry
    M = 75 : sorry

sorry

end john_books_sold_on_monday_l719_719577


namespace order_of_a_b_c_l719_719271

noncomputable def a : ℝ := (5 / 7)^(-5 / 7)
noncomputable def b : ℝ := (7 / 5)^(3 / 5)
noncomputable def c : ℝ := Real.logBase 3 (14 / 5)

theorem order_of_a_b_c : c < b ∧ b < a := by
  sorry

end order_of_a_b_c_l719_719271


namespace arbitrary_large_sum_of_digits_l719_719598

noncomputable def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem arbitrary_large_sum_of_digits (a : Nat) (h1 : 2 ≤ a) (h2 : ¬ (2 ∣ a)) (h3 : ¬ (5 ∣ a)) :
  ∃ m : Nat, sum_of_digits (a^m) > m :=
by
  sorry

end arbitrary_large_sum_of_digits_l719_719598


namespace Andrew_runs_2_miles_each_day_l719_719977

theorem Andrew_runs_2_miles_each_day
  (A : ℕ)
  (Peter_runs : ℕ := A + 3)
  (total_miles_after_5_days : 5 * (A + Peter_runs) = 35) :
  A = 2 :=
by
  sorry

end Andrew_runs_2_miles_each_day_l719_719977


namespace smallest_lambda_l719_719706

theorem smallest_lambda (n : ℕ) (hn : 2 ≤ n) :
  ∃ λ : ℝ, λ = n / (n + 1) ∧ 
  (∀ (x : Fin n → ℝ) (hx : ∀ k, 0 ≤ x k ∧ x k ≤ 1),
    ∃ (ε : Fin n → ℕ) (hε : ∀ k, ε k = 0 ∨ ε k = 1),
      ∀ (i j : Fin n), i <= j → abs (∑ k in Ico i j.succ, (ε k : ℝ) - x k) ≤ λ) :=
by
  use n / (n + 1)
  split
  · exact rfl
  sorry

end smallest_lambda_l719_719706


namespace integer_pairs_count_l719_719122

theorem integer_pairs_count :
  (∃ (m n : ℤ), m * n ≤ 0 ∧ m^3 + n^3 + 27 * m * n = 27^3) → 55 :=
by {
  -- solving or handling the proof will go here
  sorry
}

end integer_pairs_count_l719_719122


namespace triangle_PQS_isosceles_PQ_div_QR_eq_ST_div_PQ_l719_719679

variables {A B C O P S T Q R : Type*} [EuclideanGeometry ABC O]
noncomputable theory

-- Given conditions
variables (h1 : inscribed_circle ABC O)
variables (h2 : P_on_arc_AB_not_containing_C P A B C)
variables (h3 : perpendicular_from_P_on_BO PSU O intersects_side_AB_at_S_and_side_BC_at_T S T)
variables (h4 : perpendicular_from_P_on_AO Q PU O intersects_side_AB_at_Q_and_side_AC_at_R Q R)

-- Statement 1: Prove that triangle PQS is isosceles
theorem triangle_PQS_isosceles : is_isosceles_triangle P Q S :=
sorry

-- Statement 2: Prove that PQ/QR = ST/PQ
theorem PQ_div_QR_eq_ST_div_PQ : (PQ / QR) = (ST / PQ) :=
sorry

end triangle_PQS_isosceles_PQ_div_QR_eq_ST_div_PQ_l719_719679


namespace transformed_average_l719_719490

theorem transformed_average (n : ℕ) (original_average factor : ℝ) 
  (h1 : n = 15) (h2 : original_average = 21.5) (h3 : factor = 7) :
  (original_average * factor) = 150.5 :=
by
  sorry

end transformed_average_l719_719490


namespace distance_AB_is_2sqrt3_over_3_l719_719489

-- Define the conditions
variables {P A B C : Point}
variable (O : Point)
variable (radius : ℝ)
variable (PA PB PC : ℝ)
variable (h : ℝ)

-- Given conditions
axiom cone : right_circular_cone P A B C
axiom on_sphere_A : distance O A = radius
axiom on_sphere_C : distance O C = radius
axiom mutually_perpendicular : perpendicular A PB
axiom sphere_radius : radius = sqrt 3

-- Define the experiment as a mathematical statement that needs to be proven
theorem distance_AB_is_2sqrt3_over_3 :
  distance A B = 2 * sqrt 3 / 3 :=
sorry

end distance_AB_is_2sqrt3_over_3_l719_719489


namespace moles_of_NH4Cl_l719_719121

-- Define what is meant by "mole" and the substances NH3, HCl, and NH4Cl
def NH3 : Type := ℕ -- Use ℕ to represent moles
def HCl : Type := ℕ
def NH4Cl : Type := ℕ

-- Define the stoichiometry of the reaction
def reaction (n_NH3 n_HCl : ℕ) : ℕ :=
n_NH3 + n_HCl

-- Lean 4 statement: given 1 mole of NH3 and 1 mole of HCl, prove the reaction produces 1 mole of NH4Cl
theorem moles_of_NH4Cl (n_NH3 n_HCl : ℕ) (h1 : n_NH3 = 1) (h2 : n_HCl = 1) : 
  reaction n_NH3 n_HCl = 1 :=
by
  sorry

end moles_of_NH4Cl_l719_719121


namespace cubs_more_home_runs_l719_719078

noncomputable def cubs_home_runs := 2 + 1 + 2
noncomputable def cardinals_home_runs := 1 + 1

theorem cubs_more_home_runs :
  cubs_home_runs - cardinals_home_runs = 3 :=
by
  -- Proof would go here, but we are using sorry to skip it
  sorry

end cubs_more_home_runs_l719_719078


namespace marcus_mileage_l719_719406

theorem marcus_mileage (initial_mileage : ℕ) (miles_per_gallon : ℕ) (gallons_per_tank : ℕ) (tanks_used : ℕ) :
  initial_mileage = 1728 →
  miles_per_gallon = 30 →
  gallons_per_tank = 20 →
  tanks_used = 2 →
  initial_mileage + (miles_per_gallon * gallons_per_tank * tanks_used) = 2928 :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  exact rfl,
}

end marcus_mileage_l719_719406


namespace favorite_number_l719_719624

theorem favorite_number (S₁ S₂ S₃ : ℕ) (total_sum : ℕ) (adjacent_sum : ℕ) 
  (h₁ : S₁ = 8) (h₂ : S₂ = 14) (h₃ : S₃ = 12) 
  (h_total_sum : total_sum = 17) 
  (h_adjacent_sum : adjacent_sum = 12) : 
  ∃ x : ℕ, x = 5 := 
by 
  sorry

end favorite_number_l719_719624


namespace percentage_of_x_l719_719723

theorem percentage_of_x (x P: ℝ) (h1 : x = 1649.999999999999) (h2 : (P / 100) * x = (1 / 3) * x + 110) :
  abs (P - 40) < 1 :=
by
  subst h1
  subst h2
  sorry

end percentage_of_x_l719_719723


namespace sides_of_figures_intersection_l719_719979

theorem sides_of_figures_intersection (n p q : ℕ) (h1 : p ≠ 0) (h2 : q ≠ 0) :
  p + q ≤ n + 4 :=
by sorry

end sides_of_figures_intersection_l719_719979


namespace exists_fixed_point_of_function_l719_719546

theorem exists_fixed_point_of_function (a : ℝ) (h0 : 0 < a) (h1 : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (0, 1 / 2) ∧ ∀ x : ℝ, ((λ x, a^x - 1 / 2) 0) = 1 / 2 :=
by
  use (0, 1 / 2)
  split
  · refl
  · intro x
    have h : (λ x, a^x - 1 / 2) 0 = 1 / 2 := by
      simp
      assumption
    exact h
  done

end exists_fixed_point_of_function_l719_719546


namespace max_possible_score_l719_719812

-- Define the conditions of the disks
structure Disk (n : ℕ) :=
  (center : Nat)
  (radius : Nat)

-- Predicate to express the condition that a disk properly contains another disk
def properly_contains (C_i C_j : Disk) : Prop :=
  C_i.radius > C_j.radius

-- Define the score of an arrangement of disks 
def score (disks : List (Disk n)) : Nat :=
  List.sum (List.map (λ i, List.length (List.filter (λ j, properly_contains (disks.get i) (disks.get j)) disks)) (List.range (List.length disks)))

-- Theorem to state the maximum possible score
theorem max_possible_score (disks : List (Disk n)) : score disks ≤ (n-1)*(n-2)/2 := 
  sorry

end max_possible_score_l719_719812


namespace value_of_expression_l719_719703

theorem value_of_expression (x y : ℤ) (h1 : x = 7) (h2 : y = -2) :
  (x - 2 * y) ^ y = 1 / 121 := by
  rw [h1, h2]
  simp
  sorry

end value_of_expression_l719_719703


namespace max_red_squares_l719_719416

theorem max_red_squares (grid_5x5 : matrix (fin 5) (fin 5) ℕ)
    (h_no_rectangle : ∀ (i1 i2 i3 i4 j1 j2 j3 j4 : fin 5),
        i1 ≠ i2 → i1 ≠ i3 → i1 ≠ i4 → i2 ≠ i3 → i2 ≠ i4 → i3 ≠ i4 →
        j1 ≠ j2 → j1 ≠ j3 → j1 ≠ j4 → j2 ≠ j3 → j2 ≠ j4 → j3 ≠ j4 →
        grid_5x5 i1 j1 = 1 → grid_5x5 i2 j1 = 1 → grid_5x5 i1 j2 = 1 → grid_5x5 i2 j2 = 1 →
        grid_5x5 i3 j3 = 1 → grid_5x5 i4 j3 = 1 → grid_5x5 i3 j4 = 1 → grid_5x5 i4 j4 = 1 →
        false) :
    ∃ n : ℕ, n = 12 ∧ ∑ i j, grid_5x5 i j = n :=
begin
    sorry
end

end max_red_squares_l719_719416


namespace A_finishes_remaining_work_in_5_days_l719_719368

open Nat

theorem A_finishes_remaining_work_in_5_days
  (A_work_days : ℕ)
  (B_work_days : ℕ)
  (B_worked_days : ℕ)
  (A_work_rate : ℚ := 1 / A_work_days)
  (B_work_rate : ℚ := 1 / B_work_days)
  (B_completed_work : ℚ := B_work_rate * B_worked_days)
  (remaining_work : ℚ := 1 - B_completed_work):

  A_work_days = 15 ->
  B_work_days = 15 ->
  B_worked_days = 10 ->
  remaining_work / A_work_rate = 5 :=
begin
  intros hA hB hW,
  rw [hA, hB] at *,
  have work_done_B : B_work_rate * B_worked_days = 2 / 3, by {
    dsimp [B_work_rate, B_worked_days],
    rw [←div_mul_eq_mul_div, nat.cast_mul, div_eq_mul_one_div, nat.cast_bit1, nat.cast_one, mul_comm],
    simp only [mul_one_div_cancel,  mul_comm, not_le.mpr zero_lt_bit0 (one_ne_zero), one_ne_zero,
  },
  rw [remaining_work, work_done_B],
  dsimp [A_work_rate],
  field_simp [A_work_days.ne_zero],
  norm_num,
end

end A_finishes_remaining_work_in_5_days_l719_719368


namespace total_pressure_on_sphere_l719_719843

-- Question: Define the mathematical objects and problem statement.

-- Conditions
def diameter : ℝ := 4
def center_depth : ℝ := 3
def radius : ℝ := diameter / 2
def ρ : ℝ -- Density of water (constant)
def g : ℝ -- Acceleration due to gravity (constant)

-- Hypotheses
axiom density_nonnegative : ρ ≥ 0
axiom gravity_nonnegative : g ≥ 0

-- Define the pressure integral over the sphere surface as per conditions
noncomputable def pressure_integral : ℝ := 4 * π * ∫ (x : ℝ) in -radius..radius, (3 + x) dx

-- Proving the total pressure on the surface of the sphere is 64π
theorem total_pressure_on_sphere : pressure_integral = 64 * π :=
by
  sorry -- The proof is omitted

end total_pressure_on_sphere_l719_719843


namespace correct_propositions_l719_719860

variable {α β : Type}
variable {f : ℝ → ℝ}

-- Condition 1
def condition1 (f : ℝ → ℝ) : Prop := ∀ (x : ℝ), f(x) = f(-x)

-- Condition 2
def condition2 (f : ℝ → ℝ) : Prop := ∀ (x : ℝ), f(x + 2) = f(2 - x)

-- Condition 3
def condition3 (f : ℝ → ℝ) : Prop := ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 2 → f(x₁) ≤ f(x₂)

-- Proposition (II): y = f(x) is a periodic function with period 4
def prop2 (f : ℝ → ℝ) : Prop := ∀ (x : ℝ), f(x + 4) = f(x)

-- Proposition (III): y = f(x) is a decreasing function on the interval [2, 4]
def prop3 (f : ℝ → ℝ) : Prop := ∀ (x₁ x₂ : ℝ), 2 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 4 → f(x₁) ≥ f(x₂)

theorem correct_propositions (f : ℝ → ℝ) :
  (condition1 f) → (condition2 f) → (condition3 f) → (prop2 f ∧ prop3 f) :=
by
  intros h1 h2 h3
  sorry

end correct_propositions_l719_719860


namespace find_valid_n_l719_719254

noncomputable theory

def valid_permutation (σ : fin n → fin n) :=
  (∀ i, i ∈ {0, 1, ..., n - 1} ∧ ∃ j, j < n ∧ ∀ k, σ k = k → σ k = n - 1 - k)

theorem find_valid_n (n : ℕ) (hn : 0 < n) :
  (∃ σ : (fin n) → (fin n), valid_permutation σ ∧ (finset.univ.image (λ k, abs (σ (k : fin n).val - k.val))).card = n) ↔
  (n % 4 = 0 ∨ n % 4 = 1) :=
begin
  sorry
end

end find_valid_n_l719_719254


namespace min_side_b_of_triangle_l719_719922

theorem min_side_b_of_triangle (A B C a b c : ℝ) 
  (h_arith_seq : 2 * B = A + C)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides_opposite : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h_given_eq : 3 * a * c + b^2 = 25) :
  b ≥ 5 / 2 :=
  sorry

end min_side_b_of_triangle_l719_719922


namespace initial_ants_l719_719555

theorem initial_ants (A : ℕ) (h : ∀ t : ℕ, (2 ^ t) * A) :
  (let num_ants := 1600 / (2 ^ 5) in num_ants = 50) :=
by
  noncomputable def initial_ants := (1600 : ℕ) / (2 ^ 5)
  have h : initial_ants = 50
  sorry

end initial_ants_l719_719555


namespace width_of_larger_cuboid_l719_719209

def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

theorem width_of_larger_cuboid :
  ∃ W : ℝ, 
    let Volume_small := volume 5 6 3 in
    let Total_volume_small := 6 * Volume_small in
    let Volume_large := volume 18 W 2 in
    Total_volume_small = Volume_large →
    W = 15 :=
by
  let Volume_small := volume 5 6 3
  let Total_volume_small := 6 * Volume_small
  let Volume_large := volume 18 15 2
  have : Total_volume_small = Volume_large 
  { sorry }
  use 15
  split
  { exact this }
  sorry

end width_of_larger_cuboid_l719_719209


namespace problem_1_problem_2_l719_719188

variables {a : ℕ → ℕ} {S : ℕ → ℚ}

-- Condition: The sequence {a_n / n} is an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, n ≥ 1 → m ≥ 1 → (a n / n) - (a m / m) = k * (n - m)
-- Note: This condition is simplified for the purpose of statement. k is an arbitrary constant in ℕ.

-- Given conditions
def conditions (a : ℕ → ℕ) : Prop :=
a 3 - 3 * a 1 = 6 ∧ a 2 = 6

-- Question 1: Find a_n
def find_a_n (a : ℕ → ℕ) : Prop :=
a n = n * (n + 1)

-- Mathematical equivalence: Show that a_n = n(n + 1)
theorem problem_1 : conditions a → find_a_n a := sorry

-- Question 2: Find S_n given a_n
def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℚ) : Prop :=
S n = Σ (k : ℕ) in finset.range (n + 1), 1 / (a k)

-- Mathematical equivalence: Show that S_n = n / (n + 1)
theorem problem_2 (h : ∀ n, a n = n * (n + 1)) : sum_sequence a S → S n = n / (n + 1) := sorry

end problem_1_problem_2_l719_719188


namespace tetrahedron_circumscribed_sphere_surface_area_l719_719864

open Real

-- Define the problem setup
def tetrahedron_SABC (A B C S : Point3d) : Prop :=
  (right_triangle A B C ∧ (dist A B = 8)) ∧ (dist S C = 6) ∧ (perpendicular S C (plane A B C))

-- Define the result we want to prove
def surface_area_of_circumscribed_sphere (surface_area: ℝ) : Prop :=
  surface_area = 100 * π

-- The main theorem statement
theorem tetrahedron_circumscribed_sphere_surface_area 
  (A B C S : Point3d)
  (h : tetrahedron_SABC A B C S) :
  ∃ surface_area, surface_area_of_circumscribed_sphere surface_area :=
by
  sorry

end tetrahedron_circumscribed_sphere_surface_area_l719_719864


namespace percentage_of_men_is_55_l719_719222

-- Define the percentage of men among all employees
def percent_of_men (M : ℝ) := M

-- Define the percentage of women among all employees
def percent_of_women (M : ℝ) := 1 - M

-- Define the contribution to picnic attendance by men
def attendance_by_men (M : ℝ) := 0.20 * M

-- Define the contribution to picnic attendance by women
def attendance_by_women (M : ℝ) := 0.40 * (percent_of_women M)

-- Define the total attendance
def total_attendance (M : ℝ) := attendance_by_men M + attendance_by_women M

theorem percentage_of_men_is_55 : ∀ M : ℝ, total_attendance M = 0.29 → M = 0.55 :=
by
  intro M
  intro h
  sorry

end percentage_of_men_is_55_l719_719222


namespace geometric_sequence_general_term_and_sum_l719_719568

theorem geometric_sequence_general_term_and_sum :
  ∃ a_n : ℕ → ℝ, ∃ b_n : ℕ → ℝ, ∃ c_n : ℕ → ℝ, ∃ T_n : ℕ → ℝ,
  (∀ q > 1, (a_n 2 = 2) → (a_n 3 = 7) → a_n 3 = a_n 1 * q + a_n 1 * q^2 + a_n 1 * q^3) ∧
  (b_n n = log 2 (a_n n)) ∧
  (c_n n = 1 / (b_n (n + 1) * b_n (n + 2))) ∧
  (T_n n = sum (range (n + 1)) c_n = n / (n + 1)) :=
sorry

end geometric_sequence_general_term_and_sum_l719_719568


namespace max_value_f_l719_719649

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_f :
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ 1 :=
sorry

end max_value_f_l719_719649


namespace lottery_problem_l719_719210

theorem lottery_problem (x : ℕ) (h : x^2 - x - 1892 = 0) : x = 44 :=
by 
  have h_solution : x = 44 ∨ x = -43 :=
    by sorry
  cases h_solution with h_pos h_neg
  · exact h_pos
  · linarith

end lottery_problem_l719_719210


namespace candles_extinguished_at_3_75_l719_719073

def initial_height (h : ℝ) : Prop := h > 0

def burn_rate_first (h : ℝ) : ℝ := h / 5

def burn_rate_second (h : ℝ) : ℝ := h / 4

def remaining_height_first (h t : ℝ) : ℝ := h * (1 - t / 5)

def remaining_height_second (h t : ℝ) : ℝ := h * (1 - t / 4)

def burned_out_at_same_time (h t : ℝ) : Prop :=
  remaining_height_first h t = 4 * remaining_height_second h t

theorem candles_extinguished_at_3_75 (h : ℝ) (t : ℝ) (Hh : initial_height h) : 
  burned_out_at_same_time h t ↔ t = 15 / 4 :=
sorry

end candles_extinguished_at_3_75_l719_719073


namespace cyclic_quadrilaterals_l719_719595

-- Definitions
variables {α : Type*} [euclidean_space α]

-- Given a triangle ABC
variable (A B C : α)

-- M is the midpoint of [BC]
variables (M : α) (hM : midpoint B C M)

-- D is a point on the side [AC]
variables (D : α) (hd : segment A C D)

-- E is a point on the side [AB]
variables (E : α) (he : segment A B E)

-- MC = MB = ME = MD
variables (h_dist : dist M C = dist M B ∧ dist M B = dist M E ∧ dist M E = dist M D)

-- The angle bisector from A intersects (BC) at F
variables (F : α) (hF : on_angle_bisector A B C F)

-- The bisector of ∠EMD intersects (AF) at X
variables (X : α) (hX : on_angle_bisector E M D X ∧ segment (A F) X)

-- The proof statement that the quadrilaterals BEXF and DXFC are cyclic
theorem cyclic_quadrilaterals (A B C M D E F X : α)
  (hM : midpoint B C M) (hd : segment A C D) (he : segment A B E)
  (h_dist : dist M C = dist M B ∧ dist M B = dist M E ∧ dist M E = dist M D)
  (hF : on_angle_bisector A B C F) (hX : on_angle_bisector E M D X ∧ segment (A F) X) :
  cyclic_quadrilateral B E X F ∧ cyclic_quadrilateral D X F C := 
sorry

end cyclic_quadrilaterals_l719_719595


namespace exp_inequality_log_condition_l719_719177

theorem exp_inequality_log_condition (a b : ℝ) : (2^a > 2^b) ↔ (a > b) ∧ (¬(a < 0) ∨ ¬(b < 0)) := sorry

end exp_inequality_log_condition_l719_719177


namespace length_of_second_train_l719_719341

-- Define the given conditions
def speed1_kmph : ℝ := 60
def speed2_kmph : ℝ := 40
def time_seconds : ℝ := 10.439164866810657
def length1_m : ℝ := 140

-- Convert speeds from km/hr to m/s
def speed1_mps : ℝ := speed1_kmph * (5.0 / 18.0)
def speed2_mps : ℝ := speed2_kmph * (5.0 / 18.0)

-- Calculate relative speed
def relative_speed_mps : ℝ := speed1_mps + speed2_mps

-- Define the expected length of the second train
def expected_length2_m : ℝ := 150

-- Define the length of the second train
def length2_m : ℝ := relative_speed_mps * time_seconds - length1_m

-- The theorem we need to prove
theorem length_of_second_train :
  length2_m = expected_length2_m :=
by
  -- Skipping proof steps
  sorry

end length_of_second_train_l719_719341


namespace tomatoes_ready_for_sale_l719_719964

-- Define all conditions
def initial_shipment := 1000 -- kg of tomatoes on Friday
def sold_on_saturday := 300 -- kg of tomatoes sold on Saturday
def rotten_on_sunday := 200 -- kg of tomatoes rotted on Sunday
def additional_shipment := 2 * initial_shipment -- kg of tomatoes arrived on Monday

-- Define the final calculation to prove
theorem tomatoes_ready_for_sale : 
  initial_shipment - sold_on_saturday - rotten_on_sunday + additional_shipment = 2500 := 
by
  sorry

end tomatoes_ready_for_sale_l719_719964


namespace polynomial_factors_l719_719550

theorem polynomial_factors (h k : ℤ)
  (h1 : 3 * (-2)^4 - 2 * h * (-2)^2 + h * (-2) + k = 0)
  (h2 : 3 * 1^4 - 2 * h * 1^2 + h * 1 + k = 0)
  (h3 : 3 * (-3)^4 - 2 * h * (-3)^2 + h * (-3) + k = 0) :
  |3 * h - 2 * k| = 11 :=
by
  sorry

end polynomial_factors_l719_719550


namespace intersect_BB1_DE_on_ω_l719_719594

variables {A B C C1 B1 A1 D E : Type} 
variable [ConvexHexagon A B C C1 B1 A1] -- Custom structure for convex hexagon
variable [Circle ω A B C] -- Circle passing through points A, B, and C
variables [EqAEqualsBAndBCEqual A B C]
variable [PerpendicularBisectorEqual AA1 BB1 CC1]
variable [DiagonalsMeet AC1 A1C D]
variable [CircleIntersection ω A1 B C1 E]

theorem intersect_BB1_DE_on_ω : ∃ P : Type, P ∈ ω ∧ P ∈ (BB1 ∩ DE) :=
sorry

end intersect_BB1_DE_on_ω_l719_719594


namespace check_correct_proposition_l719_719090

structure Vector (ℝ : Type) :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def unit_vector {ℝ : Type} [NormedField ℝ] (v : Vector ℝ) : Prop := 
  ∥v∥ = 1

def collinear {ℝ : Type} (v w : Vector ℝ) : Prop :=
  ∃ α : ℝ, v.x = α * w.x ∧ v.y = α * w.y ∧ v.z = α * w.z

def zero_vector {ℝ : Type} : Vector ℝ := 
  ⟨0, 0, 0⟩

def equal_vectors {ℝ : Type} (v w : Vector ℝ) : Prop :=
  v.x = w.x ∧ v.y = w.y ∧ v.z = w.z

theorem check_correct_proposition (μ : Type) [NormedField μ] 
(v w : Vector μ) :
  ¬ (unit_vector v ∧ unit_vector w ∧ equal_vectors v w) ∧ 
  ((¬ collinear v w) → (v ≠ zero_vector ∧ w ≠ zero_vector)) ∧ 
  (zero_vector = ⟨0, 0, 0⟩) ∧ 
  ¬ (equal_vectors v w → (v.x ≠ 0 ∨ v.y ≠ 0 ∨ v.z ≠ 0)) :=
  sorry

end check_correct_proposition_l719_719090


namespace tim_picks_matching_pair_probability_l719_719106

def socks_probability :=
  let total_socks := 18
  let gray_socks := 10
  let white_socks := 8
  let total_ways := Nat.choose total_socks 2
  let gray_pair := Nat.choose gray_socks 2
  let white_pair := Nat.choose white_socks 2
  let matching_ways := gray_pair + white_pair
  (matching_ways : ℚ) / (total_ways : ℚ)

theorem tim_picks_matching_pair_probability :
  socks_probability = (73 : ℚ) / 153 :=
by {
  sorry
}

end tim_picks_matching_pair_probability_l719_719106


namespace sum_arithmetic_series_l719_719467

theorem sum_arithmetic_series :
  let a := -42
  let d := 2
  let l := 0
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = -462 := by
sorry

end sum_arithmetic_series_l719_719467


namespace final_water_percentage_l719_719281

theorem final_water_percentage (V_initial W_percentage V_pure : ℕ) (H1 : V_initial = 10)
  (H2 : W_percentage = 5) (H3 : V_pure = 15) :
  let W_initial := (W_percentage / 100) * V_initial in
  let V_final := V_initial + V_pure in
  let final_percentage := (W_initial / V_final : ℝ) * 100 in
  final_percentage = 2 := 
  by
    sorry

end final_water_percentage_l719_719281


namespace sum_a1_to_a14_equals_zero_l719_719882

theorem sum_a1_to_a14_equals_zero 
  (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 : ℝ) 
  (h1 : (1 + x - x^2)^3 * (1 - 2 * x^2)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 + a13 * x^13 + a14 * x^14) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 1) 
  (h3 : a = 1) : 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 = 0 := by
  sorry

end sum_a1_to_a14_equals_zero_l719_719882


namespace cube_root_5488000_eq_40_l719_719811

noncomputable def cube_root (x : ℕ) : ℕ := x^(1/3)

theorem cube_root_5488000_eq_40 (x : ℕ) (h₁ : 5488000 = x^3) : x = 40 :=
by {
  have h₂ : 5488000 = 40^3,
  {
    calc
      40^3 = (20 × 2)^3 : by norm_num
      ... = 20^3 × 2^3 : by norm_num
      ... = 8000 × 8 : by { norm_num, norm_cast, exact pow_mul' 20 2 2 }
      ... = 64000 : by norm_num,
  },
  have h₃ : 5488000 = 64000,
  {
    exact h₁,
  },
  exact eq_of_pow_eq_pow x 40 3 h₁,
}

end cube_root_5488000_eq_40_l719_719811


namespace max_area_triangle_PMN_l719_719563

noncomputable def curve_C (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 0
noncomputable def line_l (ρ : ℝ) (θ : ℝ) : Prop := θ = π / 4
noncomputable def ellipse_P (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

theorem max_area_triangle_PMN :
  ∃ (P : ℝ × ℝ) (M N : ℝ × ℝ),
  ellipse_P P.1 P.2 ∧
  curve_C M.1 M.2 ∧
  curve_C N.1 N.2 ∧
  line_l M.1 M.2 ∧
  line_l N.1 N.2 ∧
  let area := (1 / 2) * (sqrt 2) * (abs (2 * (cos (P.2 + π/6)))) in
  area = 1 :=
sorry

end max_area_triangle_PMN_l719_719563


namespace angle_CDB_eq_30_l719_719015

-- Definitions of angles used
constant A B C D : Type
constant angle_deg : A → A → ℝ

axiom angle_CAD_eq_40 : angle_deg C A D = 40
axiom angle_DBA_eq_40 : angle_deg D B A = 40
axiom angle_CAB_eq_60 : angle_deg C A B = 60
axiom angle_CBD_eq_20 : angle_deg C B D = 20

-- Mathematical proof statement without the proof, only the statement and sorry.
theorem angle_CDB_eq_30 : angle_deg C D B = 30 :=
sorry

end angle_CDB_eq_30_l719_719015


namespace lois_final_books_l719_719961

-- Definitions for the conditions given in the problem.
def initial_books : ℕ := 40
def books_given_to_nephew (b : ℕ) : ℕ := b / 4
def books_remaining_after_giving (b_given : ℕ) (b : ℕ) : ℕ := b - b_given
def books_donated_to_library (b_remaining : ℕ) : ℕ := b_remaining / 3
def books_remaining_after_donating (b_donated : ℕ) (b_remaining : ℕ) : ℕ := b_remaining - b_donated
def books_purchased : ℕ := 3
def total_books (b_final_remaining : ℕ) (b_purchased : ℕ) : ℕ := b_final_remaining + b_purchased

-- Theorem stating: Given the initial conditions, Lois should have 23 books in the end.
theorem lois_final_books : 
  total_books 
    (books_remaining_after_donating (books_donated_to_library (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books)) 
    (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books))
    books_purchased = 23 :=
  by
    sorry  -- Proof omitted as per instructions.

end lois_final_books_l719_719961


namespace least_subtraction_l719_719351

theorem least_subtraction (n : ℕ) (h : n = 427751) : 
  ∃ k : ℕ, 210 ∣ (n - k) ∧ (∀ m : ℕ, 210 ∣ (n - m) → k ≤ m) ∧ k = 91 :=
by {
  -- state the variables and assumptions
  existsi 91,
  split,
  {
    -- verify the divisibility requirement
    sorry,
  },
  split,
  {
    -- verify the minimality requirement
    sorry,
  },
  {
    -- verify k = 91
    refl,
  }
}

end least_subtraction_l719_719351


namespace expectation_of_X_l719_719393

noncomputable def seed_germination_probability : ℝ := 0.9
def num_seeds_planted : ℕ := 1000
def additional_seeds_per_non_germinated : ℕ := 2

theorem expectation_of_X :
  let p := seed_germination_probability in
  let n := num_seeds_planted in
  let q := 1 - p in
  let ξ := n * q in
  let X := 2 * ξ in
  E X = 200 :=
by
  sorry

end expectation_of_X_l719_719393


namespace find_polynomials_l719_719831

-- Definition of polynomials in Lean
noncomputable def polynomials : Type := Polynomial ℝ

-- Main theorem statement
theorem find_polynomials : 
  ∀ p : polynomials, 
    (∀ x : ℝ, p.eval (5 * x) ^ 2 - 3 = p.eval (5 * x^2 + 1)) → 
    (p.eval 0 ≠ 0 → (∃ c : ℝ, (p = Polynomial.C c) ∧ (c = (1 + Real.sqrt 13) / 2 ∨ c = (1 - Real.sqrt 13) / 2))) ∧ 
    (p.eval 0 = 0 → ∀ x : ℝ, p.eval x = 0) :=
by
  sorry

end find_polynomials_l719_719831


namespace large_square_area_l719_719675

theorem large_square_area (a b c : ℕ) (h1 : 4 * a < b) (h2 : c^2 = a^2 + b^2 + 10) : c^2 = 36 :=
  sorry

end large_square_area_l719_719675


namespace mutually_exclusive_event_l719_719691

def Event := String  -- define a simple type for events

/-- Define the events -/
def at_most_one_hit : Event := "at most one hit"
def two_hits : Event := "two hits"

/-- Define a function to check mutual exclusiveness -/
def mutually_exclusive (e1 e2 : Event) : Prop := 
  e1 ≠ e2

theorem mutually_exclusive_event :
  mutually_exclusive at_most_one_hit two_hits :=
by
  sorry

end mutually_exclusive_event_l719_719691


namespace number_of_satisfying_sets_l719_719159

def set_of_integers := {1, 2, 3, 4, 5} : Set ℕ
def subset_condition (A : Set ℕ) : Prop := A ⊆ set_of_integers ∧ A ∩ {1, 2, 3} = {1, 2}

theorem number_of_satisfying_sets : 
  {A : Set ℕ | subset_condition A}.toFinset.card = 4 := 
by 
  sorry

end number_of_satisfying_sets_l719_719159


namespace k_inequality_l719_719004

noncomputable def k_value : ℝ :=
  5

theorem k_inequality (x : ℝ) :
  (x * (2 * x + 3) < k_value) ↔ (x > -5 / 2 ∧ x < 1) :=
sorry

end k_inequality_l719_719004


namespace cos_product_identity_l719_719126

theorem cos_product_identity :
  (Real.cos (20 * Real.pi / 180)) * (Real.cos (40 * Real.pi / 180)) *
  (Real.cos (60 * Real.pi / 180)) * (Real.cos (80 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end cos_product_identity_l719_719126


namespace sequence_geometric_l719_719201

theorem sequence_geometric (a : ℕ → ℕ) (n : ℕ) (hn : 0 < n):
  (a 1 = 1) →
  (∀ n, 0 < n → a (n + 1) = 2 * a n) →
  a n = 2^(n-1) :=
by
  intros
  sorry

end sequence_geometric_l719_719201


namespace width_of_beam_l719_719311

theorem width_of_beam (L W k : ℝ) (h1 : L = k * W) (h2 : 250 = k * 1.5) : 
  (k = 166.6667) → (583.3333 = 166.6667 * W) → W = 3.5 :=
by 
  intro hk1 
  intro h583
  sorry

end width_of_beam_l719_719311


namespace series_sum_l719_719449

-- Define the infinite series
def series := ∑' k : ℕ, (k^2 : ℝ) / 3^k

-- State the theorem to prove
theorem series_sum : series = 1 := 
sorry

end series_sum_l719_719449


namespace win_sector_area_l719_719726

-- Defining the conditions
def radius : ℝ := 12
def total_area : ℝ := π * radius^2
def win_probability : ℝ := 1 / 3

-- Theorem to prove the area of the WIN sector
theorem win_sector_area : total_area * win_probability = 48 * π := by
  sorry

end win_sector_area_l719_719726


namespace nth_equation_l719_719612

open Nat

theorem nth_equation (n : ℕ) (hn : 0 < n) :
  (n + 1)/((n + 1) * (n + 1) - 1) - (1/(n * (n + 1) * (n + 2))) = 1/(n + 1) := 
by
  sorry

end nth_equation_l719_719612


namespace binomial_expansion_constant_term_l719_719565

theorem binomial_expansion_constant_term :
  let expr := (x + (2 / x)) ^ 6 in
  constant_term expr = 160 :=
by sorry

end binomial_expansion_constant_term_l719_719565


namespace acute_angle_alpha_range_l719_719163

theorem acute_angle_alpha_range (x : ℝ) (α : ℝ) (h1 : 0 < x) (h2 : x < 90) (h3 : α = 180 - 2 * x) : 0 < α ∧ α < 180 :=
by
  sorry

end acute_angle_alpha_range_l719_719163


namespace sandra_money_left_l719_719985

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 0.5
def jelly_bean_cost : ℚ := 0.2
def num_candies : ℕ := 14
def num_jelly_beans : ℕ := 20

def total_money : ℕ := sandra_savings + mother_gift + father_gift
def total_candy_cost : ℚ := num_candies * candy_cost
def total_jelly_bean_cost : ℚ := num_jelly_beans * jelly_bean_cost
def total_cost : ℚ := total_candy_cost + total_jelly_bean_cost
def money_left : ℚ := total_money - total_cost

theorem sandra_money_left : money_left = 11 := by
  sorry

end sandra_money_left_l719_719985


namespace capacitor_charge_difference_l719_719089

variable {ε₀ L d θ ξ : ℝ}
variable (h1 : d ≪ L) (h2 : θ ≈ 0)
variable (h3 : (θ * L) / d ≪ 1)

theorem capacitor_charge_difference
  (h1 : d ≪ L)
  (h2 : θ ≈ 0)
  (h3 : (θ * L) / d ≪ 1) :
  ∃ δq, δq = (ε₀ * L^3 * θ * ξ) / (d^2) :=
sorry

end capacitor_charge_difference_l719_719089


namespace ratio_proof_l719_719764

-- Define the necessary values based on the conditions.
def initial_amount : ℝ := 200
def books : ℝ := 30
def clothes : ℝ := 55
def snacks : ℝ := 25
def gift : ℝ := 20
def electronics : ℝ := 40

-- Define the total spent amount.
def total_spent : ℝ := books + clothes + snacks + gift + electronics

-- Define the unspent amount.
def unspent_money : ℝ := initial_amount - total_spent

-- Define the ratio of total spent to unspent money.
def spent_to_unspent_ratio : ℝ := total_spent / unspent_money

-- Statement of the problem: proving the ratio is 17 / 3.
theorem ratio_proof : spent_to_unspent_ratio = 17 / 3 := 
by sorry

end ratio_proof_l719_719764


namespace area_of_triangle_from_line_l719_719215

-- Define the conditions provided in the problem
def line_eq (B : ℝ) (x y : ℝ) := B * x + 9 * y = 18
def B_val := (36 : ℝ)

theorem area_of_triangle_from_line (B : ℝ) (hB : B = B_val) : 
  (∃ C : ℝ, C = 1 / 2) := by
  sorry

end area_of_triangle_from_line_l719_719215


namespace determine_abcd_l719_719580

theorem determine_abcd (a b c d : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) (h₁ : 0 ≤ b ∧ b ≤ 9) 
    (h₂ : 0 ≤ c ∧ c ≤ 9) (h₃ : 0 ≤ d ∧ d ≤ 9) 
    (h₄ : (10 * a + b) / 99 + (1000 * a + 100 * b + 10 * c + d) / 9999 = 27 / 37) :
    1000 * a + 100 * b + 10 * c + d = 3644 :=
by
  sorry

end determine_abcd_l719_719580


namespace sum_of_ys_in_range_l719_719596

theorem sum_of_ys_in_range : 
  let T := ∑ y in {y : ℝ | y > 0 ∧ y ^ (2 ^ (sqrt 3)) = (sqrt 3) ^ (2 ^ y)}, y in 
  4 < T ∧ T < 7 :=
by
  sorry

end sum_of_ys_in_range_l719_719596


namespace rectangular_circle_eqn_area_of_triangle_eqn_l719_719245

noncomputable def polar_eq_to_rect_eq (ρ θ m : ℝ) : Prop :=
  (ρ^2 - 2 * m * ρ * Real.cos θ + 4 * ρ * Real.sin θ = 1 - 2 * m) ↔
  (∀ x y : ℝ, (x = ρ * Real.cos θ) → (y = ρ * Real.sin θ) →
   (x^2 + y^2 - 2 * m * x + 4 * y = 1 - 2 * m))

theorem rectangular_circle_eqn (ρ θ m : ℝ) :
  polar_eq_to_rect_eq ρ θ m → ∀ x y, (x - m)^2 + (y + 2)^2 = m^2 - 2 * m + 5 :=
by
  intros h1 x y hx hy
  sorry
  
theorem area_of_triangle_eqn (x y : ℝ) :
  ∀ m, ((x - m)^2 + (y + 2)^2 = m^2 - 2 * m + 5 →
  (∀ x, ∃ y, y = Real.sqrt 3 * abs (x - 1) - 2) ∧ ((1, -4) = (x, y)) →
  (m = 1 → ∃ A B, area_of_triangle (1,-4) A B = 2 + Real.sqrt 3)) :=
by
  intros m h_eqn h_curve h_m
  sorry

end rectangular_circle_eqn_area_of_triangle_eqn_l719_719245


namespace binomial_divisible_by_prime_l719_719602

theorem binomial_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (h_range : 1 ≤ k ∧ k ≤ p - 1) : p ∣ Nat.choose p k :=
by
  sorry

end binomial_divisible_by_prime_l719_719602


namespace find_m_correct_l719_719588

structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  XY_length : dist X Y = 80
  XZ_length : dist X Z = 100
  YZ_length : dist Y Z = 120

noncomputable def find_m (t : Triangle) : ℝ :=
  let s := (80 + 100 + 120) / 2
  let A := 1 / 2 * 80 * 100
  let r1 := A / s
  let r2 := r1 / 2
  let r3 := r1 / 4
  let O2 := ((40 / 3), 50 + (40 / 3))
  let O3 := (40 + (20 / 3), (20 / 3))
  let O2O3 := dist O2 O3
  let m := (O2O3^2) / 10
  m

theorem find_m_correct (t : Triangle) : find_m t = 610 := sorry

end find_m_correct_l719_719588


namespace P_irreducible_l719_719601

-- Definition: Let n ≥ 2 and a₁, ..., aₙ be pairwise distinct integers.
variables (n : ℕ) (a : Fin n → ℤ)
variable (h : n ≥ 2)
variable (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)

-- Definition of the polynomial P(X) = (X - a₁)(X - a₂) ... (X - aₙ) - 1
noncomputable def P (X : ℤ) : ℤ[X] :=
  (∏ i : Fin n, X - C (a i)) - 1

-- Theorem: The polynomial P(X) is irreducible in ℤ[X].
theorem P_irreducible : irreducible (P n a) :=
by
  sorry

end P_irreducible_l719_719601


namespace plane_contains_points_l719_719835

def point := (ℝ × ℝ × ℝ)

def is_plane (A B C D : ℝ) (p : point) : Prop :=
  ∃ x y z, p = (x, y, z) ∧ A * x + B * y + C * z + D = 0

theorem plane_contains_points :
  ∃ A B C D : ℤ,
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    is_plane A B C D (2, -1, 3) ∧
    is_plane A B C D (0, -1, 5) ∧
    is_plane A B C D (-2, -3, 4) ∧
    A = 2 ∧ B = 5 ∧ C = -2 ∧ D = 7 :=
  sorry

end plane_contains_points_l719_719835


namespace find_remainder_of_Q_l719_719129

theorem find_remainder_of_Q :
  let Q := 202^1 + 20^21 + 2^21 in
  let R := Q % 1000 in
  R = 354 :=
by
  let Q := 202^1 + 20^21 + 2^21
  have : Q = 202 + 0 + (2^21 % 1000), by sorry
  have : 2^21 % 1000 = 152, by sorry
  have : Q = 202 + 0 + 152, by sorry
  exact calc
    R = Q % 1000                 : by sorry
    ... = (202 + 0 + 152) % 1000 : by sorry
    ... = 354                    : by sorry

end find_remainder_of_Q_l719_719129


namespace pictures_per_coloring_book_l719_719353

theorem pictures_per_coloring_book
    (total_colored : ℕ)
    (remaining_pictures : ℕ)
    (two_books : ℕ)
    (h1 : total_colored = 20) 
    (h2 : remaining_pictures = 68) 
    (h3 : two_books = 2) :
  (total_colored + remaining_pictures) / two_books = 44 :=
by
  sorry

end pictures_per_coloring_book_l719_719353


namespace david_total_hours_on_course_l719_719425

def hours_per_week_class := 2 * 3 + 4 -- hours per week in class
def hours_per_week_homework := 4 -- hours per week in homework
def total_hours_per_week := hours_per_week_class + hours_per_week_homework -- total hours per week

theorem david_total_hours_on_course :
  let total_weeks := 24
  in total_weeks * total_hours_per_week = 336 := by
  sorry

end david_total_hours_on_course_l719_719425


namespace emails_left_are_correct_l719_719628

-- Define the initial conditions for the problem
def initial_emails : ℕ := 400
def trash_emails : ℕ := initial_emails / 2
def remaining_after_trash : ℕ := initial_emails - trash_emails
def work_emails : ℕ := (remaining_after_trash * 40) / 100

-- Define the final number of emails left in the inbox
def emails_left_in_inbox : ℕ := remaining_after_trash - work_emails

-- The proof goal
theorem emails_left_are_correct : emails_left_in_inbox = 120 :=
by 
    -- The computations are correct based on the conditions provided
    have h_trash : trash_emails = 200 := by rfl
    have h_remaining : remaining_after_trash = 200 := by rw [← h_trash, Nat.sub_eq_iff_eq_add (Nat.le_refl 200)]
    have h_work : work_emails = 80 := by 
        rw [← h_remaining, Nat.mul_div_cancel (Nat.le_refl 8000) (Nat.lt_of_sub_one_eq_zero (by refl), 4000)]
    show emails_left_in_inbox = 120 := by
        rw [emails_left_in_inbox, h_remaining, h_work, Nat.sub_eq_iff_eq_add (Nat.le_refl 80)]
        exact rfl

end emails_left_are_correct_l719_719628


namespace sum_of_first_seven_terms_l719_719842

theorem sum_of_first_seven_terms :
  let a := (1:ℚ) / 5
  let r := (1:ℚ) / 5
  (∑ i in Finset.range 7, a * r^i) = 78124 / 312500 := by
sorry

end sum_of_first_seven_terms_l719_719842


namespace pirates_divide_coins_l719_719058

theorem pirates_divide_coins (N : ℕ) (hN : 220 ≤ N ∧ N ≤ 300) :
  ∃ n : ℕ, 
    (N - 2 - (N - 2) / 3 - 2 - (2 * ((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3) - 
    2 - (2 * (((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3)) / 3) / 3 = 84 := 
sorry

end pirates_divide_coins_l719_719058


namespace infinite_series_sum_eq_two_l719_719795

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l719_719795


namespace isosceles_triangle_angle_l719_719238

theorem isosceles_triangle_angle {α β : ℝ} (h1 : 40 + α + β = 180) (h2 : α = β) : α = 70 :=
begin
  sorry
end

end isosceles_triangle_angle_l719_719238


namespace eval_expression_at_minus_3_l719_719439

theorem eval_expression_at_minus_3 :
  (5 + 2 * x * (x + 2) - 4^2) / (x - 4 + x^2) = -5 / 2 :=
by
  let x := -3
  sorry

end eval_expression_at_minus_3_l719_719439


namespace hyperbola_center_is_correct_l719_719834

theorem hyperbola_center_is_correct :
  ∃ h k : ℝ, (∀ x y : ℝ, ((4 * y + 8)^2 / 16^2) - ((5 * x - 15)^2 / 9^2) = 1 → x - h = 0 ∧ y + k = 0) ∧ h = 3 ∧ k = -2 :=
sorry

end hyperbola_center_is_correct_l719_719834


namespace complex_number_in_first_quadrant_l719_719017

-- Define the complex number and its components
def complex_number : ℂ := i * (2 - i)

def real_part (z : ℂ) : ℝ := z.re
def imaginary_part (z : ℂ) : ℝ := z.im

-- Define the statement that the point lies in the first quadrant
theorem complex_number_in_first_quadrant : 
  real_part complex_number > 0 ∧ imaginary_part complex_number > 0 :=
sorry

end complex_number_in_first_quadrant_l719_719017


namespace two_digit_numbers_count_l719_719962

/-
Problem Statement:
M is a two-digit number whose unit digit is not zero.
After swapping the unit digit and the ten's digit of M, we get another two-digit number N.
If M - N is exactly the cube of some positive integer, then there are a total of 6 such numbers.
-/

theorem two_digit_numbers_count :
  {M : ℕ | 10 ≤ M ∧ M < 100 ∧ (∃ (c : ℕ), c > 0 ∧ c^3 < 100 ∧ (let a := M / 10, b := M % 10 in
    a ≠ 0 ∧ b ≠ 0 ∧ 9 * (a - b) = c^3))}.to_finset.card = 6 :=
by
  sorry

end two_digit_numbers_count_l719_719962


namespace new_recipe_water_amount_l719_719316

theorem new_recipe_water_amount :
  ∀ (sugar cups : ℕ), 
  (c : ℚ) (original_flour original_water original_sugar : ℚ)
  (new_flour new_water new_sugar : ℚ),
  original_flour / original_water = 7 / 2 →
  original_flour / original_sugar = 7 →
  new_flour / new_water = (2 * (original_flour / original_water)) →
  new_flour / new_sugar = (original_flour / (2 * original_sugar)) →
  new_sugar = cups →
  cups = 4 →
  new_water = c →
  c = 2 :=
by
  intros sugar cups c original_flour original_water original_sugar new_flour new_water new_sugar 
  h1 h2 h3 h4 h5 h6 h7
  sorry

end new_recipe_water_amount_l719_719316


namespace monotonic_function_a_ge_one_l719_719196

theorem monotonic_function_a_ge_one (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2 * x + a) ≥ 0) → a ≥ 1 :=
by
  intros h
  sorry

end monotonic_function_a_ge_one_l719_719196


namespace sum_of_interior_angles_hexagon_l719_719686

theorem sum_of_interior_angles_hexagon : 
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_hexagon_l719_719686


namespace each_child_apples_l719_719081

-- Define the given conditions
def total_apples : ℕ := 450
def num_adults : ℕ := 40
def num_adults_apples : ℕ := 3
def num_children : ℕ := 33

-- Define the theorem to prove
theorem each_child_apples : 
  let total_apples_eaten_by_adults := num_adults * num_adults_apples
  let total_apples_for_children := total_apples - total_apples_eaten_by_adults
  let apples_per_child := total_apples_for_children / num_children
  apples_per_child = 10 :=
by
  sorry

end each_child_apples_l719_719081


namespace sum_of_squares_gt_five_l719_719657

theorem sum_of_squares_gt_five (a b c : ℝ) (h : a + b + c = 4) : a^2 + b^2 + c^2 > 5 :=
sorry

end sum_of_squares_gt_five_l719_719657


namespace sum_series_eq_two_l719_719797

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l719_719797


namespace geometric_progression_condition_l719_719113

theorem geometric_progression_condition (a b c d : ℝ) :
  (∃ r : ℝ, (b = a * r ∨ b = a * -r) ∧
             (c = a * r^2 ∨ c = a * (-r)^2) ∧
             (d = a * r^3 ∨ d = a * (-r)^3) ∧
             (a = b / r ∨ a = b / -r) ∧
             (b = c / r ∨ b = c / -r) ∧
             (c = d / r ∨ c = d / -r) ∧
             (d = a / r ∨ d = a / -r)) ↔
  (a = b ∨ a = -b) ∧ (a = c ∨ a = -c) ∧ (a = d ∨ a = -d) := sorry

end geometric_progression_condition_l719_719113


namespace meaningful_iff_gt_3_l719_719008

section meaningful_expression

variable (a : ℝ)

def is_meaningful (a : ℝ) : Prop :=
  (a > 3)

theorem meaningful_iff_gt_3 : (∃ b, b = (a + 3) / Real.sqrt (a - 3)) ↔ is_meaningful a :=
by
  sorry

end meaningful_expression

end meaningful_iff_gt_3_l719_719008


namespace constant_term_position_l719_719826

theorem constant_term_position (x : ℝ) :
  let T (r : ℕ) := (Nat.choose 15 r) * 6^(15-r) * (-1)^r * x^((30 - 3*r)/2) in
  ∃ r : ℕ, r = 10 ∧ T r = (Nat.choose 15 10) * 6^(15-10) * (-1)^10 * x^0 :=
by
  sorry

end constant_term_position_l719_719826


namespace tom_age_ratio_l719_719678

theorem tom_age_ratio (T N : ℕ) (h1 : T = 2 * (T / 2)) (h2 : T - N = 3 * ((T / 2) - 3 * N)) : T / N = 16 :=
  sorry

end tom_age_ratio_l719_719678


namespace regular_tetrahedron_path_length_bound_l719_719286

noncomputable def path_length_bound (A B : Point) (T : Tetrahedron) : Prop :=
  ∃ (P : List Point), 
    (path_on_surface P T) ∧
    (path_length P ≤ 2 / Real.sqrt 3)

theorem regular_tetrahedron_path_length_bound :
  ∀ (T : Tetrahedron) (A B : Point),
    (is_regular T) ∧ 
    (edge_length T = 1) ∧ 
    (on_surface A T) ∧ 
    (on_surface B T) → 
    path_length_bound A B T :=
by
  sorry

end regular_tetrahedron_path_length_bound_l719_719286


namespace emails_left_are_correct_l719_719629

-- Define the initial conditions for the problem
def initial_emails : ℕ := 400
def trash_emails : ℕ := initial_emails / 2
def remaining_after_trash : ℕ := initial_emails - trash_emails
def work_emails : ℕ := (remaining_after_trash * 40) / 100

-- Define the final number of emails left in the inbox
def emails_left_in_inbox : ℕ := remaining_after_trash - work_emails

-- The proof goal
theorem emails_left_are_correct : emails_left_in_inbox = 120 :=
by 
    -- The computations are correct based on the conditions provided
    have h_trash : trash_emails = 200 := by rfl
    have h_remaining : remaining_after_trash = 200 := by rw [← h_trash, Nat.sub_eq_iff_eq_add (Nat.le_refl 200)]
    have h_work : work_emails = 80 := by 
        rw [← h_remaining, Nat.mul_div_cancel (Nat.le_refl 8000) (Nat.lt_of_sub_one_eq_zero (by refl), 4000)]
    show emails_left_in_inbox = 120 := by
        rw [emails_left_in_inbox, h_remaining, h_work, Nat.sub_eq_iff_eq_add (Nat.le_refl 80)]
        exact rfl

end emails_left_are_correct_l719_719629


namespace train_platform_length_l719_719392

theorem train_platform_length (t1 : ℝ) (t2 : ℝ) (v_train_kmph : ℝ) (v_man_kmph : ℝ) 
  (v_train_ms : ℝ) (v_man_ms : ℝ) (relative_speed : ℝ) (L_train : ℝ) (L_platform : ℝ) : 
  t1 = 30 →
  t2 = 20 →
  v_train_kmph = 54 →
  v_man_kmph = 5 →
  v_train_ms = v_train_kmph * (1000 / 3600) →
  v_man_ms = v_man_kmph * (1000 / 3600) →
  relative_speed = v_train_ms - v_man_ms →
  L_train = relative_speed * t2 →
  L_train + L_platform = v_train_ms * t1 →
  L_platform ≈ 177.778 :=
by
  intros
  sorry

end train_platform_length_l719_719392


namespace cut_into_square_l719_719248

theorem cut_into_square (A : ℕ) (h1 : ∃ (n : ℕ), A = n * n) : 
  ∃ (parts : list (grid_figure)), 
  (length parts = 3) ∧ (∀ part ∈ parts, area part + area (rest_of_figure parts) = A) ∧ 
  can_form_square_by_reassembling parts :=
sorry

end cut_into_square_l719_719248


namespace theta_terminal_side_l719_719911

theorem theta_terminal_side (alpha : ℝ) (theta : ℝ) (h1 : alpha = 1560) (h2 : -360 < theta ∧ theta < 360) :
    (theta = 120 ∨ theta = -240) := by
  -- The proof steps would go here
  sorry

end theta_terminal_side_l719_719911


namespace sum_series_eq_two_l719_719796

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l719_719796


namespace leftover_value_is_correct_l719_719749

def value_of_leftover_coins (total_quarters total_dimes quarters_per_roll dimes_per_roll : ℕ) : ℝ :=
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters * 0.25) + (leftover_dimes * 0.10)

def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40

def total_quarters : ℕ := michael_quarters + anna_quarters
def total_dimes : ℕ := michael_dimes + anna_dimes

theorem leftover_value_is_correct : 
  value_of_leftover_coins total_quarters total_dimes quarters_per_roll dimes_per_roll = 10.65 :=
by
  sorry

end leftover_value_is_correct_l719_719749


namespace complex_modulus_l719_719873

theorem complex_modulus (z : ℂ) (i : ℂ) (h : (1 - i) * z = i) (hi : i.im = 1 ∧ i.re = 0) : 
  |z| = (Real.sqrt 2) / 2 := by
  sorry

end complex_modulus_l719_719873


namespace max_tan_beta_l719_719483

theorem max_tan_beta (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2) 
  (h : α + β ≠ π / 2) (h_sin_cos : Real.sin β = 2 * Real.cos (α + β) * Real.sin α) : 
  Real.tan β ≤ Real.sqrt 3 / 3 :=
sorry

end max_tan_beta_l719_719483


namespace increasing_interval_of_exponential_quadratic_l719_719647

theorem increasing_interval_of_exponential_quadratic :
  ∀ x : ℝ, (f x = real.exp (x^2 + 2 * x)) →
  (∀ x y : ℝ, x < y →
    (real.exp (x^2 + 2 * x)) < (real.exp (y^2 + 2 * y)) ↔ (-1 < x ∧ -1 < y)) :=
by
  intro x h x₁ x₂ hx
  sorry

end increasing_interval_of_exponential_quadratic_l719_719647


namespace find_x_l719_719534

theorem find_x (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 :=
sorry

end find_x_l719_719534


namespace chord_length_l719_719537

theorem chord_length
  (a b c A B C : ℝ)
  (h₁ : c * Real.sin C = 3 * a * Real.sin A + 3 * b * Real.sin B)
  (O : ℝ → ℝ → Prop)
  (hO : ∀ x y, O x y ↔ x^2 + y^2 = 12)
  (l : ℝ → ℝ → Prop)
  (hl : ∀ x y, l x y ↔ a * x - b * y + c = 0) :
  (2 * Real.sqrt ( (2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 )) = 6 :=
by
  sorry

end chord_length_l719_719537


namespace distance_between_centers_of_circles_l719_719051

theorem distance_between_centers_of_circles 
  (a b c : ℝ)
  (h1 : a = 8)
  (h2 : b = 15)
  (h3 : c = 17)
  (right_triangle : a^2 + b^2 = c^2)
  : (dist_between_centers a b c) = 13 / 2 :=
by {
  sorry
}

end distance_between_centers_of_circles_l719_719051


namespace minimize_distance_and_profit_l719_719418

-- Definitions of the conditions
def eq1 (p1 x1 : ℝ) := 3 * x1 - 4 * p1 - 30 = 0
def eq2 (p2 x2 : ℝ) := p2^2 - 12 * p2 + x2^2 - 14 * x2 + 69 = 0
def distance (p1 x1 p2 x2 : ℝ) := real.sqrt ((x1 - x2)^2 + (p1 - p2)^2)

-- Statement that needs to be proved
theorem minimize_distance_and_profit (p1 x1 p2 x2 : ℝ) :
  eq1 p1 x1 →
  eq2 p2 x2 →
  distance p1 x1 p2 x2 = 2.6 ∧ (x1 + x2 - p1 - p2) > 0 := 
sorry

end minimize_distance_and_profit_l719_719418


namespace infinite_series_equals_two_l719_719805

noncomputable def sum_series : ℕ → ℝ := λ k, (8^k : ℝ) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_equals_two :
  (∑' k : ℕ, if k > 0 then sum_series k else 0) = 2 :=
by 
  sorry

end infinite_series_equals_two_l719_719805


namespace percy_swim_weeks_l719_719283

theorem percy_swim_weeks :
  let hours_weekdays := 2 * 5
  let hours_weekend := 3
  let hours_per_week := hours_weekdays + hours_weekend
  (52 / hours_per_week) = 4 :=
by
  let hours_weekdays := 2 * 5
  let hours_weekend := 3
  let hours_per_week := hours_weekdays + hours_weekend
  have : hours_per_week = 13 := by
    simp [hours_weekdays, hours_weekend]
  rw [this]
  norm_num

end percy_swim_weeks_l719_719283


namespace max_min_difference_l719_719072

theorem max_min_difference :
  let total_students := 3000
  ∧ let german_min := 2100
  ∧ let german_max := 2250
  ∧ let russian_min := 1200
  ∧ let russian_max := 1500
  ∧ ∃ m M, german_min + russian_min - m = total_students ∧ german_max + russian_max - M = total_students
  → (M - m = 450) :=
by {
  intros,
  sorry
}

end max_min_difference_l719_719072


namespace altitudes_concurrent_l719_719288

-- Define an acute triangle and its altitudes
variable {A B C : Point} (hAcute : acute_triangle A B C)
variable (H_A H_B H_C : Point)

-- Altitudes from vertices A, B, and C
variable (A_HA : altitude A H_A B C)
variable (B_HB : altitude B H_B A C)
variable (C_HC : altitude C H_C A B)

-- The theorem to prove in Lean 4
theorem altitudes_concurrent : ∃ H : Point, concurrent A_HA B_HB C_HC := by
  sorry

end altitudes_concurrent_l719_719288


namespace slope_at_A_is_7_l719_719320

def curve (x : ℝ) : ℝ := x^2 + 3 * x

def point_A : ℝ × ℝ := (2, 10)

theorem slope_at_A_is_7 : (deriv curve 2) = 7 := 
by
  sorry

end slope_at_A_is_7_l719_719320


namespace grassland_ratio_l719_719762

theorem grassland_ratio (h : ∀ A B : ℝ, (7 * A + (4 * A / 7)) = B * (7 + 4) → (A / B).denom = 44 → (A / B).num = 105) : 
  ∃ (a b : ℕ), gcd a b = 1 ∧ a : b = 105 : 44 :=
by
  sorry

end grassland_ratio_l719_719762


namespace calculate_expression_l719_719781

theorem calculate_expression :
  (1 / 2) ^ (-1 : ℤ) + 4 * Real.cos (Float.pi * 60 / 180) - | -3 | + Real.sqrt 9 - (-2023 : ℤ) ^ (0 : ℕ) + (-1 : ℤ) ^ (2023 - 1) = 4 :=
by sorry

end calculate_expression_l719_719781


namespace b_profit_percentage_correct_l719_719751

def profit_percentage_b (P_A : ℝ) (Profit_A : ℝ) (S_C : ℝ) : ℝ :=
  let S_B := P_A + (Profit_A / 100) * P_A in 
  let Profit_B := S_C - S_B in
  (Profit_B / S_B) * 100

theorem b_profit_percentage_correct :
  profit_percentage_b 114.94 35 225 = 44.99 :=
by
  . intro
  . skip

end b_profit_percentage_correct_l719_719751


namespace original_value_of_tempo_l719_719059

noncomputable def original_value (insured_ratio : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) : ℚ :=
(premium_amount / (premium_rate * insured_ratio))

theorem original_value_of_tempo
  (insured_ratio : ℚ := 4 / 5)
  (premium_rate : ℚ := 1.3 / 100)
  (premium_amount : ℚ := 910) :
  original_value insured_ratio premium_rate premium_amount = 87500 :=
by
  unfold original_value
  simp
  sorry

end original_value_of_tempo_l719_719059


namespace side_length_square_l719_719773

-- Define the side lengths and given constraints
def octagon := Type 
def side_length_AB (A B : octagon) : ℝ := 50
def side_length_EF (E F : octagon) : ℝ := 46 * Real.sqrt 2 - 2
def is_square (I J K L : octagon) := 
  (I ≠ J) ∧ (J ≠ K) ∧ (K ≠ L) ∧ (L ≠ I) ∧ 
  (I ≠ K) ∧ (J ≠ L) ∧ 
  (∀ (AI BI CJ DJ EK FK GL HL: octagon), 
     ∠ (AI B CJ) = 90 ∧ ∠ (EK CJ L GL) = 90 )

-- Target: Prove that the side length of the square is 25
theorem side_length_square (I J K L: octagon) 
    (h1: side_length_AB = 50)
    (h2: side_length_EF = 46 * Real.sqrt 2 - 2)
    (h3: is_square I J K L): 
     ∃ x: ℝ, x = 25 :=
    sorry

end side_length_square_l719_719773


namespace scientific_notation_of_population_l719_719614

theorem scientific_notation_of_population : (85000000 : ℝ) = 8.5 * 10^7 := 
by
  sorry

end scientific_notation_of_population_l719_719614


namespace hyperbola_equation_l719_719506

open Real

-- Definition of the hyperbola with given parameters a, b
def hyperbola_eq (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Definition of the asymptote
def asymptote_eq (y x : ℝ) : Prop :=
  y = sqrt 3 * x

-- Definition of the directrix of the parabola
def directrix_eq (x : ℝ) : Prop :=
  x = -6

-- The main theorem to be proved
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^2 + b^2 = 36)
  (h4 : sqrt 3 = b / a)
  (h5 : hyperbola_eq (-6) 0 a b ∧ asymptote_eq 0 (-6) ∧ directrix_eq (-6)) :
  hyperbola_eq x y 3 sqrt 27 :=
by
  sorry

end hyperbola_equation_l719_719506


namespace snack_bar_training_count_l719_719070

noncomputable def num_trained_in_snack_bar 
  (total_employees : ℕ) 
  (trained_in_buffet : ℕ) 
  (trained_in_dining_room : ℕ) 
  (trained_in_two_restaurants : ℕ) 
  (trained_in_three_restaurants : ℕ) : ℕ :=
  total_employees - trained_in_buffet - trained_in_dining_room + 
  trained_in_two_restaurants + trained_in_three_restaurants

theorem snack_bar_training_count : 
  num_trained_in_snack_bar 39 17 18 4 2 = 8 :=
sorry

end snack_bar_training_count_l719_719070


namespace num_undefined_values_l719_719844

theorem num_undefined_values (x k : ℝ) :
  (k ≠ 1 ∧ k ≠ -4 → (x = 1 ∨ x = -4 ∨ x = k) → 3) ∧
  ((k = 1 ∨ k = -4) → (x = 1 ∨ x = -4) → 2) :=
by
  sorry

end num_undefined_values_l719_719844


namespace find_symmetric_line_l719_719519

def is_symmetric_line (l1 l2 l: ℝ × ℝ × ℝ) : Prop :=
  ∃ M ∈ ℝ × ℝ, M.1 * l1.1 + M.2 * l1.2 + l1.3 = 0 ∧
    M.1 * l.1 + M.2 * l.2 + l.3 = 0 ∧
    M.1 * l2.1 + M.2 * l2.2 + l2.3 = 0 ∧
    (l1.1 + l2.1) / 2 = M.1 ∧ 
    (l1.2 + l2.2) / 2 = M.2

theorem find_symmetric_line : 
  is_symmetric_line (2, 1, -4) (2, 1, -6) (3, 4, -1) :=
sorry

end find_symmetric_line_l719_719519


namespace max_min_perimeter_area_ratio_l719_719255

theorem max_min_perimeter_area_ratio :
  ∃ (a b c : ℕ), let ABCD : {ABCD : Type} in
    convex_quadrilateral ABCD
    ∧ side_lengths ABCD = {25, 25, ?d₁, ?d₂, ?d₃, ?d₄}
    ∧ max_perimeter ABCD
    ∧ min_perimeter ABCD
    ∧ ratio_of_areas ABCD (/= a * sqrt b / c)
    ∧ coprime a c
    ∧ (b ∣ (sqrt b))
    ∧ a + b + c = 97 :=
  sorry

end max_min_perimeter_area_ratio_l719_719255


namespace length_of_AE_l719_719361

/-- Given the conditions on the pentagon ABCDE:
1. AB = 2, BC = 2, CD = 5, DE = 7
2. AC is the largest side in triangle ABC
3. CE is the smallest side in triangle ECD
4. In triangle ACE all sides are integers and have distinct lengths,
prove that the length of side AE is 5. -/
theorem length_of_AE
  (AB BC CD DE : ℕ)
  (hAB : AB = 2)
  (hBC : BC = 2)
  (hCD : CD = 5)
  (hDE : DE = 7)
  (AC : ℕ) 
  (hAC_large : AB < AC ∧ BC < AC)
  (CE : ℕ)
  (hCE_small : CE < CD ∧ CE < DE)
  (AE : ℕ)
  (distinct_sides : ∀ x y z : ℕ, x ≠ y → x ≠ z → y ≠ z → (AC = x ∨ CE = x ∨ AE = x) → (AC = y ∨ CE = y ∨ AE = y) → (AC = z ∨ CE = z ∨ AE = z)) :
  AE = 5 :=
sorry

end length_of_AE_l719_719361


namespace alternating_binomial_sum_l719_719005

theorem alternating_binomial_sum:
  (T : ℤ) = ∑ k in Finset.range 50, (-1)^k * Nat.choose 100 (2*k+1) → T = 0 :=
by
  sorry

end alternating_binomial_sum_l719_719005


namespace sum_group_with_10_is_22_l719_719099

def groupA := {2, 5, 9}
def groupS := {1, 3, 4, 6, 7, 8, 10}

theorem sum_group_with_10_is_22 :
  ∀ (G1 G2 G3 : Finset ℕ), 
    (G1 ∪ G2 ∪ G3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) ∧
    (groupA ⊆ G1 ∨ groupA ⊆ G2 ∨ groupA ⊆ G3) ∧
    (∀ g ∈ {G1, G2, G3}, 
      ∀ x y ∈ g, 
        x ≠ y → (x - y) ∉ g) → 
    ∃ g ∈ {G1, G2, G3}, 10 ∈ g ∧ g.sum id = 22 := sorry

end sum_group_with_10_is_22_l719_719099


namespace triangle_APR_perimeter_l719_719680

variables (A B C P Q R : Point)
variable {circle : Circle}
variables (tangent1 tangent2 tangent3 : Line)
variables (AB AC PQ PR AR : ℝ)

-- Conditions
axiom cond1 : Tangent A B circle
axiom cond2 : Tangent A C circle
axiom cond3 : Tangent A Q circle
axiom cond4 : tangent3 ∩ AB = P
axiom cond5 : tangent3 ∩ AC = R
axiom cond6 : AB = 18
axiom cond7 : MidPoint Q B C

-- Tangent Properties
axiom tangent_prop1 : AB = AC
axiom tangent_prop2 : BP = PQ
axiom tangent_prop3 : CR = QR

-- Defining the segments based on problem conditions
def AP := AB - PQ
def AR := AC - QR
def PR := PQ + QR

-- Perimeter calculation
def perimeter := AP + PR + AR

-- Proof statement
theorem triangle_APR_perimeter : perimeter = 36 :=
sorry

end triangle_APR_perimeter_l719_719680


namespace main_cost_cost_per_nautical_mile_min_cost_per_nautical_mile_l719_719437

variable (v : ℝ) (q : ℝ) (p : ℝ) (F : ℝ) (L : ℝ)
variable (h1 : 0 ≤ v ∧ v ≤ 30)
variable (h2 : L = 4050)
variable (h3 : F = 0.6 * v ^ 3)
variable (h4 : q = F + L)
variable (h5 : p = q / v)

theorem main_cost (v : ℝ) (q : ℝ) :
  0 ≤ v ∧ v ≤ 30 → L = 4050 ∧ F = 0.6 * v ^ 3 ∧ q = F + L →
  q = 0.6 * v ^ 3 + 4050 :=
by
  intros
  rw [h3, h2]
  exact h4

theorem cost_per_nautical_mile (v : ℝ) (p : ℝ) :
  0 ≤ v ∧ v ≤ 30 → L = 4050 ∧ F = 0.6 * v ^ 3 ∧ q = F + L ∧ p = q / v →
  p = (0.6 * v ^ 3 + 4050) / v :=
by
  intros
  rw [h3, h2]
  exact h5

theorem min_cost_per_nautical_mile (p : ℝ) (v : ℝ) :
  0 ≤ v ∧ v ≤ 30 ∧ v = 15 ∧ L = 4050 ∧ F = 0.6 * v ^ 3 ∧ q = F + L ∧ p = q / v →
  p = 405 :=
by
  intros
  rw [h3, h2]
  have hv : v = 15, from and.right (and.left h1),
  rw hv at *,
  exact h5
  sorry

end main_cost_cost_per_nautical_mile_min_cost_per_nautical_mile_l719_719437


namespace num_mappings_l719_719607

-- Define set A and set B
def A : Set ℕ := {0, 1}
def B : Set ℕ := {a, b, c}

-- Define the theorem to state that the number of mappings from A to B is 9
theorem num_mappings :
  ∃ (f : A → B), 3^2 = 9 :=
by {
  sorry
}

end num_mappings_l719_719607


namespace remainder_53_pow_10_div_8_l719_719318

theorem remainder_53_pow_10_div_8 : (53^10) % 8 = 1 := 
by sorry

end remainder_53_pow_10_div_8_l719_719318


namespace leak_empties_cistern_in_24_hours_l719_719371

theorem leak_empties_cistern_in_24_hours (R L : ℚ) (h1 : 6 * R = 1) (h2 : (R - L) * 8 = 1) : 
  1 / L = 24 :=
by
  -- We introduce the variables and hypothesis
  have hR : R = 1 / 6 := by
    -- Use the given condition
    linarith
  have hL : L = 1 / 24 := by
    -- Substitute R and solve for L using the second condition
    linarith
  -- Use the value of L to find the time to empty the cistern
  rw hL
  linarith

end leak_empties_cistern_in_24_hours_l719_719371


namespace calc_expr_l719_719032

theorem calc_expr : 
  (-1: ℝ)^4 - 2 * Real.tan (Real.pi / 3) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := 
by
  sorry

end calc_expr_l719_719032


namespace amoeba_growth_after_5_days_l719_719397

theorem amoeba_growth_after_5_days : (3 : ℕ)^5 = 243 := by
  sorry

end amoeba_growth_after_5_days_l719_719397


namespace complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l719_719894

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of a set
def C_A : Set ℕ := U \ A
def C_B : Set ℕ := U \ B

-- Questions rephrased as theorem statements
theorem complement_of_A_in_U : C_A = {2, 4, 5} := by sorry
theorem intersection_of_A_and_B : A ∩ B = ∅ := by sorry
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by sorry
theorem union_of_complements_of_A_and_B : C_A ∪ C_B = U := by sorry

end complement_of_A_in_U_intersection_of_A_and_B_union_of_A_and_B_union_of_complements_of_A_and_B_l719_719894


namespace a_4_value_l719_719141

-- Defining the polynomial (2x - 3)^6
def polynomial_expansion (x : ℝ) := (2 * x - 3) ^ 6

-- Given conditions polynomial expansion in terms of (x - 1)
def polynomial_coefficients (x : ℝ) (a : Fin 7 → ℝ) : ℝ :=
  a 0 + a 1 * (x - 1) + a 2 * (x - 1) ^ 2 + a 3 * (x - 1) ^ 3 + a 4 * (x - 1) ^ 4 +
  a 5 * (x - 1) ^ 5 + a 6 * (x - 1) ^ 6

-- The proof problem asking to show a_4 = 240
theorem a_4_value : 
  ∀ a : Fin 7 → ℝ, (∀ x : ℝ, polynomial_expansion x = polynomial_coefficients x a) → a 4 = 240 := by 
  sorry

end a_4_value_l719_719141


namespace plot_area_is_correct_l719_719057

noncomputable def scaled_area_in_acres
  (scale_cm_miles : ℕ)
  (area_conversion_factor_miles_acres : ℕ)
  (bottom_cm : ℕ)
  (top_cm : ℕ)
  (height_cm : ℕ) : ℕ :=
  let area_cm_squared := (1 / 2) * (bottom_cm + top_cm) * height_cm
  let area_in_squared_miles := area_cm_squared * (scale_cm_miles * scale_cm_miles)
  area_in_squared_miles * area_conversion_factor_miles_acres

theorem plot_area_is_correct :
  scaled_area_in_acres 3 640 18 14 12 = 1105920 :=
by
  sorry

end plot_area_is_correct_l719_719057


namespace gross_profit_without_discount_l719_719355

variable (C P : ℝ) -- Defining the cost and the full price as real numbers

-- Condition 1: Merchant sells an item at 10% discount (0.9P)
-- Condition 2: Makes a gross profit of 20% of the cost (0.2C)
-- SP = C + GP implies 0.9 P = 1.2 C

theorem gross_profit_without_discount :
  (0.9 * P = 1.2 * C) → ((C / 3) / C * 100 = 33.33) :=
by
  intro h
  sorry

end gross_profit_without_discount_l719_719355


namespace integer_roots_of_polynomial_l719_719385

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | (x^3 + a₂ * x^2 + a₁ * x - 18 = 0)} ⊆ {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
by sorry

end integer_roots_of_polynomial_l719_719385


namespace bears_in_stock_initially_l719_719761

theorem bears_in_stock_initially 
  (shipment_bears : ℕ) (shelf_bears : ℕ) (shelves_used : ℕ)
  (total_bears_shelved : shipment_bears + shelf_bears * shelves_used = 24) : 
  (24 - shipment_bears = 6) :=
by
  exact sorry

end bears_in_stock_initially_l719_719761


namespace prob_of_mixed_group_l719_719039

theorem prob_of_mixed_group (m f total : Nat) (hm : m = 3) (hf : f = 4) (htotal : total = 7)
  (hchoose3 : ∑ n in set.to_finset (set.range 4), combinatorial n (total - n) = combinatorial total 3) : 
  (1 - (combinatorial m 3 / combinatorial total 3 + combinatorial f 3 / combinatorial total 3)) = 6 / 7 :=
by
  rw [hm, hf, htotal] at *
  sorry

end prob_of_mixed_group_l719_719039


namespace sin_squared_sum_ge_inverse_sqrt_R_l719_719160

noncomputable def tetrahedron_problem 
  (a₁₂ a₁₃ a₁₄ a₂₃ a₂₄ a₃₄ R₁ R₂ R₃ R₄ : ℝ)
  (α β γ : ℝ) : Prop :=
  a₁₂ * a₃₄ = 1 ∧ 
  a₁₃ * a₂₄ = 1 ∧ 
  a₁₄ * a₂₃ = 1 ∧ 
  a₁₂ > 0 ∧ a₁₃ > 0 ∧ a₁₄ > 0 ∧ a₂₃ > 0 ∧ a₂₄ > 0 ∧ a₃₄ > 0 ∧ 
  0 < α ∧ α < π ∧ 0 < β ∧ β < π ∧ 0 < γ ∧ γ < π ∧ 
  0 < R₁ ∧ 0 < R₂ ∧ 0 < R₃ ∧ 0 < R₄ 

theorem sin_squared_sum_ge_inverse_sqrt_R 
  (a₁₂ a₁₃ a₁₄ a₂₃ a₂₄ a₃₄ R₁ R₂ R₃ R₄ : ℝ) 
  (α β γ : ℝ) 
  (h : tetrahedron_problem a₁₂ a₁₃ a₁₄ a₂₃ a₂₄ a₃₄ R₁ R₂ R₃ R₄ α β γ) :
  (sin α)^2 + (sin β)^2 + (sin γ)^2 ≥ 1 / (sqrt (R₁ * R₂ * R₃ * R₄)) :=
sorry

end sin_squared_sum_ge_inverse_sqrt_R_l719_719160


namespace find_k_l719_719591

def f (a b c x : ℤ) : ℤ := a * x * x + b * x + c

theorem find_k : 
  ∃ k : ℤ, 
    ∃ a b c : ℤ, 
      f a b c 1 = 0 ∧
      60 < f a b c 6 ∧ f a b c 6 < 70 ∧
      120 < f a b c 9 ∧ f a b c 9 < 130 ∧
      10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)
      ∧ k = 4 :=
by
  sorry

end find_k_l719_719591


namespace six_digit_odd_even_alternating_no_repeats_l719_719653

noncomputable def numOddEvenAlternatingNumbers (digits : List ℕ) : ℕ :=
  (digits.permutations.filter (λ l, l.length = 6 ∧ ((list.alternating_odd_even l) ∧ (list.no_duplicates l)))).length

theorem six_digit_odd_even_alternating_no_repeats :
  numOddEvenAlternatingNumbers [0, 1, 2, 3, 4, 5] = 60 :=
sorry

end six_digit_odd_even_alternating_no_repeats_l719_719653


namespace sarah_bought_new_shirts_l719_719621

-- Define the given conditions
def original_shirts : ℕ := 9
def total_shirts : ℕ := 17

-- The proof statement: Prove that the number of new shirts is 8
theorem sarah_bought_new_shirts : total_shirts - original_shirts = 8 := by
  sorry

end sarah_bought_new_shirts_l719_719621


namespace smallest_positive_period_of_f_monotonic_increasing_interval_of_f_range_of_f_on_interval_l719_719513

noncomputable def f (x : ℝ) : ℝ := √3 * (sin x ^ 2 - cos x ^ 2) + 2 * sin x * cos x

theorem smallest_positive_period_of_f :
  ∃ (p > 0), ∀ x, f (x + p) = f x ∧ p = π := 
sorry

theorem monotonic_increasing_interval_of_f :
  ∃ a b, [a, b] = [π / 6, 5 * π / 6] ∧ ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y :=
sorry

theorem range_of_f_on_interval :
  ∀ x, -π / 3 ≤ x ∧ x ≤ π / 3 → -2 ≤ f x ∧ f x ≤ √3 :=
sorry

end smallest_positive_period_of_f_monotonic_increasing_interval_of_f_range_of_f_on_interval_l719_719513


namespace max_n_rectangles_disjoint_intersect_l719_719837

theorem max_n_rectangles_disjoint_intersect (n : ℕ)
  (A B : Fin n → Set (ℝ × ℝ))
  (h_sides_parallel_to_axes : ∀ i, is_rectangle_parallel_to_axes (A i) ∧ is_rectangle_parallel_to_axes (B i))
  (h_disjoint : ∀ i, Disjoint (A i) (B i))
  (h_common_point : ∀ i j, i ≠ j → (Set.Inter (λ k, A i ∩ B j)).Nonempty) :
  n ≤ 4 :=
sorry

-- Definitions used in the theorem
def is_rectangle_parallel_to_axes (r : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ), r = { p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ b ∧ c ≤ p.2 ∧ p.2 ≤ d }

end max_n_rectangles_disjoint_intersect_l719_719837


namespace positive_conditions_triangle_sides_l719_719634

variables {a b c α β : ℝ}

-- Define the conditions given in the problem
def roots (α β : ℝ) (ha : α > 0) (hb : β ≠ 0) : Prop :=
  ∃ (a b c : ℝ), (a + b + c = 2 * α) ∧ (ab + bc + ca = α^2 + β^2)

-- First part of the problem
theorem positive_conditions (α β : ℝ) (ha : α > 0) (hb : β ≠ 0) (h : roots α β ha hb) :
  a > 0 ∧ b > 0 ∧ c > 0 := sorry

-- Second part of the problem
theorem triangle_sides (α β : ℝ) (ha : α > 0) (hb : β ≠ 0) (h : roots α β ha hb) :
  (sqrt a + sqrt b > sqrt c) ∧ (sqrt b + sqrt c > sqrt a) ∧ (sqrt c + sqrt a > sqrt b) := sorry

end positive_conditions_triangle_sides_l719_719634


namespace tickets_discount_l719_719357

/-- At a special sale, 12 tickets can be purchased for the price of 3 tickets.
Given this condition, prove that the amount saved by purchasing 12 tickets at the sale 
will be 75% of the original price of the 12 tickets. -/
theorem tickets_discount (P : ℝ) (h₁ : 12 * P) (h₂ : 3 * P) :
  ((12 * P - 3 * P) / (12 * P)) * 100 = 75 :=
by 
  sorry

end tickets_discount_l719_719357


namespace math_proof_problem_l719_719479

-- The given conditions
def condition1 {α : ℝ} : Prop := 2 * sin α * tan α = 3
def condition2 {α : ℝ} : Prop := 0 < α ∧ α < π

-- The propositions to be proved
def proposition1 : Prop := ∃ α : ℝ, condition1 α ∧ condition2 α ∧ α = π / 3
def proposition2 : Prop :=
  ∀ (α : ℝ) (hα : α = π / 3), ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 4) → 
  -1 ≤ 4 * sin x * sin (x - α) ∧ 4 * sin x * sin (x - α) ≤ 0

-- The proof problem statement
theorem math_proof_problem : proposition1 ∧ proposition2 := by
  sorry

end math_proof_problem_l719_719479


namespace find_angle_D_l719_719849

theorem find_angle_D 
  (A B C D : ℝ) 
  (h1 : A + B = 180) 
  (h2 : C = D + 10) 
  (h3 : A = 50)
  : D = 20 := by
  sorry

end find_angle_D_l719_719849


namespace smallest_positive_integer_l719_719347

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end smallest_positive_integer_l719_719347


namespace smallest_n_for_2012_solutions_l719_719258

noncomputable def fractionalPart (x : ℝ) : ℝ := x - floor x

noncomputable def f (x : ℝ) : ℝ := abs (2 * fractionalPart x - 1)

def minN (solutions_needed : ℕ) : ℕ :=
  Inf { n : ℕ | ∃ x : ℝ, (nf(xf(x)) = x).count ≥ solutions_needed }

theorem smallest_n_for_2012_solutions :
  minN 2012 = 1006 := sorry

end smallest_n_for_2012_solutions_l719_719258


namespace contrapositive_necessary_condition_l719_719904

theorem contrapositive_necessary_condition {p q : Prop} (h : p → q) : ¬p → ¬q :=
  by sorry

end contrapositive_necessary_condition_l719_719904


namespace cos_2015_eq_neg_m_l719_719175

variable (m : ℝ)

-- Given condition
axiom sin_55_eq_m : Real.sin (55 * Real.pi / 180) = m

-- The proof problem
theorem cos_2015_eq_neg_m : Real.cos (2015 * Real.pi / 180) = -m :=
by
  sorry

end cos_2015_eq_neg_m_l719_719175


namespace inverse_log_equality_l719_719887

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.log x / real.log a

theorem inverse_log_equality (a : ℝ) (h : 1 < a) (hf_inv : ∀ x, f (f⁻¹ x) a = x) :
  ∀ x, f 9 a + f 6 a = 2 := 
by
  sorry

end inverse_log_equality_l719_719887


namespace lea_bought_2_more_l719_719997

noncomputable def notebooks_difference (total_cost_tara total_cost_lea : ℝ) (unit_cost : ℝ) (ht : 1 < unit_cost) :=
  ∃ (n_tara n_lea : ℕ), 
    n_tara * unit_cost = total_cost_tara ∧ 
    n_lea * unit_cost = total_cost_lea ∧ 
    n_lea - n_tara = 2

theorem lea_bought_2_more (total_cost_tara total_cost_lea : ℝ) (h_tara : total_cost_tara = 5.2) (h_lea : total_cost_lea = 7.8) : 
  ∃ unit_cost, 1 < unit_cost ∧ notebooks_difference total_cost_tara total_cost_lea unit_cost :=
begin
  use (1.3 : ℝ),
  split,
  { norm_num }, -- Proving 1 < 1.3
  { unfold notebooks_difference,
    use [4, 6],
    simp [h_tara, h_lea],
    norm_num,
    split, 
    { exact rfl },
    { split,
      { exact rfl },
      { norm_num } } },
end

end lea_bought_2_more_l719_719997


namespace inbox_emails_after_movements_l719_719626

def initial_emails := 400
def trash_emails := initial_emails / 2
def remaining_emails := initial_emails - trash_emails
def work_emails := 0.4 * remaining_emails
def final_inbox_emails := remaining_emails - work_emails

theorem inbox_emails_after_movements : final_inbox_emails = 120 :=
by
  sorry

end inbox_emails_after_movements_l719_719626


namespace polar_to_cartesian_circle_l719_719092

theorem polar_to_cartesian_circle :
  ∀ (r : ℝ) (x y : ℝ), r = 3 → r = Real.sqrt (x^2 + y^2) → x^2 + y^2 = 9 :=
by
  intros r x y hr h
  sorry

end polar_to_cartesian_circle_l719_719092


namespace david_course_hours_l719_719428

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end david_course_hours_l719_719428


namespace sequence_arithmetic_sum_of_reciprocals_l719_719186

theorem sequence_arithmetic (a : ℕ → ℕ) (n : ℕ) (h1 : a 2 = 6) (h2 : a 3 - 3 * a 1 = 6) (h_arith : ∀ i, (a (i+1) / (i+1) - a i / i) = (a (i+2) / (i+2) - a (i+1) / (i+1))) :
  (a n = n * (n + 1)) :=
sorry

theorem sum_of_reciprocals (a : ℕ → ℕ) (n : ℕ) (S : ℕ → ℚ) (h1 : a 2 = 6) (h2 : a 3 - 3 * a 1 = 6) (h_arith : ∀ i, (a (i+1) / (i+1) - a i / i) = (a (i+2) / (i+2) - a (i+1) / (i+1))) (h_a : ∀ k, a k = k * (k + 1)) :
  (S n = ∑ i in range (n+1), 1 / a i = (n)/(n + 1)) :=
sorry

end sequence_arithmetic_sum_of_reciprocals_l719_719186


namespace circles_tangent_internally_l719_719827

noncomputable def circle_center_radius (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let h := -a / 2
  let k := -b / 2
  let r := (h * h + k * k - c).sqrt
  (h, k, r)

theorem circles_tangent_internally :
  ∀ (x y : ℝ), 
    (x ^ 2 + y ^ 2 + 2 * x + 4 * y + 1 = 0) → 
    (x ^ 2 + y ^ 2 - 4 * x + 4 * y - 17 = 0) → 
    (∃ h1 k1 r1 h2 k2 r2, 
      (h1, k1, r1) = circle_center_radius 2 4 1 ∧ 
      (h2, k2, r2) = circle_center_radius (-4) 4 (-17) ∧
      (real.sqrt ((h1 - h2) ^ 2 + (k1 - k2) ^ 2) = (r1 - r2).abs)) := 
sorry

end circles_tangent_internally_l719_719827


namespace min_value_of_exponential_expression_l719_719955

theorem min_value_of_exponential_expression 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : 2 * x + y = 6) :
  9^x + 3^y ≥ 54 :=
sorry

end min_value_of_exponential_expression_l719_719955


namespace distinct_four_digit_integers_with_product_18_l719_719661

theorem distinct_four_digit_integers_with_product_18 : 
  {n : Nat // (∀ i < 4, (1 ≤ i.digit n) ∧ (i.digit n ≤ 9)) ∧ digit_product n = 18}.card = 36 :=
sorry

end distinct_four_digit_integers_with_product_18_l719_719661


namespace probability_quitters_same_tribe_l719_719663

noncomputable def num_ways_to_choose_quitters (total_contestants : ℕ) (num_quitters : ℕ) : ℕ :=
binomial total_contestants num_quitters

noncomputable def num_ways_same_tribe (tribe_size : ℕ) (num_quitters : ℕ) : ℕ :=
binomial tribe_size num_quitters

theorem probability_quitters_same_tribe 
  (total_contestants : ℕ) (num_tribes : ℕ) (tribe_size : ℕ) (num_quitters : ℕ) 
  (h_contestants : total_contestants = 20) (h_tribes : num_tribes = 4) (h_tribe_size : tribe_size = 5) 
  (h_quitters : num_quitters = 3) :
  let total_ways := num_ways_to_choose_quitters total_contestants num_quitters,
      ways_same_tribe := num_ways_same_tribe tribe_size num_quitters * num_tribes
  in (ways_same_tribe : ℚ) / (total_ways : ℚ) = 2 / 57 := by
  sorry

end probability_quitters_same_tribe_l719_719663


namespace units_digit_factorial_sum_l719_719687

theorem units_digit_factorial_sum : 
  (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10! + 11! + 12! + 13! + 14! + 15! + 16! + 17! + 18! + 19! + 20!) % 10 = 3 :=
by
  -- Initial calculation of factorials modulo 10
  calc
    1! % 10 = 1     := by norm_num
    2! % 10 = 2     := by norm_num
    3! % 10 = 6     := by norm_num
    4! % 10 = 4     := by norm_num
    -- For n >= 5, n! is a multiple of 10, so n! % 10 = 0
    (1! + 2! + 3! + 4!) % 10 = 13 % 10 := by norm_num
    13 % 10 = 3     := by norm_num
    (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9! + 10! + 11! + 12! + 13! + 14! + 15! + 16! + 17! + 18! + 19! + 20!) % 10 = 3 :=
      -- Summing conditions for all elements from 5! to 20!
      sorry

end units_digit_factorial_sum_l719_719687


namespace chewbacca_gum_pieces_l719_719785

theorem chewbacca_gum_pieces (y : ℚ)
  (h1 : ∀ x : ℚ, x ≠ 0 → (15 - y) = 15 * (25 + 2 * y) / 25) :
  y = 5 / 2 :=
by
  sorry

end chewbacca_gum_pieces_l719_719785


namespace sampling_method_is_systematic_l719_719233

-- Define what systematic sampling means
def systematic_sampling (interval : ℕ) : Prop :=
  interval = 3

-- Given conditions for the problem
def is_inspection_coveting_interval (time_start time_end interval : ℕ) : Prop :=
  time_start = 17 ∧ time_end = 20 ∧ interval = 3

-- The proof statement
theorem sampling_method_is_systematic : 
  ∀ (time_start time_end interval : ℕ), 
  is_inspection_coveting_interval time_start time_end interval → 
  systematic_sampling interval :=
by
  intros time_start time_end interval h
  cases h
  rw h_right
  refl

end sampling_method_is_systematic_l719_719233


namespace cubs_more_home_runs_l719_719077

noncomputable def cubs_home_runs := 2 + 1 + 2
noncomputable def cardinals_home_runs := 1 + 1

theorem cubs_more_home_runs :
  cubs_home_runs - cardinals_home_runs = 3 :=
by
  -- Proof would go here, but we are using sorry to skip it
  sorry

end cubs_more_home_runs_l719_719077


namespace circle_equation_line_l2_equation_l719_719858

-- Define the conditions
variable (centerx centery : ℝ)
variable (radius : ℝ := 4)
variable (l1 : ℝ × ℝ -> ℝ := λ p, 3 * p.1 - 4 * p.2 + 8)
variable (tangent_condition : ∀ (centerx = 0), centerx = 0 ∧ centery < 0 ∧ |3 * centerx - 4 * centery + 8| / (5 : ℝ) = 4)
variable (l2 : ℝ → ℝ := λ k, k - 1 + 3)

-- The first part: finding the equation of the circle
theorem circle_equation :
  (tangent_condition centerx centery radius) →
  (centerx = 0) ∧ centery = -3 ∧ (radius = 4) →
  (∀ x y : ℝ, (x - centerx)^2 + (y - centery)^2 = radius^2) :=
by
  sorry

-- The second part: finding the equation of line l2
theorem line_l2_equation (k : ℝ) :
  let d := |k * 0 - 3 + 3| / (Math.sqrt (k^2 + 1)) in
  d = 2 * Math.sqrt 2 →
  (kx - y + 3 = 0) →
  (∀ x1 y1 x2 y2 : ℝ, (x1 - centerx)^2 + (y1 - centery)^2 = radius^2 ∧
                      (x2 - centerx)^2 + (y2 - centery)^2 = radius^2 ∧
                      (x1, y1), (x2, y2) ∈ (2 * (Math.sqrt (16 - d^2)), 8)) →
  (k = sqrt (14) / 2 ∨ k = -sqrt (14) / 2) :=
by
  sorry

end circle_equation_line_l2_equation_l719_719858


namespace min_number_of_covering_triangles_l719_719344

-- Define the side lengths
def side_length_small : ℕ := 1
def side_length_large : ℕ := 12

-- Define the areas of the triangles
def area_equilateral_triangle (side_length : ℕ) : ℝ :=
  (math.sqrt 3 / 4) * side_length^2

-- Define the problem statement
theorem min_number_of_covering_triangles : 
  (area_equilateral_triangle side_length_large) / (area_equilateral_triangle side_length_small) = 144 :=
by
  sorry 

end min_number_of_covering_triangles_l719_719344


namespace bert_puzzle_days_l719_719410

noncomputable def words_per_pencil : ℕ := 1050
noncomputable def words_per_puzzle : ℕ := 75

theorem bert_puzzle_days : words_per_pencil / words_per_puzzle = 14 := by
  sorry

end bert_puzzle_days_l719_719410


namespace parallel_vectors_solution_l719_719538

theorem parallel_vectors_solution {x y : ℝ} : 
  let a := (2 * x, 1, 3) in
  let b := (1, -2 * y, 9) in
  (∃ λ : ℝ, a = λ • b) →
  x = 1 / 6 ∧ y = -3 / 2 := 
by
  sorry

end parallel_vectors_solution_l719_719538


namespace cost_price_computer_table_l719_719026

-- Define the variables
def cost_price : ℝ := 3840
def selling_price (CP : ℝ) := CP * 1.25

-- State the conditions and the proof problem
theorem cost_price_computer_table 
  (SP : ℝ) 
  (h1 : SP = 4800)
  (h2 : ∀ CP : ℝ, SP = selling_price CP) :
  cost_price = 3840 :=
by 
  sorry

end cost_price_computer_table_l719_719026


namespace sign_pyramid_top_plus_l719_719225

theorem sign_pyramid_top_plus (a b c d e : Int) (ha : a = 1 ∨ a = -1) (hb : b = 1 ∨ b = -1) (hc : c = 1 ∨ c = -1) (hd : d = 1 ∨ d = -1) (he : e = 1 ∨ e = -1) :
    ∃ (f : Finset (Int × Int × Int × Int × Int)), f.card = 11 ∧
    ∀ (x : Int × Int × Int × Int × Int), x ∈ f → 
    (x.1 * x.2 * x.3 * x.4 * x.5 = 1) :=
sorry

end sign_pyramid_top_plus_l719_719225


namespace evaluate_ratio_l719_719821

def factorial_a (n a : ℕ) : ℕ :=
  let k := n / a
  List.product (List.map (fun i => n - i * a) (List.range (k + 1)))

theorem evaluate_ratio : (factorial_a 36 5) / (factorial_a 10 3) = 40455072 :=
  by
    sorry

end evaluate_ratio_l719_719821


namespace complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l719_719521

variable (U A B : Set ℝ)
variable (x : ℝ)

def universal_set := { x | x ≤ 4 }
def set_A := { x | -2 < x ∧ x < 3 }
def set_B := { x | -3 < x ∧ x ≤ 3 }

theorem complement_U_A : (U \ A) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem intersection_A_B : (A ∩ B) = { x | -2 < x ∧ x < 3 } := sorry

theorem complement_U_intersection_A_B : (U \ (A ∩ B)) = { x | 3 ≤ x ∧ x ≤ 4 ∨ x ≤ -2 } := sorry

theorem complement_U_A_intersection_B : ((U \ A) ∩ B) = { x | -3 < x ∧ x ≤ -2 ∨ x = 3 } := sorry

end complement_U_A_intersection_A_B_complement_U_intersection_A_B_complement_U_A_intersection_B_l719_719521


namespace find_n_satifying_cos_eq_l719_719114

theorem find_n_satifying_cos_eq :
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ (cos (n * (π / 180)) = cos (1534 * (π / 180))) :=
by
  use 154
  split
  · norm_num
  split
  · norm_num
  sorry

end find_n_satifying_cos_eq_l719_719114


namespace right_triangle_area_l719_719648

theorem right_triangle_area (h : real) (a : real) (b : real) 
  (hypotenuse_eq : h = 13) 
  (leg_eq : a = 5) 
  (pythagorean : a^2 + b^2 = h^2) : 
  (1/2) * a * b = 30 := 
by 
  sorry

end right_triangle_area_l719_719648


namespace problem_solution_l719_719585

def sequence_reach_zero_count : ℕ := 
  let S := { (a1, a2, a3) // 1 ≤ a1 ∧ a1 ≤ 15 ∧ 1 ≤ a2 ∧ a2 ≤ 15 ∧ 1 ≤ a3 ∧ a3 ≤ 15 }
  let sequence (a1 a2 a3 : ℤ) := 
    λ n: ℕ, if n = 1 then a1 else
               if n = 2 then a2 else
               if n = 3 then a3 else
               sequence (n-1) * (abs (sequence (n-2) - sequence (n-3)))
  let reaches_zero := 
    λ (a1 a2 a3 : ℤ), ∃ n: ℕ, sequence a1 a2 a3 n = 0
  Set.card (Set.filter (λ (a1 a2 a3 : ℤ) in S, reaches_zero a1 a2 a3) S)

theorem problem_solution :
  sequence_reach_zero_count = 864 :=
sorry

end problem_solution_l719_719585


namespace inscribed_square_angle_l719_719567

theorem inscribed_square_angle (A B C D E : Type) 
  (square : square A B D C) (pentagon : regular_pentagon A B C D E) (shared_side : square.shared_side A B D C) :
  ∠ ABC = 27 :=
  sorry

end inscribed_square_angle_l719_719567


namespace proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l719_719279

variables (G : Type) [Group G] (kidney testis liver : G)
variables (SudanIII gentianViolet JanusGreenB dissociationFixative : G)

-- Conditions c1, c2, c3
def c1 : Prop := True -- Meiosis occurs in gonads, we simplify this in Lean to a true condition for brevity
def c2 : Prop := True -- Steps for slide preparation
def c3 : Prop := True -- Materials available

-- Questions
def q1 : G := testis
def q2 : G := dissociationFixative
def q3 : G := gentianViolet
def q4 : List G := [kidney, dissociationFixative, gentianViolet] -- Assume these are placeholders for correct cell types

-- Answers
def a1 : G := testis
def a2 : G := dissociationFixative
def a3 : G := gentianViolet
def a4 : List G := [testis, dissociationFixative, gentianViolet] -- Correct cells

-- Proving the equivalence of questions and answers given the conditions
theorem proof_q1_a1 : c1 ∧ c2 ∧ c3 → q1 = a1 := 
by sorry

theorem proof_q2_a2 : c1 ∧ c2 ∧ c3 → q2 = a2 := 
by sorry

theorem proof_q3_a3 : c1 ∧ c2 ∧ c3 → q3 = a3 := 
by sorry

theorem proof_q4_a4 : c1 ∧ c2 ∧ c3 → q4 = a4 := 
by sorry

end proof_q1_a1_proof_q2_a2_proof_q3_a3_proof_q4_a4_l719_719279


namespace tan_sum_identity_l719_719179

theorem tan_sum_identity
  (A B C : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) :
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

end tan_sum_identity_l719_719179


namespace range_of_lambda_is_correct_l719_719206

def vector (α : Type*) [Add α] := α × α

def acute_angle (a b : vector ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product > 0

noncomputable def range_of_lambda (a b : vector ℝ) : set ℝ :=
  {λ : ℝ | acute_angle a (a.1 + λ * b.1, a.2 + λ * b.2) ∧ λ ≠ 0}

theorem range_of_lambda_is_correct :
  let a := (1, 2) : vector ℝ
  let b := (1, 1) : vector ℝ
  range_of_lambda a b = {λ : ℝ | λ ∈ (-5/3 : ℝ, 0) ∪ (0, ∞)} :=
by
  sorry

end range_of_lambda_is_correct_l719_719206


namespace not_an_algorithm_option_B_l719_719694

def is_algorithm (description : String) : Prop :=
  description = "clear and finite steps to solve a problem producing correct results when executed by a computer"

def operation_to_string (option : Char) : String :=
  match option with
  | 'A' => "Calculating the area of a circle given its radius"
  | 'B' => "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
  | 'C' => "Finding the equation of a line given two points in the coordinate plane"
  | 'D' => "The rules of addition, subtraction, multiplication, and division"
  | _ => ""

noncomputable def categorize_operation (option : Char) : Prop :=
  option = 'B' ↔ ¬ is_algorithm (operation_to_string option)

theorem not_an_algorithm_option_B :
  categorize_operation 'B' :=
by
  sorry

end not_an_algorithm_option_B_l719_719694


namespace simplify_tan_product_l719_719988

-- Mathematical Conditions
def tan_inv (x : ℝ) : ℝ := sorry
noncomputable def tan (θ : ℝ) : ℝ := sorry

-- Problem statement to be proven
theorem simplify_tan_product (x y : ℝ) (hx : tan_inv x = 10) (hy : tan_inv y = 35) :
  (1 + x) * (1 + y) = 2 :=
sorry

end simplify_tan_product_l719_719988


namespace fraction_given_away_l719_719820

-- Define the conditions
def initial_lemons : ℕ := 12
def remaining_lemons : ℕ := 9
def given_lemons : ℕ := initial_lemons - remaining_lemons

-- Prove the fraction of lemons given away
theorem fraction_given_away : (given_lemons : ℚ) / initial_lemons = 1 / 4 :=
by
  have h_given_lemons : given_lemons = 3 := by arith_simps
  -- Simplify and prove the fraction
  rw [h_given_lemons]
  norm_num

end fraction_given_away_l719_719820


namespace expected_red_balls_in_B_l719_719671

-- Define the initial conditions
def initial_red_A := 4
def initial_white_A := 3
def initial_red_B := 3
def initial_white_B := 4

-- Define the total number of balls in each box initially
def total_A := initial_red_A + initial_white_A
def total_B := initial_red_B + initial_white_B

-- Define the probability of drawing each type of ball from each box
def prob_red_A := initial_red_A.toRational / total_A.toRational
def prob_white_A := initial_white_A.toRational / total_A.toRational
def prob_red_B := initial_red_B.toRational / total_B.toRational
def prob_white_B := initial_white_B.toRational / total_B.toRational

-- Define the probabilities of each scenario after transferring the balls
def prob_xi_2 := prob_white_A * (initial_red_B.toRational / (total_B + 1).toRational)
def prob_xi_4 := prob_red_A * (initial_white_B.toRational / (total_B + 1).toRational)
def prob_xi_3 := 1.toRational - prob_xi_2 - prob_xi_4

-- Define the expected value computation
def E_xi := 2.toRational * prob_xi_2 + 3.toRational * prob_xi_3 + 4.toRational * prob_xi_4

theorem expected_red_balls_in_B : E_xi = 25.toRational / 8.toRational :=
sorry

end expected_red_balls_in_B_l719_719671


namespace exterior_angle_DEF_l719_719290

noncomputable def angle_DEF : ℝ := 117

theorem exterior_angle_DEF 
  (pentagon_internal_angle : ℝ = 108)
  (octagon_internal_angle : ℝ = 135)
  (sum_of_angles_around_E : ∑ (i : Fin 3), (fin_cases[ℝ → ℝ] [pentagon_internal_angle, octagon_internal_angle, angle_DEF]) = 360) :
  angle_DEF = 117 := 
by 
  sorry

end exterior_angle_DEF_l719_719290


namespace combined_jelly_beans_ratio_is_29_percent_l719_719847

def jelly_bean_bag : Type :=
  {total_beans : ℕ, yellow_percentage : ℕ}

def bag_A : jelly_bean_bag := {total_beans := 22, yellow_percentage := 60}
def bag_B : jelly_bean_bag := {total_beans := 33, yellow_percentage := 30}
def bag_C : jelly_bean_bag := {total_beans := 35, yellow_percentage := 25}
def bag_D : jelly_bean_bag := {total_beans := 40, yellow_percentage := 15}

theorem combined_jelly_beans_ratio_is_29_percent :
  let yellow_A := (bag_A.total_beans * bag_A.yellow_percentage) / 100
      yellow_B := (bag_B.total_beans * bag_B.yellow_percentage) / 100
      yellow_C := (bag_C.total_beans * bag_C.yellow_percentage) / 100
      yellow_D := (bag_D.total_beans * bag_D.yellow_percentage) / 100
      total_yellow := yellow_A + yellow_B + yellow_C + yellow_D
      total_beans := bag_A.total_beans + bag_B.total_beans + bag_C.total_beans + bag_D.total_beans
  in (total_yellow * 100 / total_beans ≈ 29) :=
by
  sorry

end combined_jelly_beans_ratio_is_29_percent_l719_719847


namespace sum_series_eq_two_l719_719799

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l719_719799


namespace platform_length_l719_719378

theorem platform_length (speed_kmph : ℝ) (cross_time_seconds : ℝ) (train_length_meters : ℝ) : 
  speed_kmph = 72 → cross_time_seconds = 26 → train_length_meters = 230.0416 → 
  (speed_kmph * (5/18) * cross_time_seconds - train_length_meters = 289.9584) :=
by
  intros h_speed h_time h_train_length
  have h_speed_mps : speed_kmph * (5/18) = 20 := by rw [h_speed, mul_div_cancel' 72 18 (5 : ℝ)], norm_num
  have distance_covered : speed_kmph * (5/18) * cross_time_seconds = 520 := by rw [h_speed_mps,← h_time, mul_assoc], norm_num
  have h_platform_length_eq : distance_covered - train_length_meters = 289.9584 := by rw [distance_covered, h_train_length], norm_num
  exact h_platform_length_eq

end platform_length_l719_719378


namespace infinite_series_sum_l719_719444

theorem infinite_series_sum :
  (∑ k in Nat, (k^2 : ℝ)/(3^k : ℝ)) = 6 := sorry

end infinite_series_sum_l719_719444


namespace alpha_value_l719_719533

theorem alpha_value {α : ℝ} (h1 : 0 < α ∧ α < π / 2)
  (h2 : ∀ x : ℝ, sin(2 * x + π / 4 + α) = sin(-2 * x + π / 4 + α)) :
  α = π / 4 :=
sorry

end alpha_value_l719_719533


namespace trip_savings_l719_719335

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end trip_savings_l719_719335


namespace area_equality_l719_719658

def is_equi_triangle (A P Q : Point) : Prop := dist A P = dist A Q ∧ dist A P = dist P Q

def is_rectangle (A B C D : Point) : Prop := 
  ∧ (dist A B = dist C D)
  ∧ (dist B C = dist D A)
  ∧ (right_angle A B C)
  ∧ (right_angle B C D)
  ∧ (right_angle C D A)
  ∧ (right_angle D A B)

def on_side (P B C Q D : Point) : Prop :=
  ∧ is_on_segment P B C
  ∧ is_on_segment Q D C

theorem area_equality (A B C D P Q : Point) (h1 : is_rectangle A B C D)
                                   (h2: on_side P B C Q D) 
                                   (h3: is_equi_triangle A P Q) : 
  area (triangle AQD) + area (triangle ABP) = area (triangle PCQ) := 
sorry

end area_equality_l719_719658


namespace series_sum_l719_719448

-- Define the infinite series
def series := ∑' k : ℕ, (k^2 : ℝ) / 3^k

-- State the theorem to prove
theorem series_sum : series = 1 := 
sorry

end series_sum_l719_719448


namespace perimeter_ratio_of_square_and_constructed_quadrilateral_l719_719282

theorem perimeter_ratio_of_square_and_constructed_quadrilateral (x : ℝ) (hx : 0 < x) :
  let P0 := 4 * x,
      PEMFN := 4 * x * real.sqrt (2 + real.sqrt 3) in
  PEMFN / P0 = (real.sqrt 6 + real.sqrt 2) / 2 :=
by sorry

end perimeter_ratio_of_square_and_constructed_quadrilateral_l719_719282


namespace remaining_volume_of_modified_cube_l719_719739

-- Definitions from the conditions
def side_length_cube : ℝ := 6
def radius_cylinder : ℝ := 3
def height_cylinder : ℝ := 6

-- Correct answer based on the solution steps
def correct_remaining_volume : ℝ := 216 - 54 * real.pi

-- The statement of the problem as a theorem
theorem remaining_volume_of_modified_cube :
  let V_cube := side_length_cube^3
  let V_cylinder := real.pi * radius_cylinder^2 * height_cylinder
  V_cube - V_cylinder = correct_remaining_volume :=
by
  sorry

end remaining_volume_of_modified_cube_l719_719739


namespace common_face_sum_is_9_l719_719655

noncomputable def common_sum (vertices : Fin 9 → ℕ) : ℕ :=
  let total_sum := (Finset.sum (Finset.univ : Finset (Fin 9)) vertices)
  let additional_sum := 9
  let total_with_addition := total_sum + additional_sum
  total_with_addition / 6

theorem common_face_sum_is_9 :
  ∀ (vertices : Fin 9 → ℕ), (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 9) →
  Finset.sum (Finset.univ : Finset (Fin 9)) vertices = 45 →
  common_sum vertices = 9 := 
by
  intros vertices h1 h_sum
  unfold common_sum
  sorry

end common_face_sum_is_9_l719_719655


namespace local_extrema_range_of_k_l719_719274

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x + 5

theorem local_extrema :
  (∀ x : ℝ, (f.derivative x = 0) → (x = - (real.sqrt 2) ∨ x = real.sqrt 2)) ∧ 
  f.derivative (- (real.sqrt 2)) = 0 ∧ 
  (∀ h : x = - (real.sqrt 2), is_local_max f x) ∧ 
  f.derivative (real.sqrt 2) = 0 ∧ 
  (∀ h : x = real.sqrt 2, is_local_min f x) := sorry

noncomputable def g (x : ℝ) : ℝ := (x^3 - 6 * x + 5) / (x - 1)

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, 1 < x → f x ≥ k * (x - 1)) ↔ k ≤ -3 := sorry

end local_extrema_range_of_k_l719_719274


namespace arc_intercept_length_l719_719756

noncomputable def side_length : ℝ := 4
noncomputable def diagonal_length : ℝ := Real.sqrt (side_length^2 + side_length^2)
noncomputable def radius : ℝ := diagonal_length / 2
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def arc_length_one_side : ℝ := circumference / 4

theorem arc_intercept_length :
  arc_length_one_side = Real.sqrt 2 * Real.pi :=
by
  sorry

end arc_intercept_length_l719_719756


namespace iterative_average_difference_l719_719399
-- Import necessary math library

-- Definition of iterative average calculation
def iterative_average : List ℕ → ℝ
| []       => 0
| [a]      => a
| (a :: b) => (a + iterative_average b) / 2

-- Definition of maximum and minimum iterative averages for the given list
def max_iterative_average : ℝ :=
  iterative_average [1, 2, 3, 4, 5, 6]

def min_iterative_average : ℝ :=
  iterative_average [6, 5, 4, 3, 2, 1]

-- Target theorem to prove
theorem iterative_average_difference :
  max_iterative_average - min_iterative_average = 3.0625 := by
  sorry

end iterative_average_difference_l719_719399


namespace bullet_train_speed_is_70kmph_l719_719721

noncomputable def bullet_train_speed (train_length time_man  : ℚ) (man_speed_kmph : ℕ) : ℚ :=
  let man_speed_ms : ℚ := man_speed_kmph * 1000 / 3600
  let relative_speed : ℚ := train_length / time_man
  let train_speed_ms : ℚ := relative_speed - man_speed_ms
  train_speed_ms * 3600 / 1000

theorem bullet_train_speed_is_70kmph :
  bullet_train_speed 160 7.384615384615384 8 = 70 :=
by {
  -- Proof is omitted
  sorry
}

end bullet_train_speed_is_70kmph_l719_719721


namespace inequality_true_for_all_alpha_l719_719293

noncomputable def sin3alpha (alpha : ℝ) : ℝ := 3 * sin alpha - 4 * sin alpha ^ 3
noncomputable def cos2alpha (alpha : ℝ) : ℝ := 2 * cos alpha ^ 2 - 1

theorem inequality_true_for_all_alpha :
  ∀ α : ℝ, 4 * sin3alpha α + 5 ≥ 4 * cos2alpha α + 5 * sin α :=
by 
  sorry

end inequality_true_for_all_alpha_l719_719293


namespace tetrahedron_base_count_l719_719060

theorem tetrahedron_base_count (A B C D : Type) [IsTetrahedron A B C D] : 
  num_triangular_faces A B C D = 4 := 
by
  sorry

end tetrahedron_base_count_l719_719060


namespace find_circle_radius_l719_719724

/-!
  Given an isosceles triangle ABC with AB = BC, a circle with radius r lies on AC
  such that it is tangent to sides AB and BC and divides AC into three equal parts.
  If the area of triangle ABC is 9 * sqrt 2, then prove that r = 2 * sqrt 2.
-/
theorem find_circle_radius
  (ABC : Triangle)
  (isosceles : ABC.isIsosceles)
  (circle_center_on_AC : Center_on_AC)
  (tangent_to_AB : Circle_is_tangent_to_AB)
  (tangent_to_BC : Circle_is_tangent_to_BC)
  (divides_AC_into_equal_parts : Divides_AC_into_three_equal_parts)
  (area_of_triangle_ABC : Area ABC = 9 * Real.sqrt 2) :
  radius circle = 2 * Real.sqrt 2 := 
sorry

end find_circle_radius_l719_719724


namespace trip_early_movie_savings_l719_719331

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end trip_early_movie_savings_l719_719331


namespace log_travel_time_24_l719_719938

noncomputable def time_for_log_to_travel (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) : ℝ :=
  D / v

theorem log_travel_time_24 (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) :
  time_for_log_to_travel D u v h1 h2 = 24 :=
sorry

end log_travel_time_24_l719_719938


namespace simplify_complex_expression_l719_719992

theorem simplify_complex_expression :
  (complex.normSq ((-1 + complex.I) / 2) ^ 8 + complex.normSq ((-1 - complex.I) / 2) ^ 8 = 1 / 8) :=
by 
  sorry

end simplify_complex_expression_l719_719992


namespace fraction_value_l719_719535

theorem fraction_value (m n : ℤ) (h : (m - 8) * (m - 8) + abs (n + 6) = 0) : n / m = -(3 / 4) :=
by sorry

end fraction_value_l719_719535


namespace beef_original_weight_l719_719356

noncomputable def originalWeightBeforeProcessing (weightAfterProcessing : ℝ) (lossPercentage : ℝ) : ℝ :=
  weightAfterProcessing / (1 - lossPercentage / 100)

theorem beef_original_weight : originalWeightBeforeProcessing 570 35 = 876.92 :=
by
  sorry

end beef_original_weight_l719_719356


namespace midline_theorem_medians_divide_into_four_equal_triangles_l719_719289

-- Definitions of the elements in the conditions
variables {A B C D E F : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

-- Assume D, E, and F are midpoints of sides AB, BC, and CA respectively.
variable (midpoint_D : is_midpoint D A B)
variable (midpoint_E : is_midpoint E B C)
variable (midpoint_F : is_midpoint F C A)

-- The midline theorem
theorem midline_theorem (D E F : MetricSpace) (hD : is_midpoint D A B) (hE : is_midpoint E B C) (hF : is_midpoint F C A) : 
  parallel D E C A ∧ length D E = length C A / 2 ∧ 
  parallel E F A B ∧ length E F = length A B / 2 ∧ 
  parallel F D B C ∧ length F D = length B C / 2 := sorry

theorem medians_divide_into_four_equal_triangles (A B C D E F : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (midpoint_D : is_midpoint D A B)
  (midpoint_E : is_midpoint E B C)
  (midpoint_F : is_midpoint F C A) :
  area (triangle A F D) = area (triangle D B E) ∧ area (triangle E C F) ∧ area (triangle D E F) :=
  sorry

end midline_theorem_medians_divide_into_four_equal_triangles_l719_719289


namespace max_term_sequence_l719_719880

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℕ := 3^(n-1)

theorem max_term_sequence  :
  ∃ n : ℕ, n = 14 ∧
  ∀ k : ℕ, k ∈ {1 .. 27} →
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 14 → (a_n i - 25) / b_n i ≤ (a_n k - 25) / b_n k) ∧
    (∀ i : ℕ, 14 ≤ i ∧ i ≤ 27 → (a_n i - 25) / b_n i ≤ (a_n k - 25) / b_n k) :=
sorry

end max_term_sequence_l719_719880


namespace find_m_value_l719_719500

open Nat

theorem find_m_value {m : ℕ} (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 24 :=
  sorry

end find_m_value_l719_719500


namespace gravitational_force_on_moon_l719_719646

theorem gravitational_force_on_moon :
  ∀ (d_earth d_moon : ℕ) (f_earth f_moon k : ℝ), 
  d_earth = 5000 → f_earth = 800 → k = f_earth * d_earth^2 → 
  d_moon = 250000 → k = f_moon * d_moon^2 → 
  f_moon = 8 / 25 :=
by 
  intros d_earth d_moon f_earth f_moon k h_de hf_earth h_k h_dm he_k,
  sorry

end gravitational_force_on_moon_l719_719646


namespace polar_to_cartesian_equiv_l719_719093

noncomputable def polar_to_cartesian (rho θ : ℝ) : ℝ × ℝ :=
  (rho * Real.cos θ, rho * Real.sin θ)

theorem polar_to_cartesian_equiv (ρ θ : ℝ) (h : ρ = 4 * Real.cos θ) :
  let ⟨x, y⟩ := polar_to_cartesian ρ θ in
  (x - 2)^2 + y^2 = 4 :=
by
  sorry

end polar_to_cartesian_equiv_l719_719093


namespace largest_fraction_of_three_l719_719878

theorem largest_fraction_of_three (a b c : Nat) (h1 : Nat.gcd a 6 = 1)
  (h2 : Nat.gcd b 15 = 1) (h3 : Nat.gcd c 20 = 1)
  (h4 : (a * b * c) = 60) :
  max (a / 6) (max (b / 15) (c / 20)) = 5 / 6 :=
by
  sorry

end largest_fraction_of_three_l719_719878


namespace percentage_rate_of_stock_l719_719389

-- Define the stock price and yield
def stock_price : ℝ := 250
def stock_yield : ℝ := 0.08

-- Define the annual dividend
def annual_dividend : ℝ := stock_yield * stock_price

-- Define the dividend yield calculation
def dividend_yield (annual_dividend price : ℝ) : ℝ :=
  (annual_dividend / price) * 100

theorem percentage_rate_of_stock :
  dividend_yield annual_dividend stock_price = 8 :=
by
  sorry

end percentage_rate_of_stock_l719_719389


namespace find_m_value_l719_719139

theorem find_m_value :
  ∃ m : ℕ, 144^5 + 121^5 + 95^5 + 30^5 = m^5 ∧ m = 159 := by
  use 159
  sorry

end find_m_value_l719_719139


namespace total_cupcakes_l719_719672

theorem total_cupcakes (cupcakes_per_event : ℕ) (events : ℕ) (total_cupcakes : ℕ) :
  cupcakes_per_event = 156 → events = 12 → total_cupcakes = cupcakes_per_event * events → total_cupcakes = 1872 :=
by
  intros
  subst_vars
  norm_num
  sorry

end total_cupcakes_l719_719672


namespace ordering_of_numbers_l719_719683

theorem ordering_of_numbers : 3^15 < 2^30 ∧ 2^30 < 10^10 :=
by
  -- We'll prove these inequalities one by one
  { sorry

end ordering_of_numbers_l719_719683


namespace c_minus_a_equals_90_l719_719024

variable (a b c : ℝ)

def average_a_b (a b : ℝ) : Prop := (a + b) / 2 = 45
def average_b_c (b c : ℝ) : Prop := (b + c) / 2 = 90

theorem c_minus_a_equals_90
  (h1 : average_a_b a b)
  (h2 : average_b_c b c) :
  c - a = 90 :=
  sorry

end c_minus_a_equals_90_l719_719024


namespace vampires_after_two_nights_l719_719681

-- Define the initial conditions and calculations
def initial_vampires : ℕ := 2
def transformation_rate : ℕ := 5
def first_night_vampires : ℕ := initial_vampires * transformation_rate + initial_vampires
def second_night_vampires : ℕ := first_night_vampires * transformation_rate + first_night_vampires

-- Prove that the number of vampires after two nights is 72
theorem vampires_after_two_nights : second_night_vampires = 72 :=
by sorry

end vampires_after_two_nights_l719_719681


namespace common_factor_of_polynomial_l719_719244

noncomputable def polynomial_common_factor (m : ℤ) : ℤ :=
  let polynomial := 2 * m^3 - 8 * m
  let common_factor := 2 * m
  common_factor  -- We're stating that the common factor is 2 * m

-- The theorem to verify that the common factor of each term in the polynomial is 2m
theorem common_factor_of_polynomial (m : ℤ) : 
  polynomial_common_factor m = 2 * m := by
  sorry

end common_factor_of_polynomial_l719_719244


namespace function_properties_l719_719599

variable (f : ℝ → ℝ)

theorem function_properties (h1 : ∀ x, f (10 + x) = f (10 - x)) 
                            (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
                            (∀ x, f (-x) = -f (x)) ∧ (∀ x, f(x) = f(x + 40)) :=
by
  sorry

end function_properties_l719_719599


namespace expression_value_l719_719907

theorem expression_value (x y : ℝ) (h : x + y = -1) : x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l719_719907


namespace at_least_one_distinct_root_l719_719257

theorem at_least_one_distinct_root {a b : ℝ} (ha : a > 4) (hb : b > 4) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a * x₁ + b = 0 ∧ x₂^2 + a * x₂ + b = 0) ∨
    (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + b * y₁ + a = 0 ∧ y₂^2 + b * y₂ + a = 0) :=
sorry

end at_least_one_distinct_root_l719_719257


namespace union_problem1_union_problem2_l719_719609

variable (U A B : Set ℕ)
variable [decidable_eq ℕ]

def U := {1, 2, 3, 4, 5, 6, 7}
def A := {1, 3, 5, 7}
def B := {3, 5}

theorem union_problem1 : A ∪ B = {1, 3, 5, 7} := by
  sorry

theorem union_problem2 : (U \ A) ∪ B = {2, 3, 4, 5, 6} := by
  sorry

end union_problem1_union_problem2_l719_719609


namespace max_value_expression_l719_719312

theorem max_value_expression : 
  ∀ (x y : ℝ), ∃ M : ℝ, M = 2 * y^2 - y^4 - x^2 - 3 * x ∧ M ≤ 13 / 4 :=
begin
  sorry
end

end max_value_expression_l719_719312


namespace molly_miles_per_day_l719_719969

theorem molly_miles_per_day (total_miles : ℕ) (total_years : ℕ) (days_per_year : ℕ) : 
  total_miles = 3285 → total_years = 3 → days_per_year = 365 → 
  total_miles / (total_years * days_per_year) = 3 :=
by
  intros h1 h2 h3
  calc
    total_miles / (total_years * days_per_year)
      = 3285 / (3 * 365) : by rw [h1, h2, h3]
  ... = 3 : sorry

end molly_miles_per_day_l719_719969


namespace sum_consecutive_powers_of_2_divisible_by_6_l719_719981

theorem sum_consecutive_powers_of_2_divisible_by_6 (n : ℕ) :
  ∃ k : ℕ, 2^n + 2^(n+1) = 6 * k :=
sorry

end sum_consecutive_powers_of_2_divisible_by_6_l719_719981


namespace no_such_function_exists_l719_719619

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f(f(x)) = x^2 - 2 :=
  sorry

end no_such_function_exists_l719_719619


namespace angle_A_range_l719_719558

-- Given an acute-angled triangle ABC
variables {A B C : Type} [inner_product_space ℝ (euclidean_space ℝ (fin 3))]
variables {a b c : vector ℝ (fin 3)}
hypothesis h1 : inner a b > 0 ∧ inner b c > 0 ∧ inner c a > 0 -- acute angles

-- Define the cevians intersecting at a single point
variables (AD BM CH : line ℝ (euclidean_space ℝ (fin 3)))
hypothesis h2 : AD ≠ BM ∧ BM ≠ CH ∧ CH ≠ AD -- different lines
hypothesis h3 : ∃ P, AD.contains P ∧ BM.contains P ∧ CH.contains P -- intersect at a single point P

-- Prove the range of angle A
theorem angle_A_range (h_acute : (angle a b c < π / 2) ∧ (angle b c a < π / 2) ∧ (angle c a b < π / 2)) :
  51.8333 * (π/180) < angle a b c ∧ angle a b c < π / 2 := by
  sorry

end angle_A_range_l719_719558


namespace shooter_scores_l719_719929

theorem shooter_scores
    (x y z : ℕ)
    (hx : x + y + z > 11)
    (hscore: 8 * x + 9 * y + 10 * z = 100) :
    (x + y + z = 12) ∧ ((x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end shooter_scores_l719_719929


namespace length_of_diagonal_of_regular_octagon_l719_719463

noncomputable def length_of_da (s : ℕ) := s * Real.sqrt 2

theorem length_of_diagonal_of_regular_octagon (s : ℕ) 
  (h_s : s = 12) :
  length_of_da s = 12 * Real.sqrt 2 :=
by
  unfold length_of_da
  rw h_s
  simp
  sorry

end length_of_diagonal_of_regular_octagon_l719_719463


namespace part1_increasing_on_neg1_1_part2_range_of_t_l719_719514

-- Define the function and its property conditions
variable (a b : ℝ) (h_pos : a > 0)
def f (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Assumption that f is an odd function
lemma f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Proof statement for part (1): f is increasing on (-1, 1)
theorem part1_increasing_on_neg1_1 :
  ∀ x1 x2 ∈ set.Ioo (-1 : ℝ) 1, x1 < x2 → f x1 < f x2 := by sorry

-- Proof statement for part (2): finding the range of t
theorem part2_range_of_t :
  ∀ t : ℝ, 0 < t ∧ t < (2 / 3) ↔ f (2 * t - 1) + f (t - 1) < 0 := by sorry

end part1_increasing_on_neg1_1_part2_range_of_t_l719_719514


namespace initial_investment_amount_l719_719068

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment_amount (P A r t : ℝ) (n : ℕ) (hA : A = 992.25) 
  (hr : r = 0.10) (hn : n = 2) (ht : t = 1) : P = 900 :=
by
  have h : compoundInterest P r n t = A := by sorry
  rw [hA, hr, hn, ht] at h
  simp at h
  exact sorry

end initial_investment_amount_l719_719068


namespace arrangement_of_teachers_l719_719038

theorem arrangement_of_teachers : 
  let teachers := 5
  let schools := 3
  let school_one_teachers := 1
  let school_two_teachers := 2
  let school_three_teachers := 2
  ∃ arrangements : ℕ, arrangements = (Nat.choose teachers school_one_teachers) *
                                   (Nat.choose (teachers - school_one_teachers) school_two_teachers) *
                                   (Nat.choose (teachers - school_one_teachers - school_two_teachers) school_three_teachers) /
                                   (Nat.factorial 2) * (Nat.factorial schools) ∧ 
                                   arrangements = 90 :=
by
  have h1 : teachers = 5 := rfl
  have h2 : schools = 3 := rfl
  have h3 : school_one_teachers = 1 := rfl
  have h4 : school_two_teachers = 2 := rfl
  have h5 : school_three_teachers = 2 := rfl
  use (Nat.choose teachers school_one_teachers) *
      (Nat.choose (teachers - school_one_teachers) school_two_teachers) *
      (Nat.choose (teachers - school_one_teachers - school_two_teachers) school_three_teachers) /
      (Nat.factorial 2) * (Nat.factorial schools)
  split
  · sorry
  · sorry

end arrangement_of_teachers_l719_719038


namespace perimeter_parallelogram_l719_719213

variable (A B C D : Type) [AddGroup A]
variable (AB BC CD DA : A)
variable (P : A)

def AB_eq_2 (AB : A) : Prop := AB = sorry -- Define given information AB = 2

def parallelogram_perimeter (AB BC CD DA : A) : A :=
  (AB + BC + CD + DA)

theorem perimeter_parallelogram (AB BC CD DA : A) (h : AB_eq_2 AB) :
  parallelogram_perimeter AB BC CD DA = 6 :=
sorry

end perimeter_parallelogram_l719_719213


namespace total_drink_volume_l719_719575

theorem total_drink_volume (coke_parts sprite_parts mtndew_parts : ℕ) (coke_volume : ℕ) :
  coke_parts = 2 → sprite_parts = 1 → mtndew_parts = 3 → coke_volume = 6 →
  (coke_volume / coke_parts) * (coke_parts + sprite_parts + mtndew_parts) = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end total_drink_volume_l719_719575


namespace student_correct_answers_l719_719701

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 70) : C = 90 :=
sorry

end student_correct_answers_l719_719701


namespace max_area_of_rectangle_l719_719747

theorem max_area_of_rectangle (x : ℝ) : 
  (sin x * cos x) ≤ 1/2 :=
by {
  sorry
}

end max_area_of_rectangle_l719_719747


namespace find_m_l719_719503
open Nat

theorem find_m (m : ℕ) (hm : m > 0) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 24 := 
by
  sorry

end find_m_l719_719503


namespace work_is_two_l719_719339

def log2 : ℝ := Real.log 2
def log5 : ℝ := Real.log 5

def F1 : ℝ × ℝ := (log2, log2)
def F2 : ℝ × ℝ := (log5, log2)
def S : ℝ × ℝ := (2 * log5, 1)
def F : ℝ × ℝ := (F1.1 + F2.1, F1.2 + F2.2)

def work_done (F : ℝ × ℝ) (S : ℝ × ℝ) : ℝ :=
  F.fst * S.fst + F.snd * S.snd

theorem work_is_two : work_done F S = 2 := by
  sorry

end work_is_two_l719_719339


namespace infinite_series_converges_to_3_l719_719788

noncomputable def sum_of_series := ∑' k in (Finset.range ∞).filter (λ k, k > 0), 
  (8 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))

theorem infinite_series_converges_to_3 : sum_of_series = 3 := 
  sorry

end infinite_series_converges_to_3_l719_719788


namespace boxed_boxed_11_l719_719134

-- Define what it means to be the boxed sum of factors of a number.
def sum_factors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (n % · = 0).foldr (· + ·) 0

-- Proving the specific equality for boxed boxed 11.
theorem boxed_boxed_11 : sum_factors (sum_factors 11) = 28 := by
  sorry

end boxed_boxed_11_l719_719134


namespace sin_double_angle_l719_719478

theorem sin_double_angle : ∀ (α : ℝ), (0 < α ∧ α < π) → (cos α = -12 / 13) → (sin (2 * α) = -120 / 169) := 
begin
  intros α hα hcos,
  sorry
end

end sin_double_angle_l719_719478


namespace frog_vertical_boundary_prob_l719_719376

-- Define the type of points on the grid
structure Point where
  x : ℕ
  y : ℕ

-- Define the type of the rectangle
structure Rectangle where
  left_bottom : Point
  right_top : Point

-- Conditions
def start_point : Point := ⟨2, 3⟩
def boundary : Rectangle := ⟨⟨0, 0⟩, ⟨5, 5⟩⟩

-- Define the probability function
noncomputable def P (p : Point) : ℚ := sorry

-- Symmetry relations and recursive relations
axiom symmetry_P23 : P ⟨2, 3⟩ = P ⟨3, 3⟩
axiom symmetry_P22 : P ⟨2, 2⟩ = P ⟨3, 2⟩
axiom recursive_P23 : P ⟨2, 3⟩ = 1 / 4 + 1 / 4 * P ⟨2, 2⟩ + 1 / 4 * P ⟨1, 3⟩ + 1 / 4 * P ⟨3, 3⟩

-- Main Theorem
theorem frog_vertical_boundary_prob :
  P start_point = 2 / 3 := sorry

end frog_vertical_boundary_prob_l719_719376


namespace infinite_sum_problem_l719_719807

theorem infinite_sum_problem : 
  (∑ k in (set.Ici 1), (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 1 :=
by
  sorry

end infinite_sum_problem_l719_719807


namespace repeating_8_minus_6_as_fraction_l719_719107

noncomputable def decimal_repeating_8_to_frac : ℝ :=
  let x := 0.888888888... in x

noncomputable def decimal_repeating_6_to_frac : ℝ :=
  let y := 0.666666666... in y

theorem repeating_8_minus_6_as_fraction :
  let x := decimal_repeating_8_to_frac in
  let y := decimal_repeating_6_to_frac in
  (x - y) = (2/9) :=
by sorry

end repeating_8_minus_6_as_fraction_l719_719107


namespace range_of_g_l719_719433

theorem range_of_g (A : ℝ) (h : ¬ ∃ n : ℤ, A = n * (π / 2)) :
  3 < (sin A * (5 * (cos A)^2 + (tan A)^2 + 2 * (sin A)^2 + (sin A)^2 * (cos A)^2)) / 
      (tan A * (csc A - sin A * tan A)) < 8 :=
sorry

end range_of_g_l719_719433


namespace emails_in_inbox_l719_719631

theorem emails_in_inbox :
  let total_emails := 400
  let trash_emails := total_emails / 2
  let work_emails := 0.4 * (total_emails - trash_emails)
  total_emails - trash_emails - work_emails = 120 :=
by
  sorry

end emails_in_inbox_l719_719631


namespace a_n_formula_S_n_sum_l719_719189

variable (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (S_n : ℕ → ℝ)
variable (d : ℝ)

-- Given conditions
axiom arithmetic_seq (n : ℕ) : a_n n = a_n 1 + (n - 1) * d
axiom a_2 : a_n 2 = 3
axiom log_arithmetic_seq : ∀ {a₁ a₃ a₇ : ℝ}, log 2 a₃ = (log 2 a₁ + log 2 a₇) / 2 → 
  a₃ = (a₁ * a₇) ** (1 / 2)

-- Part (1) goal:
theorem a_n_formula : a_n = λ n, n + 1 :=
sorry

-- Part (2) goal:
variable (a_n_formula : ℕ → ℝ)
axiom b_n_definition (n : ℕ) : b_n n = 1 / (a_n_formula n * a_n_formula (n + 1))
theorem S_n_sum : S_n n = n / (2 * (n + 2)) :=
sorry

end a_n_formula_S_n_sum_l719_719189


namespace total_profit_calculation_l719_719700

variables {I_B T_B : ℝ}

-- Conditions as definitions
def investment_A (I_B : ℝ) : ℝ := 3 * I_B
def period_A (T_B : ℝ) : ℝ := 2 * T_B
def profit_B (I_B T_B : ℝ) : ℝ := I_B * T_B
def total_profit (I_B T_B : ℝ) : ℝ := 7 * I_B * T_B

-- To prove
theorem total_profit_calculation
  (h1 : investment_A I_B = 3 * I_B)
  (h2 : period_A T_B = 2 * T_B)
  (h3 : profit_B I_B T_B = 4000)
  : total_profit I_B T_B = 28000 := by
  sorry

end total_profit_calculation_l719_719700


namespace original_average_speed_l719_719391

theorem original_average_speed :
  ∀ (D : ℝ),
  (V = D / (5 / 6)) ∧ (60 = D / (2 / 3)) → V = 48 :=
by
  sorry

end original_average_speed_l719_719391


namespace Petya_achieves_config_l719_719398

theorem Petya_achieves_config (n : ℕ) :
  (∃ cells : Fin 9 → ℕ, (∀ a b : Fin 9, adjacency a b → (cells a = cells b) ∨ (abs (cells a - cells b) = 1))
      ∧ set.range cells = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8}) ↔ n = 2 :=
by
  sorry

end Petya_achieves_config_l719_719398


namespace find_m_if_line_parallel_plane_l719_719879

variables (m : ℝ)

def direction_vector_line := (2, m, 1)
def normal_vector_plane := (1, 1/2, 2)

-- Given that l is parallel to alpha, the direction vector of line is perpendicular to
-- the normal vector of the plane, i.e., their dot product is zero.
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := 
  (v1.1 * v2.1) + (v1.2 * v2.2) + (v1.3 * v2.3)

-- Rewrite the problem as a true equality in Lean
theorem find_m_if_line_parallel_plane
  (h : dot_product (direction_vector_line m) normal_vector_plane = 0) : 
  m = -8 :=
sorry

end find_m_if_line_parallel_plane_l719_719879


namespace smallest_prime_angle_l719_719557

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_angle :
  ∃ (x : ℕ), is_prime x ∧ is_prime (2 * x) ∧ x + 2 * x = 90 ∧ x = 29 :=
by sorry

end smallest_prime_angle_l719_719557


namespace sphere_surface_area_l719_719862

noncomputable def surface_area_of_sphere : ℝ := 4 * Real.pi * (16 / 3)

theorem sphere_surface_area (R : ℝ) (A B C : EuclideanSpace ℝ (Fin 3))
  (h1 : dist (EuclideanSpace.origin ℝ (Fin 3)) (affineSpan ℝ {A, B, C}) = R / 2)
  (h2 : dist A B = 2)
  (h3 : dist A C = 2)
  (h4: ∠BAC = 120) : surface_area_of_sphere = (64 / 3) * Real.pi :=
by
  sorry

end sphere_surface_area_l719_719862


namespace find_a_l719_719906

variables {a b c : ℤ}

theorem find_a (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 5) : a = 3 :=
by
  sorry

end find_a_l719_719906


namespace set_equality_l719_719520

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 4})
variable (hB : B = {3, 4})

theorem set_equality : ({2, 5} : Set ℕ) = U \ (A ∪ B) :=
by
  sorry

end set_equality_l719_719520


namespace sulfur_mass_percentage_is_64_l719_719456

-- Define the molar masses of aluminum and sulfur
def molar_mass_Al : ℝ := 26.98
def molar_mass_S : ℝ := 32.06

-- Define the chemical formula molar mass calculation
def molar_mass_Al2S3 : ℝ := (2 * molar_mass_Al) + (3 * molar_mass_S)

-- Calculate the mass percentage of sulfur in Al2S3
def mass_percentage_S_in_Al2S3 : ℝ := (3 * molar_mass_S / molar_mass_Al2S3) * 100

-- The proof problem is to show that this mass percentage is approximately 64%
theorem sulfur_mass_percentage_is_64 :
  abs (mass_percentage_S_in_Al2S3 - 64) < 0.1 := 
by
  -- Proof can be given here
  sorry

end sulfur_mass_percentage_is_64_l719_719456


namespace find_m_l719_719502
open Nat

theorem find_m (m : ℕ) (hm : m > 0) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 24 := 
by
  sorry

end find_m_l719_719502


namespace sum_reciprocal_a2i_lt_half_l719_719263

def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - n

def a (n : ℕ) : ℕ := 2^n - 1

theorem sum_reciprocal_a2i_lt_half (n : ℕ) : 
  (∑ i in Finset.range n, (1 : ℝ) / a (2 * (i + 1))) < 1 / 2 := sorry

end sum_reciprocal_a2i_lt_half_l719_719263


namespace find_constants_l719_719249

theorem find_constants (a b c : ℝ) :
  (∀ n : ℕ, 0 < n → (∑ i in finset.range n.succ, i * (n^2 - i^2)) = a * n^4 + b * n^2 + c) ↔
  a = 1 / 4 ∧ b = -1 / 4 ∧ c = 0 :=
begin
  sorry
end

end find_constants_l719_719249


namespace man_is_older_l719_719743

-- Define present age of the son
def son_age : ℕ := 26

-- Define present age of the man (father)
axiom man_age : ℕ

-- Condition: in two years, the man's age will be twice the age of his son
axiom age_condition : man_age + 2 = 2 * (son_age + 2)

-- Prove that the man is 28 years older than his son
theorem man_is_older : man_age - son_age = 28 := sorry

end man_is_older_l719_719743


namespace truncated_cone_resistance_l719_719738

theorem truncated_cone_resistance (a b h : ℝ) (ρ : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h_pos : 0 < h) :
  (∫ x in (0:ℝ)..h, ρ / (π * ((a + x * (b - a) / h) / 2) ^ 2)) = 4 * ρ * h / (π * a * b) := 
sorry

end truncated_cone_resistance_l719_719738


namespace sum_series_eq_two_l719_719800

theorem sum_series_eq_two :
  ∑' k : Nat, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end sum_series_eq_two_l719_719800


namespace sum_of_valid_x_values_l719_719003

theorem sum_of_valid_x_values (x : ℝ) (h : 6 = (x^3 - 3*x^2 - 12*x) / (x - 3)) : ∑ (x : ℝ) in {6, -1}, x = 5 :=
by
  sorry

end sum_of_valid_x_values_l719_719003


namespace smallest_b_for_fourth_power_l719_719123

noncomputable def is_fourth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 4 = n

theorem smallest_b_for_fourth_power :
  ∃ b : ℕ, (0 < b) ∧ (7 + 7 * b + 7 * b ^ 2 = (7 * 1 + 7 * 18 + 7 * 18 ^ 2)) 
  ∧ is_fourth_power (7 + 7 * b + 7 * b ^ 2) := sorry

end smallest_b_for_fourth_power_l719_719123


namespace product_fraction_calc_l719_719411

theorem product_fraction_calc :
  (∏ n in (Finset.range 99).filter (λ n, n ≠ 3), (1 - (1 / (n + 2 : ℝ))) * (1 - (1 / (4 : ℝ)))) = 9 / 1600 := 
by
  sorry

end product_fraction_calc_l719_719411


namespace range_of_m_l719_719434

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 2 * Real.sin (x - Real.pi / 3) = (4 * m - 6) / (4 - m)) →
  -1 ≤ m ∧ m ≤ 7 / 3 :=
by
  intro h
  sorry

end range_of_m_l719_719434


namespace minimum_distance_to_line_l719_719166

-- Definitions of the circles
def C1 (x y : ℝ) := x^2 + y^2 = 1
def C2 (x y : ℝ) := (x - 4)^2 + (y - 2)^2 = 1

-- Definition of the condition that |MA| = |MB|
def equal_tangent_lengths (a b : ℝ) : Prop :=
  let power_C1 := a^2 + b^2 - 1
  let power_C2 := (a - 4)^2 + (b - 2)^2 - 1
  power_C1 = power_C2

-- The theorem statement
theorem minimum_distance_to_line (a b : ℝ) (h : equal_tangent_lengths a b) :
  sqrt ((a - 3)^2 + (b + 2)^2) = sqrt(5) / 5 := 
sorry

end minimum_distance_to_line_l719_719166


namespace diff_sequence_polynomial_l719_719204

theorem diff_sequence_polynomial {k n : ℕ} (un : ℕ → ℕ) (h : ∀ n, un n = n^k) :
  ∃ P : ℕ → ℕ, (∀ n, (un (n + 1) - un n) = P n) ∧ polynomial.degree (P n) = k - 1 ∧ polynomial.leading_coeff (P n) = k :=
by sorry

end diff_sequence_polynomial_l719_719204


namespace original_faculty_members_l719_719022

theorem original_faculty_members
  (x : ℝ) (h : 0.87 * x = 195) : x = 224 := sorry

end original_faculty_members_l719_719022


namespace trip_movie_savings_l719_719333

def evening_ticket_cost : ℕ := 10
def combo_cost : ℕ := 10
def ticket_discount_percentage : ℕ := 20
def combo_discount_percentage : ℕ := 50

theorem trip_movie_savings :
  let ticket_saving := evening_ticket_cost * ticket_discount_percentage / 100,
      combo_saving := combo_cost * combo_discount_percentage / 100
  in ticket_saving + combo_saving = 7 :=
by
  sorry

end trip_movie_savings_l719_719333


namespace circumcenter_XYM_on_BC_l719_719637

theorem circumcenter_XYM_on_BC 
  (A B C K X Y M : Point)
  (hA_excircle : A_excircle_touches_at K A B C)
  (hCircAKB : ∃ O1, Circumcircle AKB O1 ∧ bisector_intersects_at X A ∠A O1)
  (hCircAKC : ∃ O2, Circumcircle AKC O2 ∧ bisector_intersects_at Y A ∠A O2)
  (hMidpointM : Midpoint M B C) :
  Circumcenter_Of_Triangle_Lies_On BC XYM :=
sorry

end circumcenter_XYM_on_BC_l719_719637


namespace max_area_225_l719_719656

noncomputable def max_area_rect_perim60 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) : ℝ :=
max (x * y) (30 - x)

theorem max_area_225 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) :
  max_area_rect_perim60 x y h1 h2 = 225 :=
sorry

end max_area_225_l719_719656


namespace number_of_pairs_l719_719014

-- Define the function S(n) for sum of digits
def sum_of_digits (n : Nat) : Nat :=
n.digits.sum -- Assuming there's a concept of digits and List.sum in Mathlib

-- The main theorem statement
theorem number_of_pairs (S : Nat → Nat) : Nat :=
∃ (count : Nat),
  (∀ m n : Nat, 
    m < 100 
    ∧ n < 100 
    ∧ m > n 
    ∧ (m + S(n) = n + 2 * S(m)) 
    → count = 99) :=
sorry

end number_of_pairs_l719_719014


namespace sandra_money_left_l719_719984

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 0.5
def jelly_bean_cost : ℚ := 0.2
def num_candies : ℕ := 14
def num_jelly_beans : ℕ := 20

def total_money : ℕ := sandra_savings + mother_gift + father_gift
def total_candy_cost : ℚ := num_candies * candy_cost
def total_jelly_bean_cost : ℚ := num_jelly_beans * jelly_bean_cost
def total_cost : ℚ := total_candy_cost + total_jelly_bean_cost
def money_left : ℚ := total_money - total_cost

theorem sandra_money_left : money_left = 11 := by
  sorry

end sandra_money_left_l719_719984


namespace distance_from_center_to_plane_correct_l719_719315

noncomputable def distance_from_center_to_plane (O A B C : ℝ × ℝ × ℝ) (radius : ℝ) (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * K)
  let OD := Real.sqrt (radius^2 - R^2)
  OD

theorem distance_from_center_to_plane_correct (O A B C : ℝ × ℝ × ℝ) :
  (dist O A = 20) →
  (dist O B = 20) →
  (dist O C = 20) →
  (dist A B = 13) →
  (dist B C = 14) →
  (dist C A = 15) →
  let m := 15
  let n := 95
  let k := 8
  m + n + k = 118 := by
  sorry

end distance_from_center_to_plane_correct_l719_719315


namespace number_of_subsets_of_S_l719_719319

-- Define the set
def S : Set ℕ := {1, 3, 4}

-- Statement of the theorem
theorem number_of_subsets_of_S: S.powerset.card = 8 := by
  sorry

end number_of_subsets_of_S_l719_719319


namespace minimum_cube_size_l719_719715

def cube : Type := 
  { side_length : ℝ // side_length = 6 }

def large_sphere : Type := 
  { center : ℝ × ℝ × ℝ // center = (3, 3, 3) ∧ radius = 3 }

def small_sphere : Type := 
  { center : ℝ × ℝ × ℝ // 
    ∃ (x y z : ℝ), 
      (center = (x, y, z) ∧ 
      (x = 1 ∨ x = 5) ∧ 
      (y = 1 ∨ y = 5) ∧ 
      (z = 1 ∨ z = 5) ∧ 
      radius = 1 )}

theorem minimum_cube_size : 
  ∃ (c : cube) 
    (ls : large_sphere) 
    (ss : fin 8 → small_sphere), 
  ∀ i, 
    dist ls.center (ss i).center = ls.radius + (ss i).radius :=
sorry

end minimum_cube_size_l719_719715


namespace simplify_tan_product_l719_719989

-- Mathematical Conditions
def tan_inv (x : ℝ) : ℝ := sorry
noncomputable def tan (θ : ℝ) : ℝ := sorry

-- Problem statement to be proven
theorem simplify_tan_product (x y : ℝ) (hx : tan_inv x = 10) (hy : tan_inv y = 35) :
  (1 + x) * (1 + y) = 2 :=
sorry

end simplify_tan_product_l719_719989


namespace number_of_functions_l719_719119

theorem number_of_functions (a b c d : ℝ) :
  (∀ f : ℝ → ℝ, 
    (f = fun x => ax^3 + bx^2 + cx + d) → 
    (∀ x, f(x) * f(-x) = f(x^3)) → 
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 1 ∧ d = 0) ∨
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 1 ∧ b = 0 ∧ c = 1 ∧ d = 0)) :=
begin
  sorry
end

end number_of_functions_l719_719119


namespace trip_savings_l719_719337

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end trip_savings_l719_719337


namespace translated_expression_equals_l719_719532

variable (a b c d e : ℝ)
variable (op1 : ℝ → ℝ → ℝ)
variable (op2 : ℝ → ℝ → ℝ)
variable (op3 : ℝ → ℝ → ℝ)
variable (op4 : ℝ → ℝ → ℝ)

def translate_ops (x y z w v : ℝ) : ℝ :=
  let tmp1 := op1 x y
  let tmp2 := op2 tmp1 z
  let tmp3 := op1 tmp2 w
  op3 tmp3 v

theorem translated_expression_equals :
  a = 240 →
  b = 80 →
  c = 60 →
  d = 40 →
  e = 10 →
  op1 = (/) →
  op2 = (*) →
  op3 = (+) →
  translate_ops a b c d e = 14.5 :=
by
  intros _ _ _ _ _ _ _ _
  unfold translate_ops
  sorry

end translated_expression_equals_l719_719532


namespace find_n_l719_719349

theorem find_n : ∃ n : ℕ, 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) ∧ (n = 57) := by
  sorry

end find_n_l719_719349


namespace segment_parallel_to_bases_divides_area_l719_719618

theorem segment_parallel_to_bases_divides_area
  (a c d : ℝ)
  (p q : ℝ)
  (h_parallel : d = sqrt((q * a ^ 2 + p * c ^ 2) / (p + q))) :
  d = sqrt((q * a ^ 2 + p * c ^ 2) / (p + q)) :=
by 
  sorry

end segment_parallel_to_bases_divides_area_l719_719618


namespace find_y_value_l719_719464

theorem find_y_value : 
  ∃ (y : ℝ), (128^(y + 1) / 8^(y + 1) = 64^(3 * y - 2)) ∧ y = 8 / 7 :=
sorry

end find_y_value_l719_719464


namespace lois_books_count_l719_719959

variable (initial_books : ℕ)
variable (given_books_ratio : ℚ)
variable (donated_books_ratio : ℚ)
variable (purchased_books : ℕ)

theorem lois_books_count (h_initial : initial_books = 40)
  (h_given_ratio : given_books_ratio = 1 / 4)
  (h_donated_ratio : donated_books_ratio = 1 / 3)
  (h_purchased : purchased_books = 3) :
  let remaining_after_given := initial_books - initial_books * given_books_ratio in
  let remaining_after_donated := remaining_after_given - remaining_after_given * donated_books_ratio in
    remaining_after_donated + purchased_books = 23 := by
  sorry

end lois_books_count_l719_719959


namespace percentage_increase_in_stock_l719_719659

variable (P0 P1 P2 : ℝ)
variable (x : ℝ)

-- Definitions based on conditions
def initial_price (P0 : ℝ) : ℝ := 100
def decreased_price (P0 : ℝ) : ℝ := P0 * 0.92
def final_price (P1 : ℝ) (x : ℝ) : ℝ := P1 * (1 + x / 100)
def net_final_price (P0 : ℝ) : ℝ := P0 * 1.012

-- Theorem statement
theorem percentage_increase_in_stock 
  (h1 : P1 = decreased_price P0) 
  (h2 : P2 = final_price P1 x) 
  (h3 : P2 = net_final_price P0) : x = 10 :=
by
  sorry

end percentage_increase_in_stock_l719_719659


namespace distance_between_closest_points_of_circles_l719_719414

noncomputable def distance_center_to_line {x y: ℝ} (cx : ℝ) (cy : ℝ) : ℝ :=
  abs(cy - y)

noncomputable def distance_between_points (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_closest_points_of_circles :
  let c1 := (5 : ℝ, 5 : ℝ)
  let c2 := (20 : ℝ, 15 : ℝ)
  let y_line := (3 : ℝ)
  let r1 := distance_center_to_line c1.1 c1.2 y_line
  let r2 := distance_center_to_line c2.1 c2.2 y_line
  let d := distance_between_points c1.1 c1.2 c2.1 c2.2
  r1 = 2 →
  r2 = 12 →
  d = Real.sqrt 325 →
  (d - (r1 + r2) = Real.sqrt 325 - 14) :=
by
  intros c1 c2 y_line r1 r2 d hr1 hr2 hd
  rw hr1 at r1
  rw hr2 at r2
  rw hd at d
  sorry

end distance_between_closest_points_of_circles_l719_719414


namespace infinite_series_sum_l719_719443

theorem infinite_series_sum :
  (∑ k in Nat, (k^2 : ℝ)/(3^k : ℝ)) = 6 := sorry

end infinite_series_sum_l719_719443


namespace min_value_3x_4y_l719_719951

theorem min_value_3x_4y
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  3 * x + 4 * y = 21 :=
sorry

end min_value_3x_4y_l719_719951


namespace moles_of_NaOH_combined_l719_719120

noncomputable def moles_of_H2O_produced : ℕ := 1
noncomputable def moles_of_HCH3CO2_used : ℕ := 1
noncomputable def reaction : ℕ → ℕ := λ n, if n = 1 then 1 else 0

theorem moles_of_NaOH_combined:
  reaction(moles_of_HCH3CO2_used) = moles_of_H2O_produced :=
by sorry

end moles_of_NaOH_combined_l719_719120


namespace shaded_area_l719_719239

theorem shaded_area (r : ℝ) (sector_area : ℝ) (h1 : r = 4) (h2 : sector_area = 2 * Real.pi) : 
  sector_area - (1 / 2 * (r * Real.sqrt 2) * (r * Real.sqrt 2)) = 2 * Real.pi - 4 :=
by 
  -- Lean proof follows
  sorry

end shaded_area_l719_719239


namespace abs_sum_zero_implies_diff_eq_five_l719_719540

theorem abs_sum_zero_implies_diff_eq_five (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 :=
  sorry

end abs_sum_zero_implies_diff_eq_five_l719_719540


namespace number_of_good_subsets_is_odd_l719_719256

-- Definitions and preliminary conditions
variable (T : Finset ℕ) (hT : ∀ t ∈ T, 1 < t)

-- Definition of a good subset
def is_good_subset (S : Finset ℕ) : Prop :=
  ∀ t ∈ T, ∃ s ∈ S, Nat.gcd s t > 1

-- Main statement
theorem number_of_good_subsets_is_odd : (Finset.filter is_good_subset (Finset.powerset T)).card % 2 = 1 := 
by sorry

end number_of_good_subsets_is_odd_l719_719256


namespace correct_propositions_count_l719_719769

theorem correct_propositions_count:
  let P1 := ∀ (A : Set), ¬ (A = ∅) → ∀ (B : Set), ¬ (B ⊆ A)
  let P2 := ∀ (A : Set), (∃ (B : Set), B ⊆ A ∧ B ≠ A) → ∃ (C : Set), C ⊆ A ∧ C ≠ A ∧ C ≠ B
  let P3 := ∀ (A : Set), (A = ∅) ∨ (∃ (B : Set), A ≠ B ∧ A ⊂ B)
  let P4 := ∀ (A : Set), (∅ ⊊ (A : Set)) → A ≠ ∅
  P4 ∧ ¬P1 ∧ ¬P2 ∧ ¬P3
:= by
  sorry

end correct_propositions_count_l719_719769


namespace range_of_a_l719_719854

theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1)
  (h_decreasing : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y → (log a (2 - a * y)) < (log a (2 - a * x))) :
  1 < a ∧ a < 2 :=
sorry

end range_of_a_l719_719854


namespace volume_difference_l719_719775

-- Definitions for the conditions
def circumference_anita : ℝ := 6  -- (7 inches width minus 1 inch overlap)
def height_anita : ℝ := 10

def circumference_bernadette : ℝ := 7
def height_bernadette : ℝ := 7

def volume_anita : ℝ :=
  let r_A := circumference_anita / (2 * Real.pi)
  Real.pi * r_A^2 * height_anita

def volume_bernadette : ℝ :=
  let r_B := circumference_bernadette / (2 * Real.pi)
  Real.pi * r_B^2 * height_bernadette

-- Lean 4 statement
theorem volume_difference : Real.pi * |volume_bernadette - volume_anita| = 17 / 4 := 
by
  sorry

end volume_difference_l719_719775


namespace spinner_probability_l719_719000

theorem spinner_probability :
  let total_sections := 8
  let numbers := (1 : ℕ) :: (2 : ℕ) :: (3 : ℕ) :: (4 : ℕ) :: (5 : ℕ) :: (6 : ℕ) :: (7 : ℕ) :: (8 : ℕ) :: []
  let favorable_sections := numbers.filter (λ n, n < 4)
  let probability := (favorable_sections.length : ℚ) / (total_sections : ℚ)
  probability = 3 / 8 :=
by
  sorry

end spinner_probability_l719_719000


namespace root_reciprocal_sum_l719_719593

theorem root_reciprocal_sum :
  (∑ r in ({r : ℝ | r^2 - 2*r - 5 = 0}.to_finset), (1/r)) = -2/5 := by
sorry

end root_reciprocal_sum_l719_719593


namespace win_sector_area_l719_719727

-- Defining the conditions
def radius : ℝ := 12
def total_area : ℝ := π * radius^2
def win_probability : ℝ := 1 / 3

-- Theorem to prove the area of the WIN sector
theorem win_sector_area : total_area * win_probability = 48 * π := by
  sorry

end win_sector_area_l719_719727


namespace area_of_rectangle_with_tangent_circles_l719_719137

theorem area_of_rectangle_with_tangent_circles :
  let d := 6 in
  let r := d / 2 in
  let height := 2 * r in
  let width := 2 * d in
  let area := height * width in
  area = 72 :=
by
  let d := 6
  let r := d / 2
  let height := 2 * r
  let width := 2 * d
  let area := height * width
  show area = 72
  sorry

end area_of_rectangle_with_tangent_circles_l719_719137


namespace alpha_cubed_plus_5beta_plus_10_l719_719867

noncomputable def α: ℝ := sorry
noncomputable def β: ℝ := sorry

-- Given conditions
axiom roots_eq : ∀ x : ℝ, x^2 + 2 * x - 1 = 0 → (x = α ∨ x = β)
axiom sum_eq : α + β = -2
axiom prod_eq : α * β = -1

-- The theorem stating the desired result
theorem alpha_cubed_plus_5beta_plus_10 :
  α^3 + 5 * β + 10 = -2 :=
sorry

end alpha_cubed_plus_5beta_plus_10_l719_719867


namespace g_of_minus_1_eq_9_l719_719589

-- defining f(x) and g(f(x)), and stating the objective to prove g(-1)=9
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 5

theorem g_of_minus_1_eq_9 : g (-1) = 9 :=
  sorry

end g_of_minus_1_eq_9_l719_719589


namespace num_valid_values_n_l719_719265

theorem num_valid_values_n: 
  let valid_n := λ (n : ℕ), 
    n >= 10000 ∧ n <= 99999 ∧ 
    let q := n / 100 in 
    let r := n % 100 in 
    (3 * q + r) % 13 = 0 
  in 
  (Finset.filter valid_n (Finset.range 100000)).card = 6300 := 
sorry

end num_valid_values_n_l719_719265


namespace projection_of_a_onto_b_l719_719905

def a : ℝ × ℝ × ℝ := (1, -1, 1)
def b : ℝ × ℝ × ℝ := (-2, 2, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let k := (dot_product u v) / (magnitude v)^2 in
  (k * v.1, k * v.2, k * v.3)

theorem projection_of_a_onto_b :
  projection a b = (2/3, -2/3, -1/3) :=
by
  sorry

end projection_of_a_onto_b_l719_719905


namespace maintenance_checks_l719_719067

theorem maintenance_checks (x : ℕ) (h1 : 1.5 * x = 45) : x = 30 :=
sorry

end maintenance_checks_l719_719067


namespace outfit_choices_l719_719856

theorem outfit_choices (tops pants : ℕ) (TopsCount : tops = 4) (PantsCount : pants = 3) :
  tops * pants = 12 := by
  sorry

end outfit_choices_l719_719856


namespace number_of_candidates_l719_719390

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 132) : n = 12 :=
by
  sorry

end number_of_candidates_l719_719390


namespace smooth_transition_l719_719383

theorem smooth_transition (R : ℝ) (x₀ y₀ : ℝ) :
  ∃ m : ℝ, ∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = R^2 → y - y₀ = m * (x - x₀) :=
sorry

end smooth_transition_l719_719383


namespace student_can_escape_l719_719056

/-- Problem Statement:
Given a square pond, a student starts at the center and a teacher at one of the vertices. The teacher can run 4 times faster than the student can swim, but the student runs faster than the teacher once they reach the shore. Prove that the student can escape.
-/
theorem student_can_escape
  (side_length : ℝ)
  (student_swim_speed : ℝ)
  (teacher_run_speed : ℝ)
  (teacher_initial_position : ℝ)
  (student_run_speed : ℝ)
  (h1 : teacher_run_speed = 4 * student_swim_speed)
  (h2 : student_run_speed > teacher_run_speed)
  (h3 : teacher_initial_position ∈ {0, 1}) : -- Consider the vertices as (0,1) positions on a unit circle
  ∃ (escape_strategy : ℝ → ℝ → ℝ),
    escape_strategy = λ student_position teacher_position,
    (∀ time : ℝ,
      time * student_swim_speed >= (side_length / 2) ∧
      time * student_swim_speed <= ((2 * side_length) / (4 * student_swim_speed))) :=
sorry

end student_can_escape_l719_719056


namespace real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l719_719151

-- Define z as a complex number with components dependent on m
def z (m : ℝ) : ℂ := ⟨m^2 - m, m - 1⟩

-- Statement 1: z is real iff m = 1
theorem real_z_iff (m : ℝ) : (∃ r : ℝ, z m = ⟨r, 0⟩) ↔ m = 1 := 
    sorry

-- Statement 2: z is purely imaginary iff m = 0
theorem imaginary_z_iff (m : ℝ) : (∃ i : ℝ, z m = ⟨0, i⟩ ∧ i ≠ 0) ↔ m = 0 := 
    sorry

-- Statement 3: z is in the first quadrant iff m > 1
theorem first_quadrant_z_iff (m : ℝ) : (z m).re > 0 ∧ (z m).im > 0 ↔ m > 1 := 
    sorry

end real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l719_719151


namespace cylinder_cannot_have_triangular_front_view_cone_can_have_triangular_front_view_tetrahedron_can_have_triangular_front_view_triangularPrism_can_have_triangular_front_view_l719_719006

-- Definitions of solid geometries
inductive SolidGeometry
| Cylinder
| Cone
| Tetrahedron
| TriangularPrism

-- Function to determine if a given solid geometry can have a triangular front view
def canHaveTriangularFrontView : SolidGeometry → Prop
| SolidGeometry.Cylinder       := False
| SolidGeometry.Cone           := True
| SolidGeometry.Tetrahedron    := True
| SolidGeometry.TriangularPrism := True

-- Proof statement (without proof, so we use 'sorry')
theorem cylinder_cannot_have_triangular_front_view :
  ¬ canHaveTriangularFrontView SolidGeometry.Cylinder :=
by sorry

-- Additional theorems for completeness
theorem cone_can_have_triangular_front_view :
  canHaveTriangularFrontView SolidGeometry.Cone :=
by sorry

theorem tetrahedron_can_have_triangular_front_view :
  canHaveTriangularFrontView SolidGeometry.Tetrahedron :=
by sorry

theorem triangularPrism_can_have_triangular_front_view :
  canHaveTriangularFrontView SolidGeometry.TriangularPrism :=
by sorry

end cylinder_cannot_have_triangular_front_view_cone_can_have_triangular_front_view_tetrahedron_can_have_triangular_front_view_triangularPrism_can_have_triangular_front_view_l719_719006


namespace win_sector_area_l719_719730

-- Given Conditions
def radius := 12  -- radius of the circle in cm
def probability_of_winning := 1 / 3  -- probability of winning on one spin

-- Calculate the total area of the circle
def total_area_circle : ℝ := real.pi * radius^2

-- Calculate the area of the WIN sector
def area_of_win_sector : ℝ := probability_of_winning * total_area_circle

-- Proof Statement
theorem win_sector_area : area_of_win_sector = 48 * real.pi :=
by
  -- proof steps will go here
  sorry

end win_sector_area_l719_719730


namespace parity_of_f_f_is_increasing_range_of_m_l719_719152

noncomputable def f : ℝ → ℝ := sorry

variable {x y a m : ℝ}

-- Conditions
axiom f_defined_on_interval : ∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ y, f y = f x 
axiom f_additive : ∀ x y ∈ Icc (-1:ℝ) 1, f (x + y) = f x + f y
axiom f_positive : ∀ x, x > 0 ∧ x ≤ 1 → f x > 0
axiom f_at_one : f 1 = 1

-- Questions
theorem parity_of_f : ∀ x ∈ Icc (-1:ℝ) 1, f (-x) = -f x := sorry

theorem f_is_increasing : ∀ x1 x2 ∈ Icc (-1:ℝ) 1, x1 < x2 → f x1 < f x2 := sorry

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Icc (-1:ℝ) 1, ∀ a ∈ Icc (-1:ℝ) 1, f x < m^2 - 2 * a * m + 1) → m ∈ set.Iio (-2) ∪ set.Ioi 2 := sorry

end parity_of_f_f_is_increasing_range_of_m_l719_719152


namespace infinite_series_equals_two_l719_719803

noncomputable def sum_series : ℕ → ℝ := λ k, (8^k : ℝ) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_equals_two :
  (∑' k : ℕ, if k > 0 then sum_series k else 0) = 2 :=
by 
  sorry

end infinite_series_equals_two_l719_719803


namespace trigonometric_identity_l719_719324

theorem trigonometric_identity (α : ℝ) : 
  sin^2 (2 * π - α) + cos (π + α) * cos (π - α) + 1 = 2 :=
by
  sorry

end trigonometric_identity_l719_719324


namespace range_of_a_l719_719197

theorem range_of_a (a : ℝ) : a > real.exp (-1) → ∀ x : ℝ, 0 < x → exp(x) + (log a) / a > (log x) / a :=
sorry

end range_of_a_l719_719197


namespace student_registration_results_l719_719340

/-- Two students are registering for tests at three universities:
Zhejiang University, Fudan University, and Shanghai Jiao Tong University.
Each student can choose at most two schools.

We need to prove that the number of different registration results is 36. -/
theorem student_registration_results :
  let U := {1, 2, 3} -- representing the three universities
  ∃ (f : fin 2 → set U), (∀ i, f i ⊆ U ∧ f i.card ≤ 2) ∧
    (finset.univ.powerset.filter (λ S : finset (fin 2 → set U), S.card = 2).card = 36) :=
sorry

end student_registration_results_l719_719340


namespace expression_equality_l719_719857

-- Define the conditions
variables {a b x : ℝ}
variable (h1 : x = a / b)
variable (h2 : a ≠ 2 * b)
variable (h3 : b ≠ 0)

-- Define and state the theorem
theorem expression_equality : (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) :=
by 
  intros
  sorry

end expression_equality_l719_719857


namespace david_total_course_hours_l719_719422

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end david_total_course_hours_l719_719422


namespace exists_ab_odd_n_exists_ab_odd_n_gt3_l719_719846

-- Define the required conditions
def gcd_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define a helper function to identify odd positive integers
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem exists_ab_odd_n (n : ℕ) (h : is_odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n :=
sorry

theorem exists_ab_odd_n_gt3 (n : ℕ) (h1 : is_odd n) (h2 : n > 3) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n ∧ n ∣ (a - b) = false :=
sorry

end exists_ab_odd_n_exists_ab_odd_n_gt3_l719_719846


namespace greatest_root_of_g_l719_719836

def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 3

theorem greatest_root_of_g :
  ∃ x : ℝ, g(x) = 0 ∧ ∀ y : ℝ, g(y) = 0 → x ≥ y ∧ x = sqrt(15) / 5 :=
by
  sorry

end greatest_root_of_g_l719_719836


namespace alphazia_lost_words_l719_719243

def alphazia_letters := 128
def forbidden_letters := 2
def total_forbidden_pairs := forbidden_letters * alphazia_letters

theorem alphazia_lost_words :
  let one_letter_lost := forbidden_letters
  let two_letter_lost := 2 * alphazia_letters
  one_letter_lost + two_letter_lost = 258 :=
by
  sorry

end alphazia_lost_words_l719_719243


namespace find_largest_n_l719_719096

theorem find_largest_n : ∃ n x y z : ℕ, n > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 
  ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6
  ∧ (∀ m x' y' z' : ℕ, m > n → x' > 0 → y' > 0 → z' > 0 
  → m^2 ≠ x'^2 + y'^2 + z'^2 + 2*x'*y' + 2*y'*z' + 2*z'*x' + 3*x' + 3*y' + 3*z' - 6) :=
sorry

end find_largest_n_l719_719096


namespace david_total_course_hours_l719_719420

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end david_total_course_hours_l719_719420


namespace sandra_remaining_money_l719_719987

def sandra_savings : ℝ := 10
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def num_candies : ℝ := 14
def num_jelly_beans : ℝ := 20

theorem sandra_remaining_money : (sandra_savings + mother_contribution + father_contribution) - (num_candies * candy_cost + num_jelly_beans * jelly_bean_cost) = 11 :=
by
  sorry

end sandra_remaining_money_l719_719987


namespace side_significant_digits_l719_719825

theorem side_significant_digits
  (area : ℝ)
  (h_area : area = 0.3600) :
  (significant_digits (sqrt area) = 4) :=
sorry

end side_significant_digits_l719_719825


namespace solve_for_x_l719_719148

theorem solve_for_x (x : ℝ) (h : |3990 * x + 1995| = 1995) : x = 0 ∨ x = -1 :=
by
  sorry

end solve_for_x_l719_719148


namespace exists_same_color_parallelepiped_l719_719327

-- Definitions of the conditions
def A : Set (ℤ × ℤ × ℤ) := {v | True}

variable (color : ℤ × ℤ × ℤ → ℕ)
variable (p : ℕ)
variable (color_range : ∀ v ∈ A, color v < p)

-- The main theorem statement
theorem exists_same_color_parallelepiped : 
  ∃ (v1 v2 v3 v4 v5 v6 v7 v8 : ℤ × ℤ × ℤ),
  v1 ∈ A ∧ v2 ∈ A ∧ v3 ∈ A ∧ v4 ∈ A ∧
  v5 ∈ A ∧ v6 ∈ A ∧ v7 ∈ A ∧ v8 ∈ A ∧
  -- All vertices have the same color
  color v1 = color v2 ∧ color v2 = color v3 ∧ color v3 = color v4 ∧
  color v4 = color v5 ∧ color v5 = color v6 ∧ color v6 = color v7 ∧
  color v7 = color v8 ∧
  -- Vertices form a rectangular parallelepiped
  (∃ (x1 x2 y1 y2 z1 z2 : ℤ), 
    v1 = (x1, y1, z1) ∧ v2 = (x2 ,y1 ,z1) ∧ v3 = (x1, y2, z1) ∧
    v4 = (x2, y2, z1) ∧ v5 = (x1, y1, z2) ∧ v6 = (x2, y1, z2) ∧
    v7 = (x1, y2, z2) ∧ v8 = (x2, y2, z2)) := 
sorry

end exists_same_color_parallelepiped_l719_719327


namespace saturday_earnings_l719_719717

-- Define the variables and constants
noncomputable def W : ℝ := sorry
noncomputable def S : ℝ := 4084.09
def total_earnings := 7250
def wednesday_earnings_percentage_ticket_sales := 0.60
def wednesday_diff_ticket_sales := 142.50
def saturday_earnings_percentage_ticket_sales := 0.50

-- Define the conditions as Lean definitions
def condition1 := W + S = total_earnings
def condition2 := wednesday_earnings_percentage_ticket_sales * W 
                = saturday_earnings_percentage_ticket_sales * S - wednesday_diff_ticket_sales

-- The theorem to be proven
theorem saturday_earnings : 
  W + S = total_earnings ∧ 
  wednesday_earnings_percentage_ticket_sales * W = saturday_earnings_percentage_ticket_sales * S - wednesday_diff_ticket_sales
  → S = 4084.09 :=
by
  intro h
  cases h with h1 h2
  sorry

end saturday_earnings_l719_719717


namespace real_condition_complex_condition_pure_imaginary_condition_l719_719136

-- Definitions for our conditions
def is_real (z : ℂ) : Prop := z.im = 0
def is_complex (z : ℂ) : Prop := z.im ≠ 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- The given complex number definition
def z (m : ℝ) : ℂ := { re := m^2 + m, im := m^2 - 1 }

-- Prove that for z to be a real number, m must be ±1
theorem real_condition (m : ℝ) : is_real (z m) ↔ m = 1 ∨ m = -1 := 
sorry

-- Prove that for z to be a complex number, m must not be ±1 
theorem complex_condition (m : ℝ) : is_complex (z m) ↔ m ≠ 1 ∧ m ≠ -1 := 
sorry 

-- Prove that for z to be a pure imaginary number, m must be 0
theorem pure_imaginary_condition (m : ℝ) : is_pure_imaginary (z m) ↔ m = 0 := 
sorry 

end real_condition_complex_condition_pure_imaginary_condition_l719_719136


namespace marks_in_biology_l719_719094

theorem marks_in_biology (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) (marks_chemistry : ℕ) (average_marks : ℕ) :
  marks_english = 73 → marks_math = 69 → marks_physics = 92 → marks_chemistry = 64 → average_marks = 76 →
  (380 - (marks_english + marks_math + marks_physics + marks_chemistry)) = 82 :=
by
  intros
  sorry

end marks_in_biology_l719_719094


namespace transformed_cube_edges_l719_719754

-- Definition of the problem conditions and final problem statement

/-- Let the original cube have 12 edges.
Remove smaller cubes from each corner and cut the resulting solid.
The total number of edges after these transformations is 40. -/
theorem transformed_cube_edges (side_length_large_cube side_length_small_cube : ℕ)
    (initial_edges new_edges : ℕ) : 
  side_length_large_cube = 5 → 
  side_length_small_cube = 2 →
  initial_edges = 12 →
  new_edges = 24 →
  (initial_edges + new_edges + 4 = 40) :=
begin
  intros H1 H2 H3 H4,
  rw [H3, H4],
  norm_num,
end

end transformed_cube_edges_l719_719754


namespace range_of_a_l719_719581

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : (A ∩ B a).Nonempty) : a > 1 :=
sorry

end range_of_a_l719_719581


namespace number_of_functions_l719_719116

theorem number_of_functions : 
  ∃ (f : ℝ → ℝ) (a b c d : ℝ), 
  (∀ x, f x = a * x^3 + b * x^2 + c * x + d) ∧ 
  (∀ x, (f x) * (f (-x)) = f (x^3)) 
  ∧ (nat.card {f : ℝ → ℝ | ∃ a b c d : ℝ, 
                (∀ x, f x = a * x^3 + b * x^2 + c * x + d) 
                ∧ (∀ x, (f x) * (f (-x)) = f (x^3))}) = 6 := 
by sorry

end number_of_functions_l719_719116


namespace find_value_of_powers_l719_719052

theorem find_value_of_powers (a b : ℝ) (h₁ : a ≠ 0) 
  (h₂ : {a, b / a, 1} = {a^2, a + b, 0}) : 
  a^2003 + b^2004 = -1 := 
sorry

end find_value_of_powers_l719_719052


namespace lois_final_books_l719_719960

-- Definitions for the conditions given in the problem.
def initial_books : ℕ := 40
def books_given_to_nephew (b : ℕ) : ℕ := b / 4
def books_remaining_after_giving (b_given : ℕ) (b : ℕ) : ℕ := b - b_given
def books_donated_to_library (b_remaining : ℕ) : ℕ := b_remaining / 3
def books_remaining_after_donating (b_donated : ℕ) (b_remaining : ℕ) : ℕ := b_remaining - b_donated
def books_purchased : ℕ := 3
def total_books (b_final_remaining : ℕ) (b_purchased : ℕ) : ℕ := b_final_remaining + b_purchased

-- Theorem stating: Given the initial conditions, Lois should have 23 books in the end.
theorem lois_final_books : 
  total_books 
    (books_remaining_after_donating (books_donated_to_library (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books)) 
    (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books))
    books_purchased = 23 :=
  by
    sorry  -- Proof omitted as per instructions.

end lois_final_books_l719_719960


namespace length_shorter_diagonal_l719_719207

open Real

-- Given conditions
variables (a b : Vector3)
variables (theta : ℝ)
variables (h_a : norm a = 1)
variables (h_b : norm b = 2)
variables (h_angle : arccos (euclidean_inner_product a b / (norm a * norm b)) = theta)

-- Lean 4 statement to prove the length of the shorter diagonal
theorem length_shorter_diagonal (a b : Vector3) (theta : ℝ)
  (h_a : norm a = 1)
  (h_b : norm b = 2)
  (h_angle : arccos (euclidean_inner_product a b / (norm a * norm b)) = theta) :
  min (norm (a + b)) (norm (a - b)) = sqrt (5 + 4 * cos theta) :=
by sorry

end length_shorter_diagonal_l719_719207


namespace cubes_with_at_least_two_faces_painted_l719_719374

theorem cubes_with_at_least_two_faces_painted (n : ℕ) : 
  n = 4 → (∃ painted_cubes : ℕ, painted_cubes = 32) := 
by
  intro h
  use 32
  sorry

end cubes_with_at_least_two_faces_painted_l719_719374


namespace chord_lengths_sum_l719_719605

noncomputable def chord_sum (radius : ℝ) (distance_to_center : ℝ) : ℝ :=
  let longest := 2 * radius
  let shortest := 2 * real.sqrt (radius ^ 2 - distance_to_center ^ 2)
  let integers_between := [6, 7, 8, 9]
  longest + shortest + 2 * (integers_between.map (λ x, x)).sum

theorem chord_lengths_sum :
  ∀ (radius distance_to_center : ℝ), radius = 5 ∧ distance_to_center = 4 →
  chord_sum radius distance_to_center = 40 :=
by
  intros radius distance_to_center h
  cases h with hr hd
  rw [hr, hd]
  sorry

end chord_lengths_sum_l719_719605


namespace equal_lengths_l719_719430

noncomputable def F (x y z : ℝ) := (x+y+z) * (x+y-z) * (y+z-x) * (x+z-y)

variables {a b c d e f : ℝ}

axiom acute_angled_triangle (x y z : ℝ) : Prop

axiom altitudes_sum_greater (x y z : ℝ) : Prop

axiom cond1 : acute_angled_triangle a b c
axiom cond2 : acute_angled_triangle b d f
axiom cond3 : acute_angled_triangle a e f
axiom cond4 : acute_angled_triangle e c d

axiom cond5 : altitudes_sum_greater a b c
axiom cond6 : altitudes_sum_greater b d f
axiom cond7 : altitudes_sum_greater a e f
axiom cond8 : altitudes_sum_greater e c d

axiom cond9 : F a b c = F b d f
axiom cond10 : F a e f = F e c d

theorem equal_lengths : a = d ∧ b = e ∧ c = f := by
  sorry -- Proof not required.

end equal_lengths_l719_719430


namespace part_a_l719_719708

theorem part_a {d m b : ℕ} (h_d : d = 41) (h_m : m = 28) (h_b : b = 15) :
    d - b + m - b + b = 54 :=
  by sorry

end part_a_l719_719708


namespace no_solution_iff_a_leq_8_l719_719548

theorem no_solution_iff_a_leq_8 (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_iff_a_leq_8_l719_719548


namespace distance_between_parallel_lines_l719_719326

theorem distance_between_parallel_lines
  (O A B C D P Q : ℝ) -- Points on the circle with P and Q as defined midpoints
  (r d : ℝ) -- Radius of the circle and distance between the parallel lines
  (h_AB : dist A B = 36) -- Length of chord AB
  (h_CD : dist C D = 36) -- Length of chord CD
  (h_BC : dist B C = 40) -- Length of chord BC
  (h_OA : dist O A = r) 
  (h_OB : dist O B = r)
  (h_OC : dist O C = r)
  (h_PQ_parallel : dist P Q = d) -- Midpoints
  : d = 4 * Real.sqrt 19 / 3 :=
sorry

end distance_between_parallel_lines_l719_719326


namespace strongest_goldbach_132_l719_719321
open Nat

theorem strongest_goldbach_132 :
  ∀ p q : ℕ, prime p → prime q → p + q = 132 → p < q → (∃ d : ℕ, d = q - p ∧ ∀ r s : ℕ, prime r → prime s → r + s = 132 → r < s → (s - r) ≤ d) :=
begin
  sorry
end

end strongest_goldbach_132_l719_719321


namespace find_n_l719_719028

theorem find_n 
  (n : ℕ) 
  (h_lcm : Nat.lcm n 16 = 48) 
  (h_gcf : Nat.gcd n 16 = 18) : 
  n = 54 := 
sorry

end find_n_l719_719028


namespace cubs_more_home_runs_l719_719076

-- Define the conditions for the Chicago Cubs
def cubs_home_runs_third_inning : Nat := 2
def cubs_home_runs_fifth_inning : Nat := 1
def cubs_home_runs_eighth_inning : Nat := 2

-- Define the conditions for the Cardinals
def cardinals_home_runs_second_inning : Nat := 1
def cardinals_home_runs_fifth_inning : Nat := 1

-- Total home runs scored by each team
def total_cubs_home_runs : Nat :=
  cubs_home_runs_third_inning + cubs_home_runs_fifth_inning + cubs_home_runs_eighth_inning

def total_cardinals_home_runs : Nat :=
  cardinals_home_runs_second_inning + cardinals_home_runs_fifth_inning

-- The statement to prove
theorem cubs_more_home_runs : total_cubs_home_runs - total_cardinals_home_runs = 3 := by
  sorry

end cubs_more_home_runs_l719_719076


namespace blue_balls_count_l719_719716

theorem blue_balls_count (B : ℕ) (h : (4 / (6 + B)) * (3 / (5 + B)) = 0.13333333333333333) : B = 4 :=
sorry

end blue_balls_count_l719_719716


namespace simplify_tan_expression_l719_719991

theorem simplify_tan_expression :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have tan_add : Real.tan (10 * Real.pi / 180 + 35 * Real.pi / 180) = (Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180)) / (1 - Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180)) := by sorry
  have eq : Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180) = 1 - Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180) := by
    rw [← tan_add, tan_45]
    field_simp
    ring
  have res : (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 1 + (Real.tan (10 * Real.pi / 180) + Real.tan (35 * Real.pi / 180)) + Real.tan (10 * Real.pi / 180) * Real.tan (35 * Real.pi / 180) := by ring
  rw [eq, res]
  ring
  sorry

end simplify_tan_expression_l719_719991


namespace bucket_full_weight_l719_719690

theorem bucket_full_weight (x y c d : ℝ) 
  (h1 : x + (3/4) * y = c)
  (h2 : x + (3/5) * y = d) :
  x + y = (5/3) * c - (5/3) * d :=
by
  sorry

end bucket_full_weight_l719_719690


namespace num_people_didn_t_show_up_l719_719737

def people_invited := 45
def tables_needed := 5
def people_per_table := 2

def people_showed_up : ℕ := tables_needed * people_per_table
def people_didn_t_show_up : ℕ := people_invited - people_showed_up

theorem num_people_didn_t_show_up : people_didn_t_show_up = 35 :=
by
  unfold people_didn_t_show_up
  unfold people_showed_up
  simp [people_invited, tables_needed, people_per_table]
  sorry

end num_people_didn_t_show_up_l719_719737


namespace cross_country_race_winning_scores_l719_719553

theorem cross_country_race_winning_scores :
  (∃ (scores : Set ℕ), ∀ s ∈ scores, s ∈ {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}) 
  ∧ scores.card = 13 :=
by
  sorry

end cross_country_race_winning_scores_l719_719553


namespace value_of_ab_l719_719544

theorem value_of_ab (a b c : ℝ) (C : ℝ) (h1 : (a + b) ^ 2 - c ^ 2 = 4) (h2 : C = Real.pi / 3) : 
  a * b = 4 / 3 :=
by
  sorry

end value_of_ab_l719_719544


namespace susie_investment_l719_719996

theorem susie_investment (x : ℝ) :
  let x_slb := x,
      x_gmb := 1000 - x,
      amount_slb := x_slb * (1 + 0.04)^3,
      amount_gmb := x_gmb * (1 + 0.06)^3,
      total_amount := amount_slb + amount_gmb in
  total_amount = 1206.11 → x_slb = 228.14 :=
by
  intros x_slb x_gmb amount_slb amount_gmb total_amount h
  rw [show x_slb = x, from rfl]
  rw [show x_gmb = 1000 - x, from rfl]
  rw [show amount_slb = x * (1 + 0.04)^3, from rfl]
  rw [show amount_gmb = (1000 - x) * (1 + 0.06)^3, from rfl]
  rw [show total_amount = x * (1 + 0.04)^3 + (1000 - x) * (1 + 0.06)^3, from rfl]
  sorry

end susie_investment_l719_719996


namespace sum_of_cubes_ratio_l719_719275

theorem sum_of_cubes_ratio (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) (h2 : d + e + f = 0) :
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = (a * b * c) / (d * e * f) := 
by 
  sorry

end sum_of_cubes_ratio_l719_719275


namespace digit_in_position_206788_l719_719766
open Nat

/-- Statement: The digit that appears in the 206788th position in the sequence of all natural numbers written consecutively is 7. -/
theorem digit_in_position_206788 : 
  let s := List.join $ List.map (λ n : Nat, n.digits 10) (List.range' 1 206789)
  s.get? 206787 = some 7 := sorry

end digit_in_position_206788_l719_719766


namespace f_log2_7_eq_7_div_2_l719_719885

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2^x else f (x-1)

theorem f_log2_7_eq_7_div_2 : f (Real.log 7 / Real.log 2) = 7 / 2 :=
by
  sorry

end f_log2_7_eq_7_div_2_l719_719885


namespace ab_cardinals_l719_719597

open Set

/-- a|A| = b|B| given the conditions.
1. a and b are positive integers.
2. A and B are finite sets of integers such that:
   a. A and B are disjoint.
   b. If an integer i belongs to A or to B, then i + a ∈ A or i - b ∈ B.
-/
theorem ab_cardinals 
  (a b : ℕ) (A B : Finset ℤ) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (disjoint_AB : Disjoint A B)
  (condition_2 : ∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := 
sorry

end ab_cardinals_l719_719597


namespace slope_of_line_l719_719465

theorem slope_of_line (x y : ℝ) : 
  (∃ b : ℝ, y = - (3/2) * x + b) → 
  (∃ b : ℝ, (x / 2 + y / 3) = 2 → (y = - (3/2) * x + 6)) :=
by { intro h, intro H, exact sorry }

end slope_of_line_l719_719465


namespace tan_A_is_correct_l719_719247

def triangle_AB (a b c : ℝ) : Prop :=
a^2 + b^2 = c^2

theorem tan_A_is_correct :
  ∀ (A B C : ℝ),
    triangle_AB B C A →
    B = 13 →
    C = 2 * Real.sqrt 10 →
    Real.tan (Real.atan (C / A)) = (2 * Real.sqrt 1290) / 129 :=
by
  intro A B C
  intro h₁ h₂ h₃
  sorry

end tan_A_is_correct_l719_719247


namespace regular_tetrahedron_l719_719755

-- Define the types for points and tetrahedrons
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Tetrahedron :=
(A B C D : Point)
(insphere : Point)

-- Conditions
def sphere_touches_at_angle_bisectors (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_altitudes (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

def sphere_touches_at_medians (T : Tetrahedron) : Prop :=
-- Dummy implementation to define the condition (to be filled)
sorry

-- Main theorem statement
theorem regular_tetrahedron (T : Tetrahedron)
  (h1 : sphere_touches_at_angle_bisectors T)
  (h2 : sphere_touches_at_altitudes T)
  (h3 : sphere_touches_at_medians T) :
  T.A = T.B ∧ T.A = T.C ∧ T.A = T.D := 
sorry

end regular_tetrahedron_l719_719755


namespace initial_profit_percentage_l719_719718

theorem initial_profit_percentage
  (CP : ℝ)
  (h1 : CP = 2400)
  (h2 : ∀ SP : ℝ, 15 / 100 * CP = 120 + SP) :
  ∃ P : ℝ, (P / 100) * CP = 10 :=
by
  sorry

end initial_profit_percentage_l719_719718


namespace correct_inequality_l719_719494

noncomputable def f (x : ℝ) : ℝ := (f''(1) / 2) * (Real.exp (2 * x - 2)) + x^2 - 2 * (f(0)) * x
def g (x : ℝ) : ℝ

axiom f_condition : ∀ x : ℝ, f(x) = (f''(1) / 2) * (Real.exp (2*x - 2)) + x^2 - 2 * (f(0)) * x
axiom g_condition : ∀ x : ℝ, g''(x) + 2 * g(x) < 0

theorem correct_inequality : (g(2015) > f(2) * g(2017)) :=
    sorry

end correct_inequality_l719_719494


namespace infinite_series_equals_two_l719_719801

noncomputable def sum_series : ℕ → ℝ := λ k, (8^k : ℝ) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_equals_two :
  (∑' k : ℕ, if k > 0 then sum_series k else 0) = 2 :=
by 
  sorry

end infinite_series_equals_two_l719_719801


namespace intersecting_lines_l719_719309

-- Define the problem conditions and the expected result
def line1 (y a : ℚ) : ℚ := (1 / 3) * y + a
def line2 (x b : ℚ) : ℚ := (1 / 5) * x + b

theorem intersecting_lines (a b : ℚ) :
  (line1 5 a = 3) ∧ (line2 3 b = 5) → a + b = 86 / 15 :=
by
  intro h
  cases h with h1 h2
  sorry

end intersecting_lines_l719_719309


namespace range_of_a_l719_719889

noncomputable def f (x a : ℝ) := x^2 - a * x
noncomputable def g (x : ℝ) := Real.exp x
noncomputable def h (x : ℝ) := x - (Real.log x / x)

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (f x a = Real.log x)) ↔ (1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1) :=
by
  sorry

end range_of_a_l719_719889


namespace sum_k_squared_div_3_k_l719_719451

theorem sum_k_squared_div_3_k : ∑ k in (Finset.range n).map (λ x, x+1), (k^2 / (3^k : ℝ)) = 4 :=
by
  sorry

end sum_k_squared_div_3_k_l719_719451


namespace find_point_P_l719_719866

noncomputable def point := ℝ × ℝ

noncomputable def A : point := (4, -3)
noncomputable def B : point := (2, -1)
noncomputable def l (P : point) : ℝ := 4 * P.1 + 3 * P.2 - 2

def distance (P Q : point) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Equation of the perpendicular bisector of AB
def perp_bisector (P : point) : Prop := P.1 - P.2 - 5 = 0

-- Distance from point to line l equals 2
def dist_to_line (P : point) : Prop :=
  abs (4 * P.1 + 3 * P.2 - 2) / real.sqrt (4^2 + 3^2) = 2

theorem find_point_P : ∃ P : point, distance P A = distance P B ∧ dist_to_line P ∧ perp_bisector P ∧ P = (1, -4) :=
by
  sorry

end find_point_P_l719_719866


namespace find_n_l719_719264

theorem find_n (n : ℕ) (h : (37.5^n + 26.5^n : ℝ).natAbs > 0) : n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 :=
sorry

end find_n_l719_719264


namespace dot_product_proof_l719_719508

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (cos_theta : ℝ) (mag_a mag_b : ℝ)
variables (h_cos_theta : cos_theta = 1 / 3)
variables (h_mag_a : ∥a∥ = 1)
variables (h_mag_b : ∥b∥ = 3)

theorem dot_product_proof
  (h_cos_theta : cos_theta = 1 / 3)
  (h_mag_a : ∥a∥ = 1)
  (h_mag_b : ∥b∥ = 3) :
  (2 • a + b) • b = 11 :=
sorry

end dot_product_proof_l719_719508


namespace people_stools_chairs_l719_719228

def total_legs (x y z : ℕ) : ℕ := 2 * x + 3 * y + 4 * z 

theorem people_stools_chairs (x y z : ℕ) : 
  (x > y) → (x > z) → (x < y + z) → (total_legs x y z = 32) → 
  (x = 5 ∧ y = 2 ∧ z = 4) :=
by
  intro h1 h2 h3 h4
  sorry

end people_stools_chairs_l719_719228


namespace largest_result_l719_719909

theorem largest_result (P Q : ℝ) (hP : P = 1000) (hQ : Q = 0.01) :
  (\[ P + Q, P * Q, P / Q, Q / P, P - Q \]).maximum = P / Q := sorry

end largest_result_l719_719909


namespace derivative_at_one_l719_719193

noncomputable def f (x : ℝ) : ℝ := cos x - (1 / x)

theorem derivative_at_one : deriv f 1 = 1 - real.sin 1 :=
by
  sorry

end derivative_at_one_l719_719193


namespace tessellation_coloring_l719_719930

theorem tessellation_coloring :
  ∀ (T : Type) (colors : T → ℕ) (adjacent : T → T → Prop),
    (∀ t1 t2, adjacent t1 t2 → colors t1 ≠ colors t2) → 
    (∃ c1 c2 c3, ∀ t, colors t = c1 ∨ colors t = c2 ∨ colors t = c3) :=
sorry

end tessellation_coloring_l719_719930


namespace students_in_trumpet_or_trombone_l719_719713

theorem students_in_trumpet_or_trombone (h₁ : 0.5 + 0.12 = 0.62) : 
  0.5 + 0.12 = 0.62 :=
by
  exact h₁

end students_in_trumpet_or_trombone_l719_719713


namespace arithmetic_seq_a_arithmetic_seq_b_arithmetic_seq_c_arithmetic_seq_d_l719_719036

-- (a)
theorem arithmetic_seq_a (a₁ a₂ a₃ a₄ d : ℤ) (ha₁ : a₁ = 20) (ha₂ : a₂ = 13) (ha₃ : a₃ = 6) (ha₄ : a₄ = -1) (hd : d = a₂ - a₁) :
  a₁ + 4 * d = -8 ∧ a₁ + 5 * d = -15 := sorry

-- (b)
theorem arithmetic_seq_b (a₁ a₅ d : ℤ) (a₂ a₃ a₄ : ℤ) (ha₁ : a₁ = 2) (ha₅ : a₅ = 14)
  (hd : d = (a₅ - a₁) / 4) (ha₂ : a₂ = a₁ + d) (ha₃ : a₃ = ha₂ + d) (ha₄ : a₄ = ha₃ + d) :
  a₂ = 5 ∧ a₃ = 8 ∧ a₄ = 11 := sorry

-- (c)
theorem arithmetic_seq_c (a₁ a₂ t d : ℤ) (ht_seq : (a₁ = 7 ∨ a₂ = 7) ∧ (a₁ = 15 ∨ a₂ = 15))
  (hd : d = a₂ - a₁ ∨ d = (15 - 7) / 2) :
  (t = 23 ∨ t = -1 ∨ t = 11):= sorry

-- (d)
theorem arithmetic_seq_d (r s w x y z d : ℤ) (hseq : r ∈ [4, 20] ∨ z ∈ [4, 20])
  (hr_pre : r < z) (hd : d = (20 - 4) / 4 ∨ d = (16) / 4 ∨ d = (4) ∨ d = (26.67)):
  max (z - r) 80 ∧ min (z - r) 16 := sorry

end arithmetic_seq_a_arithmetic_seq_b_arithmetic_seq_c_arithmetic_seq_d_l719_719036


namespace percentage_employees_6_years_or_more_is_26_l719_719927

-- Define the units for different years of service
def units_less_than_2_years : ℕ := 4
def units_2_to_4_years : ℕ := 6
def units_4_to_6_years : ℕ := 7
def units_6_to_8_years : ℕ := 3
def units_8_to_10_years : ℕ := 2
def units_more_than_10_years : ℕ := 1

-- Define the total units
def total_units : ℕ :=
  units_less_than_2_years +
  units_2_to_4_years +
  units_4_to_6_years +
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- Define the units representing employees with 6 years or more of service
def units_6_years_or_more : ℕ :=
  units_6_to_8_years +
  units_8_to_10_years +
  units_more_than_10_years

-- The goal is to prove that this percentage is 26%
theorem percentage_employees_6_years_or_more_is_26 :
  (units_6_years_or_more * 100) / total_units = 26 := by
  sorry

end percentage_employees_6_years_or_more_is_26_l719_719927


namespace find_y_l719_719468

theorem find_y : (12 : ℝ)^3 * (2 : ℝ)^4 / 432 = 5184 → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end find_y_l719_719468


namespace roots_range_of_a_l719_719884

theorem roots_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + (a - 2)*|x - 3| + 9 - 2*a = 0) ↔ a > 0 ∨ a = -2 :=
sorry

end roots_range_of_a_l719_719884


namespace selling_price_correct_l719_719048

theorem selling_price_correct (C P_rate : ℝ) (hC : C = 50) (hP_rate : P_rate = 0.40) : 
  C + (P_rate * C) = 70 :=
by
  sorry

end selling_price_correct_l719_719048


namespace no_values_of_x_l719_719475

def P (x : ℝ) : ℂ := 1 + (complex.cos x + complex.sin x * complex.I) - (complex.cos (2 * x) + complex.sin (2 * x) * complex.I) + (complex.cos (3 * x) + complex.sin (3 * x) * complex.I)

theorem no_values_of_x (x : ℝ) (hx : 0 ≤ x ∧ x < 2 * real.pi) : P x ≠ 0 :=
  sorry

end no_values_of_x_l719_719475


namespace infinite_series_sum_eq_two_l719_719791

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l719_719791


namespace flour_more_than_sugar_l719_719965

-- Define the conditions.
def sugar_needed : ℕ := 9
def total_flour_needed : ℕ := 14
def salt_needed : ℕ := 40
def flour_already_added : ℕ := 4

-- Define the target proof statement.
theorem flour_more_than_sugar :
  (total_flour_needed - flour_already_added) - sugar_needed = 1 :=
by
  -- sorry is used here to skip the proof.
  sorry

end flour_more_than_sugar_l719_719965


namespace constant_ratio_of_chord_and_tangent_distance_l719_719287

theorem constant_ratio_of_chord_and_tangent_distance
  (R : ℝ) -- Radius of the circle
  (A B : ℝ) -- Representing the length of the chord AB
  (d : ℝ) -- Distance from A to the tangent at B
  (β : ℝ) -- Angle subtended by the chord at the center
  (hA : A = 2 * R * sin β) -- Chord length in terms of R and β
  (hd : d = A * sin β) -- Distance to the tangent in terms of A and β
  : (A^2 / d = 2 * R) :=
by
  sorry

end constant_ratio_of_chord_and_tangent_distance_l719_719287


namespace calculation_proof_l719_719692

-- Definitions of the square root expressions
def sqrt3 : ℝ := Real.sqrt 3
def sqrt2 : ℝ := Real.sqrt 2
def sqrt5 : ℝ := Real.sqrt 5
def sqrt32 : ℝ := Real.sqrt 32
def sqrt8 : ℝ := Real.sqrt 8

noncomputable def correct_calculation : Prop := (sqrt32 / sqrt8 = 2)

theorem calculation_proof : correct_calculation := 
  by sorry

end calculation_proof_l719_719692


namespace log_identity_of_exponentiation_l719_719869

theorem log_identity_of_exponentiation
  (a : ℝ)
  (h : log 10 a = -0.98) :
  0.01 ^ 0.01 = 1 / (10 * a) :=
by
  sorry

end log_identity_of_exponentiation_l719_719869


namespace two_identical_2x2_squares_l719_719640

def grid8x8 := fin 8 → fin 8 → bool

def is_2x2_square {α : Type*} (grid : α → α → bool) (x y : fin 7) : bool :=
  let top_left := grid x y,
      top_right := grid x (y.succ),
      bottom_left := grid (x.succ) y,
      bottom_right := grid (x.succ) (y.succ)
  in top_left ∧ top_right ∧ bottom_left ∧ bottom_right

theorem two_identical_2x2_squares (grid : grid8x8) :
  ∃ (x1 y1 x2 y2 : fin 7), (x1 ≠ x2 ∨ y1 ≠ y2) ∧
  is_2x2_square grid x1 y1 = is_2x2_square grid x2 y2 := 
sorry

end two_identical_2x2_squares_l719_719640


namespace lens_discount_l719_719611

theorem lens_discount :
  ∃ (P : ℚ), ∀ (D : ℚ),
    (300 - D = 240) →
    (P = (D / 300) * 100) →
    P = 20 :=
by
  sorry

end lens_discount_l719_719611


namespace problem_statement_l719_719273

noncomputable def f (x m : ℝ) : ℝ := x - abs (x + 2) - abs (x - 3) - m

theorem problem_statement :
  (∀ x : ℝ, (1 / m) - 4 ≥ f x m) → (m > 0 ∧ log (m + 2) / log (m + 1) > log (m + 3) / log (m + 2)) :=
by {
  sorry
}

end problem_statement_l719_719273


namespace ice_cream_flavors_l719_719531

-- We have four basic flavors and want to combine four scoops from these flavors.
def ice_cream_combinations : ℕ :=
  Nat.choose 7 3

theorem ice_cream_flavors : ice_cream_combinations = 35 :=
by
  sorry

end ice_cream_flavors_l719_719531


namespace slope_l3_l719_719957

-- Define the points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A := Point.mk 2 0
def B := Point.mk 1 2
def C := Point.mk 3 2

-- Define the lines l1, l2, and l3
def line_eq (p1 p2 : Point) (x y : ℝ) : Bool :=
  (p2.x - p1.x) * (y - p1.y) = (p2.y - p1.y) * (x - p1.x)

-- Define the equations of the lines
def l1 (x y : ℝ) : Prop := 2 * x + y = 4
def l2 (x y : ℝ) : Prop := y = 2

-- Condition: Line l3 has a positive slope and passes through A and intersects l2 at C
def l3 (p := A) (c := C) : Prop := line_eq p c p.x c.y

-- Define the area of a triangle
def area_triangle (A B C : Point) : ℝ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

-- Given conditions
def conditions : Prop :=
  l1 A.x A.y ∧
  l2 B.x B.y ∧
  l3 A C ∧
  area_triangle A B C = 2

-- Main statement: given the conditions, the slope of l3 is 2
theorem slope_l3 : conditions → (C.y - A.y) / (C.x - A.x) = 2 := by
  sorry

end slope_l3_l719_719957


namespace recover_password_exists_l719_719328

def ord (c : Char) : Nat :=
  if c.val <= 'я'.val then c.val - 'а'.val + 1 else c.val - 'а'.val + 2

def alphabet : List Char :=
  List.range 33 |>.map (λ n => Char.ofNat (n + ('а'.val - 1))) |>.filter (≠ 'и')

def c_sequence : List Nat := [2, 8, 5, 2, 8, 3, 1, 9, 8, 4, 1, 8, 4, 9, 7, 5]

noncomputable def password (xs : List Nat) (ys : List Nat) : String :=
  let x := xs.take 10
  let y := ys.take 4 ++ ys.take 4
  let cs := x.zip y |>.map (λ ⟨x, y⟩ => (x + y) % 10)
  if cs = c_sequence then
    ys.take 4.map (λ n => List.nth alphabet (n - 1)).asString
  else
    ""

theorem recover_password_exists
  (a b : Nat) (alphabet : List Char) (x0 : Nat)
  (seq_period : ∀ i, x0 = a * x0 + b % 10)
  (∀ i ≠ 0, seq_period i = seq_period 0):
  ∃ (p : String), p = "яхта" :=
by {
  sorry
}

end recover_password_exists_l719_719328


namespace Heather_total_distance_walked_l719_719526

theorem Heather_total_distance_walked :
  let d1 := 0.645
  let d2 := 1.235
  let d3 := 0.875
  let d4 := 1.537
  let d5 := 0.932
  (d1 + d2 + d3 + d4 + d5) = 5.224 := 
by
  sorry -- Proof goes here

end Heather_total_distance_walked_l719_719526


namespace solve_equation_l719_719994

open Function

theorem solve_equation (m n : ℕ) (h_gcd : gcd m n = 2) (h_lcm : lcm m n = 4) :
  m * n = (gcd m n)^2 + lcm m n ↔ (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) :=
by
  sorry

end solve_equation_l719_719994


namespace scarves_in_each_box_l719_719253

theorem scarves_in_each_box
  (num_boxes : ℕ)
  (mittens_per_box : ℕ)
  (total_clothing : ℕ)
  (total_boxes : num_boxes = 3)
  (mittens_each_box : mittens_per_box = 4)
  (total_clothes : total_clothing = 21) :
  (S : ℕ) (h : S * 3 + 12 = 21) : S = 3 :=
by
  sorry

end scarves_in_each_box_l719_719253


namespace registration_methods_l719_719919

theorem registration_methods :
  ∀ (interns : ℕ) (companies : ℕ), companies = 4 → interns = 5 → companies^interns = 1024 :=
by intros interns companies h1 h2; rw [h1, h2]; exact rfl

end registration_methods_l719_719919


namespace elroy_more_miles_l719_719103

theorem elroy_more_miles
  (rate_last_year : ℝ) (rate_this_year : ℝ) (total_earned_last_year : ℝ)
  (cost_per_5_miles : ℝ) : 
  rate_last_year = 4 →
  rate_this_year = 2.75 →
  total_earned_last_year = 44 →
  cost_per_5_miles = 3.50 →
  let miles_last_year := total_earned_last_year / rate_last_year in
  let cost_per_mile := cost_per_5_miles / 5 in
  let effective_rate_this_year := rate_this_year - cost_per_mile in
  let miles_elroy := total_earned_last_year / effective_rate_this_year in
  (miles_elroy - miles_last_year).ceil = 11 :=
by
  intros
  sorry

end elroy_more_miles_l719_719103


namespace profit_percentage_B_is_25_l719_719752

-- Define the given conditions
def cost_price_A : ℝ := 156
def selling_price_C : ℝ := 234
def profit_percentage_A : ℝ := 20 / 100

-- Calculate intermediate values based on conditions
def selling_price_B : ℝ := cost_price_A * (1 + profit_percentage_A)
def profit_B : ℝ := selling_price_C - selling_price_B

-- The Lean proof problem
theorem profit_percentage_B_is_25:
  (profit_B / selling_price_B) * 100 = 25 :=
by
  -- Assume the necessary calculations have been done
  sorry

end profit_percentage_B_is_25_l719_719752


namespace smallest_positive_integer_l719_719345

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end smallest_positive_integer_l719_719345


namespace HVAC_cost_per_vent_l719_719301

/-- 
The cost of Joe's new HVAC system is $20,000. It includes 2 conditioning zones, each with 5 vents.
Prove that the cost of the system per vent is $2,000.
-/
theorem HVAC_cost_per_vent
    (cost : ℕ := 20000)
    (zones : ℕ := 2)
    (vents_per_zone : ℕ := 5)
    (total_vents : ℕ := zones * vents_per_zone) :
    (cost / total_vents) = 2000 := by
  sorry

end HVAC_cost_per_vent_l719_719301


namespace david_course_hours_l719_719426

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end david_course_hours_l719_719426


namespace range_of_k_l719_719305

open_locale classical

def sequence (n : ℕ) (k : ℝ) : ℝ := n^2 - k * n

theorem range_of_k (k : ℝ) :
  (∀ n : ℕ, n > 0 → sequence n k ≥ sequence 3 k) →
  5 ≤ k ∧ k ≤ 7 :=
begin
  sorry
end

end range_of_k_l719_719305


namespace line_through_vertex_of_parabola_l719_719845

theorem line_through_vertex_of_parabola :
  ∃! a : ℝ, (∃ b : ℝ, (y = 2 * x + b) ∧ y = x^2 + a^2 ∧ ∀ x y, y = (2 * 0 + b) = (x^2 + a^2) ∧ (0, a^2) ) :=
sorry

end line_through_vertex_of_parabola_l719_719845


namespace functions_same_function_C_functions_same_function_D_l719_719693

theorem functions_same_function_C (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by sorry

theorem functions_same_function_D (x : ℝ) : x = (x^3)^(1/3) :=
by sorry

end functions_same_function_C_functions_same_function_D_l719_719693


namespace angle_ABM_in_square_l719_719935

theorem angle_ABM_in_square (A B C D M : Point) (h_square : square A B C D)
  (h1 : ∠DCM = 25) (h2 : ∠MAC = 25) : ∠ABM = 40 :=
by
  -- proof goes here
  sorry

end angle_ABM_in_square_l719_719935


namespace binomial_10_5_eq_252_l719_719088

theorem binomial_10_5_eq_252 : nat.choose 10 5 = 252 :=
by
  -- We will add "sorry" here as we are only required to state the theorem, not prove it.
  sorry

end binomial_10_5_eq_252_l719_719088


namespace inscribed_square_area_l719_719055

theorem inscribed_square_area (x y : ℝ) (h : (x^2 / 4) + (y^2 / 8) = 1) :
  let t := 2 * (Real.sqrt (8 / 3)) / 3 in
  let side_length := 2 * t in
  let area := side_length * side_length in
  area = 32 / 3 :=
by
  let t := Real.sqrt (8 / 3)
  let side_length := 2 * t / 3
  let area := side_length * side_length
  have : area = 32 / 3 := by sorry
  exact this

end inscribed_square_area_l719_719055


namespace fraction_diff_l719_719145

open Real

theorem fraction_diff (x y : ℝ) (hx : x = sqrt 5 - 1) (hy : y = sqrt 5 + 1) :
  (1 / x - 1 / y) = 1 / 2 := sorry

end fraction_diff_l719_719145


namespace tan_105_eq_neg2_sub_sqrt3_l719_719127

theorem tan_105_eq_neg2_sub_sqrt3 :
  let tan := Real.tan;
      A := 60 * Real.pi / 180;
      B := 45 * Real.pi / 180 in
  tan (A + B) = -2 - Real.sqrt 3 := by
  let t60 : ℝ := Real.sqrt 3
  let t45 : ℝ := 1
  have tan_sum : tan (A + B) = (t60 + t45) / (1 - t60 * t45) := by sorry
  show tan (60 * Real.pi / 180 + 45 * Real.pi / 180) = -2 - Real.sqrt 3 from by sorry

end tan_105_eq_neg2_sub_sqrt3_l719_719127


namespace problem_solution_l719_719943

theorem problem_solution (a b c : ℤ)
  (h1 : ∀ x : ℤ, |x| ≠ |a|)
  (h2 : ∀ x : ℤ, x^2 ≠ b^2)
  (h3 : ∀ x : ℤ, x * c ≤ 1):
  a + b + c = 0 :=
by sorry

end problem_solution_l719_719943


namespace evaluate_log_cube_l719_719104

theorem evaluate_log_cube :
  (log 10 (3 * log 10 1000))^3 = 0.8698 :=
by sorry

end evaluate_log_cube_l719_719104


namespace general_formula_b_n_sum_b_n_l719_719164

noncomputable def a_n (n: ℕ) : ℕ := n
def S_n (n: ℕ) : ℕ := (n * (n + 1)) / 2
noncomputable def b_n (n: ℕ) : ℚ := 2 / (n * (n + 1))

theorem general_formula_b_n (n : ℕ) (hn : n ≥ 1) : b_n n = 2 / (n * (n + 1)) :=
by 
sorry

theorem sum_b_n (n : ℕ) : (finset.range n).sum (λ k, b_n (k + 1)) < 2 :=
by 
sorry

end general_formula_b_n_sum_b_n_l719_719164


namespace volume_of_hexagonal_pyramid_l719_719131

theorem volume_of_hexagonal_pyramid (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  volume_of_pyramid h r = (2 * h^3 * (h^2 - r^2)) / (3 * r^2) :=
by
  sorry

end volume_of_hexagonal_pyramid_l719_719131


namespace correct_time_after_2011_minutes_l719_719007

def time_2011_minutes_after_midnight : String :=
  "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM"

theorem correct_time_after_2011_minutes :
  time_2011_minutes_after_midnight = "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM" :=
sorry

end correct_time_after_2011_minutes_l719_719007


namespace mom_saving_factor_l719_719079

def daily_saving : ℝ := 50 - 15

def total_saving_before_parents (days: ℕ) : ℝ :=
  days * daily_saving

def total_saving_after_dad (days: ℕ) : ℝ :=
  total_saving_before_parents days + 10

theorem mom_saving_factor (days : ℕ) (final_amount : ℝ) (x : ℝ) 
  (h1 : daily_saving = 35) 
  (h2 : days = 7) 
  (h3 : final_amount = 500) 
  (h4 : total_saving_after_dad days = 255) 
  : x = final_amount / total_saving_after_dad days :=
by
  -- Using given conditions
  have h5 : final_amount = 500 := h3
  have h6 : total_saving_after_dad 7 = 255 := h4
  -- Calculate x
  have x_calc : x = 500 / 255 := by
    rw [h5, h6]
  rw [x_calc]
  sorry

end mom_saving_factor_l719_719079


namespace f_25_over_11_neg_l719_719945

variable (f : ℚ → ℚ)
axiom f_mul : ∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b
axiom f_prime : ∀ p : ℕ, Prime p → f p = p

theorem f_25_over_11_neg : f (25 / 11) < 0 :=
by
  -- You can prove the necessary steps during interactive proof:
  -- Using primes 25 = 5^2 and 11 itself,
  -- f (25/11) = f 25 - f 11. Thus, f (25) = 2f(5) = 2 * 5 = 10 and f(11) = 11
  -- Therefore, f (25/11) = 10 - 11 = -1 < 0.
  sorry

end f_25_over_11_neg_l719_719945


namespace unique_solution_of_satisfying_function_l719_719822

noncomputable def satisfying_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f(x) + f(y) ≥ x * y) ∧ (∀ x : ℝ, ∃ y : ℝ, f(x) + f(y) = x * y)

theorem unique_solution_of_satisfying_function (f : ℝ → ℝ) :
  satisfying_function f → (∀ x : ℝ, f x = x^2 / 2) :=
by
  intro h
  -- Proof is omitted
  sorry

end unique_solution_of_satisfying_function_l719_719822


namespace calculate_dividend_l719_719020

def faceValue : ℕ := 100
def premiumPercent : ℕ := 20
def dividendPercent : ℕ := 5
def investment : ℕ := 14400
def costPerShare : ℕ := faceValue + (premiumPercent * faceValue / 100)
def numberOfShares : ℕ := investment / costPerShare
def dividendPerShare : ℕ := faceValue * dividendPercent / 100
def totalDividend : ℕ := numberOfShares * dividendPerShare

theorem calculate_dividend :
  totalDividend = 600 := 
by
  sorry

end calculate_dividend_l719_719020


namespace find_n_l719_719097

theorem find_n :
  ∃ n : ℤ, 3 ^ 3 - 7 = 4 ^ 2 + 2 + n ∧ n = 2 :=
by
  use 2
  sorry

end find_n_l719_719097


namespace raisin_cookies_difference_l719_719527

-- Definitions based on conditions:
def raisin_cookies_baked_yesterday : ℕ := 300
def raisin_cookies_baked_today : ℕ := 280

-- Proof statement:
theorem raisin_cookies_difference : raisin_cookies_baked_yesterday - raisin_cookies_baked_today = 20 := 
by
  sorry

end raisin_cookies_difference_l719_719527


namespace number_of_functions_l719_719118

theorem number_of_functions (a b c d : ℝ) :
  (∀ f : ℝ → ℝ, 
    (f = fun x => ax^3 + bx^2 + cx + d) → 
    (∀ x, f(x) * f(-x) = f(x^3)) → 
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 1 ∧ d = 0) ∨
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 1 ∧ b = 0 ∧ c = 1 ∧ d = 0)) :=
begin
  sorry
end

end number_of_functions_l719_719118


namespace remainder_of_Q_divided_by_quadratic_l719_719261

noncomputable def Q : ℚ[X] := sorry

theorem remainder_of_Q_divided_by_quadratic :
  Q.eval 17 = 41 ∧ Q.eval 93 = 13 →
  ∃ a b : ℚ, a = -7 / 19 ∧ b = 900 / 19 ∧ 
  ∀ x, Q(x) = (x - 17) * (x - 93) * (Q / ((x - 17) * (x - 93))) + (a * x + b) :=
begin
  sorry,
end

end remainder_of_Q_divided_by_quadratic_l719_719261


namespace loss_percent_two_thirds_l719_719771

theorem loss_percent_two_thirds (CP SP : ℝ) (hSP : SP = 1.35 * CP) : 
  let SP' := (2 / 3) * SP in
  ((CP - SP') / CP) * 100 = 10 :=
by
  sorry

end loss_percent_two_thirds_l719_719771


namespace sin_330_eq_neg_half_l719_719440

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by sorry

end sin_330_eq_neg_half_l719_719440


namespace expected_potato_yield_l719_719970

-- Definitions based on the conditions
def steps_length : ℕ := 3
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def yield_rate : ℚ := 3 / 4

-- Calculate the dimensions in feet
def garden_length_feet : ℕ := garden_length_steps * steps_length
def garden_width_feet : ℕ := garden_width_steps * steps_length

-- Calculate the area in square feet
def garden_area_feet : ℕ := garden_length_feet * garden_width_feet

-- Calculate the expected yield in pounds
def expected_yield_pounds : ℚ := garden_area_feet * yield_rate

-- The theorem to prove the expected yield
theorem expected_potato_yield :
  expected_yield_pounds = 3037.5 := by
  sorry  -- Proof is omitted as per the instructions.

end expected_potato_yield_l719_719970


namespace valid_combinations_l719_719063

theorem valid_combinations (h s ic : ℕ) (h_eq : h = 4) (s_eq : s = 6) (ic_eq : ic = 3) :
  h * s - ic = 21 := by
  rw [h_eq, s_eq, ic_eq]
  norm_num

end valid_combinations_l719_719063


namespace problem_b_correct_l719_719850

theorem problem_b_correct (a b : ℝ) (h₁ : a < 0) (h₂ : 0 < b) (h₃ : b < 1) : (ab^2 > ab ∧ ab > a) :=
by
  sorry

end problem_b_correct_l719_719850


namespace max_and_min_balls_in_fullest_box_l719_719284

theorem max_and_min_balls_in_fullest_box :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (max a (max b c) = 8) ∧ (min a (min b c) = 5) :=
begin
  -- Assume a, b, and c are the number of balls in the three boxes
  -- with a ≤ b ≤ c (without loss of generality).
  sorry
end

end max_and_min_balls_in_fullest_box_l719_719284


namespace rotated_function_is_inverse_neg_l719_719908

variable {X Y : Type} [TopologicalSpace X] [TopologicalSpace Y]
variable (f : X → Y) (f_inv : Y → X)

-- Conditions
def has_inverse_function : Prop := ∀ x, f_inv (f x) = x

-- Proposition: rotated function after 90-degrees counterclockwise rotation
theorem rotated_function_is_inverse_neg
  (hf : has_inverse_function f f_inv) :
  ∀ x, let y := f x in y = f_inv (-x) :=
sorry

end rotated_function_is_inverse_neg_l719_719908


namespace number_of_sets_of_positive_integers_l719_719125

theorem number_of_sets_of_positive_integers : 
  ∃ n : ℕ, n = 3333 ∧ ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → x < y → y < z → x + y + z = 203 → n = 3333 :=
by
  sorry

end number_of_sets_of_positive_integers_l719_719125


namespace original_group_size_l719_719046

theorem original_group_size (M : ℕ) 
  (h1 : ∀ work_done_by_one, work_done_by_one = 1 / (6 * M))
  (h2 : ∀ work_done_by_one, work_done_by_one = 1 / (12 * (M - 4))) : 
  M = 8 :=
by
  sorry

end original_group_size_l719_719046


namespace minimum_value_l719_719603

theorem minimum_value (a b : ℝ) (h1 : 2 * a + 3 * b = 5) (h2 : a > 0) (h3 : b > 0) : 
  (1 / a) + (1 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_l719_719603


namespace problem_statement_l719_719184

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ ⦃x y⦄, x > 4 → y > x → f y < f x)
                          (h2 : ∀ x, f (4 + x) = f (4 - x)) : f 3 > f 6 :=
by 
  sorry

end problem_statement_l719_719184


namespace determine_angle_C_l719_719924

-- Definitions for sides and angles in a triangle and area
variables (a b c S C : ℝ)

-- Condition: Area expressed in two different forms
axiom area_formula_1 : S = (sqrt 3 / 4) * (a^2 + b^2 - c^2)
axiom area_formula_2 : S = (1/2) * a * b * sin C

-- The proof problem: Prove that angle C is pi/3 given the conditions
theorem determine_angle_C (h1 : area_formula_1) (h2 : area_formula_2) :
  C = π / 3 :=
sorry

end determine_angle_C_l719_719924


namespace exists_convex_337_gon_with_area_lt_0_1_l719_719486

-- Noncomputable because we are dealing with approximations, and exact irrational numbers are typically noncomputable
noncomputable def problem_statement : Prop :=
  let radius : ℝ := 1 in
  let points : ℕ := 2022 in
  let n_gon_sides : ℕ := 337 in
  let pi_approx : ℝ := 3.142 in
  let sqrt3_approx : ℝ := 1.732 in
  ∃ (polygon_area : ℝ), 
    (polygon_area < 0.1) ∧ 
    (∃ polygon : list (ℝ × ℝ), 
      polygon.length = n_gon_sides ∧ 
      (∀ point ∈ polygon, point.1^2 + point.2^2 = radius^2))

-- Theorems
theorem exists_convex_337_gon_with_area_lt_0_1 : problem_statement :=
  sorry

end exists_convex_337_gon_with_area_lt_0_1_l719_719486


namespace max_sin_cos_eq_two_l719_719313

theorem max_sin_cos_eq_two : ∀ x : ℝ, sin (2 * x) + cos (2 * x) ≤ 2 := 
by sorry

end max_sin_cos_eq_two_l719_719313


namespace equilateral_triangle_area_decrease_l719_719772

theorem equilateral_triangle_area_decrease 
  (A : ℝ)
  (a : ℝ)
  (h1 : A = 81 * Real.sqrt 3)
  (h2 : ∃ s : ℝ, s > 0 ∧ (A = s^2 * Real.sqrt 3 / 4))
  (h3 : a = 6) :
  ∃ dA : ℝ, dA = 45 * Real.sqrt 3 :=
by
  have h4 : ∃ s : ℝ, s > 0 ∧ (s^2 = 324),
  { use 18, norm_num, }
  have h5 : ∃ A' : ℝ, A' = ((18 - 6)^2 * Real.sqrt 3 / 4),
  { use 36 * Real.sqrt 3, norm_num, }
  have h6 : ∃ dA : ℝ, dA = A - (36 * Real.sqrt 3),
  { use 45 * Real.sqrt 3, norm_num, }
  exact h6

end equilateral_triangle_area_decrease_l719_719772


namespace total_boat_license_combinations_l719_719758

theorem total_boat_license_combinations : 
  let letters := 3 in
  let digits := 10 ^ 6 in
  letters * digits = 3_000_000 :=
by
  sorry

end total_boat_license_combinations_l719_719758


namespace bottle_caps_total_l719_719080

-- Define the conditions
def groups : ℕ := 7
def caps_per_group : ℕ := 5

-- State the theorem
theorem bottle_caps_total : groups * caps_per_group = 35 :=
by
  sorry

end bottle_caps_total_l719_719080


namespace translated_vector_ab_l719_719171

-- Define points A and B, and vector a
def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (5, 2)
def a : ℝ × ℝ := (1, 2)

-- Define the vector AB
def vectorAB : ℝ × ℝ :=
  let (Ax, Ay) := A
  let (Bx, By) := B
  (Bx - Ax, By - Ay)

-- Prove that after translating vector AB by vector a, the result remains (2, -5)
theorem translated_vector_ab :
  vectorAB = (2, -5) := by
  sorry

end translated_vector_ab_l719_719171


namespace proof_problem_l719_719953

variables {R : Type*} [Field R] (p q r u v w : R)

theorem proof_problem (h₁ : 15*u + q*v + r*w = 0)
                      (h₂ : p*u + 25*v + r*w = 0)
                      (h₃ : p*u + q*v + 50*w = 0)
                      (hp : p ≠ 15)
                      (hu : u ≠ 0) : 
                      (p / (p - 15) + q / (q - 25) + r / (r - 50)) = 1 := 
by sorry

end proof_problem_l719_719953


namespace a_2_eq_half_a_3_eq_third_a_4_eq_quarter_general_term_sum_n_terms_l719_719203

noncomputable def a : ℕ+ → ℝ
| 1 => 1
| (n + 1) => a n / (1 + a n)

def sequence_property (n : ℕ+) : Prop := a (n + 1) = a n / (1 + a n)

theorem a_2_eq_half : a 2 = 1 / 2 := sorry
theorem a_3_eq_third : a 3 = 1 / 3 := sorry
theorem a_4_eq_quarter : a 4 = 1 / 4 := sorry

theorem general_term (n : ℕ+) : a n = 1 / n := sorry

theorem sum_n_terms (n : ℕ) : 
  (∑ i in Finset.range (n+1), a (.succPNat (i : ℕ+)) * a (Nat.succ (.succPNat (i : ℕ+)))) = n / (n + 1) := sorry

end a_2_eq_half_a_3_eq_third_a_4_eq_quarter_general_term_sum_n_terms_l719_719203


namespace inbox_emails_after_movements_l719_719627

def initial_emails := 400
def trash_emails := initial_emails / 2
def remaining_emails := initial_emails - trash_emails
def work_emails := 0.4 * remaining_emails
def final_inbox_emails := remaining_emails - work_emails

theorem inbox_emails_after_movements : final_inbox_emails = 120 :=
by
  sorry

end inbox_emails_after_movements_l719_719627


namespace value_range_a_l719_719670

theorem value_range_a (a : ℝ) :
  (∀ (x : ℝ), |x + 2| * |x - 3| ≥ 4 / (a - 1)) ↔ (a < 1 ∨ a = 3) :=
by
  sorry

end value_range_a_l719_719670


namespace number_of_distinct_pairs_l719_719469

theorem number_of_distinct_pairs : 
  ∃ (n : ℕ), n = 8 ∧ 
    ∃ (f m : ℕ), 
      (∀ (f m : ℕ), m ≥ 0 ∧ f ≥ 0 ∧ 
        (∃ arrangement : list (ℕ → Prop), arrangement.length = 5 ∧ 
          ∀ i, arrangement.nth i = some (λ x, 
              (if x = 1 then ∃ i, arrangement[(i + 1) % 5] = 0 else True))) ∧ 
      (∑ i in finset.range 5, if arrangement.nth i = some 1 then 1 else 0) = f ∧ 
      (∑ i in finset.range 5, if arrangement.nth i = some 0 then 1 else 0) = m) 
    sorry

end number_of_distinct_pairs_l719_719469


namespace quadratic_discriminant_zero_l719_719476

theorem quadratic_discriminant_zero (m : ℝ) : (x^2 - 4*x + m = 0) ∧ (let Δ := (-4)^2 - 4*1*m in Δ = 0) → m = 4 := by
  sorry

end quadratic_discriminant_zero_l719_719476


namespace orange_juice_percentage_is_correct_l719_719968

def pear_juice_per_pear : ℝ := 10 / 2
def orange_juice_per_orange : ℝ := 6 / 3
def total_pear_juice : ℝ := 4 * pear_juice_per_pear
def total_orange_juice : ℝ := 6 * orange_juice_per_orange
def total_juice : ℝ := total_pear_juice + total_orange_juice
def orange_juice_percentage : ℝ := 100 * total_orange_juice / total_juice

theorem orange_juice_percentage_is_correct :
  orange_juice_percentage = 37.5 := by
  sorry

end orange_juice_percentage_is_correct_l719_719968


namespace find_m_max_area_l719_719219

-- Given conditions and questions translated to Lean 4

theorem find_m {A B C : ℝ} {a b c : ℝ} 
  (h1 : √2 * sin A = √3 * cos A) (h2 : a^2 - c^2 = b^2 - m * b * c) :
  m = 1 :=
begin
  sorry -- Proof comes here
end

theorem max_area {A B C : ℝ} {a b c : ℝ} 
  (h1 : √2 * sin A = √3 * cos A) (h2 : a = 2) :
  let area := (1/2) * b * c * sin A in
  area ≤ √3 :=
begin
  sorry -- Proof comes here
end

end find_m_max_area_l719_719219


namespace quadrilateral_is_cyclic_l719_719381

-- Step 1: Define the problem's conditions.
variables {A B C M N : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space N]
variables (triangle_ABC : is_regular_triangle A B C)
variables (line_through_A : ∀ x, x ∈ line A ∧ x ∉ segment B C)
variables (M_on_line : M ∈ line_through_A)
variables (N_on_line : N ∈ line_through_A)
variable (equal_lengths : dist A M = dist A N ∧ dist A N = dist A B ∧ dist A B = dist A C)
variable (B_inside_angle : B ∈ interior_angle M A C)

-- Step 2: State the theorem to prove the quadrilateral is cyclic.
theorem quadrilateral_is_cyclic :
  is_cyclic_quadrilateral (quadrilateral.mk A B C N M) :=
sorry

end quadrilateral_is_cyclic_l719_719381


namespace simplify_fraction_addition_l719_719623

theorem simplify_fraction_addition (a b : ℚ) (h1 : a = 4 / 252) (h2 : b = 17 / 36) :
  a + b = 41 / 84 := 
by
  sorry

end simplify_fraction_addition_l719_719623


namespace total_votes_l719_719231

theorem total_votes (A B C D E : ℕ)
  (votes_A : ℕ) (votes_B : ℕ) (votes_C : ℕ) (votes_D : ℕ) (votes_E : ℕ)
  (dist_A : votes_A = 38 * A / 100)
  (dist_B : votes_B = 28 * B / 100)
  (dist_C : votes_C = 11 * C / 100)
  (dist_D : votes_D = 15 * D / 100)
  (dist_E : votes_E = 8 * E / 100)
  (redistrib_A : votes_A' = votes_A + 5 * A / 100)
  (redistrib_B : votes_B' = votes_B + 5 * B / 100)
  (redistrib_D : votes_D' = votes_D + 2 * D / 100)
  (total_A : votes_A' = 7320) :
  A = 17023 := 
sorry

end total_votes_l719_719231


namespace trajectory_of_point_l719_719913

theorem trajectory_of_point (x y : ℝ) (P A : ℝ × ℝ × ℝ) (hP : P = (x, y, 0)) (hA : A = (0, 0, 4)) (hPA : dist P A = 5) : 
  x^2 + y^2 = 9 :=
by sorry

end trajectory_of_point_l719_719913


namespace find_K_l719_719753

variables (B G P Q K : ℕ)
variables (a L : ℝ)
variables (h_B : B > 0) (h_G : G > 0) (h_P : P > 0) (h_Q : Q > 0) (h_K : K > 0)
variables (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ K ∧ 
  G ≠ P ∧ G ≠ Q ∧ G ≠ K ∧ P ≠ Q ∧ P ≠ K ∧ Q ≠ K)

/- Conditions from the problem -/
axiom thickness_geography_books : ∀ a, (∃ b : ℝ, b = 2 * a) 
axiom shelf_filled_with_biology_and_geography_books : ∀ a L : ℝ, (B > 0 ∧ G > 0) → ((B * a + G * (2 * a)) = L)
axiom shelf_filled_with_new_arrangement : ∀ a L : ℝ, (P > 0 ∧ Q > 0) → ((P * a + Q * (2 * a)) = L)
axiom shelf_filled_with_biology_books_alone : ∀ a L : ℝ, (K > 0) → (K * a = L)

/- Prove the final statement: -/
theorem find_K (B G P Q K : ℕ) (h_thickness_geography_books : thickness_geography_books a)
  (h_shelf_filled_with_biology_and_geography_books : shelf_filled_with_biology_and_geography_books a L)
  (h_shelf_filled_with_new_arrangement : shelf_filled_with_new_arrangement a L)
  (h_shelf_filled_with_biology_books_alone : shelf_filled_with_biology_books_alone a L)
  (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ K ∧ G ≠ P ∧ G ≠ Q ∧ G ≠ K ∧ P ≠ Q ∧ P ≠ K ∧ Q ≠ K): K = B + 2G :=
by 
  sorry

end find_K_l719_719753


namespace b6_b8_equals_16_l719_719861

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def b_seq : ℕ → ℝ := sorry

axiom a_arithmetic : ∃ d, ∀ n, a_seq (n + 1) = a_seq n + d
axiom b_geometric : ∃ r, ∀ n, b_seq (n + 1) = b_seq n * r
axiom a_nonzero : ∀ n, a_seq n ≠ 0
axiom a_eq : 2 * a_seq 3 - (a_seq 7)^2 + 2 * a_seq 11 = 0
axiom b7_eq_a7 : b_seq 7 = a_seq 7

theorem b6_b8_equals_16 : b_seq 6 * b_seq 8 = 16 := by
  sorry

end b6_b8_equals_16_l719_719861


namespace total_books_l719_719673

theorem total_books (num_boxes : ℕ) (books_per_box : ℕ) : 
  num_boxes = 5 → books_per_box = 20 → num_boxes * books_per_box = 100 :=
by
  intros h_num_boxes h_books_per_box
  rw h_num_boxes
  rw h_books_per_box
  exact calc
    5 * 20 = 100 : by norm_num

end total_books_l719_719673


namespace find_a_l719_719914

-- Define the conditions given in the problem
def binomial_term (r : ℕ) (a : ℝ) : ℝ :=
  Nat.choose 7 r * 2^(7-r) * (-a)^r

def coefficient_condition (a : ℝ) : Prop :=
  binomial_term 5 a = 84

-- The theorem stating the problem's solution
theorem find_a (a : ℝ) (h : coefficient_condition a) : a = -1 :=
  sorry

end find_a_l719_719914


namespace positive_numbers_satisfy_condition_l719_719111

theorem positive_numbers_satisfy_condition : 
  ∃ (x : Fin 10 → ℝ), 
    (∀ k : Fin 10, (∑ i in Finset.range k.val.succ, x i) * (∑ i in Finset.range (10 - k.val), x (⟨i + k.val, Nat.lt_of_lt_pred (Nat.lt_of_succ_le (le_of_lt k.property))⟩)) = 1) ∧ 
    (∀ i : Fin 10, 0 < x i) ∧ 
    x 0 = (Real.sqrt 6 - Real.sqrt 2) / 2 ∧ 
    x 9 = (Real.sqrt 6 - Real.sqrt 2) / 2 ∧ 
    x 1 = Real.sqrt 2 - (Real.sqrt 6) / 2 ∧ 
    x 8 = Real.sqrt 2 - (Real.sqrt 6) / 2 ∧ 
    x 2 = (2 * Real.sqrt 6 - 3 * Real.sqrt 2) / 6 ∧ 
    x 7 = (2 * Real.sqrt 6 - 3 * Real.sqrt 2) / 6 ∧ 
    x 3 = (9 * Real.sqrt 2 - 5 * Real.sqrt 6) / 6 ∧ 
    x 6 = (9 * Real.sqrt 2 - 5 * Real.sqrt 6) / 6 ∧ 
    x 4 = (3 * Real.sqrt 6 - 5 * Real.sqrt 2) / 4 ∧ 
    x 5 = (3 * Real.sqrt 6 - 5 * Real.sqrt 2) / 4 :=
  sorry

end positive_numbers_satisfy_condition_l719_719111


namespace min_distance_ST_l719_719562

theorem min_distance_ST : 
  let F := (⟨1 / 2, 0⟩ : ℝ × ℝ),
  let line_p := { p : ℝ × ℝ // p.1 = -1 / 2 },
  (Q := fun (p : line_p) => ⟨0, p.2 / 2⟩),
  (M := fun (p : line_p) (λ : ℝ) => (λ * (1 / 2), λ * 0)),
  ∃ S T : ℝ × ℝ, 
    let C := { p : ℝ × ℝ // (p.1 - 3)^2 + p.2^2 = 2 },
    S ∈ C ∧ T ∈ C ∧
    let d := dist S T,
    d = (√30 * 2) / 5 := 
  sorry

end min_distance_ST_l719_719562


namespace decrement_value_is_15_l719_719650

noncomputable def decrement_value (n : ℕ) (original_mean updated_mean : ℕ) : ℕ :=
  (n * original_mean - n * updated_mean) / n

theorem decrement_value_is_15 : decrement_value 50 200 185 = 15 :=
by
  sorry

end decrement_value_is_15_l719_719650


namespace problem_proof_l719_719852

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * x + 2 - x

-- Condition given in the problem
axiom h : ∃ a : ℝ, f a = 3

-- Theorem statement
theorem problem_proof : ∃ a : ℝ, f a = 3 → f (2 * a) = 7 :=
by
  sorry

end problem_proof_l719_719852


namespace truck_covered_distance_l719_719367

theorem truck_covered_distance (t : ℝ) (d_bike : ℝ) (d_truck : ℝ) (v_bike : ℝ) (v_truck : ℝ) :
  t = 8 ∧ d_bike = 136 ∧ v_truck = v_bike + 3 ∧ d_bike = v_bike * t →
  d_truck = v_truck * t :=
by
  sorry

end truck_covered_distance_l719_719367


namespace max_capacity_l719_719402

variable (H r h : ℝ)
variable (V_water : ℝ := 50)
variable (cylinder_volume : ℝ → ℝ → ℝ := λ r h, π * r^2 * h)

-- Given conditions
axiom half_height_condition : h = H / 2
axiom volume_condition : cylinder_volume r h = V_water

-- Prove that the maximum capacity of the container is 400 liters
theorem max_capacity (H r h : ℝ) (half_height_condition : h = H / 2) (volume_condition : π * r^2 * h = 50) : 
  (π * (2 * r)^2 * (2 * h)) = 400 :=
by
  sorry

end max_capacity_l719_719402


namespace sequence_general_term_l719_719157

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n^2 + n + 5) →
  (a 1 = 7) →
  (∀ n, n ≥ 2 → a n = 2 * n) →
  ∀ n, a n = if n = 1 then 7 else 2 * n :=
begin
  intros hS ha1 ha_gen n,
  by_cases h : n = 1,
  { rw h, exact ha1 },
  { have hn2 : n ≥ 2, 
    { linarith [nat.succ_le_of_lt (nat.one_lt_of_ne_zero (ne_of_gt (nat.pos_of_ne_zero (hne.symm h))))] },
    rw if_neg h,
    exact ha_gen n hn2,
  }
end sorry

end sequence_general_term_l719_719157


namespace part_a_part_b_part_c_l719_719029

-- Define the necessary conditions and trigonometric terms
variables {S : Type*} {a b c : S} {α β γ : S} 
variables {sin : S → S → ℝ}
variables {angle_ab_aα angle_ac_aα angle_bc_bβ angle_ba_bβ angle_ca_cγ angle_cb_cγ : ℝ}
variables {angle_b_α angle_c_α angle_c_β angle_a_β angle_a_γ angle_b_γ : ℝ}

-- Part (a)
theorem part_a (h1 : sin angle_ab_aα ≠ 0) (h2 : sin angle_ac_aα ≠ 0)
  (h3 : sin angle_bc_bβ ≠ 0) (h4 : sin angle_ba_bβ ≠ 0)
  (h5 : sin angle_ca_cγ ≠ 0) (h6 : sin angle_cb_cγ ≠ 0)
  (h7 : sin angle_b_α ≠ 0) (h8 : sin angle_c_α ≠ 0)
  (h9 : sin angle_c_β ≠ 0) (h10 : sin angle_a_β ≠ 0)
  (h11 : sin angle_a_γ ≠ 0) (h12 : sin angle_b_γ ≠ 0) : 
  (sin angle_ab_aα / sin angle_ac_aα) * (sin angle_bc_bβ / sin angle_ba_bβ) * (sin angle_ca_cγ / sin angle_cb_cγ) = 
  (sin angle_b_α / sin angle_c_α) * (sin angle_c_β / sin angle_a_β) * (sin angle_a_γ / sin angle_b_γ) := sorry

-- Part (b)
theorem part_b (h1 : sin angle_ab_aα ≠ 0) (h2 : sin angle_ac_aα ≠ 0)
  (h3 : sin angle_bc_bβ ≠ 0) (h4 : sin angle_ba_bβ ≠ 0)
  (h5 : sin angle_ca_cγ ≠ 0) (h6 : sin angle_cb_cγ ≠ 0) : 
  (sin angle_ab_aα / sin angle_ac_aα) * (sin angle_bc_bβ / sin angle_ba_bβ) * (sin angle_ca_cγ / sin angle_cb_cγ) = 1 ↔ 
  (∃ l : S, ∀ p : S, p ∈ {α, β, γ} → p ∈ l) := sorry

-- Part (c)
theorem part_c (h1 : sin angle_ab_aα ≠ 0) (h2 : sin angle_ac_aα ≠ 0)
  (h3 : sin angle_bc_bβ ≠ 0) (h4 : sin angle_ba_bβ ≠ 0)
  (h5 : sin angle_ca_cγ ≠ 0) (h6 : sin angle_cb_cγ ≠ 0) : 
  (sin angle_ab_aα / sin angle_ac_aα) * (sin angle_bc_bβ / sin angle_ba_bβ) * (sin angle_ca_cγ / sin angle_cb_cγ) = -1 ↔ 
  (∃ l : S, ∀ p : S, p ∈ {S,a,c} → ∃ l' : S, ∀ q : S, q ∈ {α, β, γ} → q ∈ l') := sorry

end part_a_part_b_part_c_l719_719029


namespace geometric_series_sum_l719_719306

theorem geometric_series_sum (a r : ℝ) 
  (h1 : a * (1 - r / (1 - r)) = 18) 
  (h2 : a * (r / (1 - r)) = 8) : r = 4 / 5 :=
by sorry

end geometric_series_sum_l719_719306


namespace gwen_spent_money_l719_719133

theorem gwen_spent_money (initial : ℕ) (remaining : ℕ) (spent : ℕ) 
  (h_initial : initial = 7) 
  (h_remaining : remaining = 5) 
  (h_spent : spent = initial - remaining) : 
  spent = 2 := 
sorry

end gwen_spent_money_l719_719133


namespace seating_arrangements_l719_719101

theorem seating_arrangements (n : ℕ) (hn : n = 8) : 
  ∃ (k : ℕ), k = 5760 :=
by
  sorry

end seating_arrangements_l719_719101


namespace fraction_difference_l719_719146

variable x y : ℝ
hypothesis hx : x = Real.sqrt 5 - 1
hypothesis hy : y = Real.sqrt 5 + 1

theorem fraction_difference : (1 / x - 1 / y = 1 / 2) :=
by 
  sorry

end fraction_difference_l719_719146


namespace black_circles_count_l719_719760

theorem black_circles_count (a1 d n : ℕ) (h1 : a1 = 2) (h2 : d = 1) (h3 : n = 16) :
  (n * (a1 + (n - 1) * d) / 2) + n ≤ 160 :=
by
  rw [h1, h2, h3]
  -- Here we will carry out the arithmetic to prove the statement
  sorry

end black_circles_count_l719_719760


namespace factor_quadratic_equation_l719_719064

theorem factor_quadratic_equation : ∀ (x : ℝ), x^2 - 2 * x - 2 = 0 → (x - 1)^2 = 3 :=
by
  intro x
  intro h
  -- sorry

end factor_quadratic_equation_l719_719064


namespace find_acute_angles_right_triangle_l719_719471

theorem find_acute_angles_right_triangle (α β : ℝ)
  (h₁ : α + β = π / 2)
  (h₂ : 0 < α ∧ α < π / 2)
  (h₃ : 0 < β ∧ β < π / 2)
  (h4 : Real.tan α + Real.tan β + Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan α ^ 3 + Real.tan β ^ 3 = 70) :
  (α = 75 * (π / 180) ∧ β = 15 * (π / 180)) 
  ∨ (α = 15 * (π / 180) ∧ β = 75 * (π / 180)) := 
sorry

end find_acute_angles_right_triangle_l719_719471


namespace Petersons_property_tax_1996_l719_719551

theorem Petersons_property_tax_1996 (tax1995 : ℝ) (increase_rate : ℝ) (surcharge : ℝ) (tax1996 : ℝ) : 
  tax1995 = 1800 → increase_rate = 0.06 → surcharge = 200 → tax1996 = tax1995 * (1 + increase_rate) + surcharge → tax1996 = 2108 :=
by
  intros h_tax1995 h_increase h_surcharge h_tax1996
  have h1 : tax1996 = 1800 * 1.06 + 200, from h_tax1996.symm.trans (by rw [h_tax1995, h_increase, h_surcharge])
  rw [← h1]
  norm_num
  rfl

end Petersons_property_tax_1996_l719_719551


namespace cubs_more_home_runs_l719_719075

-- Define the conditions for the Chicago Cubs
def cubs_home_runs_third_inning : Nat := 2
def cubs_home_runs_fifth_inning : Nat := 1
def cubs_home_runs_eighth_inning : Nat := 2

-- Define the conditions for the Cardinals
def cardinals_home_runs_second_inning : Nat := 1
def cardinals_home_runs_fifth_inning : Nat := 1

-- Total home runs scored by each team
def total_cubs_home_runs : Nat :=
  cubs_home_runs_third_inning + cubs_home_runs_fifth_inning + cubs_home_runs_eighth_inning

def total_cardinals_home_runs : Nat :=
  cardinals_home_runs_second_inning + cardinals_home_runs_fifth_inning

-- The statement to prove
theorem cubs_more_home_runs : total_cubs_home_runs - total_cardinals_home_runs = 3 := by
  sorry

end cubs_more_home_runs_l719_719075


namespace total_cost_proof_l719_719252

-- Define the conditions
def daily_rental_cost : ℕ := 30
def mileage_cost_per_mile : ℕ := 0.25
def discount_threshold_days : ℕ := 5
def discount_rate : ℕ := 0.10
def rental_days : ℕ := 6
def miles_driven : ℕ := 500

-- Define the statement to prove
theorem total_cost_proof : daily_rental_cost * rental_days - (if rental_days > discount_threshold_days then (daily_rental_cost * rental_days * discount_rate) else 0) + (mileage_cost_per_mile * miles_driven) = 287 := by
  sorry

end total_cost_proof_l719_719252


namespace win_sector_area_l719_719731

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 12) (h_p : p = 1 / 3) :
  ∃ A : ℝ, A = 48 * π :=
by {
  sorry
}

end win_sector_area_l719_719731


namespace correct_statements_l719_719507

-- Define the conditions
def point_M_on_line_l (M : ℝ × ℝ) : Prop :=
  M.1 + M.2 = 4

def circle_O (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 4

def tangent_line (M P : ℝ × ℝ) : Prop :=
  -- Placeholder definition for tangency, expand as necessary.
  sorry

def pq_parallel_to_l (PQ : ℝ → ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, PQ x y = x + y + k

def locus_length (locus : ℝ × ℝ → Prop) : ℝ :=
  -- Placeholder for length calculation, expand as necessary.
  sorry

-- Statements to prove
theorem correct_statements (M : ℝ × ℝ) (PQ : ℝ → ℝ → ℝ) (N : ℝ × ℝ → Prop) :
  point_M_on_line_l M →
  (∃ P Q : ℝ × ℝ, circle_O P ∧ circle_O Q ∧ tangent_line M P ∧ tangent_line M Q ∧ pq_parallel_to_l PQ) ∧
  locus_length N = sqrt(2) * π :=
sorry

end correct_statements_l719_719507


namespace triangle_area_minimized_sum_m_l719_719818

theorem triangle_area_minimized_sum_m:
  ∀ (m : ℤ), let points := [(2, 8), (12, 20), (8, m)] in
  (area_of_triangle points is minimized) → (m ∈ {14, 16}) →
  set.sum {14, 16} = 30 :=
by
  intros m points h_area h_m
  sorry

end triangle_area_minimized_sum_m_l719_719818


namespace find_b_in_triangle_l719_719572

theorem find_b_in_triangle (B C : ℝ) (c : ℝ) (hB : B = π / 4) (hC : C = π / 3) (hc : c = 1) : 
  ∃ b : ℝ, b = (c * Real.sin B) / (Real.sin C) :=
by 
  use (c * Real.sin B) / (Real.sin C)
  sorry

end find_b_in_triangle_l719_719572


namespace distance_product_P_to_AB_l719_719298

def alpha : Real := Real.pi / 6
def P : Point := (1, 1)

def line_parametric_eq (t : Real) : Point := 
  (1 + (√3 / 2) * t, 1 + (1 / 2) * t)

def circle_eq (x y : Real) : Prop := 
  x^2 + y^2 = 4

theorem distance_product_P_to_AB :
  (∀ t, ∃ A B : Point,
    A ∈ set_of (λ p : Point, p = line_parametric_eq t ∧ circle_eq p.1 p.2) ∧
    B ∈ set_of (λ p : Point, p = line_parametric_eq t ∧ circle_eq p.1 p.2) ∧
    P.distance A * P.distance B = 2) := sorry

end distance_product_P_to_AB_l719_719298


namespace exists_tangent_circle_l719_719485

structure Circle (α : Type*) :=
  (center : α)
  (radius : ℝ)

variables {α : Type*} [MetricSpace α]

def is_tangent_to (C1 C2 : Circle α) (A : α) : Prop :=
  dist C1.center A = C1.radius ∧ dist C2.center A = C2.radius ∧ dist C1.center C2.center = C1.radius + C2.radius

def is_tangent_to_line (C : Circle α) (l : set α) (B : α) : Prop :=
  B ∈ l ∧ dist C.center B = C.radius ∧ ∃ m : ℝ, ∀ x ∈ l, (C.center.1 - B.1) * m = (C.center.2 - B.2)

noncomputable def construct_tangent_circle
  (S : Circle α)
  (A : α)
  (l : set α)
  (O : α)
  (B : α) : Circle α :=
  sorry

theorem exists_tangent_circle
  (S : Circle α)
  (A : α)
  (l : set α)
  (O : α)
  (B : α) :
  ∃ S' : Circle α,
  is_tangent_to S S' A ∧ is_tangent_to_line S' l B :=
sorry

end exists_tangent_circle_l719_719485


namespace length_of_BC_is_10_l719_719066

-- Definitions for problem conditions and specifications
def parabola (x : ℝ) : ℝ := x^2

def A : (ℝ × ℝ) := (0, 0)

def B (b : ℝ) : (ℝ × ℝ) := (-b, parabola b)
def C (b : ℝ) : (ℝ × ℝ) := (b, parabola b)

def is_parallel_to_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.2 = p2.2

def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

-- Main theorem statement
theorem length_of_BC_is_10 (b : ℝ) (h_area : b^3 = 125) 
  (h_B : B b = (-b, parabola b))
  (h_C : C b = (b, parabola b))
  (h_parallel : is_parallel_to_x_axis (B b) (C b))
  (h_symmetric : symmetric_about_y_axis (B b) (C b))
  (h_origin : A = (0, 0)) :
  2 * b = 10 :=
by
  sorry

end length_of_BC_is_10_l719_719066


namespace infinite_series_sum_eq_two_l719_719792

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l719_719792


namespace probability_of_fourth_roll_l719_719084

-- Define the conditions 
structure Die :=
(fair : Bool) 
(biased_six : Bool)
(biased_one : Bool)

-- Define the probability function
def roll_prob (d : Die) (f : Bool) : ℚ :=
  if d.fair then 1/6
  else if d.biased_six then if f then 1/2 else 1/10
  else if d.biased_one then if f then 1/10 else 1/5
  else 0

def probability_of_fourth_six (p q : ℕ) (r1 r2 r3 : Bool) (d : Die) : ℚ :=
  (if r1 && r2 && r3 then roll_prob d true else 0) 

noncomputable def final_probability (d1 d2 d3 : Die) (prob_fair distorted_rolls : Bool) : ℚ :=
  let fair_prob := if distorted_rolls then roll_prob d1 true else roll_prob d1 false
  let biased_six_prob := if distorted_rolls then roll_prob d2 true else roll_prob d2 false
  let total_prob := fair_prob + biased_six_prob
  let fair := fair_prob / total_prob
  let biased_six := biased_six_prob / total_prob
  fair * roll_prob d1 true + biased_six * roll_prob d2 true

theorem probability_of_fourth_roll
  (d1 : Die) (d2 : Die) (d3 : Die)
  (h1 : d1.fair = true)
  (h2 : d2.biased_six = true)
  (h3 : d3.biased_one = true)
  (h4 : ∀ d, d1 = d ∨ d2 = d ∨ d3 = d)
  (r1 r2 r3 : Bool)
  : ∃ p q : ℕ, p + q = 11 ∧ final_probability d1 d2 d3 true = 5/6 := 
sorry

end probability_of_fourth_roll_l719_719084


namespace decreasing_interval_l719_719251

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x) - sqrt 3 * cos (ω * x)
noncomputable def g (x : ℝ) : ℝ := 2 * sin (4 * x + π / 3)

theorem decreasing_interval : 
  ∃ ω > 0, 
    (∀ x₁ x₂, f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ 0 < x₂ - x₁ = π / 4) → 
    (∀ x, (π / 24) < x ∧ x < (7 * π / 24) → g'(x) < 0) :=
sorry

end decreasing_interval_l719_719251


namespace sum_of_repeating_decimal_l719_719569

theorem sum_of_repeating_decimal (m n : ℕ) (h1 : m = 1) (h2 : ∑ i in range (n - m + 1), (nat_digits 2017 (i % 4)) = 2017) : n = 808 := by sorry

end sum_of_repeating_decimal_l719_719569


namespace area_equality_l719_719491

variables {A B C L N K M : Point}

def is_acute_triangle (A B C : Triangle) : Prop :=
  ∀ (α β γ : ℝ), α = ∠BAC ∧ β = ∠ABC ∧ γ = ∠BCA → α < π/2 ∧ β < π/2 ∧ γ < π/2

def angle_bisector_property (A B C L : Point) (triangle : Triangle) : Prop :=
  B.prop.linpart (line_through A L) ∧ C.prop.linpart (line_through A L) ∧ A.angle_bisector B C

def circumcircle_intersection_property (A N : Point) (circumcircle : Circle) : Prop :=
  N ∈ circumcircle ∧ circumcircle.is_orthogonal (angle_bisector_property A B C L)

def perpendicular_to_side (L K M : Point) (triangle : Triangle) : Prop :=
  point_perpendicular_to_side (Line_through AB) (Line_through AC) L K M

def points_conditions (A B C L N K M : Point) (triangle : Triangle) (circumcircle : Circle) : Prop :=
  is_acute_triangle (triangle) ∧ angle_bisector_property A B C L ∧ 
  circumcircle_intersection_property A N circumcircle ∧
  perpendicular_to_side L K M triangle

theorem area_equality 
  (A B C L N K M : Point) 
  (triangle : Triangle)
  (circumcircle : Circle) 
  (H : points_conditions A B C L N K M triangle circumcircle) : 
  area (triangle ABC) = area (Quadrilateral AKNM) :=
sorry

end area_equality_l719_719491


namespace anthony_money_left_l719_719401

theorem anthony_money_left :
  ∀ (initial_money juice_cost cupcake_cost : ℕ),
    initial_money = 75 ∧
    juice_cost = 27 ∧
    cupcake_cost = 40 →
    initial_money - (juice_cost + cupcake_cost) = 8 :=
by
  intros initial_money juice_cost cupcake_cost
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  norm_num

end anthony_money_left_l719_719401


namespace binomial_10_5_eq_252_l719_719087

theorem binomial_10_5_eq_252 : nat.choose 10 5 = 252 :=
by
  -- We will add "sorry" here as we are only required to state the theorem, not prove it.
  sorry

end binomial_10_5_eq_252_l719_719087


namespace max_disk_score_l719_719814
-- Import necessary library for mathematical proofs

-- Define our conditions and the theorem
theorem max_disk_score (n : ℕ) (C : fin n → Type) 
  (on_circumference : ∀ i : fin (n-1), ∃ p : ℝ, p ∈ C (i+1) ∧ p = C i)
  (center_circumference: ∃ p : ℝ, p ∈ C 0 ∧ p = C (n-1)) : 
  ∃ S : set (fin n × fin n), 
  (∀ i j, (i, j) ∈ S ↔ (C i).proper_contains (C j)) ∧ S.card ≤ ((n - 1) * (n - 2)) / 2 := 
sorry

end max_disk_score_l719_719814


namespace modulus_of_complex_number_l719_719105

noncomputable def complexModulusExample : ℂ := 10 - 26 * complex.I

theorem modulus_of_complex_number :
  complex.abs complexModulusExample = 2 * real.sqrt 194 := by
  sorry

end modulus_of_complex_number_l719_719105


namespace unique_solution_of_equation_l719_719840

theorem unique_solution_of_equation :
  ∃! (x : Fin 8 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + 
                                  (x 2 - x 3)^2 + (x 3 - x 4)^2 + 
                                  (x 4 - x 5)^2 + (x 5 - x 6)^2 + 
                                  (x 6 - x 7)^2 + (x 7)^2 = 1 / 9 :=
sorry

end unique_solution_of_equation_l719_719840


namespace infinite_series_sum_eq_two_l719_719793

theorem infinite_series_sum_eq_two : 
  ∑' k : ℕ, (if k = 0 then 0 else (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1))))) = 2 :=
by
  sorry

end infinite_series_sum_eq_two_l719_719793


namespace volume_ratio_l719_719083

-- Definition for the volume of a cylinder
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Conditions given in the problem
def charlie_diameter : ℝ := 8
def charlie_height : ℝ := 16
def diana_diameter : ℝ := 16
def diana_height : ℝ := 8

-- Calculations based on the conditions
def volume_charlie : ℝ := volume_cylinder (charlie_diameter / 2) charlie_height
def volume_diana : ℝ := volume_cylinder (diana_diameter / 2) diana_height

-- Statement to prove the ratio is 1/2
theorem volume_ratio : volume_charlie / volume_diana = 1 / 2 :=
by sorry

end volume_ratio_l719_719083


namespace smallest_side_length_l719_719466

-- Define the radius of the disks
def radius : ℝ := 1

-- Define the number of disks
def num_disks : ℕ := 5

-- Define the side length of the square
def side_length (a : ℝ) : Prop :=
∀ (disk_centers : Vector (ℝ × ℝ) num_disks), 
  (∀ i j, i ≠ j → dist (disk_centers.nth i) (disk_centers.nth j) ≥ 2 * radius) ∧ 
  (∀ i, abs (disk_centers.nth i).fst ≤ a / 2 ∧ abs (disk_centers.nth i).snd ≤ a / 2)

-- Define the smallest a
def smallest_a : ℝ := 2 + 2 * Real.sqrt 2

theorem smallest_side_length : side_length smallest_a := by
  sorry

end smallest_side_length_l719_719466


namespace range_a_minus_b_l719_719892

open Set

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ log x / log 2 ∧ log x / log 2 ≤ 2}

def B (a b : ℝ) : Set ℝ := Icc a b

theorem range_a_minus_b (a b : ℝ) (h1 : A ⊆ B a b) : 
  a - b ∈ Iic (-2) :=
by
  sorry

end range_a_minus_b_l719_719892


namespace seating_arrangements_l719_719065

def alice := "Alice"
def bob := "Bob"
def carla := "Carla"
def derek := "Derek"
def eric := "Eric"
def frank := "Frank"

def adjacent (x y : String) (arrangement : List String) : Prop :=
  ∃ i, arrangement.nth i = some x ∧ arrangement.nth (i+1) = some y

def valid_arrangement (arrangement : List String) : Prop :=
  ¬ (adjacent alice bob arrangement ∨ adjacent alice carla arrangement ∨ adjacent alice frank arrangement) ∧
  ¬ adjacent derek eric arrangement

theorem seating_arrangements :
  ∃ arrangements : List (List String), 
    arrangements.length = 240 ∧ 
    ∀ arrangement ∈ arrangements, valid_arrangement arrangement :=
sorry

end seating_arrangements_l719_719065


namespace sum_of_intersections_l719_719886

theorem sum_of_intersections :
  let f : ℝ → ℝ := fun x => (1/3)^x
  let g : ℝ → ℝ := fun x => log (1/3) x
  ∃ (x1 x2 x3 x4 : ℝ),
    f x1 = 5 - x1 ∧ f x2 = 5 - x2 ∧
    x3 = (1/3)^x2 ∧ x4 = (1/3)^x1 ∧
    x1 + x2 + x3 + x4 = 10 :=
by sorry

end sum_of_intersections_l719_719886


namespace dividend_calculation_l719_719684

theorem dividend_calculation (D : ℝ) (Q : ℝ) (R : ℝ) (Dividend : ℝ) (h1 : D = 47.5) (h2 : Q = 24.3) (h3 : R = 32.4)  :
  Dividend = D * Q + R := by
  rw [h1, h2, h3]
  sorry -- This skips the actual computation proof

end dividend_calculation_l719_719684


namespace general_formula_b_sum_formula_l719_719143

-- Defining the sequence {a_n} and its sum S_n with given conditions
def geometric_sequence (a : ℕ → ℝ) : Prop := ∀ n, ∃ r : ℝ, a (n + 1) = r * a n
def sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := ∀ n, S n = ∑ i in Finset.range n, a (i + 1)

-- Given conditions
def conditions (a S : ℕ → ℝ) : Prop :=
  geometric_sequence a ∧ sequence_sum S a ∧ S 3 = 21 ∧ S 6 = 189

-- Proving the general formula for {a_n}
theorem general_formula (a S : ℕ → ℝ) (h : conditions a S) :
  ∀ n, a n = 3 * 2 ^ (n - 1) :=
sorry

-- Defining the sequence {b_n} and its sum T_n
def b_n (a : ℕ → ℝ) (n : ℕ) := (-1)^n * a n
def b_sequence_sum (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop := ∀ n, T (n + 1) = ∑ i in Finset.range (n + 1), b (i + 1)

-- Proving the sum of the first n terms of {b_n}
theorem b_sum_formula (a S : ℕ → ℝ) (T : ℕ → ℝ) (h : conditions a S) :
  b_sequence_sum T (b_n a) → ∀ n, T n = -1 + (-2)^n :=
sorry

end general_formula_b_sum_formula_l719_719143


namespace find_kgs_of_apples_l719_719108

def cost_of_apples_per_kg : ℝ := 2
def num_packs_of_sugar : ℝ := 3
def cost_of_sugar_per_pack : ℝ := cost_of_apples_per_kg - 1
def weight_walnuts_kg : ℝ := 0.5
def cost_of_walnuts_per_kg : ℝ := 6
def cost_of_walnuts : ℝ := cost_of_walnuts_per_kg * weight_walnuts_kg
def total_cost : ℝ := 16

theorem find_kgs_of_apples (A : ℝ) :
  2 * A + (num_packs_of_sugar * cost_of_sugar_per_pack) + cost_of_walnuts = total_cost →
  A = 5 :=
by
  sorry

end find_kgs_of_apples_l719_719108


namespace tetrahedron_volume_l719_719560

noncomputable def volume_of_tetrahedron (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABC_ABD : ℝ) : ℝ :=
  (1/3) * area_ABC * area_ABD * (Real.sin angle_ABC_ABD) * (AB / (Real.sqrt 2))

theorem tetrahedron_volume :
  let AB := 5 -- edge AB length in cm
  let area_ABC := 18 -- area of face ABC in cm^2
  let area_ABD := 24 -- area of face ABD in cm^2
  let angle_ABC_ABD := Real.pi / 4 -- 45 degrees in radians
  volume_of_tetrahedron AB area_ABC area_ABD angle_ABC_ABD = 43.2 :=
by
  sorry

end tetrahedron_volume_l719_719560


namespace smallest_positive_period_decreasing_intervals_smallest_positive_m_for_10_zeros_l719_719517

noncomputable def f (x : ℝ) : ℝ := sin x * cos x + cos x^2 - 1/2

-- 1. Smallest positive period
theorem smallest_positive_period : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = π :=
by
  sorry

-- 2. Intervals where f(x) is monotonically decreasing
theorem decreasing_intervals : ∀ k ∈ ℤ, ∀ x ∈ [k*π + π/8, k*π + 5*π/8], f'(x) < 0 :=
by
  sorry

-- 3. Smallest positive m for 10 zeros in [0, m]
theorem smallest_positive_m_for_10_zeros : ∀ m > 0, (∀ n ∈ (0, m], f(n) = 0 → (count_zeros f [0, m] = 10) → m = 39*π/8) :=
by
  sorry

end smallest_positive_period_decreasing_intervals_smallest_positive_m_for_10_zeros_l719_719517


namespace interior_angles_sum_of_octagon_exterior_angles_sum_of_octagon_number_of_diagonals_in_octagon_l719_719323

-- Definitions for the problem conditions
def is_octagon (n : ℕ) := n = 8

-- Statements to be proven
theorem interior_angles_sum_of_octagon (n : ℕ) (h : is_octagon n) : (n-2) * 180 = 1080 :=
by {
    rw h,
    exact nat.mul_left_comm 6 180,
}

theorem exterior_angles_sum_of_octagon (n : ℕ) (h : is_octagon n) : 360 = 360 := by rfl

theorem number_of_diagonals_in_octagon (n : ℕ) (h : is_octagon n) : n * (n - 3) / 2 = 20 := 
by {
    rw h,
    exact nat.div_eq_of_eq_mul_right (dec_trivial : 2 ≠ 0) (by norm_num : 40 = 2 * 20)
}

end interior_angles_sum_of_octagon_exterior_angles_sum_of_octagon_number_of_diagonals_in_octagon_l719_719323


namespace intersection_M_N_l719_719364

theorem intersection_M_N :
  let M := { x : ℝ | abs x ≤ 2 }
  let N := {-1, 0, 2, 3}
  M ∩ N = {-1, 0, 2} :=
by
  sorry

end intersection_M_N_l719_719364


namespace median_calculation_l719_719343

-- Define the range from 1 to 2020
def num_range := (List.range' 1 2020)

-- Define the set of integers, their squares, and their cubes
def all_numbers := num_range ++ num_range.map (·^2) ++ num_range.map (·^3)

-- Median calculation function
noncomputable def median (lst : List ℕ) : ℕ :=
  let sorted_lst := lst.qsort (· <= ·)
  let n := lst.length
  if (n % 2 = 0) then 
    (sorted_lst.get! (n / 2 - 1) + sorted_lst.get! (n / 2)) / 2
  else
    sorted_lst.get! (n / 2)

-- The theorem statement
theorem median_calculation : median all_numbers = 2040201 :=
by
  -- Just a placeholder for your actual proof
  sorry

end median_calculation_l719_719343


namespace count_prime_sums_equals_five_l719_719660

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]  -- the first 13 primes
def skipped_primes : List ℕ := primes.filter_with_index (λ i p, (i + 1) % 3 ≠ 0)
def sum_series (n : ℕ) : ℕ := (List.scanl (+) 0 skipped_primes).drop(1).nth (n - 1).getD 0
def is_prime (n : ℕ) : Bool := n > 1 ∧ (List.range (n - 2)).all (λ x, (x + 2) = n ∨ (n % (x + 2)) ≠ 0)

theorem count_prime_sums_equals_five :
  (List.range 12).countp (λ i, is_prime (sum_series (i+1))) = 5 := by
  sorry

end count_prime_sums_equals_five_l719_719660


namespace Matilda_basket_problem_l719_719278

theorem Matilda_basket_problem
  (n : ℕ)
  (h1 : 30 % n = 0)
  (h2 : 45 % n = 0)
  (h3 : n ≥ 3) :
  n ≤ 15 := 
sorry

example :
  (∃ n : ℕ, 30 % n = 0 ∧ 45 % n = 0 ∧ n ≥ 3 ∧ n ≤ 15) :=
by {
  existsi (15 : ℕ),
  repeat {split},
  -- prove divisibility
  exact dvd_refl 30,
  exact dvd_refl 45,
  -- prove n ≥ 3
  exact le_refl 15,
  -- prove n ≤ 15
  exact le_refl 15,
}

end Matilda_basket_problem_l719_719278


namespace problem_solution_l719_719173

-- Definitions of the conditions as Lean statements:
def condition1 (t : ℝ) : Prop :=
  (1 + Real.sin t) * (1 - Real.cos t) = 1

def condition2 (t : ℝ) (a b c : ℕ) : Prop :=
  (1 - Real.sin t) * (1 + Real.cos t) = (a / b) - Real.sqrt c

def areRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

-- The proof problem statement:
theorem problem_solution (t : ℝ) (a b c : ℕ) (h1 : condition1 t) (h2 : condition2 t a b c) (h3 : areRelativelyPrime a b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) : a + b + c = 2 := 
sorry

end problem_solution_l719_719173


namespace number_of_true_propositions_is_one_l719_719192

theorem number_of_true_propositions_is_one :
  (∃ (α : ℝ), ∀ (x : ℝ), f x = x^α ∧ f (1/2) = 4 → α ≠ -1/2) ∧
  (∀ (p : Prop), (∃ x0 : ℝ, x0^2 - x0 - 1 > 0) → (¬ (∀ x : ℝ, x^2 - x - 1 < 0)) ∧ 
  (∀ (x : ℝ), x > 1 → x^3 > x^2) ∧
  (∀ (f : ℝ → ℝ), (f 0 = 0) ↔ (∀ x : ℝ, f (-x) = -f x) → false)) →
  true_propositions = 1 := 
sorry

end number_of_true_propositions_is_one_l719_719192


namespace number_of_proper_subsets_of_S_l719_719893

def S : Set ℕ := { x | -2 < (x: ℤ) - 1 ∧ (x: ℤ) - 1 < 4 ∧ x ≠ 1 }

theorem number_of_proper_subsets_of_S : ∃ (n : ℕ), n = 2^(Set.finite.to_finset ⟨S, sorry⟩).card - 1 ∧ n = 15 := sorry

end number_of_proper_subsets_of_S_l719_719893


namespace brown_gumdrops_count_l719_719379

def gumdrops_conditions (total : ℕ) (blue : ℕ) (brown : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ) : Prop :=
  total = blue + brown + red + yellow + green ∧
  blue = total * 25 / 100 ∧
  brown = total * 25 / 100 ∧
  red = total * 20 / 100 ∧
  yellow = total * 15 / 100 ∧
  green = 40 ∧
  green = total * 15 / 100

theorem brown_gumdrops_count: ∃ total blue brown red yellow green new_brown,
  gumdrops_conditions total blue brown red yellow green →
  new_brown = brown + blue / 3 →
  new_brown = 89 :=
by
  sorry

end brown_gumdrops_count_l719_719379


namespace magnitude_z_l719_719487

noncomputable def z (i : ℂ) := (i - 1) / i

theorem magnitude_z (i : ℂ) (hc : i ≠ 0 ∧ i ^ 2 = -1) : complex.abs (z i) = real.sqrt 2 := by
  sorry

end magnitude_z_l719_719487


namespace number_of_type1_equals_number_of_type2_l719_719620

theorem number_of_type1_equals_number_of_type2
  (n : ℕ)
  (S : finset (ℕ × ℕ))
  (coloring : (ℕ × ℕ) → Prop)
  (type1 : finset (ℕ × ℕ) → Prop)
  (type2 : finset (ℕ × ℕ) → Prop)
  (hS : ∀ ⦃h k⦄, (h, k) ∈ S ↔ h + k < n)
  (h_color : ∀ {h k h' k'}, coloring (h, k) → h' ≤ h → k' ≤ k → coloring (h', k'))
  (type1_def : ∀ T, type1 T ↔ T.card = n ∧ ∀ (a b ∈ T), a.fst ≠ b.fst)
  (type2_def : ∀ T, type2 T ↔ T.card = n ∧ ∀ (a b ∈ T), a.snd ≠ b.snd) :
  ∃ (type1_sets type2_sets : finset (finset (ℕ × ℕ))),
    (∀ T, T ∈ type1_sets ↔ type1 T ∧ ∀ x ∈ T, coloring x = false) ∧
    (∀ T, T ∈ type2_sets ↔ type2 T ∧ ∀ x ∈ T, coloring x = false) ∧
    type1_sets.card = type2_sets.card :=
by sorry

end number_of_type1_equals_number_of_type2_l719_719620


namespace Lisa_favorite_number_l719_719689

theorem Lisa_favorite_number (a b : ℕ) (h : 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b)^2 = (a + b)^3 → 10 * a + b = 27 := by
  intro h_eq
  sorry

end Lisa_favorite_number_l719_719689


namespace find_m_value_l719_719499

open Nat

theorem find_m_value {m : ℕ} (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 24 :=
  sorry

end find_m_value_l719_719499


namespace infinite_series_equals_two_l719_719804

noncomputable def sum_series : ℕ → ℝ := λ k, (8^k : ℝ) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_equals_two :
  (∑' k : ℕ, if k > 0 then sum_series k else 0) = 2 :=
by 
  sorry

end infinite_series_equals_two_l719_719804


namespace inequality_proof_l719_719030

theorem inequality_proof 
(x1 x2 y1 y2 z1 z2 : ℝ) 
(hx1 : x1 > 0) 
(hx2 : x2 > 0) 
(hineq1 : x1 * y1 - z1^2 > 0) 
(hineq2 : x2 * y2 - z2^2 > 0)
: 
  8 / ((x1 + x2)*(y1 + y2) - (z1 + z2)^2) <= 
  1 / (x1 * y1 - z1^2) + 
  1 / (x2 * y2 - z2^2) := 
sorry

end inequality_proof_l719_719030


namespace distance_cycled_l719_719394

variable (v t d : ℝ)

theorem distance_cycled (h1 : d = v * t)
                        (h2 : d = (v + 1) * (3 * t / 4))
                        (h3 : d = (v - 1) * (t + 3)) :
                        d = 36 :=
by
  sorry

end distance_cycled_l719_719394


namespace total_paint_in_gallons_l719_719400

-- Definitions based on conditions
def paint_ratio (blue green white red : ℕ) := (blue, green, white, red) = (1, 2, 5, 3)
def green_paint_liters := 6
def liter_to_gallon := 0.264172
def total_parts := 1 + 2 + 5 + 3
def one_part := green_paint_liters / 2
def blue_paint_liters := one_part * 1
def white_paint_liters := one_part * 5
def red_paint_liters := one_part * 3

-- Statement to prove
theorem total_paint_in_gallons :
  let total_liters := blue_paint_liters + green_paint_liters + white_paint_liters + red_paint_liters in
  total_liters * liter_to_gallon = 8.717676 :=
by
  sorry

end total_paint_in_gallons_l719_719400


namespace arithmetic_seq_of_equal_roots_l719_719498

theorem arithmetic_seq_of_equal_roots (a b c : ℝ) (h : b ≠ 0) 
    (h_eq_roots : ∃ x, b*x^2 - 4*b*x + 2*(a + c) = 0 ∧ (∀ y, b*y^2 - 4*b*y + 2*(a + c) = 0 → x = y)) : 
    b - a = c - b := 
by 
  -- placeholder for proof body
  sorry

end arithmetic_seq_of_equal_roots_l719_719498


namespace coefficient_x7_y2_l719_719823

theorem coefficient_x7_y2 (x y : ℝ) : 
  let term1 := (x - y) * ∑ i in finset.range (9), (nat.choose 8 i) * x^(8 - i) * y^i,
      coefficients := finset.range (9).filter (λ i, 8 - i = 7 ∧ i = 2) in
  ∑ i in coefficients, (nat.choose 8 i) * (if 8 - i = 7 then x else -y) = 20 :=
by sorry

end coefficient_x7_y2_l719_719823


namespace max_sum_of_visible_numbers_l719_719138

-- Definitions based on conditions
def cube_faces_values : List ℕ := [1, 3, 9, 27, 81, 243]

-- The maximum possible visible sum
def max_visible_sum_of_stacked_cubes : ℕ := 1446

-- The main theorem
theorem max_sum_of_visible_numbers :
  ∀ (cubes : List (List ℕ)),
    (∀ cube ∈ cubes, cube = cube_faces_values) →
    (length cubes = 4) →
    let visible_numbers := 
      let bottom_cube := cubes.head!.erase 1
      let top_cube := cubes.reverse.head!.erase 1
      let middle_cubes := cubes.tail!.init!
      bottom_cube ++ top_cube ++ (middle_cubes.map (λ c, c.erase 1.erase 3)).join
    in visible_numbers.sum = max_visible_sum_of_stacked_cubes := 
  sorry

end max_sum_of_visible_numbers_l719_719138


namespace min_goals_l719_719373

/-- Define constants and summation constraints -/
constant P : ℕ   -- Total points
constant a : ℕ   -- Number of victories
constant b : ℕ   -- Number of draws with goals
constant c : ℕ   -- Number of draws without goals
constant G : ℕ   -- Number of goals scored

axiom h1 : Π (i : ℕ), 1 ≤ i ∧ i ≤ 5 → (10 * i - P) > 0  
-- Each team ends with points 10* (position - P), where position in their rank is between 1 and 5.

axiom h2 : a + b + c = 10   
-- Total number of matches 
-- All matches must end either in victory, draw with goals or draw without goals.

axiom h3 : 5a + 4b + 2c = P   
-- Total points

axiom h4 : ∃ (g1 g2 : ℕ), G = a + 2*b ∧ g1 = 10 - a - b - c ∧ g2 = 5 ∣ 2*b + c   
-- Definition of total goals,G = a + 2*b.

theorem min_goals : G = 6 := sorry

end min_goals_l719_719373


namespace correct_result_A_plus_B_value_of_A_plus_2B_when_x_is_minus_2_l719_719013

namespace MathProof

variables (x : ℝ) (A B : ℝ → ℝ)
noncomputable def A_B : ℝ → ℝ := -7 * x^2 + 10 * x + 12
noncomputable def B_value : ℝ → ℝ := 4 * x^2 - 5 * x - 6

-- Part (1)
theorem correct_result_A_plus_B :
  (A - B = A_B ∧ B = B_value) → (A + B = x^2) :=
by
  sorry

-- Part (2)
theorem value_of_A_plus_2B_when_x_is_minus_2 :
  (A - B = A_B ∧ B = B_value) → (A + 2 * B = 24) :=
by
  sorry

end MathProof

end correct_result_A_plus_B_value_of_A_plus_2B_when_x_is_minus_2_l719_719013


namespace min_length_of_PQ_l719_719493

def circle (O : Type) := {p : (ℝ × ℝ) // p.1^2 + p.2^2 = 4}

def line_through (A : ℝ × ℝ) (P Q : ℝ × ℝ) :=
  ∃ m b : ℝ, P.1 = m * A.1 + b ∧ P.2 = m * A.2 + b ∧ Q.1 = m * A.1 + b ∧ Q.2 = m * A.2 + b

theorem min_length_of_PQ {A P Q : ℝ × ℝ}
  (hO : circle ℝ) 
  (hA : A = (1, 1))
  (hP : P ∈ hO)
  (hQ : Q ∈ hO)
  (hl_through : line_through A P Q) :
  ∃ PQ : ℝ, PQ = 2 * real.sqrt 2 :=
sorry

end min_length_of_PQ_l719_719493


namespace find_coordinates_of_Q_l719_719564

theorem find_coordinates_of_Q (x y : ℝ) (P : ℝ × ℝ) (hP : P = (1, 2))
    (perp : x + 2 * y = 0) (length : x^2 + y^2 = 5) :
    (x, y) = (-2, 1) :=
by
  -- Proof should go here
  sorry

end find_coordinates_of_Q_l719_719564


namespace product_pos_sum_ge_half_n_l719_719600

theorem product_pos_sum_ge_half_n (n : ℕ) (x : Fin n → ℝ) (h_n_pos : 0 < n)
  (h_prod : ∏ i, x i = 1) (h_pos : ∀ i, 0 < x i) :
  ∑ i in Finset.range n, (x i * Real.sqrt (∑ j in Finset.range (i + 1), (x j)^2)) ≥ (n + 1) / 2 * Real.sqrt n := 
sorry

end product_pos_sum_ge_half_n_l719_719600


namespace max_z_l719_719606

theorem max_z (x y : ℝ) (h1 : x + y ≤ 7) (h2 : x - 3y ≤ -1) (h3 : 3x - y ≥ 5) :
  ∃ (x y : ℝ), x + y ≤ 7 ∧ x - 3y ≤ -1 ∧ 3x - y ≥ 5 ∧ 2*x - y = 8 ∧
  ∀ (x y : ℝ), x + y ≤ 7 → x - 3y ≤ -1 → 3x - y ≥ 5 → 2*x - y ≤ 8 :=
by
  sorry

end max_z_l719_719606


namespace max_disk_score_l719_719815
-- Import necessary library for mathematical proofs

-- Define our conditions and the theorem
theorem max_disk_score (n : ℕ) (C : fin n → Type) 
  (on_circumference : ∀ i : fin (n-1), ∃ p : ℝ, p ∈ C (i+1) ∧ p = C i)
  (center_circumference: ∃ p : ℝ, p ∈ C 0 ∧ p = C (n-1)) : 
  ∃ S : set (fin n × fin n), 
  (∀ i j, (i, j) ∈ S ↔ (C i).proper_contains (C j)) ∧ S.card ≤ ((n - 1) * (n - 2)) / 2 := 
sorry

end max_disk_score_l719_719815


namespace abs_eq_sets_l719_719130

theorem abs_eq_sets (x : ℝ) : 
  (|x - 25| + |x - 15| = |2 * x - 40|) → (x ≤ 15 ∨ x ≥ 25) :=
by
  sorry

end abs_eq_sets_l719_719130


namespace find_n_modulo_l719_719460

theorem find_n_modulo : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 9) ∧ n ≡ -1774 [MOD 7] ∧ n = 2 :=
by
  sorry

end find_n_modulo_l719_719460


namespace largest_d_l719_719883

variable (a b c d : ℝ)

theorem largest_d (h : a + 1 = b - 2 ∧ b - 2 = c + 3 ∧ c + 3 = d - 4) : 
  d >= a ∧ d >= b ∧ d >= c :=
by
  sorry

end largest_d_l719_719883


namespace samantha_lost_pieces_l719_719069

theorem samantha_lost_pieces (total_pieces_on_board : ℕ) (arianna_lost : ℕ) (initial_pieces_per_player : ℕ) :
  total_pieces_on_board = 20 →
  arianna_lost = 3 →
  initial_pieces_per_player = 16 →
  (initial_pieces_per_player - (total_pieces_on_board - (initial_pieces_per_player - arianna_lost))) = 9 :=
by
  intros h1 h2 h3
  sorry

end samantha_lost_pieces_l719_719069


namespace area_of_figure_l719_719645

-- Define the conditions
def num_squares : ℕ := 6
def total_perimeter : ℕ := 84

-- Compute the area of the figure
theorem area_of_figure : 
  let side_length := total_perimeter / 14,
      area_of_one_square := side_length * side_length,
      total_area := num_squares * area_of_one_square
  in 
    total_area = 216 := by
  sorry

end area_of_figure_l719_719645


namespace find_UV_squared_l719_719942

-- Given Definitions
variables (A B C T U V : Type*)
variables [triangle A B C] [obtuse_scalene_triangle A B C]
def tangents_intersection_with_circuircle (T : Type*) (B C : Type*) (ω : Type*) : Prop :=
is_circumcircle ω A B C ∧ tangent_to_circumcircle_at B ω T ∧ tangent_to_circumcircle_at C ω T

def projections_of_T (U V : Type*) (T : Type*) (AB AC : Type*) : Prop :=
projected_from U T AB ∧ projected_from V T AC

variables (BT CT BC TU TV UV : ℝ)
variables [circumcircle_conditions : tangents_intersection_with_circuircle T B C ω]
variables [projections_conditions : projections_of_T U V T A]

-- Given Measurements
variable BT_eq_20 : BT = 20
variable CT_eq_20 : CT = 20
variable BC_eq_26 : BC = 26
variable TU_TV_UV_eq_1300 : TU^2 + TV^2 + UV^2 = 1300

-- Proof Goal
theorem find_UV_squared : UV^2 = 650 :=
sorry

end find_UV_squared_l719_719942


namespace area_triangle_CMD_half_area_trapezoid_ABCD_l719_719978

-- Define points A, B, C, D, and M
variables (A B C D M : Point) 

-- Define the fact that ABCD is a trapezoid with AB parallel to CD
axiom trapezoid_ABCD : trapezoid A B C D (parallel A B C D)

-- Define the fact that M is the midpoint of AB
axiom M_midpoint_AB : midpoint M A B

-- Define the function to calculate the area of triangle and trapezoid
noncomputable def area_triangle (P Q R : Point) : Real := sorry
noncomputable def area_trapezoid (P Q R S : Point) : Real := sorry

-- The theorem that the area of triangle CMD is half of the trapezoid ABCD
theorem area_triangle_CMD_half_area_trapezoid_ABCD :
  area_triangle M C D = (1/2) * area_trapezoid A B C D :=
sorry

end area_triangle_CMD_half_area_trapezoid_ABCD_l719_719978


namespace average_speed_correct_l719_719744

-- Distance definitions
def distance_total : ℝ := 1.5
def distance_incline : ℝ := 0.4
def distance_flat : ℝ := 0.6
def distance_obstacles : ℝ := 0.5

-- Time definitions (in minutes)
def time_incline : ℝ := 10
def speed_flat_kmh : ℝ := 5
def speed_flat_mpm : ℝ := speed_flat_kmh * 1000 / 60
def time_flat : ℝ := distance_flat * 1000 / speed_flat_mpm
def time_obstacles : ℝ := 15

-- Total time in hours
def total_time_minutes : ℝ := time_incline + time_flat + time_obstacles
def total_time_hours : ℝ := total_time_minutes / 60

-- Average speed calculation
def average_speed : ℝ := distance_total / total_time_hours

-- Theorem statement for the Lean proof problem
theorem average_speed_correct : average_speed ≈ 2.795 := by
  sorry

end average_speed_correct_l719_719744


namespace sock_pairing_l719_719554

def sockPicker : Prop :=
  let white_socks := 5
  let brown_socks := 5
  let blue_socks := 2
  let total_socks := 12
  let choose (n k : ℕ) := Nat.choose n k
  (choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 21) ∧
  (choose (white_socks + brown_socks) 2 = 45) ∧
  (45 = 45)

theorem sock_pairing :
  sockPicker :=
by sorry

end sock_pairing_l719_719554


namespace trip_early_movie_savings_l719_719330

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end trip_early_movie_savings_l719_719330


namespace maximize_annual_profit_l719_719722

theorem maximize_annual_profit : 
  ∃ n : ℕ, n ≠ 0 ∧ (∀ m : ℕ, m ≠ 0 → (110 * n - (n * n + n) - 90) / n ≥ (110 * m - (m * m + m) - 90) / m) ↔ n = 5 := 
by 
  -- Proof steps would go here
  sorry

end maximize_annual_profit_l719_719722


namespace interval_monotonically_increasing_l719_719115

-- Definitions
def t (x : ℝ) : ℝ := -x^2 - 2*x + 3
def domain (x : ℝ) : Prop := -3 < x ∧ x < 1

-- Theorem statement
theorem interval_monotonically_increasing : 
  ∀ x : ℝ, domain x → (t x) > 0 → ∃ I : set ℝ, I = set.Icc (-1) (1) ∧ ∀ y ∈ I, function.monotone_decr_on t I :=
sorry

end interval_monotonically_increasing_l719_719115


namespace tan_alpha_plus_pi_over_4_sin_cos_expression_l719_719480

variable (α : ℝ)
hypothesis (tanα_eq_two : tan α = 2)

theorem tan_alpha_plus_pi_over_4 :
  tan (α + π / 4) = -3 := by sorry

theorem sin_cos_expression :
  (sin (2 * α)) / ((sin α)^2 + (sin α) * (cos α) - (cos (2 * α)) - 1) = 1 := by sorry

end tan_alpha_plus_pi_over_4_sin_cos_expression_l719_719480


namespace positive_difference_of_squares_l719_719669

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 70) (h2 : a - b = 20) : a^2 - b^2 = 1400 :=
by
sorry

end positive_difference_of_squares_l719_719669


namespace prism_plane_cross_section_area_l719_719817

theorem prism_plane_cross_section_area :
  let side_length := 6
  let vertices := [
    (3 * Real.sqrt 2, 0, 0),
    (0, 3 * Real.sqrt 2, 0),
    (-3 * Real.sqrt 2, 0, 0),
    (0, -3 * Real.sqrt 2, 0)
  ]
  let plane_eq : ℝ × ℝ × ℝ → Prop :=
    λ (x, y, z), 5 * x - 3 * y + 2 * z = 20
  let cross_section_area := 
    let z_vals := [
      10 - (15 * Real.sqrt 2) / 2,
      10,
      10 + (15 * Real.sqrt 2) / 2,
      10
    ]
    let points := [
      (3 * Real.sqrt 2, 0, 10 - (15 * Real.sqrt 2) / 2),
      (0, 3 * Real.sqrt 2, 10),
      (-3 * Real.sqrt 2, 0, 10 + (15 * Real.sqrt 2) / 2),
      (0, -3 * Real.sqrt 2, 10)
    ]
    let vec1 := (-3 * Real.sqrt 2, 3 * Real.sqrt 2, 0)
    let vec2 := (-3 * Real.sqrt 2, -3 * Real.sqrt 2, 0)
    let cross_prod := (0, 0, 18)
    let area := 1 / 2 * Real.sqrt ((cross_prod.1)^2 + (cross_prod.2)^2 + (cross_prod.3)^2)
    area
  in cross_section_area = 9 :=
by
  sorry

end prism_plane_cross_section_area_l719_719817


namespace floor_add_frac_eq_154_l719_719455

theorem floor_add_frac_eq_154 (r : ℝ) (h : ⌊r⌋ + r = 15.4) : r = 7.4 := 
sorry

end floor_add_frac_eq_154_l719_719455


namespace jaya_rank_from_bottom_l719_719939

theorem jaya_rank_from_bottom (n t : ℕ) (h_n : n = 53) (h_t : t = 5) : n - t + 1 = 50 := by
  sorry

end jaya_rank_from_bottom_l719_719939


namespace magnitude_of_complex_number_l719_719267

theorem magnitude_of_complex_number (z : ℂ) (h : z^2 + complex.abs z ^ 2 = 5 - (2 * I) ^ 2) : complex.abs z ^ 2 = 1 / 2 :=
sorry

end magnitude_of_complex_number_l719_719267


namespace find_PDF_l719_719642

noncomputable def CDF (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else x^2 / (1 + x^2)

theorem find_PDF :
  ∀ x : ℝ, deriv (CDF x) = if x ≤ 0 then 0 else 2*x / (1 + x^2)^2 :=
by
  intro x
  -- Prove using differentiation rules and cases
  sorry

end find_PDF_l719_719642


namespace sin_300_eq_neg_sqrt_3_div_2_l719_719779

theorem sin_300_eq_neg_sqrt_3_div_2 : sin (300 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l719_719779


namespace tangent_line_at_point_l719_719303

noncomputable def f (x : ℝ) : ℝ := sqrt (2 * x + 1)
def point : ℝ × ℝ := (0, 1)
def tangent_line_eq (x y : ℝ) : ℝ := x - y + 1

theorem tangent_line_at_point :
  ∀ x y : ℝ, (x, y) = point → tangent_line_eq x y = 0 :=
by
  sorry

end tangent_line_at_point_l719_719303


namespace series_sum_l719_719446

-- Define the infinite series
def series := ∑' k : ℕ, (k^2 : ℝ) / 3^k

-- State the theorem to prove
theorem series_sum : series = 1 := 
sorry

end series_sum_l719_719446


namespace area_percentage_change_is_neg_4_percent_l719_719359

noncomputable def percent_change_area (L W : ℝ) : ℝ :=
  let A_initial := L * W
  let A_new := (1.20 * L) * (0.80 * W)
  ((A_new - A_initial) / A_initial) * 100

theorem area_percentage_change_is_neg_4_percent (L W : ℝ) :
  percent_change_area L W = -4 :=
by
  sorry

end area_percentage_change_is_neg_4_percent_l719_719359


namespace correct_answer_B_l719_719695

def point_slope_form (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x - 2)

def proposition_2 (k : ℝ) (x y : ℝ) : Prop :=
  ∃ k : ℝ, @point_slope_form k x y

def proposition_3 (k : ℝ) : Prop := point_slope_form k 2 (-1)

def proposition_4 (k : ℝ) : Prop := k ≠ 0

theorem correct_answer_B : 
  (∃ k : ℝ, @point_slope_form k 2 (-1)) ∧ 
  (∀ k : ℝ, @point_slope_form k 2 (-1)) ∧
  (∀ k : ℝ, k ≠ 0) → true := 
by
  intro h
  sorry

end correct_answer_B_l719_719695


namespace num_pieces_cut_l719_719023

noncomputable def rod_length_meters := 25.5
noncomputable def piece_length_cm := 85
noncomputable def rod_length_cm := rod_length_meters * 100

theorem num_pieces_cut : rod_length_cm / piece_length_cm = 30 :=
by
  sorry

end num_pieces_cut_l719_719023


namespace min_value_of_f_l719_719899

open Real

theorem min_value_of_f (x : ℝ) (λ : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ π/2) 
  (h_min : ∀ x ∈ Icc 0 (π/2), 2 * (cos x - λ) ^ 2 - 1 - 2 * λ ^ 2 ≥ -3/2) :
  λ = 1/2 :=
sorry

end min_value_of_f_l719_719899


namespace pear_juice_percentage_l719_719280

theorem pear_juice_percentage
    (total_pears total_oranges : ℕ)
    (juice_per_4_pears juice_per_3_oranges : ℕ)
    (num_pears num_oranges : ℕ)
    (pear_juice_per_pear : ℕ := juice_per_4_pears / 4)
    (orange_juice_per_orange : ℕ := juice_per_3_oranges / 3)
    (total_pear_juice : ℕ := num_pears * pear_juice_per_pear)
    (total_orange_juice : ℕ := num_oranges * orange_juice_per_orange)
    (total_juice : ℕ := total_pear_juice + total_orange_juice)
    :
    15 = total_pears ->
    15= total_oranges ->
    10 = juice_per_4_pears ->
    12 = juice_per_3_oranges ->
    5 = num_pears ->
    4 = num_oranges ->
    total_juice > 0 -> -- ensures we are not dividing by zero
    (total_pear_juice * 100 / total_juice = 4386 / 100) :=
    by
    intros h1 h2 h3 h4 h5 h6 h7
    rw [h1, h2, h3, h4, h5, h6]
    sorry

end pear_juice_percentage_l719_719280


namespace max_XY_l719_719295

variable {Point : Type} [MetricSpace Point]

-- Trapezoid ABCD with AB parallel to CD and AB perpendicular to BC
variables {A B C D X Y : Point}
variable (AB : line_segment A B)
variable (CD : line_segment C D)
variable (AD : line_segment A D)
variable (BC : line_segment B C)
variable (AC : line_segment A C)
variable (BD : line_segment B D)

-- Conditions given in the problem
axiom h_trapezoid : AB.is_parallel CD
axiom h_perpendicular : AB ⊥ BC
axiom h_bisect_ext : bisects_angle AD (angle BXC) externally
axiom h_intersection_Y : intersects_at AC BD Y
axiom AB_length : length AB = 10
axiom CD_length : length CD = 15

theorem max_XY : length (line_segment X Y) = 6 :=
sorry

end max_XY_l719_719295


namespace sum_positive_integers_leq_15_point_5_l719_719841

theorem sum_positive_integers_leq_15_point_5 :
  (∑ n in Finset.filter (λ n, 1.5 * n + 3.5 ≤ 15.5) (Finset.range 9), n) = 36 :=
by
  have h1 : (1.5 : ℝ) * 8 + 3.5 = 15.5 := by norm_num
  -- This is just a useful normalization fact, not to be strictly included in the summary relation
  sorry

end sum_positive_integers_leq_15_point_5_l719_719841


namespace dogs_neither_l719_719403

theorem dogs_neither : 
  (total_dogs : ℕ) (dogs_with_tags : ℕ) (dogs_with_collars : ℕ) (dogs_with_both : ℕ)
  (h_total : total_dogs = 80)
  (h_tags : dogs_with_tags = 45)
  (h_collars : dogs_with_collars = 40)
  (h_both : dogs_with_both = 6) : 
  total_dogs - (dogs_with_tags + dogs_with_collars - dogs_with_both) = 1 := 
  by {
    rw [h_total, h_tags, h_collars, h_both],
    norm_num,
  }

end dogs_neither_l719_719403


namespace total_pages_in_book_l719_719982

noncomputable def pages_in_chapter (n : ℕ) : ℕ :=
  if n = 1 then 13 else 13 + 3 * (n - 1)

theorem total_pages_in_book : ∑ i in Finset.range 5, pages_in_chapter (i + 1) = 95 := 
by
  sorry

end total_pages_in_book_l719_719982


namespace quadratic_trinomial_bound_l719_719200

theorem quadratic_trinomial_bound (a b : ℤ) (f : ℝ → ℝ)
  (h_def : ∀ x : ℝ, f x = x^2 + a * x + b)
  (h_bound : ∀ x : ℝ, f x ≥ -9 / 10) :
  ∀ x : ℝ, f x ≥ -1 / 4 :=
sorry

end quadratic_trinomial_bound_l719_719200


namespace parallel_vectors_eq_l719_719523

theorem parallel_vectors_eq (m : ℤ) (h : (m, 4) = (3 * k, -2 * k)) : m = -6 :=
by
  sorry

end parallel_vectors_eq_l719_719523


namespace _l719_719980

noncomputable def radical_plane {R : Type*} [inner_product_space ℝ R]
  {q1 q2 : R} {r1 r2 : ℝ} (non_concentric : q1 ≠ q2) :
  set R :=
{ r | inner (r - q1) (r - q1) - r1^2 = inner (r - q2) (r - q2) - r2^2 }

lemma radical_plane_theorem {R : Type*} [inner_product_space ℝ R]
  {q1 q2 : R} {r1 r2 : ℝ} (non_concentric : q1 ≠ q2) :
  ∃ (a b c d : ℝ), 
  (∀ r, (r ∈ radical_plane non_concentric) ↔ (a * (r - q1 + q2).x + b * (r - q1 + q2).y + c * (r - q1 + q2).z = d)) ∧
  (a * (q2 - q1).x + b * (q2 - q1).y + c * (q2 - q1).z = 0) :=
by sorry

end _l719_719980


namespace cookies_in_second_type_l719_719415

theorem cookies_in_second_type (x : ℕ) (h1 : 50 * 12 + 80 * x + 70 * 16 = 3320) : x = 20 :=
by sorry

end cookies_in_second_type_l719_719415


namespace repeating_decimal_as_fraction_simplified_l719_719212

theorem repeating_decimal_as_fraction_simplified :
  ∃ (a b : ℕ), (0.35353535... = a / b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) :=
sorry

end repeating_decimal_as_fraction_simplified_l719_719212


namespace ratio_of_areas_l719_719615

variables (A B C E K L M D : Type*)
variable [plane_geometry A B C]
variable (AE EK KC : ℝ)
variable (h_ratios : AE / EK = 3 / 5 ∧ EK / KC = 5 / 4)
variable (h_median : median D A B C)
variable (h_points_L : B E L ∧ intersection D L B E)
variable (h_points_M : B K M ∧ intersection D M B K)

theorem ratio_of_areas (h_side_AC : is_side A C K E)
(h_ratios : AE / EK = 3 / 5 ∧ EK / KC = 5 / 4)
(h_median : median D A B C)
(h_points_L : B E L ∧ intersection D L B E)
(h_points_M : B K M ∧ intersection D M B K) :
ratio_of_areas L B M A B C = 1 / 5 :=
sorry

end ratio_of_areas_l719_719615


namespace product_of_odds_last_three_digits_l719_719462

theorem product_of_odds_last_three_digits :
  let P := ∏ i in (finset.filter (λ i, i % 2 = 1) (finset.range 2012)), i
  in P % 1000 = 875 := 
sorry

end product_of_odds_last_three_digits_l719_719462


namespace chord_length_line_equation_l719_719154

noncomputable theory

open Real

def line_passes_through_origin (l : ℝ → ℝ) : Prop :=
∃ m, ∀ x, l x = m * x

def circle (x y : ℝ) : Prop :=
x^2 + y^2 - 6 * y + 4 = 0

theorem chord_length (l : ℝ → ℝ) (m : ℝ) (h1 : line_passes_through_origin l) 
  (h2 : ∀ x, l x = m * x) (h3 : m = sqrt 2) : 
  ∀ (C : ℝ × ℝ), circle C.1 C.2 → (C = (0, 3)) → 
  ∃ d : ℝ, d = 2 * sqrt 2 :=
by
  sorry

theorem line_equation (l : ℝ → ℝ) (h1 : line_passes_through_origin l)
  (h2 : ∀ x, ∃ A B, circle A.1 A.2 ∧ circle B.1 B.2 ∧ A = midpoint B ⟨0, 0⟩) :
  ∃ f : ℝ → ℝ, (∀ x, f x = x) ∨ (∀ x, f x = -x) :=
by
  sorry

noncomputable def midpoint (A B O : ℝ × ℝ) :=
⟨(A.1 + B.1) / 2, (A.2 + B.2) / 2⟩

end chord_length_line_equation_l719_719154


namespace proj_w_v_eq_v_l719_719470

open Matrix

def proj (w v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  let dot_product (a b : Fin 2 → ℝ) : ℝ := (a 0 * b 0 + a 1 * b 1)
  let scalar := (dot_product v w) / (dot_product w w)
  fun i => scalar * w i

theorem proj_w_v_eq_v :
  let v := λ (i : Fin 2) => if i = 0 then 12 else -8
  let w := λ (i : Fin 2) => if i = 0 then -9 else 6
  proj w v = v :=
by {
  dsimp [proj],
  have h_dot_vw : (v 0 * w 0 + v 1 * w 1) = -180 := by norm_num,
  have h_dot_ww : (w 0 * w 0 + w 1 * w 1) = 117 := by norm_num,
  let scalar := h_dot_vw / h_dot_ww,
  have h_scalar : scalar = -4/3 := by norm_num,
  funext i,
  fin_cases i; norm_num,
}

end proj_w_v_eq_v_l719_719470


namespace infinite_series_converges_to_3_l719_719790

noncomputable def sum_of_series := ∑' k in (Finset.range ∞).filter (λ k, k > 0), 
  (8 ^ k / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))))

theorem infinite_series_converges_to_3 : sum_of_series = 3 := 
  sorry

end infinite_series_converges_to_3_l719_719790


namespace range_of_a_l719_719153

-- Define the function under consideration
def f (a x : ℝ) : ℝ := a * x - Math.log x - 1

-- Problem statement
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Ioc 0 Real.exp, f a x < 0) ↔ a < 1 := 
sorry

end range_of_a_l719_719153


namespace twelfth_root_of_unity_l719_719431

theorem twelfth_root_of_unity :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 11 ∧
  (let sqrt3i := complex.I * real.sqrt 3 in
   let expr := (1 + sqrt3i) / (1 - sqrt3i) in
   expr = complex.exp (2 * n * real.pi * complex.I / 12)) :=
begin
  use 1,
  split,
  norm_num,
  split,
  norm_num,
  sorry
end

end twelfth_root_of_unity_l719_719431


namespace required_rate_of_return_l719_719049

theorem required_rate_of_return (total initial_investment remainder remaining_amount: ℝ)
(rate1 rate2: ℝ) (investment1 investment2: ℝ) (desired_income: ℝ)
(income1 income2: ℝ):
  investment1 = 6000 →
  investment2 = 4500 →
  rate1 = 0.03 →
  rate2 = 0.045 →
  total initial_investment = 15000 →
  desired_income = 700 →
  income1 = investment1 * rate1 →
  income2 = investment2 * rate2 →
  remaining_amount = total initial_investment - (investment1 + investment2) →
  income1 + income2 + (remaining_amount * 0.0705) = desired_income :=
begin
  intros h_inv1 h_inv2 h_rate1 h_rate2 h_total h_desired_income h_income1 h_income2 h_remaining_amount,
  simp [h_inv1, h_inv2, h_rate1, h_rate2, h_total, h_desired_income] at *,
  linarith,
end

end required_rate_of_return_l719_719049


namespace percentage_increase_in_fall_l719_719062

theorem percentage_increase_in_fall :
  ∃ x : ℝ, (1 + x / 100) * (1 - 19 / 100) = 1 + 14.95 / 100 ∧ x ≈ 41.91 :=
by
  sorry

end percentage_increase_in_fall_l719_719062


namespace exists_l_l719_719712

theorem exists_l (n : ℕ) (h : n ≥ 4011^2) : ∃ l : ℤ, n < l^2 ∧ l^2 < (1 + 1/2005) * n := 
sorry

end exists_l_l719_719712


namespace david_total_course_hours_l719_719421

-- Definitions based on the conditions
def course_weeks : ℕ := 24
def three_hour_classes_per_week : ℕ := 2
def hours_per_three_hour_class : ℕ := 3
def four_hour_classes_per_week : ℕ := 1
def hours_per_four_hour_class : ℕ := 4
def homework_hours_per_week : ℕ := 4

-- Sum of weekly hours
def weekly_hours : ℕ := (three_hour_classes_per_week * hours_per_three_hour_class) +
                         (four_hour_classes_per_week * hours_per_four_hour_class) +
                         homework_hours_per_week

-- Total hours spent on the course
def total_hours : ℕ := weekly_hours * course_weeks

-- Prove that the total number of hours spent on the course is 336 hours
theorem david_total_course_hours : total_hours = 336 := by
  sorry

end david_total_course_hours_l719_719421


namespace trip_early_movie_savings_l719_719329

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end trip_early_movie_savings_l719_719329


namespace initial_bottles_of_water_l719_719574

theorem initial_bottles_of_water {B : ℕ} (h1 : 100 - (6 * B + 5) = 71) : B = 4 :=
by
  sorry

end initial_bottles_of_water_l719_719574


namespace f_neg_example_l719_719947

-- Definitions and conditions given in the problem
def f : ℚ → ℚ := sorry

axiom condition1 (a b : ℚ) (ha : a > 0) (hb : b > 0) : f (a * b) = f a + f b
axiom condition2 (p : ℕ) (hp : nat.prime p) : f (p) = p

-- This is the statement that corresponds to the problem's question and conclusion.
theorem f_neg_example : f (25 / 11) < 0 :=
sorry

end f_neg_example_l719_719947


namespace fraction_diff_l719_719144

open Real

theorem fraction_diff (x y : ℝ) (hx : x = sqrt 5 - 1) (hy : y = sqrt 5 + 1) :
  (1 / x - 1 / y) = 1 / 2 := sorry

end fraction_diff_l719_719144


namespace train_pass_platform_time_l719_719018

-- Define the conditions given in the problem.
def train_length : ℕ := 1200
def platform_length : ℕ := 1100
def time_to_cross_tree : ℕ := 120

-- Define the calculation for speed.
def speed := train_length / time_to_cross_tree

-- Define the combined length of train and platform.
def combined_length := train_length + platform_length

-- Define the expected time to pass the platform.
def expected_time_to_pass_platform := combined_length / speed

-- The theorem to prove.
theorem train_pass_platform_time :
  expected_time_to_pass_platform = 230 :=
by {
  -- Placeholder for the proof.
  sorry
}

end train_pass_platform_time_l719_719018


namespace count_not_divisible_by_5_or_7_l719_719903

theorem count_not_divisible_by_5_or_7 :
  let n := 1000
  let count_divisible_by (m : ℕ) := Nat.floor (999 / m)
  (999 - count_divisible_by 5 - count_divisible_by 7 + count_divisible_by 35) = 686 :=
by
  sorry

end count_not_divisible_by_5_or_7_l719_719903


namespace solve_cyclic_quadrilateral_area_l719_719488

noncomputable def cyclic_quadrilateral_area (AB BC AD CD : ℝ) (cyclic : Bool) : ℝ :=
  if cyclic ∧ AB = 2 ∧ BC = 6 ∧ AD = 4 ∧ CD = 4 then 8 * Real.sqrt 3 else 0

theorem solve_cyclic_quadrilateral_area :
  cyclic_quadrilateral_area 2 6 4 4 true = 8 * Real.sqrt 3 :=
by
  sorry

end solve_cyclic_quadrilateral_area_l719_719488


namespace exists_duplicate_parenthesizations_l719_719932

def expr : List Int := List.range' 1 (1991 + 1)

def num_parenthesizations : Nat := 2 ^ 995

def num_distinct_results : Nat := 3966067

theorem exists_duplicate_parenthesizations :
  num_parenthesizations > num_distinct_results :=
sorry

end exists_duplicate_parenthesizations_l719_719932


namespace min_value_of_x3y2z_l719_719592

noncomputable def min_value_of_polynomial (x y z : ℝ) : ℝ :=
  x^3 * y^2 * z

theorem min_value_of_x3y2z
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1 / x + 1 / y + 1 / z = 9) :
  min_value_of_polynomial x y z = 1 / 46656 :=
sorry

end min_value_of_x3y2z_l719_719592


namespace total_roses_given_to_friends_l719_719525

-- Conditions
def total_money : ℕ := 300
def price_per_rose : ℕ := 2
def fraction_jenna : ℚ := 1/3
def fraction_imma : ℚ := 1/2

-- Calculations based on the problem conditions
def total_roses : ℕ := total_money / price_per_rose
def roses_for_jenna : ℕ := (fraction_jenna * total_roses).toNat
def roses_for_imma : ℕ := (fraction_imma * total_roses).toNat

-- Theorem statement
theorem total_roses_given_to_friends : roses_for_jenna + roses_for_imma = 125 := by
  sorry

end total_roses_given_to_friends_l719_719525


namespace vector_norm_and_angle_l719_719505

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (h₁ : ‖a‖ = 4)
variables (h₂ : ‖b‖ = 2)
variables (h₃ : real.angle_between a b = real.pi / 3 * 2)

theorem vector_norm_and_angle :
  ‖a + b‖ = 2 * real.sqrt 3 ∧
  real.angle_between a (a + b) = real.pi / 6 :=
  sorry

end vector_norm_and_angle_l719_719505


namespace symmetric_function_value_l719_719547

noncomputable def f (x a : ℝ) := (|x - 2| + a) / (Real.sqrt (4 - x^2))

theorem symmetric_function_value :
  ∃ a : ℝ, (∀ x : ℝ, f x a = (|x - 2| + a) / (Real.sqrt (4 - x^2)) ∧ f x a = -f (-x) a) →
  f (a / 2) a = (Real.sqrt 3) / 3 :=
by
  sorry

end symmetric_function_value_l719_719547


namespace exist_constants_a_b_c_l719_719936

theorem exist_constants_a_b_c :
  ∃ (a b c : ℚ), (∀ n : ℕ, n > 0 →
  (∑ k in Finset.range n, k * (n^2 - k^2)) = n^2 * (a * n^2 - b) + c) :=
begin
  use [(1:ℚ) / 4, (1:ℚ) / 4, 0],
  intro n, 
  intro hn,
  sorry -- proof of the equation
end

end exist_constants_a_b_c_l719_719936


namespace find_valid_triples_l719_719112

-- Define the theorem to prove the conditions and results
theorem find_valid_triples :
  ∀ (a b c : ℕ), 
    (2^a + 2^b + 1) % (2^c - 1) = 0 ↔ (a = 0 ∧ b = 0 ∧ c = 2) ∨ 
                                      (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                      (a = 2 ∧ b = 1 ∧ c = 3) := 
sorry  -- Proof omitted

end find_valid_triples_l719_719112


namespace david_total_hours_on_course_l719_719423

def hours_per_week_class := 2 * 3 + 4 -- hours per week in class
def hours_per_week_homework := 4 -- hours per week in homework
def total_hours_per_week := hours_per_week_class + hours_per_week_homework -- total hours per week

theorem david_total_hours_on_course :
  let total_weeks := 24
  in total_weeks * total_hours_per_week = 336 := by
  sorry

end david_total_hours_on_course_l719_719423


namespace car_can_drive_around_ring_l719_719622

theorem car_can_drive_around_ring (n : ℕ) (gas : ℕ → ℕ) 
  (h1 : ∀ i, gas i = gas 0)
  (h2 : ∑ i in Finset.range n, gas i ≥ 1) :
  ∃ i, (∀ k < n, gas (k + i) mod n ≥ 0) →
        (∀ j, j < n → ∃ m, m ≤ j ∧ gas (i + m) mod n ≥ 0) :=
sorry

end car_can_drive_around_ring_l719_719622


namespace sparse_subsets_cardinality_l719_719783

theorem sparse_subsets_cardinality:
  let d : ℕ → ℕ := λ n,
  if n = 1 then 2
  else if n = 2 then 3
  else if n = 3 then 4
  else if n = 4 then 5
  else d (n - 1) + d (n - 4)
  in d 15 = 181 :=
by { sorry }

end sparse_subsets_cardinality_l719_719783


namespace exactly_one_correct_proposition_l719_719865

-- Define the propositions
def proposition1 (A B C : ℝ) : Prop := sin (2 * A) = sin (2 * B) → ((A = B ∧ C ≠ π / 2) ∨ (C = π / 2 ∧ A ≠ B))
def proposition2 (A B : ℝ) : Prop := sin B = cos A → ∃ C, A + B + C = π ∧ A ≠ π / 2
def proposition3 (A B C : ℝ) : Prop := sin A ^ 2 + sin B ^ 2 > sin C ^ 2 → C < π / 2
def proposition4 (a b c A B C : ℝ) : Prop := (a / cos (A / 2) = b / cos (B / 2) ∧ b / cos (B / 2) = c / cos (C / 2)) → (A = B ∧ B = C)

-- Prove that there is exactly one correct proposition
theorem exactly_one_correct_proposition (A B C a b c : ℝ) :
  (proposition4 a b c A B C) ∧
  ¬ (proposition1 A B C) ∧
  ¬ (proposition2 A B) ∧
  ¬ (proposition3 A B C) :=
by sorry

end exactly_one_correct_proposition_l719_719865


namespace rhombus_area_l719_719025

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 15) : 
  (d1 * d2) / 2 = 75 := 
by {
    -- Assuming h_d1 : d1 = 10 and h_d2 : d2 = 15
    rw [h_d1, h_d2],
    -- Calculating the area of the rhombus
    norm_num,
    exact (75 : ℝ)
}

end rhombus_area_l719_719025
