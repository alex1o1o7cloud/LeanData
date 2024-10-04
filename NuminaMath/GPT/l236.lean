import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.CombinatorialLogic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Cast
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Set.Icc
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.Convergence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Simp
import Mathlib.Tactic.WClickApply
import Mathlib.Topology.Constructions

namespace average_monthly_income_P_and_R_l236_236031

theorem average_monthly_income_P_and_R 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : P = 4000) :
  (P + R) / 2 = 5200 :=
sorry

end average_monthly_income_P_and_R_l236_236031


namespace smallest_multiple_14_15_16_l236_236215

theorem smallest_multiple_14_15_16 : 
  Nat.lcm (Nat.lcm 14 15) 16 = 1680 := by
  sorry

end smallest_multiple_14_15_16_l236_236215


namespace find_additional_fuel_per_person_l236_236177

def num_passengers : ℕ := 30
def num_crew : ℕ := 5
def num_people : ℕ := num_passengers + num_crew
def num_bags_per_person : ℕ := 2
def num_bags : ℕ := num_people * num_bags_per_person
def fuel_empty_plane : ℕ := 20
def fuel_per_bag : ℕ := 2
def total_trip_fuel : ℕ := 106000
def trip_distance : ℕ := 400
def fuel_per_mile : ℕ := total_trip_fuel / trip_distance

def additional_fuel_per_person (x : ℕ) : Prop :=
  fuel_empty_plane + num_people * x + num_bags * fuel_per_bag = fuel_per_mile

theorem find_additional_fuel_per_person : additional_fuel_per_person 3 :=
  sorry

end find_additional_fuel_per_person_l236_236177


namespace five_eight_sided_dice_not_all_same_l236_236892

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l236_236892


namespace find_x_when_y_64_l236_236836

theorem find_x_when_y_64 (x y k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_inv_prop : x^3 * y = k) (h_given : x = 2 ∧ y = 8 ∧ k = 64) :
  y = 64 → x = 1 :=
by
  sorry

end find_x_when_y_64_l236_236836


namespace coin_loading_impossible_l236_236965

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l236_236965


namespace count_divisors_1_to_9_of_64350_l236_236321

def is_divisor (n m : Nat) : Prop := m % n = 0

theorem count_divisors_1_to_9_of_64350 : 
  (Finset.filter (is_divisor 64350) (Finset.range 10)).card = 6 :=
by 
  sorry

end count_divisors_1_to_9_of_64350_l236_236321


namespace solve_for_a_l236_236267

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - real.sqrt 2

theorem solve_for_a (a : ℝ) (h : f a (f a (real.sqrt 2)) = -real.sqrt 2) : a = real.sqrt 2 / 2 :=
by
  sorry

end solve_for_a_l236_236267


namespace skylar_current_age_l236_236027

theorem skylar_current_age (started_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) (h1 : started_age = 17) (h2 : annual_donation = 8000) (h3 : total_donation = 440000) : 
  (started_age + total_donation / annual_donation = 72) :=
by
  sorry

end skylar_current_age_l236_236027


namespace find_a_b_solve_inequality_l236_236984

-- Definitions for the given conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 6 > 4
def sol_set1 (x : ℝ) (b : ℝ) : Prop := x < 1 ∨ x > b
def root_eq (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 2 = 0

-- The final Lean statements for the proofs
theorem find_a_b (a b : ℝ) : (∀ x, (inequality1 a x) ↔ (sol_set1 x b)) → a = 1 ∧ b = 2 :=
sorry

theorem solve_inequality (c : ℝ) : 
  (∀ x, (root_eq 1 x) ↔ (x = 1 ∨ x = 2)) → 
  (c > 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (2 < x ∧ x < c)) ∧
  (c < 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (c < x ∧ x < 2)) ∧
  (c = 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ false) :=
sorry

end find_a_b_solve_inequality_l236_236984


namespace four_digit_numbers_with_property_l236_236310

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l236_236310


namespace cube_edge_probability_l236_236072

theorem cube_edge_probability :
  let num_vertices := 8
  let edges_per_vertex := 3
  let total_vertex_pairs := (Nat.choose num_vertices 2)
  let total_edges := (num_vertices * edges_per_vertex) / 2
  (total_edges : ℚ) / (total_vertex_pairs : ℚ) = 3 / 7 :=
by
  sorry

end cube_edge_probability_l236_236072


namespace four_digit_numbers_with_property_l236_236308

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l236_236308


namespace inequality_solution_l236_236793

noncomputable def cubic_roots : list ℝ := 
  [/*root1*/sorry, /*root2*/sorry, /*root3*/sorry]

theorem inequality_solution :
  ∃ r1 r2 r3 : ℝ, r1 < r2 ∧ r2 < r3 ∧
  (∀ x : ℝ, ((-4 < x ∧ x < -3) ∨ (r1 < x ∧ x < r2)) → 
  ((x^2 + 1) / (x + 3) < (4*x^2 + 5) / (3*x + 4))) :=
by {
  let r1 := cubic_roots.nth 0,
  let r2 := cubic_roots.nth 1,
  let r3 := cubic_roots.nth 2,
  have h_r1_r2_r3: ∃ r1 r2 r3 : ℝ, r1 < r2 ∧ r2 < r3 := sorry,
  use [r1, r2, r3],
  exact h_r1_r2_r3,
  intro x,
  intro h,
  have h_ineq: (x^2 + 1) / (x + 3) < (4*x^2 + 5) / (3*x + 4) := sorry,
  exact h_ineq,
}

end inequality_solution_l236_236793


namespace gcd_91_72_l236_236203

/-- Prove that the greatest common divisor of 91 and 72 is 1. -/
theorem gcd_91_72 : Nat.gcd 91 72 = 1 :=
by
  sorry

end gcd_91_72_l236_236203


namespace red_balls_count_l236_236463

theorem red_balls_count (white_balls_ratio : ℕ) (red_balls_ratio : ℕ) (total_white_balls : ℕ)
  (h_ratio : white_balls_ratio = 3 ∧ red_balls_ratio = 2)
  (h_white_balls : total_white_balls = 9) :
  ∃ (total_red_balls : ℕ), total_red_balls = 6 :=
by
  sorry

end red_balls_count_l236_236463


namespace mod_3_pow_2040_eq_1_mod_5_l236_236092

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l236_236092


namespace rectangle_maximized_area_side_length_l236_236241

theorem rectangle_maximized_area_side_length
  (x y : ℝ)
  (h_perimeter : 2 * x + 2 * y = 40)
  (h_max_area : x * y = 100) :
  x = 10 :=
by
  sorry

end rectangle_maximized_area_side_length_l236_236241


namespace four_digit_numbers_with_property_l236_236301

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l236_236301


namespace second_grandmaster_wins_perfect_play_l236_236484

inductive Player 
| First : Player
| Second : Player

def rook_placement_game := ∀ (n : ℕ), n = 8 → 
  (∃ (first_player_wins : Player) (second_player_wins : Player), 
  (∃ (perfect_play : (Player → bool)), perfect_play Player.First = false ∧ 
  perfect_play Player.Second = true))

theorem second_grandmaster_wins_perfect_play (n : ℕ) (hn : n = 8) : rook_placement_game n :=
sorry

end second_grandmaster_wins_perfect_play_l236_236484


namespace log32_eq_four_fifth_l236_236597

theorem log32_eq_four_fifth :
  log 32 4 = 2 / 5 := by
  sorry

end log32_eq_four_fifth_l236_236597


namespace total_cookies_l236_236571

theorem total_cookies (chris kenny glenn : ℕ) 
  (h1 : chris = kenny / 2)
  (h2 : glenn = 4 * kenny)
  (h3 : glenn = 24) : 
  chris + kenny + glenn = 33 := 
by
  -- Focusing on defining the theorem statement correct without entering the proof steps.
  sorry

end total_cookies_l236_236571


namespace quadratic_sequence_figure_150_l236_236592

theorem quadratic_sequence_figure_150 :
  let f (n : ℕ) := 3 * n^2 + 3 * n + 1 in
    f 150 = 67951 :=
sorry

end quadratic_sequence_figure_150_l236_236592


namespace j_inv_h_of_neg3_eq_neg3_l236_236167

variables {α β : Type} [Function.Involutive h] [Function.Involutive j]

-- Given condition
axiom h_inv_j_eq_seven_x_minus_one : ∀ x, h⁻¹ (j x) = 7 * x - 1

theorem j_inv_h_of_neg3_eq_neg3 : j⁻¹ (h (-3)) = -3 :=
by { 
  -- Sorry to skip the proof
  sorry 
}

end j_inv_h_of_neg3_eq_neg3_l236_236167


namespace lcm_of_two_numbers_l236_236607

-- Define the numbers involved
def a : ℕ := 28
def b : ℕ := 72

-- Define the expected LCM result
def lcm_ab : ℕ := 504

-- State the problem as a theorem
theorem lcm_of_two_numbers : Nat.lcm a b = lcm_ab :=
by sorry

end lcm_of_two_numbers_l236_236607


namespace work_days_l236_236500

theorem work_days (W : ℝ) (x : ℝ) 
  (A_rate : ℝ := W / 2) 
  (B_rate : ℝ := W / 6)
  (B_alone_days : ℝ := 2.0000000000000004)
  (work_eq : x * (2 * W / 3) + (B_alone_days * W / 6) = W) :
  x = 1 :=
begin
  sorry
end

end work_days_l236_236500


namespace count_3_edge_trips_l236_236817

-- Define the vertices of the cube
inductive Vertex
| A : Vertex
| B : Vertex
| D : Vertex
| C : Vertex
| F : Vertex
| E : Vertex -- assuming E for completeness
| G : Vertex -- assuming G for completeness
| H : Vertex -- assuming H for completeness

open Vertex

-- Define the edges of the cube, represented as a set of pairs
def edge : Vertex → Vertex → Prop
| A D := true
| D A := true
| B C := true
| C B := true
| D C := true
| C D := true
| D F := true
| F D := true
| F B := true
| B F := true
| _ _ := false

-- Define trip conditions
def valid_trip (v1 v2 v3 v4 : Vertex) : Prop :=
  edge v1 v2 ∧ edge v2 v3 ∧ edge v3 v4

-- Define the theorem for counting the valid 3-edge trips from A to B starting with A -> D
theorem count_3_edge_trips : 
  ∃ (p : ℕ), p = 3 ∧
  (∀ (v2 v3 : Vertex), valid_trip A D v2 v3 → v3 = B → ∃! (path : Vertex → Vertex → Prop), path A D ∧ path D v2 ∧ path v2 v3) := 
sorry

end count_3_edge_trips_l236_236817


namespace find_dihedral_angles_l236_236200

-- Define plane angles
def plane_angle1 : ℝ := 90
def plane_angle2 : ℝ := 90
def plane_angle3 : ℝ := α

-- Define the theorem for the dihedral angles
theorem find_dihedral_angles (α : ℝ) : 
  dihedral_angle1 (plane_angle1) (plane_angle2) (plane_angle3) = plane_angle1 ∧
  dihedral_angle2 (plane_angle2) (plane_angle1) (plane_angle3) = plane_angle2 ∧
  dihedral_angle3 (plane_angle3) (plane_angle1) (plane_angle2) = α := 
sorry

end find_dihedral_angles_l236_236200


namespace number_of_ways_to_place_volunteers_l236_236719

theorem number_of_ways_to_place_volunteers (volunteers : Finset ℕ) (h₁ : volunteers.card = 20)
    (h₂ : ∀ v, v ∈ volunteers → 1 ≤ v ∧ v ≤ 20) :
    let selected := Finset.filter (λ v, v = 5 ∨ v = 14) volunteers in
    selected.card = 2 →
    ∃ group1 group2 : Finset ℕ,
        group1 ∪ group2 = selected ∧
        ∀ g, g ∈ group1 ∪ group2 → g = 5 ∨ g = 14 →
        group1.card = 2 ∧
        group2.card = 2 ∧
        (∀ x y, x ∈ group1 → y ∈ group1 → x ≤ y ∨ y ≤ x) ∧
        (∀ x y, x ∈ group2 → y ∈ group2 → x ≤ y ∨ y ≤ x) ∧
        group1.card * group2.card = 24 :=
begin
  sorry
end

end number_of_ways_to_place_volunteers_l236_236719


namespace Maria_Ivanovna_grades_l236_236403

theorem Maria_Ivanovna_grades :
  let a : ℕ → ℕ
  a 1 = 3 ∧ 
  a 2 = 8 ∧ 
  (∀ n, n ≥ 3 → a n = 2 * a (n - 1) + 2 * a (n - 2)) →
  a 6 = 448 :=
by
  intros
  sorry

end Maria_Ivanovna_grades_l236_236403


namespace sum_of_cubes_l236_236487

theorem sum_of_cubes (n : ℕ) : ∑ k in Finset.range (n + 1), k^3 = (n * (n + 1) / 2)^2 := 
  sorry

end sum_of_cubes_l236_236487


namespace minimum_value_of_expression_l236_236393

variable (x y z : ℝ)
variable h1 : -0.5 < x ∧ x < 0.5
variable h2 : -0.5 < y ∧ y < 0.5
variable h3 : -0.5 < z ∧ z < 0.5
variable h4 : x = y ∧ y = z

noncomputable def expression_value (x : ℝ) : ℝ := 
  (1 / (1 - x)^3) - (1 / (1 + x)^3)

theorem minimum_value_of_expression : 
  ∀ (x y z : ℝ), 
  (-0.5 < x ∧ x < 0.5) ∧ 
  (-0.5 < y ∧ y < 0.5) ∧ 
  (-0.5 < z ∧ z < 0.5) ∧ 
  (x = y ∧ y = z) → 
  (∃ (m : ℝ), m = -5.625 ∧ ∀ a : ℝ, 
  (a = expression_value x) → a ≥ m) := 
by
  sorry

end minimum_value_of_expression_l236_236393


namespace rationalize_denominator_l236_236426

theorem rationalize_denominator (a b : ℝ) (ha : a = 5) (hb : b = 6 * real.sqrt 5 - 2) :
  (a / b) = 5 * (3 * real.sqrt 5 + 1) / 88 :=
by 
  sorry

end rationalize_denominator_l236_236426


namespace books_sale_correct_l236_236778

variable (books_original books_left : ℕ)

def books_sold (books_original books_left : ℕ) : ℕ :=
  books_original - books_left

theorem books_sale_correct : books_sold 108 66 = 42 := by
  -- Since there is no need for the solution steps, we can assert the proof
  sorry

end books_sale_correct_l236_236778


namespace profit_percentage_l236_236144

theorem profit_percentage (SP CP : ℝ) (H_SP : SP = 1800) (H_CP : CP = 1500) :
  ((SP - CP) / CP) * 100 = 20 :=
by
  sorry

end profit_percentage_l236_236144


namespace find_n_l236_236133

theorem find_n (n : ℕ) (h : (2 * n + 1) / 3 = 2022) : n = 3033 :=
sorry

end find_n_l236_236133


namespace isosceles_triangle_interior_angles_l236_236253

theorem isosceles_triangle_interior_angles (a b c : ℝ) 
  (h1 : b = c) (h2 : a + b + c = 180) (exterior : a + 40 = 180 ∨ b + 40 = 140) :
  (a = 40 ∧ b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40 ∧ c = 40) :=
by
  sorry

end isosceles_triangle_interior_angles_l236_236253


namespace general_term_sum_S_n_l236_236637

variable {a b : ℕ → ℕ}

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b (n : ℕ) : ℕ := a n * 2^n

-- Condition: common difference of arithmetic sequence is greater than 0
def common_difference_positive (d : ℕ) : Prop := d > 0

-- Conditions: specific values of a2 and a3
def condition_a2_a3 : Prop := a 2 * a 3 = 15 ∧ a 1 + a 4 = 8

-- General term formula for the sequence {a_n}
theorem general_term : common_difference_positive 2 → condition_a2_a3 → (a = λ n, 2 * n - 1) := sorry

-- Sum S_n of the first n terms of sequence {b_n}
theorem sum_S_n (n : ℕ) : (∑ i in finset.range n, b i) = 6 + (2 * n - 3) * 2 ^ (n + 1) := sorry

end general_term_sum_S_n_l236_236637


namespace shortest_side_of_right_triangle_l236_236548

theorem shortest_side_of_right_triangle (a b : ℝ) (h_a : a = 5) (h_b : b = 12) :
  a = 5.00 :=
by
  have h_c : Real.sqrt (a^2 + b^2) = 13 := sorry
  have h_min : min a b = a := sorry
  exact h_a.trans rfl

end shortest_side_of_right_triangle_l236_236548


namespace solution_l236_236666

-- Define the conditions as Lean statements
def proposition1 (a : ℝ) : Prop :=
  (∃ x1 x2 : ℝ, x1 * x2 < 0 ∧ x1 + x2 = 3 - a ∧ x1 * x2 = a) → a < 0

def proposition2 : Prop :=
  ¬(∀ x : ℝ, x ∈ [-1,1] → (sqrt (x^2 - 1) + sqrt (1 - x^2)) = 0) →
  isEven (λ x, sqrt (x^2 - 1) + sqrt (1 - x^2)) ∧ ¬isOdd (λ x, sqrt (x^2 - 1) + sqrt (1 - x^2))

def proposition3 (f : ℝ → ℝ) : Prop :=
  (∀ y : ℝ, y ∈ [-2,2] → ∃ x : ℝ, f x = y) →
  (∀ y : ℝ, y ∈ [-3,1] → ∃ x : ℝ, f (x + 1) = y)

def proposition4 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x - 1) = f (1 - x)) →
  (∀ y : ℝ, y ∈ ℝ → f (-y) = f y)

def proposition5 (a : ℝ) : Prop :=
  ¬(∃ m : ℕ, m = 1 ∧ ∃ x : ℝ, x ∈ [-sqrt(3), sqrt(3)] ∧ (3 - x^2 = a))

-- Problem Statement
def correct_propositions (a : ℝ) (f : ℝ → ℝ) : Prop :=
  proposition1 a ∧ proposition5 a

theorem solution (a : ℝ) (f : ℝ → ℝ) :
  correct_propositions a f :=
by
  sorry -- proof to be provided

end solution_l236_236666


namespace probability_not_all_same_l236_236913

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l236_236913


namespace eccentricity_of_hyperbola_l236_236646

variable (a b c : ℝ) (e : ℝ)
variable (F1 F2 P : Point) (hyperbola : Set Point)
variable (is_focus : F1 ∈ hyperbola ∧ F2 ∈ hyperbola)
variable (is_point_on_hyperbola : P ∈ hyperbola)
variable (focus_cond : |PF1| + |PF2| = 6 * a)
variable (angle_cond : internal_angle P F1 F2 = π / 6)

theorem eccentricity_of_hyperbola :
  eccentricity = real.sqrt 3 := 
sorry

end eccentricity_of_hyperbola_l236_236646


namespace intercepted_arc_60_degrees_l236_236158

noncomputable def equilateral_triangle := sorry

def circle_radius_eq_height_of_triangle (r : ℝ) (h : ℝ) : Prop := 
  r = h

def angle_measure_arc (α : ℝ) : Prop :=
  α = 60

theorem intercepted_arc_60_degrees
  (α : ℝ)
  (r : ℝ)
  (h : ℝ)
  (eq_triangle : equilateral_triangle)
  (radius_height_eq : circle_radius_eq_height_of_triangle r h)
  (rolling_along_side : True) : 
  angle_measure_arc α := 
sorry

end intercepted_arc_60_degrees_l236_236158


namespace four_digit_numbers_with_property_l236_236307

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l236_236307


namespace count_valid_numbers_l236_236312
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l236_236312


namespace corresponding_angles_not_universally_true_l236_236825

theorem corresponding_angles_not_universally_true : 
  ∀ (l1 l2 : Line) (a b : Angle), ¬ (corresponding_angles l1 l2 a b → a = b) := by
  sorry

end corresponding_angles_not_universally_true_l236_236825


namespace polynomial_degree_l236_236183

def f (x : ℝ) : ℝ := 2 - 8*x + 5*x^2 - 7*x^3 + 8*x^4
def g (x : ℝ) : ℝ := 4 - 3*x - x^3 + 4*x^4

theorem polynomial_degree (c : ℝ) (h : c = -2) : (∀ x : ℝ, f(x) + c * g(x)) = (2 - 8*x + 5*x^2 - 5*x^3) := by
  sorry

end polynomial_degree_l236_236183


namespace area_ratio_of_regular_polygons_l236_236541

noncomputable def area_ratio (r : ℝ) : ℝ :=
  let A6 := (3 * Real.sqrt 3 / 2) * r^2
  let s8 := r * Real.sqrt (2 - Real.sqrt 2)
  let A8 := 2 * (1 + Real.sqrt 2) * (s8 ^ 2)
  A8 / A6

theorem area_ratio_of_regular_polygons (r : ℝ) :
  area_ratio r = 4 * (1 + Real.sqrt 2) * (2 - Real.sqrt 2) / (3 * Real.sqrt 3) :=
  sorry

end area_ratio_of_regular_polygons_l236_236541


namespace point_on_y_axis_l236_236358

theorem point_on_y_axis (a : ℝ) 
  (h : (a - 2) = 0) : a = 2 := 
  by 
    sorry

end point_on_y_axis_l236_236358


namespace monotonic_intervals_find_minimum_l236_236768

noncomputable def g (x : ℝ) : ℝ := Real.exp x

noncomputable def f (x : ℝ) : ℝ := (x^2 / Real.exp 1) * g x - (1 / 3) * x^3 - x^2

theorem monotonic_intervals :
  (∀ x, (x ∈ Set.Ioc (-2 : ℝ) 0 → f' x > 0) ∧ (x ∈ Set.Ioc (1 : ℝ) ∞ → f' x > 0)) ∧
  (∀ x, (x ∈ Set.Ioc (-∞ : ℝ) (-2) → f' x < 0) ∧ (x ∈ Set.Ioc (0 : ℝ) 1 → f' x < 0)) :=
by sorry

theorem find_minimum :
  ∀ (x ∈ Set.Icc (-1 : ℝ) (2 * Real.log 3)),
  f x ≥ f (-1) ∧ f (-1) = (1 / Real.exp 2) - (2 / 3) :=
by sorry

end monotonic_intervals_find_minimum_l236_236768


namespace excenter_of_triangle_BCE_opposite_E_lies_on_line_LM_l236_236756

/-- Let ABCD be a cyclic convex quadrilateral. 
    \Gamma is its circumcircle.
    E is the intersection of the diagonals AC and BD.
    L is the center of the circle tangent to sides AB, BC, CD.
    M is the midpoint of the arc BC of \Gamma not containing A and D.
    Prove that the excenter of triangle BCE opposite E lies on the line LM. -/
theorem excenter_of_triangle_BCE_opposite_E_lies_on_line_LM
  (A B C D E L M : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited L] [Inhabited M]
  (h1 : cyclic_concave_quad A B C D)
  (h2 : circumcircle_Gamma A B C D)
  (h3 : intersection E (diagonal A C) (diagonal B D))
  (h4 : circle_tangent_center L (side A B) (side B C) (side C D))
  (h5 : midpoint_arc_ensuring_no_containing_A_D M (arc_B C)) :
  ∃ (N : Type), excenter (triangle B C E) opposite_E N ∧ lies_on_line N L M :=
sorry

end excenter_of_triangle_BCE_opposite_E_lies_on_line_LM_l236_236756


namespace largest_x_value_satisfies_largest_x_value_l236_236871

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l236_236871


namespace smallest_positive_debt_l236_236483

theorem smallest_positive_debt :
  ∃ (D : ℤ), (∃ (c s : ℤ), D = 400 * c + 280 * s) ∧ D > 0 ∧ ∀ (D₀ : ℤ), (∃ (c₀ s₀ : ℤ), D₀ = 400 * c₀ + 280 * s₀) → D₀ > 0 → D ≤ D₀ :=
begin
  sorry
end

end smallest_positive_debt_l236_236483


namespace probability_not_all_dice_show_different_l236_236922

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l236_236922


namespace find_smallest_n_l236_236190

-- Definitions
def a := (1 : ℝ) / 3
def r := (1 : ℝ) / 3

-- Sum of first n terms formula
def S_n (n : ℕ) : ℝ := a * (1 - r ^ n) / (1 - r)

-- Target sum
def targetSum := (80 : ℝ) / 243

-- Proof statement
theorem find_smallest_n : ∃ n : ℕ, S_n n = targetSum ∧ ∀ m : ℕ, m < n → S_n m ≠ targetSum := by
  sorry

end find_smallest_n_l236_236190


namespace first_year_after_2000_with_digit_sum_15_l236_236083

theorem first_year_after_2000_with_digit_sum_15 : ∃ y, y > 2000 ∧ (y.digits.sum = 15) ∧ ∀ z, z > 2000 ∧ (z.digits.sum = 15) → y ≤ z := 
sorry

end first_year_after_2000_with_digit_sum_15_l236_236083


namespace solution_set_f_leq_1_l236_236650

theorem solution_set_f_leq_1 (f : ℝ → ℝ)
  (H1 : ∀ x, f x = f (-x))
  (H2 : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (H3 : f (-2) = 1) :
  {x | f x ≤ 1} = set.Icc (-2 : ℝ) 2 := 
sorry

end solution_set_f_leq_1_l236_236650


namespace sum_of_common_points_eq_one_l236_236114

noncomputable def sum_of_x_coordinates : ℕ :=
  let points := { (x, y) | 0 ≤ x ∧ x < 17 ∧ 0 ≤ y ∧ y < 17 ∧ (y ≡ 7 * x + 3 [MOD 17]) ∧ (y ≡ 13 * x + 14 [MOD 17]) }
  in points.foldl (λ acc (x, _), acc + x) 0

theorem sum_of_common_points_eq_one : sum_of_x_coordinates = 1 := sorry

end sum_of_common_points_eq_one_l236_236114


namespace binomial_coefficient_19_13_l236_236647

theorem binomial_coefficient_19_13 :
  (binom 20 13 = 77520) →
  (binom 20 14 = 38760) →
  (binom 18 12 = 18564) →
  (binom 19 13 = 27132) := by
  sorry

end binomial_coefficient_19_13_l236_236647


namespace green_peaches_eq_three_l236_236344

theorem green_peaches_eq_three (p r g : ℕ) (h1 : p = r + g) (h2 : r + 2 * g = p + 3) : g = 3 := 
by 
  sorry

end green_peaches_eq_three_l236_236344


namespace number_that_multiplies_x_l236_236336

variables (n x y : ℝ)

theorem number_that_multiplies_x :
  n * x = 3 * y → 
  x * y ≠ 0 → 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 →
  n = 5 :=
by
  intros h1 h2 h3
  sorry

end number_that_multiplies_x_l236_236336


namespace quadratic_inequality_solution_l236_236436

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -1/3 < x ∧ x < 1 → -3 * x^2 + 8 * x + 1 < 0 :=
by
  intro x
  intro h
  sorry

end quadratic_inequality_solution_l236_236436


namespace corresponding_angles_not_always_equal_l236_236826

theorem corresponding_angles_not_always_equal : ¬ (∀ (l1 l2 : Line) (a1 a2 : Angle),
  corresponding_angles l1 l2 a1 a2 → a1 = a2) :=
sorry

end corresponding_angles_not_always_equal_l236_236826


namespace centroid_sum_of_coordinates_l236_236097

theorem centroid_sum_of_coordinates :
  let A := (9, 2, -1)
  let B := (5, -2, 3)
  let C := (1, 6, 5)
  let G := ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3, (A.3 + B.3 + C.3) / 3)
  in G.1 + G.2 + G.3 = 9.33 := by
  sorry

end centroid_sum_of_coordinates_l236_236097


namespace max_numbers_named_in_ap_l236_236418

/-- 
Given ten polynomials of the fifth degree, if Vasya names consecutive 
natural numbers starting from some number, and Petya substitutes 
each named number into one of the polynomials such that the results 
form an arithmetic progression, then the maximum number of numbers 
that Vasya could name is 50.
-/
theorem max_numbers_named_in_ap (P : Fin 10 → Polynomial ℚ) :
  ∃ n : ℕ, ∀ k : ℕ, k < 50 → ∃ (i : Fin 10), ∃ (m : ℕ), n + k = m ∧ P i m = n + k :=
sorry

end max_numbers_named_in_ap_l236_236418


namespace sequence_pairs_count_370_l236_236675

theorem sequence_pairs_count_370 (a b : ℕ) (h1 : x 1 = a) (h2 : x 2 = b)
  (h_rec : ∀ n, x (n + 2) = 3 * x (n + 1) + 2 * x n)
  (h_exist : ∃ k ≥ 3, x k = 2019) :
  ∃ n, (a, b) = ordered_pair_num_370 n :=
sorry

end sequence_pairs_count_370_l236_236675


namespace domain_of_f_l236_236447

def f (x : ℝ) : ℝ := Real.sqrt (2 - x) + Real.log (x + 1)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, y = f x} = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l236_236447


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236853

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236853


namespace crates_probability_numerator_l236_236071

theorem crates_probability_numerator {a b c m n : ℕ} (h1 : 3 * a + 4 * b + 6 * c = 50) (h2 : a + b + c = 12)
(h3 : Nat.gcd m n = 1) (h4 : ∃ k, m = 30690 * k ∧ n = 531441 * k) : m = 10230 :=
by {
  -- Problem translation and conditions
  sorry
}

end crates_probability_numerator_l236_236071


namespace time_for_pipe_B_l236_236073

/-- Define the rate at which pipe A fills the cistern -/
def rate_A : ℝ := 1 / 60

/-- Define the rate at which the third pipe empties the cistern -/
def rate_C : ℝ := -1 / 100.00000000000001

/-- Assuming all three pipes together fill the cistern in 50 minutes, define the combined rate -/
def combined_rate : ℝ := 1 / 50

/-- Define the rate at which pipe B fills the cistern, inferred from the combined rate -/
noncomputable def rate_B : ℝ := combined_rate - rate_A - rate_C

/-- Define the time it takes for pipe B to fill the cistern -/
noncomputable def time_for_B : ℝ := 1 / rate_B

/-- Prove that pipe B can fill the cistern in 75 minutes -/
theorem time_for_pipe_B : time_for_B = 75 := by
  sorry

end time_for_pipe_B_l236_236073


namespace subtract_value_is_34_l236_236699

theorem subtract_value_is_34 
    (x y : ℤ) 
    (h1 : (x - 5) / 7 = 7) 
    (h2 : (x - y) / 10 = 2) : 
    y = 34 := 
sorry

end subtract_value_is_34_l236_236699


namespace probability_of_not_all_same_number_l236_236903

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l236_236903


namespace product_of_two_largest_prime_factors_of_360_l236_236492

theorem product_of_two_largest_prime_factors_of_360 : 
  (∃ (p1 p2 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ p1 ≠ p2 ∧ p1 * p2 = 15 ∧ p1 * p2 = 5 * 3) :=
begin
  use [3, 5],
  repeat {split},
  { apply nat.prime_of_nat 3 },
  { apply nat.prime_of_nat 5 },
  { exact dec_trivial },
  { exact dec_trivial },
  { exact dec_trivial },
  sorry
end

end product_of_two_largest_prime_factors_of_360_l236_236492


namespace probability_not_all_same_l236_236902

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l236_236902


namespace excluded_avg_mark_l236_236802

theorem excluded_avg_mark (N A A_remaining excluded_count : ℕ)
  (hN : N = 15)
  (hA : A = 80)
  (hA_remaining : A_remaining = 90) 
  (h_excluded : excluded_count = 5) :
  (A * N - A_remaining * (N - excluded_count)) / excluded_count = 60 := sorry

end excluded_avg_mark_l236_236802


namespace find_z_equidistant_l236_236698

def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem find_z_equidistant :
  ∃ z : ℝ, distance 1 2 z 1 1 2 = distance 1 2 z 2 1 1 ∧ z = 3 / 2 :=
by
  use (3 / 2)
  sorry

end find_z_equidistant_l236_236698


namespace credit_extended_by_automobile_finance_companies_l236_236561

def percentage_of_automobile_installment_credit : ℝ := 0.36
def total_consumer_installment_credit : ℝ := 416.66667
def fraction_extended_by_finance_companies : ℝ := 0.5

theorem credit_extended_by_automobile_finance_companies :
  fraction_extended_by_finance_companies * (percentage_of_automobile_installment_credit * total_consumer_installment_credit) = 75 :=
by
  sorry

end credit_extended_by_automobile_finance_companies_l236_236561


namespace A_completion_time_l236_236523

noncomputable def A_can_work_in_nine_days_alone : Prop :=
  ∃ (x : ℕ), (1 / x + 1 / 18 = 1 / 6) ∧ x = 9

theorem A_completion_time :
  A_can_work_in_nine_days_alone :=
by
  use 9 
  constructor
  · simp
    sorry  -- Place the correct simultaneous equations solution here.
  · exact rfl

end A_completion_time_l236_236523


namespace sqrt_3x_eq_5x_largest_value_l236_236878

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l236_236878


namespace line_equation_circle_equation_l236_236035

-- Define the conditions for the line problem
def y_intercept : ℝ := 2
def inclination_angle : ℝ := real.pi / 4  -- 45 degrees in radians

-- Define the conditions for the circle problem
def center : ℝ × ℝ := (-2, 3)
def is_tangent_to_y_axis := true

-- Line problem: Equation of a line with given y-intercept and inclination angle
theorem line_equation (y_intercept : ℝ) (inclination_angle : ℝ) (h_angle : inclination_angle = real.pi / 4) : 
  (∀ x : ℝ, x + y_intercept = real.tan inclination_angle * x + y_intercept) :=
begin 
  sorry
end

-- Circle problem: Equation of a circle with given center and tangent to the y-axis
theorem circle_equation (center : ℝ × ℝ) (is_tangent_to_y_axis : Prop) (h_center : center = (-2, 3)) (h_tangent : is_tangent_to_y_axis) : 
  (∀ x y : ℝ, (x + 2)^2 + (y - 3)^2 = 4) :=
begin
  sorry
end

end line_equation_circle_equation_l236_236035


namespace impossible_load_two_coins_l236_236958

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l236_236958


namespace find_A_l236_236152

theorem find_A (A B C : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) (h4 : A < 10) (h5 : B < 10) (h6 : C < 10) (h7 : 10 * A + B + 10 * B + C = 101 * B + 10 * C) : A = 9 :=
sorry

end find_A_l236_236152


namespace perimeter_is_correct_l236_236529

noncomputable def PA : ℝ := 30
noncomputable def PB : ℝ := 40
noncomputable def PC : ℝ := 35
noncomputable def PD : ℝ := 50
noncomputable def area_ABCD : ℝ := 2500

def distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

noncomputable def AB : ℝ := distance PA PB
noncomputable def BC : ℝ := distance PB PC
noncomputable def CD : ℝ := distance PC PD
noncomputable def DA : ℝ := distance PD PA

noncomputable def perimeter_ABCD : ℝ := AB + BC + CD + DA

theorem perimeter_is_correct : perimeter_ABCD ≈ 222.49 :=
by
  sorry

end perimeter_is_correct_l236_236529


namespace probability_not_all_same_l236_236916

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l236_236916


namespace inequality_abc_l236_236431

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by
  sorry

end inequality_abc_l236_236431


namespace area_of_lawn_l236_236540

theorem area_of_lawn 
  (park_length : ℝ) (park_width : ℝ) (road_width : ℝ) 
  (H1 : park_length = 60) (H2 : park_width = 40) (H3 : road_width = 3) : 
  (park_length * park_width - (park_length * road_width + park_width * road_width - road_width ^ 2)) = 2109 := 
by
  sorry

end area_of_lawn_l236_236540


namespace sqrt_3x_eq_5x_largest_value_l236_236877

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l236_236877


namespace rest_area_location_l236_236480

theorem rest_area_location :
  ∀ (A B : ℝ), A = 50 → B = 230 → (5 / 8 * (B - A) + A = 162.5) :=
by
  intros A B hA hB
  rw [hA, hB]
  -- doing the computation to show the rest area is at 162.5 km
  sorry

end rest_area_location_l236_236480


namespace sum_first_10_terms_sum_n_terms_l236_236154

-- Problem (1): Prove the sum for n = 10
theorem sum_first_10_terms : 
  (∑ i in Finset.range 10, (i + 1) * (i + 2)) = 440 :=
by sorry

-- Problem (2): Prove the general sum for any positive integer n
theorem sum_n_terms (n : ℕ) : 
  (∑ i in Finset.range n, (i + 1) * (i + 2)) = (1 / 3 : ℚ) * n * (n + 1) * (n + 2) :=
by sorry

end sum_first_10_terms_sum_n_terms_l236_236154


namespace Maria_Ivanovna_grades_l236_236402

theorem Maria_Ivanovna_grades :
  let a : ℕ → ℕ
  a 1 = 3 ∧ 
  a 2 = 8 ∧ 
  (∀ n, n ≥ 3 → a n = 2 * a (n - 1) + 2 * a (n - 2)) →
  a 6 = 448 :=
by
  intros
  sorry

end Maria_Ivanovna_grades_l236_236402


namespace coin_loading_impossible_l236_236968

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l236_236968


namespace hyperbola_eccentricity_correct_l236_236182

open Classical

-- Definition of the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2) / (a^2) - (y^2) / (b^2) = 1

-- Conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0)

-- Perimeter condition
def min_perimeter (P F A : Pt) : ℝ :=
  -- Assuming the distance formula function dist
  (dist P A) + (dist P F) + (dist A F)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ) (ha : a > 0) (hb : b > 0),
  (7 * b = 6 * a) → eccentricity a b = (real.sqrt 85) / 7 :=
by
  intros
  sorry

end hyperbola_eccentricity_correct_l236_236182


namespace M_D_B_l236_236847

variables (O O' M N A B D B' : Type)
variables (C C' : Set Type)
variables [circle C] [circle C']
variables [tangent_line AM AB BD]

-- Conditions
def circles_meet_at_two_points (C C' : Set Type) (O O' M N : Type) : Prop :=
  centers O O' ∧ intersection C C' = {M, N}

def common_tangent_closer (C C' : Set Type) (A B : Type) : Prop :=
  tangent_touch C A ∧ tangent_touch C' B ∧ closer_to_point M {A, B}

def line_perpendicular (AM : Set Type) (B : Type) : Prop :=
  line_through_point_perpendicular_to AM B

def line_through_point (p1 p2 : Type) (D : Type) : Prop :=
  line_through_point (OO' : Set Type) D ∧ point_perpendicular B AM

def diameter (B O' B' : Type) : Prop :=
  diameter_of_circle C' B O' B'

def collinear (p1 p2 p3 : Type) : Prop :=
  are_collinear p1 p2 p3

-- Theorem
theorem M_D_B'_are_collinear 
  (h1 : circles_meet_at_two_points C C' O O' M N)
  (h2 : common_tangent_closer C C' A B)
  (h3 : line_perpendicular AM B)
  (h4 : line_through_point OO' D)
  (h5 : diameter B O' B') :
  collinear M D B' :=
sorry

end M_D_B_l236_236847


namespace probability_not_all_same_l236_236925

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l236_236925


namespace kim_knitting_total_sweaters_l236_236744

/-
Proof Problem:

Given:
1. On Day 1, Kim knit 8 sweaters.
2. On Day 2, Kim knit 2 more sweaters than on Day 1.
3. On Day 3 and Day 4, Kim knit 4 fewer sweaters than on Day 2 each day.
4. On Day 5, Kim knit 4 fewer sweaters than on Day 2.
5. On Day 6, Kim knit half the number of sweaters she knit on Day 1.
6. On Day 7, Kim didn’t knit any sweaters.
7. On Day 8 to Day 10, Kim knit 25% fewer sweaters per day than the average number of sweaters she knit from Day 1 to Day 6.
8. On Day 11, Kim knit 1/3 the number of sweaters she knit on Day 10.
9. On Day 12 and Day 13, Kim knit 50% of the number of sweaters she knit on Day 10.
10. On Day 14, Kim knit only 1 sweater.

Prove:
The total number of sweaters Kim knit during the 14 days is 61.
-/

theorem kim_knitting_total_sweaters : 
  let knit : ℕ → ℕ := λ d, if d = 1 then 8
                          else if d = 2 then 10
                          else if d = 3 ∨ d = 4 then 6
                          else if d = 5 then 6
                          else if d = 6 then 4
                          else if d = 7 then 0
                          else if d = 8 ∨ d = 9 ∨ d = 10 then 5
                          else if d = 11 then 1
                          else if d = 12 ∨ d = 13 then 2
                          else if d = 14 then 1 else 0 in
  (List.range' 1 14).sum (λ d, knit d) = 61 :=
by
  intro knit
  have h₁ : knit 1 = 8 := rfl
  have h₂ : knit 2 = 10 := rfl
  have h₃ : knit 3 = 6 := rfl
  have h₄ : knit 4 = 6 := rfl
  have h₅ : knit 5 = 6 := rfl
  have h₆ : knit 6 = 4 := rfl
  have h₇ : knit 7 = 0 := rfl
  have h₈ : knit 8 = 5 := rfl
  have h₉ : knit 9 = 5 := rfl
  have h₁₀ : knit 10 = 5 := rfl
  have h₁₁ : knit 11 = 1 := rfl
  have h₁₂ : knit 12 = 2 := rfl
  have h₁₃ : knit 13 = 2 := rfl
  have h₁₄ : knit 14 = 1 := rfl
  have sum_eq : (List.range' 1 14).sum (λ d, knit d) = 61 := by sorry
  exact sum_eq

end kim_knitting_total_sweaters_l236_236744


namespace clubsuit_expr_l236_236188

def clubsuit (a b : ℕ) : ℚ := a^b + b^(a - b)

theorem clubsuit_expr :
  (clubsuit 2 (clubsuit 3 2)) \clubsuit 3 = 8589934592 :=
by
  -- Given the definition of the operation
  have h1 : clubsuit 3 2 = 11 := by sorry
  have h2 : clubsuit 2 11 = 2048 := by sorry
  have h3 : clubsuit 2048 3 = 2048^3 := by sorry
  exact sorry

end clubsuit_expr_l236_236188


namespace negation_of_p_l236_236004

def f (a x : ℝ) : ℝ := a * x - x - a

theorem negation_of_p :
  (¬ ∀ a > 0, a ≠ 1 → ∃ x : ℝ, f a x = 0) ↔ (∃ a > 0, a ≠ 1 ∧ ¬ ∃ x : ℝ, f a x = 0) :=
by {
  sorry
}

end negation_of_p_l236_236004


namespace number_of_three_digit_palindromes_l236_236682

/-- Define a three-digit integer palindrome. -/
def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = n % 10)

/-- Define the set of three-digit palindromes. -/
def three_digit_palindromes : Finset ℕ :=
  Finset.filter is_three_digit_palindrome (Finset.Icc 100 999)

/-- Prove the number of three-digit palindromes is 90. -/
theorem number_of_three_digit_palindromes : three_digit_palindromes.card = 90 := by
  sorry

end number_of_three_digit_palindromes_l236_236682


namespace artists_contemporary_probability_l236_236846

namespace ContemporaryArtists

/-- Defines the time span for an artist's life given their birth year. -/
def lifetime (birth_year : ℝ) : set ℝ := {t : ℝ | birth_year ≤ t ∧ t ≤ birth_year + 80}

noncomputable def probability_contemporaries : ℝ := 209 / 225

/-- The condition that two artists were contemporaries. -/
def contemporaries (x y : ℝ) : Prop :=
  ∃ t, t ∈ lifetime x ∧ t ∈ lifetime y

theorem artists_contemporary_probability :
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 300 ∧ 0 ≤ y ∧ y ≤ 300 →
  (probability_contemporaries = ∫∫ x in 0..300, y in 0..300, if contemporaries x y then 1 else 0) / (300^2) :=
sorry

end ContemporaryArtists

end artists_contemporary_probability_l236_236846


namespace count_possible_values_of_x_l236_236281

theorem count_possible_values_of_x :
  let n := (set.count {x : ℕ | 25 ≤ x ∧ x ≤ 33 ∧ ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c * x < 100 ∧ 3 ≤ b ≤ 100/x}) in
  n = 9 :=
by
  -- Here we must prove the statement by the provided conditions
  sorry

end count_possible_values_of_x_l236_236281


namespace scores_are_sample_l236_236067

-- Define the total number of students
def total_students : ℕ := 5000

-- Define the number of selected students for sampling
def selected_students : ℕ := 200

-- Define a predicate that checks if a selection is a sample
def is_sample (total selected : ℕ) : Prop :=
  selected < total

-- The proposition that needs to be proven
theorem scores_are_sample : is_sample total_students selected_students := 
by 
  -- Proof of the theorem is omitted.
  sorry

end scores_are_sample_l236_236067


namespace jacket_price_is_48_l236_236536

-- Definitions according to the conditions
def jacket_problem (P S D : ℝ) : Prop :=
  S = P + 0.40 * S ∧
  D = 0.80 * S ∧
  16 = D - P

-- Statement of the theorem
theorem jacket_price_is_48 :
  ∃ P S D, jacket_problem P S D ∧ P = 48 :=
by
  sorry

end jacket_price_is_48_l236_236536


namespace mode_of_time_spent_is_6_l236_236992

noncomputable def mode_time_spent : ℕ :=
let times := [4, 5, 6, 7] in
let counts := [12, 15, 20, 3] in
let max_count := counts.max' (by decide) in
times[counts.index_of max_count]

theorem mode_of_time_spent_is_6 :
  mode_time_spent = 6 :=
by
  sorry

end mode_of_time_spent_is_6_l236_236992


namespace flower_combinations_l236_236996

theorem flower_combinations : 
  (∃ num_combinations : ℕ, 
    num_combinations = 
      (card {r c l : ℕ // r ≥ 1 ∧ c ≥ 1 ∧ l ≥ 1 ∧ 4 * r + 3 * c + 5 * l = 120}) 
      ∧ num_combinations = 17) :=
sorry

end flower_combinations_l236_236996


namespace linear_regression_forecast_l236_236709

variable (x : ℝ) (y : ℝ)
variable (b : ℝ) (a : ℝ) (center_x : ℝ) (center_y : ℝ)

theorem linear_regression_forecast :
  b=-2 → center_x=4 → center_y=50 → (center_y = b * center_x + a) →
  (a = 58) → (x = 6) → y = b * x + a → y = 46 :=
by
  intros hb hcx hcy heq ha hx hy
  sorry

end linear_regression_forecast_l236_236709


namespace five_eight_sided_dice_not_all_same_l236_236889

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l236_236889


namespace remainder_of_power_modulo_l236_236086

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end remainder_of_power_modulo_l236_236086


namespace solve_inequality_l236_236794

theorem solve_inequality (a : ℝ) : 
  (if a = 0 ∨ a = 1 then { x : ℝ | false }
   else if a < 0 ∨ a > 1 then { x : ℝ | a < x ∧ x < a^2 }
   else if 0 < a ∧ a < 1 then { x : ℝ | a^2 < x ∧ x < a }
   else ∅) = 
  { x : ℝ | (x - a) / (x - a^2) < 0 } :=
by sorry

end solve_inequality_l236_236794


namespace mary_change_l236_236771

def cost_of_berries : ℝ := 7.19
def cost_of_peaches : ℝ := 6.83
def amount_paid : ℝ := 20.00

theorem mary_change : amount_paid - (cost_of_berries + cost_of_peaches) = 5.98 := by
  sorry

end mary_change_l236_236771


namespace kevin_final_cards_l236_236743

-- Define the initial conditions and problem
def initial_cards : ℕ := 20
def found_cards : ℕ := 47
def lost_cards_1 : ℕ := 7
def lost_cards_2 : ℕ := 12
def won_cards : ℕ := 15

-- Define the function to calculate the final count
def final_cards (initial found lost1 lost2 won : ℕ) : ℕ :=
  (initial + found - lost1 - lost2 + won)

-- Statement of the problem to be proven
theorem kevin_final_cards :
  final_cards initial_cards found_cards lost_cards_1 lost_cards_2 won_cards = 63 :=
by
  sorry

end kevin_final_cards_l236_236743


namespace probability_integer_exponent_l236_236627

noncomputable def expansion (x a : ℝ) : Fin 6 → ℝ :=
  λ k, (nat.CasesOn k (Power x 5) (λ n, (nat.CasesOn n (-(5 * a * Power x 4 / sqrt x))
    (λ m, (nat.CasesOn m (10 * (Power x 3) * (Power a 2) / (Power (sqrt x) 2))
      (λ p, (nat.CasesOn p (-(10 * (Power x 2) * (Power a 3) / (Power (sqrt x) 3)))
        (λ q, (nat.CasesOn q (5 * (Power x 1) * (Power a 4) / (Power (sqrt x) 4))
          (λ r, -(Power a 5 / (Power (sqrt x) 5))))))))))))).

theorem probability_integer_exponent
  (x a : ℝ) : 
  (1 / 2) = 
    let count_even_exponents := ∑ k in Finset.range 6, if (expansion x a k) % 2 = 0 then 1 else 0 in
    count_even_exponents.to_real / 6 := 
sorry

end probability_integer_exponent_l236_236627


namespace isosceles_triangle_area_of_triangle_l236_236731

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom triangle_sides (a b c : ℝ) (A B C : ℝ) : c = 2
axiom cosine_condition (a b c : ℝ) (A B C : ℝ) : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B

-- Questions
theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ)
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B) :
  a = b :=
sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B)
  (h3 : 7 * Real.cos B = 2 * Real.cos C) 
  (h4 : a = b) :
  ∃ S : ℝ, S = Real.sqrt 15 :=
sorry

end isosceles_triangle_area_of_triangle_l236_236731


namespace unique_positive_integers_l236_236065

theorem unique_positive_integers (x y : ℕ) (h1 : x^2 + 84 * x + 2008 = y^2) : x + y = 80 :=
  sorry

end unique_positive_integers_l236_236065


namespace reassemble_eye_to_center_l236_236014

-- Definitions used in the conditions
def Circle := {x : ℝ × ℝ // x.1^2 + x.2^2 = 1}
def Eye (c : Circle) := Point
def Dragon := Circle × Eye -- Dragon is a pair of a Circle and its Eye position
variable (d1 d2 : Dragon)

-- The condition on the first sheet
def dragon_eye_centered (d : Dragon) : Prop := d.2 = (0, 0)

-- Theorem to be proven
theorem reassemble_eye_to_center (d1 d2 : Dragon) 
  (h1 : dragon_eye_centered d1)
  (h2 : d2.2 ≠ (0, 0)) : 
  ∃ (p1 p2 : Circle), (p1 ∪ p2 = d2.1 ∧ p1 ∩ p2 = ∅) ∧
    reassemble p1 p2 = d1.1 := 
sorry

end reassemble_eye_to_center_l236_236014


namespace triangle_side_difference_l236_236363

theorem triangle_side_difference (x : ℕ) (h1 : x + 10 > 8) (h2 : x + 8 > 10) (h3 : 10 + 8 > x) :
  (∃ x_min x_max, (∀ y, x_min ≤ y ∧ y ≤ x_max → ∃ z, z = y ∧ z ∈ set.Icc 3 17) ∧ x_max - x_min = 14) :=
by sorry

end triangle_side_difference_l236_236363


namespace proposition_1_proposition_2_proposition_4_l236_236623

noncomputable def f : ℝ → ℝ
| x => if x ∈ Icc 0 2 then sin (π * x) else (1 / 2) * f (x - 2)

theorem proposition_1 : ∀ (x1 x2 : ℝ), x1 ∈ Icc 0 ⨿ infinite_interval_left 2 → 
x2 ∈ Icc 0 ⨿ infinite_interval_left 2 → 
abs (f x1 - f x2) ≤ 2 := 
sorry
-- Where "infinite_interval_left 2" is used for the interval (2, +∞).
-- You will have to define this interval separately if not provided in Mathlib.

theorem proposition_2 : ∀ (x : ℝ) (k : ℕ), k > 0 → 
x ∈ Icc 0 ⨿ infinite_interval_left 2 → 
f x = 2^k * f (x + 2*k) :=
sorry

theorem proposition_4 : ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 
∀ (x : ℝ), (x = x1 ∨ x = x2 ∨ x = x3) → 
( f (x) - log(x - 1) = 0 ) :=
sorry

def correct_propositions : Set ℕ := {1, 2, 4}

end proposition_1_proposition_2_proposition_4_l236_236623


namespace number_of_three_digit_palindromes_l236_236681

/-- Define a three-digit integer palindrome. -/
def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = n % 10)

/-- Define the set of three-digit palindromes. -/
def three_digit_palindromes : Finset ℕ :=
  Finset.filter is_three_digit_palindrome (Finset.Icc 100 999)

/-- Prove the number of three-digit palindromes is 90. -/
theorem number_of_three_digit_palindromes : three_digit_palindromes.card = 90 := by
  sorry

end number_of_three_digit_palindromes_l236_236681


namespace part1_part2_part3_l236_236271

noncomputable def f (x : ℝ) (a : ℝ) := log x + a * x
noncomputable def g (x : ℝ) (a : ℝ) := a * x^2 + 2 * x

theorem part1 (h : a = 1) : 
  let f := λ x : ℝ, log x + x in 
  let f' := derivative f in 
  let tf := tangent f 1 in
  tf = 2*x - y - 1 :=
sorry

theorem part2 (h : ∀ x, x > 0 → f x a ≤ -2) : 
  let f := λ x : ℝ, log x + a * x in 
  a = -exp 1 :=
sorry

theorem part3 (h1 : a < 0) (h2 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ exp 1 → f x a ≤ g x a) : 
  a ∈ Icc ((1 - 2 * exp 1) / (exp 1 ^ 2 - exp 1)) 0 :=
sorry

end part1_part2_part3_l236_236271


namespace flowers_per_pot_l236_236059

def total_gardens : ℕ := 10
def pots_per_garden : ℕ := 544
def total_flowers : ℕ := 174080

theorem flowers_per_pot  :
  (total_flowers / (total_gardens * pots_per_garden)) = 32 :=
by
  -- Here would be the place to provide the proof, but we use sorry for now
  sorry

end flowers_per_pot_l236_236059


namespace smallest_odd_prime_factor_2047_pow4_plus_1_l236_236608

-- We define the problem:
theorem smallest_odd_prime_factor_2047_pow4_plus_1 :
  ∃ p : ℕ, prime p ∧ p % 2 = 1 ∧ (2047^4 + 1) % p = 0 ∧
  ∀ q : ℕ, prime q ∧ q % 2 = 1 ∧ (2047^4 + 1) % q = 0 → q ≥ p := 
begin
  -- We want to prove that the smallest odd prime p such that
  -- 2047^4 + 1 is divisible by p is 41.
  use 41,
  split,
  { -- Prove 41 is prime
    exact prime_of_nat_prime (dec_trivial : nat.prime 41), 
  },
  split,
  { -- 41 is odd (41 mod 2 = 1)
    norm_num,
  },
  split,
  { -- 2047^4 + 1 is divisible by 41
    sorry,
  },
  { -- If q is a smaller odd prime such that 2047^4 + 1 is divisible by q,
    -- then q >= 41
    intros q hqprime hqodd hqdiv,
    sorry,
  }
end

end smallest_odd_prime_factor_2047_pow4_plus_1_l236_236608


namespace total_cookies_l236_236572

theorem total_cookies (chris kenny glenn : ℕ) 
  (h1 : chris = kenny / 2)
  (h2 : glenn = 4 * kenny)
  (h3 : glenn = 24) : 
  chris + kenny + glenn = 33 := 
by
  -- Focusing on defining the theorem statement correct without entering the proof steps.
  sorry

end total_cookies_l236_236572


namespace range_of_a_l236_236263

def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x - x^2 else Real.log (x + 1)

theorem range_of_a :
  (∀ x : ℝ, |f x| ≥ a * x) ↔ (a ∈ Set.Icc (-3 : ℝ) (0 : ℝ)) := 
by 
sorrry

end range_of_a_l236_236263


namespace letters_in_mailboxes_l236_236680

theorem letters_in_mailboxes : (number_of_ways : ℕ) (letters mailboxes : ℕ) (h_letters : letters = 5) (h_mailboxes : mailboxes = 3) :
  number_of_ways = mailboxes ^ letters :=
by
  sorry

end letters_in_mailboxes_l236_236680


namespace largest_prime_factor_of_460_l236_236852

theorem largest_prime_factor_of_460 : ∃ p, prime p ∧ p ∣ 460 ∧ (∀ q, prime q ∧ q ∣ 460 → q ≤ p) :=
begin
  sorry
end

end largest_prime_factor_of_460_l236_236852


namespace domain_of_g_l236_236201

noncomputable def g (x : ℝ) : ℝ := (x^3 + 11 * x - 2) / (|x - 3| + |x + 1|)

theorem domain_of_g : ∀ x : ℝ, |x - 3| + |x + 1| ≠ 0 :=
by
  intro x
  have h1 : |x - 3| ≥ 0 := abs_nonneg (x - 3)
  have h2 : |x + 1| ≥ 0 := abs_nonneg (x + 1)
  by_contra h3
  have h4 : |x - 3| = 0 ∧ |x + 1| = 0 := sorry
  have h5 : x = 3 ∧ x = -1 := sorry
  exact h4.elim (λ h, h5.1.symm ▸ h5.2.symm)

end domain_of_g_l236_236201


namespace algebraic_expression_value_l236_236694

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := by
  sorry

end algebraic_expression_value_l236_236694


namespace area_of_triangle_bounded_by_lines_l236_236199

theorem area_of_triangle_bounded_by_lines (y : ℝ) (x : ℝ) : 
  let line1_y_intercept := 1
  let line2_y_intercept := 4
  let base := line2_y_intercept - line1_y_intercept
  let intersection_x := (12 : ℝ) / 9
  let height := intersection_x
  let area := (1 / 2) * base * height
  in area = 2 :=
by
  sorry

end area_of_triangle_bounded_by_lines_l236_236199


namespace zika_virus_scientific_notation_l236_236106

def scientific_notation_form (a : ℝ) (n : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ a * 10^n = 0.0000021

theorem zika_virus_scientific_notation :
  ∃ (a n : ℝ), scientific_notation_form a n ∧ a = 2.1 ∧ n = -6 :=
by
  use 2.1
  use -6
  unfold scientific_notation_form
  norm_num
  split; norm_num
  sorry

end zika_virus_scientific_notation_l236_236106


namespace poly_division_remainder_eq_l236_236101

theorem poly_division_remainder_eq :
  ∃ (k a : ℤ), 
    let P := (λ x : ℝ, x^5 - 4 * x^4 + 12 * x^3 - 20 * x^2 + 15 * x - 4) in
    let D := (λ x : ℝ, x^2 - x + k) in
    (3*x + a = (P x) % (D x)) :=
begin
  sorry
end

end poly_division_remainder_eq_l236_236101


namespace num_four_digit_div_by_6_l236_236384

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_6 (n : ℕ) : Prop := is_divisible_by_2 n ∧ is_divisible_by_3 n

theorem num_four_digit_div_by_6 : (∑ n in finset.range 10000, if is_four_digit n ∧ is_divisible_by_6 n then 1 else 0) = 1500 := by
  sorry

end num_four_digit_div_by_6_l236_236384


namespace part1_part2_l236_236174

noncomputable def problem1 : ℝ :=
  8^(2/3) - (0.5)^(-3) + (1 / Real.sqrt 3)^(-2) * (81 / 16)^(-1/4)

noncomputable def problem2 : ℝ :=
  Real.log 5 * Real.log 8000 + (Real.log (2^(Real.sqrt 3)))^2 + Real.exp (Real.log 1) + Real.log (Real.exp 1 * Real.sqrt (Real.exp 1))

theorem part1 : problem1 = -2 := by
  sorry

theorem part2 : problem2 = 11/2 := by
  sorry

end part1_part2_l236_236174


namespace sum_of_n_for_continuity_of_f_l236_236003

def f (x n : ℝ) : ℝ :=
if x < n then x^2 - 3 * x + 1 else 3 * x + 4

theorem sum_of_n_for_continuity_of_f : ∑ n in {n : ℝ | continuous_at (λ x, f x n) n}, n = 6 := by
sorry

end sum_of_n_for_continuity_of_f_l236_236003


namespace range_x_satisfying_l236_236654

open Function

noncomputable def f : ℝ → ℝ := sorry -- Assume some appropriate function definition

theorem range_x_satisfying :
  Even f ∧ MonotoneDecreasingOn f (Icc (-3 : ℝ) (0 : ℝ)) ∧ ∀ x ∈ Icc (0 : ℝ) (√2 : ℝ), f (-x^2 + 2*x - 3) < f (x^2 + 1) :=
by
  sorry

end range_x_satisfying_l236_236654


namespace find_angle_A_find_area_l236_236701

noncomputable section

-- Definitions of conditions
variables {A B C : ℝ} {a b c : ℝ}
variables (A_pos : 0 < A) (A_lt_pi : A < π) (B_pos : 0 < B) (B_lt_pi : B < π)
variables (cosA_eq_half : cos A = 1 / 2) (a_eq_3 : a = 3) (b_eq_2c : b = 2 * c)
variables (cos_reln : (2 * b - c) * cos A = a * cos C)

-- Statement 1: Given the cosine relation, A = π / 3
theorem find_angle_A (h : (2 * b - c) * cos A = a * cos C) : A = π / 3 := sorry

-- Statement 2: Given a = 3 and b = 2c, find the area
theorem find_area (a_eq_3 : a = 3) (b_eq_2c : b = 2 * c) (cosA_eq_half : cos A = 1 / 2) :
  let s := sqrt (a * (a - b + c) * (a + b - c) * (a + b + c) / 4) in
  s = 3 * sqrt 3 / 2 := sorry

end find_angle_A_find_area_l236_236701


namespace collinear_vectors_l236_236679

variable (m : ℝ)
noncomputable def a : ℝ × ℝ := (2, 3)
noncomputable def b : ℝ × ℝ := (-1, 2)

theorem collinear_vectors : m•a + 4•b = (2,3) + 4•(-1,2) → m = -2 :=
by
  sorry

end collinear_vectors_l236_236679


namespace probability_not_all_same_l236_236912

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l236_236912


namespace positive_difference_of_perimeters_l236_236519

theorem positive_difference_of_perimeters (L W : ℕ) (n : ℕ) 
    (hLW : L = 10) (hW : W = 5) (hn : n = 5) : 
    (let minPerimeter := min (2 * (L / n) + 2 * W) (2 * (W / n) + 2 * L) in
    let maxPerimeter := max (2 * (L / n) + 2 * W) (2 * (W / n) + 2 * L) in
    maxPerimeter - minPerimeter = 8) :=
by 
    sorry

end positive_difference_of_perimeters_l236_236519


namespace N_eq_M_union_P_l236_236007

open Set

def M : Set ℝ := { x | ∃ n : ℤ, x = n }
def N : Set ℝ := { x | ∃ n : ℤ, x = n / 2 }
def P : Set ℝ := { x | ∃ n : ℤ, x = n + 1/2 }

theorem N_eq_M_union_P : N = M ∪ P := 
sorry

end N_eq_M_union_P_l236_236007


namespace range_of_c_solution_set_inequality_exclusive_cond_l236_236630

theorem range_of_c (c : ℝ) (hc : c > 0) :
  (∀ x : ℝ, y = c^x → StrictMono y = false) ↔ (0 < c ∧ c < 1) :=
begin
  sorry
end

theorem solution_set_inequality (c : ℝ) (hc : c > 0) :
  (∀ x : ℝ, x + |x - (2 * c)| > 1) ↔ (c ≥ 1) :=
begin
  sorry
end

theorem exclusive_cond (c : ℝ) (hc : c > 0) :
  (0 < c ∧ c ≤ 0.5) ∨ (1 ≤ c) :=
begin
  by_cases hp : (0 < c) ∧ (c < 1),
  { 
    -- Prove exactly one of p or q holds
    sorry
  },
  {
    -- Prove exactly one of p or q holds
    sorry
  }
end

end range_of_c_solution_set_inequality_exclusive_cond_l236_236630


namespace sin_double_angle_l236_236329

variable (α : ℝ)

-- Define the conditions
axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : cos (π / 4 - α) = 2 * sqrt 2 * cos (2 * α)

-- The goal to verify
theorem sin_double_angle (h1 : 0 < α ∧ α < π / 2) (h2 : cos (π / 4 - α) = 2 * sqrt 2 * cos (2 * α)) :
  sin (2 * α) = 15 / 16 := 
by
  sorry

end sin_double_angle_l236_236329


namespace coin_loading_impossible_l236_236970

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l236_236970


namespace probability_of_odd_sums_l236_236821

-- Definition of the problem conditions
def numbers := { x | ∃ n, x = (n : ℕ) ∧ n ∈ (finset.range 1 10)}
def grid := fin 3 × fin 3

namespace grid_numbers
open finset

-- Condition to check if the sum of the numbers in rows, columns, and the main diagonal is odd
def sum_of_rows_and_columns_and_diagonal_is_odd (f : grid → ℕ) : Prop :=
  (∀ i, (∑ j, f ⟨i, j⟩) % 2 = 1) ∧
  (∀ j, (∑ i, f ⟨i, j⟩) % 2 = 1) ∧
  ((∑ k, f ⟨k,k⟩) % 2 = 1)

-- Statement of the probability problem
theorem probability_of_odd_sums :
  (∃ (f : grid → ℕ), 
    (∀ i j, f ⟨i,j⟩ ∈ numbers) ∧ 
    (finset.univ.image f).card = 9 ∧ 
    sum_of_rows_and_columns_and_diagonal_is_odd f) →
  1 / 126 := 
sorry

end grid_numbers

end probability_of_odd_sums_l236_236821


namespace present_worth_of_bill_l236_236468

theorem present_worth_of_bill (P : ℝ) (TD BD : ℝ) 
  (hTD : TD = 36) (hBD : BD = 37.62) 
  (hFormula : BD = (TD * (P + TD)) / P) : P = 800 :=
by
  sorry

end present_worth_of_bill_l236_236468


namespace xiao_ming_needs_14_correct_answers_at_least_l236_236713

theorem xiao_ming_needs_14_correct_answers_at_least :
  ∀ (num_questions : ℕ)
    (points_correct : ℤ)
    (points_wrong : ℤ)
    (num_unanswered : ℕ)
    (min_score : ℤ),
    num_questions = 20 →
    points_correct = 5 →
    points_wrong = -2 →
    num_unanswered = 2 →
    min_score = 60 →
    let num_attempted := num_questions - num_unanswered in
    ∃ (x : ℕ), (5 * x - 2 * (num_attempted - x)) ≥ min_score ∧ x ≥ 14 :=
by
  intros num_questions points_correct points_wrong num_unanswered min_score
  dsimp
  intros h_num_questions h_points_correct h_points_wrong h_num_unanswered h_min_score
  use 14
  split
  · sorry
  · exact le_refl 14

end xiao_ming_needs_14_correct_answers_at_least_l236_236713


namespace exists_omega_cyclotomic_containing_sqrt_l236_236687

theorem exists_omega_cyclotomic_containing_sqrt (d : ℚ) :
  ∃ (ω : ℂ) (n : ℕ), ω^n = 1 ∧ (adjoin ℚ {ω}).to_subfield.contains (d.sqrt).to_subfield :=
sorry

end exists_omega_cyclotomic_containing_sqrt_l236_236687


namespace necessary_but_not_sufficient_l236_236632

variables {Point Line Plane : Type}
variables (P : Point) (l : Line) (α β : Plane)
variables (P_in_l : P ∈ l) (P_in_α : P ∈ α) (α_perp_β : α ⊥ β)

-- We need to prove l ⊆ α is necessary but not sufficient for l ⊥ β
theorem necessary_but_not_sufficient (P : Point) (l : Line) (α β : Plane) 
  (P_in_l : P ∈ l) (P_in_α : P ∈ α) (α_perp_β : α ⊥ β) :
  (l ⊆ α → l ⊥ β) = false ∧ (l ⊥ β → l ⊆ α) = true :=
sorry

end necessary_but_not_sufficient_l236_236632


namespace transformed_circle_equation_l236_236673

variable {x y x₀ y₀ : ℝ}

def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0], ![0, 1]]

def original_circle (x₀ y₀ : ℝ) : Prop :=
  x₀^2 + y₀^2 = 1

def transformed_point (x y x₀ y₀ : ℝ) : Prop :=
  (Matrix.vecMul transformation_matrix ![x₀, y₀]) = ![x, y]

def transformed_curve (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

theorem transformed_circle_equation (x₀ y₀ x y : ℝ) :
  original_circle x₀ y₀ → transformed_point x y x₀ y₀ → transformed_curve x y :=
by
  -- skip the proof as per instructions
  sorry

end transformed_circle_equation_l236_236673


namespace problem1_problem2_problem3_l236_236814

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 1 ≤ x then x^2 - 2 * a * x + a
  else if 0 < x then 2 * x + a / x
  else 0 -- Undefined for x ≤ 0

theorem problem1 (a : ℝ) :
  (∀ x y : ℝ, (0 < x ∧ x < y) → f a x < f a y) ↔ (a ≤ -1 / 2) :=
sorry
  
theorem problem2 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ f a x1 = 1 ∧ f a x2 = 1 ∧ f a x3 = 1) ↔ (0 < a ∧ a < 1 / 8) :=
sorry

theorem problem3 (a : ℝ) :
  (∀ x : ℝ, f a x ≥ x - 2 * a) ↔ (0 ≤ a ∧ a ≤ 1 + Real.sqrt 3 / 2) :=
sorry

end problem1_problem2_problem3_l236_236814


namespace impossible_load_two_coins_l236_236957

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l236_236957


namespace line_passes_through_vertex_of_parabola_l236_236618

theorem line_passes_through_vertex_of_parabola : 
  { b : ℝ | (∃ x : ℝ, x + b = 2 * b^2) }.finite.card = 2 :=
by
  sorry

end line_passes_through_vertex_of_parabola_l236_236618


namespace interest_rate_is_correct_l236_236545

def SI : ℝ := 70
def P : ℝ := 388.89
def T : ℝ := 4

def R : ℝ := (SI / (P * T)) * 100

theorem interest_rate_is_correct : R ≈ 4.497 := 
    sorry

end interest_rate_is_correct_l236_236545


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236854

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236854


namespace larger_number_of_two_l236_236485

noncomputable def quadratic_root : ℤ → ℤ → ℤ → ℚ :=
fun a b c =>
  let discriminant := b * b - 4 * a * c
  if discriminant < 0 then 0    -- handling negative for simplicity
  else (real.sqrt discriminant : ℚ)

theorem larger_number_of_two (x y : ℚ) (h1 : x - y = 5) (h2 : x * y = 156) : x = (5 + quadratic_root 1 (-5) (-156)) / 2 :=
by
  sorry

end larger_number_of_two_l236_236485


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236934

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236934


namespace sixty_five_percent_of_40_minus_four_fifths_of_25_l236_236686

theorem sixty_five_percent_of_40_minus_four_fifths_of_25 : 
  (0.65 * 40) - (0.8 * 25) = 6 := 
by
  sorry

end sixty_five_percent_of_40_minus_four_fifths_of_25_l236_236686


namespace wendy_baked_4_cupcakes_l236_236849

def pastries_before_sale (left sold : Nat) : Nat := left + sold

def cupcakes_baked (total_pasties cookies : Nat) : Nat := total_pasties - cookies

theorem wendy_baked_4_cupcakes (left sold cookies : Nat) :
    left = 24 → sold = 9 → cookies = 29 → pastries_before_sale left sold - cookies = 4 :=
by
  intros h_left h_sold h_cookies
  rw [h_left, h_sold, h_cookies]
  unfold pastries_before_sale
  simp
  -- Proof can be filled in here
  sorry

end wendy_baked_4_cupcakes_l236_236849


namespace unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l236_236041

theorem unique_solution_x_ln3_plus_x_ln4_eq_x_ln5 :
  ∃! x : ℝ, 0 < x ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) := sorry

end unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l236_236041


namespace kate_money_ratio_l236_236380

-- Define the cost of the pen and the amount Kate needs
def pen_cost : ℕ := 30
def additional_money_needed : ℕ := 20

-- Define the amount of money Kate has
def kate_savings : ℕ := pen_cost - additional_money_needed

-- Define the ratio of Kate's money to the cost of the pen
def ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- The target property: the ratio of Kate's savings to the cost of the pen
theorem kate_money_ratio : ratio kate_savings pen_cost = (1, 3) :=
by
  sorry

end kate_money_ratio_l236_236380


namespace length_broken_line_path_l236_236191

theorem length_broken_line_path (A B O C D P : Point) (r : ℝ) (diam_AB : dist A B = 12)
  (center_O: midpoint A B = O) (radius_O: dist O A = 6) 
  (dist_CA : dist C A = 3) (dist_DB : dist D B = 3)
  (on_circle_P : dist O P = 6) (right_angle_CPD : angle C P D = pi/2):
  dist C P + dist P D = 6 * sqrt 5 :=
by sorry

end length_broken_line_path_l236_236191


namespace locus_centers_annular_region_l236_236820

noncomputable def locus_of_centers (P : Point) (a b : ℝ) (h : a < b) : Set Point :=
  { O | ∃ (r : ℝ), a ≤ r ∧ r ≤ b ∧ dist O P = r }

theorem locus_centers_annular_region
  (P : Point) (a b : ℝ) (h : a < b) :
  locus_of_centers P a b h = { O | a ≤ dist O P ∧ dist O P ≤ b } :=
sorry

end locus_centers_annular_region_l236_236820


namespace michael_ratio_zero_l236_236988

theorem michael_ratio_zero (M : ℕ) (h1: M ≤ 60) (h2: 15 = (60 - M) / 2 - 15) : M = 0 := by
  sorry 

end michael_ratio_zero_l236_236988


namespace four_digit_numbers_property_l236_236319

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l236_236319


namespace needle_intersection_probability_l236_236822

noncomputable def needle_probability (a l : ℝ) (h : l < a) : ℝ :=
  (2 * l) / (a * Real.pi)

theorem needle_intersection_probability (a l : ℝ) (h : l < a) :
  needle_probability a l h = 2 * l / (a * Real.pi) :=
by
  -- This is the statement to be proved
  sorry

end needle_intersection_probability_l236_236822


namespace find_constants_l236_236678

open Set

variable {α : Type*} [LinearOrderedField α]

def Set_1 : Set α := {x | x^2 - 3*x + 2 = 0}

def Set_2 (a : α) : Set α := {x | x^2 - a*x + (a-1) = 0}

def Set_3 (m : α) : Set α := {x | x^2 - m*x + 2 = 0}

theorem find_constants (a m : α) :
  (Set_1 ∪ Set_2 a = Set_1) ∧ (Set_1 ∩ Set_2 a = Set_3 m) → 
  a = 3 ∧ m = 3 :=
by sorry

end find_constants_l236_236678


namespace intersection_with_y_axis_l236_236562

def point := (ℕ × ℕ)

theorem intersection_with_y_axis (p1 p2 : point) (hx1 : p1 = (1, 7)) (hx2 : p2 = (3, 11)) :
  ∃ y : ℕ, (0, y) ∈ (set_of (λ p : ℕ × ℕ,  (p2.2 - p1.2) = (2 * (p2.1 - p1.1))) ∧ y = 5 :=
by
  sorry

end intersection_with_y_axis_l236_236562


namespace custom_op_equality_l236_236583

def custom_op (x y : ℕ) : ℕ :=
  x * y - 2 * x + 3 * y

theorem custom_op_equality : custom_op 8 5 - custom_op 5 8 = -15 :=
  by
    sorry

end custom_op_equality_l236_236583


namespace exterior_angle_bisectors_collinear_tangents_to_circumcircle_collinear_l236_236948

-- Part (a)
theorem exterior_angle_bisectors_collinear (A B C D E F : Point) 
  (h1: triangle A B C )
  (h2: is_exterior_angle_bisector A B D)
  (h3: is_exterior_angle_bisector B C E)
  (h4: is_exterior_angle_bisector C A F)
  (h5: intersects_extension D B C A)
  (h6: intersects_extension E C A B)
  (h7: intersects_extension F A B C):
  collinear [D, E, F] := 
sorry

-- Part (b)
theorem tangents_to_circumcircle_collinear (A B C G H I : Point) 
  (h1: triangle A B C )
  (h2: has_circumcircle A B C)
  (h3: is_tangent_to_circumcircle A G A B C)
  (h4: is_tangent_to_circumcircle B H A B C)
  (h5: is_tangent_to_circumcircle C I A B C)
  (h6: intersects_opposite_side G B C)
  (h7: intersects_opposite_side H C A)
  (h8: intersects_opposite_side I A B):
  collinear [G, H, I] := 
sorry

end exterior_angle_bisectors_collinear_tangents_to_circumcircle_collinear_l236_236948


namespace total_cookies_l236_236575

variable (glenn_cookies : ℕ) (kenny_cookies : ℕ) (chris_cookies : ℕ)
hypothesis (h1 : glenn_cookies = 24)
hypothesis (h2 : glenn_cookies = 4 * kenny_cookies)
hypothesis (h3 : chris_cookies = kenny_cookies / 2)

theorem total_cookies : glenn_cookies + kenny_cookies + chris_cookies = 33 :=
by sorry

end total_cookies_l236_236575


namespace cone_volume_proof_l236_236477

noncomputable def cone_volume (r : ℝ) (theta : ℝ) (V : ℝ) : Prop :=
  r = 1 ∧ theta = 90 ∧ V = (sqrt 15 / 3) * π

theorem cone_volume_proof : 
  cone_volume 1 90 ((sqrt 15 / 3) * π) :=
by
  -- Code to prove the statement
  sorry

end cone_volume_proof_l236_236477


namespace ratio_of_hexagon_areas_l236_236075

-- Definition of the problem parameters
def radius (r : ℝ) := r > 0

-- The areas of the hexagons in terms of the radius of the circle
def area_inscribed_hexagon (r : ℝ) : ℝ := (3 * Real.sqrt 3) / 2 * r^2
def area_circumscribed_hexagon (r : ℝ) : ℝ := 3 * Real.sqrt 3 * r^2

-- Problem statement to prove the ratio of the areas
theorem ratio_of_hexagon_areas (r : ℝ) (hr : radius r) : 
  (area_inscribed_hexagon r) / (area_circumscribed_hexagon r) = 3 / 4 := 
by
  sorry

end ratio_of_hexagon_areas_l236_236075


namespace playground_area_l236_236456

theorem playground_area
  (w l : ℕ)
  (h₁ : l = 2 * w + 25)
  (h₂ : 2 * (l + w) = 650) :
  w * l = 22500 := 
sorry

end playground_area_l236_236456


namespace solution_set_inequality_l236_236473

theorem solution_set_inequality (x : ℝ) : (x-2)/(x+3) > 0 ↔ x ∈ set.Ioo (-∞) (-3) ∪ set.Ioo (2) (+∞) :=
sorry

end solution_set_inequality_l236_236473


namespace tan_double_angle_value_l236_236670

def f (x : ℝ) : ℝ := sin x - cos x

theorem tan_double_angle_value (x : ℝ) (h : ∀ x, deriv (deriv f x) = (1 / 2) * f x) : 
  tan (2 * x) = 3 / 4 :=
by 
  sorry

end tan_double_angle_value_l236_236670


namespace divisible_by_11_l236_236235

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1
else if n = 2 then 3
else (n + 1) * a (n - 1) - n * a (n - 2)

theorem divisible_by_11 :
  ∀ n : ℕ, n = 4 ∨ n = 8 ∨ n = 10 ∨ n ≥ 10 → a n % 11 = 0 := by
  sorry

end divisible_by_11_l236_236235


namespace mary_total_baseball_cards_l236_236010

noncomputable def mary_initial_baseball_cards : ℕ := 18
noncomputable def torn_baseball_cards : ℕ := 8
noncomputable def fred_given_baseball_cards : ℕ := 26
noncomputable def mary_bought_baseball_cards : ℕ := 40

theorem mary_total_baseball_cards :
  mary_initial_baseball_cards - torn_baseball_cards + fred_given_baseball_cards + mary_bought_baseball_cards = 76 :=
by
  sorry

end mary_total_baseball_cards_l236_236010


namespace prob_C_eq_prob_CD_l236_236125

-- Define the probabilities for regions A and B
def prob_A : ℚ := 1 / 2
def prob_B : ℚ := 1 / 8

-- Define the probability for regions C and D as equal
def prob_CD : ℚ := 3 / 16

-- State the theorem to prove the probability of region C
theorem prob_C_eq_prob_CD :
  (prob_A + prob_B + prob_CD + prob_CD = 1) →
  prob_CD = 3 / 16 :=
by
  intro h,
  exact sorry

end prob_C_eq_prob_CD_l236_236125


namespace largest_x_value_satisfies_largest_x_value_l236_236873

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l236_236873


namespace max_y_difference_l236_236586

theorem max_y_difference : (∃ x, (5 - 2 * x^2 + 2 * x^3 = 1 + x^2 + x^3)) ∧ 
                           (∀ y1 y2, y1 = 5 - 2 * (2^2) + 2 * (2^3) ∧ y2 = 5 - 2 * (1/2)^2 + 2 * (1/2)^3 → 
                           (y1 - y2 = 11.625)) := sorry

end max_y_difference_l236_236586


namespace sum_of_center_coords_l236_236216

theorem sum_of_center_coords (x y : ℝ) :
  (∃ k : ℝ, (x + 2)^2 + (y + 3)^2 = k ∧ (x^2 + y^2 = -4 * x - 6 * y + 5)) -> x + y = -5 :=
by
sorry

end sum_of_center_coords_l236_236216


namespace idle_day_forfeit_amount_l236_236549

theorem idle_day_forfeit_amount
  (daily_wage: ℕ) (total_days: ℕ) (net_earnings: ℕ) (days_worked: ℕ) (forfeit: ℕ) :
  daily_wage = 20 ∧ total_days = 25 ∧ net_earnings = 450 ∧ days_worked = 23 →
  forfeit = 25 :=
by
  intros conditions
  cases conditions with daily_wage_h rest
  cases rest with total_days_h rest
  cases rest with net_earnings_h rest
  cases rest with days_worked_h forfeit_h
  sorry

end idle_day_forfeit_amount_l236_236549


namespace first_year_after_2000_with_digit_sum_15_l236_236084

theorem first_year_after_2000_with_digit_sum_15 : ∃ y, y > 2000 ∧ (y.digits.sum = 15) ∧ ∀ z, z > 2000 ∧ (z.digits.sum = 15) → y ≤ z := 
sorry

end first_year_after_2000_with_digit_sum_15_l236_236084


namespace largest_x_satisfying_equation_l236_236862

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l236_236862


namespace amelia_drove_tuesday_l236_236034

-- Define the known quantities
def total_distance : ℕ := 8205
def distance_monday : ℕ := 907
def remaining_distance : ℕ := 6716

-- Define the distance driven on Tuesday and state the theorem
def distance_tuesday : ℕ := total_distance - (distance_monday + remaining_distance)

-- Theorem stating the distance driven on Tuesday is 582 kilometers
theorem amelia_drove_tuesday : distance_tuesday = 582 := 
by
  -- We skip the proof for now
  sorry

end amelia_drove_tuesday_l236_236034


namespace boys_from_Maple_l236_236115

theorem boys_from_Maple : 
  ∀ (students_total boys_total girls_total jonas_total clay_total maple_total jonas_girls_total clay_girls_total : ℕ),
  students_total = 150 →
  boys_total = 85 →
  girls_total = 65 →
  jonas_total = 50 →
  clay_total = 70 →
  maple_total = 30 →
  jonas_girls_total = 25 →
  clay_girls_total = 30 →
  boys_total + girls_total = students_total →
  jonas_total + clay_total + maple_total = students_total →
  jonas_girls_total + clay_girls_total ≤ girls_total →
  𝑏𝑜𝑦𝑠 = students_total - girls_total →
  ∃ maple_boys, maple_boys = 20 := 
by
  sorry

end boys_from_Maple_l236_236115


namespace problem1_problem2_problem3_l236_236266

noncomputable def f (x : ℝ) := Real.log x - x

theorem problem1 : 
  let y := -1,
  let point := (1 : ℝ, -1 : ℝ),
  ∃ l : ℝ → ℝ, l = λ x, y := 
by
  sorry

theorem problem2 (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) : 
  |f x₁| > (Real.log x₂) / x₂ :=
by
  sorry

theorem problem3 (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) : 
  (f m + m - (f n + n)) / (m - n) > m / (m^2 + n^2) :=
by
  sorry

end problem1_problem2_problem3_l236_236266


namespace nine_possible_xs_l236_236284

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end nine_possible_xs_l236_236284


namespace maximum_m2_n2_l236_236587

theorem maximum_m2_n2 
  (m n : ℤ)
  (hm : 1 ≤ m ∧ m ≤ 1981) 
  (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m*n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end maximum_m2_n2_l236_236587


namespace notebooks_cost_l236_236446

theorem notebooks_cost 
  (P N : ℝ)
  (h1 : 96 * P + 24 * N = 520)
  (h2 : ∃ x : ℝ, 3 * P + x * N = 60)
  (h3 : P + N = 15.512820512820513) :
  ∃ x : ℕ, x = 4 :=
by
  sorry

end notebooks_cost_l236_236446


namespace approx_1_08_pow_3_96_approx_frac_sin_arctg_div_pow_l236_236170

theorem approx_1_08_pow_3_96 : 1.08 ^ 3.96 ≈ 1.32 :=
by sorry

theorem approx_frac_sin_arctg_div_pow :
  (sin 1.49 * arctan 0.07) / (2^2.95) ≈ 0.00875 :=
by sorry

end approx_1_08_pow_3_96_approx_frac_sin_arctg_div_pow_l236_236170


namespace E_xi_eq_27_div_2_l236_236064

theorem E_xi_eq_27_div_2 :
  (∃ p : ℕ → ℝ,
    (p 1 = 0) ∧ (p 2 = 0) ∧ (p 3 = 0) ∧ (p 4 = 1 / 8) ∧ (p 5 = 1 / 16) ∧
    (∀ n ≥ 6, p n = p (n - 1) - (1 / 16) * p (n - 4)) ∧
    sum (λ k, k * p k) 5 (max_supported k) = 27 / 2) :=
sorry

end E_xi_eq_27_div_2_l236_236064


namespace algebraic_expression_value_l236_236695

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := by
  sorry

end algebraic_expression_value_l236_236695


namespace sqrt_3x_eq_5x_largest_value_l236_236881

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l236_236881


namespace total_cookies_l236_236568

variable (ChrisCookies KennyCookies GlennCookies : ℕ)
variable (KennyHasCookies : GlennCookies = 4 * KennyCookies)
variable (ChrisHasCookies : ChrisCookies = KennyCookies / 2)
variable (GlennHas24Cookies : GlennCookies = 24)

theorem total_cookies : GlennCookies + KennyCookies + ChrisCookies = 33 := 
by
  have KennyCookiesEq : KennyCookies = 24 / 4 := by 
    rw [GlennHas24Cookies, mul_div_cancel_left, nat.mul_comm, nat.one_div, nat.div_self] ; trivial
  have ChrisCookiesEq : ChrisCookies = 6 / 2 := by 
    rw [KennyCookiesEq, ChrisHasCookies]
  rw [ChrisCookiesEq, KennyCookiesEq, GlennHas24Cookies]
  exact sorry

end total_cookies_l236_236568


namespace minimum_rubles_to_reverse_order_of_chips_100_l236_236354

noncomputable def minimum_rubles_to_reverse_order_of_chips (n : ℕ) : ℕ :=
if n = 100 then 61 else 0

theorem minimum_rubles_to_reverse_order_of_chips_100 :
  minimum_rubles_to_reverse_order_of_chips 100 = 61 :=
by sorry

end minimum_rubles_to_reverse_order_of_chips_100_l236_236354


namespace parabola_constant_l236_236538

theorem parabola_constant (b c : ℝ)
  (h₁ : -20 = 2 * (-2)^2 + b * (-2) + c)
  (h₂ : 24 = 2 * 2^2 + b * 2 + c) : 
  c = -6 := 
by 
  sorry

end parabola_constant_l236_236538


namespace coin_loading_impossible_l236_236961

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l236_236961


namespace a_b_finish_job_in_15_days_l236_236501

theorem a_b_finish_job_in_15_days (A B C : ℝ) 
  (h1 : A + B + C = 1 / 5)
  (h2 : C = 1 / 7.5) : 
  (1 / (A + B)) = 15 :=
by
  sorry

end a_b_finish_job_in_15_days_l236_236501


namespace charles_more_than_robin_l236_236508

-- Definitions and basic setup
variables {E R C : ℝ} -- Let E, R, and C be wages earned by Erica, Robin, and Charles respectively.

-- Condition: Robin's wage is 30% more than Erica's wage.
def robin_wage (E : ℝ) : ℝ := E + 0.30 * E

-- Condition: Charles's wage is 60% more than Erica's wage.
def charles_wage (E : ℝ) : ℝ := E + 0.60 * E

-- Percentage more calculation
def percentage_more_than (high low : ℝ) : ℝ :=
  ((high - low) / low) * 100

-- Theorem statement
theorem charles_more_than_robin (E : ℝ) :
  percentage_more_than (charles_wage E) (robin_wage E) = (3 / 13) * 100 :=
by
  sorry

end charles_more_than_robin_l236_236508


namespace first_year_sum_of_digits_15_l236_236082

theorem first_year_sum_of_digits_15 : ∃ y : ℕ, y > 2000 ∧ sum_of_digits y = 15 ∧ ∀ z, (z > 2000 ∧ sum_of_digits z = 15) → y ≤ z :=
by
  sorry

-- Helper function to calculate the sum of digits of a given number.
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

end first_year_sum_of_digits_15_l236_236082


namespace area_ratio_of_triangles_l236_236330

theorem area_ratio_of_triangles (AC AD : ℝ) (h : ℝ) (hAC : AC = 1) (hAD : AD = 4) :
  (AC * h / 2) / ((AD - AC) * h / 2) = 1 / 3 :=
by
  sorry

end area_ratio_of_triangles_l236_236330


namespace solve_exponential_equation_l236_236792

theorem solve_exponential_equation (x : ℝ) :
  3 * (16^x) + 37 * (36^x) = 26 * (81^x) ↔ x = 1/2 :=
by
  have h1 : 16 = (2:ℝ)^4 := by norm_num
  have h2 : 36 = (2:ℝ * 3)^2 := by norm_num
  have h3 : 81 = (3:ℝ)^4 := by norm_num
  sorry

end solve_exponential_equation_l236_236792


namespace taxi_division_number_of_ways_to_divide_six_people_l236_236026

theorem taxi_division (people : Finset ℕ) (h : people.card = 6) (taxi1 taxi2 : Finset ℕ) 
  (h1 : taxi1.card ≤ 4) (h2 : taxi2.card ≤ 4) (h_union : people = taxi1 ∪ taxi2) (h_disjoint : Disjoint taxi1 taxi2) :
  (taxi1.card = 3 ∧ taxi2.card = 3) ∨ 
  (taxi1.card = 4 ∧ taxi2.card = 2) :=
sorry

theorem number_of_ways_to_divide_six_people : 
  ∃ n : ℕ, n = 50 :=
sorry

end taxi_division_number_of_ways_to_divide_six_people_l236_236026


namespace sqrt_3x_eq_5x_largest_value_l236_236882

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l236_236882


namespace no_valid_sequence_exists_l236_236533

-- Define a structure to represent the chessboard and movements
structure Chessboard :=
  (rows : Fin 6)
  (cols : Fin 6)

-- Define the possible moves, alternating 1 and 2 squares
inductive Move
| move1 : Chessboard → Chessboard → Prop
| move2 : Chessboard → Chessboard → Prop

-- Define the condition that no square can be revisited
def no_reentry (sq : Fin 6 → Fin 6 → Prop) : Prop :=
  ∀ i j, (i, j) ≠ sq (i-1) j ∧ (i, j) ≠ sq i (j-1)

-- Define the sequence that satisfies the conditions
def valid_sequence : Prop :=
  ∃ seq : List (Fin 6 × Fin 6) (h : seq.length = 36), -- Ensure the sequence covers 36 distinct squares
  (∀ i, (seq.get? i).is_some → no_reentry seq) ∧       -- Ensure no square is revisited
  (∀ i, i < 35 →       -- Ensure alternating move lengths
    (if i % 2 = 0 then Move.move1 
                    (seq.nth_le i (by linarith)) 
                    (seq.nth_le (i+1) (by linarith)) else
     Move.move2 
                    (seq.nth_le i (by linarith)) 
                    (seq.nth_le (i+1) (by linarith))))

-- Now we state the theorem that no such valid sequence exists
theorem no_valid_sequence_exists : ¬ valid_sequence :=
by
  sorry -- The detailed proof steps are omitted as per the instructions.

end no_valid_sequence_exists_l236_236533


namespace OS_length_l236_236381

-- Defining the basic setting
structure Square (A B C D O P Q R S X Y Z : Type) :=
  (AB : segment A B)
  (BC : segment B C)
  (CD : segment C D)
  (DA : segment D A)
  (OA : segment O A)
  (OB : segment O B)
  (OC : segment O C)
  (OD : segment O D)
  (PQ : segment P Q)
  (QR : segment Q R)
  (RS : segment R S)

-- Assumptions of the problem
axiom OP_len (O P : Type) : segment O P := 3
axiom OQ_len (O Q : Type) : segment O Q := 5
axiom OR_len (O R : Type) : segment O R := 4
axiom collinear_intersections (A B C D O P Q R S X Y Z : Type) 
  (SQR : (segment A B) ∩ (segment P Q) = X) 
  (TBC : (segment B C) ∩ (segment Q R) = Y) 
  (UCD : (segment C D) ∩ (segment R S) = Z)
  : collinear X Y Z

-- Goal to prove
theorem OS_length (A B C D O P Q R S X Y Z : Type)
  [Square A B C D O P Q R S X Y Z]
  (hOPO : segment O P = OP_len O P)
  (hOQO : segment O Q = OQ_len O Q)
  (hORO : segment O R = OR_len O R)
  (hCollinear : collinear_intersections A B C D O P Q R S X Y Z)
  : segment O S = (60/23) :=
sorry -- Proof omitted

end OS_length_l236_236381


namespace correct_statements_l236_236667

theorem correct_statements (a : ℝ) (x : ℝ) :
  (¬ (Real.pow (4 : ℝ) (Real.log (-2))) = ±2) ∧
  (¬ (∀ x ∈ set.Icc (-1 : ℝ) (2 : ℝ), ((λ x, x ^ 2 + 1) x ∈ set.Icc (2 : ℝ) (5 : ℝ)))) ∧
  (∀ x : ℝ, ¬ (f (x) < 0 && x > 0)) ∧
  (a > 0 ∧ a ≠ 1 → (f(-1 : ℝ) = -1)) ∧
  (¬ (log a < 1 → a ∈ set.Icc (-1) (Real.exp 1))) → ((3 = 3) ∧ (4 = 4)) :=
by
  intro h
  sorry

end correct_statements_l236_236667


namespace machine_shop_tool_distance_l236_236534

theorem machine_shop_tool_distance :
  ∃ a b : ℝ, (a^2 + b^2 = 61) ∧ (a^2 + (b + 8)^2 = 72 ∧ (a + 3)^2 + b^2 = 72) :=
begin
  sorry
end

end machine_shop_tool_distance_l236_236534


namespace fourth_intersection_point_l236_236360

noncomputable def curve (x : ℝ) : ℝ := 2 / x

def circle (a b r x : ℝ) : Prop :=
  let y := curve x
  (x - a)^2 + (y - b)^2 = r^2

theorem fourth_intersection_point
  (a b r : ℝ)
  (h1 : circle a b r 3)
  (h2 : circle a b r (-4))
  (h3 : circle a b r (1/4))
  : circle a b r (-1/3) :=
sorry

end fourth_intersection_point_l236_236360


namespace product_divisibility_l236_236002

theorem product_divisibility (a b c : ℤ)
  (h₁ : (a + b + c) ^ 2 = -(a * b + a * c + b * c))
  (h₂ : a + b ≠ 0)
  (h₃ : b + c ≠ 0)
  (h₄ : a + c ≠ 0) :
  (a + b) * (a + c) % (b + c) = 0 ∧
  (a + b) * (b + c) % (a + c) = 0 ∧
  (a + c) * (b + c) % (a + b) = 0 := by
  sorry

end product_divisibility_l236_236002


namespace coin_loading_impossible_l236_236962

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l236_236962


namespace area_inside_C_but_outside_A_B_D_l236_236579

-- Definition of circles' radii
def radius_A : ℝ := 2
def radius_B : ℝ := 2
def radius_C : ℝ := 2
def radius_D : ℝ := 1

-- Conditions of the problem
axiom tangency_A_B : True -- Circle A and B are tangent
axiom tangency_C_M : True -- Circle C is tangent at midpoint M of AB
axiom tangency_D_C : True -- Circle D is externally tangent to Circle C
axiom tangency_D_A : True -- Circle D is internally tangent to Circle A
axiom tangency_D_B : True -- Circle D is internally tangent to Circle B

-- Proposition stating the area inside C but outside A, B, and D is 4 - 0.5*pi
theorem area_inside_C_but_outside_A_B_D : 
  let area_C := π * radius_C^2 in
  let area_A := π * radius_A^2 in
  let area_B := π * radius_B^2 in
  let area_D := π * radius_D^2 in
  (area_C - (0.5 * area_A + 0.5 * area_B + area_D)) = 4 - 0.5 * π :=
sorry

end area_inside_C_but_outside_A_B_D_l236_236579


namespace relation_among_a_b_c_l236_236265

noncomputable def f (x : ℝ) : ℝ := 2 ^ |x|

def a : ℝ := f (Real.log 3 / Real.log (1 / 2))
def b : ℝ := Real.log 5 / Real.log 2
def c : ℝ := f 0

theorem relation_among_a_b_c : c < b ∧ b < a :=
by
  have fa : a = 3 := by sorry
  have fb : 2 < b ∧ b < 3 := by sorry
  have fc : c = 1 := by sorry
  sorry

end relation_among_a_b_c_l236_236265


namespace air_conditioner_savings_l236_236051

theorem air_conditioner_savings :
  let price_X := 575
      surcharge_rate_X := 0.04
      install_charge_X := 82.50
      price_Y := 530
      surcharge_rate_Y := 0.03
      install_charge_Y := 93.00
      surcharge_X := price_X * surcharge_rate_X
      total_charge_X := price_X + surcharge_X + install_charge_X
      surcharge_Y := price_Y * surcharge_rate_Y
      total_charge_Y := price_Y + surcharge_Y + install_charge_Y
      savings := total_charge_X - total_charge_Y
  in savings = 41.60 :=
by
  sorry

end air_conditioner_savings_l236_236051


namespace quadratic_has_real_root_l236_236066

theorem quadratic_has_real_root (a b : ℝ) : (∃ x : ℝ, x^2 + a * x + b = 0) :=
by
  -- To use contradiction, we assume the negation
  have h : ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry
  -- By contradiction, this assumption should lead to a contradiction
  sorry

end quadratic_has_real_root_l236_236066


namespace inequality_with_equality_condition_l236_236024

variables {a b c d : ℝ}

theorem inequality_with_equality_condition (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 1) : 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) ∧ 
    (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1 / 2 ↔ a = b ∧ b = c ∧ c = d) := 
sorry

end inequality_with_equality_condition_l236_236024


namespace cos_B_correct_l236_236732

-- Given conditions for the triangle
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle
variables (h_triangle : ∀ (x y z : ℝ), x + y + z = π)
variables (h_sine_ratio : ∀ (x y z : ℝ), x / y / z = 3 / 4 / 6)
variables (h_law_of_sines : a / sin A = b / sin B = c / sin C)

noncomputable def cosB_calc : ℝ :=
  (a^2 + c^2 - b^2) / (2 * a * c)

theorem cos_B_correct : cosB_calc = 29 / 36 :=
by sorry

end cos_B_correct_l236_236732


namespace probability_not_all_dice_show_different_l236_236919

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l236_236919


namespace sum_zeros_of_f_l236_236476

def f (x : ℝ) : ℝ := (x - 1) * Real.sin (π * x) - 1

theorem sum_zeros_of_f :
  let Z := {x : ℝ | f x = 0 ∧ -1 < x ∧ x < 3} in
  ∑ (x : ℝ) in Z, x = 1.5 :=
by 
  -- proof here
  sorry

end sum_zeros_of_f_l236_236476


namespace smallest_number_greater_than_l236_236211

theorem smallest_number_greater_than : 
  ∀ (S : Set ℝ), S = {0.8, 0.5, 0.3} → 
  (∃ x ∈ S, x > 0.4 ∧ (∀ y ∈ S, y > 0.4 → x ≤ y)) → 
  x = 0.5 :=
by
  sorry

end smallest_number_greater_than_l236_236211


namespace factors_of_12_factors_of_18_l236_236808

def is_factor (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

theorem factors_of_12 : 
  {k : ℕ | is_factor 12 k} = {1, 12, 2, 6, 3, 4} :=
by
  sorry

theorem factors_of_18 : 
  {k : ℕ | is_factor 18 k} = {1, 18, 2, 9, 3, 6} :=
by
  sorry

end factors_of_12_factors_of_18_l236_236808


namespace diagonals_intersect_at_one_point_l236_236717

variables (A B C D E F : Point)
variables (AB BC CD DE EF FA : ℝ)
variables (angle_A angle_C angle_E : ℝ)
variables (α β γ : Line) -- where α, β, γ are the diagonals BO, DO, FO respectively

-- Given conditions
def conditions (hAB : AB = BC) 
               (hCD : CD = DE) 
               (hEF : EF = FA) 
               (hangle : angle_A = angle_C ∧ angle_C = angle_E) : Prop := 
  true

-- Theorem: The main diagonals of the hexagon intersect at a single point
theorem diagonals_intersect_at_one_point
  (h : conditions AB BC CD DE EF FA angle_A angle_C angle_E) :
  ∃ O : Point, is_intersecting_at α β O ∧ is_intersecting_at β γ O ∧ is_intersecting_at γ α O :=
sorry

end diagonals_intersect_at_one_point_l236_236717


namespace probability_not_all_same_l236_236927

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l236_236927


namespace julia_average_speed_l236_236742

-- Define the conditions as constants
def total_distance : ℝ := 28
def total_time : ℝ := 4

-- Define the theorem stating Julia's average speed
theorem julia_average_speed : total_distance / total_time = 7 := by
  sorry

end julia_average_speed_l236_236742


namespace area_quadrilateral_ABCD_l236_236424

theorem area_quadrilateral_ABCD
  (A B C D E : Type)
  (angle_ABC: ∠ABC = 90)
  (angle_ACD: ∠ACD = 90)
  (AC : dist A C = 24)
  (CD : dist C D = 40)
  (AE : dist A E = 6) :
  area ABCD = 624 := 
sorry

end area_quadrilateral_ABCD_l236_236424


namespace evaluate_expression_l236_236196

theorem evaluate_expression (a : ℤ) : ((a + 10) - a + 3) * ((a + 10) - a - 2) = 104 := by
  sorry

end evaluate_expression_l236_236196


namespace quadratic_inequality_solution_l236_236829

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (x^2 - 4*x + 3) < 0 ↔ 1 < x ∧ x < 3 :=
by
  sorry

end quadratic_inequality_solution_l236_236829


namespace range_of_a_l236_236264

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * log x + a * x ^ 2 + 1

theorem range_of_a (a : ℝ) (h_a : a < -1) (h_f: ∀ x1 x2 ∈ Ioi 0, |f a x1 - f a x2| ≥ 4 * |x1 - x2|) : a ≤ -2 :=
sorry

end range_of_a_l236_236264


namespace largest_x_satisfies_eq_l236_236866

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236866


namespace digital_root_frequency_l236_236567

theorem digital_root_frequency :
  let sequence := list.range (10^9 + 1)
  let digital_root (n : ℕ) : ℕ := if n = 0 then 0 else 1 + (n - 1) % 9
  let count_digital_root r := (sequence.filter (λ n, digital_root n = r)).length
  in count_digital_root 1 > count_digital_root 2 :=
by
  sorry

end digital_root_frequency_l236_236567


namespace find_three_numbers_l236_236326

theorem find_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a + b - c = 10) 
  (h3 : a - b + c = 8) : 
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := 
by 
  sorry

end find_three_numbers_l236_236326


namespace find_t_from_tan_conditions_l236_236359

theorem find_t_from_tan_conditions 
  (α t : ℝ)
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + Real.pi / 4) = 4 / t)
  (h3 : Real.tan (α + Real.pi / 4) = (Real.tan (Real.pi / 4) + Real.tan α) / (1 - Real.tan (Real.pi / 4) * Real.tan α)) :
  t = 2 := 
  by
  sorry

end find_t_from_tan_conditions_l236_236359


namespace molecular_weight_correct_l236_236171

-- Define atomic weights of elements
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_D : ℝ := 2.01

-- Define the number of each type of atom in the compound
def num_Ba : ℕ := 2
def num_O : ℕ := 3
def num_H : ℕ := 4
def num_D : ℕ := 1

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  (num_Ba * atomic_weight_Ba) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_D * atomic_weight_D)

-- Theorem stating the molecular weight is 328.71 g/mol
theorem molecular_weight_correct :
  molecular_weight = 328.71 :=
by
  -- The proof will go here
  sorry

end molecular_weight_correct_l236_236171


namespace find_b_of_parabola_axis_of_symmetry_l236_236804

theorem find_b_of_parabola_axis_of_symmetry (b : ℝ) :
  (∀ (x : ℝ), (x = 1) ↔ (x = - (b / (2 * 2))) ) → b = 4 :=
by
  intro h
  sorry

end find_b_of_parabola_axis_of_symmetry_l236_236804


namespace largest_x_satisfying_equation_l236_236861

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l236_236861


namespace probability_not_all_same_l236_236898

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l236_236898


namespace graph_passes_through_point_l236_236823

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 1

theorem graph_passes_through_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : f a 2 = 2 :=
by
  sorry

end graph_passes_through_point_l236_236823


namespace rectangle_area_ratio_l236_236179

-- Define the scenario and parameters
def point := (ℝ × ℝ × ℝ)

-- Define the edges of the cube and points K, L, M, N
def A : point := (0, 0, 0)
def B : point := (2, 0, 0)
def C : point := (2, 0, 2)
def D : point := (0, 0, 2)
def E : point := (0, 2, 0)
def F : point := (2, 2, 0)
def G : point := (2, 2, 2)
def H : point := (0, 2, 2)
def K : point := ((2 + 0) / 2, 0, 0)
def L : point := ((2 + 0) / 2, 2, 2)
def M : point := ((1 + 1) / 2, (0 + 2) / 2, (0 + 2) / 2)
def N : point := ((2 + 2) / 2, 0, (2 + 2) / 2)

-- Calculate the lengths of the rectangle sides
noncomputable def length_EM : ℝ := real.sqrt (((1 - 0)^2 + (1 - 2)^2 + (1 - 0)^2))
noncomputable def length_EN : ℝ := real.sqrt (((1 - 2)^2 + (1 - 0)^2 + (1 - 2)^2))

-- Calculate the areas
noncomputable def area_EMFN : ℝ := length_EM * length_EN
def area_face : ℝ := 2 * 2 -- since each edge length is 2 units

-- The ratio S
noncomputable def ratio_S : ℝ := area_EMFN / area_face

theorem rectangle_area_ratio :
  ratio_S = 3 / 4 :=
sorry

end rectangle_area_ratio_l236_236179


namespace area_of_region_l236_236442

theorem area_of_region :
  let f := λ x : ℝ, 2 * x + 3
  let g := λ x : ℝ, x ^ 2
  let a := -1
  let b := 3
  (∫ x in a..b, f x - g x) = 32 / 3 := sorry

end area_of_region_l236_236442


namespace lambda_range_correct_l236_236638

noncomputable def find_lambda_range (k : ℝ) (h_k : k ≥ sqrt 3) : set ℝ := 
{lambda : ℝ | sqrt 2 < lambda ∧ lambda ≤ (2 * sqrt 6) / 3}

theorem lambda_range_correct :
  ∃ (k : ℝ) (h_k : k ≥ sqrt 3), 
    find_lambda_range k h_k = (λ u, sqrt 2 < u ∧ u ≤ (2 * sqrt 6) / 3) :=
by
  sorry

end lambda_range_correct_l236_236638


namespace arithmetic_progression_count_l236_236760

theorem arithmetic_progression_count (n : ℕ) : 
  let S := finset.range (n + 1) \ {0} in
  let num_sequences := finset.card S / 4 in
  S.card ≥ 2 →
  num_sequences = Nat.floor (n^2 / 4) :=
by sorry

end arithmetic_progression_count_l236_236760


namespace dots_per_blouse_l236_236376

theorem dots_per_blouse (num_bottles : ℕ) (vol_per_bottle : ℕ) (num_blouses : ℕ) (dye_per_dot : ℕ) :
  num_bottles = 50 →
  vol_per_bottle = 400 →
  num_blouses = 100 →
  dye_per_dot = 10 →
  (num_bottles * vol_per_bottle) / num_blouses / dye_per_dot = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end dots_per_blouse_l236_236376


namespace monotonicity_of_function_l236_236037

theorem monotonicity_of_function : 
  ∀ x : ℝ, (0 < x ∧ x < 5) → (x < 1/e → (deriv (λ x, x * log x)) x < 0) ∧ (x > 1/e → (deriv (λ x, x * log x)) x > 0) := 
begin
  assume x hx,
  let f := λ x : ℝ, x * log x,
  have h_deriv : deriv f x = log x + 1, 
  {
    calc deriv f x = deriv (λ x, x * log x) x : by sorry
             ...   = log x + 1 : by sorry,
  },
  split;
  {
    assume h,
    {
      rw h_deriv,
      exact sorry,
    },
    {
      rw h_deriv,
      exact sorry,
    }
  }
end

end monotonicity_of_function_l236_236037


namespace symmetric_line_eqn_l236_236450

theorem symmetric_line_eqn :
  ∀ (x y : ℝ), let line1 := x + 2*y - 1 = 0,
                   point := (1 : ℝ, -1 : ℝ)
               in line_symmetric line1 point = (x + 2*y - 3 = 0) :=
begin
  sorry
end

end symmetric_line_eqn_l236_236450


namespace largest_x_value_satisfies_largest_x_value_l236_236875

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l236_236875


namespace count_distinct_parabolas_l236_236361

def S := {-3, -2, 0, 1, 2, 3}

def is_parabola (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ 0 ∧ b ≠ 0

theorem count_distinct_parabolas : 
  (∃ n : ℕ, n = 62 ∧ 
    ∀ a b c, is_parabola a b c → ay = b^2x^2 + c → --- ∈ distinct_parabolas_count = n ) :=
sorry

end count_distinct_parabolas_l236_236361


namespace probability_both_defective_l236_236143

open Real

/-- Let N be the total number of smartphones, D be the number of defective smartphones.
    The probability that both phones purchased at random are defective is approximately 0.071. -/
theorem probability_both_defective (N D : ℕ) (hN : N = 250) (hD : D = 67) : 
  abs ((D / N) * ((D - 1) / (N - 1)) - 0.071) < 0.01 :=
by
  have P_A := (67 : ℝ) / 250
  have P_B_given_A := (66 : ℝ) / 249
  have P_def := P_A * P_B_given_A
  show abs (P_def - 0.071) < 0.01, from sorry

end probability_both_defective_l236_236143


namespace quadratic_roots_equal_l236_236633

theorem quadratic_roots_equal (m : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + m - 1 = 0 ∧ (∀ y : ℝ, y^2 - 4*y + m-1 = 0 → y = x)) ↔ (m = 5 ∧ (∀ x, x^2 - 4 * x + 4 = 0 ↔ x = 2)) :=
by
  sorry

end quadratic_roots_equal_l236_236633


namespace average_of_list_l236_236437

theorem average_of_list (n : ℕ) (h : (2 + 9 + 4 + n + 2 * n) / 5 = 6) : n = 5 := 
by
  sorry

end average_of_list_l236_236437


namespace part1_part2_l236_236690

variables (a b c d m : Real) 

-- Condition: a and b are opposite numbers
def opposite_numbers (a b : Real) : Prop := a = -b

-- Condition: c and d are reciprocals
def reciprocals (c d : Real) : Prop := c = 1 / d

-- Condition: |m| = 3
def absolute_value_three (m : Real) : Prop := abs m = 3

-- Statement for part 1
theorem part1 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) :
  a + b = 0 ∧ c * d = 1 ∧ (m = 3 ∨ m = -3) :=
by
  sorry

-- Statement for part 2
theorem part2 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) (h4 : m < 0) :
  m^3 + c * d + (a + b) / m = -26 :=
by
  sorry

end part1_part2_l236_236690


namespace hyperbola_asymptotes_l236_236645

theorem hyperbola_asymptotes (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (F₁ F₂ P : ℝ × ℝ)
  (h₃ : P ∈ {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}) 
  (h₄ : dist P F₂ = a) (h₅ : dist P F₁ = 3 * a) (h₆ : angle P F₁ F₂ = π / 2):
  (∀ x : ℝ, ∃ y : ℝ, y = sqrt(6) / 2 * x ∨ y = -sqrt(6) / 2 * x) := sorry

end hyperbola_asymptotes_l236_236645


namespace impossibility_exchange_l236_236736

theorem impossibility_exchange :
  ¬ ∃ (x y z : ℕ), (x + y + z = 10) ∧ (x + 3 * y + 5 * z = 25) := 
by
  sorry

end impossibility_exchange_l236_236736


namespace paraplex_line_intersection_l236_236621

theorem paraplex_line_intersection :
  let parabola_vertex (b : ℝ) := (0 : ℝ, 2 * b ^ 2)
  let line_eq (x b : ℝ) := x + b
  ∃ bs : Finset ℝ, bs.card = 2 ∧ ∀ b ∈ bs, line_eq 0 b = parabola_vertex b.2 :=
sorry

end paraplex_line_intersection_l236_236621


namespace impossible_to_load_two_coins_l236_236973

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l236_236973


namespace incorrect_relations_count_l236_236160

theorem incorrect_relations_count :
  (¬ (1 : ∀ α, Set α) ⊆ ({0, 1, 2, 3} : Set ℕ)) ∧
  (¬ ({1} ∈ ({0, 1, 2, 3} : Set (Set ℕ)))) ∧
  (({0, 1, 2, 3} : Set ℕ) ⊆ {0, 1, 2, 3}) ∧
  ((∅ : Set ℕ) ⊂ ({0} : Set ℕ)) →
  2 = 2 :=
begin
  sorry
end

end incorrect_relations_count_l236_236160


namespace max_cardinality_of_A_l236_236639

theorem max_cardinality_of_A (p n : ℕ) (hp : Nat.prime p) (h_cond : p ≥ n ∧ n ≥ 3) :
  ∃ (A : Finset (Fin n → Fin p)), (∀ x y ∈ A, x ≠ y → ∃ (k l m : Fin n), k ≠ l ∧ l ≠ m ∧ m ≠ k ∧ x k ≠ y k ∧ x l ≠ y l ∧ x m ≠ y m) ∧ A.card = p^(n-2) := sorry

end max_cardinality_of_A_l236_236639


namespace count_valid_numbers_l236_236311
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l236_236311


namespace largest_number_in_set_is_2_l236_236556

-- Define the set of real numbers
def numbers : set ℝ := {1, real.sqrt 3, 0, 2}

-- Prove that the maximum of this set is 2
theorem largest_number_in_set_is_2 : Sup numbers = 2 :=
by
  sorry

end largest_number_in_set_is_2_l236_236556


namespace four_digit_numbers_property_l236_236318

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l236_236318


namespace coin_loading_impossible_l236_236966

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l236_236966


namespace eval_log32_4_l236_236595

noncomputable def log_base_change (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem eval_log32_4 : log_base_change 32 4 = 2 / 5 := 
by 
  sorry

end eval_log32_4_l236_236595


namespace value_of_expression_l236_236692

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := 
by 
  have h₁ : x^2 - 2*x = 2 := by linarith
  calc
    3*x^2 - 6*x + 9 = 3*(x^2 - 2*x) + 9 : by ring
                ... = 3*2 + 9           : by rw [h₁]
                ... = 15                : by norm_num

end value_of_expression_l236_236692


namespace monotonic_increasing_on_interval_range_on_interval_l236_236669

-- Definition of the function
def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

-- Monotonicity on the interval (-1, +∞)
theorem monotonic_increasing_on_interval : ∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → f x1 < f x2 :=
by 
  intro x1 x2 x1_gt_minus1 x2_gt_minus1 x1_lt_x2
  sorry

-- Range of the function on the interval [0, 2]
theorem range_on_interval : set.range (λ x, f x) (set.Icc 0 2) = set.Icc (1 : ℝ) (5/3 : ℝ) :=
by
  sorry

end monotonic_increasing_on_interval_range_on_interval_l236_236669


namespace probability_not_all_same_l236_236899

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l236_236899


namespace sum_of_first_five_terms_l236_236721

variables (a : ℕ → ℝ) (d : ℝ)

-- Definitions and conditions
def arithmetic_sequence := ∀ n, a (n + 1) = a n + d

axiom a2a3a4_eq_3 : a 2 + a 3 + a 4 = 3

def S (n : ℕ) := ∑ i in finset.range n, a i

-- Theorem statement
theorem sum_of_first_five_terms :
  arithmetic_sequence a d →
  a (3 : ℕ) = 1 →
  S a 5 = 5 :=
begin
  intros h_seq h_a3,
  sorry
end

end sum_of_first_five_terms_l236_236721


namespace cylinder_radius_unique_l236_236735

theorem cylinder_radius_unique
  (r : ℝ) (h : ℝ) (V : ℝ) (y : ℝ)
  (h_eq : h = 2)
  (V_eq : V = 2 * Real.pi * r ^ 2)
  (y_eq_increase_radius : y = 2 * Real.pi * ((r + 6) ^ 2 - r ^ 2))
  (y_eq_increase_height : y = 6 * Real.pi * r ^ 2) :
  r = 6 :=
by
  sorry

end cylinder_radius_unique_l236_236735


namespace is_param_eq_l236_236513

-- Define the problem's parameters
variables (A B C : Type) -- the vertices of the triangle
variables (r : ℝ) -- the radius of the circle
variables (d : ℝ) -- the distance between C and either A or B (since C is equidistant from A and B)
variables (s : ℝ) -- the sum of AC and BC

-- Define conditions based on the problem
def is_isosceles_triangle (A B C : Type) (d : ℝ) : Prop :=
  (dist A C = d) ∧ (dist B C = d)

def is_inscribed_circle (A B C : Type) (r : ℝ) : Prop := 
  -- inscribed means that the distance from the center to any vertex equals the radius
  ∃ O : Type, dist O A = r ∧ dist O B = r ∧ dist O C = r

def base_is_chord (A B : Type) : Prop :=
  -- base AB is a chord of the circle
  ¬ is_diameter A B

-- The desired proof statement
theorem is_param_eq (A B C : Type) (r : ℝ) (d : ℝ) (s : ℝ)
  (h1 : is_isosceles_triangle A B C d)
  (h2 : is_inscribed_circle A B C r)
  (h3 : base_is_chord A B) :
  s^2 = 8 * r^2 := sorry
 
end is_param_eq_l236_236513


namespace linear_equation_in_options_l236_236496

def is_linear_equation_with_one_variable (eqn : String) : Prop :=
  eqn = "3 - 2x = 5"

theorem linear_equation_in_options :
  is_linear_equation_with_one_variable "3 - 2x = 5" :=
by
  sorry

end linear_equation_in_options_l236_236496


namespace total_cookies_l236_236576

variable (glenn_cookies : ℕ) (kenny_cookies : ℕ) (chris_cookies : ℕ)
hypothesis (h1 : glenn_cookies = 24)
hypothesis (h2 : glenn_cookies = 4 * kenny_cookies)
hypothesis (h3 : chris_cookies = kenny_cookies / 2)

theorem total_cookies : glenn_cookies + kenny_cookies + chris_cookies = 33 :=
by sorry

end total_cookies_l236_236576


namespace elgin_has_10_dollars_l236_236165

theorem elgin_has_10_dollars (A B C D E : ℕ) 
  (hABC : |A + B + C + D + E| = 56)
  (hAB : |A - B| = 19)
  (hBC : |B - C| = 7)
  (hCD : |C - D| = 5)
  (hDE : |D - E| = 4)
  (hEA : |E - A| = 11) :
  E = 10 := 
  by
  sorry

end elgin_has_10_dollars_l236_236165


namespace least_even_integer_square_l236_236221

theorem least_even_integer_square (E : ℕ) (h_even : E % 2 = 0) (h_square : ∃ (I : ℕ), 300 * E = I^2) : E = 6 ∧ ∃ (I : ℕ), I = 30 ∧ 300 * E = I^2 :=
sorry

end least_even_integer_square_l236_236221


namespace find_tenth_number_l236_236063

open Finset

noncomputable def digits : Finset ℕ := {1, 3, 6, 9}

theorem find_tenth_number : 
  let permutations := (digits.perm 4).toList.map (λ l, l.foldr (λ x acc, 10 * acc + x) 0)
  let sorted_permutations := permutations.qsort (≤)
  sorted_permutations.nth 9 = some 3691 :=
sorry

end find_tenth_number_l236_236063


namespace number_of_ways_to_assign_6_grades_l236_236400

def valid_grades (grades : List ℕ) : Prop :=
  grades.all (λ g, g = 2 ∨ g = 3 ∨ g = 4) ∧
  ¬ grades.contains_adjacent (λ g => g = 2)

def a (n : ℕ) : ℕ
| 0         := 1
| 1         := 3
| 2         := 8
| (n + 1) := 2 * a n + 2 * a (n - 1)

theorem number_of_ways_to_assign_6_grades : 
  a 6 = 448 :=
by
  sorry

end number_of_ways_to_assign_6_grades_l236_236400


namespace arrangement_volunteers_l236_236022

-- Definitions based on conditions:
def num_volunteers : ℕ := 5

def ways_friday : ℕ := num_volunteers.choose 1
def ways_saturday : ℕ := (num_volunteers - 1).choose 2
def ways_sunday : ℕ := (num_volunteers - 1 - 2).choose 1

-- The proof problem statement:
theorem arrangement_volunteers :
  ways_friday * ways_saturday * ways_sunday = 60 :=
by
  sorry

end arrangement_volunteers_l236_236022


namespace collinear_A_S_P_l236_236070

-- Definitions for the conditions
variables 
  (Ω : Type*) [metric_space Ω] [normed_group Ω] [hnc : normed_space ℝ Ω]
  (A B C O S P : Ω)
  (circumcircle_ABC : circle Ω)
  (circumcircle_OBC : circle Ω)
  (circle_AO : circle Ω)

-- Assumptions based on conditions
def triangle_inscribed := (A ∈ circumcircle_ABC) ∧ (B ∈ circumcircle_ABC) ∧ (C ∈ circumcircle_ABC)
def center_O := center circumcircle_ABC = O
def AO_diameter_intersects := (diameter circle_AO = ⟨A, O⟩) ∧ (S ∈ circle_AO) ∧ (S ≠ O) ∧ (S ∈ circumcircle_OBC)
def tangents_intersect_at_P := (tangent line_from_B := B) = P ∧ (tangent line_from_C := C) = P

-- Proof that A, S, and P are collinear
theorem collinear_A_S_P :
  triangle_inscribed Ω A B C O circumcircle_ABC →
  center_O Ω A B C O circumcircle_ABC →
  AO_diameter_intersects Ω A B C O S circle_AO circumcircle_OBC →
  tangents_intersect_at_P Ω B C P →
  collinear [A, S, P] := 
by 
  sorry

end collinear_A_S_P_l236_236070


namespace four_digit_numbers_with_property_l236_236303

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l236_236303


namespace coin_loading_impossible_l236_236963

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l236_236963


namespace none_of_inequalities_true_l236_236012

theorem none_of_inequalities_true (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) 
(hxa : x < a) (hyb : y ≠ b) :
  ¬ ((x + y < a + b) ∨ (x - y < a - b) ∨ (x * y < a * b) ∨ (x / y < a / b)). 
by
  sorry

end none_of_inequalities_true_l236_236012


namespace find_D_l236_236383

theorem find_D (A B D : ℕ) (h1 : (100 * A + 10 * B + D) * (A + B + D) = 1323) (h2 : A ≥ B) : D = 1 :=
sorry

end find_D_l236_236383


namespace loss_percentage_is_75_l236_236507

-- Given conditions
def cost_price_one_book (C : ℝ) : Prop := C > 0
def selling_price_one_book (S : ℝ) : Prop := S > 0
def cost_price_5_equals_selling_price_20 (C S : ℝ) : Prop := 5 * C = 20 * S

-- Proof goal
theorem loss_percentage_is_75 (C S : ℝ) (h1 : cost_price_one_book C) (h2 : selling_price_one_book S) (h3 : cost_price_5_equals_selling_price_20 C S) : 
  ((C - S) / C) * 100 = 75 :=
by
  sorry

end loss_percentage_is_75_l236_236507


namespace find_city_with_at_most_one_outgoing_road_l236_236044

theorem find_city_with_at_most_one_outgoing_road (n : ℕ) (road_direction : fin n → fin n → option bool) : 
  ∃ m, m < 4 * n ∧ ∀ A, (∃ outgoing : list (fin n), outgoing.length ≤ 1 ∧ ∀ B ∈ outgoing, road_direction A B = some true) :=
sorry

end find_city_with_at_most_one_outgoing_road_l236_236044


namespace second_number_in_sequence_l236_236809

noncomputable def sequence (n : ℕ) : ℕ :=
  34 + n * 11

theorem second_number_in_sequence :
  sequence 0 = 34 ∧ sequence 1 = 45 ∧ sequence (sequence_length - 1) = 89 → sequence 1 = 45 :=
begin
  sorry
end

end second_number_in_sequence_l236_236809


namespace largest_value_l236_236585

def x1 : ℝ := 12345 + 1/5678
def x2 : ℝ := 12345 - 1/5678
def x3 : ℝ := 12345 * (1/5678)
def x4 : ℝ := 12345 * 5678
def x5 : ℝ := 12345.5678

theorem largest_value (i : ℕ) (h : i ∈ [1, 2, 3, 5]) : x4 > (list.nth [x1, x2, x3, x5] (i - 1)).get_or_else 0 :=
by
  sorry

end largest_value_l236_236585


namespace jason_text_messages_per_day_l236_236740

theorem jason_text_messages_per_day
  (monday_messages : ℕ)
  (tuesday_messages : ℕ)
  (total_messages : ℕ)
  (average_per_day : ℕ)
  (messages_wednesday_friday_per_day : ℕ) :
  monday_messages = 220 →
  tuesday_messages = monday_messages / 2 →
  average_per_day = 96 →
  total_messages = 5 * average_per_day →
  total_messages - (monday_messages + tuesday_messages) = 3 * messages_wednesday_friday_per_day →
  messages_wednesday_friday_per_day = 50 :=
by
  intros
  sorry

end jason_text_messages_per_day_l236_236740


namespace probability_not_all_same_l236_236924

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l236_236924


namespace digits_in_sequence_equals_zeros_in_next_sequence_l236_236017

def digits_count (n : ℕ) : ℕ :=
  (list.range (n + 1)).map (λ x => x.digits 10 ∑ (list.map (λ d => 1) d).sum).sum

def zeros_count (n : ℕ) : ℕ :=
  (list.range (n + 1)).map (λ x => x.digits 10 ∑ (list.map (λ d => if d = 0 then 1 else 0) d).sum).sum

theorem digits_in_sequence_equals_zeros_in_next_sequence (k : ℕ) :
  digits_count (10^k) = zeros_count (10^(k+1)) :=
sorry

end digits_in_sequence_equals_zeros_in_next_sequence_l236_236017


namespace trapezoid_circle_geometry_l236_236245

-- Definitions of the problem conditions
variables {A B C D E : Type} [EuclideanSpace Geometry]
variables {AD BC : A → B}
variables {CE ED : ℕ}
-- Simulate the isosceles trapezoid condition and parallel sides
axiom is_isosceles_trapezoid : ∀ (A B C D : Point), AD ∥ BC ∧ AD > BC

-- Simulate circle properties and point tangency
axiom circle_properties : ∀ (Ω : Circle) (P : Point), tangent Ω P

theorem trapezoid_circle_geometry
  (AD_parallel_BC : AD ∥ BC)
  (AD_gt_BC : AD > BC)
  (circle_tangency_CE : tangent Ω C)
  (CE_eq_9 : CE = 9)
  (ED_eq_16 : ED = 16) :
  (radius Ω = 15 / 2) ∧ (area_trapezoid = 675 / 2) :=
sorry

end trapezoid_circle_geometry_l236_236245


namespace golden_section_AC_length_l236_236655

theorem golden_section_AC_length
  (A B C : ℝ)
  (h1 : C = (√5 - 1) / 2 * B)
  (h2 : AB = 200)
  (h3 : AC > BC) :
  AC = 100 * (√5 - 1) := 
sorry

end golden_section_AC_length_l236_236655


namespace purple_candies_in_the_box_l236_236521

-- Definitions and conditions
variables (P Y G : ℕ)

noncomputable def question := P = 10

def condition1 := Y = P + 4
def condition2 := G = Y - 2
def condition3 := P + Y + G = 36

-- The theorem we need to prove
theorem purple_candies_in_the_box :
  (∃ P Y G, condition1 ∧ condition2 ∧ condition3) → question :=
begin
  intros h,
  sorry
end

end purple_candies_in_the_box_l236_236521


namespace largest_x_satisfies_eq_l236_236887

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236887


namespace EmilySpeed_l236_236594

/-- Define the conditions: Distance and Time --/
def Distance : ℝ := 10 -- miles
def Time : ℝ := 2 -- hours

/-- Define Speed as a quotient of Distance by Time --/
def Speed : ℝ := Distance / Time

/-- Theorem to prove --/
theorem EmilySpeed :
  Speed = 5 := by
  sorry

end EmilySpeed_l236_236594


namespace five_eight_sided_dice_not_all_same_l236_236895

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l236_236895


namespace largest_x_satisfying_equation_l236_236863

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l236_236863


namespace largest_x_satisfies_eq_l236_236865

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236865


namespace ways_to_turn_off_lights_l236_236839

/-- Number of ways to turn off 3 non-adjacent lights out of 8 lights in a corridor. -/
theorem ways_to_turn_off_lights (total_lights turned_off : ℕ) (h_total : total_lights = 8) (h_turned_off : turned_off = 3) (h_non_adjacent : 3 ≤ total_lights - turned_off) :
  nat.choose (total_lights - turned_off - 1) turned_off = 20 :=
by
  cases h_total, h_turned_off
  simp [nat.choose]
  sorry

end ways_to_turn_off_lights_l236_236839


namespace coefficient_x5_l236_236490

theorem coefficient_x5 :
  let poly1 := 2 * x^5 - 4 * x^4 + 3 * x^3 - x^2 + 2 * x - 1
  let poly2 := x^3 + 3 * x^2 - 2 * x + 4
  let product := poly1 * poly2
  (coeff product 5) = 24 :=
by
  sorry

end coefficient_x5_l236_236490


namespace chip_cost_l236_236159

noncomputable def bag_cost (price_total: ℝ) (quantity: ℕ) : ℝ :=
  price_total / quantity

theorem chip_cost (money: ℝ) (candy_cost: ℝ) (candy_ounces: ℕ) (chips_ounces: ℕ) (total_ounces: ℕ) :
  money = 7 → candy_cost = 1 → candy_ounces = 12 → chips_ounces = 17 → total_ounces = 85 →
  bag_cost 7 5 = 1.4 :=
by
  -- Definitions for the context
  assume h1: money = 7,
  assume h2: candy_cost = 1,
  assume h3: candy_ounces = 12,
  assume h4: chips_ounces = 17,
  assume h5: total_ounces = 85,
  -- The proof is omitted
  sorry

end chip_cost_l236_236159


namespace alice_probability_after_three_turns_l236_236155

-- Definitions of the probabilities given in the conditions
def p_Alice_to_Bob := 1 / 3
def p_Alice_to_Alice := 1 / 3
def p_Alice_keep := 1 / 3
def p_Bob_to_Alice := 3 / 5
def p_Bob_keep := 2 / 5

-- Defining the initial state that Alice starts with the ball
def initial_state := "Alice"

-- The target probability that Alice has the ball after three turns
def target_probability := 7 / 45

theorem alice_probability_after_three_turns :
  (probability Alice_has_ball_after_three_turns initial_state p_Alice_to_Bob p_Alice_to_Alice p_Alice_keep p_Bob_to_Alice p_Bob_keep) = target_probability := 
  sorry

end alice_probability_after_three_turns_l236_236155


namespace angle_DPO_eq_angle_CPO_l236_236759

theorem angle_DPO_eq_angle_CPO
  {O P A B C D M : Point}
  (h1 : Tangent PA (circle O A B))
  (h2 : Tangent PB (circle O A B))
  (h3 : LineIntersectsAt PO AB M)
  (h4 : ChordPassesThrough C D M) :
  ∠ D P O = ∠ C P O := sorry

end angle_DPO_eq_angle_CPO_l236_236759


namespace same_centroid_l236_236757

structure Point :=
  (x : ℝ) (y : ℝ)

structure Triangle :=
  (A B C : Point)

def trisection_point (P Q : Point) (ratio : ℝ) : Point :=
  { x := (P.x + ratio * Q.x) / (1 + ratio), y := (P.y + ratio * Q.y) / (1 + ratio) }

def centroid (T : Triangle) : Point :=
  { x := (T.A.x + T.B.x + T.C.x) / 3, y := (T.A.y + T.B.y + T.C.y) / 3 }

theorem same_centroid (ABC : Triangle) :
  let
    D := trisection_point ABC.B ABC.C (1/2),
    E := trisection_point ABC.C ABC.A (1/2),
    F := trisection_point ABC.A ABC.B (1/2),
    DEF := Triangle.mk D E F
  in
    centroid ABC = centroid DEF :=
begin
  sorry
end

end same_centroid_l236_236757


namespace remainder_valid_tilings_l236_236121

-- Define the board size
def board_length : ℕ := 8

-- Define the colors available
inductive Color
| red
| blue
| green

-- Define the function to count valid tilings using at least two different colors
noncomputable def count_valid_tilings : ℕ → ℕ
| n := 
  let partitions := 
    [ (7.choose 1 * (3 ^ 2 - 3)),   -- Two pieces: 3^2 - 3 valid colorings
      (7.choose 2 * (3 ^ 3 - 3)),   -- Three pieces: 3^3 - 3 valid colorings
      (7.choose 3 * (3 ^ 4 - 3)),   -- Four pieces: 3^4 - 3 valid colorings
      (7.choose 4 * (3 ^ 5 - 3)),   -- Five pieces: 3^5 - 3 valid colorings
      (7.choose 5 * (3 ^ 6 - 3)),   -- Six pieces: 3^6 - 3 valid colorings
      (7.choose 6 * (3 ^ 7 - 3)),   -- Seven pieces: 3^7 - 3 valid colorings
      (7.choose 7 * (3 ^ 8 - 3)) ]  -- Eight pieces: 3^8 - 3 valid colorings in the last piece
  in partitions.sum

-- Define the problem statement
theorem remainder_valid_tilings : count_valid_tilings board_length % 1000 = 768 :=
by
  sorry

end remainder_valid_tilings_l236_236121


namespace triangle_side_length_l236_236385

theorem triangle_side_length
  (a b c : ℝ)
  (area : a * c / 4 = sqrt 3)
  (angle_B : real.cos (real.pi / 3) = 1 / 2)
  (cond : a ^ 2 + c ^ 2 = 3 * a * c) :
  b = 2 * sqrt 2 := 
by
  sorry

end triangle_side_length_l236_236385


namespace general_term_of_sequence_l236_236634

open_locale big_operators

def sequence_an (a : ℕ → ℤ) := ∀ n : ℕ, a (n + 1) = a n + 3

theorem general_term_of_sequence 
  (a : ℕ → ℤ)
  (h1 : a 1 = 5)
  (h_seq : sequence_an a) :
  ∀ n : ℕ, a n = 3 * n + 2 :=
begin
  sorry
end

end general_term_of_sequence_l236_236634


namespace length_of_first_train_l236_236848

theorem length_of_first_train 
  (speed_first_train : ℝ) (speed_second_train : ℝ) 
  (time_clearance : ℝ) (length_second_train : ℝ) : 
  speed_first_train = 80 → speed_second_train = 65 → 
  time_clearance = 7.199424046076314 → length_second_train = 165 →
  let relative_speed := (speed_first_train + speed_second_train) * (1000 / 3600) in
  let total_distance := relative_speed * time_clearance in
  let length_first_train := total_distance - length_second_train in
  length_first_train = 125 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end length_of_first_train_l236_236848


namespace car_total_distance_l236_236129

theorem car_total_distance (g_total : ℕ) (h1 : g_total = 6) 
                          (d_hw1 : ℕ) (g_hw1 : ℕ) (h2 : d_hw1 = 120) (h3 : g_hw1 = 3) 
                          (d_city1 : ℕ) (g_city1 : ℕ) (h4 : d_city1 = 90) (h5 : g_city1 = 3) 
                          (g_hw2 : ℕ) (h6 : g_hw2 = 4) 
                          (g_city2 : ℕ) (h7 : g_city2 = 2) : 
                          let mpg_hw1 := d_hw1 / g_hw1,
                              mpg_city1 := d_city1 / g_city1,
                              d_hw2 := mpg_hw1 * g_hw2,
                              d_city2 := mpg_city1 * g_city2,
                              d_additional := d_hw2 + d_city2,
                              d_initial := d_hw1 + d_city1,
                              d_total := d_initial + d_additional in
                          d_total = 430 := 
by 
  let mpg_hw1 := 40
  let mpg_city1 := 30
  have h8 : mpg_hw1 = d_hw1 / g_hw1 := by exact rfl
  have h9 : mpg_city1 = d_city1 / g_city1 := by exact rfl
  let d_hw2 := mpg_hw1 * g_hw2
  let d_city2 := mpg_city1 * g_city2
  have h10 : d_hw2 = 160 := by exact rfl
  have h11 : d_city2 = 60 := by exact rfl
  let d_additional := d_hw2 + d_city2
  have h12 : d_additional = 220 := by exact rfl
  let d_initial := d_hw1 + d_city1
  have h13 : d_initial = 210 := by exact rfl
  let d_total := d_initial + d_additional
  have h14 : d_total = 430 := by exact rfl
  exact h14
  sorry

end car_total_distance_l236_236129


namespace cos_c_of_triangle_l236_236251

theorem cos_c_of_triangle {A B C : ℝ} (h1 : ∀ x, x^2 - 10*x + 6 = 0 → (x = tan A ∨ x = tan B)) 
                          (h2 : A + B + C = Real.pi) : cos C = sqrt(5)/5 := 
by 
  sorry

end cos_c_of_triangle_l236_236251


namespace problem_statement_l236_236748

variable (k : ℕ) (a : ℕ → ℝ)

theorem problem_statement (h₀ : k ≥ 2)
    (h₁ : ∀ n, 1 ≤ n ∧ n ≤ 2021 → a n ≥ 0)
    (h₂ : ∀ n, 1 ≤ n ∧ n < 2021 → a n ≥ a (n + 1))
    (h₃ : ∀ n, 1 ≤ n ∧ n ≤ 2021 → ∑ i in finset.range (2022 - n) + n, a i ≤ k * a n) :
    a 2021 ≤ 4 * (1 - 1 / k) ^ 2021 * a 1 := by
  sorry

end problem_statement_l236_236748


namespace light_ray_exits_angle_l236_236277

theorem light_ray_exits_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∃ (d : ℝ), d > 0 ∧ ∀ t > d, ray_exit_angle a b t :=
sorry

-- Definition which represents the concept of the light ray exiting the angle
def ray_exit_angle (a b t : ℝ) : Prop :=
∀ (x : ℝ), x = t → (x > a ∨ x > b)

end light_ray_exits_angle_l236_236277


namespace max_connected_nodes_l236_236350

theorem max_connected_nodes (N : ℕ) (hN : N > 3) 
  (h : ∃ n : ℕ, n < N ∧ ∀ (i : ℕ), i ≠ n → ¬ (n ~ i)) : 
  ∃ m : ℕ, m = N - 1 :=
by
  sorry

end max_connected_nodes_l236_236350


namespace degree_sum_of_polynomials_l236_236754

noncomputable def f (a_3 a_2 a_1 a_0 : ℂ) (z : ℂ) : ℂ :=
  a_3 * z^3 + a_2 * z^2 + a_1 * z + a_0

noncomputable def g (b_2 b_1 b_0 : ℂ) (z : ℂ) : ℂ :=
  b_2 * z^2 + b_1 * z + b_0

theorem degree_sum_of_polynomials 
  (a_3 a_2 a_1 a_0 b_2 b_1 b_0 : ℂ)
  (ha3 : a_3 ≠ 0) :
  ∃ z : ℂ, degree (f a_3 a_2 a_1 a_0 z + g b_2 b_1 b_0 z) = 3 := 
sorry

end degree_sum_of_polynomials_l236_236754


namespace find_beta_l236_236653

theorem find_beta
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcos_α : real.cos α = 1 / 7)
  (hcos_αβ : real.cos (α + β) = -11 / 14) :
  β = π / 3 :=
sorry

end find_beta_l236_236653


namespace ratio_simplified_l236_236777

theorem ratio_simplified {total_students bleachers_students : ℕ} (h_total : total_students = 26) (h_bleachers : bleachers_students = 4) :
  let floor_students := total_students - bleachers_students in
  (floor_students : ℚ) / total_students = 11 / 13 :=
by 
  sorry

end ratio_simplified_l236_236777


namespace david_avg_monthly_balance_l236_236582

theorem david_avg_monthly_balance : 
  let jan := 150
      feb := 200
      mar := 250
      apr := 250
      may := 200
      jun := 300
      total_sum := jan + feb + mar + apr + may + jun
      num_months := 6 in 
  (total_sum / num_months) = 225 :=
by
  let jan := 150
  let feb := 200
  let mar := 250
  let apr := 250
  let may := 200
  let jun := 300
  let total_sum := jan + feb + mar + apr + may + jun
  let num_months := 6
  have h : (total_sum / num_months) = 225 := sorry
  exact h

end david_avg_monthly_balance_l236_236582


namespace number_of_buses_required_l236_236471

def total_seats : ℕ := 28
def students_per_bus : ℝ := 14.0

theorem number_of_buses_required :
  (total_seats / students_per_bus) = 2 := 
by
  -- The actual proof is intentionally left out.
  sorry

end number_of_buses_required_l236_236471


namespace find_x_given_y64_l236_236835

variable (x y k : ℝ)

def inversely_proportional (x y : ℝ) := (x^3 * y = k)

theorem find_x_given_y64
  (h_pos : x > 0 ∧ y > 0)
  (h_inversely : inversely_proportional x y)
  (h_given : inversely_proportional 2 8)
  (h_y64 : y = 64) :
  x = 1 := by
  sorry

end find_x_given_y64_l236_236835


namespace elizabeth_wedding_gift_cost_l236_236194

-- Defining the given conditions
def cost_steak_knife_set : ℝ := 80.00
def num_steak_knife_sets : ℝ := 2
def cost_dinnerware_set : ℝ := 200.00
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Calculating total expense
def total_cost (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set : ℝ) : ℝ :=
  (cost_steak_knife_set * num_steak_knife_sets) + cost_dinnerware_set

def discounted_price (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (total_cost * discount_rate)

def final_price (discounted_price sales_tax_rate : ℝ) : ℝ :=
  discounted_price + (discounted_price * sales_tax_rate)

def elizabeth_spends (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate : ℝ) : ℝ :=
  final_price (discounted_price (total_cost cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set) discount_rate) sales_tax_rate

theorem elizabeth_wedding_gift_cost
  (cost_steak_knife_set : ℝ)
  (num_steak_knife_sets : ℝ)
  (cost_dinnerware_set : ℝ)
  (discount_rate : ℝ)
  (sales_tax_rate : ℝ) :
  elizabeth_spends cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate = 340.20 := 
by
  sorry -- Proof is to be completed

end elizabeth_wedding_gift_cost_l236_236194


namespace minimize_sum_distances_l236_236033

theorem minimize_sum_distances (k : ℚ) :
  let A := (3, 4) : ℚ × ℚ,
      B := (6, 2) : ℚ × ℚ,
      C := (-2, k) : ℚ × ℚ in
  k = 34 / 9 ↔
  ∀ k', (dist A C + dist B C) ≥ (dist A (−2, 34 / 9) + dist B (−2, 34 / 9)) := by
    sorry

end minimize_sum_distances_l236_236033


namespace opposite_of_neg_half_l236_236464

-- Define the opposite of a number
def opposite (x : ℝ) : ℝ := -x

-- The theorem we want to prove
theorem opposite_of_neg_half : opposite (-1/2) = 1/2 :=
by
  -- Proof goes here
  sorry

end opposite_of_neg_half_l236_236464


namespace jessica_older_than_claire_l236_236741

-- Define the current age of Claire
def claire_current_age := 20 - 2

-- Define the current age of Jessica
def jessica_current_age := 24

-- Prove that Jessica is 6 years older than Claire
theorem jessica_older_than_claire : jessica_current_age - claire_current_age = 6 :=
by
  -- Definitions of the ages
  let claire_current_age := 18
  let jessica_current_age := 24

  -- Prove the age difference
  sorry

end jessica_older_than_claire_l236_236741


namespace assignment_students_groups_proof_l236_236716

/-
  Given four students and three interest groups: Chinese, Mathematics, and English, 
  prove that the number of ways to assign the students such that every interest group 
  has at least one student follows the conditions defined by the provided combinations 
  and permutations.
-/

def num_ways_assign_students_each_group_has_one (students groups : ℕ) (at_least_one : ∀ g : ℕ, g < groups → g ≥ 1) : Prop :=
  ∃ (assignments : ℕ), assignments = (Nat.C 4 2 * Nat.A 3 3) ∧ assignments = (Nat.C 3 1 * Nat.C 4 2 * Nat.A 2 2)

theorem assignment_students_groups_proof : 
  num_ways_assign_students_each_group_has_one 4 3 (by { intro g, intro hg, exact Nat.le_of_lt hg }) :=
  sorry

end assignment_students_groups_proof_l236_236716


namespace num_mappings_A_to_A_num_bijective_mappings_A_to_A_l236_236625

-- Defining the set A
def A : Set := {a, b, c, d}

-- Theorem stating the number of functions from A to A is 256
theorem num_mappings_A_to_A : ∃ n, n = 256 ∧ (∀ f : A → A, true) := sorry

-- Theorem stating the number of bijections from A to A is 24
theorem num_bijective_mappings_A_to_A : ∃ n, n = 24 ∧ (∀ f : A → A, Function.Bijective f → true) := sorry

end num_mappings_A_to_A_num_bijective_mappings_A_to_A_l236_236625


namespace sequence_general_term_l236_236674

theorem sequence_general_term (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 5)
  (h4 : a 4 = 7) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l236_236674


namespace yeri_change_l236_236497

theorem yeri_change :
  let cost_candies := 5 * 120
  let cost_chocolates := 3 * 350
  let total_cost := cost_candies + cost_chocolates
  let amount_handed_over := 2500
  amount_handed_over - total_cost = 850 :=
by
  sorry

end yeri_change_l236_236497


namespace four_digit_numbers_with_property_l236_236309

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l236_236309


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236931

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236931


namespace count_valid_numbers_l236_236315
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l236_236315


namespace suff_not_necessary_perpendicular_lines_l236_236511

theorem suff_not_necessary_perpendicular_lines (a : ℝ) :
  (a = 3 → (a ≠ -2 → ∃ m1 m2 : ℝ, 
  (∃ b1 b2 : ℝ, ax - 2 * y + 3 * a = m1 * x + b1 ∧ (a - 1) * x + 3 * y + a² - a + 3 = m2 * x + b2) 
  → m1 * m2 = -1)) ∧ 
  (∃ a1 : ℝ, a1 ≠ 3 ∧ (∃ m1 m2 : ℝ,  
  (∃ b1 b2 : ℝ, a1x - 2 * y + 3 * a1 = m1 * x + b1 ∧ (a1 - 1) * x + 3 * y + a1² - a1 + 3 = m2 * x + b2) 
  → m1 * m2 = -1)) :=
by
  sorry

end suff_not_necessary_perpendicular_lines_l236_236511


namespace line_outside_plane_l236_236495

-- Definitions of the conditions
variable (Point Line Plane : Type)
variable (contains_point : Plane → Point → Prop)
variable (on_line : Line → Point → Prop)
variable (intersects : Line → Plane → Prop)

-- Definition of at most one common point
def at_most_one_common_point (l : Line) (p : Plane) : Prop :=
  ∀ (x y : Point), on_line l x → contains_point p x → on_line l y → contains_point p y → x = y

-- Proof statement
theorem line_outside_plane (l : Line) (p : Plane) :
  (at_most_one_common_point l p) → (∃ (x : Point), ¬ contains_point p x) :=
sorry

end line_outside_plane_l236_236495


namespace sum_of_coefficients_is_8_l236_236766

noncomputable def sequence (u : ℕ → ℕ) : Prop :=
  u 1 = 8 ∧ ∀ n, u (n + 1) - u n = 5 + 2 * (n - 1)

theorem sum_of_coefficients_is_8 (u : ℕ → ℕ) (h : sequence u) : 
  ∃ a b c : ℕ, (∀ n, u n = a * n ^ 2 + b * n + c) ∧ a + b + c = 8 :=
by 
  sorry

end sum_of_coefficients_is_8_l236_236766


namespace prove_f2_value_l236_236217

noncomputable def f : ℝ → ℝ := sorry

theorem prove_f2_value :
  (∀ x : ℝ, x ≠ 0 → f(x) + 3 * f(1 / x) = x^2) →
  f(2) = -13 / 32 :=
by
  intro h
  sorry

end prove_f2_value_l236_236217


namespace find_breadth_l236_236441

-- Define the conditions
def rectangular_plot (breadth : ℝ) (length : ℝ) : Prop :=
  length = 0.75 * breadth ∧ length * breadth = 360

-- State the problem
theorem find_breadth (breadth : ℝ) (length : ℝ) (h : rectangular_plot breadth length) : breadth ≈ 21.91 :=
by 
  sorry

end find_breadth_l236_236441


namespace reciprocal_of_neg3_l236_236043

theorem reciprocal_of_neg3 : 1 / (-3: ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l236_236043


namespace minimum_value_fraction_l236_236257

theorem minimum_value_fraction (m n : ℝ) (h_line : 2 * m * 2 + n * 2 - 4 = 0) (h_pos_m : m > 0) (h_pos_n : n > 0) :
  (m + n / 2 = 1) -> ∃ (m n : ℝ), (m > 0 ∧ n > 0) ∧ (3 + 2 * Real.sqrt 2 ≤ (1 / m + 4 / n)) :=
by
  sorry

end minimum_value_fraction_l236_236257


namespace paraplex_line_intersection_l236_236620

theorem paraplex_line_intersection :
  let parabola_vertex (b : ℝ) := (0 : ℝ, 2 * b ^ 2)
  let line_eq (x b : ℝ) := x + b
  ∃ bs : Finset ℝ, bs.card = 2 ∧ ∀ b ∈ bs, line_eq 0 b = parabola_vertex b.2 :=
sorry

end paraplex_line_intersection_l236_236620


namespace find_f_of_3_l236_236648

open Real

noncomputable def fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : ℝ × ℝ :=
  let x := sqrt 2
  let y := 2
  (x, y)

noncomputable def power_function (x : ℝ) : ℝ :=
  x^2

theorem find_f_of_3 (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  let P := fixed_point a h₀ h₁ in
  (1 > 0 ∧ a > 0 ∧ a ≠ 1) → 
  P.1 = sqrt 2 ∧ P.2 = 2 →
  let f := power_function in
  f(3) = 9 :=
by
  intros
  sorry

end find_f_of_3_l236_236648


namespace num_valid_mappings_count_l236_236649

def M : Set ℕ := {a, b, c, d}
def N : Set ℕ := {0, 1, 2}

noncomputable def f : ℕ → ℕ := sorry

def valid_mappings (f : ℕ → ℕ) : Prop :=
  f a + f b + f c + f d = 4

theorem num_valid_mappings_count : ∃ (f : ℕ → ℕ), Prod (valid_mappings f) = 19 :=
by
  sorry

end num_valid_mappings_count_l236_236649


namespace cannot_form_right_triangle_l236_236162

theorem cannot_form_right_triangle (a b c : ℕ) (h : a ≤ b ∧ b ≤ c) :
  ¬ (a^2 + b^2 = c^2) :=
by
  -- Input the specific values for the conditions
  have ha4 : a = 4 := sorry
  have hb6 : b = 6 := sorry
  have hc8 : c = 8 := sorry
  -- Show that a^2 + b^2 ≠ c^2 for these values
  calc
    a^2 + b^2 = 4*4 + 6*6 := by rw [ha4, hb6]
            ... = 16 + 36 := by norm_num
            ... = 52     := by norm_num
    _ = c^2   := by rw hc8
    _ = 8*8   := rfl
    _ = 64    := by norm_num
  have hneq : 52 ≠ 64 := by norm_num
  exact hneq

end cannot_form_right_triangle_l236_236162


namespace flower_bouquet_total_length_l236_236800

theorem flower_bouquet_total_length :
  let n := 50 in
  let person_space := 0.4 in
  let gap_space := 0.5 in
  let num_gaps := n - 1 in
  let total_gap_length := num_gaps * gap_space in
  let total_person_space := n * person_space in
  total_gap_length + total_person_space = 44.5 :=
by
  sorry

end flower_bouquet_total_length_l236_236800


namespace complex_number_in_first_quadrant_l236_236631

noncomputable def z (x : ℂ) : ℂ := 2 * complex.I * complex.sqrt 3 / (1 + complex.sqrt 3 * complex.I)

theorem complex_number_in_first_quadrant :
  let z := 2 * complex.I * complex.sqrt 3 / (1 + complex.sqrt 3 * complex.I)
  z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_number_in_first_quadrant_l236_236631


namespace triangle_altitude_segment_length_l236_236151

noncomputable theory

-- Definitions
def triangle_sides (a b c : ℕ) : Prop := (a = 30 ∧ b = 70 ∧ c = 80)

-- The theorem to prove
theorem triangle_altitude_segment_length {a b c d : ℕ} 
  (h_sides : triangle_sides a b c) : 
  d = 65 
:= sorry

end triangle_altitude_segment_length_l236_236151


namespace nine_by_nine_grid_equal_columns_impossible_l236_236591

theorem nine_by_nine_grid_equal_columns_impossible :
  ∀ (grid : Matrix (Fin 9) (Fin 9) ℕ), 
    (∀ i j, grid i j = 0) →
    (∀ row : Fin 9, ∀ k : ℕ, 
      k > 0 → 
      ∃ m n : Fin 9, 
      n = m + 1 ∧ 
      ∃ p : ℕ, 
      (∀ r : Fin 9, grid (row : Fin 9) r = 
        if r = m 
        then grid row m + p 
        else if r = n 
        then grid row n + p 
        else grid row r
      )) →
    ∃ S : ℕ, (∀ j : Fin 9, 
      S = ∑ i : Fin 9, grid i j) →
    ∀ i : Fin 9, S > 0 →
    False
:= 
begin
  sorry
end

end nine_by_nine_grid_equal_columns_impossible_l236_236591


namespace max_points_on_circle_five_cm_l236_236421

-- Defining the circle and point with their properties
structure Point := (x : ℝ) (y : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Axiom stating that the point is outside the circle
axiom point_outside_circle {P : Point} {C : Circle} (h : (P.x - C.center.x)^2 + (P.y - C.center.y)^2 > C.radius^2)

-- Function to check if a point is exactly 5 cm from another point
def is_five_cm_from (A B : Point) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = 25

-- The theorem to prove
theorem max_points_on_circle_five_cm (P : Point) (C : Circle) (h : point_outside_circle P C) :
  ∃ p1 p2 : Point, is_five_cm_from p1 P ∧ is_five_cm_from p2 P ∧ p1 ≠ p2 ∧ p1 ∈ C ∧ p2 ∈ C :=
sorry

end max_points_on_circle_five_cm_l236_236421


namespace three_pow_2040_mod_5_l236_236090

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l236_236090


namespace find_b_of_parabola_axis_of_symmetry_l236_236805

theorem find_b_of_parabola_axis_of_symmetry (b : ℝ) :
  (∀ (x : ℝ), (x = 1) ↔ (x = - (b / (2 * 2))) ) → b = 4 :=
by
  intro h
  sorry

end find_b_of_parabola_axis_of_symmetry_l236_236805


namespace find_f_inv_128_l236_236691

noncomputable def f : ℕ → ℕ := sorry

axiom f_at_5 : f 5 = 2
axiom f_doubling : ∀ x : ℕ, f (2 * x) = 2 * f x

theorem find_f_inv_128 : f 320 = 128 :=
by sorry

end find_f_inv_128_l236_236691


namespace main_theorem_l236_236661

variables {n : ℕ} {a : Fin (2*n) → Fin (2*n) → ℝ} {δ : ℝ}

def tangency_condition (a : Fin (2 * n) → Fin (2 * n) → ℝ) (δ : ℝ) : Prop :=
  (∏ i in Finset.range n, a (⟨2*i + 1, sorry⟩ : Fin (2*n)) (⟨2*i + 2 % (2*n), sorry⟩ : Fin (2*n))) = δ

theorem main_theorem (hn : 0 < n) (H : tangency_condition a δ) :
  ∃ i ∈ Finset.range n, a (⟨2*i, sorry⟩ : Fin (2*n)) (⟨2*i + 1 % (2*n), sorry⟩ : Fin (2*n)) ≥ real.pow δ (1/n) ∧
  ∃ j ∈ Finset.range n, a (⟨2*j, sorry⟩ : Fin (2*n)) (⟨2*j + 1 % (2*n), sorry⟩ : Fin (2*n)) ≤ real.pow δ (1/n) :=
sorry

end main_theorem_l236_236661


namespace solve_problem_l236_236452

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 - x - 3

theorem solve_problem : f (1 / f 3) = 8 / 9 := by
  sorry

end solve_problem_l236_236452


namespace bob_cleaning_time_l236_236737

theorem bob_cleaning_time :
  let carol_time := 30 in
  let bob_fraction_of_carol := 1 / 6 in
  let bob_time := bob_fraction_of_carol * carol_time in
  bob_time = 5 :=
by
  let carol_time := 30
  let bob_fraction_of_carol := 1 / 6
  let bob_time := bob_fraction_of_carol * carol_time
  have h : bob_time = 5 := by sorry
  exact h

end bob_cleaning_time_l236_236737


namespace probability_of_not_all_same_number_l236_236906

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l236_236906


namespace exists_divalent_radical_and_bound_l236_236746

-- Define the conditions as hypotheses
def is_divalent_radical (A : set ℕ) : Prop :=
  ∀ k, (∃ a b ∈ A, a + b = k)

def satisfies_bound (A : set ℕ) (C : ℝ) : Prop :=
  ∀ x : ℕ, x ≥ 1 → (A ∩ set.Iic x).finite ∧ ((A ∩ set.Iic x).to_finset.card : ℝ) ≤ C * real.sqrt x

-- Declare the main theorem to be proven
theorem exists_divalent_radical_and_bound :
  ∃ (A : set ℕ) (C : ℝ), A.nonempty ∧ is_divalent_radical A ∧ C > 0 ∧ satisfies_bound A C :=
begin
  sorry
end

end exists_divalent_radical_and_bound_l236_236746


namespace num_elements_in_M_l236_236275

open Set

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {4, 5}

def M : Set ℕ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a + b}

theorem num_elements_in_M : M.to_finset.card = 4 :=
  by
  sorry

end num_elements_in_M_l236_236275


namespace smallest_positive_integer_divisible_by_14_15_16_l236_236212

theorem smallest_positive_integer_divisible_by_14_15_16 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (14 ∣ m) ∧ (15 ∣ m) ∧ (16 ∣ m) → n ≤ m) :=
  ∃ n : ℕ, n = 1680 ∧ ∀ m : ℕ, (14 ∣ m) ∧ (15 ∣ m) ∧ (16 ∣ m) ∧ m > 0 → n ≤ m

end smallest_positive_integer_divisible_by_14_15_16_l236_236212


namespace quadrant_rotation_l236_236334

def in_fourth_quadrant (θ : ℝ) : Prop :=
  3 * π / 2 < θ ∧ θ < 2 * π

def in_first_quadrant (φ : ℝ) : Prop :=
  0 < φ ∧ φ < π / 2

theorem quadrant_rotation (θ : ℝ) (h : in_fourth_quadrant θ) : in_first_quadrant (π / 2 + θ) :=
by
  sorry

end quadrant_rotation_l236_236334


namespace largest_x_satisfies_eq_l236_236883

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236883


namespace find_cost_prices_l236_236785

noncomputable def dining_set_cost := 7576
noncomputable def chandelier_cost := 7500
noncomputable def sofa_set_cost := 11429

theorem find_cost_prices (
  h_dining_set: ∀ D, 0.82 * D + 2500 = 1.15 * D -> D = dining_set_cost,
  h_chandelier: ∀ C, 1.20 * C - 3000 = 0.80 * C -> C = chandelier_cost,
  h_sofa_set: ∀ S, 0.90 * S + 4000 = 1.25 * S -> S = sofa_set_cost
) : 
  dining_set_cost = 7576 ∧ 
  chandelier_cost = 7500 ∧ 
  sofa_set_cost = 11429 := 
by
  split;
  sorry

end find_cost_prices_l236_236785


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236855

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236855


namespace determine_a_l236_236189

def quadratic_condition (a : ℝ) (x : ℝ) : Prop := 
  abs (x^2 + 2 * a * x + 3 * a) ≤ 2

theorem determine_a : {a : ℝ | ∃! x : ℝ, quadratic_condition a x} = {1, 2} :=
sorry

end determine_a_l236_236189


namespace julias_change_l236_236349

theorem julias_change :
  let snickers := 2
  let mms := 3
  let cost_snickers := 1.5
  let cost_mms := 2 * cost_snickers
  let money_given := 2 * 10
  let total_cost := snickers * cost_snickers + mms * cost_mms
  let change := money_given - total_cost
  change = 8 :=
by
  sorry

end julias_change_l236_236349


namespace degrees_to_radians_l236_236186

theorem degrees_to_radians (d : ℝ) (hd : d = -885) : 
  let radians := d * (π / 180) in radians = - (59 / 12) * π :=
by
  simp [hd]
  sorry

end degrees_to_radians_l236_236186


namespace shaded_region_area_l236_236343

theorem shaded_region_area
  (pi : ℝ := 3)
  (r : ℝ := 4)
  (side_length : ℝ := 1)
  (grid_width : ℕ := 10)
  (grid_height : ℕ := 12)
  (a_b_combined_area : ℝ := 4.5)
  (c_d_combined_area : ℝ := 12)
  (rectangle_area : ℝ := 10)
  (largest_circle_area : ℝ := 48)
  :
  let shaded_area := largest_circle_area - a_b_combined_area - c_d_combined_area - rectangle_area in
  shaded_area = 21.5 :=
by
  sorry

end shaded_region_area_l236_236343


namespace divisors_ending_in_3_at_most_half_l236_236616

theorem divisors_ending_in_3_at_most_half (n : ℕ) (h : 0 < n):
  let S_n := {d : ℕ // d ∣ n}
  in (2 * (S_n.filter (λ d => (d.val % 10 = 3))).card ≤ S_n.card) :=
by sorry

end divisors_ending_in_3_at_most_half_l236_236616


namespace eccentricity_ellipse_eq_l236_236629

variables {a b : ℝ}
variables (h1 : a > b) (h2 : b > 0)
def ellipse : Prop := ∃ x y, (x^2 / a^2 + y^2 / b^2 = 1)
def hyperbola : Prop := ∃ x y, (x^2 / a^2 - y^2 / b^2 = 1)

theorem eccentricity_ellipse_eq (h3 : a ^ 2 = 2 * b ^ 2)
  (h4 : ∀ e1 e2 : ℝ, (e1 = sqrt ((a ^ 2 - b ^ 2) / a ^ 2)) ∧ (e2 = sqrt ((a ^ 2 + b ^ 2) / a ^ 2)) ∧ (e1 * e2 = sqrt 3 / 2)) :
  sqrt (1 - b^2 / a^2) = sqrt 2 / 2 :=
by sorry

end eccentricity_ellipse_eq_l236_236629


namespace complex_number_multiplication_l236_236252

theorem complex_number_multiplication (i : ℂ) (hi : i * i = -1) : i * (1 + i) = -1 + i :=
by sorry

end complex_number_multiplication_l236_236252


namespace find_x_l236_236327

noncomputable def x_value (x z : ℝ) : Prop :=
  x ≠ 0 ∧ (x / 3 = z^2 + 1) ∧ (x / 5 = 5z + 2)

theorem find_x (x z : ℝ) (h : x_value x z) :
  x = (685 + 25 * Real.sqrt 541) / 6 :=
sorry

end find_x_l236_236327


namespace sequence_a_4_is_5_over_3_l236_236366

noncomputable def sequence : ℕ → ℚ
| 1     := 1
| (n+1) := 1 + 1 / sequence n

theorem sequence_a_4_is_5_over_3 : sequence 4 = 5/3 := by
  sorry

end sequence_a_4_is_5_over_3_l236_236366


namespace travel_time_difference_l236_236193

theorem travel_time_difference (distance : ℝ) (speed1 speed2 : ℝ) :
  distance = 6 → speed1 = 55 → speed2 = 35 → (60 * (distance / speed2 - distance / speed1)) ≈ 3.74 :=
by
  intros h_distance h_speed1 h_speed2
  rw [h_distance, h_speed1, h_speed2]
  sorry

end travel_time_difference_l236_236193


namespace first_candidate_fails_by_60_marks_l236_236128

theorem first_candidate_fails_by_60_marks (T P F : ℝ) (h1 : 0.30 * T = P - F) (h2 : 0.45 * T = P + 30) (h3 : P ≈ 240) : F = 60 := by
  sorry

end first_candidate_fails_by_60_marks_l236_236128


namespace Maria_Ivanovna_solution_l236_236407

noncomputable def Maria_grades_problem : Prop :=
  let a : ℕ → ℕ := λ n, if n = 1 then 3
                        else if n = 2 then 8
                        else 2 * a (n - 1) + 2 * a (n - 2) in
  a 6 = 448

theorem Maria_Ivanovna_solution : Maria_grades_problem := by
  sorry

end Maria_Ivanovna_solution_l236_236407


namespace remainder_is_correct_l236_236210

def p (x : ℝ) := x^6 - 3*x^5 + 3*x^3 - x^2 - 2*x
def d (x : ℝ) := (x^2 - 1) * (x - 2)
def r (x : ℝ) := - (16/3) * x^2 + 2 * x + (4/3)

theorem remainder_is_correct : 
  ∀ x : ℝ, p(x) % d(x) = r(x) := 
by sorry

end remainder_is_correct_l236_236210


namespace solve_cos_eq_l236_236435

theorem solve_cos_eq :
  ∃ k : ℤ, ∀ x : ℝ,
    x = (π / 2) + 2 * k * π ∨
    x = (3 * π / 2) + 2 * k * π ∨
    x = (π / 5) + (2 * k * π / 5) ∨
    x = (3 * π / 5) + (2 * k * π / 5) ∨
    x = π + (2 * k * π / 5) ∨
    x = (7 * π / 5) + (2 * k * π / 5) ∨
    x = (9 * π / 5) + (2 * k * π / 5) ↔
    cos x + cos (2 * x) + cos (3 * x) + cos (4 * x) = 0 :=
begin
  sorry
end

end solve_cos_eq_l236_236435


namespace problem_solution_l236_236395

noncomputable theory

variables {a k m : ℝ}

-- Conditions from the problem
def f (x : ℝ) : ℝ := k * a^x - a^(-x)
def g (x : ℝ) : ℝ := a^(2 * x) - a^(-2 * x) - 2 * m * f(x)

-- Main statement to prove
theorem problem_solution (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f 1 = 8 / 3) (h₃ : ∀ x ∈ set.Ici (1 : ℝ), g(x) ≥ -2) :
  k = 1 ∧ m = 25 / 12 :=
by
  sorry

end problem_solution_l236_236395


namespace question_mark_value_l236_236116

theorem question_mark_value :
  ∀ (x : ℕ), ( ( (5568: ℝ) / (x: ℝ) )^(1/3: ℝ) + ( (72: ℝ) * (2: ℝ) )^(1/2: ℝ) = (256: ℝ)^(1/2: ℝ) ) → x = 87 :=
by
  intro x
  intro h
  sorry

end question_mark_value_l236_236116


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236858

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236858


namespace max_integer_a_real_roots_l236_236225

theorem max_integer_a_real_roots :
  ∀ (a : ℤ), (∃ (x : ℝ), (a + 1 : ℝ) * x^2 - 2 * x + 3 = 0) → a ≤ -2 :=
by
  sorry

end max_integer_a_real_roots_l236_236225


namespace count_valid_numbers_l236_236314
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l236_236314


namespace largest_x_value_satisfies_largest_x_value_l236_236876

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l236_236876


namespace car_trip_distance_l236_236130

noncomputable def total_distance_of_trip (original_speed : ℝ) (delayed_speed_factor : ℝ) (duration_before_accident : ℝ)
  (delay_after_accident : ℝ) (total_delay : ℝ) (extra_distance_if_delayed : ℝ) 
  (total_distance : ℝ) : Prop :=  
  let remaining_distance := total_distance - (original_speed * duration_before_accident)
  let reduced_speed := original_speed * delayed_speed_factor
  let time_after_accident := remaining_distance / reduced_speed
  let total_time_first_scenario := duration_before_accident + delay_after_accident + time_after_accident

  let new_distance_before_accident := (original_speed * duration_before_accident) + extra_distance_if_delayed
  let new_remaining_distance := total_distance - new_distance_before_accident
  let new_time_after_accident := new_remaining_distance / reduced_speed
  let total_time_second_scenario := (new_distance_before_accident / original_speed) + delay_after_accident + new_time_after_accident

  (total_time_first_scenario - total_delay) = duration_before_accident + (remaining_distance / original_speed) ∧
  (total_time_second_scenario - (total_delay - 0.5)) = (new_distance_before_accident / original_speed) + (new_remaining_distance / reduced_speed)

theorem car_trip_distance : total_distance_of_trip 40 (5/6) 2 0.33 2 60 280 := 
begin 
  sorry 
end

end car_trip_distance_l236_236130


namespace find_matrix_M_l236_236206

theorem find_matrix_M :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℚ),
    M.mul (Matrix.of '[[2, -3, 0, 1], [-4, 6, 0, -2], [0, 0, 1, 0], [1, -1.5, 0, 0.5]]) = 1 := sorry

end find_matrix_M_l236_236206


namespace simplify_sqrt_5_4_simplify_sqrt_n_n1_series_calculation_l236_236613

-- Problem 1
theorem simplify_sqrt_5_4 : 1 / (Real.sqrt 5 + Real.sqrt 4) = Real.sqrt 5 - 2 :=
by sorry

-- Problem 2
theorem simplify_sqrt_n_n1 (n : ℕ) (h : 0 < n) : 1 / (Real.sqrt (n + 1) + Real.sqrt n) = Real.sqrt (n + 1) - Real.sqrt n :=
by sorry

-- Problem 3
theorem series_calculation : (∑ k in Finset.range 2022, 1 / (Real.sqrt (k + 2) + Real.sqrt (k + 1))) * (Real.sqrt 2023 + 1) = 2022 :=
by sorry

end simplify_sqrt_5_4_simplify_sqrt_n_n1_series_calculation_l236_236613


namespace incorrect_statement_A_l236_236982

/-- Let prob_beijing be the probability of rainfall in Beijing and prob_shanghai be the probability
of rainfall in Shanghai. We assert that statement (A) which claims "It is certain to rain in Beijing today, 
while it is certain not to rain in Shanghai" is incorrect given the probabilities. 
-/
theorem incorrect_statement_A (prob_beijing prob_shanghai : ℝ) 
  (h_beijing : prob_beijing = 0.8)
  (h_shanghai : prob_shanghai = 0.2)
  (statement_A : ¬ (prob_beijing = 1 ∧ prob_shanghai = 0)) : 
  true := 
sorry

end incorrect_statement_A_l236_236982


namespace mutually_exclusive_but_not_opposite_l236_236192

theorem mutually_exclusive_but_not_opposite
  (cards : Type) [fintype cards] [decidable_eq cards] (hearts spades diamonds clubs : cards)
  (people : Type) [fintype people] [decidable_eq people] (A B C D : people)
  (distribution : people → cards) :
  (distribution A = clubs ∧ distribution B ≠ clubs) ∨ (distribution B = clubs ∧ distribution A ≠ clubs) ∨ (distribution A ≠ clubs ∧ distribution B ≠ clubs) :=
begin
  sorry,
end

end mutually_exclusive_but_not_opposite_l236_236192


namespace even_numbers_count_l236_236135

def digits : Set ℕ := {0, 1, 2, 3, 4, 5}

theorem even_numbers_count : 
  (∃ (even_numbers : Finset (Fin 5 → ℕ)), 
    even_numbers.card = 240 ∧ 
    ∀ num ∈ even_numbers, 
      (num 0 ≠ 0 ∧ 
       (num 4 = 0 ∨ num 4 = 2 ∨ num 4 = 4) ∧ 
       list.nodup (list.of_fn num) ∧ 
       ∀ (i : Fin 5), num i ∈ digits ∧ 
       ((num 0 : ℕ) < 5) ∧ 
       (∀ (i j : Fin 5) num_i num_j, i ≠ j → num i ≠ num j))
  ) :=
begin
  sorry
end

end even_numbers_count_l236_236135


namespace sqrt_3x_eq_5x_largest_value_l236_236880

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l236_236880


namespace sum_of_consecutive_integers_420_l236_236325

theorem sum_of_consecutive_integers_420 : 
  ∃ (k n : ℕ) (h1 : k ≥ 2) (h2 : k * n + k * (k - 1) / 2 = 420), 
  ∃ K : Finset ℕ, K.card = 6 ∧ (∀ x ∈ K, k = x) :=
by
  sorry

end sum_of_consecutive_integers_420_l236_236325


namespace distance_to_focus_l236_236337

open Real

-- Define the parabola y^2 = -4x
def parabola := { P : ℝ × ℝ | P.2^2 = -4 * P.1 }

-- Define the focus of the parabola y^2 = -4x
def focus : ℝ × ℝ := (-1, 0)

-- Define the distance function
def dist (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Main theorem statement
theorem distance_to_focus {P : ℝ × ℝ} (hP : P ∈ parabola) (dist_y_axis : P.1^2 = 25) :
  dist P focus = 6 := 
sorry

end distance_to_focus_l236_236337


namespace num_positive_x_count_num_positive_x_l236_236294

theorem num_positive_x (x : ℕ) : (3 * x < 100) ∧ (4 * x ≥ 100) → x ≥ 25 ∧ x ≤ 33 := by
  sorry

theorem count_num_positive_x : 
  (∃ x : ℕ, (3 * x < 100) ∧ (4 * x ≥ 100)) → 
  (finset.range 34).filter (λ x, (3 * x < 100 ∧ 4 * x ≥ 100)).card = 9 := by
  sorry

end num_positive_x_count_num_positive_x_l236_236294


namespace evaporation_duration_l236_236136

-- Definitions based on conditions
def initial_volume : ℝ := 40 -- Ounces
def daily_evaporation_rate : ℝ := 0.01 -- Ounces per day
def evaporation_percentage : ℝ := 0.005 -- 0.5%

-- Theorem statement using conditions
theorem evaporation_duration :
  let total_evaporated_volume := evaporation_percentage * initial_volume in
  let days := total_evaporated_volume / daily_evaporation_rate in
  days = 20 :=
  by
    sorry

end evaporation_duration_l236_236136


namespace probability_not_all_same_l236_236910

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l236_236910


namespace seeds_distributed_equally_l236_236195

theorem seeds_distributed_equally (S G n seeds_per_small_garden : ℕ) 
  (hS : S = 42) 
  (hG : G = 36) 
  (hn : n = 3) 
  (h_seeds : seeds_per_small_garden = (S - G) / n) : 
  seeds_per_small_garden = 2 := by
  rw [hS, hG, hn] at h_seeds
  simp at h_seeds
  exact h_seeds

end seeds_distributed_equally_l236_236195


namespace measure_of_angle_A_perimeter_of_triangle_l236_236730

-- Definition of the conditions
def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  A + B + C = π ∧
  a = 4 ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos(A) ∧
  3 * sin(B) + sqrt(3) * cos(A) = 0 ∧
  1 / 2 * b * c * sin(A) = 2 * sqrt(3)

-- Prove the measure of angle A
theorem measure_of_angle_A (a b c A B C : ℝ) (h : triangle a b c A B C) : 
  A = 5 * π / 6 :=
sorry

-- Prove the perimeter given extra conditions and assuming A = 5π/6
theorem perimeter_of_triangle (a b c A : ℝ) (h1 : triangle a b c A A B C) 
  (h2 : A = 5 * π / 6) (S : ℝ) (h3 : S = 2 * sqrt(3)) :
  a + b + c = 4 + 2 * sqrt(13) + 2 * sqrt(3) :=
sorry

end measure_of_angle_A_perimeter_of_triangle_l236_236730


namespace roots_of_f_m_l236_236510

namespace RootsPolynomial

-- Definitions given in the problem
variables {K : Type*} [field K] (p : ℕ) (n : ℕ) (hp : nat.prime p) (hn : n ≠ 0)
variable [fintype K] (char_K : char_p K p) (card_K : fintype.card K = p^n)
noncomputable def unity := (1 : K)
noncomputable def m_h (m : ℕ) := finset.sum (finset.range m) (λ _, unity)
noncomputable def f_m (m : ℕ) : polynomial K := 
  finset.sum (finset.range (m+1)) (λ k, (-1)^(m - k) * (nat.choose m k) * X^(p^k))

-- The theorem to be proved
theorem roots_of_f_m {m : ℕ} (hm : m > 0) : 
  ∀ (k : ℕ), (k < p) → is_root (f_m p n) (m_h p n k) :=
begin
  sorry
end

end RootsPolynomial

end roots_of_f_m_l236_236510


namespace range_of_f_l236_236672

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_f : set.Icc (1 : ℝ) 10 = set.range (λ x, f x) ∩ set.Icc 1 5 := 
by {
  sorry
}

end range_of_f_l236_236672


namespace parallel_BE_DF_l236_236000

open EuclideanGeometry

variables {A B C D E F : Point}

/-- Let \( ABC \) be a triangle satisfying \( 2 \cdot \angle CBA = 3 \cdot \angle ACB \). 
Let \( D \) and \( E \) be points on the side \( AC \) such that \( BD \) and \( BE \) divide 
\( \angle CBA \) into three equal angles and such that \( D \) lies between \( A \) and \( E \). 
Furthermore, let \( F \) be the intersection of \( AB \) and the angle bisector of \( \angle ACB \). 
Show that \( BE \) and \( DF \) are parallel. -/
theorem parallel_BE_DF 
  (h1 : 2 * ∠ C B A = 3 * ∠ A C B)
  (h2 : trisection (angle B D A) (angle B E A) (angle E B A))
  (h3 : between (A, D, E))
  (h4 : F = intersection (line A B) (angle_bisector F (angle A C B))) :
  parallel (line B E) (line D F) := by
  sorry

end parallel_BE_DF_l236_236000


namespace sum_of_squares_due_to_regression_eq_72_l236_236338

theorem sum_of_squares_due_to_regression_eq_72
    (total_squared_deviations : ℝ)
    (correlation_coefficient : ℝ)
    (h1 : total_squared_deviations = 120)
    (h2 : correlation_coefficient = 0.6)
    : total_squared_deviations * correlation_coefficient^2 = 72 :=
by
  -- Proof goes here
  sorry

end sum_of_squares_due_to_regression_eq_72_l236_236338


namespace simplify_expression_l236_236790

variables {a b : ℝ}

theorem simplify_expression (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := 
by
  sorry

end simplify_expression_l236_236790


namespace four_digit_num_condition_l236_236300

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l236_236300


namespace line_passes_through_vertex_of_parabola_l236_236619

theorem line_passes_through_vertex_of_parabola : 
  { b : ℝ | (∃ x : ℝ, x + b = 2 * b^2) }.finite.card = 2 :=
by
  sorry

end line_passes_through_vertex_of_parabola_l236_236619


namespace min_value_of_expression_l236_236124

variable (a b c : ℝ)
variable (h1 : a + b + c = 1)
variable (h2 : 0 < a ∧ a < 1)
variable (h3 : 0 < b ∧ b < 1)
variable (h4 : 0 < c ∧ c < 1)
variable (h5 : 3 * a + 2 * b = 2)

theorem min_value_of_expression : (2 / a + 1 / (3 * b)) ≥ 16 / 3 := 
  sorry

end min_value_of_expression_l236_236124


namespace classroom_boys_count_l236_236953

theorem classroom_boys_count (total_students girls_per_boys_ratio boys_per_girls_ratio : ℕ) 
(total_count : total_students = 30) (ratio_constraint : girls_per_boys_ratio = 1 ∧ boys_per_girls_ratio = 2)
: 
let parts := girls_per_boys_ratio + boys_per_girls_ratio in 
let students_per_part := total_students / parts in
let boys := students_per_part * boys_per_girls_ratio in
boys = 20 :=
by sorry

end classroom_boys_count_l236_236953


namespace vertex_of_parabola_l236_236445

theorem vertex_of_parabola (a b c : ℝ) (h k : ℝ) (x y : ℝ) :
  (∀ x, y = (1/2) * (x - 1)^2 + 2) → (h, k) = (1, 2) :=
by
  intro hy
  exact sorry

end vertex_of_parabola_l236_236445


namespace quadratic_eq1_solution_quadratic_eq2_solution_l236_236795

-- Quadratic equation (1)
theorem quadratic_eq1_solution (x : ℝ) : (x-5)^2 - 16 = 0 → x = 9 ∨ x = 1 := 
by sorry

-- Quadratic equation (2)
theorem quadratic_eq2_solution (x : ℝ) : x^2 - 4x + 1 = 0 → x = 2 + real.sqrt 3 ∨ x = 2 - real.sqrt 3 := 
by sorry

end quadratic_eq1_solution_quadratic_eq2_solution_l236_236795


namespace cos_diff_alpha_beta_l236_236723

open Real

theorem cos_diff_alpha_beta (α β : ℝ) (h1 : sin α = 1 / 4) (h2 : β = -α) :
  cos (α - β) = 7 / 8 :=
by
  sorry

end cos_diff_alpha_beta_l236_236723


namespace average_speed_ratio_l236_236373

theorem average_speed_ratio 
  (jack_marathon_distance : ℕ) (jack_marathon_time : ℕ) 
  (jill_marathon_distance : ℕ) (jill_marathon_time : ℕ)
  (h1 : jack_marathon_distance = 40) (h2 : jack_marathon_time = 45) 
  (h3 : jill_marathon_distance = 40) (h4 : jill_marathon_time = 40) :
  (889 : ℕ) / 1000 = (jack_marathon_distance / jack_marathon_time) / 
                      (jill_marathon_distance / jill_marathon_time) :=
by
  sorry

end average_speed_ratio_l236_236373


namespace correct_function_at_x_equals_1_l236_236254

noncomputable def candidate_A (x : ℝ) : ℝ := (x - 1)^3 + 3 * (x - 1)
noncomputable def candidate_B (x : ℝ) : ℝ := 2 * (x - 1)^2
noncomputable def candidate_C (x : ℝ) : ℝ := 2 * (x - 1)
noncomputable def candidate_D (x : ℝ) : ℝ := x - 1

theorem correct_function_at_x_equals_1 :
  (deriv candidate_A 1 = 3) ∧ 
  (deriv candidate_B 1 ≠ 3) ∧ 
  (deriv candidate_C 1 ≠ 3) ∧ 
  (deriv candidate_D 1 ≠ 3) := 
by
  sorry

end correct_function_at_x_equals_1_l236_236254


namespace exists_k_composite_l236_236019

theorem exists_k_composite (k : ℕ) : 
  (∀ n : ℕ, ∃ p : ℕ, p > 1 ∧ p ∣ (k * 2^n + 1)) :=
begin
  sorry
end

end exists_k_composite_l236_236019


namespace min_value_of_quadratic_l236_236100

theorem min_value_of_quadratic (x : ℝ) : 
  (∃ x_min : ℝ, (∀ x : ℝ, x^2 - 10 * x + 24 ≥ x_min) ∧ x^2 - 10 * x + 24 = x_min) → x = 5 :=
begin
  sorry
end

end min_value_of_quadratic_l236_236100


namespace graph_contains_quadrilateral_l236_236635

variable {q : ℕ} (n l : ℕ) (points : set (ℝ × ℝ × ℝ))
variable (segments : set (ℝ × ℝ))
variable (A B C D : ℝ × ℝ × ℝ)
variable (q_ge_two : q ≥ 2)
variable (q_nat : q ∈ ℕ)
variable (n_def : n = q^2 + q + 1)
variable (l_min : l ≥ (1 / 2 : ℚ) * q * (q + 1)^2 + 1)
variable (not_coplanar : ∀ {A B C D : ℝ × ℝ × ℝ}, A ∈ points → B ∈ points → C ∈ points → D ∈ points → ¬ ∃ plane : set (ℝ × ℝ × ℝ), {A, B, C, D} ⊆ plane ∧ (plane is 2D))
variable (one_segment : ∀ {A : ℝ × ℝ × ℝ}, A ∈ points →  ∃ B : ℝ × ℝ × ℝ, B ∈ points ∧ (A, B) ∈ segments)
variable (exist_qplus2_segments : ∃ A : ℝ × ℝ × ℝ, A ∈ points ∧ ∃ neighbs : finset (ℝ × ℝ × ℝ), neighbs.card ≥ q + 2 ∧ ∀ B ∈ neighbs, (A, B) ∈ segments)

theorem graph_contains_quadrilateral : 
  ∃ A B C D : ℝ × ℝ × ℝ, (A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ D ∈ points) ∧ 
    ((A, B) ∈ segments ∧ (B, C) ∈ segments ∧ (C, D) ∈ segments ∧ (D, A) ∈ segments) :=
sorry

end graph_contains_quadrilateral_l236_236635


namespace halve_to_goal_in_8_l236_236030

-- Define the operation of halving and rounding down
def halve_round_down (n : ℕ) : ℕ := n / 2

-- Define the initial condition
def initial_value := 150

-- Define the goal condition
def goal_value := 1

-- Define the number of operations needed to reach the goal
def num_operations := 8

-- Define a recursive function to simulate the process and count operations
def count_operations (n : ℕ) : ℕ :=
  if n = goal_value then 0
  else 1 + count_operations (halve_round_down n)

-- The theorem we need to prove
theorem halve_to_goal_in_8 :
  count_operations initial_value = num_operations :=
sorry

end halve_to_goal_in_8_l236_236030


namespace find_MV_l236_236356

-- Definitions of the conditions
variables (J K L M P U V Q R : Type)
variables [rectangle J K L M]
variables [point_on_segment P L M]
variables [right_angle P K M]
variables [perpendicular U V L M]
variables [equal_lengths L U U P]
variables [line_intersect PK UV Q]
variables [point_on_segment R K M]
variables [line_through R K Q]
variables (PK KQ PQ : ℝ)
variables [equals PK 30]
variables [equals KQ 18]
variables [equals PQ 24]

-- Statement of the theorem
theorem find_MV : MV = 12.8 :=
sorry

end find_MV_l236_236356


namespace avg_salary_officers_correct_l236_236715

def total_employees := 465
def avg_salary_employees := 120
def non_officers := 450
def avg_salary_non_officers := 110
def officers := 15

theorem avg_salary_officers_correct : (15 * 420) = ((total_employees * avg_salary_employees) - (non_officers * avg_salary_non_officers)) := by
  sorry

end avg_salary_officers_correct_l236_236715


namespace max_students_seated_l236_236950

/-- Problem statement:
There are a total of 8 rows of desks.
The first row has 10 desks.
Each subsequent row has 2 more desks than the previous row.
We need to prove that the maximum number of students that can be seated in the class is 136.
-/
theorem max_students_seated : 
  let n := 8      -- number of rows
  let a1 := 10    -- desks in the first row
  let d := 2      -- common difference
  let an := a1 + (n - 1) * d  -- desks in the n-th row
  let S := n / 2 * (a1 + an)  -- sum of the arithmetic series
  S = 136 :=
by
  sorry

end max_students_seated_l236_236950


namespace chromium_percentage_correct_l236_236951

-- Definitions based on the problem conditions
def weight_alloy1 : ℝ := 15.0
def chromium_percent_alloy1 : ℝ := 0.15
def weight_alloy2 : ℝ := 35.0
def chromium_percent_alloy2 : ℝ := 0.08

-- Definitions for intermediate calculations
def chromium_alloy1 : ℝ := weight_alloy1 * chromium_percent_alloy1
def chromium_alloy2 : ℝ := weight_alloy2 * chromium_percent_alloy2
def total_chromium : ℝ := chromium_alloy1 + chromium_alloy2
def total_weight : ℝ := weight_alloy1 + weight_alloy2
def chromium_percentage_new_alloy : ℝ := (total_chromium / total_weight) * 100

-- The theorem to be proven
theorem chromium_percentage_correct : chromium_percentage_new_alloy = 10.1 := by
  sorry

end chromium_percentage_correct_l236_236951


namespace find_N_matrix_l236_236207

-- Define the conditions for the matrices
variables {a b c d e f g h : ℝ}

-- Define the original matrix and the transformation matrix
def original_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![c, d]]

def N_matrix (e f g h : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![e, f], ![g, h]]

def transformed_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4 * a, 4 * b], ![2 * c, 2 * d]]

-- The problem statement
theorem find_N_matrix :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    (N ⬝ original_matrix = transformed_matrix) → 
    (N = ![![4, 0], ![0, 2]]) :=
by
  sorry

end find_N_matrix_l236_236207


namespace appended_number_condition_l236_236525

theorem appended_number_condition (a x : ℕ) (h_pos_a : a > 0) (h_digit_x : x < 10) :
  let new_num := 10 * a + x in
  new_num - a * a = (11 - x) * a ↔ x = a :=
by {
  sorry,
}

end appended_number_condition_l236_236525


namespace largest_integer_in_mean_set_l236_236443

theorem largest_integer_in_mean_set :
  ∃ (A B C D : ℕ), 
    A < B ∧ B < C ∧ C < D ∧
    (A + B + C + D) = 4 * 68 ∧
    A ≥ 5 ∧
    D = 254 :=
sorry

end largest_integer_in_mean_set_l236_236443


namespace flight_duration_l236_236378

theorem flight_duration 
  (h m : ℕ) 
  (condition1 : 1 ≤ m) 
  (condition2 : m < 60) 
  (take_off : time := 3 + 15 / 60) 
  (land : time := 6 + 45 / 60) 
  (time_zone_diff : 1) 
  (h_eq : h = 2) 
  (m_eq : m = 30) 
  : h + m = 32 := 
by 
  sorry

end flight_duration_l236_236378


namespace horses_least_meeting_time_l236_236054

theorem horses_least_meeting_time :
  ∃ (T : ℕ), T > 0 ∧ (∃ (horses : Finset ℕ), horses.card ≥ 6 ∧
  (∀ k ∈ horses, k ≤ 12 ∧ T % k = 0) ∧ Nat.digits 10 T.sum = 6) ∧ T = 60 :=
sorry

end horses_least_meeting_time_l236_236054


namespace find_a_1001_l236_236351

-- Define the sequence a_n
noncomputable def a : ℕ → ℤ
| 0 := 2010
| 1 := 2011
| (n + 2) := a n + 2

-- Define the conditions
axiom h : ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 2 * n

-- State the theorem with the given conditions and answer
theorem find_a_1001 : a 1000 = 2678 :=
sorry

end find_a_1001_l236_236351


namespace repeating_decimals_sum_l236_236599

-- Define the repeating decimals as rational numbers
def dec_0_3 : ℚ := 1 / 3
def dec_0_02 : ℚ := 2 / 99
def dec_0_0004 : ℚ := 4 / 9999

-- State the theorem that we need to prove
theorem repeating_decimals_sum :
  dec_0_3 + dec_0_02 + dec_0_0004 = 10581 / 29889 :=
by
  sorry

end repeating_decimals_sum_l236_236599


namespace area_of_right_triangle_l236_236662

-- Define conditions
def side1 : ℝ := Real.sqrt 36
def side2 : ℝ := Real.sqrt 64
def hypotenuse : ℝ := Real.sqrt 100

-- Prove the area of the right triangle
theorem area_of_right_triangle : (side1 = 6) → (side2 = 8) → (hypotenuse = 10) → (side1^2 + side2^2 = hypotenuse^2) → (1 / 2 * side1 * side2 = 24) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2]
  sorry

end area_of_right_triangle_l236_236662


namespace sum_of_roots_eq_three_l236_236481

theorem sum_of_roots_eq_three {a b : ℝ} (h₁ : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (11 - a) = 0)
  (h₂ : a * 2^3 + (a + 3 * b) * 2^2 + (b - 4 * a) * 2 + (11 - a) = 0)
  (h₃ : a * 4^3 + (a + 3 * b) * 4^2 + (b - 4 * a) * 4 + (11 - a) = 0) :
  (-3) + 2 + 4 = 3 :=
by
  sorry

end sum_of_roots_eq_three_l236_236481


namespace exponent_simplification_l236_236333

theorem exponent_simplification (a b : ℝ) (ha : 80^a = 2) (hb : 80^b = 5) :
  16^((1 - a - b) / (2 * (1 - b))) = 4 :=
by
  sorry

end exponent_simplification_l236_236333


namespace greatest_two_digit_multiple_of_7_l236_236851

theorem greatest_two_digit_multiple_of_7 : ∃ n, 10 ≤ n ∧ n < 100 ∧ n % 7 = 0 ∧ ∀ m, 10 ≤ m ∧ m < 100 ∧ m % 7 = 0 → n ≥ m := 
by
  sorry

end greatest_two_digit_multiple_of_7_l236_236851


namespace probability_not_all_dice_show_different_l236_236921

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l236_236921


namespace clique_of_six_exists_min_degree_clique_six_l236_236532

theorem clique_of_six_exists (G : SimpleGraph) (h_order : fintype.card G.vertex = 1991)
  (h_degree : ∀ v : G.vertex, G.degree v ≥ 1593) :
  ∃ (K₆ : finset G.vertex), K₆.card = 6 ∧ (∀ u v ∈ K₆, u ≠ v → G.adj u v) :=
sorry

theorem min_degree_clique_six (G : SimpleGraph) :
  (∃ (d : ℕ), (∀ G : SimpleGraph, fintype.card G.vertex = 1991 → 
  ∀ v : G.vertex, G.degree v ≥ d → ∃ (K₆ : finset G.vertex), K₆.card = 6 ∧ (∀ u v ∈ K₆, u ≠ v → G.adj u v))) ↔ (d = 1593) :=
sorry

end clique_of_six_exists_min_degree_clique_six_l236_236532


namespace center_of_symmetry_sum_of_values_l236_236622

noncomputable def f (x : ℝ) := (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12)

theorem center_of_symmetry : 
  let f := (λ x : ℝ, (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12))
  in inflection_point f = (1/2, 1) := by
  sorry

theorem sum_of_values :
  let f := (λ x : ℝ, (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12))
  in ∑ k in (finset.range 2012).image (λ n, 1 / 2013 * n), f k = 2012 := by
  sorry

end center_of_symmetry_sum_of_values_l236_236622


namespace min_value_l236_236652

-- Conditions
variables {x y : ℝ}
variable (hx : x > 0)
variable (hy : y > 0)
variable (hxy : x + y = 2)

-- Theorem
theorem min_value (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  ∃ x y, (x > 0) ∧ (y > 0) ∧ (x + y = 2) ∧ (1/x + 4/y = 9/2) := 
by
  sorry

end min_value_l236_236652


namespace valid_vector_parameterizations_of_line_l236_236459

theorem valid_vector_parameterizations_of_line (t : ℝ) :
  (∃ t : ℝ, (∃ x y : ℝ, (x = 1 + t ∧ y = t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = -t ∧ y = -1 - t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = 2 + 0.5 * t ∧ y = 1 + 0.5 * t ∧ y = x - 1))) :=
by sorry

end valid_vector_parameterizations_of_line_l236_236459


namespace triangle_side_length_l236_236386

theorem triangle_side_length
  (a b c : ℝ)
  (area : a * c / 4 = sqrt 3)
  (angle_B : real.cos (real.pi / 3) = 1 / 2)
  (cond : a ^ 2 + c ^ 2 = 3 * a * c) :
  b = 2 * sqrt 2 := 
by
  sorry

end triangle_side_length_l236_236386


namespace factorization_of_polynomial_l236_236580

theorem factorization_of_polynomial : 
  ∀ (x : ℝ), 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) :=
by sorry

end factorization_of_polynomial_l236_236580


namespace James_comics_l236_236374

theorem James_comics (days_in_year : ℕ) (years : ℕ) (writes_every_other_day : ℕ) (no_leap_years : ℕ) 
  (h1 : days_in_year = 365) (h2 : years = 4) (h3 : writes_every_other_day = 2) : 
  (days_in_year * years) / writes_every_other_day = 730 := 
by
  sorry

end James_comics_l236_236374


namespace perpendiculars_concurrent_l236_236466

theorem perpendiculars_concurrent 
  (A B C D E F : Type*) 
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F]
  [triangle ABC A B C] [triangle DEF D E F] :
  (∃ P, is_intersection_point_of_perpendiculars_from_ABC_to_DEF P A B C D E F) →
  (∃ Q, is_intersection_point_of_perpendiculars_from_DEF_to_ABC Q D E F A B C) :=
  sorry

-- Definitions (placeholders for the actual geometric properties of the triangles and perpendiculars)
def is_intersection_point_of_perpendiculars_from_ABC_to_DEF 
  (P : Type*) (A B C D E F : Type*) : Prop :=
  sorry

def is_intersection_point_of_perpendiculars_from_DEF_to_ABC 
  (Q : Type*) (D E F A B C : Type*) : Prop :=
  sorry

end perpendiculars_concurrent_l236_236466


namespace women_reseat_ways_l236_236798

def S : ℕ → ℕ
| 0 := 0
| 1 := 1
| 2 := 2
| (n+3) := S (n+2) + S (n+1)

theorem women_reseat_ways : S 10 = 89 := by
  sorry

end women_reseat_ways_l236_236798


namespace rotated_parabola_equation_l236_236040

namespace Parabola

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 - 5 * x + 9

-- Define the transformation for 180-degree rotation
def rotated_parabola (x y : ℝ) : Prop := y = -x^2 - 5 * x - 9

-- State the theorem to prove the new equation after rotation
theorem rotated_parabola_equation (x : ℝ): 
  rotated_parabola x (- original_parabola x) := 
begin
  -- We know that original_parabola x = x^2 - 5 * x + 9
  unfold original_parabola,
  -- We need to prove: -(x^2 - 5 * x + 9) = -x^2 - 5 * x - 9
  simp,
end

end Parabola

end rotated_parabola_equation_l236_236040


namespace lcm_of_two_numbers_l236_236606

-- Define the numbers involved
def a : ℕ := 28
def b : ℕ := 72

-- Define the expected LCM result
def lcm_ab : ℕ := 504

-- State the problem as a theorem
theorem lcm_of_two_numbers : Nat.lcm a b = lcm_ab :=
by sorry

end lcm_of_two_numbers_l236_236606


namespace ratio_of_money_with_Ram_and_Gopal_l236_236470

noncomputable section

variable (R K G : ℕ)

theorem ratio_of_money_with_Ram_and_Gopal 
  (hR : R = 735) 
  (hK : K = 4335) 
  (hRatio : G * 17 = 7 * K) 
  (hGCD : Nat.gcd 735 1785 = 105) :
  R * 17 = 7 * G := 
by
  sorry

end ratio_of_money_with_Ram_and_Gopal_l236_236470


namespace probability_odd_divisor_15_factorial_l236_236460

theorem probability_odd_divisor_15_factorial:
  let n := 15!
  let total_divisors := (11+1) * (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let odd_divisors := (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let probability := odd_divisors.to_rat / total_divisors.to_rat
  probability = (1 : ℚ) / 6 := 
by {
  sorry
}

end probability_odd_divisor_15_factorial_l236_236460


namespace odd_divisors_iff_perfect_square_l236_236488

theorem odd_divisors_iff_perfect_square (n : ℕ) : 
  (∃ (d : ℕ), n % d = 0 ∧ ∀ (k : ℕ), (n % k = 0 → k = d ∨ k = n / d) ∧ #(finset.filter (λ k, n % k = 0) (finset.range (n + 1))) % 2 = 1) ↔ (∃ k : ℕ, n = k * k) :=
sorry

end odd_divisors_iff_perfect_square_l236_236488


namespace arithmetic_geometric_problem_l236_236364

-- Define the sequence properties and condition of arithmetic sequence
variable {a : ℕ → ℝ}
variable (q : ℝ) (h_q_pos : 0 < q)

-- Arithmetic geometric sequence condition (recursive geometric relation)
axiom h_seq : ∀ n, a (n + 1) = a n * q

-- Arithmetic sequence condition
axiom h_arith : 3 * a 1, (1/2) * a 3, 2 * a 2 form an arithmetic sequence

theorem arithmetic_geometric_problem 
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h_arith : 3 * a 1 = (1/2) * a 3 - 2 * a 2) :
  ∀ (a₀ a₁ a₂ a₃ : ℝ), 
    a 2016 q^2 - a 2017 q^3 = (a 2014) (1 - q) → 
    (a 2014) (q^2 - q^3) / (a 2014) (1 - q) = 9 := 
by 
  sorry

end arithmetic_geometric_problem_l236_236364


namespace cube_volume_ratio_l236_236085

theorem cube_volume_ratio (edge1 edge2 : ℕ) (h1 : edge1 = 10) (h2 : edge2 = 36) :
  (edge1^3 : ℚ) / (edge2^3) = 125 / 5832 :=
by
  sorry

end cube_volume_ratio_l236_236085


namespace five_eight_sided_dice_not_all_same_l236_236891

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l236_236891


namespace impossible_load_two_coins_l236_236960

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l236_236960


namespace largest_x_satisfies_eq_l236_236869

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236869


namespace circle_radius_l236_236728

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem circle_radius :
  let center : (ℝ × ℝ) := (2, 1),
      inside_point : (ℝ × ℝ) := (-2, 1),
      outside_point : (ℝ × ℝ) := (2, -5),
      r_inside := distance center inside_point,
      r_outside := distance center outside_point,
      r := 5 in
  r_inside < r ∧ r < r_outside ∧ r ∈ ℤ :=
by {
  let center : (ℝ × ℝ) := (2, 1),
      inside_point : (ℝ × ℝ) := (-2, 1),
      outside_point : (ℝ × ℝ) := (2, -5),
      r_inside := distance center inside_point,
      r_outside := distance center outside_point,
      r := 5,
  -- Proof starts
  sorry
}

end circle_radius_l236_236728


namespace cost_price_of_computer_table_l236_236465

theorem cost_price_of_computer_table (SP : ℝ) (h1 : SP = 1.15 * CP ∧ SP = 6400) : CP = 5565.22 :=
by
  sorry

end cost_price_of_computer_table_l236_236465


namespace probability_not_all_same_l236_236928

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l236_236928


namespace least_integer_in_ratio_1_3_5_l236_236844

theorem least_integer_in_ratio_1_3_5 (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 90) (h_ratio : a * 3 = b ∧ a * 5 = c) : a = 10 :=
sorry

end least_integer_in_ratio_1_3_5_l236_236844


namespace solve_for_a_l236_236268

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - real.sqrt 2

theorem solve_for_a (a : ℝ) (h : f a (f a (real.sqrt 2)) = -real.sqrt 2) : a = real.sqrt 2 / 2 :=
by
  sorry

end solve_for_a_l236_236268


namespace find_cost_price_of_article_l236_236139

theorem find_cost_price_of_article 
  (C : ℝ) 
  (h1 : 1.05 * C - 2 = 1.045 * C) 
  (h2 : 0.005 * C = 2) 
: C = 400 := 
by 
  sorry

end find_cost_price_of_article_l236_236139


namespace pairings_equal_l236_236353

-- Definitions for City A
def A_girls (n : ℕ) : Type := Fin n
def A_boys (n : ℕ) : Type := Fin n
def A_knows (n : ℕ) (g : A_girls n) (b : A_boys n) : Prop := True

-- Definitions for City B
def B_girls (n : ℕ) : Type := Fin n
def B_boys (n : ℕ) : Type := Fin (2 * n - 1)
def B_knows (n : ℕ) (i : Fin n) (j : Fin (2 * n - 1)) : Prop :=
  j.val < 2 * (i.val + 1)

-- Function to count the number of ways to pair r girls and r boys in city A
noncomputable def A (n r : ℕ) : ℕ := 
  if h : r ≤ n then 
    Nat.choose n r * Nat.choose n r * (r.factorial)
  else 0

-- Recurrence relation for city B
noncomputable def B (n r : ℕ) : ℕ :=
  if r = 0 then 1 else if r > n then 0 else
  if n < 2 then if r = 1 then (2 - 1) * 2 else 0 else
  B (n - 1) r + (2 * n - r) * B (n - 1) (r - 1)

-- We want to prove that number of pairings in city A equals number of pairings in city B for any r <= n
theorem pairings_equal (n r : ℕ) (h : r ≤ n) : A n r = B n r := sorry

end pairings_equal_l236_236353


namespace coin_loading_impossible_l236_236978

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l236_236978


namespace probability_of_not_all_same_number_l236_236907

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l236_236907


namespace woman_lawyer_probability_l236_236986

theorem woman_lawyer_probability (total_members women_count lawyer_prob : ℝ) 
  (h1: total_members = 100) 
  (h2: women_count = 0.70 * total_members) 
  (h3: lawyer_prob = 0.40) : 
  (0.40 * 0.70) = 0.28 := by sorry

end woman_lawyer_probability_l236_236986


namespace trigonometric_identity_l236_236512

theorem trigonometric_identity :
  sin 60 + tan 45 - cos 30 * tan 60 = (sqrt 3 - 1) / 2 := by
  sorry

end trigonometric_identity_l236_236512


namespace units_digit_sum_sequence_l236_236098

theorem units_digit_sum_sequence :
  let seq := [1!, 2!, 3!, 4!, 5!, 6!, 7!, 8!, 9!, 10!].map (λ n, n + 10) in
  let unit_digits := [11, 12, 16, 34].map (λ n, n % 10) ++ [0, 0, 0, 0, 0, 0] in
  (unit_digits.sum % 10 = 3) := 
by
  sorry

end units_digit_sum_sequence_l236_236098


namespace arithmetic_sequence_sum_l236_236248

variable {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ d : ℝ, a (n + 1) = a n + d

-- Condition 2: Given property
def property (a : ℕ → ℝ) : Prop :=
a 7 + a 13 = 20

theorem arithmetic_sequence_sum (h_seq : is_arithmetic_sequence a) (h_prop : property a) :
  a 9 + a 10 + a 11 = 30 := 
sorry

end arithmetic_sequence_sum_l236_236248


namespace largest_number_of_plates_l236_236138

theorem largest_number_of_plates (n : ℕ) (n ≥ 2) : 
  ∃ S : set (vector ℤ n), 
  (∀ (a b : vector ℤ n), a ∈ S → b ∈ S → a ≠ b → (∃ i j, i ≠ j ∧ (a.nth i ≠ b.nth i ∧ a.nth j ≠ b.nth j))) ∧
  card S = 10 ^ (n - 1) := 
sorry

end largest_number_of_plates_l236_236138


namespace sum_of_six_real_roots_of_symmetric_function_l236_236812

theorem sum_of_six_real_roots_of_symmetric_function
  (g : ℝ → ℝ)
  (h_symm : ∀ x : ℝ, g (4 + x) = g (4 - x))
  (h_roots : ∃ s : fin 6 → ℝ, (∀ i, g (s i) = 0) ∧ function.injective s) :
  ∑ i : fin 6, (fin 6 → ℝ) i = 24 :=
sorry

end sum_of_six_real_roots_of_symmetric_function_l236_236812


namespace evan_can_reflect_point_evan_cannot_construct_altitude_foot_l236_236764

def point := ℝ × ℝ
def line := point → Prop

variables (A : point) (ℓ : line) (can_draw_circle : set point → Prop)
  (can_mark_intersection : set point → point) (can_mark_arbitrary_point : point → point)

-- Condition: A point A in the plane
-- Condition: A line ℓ not passing through A
-- Condition: Evan has the ability to draw a circle through three distinct non-collinear points
-- Condition: Evan can mark the intersections between two drawn objects
-- Condition: Evan can mark an arbitrary point on a given object or on the plane

-- Problem (i): Evan can construct the reflection of A over ℓ
theorem evan_can_reflect_point (h1 : ¬ ℓ A) :
  ∃ R, _ := sorry

-- Problem (ii): Evan cannot construct the foot of the altitude from A to ℓ
theorem evan_cannot_construct_altitude_foot (h1 : ¬ ℓ A) :
  ¬ ∃ F, _ := sorry

end evan_can_reflect_point_evan_cannot_construct_altitude_foot_l236_236764


namespace similar_triangle_PVN_PBF_l236_236347

open EuclideanGeometry

noncomputable def triangle_similarity (circle : Type) (center : circle) (radius : ℝ)
(chord_PQ chord_RS : circle → circle → Prop) 
(perpendicular_bisector : chord_PQ → chord_RS → Prop)
(intersect_at_N : (PQ : circle) (RS : circle) (N : circle))
(point_V_between_R_and_N : (V : circle) (R : circle) (N : circle))
(extend_PV_meets_circle_at_B : (P : circle) (V : circle) (B : circle)) 
(similar_triangles : triangle PVN → triangle PBF → Prop) : Prop :=
∀ (circle : Type) (center : circle) (radius : ℝ) 
(chord_PQ chord_RS : circle → circle → Prop) 
(perpendicular_bisector : chord_PQ → chord_RS → Prop)
(intersect_at_N : (PQ : circle) (RS : circle) (N : circle))
(point_V_between_R_and_N : (V : circle) (R : circle) (N : circle))
(extend_PV_meets_circle_at_B : (P : circle) (V : circle) (B : circle)), 
similar_triangles PVN PBF

theorem similar_triangle_PVN_PBF (circle : Type) (center : circle) 
(radius : ℝ) (chord_PQ chord_RS : circle → circle → Prop)
(perpendicular_bisector : chord_PQ → chord_RS → Prop)
(intersect_at_N : (PQ : circle) (RS : circle) (N : circle))
(point_V_between_R_and_N : (V : circle) (R : circle) (N : circle))
(extend_PV_meets_circle_at_B : (P : circle) (V : circle) (B : circle)) :
similar_triangles PVN PBF :=
by
  sorry

end similar_triangle_PVN_PBF_l236_236347


namespace new_perimeter_is_20_l236_236438

/-
Ten 1x1 square tiles are arranged to form a figure whose outside edges form a polygon with a perimeter of 16 units.
Four additional tiles of the same size are added to the figure so that each new tile shares at least one side with 
one of the squares in the original figure. Prove that the new perimeter of the figure could be 20 units.
-/

theorem new_perimeter_is_20 (initial_perimeter : ℕ) (num_initial_tiles : ℕ) 
                            (num_new_tiles : ℕ) (shared_sides : ℕ) 
                            (total_tiles : ℕ) : 
  initial_perimeter = 16 → num_initial_tiles = 10 → num_new_tiles = 4 → 
  shared_sides ≤ 8 → total_tiles = 14 → (initial_perimeter + 2 * (num_new_tiles - shared_sides)) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end new_perimeter_is_20_l236_236438


namespace unique_cycle_exists_l236_236818

theorem unique_cycle_exists (n : ℕ) (h : n > 0) : 
  ∃ (assign : fin (2^n) → char), 
    (∀ (i j : fin (2^n)), (i ≠ j) → 
    (vector.map assign (fin_range n i) ≠ vector.map assign (fin_range n j))) :=
sorry


end unique_cycle_exists_l236_236818


namespace center_on_incenter_circumcenter_line_l236_236806

open EuclideanGeometry

variable {α β γ δ : Circle}
variable {A B C : Point}
variable {α_center β_center γ_center δ_center : Point}
variable {O1 O2 : Point}
variable {ABC : Triangle}

axiom circles_tangent_to_sides_near_vertices (α β γ : Circle) (A B C : Point) :
  α.Center = α_center ∧ β.Center = β_center ∧ γ.Center = γ_center ∧
  (∃ r, α.radius = r ∧ β.radius = r ∧ γ.radius = r) ∧
  α.TangentToSideNearVertex ABC A ∧ β.TangentToSideNearVertex ABC B ∧ γ.TangentToSideNearVertex ABC C

axiom δ_tangent_to_αβγ (δ : Circle) (α β γ : Circle) :
  δ.Center = δ_center ∧ δ.Tangent(α) ∧ δ.Tangent(β) ∧ δ.Tangent(γ)

axiom incircle_center (T : Triangle) (O1 : Point) : T.Incircle.Center = O1

axiom circumcircle_center (T : Triangle) (O2 : Point) : T.Circumcircle.Center = O2

theorem center_on_incenter_circumcenter_line (ABC : Triangle) (α β γ δ : Circle) (O1 O2 δ_center : Point) :
  circles_tangent_to_sides_near_vertices α β γ ABC.ABC →
  δ_tangent_to_αβγ δ α β γ →
  incircle_center ABC O1 →
  circumcircle_center ABC O2 →
  Collinear O1 O2 δ_center :=
sorry

end center_on_incenter_circumcenter_line_l236_236806


namespace option_A_equivalence_option_B_inequivalence_option_C_equivalence_option_D_inequivalence_final_result_l236_236554

-- Definitions of functions for each option
def f_A (x : ℝ) : ℝ := x
def g_A (x : ℝ) : ℝ := x^(3:ℝ).cbrt
def f_B (x : ℝ) : ℝ := x + 1
def g_B (x : ℝ) : ℝ := (x^2 - 1) / (x - 1)
def f_C (x : ℝ) : ℝ := √x + 1 / x
def g_C (t : ℝ) : ℝ := √t + 1 / t
def f_D (x : ℝ) : ℝ := √(x^2 - 1)
def g_D (x : ℝ) : ℝ := √(x + 1) * √(x - 1)

-- Theorem statements to verify the equivalence of functions
theorem option_A_equivalence : ∀ x : ℝ, f_A x = g_A x := 
by 
  intro x
  sorry

theorem option_B_inequivalence : ∃ x : ℝ, f_B x ≠ g_B x := 
by 
  sorry

theorem option_C_equivalence : ∀ x : ℝ, x > 0 → f_C x = g_C x :=
by 
  intro x hx
  sorry

theorem option_D_inequivalence : ∃ x : ℝ, f_D x ≠ g_D x := 
by 
  sorry

-- Combining results from all options
theorem final_result : option_A_equivalence ∧ option_C_equivalence ∧ option_B_inequivalence ∧ option_D_inequivalence :=
by 
  sorry

end option_A_equivalence_option_B_inequivalence_option_C_equivalence_option_D_inequivalence_final_result_l236_236554


namespace increasing_quadratic_l236_236588

noncomputable def f (a x : ℝ) : ℝ := 3 * x^2 - a * x + 4

theorem increasing_quadratic {a : ℝ} :
  (∀ x ≥ -5, 6 * x - a ≥ 0) ↔ a ≤ -30 :=
by
  sorry

end increasing_quadratic_l236_236588


namespace ferris_wheel_capacity_l236_236799

theorem ferris_wheel_capacity :
  ∀ (s_total s_broken p : ℕ), s_total = 18 → s_broken = 10 → p = 15 →
  (s_total - s_broken) * p = 120 :=
by
  intros s_total s_broken p ht hs hp
  rw [ht, hs, hp]
  simp
  sorry

end ferris_wheel_capacity_l236_236799


namespace max_a_for_inequality_l236_236233

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

theorem max_a_for_inequality :
  ∀ (x a : ℝ), x ∈ set.Icc a (a + 1) →
  f (x + a) ≥ f (2 * a - x) ↔ a ≤ -2 := 
sorry

end max_a_for_inequality_l236_236233


namespace sum_of_numbers_le_threshold_l236_236061
-- Import the necessary libraries

-- Definitions according to the conditions
def numbers : set ℝ := {0.8, 1/2, 0.9, 1/3}
def threshold : ℝ := 0.3

-- Statement of the problem in Lean 4
theorem sum_of_numbers_le_threshold : ∑ x in (numbers.filter (λ x, x ≤ threshold)), x = 0 := 
by {
  sorry
}

end sum_of_numbers_le_threshold_l236_236061


namespace probability_of_not_all_same_number_l236_236909

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l236_236909


namespace rationalize_denominator_correct_l236_236782

theorem rationalize_denominator_correct :
  ∃ (A C E : ℤ) (B D F : ℕ),
    B = 2 ∧ D = 2 ∧ A = 7 ∧ C = -7 ∧ E = -7 ∧ F = 3 ∧
    (∀ (p : ℕ), p.prime → p^2 ∣ B → False) ∧
    (∀ (p : ℕ), p.prime → p^2 ∣ D → False) ∧
    Nat.gcd (Nat.gcd A.natAbs (Nat.gcd C.natAbs (Nat.gcd E.natAbs F))) = 1 ∧
    A + B + C + D + E + F = 0 :=
by
  sorry

end rationalize_denominator_correct_l236_236782


namespace total_senior_students_l236_236614

theorem total_senior_students 
  (S : ℕ)
  (h_full_scholarship : (0.05 * S).floor = 0.05 * S)  -- Five percent of the senior students got a full merit scholarship
  (h_half_scholarship : (0.10 * S).floor = 0.10 * S)  -- Ten percent of the senior students got a half merit scholarship
  (h_no_scholarship : 255 = (0.85 * S).floor)         -- 255 senior students did not get any scholarships
  : S = 300 :=
by
  sorry

end total_senior_students_l236_236614


namespace probability_not_all_same_l236_236900

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l236_236900


namespace ball_label_problem_l236_236236

open Nat Real

theorem ball_label_problem
    (n : ℕ)
    (h_eq : n / (n + 2) = 2 / 5) :
  n = 2 ∧
  (let prob_A := (4 : ℝ) / 6 in prob_A = 2 / 3) ∧
  (let prob_sqrt := 1 - π / 4 in ∀ x y : ℝ, 0 ≤ x → x ≤ 4 → 0 ≤ y → y ≤ 4 → prob_sqrt = 1 - π / 4) :=
by
  sorry

end ball_label_problem_l236_236236


namespace jungkook_english_score_l236_236379

-- Definitions according to conditions
def initial_average : ℕ := 92
def new_average : ℕ := initial_average + 2
def number_of_subjects_initial : ℕ := 3
def number_of_subjects_total : ℕ := 4
def total_initial_score : ℕ := number_of_subjects_initial * initial_average

-- Proposition we want to prove
theorem jungkook_english_score :
  let total_new_score := number_of_subjects_total * new_average in 
  let english_score := total_new_score - total_initial_score in 
  english_score = 100 := by
  sorry

end jungkook_english_score_l236_236379


namespace dividend_is_2160_l236_236707

theorem dividend_is_2160 (d q r : ℕ) (h₁ : d = 2016 + d) (h₂ : q = 15) (h₃ : r = 0) : d = 2160 :=
by
  sorry

end dividend_is_2160_l236_236707


namespace total_cookies_l236_236573

theorem total_cookies (chris kenny glenn : ℕ) 
  (h1 : chris = kenny / 2)
  (h2 : glenn = 4 * kenny)
  (h3 : glenn = 24) : 
  chris + kenny + glenn = 33 := 
by
  -- Focusing on defining the theorem statement correct without entering the proof steps.
  sorry

end total_cookies_l236_236573


namespace sphere_surface_area_l236_236372

theorem sphere_surface_area (S A B C O : Type) [MetricSpace O] [Metric O] 
  (h1 : distance S A = 1) (h2 : distance A B = 1) (h3 : distance B C = sqrt 2) 
  (h4 : ∃ plane ABC, right_angle (line S A) (plane ABC) ∧ right_angle (line A B) (line B C))
  : surface_area_of_sphere O = 4 * π :=
by
  sorry

end sphere_surface_area_l236_236372


namespace number_of_ways_to_assign_6_grades_l236_236399

def valid_grades (grades : List ℕ) : Prop :=
  grades.all (λ g, g = 2 ∨ g = 3 ∨ g = 4) ∧
  ¬ grades.contains_adjacent (λ g => g = 2)

def a (n : ℕ) : ℕ
| 0         := 1
| 1         := 3
| 2         := 8
| (n + 1) := 2 * a n + 2 * a (n - 1)

theorem number_of_ways_to_assign_6_grades : 
  a 6 = 448 :=
by
  sorry

end number_of_ways_to_assign_6_grades_l236_236399


namespace largest_x_satisfying_equation_l236_236859

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l236_236859


namespace determine_true_propositions_l236_236843

-- Definitions for lines and their relationships
variables (a b c : Type)
variables [PlaneLine a] [PlaneLine b] [PlaneLine c]

-- Propositions
def prop1 (hab : a ∥ b) (hac : a ⟂ c) : b ⟂ c := sorry
def prop2 (hba : b ∥ a) (hca : c ∥ a) : b ⟂ c := sorry
def prop3 (hba : b ⟂ a) (hca : c ⟂ a) : b ⟂ c := sorry
def prop4 (hba : b ⟂ a) (hca : c ⟂ a) : b ∥ c := sorry

-- Mathematical Problem Statement
theorem determine_true_propositions :
  (prop1 a b c) ∧ (prop4 a b c) ∧ ¬ (prop2 a b c) ∧ ¬ (prop3 a b c) := sorry

end determine_true_propositions_l236_236843


namespace largest_x_satisfies_eq_l236_236884

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236884


namespace angle_ADC_l236_236348

noncomputable section

-- Define the conditions as hypotheses
variables (A B C D : Type) [AddGroup A] [AffineSpace A B] [AddGroup C] [AffineSpace C D]
variable [euclidean_geometry ℝ A] [euclidean_geometry ℝ B] [euclidean_geometry ℝ C] [euclidean_geometry ℝ D]

-- Given conditions of the problem
/-- Triangle ABC is isosceles with AB = AC --/
variable (AB AC : line_segment ℝ A) (BAC : ℝ)
variable (AD BD CD : line_segment ℝ D) 

-- Assumptions
axiom isosceles : AB = AC
axiom angle_BAC : BAC = 36
axiom point_D_inside_triangle : angle(D A B) = 90 ∧ AD = BD ∧ AD = CD

-- The theorem to prove
theorem angle_ADC (H1 : isosceles) (H2 : angle_BAC) (H3 : point_D_inside_triangle) : angle(A D C) = 90 := by
  sorry

end angle_ADC_l236_236348


namespace min_shift_for_symmetry_l236_236023

noncomputable def cos_shifted (x m : ℝ) : ℝ := 3 * Real.cos (2 * x - 2 * m + π / 3)

theorem min_shift_for_symmetry (m : ℝ) (k : ℤ) :
  (cos_shifted (-x m) = cos_shifted x m) → m > 0 → m = 5 * π / 12 :=
by
  intro h1 h2
  sorry

end min_shift_for_symmetry_l236_236023


namespace laplace_transform_of_f_l236_236780

noncomputable def eta : ℝ → ℝ
| t => if t < 0 then 0 else 1

def f (a : ℂ) (t : ℝ) : ℂ := eta t * Complex.exp (a * Complex.ofReal t)

theorem laplace_transform_of_f (a : ℂ) (p : ℂ) (h : p.re > a.re) :
  ∫ (t : ℝ) in 0..∞, f a t * Complex.exp (-p * Complex.ofReal t) = 1 / (p - a) :=
by
  sorry

end laplace_transform_of_f_l236_236780


namespace largest_x_satisfying_equation_l236_236860

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l236_236860


namespace prob_not_A_l236_236020

variables {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω}

-- Define the events
def event_A : Set Ω := {x | x = 0}  -- Placeholder, actual event content is irrelevant for probability assignment
def event_B : Set Ω := {x | x = 1}  -- Placeholder
def event_C : Set Ω := {x | x = 2}  -- Placeholder

-- Assigning probabilities to events
axiom prob_A : μ event_A = 0.7
axiom prob_B : μ event_B = 0.2
axiom prob_C : μ event_C = 0.1

-- The proof goal
theorem prob_not_A : μ event_Aᶜ = 0.3 :=
by {
    sorry
}

end prob_not_A_l236_236020


namespace length_AB_l236_236811

noncomputable def parabola_p := 3
def x1_x2_sum := 6

theorem length_AB (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : x1 + x2 = x1_x2_sum)
  (h2 : (y1^2 = 6 * x1) ∧ (y2^2 = 6 * x2))
  : abs (x1 + parabola_p / 2 - (x2 + parabola_p / 2)) = 9 := by
  sorry

end length_AB_l236_236811


namespace slope_of_line_connecting_tangent_points_and_focus_is_one_l236_236273

noncomputable def point := (-1, 2)
def parabola (x y : ℝ) := y^2 = 4 * x
def focus := (1, 0)

theorem slope_of_line_connecting_tangent_points_and_focus_is_one
  (k : ℝ)
  (tangent_line : ℝ → ℝ → Prop)
  (tangent_points : set (ℝ × ℝ)) :
  (∀ x y : ℝ, tangent_line x y ↔ (y - 2 = k * (x + 1))) →
  (∀ x y : ℝ, parabola x y → tangent_line x y → (x, y) ∈ tangent_points) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ∈ tangent_points → (x₂, y₂) ∈ tangent_points → x₁ ≠ x₂ → 
    let slope_1 := (y₁ - focus.2) / (x₁ - focus.1),
        slope_2 := (y₂ - focus.2) / (x₂ - focus.1)
    in slope_1 = 1 ∧ slope_2 = 1)
  : True :=
sorry

end slope_of_line_connecting_tangent_points_and_focus_is_one_l236_236273


namespace odd_function_expression_l236_236658

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then cos (3 * x) + sin (2 * x) else
  if x < 0 then sin (2 * x) - cos (3 * x) else sorry

theorem odd_function_expression (x : ℝ) (h_odd : ∀ x, f(-x) = -f(x)) (hx_neg : x < 0) :
  f(x) = sin (2 * x) - cos (3 * x) :=
by
  sorry

end odd_function_expression_l236_236658


namespace tan_theta_identity_l236_236076

-- Define the conditions and the theorem
theorem tan_theta_identity 
  (θ α β : ℝ) 
  (h1 : 0 < θ ∧ θ < π / 2) -- acute angle θ
  (h2 : β < α): -- condition β < α
  tan θ = (2 * sin α * sin β) / (sin (α - β)) :=
sorry

end tan_theta_identity_l236_236076


namespace coin_loading_impossible_l236_236967

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l236_236967


namespace friends_courses_l236_236718

-- Define the notions of students and their properties
structure Student :=
  (first_name : String)
  (last_name : String)
  (year : ℕ)

-- Define the specific conditions from the problem
def students : List Student := [
  ⟨"Peter", "Krylov", 1⟩,
  ⟨"Nikolay", "Ivanov", 2⟩,
  ⟨"Boris", "Karpov", 3⟩,
  ⟨"Vasily", "Orlov", 4⟩
]

-- The main statement of the problem
theorem friends_courses :
  ∀ (s : Student), s ∈ students →
    (s.first_name = "Peter" → s.last_name = "Krylov" ∧ s.year = 1) ∧
    (s.first_name = "Nikolay" → s.last_name = "Ivanov" ∧ s.year = 2) ∧
    (s.first_name = "Boris" → s.last_name = "Karpov" ∧ s.year = 3) ∧
    (s.first_name = "Vasily" → s.last_name = "Orlov" ∧ s.year = 4) :=
by
  sorry

end friends_courses_l236_236718


namespace basketball_opponents_score_l236_236989

theorem basketball_opponents_score (games : List ℕ)
    (team_scores := [2, 3, 4, 5, 6, 8, 10, 12])
    (lost_games : ℕ × ℕ × ℕ × ℕ := (3, 5, 9, 11))
    (lost_by_one : ∀ {x : ℕ}, x ∈ lost_games → ∃ y, y = x + 1):
  let s₁ := [4, 6, 10, 12]
  let s₂ := [6, 4]
  let total_points := s₁.sum + s₂.sum 
  total_points = 38 := 
  by 
  {
    sorry 
  }

end basketball_opponents_score_l236_236989


namespace find_a_find_angle_B_l236_236342

-- Definition of the conditions
variables (A B C a b c : ℝ)

-- Given conditions as hypotheses
-- Translate these conditions into Lean hypotheses
hypothesis h1 : b = 2
hypothesis h2 : real.cos (A - (real.pi / 3)) = 2 * real.cos A
hypothesis h3 : ∃ S : ℝ, S = (1/2) * b * c * real.sin A ∧ S = 3 * real.sqrt 3

-- Function for area calculation
def area (b c A : ℝ) : ℝ := (1 / 2) * b * c * real.sin A

-- Translate the questions to Lean theorem statements
theorem find_a :
  ∃ a : ℝ, b = 2 ∧
          real.cos (A - (real.pi / 3)) = 2 * real.cos A ∧
          (∃ S, S = (1/2) * b * c * real.sin A ∧ S = 3 * real.sqrt 3) →
          a = 2 * real.sqrt 7 :=
sorry

theorem find_angle_B :
  ∃ B : ℝ, b = 2 ∧
           real.cos (A - (real.pi / 3)) = 2 * real.cos A ∧
           (∃ S, S = (1/2) * b * c * real.sin A ∧ S = 3 * real.sqrt 3) ∧
           (real.cos (2 * C) = 1 - (a^2) / (6 * b^2)) →
           (B = real.pi / 12 ∨ B = 7 * real.pi / 12) :=
sorry

end find_a_find_angle_B_l236_236342


namespace smallest_sum_of_56_and_78_is_134_l236_236419

theorem smallest_sum_of_56_and_78_is_134 : 
  ∃ (a b c d : ℕ), {a, b, c, d} = {5, 6, 7, 8} ∧ 
  a * 10 + b + c * 10 + d = 134 :=
sorry

end smallest_sum_of_56_and_78_is_134_l236_236419


namespace count_possible_values_of_x_l236_236280

theorem count_possible_values_of_x :
  let n := (set.count {x : ℕ | 25 ≤ x ∧ x ≤ 33 ∧ ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c * x < 100 ∧ 3 ≤ b ≤ 100/x}) in
  n = 9 :=
by
  -- Here we must prove the statement by the provided conditions
  sorry

end count_possible_values_of_x_l236_236280


namespace percentage_of_pure_acid_l236_236056

theorem percentage_of_pure_acid (volume_pure_acid : ℝ) (total_volume_solution : ℝ) (h1 : volume_pure_acid = 2.5) (h2 : total_volume_solution = 10) : 
  let percentage_pure_acid := (volume_pure_acid / total_volume_solution) * 100 in
  percentage_pure_acid = 25 :=
by
  sorry

end percentage_of_pure_acid_l236_236056


namespace smallest_n_verify_n_eq_3_smallest_n_value_l236_236784

-- Define the conditions given in the problem.
def boxes (n : ℕ) : ℕ := 15 * n
def remaining_cookies (n : ℕ) : ℕ := boxes n - 3
def bags (n : ℕ) : Prop := remaining_cookies n % 7 = 0

-- The statement: proving the smallest n.
theorem smallest_n (n : ℕ) : bags n → n ≥ 3 :=
by
  sorry

theorem verify_n_eq_3 : bags 3 :=
by
  sorry

-- The smallest n satisfying the condition
theorem smallest_n_value : ∃ n, bags n ∧ n = 3 :=
by
  use 3
  exact verify_n_eq_3

end smallest_n_verify_n_eq_3_smallest_n_value_l236_236784


namespace largest_x_satisfies_eq_l236_236886

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236886


namespace equal_segments_AY_DY_l236_236411

open EuclideanGeometry

variable {P Q R S T U : Type*}
variable (A B C D X Z : P)
variable (Y : P)
variable [trapezoid A B C D]
variable [points_on_lats A B C D X Z]
variable [lines_intersects cx bx Y]
variable [pentagon_inscribed A X Y Z D]

theorem equal_segments_AY_DY 
  (h_trapezoid : trapezoid A B C D)
  (h_points_on_lats : points_on_lats A B C D X Z)
  (h_lines_intersects : lines_intersects cx bx Y)
  (h_pentagon_inscribed : pentagon_inscribed A X Y Z D)
  : segment_eq A Y D Y := sorry

end equal_segments_AY_DY_l236_236411


namespace solve_f_l236_236600

-- Adding the function definitions and conditions as definitions in Lean 4
variable {f : ℝ → ℝ}

-- Assuming function domain and range
axiom f_domain : ∀ x, 0 < x → x < (1 / 0) → f(x) ∈ ℝ

-- Functional equations and conditions
axiom functional_eq : ∀ x y : ℝ, 0 < x → 0 < y →
  f(x) * f(y) + f(2008 / x) * f(2008 / y) = 2 * f(x * y)

axiom f_2008 : f(2008) = 1

-- The proof goal
theorem solve_f : ∀ x : ℝ, 0 < x → x < (1 / 0) → f(x) = 1 :=
by
  intro x hx
  sorry

end solve_f_l236_236600


namespace probability_not_all_dice_show_different_l236_236917

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l236_236917


namespace box_depth_is_10_l236_236990

variable (depth : ℕ)

theorem box_depth_is_10 
  (length width : ℕ)
  (cubes : ℕ)
  (h1 : length = 35)
  (h2 : width = 20)
  (h3 : cubes = 56)
  (h4 : ∃ (cube_size : ℕ), ∀ (c : ℕ), c = cube_size → (length % cube_size = 0 ∧ width % cube_size = 0 ∧ 56 * cube_size^3 = length * width * depth)) :
  depth = 10 :=
by
  sorry

end box_depth_is_10_l236_236990


namespace find_least_x_l236_236205

theorem find_least_x (x : ℤ) (h : x + 3490 ≡ 2801 [MOD 15]) : x = 11 :=
sorry

end find_least_x_l236_236205


namespace hyperbola_y_relation_l236_236643

theorem hyperbola_y_relation {k y₁ y₂ : ℝ} 
  (A_on_hyperbola : y₁ = k / 2) 
  (B_on_hyperbola : y₂ = k / 3) 
  (k_positive : 0 < k) : 
  y₁ > y₂ := 
sorry

end hyperbola_y_relation_l236_236643


namespace four_digit_num_condition_l236_236298

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l236_236298


namespace distance_and_triangle_l236_236708

variable (x1 y1 : ℝ) 

-- Given points
def origin := (0, 0)
def pointA := (12, 0)
def pointB := (12, -5)

-- Distance formula from the origin to another point
def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt ((x - 0)^2 + (y - 0)^2)

theorem distance_and_triangle :
  distance_from_origin 12 (-5) = 13 ∧ 
  ( (pointA.1 - origin.1 = 12 ∧ pointB.2 - pointA.2 = -5) → 
    (pointA.1 = pointB.1 ∧ pointA.2 = origin.2) → 
    (pointA.2 = origin.2 ∧ pointB.1 = pointA.1 - origin.1) → 
    pointA.1 = pointB.1 ∧ pointA.2 = origin.2 ∧ (pointB.2 - origin.2) ≠ 0 →
    true
  ) :=
by 
  -- Skipping the proof
  sorry

end distance_and_triangle_l236_236708


namespace investment_schemes_correct_l236_236993

-- Define the parameters of the problem
def num_projects : Nat := 3
def num_districts : Nat := 4

-- Function to count the number of valid investment schemes
def count_investment_schemes (num_projects num_districts : Nat) : Nat :=
  let total_schemes := num_districts ^ num_projects
  let invalid_schemes := num_districts
  total_schemes - invalid_schemes

-- Theorem statement
theorem investment_schemes_correct :
  count_investment_schemes num_projects num_districts = 60 := by
  sorry

end investment_schemes_correct_l236_236993


namespace line_through_origin_and_conditions_l236_236118

-- Definitions:
def system_defines_line (m n p x y z : ℝ) : Prop :=
  (x / m = y / n) ∧ (y / n = z / p)

def lies_in_coordinate_plane (m n p : ℝ) : Prop :=
  (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)

def coincides_with_coordinate_axis (m n p : ℝ) : Prop :=
  (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)

-- Theorem statement:
theorem line_through_origin_and_conditions (m n p x y z : ℝ) :
  system_defines_line m n p x y z →
  (∀ m n p, lies_in_coordinate_plane m n p ↔ (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)) ∧
  (∀ m n p, coincides_with_coordinate_axis m n p ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)) :=
by
  sorry

end line_through_origin_and_conditions_l236_236118


namespace probability_same_color_correct_l236_236520

def number_of_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def probability_two_balls_same_color : ℚ :=
  ((green_balls / number_of_balls)^2 + (red_balls / number_of_balls)^2 + (blue_balls / number_of_balls)^2)

theorem probability_same_color_correct :
  probability_two_balls_same_color = 49 / 128 := sorry

end probability_same_color_correct_l236_236520


namespace problem_solution_l236_236640

section math_proof_problem

variables {P Q M N : Point}
variables (C : Circle) (line_l line_l1 : Line)
variables {d : ℝ} (l_eq1 l_eq2 q_eq : Equation)

-- Define the Circle C with its equation
def circle_C : Circle := { center := (3, -2), radius := 3 }

-- Define point P(2,0)
def point_P : Point := (2, 0)

-- Given line l passes through point P and has a distance of 1 from the center of C
def line_l_eq1 (l : Line) : Prop :=
  -- Equation of the line 3x + 4y - 6 = 0
  l = { a := 3, b := 4, c := -6 }

def line_l_eq2 (l : Line) : Prop :=
  -- Equation of the line x = 2
  l = { a := 1, b := 0, c := -2 }

-- The two possible equations of the line l
def possible_line_eqs (l : Line) : Prop :=
  line_l_eq1 l ∨ line_l_eq2 l

-- Define the intersection points M and N and their distance being 4
def intersection_points (l : Line) (C : Circle) (M N : Point) : Prop :=
  -- l passes through P and intersects C at M and N
  l = line_l1 ∧ passes_through l point_P ∧
  intersects l C M ∧ intersects l C N ∧
  distance M N = 4

-- Define circle Q with diameter MN
def circle_Q : Circle := { center := (2, 0), radius := 2 }

-- Prove the conditions stated in the problem
theorem problem_solution :
  (∀ l : Line, (passes_through l point_P → distance_from_center C l = 1 → possible_line_eqs l)) ∧
  (∀ M N : Point, intersection_points line_l1 C M N → circle_eq_with_diameter M N q_eq → q_eq = (x-2)^2 + y^2 = 4 )
:= by
-- Proof would go here
sorry

end math_proof_problem

end problem_solution_l236_236640


namespace correct_operation_l236_236942

theorem correct_operation (a b : ℝ) :
  ((a^2)^3 = a^6 ∧ ¬(3 * a - 2 * a = 1) ∧ ¬((a * b^2)^2 = a * b^4) ∧ ¬(a^6 / a^2 = a^3)) :=
by {
  -- Proof statements each condition here
  sorry
}

end correct_operation_l236_236942


namespace probability_both_white_given_same_color_l236_236479

open Finset

def bag := ["white", "white", "white", "black", "black"]

def same_color_pairs := ((choose (bag.filter (λ c, c = "white")) ⟨2, by sorry⟩) ++ (choose (bag.filter (λ c, c = "black")) ⟨2, by sorry⟩))

theorem probability_both_white_given_same_color : 
  (3 : ℚ) / 4 = 3 / 10 := 
sorry

end probability_both_white_given_same_color_l236_236479


namespace rest_time_after_every_ten_miles_l236_236535

namespace toProof

def walk_rate : ℝ := 10 -- The walking rate in mph
def total_time : ℝ := 332 -- The total time taken in minutes
def distance : ℝ := 50 -- The total distance in miles
def walk_time : ℝ := (distance / walk_rate) * 60 -- Time to walk 50 miles in minutes (converted from hours)
def number_of_rests : ℝ := distance / 10 - 1 -- The number of 10-mile segments (distances minus one for the last stop)
def total_rest_time : ℝ := total_time - walk_time -- The total rest time

-- The proposition that the rest time after every ten miles is 8 minutes
theorem rest_time_after_every_ten_miles : (total_rest_time / number_of_rests) = 8 :=
by
  sorry

end toProof

end rest_time_after_every_ten_miles_l236_236535


namespace pairs_same_function_l236_236161

def f1 (x : ℝ) : ℝ := x
def g1 (x : ℝ) : ℝ := (sqrt x) ^ 2

def f2 (x : ℝ) : ℝ := x - 2
def g2 (x : ℝ) : ℝ := sqrt (x^2 - 4*x + 4)

def f3 (x : ℝ) (h : x ≥ 0) : ℝ := real.pi * x^2
def g3 (r : ℝ) (h : r ≥ 0) : ℝ := real.pi * r^2

def f4 (x : ℝ) : ℝ := abs x
def g4 (x : ℝ) : ℝ := if x ≥ 0 then x else -x

theorem pairs_same_function : 
  (∀ x, f3 x (le_refl x) = g3 x (le_refl x)) ∧ 
  (∀ x, f4 x = g4 x) :=
by {
  sorry,
  sorry
}

end pairs_same_function_l236_236161


namespace expected_dietary_restriction_l236_236413

theorem expected_dietary_restriction (n : ℕ) (p : ℚ) (sample_size : ℕ) (expected : ℕ) :
  p = 1 / 4 ∧ sample_size = 300 ∧ expected = sample_size * p → expected = 75 := by
  sorry

end expected_dietary_restriction_l236_236413


namespace original_number_l236_236208

theorem original_number (n : ℕ) (h1 : 2319 % 21 = 0) (h2 : 2319 = 21 * (n + 1) - 1) : n = 2318 := 
sorry

end original_number_l236_236208


namespace linear_function_unique_l236_236391

/-
  Let f(x) be a linear function such that f(f(x)) = 4x - 1 and f(3) = -5.
  Prove that f(x) = -2x + 1.
-/

theorem linear_function_unique (f : ℝ → ℝ)
  (h1 : ∃a b : ℝ, ∀ x : ℝ, f(x) = a * x + b)
  (h2 : ∀ x : ℝ, f(f(x)) = 4 * x - 1)
  (h3 : f(3) = -5) :
  ∀ x : ℝ, f(x) = -2 * x + 1 :=
by
  sorry

end linear_function_unique_l236_236391


namespace charlie_paints_150_square_feet_l236_236157

-- Definitions based on conditions from a)
def ratio_alice_bob_charlie : ℕ × ℕ × ℕ := (3, 4, 5)
def total_area : ℕ := 360

-- Problem statement in Lean 4
theorem charlie_paints_150_square_feet : 
  let (a, b, c) := ratio_alice_bob_charlie in
  a + b + c = 12 → 
  total_area / (a + b + c) * c = 150 :=
by
  sorry

end charlie_paints_150_square_feet_l236_236157


namespace max_PA2_PB2_PC2_l236_236642

-- Define the points A, B, and C.
def A : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (4, -2)

-- Define the function that calculates the squared distance between two points.
def squared_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Define the function for the sum of squared distances from point P to points A, B, and C.
def sum_of_squared_distances (P : ℝ × ℝ) : ℝ :=
  squared_distance P A + squared_distance P B + squared_distance P C

-- Define the set of points on the circle with radius 2 centered at the origin.
def circle : set (ℝ × ℝ) := {P | P.1^2 + P.2^2 = 4}

-- Define the maximum value of the sum of squared distances when P is on the circle.
theorem max_PA2_PB2_PC2 : ∃ P ∈ circle, sum_of_squared_distances P = 88 :=
sorry

end max_PA2_PB2_PC2_l236_236642


namespace angle_CGH_l236_236749

noncomputable def angle (A B C: Point): ℝ := sorry -- Assuming we have an angle definition

-- Define the problem conditions
variable (O A B F G H: Point)
variable (circle: Circle)
variable (tangentAtB tangentAtF: Line)
variable (diameter: Segment)
variable (angle_BAF: ℝ)

-- Explicit conditions from the problem
-- Let AB be a diameter of a circle centered at O
hypothesis h1: circle.center = O
hypothesis h2: diameter = AB
hypothesis h3: is_diameter O A B
-- Let F be a point on the circle
hypothesis h4: F ∈ circle
-- Let the tangent at B intersect the tangent at F and AF at G and H respectively
hypothesis h5: tangent_to_circle B circle tangentAtB
hypothesis h6: tangent_to_circle F circle tangentAtF
hypothesis h7: G = tangentAtB ∩ tangentAtF
hypothesis h8: H = tangentAtB ∩ AF
-- \(\angle BAF = 30^\circ\)
hypothesis h9: angle B A F = 30

-- The proof statement
theorem angle_CGH (h1 h2 h3 h4 h5 h6 h7 h8 h9) :
  angle C G H = 60 :=
sorry

end angle_CGH_l236_236749


namespace walk_to_school_l236_236339

theorem walk_to_school (W P : ℕ) (h1 : W + P = 41) (h2 : W = P + 3) : W = 22 :=
by 
  sorry

end walk_to_school_l236_236339


namespace cost_price_of_book_l236_236999

theorem cost_price_of_book (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 90) 
  (h2 : rate_of_profit = 0.8) 
  (h3 : rate_of_profit = (SP - CP) / CP) : 
  CP = 50 :=
sorry

end cost_price_of_book_l236_236999


namespace impossible_to_load_two_coins_l236_236975

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l236_236975


namespace transformation_is_projective_l236_236018

noncomputable section

-- Define the transformation
def P (x y : ℝ) : ℝ × ℝ :=
  (1 / x, y / x)

-- Prove that the transformation is projective
theorem transformation_is_projective :
  (P : ℝ × ℝ → ℝ × ℝ) :=
by
  sorry

end transformation_is_projective_l236_236018


namespace almost_surely_equal_l236_236432

noncomputable theory
open ProbabilityTheory

variables {Ω : Type*} {ξ ζ : Ω → ℝ} {ξ_n : ℕ → Ω → ℝ}

axiom conv_prob_ξ : ∀ ε > 0, lim_sup (λ n, probability (λ ω, abs (ξ_n n ω - ξ ω) ≥ ε)) = 0
axiom conv_prob_ζ : ∀ ε > 0, lim_sup (λ n, probability (λ ω, abs (ξ_n n ω - ζ ω) ≥ ε)) = 0

theorem almost_surely_equal :
  probability (λ ω, ξ ω = ζ ω) = 1 :=
sorry

end almost_surely_equal_l236_236432


namespace series_sum_l236_236557

theorem series_sum : (∑ n in Finset.range 50, 2 / ((2 * n + 1) * (2 * n + 3))) = 100 / 101 :=
by
  sorry

end series_sum_l236_236557


namespace omega_eq_six_l236_236068

theorem omega_eq_six (A ω : ℝ) (φ : ℝ) (f : ℝ → ℝ) (h1 : A ≠ 0) (h2 : ω > 0)
  (h3 : -π / 2 < φ ∧ φ < π / 2) (h4 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h5 : ∀ x, f (-x) = -f x) 
  (h6 : ∀ x, f (x + π / 6) = -f (x - π / 6)) :
  ω = 6 :=
sorry

end omega_eq_six_l236_236068


namespace compare_sizes_l236_236657

variables {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

variable (h_even : even_function f)
variable (h_decreasing : decreasing_on f 0 2)

noncomputable def a := f 1
noncomputable def b := f 2
noncomputable def c := f (1 / 2)

theorem compare_sizes (h_even : even_function f) (h_decreasing : decreasing_on f 0 2):
  c > a ∧ a > b :=
by {
  sorry
}

end compare_sizes_l236_236657


namespace number_of_correct_conclusions_l236_236062

def condition1 (a b : ℝ) : Prop := abs (a * b) = abs a * abs b
def condition2 (a b : ℝ) : Prop := ∀ (𝒶 𝒷 : Vector ℝ), abs (𝒶 • 𝒷) = abs 𝒶 * abs 𝒷
def condition3 (𝒶 : Vector ℝ) : Prop := (𝒶 • 𝒶) = (abs 𝒶) ^ 2
def condition4 (z : ℂ) : Prop := z ^ 2 = abs z ^ 2
def condition5 (z : ℂ) : Prop := abs (z ^ 2) = abs z ^ 2

theorem number_of_correct_conclusions : (∃ (c₁ c₂ c₃ c₄ c₅ : Prop),
  (c₁ = condition1 ∧ c₁ ∧
   c₂ = condition2 ∧ ¬c₂ ∧
   c₃ = condition3 ∧ c₃ ∧
   c₄ = condition4 ∧ ¬c₄ ∧
   c₅ = condition5 ∧ c₅) → 3) :=
begin
  sorry
end

end number_of_correct_conclusions_l236_236062


namespace stratified_sampling_selection_l236_236543

theorem stratified_sampling_selection (l s kl ks n : ℕ) (hl : l = 5) (hs : s = 10) (hkl : kl = 2) (hks : ks = 4) (hn : n = 6) : 
  l + s = n ∧ kl + ks = n → (Nat.choose l kl * Nat.choose s ks = 2100) :=
by 
  intros
  rw [hl, hs, hkl, hks, hn]
  sorry

end stratified_sampling_selection_l236_236543


namespace sports_club_membership_l236_236710

theorem sports_club_membership 
  (B : ℕ) (T : ℕ) (Both : ℕ) (Neither : ℕ)
  (hB : B = 17) (hT : T = 19) (hBoth : Both = 8) (hNeither : Neither = 2) :
  B + T - Both + Neither = 30 := 
  by 
    -- conditions are replaced by provided proofs
    rw [hB, hT, hBoth, hNeither]
    -- the actual arithmetic is performed
    sorry

end sports_club_membership_l236_236710


namespace geom_seq_min_value_l236_236239

theorem geom_seq_min_value  {a : ℕ → ℝ} (h_pos : ∀ n, 0 < a n)
                             (h_geom : ∀ n, a (n + 1) = 2 * a n)
                             (h_eq : a 2018 = a 2017 + 2 * a 2016)
                             (m n : ℕ) (h_mn : m + n = 6)
                             (h_sqrt : ∀ (m n : ℕ), (sqrt (a m * a n) = 4 * a 1))
                             : 1/m + 5/n = 7/4 :=
by
  sorry

end geom_seq_min_value_l236_236239


namespace distribute_students_l236_236544

theorem distribute_students (n : ℕ) (k : ℕ) (spots : ℕ) (classes : ℕ) (h_n : n = 8) (h_k : k = 6)
  (h_spots : spots = 8) (h_classes : classes = 6) :
  (∑ (i : ℕ) in finset.Icc 1 k, k - i + 1) + nat.binom k 2 = 21 :=
by
  -- Proof omitted
  sorry

end distribute_students_l236_236544


namespace find_radius_l236_236462

open Real

noncomputable def radius_of_circumscribed_circle (r : ℝ) : Prop := 
  3 * r * sqrt 3 = pi * r^2

theorem find_radius : ∃ r : ℝ, radius_of_circumscribed_circle r ∧ r = 3 * sqrt 3 / pi :=
  sorry

end find_radius_l236_236462


namespace isosceles_triangle_area_l236_236079

theorem isosceles_triangle_area (a b c : ℝ) (h: a = 5 ∧ b = 5 ∧ c = 6)
  (altitude_splits_base : ∀ (h : 3^2 + x^2 = 25), x = 4) : 
  ∃ (area : ℝ), area = 12 := 
by
  sorry

end isosceles_triangle_area_l236_236079


namespace parabola_points_l236_236472

theorem parabola_points :
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} :=
by
  sorry

end parabola_points_l236_236472


namespace correct_propositions_l236_236250

-- Definitions of lines and planes
variable (m l : Line)
variable (α β : Plane)

-- Propositions representing the given conditions
def proposition1 : Prop := 
  ∀ (p q : Line), (p ∈ α ∧ q ∈ α ∧ p ≠ q ∧ l ⊥ p ∧ l ⊥ q) → l ⊥ α

def proposition2 : Prop := 
  ∀ (p : Line), (l ∥ α → l ∥ p) ∧ (p ∈ α)

def proposition3 : Prop := 
  (m ⊂ α ∧ l ⊂ β ∧ l ⊥ m) → α ⊥ β

def proposition4 : Prop := 
  (l ⊂ β ∧ l ⊥ α) → α ⊥ β

def proposition5 : Prop := 
  (m ⊂ α ∧ l ⊂ β ∧ α ∥ β) → l ∥ m

-- Correct answer: Propositions 1 and 4 are correct
theorem correct_propositions :
  (proposition1 m l α β ∧ proposition4 m l α β)
  ∧
  ¬ (proposition2 m l α β ∧ proposition3 m l α β ∧ proposition5 m l α β) := by
  sorry

end correct_propositions_l236_236250


namespace angle_MKD_is_90_l236_236021

variables {Point : Type}
variables (M H D L K A B : Point)
variables (angle : Point → Point → Point → ℝ)
variables (dist : Point → Point → ℝ)

-- Conditions
axiom cyclic_quadrilateral (hMHD_MLD : angle M H D + angle M L D = 180)
axiom lengths_equal (hKH : dist K H = (1 / 2) * dist A B)
axiom hAK : dist A K = dist K H
axiom hDL : dist D L = dist K H

-- Target to prove
theorem angle_MKD_is_90 :
  angle M K D = 90 :=
sorry

end angle_MKD_is_90_l236_236021


namespace proving_b_and_extreme_value_of_f_l236_236256

noncomputable def f (x : ℝ) (b : ℝ) := x^2 + b * Real.log x
noncomputable def g (x : ℝ) := (x - 10) / (x - 4)

theorem proving_b_and_extreme_value_of_f 
  (h_parallel: (deriv (λ x, f x b) 5 = deriv g 5))
  (b_value: b = -20) :
  b = -20 ∧ (∃ x_extreme : ℝ, x_extreme = Real.sqrt 10 ∧ f x_extreme (-20) = 10 - 10 * Real.log 10) :=
by
  have h_b : b = -20 := by sorry
  have h_extreme : ∃ x_extreme : ℝ, x_extreme = Real.sqrt 10 ∧ f x_extreme (-20) = 10 - 10 * Real.log 10 := by sorry
  exact ⟨h_b, h_extreme⟩

end proving_b_and_extreme_value_of_f_l236_236256


namespace rotten_apples_did_not_smell_l236_236415

theorem rotten_apples_did_not_smell (total_apples : ℕ) (rotten_percentage smell_percentage : ℕ) :
  total_apples = 200 →
  rotten_percentage = 40 →
  smell_percentage = 70 →
  let rotten_apples := (rotten_percentage * total_apples) / 100 in
  let smelling_rotten_apples := (smell_percentage * rotten_apples) / 100 in
  let non_smelling_rotten_apples := rotten_apples - smelling_rotten_apples in
  non_smelling_rotten_apples = 24 :=
by {
  intros h1 h2 h3,
  have h_rotten_apples : rotten_apples = 80,
  { calc
      rotten_apples
          = (rotten_percentage * total_apples) / 100 : by rw [←h2, ←h1, nat.mul_div_cancel' (by norm_num : total_apples % 100 = 0)]
      ... = 80 : by norm_num },
  have h_smelling_rotten_apples : smelling_rotten_apples = 56,
  { calc
      smelling_rotten_apples
          = (smell_percentage * rotten_apples) / 100 : by rw [←h3, ←h_rotten_apples, nat.mul_div_cancel' (by norm_num : rotten_apples % 100 = 0)]
      ... = 56 : by norm_num },
  have h_non_smelling_rotten_apples : non_smelling_rotten_apples = 24,
  { calc
      non_smelling_rotten_apples
          = rotten_apples - smelling_rotten_apples : rfl
      ... = 80 - 56 : by rw [h_rotten_apples, h_smelling_rotten_apples]
      ... = 24 : by norm_num },
  exact h_non_smelling_rotten_apples,
}

end rotten_apples_did_not_smell_l236_236415


namespace find_x_approximate_l236_236729

-- Define the conditions given in the problem
def y : ℝ := 7
def z : ℝ := 3
def cos_with_angle_diff : ℝ := 40 / 41

-- Define the goal to prove
theorem find_x_approximate : 
  ∃ x : ℝ, abs (x - 7.49) < 0.01 ∧ 
           (∃ cos_X : ℝ, x^2 = y^2 + z^2 - 2 * y * z * cos_X ∧ 
                         cos_with_angle_diff + cos_X = 42 * (1 - cos_X^2) / (y^2 + z^2 - 2 * y * z * cos_X)) :=
sorry

end find_x_approximate_l236_236729


namespace find_x_given_y64_l236_236834

variable (x y k : ℝ)

def inversely_proportional (x y : ℝ) := (x^3 * y = k)

theorem find_x_given_y64
  (h_pos : x > 0 ∧ y > 0)
  (h_inversely : inversely_proportional x y)
  (h_given : inversely_proportional 2 8)
  (h_y64 : y = 64) :
  x = 1 := by
  sorry

end find_x_given_y64_l236_236834


namespace top_three_positions_l236_236711

theorem top_three_positions (athletes : Finset String) (h_athletes : athletes = {"Alex", "Ben", "Carl", "Danny", "Emma", "Fiona"}) : 
  (∃ outcomes : ℕ, outcomes = 6 * 5 * 4 ∧ outcomes = 120) :=
by
  have h_size : athletes.card = 6,
  { rw h_athletes, simp },
  use 6 * 5 * 4,
  split,
  { refl },
  { norm_num }

end top_three_positions_l236_236711


namespace polygon_interior_angle_sum_360_l236_236539

theorem polygon_interior_angle_sum_360 (n : ℕ) (h : (n-2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angle_sum_360_l236_236539


namespace total_cookies_l236_236570

variable (ChrisCookies KennyCookies GlennCookies : ℕ)
variable (KennyHasCookies : GlennCookies = 4 * KennyCookies)
variable (ChrisHasCookies : ChrisCookies = KennyCookies / 2)
variable (GlennHas24Cookies : GlennCookies = 24)

theorem total_cookies : GlennCookies + KennyCookies + ChrisCookies = 33 := 
by
  have KennyCookiesEq : KennyCookies = 24 / 4 := by 
    rw [GlennHas24Cookies, mul_div_cancel_left, nat.mul_comm, nat.one_div, nat.div_self] ; trivial
  have ChrisCookiesEq : ChrisCookies = 6 / 2 := by 
    rw [KennyCookiesEq, ChrisHasCookies]
  rw [ChrisCookiesEq, KennyCookiesEq, GlennHas24Cookies]
  exact sorry

end total_cookies_l236_236570


namespace range_condition_l236_236668

def f (x: ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) else 1

theorem range_condition : {x : ℝ | f(x + 1) < f(2 * x)} = {x : ℝ | x < 0} :=
  sorry

end range_condition_l236_236668


namespace g_of_3_to_6_eq_81_l236_236390

noncomputable theory

-- Declaration of properties of functions f and g
variables (f g : ℝ → ℝ)

-- Conditions
axiom h1 : ∀ x, x ≥ 1 → f (g x) = x^4
axiom h2 : ∀ x, x ≥ 1 → g (f x) = x^6
axiom h3 : g 81 = 81

-- Theorem statement
theorem g_of_3_to_6_eq_81 : [g 3]^6 = 81 :=
by {
  sorry
}

end g_of_3_to_6_eq_81_l236_236390


namespace find_x_l236_236656

def f : ℕ → ℕ := sorry
def g : ℕ → ℕ := sorry
def f_domain : set ℕ := {1, 2, 3}
def g_domain : set ℕ := {1, 2, 3}
axiom f_in_domain : ∀ x, x ∈ f_domain → f x ∈ f_domain
axiom g_in_domain : ∀ x, x ∈ g_domain → g x ∈ g_domain
axiom f1 : f 1 = 1
axiom f2 : f 2 = 3
axiom f3 : f 3 = 1
axiom g_def : ∀ x ∈ g_domain, g x + x = 4

theorem find_x :
  ∃ x ∈ g_domain, f (g x) > g (f x) ∧ x = 2 :=
sorry

end find_x_l236_236656


namespace correct_statement_l236_236103

-- Definition of quadrants
def is_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def is_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_third_quadrant (θ : ℝ) : Prop := -180 < θ ∧ θ < -90
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement of the problem
theorem correct_statement : is_obtuse_angle θ → is_second_quadrant θ :=
by sorry

end correct_statement_l236_236103


namespace flat_fee_l236_236997

theorem flat_fee (f n : ℝ) 
  (h1 : f + 3 * n = 205) 
  (h2 : f + 6 * n = 350) : 
  f = 60 := 
by
  sorry

end flat_fee_l236_236997


namespace monthly_salary_l236_236502

theorem monthly_salary (S : ℝ) (E : ℝ) 
  (h1 : S - 1.20 * E = 220)
  (h2 : E = 0.80 * S) :
  S = 5500 :=
by
  sorry

end monthly_salary_l236_236502


namespace sequence_geometric_l236_236725

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, a n ≠ 0)
  (h_arith : 2 * a 2 = a 1 + a 3)
  (h_geom : a 3 ^ 2 = a 2 * a 4)
  (h_recip_arith : 2 / a 4 = 1 / a 3 + 1 / a 5) :
  a 3 ^ 2 = a 1 * a 5 :=
sorry

end sequence_geometric_l236_236725


namespace math_proof_l236_236617

open BigOperators

noncomputable def problem_statement : Prop :=
  let floor_log_floor_sum := ∑ a in Finset.range 244, (Real.log a) / (Real.log 3).floor
  floor_log_floor_sum = 857

theorem math_proof : problem_statement := by
  -- problem and conditions
  sorry

end math_proof_l236_236617


namespace three_pow_2040_mod_5_l236_236089

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l236_236089


namespace num_positive_x_count_num_positive_x_l236_236295

theorem num_positive_x (x : ℕ) : (3 * x < 100) ∧ (4 * x ≥ 100) → x ≥ 25 ∧ x ≤ 33 := by
  sorry

theorem count_num_positive_x : 
  (∃ x : ℕ, (3 * x < 100) ∧ (4 * x ≥ 100)) → 
  (finset.range 34).filter (λ x, (3 * x < 100 ∧ 4 * x ≥ 100)).card = 9 := by
  sorry

end num_positive_x_count_num_positive_x_l236_236295


namespace cost_of_first_variety_l236_236368

theorem cost_of_first_variety (x : ℝ) (cost2 : ℝ) (cost_mix : ℝ) (ratio : ℝ) :
    cost2 = 8.75 →
    cost_mix = 7.50 →
    ratio = 0.625 →
    (x - cost_mix) / (cost2 - cost_mix) = ratio →
    x = 8.28125 := 
by
  intros h1 h2 h3 h4
  sorry

end cost_of_first_variety_l236_236368


namespace distance_small_sphere_to_table_l236_236772

-- Definitions based on the conditions in (a)
def Sphere (r : ℝ) := {center : ℝ × ℝ × ℝ // (center.1 ^ 2 + center.2 ^ 2 + center.3 ^ 2) = r ^ 2}

variables (α : ℝ) (R : ℝ)
variables (sphere1 sphere2 sphere3 sphere4 : Sphere (2 * R))
variables (small_sphere : Sphere R)

-- Condition: The centers of the 4 large spheres form the vertices of a square on the table surface α.
def centers_form_square : Prop := 
  ∃ (A B C D : ℝ × ℝ × ℝ), 
    A ∈ sphere1 ∧ B ∈ sphere2 ∧ C ∈ sphere3 ∧ D ∈ sphere4 ∧
    (A.1 = B.1) ∧ (A.2 = D.2) ∧ (B.2 = C.2) ∧ (D.1 = C.1) ∧ 
    ((A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = (4 * R) ^ 2)

-- Condition: The small sphere is tangent to the centers of the 4 large spheres below.
def small_sphere_tangent : Prop := 
  ∃ (E : ℝ × ℝ × ℝ), E ∈ small_sphere ∧ 
  ((E.1 - fst sphere1.val).sqrt + (E.2 - snd sphere1.val).sqrt + (E.3 - 0.0.snd sphere1.val) = (3 * R))

-- The theorem to prove
theorem distance_small_sphere_to_table (h1 : centers_form_square α sphere1 sphere2 sphere3 sphere4)
  (h2 : small_sphere_tangent small_sphere sphere1 sphere2 sphere3 sphere4) :
  ∃ (d : ℝ), d = 3 * R :=
sorry

end distance_small_sphere_to_table_l236_236772


namespace sum_arithmetic_series_remainder_l236_236095

theorem sum_arithmetic_series_remainder :
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S % 9 = 5 :=
by
  let a := 2
  let l := 12
  let d := 1
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  show S % 9 = 5
  sorry

end sum_arithmetic_series_remainder_l236_236095


namespace largest_vector_magnitude_l236_236247

-- Definitions for the conditions in a)
def e1 : ℝ^3 := sorry -- Assume e1 is a unit vector in ℝ³
def e2 : ℝ^3 := sorry -- Assume e2 is a unit vector in ℝ³
axiom non_collinear : ¬collinear e1 e2
axiom unit_e1 : ‖e1‖ = 1
axiom unit_e2 : ‖e2‖ = 1
noncomputable def option_a := 1 / 2 * e1 + 1 / 2 * e2
noncomputable def option_b := 1 / 3 * e1 + 2 / 3 * e2
noncomputable def option_c := 2 / 5 * e1 + 3 / 5 * e2
noncomputable def option_d := 1 / 4 * e1 + 3 / 4 * e2

-- The problem's statement in Lean, proving D has the largest magnitude
theorem largest_vector_magnitude : 
  ‖option_d‖ >= ‖option_a‖ ∧ ‖option_d‖ >= ‖option_b‖ ∧ ‖option_d‖ >= ‖option_c‖ := 
sorry

end largest_vector_magnitude_l236_236247


namespace volume_of_region_l236_236219

-- Define the conditions
def condition1 (x y z : ℝ) := abs (x + y + 2 * z) + abs (x + y - 2 * z) ≤ 12
def condition2 (x : ℝ) := x ≥ 0
def condition3 (y : ℝ) := y ≥ 0
def condition4 (z : ℝ) := z ≥ 0

-- Define the volume function
def volume (x y z : ℝ) := 18 * 3

-- Proof statement
theorem volume_of_region : ∀ (x y z : ℝ),
  condition1 x y z →
  condition2 x →
  condition3 y →
  condition4 z →
  volume x y z = 54 := by
  sorry

end volume_of_region_l236_236219


namespace probability_of_eight_is_zero_l236_236775

noncomputable def repeating_block (n : ℚ) : ℕ → ℕ := fun i => nat.digits 10 n.fract_part.denom

def contains_eight (n : ℚ) :=
  ∃ i : ℕ, repeating_block n i = 8

theorem probability_of_eight_is_zero :
  repeating_block (3 / 11) = [2, 7] →
  ¬ contains_eight (3 / 11) →
  ∀ digit, digit ∈ repeating_block (3 / 11) → digit ≠ 8 :=
by
  sorry

end probability_of_eight_is_zero_l236_236775


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236936

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236936


namespace average_ABC_l236_236184

/-- Given three numbers A, B, and C such that 1503C - 3006A = 6012 and 1503B + 4509A = 7509,
their average is 3  -/
theorem average_ABC (A B C : ℚ) 
  (h1 : 1503 * C - 3006 * A = 6012) 
  (h2 : 1503 * B + 4509 * A = 7509) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_l236_236184


namespace angles_of_triangle_A2B2C2_l236_236412

theorem angles_of_triangle_A2B2C2 (ABC A1 B1 C1 A2 B2 C2 : Type)
  [equilateral_triangle ABC]
  (isosceles_A1BC : ∀ {α}, α + β + γ = 60)
  (isosceles_AB1C : isIsoTriangle ABC)
  (isosceles_ABC1 : isIsoTriangle ABC)
  (hA2 : intersects BC1 B1C A2)
  (hB2 : intersects AC1 A1C B2)
  (hC2 : intersects AB1 A1B C2) :
  ∃ (α β γ : ℝ), angle A2 B2 C2 = 3 * α ∧ angle B2 C2 A2 = 3 * β ∧ angle C2 A2 B2 = 3 * γ :=
sorry

end angles_of_triangle_A2B2C2_l236_236412


namespace remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l236_236096

theorem remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14 
  (a b c d e f g h : ℤ) 
  (h1 : a = 11085)
  (h2 : b = 11087)
  (h3 : c = 11089)
  (h4 : d = 11091)
  (h5 : e = 11093)
  (h6 : f = 11095)
  (h7 : g = 11097)
  (h8 : h = 11099) :
  (2 * (a + b + c + d + e + f + g + h)) % 14 = 2 := 
by
  sorry

end remainder_when_sum_of_8_consecutive_odds_is_multiplied_by_2_and_divided_by_14_l236_236096


namespace count_palindromes_l236_236683

def is_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = n % 10)

theorem count_palindromes : 
  {n : ℕ | is_palindrome n}.to_finset.card = 90 := 
by
  sorry

end count_palindromes_l236_236683


namespace alice_birthday_2025_l236_236550

def isLeapYear (y : Nat) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def dayShift (startDay : Nat) (years : Nat) : Nat :=
  (List.range years).foldl (λ acc i => acc + (if isLeapYear (2012 + i) then 2 else 1)) startDay

def nextMondayAfter2012 (startDay : Nat) : Nat :=
  let daysInWeek := 7
  let targetDay := 1 -- Monday
  (List.range (2025 - 2012)).foldl (λ (year, currentDay) _ =>
    let newDay := (currentDay + (if isLeapYear year then 2 else 1)) % daysInWeek
    if newDay = targetDay then (year, newDay) else (year + 1, newDay)
  ) (2012, startDay)

theorem alice_birthday_2025 : nextMondayAfter2012 4 = 2025 :=
  sorry

end alice_birthday_2025_l236_236550


namespace arithmetic_sequence_tenth_term_l236_236636

/- 
  Define the arithmetic sequence in terms of its properties 
  and prove that the 10th term is 18.
-/

theorem arithmetic_sequence_tenth_term (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 8) : a 10 = 18 := 
by 
  sorry

end arithmetic_sequence_tenth_term_l236_236636


namespace polynomial_characterization_l236_236761

theorem polynomial_characterization (k : ℕ) (P : ℝ → ℝ) (h_deg : degree P = k) (h_zeros : ∀ a, P a = 0 → P (a + 1) = 1) : 
  ∃ c : ℝ, ∀ x : ℝ, P x = x + c :=
by sorry

end polynomial_characterization_l236_236761


namespace five_eight_sided_dice_not_all_same_l236_236890

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l236_236890


namespace infinitely_many_lines_parallel_to_plane_unique_plane_parallel_to_given_plane_l236_236322

-- Condition: There is a plane and a point not on the plane.
variable {P : Type} -- P is the type of the points in space
variable {Plane : Type} -- Plane is the type of the planes in space
variable [affine_space P Plane] -- P forms an affine space with planes

-- Given: a plane π and a point A not on π
variable (π : Plane) (A : P)
variable (hA_not_on_π : ¬ A ∈ π)

-- Proof Problem 1: Prove that there are infinitely many lines parallel to π through A.
theorem infinitely_many_lines_parallel_to_plane (π : Plane) (A : P) (hA_not_on_π : ¬ A ∈ π) :
  ∃ L : set (set P), (∀ l ∈ L, (A ∈ l ∧ ¬ ∃ P' ∈ l, P' ∈ π)) ∧ infinite L := sorry

-- Proof Problem 2: Prove that there is exactly one plane parallel to π through A.
theorem unique_plane_parallel_to_given_plane (π : Plane) (A : P) (hA_not_on_π : ¬ A ∈ π) :
  ∃! π' : Plane, (∀ B : P, B ∈ π' ↔ (A ∈ π' ∧ ∃ Q ∈ π, ∀ R ∈ π', R-Q ∈ (π' - A))) := sorry

end infinitely_many_lines_parallel_to_plane_unique_plane_parallel_to_given_plane_l236_236322


namespace class_president_is_yi_l236_236551

variable (Students : Type)
variable (Jia Yi Bing StudyCommittee SportsCommittee ClassPresident : Students)
variable (age : Students → ℕ)

-- Conditions
axiom bing_older_than_study_committee : age Bing > age StudyCommittee
axiom jia_age_different_from_sports_committee : age Jia ≠ age SportsCommittee
axiom sports_committee_younger_than_yi : age SportsCommittee < age Yi

-- Prove that Yi is the class president
theorem class_president_is_yi : ClassPresident = Yi :=
sorry

end class_president_is_yi_l236_236551


namespace added_grape_juice_10_gallons_l236_236991

-- Define the initial conditions
def initial_volume := 30 -- 30 gallons
def initial_percent_juice := 0.10 -- 10 percent

-- Define the resulting condition
def resulting_percent_juice := 0.325 -- 32.5 percent

-- Define the initial amount of grape juice in the mixture
def initial_juice := initial_volume * initial_percent_juice

-- Let x be the amount of grape juice added
variable (x : ℝ)

-- Define the total volume and total amount of grape juice after adding x gallons
def total_volume := initial_volume + x
def total_juice := initial_juice + x

-- The resulting mixture condition
def resulting_condition := total_juice = resulting_percent_juice * total_volume

-- The theorem to be proved
theorem added_grape_juice_10_gallons (x : ℝ) : resulting_condition x → x = 10 := by
  sorry

end added_grape_juice_10_gallons_l236_236991


namespace find_a2016_l236_236274

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = (1 + a n) / (1 - a n)

theorem find_a2016 (a : ℕ → ℚ) (h : sequence a) : a 2016 = 1 / 3 :=
by
  sorry

end find_a2016_l236_236274


namespace find_m_plus_n_plus_d_l236_236345

noncomputable def calculate_m_n_d (r L_dist O_E_dist: ℝ) : ℝ :=
  let m := 2500 - 20736 + 119 * π
  let n := 2 * sqrt 119
  let d := 119
  m + n + d

theorem find_m_plus_n_plus_d :
  let r := 50
  let L := 96
  let O_E_dist := 24
  let form_area := calculate_m_n_d r L O_E_dist = 173
  form_area = 173 :=
by
  -- You can fill in the actual proof here if desired
  sorry

end find_m_plus_n_plus_d_l236_236345


namespace antoine_wins_l236_236029

-- Define the conditions of the game
def game_condition (current_score : ℕ) : Prop :=
  current_score <= 100 ∧ ∀ n, 1 ≤ n ∧ n ≤ 10 → ∃ m, 1 ≤ m ∧ m ≤ 10 ∧ (current_score + m ≤ 100)

-- Define the condition of the winning strategy
def winning_strategy (player : ℕ -> ℕ -> Prop) (start_score : ℕ) : Prop :=
  player start_score 0 = 1 ∧ ∀ opp_move, 1 ≤ opp_move ∧ opp_move ≤ 10 → player (start_score + opp_move) 0 = 1

-- Statement of the problem: Antoine (player 1) has the winning strategy
theorem antoine_wins : ∃ (strategy : ℕ -> ℕ -> Prop), game_condition 0 ∧ winning_strategy strategy 0 :=
sorry

end antoine_wins_l236_236029


namespace min_yellow_surface_area_is_one_fourth_l236_236185

def yellow_surface_area_min_fraction := 1 / 4

def is_min_yellow_surface_area_exposed (n : ℕ) (m : ℕ) (k : ℕ) : Prop :=
  let total_surface_area := 6 * (n * n)
  let inner_cubes := (n - 2) * (n - 2) * (n - 2)
  let surface_yellow_cubes := m - inner_cubes
  let exposed_yellow_area := surface_yellow_cubes
  (exposed_yellow_area : ℚ) / total_surface_area = yellow_surface_area_min_fraction

theorem min_yellow_surface_area_is_one_fourth :
  ∀ (n : ℕ) (m : ℕ) (k : ℕ),
  n = 4 →
  m = 32 →
  k = 32 →
  n * n * n = m + k →
  is_min_yellow_surface_area_exposed n m k :=
begin
  intros n m k hn hm hk htotal,
  sorry -- Proof goes here
end

end min_yellow_surface_area_is_one_fourth_l236_236185


namespace positive_difference_l236_236172

theorem positive_difference:
  let a := (7^3 + 7^3) / 7
  let b := (7^3)^2 / 7
  b - a = 16709 :=
by
  sorry

end positive_difference_l236_236172


namespace gardener_prob_l236_236531

noncomputable def total_permutations : ℕ :=
  (Nat.factorial 12) / ((Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 5))

noncomputable def ways_to_arrange_maples_and_oaks : ℕ :=
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 4))

noncomputable def ways_to_place_birch_trees : ℕ :=
  Nat.choose 8 5

noncomputable def favorable_permutations : ℕ :=
  ways_to_arrange_maples_and_oaks * ways_to_place_birch_trees

noncomputable def probability_fraction : ℚ :=
  favorable_permutations / total_permutations

noncomputable def m_n_sum : ℕ :=
  let f := probability_fraction.num
  let d := probability_fraction.denom
  f + d

theorem gardener_prob (m n : ℕ) (h1 : probability_fraction = m / n) : m + n = 106 :=
sorry

end gardener_prob_l236_236531


namespace cube_surface_area_of_given_volume_l236_236053

noncomputable def cube_side_length (V : ℕ) := ℝ.cbrt V
noncomputable def cube_surface_area (s : ℝ) := 6 * s^2

theorem cube_surface_area_of_given_volume :
  cube_surface_area (cube_side_length 3375) = 1350 :=
by
  sorry

end cube_surface_area_of_given_volume_l236_236053


namespace Sarah_score_l236_236787

theorem Sarah_score (G S : ℕ) (h1 : S = G + 60) (h2 : (S + G) / 2 = 108) : S = 138 :=
by
  sorry

end Sarah_score_l236_236787


namespace max_min_dist_product_correct_l236_236001

noncomputable def prod_max_min_dist (A B : ℝ × ℝ) (hA : A.1^2 + 3 * A.2^2 = 1) (hB : B.1^2 + 3 * B.2^2 = 1) (h_perp : A.1 * B.1 + 3 * A.2 * B.2 = 0) : ℝ :=
  let dist_AB := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
  -- Compute the product of maximum and minimum distances
  (2 * real.sqrt 3 / 3)^2

theorem max_min_dist_product_correct (A B : ℝ × ℝ) (hA : A.1^2 + 3 * A.2^2 = 1) (hB : B.1^2 + 3 * B.2^2 = 1) (h_perp : A.1 * B.1 + 3 * A.2 * B.2 = 0) :
  prod_max_min_dist A B hA hB h_perp = 2 * real.sqrt 3 / 3 :=
sorry

end max_min_dist_product_correct_l236_236001


namespace distinct_remainders_l236_236747

theorem distinct_remainders
  (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n)
  (h_div : n ∣ a^n - 1) :
  ∀ i j : ℕ, i ∈ (Finset.range n).image (· + 1) →
            j ∈ (Finset.range n).image (· + 1) →
            (a^i + i) % n = (a^j + j) % n →
            i = j :=
by
  intros i j hi hj h
  sorry

end distinct_remainders_l236_236747


namespace train_length_l236_236148

noncomputable def speed_kmh_to_ms (v_kmh : ℕ) : ℕ := v_kmh * 1000 / 3600

theorem train_length (speed_kmh : ℕ) (time_sec : ℕ) (length_m : ℕ) :
  speed_kmh = 36 → time_sec = 20 → length_m = speed_kmh_to_ms speed_kmh * time_sec → length_m = 200 :=
by
  intro h_speed h_time h_length
  rw [← h_speed, ← h_time] at h_length
  simp [speed_kmh_to_ms] at h_length
  exact h_length

end train_length_l236_236148


namespace min_x_plus_y_l236_236335

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) :
  x + y ≥ 16 :=
sorry

end min_x_plus_y_l236_236335


namespace count_palindromes_l236_236684

def is_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = n % 10)

theorem count_palindromes : 
  {n : ℕ | is_palindrome n}.to_finset.card = 90 := 
by
  sorry

end count_palindromes_l236_236684


namespace smallest_n_exists_l236_236774

theorem smallest_n_exists {a b : ℕ} (connected : bool)
  (h1 : ¬connected → (Nat.gcd (a + b) 15) = 1)
  (h2 : connected → (Nat.gcd (a + b) 15) > 1) : 
  ∃ n : ℕ, (∀ a b : ℕ, (¬connected → (Nat.gcd (a + b) n) = 1) ∧ (connected → (Nat.gcd (a + b) n) > 1)) ∧ n = 15 :=
by
  sorry

end smallest_n_exists_l236_236774


namespace correct_propositions_l236_236555

def P1 : Prop := ∀ x y : ℝ, (y^2 = 4 * x) → ((x - 1)^2 + y^2 = 1) → x = 0 ∧ y = 0

def P2 (m : ℝ) : Prop := 
  m = -2 → 
  ((m + 2) * x + m * y + 1 = 0 → y = (1/2) * x - 1) ∧
  ((m - 2) * x + (m + 2) * y - 3 = 0 → x = -3/2)

def P3 : Prop := 
  (∃ x : ℝ, x^2 + 3 * x + 4 = 0) ↔ 
  (∀ x : ℝ, x^2 + 3 * x + 4 ≠ 0)

def P4 : Prop :=
  ∀ x : ℝ, y = sin (2 * x) → 
  y = sin (2 * (x - (π / 3))) → 
  y = sin (2 * x - (2 * π / 3))

theorem correct_propositions : P1 ∧ P2 -2 ∧ P3 :=
by
  sorry

end correct_propositions_l236_236555


namespace calculate_oplus_l236_236689

def op (X Y : ℕ) : ℕ :=
  (X + Y) / 2

theorem calculate_oplus : op (op 6 10) 14 = 11 := by
  sorry

end calculate_oplus_l236_236689


namespace trajectory_midpoint_l236_236238

-- Defining the point A(-2, 0)
def A : ℝ × ℝ := (-2, 0)

-- Defining the curve equation
def curve (x y : ℝ) : Prop := 2 * y^2 = x

-- Coordinates of P based on the midpoint formula
def P (x y : ℝ) : ℝ × ℝ := (2 * x + 2, 2 * y)

-- The target trajectory equation
def trajectory_eqn (x y : ℝ) : Prop := x = 4 * y^2 - 1

-- The theorem to be proved
theorem trajectory_midpoint (x y : ℝ) :
  curve (2 * y) (2 * x + 2) → 
  trajectory_eqn x y :=
sorry

end trajectory_midpoint_l236_236238


namespace largest_n_l236_236491

def polynomial (n : ℕ) : ℤ := 9 * (n - 3)^5 - 2 * n^3 + 17 * n - 33

theorem largest_n : ∃ n < 100000, polynomial n % 7 = 0 ∧ ∀ m < 100000, polynomial m % 7 = 0 → m ≤ n :=
  exists.intro 99999 (and.intro (by
    rw [polynomial]
    sorry) (by
    intro m hm
    sorry))

end largest_n_l236_236491


namespace find_length_AC_l236_236420

variables {Ω : Type*} [metric_space Ω] {A B C : Ω}
variable radius : ℝ
variable {circle : set Ω}
variable on_circle : ∀ {x : Ω}, x ∈ circle → dist x (classical.some metric_space.exist_basis) = radius
variable radius_nonneg : 0 ≤ radius
variable tangent_point A_on_circle : A ∈ circle
variable tangent_length : dist A B = 65
variable C_on_circle : C ∈ circle
variable dist_BC : dist B C = 25

theorem find_length_AC (radius : ℝ) (tangent_length : dist A B = 65) (dist_BC : dist B C = 25)
  (radius_nonneg : 0 ≤ radius) (A_on_circle : A ∈ circle) (C_on_circle : C ∈ circle) :
  dist A C = 60 :=
sorry

end find_length_AC_l236_236420


namespace sum_and_nth_term_l236_236276

-- Definitions for sum of arithmetic progression and nth term
def arithmetic_sum (n : ℕ) (a d : ℝ) : ℝ := n / 2 * (2 * a + (n - 1) * d)

def nth_term (n : ℕ) (a d : ℝ) : ℝ := a + (n - 1) * d

-- Conditions
def cond1 : Prop := arithmetic_sum 5 a d = 10
def cond2 : Prop := arithmetic_sum 50 a d = 150

-- Proof problem
theorem sum_and_nth_term (a d : ℝ) (h1 : cond1) (h2 : cond2) :
  arithmetic_sum 55 a d = 171 ∧ nth_term 55 a d = 4.31 :=
by
  sorry

end sum_and_nth_term_l236_236276


namespace probability_of_not_all_same_number_l236_236904

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l236_236904


namespace simplify_sin_formula_l236_236433

theorem simplify_sin_formula : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := by
  -- Conditions and values used in the proof
  sorry

end simplify_sin_formula_l236_236433


namespace num_people_need_life_jackets_l236_236842

-- Definitions of conditions
def raft_capacity_no_life_jackets : ℕ := 21
def reduced_capacity_due_to_jackets : ℕ := 7
def raft_capacity_with_some_life_jackets (x : ℕ) : ℕ := 17
def space_per_person_with_life_jacket : ℝ := 21 / 14

-- Statement of the problem to prove
theorem num_people_need_life_jackets :
  ∃ x : ℕ, (raft_capacity_with_some_life_jackets x - x + x * (space_per_person_with_life_jacket)) = raft_capacity_no_life_jackets :=
sorry

end num_people_need_life_jackets_l236_236842


namespace tank_capacity_is_48_l236_236042

-- Define the conditions
def num_4_liter_bucket_used : ℕ := 12
def num_3_liter_bucket_used : ℕ := num_4_liter_bucket_used + 4

-- Define the capacities of the buckets and the tank
def bucket_4_liters_capacity : ℕ := 4 * num_4_liter_bucket_used
def bucket_3_liters_capacity : ℕ := 3 * num_3_liter_bucket_used

-- Tank capacity
def tank_capacity : ℕ := 48

-- Statement to prove
theorem tank_capacity_is_48 : 
    bucket_4_liters_capacity = tank_capacity ∧
    bucket_3_liters_capacity = tank_capacity := by
  sorry

end tank_capacity_is_48_l236_236042


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236932

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236932


namespace fourth_vertex_of_square_l236_236626

-- Conditions
def vertex1 : ℂ := 1 + I
def vertex2 : ℂ := -1 + 3 * I
def vertex3 : ℂ := -3 - I

-- Question & Correct Answer
theorem fourth_vertex_of_square (v1 v2 v3 : ℂ) (h1 : v1 = 1 + I) (h2 : v2 = -1 + 3 * I) (h3 : v3 = -3 - I) : 
  ∃ v4 : ℂ, (v1 = 1 + I ∨ v1 = -1 + 3 * I ∨ v1 = -3 - I ∨ v1 = v4) ∧
             (v2 = 1 + I ∨ v2 = -1 + 3 * I ∨ v2 = -3 - I ∨ v2 = v4) ∧
             (v3 = 1 + I ∨ v3 = -1 + 3 * I ∨ v3 = -3 - I ∨ v3 = v4) ∧
             v4 = -1 - I :=
by
  use -1 - I
  sorry

end fourth_vertex_of_square_l236_236626


namespace maximize_area_of_ABCD_l236_236750

-- Define the vertices of the quadrilateral
structure Point where
  x : ℝ
  y : ℝ

structure Quadrilateral where
  A B C D : Point

-- Side lengths
def side_AB (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.B.x - q.A.x)^2 + (q.B.y - q.A.y)^2)

def side_BC (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.C.x - q.B.x)^2 + (q.C.y - q.B.y)^2)

def side_CD (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.D.x - q.C.x)^2 + (q.D.y - q.C.y)^2)

-- Centroids of the triangles
def centroid_triangle_ABC (q : Quadrilateral) : Point :=
  ⟨(q.A.x + q.B.x + q.C.x) / 3, (q.A.y + q.B.y + q.C.y) / 3⟩

def centroid_triangle_BCD (q : Quadrilateral) : Point :=
  ⟨(q.B.x + q.C.x + q.D.x) / 3, (q.B.y + q.C.y + q.D.y) / 3⟩

def centroid_triangle_ABD (q : Quadrilateral) : Point :=
  ⟨(q.A.x + q.B.x + q.D.x) / 3, (q.A.y + q.B.y + q.D.y) / 3⟩

-- Check if centroids form an equilateral triangle
def centroids_form_equilateral (q : Quadrilateral) : Prop :=
  let c1 := centroid_triangle_ABC q
  let c2 := centroid_triangle_BCD q
  let c3 := centroid_triangle_ABD q
  (Real.sqrt ((c1.x - c2.x)^2 + (c1.y - c2.y)^2) = Real.sqrt ((c2.x - c3.x)^2 + (c2.y - c3.y)^2)) ∧
  (Real.sqrt ((c2.x - c3.x)^2 + (c2.y - c3.y)^2) = Real.sqrt ((c3.x - c1.x)^2 + (c3.y - c1.y)^2))

-- Define the function to calculate the area of the quadrilateral
def area (q : Quadrilateral) : ℝ :=
  let A := (q.A.x * (q.B.y - q.D.y) + q.B.x * (q.D.y - q.A.y) + q.D.x * (q.A.y - q.B.y)) / 2
  let B := (q.B.x * (q.C.y - q.A.y) + q.C.x * (q.A.y - q.B.y) + q.A.x * (q.B.y - q.C.y)) / 2
  let C := (q.C.x * (q.D.y - q.B.y) + q.D.x * (q.B.y - q.C.y) + q.B.x * (q.C.y - q.D.y)) / 2
  Real.abs (A + B + C)

-- Maximize area
theorem maximize_area_of_ABCD : ∃ (q : Quadrilateral), side_AB q = 3 ∧ side_BC q = 4 ∧ side_CD q = 5 ∧ centroids_form_equilateral q ∧ area q = 27 := sorry

end maximize_area_of_ABCD_l236_236750


namespace corresponding_angles_not_universally_true_l236_236824

theorem corresponding_angles_not_universally_true : 
  ∀ (l1 l2 : Line) (a b : Angle), ¬ (corresponding_angles l1 l2 a b → a = b) := by
  sorry

end corresponding_angles_not_universally_true_l236_236824


namespace find_x_when_y_64_l236_236837

theorem find_x_when_y_64 (x y k : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_inv_prop : x^3 * y = k) (h_given : x = 2 ∧ y = 8 ∧ k = 64) :
  y = 64 → x = 1 :=
by
  sorry

end find_x_when_y_64_l236_236837


namespace det_example_l236_236222

theorem det_example : (1 * 4 - 2 * 3) = -2 :=
by
  -- Skip the proof with sorry
  sorry

end det_example_l236_236222


namespace total_earnings_l236_236119

def wage_per_person : ℕ → ℕ := λ w, 14 * w

theorem total_earnings (W B : ℕ) (hW : 5 = W) (hB : W = 8) : 
  wage_per_person 5 + wage_per_person W + wage_per_person B = 210 :=
by 
  sorry

end total_earnings_l236_236119


namespace ratio_child_to_jane_babysit_l236_236375

-- Definitions of the conditions
def jane_current_age : ℕ := 32
def years_since_jane_stopped_babysitting : ℕ := 10
def oldest_person_current_age : ℕ := 24

-- Derived definitions
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped_babysitting
def oldest_person_age_when_jane_stopped : ℕ := oldest_person_current_age - years_since_jane_stopped_babysitting

-- Statement of the problem to be proven in Lean 4
theorem ratio_child_to_jane_babysit :
  (oldest_person_age_when_jane_stopped : ℚ) / (jane_age_when_stopped : ℚ) = 7 / 11 :=
by
  sorry

end ratio_child_to_jane_babysit_l236_236375


namespace time_to_pass_tree_l236_236947

def train_length : ℝ := 240 -- in meters
def train_speed_kmh : ℝ := 108 -- in kilometers per hour

def kmh_to_ms (v : ℝ) : ℝ := v * 1000 / 3600 -- conversion from km/hr to m/s

def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

theorem time_to_pass_tree : train_length / train_speed_ms = 8 :=
by
  -- Lean proof would go here
  sorry

end time_to_pass_tree_l236_236947


namespace four_digit_numbers_with_property_l236_236304

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l236_236304


namespace red_unit_cubes_exist_l236_236763

noncomputable def red_points_problem
  (n k : ℕ) 
  (T : set (ℤ × ℤ × ℤ)) 
  (R : set (ℤ × ℤ × ℤ)) : Prop :=
  let total_red_points := 3 * n^2 - 3 * n + 1 + k in
  let unit_cube_vertices := 
    { (xi, yi, zi) | xi, yi, zi ∈ T ∧ ∀ x x', (xi, yi, zi) ∈ R ∧ (x', yi, zi) ∈ R
    ∧ ∀ y y', (xi, yi, zi) ∈ R ∧ (xi, y', zi) ∈ R ∧ ∀ z z', (xi, yi, zi) ∈ R
    ∧ (xi, yi, z') ∈ R } in
  (3 * n^2 - 3 * n + 1) + k ≤ box.num_points && 
  all_points_on_line_R ||
  k > 0 &&
  ∀x y z, total_red_points ∧ exists! x unit_cube_vertices ≥ k ∧ R else sorry
  
theorem red_unit_cubes_exist
  (n k : ℤ) 
  (T : set (ℤ × ℤ × ℤ)) 
  (R : set (ℤ × ℤ × ℤ))
  (H1 : 0 < n) 
  (H2 : 0 < k)
  (H3 : T = { p | ∃ x y z, p = (x, y, z) ∧ 1 ≤ x ∧ x ≤ n ∧ 1 ≤ y ∧ y ≤ n ∧ 1 ≤ z ∧ z ≤ n }) 
  (H4 : ∃ S, S.card = 3 * n^2 - 3 * n + 1 + k ∧ S ⊆ R ∧ ∀ P Q : ℤ × ℤ × ℤ, P ∈ R ∧ Q ∈ R ∧ (∃ i : {0, 1, 2}, P = Q) ∧ (P, Q) ∈ set.prod (set.univ) (set.univ)) : 
  ∃ c ≥  k := 
    begin 
      let box.num_points in sorry
    end

end red_unit_cubes_exist_l236_236763


namespace shirt_final_price_percent_l236_236107

theorem shirt_final_price_percent (original_price : ℝ) :
  let sale_price := original_price * 0.7 in
  let final_price := sale_price * 0.9 in
  final_price / original_price = 0.63 :=
by
  -- Let's define our variables
  let sale_price := original_price * 0.7
  let final_price := sale_price * 0.9
  -- The proof will follow from these definitions
  sorry

end shirt_final_price_percent_l236_236107


namespace students_swimming_class_l236_236478

/-- Given:
1. There are 1000 students in a school.
2. 25% of the students attend chess class.
3. 50% of the students in the chess class are enrolled for swimming.
4. All enrolled students attend the swimming class.

Prove that the number of students attending the swimming class is 125.
-/
theorem students_swimming_class (total_students : ℕ) (chess_percent : ℝ) (swimming_percent : ℝ) :
  total_students = 1000 →
  chess_percent = 0.25 →
  swimming_percent = 0.50 →
  (total_students * chess_percent * swimming_percent).toNat = 125 :=
by
  intros h1 h2 h3
  sorry

end students_swimming_class_l236_236478


namespace problem_solution_l236_236430

-- Define the given curves and transformations.
def curve_B (x y: ℝ) : Prop := x^2 + y^2 = 1

def transform_x (x: ℝ) : ℝ := 3 * x
def transform_y (y: ℝ) : ℝ := y

def curve_C (x' y': ℝ) : Prop := (x' / 3)^2 + y'^2 = 1

-- Define the line and the distance function.
def line_l (x y: ℝ) : Prop := x + 4 * y - 8 = 0

def distance_to_line (x y: ℝ) : ℝ := abs (x + 4 * y - 8) / real.sqrt 17

-- Define the parametric form of point D on curve C.
def point_D (θ: ℝ) : ℝ × ℝ := (3 * real.cos θ, real.sin θ)

-- The minimum distance condition translated:
def min_distance_condition (φ θ: ℝ) : Prop := real.sin (φ + θ) = 1

-- Prove the given statements.
theorem problem_solution:
  (∀ (x y: ℝ), curve_B x y → curve_C (transform_x x) (transform_y y)) ∧
  (∃ (θ: ℝ), (point_D θ) = (3 * real.cos θ, real.sin θ) ∧ distance_to_line (3 * real.cos θ) (real.sin θ) = (5 * abs (real.sin (φ + θ) - 8)) / real.sqrt 17 ∧ min_distance_condition φ θ ∧ 
     point_D θ = (9/5, 4/5)) :=
by
  sorry

end problem_solution_l236_236430


namespace intersection_product_of_circles_l236_236938

theorem intersection_product_of_circles :
  (∀ x y : ℝ, (x^2 + 2 * x + y^2 + 4 * y + 5 = 0) ∧ (x^2 + 6 * x + y^2 + 4 * y + 9 = 0) →
  x * y = 2) :=
sorry

end intersection_product_of_circles_l236_236938


namespace largest_x_satisfying_equation_l236_236864

theorem largest_x_satisfying_equation :
  ∃ x : ℝ, x = 3 / 25 ∧ (∀ y, (y : ℝ) ∈ {z | sqrt (3 * z) = 5 * z} → y ≤ x) :=
by
  sorry

end largest_x_satisfying_equation_l236_236864


namespace solution1_solution2_l236_236175

noncomputable def problem1 : ℝ :=
  40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12

theorem solution1 : problem1 = 43 := by
  sorry

noncomputable def problem2 : ℝ :=
  (-1 : ℝ) ^ 2021 + |(-9 : ℝ)| * (2 / 3) + (-3) / (1 / 5)

theorem solution2 : problem2 = -10 := by
  sorry

end solution1_solution2_l236_236175


namespace P_neg2_lt_xi_lt_0_l236_236660

variables {Ω : Type*} [ProbabilitySpace Ω]

noncomputable def standard_normal : MeasureTheory.Measure Ω := sorry -- placeholder for the standard normal distribution

variable (ξ : Ω → ℝ)

axiom normal_dist (hξ : ∀ ω, ξ ω ∼ (MeasureTheory.ProbabilityMeasure standard_normal)) : 
  MeasureTheory.ProbabilityMeasure standard_normal ξ

axiom P_xi_gt_2 (hξ : ∀ ω, ξ ω ∼ (MeasureTheory.ProbabilityMeasure standard_normal)) :
  MeasureTheory.Probability (MeasureTheory.measurable_set (λ ω, 2 < ξ ω)) = p

theorem P_neg2_lt_xi_lt_0 (hξ : ∀ ω, ξ ω ∼ (MeasureTheory.ProbabilityMeasure standard_normal)) :
  MeasureTheory.Probability (MeasureTheory.measurable_set (λ ω, (-2:ℝ) < ξ ω ∧ ξ ω < 0)) = 1/2 - p :=
sorry

end P_neg2_lt_xi_lt_0_l236_236660


namespace milton_apple_pie_slices_l236_236168

theorem milton_apple_pie_slices :
  ∀ (A : ℕ),
  (∀ (peach_pie_slices_per : ℕ), peach_pie_slices_per = 6) →
  (∀ (apple_pie_slices_sold : ℕ), apple_pie_slices_sold = 56) →
  (∀ (peach_pie_slices_sold : ℕ), peach_pie_slices_sold = 48) →
  (∀ (total_pies_sold : ℕ), total_pies_sold = 15) →
  (∃ (apple_pie_slices : ℕ), apple_pie_slices = 56 / (total_pies_sold - (peach_pie_slices_sold / peach_pie_slices_per))) → 
  A = 8 :=
by sorry

end milton_apple_pie_slices_l236_236168


namespace domain_of_tan_sub_pi_over_4_l236_236448

theorem domain_of_tan_sub_pi_over_4 :
  ∀ x : ℝ, (∃ k : ℤ, x = k * π + 3 * π / 4) ↔ ∃ y : ℝ, y = (x - π / 4) ∧ (∃ k : ℤ, y = (2 * k + 1) * π / 2) := 
sorry

end domain_of_tan_sub_pi_over_4_l236_236448


namespace part_a_part_b_part_c_l236_236985

-- Declare the existence of nine distinct squares that form a rectangle
theorem part_a : ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ), 
  (a_1 ≠ a_2 ∧ a_1 ≠ a_3 ∧ a_1 ≠ a_4 ∧ a_1 ≠ a_5 ∧ a_1 ≠ a_6 ∧ a_1 ≠ a_7 ∧ a_1 ≠ a_8 ∧ a_1 ≠ a_9 ∧
   a_2 ≠ a_3 ∧ a_2 ≠ a_4 ∧ a_2 ≠ a_5 ∧ a_2 ≠ a_6 ∧ a_2 ≠ a_7 ∧ a_2 ≠ a_8 ∧ a_2 ≠ a_9 ∧
   a_3 ≠ a_4 ∧ a_3 ≠ a_5 ∧ a_3 ≠ a_6 ∧ a_3 ≠ a_7 ∧ a_3 ≠ a_8 ∧ a_3 ≠ a_9 ∧
   a_4 ≠ a_5 ∧ a_4 ≠ a_6 ∧ a_4 ≠ a_7 ∧ a_4 ≠ a_8 ∧ a_4 ≠ a_9 ∧
   a_5 ≠ a_6 ∧ a_5 ≠ a_7 ∧ a_5 ≠ a_8 ∧ a_5 ≠ a_9 ∧
   a_6 ≠ a_7 ∧ a_6 ≠ a_8 ∧ a_6 ≠ a_9 ∧
   a_7 ≠ a_8 ∧ a_7 ≠ a_9 ∧
   a_8 ≠ a_9) ∧ 
  (∃ (l_1 l_2 : ℕ), l_1 * l_2 = (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9)) := 
sorry

-- Declare the property for 10 distinct squares forming a specific rectangle
theorem part_b : ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℕ),
  (a_1 ≠ a_2 ∧ a_1 ≠ a_3 ∧ a_1 ≠ a_4 ∧ a_1 ≠ a_5 ∧ a_1 ≠ a_6 ∧ a_1 ≠ a_7 ∧ a_1 ≠ a_8 ∧ a_1 ≠ a_9 ∧ a_1 ≠ a_10 ∧
   a_2 ≠ a_3 ∧ a_2 ≠ a_4 ∧ a_2 ≠ a_5 ∧ a_2 ≠ a_6 ∧ a_2 ≠ a_7 ∧ a_2 ≠ a_8 ∧ a_2 ≠ a_9 ∧ a_2 ≠ a_10 ∧
   a_3 ≠ a_4 ∧ a_3 ≠ a_5 ∧ a_3 ≠ a_6 ∧ a_3 ≠ a_7 ∧ a_3 ≠ a_8 ∧ a_3 ≠ a_9 ∧ a_3 ≠ a_10 ∧
   a_4 ≠ a_5 ∧ a_4 ≠ a_6 ∧ a_4 ≠ a_7 ∧ a_4 ≠ a_8 ∧ a_4 ≠ a_9 ∧ a_4 ≠ a_10 ∧
   a_5 ≠ a_6 ∧ a_5 ≠ a_7 ∧ a_5 ≠ a_8 ∧ a_5 ≠ a_9 ∧ a_5 ≠ a_10 ∧
   a_6 ≠ a_7 ∧ a_6 ≠ a_8 ∧ a_6 ≠ a_9 ∧ a_6 ≠ a_10 ∧
   a_7 ≠ a_8 ∧ a_7 ≠ a_9 ∧ a_7 ≠ a_10 ∧
   a_8 ≠ a_9 ∧ a_8 ≠ a_10 ∧
   a_9 ≠ a_10) ∧
  (∃ (a b : ℕ), a * b = 47 * 65) :=
sorry

-- Declare the general case for any \( n > 8 \)
theorem part_c (n : ℕ) (h : n > 8) : ∃ (a : fin n → ℕ),
  (∀ i j : fin n, i ≠ j → a i ≠ a j) ∧ ∃ (l_1 l_2 : ℕ), l_1 * l_2 = (∑ i, a i) :=
sorry

end part_a_part_b_part_c_l236_236985


namespace correct_calculation_l236_236945

theorem correct_calculation (x : ℝ) (h : 3 * x - 12 = 60) : (x / 3) + 12 = 20 :=
by 
  sorry

end correct_calculation_l236_236945


namespace max_3a_b_equals_4_l236_236388

noncomputable def max_value_3a_b (a b : ℝ) : ℝ :=
if 9 * a^2 + b^2 - 6 * a - 2 * b = 0 then 3 * a + b else -∞

theorem max_3a_b_equals_4
  (a b : ℝ)
  (h : 9 * a^2 + b^2 - 6 * a - 2 * b = 0) :
  ∃ a_max b_max : ℝ, max_value_3a_b a b = 4 :=
sorry

end max_3a_b_equals_4_l236_236388


namespace avg_score_first_4_l236_236803

-- Definitions based on conditions
def average_score_all_7 : ℝ := 56
def total_matches : ℕ := 7
def average_score_last_3 : ℝ := 69.33333333333333
def matches_first : ℕ := 4
def matches_last : ℕ := 3

-- Calculation of total runs from average scores.
def total_runs_all_7 : ℝ := average_score_all_7 * total_matches
def total_runs_last_3 : ℝ := average_score_last_3 * matches_last

-- Total runs for the first 4 matches
def total_runs_first_4 : ℝ := total_runs_all_7 - total_runs_last_3

-- Prove the average score for the first 4 matches.
theorem avg_score_first_4 :
  (total_runs_first_4 / matches_first) = 46 := 
sorry

end avg_score_first_4_l236_236803


namespace trapezoid_non_parallel_sides_equal_length_l236_236149

theorem trapezoid_non_parallel_sides_equal_length (radius length_parallel length_non_parallel : ℝ) 
  (h_circle : radius = 300)
  (h_parallel : length_parallel = 150)
  (h_inscribed : ∀ ABCD : set ℝ, (ABCD ⊆ ∂ball 0 radius) → is_trapezoid ABCD)
  (h_equal_sides : ∀ ABCD : set ℝ, is_trapezoid ABCD → BC = DA) :
  length_non_parallel = 300 :=
sorry

end trapezoid_non_parallel_sides_equal_length_l236_236149


namespace count_valid_numbers_l236_236313
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l236_236313


namespace nine_possible_xs_l236_236287

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end nine_possible_xs_l236_236287


namespace smallest_possible_value_of_d_l236_236427

theorem smallest_possible_value_of_d (c d : ℝ) (hc : 1 < c) (hd : c < d)
  (h_triangle1 : ¬(1 + c > d ∧ c + d > 1 ∧ 1 + d > c))
  (h_triangle2 : ¬(1 / c + 1 / d > 1 ∧ 1 / d + 1 > 1 / c ∧ 1 / c + 1 > 1 / d)) :
  d = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_d_l236_236427


namespace barry_minimal_money_l236_236176

theorem barry_minimal_money (num_coins : ℕ)
    (denoms : list ℝ)
    (counts : list ℕ)
    (h1 : num_coins = 12)
    (h2 : denoms = [2.00, 1.00, 0.25, 0.10, 0.05])
    (h3 : ∀ d ∈ denoms, ∃ c ∈ counts, c ≥ 1)
    : ∃ value : ℝ, value = 3.75 :=
by
  sorry

end barry_minimal_money_l236_236176


namespace num_positive_x_count_num_positive_x_l236_236292

theorem num_positive_x (x : ℕ) : (3 * x < 100) ∧ (4 * x ≥ 100) → x ≥ 25 ∧ x ≤ 33 := by
  sorry

theorem count_num_positive_x : 
  (∃ x : ℕ, (3 * x < 100) ∧ (4 * x ≥ 100)) → 
  (finset.range 34).filter (λ x, (3 * x < 100 ∧ 4 * x ≥ 100)).card = 9 := by
  sorry

end num_positive_x_count_num_positive_x_l236_236292


namespace vitali_hahn_saks_part_a_vitali_hahn_saks_part_b_l236_236382

-- Define the necessary objects and conditions
variables {Ω : Type*} {ℱ : set (set Ω)}
variables (P : (set Ω) → ℝ) (Pn : ℕ → (set Ω) → ℝ)

-- Measurable space and probability measure sequence
def is_measurable_space (Ω : Type*) (ℱ : set (set Ω)) := True  -- Abstract measurable space definition
def prob_meas (P : (set Ω) → ℝ) := True  -- Abstract probability measure definition

-- Convergence condition
axiom prob_convergence (h : ∀ A ∈ ℱ, tendsto (λ n, Pn n A) at_top (𝓝 (P A)))

-- Vitali-Hahn-Saks theorem part (a)
theorem vitali_hahn_saks_part_a 
  (meas_space : is_measurable_space Ω ℱ)
  (seq_prob_meas : ∀ n, prob_meas (Pn n))
  (h : ∀ A ∈ ℱ, tendsto (λ n, Pn n A) at_top (𝓝 (P A))) :
  prob_meas P :=
sorry

-- Define decreasing sequence and limit supremum condition
def decreasing_sequence {Ω : Type*} (ℱ : set (set Ω)) (A : ℕ → set Ω) :=
  ∀ k, A (k + 1) ⊆ A k ∧ A k ∈ ℱ

def sup_lim_zero {Pn : ℕ → (set Ω) → ℝ} (A : ℕ → set Ω) :=
  tendsto (λ k, supr (λ n, Pn n (A k))) at_top (𝓝 0)

-- Vitali-Hahn-Saks theorem part (b)
theorem vitali_hahn_saks_part_b 
  (meas_space : is_measurable_space Ω ℱ)
  (seq_prob_meas : ∀ n, prob_meas (Pn n))
  (h : ∀ A ∈ ℱ, tendsto (λ n, Pn n A) at_top (𝓝 (P A)))
  (A : ℕ → set Ω)
  (decr_seq : decreasing_sequence ℱ A)
  (h_empty : tendsto (λ k, A k) at_top (𝓝 ∅)) :
  sup_lim_zero A :=
sorry

end vitali_hahn_saks_part_a_vitali_hahn_saks_part_b_l236_236382


namespace combined_sets_range_is_91_l236_236788

def two_digit_primes : Set ℕ := { n | 10 ≤ n ∧ n ≤ 99 ∧ Nat.Prime n }
def multiples_of_6_less_than_100 : Set ℕ := { n | 0 < n ∧ n < 100 ∧ n % 6 = 0 }

def combined_set_range {S T : Set ℕ} : ℕ := (S ∪ T).to_finset.sup id - (S ∪ T).to_finset.inf id

theorem combined_sets_range_is_91 :
  combined_set_range two_digit_primes multiples_of_6_less_than_100 = 91 :=
sorry

end combined_sets_range_is_91_l236_236788


namespace solve_system_l236_236389

theorem solve_system (c d : ℝ) 
  (h1 : (¬∃ x : ℝ, -c = -5 ∨ -d = -5))
  (h2 : -d ≠ -10)
  (h3 : ∃! x : ℝ, (∃ x : ℝ, x + 3 * c = 0) ∨ (∃ x : ℝ, x + 2 = 0) ∨ (∃ x : ℝ, x + 4 = 0) ∧ ¬(x = -d ∨ x = -10)) :
  50 * c + 10 * d = 103.33 :=
begin
  sorry
end

end solve_system_l236_236389


namespace probability_two_or_fewer_distinct_digits_l236_236987

def digits : Set ℕ := {1, 2, 3}

def total_3_digit_numbers : ℕ := 27

def distinct_3_digit_numbers : ℕ := 6

def at_most_two_distinct_numbers : ℕ := total_3_digit_numbers - distinct_3_digit_numbers

theorem probability_two_or_fewer_distinct_digits :
  (at_most_two_distinct_numbers : ℚ) / total_3_digit_numbers = 7 / 9 := by
  sorry

end probability_two_or_fewer_distinct_digits_l236_236987


namespace intersecting_circles_range_of_m_l236_236578

theorem intersecting_circles_range_of_m
  (x y m : ℝ)
  (C₁_eq : x^2 + y^2 - 2 * m * x + m^2 - 4 = 0)
  (C₂_eq : x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0)
  (intersect : ∃ x y : ℝ, (x^2 + y^2 - 2 * m * x + m^2 - 4 = 0) ∧ (x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0))
  : m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := 
sorry

end intersecting_circles_range_of_m_l236_236578


namespace sum_of_possible_values_of_x_l236_236394

def f (x : ℝ) : ℝ :=
if x < 1 then 5 * x + 10 else 3 * x - 9

theorem sum_of_possible_values_of_x (x1 x2 : ℝ) (h1 : f x1 = 1) (h2 : f x2 = 1) (hx1 : x1 < 1) (hx2 : x2 ≥ 1) :
  x1 + x2 = 23 / 15 :=
sorry

end sum_of_possible_values_of_x_l236_236394


namespace total_cookies_l236_236574

variable (glenn_cookies : ℕ) (kenny_cookies : ℕ) (chris_cookies : ℕ)
hypothesis (h1 : glenn_cookies = 24)
hypothesis (h2 : glenn_cookies = 4 * kenny_cookies)
hypothesis (h3 : chris_cookies = kenny_cookies / 2)

theorem total_cookies : glenn_cookies + kenny_cookies + chris_cookies = 33 :=
by sorry

end total_cookies_l236_236574


namespace factory_pass_rate_nine_possible_l236_236258

theorem factory_pass_rate_nine_possible :
  let n := 10
  let p := 0.9
  ∃ k, k = 9 ∧ (probability (binomial n p) k > 0) :=
by
  sorry

end factory_pass_rate_nine_possible_l236_236258


namespace five_eight_sided_dice_not_all_same_l236_236893

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l236_236893


namespace trees_needed_on_road_l236_236946

theorem trees_needed_on_road (length interval : ℕ) (starts_ends : bool) :
  length = 100 ∧ interval = 10 ∧ starts_ends = true → 
  let N := (length / interval) + 1 in
  N = 11 :=
by
  intros
  sorry

end trees_needed_on_road_l236_236946


namespace total_cookies_l236_236569

variable (ChrisCookies KennyCookies GlennCookies : ℕ)
variable (KennyHasCookies : GlennCookies = 4 * KennyCookies)
variable (ChrisHasCookies : ChrisCookies = KennyCookies / 2)
variable (GlennHas24Cookies : GlennCookies = 24)

theorem total_cookies : GlennCookies + KennyCookies + ChrisCookies = 33 := 
by
  have KennyCookiesEq : KennyCookies = 24 / 4 := by 
    rw [GlennHas24Cookies, mul_div_cancel_left, nat.mul_comm, nat.one_div, nat.div_self] ; trivial
  have ChrisCookiesEq : ChrisCookies = 6 / 2 := by 
    rw [KennyCookiesEq, ChrisHasCookies]
  rw [ChrisCookiesEq, KennyCookiesEq, GlennHas24Cookies]
  exact sorry

end total_cookies_l236_236569


namespace factorial_sum_remainder_l236_236566

theorem factorial_sum_remainder :
  (1! + 2! + 3! + 4! + 5! + 6!) % 60 = 33 := 
by 
  sorry

end factorial_sum_remainder_l236_236566


namespace rotten_apples_did_not_smell_l236_236414

theorem rotten_apples_did_not_smell (total_apples : ℕ) (rotten_percentage smell_percentage : ℕ) :
  total_apples = 200 →
  rotten_percentage = 40 →
  smell_percentage = 70 →
  let rotten_apples := (rotten_percentage * total_apples) / 100 in
  let smelling_rotten_apples := (smell_percentage * rotten_apples) / 100 in
  let non_smelling_rotten_apples := rotten_apples - smelling_rotten_apples in
  non_smelling_rotten_apples = 24 :=
by {
  intros h1 h2 h3,
  have h_rotten_apples : rotten_apples = 80,
  { calc
      rotten_apples
          = (rotten_percentage * total_apples) / 100 : by rw [←h2, ←h1, nat.mul_div_cancel' (by norm_num : total_apples % 100 = 0)]
      ... = 80 : by norm_num },
  have h_smelling_rotten_apples : smelling_rotten_apples = 56,
  { calc
      smelling_rotten_apples
          = (smell_percentage * rotten_apples) / 100 : by rw [←h3, ←h_rotten_apples, nat.mul_div_cancel' (by norm_num : rotten_apples % 100 = 0)]
      ... = 56 : by norm_num },
  have h_non_smelling_rotten_apples : non_smelling_rotten_apples = 24,
  { calc
      non_smelling_rotten_apples
          = rotten_apples - smelling_rotten_apples : rfl
      ... = 80 - 56 : by rw [h_rotten_apples, h_smelling_rotten_apples]
      ... = 24 : by norm_num },
  exact h_non_smelling_rotten_apples,
}

end rotten_apples_did_not_smell_l236_236414


namespace quadratic_eqn_l236_236255

-- Define the problem conditions
def vertex : ℝ × ℝ := (-1, 4)
def point : ℝ × ℝ := (2, -5)

-- Define the quadratic function with given vertex and some coefficient
def quadratic (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)^2 + 4

-- The final proof statement
theorem quadratic_eqn : ∃ a, ∀ x, (vertex, point) = ((-1, 4), (2, -5)) → quadratic a x = -x^2 - 2x + 3 :=
by
  sorry

end quadratic_eqn_l236_236255


namespace problem_solution_l236_236223

def count_valid_n : ℕ :=
  let count_mult_3 := (3000 / 3)
  let count_mult_6 := (3000 / 6)
  count_mult_3 - count_mult_6

theorem problem_solution : count_valid_n = 500 := 
sorry

end problem_solution_l236_236223


namespace lower_limit_of_range_l236_236140

theorem lower_limit_of_range (x y : ℝ) (hx1 : 3 < x) (hx2 : x < 8) (hx3 : y < x) (hx4 : x < 10) (hx5 : x = 7) : 3 < y ∧ y ≤ 7 :=
by
  sorry

end lower_limit_of_range_l236_236140


namespace count_possible_values_of_x_l236_236282

theorem count_possible_values_of_x :
  let n := (set.count {x : ℕ | 25 ≤ x ∧ x ≤ 33 ∧ ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c * x < 100 ∧ 3 ≤ b ≤ 100/x}) in
  n = 9 :=
by
  -- Here we must prove the statement by the provided conditions
  sorry

end count_possible_values_of_x_l236_236282


namespace asymptotes_of_hyperbola_l236_236816

-- Definitions
variables (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)

-- Theorem: Equation of the asymptotes of the given hyperbola
theorem asymptotes_of_hyperbola (h_equiv : b = 2 * a) :
  ∀ x y : ℝ, 
    (x ≠ 0 ∧ y ≠ 0 ∧ (y = (2 : ℝ) * x ∨ y = - (2 : ℝ) * x)) ↔ (x, y) ∈ {p : ℝ × ℝ | (x^2 / a^2) - (y^2 / b^2) = 1} := 
sorry

end asymptotes_of_hyperbola_l236_236816


namespace two_complex_numbers_cannot_be_compared_in_size_l236_236943

theorem two_complex_numbers_cannot_be_compared_in_size (z1 z2 : Complex) (h1 : z1.im ≠ 0 ∨ z2.im ≠ 0) : ¬ (z1, z2 : ℂ) -> (z1 ≤ z2 ∨ z2 ≤ z1) :=
by
  sorry

end two_complex_numbers_cannot_be_compared_in_size_l236_236943


namespace books_on_shelf_after_removal_l236_236057

theorem books_on_shelf_after_removal :
  let initial_books : ℝ := 38.0
  let books_removed : ℝ := 10.0
  initial_books - books_removed = 28.0 :=
by 
  sorry

end books_on_shelf_after_removal_l236_236057


namespace probability_not_all_same_l236_236901

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l236_236901


namespace first_year_sum_of_digits_15_l236_236081

theorem first_year_sum_of_digits_15 : ∃ y : ℕ, y > 2000 ∧ sum_of_digits y = 15 ∧ ∀ z, (z > 2000 ∧ sum_of_digits z = 15) → y ≤ z :=
by
  sorry

-- Helper function to calculate the sum of digits of a given number.
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

end first_year_sum_of_digits_15_l236_236081


namespace percentage_gain_correct_l236_236142

noncomputable def percentage_gain : ℝ :=
  let bought_bowls := 114
  let cp_per_bowl := 13
  let sold_bowls := 108
  let sp_per_bowl := 17
  let total_cost_price := bought_bowls * cp_per_bowl
  let total_selling_price := sold_bowls * sp_per_bowl
  let profit := total_selling_price - total_cost_price
  (profit / total_cost_price) * 100

theorem percentage_gain_correct : percentage_gain ≈ 23.88 :=
by
  sorry

end percentage_gain_correct_l236_236142


namespace length_AD_l236_236025

-- Define the given conditions
def AB : ℝ := 6
def BC : ℝ := 8
def CD : ℝ := 15
def sin_C : ℝ := 4 / 5
def neg_cos_B : ℝ := -4 / 5
def angle_is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Hypothesize the obtuse angles
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := sorry
axiom B_obtuse : angle_is_obtuse B
axiom C_obtuse : angle_is_obtuse C
axiom sin_C_is_correct : real.sin C = sin_C
axiom cos_B_is_correct : real.cos B = -neg_cos_B

-- The statement to prove: length of side AD is 12.95
theorem length_AD : 
  let AD : ℝ := 12.95 in
  ∀ (AB BC CD : ℝ) (sin_C neg_cos_B : ℝ) (angle_is_obtuse : ℝ → Prop) (B C AD : ℝ),
    AB = 6 ∧ BC = 8 ∧ CD = 15 ∧ sin_C = 4 / 5 ∧ neg_cos_B = -4 / 5 ∧
    angle_is_obtuse B ∧ angle_is_obtuse C ∧ real.sin C = sin_C ∧ real.cos B = -neg_cos_B
    → 
    AD = 12.95 := sorry

end length_AD_l236_236025


namespace solve_quadratic_l236_236328

theorem solve_quadratic (x : ℝ) : x^2 - 4 * x + 3 = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end solve_quadratic_l236_236328


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236856

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236856


namespace max_sides_cross_section_of_cube_l236_236552

theorem max_sides_cross_section_of_cube : ∀ (plane : ℝ^3 → Prop) (cube : ℝ^3 → Prop), 
  (∀ p : ℝ^3, cube p → ∃ side: ℕ, 1 ≤ side ∧ side ≤ 6 ∧ sides_of_cross_section plane cube ≤ side) →
  sides_of_cross_section plane cube ≤ 6 :=
sorry

end max_sides_cross_section_of_cube_l236_236552


namespace largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236857

theorem largest_value_of_x_satisfying_sqrt3x_eq_5x : 
  ∃ (x : ℚ), sqrt (3 * x) = 5 * x ∧ (∀ y : ℚ, sqrt (3 * y) = 5 * y → y ≤ x) ∧ x = 3 / 25 :=
sorry

end largest_value_of_x_satisfying_sqrt3x_eq_5x_l236_236857


namespace plywood_cut_perimeter_difference_l236_236516

theorem plywood_cut_perimeter_difference :
  ∃ (rectangles : list (ℝ × ℝ)), 
    length rectangles = 5 ∧
    (∀ r ∈ rectangles, r.1 * r.2 = 10) ∧
    let perimeters := rectangles.map (λ r, 2 * r.1 + 2 * r.2) in
    |(max (perimeters) - min (perimeters))| = 8 :=
sorry

end plywood_cut_perimeter_difference_l236_236516


namespace largest_x_satisfies_eq_l236_236870

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236870


namespace find_d_l236_236696

theorem find_d (d : ℤ) (h : ∀ x : ℤ, 8 * x^3 + 23 * x^2 + d * x + 45 = 0 → 2 * x + 5 = 0) : 
  d = 163 := 
sorry

end find_d_l236_236696


namespace original_price_of_sarees_l236_236045

theorem original_price_of_sarees (sale_price : ℝ) (h_sale_price : sale_price = 381.48) : 
  ∃ P : ℝ, 0.85 * 0.88 * P = sale_price ∧ P ≈ 510 :=
by
  existsi (381.48 / (0.85 * 0.88))
  split
  { 
    sorry 
  }
  { 
    -- The exact proof about approximation can be formalized but here we skip this
    sorry 
  }

end original_price_of_sarees_l236_236045


namespace correct_options_count_l236_236727

noncomputable def A (n : ℕ) : ℝ := Real.sqrt n

theorem correct_options_count :
  let a := Real.frac (A 6),
      b := 3,
      c := 1 in
  (2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 ∧
   (a = Real.sqrt 6 - 2 → ¬ ((2 : ℝ) / a = Real.sqrt 6 + 2)) ∧ 
   (Real.sqrt 4 - Real.sqrt 3 ≠ 0 ∧ 
    b / (Real.sqrt 4 - Real.sqrt 3) - c / (Real.sqrt 3 + Real.sqrt 4) = 4 * Real.sqrt 3 + 4 → b = 3 * c) ∧ 
   ((Σ i in Finset.range 2023, 1 / (i + 2) * A (i + 1 + 1) + (i + 1) * A i) = 1 - (Real.sqrt 2023 / 2023)) 
) →
  2 :=
by sorry

end correct_options_count_l236_236727


namespace fifth_inequality_l236_236013

theorem fifth_inequality :
  1 + (1 / (2^2 : ℝ)) + (1 / (3^2 : ℝ)) + (1 / (4^2 : ℝ)) + (1 / (5^2 : ℝ)) + (1 / (6^2 : ℝ)) < (11 / 6 : ℝ) :=
by
  sorry

end fifth_inequality_l236_236013


namespace positive_difference_of_perimeters_l236_236518

theorem positive_difference_of_perimeters (L W : ℕ) (n : ℕ) 
    (hLW : L = 10) (hW : W = 5) (hn : n = 5) : 
    (let minPerimeter := min (2 * (L / n) + 2 * W) (2 * (W / n) + 2 * L) in
    let maxPerimeter := max (2 * (L / n) + 2 * W) (2 * (W / n) + 2 * L) in
    maxPerimeter - minPerimeter = 8) :=
by 
    sorry

end positive_difference_of_perimeters_l236_236518


namespace probability_of_inner_square_is_correct_l236_236409

-- Define the number of total squares on the chessboard
def total_squares := 10 * 10

-- Define the number of squares on the outermost two rows and columns
def outermost_row_squares := 2 * 10
def outermost_col_squares := 2 * 10
def double_counted_corners := 2 * 4

-- Calculate the total number of squares in the outermost two rows or columns
def outermost_squares := outermost_row_squares + outermost_col_squares - double_counted_corners

-- Define the number of squares not in the outermost two rows or columns
def inner_squares := total_squares - outermost_squares

-- Define the probability as a fraction
def probability := inner_squares.to_rat / total_squares.to_rat

-- Statement to be proven
theorem probability_of_inner_square_is_correct : probability = 17 / 25 := by
  sorry

end probability_of_inner_square_is_correct_l236_236409


namespace Maria_Ivanovna_solution_l236_236406

noncomputable def Maria_grades_problem : Prop :=
  let a : ℕ → ℕ := λ n, if n = 1 then 3
                        else if n = 2 then 8
                        else 2 * a (n - 1) + 2 * a (n - 2) in
  a 6 = 448

theorem Maria_Ivanovna_solution : Maria_grades_problem := by
  sorry

end Maria_Ivanovna_solution_l236_236406


namespace coin_loading_impossible_l236_236976

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l236_236976


namespace min_perimeter_triangle_BC_length_l236_236700

noncomputable def minBC_length (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : ℝ :=
  1 + real.sqrt(2) / 2

theorem min_perimeter_triangle_BC_length :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], 
  ∠ C A B = 60 ∧ (dist A B > 1) ∧ (dist B C = dist A B + 1 / 2) →
  dist A C = minBC_length A B C :=
by
  sorry

end min_perimeter_triangle_BC_length_l236_236700


namespace grid_configuration_count_l236_236122

theorem grid_configuration_count :
  let grid_size := 5
  let start_value := -3
  let end_value := 3
  let valid_difference_condition (a b : Int) : Prop := (a - b).abs = 1
  let is_adjacent (i j : Nat) (x y : Nat) : Prop := 
    ((i = x ∧ |j - y| = 1) ∨ (j = y ∧ |i - x| = 1))
  ∃ (f : Fin grid_size → Fin grid_size → Int), 
    (f 0 0 = start_value) ∧ 
    (f (grid_size - 1) (grid_size - 1) = end_value) ∧
    (∀ i j x y, is_adjacent i j x y → valid_difference_condition (f i j) (f x y)) ∧ 
    count_valid_configurations f = 250 :=
begin
  sorry
end

end grid_configuration_count_l236_236122


namespace max_sectional_area_distance_l236_236047

theorem max_sectional_area_distance (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
    let d := (a * (real.sqrt (8 * b^2 - a^2))) / (4 * b)
    d = (a * (real.sqrt (8 * b^2 - a^2))) / (4 * b) :=
by 
  -- Definitions for the parameters and assumptions used
  sorry

end max_sectional_area_distance_l236_236047


namespace log_equality_implication_l236_236665

theorem log_equality_implication (p q : ℝ) (hq : q ≠ 2) : (log p + log q = log (2 * p + 3 * q)) → (p = 3 * q / (q - 2)) :=
  sorry

end log_equality_implication_l236_236665


namespace eval_log32_4_l236_236596

noncomputable def log_base_change (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem eval_log32_4 : log_base_change 32 4 = 2 / 5 := 
by 
  sorry

end eval_log32_4_l236_236596


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236937

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236937


namespace compare_xyz_l236_236259

noncomputable def x := log 5 (1 / 2)
noncomputable def y := (1 / 2) ^ 0.1
noncomputable def z := 2 ^ (1 / 3)

theorem compare_xyz : x < y ∧ y < z :=
by sorry

end compare_xyz_l236_236259


namespace prob_at_least_three_min_k_l236_236797

noncomputable def prob_A := 0.6
noncomputable def prob_B := 0.5
noncomputable def prob_C := 0.5
noncomputable def prob_D := 0.4

/-- 
Problem 1: Calculate the probability that at least three people need the device on the same workday.
- We need to prove that the probability is 0.31.
- Given conditions:
  - The probabilities of four people (A, B, C, and D) needing the device are \(0.6, 0.5, 0.5, 0.4\), respectively.
  - The events of each person needing the device are independent of each other.
-/
theorem prob_at_least_three : 
  -- Given conditions
  (Prob_A = 0.6 ∧ Prob_B = 0.5 ∧ Prob_C = 0.5 ∧ Prob_D = 0.4) ∧ 
  (independent_events A B ∧ independent_events A C ∧ independent_events A D ∧ 
   independent_events B C ∧ independent_events B D ∧ independent_events C D) → 
  -- Proof of correctness
  prob_at_least_n_device(3) = 0.31 := sorry

/--
Problem 2: Find the minimum value of \( k \) such that the probability of more than \( k \) people needing the device is less than 0.1.
- We need to prove that the minimum value of \( k \) is 3.
- Given conditions:
  - The probabilities of four people (A, B, C, and D) needing the device are \(0.6, 0.5, 0.5, 0.4\), respectively.
  - The events of each person needing the device are independent of each other.
-/
theorem min_k (k : ℕ) : 
  -- Given conditions
  (Prob_A = 0.6 ∧ Prob_B = 0.5 ∧ Prob_C = 0.5 ∧ Prob_D = 0.4) ∧ 
  (independent_events A B ∧ independent_events A C ∧ independent_events A D ∧ 
   independent_events B C ∧ independent_events B D ∧ independent_events C D) →
  -- Finding the minimum k
  prob_more_than_k_device(k) < 0.1 ↔ k = 3 := sorry

end prob_at_least_three_min_k_l236_236797


namespace coefficient_x4_l236_236850

noncomputable def p (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 - 8 * x + 2
noncomputable def q (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 3

theorem coefficient_x4 :
  (∀ x : ℝ, (p x * q x).coeff 4 = -29) := sorry

end coefficient_x4_l236_236850


namespace find_three_digit_number_l236_236147

theorem find_three_digit_number : ∃ n : ℕ, n * 6 = 41 * 18 ∧ 100 ≤ n ∧ n < 1000 :=
by
  use 123
  split
  sorry
  split
  sorry
  sorry

end find_three_digit_number_l236_236147


namespace solution_satisfies_x_eq_2_l236_236832

theorem solution_satisfies_x_eq_2 :
  ∃ (eq : ℝ → Prop), eq 2 ∧ (eq = (λ x, - (1 / 3) * x + 2 / 3 = 0)) ∧ 
  (eq = (λ x, 4 * x + 8 = 0) ∨ eq = (λ x, - (1 / 3) * x + 2 / 3 = 0) ∨ 
   eq = (λ x, 2 / 3 * x = 2) ∨ eq = (λ x, 1 - 3 * x = 5)) :=
by
  sorry

end solution_satisfies_x_eq_2_l236_236832


namespace ratio_d1_d2_value_of_d3_l236_236796

-- Define constants and variables
variables {c d k c_1 c_2 d_1 d_2 c_3 d_3 : Real}
variable h1 : c * d = k
variable h2 : c_1 / c_2 = 4 / 5
variable h3 : c_3 = 2 * c_1

-- First proof problem: ratio of d_1 to d_2
theorem ratio_d1_d2 (h4 : c_1 * d_1 = c_2 * d_2) : d_1 / d_2 = 5 / 4 :=
by sorry

-- Second proof problem: value of d_3
theorem value_of_d3 (h5 : c_3 * d_3 = k) (h6 : c_1 * d_1 = k) : d_3 = d_1 / 2 :=
by sorry

end ratio_d1_d2_value_of_d3_l236_236796


namespace three_pow_2040_mod_5_l236_236091

theorem three_pow_2040_mod_5 : (3^2040) % 5 = 1 := by
  sorry

end three_pow_2040_mod_5_l236_236091


namespace probability_not_all_dice_show_different_l236_236923

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l236_236923


namespace prob_sum_six_l236_236340

variable (α : Type) [ProbabilityTheory α]

def dice_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def dice_rolls : Finset (ℕ × ℕ) := (dice_faces α).product (dice_faces α)

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 5), (5, 1), (2, 4), (4, 2), (3, 3)}

theorem prob_sum_six :
  (favorable_outcomes α).card = 5 →
  (dice_rolls α).card = 36 →
  (favorable_outcomes α).card.toRat / (dice_rolls α).card.toRat = 5 / 36 :=
by
  intro favorable_card total_card
  rw [favorable_card, total_card]
  norm_cast
  rw div_eq_of_eq_mul_right
  sorry

end prob_sum_six_l236_236340


namespace ball_count_difference_l236_236783

open Nat

theorem ball_count_difference :
  (total_balls = 145) →
  (soccer_balls = 20) →
  (basketballs > soccer_balls) →
  (tennis_balls = 2 * soccer_balls) →
  (baseballs = soccer_balls + 10) →
  (volleyballs = 30) →
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  (basketballs - soccer_balls = 5) :=
by
  intros
  let tennis_balls := 2 * soccer_balls
  let baseballs := soccer_balls + 10
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  exact sorry

end ball_count_difference_l236_236783


namespace probability_of_not_all_same_number_l236_236908

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l236_236908


namespace complementary_not_supplementary_l236_236475

theorem complementary_not_supplementary (α β : ℝ) (h₁ : α + β = 90) (h₂ : α + β ≠ 180) : (α + β = 180) = false :=
by 
  sorry

end complementary_not_supplementary_l236_236475


namespace sufficient_condition_not_necessary_condition_l236_236234

variables (p q : Prop)
def φ := ¬p ∧ ¬q
def ψ := ¬p

theorem sufficient_condition : φ p q → ψ p := 
sorry

theorem not_necessary_condition : ψ p → ¬ (φ p q) :=
sorry

end sufficient_condition_not_necessary_condition_l236_236234


namespace find_x_l236_236110

theorem find_x (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by
  -- Proof goes here
  sorry

end find_x_l236_236110


namespace count_squares_in_region_l236_236565

theorem count_squares_in_region :
  let region := { p : ℤ × ℤ | p.1 ≥ 0 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 * p.1 }
  let squares_count := ∑ x in Finset.range 6, (x + 1) * (x + 1)
  squares_count = 51 := by
    sorry

end count_squares_in_region_l236_236565


namespace area_of_tangent_triangle_l236_236602

noncomputable def f : ℝ → ℝ := λ x, x * Real.exp x

def tangent_at (f : ℝ → ℝ) (x0 : ℝ) :=
  let f' := (fun h : ℝ => (f (x0 + h) - f x0) / h) in
  λ x, f x0 + f' 1 * (x - x0)

def area_of_triangle (a b : ℝ) : ℝ := (1 / 2) * |a * b|

theorem area_of_tangent_triangle :
  area_of_triangle (1 / 2) (Real.exp 1) = (1 / 4) * (Real.exp 1) :=
by
  -- The proof will show that the area calculation is correct
  sorry

end area_of_tangent_triangle_l236_236602


namespace volleyball_team_selection_l236_236776

/-- A set representing players on the volleyball team -/
def players : Finset String := {
  "Missy", "Lauren", "Liz", -- triplets
  "Anna", "Mia",           -- twins
  "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10" -- other players
}

/-- The triplets -/
def triplets : Finset String := {"Missy", "Lauren", "Liz"}

/-- The twins -/
def twins : Finset String := {"Anna", "Mia"}

/-- The number of ways to choose 7 starters given the restrictions -/
theorem volleyball_team_selection : 
  let total_ways := (players.card.choose 7)
  let select_3_triplets := (players \ triplets).card.choose 4
  let select_2_twins := (players \ twins).card.choose 5
  let select_all_restriction := (players \ (triplets ∪ twins)).card.choose 2
  total_ways - select_3_triplets - select_2_twins + select_all_restriction = 9778 := by
  sorry

end volleyball_team_selection_l236_236776


namespace range_of_a_max_min_l236_236237

-- Given conditions
def given_function (a x : ℝ) : ℝ := (1/3) * a * x^3 + x^2 + a * x + 1

def has_maximum_and_minimum (f : ℝ → ℝ) : Prop := 
  ∃ xₘ xₙ : ℝ, xₘ ≠ xₙ ∧ (¬∃ d : ℝ, d > 0 ∧ f xₘ = f xₙ * (1 + d) ∧ f xₘ = f xₙ * (1 - d))

-- What to prove: the range of a such that f(x) has maximum an minimum
theorem range_of_a_max_min : 
  (∀ (a : ℝ), (has_maximum_and_minimum (given_function a)) ↔ ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1))) :=
begin
  sorry -- proof is not required
end

end range_of_a_max_min_l236_236237


namespace min_value_of_a_l236_236243

-- Given definitions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

def sum_of_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  n * a + d * (n * (n - 1) / 2)

theorem min_value_of_a (a : ℝ) (m : ℕ) (hpos : 0 < m) (h : sum_of_arithmetic_sequence a (-4) m = 36) : a = 15 :=
sorry

end min_value_of_a_l236_236243


namespace max_value_M_l236_236220

def J_k (k : ℕ) : ℕ := 10^(k + 3) + 1600

def M (k : ℕ) : ℕ := (J_k k).factors.count 2

theorem max_value_M : ∃ k > 0, (M k) = 7 ∧ ∀ m > 0, M m ≤ 7 :=
by 
  sorry

end max_value_M_l236_236220


namespace solve_for_m_l236_236036

noncomputable def is_decreasing_on_Ioi (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x y : ℝ, a < x → x < y → f y < f x

theorem solve_for_m 
  (m : ℝ)
  (h : ∀ x : ℝ, x > 0 → ((m^2 - m - 1) * (x ^ (2 * m - 3))) = (m^2 - m - 1) * (x ^ (2 * m - 3))) :
  (∀ x : ℝ, x > 0 →  (if m = -1 then is_decreasing_on_Ioi (λ x => (m^2 - m - 1) * x^(2*m-3)) 0 else false)) :=
begin
  intros x hx,
  sorry,
end

end solve_for_m_l236_236036


namespace Ryan_has_28_marbles_l236_236577

theorem Ryan_has_28_marbles :
  ∃ R : ℕ, (12 + R) - (1/4 * (12 + R)) * 2 = 20 ∧ R = 28 :=
by
  sorry

end Ryan_has_28_marbles_l236_236577


namespace fresh_grapes_weight_l236_236226

theorem fresh_grapes_weight (D : ℝ) (F : ℝ) 
  (hD : D = 33.33333333333333) 
  (hFreshWaterWeight : ∀ (x : ℝ), x * 0.30 = 0.90 * D) : 
  F = 100 :=
by
  have h1 : 0.30 * F = 0.90 * D := hFreshWaterWeight F
  rw [hD] at h1
  have h2 : 0.30 * F = 0.90 * 33.33333333333333 := h1
  norm_num at h2
  norm_num
  sorry

end fresh_grapes_weight_l236_236226


namespace english_books_published_outside_country_l236_236365

-- Definitions of the given conditions
def total_books : ℕ := 2300
def percent_english : ℝ := 0.80
def percent_published_in_country : ℝ := 0.60

-- Defining the number of English-language books and the number published in the country
def english_books : ℕ := (percent_english * total_books).to_nat
def published_in_country : ℕ := (percent_published_in_country * english_books).to_nat

-- The statement to prove
theorem english_books_published_outside_country : english_books - published_in_country = 736 :=
by sorry

end english_books_published_outside_country_l236_236365


namespace num_tiles_needed_l236_236542

noncomputable def tiles_needed (length_room width_room : ℝ) (length_tile_inches width_tile_inches : ℝ) (inches_to_feet : ℝ) : ℝ :=
  let length_tile_feet := length_tile_inches / inches_to_feet in
  let width_tile_feet := width_tile_inches / inches_to_feet in
  let area_tile := length_tile_feet * width_tile_feet in
  let area_room := length_room * width_room in
  area_room / area_tile

theorem num_tiles_needed 
  (length_room width_room : ℝ)
  (length_tile_inches width_tile_inches : ℝ)
  (inches_to_feet : ℝ) :
  tiles_needed length_room width_room length_tile_inches width_tile_inches inches_to_feet = 1600 :=
by
  have h_length_room : length_room = 15 := rfl
  have h_width_room : width_room = 20 := rfl
  have h_length_tile_inches : length_tile_inches = 3 := rfl
  have h_width_tile_inches : width_tile_inches = 9 := rfl
  have h_inches_to_feet : inches_to_feet = 12 := rfl
  rw [h_length_room, h_width_room, h_length_tile_inches, h_width_tile_inches, h_inches_to_feet],
  sorry

end num_tiles_needed_l236_236542


namespace quadratic_solution_l236_236049

theorem quadratic_solution :
  (∀ x : ℝ, (x^2 - x - 1 = 0) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2)) :=
by
  intro x
  rw [sub_eq_neg_add, sub_eq_neg_add]
  sorry

end quadratic_solution_l236_236049


namespace beef_original_weight_l236_236146

theorem beef_original_weight (W : ℝ) (h : 0.65 * W = 546): W = 840 :=
sorry

end beef_original_weight_l236_236146


namespace problem_statement_l236_236840

noncomputable def probability_replacement (n k : ℕ) : ℚ :=
  -- Calculate the probability of exactly 4 lanterns needing replacement 
  sorry

noncomputable def expected_replacement (n : ℕ) : ℚ :=
  -- Calculate the expected number of lanterns needing replacement
  sorry

theorem problem_statement :
  probability_replacement 9 4 = 25 / 84 ∧ expected_replacement 9 ≈ 3.32 :=
by 
  simp [probability_replacement, expected_replacement]
  sorry

end problem_statement_l236_236840


namespace tangent_line_equation_l236_236052

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x + 3

noncomputable def line_perpendicular (x y : ℝ) : Prop :=
    x + 3 * y + 1 = 0

theorem tangent_line_equation :
  (∀ A : ℝ, (f' x = 3) → (∃ y, line_perpendicular x y ∧ y = f x)
  → (∀ x y : ℝ, (3 * x - y + 2) = 0)) :=
by
  sorry

end tangent_line_equation_l236_236052


namespace percentage_increase_is_20_l236_236738

-- Defining the original cost and new cost
def original_cost := 200
def new_total_cost := 480

-- Doubling the capacity means doubling the original cost
def doubled_old_cost := 2 * original_cost

-- The increase in cost
def increase_cost := new_total_cost - doubled_old_cost

-- The percentage increase in cost
def percentage_increase := (increase_cost / doubled_old_cost) * 100

-- The theorem we need to prove
theorem percentage_increase_is_20 : percentage_increase = 20 :=
  by
  sorry

end percentage_increase_is_20_l236_236738


namespace probability_not_all_same_l236_236926

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l236_236926


namespace find_denominator_x_l236_236944

noncomputable def sum_fractions : ℝ := 
    3.0035428163476343

noncomputable def fraction1 (x : ℝ) : ℝ :=
    2007 / x

noncomputable def fraction2 : ℝ :=
    8001 / 5998

noncomputable def fraction3 : ℝ :=
    2001 / 3999

-- Problem statement in Lean
theorem find_denominator_x (x : ℝ) :
  sum_fractions = fraction1 x + fraction2 + fraction3 ↔ x = 1717 :=
by sorry

end find_denominator_x_l236_236944


namespace largest_x_satisfies_eq_l236_236888

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236888


namespace probability_not_all_same_l236_236896

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l236_236896


namespace part1_part2_l236_236397

-- Define the sets A and B based on the given conditions
def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

-- Lean statement for part (1)
theorem part1 (A_eq: A = {5, 3}) (a : ℝ) (h_a : a = 1/5) : B a ⊂ A := 
  sorry

-- Lean statement for part (2)
theorem part2 (A_eq: A = {5, 3}) (B_sub_A : ∀ a, (B a).Subset A) : {a | B a ⊆ A} = {0, 1/3, 1/5} :=
  sorry

end part1_part2_l236_236397


namespace quadratic_condition_l236_236112

theorem quadratic_condition (p q : ℝ) (x1 x2 : ℝ) (hx : x1 + x2 = -p) (hq : x1 * x2 = q) :
  p + q = 0 := sorry

end quadratic_condition_l236_236112


namespace four_digit_numbers_with_property_l236_236306

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l236_236306


namespace num_valid_x_values_l236_236290

noncomputable def count_valid_x : ℕ :=
  ((Finset.range 34).filter (λ x, x ≥ 25 ∧ 3 * x < 100 ∧ 4 * x > 99)).card

theorem num_valid_x_values : count_valid_x = 9 := by
  sorry

end num_valid_x_values_l236_236290


namespace delete_column_preserves_distinctness_l236_236712

def table (N : ℕ) := matrix (fin N) (fin N) ℕ

def rows_distinct {N : ℕ} (T : table N) : Prop := ∀ i j : fin N, i ≠ j → (∀ k : fin N, T i k ≠ T j k)

theorem delete_column_preserves_distinctness {N : ℕ} (T : table N)
  (h : rows_distinct T) : ∃ c : fin N, rows_distinct (λ i : fin N, λ j : fin (N - 1), if j.val < c.val then T i j else T i ⟨j.val + 1, sorry⟩) :=
sorry

end delete_column_preserves_distinctness_l236_236712


namespace impossible_load_two_coins_l236_236956

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l236_236956


namespace largest_x_value_satisfies_largest_x_value_l236_236874

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l236_236874


namespace product_less_than_5_probability_l236_236227

open_locale classical

-- Define the set of numbers
def num_set : Finset ℕ := {1, 2, 3, 4}

-- Define the pairs of different numbers
def pairs : Finset (ℕ × ℕ) := num_set.product num_set.filter (λ (a : ℕ × ℕ), a.1 < a.2)

-- Define the pairs whose product is less than 5
def pairs_less_than_5 : Finset (ℕ × ℕ) := pairs.filter (λ (p : ℕ × ℕ), p.1 * p.2 < 5)

-- Define the probability computation
def probability : ℚ := pairs_less_than_5.card / pairs.card

-- Theorem statement
theorem product_less_than_5_probability : probability = 1 / 2 := by
  sorry

end product_less_than_5_probability_l236_236227


namespace intersect_on_circumcircle_l236_236767

noncomputable theory

-- Defining geometrical entities
variables {A B C D E M T P N l1 l2 : Type}
variables (PM PT : Type) 

-- Geometrical conditions
def conditions (A B C D E M T P N l1 l2 : Type) (PM PT : Type) [Circumcircle : Type] : Prop :=
  let Nagel_point := N in
  let intersect_B_C_N := ((B ∉ E) ∧ (C ∉ D)) in
  let midpoints := (M = midpoint (B, E)) ∧ (T = midpoint (C, D)) in
  let P_second_intersect := true in
  let perpendiculars := (is_perpendicular PM l1 M) ∧ (is_perpendicular PT l2 T) in
  Nagel_point ∧ intersect_B_C_N ∧ midpoints ∧ P_second_intersect ∧ perpendiculars

-- The proof problem statement
theorem intersect_on_circumcircle
  (intersect : conditions A B C D E M T P N l1 l2 PM PT)
  (circumcircle_ABC : Circumcircle A B C) :
  ∃ X, X ∈ circumcircle_ABC ∧ (l1 = l2) :=
sorry

end intersect_on_circumcircle_l236_236767


namespace find_starting_point_of_a_l236_236514

def point := ℝ × ℝ
def vector := ℝ × ℝ

def B : point := (1, 0)

def b : vector := (-3, -4)
def c : vector := (1, 1)

def a : vector := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

theorem find_starting_point_of_a (hb : b = (-3, -4)) (hc : c = (1, 1)) (hB : B = (1, 0)) :
    let a := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)
    let start_A := (B.1 - a.1, B.2 - a.2)
    start_A = (12, 14) :=
by
  rw [hb, hc, hB]
  let a := (3 * (-3) - 2 * (1), 3 * (-4) - 2 * (1))
  let start_A := (1 - a.1, 0 - a.2)
  simp [a]
  sorry

end find_starting_point_of_a_l236_236514


namespace gcd_1037_425_l236_236202

theorem gcd_1037_425 : Int.gcd 1037 425 = 17 :=
by
  sorry

end gcd_1037_425_l236_236202


namespace num_valid_x_values_l236_236288

noncomputable def count_valid_x : ℕ :=
  ((Finset.range 34).filter (λ x, x ≥ 25 ∧ 3 * x < 100 ∧ 4 * x > 99)).card

theorem num_valid_x_values : count_valid_x = 9 := by
  sorry

end num_valid_x_values_l236_236288


namespace coin_loading_impossible_l236_236979

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l236_236979


namespace parallelogram_proj_rectangle_condition_l236_236429

-- Define our geometric objects
structure Point := (x y z : ℝ)
structure Rectangle := (A B C D : Point)
structure Plane := (a b c d : Point) -- Plane defined by points a, b, c, d (though we really only need a normal vector usually)

-- Basic properties and projections
def is_perpendicular (p : Point) (π : Plane) : Prop := 
  -- Placeholder for the real definition, you would use actual vector math here
  sorry 

def projection (p : Point) (π : Plane) : Point := 
  -- Placeholder for projection logic
  sorry

-- Conditions
variable (ABCD : Rectangle)
variable (α : Plane)

-- Vertices projections
def A' := projection ABCD.A α
def B' := projection ABCD.B α
def C' := projection ABCD.C α
def D' := projection ABCD.D α

-- Given conditions
axiom h_perp_A : is_perpendicular ABCD.A α
axiom h_perp_B : is_perpendicular ABCD.B α
axiom h_perp_C : is_perpendicular ABCD.C α
axiom h_perp_D : is_perpendicular ABCD.D α
axiom h_noncoincide : (A'.x ≠ C'.x ∨ A'.y ≠ C'.y ∨ A'.z ≠ C'.z)

-- Proof problems
theorem parallelogram_proj : 
  -- Prove A'B'C'D' is a parallelogram
  sorry

theorem rectangle_condition :
  -- Prove A'B'C'D' is a rectangle if and only if one side is parallel to α or lies within α
  sorry

end parallelogram_proj_rectangle_condition_l236_236429


namespace calculate_supplies_percentage_l236_236527

noncomputable def budget_salaries_percentage (degrees_salaries : ℝ) (total_degrees : ℝ) : ℝ :=
  (degrees_salaries / total_degrees) * 100

theorem calculate_supplies_percentage :
  (∀ (transportation research_development utilities equipment : ℝ) 
    (degrees_salaries : ℝ) (total_degrees : ℝ),
    transportation = 15 → 
    research_development = 9 →
    utilities = 5 →
    equipment = 4 →
    degrees_salaries = 234 →
    total_degrees = 360 →
    let salaries_percentage := budget_salaries_percentage degrees_salaries total_degrees in
    let known_percentages := transportation + research_development + utilities + equipment in
    let supplies_percentage := 100 - salaries_percentage - known_percentages in
    supplies_percentage = 2) :=
by
  intros transportation research_development utilities equipment degrees_salaries total_degrees 
    h_transportation h_rd h_utilities h_equipment h_degrees_salaries h_total_degrees,
  have salaries_percentage := budget_salaries_percentage degrees_salaries total_degrees,
  have known_percentages := transportation + research_development + utilities + equipment,
  have supplies_percentage := 100 - salaries_percentage - known_percentages,
  rw [h_transportation, h_rd, h_utilities, h_equipment, h_degrees_salaries, h_total_degrees],
  unfold budget_salaries_percentage,
  simp,
  norm_num,
  exact rfl

end calculate_supplies_percentage_l236_236527


namespace statement_a_statement_b_statement_c_statement_d_l236_236231

theorem statement_a (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : ab - a - 2b = 0) : a + 2b ≥ 8 :=
by sorry

theorem statement_b (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 2) : ¬ (frac b a + frac 4 b ≥ 5) :=
by sorry

theorem statement_c (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1) : sqrt (2 * a + 4) + sqrt (b + 1) ≤ 2 * sqrt 3 :=
by sorry

theorem statement_d (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 1 / (a + 1) + 1 / (b + 2) = 1 / 3) : ab + a + b ≥ 14 + 6 * sqrt 6 :=
by sorry

end statement_a_statement_b_statement_c_statement_d_l236_236231


namespace find_slope_l_l236_236641

-- Definitions for points A and B
def A := (-1 : ℝ, -5 : ℝ)
def B := (3 : ℝ, 3 : ℝ)

-- Definition of slope calculation function
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Definition of the slope of line AB
def slope_AB : ℝ := slope A B

-- Definition of tan_alpha based on the slope of AB
def tan_alpha : ℝ := slope_AB

-- Using the double angle formula for tangent to define the slope of line l
def slope_l : ℝ := (2 * tan_alpha) / (1 - tan_alpha ^ 2)

theorem find_slope_l : slope_l = -4 / 3 := by
  sorry

end find_slope_l_l236_236641


namespace track_length_correct_l236_236169

noncomputable def length_of_track : ℕ :=
  let x := 520 in
  ∀ (brenda_sue_run_opposite : Bool) 
    (first_meeting: ℕ) 
    (second_meeting : ℕ) 
    (constant_speeds: Bool),
  brenda_sue_run_opposite = true →
  first_meeting = 80 →
  second_meeting = 180 →
  constant_speeds = true →
  x = 520

-- Theorem to state the problem
theorem track_length_correct
  (brenda_sue_run_opposite : Bool) 
  (first_meeting: ℕ) 
  (second_meeting : ℕ) 
  (constant_speeds: Bool) :
  brenda_sue_run_opposite = true →
  first_meeting = 80 →
  second_meeting = 180 →
  constant_speeds = true →
  length_of_track = 520 :=
by sorry

end track_length_correct_l236_236169


namespace f_is_increasing_l236_236451

def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem f_is_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end f_is_increasing_l236_236451


namespace limit_problem_l236_236224

noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉₊

theorem limit_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f (x : ℝ) := (ceil x : ℝ)
  if a ≠ b then
    (∀ c : ℝ, (∀ x : ℝ, x > 0 →
      x^c * (1 / f(a * x - 7) - 1 / f(b * x + 3)) = x^(c - 1) * ((b - a) / (a * b)) →
      c ≤ 1) ∧
    ∀ x : ℝ, x > 0 →
      x * (1 / f(a * x - 7) - 1 / f(b * x + 3)) = (b - a) / (a * b))
  else
    (∀ c : ℝ, (∀ x : ℝ, x > 0 →
      x^c * (1 / f(a * x - 7) - 1 / f(a * x + 3)) = x^(c - 2) * (10 / (a^2)) →
      c ≤ 2) ∧
    ∀ x : ℝ, x > 0 →
      x^2 * (1 / f(a * x - 7) - 1 / f(a * x + 3)) = 10 / (a^2)) :=
by sorry

end limit_problem_l236_236224


namespace lcm_of_36_and_45_l236_236204

theorem lcm_of_36_and_45 : Nat.lcm 36 45 = 180 := by
  sorry

end lcm_of_36_and_45_l236_236204


namespace cos_alpha_fraction_sum_l236_236706

theorem cos_alpha_fraction_sum {α β : ℝ} (h1 : α + β < π) (h2 : ∃ q : ℚ, q > 0 ∧ q = cos α)
    (L3 : ∃ r : ℝ, r > 0 ∧ r = cos((α + β) / 2)) :
    1 / 3 = cos α → (1 + 3) = 4 :=
begin
  sorry
end

end cos_alpha_fraction_sum_l236_236706


namespace product_remainder_l236_236831

theorem product_remainder (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) (h4 : (a + b + c) % 7 = 3) : 
  (a * b * c) % 7 = 2 := 
by sorry

end product_remainder_l236_236831


namespace triangle_area_l236_236150

theorem triangle_area (a b c : ℝ) (angle_A : ℝ) 
(h1 : a = 14) 
(h2 : angle_A = 60) 
(h3 : b / c = 8 / 5) 
(h4 : c = (b * 5) / 8) : 
a * b * c * real.sin (angle_A / 2) = 40 * real.sqrt 3 := 
sorry

end triangle_area_l236_236150


namespace four_digit_numbers_property_l236_236320

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l236_236320


namespace quadratic_radical_type_equivalence_l236_236553

def is_same_type_as_sqrt2 (x : ℝ) : Prop := ∃ k : ℚ, x = k * (Real.sqrt 2)

theorem quadratic_radical_type_equivalence (A B C D : ℝ) (hA : A = (Real.sqrt 8) / 7)
  (hB : B = Real.sqrt 3) (hC : C = Real.sqrt (1 / 3)) (hD : D = Real.sqrt 12) :
  is_same_type_as_sqrt2 A ∧ ¬ is_same_type_as_sqrt2 B ∧ ¬ is_same_type_as_sqrt2 C ∧ ¬ is_same_type_as_sqrt2 D :=
by
  sorry

end quadratic_radical_type_equivalence_l236_236553


namespace A_work_days_l236_236127

theorem A_work_days (x : ℝ) (H : 3 * (1 / x + 1 / 20) = 0.35) : x = 15 := 
by
  sorry

end A_work_days_l236_236127


namespace B_wins_polynomial_game_l236_236509

def polynomial_game_winning_strategy (n : ℕ) (f : ℝ[X]) : Prop :=
  degree f = 2 * n ∧
  ∃ (g : List (ℕ × ℝ)), 
    (∀ (p : ℕ × ℝ), p ∈ g → p.1 > 0 ∧ p.1 < degree f) ∧
    (∀ (x : ℝ), is_root f x → is_root (eval₂_ring_hom p) x)

theorem B_wins_polynomial_game (n : ℕ) (h : n ≥ 2) : 
  ∀ f : ℝ[X], 
  degree f = 2 * n ∧ 
  (∀ (coeffs : fin (2 * n) → ℝ), 
    (∀ x : ℝ, x ≠ 0 → ¬ is_root (f.coeff ∘ coeffs)) := sorry.

end B_wins_polynomial_game_l236_236509


namespace smallest_multiple_14_15_16_l236_236214

theorem smallest_multiple_14_15_16 : 
  Nat.lcm (Nat.lcm 14 15) 16 = 1680 := by
  sorry

end smallest_multiple_14_15_16_l236_236214


namespace polynomial_divisibility_l236_236392

theorem polynomial_divisibility (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (a : Fin n.succ → ℚ) 
(h_sum : (Finset.univ.sum (λ i, a i)) = 0)
(f : ℚ → ℚ := fun x => Finset.univ.sum (λ i, a i * x ^ (n - i))) :
  ∃ φ : ℚ → ℚ, ∀ x : ℚ, f (x ^ (k + 1)) = (x ^ (k + 1) - 1) * φ (x ^ (k + 1)) → 
  (∃ ψ : ℚ → ℚ, ∀ x : ℚ, f (x ^ (k + 1)) = (x - 1) * (Finset.range (k + 1)).sum (λ i, x ^ i) * ψ (x ^ (k + 1))) := sorry

end polynomial_divisibility_l236_236392


namespace four_digit_num_condition_l236_236296

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l236_236296


namespace four_digit_num_condition_l236_236297

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l236_236297


namespace probability_not_all_dice_show_different_l236_236918

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l236_236918


namespace parabola_other_intercept_l236_236624

theorem parabola_other_intercept (a b c : ℝ)
  (vertex : (ℝ × ℝ))
  (x_intercept : ℝ)
  (h_vertex : vertex = (4, 10))
  (h_x_intercept : x_intercept = 1)
  (h_eq : ∀ x : ℝ, y = a * x^2 + b * x + c) :
  ∃ (x_other : ℝ), 2 * 4 - x_intercept = x_other ∧ x_other = 7 :=
by
  rw h_vertex at *
  rw h_x_intercept at *
  use 7
  split
  · exact sub_self_add_self 4 1
  · refl

end parabola_other_intercept_l236_236624


namespace dacid_average_l236_236187

noncomputable def average (a b : ℕ) : ℚ :=
(a + b) / 2

noncomputable def overall_average (a b c d e : ℕ) : ℚ :=
(a + b + c + d + e) / 5

theorem dacid_average :
  ∀ (english mathematics physics chemistry biology : ℕ),
  english = 86 →
  mathematics = 89 →
  physics = 82 →
  chemistry = 87 →
  biology = 81 →
  (average english mathematics < 90) ∧
  (average english physics < 90) ∧
  (average english chemistry < 90) ∧
  (average english biology < 90) ∧
  (average mathematics physics < 90) ∧
  (average mathematics chemistry < 90) ∧
  (average mathematics biology < 90) ∧
  (average physics chemistry < 90) ∧
  (average physics biology < 90) ∧
  (average chemistry biology < 90) ∧
  overall_average english mathematics physics chemistry biology = 85 := by
  intros english mathematics physics chemistry biology
  intros h_english h_mathematics h_physics h_chemistry h_biology
  simp [average, overall_average]
  rw [h_english, h_mathematics, h_physics, h_chemistry, h_biology]
  sorry

end dacid_average_l236_236187


namespace vector_dot_product_l236_236164

open Real EuclideanGeometry

/-- Given a right triangle ABC with hypotenuse AB and AC = BC = 2, if point P lies on AB such that BP = 2PA,
prove that the vector product CP ⋅ CA + CP ⋅ CB equals 4. -/
theorem vector_dot_product (A B C P : Point ℝ) (h : ∠ A C B = π / 2)
  (h1 : dist A C = 2) (h2 : dist B C = 2)
  (h3 : collinear {A, B, P} ∧ ∃ k : ℝ, k > 0 ∧ k ≤ 1 ∧ dist B P = 2 * dist P A) :
  (vector.coord (C -ᵥ P) 0) * (vector.coord (C -ᵥ A) 0) + (vector.coord (C -ᵥ P) 1) * (vector.coord (C -ᵥ A) 1) +
  (vector.coord (C -ᵥ P) 0) * (vector.coord (C -ᵥ B) 0) + (vector.coord (C -ᵥ P) 1) * (vector.coord (C -ᵥ B) 1) = 4 :=
sorry

end vector_dot_product_l236_236164


namespace reserve_bird_percentage_l236_236704

theorem reserve_bird_percentage (total_birds hawks paddyfield_warbler_percentage kingfisher_percentage woodpecker_percentage owl_percentage : ℕ) 
  (h1 : total_birds = 5000)
  (h2 : hawks = 30 * total_birds / 100)
  (h3 : paddyfield_warbler_percentage = 40)
  (h4 : kingfisher_percentage = 25)
  (h5 : woodpecker_percentage = 15)
  (h6 : owl_percentage = 15) :
  let non_hawks := total_birds - hawks
  let paddyfield_warblers := paddyfield_warbler_percentage * non_hawks / 100
  let kingfishers := kingfisher_percentage * paddyfield_warblers / 100
  let woodpeckers := woodpecker_percentage * non_hawks / 100
  let owls := owl_percentage * non_hawks / 100
  let specified_non_hawks := paddyfield_warblers + kingfishers + woodpeckers + owls
  let unspecified_non_hawks := non_hawks - specified_non_hawks
  let percentage_unspecified := unspecified_non_hawks * 100 / total_birds
  percentage_unspecified = 14 := by
  sorry

end reserve_bird_percentage_l236_236704


namespace num_positive_x_count_num_positive_x_l236_236293

theorem num_positive_x (x : ℕ) : (3 * x < 100) ∧ (4 * x ≥ 100) → x ≥ 25 ∧ x ≤ 33 := by
  sorry

theorem count_num_positive_x : 
  (∃ x : ℕ, (3 * x < 100) ∧ (4 * x ≥ 100)) → 
  (finset.range 34).filter (λ x, (3 * x < 100 ∧ 4 * x ≥ 100)).card = 9 := by
  sorry

end num_positive_x_count_num_positive_x_l236_236293


namespace incorrect_option_c_l236_236387

variable (a : ℕ → ℝ)

def Sn (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i

theorem incorrect_option_c (h1 : Sn a 5 < Sn a 6)
                          (h2 : Sn a 6 = Sn a 7)
                          (h3 : Sn a 7 > Sn a 8) :
  Sn a 9 ≤ Sn a 5 := 
sorry

end incorrect_option_c_l236_236387


namespace gain_percent_correct_l236_236530

def CostPrice : ℝ := 900
def SellingPrice : ℝ := 1150

def gainPercent (CP SP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem gain_percent_correct : gainPercent CostPrice SellingPrice ≈ 27.78 := 
by
  -- This part will contain the proof but mentioned to use sorry
  sorry

end gain_percent_correct_l236_236530


namespace change_in_points_l236_236954

-- Definitions for the problem
variables (n : ℕ) (players : Type) [fintype players] (score : players → ℕ → ℚ)

-- Conditions for the problem
def round_robin_twice (s1 s2 : players → ℕ → ℚ) :=
  ∀ p : players, abs (s2 p - s1 p) ≥ n

-- Theorem to be proved
theorem change_in_points (n : ℕ) (players : Type) [fintype players] (score1 score2 : players → ℕ → ℚ)
    (h : round_robin_twice n players score1 score2) :
  ∀ p : players, abs (score2 p - score1 p) = n :=
begin
  sorry
end

end change_in_points_l236_236954


namespace sum_in_base6_l236_236612

theorem sum_in_base6 :
  @add (Fin 6) (Fin.hasAdd 6) (Fin 6) (Fin 6) (⟨2015, sorry⟩ : Fin 6) +
  @add (Fin 6) (Fin.hasAdd 6) (Fin 6) (Fin 6) (⟨251, sorry⟩ : Fin 6) +
  @add (Fin 6) (Fin.hasAdd 6) (Fin 6) (Fin 6) (⟨25, sorry⟩ : Fin 6)
  = ⟨2335, sorry⟩ := sorry

end sum_in_base6_l236_236612


namespace smallest_x_absolute_value_l236_236610

theorem smallest_x_absolute_value : ∃ x : ℤ, |x + 3| = 15 ∧ ∀ y : ℤ, |y + 3| = 15 → x ≤ y :=
sorry

end smallest_x_absolute_value_l236_236610


namespace sqrt_64_eq_pm_8_l236_236474

theorem sqrt_64_eq_pm_8 : ∃x : ℤ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
by
  sorry

end sqrt_64_eq_pm_8_l236_236474


namespace trapezoid_area_correct_l236_236113

variable (AB CD : ℝ) (r : ℝ) (angle : ℝ)

def trapezoid_area (AB CD r angle : ℝ) : ℝ :=
  let height := r + r * (1 / 2) -- Distance from O to AB and O to XY
  (1 / 2) * height * (AB + CD)

theorem trapezoid_area_correct :
  AB = 10 → CD = 15 → r = 6 → angle = 120 → trapezoid_area AB CD r angle = 225 / 2 := by
  intros h1 h2 h3 h4
  sorry

end trapezoid_area_correct_l236_236113


namespace lemon_bag_mass_l236_236134

variable (m : ℝ)  -- mass of one bag of lemons in kg

-- Conditions
def max_load := 900  -- maximum load in kg
def num_bags := 100  -- number of bags
def extra_load := 100  -- additional load in kg

-- Proof statement (target)
theorem lemon_bag_mass : num_bags * m + extra_load = max_load → m = 8 :=
by
  sorry

end lemon_bag_mass_l236_236134


namespace problem_l236_236688

theorem problem (a b : ℕ) (h1 : 2^4 + 2^4 = 2^a) (h2 : 3^5 + 3^5 + 3^5 = 3^b) : a + b = 11 :=
by {
  sorry
}

end problem_l236_236688


namespace playground_area_22500_l236_236455

noncomputable def rectangle_playground_area (w l : ℕ) : ℕ :=
  w * l

theorem playground_area_22500 (w l : ℕ) (h1 : l = 2 * w + 25) (h2 : 2 * l + 2 * w = 650) :
  rectangle_playground_area w l = 22500 := by
  sorry

end playground_area_22500_l236_236455


namespace largest_x_value_satisfies_largest_x_value_l236_236872

theorem largest_x_value (x : ℚ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

theorem satisfies_largest_x_value : 
  ∃ x : ℚ, Real.sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
sorry

end largest_x_value_satisfies_largest_x_value_l236_236872


namespace sum_real_imaginary_zero_l236_236032

open Complex

noncomputable def z : ℂ := sorry -- We introduce a placeholder for the complex number z

theorem sum_real_imaginary_zero (hz : conj z * (1 - I) = abs (1 + I)) : (z.re + z.im) = 0 := 
by sorry

end sum_real_imaginary_zero_l236_236032


namespace vertically_opposite_angles_l236_236724

-- Definitions of the conditions
def intersecting_lines (l1 l2 l3 : Line) : Prop := 
  ∃ (p1 p2 p3 : Point), p1 ∈ l1 ∧ p1 ∈ l2 ∧
                        p2 ∈ l1 ∧ p2 ∈ l3 ∧
                        p3 ∈ l2 ∧ p3 ∈ l3

-- Statement of the problem to be proved
theorem vertically_opposite_angles (l1 l2 l3 : Line) 
  (h : intersecting_lines l1 l2 l3) : 
  ∃ n : ℕ, n = 6 := 
by
  sorry

end vertically_opposite_angles_l236_236724


namespace largest_square_factor_of_10_fact_l236_236015

theorem largest_square_factor_of_10_fact (n : ℕ) : 
  n = 6 →
  let ten_factorial := (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10) in
  (6!)^2 ∣ ten_factorial :=
begin
  intro h,
  rw h,
  sorry
end

end largest_square_factor_of_10_fact_l236_236015


namespace rotten_oranges_are_50_l236_236838

def oranges_in_bags (bags : ℕ) (oranges_per_bag : ℕ) : ℕ :=
  bags * oranges_per_bag

def remaining_oranges (total_oranges : ℕ) (juice_oranges : ℕ) : ℕ :=
  total_oranges - juice_oranges

def rotten_oranges (total_sellable_oranges : ℕ) (sold_oranges : ℕ) : ℕ :=
  total_sellable_oranges - sold_oranges

theorem rotten_oranges_are_50 : 
  oranges_in_bags 10 30 = 300 →
  remaining_oranges 300 30 = 270 →
  rotten_oranges 270 220 = 50 :=
by
  intros h1 h2
  rw [←h1, ←h2]
  sorry

end rotten_oranges_are_50_l236_236838


namespace hexagon_angle_geometric_progression_l236_236440

theorem hexagon_angle_geometric_progression (a r : ℝ) 
  (h1 : 720 = a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5) 
  (h2 : r = 2) :
  ∃ angle : ℝ, angle = 90 := 
by
  sorry

end hexagon_angle_geometric_progression_l236_236440


namespace equation_of_chord_bisected_by_point_l236_236664
noncomputable def ellipse := { p : ℝ × ℝ | p.1^2 / 36 + p.2^2 / 9 = 1 }
def midpoint := (2 : ℝ, 2 : ℝ)

theorem equation_of_chord_bisected_by_point :
  (∃ A B : ℝ × ℝ,
    A ∈ ellipse ∧ B ∈ ellipse ∧
    (A.1 + B.1) / 2 = midpoint.1 ∧
    (A.2 + B.2) / 2 = midpoint.2 ∧
    ∃ m b : ℝ, m = -1/4 ∧ ∀ x y : ℝ, y = m * x + b ↔ x + 4 * y - 10 = 0) :=
sorry

end equation_of_chord_bisected_by_point_l236_236664


namespace Maria_Ivanovna_solution_l236_236405

noncomputable def Maria_grades_problem : Prop :=
  let a : ℕ → ℕ := λ n, if n = 1 then 3
                        else if n = 2 then 8
                        else 2 * a (n - 1) + 2 * a (n - 2) in
  a 6 = 448

theorem Maria_Ivanovna_solution : Maria_grades_problem := by
  sorry

end Maria_Ivanovna_solution_l236_236405


namespace proof_problem_l236_236644

variable {α β : Real}
variable {A : Set ℝ}

def proposition_p := ∀ α β : ℝ, (tan α = tan β ↔ α = β)
def proposition_q := ∅ ⊆ A

theorem proof_problem :
  (proposition_p ∨ proposition_q) ∧ ¬(proposition_p ∧ proposition_q) ∧ ¬proposition_p ∧ proposition_q := by
  sorry

end proof_problem_l236_236644


namespace valid_numbers_count_l236_236486

def count_valid_numbers (n : ℕ) : ℕ := 1 / 4 * (5^n + 2 * 3^n + 1)

theorem valid_numbers_count (n : ℕ) : count_valid_numbers n = (1 / 4) * (5^n + 2 * 3^n + 1) :=
by sorry

end valid_numbers_count_l236_236486


namespace solve_quadratic_l236_236434

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, 
  (-6) * x1^2 + 11 * x1 - 3 = 0 ∧ (-6) * x2^2 + 11 * x2 - 3 = 0 ∧ x1 = 1.5 ∧ x2 = 1 / 3 :=
by
  sorry

end solve_quadratic_l236_236434


namespace num_valid_x_values_l236_236291

noncomputable def count_valid_x : ℕ :=
  ((Finset.range 34).filter (λ x, x ≥ 25 ∧ 3 * x < 100 ∧ 4 * x > 99)).card

theorem num_valid_x_values : count_valid_x = 9 := by
  sorry

end num_valid_x_values_l236_236291


namespace electric_bicycle_sales_l236_236145

theorem electric_bicycle_sales (a : ℝ) :
  (let first_quarter_total := 1 in
   let mA_first_quarter := 0.56 * first_quarter_total in
   let mB_C_first_quarter := 1 - mA_first_quarter in
   let mA_second_quarter := mA_first_quarter * 1.23 in
   let mB_C_second_quarter := mB_C_first_quarter * (1 - a / 100) in
   let second_quarter_total := mA_second_quarter + mB_C_second_quarter in
   second_quarter_total = 1.12) → a = 2 :=
by
  intros h
  sorry

end electric_bicycle_sales_l236_236145


namespace probability_not_all_same_l236_236915

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l236_236915


namespace probability_each_gets_one_of_each_correct_sum_m_n_l236_236998

-- Define the problem setup
def guests (n : ℕ) := {i | i < n}
def rolls := {rollType : Type} -- type to differentiate cheese from fruit

-- Assuming four cheese and four fruit, wrapped and indistinguishable.
constant total_rolls : set (guests 8)
constant cheese_rolls : set (guests 4)
constant fruit_rolls  : set (guests 4)

axiom cheese_card : cheese_rolls.finite_card = 4
axiom fruit_card  : fruit_rolls.finite_card = 4
axiom total_card : total_rolls.finite_card = 8

-- Probability calculation setup
noncomputable def probability_each_gets_one_of_each := (2/7) * (3/10) * (1/3)

-- Prove the probability that each guest gets one roll of each type
theorem probability_each_gets_one_of_each_correct :
  probability_each_gets_one_of_each = (1/35) := sorry

-- Prove the sum of m and n where probability is m/n and m, n are relatively prime.
theorem sum_m_n : (1 + 35 = 36) :=
by norm_num

end probability_each_gets_one_of_each_correct_sum_m_n_l236_236998


namespace log_ordering_correct_l236_236983

noncomputable def log_ordering : Prop :=
  let a := 20.3
  let b := 0.32
  let c := Real.log b
  (0 < b ∧ b < 1) ∧ (c < 0) ∧ (c < b ∧ b < a)

theorem log_ordering_correct : log_ordering :=
by
  -- skipped proof
  sorry

end log_ordering_correct_l236_236983


namespace compare_shaded_areas_l236_236355

-- Definitions based on conditions
def area_of_square (s : ℝ) := s * s
def shaded_area_I (s : ℝ) := (1/4) * area_of_square s
def shaded_area_II (s : ℝ) := (1/4) * area_of_square s
def shaded_area_III (s : ℝ) := (3/16) * area_of_square s

-- Theorem to prove the correct option
theorem compare_shaded_areas (s : ℝ) (h : s > 0) : 
  shaded_area_I s = shaded_area_II s ∧
  shaded_area_I s ≠ shaded_area_III s := 
by
  sorry

end compare_shaded_areas_l236_236355


namespace division_result_l236_236939

def numerator : ℕ := 3 * 4 * 5
def denominator : ℕ := 2 * 3
def quotient : ℕ := numerator / denominator

theorem division_result : quotient = 10 := by
  sorry

end division_result_l236_236939


namespace nine_possible_xs_l236_236285

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end nine_possible_xs_l236_236285


namespace first_term_geometric_sequence_l236_236117

theorem first_term_geometric_sequence (a r : ℝ)
  (h1 : a * r^6 = fact 9)
  (h2 : a * r^10 = fact 11) :
  a = (fact 9) / (real.sqrt (real.sqrt 110))^6 := 
sorry

end first_term_geometric_sequence_l236_236117


namespace train_speed_clicks_l236_236469

theorem train_speed_clicks
  (rails_length : ℕ)
  (time_in_seconds_per_click : ℕ → ℕ → ℕ)
  (correct_time_in_seconds : ℕ)
  (feet_per_mile : ℕ := 5280)
  (seconds_per_hour : ℕ := 3600)
  (none_of_these_time : ℕ)
  (none_of_these_time := 27) :
  rails_length = 40 →correct_time_in_seconds = 27 :=
begin
  intros,
  sorry
end

end train_speed_clicks_l236_236469


namespace product_of_consecutive_integers_sqrt_73_l236_236833

theorem product_of_consecutive_integers_sqrt_73 : 
  ∃ (m n : ℕ), (m < n) ∧ ∃ (j k : ℕ), (j = 8) ∧ (k = 9) ∧ (m = j) ∧ (n = k) ∧ (m * n = 72) := by
  sorry

end product_of_consecutive_integers_sqrt_73_l236_236833


namespace remainder_of_power_modulo_l236_236087

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end remainder_of_power_modulo_l236_236087


namespace calculate_value_of_expression_l236_236173

theorem calculate_value_of_expression :
  3.5 * 7.2 * (6.3 - 1.4) = 122.5 :=
  by
  sorry

end calculate_value_of_expression_l236_236173


namespace increasing_interval_m_range_l236_236659

def y (x m : ℝ) : ℝ := x^2 + 2 * m * x + 10

theorem increasing_interval_m_range (m : ℝ) : (∀ x, 2 ≤ x → ∀ x', x' ≥ x → y x m ≤ y x' m) → (-2 : ℝ) ≤ m :=
sorry

end increasing_interval_m_range_l236_236659


namespace arithmetic_sequence_general_term_and_sum_l236_236242

theorem arithmetic_sequence_general_term_and_sum :
  (∀ (a : ℕ → ℝ), 
    (a 1 = 1 / 2 ∧ 
    (2 * a 2 = a 1 + a 3 - 1 / 8) ∧
    (0 < (1 / 2 : ℝ) ∧ (1 / 2 : ℝ) < 1))
    → (∀ n, a n = 1 / (2^n)) ∧ 
    (∀ S a, (S n = ∑ i in Finset.range n, i * a i)
           → S n = 2 - (n + 2) * (1 / 2)^n))
:= by sorry

end arithmetic_sequence_general_term_and_sum_l236_236242


namespace rotten_apples_did_not_smell_l236_236416

theorem rotten_apples_did_not_smell:
  ∀ (total_apples rotten_percentage smelly_percentage : ℕ),
  total_apples = 200 →
  rotten_percentage = 40 →
  smelly_percentage = 70 →
  (total_apples * rotten_percentage / 100 - total_apples * rotten_percentage / 100 * smelly_percentage / 100) = 24 :=
by
  intros total_apples rotten_percentage smelly_percentage hab hbp hsp
  have h1 : total_apples * rotten_percentage / 100 = 80 := by sorry
  have h2 : (total_apples * rotten_percentage / 100) * smelly_percentage / 100 = 56 := by sorry
  show 80 - 56 = 24, by sorry

end rotten_apples_did_not_smell_l236_236416


namespace isosceles_triangle_third_side_l236_236352

theorem isosceles_triangle_third_side (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h₃ : a = b ∨ ∃ c, c = 9 ∧ (a = c ∨ b = c) ∧ (a + b > c ∧ a + c > b ∧ b + c > a)) :
  a = 9 ∨ b = 9 :=
by
  sorry

end isosceles_triangle_third_side_l236_236352


namespace monotonic_intervals_log_sum_greater_than_two_l236_236671

noncomputable def f (a b : ℝ) (x : ℝ) := a * log x - b * x - 3
noncomputable def g (b : ℝ) (x : ℝ) := log x - b * x

theorem monotonic_intervals (a : ℝ) (ha : a ≠ 0) (b : ℝ) (hab : a = b) :
  if a > 0 then
    ∀ x, (0 < x ∧ x < 1 → f a b x = a * log x - a * x - 3 ∧ f' a b x > 0)
    ∧ (1 < x ∧ x < ∞ → f a b x = a * log x - a * x - 3 ∧ f' a b x < 0)
  else
    ∀ x, (1 < x ∧ x < ∞ → f a b x = a * log x - a * x - 3 ∧ f' a b x > 0)
    ∧ (0 < x ∧ x < 1 → f a b x = a * log x - a * x - 3 ∧ f' a b x < 0) :=
sorry

theorem log_sum_greater_than_two (x1 x2 b : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (hx1x2 : x1 ≠ x2) (h : g b x1 = 0 ∧ g b x2 = 0) :
  log x1 + log x2 > 2 :=
sorry

end monotonic_intervals_log_sum_greater_than_two_l236_236671


namespace four_digit_numbers_property_l236_236316

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l236_236316


namespace students_in_all_three_events_l236_236703

-- Definitions based on the conditions
def total_students : ℕ := 45
def tug_of_war_students : ℕ := 45
def kick_shuttlecock_students : ℕ := 39
def basketball_shoot_students : ℕ := 28

-- Lean theorem statement to prove the number of students participating in all three events
theorem students_in_all_three_events : ∃ N_T : ℕ, N_T = 22 :=
begin
  -- Here you can detail the proof, for now, we use sorry as requested
  sorry
end

end students_in_all_three_events_l236_236703


namespace largest_x_satisfies_eq_l236_236867

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236867


namespace mary_shirts_left_l236_236011

theorem mary_shirts_left :
  let blue_shirts := 35
  let brown_shirts := 48
  let red_shirts := 27
  let yellow_shirts := 36
  let green_shirts := 18
  let blue_given_away := 4 / 5 * blue_shirts
  let brown_given_away := 5 / 6 * brown_shirts
  let red_given_away := 2 / 3 * red_shirts
  let yellow_given_away := 3 / 4 * yellow_shirts
  let green_given_away := 1 / 3 * green_shirts
  let blue_left := blue_shirts - blue_given_away
  let brown_left := brown_shirts - brown_given_away
  let red_left := red_shirts - red_given_away
  let yellow_left := yellow_shirts - yellow_given_away
  let green_left := green_shirts - green_given_away
  blue_left + brown_left + red_left + yellow_left + green_left = 45 := by
  sorry

end mary_shirts_left_l236_236011


namespace coin_loading_impossible_l236_236964

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l236_236964


namespace vector_addition_vector_magnitude_unit_vector_coordinates_l236_236229

noncomputable def vec := ℝ × ℝ

def a : vec := (2, 1)
def b : vec := (-3, -4)

def dot_product (v1 v2: vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def norm (v: vec) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def is_perpendicular (v1 v2: vec) : Prop := dot_product v1 v2 = 0

def is_unit_vector (v: vec) : Prop := norm v = 1

theorem vector_addition : 2 • a + 3 • b = (-5, -10) := 
by
  sorry

theorem vector_magnitude : norm (a - 2 • b) = Real.sqrt 145 := 
by
  sorry

theorem unit_vector_coordinates (c: vec) (h1: is_unit_vector c) (h2: is_perpendicular c (a - b)) : 
  c = (Real.sqrt 2 / 2, -Real.sqrt 2 / 2) ∨ c = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2) :=
by
  sorry

end vector_addition_vector_magnitude_unit_vector_coordinates_l236_236229


namespace silver_car_percentage_l236_236524

def initial_lot_size : ℕ := 40
def percentage_silver_initial : ℝ := 0.15
def new_shipment_size : ℕ := 80
def percentage_not_silver_new_shipment : ℝ := 0.30
def percentage_silver_new_shipment := 1 - percentage_not_silver_new_shipment

theorem silver_car_percentage :
  ((percentage_silver_initial * initial_lot_size + percentage_silver_new_shipment * new_shipment_size) 
  / (initial_lot_size + new_shipment_size)) * 100 ≈ 51.67 := 
by
  sorry

end silver_car_percentage_l236_236524


namespace mod_3_pow_2040_eq_1_mod_5_l236_236093

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l236_236093


namespace shirts_total_cost_l236_236377

noncomputable def firstShirtCost : ℝ := 15
noncomputable def priceDifference : ℝ := 6
noncomputable def discountFirstShirt : ℝ := 0.15
noncomputable def discountSecondShirt : ℝ := 0.10
noncomputable def salesTax : ℝ := 0.07

theorem shirts_total_cost 
  (firstShirtCost = 15)
  (priceDifference = 6)
  (discountFirstShirt = 0.15)
  (discountSecondShirt = 0.10)
  (salesTax = 0.07) : 
  (firstShirtCost + (firstShirtCost - priceDifference) = 24) ∧ 
  ((firstShirtCost - discountFirstShirt * firstShirtCost) + 
   ((firstShirtCost - priceDifference) - discountSecondShirt * (firstShirtCost - priceDifference)) * 
   (1 + salesTax) = 22.31) := sorry

end shirts_total_cost_l236_236377


namespace manager_wage_l236_236560

variable (M D C : ℝ)

def condition1 : Prop := D = M / 2
def condition2 : Prop := C = 1.25 * D
def condition3 : Prop := C = M - 3.1875

theorem manager_wage (h1 : condition1 M D) (h2 : condition2 D C) (h3 : condition3 M C) : M = 8.5 :=
by
  sorry

end manager_wage_l236_236560


namespace solve_quadrilateral_problem_l236_236425

noncomputable def quadrilateral_problem : Prop :=
  ∃ (O : Type) (ABC : O) (D : O) (A : O) (B : O) (C : O) (X : O) (Y : O) (Z : O) (V : O) (H : O)
  (BD_length : ℝ) (AC_length : ℝ),
  (BD_length = 71 / AC_length) ∧
  (∃ (AB BC CD DA : ℝ), AB = 5 ∧ BC = 4 ∧ CD = 7 ∧ DA = 9) ∧
  (DX = 1/3 * BD_length) ∧ (BY = 1/4 * BD_length) ∧
  (XZ * XD = 12.33)

theorem solve_quadrilateral_problem :
  quadrilateral_problem := 
sorry

end solve_quadrilateral_problem_l236_236425


namespace michael_passes_donovan_l236_236949

theorem michael_passes_donovan
  (track_length : ℕ)
  (donovan_lap_time : ℕ)
  (michael_lap_time : ℕ)
  (start_time : ℕ)
  (L : ℕ)
  (h1 : track_length = 500)
  (h2 : donovan_lap_time = 45)
  (h3 : michael_lap_time = 40)
  (h4 : start_time = 0)
  : L = 9 :=
by
  sorry

end michael_passes_donovan_l236_236949


namespace largest_x_satisfies_eq_l236_236868

theorem largest_x_satisfies_eq (x : ℝ) (h : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236868


namespace trigonometric_equation_has_no_solutions_l236_236323

theorem trigonometric_equation_has_no_solutions :
  ∀ x ∈ Set.Icc (0 : ℝ) (π / 2), ∨ (cos(π / 3 * sin x) ≠ sin(π / 3 * cos x)) = false :=
by
  sorry

end trigonometric_equation_has_no_solutions_l236_236323


namespace bob_total_miles_l236_236563

def total_miles_day1 (T : ℝ) := 0.20 * T
def remaining_miles_day1 (T : ℝ) := T - total_miles_day1 T
def total_miles_day2 (T : ℝ) := 0.50 * remaining_miles_day1 T
def remaining_miles_day2 (T : ℝ) := remaining_miles_day1 T - total_miles_day2 T
def total_miles_day3 (T : ℝ) := 28

theorem bob_total_miles (T : ℝ) (h : total_miles_day3 T = remaining_miles_day2 T) : T = 70 :=
by
  sorry

end bob_total_miles_l236_236563


namespace playground_area_l236_236457

theorem playground_area
  (w l : ℕ)
  (h₁ : l = 2 * w + 25)
  (h₂ : 2 * (l + w) = 650) :
  w * l = 22500 := 
sorry

end playground_area_l236_236457


namespace peter_weight_l236_236077

theorem peter_weight (sam_weight : ℕ) (tyler_more : ℕ) (tyler_half : ℕ):
  sam_weight = 105 →
  tyler_more = 25 →
  tyler_half = 2 →
  let tyler_weight := sam_weight + tyler_more in
  let peter_weight := tyler_weight / tyler_half in
  peter_weight = 65 :=
begin
  intros h1 h2 h3,
  simp [h1, h2, h3],
  sorry
end

end peter_weight_l236_236077


namespace books_sold_l236_236126

theorem books_sold (x : ℕ) 
  (h1 : 4 - x + 10 = 11) : x = 3 :=
begin
  sorry
end

end books_sold_l236_236126


namespace exist_distinct_nat_numbers_l236_236559

theorem exist_distinct_nat_numbers :
  ∃ (m n p q : ℕ),
    m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q ∧ 
    m + n = p + q ∧ 
    (real.sqrt m + real.cbrt n = real.sqrt p + real.cbrt q) ∧
    (real.sqrt m + real.cbrt n > 2004) :=
begin
  sorry
end

end exist_distinct_nat_numbers_l236_236559


namespace exists_quadratic_function_l236_236371

theorem exists_quadratic_function :
  (∃ (a b c : ℝ), ∀ (k : ℕ), k > 0 → (a * (5 / 9 * (10^k - 1))^2 + b * (5 / 9 * (10^k - 1)) + c = 5/9 * (10^(2*k) - 1))) :=
by
  have a := 9 / 5
  have b := 2
  have c := 0
  use a, b, c
  intros k hk
  sorry

end exists_quadratic_function_l236_236371


namespace find_f_65_l236_236813

theorem find_f_65 (f : ℝ → ℝ) (h_eq : ∀ x y : ℝ, f (x * y) = x * f y) (h_f1 : f 1 = 40) : f 65 = 2600 :=
by
  sorry

end find_f_65_l236_236813


namespace mod_3_pow_2040_eq_1_mod_5_l236_236094

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l236_236094


namespace graph_of_function_not_in_third_quadrant_l236_236039

theorem graph_of_function_not_in_third_quadrant :
  ∀ (x : ℝ), ¬((x < 0) ∧ (y < 0) ∧ (y = -2 * x + 2)) :=
by {
  intros x y h,
  sorry
}

end graph_of_function_not_in_third_quadrant_l236_236039


namespace linear_function_is_C_l236_236941

theorem linear_function_is_C :
  ∀ (f : ℤ → ℤ), (f = (λ x => 2 * x^2 - 1) ∨ f = (λ x => -1/x) ∨ f = (λ x => (x+1)/3) ∨ f = (λ x => 3 * x + 2 * x^2 - 1)) →
  (f = (λ x => (x+1)/3)) ↔ 
  (∃ (m b : ℤ), ∀ x : ℤ, f x = m * x + b) :=
by
  sorry

end linear_function_is_C_l236_236941


namespace number_of_girls_in_class_l236_236702

theorem number_of_girls_in_class: ∃ g b : ℕ, (3 * b = 4 * g) ∧ (g + b = 35) ∧ (g = 15) :=
by
  use 15, 20
  split
  . apply nat.succ_pos _
  . split
  . exact rfl
  . exact rfl
. sorry

end number_of_girls_in_class_l236_236702


namespace solve_for_b_l236_236791

theorem solve_for_b (b x : ℚ)
  (h₁ : 3 * x + 5 = 1)
  (h₂ : b * x + 6 = 0) :
  b = 9 / 2 :=
sorry   -- The proof is omitted as per instruction.

end solve_for_b_l236_236791


namespace four_digit_numbers_property_l236_236317

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l236_236317


namespace painting_time_eq_l236_236156

theorem painting_time_eq (t : ℝ) :
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 ↔ t = 8 / 3 :=
by
  sorry

end painting_time_eq_l236_236156


namespace tadpoles_kept_l236_236069

theorem tadpoles_kept (total_caught : ℕ) (percentage_released : ℚ)
  (h1 : total_caught = 180)
  (h2 : percentage_released = 75) : 
  let percentage_kept := 100 - percentage_released in
  let kept := percentage_kept / 100 * total_caught in
  kept = 45 :=
by 
  sorry

end tadpoles_kept_l236_236069


namespace impossible_load_two_coins_l236_236959

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l236_236959


namespace max_area_triangle_l236_236758

theorem max_area_triangle
  (A B C P Q R : Type)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space P] [metric_space Q] [metric_space R]
  (f : A → B) (g : B → C) (h : C → A)
  (d_AP : ℝ) (d_PQ : ℝ) (d_QR : ℝ) (d_RC : ℝ)
  (P_on_AB : P ∈ f A B) (Q_on_BC : Q ∈ g B C) (R_on_CA : R ∈ h C A)
  (BP : dist B P = 1) (PQ : dist P Q = 1) (QR : dist Q R = 1) (RC : dist R C = 1) :

  ∃ (S : ℝ), S = 2 := 
  sorry

end max_area_triangle_l236_236758


namespace x_coordinate_P_l236_236663

variables {x y m n: ℝ}
def ellipse (x y: ℝ) : Prop :=  x^2 / 4 + y^2 / 3 = 1
def right_focus (f: ℕ × ℕ) : Prop := (f = (1, 0))

variables {x₁ y₁ x₂ y₂: ℝ}
def line_l_through_points (x₁ y₁ x₂ y₂: ℝ) (F: ℕ × ℕ) : Prop := x₁ ≠ x₂ ∧ F ≠ (x₁, y₁) ∧ F ≠ (x₂, y₂)
def intersect_ellipse_points (x₁ y₁ x₂ y₂: ℝ) : Prop := ellipse x₁ y₁ ∧ ellipse x₂ y₂

def external_angle_bisector (x₁ y₁ x₂ y₂ m n: ℝ) (F: ℕ × ℕ) : Prop :=
  (| x₁-1 | / | x₂-1 |) = (x₁ - m) / (x₂ - m)

theorem x_coordinate_P (h1 : ellipse x₁ y₁) 
                      (h2 : ellipse x₂ y₂) 
                      (h3 : right_focus (1, 0)) 
                      (h4 : line_l_through_points x₁ y₁ x₂ y₂ (1, 0)) 
                      (h5 : intersect_ellipse_points x₁ y₁ x₂ y₂)
                      (h6 : external_angle_bisector x₁ y₁ x₂ y₂ 4 n (1, 0)) :
  m = 4 := sorry

end x_coordinate_P_l236_236663


namespace sine_symmetry_axis_is_pi_over_2_l236_236050

theorem sine_symmetry_axis_is_pi_over_2 :
  ∃ k : ℤ, (∀ x : ℝ, sin x = sin (2 * (k + 1) * π - x)) → k = 0 → x = π / 2 :=
by
  sorry

end sine_symmetry_axis_is_pi_over_2_l236_236050


namespace coin_loading_impossible_l236_236977

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l236_236977


namespace find_x_squared_plus_y_squared_l236_236331

variable (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : y + 7 = (x - 3)^2) (h2 : x + 7 = (y - 3)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 17 :=
by
  sorry  -- Proof to be provided

end find_x_squared_plus_y_squared_l236_236331


namespace inequality_sum_l236_236016

theorem inequality_sum {n : ℕ} (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) : 
  (∑ i in Finset.range n, (i + 1) / (∑ j in Finset.range (i + 1), a j)) < 
  4 * ∑ i in Finset.range n, (1 / a i) := 
sorry

end inequality_sum_l236_236016


namespace four_digit_numbers_with_property_l236_236302

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l236_236302


namespace james_bike_ride_percentage_l236_236739

theorem james_bike_ride_percentage :
  ∃ (d1 d2 : ℕ), d2 = 18 ∧ 
  d2 = (6 / 5) * d1 ∧ 
  d1 + d2 + (55.5 - (d1 + d2)) = 55.5 ∧ 
  let d3 := 55.5 - (d1 + d2) in
  (d3 - d2) / d2 * 100 = 25 := by
sorry

end james_bike_ride_percentage_l236_236739


namespace age_of_teacher_l236_236801

variables (S T : ℕ)

-- Conditions
def total_age_of_students (avg_student_age : ℕ) (num_students : ℕ) : ℕ := avg_student_age * num_students
def new_total_age_with_teacher (new_avg_age : ℕ) (num_people_with_teacher : ℕ) : ℕ := new_avg_age * num_people_with_teacher

-- Given data
def avg_student_age : ℕ := 15
def num_students : ℕ := 30
def new_avg_age : ℕ := 16
def num_people_with_teacher : ℕ := 31

-- Total age of students is 450
def S : ℕ := total_age_of_students avg_student_age num_students

-- New total age with teacher is 496
def St : ℕ := new_total_age_with_teacher new_avg_age num_people_with_teacher

-- Proof that the age of the teacher T is 46
theorem age_of_teacher : S + T = St → T = 46 :=
by
  sorry

end age_of_teacher_l236_236801


namespace bananas_used_l236_236593

-- Define the conditions
def bananas_per_loaf := 4
def loaves_monday := 3
def loaves_tuesday := 2 * loaves_monday

-- Define the total bananas used
def bananas_monday := loaves_monday * bananas_per_loaf
def bananas_tuesday := loaves_tuesday * bananas_per_loaf
def total_bananas := bananas_monday + bananas_tuesday

-- Theorem statement to prove the total bananas used is 36
theorem bananas_used : total_bananas = 36 := by
  sorry

end bananas_used_l236_236593


namespace total_new_people_l236_236745

theorem total_new_people (born : ℕ) (immigrated : ℕ) : born = 90171 → immigrated = 16320 → born + immigrated = 106491 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end total_new_people_l236_236745


namespace total_flour_amount_l236_236123

-- Define the initial amount of flour in the bowl
def initial_flour : ℝ := 2.75

-- Define the amount of flour added by the baker
def added_flour : ℝ := 0.45

-- Prove that the total amount of flour is 3.20 kilograms
theorem total_flour_amount : initial_flour + added_flour = 3.20 :=
by
  sorry

end total_flour_amount_l236_236123


namespace mike_initial_nickels_l236_236408

theorem mike_initial_nickels (quarters : ℕ) (borrowed_nickels : ℕ) (current_nickels : ℕ) 
  (h_borrowed : borrowed_nickels = 75) 
  (h_current : current_nickels = 12) : 
  let initial_nickels := current_nickels + borrowed_nickels in
  initial_nickels = 87 := 
by 
  sorry

end mike_initial_nickels_l236_236408


namespace logarithm_equation_l236_236099

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem logarithm_equation (a : ℝ) : 
  (1 / log_base 2 a + 1 / log_base 3 a + 1 / log_base 4 a = 1) → a = 24 :=
by
  sorry

end logarithm_equation_l236_236099


namespace nine_possible_xs_l236_236286

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end nine_possible_xs_l236_236286


namespace cost_of_pencil_pen_eraser_l236_236807

variables {p q r : ℝ}

theorem cost_of_pencil_pen_eraser 
  (h1 : 4 * p + 3 * q + r = 5.40)
  (h2 : 2 * p + 2 * q + 2 * r = 4.60) : 
  p + 2 * q + 3 * r = 4.60 := 
by sorry

end cost_of_pencil_pen_eraser_l236_236807


namespace typing_order_count_l236_236714

/-- The office problem: Given that letter 10 has been typed and
letters 1 through 9 could be in a stack with letters 11 and 12 possibly added,
compute the number of distinct orders to type the remaining letters. -/
theorem typing_order_count :
  let k (n : Nat) := Nat.binomial 9 n * (n + 1) * (n + 2) in
  (List.range 10).sum k = 5166 :=
by
  intro k
  rw [List.range, List.sum]
  sorry

end typing_order_count_l236_236714


namespace monotonic_invertible_function_l236_236601

theorem monotonic_invertible_function (f : ℝ → ℝ) (c : ℝ) (h_mono : ∀ x y, x < y → f x < f y) (h_inv : ∀ x, f (f⁻¹ x) = x) :
  (∀ x, f x + f⁻¹ x = 2 * x) ↔ ∀ x, f x = x + c :=
sorry

end monotonic_invertible_function_l236_236601


namespace five_eight_sided_dice_not_all_same_l236_236894

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  1 - (same_number_outcomes / total_outcomes)

theorem five_eight_sided_dice_not_all_same :
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end five_eight_sided_dice_not_all_same_l236_236894


namespace product_of_two_numbers_l236_236453

open_locale big_operators

noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem product_of_two_numbers (a b : ℕ) (h_lcm : lcm a b = 120) (h_gcd : gcd a b = 8) : a * b = 960 :=
by
  sorry

end product_of_two_numbers_l236_236453


namespace complement_intersection_l236_236398

noncomputable def A : Set ℝ := {x : ℝ | |x - 2| ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}

theorem complement_intersection (A B : Set ℝ) :
  (C : Set ℝ) = {x : ℝ | x ≤ 1 ∨ 5 < x} :=
by
  let A := {x : ℝ | |x - 2| ≤ 3}
  let B := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
  let intersection := {x : ℝ | 1 < x ∧ x ≤ 5}
  let complement := {x : ℝ | x ≤ 1 ∨ 5 < x}
  have h : complement = C := sorry
  exact h

end complement_intersection_l236_236398


namespace apex_projection_center_of_circle_l236_236467

variables {α : Type*} [linear_ordered_field α] {P : Type*} [add_comm_group P] [module α P]
variables (A B C D : P) -- Vertices of the pyramid
variable (h : α) -- Height from apex D to base plane ABC
variable (α : α) -- Equal dihedral angle between lateral faces and base plane

-- Assume the planes of the lateral faces form equal angles with the plane of the base via angle α
def planes_form_equal_angles_with_base (ABC_plane lateral_face_plane : set P) : Prop :=
  ∀ lateral_face ∈ {DAB, DBC, DCA} (abc : unit), 
  let α := dihedral_angle ABC_plane lateral_face_plane in
  α = α -- Given α as the angle of dihedral planes which is constant.

def projection_to_base_plane (D : P) (ABC_plane : set P) : P := sorry -- Definition of the projection

def is_incenter_or_excenter (P XYZ : P) : Prop := sorry -- Definition to check if P is incenter or excenter of triangle XYZ

theorem apex_projection_center_of_circle (planes_form_equal_angles_with_base : Prop) :
  let P := projection_to_base_plane D (affine_span ℝ {A, B, C}) in
  is_incenter_or_excenter P (affine_span ℝ {A, B, C}) :=
begin
  sorry -- Proof is omitted
end

end apex_projection_center_of_circle_l236_236467


namespace capital_of_a_l236_236499

variable (P : ℝ) -- Total profit
variable (C_A : ℝ) -- Capital of A
variable (income_increase : ℝ) (rate_increase : ℝ) -- Income increase and rate increase
variable (original_rate : ℝ) -- Original rate

-- Given conditions
axiom profit_share_a : ∀ P, a_profit = (2/3) * P
axiom income_increase_condition : income_increase = 200
axiom rate_change_condition : rate_increase = 0.02
axiom original_rate_condition : original_rate = 0.05

-- The formal statement to be proven
theorem capital_of_a (h : (2/3) * rate_increase * P = income_increase) :
  C_A = 300000 :=
by 
  let P := 15000
  let C_A := 300000
  -- sorry, proof steps are not given here
  sorry

end capital_of_a_l236_236499


namespace father_l236_236503

theorem father's_age (M F : ℕ) 
  (h1 : M = (2 / 5 : ℝ) * F)
  (h2 : M + 14 = (1 / 2 : ℝ) * (F + 14)) : 
  F = 70 := 
  sorry

end father_l236_236503


namespace calculate_f8_f4_l236_236752

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 3

theorem calculate_f8_f4 : f 8 - f 4 = -2 := by
  sorry

end calculate_f8_f4_l236_236752


namespace probability_at_least_four_at_least_four_times_in_five_rolls_l236_236995

noncomputable def probability_at_least_four (n : ℕ) : ℚ :=
  if n = 0 then 1 / 2 else (1 / 2) ^ n

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def prob_exactly_k_times (k n : ℕ) : ℚ :=
  (binomial_coefficient n k) * (probability_at_least_four k) * (probability_at_least_four (n - k))

theorem probability_at_least_four_at_least_four_times_in_five_rolls :
  let prob_four_times := prob_exactly_k_times 4 5
  let prob_five_times := prob_exactly_k_times 5 5
  prob_four_times + prob_five_times = 3 / 16 :=
by
  sorry

end probability_at_least_four_at_least_four_times_in_five_rolls_l236_236995


namespace calculator_squaring_number_l236_236522

theorem calculator_squaring_number (x_init : ℕ) : x_init = 5 → ∃ n : ℕ, n = 3 ∧ x_init^(2^n) > 10000 :=
by
  intros
  let display1 := x_init^2
  let display2 := display1^2
  let display3 := display2^2
  use 3
  { 
    split
    { 
      reflexivity
    },
    sorry
  }

end calculator_squaring_number_l236_236522


namespace rotten_apples_did_not_smell_l236_236417

theorem rotten_apples_did_not_smell:
  ∀ (total_apples rotten_percentage smelly_percentage : ℕ),
  total_apples = 200 →
  rotten_percentage = 40 →
  smelly_percentage = 70 →
  (total_apples * rotten_percentage / 100 - total_apples * rotten_percentage / 100 * smelly_percentage / 100) = 24 :=
by
  intros total_apples rotten_percentage smelly_percentage hab hbp hsp
  have h1 : total_apples * rotten_percentage / 100 = 80 := by sorry
  have h2 : (total_apples * rotten_percentage / 100) * smelly_percentage / 100 = 56 := by sorry
  show 80 - 56 = 24, by sorry

end rotten_apples_did_not_smell_l236_236417


namespace probability_not_all_same_l236_236914

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l236_236914


namespace find_a_value_l236_236269

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - real.sqrt 2

theorem find_a_value (a : ℝ) :
  (f a (f a (real.sqrt 2)) = -real.sqrt 2) ↔ a = real.sqrt 2 / 2 :=
by
  sorry

end find_a_value_l236_236269


namespace imaginary_part_of_complex_division_l236_236651

theorem imaginary_part_of_complex_division : 
  let i := Complex.I
  let z := (1 - 2 * i) / (2 - i)
  Complex.im z = -3 / 5 :=
by
  sorry

end imaginary_part_of_complex_division_l236_236651


namespace probability_same_parity_l236_236272

-- Define the functions
def f1 (x : ℝ) : ℝ := x^3 + 3*x^2
def f2 (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2
def f3 (x : ℝ) : ℝ := Real.log 2 ((3 - x) / (3 + x))
def f4 (x : ℝ) : ℝ := x * Real.sin x

-- Define even and odd properties
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) := ∀ x, f x = - f (-x)

-- Define the parity of the functions
def parity_f1 : Prop := ¬ is_even f1 ∧ ¬ is_odd f1
def parity_f2 : Prop := is_even f2
def parity_f3 : Prop := is_odd f3
def parity_f4 : Prop := is_even f4

-- Define the problem statement
theorem probability_same_parity : 
  let n := Nat.choose 4 2,
      m := Nat.choose 2 2,
  m / n = 1 / 6 := 
  by 
    let n := Nat.choose 4 2
    let m := Nat.choose 2 2
    have h : m = 1 := sorry
    have h' : n = 6 := sorry
    show m / n = 1 / 6 by 
      rw [h, h']
      norm_num

end probability_same_parity_l236_236272


namespace find_B_l236_236153

def AB_CB_add_CC6 (A B C : ℕ) : Prop :=
  let AB := 10 * A + B in
  let CB := 10 * C + B in
  ∃ (carry : ℕ), AB + CB = 100 * C + 10 * C + 6 + carry 

theorem find_B : 
  ∃ (A B C : ℕ), AB_CB_add_CC6 A B C ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A ≠ 0 ∧ C ≠ 0 ∧ B = 8 := sorry

end find_B_l236_236153


namespace inverse_of_half_l236_236261

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else log x / log 2

theorem inverse_of_half : f⁻¹(1 / 2) = -1 := by
  sorry

end inverse_of_half_l236_236261


namespace radius_of_circle_l236_236828

theorem radius_of_circle (p q : ℕ) (m n r : ℕ)
  (hp : p.prime) (hq : q.prime) (hm : 0 < m) (hn : 0 < n) (hr : r % 2 = 1)
  (h_on_circle : p^(2*m) + q^(2*n) = r^2) : r = 5 :=
sorry

end radius_of_circle_l236_236828


namespace smallest_positive_integer_divisible_by_14_15_16_l236_236213

theorem smallest_positive_integer_divisible_by_14_15_16 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (14 ∣ m) ∧ (15 ∣ m) ∧ (16 ∣ m) → n ≤ m) :=
  ∃ n : ℕ, n = 1680 ∧ ∀ m : ℕ, (14 ∣ m) ∧ (15 ∣ m) ∧ (16 ∣ m) ∧ m > 0 → n ≤ m

end smallest_positive_integer_divisible_by_14_15_16_l236_236213


namespace votes_to_win_l236_236506

theorem votes_to_win (total_votes : ℕ) (geoff_votes_percent : ℝ) (additional_votes : ℕ) (x : ℝ) 
(h1 : total_votes = 6000)
(h2 : geoff_votes_percent = 0.5)
(h3 : additional_votes = 3000)
(h4 : x = 50.5) :
  ((geoff_votes_percent / 100 * total_votes) + additional_votes) / total_votes * 100 = x :=
by
  sorry

end votes_to_win_l236_236506


namespace first_nonzero_digit_of_inv_127_l236_236080

def first_nonzero_decimal_digit (x : ℚ) : ℕ :=
  have hx : x ≠ 0 := sorry
  let x_digits := (x - x.floor) * 10 ^ (-(x.log10_ceil) + 1)
  (x_digits.toNat % 10).nat_abs

theorem first_nonzero_digit_of_inv_127 :
  first_nonzero_decimal_digit (1 / 127) = 7 :=
sorry

end first_nonzero_digit_of_inv_127_l236_236080


namespace subtractions_to_zero_l236_236324

-- Definitions used in the conditions
def initial_value : ℕ := 792
def subtract_value : ℕ := 8
def number_of_subtractions : ℕ := initial_value / subtract_value

-- The proof problem statement
theorem subtractions_to_zero :
  initial_value - number_of_subtractions * subtract_value = 0 :=
by
  have : initial_value = number_of_subtractions * subtract_value :=
    by sorry -- Placeholder for the proof on correctness of the division
  rw [this]
  rw [sub_eq_zero]
  exact zero_mul subtract_value

end subtractions_to_zero_l236_236324


namespace subset_iff_union_complement_eq_univ_l236_236009

variable (U : Type) (A B : Set U) (CU : U → Set U)

theorem subset_iff_union_complement_eq_univ :
  (B ⊆ A) ↔ (A ∪ (CU B) = Set.univ) :=
sorry

end subset_iff_union_complement_eq_univ_l236_236009


namespace find_a_value_l236_236270

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - real.sqrt 2

theorem find_a_value (a : ℝ) :
  (f a (f a (real.sqrt 2)) = -real.sqrt 2) ↔ a = real.sqrt 2 / 2 :=
by
  sorry

end find_a_value_l236_236270


namespace postage_count_is_three_l236_236439

def envelope (length height : ℕ) : ℕ × ℕ := (length, height)

def requires_extra_postage (env : ℕ × ℕ) : Bool :=
  let (length, height) := env
  let ratio := length.toFloat / height.toFloat
  ratio < 1.3 ∨ ratio > 2.5

def count_extra_postage (envelopes : List (ℕ × ℕ)) : Nat :=
  envelopes.countp requires_extra_postage

def envelopeA := envelope 6 4
def envelopeB := envelope 9 3
def envelopeC := envelope 6 6
def envelopeD := envelope 11 4

def envelopes := [envelopeA, envelopeB, envelopeC, envelopeD]

theorem postage_count_is_three : count_extra_postage envelopes = 3 := by
  sorry

end postage_count_is_three_l236_236439


namespace distance_between_planes_l236_236603

theorem distance_between_planes :
  let plane1 := λ (x y z : ℝ), 2 * x - 4 * y + 4 * z = 10 in
  let plane2 := λ (x y z : ℝ), 2 * x - 4 * y + 4 * z = 9 in
  let distance := λ (p1 p2 : ℝ), p1 = 10 /\ p2 = 9 -> (abs (p2 - p1) / (real.sqrt (2^2 + (-4)^2 + 4^2))) = 1 / 6 in
  distance 10 9 :=
by sorry

end distance_between_planes_l236_236603


namespace percentage_of_shaded_squares_l236_236940

theorem percentage_of_shaded_squares (total_squares shaded_squares : ℕ) (H1 : total_squares = 49) (H2 : shaded_squares = 20) : 
  (shaded_squares : ℝ) / (total_squares : ℝ) * 100 ≈ 40.82 := by
sorry

end percentage_of_shaded_squares_l236_236940


namespace PS_eq_QS_l236_236482

noncomputable def problem (A B P Q S : Point) (C1 C2 C3 : Circle) : Prop :=
  ∃ C1 C2 C3,
    (C1 ∩ C2 = {S}) ∧
    (is_tangent C1 S) ∧ (is_tangent C2 S) ∧
    (tangent_point C1 S ≠ S) ∧ (tangent_point C2 S ≠ S) ∧
    (is_on_circle C3 A) ∧ (is_on_circle C3 B) ∧ (is_on_circle C3 S) ∧
    (tangent_point C3 S ≠ S)

theorem PS_eq_QS (A B P Q S : Point) (C1 C2 C3 : Circle)
  (h1 : problem A B P Q S C1 C2 C3) : dist P S = dist Q S :=
sorry

end PS_eq_QS_l236_236482


namespace angle_ABC_twice_angle_BAC_l236_236367

noncomputable def is_bisector (A B C D : Point) : Prop := 
∡ ABD = ∡ DBC

noncomputable def are_parallel (L1 L2 : Line) : Prop := 
∀ {P Q R S : Point}, P ∈ L1 ∧ Q ∈ L1 ∧ R ∈ L2 ∧ S ∈ L2 → 
  (∡ PQR = ∡ QRS ∨ ∡ QRP = ∡ PQS)

variables {A B C D E F : Point}
variables {BD EF : Line} 

axiom triangle_ABC : is_triangle A B C
axiom triangle_BDC : is_triangle B C D
axiom triangle_DEC : is_triangle D E C

axiom bisector_BD : is_bisector A B C D
axiom bisector_DE : is_bisector B D C E
axiom bisector_EF : is_bisector D E C F
axiom parallel_BD_EF : are_parallel BD EF

theorem angle_ABC_twice_angle_BAC :
  ∡ ABC = 2 * ∡ BAC := 
sorry

end angle_ABC_twice_angle_BAC_l236_236367


namespace determine_x_l236_236955

theorem determine_x (x r1 r2 r3 : ℕ) (h1 : x % 3 = r1) (h2 : x % 5 = r2) (h3 : x % 7 = r3) (h4 : 1 ≤ x) (h5 : x ≤ 100) :
  x ≡ (70 * r1 + 21 * r2 + 15 * r3) % 105 :=
by
  sorry

end determine_x_l236_236955


namespace solve_quadratic_eq_l236_236028

theorem solve_quadratic_eq (a b x : ℝ) :
  12 * a * b * x^2 - (16 * a^2 - 9 * b^2) * x - 12 * a * b = 0 ↔ (x = 4 * a / (3 * b)) ∨ (x = -3 * b / (4 * a)) :=
by
  sorry

end solve_quadratic_eq_l236_236028


namespace probability_not_all_same_l236_236897

theorem probability_not_all_same :
  (let total_outcomes := 8 ^ 5 in
   let same_number_outcomes := 8 in
   let probability_all_same := same_number_outcomes / total_outcomes in
   let probability_not_same := 1 - probability_all_same in
   probability_not_same = 4095 / 4096) :=
begin
  sorry
end

end probability_not_all_same_l236_236897


namespace billy_total_questions_l236_236109

theorem billy_total_questions :
  ∃ (x : ℕ), 3 * x = 132 ∧ (x + 2 * x + 3 * x) = 264 :=
begin
  sorry
end

end billy_total_questions_l236_236109


namespace min_value_of_exponential_difference_l236_236609

open Real

theorem min_value_of_exponential_difference :
  ∃ x : ℝ, 3^x - 9^x = 1/4 :=
by
  sorry

end min_value_of_exponential_difference_l236_236609


namespace playground_area_22500_l236_236454

noncomputable def rectangle_playground_area (w l : ℕ) : ℕ :=
  w * l

theorem playground_area_22500 (w l : ℕ) (h1 : l = 2 * w + 25) (h2 : 2 * l + 2 * w = 650) :
  rectangle_playground_area w l = 22500 := by
  sorry

end playground_area_22500_l236_236454


namespace tangent_line_through_point_l236_236449

theorem tangent_line_through_point (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) : 
  (∃ k : ℝ, 15 * x - 8 * y - 13 = 0) ∨ x = 3 := sorry

end tangent_line_through_point_l236_236449


namespace xenia_weekly_earnings_l236_236105

theorem xenia_weekly_earnings
  (hours_week_1 : ℕ)
  (hours_week_2 : ℕ)
  (week2_additional_earnings : ℕ)
  (hours_week_3 : ℕ)
  (bonus_week_3 : ℕ)
  (hourly_wage : ℚ)
  (earnings_week_1 : ℚ)
  (earnings_week_2 : ℚ)
  (earnings_week_3 : ℚ)
  (total_earnings : ℚ) :
  hours_week_1 = 18 →
  hours_week_2 = 25 →
  week2_additional_earnings = 60 →
  hours_week_3 = 28 →
  bonus_week_3 = 30 →
  hourly_wage = (60 : ℚ) / (25 - 18) →
  earnings_week_1 = hours_week_1 * hourly_wage →
  earnings_week_2 = hours_week_2 * hourly_wage →
  earnings_week_2 = earnings_week_1 + 60 →
  earnings_week_3 = hours_week_3 * hourly_wage + 30 →
  total_earnings = earnings_week_1 + earnings_week_2 + earnings_week_3 →
  hourly_wage = (857 : ℚ) / 1000 ∧
  total_earnings = (63947 : ℚ) / 100
:= by
  intros h1 h2 h3 h4 h5 hw he1 he2 he2_60 he3 hte
  sorry

end xenia_weekly_earnings_l236_236105


namespace probability_of_desired_roll_l236_236770

-- Definitions of six-sided dice rolls and probability results
def is_greater_than_four (n : ℕ) : Prop := n > 4
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5

-- Definitions of probabilities based on dice outcomes
def prob_greater_than_four : ℚ := 2 / 6
def prob_prime : ℚ := 3 / 6

-- Definition of joint probability for independent events
def joint_prob : ℚ := prob_greater_than_four * prob_prime

-- Theorem to prove
theorem probability_of_desired_roll : joint_prob = 1 / 6 := 
by
  sorry

end probability_of_desired_roll_l236_236770


namespace total_selling_price_calculation_l236_236137

noncomputable def laptop_price : ℝ := 1200
noncomputable def discount_percentage : ℝ := 0.30
noncomputable def tax_percentage : ℝ := 0.12

theorem total_selling_price_calculation :
  let discount := discount_percentage * laptop_price in
  let sale_price := laptop_price - discount in
  let tax := tax_percentage * sale_price in
  let total_selling_price := sale_price + tax in
  total_selling_price = 940.8 :=
by
  sorry

end total_selling_price_calculation_l236_236137


namespace base_conversion_sum_l236_236197

-- Definition of conversion from base 13 to base 10
def base13_to_base10 (n : ℕ) : ℕ :=
  3 * (13^2) + 4 * (13^1) + 5 * (13^0)

-- Definition of conversion from base 14 to base 10 where C = 12 and D = 13
def base14_to_base10 (m : ℕ) : ℕ :=
  4 * (14^2) + 12 * (14^1) + 13 * (14^0)

theorem base_conversion_sum :
  base13_to_base10 345 + base14_to_base10 (4 * 14^2 + 12 * 14 + 13) = 1529 := 
by
  sorry -- proof to be provided

end base_conversion_sum_l236_236197


namespace increasing_interval_of_translated_sine_l236_236845

theorem increasing_interval_of_translated_sine :
  ∀ x : ℝ, 
    (y = 3 * sin (2 * x - 2 * π / 3)) → 
    (π / 12 ≤ x ∧ x ≤ 7 * π / 12) → 
    ∀ x₁ x₂, (π / 12 ≤ x₁) → (x₁ < x₂) → (x₂ ≤ 7 * π / 12) → y x₁ < y x₂ :=
sorry

end increasing_interval_of_translated_sine_l236_236845


namespace good_number_10_l236_236104

def good_number : ℕ → ℕ 
| 0 := 1
| 1 := 2
| 2 := 4
| n := 2 * good_number (n-1) + good_number (n-2)

theorem good_number_10 : good_number 10 = 4756 :=
by
  sorry

end good_number_10_l236_236104


namespace min_value_of_f_in_interval_smallest_value_of_f_in_interval_l236_236048

def f (x : ℝ) : ℝ :=
  |x| + |(1 - 2013 * x) / (2013 - x)|

theorem min_value_of_f_in_interval : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), (f x) ≥ (1 / 2013 : ℝ) :=
by
  sorry

theorem smallest_value_of_f_in_interval : ∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), (f x) = (1 / 2013 : ℝ) :=
by
  sorry

end min_value_of_f_in_interval_smallest_value_of_f_in_interval_l236_236048


namespace remainder_div_19_l236_236504

theorem remainder_div_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
sorry

end remainder_div_19_l236_236504


namespace sum_of_series_eq_l236_236370

open BigOperators

theorem sum_of_series_eq (n : ℕ) (h : 0 < n) : 
  ∑ i in Finset.range n, (i.succ ^ 2) / ((2 * i.succ - 1) * (2 * i.succ + 1)) = (n * n + n) / (4 * n + 2) := by
  sorry

end sum_of_series_eq_l236_236370


namespace smallest_x_absolute_value_l236_236611

theorem smallest_x_absolute_value : ∃ x : ℤ, |x + 3| = 15 ∧ ∀ y : ℤ, |y + 3| = 15 → x ≤ y :=
sorry

end smallest_x_absolute_value_l236_236611


namespace sequence_geometric_sum_expression_l236_236726

section Problem1

variable (a : ℕ → ℤ) (b : ℕ → ℤ → ℚ) (a₁ : a 1 = -1) (rec : ∀ n, a (n + 1) = 2 * a n - n + 1)

theorem sequence_geometric :
  let s := λ n => a n - n
  ∃ r : ℤ, r = 2 ∧ ∀ n, s (n + 1) = r * s n ∧ s 1 = -2 := 
sorry

end Problem1

section Problem2

variable (a : ℕ → ℤ) (b : ℕ → ℚ) (a₁ : a 1 = -1) (rec : ∀ n, a (n + 1) = 2 * a n - n + 1)

noncomputable def b (n : ℕ) :=
  (a n) / (2 ^ n : ℤ)

noncomputable def S (n : ℕ) := 
  ∑ i in Finset.range n, b (i + 1)

theorem sum_expression (n : ℕ) :
  S n = 2 - (n + 2) / (2 ^ n : ℤ) - n := 
sorry

end Problem2

end sequence_geometric_sum_expression_l236_236726


namespace smallest_sum_two_3_digit_numbers_l236_236493

theorem smallest_sum_two_3_digit_numbers : ∃ (a b c d e f : ℕ), 
  {a, b, c, d, e, f} = {1, 2, 3, 7, 8, 9} ∧ 
  a ≠ d ∧ b ≠ e ∧ c ≠ f ∧
  a < b ∧ b < c ∧ d < e ∧ e < f ∧
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) = 417 :=
sorry

end smallest_sum_two_3_digit_numbers_l236_236493


namespace circumcircle_radius_l236_236341

theorem circumcircle_radius (b A S : ℝ) (h_b : b = 2) 
  (h_A : A = 120 * Real.pi / 180) (h_S : S = Real.sqrt 3) : 
  ∃ R, R = 2 := 
by
  sorry

end circumcircle_radius_l236_236341


namespace trigonometric_identity_l236_236628

theorem trigonometric_identity 
  (θ : ℝ) (k : ℤ)
  (h : ∀ (k : ℤ), sin (θ + k * π) = -2 * cos (θ + k * π)) :
  (4 * sin θ - 2 * cos θ) / (5 * cos θ + 3 * sin θ) = 10 :=
by {
  sorry
}

end trigonometric_identity_l236_236628


namespace base_10_to_base_6_l236_236489

theorem base_10_to_base_6 (n : ℕ) (h : n = 515) : ∃ k : ℕ, k = 2215 ∧ (515:ℕ) = k % 6 + 6 * (k / 6 % 6) + 6^2 * (k / 6^2 % 6) + 6^3 * (k / 6^3 % 6) :=
by
  use 2215
  split
  . refl
  . calc
    515 = 2215 % 6 + (6 * (2215 / 6 % 6) + (6^2 * (2215 / 6^2 % 6) + (6^3 * (2215 / 6^3 % 6)))) : sorry

end base_10_to_base_6_l236_236489


namespace defective_percentage_is_correct_l236_236558

noncomputable def percentage_defective (defective : ℕ) (total : ℝ) : ℝ := 
  (defective / total) * 100

theorem defective_percentage_is_correct : 
  percentage_defective 2 3333.3333333333335 = 0.06000600060006 :=
by
  sorry

end defective_percentage_is_correct_l236_236558


namespace sum_first_100_triangular_numbers_l236_236810

theorem sum_first_100_triangular_numbers :
  (∑ n in Finset.range 101, n * (n + 1) / 2) = 171700 := 
by
  sorry

end sum_first_100_triangular_numbers_l236_236810


namespace fireflies_joined_l236_236773

theorem fireflies_joined (x : ℕ) : 
  let initial_fireflies := 3
  let flew_away := 2
  let remaining_fireflies := 9
  initial_fireflies + x - flew_away = remaining_fireflies → x = 8 := by
  sorry

end fireflies_joined_l236_236773


namespace remainder_three_l236_236102

-- Define the condition that x % 6 = 3
def condition (x : ℕ) : Prop := x % 6 = 3

-- Proof statement that if condition is met, then (3 * x) % 6 = 3
theorem remainder_three {x : ℕ} (h : condition x) : (3 * x) % 6 = 3 :=
sorry

end remainder_three_l236_236102


namespace chess_team_girls_l236_236526

theorem chess_team_girls (B G : ℕ) (h1 : B + G = 26) (h2 : (G / 2) + B = 16) : G = 20 := by
  sorry

end chess_team_girls_l236_236526


namespace cos_angle_BAC_l236_236720

-- Define points A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, -1)

-- Define a function to calculate the distance between two points
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt((Q.fst - P.fst)^2 + (Q.snd - P.snd)^2)

-- Distances AB, AC, and BC
def AB := dist A B
def AC := dist A C
def BC := dist B C

-- Define a function for cosine using the Law of Cosines
def cos_angle (d1 d2 d3 : ℝ) : ℝ :=
  (d1^2 + d3^2 - d2^2) / (2 * d1 * d3)

-- Theorem to prove that the cosine of angle BAC is sqrt(10)/10
theorem cos_angle_BAC : cos_angle AB BC AC = real.sqrt(10) / 10 := by
  sorry

end cos_angle_BAC_l236_236720


namespace log32_eq_four_fifth_l236_236598

theorem log32_eq_four_fifth :
  log 32 4 = 2 / 5 := by
  sorry

end log32_eq_four_fifth_l236_236598


namespace focus_of_parabola_l236_236444

theorem focus_of_parabola (a : ℝ) (h : a = 3) : 
  let focus := (0, 1 / 12) in
  ∃ p, p = 1 / 12 ∧ focus = (0, p) :=
by 
  have p_eq : 1 / (3 * 4) = 1 / 12 := by norm_num,
  use 1 / 12,
  exact ⟨p_eq, rfl⟩

end focus_of_parabola_l236_236444


namespace no_possible_arrangement_l236_236369

theorem no_possible_arrangement :
  ¬ ∃ (M : Matrix (Fin 4) (Fin 4) ℤ),
  (∀ i, ∑ j, M i j = 38) ∧
  ∀ i j1 j2, (j1 ≠ j2 → ¬ (M i j1 ∣ M i j2 ∨ M i j2 ∣ M i j1))
  where
    Matrix := List (List Int)
    from_2_to_17 := (2 : 17 : [])
:= sorry

end no_possible_arrangement_l236_236369


namespace probability_not_all_same_l236_236930

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l236_236930


namespace increasing_on_interval_l236_236765

theorem increasing_on_interval : 
  ∀ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), 
    deriv (λ x : ℝ, -3 * x ^ 2 + 1) x > 0 :=
by
  sorry

end increasing_on_interval_l236_236765


namespace relationship_among_a_b_c_l236_236232

noncomputable def a : ℝ := ∫ x in 0..1, x
noncomputable def b : ℝ := ∫ x in 0..1, x^2
noncomputable def c : ℝ := ∫ x in 0..1, Real.sqrt x

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l236_236232


namespace corresponding_angles_not_always_equal_l236_236827

theorem corresponding_angles_not_always_equal : ¬ (∀ (l1 l2 : Line) (a1 a2 : Angle),
  corresponding_angles l1 l2 a1 a2 → a1 = a2) :=
sorry

end corresponding_angles_not_always_equal_l236_236827


namespace largest_x_satisfies_eq_l236_236885

theorem largest_x_satisfies_eq (x : ℝ) (hx : sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 :=
sorry

end largest_x_satisfies_eq_l236_236885


namespace Maria_Ivanovna_grades_l236_236404

theorem Maria_Ivanovna_grades :
  let a : ℕ → ℕ
  a 1 = 3 ∧ 
  a 2 = 8 ∧ 
  (∀ n, n ≥ 3 → a n = 2 * a (n - 1) + 2 * a (n - 2)) →
  a 6 = 448 :=
by
  intros
  sorry

end Maria_Ivanovna_grades_l236_236404


namespace maximize_expr_l236_236677

def set_a := {-3, -2, -1, 0, 1, 2, 3}

noncomputable def v := 2

theorem maximize_expr (y z : ℤ) (h₁ : y ∈ set_a) (h₂ : z ∈ set_a) (h₃ : ∀ (x : ℤ), x ∈ set_a → v * x - y * z ≤ 15) : y = -3 ∧ z = 3 :=
by
  have H : ∀ (x : ℤ), x ∈ set_a → v * x - y * z ≤ 15 := h₃
  sorry

end maximize_expr_l236_236677


namespace day_of_week_50th_day_of_year_N_minus_1_l236_236734

def day_of_week (d : ℕ) (first_day : ℕ) : ℕ :=
  (first_day + d - 1) % 7

theorem day_of_week_50th_day_of_year_N_minus_1 
  (N : ℕ) 
  (day_250_N : ℕ) 
  (day_150_N_plus_1 : ℕ) 
  (h1 : day_250_N = 3)  -- 250th day of year N is Wednesday (3rd day of week, 0 = Sunday)
  (h2 : day_150_N_plus_1 = 3) -- 150th day of year N+1 is also Wednesday (3rd day of week, 0 = Sunday)
  : day_of_week 50 (day_of_week 1 ((day_of_week 1 day_250_N - 1 + 250) % 365 - 1 + 366)) = 6 := 
sorry

-- Explanation:
-- day_of_week function calculates the day of the week given the nth day of the year and the first day of the year.
-- Given conditions that 250th day of year N and 150th day of year N+1 are both Wednesdays (represented by 3 assuming Sunday = 0).
-- We need to derive that the 50th day of year N-1 is a Saturday (represented by 6 assuming Sunday = 0).

end day_of_week_50th_day_of_year_N_minus_1_l236_236734


namespace scientific_notation_of_0_0000007_l236_236428

theorem scientific_notation_of_0_0000007 :
  0.0000007 = 7 * 10 ^ (-7) :=
  by
  sorry

end scientific_notation_of_0_0000007_l236_236428


namespace cistern_fill_time_l236_236132

theorem cistern_fill_time (filling_rate emptying_rate : ℚ)
  (h₁ : filling_rate = 1/5) (h₂ : emptying_rate = 1/9) : 
  (1 / (filling_rate - emptying_rate)) = 45/4 :=
by
  have net_rate := filling_rate - emptying_rate
  rw [h₁, h₂] at net_rate
  norm_num at net_rate
  rw [← net_rate]
  norm_num
  sorry

end cistern_fill_time_l236_236132


namespace lcm_28_72_l236_236605

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l236_236605


namespace number_of_monomials_degree_and_coefficient_terms_of_f_degree_and_term_count_l236_236841

-- Definitions
def f (x y : ℝ) := x^4 + 3 * x * y - 2 * x * y^4 - 5 * x^3 * y^3 - 1
def const_expr : ℝ := 2024
def linear_expr (x : ℝ) := -x

-- Proof Statements
theorem number_of_monomials : 
  (@has_monomial.f f = 5) ∧ (@has_monomial.constant const_expr = 1) ∧ (@has_monomial.scaled linear_expr = 1) :=
sorry

theorem degree_and_coefficient : 
  (degree const_expr = 0) ∧ (coefficient (linear_expr 1) = -1) :=
sorry

theorem terms_of_f :
  (quadratic_term f = none) ∧ (constant_term f = -1) :=
sorry

theorem degree_and_term_count :
  (degree f = 5) ∧ (term_count f = 5) :=
sorry

end number_of_monomials_degree_and_coefficient_terms_of_f_degree_and_term_count_l236_236841


namespace combined_population_lake_bright_and_sunshine_hills_l236_236615

theorem combined_population_lake_bright_and_sunshine_hills
  (p_toadon p_gordonia p_lake_bright p_riverbank p_sunshine_hills : ℕ)
  (h1 : p_toadon + p_gordonia + p_lake_bright + p_riverbank + p_sunshine_hills = 120000)
  (h2 : p_gordonia = 1 / 3 * 120000)
  (h3 : p_toadon = 3 / 4 * p_gordonia)
  (h4 : p_riverbank = p_toadon + 2 / 5 * p_toadon) :
  p_lake_bright + p_sunshine_hills = 8000 :=
by
  sorry

end combined_population_lake_bright_and_sunshine_hills_l236_236615


namespace farmer_rewards_l236_236994

theorem farmer_rewards (x y : ℕ) (h1 : x + y = 60) (h2 : 1000 * x + 3000 * y = 100000) : x = 40 ∧ y = 20 :=
by {
  sorry
}

end farmer_rewards_l236_236994


namespace slope_range_l236_236830

theorem slope_range (α : Real) (hα : -1 ≤ Real.cos α ∧ Real.cos α ≤ 1) :
  ∃ k ∈ Set.Icc (- Real.sqrt 3 / 3) (Real.sqrt 3 / 3), ∀ x y : Real, x * Real.cos α - Real.sqrt 3 * y - 2 = 0 → y = k * x - (2 / Real.sqrt 3) :=
by
  sorry

end slope_range_l236_236830


namespace number_of_ordered_pairs_l236_236209

noncomputable def eighth_roots_of_unity : set ℂ :=
  { z | z^8 = 1 }

noncomputable def third_roots_of_unity : set ℂ :=
  { z | z^3 = 1 }

theorem number_of_ordered_pairs :
  (∑ a in eighth_roots_of_unity, ∑ b in third_roots_of_unity, (a^4 * b^6 = 1) ∧ (a^8 * b^3 = 1)) = 24 :=
by sorry

end number_of_ordered_pairs_l236_236209


namespace print_width_l236_236141

noncomputable def aspect_ratio (h w : ℝ) : ℝ := w / h

theorem print_width (h_paint h_print w_paint : ℝ) 
  (h_paint_pos : h_paint = 10) 
  (w_paint_pos : w_paint = 15) 
  (h_print_pos : h_print = 25) : 
  (w_paint * h_print / h_paint) = 37.5 := 
by 
  have r : ℝ := aspect_ratio 10 15 
  have aspect_ratio_calc : r = 1.5 := by 
    simp [aspect_ratio, h_paint_pos, w_paint_pos]
  have width_calc : (w_paint * h_print / h_paint) = h_print * r := by
    rw [aspect_ratio]
    rw [h_paint_pos, w_paint_pos]
    simp
  rw [width_calc]
  rw [h_print_pos, aspect_ratio_calc]
  norm_num

end print_width_l236_236141


namespace problem_statement_l236_236753

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 2 * x - 1

theorem problem_statement : f (g 3) - g (f 3) = -5 := by
  sorry

end problem_statement_l236_236753


namespace count_numbers_containing_zero_l236_236685

def contains_zero (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ (n / (10^k) % 10) = 0

theorem count_numbers_containing_zero : 
  (finset.filter contains_zero (finset.Icc 1 2500)).card = 651 := 
sorry

end count_numbers_containing_zero_l236_236685


namespace min_colors_tessellation_l236_236547

theorem min_colors_tessellation (hex_adj_squares : ℕ) (sqr_adj_hexagons : ℕ) :
  (hex_adj_squares = 6 ∧ sqr_adj_hexagons = 4) → ∃ n : ℕ, n = 5 := 
by
  intros h
  let min_colors := 5
  use min_colors
  sorry

end min_colors_tessellation_l236_236547


namespace functional_equation_l236_236581

noncomputable def f : ℚ+ → ℚ+ :=
sorry

theorem functional_equation (x y : ℚ+) : f (x * f y) = f x / y :=
sorry

end functional_equation_l236_236581


namespace necessary_but_not_sufficient_l236_236981

noncomputable def condition_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < (real.pi / 2) → k * (real.sin x) * (real.cos x) < x

theorem necessary_but_not_sufficient (k : ℝ) : 
  (condition_holds k → k < 1) ∧ (¬(k < 1) → ¬condition_holds k) :=
sorry

end necessary_but_not_sufficient_l236_236981


namespace shaded_area_is_correct_l236_236181

noncomputable def octagon_side_length := 3
noncomputable def octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side_length^2
noncomputable def semicircle_radius := octagon_side_length / 2
noncomputable def semicircle_area := (1 / 2) * Real.pi * semicircle_radius^2
noncomputable def total_semicircle_area := 8 * semicircle_area
noncomputable def shaded_region_area := octagon_area - total_semicircle_area

theorem shaded_area_is_correct : shaded_region_area = 54 + 36 * Real.sqrt 2 - 9 * Real.pi :=
by
  -- Proof goes here, but we're inserting sorry to skip it
  sorry

end shaded_area_is_correct_l236_236181


namespace Nicole_fewer_questions_l236_236410

-- Definitions based on the given conditions
def Nicole_correct : ℕ := 22
def Cherry_correct : ℕ := 17
def Kim_correct : ℕ := Cherry_correct + 8

-- Theorem to prove the number of fewer questions Nicole answered compared to Kim
theorem Nicole_fewer_questions : Kim_correct - Nicole_correct = 3 :=
by
  -- We set up the definitions
  let Nicole_correct := 22
  let Cherry_correct := 17
  let Kim_correct := Cherry_correct + 8
  -- The proof will be filled in here. 
  -- The goal theorem statement is filled with 'sorry' to bypass the actual proof.
  have : Kim_correct - Nicole_correct = 3 := sorry
  exact this

end Nicole_fewer_questions_l236_236410


namespace lloyd_total_earnings_l236_236769

theorem lloyd_total_earnings
  (hourly_rate : ℕ)
  (overtime_multiplier : ℝ)
  (saturday_multiplier : ℝ)
  (normal_hours : ℝ)
  (monday_hours : ℝ)
  (tuesday_hours : ℝ)
  (saturday_hours : ℝ) :
  hourly_rate = 5 →
  overtime_multiplier = 1.5 →
  saturday_multiplier = 2 →
  normal_hours = 8 →
  monday_hours = 10.5 →
  tuesday_hours = 9 →
  saturday_hours = 6 →
  let monday_pay := normal_hours * hourly_rate + (monday_hours - normal_hours) * hourly_rate * overtime_multiplier,
      tuesday_pay := normal_hours * hourly_rate + (tuesday_hours - normal_hours) * hourly_rate * overtime_multiplier,
      saturday_pay := saturday_hours * hourly_rate * saturday_multiplier,
      total_earnings := monday_pay + tuesday_pay + saturday_pay
  in total_earnings = 166.25 :=
by intros; sorry

end lloyd_total_earnings_l236_236769


namespace number_of_arrangements_l236_236058

section

variables (teachers schools : Fin 4)

noncomputable def arrangements : ℕ := (Finset.univ.perm.univ : Finset (Equiv.Perm (Fin 4))).card

theorem number_of_arrangements : arrangements teachers schools = 24 := 
sorry

end

end number_of_arrangements_l236_236058


namespace find_a_l236_236751

noncomputable def a_b_c_complex (a b c : ℂ) : Prop :=
  a.re = a ∧ a + b + c = 4 ∧ a * b + b * c + c * a = 6 ∧ a * b * c = 8

theorem find_a (a b c : ℂ) (h : a_b_c_complex a b c) : a = 3 :=
by
  sorry

end find_a_l236_236751


namespace value_of_expression_l236_236693

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := 
by 
  have h₁ : x^2 - 2*x = 2 := by linarith
  calc
    3*x^2 - 6*x + 9 = 3*(x^2 - 2*x) + 9 : by ring
                ... = 3*2 + 9           : by rw [h₁]
                ... = 15                : by norm_num

end value_of_expression_l236_236693


namespace find_last_number_l236_236111

theorem find_last_number
  (A1 A2 A3 A4 A5 A6 A7 A8 : ℝ)
  (h_sum_8 : A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 = 200)
  (h_sum_2 : A1 + A2 = 40)
  (h_sum_3 : A3 + A4 + A5 = 78)
  (h_rel_6_7 : A6 + 4 = A7)
  (h_rel_6_8 : A6 + 6 = A8) : A8 = 30 := 
sorry

end find_last_number_l236_236111


namespace mod_congruence_l236_236332

theorem mod_congruence {x : ℤ} (h : 4 * x + 9 ≡ 3 [MOD 20]) : 
  3 * x + 15 ≡ 10 [MOD 20] :=
sorry

end mod_congruence_l236_236332


namespace remainder_of_power_modulo_l236_236088

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end remainder_of_power_modulo_l236_236088


namespace find_a_l236_236055

theorem find_a (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 = 180)
  (h2 : x2 = 182)
  (h3 : x3 = 173)
  (h4 : x4 = 175)
  (h6 : x6 = 178)
  (h7 : x7 = 176)
  (h_avg : (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 178) : x5 = 182 := by
  sorry

end find_a_l236_236055


namespace probability_not_all_dice_show_different_l236_236920

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l236_236920


namespace swimmer_speed_in_still_water_l236_236546

-- Define the conditions
def current_speed : ℝ := 2   -- Speed of the water current is 2 km/h
def swim_time : ℝ := 2.5     -- Time taken to swim against current is 2.5 hours
def distance : ℝ := 5        -- Distance swum against current is 5 km

-- Main theorem proving the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) (h : v - current_speed = distance / swim_time) : v = 4 :=
by {
  -- Skipping the proof steps as per the requirements
  sorry
}

end swimmer_speed_in_still_water_l236_236546


namespace product_of_possible_values_l236_236362

theorem product_of_possible_values : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 2 ∨ x = 8)) → (2 * 8) = 16 :=
by 
  sorry

end product_of_possible_values_l236_236362


namespace finance_specialization_percentage_l236_236705

theorem finance_specialization_percentage (F : ℝ) :
  (76 - 43.333333333333336) = (90 - F) → 
  F = 57.333333333333336 :=
by
  sorry

end finance_specialization_percentage_l236_236705


namespace basketball_team_lineups_l236_236178

theorem basketball_team_lineups (n k : ℕ) (h_n : n = 20) (h_k : k = 5) :
  (∃ lineup : ℕ, lineup = 20 * Nat.choose 19 4 ∧ lineup = 77520) :=
by
  let point_guard_choice := 20
  let remaining_choices := Nat.choose 19 4
  have : point_guard_choice * remaining_choices = 77520 := by sorry
  use point_guard_choice * remaining_choices
  split
  · 
    exact rfl
    
  ·
    assumption

end basketball_team_lineups_l236_236178


namespace range_f_l236_236584

def operation (a b : ℝ) : ℝ :=
if a ≤ b then a else b

def f (x : ℝ) : ℝ := abs ((operation (2^x) (2^(-x))) - 1)

theorem range_f : set.range f = set.Ico 0 1 :=
by
  sorry

end range_f_l236_236584


namespace ellipse_properties_l236_236244

theorem ellipse_properties (a b : ℝ) (h : a > b) (h_b : b > 0) (h_a : a^2 = b^2 + 4) :
  (∀ x y : ℝ, x = 2 → y = -real.sqrt 2 → x^2 / a^2 + y^2 / b^2 = 1) →
  (∀ k : ℝ, k ≠ 0 →
   ∃ x0 : ℝ, (x0 = 2 ∨ x0 = -2) → 
             ∀ x1 y1 : ℝ, y1 = k * x1 → 
             (1 + 2 * k^2) * x1^2 - 8 = 0 →
             y1 = (2 * real.sqrt 2 * k) / real.sqrt (1 + 2 * k^2) →
             (λ (M N : ℝ × ℝ), M = (0, ((2 * real.sqrt 2 * k) / (1 + real.sqrt (1 + 2 * k^2)))) ∧
                             N = (0, ((2 * real.sqrt 2 * k) / (1 - real.sqrt (1 + 2 * k^2)))) → 
                  ∃ P : ℝ × ℝ, (P = (x0, 0)) → 
                                let PM := (-x0, ((2 * real.sqrt 2 * k) / (1 + real.sqrt (1 + 2 * k^2)))) in
                                let PN := (-x0, ((2 * real.sqrt 2 * k) / (1 - real.sqrt (1 + 2 * k^2)))) in
                                PM.1 * PN.1 + PM.2 * PN.2 = 0)) :=
sorry

end ellipse_properties_l236_236244


namespace count_possible_values_of_x_l236_236283

theorem count_possible_values_of_x :
  let n := (set.count {x : ℕ | 25 ≤ x ∧ x ≤ 33 ∧ ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c * x < 100 ∧ 3 ≤ b ≤ 100/x}) in
  n = 9 :=
by
  -- Here we must prove the statement by the provided conditions
  sorry

end count_possible_values_of_x_l236_236283


namespace mu_sq_minus_lambda_sq_l236_236458

noncomputable section

open Real

def point (α : Type _) := (α × α)
def vec (α : Type _) := point α

def line_eqn := λ x y : ℝ, (sqrt 3) * x - y - (sqrt 3) = 0
def parabola_eqn := λ x y : ℝ, y^2 = 4 * x

def F : point ℝ := (1, 0)
def A : point ℝ := (3, 2 * sqrt 3)
def B : point ℝ := ((1/3), -(2 * sqrt 3) / 3)

def vec_OA := A
def vec_OB := B
def vec_OF := F

def lambda := (1 / 4 : ℝ)
def mu := (3 / 4 : ℝ)

theorem mu_sq_minus_lambda_sq :
  let µ := mu
  let λ := lambda
  (µ^2 - λ^2) = 1 / 2 := 
by
  sorry

end mu_sq_minus_lambda_sq_l236_236458


namespace find_conjugate_l236_236260

noncomputable def conjugate (z : ℂ) := conj z

theorem find_conjugate (z : ℂ) (h : (z - 1) * complex.I = 1 + complex.I) : conjugate z = 2 + complex.I :=
by
  sorry

end find_conjugate_l236_236260


namespace probability_0_to_1_l236_236240

variable {μ δ : ℝ}
variable (ξ : ℝ → ℝ) -- representing ξ as a function which we'd interpret as the random variable

noncomputable def normal_distribution (x : ℝ) := (1 / (δ * sqrt (2 * π))) * exp (- ((x - μ)^2) / (2 * δ^2))

theorem probability_0_to_1 :
  (∃ μ δ, ∀ x, ξ x = normal_distribution x) → (P(ξ < 1) = 0.5) → (P(ξ > 2) = 0.4) → P (0 < ξ < 1) = 0.1 :=
by
sorry

end probability_0_to_1_l236_236240


namespace fraction_of_area_below_line_l236_236819

noncomputable def rectangle_area_fraction (x1 y1 x2 y2 : ℝ) (x3 y3 x4 y4 : ℝ) : ℝ :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  let y_intercept := b
  let base := x4 - x1
  let height := y4 - y3
  let triangle_area := 0.5 * base * height
  triangle_area / (base * height)

theorem fraction_of_area_below_line : 
  rectangle_area_fraction 1 3 5 1 1 0 5 4 = 1 / 8 := 
by
  sorry

end fraction_of_area_below_line_l236_236819


namespace target_hit_by_A_given_target_hit_l236_236779

-- Definitions of hit rates
def hit_rate_A : ℝ := 0.6
def hit_rate_B : ℝ := 0.5

-- Probability that the target is hit
def P_C : ℝ := 1 - (1 - hit_rate_A) * (1 - hit_rate_B)

-- Conditional probability: given the target is hit, it was hit by person A
def conditional_prob : ℝ := hit_rate_A / P_C

theorem target_hit_by_A_given_target_hit : conditional_prob = 0.75 := 
by
  unfold hit_rate_A hit_rate_B P_C conditional_prob
  -- Mathematical calculations
  sorry

end target_hit_by_A_given_target_hit_l236_236779


namespace num_valid_x_values_l236_236289

noncomputable def count_valid_x : ℕ :=
  ((Finset.range 34).filter (λ x, x ≥ 25 ∧ 3 * x < 100 ∧ 4 * x > 99)).card

theorem num_valid_x_values : count_valid_x = 9 := by
  sorry

end num_valid_x_values_l236_236289


namespace coin_loading_impossible_l236_236969

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l236_236969


namespace h2o_formation_l236_236198

theorem h2o_formation (n_HCl : ℕ) (n_CaCO3 : ℕ) (eqn : "CaCO3 + 2HCl → CaCl2 + CO2 + H2O") (h_HCl : n_HCl = 2) (h_CaCO3 : n_CaCO3 = 1) : 
  ∃ n_H2O : ℕ, n_H2O = 1 :=
by
  sorry

end h2o_formation_l236_236198


namespace four_digit_numbers_with_property_l236_236305

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l236_236305


namespace largest_red_points_l236_236722

noncomputable def maximum_red_points {n : ℕ} (points : Finset (ℝ × ℝ))
  (h_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬ collinear p1 p2 p3)
  (coloring : points → Finset (ℝ × ℝ) × Finset (ℝ × ℝ))
  (h_coloring : ∀ t : Finset (ℝ × ℝ), t ⊆ points → t.card = 3 → 
    (∀ p ∈ t, p ∈ (coloring points).1) → ∃ b ∈ (coloring points).2, b ∈ convex_hull (t : Set (ℝ × ℝ))) : ℕ :=
1012

theorem largest_red_points {n : ℕ} (points : Finset (ℝ × ℝ)) :
  n = 2022 →
  (∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬ collinear p1 p2 p3) →
  ∃ coloring : points → Finset (ℝ × ℝ) × Finset (ℝ × ℝ), 
    (∀ t : Finset (ℝ × ℝ), t ⊆ points → t.card = 3 → 
      (∀ p ∈ t, p ∈ (coloring points).1) → ∃ b ∈ (coloring points).2, b ∈ convex_hull (t : Set (ℝ × ℝ))) →
  maximum_red_points points _ coloring _ = 1012 :=
by
  intros h_n h_collinear
  classical
  use (λ pts, (pt : ℝ × ℝ)
    let reds := {p ∈ pts | cond1 p} -- Example condition for red points
    let blues := pts \ reds
    (reds, blues)
  sorry

end largest_red_points_l236_236722


namespace complex_division_derivative_value_l236_236218

-- Problem (1)
theorem complex_division : (2 - I) / (3 + 4 * I) = (2 / 25) - (11 / 25) * I := by
  sorry

-- Problem (2)
noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * (f 2).derivative

theorem derivative_value : (1 + (derivative f 1)) = -3 := by
  sorry

end complex_division_derivative_value_l236_236218


namespace remainder_division_l236_236537

theorem remainder_division (N : ℤ) (R1 : ℤ) (Q2 : ℤ) 
  (h1 : N = 44 * 432 + R1)
  (h2 : N = 38 * Q2 + 8) : 
  R1 = 0 := by
  sorry

end remainder_division_l236_236537


namespace students_remaining_after_third_stop_l236_236697

theorem students_remaining_after_third_stop
  (initial_students : ℕ)
  (third : ℚ) (stops : ℕ)
  (one_third_off : third = 1 / 3)
  (initial_students_eq : initial_students = 64)
  (stops_eq : stops = 3)
  : 64 * ((2 / 3) ^ 3) = 512 / 27 :=
by 
  sorry

end students_remaining_after_third_stop_l236_236697


namespace visible_cubes_from_corner_l236_236120

theorem visible_cubes_from_corner : 
  ∀ (n : ℕ), 
  n = 12 → 
  let face_cubes := n * n, 
      total_cubes := 3 * face_cubes, 
      shared_edges := 3 * (n - 1), 
      corner_cube := 1 
  in total_cubes - shared_edges + corner_cube = 400 := 
by
  intros n hn
  let face_cubes := n * n
  let total_cubes := 3 * face_cubes
  let shared_edges := 3 * (n - 1)
  let corner_cube := 1
  calc
    total_cubes - shared_edges + corner_cube
        = 3 * (n * n) - 3 * (n - 1) + 1 : by rw [face_cubes, total_cubes, shared_edges, corner_cube]
    ... = 400 : by sorry

end visible_cubes_from_corner_l236_236120


namespace probability_of_different_colors_l236_236060

noncomputable def num_blue := 7
noncomputable def num_yellow := 4
noncomputable def num_green := 5
noncomputable def total_chips := num_blue + num_yellow + num_green

theorem probability_of_different_colors :
  let p_blue := (num_blue : ℚ) / total_chips,
      p_yellow := (num_yellow : ℚ) / total_chips,
      p_green := (num_green : ℚ) / total_chips,
      p_diff_colors := 
        (p_blue * (num_yellow : ℚ) / total_chips) +
        (p_blue * (num_green : ℚ) / total_chips) +
        (p_yellow * (num_blue : ℚ) / total_chips) +
        (p_yellow * (num_green : ℚ) / total_chips) +
        (p_green * (num_blue : ℚ) / total_chips) +
        (p_green * (num_yellow : ℚ) / total_chips)
  in p_diff_colors = 83 / 128 := sorry

end probability_of_different_colors_l236_236060


namespace sqrt_E_minus_F_l236_236590

noncomputable def A (k : ℕ) : ℕ := 10^k - 1 / 9

theorem sqrt_E_minus_F (k : ℕ) (E : ℕ) (F : ℕ) (hE : E = A k * 10^k + A k) (hF : F = 2 * A k) :
  nat.sqrt (E - F) = 3 * A k :=
by
  sorry

end sqrt_E_minus_F_l236_236590


namespace ball_distribution_count_l236_236246

theorem ball_distribution_count :
  ∃ (n : ℕ), n = 180 ∧ -- We need to prove n is 180
  (∀ (balls : fin 5) (boxes : fin 3),
    (∀ (distribution : Π (b : fin 4), boxes), 
      ∃ (f : fin 3 → fin 5 → ℕ), 
        (∀ b : fin 3, f b ≠ 0) →
        (∑ b : fin 3, f b = 4) →
        (so that different ways distribution equals n))) := sorry

end ball_distribution_count_l236_236246


namespace greatest_possible_remainder_when_dividing_by_7_l236_236279

theorem greatest_possible_remainder_when_dividing_by_7 :
  ∀ (x : ℕ), ∃ r : ℕ, r < 7 ∧ x % 7 = r ∧ r = 6 :=
by
  intro x
  use 6
  split
  . exact lt_of_le_of_lt (Nat.zero_le 6) (by norm_num)
  split
  . sorry -- proof that x % 7 can equal 6
  . refl

end greatest_possible_remainder_when_dividing_by_7_l236_236279


namespace divide_400_l236_236461

theorem divide_400 (a b c d : ℕ) (h1 : a + b + c + d = 400) 
  (h2 : a + 1 = b - 2) (h3 : a + 1 = 3 * c) (h4 : a + 1 = d / 4) 
  : a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 :=
sorry

end divide_400_l236_236461


namespace impossible_to_load_two_coins_l236_236972

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l236_236972


namespace problem_statement_l236_236006

def f (x p q : ℝ) := (x + p) * (x + q) + 2

theorem problem_statement
  (p q : ℝ)
  (hp : 2^p + p + 2 = 0)
  (hq : log 2 q + q + 2 = 0) :
  f 2 p q = f 0 p q ∧ f 0 p q < f 3 p q :=
by
  sorry -- proof not required

end problem_statement_l236_236006


namespace time_to_fill_remaining_cistern_l236_236074

-- Define the conditions
def pipe_p_fill_time : ℝ := 12
def pipe_q_fill_time : ℝ := 15
def both_opened_time : ℝ := 2

-- Calculate the fill rates
def fill_rate_pipe_p : ℝ := 1 / pipe_p_fill_time
def fill_rate_pipe_q : ℝ := 1 / pipe_q_fill_time

def combined_fill_rate : ℝ := fill_rate_pipe_p + fill_rate_pipe_q
def fill_in_both_opened : ℝ := both_opened_time * combined_fill_rate

-- Calculate the remaining part of the cistern to be filled
def remaining_cistern : ℝ := 1 - fill_in_both_opened

-- Define the theorem
theorem time_to_fill_remaining_cistern : remaining_cistern / fill_rate_pipe_q = 10.5 := by
  sorry

end time_to_fill_remaining_cistern_l236_236074


namespace difference_of_squares_example_l236_236564

theorem difference_of_squares_example: (635^2 - 365^2 = 270000) :=
by
  let a := 635
  let b := 365
  have h1: a = 635 := rfl
  have h2: b = 365 := rfl
  calc
    a^2 - b^2 
        = (a + b) * (a - b) : by rw [←sub_eq_add_neg, ←add_mul, ←mul_sub]
    ... = 1000 * 270 : by rw [show a + b = 1000, from rfl, show a - b = 270, from rfl]
    ... = 270000 : by norm_num

end difference_of_squares_example_l236_236564


namespace initial_mixture_l236_236515

theorem initial_mixture (M : ℝ) (h1 : 0.20 * M + 20 = 0.36 * (M + 20)) : 
  M = 80 :=
by
  sorry

end initial_mixture_l236_236515


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236933

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236933


namespace infinite_n_prime_divisor_l236_236781

theorem infinite_n_prime_divisor (h1 : ∀ n : ℕ, ∃ p : ℕ, p ∣ (n^2 + 1) ∧ Prime p) :
  ∃∞ n : ℕ, ∃ p : ℕ, p ∣ (n^2 + 1) ∧ Prime p ∧ p > 2 * n + Real.sqrt (2 * n) :=
sorry 

end infinite_n_prime_divisor_l236_236781


namespace find_f_for_negative_x_find_g_of_a_l236_236249

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x >= 0 then (if x <= 3 then x * (3 - x) else (x - 3) * (a - x))
  else if -3 <= x then -x * (x + 3) else -(x + 3) * (a + x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem find_f_for_negative_x (a : ℝ) :
  is_even_function (λ x, f x a) →
  ∀ x, x < 0 →
  f x a =
    if -3 <= x then -x * (x + 3)
    else -(x + 3) * (a + x) :=
sorry

theorem find_g_of_a (a : ℝ) :
  is_even_function (λ x, f x a) →
  ∃ g : ℝ → ℝ,
    (a ≤ 6 → g a = 9 / 4) ∧
    (6 < a ∧ a ≤ 7 → g a = (a - 3) ^ 2 / 4) ∧
    (a > 7 → g a = 2 * (a - 5)) :=
sorry

end find_f_for_negative_x_find_g_of_a_l236_236249


namespace simplify_expression_l236_236789

theorem simplify_expression :
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) := by
  sorry

end simplify_expression_l236_236789


namespace find_f_expression_find_area_l236_236166

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c

def has_equal_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = 0 ↔ x = α

variables {f : ℝ → ℝ}

theorem find_f_expression
  (hf1 : quadratic_function f)
  (hf2 : has_equal_real_roots f)
  (hf3 : ∀ x, deriv f x = 2 * x + 2) :
  f = λ x, x^2 + 2 * x + 1 := sorry

theorem find_area
  (hf1 : quadratic_function f)
  (hf2 : ∀ x, deriv f x = 2 * x + 2)
  (hf3 : f = λ x, x^2 + 2 * x + 1) :
  ∫ x in -1..0, f x = 1 / 3 := sorry

end find_f_expression_find_area_l236_236166


namespace sqrt_3x_eq_5x_largest_value_l236_236879

theorem sqrt_3x_eq_5x_largest_value (x : ℝ) (h : Real.sqrt (3 * x) = 5 * x) : x ≤ 3 / 25 := 
by
  sorry

end sqrt_3x_eq_5x_largest_value_l236_236879


namespace shape_from_intersection_l236_236180

def point := (ℝ × ℝ)
def rectangle (A B C D : point) := (A = (0, 0)) ∧ (B = (0, 5)) ∧ (C = (8, 5)) ∧ (D = (8, 0))

def line_from_point_at_angle (p : point) (θ : ℝ) : set (point) :=
  { q | ∃ t : ℝ, q = (p.1 + t * math.cos θ, p.2 + t * math.sin θ) }

theorem shape_from_intersection :
  ∀ (A B C D : point),
    rectangle A B C D →
    ∃ (P1 P2 : point),
      P1 ∈ line_from_point_at_angle A (π/4) ∧ P1 ∈ line_from_point_at_angle B (-π/4) ∧
      P2 ∈ line_from_point_at_angle A (5*π/12) ∧ P2 ∈ line_from_point_at_angle B (-5*π/12) ∧
      (P1.2 = P2.2) ∧ (P1.2 = (A.2 + B.2) / 2) :=
by
  -- Proof structure goes here
  sorry

end shape_from_intersection_l236_236180


namespace inequality_proof_l236_236762

variables {n : ℕ} (h₁ : n ≥ 3) {x : ℕ → ℝ} (h₂ : ∀ i j, i < j → x i < x j)

theorem inequality_proof :
  (n * (n - 1) / 2) * ∑ (i : ℕ) in finset.range(n - 1), ∑ (j : ℕ) in finset.range(i + 1, n), (x i) * (x j)
  > ∑ (i : ℕ) in finset.range(n - 1), (n - i) * (x i) * ∑ (j : ℕ) in finset.range(2, n + 1), (j - 1) * (x (j-1)) :=
sorry

end inequality_proof_l236_236762


namespace find_theta_l236_236230

theorem find_theta (θ : Real) (h : abs θ < π / 2) (h_eq : Real.sin (π + θ) = -Real.sqrt 3 * Real.cos (2 * π - θ)) :
  θ = π / 3 :=
sorry

end find_theta_l236_236230


namespace tank_capacity_l236_236815

theorem tank_capacity (x : ℝ) (h : 0.50 * x = 75) : x = 150 :=
by sorry

end tank_capacity_l236_236815


namespace movement_in_space_is_four_symmetries_l236_236505

theorem movement_in_space_is_four_symmetries :
  ∀ T : (EuclideanSpace ℝ 3 → EuclideanSpace ℝ 3), 
    ∃ S1 S2 S3 S4 : (EuclideanSpace ℝ 3 → EuclideanSpace ℝ 3), 
      is_reflection_plane S1 ∧ is_reflection_plane S2 ∧ is_reflection_plane S3 ∧ is_reflection_plane S4 ∧
      (T = S1 ∘ S2 ∘ S3 ∘ S4) :=
sorry

end movement_in_space_is_four_symmetries_l236_236505


namespace combined_cost_is_3490_l236_236228

-- Definitions for the quantities of gold each person has and their respective prices per gram
def Gary_gold_grams : ℕ := 30
def Gary_gold_price_per_gram : ℕ := 15

def Anna_gold_grams : ℕ := 50
def Anna_gold_price_per_gram : ℕ := 20

def Lisa_gold_grams : ℕ := 40
def Lisa_gold_price_per_gram : ℕ := 18

def John_gold_grams : ℕ := 60
def John_gold_price_per_gram : ℕ := 22

-- Combined cost
def combined_cost : ℕ :=
  Gary_gold_grams * Gary_gold_price_per_gram +
  Anna_gold_grams * Anna_gold_price_per_gram +
  Lisa_gold_grams * Lisa_gold_price_per_gram +
  John_gold_grams * John_gold_price_per_gram

-- Proof that the combined cost is equal to $3490
theorem combined_cost_is_3490 : combined_cost = 3490 :=
  by
  -- proof skipped
  sorry

end combined_cost_is_3490_l236_236228


namespace ln_f_x_gt_1_max_value_a_l236_236396

-- (I) Prove that for the function \( f(x) = |x-2| + |x+1| \), the inequality ln(f(x)) > 1 holds for any \( x \in \mathbb{R} \).
theorem ln_f_x_gt_1 (x : ℝ) : 
  let f : ℝ → ℝ := fun x => abs (x - 2) + abs (x + 1)
  in ln (f x) > 1 := by sorry

-- (II) Find the maximum value of \( a \) such that the inequality \( f(x) = |x-2| + |x-a| \geqslant a \) holds for any \( x \in \mathbb{R} \) and prove that the maximum value is \( 1 \).
theorem max_value_a : ∃ a : ℝ, (∀  (x : ℝ) , (|x - 2| + |x - a|) ≥ a) ∧ a = 1 := by sorry

end ln_f_x_gt_1_max_value_a_l236_236396


namespace unique_sup_and_inf_l236_236423

open Set

-- Define the problem in Lean 4
theorem unique_sup_and_inf {A : Set ℝ} (h1 : A.nonempty) :
  (BoundedAbove A → ∃! s, IsSup A s) ∧ (BoundedBelow A → ∃! i, IsInf A i) :=
by
  sorry

end unique_sup_and_inf_l236_236423


namespace correct_statements_are_2_l236_236163

def stmt1 := "The prism with the least number of faces has 6 vertices."
def stmt2 := "A frustum is the middle part of a cone cut by two parallel planes."
def stmt3 := "A plane passing through the vertex of a cone cuts the cone into a section that is an isosceles triangle."
def stmt4 := "Equal angles remain equal in perspective drawings."

noncomputable def statement := 4
noncomputable def correct_statements := 2

theorem correct_statements_are_2 :
  (stmt1_correct : true) ∧ 
  (stmt2_correct : false) ∧ 
  (stmt3_correct : true) ∧ 
  (stmt4_correct : false) ∧ 
  (statement_count = correct_statements) :=
  sorry

end correct_statements_are_2_l236_236163


namespace area_triangle_ABC_l236_236422

def point := (ℝ × ℝ)

noncomputable def area_of_triangle (A B C : point) : ℝ :=
  0.5 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

theorem area_triangle_ABC :
  let A := (0, 0) : point
  let B := (1, 3) : point
  let C := (4, 1) : point
  area_of_triangle A B C = 11 / 2 := 
by
  let A := (0, 0) : point
  let B := (1, 3) : point
  let C := (4, 1) : point
  exact sorry

end area_triangle_ABC_l236_236422


namespace impossible_to_load_two_coins_l236_236974

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l236_236974


namespace partnership_investment_l236_236108

theorem partnership_investment
  (a_investment : ℕ := 30000)
  (b_investment : ℕ)
  (c_investment : ℕ := 50000)
  (c_profit_share : ℕ := 36000)
  (total_profit : ℕ := 90000)
  (total_investment := a_investment + b_investment + c_investment)
  (c_defined_share : ℚ := 2/5)
  (profit_proportionality : (c_profit_share : ℚ) / total_profit = (c_investment : ℚ) / total_investment) :
  b_investment = 45000 :=
by
  sorry

end partnership_investment_l236_236108


namespace four_digit_num_condition_l236_236299

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l236_236299


namespace circle_radius_l236_236528

def hexagon_side_length := 3
def visibility_probability := 1 / 4
def correct_radius := 4 * Real.sqrt 3

theorem circle_radius (r : ℝ) :
  (∃ (hexagon_side_length : ℝ) (visibility_probability : ℝ),
    hexagon_side_length = 3 ∧ visibility_probability = 1 / 4) →
  r = correct_radius :=
by {
  sorry
}

end circle_radius_l236_236528


namespace triangle_inequality_sqrt_l236_236008

theorem triangle_inequality_sqrt {a b c : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (habc1: a + b > c) (habc2: a + c > b) (habc3: b + c > a) :
  let p := (a + b + c) / 2 in
  sqrt (p - a) + sqrt (p - b) + sqrt (p - c) ≤ sqrt (3 * p) :=
by
  sorry

end triangle_inequality_sqrt_l236_236008


namespace how_many_years_younger_l236_236786

-- Define conditions
def age_ratio (sandy_age moll_age : ℕ) := sandy_age * 9 = moll_age * 7
def sandy_age := 70

-- Define the theorem to prove
theorem how_many_years_younger 
  (molly_age : ℕ) 
  (h1 : age_ratio sandy_age molly_age) 
  (h2 : sandy_age = 70) : molly_age - sandy_age = 20 := 
sorry

end how_many_years_younger_l236_236786


namespace sum_of_possible_m_l236_236755

theorem sum_of_possible_m (x y z m : ℝ) (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z)
  (h4 : x / (2 - y) = m) (h5 : y / (2 - z) = m) (h6 : z / (2 - x) = m) :
  m = 1 ∨ m = 2 :=
begin
  sorry
end

end sum_of_possible_m_l236_236755


namespace probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236935

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end probability_of_not_all_8_sided_dice_rolling_the_same_number_l236_236935


namespace cartesian_eq_C1_cartesian_eq_C2_range_distance_MN_l236_236357

open Real

namespace CartesianCoordinates

-- Conditions for the problem
def parametric_C1 (φ : ℝ) : ℝ × ℝ := (2 * cos φ, sin φ)
def polar_center_C2 : ℝ × ℝ := (0, 3)

-- Cartesian equation of curve C1
theorem cartesian_eq_C1 (x y : ℝ) (h : ∃ φ : ℝ, x = 2 * cos φ ∧ y = sin φ) :
  x ^ 2 / 4 + y ^ 2 = 1 := sorry

-- Cartesian equation of curve C2
theorem cartesian_eq_C2 (x y : ℝ) :
  (x - polar_center_C2.1) ^ 2 + (y - polar_center_C2.2) ^ 2 = 1 ↔ x ^ 2 + (y - 3) ^ 2 = 1 := sorry

-- Range of values for the distance |MN|
theorem range_distance_MN (φ θ : ℝ) :
  let M := parametric_C1 φ;
  let N := (polar_center_C2.1 + cos θ, polar_center_C2.2 + sin θ);
  let distance := sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2) in
  1 ≤ distance ∧ distance ≤ 5 := sorry

end CartesianCoordinates

end cartesian_eq_C1_cartesian_eq_C2_range_distance_MN_l236_236357


namespace probability_not_all_same_l236_236911

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l236_236911


namespace basketball_game_half_points_l236_236589

noncomputable def eagles_geometric_sequence (a r : ℕ) (n : ℕ) : ℕ :=
  a * r ^ n

noncomputable def lions_arithmetic_sequence (b d : ℕ) (n : ℕ) : ℕ :=
  b + n * d

noncomputable def total_first_half_points (a r b d : ℕ) : ℕ :=
  eagles_geometric_sequence a r 0 + eagles_geometric_sequence a r 1 +
  lions_arithmetic_sequence b d 0 + lions_arithmetic_sequence b d 1

theorem basketball_game_half_points (a r b d : ℕ) (h1 : a + a * r = b + (b + d)) (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2*d) + (b + 3*d)) :
  total_first_half_points a r b d = 8 :=
by sorry

end basketball_game_half_points_l236_236589


namespace number_of_ways_to_assign_6_grades_l236_236401

def valid_grades (grades : List ℕ) : Prop :=
  grades.all (λ g, g = 2 ∨ g = 3 ∨ g = 4) ∧
  ¬ grades.contains_adjacent (λ g => g = 2)

def a (n : ℕ) : ℕ
| 0         := 1
| 1         := 3
| 2         := 8
| (n + 1) := 2 * a n + 2 * a (n - 1)

theorem number_of_ways_to_assign_6_grades : 
  a 6 = 448 :=
by
  sorry

end number_of_ways_to_assign_6_grades_l236_236401


namespace sum_of_numbers_is_twenty_l236_236952

-- Given conditions
variables {a b c : ℝ}

-- Prove that the sum of a, b, and c is 20 given the conditions
theorem sum_of_numbers_is_twenty (h1 : a^2 + b^2 + c^2 = 138) (h2 : ab + bc + ca = 131) :
  a + b + c = 20 :=
by
  sorry

end sum_of_numbers_is_twenty_l236_236952


namespace correct_student_result_l236_236498

theorem correct_student_result 
  (α β : ℝ) (hα : 0 < α ∧ α < 90) (hβ : 90 < β ∧ β < 180) 
  (student_result : ℝ) 
  (h_result : student_result ∈ {17, 42, 56, 73}) 
  (h_correct : student_result = (1/5) * (α + β)) : 
  student_result = 42 :=
by
  have h_range : 90 < α + β ∧ α + β < 270 := ⟨
    by linarith [hα.left, hβ.left],
    by linarith [hα.right, hβ.right]
  ⟩
  have h_fifth_range : 18 < (1/5) * (α + β) ∧ (1/5) * (α + β) < 54 := ⟨
    by linarith [h_range.left],
    by linarith [h_range.right]
  ⟩
  have h_valid : 18 < student_result ∧ student_result < 54 :=
    by simp [h_correct, h_fifth_range]
  rcases h_valid with ⟨h1, h2⟩
  interval_cases student_result
  case inl => linarith
  case inr _ _ =>
    repeat { case inl { linarith }, case inr _ _ {} }
  case inr => linarith
  case out => rfl -- proving 42° is within the bounds and correct

end correct_student_result_l236_236498


namespace probability_not_all_same_l236_236929

theorem probability_not_all_same (n : ℕ) (h : n = 5) : 
  let total_outcomes := 8^n in
  let same_number_outcomes := 8 in
  let prob_all_same_number := (same_number_outcomes : ℚ) / total_outcomes in
  let prob_not_all_same_number := 1 - prob_all_same_number in
  prob_not_all_same_number = 4095 / 4096 :=
by 
  sorry

end probability_not_all_same_l236_236929


namespace find_x_y_z_l236_236278

variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables {a b c : V}
variables (x y z : ℝ)
variables (O A B C D M N P : V)

-- Definitions of the conditions
def OA := a
def OB := b
def OC := c
def OP := x • a + y • b + z • c
def M := 1/3 • (2 • A + D + C)
def N := 1/3 • (B + D + C)

-- Relationship in the problem
def MP := M - P
def PN := P - N

-- The proof goal
theorem find_x_y_z (h1 : OA = a) (h2 : OB = b) (h3 : OC = c)
    (h4 : OP = x • a + y • b + z • c)
    (h5 : M = 1/3 • (2 • A + D + C))
    (h6 : N = 1/3 • (B + D + C))
    (h7 : MP = 2 • PN) :
    x = -2/9 ∧ y = 4/9 ∧ z = 5/9 :=
by { sorry }

end find_x_y_z_l236_236278


namespace angle_B_equals_pi_div_3_right_triangle_when_lambda_sqrt3_obtuse_triangle_lambda_range_l236_236733

-- Definition of triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables (A B C a b c : Real) (λ : Real)

-- Conditions of the problem
axiom h1 : (2 * a - c) * cos B = b * cos C
axiom h2 : sin A ^ 2 = sin B ^ 2 + sin C ^ 2 - λ * sin B * sin C
axiom h3 : 0 < A ∧ A < π
axiom h4 : 0 < B ∧ B < π
axiom h5 : 0 < C ∧ C < π
axiom h6 : A + B + C = π

-- 1. Prove that B = π / 3
theorem angle_B_equals_pi_div_3 : B = π / 3 := sorry

-- 2. Prove triangle ABC is a right triangle for λ = √3
theorem right_triangle_when_lambda_sqrt3 (h : λ = Real.sqrt 3) : C = π / 2 := sorry

-- 3. Prove range of λ for obtuse triangle ABC
theorem obtuse_triangle_lambda_range : (B > π / 2 ∨ C > π / 2) → (-1 < λ ∧ λ < 0) ∨ (Real.sqrt 3 < λ ∧ λ < 2) := sorry

end angle_B_equals_pi_div_3_right_triangle_when_lambda_sqrt3_obtuse_triangle_lambda_range_l236_236733


namespace locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l236_236131

/-- Given a circle with center at point P passes through point A (1,0) 
    and is tangent to the line x = -1, the locus of point P is the parabola C. -/
theorem locus_of_P_is_parabola (P A : ℝ × ℝ) (x y : ℝ):
  (A = (1, 0)) → (P.1 + 1)^2 + P.2^2 = 0 → y^2 = 4 * x := 
sorry

/-- If the line passing through point H(4, 0) intersects the parabola 
    C (denoted by y^2 = 4x) at points M and N, and T is any point on 
    the line x = -4, then the slopes of lines TM, TH, and TN form an 
    arithmetic sequence. -/
theorem slopes_form_arithmetic_sequence (H M N T : ℝ × ℝ) (m n k : ℝ): 
  (H = (4, 0)) → (T.1 = -4) → 
  (M.1, M.2) = (k^2, 4*k) ∧ (N.1, N.2) = (m^2, 4*m) → 
  ((T.2 - M.2) / (T.1 - M.1) + (T.2 - N.2) / (T.1 - N.1)) = 
  2 * (T.2 / -8) := 
sorry

end locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l236_236131


namespace max_sum_squares_l236_236005

theorem max_sum_squares (a : Fin 100 → ℝ)
  (h1 : ∀ i j : Fin 100, i ≤ j → a i ≥ a j)
  (h2 : a 0 + a 1 ≤ 100)
  (h3 : ∑ i in Finset.filter (λ i, 2 ≤ i) Finset.univ, a i ≤ 100) :
  (∑ i, (a i)^2) ≤ 10000 :=
sorry

end max_sum_squares_l236_236005


namespace coin_loading_impossible_l236_236980

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l236_236980


namespace g_1_5_l236_236038

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_defined (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g x ≠ 0

axiom g_zero : g 0 = 0

axiom g_mono (x y : ℝ) (hx : 0 ≤ x ∧ x < y ∧ y ≤ 1) : g x ≤ g y

axiom g_symmetry (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (1 - x) = 1 - g x

axiom g_scaling (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (x/4) = g x / 2

theorem g_1_5 : g (1 / 5) = 1 / 4 := 
sorry

end g_1_5_l236_236038


namespace lcm_28_72_l236_236604

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l236_236604


namespace max_min_f_f_below_g_inequality_f_prime_l236_236262

-- f(x) = x^2 + ln(x) - 1
def f (x : ℝ) := x^2 + Real.log x - 1
-- g(x) = x^3
def g (x : ℝ) := x^3

-- 1. Prove the maximum and minimum values in the interval [1, e].
theorem max_min_f :
  max (f 1) (f Real.exp) = f Real.exp ∧ min (f 1) (f Real.exp) = f 1 :=
by
  sorry

-- 2. Prove that f(x) < g(x) for x in (1, ∞).
theorem f_below_g (x : ℝ) (hx : x > 1) : f x < g x :=
by
  sorry

-- 3. Prove that [f'(x)]^n - f'(x^n) ≥ 2^n - 2 for n ∈ ℕ+.
theorem inequality_f_prime (x : ℝ) (hx : x > 0) (n : ℕ) (hn : n ≥ 1) :
  (2 * x + 1 / x) ^ n - (2 * x ^ n + 1 / (x ^ n)) ≥ 2 ^ n - 2 :=
by
  sorry

end max_min_f_f_below_g_inequality_f_prime_l236_236262


namespace impossible_to_load_two_coins_l236_236971

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l236_236971


namespace plywood_cut_perimeter_difference_l236_236517

theorem plywood_cut_perimeter_difference :
  ∃ (rectangles : list (ℝ × ℝ)), 
    length rectangles = 5 ∧
    (∀ r ∈ rectangles, r.1 * r.2 = 10) ∧
    let perimeters := rectangles.map (λ r, 2 * r.1 + 2 * r.2) in
    |(max (perimeters) - min (perimeters))| = 8 :=
sorry

end plywood_cut_perimeter_difference_l236_236517


namespace sine_of_central_angle_subtended_by_minor_arc_AB_is_120_over_169_l236_236346

theorem sine_of_central_angle_subtended_by_minor_arc_AB_is_120_over_169 :
  ∀ (O A B C D M N : Type)
    [h1 : ∀(O A B C D : Type), AD.has_radius 13]
    [h2 : ∀(O A B C D : Type), chord.intersect BC AD]
    [h3 : ∀(O A B C D : Type), chord.bisect BC AD]
    [h4 : BC.length = 10]
    [h5 : ∀ {O A B C D : Type}, (∀ {P : Type}, chord.intersect P AD → P = BC)]
  , (sin (angle (∠AOB)) = 120 / 169) := sorry

end sine_of_central_angle_subtended_by_minor_arc_AB_is_120_over_169_l236_236346


namespace select_student_B_l236_236046

-- Define the average scores for the students A, B, C, D
def avg_A : ℝ := 85
def avg_B : ℝ := 90
def avg_C : ℝ := 90
def avg_D : ℝ := 85

-- Define the variances for the students A, B, C, D
def var_A : ℝ := 50
def var_B : ℝ := 42
def var_C : ℝ := 50
def var_D : ℝ := 42

-- Theorem stating the selected student should be B
theorem select_student_B (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ)
  (h_avg_A : avg_A = 85) (h_avg_B : avg_B = 90) (h_avg_C : avg_C = 90) (h_avg_D : avg_D = 85)
  (h_var_A : var_A = 50) (h_var_B : var_B = 42) (h_var_C : var_C = 50) (h_var_D : var_D = 42) :
  (avg_B = 90 ∧ avg_C = 90 ∧ avg_B ≥ avg_A ∧ avg_B ≥ avg_D ∧ var_B < var_C) → 
  (select_student = "B") :=
by
  sorry

end select_student_B_l236_236046


namespace induction_problem_l236_236078

theorem induction_problem (n : ℕ) (h : n ≥ 2) : 
  ∏ i in finset.range (n + 1), (n + i) = 2^n * ∏ i in finset.range n, (2 * i + 1) :=
by sorry

end induction_problem_l236_236078


namespace possible_sets_B_l236_236676

def A : Set ℤ := {-1}

def isB (B : Set ℤ) : Prop :=
  A ∪ B = {-1, 3}

theorem possible_sets_B : ∀ B : Set ℤ, isB B → B = {3} ∨ B = {-1, 3} :=
by
  intros B hB
  sorry

end possible_sets_B_l236_236676


namespace probability_of_not_all_same_number_l236_236905

theorem probability_of_not_all_same_number :
  ∀ (D : ℕ) (n : ℕ),
  D = 8 → n = 5 → 
  let total_outcomes := D^n in
  let same_outcome_probability := D / total_outcomes  in
  let not_all_same_probability := 1 - same_outcome_probability in
  (not_all_same_probability = 4095 / 4096) :=
begin
  intros D n D_eq n_eq,
  simp only [D_eq, n_eq],
  let total_outcomes := 8^5,
  let same_outcome_probability := 8 / total_outcomes,
  let not_all_same_probability := 1 - same_outcome_probability,
  have total_outcome_calc : total_outcomes = 8^5 := by sorry,
  have same_outcome_prob_calc : same_outcome_probability = 1 / 4096 := by sorry,
  have not_all_same_prob_calc : not_all_same_probability = 4095 / 4096 := by sorry,
  exact not_all_same_prob_calc,
end

end probability_of_not_all_same_number_l236_236905


namespace curve_properties_l236_236494

noncomputable def curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

theorem curve_properties :
  curve 1 0 ∧ curve 0 1 ∧ curve (1/4) (1/4) ∧ 
  (∀ p : ℝ × ℝ, curve p.1 p.2 → curve p.2 p.1) :=
by
  sorry

end curve_properties_l236_236494
