import Mathlib
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.FieldPower
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticEquations
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.Calculus.Parametric
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics.Basic
import Mathlib.Data.Nat.Pow
import Mathlib.Data.Pi.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Set.Intervals
import Mathlib.Data.Set.Intervals.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ConditionalProbability
import Mathlib.Tactic
import Mathlib.Tactic.SolveByElim

namespace sequence_geometric_l603_603300

theorem sequence_geometric (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 = 1)
  (h_geom : ∀ k : ℕ, a (k + 1) - a k = (1 / 3) ^ k) :
  a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by
  sorry

end sequence_geometric_l603_603300


namespace line_positional_relationship_l603_603215

variables {Point Line Plane : Type}

-- Definitions of the conditions
def is_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
def is_within_plane (b : Line) (α : Plane) : Prop := sorry
def no_common_point (a b : Line) : Prop := sorry
def parallel_or_skew (a b : Line) : Prop := sorry

-- Proof statement in Lean
theorem line_positional_relationship
  (a b : Line) (α : Plane)
  (h₁ : is_parallel_to_plane a α)
  (h₂ : is_within_plane b α)
  (h₃ : no_common_point a b) :
  parallel_or_skew a b :=
sorry

end line_positional_relationship_l603_603215


namespace tangent_line_perpendicular_eq_l603_603818

-- Definitions based on the conditions from part a)
def is_tangent_line (f : ℝ → ℝ) (p : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
∀ x : ℝ, l x = (deriv f p.1) * (x - p.1) + p.2

def perpendicular_lines (m1 m2 : ℝ) : Prop :=
m1 * m2 = -1

-- Main statement
theorem tangent_line_perpendicular_eq :
  let l := λ (x : ℝ), 4 * x - 3
  let curve := λ (x : ℝ), x ^ 4
  let lin_eq := λ (x : ℝ), - (1 / 4) * x + 2 in
  is_tangent_line curve (1, 1) l ∧ perpendicular_lines (4) (- (1 / 4)) →
  ∃ m : ℝ, l = λ (x : ℝ), m * x - 3 :=
by sorry

end tangent_line_perpendicular_eq_l603_603818


namespace partition_two_connected_graph_l603_603744

open SimpleGraph

variables {V : Type} [Fintype V]

/-- Represents that a graph is 2-connected -/
def is_two_connected (G : SimpleGraph V) : Prop :=
∀ (v : V), G.delete_vert v.connected

theorem partition_two_connected_graph (G : SimpleGraph (Fin 100)) 
  (hG : is_two_connected G) :
  ∃ (G1 G2 : SimpleGraph (Fin 50)), G1.connected ∧ G2.connected :=
by
  sorry

end partition_two_connected_graph_l603_603744


namespace find_x_l603_603258

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l603_603258


namespace minimum_positive_period_of_f_range_of_f_for_given_interval_l603_603771

noncomputable def f (ω x : ℝ) : ℝ := 
  2 * sin (ω * x) * (sqrt 3 * sin (ω * x) + cos (ω * x)) - sqrt 3

theorem minimum_positive_period_of_f : 
  (∃ ω > 0, ∀ x, f ω (x + π / ω) = f ω x) ↔ ω = 1 :=
by
  sorry

theorem range_of_f_for_given_interval :
  (∃ x : set ℝ, x = set.Icc (-π / 6) (π / 6) →
    set.range (λ y, f 1 y) = set.Icc (-2) 0) :=
by
  sorry

end minimum_positive_period_of_f_range_of_f_for_given_interval_l603_603771


namespace find_integer_n_l603_603142

theorem find_integer_n (n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 12) (h2 : n ≡ -4376 [MOD 10]) : n = 4 := 
sorry

end find_integer_n_l603_603142


namespace dream_clock_time_condition_l603_603283

theorem dream_clock_time_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 1)
  (h3 : (120 + 0.5 * 60 * x) = (240 - 6 * 60 * x)) :
  (4 + x) = 4 + 36 + 12 / 13 := by sorry

end dream_clock_time_condition_l603_603283


namespace integer_values_satisfying_sqrt_inequality_l603_603813

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l603_603813


namespace nine_cards_drawn_product_even_is_even_l603_603526

theorem nine_cards_drawn_product_even_is_even :
  ∃ cards : Finset ℕ, cards.card = 9 ∧ 
  (∀ card ∈ cards, card ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) ∧
  (∃ card ∈ cards, card % 2 = 0) :=
by
  sorry

end nine_cards_drawn_product_even_is_even_l603_603526


namespace train_length_l603_603888

theorem train_length (t_platform t_pole : ℕ) (platform_length : ℕ) (train_length : ℕ) :
  t_platform = 39 → t_pole = 18 → platform_length = 350 →
  (train_length + platform_length) / t_platform = train_length / t_pole →
  train_length = 300 :=
by
  intros ht_platform ht_pole hplatform_length hspeeds 
  have h1 : train_length / 18 = (train_length + 350) / 39, from hspeeds
  have h2 : 39 * (train_length / 18) = 39 * ((train_length + 350) / 39), from congrArg (λ x, 39 * x) h1
  sorry

end train_length_l603_603888


namespace problem_1_problem_2_l603_603232

-- Proposition p
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2 * a * x + 2 - a)

-- Proposition q
def q (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2 * x + a ≥ 0

-- Problem 1: Prove that if p is true then a ≤ -2 or a ≥ 1
theorem problem_1 (a : ℝ) (hp : p a) : a ≤ -2 ∨ a ≥ 1 := sorry

-- Problem 2: Prove that if p ∨ q is true then a ≤ -2 or a ≥ 0
theorem problem_2 (a : ℝ) (hpq : p a ∨ q a) : a ≤ -2 ∨ a ≥ 0 := sorry

end problem_1_problem_2_l603_603232


namespace sin_lt_tan_of_acute_l603_603032

theorem sin_lt_tan_of_acute (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) :
  Real.sin α < Real.tan α :=
by
  have h3 : Real.cos α > 0 := Real.cos_pos_of_mem_Ioo ⟨h1, h2⟩
  have h4 : Real.cos α < 1 := by
    apply Real.cos_lt_one_of_nonzero_aux
    linarith
  rw Real.tan_eq_sin_div_cos
  have h5 : 1 / Real.cos α > 1 := by
    rw one_div_pos
    exact h4
    exact h3
  linarith

end sin_lt_tan_of_acute_l603_603032


namespace rectangle_enclosed_by_lines_l603_603166

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l603_603166


namespace ab_cd_not_prime_l603_603331

theorem ab_cd_not_prime {a b c d : ℤ}
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : d > 0)
  (h5 : ac_bd_eq : a * c + b * d = (b + d + a - c) * (b + d - a + c)) :
  ¬Prime (a * b + c * d) :=
sorry

end ab_cd_not_prime_l603_603331


namespace product_of_three_numbers_l603_603398

theorem product_of_three_numbers (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x = 4 * (y + z)) 
  (h3 : y = 7 * z) :
  x * y * z = 28 := 
by 
  sorry

end product_of_three_numbers_l603_603398


namespace distance_to_intersection_of_quarter_circles_eq_zero_l603_603673

open Real

theorem distance_to_intersection_of_quarter_circles_eq_zero (s : ℝ) :
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let center := (s / 2, s / 2)
  let arc_from_A := {p : ℝ × ℝ | p.1^2 + p.2^2 = s^2}
  let arc_from_C := {p : ℝ × ℝ | (p.1 - s)^2 + (p.2 - s)^2 = s^2}
  (center ∈ arc_from_A ∧ center ∈ arc_from_C) →
  let (ix, iy) := (s / 2, s / 2)
  dist (ix, iy) center = 0 :=
by
  sorry

end distance_to_intersection_of_quarter_circles_eq_zero_l603_603673


namespace question1_question2_question3_l603_603642

variables (A B C : ℝ) (a b c : ℝ)
variable (triangle_ABC : triangle ℝ)

-- Condition 1
def condition1 := 1 + (Real.tan C) / (Real.tan B) = 2 * a / b

-- Question 1 - Prove C == π / 3
theorem question1 (h : condition1) : C = π / 3 := sorry

-- Condition 2
def condition2 := Real.cos (B + π / 6) = 1 / 3

-- Question 2 - Prove sin A == (2 * sqrt 6 + 1) / 6
theorem question2 (h1 : condition2) (h2 : C = π / 3) : Real.sin A = (2 * Real.sqrt 6 + 1) / 6 := sorry

-- Condition 3
def condition3 := (a + b)^2 - c^2 = 4

-- Question 3 - Prove minimum value of 3a + b == 4
theorem question3 (h1 : condition3) : ∃ a b, 3 * a + b = 4 := sorry

end question1_question2_question3_l603_603642


namespace choose_three_consecutive_circles_l603_603970

-- Given a figure consisting of 33 circles, prove that the number of ways to choose three consecutive circles is 57.
theorem choose_three_consecutive_circles :
  (number_of_ways_to_choose_three_consecutive_circles 33 = 57) :=
sorry

end choose_three_consecutive_circles_l603_603970


namespace each_hedgehog_ate_1050_strawberries_l603_603829

-- Definitions based on given conditions
def total_strawberries : ℕ := 3 * 900
def remaining_fraction : ℚ := 2 / 9
def remaining_strawberries : ℕ := remaining_fraction * total_strawberries

-- The two hedgehogs and the amount they ate
def two_hedgehogs : ℕ := 2
def total_strawberries_eaten : ℕ := total_strawberries - remaining_strawberries
def strawberries_per_hedgehog : ℕ := total_strawberries_eaten / two_hedgehogs

-- Proof goal: Prove that each hedgehog ate 1050 strawberries
theorem each_hedgehog_ate_1050_strawberries : strawberries_per_hedgehog = 1050 :=
by
  sorry

end each_hedgehog_ate_1050_strawberries_l603_603829


namespace jan_drove_more_distance_than_ian_l603_603846

variables (d : ℝ) (t : ℝ) (s : ℝ) (m : ℝ) (n : ℝ)

-- Conditions from the problem statement
def han_drove_more_time : Prop := t + 1.5
def han_faster_speed : Prop := s + 6
def han_drove_more_distance : Prop := ((s + 6) * (t + 1.5)) = d + 84
def jan_drove_more_time : Prop := t + 3
def jan_faster_speed : Prop := s + 8

-- Additional distance traveled by Jan compared to Ian
def jan_more_distance : Prop := m - d = n

-- Equation to solve
def equation_to_solve : Prop := 8 * t + 3 * s + 24 = 174

theorem jan_drove_more_distance_than_ian (d t s : ℝ) (h : han_drove_more_distance d t s) : jan_more_distance m d n :=
by
  sorry

end jan_drove_more_distance_than_ian_l603_603846


namespace arithmetic_geometric_sequence_result_l603_603984

theorem arithmetic_geometric_sequence_result :
  (∃ d : ℕ, d ≠ 0 ∧ ∀ n : ℕ, a (n : ℕ) = 2 + (n - 1) * d) ∧
  (a 1 = 2 ∧ (a 3)^2 = (a 1) * (a 7)) →
  a 2023 = 2024 :=
sorry

end arithmetic_geometric_sequence_result_l603_603984


namespace prob_X_ge_zero_l603_603481

noncomputable def standard_normal_distribution := sorry

def normal_distribution_mean_variance (μ σ² : ℝ) 
    : sorry := sorry

def probability (event : set ℝ) (dist : sorry) : ℝ := sorry

theorem prob_X_ge_zero (σ : ℝ) 
    (h1 : normal_distribution_mean_variance 1 σ^2 = standard_normal_distribution)
    (h2 : probability {x : ℝ | abs (x - 1) < 1} standard_normal_distribution = 2 / 3)
    : probability {x : ℝ | x ≥ 0} standard_normal_distribution = 5 / 6 :=
sorry

end prob_X_ge_zero_l603_603481


namespace line_circle_no_intersection_l603_603620

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), 3 * x + 4 * y ≠ 12 ∧ x^2 + y^2 = 4 :=
by
  sorry

end line_circle_no_intersection_l603_603620


namespace false_statement_about_isosceles_right_triangle_l603_603030

def is_isosceles_right_triangle (A B C : Type) [triangle A B C] : Prop :=
    ∃ (a b c : ℝ), right_triangle A B C ∧ isosceles A B C

def regular_polygon (P : Type) [polygon P] : Prop :=
    equiangular P ∧ equilateral P

theorem false_statement_about_isosceles_right_triangle :
    ¬ ∀ (A B C : Type) [triangle A B C], is_isosceles_right_triangle A B C → regular_polygon (triangle A B C) :=
  sorry

end false_statement_about_isosceles_right_triangle_l603_603030


namespace find_m_plus_n_l603_603528

def conditions (A B C D E T : Type) (radius_A radius_B radius_C radius_D x m n : ℕ) :=
  radius_A = 10 ∧
  radius_B = 3 ∧
  radius_C = 2 ∧
  radius_D = 2 ∧
  (x * n = 27 * m) ∧
  (m.gcd n = 1) -- relatively prime positive integers

theorem find_m_plus_n (A B C D E T : Type) 
  (radius_A radius_B radius_C radius_D x m n : ℕ) 
  (h : conditions A B C D E T radius_A radius_B radius_C radius_D x m n) : 
  m + n = 32 :=
begin
  -- Proof here
  sorry
end

end find_m_plus_n_l603_603528


namespace count_female_worker_ants_l603_603371

theorem count_female_worker_ants
  (total_ants : ℕ)
  (half_worker_ratio : total_ants / 2)
  (male_worker_ratio : ℕ → ℕ → Prop)
  (twenty_percent_of_worker_ants : male_worker_ratio (total_ants / 2) (total_ants / 2 * 2 / 10))
  (total_ants_eq : total_ants = 110) :
  total_ants / 2 - (total_ants / 2 * 20 / 100) = 44 :=
by
  rw [total_ants_eq]
  sorry

end count_female_worker_ants_l603_603371


namespace rectangle_enclosed_by_lines_l603_603160

theorem rectangle_enclosed_by_lines :
  ∀ (H : ℕ) (V : ℕ), H = 5 → V = 4 → (nat.choose H 2) * (nat.choose V 2) = 60 :=
by
  intros H V h_eq H_eq
  rw [h_eq, H_eq]
  simp
  sorry

end rectangle_enclosed_by_lines_l603_603160


namespace speed_ratio_l603_603039

theorem speed_ratio (L v_a v_b : ℝ) (h1 : v_a = c * v_b) (h2 : (L / v_a) = (0.8 * L / v_b)) :
  v_a / v_b = 5 / 4 :=
by
  sorry

end speed_ratio_l603_603039


namespace forest_edge_replacement_l603_603238

theorem forest_edge_replacement {A B : SimpleGraph V} (hA : A.isForest) (hB : B.isForest)
  (h_vertex_set : A.vertices = B.vertices) (h_edge_count : A.edgeCount > B.edgeCount) :
  ∃ e ∈ A.edges, ¬B.edge_adjacency (e.1, e.2) → (B.edgeSet ∪ {e}) is_forest :=
sorry

end forest_edge_replacement_l603_603238


namespace problem1_problem2_problem3_l603_603404

def total_students : ℕ := 5
def boys : ℕ := 2
def girls : ℕ := 3

-- Problem 1
theorem problem1 : (number_of_arrangements where 2 boys stand together and 3 girls stand together) = 24 :=
sorry

-- Problem 2
theorem problem2 : (number_of_arrangements where boys and girls alternate) = 12 :=
sorry

-- Problem 3
theorem problem3 (boyA_not_on_ends : boy A cannot stand at either end) (girlB_not_in_middle: girl B cannot stand in the middle) :
  (number_of_arrangements where boy A is not at an end and girl B is not in the middle) = 60 :=
sorry

end problem1_problem2_problem3_l603_603404


namespace average_revenue_per_hour_l603_603340

theorem average_revenue_per_hour 
    (sold_A_hour1 : ℕ) (sold_B_hour1 : ℕ) (sold_A_hour2 : ℕ) (sold_B_hour2 : ℕ)
    (price_A_hour1 : ℕ) (price_A_hour2 : ℕ) (price_B_constant : ℕ) : 
    (sold_A_hour1 = 10) ∧ (sold_B_hour1 = 5) ∧ (sold_A_hour2 = 2) ∧ (sold_B_hour2 = 3) ∧
    (price_A_hour1 = 3) ∧ (price_A_hour2 = 4) ∧ (price_B_constant = 2) →
    (54 / 2 = 27) :=
by
  intros
  sorry

end average_revenue_per_hour_l603_603340


namespace connie_initial_marbles_l603_603514

theorem connie_initial_marbles (m_juan m_lisa m_left m_start : ℕ) 
  (h1 : m_juan = 1835) 
  (h2 : m_lisa = 985) 
  (h3 : m_left = 5930) 
  (h_start : m_start = m_juan + m_lisa + m_left) : 
  m_start = 8750 := 
by 
  rw [h1, h2, h3] at h_start 
  rw [nat.add_comm (m_juan + m_lisa) m_left, nat.add_assoc] at h_start 
  exact h_start

end connie_initial_marbles_l603_603514


namespace cubic_no_maximum_value_l603_603426

theorem cubic_no_maximum_value (x : ℝ) : ¬ ∃ M, ∀ x : ℝ, 3 * x^2 + 6 * x^3 + 27 * x + 100 ≤ M := 
by
  sorry

end cubic_no_maximum_value_l603_603426


namespace compute_value_l603_603922

theorem compute_value : 12 - 4 * (5 - 10)^3 = 512 :=
by
  sorry

end compute_value_l603_603922


namespace andrew_ruined_planks_l603_603098

variable (b L k g h leftover plank_total ruin_bedroom ruin_guest : ℕ)

-- Conditions
def bedroom_planks := b
def living_room_planks := L
def kitchen_planks := k
def guest_bedroom_planks := g
def hallway_planks := h
def planks_leftover := leftover

-- Values
axiom bedroom_planks_val : bedroom_planks = 8
axiom living_room_planks_val : living_room_planks = 20
axiom kitchen_planks_val : kitchen_planks = 11
axiom guest_bedroom_planks_val : guest_bedroom_planks = bedroom_planks - 2
axiom hallway_planks_val : hallway_planks = 4
axiom planks_leftover_val : planks_leftover = 6

-- Total planks used and total planks had
def total_planks_used := bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + (2 * hallway_planks)
def total_planks_had := total_planks_used + planks_leftover

-- Planks ruined
def planks_ruined_in_bedroom := ruin_bedroom
def planks_ruined_in_guest_bedroom := ruin_guest

-- Theorem to be proven
theorem andrew_ruined_planks :
  (planks_ruined_in_bedroom = total_planks_had - total_planks_used) ∧
  (planks_ruined_in_guest_bedroom = planks_ruined_in_bedroom) :=
by
  sorry

end andrew_ruined_planks_l603_603098


namespace natural_numbers_solution_l603_603116

theorem natural_numbers_solution :
  ∃ (a b c d : ℕ), 
    ab = c + d ∧ a + b = cd ∧
    ((a, b, c, d) = (2, 2, 2, 2) ∨ (a, b, c, d) = (2, 3, 5, 1) ∨ 
     (a, b, c, d) = (3, 2, 5, 1) ∨ (a, b, c, d) = (2, 2, 1, 5) ∨ 
     (a, b, c, d) = (3, 2, 1, 5) ∨ (a, b, c, d) = (2, 3, 1, 5)) :=
by
  sorry

end natural_numbers_solution_l603_603116


namespace inverse_of_128_l603_603629

def f : ℕ → ℕ := sorry
axiom f_at_5 : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

theorem inverse_of_128 : f⁻¹ 128 = 320 :=
by {
  have basic_values : f 5 = 2 ∧ f (2 * 5) = 4 ∧ f (4 * 5) = 8 ∧ f (8 * 5) = 16 ∧
                       f (16 * 5) = 32 ∧ f (32 * 5) = 64 ∧ f (64 * 5) = 128,
  {
    split, exact f_at_5,
    split, rw [f_property, f_at_5],
    split, rw [f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_property, f_at_5],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 4, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_at_5],
             rw [mul_comm, ← mul_assoc, f_property, mul_comm 8, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_property, f_at_5],
               rw [mul_comm, ← mul_assoc, f_property, mul_comm 16, f_property, mul_comm, f_property],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 32],
         rw [mul_comm, ← mul_assoc, mul_comm 8],
    tauto,
  },
  exact sorry
}

end inverse_of_128_l603_603629


namespace rectangles_from_lines_l603_603155

theorem rectangles_from_lines : 
  (∃ (h_lines : ℕ) (v_lines : ℕ), (h_lines = 5 ∧ v_lines = 4) ∧ (nat.choose h_lines 2 * nat.choose v_lines 2 = 60)) :=
by
  let h_lines := 5
  let v_lines := 4
  have h_choice := nat.choose h_lines 2
  have v_choice := nat.choose v_lines 2
  have answer := h_choice * v_choice
  exact ⟨h_lines, v_lines, ⟨rfl, rfl⟩, rfl⟩

end rectangles_from_lines_l603_603155


namespace find_lambda_l603_603995

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def not_collinear (a b : V) : Prop :=
  ∀ (k : ℝ), a ≠ k • b

theorem find_lambda
  (a b : V)
  (h1 : not_collinear a b)
  (h2 : ∃ k : ℝ, ∀ l : ℝ, λ l, b = k • (a + 2 • b)) :
  ∃ λ : ℝ, λ = 1/2 := 
sorry

end find_lambda_l603_603995


namespace new_person_weight_proof_l603_603752

-- Definitions for the conditions:
def initial_weight (persons: ℕ) (weight: ℕ) : Prop := persons = 8 ∧ weight = 65
def weight_increase (increase: ℕ) : Prop := increase = 16
def average_increase (persons: ℕ) (increase: ℕ) : Prop := persons * 2 = increase

-- Define the weight of the new person
def new_person_weight (old_weight: ℕ) (increase: ℕ): ℕ := old_weight + increase

-- The theorem to prove
theorem new_person_weight_proof : 
  ∀ old_weight persons increase, 
    initial_weight persons old_weight ∧ weight_increase increase ∧ average_increase persons increase 
    → new_person_weight old_weight increase = 81 :=
by
  intro old_weight persons increase
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  have h_persons : persons = 8 := h1.left
  have h_old_weight : old_weight = 65 := h1.right
  have h_increase : increase = 16 := h3
  have : 8 * 2 = 16 := h4
  rw [h_old_weight, h_increase]
  exact rfl

end new_person_weight_proof_l603_603752


namespace isosceles_right_triangle_exists_point_M_l603_603906

noncomputable theory
open Complex

-- Definitions based on given conditions
def is_isosceles_right_triangle (A B C : ℂ) : Prop :=
  (B - A) * (C - A) = I * (norm_sq (B - A) : ℂ)

def exists_point_M (A B C D E : ℂ) (M : ℂ) : Prop :=
  (M.re * 2 = (C.re + E.re)) ∧ (M.im * 2 = (C.im + E.im))

-- The theorem statement proving the problem
theorem isosceles_right_triangle_exists_point_M
  (a b C E A : ℂ)
  (hAC : is_isosceles_right_triangle A C a)
  (hAE : is_isosceles_right_triangle A E b) :
  ∃ M, ∀ θ : ℝ, exists_point_M A a b C (exp (I * θ) * E) M ∧ is_isosceles_right_triangle B M D :=
sorry

end isosceles_right_triangle_exists_point_M_l603_603906


namespace find_f_lg_frac_one_a_l603_603600

noncomputable def f (x : Real) : Real := sin x * cos x + (sin x) / (cos x) + 3

theorem find_f_lg_frac_one_a (a : Real) (h1 : f (log a) = 4) (h2 : f x + f (-x) = 6) :
  f (log (1 / a)) = 2 := 
by
  sorry

end find_f_lg_frac_one_a_l603_603600


namespace correct_system_of_equations_l603_603525

-- Define the given problem conditions.
def cost_doll : ℝ := 60
def cost_keychain : ℝ := 20
def total_cost : ℝ := 5000

-- Define the condition that each gift set needs 1 doll and 2 keychains.
def gift_set_relation (x y : ℝ) : Prop := 2 * x = y

-- Define the system of equations representing the problem.
def system_of_equations (x y : ℝ) : Prop :=
  2 * x = y ∧
  60 * x + 20 * y = total_cost

-- State the theorem to prove that the given system correctly models the problem.
theorem correct_system_of_equations (x y : ℝ) :
  system_of_equations x y ↔ (2 * x = y ∧ 60 * x + 20 * y = 5000) :=
by sorry

end correct_system_of_equations_l603_603525


namespace simplify_and_compute_l603_603111

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end simplify_and_compute_l603_603111


namespace crate_dimensions_l603_603061

-- Define the dimensions of the crate
variables (l w h : ℝ)

-- Define the radius of the cylindrical gas tank
def radius : ℝ := 10

-- Define the diameter of the cylindrical gas tank
def diameter : ℝ := radius * 2

-- Define a condition stating that the cylindrical gas tank must stand upright in the crate
def stands_upright (l w h : ℝ) (r : ℝ) : Prop := 
  l ≥ diameter ∧ w ≥ diameter ∨ l ≥ diameter ∧ h ≥ diameter ∨ w ≥ diameter ∧ h ≥ diameter

theorem crate_dimensions (l w h : ℝ) (hlw : l ≥ 20) (hlh : h ≥ 20) (hwh : w ≥ 20) 
  (upright: stands_upright l w h radius) : (l ≥ 20 ∧ w ≥ 20) ∨ (l ≥ 20 ∧ h ≥ 20) ∨ (w ≥ 20 ∧ h ≥ 20) :=
begin
  sorry
end

end crate_dimensions_l603_603061


namespace power_relationship_l603_603180

variable (a b : ℝ)

theorem power_relationship (h : 0 < a ∧ a < b ∧ b < 2) : a^b < b^a :=
sorry

end power_relationship_l603_603180


namespace integer_solutions_count_l603_603783

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l603_603783


namespace range_f_greater_than_zero_l603_603991

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

theorem range_f_greater_than_zero :
  (∀ x : ℝ, f'(x) = -f'(-x)) →
  f(-2) = 0 →
  (∀ x : ℝ, 0 < x → f(x) + (x/3) * f'(x) > 0) →
  {x : ℝ | f(x) > 0} = {x : ℝ | (-2 < x ∧ x < 0) ∨ (2 < x)} :=
begin
  intros,
  sorry
end

end range_f_greater_than_zero_l603_603991


namespace find_b_l603_603103

noncomputable def cosine_wave_period {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : Prop :=
  let f := λ (x : ℝ), a * Real.cos (b * x + c) + d in
  let T := 2 * Real.pi / b in
  T = Real.pi

theorem find_b (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  cosine_wave_period ha hb hc hd → b = 2 :=
by
  sorry

end find_b_l603_603103


namespace element_not_in_range_l603_603641

noncomputable def f : ℝ → ℝ := λ x, if h : x ≠ 2 then (-3 * x + 1) / (x - 2) else 0

theorem element_not_in_range : ¬ ∃ x : ℝ, x ≠ 2 ∧ f x = -3 := by
  sorry

end element_not_in_range_l603_603641


namespace find_nabla_l603_603624

theorem find_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by {
  sorry
}

end find_nabla_l603_603624


namespace action_figure_cost_l603_603678

def initial_figures : ℕ := 7
def total_figures_needed : ℕ := 16
def total_cost : ℕ := 72

theorem action_figure_cost :
  total_cost / (total_figures_needed - initial_figures) = 8 := by
  sorry

end action_figure_cost_l603_603678


namespace boys_assigned_l603_603820

theorem boys_assigned (B G : ℕ) (h1 : B + G = 18) (h2 : B = G - 2) : B = 8 :=
sorry

end boys_assigned_l603_603820


namespace reporter_earnings_per_hour_l603_603081

/-- A reporter's expected earnings per hour if she writes the entire time. -/
theorem reporter_earnings_per_hour :
  ∀ (word_earnings: ℝ) (article_earnings: ℝ) (stories: ℕ) (hours: ℝ) (words_per_minute: ℕ),
  word_earnings = 0.1 →
  article_earnings = 60 →
  stories = 3 →
  hours = 4 →
  words_per_minute = 10 →
  (stories * article_earnings + (hours * 60 * words_per_minute) * word_earnings) / hours = 105 :=
by
  intros word_earnings article_earnings stories hours words_per_minute
  intros h_word_earnings h_article_earnings h_stories h_hours h_words_per_minute
  sorry

end reporter_earnings_per_hour_l603_603081


namespace simplify_and_compute_l603_603112

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end simplify_and_compute_l603_603112


namespace product_of_factors_of_5_pow_15_l603_603395

theorem product_of_factors_of_5_pow_15 :
  ∏ k in (finset.range (15 + 1)), 5 ^ k = 5 ^ 120 :=
by sorry

end product_of_factors_of_5_pow_15_l603_603395


namespace lambda_value_on_line_lambda_range_second_quadrant_l603_603201

variables {λ : ℝ}
variables (A B C P : ℝ × ℝ)
variables (A_coords : A = (2, 3)) (B_coords : B = (5, 4)) (C_coords : C = (10, 8))
variables (AP_eq : ∃ P, P = A.coords + λ • (A.coords - C.coords))

/-- 
    Given points A(2, 3), B(5, 4), C(10, 8), if $\overrightarrow{AP} = \overrightarrow{AB} + \lambda \overrightarrow{AC}$ 
    with $\lambda \in \mathbb{R}$, then:
    1. Prove that $\lambda = -\frac{1}{3}$ when point P is on the line y = x.
    2. Prove that $-\frac{5}{8} < \lambda < -\frac{4}{5}$ for point P to be in the second quadrant.
-/ 
theorem lambda_value_on_line (
    hP_line : ∃ P, P ∈ (λ P, P.1 = P.2) 
): λ = -1 / 3 := sorry

theorem lambda_range_second_quadrant (
    hP_second_quadrant : ∃ P, P.1 < 0 ∧ P.2 > 0
): -5 / 8 < λ ∧ λ < -4 / 5 := sorry

end lambda_value_on_line_lambda_range_second_quadrant_l603_603201


namespace find_b_l603_603982

theorem find_b (b : ℝ) (h : ∃ x : ℝ, x^2 + b*x - 35 = 0 ∧ x = -5) : b = -2 :=
by
  sorry

end find_b_l603_603982


namespace charles_total_money_l603_603106

-- Definitions based on the conditions in step a)
def number_of_pennies : ℕ := 6
def number_of_nickels : ℕ := 3
def value_of_penny : ℕ := 1
def value_of_nickel : ℕ := 5

-- Calculations in Lean terms
def total_pennies_value : ℕ := number_of_pennies * value_of_penny
def total_nickels_value : ℕ := number_of_nickels * value_of_nickel
def total_money : ℕ := total_pennies_value + total_nickels_value

-- The final proof statement based on step c)
theorem charles_total_money : total_money = 21 := by
  sorry

end charles_total_money_l603_603106


namespace smallest_integer_of_consecutive_odds_l603_603770

theorem smallest_integer_of_consecutive_odds (median greatest : ℤ) 
  (h_median : median = 147) 
  (h_greatest : greatest = 155) 
  (h_consecutive_odds : ∀ n : ℤ, n ∈ set.Icc (median - 8 * 2 + 1) greatest → n % 2 = 1) :
  ∃ smallest : ℤ, smallest ∈ set.Icc (median - 8 * 2 + 1) greatest ∧ smallest = 133 :=
by {
  sorry
}

end smallest_integer_of_consecutive_odds_l603_603770


namespace area_is_eight_l603_603104

noncomputable def area_between_curves : ℝ :=
  ∫ x in 0..4, (4 * x - 8) - (x - 2)^3

theorem area_is_eight : area_between_curves = 8 := by
  sorry

end area_is_eight_l603_603104


namespace largest_possible_difference_l603_603107

-- Defining the estimates and tolerances
def Charlie_estimate : ℝ := 80000
def Chicago_tolerance : ℝ := 0.12
def Daisy_estimate : ℝ := 70000
def Denver_tolerance : ℝ := 0.15
def Ed_estimate : ℝ := 65000

-- Representing the conditions
def Chicago_attendance : set ℝ :=
  {C | 70400 ≤ C ∧ C ≤ 89600}

def Denver_attendance : set ℝ :=
  {D | 60869 ≤ D ∧ D ≤ 82353}

def Edmonton_attendance : set ℝ :=
  {E | E = 65000}

-- The proof problem as a Lean statement
theorem largest_possible_difference
  (C : ℝ) (D : ℝ) (E : ℝ)
  (hC : C ∈ Chicago_attendance)
  (hD : D ∈ Denver_attendance)
  (hE : E ∈ Edmonton_attendance) :
  abs (max (max C D) E - min (min C D) E) = 29000 :=
sorry

end largest_possible_difference_l603_603107


namespace emily_order_cost_l603_603527

theorem emily_order_cost :
  let curtain_price := 30.00
  let curtains := 2 * curtain_price
  let wall_print_price := 15.00
  let wall_prints := 9 * wall_print_price
  let discount_percent := 0.10
  let tax_percent := 0.08
  let installation_service := 50.00
  let total_before_discount := curtains + wall_prints
  let discount_amount := discount_percent * total_before_discount
  let total_after_discount := total_before_discount - discount_amount
  let sales_tax := tax_percent * total_after_discount
  let total_after_tax := total_after_discount + sales_tax
  let total_order_cost := total_after_tax + installation_service
  in total_order_cost = 239.54 := sorry

end emily_order_cost_l603_603527


namespace largest_x_value_not_defined_eq_2_l603_603836

theorem largest_x_value_not_defined_eq_2:
  ∀ (x : ℝ), (10 * x ^ 2 - 30 * x + 20 = 0) → x = 2 → x = 2 :=
by
  intro x hx h2
  exact h2
  sorry

end largest_x_value_not_defined_eq_2_l603_603836


namespace intersection_distance_l603_603661

open Real EuclideanSpace

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 2 * t, 2 + 4 * t, 3 + 4 * t)

def unit_sphere (p : ℝ × ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + p.3^2 = 1

theorem intersection_distance :
  let t1 := (-52 + Real.sqrt (2704 - 2184)) / 84
      t2 := (-52 - Real.sqrt (2704 - 2184)) / 84 in
  let dist := Real.sqrt ( (2 * (t1 - t2))^2 + (4 * (t1 - t2))^2 + (4 * (t1 - t2))^2 ) in
  dist = 12 * Real.sqrt 145 / 33 :=
by
  sorry

end intersection_distance_l603_603661


namespace sum_of_valid_ns_l603_603989

theorem sum_of_valid_ns : 
  ∑ n in Finset.filter (λ n, 5 * n ^ 2 + 3 * n - 5 % 15 = 0) (Finset.range 100), n = 635 :=
by
  sorry

end sum_of_valid_ns_l603_603989


namespace calculate_female_worker_ants_l603_603369

theorem calculate_female_worker_ants :
  ∀ (total_ants : ℕ) (half_worker_ants_rate male_worker_ants_rate : ℚ), 
  total_ants = 110 →
  half_worker_ants_rate = 1 / 2 →
  male_worker_ants_rate = 1 / 5 →
  let total_worker_ants := total_ants * half_worker_ants_rate,
      male_worker_ants := total_worker_ants * male_worker_ants_rate in
  total_worker_ants - male_worker_ants = 44 := by
  sorry

end calculate_female_worker_ants_l603_603369


namespace sum_of_interior_angles_l603_603636

noncomputable def exterior_angle (n : ℕ) := 360 / n

theorem sum_of_interior_angles (n : ℕ) (h : exterior_angle n = 45) :
  180 * (n - 2) = 1080 :=
by
  sorry

end sum_of_interior_angles_l603_603636


namespace largest_fraction_of_consecutive_odds_is_three_l603_603583

theorem largest_fraction_of_consecutive_odds_is_three
  (p q r s : ℕ)
  (h1 : 0 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h_odd1 : p % 2 = 1)
  (h_odd2 : q % 2 = 1)
  (h_odd3 : r % 2 = 1)
  (h_odd4 : s % 2 = 1)
  (h_consecutive1 : q = p + 2)
  (h_consecutive2 : r = q + 2)
  (h_consecutive3 : s = r + 2) :
  (r + s) / (p + q) = 3 :=
sorry

end largest_fraction_of_consecutive_odds_is_three_l603_603583


namespace prob_rain_12_to_14_l603_603062

-- Definitions of probabilities and events
def P (E : Prop) : ℝ := sorry

-- Conditions
axiom PA : P(A) = 0.5
axiom PB : P(B) = 0.4
axiom A_B_independent : P(A ∧ B) = P(A) * P(B)

-- Definitions of no-rain events
def notA : Prop := ¬A
def notB : Prop := ¬B
def notD : Prop := ¬(A ∨ B)

-- Given event D
def D : Prop := A ∨ B

-- Probability of the combined no-rain event
axiom Prob_noA : P(notA) = 1 - P(A)
axiom Prob_noB : P(notB) = 1 - P(B)
axiom Prob_noD : P(notD) = P(notA) * P(notB)

-- Proof statement
theorem prob_rain_12_to_14 : P(D) = 0.7 :=
by
  -- Calculate the probability of D using the given conditions
  sorry

end prob_rain_12_to_14_l603_603062


namespace find_x_l603_603260

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l603_603260


namespace infinite_n_multiples_of_six_available_l603_603726

theorem infinite_n_multiples_of_six_available :
  ∃ (S : Set ℕ), (∀ n ∈ S, ∃ (A : Matrix (Fin 3) (Fin (n : ℕ)) Nat),
    (∀ (i : Fin n), (A 0 i + A 1 i + A 2 i) % 6 = 0) ∧ 
    (∀ (i : Fin 3), (Finset.univ.sum (λ j => A i j)) % 6 = 0)) ∧
  Set.Infinite S :=
sorry

end infinite_n_multiples_of_six_available_l603_603726


namespace find_P_B_given_A_l603_603412

noncomputable def coin_toss_three_times : Type := Fin 2 → Fin 2

def event_A (outcome : coin_toss_three_times) : Prop :=
  ∃ i, outcome i = 1  -- at least one tail

def event_B (outcome : coin_toss_three_times) : Prop :=
  (∑ (i : Fin 3), outcome i) = 1  -- exactly one head

theorem find_P_B_given_A : @cond_prob _ _
  { outcome: coin_toss_three_times // event_A outcome }
  { outcome: coin_toss_three_times // event_A outcome ∧ event_B outcome }
  = 3 / 7 :=
by
  sorry

end find_P_B_given_A_l603_603412


namespace parallel_lines_slope_l603_603588

theorem parallel_lines_slope (a : ℝ) (h : ∀ x y : ℝ, (x + a * y + 6 = 0) → ((a - 2) * x + 3 * y + 2 * a = 0)) : a = -1 :=
by
  sorry

end parallel_lines_slope_l603_603588


namespace ratio_PQ_QR_l603_603723

noncomputable def radius : ℝ := 2
noncomputable def PQ : ℝ := 4
noncomputable def QR : ℝ := 2 * real.pi
noncomputable def ratio := PQ / QR

theorem ratio_PQ_QR : ratio = 2 / real.pi := by {
  have hPQ : PQ = 2 * radius := by sorry,
  have hQR : QR = 2 * real.pi := by sorry,
  exact sorry,
}

end ratio_PQ_QR_l603_603723


namespace value_of_x_l603_603278

theorem value_of_x (x : ℝ) (h : x = 90 + (11 / 100) * 90) : x = 99.9 :=
by {
  sorry
}

end value_of_x_l603_603278


namespace triangle_angles_l603_603308

theorem triangle_angles (A B C M N: Point) (BM AN: Line)
  (cond1: is_median BM ∧ length BM = (1/2) * length AN ∧ is_angle_bisector AN)
  (cond2: ∠C B M = 3 * ∠C A N) :
  ∠A = 36 ∧ ∠B = 108 ∧ ∠C = 36 :=
by
  sorry

end triangle_angles_l603_603308


namespace find_a_from_circle_and_chord_l603_603593

theorem find_a_from_circle_and_chord 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0)
  (line_eq : ∀ x y : ℝ, x + y + 2 = 0)
  (chord_length : ∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 + 2*x1 - 2*y1 + a = 0 ∧ x2^2 + y2^2 + 2*x2 - 2*y2 + a = 0 ∧ x1 + y1 + 2 = 0 ∧ x2 + y2 + 2 = 0 → (x1 - x2)^2 + (y1 - y2)^2 = 16) :
  a = -4 :=
by
  sorry

end find_a_from_circle_and_chord_l603_603593


namespace value_of_f_at_2008_l603_603502

-- Conditions
variable {f : ℝ → ℝ}
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f(-x) = -f(x)
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f(1-x) = f(1+x)

-- Statement to prove
theorem value_of_f_at_2008 (h_odd : odd_function f) (h_eq : functional_equation f) : f 2008 = 0 :=
by
  sorry

end value_of_f_at_2008_l603_603502


namespace odd_and_monotonic_l603_603093

-- Definitions based on the conditions identified
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_monotonic_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x ≤ f y
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement without the proof
theorem odd_and_monotonic :
  is_odd f ∧ is_monotonic_increasing f :=
sorry

end odd_and_monotonic_l603_603093


namespace find_a5_l603_603323

variable {a_1 d : ℝ}

def sum_of_first_n_terms (n : ℕ) : ℝ := (n / 2) * (2 * a_1 + (n - 1) * d)
def a_seq (n : ℕ) : ℝ := a_1 + (n - 1) * d
def S2_eq_S6 := sum_of_first_n_terms 2 = sum_of_first_n_terms 6
def a4_eq_1 := a_seq 4 = 1

theorem find_a5 (h1 : S2_eq_S6) (h2 : a4_eq_1) : a_seq 5 = -1 :=
by
  sorry

end find_a5_l603_603323


namespace distance_to_water_source_l603_603091

theorem distance_to_water_source (d : ℝ) :
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 5)) → 7 < d ∧ d < 8 :=
by
  sorry

end distance_to_water_source_l603_603091


namespace count_correct_statements_l603_603669

noncomputable theory
open_locale real

variables {P A B C D E : Type*} [regular_tetrahedron P A B C] [midpoint D A B] [midpoint E B C]

def is_perpendicular (plane1 plane2 : Type*) : Prop := sorry
def is_parallel (line : Type*) (plane : Type*) : Prop := sorry

theorem count_correct_statements (h1 : ¬ is_perpendicular (plane P A C) (plane P B D))
                                 (h2 : is_parallel (line A C) (plane P D E))
                                 (h3 : is_perpendicular (line A B) (plane P D C)) :
    2 = (ite h1 0 1) + (ite h2 1 0) + (ite h3 1 0) :=
begin
  sorry
end

end count_correct_statements_l603_603669


namespace max_principals_in_period_l603_603936

-- Define the setting with the conditions
def principal_term : ℕ := 3
def period_length : ℕ := 8

-- Statement of the problem equivalent to proving the math problem
theorem max_principals_in_period (principal_term : ℕ) (period_length : ℕ) 
  (h_term : principal_term = 3) (h_period : period_length = 8) : 
  ∃ max_principals : ℕ, max_principals = 4 :=
by
  -- sorry is a placeholder for the proof.
  use 4
  trivial

end max_principals_in_period_l603_603936


namespace probability_even_tails_same_flip_l603_603826

/-- Probability that all three individuals (Tom, Dick, and Harry) flip their coins 
an even number of times and get their first tail on the same flip is 1/63 --/
theorem probability_even_tails_same_flip : 
  let p := (λ k : ℕ, (1 / 2)^(2 * k)) in
  let P := ∑' k : ℕ, p(k)^3 in
  P = 1 / 63 :=
by
  sorry

end probability_even_tails_same_flip_l603_603826


namespace value_of_6z_l603_603279

theorem value_of_6z (x y z : ℕ) (h1 : 6 * z = 2 * x) (h2 : x + y + z = 26) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 6 * z = 36 :=
by
  sorry

end value_of_6z_l603_603279


namespace not_simplifiable_by_difference_of_squares_l603_603438

theorem not_simplifiable_by_difference_of_squares :
  ¬(∃ a b : ℝ, (-x + y) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (-x - y) * (-x + y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y + x) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y - x) * (x + y) = a^2 - b^2) :=
sorry

end not_simplifiable_by_difference_of_squares_l603_603438


namespace worker_A_time_l603_603845

-- Define the variables and their relationships
theorem worker_A_time (A : ℝ) (H1 : ∀ B: ℝ, B = 10) (H2 : ∀ AB: ℝ, AB = 4.444444444444445)
  (H3 : 1 / A + 1 / H1(10) = 1 / H2(4.444444444444445)) : A = 8 :=
by
  sorry

end worker_A_time_l603_603845


namespace perfect_square_of_sides_of_triangle_l603_603688

theorem perfect_square_of_sides_of_triangle 
  (a b c : ℤ) 
  (h1: a > 0 ∧ b > 0 ∧ c > 0)
  (h2: a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_abc: Int.gcd (Int.gcd a b) c = 1)
  (h3: (a^2 + b^2 - c^2) % (a + b - c) = 0)
  (h4: (b^2 + c^2 - a^2) % (b + c - a) = 0)
  (h5: (c^2 + a^2 - b^2) % (c + a - b) = 0) : 
  ∃ n : ℤ, n^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
  ∃ m : ℤ, m^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end perfect_square_of_sides_of_triangle_l603_603688


namespace part1_part2_l603_603186

variable {f : ℝ → ℝ}

theorem part1 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : f 1 = 0 :=
by sorry

theorem part2 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f (-x) + f (3 - x) ≥ 2 :=
by sorry

end part1_part2_l603_603186


namespace probability_of_shaded_shape_l603_603299

   def total_shapes : ℕ := 4
   def shaded_shapes : ℕ := 1

   theorem probability_of_shaded_shape : shaded_shapes / total_shapes = 1 / 4 := 
   by
     sorry
   
end probability_of_shaded_shape_l603_603299


namespace valid_outcomes_when_X_eq_1_l603_603864

def is_valid_outcome (s : String) : Prop :=
  s.length = 3 ∧ s.count('H') = 1 ∧ s.count('T') = 2

theorem valid_outcomes_when_X_eq_1 :
  {s | is_valid_outcome s} = {"HTT", "THT", "TTH"} :=
  sorry

end valid_outcomes_when_X_eq_1_l603_603864


namespace jogger_distance_l603_603043

theorem jogger_distance (t : ℝ) (h : 16 * t = 12 * t + 10) : 12 * t = 30 :=
by
  have ht : t = 2.5,
  {
    linarith,
  },
  rw ht,
  norm_num,
  sorry

end jogger_distance_l603_603043


namespace domain_of_g_l603_603226

theorem domain_of_g (a b : ℝ) (f : ℝ → ℝ) (h_even : ∀ x, f(x) = f(-x)) :
  (g : ℝ → ℝ) (g = λ x, sqrt (log a x - 1)) → 
  a = 1 / 2 →
  ∀ x, g x ≠ NaN ↔ (0 < x ∧ x ≤ 1 / 2) :=
by
  sorry

end domain_of_g_l603_603226


namespace find_x_l603_603259

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l603_603259


namespace solve_for_y_l603_603364

theorem solve_for_y (y : ℝ) : 4 * y + 6 * y = 450 - 10 * (y - 5) → y = 25 :=
by
  sorry

end solve_for_y_l603_603364


namespace calculate_female_worker_ants_l603_603368

theorem calculate_female_worker_ants :
  ∀ (total_ants : ℕ) (half_worker_ants_rate male_worker_ants_rate : ℚ), 
  total_ants = 110 →
  half_worker_ants_rate = 1 / 2 →
  male_worker_ants_rate = 1 / 5 →
  let total_worker_ants := total_ants * half_worker_ants_rate,
      male_worker_ants := total_worker_ants * male_worker_ants_rate in
  total_worker_ants - male_worker_ants = 44 := by
  sorry

end calculate_female_worker_ants_l603_603368


namespace infinitely_many_n_dividing_floor_sqrt_3_mul_d_l603_603930
noncomputable def floor_sqrt_3_mul_d (n : ℕ) : ℕ :=
  int.to_nat (real.floor (real.sqrt 3 * (real.of_nat (nat.divisors n).length)))

theorem infinitely_many_n_dividing_floor_sqrt_3_mul_d :
  ∃ᶠ (n : ℕ) in filter.at_top, n % (floor_sqrt_3_mul_d n) = 0 := sorry

end infinitely_many_n_dividing_floor_sqrt_3_mul_d_l603_603930


namespace sequence_a_proof_l603_603191

def sequence_a (n : ℕ) : ℕ := if n = 0 then 0 else (n * (n + 1))

def sum_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else ∑ i in finset.range(1, n+1), sequence_a i

theorem sequence_a_proof (n : ℕ) (hn : n ≠ 0) :
  let S_n := sum_seq n in
  a_1 = 2 ∧ (∀ k > 0, 3 * S_k = sequence_a k * (k + 2)) → 
  sequence_a n = n * (n + 1) :=
by {
  intros,
  sorry,
}

end sequence_a_proof_l603_603191


namespace number_of_white_balls_l603_603658

theorem number_of_white_balls (total : ℕ) (freq_red freq_black : ℚ) (h1 : total = 120) 
                              (h2 : freq_red = 0.15) (h3 : freq_black = 0.45) : 
                              (total - total * freq_red - total * freq_black = 48) :=
by sorry

end number_of_white_balls_l603_603658


namespace train_passing_time_l603_603890

noncomputable def length_of_train : ℝ := 450
noncomputable def speed_kmh : ℝ := 80
noncomputable def length_of_station : ℝ := 300
noncomputable def speed_m_per_s : ℝ := speed_kmh * 1000 / 3600 -- Convert km/hour to m/second
noncomputable def total_distance : ℝ := length_of_train + length_of_station
noncomputable def passing_time : ℝ := total_distance / speed_m_per_s

theorem train_passing_time : abs (passing_time - 33.75) < 0.01 :=
by
  sorry

end train_passing_time_l603_603890


namespace retailer_bought_120_pens_l603_603085

theorem retailer_bought_120_pens (P : ℝ) : 
  (N : ℝ) N = 120 :=
begin
  -- Given conditions
  let cost_price := 36 * P,
  let selling_price_per_pen := 0.99 * P,
  let profit_percentage := 2.3,
  -- Profit equation
  let profit := profit_percentage * cost_price,
  let selling_price_total := N * selling_price_per_pen,
  -- Main equation derived from profit and cost
  have eq1 : profit = selling_price_total - cost_price,
  -- Solve for N
  calc
    profit = profit_percentage * cost_price : by sorry
    _ := sorry,
end

end retailer_bought_120_pens_l603_603085


namespace lcm_9_12_15_l603_603005

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l603_603005


namespace area_is_13pi_l603_603722

-- Define the points C and D
def C : ℝ × ℝ := (-2, 3)
def D : ℝ × ℝ := (4, -1)

-- Define the distance formula between two points (x1, y1) and (x2, y2)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the radius as half the distance between C and D
noncomputable def radius : ℝ :=
  distance C D / 2

-- Define the area of the circle in terms of π
noncomputable def area_of_circle : ℝ :=
  real.pi * (radius ^ 2)

-- The theorem stating the area of the circle is 13π
theorem area_is_13pi : area_of_circle = 13 * real.pi :=
by sorry

end area_is_13pi_l603_603722


namespace each_hedgehog_ate_1050_strawberries_l603_603828

noncomputable def total_strawberries_per_basket : ℕ := 900
noncomputable def number_of_baskets : ℕ := 3
noncomputable def fraction_remaining : ℚ := 2 / 9
noncomputable def number_of_hedgehogs : ℕ := 2

theorem each_hedgehog_ate_1050_strawberries :
  let total_strawberries := number_of_baskets * total_strawberries_per_basket,
      strawberries_remaining := (fraction_remaining * total_strawberries).toNat,
      strawberries_eaten := total_strawberries - strawberries_remaining,
      strawberries_per_hedgehog := strawberries_eaten / number_of_hedgehogs
  in strawberries_per_hedgehog = 1050 := 
by
  -- Proof here
  sorry

end each_hedgehog_ate_1050_strawberries_l603_603828


namespace geometric_sequence_formula_sum_bn_formula_l603_603188

noncomputable def geometric_sequence (an : ℕ → ℕ) : Prop :=
an 1 = 2 ∧ (∀ n, an n = 2 * 2 ^ (n - 1))

noncomputable def bn (an : ℕ → ℕ) (n : ℕ) : ℕ :=
an n * Int.log an n

noncomputable def Tn (an: ℕ → ℕ) (n: ℕ) : ℕ :=
 ∑ i in finset.range n, bn an i

theorem geometric_sequence_formula (an : ℕ → ℕ) :
  an 1 = 2 → (∀ n, 2 * (an 3 + 2) = an 2 + an 4) → (∀ n, an n = 2 ^ n) :=
by
  sorry

theorem sum_bn_formula (an : ℕ → ℕ) (n : ℕ) :
  (∀ n, an n = 2 ^ n) →
  Tn an n = 
  →

by
  sorry

end geometric_sequence_formula_sum_bn_formula_l603_603188


namespace problem_solution_l603_603699

noncomputable def solve_problem : ℝ :=
  let x y : ℝ
  let h1 : 3 * x^2 + 6 * x * y + 4 * y^2 = 1
  let m := (1 - Real.sqrt 3) / 2
  let M := (1 + Real.sqrt 3) / 2
  m * M

theorem problem_solution (x y : ℝ)
  (h1 : 3 * x^2 + 6 * x * y + 4 * y^2 = 1) :
  solve_problem = -1 / 2 :=
by
  sorry

end problem_solution_l603_603699


namespace find_nabla_l603_603623

theorem find_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by {
  sorry
}

end find_nabla_l603_603623


namespace a_56_equals_neg_sqrt3_l603_603605

def seq (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  (seq (n-1) - Real.sqrt 3) / (Real.sqrt 3 * seq (n-1) + 1)

theorem a_56_equals_neg_sqrt3 :
  seq 56 = - Real.sqrt 3 :=
sorry

end a_56_equals_neg_sqrt3_l603_603605


namespace number_of_lines_determined_by_12_points_l603_603178

theorem number_of_lines_determined_by_12_points : 
  ∀ (P : Finset (Fin 12)), (∀ p₁ p₂ p₃ ∈ P, p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ → ¬ (collinear ℝ {p₁, p₂, p₃})) → 
  (P.card.choose 2) = 66 :=
by
  intros P hP
  have h_card : P.card = 12 := by sorry
  rw [Finset.card_choose_two, h_card]
  norm_num
  -- exact proof we'll write specifics here 

end number_of_lines_determined_by_12_points_l603_603178


namespace length_of_AB_l603_603720

theorem length_of_AB :
  ∀ (A B C D E F G : Type) [add_group A] [has_smul ℚ A] 
    (midpoint : A → A → A)
    (C_mid : C = midpoint A B) 
    (D_mid : D = midpoint A C)
    (E_mid : E = midpoint A D)
    (F_mid : F = midpoint A E)
    (G_mid : G = midpoint A F)
    (AG_length : dist A G = 2),
  dist A B = 64 := 
sorry

end length_of_AB_l603_603720


namespace math_proof_problem_l603_603229

noncomputable def f (x : ℝ) := Real.log (Real.sin x) * Real.log (Real.cos x)

def domain (k : ℤ) : Set ℝ := { x | 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi / 2 }

def is_even_shifted : Prop :=
  ∀ x, f (x + Real.pi / 4) = f (- (x + Real.pi / 4))

def has_unique_maximum : Prop :=
  ∃! x, 0 < x ∧ x < Real.pi / 2 ∧ ∀ y, 0 < y ∧ y < Real.pi / 2 → f y ≤ f x

theorem math_proof_problem (k : ℤ) :
  (∀ x, x ∈ domain k → f x ∈ domain k) ∧
  ¬ (∀ x, f (-x) = f x) ∧
  is_even_shifted ∧
  has_unique_maximum :=
by
  sorry

end math_proof_problem_l603_603229


namespace qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l603_603355

variable (m : Int)

theorem qiqi_initial_batteries (m : Int) : 
  let Qiqi_initial := 2 * m - 2
  Qiqi_initial = 2 * m - 2 := sorry

theorem qiqi_jiajia_difference_after_transfer (m : Int) : 
  let Qiqi_after := 2 * m - 2 - 2
  let Jiajia_after := m + 2
  Qiqi_after - Jiajia_after = m - 6 := sorry

end qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l603_603355


namespace investment_ratio_same_period_l603_603042

-- Define the profits of A and B
def profit_A : ℕ := 60000
def profit_B : ℕ := 6000

-- Define their investment ratio given the same time period
theorem investment_ratio_same_period : profit_A / profit_B = 10 :=
by
  -- Proof skipped 
  sorry

end investment_ratio_same_period_l603_603042


namespace integer_satisfying_values_l603_603801

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l603_603801


namespace complement_A_inter_B_l603_603337

universe u

variables {α : Type u}

-- Define the sets U, A, and B
def U := set α
def A : set ℝ := {x | x ≥ 1}
def B : set ℝ := {x | -1 ≤ x ∧ x < 2}

-- Complement of intersection of A and B in U
def A_inter_B : set ℝ := {x | 1 ≤ x ∧ x < 2}
def complement_in_U (s : set α) : set α := { x | x ∈ U ∧ x ∉ s }

theorem complement_A_inter_B :
  complement_in_U A_inter_B = {x : ℝ | x < 1 ∨ x ≥ 2} :=
sorry

end complement_A_inter_B_l603_603337


namespace designated_time_to_B_l603_603064

theorem designated_time_to_B (s v : ℝ) (x : ℝ) (V' : ℝ)
  (h1 : s / 2 = (x + 2) * V')
  (h2 : s / (2 * V') + 1 + s / (2 * (V' + v)) = x) :
  x = (v + Real.sqrt (9 * v ^ 2 + 6 * v * s)) / v :=
by
  sorry

end designated_time_to_B_l603_603064


namespace positive_integers_of_m_n_l603_603954

theorem positive_integers_of_m_n (m n : ℕ) (p : ℕ) (a : ℕ) (k : ℕ) (h_m_ge_2 : m ≥ 2) (h_n_ge_2 : n ≥ 2) 
  (h_prime_q : Prime (m + 1)) (h_4k_1 : m + 1 = 4 * k - 1) 
  (h_eq : (m ^ (2 ^ n - 1) - 1) / (m - 1) = m ^ n + p ^ a) : 
  (m, n) = (p - 1, 2) ∧ Prime p ∧ ∃k, p = 4 * k - 1 := 
by {
  sorry
}

end positive_integers_of_m_n_l603_603954


namespace probability_difference_l603_603465

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_at_least_61_heads : ℝ :=
  (∑ k in Finset.range (110 - 60), (binom 110 (61 + k) / 2^110))

noncomputable def probability_less_than_49_heads : ℝ :=
  (∑ k in Finset.range 49, (binom 110 k / 2^110))

theorem probability_difference :
  probability_at_least_61_heads - probability_less_than_49_heads =
  binom 110 61 / 2^110 :=
by
  sorry

end probability_difference_l603_603465


namespace fraction_inequality_solution_l603_603923

theorem fraction_inequality_solution (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) :
  3 * x + 2 < 2 * (5 * x - 4) → (10 / 7) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l603_603923


namespace concur_point_l603_603289

-- Definitions based on conditions
variable {ABC : Triangle}
variable {circumcircle : Circle ABC}
variable {incircle_A : Incircle (angle A)}
variable {incircle_B : Incircle (angle B)}
variable {incircle_C : Incircle (angle C)}
variable {A1 : Point}
variable {B1 : Point}
variable {C1 : Point}

-- Incircles touch the circumcircle
axiom hA1 : incircle_A.touches circumcircle A1
axiom hB1 : incircle_B.touches circumcircle B1
axiom hC1 : incircle_C.touches circumcircle C1

-- Main statement
theorem concur_point : ConcurrentLines (line_through_points A A1) (line_through_points B B1) (line_through_points C C1) := 
  sorry

end concur_point_l603_603289


namespace sarah_house_units_digit_l603_603938

-- Sarah's house number has two digits
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- The four statements about Sarah's house number
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def has_digit_7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Exactly three out of the four statements are true
def exactly_three_true (n : ℕ) : Prop :=
  (is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ is_odd n ∧ ¬is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (is_multiple_of_5 n ∧ ¬is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n) ∨
  (¬is_multiple_of_5 n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_7 n)

-- Main statement
theorem sarah_house_units_digit : ∃ n : ℕ, is_two_digit n ∧ exactly_three_true n ∧ n % 10 = 5 :=
by
  sorry

end sarah_house_units_digit_l603_603938


namespace number_of_girls_l603_603349

theorem number_of_girls (total_children boys girls : ℕ) 
    (total_children_eq : total_children = 60)
    (boys_eq : boys = 22)
    (compute_girls : girls = total_children - boys) : 
    girls = 38 :=
by
    rw [total_children_eq, boys_eq] at compute_girls
    simp at compute_girls
    exact compute_girls

end number_of_girls_l603_603349


namespace circumcenter_fixed_circle_l603_603572

open EuclideanGeometry

theorem circumcenter_fixed_circle {S1 S2 : Circle}
  (P Q : Point)
  (h_intersect : P ≠ Q ∧ P ∈ S1 ∧ Q ∈ S1 ∧ P ∈ S2 ∧ Q ∈ S2)
  (A1 B1 : Point)
  (h_A1B1 : A1 ≠ P ∧ A1 ≠ Q ∧ B1 ≠ P ∧ B1 ≠ Q)
  (A2 B2 : Point)
  (h_A2 : A2 ∈ S2 ∧ A2 ≠ P ∧ collinear {A1, P, A2})
  (h_B2 : B2 ∈ S2 ∧ B2 ≠ P ∧ collinear {B1, P, B2})
  (C : Point)
  (h_C : collinear {A1, B1, C} ∧ collinear {A2, B2, C}) :
  ∃ (O : Point), O = circumcenter (triangle.mk A1 A2 C) ∧ O ∈ (Miquel_circle S1 S2 P Q) := 
begin
  sorry
end

end circumcenter_fixed_circle_l603_603572


namespace sum_range_median_is_118_l603_603488

theorem sum_range_median_is_118 :
  ∀ (scores : list ℝ),
    (list.length scores = 22) →
    (list.maximum scores = some 98) →
    (list.minimum scores = some 56) →
    (scores.nth_le 10 sorry = 76) →
    (scores.nth_le 11 sorry = 76) →
    list.sum scores =
    (98 - 56) + (76 + 76) / 2 := by
  sorry

end sum_range_median_is_118_l603_603488


namespace corset_total_cost_l603_603918

def purple_bead_cost : ℝ := 50 * 20 * 0.12
def blue_bead_cost : ℝ := 40 * 18 * 0.10
def gold_bead_cost : ℝ := 80 * 0.08
def red_bead_cost : ℝ := 30 * 15 * 0.09
def silver_bead_cost : ℝ := 100 * 0.07

def total_cost : ℝ := purple_bead_cost + blue_bead_cost + gold_bead_cost + red_bead_cost + silver_bead_cost

theorem corset_total_cost : total_cost = 245.90 := by
  sorry

end corset_total_cost_l603_603918


namespace f_2016_eq_neg1_l603_603587

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_property : ∀ x y : ℝ, f x * f y = f (x + y) + f (x - y)

theorem f_2016_eq_neg1 : f 2016 = -1 := 
by 
  sorry

end f_2016_eq_neg1_l603_603587


namespace sum_of_digits_of_binary_300_l603_603018

theorem sum_of_digits_of_binary_300 : 
  ∑ digit in (Nat.digits 2 300), digit = 3 :=
by
  sorry

end sum_of_digits_of_binary_300_l603_603018


namespace stock_fall_afternoon_l603_603940

-- Given definitions and conditions stated in the problem
axiom stock_rise_morning : ℝ := 2
axiom initial_stock_value : ℝ := 100
axiom target_stock_value : ℝ := 200
axiom day_reaches_target : ℕ := 100

-- Lean statement to prove how much the stock falls in the afternoon
theorem stock_fall_afternoon (x : ℝ) 
  (H1 : stock_rise_morning > 0)
  (H2 : initial_stock_value > 0)
  (H3 : target_stock_value > initial_stock_value)
  (H4 : day_reaches_target > 0)
  (H5 : initial_stock_value + 99 * (stock_rise_morning - x) = 200) : 
  x = 98 / 99 :=
sorry

end stock_fall_afternoon_l603_603940


namespace rectangle_lines_combinations_l603_603163

theorem rectangle_lines_combinations : 5.choose 2 * 4.choose 2 = 60 := by
  sorry

end rectangle_lines_combinations_l603_603163


namespace conic_problem_l603_603198

theorem conic_problem (ρ θ x y t₁ t₂ : ℝ) 
    (C_polar : ∀ θ, ρ^2 = 12 / (3 + sin θ ^ 2))
    (A_fixed : (0, -sqrt 3))
    (F₁_foci : (-1, 0))
    (F₂_foci : (1, 0)) :
    (∀ x y, 3 * x^2 + 4 * y^2 = 12 ↔ (∃ ρ θ, ρ^2 = 12 / (3 + sin θ ^ 2) ∧ x = ρ * cos θ ∧ y = ρ * sin θ)) ∧
    (∀ x, (∃ k, k = sqrt 3 ∧ A_fixed.2= k * (x - A_fixed.1)) → (y = sqrt 3 * (x + 1))) ∧ 
    (∀ t₁ t₂, (5 * t₁^2 - 4 * t₁ - 12 = 0 ∧ 5 * t₂^2 - 4 * t₂ - 12 = 0) → (abs t₁ * abs t₂ = 12  / 5)) := 
sorry

end conic_problem_l603_603198


namespace intersection_A_B_l603_603580

open Set

-- Definitions from conditions
def A := { x : ℝ | x^2 + x - 6 < 0 }
def B := { y : ℝ | ∃ x : ℝ, y = sqrt (x + 1) }

-- The statement to prove
theorem intersection_A_B : A ∩ B = { z : ℝ | 0 ≤ z ∧ z < 2 } :=
sorry

end intersection_A_B_l603_603580


namespace find_two_digit_number_l603_603075

def is_positive (n : ℕ) := n > 0
def is_even (n : ℕ) := n % 2 = 0
def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def product_of_digits_is_square (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  ∃ k : ℕ, (tens * units) = k * k

theorem find_two_digit_number (N : ℕ) 
  (h_pos : is_positive N) 
  (h_ev : is_even N) 
  (h_mult_9 : is_multiple_of_9 N)
  (h_prod_square : product_of_digits_is_square N) 
: N = 90 := by 
  sorry

end find_two_digit_number_l603_603075


namespace dressing_q_vinegar_percentage_l603_603360

/-- 
Given:
1. P is 30% vinegar and 70% oil.
2. Q is V% vinegar and the rest is oil.
3. The new dressing is produced from 10% of P and 90% of Q and is 12% vinegar.
Prove:
The percentage of vinegar in dressing Q is 10%.
-/
theorem dressing_q_vinegar_percentage (V : ℝ) (h : 0.10 * 0.30 + 0.90 * V = 0.12) : V = 0.10 :=
by 
    sorry

end dressing_q_vinegar_percentage_l603_603360


namespace rectangles_from_lines_l603_603152

theorem rectangles_from_lines : 
  (∃ (h_lines : ℕ) (v_lines : ℕ), (h_lines = 5 ∧ v_lines = 4) ∧ (nat.choose h_lines 2 * nat.choose v_lines 2 = 60)) :=
by
  let h_lines := 5
  let v_lines := 4
  have h_choice := nat.choose h_lines 2
  have v_choice := nat.choose v_lines 2
  have answer := h_choice * v_choice
  exact ⟨h_lines, v_lines, ⟨rfl, rfl⟩, rfl⟩

end rectangles_from_lines_l603_603152


namespace percentage_increase_in_expenses_l603_603477

noncomputable def monthlySalary : ℝ := 4166.67
noncomputable def initialSavingsPercentage : ℝ := 0.20
noncomputable def newSavingsAmount : ℝ := 500

theorem percentage_increase_in_expenses :
  let initialSavings := initialSavingsPercentage * monthlySalary,
      increaseInExpenses := initialSavings - newSavingsAmount,
      originalExpenses := monthlySalary - initialSavings in
  ((increaseInExpenses / originalExpenses) * 100) = 10 :=
by
  sorry

end percentage_increase_in_expenses_l603_603477


namespace expr_B_not_simplified_using_difference_of_squares_l603_603436

def expr_A (x y : ℝ) := (-x - y) * (-x + y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (y + x) * (x - y)
def expr_D (x y : ℝ) := (y - x) * (x + y)

theorem expr_B_not_simplified_using_difference_of_squares (x y : ℝ) :
  ∃ x y, ¬ ∃ a b, expr_B x y = a^2 - b^2 :=
sorry

end expr_B_not_simplified_using_difference_of_squares_l603_603436


namespace num_candied_apples_l603_603108

-- Definitions for conditions
variables {A : ℕ} -- Number of candied apples
constant price_apple : ℕ := 2
constant num_grapes : ℕ := 12
constant price_grape : ℚ := 1.5
constant total_earnings : ℚ := 48

-- Condition stating the total earning equation
axiom earnings_eq : price_apple * A + num_grapes * price_grape = total_earnings

-- The proof goal
theorem num_candied_apples : A = 15 :=
by {
  -- Setup the conditions
  exact earnings_eq,
  -- This is where the proof would be finished, normally
  sorry
}

end num_candied_apples_l603_603108


namespace friday_birth_of_dickens_l603_603498

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem friday_birth_of_dickens :
  let regular_year_days := 365
  let leap_year_days := 366
  let total_years := 200
  let leap_years := 49  -- already computed as 49 in the steps
  let regular_years := total_years - leap_years
  let total_days_in_regular_years := regular_years * regular_year_days
  let total_days_in_leap_years := leap_years * leap_year_days
  let total_days := total_days_in_regular_years + total_days_in_leap_years
  (total_days % 7 = 4) : 
  day_of_week (date.add_days (date.mk 2012 2 7) (-total_days)) = "Friday"
:= sorry

end friday_birth_of_dickens_l603_603498


namespace semicircle_radius_l603_603485

theorem semicircle_radius (b h : ℝ) (base_eq_b : b = 16) (height_eq_h : h = 15) :
  let s := (2 * 17) / 2
  let area := 240 
  s * (r : ℝ) = area → r = 120 / 17 :=
  by
  intros s area
  sorry

end semicircle_radius_l603_603485


namespace _l603_603138

noncomputable theorem polynomial_characterization (n : ℕ) :
  (∀ (P : Polynomial ℝ), P.degree ≤ n ∧ (∀ i : ℕ, P.coeff i ≥ 0) ∧ (∀ x : ℝ, x > 0 → P.eval x * P.eval (1 / x) ≤ (P.eval 1) ^ 2) ↔
  (∃ j : ℕ, j ≤ n ∧ ∃ a_j : ℝ, a_j ≥ 0 ∧ P = Polynomial.C a_j * Polynomial.X ^ j)) := sorry

end _l603_603138


namespace dog_food_consumption_l603_603067

theorem dog_food_consumption :
  (∀ dog_cups_per_feeding: ℝ,
    let cup_weight := 1 / 4,
    let total_weight := 9 * 20,
    let dogs := 2,
    let feeds_per_day := 2,
    let days_per_month := 30 in
    (dog_cups_per_feeding = 
     (total_weight / dogs) / ((feeds_per_day * days_per_month) * cup_weight)) → 
    dog_cups_per_feeding = 6) :=
begin
  intro dog_cups_per_feeding,
  simp only [dog_cups_per_feeding],
  let cup_weight := 1 / 4,
  let total_weight := 9 * 20,
  let dogs := 2,
  let feeds_per_day := 2,
  let days_per_month := 30,
  calc dog_cups_per_feeding
      = (total_weight / dogs) / ((feeds_per_day * days_per_month) * cup_weight) : by rfl
  ... = 6 : sorry
end

end dog_food_consumption_l603_603067


namespace conjugate_of_z_l603_603967

noncomputable def z : ℂ := 5 * complex.I / (1 - 2 * complex.I)

theorem conjugate_of_z : complex.conj z = -2 - complex.I := by
  sorry

end conjugate_of_z_l603_603967


namespace count_integer_values_satisfying_condition_l603_603807

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l603_603807


namespace average_weight_increase_l603_603379

variable (A N X : ℝ)

theorem average_weight_increase (hN : N = 135.5) (h_avg : A + X = (9 * A - 86 + N) / 9) : 
  X = 5.5 :=
by
  sorry

end average_weight_increase_l603_603379


namespace integer_satisfying_values_l603_603796

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l603_603796


namespace height_of_door_l603_603384

/-- Given the dimensions of a room, three windows, the cost per square foot for whitewashing, 
    and the total cost, prove that the height of the door is 3 feet. -/
theorem height_of_door 
  (length width height : ℕ)
  (door_width door_unknown_height : ℕ)
  (window_width window_height : ℕ)
  (number_of_windows : ℕ)
  (cost_per_square_foot total_cost : ℕ)
  (h : ℕ)
  (h_proof : 
    let perimeter := 2 * (length + width),
        area_of_walls := perimeter * height,
        area_of_door := door_width * door_unknown_height,
        area_of_window := window_width * window_height,
        total_area_of_windows := number_of_windows * area_of_window,
        area_to_be_whitewashed := area_of_walls - (area_of_door + total_area_of_windows),
        calculated_cost := area_to_be_whitewashed * cost_per_square_foot
    in
    calculated_cost = total_cost
  ) 
  : h = 3
:= sorry

end height_of_door_l603_603384


namespace measure_minor_arc_BD_l603_603282

open Real
open EuclideanGeometry

-- Define the geometric context and the angle condition
variables {O B C D : Point}  -- Points O, B, C, D
variable (h : Circle O B)   -- Circle with center O and passing through B
variable (inscribed_angle : ∠ B C D = 30)  -- ∠ B C D is 30 degrees

-- The statement to prove: the measure of minor arc BD is 60 degrees
theorem measure_minor_arc_BD (h : Circle O B) (inscribed_angle : ∠ B C D = 30) :
  arc_measure O B D = 60 :=
sorry

end measure_minor_arc_BD_l603_603282


namespace sum_of_digits_of_binary_300_l603_603015

theorem sum_of_digits_of_binary_300 : 
  ∑ digit in (Nat.digits 2 300), digit = 3 :=
by
  sorry

end sum_of_digits_of_binary_300_l603_603015


namespace evaluate_product_l603_603130

theorem evaluate_product (a : ℤ) (h : a = -1) : ((a - 3) * (a - 2) * (a - 1) * a = 0) :=
by
  rw [h]
  norm_num

end evaluate_product_l603_603130


namespace price_percentage_gain_l603_603486

theorem price_percentage_gain (P : ℝ) : 
  let increased_price := P * 1.31
      first_discounted_price := increased_price * 0.90
      final_price := first_discounted_price * 0.85
      percentage_gain := (final_price - P) / P * 100
  in percentage_gain = 0.215 :=
by
  sorry

end price_percentage_gain_l603_603486


namespace no_such_function_exists_l603_603523

noncomputable def f (x : ℝ) : ℝ := sorry

theorem no_such_function_exists :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f (f x) = x^2 - 2) → false :=
begin
  assume f,
  assume h : ∀ x : ℝ, f (f x) = x^2 - 2,
  sorry
end

end no_such_function_exists_l603_603523


namespace problem6_l603_603295

noncomputable def AcknowledgingCircumcircleTangent (ABC : Type) :=
  ∀ (A B C H F M Q K : Point) 
  (circ_ABC circ_KQH circ_FKM: Circle) 
  (angleHQA : Angle) 
  (angleHKQ : Angle), 
  abcAcute : isAcute ∆ABC
  ∧ AB > AC 
  ∧ isCircumcircle circ_ABC ∆ABC 
  ∧ isOrthocenter H ∆ABC 
  ∧ isAltitudeFoot F A ∆ABC 
  ∧ isMidpoint M B C 
  ∧ isPointOnCircle Q circ_ABC 
  ∧ isRightAngle angleHQA 
  ∧ isAngleHQA angleHQA H Q A 
  ∧ isPointOnCircle K circ_ABC 
  ∧ isRightAngle angleHKQ 
  ∧ isAngleHKQ angleHKQ H K Q
  ∧ isCircumcircle circ_KQH ∆KQH 
  ∧ isCircumcircle circ_FKM ∆FKM 
  → isTangent circ_KQH circ_FKM

-- Note: sorry is used here to skip the proof, as implementing a complete proof is outside the current scope.
theorem problem6 : AcknowledgingCircumcircleTangent := sorry

end problem6_l603_603295


namespace least_number_when_increased_by_8_divisible_l603_603046

theorem least_number_when_increased_by_8_divisible (n : ℕ) :
  (∀ k, k ∈ {24, 32, 36, 54} → (n + 8) % k = 0) → n = 856 :=
by
  intro h
  have h1 := h 24 (by simp)
  have h2 := h 32 (by simp)
  have h3 := h 36 (by simp)
  have h4 := h 54 (by simp)
  -- Proof requires showing n + 8 = 864 and n = 856, skipping for brevity
  -- as the requirement is only for the theorem statement.
  sorry

end least_number_when_increased_by_8_divisible_l603_603046


namespace probability_multiple_of_3_l603_603774

theorem probability_multiple_of_3 :
  let golf_balls := Finset.range 21
  let total_balls := golf_balls.card
  let multiples_of_3 := {x ∈ golf_balls | x % 3 = 0}
  let favorable_cases := multiples_of_3.card
  let probability := favorable_cases / total_balls
  in probability = 3 / 10 :=
by
  sorry

end probability_multiple_of_3_l603_603774


namespace f_C_even_and_monotonically_increasing_l603_603026

def log_base (b x : ℝ) : ℝ := log x / log b

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x y ∈ domain, x < y → f x < f y

def f_C (x : ℝ) : ℝ := 1 - log_base (1/2) (abs x)

def domain : Set ℝ := {x | 0 < x}

theorem f_C_even_and_monotonically_increasing :
  is_even f_C ∧ is_monotonically_increasing f_C domain :=
by
  sorry

end f_C_even_and_monotonically_increasing_l603_603026


namespace rectangle_enclosed_by_lines_l603_603167

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l603_603167


namespace a_formula_T_sum_l603_603567

open Nat

-- Given sequence \( \{a_n\} \) with the sum of the first \( n \) terms denoted as \( S_n \) and satisfying \( 2S_n = n^2 - n \)
def S (n : ℕ) := (n * (n - 1)) / 2

-- Define \( a_n \)
def a (n : ℕ) := n - 1

-- Define the sequence \( b_n \)
def b (n : ℕ) : ℚ :=
  if h : n % 2 = 0 then
    (1 : ℚ) / (n.toReal^2 + 2 * n.toReal)
  else
    (2 : ℚ)^(n - 1)

-- Sum the sequence \( b_n \) to get \( T_n \)
noncomputable def T (n : ℕ) : ℚ :=
  if h : n % 2 = 0 then
    (2^n - 1) / 3 + (n) / (4 * (n + 2))
  else
    (2^(n + 1) - 1) / 3 + (n - 1) / (4 * (n + 1))

-- Proof that \( a_n = n - 1 \)
theorem a_formula (n : ℕ) : a n = n - 1 :=
  sorry

-- Proof that \( T_n \) equals the given formula
theorem T_sum (n : ℕ) : T n = 
  if h : n % 2 = 0 then 
    (2^n - 1) / 3 + (n) / (4 * (n + 2))
  else
    (2^(n + 1) - 1) / 3 + (n - 1) / (4 * (n + 1)) :=
  sorry

end a_formula_T_sum_l603_603567


namespace mother_daughter_age_relation_l603_603821

theorem mother_daughter_age_relation (x : ℕ) (hc1 : 43 - x = 5 * (11 - x)) : x = 3 := 
sorry

end mother_daughter_age_relation_l603_603821


namespace distribution_methods_l603_603406

theorem distribution_methods :
  (Nat.choose 9 3) = (number_of_distribution_methods 6 3 9) :=
begin
  -- Let's define the number of different distribution methods
  def number_of_distribution_methods (math_books : ℕ) (lang_books : ℕ) (people : ℕ) : ℕ :=
    Nat.choose people lang_books,

  sorry,
end

end distribution_methods_l603_603406


namespace rectangle_enclosed_by_lines_l603_603170

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l603_603170


namespace inverse_of_128_l603_603628

def f : ℕ → ℕ := sorry
axiom f_at_5 : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

theorem inverse_of_128 : f⁻¹ 128 = 320 :=
by {
  have basic_values : f 5 = 2 ∧ f (2 * 5) = 4 ∧ f (4 * 5) = 8 ∧ f (8 * 5) = 16 ∧
                       f (16 * 5) = 32 ∧ f (32 * 5) = 64 ∧ f (64 * 5) = 128,
  {
    split, exact f_at_5,
    split, rw [f_property, f_at_5],
    split, rw [f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_property, f_at_5],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 4, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_at_5],
             rw [mul_comm, ← mul_assoc, f_property, mul_comm 8, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_property, f_at_5],
               rw [mul_comm, ← mul_assoc, f_property, mul_comm 16, f_property, mul_comm, f_property],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 32],
         rw [mul_comm, ← mul_assoc, mul_comm 8],
    tauto,
  },
  exact sorry
}

end inverse_of_128_l603_603628


namespace max_triangle_area_and_coordinates_l603_603353

noncomputable def ellipse (x y : ℝ) : Prop := ((x + 4)^2) / 9 + (y^2) / 16 = 1
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Define the area function for the triangle PAB
noncomputable def triangle_area (x0 y0 : ℝ) : ℝ := 
  (1 / 2) * ((y0^2 - 4 * x0) / 2) * 2 * real.sqrt(y0^2 - 4 * x0)

-- Point P coordinates
def P (θ : ℝ) : ℝ × ℝ := (-4 + 3 * real.cos θ, 4 * real.sin θ)

-- Proof that the maximum area is given by the expected value and occurs at the specified coordinates
theorem max_triangle_area_and_coordinates :
  ∃ θ, P θ = (-41 / 8, real.sqrt 55 / 2) ∨ P θ = (-41 / 8, - real.sqrt 55 / 2) ∧ 
        ∀ x y, ellipse x y → triangle_area x y ≤ 137 * real.sqrt 137 / 16 :=
sorry

end max_triangle_area_and_coordinates_l603_603353


namespace limit_solution_l603_603921

open Real
open Topology.Filter

noncomputable def limit_problem : ℝ :=
  lim (atTop.map (λ x : ℝ, x)) (λ x, (1 - ln (1 + x^3)) ^ (3 / (x^2 * arcsin x)))

theorem limit_solution : limit_problem = exp (-3) := 
by {
  sorry
}

end limit_solution_l603_603921


namespace arithmetic_sequence_term_1023_l603_603763

theorem arithmetic_sequence_term_1023 (p r : ℚ) (h1 : 2 * p - 2 * r = 15) (h2 : 15 - 2 * r = 4 * p + r) :
  let d := -2 * r in
  let a₁ := 2 * p in
  let aₙ := a₁ + (1022) * d in
  aₙ = 61215 / 14 :=
by
  let d := -2 * r
  let a₁ := 2 * p 
  sorry

end arithmetic_sequence_term_1023_l603_603763


namespace number_of_transform_sequences_l603_603924

-- Definitions to represent transformations:
inductive Transform
| R1 | R2 | H1 | H2

open Transform

-- Assume we have a regular hexagon with vertices at specific coordinates
structure Hexagon :=
(a b c d e f : ℤ × ℤ)

-- Define the transformations R1, R2, H1, H2
def R1_transformation : Hexagon → Hexagon := sorry
def R2_transformation : Hexagon → Hexagon := sorry
def H1_transformation : Hexagon → Hexagon := sorry
def H2_transformation : Hexagon → Hexagon := sorry

-- Apply a sequence of transformations to a hexagon
def apply_transforms (transforms : List Transform) (hex : Hexagon) : Hexagon :=
  transforms.foldr (λ t hex,
    match t with
    | R1 => R1_transformation hex
    | R2 => R2_transformation hex
    | H1 => H1_transformation hex
    | H2 => H2_transformation hex) hex

-- Initial hexagon configuration based on the given coordinates
def initial_hexagon : Hexagon :=
{ a := (1, nat_sqrt 3) -- (1, √3)
, b := (-1, nat_sqrt 3) -- (-1, √3)
, c := (-2, 0)
, d := (-1, -nat_sqrt 3) -- (-1, -√3)
, e := (1, -nat_sqrt 3) -- (1, -√3)
, f := (2, 0) }

-- State that the number of sequences of 15 transformations sending the hexagon back to the initial position is 4^14
theorem number_of_transform_sequences :
  (filter (λ s,
    apply_transforms s initial_hexagon = initial_hexagon)
    (List.replicateM 15 [R1, R2, H1, H2])).length = 4^14 := sorry

end number_of_transform_sequences_l603_603924


namespace integer_satisfying_values_l603_603798

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l603_603798


namespace total_windows_needed_l603_603478

theorem total_windows_needed (w_installed h_per_window h_remaining : Nat) (h1 : w_installed = 6) (h2 : h_per_window = 6) (h3 : h_remaining = 18) : 
  let w_remaining := h_remaining / h_per_window in
  let total_windows := w_installed + w_remaining in
  total_windows = 9 := 
by 
  sorry

end total_windows_needed_l603_603478


namespace polyhedron_is_regular_icosahedron_l603_603719

theorem polyhedron_is_regular_icosahedron (a : ℝ) (polyhedron : Type)
  [linear_ordered_field ℝ] [linear_ordered_ring ℝ] [nontrivial ℝ]
  (has_6_vertices : ∀ p : polyhedron, (vertex_count p) = 6)
  (has_5_faces : ∀ p : polyhedron, (face_count p) = 5)
  (is_positioned_by_square_face : ∀ p1 p2 : polyhedron, 
    attached_by_square_face p1 p2 → perpendicular_EF_edges p1 p2)
  (distance_condition : ∀ b : ℝ, (a^2 = b + b^2) → 
    (let m := (b / 2) in m^2 = (a * (a - b)) / 4))
  : convex_polyhedron_formed_by(polyhedron, a) = regular_icosahedron := 
sorry

end polyhedron_is_regular_icosahedron_l603_603719


namespace inequality_log_div_l603_603266

theorem inequality_log_div (x y a b : ℝ) (hx : 0 < x) (hxy : x < y) (hy : y < 1)
  (hb : 1 < b) (hba : b < a) :
  (ln x) / b < (ln y) / a := 
sorry

end inequality_log_div_l603_603266


namespace integer_values_satisfying_sqrt_inequality_l603_603810

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l603_603810


namespace alternating_tree_planting_l603_603405

theorem alternating_tree_planting (willows poplars : ℕ) (h1 : willows = 4) (h2 : poplars = 4) : 
  ∃ (n : ℕ), (n = 2 * (4!) * (4!)) ∧ n = 1152 :=
by 
  use 1152
  split
  sorry
  rfl

end alternating_tree_planting_l603_603405


namespace star_sum_of_angles_l603_603127

theorem star_sum_of_angles (n : ℕ) (h₀ : n = 8) (h₁ : ∀ k, 0 ≤ k ∧ k < 8 → 
(∃ angles : vector ℝ 8, (∀ i j, (i ≠ j) → (angles.nth i < angles.nth j))) ) : 
  (∑ k in finset.range 8, (angle_at_tip k 8)) = 1440 :=
by
  sorry

end star_sum_of_angles_l603_603127


namespace first_term_a1_geometric_sequence_general_term_a_existence_of_m_l603_603993

-- Definitions of the sequences and conditions are based on the given problem.

noncomputable def S (n : ℕ) : ℚ := 3 * a n - 5 * n
noncomputable def a (n : ℕ) : ℚ := if n = 1 then 5 / 2 else (3 / 2) * a (n - 1) + 5 / 2
noncomputable def b (n : ℕ) : ℚ := (9 * n + 4) / (a n + 5)

-- Theorems to prove
theorem first_term_a1 : a 1 = 5 / 2 :=
sorry

theorem geometric_sequence : (a n + 5) = (3 / 2) * (a (n - 1) + 5) :=
sorry

theorem general_term_a : ∀ n, a n = (15 / 2) * (3 / 2)^(n - 1) - 5 :=
sorry

theorem existence_of_m : ∃ m : ℚ, (m > 88 / 45) ∧ ∀ n : ℕ, b n < m :=
sorry

end first_term_a1_geometric_sequence_general_term_a_existence_of_m_l603_603993


namespace sam_has_75_dollars_l603_603912

variable (S B : ℕ)

def condition1 := B = 2 * S - 25
def condition2 := S + B = 200

theorem sam_has_75_dollars (h1 : condition1 S B) (h2 : condition2 S B) : S = 75 := by
  sorry

end sam_has_75_dollars_l603_603912


namespace each_hedgehog_ate_1050_strawberries_l603_603830

-- Definitions based on given conditions
def total_strawberries : ℕ := 3 * 900
def remaining_fraction : ℚ := 2 / 9
def remaining_strawberries : ℕ := remaining_fraction * total_strawberries

-- The two hedgehogs and the amount they ate
def two_hedgehogs : ℕ := 2
def total_strawberries_eaten : ℕ := total_strawberries - remaining_strawberries
def strawberries_per_hedgehog : ℕ := total_strawberries_eaten / two_hedgehogs

-- Proof goal: Prove that each hedgehog ate 1050 strawberries
theorem each_hedgehog_ate_1050_strawberries : strawberries_per_hedgehog = 1050 :=
by
  sorry

end each_hedgehog_ate_1050_strawberries_l603_603830


namespace find_constant_and_increasing_interval_l603_603601

def g (x m : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 + m

theorem find_constant_and_increasing_interval (m : ℝ) (k : ℤ):
  (∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), g x m = 6) →
  m = 3 ∧ (∀ x ∈ Set.Icc (k * Real.pi - 2 * Real.pi / 3) (k * Real.pi - Real.pi / 6), 
           g (-x) 3 ≤ g (2 -x) 3 → g (-x) 3 < g (-x + ε) 3) := 
sorry

end find_constant_and_increasing_interval_l603_603601


namespace sum_first_n_terms_sequence_general_term_b_l603_603234

-- Problem (I)
theorem sum_first_n_terms_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)) 
  (h2 : a 5 = 5) 
  (h3 : S 7 = 28) 
  (h4 : ∀ n : ℕ, S n = (n * (n + 1)) / 2) 
  : ∀ n : ℕ, T n = 2 * (1 - (1 / (n + 1))) := 
sorry

-- Problem (II)
theorem general_term_b (a : ℕ → ℕ) (b : ℕ → ℕ) (q : ℕ → ℝ) 
  (h1 : b 1 = 1) 
  (h2 : ∀ n : ℕ, b (n + 1) = b n + q (a n)) 
  : ∀ n : ℕ, b n = if q 1 = 1 then n else (1 - q n) / (1 - q 1) := 
sorry

end sum_first_n_terms_sequence_general_term_b_l603_603234


namespace compute_sum_sq_roots_of_polynomial_l603_603513

theorem compute_sum_sq_roots_of_polynomial :
  (∃ p q r : ℚ, (∀ x : ℚ, polynomial.eval x (3 * X^3 - 2 * X^2 + 6 * X - 9) = 0 → (x = p ∨ x = q ∨ x = r)) ∧
     p^2 + q^2 + r^2 = -32/9) :=
sorry

end compute_sum_sq_roots_of_polynomial_l603_603513


namespace derivative_at_zero_l603_603102

def f (x : ℝ) : ℝ :=
if x = 0 then 0 else arctan (x^3 - x^(3/2) * sin (1/(3*x)))

theorem derivative_at_zero : deriv f 0 = 0 :=
sorry

end derivative_at_zero_l603_603102


namespace number_of_regions_on_sphere_l603_603052

theorem number_of_regions_on_sphere (n : ℕ) (h : ∀ {a b c: ℤ}, a ≠ b → b ≠ c → a ≠ c → True) : 
  ∃ a_n, a_n = n^2 - n + 2 := 
by
  sorry

end number_of_regions_on_sphere_l603_603052


namespace proof_equivalence_l603_603609

-- Definitions for the conditions
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + y - 1 = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, (3 * a - 4) * x - y - 2 = 0
def parallel (k1 k2 : ℝ) : Prop := k1 = k2
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 1) ^ 2 = 8

-- Main statement combining the questions and answers
theorem proof_equivalence (a : ℝ) (l1_parallel_l2 : parallel (-a) (3 * a - 4)) :
  (a = 1) ∧ (∀ x y, line2 a x y → circle_eq x y) :=
by
  sorry

end proof_equivalence_l603_603609


namespace integer_values_satisfying_sqrt_inequality_l603_603815

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l603_603815


namespace tangent_secant_problem_l603_603320

theorem tangent_secant_problem (P O T A B : Point)
    (h1 : ¬ (P ∈ O))  -- P is outside of circle O
    (h2 : PT tangent O at T)  -- PT is tangent to circle O at T
    (h3 : secant P A B ∈ O) -- Secant PAB intersects O at A and B
    (h4 : distance P A < distance P B)
    (h5 : distance P A = 3)
    (h6 : distance P T = distance A B - distance P A)
  : distance P B = 12 := 
sorry

end tangent_secant_problem_l603_603320


namespace no_polynomial_satisfies_inequality_l603_603727

variables (P : Real → Real)

theorem no_polynomial_satisfies_inequality (hP : ∀ x : Real, differentiable ℝ P ∧ differentiable ℝ (P') ∧ differentiable ℝ (P'') ∧ differentiable ℝ (P''') ∧ is_poly P) :
  ¬ (∀ x : ℝ, (P' x) * (P'' x) > (P x) * (P''' x)) :=
sorry

#check no_polynomial_satisfies_inequality

end no_polynomial_satisfies_inequality_l603_603727


namespace inverse_of_128_l603_603630

def f : ℕ → ℕ := sorry
axiom f_at_5 : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

theorem inverse_of_128 : f⁻¹ 128 = 320 :=
by {
  have basic_values : f 5 = 2 ∧ f (2 * 5) = 4 ∧ f (4 * 5) = 8 ∧ f (8 * 5) = 16 ∧
                       f (16 * 5) = 32 ∧ f (32 * 5) = 64 ∧ f (64 * 5) = 128,
  {
    split, exact f_at_5,
    split, rw [f_property, f_at_5],
    split, rw [f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_at_5],
    split, rw [f_property, f_property, f_property, f_property, f_at_5],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 4, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_at_5],
             rw [mul_comm, ← mul_assoc, f_property, mul_comm 8, f_property, mul_comm, f_property],
    split, rw [f_property, f_property, f_property, f_property, f_property, f_property, f_at_5],
               rw [mul_comm, ← mul_assoc, f_property, mul_comm 16, f_property, mul_comm, f_property],
         rw [mul_comm, ← mul_assoc, f_property, mul_comm 32],
         rw [mul_comm, ← mul_assoc, mul_comm 8],
    tauto,
  },
  exact sorry
}

end inverse_of_128_l603_603630


namespace find_ellipse_eq_find_slope_range_l603_603903

open Real

-- Definitions of conditions
def ellipse (x y a b : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1
def vertex (x y : ℝ) := (x, y)
def quadrilateral_area (area : ℝ) (a b : ℝ) := 2 * a * b = area

-- Theorem for Part I
theorem find_ellipse_eq (a b : ℝ) (ha : a > b) (hb : b > 0) (hA : vertex 0 (-2)) 
  (hquad : quadrilateral_area (4 * sqrt 5) a b) :
  ellipse 5 4 a b := 
sorry

-- Definitions for Part II
def line_through_point_with_slope (x y k : ℝ) := y = k * x - 3
def intersects (ellipse_eq : Prop) (line_eq : Prop) := -- Intersection condition to be filled in
def intersection_points (x1 y1 x2 y2 : ℝ) := (x1, y1) = ('B') ∧ (x2, y2) = ('C')
def intersection_conditions (m n : ℝ) := -- Intersection M and N on y = -3

-- Theorem for Part II
theorem find_slope_range (P : vertex 0 (-3)) (k : ℝ) :
  (abs k > 1 ∧ abs k ≤ 3) → 
  (|PM| + |PN| ≤ 15) :=
sorry

end find_ellipse_eq_find_slope_range_l603_603903


namespace sales_discount_percentage_l603_603066

theorem sales_discount_percentage :
  ∀ (P N : ℝ) (D : ℝ),
  (N * 1.12 * (P * (1 - D / 100)) = P * N * (1 + 0.008)) → D = 10 :=
by
  intros P N D h
  sorry

end sales_discount_percentage_l603_603066


namespace symmetric_point_with_respect_to_x_axis_l603_603382

-- Definition of point M
def point_M : ℝ × ℝ := (3, -4)

-- Define the symmetry condition with respect to the x-axis
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Statement that the symmetric point to point M with respect to the x-axis is (3, 4)
theorem symmetric_point_with_respect_to_x_axis : symmetric_x point_M = (3, 4) :=
by
  -- This is the statement of the theorem; the proof will be added here.
  sorry

end symmetric_point_with_respect_to_x_axis_l603_603382


namespace number_of_rectangles_l603_603147

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 4) : 
  (nat.choose H 2) * (nat.choose V 2) = 60 := by
  rw [hH, hV]
  norm_num

end number_of_rectangles_l603_603147


namespace num_three_digit_div_by_three_l603_603552

-- Necessary conditions: three-digit number, using digits 0, 1, 2, 3, without repetition, divisible by 3.
theorem num_three_digit_div_by_three : 
  let digits := {0, 1, 2, 3}
  (∀ n : ℕ, (∃ h t u : ℕ, h ∈ digits ∧ t ∈ digits ∧ u ∈ digits ∧ h ≠ t ∧ t ≠ u ∧ h ≠ u ∧ 
  100 * h + 10 * t + u = n ∧ 
  (h ≠ 0) ∧ 
  (n % 3 = 0)) ↔ n ∈ (Icc 100 999)) →
  card {n : ℕ | ∃ h t u : ℕ, h ∈ digits ∧ t ∈ digits ∧ u ∈ digits ∧ h ≠ t ∧ t ≠ u ∧ h ≠ u ∧ 
  100 * h + 10 * t + u = n ∧ 
  (h ≠ 0) ∧ 
  (n % 3 = 0)} = 10 :=
sorry

end num_three_digit_div_by_three_l603_603552


namespace f_inv_128_l603_603633

noncomputable def f : ℕ → ℕ := sorry -- Placeholder for the function definition.

axiom f_5 : f 5 = 2           -- Condition 1: f(5) = 2
axiom f_2x : ∀ x, f (2 * x) = 2 * f x  -- Condition 2: f(2x) = 2f(x) for all x

theorem f_inv_128 : f⁻¹ 128 = 320 := sorry -- Prove that f⁻¹(128) = 320 given the conditions

end f_inv_128_l603_603633


namespace fraction_zero_l603_603433

theorem fraction_zero (x : ℝ) (h : (x^2 - 1) / (x + 1) = 0) : x = 1 := 
sorry

end fraction_zero_l603_603433


namespace son_l603_603037

variable (S M : ℤ)

-- Conditions
def condition1 : Prop := M = S + 24
def condition2 : Prop := M + 2 = 2 * (S + 2)

theorem son's_age : condition1 S M ∧ condition2 S M → S = 22 :=
by
  sorry

end son_l603_603037


namespace point_not_center_of_symmetry_l603_603440

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt 3 * real.sin x * real.cos x + real.cos x ^ 2 - 1 / 2

theorem point_not_center_of_symmetry : ¬ (∃ x₀ y₀, f (2 * (π / 6) - x₀) = 2 * y₀ - f x₀) :=
begin
  -- sorry to skip proof
  sorry
end

end point_not_center_of_symmetry_l603_603440


namespace solve_equation_l603_603366

-- Defining the conditions
def greatestInt (x : ℝ) : ℤ := floor x
def fractionalPart (x : ℝ) : ℝ := x - ↑(greatestInt x)

-- The main theorem
theorem solve_equation (x : ℝ) : 
  0 < x ∧ x ≠ greatestInt x ∧
  (8 / fractionalPart x = 9 / x + 10 / (greatestInt x : ℝ)) → 
  x = 3 / 2 :=
by
  sorry

end solve_equation_l603_603366


namespace problem_solution_l603_603216

noncomputable def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(-x) = f(x)

noncomputable def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f(x) ≤ f(y)

variables (f : ℝ → ℝ)

theorem problem_solution (hf_even : is_even f)
  (hf_mono : is_monotonic f 0 5)
  (h_inequality : f (-3) < f (-1)) :
  f 0 > f 1 :=
begin
  sorry
end

end problem_solution_l603_603216


namespace count_integer_values_satisfying_condition_l603_603805

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l603_603805


namespace distinct_products_count_l603_603613

def distinct_products (s : set ℕ) : set ℕ :=
  {p | ∃ a b, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ p = a * b} ∪
  {p | ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ p = a * b * c} ∪
  {p | ∃ a b c d, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ p = a * b * c * d}

theorem distinct_products_count : 
  distinct_products {1, 2, 4, 7, 13} = 11 := 
by 
  sorry

end distinct_products_count_l603_603613


namespace strawberries_for_mom_l603_603343

-- Define the conditions as Lean definitions
def dozen : ℕ := 12
def strawberries_picked : ℕ := 2 * dozen
def strawberries_eaten : ℕ := 6

-- Define the statement to be proven
theorem strawberries_for_mom : (strawberries_picked - strawberries_eaten) = 18 := by
  sorry

end strawberries_for_mom_l603_603343


namespace length_of_train_l603_603881

variables (L : ℝ) (t1 t2 : ℝ) (length_platform : ℝ)

-- Conditions
def condition1 := t1 = 39
def condition2 := t2 = 18
def condition3 := length_platform = 350

-- The goal is to prove the length of the train
theorem length_of_train : condition1 ∧ condition2 ∧ condition3 → L = 300 :=
by
  intros h
  sorry

end length_of_train_l603_603881


namespace arithmetic_mean_median_l603_603377

theorem arithmetic_mean_median (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : a = 0) (h4 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end arithmetic_mean_median_l603_603377


namespace geometric_sequence_arithmetic_median_l603_603214

theorem geometric_sequence_arithmetic_median 
  (a : ℕ → ℝ) 
  (hpos : ∀ n, 0 < a n) 
  (h_arith : 2 * a 1 + a 2 = 2 * a 3) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 :=
sorry

end geometric_sequence_arithmetic_median_l603_603214


namespace ramesh_discount_correct_l603_603729

noncomputable def labelled_price : ℝ := 17250
def purchase_price : ℝ := 13500
def selling_price : ℝ := 18975
def desired_profit_rate : ℝ := 0.10
def discount_percentage := ((labelled_price - purchase_price) / labelled_price) * 100

theorem ramesh_discount_correct : discount_percentage ≈ 21.74 :=
by
  have h1 : 1.10 * labelled_price = selling_price :=
    sorry
  have h2 : labelled_price = 17250 :=
    sorry
  exacte sorry

end ramesh_discount_correct_l603_603729


namespace lines_intersect_hyperbola_exactly_once_l603_603474

theorem lines_intersect_hyperbola_exactly_once :
  let P := (0, 4)
  let ℋ : ℝ × ℝ → Prop := λ (x, y), y^2 - 4 * x^2 = 16
  ∃ L : ℝ → ℝ, (∀ x : ℝ, (x, L x) = P) ∧ (set_of (λ (x, y), ℋ (x, y)) ∩ set_of (λ (x: ℝ, (x, L x))) = 1) :=
begin
  sorry
end

end lines_intersect_hyperbola_exactly_once_l603_603474


namespace dalton_movies_l603_603518

variable (D : ℕ) -- Dalton's movies
variable (Hunter : ℕ := 12) -- Hunter's movies
variable (Alex : ℕ := 15) -- Alex's movies
variable (Together : ℕ := 2) -- Movies watched together
variable (TotalDifferentMovies : ℕ := 30) -- Total different movies

theorem dalton_movies (h : D + Hunter + Alex - Together * 3 = TotalDifferentMovies) : D = 9 := by
  sorry

end dalton_movies_l603_603518


namespace equivalenceOfPandQ_l603_603575

variable (A B C : ℝ)
variable (a b : ℝ)
variable (k : ℝ)
variable (ΔABC : A + B + C = π)  -- Triangle Angle Sum Property

-- Law of Sines
axiom lawOfSines : a / sin A = b / sin B
axiom sidesPos : a > 0 ∧ b > 0

-- Propositions
def p : Prop := sin A > sin B
def q : Prop := A > B

theorem equivalenceOfPandQ :
  (A > B) ↔ (sin A > sin B) := by
  sorry

end equivalenceOfPandQ_l603_603575


namespace gcd_euclidean_algorithm_l603_603724

theorem gcd_euclidean_algorithm (a b : ℕ) : 
  ∃ d : ℕ, d = gcd a b ∧ ∀ m : ℕ, (m ∣ a ∧ m ∣ b) → m ∣ d :=
by
  sorry

end gcd_euclidean_algorithm_l603_603724


namespace find_growth_rate_calculate_fourth_day_donation_l603_603420

-- Define the conditions
def first_day_donation : ℝ := 3000
def third_day_donation : ℝ := 4320
def growth_rate (x : ℝ) : Prop := (1 + x)^2 = third_day_donation / first_day_donation

-- Since the problem states growth rate for second and third day is the same,
-- we need to find that rate which is equivalent to solving the above proposition for x.

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.2 := by
  sorry

-- Calculate the fourth day's donation based on the growth rate found.
def fourth_day_donation (third_day : ℝ) (growth_rate : ℝ) : ℝ :=
  third_day * (1 + growth_rate)

theorem calculate_fourth_day_donation : 
  ∀ x : ℝ, growth_rate x → x = 0.2 → fourth_day_donation third_day_donation x = 5184 := by 
  sorry

end find_growth_rate_calculate_fourth_day_donation_l603_603420


namespace problem_l603_603773

/-- 
Let m, n be non-negative integers.
Given that 210 = (a_1! * a_2! * ... * a_m!) / (b_1! * b_2! * ... * b_n!), where 
a_1 ≥ a_2 ≥ ... ≥ a_m, b_1 ≥ b_2 ≥ ... ≥ b_n,
a_1 + b_1 is minimized and a_1 - b_1 is even,
then |a_1 - b_1| = 2
--/
theorem problem {m n : ℕ} {a b : ℕ → ℕ} (h1 : 210 = (∏ i in finset.range m, (a i)!) / (∏ j in finset.range n, (b j)!))
  (h2 : ∀ i, i < m - 1 → a i ≥ a (i + 1))
  (h3 : ∀ j, j < n - 1 → b j ≥ b (j + 1))
  (h4 : ∀ a1 b1, (a1 + b1) ≤ (a 0 + b 0) → (a0 - b0) % 2 = 0) :
  |(a 0) - (b 0)| = 2 :=
by sorry

end problem_l603_603773


namespace board_total_length_l603_603058

-- Definitions based on conditions
def S : ℝ := 2
def L : ℝ := 2 * S

-- Define the total length of the board
def T : ℝ := S + L

-- The theorem asserting the total length of the board is 6 ft
theorem board_total_length : T = 6 := 
by
  sorry

end board_total_length_l603_603058


namespace area_of_circle_above_line_l603_603422

theorem area_of_circle_above_line :
  ∃ c : ℝ × ℝ, ∃ r : ℝ, (∀ y : ℝ, y = -1 → (∃ x : ℝ, (x - 3)^2 + (y - 1)^2 = 8) → 
  (π * (2 * real.sqrt 2)^2) = 8 * π) :=
sorry

end area_of_circle_above_line_l603_603422


namespace equation_of_ellipse_maximum_area_triangle_AOB_l603_603222

noncomputable def eccentricity : ℝ := real.sqrt (6 / 3)

variable (a b : ℝ)
variable (h1 : a > b > 0)
variable (h2 : y = real.sqrt 3 * x - 2 * real.sqrt 3)
variable (h3 : ∃ f : ℝ × ℝ, f = (2, 0))
variable (ellipse_equation : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1)
variable (h4 : a = real.sqrt 6)

theorem equation_of_ellipse :
  (∃ a b : ℝ, a = real.sqrt 6 ∧ b^2 = a^2 - 4 ∧ h1 ∧ ellipse_equation (6, 2)) :=
begin
  sorry
end

variable (D : ℝ × ℝ)
variable (A B : ℝ × ℝ)
variable (triangle_area : ℝ)

theorem maximum_area_triangle_AOB :
  (∃ k : ℝ, ∀ x1 x2 : ℝ, x1 + x2 = -6 * k / (3 * k^2 + 1) ∧ x1 * x2 = 3 / (3 * k^2 + 1) →
  S_ABC = real.sqrt 3 * (real.sqrt ((6 * k^2 + 1) / (3 * k^2 + 1)^2)) ≤ real.sqrt 3) :=
begin
  sorry
end

end equation_of_ellipse_maximum_area_triangle_AOB_l603_603222


namespace correct_option_A_l603_603843

theorem correct_option_A : 
  (sqrt ((-4 : ℝ)^2) = 4) ∧ 
  (sqrt (-4) ≠ -2) ∧ 
  (sqrt 16 ≠ 4 ∨ sqrt 16 ≠ -4) ∧ 
  ( (sqrt 4 = 2 ∧ sqrt 4 = -2) -> ((sqrt 4 = 2) ∨ (sqrt 4 = -2)) ) :=
by
  sorry

end correct_option_A_l603_603843


namespace integer_values_satisfying_sqrt_inequality_l603_603812

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l603_603812


namespace range_of_p_l603_603596

theorem range_of_p (a b : ℝ) :
  (∀ x y p q : ℝ, p + q = 1 → (p * (x^2 + a * x + b) + q * (y^2 + a * y + b) ≥ ((p * x + q * y)^2 + a * (p * x + q * y) + b))) →
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 1) :=
sorry

end range_of_p_l603_603596


namespace Pascal_triangle_first_21_rows_l603_603614

theorem Pascal_triangle_first_21_rows :
  let total_numbers := (21 * 22) / 2
  in total_numbers = 231 := 
by
  let total_numbers := (21 * 22) / 2
  show total_numbers = 231,
  by
    sorry -- Proof goes here

end Pascal_triangle_first_21_rows_l603_603614


namespace intersection_points_l603_603928

noncomputable def y := (f : ℝ → ℝ)

variables (f : ℝ → ℝ)

-- Condition: y = f(x) is an even function on ℝ
def even_function : Prop := ∀ x, f (-x) = f x
-- Condition: when x > 0, y = f(x) is monotonically increasing
def monotonically_increasing : Prop := ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f x < f y
-- Condition: f(1) * f(2) < 0
def opposite_signs : Prop := f 1 * f 2 < 0

theorem intersection_points (h1 : even_function f) (h2 : monotonically_increasing f) (h3 : opposite_signs f) : 
  ∃ n : ℕ, n = 2 ∧ ∀ x, f x = 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 :=
sorry

end intersection_points_l603_603928


namespace determine_M_l603_603120

theorem determine_M : ∃ M : ℕ, 36^2 * 75^2 = 30^2 * M^2 ∧ M = 90 := 
by
  sorry

end determine_M_l603_603120


namespace distance_focus_asymptote_l603_603603

-- Definitions based on problem conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def eccentricity (a c : ℝ) : Prop := c^2 = a^2 + b^2
def axis_length (a : ℝ) : Prop := 2 * a = 2

-- Main theorem stating the proof problem
theorem distance_focus_asymptote 
  (a b c : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : hyperbola a b 1 0) 
  (h4 : eccentricity a c) 
  (h5 : axis_length a) : 
  (distance_focus_asymptote = 2) :=
sorry

end distance_focus_asymptote_l603_603603


namespace pies_eaten_by_ashley_l603_603115

theorem pies_eaten_by_ashley (daily_production : ℕ) (days : ℕ) (remaining_pies : ℕ) : daily_production = 7 → days = 12 → remaining_pies = 34 → (daily_production * days - remaining_pies = 50) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl  -- This would usually lead to the proof but we skip it


end pies_eaten_by_ashley_l603_603115


namespace arithmetic_mean_of_sides_l603_603929

noncomputable theory

open Classical

theorem arithmetic_mean_of_sides (r : ℝ) (a b c : ℝ) (h1 : a = r * (Real.sqrt 2)) (h2 : b = 2 * r) (h3 : c = r * (Real.sqrt (2 - Real.sqrt 2))) : 
  a = (b + c) / 2 :=
by
  sorry

end arithmetic_mean_of_sides_l603_603929


namespace smallest_B_for_divisibility_by_4_l603_603301

theorem smallest_B_for_divisibility_by_4 : 
  ∃ (B : ℕ), B < 10 ∧ (4 * 1000000 + B * 100000 + 80000 + 3961) % 4 = 0 ∧ ∀ (B' : ℕ), (B' < B ∧ B' < 10) → ¬ ((4 * 1000000 + B' * 100000 + 80000 + 3961) % 4 = 0) := 
sorry

end smallest_B_for_divisibility_by_4_l603_603301


namespace angle_F1PF2_l603_603212

open Real

theorem angle_F1PF2 (F1 F2 P : ℝ × ℝ) (hF1 : F1 = (-√2, 0)) (hF2 : F2 = (√2, 0))
  (hP : ∃ x y : ℝ, P = (x, y) ∧ x ^ 2 - y ^ 2 = 1)
  (hArea : ∃ x y : ℝ, P = (x, y) ∧ abs ((-√2) * y + √2 * y) = 2 * y = sqrt 3) :
  ∃ θ : ℝ, cos θ = 1 / 2 ∧ θ = π / 3 :=
begin
  sorry
end

end angle_F1PF2_l603_603212


namespace octal_to_binary_correct_l603_603927

-- Definition stating the given octal number
def octal_num : ℕ := 1 * 8^2 + 2 * 8^1 + 7 * 8^0

-- The equivalent decimal number of the given octal number
def decimal_num : ℕ := 87

-- The expected binary representation of the decimal number
def binary_rep : string := "1010111"

-- The main statement to be proven
theorem octal_to_binary_correct : 
  (nat.to_digits 2 decimal_num).as_string = binary_rep :=
by
  -- This is where the proof would go, but we are only stating the theorem here
  sorry

end octal_to_binary_correct_l603_603927


namespace find_a6_l603_603189

-- Define the geometric sequence conditions
noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the specific sequence with given initial conditions and sum of first three terms
theorem find_a6 : 
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (0 < q) ∧ (q ≠ 1) ∧ geom_seq a q ∧ 
    a 1 = 96 ∧ 
    (a 1 + a 2 + a 3 = 168) ∧
    a 6 = 3 := 
by
  sorry

end find_a6_l603_603189


namespace work_completion_days_l603_603848

theorem work_completion_days (D_a : ℝ) (R_a R_b : ℝ)
  (h1 : R_a = 1 / D_a)
  (h2 : R_b = 1 / (1.5 * D_a))
  (h3 : R_a = 1.5 * R_b)
  (h4 : 1 / 18 = R_a + R_b) : D_a = 30 := 
by
  sorry

end work_completion_days_l603_603848


namespace BM_bisects_AC_l603_603712

variables {A B C D K L M : Point}
variables {circle : Circle D}
variables {triangle : Triangle A B C}

-- Conditions provided by the problem
def altitude (BD : Line) (triangle : Triangle A B C) : Prop := 
  orthogonal BD (line_through A C)

def intersects_AB_BC (circle : Circle D) (AB BC : Line) (K L : Point) : Prop := 
  intersect circle AB = some K ∧ intersect circle BC = some L

def tangents_intersect (circle : Circle D) (K L : Point) (M : Point) : Prop :=
  tangent_at K circle ∩ tangent_at L circle = some M

def bisects (BM : Line) (AC : Segment) : Prop := 
  midpoint AC (intersection_point BM AC)

-- Theorem statement
theorem BM_bisects_AC
  (BD : Line) 
  (h_alt : altitude BD triangle)
  (h_circle : is_diameter BD circle)
  (h_inter_AB_BC : intersects_AB_BC circle (line_through A B) (line_through B C) K L)
  (h_tangent_intersect : tangents_intersect circle K L M) :
  bisects (line_through B M) (segment A C) :=
sorry

end BM_bisects_AC_l603_603712


namespace circumcenter_lies_on_reflected_line_l603_603329

-- Given a chordal quadrilateral ABCD with a circumcenter O
-- The diagonals AC and BD are perpendicular to each other
-- Let g be the reflection of the diagonal AC at the angle bisector of ∠BAD

noncomputable theory
open_locale classical

variables {A B C D O : Point}

-- Definition of a Point and Line in our geometric setup
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ) -- ax + by + c = 0

def is_circumcenter (O : Point) (A B C D : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def reflection (AC : Line) (BAD_bisector : Line) : Line := -- Placeholder_logic
  sorry

def lies_on (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- The actual theorem statement
theorem circumcenter_lies_on_reflected_line :
  chordal_quadrilateral ABCD → 
  is_circumcenter O A B C D →
  perpendicular (diagonal AC) (diagonal BD) →
  let g := reflection (diagonal AC) (angle_bisector BAD) in
  lies_on O g :=
begin
  sorry
end

end circumcenter_lies_on_reflected_line_l603_603329


namespace find_a_odd_function_l603_603335

variables (a x : ℝ)
def f (x : ℝ) : ℝ := Real.exp x + a * Real.exp x

theorem find_a_odd_function (h : ∀ x : ℝ, f a (-x) = - f a x) : a = -1 :=
by
  -- Given the precondition that f(x) is an odd function
  sorry

end find_a_odd_function_l603_603335


namespace complement_intersection_eq_l603_603338

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) 

-- Given conditions:
def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x : ℝ | x^2 - 3*x ≥ 0}
def set_B : Set ℝ := {x : ℝ | x ∈ ℤ ∧ x ≤ 3}

-- Mathematical goal:
theorem complement_intersection_eq : (compl set_A ∩ set_B) = {1, 2} :=
sorry

end complement_intersection_eq_l603_603338


namespace smallest_k_equals_26_l603_603522

open Real

-- Define the condition
def cos_squared_eq_one (θ : ℝ) : Prop :=
  cos θ ^ 2 = 1

-- Define the requirement for θ to be in the form 180°n
def theta_condition (n : ℤ) : Prop :=
  ∃ (k : ℤ), k ^ 2 + k + 81 = 180 * n

-- The problem statement in Lean: Find the smallest positive integer k such that
-- cos squared of (k^2 + k + 81) degrees = 1
noncomputable def smallest_k_satisfying_cos (k : ℤ) : Prop :=
  (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (k ^ 2 + k + 81)) ∧ (∀ m : ℤ, m > 0 ∧ m < k → 
   (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (m ^ 2 + m + 81)) → false)

theorem smallest_k_equals_26 : smallest_k_satisfying_cos 26 := 
  sorry

end smallest_k_equals_26_l603_603522


namespace calculate_expr_eq_two_l603_603105

def calculate_expr : ℕ :=
  3^(0^(2^8)) + (3^0^2)^8

theorem calculate_expr_eq_two : calculate_expr = 2 := 
by
  sorry

end calculate_expr_eq_two_l603_603105


namespace numeral_diff_local_face_value_l603_603428

theorem numeral_diff_local_face_value (P : ℕ) :
  7 * (10 ^ P - 1) = 693 → P = 2 ∧ (N = 700) :=
by
  intro h
  -- The actual proof is not required hence we insert sorry
  sorry

end numeral_diff_local_face_value_l603_603428


namespace expected_value_abs_diff_HT_l603_603342

noncomputable def expected_abs_diff_HT : ℚ :=
  let F : ℕ → ℚ := sorry -- Recurrence relation omitted for brevity
  F 0

theorem expected_value_abs_diff_HT :
  expected_abs_diff_HT = 24 / 7 :=
sorry

end expected_value_abs_diff_HT_l603_603342


namespace rahim_pillows_l603_603728

theorem rahim_pillows (x T : ℕ) (h1 : T = 5 * x) (h2 : (T + 10) / (x + 1) = 6) : x = 4 :=
by
  sorry

end rahim_pillows_l603_603728


namespace solve_for_x_l603_603249

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l603_603249


namespace bank_check_problem_l603_603072

theorem bank_check_problem :
  ∃ (x y : ℕ), (0 ≤ y ∧ y ≤ 99) ∧ (y + (x : ℚ) / 100 - 0.05 = 2 * (x + (y : ℚ) / 100)) ∧ x = 31 ∧ y = 63 :=
by
  -- Definitions and Conditions
  sorry

end bank_check_problem_l603_603072


namespace unit_digit_15_pow_100_l603_603431

theorem unit_digit_15_pow_100 : ((15^100) % 10) = 5 := 
by sorry

end unit_digit_15_pow_100_l603_603431


namespace arctan_sum_eq_pi_sub_arctan_l603_603511

noncomputable theory

open Real

theorem arctan_sum_eq_pi_sub_arctan :
  arctan (3 / 4) + 2 * arctan (4 / 3) = π - arctan (3 / 4) :=
sorry

end arctan_sum_eq_pi_sub_arctan_l603_603511


namespace parallel_lines_line_above_x_axis_l603_603558

-- Definitions for the lines l1 and l2 and the conditions
def l1 (k : ℝ) (x : ℝ) : ℝ := (k / 2) * x + 1
def l2 (k : ℝ) (x : ℝ) : ℝ := (1 / (k - 1)) * x - k

-- Problem (1): l1 parallel to l2 if and only if k = 2
theorem parallel_lines (k : ℝ) (h : k ≠ 1) : (∀ x, l1 k x = l2 k x) ↔ k = 2 := sorry

-- Problem (2): l1 always above the x-axis on the interval [-1, 2] if and only if -1 < k < 2
theorem line_above_x_axis (k : ℝ) (h : k ≠ 1) : 
  (∀ x ∈ set.Icc (-1 : ℝ) 2, l1 k x > 0) ↔ -1 < k ∧ k < 2 := sorry

end parallel_lines_line_above_x_axis_l603_603558


namespace find_positive_integer_l603_603849

theorem find_positive_integer 
  (n : ℕ) 
  (a : ℕ := (4 + 8 + 12 + 16 + 20 + 24 + 28) / 7)
  (b : ℕ := 2 * n)
  (h : a^2 - b^2 = 0) : 
  n = 8 := 
by
  have ha : a = 16 := by decide
  rw [← ha, h] at h
  rw [pow_two, pow_two] at h
  linarith

end find_positive_integer_l603_603849


namespace slope_angle_of_line_l603_603837

theorem slope_angle_of_line : 
  ∀ (α : ℝ), (0 ≤ α ∧ α < π) → (∀ m: ℝ, m = -1 → ∃ α : ℝ, tan α = m) → α = 3 * π / 4 := 
  by
  sorry

end slope_angle_of_line_l603_603837


namespace total_animals_safari_l603_603939

def animals_week1 :=
  [(5, 3), (2, 5), (5, 3), (7, 4), (6, 2), (3, 4), (8, 5)]

def animals_week2 :=
  [(6, 1), (3, 2, 4), (4, 5, 1), (11, 3, 7), (4, 7), (2, 5), (9, 7, 1)]

def daily_totals (week: List (List Nat)) : Nat :=
  week.map (λ day => day.sum).sum

theorem total_animals_safari : 
  daily_totals (animals_week1.map List.sum) + daily_totals (animals_week2.map List.sum) = 144 := 
  sorry

end total_animals_safari_l603_603939


namespace correlation_relationship_is_D_l603_603028

def relationship_A := (angle : ℝ) → Real.cos angle
def relationship_B := (side_length : ℝ) → side_length ^ 2
def relationship_C (V : ℝ) := (I : ℝ) → V / I
def relationship_D := (amount_sunlight : ℝ) → per_acre_yield_of_rice (amount_sunlight)

axiom per_acre_yield_of_rice : ℝ → ℝ

def is_correlation (relation : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → relation x ≠ relation y

theorem correlation_relationship_is_D:
  ¬ is_correlation relationship_A ∧ 
  ¬ is_correlation relationship_B ∧ 
  ¬ is_correlation (relationship_C 1) ∧  -- assuming a constant voltage of 1 for simplicity
  is_correlation relationship_D := 
sorry

end correlation_relationship_is_D_l603_603028


namespace vector_v_satisfies_conditions_l603_603324

open Matrix
open Matrix.SpecialLinearGroup

def vector_a : Vector 3 ℝ := ![2, 1, 1]
def vector_b : Vector 3 ℝ := ![3, -1, 0]
def vector_v : Vector 3 ℝ := ![5, 0, 1]

theorem vector_v_satisfies_conditions :
  (crossProduct vector_v vector_a = crossProduct vector_b vector_a) ∧
  (crossProduct vector_v vector_b = crossProduct vector_a vector_b) :=
by
  sorry

end vector_v_satisfies_conditions_l603_603324


namespace power_relationship_l603_603181

variable (a b : ℝ)

theorem power_relationship (h : 0 < a ∧ a < b ∧ b < 2) : a^b < b^a :=
sorry

end power_relationship_l603_603181


namespace algebra_expression_value_l603_603592

theorem algebra_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 11) : 3 * x^2 + 9 * x + 12 = 30 := 
by
  sorry

end algebra_expression_value_l603_603592


namespace find_sum_of_coefficients_l603_603573

-- Define the complex numbers
def z1 : Complex := (-1 : ℝ) + (2 : ℝ) * Complex.I
def z2 : Complex := (1 : ℝ) - Complex.I
def z3 : Complex := (3 : ℝ) - (2 : ℝ) * Complex.I

-- Define the points corresponding to the complex numbers
def A := (z1.re, z1.im)
def B := (z2.re, z2.im)
def C := (z3.re, z3.im)

-- Define the vectors from the origin to the points
def OA := (z1.re, z1.im)
def OB := (z2.re, z2.im)
def OC := (z3.re, z3.im)

-- State the condition given in the problem
def condition (x y : ℝ) : Prop :=
  OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2)

-- Theorem to prove
theorem find_sum_of_coefficients :
  ∃ x y : ℝ, condition x y ∧ x + y = 5 :=
begin
  -- Proof skipped
  sorry
end

end find_sum_of_coefficients_l603_603573


namespace integer_solutions_count_l603_603785

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l603_603785


namespace sin_integer_and_not_divisible_by_5_l603_603362

theorem sin_integer_and_not_divisible_by_5 (n : ℕ) (hα : sin α = 3 / 5)
  (hpos : n > 0) : ∃ k : ℤ, 5^n * sin (n * α) = k ∧ ¬ (5 ∣ 5^n * sin (n * α)) :=
by
  sorry

end sin_integer_and_not_divisible_by_5_l603_603362


namespace trapezoid_PR_length_l603_603415

noncomputable def PR_length (PQ RS QS PR : ℝ) (angle_QSP angle_SRP : ℝ) : Prop :=
  PQ < RS ∧ 
  QS = 2 ∧ 
  angle_QSP = 30 ∧ 
  angle_SRP = 60 ∧ 
  RS / PQ = 7 / 3 ∧ 
  PR = 8 / 3

theorem trapezoid_PR_length (PQ RS QS PR : ℝ) 
  (angle_QSP angle_SRP : ℝ) 
  (h1 : PQ < RS) 
  (h2 : QS = 2) 
  (h3 : angle_QSP = 30) 
  (h4 : angle_SRP = 60) 
  (h5 : RS / PQ = 7 / 3) :
  PR = 8 / 3 := 
by
  sorry

end trapezoid_PR_length_l603_603415


namespace number_of_solutions_l603_603616

def is_solution (x y z : ℕ) : Prop :=
  Nat.lcm x y = 72 ∧ Nat.lcm x z = 600 ∧ Nat.lcm y z = 900

theorem number_of_solutions :
  {n : ℕ // ∃ (triples : Finset (ℕ × ℕ × ℕ)), triples.filter (λ t, is_solution t.1 t.2.1 t.2.2) = n} = 15 :=
sorry

end number_of_solutions_l603_603616


namespace original_number_1496_or_2996_l603_603650

theorem original_number_1496_or_2996 (N : ℕ) : 
  (N >= 1000 ∧ N < 10000) ∧ 
  (∃ y, y >= 100 ∧ y < 1000 ∧ N = 1000 * (N / 1000) + y) ∧ 
  (∃ y, y = 1000 * (N / 1000) + N % 1000 - 3 * y = 8) →
  N = 1496 ∨ N = 2996 := 
sorry

end original_number_1496_or_2996_l603_603650


namespace chord_line_equation_l603_603220

theorem chord_line_equation 
  (x y : ℝ)
  (ellipse_eq : x^2 / 4 + y^2 / 3 = 1)
  (midpoint_condition : ∃ x1 y1 x2 y2 : ℝ, (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1
   ∧ (x1^2 / 4 + y1^2 / 3 = 1) ∧ (x2^2 / 4 + y2^2 / 3 = 1))
  : 3 * x - 4 * y + 7 = 0 :=
sorry

end chord_line_equation_l603_603220


namespace cos_alpha_value_cos_alpha_plus_half_beta_value_l603_603582

variable (α β : ℝ)

def main_problem :=
  0 < α ∧ α < π / 2 ∧
  -π / 2 < β ∧ β < 0 ∧
  cos (π / 4 + α) = 1 / 3 ∧
  cos (π / 4 - β / 2) = sqrt 3 / 3

theorem cos_alpha_value (h : main_problem α β) :
  cos α = (sqrt 2 + 4) / 6 := sorry

theorem cos_alpha_plus_half_beta_value (h : main_problem α β) :
  cos (α + β / 2) = 5 * sqrt 3 / 9 := sorry

end cos_alpha_value_cos_alpha_plus_half_beta_value_l603_603582


namespace kamals_salary_change_l603_603682

theorem kamals_salary_change : 
  ∀ (S : ℝ), ((S * 0.5 * 1.3 * 0.8 - S) / S) * 100 = -48 :=
by
  intro S
  sorry

end kamals_salary_change_l603_603682


namespace proposition2_correct_proposition6_correct_l603_603094

variable (a b c : ℝ)

-- Condition for proposition 2
def prop2_condition (a b : ℝ) (ab_zero : a * b = 0) : Bool :=
  abs (a + b) = abs (a - b)

-- Statement of proposition 2 proof 
theorem proposition2_correct (a b : ℝ) (ab_zero : a * b = 0) : 
  prop2_condition a b ab_zero = true := 
by
  sorry

-- Condition for proposition 6
def prop6_condition (A B : ℝ) : Bool :=
  (A > B) → (Real.sin A > Real.sin B)

-- Statement of proposition 6 proof
theorem proposition6_correct (A B : ℝ) (h : A > B) : 
  prop6_condition A B =
  true := by
  sorry

end proposition2_correct_proposition6_correct_l603_603094


namespace lakota_new_cds_l603_603684

theorem lakota_new_cds (U : ℝ) (N : ℝ) (L_new_cds : ℕ) (M_new_cds : ℕ) (L_total : ℝ) (M_total : ℝ) 
(H1 : U = 9.99) (H2 : L_total = N * L_new_cds + U * 2) (H3 : M_total = N * 3 + U * 8) 
(H4 : L_total = 127.92) (H5 : M_total = 133.89) : L_new_cds = 6 := 
by 
  sorry

end lakota_new_cds_l603_603684


namespace smallest_positive_solution_l603_603949

theorem smallest_positive_solution (x : ℝ) : (∃ x > 0, tan (4 * x) + tan (5 * x) = sec (5 * x)) → (∃ x > 0, x = 2 * pi / 13) :=
by {
  sorry
}

end smallest_positive_solution_l603_603949


namespace find_x_l603_603861

variable (x : ℝ)
variable (y : ℝ := x * 3.5)
variable (z : ℝ := y / 0.00002)

theorem find_x (h : z = 840) : x = 0.0048 :=
sorry

end find_x_l603_603861


namespace relationship_between_p_and_q_l603_603556

variable {a b : ℝ}

theorem relationship_between_p_and_q 
  (ha : a < 0) (hb : b < 0) : 
  let p := b^2 / a + a^2 / b
  let q := a + b
  in p ≤ q := 
sorry

end relationship_between_p_and_q_l603_603556


namespace potential_zero_of_polynomial_l603_603875

theorem potential_zero_of_polynomial :
  ∃ (p q : ℤ) (α β : ℤ), 
  let P := λ x : ℂ, (x - p) * (x - q) * (x^2 + α * x + β) in
  (∀ x : ℂ, P x = 0 → x ∈ {(p : ℂ), (q : ℂ), (↑(-α / 2) + (complex.I * (complex.sqrt (4 * β - α^2) / 2)) : ℂ), (↑(-α / 2) - (complex.I * (complex.sqrt (4 * β - α^2) / 2)) : ℂ)}) ∧
  (P (3/2 + complex.I * (complex.sqrt 15 / 2)) = 0) :=
begin
  sorry
end

end potential_zero_of_polynomial_l603_603875


namespace find_c_l603_603627

theorem find_c (a c : ℝ) :
  (∀ x : ℝ, x^2 + a * x + a / 4 + 1 / 2 > 0) →
  (∃! b : ℝ, (∀ x : ℝ, x^2 - x + b < 0)) →
  c = -2 :=
by
  sorry

end find_c_l603_603627


namespace integer_solutions_count_l603_603787

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l603_603787


namespace exponent_equation_solution_l603_603545

theorem exponent_equation_solution (x : ℝ) (h : 3^(4*x^2 - 7*x + 3) = 3^(4*x^2 + 9*x - 5)) : x = 1/2 :=
sorry

end exponent_equation_solution_l603_603545


namespace percent_other_birds_is_31_l603_603647

noncomputable def initial_hawk_percentage : ℝ := 0.30
noncomputable def initial_paddyfield_warbler_percentage : ℝ := 0.25
noncomputable def initial_kingfisher_percentage : ℝ := 0.10
noncomputable def initial_hp_k_total : ℝ := initial_hawk_percentage + initial_paddyfield_warbler_percentage + initial_kingfisher_percentage

noncomputable def migrated_hawk_percentage : ℝ := 0.8 * initial_hawk_percentage
noncomputable def migrated_kingfisher_percentage : ℝ := 2 * initial_kingfisher_percentage
noncomputable def migrated_hp_k_total : ℝ := migrated_hawk_percentage + initial_paddyfield_warbler_percentage + migrated_kingfisher_percentage

noncomputable def other_birds_percentage : ℝ := 1 - migrated_hp_k_total

theorem percent_other_birds_is_31 : other_birds_percentage = 0.31 := sorry

end percent_other_birds_is_31_l603_603647


namespace radius_of_incircle_l603_603904

theorem radius_of_incircle (a : ℝ) : 
  ∃ (r : ℝ), 
  (∀ (AB BC : ℝ) (angle_ABC : ℝ), 
  AB = a ∧ BC = a ∧ angle_ABC = 120 → 
  r = a * sqrt 3 * (2 - sqrt 3) / 2) :=
sorry

end radius_of_incircle_l603_603904


namespace vertex_on_x_axis_l603_603402

theorem vertex_on_x_axis (d : ℝ) :
  let y := (x : ℝ) ↦ x^2 - 6 * x + d in
  (∃ x : ℝ, y x = 0) ↔ d = 9 :=
by
  sorry

end vertex_on_x_axis_l603_603402


namespace geometric_sequence_problem_l603_603563

open Classical

noncomputable def a_n (n : ℕ) : ℤ

variables (a_5 : ℤ) (a_4 a_7 : ℤ)

def geometric_sequence : Prop :=
  a_5 = 3 ∧ a_4 * a_7 = 45

theorem geometric_sequence_problem (h : geometric_sequence a_5 a_4 a_7) :
  let a_6 := a_4 * a_7 / a_5 in
  let q := a_6 / a_5 in
  (a_7 - a_7 * q) = 25 :=
by
  sorry

end geometric_sequence_problem_l603_603563


namespace sequence_exists_l603_603117

theorem sequence_exists :
  ∃ a : ℕ → ℕ,
    (∀ n : ℕ, 0 < a n) ∧
    (∀ m : ℕ, ∃ n : ℕ, a n = m) ∧
    (∀ n : ℕ, ∃ k : ℕ, (∏ i in finset.range n.succ, a i) = k ^ (n + 1)) :=
sorry

end sequence_exists_l603_603117


namespace height_of_box_l603_603857

theorem height_of_box (r_large r_small : ℝ) (h : ℝ) (is_tangent : ∀ (i : fin 8), sphere_center i = box_corner i)
  (box_tangent_large : ∀ (i j k : fin 5), large_sphere_tangent i j k)
  (box_tangent_small : ∀ (i j k n : fin 5), small_sphere_tangent i j k n)
  (sphere_tangent : ∀ (i : fin 8), small_sphere_tangent_large i) :
  r_large = 3 ∧ r_small = 1 ∧ h = 13 :=
by
  sorry

end height_of_box_l603_603857


namespace boat_speed_l603_603446

def speed_of_boat_in_still_water (b s : ℝ) : Prop :=
  (b + s = 11) ∧ (b - s = 3) ∧ b = 7

theorem boat_speed:
  ∃ (b s : ℝ), speed_of_boat_in_still_water b s :=
by {
  use 7,
  use 4,
  split,
  { rw add_comm,
    exact eq.refl 11 },
  split,
  { exact eq.refl 3 },
  { exact eq.refl 7 }
}

end boat_speed_l603_603446


namespace correct_expression_l603_603025

-- Define the expressions
def exprA : Prop := ¬ (\((a + b) / c) = ((a + b) / c))
def exprB : Prop := ¬ (2 * (1 / 5) = 2 / 5)
def exprC : Prop := (n / m = n / m)
def exprD : Prop := ¬ (x * y * 2 = 2 * x * y)

-- Define the theorem stating the correct expression
theorem correct_expression : exprA ∧ exprB ∧ exprC ∧ exprD → exprC :=
by
  sorry

end correct_expression_l603_603025


namespace train_crosses_pole_in_9_seconds_l603_603492

noncomputable def time_to_cross_pole (speed_kmph : ℕ) (length_m : ℕ) : ℕ :=
  let speed_mps := speed_kmph * 1000 / 3600 in
  length_m / speed_mps

theorem train_crosses_pole_in_9_seconds :
  time_to_cross_pole 36 90 = 9 :=
by
  -- Let speed_kmph := 36
  -- Let length_m := 90
  -- Convert speed to mps: speed_mps = 36 * 1000 / 3600 = 10
  -- Calculate time: time = length_m / speed_mps = 90 / 10 = 9
  sorry

end train_crosses_pole_in_9_seconds_l603_603492


namespace subsets_S_union_T_l603_603276

theorem subsets_S_union_T (a : ℤ) (S T P : Set ℤ) (S_def : S = {3, a^2}) (T_def : T = {x | 0 < x + a ∧ x + a < 3 ∧ x ∈ ℤ}) :
  (S ∩ T = {1}) →
  (P = S ∪ T) →
  (a = 1 → (P = {0, 1, 3} ∧
    {∅, {0}, {1}, {3}, {0, 1}, {1, 3}, {3, 0}, {0, 1, 3}} ⊆ (Set.powerset P))) ∧
  (a = -1 → False) :=
begin
  sorry
end

end subsets_S_union_T_l603_603276


namespace sleep_hours_l603_603655

-- Define the times Isaac wakes up, goes to sleep, and takes naps
def monday : ℝ := 16 - 9
def tuesday_night : ℝ := 12 - 6.5
def tuesday_nap : ℝ := 1
def wednesday : ℝ := 9.75 - 7.75
def thursday_night : ℝ := 15.5 - 8
def thursday_nap : ℝ := 1.5
def friday : ℝ := 12 - 7.25
def saturday : ℝ := 12.75 - 9
def sunday_night : ℝ := 10.5 - 8.5
def sunday_nap : ℝ := 2

noncomputable def total_sleep : ℝ := 
  monday +
  (tuesday_night + tuesday_nap) +
  wednesday +
  (thursday_night + thursday_nap) +
  friday +
  saturday +
  (sunday_night + sunday_nap)

theorem sleep_hours (total_sleep : ℝ) : total_sleep = 36.75 := 
by
  -- Here, you would provide the steps used to add up the hours, but we will skip with sorry
  sorry

end sleep_hours_l603_603655


namespace total_tickets_sales_l603_603825

theorem total_tickets_sales:
    let student_ticket_price := 6
    let adult_ticket_price := 8
    let number_of_students := 20
    let number_of_adults := 12
    number_of_students * student_ticket_price + number_of_adults * adult_ticket_price = 216 :=
by
    intros
    sorry

end total_tickets_sales_l603_603825


namespace length_of_train_l603_603883

variables (L : ℝ) (t1 t2 : ℝ) (length_platform : ℝ)

-- Conditions
def condition1 := t1 = 39
def condition2 := t2 = 18
def condition3 := length_platform = 350

-- The goal is to prove the length of the train
theorem length_of_train : condition1 ∧ condition2 ∧ condition3 → L = 300 :=
by
  intros h
  sorry

end length_of_train_l603_603883


namespace equivalent_annual_rate_l603_603524

def quarterly_to_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4 - 1

def to_percentage (rate : ℝ) : ℝ :=
  rate * 100

theorem equivalent_annual_rate (quarterly_rate : ℝ) (annual_rate : ℝ) :
  quarterly_rate = 0.02 →
  annual_rate = quarterly_to_annual_rate quarterly_rate →
  to_percentage annual_rate = 8.24 :=
by
  intros
  sorry

end equivalent_annual_rate_l603_603524


namespace profit_percentage_example_l603_603442

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℕ) (sp_total : ℝ) (sp_count : ℕ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  (profit_per_article / cp_per_article) * 100

theorem profit_percentage_example : profit_percentage 25 15 33 12 = 65 :=
by
  sorry

end profit_percentage_example_l603_603442


namespace min_sum_distances_l603_603772

theorem min_sum_distances : 
  ∀ (x : ℝ), 
  (min_dist (x: ℝ, 0) (0, 2) (1, 1)) = sqrt (10)
    where
    min_dist (P: ℝ × ℝ) (A: ℝ × ℝ) (B: ℝ × ℝ): ℝ :=
        dist P A  + dist P B 
    dist (C: ℝ × ℝ) (D: ℝ × ℝ): ℝ :=
    sqrt ((D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2) מצ sorry

end min_sum_distances_l603_603772


namespace units_digit_p_plus_5_l603_603444

theorem units_digit_p_plus_5 (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 = 6) (h3 : (p^3 % 10) - (p^2 % 10) = 0) : (p + 5) % 10 = 1 :=
by
  sorry

end units_digit_p_plus_5_l603_603444


namespace minimum_M_grid_covering_l603_603969

def grid_covering_problem : Prop :=
  ∃ (M : ℕ), (∀ (m : ℕ), (m < M → ¬ ∀ (grid_rectangles : Fin m → Fin 2016 × Fin 2016 × Fin 2016 × Fin 2016),
  ∀ (i j : Fin 2017) (k : Fin 2016), ∃ (r : Fin m), 
  ((grid_rectangles r).fst₀ ≤ i ∧ i ≤ (grid_rectangles r).snd₀) ∧
  ((grid_rectangles r).fst₁ ≤ j ∧ j ≤ (grid_rectangles r).snd₁) ∧
  ((grid_rectangles r).fst₂ ≤ k ∧ k ≤ (grid_rectangles r).snd₂))) ∧
  (M = 2017))

theorem minimum_M_grid_covering : grid_covering_problem := by
  sorry

end minimum_M_grid_covering_l603_603969


namespace deriv_lt_2f_l603_603701

variable {R : Type} [LinearOrder R] [IsDifferentiable R] 
variable {f : R → R} (hf1 : ∀ x, 0 < f x) (hf2 : ∀ x, 0 < deriv f x)
          (hf3 : ∀ x, 0 < deriv (deriv f) x) (hf4 : ∀ x, 0 < deriv (deriv (deriv f)) x)
          (hf5 : ∀ x, f x ≥ deriv (deriv (deriv f)) x)

theorem deriv_lt_2f : ∀ x, deriv f x < 2 * f x :=
by sorry

end deriv_lt_2f_l603_603701


namespace chord_square_length_l603_603510

/-- Given three circles with radii 4, 8, and 16, such that the first two are externally tangent to each other and both are internally tangent to the third, if a chord in the circle with radius 16 is a common external tangent to the other two circles, then the square of the length of this chord is 7616/9. -/
theorem chord_square_length (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 8) (h3 : r3 = 16)
  (tangent_condition : ∀ (O4 O8 O16 : ℝ), O4 = r1 + r2 ∧ O8 = r2 + r3 ∧ O16 = r1 + r3) :
  (16^2 - (20/3)^2) * 4 = 7616 / 9 :=
by
  sorry

end chord_square_length_l603_603510


namespace base_length_of_parallelogram_l603_603564

theorem base_length_of_parallelogram (area : ℝ) (b h : ℝ) (angle : ℝ)
  (h1 : area = 162)
  (h2 : h = 2 * b)
  (h3 : angle = real.pi / 3) : b = 9 :=
by
  sorry

end base_length_of_parallelogram_l603_603564


namespace box_with_20_aluminium_80_plastic_weighs_494_l603_603859

def weight_of_box_with_100_aluminium_balls := 510 -- in grams
def weight_of_box_with_100_plastic_balls := 490 -- in grams
def number_of_aluminium_balls := 100
def number_of_plastic_balls := 100

-- Define the weights per ball type by subtracting the weight of the box
def weight_per_aluminium_ball := (weight_of_box_with_100_aluminium_balls - weight_of_box_with_100_plastic_balls) / number_of_aluminium_balls
def weight_per_plastic_ball := (weight_of_box_with_100_plastic_balls - weight_of_box_with_100_plastic_balls) / number_of_plastic_balls

-- Condition: The weight of the box alone (since it's present in both conditions)
def weight_of_empty_box := weight_of_box_with_100_plastic_balls - (weight_per_plastic_ball * number_of_plastic_balls)

-- Function to compute weight of the box with given number of aluminium and plastic balls
def total_weight (num_al : ℕ) (num_pl : ℕ) : ℕ :=
  weight_of_empty_box + (weight_per_aluminium_ball * num_al) + (weight_per_plastic_ball * num_pl)

-- The theorem to be proven
theorem box_with_20_aluminium_80_plastic_weighs_494 :
  total_weight 20 80 = 494 := sorry

end box_with_20_aluminium_80_plastic_weighs_494_l603_603859


namespace smallest_M_l603_603144

theorem smallest_M (M : ℕ) :
  (M > 0) ∧ ((∃ k, M = 4 * k) ∨ (∃ k, M + 1 = 4 * k) ∨ (∃ k, M + 2 = 4 * k)) ∧
  ((∃ k, M = 9 * k) ∨ (∃ k, M + 1 = 9 * k) ∨ (∃ k, M + 2 = 9 * k)) ∧
  ((∃ k, M = 49 * k) ∨ (∃ k, M + 1 = 49 * k) ∨ (∃ k, M + 2 = 49 * k)) ∧
  ((∃ k, M = 121 * k) ∨ (∃ k, M + 1 = 121 * k) ∨ (∃ k, M + 2 = 121 * k)) →
  M = 362 :=
begin
  sorry
end

end smallest_M_l603_603144


namespace stamp_arrangements_equals_76_l603_603124

-- Define the conditions of the problem
def stamps_available : List (ℕ × ℕ) := 
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), 
   (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), 
   (17, 17), (18, 18), (19, 19)]

-- Define a function to compute the number of different arrangements
noncomputable def count_stamp_arrangements : ℕ :=
  -- This is a placeholder for the actual implementation
  sorry

-- State the theorem to be proven
theorem stamp_arrangements_equals_76 : count_stamp_arrangements = 76 :=
sorry

end stamp_arrangements_equals_76_l603_603124


namespace polynomial_solutions_l603_603535

theorem polynomial_solutions :
  (∀ x : ℂ, (x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = 0) ↔ (x = -1 ∨ x = Complex.I ∨ x = -Complex.I)) :=
by
  sorry

end polynomial_solutions_l603_603535


namespace sum_of_digits_base2_300_l603_603007

theorem sum_of_digits_base2_300 : 
  let n := 300
  in (Int.digits 2 n).sum = 4 :=
by
  let n := 300
  have h : Int.digits 2 n = [1, 0, 0, 1, 0, 1, 1, 0, 0] := by sorry
  rw h
  norm_num
  -- or directly
  -- exact rfl

end sum_of_digits_base2_300_l603_603007


namespace length_of_train_l603_603886

theorem length_of_train
  (T_platform : ℕ)
  (T_pole : ℕ)
  (L_platform : ℕ)
  (h1: T_platform = 39)
  (h2: T_pole = 18)
  (h3: L_platform = 350)
  (L : ℕ)
  (h4 : 39 * L = 18 * (L + 350)) :
  L = 300 :=
by
  sorry

end length_of_train_l603_603886


namespace time_to_fill_tank_correct_l603_603718

noncomputable def time_to_fill_tank (tank_capacity : ℕ) (rate_a rate_b rate_c : ℕ) (rate_d_initial rate_d_increment : ℕ) (cycle_durations : list ℕ) : ℕ :=
  let fill_rate_per_cycle (cycle : ℕ) : ℕ :=
    rate_a - rate_c + rate_b * 2 + (rate_d_initial + (cycle / 2) * rate_d_increment)
  let net_fill_in_cycle (cycle : ℕ) : ℕ :=
    max 0 (fill_rate_per_cycle cycle) * cycle_durations.sum
  let fill_tank (acc_water : ℕ) (cycle : ℕ) : ℕ :=
    if acc_water >= tank_capacity then 0 else cycle_durations.sum + fill_tank (acc_water + net_fill_in_cycle cycle) (cycle + 1)
  fill_tank 0 1

theorem time_to_fill_tank_correct : time_to_fill_tank 1000 200 50 25 75 25 [1, 2, 2, 1] = 18 :=
by sorry

end time_to_fill_tank_correct_l603_603718


namespace son_work_rate_l603_603475

-- Definitions for the conditions
def man_work_rate : ℝ := 1 / 5
def combined_work_rate : ℝ := 1 / 3

-- The goal is to prove that the son can do the work alone in 7.5 days
theorem son_work_rate :
  (1 / (combined_work_rate - man_work_rate)) = 7.5 :=
by
  -- Normal Lean practice would include details in the proof here,
  -- but we are instructed to use 'sorry' to omit the proof steps.
  sorry

end son_work_rate_l603_603475


namespace small_n_non_isosceles_triangle_l603_603539

noncomputable def no_three_collinear {n : ℕ} (points : Finₙ → ℝ × ℝ) : Prop :=
  ∀ (i j k : Finₙ), i ≠ j → j ≠ k → i ≠ k → ¬ collinear (points i) (points j) (points k)

noncomputable def is_isosceles {n : ℕ} (triangle : Finₙ → ℝ × ℝ) : Prop :=
  let a := dist (triangle 0) (triangle 1)
  let b := dist (triangle 1) (triangle 2)
  let c := dist (triangle 2) (triangle 0)
  a = b ∨ b = c ∨ a = c

theorem small_n_non_isosceles_triangle :
  ∃ n : ℕ, n ≥ 3 ∧ (∀ (points : Finₙ → ℝ × ℝ), no_three_collinear points →
  ∃ (i j k : Finₙ), ¬ is_isosceles (λ m => points (Finₙ.fintype.equiv m))) → n = 7 :=
sorry

end small_n_non_isosceles_triangle_l603_603539


namespace goods_train_length_l603_603872

-- Conditions
def train1_speed := 60 -- kmph
def train2_speed := 52 -- kmph
def passing_time := 9 -- seconds

-- Conversion factor from kmph to meters per second
def kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps := kmph_to_mps (train1_speed + train2_speed)

-- Final theorem statement
theorem goods_train_length :
  relative_speed_mps * passing_time = 280 :=
sorry

end goods_train_length_l603_603872


namespace p_is_necessary_but_not_sufficient_for_q_l603_603204

-- Conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0
def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- Proof target
theorem p_is_necessary_but_not_sufficient_for_q : 
  (∀ a : ℝ, p a → q a) ∧ ¬(∀ a : ℝ, q a → p a) :=
sorry

end p_is_necessary_but_not_sufficient_for_q_l603_603204


namespace find_n_l603_603407

def calls_between_any_group (n : ℕ) (λ : ℕ → ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ (s : Finset ℕ), s.card = n - 2 → (∑ t in s, ∑ u in s, λ t u) = 3^k

theorem find_n (n : ℕ) (λ : ℕ → ℕ → ℕ) (k : ℕ) :
  (∀ i j : ℕ, λ i j ≤ 1) →
  calls_between_any_group n λ k →
  n = 5 := 
sorry

end find_n_l603_603407


namespace can_catch_up_and_max_boat_speed_l603_603878
  
-- Define the given data as constants
def boat_speed_initial : ℝ := 2.5
def angle_with_shore : ℝ := 15
def person_run_speed : ℝ := 4
def person_swim_speed : ℝ := 2
def max_boat_speed : ℝ := 2 * Real.sqrt 2

-- State the theorem
theorem can_catch_up_and_max_boat_speed (v : ℝ) (t : ℝ) (k : ℝ) 
  (h_angle : angle_with_shore = 15) 
  (h_run_speed : person_run_speed = 4)
  (h_swim_speed : person_swim_speed = 2) :
  
  (v = boat_speed_initial → 
    ∃ t k, let distance_shore := 4 * k * t,
               distance_water := 2 * (1 - k) * t,
               distance_boat := v * t in 
    distance_boat^2 = distance_shore^2 + distance_water^2 - 
                      2 * distance_shore * distance_water * Real.cos(angle_with_shore * Real.pi / 180) ∧ 
    0 ≤ k ∧ k < 1) ∧
  
  (v = max_boat_speed →
    ∃ t k, let distance_shore := 4 * k * t,
               distance_water := 2 * (1 - k) * t,
               distance_boat := v * t in 
    distance_boat^2 = distance_shore^2 + distance_water^2 - 
                      2 * distance_shore * distance_water * Real.cos(angle_with_shore * Real.pi / 180) ∧ 
    0 ≤ k ∧ k < 1) :=

sorry

end can_catch_up_and_max_boat_speed_l603_603878


namespace count_integer_values_satisfying_condition_l603_603804

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l603_603804


namespace fraction_of_geese_not_survived_l603_603711

-- Defining the conditions
def total_eggs : ℕ := 320 -- We use this from the solution
def hatched_geese := (1/2 : ℝ) * total_eggs
def survived_first_month := (3/4 : ℝ) * hatched_geese
def survived_first_year : ℝ := 120
def no_more_than_one_goose_per_egg : Prop := true

-- Defining the fraction of the geese that survived the first month but did not survive the first year
def fraction_not_survived_first_year := (survived_first_month - survived_first_year) / survived_first_month

-- The proof statement
theorem fraction_of_geese_not_survived (h : no_more_than_one_goose_per_egg) :
  fraction_not_survived_first_year = 0 :=
by {
  -- Since (survived_first_month - survived_first_year) = 0 under the given conditions,
  -- The fraction_not_survived_first_year is 0.
  sorry
}

end fraction_of_geese_not_survived_l603_603711


namespace April_production_l603_603487

theorem April_production (P_Jan : ℝ) (r : ℝ) (n : ℕ) (P_Apr : ℝ) :
  P_Jan = 800000 →
  r = 0.05 →
  n = 3 →
  P_Apr = P_Jan * (1 + r)^n →
  P_Apr = 926100 :=
begin
  intros hPJan hr hn hPApr,
  rw [hPJan, hr, hn] at hPApr,
  exact hPApr,
end

end April_production_l603_603487


namespace part_a_part_b_l603_603687

-- Definition of the set C
def C (n : ℕ) : Prop := ∃ s t : ℕ, n = 1999 * s + 2000 * t

-- Part (a): Show that 3,994,001 is not in C
theorem part_a : ¬ C 3_994_001 :=
sorry

-- Part (b): Show that if 0 ≤ n ≤ 3,994,001 and n is not in C, then 3,994,001 - n is in C
theorem part_b (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 3_994_001) (hnotin : ¬ C n) : C (3_994_001 - n) :=
sorry

end part_a_part_b_l603_603687


namespace sum_of_three_consecutive_eq_product_of_distinct_l603_603574

theorem sum_of_three_consecutive_eq_product_of_distinct (n : ℕ) (h : 100 < n) :
  ∃ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧
  ((n + (n+1) + (n+2) = a * b * c) ∨
   ((n+1) + (n+2) + (n+3) = a * b * c) ∨
   (n + (n+1) + (n+3) = a * b * c) ∨
   (n + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end sum_of_three_consecutive_eq_product_of_distinct_l603_603574


namespace evalA_eq_1_div_cbrt_y_l603_603033

noncomputable def evalA (x y : ℝ) : ℝ :=
  ( (real.cbrt (8 * x - y - 6 * (2 * real.cbrt (x^2 * y) - real.cbrt (x * y^2))))
  * (4 * real.cbrt (x^2) + 2 * real.cbrt (x * y) + real.cbrt (y^2)) )
  / (8 * x * real.cbrt y - real.cbrt (y^4))

theorem evalA_eq_1_div_cbrt_y 
  {x y : ℝ} (h₁ : y ≠ 0) (h₂ : y ≠ (8 * x)) : evalA x y = 1 / real.cbrt y := 
sorry

end evalA_eq_1_div_cbrt_y_l603_603033


namespace compute_expression_at_4_l603_603110

theorem compute_expression_at_4 (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end compute_expression_at_4_l603_603110


namespace a44_is_14_l603_603054

def isMagicSquare (M : Matrix (Fin 4) (Fin 4) ℕ) (s : ℕ) : Prop :=
  (∀ i, (Σ j, M i j) = s) ∧              -- All rows have the same sum
  (∀ j, (Σ i, M i j) = s) ∧              -- All columns have the same sum
  ((Σ i, M i i) = s) ∧                   -- Main diagonal has the same sum
  ((Σ i, M i (3 - i)) = s) ∧             -- Anti-diagonal has the same sum
  ((∃ unique n, Matrix.finSum (λ _ _, n) M = 136) ∧ -- Sum of all numbers is 136
  ∀ i j, 1 ≤ M i j ∧ M i j ≤ 16)          -- All numbers are from 1 to 16 

def magic_square_conditions (M : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  M 1 1 = 1 ∧ M 1 3 = 2                   -- Specific given values

theorem a44_is_14 (M : Matrix (Fin 4) (Fin 4) ℕ) : 
  isMagicSquare M 34 ∧ magic_square_conditions M → M 3 3 = 14 := 
by 
  sorry

end a44_is_14_l603_603054


namespace analytical_expression_range_of_a_l603_603225

noncomputable theory

-- Define the function with constraints
def f (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)
variables (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : |φ| < Real.pi / 2)
-- Given points A and B
variables (hA_point : f (Real.pi / 3) = 2) (hB_point : f (-Real.pi / 6) = -2)
-- Given symmetry of the graph
variable (h_symmetry : ∀ x, f (Real.pi / 3 - x) = f (Real.pi / 3 + x))
-- Prove analytical expression
theorem analytical_expression : f = λ x, 2 * Real.sin (2 * x - Real.pi / 6) :=
  sorry

-- Prove the range for a
theorem range_of_a : ∀ x ∈ set.Icc (3 * Real.pi / 4) (7 * Real.pi / 6), 
  f x ≤ 2 * a + 3 → a ∈ set.Ici (-5 / 2) :=
  sorry

end analytical_expression_range_of_a_l603_603225


namespace rectangle_lines_combinations_l603_603165

theorem rectangle_lines_combinations : 5.choose 2 * 4.choose 2 = 60 := by
  sorry

end rectangle_lines_combinations_l603_603165


namespace difference_of_squares_l603_603399

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 20) : a^2 - b^2 = 1200 := 
sorry

end difference_of_squares_l603_603399


namespace problem_solution_l603_603416

def triangle (A B C : Type) := ∃ AB BC CA : ℝ, AB = 6 ∧ BC = 8 ∧ CA = 10
def circle_tangent_at (ω : Type) (P : Type) (A : Type) := True

def problem_statement : Prop :=
  ∀ (A B C : Type) (ω1 ω2 : Type) (K : Type),
  triangle A B C →
  circle_tangent_at ω1 B A →
  circle_tangent_at ω2 C A →
  K ≠ A →
  let AK := (3.75 : ℝ) in
  AK = 3.75

theorem problem_solution : problem_statement := by
  sorry

end problem_solution_l603_603416


namespace sum_of_digits_base_2_of_300_l603_603013

theorem sum_of_digits_base_2_of_300 : 
  let n := 300
  let binary_representation := nat.binary_repr n
  nat.digits_sum 2 binary_representation = 4 :=
by
  let n := 300
  let binary_representation := nat.binary_repr n
  have h1 : binary_representation = [1,0,0,1,0,1,1,0,0] := sorry
  have h2 : nat.digits_sum 2 binary_representation = 1+0+0+1+0+1+1+0+0 := sorry
  show nat.digits_sum 2 binary_representation = 4 from by sorry

end sum_of_digits_base_2_of_300_l603_603013


namespace dealer_gain_percent_is_100_l603_603867

variable (L : ℝ)

def purchase_price (L : ℝ) := (3/4) * L
def selling_price (L : ℝ) := (3/2) * L
def gain (L : ℝ) := selling_price L - purchase_price L
def gain_percent (L : ℝ) := (gain L / purchase_price L) * 100

theorem dealer_gain_percent_is_100 : gain_percent L = 100 := by
  sorry

end dealer_gain_percent_is_100_l603_603867


namespace smaller_cube_count_l603_603466

-- Cube problem where main cube edge length is 5 cm and smaller cubes are of integer lengths.
theorem smaller_cube_count
  (edge_length_main_cube : ℕ) -- Edge length of the main cube
  (edge_length_main_cube = 5)
  (smaller_cube_lengths : list ℕ) -- List of smaller cube edge lengths
  (∀ x ∈ smaller_cube_lengths, x ∈ [1, 2, 3, 4]) -- Smaller cube lengths are 1, 2, 3, or 4
  : ∃ N, N = 25 := -- Resulting number of smaller cubes should be 25
sorry

end smaller_cube_count_l603_603466


namespace squares_area_ratio_l603_603777

theorem squares_area_ratio (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 8) :
  a + b + c = 16 :=
by
  rw [h₁, h₂, h₃]
  rw [add_assoc]
  rfl

end squares_area_ratio_l603_603777


namespace number_of_rectangles_l603_603149

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 4) : 
  (nat.choose H 2) * (nat.choose V 2) = 60 := by
  rw [hH, hV]
  norm_num

end number_of_rectangles_l603_603149


namespace expression_undefined_at_2_l603_603667

theorem expression_undefined_at_2 :
  ∀ x : ℝ, x = 2 → (let y := (x + 2) / (x - 2) in (y + 2) / (y - 2)) = 0 :=
by
  intro x hx
  rw hx
  have h : (2 + 2) / (2 - 2) = 0 := by sorry
  rw h
  sorry

end expression_undefined_at_2_l603_603667


namespace num_int_values_x_l603_603790

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l603_603790


namespace find_prices_possible_purchasing_schemes_maximize_profit_l603_603464

namespace PurchasePriceProblem

/-- 
Define the purchase price per unit for bean sprouts and dried tofu,
and show that they satisfy the given conditions. 
--/
theorem find_prices (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 240)
  (h2 : 3 * x + 4 * y = 340) :
  x = 60 ∧ y = 40 := 
by sorry

/-- 
Given the conditions on the purchase price of bean sprouts and dried tofu
and the need of purchasing a total of 200 units for no more than $10440, 
determine the valid purchasing schemes.
--/
theorem possible_purchasing_schemes 
  (a : ℤ)
  (h1 : 60 * a + 40 * (200 - a) ≤ 10440)
  (h2 : a ≥ 3 / 2 * (200 - a)) :
  120 ≤ a ∧ a ≤ 122 := 
by sorry
  
/-- 
Maximize profit based on the purchasing schemes that satisfy the conditions.
--/
theorem maximize_profit 
  (a : ℤ) 
  (h_valid : 120 ≤ a ∧ a ≤ 122) 
  (h_max : ∀ b, 120 ≤ b ∧ b ≤ 122 → 5 * a + 3000 ≥ 5 * b + 3000) :
  (a = 122) → 
  let beans_profit := 5 * a + 3000 
  in beans_profit = 3610 := 
by sorry

end PurchasePriceProblem

end find_prices_possible_purchasing_schemes_maximize_profit_l603_603464


namespace location_of_z_l603_603997

def z : ℂ := (i^2016) / (3 + 2 * i)

theorem location_of_z : z.re > 0 ∧ z.im < 0 := sorry

end location_of_z_l603_603997


namespace digits_difference_l603_603648

def count_digit (digit num : ℕ) : ℕ :=
  (num.digits 10).count digit

def total_count_digit (digit : ℕ) (n : ℕ) : ℕ :=
  (list.range (n + 1)).sum (count_digit digit)

theorem digits_difference:
  total_count_digit 3 625 - total_count_digit 7 625 = 100 := by
  sorry

end digits_difference_l603_603648


namespace min_balls_to_guarantee_20_l603_603860

theorem min_balls_to_guarantee_20 (red green yellow blue white black : ℕ) :
  red = 36 → green = 24 → yellow = 18 → blue = 15 → white = 12 → black = 10 →
  ∃ (n : ℕ), n = 94 ∧ ∀ (draws : ℕ), draws ≥ n →
  (∃ (r g y b w bl : ℕ), r + g + y + b + w + bl = draws ∧ 
  (r = 20 ∨ g = 20 ∨ y = 20 ∨ b = 20 ∨ w = 20 ∨ bl = 20)) :=
begin
  intros h_red h_green h_yellow h_blue h_white h_black,
  use 94,
  split,
  { refl },
  { intros draws h_draws,
    sorry -- Proof will be filled here
  }
end

end min_balls_to_guarantee_20_l603_603860


namespace river_width_after_30_seconds_l603_603359

noncomputable def width_of_river (initial_width : ℝ) (width_increase_rate : ℝ) (rowing_rate : ℝ) (time_taken : ℝ) : ℝ :=
  initial_width + (time_taken * rowing_rate * (width_increase_rate / 10))

theorem river_width_after_30_seconds :
  width_of_river 50 2 5 30 = 80 :=
by
  -- it suffices to check the calculations here
  sorry

end river_width_after_30_seconds_l603_603359


namespace cheese_balance_exists_l603_603408

-- Definitions: weights of the cheeses and their distinct property.
variables (a b c d e f : ℝ)
variables (h_abc : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f)

-- Main proof problem statement.
theorem cheese_balance_exists (h_total_weight : a + b + c + d + e + f = 2) :
  ∃ (S1 S2 : set ℝ), 
    S1.card = 3 ∧ S2.card = 3 ∧ S1 ∪ S2 = {a, b, c, d, e, f} ∧ S1 ∩ S2 = ∅ ∧
    (∑ x in S1, x = 1 ∧ ∑ x in S2, x = 1) := 
sorry

end cheese_balance_exists_l603_603408


namespace monotonic_intervals_max_k_l603_603971

def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 2
def f' (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

theorem monotonic_intervals 
  (a : ℝ) :
  (if a ≤ 0 then ∀ x₁ x₂, x₁ < x₂ → f x₁ a ≤ f x₂ a else
  ∀ x₁ x₂, (x₁ < x₂ ∧ x₁ < Real.log a ∧ x₂ < Real.log a → f x₁ a ≥ f x₂ a) ∧ 
           (x₁ < x₂ ∧ x₁ > Real.log a ∧ x₂ > Real.log a → f x₁ a ≤ f x₂ a))
:= sorry

theorem max_k 
  (k : ℤ) (a : ℝ) (h₁ : a = 1) (h₂ : ∀ x > 0, (k - x) / (x + 1) * f' x a < 1) : 
  k ≤ 2 :=
begin
  have f'_1 : ∀ x > 0, f' x 1 = Real.exp x - 1, { 
    intros x hx, 
    exact rfl 
  },
  sorry
end

end monotonic_intervals_max_k_l603_603971


namespace inconsistent_b_positive_l603_603589

theorem inconsistent_b_positive
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 / 2 → ax^2 + bx + c > 0) :
  ¬ b > 0 :=
sorry

end inconsistent_b_positive_l603_603589


namespace find_x_l603_603697

theorem find_x (a b x : ℝ) (hb : b ≠ 0) : 
  let r := (3 * a)^(3 * b) in
  r = a^b * x^(3 * b) →
  x = 3 * a^(2 / 3) :=
by
  intros r hr
  sorry

end find_x_l603_603697


namespace cos_2a_2b_2c_sum_l603_603332

theorem cos_2a_2b_2c_sum (a b c : ℝ) (h1 : cos a + cos b + cos c = 1) (h2 : sin a + sin b + sin c = 1) :
  cos (2 * a) + cos (2 * b) + cos (2 * c) = -2 :=
sorry

end cos_2a_2b_2c_sum_l603_603332


namespace expected_value_two_point_distribution_l603_603336

theorem expected_value_two_point_distribution (X : Type) [Fintype X] 
  (p0 p1 : ℝ) (h0 : p0 + p1 = 1) (h1 : p1 - p0 = 0.4) : 
  ∑ x in ({1, 0} : Finset X), (if x = 1 then p1 else p0) * (x : ℝ) = 0.7 :=
by
  sorry

end expected_value_two_point_distribution_l603_603336


namespace percentage_fewer_than_50000_l603_603385

def percentage_lt_20000 : ℝ := 35
def percentage_20000_to_49999 : ℝ := 45
def percentage_lt_50000 : ℝ := 80

theorem percentage_fewer_than_50000 :
  percentage_lt_20000 + percentage_20000_to_49999 = percentage_lt_50000 := 
by
  sorry

end percentage_fewer_than_50000_l603_603385


namespace cost_of_480_chocolates_l603_603059

theorem cost_of_480_chocolates :
  (40 * 8) = 320 :=
begin
 sorry
end

end cost_of_480_chocolates_l603_603059


namespace count_ordered_triples_l603_603617

theorem count_ordered_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : Nat.lcm x y = 72) (h2 : Nat.lcm x z = 600) (h3 : Nat.lcm y z = 900) :
  (Finset.univ.filter (λ t : ℕ × ℕ × ℕ, 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 
    Nat.lcm t.1 t.2.1 = 72 ∧ 
    Nat.lcm t.1 t.2.2 = 600 ∧ 
    Nat.lcm t.2.1 t.2.2 = 900)).card = 15 := 
sorry

end count_ordered_triples_l603_603617


namespace find_a3_l603_603581

noncomputable def a_coeffs (n : ℕ) (f : ℕ → ℕ) : ℕ → ℕ
| 0 := 1
| (k + 1) := a_coeffs k f * (n - k) * f k / (k + 1)

noncomputable def binomial_expansion (a : ℕ) (x : ℕ) (n : ℕ) : ℕ → ℕ
| 0 := (a : ℕ) ^ n
| (k + 1) := binomial_expansion k + (n.choose (k + 1)) * (a ^ (n - k - 1)) * (x ^ (k + 1))

theorem find_a3  : let a := 2 in let x := 2 in let n := 5 in let k := 3 in
  (binomial_expansion a x n) k = 80 :=
by
  let a := 2
  let x := 2
  let n := 5
  let k := 3
  sorry

end find_a3_l603_603581


namespace line_length_400_l603_603484

noncomputable def length_of_line (speed_march_kmh speed_run_kmh total_time_min: ℝ) : ℝ :=
  let speed_march_mpm := (speed_march_kmh * 1000) / 60
  let speed_run_mpm := (speed_run_kmh * 1000) / 60
  let len_eq := 1 / (speed_run_mpm - speed_march_mpm) + 1 / (speed_run_mpm + speed_march_mpm)
  (total_time_min * 200 * len_eq) * 400 / len_eq

theorem line_length_400 :
  length_of_line 8 12 7.2 = 400 := by
  sorry

end line_length_400_l603_603484


namespace gcf_lcm_statement_l603_603700

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_statement :
  gcd (lcm 18 21) (lcm 9 14) = 126 :=
by
  sorry

end gcf_lcm_statement_l603_603700


namespace red_ball_higher_probability_l603_603877

theorem red_ball_higher_probability : 
  let prob_red := λ (k : ℕ), 3 ^ (- (k + 1))
  let prob_green := λ (k : ℕ), 2 ^ (- (k + 1))
  (∃ (k i : ℕ), k > i ∧ prob_red k * prob_green i) = 1 / 5 := sorry

end red_ball_higher_probability_l603_603877


namespace minimize_deviation_chebyshev_polynomial_l603_603856

noncomputable def chebyshev_polynomial (n : ℕ) : ℝ → ℝ :=
  λ x, Real.cos (n * Real.arccos x)

noncomputable def transformed_chebyshev_polynomial (n : ℕ) : ℝ → ℝ :=
  λ x, 2 * chebyshev_polynomial n (x / 2)

def deviation_from_zero_on_interval (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  max (Real.abs (f a)) (Real.abs (f b))

theorem minimize_deviation_chebyshev_polynomial (n : ℕ) :
  ∀ P : ℝ → ℝ, (∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, P x = x^n * a) →
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → Real.abs (P x) ≥ 2) :=
  sorry

end minimize_deviation_chebyshev_polynomial_l603_603856


namespace max_x_condition_l603_603118

theorem max_x_condition (a b c x : ℤ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  2^a + 3^b = 2^c + 3^x → x ≤ b :=
begin
  sorry
end

end max_x_condition_l603_603118


namespace bracelet_arrangements_l603_603290

-- Definitions for the problem
def num_beads : ℕ := 8
def num_identical_red_beads : ℕ := 2

-- The main theorem statement
theorem bracelet_arrangements : 
  let total_arrangements := (Nat.factorial num_beads) / (Nat.factorial num_identical_red_beads * num_beads * 2) in
  total_arrangements = 1260 :=
by
  sorry

end bracelet_arrangements_l603_603290


namespace smallest_x_l603_603706

noncomputable def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 4 then x^2 - 4 * x + 5 else sorry

theorem smallest_x (x : ℝ) (h₁ : ∀ x > 0, f (4 * x) = 4 * f x)
  (h₂ : ∀ x, (1 ≤ x ∧ x ≤ 4) → f x = x^2 - 4 * x + 5) :
  ∃ x₀, x₀ > 0 ∧ f x₀ = 1024 ∧ (∀ y, y > 0 ∧ f y = 1024 → y ≥ x₀) :=
sorry

end smallest_x_l603_603706


namespace symmetric_with_respect_to_xoy_l603_603670

-- Definition of symmetry with respect to the xoy plane
def symmetric_point (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (x, y, -z)

-- The point P and its coordinates
def point_P := (1 : ℝ, 3 : ℝ, -5 : ℝ)

-- The symmetric point of P with respect to the xoy plane
def symmetric_point_P := (1 : ℝ, 3 : ℝ, 5 : ℝ)

-- Proof statement
theorem symmetric_with_respect_to_xoy :
  symmetric_point 1 3 (-5) = symmetric_point_P :=
by
  simp [symmetric_point, symmetric_point_P]
  done

end symmetric_with_respect_to_xoy_l603_603670


namespace goals_last_season_l603_603681

theorem goals_last_season : 
  ∀ (goals_last_season goals_this_season total_goals : ℕ), 
  goals_this_season = 187 → 
  total_goals = 343 → 
  total_goals = goals_last_season + goals_this_season → 
  goals_last_season = 156 := 
by 
  intros goals_last_season goals_this_season total_goals 
  intro h_this_season 
  intro h_total_goals 
  intro h_equation 
  calc 
    goals_last_season = total_goals - goals_this_season : by rw [h_equation, Nat.add_sub_cancel_left]
    ... = 343 - 187 : by rw [h_this_season, h_total_goals]
    ... = 156 : by norm_num

end goals_last_season_l603_603681


namespace values_of_abc_l603_603547

noncomputable def polynomial_divisibility (a b c : ℤ) : Prop :=
  let f := λ x:ℤ, x^4 + a * x^2 + b * x + c
  in (∀ x:ℤ, f (x-1) = (x-1)^3 * (x * (x + 1) + (a + b + 1) - 1) + (a + b + c + 1) - 1)

theorem values_of_abc {a b c : ℤ} :
  polynomial_divisibility a b c ->
  a = -6 ∧ b = 8 ∧ c = -3 :=
sorry

end values_of_abc_l603_603547


namespace buratino_made_error_l603_603040

theorem buratino_made_error : 
  ¬ ∃ cube : Array (Fin 6) (Array (Fin 4) Bool), 
    ∀ i j,  
      (cube[i]![j] || 
       (i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5) ∧ 
       (j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3)) := 
sorry

end buratino_made_error_l603_603040


namespace total_marks_scored_l603_603314

theorem total_marks_scored :
  let Keith_score := 3.5
  let Larry_score := Keith_score * 3.2
  let Danny_score := Larry_score + 5.7
  let Emma_score := (Danny_score * 2) - 1.2
  let Fiona_score := (Keith_score + Larry_score + Danny_score + Emma_score) / 4
  Keith_score + Larry_score + Danny_score + Emma_score + Fiona_score = 80.25 :=
by
  sorry

end total_marks_scored_l603_603314


namespace arithmetic_sequence_a_inv_sum_b_n_l603_603233

noncomputable def sequence_a (n : ℕ) : ℤ := 
if n = 1 then 1 else sorry

def sequence_b (n : ℕ) : ℚ := 
1 / (2^n * sequence_a n)

def Sn (n : ℕ) : ℚ := 
∑ i in finset.range n + 1, sequence_b i

theorem arithmetic_sequence_a_inv : ∀ n, ∃ k d : ℚ, (sequence_a (n + 1))⁻¹ = k + n * d := 
sorry

theorem sum_b_n : ∀ n : ℕ, Sn n = 3 - ((2 * n + 3) / 2^n) :=
sorry

end arithmetic_sequence_a_inv_sum_b_n_l603_603233


namespace matrix_multiplication_l603_603512

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, -1], ![5, 2]]

def matrix2 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 6], ![-1, 3]]

def expected_product : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, 15], ![8, 36]]

theorem matrix_multiplication :
  matrix1 ⬝ matrix2 = expected_product :=
by {
  sorry
}

end matrix_multiplication_l603_603512


namespace eighth_term_is_one_over_32_l603_603748

-- Define the sequence according to the conditions
def a (n : ℕ) : ℚ :=
  (-1)^(n+1) * (n / 2^n)

-- Prove that the 8th term is 1/32
theorem eighth_term_is_one_over_32 :
  a 8 = 1/32 :=
by
  rw [a, Nat.cast_succ, pow_succ', pow_add, pow_one, mul_comm, mul_one, div_pow, nat.pow_succ, mul_assoc]
  -- rest of the proof steps skipped
  sorry

end eighth_term_is_one_over_32_l603_603748


namespace rainbow_four_digit_numbers_l603_603270

theorem rainbow_four_digit_numbers : 
  ∃ n : ℕ, n = 3645 ∧ 
         (∀ a b c d : ℕ, 
          0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ 
          (a - b) * (c - d) < 0 → 
          n = (number_of_combinations (λ (a b : ℕ), a ≠ 0 ∧ b > a ∧ possible_combinations (c > d)) + 
               number_of_combinations (λ (a b : ℕ), a ≠ 0 ∧ b < a ∧ possible_combinations (d > c)))) :=
by sorry

end rainbow_four_digit_numbers_l603_603270


namespace area_of_given_rhombus_l603_603757

def area_of_rhombus (d1 d2 : ℝ) := (d1 * d2) / 2

theorem area_of_given_rhombus :
  area_of_rhombus 17 20 = 170 := 
sorry

end area_of_given_rhombus_l603_603757


namespace fruit_cups_francis_l603_603177

-- Definitions for the problem
def m := 2 -- cost of a muffin
def f := 3 -- cost of a fruit cup
def Fm := 2 -- number of muffins Francis had
def Ff : ℕ -- number of fruit cups Francis had, yet to be determined
def Km := 2 -- number of muffins Kiera had
def Kf := 1 -- number of fruit cups Kiera had
def total_cost := 17 -- total cost of breakfast

-- Cost equations
def cost_francis (Ff : ℕ) := m * Fm + f * Ff
def cost_kiera := m * Km + f * Kf
def total (Ff : ℕ) := cost_francis Ff + cost_kiera

-- The statement asserting that the number of fruit cups Francis had
-- such that their combined breakfast cost is $17.
theorem fruit_cups_francis : total 2 = total_cost :=
by
  -- Full proof is not required, placeholder to indicate the problem
  sorry

end fruit_cups_francis_l603_603177


namespace findNumberOfSatisfyingPermutations_l603_603143

-- Define the problem conditions and solution statement.

noncomputable def numberOfSatisfyingPermutations : Nat :=
  let S := ([-3, -2, -1, 0, 1, 2, 3, 4] : List Int)
  let permutations := List.permutations S
  let satisfyingPermutations := permutations.filter (λ l => 
    l.length = 8 ∧ 
    ((l[0] * l[1] ≤ l[1] * l[2]) ∧ (l[1] * l[2] ≤ l[2] * l[3]) ∧ 
     (l[2] * l[3] ≤ l[3] * l[4]) ∧ (l[3] * l[4] ≤ l[4] * l[5]) ∧ 
     (l[4] * l[5] ≤ l[5] * l[6]) ∧ (l[5] * l[6] ≤ l[6] * l[7])))
  satisfyingPermutations.length

theorem findNumberOfSatisfyingPermutations : numberOfSatisfyingPermutations = 21 := 
  by 
    sorry

end findNumberOfSatisfyingPermutations_l603_603143


namespace integer_solutions_count_l603_603781

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l603_603781


namespace simplify_fraction_1_210_plus_17_35_l603_603363

theorem simplify_fraction_1_210_plus_17_35 :
  1 / 210 + 17 / 35 = 103 / 210 :=
by sorry

end simplify_fraction_1_210_plus_17_35_l603_603363


namespace rectangle_area_in_triangle_is_48_l603_603730

def rectangle_area_in_triangle_proof_problem : Prop :=
  ∃ (AD AB : ℝ), 
    let PR := 12 in
    let altitude_Q_to_PR := 8 in
    let AB : ℝ := 1 / 3 * AD in
    AB * AD = 48

theorem rectangle_area_in_triangle_is_48 : rectangle_area_in_triangle_proof_problem := by
  sorry

end rectangle_area_in_triangle_is_48_l603_603730


namespace thirteen_pow_seven_mod_eight_l603_603742

theorem thirteen_pow_seven_mod_eight : 
  (13^7) % 8 = 5 := by
  sorry

end thirteen_pow_seven_mod_eight_l603_603742


namespace parallelogram_area_bound_l603_603074

def parallelogram_inscribed (ABCD : Type) (M : Type) [parallelogram ABCD] [regular_hexagon M] (center_symm : center_of_symmetry ABCD = center_of_hexagon M) : Prop :=
  area ABCD ≤ (2 / 3) * area M

theorem parallelogram_area_bound (ABCD : Type) (M : Type) [parallelogram ABCD] [regular_hexagon M] (center_symm : center_of_symmetry ABCD = center_of_hexagon M) :
  area ABCD ≤ (2 / 3 * area M) := 
sorry

end parallelogram_area_bound_l603_603074


namespace algebraic_expression_value_l603_603434

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 23 - 1) : x^2 + 2 * x + 2 = 24 :=
by
  -- Start of the proof
  sorry -- Proof is omitted as per instructions

end algebraic_expression_value_l603_603434


namespace triangle_area_on_ellipse_l603_603333

theorem triangle_area_on_ellipse :
  ∀ (P : Type) 
    (F1 F2 : Type) 
    (x y : ℝ),
  (x^2 / 9 + y^2 / 5 = 1) →
  (angle : ℝ) (HF1PF2 : angle = 60) →
  (dist_F1_F2 : ℝ) (dist_F1_F2 = 4) →
  (F1P_plus_PF2 : ℝ) (F1P_plus_PF2 = 6) →
  ∃ (area : ℝ), area = (5 * Real.sqrt 3 / 3) := 
by
  intros
  sorry

end triangle_area_on_ellipse_l603_603333


namespace kelly_students_l603_603315

theorem kelly_students (S : ℕ) :
  (3 * S + 6) / 2 + 5 = 20 ↔ S = 8 :=
by
  split
  · intro h
    have h1 : (3 * S + 6) / 2 = 15 := by linarith
    have h2 : 3 * S + 6 = 30 := by linarith only [mul_right_cancel' h1]
    have h3 : 3 * S = 24 := by linarith      
    have h4 : S = 8 := nat.eq_of_mul_eq_mul_left three_ne_zero h3
    exact h4
    
  · intro h
    rw h
    linarith


end kelly_students_l603_603315


namespace isosceles_triangle_angle_l603_603862

-- Definition of required angles and the given geometric context
variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]
variables (angleBAC : ℝ) (angleBCA : ℝ)

-- Given: shared vertex A, with angle BAC of pentagon
axiom angleBAC_def : angleBAC = 108

-- To Prove: determining the measure of angle BCA in the isosceles triangle
theorem isosceles_triangle_angle (h : 180 > 2 * angleBAC) : angleBCA = (180 - angleBAC) / 2 :=
  sorry

end isosceles_triangle_angle_l603_603862


namespace angles_are_equal_l603_603715

variable {α : Type}
variable (Q : Quadrilateral α) -- assuming Q is the square

-- Conditions
variable (A B C D M L K P : α)
variable (segment_length : ℝ) -- the equal segments

-- Assume sides AB, BC, CD, DA are equal and form a square
variable (is_square : is_square Q A B C D)

-- Assume points M, L, K, P are such that AM = BL = CK = DP = segment_length
variable (AM_length : distance A M = segment_length)
variable (BL_length : distance B L = segment_length)
variable (CK_length : distance C K = segment_length)
variable (DP_length : distance D P = segment_length)

-- Assume angles that are in question
variable (angle_CML : angle C M L = α)
variable (angle_PKM : angle P K M = α)

-- Proof statement
theorem angles_are_equal : angle C M L = angle P K M :=
by sorry

end angles_are_equal_l603_603715


namespace sum_seven_consecutive_l603_603373

theorem sum_seven_consecutive (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 7 * n + 7 :=
by
  sorry

end sum_seven_consecutive_l603_603373


namespace max_profit_l603_603065

noncomputable def profit (x : ℕ) : ℝ := -0.15 * (x : ℝ)^2 + 3.06 * (x : ℝ) + 30

theorem max_profit :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 ∧ ∀ y : ℕ, 0 ≤ y ∧ y ≤ 15 → profit y ≤ profit x :=
by
  sorry

end max_profit_l603_603065


namespace range_of_k_l603_603638

theorem range_of_k (k : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (k + 2) * x1 - 1 > (k + 2) * x2 - 1) → k < -2 := by
  sorry

end range_of_k_l603_603638


namespace triangle_bqc_max_area_l603_603303

noncomputable def incenter (a b c : ℝ) : ℝ := sorry -- Placeholder for incenter computation

theorem triangle_bqc_max_area (a b c : ℝ) (E : ℝ) (I_B I_C : ℝ) (Q : ℝ) (h1 : a = 13) (h2 : b = 15) (h3 : c = 14)
  (h4 : E < b) (h5 : E > 0) (h6 : I_B = incenter a b E) (h7 : I_C = incenter a c E) (h8 : Q ≠ 0): 
  ∃ a b c : ℝ, a = 112.5 ∧ b = 56.25 ∧ c = 3 ∧ (max_area : ℝ) = a - b * real.sqrt c :=
  sorry 

end triangle_bqc_max_area_l603_603303


namespace percentage_error_formula_l603_603095

noncomputable def percentage_error_in_area (a b : ℝ) (x y : ℝ) :=
  let actual_area := a * b
  let measured_area := a * (1 + x / 100) * b * (1 + y / 100)
  let error_percentage := ((measured_area - actual_area) / actual_area) * 100
  error_percentage

theorem percentage_error_formula (a b x y : ℝ) :
  percentage_error_in_area a b x y = x + y + (x * y / 100) :=
by
  sorry

end percentage_error_formula_l603_603095


namespace part1_part2_l603_603228

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x + 1)

-- Definition of the function g(x)
def g (x : ℝ) : ℝ := f x + abs (x + 1)

-- Defining the range M of the function g(x)
def M : set ℝ := {t | 3 ≤ t}

-- To prove the inequality for f(x) given the interval -1 ≤ x ≤ 1
theorem part1 (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) : f x ≤ 3 := sorry

-- To prove the inequality for t^2 + 1 given t in the range of g(x)
theorem part2 (t : ℝ) (ht : t ∈ M) : t^2 + 1 ≥ 3 / t + 3 * t := sorry

end part1_part2_l603_603228


namespace percentage_of_students_in_biology_l603_603505

variable (t : ℕ) (n : ℕ)

theorem percentage_of_students_in_biology (h_t : t = 880) (h_n : n = 572) : 
  ((t - n).toRat / t.toRat) * 100 = 35 := by
    sorry

end percentage_of_students_in_biology_l603_603505


namespace problem_statement_l603_603965

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 3 + 1) : x^2 - 2*x + 1 = 3 :=
sorry

end problem_statement_l603_603965


namespace sufficient_but_not_necessary_condition_l603_603968

variable (x : ℝ)

def p := x > 2
def q := x^2 > 4

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l603_603968


namespace triangles_areas_equal_l603_603909

noncomputable def y_direct (x : ℝ) : ℝ := x
noncomputable def y_direct_a (a x : ℝ) : ℝ := a * x
noncomputable def y_inverse (k x : ℝ) : ℝ := k / x

def point_A (k : ℝ) : ℝ × ℝ :=
  let x := Real.sqrt k in
  (x, x)

def point_C (a k : ℝ) : ℝ × ℝ :=
  let x := Real.sqrt (k / a) in
  (x, Real.sqrt (a * k))

noncomputable def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem triangles_areas_equal (a k : ℝ) (ha : 0 < a) (hk : 0 < k) :
  let A := point_A k
  let C := point_C a k
  let S1 := triangle_area A.1 A.2
  let S2 := triangle_area C.1 C.2
  S1 = S2 := by
  sorry

end triangles_areas_equal_l603_603909


namespace compare_ND_NH_l603_603656

-- Here we define the entities and conditions from the problem context.
variables {Point : Type} [metric_space Point]
variables (A B C D K M N H : Point)
variables (l : set Point)
variables (AB : line Point) (AC : segment Point) (BC : segment Point) (KM : segment Point)
variables (perp_to_line_l : ∀ (X : Point), X ∈ l ⟺ ∃ Y ∈ l, metric.dist X Y = 0) 
variables (isosceles_right_triangle : ∀ (A B C : Point), is_isosceles_right_triangle ABC)
variables (D_midpoint : ∀ (A B : Point), is_midpoint D A B)
variables (parallel_line : ∀ (A B C : Point), are_parallel (line_through C) (line_through A B))
variables (CK_AK_ratio BM_MC_ratio FN_NK_ratio : ℝ)
variables (ratio_condition : CK_AK_ratio = BM_MC_ratio ∧ BM_MC_ratio = FN_NK_ratio)
variables (H_perpendicular : ∀ (N : Point), is_perpendicular_to_line N l H)

theorem compare_ND_NH : 
  is_isosceles_right_triangle ABC → 
  is_midpoint D A B → 
  are_parallel (line_through C) (line_through A B) →  
  ((ratio_condition) → 
  (is_perpendicular_to_line N l H) →  
  metric.dist N D = metric.dist N H) :=
begin
  -- proof would go here. This is skipped as per instructions with sorry.
  sorry
end

end compare_ND_NH_l603_603656


namespace x_coordinate_point_P_l603_603273

theorem x_coordinate_point_P (x y : ℝ) (h_on_parabola : y^2 = 4 * x) 
  (h_distance : dist (x, y) (1, 0) = 3) : x = 2 :=
sorry

end x_coordinate_point_P_l603_603273


namespace relationship_among_abc_l603_603964

noncomputable def a : ℝ := (1 / 5)^(1 / 2)
noncomputable def b : ℝ := Real.log (1 / 3) / Real.log 5
noncomputable def c : ℝ := Real.log (1 / 3) / Real.log (1 / 5)

theorem relationship_among_abc : c > a ∧ a > b := 
by 
have ha : a = (1 / 5)^(1 / 2) := rfl
have hb : b = log (1 / 3) / log 5 := rfl
have hc : c = log (1 / 3) / log (1 / 5) := rfl
sorry

end relationship_among_abc_l603_603964


namespace rectangle_lines_combinations_l603_603161

theorem rectangle_lines_combinations : 5.choose 2 * 4.choose 2 = 60 := by
  sorry

end rectangle_lines_combinations_l603_603161


namespace quilt_block_fraction_shaded_l603_603516

theorem quilt_block_fraction_shaded :
  let total_squares := 16
      divided_squares := 4
      shaded_per_divided_square := 1 / 2
      total_shaded := divided_squares * shaded_per_divided_square
      fraction_shaded := total_shaded / total_squares
  in
  fraction_shaded = 1 / 8 := 
by
  sorry

end quilt_block_fraction_shaded_l603_603516


namespace comprehensive_survey_suitability_l603_603439
noncomputable theory

-- Define the question
def question : Type := "Which of the following is not suitable for a comprehensive survey?"

-- Define each option
def optionA : Type := "Security check for passengers before boarding a plane"
def optionB : Type := "School recruiting teachers and conducting interviews for applicants"
def optionC : Type := "Understanding the extracurricular reading time of seventh-grade students in a school"
def optionD : Type := "Understanding the service life of a batch of light bulbs"

-- Define the comprehensive survey suitability for each option as conditions
def suitable_for_comprehensive_survey (option : Type) : Prop :=
  option = optionA ∨ option = optionB ∨ option = optionC ∨ option = optionD

-- Express that option D is not suitable for a comprehensive survey
def not_suitable_for_comprehensive_survey (option : Type) : Prop :=
  option = optionD

-- The proof statement
theorem comprehensive_survey_suitability :
  suitable_for_comprehensive_survey optionA ∧
  suitable_for_comprehensive_survey optionB ∧
  suitable_for_comprehensive_survey optionC ∧
  ¬ suitable_for_comprehensive_survey optionD :=
by 
  -- Assume all options except D are suitable for a comprehensive survey
  sorry

end comprehensive_survey_suitability_l603_603439


namespace range_of_a_l603_603219

theorem range_of_a (a : ℝ) (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a)) (h_local_max : ∃ f, ∃ x, f(x : ℝ) → ∀ x, deriv f x = a * (x + 1) * (x - a) → has_local_max f x) :
  a ∈ Ioo (-1 : ℝ) (0 : ℝ) :=
sorry

end range_of_a_l603_603219


namespace angle_AC1B_l603_603686

theorem angle_AC1B (ABC : Triangle) (A' B' C1 : Point)
  (φ : ℝ) (h_non_isosceles : ¬ is_isosceles ABC)
  (h_isosceles_AB'C : Isosceles (triangle_AB'C ABC) (base AC', ABC))
  (h_isosceles_CA'B : Isosceles (triangle_CA'B ABC) (base BC', ABC))
  (h_base_angle_AB'C : base_angle (triangle_AB'C ABC) = φ)
  (h_base_angle_CA'B : base_angle (triangle_CA'B ABC) = φ)
  (h_C1_definition : C1 = point_of_intersection (perpendicular_from C to A'B') (perpendicular_bisector_of_segment AB)) :
  angle [A, C1, B] = 180 - φ := by
  sorry

end angle_AC1B_l603_603686


namespace symmetric_points_line_l603_603200

theorem symmetric_points_line (a b : ℝ) : 
  (∃ m : ℝ, m = (1, 4)) ∧ (-1 + 1)*(a / (-1 - 3) + (3 - 4) / 1) = -1 ∧ (a + 4 - b = 0) → a / b = 1 / 3 :=
by
  sorry

end symmetric_points_line_l603_603200


namespace allison_upload_ratio_l603_603898

theorem allison_upload_ratio :
  ∃ (x y : ℕ), (x + y = 30) ∧ (10 * x + 20 * y = 450) ∧ (x / 30 = 1 / 2) :=
by
  sorry

end allison_upload_ratio_l603_603898


namespace value_of_x_l603_603254

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l603_603254


namespace quadratic_root_proof_l603_603458

theorem quadratic_root_proof : 
  (∃ x, x = (-5 + Real.sqrt(5^2 + 4 * 3 * 1)) / (2 * 3) ∨ x = (-5 - Real.sqrt(5^2 + 4 * 3 * 1)) / (2 * 3)) →
  ∃ x, (3 * x^2 + 5 * x - 1 = 0) :=
by
  intro h
  cases h with x hx
  exists x
  simp [hx]
  sorry

end quadratic_root_proof_l603_603458


namespace markup_percentage_l603_603073

-- Define the purchase price and the gross profit
def purchase_price : ℝ := 54
def gross_profit : ℝ := 18

-- Define the sale price after discount
def sale_discount : ℝ := 0.8

-- Given that the sale price after the discount is purchase_price + gross_profit
theorem markup_percentage (M : ℝ) (SP : ℝ) : 
  SP = purchase_price * (1 + M / 100) → -- selling price as function of markup
  (SP * sale_discount = purchase_price + gross_profit) → -- sale price after 20% discount
  M = 66.67 := 
by
  -- sorry to skip the proof
  sorry

end markup_percentage_l603_603073


namespace sum_of_last_three_coeffs_l603_603024

theorem sum_of_last_three_coeffs (a : ℝ) (h : a ≠ 0) :
  let expr := (1 - (1 / a)) ^ 7,
  let last_three_sum := 1 - 7 + 21 in
  last_three_sum = 15 :=
by
  sorry

end sum_of_last_three_coeffs_l603_603024


namespace count_integer_values_satisfying_condition_l603_603808

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l603_603808


namespace example_3_is_analogical_reasoning_l603_603902

-- Definitions based on the conditions of the problem:
def is_analogical_reasoning (reasoning: String): Prop :=
  reasoning = "from one specific case to another similar specific case"

-- Example of reasoning given in the problem.
def example_3 := "From the fact that the sum of the distances from a point inside an equilateral triangle to its three sides is a constant, it is concluded that the sum of the distances from a point inside a regular tetrahedron to its four faces is a constant."

-- Proof statement based on the conditions and correct answer.
theorem example_3_is_analogical_reasoning: is_analogical_reasoning example_3 :=
by 
  sorry

end example_3_is_analogical_reasoning_l603_603902


namespace dickens_birth_day_l603_603499

def is_leap_year (year : ℕ) : Prop :=
  (year % 400 = 0) ∨ (year % 4 = 0 ∧ year % 100 ≠ 0)

theorem dickens_birth_day :
  let day_of_week_2012 := 2 -- 0: Sunday, 1: Monday, ..., 2: Tuesday
  let years := 200
  let regular_years := 151
  let leap_years := 49
  let days_shift := regular_years + 2 * leap_years
  let day_of_week_birth := (day_of_week_2012 + days_shift) % 7
  day_of_week_birth = 5 -- 5: Friday
:= 
sorry -- proof not supplied

end dickens_birth_day_l603_603499


namespace shaded_area_fraction_l603_603086

-- Define the initial conditions and the infinite process
noncomputable def shaded_fractional_part : ℚ :=
  let f : ℕ → ℚ := λ n, (1 / 16 : ℚ)^n
  4 / 15

-- Statement of the theorem to prove the fractional part that is shaded
theorem shaded_area_fraction : 
  let total_area : ℚ := 1 in
  let shaded_area := 𝓝 sum_range_0_inf (λ (n : ℕ), (4 / 16 ^ (n + 1) : ℚ)) in
  shaded_area = (4 / 15 : ℚ) :=
  sorry

end shaded_area_fraction_l603_603086


namespace seq_arith_specific_max_sum_another_arith_seq_cond_geom_seq_l603_603844

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions for arithmetic sequence
axiom arith_seq (d : ℝ) : ∀ n, S n = (n * (2*a 1 + (n-1)*d)) / 2
axiom S4_arith : S 4 = 4 * a 1 + 6 * d
axiom S8_arith : S 8 - S 4 = 4 * a 1 + 22 * d
axiom S12_arith : S 12 - S 8 = 4 * a 1 + 38 * d

-- Conditions for specific sequence (a_n = 26 - 2n)
axiom specific_seq : ∀ n, a n = 26 - 2 * n
axiom max_sum_n : ∃ n, n ≤ 13 ∧ ∀ m ≤ 13, S n ≥ S m

-- Conditions for another arithmetic sequence
axiom another_arith_seq : a 1 > 0
axiom sum_equal : S 10 = S 20
axiom S_n_pos : ∀ n, n < 32 → S n < 0

-- Conditions for geometric sequence
axiom pos_geom_seq : ∀ n, a n > 0
axiom a6_one : a 6 = 1
axiom T11_geom : T 11 = 1

-- Proof statements
theorem seq_arith : S 4 = 4 * a 1 + 6 * d ∧ S 8 - S 4 = 4 * a 1 + 22 * d ∧ S 12 - S 8 = 4 * a 1 + 38 * d :=
  by sorry

theorem specific_max_sum : ¬ (∀ n, n = 13 → maximize S n) :=
  by sorry

theorem another_arith_seq_cond : ∀ n, n < 32 → S n < 0 :=
  by sorry

theorem geom_seq : T 11 = 1 :=
  by sorry

end seq_arith_specific_max_sum_another_arith_seq_cond_geom_seq_l603_603844


namespace lcm_9_12_15_l603_603002

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l603_603002


namespace find_xy_l603_603210

variable (x y : ℚ)

theorem find_xy (h1 : 1/x + 3/y = 1/2) (h2 : 1/y - 3/x = 1/3) : 
    x = -20 ∧ y = 60/11 := 
by
  sorry

end find_xy_l603_603210


namespace books_remaining_after_second_day_l603_603070

variable (x a b c d : ℕ)

theorem books_remaining_after_second_day :
  let books_borrowed_first_day := a * b
  let books_borrowed_second_day := c
  let books_returned_second_day := (d * books_borrowed_first_day) / 100
  x - books_borrowed_first_day - books_borrowed_second_day + books_returned_second_day =
  x - (a * b) - c + ((d * (a * b)) / 100) :=
sorry

end books_remaining_after_second_day_l603_603070


namespace number_of_solutions_l603_603615

def is_solution (x y z : ℕ) : Prop :=
  Nat.lcm x y = 72 ∧ Nat.lcm x z = 600 ∧ Nat.lcm y z = 900

theorem number_of_solutions :
  {n : ℕ // ∃ (triples : Finset (ℕ × ℕ × ℕ)), triples.filter (λ t, is_solution t.1 t.2.1 t.2.2) = n} = 15 :=
sorry

end number_of_solutions_l603_603615


namespace arrangement_plans_l603_603126

-- Definitions based on the conditions
def num_students := 4
def num_events := 3
def events := {A, B, C}
def condition_1 := ∀ event ∈ events, ∃ student : students, serves student event
def condition_2 := ∀ student : students, ∃! event ∈ events, serves student event
def student_A_not_in_A := ∀ event ∈ events, serves studentA event → event ≠ A

-- The equivalent Lean statement for the proof problem
theorem arrangement_plans :
  (condition_1 ∧ condition_2 ∧ student_A_not_in_A) →
  ∃ n : ℕ, n = 24 :=
by
  sorry

end arrangement_plans_l603_603126


namespace range_of_q_l603_603705

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m ∣ n → m = n

def greatest_prime_factor (n : ℕ) : ℕ :=
  if is_prime n then n else
    nat.find_greatest (λ k, is_prime k ∧ k ∣ n) (n-1)

def q (x : ℕ) : ℕ :=
  if x >= 2 ∧ x <= 20 then
    if is_prime (⌊x⌋) then 
      ⌊x⌋ + 2
    else
      let y := greatest_prime_factor (⌊x⌋) in
      q(y) + (x + 1 - ⌊x⌋) + 2
  else 0

theorem range_of_q : set.range q = set.Ico 4 23 :=
  sorry

end range_of_q_l603_603705


namespace disneyland_attraction_order_l603_603677

theorem disneyland_attraction_order :
  let num_attractions := 6
  let sm_after_hm := true
  (num_attractions = 6 ∧ sm_after_hm) →
  ∃ (n : ℕ), n = 120 :=
by
  let units := 5
  let factorial_5 := 120
  let num_ways := factorial_5
  trivial

end disneyland_attraction_order_l603_603677


namespace factor_quadratic_l603_603531

theorem factor_quadratic (x : ℝ) : 
  (x^2 + 6 * x + 9 - 16 * x^4) = (-4 * x^2 + 2 * x + 3) * (4 * x^2 + 2 * x + 3) := 
by 
  sorry

end factor_quadratic_l603_603531


namespace cistern_fill_time_l603_603824

-- Define the rates at which pipes p, q, and r can fill/drain the cistern.
def rate_p := 1/10
def rate_q := 1/15
def rate_r := -1/30

-- Define the time pipes p and q are open together.
def time_pq_open := 4

-- Define the remaining fraction of the cistern to be filled after 4 minutes.
def filled_cistern_after_4_minutes : ℚ := (rate_p + rate_q) * time_pq_open
def remaining_cistern : ℚ := 1 - filled_cistern_after_4_minutes

-- Define the combined rate of pipes q and r.
def combined_rate_q_r := rate_q + rate_r

-- Prove that the time it takes to fill the remaining cistern at the combined rate is 10 minutes.
theorem cistern_fill_time : 
  remaining_cistern / combined_rate_q_r = 10 := 
by 
  sorry

end cistern_fill_time_l603_603824


namespace PQ_perp_RS_l603_603756

-- Definitions and problem assumptions
variables {A B C D M P Q R S : Point}
variables [convex_quadrilateral A B C D]
variables (M_property : diagonals_intersect A B C D M)
variables (P_centroid : centroid A M D P)
variables (Q_centroid : centroid C M B Q)
variables (R_orthocenter : orthocenter D M C R)
variables (S_orthocenter : orthocenter M A B S)

-- Proof statement
theorem PQ_perp_RS : perpendicular P Q R S :=
sorry

end PQ_perp_RS_l603_603756


namespace at_most_two_integer_solutions_proof_l603_603579

noncomputable def at_most_two_integer_solutions (a b c : ℝ) (h : a > 100) : Prop :=
  ∀ x1 x2 x3 : ℤ, 
    abs (a * (x1 : ℝ)^2 + b * (x1 : ℝ) + c) ≤ 50 ∧ 
    abs (a * (x2 : ℝ)^2 + b * (x2 : ℝ) + c) ≤ 50 ∧ 
    abs (a * (x3 : ℝ)^2 + b * (x3 : ℝ) + c) ≤ 50 
    → (x1 = x2) ∨ (x2 = x3) ∨ (x3 = x1)

theorem at_most_two_integer_solutions_proof (a b c : ℝ) (h : a > 100) :
  at_most_two_integer_solutions a b c h :=
begin
  sorry,
end

end at_most_two_integer_solutions_proof_l603_603579


namespace range_of_A_value_of_b_l603_603643

-- Assuming the basic setup for a triangle
variables {A B C : Real} {a b c S : Real}
-- Definitions for angles and sides in a triangle.
def angleA := A
def angleB := B
def angleC := C
def side_a := a
def side_b := b
def side_c := c
def area := S
def dot_product := side_b * side_c * Real.cos angleA
def area_formula := (1 / 2) * side_b * side_c * Real.sin angleA

-- Given Conditions
axiom tan_ratios : (Real.tan angleA) : (Real.tan angleB) : (Real.tan angleC) = 1 : 2 : 3
axiom c_is_1 : side_c = 1

-- Part (1): Proving the range of angle A given the inequality and the area
theorem range_of_A : 
  dot_product ≤ 2 * Real.sqrt 3 * area → (π / 6) ≤ angleA ∧ angleA < π := sorry

-- Part (2): Proving the value of b given the tan ratios and c = 1
theorem value_of_b : 
  tan_ratios ∧ c_is_1 → side_b = (2 * Real.sqrt 2) / 3 := sorry

end range_of_A_value_of_b_l603_603643


namespace tan_sec_solution_l603_603950

noncomputable def smallest_positive_solution (x : ℝ) : ℝ :=
  ∃ n : ℤ, x = (Real.pi / 26) + (2 * Real.pi * n) / 13 ∧ x > 0

theorem tan_sec_solution (x : ℝ) :
  (x = Real.pi / 26) ↔ (smallest_positive_solution x ∧ (Real.tan (4 * x) + Real.tan (5 * x) = Real.sec (5 * x))) :=
by
  sorry

end tan_sec_solution_l603_603950


namespace find_prices_possible_purchasing_schemes_maximize_profit_l603_603463

namespace PurchasePriceProblem

/-- 
Define the purchase price per unit for bean sprouts and dried tofu,
and show that they satisfy the given conditions. 
--/
theorem find_prices (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 240)
  (h2 : 3 * x + 4 * y = 340) :
  x = 60 ∧ y = 40 := 
by sorry

/-- 
Given the conditions on the purchase price of bean sprouts and dried tofu
and the need of purchasing a total of 200 units for no more than $10440, 
determine the valid purchasing schemes.
--/
theorem possible_purchasing_schemes 
  (a : ℤ)
  (h1 : 60 * a + 40 * (200 - a) ≤ 10440)
  (h2 : a ≥ 3 / 2 * (200 - a)) :
  120 ≤ a ∧ a ≤ 122 := 
by sorry
  
/-- 
Maximize profit based on the purchasing schemes that satisfy the conditions.
--/
theorem maximize_profit 
  (a : ℤ) 
  (h_valid : 120 ≤ a ∧ a ≤ 122) 
  (h_max : ∀ b, 120 ≤ b ∧ b ≤ 122 → 5 * a + 3000 ≥ 5 * b + 3000) :
  (a = 122) → 
  let beans_profit := 5 * a + 3000 
  in beans_profit = 3610 := 
by sorry

end PurchasePriceProblem

end find_prices_possible_purchasing_schemes_maximize_profit_l603_603463


namespace gain_percent_correct_l603_603847

-- Definitions corresponding to the conditions
def cost_price : ℝ := 900
def selling_price : ℝ := 1150

-- Calculating the gain
def gain : ℝ := selling_price - cost_price

-- Formula for gain percent
def gain_percent : ℝ := (gain / cost_price) * 100

-- Statement to prove the gain percent
theorem gain_percent_correct : gain_percent ≈ 27.78 := sorry

end gain_percent_correct_l603_603847


namespace jacob_has_5_times_more_l603_603099

variable (A J D : ℕ)
variable (hA : A = 75)
variable (hAJ : A = J / 2)
variable (hD : D = 30)

theorem jacob_has_5_times_more (hA : A = 75) (hAJ : A = J / 2) (hD : D = 30) : J / D = 5 :=
sorry

end jacob_has_5_times_more_l603_603099


namespace karen_box_crayons_l603_603683

theorem karen_box_crayons (judah_crayons : ℕ) (gilbert_crayons : ℕ) (beatrice_crayons : ℕ) (karen_crayons : ℕ)
  (h1 : judah_crayons = 8)
  (h2 : gilbert_crayons = 4 * judah_crayons)
  (h3 : beatrice_crayons = 2 * gilbert_crayons)
  (h4 : karen_crayons = 2 * beatrice_crayons) :
  karen_crayons = 128 :=
by
  sorry

end karen_box_crayons_l603_603683


namespace find_f_inv_8_l603_603560

variable (f : ℝ → ℝ)

-- Given conditions
axiom h1 : f 5 = 1
axiom h2 : ∀ x, f (2 * x) = 2 * f x

-- Theorem to prove
theorem find_f_inv_8 : f ⁻¹' {8} = {40} :=
by sorry

end find_f_inv_8_l603_603560


namespace sum_of_digits_double_l603_603322

theorem sum_of_digits_double (k : ℕ) (S : ℕ → ℕ) (H1 : S k = 2187) (H2 : ∀ digit, digit ∈ list.ofDigits 10 (2 * k) → digit ≤ 7) :
  S (2 * k) = 4374 :=
sorry

end sum_of_digits_double_l603_603322


namespace sum_of_digits_base2_of_300_l603_603020

theorem sum_of_digits_base2_of_300 : (nat.binary_digits 300).sum = 4 :=
by
  sorry

end sum_of_digits_base2_of_300_l603_603020


namespace number_of_rectangles_l603_603146

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 4) : 
  (nat.choose H 2) * (nat.choose V 2) = 60 := by
  rw [hH, hV]
  norm_num

end number_of_rectangles_l603_603146


namespace intersection_points_of_curve_and_line_max_distance_condition_l603_603294

open Real

noncomputable def curve (θ : ℝ) : ℝ × ℝ := (3 * cos θ, sin θ)

noncomputable def line (a t : ℝ) : ℝ × ℝ := (a + 4 * t, 1 - t)

theorem intersection_points_of_curve_and_line (a : ℝ) (h₁ : a = -1) :
  (∃ t θ,
    curve θ = line a t) ↔
  (∃ t θ,
    (curve θ = (3, 0) ∨ curve θ = (-21/25, 24/25))) :=
by
  sorry

theorem max_distance_condition (a : ℝ) (h₂ : ∃ θ d,
  d = √17 ∧ 
  d = abs (3 * cos θ + 4 * sin θ - a - 4) / √17) :
  a = -16 ∨ a = 8 :=
by 
  sorry

end intersection_points_of_curve_and_line_max_distance_condition_l603_603294


namespace lcm_9_12_15_l603_603003

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l603_603003


namespace total_earnings_proof_l603_603034

-- Definitions for the conditions based on given investments and returns
variables (x y : ℝ)

-- Ratio condition for investments
def investment_a := 3 * x
def investment_b := 4 * x
def investment_c := 5 * x

-- Ratio condition for returns
def return_a := 18 * x * y
def return_b := 20 * x * y
def return_c := 20 * x * y

-- Additional condition given in the problem
def condition_b_earns_more := return_b - return_a = 250

-- Total earnings computation
def total_earnings := return_a + return_b + return_c

-- Main theorem stating the total earnings
theorem total_earnings_proof : condition_b_earns_more x y → total_earnings x y = 7250 := by
  sorry

end total_earnings_proof_l603_603034


namespace geom_seq_extreme_points_l603_603298

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x^2 + 4 * x - 1

theorem geom_seq_extreme_points (a : ℕ → ℝ) (a3 a7 : ℝ) (r : ℝ) (h3 : a 3 = a3) (h7 : a 7 = a7)
  (nth_term : ∀ n, a n = a 0 * r ^ n) (h_f : ∀ x, f'(x) = 0 ↔ x = a3 ∨ x = a7) :
  a 5 = 2 :=
by
  sorry

end geom_seq_extreme_points_l603_603298


namespace find_x_for_parallel_vectors_l603_603576

theorem find_x_for_parallel_vectors :
  ∀ (x : ℚ), (∃ a b : ℚ × ℚ, a = (2 * x, 3) ∧ b = (1, 9) ∧ (∃ k : ℚ, (2 * x, 3) = (k * 1, k * 9))) ↔ x = 1 / 6 :=
by 
  sorry

end find_x_for_parallel_vectors_l603_603576


namespace proof_main_problem_l603_603069

variable (length1 length2 length3 length4 : Real)
variable (old_speed1 old_speed2 old_speed3 old_speed4 : Real)
variable (new_speed1 new_speed2 new_speed3 new_speed4 : Real)

def additional_time_section (length old_speed new_speed : Real) : Real :=
  (length / new_speed - length / old_speed) * 60

def total_additional_time : Real :=
  additional_time_section length1 old_speed1 new_speed1 +
  additional_time_section length2 old_speed2 new_speed2 +
  additional_time_section length3 old_speed3 new_speed3 +
  additional_time_section length4 old_speed4 new_speed4

def main_problem (length1 length2 length3 length4 : Real)
                 (old_speed1 old_speed2 old_speed3 old_speed4 : Real)
                 (new_speed1 new_speed2 new_speed3 new_speed4 : Real) : Prop :=
  total_additional_time length1 length2 length3 length4 old_speed1 old_speed2 old_speed3 old_speed4 new_speed1 new_speed2 new_speed3 new_speed4 ≈ 14.52

theorem proof_main_problem :
  main_problem 6 8 3 10 60 65 55 70 40 50 45 35 := by
  sorry

end proof_main_problem_l603_603069


namespace rectangles_in_octagon_l603_603482

theorem rectangles_in_octagon :
  ∀ (octagon : fin 8 → ℝ × ℝ), 
    (∀ i, dist (octagon i) (octagon ((i + 1) % 8)) = 1) →
    (∃ r1 r2 : parallelogram, is_rectangle r1 ∧ is_rectangle r2 ∧
    sum_of_areas_of_rectangles octagon = 2) :=
by
  sorry

end rectangles_in_octagon_l603_603482


namespace number_of_cakes_sold_l603_603101

namespace Bakery

variables (cakes pastries sold_cakes sold_pastries : ℕ)

-- Defining the conditions
def pastries_sold := 154
def more_pastries_than_cakes := 76

-- Defining the problem statement
theorem number_of_cakes_sold (h1 : sold_pastries = pastries_sold) 
                             (h2 : sold_pastries = sold_cakes + more_pastries_than_cakes) : 
                             sold_cakes = 78 :=
by {
  sorry
}

end Bakery

end number_of_cakes_sold_l603_603101


namespace distinct_numbers_in_set_l603_603508

theorem distinct_numbers_in_set (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬ (∃ s : finset ℝ, s.card = 3 ∧ {a, b, c, a^2 / b, b^2 / c, c^2 / a}.to_finset = s) :=
sorry

end distinct_numbers_in_set_l603_603508


namespace find_divisor_l603_603841

variable (N : ℝ) (C : ℝ) (R : ℝ)

theorem find_divisor (h : (N / 6) * C = R) : N / (N * C / R) = 6 :=
by
  have h1 : N * C = R * (N / 6) := by rw [h, div_mul_cancel]; exact zero_ne_one.symm
  have h2 : N * C = R * (N / 6) := by rw [h1, mul_div_cancel']; exact zero_ne_one.symm
  have h3 : R = N * C / N := div_eq_iff (zero_ne_one.symm') ▸ h2
  have correct_answer : 6 = N * C / R := by rw [← h3, mul_comm, div_div_div_eq, one_mul, div_mul_cancel, div_self]; exact zero_ne_one.symm
  have final_answer : (N / (N * C / R)) = 6 := div_eq_iff (zero_ne_one.symm') ▸ correct_answer

  sorry

end find_divisor_l603_603841


namespace four_points_concyclic_l603_603286

open Triangle

variable {α : Type*} [EuclideanGeometry α]

theorem four_points_concyclic (A B C E F M N P Q : α)
  (hABC: triangle.is_acute A B C)
  (hBE: altitude B E A C)
  (hCF: altitude C F A B)
  (hCircle1: circle_with_diameter A B (line_through (pt C F)) M N)
  (hCircle2: circle_with_diameter A C (line_through (pt B E)) P Q) :
  concyclic M P N Q := sorry

end four_points_concyclic_l603_603286


namespace students_not_liking_either_l603_603653

variable (U : Finset ℕ)
variable (A B : Finset ℕ)
variable (n_U : ℕ)
variable (n_A : ℕ)
variable (n_B : ℕ)
variable (n_A_inter_B : ℕ)

theorem students_not_liking_either (h1 : n_U = 70)
                                (h2 : n_A = 37)
                                (h3 : n_B = 49)
                                (h4 : n_A_inter_B = 20) :
  (n_U - n_A - n_B + n_A_inter_B) = 4 :=
by {
  simp [h1, h2, h3, h4],
  nat,
  sorry
}

end students_not_liking_either_l603_603653


namespace dhoni_savings_percent_l603_603935

variable (E : ℝ) -- Assuming E is Dhoni's last month's earnings

-- Condition 1: Dhoni spent 25% of his earnings on rent
def spent_on_rent (E : ℝ) : ℝ := 0.25 * E

-- Condition 2: Dhoni spent 10% less than what he spent on rent on a new dishwasher
def spent_on_dishwasher (E : ℝ) : ℝ := 0.225 * E

-- Prove the percentage of last month's earnings Dhoni had left over
theorem dhoni_savings_percent (E : ℝ) : 
    52.5 / 100 * E = E - (spent_on_rent E + spent_on_dishwasher E) :=
by
  sorry

end dhoni_savings_percent_l603_603935


namespace slope_probability_eq_l603_603692

theorem slope_probability_eq (P : ℝ × ℝ) 
  (hP : P.1 ∈ Ioo 0 1 ∧ P.2 ∈ Ioo 0 1) 
  (hS : P.2 - (1/2 : ℝ) ≥ 1 * (P.1 - (3/4 : ℝ))) : 
  (3 : ℕ) + 16 = 19 := 
by sorry

end slope_probability_eq_l603_603692


namespace each_hedgehog_ate_1050_strawberries_l603_603827

noncomputable def total_strawberries_per_basket : ℕ := 900
noncomputable def number_of_baskets : ℕ := 3
noncomputable def fraction_remaining : ℚ := 2 / 9
noncomputable def number_of_hedgehogs : ℕ := 2

theorem each_hedgehog_ate_1050_strawberries :
  let total_strawberries := number_of_baskets * total_strawberries_per_basket,
      strawberries_remaining := (fraction_remaining * total_strawberries).toNat,
      strawberries_eaten := total_strawberries - strawberries_remaining,
      strawberries_per_hedgehog := strawberries_eaten / number_of_hedgehogs
  in strawberries_per_hedgehog = 1050 := 
by
  -- Proof here
  sorry

end each_hedgehog_ate_1050_strawberries_l603_603827


namespace sequence_diff_l603_603190

theorem sequence_diff (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hSn : ∀ n, S n = n^2)
  (hS1 : a 1 = S 1)
  (ha_n : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 3 - a 2 = 2 := sorry

end sequence_diff_l603_603190


namespace cats_left_l603_603853

theorem cats_left (siamese house sold : ℕ) (h1 : siamese = 12) (h2 : house = 20) (h3 : sold = 20) :  
  (siamese + house) - sold = 12 := 
by
  sorry

end cats_left_l603_603853


namespace boxes_per_hand_l603_603476

theorem boxes_per_hand (total_people : ℕ) (total_boxes : ℕ) (boxes_per_person : ℕ) (hands_per_person : ℕ) 
  (h1: total_people = 10) (h2: total_boxes = 20) (h3: boxes_per_person = total_boxes / total_people) 
  (h4: hands_per_person = 2) : boxes_per_person / hands_per_person = 1 := 
by
  sorry

end boxes_per_hand_l603_603476


namespace original_price_of_cycle_l603_603071

theorem original_price_of_cycle (SP : ℕ) (P : ℕ) (h1 : SP = 1800) (h2 : SP = 9 * P / 10) : P = 2000 :=
by
  have hSP_eq : SP = 1800 := h1
  have hSP_def : SP = 9 * P / 10 := h2
  -- Now we need to combine these to prove P = 2000
  sorry

end original_price_of_cycle_l603_603071


namespace cookie_cost_l603_603745

variables (m o c : ℝ)
variables (H1 : m = 2 * o)
variables (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c)

theorem cookie_cost (H1 : m = 2 * o) (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c) : c = (13 / 4) * o :=
by sorry

end cookie_cost_l603_603745


namespace tan_double_angle_l603_603213

theorem tan_double_angle (α : ℝ) (h : tan α = -2) : tan (2 * α) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l603_603213


namespace compute_sum_g_l603_603517

noncomputable def g (x : ℝ) : ℝ := 5 / (16^x + 5)

theorem compute_sum_g :
  (∑ k in finset.range 2002, g ((k+1) / 2003)) = 1001 :=
begin
  sorry
end

end compute_sum_g_l603_603517


namespace distance_between_points_l603_603668

theorem distance_between_points :
  let l := λ t : ℝ, (x := -2 - 3 * t, y := 2 - 4 * t)
  let C := λ (x y : ℝ), (y - 2)^2 - x^2 = 1
  ∀ t1 t2 : ℝ, C (l t1).1 (l t1).2 → C (l t2).1 (l t2).2 →
  let distance := (3^2 + 4^2)^(1/2) * abs (t1 - t2)
  distance = (10 * real.sqrt 71) / 7 := 
sorry

end distance_between_points_l603_603668


namespace triangle_side_length_l603_603671

open Real

theorem triangle_side_length (PQ PR : ℝ) (QR PM : ℝ) (M : Point)
  (hPQ : PQ = 2) (hPR : PR = 3) (hM : midpoint M Q R) (hPM : PM = QR) :
  QR = Real.sqrt (26 * 0.2) :=
begin
  sorry
end

end triangle_side_length_l603_603671


namespace reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l603_603725

theorem reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs
  (a b c h : Real)
  (area_legs : ℝ := (1 / 2) * a * b)
  (area_hypotenuse : ℝ := (1 / 2) * c * h)
  (eq_areas : a * b = c * h)
  (height_eq : h = a * b / c)
  (pythagorean_theorem : c ^ 2 = a ^ 2 + b ^ 2) :
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 := 
by
  sorry

end reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l603_603725


namespace gopi_servant_salary_l603_603611

theorem gopi_servant_salary (S : ℕ) (turban_price : ℕ) (cash_received : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  turban_price = 70 →
  cash_received = 50 →
  months_worked = 9 →
  total_months = 12 →
  S = 160 :=
by
  sorry

end gopi_servant_salary_l603_603611


namespace problem1_l603_603854

theorem problem1 :
  [((3 + 13/81)^(-3))^(1/6) - log10 (1/100) - (log (sqrt real.exp 1))^(-1) + 0.1^(-2) -
  ((2 + 10/27)^(-2/3)) - (1 / (2 + sqrt 3))^(0) + 2^(-1 - log 2 (1/6))] = 209/2 := by
sorry

end problem1_l603_603854


namespace length_of_train_l603_603885

theorem length_of_train
  (T_platform : ℕ)
  (T_pole : ℕ)
  (L_platform : ℕ)
  (h1: T_platform = 39)
  (h2: T_pole = 18)
  (h3: L_platform = 350)
  (L : ℕ)
  (h4 : 39 * L = 18 * (L + 350)) :
  L = 300 :=
by
  sorry

end length_of_train_l603_603885


namespace percentage_of_ducks_among_non_herons_l603_603646

theorem percentage_of_ducks_among_non_herons
    (ducks : ℕ) (swans : ℕ) (herons : ℕ) (geese : ℕ) -- All values in percentage points
    (h_ducks : ducks = 40)
    (h_swans : swans = 20)
    (h_herons : herons = 15)
    (h_geese : geese = 25) :
    (ducks / (100 - herons) * 100 = 47) :=
begin
  sorry
end

end percentage_of_ducks_among_non_herons_l603_603646


namespace sphere_always_has_circular_cross_section_l603_603027

theorem sphere_always_has_circular_cross_section
    (cone_cross_section_not_always_circular : ∀(H : ∀ (P: Plane), ¬ parallel_to_base P → ¬ is_circle (cross_section Cone P)))
    (cylinder_cross_section_not_always_circular : ∀(H : ∀ (P: Plane), ¬ parallel_to_base P → ¬ is_circle (cross_section Cylinder P)))
    (prism_cross_section_never_circular : ∀(H : ∀ (P: Plane), ¬ is_circle (cross_section Prism P)))
    (sphere_all_circular : ∀ (P: Plane), is_circle (cross_section Sphere P)) :
    (∀ (g : GeometricShape), (g = Cone ∨ g = Cylinder ∨ g = Prism → ∀ (P : Plane), ¬ is_circle (cross_section g P)) ∧
    (g = Sphere → ∀ (P : Plane), is_circle (cross_section g P))) :=
by
    intros g H
    cases H
    case inl =>
        intro P
        cases g
        case Cone => exact cone_cross_section_not_always_circular
        case Cylinder => exact cylinder_cross_section_not_always_circular
        case Prism => exact prism_cross_section_never_circular
    case inr =>
        intro P
        exact sphere_all_circular
  
structure Plane
axiom parallel_to_base : Plane → Prop
axiom cross_section : GeometricShape → Plane → Shape
axiom is_circle : Shape → Prop

inductive GeometricShape
| Cone
| Sphere
| Cylinder
| Prism

end sphere_always_has_circular_cross_section_l603_603027


namespace ginger_cakes_l603_603554

def cakes_per_year (children: ℕ) (cakes_per_child: ℕ) (husband_cakes: ℕ) (parents: ℕ) (cakes_per_parent: ℕ) : ℕ :=
  (children * cakes_per_child) + husband_cakes + (parents * cakes_per_parent)

theorem ginger_cakes (children: ℕ) (cakes_per_child: ℕ) (husband_cakes: ℕ) (parents: ℕ) (cakes_per_parent: ℕ) (years: ℕ) :
  children = 2 → cakes_per_child = 4 → husband_cakes = 6 → parents = 2 → cakes_per_parent = 1 → years = 10 →
  cakes_per_year children cakes_per_child husband_cakes parents cakes_per_parent * years = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  simp [cakes_per_year]
  norm_num
  sorry

end ginger_cakes_l603_603554


namespace problem_solution_l603_603223

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + (2 - Real.exp 1) * x

-- Define the condition for the tangent line at x = 0
def tangent_condition (a : ℝ) : Prop := (f a 0) + (2 - Real.exp 1) = 3 - Real.exp 1

-- Define the function g(x) used in the second proof
def g (x : ℝ) : ℝ := Real.exp x - (x + 1)

-- Main theorem
theorem problem_solution :
  (∃ a : ℝ, tangent_condition a ∧
    (∀ x ≥ 0, f a x ≠ 0) ∧
    (a = 1) ∧
    (∀ x : ℝ, x > 0 → f a x - 1 > x * Real.log (x + 1))
  ) :=
sorry

end problem_solution_l603_603223


namespace rectangle_lines_combinations_l603_603164

theorem rectangle_lines_combinations : 5.choose 2 * 4.choose 2 = 60 := by
  sorry

end rectangle_lines_combinations_l603_603164


namespace sqrt_of_16_l603_603840

theorem sqrt_of_16 : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_16_l603_603840


namespace correct_statements_l603_603441

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

-- Statement A: If a is parallel to b and b is parallel to c, then a is parallel to c.
def statement_A : Prop := 
  (∀ {d e : V}, d ∥ e ↔ ∃ k : ℝ, e = k • d) 
    
-- Statement B: |(a ⋅ b) ⋅ c| ≤ |a||b||c|.
def statement_B : Prop :=
  ∥(a ⬝ b) • c∥ ≤ ∥a∥ * ∥b∥ * ∥c∥

-- Statement C: If a ⊥ (b - c), then a ⋅ b = a ⋅ c.
def statement_C : Prop := 
  a ⊥ (b - c) → (a ⬝ b = a ⬝ c)

-- Statement D: (a ⋅ b) ⋅ b = a ⋅ (b^2).
def statement_D : Prop := 
  (a ⬝ b) • b = a ⬝ (b * b)

-- Prove the correct statements are B and C.
theorem correct_statements :
  statement_B a b c ∧ statement_C a b c :=
by {
  -- Proof omitted
  sorry
}

end correct_statements_l603_603441


namespace simplify_fractions_l603_603735

theorem simplify_fractions : (360 / 32) * (10 / 240) * (16 / 6) = 10 := by
  sorry

end simplify_fractions_l603_603735


namespace proper_subsets_of_B_l603_603606

theorem proper_subsets_of_B (b a : ℝ) 
  (hA : {x : ℝ | x^2 + (b + 2) * x + b + 1 = 0} = {a})
  (hB : ∀ x, x ∈ ({x | x^2 + a * x + b = 0} : set ℝ) ↔ x = 0 ∨ x = 1) :
  ({ ∅, {0}, {1} }: set (set ℝ)) ⊆ (powerset {0, 1} \ {0, 1}) := 
by
  sorry

end proper_subsets_of_B_l603_603606


namespace exists_fixed_point_l603_603372

variable {S : Set ℝ} [Finite S] (f : S → S)

theorem exists_fixed_point (h : ∀ s1 s2 : S, |f s1 - f s2| ≤ (1/2) * |s1 - s2|) : ∃ x : S, f x = x :=
sorry

end exists_fixed_point_l603_603372


namespace analytical_expression_of_f_l603_603586

-- Define properties for even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Given conditions
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 2 * x else x^2 + 2 * x

axiom even_f : is_even f

-- Prove the desired property
theorem analytical_expression_of_f :
  ∀ x, f x = if x > 0 then x^2 + 2 * x else x^2 - 2 * x :=
sorry

end analytical_expression_of_f_l603_603586


namespace events_complementary_l603_603994

def defective_probability (total: ℕ) (defective: ℕ) (selection: ℕ) : Prop :=
  ∃ E F G : set (finset ℕ),
  -- Define event E: all 3 products are non-defective
  E = { s ∈ finset.powerset_len selection (finset.range total) | ∀ x ∈ s, x < total - defective } ∧
  -- Define event F: all 3 products are defective
  F = { s ∈ finset.powerset_len selection (finset.range total) | ∀ x ∈ s, x ≥ total - defective } ∧
  -- Define event G: at least one of the 3 products is defective
  G = { s ∈ finset.powerset_len selection (finset.range total) | ∃ x ∈ s, x ≥ total - defective } ∧
  -- E and G are complementary events
  E ∪ G = finset.powerset_len selection (finset.range total) ∧
  E ∩ G = ∅

theorem events_complementary (total defective selection : ℕ) (h1 : total = 100) (h2 : defective = 5) (h3 : selection = 3) :
  defective_probability total defective selection :=
by
  sorry

end events_complementary_l603_603994


namespace area_of_ellipse_l603_603139

theorem area_of_ellipse (x y : ℝ) (h : x^2 + 6 * x + 4 * y^2 - 8 * y + 9 = 0) : 
  area = 2 * Real.pi :=
sorry

end area_of_ellipse_l603_603139


namespace count_integer_values_satisfying_condition_l603_603802

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l603_603802


namespace range_of_a_l603_603597

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, f x < |a - 1|) : a ∈ set.Ioo (-∞) (-3) ∪ set.Ioo 5 ∞ :=
sorry

end range_of_a_l603_603597


namespace length_first_train_l603_603419

noncomputable def length_second_train : ℝ := 200
noncomputable def speed_first_train_kmh : ℝ := 42
noncomputable def speed_second_train_kmh : ℝ := 30
noncomputable def time_seconds : ℝ := 14.998800095992321

noncomputable def speed_first_train_ms : ℝ := speed_first_train_kmh * 1000 / 3600
noncomputable def speed_second_train_ms : ℝ := speed_second_train_kmh * 1000 / 3600

noncomputable def relative_speed : ℝ := speed_first_train_ms + speed_second_train_ms
noncomputable def combined_length : ℝ := relative_speed * time_seconds

theorem length_first_train : combined_length - length_second_train = 99.9760019198464 :=
by
  sorry

end length_first_train_l603_603419


namespace wizard_safe_combinations_l603_603496

theorem wizard_safe_combinations (herbs crystals unsafe_pairs : ℕ) (h_herbs : herbs = 4) (h_crystals : crystals = 6) (h_unsafe_pairs : unsafe_pairs = 3) :
  herbs * crystals - unsafe_pairs = 21 :=
by
  rw [h_herbs, h_crystals, h_unsafe_pairs]
  rfl

end wizard_safe_combinations_l603_603496


namespace trihedral_angle_exists_l603_603925

open EuclideanGeometry

noncomputable def trihedral_angle (O : Point) : Prop := 
  ∃ A B C : Point, 
  (dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O A) ∧ 
  ∀ X Y Z : Point, 
  (angle O A B < π/2 ∧ angle O B A < π/2 ∧ 
   angle O B C < π/2 ∧ angle O C B < π/2 ∧ 
   angle O A C < π/2 ∧ angle O C A < π/2)

theorem trihedral_angle_exists (O : Point) : trihedral_angle O :=
by sorry

end trihedral_angle_exists_l603_603925


namespace infinite_primes_dividing_polynomial_l603_603543

noncomputable def polynomial_with_integer_coefficients (f : Polynomial ℤ) : Prop :=
  ∃ (a : ℤ) (n : ℕ), 1 ≤ n ∧ f = ∑ i in Finset.range n, Polynomial.C (a ^ i)

theorem infinite_primes_dividing_polynomial (f : Polynomial ℤ) (Hf : polynomial_with_integer_coefficients f) (deg_ge_one : f.degree ≥ 1) :
  ∃! (p : ℕ), nat.prime p ∧ ∃ (n : ℤ), f.eval n ≠ 0 ∧ p ∣ int.nat_abs (f.eval n) :=
sorry

end infinite_primes_dividing_polynomial_l603_603543


namespace center_of_circle_minimum_tangent_length_value_of_m_inequality_proof_l603_603053

open Real

-- Problem I (a)
theorem center_of_circle 
  (C : Type*) [metric_space C] [normed_add_comm_group C] [normed_space ℝ C] 
  (ρ θ : ℝ) (h : ρ = 2 * cos(θ + π/4)) : 
  ∃ x y, (x - sqrt(2) / 2) ^ 2 + (y + sqrt(2) / 2) ^ 2 = 1 :=
by 
  sorry

-- Problem I (b)
theorem minimum_tangent_length 
  (t : ℝ) 
  (x1 y1 : ℝ) 
  (h_line : x1 = sqrt(2)/2 * t ∧ y1 = sqrt(2)/2 * t + 4*sqrt(2)) 
  (h_center : ∃ x y, (x - sqrt(2) / 2) ^ 2 + (y + sqrt(2) / 2) ^ 2 = 1) :
  ∃ len, minimum len (t^2 + 8*t + 40).sqrt :=
by
  sorry

-- Problem II (a)
theorem value_of_m (m x : ℝ) 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f(x) = m - abs(x-2)) 
  (h_sol : ∀ x, f(x + 2) ≥ 0 ↔ -1 ≤ x ∧ x ≤ 1) : 
  m = 1 :=
by 
  sorry

-- Problem II (b)
theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : 1/a + 1/(2*b) + 1/(3*c) = 1) : 
  a + 2*b + 3*c ≥ 9 :=
by 
  sorry

end center_of_circle_minimum_tangent_length_value_of_m_inequality_proof_l603_603053


namespace range_of_a_l603_603209

def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def f (a : ℝ) : ℝ → ℝ := 
  λ x, if x ≥ 1 then a / x else -x + 3 * a

theorem range_of_a (a : ℝ) : 
  monotonic (f a) ↔ a ∈ set.Ici (1 / 2) := 
sorry

end range_of_a_l603_603209


namespace remove_to_maximize_probability_l603_603832

def integer_list : List ℤ := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def remaining_list_after_removal (n : ℤ) : List ℤ := integer_list.erase n

def count_pairs_with_sum (lst : List ℤ) (sum : ℤ) : Nat :=
  lst.filter (λ x => x ≠ sum - x ∧ (sum - x) ∈ lst).length / 2

theorem remove_to_maximize_probability : let n := 8 in
  ∀ l, l = integer_list.erase n → (λ l, count_pairs_with_sum l 16) (remaining_list_after_removal 8) = (λ l, count_pairs_with_sum l 16) l :=
by
  sorry

end remove_to_maximize_probability_l603_603832


namespace coeff_x3_in_expansion_l603_603140

theorem coeff_x3_in_expansion : 
  (coeff (x^3) ((1 - x)^5 * (3 + x)) = -20) :=
by
  sorry

end coeff_x3_in_expansion_l603_603140


namespace train_crosses_pole_in_9_seconds_l603_603491

noncomputable def time_to_cross_pole (speed_kmph : ℕ) (length_m : ℕ) : ℕ :=
  let speed_mps := speed_kmph * 1000 / 3600 in
  length_m / speed_mps

theorem train_crosses_pole_in_9_seconds :
  time_to_cross_pole 36 90 = 9 :=
by
  -- Let speed_kmph := 36
  -- Let length_m := 90
  -- Convert speed to mps: speed_mps = 36 * 1000 / 3600 = 10
  -- Calculate time: time = length_m / speed_mps = 90 / 10 = 9
  sorry

end train_crosses_pole_in_9_seconds_l603_603491


namespace least_number_remainder_l603_603425

theorem least_number_remainder (N k : ℕ) (h : N = 18 * k + 4) : N = 256 :=
by
  sorry

end least_number_remainder_l603_603425


namespace number_of_solutions_l603_603931

theorem number_of_solutions : 
  ∃ (xs : Set ℝ), xs = { x | |x+1| = |x-1| + |x-4| } ∧ xs.card = 2 :=
by
  sorry

end number_of_solutions_l603_603931


namespace find_common_difference_find_range_a1_l603_603207

-- Given conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = n/2 * (2 * a 1 + (n - 1) * (a 1 - a 0))

axiom S4_eq_2S2_plus_4 (a : ℕ → ℝ) (S : ℕ → ℝ) :
S 4 = 2 * S 2 + 4

axiom Sn_ge_S8 (a : ℕ → ℝ) (S : ℕ → ℝ) :
∀ n : ℕ, n > 0 → S n ≥ S 8

-- Proof Problem
theorem find_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  sum_first_n_terms a S →
  S4_eq_2S2_plus_4 a S →
  d = 1 :=
sorry

theorem find_range_a1 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → S n ≥ S 8) →
  sum_first_n_terms a S →
  ∀ n : ℕ, S n = n/2 * (2 * a 1 + (n - 1)) →
  (-8) ≤ a 1 ∧ a 1 ≤ (-7) :=
sorry

end find_common_difference_find_range_a1_l603_603207


namespace find_principal_sum_l603_603758

variable (P : ℝ)
variable (R : ℝ := 0.04)
variable (T : ℕ := 5)
variable (n : ℕ := 2)

-- Define Simple Interest (SI)
def SI := P * R * (T : ℝ) / 100

-- Define Compound Interest (CI)
def CI := P * ((1 + R / (n : ℝ)) ^ (n * T) - 1)

-- Given condition: CI - SI = 1
theorem find_principal_sum (h : CI P R T n - SI P R T = 1) : P = 52.66 :=
sorry

#print axioms find_principal_sum

end find_principal_sum_l603_603758


namespace probability_at_least_seven_stayed_l603_603639

variable (total_people : ℕ)
variable (unsure_people : ℕ)
variable (sure_probability : ℚ)

def at_least_seven_stay_probability : ℚ :=
  (nat.choose 5 3) * (sure_probability^3) * ((1 - sure_probability)^2)
  + (nat.choose 5 4) * (sure_probability^4) * ((1 - sure_probability)^1)
  + (nat.choose 5 5) * (sure_probability^5)

theorem probability_at_least_seven_stayed
  (h1 : total_people = 9)
  (h2 : unsure_people = 5)
  (h3 : sure_probability = 1/3):
  at_least_seven_stay_probability total_people unsure_people sure_probability = 17 / 81 :=
by
  sorry

end probability_at_least_seven_stayed_l603_603639


namespace P_sufficient_but_not_necessary_for_Q_l603_603203

noncomputable theory

variables {a b : ℝ}

def P (a b : ℝ) : Prop := a > b ∧ b > 0
def Q (a b : ℝ) : Prop := a^2 > b^2

theorem P_sufficient_but_not_necessary_for_Q (a b : ℝ) :
  (P a b → Q a b) ∧ (Q a b) ∧ ¬(Q a b → P a b) :=
sorry

end P_sufficient_but_not_necessary_for_Q_l603_603203


namespace perpendicular_line_equation_l603_603762

theorem perpendicular_line_equation 
  (x y : ℝ) 
  (line1 : 2 * x - 3 * y + 4 = 0) 
  (line2 : 3 * x + 2 * y + m = 0) 
  (point : (-1, 2)) :
  ∃ (m : ℝ), 3 * x + 2 * y - 1 = 0 :=
sorry

end perpendicular_line_equation_l603_603762


namespace first_special_saturday_l603_603863

-- Define the structure of the problem with all necessary conditions
def starts_on_monday (y : ℕ) (m : ℕ) (d : ℕ) : Prop :=
  y = 2018 ∧ m = 1 ∧ d = 8

def special_saturday (y : ℕ) (m : ℕ) (d : ℕ) : Prop :=
  d = 31 ∧ m = 3 ∧ y = 2018

theorem first_special_saturday {y m d : ℕ} 
  (club_start : starts_on_monday y m d) (january_days : 31 = 31)
  (february_days : 28 = 28) (march_days : 31 = 31) :
  ∃ y m d, special_saturday y m d :=
begin
  use [2018, 3, 31],
  split, exact rfl,
  split, exact rfl,
  exact rfl,
end

end first_special_saturday_l603_603863


namespace tan_Y_of_right_triangle_l603_603134

theorem tan_Y_of_right_triangle (X Y Z : Point) (hxz : distance X Z = 40)
  (hxy : distance X Y = 30) (hyz : distance Y Z = 50) (right_angle : right_angle_at X Y Z) :
  tan Y = 4 / 3 :=
sorry

end tan_Y_of_right_triangle_l603_603134


namespace necessary_but_not_sufficient_condition_l603_603454

variable {m : ℝ}

theorem necessary_but_not_sufficient_condition (h : (∃ x1 x2 : ℝ, (x1 ≠ 0 ∧ x1 = -x2) ∧ (x1^2 + x1 + m^2 - 1 = 0))): 
  0 < m ∧ m < 1 :=
by 
  sorry

end necessary_but_not_sufficient_condition_l603_603454


namespace value_of_x_l603_603253

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l603_603253


namespace frank_spent_fraction_l603_603961

theorem frank_spent_fraction (F : ℚ) : 
  ( ∃ (F : ℚ), 
    (
      (F * 600 = 600 - (1 - (1 / 4)) * (1 - F) * 600) 
      ∧ (1 - F) * 600 - (1 / 4) * (1 - F) * 600 = 360
      ∧ (600: ℚ) > 0
    )
  ) → F = 1 / 5 :=
by 
  intro h
  cases h with F hF
  sorry 

end frank_spent_fraction_l603_603961


namespace chandra_pairings_l603_603919

theorem chandra_pairings (bowls glasses : ℕ) (h_bowls : bowls = 5) (h_glasses : glasses = 5) : bowls * glasses = 25 := by
  rw [h_bowls, h_glasses]
  exact Nat.mul_comm 5 5
  exact Nat.mul_self 5
  sorry

end chandra_pairings_l603_603919


namespace arithmetic_sequence_sum_l603_603976

theorem arithmetic_sequence_sum
  (a₁ d : ℝ) (n : ℕ)
  (h₁ : ∀ x y : ℝ, (x, y) ∈ { p | (p.1 - 2)^2 + p.2^2 = 1 } → p ∈ { p | p.2 = a₁ * p.1 })
  (h₂ : symmetric_about_line { p : ℝ × ℝ | (p.fst - 2)^2 + p.snd^2 = 1 } { q : ℝ × ℝ | q.2 = a₁ * q.1 } { r : ℝ × ℝ | r.1 + r.2 + d = 0 }) :
  S n = 2 * n - n^2 := sorry

end arithmetic_sequence_sum_l603_603976


namespace smallest_n_power_2013_ends_001_l603_603430

theorem smallest_n_power_2013_ends_001 :
  ∃ n : ℕ, n > 0 ∧ 2013^n % 1000 = 1 ∧ ∀ m : ℕ, m > 0 ∧ 2013^m % 1000 = 1 → n ≤ m := 
sorry

end smallest_n_power_2013_ends_001_l603_603430


namespace combination_pattern_l603_603555

theorem combination_pattern (n : ℕ) :
  (C (4*n+1) 1 + C (4*n+1) 5 + ... + C (4*n+1) (4*n+1)) =
  2^(4*n-1) - 2^(2*n-1) :=
by
  sorry

end combination_pattern_l603_603555


namespace trapezoid_total_area_l603_603495
open Real

theorem trapezoid_total_area (h : ℝ) (h_pos : 0 < h) :
  let lower_base := 4 * h in
  let upper_base := 5 * h in
  let mid_height := h / 2 in
  let imaginary_mid_base := (lower_base + upper_base) / 2 in
  let area_lower := (((4 * h) + (4.5 * h)) / 2) * (h / 2) in
  let area_upper := (((4.5 * h) + (5 * h)) / 2) * (h / 2) in
  let total_area := area_lower + area_upper in
    total_area = (9 * h^2) / 2 :=
by
  sorry

end trapezoid_total_area_l603_603495


namespace count_integer_values_satisfying_condition_l603_603806

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l603_603806


namespace rectangle_lines_combinations_l603_603162

theorem rectangle_lines_combinations : 5.choose 2 * 4.choose 2 = 60 := by
  sorry

end rectangle_lines_combinations_l603_603162


namespace negation_correct_l603_603393

-- Definitions needed from the conditions:
def is_positive (m : ℝ) : Prop := m > 0
def square (m : ℝ) : ℝ := m * m

-- The original proposition
def original_proposition (m : ℝ) : Prop := is_positive m → square m > 0

-- The negation of the proposition
def negated_proposition (m : ℝ) : Prop := ¬is_positive m → ¬(square m > 0)

-- The theorem to prove that the negated proposition is the negation of the original proposition
theorem negation_correct (m : ℝ) : (original_proposition m) ↔ (negated_proposition m) :=
by
  sorry

end negation_correct_l603_603393


namespace rectangle_enclosed_by_lines_l603_603158

theorem rectangle_enclosed_by_lines :
  ∀ (H : ℕ) (V : ℕ), H = 5 → V = 4 → (nat.choose H 2) * (nat.choose V 2) = 60 :=
by
  intros H V h_eq H_eq
  rw [h_eq, H_eq]
  simp
  sorry

end rectangle_enclosed_by_lines_l603_603158


namespace rectangles_from_lines_l603_603153

theorem rectangles_from_lines : 
  (∃ (h_lines : ℕ) (v_lines : ℕ), (h_lines = 5 ∧ v_lines = 4) ∧ (nat.choose h_lines 2 * nat.choose v_lines 2 = 60)) :=
by
  let h_lines := 5
  let v_lines := 4
  have h_choice := nat.choose h_lines 2
  have v_choice := nat.choose v_lines 2
  have answer := h_choice * v_choice
  exact ⟨h_lines, v_lines, ⟨rfl, rfl⟩, rfl⟩

end rectangles_from_lines_l603_603153


namespace angle_between_vectors_l603_603241

variables (a b : ℝ^3)

-- Conditions
def perp_condition : Prop := (a - 2 • b) ⬝ (3 • a + b) = 0
def norm_condition : Prop := ∥a∥ = (1 / 2) * ∥b∥

-- Theorem
theorem angle_between_vectors (h₁ : perp_condition a b) (h₂ : norm_condition a b) :
  ∀ θ : ℝ, θ = real.arccos (-1 / 2) -> θ = 2 * real.pi / 3 :=
by
  sorry

end angle_between_vectors_l603_603241


namespace eight_applications_of_s_l603_603328

def s (θ : ℝ) : ℝ := 1 / (2 - θ)

theorem eight_applications_of_s (θ : ℝ) (h : θ = 50) : 
  s (s (s (s (s (s (s (s θ)))))))) = 50 :=
  sorry

end eight_applications_of_s_l603_603328


namespace function_bounds_l603_603975

theorem function_bounds (f : ℕ+ → ℕ+) (k : ℕ+) (strict_increasing : ∀ m n : ℕ+, m < n → f m < f n)
  (functional_eq : ∀ n : ℕ+, f (f n) = k * n) (n : ℕ+) :
  (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 :=
by
  -- skipping the proof
  sorry

end function_bounds_l603_603975


namespace increasing_function_solution_l603_603944

noncomputable def solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y

theorem increasing_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x + y) * (f x + f y) = f x * f y)
  ∧ (∀ x y : ℝ, x < y → f x < f y)
  → ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = 1 / (a * x) :=
by {
  sorry
}

end increasing_function_solution_l603_603944


namespace find_x_l603_603256

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l603_603256


namespace greatest_AB_CBA_div_by_11_l603_603876

noncomputable def AB_CBA_max_value (A B C : ℕ) : ℕ := 10001 * A + 1010 * B + 100 * C + 10 * B + A

theorem greatest_AB_CBA_div_by_11 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  2 * A - 2 * B + C % 11 = 0 ∧ 
  ∀ (A' B' C' : ℕ),
    A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
    2 * A' - 2 * B' + C' % 11 = 0 → 
    AB_CBA_max_value A B C ≥ AB_CBA_max_value A' B' C' :=
  by sorry

end greatest_AB_CBA_div_by_11_l603_603876


namespace sum_of_digits_base_2_of_300_l603_603011

theorem sum_of_digits_base_2_of_300 : 
  let n := 300
  let binary_representation := nat.binary_repr n
  nat.digits_sum 2 binary_representation = 4 :=
by
  let n := 300
  let binary_representation := nat.binary_repr n
  have h1 : binary_representation = [1,0,0,1,0,1,1,0,0] := sorry
  have h2 : nat.digits_sum 2 binary_representation = 1+0+0+1+0+1+1+0+0 := sorry
  show nat.digits_sum 2 binary_representation = 4 from by sorry

end sum_of_digits_base_2_of_300_l603_603011


namespace intersect_sets_l603_603237

open Set

noncomputable def P : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}

theorem intersect_sets (U : Set ℝ) (P : Set ℝ) (Q : Set ℝ) :
  U = univ → P = {x : ℝ | x^2 - 2 * x ≤ 0} → Q = {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x} →
  P ∩ Q = Icc (0 : ℝ) (2 : ℝ) :=
by
  intros
  sorry

end intersect_sets_l603_603237


namespace hyperbola_eccentricity_l603_603988

variables (a b : ℝ) (x y : ℝ) (PF1 PF2 F1 F2 P : ℝ)

theorem hyperbola_eccentricity 
  (ha : a > 0)
  (hb : b > 0)
  (hP : (x/a)^2 - (y/b)^2 = 1)
  (hF : PF1^2 + PF2^2 = (F1F2)^2)
  (hp_orthogonal : PF1 ⊥ PF2)
  (h_arithmetic : (PF2 - PF1) + PF1 = F1F2) :
  eccentricity (hyperbola a b) = 5 := 
sorry

end hyperbola_eccentricity_l603_603988


namespace percentage_rotten_oranges_l603_603879

-- Conditions
def total_fruits := 600 + 400
def rotten_bananas := 0.04 * 400
def good_fruits := 0.894 * total_fruits

-- Question and its expected answer
theorem percentage_rotten_oranges
  (total_fruits = 1000 : Prop)
  (rotten_bananas = 16 : Prop)
  (good_fruits = 894 : Prop)
  (total_rotten_fruits := total_fruits - good_fruits)
  (rotten_oranges := total_rotten_fruits - rotten_bananas) :
  ((rotten_oranges / 600) * 100 = 15) := sorry

end percentage_rotten_oranges_l603_603879


namespace students_not_in_biology_l603_603445

theorem students_not_in_biology (total_students : ℕ) (percentage_in_biology : ℚ) 
  (h1 : total_students = 880) (h2 : percentage_in_biology = 27.5 / 100) : 
  total_students - (total_students * percentage_in_biology) = 638 := 
by
  sorry

end students_not_in_biology_l603_603445


namespace range_of_a_for_three_distinct_real_roots_l603_603221

theorem range_of_a_for_three_distinct_real_roots (a : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x, f x = x^3 - 3*x^2 - a ∧ ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end range_of_a_for_three_distinct_real_roots_l603_603221


namespace cubic_poly_real_roots_l603_603866

theorem cubic_poly_real_roots (a b c d : ℝ) (h : a ≠ 0) : 
  ∃ (min_roots max_roots : ℕ), 1 ≤ min_roots ∧ max_roots ≤ 3 ∧ min_roots = 1 ∧ max_roots = 3 :=
by
  sorry

end cubic_poly_real_roots_l603_603866


namespace probability_of_point_closer_to_center_l603_603063

noncomputable def probability_point_closer_to_center (r_outer r_inner : ℝ) (h_radius : r_outer = 4) (h_inner_radius : r_inner = 1) : ℝ :=
  let area_outer := real.pi * (r_outer ^ 2)
  let area_inner := real.pi * (r_inner ^ 2)
  area_inner / area_outer

theorem probability_of_point_closer_to_center
  (r_outer r_inner : ℝ)
  (h_radius : r_outer = 4)
  (h_inner_radius : r_inner = 1)
  : probability_point_closer_to_center r_outer r_inner h_radius h_inner_radius = 1 / 16 :=
by
    sorry

end probability_of_point_closer_to_center_l603_603063


namespace fabric_price_l603_603467

theorem fabric_price (x: ℕ) : ∃ y, y = 8.3 * x :=
by
  sorry

end fabric_price_l603_603467


namespace red_quadrilaterals_equal_area_area_one_red_quadrilateral_l603_603674

theorem red_quadrilaterals_equal_area (A A₀ B₀ C₀ C₁ B₁: Point)
    (blue_triangle_areas_equal : ∀ (A A₀ B₀ : Point), area A A₀ B₀ = 1)
    : are_equal_in_area [A, B₀, A₀, B₁] [A₀, C₀, C₁, B₀] ∧
      are_equal_in_area [B, C₀, C₁, B₁] :=
  sorry

theorem area_one_red_quadrilateral (A A₀ B₀ C₀ C₁ B₁ : Point)
    (blue_triangle_areas_equal : ∀ (A A₀ B₀ : Point), area A A₀ B₀ = 1)
    : ∃ (s : ℝ), s = 1 + sqrt 5 ∧ area_four_sided A B₀ A₀ B₁ = s :=
  sorry

end red_quadrilaterals_equal_area_area_one_red_quadrilateral_l603_603674


namespace find_x_l603_603257

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l603_603257


namespace sin_cos_tangent_relation_l603_603217

variable (A B C : ℝ)

-- Definitions based on the given conditions
def condition1 := cos A = sin B
def condition2 := sin B = 2 * tan (C / 2)
def condition3 := A + B + C = π

-- Proof statement
theorem sin_cos_tangent_relation
  (h1 : condition1 A B C)
  (h2 : condition2 B C)
  (h3 : condition3 A B C) : sin A + cos A + 2 * tan A = 2 :=
by
  sorry

end sin_cos_tangent_relation_l603_603217


namespace determine_all_tables_l603_603400

noncomputable def initial_table : matrix (fin 4) (fin 4) char :=
![![ 'A', 'B', 'C', 'D' ],
  ![ 'D', 'C', 'B', 'A' ],
  ![ 'C', 'A', 'C', 'A' ],
  ![ 'B', 'D', 'B', 'D' ]]

noncomputable def result_table : matrix (fin 4) (fin 4) char :=
![![ 'C', 'D', 'A', 'B' ],
  ![ 'B', 'A', 'D', 'C' ],
  ![ 'A', 'C', 'A', 'C' ],
  ![ 'D', 'B', 'D', 'B' ]]

-- Predicate to check the quadrant consistency
def is_valid_transformation
  (T1 T2 : matrix (fin 4) (fin 4) char) : Prop :=
     (T1 0 0 = T2 0 0 ∧ 
      T1 0 1 = T2 0 1 ∧ 
      T1 1 0 = T2 1 0 ∧ 
      T1 1 1 = T2 1 1) ∧
     (T1 2 2 = T2 2 2 ∧
      T1 2 3 = T2 2 3 ∧
      T1 3 2 = T2 3 2 ∧
      T1 3 3 = T2 3 3)

theorem determine_all_tables :
  is_valid_transformation initial_table result_table ∧
  result_table = ![![ 'C', 'D', 'A', 'B' ],
                     ![ 'B', 'A', 'D', 'C' ],
                     ![ 'A', 'C', 'A', 'C' ],
                     ![ 'D', 'B', 'D', 'B' ]] ∧
  is_valid_transformation initial_table initial_table :=
  sorry

end determine_all_tables_l603_603400


namespace range_of_m_l603_603992

theorem range_of_m (m : ℝ) (h : (8 - m) / (m - 5) > 1) : 5 < m ∧ m < 13 / 2 :=
by
  sorry

end range_of_m_l603_603992


namespace tangent_line_eq_2x_minus_1_at_1_1_l603_603387

theorem tangent_line_eq_2x_minus_1_at_1_1 :
  ∀ (x : ℝ), ∀ (f : ℝ → ℝ), f x = x * real.exp (x - 1) → 
  (f 1 = 1) → ∃ (m b : ℝ), (m = 2) ∧ (b = -1) ∧ (∀ (x : ℝ), f x = m * x + b) :=
by
  sorry

end tangent_line_eq_2x_minus_1_at_1_1_l603_603387


namespace tangent_line_normal_line_l603_603141

noncomputable def x (t : ℝ) : ℝ := 2 * Real.tan t
noncomputable def y (t : ℝ) : ℝ := 2 * Real.sin t ^ 2 + Real.sin (2 * t)
def t₀ : ℝ := Real.pi / 4

def pt_x₀ : ℝ := x t₀
def pt_y₀ : ℝ := y t₀

noncomputable def dx_dt (t : ℝ) : ℝ := 2 / (Real.cos t ^ 2)
noncomputable def dy_dt (t : ℝ) : ℝ := 4 * Real.sin t * Real.cos t + 2 * Real.cos (2 * t)

def slope_tangent (t : ℝ) : ℝ := (dy_dt t) / (dx_dt t)

def tangent_line_eq (x : ℝ) : ℝ := slope_tangent t₀ * (x - pt_x₀) + pt_y₀
def normal_line_eq (x : ℝ) : ℝ := -1 / (slope_tangent t₀) * (x - pt_x₀) + pt_y₀

theorem tangent_line : tangent_line_eq = λ x, (1/2) * x + 1 := sorry

theorem normal_line : normal_line_eq = λ x, -2 * x + 6 := sorry

end tangent_line_normal_line_l603_603141


namespace same_solution_sets_l603_603277

theorem same_solution_sets (a : ℝ) :
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := 
by
  sorry

end same_solution_sets_l603_603277


namespace magnitude_z_of_equation_l603_603562

noncomputable def magnitude_of_z : ℂ → ℝ := sorry
-- Given: a complex number z satisfying z * (1 - i)^2 = 1 + i
-- Prove that the magnitude of z is sqrt(2)/2
theorem magnitude_z_of_equation (z : ℂ) (h : z * (1 - i)^2 = 1 + i) : abs z = real.sqrt 2 / 2 := sorry

end magnitude_z_of_equation_l603_603562


namespace integer_solutions_count_l603_603784

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l603_603784


namespace train_speed_kmh_l603_603490

def man_speed_kmh : ℝ := 3 -- The man's speed in km/h
def train_length_m : ℝ := 110 -- The train's length in meters
def passing_time_s : ℝ := 12 -- Time taken to pass the man in seconds

noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600 -- Convert man's speed to m/s

theorem train_speed_kmh :
  (110 / 12) - (5 / 6) * (3600 / 1000) = 30 := by
  -- Omitted steps will go here
  sorry

end train_speed_kmh_l603_603490


namespace clover_walk_distance_l603_603920

theorem clover_walk_distance (total_distance days walks_per_day : ℝ) (h1 : total_distance = 90) (h2 : days = 30) (h3 : walks_per_day = 2) :
  (total_distance / days / walks_per_day = 1.5) :=
by
  sorry

end clover_walk_distance_l603_603920


namespace magnitude_of_vector_a_l603_603568

def vector_a : ℝ × ℝ × ℝ := (1, 2, -1)

theorem magnitude_of_vector_a :
  let magnitude := (vector_a.1^2 + vector_a.2^2 + vector_a.3^2).sqrt
  magnitude = Real.sqrt 6 := 
by
  sorry

end magnitude_of_vector_a_l603_603568


namespace angle_DOB_is_90_degrees_l603_603541

theorem angle_DOB_is_90_degrees :
  ∀ (x : ℝ), 
  let AOB := x
  let COB := 2 * AOB
  let DOC := 3 * AOB
  let EOD := 2 * COB
  (AOB + COB + DOC + EOD = 180) →
  (∠ AOB = x ∧ ∠ COB = 2 * x ∧ ∠ DOC = 3 * x ∧ ∠ EOD = 4 * x) →
  (x = 18) →
  (∠ DOB = ∠ DOC + ∠ COB) →
  ∠ DOB = 90 :=
by sorry

end angle_DOB_is_90_degrees_l603_603541


namespace find_y_l603_603534

theorem find_y (y : ℝ) (h₁ : (y^2 - 7*y + 12) / (y - 3) + (3*y^2 + 5*y - 8) / (3*y - 1) = -8) : y = -6 :=
sorry

end find_y_l603_603534


namespace determine_d_l603_603123

theorem determine_d (d : ℝ) : (∀ x : ℝ, x ∈ Ioo (-4 : ℝ) 1 → x * (2 * x + 4) < d) ↔ d = 8 :=
by
  sorry

end determine_d_l603_603123


namespace lcm_9_12_15_l603_603001

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l603_603001


namespace solve_equation_l603_603738

theorem solve_equation (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + y * z + z * x - x * y * z = 2) ↔ 
  ({x, y, z} = {1, 1, 1} ∨ {x, y, z} ∈ {2, 3, 4}) :=
sorry

end solve_equation_l603_603738


namespace G_a_subgroup_l603_603318

variables {G : Type*} [group G] (a : G)

def G_a : set G := {x | ∃ n : ℤ, x = a ^ n}

theorem G_a_subgroup : is_subgroup (G_a a) :=
by {
  sorry
}

end G_a_subgroup_l603_603318


namespace hyperbola_problem_l603_603869

noncomputable def is_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) - ((y - 2)^2 / b^2) = 1

variables (s : ℝ)

theorem hyperbola_problem
  (h₁ : is_hyperbola 0 5 a b)
  (h₂ : is_hyperbola (-1) 6 a b)
  (h₃ : is_hyperbola s 3 a b)
  (hb : b^2 = 9)
  (ha : a^2 = 9 / 25) :
  s^2 = 2 / 5 :=
sorry

end hyperbola_problem_l603_603869


namespace solve_for_nabla_l603_603622

theorem solve_for_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by
  sorry

end solve_for_nabla_l603_603622


namespace factorization_l603_603532

theorem factorization (m : ℝ) : m^2 - 3 * m = m * (m - 3) :=
by sorry

end factorization_l603_603532


namespace evaluate_ff_l603_603230

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else 3 ^ x

theorem evaluate_ff (h : f (f (1 / 2)) = 1 / 3) : f (f (1 / 2)) = 1 / 3 := 
by
  sorry

end evaluate_ff_l603_603230


namespace candles_ratio_l603_603092

def totalCandles : ℕ := 40
def remainingCandles : ℕ := 6
def chelseaUsagePercent : ℝ := 0.7

theorem candles_ratio (A : ℕ) (hA : A + chelseaUsagePercent * (totalCandles - A) = 
                                            totalCandles - remainingCandles) :
  A.to_real / totalCandles.to_real = 1 / 2 := by
  sorry

end candles_ratio_l603_603092


namespace sides_of_DBE_are_one_third_l603_603852

open EuclideanGeometry

noncomputable def side_lengths_one_third (A B C D E : Point) : Prop :=
  let AB := dist A B;
  let BC := dist B C;
  let AC := dist A C;
  let DB := dist D B;
  let DE := dist D E;
  let BE := dist B E
  in
  DB = AB / 3 ∧ BE = BC / 3 ∧ DE = AC / 3

theorem sides_of_DBE_are_one_third (A B C D E : Point)
  (h1 : divides A D B (2 / 3))
  (h2 : parallel_for_line_through D AC E BC) :
  side_lengths_one_third A B C D E :=
by
  sorry

end sides_of_DBE_are_one_third_l603_603852


namespace sum_of_digits_base2_300_l603_603009

theorem sum_of_digits_base2_300 : 
  let n := 300
  in (Int.digits 2 n).sum = 4 :=
by
  let n := 300
  have h : Int.digits 2 n = [1, 0, 0, 1, 0, 1, 1, 0, 0] := by sorry
  rw h
  norm_num
  -- or directly
  -- exact rfl

end sum_of_digits_base2_300_l603_603009


namespace proof_problem_l603_603602

noncomputable def f (x : ℝ) := real.cos x + real.log x / real.log 2

theorem proof_problem
  (a : ℝ) (h_pos : 0 < a) (h_eq : f a = f (2 * a)) :
  f (2 * a) - f (4 * a) = -1 :=
sorry

end proof_problem_l603_603602


namespace rabbit_start_moving_away_point_l603_603469

noncomputable def fox : (ℝ × ℝ) := (10, 8)
noncomputable def rabbit_line : ℝ → ℝ := λ x, -3 * x + 14

theorem rabbit_start_moving_away_point :
    ∃ (c d : ℝ), 
        (c, d) = (2.8, 5.6) ∧ 
        ∃ m₁ m₂ : ℝ, m₁ * m₂ = -1 ∧
        (d - 8) = (m₂) * (c - 10) ∧
        (d = rabbit_line c) ∧
        (c + d = 8.4) := 
sorry

end rabbit_start_moving_away_point_l603_603469


namespace no_14_consecutive_integers_divisible_by_primes_2_to_11_exists_21_consecutive_integers_divisible_by_primes_2_to_13_l603_603051

-- Problem 1
theorem no_14_consecutive_integers_divisible_by_primes_2_to_11 :
  ¬ ∃ (N : ℕ), ∀ i ∈ (list.range' N 14),
    (∃ p ∈ [2, 3, 5, 7, 11], p ∣ i) :=
sorry

-- Problem 2
theorem exists_21_consecutive_integers_divisible_by_primes_2_to_13 :
  ∃ (N : ℕ), ∀ i ∈ (list.range' N 21),
    (∃ p ∈ [2, 3, 5, 7, 11, 13], p ∣ i) :=
sorry

end no_14_consecutive_integers_divisible_by_primes_2_to_11_exists_21_consecutive_integers_divisible_by_primes_2_to_13_l603_603051


namespace identify_factoring_correct_option_l603_603401

theorem identify_factoring_correct_option (x a b : ℝ) :
  (x^2 - x - 2 ≠ x * (x - 1) - 2) ∧
  ((a + b) * (a - b) = a^2 - b^2) ∧
  (x^2 - 4 = (x + 2) * (x - 2)) ∧
  (x - 1 ≠ x * (1 - 1/x)) → 
  (x^2 - 4 = (x + 2) * (x - 2)) = true := 
begin
  sorry -- Proof not required
end

end identify_factoring_correct_option_l603_603401


namespace power_of_complex_fraction_l603_603987

theorem power_of_complex_fraction (i : ℂ) (h : i = complex.I) :
  (complex.div (1 + i) (1 - i)) ^ 2013 = i :=
by sorry

end power_of_complex_fraction_l603_603987


namespace percent_increase_hypotenuse_l603_603097

theorem percent_increase_hypotenuse :
  let l1 := 3
  let l2 := 1.25 * l1
  let l3 := 1.25 * l2
  let l4 := 1.25 * l3
  let h1 := l1 * Real.sqrt 2
  let h4 := l4 * Real.sqrt 2
  ((h4 - h1) / h1) * 100 = 95.3 :=
by
  sorry

end percent_increase_hypotenuse_l603_603097


namespace total_profit_correct_l603_603056

-- Define the initial investments and changes
def initial_investment_A := 3000
def initial_investment_B := 4000
def months_initial := 8
def months_remaining := 4
def change_A := -1000
def change_B := 1000

-- Define A's share of profit
def A_share := 240

-- Define investment-months calculations
def investment_months_A := initial_investment_A * months_initial + (initial_investment_A + change_A) * months_remaining
def investment_months_B := initial_investment_B * months_initial + (initial_investment_B + change_B) * months_remaining

-- Define the profit ratio
def ratio_A_B : ℕ × ℕ := (investment_months_A, investment_months_B)
def total_parts := ratio_A_B.1 + ratio_A_B.2

-- Define the total profit
def total_profit : ℝ := (A_share * total_parts) / ratio_A_B.1

theorem total_profit_correct :
  total_profit = 630 := sorry

end total_profit_correct_l603_603056


namespace logically_follows_l603_603348

-- Define the predicates P and Q
variables {Student : Type} {P Q : Student → Prop}

-- The given condition
axiom Turner_statement : ∀ (x : Student), P x → Q x

-- The statement that necessarily follows
theorem logically_follows : (∀ (x : Student), ¬ Q x → ¬ P x) :=
sorry

end logically_follows_l603_603348


namespace hyperbola_passing_through_parabola_focus_l603_603293

noncomputable def hyperbola_eccentricity : ℝ := 
  let a : ℝ := 2
  let c : ℝ := Real.sqrt (a^2 + a^2)
  c / a

theorem hyperbola_passing_through_parabola_focus :
  ∀ (a : ℝ), a > 0 →
  ∃ e : ℝ, e = Real.sqrt 2 → 
  ∃ x y : ℝ, (x, y) = (2, 0) ∧ (x^2 / a^2 - y^2 = 1) :=
begin
  sorry
end

end hyperbola_passing_through_parabola_focus_l603_603293


namespace factorization_l603_603533

theorem factorization (m : ℝ) : m^2 - 3 * m = m * (m - 3) :=
by sorry

end factorization_l603_603533


namespace millet_constitutes_more_than_half_l603_603345

theorem millet_constitutes_more_than_half :
  ∃ (n : ℕ), n = 3 ∧
  let seeds_on_day := λ n : ℕ, if n = 1 then 0.3 else if n = 2 then 0.25 else if n = 3 then 0.2 else 0.15 in
  let eaten_by_birds := λ n : ℕ, 0.25 - (n-1) * 0.05 in
  let total_millet := λ n : ℕ, (seeds_on_day 1) * (1 - eaten_by_birds 1) + seeds_on_day 2 + (if n ≥ 2 then seeds_on_day 2 * (1 - eaten_by_birds 2) else 0) + (if n ≥ 3 then seeds_on_day 3 * (1 - eaten_by_birds 3) else 0) in
  total_millet 3 > 0.5 :=
begin
  -- proof to be filled
  sorry
end

end millet_constitutes_more_than_half_l603_603345


namespace graph_of_abs_f_is_D_l603_603767

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Assuming f(x) is 0 outside the specified intervals

noncomputable def g_D (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then 2 + x
  else if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- Assuming g_D(x) is 0 outside the specified intervals

theorem graph_of_abs_f_is_D : 
  ∀ x : ℝ, |f x| = g_D x :=
by
  intro x
  cases' lt_or_ge (f x) 0 with h h
  · rw [abs_of_neg h, g_D]
    split_ifs
    sorry
  · rw [abs_of_nonneg h, g_D]
    split_ifs
    sorry


end graph_of_abs_f_is_D_l603_603767


namespace tip_calculation_l603_603553

def pizza_price : ℤ := 10
def number_of_pizzas : ℤ := 4
def total_pizza_cost := pizza_price * number_of_pizzas
def bill_given : ℤ := 50
def change_received : ℤ := 5
def total_spent := bill_given - change_received
def tip_given := total_spent - total_pizza_cost

theorem tip_calculation : tip_given = 5 :=
by
  -- skipping the proof
  sorry

end tip_calculation_l603_603553


namespace num_int_values_x_l603_603794

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l603_603794


namespace female_users_selected_correct_prob_one_male_correct_chi_squared_test_correct_liking_disliking_related_to_gender_l603_603895

noncomputable def num_female_users_selected : ℕ :=
  let total_dislikes := 27 + 18
  let female_proportion := 18 / total_dislikes
  round (female_proportion * 5)

theorem female_users_selected_correct : 
  num_female_users_selected = 2 :=
by
  -- Lean code to assert round ((18/45) * 5) ≈ 2
  sorry

noncomputable def prob_exactly_one_male : ℚ :=
  let total_pairs := 10
  let pairs_with_exactly_one_male := 6
  pairs_with_exactly_one_male / total_pairs

theorem prob_one_male_correct :
  prob_exactly_one_male = 3 / 5 :=
by
  -- Lean code to assert (6 / 10) = 3 / 5
  sorry

noncomputable def chi_squared_value : ℚ :=
  let n := 100
  let a := 13
  let b := 27
  let c := 42
  let d := 18
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_test_correct :
  chi_squared_value > 6.635 :=
by
  -- Lean code to assert chi_squared_value ≈ 13.64 > 6.635
  sorry

theorem liking_disliking_related_to_gender :
  num_female_users_selected = 2 ∧ 
  prob_exactly_one_male = 3 / 5 ∧ 
  chi_squared_value > 6.635 :=
by
  apply and.intro
  apply female_users_selected_correct
  apply and.intro
  apply prob_one_male_correct
  apply chi_squared_test_correct

end female_users_selected_correct_prob_one_male_correct_chi_squared_test_correct_liking_disliking_related_to_gender_l603_603895


namespace value_of_5_minus_y_star_when_x_is_5_l603_603958

def y (x : ℝ) : ℝ := x^2 - 3 * x + 7

def y_star (y : ℝ) : ℝ := 
  if y > 1 then floor (y / 2) * 2 else 0  -- Computes the greatest even integer <= y

theorem value_of_5_minus_y_star_when_x_is_5 : 5.0 - y_star (y 5) = -11 := by
  sorry

end value_of_5_minus_y_star_when_x_is_5_l603_603958


namespace find_y_l603_603244

theorem find_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 :=
by
  sorry

end find_y_l603_603244


namespace integer_satisfying_values_l603_603795

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l603_603795


namespace sum_f_values_l603_603598

noncomputable def A : ℝ := 2
noncomputable def w : ℝ := π / 4
noncomputable def φ : ℝ := π / 4

def f (x : ℝ) : ℝ := A * cos(wx + φ)^2 + 1

theorem sum_f_values :
  A > 0 ∧ w > 0 ∧ 0 < φ ∧ φ < π / 2 ∧ 
  (∀ x, f(x) ≤ 3) ∧
  f(0) = 2 ∧ 
  (∃ P, f(x + P) = f(x) ∧ P = 2) →
  ∑ i in (finset.range 2018).map (finset.nat_cast), f(i) = 4035 :=
by sorry

end sum_f_values_l603_603598


namespace area_of_triangle_l603_603240

open Real

def a := (4, -1)
def b := (1, 3)
def c := (2, 1)

def area_triangle (a b : ℝ × ℝ) : ℝ :=
  let det := (a.1 * b.2 - a.2 * b.1) in
  (1/2) * abs det

theorem area_of_triangle :
  area_triangle a b = 6.5 :=
by
  sorry

end area_of_triangle_l603_603240


namespace general_term_a_n_product_b_n_l603_603590

-- First part: General term formula for the sequence {a_n}
theorem general_term_a_n (a : Real) (n : Nat) (hn : n ≥ 1) :
  let S_n := ln (n + 1) - a
  let a_n := if a = 0 then ln ((n + 1: Real) / n) else 
    if n = 1 then ln 2 - a else ln ((n + 1: Real) / n)
  true := sorry

-- Second part: Sum of b_n terms where b_n = exp(a_n)
theorem product_b_n (a : Real) (n : Nat) (hn : n ≥ 1) :
  let S_n := ln (n + 1) - a
  let a_n := if a = 0 then ln ((n + 1: Real) / n) else 
    if n = 1 then ln 2 - a else ln ((n + 1: Real) / n)
  let b_n := exp(a_n)
  let product := ∏ k in Finset.range n, b_n
  (if a = 0 then product = (n + 1: Real) else 
   product = (n + 1: Real) / exp(a)) := sorry

end general_term_a_n_product_b_n_l603_603590


namespace coloring_of_convex_polygon_l603_603515

theorem coloring_of_convex_polygon (n : ℕ) (h : n ≥ 2) :
  ∀ (polygon : Type) [convex_polygon polygon (2 * n + 1)],
    (∀ adjacent_vertices (v1 v2 : polygon), v1 ≠ v2) →
    ∃ (triangulation : list (triangle polygon)),
      (∀ (t : triangle polygon) (v1 v2 : polygon),
         diagonal_in_triangle t v1 v2 → v1 ≠ v2) :=
by sorry

end coloring_of_convex_polygon_l603_603515


namespace positive_integer_pairs_l603_603137

theorem positive_integer_pairs (t : ℕ) (ht : 0 < t) :
  (let x := 2 * t in x^2 ∣ 2 * x * 1^2 - 1^3 + 1) ∧
  (let x := t, y := 2 * t in x^2 ∣ 2 * x * y^2 - y^3 + 1) ∧
  (let x := 8 * t^4 - t, y := 2 * t in x^2 ∣ 2 * x * y^2 - y^3 + 1) :=
by
  sorry

end positive_integer_pairs_l603_603137


namespace kite_area_correct_l603_603946

noncomputable def area_of_kite (K L M N : Type) 
  [MetricSpace K] [MetricSpace L] [MetricSpace M] [MetricSpace N]
  (KL : ℝ) (LM : ℝ) (MN : ℝ) (NK : ℝ) 
  (R₁ : ℝ) (R₂ : ℝ)
  (h1 : KL = LM) (h2 : LM = MN) (h3 : MN = NK) 
  (h4 : R₁ = 8) (h5 : R₂ = 16) : ℝ :=
  if h1 : KL = 8 ∧ h5 : R₂ = 16 ∧ h3 : NK = KL then 256 else 0

theorem kite_area_correct (K L M N : Type) 
  [MetricSpace K] [MetricSpace L] [MetricSpace M] [MetricSpace N]
  (KL : ℝ) (LM : ℝ) (MN : ℝ) (NK : ℝ)
  (R₁ : ℝ) (R₂ : ℝ) 
  (h1 : KL = LM)
  (h2 : LM = MN)
  (h3 : MN = NK)
  (h4 : R₁ = 8)
  (h5 : R₂ = 16) : 
  area_of_kite K L M N KL LM MN NK R₁ R₂ h1 h2 h3 h4 h5 = 256 := 
sorry

end kite_area_correct_l603_603946


namespace large_circle_diameter_proof_l603_603128

noncomputable def large_circle_diameter (r: ℝ) (count: ℕ) : ℝ :=
  let s := 2 * r
  s / (2 * Real.sin (Real.pi / count))

theorem large_circle_diameter_proof :
  large_circle_diameter 4 8 ≈ 20.94 :=
by
  let r := 4
  let count := 8
  let diameter := 2 * large_circle_diameter r count
  have : diameter ≈ 10.47 * 2 :=
    by sorry
  exact this

end large_circle_diameter_proof_l603_603128


namespace cos_sin_ratio_l603_603206

theorem cos_sin_ratio (x : ℝ) (h : cos x - sin x = 3 * sqrt 2 / 5) : 
  cos (2 * x) / sin (x + π / 4) = 6 / 5 :=
by
  sorry

end cos_sin_ratio_l603_603206


namespace root_of_quadratic_equation_l603_603456

def quadratic_formula (a b c : ℝ) : Set ℝ := 
  {x | x = (-b + real.sqrt (b^2 - 4*a*c))/(2*a) ∨ x = (-b - real.sqrt (b^2 - 4*a*c))/(2*a)}

theorem root_of_quadratic_equation :
  quadratic_formula 3 5 (-1) = {x | x = (-5 + real.sqrt ((5^2) + 4 * 3 * 1))/ (2 * 3) ∨ 
                                        x = (-5 - real.sqrt ((5^2) + 4 * 3 * 1))/ (2 * 3)} :=
sorry

end root_of_quadratic_equation_l603_603456


namespace integer_values_satisfying_sqrt_inequality_l603_603814

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l603_603814


namespace median_of_combined_sequence_l603_603427

-- Definitions based on problem conditions
def sequence_length : ℕ := 4060
def integers_upto_n : list ℕ := list.range 2031 -- integers from 1 to 2030
def squares_upto_n : list ℕ := list.map (λ n, n * n) (list.range 2031) -- squares from 1^2 to 2030^2
def combined_sequence : list ℕ := integers_upto_n ++ squares_upto_n

-- Median calculation
def median (l : list ℕ) : ℚ := 
  if h : l.length % 2 = 0 then 
    ((l.nth_le (l.length / 2 - 1) (sorry)) + (l.nth_le (l.length / 2) (sorry))) / 2
  else 
    l.nth_le (l.length / 2) (sorry)

-- The proof problem to be stated
theorem median_of_combined_sequence : median combined_sequence = 1015.5 := sorry

end median_of_combined_sequence_l603_603427


namespace tan_alpha_eq_neg_sqrt_3_l603_603569

noncomputable def x : ℝ := - (real.sqrt 3) / 3

theorem tan_alpha_eq_neg_sqrt_3 (α : ℝ) (P : ℝ × ℝ) (hP : P.1 = x ∧ P.2 = 1) 
  (hcos : real.cos α = -1 / 2): 
  real.tan α = - real.sqrt 3 :=
sorry

end tan_alpha_eq_neg_sqrt_3_l603_603569


namespace geom_sequence_ratio_l603_603297

-- Definitions and assumptions for the problem
noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_ratio (a : ℕ → ℝ) (r : ℝ) 
  (h_geom: geom_seq a)
  (h_r: 0 < r ∧ r < 1)
  (h_seq: ∀ n : ℕ, a (n + 1) = a n * r)
  (ha1: a 7 * a 14 = 6)
  (ha2: a 4 + a 17 = 5) :
  (a 5 / a 18) = (3 / 2) :=
sorry

end geom_sequence_ratio_l603_603297


namespace meals_initially_available_for_adults_l603_603471

theorem meals_initially_available_for_adults (A C : ℕ) (hC : C = 90) (h_equiv : A - 35 = C - 45) :
  A = 80 :=
by {
  rw hC at h_equiv,
  simp at h_equiv,
  exact h_equiv,
}

end meals_initially_available_for_adults_l603_603471


namespace Acme_Vowel_Soup_Sequences_l603_603896

theorem Acme_Vowel_Soup_Sequences:
  let num_vowels := 5  -- Number of vowels (A, E, I, O, U)
  let positions := 4  -- Number of positions in the sequence
  (num_vowels ^ positions) = 625 := 
by 
  have num_vowels_def : num_vowels = 5 := rfl
  have positions_def : positions = 4 := rfl
  rw [num_vowels_def, positions_def]
  norm_num
  done

end Acme_Vowel_Soup_Sequences_l603_603896


namespace power_function_properties_l603_603388

theorem power_function_properties :
  ∀ {α x : ℝ}, (-2) ^ α = -8 ∧ x ^ α = 27 → x = 3 :=
by
  intros α x h
  have hₐ : (-2) ^ α = -8 := h.1
  have hₓ : x ^ α = 27 := h.2
  sorry

end power_function_properties_l603_603388


namespace square_perimeter_eq_l603_603941

theorem square_perimeter_eq :
  (∀ A B C : ℝ,
    (A = 20) →
    (B = 40) →
    C = 4 * ((A / 4) + 2 * (B / 4)) →
    C = 100) :=
by
  intro A B C hA hB hC
  rw [hA, hB] at hC
  simp at hC
  rw hC
  trivial

end square_perimeter_eq_l603_603941


namespace find_angles_of_triangle_BKD_l603_603721

noncomputable def angles_of_triangle_BKD : Prop :=
  ∃ (A B C E K D : Type) [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint E] [IsPoint K] [IsPoint D],
    (IsEquilateralTriangle A B C) ∧
    (ExtensionOf E A C) ∧
    (Midpoint K C E) ∧
    (PerpendicularLineThrough A B D) ∧
    (PerpendicularLineThrough E B D) ∧
    (AnglesOfTriangle B K D [90, 60, 30])

-- Theorem to state the problem's requirement
theorem find_angles_of_triangle_BKD : angles_of_triangle_BKD :=
sorry

end find_angles_of_triangle_BKD_l603_603721


namespace arithmetic_sequence_average_l603_603819

theorem arithmetic_sequence_average :
  ∀ (seq : List ℕ), seq.length = 10 → seq.nth 9 = some 25 →
  (seq.sum / 10 : ℚ) = 20.5 := sorry

end arithmetic_sequence_average_l603_603819


namespace students_did_not_participate_l603_603285

noncomputable def total_students : ℕ := 1170
noncomputable def students_met_standards : ℕ := 900
noncomputable def percentage_did_not_meet_standards : ℚ := 0.25
noncomputable def percentage_did_not_participate : ℚ := 0.04

theorem students_did_not_participate :
  let total_tested := students_met_standards * (1 + percentage_did_not_meet_standards)
  in  total_students = total_tested / (1 - percentage_did_not_participate) →
  (total_students - total_tested) * percentage_did_not_participate = 47 :=
by 
  sorry

end students_did_not_participate_l603_603285


namespace maximum_height_l603_603858

noncomputable def h (t : ℝ) : ℝ := -5 * t^2 + 20 * t + 10

theorem maximum_height : ∃ t : ℝ, h t = 30 :=
by {
  use 2,
  show h 2 = 30,
  have ht : h 2 = -5 * 2^2 + 20 * 2 + 10 := rfl,
  calc
    h 2 = -5 * 2^2 + 20 * 2 + 10 : ht
    ... = -20 + 40 + 10
    ... = 30
}

end maximum_height_l603_603858


namespace collinear_O1_O2_A_l603_603907

variable {A B C M N O1 O2 : Type*}

-- Assume acute triangle ABC
variables [triangle A B C] (h_acute : triangle.acute A B C) (h_AB_gt_AC : dist A B > dist A C)

-- Define the circumcenters
variables (circumcenter_ABC : circumcenter A B C = O1)
variables (circumcenter_AMN : circumcenter A M N = O2)

-- Define the conditions
variables (M_bc : segment BC.contains M) (N_bc : segment BC.contains N)
variables (angle_BAM_eq_CAN : angle A B M = angle A C N)

theorem collinear_O1_O2_A :
  collinear {O1, O2, A} :=
sorry

end collinear_O1_O2_A_l603_603907


namespace solve_system_equations_l603_603236

theorem solve_system_equations (a b c x y z : ℝ) (h1 : x + y + z = 0)
(h2 : c * x + a * y + b * z = 0)
(h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
(x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = a - b ∧ y = b - c ∧ z = c - a) := 
sorry

end solve_system_equations_l603_603236


namespace triangle_ratios_sum_l603_603304

theorem triangle_ratios_sum (A B C D E F : Type) [Affine ℝ A B C D E F]
  (hD : collinear {B, D, C}) (hD_ratio : (BD.toRealLineSegment).contains D 2 1)
  (hE : collinear {A, E, C}) (hE_ratio : (AE.toRealLineSegment).contains E 1 2) : 
  (EF.length / FC.length) + (AF.length / FD.length) = 10 / 3 := 
sorry

end triangle_ratios_sum_l603_603304


namespace saucer_area_is_28_27_cm_squared_l603_603778

-- Definitions and conditions
def radius : ℝ := 3 -- Radius of Mika's saucer in centimeters
def pi_approx : ℝ := 3.14159 -- Approximate value of π
def area (r : ℝ) : ℝ := pi_approx * r^2 -- Formula for the area of a circle using the approximate π value

-- Statement of the problem
theorem saucer_area_is_28_27_cm_squared : area radius ≈ 28.27 :=
by
  -- The statement of the theorem, proof omitted
  sorry

end saucer_area_is_28_27_cm_squared_l603_603778


namespace recommended_intake_of_added_sugar_l603_603341

-- Define the constants and variables used in the conditions
noncomputable def soft_drink_calories := 2500
noncomputable def soft_drink_added_sugar_fraction := 0.05
noncomputable def candy_bars_count := 7
noncomputable def candy_bar_added_sugar_calories := 25
noncomputable def exceeded_recommended_intake_fraction := 1.0

-- Define the main proof statement
theorem recommended_intake_of_added_sugar :
  let R := 150 in
  let added_sugar_from_soft_drink := soft_drink_added_sugar_fraction * soft_drink_calories in
  let added_sugar_from_candy_bars := candy_bars_count * candy_bar_added_sugar_calories in
  let total_added_sugar_consumed := added_sugar_from_soft_drink + added_sugar_from_candy_bars in
  total_added_sugar_consumed = (1 + exceeded_recommended_intake_fraction) * R :=
by {
  sorry
}

end recommended_intake_of_added_sugar_l603_603341


namespace inequality_solution_l603_603595

theorem inequality_solution :
  { x : ℝ // x > 1/3 ∧ x < 1 } →
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = Real.exp (1 + |x|) - 1/(1 + x^2)) → 
  (x ∈ { x | f(x) > f(2*x-1) }) :=
sorry

end inequality_solution_l603_603595


namespace less_than_reciprocal_l603_603029

theorem less_than_reciprocal (a b c d e : ℝ) (ha : a = -3) (hb : b = -1/2) (hc : c = 0.5) (hd : d = 1) (he : e = 3) :
  (a < 1 / a) ∧ (c < 1 / c) ∧ ¬(b < 1 / b) ∧ ¬(d < 1 / d) ∧ ¬(e < 1 / e) :=
by
  sorry

end less_than_reciprocal_l603_603029


namespace total_distance_traveled_l603_603347

/-- The total distance traveled by Mr. and Mrs. Hugo over three days. -/
theorem total_distance_traveled :
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  first_day + second_day + third_day = 525 := by
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  have h1 : first_day + second_day + third_day = 525 := by
    sorry
  exact h1

end total_distance_traveled_l603_603347


namespace sum_of_digits_base2_of_300_l603_603022

theorem sum_of_digits_base2_of_300 : (nat.binary_digits 300).sum = 4 :=
by
  sorry

end sum_of_digits_base2_of_300_l603_603022


namespace probability_region_D_l603_603089

noncomputable def P_A : ℝ := 1 / 4
noncomputable def P_B : ℝ := 1 / 3
noncomputable def P_C : ℝ := 1 / 6

theorem probability_region_D (P_D : ℝ) (h : P_A + P_B + P_C + P_D = 1) : P_D = 1 / 4 :=
by
  sorry

end probability_region_D_l603_603089


namespace median_of_list_l603_603521

theorem median_of_list (numbers : List ℕ) (h₁ : numbers = List.range' 1 1751 ++ List.map (λ x, x^2) (List.range' 1 1751)) : 
  (findMedian numbers) = 1757 :=
sorry

end median_of_list_l603_603521


namespace mike_games_l603_603344

theorem mike_games (initial_money spent_money game_cost remaining_games : ℕ)
  (h1 : initial_money = 101)
  (h2 : spent_money = 47)
  (h3 : game_cost = 6)
  (h4 : remaining_games = (initial_money - spent_money) / game_cost) :
  remaining_games = 9 := by
  sorry

end mike_games_l603_603344


namespace negation_of_p_l603_603202

theorem negation_of_p :
  ¬(∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔ ∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end negation_of_p_l603_603202


namespace problem_l603_603959

noncomputable def h (p x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15

noncomputable def k (q r x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

theorem problem
  (p q r : ℝ)
  (h_has_distinct_roots: ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ h p a = 0 ∧ h p b = 0 ∧ h p c = 0)
  (h_roots_are_k_roots: ∀ x, h p x = 0 → k q r x = 0) :
  k q r 1 = -3322.25 :=
sorry

end problem_l603_603959


namespace average_length_of_strings_l603_603717

theorem average_length_of_strings : 
  let length1 : ℝ := 2.5
  let length2 : ℝ := 3.5
  let length3 : ℝ := 4.5
  let average_length : ℝ := (length1 + length2 + length3) / 3
  average_length = 3.5 :=
by
  let length1 := 2.5
  let length2 := 3.5
  let length3 := 4.5
  let average_length := (length1 + length2 + length3) / 3
  show average_length = 3.5
  sorry

end average_length_of_strings_l603_603717


namespace num_int_values_x_l603_603793

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l603_603793


namespace solve_for_a_l603_603584

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = -2 → x^2 - a * x + 7 = 0) → a = -11 / 2 :=
by 
  sorry

end solve_for_a_l603_603584


namespace num_int_values_x_l603_603792

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l603_603792


namespace angle_ZQY_l603_603361

-- Definitions
noncomputable def length_XY : ℝ := 2 * r

noncomputable def radius_XY : ℝ := r

noncomputable def radius_YZ : ℝ := r / 2

noncomputable def area_large_semicircle : ℝ := (1 / 2) * real.pi * r^2

noncomputable def area_small_semicircle : ℝ := (1 / 2) * real.pi * (r / 2)^2

noncomputable def total_area : ℝ := area_large_semicircle + area_small_semicircle

noncomputable def split_area : ℝ := total_area / 2

noncomputable def angle_theta : ℝ := 360 * (split_area / area_large_semicircle)

-- The Lean statement
theorem angle_ZQY : angle_theta = 112.5 := 
by sorry

end angle_ZQY_l603_603361


namespace trajectory_equation_max_area_quadrilateral_l603_603664

-- Definitions based on conditions
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def C : set (ℝ × ℝ) := {P | let (x, y) := P in (x + 1)^2 + y^2 + (x - 1)^2 + y^2 = 16}

-- Proving the trajectory equation
theorem trajectory_equation : ∀ P : ℝ × ℝ, P ∈ C ↔ let (x, y) := P in (x^2 / 4 + y^2 / 3 = 1) :=
sorry

-- Proving the maximum area of the quadrilateral
theorem max_area_quadrilateral : ∃ A B : ℝ × ℝ, 
  parallellines_intersct_ellipse C A B → 
  ∃ area : ℝ, area = 3 :=
sorry

end trajectory_equation_max_area_quadrilateral_l603_603664


namespace number_of_days_with_exactly_one_visitor_between_march_and_june_l603_603410

/-
  Define the problem conditions:
  - Total number of days from March 1 to June 30.
  - A's visiting frequency.
  - B's visiting frequency.
  - C's visiting frequency.
  - The specific calculation to check exactly one visitor per library visit day.
-/

def total_days_march_to_june : ℕ := 31 + 30 + 31 + 30

def A_visiting_days (days : ℕ) : set ℕ := { d | d ≤ days }
def B_visiting_days (days : ℕ) : set ℕ := { d | d % 2 = 0 ∧ d ≤ days }
def C_visiting_days (days : ℕ) : set ℕ := { d | d % 3 = 0 ∧ d ≤ days }

def days_with_exactly_one_visitor (total_days : ℕ) : ℕ :=
  let A_days := A_visiting_days total_days
  let B_days := B_visiting_days total_days
  let C_days := C_visiting_days total_days
  (A_days \ (B_days ∪ C_days)).card +
  (B_days \ (A_days ∪ C_days)).card +
  (C_days \ (A_days ∪ B_days)).card

theorem number_of_days_with_exactly_one_visitor_between_march_and_june :
  days_with_exactly_one_visitor total_days_march_to_june = 41 :=
sorry

end number_of_days_with_exactly_one_visitor_between_march_and_june_l603_603410


namespace log_fraction_inequality_l603_603268

theorem log_fraction_inequality (x y a b : ℝ) (hx : 0 < x) (hxy : x < y) (hy : y < 1) (hb : 1 < b) (hba : b < a) :
  (ln x / b) < (ln y / a) := 
sorry

end log_fraction_inequality_l603_603268


namespace m_value_for_perfect_square_l603_603625

theorem m_value_for_perfect_square (m : ℤ) (x y : ℤ) :
  (∃ k : ℤ, 4 * x^2 - m * x * y + 9 * y^2 = k^2) → m = 12 ∨ m = -12 :=
by
  sorry

end m_value_for_perfect_square_l603_603625


namespace tangent_construction_possible_l603_603218

noncomputable def tangent_construction_part_a (α a : ℝ) : Prop :=
  (π / 2 < α) ∧ 
  (π / 2 < a) ∧ 
  (a < α) ∧ 
  ∃ (x y : ℝ), (x, y) = (a, Real.sin a) ∧
  ∃ (B : ℝ × ℝ) (mid_AB : ℝ × ℝ) (segment_perp : ℝ × ℝ), 
  B = (π - a, Real.sin a) ∧
  mid_AB = ((a + (π - a)) / 2, (Real.sin a + Real.sin a) / 2) ∧ 
  segment_perp = (π / 2, 1) ∧
  (Real.sin a = 1)

noncomputable def tangent_construction_part_b (α a b : ℝ) : Prop :=
  (0 < α) ∧ 
  (α < π / 2) ∧ 
  (0 < b) ∧ 
  (b < a) ∧ 
  (a < α) ∧ 
  ∃ (x y : ℝ), (x, y) = (a + b, 2 * Real.sin ((a + b) / 2) * Real.cos ((a - b) / 2)) ∧
  ∃ (B : ℝ × ℝ) (C : ℝ), 
  B = (b, Real.sin b) ∧ 
  ∃ D : ℝ × ℝ, E : ℝ, 
  D = (b + (a - b) / 2, Real.sin ((a - b) / 2)) ∧
  E = 1

theorem tangent_construction_possible (α a b : ℝ) :
  tangent_construction_part_a α a ∨ tangent_construction_part_b α a b := 
sorry

end tangent_construction_possible_l603_603218


namespace count_integer_values_satisfying_condition_l603_603803

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l603_603803


namespace length_of_train_l603_603884

theorem length_of_train
  (T_platform : ℕ)
  (T_pole : ℕ)
  (L_platform : ℕ)
  (h1: T_platform = 39)
  (h2: T_pole = 18)
  (h3: L_platform = 350)
  (L : ℕ)
  (h4 : 39 * L = 18 * (L + 350)) :
  L = 300 :=
by
  sorry

end length_of_train_l603_603884


namespace sum_of_solutions_eq_104_l603_603696

noncomputable def f (x : ℝ) : ℝ := 25 * x + 4

theorem sum_of_solutions_eq_104 : 
  let f_inv (y : ℝ) : ℝ := (y - 4) / 25 in
  ∃ s : list ℝ, (∀ x ∈ s, f_inv x = f (x⁻²)) ∧ s.sum = 104 :=
by 
  let f_x_inv (x : ℝ) := 1 / x ^ 2
  let equation := λ x, x^3 - 104 * x^2 - 625
  have h : ∀ r, equation r = 0 ↔ f_inv (f (r⁻²)) = r := sorry
  let solutions := {r | equation r = 0}.to_finset
  exists (solutions.to_list : list ℝ)
  split
  -- This implies that each list element satisfies the inverse function equation
  {
    intros x hx,
    rw list.mem_to_finset at hx,
    exact (h x).mpr hx
  },
  -- This ensures that the sum is the roots sum, here assumed as 104.
  {
    rw list.sum_to_finset,
    exact sorry
  }

end sum_of_solutions_eq_104_l603_603696


namespace choose_4_from_15_l603_603291

theorem choose_4_from_15 : (Nat.choose 15 4) = 1365 :=
by
  sorry

end choose_4_from_15_l603_603291


namespace root_of_quadratic_equation_l603_603455

def quadratic_formula (a b c : ℝ) : Set ℝ := 
  {x | x = (-b + real.sqrt (b^2 - 4*a*c))/(2*a) ∨ x = (-b - real.sqrt (b^2 - 4*a*c))/(2*a)}

theorem root_of_quadratic_equation :
  quadratic_formula 3 5 (-1) = {x | x = (-5 + real.sqrt ((5^2) + 4 * 3 * 1))/ (2 * 3) ∨ 
                                        x = (-5 - real.sqrt ((5^2) + 4 * 3 * 1))/ (2 * 3)} :=
sorry

end root_of_quadratic_equation_l603_603455


namespace minimum_function_C_value_is_2_l603_603900

noncomputable def func_C (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem minimum_function_C_value_is_2 : ∃ x : ℝ, func_C x = 2 :=
by {
  sorry,
}

end minimum_function_C_value_is_2_l603_603900


namespace representatives_function_l603_603483

theorem representatives_function (x : ℕ) : 
  let y := if x % 10 > 6 then x / 10 + 1 else x / 10 in
  y = (x + 3) / 10 :=
by
  sorry

end representatives_function_l603_603483


namespace total_length_of_circle_shaped_tapes_l603_603459

theorem total_length_of_circle_shaped_tapes :
  let number_pieces := 16
  let length_each_piece := 10.4
  let overlap := 3.5
  let contributing_length_per_piece := length_each_piece - overlap
  let total_length := number_pieces * contributing_length_per_piece
  total_length = 110.4 := by
  -- Definitions
  let number_pieces := 16
  let length_each_piece := 10.4
  let overlap := 3.5
  let contributing_length_per_piece := length_each_piece - overlap
  let total_length := number_pieces * contributing_length_per_piece
  -- Proof
  sorry

end total_length_of_circle_shaped_tapes_l603_603459


namespace num_int_values_x_l603_603788

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l603_603788


namespace complex_is_pure_imaginary_l603_603272

theorem complex_is_pure_imaginary (a : ℝ) : 
  let z := (a - complex.I)^2 in 
  (complex.re z = 0) → (a = 1 ∨ a = -1) :=
by
  let z := (a - complex.I)^2
  have hz : complex.re z = 0 → (a^2 - 1 = 0) := by sorry
  show (complex.re z = 0) → (a = 1 ∨ a = -1) from
    λ h, hz h

end complex_is_pure_imaginary_l603_603272


namespace greatest_x_l603_603375

-- Define x as a positive multiple of 4.
def is_positive_multiple_of_four (x : ℕ) : Prop :=
  x > 0 ∧ ∃ k : ℕ, x = 4 * k

-- Statement of the equivalent proof problem
theorem greatest_x (x : ℕ) (h1: is_positive_multiple_of_four x) (h2: x^3 < 4096) : x ≤ 12 :=
by {
  sorry
}

end greatest_x_l603_603375


namespace train_length_calculation_l603_603088

def speed_kmh := 36 -- speed of the train in km/hr
def time_sec := 5.5 -- time for the train to cross the pole in seconds
def expected_length := 55 -- expected length of the train in meters

theorem train_length_calculation (speed_kmh: ℕ) (time_sec: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600 in
  speed_ms * time_sec

example : train_length_calculation speed_kmh time_sec = expected_length := by
  sorry

end train_length_calculation_l603_603088


namespace find_m_n_l603_603183

variables (a b : ℝ × ℝ)
variables (m n : ℝ)

def vector_a := (-3, 1)
def vector_b := (-1, 2)

theorem find_m_n (h : m * vector_a + n * vector_b = (10, 0)) : m = -4 ∧ n = -2 :=
sorry

end find_m_n_l603_603183


namespace bullet_train_pass_time_l603_603035

-- Definitions based on conditions
def train_length : ℝ := 120  -- Length of the train in meters
def train_speed_kmph : ℝ := 50  -- Speed of the train in km/h
def man_speed_kmph : ℝ := 4  -- Speed of the man in km/h

-- Converting speeds from km/h to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := (speed_kmph * 1000) / 3600

-- Relative speed in m/s when two objects move in opposite directions
def relative_speed_mps : ℝ := kmph_to_mps (train_speed_kmph + man_speed_kmph)

-- Calculate the time it takes for the train to pass the man
def time_to_pass : ℝ := train_length / relative_speed_mps

theorem bullet_train_pass_time : time_to_pass = 8 :=
by
  -- This statement ensures that Lean checks if the defined time_to_pass equals 8 seconds
  sorry  -- Proof can be provided here if required

end bullet_train_pass_time_l603_603035


namespace tank_capacity_l603_603489

noncomputable def inflow_A (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_B (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_C (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_X (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_Y (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

theorem tank_capacity
  (fA : ℕ := inflow_A 8 7)
  (fB : ℕ := inflow_B 12 3)
  (fC : ℕ := inflow_C 6 4)
  (oX : ℕ := outflow_X 20 7)
  (oY : ℕ := outflow_Y 15 5) :
  fA + fB + fC = 6960 ∧ oX + oY = 12900 ∧ 12900 - 6960 = 5940 :=
by
  sorry

end tank_capacity_l603_603489


namespace differentiable_F_initial_condition_F_differential_eq_F_l603_603760

noncomputable def F (x : ℝ) : ℝ := -cos (sin (sin (sin x)))

theorem differentiable_F : Differentiable ℝ F := by 
  sorry

theorem initial_condition_F : F 0 = -1 := by
  sorry

theorem differential_eq_F : (fun x => deriv F x) = (fun x => sin (sin (sin (sin x))) * cos (sin (sin x)) * cos (sin x) * cos x) := by
  sorry

end differentiable_F_initial_condition_F_differential_eq_F_l603_603760


namespace number_of_white_balls_l603_603657

theorem number_of_white_balls (total : ℕ) (freq_red freq_black : ℚ) (h1 : total = 120) 
                              (h2 : freq_red = 0.15) (h3 : freq_black = 0.45) : 
                              (total - total * freq_red - total * freq_black = 48) :=
by sorry

end number_of_white_balls_l603_603657


namespace train_cross_pole_time_l603_603494

-- Define the conditions
def speed_km_per_hr : ℝ := 36
def train_length_meters : ℝ := 90

-- Convert speed from km/hr to m/s
def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)

-- State the theorem
theorem train_cross_pole_time : train_length_meters / speed_m_per_s = 9 := by
  sorry

end train_cross_pole_time_l603_603494


namespace continued_fraction_solution_l603_603432

noncomputable def continued_fraction : ℝ := 
  3 + (5 / (2 + (5 / (3 + (5 / (2 + ...)))))

theorem continued_fraction_solution :
  (∃ y : ℝ, y = continued_fraction ∧ y = (3 + real.sqrt 69) / 2) :=
sorry

end continued_fraction_solution_l603_603432


namespace imaginary_part_of_fraction_l603_603986

theorem imaginary_part_of_fraction {i : ℂ} (hi : i = complex.I) :
  complex.imag (1 + 2 * i) / (i - 2) = -1 := 
by 
  sorry

end imaginary_part_of_fraction_l603_603986


namespace collinear_A_F_C_l603_603417

open_locale classical

variables {S₁ S₂ : Type} [metric_space S₁] [metric_space S₂] 
variables {F A B C D E : S₁}
variables (h_touch : ∀ x ∈ S₁, ∃ y ∈ S₂, dist x y = 0 ∨ dist x y = 2 * radius S₁ + 2 * radius S₂) 
variables (h_tangent_A : ∀ {p : S₁}, tangent_point S₁ A p ↔ orthogonal p (center S₁, A)) 
variables (h_tangent_B : ∀ {p : S₂}, tangent_point S₂ B p ↔ orthogonal p (center S₂, B)) 
variables (h_parallel : parallel (path A B) (path C D E))
variables (h_tangent_C : tangent_point S₂ C (path C D E))

theorem collinear_A_F_C : collinear {A, F, C} :=
sorry

end collinear_A_F_C_l603_603417


namespace find_x_l603_603255

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end find_x_l603_603255


namespace gain_in_transaction_per_year_l603_603480

noncomputable def borrowing_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def lending_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def gain_per_year (borrow_principal : ℕ) (borrow_rate : ℚ) 
  (borrow_time : ℕ) (lend_principal : ℕ) (lend_rate : ℚ) (lend_time : ℕ) : ℚ :=
  (lending_interest lend_principal lend_rate lend_time - borrowing_interest borrow_principal borrow_rate borrow_time) / borrow_time

theorem gain_in_transaction_per_year :
  gain_per_year 4000 (4 / 100) 2 4000 (6 / 100) 2 = 80 := 
sorry

end gain_in_transaction_per_year_l603_603480


namespace harriet_travel_time_l603_603449

theorem harriet_travel_time (D : ℝ) (h : (D / 90 + D / 160 = 5)) : (D / 90) * 60 = 192 := 
by sorry

end harriet_travel_time_l603_603449


namespace gnomon_magic_square_diagonals_equal_l603_603055

structure GnomonMagicSquare (α : Type) [Add α] :=
  (a11 a12 a13 : α)
  (a21 a22 a23 : α)
  (a31 a32 a33 : α)
  (condition :
    a21 + a22 + a31 + a32 = a11 + a12 + a21 + a22 ∧
    a11 + a12 + a21 + a22 = a12 + a13 + a22 + a23 ∧
    a12 + a13 + a22 + a23 = a22 + a23 + a32 + a33)

open GnomonMagicSquare

theorem gnomon_magic_square_diagonals_equal {α : Type} [Add α] [Eq α] (s : GnomonMagicSquare α) :
  s.a11 + s.a22 + s.a33 = s.a13 + s.a22 + s.a31 :=
  sorry

end gnomon_magic_square_diagonals_equal_l603_603055


namespace orthocenter_of_triangle_l603_603309

-- Define the conditions
variables {A B C P : Type}
variables [metric_space P]
variables [nonempty A]
variables [nonempty B]
variables [nonempty C]

-- Assume a triangle ABC and internal point P with the properties:
-- Each altitude is drawn to intersect the opposite side.
-- On each of these segments, a circle is constructed with the segment as the diameter.
-- A chord through point P is drawn perpendicular to these diameters.
-- The three chords have the same length.

-- Define triangle ABC and the internal point P
structure Triangle :=
  (A B C : P)
  (inside : P)

-- Define altitude intersection points and circles
structure AltitudeIntersection :=
  (alt_A : P × P)
  (alt_B : P × P)
  (alt_C : P × P)

-- Define the property of equal length chords perpendicular to the diameters
structure EqualLengthChords :=
  (chord_len : ℝ)
  (chord_A : P)
  (chord_B : P)
  (chord_C : P)
  (perp_A : is_perpendicular chord_A P)
  (perp_B : is_perpendicular chord_B P)
  (perp_C : is_perpendicular chord_C P)
  (equal_len : chord_len = dist (perp_A) (perp_B))

-- Finally, the theorem to prove
theorem orthocenter_of_triangle (T : Triangle) (AIs : AltitudeIntersection) (ELCs : EqualLengthChords) : 
  is_orthocenter T P := sorry

end orthocenter_of_triangle_l603_603309


namespace find_x_l603_603262

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l603_603262


namespace pencils_and_notebooks_cost_l603_603816

theorem pencils_and_notebooks_cost
    (p n : ℝ)
    (h1 : 8 * p + 10 * n = 5.36)
    (h2 : 12 * (p - 0.05) + 5 * n = 4.05) :
    15 * (p - 0.05) + 12 * n = 7.01 := 
sorry

end pencils_and_notebooks_cost_l603_603816


namespace find_positive_integers_l603_603945

theorem find_positive_integers (a b : ℕ) (h1 : a > 1) (h2 : b ∣ (a - 1)) (h3 : (2 * a + 1) ∣ (5 * b - 3)) : a = 10 ∧ b = 9 :=
sorry

end find_positive_integers_l603_603945


namespace integer_values_satisfying_sqrt_inequality_l603_603809

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l603_603809


namespace non_integer_rational_condition_l603_603176

theorem non_integer_rational_condition (n : ℕ) (h_pos: 0 < n) :
  (∃ (x : ℚ), ¬ is_integer x ∧ is_integer (x^n + (x + 1)^n)) → odd n :=
sorry

end non_integer_rational_condition_l603_603176


namespace decreased_percentage_l603_603759

def original_number : ℝ := 80
def increased_value := original_number + (12.5 / 100) * original_number
def decreased_value (x : ℝ) := original_number - (x / 100) * original_number
def difference (x : ℝ) := increased_value - decreased_value x

theorem decreased_percentage (x : ℝ) : difference x = 30 → x = 25 := by
  sorry

end decreased_percentage_l603_603759


namespace center_and_radius_of_inner_circle_l603_603707

noncomputable def circle_center_and_radius {D : set (ℝ × ℝ)} (Q : ℝ × ℝ) (radius_D : ℝ) (B : ℝ × ℝ) (dist_QB : ℝ) :=
  {A | (dist A B) ≤ ∀ point_on_boundary ∈ (boundary D), dist A point_on_boundary}

theorem center_and_radius_of_inner_circle {Q : ℝ × ℝ} {B : ℝ × ℝ} :
  dist Q B = 6 →
  circle_center_and_radius Q 10 B 6 = {A | dist A B <= 4} :=
begin
  sorry
end

end center_and_radius_of_inner_circle_l603_603707


namespace correct_answer_l603_603591

-- Define the sequence {a_n}
def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n * a (n + 1) = 2 ^ n

-- Define the sum S_n of the first n terms of the sequence {a_n}
def sum_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in range (n + 1), a i

-- Given conditions
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
hypothesis h_seq : sequence a
hypothesis h_a3 : a 3 = 2
hypothesis h_sum : sum_sequence a S

-- Statement to prove
theorem correct_answer : S 100 = 3 * (2 ^ 50 - 1) :=
sorry

end correct_answer_l603_603591


namespace problem1_problem2_l603_603242

-- Define the given conditions
variables (α : ℝ) (k : ℤ)
def m : ℝ × ℝ := (sin α - 2, -cos α)
def n : ℝ × ℝ := (-sin α, cos α)

-- Problem 1: If m ⊥ n, find α
theorem problem1 (h : (m α).1 * (n α).1 + (m α).2 * (n α).2 = 0) : 
  ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 6 ∨ α = 2 * k * Real.pi + 5 * Real.pi / 6 := 
sorry

-- Problem 2: If |m - n| = √2, find cos(2α)
theorem problem2 (h : ((m α).1 - (n α).1)^2 + ((m α).2 - (n α).2)^2 = 2) : 
  cos (2 * α) = -1 / 8 := 
sorry

end problem1_problem2_l603_603242


namespace sum_first_n_terms_S_n_correct_l603_603326

noncomputable def f : ℝ → ℝ := sorry
noncomputable def a (n : ℕ) (h : 0 < n) : ℝ := f n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1) (nat.succ_pos i)

axiom condition1 : ∀ x y : ℝ, f (x + y) = f x + f y - 1
axiom condition2 : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ f (x^2 - x - 3) < 3
axiom condition3 : ∀ n : ℕ, 0 < n → f n = f 1 + (n - 1) * 2 / 3

theorem sum_first_n_terms (n : ℕ) : 
  S n = n * f 1 + (n * (n - 1) / 2) * (2 / 3) := sorry

theorem S_n_correct (n : ℕ) : 
  S n = n * (n + 4) / 3 := sorry

end sum_first_n_terms_S_n_correct_l603_603326


namespace compute_sixth_power_sum_l603_603325

theorem compute_sixth_power_sum (ζ1 ζ2 ζ3 : ℂ) 
  (h1 : ζ1 + ζ2 + ζ3 = 2)
  (h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5)
  (h3 : ζ1^4 + ζ2^4 + ζ3^4 = 29) :
  ζ1^6 + ζ2^6 + ζ3^6 = 101.40625 := 
by
  sorry

end compute_sixth_power_sum_l603_603325


namespace f_inv_128_l603_603631

noncomputable def f : ℕ → ℕ := sorry -- Placeholder for the function definition.

axiom f_5 : f 5 = 2           -- Condition 1: f(5) = 2
axiom f_2x : ∀ x, f (2 * x) = 2 * f x  -- Condition 2: f(2x) = 2f(x) for all x

theorem f_inv_128 : f⁻¹ 128 = 320 := sorry -- Prove that f⁻¹(128) = 320 given the conditions

end f_inv_128_l603_603631


namespace find_a_value_l603_603998

theorem find_a_value (a : ℝ) (h1 : ∀ x, f x = 2 * a * x^3 - a)
  (h2 : ∀ (p : ℝ × ℝ), p = (1, a))
  (h3 : ∀ (g : ℝ → ℝ), deriv g 1 = 6 * a)
  (h4 : ∀ (h : ℝ → ℝ), deriv h = 2) :
  a = 1 / 3 := 
by
  sorry

end find_a_value_l603_603998


namespace find_angle_sum_l603_603296

theorem find_angle_sum (α β : ℝ)
  (ha : cos α = (sqrt 5) / 5 ∧ sin α = (2 * sqrt 5) / 5 ∧ α ∈ (Ioo (π/4) (π/2)))
  (hb : cos β = (7 * sqrt 2) / 10 ∧ sin β = (sqrt 2) / 10 ∧ β ∈ (Ioo 0 (π/6))) :
  2 * α + β = 3 * π / 4 := sorry

end find_angle_sum_l603_603296


namespace sum_of_digits_base2_300_l603_603010

theorem sum_of_digits_base2_300 : 
  let n := 300
  in (Int.digits 2 n).sum = 4 :=
by
  let n := 300
  have h : Int.digits 2 n = [1, 0, 0, 1, 0, 1, 1, 0, 0] := by sorry
  rw h
  norm_num
  -- or directly
  -- exact rfl

end sum_of_digits_base2_300_l603_603010


namespace max_sum_of_coprime_composites_l603_603608

theorem max_sum_of_coprime_composites :
  ∃ (A B C : ℕ), 
    (composite A) ∧ (composite B) ∧ (composite C) ∧ 
    (pairwise coprime [A, B, C]) ∧ 
    (A * B * C = 11011 * 28) ∧
    (A + B + C = 1626) :=
by {
  sorry
}

end max_sum_of_coprime_composites_l603_603608


namespace distance_between_parallel_lines_l603_603981

theorem distance_between_parallel_lines (b : ℝ) 
  (h_parallel : ∀ x y, 2 * (x + 2 * y + 1) = b * (2 * x + by - 4)) : 
  ∃ d : ℝ, d = (3 * Real.sqrt 5) / 5 ∧
  ∀ p : (ℝ × ℝ), (p.fst + 2 * p.snd - 2 = 0) → 
  p = (0, 1) := 
sorry

end distance_between_parallel_lines_l603_603981


namespace solution_25_21_l603_603327

noncomputable def g : ℚ+ → ℤ := sorry

axiom functional_equation (x y : ℚ+) : g (x * y) = g x + g y
axiom prime_condition (n : ℕ) (h : Prime n) : g ⟨n, sorry⟩ = n^2

theorem solution_25_21 : g ⟨25, sorry⟩ - g ⟨21, sorry⟩ = -8 ∧ g ⟨25, sorry⟩ - g ⟨21, sorry⟩ < 16 :=
by {
  sorry
}

end solution_25_21_l603_603327


namespace sapsan_signal_duration_l603_603390

theorem sapsan_signal_duration
    (v_kmh : ℝ) (h_v_kmh : v_kmh = 216)
    (Δt : ℝ) (h_Δt : Δt = 5)
    (c : ℝ) (h_c : c = 340) :
    ∃ Δt₁ : ℝ, Δt₁ = (Δt * (c - (v_kmh * 1000 / 3600)) / c) :=
begin
    use Δt * ((c - (v_kmh * 1000 / 3600)) / c),
    rw [h_v_kmh, h_Δt, h_c],
    norm_num,
    sorry
end

end sapsan_signal_duration_l603_603390


namespace num_four_letter_initials_sets_l603_603612

def num_initials_sets : ℕ := 8 ^ 4

theorem num_four_letter_initials_sets:
  num_initials_sets = 4096 :=
by
  rw [num_initials_sets]
  norm_num

end num_four_letter_initials_sets_l603_603612


namespace compute_expression_at_4_l603_603109

theorem compute_expression_at_4 (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end compute_expression_at_4_l603_603109


namespace rate_of_current_l603_603780

theorem rate_of_current (c : ℝ) : 
  (∀ t : ℝ, t = 0.4 → ∀ d : ℝ, d = 9.6 → ∀ b : ℝ, b = 20 →
  d = (b + c) * t → c = 4) :=
sorry

end rate_of_current_l603_603780


namespace problem_statement_l603_603566

noncomputable def a : Nat → ℕ
| 1 => 1
| 2 => 2
| n+1 => sorry

def S : ℕ → ℕ
| 0 => 0
| n+1 => S n + a (n+1)

def k := 2

def T (n : ℕ) := (Finset.range n).sum S

/-- 
(Ⅰ) The value of k is 2.
(Ⅱ) The sequence {a_n} is a geometric sequence with common ratio 2.
(Ⅲ) T_{10} is 2036.
-/
theorem problem_statement :
    (∀ n, S (n+1) = k * S n + 1) ∧
    (∀ n, S (n+1) = 2 * S n + 1) ∧
    (∀ n, n ≥ 2 → a (n+1) = 2 * a n) ∧
    (S 1 = 1 ∧ S 2 = 3) ∧
    (S n = 2^n - 1) ∧
    T 10 = 2036 :=
by
  sorry

end problem_statement_l603_603566


namespace largest_base_conversion_l603_603501

theorem largest_base_conversion :
  let A := 11011₂
  let B := 103₄
  let C := 44₅
  let D := 25 in
  A = 27 ∧ 
  (B = 19 ∧ C = 24 ∧ D = 25) ∧
  (27 > 19 ∧ 27 > 24 ∧ 27 > 25) :=
by
  let A := 27
  let B := 19
  let C := 24
  let D := 25
  sorry

end largest_base_conversion_l603_603501


namespace no_real_roots_l603_603933

theorem no_real_roots (k : ℝ) (h : k ≠ 0) : ¬∃ x : ℝ, x^2 + k * x + 3 * k^2 = 0 :=
by
  sorry

end no_real_roots_l603_603933


namespace ellipse_properties_l603_603570

-- Definitions for the problem's conditions:
def e : ℝ := sqrt 3 / 2
def a : ℝ := 2
def b : ℝ := 1
def point_P : ℝ × ℝ := (sqrt 3, 1 / 2)

-- Given the assumptions above, we need to show:
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

def line_intersects_ellipse (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 / 4) + ((-x + m)^2) = 1

def perpendicular_condition (m : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), (x1 ^ 2 / 4 + y1 ^ 2 = 1) ∧ (x2 ^ 2 / 4 + y2 ^ 2 = 1) ∧
  (y1 = -x1 + m) ∧ (y2 = -x2 + m) ∧
  (x1 * x2 + y1 * y2 = 0)

theorem ellipse_properties :
  (∀ (x y : ℝ), ellipse_equation x y) ∧
  (∃ (m : ℝ), line_intersects_ellipse m ∧ perpendicular_condition m) :=
  by
    sorry

end ellipse_properties_l603_603570


namespace minimum_cos_minus_sin_l603_603172

noncomputable def cos_minus_sin_min_value : Prop :=
  ∀ (x y z : ℝ),
  (2 * Real.sin x = Real.tan y) ∧
  (2 * Real.cos y = Real.cot z) ∧
  (Real.sin z = Real.tan x) →
  Real.cos x - Real.sin z = - (5 * Real.sqrt 3) / 6

theorem minimum_cos_minus_sin (x y z : ℝ) :
  2 * Real.sin x = Real.tan y →
  2 * Real.cos y = Real.cot z →
  Real.sin z = Real.tan x →
  Real.cos x - Real.sin z = - (5 * Real.sqrt 3) / 6 := sorry

end minimum_cos_minus_sin_l603_603172


namespace range_of_m_l603_603235

open Set

noncomputable def A (m : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x^2 + m * x - y + 2 = 0} 

noncomputable def B : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x - y + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ B ≠ ∅) → (m ≤ -1 ∨ m ≥ 3) := 
sorry

end range_of_m_l603_603235


namespace min_m_for_dice_rolls_l603_603732

theorem min_m_for_dice_rolls :
  (∀ a b : ℕ, 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 → a^2 + b^2 ≤ 72) :=
begin
  intros a b ha hb,
  sorry -- Proof omitted
end

end min_m_for_dice_rolls_l603_603732


namespace evaluate_expression_l603_603544

def greatest_integer (x : ℝ) : ℤ :=
  Int.floor x

theorem evaluate_expression (y : ℝ) (h : y = 2) :
  greatest_integer 6.5 * greatest_integer (2 / 3) + greatest_integer y * 7.2 + greatest_integer 8.4 - 6.2 = 16.2 :=
by
  sorry

end evaluate_expression_l603_603544


namespace problem_1_equals_minus_4_problem_2_equals_2_l603_603916
noncomputable def problem_1 := 0.25 * (1 / 2) ^ -4 - 4 / (sqrt 5 - 1) ^ 0 - (1 / 16) ^ (-1 / 2)
noncomputable def problem_2 := log 25 + log 2 * log 50 + (log 2) ^ 2

theorem problem_1_equals_minus_4 : problem_1 = -4 := by sorry

theorem problem_2_equals_2 : problem_2 = 2 := by sorry

end problem_1_equals_minus_4_problem_2_equals_2_l603_603916


namespace find_two_numbers_l603_603953

theorem find_two_numbers (x y : ℕ) : 
  (x + y = 20) ∧
  (x * y = 96) ↔ 
  ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := 
by
  sorry

end find_two_numbers_l603_603953


namespace triangle_median_ratio_l603_603714

theorem triangle_median_ratio (A B C A1 M : Point)
  (h1 : is_midpoint A1 B C) (h2 : on_median A A1 M (1/3)) : divides BM AC (1/6) :=
sorry

end triangle_median_ratio_l603_603714


namespace length_MN_proof_l603_603662

-- Declare a noncomputable section to avoid computational requirements
noncomputable section

-- Define the quadrilateral ABCD with given sides
structure Quadrilateral :=
  (BC AD AB CD : ℕ)
  (BC_AD_parallel : Prop)

-- Define a theorem to calculate the length MN
theorem length_MN_proof (ABCD : Quadrilateral) 
  (M N : ℝ) (BisectorsIntersect_M : Prop) (BisectorsIntersect_N : Prop) : 
  ABCD.BC = 26 → ABCD.AD = 5 → ABCD.AB = 10 → ABCD.CD = 17 → 
  (MN = 2 ↔ (BC + AD - AB - CD) / 2 = 2) :=
by
  sorry

end length_MN_proof_l603_603662


namespace goals_last_season_l603_603680

theorem goals_last_season : 
  ∀ (goals_last_season goals_this_season total_goals : ℕ), 
  goals_this_season = 187 → 
  total_goals = 343 → 
  total_goals = goals_last_season + goals_this_season → 
  goals_last_season = 156 := 
by 
  intros goals_last_season goals_this_season total_goals 
  intro h_this_season 
  intro h_total_goals 
  intro h_equation 
  calc 
    goals_last_season = total_goals - goals_this_season : by rw [h_equation, Nat.add_sub_cancel_left]
    ... = 343 - 187 : by rw [h_this_season, h_total_goals]
    ... = 156 : by norm_num

end goals_last_season_l603_603680


namespace number_of_false_statements_l603_603634

theorem number_of_false_statements (p q : Prop) (hp : p = True) (hq : q = False) :
  (∃ f1 f2 f3 f4 : Prop, 
    f1 = (p ∧ q) ∧ 
    f2 = (p ∨ q) ∧ 
    f3 = (¬ p) ∧
    f4 = (¬ q) ∧
    ((f1 = False ∧ f2 = False ∧ f3 = False ∧ f4 = False) = 2)) := sorry

end number_of_false_statements_l603_603634


namespace chess_tournament_students_l603_603351

variable (total_students : ℕ) (chess_percentage : ℝ) (tournament_fraction : ℝ)

theorem chess_tournament_students (h1 : total_students = 120)
                                  (h2 : chess_percentage = 0.45)
                                  (h3 : tournament_fraction = 2/5) :
                                  let chess_students := total_students * chess_percentage in
                                  let attending_students := chess_students * tournament_fraction in
                                  attending_students.toNat = 21 :=
by
  sorry

end chess_tournament_students_l603_603351


namespace conjugate_of_z_l603_603755

noncomputable def z : ℂ := (-2 + 2 * complex.i) / (1 + complex.i)

theorem conjugate_of_z : complex.conj z = -2 * complex.i :=
by
  sorry

end conjugate_of_z_l603_603755


namespace exists_divisible_by_sum_of_digits_l603_603356

theorem exists_divisible_by_sum_of_digits (a : ℕ) (h_a : 100 ≤ a ∧ a ≤ 982) :
  ∃ n, n ∈ (finset.range 18).image (λ i, a + i) ∧ (n % (n.digits 10).sum = 0) :=
sorry

end exists_divisible_by_sum_of_digits_l603_603356


namespace inequality_log_div_l603_603265

theorem inequality_log_div (x y a b : ℝ) (hx : 0 < x) (hxy : x < y) (hy : y < 1)
  (hb : 1 < b) (hba : b < a) :
  (ln x) / b < (ln y) / a := 
sorry

end inequality_log_div_l603_603265


namespace color_pairs_exist_l603_603546

theorem color_pairs_exist (n : ℕ) (hn : n > 0) : 
  ∃ f : fin (2 * n) → fin n, 
    (∀ k : fin n, ∃! (x y : fin (2 * n)), 
      x ≠ y ∧ f x = k ∧ f y = k ∧ |x - y| ∈ (fin n)) :=
sorry

end color_pairs_exist_l603_603546


namespace num_int_values_x_l603_603791

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l603_603791


namespace hyperbola_canonical_equation_l603_603960

-- Given conditions:
def eccentricity (ε : ℝ) := ε = 1.5
def focal_distance (d : ℝ) := d = 6

-- The problem to prove:
theorem hyperbola_canonical_equation (ε d c a b : ℝ) 
  (h₁ : eccentricity ε) 
  (h₂ : focal_distance d) 
  (h₃ : d = 2 * c) 
  (h₄ : ε = c / a) 
  (h₅ : c^2 = a^2 + b^2)
  (h₆ : b^2 = 5)  
  : (a = 2) → (c = 3) → (b^2 = 5) → (ε = 1.5) → (d = 2 * c) → 
    ∀ x y : ℝ, (x^2 / (a^2)) - (y^2 / b^2) = 1 :=
by
  intros h7 h8 h9 h10 h11 x y
  rw [h7, h8, h9]
  sorry


end hyperbola_canonical_equation_l603_603960


namespace intersection_area_eq_l603_603823

-- Define the vertices of the rectangle
def vertices := [(5, 11), (16, 11), (16, -2), (5, -2)]

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := (x - 5) ^ 2 + (y + 2) ^ 2 = 9

-- Define the theorem that states the area of the intersection
theorem intersection_area_eq : 
  let rect := {p : ℝ × ℝ | p ∈ [(5, 11), (16, 11), (16, -2), (5, -2)]}
  let circle := {p : ℝ × ℝ | circle_eq (p.1) (p.2)} in
  (∃ p ∈ rect, p ∈ circle) → 
  (area_of_intersection rect circle = (9/4) * Real.pi) := 
sorry

end intersection_area_eq_l603_603823


namespace find_a_l603_603604

open Real

def is_chord_length_correct (a : ℝ) : Prop :=
  let x_line := fun t : ℝ => 1 + t
  let y_line := fun t : ℝ => a - t
  let x_circle := fun α : ℝ => 2 + 2 * cos α
  let y_circle := fun α : ℝ => 2 + 2 * sin α
  let distance_from_center := abs (3 - a) / sqrt 2
  let chord_length := 2 * sqrt (4 - distance_from_center ^ 2)
  chord_length = 2 * sqrt 2 

theorem find_a (a : ℝ) : is_chord_length_correct a → a = 1 ∨ a = 5 :=
by
  sorry

end find_a_l603_603604


namespace y_in_terms_of_x_l603_603263

theorem y_in_terms_of_x (p x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) := 
by 
  sorry

end y_in_terms_of_x_l603_603263


namespace age_difference_proof_l603_603339

noncomputable def lexie_age : ℕ := 8
noncomputable def brother_age : ℕ := lexie_age - 6
noncomputable def sister_age : ℕ := 2 * lexie_age
noncomputable def age_difference : ℕ := sister_age - brother_age

theorem age_difference_proof : age_difference = 14 := 
by
  have h1 : lexie_age = 8 := rfl
  have h2 : brother_age = lexie_age - 6 := rfl
  have h3 : sister_age = 2 * lexie_age := rfl
  have h4 : age_difference = sister_age - brother_age := rfl
  rw [h1, h2, h3, h4]
  sxrwa/norm_num
  :
  assageifference := h3
  sorry

end age_difference_proof_l603_603339


namespace least_positive_A_l603_603702

theorem least_positive_A (W S : ℕ) (hW : W = 8) (hS : S = 5) : 
  ∃ (A : ℕ), 6 < A ∧ A < 10 ∧ A = 7 := 
by {
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { exact dec_trivial }
}

end least_positive_A_l603_603702


namespace find_PB_l603_603048

variables (P A B C D : Point) (PA PD PC PB : ℝ)
-- Assume P is interior to rectangle ABCD
-- Conditions
axiom hPA : PA = 3
axiom hPD : PD = 4
axiom hPC : PC = 5

-- The main statement to prove
theorem find_PB (P A B C D : Point) (PA PD PC PB : ℝ)
  (hPA : PA = 3) (hPD : PD = 4) (hPC : PC = 5) : PB = 3 * Real.sqrt 2 :=
by
  sorry

end find_PB_l603_603048


namespace count_female_worker_ants_l603_603370

theorem count_female_worker_ants
  (total_ants : ℕ)
  (half_worker_ratio : total_ants / 2)
  (male_worker_ratio : ℕ → ℕ → Prop)
  (twenty_percent_of_worker_ants : male_worker_ratio (total_ants / 2) (total_ants / 2 * 2 / 10))
  (total_ants_eq : total_ants = 110) :
  total_ants / 2 - (total_ants / 2 * 20 / 100) = 44 :=
by
  rw [total_ants_eq]
  sorry

end count_female_worker_ants_l603_603370


namespace minimal_surface_area_l603_603905

-- Definitions based on the conditions in the problem.
def unit_cube (a b c : ℕ) : Prop := a * b * c = 25
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

-- The proof problem statement.
theorem minimal_surface_area : ∃ (a b c : ℕ), unit_cube a b c ∧ surface_area a b c = 54 := 
sorry

end minimal_surface_area_l603_603905


namespace sum_of_divisors_of_3000_l603_603952

theorem sum_of_divisors_of_3000 :
  let nums_with_common_divisor := {n | n ∈ Finset.range 3001 ∧ ∃ d > 1, d ∣ 3000 ∧ d ∣ n} in
  (Finset.sum nums_with_common_divisor (λ x, x)) = 3301500 :=
by
  sorry

end sum_of_divisors_of_3000_l603_603952


namespace each_sack_weight_l603_603506

theorem each_sack_weight
  (number_of_dogs : ℕ)
  (meals_per_day : ℕ)
  (food_per_meal_g : ℕ)
  (number_of_days : ℕ)
  (number_of_sacks : ℕ)
  (grams_per_kg : ℕ)
  (total_food_kg : ℕ) :
  number_of_dogs = 4 →
  meals_per_day = 2 →
  food_per_meal_g = 250 →
  number_of_days = 50 →
  number_of_sacks = 2 →
  grams_per_kg = 1000 →
  total_food_kg = (number_of_dogs * meals_per_day * food_per_meal_g * number_of_days) / grams_per_kg →
  (total_food_kg / number_of_sacks) = 50 :=
begin
  sorry
end

end each_sack_weight_l603_603506


namespace count_valid_integer_area_triangles_l603_603926

def is_valid_point (x y : ℕ) : Prop :=
  41 * x + y = 2009

def triangle_area (x1 y1 x2 y2 : ℕ) : ℕ :=
  (x1 * y2 - x2 * y1).natAbs / 2

def is_distinct (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 ≠ x2) ∨ (y1 ≠ y2)

def is_integer_area (area : ℕ) : Prop :=
  area > 0

theorem count_valid_integer_area_triangles :
  let points := { p : ℕ × ℕ // is_valid_point p.1 p.2 }
  let valid_triangles :=
    { (p1, p2) : points × points //
      is_distinct p1.val.1 p1.val.2 p2.val.1 p2.val.2 ∧
      is_integer_area (triangle_area p1.val.1 p1.val.2 p2.val.1 p2.val.2) }
  in
  (finset.univ.card : valid_triangles) = 600 :=
by
  sorry

end count_valid_integer_area_triangles_l603_603926


namespace sin_sum_identity_l603_603049

theorem sin_sum_identity :
  sin 17 * cos 43 + cos 17 * sin 43 = sqrt 3 / 2 :=
by
  -- sin(17° + 43°) = sin 60°
  sorry

end sin_sum_identity_l603_603049


namespace time_to_pass_tunnel_is_92_l603_603891

def train_speed_km_hr : ℝ := 90
def train_speed_m_s : ℝ := (90 * 1000) / 3600
def train_length : ℝ := 500
def tunnel_length : ℝ := 1800
def total_distance : ℝ := train_length + tunnel_length
def time_to_pass_tunnel : ℝ := total_distance / train_speed_m_s

theorem time_to_pass_tunnel_is_92 :
  time_to_pass_tunnel = 92 := by
  sorry

end time_to_pass_tunnel_is_92_l603_603891


namespace not_simplifiable_by_difference_of_squares_l603_603437

theorem not_simplifiable_by_difference_of_squares :
  ¬(∃ a b : ℝ, (-x + y) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (-x - y) * (-x + y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y + x) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y - x) * (x + y) = a^2 - b^2) :=
sorry

end not_simplifiable_by_difference_of_squares_l603_603437


namespace find_m_l603_603577

theorem find_m (m : ℝ) 
  (A : ℝ × ℝ := (-2, m))
  (B : ℝ × ℝ := (m, 4))
  (h_slope : ((B.snd - A.snd) / (B.fst - A.fst)) = -2) : 
  m = -8 :=
by 
  sorry

end find_m_l603_603577


namespace club_selection_l603_603350

theorem club_selection :
  let members := 24
  let boys := 12
  let girls := 12
  let boys_who_wear_glasses := 6
  let girls_who_wear_glasses := 6
  (∀ president, ∀ vice_president, president ≠ vice_president → 
    (∃ ways_boys, ways_boys = boys_who_wear_glasses * (boys_who_wear_glasses - 1)) ∧
    (∃ ways_girls, ways_girls = girls_who_wear_glasses * (girls_who_wear_glasses - 1)) ∧
    ((ways_boys + ways_girls) = 60)) :=
by sorry

end club_selection_l603_603350


namespace speed_of_train_l603_603880

-- Defining all the conditions given.
def length_of_train_meters : ℝ := 140
def speed_of_man_kmph : ℝ := 6
def time_to_pass_seconds : ℝ := 6

-- Conversion factor from m/s to kmph
def conversion_factor : ℝ := 3.6

-- Define the relative speed
def relative_speed_mps := length_of_train_meters / time_to_pass_seconds
def relative_speed_kmph := relative_speed_mps * conversion_factor

-- Lean 4 statement: Proving the speed of the train in kmph
theorem speed_of_train : relative_speed_kmph - speed_of_man_kmph = 78 := by
  sorry

end speed_of_train_l603_603880


namespace enclosed_area_computation_l603_603753

noncomputable def enclosed_area : ℝ :=
  let arc_length := (3 * Real.pi) / 4
  let num_arcs := 12
  let side_length := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * side_length ^ 2
  let sector_area := (3 * Real.pi / 4) * ((3 / 2) ^ 2 * Real.pi / 4) 
  let total_sector_area := num_arcs * (9 * Real.pi / 4) in
  octagon_area + total_sector_area

theorem enclosed_area_computation : enclosed_area = 54 + 54 * Real.sqrt 2 + 4 * Real.pi :=
  sorry

end enclosed_area_computation_l603_603753


namespace part1_geometric_seq_part2_general_formula_part2_sum_l603_603192

-- Given conditions
def a : ℕ → ℚ
| 1       := 7 / 2
| (n + 1) := 3 * a n - 1

-- Part (1): Prove that the sequence {a_n - 1 / 2} is a geometric sequence with ratio 3
theorem part1_geometric_seq (n : ℕ) (hn : n ≥ 1) : 
    (a (n + 1) - 1 / 2) = 3 * (a n - 1 / 2) :=
sorry

-- Part (2): Prove the general formula for a_n
theorem part2_general_formula (n : ℕ) : 
    a n = 3 ^ n + 1 / 2 :=
sorry

-- Part (2): Prove the sum of the first n terms
theorem part2_sum (n : ℕ) : 
    (finset.range n).sum (λ k, a (k + 1)) = (3 ^ (n + 1) - 3 + n) / 2 := 
sorry

end part1_geometric_seq_part2_general_formula_part2_sum_l603_603192


namespace problem1_problem2_problem3_l603_603195

variables {m k : ℝ} (h : m > 0) (k_pos : k > 0) (k_ne_3 : k ≠ 3)

-- Problem 1: Prove the range of dot product for given conditions.
theorem problem1 
  (F1 F2 K : ℝ × ℝ) 
  (hF1 : F1 = (0, 2 * real.sqrt 2))
  (hF2 : F2 = (0, -2 * real.sqrt 2))
  (hK : K = (real.cos 0, 3 * real.sin 0)) : 
  ∃ α, -7 ≤ real.cos α ^ 2 + 9 * real.sin α ^ 2 - 8 ≤ 1 :=
sorry

-- Problem 2: Prove the product of slopes is a constant -9.
theorem problem2 
  (OM l : ℝ → ℝ)
  (slope_OM : ∀ x, OM x = (-9 / k) * x)
  (slope_l : ∀ x, l x = k * x) :
  (∀ x, (slope_OM x) * (slope_l x)) = - 9 := 
sorry

-- Problem 3: Verify the conditions for making the quadrilateral a parallelogram.
theorem problem3
  (O : ℝ × ℝ) 
  (A B M : ℝ × ℝ) 
  (P : ℝ × ℝ)
  (hl : ∀ x, B = (m / 3, m))
  (h_slope_l : l x = k * (x - m / 3) + m) :
  (2 * (-((m - m * k / 3) * k) / (k ^ 2 + 9)) = real.sqrt (m^2 * k^2 / (9 * k^2 + 81))) →
  (k = 4 + real.sqrt 7 ∨ k = 4 - real.sqrt 7) := 
sorry

end problem1_problem2_problem3_l603_603195


namespace number_of_blue_fish_l603_603731

def total_fish : ℕ := 22
def goldfish : ℕ := 15
def blue_fish : ℕ := total_fish - goldfish

theorem number_of_blue_fish : blue_fish = 7 :=
by
  -- proof goes here
  sorry

end number_of_blue_fish_l603_603731


namespace average_speed_l603_603447

-- Definitions based on the conditions from part (a)
def total_distance : ℝ := 250
def first_leg_distance : ℝ := 100
def first_leg_speed : ℝ := 20
def second_leg_distance : ℝ := total_distance - first_leg_distance
def second_leg_speed : ℝ := 15

-- The statement proving the average speed is 16.67 km/h
theorem average_speed (total_distance = 250) (first_leg_distance = 100)
    (first_leg_speed = 20) (second_leg_distance = total_distance - first_leg_distance)
    (second_leg_speed = 15) :
    ((total_distance) / ((first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed))) = 16.67 := 
sorry

end average_speed_l603_603447


namespace geometric_sequence_sum_l603_603208

-- Define the relations for geometric sequences
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (m n p q : ℕ), m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n, a n > 0)
  (h_cond : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 :=
sorry

end geometric_sequence_sum_l603_603208


namespace sum_of_exterior_angles_convex_pentagon_l603_603839

theorem sum_of_exterior_angles_convex_pentagon (P : Type) [convex_polygon P] (h: sides_count P = 5) : 
  sum_exterior_angles P = 360 :=
sorry

end sum_of_exterior_angles_convex_pentagon_l603_603839


namespace find_n_l603_603264

theorem find_n (n : ℕ) : (1/2)^n * (1/81)^12.5 = 1/(18^25) → n = 25 :=
by
  sorry

end find_n_l603_603264


namespace find_a_l603_603135

theorem find_a :
  ∃ a : ℝ, (∀ t1 t2 : ℝ, t1 + t2 = -a ∧ t1 * t2 = -2017 ∧ 2 * t1 = 4) → a = 1006.5 :=
by
  sorry

end find_a_l603_603135


namespace rank_inequality_l603_603685

noncomputable def A : Matrix (Fin n) (Fin n) ℝ := sorry
noncomputable def B : Matrix (Fin n) (Fin n) ℝ := sorry
def n : ℕ := sorry
def n_ge_2 : n ≥ 2 := sorry
def B_square_eq_B : B ⬝ B = B := sorry

theorem rank_inequality (A B : Matrix (Fin n) (Fin n) ℝ) (n_ge_2 : n ≥ 2) (B_square_eq_B : B ⬝ B = B) :
  rank (A ⬝ B - B ⬝ A) ≤ rank (A ⬝ B + B ⬝ A) := 
sorry

end rank_inequality_l603_603685


namespace centroid_positions_count_l603_603376

theorem centroid_positions_count :
  let vertices := [(0, 0), (12, 0), (12, 8), (0, 8)]
  let points := set_of (x, y) | 
    (x = 0 ∧ 0 ≤ y ∧ y ≤ 8 ∧ y % 1 = 0) ∨
    (x = 12 ∧ 0 ≤ y ∧ y ≤ 8 ∧ y % 1 = 0) ∨
    (y = 0 ∧ 0 ≤ x ∧ x ≤ 12 ∧ x % 1.33 = 0) ∨
    (y = 8 ∧ 0 ≤ x ∧ x ≤ 12 ∧ x % 1.33 = 0)
  let centroid (P Q R : (ℝ × ℝ)) := 
    ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)
  let possible_centroids := set_of (Gx, Gy) | 
    ∃ P Q R : (ℝ × ℝ), P ∈ points ∧ Q ∈ points ∧ R ∈ points ∧
    P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧ centroid P Q R = (Gx, Gy)
  in possible_centroids.card = 925 := sorry

end centroid_positions_count_l603_603376


namespace polynomial_abs_sum_l603_603330

theorem polynomial_abs_sum : 
  let a := 2 in let b := -1 in 
  (a * x + b)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0 → 
  (|a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6|) = 3^6 :=
by
  intro a b
  sorry

end polynomial_abs_sum_l603_603330


namespace product_closest_to_1700_l603_603023

noncomputable def approximate_product_is_1700 : Prop :=
  let a := 0.000258
  let b := 6539721
  let p := a * b
  let approximations := 1700
  (approximations - p).abs <= List.minimum ((List.map (fun x => (x - p).abs) [1600, 1700, 1800, 1900, 2000]))

theorem product_closest_to_1700 : approximate_product_is_1700 := sorry

end product_closest_to_1700_l603_603023


namespace yellow_beads_count_l603_603090

theorem yellow_beads_count (Y : ℕ) :
    let total_beads := 23 + Y in 
    let total_removed := 3 * 10 in 
    let beads_per_part_doubled := 6 in 
    let beads_before_doubling := beads_per_part_doubled / 2 in 
    let total_remaining_beads := 3 * beads_before_doubling in 
    let total_beads_before_removal := total_remaining_beads + total_removed in 
    let blue_beads := 23 in 
    Y = total_beads_before_removal - blue_beads →
    Y = 16 :=
by
    intros h
    rw [← h]
    rfl


end yellow_beads_count_l603_603090


namespace intersection_A_B_l603_603205

open Set

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }
def B : Set ℕ := { x | x^2 - 3 * x - 4 < 0 }

theorem intersection_A_B : (A ∩ B) = {0, 1, 2} := sorry

end intersection_A_B_l603_603205


namespace quadratic_eq_roots_quadratic_eq_range_l603_603973

theorem quadratic_eq_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0 ∧ x1 + 3 * x2 = 2 * m + 8) →
  (m = -1 ∨ m = -2) :=
sorry

theorem quadratic_eq_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 2 * x1 + m + 1 = 0 ∧ x2^2 - 2 * x2 + m + 1 = 0) →
  m ≤ 0 :=
sorry

end quadratic_eq_roots_quadratic_eq_range_l603_603973


namespace nonnegative_solution_exists_l603_603185

theorem nonnegative_solution_exists
  (a b c d n : ℕ)
  (h_npos : 0 < n)
  (h_gcd_abc : Nat.gcd (Nat.gcd a b) c = 1)
  (h_gcd_ab : Nat.gcd a b = d)
  (h_conds : n > a * b / d + c * d - a - b - c) :
  ∃ x y z : ℕ, a * x + b * y + c * z = n := 
by
  sorry

end nonnegative_solution_exists_l603_603185


namespace rectangles_from_lines_l603_603154

theorem rectangles_from_lines : 
  (∃ (h_lines : ℕ) (v_lines : ℕ), (h_lines = 5 ∧ v_lines = 4) ∧ (nat.choose h_lines 2 * nat.choose v_lines 2 = 60)) :=
by
  let h_lines := 5
  let v_lines := 4
  have h_choice := nat.choose h_lines 2
  have v_choice := nat.choose v_lines 2
  have answer := h_choice * v_choice
  exact ⟨h_lines, v_lines, ⟨rfl, rfl⟩, rfl⟩

end rectangles_from_lines_l603_603154


namespace earliest_time_for_84_degrees_l603_603645

noncomputable def temperature (t : ℝ) : ℝ := -t^2 + 14 * t + 40

theorem earliest_time_for_84_degrees : ∃ t : ℝ, t = 22 ∧ temperature t = 84 :=
by {
  use 22,
  split,
  exact rfl,
  sorry -- Actual proof to be filled in.
}

end earliest_time_for_84_degrees_l603_603645


namespace estimate_fitness_population_l603_603411

theorem estimate_fitness_population :
  ∀ (sample_size total_population : ℕ) (sample_met_standards : Nat) (percentage_met_standards estimated_met_standards : ℝ),
  sample_size = 1000 →
  total_population = 1200000 →
  sample_met_standards = 950 →
  percentage_met_standards = (sample_met_standards : ℝ) / (sample_size : ℝ) →
  estimated_met_standards = percentage_met_standards * (total_population : ℝ) →
  estimated_met_standards = 1140000 := by sorry

end estimate_fitness_population_l603_603411


namespace sin_squared_plus_sin_range_l603_603538

theorem sin_squared_plus_sin_range : set.range (λ x : ℝ, (Real.sin x)^2 + Real.sin x - 1) = set.Icc (-(5 / 4)) 1 :=
by
  sorry

end sin_squared_plus_sin_range_l603_603538


namespace correct_statement_l603_603765

theorem correct_statement :
  ∃ (s : Nat), (s = 4) ∧
  ( ∀ (l : Line) (p : Point), ¬(p ∈ l) → ∃ (m : Line), (m ≠ l) ∧ (p ∈ m) ∧ (∀ (q : Point), q ∈ l → distance p q = distance p l)) :=
by sorry

end correct_statement_l603_603765


namespace rope_length_in_inches_l603_603542

theorem rope_length_in_inches :
  let week1 := 6 in
  let week2 := 2 * week1 in
  let week3 := week2 - 4 in
  let week4 := week3 / 2 in
  let total_feet := (week1 + week2 + week3) - week4 in
  let total_inches := total_feet * 12 in
  total_inches = 264 :=
by
  simp [total_inches]
  sorry

end rope_length_in_inches_l603_603542


namespace other_root_l603_603271

theorem other_root (m : ℤ) (h : (∀ x : ℤ, x^2 - x + m = 0 → (x = 2))) : (¬ ∃ y : ℤ, (y^2 - y + m = 0 ∧ y ≠ 2 ∧ y ≠ -1) ) := 
by {
  sorry
}

end other_root_l603_603271


namespace exponential_monotone_l603_603985

theorem exponential_monotone {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b :=
sorry

end exponential_monotone_l603_603985


namespace preliminary_articles_day_is_tuesday_l603_603749

-- Define the problem conditions
def TreatyOfParisDay : Nat := 4  -- Thursday corresponds to 4 (assuming Monday is 1)
def DaysBefore : Nat := 621

-- Define a function to calculate day of the week after moving x days
def day_of_week (initial_day : Nat) (days_change : Int) : Nat := 
  ((initial_day + days_change) % 7 + 7) % 7

-- State the theorem to prove
theorem preliminary_articles_day_is_tuesday :
  day_of_week TreatyOfParisDay (- (DaysBefore : Int)) = 2 := by 
  sorry

end preliminary_articles_day_is_tuesday_l603_603749


namespace compute_3_oplus_4_l603_603519

def op (a b x y : ℝ) : ℝ := a * x + b * y - a * b

axiom a_nonzero {a : ℝ} : a ≠ 0
axiom b_nonzero {b : ℝ} : b ≠ 0
axiom condition1 {a b : ℝ} : op a b 1 2 = 3
axiom condition2 {a b : ℝ} : op a b 2 3 = 6

theorem compute_3_oplus_4 (a b : ℝ) [h1: a_nonzero] [h2: b_nonzero] [cond1: condition1] [cond2: condition2] :
  op a b 3 4 = 9 :=
sorry

end compute_3_oplus_4_l603_603519


namespace black_marbles_count_l603_603403

theorem black_marbles_count :
  ∀ (white_marbles total_marbles : ℕ), 
  white_marbles = 19 → total_marbles = 37 → total_marbles - white_marbles = 18 :=
by
  intros white_marbles total_marbles h_white h_total
  sorry

end black_marbles_count_l603_603403


namespace train_cross_pole_time_l603_603493

-- Define the conditions
def speed_km_per_hr : ℝ := 36
def train_length_meters : ℝ := 90

-- Convert speed from km/hr to m/s
def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)

-- State the theorem
theorem train_cross_pole_time : train_length_meters / speed_m_per_s = 9 := by
  sorry

end train_cross_pole_time_l603_603493


namespace trigonometric_identity_in_triangle_l603_603281

theorem trigonometric_identity_in_triangle
  (A B C : ℝ) (a b c h : ℝ)
  (triangle_ABC : a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π ∧ a^2 + b^2 = c^2)
  (height_condition : c - a = h)
  : (cos (A / 2) - sin (A / 2)) * (sin (C / 2) + cos (C / 2)) = 1 := by
sorry

end trigonometric_identity_in_triangle_l603_603281


namespace integer_values_satisfying_sqrt_inequality_l603_603811

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l603_603811


namespace rectangle_enclosed_by_lines_l603_603156

theorem rectangle_enclosed_by_lines :
  ∀ (H : ℕ) (V : ℕ), H = 5 → V = 4 → (nat.choose H 2) * (nat.choose V 2) = 60 :=
by
  intros H V h_eq H_eq
  rw [h_eq, H_eq]
  simp
  sorry

end rectangle_enclosed_by_lines_l603_603156


namespace Alden_nephews_10_years_ago_l603_603897

noncomputable def nephews_Alden_now : ℕ := sorry
noncomputable def nephews_Alden_10_years_ago (N : ℕ) : ℕ := N / 2
noncomputable def nephews_Vihaan_now (N : ℕ) : ℕ := N + 60
noncomputable def total_nephews (N : ℕ) : ℕ := N + (nephews_Vihaan_now N)

theorem Alden_nephews_10_years_ago (N : ℕ) (h1 : total_nephews N = 260) : 
  nephews_Alden_10_years_ago N = 50 :=
by
  sorry

end Alden_nephews_10_years_ago_l603_603897


namespace find_x_in_inches_l603_603868

noncomputable def x_value (x : ℝ) : Prop :=
  let area_larger_square := (4 * x) ^ 2
  let area_smaller_square := (3 * x) ^ 2
  let area_triangle := (1 / 2) * (3 * x) * (4 * x)
  let total_area := area_larger_square + area_smaller_square + area_triangle
  total_area = 1100 ∧ x = Real.sqrt (1100 / 31)

theorem find_x_in_inches (x : ℝ) : x_value x :=
by sorry

end find_x_in_inches_l603_603868


namespace prime_factors_M_l603_603243

noncomputable def M : ℕ := sorry

theorem prime_factors_M :
  (∃ M : ℕ, log 3 (log 5 (log 7 (log 11 M))) = 7 → count_distinct_prime_factors M = 1) :=
sorry

end prime_factors_M_l603_603243


namespace number_of_white_balls_l603_603659

-- Definitions based on the problem conditions
def total_balls : Nat := 120
def red_freq : ℝ := 0.15
def black_freq : ℝ := 0.45

-- Result to prove
theorem number_of_white_balls :
  let red_balls := total_balls * red_freq
  let black_balls := total_balls * black_freq
  total_balls - red_balls - black_balls = 48 :=
by
  sorry

end number_of_white_balls_l603_603659


namespace number_of_rectangles_l603_603150

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 4) : 
  (nat.choose H 2) * (nat.choose V 2) = 60 := by
  rw [hH, hV]
  norm_num

end number_of_rectangles_l603_603150


namespace train_length_l603_603889

theorem train_length (t_platform t_pole : ℕ) (platform_length : ℕ) (train_length : ℕ) :
  t_platform = 39 → t_pole = 18 → platform_length = 350 →
  (train_length + platform_length) / t_platform = train_length / t_pole →
  train_length = 300 :=
by
  intros ht_platform ht_pole hplatform_length hspeeds 
  have h1 : train_length / 18 = (train_length + 350) / 39, from hspeeds
  have h2 : 39 * (train_length / 18) = 39 * ((train_length + 350) / 39), from congrArg (λ x, 39 * x) h1
  sorry

end train_length_l603_603889


namespace sequence_solution_l603_603974

variable {r : ℝ} -- assume r is a real number

def a_n (n : ℕ) : ℕ → ℝ 
| 0 => 1  -- indexing starts at 1 for a_1
| (n+1) => a_n n + r^n -- recursive definition reflecting a_{i+1} - a_i = r^i

theorem sequence_solution (n : ℕ) : a_n n = (1 - r^(n-1)) / (1 - r) :=
sorry

end sequence_solution_l603_603974


namespace managers_non_managers_ratio_l603_603649

theorem managers_non_managers_ratio
  (M N : ℕ)
  (h_ratio : M / N > 7 / 24)
  (h_max_non_managers : N = 27) :
  ∃ M, 8 ≤ M ∧ M / 27 > 7 / 24 :=
by
  sorry

end managers_non_managers_ratio_l603_603649


namespace max_population_supported_l603_603910

theorem max_population_supported
  (growth_constant : ∀ (t : ℕ), growth t.succ - growth t = growth 1)
  (support_11_90 : ∀ t, t <= 90 → resources t 11 * t = 9900)
  (support_9_210 : ∀ t, t <= 210 → resources t 9 * t = 18900):
  ( ∀ t, growth t = growth_rate * t ) ∧
  ( ∀ p, resources t p = p ) → 
  ( ∀ t, growth t = resources t 75 ) := 
begin
  sorry
end

end max_population_supported_l603_603910


namespace expected_earnings_per_hour_l603_603082

def earnings_per_hour (words_per_minute earnings_per_word : ℝ) (earnings_per_article num_articles total_hours : ℕ) : ℝ :=
  let minutes_in_hour := 60
  let total_time := total_hours * minutes_in_hour
  let total_words := total_time * words_per_minute
  let word_earnings := total_words * earnings_per_word
  let article_earnings := earnings_per_article * num_articles
  (word_earnings + article_earnings) / total_hours

theorem expected_earnings_per_hour :
  earnings_per_hour 10 0.1 60 3 4 = 105 := by
  sorry

end expected_earnings_per_hour_l603_603082


namespace function_relationship_l603_603470

variable {R : Type} [LinearOrderedField R] {f : R → R} {f'' : R → R}
variables {x_1 x_2 : R}

def is_symmetric_about (f : R → R) (a : R) : Prop :=
∀ x, f (2*a - x) = f x

def concavity_condition (x : R) (f'' : R → R) : Prop :=
(x - 1) * f'' x < 0

theorem function_relationship 
  (h_deriv : ∀ x, deriv (deriv f x) = f'' x)
  (h_symm : is_symmetric_about f 1)
  (h_concave : ∀ x, concavity_condition x f'')
  (h_less : x_1 < x_2)
  (h_sum : x_1 + x_2 > 2) : 
  f x_1 > f x_2 :=
sorry

end function_relationship_l603_603470


namespace sin_25_over_6_pi_l603_603145

noncomputable def sin_value : ℝ :=
  Real.sin (25 / 6 * Real.pi)

theorem sin_25_over_6_pi : sin_value = 1 / 2 := by
  sorry

end sin_25_over_6_pi_l603_603145


namespace value_after_four_bailouts_correct_l603_603386

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 1 / 980
  else a (n - 1) * (2^(2^(n - 1)) + 1) / 2^(2^(n - 1))

def b_c : ℕ × ℕ :=
  (490, 16)

theorem value_after_four_bailouts_correct :
  let (b, c) := b_c in
  a 4 = (1 : ℚ) / b * (1 - 1 / (2^c)) → b + c = 506 :=
begin
  intro h,
  sorry, -- Proof goes here
end

end value_after_four_bailouts_correct_l603_603386


namespace number_of_odd_coefficients_l603_603319

theorem number_of_odd_coefficients (a : ℕ → ℕ) (h : (∑ i in Finset.range 9, a i * x^i) = (1 + x)^8) :
  ∃ n: ℕ, n = 2 ∧ (Finset.filter (λ i, Nat.odd (a i)) (Finset.range 9)).card = n :=
sorry

end number_of_odd_coefficients_l603_603319


namespace minimum_cos_minus_sin_l603_603171

noncomputable def cos_minus_sin_min_value : Prop :=
  ∀ (x y z : ℝ),
  (2 * Real.sin x = Real.tan y) ∧
  (2 * Real.cos y = Real.cot z) ∧
  (Real.sin z = Real.tan x) →
  Real.cos x - Real.sin z = - (5 * Real.sqrt 3) / 6

theorem minimum_cos_minus_sin (x y z : ℝ) :
  2 * Real.sin x = Real.tan y →
  2 * Real.cos y = Real.cot z →
  Real.sin z = Real.tan x →
  Real.cos x - Real.sin z = - (5 * Real.sqrt 3) / 6 := sorry

end minimum_cos_minus_sin_l603_603171


namespace johns_pace_l603_603312

variable {J : ℝ} -- John's pace during his final push

theorem johns_pace
  (steve_speed : ℝ := 3.8)
  (initial_gap : ℝ := 15)
  (finish_gap : ℝ := 2)
  (time : ℝ := 42.5)
  (steve_covered : ℝ := steve_speed * time)
  (john_covered : ℝ := steve_covered + initial_gap + finish_gap)
  (johns_pace_equation : J * time = john_covered) :
  J = 4.188 :=
by
  sorry

end johns_pace_l603_603312


namespace earnings_per_hour_l603_603078

-- Define the conditions and the respective constants
def words_per_minute : ℕ := 10
def earnings_per_word : ℝ := 0.1
def earnings_per_article : ℝ := 60
def number_of_articles : ℕ := 3
def total_hours : ℕ := 4
def minutes_per_hour : ℕ := 60

theorem earnings_per_hour :
  let total_words := words_per_minute * minutes_per_hour * total_hours in
  let earnings_from_words := earnings_per_word * total_words in
  let earnings_from_articles := earnings_per_article * number_of_articles in
  let total_earnings := earnings_from_words + earnings_from_articles in
  let expected_earnings_per_hour := total_earnings / total_hours in
  expected_earnings_per_hour = 105 := 
  sorry

end earnings_per_hour_l603_603078


namespace hexagon_coloring_count_l603_603937

def num_possible_colorings : Nat :=
by
  /- There are 7 choices for first vertex A.
     Once A is chosen, there are 6 choices for the remaining vertices B, C, D, E, F considering the diagonal restrictions. -/
  let total_colorings := 7 * 6 ^ 5
  let restricted_colorings := 7 * 6 ^ 3
  let valid_colorings := total_colorings - restricted_colorings
  exact valid_colorings

theorem hexagon_coloring_count : num_possible_colorings = 52920 :=
  by
    /- Computation steps above show that the number of valid colorings is 52920 -/
    sorry   -- Proof computation already indicated

end hexagon_coloring_count_l603_603937


namespace equivalent_expression_l603_603119

open Complex

noncomputable def sum_complex_expressions : ℂ :=
  (15:ℂ) * exp (π * I / 5) + (15:ℂ) * exp (7 * π * I / 10)

theorem equivalent_expression :
  sum_complex_expressions = (30 * cos (π / 10)) * exp((9 * π * I) / 20) :=
sorry

end equivalent_expression_l603_603119


namespace expr_same_type_l603_603899

-- Defining the expressions
def expr1 (a b : ℕ) : ℕ := -2 * a^2 * b
def expr2 (a b : ℕ) : ℕ := -2 * a * b
def expr3 (a b : ℕ) : ℕ := 2 * a * b^2
def expr4 (a b : ℕ) : ℕ := 2 * a^2
def target_expr (a b : ℕ) : ℕ := 3 * a^2 * b

-- Proving that expr1 is the same type as target_expr
theorem expr_same_type (a b : ℕ) : 
  (expr1 a b = target_expr a b) ∨ (expr2 a b = target_expr a b) ∨ (expr3 a b = target_expr a b) ∨ (expr4 a b = target_expr a b) → 
  (expr1 a b = target_expr a b) :=
by {
  sorry
}


end expr_same_type_l603_603899


namespace simplify_expression_l603_603736

theorem simplify_expression (x : ℝ) :
  ((3 * x^2 + 2 * x - 1) + 2 * x^2) * 4 + (5 - 2 / 2) * (3 * x^2 + 6 * x - 8) = 32 * x^2 + 32 * x - 36 :=
sorry

end simplify_expression_l603_603736


namespace find_max_constant_c_l603_603565

def shining_vector (n : ℕ) (x : Fin n → ℝ) :=
  ∀ (y : Fin n → ℝ), (∀ i, y i ∈ (Finset.finRange n).val.map x) →
  ∑ i in Finset.range (n - 1), y i * y ⟨i + 1, Nat.lt_of_succ_lt_succ (Finset.mem_range.1 (Finset.mem_range_sub1.2 (Nat.succ_pos n)))⟩ ≥ -1

theorem find_max_constant_c (n : ℕ) (hn : n ≥ 3) (x : Fin n → ℝ) (hx : shining_vector n x) :
  ∑ i in Finset.range n, ∑ j in Finset.Ico (i + 1) n, x i * x j ≥ -(n - 1) / 2 :=
sorry

end find_max_constant_c_l603_603565


namespace proof_problem_l603_603957

def f (n : ℕ) (x : ℝ) : ℝ :=
  let frac_part := x - (x.floor : ℝ)
  in (1 - frac_part) * real.binom n x.floor + frac_part * real.binom n (x.floor + 1)

theorem proof_problem (n m : ℕ) (h1 : 2 ≤ n) (h2 : 2 ≤ m)
(h3 : f m (1 / n) + f m (2 / n) + f m ((m * n - 1) / n) = 123) :
  f n (1 / m) + f n (2 / m) + f n ((m * n - 1) / m) = 74 := by
  sorry

end proof_problem_l603_603957


namespace fred_has_6_cards_l603_603551

-- Define initial number of baseball cards
def initial_cards : ℤ := 5
-- Define the number of cards given away to Melanie
def cards_given_melanie : ℤ := 2
-- Define the number of cards traded with Sam (net change is 0)
def cards_traded_sam : ℤ := 0
-- Define the number of cards gifted by Lisa
def cards_received_lisa : ℤ := 3

-- Define Fred's final number of baseball cards
def fred_final_cards : ℤ := initial_cards - cards_given_melanie + cards_traded_sam + cards_received_lisa

-- Prove that Fred has 6 baseball cards now
theorem fred_has_6_cards : fred_final_cards = 6 :=
by
  unfold fred_final_cards
  unfold initial_cards cards_given_melanie cards_traded_sam cards_received_lisa
  calc
    5 - 2 + 0 + 3 = 3 + 3 : by simp
    ... = 6 : by simp

end fred_has_6_cards_l603_603551


namespace intersection_of_lines_l603_603932

noncomputable def intersection_point : ℝ × ℝ := (2.1, 2.3)

theorem intersection_of_lines :
  ∃ (x y : ℝ), (y = 3 * x - 4) ∧ (y = - (1 / 3) * x + 3) ∧ (x, y) = (2.1, 2.3) :=
begin
  use 2.1,
  use 2.3,
  split,
  { -- First line equation y = 3x - 4
    calc 2.3 = 3 * 2.1 - 4 : by norm_num
  },
  split,
  { -- Second line equation y = -(1/3)x + 3
    calc 2.3 = -(1 / 3) * 2.1 + 3 : by norm_num
  },
  { -- Intersection point verification
    refl
  }
end

end intersection_of_lines_l603_603932


namespace fraction_to_decimal_l603_603132

theorem fraction_to_decimal :
  (17 : ℚ) / (2^2 * 5^4) = 0.0068 :=
by
  sorry

end fraction_to_decimal_l603_603132


namespace count_ordered_triples_l603_603618

theorem count_ordered_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : Nat.lcm x y = 72) (h2 : Nat.lcm x z = 600) (h3 : Nat.lcm y z = 900) :
  (Finset.univ.filter (λ t : ℕ × ℕ × ℕ, 
    t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 
    Nat.lcm t.1 t.2.1 = 72 ∧ 
    Nat.lcm t.1 t.2.2 = 600 ∧ 
    Nat.lcm t.2.1 t.2.2 = 900)).card = 15 := 
sorry

end count_ordered_triples_l603_603618


namespace finance_charge_rate_l603_603709

theorem finance_charge_rate (original_balance total_payment finance_charge_rate : ℝ)
    (h1 : original_balance = 150)
    (h2 : total_payment = 153)
    (h3 : finance_charge_rate = ((total_payment - original_balance) / original_balance) * 100) :
    finance_charge_rate = 2 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end finance_charge_rate_l603_603709


namespace max_value_l603_603980

variable (a b c d : ℝ)

theorem max_value 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) 
  (h5 : b ≠ d) (h6 : c ≠ d)
  (cond1 : a / b + b / c + c / d + d / a = 4)
  (cond2 : a * c = b * d) :
  (a / c + b / d + c / a + d / b) ≤ -12 :=
sorry

end max_value_l603_603980


namespace perpendicular_vectors_x_value_l603_603610

theorem perpendicular_vectors_x_value 
  (x : ℝ) (a b : ℝ × ℝ) (hₐ : a = (1, -2)) (hᵦ : b = (3, x)) (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 / 2 :=
by
  -- The proof is not required, hence we use 'sorry'
  sorry

end perpendicular_vectors_x_value_l603_603610


namespace ratio_of_segments_angle_equality_l603_603503

-- Define the geometric configuration
variables {α : Type*} [euclidean_space α]
variables (A B C P D E F : α)
variables {O : circle A B C}

-- Define segments and angles
noncomputable def segments := sorry
noncomputable def angles := sorry 

-- Conditions
def is_tangent_to_circle (P B : α) (O : circle A B C) : Prop := sorry 
def is_parallel (a b : α) : Prop := sorry 
def intersects (a b c : α) : α := sorry 

variable (DE_parallel_BA : is_parallel D E B A)
variable (DF_parallel_CA : is_parallel D F C A)
variable (AP_intersects_BC_at_D : intersects A P B C = D)
variable (PB_tangent : is_tangent_to_circle P B O)
variable (PC_tangent : is_tangent_to_circle P C O)

-- Lean 4 statements for proofs required
theorem ratio_of_segments (h1 : AP_intersects_BC_at_D) 
                          (h2 : DE_parallel_BA) 
                          (h3 : DF_parallel_CA) 
                          (h4 : PB_tangent) 
                          (h5 : PC_tangent) :
  (segments B D / segments C D) = (segments A B ^ 2 / segments A C ^ 2) :=
sorry

theorem angle_equality (h6 : AP_intersects_BC_at_D) 
                       (h7 : DE_parallel_BA) 
                       (h8 : DF_parallel_CA) 
                       (h9 : PB_tangent) 
                       (h10 : PC_tangent) :
  angles B C F = angles B E F :=
sorry

end ratio_of_segments_angle_equality_l603_603503


namespace find_digits_l603_603211

-- Define the digits a, b, c, and d as natural numbers
variables (a b c d : ℕ)

-- Conditions based on the problem
def condition_1 : 34! % 10000000 = 0 := sorry
def condition_2 : (352 + a) % 8 = 0 := sorry
def condition_3 : (295232799 + 100000 * c + 10000 * d + 960414084761860964352) % 9 = 0 := sorry
def condition_4 : (80 + d - 61 - c) % 11 = 0 := sorry

-- Prove that the digits a, b, c, d satisfy the given conditions
theorem find_digits : a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3 :=
by
  have h1 : condition_1 := sorry,
  have h2 : condition_2 := sorry,
  have h3 : condition_3 := sorry,
  have h4 : condition_4 := sorry,
  have : a = 2 := sorry,
  have : b = 0 := sorry,
  have : c = 0 := sorry,
  have : d = 3 := sorry,
  exact ⟨‹a = 2›, ‹b = 0›, ‹c = 0›, ‹d = 3›⟩

end find_digits_l603_603211


namespace sum_of_digits_of_binary_300_l603_603017

theorem sum_of_digits_of_binary_300 : 
  ∑ digit in (Nat.digits 2 300), digit = 3 :=
by
  sorry

end sum_of_digits_of_binary_300_l603_603017


namespace sales_tax_difference_correct_l603_603850

def sales_tax_difference : ℕ := 7800
def original_rate : ℚ := 7 / 200
def new_rate : ℚ := 10 / 300

theorem sales_tax_difference_correct :
  (sales_tax_difference * original_rate - sales_tax_difference * new_rate) ≈ 13.26 := sorry

end sales_tax_difference_correct_l603_603850


namespace friday_birth_of_dickens_l603_603497

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem friday_birth_of_dickens :
  let regular_year_days := 365
  let leap_year_days := 366
  let total_years := 200
  let leap_years := 49  -- already computed as 49 in the steps
  let regular_years := total_years - leap_years
  let total_days_in_regular_years := regular_years * regular_year_days
  let total_days_in_leap_years := leap_years * leap_year_days
  let total_days := total_days_in_regular_years + total_days_in_leap_years
  (total_days % 7 = 4) : 
  day_of_week (date.add_days (date.mk 2012 2 7) (-total_days)) = "Friday"
:= sorry

end friday_birth_of_dickens_l603_603497


namespace given_problem_proof_l603_603280

open Triangle
open Angle

theorem given_problem_proof
  (A B C D E F : Point)
  (hABC : RightTriangle A B C)
  (hBA2C : Angle B = 2 * Angle C)
  (hBisector : angleBisector B D A (AC intersect Point D))
  (hAEperpBC : Perpendicular AE BC)
  (hDFperpBC : Perpendicular DF BC) :
  (1 / (Distance BE * Distance DF)) = (1 / (Distance AE * Distance BF)) + (1 / (Distance AE * Distance BE)) :=
sorry

end given_problem_proof_l603_603280


namespace arithmetic_sequence_common_difference_l603_603651

theorem arithmetic_sequence_common_difference 
  (a l S : ℕ) (h1 : a = 5) (h2 : l = 50) (h3 : S = 495) :
  (∃ d n : ℕ, l = a + (n-1) * d ∧ S = n * (a + l) / 2 ∧ d = 45 / 17) :=
by
  sorry

end arithmetic_sequence_common_difference_l603_603651


namespace max_number_of_satellites_satellite_coordinates_valid_l603_603452

noncomputable def earth_radius : ℝ := sorry

noncomputable def orbit_altitude : ℝ := sorry

noncomputable def satellite_distance (R H : ℝ) : ℝ := sqrt (2 * (R + H) * (R + H))

def max_satellites_orbit (R H : ℝ) : ℕ := sorry

def satellite_coordinates (R H : ℝ) : list (ℝ × ℝ × ℝ) :=
  let RH := R + H in
  [ (RH, 0, 0),
    (-RH / 3, 0, (RH * sqrt 8) / 3),
    (-RH / 3, (RH * sqrt 8) / 6, - (RH * sqrt 8) / 6),
    (-RH / 3, - (RH * sqrt 8) / 6, - (RH * sqrt 8) / 6)
  ]

theorem max_number_of_satellites (R H : ℝ) (h1 : R > 0) (h2 : H > 0)
  : max_satellites_orbit R H = 4 :=
sorry

theorem satellite_coordinates_valid (R H : ℝ) (h1 : R > 0) (h2 : H > 0)
  : satellite_coordinates R H = 
    [ (R + H, 0, 0),
      (- (R + H) / 3, 0, ((R + H) * sqrt 8) / 3),
      (- (R + H) / 3, ((R + H) * sqrt 8) / 6, -((R + H) * sqrt 8) / 6),
      (- (R + H) / 3, - ((R + H) * sqrt 8) / 6, -((R + H) * sqrt 8) / 6)
    ] :=
sorry

end max_number_of_satellites_satellite_coordinates_valid_l603_603452


namespace rectangle_enclosed_by_lines_l603_603157

theorem rectangle_enclosed_by_lines :
  ∀ (H : ℕ) (V : ℕ), H = 5 → V = 4 → (nat.choose H 2) * (nat.choose V 2) = 60 :=
by
  intros H V h_eq H_eq
  rw [h_eq, H_eq]
  simp
  sorry

end rectangle_enclosed_by_lines_l603_603157


namespace hydrangeas_percent_l603_603460

theorem hydrangeas_percent (total_flowers : ℕ) (blue_flowers tulips hydrangeas yellow_flowers daisies : ℕ)
  (h1 : blue_flowers = total_flowers * 3 / 5)
  (h2 : yellow_flowers = total_flowers * 2 / 5)
  (h3 : tulips = blue_flowers / 4)
  (h4 : hydrangeas = blue_flowers * 3 / 4) :
  hydrangeas = total_flowers * 9 / 20 :=
by
  sorry

end hydrangeas_percent_l603_603460


namespace number_of_tuples_l603_603537

theorem number_of_tuples :
  let n := 2012
  in let range := Finset.Ico 0 n
  in let tuples := finfun fun i : fin n => range
  in
    finset.card { (x : fin n → ℕ) | (∀ i, x i ∈ range) ∧
                                     (∑ i in finset.range n, (i + 1) * x ⟨i, sorry⟩) % n = 0 } = n^(n - 1) :=
begin
  sorry
end

end number_of_tuples_l603_603537


namespace geometric_sequence_sum_l603_603288

noncomputable def sum_of_geometric_sequence (a : ℕ → ℝ) (a2 : a 2 = 12) (a3 : a 3 = 3) : ℝ :=
  ∑' n, a n

theorem geometric_sequence_sum :
  ∀ (a : ℕ → ℝ), (a 2 = 12) → (a 3 = 3) → sum_of_geometric_sequence a (by assumption) (by assumption) = 64 := by
  sorry

end geometric_sequence_sum_l603_603288


namespace probability_divisor_of_12_on_8_sided_die_l603_603468

-- Define the 8-sided die
def die_faces := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the divisors of 12 that are within the die faces
def valid_divisors := {n ∈ die_faces | n ∣ 12}

-- Assuming die_faces and valid_divisors are finite sets,
-- calculate the probability of rolling a valid divisor
theorem probability_divisor_of_12_on_8_sided_die : 
  (finset.card valid_divisors) / (finset.card die_faces) = 5 / 8 := 
sorry

end probability_divisor_of_12_on_8_sided_die_l603_603468


namespace reflected_midpoint_sum_l603_603354

theorem reflected_midpoint_sum (A B N : ℝ × ℝ) (x₁ x₂ y₁ y₂ : ℝ) (hx₁ : A = (x₁, y₁)) (hx₂ : B = (x₂, y₂)) (hmx : N = ((x₁ + x₂) / 2, (y₁ + y₂) / 2)) :
  let N' := (-N.1, N.2) in N'.1 + N'.2 = 3 :=
by
  sorry

end reflected_midpoint_sum_l603_603354


namespace maximal_odd_numbers_in_polygon_l603_603733

/-- 
  Sasha and Serg play a game with a 100-angled regular polygon. 
  Initially, Sasha sets natural numbers at each vertex. 
  On Serg's turn, he adds 1 to two opposite vertices. 
  On Sasha's turn, she adds 1 to two neighboring vertices. 
  Prove that the maximal number of odd numbers Serg can achieve no matter how Sasha plays is 27. 
-/
theorem maximal_odd_numbers_in_polygon:
  ∀ (a : Fin 100 → ℕ), 
  (∀ n, (exists opposite_edges (Serg_turn_turn n a) xor_vertex_turn Sasha_turn_turn)), 
  max_odd_numbers a = 27 :=
sorry

end maximal_odd_numbers_in_polygon_l603_603733


namespace midpoint_quadrilateral_area_half_equal_diagonals_area_midpoints_product_l603_603041

-- Part (a): Prove that the midpoint quadrilateral's area is half of the original quadrilateral's area
theorem midpoint_quadrilateral_area_half (A B C D E F G H : Point) 
  (h_convex: Convex ABCD)
  (h_E: E = midpoint A B)
  (h_F: F = midpoint B C)
  (h_G: G = midpoint C D)
  (h_H: H = midpoint D A) :
  area (Quadrilateral E F G H) = (1 / 2) * area (Quadrilateral A B C D) :=
sorry

-- Part (b): Prove that if the diagonals are equal, the area is the product of the lengths of the segments joining midpoints of opposite sides
theorem equal_diagonals_area_midpoints_product (A B C D E F G H : Point)
  (h_convex: Convex ABCD)
  (h_diagonals_equal: distance A C = distance B D)
  (h_E: E = midpoint A B)
  (h_F: F = midpoint B C)
  (h_G: G = midpoint C D)
  (h_H: H = midpoint D A) :
  area (Quadrilateral A B C D) = distance E G * distance F H :=
sorry

end midpoint_quadrilateral_area_half_equal_diagonals_area_midpoints_product_l603_603041


namespace zero_point_interval_l603_603227

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

theorem zero_point_interval : ∃ c ∈ set.Ioo (1 : ℝ) 2, f c = 0 :=
begin
  sorry
end

end zero_point_interval_l603_603227


namespace lcm_9_12_15_l603_603004

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l603_603004


namespace angle_C_length_CD_l603_603672

variable {a b c : ℝ} {A B C : ℝ}

def condition₁ (a b c A B C : ℝ) := 2 * a^2 * sin B * sin C = sqrt 3 * (a^2 + b^2 - c^2) * sin A

theorem angle_C (h : condition₁ a b c A B C) : C = π / 3 :=
sorry

theorem length_CD (a b : ℝ) (h : a = 1 ∧ b = 2) : let D := (a + b) / 2 in sqrt ((h.1 + h.2 + (h.2 - h.1)^2) / 4) = sqrt 7 / 2 :=
sorry

end angle_C_length_CD_l603_603672


namespace sum_of_digits_base2_300_l603_603008

theorem sum_of_digits_base2_300 : 
  let n := 300
  in (Int.digits 2 n).sum = 4 :=
by
  let n := 300
  have h : Int.digits 2 n = [1, 0, 0, 1, 0, 1, 1, 0, 0] := by sorry
  rw h
  norm_num
  -- or directly
  -- exact rfl

end sum_of_digits_base2_300_l603_603008


namespace average_results_combined_l603_603378

noncomputable def combined_average (sum1 sum2 : ℕ) (count1 count2 : ℕ) : ℕ :=
(sum1 + sum2) / (count1 + count2)

theorem average_results_combined :
  ∀ (count1 count2 sum1 sum2 : ℕ), 
  count1 = 80 → count2 = 50 →
  sum1 = 2560 → sum2 = 2800 →
  combined_average sum1 sum2 count1 count2 = 41 :=
begin
  intros,
  sorry
end

end average_results_combined_l603_603378


namespace annika_hike_distance_l603_603100

-- Define the conditions as definitions
def hiking_rate : ℝ := 10  -- rate of 10 minutes per kilometer
def total_minutes : ℝ := 35 -- total available time in minutes
def total_distance_east : ℝ := 3 -- total distance hiked east

-- Define the statement to prove
theorem annika_hike_distance : ∃ (x : ℝ), (x / hiking_rate) + ((total_distance_east - x) / hiking_rate) = (total_minutes - 30) / hiking_rate :=
by
  sorry

end annika_hike_distance_l603_603100


namespace relationship_among_abc_l603_603184

-- Given conditions
def a : ℝ := (5 / 3) ^ 0.2
def b : ℝ := (2 / 3) ^ 10
def c : ℝ := Real.logBase 0.3 6

-- to prove
theorem relationship_among_abc : a > b ∧ b > c :=
by {
  -- Proof is omitted
  sorry
}

end relationship_among_abc_l603_603184


namespace soup_ingredients_l603_603414

theorem soup_ingredients :
  ∃ (water grains potatoes onions fat : ℝ),
  water = weight_of_grains + potatoes + onions + fat ∧
  grains = potatoes + onions + fat ∧
  potatoes = onions + fat ∧
  fat = onions / 2 ∧
  water + grains + potatoes + onions + fat = 12 ∧
  water = 6 ∧
  grains = 3 ∧
  potatoes = 1.5 ∧
  onions = 1 ∧
  fat = 0.5 :=
begin
  -- Proof goes here
  sorry
end

end soup_ingredients_l603_603414


namespace selling_price_750_max_daily_profit_l603_603292

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 10) * (-10 * x + 300)

theorem selling_price_750 (x : ℝ) : profit x = 750 ↔ (x = 15 ∨ x = 25) :=
by sorry

theorem max_daily_profit : (∀ x : ℝ, profit x ≤ 1000) ∧ (profit 20 = 1000) :=
by sorry

end selling_price_750_max_daily_profit_l603_603292


namespace percentage_of_first_to_second_l603_603448

theorem percentage_of_first_to_second (x : ℝ) :
  let first := 1.71 * x in
  let second := 1.80 * x in
  first / second * 100 = 95 := by
  sorry

end percentage_of_first_to_second_l603_603448


namespace evaluate_expression_l603_603129

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end evaluate_expression_l603_603129


namespace solve_for_nabla_l603_603621

theorem solve_for_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by
  sorry

end solve_for_nabla_l603_603621


namespace continuous_linear_function_continuous_quadratic_function_l603_603695

open Real

-- First part of the proof
theorem continuous_linear_function (f : ℝ → ℝ) (a : ℝ) 
  (hf_cont : Continuous f) 
  (hf_zero : f 0 = 0)
  (hf_add : ∀ x y : ℝ, f (x + y) ≥ f x + f y)
  (hf_form : ∀ x : ℝ, f x = a * x) : 
  ∀ x : ℝ, f x = a * x := 
sorry

-- Second part of the proof
theorem continuous_quadratic_function (f : ℝ → ℝ) (a : ℝ)
  (hf_cont : Continuous f) 
  (hf_nonneg : ∀ x : ℝ, f x ≥ 0)
  (hf_zero : f 0 = 0) 
  (hf_add : ∀ x y : ℝ, f (x + y) ≥ f x + f y + 2 * real.sqrt (f x * f y)) :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = a * x ^ 2 := 
sorry

end continuous_linear_function_continuous_quadratic_function_l603_603695


namespace f_divisible_by_13_l603_603703

theorem f_divisible_by_13 (f : ℕ → ℤ) : 
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ v, v ≥ 1 → f (v+2) = 4 ^ (v+2) * f (v+1) - 16 ^ (v+1) * f v + v * 2 ^ (v^2)) →
  f 1989 % 13 = 0 ∧ f 1990 % 13 = 0 ∧ f 1991 % 13 = 0 :=
by {
  intro h,
  cases h with h0 h,
  cases h with h1 hrec,
  sorry
}

end f_divisible_by_13_l603_603703


namespace line_intersects_circle_l603_603871

theorem line_intersects_circle (m : ℝ) : 
  ∃ (x y : ℝ), y = m * x - 3 ∧ x^2 + (y - 1)^2 = 25 :=
sorry

end line_intersects_circle_l603_603871


namespace dickens_birth_day_l603_603500

def is_leap_year (year : ℕ) : Prop :=
  (year % 400 = 0) ∨ (year % 4 = 0 ∧ year % 100 ≠ 0)

theorem dickens_birth_day :
  let day_of_week_2012 := 2 -- 0: Sunday, 1: Monday, ..., 2: Tuesday
  let years := 200
  let regular_years := 151
  let leap_years := 49
  let days_shift := regular_years + 2 * leap_years
  let day_of_week_birth := (day_of_week_2012 + days_shift) % 7
  day_of_week_birth = 5 -- 5: Friday
:= 
sorry -- proof not supplied

end dickens_birth_day_l603_603500


namespace decimal_subtraction_l603_603006

theorem decimal_subtraction (a b : ℝ) (h1 : a = 3.79) (h2 : b = 2.15) : a - b = 1.64 := by
  rw [h1, h2]
  -- This follows from the correct calculation rule
  sorry

end decimal_subtraction_l603_603006


namespace jeremie_friends_l603_603310

-- Define the costs as constants.
def ticket_cost : ℕ := 18
def snack_cost : ℕ := 5
def total_cost : ℕ := 92
def per_person_cost : ℕ := ticket_cost + snack_cost

-- Define the number of friends Jeremie is going with (to be solved/proven).
def number_of_friends (total_cost : ℕ) (per_person_cost : ℕ) : ℕ :=
  let total_people := total_cost / per_person_cost
  total_people - 1

-- The statement that we want to prove.
theorem jeremie_friends : number_of_friends total_cost per_person_cost = 3 := by
  sorry

end jeremie_friends_l603_603310


namespace vector_problem_l603_603750

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem vector_problem
  (a b : ℝ × ℝ)
  (h1 : dot_product a b = 0)
  (h2 : vector_length a = 1)
  (h3 : vector_length b = 1) :
  vector_length (3 * a.1 - 2 * b.1, 3 * a.2 - 2 * b.2) = 1 :=
by
  sorry

end vector_problem_l603_603750


namespace expected_earnings_per_hour_l603_603084

def earnings_per_hour (words_per_minute earnings_per_word : ℝ) (earnings_per_article num_articles total_hours : ℕ) : ℝ :=
  let minutes_in_hour := 60
  let total_time := total_hours * minutes_in_hour
  let total_words := total_time * words_per_minute
  let word_earnings := total_words * earnings_per_word
  let article_earnings := earnings_per_article * num_articles
  (word_earnings + article_earnings) / total_hours

theorem expected_earnings_per_hour :
  earnings_per_hour 10 0.1 60 3 4 = 105 := by
  sorry

end expected_earnings_per_hour_l603_603084


namespace age_of_b_l603_603443

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_of_b_l603_603443


namespace magnitude_difference_l603_603585

variable (a b : ℝ^3)
variable (angle_ab : ℝ)
variable (len_a len_b : ℝ)

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (v1 v2 : ℝ^3) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def cos_angle (θ : ℝ) : ℝ := real.cos θ

axiom angle_a_b_is_120 : angle_ab = real.pi * 2 / 3 -- 120 degrees in radians
axiom a_magnitude_is_2 : magnitude a = 2
axiom b_magnitude_is_2 : magnitude b = 2

theorem magnitude_difference :
  magnitude (a - 3 • b) = 2 * real.sqrt 13 :=
by
  sorry

end magnitude_difference_l603_603585


namespace largest_number_with_digits_sum_19_l603_603835

def digits_sum_to_n (n : Nat) (d : List Nat) : Prop :=
  d.sum = n ∧ (d.nodup)

theorem largest_number_with_digits_sum_19 : ∃ d : List Nat, digits_sum_to_n 19 d ∧ 
  (∀ (d' : List Nat), digits_sum_to_n 19 d' → nat_of_digits d ≥ nat_of_digits d') ∧ 
  nat_of_digits d = 982 :=
begin
  sorry
end

end largest_number_with_digits_sum_19_l603_603835


namespace price_per_large_bottle_l603_603313

theorem price_per_large_bottle : 
  ∃ P : ℝ, 
    (P ≈ 1.89) ∧ 
    (1325 * P + 750 * 1.38) / (1325 + 750) = 1.7057 
:= 
by sorry

end price_per_large_bottle_l603_603313


namespace constant_remainder_polynomial_division_l603_603943

theorem constant_remainder_polynomial_division (b : ℚ) :
  (∃ (r : ℚ), ∀ x : ℚ, r = (8 * x^3 - 9 * x^2 + b * x + 10) % (3 * x^2 - 2 * x + 5)) ↔ b = 118 / 9 :=
by
  sorry

end constant_remainder_polynomial_division_l603_603943


namespace not_profitable_for_large_output_daily_profit_function_maximum_profit_at_84_l603_603068

def defective_rate (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 94 then 1 / (96 - x)
  else if x > 94 then 2 / 3
  else 0

def daily_profit (x : ℕ) (A : ℚ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 94 then
    let p := defective_rate x in
    (x - (3 * x / (2 * (96 - x)))) * A
  else 0

theorem not_profitable_for_large_output (x : ℕ) (A : ℚ) (hx : x > 94) :
  daily_profit x A = 0 :=
by sorry

theorem daily_profit_function (x : ℕ) (A : ℚ) (hx : 1 ≤ x ∧ x ≤ 94) :
  daily_profit x A = (x - (3 * x / (2 * (96 - x)))) * A :=
by sorry

theorem maximum_profit_at_84 (x : ℕ) (A : ℚ) :
  ∃ (T_max : ℚ), T_max = daily_profit 84 A ∧ ∀ y : ℕ, 1 ≤ y ∧ y ≤ 94 → daily_profit y A ≤ T_max :=
by sorry

end not_profitable_for_large_output_daily_profit_function_maximum_profit_at_84_l603_603068


namespace values_of_abc_l603_603548

noncomputable def polynomial_divisibility (a b c : ℤ) : Prop :=
  let f := λ x:ℤ, x^4 + a * x^2 + b * x + c
  in (∀ x:ℤ, f (x-1) = (x-1)^3 * (x * (x + 1) + (a + b + 1) - 1) + (a + b + c + 1) - 1)

theorem values_of_abc {a b c : ℤ} :
  polynomial_divisibility a b c ->
  a = -6 ∧ b = 8 ∧ c = -3 :=
sorry

end values_of_abc_l603_603548


namespace sequence_problem_l603_603972

noncomputable def a (n : ℕ) : ℕ := 
if n = 1 then 1 else 2^(2 * (n - 2)) * 3

def S (n : ℕ) : ℕ :=
if n = 0 then 0 else 4^(n-1)

def b (n : ℕ) := Int.log2 (a (n + 1) / 6)

theorem sequence_problem (n : ℕ) (h : n ≥ 1) : b n = 2 * n - 3 := 
by 
  sorry

end sequence_problem_l603_603972


namespace probability_red_or_blue_l603_603409

noncomputable def total_marbles : ℕ := 100

noncomputable def probability_white : ℚ := 1 / 4

noncomputable def probability_green : ℚ := 1 / 5

theorem probability_red_or_blue :
  (1 - (probability_white + probability_green)) = 11 / 20 :=
by
  -- Proof is omitted
  sorry

end probability_red_or_blue_l603_603409


namespace farmer_bales_left_l603_603472

theorem farmer_bales_left
    (bales_per_month_per_5_acres : ℕ)
    (additional_acres : ℕ)
    (initial_acres : ℕ)
    (num_horses : ℕ)
    (bales_per_horse_per_day : ℕ)
    (days_sep_to_dec : ℕ)
    (total_days : ℕ) :
    bales_per_month_per_5_acres = 560 →
    additional_acres = 7 →
    initial_acres = 5 →
    num_horses = 9 →
    bales_per_horse_per_day = 3 →
    days_sep_to_dec = 30 + 31 + 30 + 31 →
    let total_acres := initial_acres + additional_acres,
        bales_per_acre_per_month := bales_per_month_per_5_acres / initial_acres,
        bales_per_month := bales_per_acre_per_month * total_acres,
        total_bales_harvested := bales_per_month * 4,
        total_consumption := num_horses * bales_per_horse_per_day * days_sep_to_dec in
    total_bales_harvested - total_consumption = 2082 :=
begin
    intros,
    let initial_acres := 5,
    let total_acres := initial_acres + additional_acres,
    let bales_per_acre_per_month := bales_per_month_per_5_acres / initial_acres,
    let bales_per_month := bales_per_acre_per_month * total_acres,
    let total_bales_harvested := bales_per_month * 4,
    let total_consumption := num_horses * bales_per_horse_per_day * days_sep_to_dec,
    sorry, -- Proof skipped as instructed
end

end farmer_bales_left_l603_603472


namespace probability_heads_exactly_2_times_three_tosses_uniform_coin_l603_603413

noncomputable def probability_heads_exactly_2_times (n k : ℕ) (p : ℚ) : ℚ :=
(n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_heads_exactly_2_times_three_tosses_uniform_coin :
  probability_heads_exactly_2_times 3 2 (1/2) = 3 / 8 :=
by
  sorry

end probability_heads_exactly_2_times_three_tosses_uniform_coin_l603_603413


namespace fraction_telephone_numbers_end_with_5_l603_603504

theorem fraction_telephone_numbers_end_with_5 :
  (∃ (total valid_phone_numbers end_5 : ℕ),
    total = 7 * 10^6 ∧
    valid_phone_numbers = 7 * 10^5 ∧
    end_5 = valid_phone_numbers / total ∧
    end_5 = 1 / 10) :=
by {
  use 7 * 10^6,
  use 7 * 10^5,
  split,
  { exact 7 * 10^6, },
  split,
  { exact 7 * 10^5, },
  split,
  { exact (7 * 10^5) / (7 * 10^6), },
  { norm_num, },
  sorry,
}

end fraction_telephone_numbers_end_with_5_l603_603504


namespace part1_part2_l603_603557

noncomputable def f (x a : ℝ) := (4 * x + a) * log x / (3 * x + 1)

theorem part1 (a : ℝ) (h : deriv (λ x, f x a) 1 = 1) : a = 0 :=
by sorry

noncomputable def f_fixed (x : ℝ) := (4 * x * log x) / (3 * x + 1)

theorem part2 (m : ℝ) (h : ∀ x, 1 ≤ x → f_fixed x ≤ m * (x - 1)) : 1 ≤ m :=
by sorry

end part1_part2_l603_603557


namespace circumcenter_BCD_on_circumcircle_ABC_l603_603911

open EuclideanGeometry

variables {P : Type*} [EuclideanSpace P]

-- Variables for points on the circle and tangent lines
variables (A B C D : P)
variables (S : Circle P) (S' : Circle P)

-- Conditions
axiom point_on_circle_B : B ∈ S
axiom point_on_tangent_A : A ∉ S ∧ tangent_to_circle_at S B A
axiom point_not_on_S_C : C ∉ S ∧ (∃ P Q, P ≠ Q ∧ P ∈ AC ∧ Q ∈ AC ∧ P ∈ S ∧ Q ∈ S)
axiom circle_S'_conditions : S'.tangency_points AC.contains C ∧ tangency_between_two_circles S S' D ∧ (B ≠ D ∧ opposite_sides_of_line B D AC)

-- The theorem to be proved
theorem circumcenter_BCD_on_circumcircle_ABC (h1 : B ∈ S) 
                                             (h2 : A ∉ S ∧ tangent_to_circle_at S B A) 
                                             (h3 : C ∉ S ∧ (∃ P Q, P ≠ Q ∧ P ∈ AC ∧ Q ∈ AC ∧ P ∈ S ∧ Q ∈ S)) 
                                             (h4 : tangent_to_circle_at S' AC C ∧ tangency_between_two_circles S S' D ∧ (B ≠ D ∧ opposite_sides_of_line B D AC)) : 
                                             circumcenter (triangle B C D) ∈ circumcircle (triangle A B C) := 
sorry

end circumcenter_BCD_on_circumcircle_ABC_l603_603911


namespace lcm_9_12_15_l603_603000

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l603_603000


namespace y_intercept_range_l603_603199

-- Define the points A and B
def pointA : ℝ × ℝ := (-1, -2)
def pointB : ℝ × ℝ := (2, 3)

-- We define the predicate for the line intersection condition
def line_intersects_segment (c : ℝ) : Prop :=
  let x_val_a := -1
  let y_val_a := -2
  let x_val_b := 2
  let y_val_b := 3
  -- Line equation at point A
  let eqn_a := x_val_a + y_val_a - c
  -- Line equation at point B
  let eqn_b := x_val_b + y_val_b - c
  -- We assert that the line must intersect the segment AB
  eqn_a ≤ 0 ∧ eqn_b ≥ 0 ∨ eqn_a ≥ 0 ∧ eqn_b ≤ 0

-- The main theorem to prove the range of c
theorem y_intercept_range : 
  ∃ c_min c_max : ℝ, c_min = -3 ∧ c_max = 5 ∧
  ∀ c, line_intersects_segment c ↔ c_min ≤ c ∧ c ≤ c_max :=
by
  existsi -3
  existsi 5
  sorry

end y_intercept_range_l603_603199


namespace fibonacci_polynomial_property_l603_603047

def fib : ℕ → ℕ 
| 0     := 1
| 1     := 1
| (n+2) := fib (n+1) + fib n

def p : ℕ → ℕ := sorry

theorem fibonacci_polynomial_property :
  (∀ k, k ∈ { (992 : ℕ), 993, ..., 1982 } → p k = fib k) → p 1983 = fib 1083 - 1 :=
by
  assume h : ∀ k, k ∈ { (992 : ℕ), 993, ..., 1982 } → p k = fib k
  sorry

end fibonacci_polynomial_property_l603_603047


namespace f_odd_f_decreasing_f_inequality_range_l603_603187

noncomputable def f : ℝ → ℝ := sorry -- this definition is noncomputable based on given conditions

axiom f_fun_axiom : ∀ x y : ℝ, f(x) + f(y - x) = f(y)

axiom f_neg_axiom : ∀ x : ℝ, x > 0 → f(x) < 0

theorem f_odd : ∀ x : ℝ, f(-x) = -f(x) :=
sorry

theorem f_decreasing : ∀ x y : ℝ, x < y → f(x) > f(y) :=
sorry

theorem f_inequality_range (t : ℝ) (h_t : t ∈ Icc 1 2) : 
  f(t * x^2 - 2 * x) < f(t + 2) → x > 3 ∨ x < -1 :=
sorry

end f_odd_f_decreasing_f_inequality_range_l603_603187


namespace monotonic_increase_interval_value_of_b_plus_c_l603_603599

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin (x + π / 6) + 2 * sin (x / 2) ^ 2

-- Problem I: Interval of monotonic increase
theorem monotonic_increase_interval (k : ℤ) :
  (∀ x, ∀ y, (2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + 2 * π / 3) → 
  (2 * k * π - π / 3 ≤ y ∧ y ≤ 2 * k * π + 2 * π / 3) → x ≤ y → f x ≤ f y) :=
sorry

-- Define the area function for triangle
def area (a b C : ℝ) : ℝ := (1 / 2) * a * b * sin C

-- Problem II: Value of b + c
theorem value_of_b_plus_c (A B C a b c S : ℝ) 
  (h1 : f A = 3 / 2) 
  (h2 : A = π / 3) 
  (h3 : a = sqrt 3) 
  (h4 : area a b C = sqrt 3 / 2)
  (h5 : 0 < A ∧ A < π) 
  (h6 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos (π / 3)) :
  b + c = 3 :=
sorry

end monotonic_increase_interval_value_of_b_plus_c_l603_603599


namespace sqrt_cubic_sqrt_decimal_l603_603507

theorem sqrt_cubic_sqrt_decimal : 
  (Real.sqrt (0.0036 : ℝ))^(1/3) = 0.3912 :=
sorry

end sqrt_cubic_sqrt_decimal_l603_603507


namespace jasons_monthly_payment_l603_603676

-- Translating the conditions to Lean definitions
def car_cost : ℝ := 32000
def down_payment : ℝ := 8000
def loan_amount : ℝ := car_cost - down_payment
def monthly_payments : ℕ := 48
def annual_interest_rate : ℝ := 0.05
def monthly_interest_rate : ℝ := annual_interest_rate / 12
def monthly_payment_without_interest : ℝ := loan_amount / monthly_payments
def interest_per_month : ℝ := monthly_payment_without_interest * monthly_interest_rate
def total_monthly_payment : ℝ := monthly_payment_without_interest + interest_per_month

-- Proving that the total monthly payment is $502.08
theorem jasons_monthly_payment : total_monthly_payment = 502.08 := by
  sorry

end jasons_monthly_payment_l603_603676


namespace connie_marbles_l603_603453

theorem connie_marbles (a b : ℕ) (h1 : a = 183) (h2 : b = 593) : a + b = 776 :=
by
  rw [h1, h2]
  exact rfl

end connie_marbles_l603_603453


namespace fraction_shaded_area_l603_603473

theorem fraction_shaded_area (l w : ℕ) (h_l : l = 15) (h_w : w = 20)
  (h_qtr : (1 / 4: ℝ) * (l * w) = 75) (h_shaded : (1 / 5: ℝ) * 75 = 15) :
  (15 / (l * w): ℝ) = 1 / 20 :=
by
  sorry

end fraction_shaded_area_l603_603473


namespace gcf_lcm_360_210_l603_603423

theorem gcf_lcm_360_210 :
  let factorization_360 : ℕ × ℕ × ℕ × ℕ := (3, 2, 1, 0) -- Prime exponents for 2, 3, 5, 7
  let factorization_210 : ℕ × ℕ × ℕ × ℕ := (1, 1, 1, 1) -- Prime exponents for 2, 3, 5, 7
  gcd (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 30 ∧
  lcm (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 2520 :=
by {
  let factorization_360 := (3, 2, 1, 0)
  let factorization_210 := (1, 1, 1, 1)
  sorry
}

end gcf_lcm_360_210_l603_603423


namespace smallest_value_of_x_l603_603838

theorem smallest_value_of_x (x : ℝ) (h : 5 * x^2 + 8 * x + 3 = 9) : x = ( -8 - 2 * real.sqrt 46 ) / 10 :=
sorry

end smallest_value_of_x_l603_603838


namespace min_value_of_a_l603_603607

theorem min_value_of_a (a : ℕ) (x : ℤ) (h : x ∈ {x | 5 * x - a ≤ 0}) : 
  5 ∈ { x : ℤ | 5 * x - a ≤ 0 } → a ≥ 25 :=
begin
  sorry
end

end min_value_of_a_l603_603607


namespace find_x_l603_603261

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l603_603261


namespace usual_time_is_1_l603_603038

-- Define the problem conditions
def train_moves_slower (T : ℝ) : Prop :=
  let new_time : ℝ := T + 15 / 60
  in (6 / 7) * new_time = T

theorem usual_time_is_1.5 :
  ∃ T : ℝ, train_moves_slower T ∧ T = 1.5 := sorry

end usual_time_is_1_l603_603038


namespace number_of_rectangles_l603_603148

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 4) : 
  (nat.choose H 2) * (nat.choose V 2) = 60 := by
  rw [hH, hV]
  norm_num

end number_of_rectangles_l603_603148


namespace divide_crowns_l603_603741

theorem divide_crowns (bread1 bread2 : ℕ) (total_crowns : ℚ)
  (h_bread1 : bread1 = 5)
  (h_bread2 : bread2 = 3)
  (h_total_crowns : total_crowns = 2) :
  let total_bread := bread1 + bread2,
      per_person_bread := (total_bread : ℚ) / 3,
      bread_given1 := (bread1 : ℚ) - per_person_bread,
      bread_given2 := (bread2 : ℚ) - per_person_bread,
      total_given := bread_given1 + bread_given2,
      ratio := (bread_given1 / total_given),
      share1 := ratio * total_crowns,
      share2 := (1 - ratio) * total_crowns
  in share1 = 1.75 ∧ share2 = 0.25 :=
by
  sorry

end divide_crowns_l603_603741


namespace ae_length_l603_603302

open EuclideanGeometry

-- Define the setting: Triangle ABC
variables {A B C D E : Point} -- points in the plane
variable {triangle_ABC : Triangle} -- declaring triangle ABC
variable [eq_length_AB_AC : length AB = 3.6]
variable [eq_length_AC : length AC = 3.6]
variable [point_D_on_AB : D ∈ line_segment AB]
variable [dist_AD : length AD = 1.2]

-- Define the condition: Area of triangles ABC and ADE are equal
variable [area_ABC_ADE_eq : area (Triangle.mk A B C) = area (Triangle.mk A E D)]

-- The theorem: We aim to prove length AE = 10.8
theorem ae_length :
  length AE = 10.8 :=
sorry

end ae_length_l603_603302


namespace rectangle_enclosed_by_lines_l603_603168

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l603_603168


namespace unit_prices_purchasing_schemes_maximize_profit_l603_603461

-- Define the conditions and variables
def purchase_price_system (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 240) ∧ (3 * x + 4 * y = 340)

def possible_schemes (a b : ℕ) : Prop :=
  (a + b = 200) ∧ (60 * a + 40 * b ≤ 10440) ∧ (a ≥ 3 * b / 2)

def max_profit (x y : ℝ) (a b : ℕ) : ℝ :=
  (x - 60) * a + (y - 40) * b

-- Prove the unit prices are $60 and $40
theorem unit_prices : ∃ x y, purchase_price_system x y ∧ x = 60 ∧ y = 40 :=
by
  sorry

-- Prove the possible purchasing schemes
theorem purchasing_schemes : ∀ a b, possible_schemes a b → 
  (a = 120 ∧ b = 80 ∨ a = 121 ∧ b = 79 ∨ a = 122 ∧ b = 78) :=
by
  sorry

-- Prove the maximum profit is 3610 with the purchase amounts (122, 78)
theorem maximize_profit :
  ∃ (a b : ℕ), max_profit 80 55 a b = 3610 ∧ purchase_price_system 60 40 ∧ possible_schemes a b ∧ a = 122 ∧ b = 78 :=
by
  sorry

end unit_prices_purchasing_schemes_maximize_profit_l603_603461


namespace k_value_l603_603594

theorem k_value (k : ℝ) :
    (∀ r s : ℝ, (r + s = -k ∧ r * s = 9) ∧ ((r + 3) + (s + 3) = k)) → k = -3 :=
by
    intro h
    sorry

end k_value_l603_603594


namespace find_fraction_l603_603635

theorem find_fraction
  (x : ℝ)
  (h : (x)^35 * (1/4)^18 = 1 / (2 * 10^35)) : x = 1/5 :=
by 
  sorry

end find_fraction_l603_603635


namespace angle_KPM_right_l603_603776

theorem angle_KPM_right :
  ∀ (A B C D K M P: Point) (AC_circle : Circle) (BD_line : Line),
  -- Given conditions
  AC_circle.is_diameter A C →
  AC_circle.is_inscribed (Quadrilateral.mk A B C D) →
  BD_line.is_projection A K → -- K is the projection of A onto BD
  BD_line.is_projection C M → -- M is the projection of C onto BD
  is_parallel (Line.mk K P) (Line.mk B C) → -- Line through K is parallel to BC and intersects AC at P
  is_on_line P (Line.mk A C) →
  -- To prove
  ∠ K P M = 90° := 
  by sorry

end angle_KPM_right_l603_603776


namespace find_a_range_l603_603231

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := 3 * Real.exp x + a

theorem find_a_range (a : ℝ) :
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x > g x a) → a < Real.exp 2 :=
by
  sorry

end find_a_range_l603_603231


namespace min_cos_x_minus_sin_z_l603_603174

noncomputable def min_value (x y z : ℝ) : ℝ :=
  cos x - sin z

theorem min_cos_x_minus_sin_z (x y z : ℝ) (hx : 2 * sin x = tan y) (hy : 2 * cos y = cot z) (hz : sin z = tan x) :
  ∃ a ∈ set.range (λ (x y z : ℝ), min_value x y z)
  (hx : 2 * sin x = tan y) (hy : 2 * cos y = cot z) (hz : sin z = tan x), a = (-5 * Real.sqrt 3 / 6) :=
sorry

end min_cos_x_minus_sin_z_l603_603174


namespace kidsFromOutsideCountyAtCamp_l603_603125

def numKidsFromLawrenceAtCamp : Nat := 34044
def totalKidsAtCamp : Nat := 458988
def numKidsFromOutsideTheCountyAtCamp : Nat := totalKidsAtCamp - numKidsFromLawrenceAtCamp

theorem kidsFromOutsideCountyAtCamp (numKidsFromLawrenceAtCamp totalKidsAtCamp : Nat) :
  numKidsFromLawrenceAtCamp = 34044 → totalKidsAtCamp = 458988 → numKidsFromOutsideTheCountyAtCamp = 424944 :=
by
  intros hLawrence hTotal
  rw [hLawrence, hTotal]
  unfold numKidsFromOutsideTheCountyAtCamp
  simp
  sorry

end kidsFromOutsideCountyAtCamp_l603_603125


namespace figure_M_area_l603_603665

open Real

theorem figure_M_area :
  ∫ x in 1..5, ((4 - x) - (1 / 2 * (x - 2)^2 - 1)) = 4 :=
by
  sorry

end figure_M_area_l603_603665


namespace no_such_triangle_l603_603734

theorem no_such_triangle :
  ¬∃ (α β γ : ℝ), α + β + γ = π ∧
    ((3 * cos α - 2) * (14 * sin α ^ 2 + sin (2 * α) - 12) = 0) ∧
    ((3 * cos β - 2) * (14 * sin β ^ 2 + sin (2 * β) - 12) = 0) ∧
    ((3 * cos γ - 2) * (14 * sin γ ^ 2 + sin (2 * γ) - 12) = 0) :=
by
  sorry

end no_such_triangle_l603_603734


namespace remainder_7_pow_63_mod_8_l603_603429

theorem remainder_7_pow_63_mod_8 : 7^63 % 8 = 7 :=
by sorry

end remainder_7_pow_63_mod_8_l603_603429


namespace angle_bisector_BP_MPN_l603_603894

open Triangle Circle Geometry

variable {A B C M N P : Point} [Geometry]

-- Hypotheses
constant h1 : AcuteAngledTriangle A B C
constant h2 : TangentsToCircumcircleMeetAtPoints M N A C B
constant h3 : AltitudeFromBeesonACMeetsP A B C P

theorem angle_bisector_BP_MPN :
  AngleBisector (Line BP) (Angle MPN) := 
sorry

end angle_bisector_BP_MPN_l603_603894


namespace effective_annual_rate_correct_l603_603045

noncomputable def nominal_annual_interest_rate : ℝ := 0.10
noncomputable def compounding_periods_per_year : ℕ := 2
noncomputable def effective_annual_rate : ℝ := (1 + nominal_annual_interest_rate / compounding_periods_per_year) ^ compounding_periods_per_year - 1

theorem effective_annual_rate_correct :
  effective_annual_rate = 0.1025 :=
by
  sorry

end effective_annual_rate_correct_l603_603045


namespace jill_account_balance_correct_l603_603311

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def jill_end_of_year_two (initial_investment : ℝ) 
                          (first_year_interest_rate : ℝ)
                          (second_year_interest_rate : ℝ) 
                          (first_year_compounding_frequency : ℕ) 
                          (second_year_compounding_frequency : ℕ) 
                          (additional_deposit : ℝ) 
                          (withdrawal : ℝ) 
                          (years : ℕ) : ℝ :=
  let A1 := compound_interest initial_investment first_year_interest_rate first_year_compounding_frequency 1
  let A1' := A1 + additional_deposit - withdrawal
  compound_interest A1' second_year_interest_rate second_year_compounding_frequency 1

theorem jill_account_balance_correct :
  jill_end_of_year_two 10000 0.0396 0.0435 2 4 2000 500 2 ≈ 12429.35 :=
sorry

end jill_account_balance_correct_l603_603311


namespace solve_for_x_l603_603250

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l603_603250


namespace euler_line_parallel_l603_603746

/-- 
  Given the vertices of a triangle \(A(-3,0)\), \(B(3,0)\), and \(C(3,3)\),
  prove that the line defined by \(ax + (a^2 - 3)y - 9 = 0\) is parallel to
  the Euler line of the triangle \(\triangle ABC\) if and only if \(a = -1\).
-/
theorem euler_line_parallel (a : ℝ) :
  let A := (-3, 0)
      B := (3, 0)
      C := (3, 3) 
      centroid := (1, 1)
      circumcenter := (0, 3/2)
      euler_slope := -1 / 2
      given_line := λ a:ℝ, (a, (a^2 - 3))
  in
  given_line a.1 = euler_slope ↔ a = -1 := by 
  sorry

end euler_line_parallel_l603_603746


namespace same_function_ln_exp_l603_603934

theorem same_function_ln_exp (x : ℝ) : (λ x, x) = (λ x, Real.log (Real.exp x)) :=
by
  sorry

end same_function_ln_exp_l603_603934


namespace highest_rectangle_middle_position_is_mode_l603_603775

theorem highest_rectangle_middle_position_is_mode
  (data : List ℚ) 
  (histogram : FrequencyDistributionHistogram data) 
  (highest_rectangle : Rectangle) 
  (H1 : highest_rectangle ∈ histogram.rectangles) 
  (H2 : highest_rectangle.is_highest) 
  (middle_position := (highest_rectangle.base.left + highest_rectangle.base.right) / 2) :
  histogram.characteristic_at middle_position = histogram.mode :=
sorry

end highest_rectangle_middle_position_is_mode_l603_603775


namespace solve_exponential_eq_l603_603779

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_exponential_eq (x : ℝ) : 3^(2 * x) - 3^(x + 1) - 4 = 0 ↔ x = log_base 3 4 :=
by
  sorry

end solve_exponential_eq_l603_603779


namespace boyden_family_tickets_l603_603346

theorem boyden_family_tickets (child_ticket_cost : ℕ) (adult_ticket_cost : ℕ) (total_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  adult_ticket_cost = child_ticket_cost + 6 →
  total_cost = 77 →
  adult_ticket_cost = 19 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost →
  num_adults + num_children = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end boyden_family_tickets_l603_603346


namespace train_length_correct_l603_603892

noncomputable def length_of_train (train_speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * cross_time_s
  total_distance - bridge_length_m

theorem train_length_correct :
  length_of_train 45 30 205 = 170 :=
by
  sorry

end train_length_correct_l603_603892


namespace trajectory_of_N_sum_of_slopes_k_AD_AE_l603_603561

open Real

-- Given conditions and definitions
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 6
def point_on_circle_O (P : ℝ × ℝ) : Prop := circle_O P.1 P.2
def perpendicular_line_PM (P : ℝ × ℝ) : ℝ × ℝ := (P.1, 0)
def vector_PM (P : ℝ × ℝ) : ℝ × ℝ := (0, -P.2)
def point_N (P M N : ℝ × ℝ) : Prop :=
  let NM := (M.1 - N.1, -N.2)
  vector_PM P = (sqrt 2) • NM

-- Equation of the trajectory C
def trajectory_C (x y : ℝ) : Prop := (x^2) / 6 + (y^2) / 3 = 1

-- Question 1: Prove the equation of the trajectory of point N
theorem trajectory_of_N (P N : ℝ × ℝ) (hP : point_on_circle_O P) (hN : point_N P (perpendicular_line_PM P) N) :
  trajectory_C N.1 N.2 :=
sorry

-- Point definitions
def point_A := (2, 1) : ℝ × ℝ
def point_B := (3, 0) : ℝ × ℝ

-- Intersecting points on curve C through B
def line_through_B (k : ℝ) (x : ℝ) : ℝ := k * (x - 3)
def intersect_curve_C (k : ℝ) (C : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  y = line_through_B k x ∧ C x y

-- Computation of slopes k_AD and k_AE and their sum
def slope_k (A D : ℝ × ℝ) : ℝ := (D.2 - A.2) / (D.1 - A.1)
def k_AD_AE_constant (k : ℝ) (D E : ℝ × ℝ) : Prop :=
  slope_k point_A D + slope_k point_A E = -2

-- Question 2: Sum of slopes k_AD + k_AE
theorem sum_of_slopes_k_AD_AE (k : ℝ) (D E : ℝ × ℝ) (hD : intersect_curve_C k trajectory_C D.1 D.2) (hE : intersect_curve_C k trajectory_C E.1 E.2) :
  k_AD_AE_constant k D E :=
sorry

end trajectory_of_N_sum_of_slopes_k_AD_AE_l603_603561


namespace green_hat_cost_l603_603833

theorem green_hat_cost (G : ℕ) (H : 85 = 47 + 38) 
    (C_blue : ∀ (blue_count : ℕ), blue_count * 6 = 282)
    (total_cost : ∀ (green_count : ℕ), 282 + (green_count * G) = 548)
    (green_count : 38)
    (blue_count : 47)
    (total_price : 548):
  G = 7 := by
  -- Here you would proceed with the proof.
  sorry

end green_hat_cost_l603_603833


namespace player_B_prevents_A_winning_l603_603831

-- Definitions based on conditions
structure Game :=
(infinite_grid : ℕ → ℕ → option (Bool))
(player_A_mark : Bool := tt)
(player_B_mark : Bool := ff)

-- Axioms related to the game
axiom A_first_turn {g : Game} : ∀ n m : ℕ, g.infinite_grid n m = none
axiom A_win_condition {g : Game} : ∀ n m : ℕ, (n ≤ m) → (∀ i : ℕ, i < 11 → g.infinite_grid (n + i) m = some g.player_A_mark ∨ g.infinite_grid n (m + i) = some g.player_A_mark ∨ g.infinite_grid (n + i) (m + i) = some g.player_A_mark → false)

-- Theorem statement
theorem player_B_prevents_A_winning (g : Game) : ∀ n m : ℕ, ∃ i j : ℕ, g.infinite_grid i j = some g.player_B_mark ∧ ¬A_win_condition :=
sorry

end player_B_prevents_A_winning_l603_603831


namespace g_fraction_eq_five_halves_l603_603766

theorem g_fraction_eq_five_halves 
  (g : ℝ → ℝ)
  (h : ∀ (c d : ℝ), c^2 * g(d) = d^2 * g(c))
  (hg4 : g(4) ≠ 0) : 
  (g(7) - g(3)) / g(4) = 5 / 2 := 
by 
  sorry

end g_fraction_eq_five_halves_l603_603766


namespace log_fraction_inequality_l603_603267

theorem log_fraction_inequality (x y a b : ℝ) (hx : 0 < x) (hxy : x < y) (hy : y < 1) (hb : 1 < b) (hba : b < a) :
  (ln x / b) < (ln y / a) := 
sorry

end log_fraction_inequality_l603_603267


namespace find_d_find_a_n_inequality_l603_603114

variables {a : ℕ → ℕ} {S : ℕ → ℕ} {b : ℕ → ℕ} {d : ℕ} {n : ℕ}
variable (hn : n > 0)

-- Condition: a_1 = 1 and a_2 = 1
def a_conditions : Prop := a 1 = 1 ∧ a 2 = 1

-- Condition: Sum of the first n terms of a_n is S_n
def S_definition : Prop := ∀ n : ℕ, S n = ∑ i in range (n + 1), a i

-- Condition: b_n = nS_n + (n + 2)a_n
def b_conditions : Prop := ∀ n : ℕ, b n = n * S n + (n + 2) * a n

-- Condition: {b_n} is an arithmetic sequence with common difference d
def b_arithmetic_sequence : Prop := ∀ n, b (n + 1) - b n = d

-- Equivalent proof problem in Lean 4
theorem find_d (h1 : a_conditions) (h2 : S_definition) (h3 : b_conditions) (h4 : b_arithmetic_sequence hn) : d = 4 := sorry

theorem find_a_n (h1 : a_conditions) (h2 : S_definition) (h3 : b_conditions) (h4 : b_arithmetic_sequence hn) : 
  ∀ n : ℕ, n > 0 → a n = n / 2 ^ (n - 1) := sorry

theorem inequality (h1 : a_conditions) (h2 : S_definition) (h3 : b_conditions) (h4 : b_arithmetic_sequence hn) : 
  (∏ i in range (n + 1), a i) * 
  (∏ i in range (n + 1), S i) < 2 ^ (2 * n + 1) / ((n + 1) * (n + 2)) := sorry

end find_d_find_a_n_inequality_l603_603114


namespace area_triangle_BCD_l603_603284

theorem area_triangle_BCD (h : ℝ) : ∃ (h : ℝ), 
  2 * 20 = 4 * h → 
  let area_BCD := 1 / 2 * 30 * h in 
  area_BCD = 150 :=
by 
  assume (h : ℝ) (cond : 2 * 20 = 4 * h),
  use h,
  sorry

end area_triangle_BCD_l603_603284


namespace series_sum_l603_603131

theorem series_sum (n : ℕ) : 
  (\sum k in Finset.range (n + 1), 1 / ((3 * k - 1 : ℤ) * (3 * k + 2 : ℤ))) = n / (2 * (3 * n + 2) : ℤ) := 
by sorry

end series_sum_l603_603131


namespace third_side_tangent_l603_603096

-- Definitions for the conditions
def is_tangent_to_parabola (line_eq parab_eq : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, parab_eq t = 2 * line_eq t

def inscribed_triangle_of_parabola
  {A B C : ℝ × ℝ}
  (parab_eq : ℝ → ℝ) :=
  parab_eq A.snd ^ 2 = 2 * parab_eq A.fst ∧
  parab_eq B.snd ^ 2 = 2 * parab_eq B.fst ∧
  parab_eq C.snd ^ 2 = 2 * parab_eq C.fst

-- Given conditions
variables {A B C : ℝ × ℝ}
variables (parab1 parab2 : ℝ → ℝ) -- parab1: y^2 = 2px, parab2: x^2 = 2qy

-- Prove that if two sides of the inscribed triangle are tangent, then the third side is also tangent
theorem third_side_tangent
  (h1 : inscribed_triangle_of_parabola parab1 A B C)
  (h2 : is_tangent_to_parabola (λ t, (1/(A.snd + B.snd)) * (t - 2 * (B.snd)^2) + 2 * (B.snd)) parab2)
  (h3 : is_tangent_to_parabola (λ t, (1/(B.snd + C.snd)) * (t - 2 * (C.snd)^2) + 2 * (C.snd)) parab2)
  : is_tangent_to_parabola (λ t, (1/(A.snd + C.snd)) * (t - 2 * (C.snd)^2) + 2 * (C.snd)) parab2 :=
sorry

end third_side_tangent_l603_603096


namespace integer_satisfying_values_l603_603797

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l603_603797


namespace reporter_earnings_per_hour_l603_603080

/-- A reporter's expected earnings per hour if she writes the entire time. -/
theorem reporter_earnings_per_hour :
  ∀ (word_earnings: ℝ) (article_earnings: ℝ) (stories: ℕ) (hours: ℝ) (words_per_minute: ℕ),
  word_earnings = 0.1 →
  article_earnings = 60 →
  stories = 3 →
  hours = 4 →
  words_per_minute = 10 →
  (stories * article_earnings + (hours * 60 * words_per_minute) * word_earnings) / hours = 105 :=
by
  intros word_earnings article_earnings stories hours words_per_minute
  intros h_word_earnings h_article_earnings h_stories h_hours h_words_per_minute
  sorry

end reporter_earnings_per_hour_l603_603080


namespace Dmitry_before_father_l603_603851

noncomputable def probability_dmitry_before_father (m : ℝ) (h : 0 < m) : ℝ :=
let x := uniform_open 0 m in
let y := uniform_open 0 m in
let z := uniform_open 0 m in
if h₁ : x < m ∧ y < z ∧ z < m then
  (measure_theory.volume {p : ℝ × ℝ × ℝ | 0 < p.1 ∧ 0 < p.2 ∧ p.2 < p.1 ∧ p.1 < m ∧ 0 < p.3 ∧ p.2 < p.3 ∧ p.3 < m}) /
  (measure_theory.volume {p : ℝ × ℝ × ℝ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 < m ∧ 0 < p.3 ∧ p.2 < p.3 ∧ p.3 < m})
else 0

theorem Dmitry_before_father : ∀ m : ℝ, ∀ h : 0 < m, probability_dmitry_before_father m h = 2 / 3 := 
by intros; sorry

end Dmitry_before_father_l603_603851


namespace pea_patch_part_size_l603_603479

def PeaPatchSize (R : ℕ) : ℕ := 2 * R

def PartOfPeaPatch (P : ℕ) : ℕ := P / 6

theorem pea_patch_part_size
  (R : ℕ) 
  (hR : R = 15)
  (P : ℕ) 
  (hP : P = PeaPatchSize R) :
  PartOfPeaPatch P = 5 :=
by
  rw [hP, hR, PeaPatchSize, PartOfPeaPatch]
  norm_num

end pea_patch_part_size_l603_603479


namespace part1_part2_l603_603182

namespace Problem

open Set

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

-- Part (1)
theorem part1 : A ∩ (B ∩ C) = {3} := by 
  sorry

-- Part (2)
theorem part2 : A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0} := by 
  sorry

end Problem

end part1_part2_l603_603182


namespace no_n_for_equal_sums_l603_603698

def sum_arith_seq (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem no_n_for_equal_sums (n : ℕ) (h : n ≠ 0) :
  let t1 := sum_arith_seq 5 6 n;
      t2 := sum_arith_seq 14 6 n
  in t1 ≠ t2 :=
by
  sorry

end no_n_for_equal_sums_l603_603698


namespace min_cos_x_minus_sin_z_l603_603173

noncomputable def min_value (x y z : ℝ) : ℝ :=
  cos x - sin z

theorem min_cos_x_minus_sin_z (x y z : ℝ) (hx : 2 * sin x = tan y) (hy : 2 * cos y = cot z) (hz : sin z = tan x) :
  ∃ a ∈ set.range (λ (x y z : ℝ), min_value x y z)
  (hx : 2 * sin x = tan y) (hy : 2 * cos y = cot z) (hz : sin z = tan x), a = (-5 * Real.sqrt 3 / 6) :=
sorry

end min_cos_x_minus_sin_z_l603_603173


namespace unit_prices_purchasing_schemes_maximize_profit_l603_603462

-- Define the conditions and variables
def purchase_price_system (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 240) ∧ (3 * x + 4 * y = 340)

def possible_schemes (a b : ℕ) : Prop :=
  (a + b = 200) ∧ (60 * a + 40 * b ≤ 10440) ∧ (a ≥ 3 * b / 2)

def max_profit (x y : ℝ) (a b : ℕ) : ℝ :=
  (x - 60) * a + (y - 40) * b

-- Prove the unit prices are $60 and $40
theorem unit_prices : ∃ x y, purchase_price_system x y ∧ x = 60 ∧ y = 40 :=
by
  sorry

-- Prove the possible purchasing schemes
theorem purchasing_schemes : ∀ a b, possible_schemes a b → 
  (a = 120 ∧ b = 80 ∨ a = 121 ∧ b = 79 ∨ a = 122 ∧ b = 78) :=
by
  sorry

-- Prove the maximum profit is 3610 with the purchase amounts (122, 78)
theorem maximize_profit :
  ∃ (a b : ℕ), max_profit 80 55 a b = 3610 ∧ purchase_price_system 60 40 ∧ possible_schemes a b ∧ a = 122 ∧ b = 78 :=
by
  sorry

end unit_prices_purchasing_schemes_maximize_profit_l603_603462


namespace sum_of_digits_of_binary_300_l603_603016

theorem sum_of_digits_of_binary_300 : 
  ∑ digit in (Nat.digits 2 300), digit = 3 :=
by
  sorry

end sum_of_digits_of_binary_300_l603_603016


namespace value_of_k_l603_603520

open Set
open Real

noncomputable def floorPart (x : ℝ) : ℝ :=
  ⌊x⌋

noncomputable def fracPart (x : ℝ) : ℝ :=
  x - ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  floorPart x * fracPart x

noncomputable def g (x : ℝ) : ℝ :=
  x - 1

def lengthOfSolutionSet (k : ℝ) : ℝ :=
  ∑ d in {d | ∃ a b, a ≤ b ∧ (range (f ⊆ g)) ∈ Ioo 0 k, b - a}

theorem value_of_k :
  ∃ k : ℝ, { x | 0 ≤ x ∧ x ≤ k ∧ f x < g x }.measure = 5 :=
  ∃ k, k = 7


end value_of_k_l603_603520


namespace area_increases_when_sides_increased_l603_603654

-- Definition of Heron's formula
def herons_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem area_increases_when_sides_increased (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  herons_area (a + 1) (b + 1) (c + 1) > herons_area a b c :=
by
  sorry

end area_increases_when_sides_increased_l603_603654


namespace carlos_gummy_worms_l603_603917

theorem carlos_gummy_worms :
  ∀ (initial_gummies : ℕ), initial_gummies = 64 →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 4 → 
    let g := [64, 32, 16, 8, 4] in 
    initial_gummies / 2^n = g.getD (n - 1) 0) →
  initial_gummies / 2^4 = 4 :=
by
  introv h_initial h_pattern
  have h_day1 : initial_gummies / 2 = 32, from h_pattern 1 (by linarith)
  have h_day2 : initial_gummies / 2^2 = 16, from h_pattern 2 (by linarith)
  have h_day3 : initial_gummies / 2^3 = 8, from h_pattern 3 (by linarith)
  have h_day4 : initial_gummies / 2^4 = 4, from h_pattern 4 (by linarith)
  exact h_day4

end carlos_gummy_worms_l603_603917


namespace points_on_hyperbola_l603_603743

theorem points_on_hyperbola (s : ℝ) :
  ∃ u v : ℝ, (u = 2 * Real.cosh(s)) ∧ (v = 2 * Real.sqrt 2 * Real.sinh(s)) ∧
             (u^2 / 4 - v^2 / 8 = 1) :=
by
  sorry

end points_on_hyperbola_l603_603743


namespace probability_divisible_by_15_l603_603269

open Finset

def is_prime_digit (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (∀ d, d ∈ n.digits 10 → is_prime_digit d)

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

theorem probability_divisible_by_15 : 
  let S := {n ∈ (Ico 100 1000) | is_valid_three_digit n}
  let favorable := {n ∈ S | divisible_by_15 n}
  (favorable.card : ℚ) / S.card = 1 / 16 :=
by
  sorry

end probability_divisible_by_15_l603_603269


namespace optimal_path_exists_l603_603768

-- Define the points and connections in a graph representing the tennis court
structure Point :=
  (name : String)

structure Edge :=
  (start : Point)
  (end : Point)

-- Define the tennis court with points and edges
def points : List Point :=
  [⟨"A"⟩, ⟨"B"⟩, ⟨"C"⟩, ⟨"D"⟩, ⟨"E"⟩, ⟨"F"⟩, ⟨"G"⟩, ⟨"H"⟩, ⟨"I"⟩, ⟨"J"⟩]

def edges : List Edge :=
  [⟨⟨"A"⟩, ⟨"B"⟩⟩, ⟨⟨"B"⟩, ⟨"C"⟩⟩, ⟨⟨"C"⟩, ⟨"D"⟩⟩, ⟨⟨"D"⟩, ⟨"E"⟩⟩,
   ⟨⟨"E"⟩, ⟨"F"⟩⟩, ⟨⟨"F"⟩, ⟨"G"⟩⟩, ⟨⟨"G"⟩, ⟨"H"⟩⟩, ⟨⟨"H"⟩, ⟨"I"⟩⟩,
   ⟨⟨"I"⟩, ⟨"J"⟩⟩]

-- Define the main theorem
theorem optimal_path_exists :
  ∃ path : List Point, path = points ∧
  ── (The constructed path traces each edge exactly once) ──
  some condition to ensure minimal repeats and uninterrupted tracing
sorry

end optimal_path_exists_l603_603768


namespace polyhedron_volume_is_eleven_l603_603666

noncomputable def volume_of_polyhedron : ℕ :=
  let side_length := 2
  let cube_volume := side_length ^ 3
  let tetrahedron_volume := (side_length * Math.sqrt 3) ^ 2 / 3
  let total_volume := cube_volume + 3 * tetrahedron_volume
  total_volume

theorem polyhedron_volume_is_eleven :
  volume_of_polyhedron = 11 :=
by 
  sorry

end polyhedron_volume_is_eleven_l603_603666


namespace math_problem_statement_l603_603689

noncomputable def count_odd_difference_quadruples : ℕ :=
  let S := {1, 2, 3, 4}
  let odd (x : ℕ) : Prop := x % 2 = 1
  let even (x : ℕ) : Prop := x % 2 = 0
  let is_odd_difference (a b c d : ℕ) : Prop := odd (a * d - b * c)
  finset.card { (a, b, c, d) ∈ (S × S × S × S) | is_odd_difference a b c d }

theorem math_problem_statement :
  count_odd_difference_quadruples = 96 :=
by
  sorry

end math_problem_statement_l603_603689


namespace sin_13pi_div_3_l603_603942

theorem sin_13pi_div_3 : Real.sin (13 * Real.pi / 3) = sqrt 3 / 2 := by
  -- Conditions
  have h1 : Real.sin 780 = Real.sin (780 - 2 * 360) := by
    rw [Real.sin_periodic]
  have h2 : 780 - 2 * 360 = 60 := by
    norm_num
  -- Use the known value of sin 60 degrees
  have h3 : Real.sin 60 = sqrt 3 / 2 := by
    rw [Real.sin_eq_sin_deg]; norm_num
  sorry

end sin_13pi_div_3_l603_603942


namespace isosceles_triangle_third_side_l603_603571

noncomputable def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

theorem isosceles_triangle_third_side :
  ∀ (a b c : ℕ), is_isosceles a b c ∧ (a = 2 ∨ b = 2 ∨ c = 2) ∧ (a = 5 ∨ b = 5 ∨ c = 5) →
  (a = b → c = 5) ∧ (a = c → b = 5) ∧ (b = c → a = 5) :=
by
  intro a b c
  intro h
  cases h with h_iso h_rest
  cases h_rest with h_2 h_5
  sorry

end isosceles_triangle_third_side_l603_603571


namespace inverse_of_projection_matrix_is_zero_matrix_l603_603693

noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let u := (1 / Real.sqrt (v.1^2 + v.2^2)) * v
  let u_vec := ![u.1, u.2]
  (u_vec ⬝ u_vec.transpose)⟩

theorem inverse_of_projection_matrix_is_zero_matrix :
  let v := (1, 3)
  let proj_mat := projection_matrix v
  proj_mat.det = 0 →
  (∀ P_inv : Matrix (Fin 2) (Fin 2) ℝ, proj_mat ⬝ P_inv = 1 ∧ P_inv ⬝ proj_mat = 1 → false) ∧ ∀ i j, (proj_mat⁻¹)[i, j] = 0 :=
by
  intros
  sorry

end inverse_of_projection_matrix_is_zero_matrix_l603_603693


namespace solve_trig_eq_l603_603365

theorem solve_trig_eq (x : ℝ) (n s : ℤ) :
  sqrt (sqrt (1 - cos (3 * x) ^ 15 * cos (5 * x) ^ 2)) = sin (5 * x) →
  (∃ n : ℤ, x = π / 10 + 2 * n * π / 5) ∨ (∃ s : ℤ, x = 2 * s * π) :=
by
  sorry

end solve_trig_eq_l603_603365


namespace number_in_124th_position_l603_603764

-- Define the sequence conditions
def sequence (a b : ℤ) : ℕ → ℤ
| 0     => a
| 1     => b
| (n+2) => sequence a b (n+1) - sequence a b n

theorem number_in_124th_position (a b : ℤ) :
  sequence a b 123 = -a := by
  sorry

end number_in_124th_position_l603_603764


namespace model_M_time_l603_603865

theorem model_M_time
  (T : ℕ)
  (rate_M : 1 / T)              -- rate of one model M computer
  (rate_8M : 8 / T)             -- rate of eight model M computers
  (rate_N : 1 / 12)             -- rate of one model N computer
  (rate_8N : 8 / 12)            -- rate of eight model N computers
  (combined_rate : 8 / T + 8 / 12 = 1)  -- combined rate when using both types of computers
  : T = 24 := by
  sorry

end model_M_time_l603_603865


namespace alice_tower_heights_l603_603450

theorem alice_tower_heights (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_travel_a : 4 * a * b = 26^2 - 16^2)
  (h_diff : abs (a - b) < 16) (h_sum : a + b < 42) : 
  a = 7 ∨ a = 15 := 
by 
  sorry

end alice_tower_heights_l603_603450


namespace sum_of_digits_base2_of_300_l603_603019

theorem sum_of_digits_base2_of_300 : (nat.binary_digits 300).sum = 4 :=
by
  sorry

end sum_of_digits_base2_of_300_l603_603019


namespace cartesian_eq_C1_cartesian_eq_C2_min_dist_PQ_l603_603663

noncomputable def parametric_eq_C1 (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos α, Real.sin α)

noncomputable def polar_eq_C2 (θ : ℝ) : Prop :=
  ∃ ρ : ℝ, ρ * Real.sin (θ + π / 4) = 2 * sqrt 2

@[simp]
theorem cartesian_eq_C1 (x y : ℝ) : 
  (∃ α : ℝ, (x, y) = parametric_eq_C1 α) ↔ (x^2 / 3 + y^2 = 1) :=
by { sorry }

@[simp]
theorem cartesian_eq_C2 (x y : ℝ) : 
  (∃ θ : ℝ, polar_eq_C2 θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ (x + y = 4) :=
by { sorry }

theorem min_dist_PQ (PQ_min : ℝ) (P : ℝ × ℝ) :
  (P ∈ (λ ⟨x, y⟩, x^2 / 3 + y^2 = 1) ∧ 
   ∃ Q : ℝ × ℝ, Q ∈ (λ ⟨x, y⟩, x + y = 4) ∧ 
   PQ_min = sqrt(2) ∧ 
   P = (3/2, 1/2)) :=
by { sorry }

end cartesian_eq_C1_cartesian_eq_C2_min_dist_PQ_l603_603663


namespace distribution_less_than_m_plus_d_l603_603036

theorem distribution_less_than_m_plus_d
  (m d : ℝ)
  (h1 : ∀ x, f(x + d) = f(m - x))
  (h2 : ∀ x, (f(x) - f(m - d)) + (f(m + d) - f(x)) = 0.36) :
  (∀ x, f(x) < f(m + d)) = 0.68 :=
by
  sorry

end distribution_less_than_m_plus_d_l603_603036


namespace exists_polygon_partition_l603_603979

noncomputable def M_n : ℕ → Type := sorry

theorem exists_polygon_partition (n : ℕ) : 
  ∃ (M : M_n n), 
    (∃ f : M → fin n → fin n × fin n, 
      ∀ i : fin n, ∃ R : ℕ × ℕ, R = (2, 1) ∧ M.partition (f i) = R) :=
sorry

end exists_polygon_partition_l603_603979


namespace proportion_a_value_l603_603637

theorem proportion_a_value (a b c d : ℝ) (h1 : b = 3) (h2 : c = 4) (h3 : d = 6) (h4 : a / b = c / d) : a = 2 :=
by sorry

end proportion_a_value_l603_603637


namespace polynomial_divisibility_l603_603550

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, (x-1)^3 ∣ x^4 + a * x^2 + b * x + c) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
by
  sorry

end polynomial_divisibility_l603_603550


namespace hyperbola_axes_asymptotes_ellipse_equation_intersect_condition_l603_603978

theorem hyperbola_axes_asymptotes :
  let a := sqrt 3
  let b := 1
  let c := 2
  in (2 * a = 2 * sqrt 3) ∧
     (2 * b = 2) ∧
     (∀ x, (y = sqrt 3 * x ∨ y = -sqrt 3 * x) ↔ (y^2 / 3 - x^2 = 0)) := by
  sorry

theorem ellipse_equation :
  (∀ (x y : ℝ), (x^2 + y^2 / 4 = 1) ↔
    ((0, - sqrt 3), (0, sqrt 3), 2) ∧ (sqrt 4 - 3 = 1)) := by sorry

theorem intersect_condition :
  (∀ (m : ℝ), (∃ x y : ℝ, y = x + m ∧ x^2 + y^2 / 4 = 1) ↔
    (-sqrt 5 ≤ m ∧ m ≤ sqrt 5)) := by
  sorry

end hyperbola_axes_asymptotes_ellipse_equation_intersect_condition_l603_603978


namespace triangle_area_l603_603307

theorem triangle_area (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 3) (h₃ : c = 4) : 
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  let sin_B := real.sqrt (1 - cos_B^2)
  let area := (1 / 2) * a * c * sin_B
  area = real.sqrt 135 / 8 :=
by
  sorry

end triangle_area_l603_603307


namespace grasshopper_cannot_return_to_origin_after_2222_jumps_l603_603713

theorem grasshopper_cannot_return_to_origin_after_2222_jumps :
  ¬∃ (x y : ℤ) (n : ℕ), 
    (∀ i ≤ n, (∃ xi yi : ℤ, xi^2 + yi^2 = i^2) ∧ 
              (xi - xi.pred i ≤ 1) ∧ 
              (yi - yi.pred i ≤ 1) ∧
              (x = 0) ∧ (y = 0) ∧
              (i = 2222 → xi = 0 ∧ yi = 0)) := 
begin
  sorry
end

end grasshopper_cannot_return_to_origin_after_2222_jumps_l603_603713


namespace sufficient_but_not_necessary_l603_603966

variable (x : ℝ)

example (h1 : |x - 2| < 1) : x^2 + x - 2 > 0 :=
by 
  sorry

example (h1 : x^2 + x - 2 > 0) : |x - 2| < 1 → False :=
by 
  sorry

theorem sufficient_but_not_necessary (h1 : |x - 2| < 1) : 
  (x^2 + x - 2 > 0) ∧ (∃ y : ℝ, (y^2 + y - 2 > 0 ∧ ¬(|y - 2| < 1))) :=
  ⟨suff_proof, not_nec_proof⟩

where
  suff_proof : x^2 + x - 2 > 0 := 
  by 
    sorry,  -- this shows |x - 2| < 1 implies x^2 + x - 2 > 0 (sufficiency)

  not_nec_proof : ∃ y : ℝ, y^2 + y - 2 > 0 ∧ ¬(|y - 2| < 1) := 
  by 
    sorry,  -- this finds some y s.t. y^2 + y - 2 > 0 but ¬(|y - 2| < 1) (not necessity)

end sufficient_but_not_necessary_l603_603966


namespace expr_B_not_simplified_using_difference_of_squares_l603_603435

def expr_A (x y : ℝ) := (-x - y) * (-x + y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (y + x) * (x - y)
def expr_D (x y : ℝ) := (y - x) * (x + y)

theorem expr_B_not_simplified_using_difference_of_squares (x y : ℝ) :
  ∃ x y, ¬ ∃ a b, expr_B x y = a^2 - b^2 :=
sorry

end expr_B_not_simplified_using_difference_of_squares_l603_603435


namespace new_cost_percentage_l603_603383

theorem new_cost_percentage (t b : ℝ) (C : ℝ) (hC : C = t * b ^ 4) :
  let e := t * (2 * b) ^ 4 in
  e = 16 * C :=
by
  rcases hC with rfl
  sorry

end new_cost_percentage_l603_603383


namespace pseudocode_execution_l603_603530

theorem pseudocode_execution :
  let S_init := 1
  let I_init := 1
  let final_S := nat.iterate (λ s, s + 2) 4 S_init
  let I_after_loop := nat.iterate (λ i, i + 1) 4 I_init
  (final_S = 9) :=
by
  let S_init := 1
  let I_init := 1
  let final_S := nat.iterate (λ s, s + 2) 4 S_init
  let I_after_loop := nat.iterate (λ i, i + 1) 4 I_init
  have h_final_S : final_S = 9 := by norm_num
  exact h_final_S

end pseudocode_execution_l603_603530


namespace vec_perpendicular_l603_603983

noncomputable def e1 : ℝ^2 := ⟨1, 0⟩
noncomputable def e2 : ℝ^2 := ⟨0, 1⟩

noncomputable def a : ℝ^2 := e1 + 2 • e2
noncomputable def b : ℝ^2 := 4 • e1 - 2 • e2

theorem vec_perpendicular :
  a ⬝ b = 0 :=
by
  sorry

end vec_perpendicular_l603_603983


namespace compare_values_l603_603334

variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem compare_values (hf_even : is_even f) (hf_inc : is_increasing_on_nonneg f) :
  f π > f (-3) ∧ f (-3) > f (-√7) :=
by
  sorry

end compare_values_l603_603334


namespace correct_inequalities_count_l603_603694

theorem correct_inequalities_count 
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : 0 < a)
  (cond1 : |a - b| ≤ |a - c| + |b - c|)
  (cond2 : a^2 + 1/a^2 ≥ a + 1/a)
  (cond3 : |a - b| + 1/|a - b| ≥ 2)
  (cond4 : sqrt (a + 3) - sqrt (a + 1) ≤ sqrt (a + 2) - sqrt a) : 
  3 = 1 + 1 + 0 + 1 :=
sorry

end correct_inequalities_count_l603_603694


namespace integer_satisfying_values_l603_603800

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l603_603800


namespace light_path_in_cube_l603_603690

/-- Let ABCD and EFGH be two faces of a cube with AB = 10. A beam of light is emitted 
from vertex A and reflects off face EFGH at point Q, which is 6 units from EH and 4 
units from EF. The length of the light path from A until it reaches another vertex of 
the cube for the first time is expressed in the form s√t, where s and t are integers 
with t having no square factors. Provide s + t. -/
theorem light_path_in_cube :
  let AB := 10
  let s := 10
  let t := 152
  s + t = 162 := by
  sorry

end light_path_in_cube_l603_603690


namespace minimum_combinations_to_open_safe_l603_603392

open Fin

-- Definition of the conditions
def wheels := 3
def positions := 8

def opens_if_two_correct (x y z : Fin 8) : Prop :=
  x = y ∨ y = z ∨ z = x

-- The theorem statement
theorem minimum_combinations_to_open_safe : ∃ (n : ℕ), n = 32 ∧ ∀ (attempts : Fin (positions ^ wheels) → Fin (positions × positions × positions)), 
  (∀ t : Fin (positions ^ wheels), ∃ x y z, opens_if_two_correct (attempts t).1 (attempts t).2.1 (attempts t).2.2) → n = 32 :=
sorry

end minimum_combinations_to_open_safe_l603_603392


namespace decimal_division_l603_603421

theorem decimal_division : (0.05 : ℝ) / (0.005 : ℝ) = 10 := 
by 
  sorry

end decimal_division_l603_603421


namespace expected_value_of_odd_faces_l603_603893

theorem expected_value_of_odd_faces (dice_faces : Fin 12 → ℕ) (h_faces : dice_faces = (λ n, n + 5)) : 
  (∑ i in Finset.filter (λ x, (dice_faces x) % 2 = 1) (Finset.univ : Finset (Fin 12)), dice_faces i) / 6 = 10 :=
by
  sorry

end expected_value_of_odd_faces_l603_603893


namespace number_of_white_balls_l603_603660

-- Definitions based on the problem conditions
def total_balls : Nat := 120
def red_freq : ℝ := 0.15
def black_freq : ℝ := 0.45

-- Result to prove
theorem number_of_white_balls :
  let red_balls := total_balls * red_freq
  let black_balls := total_balls * black_freq
  total_balls - red_balls - black_balls = 48 :=
by
  sorry

end number_of_white_balls_l603_603660


namespace pens_multiple_91_l603_603769

theorem pens_multiple_91 (S : ℕ) (P : ℕ) (total_pencils : ℕ) 
  (h1 : S = 91) (h2 : total_pencils = 910) (h3 : total_pencils % S = 0) :
  ∃ (x : ℕ), P = S * x :=
by 
  sorry

end pens_multiple_91_l603_603769


namespace largest_square_tile_is_one_l603_603710

-- Definition of the problem
def width : ℕ := 19
def height : ℕ := 29
def largest_square_tile (width height : ℕ) : ℕ := Nat.gcd width height

-- Theorem statement
theorem largest_square_tile_is_one : largest_square_tile width height = 1 := by
  sorry

end largest_square_tile_is_one_l603_603710


namespace faster_completion_l603_603057

variables {x y z V : ℝ}
def productivity_condition : Prop :=
  (1 / (y + z) + 1 / (x + z) + 1 / (x + y) = 1)

def combined_time_faster : Prop :=
  let individual_time := (V / (y + z)) + (V / (x + z)) + (V / (x + y)) in
  let combined_time := V / (x + y + z) in
  individual_time / combined_time = 4

theorem faster_completion (h : productivity_condition) : combined_time_faster :=
by
  sorry

end faster_completion_l603_603057


namespace eval_f3_minus_f4_l603_603246

variable {R : Type*}
variables (f : ℝ → ℝ)

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f(x + p) = f(x)
def f_at_1 (f : ℝ → ℝ) : Prop := f(1) = 1
def f_at_2 (f : ℝ → ℝ) : Prop := f(2) = 2

-- Proof problem
theorem eval_f3_minus_f4 
  (hf_odd : is_odd_function f)
  (hf_periodic : is_periodic_function f 5)
  (hf_at_1 : f_at_1 f)
  (hf_at_2 : f_at_2 f)
  : f(3) - f(4) = -1 :=
sorry

end eval_f3_minus_f4_l603_603246


namespace number_of_squares_and_triangles_l603_603908

theorem number_of_squares_and_triangles (ABCD CEFG BEGD: Set Point)
  (is_square_ABCD : is_square ABCD)
  (is_square_CEFG : is_square CEFG)
  (side_lengths_equal : side_length ABCD = side_length CEFG)
  (B_C_G_collinear : collinear {B, C, G}) :
  (count_squares {ABCD, CEFG, BEGD} = 3) ∧ (count_isosceles_right_triangles {ABCD, CEFG, BEGD} = 22) := 
by 
  sorry

end number_of_squares_and_triangles_l603_603908


namespace probability_of_meeting_l603_603418

theorem probability_of_meeting (α x y : ℝ) (hα : α = 10) (hx : 0 ≤ x ∧ x ≤ 60) (hy : 0 ≤ y ∧ y ≤ 60) :
  let p := 1 - ((1 - α / 60) ^ 2) in p = 11 / 36 :=
by
  have h1 : α / 60 = 10 / 60 := by rw hα
  have h2 : 1 - 10 / 60 = 5 / 6 := by norm_num
  have h3 : (5 / 6) ^ 2 = 25 / 36 := by norm_num
  have h4 : 1 - 25 / 36 = 11 / 36 := by norm_num
  rw [h1, h2, h3, h4]
  norm_num

end probability_of_meeting_l603_603418


namespace count_factors_of_48_multiple_of_6_l603_603619

-- We define the number 48
def n : ℕ := 48

-- We define what it means to be a factor of 48 and also a multiple of 6
def is_factor_and_multiple_of_six (k : ℕ) : Prop :=
  (k ∣ n) ∧ (6 ∣ k)

-- We count the number of such factors
def count_factors_and_multiples_of_six : ℕ :=
  (Finset.filter is_factor_and_multiple_of_six (Finset.range (n + 1))).card

-- The theorem stating there are exactly 4 such factors
theorem count_factors_of_48_multiple_of_6 : count_factors_and_multiples_of_six = 4 :=
by
  sorry

end count_factors_of_48_multiple_of_6_l603_603619


namespace sum_of_digits_base_2_of_300_l603_603012

theorem sum_of_digits_base_2_of_300 : 
  let n := 300
  let binary_representation := nat.binary_repr n
  nat.digits_sum 2 binary_representation = 4 :=
by
  let n := 300
  let binary_representation := nat.binary_repr n
  have h1 : binary_representation = [1,0,0,1,0,1,1,0,0] := sorry
  have h2 : nat.digits_sum 2 binary_representation = 1+0+0+1+0+1+1+0+0 := sorry
  show nat.digits_sum 2 binary_representation = 4 from by sorry

end sum_of_digits_base_2_of_300_l603_603012


namespace int_cubed_bound_l603_603196

theorem int_cubed_bound (a : ℤ) (h : 0 < a^3 ∧ a^3 < 9) : a = 1 ∨ a = 2 :=
sorry

end int_cubed_bound_l603_603196


namespace number_of_people_in_group_l603_603380

theorem number_of_people_in_group (weight_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (n : ℝ) (h : new_weight - old_weight = weight_increase * n) :
  n = 8 :=
by
  have diff := new_weight - old_weight
  have avg_increase := weight_increase * n
  have h_diff : diff = 20 := by simp [*]
  have h_avg_increase : avg_increase = 2.5 * n := by simp [*]
  have h0 : 20 = 2.5 * n := by rw [h_diff, h_avg_increase]
  field_simp at h0
  exact eq_of_mul_eq_mul_right (by norm_num) h0

# Test with provided values
example : number_of_people_in_group 2.5 55 75 (8 : ℝ) (by norm_num) := by norm_num

end number_of_people_in_group_l603_603380


namespace count_subsets_no_isolated_correct_l603_603956

noncomputable def count_subsets_no_isolated (n k : ℕ) : ℕ :=
  let r := min (k / 2) (n - k + 1)
  ∑ l in finset.range r.succ, nat.choose (k - l - 1) (l - 1) * nat.choose (n - k + 1) l

theorem count_subsets_no_isolated_correct (n k : ℕ) (hn : 3 ≤ n) (hk : 3 ≤ k) :
  count_subsets_no_isolated n k = ∑ l in finset.range (min (k / 2) (n - k + 1)).succ, 
    nat.choose (k - l - 1) (l - 1) * nat.choose (n - k + 1) l :=
by sorry

end count_subsets_no_isolated_correct_l603_603956


namespace prove_zero_l603_603394

variable {a b c : ℝ}

theorem prove_zero (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
by
  sorry

end prove_zero_l603_603394


namespace kim_total_water_drank_l603_603316

noncomputable def total_water_kim_drank : Float :=
  let water_from_bottle := 1.5 * 32
  let water_from_can := 12
  let shared_bottle := (3 / 5) * 32
  water_from_bottle + water_from_can + shared_bottle

theorem kim_total_water_drank :
  total_water_kim_drank = 79.2 :=
by
  -- Proof skipped
  sorry

end kim_total_water_drank_l603_603316


namespace train_length_l603_603887

theorem train_length (t_platform t_pole : ℕ) (platform_length : ℕ) (train_length : ℕ) :
  t_platform = 39 → t_pole = 18 → platform_length = 350 →
  (train_length + platform_length) / t_platform = train_length / t_pole →
  train_length = 300 :=
by
  intros ht_platform ht_pole hplatform_length hspeeds 
  have h1 : train_length / 18 = (train_length + 350) / 39, from hspeeds
  have h2 : 39 * (train_length / 18) = 39 * ((train_length + 350) / 39), from congrArg (λ x, 39 * x) h1
  sorry

end train_length_l603_603887


namespace ambiguous_dates_count_l603_603834

theorem ambiguous_dates_count : 
  ∃ n : ℕ, n = 132 ∧ ∀ d m : ℕ, 1 ≤ d ∧ d ≤ 31 ∧ 1 ≤ m ∧ m ≤ 12 →
  ((d ≥ 1 ∧ d ≤ 12 ∧ m ≥ 1 ∧ m ≤ 12) → n = 132)
  :=
by 
  let ambiguous_days := 12 * 12
  let non_ambiguous_days := 12
  let total_ambiguous := ambiguous_days - non_ambiguous_days
  use total_ambiguous
  sorry

end ambiguous_dates_count_l603_603834


namespace earnings_per_hour_l603_603076

-- Define the conditions and the respective constants
def words_per_minute : ℕ := 10
def earnings_per_word : ℝ := 0.1
def earnings_per_article : ℝ := 60
def number_of_articles : ℕ := 3
def total_hours : ℕ := 4
def minutes_per_hour : ℕ := 60

theorem earnings_per_hour :
  let total_words := words_per_minute * minutes_per_hour * total_hours in
  let earnings_from_words := earnings_per_word * total_words in
  let earnings_from_articles := earnings_per_article * number_of_articles in
  let total_earnings := earnings_from_words + earnings_from_articles in
  let expected_earnings_per_hour := total_earnings / total_hours in
  expected_earnings_per_hour = 105 := 
  sorry

end earnings_per_hour_l603_603076


namespace H2O_formed_l603_603536

-- Definition of the balanced chemical equation
def balanced_eqn : Prop :=
  ∀ (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ), HCH3CO2 + NaOH = NaCH3CO2 + H2O

-- Statement of the problem
theorem H2O_formed (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ) 
  (h1 : HCH3CO2 = 1)
  (h2 : NaOH = 1)
  (balanced : balanced_eqn):
  H2O = 1 :=
by sorry

end H2O_formed_l603_603536


namespace at_most_one_head_and_exactly_two_heads_mutually_exclusive_l603_603842

noncomputable def possible_outcomes : set (ℕ × ℕ) := { (0, 0), (0, 1), (1, 0), (1, 1) }

-- Event definitions
def at_most_one_head (outcome : ℕ × ℕ) : Prop :=
  outcome.1 + outcome.2 ≤ 1

def exactly_two_heads (outcome : ℕ × ℕ) : Prop :=
  outcome.1 + outcome.2 = 2

-- Definition of mutually exclusive events
def mutually_exclusive (e1 e2 : ℕ × ℕ → Prop) : Prop :=
  ∀ outcome, e1 outcome → ¬e2 outcome

-- The actual theorem to be proved
theorem at_most_one_head_and_exactly_two_heads_mutually_exclusive :
  mutually_exclusive at_most_one_head exactly_two_heads :=
sorry

end at_most_one_head_and_exactly_two_heads_mutually_exclusive_l603_603842


namespace pokemon_card_cost_l603_603962

theorem pokemon_card_cost 
  (football_cost : ℝ)
  (num_football_packs : ℕ) 
  (baseball_cost : ℝ) 
  (total_spent : ℝ) 
  (h_football : football_cost = 2.73)
  (h_num_football_packs : num_football_packs = 2)
  (h_baseball : baseball_cost = 8.95)
  (h_total : total_spent = 18.42) :
  (total_spent - (num_football_packs * football_cost + baseball_cost) = 4.01) :=
by
  -- Proof goes here
  sorry

end pokemon_card_cost_l603_603962


namespace TE_eq_TF_l603_603321

variables {R S T E D F : Type} [EuclideanGeometry R S T E D F]

-- Definitions and conditions in Lean 4
def triangle (R S T : Type) : Prop := ∃ (a b c : R), a ≠ b ∧ b ≠ c ∧ c ≠ a
def angle_bisector (R S T E : Type) : Prop := ∃ (r s t e : R), ∠RSE = ∠SET / 2

-- Given conditions
axiom angle_bisector_RE : angle_bisector R S T E
axiom D_on_RS : D ∈ RS
axiom ED_parallel_RT : parallel ED RT
axiom F_intersection_TD_RE : F ∈ TD ∧ F ∈ RE
axiom SD_eq_RT : distance S D = distance R T

-- Proof goal
theorem TE_eq_TF : distance T E = distance T F :=
by
  -- proof steps would go here, but are omitted as instructed
  exact sorry

end TE_eq_TF_l603_603321


namespace smallest_positive_solution_l603_603948

theorem smallest_positive_solution (x : ℝ) : (∃ x > 0, tan (4 * x) + tan (5 * x) = sec (5 * x)) → (∃ x > 0, x = 2 * pi / 13) :=
by {
  sorry
}

end smallest_positive_solution_l603_603948


namespace polynomial_factorization_l603_603133

noncomputable def factorize_polynomial (a b : ℝ) : ℝ :=
  -3 * a^3 * b + 6 * a^2 * b^2 - 3 * a * b^3

theorem polynomial_factorization (a b : ℝ) : 
  factorize_polynomial a b = -3 * a * b * (a - b)^2 := 
by
  sorry

end polynomial_factorization_l603_603133


namespace multiple_of_4_and_8_l603_603374

theorem multiple_of_4_and_8 (a b : ℤ) (h1 : ∃ k1 : ℤ, a = 4 * k1) (h2 : ∃ k2 : ℤ, b = 8 * k2) :
  (∃ k3 : ℤ, b = 4 * k3) ∧ (∃ k4 : ℤ, a - b = 4 * k4) :=
by
  sorry

end multiple_of_4_and_8_l603_603374


namespace shape_of_phi_eq_d_in_spherical_coordinates_l603_603955

theorem shape_of_phi_eq_d_in_spherical_coordinates (d : ℝ) : 
  (∃ (ρ θ : ℝ), ∀ (φ : ℝ), φ = d) ↔ ( ∃ cone_vertex : ℝ × ℝ × ℝ, ∃ opening_angle : ℝ, cone_vertex = (0, 0, 0) ∧ opening_angle = d) :=
sorry

end shape_of_phi_eq_d_in_spherical_coordinates_l603_603955


namespace solve_log_inequality_l603_603740

theorem solve_log_inequality :
  ∀ x : ℝ, 4 < x → log (1 / 3) (x^2 - 6 * x + 18) - 2 * log (1 / 3) (x - 4) < 0 ↔ 4 < x :=
by
  intro x
  assume h : 4 < x
  sorry

end solve_log_inequality_l603_603740


namespace range_of_a_l603_603578

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1 / (x - 1) < 1
def q (x a : ℝ) : Prop := x^2 + (a - 1) * x - a > 0

-- The main theorem to prove
theorem range_of_a {a : ℝ} : (∀ x : ℝ, p x → q x a) ∧ (¬ ∀ x : ℝ, q x a → p x) ↔ a ∈ set.Ioc (-2 : ℝ) (-1) := sorry

end range_of_a_l603_603578


namespace sum_of_digits_base2_of_300_l603_603021

theorem sum_of_digits_base2_of_300 : (nat.binary_digits 300).sum = 4 :=
by
  sorry

end sum_of_digits_base2_of_300_l603_603021


namespace absolute_inequality_l603_603559

theorem absolute_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := 
sorry

end absolute_inequality_l603_603559


namespace rectangle_enclosed_by_lines_l603_603159

theorem rectangle_enclosed_by_lines :
  ∀ (H : ℕ) (V : ℕ), H = 5 → V = 4 → (nat.choose H 2) * (nat.choose V 2) = 60 :=
by
  intros H V h_eq H_eq
  rw [h_eq, H_eq]
  simp
  sorry

end rectangle_enclosed_by_lines_l603_603159


namespace num_men_in_first_group_l603_603367

variable {x m w : ℝ}

theorem num_men_in_first_group (h1 : x * m + 8 * w = 6 * m + 2 * w)
  (h2 : 2 * m + 3 * w = 0.5 * (x * m + 8 * w)) : 
  x = 3 :=
sorry

end num_men_in_first_group_l603_603367


namespace length_of_train_l603_603882

variables (L : ℝ) (t1 t2 : ℝ) (length_platform : ℝ)

-- Conditions
def condition1 := t1 = 39
def condition2 := t2 = 18
def condition3 := length_platform = 350

-- The goal is to prove the length of the train
theorem length_of_train : condition1 ∧ condition2 ∧ condition3 → L = 300 :=
by
  intros h
  sorry

end length_of_train_l603_603882


namespace tan_sec_solution_l603_603951

noncomputable def smallest_positive_solution (x : ℝ) : ℝ :=
  ∃ n : ℤ, x = (Real.pi / 26) + (2 * Real.pi * n) / 13 ∧ x > 0

theorem tan_sec_solution (x : ℝ) :
  (x = Real.pi / 26) ↔ (smallest_positive_solution x ∧ (Real.tan (4 * x) + Real.tan (5 * x) = Real.sec (5 * x))) :=
by
  sorry

end tan_sec_solution_l603_603951


namespace rectangles_from_lines_l603_603151

theorem rectangles_from_lines : 
  (∃ (h_lines : ℕ) (v_lines : ℕ), (h_lines = 5 ∧ v_lines = 4) ∧ (nat.choose h_lines 2 * nat.choose v_lines 2 = 60)) :=
by
  let h_lines := 5
  let v_lines := 4
  have h_choice := nat.choose h_lines 2
  have v_choice := nat.choose v_lines 2
  have answer := h_choice * v_choice
  exact ⟨h_lines, v_lines, ⟨rfl, rfl⟩, rfl⟩

end rectangles_from_lines_l603_603151


namespace divisor_of_polynomial_l603_603175

-- Definitions
def P (b : ℤ) (q : Polynomial ℤ) : Prop := (Polynomial.x ^ 2 - 2 * Polynomial.x + Polynomial.C b) * q = Polynomial.x ^ 15 + 2 * Polynomial.x + 180

-- Statement to prove
theorem divisor_of_polynomial : ∃ q : Polynomial ℤ, P 4 q :=
by
  sorry

end divisor_of_polynomial_l603_603175


namespace maximize_profit_l603_603874

variables (x : ℝ)
def sales_volume (x : ℝ) := 3 - (2 / (x + 1))
def fixed_investment : ℝ := 8 -- in ten thousand yuan
def additional_investment (m : ℝ) := 16 * m -- in ten thousand yuan
def production_cost (m : ℝ) := fixed_investment + additional_investment m -- in ten thousand yuan
def sales_revenue (m : ℝ) := 1.5 * production_cost m -- in ten thousand yuan
def profit (x : ℝ) : ℝ := 
  let m := sales_volume x 
  in sales_revenue m - (fixed_investment + additional_investment m + x)

theorem maximize_profit : 
  ∀ x ≥ 0, profit x ≤ 21 ∧ (profit x = 21 ↔ x = 3) := 
sorry -- Proof to be filled in

end maximize_profit_l603_603874


namespace largest_six_digit_product_l603_603424

theorem largest_six_digit_product:
  ∃ (n : ℕ), n = 987520 ∧ (∀ d ∈ Int.toDigits n, d ≠ 0 ∨ d ≠ 1 ∨ d ≠ 3 ∨ d ≠ 4 ∨ d ≠ 6 ∨ d ≠ 7 ∨ d ≠ 8) ∧
              (∏ d in Int.toFinset n.digits, d) = 40320 := sorry

end largest_six_digit_product_l603_603424


namespace area_RegionR_l603_603358

-- Definitions related to the rhombus ABCD
def Rhombus (A B C D : ℝ × ℝ) : Prop :=
  (dist A B = 4) ∧ (dist B C = 4) ∧ (dist C D = 4) ∧ (dist D A = 4) ∧
  (angle A B C = 60) ∧ (angle B C D = 60) ∧ (angle C D A = 60) ∧ (angle D A B = 60)

-- Definition of region R being the area closer to vertex B than to any other vertices 
def RegionR (A B C D : ℝ × ℝ) : set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist P B < dist P A ∧ dist P B < dist P C ∧ dist P B < dist P D}

-- Proof problem: the area of region R is 3
theorem area_RegionR {A B C D : ℝ × ℝ} (h : Rhombus A B C D) : 
  let R := RegionR A B C D in 
  area R = 3 :=
sorry  -- proof is omitted

end area_RegionR_l603_603358


namespace coefficient_of_x5_l603_603754

-- Definitions used in Lean for the given condition
def binomial_term (a b : ℂ) (n k : ℕ) : ℂ := (nat.choose n k : ℂ) * a ^ (n - k) * b ^ k

-- Function to get the k-th term of the binomial expansion
def expansion_term (f : ℂ → ℂ) (k : ℕ) := binomial_term (1 / 3 : ℂ) (-3 : ℂ) (7 : ℕ) k * f (7 - 2 * k + 2 * k - 1)

-- Define the problem
theorem coefficient_of_x5 :
  ∑ k in (finset.range 8), 
  expansion_term (λ n, x ^ (14 - 3 * k)) k = (- 35 / 3 : ℂ) * x ^ 5 :=
sorry

end coefficient_of_x5_l603_603754


namespace light_bulb_probability_l603_603708

theorem light_bulb_probability : 
  let prob_change_state (n : ℕ) := 1 / (2 * (n + 1)^2)
  let prob_not_change_state (n : ℕ) := 1 - prob_change_state n
  let A (x : ℝ) := ∏ n in finset.range 100, (prob_not_change_state n + prob_change_state n * x)
  (1 / 2 * (A 1 - A (-1))) = (101 - 2^100) / (2 * 101) :=
by sorry

end light_bulb_probability_l603_603708


namespace num_real_roots_sinx_eq_logx_l603_603761

theorem num_real_roots_sinx_eq_logx : 
  ∀ x : ℝ, 0 < x ∧ x ≤ 10 → (sin x = log x → x = 3) :=
by
  sorry

end num_real_roots_sinx_eq_logx_l603_603761


namespace pond_water_after_20_days_l603_603870

theorem pond_water_after_20_days :
  let initial_volume := 500
  let evaporation_rate := λ (n : ℕ), 0.5 + 0.1 * (n - 1)
  let total_evaporation := ∑ n in Finset.range 20, evaporation_rate (n + 1)
  in initial_volume - total_evaporation = 471 :=
begin
  sorry
end

end pond_water_after_20_days_l603_603870


namespace inequality_system_solution_l603_603397

theorem inequality_system_solution (x : ℝ) (h1 : 5 - 2 * x ≤ 1) (h2 : x - 4 < 0) : 2 ≤ x ∧ x < 4 :=
  sorry

end inequality_system_solution_l603_603397


namespace polynomial_divisibility_l603_603549

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, (x-1)^3 ∣ x^4 + a * x^2 + b * x + c) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
by
  sorry

end polynomial_divisibility_l603_603549


namespace integer_solutions_count_l603_603782

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l603_603782


namespace find_a_from_symmetry_l603_603179

theorem find_a_from_symmetry (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = (a - x) / (x - a - 1)) (h2 : ∃ c, (c = (3, -1)) ∧ ∀ x, f(x) = 2 * c.1 - x -(2 * c.2 - f x)) : a = 2 :=
sorry

end find_a_from_symmetry_l603_603179


namespace density_change_l603_603451

theorem density_change (V : ℝ) (Δa : ℝ) (decrease_percent : ℝ) (initial_volume : V = 27) (edge_increase : Δa = 0.9) : 
    decrease_percent = 8 := 
by 
  sorry

end density_change_l603_603451


namespace integer_satisfying_values_l603_603799

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l603_603799


namespace smallest_n_square_19_and_ends_89_l603_603947

theorem smallest_n_square_19_and_ends_89 : ∃ n : ℕ, (n^2 % 100 = 89) ∧ (n^2 / 10^(nat.log10 (n^2) - 1) = 19) ∧ n = 1383 :=
by
  sorry

end smallest_n_square_19_and_ends_89_l603_603947


namespace quadratic_root_proof_l603_603457

theorem quadratic_root_proof : 
  (∃ x, x = (-5 + Real.sqrt(5^2 + 4 * 3 * 1)) / (2 * 3) ∨ x = (-5 - Real.sqrt(5^2 + 4 * 3 * 1)) / (2 * 3)) →
  ∃ x, (3 * x^2 + 5 * x - 1 = 0) :=
by
  intro h
  cases h with x hx
  exists x
  simp [hx]
  sorry

end quadratic_root_proof_l603_603457


namespace ratio_of_areas_l603_603691

theorem ratio_of_areas (s : ℝ) : 
  let h := s * (Real.sqrt 3 / 2) in
  let area_ABCDE := (1/4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * s^2 in
  let side_PQRST := s + 2 * h / 3 in
  let area_PQRST := (s + 2 * h / 3)^2 * (1/4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) in
  (area_PQRST / area_ABCDE) = (3 + 2 * Real.sqrt 3)^2 / 9 :=
by
  sorry

end ratio_of_areas_l603_603691


namespace inequality_of_products_l603_603396

theorem inequality_of_products
  (a b c d : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_of_products_l603_603396


namespace boys_in_school_l603_603044

theorem boys_in_school (x : ℕ) (boys girls : ℕ) (h1 : boys = 5 * x) 
  (h2 : girls = 13 * x) (h3 : girls - boys = 128) : boys = 80 :=
by
  sorry

end boys_in_school_l603_603044


namespace integral_x_ex2_I_recursion_l603_603855

-- Part 1: Prove the integral of xe^{x^2} from 0 to 1 equals (e - 1) / 2.
theorem integral_x_ex2 : ∫ x in 0..1, x * real.exp (x^2) = (real.exp 1 - real.exp 0) / 2 := sorry

-- Part 2: Express I_{n+1} in terms of I_n given the definition of I_n.
def I (n : ℕ) : ℝ := ∫ x in 0..1, x^(2*n - 1) * real.exp (x^2)

theorem I_recursion (n : ℕ) : I (n + 1) = real.exp 1 / 2 - n * I n := sorry

end integral_x_ex2_I_recursion_l603_603855


namespace tan_alpha_value_l603_603963

theorem tan_alpha_value (α : Real) (h1 : Real.sin (2 * α) = -Real.sin α) (h2 : α ∈ Ioo (Real.pi / 2) Real.pi) : 
  Real.tan α = -Real.sqrt 3 := 
sorry

end tan_alpha_value_l603_603963


namespace num_int_values_x_l603_603789

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l603_603789


namespace find_AB_find_area_AEC_l603_603306

-- Define the triangle and initial conditions
structure Triangle :=
(A B C : Point)

-- Define points D and E
structure PointsOnBC :=
(D E : Point)

-- Define the conditions
variables (T : Triangle) (P : PointsOnBC)
variable (α : ℝ)
variable (BD DE AB : ℝ)

-- Conditions given in the problem
axiom angle_ABC_pi_over_2 : ∠ T.B T.A T.C = π / 2
axiom points_D_E_on_BC : ∃ P, P.D ∈ Segment T.B T.C ∧ P.E ∈ Segment T.B T.C
axiom angles_equal : ∠ (T.A) (T.B) (P.D) = α ∧ ∠ (T.D) (P.E) (T.C) = α
axiom BD_length : BD = 3
axiom DE_length : DE = 5

-- Theorem to prove AB = 6
theorem find_AB (x : ℝ) : AB = 6 :=
by
  have hAB : AB = 6, sorry
  exact hAB

-- Theorem to prove the area of ΔAEC = 75
theorem find_area_AEC (area : ℝ) : area = 75 :=
by
  have hArea : area = 75, sorry
  exact hArea

end find_AB_find_area_AEC_l603_603306


namespace proof_point_on_ellipse_l603_603977

def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_focus (x y : ℝ) : Prop :=
  x = 1 ∧ y = 0

def is_symmetric (x₀ y₀ : ℝ) : Prop :=
  y₀ ≠ 0

def line_eq (m_x₀ m_y₀ n_x₀ n_y₀ : ℝ) (x : ℝ) : ℝ :=
  (m_x₀ - 1) / m_y₀ * x + 1

def line_intersection (m_x₀ m_y₀ n_x₀ n_y₀ : ℝ) (x : ℝ) : Prop :=
  (3 * m_x₀ - 4) / (2 * m_x₀ - 3)

theorem proof_point_on_ellipse 
  (a b : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : a = sqrt 2) (h4 : b = 1)
  (x₀ y₀ : ℝ) (hy : is_symmetric x₀ y₀)
  : ellipse_equation ((3 * x₀ - 4) / (2 * x₀ - 3)) (y₀ / (2 * x₀ - 3)) (sqrt 2) 1 :=
by
  sorry

end proof_point_on_ellipse_l603_603977


namespace sum_of_squares_of_real_solutions_l603_603540

theorem sum_of_squares_of_real_solutions :
  (finset.univ.filter (λ x : ℝ, x^128 = 64^16)).sum (λ x, x^2) = 2^(5/2) :=
sorry

end sum_of_squares_of_real_solutions_l603_603540


namespace altitudes_iff_area_l603_603990

variables {A B C D E F : Type*}
variables (R : ℝ) (S : ℝ) (EF FD DE : ℝ)
variables [acute_triangle A B C] [on_side D A B] [on_side E B C] [on_side F C A]
variables [circumradius A B C R] 

theorem altitudes_iff_area (h : S = R / 2 * (EF + FD + DE)) :
  (altitude AD A B C ∧ altitude BE B C A ∧ altitude CF C A B) ↔ S = R / 2 * (EF + FD + DE) :=
sorry

end altitudes_iff_area_l603_603990


namespace tan_complex_l603_603915

def tan_p7 := Real.tan (Real.pi / 7)
def cos_4p14 := Real.cos (4 * Real.pi / 14)
def sin_4p14 := Real.sin (4 * Real.pi / 14)
def z := (tan_p7 + Complex.i) / (tan_p7 - Complex.i)
def root_unity := cos_4p14 + Complex.i * sin_4p14

theorem tan_complex (h: z = root_unity) : ∃ n : ℕ, n ≥ 0 ∧ n < 14 ∧ z = cos_4p14 + Complex.i * sin_4p14 :=
begin
  -- By definition of z and root_unity, we have equality
  use 2,
  split,
  { norm_num }, -- prove that 2 >= 0
  split,
  { norm_num }, -- prove that 2 < 14
  { exact h }, -- use hypothesis
end

end tan_complex_l603_603915


namespace prime_modulo_quadratic_congruence_l603_603031

theorem prime_modulo_quadratic_congruence (p : ℕ) [Fact p.prime] :
  ((∃ x : ℤ, (x^2 + x + 3 ≡ 0 [ZMOD p])) ↔ (∃ x : ℤ, (x^2 + x + 25 ≡ 0 [ZMOD p]))) ∧
  ((¬ ∃ x : ℤ, (x^2 + x + 3 ≡ 0 [ZMOD p])) ↔ (¬ ∃ x : ℤ, (x^2 + x + 25 ≡ 0 [ZMOD p]))) := sorry

end prime_modulo_quadratic_congruence_l603_603031


namespace solve_for_x_l603_603248

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l603_603248


namespace number_of_negative_numbers_l603_603197

theorem number_of_negative_numbers :
  let a := (-1) ^ 2023
  let b := abs (-2)
  let c := -(-1.2)
  let d := -(3 ^ 2)
  (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) ∧
  (a < 0 → 1) + (b < 0 → 1) + (c < 0 → 1) + (d < 0 → 1) = 2 :=
by
  -- leaving the proof as an exercise
  sorry

end number_of_negative_numbers_l603_603197


namespace arithmetic_sequence_l603_603193

theorem arithmetic_sequence (
  a : ℕ → ℝ, 
  h1 : a 1 + a 2 = 6, 
  h2 : a 2 + a 3 = 10, 
  common_difference : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) : 
  (∀ n : ℕ, a n = 2 * n) ∧ 
  (∀ n : ℕ, (∑ i in finset.range n, (a i + a (i + 1))) = 2 * n^2 + 4 * n) := 
by 
  sorry

end arithmetic_sequence_l603_603193


namespace value_of_x_l603_603252

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l603_603252


namespace integral_evaluation_l603_603529

noncomputable def integral_problem : Prop :=
  ∫ x in 0..Real.pi, (x + Real.cos x) = Real.pi^2 / 2

theorem integral_evaluation : integral_problem :=
by sorry

end integral_evaluation_l603_603529


namespace radius_of_circumscribed_circle_l603_603389

-- Definitions based on conditions
def triangle (α β γ : ℝ) : Prop := 
  (α > 0) ∧ (β > 0) ∧ (γ > 0) ∧ (α + β > γ) ∧ (β + γ > α) ∧ (γ + α > β)

def height (α β γ : ℝ) : ℝ :=
  sorry -- Assume height is defined

def median (α β γ : ℝ) : ℝ :=
  sorry -- Assume median is defined

def equal_angles (α β γ h m : ℝ) : Prop :=
  sorry -- Assume the equality of angles is defined

def distinct (h m : ℝ) : Prop :=
  h ≠ m

def circumscribed_circle_radius (α β γ : ℝ) : ℝ :=
  sorry -- Assume circumscribed circle radius is defined

-- Statement of the problem
theorem radius_of_circumscribed_circle (α β γ h m : ℝ) 
  (ht : triangle α β γ) 
  (hh : height α β γ = h)
  (hm : median α β γ = m)
  (he : equal_angles α β γ h m)
  (hd : distinct h m) :
  circumscribed_circle_radius α β γ = m :=
sorry

end radius_of_circumscribed_circle_l603_603389


namespace binom_15_3_eq_455_l603_603914

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement: Prove that binom 15 3 = 455
theorem binom_15_3_eq_455 : binom 15 3 = 455 := sorry

end binom_15_3_eq_455_l603_603914


namespace sin_7pi_over_6_l603_603050

theorem sin_7pi_over_6 : sin (7 * π / 6) = -1 / 2 := by
  -- conditions as definitions:
  have h1 : 7 * π / 6 = π + π / 6 := by sorry
  have h2 : sin (π + π / 6) = -sin (π / 6) := by rw sin_add_pi
  have h3 : sin (π / 6) = 1 / 2 := by sorry
  -- question == answer:
  rw [h1, h2, h3]
  norm_num

end sin_7pi_over_6_l603_603050


namespace odd_func_value_l603_603640

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x - 3 else 0 -- f(x) is initially set to 0 when x ≤ 0, since we will not use this part directly.

theorem odd_func_value (x : ℝ) (h : x < 0) (hf : isOddFunction f) (hfx : ∀ x > 0, f x = 2 * x - 3) :
  f x = 2 * x + 3 :=
by
  sorry

end odd_func_value_l603_603640


namespace solve_expression_l603_603245

theorem solve_expression (x : ℝ) (h : 3 * x - 5 = 10 * x + 9) : 4 * (x + 7) = 20 :=
by
  sorry

end solve_expression_l603_603245


namespace sum_series_l603_603122

theorem sum_series : 
  (5005 + ∑ i in Finset.range 5000, (5005 - i) / (2^i : ℝ)) = 5009 :=
begin
  sorry
end

end sum_series_l603_603122


namespace perimeter_of_parallelogram_ADEF_l603_603644

def triangle_ABC (A B C D E F : Type*) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty E] [Nonempty F] :=
  ∃ (AB AC BC : ℕ) (BD DE EF FC AF AD : ℕ),
  AB = 24 ∧ AC = 24 ∧ BC = 20 ∧
  (DE ∥ AC) ∧ (EF ∥ AB) ∧
  (BD = DE) ∧ (EF = FC) ∧
  (AD + BD + EF + AF) = (AB + AC)

theorem perimeter_of_parallelogram_ADEF {A B C D E F : Type*} [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty E] [Nonempty F]
  (h : triangle_ABC A B C D E F) : (AD + BD + EF + AF) = 48 :=
  sorry

end perimeter_of_parallelogram_ADEF_l603_603644


namespace max_sum_of_factors_l603_603287

theorem max_sum_of_factors (A B C : ℕ) (h1 : A * B * C = 2310) (h2 : A ≠ B) (h3 : B ≠ C) (h4 : A ≠ C) (h5 : 0 < A) (h6 : 0 < B) (h7 : 0 < C) : 
  A + B + C ≤ 42 := 
sorry

end max_sum_of_factors_l603_603287


namespace value_of_x_l603_603251

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l603_603251


namespace smallest_possible_product_l603_603352

def digits : Set ℕ := {2, 4, 5, 8}

def is_valid_pair (a b : ℤ) : Prop :=
  let (d1, d2, d3, d4) := (a / 10, a % 10, b / 10, b % 10)
  {d1.toNat, d2.toNat, d3.toNat, d4.toNat} ⊆ digits ∧ {d1.toNat, d2.toNat, d3.toNat, d4.toNat} = digits

def smallest_product : ℤ :=
  1200

theorem smallest_possible_product :
  ∀ (a b : ℤ), is_valid_pair a b → a * b ≥ smallest_product :=
by
  intro a b h
  sorry

end smallest_possible_product_l603_603352


namespace toys_profit_l603_603873

theorem toys_profit (sp cp : ℕ) (x : ℕ) (h1 : sp = 25200) (h2 : cp = 1200) (h3 : 18 * cp + x * cp = sp) :
  x = 3 :=
by
  sorry

end toys_profit_l603_603873


namespace john_remaining_money_l603_603679

theorem john_remaining_money (q : ℝ) : 
  let drink_cost := 5 * q
  let medium_pizza_cost := 3 * 2 * q
  let large_pizza_cost := 2 * 3 * q
  let dessert_cost := 4 * (1 / 2) * q
  let total_cost := drink_cost + medium_pizza_cost + large_pizza_cost + dessert_cost
  let initial_money := 60
  initial_money - total_cost = 60 - 19 * q :=
by
  sorry

end john_remaining_money_l603_603679


namespace rectangle_enclosed_by_lines_l603_603169

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l603_603169


namespace find_m_value_prove_inequality_l603_603224

noncomputable def f (x m : ℝ) := |x - m| - |x + 2 * m|

-- The problem conditions
variables (a b m : ℝ)
variable hyp1 : f x m ≤ 3
variable hyp2 : m > 0
variable hyp3 : a > 0
variable hyp4 : b > 0
variable hyp5 : a^2 + b^2 = m^2

-- Statements that need to be proven:
theorem find_m_value (hm : ∀ x : ℝ, f x m ≤ 3) (hm_pos : m > 0) : m = 1 := sorry

theorem prove_inequality (hab : a > 0) (hb : b > 0) (hab2 : a^2 + b^2 = 1) : 
  a^3 / b + b^3 / a ≥ 1 := sorry

end find_m_value_prove_inequality_l603_603224


namespace sum_of_digits_base_2_of_300_l603_603014

theorem sum_of_digits_base_2_of_300 : 
  let n := 300
  let binary_representation := nat.binary_repr n
  nat.digits_sum 2 binary_representation = 4 :=
by
  let n := 300
  let binary_representation := nat.binary_repr n
  have h1 : binary_representation = [1,0,0,1,0,1,1,0,0] := sorry
  have h2 : nat.digits_sum 2 binary_representation = 1+0+0+1+0+1+1+0+0 := sorry
  show nat.digits_sum 2 binary_representation = 4 from by sorry

end sum_of_digits_base_2_of_300_l603_603014


namespace maximum_different_products_24_l603_603113

-- We state the conditions of the problem
def condition_360 (marble_labels : Fin 13 → ℕ) : Prop :=
  ∏ i, marble_labels i = 360

-- We state the goal we want to achieve
def max_different_products (marble_labels : Fin 13 → ℕ) : ℕ :=
  Set.card {p | ∃ (s : Finset (Fin 13)), s.card = 5 ∧ p = (∏ x in s, marble_labels x)}

-- We state the final theorem
theorem maximum_different_products_24 :
  ∀ (marble_labels : Fin 13 → ℕ),
    condition_360 marble_labels →
    max_different_products marble_labels = 24 :=
begin
  intros,
  sorry
end

end maximum_different_products_24_l603_603113


namespace purchase_price_calculation_l603_603087

theorem purchase_price_calculation
    (down_payment : ℝ)
    (monthly_payment : ℝ)
    (num_months : ℕ)
    (interest_rate : ℝ)
    (total_paid : ℝ)
    (purchase_price : ℝ) :
    down_payment = 18 →
    monthly_payment = 10 →
    num_months = 12 →
    interest_rate = 0.15254237288135593 →
    total_paid = down_payment + (monthly_payment * num_months) →
    total_paid = purchase_price * (1 + interest_rate) →
    purchase_price ≈ 119.83 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end purchase_price_calculation_l603_603087


namespace sum_of_coefficients_of_expansion_l603_603121

theorem sum_of_coefficients_of_expansion (x y : ℝ) :
  (3*x - 4*y) ^ 20 = 1 :=
by 
  sorry

end sum_of_coefficients_of_expansion_l603_603121


namespace nested_sqrt_inequality_l603_603357

theorem nested_sqrt_inequality (n : ℕ) (h : n ≥ 2) : 
  (sqrt ((2 : ℝ) * sqrt ((3 : ℝ) * sqrt (∏ i in finset.range (n-2), sqrt ((i+1 : ℕ) + 1))))) < 3 := 
by
  sorry

end nested_sqrt_inequality_l603_603357


namespace probability_three_friends_same_lunch_group_l603_603317

noncomputable def probability_three_friends_same_group : ℝ :=
  let groups := 4
  let probability := (1 / groups) * (1 / groups)
  probability

theorem probability_three_friends_same_lunch_group :
  probability_three_friends_same_group = 1 / 16 :=
by
  unfold probability_three_friends_same_group
  sorry

end probability_three_friends_same_lunch_group_l603_603317


namespace base7_minus_base5_eq_l603_603913

theorem base7_minus_base5_eq :
  let x := 3 * 7^2 + 2 * 7^1 + 5 * 7^0,
      y := 1 * 5^2 + 6 * 5^1 + 4 * 5^0
  in x - y = 107 :=
by
  let x := 3 * 7^2 + 2 * 7^1 + 5 * 7^0
  let y := 1 * 5^2 + 6 * 5^1 + 4 * 5^0
  have h1 : x = 166 := by sorry
  have h2 : y = 59 := by sorry
  show x - y = 107
  sorry

end base7_minus_base5_eq_l603_603913


namespace sum_of_real_roots_eq_zero_l603_603817

noncomputable def polynomial_p : Polynomial ℝ := Polynomial.Coeff 0 (-2) 
  + Polynomial.Coeff 1 (-7) 
  + Polynomial.Coeff 2 (-7) 
  + Polynomial.Coeff 3 (-4) 
  + Polynomial.Coeff 4 1 
  + Polynomial.Coeff 5 1

theorem sum_of_real_roots_eq_zero : 
  ∑ r in (polynomial_p.roots.to_finset.filter Polynomial.is_real_root), r = 0 := 
sorry

end sum_of_real_roots_eq_zero_l603_603817


namespace correct_slope_abs_l603_603822

noncomputable def eq_slope_abs_val : Prop :=
  ∃ (m : ℝ),
    (abs m = 0.5) ∧ 
    (∀ (radius : ℝ) 
       (center1 center2 center3 : ℝ × ℝ) 
       (line_pt : ℝ × ℝ),
       radius = 4 ∧ 
       center1 = (10, 100) ∧
       center2 = (13, 82) ∧
       center3 = (15, 90) ∧
       line_pt = (13, 82) →
          (by
            let translated_center1 := (0, 18)
            let translated_center2 := (0, 0)
            let translated_center3 := (2, 8)
            let dist_center1 := (m * 0 - 18) / sqrt (m^2 + 1)
            let dist_center2 := 0
            let dist_center3 := (2 * m + 8) / sqrt (m^2 + 1 )
            dist_center1 = dist_center3 ∨ dist_center1 = -dist_center3))

theorem correct_slope_abs : eq_slope_abs_val :=
begin
  -- This is where the proof would be constructed.
  -- Proof outline:
  -- 1. Calculate the distances based on transformed coordinates.
  -- 2. Constrain distances to be equal.
  -- 3. Prove absolute value of the slope must be 0.5.
  sorry
end

end correct_slope_abs_l603_603822


namespace f_inv_128_l603_603632

noncomputable def f : ℕ → ℕ := sorry -- Placeholder for the function definition.

axiom f_5 : f 5 = 2           -- Condition 1: f(5) = 2
axiom f_2x : ∀ x, f (2 * x) = 2 * f x  -- Condition 2: f(2x) = 2f(x) for all x

theorem f_inv_128 : f⁻¹ 128 = 320 := sorry -- Prove that f⁻¹(128) = 320 given the conditions

end f_inv_128_l603_603632


namespace length_ZR_eq_43_l603_603305

-- Definitions of points and lengths
structure Triangle :=
  (X Y Z : Point)
  (XY : ℝ)
  (YZ : ℝ)
  (ZX : ℝ)

def length_XY (t : Triangle) : ℝ := t.XY
def length_YZ (t : Triangle) : ℝ := t.YZ
def length_ZX (t : Triangle) : ℝ := t.ZX

-- Defining conditions
variables (t : Triangle)
variables (W V R : Point)
variables (h_XY : t.XY = 13)
variables (h_YZ : t.YZ = 30)
variables (h_ZX : t.ZX = 26)

-- Definitions involved with angle bisectors, circumcircles, and intersection points
def angle_bisector_cond (X Y Z W : Point) : Prop := true
def circumcircle_cond (Y W V R : Point) : Prop := true

-- Condition ensemble for the problem
axiom cond_angle_bisector : angle_bisector_cond t.X t.Y t.Z W
axiom cond_circumcircle : circumcircle_cond t.Y W V R

-- Main statement to prove
theorem length_ZR_eq_43 
  (h_one : angle_bisector_cond t.X t.Y t.Z W)
  (h_two : circumcircle_cond t.Y W V R)
  (h_XY : t.XY = 13)
  (h_YZ : t.YZ = 30)
  (h_ZX : t.ZX = 26) :
  length t.Z R = 43 :=
sorry

end length_ZR_eq_43_l603_603305


namespace cannot_determine_letters_afternoon_l603_603675

theorem cannot_determine_letters_afternoon
  (emails_morning : ℕ) (letters_morning : ℕ)
  (emails_afternoon : ℕ) (letters_afternoon : ℕ)
  (h1 : emails_morning = 10)
  (h2 : letters_morning = 12)
  (h3 : emails_afternoon = 3)
  (h4 : emails_morning = emails_afternoon + 7) :
  ¬∃ (letters_afternoon : ℕ), true := 
sorry

end cannot_determine_letters_afternoon_l603_603675


namespace expected_earnings_per_hour_l603_603083

def earnings_per_hour (words_per_minute earnings_per_word : ℝ) (earnings_per_article num_articles total_hours : ℕ) : ℝ :=
  let minutes_in_hour := 60
  let total_time := total_hours * minutes_in_hour
  let total_words := total_time * words_per_minute
  let word_earnings := total_words * earnings_per_word
  let article_earnings := earnings_per_article * num_articles
  (word_earnings + article_earnings) / total_hours

theorem expected_earnings_per_hour :
  earnings_per_hour 10 0.1 60 3 4 = 105 := by
  sorry

end expected_earnings_per_hour_l603_603083


namespace perp_lines_intersection_point_parallel_lines_m_distance_l603_603239

noncomputable def line1 (x y : ℝ) : Prop := 2 * x + y - 2 = 0
noncomputable def line2 (x y m : ℝ) : Prop := 2 * x - m * y + 4 = 0

theorem perp_lines_intersection_point :
  let l₁ := line1
  let l₂ := (λ x y, line2 x y 4)
  (∀ x y : ℝ, l₁ x y → l₂ x y → x = 0.4 ∧ y = 1.2) :=
by
  sorry

theorem parallel_lines_m_distance :
  let l₁ := line1
  let m := -1
  let l₂ := (λ x y, line2 x y m)
  m = -1 ∧ ∀ d : ℝ,
    (∀ x y : ℝ, l₁ x y → l₂ x y → false) →
    d = 6 * Real.sqrt 5 / 5 :=
by
  sorry

end perp_lines_intersection_point_parallel_lines_m_distance_l603_603239


namespace circumference_irrational_if_radius_rational_l603_603275

theorem circumference_irrational_if_radius_rational
  (a b : Int)
  (h_b : b ≠ 0)
  (h_rational : ∃ a b : Int, b ≠ 0 ∧ r = a / b):
  ¬ ( ∃ c : Rat, c = 2 * π * r):
sorry

end circumference_irrational_if_radius_rational_l603_603275


namespace simplify_fraction_l603_603737

theorem simplify_fraction (n : ℕ) (h : 2 ^ n ≠ 0) : 
  (2 ^ (n + 5) - 3 * 2 ^ n) / (3 * 2 ^ (n + 4)) = 29 / 48 := 
by
  sorry

end simplify_fraction_l603_603737


namespace carrie_is_left_with_50_l603_603509

-- Definitions for the conditions given in the problem
def amount_given : ℕ := 91
def cost_of_sweater : ℕ := 24
def cost_of_tshirt : ℕ := 6
def cost_of_shoes : ℕ := 11

-- Definition of the total amount spent
def total_spent : ℕ := cost_of_sweater + cost_of_tshirt + cost_of_shoes

-- Definition of the amount left
def amount_left : ℕ := amount_given - total_spent

-- The theorem we want to prove
theorem carrie_is_left_with_50 : amount_left = 50 :=
by
  have h1 : amount_given = 91 := rfl
  have h2 : total_spent = 41 := rfl
  have h3 : amount_left = 50 := rfl
  exact rfl

end carrie_is_left_with_50_l603_603509


namespace area_of_yard_l603_603391

def length {w : ℝ} : ℝ := 2 * w + 30

def perimeter {w l : ℝ} (cond_len : l = 2 * w + 30) : Prop := 2 * w + 2 * l = 700

theorem area_of_yard {w l A : ℝ} 
  (cond_len : l = 2 * w + 30) 
  (cond_perim : 2 * w + 2 * l = 700) : 
  A = w * l := 
  sorry

end area_of_yard_l603_603391


namespace part1_part2_l603_603996

-- Defining the complex number z
def z : ℂ := 1 - I

-- Definitions for part 1
def w : ℂ := z^2 + 3 * conj z - 4
def w_trig_form : ℂ := complex.abs w * (complex.cos (3 * real.pi / 4) + I * complex.sin (3 * real.pi / 4))

-- Proof statements for part 1
theorem part1 : w = w_trig_form :=
sorry

-- Definitions for part 2
def a : ℝ := 6
def b : ℝ := 8

-- Definitions for the polynomial equation
def lhs : ℂ := z^2 - a * z + b
def rhs : ℂ := 2 + 4 * I

-- Proof statements for part 2
theorem part2 : lhs = rhs :=
sorry

end part1_part2_l603_603996


namespace max_safe_wise_men_l603_603716

open Set

-- Definitions based on conditions
def num_wise_men : Nat := 100
def num_cars : Nat := 12
def initial_controller_positions := {1, 2}
def max_moves_per_wise_man : Nat := 3

-- A wise man can see an inspector if they're in a neighboring car or two cars away
def can_see (wise_car inspector_car : Nat) : Prop :=
  inspector_car ≠ wise_car ∧ abs (wise_car - inspector_car) ≤ 2

-- The mathematical proof problem
theorem max_safe_wise_men :
  ∃ (max_safe : Nat), max_safe = 82 ∧
    ∀ (wise_men : Fin num_wise_men → Fin num_cars) (controller_pos : Fin 2 → Fin num_cars),
      -- Controllers initially board cars 1 and 2
      (controller_pos 0 = 1 ∧ controller_pos 1 = 2) →
      -- Each wise man can move to an adjacent car up to 3 cars away at each station
      (∀ i, ∃ j, abs (wise_men i - wise_men j) ≤ 3) →
      -- Wise men should be distributed so as many as possible never share a car with controllers
      (∀ i, ¬∃ j, wise_men i = controller_pos j)
:= sorry

end max_safe_wise_men_l603_603716


namespace sum_nine_terms_l603_603194

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- The sequence a_n is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

-- Given conditions
variable ha : is_arithmetic_sequence a
variable hsum : a 2 + a 3 + a 4 + a 5 + a 6 = 150

-- The sum of the first n terms
def S (n : ℕ) : ℝ := (n + 1) / 2 * (a 0 + a n)

theorem sum_nine_terms : S 8 = 270 :=
by
  sorry

end sum_nine_terms_l603_603194


namespace age_of_other_man_l603_603751

theorem age_of_other_man
  (n : ℕ) (average_age_before : ℕ) (average_age_after : ℕ) (age_of_one_man : ℕ) (average_age_women : ℕ) 
  (h1 : n = 9)
  (h2 : average_age_after = average_age_before + 4)
  (h3 : age_of_one_man = 36)
  (h4 : average_age_women = 52) :
  (68 - 36 = 32) := 
by
  sorry

end age_of_other_man_l603_603751


namespace odd_card_draw_even_product_draw_l603_603060

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f(x)

def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := x
def f4 (x : ℝ) : ℝ := Real.cos x
def f5 (x : ℝ) : ℝ := Real.sin x
def f6 (x : ℝ) : ℝ := 2 - x
def f7 (x : ℝ) : ℝ := x + 2

theorem odd_card_draw :
  (C 3 1 * C 4 1) + (C 3 2) = 15 := sorry

theorem even_product_draw :
  (C 2 2) + (C 3 2) + 1 = 5 := sorry

end odd_card_draw_even_product_draw_l603_603060


namespace max_val_z_lt_2_l603_603704

-- Definitions for the variables and constraints
variable {x y m : ℝ}
variable (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1)

-- Theorem statement
theorem max_val_z_lt_2 (h1 : y ≥ x) (h2 : y ≤ m * x) (h3 : x + y ≤ 1) (h4 : m > 1) : 
  (∀ x y, y ≥ x → y ≤ m * x → x + y ≤ 1 → x + m * y < 2) ↔ 1 < m ∧ m < 1 + Real.sqrt 2 :=
sorry

end max_val_z_lt_2_l603_603704


namespace derivative_of_f_at_1_l603_603999

theorem derivative_of_f_at_1 :
  (∃ (f : ℝ → ℝ), (∀ x, deriv f x = (fun x => 2 * (deriv f 1) + 2 * x)) ∧
                   (f = (λ x, 2 * x * (deriv f 1) + x^2))) → deriv (f : ℝ → ℝ) 1 = -2 :=
by 
  sorry

end derivative_of_f_at_1_l603_603999


namespace quadrilateral_inequality_l603_603652

open EuclideanGeometry

/-- 
  Given a quadrilateral ABCD such that ∠A = ∠B and ∠D > ∠C, 
  prove that the length AD is less than the length BC.
-/
theorem quadrilateral_inequality (A B C D : Point) (hABCD : IsQuadrilateral A B C D)
  (hAngle_eq : ∠A = ∠B) (hAngle_ineq : ∠D > ∠C) : dist A D < dist B C :=
by
  sorry

end quadrilateral_inequality_l603_603652


namespace cube_root_sum_zero_implies_opposite_l603_603626

theorem cube_root_sum_zero_implies_opposite (x y : ℝ) (h : (√[3] x) + (√[3] y) = 0) : x = -y :=
sorry

end cube_root_sum_zero_implies_opposite_l603_603626


namespace reporter_earnings_per_hour_l603_603079

/-- A reporter's expected earnings per hour if she writes the entire time. -/
theorem reporter_earnings_per_hour :
  ∀ (word_earnings: ℝ) (article_earnings: ℝ) (stories: ℕ) (hours: ℝ) (words_per_minute: ℕ),
  word_earnings = 0.1 →
  article_earnings = 60 →
  stories = 3 →
  hours = 4 →
  words_per_minute = 10 →
  (stories * article_earnings + (hours * 60 * words_per_minute) * word_earnings) / hours = 105 :=
by
  intros word_earnings article_earnings stories hours words_per_minute
  intros h_word_earnings h_article_earnings h_stories h_hours h_words_per_minute
  sorry

end reporter_earnings_per_hour_l603_603079


namespace solve_for_x_l603_603247

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l603_603247


namespace integer_solutions_count_l603_603786

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l603_603786


namespace icosahedron_colorings_l603_603747

theorem icosahedron_colorings :
  let n := 10
  let f := 9
  n! / 5 = 72576 :=
by
  sorry

end icosahedron_colorings_l603_603747


namespace find_pairs_l603_603136

def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0

theorem find_pairs :
  { (a, b) : ℕ × ℕ | is_solution a b } = {(1, 3), (2, 2), (3, 3)} :=
begin
  sorry
end

end find_pairs_l603_603136


namespace number_of_correct_statements_l603_603901

-- Definitions based on the statements and conditions
def statement1 := ∀ (data : List ℝ) (c : ℝ),
  (data.map (λ x => x + c)).variance = data.variance

def statement2 := ∀ (x : ℝ), 
  let y := 3 - 5*x in
  (y - 3) / (-5) = x - 1

def statement3 := ∀ (ids : List ℕ),
  ids = List.range' 5 50 5 →
  let n := 50 in
  n % 5 = 0

-- Prove that the number of correct statements is 2
theorem number_of_correct_statements : 
  (Cond1 : statement1) → 
  (Cond2 : ¬statement2) → 
  (Cond3 : statement3) → 
  2 = 2 := 
by 
  intros 
  sorry

end number_of_correct_statements_l603_603901


namespace solve_system_of_equations_l603_603739

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y = z) ∧ (x * z = y) ∧ (y * z = x) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) ∨
  (x = 0 ∧ y = 0 ∧ z = 0) := by
  sorry

end solve_system_of_equations_l603_603739


namespace condition_sufficient_not_necessary_l603_603381

variable (a : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)

def fx := λ (x : ℝ), |x - a|
def gx := λ (x : ℝ), |x - 1|

theorem condition_sufficient_not_necessary 
  (h1 : ∀ x : ℝ, 1 ≤ x → gx x ≤ gx (x + 1))
  (h2 : ∀ x : ℝ, 1 ≤ x → fx x ≤ fx (x + 1) → a = 1) :
  "a = 1" is a sufficient but not necessary condition for the function \( f(x) = |x-a| \) to be increasing on the interval \([1, +\infty)\): 
sorry

end condition_sufficient_not_necessary_l603_603381


namespace earnings_per_hour_l603_603077

-- Define the conditions and the respective constants
def words_per_minute : ℕ := 10
def earnings_per_word : ℝ := 0.1
def earnings_per_article : ℝ := 60
def number_of_articles : ℕ := 3
def total_hours : ℕ := 4
def minutes_per_hour : ℕ := 60

theorem earnings_per_hour :
  let total_words := words_per_minute * minutes_per_hour * total_hours in
  let earnings_from_words := earnings_per_word * total_words in
  let earnings_from_articles := earnings_per_article * number_of_articles in
  let total_earnings := earnings_from_words + earnings_from_articles in
  let expected_earnings_per_hour := total_earnings / total_hours in
  expected_earnings_per_hour = 105 := 
  sorry

end earnings_per_hour_l603_603077


namespace find_k_value_l603_603274

theorem find_k_value (k : ℝ) : 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ x4 ≠ 0 ∧
    (x1^2 - 1) * (x1^2 - 4) = k ∧
    (x2^2 - 1) * (x2^2 - 4) = k ∧
    (x3^2 - 1) * (x3^2 - 4) = k ∧
    (x4^2 - 1) * (x4^2 - 4) = k ∧
    x1 ≠ x2 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
    x4 - x3 = x3 - x2 ∧ x2 - x1 = x4 - x3) → 
  k = 7/4 := 
by
  sorry

end find_k_value_l603_603274
