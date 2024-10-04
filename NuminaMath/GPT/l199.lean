import MathLib.Trigonometry.Basic
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.LinearAlgebra.Determinant
import Mathlib.Algebra.Module.Pi
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Special_Functions.Trigonometric
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Fin2
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Divisors
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Logic.Basic
import Mathlib.Statistics
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace problem_solution_l199_199579

-- Define f as a function on real numbers ℝ
variable (f : ℝ → ℝ) 

-- Given conditions
axiom h1 : ∀ x : ℝ, f(x) + f''(x) > 1
axiom h2 : f(0) = 2

-- Prove the solution set of the inequality is { x | x > 0 }
theorem problem_solution : {x : ℝ | exp x * f x > exp x + 1} = {x : ℝ | x > 0} :=
by
  sorry

end problem_solution_l199_199579


namespace charlene_necklaces_l199_199732

theorem charlene_necklaces :
  ∀ (total : ℕ) (sold_percent : ℝ) (given_percent : ℝ),
    total = 360 →
    sold_percent = 0.45 →
    given_percent = 0.25 →
    let remaining_after_sold := total - (total * sold_percent).to_nat in
    let given_away := (remaining_after_sold * given_percent).to_nat in
    let remaining_after_given := remaining_after_sold - given_away in
    remaining_after_given = 149 :=
begin
  intros total sold_percent given_percent,
  intro h_total,
  intro h_sold_percent,
  intro h_given_percent,
  let remaining_after_sold := total - (total * sold_percent).to_nat,
  let given_away := (remaining_after_sold * given_percent).to_nat,
  let remaining_after_given := remaining_after_sold - given_away,
  sorry
end

end charlene_necklaces_l199_199732


namespace inclination_angle_range_l199_199594

theorem inclination_angle_range (θ : ℝ) :
  (∃ α : ℝ, 0 ≤ α ∧ α < π ∧ (x * real.cos θ + real.sqrt 3 * y - 2 = 0) → α ∈ (set.Icc 0 (real.pi / 6) ∪ set.Icc (5 * real.pi / 6) real.pi)) :=
sorry

end inclination_angle_range_l199_199594


namespace other_root_of_equation_l199_199451

theorem other_root_of_equation (c : ℝ) (h : 3^2 - 5 * 3 + c = 0) : 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5 * x + c = 0 ∧ x = 2 := 
by 
  sorry

end other_root_of_equation_l199_199451


namespace find_C_coordinates_l199_199794

-- Given points A and B
def A : ℝ × ℝ × ℝ := (2, 2, 7)
def B : ℝ × ℝ × ℝ := (-2, 4, 3)

-- Define vector subtraction
def vector_sub (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (P.1 - Q.1, P.2 - Q.2, P.3 - Q.3)

-- Define scalar multiplication on vectors
def scalar_mul (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (k * v.1, k * v.2, k * v.3)

-- Define vector equality
def vector_eq (v w : ℝ × ℝ × ℝ) : Prop :=
  v.1 = w.1 ∧ v.2 = w.2 ∧ v.3 = w.3

-- Prove that given the conditions, the coordinates of C are (0, 3, 5)
theorem find_C_coordinates :
  ∃ C : ℝ × ℝ × ℝ, scalar_mul (1/2) (vector_sub B A) = vector_sub C A ∧ C = (0, 3, 5) :=
by
  -- Ensure the proof environment recognizes that the theorem statement is non-trivial
  sorry

end find_C_coordinates_l199_199794


namespace replace_star_with_2x_l199_199548

theorem replace_star_with_2x (c : ℤ) (x : ℤ) :
  (x^3 - 2)^2 + (x^2 + c)^2 = x^6 + x^4 + 4x^2 + 4 ↔ c = 2 * x :=
by
  sorry

end replace_star_with_2x_l199_199548


namespace regular_tetrahedron_volume_l199_199793

-- Definitions from conditions
def alpha (i : ℕ) : ℝ := i
def A_i_on_alpha (i : ℕ) (c : ℝ) : Prop := c = alpha i

-- Proving the volume of regular tetrahedron given conditions
theorem regular_tetrahedron_volume :
  ∃ (T : Type) (A₁ A₂ A₃ A₄ : T) (vol : ℝ),
  (A_i_on_alpha 1 A₁ ∧ A_i_on_alpha 2 A₂ ∧ A_i_on_alpha 3 A₃ ∧ A_i_on_alpha 4 A₄) ∧
  is_regular_tetrahedron A₁ A₂ A₃ A₄ ∧ 
  vol = ((5 * sqrt 5) / 3) := sorry

end regular_tetrahedron_volume_l199_199793


namespace red_black_squares_sum_equal_l199_199499

theorem red_black_squares_sum_equal (n : ℕ) (hn_even : n % 2 = 0) :
  let grid := λ k l, (k - 1) * n + l
  let red_black_sum (coloring: (ℕ × ℕ) → Prop) := 
    ∑ i in finset.range n, ∑ j in finset.range n, if coloring (i, j) then grid (i + 1) (j + 1) else 0
  ∃ (coloring: (ℕ × ℕ) → Prop), 
    (∀ i, ∑ j in finset.range n, if coloring (i, j) then 1 else 0 = n / 2) ∧
    (∀ j, ∑ i in finset.range n, if coloring (i, j) then 1 else 0 = n / 2) ∧
    red_black_sum coloring = red_black_sum (λ p, ¬coloring p) :=
begin
  sorry
end

end red_black_squares_sum_equal_l199_199499


namespace simplify_and_rationalize_denominator_l199_199562

theorem simplify_and_rationalize_denominator :
  (sqrt 3 / sqrt 4) * (sqrt 5 / sqrt 6) * (sqrt 7 / sqrt 8) = sqrt 70 / 8 :=
by {
  have h1 : sqrt 4 = 2 := sorry,
  have h2 : sqrt 6 = sqrt 2 * sqrt 3 := sorry,
  have h3 : sqrt 8 = 2 * sqrt 2 := sorry,
  sorry
}

end simplify_and_rationalize_denominator_l199_199562


namespace robin_pieces_of_candy_l199_199050

theorem robin_pieces_of_candy (packages : ℕ) (pieces_per_package : ℕ) (h1 : packages = 45) (h2 : pieces_per_package = 9) : packages * pieces_per_package = 405 :=
by
  rw [h1, h2]
  norm_num

end robin_pieces_of_candy_l199_199050


namespace bottle_caps_total_l199_199608

theorem bottle_caps_total (caps_per_box : ℝ) (num_boxes : ℝ) (total_caps : ℝ) :
  caps_per_box = 35.0 → num_boxes = 7.0 → total_caps = caps_per_box * num_boxes → total_caps = 245.0 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end bottle_caps_total_l199_199608


namespace balls_into_boxes_l199_199086

theorem balls_into_boxes (balls boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 4) :
  (number_of_ways_to_distribute_balls_with_each_box_having_at_least_one_ball balls boxes) = 240 :=
by
  sorry

end balls_into_boxes_l199_199086


namespace solution_to_g_inv_2_l199_199016

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := 1 / (c * x + d)

theorem solution_to_g_inv_2 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
    ∃ x : ℝ, g x c d = 2 ↔ x = (1 - 2 * d) / (2 * c) :=
by
  sorry

end solution_to_g_inv_2_l199_199016


namespace mini_van_tank_capacity_is_65_l199_199465

noncomputable def mini_van_tank_capacity (V : ℝ) : Prop :=
  let service_cost_per_vehicle := 2.30
      fuel_cost_per_liter := 0.70
      num_mini_vans := 4
      num_trucks := 2
      total_cost := 396
      truck_capacity_ratio := 1.20
  in
  let total_service_cost := (num_mini_vans + num_trucks) * service_cost_per_vehicle
      fuel_budget := total_cost - total_service_cost
      total_fuel_liters := fuel_budget / fuel_cost_per_liter
      total_vehicle_fuel := num_mini_vans * V + num_trucks * (1 + truck_capacity_ratio) * V
  in
    total_vehicle_fuel = total_fuel_liters ∧ V = 65

theorem mini_van_tank_capacity_is_65 : ∃ V, mini_van_tank_capacity V :=
  sorry

end mini_van_tank_capacity_is_65_l199_199465


namespace problem1_problem2_problem3_l199_199818

theorem problem1 (a b : ℝ) (f : ℝ → ℝ)
    (h₀ : f = λ x, b / (x - a))
    (h₁ : f 0 = 3 / 2)
    (h₂ : f 3 = 3) :
    f = λ x, 3 / (x - 2) := sorry

theorem problem2 (f : ℝ → ℝ)
    (h₀ : f = λ x, 3 / (x - 2)) :
    ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂ := sorry

theorem problem3 (f : ℝ → ℝ)
    (m n : ℝ)
    (h₀ : m > 2)
    (h₁ : n > 2)
    (h₂ : f = λ x, 3 / (x - 2))
    (h₃ : ∀ y, y ∈ set.Icc m n ↔ f y ∈ set.Icc 1 3) :
    m + n = 8 := sorry

end problem1_problem2_problem3_l199_199818


namespace total_storage_l199_199488

variable (barrels largeCasks smallCasks : ℕ)
variable (cap_barrel cap_largeCask cap_smallCask : ℕ)

-- Given conditions
axiom h1 : barrels = 4
axiom h2 : largeCasks = 3
axiom h3 : smallCasks = 5
axiom h4 : cap_largeCask = 20
axiom h5 : cap_smallCask = cap_largeCask / 2
axiom h6 : cap_barrel = 2 * cap_largeCask + 3

-- Target statement
theorem total_storage : 4 * cap_barrel + 3 * cap_largeCask + 5 * cap_smallCask = 282 := 
by
  -- Proof is not required
  sorry

end total_storage_l199_199488


namespace a2016_minus_a2014_l199_199813

noncomputable def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
variables (a : ℕ → ℝ) (d : ℝ)
hypothesis h1 : arithmetic_sequence a d
hypothesis h2 : (Sn 2016 a) / 2016 - (Sn 2015 a) / 2015 = 3

-- To prove
theorem a2016_minus_a2014 : a 2016 - a 2014 = 12 :=
sorry

end a2016_minus_a2014_l199_199813


namespace distinct_intersection_points_l199_199353

-- Definitions for the given conditions
def equation1 (x y : ℝ) : Prop := (x + y = 4 ∨ 3 * x - 2 * y = -6)
def equation2 (x y : ℝ) : Prop := (2 * x - y = 3 ∨ x + 2 * y = 11)

-- Problem statement in Lean 4: proving the number of distinct intersection points is 4
theorem distinct_intersection_points : 
  {p : ℝ × ℝ | equation1 p.1 p.2} ∩ {p : ℝ × ℝ | equation2 p.1 p.2} = 4 := sorry

end distinct_intersection_points_l199_199353


namespace angle_Z_130_degrees_l199_199922

theorem angle_Z_130_degrees {p q : Line} {X Y Z : Point}
  (h_parallel : p ∥ q)
  (h_angle_X : m∠X = 100)
  (h_angle_Y : m∠Y = 130) :
  m∠Z = 130 :=
sorry

end angle_Z_130_degrees_l199_199922


namespace player_A_has_no_winning_strategy_l199_199935

theorem player_A_has_no_winning_strategy
  (r : ℕ → ℤ)
  (x : ℕ → ℝ)
  (y : ℕ → ℤ)
  (m : ℕ)
  (A_paint : ℕ → ℝ)
  (B_paint : ℕ → ℝ) :
  (∀ n, 0 ≤ x n ≤ 1) →
  (x 0 = 0) →
  (∀ n m, ∃ k, A_paint n = 1 / 2 ^ m) →
  (∀ n m, ∃ k, B_paint n = ((k : ℝ) / 2 ^ m, ((k + 1) : ℝ) / 2 ^ m)) →
  (∃ N, ∑ i in range N, A_paint i = 4 ∧ (interval (B_paint i).fst (B_paint i).snd) ⊆ [0,1]) →
  (∃ N, x N = 1) :=
begin
  sorry
end

end player_A_has_no_winning_strategy_l199_199935


namespace non_negative_integer_solutions_l199_199590

theorem non_negative_integer_solutions (x : ℕ) : 3 * x - 2 < 7 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end non_negative_integer_solutions_l199_199590


namespace triangle_ACM_area_l199_199886

theorem triangle_ACM_area (AC BC : ℝ) (angle_C : ℝ) (hAC : AC = 6) (hBC : BC = 3 * Real.sqrt 3) (h_angle_C : angle_C = Real.pi / 6) :
  let area_ABC := (1 / 2) * AC * BC * Real.sin angle_C in
  let area_ACM := area_ABC / 2 in
  area_ACM = (9 * Real.sqrt 3) / 2 := 
by
  sorry

end triangle_ACM_area_l199_199886


namespace Jacqueline_hours_per_week_l199_199891

theorem Jacqueline_hours_per_week
  (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_weeks : ℕ) (school_earnings_goal : ℕ) :
  summer_weeks = 8 → summer_hours_per_week = 60 → summer_earnings = 6000 →
  school_weeks = 40 → school_earnings_goal = 7500 →
  (let summer_hours := summer_weeks * summer_hours_per_week in
   let hourly_wage := summer_earnings / summer_hours in
   let total_hours_school := school_earnings_goal / hourly_wage in
   total_hours_school / school_weeks = 15) :=
by
  intros h1 h2 h3 h4 h5
  let summer_hours := 8 * 60
  let hourly_wage := 6000 / summer_hours
  let total_hours_school := 7500 / hourly_wage
  show total_hours_school / 40 = 15, from sorry

end Jacqueline_hours_per_week_l199_199891


namespace closest_approach_l199_199766

noncomputable def vector_v (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 8 * t, -2 + 4 * t, -4 - 2 * t)

def vector_a : ℝ × ℝ × ℝ :=
  (3, 3, 2)

def direction_vector : ℝ × ℝ × ℝ :=
  (8, 4, -2)

def orthogonal (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

theorem closest_approach : orthogonal (vector_v (2/7) - vector_a) direction_vector := by
  -- Proof goes here
  sorry

end closest_approach_l199_199766


namespace arccos_one_half_is_pi_div_three_l199_199200

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199200


namespace incorrect_probability_l199_199401

variable {Ω : Type*} [fintype Ω] (A B : set Ω)
variables [decidable_pred (λ x, x ∈ A)] [decidable_pred (λ x, x ∈ B)]

-- Given conditions
def nOmega : ℕ := 12
def nA : ℕ := 6
def nB : ℕ := 4
def nA_union_B : ℕ := 8

-- Definitions based on conditions
def P (s : set Ω) : ℚ := fintype.card s.to_finset / fintype.card Ω.to_finset
def complement (s : set Ω) : set Ω := { x | x ∉ s }

def P_A := P A
def P_B := P B
def P_A_union_B := P (A ∪ B)
def P_complement_A := P (complement A)
def P_complement_B := P (complement B)

-- Probabilities given in the question
def P_intersection := P (A ∩ B)
def P_union := P (A ∪ B)
def P_complement_intersection := P (complement A ∩ B)
def P_complement_complement := P (complement A ∩ complement B)

-- The problem statement
theorem incorrect_probability :
  (P_A = 6 / 12) ∧
  (P_B = 4 / 12) ∧
  (P_A_union_B = 8 / 12) ∧
  (P_intersection = 1 / 6) ∧
  (P_union = 2 / 3) ∧
  (P_complement_intersection = 1 / 6) ∧
  (P_complement_complement ≠ 2 / 3) :=
by sorry

end incorrect_probability_l199_199401


namespace amelia_wins_probability_l199_199716

/--
Amelia has a coin that lands heads with probability 1/4, 
and Blaine has a coin that lands heads with probability 3/7.
They toss their coins alternately starting with Amelia until one of them gets a head.
Prove that the probability Amelia wins the game is 7/16.
-/
theorem amelia_wins_probability : 
  let pA := 1 / 4
  let pB := 3 / 7
  P_amelia_wins (pA pB) = 7 / 16 :=
proof
  sorry

end amelia_wins_probability_l199_199716


namespace second_number_value_l199_199412

theorem second_number_value
  (a b : ℝ)
  (h1 : a * (a - 6) = 7)
  (h2 : b * (b - 6) = 7)
  (h3 : a ≠ b)
  (h4 : a + b = 6) :
  b = 7 := by
sorry

end second_number_value_l199_199412


namespace b_contribution_is_correct_l199_199119

-- Definitions based on the conditions
def A_investment : ℕ := 35000
def B_join_after_months : ℕ := 5
def profit_ratio_A_B : ℕ := 2
def profit_ratio_B_A : ℕ := 3
def A_total_months : ℕ := 12
def B_total_months : ℕ := 7
def profit_ratio := (profit_ratio_A_B, profit_ratio_B_A)
def total_investment_time_ratio : ℕ := 12 * 35000 / 7

-- The property to be proven
theorem b_contribution_is_correct (X : ℕ) (h : 35000 * 12 / (X * 7) = 2 / 3) : X = 90000 :=
by
  sorry

end b_contribution_is_correct_l199_199119


namespace coefficient_of_x_squared_in_binomial_expansion_l199_199370

noncomputable def binomial_general_term (n k : ℕ) : ℕ :=
  (Nat.choose n k) * (-3)^k

noncomputable def coefficient_x_squared : ℕ :=
  binomial_general_term 5 2

theorem coefficient_of_x_squared_in_binomial_expansion :
  coefficient_x_squared = 90 :=
by
  sorry

end coefficient_of_x_squared_in_binomial_expansion_l199_199370


namespace sufficient_but_not_necessary_cond_l199_199507

variable {Point : Type} [IncidenceGeometry Point]

variables (m n : Line Point) (α β : Plane Point)

-- Assume that m and n are lines in the plane α
axiom lines_in_plane : m ∈ α ∧ n ∈ α

-- Define parallelism
def parallel_planes (α β : Plane Point) : Prop :=
  ∀ (p : Point), p ∈ α → p ∉ β

def parallel_line_to_plane (l : Line Point) (β : Plane Point) : Prop :=
  ∀ (p : Point), p ∈ l → p ∉ β

theorem sufficient_but_not_necessary_cond (m n : Line Point) (α β : Plane Point)
    (h1 : m ∈ α) (h2 : n ∈ α) :
  (parallel_planes α β) ↔
  ((parallel_line_to_plane m β) ∧ (parallel_line_to_plane n β)) :=
sorry

end sufficient_but_not_necessary_cond_l199_199507


namespace expand_expression_l199_199365

theorem expand_expression (x y : ℝ) : 
  (2 * x + 3) * (5 * y + 7) = 10 * x * y + 14 * x + 15 * y + 21 := 
by sorry

end expand_expression_l199_199365


namespace solve_equation1_solve_equation2_l199_199949

-- Define the first equation and its expected solutions:
def equation1 (x : ℝ) : Prop := x^2 - 6 * x - 2 = 0
def solution1_1 : ℝ := 3 + Real.sqrt 11
def solution1_2 : ℝ := 3 - Real.sqrt 11

-- Define the second equation and its expected solutions:
def equation2 (x : ℝ) : Prop := (2 * x + 1)^2 + 6 * x + 3 = 0
def solution2_1 : ℝ := -1 / 2
def solution2_2 : ℝ := -2

theorem solve_equation1 :
  ∀ x : ℝ, (equation1 x ↔ x = solution1_1 ∨ x = solution1_2) := by
  sorry

theorem solve_equation2 :
  ∀ x : ℝ, (equation2 x ↔ x = solution2_1 ∨ x = solution2_2) := by
  sorry

end solve_equation1_solve_equation2_l199_199949


namespace part_1_part_2_l199_199826

noncomputable def f (x b c : ℝ) := x^2 + b * x + c
noncomputable def f' (x b c : ℝ) := 2 * x + b

theorem part_1 (b c : ℝ) (h1 : ∀ x : ℝ, f' x b c ≤ f x b c) (x : ℝ) (hx : 0 ≤ x) :
  f x b c ≤ (x + c)^2 :=
sorry

theorem part_2 (b c M : ℝ) (h1 : ∀ b c : ℝ, f(c) - f(b) ≤ M * (c^2 - b^2)) :
  M ≥ 3 / 2 :=
sorry

end part_1_part_2_l199_199826


namespace sin_theta_of_geometric_cos_progression_l199_199011

theorem sin_theta_of_geometric_cos_progression
  (θ: ℝ)
  (h1: 0 < θ ∧ θ < π / 2)
  (h2: ∃ a b c : ℝ, (a = cos θ ∧ b = cos (2 * θ) ∧ c = cos (3 * θ) ∧
                      ∀ (x y z : ℝ), (x = a ∧ y = b ∧ z = c  ∨ x = a ∧ y = c ∧ z = b
                                      ∨ x = b ∧ y = a ∧ z = c  ∨ x = b ∧ y = c ∧ z = a
                                      ∨ x = c ∧ y = a ∧ z = b  ∨ x = c ∧ y = b ∧ z = a) →
                                      y = x * x ∧ z = x * y)) :
  sin θ = 0 :=
  sorry

end sin_theta_of_geometric_cos_progression_l199_199011


namespace log_ineq_solution_f_max_min_l199_199431

noncomputable def set_B := {x : ℝ | real.sqrt 2 ≤ x ∧ x ≤ 4}

theorem log_ineq_solution (x : ℝ) :
  (2 * (real.log x / real.log 2)^2 - 5 * (real.log x / real.log 2) + 2 ≤ 0) ↔ (real.sqrt 2 ≤ x ∧ x ≤ 4) := 
sorry

theorem f_max_min (x : ℝ) (hx : (real.sqrt 2 ≤ x ∧ x ≤ 4)) :
  let f := (λ x : ℝ, (real.log (x / 8) / real.log 2) * (real.log (2 * x) / real.log 2)) in
  (f x ≤ 5 ∧ f x ≥ -4) :=
sorry

end log_ineq_solution_f_max_min_l199_199431


namespace sum_of_inverses_mod_17_l199_199104

theorem sum_of_inverses_mod_17 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 17 = 15 :=
by
  sorry

end sum_of_inverses_mod_17_l199_199104


namespace arccos_one_half_is_pi_div_three_l199_199197

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199197


namespace money_made_from_milk_sales_l199_199730

namespace BillMilkProblem

def total_gallons_milk : ℕ := 16
def fraction_for_sour_cream : ℚ := 1 / 4
def fraction_for_butter : ℚ := 1 / 4
def milk_to_sour_cream_ratio : ℕ := 2
def milk_to_butter_ratio : ℕ := 4
def price_per_gallon_butter : ℕ := 5
def price_per_gallon_sour_cream : ℕ := 6
def price_per_gallon_whole_milk : ℕ := 3

theorem money_made_from_milk_sales : ℕ :=
  let milk_for_sour_cream := (fraction_for_sour_cream * total_gallons_milk).toNat
  let milk_for_butter := (fraction_for_butter * total_gallons_milk).toNat
  let sour_cream_gallons := milk_for_sour_cream / milk_to_sour_cream_ratio
  let butter_gallons := milk_for_butter / milk_to_butter_ratio
  let milk_remaining := total_gallons_milk - milk_for_sour_cream - milk_for_butter
  let money_from_butter := butter_gallons * price_per_gallon_butter
  let money_from_sour_cream := sour_cream_gallons * price_per_gallon_sour_cream
  let money_from_whole_milk := milk_remaining * price_per_gallon_whole_milk
  money_from_butter + money_from_sour_cream + money_from_whole_milk = 41 :=
by
  sorry 

end BillMilkProblem

end money_made_from_milk_sales_l199_199730


namespace arccos_half_eq_pi_div_three_l199_199338

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199338


namespace area_quadrilateral_ABCD_l199_199558

noncomputable theory
open Real

def length_AB : ℝ := 3
def length_BC : ℝ := 4
def length_CD : ℝ := 12
def length_DA : ℝ := 13
def angle_CBA : ℝ := π / 2  -- 90 degrees in radians

theorem area_quadrilateral_ABCD : 
  let AB := length_AB,
      BC := length_BC,
      CD := length_CD,
      DA := length_DA,
      right_angle := angle_CBA in
  ∠CBA = right_angle →
  (AB * AB + BC * BC = 5 * 5) →
  (5 * 5 + CD * CD = DA * DA) →
  let area_ABC := 1 / 2 * AB * BC,
      area_ACD := 1 / 2 * 5 * CD in
  area_ABC + area_ACD = 36 :=
by
  intros h1 h2 h3 h4
  have h5 : area_ABC = 1 / 2 * length_AB * length_BC := by sorry
  have h6 : area_ACD = 1 / 2 * 5 * length_CD := by sorry
  have h_area_ABC := h5
  have h_area_ACD := h6
  have h_total_area := h_area_ABC + h_area_ACD
  exact h_total_area

end area_quadrilateral_ABCD_l199_199558


namespace islander_parity_l199_199751

-- Define the concept of knights and liars
def is_knight (x : ℕ) : Prop := x % 2 = 0 -- Knight count is even
def is_liar (x : ℕ) : Prop := ¬(x % 2 = 1) -- Liar count being odd is false, so even

-- Define the total inhabitants on the island and conditions
theorem islander_parity (K L : ℕ) (h₁ : is_knight K) (h₂ : is_liar L) (h₃ : K + L = 2021) : false := sorry

end islander_parity_l199_199751


namespace bishop_safe_squares_l199_199892

def chessboard_size : ℕ := 64
def total_squares_removed_king : ℕ := chessboard_size - 1
def threat_squares : ℕ := 7

theorem bishop_safe_squares : total_squares_removed_king - threat_squares = 30 :=
by
  sorry

end bishop_safe_squares_l199_199892


namespace a₈_equals_six_l199_199601

-- We define the initial conditions of the arithmetic sequence sum and also the specific terms in the sequence.
variable {a₁ d : ℝ} -- Introducing the first term a₁ and the common difference d as real numbers
variable (S₁₅ : ℝ) -- Introducing the sum of the first 15 terms S₁₅ as a real number

-- Condition: The sum of the first 15 terms of the arithmetic sequence is zero.
def S₁₅_def : S₁₅ = 0 := sorry

-- Calculate the sum of the first 15 terms using the arithmetic sequence sum formula
def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) / 2) * d

-- Specific instance that sum of the first 15 terms equals zero.
def sum_15_terms : arithmetic_sum a₁ d 15 = S₁₅ :=
  by { unfold arithmetic_sum, sorry }

-- Calculate a₈ in terms of a₁ and d
def a₈ (a₁ d : ℝ) : ℝ := a₁ + 7 * d

-- Prove a₈ equals 6 given S₁₅ equals 0
theorem a₈_equals_six :
  S₁₅ = 0 → a₈ a₁ d = 6 :=
by sorry

end a₈_equals_six_l199_199601


namespace equal_segments_CK_DL_l199_199477

variables {Point : Type}

variables (A B C D E K L : Point)

-- Assume the definitions of angle bisectors
axiom BD_bisects_CBE : ∀ (BD : Point → Point → Set Point), bisects (BD B D) (\<angle C B E\>)
axiom BD_bisects_CDA : ∀ (BD : Point → Point → Set Point), bisects (BD B D) (\<angle C D A\>)
axiom CE_bisects_ACD : ∀ (CE : Point → Point → Set Point), bisects (CE C E) (\<angle A C D\>)
axiom CE_bisects_BED : ∀ (CE : Point → Point → Set Point), bisects (CE C E) (\<angle B E D\>)

-- Assume the intersections
axiom BE_intersects_AC_at_K : intersects (diagonal BE) (diagonal AC) K
axiom BE_intersects_AD_at_L : intersects (diagonal BE) (diagonal AD) L

-- The proof statement
theorem equal_segments_CK_DL : distance C K = distance D L :=
by
  -- Assume the proof steps (left intentionally as sorry)
  sorry

end equal_segments_CK_DL_l199_199477


namespace number_of_sets_l199_199773

   theorem number_of_sets :
     let M := {a, b}
     ∃ (N : set α), (M ∪ N = {a, b, c}) ∧ set.card {N | M ∪ N = {a, b, c}} = 4 := 
   by
     sorry
   
end number_of_sets_l199_199773


namespace part_a_proof_part_b_proof_l199_199470

variables (A B C A' A1 A2 M O : Type*)
variables (a b c : ℝ)

-- Conditions of the problem
axiom triangle : ∃ (ABC : triangle), true
axiom med_meets_BC_A' : median A B C A' ∧ midpoint B C A'
axiom med_meets_circumcircle_A1 : median_on_circumcircle A B C A1
axiom symmed_meets_BC_M : symmedian A B C M ∧ meets_BC A B C M
axiom symmed_meets_circumcircle_A2 : symmedian_on_circumcircle A B C A2
axiom A1A2_contains_O : line_contains A1 A2 O

-- Part (a) statement
noncomputable def part_a : ℝ := (AA' / AM)
noncomputable def part_a_answer : ℝ := (b^2 + c^2) / (2 * b * c)

theorem part_a_proof : part_a = part_a_answer := sorry

-- Part (b) statement
noncomputable def part_b : ℝ := 1 + 4 * b^2 * c^2
noncomputable def part_b_answer : ℝ := a^2 * (b^2 + c^2)

theorem part_b_proof : part_b = part_b_answer := sorry

end part_a_proof_part_b_proof_l199_199470


namespace total_distance_is_210_l199_199702

noncomputable def ring_sequence (n : ℕ) : ℕ → ℕ
| 0     := 28
| (n+1) := ring_sequence n - 2

def total_vertical_distance (n : ℕ) : ℕ :=
  n * 1

theorem total_distance_is_210 :
  let rings := 14 in
  total_vertical_distance rings = 210 :=
by 
  sorry

end total_distance_is_210_l199_199702


namespace initial_amount_l199_199117

theorem initial_amount (x : ℝ) (h1 : x = (2*x - 10) / 2) (h2 : x = (4*x - 30) / 2) (h3 : 8*x - 70 = 0) : x = 8.75 :=
by
  sorry

end initial_amount_l199_199117


namespace find_angle_Z_l199_199920

variables (p q : Line) (X Y Z : Angle)
variables (h1 : parallel p q) 
variables (h2 : mangle X = 100) 
variables (h3 : mangle Y = 130)

theorem find_angle_Z (p q : Line) (X Y Z : Angle) 
  (h1 : parallel p q) 
  (h2 : mangle X = 100)
  (h3 : mangle Y = 130) :
  mangle Z = 130 :=
sorry

end find_angle_Z_l199_199920


namespace sin_theta_value_l199_199446

theorem sin_theta_value (θ : ℝ) (h1 : 10 * tan θ = 2 * cos θ) (h2 : 0 < θ) (h3 : θ < π) : 
  sin θ = 0.1925 :=
sorry

end sin_theta_value_l199_199446


namespace sum_of_elements_relatively_prime_to_120_is_correct_l199_199513

noncomputable def gcd (a b : ℕ) : ℕ := nat.gcd a b

def sum_of_elements_relatively_prime_to_120 (n : ℕ) : ℕ :=
  ∑ i in finset.range n, if gcd i 120 = 1 then i else 0

theorem sum_of_elements_relatively_prime_to_120_is_correct :
  sum_of_elements_relatively_prime_to_120 120 = 6084 := sorry

end sum_of_elements_relatively_prime_to_120_is_correct_l199_199513


namespace scientific_notation_correct_l199_199879

def n : ℝ := 12910000

theorem scientific_notation_correct : n = 1.291 * 10^7 := 
by
  sorry

end scientific_notation_correct_l199_199879


namespace converge_to_derivative_at_zero_l199_199498

noncomputable theory

open_locale classical

theorem converge_to_derivative_at_zero 
  (f : ℝ → ℝ)
  (f_deriv_exists : ∃ x, -1 < x ∧ x < 1 ∧ has_deriv_at f (f x) x)
  (a_n b_n : ℕ → ℝ)
  (h_a_n : ∀ n, -1 < a_n n)
  (h_a_n_below_0 : ∀ n, a_n n < 0)
  (h_b_n : ∀ n, 0 < b_n n)
  (h_b_n_below_1 : ∀ n, b_n n < 1)
  (lim_a_n : filter.tendsto a_n filter.at_top (nhds 0))
  (lim_b_n : filter.tendsto b_n filter.at_top (nhds 0))
  : filter.tendsto (λ n, (f (b_n n) - f (a_n n)) / (b_n n - a_n n)) filter.at_top (nhds (deriv f 0)) :=
sorry

end converge_to_derivative_at_zero_l199_199498


namespace arccos_half_eq_pi_div_3_l199_199275

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199275


namespace jack_should_leave_300_in_till_l199_199889

-- Defining the amounts of each type of bill
def num_100_bills := 2
def num_50_bills := 1
def num_20_bills := 5
def num_10_bills := 3
def num_5_bills := 7
def num_1_bills := 27

-- The amount he needs to hand in
def amount_to_hand_in := 142

-- Calculating the total amount in notes
def total_in_notes := 
  (num_100_bills * 100) + 
  (num_50_bills * 50) + 
  (num_20_bills * 20) + 
  (num_10_bills * 10) + 
  (num_5_bills * 5) + 
  (num_1_bills * 1)

-- Calculating the amount to leave in the till
def amount_to_leave := total_in_notes - amount_to_hand_in

-- Proof statement
theorem jack_should_leave_300_in_till :
  amount_to_leave = 300 :=
by sorry

end jack_should_leave_300_in_till_l199_199889


namespace initial_investments_l199_199677

theorem initial_investments (x y : ℝ) : 
  -- Conditions
  5000 = y + (5000 - y) ∧
  (y * (1 + x / 100) = 2100) ∧
  ((5000 - y) * (1 + (x + 1) / 100) = 3180) →
  -- Conclusion
  y = 2000 ∧ (5000 - y) = 3000 := 
by 
  sorry

end initial_investments_l199_199677


namespace tom_total_money_l199_199495

theorem tom_total_money :
  let initial_amount := 74
  let additional_amount := 86
  initial_amount + additional_amount = 160 :=
by
  let initial_amount := 74
  let additional_amount := 86
  show initial_amount + additional_amount = 160
  sorry

end tom_total_money_l199_199495


namespace minimize_quadratic_l199_199386

theorem minimize_quadratic (c : ℝ) : ∃ b : ℝ, (∀ x : ℝ, 3 * x^2 + 2 * x + c ≥ 3 * b^2 + 2 * b + c) ∧ b = -1/3 :=
by
  sorry

end minimize_quadratic_l199_199386


namespace pyramid_volume_correct_l199_199090

noncomputable def volume_of_pyramid (l α β : ℝ) (Hα : α = π/8) (Hβ : β = π/4) :=
  (1 / 3) * (l^3 / 24) * Real.sqrt (Real.sqrt 2 + 1)

theorem pyramid_volume_correct :
  ∀ (l : ℝ), l = 6 → volume_of_pyramid l (π/8) (π/4) (rfl) (rfl) = 9 * Real.sqrt (Real.sqrt 2 + 1) :=
by
  intros l hl
  rw [hl]
  norm_num
  sorry

end pyramid_volume_correct_l199_199090


namespace monotonic_intervals_max_value_l199_199423

def f (x a : ℝ) : ℝ := cos (2 * x + π / 3) + sqrt 3 * sin (2 * x) + 2 * a

theorem monotonic_intervals (k : ℤ) (x : ℝ) :
  -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π :=
sorry

theorem max_value (a x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4) (hmin : f x a = 0) :
  a = -1 / 4 → (∀ y ∈ set.Icc 0 (π / 4), f y a ≤ 1) :=
sorry

end monotonic_intervals_max_value_l199_199423


namespace cents_saved_eq_zero_l199_199987

def in_store_price : ℝ := 139.99
def advertisement_payment : ℝ := 33.00
def shipping_handling : ℝ := 11.99

theorem cents_saved_eq_zero : 
  let total_advertisement_cost := (4 * advertisement_payment) + shipping_handling in
  let cents_saved := ((in_store_price - total_advertisement_cost) * 100) in
  cents_saved = 0 :=
by 
  sorry

end cents_saved_eq_zero_l199_199987


namespace find_digit_set_l199_199694
noncomputable def P (d : ℕ) : ℝ := 
  if d ≠ 0 then log (d+1) / log 10 - log d / log 10 else 0

theorem find_digit_set :
  (P 2 = ((P 4) + (P 5) + (P 6) + (P 7) + (P 8)) / 2) :=
by
  sorry

end find_digit_set_l199_199694


namespace triangle_length_BC_l199_199884

theorem triangle_length_BC 
  (A B C : ℝ)
  (AB AC BC : ℝ) 
  (h1 : cos (2 * A - B) + sin (A + B) = sqrt 2 + 1)
  (h2 : AB = 6)
  (h3 : ∠A + ∠B = 90)
  (h4 : ∠A = 45)
  (h5 : ∠B = 45) 
  : BC = 6 * sqrt 2 := 
by
  sorry

end triangle_length_BC_l199_199884


namespace lesser_of_two_numbers_l199_199985

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x * y = 1050) : min x y = 30 :=
sorry

end lesser_of_two_numbers_l199_199985


namespace scientific_notation_l199_199882

theorem scientific_notation (n : ℤ) (hn : n = 12910000) : ∃ a : ℝ, (a = 1.291 ∧ hn = (a * 10^7).to_int) :=
by
  sorry

end scientific_notation_l199_199882


namespace minimum_players_multiple_of_10_l199_199869

-- Define the jersey numbers as a 3x9 matrix
def jersey_numbers : Matrix (Fin 3) (Fin 9) ℕ := 
  -- insert the specific jersey numbers here if known, or assume an arbitrary matrix for the sake of definition
  Matrix.of_list [[1, 2, 3, 4, 5, 6, 7, 8, 9], 
                  [10, 11, 12, 13, 14, 15, 16, 17, 18], 
                  [19, 20, 21, 22, 23, 24, 25, 26, 27]]

-- Define a function that computes the sum of elements in a sub-rectangle
def sum_sub_rect (m : Matrix (Fin 3) (Fin 9) ℕ) (r1 r2 : Fin 3) (c1 c2 : Fin 9) : ℕ :=
  Finset.sum (Finset.fin_range (r2 - r1 + 1)) (λ i, Finset.sum (Finset.fin_range (c2 - c1 + 1)) (λ j, m (r1 + i) (c1 + j)))

-- Define the main theorem for the problem
theorem minimum_players_multiple_of_10 :
  ∃ (s : ℕ) (r1 r2 : Fin 3) (c1 c2 : Fin 9), (1 ≤ r2 - r1 + 1) ∧ (r2 - r1 + 1 ≤ 3) ∧ 
                                                (1 ≤ c2 - c1 + 1) ∧ (c2 - c1 + 1 ≤ 9) ∧
                                                (sum_sub_rect jersey_numbers r1 r2 c1 c2 % 10 = 0) ∧ 
                                                (r2 - r1 + 1) * (c2 - c1 + 1) = 2 :=
sorry

end minimum_players_multiple_of_10_l199_199869


namespace inequality_solution_l199_199824

noncomputable def f : ℝ → ℝ := sorry

axiom f_deriv : ∀ x : ℝ, deriv f x - 2 * f x - 4 > 0
axiom f_at_0 : f 0 = -1

theorem inequality_solution : ∀ x : ℝ, x > 0 → f x > real.exp (2 * x) - 2 :=
by
  intro x hx
  sorry

end inequality_solution_l199_199824


namespace vertex_farthest_from_origin_l199_199564

theorem vertex_farthest_from_origin (center : ℝ × ℝ) (area : ℝ) (top_side_horizontal : Prop) (dilation_center : ℝ × ℝ) (scale_factor : ℝ) :
  center = (10, -5) ∧ area = 16 ∧ top_side_horizontal ∧ dilation_center = (0, 0) ∧ scale_factor = 3 →
  ∃ (vertex_farthest : ℝ × ℝ), vertex_farthest = (36, -21) :=
by
  sorry

end vertex_farthest_from_origin_l199_199564


namespace mapping_sum_l199_199417

theorem mapping_sum (f : ℝ × ℝ → ℝ × ℝ) (a b : ℝ)
(h1 : ∀ x y, f (x, y) = (x, x + y))
(h2 : (a, b) = f (1, 3)) :
  a + b = 5 :=
sorry

end mapping_sum_l199_199417


namespace line_single_point_not_necessarily_tangent_l199_199689

-- Define a curve
def curve : Type := ℝ → ℝ

-- Define a line
def line (m b : ℝ) : curve := λ x => m * x + b

-- Define a point of intersection
def intersects_at (l : curve) (c : curve) (x : ℝ) : Prop :=
  l x = c x

-- Define the property of having exactly one common point
def has_single_intersection (l : curve) (c : curve) : Prop :=
  ∃ x, ∀ y ≠ x, l y ≠ c y

-- Define the tangent line property
def is_tangent (l : curve) (c : curve) (x : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ((c (x + h) - c x) / h - (l (x + h) - l x) / h) < ε

-- The proof statement: There exists a curve c and a line l such that l has exactly one intersection point with c, but l is not necessarily a tangent to c.
theorem line_single_point_not_necessarily_tangent :
  ∃ c : curve, ∃ l : curve, has_single_intersection l c ∧ ∃ x, ¬ is_tangent l c x :=
sorry

end line_single_point_not_necessarily_tangent_l199_199689


namespace problem_l199_199950

variable (a : ℝ)

theorem problem (h : 5 = a + a⁻¹) : a^4 + a⁻⁴ = 527 :=
by {
  sorry -- The proof goes here
}

end problem_l199_199950


namespace greatest_multiple_of_5_and_7_less_than_1000_l199_199631

theorem greatest_multiple_of_5_and_7_less_than_1000 : ∃ x : ℕ, (x % 5 = 0) ∧ (x % 7 = 0) ∧ (x < 1000) ∧ (∀ y : ℕ, (y % 5 = 0) ∧ (y % 7 = 0) ∧ (y < 1000) → y ≤ x) ∧ x = 980 :=
begin
  sorry
end

end greatest_multiple_of_5_and_7_less_than_1000_l199_199631


namespace determine_a_if_roots_are_prime_l199_199804
open Nat

theorem determine_a_if_roots_are_prime (a x1 x2 : ℕ) (h1 : Prime x1) (h2 : Prime x2) 
  (h_eq : x1^2 - x1 * a + a + 1 = 0) :
  a = 5 :=
by
  -- Placeholder for the proof
  sorry

end determine_a_if_roots_are_prime_l199_199804


namespace factors_of_10_in_factorial_product_l199_199565

theorem factors_of_10_in_factorial_product (a b c m n : ℕ) 
  (h1 : b = 2 * a) 
  (h2 : c = 3 * a) 
  (h3 : a + b + c = 6024)
  (h4 : a * b * c = m * 10^n) : 
  ∑ k in {1, 2}, (∑ k in {1, 2, 3}, floor (a / (5^k)) + floor (b / (5^k)) + floor (c / (5^k))) = 1491 :=
sorry

end factors_of_10_in_factorial_product_l199_199565


namespace trigonometric_identity_l199_199563

theorem trigonometric_identity (α : ℝ) :
  (cos (2 * α) - sin (4 * α) - cos (6 * α)) / (cos (2 * α) + sin (4 * α) - cos (6 * α)) 
  = (tan (α - real.pi / 12) * cot (α + real.pi / 12)) :=
by sorry

end trigonometric_identity_l199_199563


namespace arccos_half_eq_pi_over_three_l199_199207

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199207


namespace leftover_yarn_after_square_l199_199100

theorem leftover_yarn_after_square (total_yarn : ℕ) (side_length : ℕ) (left_yarn : ℕ) :
  total_yarn = 35 →
  (4 * side_length ≤ total_yarn ∧ (∀ s : ℕ, s > side_length → 4 * s > total_yarn)) →
  left_yarn = total_yarn - 4 * side_length →
  left_yarn = 3 :=
by
  sorry

end leftover_yarn_after_square_l199_199100


namespace house_numbers_count_l199_199359

theorem house_numbers_count : 
  let two_digit_primes := {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47} in
  let potential_wx := (finset.filter (λ n, n < 50) two_digit_primes).to_finset in
  let potential_yz := (finset.filter (λ n, n < 50) two_digit_primes).to_finset in
  potential_wx.card * (potential_yz.card - 1) = 110 :=
by
  sorry

end house_numbers_count_l199_199359


namespace deductive_reasoning_option_l199_199171

inductive ReasoningType
| deductive
| inductive
| analogical

-- Definitions based on conditions
def option_A : ReasoningType := ReasoningType.inductive
def option_B : ReasoningType := ReasoningType.deductive
def option_C : ReasoningType := ReasoningType.inductive
def option_D : ReasoningType := ReasoningType.analogical

-- The main theorem to prove
theorem deductive_reasoning_option : option_B = ReasoningType.deductive :=
by sorry

end deductive_reasoning_option_l199_199171


namespace radius_of_circumcircle_of_intersections_is_one_l199_199611

theorem radius_of_circumcircle_of_intersections_is_one
  (A1 B1 C1 P A B C : Point)
  (r : ℝ)
  (hA1 : dist A1 P = 1)
  (hB1 : dist B1 P = 1)
  (hC1 : dist C1 P = 1)
  (hPA : ∃ Q, ∀ R, (R ≠ Q → on_circle A1 1 R ∧ on_circle C1 1 R))
  (hPB : ∃ Q, ∀ R, (R ≠ Q → on_circle A1 1 R ∧ on_circle B1 1 R))
  (hPC : ∃ Q, ∀ R, (R ≠ Q → on_circle B1 1 R ∧ on_circle C1 1 R))
  (hIntersectionA : on_circle A1 1 A ∧ on_circle C1 1 A)
  (hIntersectionB : on_circle A1 1 B ∧ on_circle B1 1 B)
  (hIntersectionC : on_circle B1 1 C ∧ on_circle C1 1 C) :
  radius_of_circumcircle A B C = 1 :=
sorry

end radius_of_circumcircle_of_intersections_is_one_l199_199611


namespace maya_model_height_l199_199682

noncomputable def calculate_miniature_height 
  (height_actual : ℝ) 
  (volume_actual : ℝ) 
  (volume_miniature : ℝ) 
: ℝ :=
  height_actual / (volume_actual / volume_miniature)^(1 / 3)

theorem maya_model_height :
  let height_actual := 50
  let volume_actual := 150000
  let volume_miniature := 0.2
  let target_height_miniature := 0.55
  calculate_miniature_height height_actual volume_actual volume_miniature ≈ target_height_miniature
:= by
  sorry

end maya_model_height_l199_199682


namespace final_number_blackboard_l199_199971

theorem final_number_blackboard :
  (∏ i in Finset.range 2010 | i > 0, (1 + (1 / i: ℝ))) - 1 = 2010 :=
by
  sorry

end final_number_blackboard_l199_199971


namespace cauchy_schwarz_inequality_example_l199_199511

theorem cauchy_schwarz_inequality_example
  (n : ℕ)
  (a b : ℕ → ℝ)
  (pos_a : ∀ i, 0 < a i)
  (pos_b : ∀ i, 0 < b i)
  (h : ∑ i in finset.range n, a i = ∑ i in finset.range n, b i) :
  ∑ i in finset.range n, (a i)^2 / (a i + b i) ≥ ∑ i in finset.range n, a i / 2 :=
sorry

end cauchy_schwarz_inequality_example_l199_199511


namespace polar_eqn_of_ellipse_range_of_x_plus_2y_l199_199406

noncomputable def ellipse_parametric_eqn := ∃ θ : ℝ, (x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ)

theorem polar_eqn_of_ellipse (h1 : ellipse_parametric_eqn) :
  ∃ ρ θ : ℝ, ρ^2 = 36 / (4 + 5 * Real.sin θ^2) :=
sorry

theorem range_of_x_plus_2y (h1 : ellipse_parametric_eqn) :
  ∀ θ : ℝ, -5 ≤ 3 * Real.cos θ + 4 * Real.sin θ ∧ 3 * Real.cos θ + 4 * Real.sin θ ≤ 5 :=
sorry

end polar_eqn_of_ellipse_range_of_x_plus_2y_l199_199406


namespace line_KL_passes_through_O_l199_199910

-- Definitions
variables {O : Type} [metric_space O] [normed_group O] [normed_space ℝ O]
variables {A B C D E F K L : O}
variables (circle : set O)
variables (cyclic_quad : is_cyclic_quad A B C D)
variables (midpoint_arc_AB : is_midpoint_arc E A B)
variables (midpoint_arc_CD : is_midpoint_arc F C D)
variables (parallel_diag_E : line_through E ∥ diagonal_AB_CD)
variables (parallel_diag_F : line_through F ∥ diagonal_CD_AB)

-- Theorem
theorem line_KL_passes_through_O :
  passes_through (line K L) O :=
sorry

end line_KL_passes_through_O_l199_199910


namespace sum_of_divisors_252_l199_199643

open BigOperators

-- Definition of the sum of divisors for a given number n
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in Nat.divisors n, d

-- Statement of the problem
theorem sum_of_divisors_252 : sum_of_divisors 252 = 728 := 
sorry

end sum_of_divisors_252_l199_199643


namespace arccos_of_half_eq_pi_over_three_l199_199295

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199295


namespace arccos_half_eq_pi_div_three_l199_199245

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199245


namespace arccos_half_eq_pi_over_three_l199_199204

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199204


namespace fried_frog_probability_l199_199390

def Grid : Type :=
| Center
| CornerA
| CornerB
| CornerD
| CornerE
| EdgeF
| EdgeG
| EdgeH
| EdgeI

-- Transition probabilities for movement including wrap-around rules
def transition_prob (state : Grid) : Grid → ℚ :=
  match state with
  | Grid.Center => fun g => if g = Grid.EdgeF ∨ g = Grid.EdgeG ∨ g = Grid.EdgeH ∨ g = Grid.EdgeI then 1/4 else 0
  | Grid.EdgeF => fun g => if g = Grid.Center ∨ g = Grid.CornerA ∨ g = Grid.EdgeG ∨ g = Grid.EdgeH then 1/4 else 0
  | Grid.EdgeG => fun g => if g = Grid.Center ∨ g = Grid.CornerB ∨ g = Grid.EdgeF ∨ g = Grid.EdgeI then 1/4 else 0
  | Grid.EdgeH => fun g => if g = Grid.Center ∨ g = Grid.CornerD ∨ g = Grid.EdgeF ∨ g = Grid.EdgeI then 1/4 else 0
  | Grid.EdgeI => fun g => if g = Grid.Center ∨ g = Grid.CornerE ∨ g = Grid.EdgeG ∨ g = Grid.EdgeH then 1/4 else 0
  | _ => fun _ => 0

-- Recursive probability of reaching a corner square in n hops
def p_n (n : ℕ) : Grid → ℚ
| 0 => fun state => if state = Grid.CornerA ∨ state = Grid.CornerB ∨ state = Grid.CornerD ∨ state = Grid.CornerE then 1 else 0
| (n+1) => fun state => (∑ t : Grid, transition_prob state t * p_n n t)

theorem fried_frog_probability : p_n 4 Grid.Center = 25/32 :=
by
  sorry

end fried_frog_probability_l199_199390


namespace arccos_half_eq_pi_div_three_l199_199247

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199247


namespace sum_of_divisors_252_l199_199641

open BigOperators

-- Definition of the sum of divisors for a given number n
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in Nat.divisors n, d

-- Statement of the problem
theorem sum_of_divisors_252 : sum_of_divisors 252 = 728 := 
sorry

end sum_of_divisors_252_l199_199641


namespace zero_point_neg_x0_l199_199798

variables {f : ℝ → ℝ}

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Given function satisfies the condition of being odd
def is_zero_point (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f x0 + exp x0 = 0

-- The function to evaluate -x0
def target_function (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  exp x * f x - 1

-- The theorem stating that -x0 is a zero point of the target function
theorem zero_point_neg_x0 (h1 : is_odd_function f) (h2 : is_zero_point f x0) : target_function f (-x0) = 0 :=
sorry

end zero_point_neg_x0_l199_199798


namespace arccos_half_eq_pi_div_three_l199_199246

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199246


namespace figure_perimeter_l199_199059

theorem figure_perimeter 
  (side_length : ℕ)
  (inner_large_square_sides : ℕ)
  (shared_edge_length : ℕ)
  (rectangle_dimension_1 : ℕ)
  (rectangle_dimension_2 : ℕ) 
  (h1 : side_length = 2)
  (h2 : inner_large_square_sides = 4)
  (h3 : shared_edge_length = 2)
  (h4 : rectangle_dimension_1 = 2)
  (h5 : rectangle_dimension_2 = 1) : 
  let large_square_perimeter := inner_large_square_sides * side_length
  let horizontal_perimeter := large_square_perimeter - shared_edge_length + rectangle_dimension_1 + rectangle_dimension_2
  let vertical_perimeter := large_square_perimeter
  horizontal_perimeter + vertical_perimeter = 33 := 
by
  sorry

end figure_perimeter_l199_199059


namespace inv_prop_x_y_l199_199567

theorem inv_prop_x_y (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 4) (h3 : y = 2) (h4 : y = 10) : x = 4 / 5 :=
by
  sorry

end inv_prop_x_y_l199_199567


namespace intersection_lines_on_PM_l199_199041

open EuclideanGeometry

variable {A B C A₁ B₁ C₁ P M Q : Point}

/-- Conditions:
1. Points A₁, B₁, and C₁ lie on sides BC, CA, and AB of triangle ABC respectively.
2. Segments AA₁, BB₁, and CC₁ intersect at point P.
3. la is the line connecting the midpoints of BC and B₁C₁.
4. lb is the line connecting the midpoints of CA and C₁A₁.
5. lc is the line connecting the midpoints of AB and A₁B₁. -/
axiom condition1 : ∃ (D E F : Point), Collinear D B C ∧ Collinear E C A ∧ Collinear F A B ∧ A₁ = Intersection D E ∧ B₁ = Intersection E F ∧ C₁ = Intersection F D
axiom condition2 : Intersection (LineThrough A A₁) (Intersection (LineThrough B B₁) (LineThrough C C₁)) = P
axiom condition3 : ∃ (D₁ E₁ : Point), D₁ = Midpoint B C ∧ E₁ = Midpoint B₁ C₁ ∧ (LineThrough D₁ E₁) = la
axiom condition4 : ∃ (D₂ E₂ : Point), D₂ = Midpoint C A ∧ E₂ = Midpoint C₁ A₁ ∧ (LineThrough D₂ E₂) = lb
axiom condition5 : ∃ (D₃ E₃ : Point), D₃ = Midpoint A B ∧ E₃ = Midpoint A₁ B₁ ∧ (LineThrough D₃ E₃) = lc
axiom condition6 : M = Centroid A B C

theorem intersection_lines_on_PM :
  ∃ Q : Point, Intersection la lb lc = Q ∧ OnSegment P M Q := by
  sorry

end intersection_lines_on_PM_l199_199041


namespace count_integer_length_chords_l199_199936

/-- Point P is 9 units from the center of a circle with radius 15. -/
def point_distance_from_center : ℝ := 9

def circle_radius : ℝ := 15

/-- Correct answer to the number of different chords that contain P and have integer lengths. -/
def correct_answer : ℕ := 7

/-- Proving the number of chords containing P with integer lengths given the conditions. -/
theorem count_integer_length_chords : 
  ∀ (r_P : ℝ) (r_circle : ℝ), r_P = point_distance_from_center → r_circle = circle_radius → 
  (∃ n : ℕ, n = correct_answer) :=
by 
  intros r_P r_circle h1 h2
  use 7 
  sorry

end count_integer_length_chords_l199_199936


namespace sum_of_divisors_of_252_l199_199649

theorem sum_of_divisors_of_252 :
  ∑ (d : ℕ) in (finset.filter (λ x, 252 % x = 0) (finset.range (252 + 1))), d = 728 :=
by
  sorry

end sum_of_divisors_of_252_l199_199649


namespace sum_arithmetic_sequence_l199_199411

theorem sum_arithmetic_sequence {a : ℕ → ℤ} (S : ℕ → ℤ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 13 = S 2000 →
  S 2013 = 0 :=
by
  sorry

end sum_arithmetic_sequence_l199_199411


namespace replace_star_with_2x_l199_199555

theorem replace_star_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_star_with_2x_l199_199555


namespace arccos_one_half_l199_199225

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199225


namespace Edmund_earns_64_dollars_l199_199755

-- Conditions
def chores_per_week : Nat := 12
def pay_per_extra_chore : Nat := 2
def chores_per_day : Nat := 4
def weeks : Nat := 2
def days_per_week : Nat := 7

-- Goal
theorem Edmund_earns_64_dollars :
  let total_chores_without_extra := chores_per_week * weeks
  let total_chores_with_extra := chores_per_day * (days_per_week * weeks)
  let extra_chores := total_chores_with_extra - total_chores_without_extra
  let earnings := pay_per_extra_chore * extra_chores
  earnings = 64 :=
by
  sorry

end Edmund_earns_64_dollars_l199_199755


namespace power_difference_mod_7_l199_199189

theorem power_difference_mod_7 :
  (45^2011 - 23^2011) % 7 = 5 := by
  have h45 : 45 % 7 = 3 := by norm_num
  have h23 : 23 % 7 = 2 := by norm_num
  sorry

end power_difference_mod_7_l199_199189


namespace day_of_week_2_2312_wednesday_l199_199163

def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ ((y % 4 = 0) ∧ (y % 100 ≠ 0))

theorem day_of_week_2_2312_wednesday (birth_year : ℕ) (birth_day : String) 
  (h1 : birth_year = 2312 - 300)
  (h2 : birth_day = "Wednesday") :
  "Monday" = "Monday" :=
sorry

end day_of_week_2_2312_wednesday_l199_199163


namespace probability_10101_before_010101_l199_199685

def fairCoin : Prob := { heads := 1 / 2, tails := 1 / 2 }

def winningProb (x y : Prob) : Prob :=
(x + y) / 2

variable {x y : Prob}

axiom x_def : x = x * (5/8) + y * (5/16) + (1/16)
axiom y_def : y = x * (5/16) + y * (23/32)

theorem probability_10101_before_010101 : winningProb (9 / 29) (5 / 29) = 7 / 29 := by 
  sorry

end probability_10101_before_010101_l199_199685


namespace surface_area_equation_l199_199008

-- Definitions based on conditions
def cube_side_length : ℝ := 10
def point_P := (3 : ℝ)
def point_Q := (3 : ℝ)
def point_R := (3 : ℝ)

/- The surface area of the solid T, including the walls of the tunnel, 
is expressed as x + y * sqrt z, where x, y, and z are positive integers and 
z is not divisible by the square of any prime. We need to prove that x + y + z = 945.
-/
theorem surface_area_equation :
  ∃ (x y z : ℕ), x = 882 ∧ y = 51 ∧ z = 12 ∧ (z > 1 ∧ ∀ p : ℕ, p ∣ z → Nat.Prime p → p * p ∣ z → false) ∧
  x + y + z = 945 :=
begin
  -- skip proof, provided statement only
  sorry
end

end surface_area_equation_l199_199008


namespace discount_per_bear_l199_199998

/-- Suppose the price of the first bear is $4.00 and Wally pays $354.00 for 101 bears.
 Prove that the discount per bear after the first bear is $0.50. -/
theorem discount_per_bear 
  (price_first : ℝ) (total_bears : ℕ) (total_paid : ℝ) (price_rest_bears : ℝ )
  (h1 : price_first = 4.0) (h2 : total_bears = 101) (h3 : total_paid = 354.0) : 
  (price_first + (total_bears - 1) * price_rest_bears - total_paid) / (total_bears - 1) = 0.50 :=
sorry

end discount_per_bear_l199_199998


namespace find_AC_length_l199_199528

-- Define the problem conditions
variables (A B C D E : Type)
variables [noncomputable] [Add A] [Add B] [Add C] [Sub D] [Sub E] -- Use appropriate types for geometry

-- Define lengths and angles as constants or variables
constants (AD EC BD DE BE AB : ℝ)
constants (angleBDC angleDEB : ℝ)

-- Define geometric constraints
axiom h1 : AD = EC
axiom h2 : BD = DE
axiom h3 : angleBDC = angleDEB
axiom h4 : AB = 7
axiom h5 : BE = 2

-- The theorem to prove
theorem find_AC_length : AD + EC = 12 :=
by
  -- formalize the conditions
  have hAD_EC : AD = 5 := sorry
  have hDC_AB : EC = 7 := sorry
  have hAC_length : AD + EC = 12 := sorry
  exact hAC_length

end find_AC_length_l199_199528


namespace arccos_of_half_eq_pi_over_three_l199_199298

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199298


namespace unique_arrangement_exists_l199_199770

theorem unique_arrangement_exists (n : ℕ) (h : n ≥ 2) :
  (∃ (a : Fin n → ℕ), (∀ k:ℕ, 1 ≤ k → k ≤ n → (Finset.sum (Finset.range k) (λ i, a ⟨i, sorry⟩) % k = 0))) ↔ (n = 3) :=
sorry

end unique_arrangement_exists_l199_199770


namespace mean_less_than_median_l199_199346

noncomputable def sick_days_table : List (ℕ × ℕ) := [(0, 4), (1, 2), (2, 5), (3, 2), (4, 1), (5, 1)]

def total_students : ℕ := sick_days_table.foldr (λ x acc => x.2 + acc) 0

def median (table : List (ℕ × ℕ)) : ℚ :=
  let cum_freq := table.scanl (λ acc x => acc + x.2) 0
  let n := (total_students + 1) / 2
  table.enum.find (λ x => cum_freq.get! x.fst >= n).get! .snd

def mean (table : List (ℕ × ℕ)) : ℚ :=
  table.foldr (λ x acc => acc + x.1 * x.2) 0 / total_students

theorem mean_less_than_median :
  mean sick_days_table - median sick_days_table = -1 / 5 :=
sorry

end mean_less_than_median_l199_199346


namespace arccos_one_half_l199_199235

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199235


namespace coefficient_of_x2_in_expansion_l199_199578

-- Define the problem statement and the necessary conditions.

theorem coefficient_of_x2_in_expansion :
  (coeff (expand ((1 - X)^6 * (1 + X)^4)) 2) = -3 :=
by
  sorry -- No proof is provided here, just the statement.

end coefficient_of_x2_in_expansion_l199_199578


namespace geometric_series_sum_l199_199503

noncomputable theory
open Classical

theorem geometric_series_sum (a b : ℝ) (hb : b ≠ 0) 
  (h : (a / b) / (1 - 1 / b) = 6) : 
  (a / (a + b)) / (1 - 1 / (a + b)) = 6 / 7 := 
by 
  sorry

end geometric_series_sum_l199_199503


namespace multiply_digits_correctness_l199_199523

theorem multiply_digits_correctness (a b c : ℕ) :
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c :=
by sorry

end multiply_digits_correctness_l199_199523


namespace outfits_count_l199_199445

def num_outfits (n : Nat) (total_colors : Nat) : Nat :=
  let total_combinations := n * n * n
  let undesirable_combinations := total_colors
  total_combinations - undesirable_combinations

theorem outfits_count : num_outfits 5 5 = 120 :=
  by
  sorry

end outfits_count_l199_199445


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199328

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199328


namespace parallel_lines_slope_l199_199454

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + y + 2a = 0) ∧ (∀ x y : ℝ, x + ay + 3 = 0) →
  a = 1 ∨ a = -1 :=
by
  sorry

end parallel_lines_slope_l199_199454


namespace calculate_z_l199_199029

noncomputable def z : ℂ := (Complex.sqrt 3 + Complex.i) / 2

theorem calculate_z {
  ∑ k in (Finset.range 8).map (λ k, z ^ (k + 1)^3) *
  ∑ k in (Finset.range 8).map (λ k, z ^ (-(k + 1)^3)) 
  = 64 := by
  sorry

end calculate_z_l199_199029


namespace curve_is_parabola_l199_199373

-- Define the condition: the curve is defined by the given polar equation
def polar_eq (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

-- The main theorem statement: Prove that the curve defined by the equation is a parabola
theorem curve_is_parabola (r θ : ℝ) (h : polar_eq r θ) : ∃ x y : ℝ, x = 1 + 2 * y :=
sorry

end curve_is_parabola_l199_199373


namespace probability_heads_penny_dime_quarter_l199_199571

theorem probability_heads_penny_dime_quarter :
  let total_outcomes := 2^5 in
  let favorable_outcomes := 2^2 in
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 8 :=
by
  let total_outcomes : ℚ := 32
  let favorable_outcomes : ℚ := 4
  have probability : ℚ := favorable_outcomes / total_outcomes
  have expected : ℚ := 1 / 8
  show probability = expected
  sorry

end probability_heads_penny_dime_quarter_l199_199571


namespace emily_lives_after_level_l199_199127

theorem emily_lives_after_level : 
  let initial_lives := 42
  let lives_lost := 25
  let lives_gained := 24
  (initial_lives - lives_lost + lives_gained) = 41 :=
by
  let initial_lives := 42
  let lives_lost := 25
  let lives_gained := 24
  show (initial_lives - lives_lost + lives_gained) = 41
  from sorry

end emily_lives_after_level_l199_199127


namespace decreasing_interval_l199_199068

def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x + 2

theorem decreasing_interval :
  {x : ℝ | fderiv ℝ f x < 0} = set.Ioo (-2 : ℝ) (2 : ℝ) :=
by {
  sorry
}

end decreasing_interval_l199_199068


namespace infinite_monochromatic_rectangles_l199_199019

-- Given conditions
axiom exists_coloring : ∀ (n : ℕ) (h : n ≥ 2), (ℝ × ℝ) → Fin n → Prop

-- Given a natural number n (n ≥ 2) and a coloring of the plane with n colors,
-- show that there exists an infinite number of rectangles where all vertices have the same color.
theorem infinite_monochromatic_rectangles (n : ℕ) (h : n ≥ 2)
  (coloring : (ℝ × ℝ) → Fin n) : ∃ (inf_rectangles : ℕ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)),
  ∀ k : ℕ, coloring (inf_rectangles k).1.1 = coloring (inf_rectangles k).1.2 ∧
           coloring (inf_rectangles k).1.1 = coloring (inf_rectangles k).2.1 ∧
           coloring (inf_rectangles k).1.1 = coloring (inf_rectangles k).2.2 :=
sorry

end infinite_monochromatic_rectangles_l199_199019


namespace last_two_nonzero_digits_of_70_factorial_are_04_l199_199075

-- Given conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial_are_04 :
  let n := 70;
  ∀ t : ℕ, 
    t = factorial n → t % 100 ≠ 0 → (t % 100) / 10 != 0 → 
    (t % 100) = 04 :=
sorry

end last_two_nonzero_digits_of_70_factorial_are_04_l199_199075


namespace replace_asterisk_with_2x_l199_199540

-- Defining conditions for Lean
def expr_with_monomial (a : ℤ) : ℤ := (x : ℤ) → (x^3 - 2)^2 + (x^2 + a * x)^2

-- The statement of the proof in Lean 4
theorem replace_asterisk_with_2x : expr_with_monomial 2 = (x : ℤ) → x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_asterisk_with_2x_l199_199540


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199331

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199331


namespace additional_toothpicks_needed_l199_199723

theorem additional_toothpicks_needed :
  ∀ stairs3 stairs6 : ℕ, 
    stairs3 = 18 → 
    stairs6 = stairs3 + 10 + 12 + 14 →
    stairs6 - stairs3 = 36 :=
by
  intros stairs3 stairs6 h3 h6
  rw h6
  rw h3
  norm_num


end additional_toothpicks_needed_l199_199723


namespace tangent_to_incircle_l199_199066

noncomputable def triangleABC (A B C : Point) : Prop := true -- triangulation is given

noncomputable def incenter (I : Point) (A B C : Point) : Prop := 
  I is the incenter of triangleABC A B C

noncomputable def perpendicular_line_through_point (l : Line) (I C : Point) : Prop := 
  l is perpendicular to line through I and C

noncomputable def intersects (l : Line) (line1 line2 : Line) (A' B' : Point) : Prop := 
  l intersects line AC at A' and line BC at B'

noncomputable def symmetric_point (p1 p2 : Point) (mid : Point) : Prop := 
  mid is the midpoint of p1 and p2

theorem tangent_to_incircle 
  (A B C I A' B' A'' B'' : Point) (e: Line)
  (h1 : incenter I A B C)
  (h2 : perpendicular_line_through_point e I C)
  (h3 : intersects e (line_through A C) (line_through B C) A' B')
  (h4 : symmetric_point A A'' A')
  (h5 : symmetric_point B B'' B')
  : tangent A'' B'' (incircle (triangleABC A B C)) := 
sorry

end tangent_to_incircle_l199_199066


namespace degree_of_polynomial_l199_199629

-- Define the polynomial function in Lean
def polynomial : ℝ[X] := 3 + 7 * (X^5) + 15 + 4 * real.exp(1) * (X^6) + 7 * real.sqrt(3) * (X^3) + 11

-- State the theorem to prove the degree of the polynomial
theorem degree_of_polynomial : polynomial.degree = 6 := by sorry

end degree_of_polynomial_l199_199629


namespace num_rectangular_arrays_with_48_chairs_l199_199142

theorem num_rectangular_arrays_with_48_chairs : 
  ∃ n, (∀ (r c : ℕ), 2 ≤ r ∧ 2 ≤ c ∧ r * c = 48 → (n = 8 ∨ n = 0)) ∧ (n = 8) :=
by 
  sorry

end num_rectangular_arrays_with_48_chairs_l199_199142


namespace ratio_expression_l199_199849

variable {x : ℚ}
def A := 3 * x
def B := 2 * x
def C := 5 * x

theorem ratio_expression :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
  sorry

end ratio_expression_l199_199849


namespace arccos_half_eq_pi_over_three_l199_199203

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199203


namespace problem_proof_l199_199506

noncomputable def a : ℝ := 4 ^ 0.1
noncomputable def b : ℝ := Real.log 0.1 / Real.log 4
noncomputable def c : ℝ := 0.4 ^ 0.2

theorem problem_proof : a > c ∧ c > b := by
  have h1 : a = 4 ^ 0.1 := rfl
  have h2 : b = Real.log 0.1 / Real.log 4 := rfl
  have h3 : c = 0.4 ^ 0.2 := rfl
  sorry

end problem_proof_l199_199506


namespace number_of_taxis_l199_199972

-- Define the conditions explicitly
def number_of_cars : ℕ := 3
def people_per_car : ℕ := 4
def number_of_vans : ℕ := 2
def people_per_van : ℕ := 5
def people_per_taxi : ℕ := 6
def total_people : ℕ := 58

-- Define the number of people in cars and vans
def people_in_cars := number_of_cars * people_per_car
def people_in_vans := number_of_vans * people_per_van
def people_in_taxis := total_people - (people_in_cars + people_in_vans)

-- The theorem we need to prove
theorem number_of_taxis : people_in_taxis / people_per_taxi = 6 := by
  sorry

end number_of_taxis_l199_199972


namespace inequality_proof_l199_199026

theorem inequality_proof (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

end inequality_proof_l199_199026


namespace problem_g6_l199_199070

theorem problem_g6 : (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x + y) = g x * g y) 
  (h2 : g 2 = 5) : 
  g 6 = 125 :=
sorry

end problem_g6_l199_199070


namespace longest_side_is_43_in_l199_199711

def height_of_pane (x : ℕ) := 7 * x
def width_of_pane (x : ℕ) := 2 * x
def border_width := 3
def total_height (x : ℕ) := 4 * height_of_pane x + 5 * border_width
def total_width (x : ℕ) := 2 * width_of_pane x + 3 * border_width

theorem longest_side_is_43_in (x : ℕ) (hx : x = 1) : 
  total_height x = 43 := 
by
  rw [hx, total_height, height_of_pane, border_width]
  norm_num
  sorry

end longest_side_is_43_in_l199_199711


namespace combined_alloy_force_l199_199715

-- Define the masses and forces exerted by Alloy A and Alloy B
def mass_A : ℝ := 6
def force_A : ℝ := 30
def mass_B : ℝ := 3
def force_B : ℝ := 10

-- Define the combined mass and force
def combined_mass : ℝ := mass_A + mass_B
def combined_force : ℝ := force_A + force_B

-- Theorem statement
theorem combined_alloy_force :
  combined_force = 40 :=
by
  -- The proof is omitted.
  sorry

end combined_alloy_force_l199_199715


namespace angle_in_second_quadrant_l199_199409

noncomputable theory

-- Definitions based on the conditions
def second_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (π/2 + 2 * k * π < α ∧ α < π + 2 * k * π)

def cos_condition (α : ℝ) : Prop :=
  |Real.cos (α / 3)| = -Real.cos (α / 3)

-- The main theorem to be proved
theorem angle_in_second_quadrant (α : ℝ) (h1 : second_quadrant α) (h2 : cos_condition α) : 
  ∃ k : ℤ, (π/6 + 2 * k * π/3 < α / 3 ∧ α / 3 < π/3 + 2 * k * π/3 ∧ Real.cos (α / 3) < 0) :=
sorry

end angle_in_second_quadrant_l199_199409


namespace arccos_half_eq_pi_div_three_l199_199251

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199251


namespace part1_part2_l199_199808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

theorem part1 (a : ℝ) : 
  (∀ x, Derivative (f a) 0 = Derivative g 1) → a = 1 := 
by 
  sorry

theorem part2 :
  (∀ x, Derivative (f 1) 0 = Derivative g 1) → 
  (∀ x > 0, h 1 x > 2) :=
by 
  sorry

end part1_part2_l199_199808


namespace large_monkey_doll_cost_l199_199666

theorem large_monkey_doll_cost :
  ∃ (L : ℝ), (300 / L - 300 / (L - 2) = 25) ∧ L > 0 := by
  sorry

end large_monkey_doll_cost_l199_199666


namespace calculate_change_l199_199183

def price_cappuccino := 2
def price_iced_tea := 3
def price_cafe_latte := 1.5
def price_espresso := 1

def quantity_cappuccinos := 3
def quantity_iced_teas := 2
def quantity_cafe_lattes := 2
def quantity_espressos := 2

def amount_paid := 20

theorem calculate_change :
  let total_cost := (price_cappuccino * quantity_cappuccinos) + 
                    (price_iced_tea * quantity_iced_teas) + 
                    (price_cafe_latte * quantity_cafe_lattes) + 
                    (price_espresso * quantity_espressos) in
  amount_paid - total_cost = 3 :=
by
  sorry

end calculate_change_l199_199183


namespace replace_asterisk_with_2x_l199_199542

-- Defining conditions for Lean
def expr_with_monomial (a : ℤ) : ℤ := (x : ℤ) → (x^3 - 2)^2 + (x^2 + a * x)^2

-- The statement of the proof in Lean 4
theorem replace_asterisk_with_2x : expr_with_monomial 2 = (x : ℤ) → x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_asterisk_with_2x_l199_199542


namespace find_f_3_l199_199398

def f : ℤ → ℤ
| x := if x >= 6 then x - 5 else f (x + 2)

theorem find_f_3 : f 3 = 2 := by
  sorry

end find_f_3_l199_199398


namespace ladder_height_l199_199675

theorem ladder_height (c b : ℕ) (hc : c = 13) (hb : b = 5) : ∃ a : ℕ, a = 12 :=
by {
  let a := 12,
  have h : c^2 = a^2 + b^2 := by sorry,
  have hc' : 13^2 = a^2 + 5^2 := by simp [h, hc, hb],
  have ha : a^2 = 144 := by {
    simp only [pow_two, nat.pow, Nat.mul] at hc',
    sorry -- complete the proof here
  },
  have hr : a = 12 := by rw nat.sqrt_eq_iff² at ha,
  use a,
  exact hr
}

end ladder_height_l199_199675


namespace arccos_half_is_pi_div_three_l199_199256

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199256


namespace find_lesser_number_l199_199984

theorem find_lesser_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by sorry

end find_lesser_number_l199_199984


namespace correct_propositions_l199_199774

variables {α β : Type} [plane α] [plane β] 
variables {m n : Type} [line m] [line n]

-- Proposition 2
def prop2 (h₁ : m ⟂ α) (h₂ : m ⟂ β) : α ∥ β := sorry

-- Proposition 3
def prop3 (h₁ : α ∩ β = n) (h₂ : m ∥ α) (h₃ : m ∥ β) : m ∥ n := sorry

-- Proposition 4
def prop4 (h₁ : α ⟂ β) (h₂ : m ⟂ α) (h₃ : n ⟂ β) : m ⟂ n := sorry

theorem correct_propositions : prop2 ∧ prop3 ∧ prop4 := 
by
  exact and.intro prop2 (and.intro prop3 prop4)

end correct_propositions_l199_199774


namespace final_number_blackboard_l199_199970

theorem final_number_blackboard :
  (∏ i in Finset.range 2010 | i > 0, (1 + (1 / i: ℝ))) - 1 = 2010 :=
by
  sorry

end final_number_blackboard_l199_199970


namespace sum_m_n_l199_199045

-- We define the conditions and problem
variables (m n : ℕ)

-- Conditions
def conditions := m > 50 ∧ n > 50 ∧ Nat.lcm m n = 480 ∧ Nat.gcd m n = 12

-- Statement to prove
theorem sum_m_n : conditions m n → m + n = 156 := by sorry

end sum_m_n_l199_199045


namespace prob_indep_event_intersection_l199_199667

variable {α : Type}  -- General type for our sample space
variable [probability_space α]  -- Assume α is a probability space

variable (A B : set α)  -- Events A and B

-- Assume the probabilities of A and B and their independence
variable (hA : probability A = 2/5)
variable (hB : probability B = 2/5)
variable (h_indep : indep_events A B)

-- Prove the probability of A intersect B
theorem prob_indep_event_intersection :
  probability (A ∩ B) = 4/25 :=
by sorry

end prob_indep_event_intersection_l199_199667


namespace house_construction_count_l199_199928

/-- Define the condition of having six neighboring houses that can be made of brick or wood,
but no two wooden houses can be adjacent. -/
def valid_house_construction (houses : List Char) : Prop :=
  houses.length = 6 ∧
  ∀ i, (i < 5) → houses[i] = 'B' ∨ houses[i] = 'W' → houses[i + 1] ≠ 'W'

/-- The main theorem: There are exactly 21 valid constructions of six houses where no two wooden houses are adjacent. -/
theorem house_construction_count :  
  (List.filter valid_house_construction (List.replicate 6 ['B', 'W'])).length = 21 :=
sorry

end house_construction_count_l199_199928


namespace advanced_moves_equal_twice_l199_199384

theorem advanced_moves_equal_twice (n : ℕ) (h1 : n ≥ 4) : 
  let a : ℕ → ℕ := λ n, -- placeholder for the actual counting function
    sorry -- since we omit solution steps, we do not define the function here
  in a (n-1) + a n = 2^n :=
by {
  -- We assume a counts the number of valid ways to advance twice around the circle
  sorry
}

end advanced_moves_equal_twice_l199_199384


namespace area_RGJHK_l199_199188

noncomputable def shaded_region_area (radius PR RG PQ distance: ℝ) : ℝ :=
  let rect_width := PQ / 2 * √ 3
  let rect_height := RG
  let rect_area := rect_width * rect_height
  let triangle_area := (1 / 2) * radius * RG
  let sector_area := (1 / 8) * 9 * Real.pi
  let triangles_and_sectors_area := 2 * (triangle_area + sector_area)
  rect_area - triangles_and_sectors_area

theorem area_RGJHK :
  let radius := 3
  let PR := 3 * √ 3
  let RG := √ ((3 * √ 3) ^ 2 - 3 ^ 2)
  let PQ := 2 * 3 * √ 3
  shaded_region_area radius PR RG PQ 18 = 18 * √ 6 - 9 * √ 2 - (9 * Real.pi / 4) :=
by
  let radius := 3
  let PR := 3 * √ 3
  let RG := 3 * √ 2
  let PQ := 6 * √ 3
  sorry

end area_RGJHK_l199_199188


namespace replacement_times_l199_199391

noncomputable def find_replacement_times (V : ℝ) (n : ℕ) : ℝ :=
  let final_milk_volume := 0.8^n * V
  in final_milk_volume

theorem replacement_times (V : ℝ) : ∃ n : ℕ, find_replacement_times V n = 0.512 * V :=
by
  use 3
  sorry

end replacement_times_l199_199391


namespace books_distribution_l199_199134

noncomputable def distribution_ways : ℕ :=
  let books := 5
  let people := 4
  let combination := Nat.choose books 2
  let arrangement := Nat.factorial people
  combination * arrangement ^ people

theorem books_distribution : distribution_ways = 240 := by
  sorry

end books_distribution_l199_199134


namespace arccos_half_eq_pi_div_three_l199_199333

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199333


namespace problem_solution_l199_199049

-- Define the condition that specifies the rule for the changes in decimal points
def decimal_point_rule  (n : ℝ) (d : ℕ) : Prop :=
  sqrt(n * (10 : ℝ) ^ d) = sqrt(n) * (10 : ℝ) ^ (d / 2)

-- State the main theorem encapsulating all the questions
theorem problem_solution :
  decimal_point_rule 0.0625 2 ∧ 
  decimal_point_rule 0.625 2 ∧ 
  decimal_point_rule 6.25 2 ∧ 
  decimal_point_rule 62.5 2 ∧ 
  decimal_point_rule 625 2 ∧ 
  decimal_point_rule 6250 2 ∧ 
  decimal_point_rule 62500 2 ∧
  -- ∀ n, decimal_point_rule n (2 : ℕ) -- General rule for all n
  -- Specific cases for the given values:
  sqrt 0.03 ≈ 0.1732 ∧
  sqrt 300 ≈ 17.32 ∧ 
  ¬(exists x, sqrt 30 = x * sqrt 3) := 
by 
  apply And.intro; try {
    exact sorry
  };
  apply And.intro; try {
    exact sorry
  };
  apply And.intro; try {
    exact sorry
  };
  exact sorry
  ;
  apply And.intro; try {
    exact sorry
  };
  apply And.intro; try {
    exact sorry
  };
  apply And.intro; try {
    exact sorry
  };
  apply And.intro; try {
    exact sorry
  };
  exact sorry
  ;
  apply And.intro; try {
    exact sorry
  };
  exact sorry

end problem_solution_l199_199049


namespace min_value_of_f_l199_199966

noncomputable def f (x : ℝ) : ℝ :=
  sin (x - π / 2) * cos x - (cos (2 * x)) ^ 2

theorem min_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = -2 :=
by
  sorry

end min_value_of_f_l199_199966


namespace replace_star_with_2x_l199_199550

theorem replace_star_with_2x (c : ℤ) (x : ℤ) :
  (x^3 - 2)^2 + (x^2 + c)^2 = x^6 + x^4 + 4x^2 + 4 ↔ c = 2 * x :=
by
  sorry

end replace_star_with_2x_l199_199550


namespace minimum_photos_needed_l199_199873

theorem minimum_photos_needed 
  (total_photos : ℕ) 
  (photos_IV : ℕ)
  (photos_V : ℕ) 
  (photos_VI : ℕ) 
  (photos_VII : ℕ) 
  (photos_I_III : ℕ) 
  (H : total_photos = 130)
  (H_IV : photos_IV = 35)
  (H_V : photos_V = 30)
  (H_VI : photos_VI = 25)
  (H_VII : photos_VII = 20)
  (H_I_III : photos_I_III = total_photos - (photos_IV + photos_V + photos_VI + photos_VII)) :
  77 = 77 :=
by
  sorry

end minimum_photos_needed_l199_199873


namespace correct_conclusion_l199_199657

noncomputable def algebraic_expression_1 := ∀ (x : ℝ), π * x^2 + 4 * x - 3
noncomputable def algebraic_expression_2 := ∀ (x y : ℝ), 3 * x^2 * y = -2 * x * y^2
noncomputable def algebraic_expression_3 := ∀ (x : ℝ), x^2 + 4 * x - 3
noncomputable def monomial := ∀ (x y : ℝ), -3 * x^2 * y / 5

theorem correct_conclusion :
    (¬ algebraic_expression_1 ∧ ¬ algebraic_expression_2 ∧ ¬ algebraic_expression_3 ∧
    monomial = -3/5 * x^2 * y ∧ ∃ d : ℕ, d = 3) :=
begin
  sorry
end

end correct_conclusion_l199_199657


namespace arccos_half_eq_pi_div_3_l199_199280

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199280


namespace total_sunglasses_count_l199_199039

/-- 
Given:
1. There are 3000 adults consisting of men and women on the ocean liner.
2. 55% of the adults are men.
3. 12% of the women are wearing sunglasses.
4. 15% of the men are wearing sunglasses.

Prove that the total number of adults wearing sunglasses is equal to 409.
-/
theorem total_sunglasses_count :
  let (total_adults men women : ℕ) := (3000, 1650, 1350) in
  let total_adults_sunglasses := (0.12 * women + 0.15 * men).to_nat in
  total_adults_sunglasses = 409 :=
by
  let total_adults := 3000
  let men := (0.55 * total_adults).to_nat
  let women := total_adults - men
  let women_sunglasses := (0.12 * women).to_nat
  let men_sunglasses := (0.15 * men).to_nat
  let total_adults_sunglasses := women_sunglasses + men_sunglasses
  have h : total_adults_sunglasses = 409 := sorry
  exact h

end total_sunglasses_count_l199_199039


namespace arccos_half_eq_pi_div_three_l199_199286

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199286


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199330

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199330


namespace calc_1_calc_2_calc_3_calc_4_l199_199181

section
variables {m n x y z : ℕ} -- assuming all variables are natural numbers for simplicity.
-- Problem 1
theorem calc_1 : (2 * m * n) / (3 * m ^ 2) * (6 * m * n) / (5 * n) = (4 * n) / 5 :=
sorry

-- Problem 2
theorem calc_2 : (5 * x - 5 * y) / (3 * x ^ 2 * y) * (9 * x * y ^ 2) / (x ^ 2 - y ^ 2) = 
  15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem calc_3 : ((x ^ 3 * y ^ 2) / z) ^ 2 * ((y * z) / x ^ 2) ^ 3 = y ^ 7 * z :=
sorry

-- Problem 4
theorem calc_4 : (4 * x ^ 2 * y ^ 2) / (2 * x + y) * (4 * x ^ 2 + 4 * x * y + y ^ 2) / (2 * x + y) / 
  ((2 * x * y) * (2 * x - y) / (4 * x ^ 2 - y ^ 2)) = 4 * x ^ 2 * y + 2 * x * y ^ 2 :=
sorry
end

end calc_1_calc_2_calc_3_calc_4_l199_199181


namespace arccos_of_half_eq_pi_over_three_l199_199294

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199294


namespace skating_speeds_ratio_l199_199686

theorem skating_speeds_ratio (v_s v_f : ℝ) (h1 : v_f > v_s) (h2 : |v_f + v_s| / |v_f - v_s| = 5) :
  v_f / v_s = 3 / 2 :=
by
  sorry

end skating_speeds_ratio_l199_199686


namespace wax_he_has_l199_199037

def total_wax : ℕ := 353
def additional_wax : ℕ := 22

theorem wax_he_has : total_wax - additional_wax = 331 := by
  sorry

end wax_he_has_l199_199037


namespace arccos_one_half_l199_199217

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199217


namespace radius_of_sector_l199_199574

theorem radius_of_sector
  (A : ℝ) (θ : ℝ)
  (hA : A = 118.8)
  (hθ : θ = 42)
  : ∃ r : ℝ, 118.8 = (42 / 360) * Real.pi * r^2 ∧ r ≈ 17.99 :=
by
  sorry

end radius_of_sector_l199_199574


namespace num_int_pairs_l199_199762

theorem num_int_pairs :
  {p : ℕ × ℕ // 
    let m := p.1, n := p.2 in
    1 ≤ m ∧ m ≤ 99 ∧ 1 ≤ n ∧ n ≤ 99 ∧ ∃ k : ℕ, (m + n)^2 + 3 * m + n = k^2}.card = 98 :=
by sorry

end num_int_pairs_l199_199762


namespace arccos_one_half_l199_199218

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199218


namespace prime_factors_of_2008006_l199_199847

theorem prime_factors_of_2008006 : set.finite {p : ℕ | nat.prime p ∧ p ∣ 2008006} ∧ set.card {p : ℕ | nat.prime p ∧ p ∣ 2008006} = 6 :=
by
  sorry

end prime_factors_of_2008006_l199_199847


namespace simplify_expression_l199_199947

theorem simplify_expression (x : ℝ) : 
  x^2 * (4 * x^3 - 3 * x + 1) - 6 * (x^3 - 3 * x^2 + 4 * x - 5) = 
  4 * x^5 - 9 * x^3 + 19 * x^2 - 24 * x + 30 := by
  sorry

end simplify_expression_l199_199947


namespace knights_not_less_than_liars_l199_199929

theorem knights_not_less_than_liars
  (p : Type) -- the type of people
  (liar knight : p → Prop)
  (acquainted : p → p → Prop)
  (H_exclusive : ∀ x, liar x ∨ knight x ∧ ¬ (liar x ∧ knight x))
  (H_liar : ∀ x, liar x → (∀ y, acquainted x y ↔ (acquainted x y ∧ ∀ z, acquainted y z)))
  (H_knight : ∀ x, knight x → (∀ y, acquainted x y → ∀ z, acquainted y z))
  (H_statements : ∀ x, (∀ y, acquainted x y) → ((∃ L K, liar x ∧ knight y) → true)) :
  ∀ L K, (∀ x, (liar x ∨ knight x) ∧ (liar x → ∃ y, acquainted x y) ∧ (liar x ∧ ∀ y, liar y → acquainted x y)) → K ≥ L :=
by
  sorry

end knights_not_less_than_liars_l199_199929


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199329

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199329


namespace arccos_one_half_l199_199228

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199228


namespace decrease_in_average_age_l199_199953

theorem decrease_in_average_age (original_avg_age : ℕ) (new_students_avg_age : ℕ) 
    (original_strength : ℕ) (new_students_strength : ℕ) 
    (h1 : original_avg_age = 40) (h2 : new_students_avg_age = 32) 
    (h3 : original_strength = 8) (h4 : new_students_strength = 8) : 
    (original_avg_age - ((original_strength * original_avg_age + new_students_strength * new_students_avg_age) / (original_strength + new_students_strength))) = 4 :=
by 
  sorry

end decrease_in_average_age_l199_199953


namespace find_d_l199_199582

open Complex

def g (z : ℂ) : ℂ := ((1 - I * real.sqrt 3) * z + (2 * real.sqrt 3 + 12 * I)) / 3

theorem find_d :
  ∃ d : ℂ, g d = d ∧ d = ((16 * real.sqrt 3) / 7) + ((18 : ℚ) / 7) * I :=
by
  sorry

end find_d_l199_199582


namespace symmetry_axis_l199_199347

def determinant_2x2 (a1 a2 a3 a4 : ℝ) := a1 * a4 - a2 * a3

noncomputable def f (x : ℝ) : ℝ :=
  determinant_2x2 (sin (2 * x)) 1 (cos (2 * x)) (sqrt 3)

noncomputable def f_shifted (x : ℝ) : ℝ :=
  f (x - π / 3)

theorem symmetry_axis : ∃ (x : ℝ), f_shifted (x) = f (π / 6) :=
by
  sorry

end symmetry_axis_l199_199347


namespace johns_average_speed_l199_199893

theorem johns_average_speed (driving_time_min scooter_time_min : ℕ) 
  (driving_speed_mph scooter_speed_mph : ℕ) : 
  driving_time_min = 45 → 
  driving_speed_mph = 30 → 
  scooter_time_min = 60 → 
  scooter_speed_mph = 10 → 
  let total_distance := (driving_speed_mph * driving_time_min / 60) + (scooter_speed_mph * scooter_time_min / 60) in
  let total_time := (driving_time_min / 60 : ℝ) + (scooter_time_min / 60 : ℝ) in
  let average_speed := total_distance / total_time in
  average_speed ≈ 18.57 := 
by
  sorry

end johns_average_speed_l199_199893


namespace sum_of_divisors_252_l199_199646

theorem sum_of_divisors_252 :
  let n := 252
  let prime_factors := [2, 2, 3, 3, 7]
  sum_of_divisors n = 728 :=
by
  sorry

end sum_of_divisors_252_l199_199646


namespace arccos_half_is_pi_div_three_l199_199257

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199257


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199325

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199325


namespace midpoint_line_parallel_bisector_l199_199930

-- We define the points and distances:
variables {X O Y A B C D : Type}
variables [MetricSpace O]

-- Conditions given in the problem:
variables (OX OY : Line O) 
variables (A B : Point OX) (C D : Point OY)
variables (AB_eq_CD : distance A B = distance C D)

-- Define the bisector of the angle XOY:
noncomputable def bisector (X O Y : Type) [MetricSpace O] : Line O := sorry

-- Midpoints of segments AC and BD:
noncomputable def midpoint (P Q : Type) [MetricSpace P] : Point Q := sorry

-- The goal: the line connecting the midpoints of AC and BD is parallel to the bisector of the angle XOY.
theorem midpoint_line_parallel_bisector (X O Y : Type) [MetricSpace O] (A B C D : Point O) 
  (H1 : OX OY A B C D : Type) (H2 : distance A B = distance C D) 
  (mid_AC := midpoint A C) (mid_BD := midpoint B D) 
  (bisec_XOY := bisector X O Y) :
  is_parallel_to (Line.join mid_AC mid_BD) bisec_XOY := 
sorry


end midpoint_line_parallel_bisector_l199_199930


namespace replace_asterisk_with_2x_l199_199535

theorem replace_asterisk_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by sorry

end replace_asterisk_with_2x_l199_199535


namespace arccos_one_half_l199_199227

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199227


namespace angles_parallel_sides_l199_199858

theorem angles_parallel_sides
  (α β : Angle)
  (h1 : α = β)
  (sides1_parallel : ∃ l1 l2 : Line, are_parallel l1 l2 ∧ corresponding_sides_of α β l1 l2) :
  ¬(∃ l3 l4 : Line, are_parallel l3 l4 ∨ intersect l3 l4 ∨ skew l3 l4) :=
sorry

end angles_parallel_sides_l199_199858


namespace surface_area_unchanged_l199_199099

theorem surface_area_unchanged (volume_of_small_cube : ℕ) (side_length : ℕ) (surface_area : ℕ) :
  volume_of_small_cube = 1 ∧ side_length = 4 ∧ surface_area = 96 →
  surface_area = 96 :=
by
  intros h,
  cases h,
  sorry

end surface_area_unchanged_l199_199099


namespace largest_odd_integer_sum_squares_l199_199983

/-- The sum of the squares of 50 consecutive odd integers is 300850. Prove that the largest odd integer whose square is the last term of this sum is 121. -/
theorem largest_odd_integer_sum_squares (sum_squares_eq : ∑ k in (finset.range 50), (2 * k + X)^2 = 300850)
: X + 98 = 121 :=
sorry

end largest_odd_integer_sum_squares_l199_199983


namespace derivative_expression_at_one_l199_199416

theorem derivative_expression_at_one :
  (∃ (f : ℝ → ℝ), f = (λ x, (x - 1) ^ 2 + 3 * (x - 1)) ∧ deriv f 1 = 3) :=
sorry

end derivative_expression_at_one_l199_199416


namespace point_in_second_quadrant_l199_199875

-- Define points in the Cartesian coordinate system
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the quadrants
inductive Quadrant
| First
| Second
| Third
| Fourth
  
-- Function to determine the quadrant of a point
def pointQuadrant (P : Point) : Quadrant :=
  if P.x > 0 ∧ P.y > 0 then Quadrant.First
  else if P.x < 0 ∧ P.y > 0 then Quadrant.Second
  else if P.x < 0 ∧ P.y < 0 then Quadrant.Third
  else Quadrant.Fourth

-- The point in question
def P : Point := { x := -2, y := 3 }

-- The proof problem
theorem point_in_second_quadrant : pointQuadrant P = Quadrant.Second := by 
  simp [pointQuadrant, P]
  sorry

end point_in_second_quadrant_l199_199875


namespace inequality_proof_l199_199510

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)^2 + bc / (b + c)^2 + ca / (c + a)^2) + (3 * (a^2 + b^2 + c^2)) / (a + b + c)^2 ≥ 7 / 4 := 
by
  sorry

end inequality_proof_l199_199510


namespace replace_star_with_2x_l199_199549

theorem replace_star_with_2x (c : ℤ) (x : ℤ) :
  (x^3 - 2)^2 + (x^2 + c)^2 = x^6 + x^4 + 4x^2 + 4 ↔ c = 2 * x :=
by
  sorry

end replace_star_with_2x_l199_199549


namespace hexagon_area_error_l199_199719

theorem hexagon_area_error (s : ℝ) :
  let A := (3 * Real.sqrt 3 / 2) * s^2 in
  let s' := 1.08 * s in
  let A' := (3 * Real.sqrt 3 / 2) * (s')^2 in
  (A' - A) / A * 100 = 16.64 :=
by
  let A := (3 * Real.sqrt 3 / 2) * s^2
  let s' := 1.08 * s
  let A' := (3 * Real.sqrt 3 / 2) * (s')^2
  sorry

end hexagon_area_error_l199_199719


namespace delores_initial_money_l199_199350

-- Definitions and conditions based on the given problem
def original_computer_price : ℝ := 400
def original_printer_price : ℝ := 40
def original_headphones_price : ℝ := 60

def computer_discount : ℝ := 0.10
def computer_tax : ℝ := 0.08
def printer_tax : ℝ := 0.05
def headphones_tax : ℝ := 0.06

def leftover_money : ℝ := 10

-- Final proof problem statement
theorem delores_initial_money :
  original_computer_price * (1 - computer_discount) * (1 + computer_tax) +
  original_printer_price * (1 + printer_tax) +
  original_headphones_price * (1 + headphones_tax) + leftover_money = 504.40 := by
  sorry -- Proof is not required

end delores_initial_money_l199_199350


namespace return_path_length_l199_199033

variables (C_lower C_upper : ℝ) (theta : ℝ)
def R := C_lower / (2 * Real.pi)
def r := C_upper / (2 * Real.pi)
def l := R / Real.cos theta
def h := l * Real.sin theta

theorem return_path_length (hC_lower : C_lower = 8) (hC_upper : C_upper = 6) (htheta : theta = Real.pi / 3) :
  h = 4 * Real.sqrt 3 / Real.pi :=
by
  rw [hC_lower, hC_upper, htheta]
  dsimp [R, r, l, h]
  sorry

end return_path_length_l199_199033


namespace dodecagon_diagonals_l199_199683

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem dodecagon_diagonals : num_diagonals 12 = 54 :=
by
  -- by sorry means we skip the actual proof
  sorry

end dodecagon_diagonals_l199_199683


namespace no_real_solution_for_ap_l199_199357

theorem no_real_solution_for_ap : 
  (¬∃ (a b : ℝ), 15, a, b, a * b form_an_arithmetic_progression) :=
sorry

end no_real_solution_for_ap_l199_199357


namespace triangle_inequality_l199_199959

theorem triangle_inequality (a b c : ℕ) : a + b > c ∧ a + c > b ∧ b + c > a ↔ true := sorry

example (a b c : ℕ) (h : a = 13 ∧ b = 12 ∧ c = 20) : a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rw [← h.1, ← h.2.1, ← h.2.2]
  exact triangle_inequality 13 12 20

end triangle_inequality_l199_199959


namespace a_2017_value_l199_199078

noncomputable def sequence (a : ℕ → ℚ) : ℕ → ℚ
| 0       := a 0
| (n + 1) := if 0 ≤ a n ∧ a n ≤ (1/2 : ℚ) then 2 * a n else 2 * a n - 1

def a_2017 := sequence (λ _ => 6/7) 2016
-- Prove a_2017 = 6 / 7
theorem a_2017_value : a_2017 = 6 / 7 := sorry

end a_2017_value_l199_199078


namespace arccos_half_eq_pi_div_3_l199_199276

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199276


namespace problem_f_17_l199_199725

/-- Assume that f(1) = 0 and f(m + n) = f(m) + f(n) + 4 * (9 * m * n - 1) for all natural numbers m and n.
    Prove that f(17) = 4832.
-/
theorem problem_f_17 (f : ℕ → ℤ) 
  (h1 : f 1 = 0) 
  (h_func : ∀ m n : ℕ, f (m + n) = f m + f n + 4 * (9 * m * n - 1)) 
  : f 17 = 4832 := 
sorry

end problem_f_17_l199_199725


namespace drug_price_reduction_l199_199093

theorem drug_price_reduction :
  ∃ x : ℝ, 56 * (1 - x)^2 = 31.5 :=
by
  sorry

end drug_price_reduction_l199_199093


namespace arccos_half_eq_pi_div_three_l199_199336

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199336


namespace correlation_problem_l199_199419

-- Given: y_hat = -0.1 * z + 1 and y negatively correlated with x
variables {x y z : ℝ}
axiom correlation : ∀ {a b : ℝ}, correlated_neg a b ↔ correlated_neg b a
def y_hat (z : ℝ) : ℝ := -0.1 * z + 1

-- Condition: y is negatively correlated with x
axiom neg_corr_y_x : correlated_neg y x

-- Condition: y is negatively correlated with z from y_hat equation
def neg_corr_y_z : correlated_neg (y_hat z) z := sorry

theorem correlation_problem : (correlated_neg y x) ∧ (correlated_neg (y_hat z) z) →
                            (correlated_neg x y) ∧ (correlated_pos x z) :=
begin
  sorry
end

end correlation_problem_l199_199419


namespace replace_star_with_2x_l199_199553

theorem replace_star_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_star_with_2x_l199_199553


namespace girl_scouts_with_permission_slip_l199_199664

theorem girl_scouts_with_permission_slip (total_scouts boy_scouts girl_scouts signed_permission_slips boy_scouts_with_slips girl_scouts_with_slips : ℕ)
  (h1 : total_scouts = 100)
  (h2 : boy_scouts = 40)
  (h3 : girl_scouts = total_scouts - boy_scouts)
  (h4 : signed_permission_slips = 80)
  (h5 : boy_scouts_with_slips = 30)
  (h6 : girl_scouts_with_slips = signed_permission_slips - boy_scouts_with_slips) :
  (girl_scouts_with_slips * 100 / girl_scouts)%ℕ = 83 :=
by
  sorry

end girl_scouts_with_permission_slip_l199_199664


namespace ratio_QZ_ZX_l199_199460

def triangle_ratio (XY YZ : ℝ) : Prop :=
  XY / YZ = 4 / 5

def bisector_intersect (c d : ℝ) (YQ : ℝ) (angle_bisector : ℝ) (YZ : ℝ) (ZX QZ : ℝ) : Prop :=
  angle_bisector = (QZ / ZX) ↔ (XY / YZ) = 4 / 5

theorem ratio_QZ_ZX (XY YZ ZX QZ : ℝ) (h1 : triangle_ratio XY YZ)
  (h2 : bisector_intersect XY YZ ZX QZ) : QZ / ZX = 9 / 4 :=
sorry

end ratio_QZ_ZX_l199_199460


namespace sun_radius_scientific_notation_l199_199077

theorem sun_radius_scientific_notation : 
  (369000 : ℝ) = 3.69 * 10^5 :=
by
  sorry

end sun_radius_scientific_notation_l199_199077


namespace circle_properties_l199_199681

-- Define the points A and B
def A : Point := ⟨4, 1⟩
def B : Point := ⟨2, 1⟩

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the circle equation
def circle_eqn (h k r x y : ℝ) : Prop := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Given conditions:
variable (C_passes_through_A : circle_eqn 3 0 (sqrt 2) 4 1)
variable (C_tangent_to_line_B : tangent_line 2 1)

-- Proof statement
theorem circle_properties : 
  ∃ h k r, circle_eqn h k r x y ∧ h = 3 ∧ k = 0 ∧ r = sqrt 2 :=
sorry

end circle_properties_l199_199681


namespace greatest_multiple_of_5_and_7_less_than_1000_l199_199630

theorem greatest_multiple_of_5_and_7_less_than_1000 : ∃ x : ℕ, (x % 5 = 0) ∧ (x % 7 = 0) ∧ (x < 1000) ∧ (∀ y : ℕ, (y % 5 = 0) ∧ (y % 7 = 0) ∧ (y < 1000) → y ≤ x) ∧ x = 980 :=
begin
  sorry
end

end greatest_multiple_of_5_and_7_less_than_1000_l199_199630


namespace incorrect_statement_C_l199_199659

-- Define the parabola function
def parabola (x : ℝ) : ℝ := -x^2 + 3

-- State the theorem that statement C is incorrect
theorem incorrect_statement_C : ∀ x > 0, parabola x > parabola (x + 1) :=
by { intros x hx, sorry }

end incorrect_statement_C_l199_199659


namespace cone_volume_ratio_l199_199621

noncomputable def ratio_of_volumes (l : ℝ) (r : ℝ) (R : ℝ) (h : ℝ) (H : ℝ) : ℝ :=
  (1/3) * real.pi * r^2 * h / ((1/3) * real.pi * R^2 * H)

theorem cone_volume_ratio (l r R h H : ℝ) (h_eq : h = real.sqrt (l^2 - r^2))
  (H_eq : H = real.sqrt (l^2 - R^2)) (R_eq : R = 2 * r) (l_eq : l = 3 * r) :
  ratio_of_volumes l r R h H = 1 / real.sqrt 10 := 
  sorry

end cone_volume_ratio_l199_199621


namespace find_m_range_l199_199515

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - (1 / 2) * x^2 - 2 * x + 5

-- Statement of the problem
theorem find_m_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 2 → f x < m) ↔ m ∈ set.Ioi 7 :=
sorry

end find_m_range_l199_199515


namespace sum_reciprocal_of_shifted_roots_l199_199505

noncomputable def roots_of_cubic (a b c : ℝ) : Prop := 
    ∀ x : ℝ, x^3 - x - 2 = (x - a) * (x - b) * (x - c)

theorem sum_reciprocal_of_shifted_roots (a b c : ℝ) 
    (h : roots_of_cubic a b c) : 
    (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) = 1 :=
by
  sorry

end sum_reciprocal_of_shifted_roots_l199_199505


namespace replace_asterisk_with_2x_l199_199543

-- Defining conditions for Lean
def expr_with_monomial (a : ℤ) : ℤ := (x : ℤ) → (x^3 - 2)^2 + (x^2 + a * x)^2

-- The statement of the proof in Lean 4
theorem replace_asterisk_with_2x : expr_with_monomial 2 = (x : ℤ) → x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_asterisk_with_2x_l199_199543


namespace type_1_all_colors_type_2_not_all_colors_l199_199058

/-- Define the coloring function based on coordinates (x, y) -/
def color (x y : ℤ) : ℤ := (x + 2 * y) % 5

/-- Define what figure type 1 is -/
def figure_type_1 (x y : ℤ) : set (ℤ × ℤ) := 
  {(x, y), (x+1, y), (x+2, y), (x+1, y+1), (x+1, y-1)}

/-- Definition that checks if a set of coordinates covers all 5 colors -/
def covers_all_colors (fig : set (ℤ × ℤ)) : Prop :=
  {color (fst p) (snd p) | p ∈ fig} = {0, 1, 2, 3, 4}

/-- Define what figure type 2 is -/
def figure_type_2 (fig : set (ℤ × ℤ)) : Prop := 
  ¬ covers_all_colors fig

/-- Prove that in any figure of type 1, there will be cells of all five colors. -/
theorem type_1_all_colors (x y : ℤ) :
  covers_all_colors (figure_type_1 x y) :=
  sorry

/-- Theorem that figure type 2 do not necessarily cover all five colors. -/
theorem type_2_not_all_colors {fig : set (ℤ × ℤ)} :
  figure_type_2 fig → ¬ covers_all_colors fig :=
  sorry

end type_1_all_colors_type_2_not_all_colors_l199_199058


namespace bob_shucks_240_oysters_in_2_hours_l199_199175

-- Definitions based on conditions provided:
def oysters_per_minute (oysters : ℕ) (minutes : ℕ) : ℕ :=
  oysters / minutes

def minutes_in_hour : ℕ :=
  60

def oysters_in_two_hours (oysters_per_minute : ℕ) (hours : ℕ) : ℕ :=
  oysters_per_minute * (hours * minutes_in_hour)

-- Parameters given in the problem:
def initial_oysters : ℕ := 10
def initial_minutes : ℕ := 5
def hours : ℕ := 2

-- The main theorem we need to prove:
theorem bob_shucks_240_oysters_in_2_hours :
  oysters_in_two_hours (oysters_per_minute initial_oysters initial_minutes) hours = 240 :=
by
  sorry

end bob_shucks_240_oysters_in_2_hours_l199_199175


namespace largest_value_l199_199909

noncomputable def y : ℝ := 10 ^ (-2024)

theorem largest_value : ∀ x : ℝ, x ∈ {5 + y, 5 - y, 5 * y, 5 / y, y / 5} → x ≤ 5 / y :=
by
  sorry

end largest_value_l199_199909


namespace arccos_one_half_l199_199236

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199236


namespace money_made_from_milk_sales_l199_199729

namespace BillMilkProblem

def total_gallons_milk : ℕ := 16
def fraction_for_sour_cream : ℚ := 1 / 4
def fraction_for_butter : ℚ := 1 / 4
def milk_to_sour_cream_ratio : ℕ := 2
def milk_to_butter_ratio : ℕ := 4
def price_per_gallon_butter : ℕ := 5
def price_per_gallon_sour_cream : ℕ := 6
def price_per_gallon_whole_milk : ℕ := 3

theorem money_made_from_milk_sales : ℕ :=
  let milk_for_sour_cream := (fraction_for_sour_cream * total_gallons_milk).toNat
  let milk_for_butter := (fraction_for_butter * total_gallons_milk).toNat
  let sour_cream_gallons := milk_for_sour_cream / milk_to_sour_cream_ratio
  let butter_gallons := milk_for_butter / milk_to_butter_ratio
  let milk_remaining := total_gallons_milk - milk_for_sour_cream - milk_for_butter
  let money_from_butter := butter_gallons * price_per_gallon_butter
  let money_from_sour_cream := sour_cream_gallons * price_per_gallon_sour_cream
  let money_from_whole_milk := milk_remaining * price_per_gallon_whole_milk
  money_from_butter + money_from_sour_cream + money_from_whole_milk = 41 :=
by
  sorry 

end BillMilkProblem

end money_made_from_milk_sales_l199_199729


namespace triangle_height_le_half_l199_199599

theorem triangle_height_le_half (a : ℝ) : 
  let s1 := Real.sqrt (a^2 - a + 1),
      s2 := Real.sqrt (a^2 + a + 1),
      s3 := Real.sqrt (4 * a^2 + 3) in
  s2 * s2 + (a - a^2 - 1) ^ 2 ≤ (1 / 4) * 4 :=
  sorry

end triangle_height_le_half_l199_199599


namespace max_f_eq_find_a_l199_199439

open Real

noncomputable def f (α : ℝ) : ℝ :=
  let a := (sin α, cos α)
  let b := (6 * sin α + cos α, 7 * sin α - 2 * cos α)
  a.1 * b.1 + a.2 * b.2

theorem max_f_eq : 
  ∃ α : ℝ, f α = 4 * sqrt 2 + 2 :=
sorry

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_A : ℝ)

noncomputable def f_triangle (A : ℝ) : ℝ :=
  let a := (sin A, cos A)
  let b := (6 * sin A + cos A, 7 * sin A - 2 * cos A)
  a.1 * b.1 + a.2 * b.2

axiom f_A_eq (A : ℝ) : f_triangle A = 6

theorem find_a (A B C a b c : ℝ) (h₁ : f_triangle A = 6) (h₂ : 1 / 2 * b * c * sin A = 3) (h₃ : b + c = 2 + 3 * sqrt 2) :
  a = sqrt 10 :=
sorry

end max_f_eq_find_a_l199_199439


namespace arccos_of_half_eq_pi_over_three_l199_199300

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199300


namespace jasmine_dinner_time_proof_l199_199490

def time_addition (hour: ℕ) (minute: ℕ) (added_minute: ℕ) : ℕ × ℕ :=
let total_minutes := minute + added_minute in
(hour + total_minutes / 60, total_minutes % 60)

def jasmine_dinner_time : Prop :=
∃ (hour minute : ℕ),
  hour = 4 ∧ 
  minute = 0 ∧ 
  let commute_time := 30 in
  let grocery_time := 30 in
  let dry_cleaning_time := 10 in
  let dog_grooming_time := 20 in
  let cooking_time := 90 in
  let total_time := commute_time + grocery_time + dry_cleaning_time + dog_grooming_time + cooking_time in
  time_addition hour minute total_time = (7, 0)

theorem jasmine_dinner_time_proof : jasmine_dinner_time := 
by 
  -- to be filled with the actual proof 
  sorry

end jasmine_dinner_time_proof_l199_199490


namespace textbooks_probability_l199_199035

theorem textbooks_probability :
  let total_ways := (15.factorial / (4.factorial * 5.factorial * 6.factorial))
  let favorable_4 := 12 * (11.factorial / (5.factorial * 6.factorial))
  let favorable_5 := 66 * (10.factorial / (4.factorial * 6.factorial))
  let favorable_6 := 220 * (9.factorial / (4.factorial * 5.factorial))
  let favorable_total := favorable_4 + favorable_5 + favorable_6
  let probability := (favorable_total.toRat / total_ways.toRat)
  let reduced_probability := probability.reduce
  (reduced_probability.num = 9 ∧ reduced_probability.den = 121) →
  9 + 121 = 130 :=
by
  intro h
  sorry

end textbooks_probability_l199_199035


namespace sum_of_divisors_252_l199_199640

open BigOperators

-- Definition of the sum of divisors for a given number n
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in Nat.divisors n, d

-- Statement of the problem
theorem sum_of_divisors_252 : sum_of_divisors 252 = 728 := 
sorry

end sum_of_divisors_252_l199_199640


namespace sequence_properties_l199_199069

variables {α : Type*} [linear_ordered_field α]

def is_arithmetic_sequence (a b c : α) : Prop :=
  b - a = c - b

def is_geometric_sequence (a b c : α) : Prop :=
  b / a = c / b

noncomputable def a : ℕ → α
| 0 := 1
| 4 := 12
| 8 := 192
| _ := 0 -- Filler definition for other indices

theorem sequence_properties:
  let a := a in
  is_arithmetic_sequence a 0 a 1 a 2 ∧
  is_geometric_sequence a 2 a 3 a 4 ∧ 
  is_geometric_sequence a 4 a 5 a 6 ∧
  is_geometric_sequence a 5 a 6 a 7 ∧
  is_geometric_sequence a 6 a 7 a 8 ∧
  is_geometric_sequence a 7 a 8 a 9 →
  a 6 = 48 ∧
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8) = 384 :=
by sorry

end sequence_properties_l199_199069


namespace simplify_and_rationalize_denominator_l199_199561

theorem simplify_and_rationalize_denominator :
  (sqrt 3 / sqrt 4) * (sqrt 5 / sqrt 6) * (sqrt 7 / sqrt 8) = sqrt 70 / 8 :=
by {
  have h1 : sqrt 4 = 2 := sorry,
  have h2 : sqrt 6 = sqrt 2 * sqrt 3 := sorry,
  have h3 : sqrt 8 = 2 * sqrt 2 := sorry,
  sorry
}

end simplify_and_rationalize_denominator_l199_199561


namespace arccos_half_eq_pi_div_three_l199_199254

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199254


namespace arccos_half_eq_pi_div_3_l199_199277

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199277


namespace min_length_of_tangent_lines_l199_199973

noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def pi_over_4 : ℝ := Real.pi / 4

def circle_polar_radius (theta : ℝ) : ℝ := 2 * Real.cos (theta + pi_over_4)

def line_parametric_x (t : ℝ) : ℝ := (sqrt2 / 2) * t
def line_parametric_y (t : ℝ) : ℝ := (sqrt2 / 2) * t + 4 * sqrt2

def point_to_line_distance (px py a b c : ℝ) : ℝ := 
  ((a * px + b * py + c).abs) / Real.sqrt (a^2 + b^2)

def min_tangent_length : ℝ := 2 * Real.sqrt 6

theorem min_length_of_tangent_lines : 
    ∀ t : ℝ, 
    let x := line_parametric_x t,
        y := line_parametric_y t,
        cx := sqrt2 / 2,
        cy := -sqrt2 / 2,
        d := point_to_line_distance cx cy sqrt2 (-sqrt2) (4 * sqrt2)
    in Real.sqrt (d^2 - 1^2) = min_tangent_length := 
by 
    sorry

end min_length_of_tangent_lines_l199_199973


namespace probability_reach_corner_within_four_hops_l199_199388

/-- 
  Frieda the frog starts from the center of a 3x3 grid. 
  She can hop up, down, left, or right with equal probability. 
  If her hop would take her off the grid, she wraps around to the opposite edge.
  She stops if she lands on a corner square. What is the probability that she reaches a corner square within four hops?
-/
theorem probability_reach_corner_within_four_hops :
  -- Define the initial state and the transition probabilities
  let grid_size := 3
  let num_hops := 4
  let corner_probability : ℚ := 11 / 16
  ∀ (initial_state : ℕ × ℕ),
    initial_state = (2, 2) →
    (∑ p in all_possible_paths initial_state grid_size num_hops, reach_corner p grid_size) / (num_possible_paths initial_state grid_size num_hops) = corner_probability :=
sorry

end probability_reach_corner_within_four_hops_l199_199388


namespace tiles_needed_l199_199154

theorem tiles_needed (room_width room_length tile_width_inches tile_length_inches : ℝ)
  (hw_pos : room_width > 0) (hl_pos : room_length > 0)
  (tw_pos : tile_width_inches > 0) (tl_pos : tile_length_inches > 0)
  (room_dimensions : room_width = 15) (room_length_dimensions : room_length = 18)
  (tile_width_dimension : tile_width_inches = 3) (tile_length_dimension : tile_length_inches = 9) :
  let tile_width_feet := tile_width_inches / 12,
      tile_length_feet := tile_length_inches / 12,
      floor_area := room_width * room_length,
      tile_area := tile_width_feet * tile_length_feet in
  (floor_area / tile_area) = 1440 :=
  sorry

end tiles_needed_l199_199154


namespace arccos_half_eq_pi_over_three_l199_199209

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199209


namespace improper_integrals_diverge_l199_199053

open Topology
open Integral

noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (x : ℝ) : ℝ := sin x

theorem improper_integrals_diverge :
  (∫ x in 0..∞, f x = ⊤) ∧ ¬ ∃ l, tendsto (λ b, ∫ x in 0..b, g x) (Filter.atTop) (nhds l) :=
by sorry

end improper_integrals_diverge_l199_199053


namespace arccos_one_half_l199_199216

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199216


namespace arccos_one_half_l199_199229

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199229


namespace solve_quad_linear_system_l199_199449

theorem solve_quad_linear_system :
  (∃ x y : ℝ, x^2 - 6 * x + 8 = 0 ∧ y + 2 * x = 12 ∧ ((x, y) = (4, 4) ∨ (x, y) = (2, 8))) :=
sorry

end solve_quad_linear_system_l199_199449


namespace ratio_of_linear_combination_l199_199745

theorem ratio_of_linear_combination (a b x y : ℝ) (hb : b ≠ 0) 
  (h1 : 4 * x - 2 * y = a) (h2 : 5 * y - 10 * x = b) :
  a / b = -2 / 5 :=
by {
  sorry
}

end ratio_of_linear_combination_l199_199745


namespace find_ratio_CP_PA_l199_199482

-- Define the setup of the problem
variables {A B C D M P Q : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace P] [MetricSpace Q]
variables [HasDist A B C D M P Q] [HasMidpoint A D M]

-- Mention the conditions from the problem
variables (AB AC : ℝ)
variables (H1 : dist A B = 24) -- AB = 24
variables (H2 : dist A C = 13) -- AC = 13
variable (H3 : angleBisector A B C D) -- D is the intersection of angle bisector of ∠A with BC
variables (H4 : midpoint A D M) -- M is the midpoint of AD
variables (H5 : intersectionOnLine AC BM P) -- P is the intersection of AC and BM
variable (H6 : perpendicular AQ BC Q) -- AQ is perpendicular to BC

theorem find_ratio_CP_PA (h : dist C P / dist P A = 1) : 1 + 1 = 2 := sorry

end find_ratio_CP_PA_l199_199482


namespace arccos_half_eq_pi_over_three_l199_199205

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199205


namespace replace_asterisk_with_2x_l199_199539

-- Defining conditions for Lean
def expr_with_monomial (a : ℤ) : ℤ := (x : ℤ) → (x^3 - 2)^2 + (x^2 + a * x)^2

-- The statement of the proof in Lean 4
theorem replace_asterisk_with_2x : expr_with_monomial 2 = (x : ℤ) → x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_asterisk_with_2x_l199_199539


namespace simplify_and_rationalize_l199_199559

theorem simplify_and_rationalize :
  ( √3 / √4 ) * ( √5 / √6 ) * ( √7 / √8 ) = √70 / 16 :=
by
  sorry

end simplify_and_rationalize_l199_199559


namespace arccos_half_eq_pi_div_3_l199_199269

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199269


namespace trains_clear_time_correct_l199_199995

def length_of_first_train : ℝ := 60
def length_of_second_train : ℝ := 280
def speed_of_first_train : ℝ := 42 * 1000 / 3600  -- in m/s
def speed_of_second_train : ℝ := 30 * 1000 / 3600  -- in m/s
def total_length : ℝ := length_of_first_train + length_of_second_train
def relative_speed : ℝ := speed_of_first_train + speed_of_second_train
def correct_time : ℝ := 17

theorem trains_clear_time_correct :
  (total_length / relative_speed) = correct_time := by
  sorry

end trains_clear_time_correct_l199_199995


namespace train_passing_bridge_time_l199_199158

theorem train_passing_bridge_time (L_train L_bridge v t_bridge : ℕ) (h1 : L_train = 18) (h2 : L_bridge = 36) (h3 : v = L_train / 9) (h4 : t_bridge = (L_train + L_bridge) / v) : t_bridge = 27 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end train_passing_bridge_time_l199_199158


namespace pants_price_100_l199_199517

-- Define the variables and conditions
variables (x y : ℕ)

-- Define the prices according to the conditions
def coat_price_pants := x + 340
def coat_price_shoes_pants := y + x + 180
def total_price := (coat_price_pants x) + x + y

-- The theorem to prove
theorem pants_price_100 (h1: coat_price_pants x = coat_price_shoes_pants x y) (h2: total_price x y = 700) : x = 100 :=
sorry

end pants_price_100_l199_199517


namespace partitions_of_Hamiltonian_cycles_l199_199155

def HamiltonianCycle (E : Type) [Fintype E] [DecidableEq E] (edges : Finset (Finset E)) : Prop :=
  ∃ cyclic_permutation_of_edges : edges ≃ edges, 
  ∀ vertex ∈ edges, ∃! another_vertex ∈ edges, (vertex, another_vertex) ∈ edges

def Octahedron (V : Type) [Fintype V] (edges : Finset (Finset V)) : Prop :=
  (Finset.card edges = 12) ∧ (Finset.card V = 8)

theorem partitions_of_Hamiltonian_cycles (V : Type) [Fintype V] [DecidableEq V]
  (edges : Finset (Finset V)) (h1 : Octahedron V edges) : 
  ∃ partitions, partitions.length = 6 := 
sorry

end partitions_of_Hamiltonian_cycles_l199_199155


namespace greatest_multiple_of_5_and_7_less_than_1000_l199_199632

theorem greatest_multiple_of_5_and_7_less_than_1000 : ∃ x : ℕ, (x % 5 = 0) ∧ (x % 7 = 0) ∧ (x < 1000) ∧ (∀ y : ℕ, (y % 5 = 0) ∧ (y % 7 = 0) ∧ (y < 1000) → y ≤ x) ∧ x = 980 :=
begin
  sorry
end

end greatest_multiple_of_5_and_7_less_than_1000_l199_199632


namespace geometric_mean_le_3_sqrt_11_l199_199497

noncomputable def geometric_mean (a : List ℕ) (m : ℕ) : ℝ :=
  Real.exp (a.foldr (λ x y => x * log (x : ℝ) + y) 0 / m)

theorem geometric_mean_le_3_sqrt_11 (a : List ℕ) (m : ℕ) (h_len : a.length = m)
  (h_all_pos : ∀ i, i < m → 0 < a.nth_le i sorry)
  (h_sum : a.sum = 10 * m)
  (h_all_not_10 : ∀ i, i < m → a.nth_le i sorry ≠ 10) :
  geometric_mean a m ≤ 3 * Real.sqrt 11 :=
sorry

end geometric_mean_le_3_sqrt_11_l199_199497


namespace average_of_all_25_results_l199_199607

theorem average_of_all_25_results : 
  ∀ (results : Fin 25 → ℝ), 
  (∀ i : Fin 12, (results i) = 14) → (∀ i : Fin 12, (results (Fin.mk (24 - i.val) _)) = 17) →
  results 12 = 103 → 
  (1 / 25) * (∑ i, results i) = 19 := 
by 
  intros results h1 h2 h3
  unfold average
  have sum_first12 := ∑ i : Fin 12, (results i)
  have sum_last12 := ∑ i : Fin 12, results (Fin.mk (24 - i.val) _)
  have sum_13th : results 12 = 103 := by simp [h3]
  have total_sum := sorry
  have average_all_25 : (1 / 25) * (∑ i : Fin 25, results i) = 19 := by simp [total_sum]
  exact average_all_25

end average_of_all_25_results_l199_199607


namespace train_speed_180_kmph_l199_199159

def train_speed_in_kmph (length_meters : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_m_per_s := length_meters / time_seconds
  let speed_km_per_h := speed_m_per_s * 36 / 10
  speed_km_per_h

theorem train_speed_180_kmph:
  train_speed_in_kmph 400 8 = 180 := by
  sorry

end train_speed_180_kmph_l199_199159


namespace jeremy_total_songs_l199_199661

noncomputable def total_songs (x y : ℕ) : ℕ := 9 + x + y

theorem jeremy_total_songs :
  ∃ (x y : ℕ), 9 = 2 * nat.sqrt x - 5 ∧ x % 2 = 0 ∧ y = 1 / 2 * (9 + x) ∧ total_songs x y = 110 := by
{
  -- next steps to construct the proof, if necessary
  sorry
}

end jeremy_total_songs_l199_199661


namespace resulting_vector_correct_l199_199162

-- Define the initial vector.
def initial_vector : ℝ × ℝ × ℝ := (2, 1, 1)

-- Define the magnitude of the initial vector.
def magnitude_initial_vector : ℝ := Real.sqrt (2^2 + 1^2 + 1^2)

-- Define the resulting vector after a 90-degree rotation.
def resulting_vector : ℝ × ℝ × ℝ := (Real.sqrt (6 / 11), -3 * Real.sqrt (6 / 11), Real.sqrt (6 / 11))

-- Define a function to calculate the dot product between two vectors.
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the orthogonality condition: dot product of initial_vector and resulting_vector = 0
def orthogonality_condition : Prop := dot_product initial_vector resulting_vector = 0

-- Lemma stating the magnitude of the resulting vector remains the same as the initial vector.
lemma magnitude_preserved : (resulting_vector.1^2 + resulting_vector.2^2 + resulting_vector.3^2) = 6 :=
  sorry

-- The main theorem stating the resulting_vector under the given conditions.
theorem resulting_vector_correct : 
  ∃ v : ℝ × ℝ × ℝ, v = resulting_vector ∧ orthogonality_condition ∧ (v.1^2 + v.2^2 + v.3^2) = 6 :=
  by
    use resulting_vector
    split
    . reflexivity
    -- Proof of orthogonality
    . unfold orthogonality_condition
      sorry -- complete the proof of dot_product = 0
    -- Proof of magnitude preservation
    . apply magnitude_preserved
      sorry

end resulting_vector_correct_l199_199162


namespace find_rate_of_current_l199_199979

noncomputable def rate_of_current (speed_boat : ℝ) (distance_downstream : ℝ) (time_downstream_min : ℝ) : ℝ :=
 speed_boat + 3

theorem find_rate_of_current : ∀ (speed_boat distance_downstream time_downstream_min : ℝ), 
  speed_boat = 15 → 
  distance_downstream = 3.6 → 
  time_downstream_min = 12 → 
  rate_of_current speed_boat distance_downstream time_downstream_min = 18 - speed_boat :=
by
  intros speed_boat distance_downstream time_downstream_min h_speed h_distance h_time
  rw [h_speed, h_distance, h_time]
  have time_hours : ℝ := (12 / 60 : ℝ)
  calc
    (rate_of_current 15 3.6 12) = speed_boat + 3 : rfl
    ... = 15 + 3 : by rw [h_speed]
    ... = 18 - speed_boat : by rw [h_speed]

end find_rate_of_current_l199_199979


namespace total_presents_l199_199738

variables (ChristmasPresents BirthdayPresents EasterPresents HalloweenPresents : ℕ)

-- Given conditions
def condition1 : ChristmasPresents = 60 := sorry
def condition2 : BirthdayPresents = 3 * EasterPresents := sorry
def condition3 : EasterPresents = (ChristmasPresents / 2) - 10 := sorry
def condition4 : HalloweenPresents = BirthdayPresents - EasterPresents := sorry

-- Proof statement
theorem total_presents (h1 : ChristmasPresents = 60)
    (h2 : BirthdayPresents = 3 * EasterPresents)
    (h3 : EasterPresents = (ChristmasPresents / 2) - 10)
    (h4 : HalloweenPresents = BirthdayPresents - EasterPresents) :
    ChristmasPresents + BirthdayPresents + EasterPresents + HalloweenPresents = 180 :=
sorry

end total_presents_l199_199738


namespace arccos_one_half_l199_199238

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199238


namespace arccos_half_eq_pi_div_three_l199_199249

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199249


namespace urn_exchange_theorem_l199_199840

noncomputable def urn_exchange_problem (m n k : ℕ) : Prop :=
  let p := k - (k + m - m) in
  let b := k - p in
  b = (k - p)

theorem urn_exchange_theorem (m n k : ℕ) : urn_exchange_problem m n k :=
by {
  sorry
}

end urn_exchange_theorem_l199_199840


namespace pool_surface_area_l199_199152

/-
  Given conditions:
  1. The width of the pool is 3 meters.
  2. The length of the pool is 10 meters.

  To prove:
  The surface area of the pool is 30 square meters.
-/
def width : ℕ := 3
def length : ℕ := 10
def surface_area (length width : ℕ) : ℕ := length * width

theorem pool_surface_area : surface_area length width = 30 := by
  unfold surface_area
  rfl

end pool_surface_area_l199_199152


namespace parallel_lines_eq_slope_l199_199456

theorem parallel_lines_eq_slope (a : ℝ) (l1_parallel_l2 : ∀ x y, (ax + y + 2a = 0) → (x + ay + 3 = 0)) : a = 1 ∨ a = -1 :=
by sorry

end parallel_lines_eq_slope_l199_199456


namespace arithmetic_b_general_formula_a_l199_199596

def sequence_a (n : ℕ) : ℝ
  | 1 => 1
  | (n + 1) => (3 * sequence_a n) / (sequence_a n + 3)

def sequence_b (n : ℕ) : ℝ := 1 / (sequence_a n)

theorem arithmetic_b (n : ℕ) : sequence_b (n + 1) = sequence_b n + 1 / 3 := 
sorry

theorem general_formula_a (n : ℕ) : sequence_a n = 3 / (n + 2) := 
sorry

end arithmetic_b_general_formula_a_l199_199596


namespace arccos_half_eq_pi_div_three_l199_199281

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199281


namespace distance_to_bus_stand_l199_199121

theorem distance_to_bus_stand :
  ∀ D : ℝ, (D / 5 - 0.2 = D / 6 + 0.25) → D = 13.5 :=
by
  intros D h
  sorry

end distance_to_bus_stand_l199_199121


namespace quadrilateral_perimeter_l199_199697

/-- The vertices of the quadrilateral in 2D space -/
def vertices : List (ℝ × ℝ) := [(1, 2), (4, 6), (6, 5), (4, 1)]

/-- The distance formula between two points in 2D space -/
def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

-- Calculate the perimeter of the quadrilateral
noncomputable def perimeter : ℝ :=
  dist (1, 2) (4, 6) + dist (4, 6) (6, 5) + dist (6, 5) (4, 1) + dist (4, 1) (1, 2)

/-- The target form of the perimeter is c * sqrt(5) + d * sqrt(13) -/
def target_perimeter_form (c d : ℤ) : ℝ :=
  c * Real.sqrt 5 + d * Real.sqrt 13

/-- Prove that the perimeter can be expressed in the form c * sqrt(5) + d * sqrt(13) with c + d = 3 and that corresponds to the sum of the distances of the quadrilateral's vertices. -/
theorem quadrilateral_perimeter :
  ∃ (c d : ℤ), (perimeter = target_perimeter_form c d) ∧ (c + d = 3) := by
  sorry

end quadrilateral_perimeter_l199_199697


namespace arccos_half_eq_pi_over_three_l199_199213

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199213


namespace P_and_Q_equivalent_l199_199457

def P (x : ℝ) : Prop := 3 * x - x^2 ≤ 0
def Q (x : ℝ) : Prop := |x| ≤ 2
def P_intersection_Q (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

theorem P_and_Q_equivalent : ∀ x, (P x ∧ Q x) ↔ P_intersection_Q x :=
by {
  sorry
}

end P_and_Q_equivalent_l199_199457


namespace typist_times_l199_199691

theorem typist_times (x y : ℕ) (hx : x = 10) (hy : y = 16)
  (h1 : ∀ (x y : ℕ), (40 / y) - (40 / x) = 3)
  (h2 : ∀ (x y : ℕ), 5 * ((80 / x) + (80 / y)) = 65) :
  (x = 10) ∧ (y = 16) :=
by
  apply And.intro
  apply hx
  apply hy
  sorry -- to skip the proof

end typist_times_l199_199691


namespace sphere_to_cylinder_volume_ratio_l199_199144

-- Definitions from conditions
def sphere_radius (R : ℝ) : ℝ := R
def cylinder_height (R : ℝ) : ℝ := 2 * R
def volume_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3
def volume_cylinder (R : ℝ) : ℝ := Real.pi * R^2 * cylinder_height R

-- Lean theorem statement
theorem sphere_to_cylinder_volume_ratio (R : ℝ) (h : ℝ) (hs : h = 2 * R) :
  volume_sphere R / volume_cylinder R = (2 / 3) :=
by
  sorry

end sphere_to_cylinder_volume_ratio_l199_199144


namespace complement_intersection_in_U_l199_199834

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_in_U : (U \ (A ∩ B)) = {1, 4, 5, 6, 7, 8} :=
by {
  sorry
}

end complement_intersection_in_U_l199_199834


namespace arccos_half_eq_pi_div_3_l199_199273

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199273


namespace replace_asterisk_with_2x_l199_199544

-- Defining conditions for Lean
def expr_with_monomial (a : ℤ) : ℤ := (x : ℤ) → (x^3 - 2)^2 + (x^2 + a * x)^2

-- The statement of the proof in Lean 4
theorem replace_asterisk_with_2x : expr_with_monomial 2 = (x : ℤ) → x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_asterisk_with_2x_l199_199544


namespace arccos_one_half_l199_199222

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199222


namespace proof_a_squared_plus_b_squared_l199_199447

theorem proof_a_squared_plus_b_squared (a b : ℝ) (h1 : (a + b) ^ 2 = 4) (h2 : a * b = 1) : a ^ 2 + b ^ 2 = 2 := 
by 
  sorry

end proof_a_squared_plus_b_squared_l199_199447


namespace locus_is_radical_axis_or_infinite_line_l199_199734

noncomputable def locus_of_points (k : Circle) (K : Point) : Set Point :=
  {M : Point | ∃ P Q : Point, P ≠ Q ∧ P ∈ k ∧ Q ∈ k ∧
    let k' := Circle_through K P Q in 
    tangent_line k' K ∩ line_through P Q = M }

theorem locus_is_radical_axis_or_infinite_line (k : Circle) (K : Point) :
  ∃ l : Line, (l = radical_axis k (Circle_through K (some_point k) (another_point k)))
  ∧ (∀ M : Point, M ∈ (locus_of_points k K) ↔ M ∈ l) ∨
  (K = center k ∧ ∀ M : Point, M ∈ (locus_of_points k K) ↔ true) :=
sorry

end locus_is_radical_axis_or_infinite_line_l199_199734


namespace count_minks_l199_199872

structure Creature :=
  (is_unicorn : Bool)

structure Statements :=
  (susan_diff_alice : Bool)
  (tim_henry_mink : Bool)
  (henry_tim_mink : Bool)
  (bob_least_three_unicorns : Bool)
  (alice_same_susan_bob : Bool)

def creatures : List Creature := [
  { is_unicorn := true },   -- Example, needs actual logic to define it correctly
  { is_unicorn := true },
  { is_unicorn := false },
  { is_unicorn := true },
  { is_unicorn := false }
]

def example_statements : Statements :=
  { susan_diff_alice := (creatures.nth 0).get.is_unicorn != (creatures.nth 4).get.is_unicorn,
    tim_henry_mink := not (creatures.nth 2).get.is_unicorn,
    henry_tim_mink := not (creatures.nth 1).get.is_unicorn,
    bob_least_three_unicorns := creatures.count (λ c => c.is_unicorn) >= 3,
    alice_same_susan_bob := (creatures.nth 4).get.is_unicorn = (creatures.nth 0).get.is_unicorn &&
                            (creatures.nth 4).get.is_unicorn = (creatures.nth 3).get.is_unicorn }

theorem count_minks : example_statements → creatures.count (λ c => not c.is_unicorn) = 2 := 
by
  sorry

end count_minks_l199_199872


namespace range_of_mu_l199_199781

noncomputable def problem_statement (a b μ : ℝ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < μ) ∧ (1 / a + 9 / b = 1) → (0 < μ ∧ μ ≤ 16)

theorem range_of_mu (a b μ : ℝ) : problem_statement a b μ :=
  sorry

end range_of_mu_l199_199781


namespace sum_of_ages_l199_199917

theorem sum_of_ages {l t : ℕ} (h1 : t > l) (h2 : t * t * l = 72) : t + t + l = 14 :=
sorry

end sum_of_ages_l199_199917


namespace arccos_half_is_pi_div_three_l199_199260

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199260


namespace length_of_CE_l199_199478

-- Given conditions
variable (A B C D E : Type) [InnerProductSpace ℝ E]
variable (haebe : angle A E B = π / 2)
variable (hbce : angle B E C = π / 3)
variable (hced : angle C E D = π / 3)
variable (aelength : ∥A - E∥ = 30)

-- Hypotenuse length calculation
def BE_length : ℝ := 30 * (Real.cos (π / 4))

-- Target length calculation using triangle BCE
def CE_length : ℝ := BE_length * (Real.sin (π / 3))

theorem length_of_CE : CE_length = 15 * Real.sqrt 6 / 2 := by
  sorry

end length_of_CE_l199_199478


namespace replace_star_with_2x_l199_199551

theorem replace_star_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_star_with_2x_l199_199551


namespace arccos_half_is_pi_div_three_l199_199265

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199265


namespace arccos_one_half_is_pi_div_three_l199_199194

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199194


namespace cosine_sequence_count_l199_199856

def is_cosine_sequence (a : List ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < a.length - 1 → (if i % 2 = 0 then a.get ⟨i, _⟩ > a.get ⟨i+1, _⟩ else a.get ⟨i, _⟩ < a.get ⟨i+1, _⟩)

def total_arrangements_equal_28 (l : List ℕ) (h₀ : l.sort = [1, 1, 2, 3, 4, 5]) (h₁ : ∀ a, a ∈ l → 1 ≤ a ∧ a ≤ 5) : Prop :=
  l.permutations.count (λ a, is_cosine_sequence a) = 28

theorem cosine_sequence_count : total_arrangements_equal_28 [1, 1, 2, 3, 4, 5] sorry sorry :=
sorry

end cosine_sequence_count_l199_199856


namespace similarity_of_ABCD_and_A2B2C2D2_ratio_of_similitude_l199_199894

variables {A B C D : Point} -- Assuming a suitable definition for Points and appropriate geometry library.

-- Conditions based on problem statements
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry -- Proper definition goes here
def T (A B C D : Point) : Quad := sorry -- Proper definition goes here

-- Proof Problem Statements for (a) and (b)
theorem similarity_of_ABCD_and_A2B2C2D2
  (Hconvex : is_convex_quadrilateral A B C D)
  (Hnot_on_circle : ¬ collinear A B C D) :
  similar (T (T A B C D)) (A B C D) :=
sorry

theorem ratio_of_similitude
  (Hconvex : is_convex_quadrilateral A B C D)
  (Hnot_on_circle : ¬ collinear A B C D)
  (alpha beta gamma delta : Angle) :
  ratio (T (T A B C D)) (A B C D) = 
  (abs (sin (alpha + gamma) * sin (beta + delta)) / (4 * sin alpha * sin beta * sin gamma * sin delta)) ^ 2 :=
sorry

end similarity_of_ABCD_and_A2B2C2D2_ratio_of_similitude_l199_199894


namespace sample_size_l199_199461

theorem sample_size (w_under30 : ℕ) (w_30to40 : ℕ) (w_40plus : ℕ) (sample_40plus : ℕ) (total_sample : ℕ) :
  w_under30 = 2400 →
  w_30to40 = 3600 →
  w_40plus = 6000 →
  sample_40plus = 60 →
  total_sample = 120 :=
by
  intros
  sorry

end sample_size_l199_199461


namespace translation_correct_l199_199811

def vector_a : ℝ × ℝ := (1, 1)

def translate_right (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1 + d, v.2)
def translate_down (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1, v.2 - d)

def vector_b := translate_down (translate_right vector_a 2) 1

theorem translation_correct :
  vector_b = (3, 0) :=
by
  -- proof steps will go here
  sorry

end translation_correct_l199_199811


namespace white_pieces_total_l199_199990

theorem white_pieces_total (B W : ℕ) 
  (h_total_pieces : B + W = 300) 
  (h_total_piles : 100 * 3 = B + W) 
  (h_piles_1_white : {n : ℕ | n = 27}) 
  (h_piles_2_3_black : {m : ℕ | m = 42}) 
  (h_piles_3_black_3_white : 15 = 15) :
  W = 158 :=
by
  sorry

end white_pieces_total_l199_199990


namespace koala_food_consumed_l199_199005

theorem koala_food_consumed (x y : ℝ) (h1 : 0.40 * x = 12) (h2 : 0.20 * y = 2) : 
  x = 30 ∧ y = 10 := 
by
  sorry

end koala_food_consumed_l199_199005


namespace sum_of_divisors_252_l199_199644

theorem sum_of_divisors_252 :
  let n := 252
  let prime_factors := [2, 2, 3, 3, 7]
  sum_of_divisors n = 728 :=
by
  sorry

end sum_of_divisors_252_l199_199644


namespace choose_points_not_in_same_plane_l199_199083

-- Variables for the problem
def vertices : ℕ := 4
def midpoints : ℕ := 6
def total_points : ℕ := vertices + midpoints

theorem choose_points_not_in_same_plane : 
  let total_ways := 
    (Mathlib.combinatorics.choose vertices 3) * midpoints + 
    (Mathlib.combinatorics.choose vertices 2) * 
    (Mathlib.combinatorics.choose midpoints 2) + 
    vertices * (Mathlib.combinatorics.choose midpoints 3) in
  total_ways - 4 = 190 :=
by sorry

end choose_points_not_in_same_plane_l199_199083


namespace find_range_of_k_l199_199787

noncomputable def vector_a (x k : ℝ) : ℝ × ℝ :=
(Real.sqrt 3 * Real.sin x, k * Real.cos x)

noncomputable def vector_b (x k : ℝ) : ℝ × ℝ :=
(2 * k * Real.cos x, 2 * Real.cos x)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

noncomputable def g (x k : ℝ) : ℝ :=
dot_product (vector_a x k) (vector_b x k) - k + 1

noncomputable def h (x : ℝ) : ℝ :=
Real.sin (2 * x) - (13 * Real.sqrt 2 / 5) * Real.sin (x + Real.pi / 4) + 369 / 100

def is_constant_triangle_function (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
∀ a b c ∈ domain, a ≠ b → b ≠ c → c ≠ a → f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b

theorem find_range_of_k :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), is_constant_triangle_function (g · k) (Set.Icc 0 (Real.pi / 2))) →
  (∀ x1 ∈ Set.Icc 0 (Real.pi / 2), ∃ x2 ∈ Set.Icc 0 (Real.pi / 2), g x2 k = h x1) →
  (9 / 200 ≤ k ∧ k < 1 / 4) ∨ (-1/5 < k ∧ k ≤ -9 / 100) :=
by
  sorry

end find_range_of_k_l199_199787


namespace arccos_half_eq_pi_div_three_l199_199242

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199242


namespace problem_statement_l199_199172

-- Definitions for equilateral triangle, distances, and the three triangles PS_kT_k
structure EquilateralTriangle (P Q R : Type) :=
(length : ℝ)
(side_lengths : length = 11)

structure CongruentTriangle (P Q R S T : Type) :=
(length : ℝ)
(side_lengths : length = 11)
(QS_eq : QS = 5)

def sum_of_squares_of_RTk (P Q R S₁ T₁ S₂ T₂ S₃ T₃ : Type) [EquilateralTriangle P Q R] [CongruentTriangle P Q R S₁ T₁] [CongruentTriangle P Q R S₂ T₂] [CongruentTriangle P Q R S₃ T₃] : ℝ :=
  3 * 121

theorem problem_statement (P Q R S₁ T₁ S₂ T₂ S₃ T₃ : Type) [EquilateralTriangle P Q R] [CongruentTriangle P Q R S₁ T₁] [CongruentTriangle P Q R S₂ T₂] [CongruentTriangle P Q R S₃ T₃] : sum_of_squares_of_RTk P Q R S₁ T₁ S₂ T₂ S₃ T₃ = 363 := by
  sorry

end problem_statement_l199_199172


namespace exists_nat_a_b_l199_199022

theorem exists_nat_a_b (N : ℕ) (hN : 0 < N) :
  ∃ (a b : ℕ), 1 ≤ b ∧ b ≤ N ∧ |(a : ℝ) - b * real.sqrt 2| ≤ 1 / N :=
by
  sorry

end exists_nat_a_b_l199_199022


namespace probability_x_lt_y_l199_199149

noncomputable def rectangle_vertices := [(0, 0), (4, 0), (4, 3), (0, 3)]

theorem probability_x_lt_y :
  let area_triangle := (1 / 2) * 3 * 3
  let area_rectangle := 4 * 3
  let probability := area_triangle / area_rectangle
  probability = (3 / 8) := 
by
  sorry

end probability_x_lt_y_l199_199149


namespace distinct_real_roots_range_l199_199458

theorem distinct_real_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ax^2 + 2 * x + 1 = 0 ∧ ay^2 + 2 * y + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
by
  sorry

end distinct_real_roots_range_l199_199458


namespace bob_distance_walked_l199_199932

theorem bob_distance_walked
  (Y_walk_rate : ℕ) (B_walk_rate : ℕ) (total_distance : ℕ)
  (Y_extra_time : ℕ) :
  Y_walk_rate = 5 
  ∧ B_walk_rate = 7 
  ∧ total_distance = 65 
  ∧ Y_extra_time = 1 
  → (∃ B : ℕ, B = 35 ∧ B + Y_walk_rate * (B / B_walk_rate + Y_extra_time) = total_distance) :=
begin
  intro h,
  sorry
end

end bob_distance_walked_l199_199932


namespace exists_rectangle_in_parallelograms_l199_199103

theorem exists_rectangle_in_parallelograms (octagon_decomposed : ∃ P : list parallelogram, decomposes_to_parallelograms P regular_octagon) :
  ∃ Q : parallelogram, is_rectangle Q ∧ Q ∈ P :=
sorry

end exists_rectangle_in_parallelograms_l199_199103


namespace Q1_Q2_l199_199827

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := real.log x - m * x ^ 2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * m * x ^ 2 + x
noncomputable def F (x : ℝ) (m : ℝ) : ℝ := f x m + g x m

theorem Q1 (x : ℝ) (hx : 0 < x) : f x (1 / 2) > 0 := 
by 
  sorry

theorem Q2 (m : ℤ) (hF : ∀ x > 0, F x m.to_real ≤ m.to_real * x - 1) : m ≥ 2 :=
by 
  sorry

end Q1_Q2_l199_199827


namespace intersection_tangent_value_l199_199864

theorem intersection_tangent_value (α : ℝ) (h_nonzero : α ≠ 0) (h_eq : tan α = -α) : (α^2 + 1) * (1 + cos (2 * α)) = 2 := by
  sorry

end intersection_tangent_value_l199_199864


namespace third_last_digit_square_ends_in_5_l199_199940

theorem third_last_digit_square_ends_in_5 (k : ℤ) : 
  let n := 10 * k + 5 in 
  ∃ d : ℕ, d ∈ {0, 2, 6} ∧ (n^2 / 100) % 10 = d := 
by
  sorry

end third_last_digit_square_ends_in_5_l199_199940


namespace inequality_proof_l199_199024

theorem inequality_proof (n : ℕ) (h₁ : n ≥ 2) (x : fin n → ℝ)
  (h₂ : (∀ i, 0 < x i)) (h₃ : (∑ i, x i = 1)) :
  (∑ i, x i / real.sqrt (1 - x i)) ≥
  (∑ i, real.sqrt (x i)) / real.sqrt (n - 1) :=
sorry

end inequality_proof_l199_199024


namespace john_additional_correct_questions_l199_199002

theorem john_additional_correct_questions :
  ∃ (x : ℕ), 
  let total_questions := 90,
      ancient_history := 20,
      modern_history := 40,
      contemporary_history := 30,
      correct_ancient := 0.50 * ancient_history,
      correct_modern := 0.25 * modern_history,
      correct_contemporary := 0.70 * contemporary_history,
      total_correct := correct_ancient + correct_modern + correct_contemporary,
      pass_threshold := 0.55 * total_questions,
      additional_needed := pass_threshold - total_correct in
  x = nat.ceil (additional_needed) ∧ x = 9 := by
  sorry

end john_additional_correct_questions_l199_199002


namespace a_plus_c_eq_neg800_l199_199906

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem a_plus_c_eq_neg800 (a b c d : ℝ) (h1 : g (-a / 2) c d = 0)
  (h2 : f (-c / 2) a b = 0) (h3 : ∀ x, f x a b ≥ f (-a / 2) a b)
  (h4 : ∀ x, g x c d ≥ g (-c / 2) c d) (h5 : f (-a / 2) a b = g (-c / 2) c d)
  (h6 : f 200 a b = -200) (h7 : g 200 c d = -200) :
  a + c = -800 := sorry

end a_plus_c_eq_neg800_l199_199906


namespace problem_statement_l199_199758

open Nat

def greatest_odd_divisor (x : ℕ) : ℕ :=
  if x % 2 = 0 then greatest_odd_divisor (x / 2) else x

theorem problem_statement (k : ℕ) (hk : ∃ l : ℕ, l ≥ 2 ∧ k = 2^l - 1) :
  ∀ n : ℕ, n ≥ 2 → ¬ (n ∣ greatest_odd_divisor (k^n + 1)) :=
by
  sorry

end problem_statement_l199_199758


namespace cupboard_cost_price_l199_199101

theorem cupboard_cost_price
  (C : ℝ)
  (h1 : ∀ (S : ℝ), S = 0.84 * C) -- Vijay sells a cupboard at 84% of the cost price.
  (h2 : ∀ (S_new : ℝ), S_new = 1.16 * C) -- If Vijay got Rs. 1200 more, he would have made a profit of 16%.
  (h3 : ∀ (S_new S : ℝ), S_new - S = 1200) -- The difference between new selling price and original selling price is Rs. 1200.
  : C = 3750 := 
sorry -- Proof is not required.

end cupboard_cost_price_l199_199101


namespace domain_of_g_l199_199440

-- Assume f is a function from ℝ to ℝ with domain [0,2]
variable (f : ℝ → ℝ)
variable (hf_domain : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f x = y)

-- Define g as equivalent to f
def g := f

-- Prove that the domain of g is [0,2] given the domain of f is [0,2]
theorem domain_of_g (x : ℝ) : (0 ≤ x ∧ x ≤ 2) ↔ ∃ y, g x = y := by
  sorry

end domain_of_g_l199_199440


namespace always_positive_inequality_l199_199962

theorem always_positive_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end always_positive_inequality_l199_199962


namespace silver_dollars_total_l199_199925

theorem silver_dollars_total :
  ∀ (MrHa MrPhung MrChiu MsLin : ℕ),
  MrChiu = 56 →
  MrPhung = MrChiu + 16 →
  MrHa = MrPhung + 5 →
  MsLin = (MrHa + MrPhung + MrChiu) + 25 →
  MrHa + MrPhung + MrChiu + MsLin = 435 :=
by
  intros MrHa MrPhung MrChiu MsLin h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end silver_dollars_total_l199_199925


namespace salary_reduction_l199_199595

variable (S R : ℝ) (P : ℝ)
variable (h1 : R = S * (1 - P/100))
variable (h2 : S = R * (1 + 53.84615384615385 / 100))

theorem salary_reduction : P = 35 :=
by sorry

end salary_reduction_l199_199595


namespace largest_integral_solution_l199_199761

theorem largest_integral_solution (x : ℤ) : (1 / 4 : ℝ) < (x / 7 : ℝ) ∧ (x / 7 : ℝ) < (3 / 5 : ℝ) → x = 4 :=
by {
  sorry
}

end largest_integral_solution_l199_199761


namespace main_theorem_l199_199784

variable (x y z : ℝ) (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1)

theorem main_theorem (h₀ : 0 < x ∧ 0 < y ∧ 0 < z) (h₁ : x^3 + y^3 + z^3 = 1):
  (x^2 / (1 - x^2)) + (y^2 / (1 - y^2)) + (z^2 / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := 
by
  sorry

end main_theorem_l199_199784


namespace num_pairs_in_arithmetic_progression_l199_199355

theorem num_pairs_in_arithmetic_progression : 
  ∃ n : ℕ, n = 2 ∧ ∀ a b : ℝ, (a = (15 + b) / 2 ∧ (a + a * b = 2 * b)) ↔ 
  (a = (9 + 3 * Real.sqrt 7) / 2 ∧ b = -6 + 3 * Real.sqrt 7)
  ∨ (a = (9 - 3 * Real.sqrt 7) / 2 ∧ b = -6 - 3 * Real.sqrt 7) :=    
  sorry

end num_pairs_in_arithmetic_progression_l199_199355


namespace arccos_half_eq_pi_div_three_l199_199312

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199312


namespace arccos_half_eq_pi_div_3_l199_199270

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199270


namespace increasing_range_of_a_l199_199863

noncomputable def f (a x : ℝ) := (1 / 3) * x ^ 3 - (1 / 2) * a * x ^ 2 + x

theorem increasing_range_of_a (a : ℝ) : 
  (∀ x ∈ Ioo 0 1, deriv (f a) x ≥ 0) → a ≤ 2 := 
sorry

end increasing_range_of_a_l199_199863


namespace replace_star_with_2x_l199_199552

theorem replace_star_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_star_with_2x_l199_199552


namespace isosceles_triangle_QSR_l199_199404

theorem isosceles_triangle_QSR (P Q R S : Type)
  (QR PR : P ≃ Q)
  (anglePQR : angle P Q R = 60)
  (bisects_PS : bisects (overline P S) (angle Q P R))
  (bisects_RS : bisects (overline R S) (angle P R Q)) :
  angle Q S R = 120 :=
by
  sorry

end isosceles_triangle_QSR_l199_199404


namespace roots_are_prime_then_a_is_five_l199_199806

theorem roots_are_prime_then_a_is_five (x1 x2 a : ℕ) (h_prime_x1 : Prime x1) (h_prime_x2 : Prime x2)
  (h_eq : x1 + x2 = a) (h_eq_mul : x1 * x2 = a + 1) : a = 5 :=
sorry

end roots_are_prime_then_a_is_five_l199_199806


namespace rectangle_dimensions_l199_199700

theorem rectangle_dimensions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_area : x * y = 36) (h_perimeter : 2 * x + 2 * y = 30) : 
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) :=
by
  sorry

end rectangle_dimensions_l199_199700


namespace fill_space_without_gaps_or_overlaps_l199_199392

-- Define the given shape (polyomino) composed of six unit cubes.
def polyomino : Type := ℝ × ℝ × ℝ

-- Define the condition that the polyomino is constructed from six unit cubes.
def is_composed_of_six_unit_cubes (p : polyomino) : Prop :=
  ∃ (cubes : fin 6 → polyomino), ∀ i, ∃ x y z, cubes i = (x, y, z) ∧
  (∀ j, i ≠ j → cubes i ≠ cubes j) -- each cube is distinct

-- State the main question: can space be filled without gaps and overlaps using congruent copies?
theorem fill_space_without_gaps_or_overlaps :
  ∀ (p : polyomino), (is_composed_of_six_unit_cubes p) →
  ∃ (copies : set polyomino), ∀ x y z, (x, y, z) ∈ copies ∧
  (∀ i j, i ≠ j → cubes i ∩ cubes j = ∅) := sorry

end fill_space_without_gaps_or_overlaps_l199_199392


namespace final_number_on_blackboard_is_2010_l199_199969

theorem final_number_on_blackboard_is_2010 :
  ∃ n : ℕ, n = 2010 ∧
  let a : List ℚ := List.map (fun i => 1 / i) (List.range' 1 2010) in
  let operation := fun x y => x + y + x * y in
  (∃ b, b = List.foldl operation 0  a ∧ b = n) :=
sorry

end final_number_on_blackboard_is_2010_l199_199969


namespace zero_points_sum_gt_one_l199_199427

noncomputable def f (x : ℝ) : ℝ := Real.log x + (1 / (2 * x))

noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (hx₁ : g x₁ m = 0) (hx₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := 
  by
    sorry

end zero_points_sum_gt_one_l199_199427


namespace probability_of_at_least_three_same_l199_199383

/-- Given five six-sided dice rolled where there is no three-of-a-kind 
but there is a pair of dice showing the same number, and the pair is set aside 
and the remaining three dice are re-rolled, the probability that at least three of the five dice 
show the same number after re-rolling the three dice is 4/9. -/
theorem probability_of_at_least_three_same : 
  let total_outcomes := 6^3 in
  let matching_pair := total_outcomes - 5^3 in
  let all_three_same := 6 in
  let overcount_correction := 1 in
  let successful_outcomes := matching_pair + all_three_same - overcount_correction in
  successful_outcomes / total_outcomes = (4 : ℚ) / 9 :=
by
  repeat { sorry }

end probability_of_at_least_three_same_l199_199383


namespace min_value_of_a_plus_mb_l199_199904

theorem min_value_of_a_plus_mb {a b : ℝ} (ha : a.norm = 1) (hb : b.norm = 1) (dot_ab : a.dot_product b = 3/5) (m : ℝ) : 
  ∃ m : ℝ, (a + m * b).norm = 4/5 := sorry

end min_value_of_a_plus_mb_l199_199904


namespace greatest_sum_of_pairs_exists_greatest_sum_of_pairs_l199_199989

theorem greatest_sum_of_pairs (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 8 :=
begin
  sorry
end

theorem exists_greatest_sum_of_pairs : 
  ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ ∀ (a b : ℤ), a^2 + b^2 = 50 → a + b ≤ x + y :=
begin
  use [7, 1],
  split,
  { norm_num },
  { intros a b hab,
    have key : ∀ (x y : ℤ), x^2 + y^2 = 50 → x + y ≤ 8,
    { intros x y hx,
      sorry },
    exact key a b hab }
end

end greatest_sum_of_pairs_exists_greatest_sum_of_pairs_l199_199989


namespace number_of_triangles_l199_199736

-- Definition of the problem
def is_valid_point (x y : ℕ) : Prop := 17 * x + y = 144

-- Definition of the conditions for points P and Q
def valid_triangle (P Q : ℕ × ℕ) : Prop :=
  let (x₁, y₁) := P;
  let (x₂, y₂) := Q;
  is_valid_point x₁ y₁ ∧ is_valid_point x₂ y₂ ∧ x₁ ≠ x₂

-- Definition to check the area requirement
def valid_area (P Q : ℕ × ℕ) : Prop :=
  let (x₁, y₁) := P;
  let (x₂, y₂) := Q;
  abs (x₁ * y₂ - x₂ * y₁) ≠ 0

-- Statement of the problem that needs to be proven
theorem number_of_triangles : 
  (finset.univ.filter (λ P : ℕ × ℕ, ∃ Q : ℕ × ℕ, valid_triangle P Q ∧ valid_area P Q)).card = 36 :=
sorry

end number_of_triangles_l199_199736


namespace probability_of_first_three_red_cards_l199_199692

theorem probability_of_first_three_red_cards :
  let total_cards := 60
  let red_cards := 36
  let black_cards := total_cards - red_cards
  let total_ways := total_cards * (total_cards - 1) * (total_cards - 2)
  let red_ways := red_cards * (red_cards - 1) * (red_cards - 2)
  (red_ways / total_ways) = 140 / 673 :=
by
  sorry

end probability_of_first_three_red_cards_l199_199692


namespace arccos_half_is_pi_div_three_l199_199267

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199267


namespace algae_coverage_10_percent_l199_199573

-- Define constants for the given problem
def algae_coverage (day : ℕ) : ℝ := 2^(day - 24)

-- Define our goal, which is to find the day when the coverage was approximately 10%
def day_when_10_percent_covered := 21

-- Define the proposition to prove
theorem algae_coverage_10_percent : algae_coverage day_when_10_percent_covered ≈ 0.10 :=
sorry

end algae_coverage_10_percent_l199_199573


namespace final_number_on_blackboard_is_2010_l199_199968

theorem final_number_on_blackboard_is_2010 :
  ∃ n : ℕ, n = 2010 ∧
  let a : List ℚ := List.map (fun i => 1 / i) (List.range' 1 2010) in
  let operation := fun x y => x + y + x * y in
  (∃ b, b = List.foldl operation 0  a ∧ b = n) :=
sorry

end final_number_on_blackboard_is_2010_l199_199968


namespace arccos_half_eq_pi_div_three_l199_199243

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199243


namespace complex_number_quadrant_l199_199363

theorem complex_number_quadrant (θ : Real) (hθ : θ = 2 * Real.pi / 3) : 
  let z := Complex.exp (Complex.I * θ)
  in z.re < 0 ∧ z.im > 0 :=
by
  let θ := 2 * Real.pi / 3
  have hθ : θ = 2 * Real.pi / 3 := rfl
  let z := Complex.exp (Complex.I * θ)
  have hz : z = Complex.mk (Real.cos θ) (Real.sin θ) := complex_exp_eq_cos_add_sin θ
  have hz2 : z = Complex.mk (-1 / 2) (Real.sqrt 3 / 2) := by 
    rw [θ, Real.cos, Real.sin, Real.sqrt]
  sorry

end complex_number_quadrant_l199_199363


namespace probability_girl_from_family_E_expectation_X_l199_199076

/-- Given the distribution of boys and girls among families A to E, if a girl is randomly selected,
    the probability that the girl is from family E is 1/2. --/
theorem probability_girl_from_family_E :
  let boys := [0, 1, 0, 1, 1]
  let girls := [0, 0, 1, 1, 2]
  let total_girls := girls.sum
  let girls_in_E := girls[4]
  (girls_in_E : ℚ) / (total_girls : ℚ) = 1 / 2 :=
by
  sorry

/-- Given the distribution of boys and girls among families A to E, the expectation of X,
    the number of families where girls outnumber boys when three families are selected randomly,
    is 6/5. --/
theorem expectation_X :
  let boys := [0, 1, 0, 1, 1]
  let girls := [0, 0, 1, 1, 2]
  let possible_combinations := [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4],
                                [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
  let girls_outnumber_boys (families : List Nat) : Nat := families.count (fun i => girls[i] > boys[i])
  let distribution := (possible_combinations.map (fun comb => girls_outnumber_boys comb)).count
  let P_X_0 := distribution.count (· = 0) / possible_combinations.length
  let P_X_1 := distribution.count (· = 1) / possible_combinations.length
  let P_X_2 := distribution.count (· = 2) / possible_combinations.length
  let E_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2
  (E_X : ℚ) = 6 / 5 :=
by
  sorry

end probability_girl_from_family_E_expectation_X_l199_199076


namespace tangent_lines_with_equal_intercepts_l199_199443

/-
  The circle is defined by the equation (x + 2 * sqrt(2))^2 + (y - 3 * sqrt(2))^2 = 1.
  We need to prove that the number of lines that are tangent to this circle and have
  equal intercepts on the x-axis and y-axis is 4.
-/

def circle (x y : ℝ) : Prop :=
  (x + 2 * Real.sqrt 2) ^ 2 + (y - 3 * Real.sqrt 2) ^ 2 = 1

theorem tangent_lines_with_equal_intercepts :
  ∃ L : ℕ, (∀ l, l ∈ L → tangent_to_circle l ∧ equal_intercepts l) ∧ L = 4 :=
  sorry

-- Definitions for tangent_to_circle and equal_intercepts can be added properly.
-- tangent_to_circle should capture the property of being tangent to the circle.
-- equal_intercepts should capture the property of having equal intercepts on x and y axes.

end tangent_lines_with_equal_intercepts_l199_199443


namespace anne_distance_l199_199855
  
theorem anne_distance (S T : ℕ) (H1 : S = 2) (H2 : T = 3) : S * T = 6 := by
  -- Given that speed S = 2 miles/hour and time T = 3 hours, we need to show the distance S * T = 6 miles.
  sorry

end anne_distance_l199_199855


namespace jonas_needs_35_pairs_of_socks_l199_199494

def JonasWardrobeItems (socks_pairs shoes_pairs pants_items tshirts : ℕ) : ℕ :=
  2 * socks_pairs + 2 * shoes_pairs + pants_items + tshirts

def itemsNeededToDouble (initial_items : ℕ) : ℕ :=
  2 * initial_items - initial_items

theorem jonas_needs_35_pairs_of_socks (socks_pairs : ℕ) 
                                      (shoes_pairs : ℕ) 
                                      (pants_items : ℕ) 
                                      (tshirts : ℕ) 
                                      (final_socks_pairs : ℕ) 
                                      (initial_items : ℕ := JonasWardrobeItems socks_pairs shoes_pairs pants_items tshirts) 
                                      (needed_items : ℕ := itemsNeededToDouble initial_items) 
                                      (needed_pairs_of_socks := needed_items / 2) : 
                                      final_socks_pairs = 35 :=
by
  sorry

end jonas_needs_35_pairs_of_socks_l199_199494


namespace smallest_n_is_7_l199_199742

noncomputable def smallest_n : ℕ :=
  Inf {n : ℕ | ∃ (x : fin n → ℝ), (∑ i, x i = 500) ∧ (∑ i, (x i)^4 = 256000)}

theorem smallest_n_is_7 : smallest_n = 7 := 
by
  sorry

end smallest_n_is_7_l199_199742


namespace probability_x_lt_y_l199_199148

noncomputable def rectangle_vertices := [(0, 0), (4, 0), (4, 3), (0, 3)]

theorem probability_x_lt_y :
  let area_triangle := (1 / 2) * 3 * 3
  let area_rectangle := 4 * 3
  let probability := area_triangle / area_rectangle
  probability = (3 / 8) := 
by
  sorry

end probability_x_lt_y_l199_199148


namespace token_exits_at_A2_l199_199868

-- Define the initial grid state, positions, and movement rules.

structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (initial_position : ℕ × ℕ) -- (row, col)
  (direction : ℕ × ℕ → ℕ × ℕ) -- movement arrow mapping from one cell to the next

constant change_direction : Grid → ℕ × ℕ → Grid -- function to change the direction after each move

-- Define the specific condition for our problem
def init_grid : Grid :=
  { rows := 4,
    cols := 4,
    initial_position := (3, 1), -- Lets assume (3, 1) corresponds to C2
    direction := λ cell, -- Based on the given path in the solution, we need to define this
      match cell with
      | (3, 1) => (3, 2)
      | (3, 2) => (2, 2)
      | (2, 2) => (1, 2)
      | (1, 2) => (1, 3)
      | (1, 3) => (2, 3)
      | (2, 3) => (2, 2)
      | (2, 2) => (3, 2)
      | (3, 2) => (4, 2)
      | (4, 2) => (4, 1)
      | (4, 1) => (3, 1)
      | (3, 1) => (3, 0)
      | (3, 0) => (2, 0)
      | (2, 0) => (1, 0)
      | (1, 0) => (1, 1)
      | _ => (0, 0) -- Default for the end of movement
      end }

-- Function to check if a cell is out of the grid bounds
def is_outside_grid (g : Grid) (pos : ℕ × ℕ) : Prop :=
  pos.1 < 1 ∨ pos.1 > g.rows ∨ pos.2 < 1 ∨ pos.2 > g.cols

-- Prove that the token exits at the specific cell
theorem token_exits_at_A2 (g : Grid) :
  g = init_grid →
  let mut pos := g.initial_position
  let move := g.direction pos
  -- Simulate the moves according to the sequence in the solution
  is_outside_grid g move :=
by
  intros h
  simp [init_grid, is_outside_grid]
  sorry -- Skip proof details for the sake of this example

end token_exits_at_A2_l199_199868


namespace min_moves_to_target_l199_199485

def initial_tuplets : List (Fin 31 → ℕ) :=
  List.init 31 (λ i, (λ j, if j = i then 1 else 0))

def target_tuplets : List (Fin 31 → ℕ) :=
  List.init 31 (λ i, (λ j, if j = i then 0 else 1))

noncomputable def min_moves (initial : List (Fin 31 → ℕ)) (target : List (Fin 31 → ℕ)) : ℕ :=
  87

theorem min_moves_to_target :
  ∀ initial ∈ initial_tuplets,
    ∀ target ∈ target_tuplets,
    min_moves initial_tuplets target_tuplets = 87 :=
by sorry

end min_moves_to_target_l199_199485


namespace data_set_properties_l199_199790

open Finset
open Real

theorem data_set_properties :
  let data := {4, 5, 12, 7, 11, 9, 8} in
  let sorted_data := sort data in
  let n := sorted_data.card in
  let mean := (sorted_data.sum : ℝ) / n in
  let median := sorted_data.val[n / 2] in
  let variance := (sorted_data.sum (λ x, (x - mean) ^ 2) / n) in
  mean = 8 ∧ median = 8 ∧ variance = 52 / 7 := 
by {
  sorry
}

end data_set_properties_l199_199790


namespace time_saved_correct_l199_199034

-- Define the conditions as constants
def section1_problems : Nat := 20
def section2_problems : Nat := 15

def time_with_calc_sec1 : Nat := 3
def time_without_calc_sec1 : Nat := 8

def time_with_calc_sec2 : Nat := 5
def time_without_calc_sec2 : Nat := 10

-- Calculate the total times
def total_time_with_calc : Nat :=
  (section1_problems * time_with_calc_sec1) +
  (section2_problems * time_with_calc_sec2)

def total_time_without_calc : Nat :=
  (section1_problems * time_without_calc_sec1) +
  (section2_problems * time_without_calc_sec2)

-- The time saved using a calculator
def time_saved : Nat :=
  total_time_without_calc - total_time_with_calc

-- State the proof problem
theorem time_saved_correct :
  time_saved = 175 := by
  sorry

end time_saved_correct_l199_199034


namespace arccos_half_eq_pi_over_three_l199_199206

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199206


namespace range_of_a_l199_199974

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + (a^2 + 1) * x + a - 2

theorem range_of_a (a : ℝ) :
  (f a 1 < 0) ∧ (f a (-1) < 0) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l199_199974


namespace arccos_half_eq_pi_div_3_l199_199268

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199268


namespace range_m_for_P_inter_not_Q_l199_199131

def P (m : ℝ) : Prop := 2 ≤ m ∧ m ≤ 8

def has_max_and_min_value (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' x₁ = 0 ∧ f' x₂ = 0

def Q (m : ℝ) : Prop := has_max_and_min_value (λ x, x^3 + m * x^2 + (m + 6) * x + 1)

theorem range_m_for_P_inter_not_Q :
  ∀ (m : ℝ), (P m ∧ ¬ (Q m)) ↔ (m ≥ 2 ∧ m ≤ 6) :=
by
  sorry

end range_m_for_P_inter_not_Q_l199_199131


namespace milk_left_l199_199918

theorem milk_left (milk_initial : ℚ) (milk_given : ℚ) (milk_left_expected : ℚ) : 
  milk_initial = 8 → 
  milk_given = 18/7 → 
  milk_left_expected = 38/7 → 
  milk_initial - milk_given = milk_left_expected :=
by
  intros h_initial h_given h_expected
  rw [h_initial, h_given] -- Replace milk_initial with 8 and milk_given with 18/7
  norm_num -- Perform the arithmetic operations
  assumption -- Conclude using h_expected

end milk_left_l199_199918


namespace arccos_half_is_pi_div_three_l199_199258

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199258


namespace daughter_and_child_weight_l199_199589

variables (M D C : ℝ)

-- Conditions
def condition1 : Prop := M + D + C = 160
def condition2 : Prop := D = 40
def condition3 : Prop := C = (1/5) * M

-- Goal (Question)
def goal : Prop := D + C = 60

theorem daughter_and_child_weight
  (h1 : condition1 M D C)
  (h2 : condition2 D)
  (h3 : condition3 M C) : goal D C :=
by
  sorry

end daughter_and_child_weight_l199_199589


namespace arccos_half_eq_pi_div_three_l199_199244

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199244


namespace value_v3_at_1_horners_method_l199_199098

def f (x : ℝ) : ℝ := 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem value_v3_at_1_horners_method :
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  v3 = 7.9 :=
by
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  exact sorry

end value_v3_at_1_horners_method_l199_199098


namespace circle_equation_center_2_0_origin_l199_199957

theorem circle_equation_center_2_0_origin :
  ∃ r : ℝ, (∀ x y : ℝ, ((x - 2)^2 + y^2 = r^2) ↔ (center (2,0) ∧ passes_through (0,0) r)) :=
begin
  let center := (2, 0),
  let passes_through := (0,0),
  existsi 2, -- This radius satisfies the condition
  sorry
end

end circle_equation_center_2_0_origin_l199_199957


namespace find_exponent_M_l199_199652

theorem find_exponent_M (M : ℕ) : (32^4) * (4^6) = 2^M → M = 32 := by
  sorry

end find_exponent_M_l199_199652


namespace cofactor_of_3_is_minus_68_l199_199420

theorem cofactor_of_3_is_minus_68 : 
  let M := ![![8, 1, 6], ![3, 5, 7], ![4, 9, 2]]
  algebraic_complement M 2 1 = -68 :=
by 
  sorry

end cofactor_of_3_is_minus_68_l199_199420


namespace unique_prime_p_l199_199006

def f (x : ℤ) : ℤ := x^3 + 7 * x^2 + 9 * x + 10

theorem unique_prime_p (p : ℕ) (hp : p = 5 ∨ p = 7 ∨ p = 11 ∨ p = 13 ∨ p = 17) :
  (∀ a b : ℤ, f a ≡ f b [ZMOD p] → a ≡ b [ZMOD p]) ↔ p = 11 :=
by
  sorry

end unique_prime_p_l199_199006


namespace worker_followed_instructions_l199_199874

def initial_trees (grid_size : ℕ) : ℕ := grid_size * grid_size

noncomputable def rows_of_trees (rows left each_row : ℕ) : ℕ := rows * each_row

theorem worker_followed_instructions :
  initial_trees 7 = 49 →
  rows_of_trees 5 20 4 = 20 →
  rows_of_trees 5 10 4 = 39 →
  (∃ T : Finset (Fin 7 × Fin 7), T.card = 10) :=
by
  sorry

end worker_followed_instructions_l199_199874


namespace bill_earnings_l199_199728

theorem bill_earnings
  (milk_total : ℕ)
  (fraction : ℚ)
  (milk_to_butter_ratio : ℕ)
  (milk_to_sour_cream_ratio : ℕ)
  (butter_price_per_gallon : ℚ)
  (sour_cream_price_per_gallon : ℚ)
  (whole_milk_price_per_gallon : ℚ)
  (milk_for_butter : ℚ)
  (milk_for_sour_cream : ℚ)
  (remaining_milk : ℚ)
  (total_earnings : ℚ) :
  milk_total = 16 →
  fraction = 1/4 →
  milk_to_butter_ratio = 4 →
  milk_to_sour_cream_ratio = 2 →
  butter_price_per_gallon = 5 →
  sour_cream_price_per_gallon = 6 →
  whole_milk_price_per_gallon = 3 →
  milk_for_butter = milk_total * fraction / milk_to_butter_ratio →
  milk_for_sour_cream = milk_total * fraction / milk_to_sour_cream_ratio →
  remaining_milk = milk_total - 2 * (milk_total * fraction) →
  total_earnings = (remaining_milk * whole_milk_price_per_gallon) + 
                   (milk_for_sour_cream * sour_cream_price_per_gallon) + 
                   (milk_for_butter * butter_price_per_gallon) →
  total_earnings = 41 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end bill_earnings_l199_199728


namespace min_and_max_f_l199_199588

noncomputable def f (x : ℝ) : ℝ := -2 * x + 1

theorem min_and_max_f :
  (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≥ -9) ∧ (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ 1) :=
by
  sorry

end min_and_max_f_l199_199588


namespace base_b_addition_proof_l199_199765

theorem base_b_addition_proof : 
  ∃ b : ℕ, b = 10 ∧ bdigit_sum (list.reverse [4, 6, 3, 7, 8]) b + 
                      bdigit_sum (list.reverse [7, 1, 4, 2, 9]) b = 
                      bdigit_sum (list.reverse [1, 7, 8, 5, 8, 1]) b :=
by
  sorry

/--
 Helper function to convert a list of digits and a base into the corresponding natural number.
-/
def bdigit_sum (digits : list ℕ) (b : ℕ) : ℕ :=
  digits.foldr (λ d acc, d + b * acc) 0

end base_b_addition_proof_l199_199765


namespace extreme_value_of_f_l199_199822

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 * (f' 1)

noncomputable def f' (x : ℝ) : ℝ := (1 / x) + 2 * x * (f' 1)

theorem extreme_value_of_f :
  ∃ x : ℝ, f (Real.sqrt 2 / 2) = -1 / 2 - (Real.log 2) / 2 :=
sorry

end extreme_value_of_f_l199_199822


namespace weight_of_replaced_person_l199_199576

theorem weight_of_replaced_person :
  (∃ avg_increase : ℝ, avg_increase = 2.5) →
  (∃ new_person_weight : ℝ, new_person_weight = 95) →
  (∃ old_person_weight : ℝ, old_person_weight = 75) :=
by
  intros h1 h2
  use 75
  sorry

end weight_of_replaced_person_l199_199576


namespace sum_distances_constant_l199_199531

-- Define a convex polygon with equal sides and angles
structure RegularConvexPolygon (n : ℕ) :=
(length : ℝ)  -- Length of each side
(angle : ℝ)   -- Measure of each interior angle
(area : ℝ)    -- Area of the polygon
(property : length > 0 ∧ angle > 0 ∧ area > 0)

-- Define a function to calculate the sum of distances from a point to the sides
noncomputable def sum_distances (poly : RegularConvexPolygon n) 
  (distances : Fin n → ℝ) : ℝ := distances.sum

-- State the theorem to prove
theorem sum_distances_constant {n : ℕ}
  (poly : RegularConvexPolygon n)
  (distances : Fin n → ℝ) :
  ∃ c : ℝ, ∀ (p : Fin n → ℝ), sum_distances poly p = c := 
sorry

end sum_distances_constant_l199_199531


namespace eight_digit_numbers_cyclic_permutation_l199_199684

-- Define the cyclic permutation function
def cyclic_permutation (a : Nat) : Nat :=
  let digits := a.digits 10
  if digits.head = 0 then 0
  else digits.last * 10 ^ (digits.length - 1) + digits.init.foldl (λ acc d => acc * 10 + d) 0

-- Define the function that applies the cyclic permutation 4 times
def P4 (a : Nat) : Nat := cyclic_permutation (cyclic_permutation (cyclic_permutation (cyclic_permutation a)))

-- Prove the number of eight-digit numbers a where P(P(P(P(a)))) = a
theorem eight_digit_numbers_cyclic_permutation :
  {a : Nat // a.digits 10 |>.length = 8 ∧ a.digits 10 |>.head ≠ 0 ∧ P4 a = a}.card = 9^4 := sorry

end eight_digit_numbers_cyclic_permutation_l199_199684


namespace all_numbers_equal_l199_199747

-- Defining the dimension of the table
def n : ℕ := 10

-- Defining the table as a matrix of size n x n with real numbers
def table : matrix (fin n) (fin n) ℝ := sorry

-- Condition 1: In each row, the largest number is underlined
def is_row_max (i : fin n) (j : fin n) : Prop :=
  ∀ k : fin n, table i j ≥ table i k

-- Condition 2: In each column, the smallest number is underlined
def is_col_min (i : fin n) (j : fin n) : Prop :=
  ∀ k : fin n, table i j ≤ table k j

-- Condition 3: Each number is underlined exactly twice (i.e., it is a row max and a col min)
def is_underlined_twice (i : fin n) (j : fin n) : Prop :=
  is_row_max i j ∧ is_col_min i j

-- Theorem Statement: All numbers in the table are equal
theorem all_numbers_equal : ∀ i j k l : fin n, table i j = table k l :=
by
  sorry

end all_numbers_equal_l199_199747


namespace shaded_region_area_l199_199619

noncomputable def area_of_shaded_region (r : ℝ) (oa : ℝ) (ab_length : ℝ) : ℝ :=
  18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4)

theorem shaded_region_area (r : ℝ) (oa : ℝ) (ab_length : ℝ) : 
  r = 3 ∧ oa = 3 * Real.sqrt 2 ∧ ab_length = 6 * Real.sqrt 2 → 
  area_of_shaded_region r oa ab_length = 18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4) :=
by
  intro h
  obtain ⟨hr, hoa, hab⟩ := h
  rw [hr, hoa, hab]
  exact rfl

end shaded_region_area_l199_199619


namespace area_increase_l199_199153

theorem area_increase (original_length original_width new_length : ℝ)
  (h1 : original_length = 20)
  (h2 : original_width = 5)
  (h3 : new_length = original_length + 10) :
  (new_length * original_width - original_length * original_width) = 50 := by
  sorry

end area_increase_l199_199153


namespace find_vidya_age_l199_199625

theorem find_vidya_age (V M : ℕ) (h1: M = 3 * V + 5) (h2: M = 44) : V = 13 :=
by {
  sorry
}

end find_vidya_age_l199_199625


namespace monotonic_intervals_solve_inequality_range_of_a_l199_199819

open Function
open Real

noncomputable def f (x a : ℝ) : ℝ := x * |2 * x - a|
noncomputable def g (x a : ℝ) : ℝ := (x^2 - a) / (x - 1)

-- Statement for intervals of monotonic increase
theorem monotonic_intervals (a : ℝ) :
  ∃ I1 I2 : Set ℝ, (a < 0 → I1 = Ioo (-∞) (a / 2) ∧ I2 = Ioo (a / 4) +∞) ∧
                 (a > 0 → I1 = Ioo (-∞) (a / 4) ∧ I2 = Ioo (a / 2) +∞) ∧
                 (a = 0 → I1 = Ioi -∞ ∧ I2 = Ioi -∞) := sorry

-- Statement for inequality solution if a < 0
theorem solve_inequality (a x : ℝ) (h : a < 0) :
  (a ≥ -8 → x ≥ (a - sqrt (a^2 - 8 * a)) / 4) ∧
  (a < -8 → (x ≥ (a - sqrt (a^2 - 8 * a)) / 4 ∧ x ≤ (a - sqrt (a^2 + 8 * a)) / 4)
            ∨ x ≥ (a + sqrt (a^2 + 8 * a)) / 4) := sorry

-- Statement for range of values for a
theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 12) :
  (∀ t : ℝ, t ∈ Icc (3:ℝ) 5 → ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ f y1 a = g y1 a ∧ f y2 a = g y2 a) →
  97 / 13 ≤ a ∧ a < 9 := sorry

end monotonic_intervals_solve_inequality_range_of_a_l199_199819


namespace algebra_expression_value_l199_199395

theorem algebra_expression_value (a b c : ℝ) (h1 : a - b = 3) (h2 : b + c = -5) : 
  ac - bc + a^2 - ab = -6 := by
  sorry

end algebra_expression_value_l199_199395


namespace arccos_of_half_eq_pi_over_three_l199_199304

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199304


namespace projection_of_a_onto_b_l199_199997

open Real

variables (i j : ℝ) -- to represent the standard basis vectors
noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (3, 4)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def projection (a b : ℝ × ℝ) : ℝ := dot_product a b / magnitude b

theorem projection_of_a_onto_b : projection a b = 11 / 5 := by
  sorry

end projection_of_a_onto_b_l199_199997


namespace triangle_congruence_example_l199_199780

variable {A B C : Type}
variable (A' B' C' : Type)

def triangle (A B C : Type) : Prop := true

def congruent (t1 t2 : Prop) : Prop := true

variable (P : ℕ)

def perimeter (t : Prop) (p : ℕ) : Prop := true

def length (a b : Type) (l : ℕ) : Prop := true

theorem triangle_congruence_example :
  ∀ (A B C A' B' C' : Type) (h_cong : congruent (triangle A B C) (triangle A' B' C'))
    (h_perimeter : perimeter (triangle A B C) 20)
    (h_AB : length A B 8)
    (h_BC : length B C 5),
    length A C 7 :=
by sorry

end triangle_congruence_example_l199_199780


namespace arccos_half_eq_pi_div_three_l199_199340

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199340


namespace fruits_harvested_daily_total_l199_199486

theorem fruits_harvested_daily_total :
  let apples_kg_per_section := 450
  let apples_sections := 8

  let oranges_kg_per_crate := 8
  let oranges_crates_per_section := 60
  let oranges_sections := 10

  let peaches_kg_per_sack := 12
  let peaches_sacks_per_section := 55
  let peaches_sections := 3

  let cherries_kg_per_basket := 3.5
  let cherries_baskets_per_field := 50
  let cherries_fields := 5

  let total_apples := apples_kg_per_section * apples_sections
  let total_oranges := oranges_kg_per_crate * oranges_crates_per_section * oranges_sections
  let total_peaches := peaches_kg_per_sack * peaches_sacks_per_section * peaches_sections
  let total_cherries := cherries_kg_per_basket * cherries_baskets_per_field * cherries_fields in

  total_apples + total_oranges + total_peaches + total_cherries = 11255 :=
by 
  sorry

end fruits_harvested_daily_total_l199_199486


namespace original_number_recovery_l199_199934

theorem original_number_recovery (N : ℕ) (hN : 100 ≤ N ∧ N ≤ 999) :
  let Q := (((1001 * N + 3514) / 7 + 3524) / 11 + 3534) / 13 in
  N = Q - 300 :=
by {
  sorry
}

end original_number_recovery_l199_199934


namespace projection_onto_plane_Q_l199_199901

variables {R : Type} [LinearOrderedField R]

def projection (v n : R × R × R) : R × R × R :=
  let k := (v.1 * n.1 + v.2 * n.2 + v.3 * n.3) / (n.1 * n.1 + n.2 * n.2 + n.3 * n.3)
  (k * n.1, k * n.2, k * n.3)

theorem projection_onto_plane_Q :
  let Q := (λ x : R × R × R, x.1 - x.2 + (5 / 3) * x.3 = 0)
  let v1 := (7, 4, 7) : R × R × R
  let p1 := (4, 7, 2) : R × R × R
  let v2 := (5, 1, 8) : R × R × R
  v1 - (p1 : ℝ × ℝ × ℝ) = (typical 
  have h : Q v1 ∧ Q p1 from ⟨rfl, rfl⟩
  have n := (1, -1, 5/3) : R × R × R
  projection v2 n = (7/35, 343/35, 98/21) :=
sorry

end projection_onto_plane_Q_l199_199901


namespace area_of_closed_figure_l199_199063

noncomputable def area_closed_figure : ℝ :=
∫ y in (1/2)..2, 1 / y

theorem area_of_closed_figure :
  area_closed_figure = 2 * real.log 2 :=
by
  sorry

end area_of_closed_figure_l199_199063


namespace second_bucket_capacity_l199_199138

-- Define the initial conditions as given in the problem.
def tank_capacity : ℕ := 48
def bucket1_capacity : ℕ := 4

-- Define the number of times the 4-liter bucket is used.
def bucket1_uses : ℕ := tank_capacity / bucket1_capacity

-- Define a condition related to bucket uses.
def buckets_use_relation (x : ℕ) : Prop :=
  bucket1_uses = (tank_capacity / x) - 4

-- Formulate the theorem that states the capacity of the second bucket.
theorem second_bucket_capacity (x : ℕ) (h : buckets_use_relation x) : x = 3 :=
by {
  sorry
}

end second_bucket_capacity_l199_199138


namespace john_bench_press_l199_199493

theorem john_bench_press (initial_weight : ℝ) (decrease_pct : ℝ) (training_factor : ℝ) (final_weight : ℝ) :
  initial_weight = 500 ∧ decrease_pct = 0.80 ∧ training_factor = 3 ∧ final_weight = 
  (initial_weight * (1 - decrease_pct)) * training_factor → final_weight = 300 := 
by
  intros h
  cases h with i_d t_f
  cases i_d with i_weight d_pct
  cases t_f with d_pct t_fact
  cases t_fact with t_factor f_weight
  sorry

end john_bench_press_l199_199493


namespace replace_star_with_2x_l199_199554

theorem replace_star_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_star_with_2x_l199_199554


namespace area_of_BOC_l199_199483

variable (a b c S : ℝ)

-- Given conditions
def triangle_ABC (BC AC AB : ℝ) (S : ℝ) := BC = a ∧ AC = b ∧ AB = c ∧ area_ABC S

-- Definitions related to the problem
def area_ABC (S : ℝ) : Prop := true  -- Assume this represents the condition that the area of the triangle is S

def area_BOC (a b c S : ℝ) := (a * S) / (a + b + c)

theorem area_of_BOC (BC AC AB S : ℝ) (h : triangle_ABC BC AC AB S) :
  area_BOC a b c S = (a * S) / (a + b + c) :=
sorry

end area_of_BOC_l199_199483


namespace equilateral_pentagon_CGJHK_l199_199999

theorem equilateral_pentagon_CGJHK (GH HJ CJ: ℝ) (GH_HJ_perp: GH = HJ ∧ GH ≠ 0 ∧ HJ ≠ 0) (pentagon_sym_jj' : Symmetric_pentagon JJ' CGHJK) : 
  CJ = (Real.sqrt 7 - 1) / 6 :=
by
  sorry

end equilateral_pentagon_CGJHK_l199_199999


namespace expansion_coefficient_l199_199415

theorem expansion_coefficient (a : ℤ) : 
  (∃ (a : ℤ), (∃ (coeff : ℤ), coeff = -35 ∧ (∑ i in Finset.range 8, a ^ i * (nat.choose 7 i) * x ^ (7 - i)) = coeff * x^4)) → a = -1 :=
by sorry

end expansion_coefficient_l199_199415


namespace birgit_time_to_travel_8km_l199_199057

theorem birgit_time_to_travel_8km
  (total_hours : ℝ)
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (distance_to_travel : ℝ)
  (total_minutes := total_hours * 60)
  (average_speed := total_minutes / total_distance)
  (birgit_speed := average_speed - speed_difference) :
  total_hours = 3.5 →
  total_distance = 21 →
  speed_difference = 4 →
  distance_to_travel = 8 →
  (birgit_speed * distance_to_travel) = 48 :=
by
  sorry

end birgit_time_to_travel_8km_l199_199057


namespace arccos_one_half_is_pi_div_three_l199_199192

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199192


namespace james_has_12_pairs_of_pants_l199_199489

noncomputable def pairs_of_pants (P : ℕ) : Prop :=
  let cost_per_shirt := 1.5 * 30
  let cost_shirts := 10 * cost_per_shirt
  let cost_per_pants := (1.5 * 2) * 30
  let total_cost := cost_shirts + (P * cost_per_pants)
  total_cost = 1530

theorem james_has_12_pairs_of_pants : pairs_of_pants 12 :=
by
  sorry

end james_has_12_pairs_of_pants_l199_199489


namespace arccos_half_eq_pi_div_three_l199_199339

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199339


namespace ellipse_equation_line_intersection_l199_199501

variable {b k : ℝ} (P A B A' : ℝ × ℝ)
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / b = 1

def foci1 : ℝ × ℝ := (-sqrt (4 - b), 0)
def foci2 : ℝ × ℝ := (sqrt (4 - b), 0)

def dot_product (u v : ℝ × ℝ) :=
  u.1 * v.1 + u.2 * v.2 

def line_eq (x y : ℝ) : Prop := x = k * y - 1

def symmetric_about_x (A A' : ℝ × ℝ) :=
  A' = (A.1, -A.2)

noncomputable def max_dot_product : ℝ :=
  max (dot_product (foci1) (P)) (dot_product (foci2) (P))

theorem ellipse_equation :
  max_dot_product = 1 → b = 1 := sorry

theorem line_intersection (h1 : ellipse P.1 P.2)
  (h2 : line_eq A.1 A.2) (h3 : line_eq B.1 B.2)
  (h4 : symmetric_about_x A A') :
  ∃ x, line_eq x 0 ∧ x = -4 := sorry

end ellipse_equation_line_intersection_l199_199501


namespace arccos_one_half_is_pi_div_three_l199_199202

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199202


namespace no_real_solution_for_ap_l199_199356

theorem no_real_solution_for_ap : 
  (¬∃ (a b : ℝ), 15, a, b, a * b form_an_arithmetic_progression) :=
sorry

end no_real_solution_for_ap_l199_199356


namespace arccos_half_eq_pi_over_three_l199_199214

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199214


namespace arccos_of_half_eq_pi_over_three_l199_199303

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199303


namespace inequality_sum_lt_two_thirds_l199_199896

-- Lean statement for the problem
theorem inequality_sum_lt_two_thirds 
  (n : ℕ) (hn : 0 < n) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 < x i) :
  ∑ i in Finset.range n, (x i * (1 - (∑ j in Finset.range (i + 1), x j)^2)) < 2 / 3 := 
sorry

end inequality_sum_lt_two_thirds_l199_199896


namespace dirichlet_bvp_solution_l199_199948

section DirichletBVP
open Real

-- Define the function u(r, φ) 
noncomputable def u (r φ : ℝ) : ℝ :=
  (r^5 / 24 - 107/40 * r + 79/30 * r⁻¹) * cos φ +
  (1/17 * r + 16/17 * r⁻¹) * cos (2 * φ) +
  (16/195 * r - 16/195 * r⁻¹) * sin (3 * φ)

-- Condition 1: \Delta u = r^3 \cos φ for 1 < r < 2
axiom delta_u_eq_r3_cos_phi (r φ : ℝ) (hr : 1 < r) (hr2 : r < 2) :
  (real.laplacian (u r φ)) = r^3 * cos φ

-- Condition 2: u|_{r=1} = \cos 2φ
axiom u_at_r_1 (φ : ℝ) : u 1 φ = cos (2 * φ)

-- Condition 3: ∂u/∂r |_{r=2} = sin 3φ
axiom dudr_at_r_2 (φ : ℝ) : 
  (deriv (λ r, u r φ)) 2 = sin (3 * φ)

-- The conclusion: Prove that the function u satisfies these conditions
theorem dirichlet_bvp_solution :
  (∀ r φ, (1 < r ∧ r < 2) → (real.laplacian (u r φ) = r^3 * cos φ)) ∧ 
  (∀ φ, u 1 φ = cos (2 * φ)) ∧ 
  (∀ φ, (deriv (λ r, u r φ)) 2 = sin (3 * φ)) :=
by
  split
  · intros r φ hr
    exact delta_u_eq_r3_cos_phi r φ hr.1 hr.2
  · intro φ
    exact u_at_r_1 φ
  · intro φ
    exact dudr_at_r_2 φ

end dirichlet_bvp_solution_l199_199948


namespace distance_of_each_race_l199_199996

theorem distance_of_each_race (d : ℝ) : 
  (∃ (d : ℝ), 
    let lake_speed := 3 
    let ocean_speed := 2.5 
    let num_races := 10 
    let total_time := 11
    let num_lake_races := num_races / 2
    let num_ocean_races := num_races / 2
    (num_lake_races * (d / lake_speed) + num_ocean_races * (d / ocean_speed) = total_time)) →
  d = 3 :=
sorry

end distance_of_each_race_l199_199996


namespace inequality_proof_l199_199529

theorem inequality_proof (α : ℝ) (n : ℕ) (x : ℕ → ℝ) (hα : α ≤ 1) (hx : ∀ i, i ≤ n → x i > 0) (hx2 : ∀ i j, i ≤ j → j ≤ n → x i ≥ x j) :
  (1 + (Finset.sum (Finset.range(n+1)) (λ i, x i))) ^ α ≤ 1 + Finset.sum (Finset.range(n+1)) (λ i, ((i + 1) ^ (α - 1)) * (x i ^ α)) :=
by
  sorry

end inequality_proof_l199_199529


namespace rest_days_coincide_l199_199167

theorem rest_days_coincide :
  let al_schedule := λ n : ℕ, n % 3 = 2
  let barb_schedule := λ n : ℕ, n % 7 = 5 ∨ n % 7 = 6
  let common_rest_days := λ n : ℕ, al_schedule n ∧ barb_schedule n
  (finset.filter common_rest_days (finset.range 1000)).card = 94 := by sorry

end rest_days_coincide_l199_199167


namespace train_pass_platform_in_correct_time_l199_199676

def length_of_train : ℝ := 2500
def time_to_cross_tree : ℝ := 90
def length_of_platform : ℝ := 1500

noncomputable def speed_of_train : ℝ := length_of_train / time_to_cross_tree
noncomputable def total_distance_to_cover : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance_to_cover / speed_of_train

theorem train_pass_platform_in_correct_time :
  abs (time_to_pass_platform - 143.88) < 0.01 :=
sorry

end train_pass_platform_in_correct_time_l199_199676


namespace rancher_feed_cost_l199_199699

theorem rancher_feed_cost (num_sheep : ℕ) (num_cattle : ℕ) (cow_grass_consumption : ℕ) (sheep_grass_consumption : ℕ)
  (bag_cost : ℕ) (cow_feed_duration : ℕ) (sheep_feed_duration : ℕ) (total_grass : ℕ) (months_in_year : ℕ) :
  num_sheep = 8 →
  num_cattle = 5 →
  cow_grass_consumption = 2 →
  sheep_grass_consumption = 1 →
  bag_cost = 10 →
  cow_feed_duration = 1 →
  sheep_feed_duration = 2 →
  total_grass = 144 →
  months_in_year = 12 →
  (num_cattle * cow_grass_consumption + num_sheep * sheep_grass_consumption) * (months_in_year - (total_grass / (num_cattle * cow_grass_consumption + num_sheep * sheep_grass_consumption))) * bag_cost / cow_feed_duration + (num_sheep * sheep_feed_duration⁻¹ * bag_cost * (months_in_year - (total_grass / (num_cattle * cow_grass_consumption + num_sheep * sheep_grass_consumption)) * cow_feed_duration * (num_cattle / cow_duration + num_sheep * (sheep_duration / months_in_year))) = 360 :=
by sorry

end rancher_feed_cost_l199_199699


namespace sequence_100th_term_eq_981_l199_199586

-- Conditions: The sequence consists of positive integers that are powers of 3 
-- or sums of distinct powers of 3.
def is_valid_sequence (n : ℕ) : Prop :=
  ∃ (binary_representation : ℕ), binary_representation < 2^(n - 1) ∧ 
    n = ∑ i in finset.range (nat.bitsize binary_representation), 
          (if test_bit binary_representation i then 3^i else 0)

-- Proving the 100th term is equal to 981.
theorem sequence_100th_term_eq_981 : 
  ∃ n, is_valid_sequence n ∧  nat.sublist_nth n = 100
       := 981 :=
sorry

end sequence_100th_term_eq_981_l199_199586


namespace range_of_a_l199_199831

noncomputable def is_decreasing (a : ℝ) : Prop :=
∀ n : ℕ, 0 < n → n ≤ 6 → (1 - 3 * a) * n + 10 * a > (1 - 3 * a) * (n + 1) + 10 * a ∧ 0 < a ∧ a < 1 ∧ ((1 - 3 * a) * 6 + 10 * a > 1)

theorem range_of_a (a : ℝ) : is_decreasing a ↔ (1/3 < a ∧ a < 5/8) :=
sorry

end range_of_a_l199_199831


namespace largest_neg_int_property_l199_199853

theorem largest_neg_int_property (x : ℤ) (h : x = -1) : -(-(-x)) = 1 :=
by
  rw [h]
  simp [neg_neg]
  sorry

end largest_neg_int_property_l199_199853


namespace arccos_half_is_pi_div_three_l199_199259

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199259


namespace parabolas_intersect_diff_l199_199592

theorem parabolas_intersect_diff (a b c d : ℝ) (h1 : c ≥ a)
  (h2 : b = 3 * a^2 - 6 * a + 3)
  (h3 : d = 3 * c^2 - 6 * c + 3)
  (h4 : b = -2 * a^2 - 4 * a + 6)
  (h5 : d = -2 * c^2 - 4 * c + 6) :
  c - a = 1.6 :=
sorry

end parabolas_intersect_diff_l199_199592


namespace prove_bd_greater_dc_l199_199938

variables (A B C D M : Type)
variables [ordered_comm_group A] [ordered_comm_group B] [ordered_comm_group C] [ordered_comm_group D] [ordered_comm_group M]
variables (a b c d : A) (ab ac bd dc : B) (abcd : Prop)

def convex_quadrilateral (abcd : Prop) : Prop :=
    sorry  -- Definition of convex quadrilateral

theorem prove_bd_greater_dc
    (hab : ab ≥ ac)
    (hconvex : convex_quadrilateral abcd) :
    bd > dc :=
    sorry

end prove_bd_greater_dc_l199_199938


namespace triangle_altitudes_concurrent_l199_199707

theorem triangle_altitudes_concurrent
    (A B C : Type) [TopologicalSpace A] [TopologicalSpace B] [TopologicalSpace C]
    (AB BC CA : A → B → C) :
    ∃ (H : A), (∃ (h₁ h₂ h₃ : Line A), 
    is_altitude h₁ AB ∧ is_altitude h₂ BC ∧ is_altitude h₃ CA ∧ concurrent h₁ h₂ h₃ H) :=
sorry

end triangle_altitudes_concurrent_l199_199707


namespace count_valid_numbers_4_l199_199444

def is_digit_sum_multiple (n : ℕ) (k : ℕ) : Prop :=
  7 * (list.sum (nat.digits 10 n)) = k

def valid_numbers : set ℕ := {n | n < 1000 ∧ 7 * (list.sum (nat.digits 10 n)) = n}

theorem count_valid_numbers_4 : (finset.card (finset.filter (λ n, 7 * (list.sum (nat.digits 10 n)) = n) (finset.range 1000))) = 4 :=
sorry

end count_valid_numbers_4_l199_199444


namespace sum_of_divisors_252_l199_199645

theorem sum_of_divisors_252 :
  let n := 252
  let prime_factors := [2, 2, 3, 3, 7]
  sum_of_divisors n = 728 :=
by
  sorry

end sum_of_divisors_252_l199_199645


namespace speed_at_perigee_l199_199133

-- Define the conditions
def semi_major_axis (a : ℝ) := a > 0
def perigee_distance (a : ℝ) := 0.5 * a
def point_P_distance (a : ℝ) := 0.75 * a
def speed_at_P (v1 : ℝ) := v1 > 0

-- Define what we need to prove
theorem speed_at_perigee (a v1 v2 : ℝ) (h1 : semi_major_axis a) (h2 : speed_at_P v1) :
  v2 = (3 / Real.sqrt 5) * v1 :=
sorry

end speed_at_perigee_l199_199133


namespace same_speed_4_l199_199890

theorem same_speed_4 {x : ℝ} (hx : x ≠ -7)
  (H1 : ∀ (x : ℝ), (x^2 - 7*x - 60)/(x + 7) = x - 12) 
  (H2 : ∀ (x : ℝ), x^3 - 5*x^2 - 14*x + 104 = x - 12) :
  ∃ (speed : ℝ), speed = 4 :=
by
  sorry

end same_speed_4_l199_199890


namespace arccos_one_half_l199_199233

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199233


namespace arccos_half_is_pi_div_three_l199_199255

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199255


namespace probability_sum_five_l199_199114

theorem probability_sum_five (total_outcomes : ℕ) (favorable_outcomes : ℕ):
  total_outcomes = 36 ∧ favorable_outcomes = 4 → 
  (favorable_outcomes.toRat / total_outcomes.toRat = (1 / 9 : ℚ)) :=
by
  sorry

end probability_sum_five_l199_199114


namespace replace_star_with_2x_l199_199547

theorem replace_star_with_2x (c : ℤ) (x : ℤ) :
  (x^3 - 2)^2 + (x^2 + c)^2 = x^6 + x^4 + 4x^2 + 4 ↔ c = 2 * x :=
by
  sorry

end replace_star_with_2x_l199_199547


namespace difference_in_base_seven_l199_199374

theorem difference_in_base_seven : 
  let base := 7
  then (base^3 - 1) - (base^2*(base-1) + base*(base-1) + (base-1))  = 1 :=
by 
  let base := 7
  have h0 : base = 7, by rfl
  calc
    (base^3 - 1) - (base^2*(base-1) + base*(base-1) + (base-1))
      = (7^3 - 7^0) - (7^2 * 6 + 7 * 6 + 6) : by rw h0
  ... = (343 - 1) - (49 * 6 + 7 * 6 + 6) : by norm_num
  ... = (343 - 1) - (294 + 42 + 6) : by norm_num
  ... = 342 - 342 : by norm_num
  ... = 0 : by norm_num
  ... = 0 + 1 : by norm_num
  ... = 1 : by norm_num

end difference_in_base_seven_l199_199374


namespace perimeter_of_ACED_l199_199709

theorem perimeter_of_ACED :
  let D B E A C: Point,
  let s: Length := 4.5,
  let t: Length := 1.5,
  let quadrilateral_perimeter (a b c d: Length) := a + b + c + d,
  is_equilateral_triangle ABC s →
  is_equilateral_triangle DBE t →
  contains_triangle ABC DBE →
  perimeter (remaining_quadrilateral ABC DBE) = 12 :=
begin
  sorry
end

end perimeter_of_ACED_l199_199709


namespace sqrt_diff_eq_l199_199662

theorem sqrt_diff_eq
  (p q : ℝ) 
  (hp : p > 0) 
  (hq1 : 0 ≤ q) 
  (hq2 : q ≤ 5 * p) :
  sqrt (10 * p + 2 * sqrt (25 * p^2 - q^2)) - sqrt (10 * p - 2 * sqrt (25 * p^2 - q^2)) = 2 * sqrt (5 * p - q) := 
sorry

end sqrt_diff_eq_l199_199662


namespace count_four_digit_solutions_l199_199844

theorem count_four_digit_solutions :
  (∃ x : ℕ, 3874 * x + 481 ≡ 1205 [MOD 31] ∧ 1000 ≤ x ∧ x ≤ 9999) →
  (∀ a b c, (3874 * a + 481 ≡ 1205 [MOD 31]) → (1000 ≤ a) ∧ (a ≤ 9999) → (3874 * b + 481 ≡ 1205 [MOD 31]) → (1000 ≤ b) ∧ (b ≤ 9999) → a ≠ b → false) →
  ∀ n : ℕ, ((∃ x : ℕ, 3874 * x + 481 ≡ 1205 [MOD 31] ∧ 1000 ≤ x ∧ x ≤ 9999) → n = 290) :=
by
  sorry

end count_four_digit_solutions_l199_199844


namespace largest_coefficient_term_in_binomial_expansion_l199_199877

theorem largest_coefficient_term_in_binomial_expansion (x : ℝ) :
  (binom_expand (\(\frac{1}{x} - 1 \)^5)).max_coef_term = \(\frac{10}{x^3}\) :=
sorry

end largest_coefficient_term_in_binomial_expansion_l199_199877


namespace determine_a_if_roots_are_prime_l199_199803
open Nat

theorem determine_a_if_roots_are_prime (a x1 x2 : ℕ) (h1 : Prime x1) (h2 : Prime x2) 
  (h_eq : x1^2 - x1 * a + a + 1 = 0) :
  a = 5 :=
by
  -- Placeholder for the proof
  sorry

end determine_a_if_roots_are_prime_l199_199803


namespace volume_of_tetrahedron_is_zero_l199_199382

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

-- Definition of the points as given in the problem
def P1 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci n, fibonacci (n + 1), fibonacci (n + 2))
def P2 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci (n + 3), fibonacci (n + 4), fibonacci (n + 5))
def P3 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci (n + 6), fibonacci (n + 7), fibonacci (n + 8))
def P4 (n : ℕ) : ℝ × ℝ × ℝ := (fibonacci (n + 9), fibonacci (n + 10), fibonacci (n + 11))

-- Function to calculate the volume of the tetrahedron
noncomputable def tetrahedron_volume (v1 v2 v3 v4 : ℝ × ℝ × ℝ) : ℝ :=
  let matrix := λ i j, match (i, j) with
  | (0, 0) => v2.1 - v1.1 | (0, 1) => v2.2 - v1.2 | (0, 2) => v2.3 - v1.3
  | (1, 0) => v3.1 - v1.1 | (1, 1) => v3.2 - v1.2 | (1, 2) => v3.3 - v1.3
  | (2, 0) => v4.1 - v1.1 | (2, 1) => v4.2 - v1.2 | (2, 2) => v4.3 - v1.3
  | _ => 0 end
  (1/6) * (matrix.det).abs

-- Statement of the theorem
theorem volume_of_tetrahedron_is_zero (n : ℕ) : 
  tetrahedron_volume (P1 n) (P2 n) (P3 n) (P4 n) = 0 :=
sorry

end volume_of_tetrahedron_is_zero_l199_199382


namespace greatest_three_digit_number_l199_199105

theorem greatest_three_digit_number : ∃ n : ℕ, n < 1000 ∧ n >= 100 ∧ (n + 1) % 8 = 0 ∧ (n - 4) % 7 = 0 ∧ n = 967 :=
by
  sorry

end greatest_three_digit_number_l199_199105


namespace correct_option_d_l199_199782

-- Definitions for lines, planes, and relationships
variables {Line Plane : Type} [linear_space : LinearSpace Line Plane]
variables (m n : Line) (alpha beta : Plane)

-- Conditions for the problem
def option_d_valid_condition (m_parallel_n : m∥n) (m_parallel_alpha : m∥alpha) (n_perpendicular_beta : n⊥beta) : Prop :=
  alpha⊥beta

-- The theorem statement to be proved
theorem correct_option_d (m_parallel_n : m ∥ n) (m_parallel_alpha : m ∥ alpha) (n_perpendicular_beta : n ⊥ beta) : 
    option_d_valid_condition m_parallel_n m_parallel_alpha n_perpendicular_beta :=
sorry

end correct_option_d_l199_199782


namespace rectangle_new_area_l199_199062

theorem rectangle_new_area (original_area : ℝ) (new_length_factor : ℝ) (new_width_factor : ℝ) 
  (h1 : original_area = 560) (h2 : new_length_factor = 1.2) (h3 : new_width_factor = 0.85) : 
  new_length_factor * new_width_factor * original_area = 571 := 
by 
  sorry

end rectangle_new_area_l199_199062


namespace arccos_half_eq_pi_div_three_l199_199334

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199334


namespace sum_of_angles_outside_pentagon_l199_199147

-- Definitions based on conditions
variable {α β γ δ ε : ℝ}
variable {A B C D E : Point}

-- The problem statement
theorem sum_of_angles_outside_pentagon (h1 : InscribedInCircle A B C D E) 
  (h2 : InscribedAngle α A B C ∧ InscribedAngle β B C D ∧ InscribedAngle γ C D E ∧ InscribedAngle δ D E A ∧ InscribedAngle ε E A B) :
  α + β + γ + δ + ε = 900 :=
sorry

end sum_of_angles_outside_pentagon_l199_199147


namespace arccos_one_half_is_pi_div_three_l199_199190

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199190


namespace intersecting_x_value_l199_199623

theorem intersecting_x_value : 
  (∃ x y : ℝ, y = 3 * x - 17 ∧ 3 * x + y = 103) → 
  (∃ x : ℝ, x = 20) :=
by
  sorry

end intersecting_x_value_l199_199623


namespace sum_of_areas_of_circles_l199_199975

theorem sum_of_areas_of_circles :
  (∑' n : ℕ, π * (9 / 16) ^ n) = π * (16 / 7) :=
by
  sorry

end sum_of_areas_of_circles_l199_199975


namespace largest_circle_radius_l199_199604

noncomputable def center_case2 : Real := 2
noncomputable def radius_case2 : Real := (Real.sqrt 65 + Real.sqrt 97) / 8

theorem largest_circle_radius :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 2 : ℝ)
  let C := (4 : ℝ, 0 : ℝ)
  let D := (1 : ℝ, -2 : ℝ)
   in radius_case2 = 2.23889 :=
by
  sorry

end largest_circle_radius_l199_199604


namespace sequence_sixth_term_l199_199466

theorem sequence_sixth_term (a b c d : ℚ) : 
  (a = 1/4 * (5 + b)) →
  (b = 1/4 * (a + 45)) →
  (45 = 1/4 * (b + c)) →
  (c = 1/4 * (45 + d)) →
  d = 1877 / 3 :=
by
  sorry

end sequence_sixth_term_l199_199466


namespace minimum_rows_required_l199_199688

theorem minimum_rows_required (n : ℕ) : (3 * n * (n + 1)) / 2 ≥ 150 ↔ n ≥ 10 := 
by
  sorry

end minimum_rows_required_l199_199688


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199324

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199324


namespace parallel_lines_slope_l199_199459

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0) ∧ (∀ x y : ℝ, 2 * x + (m + 5) * y - 8 = 0) →
  m = -7 :=
by
  intro H
  sorry

end parallel_lines_slope_l199_199459


namespace translated_graph_symmetric_l199_199961

noncomputable def f (x : ℝ) : ℝ := sorry

theorem translated_graph_symmetric (f : ℝ → ℝ)
  (h_translate : ∀ x, f (x - 1) = e^x)
  (h_symmetric : ∀ x, f x = f (-x)) :
  ∀ x, f x = e^(-x - 1) :=
by
  sorry

end translated_graph_symmetric_l199_199961


namespace arccos_half_eq_pi_div_three_l199_199344

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199344


namespace min_positive_period_of_f_max_value_of_f_l199_199965

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 * Real.tan x + Real.cos (2 * x)

theorem min_positive_period_of_f :
  Function.periodic f π :=
sorry

theorem max_value_of_f :
  ∀ x, f x ≤ sqrt 2 :=
sorry

end min_positive_period_of_f_max_value_of_f_l199_199965


namespace MH_eq_MK_l199_199919

-- Define points and segments in a plane
variables (Point : Type) [MetricSpace Point]
variables (H K B C M : Point)
variables (HK BC : Set Point)

-- Conditions given in the problem
variable (plane : ∀ (x ∈ HK) (y ∈ BC), x ≠ y → ∃ z, z ≠ x ∧ z ≠ y)
variable (midpointM : dist M B = dist M C)
variable (perpendicularBH : ∀ x ∈ HK, dist (dist B x) (dist H x))
variable (perpendicularCK : ∀ x ∈ HK, dist (dist C x) (dist K x))

-- Statement of the theorem to be proved
theorem MH_eq_MK : dist M H = dist M K :=
by
  sorry -- Proof steps would go here

end MH_eq_MK_l199_199919


namespace number_of_k_l199_199768

theorem number_of_k (k : ℕ) : 
  let a := k.factorization 2 
  let b := k.factorization 3 in
  lcm (2^6 * 3^6) (lcm (2^24) (2^a * 3^b)) = 2^24 * 3^12 →
  25 = (finset.range 25).card :=
by {
  intros,
  sorry
}

end number_of_k_l199_199768


namespace y_coordinate_of_intersection_l199_199795

noncomputable def parabola_y (x : ℝ) : ℝ := x^2 / 2

def point_P : ℝ × ℝ := (4, parabola_y 4)
def point_Q : ℝ × ℝ := (-2, parabola_y (-2))

def tangent_slope (x : ℝ) : ℝ := x

def tangent_line (x₀ y₀ m : ℝ) (x : ℝ) : ℝ := y₀ + m * (x - x₀)

def tangent_at_P (x : ℝ) : ℝ := tangent_line 4 (parabola_y 4) (tangent_slope 4) x
def tangent_at_Q (x : ℝ) : ℝ := tangent_line (-2) (parabola_y (-2)) (tangent_slope (-2)) x

theorem y_coordinate_of_intersection :
  let A := (1, tangent_at_P 1) in A.snd = -4 :=
by
  sorry

end y_coordinate_of_intersection_l199_199795


namespace part1_part2_l199_199433

noncomputable def S (A B : ℚ) (n : ℕ) : ℚ := A * n^2 + B * n
def a (A B : ℚ) (n : ℕ) : ℚ := S A B n - S A B (n-1)

theorem part1 (A B : ℚ) (hA : A = 3 / 2) (hB : B = 1 / 2) :
  a A B = λ n, 3 * n - 1 := sorry

noncomputable def b (A B : ℚ) (n : ℕ) : ℚ := 1 / (a A B n * a A B (n + 1))
noncomputable def T (A B : ℚ) (n : ℕ) : ℚ := (Finset.range n).sum (b A B)

theorem part2 (A B : ℚ) (hA : A = 3 / 2) (hB : B = 1 / 2) :
  T A B n = n / (6 * n + 4) := sorry

end part1_part2_l199_199433


namespace largest_unshaded_area_l199_199156

theorem largest_unshaded_area (s : ℝ) (π_approx : ℝ) :
    (let r := s / 2
     let area_square := s^2
     let area_circle := π_approx * r^2
     let area_triangle := (1 / 2) * (s / 2) * (s / 2)
     let unshaded_square := area_square - area_circle
     let unshaded_circle := area_circle - area_triangle
     unshaded_circle) > (unshaded_square) := by
        sorry

end largest_unshaded_area_l199_199156


namespace sign_of_x_minus_y_l199_199854

theorem sign_of_x_minus_y (x y a : ℝ) (h1 : x + y > 0) (h2 : a < 0) (h3 : a * y > 0) : x - y > 0 := 
by 
  sorry

end sign_of_x_minus_y_l199_199854


namespace middle_term_geometric_l199_199502

variables {α β : Type} [LinearOrderedField α] [OrderedRing β]
variables (a_n : ℕ → α) (b_n : ℕ → β) (d : α) (q : β)

-- Assume d ≠ 0
axiom d_nonzero : d ≠ 0

-- Assume q ≠ 1
axiom q_nonone : q ≠ 1

-- Define S_n as the sum of the first n terms of the arithmetic sequence a_n
def S_n (n : ℕ) : α := ∑ i in finset.range n, a_n i

-- Define T_n as the product of the first n terms of the geometric sequence b_n
def T_n (n : ℕ) : β := ∏ i in finset.range n, b_n i

-- Assume S_n = S_(2011 - n) for n < 2011
axiom Sn_symmetry (n : ℕ) (h : n < 2011) : S_n a_n n = S_n a_n (2011 - n)

-- Assume T_n = T_(23 - n) for n < 23
axiom Tn_symmetry (n : ℕ) (h : n < 23) : T_n b_n n = T_n b_n (23 - n)

theorem middle_term_geometric : b_n 12 = 1 :=
sorry

end middle_term_geometric_l199_199502


namespace point_on_curve_l199_199040

-- Define the parametric curve equations
def onCurve (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.sin (2 * θ) ∧ y = Real.cos θ + Real.sin θ

-- Define the general form of the curve
def curveEquation (x y : ℝ) : Prop :=
  y^2 = 1 + x

-- The proof statement
theorem point_on_curve : 
  curveEquation (-3/4) (1/2) ∧ ∃ θ : ℝ, onCurve θ (-3/4) (1/2) :=
by
  sorry

end point_on_curve_l199_199040


namespace find_exponent_M_l199_199653

theorem find_exponent_M (M : ℕ) : (32^4) * (4^6) = 2^M → M = 32 := by
  sorry

end find_exponent_M_l199_199653


namespace f_positive_for_all_x_f_increasing_solution_set_inequality_l199_199348

namespace ProofProblem

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

axiom f_zero_ne_zero : f 0 ≠ 0
axiom f_one_eq_two : f 1 = 2
axiom f_pos_when_pos : ∀ x : ℝ, x > 0 → f x > 1
axiom f_add_mul : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(x) > 0 for all x ∈ ℝ
theorem f_positive_for_all_x : ∀ x : ℝ, f x > 0 := sorry

-- Problem 2: Prove that f(x) is increasing on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 3: Find the solution set of the inequality f(3-2x) > 4
theorem solution_set_inequality : { x : ℝ | f (3 - 2 * x) > 4 } = { x | x < 1 / 2 } := sorry

end ProofProblem

end f_positive_for_all_x_f_increasing_solution_set_inequality_l199_199348


namespace each_child_plays_for_90_minutes_l199_199871

-- Definitions based on the conditions
def total_playing_time : ℕ := 180
def children_playing_at_a_time : ℕ := 3
def total_children : ℕ := 6

-- The proof problem statement
theorem each_child_plays_for_90_minutes :
  (children_playing_at_a_time * total_playing_time) / total_children = 90 := by
  sorry

end each_child_plays_for_90_minutes_l199_199871


namespace arccos_half_eq_pi_div_3_l199_199274

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199274


namespace arccos_half_eq_pi_div_three_l199_199313

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199313


namespace minimize_avg_cost_processing_at_loss_maximize_profit_l199_199140

def processing_cost (x : ℝ) : ℝ := (1/2) * x^2 + 40 * x + 3200
def avg_cost_per_ton (x : ℝ) : ℝ := processing_cost x / x
def selling_price_per_ton : ℝ := 110
def subsidy_option1 : ℝ := 2300
def subsidy_option2 (x : ℝ) : ℝ := 30 * x

-- Prove the various parts of the problem
theorem minimize_avg_cost (x : ℝ) (h1 : 70 ≤ x ∧ x ≤ 100) : 
  avg_cost_per_ton x = (x / 2) + (3200 / x) + 40 → 
  x = 80 → 
  avg_cost_per_ton x = 120 := 
sorry

theorem processing_at_loss (x : ℝ) (h1 : 70 ≤ x ∧ x ≤ 100) : 
  x = 80 →
  selling_price_per_ton < avg_cost_per_ton x →
  true := 
sorry

theorem maximize_profit (x : ℝ) (h1 : 70 ≤ x ∧ x ≤ 100) : 
  let profit1 := selling_price_per_ton * x - processing_cost x + subsidy_option1,
      profit2 := selling_price_per_ton * x - processing_cost x + subsidy_option2 x in 
  (profit1 ≤ profit2) :=
sorry

end minimize_avg_cost_processing_at_loss_maximize_profit_l199_199140


namespace at_least_one_perpendicular_foot_on_side_l199_199512

-- Define the Lean 4 statement.

theorem at_least_one_perpendicular_foot_on_side (n : ℕ) (A : ℕ → Point) (M : Point) 
  (h_convex : convex_polygon A n) (h_inside : inside_polygon A n M) :
  ∃ (i : ℕ), (1 ≤ i ∧ i ≤ n) ∧ foot_on_side A (i % n) ((i+1) % n) M :=
sorry

end at_least_one_perpendicular_foot_on_side_l199_199512


namespace arccos_one_half_l199_199239

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199239


namespace arccos_half_eq_pi_div_three_l199_199288

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199288


namespace geom_seq_problem_l199_199413

variable (a : ℕ → ℝ)

-- Given conditions
axiom geom_seq_pos (n m : ℕ) : (a n) * (a m) > 0
axiom log_condition : log10 (a 3 * a 8 * a 13) = 6

-- Required to prove
theorem geom_seq_problem : a 1 * a 15 = 10000 := by
  sorry

end geom_seq_problem_l199_199413


namespace triple_angle_l199_199617

theorem triple_angle (α : ℝ) : 3 * α = α + α + α := 
by sorry

end triple_angle_l199_199617


namespace find_angle_Z_l199_199921

variables (p q : Line) (X Y Z : Angle)
variables (h1 : parallel p q) 
variables (h2 : mangle X = 100) 
variables (h3 : mangle Y = 130)

theorem find_angle_Z (p q : Line) (X Y Z : Angle) 
  (h1 : parallel p q) 
  (h2 : mangle X = 100)
  (h3 : mangle Y = 130) :
  mangle Z = 130 :=
sorry

end find_angle_Z_l199_199921


namespace cost_of_renting_per_month_l199_199393

namespace RentCarProblem

def cost_new_car_per_month : ℕ := 30
def months_per_year : ℕ := 12
def yearly_difference : ℕ := 120

theorem cost_of_renting_per_month (R : ℕ) :
  (cost_new_car_per_month * months_per_year + yearly_difference) / months_per_year = R → 
  R = 40 :=
by
  sorry

end RentCarProblem

end cost_of_renting_per_month_l199_199393


namespace cone_volume_l199_199696

theorem cone_volume (d : ℝ) (h : ℝ) (π : ℝ) (volume : ℝ) 
  (hd : d = 10) (hh : h = 0.8 * d) (hπ : π = Real.pi) : 
  volume = (200 / 3) * π :=
by
  sorry

end cone_volume_l199_199696


namespace meet_time_first_l199_199135

-- Define the conditions
def track_length := 1800 -- track length in meters
def speed_A_kmph := 36   -- speed of A in kmph
def speed_B_kmph := 54   -- speed of B in kmph

-- Convert speeds from kmph to mps
def speed_A_mps := speed_A_kmph * 1000 / 3600 -- speed of A in meters per second
def speed_B_mps := speed_B_kmph * 1000 / 3600 -- speed of B in meters per second

-- Calculate lap times in seconds
def lap_time_A := track_length / speed_A_mps
def lap_time_B := track_length / speed_B_mps

-- Problem statement
theorem meet_time_first :
  Nat.lcm (Int.natAbs lap_time_A) (Int.natAbs lap_time_B) = 360 := by
  sorry

end meet_time_first_l199_199135


namespace monotonically_increasing_interval_l199_199825

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem monotonically_increasing_interval :
  ∀ x, 0 < x ∧ x ≤ π / 6 → ∀ y, x ≤ y ∧ y < π / 2 → f x ≤ f y :=
by
  intro x hx y hy
  sorry

end monotonically_increasing_interval_l199_199825


namespace scientific_notation_correct_l199_199878

def n : ℝ := 12910000

theorem scientific_notation_correct : n = 1.291 * 10^7 := 
by
  sorry

end scientific_notation_correct_l199_199878


namespace ramu_profit_percent_correct_l199_199942

def cost_of_car : ℝ := 25000
def cost_of_repairs : ℝ := 8500
def shipping_cost : ℝ := 2500
def import_tax_rate : ℝ := 0.15
def exchange_rate_purchase : ℝ := 75
def exchange_rate_sale : ℝ := 73
def sale_price_inr : ℝ := 4750000

-- Calculate the total cost in USD before taxes
def total_cost_before_taxes : ℝ := cost_of_car + cost_of_repairs + shipping_cost

-- Calculate import taxes and fees
def import_taxes_fees : ℝ := import_tax_rate * total_cost_before_taxes

-- Calculate the total cost in USD including taxes
def total_cost_usd : ℝ := total_cost_before_taxes + import_taxes_fees

-- Convert the total cost to INR
def total_cost_inr : ℝ := total_cost_usd * exchange_rate_purchase

-- Convert the sale price to USD using the new exchange rate
def sale_price_usd : ℝ := sale_price_inr / exchange_rate_sale

-- Calculate Ramu's profit in USD
def profit_usd : ℝ := sale_price_usd - total_cost_usd

-- Calculate Ramu's profit percent
def profit_percent : ℝ := (profit_usd / total_cost_usd) * 100

theorem ramu_profit_percent_correct : profit_percent = 57.15 := 
by 
  sorry

end ramu_profit_percent_correct_l199_199942


namespace intersection_complement_l199_199032

def real_set_M : Set ℝ := {x | 1 < x}
def real_set_N : Set ℝ := {x | x > 4}

theorem intersection_complement (x : ℝ) : x ∈ (real_set_M ∩ (real_set_Nᶜ)) ↔ 1 < x ∧ x ≤ 4 :=
by
  sorry

end intersection_complement_l199_199032


namespace total_parking_spaces_l199_199089

-- Definitions of conditions
def caravan_space : ℕ := 2
def number_of_caravans : ℕ := 3
def spaces_left : ℕ := 24

-- Proof statement
theorem total_parking_spaces :
  (number_of_caravans * caravan_space + spaces_left) = 30 :=
by
  sorry

end total_parking_spaces_l199_199089


namespace both_A_and_B_l199_199717

theorem both_A_and_B (total childrenA childrenB: ℕ) (h_total: total = 48) (h_A: childrenA = 38) (h_B: childrenB = 29) (h_neither: total = childrenA + childrenB - X) :
  X = 19 :=
by
  rw h_total at h_neither
  rw h_A at h_neither
  rw h_B at h_neither
  sorry

end both_A_and_B_l199_199717


namespace problem1_problem2_l199_199182

-- Problem 1
theorem problem1 (b : ℝ) :
  4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1) :=
by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) :
  a - a * abs (-a^2 - 1) < 1 - a^2 * (a - 1) :=
by
  sorry

end problem1_problem2_l199_199182


namespace inequality_solution_set_l199_199786

theorem inequality_solution_set (a b : ℝ) (h1 : a = -2) (h2 : b = 1) :
  {x : ℝ | |2 * x + a| + |x - b| < 6} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_set_l199_199786


namespace remainder_of_b_mod_11_l199_199027

theorem remainder_of_b_mod_11 (n : ℕ) (h_pos : 0 < n) : 
  let b := (5 ^ (2 * n) + 6)⁻¹ in
  b % 11 = 5 := 
by
  sorry

end remainder_of_b_mod_11_l199_199027


namespace magnitude_of_b_l199_199836

variable (a b : ℝ^3)
variable (angle_ab : Real.Angle)
variable (norm_a norm_b norm_a_minus_2b : ℝ)

-- Given conditions
def conditions : Prop :=
  angle_ab = Real.pi / 3 ∧
  ‖a‖ = 2 ∧
  ‖a - 2 • b‖ = 2 * Real.sqrt 7

-- Prove that |b| = 3 given the conditions
theorem magnitude_of_b : conditions a b angle_ab norm_a norm_b norm_a_minus_2b → ‖b‖ = 3 := by
  sorry

end magnitude_of_b_l199_199836


namespace two_tailed_coin_probability_l199_199572

variable {Ω : Type} -- Our probability space

-- Defining the probability measure
variable (P : MeasureTheory.Measure Ω)

-- Events
variable (A B : Set Ω)

-- Probabilities of the events and conditional probabilities
variable (h_PA : P A = 1 / 10)
variable (h_PBA : P B ∩ A / P A = 1)
variable (h_PfairB : P B ∩ (fun x, x ∉ A) / P (fun x, x ∉ A) = 1 / 2)

-- Number of fair coins
def Pfair : ℝ := 9 / 10

-- Calculate Bayes' Theorem
theorem two_tailed_coin_probability :
  (P B ∩ A / P B) = 2 / 11 :=
by
  -- we will leave the proof blank and only focus on the statement itself
  sorry

end two_tailed_coin_probability_l199_199572


namespace determinant_of_matrix_l199_199364

variable (α β : ℝ)

def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![cos α * cos β, tan α * cos β, -sin α],
    ![-sin β, cos β, 0],
    ![tan α * cos β, sin α * sin β, cos α]
  ]

theorem determinant_of_matrix :
  Matrix.det (matrix α β) = cos α ^ 2 * cos β ^ 2 + cos β * tan α + sin α ^ 2 * sin β ^ 2 := by
  sorry

end determinant_of_matrix_l199_199364


namespace arccos_half_eq_pi_div_three_l199_199285

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199285


namespace sequence_a_formula_series_value_l199_199418

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 ^ n

def b (n : ℕ) : ℕ :=
  a n - 1

def S : ℕ → ℕ
| 3 := 14
| (n + 1) := (1 / 2) * (a 2 * S n) + a 1

theorem sequence_a_formula :
  ∀ n : ℕ, a n = 2 ^ n :=
sorry

theorem series_value (n : ℕ) :
  (∑ i in finset.range n, a i / (b i * b (i + 1))) = 1 - 1 / (2 ^ (n + 1) - 1) :=
sorry

end sequence_a_formula_series_value_l199_199418


namespace arccos_half_eq_pi_div_three_l199_199314

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199314


namespace find_angle_C_max_area_ABC_l199_199405

-- Definitions and Conditions
variables (A B C a b c : Real)
variables (cos_bar A_cos_bar : Real)

-- Conditions
axiom angle_conditions : A > 0 ∧ A < Real.pi ∧ B > 0 ∧ B < Real.pi ∧ C > 0 ∧ C < Real.pi

axiom side_conditions : a > 0 ∧ b > 0 ∧ c > 0

axiom cosine_condition : a * Real.cos B + b * Real.cos A - Real.sqrt 2 * c * Real.cos C = 0

-- Theorem to be proved
theorem find_angle_C :
  C = Real.pi / 4 :=
sorry

theorem max_area_ABC (c_eq_two : c = 2) :
  exists ab_max : Real, ab_max = (1 + Real.sqrt 2) ∧
  let S_ABC := (1 / 2) * a * b * Real.sin C in
  S_ABC ≤ ab_max :=
sorry

end find_angle_C_max_area_ABC_l199_199405


namespace solve_for_m_l199_199145

-- Definition of the problem constraints
def rectangular_field_dimensions (m : ℝ) : (ℝ × ℝ) := (3 * m + 5, m - 1)
def field_area (m : ℝ) : ℝ := (3 * m + 5) * (m - 1)

-- The given area of the rectangle
def given_area : ℝ := 104

-- The main proof statement
theorem solve_for_m (m : ℝ) (h1 : field_area m = given_area) : 
  (3 * m ^ 2 + 2 * m - 109 = 0) ∧ (m ≈ 5.70) :=
by {
  sorry
}

end solve_for_m_l199_199145


namespace arccos_half_eq_pi_div_three_l199_199318

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199318


namespace Zaporozhets_eqdist_when_passing_observer_l199_199169

variables {point : Type} [MetricSpace point] 
variables (O Z0 N0 M1 Z1 M2 N2 : point)
variables (t1 t2 : ℝ)
variables (vZ vN vM : ℝ) 
variables (car_pos : ℝ → point → point)

-- Definitions of car positions at given times assuming constant speeds.
def pos_Zaporozhets (t : ℝ) := car_pos t Z0 + vZ * (t - t1)
def pos_Niva (t : ℝ) := car_pos t N0 + vN * (t - t2)
def pos_Moskvich (t : ℝ) := car_pos t M1 + vM * (t - t1)

-- Main condition when the Moskvich is level with the observer
axiom Moskvich_eqdist (h_m : (dist Z0 O) = (dist O N0))

-- Main condition when the Niva is level with the observer
axiom Niva_eqdist (h_n : (dist M1 O) = (dist O Z1))

-- Wanted to prove the case for "Zaporozhets" passing the observer.
theorem Zaporozhets_eqdist_when_passing_observer :
  (dist (pos_Moskvich t1) (pos_Zaporozhets t1) = dist (pos_Zaporozhets t1) (pos_Niva t1)) :=
by sorry

end Zaporozhets_eqdist_when_passing_observer_l199_199169


namespace probability_hare_killed_l199_199612

theorem probability_hare_killed (P_hit_1 P_hit_2 P_hit_3 : ℝ)
  (h1 : P_hit_1 = 3 / 5) (h2 : P_hit_2 = 3 / 10) (h3 : P_hit_3 = 1 / 10) :
  (1 - ((1 - P_hit_1) * (1 - P_hit_2) * (1 - P_hit_3))) = 0.748 :=
by
  sorry

end probability_hare_killed_l199_199612


namespace bob_shucks_240_oysters_in_2_hours_l199_199174

-- Definitions based on conditions provided:
def oysters_per_minute (oysters : ℕ) (minutes : ℕ) : ℕ :=
  oysters / minutes

def minutes_in_hour : ℕ :=
  60

def oysters_in_two_hours (oysters_per_minute : ℕ) (hours : ℕ) : ℕ :=
  oysters_per_minute * (hours * minutes_in_hour)

-- Parameters given in the problem:
def initial_oysters : ℕ := 10
def initial_minutes : ℕ := 5
def hours : ℕ := 2

-- The main theorem we need to prove:
theorem bob_shucks_240_oysters_in_2_hours :
  oysters_in_two_hours (oysters_per_minute initial_oysters initial_minutes) hours = 240 :=
by
  sorry

end bob_shucks_240_oysters_in_2_hours_l199_199174


namespace function_is_increasing_l199_199862

noncomputable def is_increasing (f : ℝ → ℝ) : Prop :=
∀ (x1 x2 : ℝ), x1 < x2 → f(x1) < f(x2)

theorem function_is_increasing (f : ℝ → ℝ)
  (H : ∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 * f(x1) + x2 * f(x2) > x1 * f(x2) + x2 * f(x1)) :
  is_increasing f :=
sorry

end function_is_increasing_l199_199862


namespace sum_of_perimeters_eq_l199_199598

theorem sum_of_perimeters_eq (a : ℝ) : 
  let perimeters : ℕ → ℝ  := λ n, 3 * a / (2^n) in
  ∑' n, perimeters n = 6 * a :=
sorry

end sum_of_perimeters_eq_l199_199598


namespace marie_can_give_away_l199_199519

noncomputable def stamps_given_away : ℕ :=
  let notebooks := 40 * 150
  let binders := 12 * 315
  let folders := 25 * 80
  let total_stamps := notebooks + binders + folders
  let stamps_kept := (total_stamps * 285) / 1000
  let stamps_rounded_kept := stamps_kept.toNat
  total_stamps - stamps_rounded_kept

theorem marie_can_give_away : stamps_given_away = 8423 := by
  sorry

end marie_can_give_away_l199_199519


namespace volume_ratio_john_emma_l199_199492

theorem volume_ratio_john_emma (r_J h_J r_E h_E : ℝ) (diam_J diam_E : ℝ)
  (h_diam_J : diam_J = 8) (h_r_J : r_J = diam_J / 2) (h_h_J : h_J = 15)
  (h_diam_E : diam_E = 10) (h_r_E : r_E = diam_E / 2) (h_h_E : h_E = 12) :
  (π * r_J^2 * h_J) / (π * r_E^2 * h_E) = 4 / 5 := by
  sorry

end volume_ratio_john_emma_l199_199492


namespace sum_of_x_coordinates_l199_199616

open Real

noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem sum_of_x_coordinates (B C F G : ℝ × ℝ)
   (hB : B = (0, 0))
   (hC : C = (331, 0))
   (hF : F = (800, 450))
   (hG : G = (813, 463))
   (hABC_area : triangle_area B.1 B.2 C.1 C.2 A.1 A.2 = 3009)
   (hAFG_area : triangle_area A.1 A.2 F.1 F.2 G.1 G.2 = 9003) :
   ∃ A : ℝ × ℝ, (sum_of_all_possible_x_coordinates A = 1400) :=
begin
  sorry
end

end sum_of_x_coordinates_l199_199616


namespace problem_I_problem_II_l199_199833

noncomputable def A := {x : ℝ | -1 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2m + 3}

theorem problem_I (m : ℝ) : (A ∪ B m = A) ↔ (m ∈ Set.Iio (-2) ∪ Set.Ioo (-2, -1/2)) := by
  sorry

theorem problem_II (m : ℝ) : (A ∩ B m ≠ ∅) ↔ (m ∈ Set.Ioo (-2, 1)) := by
  sorry

end problem_I_problem_II_l199_199833


namespace simplest_form_of_expression_l199_199112

theorem simplest_form_of_expression :
  let a := (√3 : ℝ)
  let b := (√(3+5) : ℝ)
  let c := (√(3+5+7) : ℝ)
  let d := (√(3+5+7+9) : ℝ)
  a + b + c + d = √3 + 2*√2 + √15 + 2*√6 :=
by
  sorry

end simplest_form_of_expression_l199_199112


namespace problemI_problemII_l199_199132

-- Proof Problem I Statement in Lean 4
theorem problemI (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : α + β = π/4) :
  (1 + tan α) * (1 + tan β) = 2 :=
sorry

-- Proof Problem II Statement in Lean 4
theorem problemII (α β : ℝ) 
  (h1 : π/2 < β ∧ β < α ∧ α < 3 * π / 4) 
  (h2 : cos (α - β) = 12 / 13) 
  (h3 : sin (α + β) = -3 / 5) :
  sin (2 * α) = -56 / 65 :=
sorry

end problemI_problemII_l199_199132


namespace ratio_mn_eq_x_plus_one_over_two_x_plus_one_l199_199566

theorem ratio_mn_eq_x_plus_one_over_two_x_plus_one (x : ℝ) (m n : ℝ) 
  (hx : x > 0) 
  (hmn : m * n ≠ 0) 
  (hineq : m * x > n * x + n) : 
  m / (m + n) = (x + 1) / (2 * x + 1) := 
by 
  sorry

end ratio_mn_eq_x_plus_one_over_two_x_plus_one_l199_199566


namespace _l199_199028

noncomputable theorem absolute_value_xyz (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) 
(h_cond : x + 2/y = y + 2/z ∧ y + 2/z = z + 2/x) : |x * y * z| = 2 :=
sorry

end _l199_199028


namespace range_of_a_l199_199013

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1/x) + a

theorem range_of_a (a : ℝ) (h : f a 0 = a^2) : (f a 0 = f a 0 -> 0 ≤ a ∧ a ≤ 2) := by
  sorry

end range_of_a_l199_199013


namespace find_real_roots_of_polynomial_l199_199381
-- Import the necessary libraries

-- Define the polynomial function f
def f (x : ℝ) : ℝ := x^4 + x^3 - 7*x^2 - x + 6

-- Statement of the theorem
theorem find_real_roots_of_polynomial :
    ∃ r1 r2 r3 : ℝ, (r1 = -2 ∧ r2 = -1 ∧ r3 = 3) ∧
                    (∃ m2 : ℕ, m2 = 2 ∧ roots_multiplicity f r2 m2) ∧
                    (roots f = [r1, r2, r2, r3]) :=
by
  sorry

end find_real_roots_of_polynomial_l199_199381


namespace smallest_n_l199_199639

theorem smallest_n (n : ℕ) (h1 : 1826 % 26 = 6) (h2 : 5 * n % 26 = 6) : n = 20 :=
sorry

end smallest_n_l199_199639


namespace no_2021_residents_possible_l199_199748

-- Definition: Each islander is either a knight or a liar
def is_knight_or_liar (i : ℕ) : Prop := true -- Placeholder definition for either being a knight or a liar

-- Definition: Knights always tell the truth
def knight_tells_truth (i : ℕ) : Prop := true -- Placeholder definition for knights telling the truth

-- Definition: Liars always lie
def liar_always_lies (i : ℕ) : Prop := true -- Placeholder definition for liars always lying

-- Definition: Even number of knights claimed by some islanders
def even_number_of_knights : Prop := true -- Placeholder definition for the claim of even number of knights

-- Definition: Odd number of liars claimed by remaining islanders
def odd_number_of_liars : Prop := true -- Placeholder definition for the claim of odd number of liars

-- Question and proof problem
theorem no_2021_residents_possible (K L : ℕ) (h1 : K + L = 2021) (h2 : ∀ i, is_knight_or_liar i) 
(h3 : ∀ k, knight_tells_truth k → even_number_of_knights) 
(h4 : ∀ l, liar_always_lies l → odd_number_of_liars) : 
  false := sorry

end no_2021_residents_possible_l199_199748


namespace find_mn_l199_199474

variable {AB CD BD BC : ℚ}
variable {BAD ADC ABD BCD : Prop}

theorem find_mn (h₁ : ∠ BAD = ∠ ADC) (h₂ : ∠ ABD = ∠ BCD)
  (h₃ : AB = 10) (h₄ : BD = 12) (h₅ : BC = 7) : ∃ (m n : ℕ), m + n = 43 :=
by
  let m := 40
  let n := 3
  use m
  use n
  have h₆ : m + n = 43 := by norm_num
  exact h₆

end find_mn_l199_199474


namespace alicia_tax_deduction_is_50_cents_l199_199712

def alicia_hourly_wage_dollars : ℝ := 25
def deduction_rate : ℝ := 0.02

def alicia_hourly_wage_cents : ℝ := alicia_hourly_wage_dollars * 100
def tax_deduction_cents : ℝ := alicia_hourly_wage_cents * deduction_rate

theorem alicia_tax_deduction_is_50_cents : tax_deduction_cents = 50 := by
  sorry

end alicia_tax_deduction_is_50_cents_l199_199712


namespace PC_length_l199_199614

theorem PC_length {P A B C : Type}
  [metric_space P] [metric_space A] [metric_space B] [metric_space C]
  (dist_PA : dist P A = 7) (dist_PB : dist P B = 9) (angle_APB : ∠ A P B = 120)
  (angle_BPC : ∠ B P C = 120) (angle_CPA : ∠ C P A = 120) :
  dist P C = 2 :=
by
  sorry

end PC_length_l199_199614


namespace positive_root_condition_negative_root_condition_zero_root_condition_l199_199372

-- Positive root condition
theorem positive_root_condition {a b : ℝ} (h : a * b < 0) : ∃ x : ℝ, a * x + b = 0 ∧ x > 0 :=
by
  sorry

-- Negative root condition
theorem negative_root_condition {a b : ℝ} (h : a * b > 0) : ∃ x : ℝ, a * x + b = 0 ∧ x < 0 :=
by
  sorry

-- Root equal to zero condition
theorem zero_root_condition {a b : ℝ} (h₁ : b = 0) (h₂ : a ≠ 0) : ∃ x : ℝ, a * x + b = 0 ∧ x = 0 :=
by
  sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l199_199372


namespace calculate_expression_l199_199508

theorem calculate_expression (p q : ℝ) (hp : p + q = 7) (hq : p * q = 12) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 3691 := 
by sorry

end calculate_expression_l199_199508


namespace orthocenter_on_euler_line_of_incenter_and_circumcenter_l199_199899

theorem orthocenter_on_euler_line_of_incenter_and_circumcenter
  (ABC : Triangle)
  (h_non_eq : ¬Equilateral ABC)
  (incenter : Point)
  (circumcenter : Point)
  (A1 B1 C1 : Points)
  (ha1 : TangentCircle A1 (side ABC BC))
  (hb1 : TangentCircle B1 (side ABC CA))
  (hc1 : TangentCircle C1 (side ABC AB))
  (M : Point)
  (hM : Orthocenter M A1 B1 C1) :
    LiesOnLine M (line incenter circumcenter) :=
by
  sorry

end orthocenter_on_euler_line_of_incenter_and_circumcenter_l199_199899


namespace min_dist_circle_to_line_l199_199379

theorem min_dist_circle_to_line : 
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let line (x y : ℝ) := x + sqrt(3) * y = 6
  ∀ x y : ℝ, circle x y → ∃ d : ℝ, d = 1 ∧ (∃ p q : ℝ, line p q ∧ sqrt((x-p)^2 + (y-q)^2) = d) :=
by
  sorry

end min_dist_circle_to_line_l199_199379


namespace arccos_half_eq_pi_div_three_l199_199283

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199283


namespace triangle_ratio_proof_l199_199141

noncomputable def triangle_side_ratios (a x : ℝ) (h1 : 0 < a) (h2 : 0 < x) : Prop :=
  let BM := 3 * a in
  let side_CA := a * Real.sqrt 2 + x in
  let side_AB := 3 * a * Real.sqrt 2 + x in
  let side_BC := 2 * a * Real.sqrt 2 + 2 * x in
  let ratio_BC_CA := 10 / 5 in
  let ratio_AB_CA := 13 / 5 in
  |BC| / |CA| = ratio_BC_CA ∧ |AB| / |CA| = ratio_AB_CA

theorem triangle_ratio_proof (a x : ℝ) (h1 : 0 < a) (h2 : 0 < x) (h : triangle_side_ratios a x h1 h2) :
  |let side_BC := 2 * a * Real.sqrt 2 + 2 * x in
  let side_CA := a * Real.sqrt 2 + x in
  let side_AB := 3 * a * Real.sqrt 2 + x in
  (side_BC / side_CA = 10 / 5 ∧ side_AB / side_CA = 13 / 5) 
| := 
sory

end triangle_ratio_proof_l199_199141


namespace weight_jordan_after_exercise_l199_199003

def initial_weight : ℕ := 250
def first_4_weeks_loss : ℕ := 3 * 4
def next_8_weeks_loss : ℕ := 2 * 8
def total_weight_loss : ℕ := first_4_weeks_loss + next_8_weeks_loss
def final_weight : ℕ := initial_weight - total_weight_loss

theorem weight_jordan_after_exercise : final_weight = 222 :=
by 
  sorry

end weight_jordan_after_exercise_l199_199003


namespace sum_of_integers_is_18_l199_199593

theorem sum_of_integers_is_18 (a b : ℕ) (h1 : b = 2 * a) (h2 : a * b + a + b = 156) (h3 : Nat.gcd a b = 1) (h4 : a < 25) : a + b = 18 :=
by
  sorry

end sum_of_integers_is_18_l199_199593


namespace arccos_half_is_pi_div_three_l199_199261

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199261


namespace roots_equation_value_l199_199778

theorem roots_equation_value (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α + β = 1) :
    α^4 + 3 * β = 5 := by
sorry

end roots_equation_value_l199_199778


namespace alpha_beta_expression_l199_199775

noncomputable theory

variables (α β : ℝ)

-- Conditions
axiom root_equation : ∀ x, (x = α ∨ x = β) → x^2 - x - 1 = 0
axiom alpha_square : α^2 = α + 1
axiom alpha_beta_sum : α + β = 1

-- The statement
theorem alpha_beta_expression : α^4 + 3 * β = 5 :=
sorry

end alpha_beta_expression_l199_199775


namespace problem_solution_l199_199991

theorem problem_solution (x y z : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (h1 : x^2 + y^2 = 9) 
  (h2 : y^2 + y*z + z^2 = 16) 
  (h3 : x^2 + sqrt 3 * x * z + z^2 = 25) : 
  2 * x * y + x * z + sqrt 3 * y * z = 24 := 
  sorry

end problem_solution_l199_199991


namespace total_volume_collection_l199_199113

-- Define the conditions
def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def cost_per_box : ℚ := 0.5
def minimum_total_cost : ℚ := 255

-- Define the volume of one box
def volume_of_one_box : ℕ := box_length * box_width * box_height

-- Define the number of boxes needed
def number_of_boxes : ℚ := minimum_total_cost / cost_per_box

-- Define the total volume of the collection
def total_volume : ℚ := volume_of_one_box * number_of_boxes

-- The goal is to prove that the total volume of the collection is as calculated
theorem total_volume_collection :
  total_volume = 3060000 := by
  sorry

end total_volume_collection_l199_199113


namespace arccos_half_eq_pi_div_three_l199_199309

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199309


namespace exactly_two_problems_use_loop_l199_199658

-- Definitions based on conditions
def problem1_uses_loop_statement : Prop :=
∃ s : ℕ → ℕ, s 0 = 1 ∧ 
  (∀ n, s (n + 1) = s n + 3^n) ∧ 
  (s 9 = ∑ i in finset.range 10, 3^i)

def problem2_uses_loop_statement : Prop :=
false

def problem3_uses_loop_statement : Prop :=
false

def problem4_uses_loop_statement : Prop :=
∃ n : ℕ, n^2 < 100 ∧ ∀ m : ℕ, (m^2 < 100 → m ≤ n)

-- Main statement
theorem exactly_two_problems_use_loop : 
  (count ({problem1_uses_loop_statement, problem2_uses_loop_statement, problem3_uses_loop_statement, problem4_uses_loop_statement} : finset Prop).1) = 2 :=
sorry

end exactly_two_problems_use_loop_l199_199658


namespace arccos_half_eq_pi_div_three_l199_199252

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199252


namespace mia_age_at_end_of_period_l199_199036

theorem mia_age_at_end_of_period :
  ∀ (work_days : ℕ) (hours_per_day : ℕ) (pay_rate_per_hour : ℕ → ℝ) (total_earnings : ℝ),
  work_days = 80 →
  hours_per_day = 3 →
  (∀ (age : ℕ), pay_rate_per_hour age = 0.40 * age) →
  total_earnings = 960 →
  (∃ (age : ℕ), age = 11) :=
by
  intros work_days hours_per_day pay_rate_per_hour total_earnings h_days h_hours h_rate h_earnings
  refine ⟨11, _⟩
  sorry

end mia_age_at_end_of_period_l199_199036


namespace arccos_half_eq_pi_div_three_l199_199316

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199316


namespace arithmetic_sequence_solution_l199_199791

theorem arithmetic_sequence_solution 
  (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 4 = 7) (h2 : a 10 = 19) 
  (a_general : ∀ n, a n = 2 * n - 1) 
  (S_formula : ∀ n, S n = n^2) 
  (b_formula : ∀ n, b n = 1 / ((a n) * (a (n + 1)))) :
  (∀ n, T n = ∑ i in finset.range n, b i) → 
  (∀ n, T n = n / (2 * n + 1)) :=
by {
  sorry
}

end arithmetic_sequence_solution_l199_199791


namespace andrea_rhinestones_l199_199173

theorem andrea_rhinestones (T : ℝ) 
  (h1 : T / 3) 
  (h2 : T / 5) 
  (h3 : T - (T / 3 + T / 5) = 21) : T = 45 :=
by 
  sorry

end andrea_rhinestones_l199_199173


namespace sum_and_count_valid_primes_l199_199110

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function checking the primality of a number.
def interchange_digits (n : ℕ) : ℕ := sorry -- Assume we have a function to interchange digits.

def valid_primes : List ℕ :=
  List.filter (λ p : ℕ, is_prime p ∧ 20 < p ∧ p < 99 ∧ is_prime (interchange_digits p))
  [21, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def sum_valid_primes : ℕ := (valid_primes).sum
def count_valid_primes : ℕ := (valid_primes).length

theorem sum_and_count_valid_primes :
  sum_valid_primes = 388 ∧ count_valid_primes = 6 :=
by
  -- Statements to be proven.
  sorry

end sum_and_count_valid_primes_l199_199110


namespace sphere_surface_area_l199_199084

theorem sphere_surface_area (V : ℝ) (hV : V = 72 * Real.pi) :
  ∃ r : ℝ, 4 * Real.pi * r^2 = 36 * Real.pi * (Real.cbrt 2)^2 :=
by
  use 3 * Real.cbrt 2
  sorry

end sphere_surface_area_l199_199084


namespace equation_of_circle_and_line_l199_199432

-- Definition of the circle
def circle_eq (x y : ℝ) := x^2 + y^2 + 4 * x = 0

-- Given conditions: the parabola C, and the circle passing through certain points
def parabola_C (x y : ℝ) := y^2 = 4 * x
def passes_through_points (x y : ℝ) := (x=0 ∧ y=0) ∨ (x=-2 ∧ y= 2) ∨ (x=-1 ∧ y=Real.sqrt 3)

-- Given intersection and perpendicularity conditions
def lines_through_origin (l m : ℝ → ℝ) := ∀ x, l x = k * x ∧ m x = -x / k

-- Point definition based on lines and the given curve
def point_M (k : ℝ) : ℝ := 4 / (k^2)
def point_N (k : ℝ) : ℝ := 4 / (1 + k^2)
def point_P (k : ℝ) : ℝ := 4 * (k^2)
def point_Q (k : ℝ) : ℝ := 4 * (k^2) / (1 + k^2)

-- Ratio of triangle areas for minimization condition
def ratio_triangle_areas (k : ℝ) := (1 + k^2)^2 / k^2

-- Equation of line l
def line_eq_l (k : ℝ) := y = if k = 1 ∨ k = -1 then x else y 

theorem equation_of_circle_and_line :
  (∀ x y, passes_through_points x y → circle_eq x y) ∧
  (∀ k, parabola_C (point_M k) (k * point_M k) → parabola_C (point_P k) (- point_M k / k) →
    parabola_C (point_N k) (k * point_N k) → parabola_C (point_Q k) (- point_N k / k) →
    lines_through_origin (λ x, k * x) (λ x, -x / k) →
    ratio_triangle_areas k ≥ 4 →
    line_eq_l k) :=
begin
  sorry
end

end equation_of_circle_and_line_l199_199432


namespace line_from_complex_condition_l199_199850

theorem line_from_complex_condition (z : ℂ) (h : ∃ x y : ℝ, z = x + y * I ∧ (3 * y + 4 * x = 0)) : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), z = x + y * I → 3 * y + 4 * x = 0 → z = a + b * I ∧ 4 * x + 3 * y = 0) := 
sorry

end line_from_complex_condition_l199_199850


namespace min_sum_of_dimensions_l199_199380

theorem min_sum_of_dimensions (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 3003) : 
  a + b + c ≥ 57 := sorry

end min_sum_of_dimensions_l199_199380


namespace length_of_GH_l199_199108

theorem length_of_GH 
(PQ RS GH : ℝ) 
(h1 : PQ = 180) 
(h2 : RS = 120) 
(h3 : PQ ∥ RS) 
(h4 : RS ∥ GH) :
GH = 360 / 7 := 
by
  sorry

end length_of_GH_l199_199108


namespace replace_asterisk_with_2x_l199_199538

theorem replace_asterisk_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by sorry

end replace_asterisk_with_2x_l199_199538


namespace number_of_cards_per_page_l199_199941

variable (packs : ℕ) (cards_per_pack : ℕ) (total_pages : ℕ)

def number_of_cards (packs cards_per_pack : ℕ) : ℕ :=
  packs * cards_per_pack

def cards_per_page (total_cards total_pages : ℕ) : ℕ :=
  total_cards / total_pages

theorem number_of_cards_per_page
  (packs := 60) (cards_per_pack := 7) (total_pages := 42)
  (total_cards := number_of_cards packs cards_per_pack)
    : cards_per_page total_cards total_pages = 10 :=
by {
  sorry
}

end number_of_cards_per_page_l199_199941


namespace arccos_half_eq_pi_div_three_l199_199311

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199311


namespace findWorkRateB_l199_199139

-- Define the work rates of A and C given in the problem
def workRateA : ℚ := 1 / 8
def workRateC : ℚ := 1 / 16

-- Combined work rate when A, B, and C work together to complete the work in 4 days
def combinedWorkRate : ℚ := 1 / 4

-- Define the work rate of B that we need to prove
def workRateB : ℚ := 1 / 16

-- Theorem to prove that workRateB is equal to B's work rate given the conditions
theorem findWorkRateB : workRateA + workRateB + workRateC = combinedWorkRate :=
  by
  sorry

end findWorkRateB_l199_199139


namespace hyperbola_equation_l199_199980

theorem hyperbola_equation :
  ∃ (a b : ℝ),
    (c = 2) → 
    (c^2 = a^2 + b^2) →
    (∃ (x y : ℝ),
      (x = sqrt 2) ∧ 
      (y = sqrt 3) ∧ 
      (∀ (x y : ℝ),
         (x^2 / a^2 - y^2 / b^2 = 1) →
        (2 / a^2 - 3 / b^2 = 1))) →
    (a^2 = 1) ∧ (b^2 = 3)
|>
sorry

end hyperbola_equation_l199_199980


namespace arccos_half_eq_pi_div_three_l199_199317

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199317


namespace polynomial_q_evaluation_l199_199020

noncomputable def q (x : ℝ) : ℝ :=
    (x - 1) * (x - 2) * (x - 3) * (x - 5) + 13 * x

theorem polynomial_q_evaluation :
    q 1 = 29 ∧ q 2 = 46 ∧ q 3 = 63 → q 0 + q 5 = 95 :=
begin
    intros h,
    sorry
end

end polynomial_q_evaluation_l199_199020


namespace islander_parity_l199_199750

-- Define the concept of knights and liars
def is_knight (x : ℕ) : Prop := x % 2 = 0 -- Knight count is even
def is_liar (x : ℕ) : Prop := ¬(x % 2 = 1) -- Liar count being odd is false, so even

-- Define the total inhabitants on the island and conditions
theorem islander_parity (K L : ℕ) (h₁ : is_knight K) (h₂ : is_liar L) (h₃ : K + L = 2021) : false := sorry

end islander_parity_l199_199750


namespace range_of_m_l199_199809

noncomputable def quadratic_roots (m : ℝ) [hm : (x^2 - 2*x + m = 0)] : ℝ × ℝ :=
if discriminant_ge_zero : (4 - 4*m) ≥ 0 then 
  let sqrt_disc := Real.sqrt(4 - 4*m) in
  (1 + Real.sqrt(1 - m), 1 - Real.sqrt(1 - m))
else
  (0, 0)

theorem range_of_m (m : ℝ) : 
  (∃ a b : ℝ, quadratic_roots m = (a, b) ∧ a + b = 2 ∧ 1 - m ≥ 0 ∧ a + b > 1 ∧ abs (a - b) < 1) 
  → (3 / 4 < m ∧ m ≤ 1) :=
by
  intro h
  sorry

end range_of_m_l199_199809


namespace pump1_half_drain_time_l199_199042

-- Definitions and Conditions
def time_to_drain_half_pump1 (t : ℝ) : Prop :=
  ∃ rate1 rate2 : ℝ, 
    rate1 = 1 / (2 * t) ∧
    rate2 = 1 / 1.25 ∧
    rate1 + rate2 = 2

-- Equivalent Proof Problem
theorem pump1_half_drain_time (t : ℝ) : time_to_drain_half_pump1 t → t = 5 / 12 := sorry

end pump1_half_drain_time_l199_199042


namespace village_population_l199_199626

variable (Px : ℕ) (t : ℕ) (dX dY : ℕ)
variable (Py : ℕ := 42000) (rateX : ℕ := 1200) (rateY : ℕ := 800) (timeYears : ℕ := 15)

theorem village_population : (Px - rateX * timeYears = Py + rateY * timeYears) → Px = 72000 :=
by
  sorry

end village_population_l199_199626


namespace fish_remain_approximately_correct_l199_199087

noncomputable def remaining_fish : ℝ :=
  let west_initial := 1800
  let east_initial := 3200
  let north_initial := 500
  let south_initial := 2300
  let a := 3
  let b := 4
  let c := 2
  let d := 5
  let e := 1
  let f := 3
  let west_caught := (a / b) * west_initial
  let east_caught := (c / d) * east_initial
  let south_caught := (e / f) * south_initial
  let west_left := west_initial - west_caught
  let east_left := east_initial - east_caught
  let south_left := south_initial - south_caught
  let north_left := north_initial
  west_left + east_left + south_left + north_left

theorem fish_remain_approximately_correct :
  abs (remaining_fish - 4403) < 1 := 
  sorry

end fish_remain_approximately_correct_l199_199087


namespace line_equation_l199_199802

theorem line_equation (a b : ℝ) (h_intercept_eq : a = b) (h_pass_through : 3 * a + 2 * b = 2 * a + 5) : (3 + 2 = 5) ↔ (a = 5 ∧ b = 5) :=
sorry

end line_equation_l199_199802


namespace probability_x_lt_y_in_rectangle_l199_199151

noncomputable def probability_point_in_triangle : ℚ :=
  let rectangle_area : ℚ := 4 * 3
  let triangle_area : ℚ := (1/2) * 3 * 3
  let probability : ℚ := triangle_area / rectangle_area
  probability

theorem probability_x_lt_y_in_rectangle :
  probability_point_in_triangle = 3 / 8 :=
by
  sorry

end probability_x_lt_y_in_rectangle_l199_199151


namespace greatest_multiple_of_35_less_than_1000_l199_199633

theorem greatest_multiple_of_35_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 35 = 0 ∧ (∀ k : ℕ, k < 1000 → k % 35 = 0 → k ≤ n) ∧ n = 980 :=
begin
  sorry
end

end greatest_multiple_of_35_less_than_1000_l199_199633


namespace rational_numbers_sum_reciprocal_integer_l199_199986

theorem rational_numbers_sum_reciprocal_integer (p1 q1 p2 q2 : ℤ) (k m : ℤ)
  (h1 : Int.gcd p1 q1 = 1)
  (h2 : Int.gcd p2 q2 = 1)
  (h3 : p1 * q2 + p2 * q1 = k * q1 * q2)
  (h4 : q1 * p2 + q2 * p1 = m * p1 * p2) :
  (p1, q1, p2, q2) = (x, y, -x, y) ∨
  (p1, q1, p2, q2) = (2, 1, 2, 1) ∨
  (p1, q1, p2, q2) = (-2, 1, -2, 1) ∨
  (p1, q1, p2, q2) = (1, 1, 1, 1) ∨
  (p1, q1, p2, q2) = (-1, 1, -1, 1) ∨
  (p1, q1, p2, q2) = (1, 2, 1, 2) ∨
  (p1, q1, p2, q2) = (-1, 2, -1, 2) :=
sorry

end rational_numbers_sum_reciprocal_integer_l199_199986


namespace find_x_l199_199124

theorem find_x :
  ∃ (x : ℕ), 3 * x = (62 - x) + 26 ∧ x = 22 :=
by
  use 22
  split
  {
    calc
      3 * 22 = 66 : by norm_num
      ... = 36 + 30 : by norm_num
      ... = (62 - 22) + 26 : by norm_num
  }
  rfl

end find_x_l199_199124


namespace order_of_magnitude_l199_199741

noncomputable def cube_root_seven : ℝ := real.cbrt 7
noncomputable def fractional_power : ℝ := (0.3:ℝ) ^ 7
noncomputable def log_base_three : ℝ := real.logb 0.3 0.001

theorem order_of_magnitude :
  cube_root_seven > fractional_power ∧
  fractional_power > log_base_three := by
  sorry

end order_of_magnitude_l199_199741


namespace geometric_sequence_correspondence_l199_199471

-- Definitions
def is_geometric_sequence (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

-- The problem statement rewritten in Lean
theorem geometric_sequence_correspondence (b : ℕ → ℝ) (h_geo : is_geometric_sequence b) (h_len : ∀ n, n ≤ 2010 → b n ≠ 0) :
  (b 0 + b 2 + b 4 + ... + b 2010) - (b 1 + b 3 + ... + b 2009) = b 1006 := 
sorry

end geometric_sequence_correspondence_l199_199471


namespace television_screen_horizontal_length_l199_199060

theorem television_screen_horizontal_length (d : ℝ) (h_ratio v_ratio : ℝ) (a : ℝ) 
  (aspect_ratio : h_ratio = 16 ∧ v_ratio = 9 ∧ d = 40 ∧ a = real.sqrt (16^2 + 9^2)) :
  ∃ l : ℝ, l = 16 * 40 / real.sqrt (16^2 + 9^2) :=
by
  sorry

end television_screen_horizontal_length_l199_199060


namespace probability_x_lt_y_in_rectangle_l199_199150

noncomputable def probability_point_in_triangle : ℚ :=
  let rectangle_area : ℚ := 4 * 3
  let triangle_area : ℚ := (1/2) * 3 * 3
  let probability : ℚ := triangle_area / rectangle_area
  probability

theorem probability_x_lt_y_in_rectangle :
  probability_point_in_triangle = 3 / 8 :=
by
  sorry

end probability_x_lt_y_in_rectangle_l199_199150


namespace Q_correct_l199_199009

noncomputable def Q(x : ℝ) : ℝ := -3 + 4*x^2 - x^3

theorem Q_correct :
  Q(0) + Q(2)*x^2 + Q(3)*x^3 = -3 + 4*x^2 - x^3 ∧
  Q(-1) = 2 ∧
  Q(0) = -3 ∧
  Q(2) = 4 ∧
  Q(3) = -1 ∧
  Q(1) = -3 + 4*1^2 - 1^3 ∧
  Q(2) = -3 + 4*2^2 - 2^3 ∧
  Q(3) = -3 + 4*3^2 - 3^3
  :=
by
  sorry

end Q_correct_l199_199009


namespace football_area_of_overlapping_arcs_l199_199532

theorem football_area_of_overlapping_arcs 
  (A B C D : Point)
  (M N : Point)
  {r : ℝ} (h_square : square A B C D)
  (h_side_length : dist A B = 4)
  (h_M_mid_AD : midpoint M A D)
  (h_N_mid_BC : midpoint N B C)
  (h_circle_1 : circle M r)
  (h_circle_2 : circle N r)
  (h_r : r = 4)
  (h_B_on_circle_1 : on_circle B M r)
  (h_D_on_circle_2 : on_circle D N r)
  (E F : Point)
  (h_intersect_1 : intersects_circle E M r N r)
  (h_intersect_2 : intersects_circle F M r N r) :
  area_of_overlap_arcs E M D F N B = 8 * π := sorry

end football_area_of_overlapping_arcs_l199_199532


namespace functions_equal_l199_199718

noncomputable def f (x : ℝ) : ℝ := x^0
noncomputable def g (x : ℝ) : ℝ := x / x

theorem functions_equal (x : ℝ) (hx : x ≠ 0) : f x = g x :=
by
  unfold f g
  sorry

end functions_equal_l199_199718


namespace lattice_point_count_l199_199876

theorem lattice_point_count :
  (∃ (S : Finset (ℤ × ℤ)), S.card = 16 ∧ ∀ (p : ℤ × ℤ), p ∈ S → (|p.1| - 1) ^ 2 + (|p.2| - 1) ^ 2 < 2) :=
sorry

end lattice_point_count_l199_199876


namespace parallel_vector_lambda_l199_199837

theorem parallel_vector_lambda (λ : ℝ) :
  let a := (-1, 2 : ℝ × ℝ)
  let b := (λ, 1 : ℝ × ℝ)
  (∃ k : ℝ, a + b = k • a) → λ = -1 / 2 :=
by
  intros a b h
  let a := (-1, 2 : ℝ × ℝ)
  let b := (λ, 1 : ℝ × ℝ)
  have h1 : a + b = (-1 + λ, 2 + 1),
  {
    simp,
  },
  rw h1 at h,
  exact sorry

end parallel_vector_lambda_l199_199837


namespace train_speed_and_length_l199_199137

theorem train_speed_and_length 
  (x y : ℝ)
  (h1 : 60 * x = 1000 + y)
  (h2 : 40 * x = 1000 - y) :
  x = 20 ∧ y = 200 :=
by
  sorry

end train_speed_and_length_l199_199137


namespace proof_problem_l199_199438

def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-2, 2)
def vec_c (k : ℝ) : ℝ × ℝ := (4, k)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scala := dot_product u v / (dot_product v v)
  (scala * v.1, scala * v.2)

def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem proof_problem (k : ℝ) :
  dot_product vec_a vec_b = 2 ∧
  (is_perpendicular (vec_a.1 - vec_b.1, vec_a.2 + vec_b.2) (vec_c k) → k = 1) ∧
  projection vec_a vec_b = (-1 / 2, 1 / 2) ∧
  (is_parallel (vec_a.1 - vec_b.1, vec_a.2 + vec_b.2) (vec_c k) → k = 1) := 
by
  sorry

end proof_problem_l199_199438


namespace dice_five_prob_l199_199109

-- Define a standard six-sided die probability
def prob_five : ℚ := 1 / 6

-- Define the probability of all four dice showing five
def prob_all_five : ℚ := prob_five * prob_five * prob_five * prob_five

-- State the theorem
theorem dice_five_prob : prob_all_five = 1 / 1296 := by
  sorry

end dice_five_prob_l199_199109


namespace exponent_equality_l199_199655

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality_l199_199655


namespace largest_multiple_5_6_lt_1000_is_990_l199_199107

theorem largest_multiple_5_6_lt_1000_is_990 : ∃ n, (n < 1000) ∧ (n % 5 = 0) ∧ (n % 6 = 0) ∧ n = 990 :=
by 
  -- Needs to follow the procedures to prove it step-by-step
  sorry

end largest_multiple_5_6_lt_1000_is_990_l199_199107


namespace find_a_l199_199429

def f (a x : ℝ) : ℝ := 1 / x + a * x^3
def f' (a x : ℝ) : ℝ := -1 / (x^2) + 3 * a * (x^2)

theorem find_a (a : ℝ) (h : f'(a, 1) = 5) : a = 2 :=
by
  -- Proof goes here, but we use sorry to skip it as instructed.
  sorry

end find_a_l199_199429


namespace Edmund_earns_64_dollars_l199_199756

-- Conditions
def chores_per_week : Nat := 12
def pay_per_extra_chore : Nat := 2
def chores_per_day : Nat := 4
def weeks : Nat := 2
def days_per_week : Nat := 7

-- Goal
theorem Edmund_earns_64_dollars :
  let total_chores_without_extra := chores_per_week * weeks
  let total_chores_with_extra := chores_per_day * (days_per_week * weeks)
  let extra_chores := total_chores_with_extra - total_chores_without_extra
  let earnings := pay_per_extra_chore * extra_chores
  earnings = 64 :=
by
  sorry

end Edmund_earns_64_dollars_l199_199756


namespace arccos_of_half_eq_pi_over_three_l199_199305

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199305


namespace algebraic_expression_value_l199_199408

variable {a b c : ℝ}

theorem algebraic_expression_value
  (h1 : (a + b) * (b + c) * (c + a) = 0)
  (h2 : a * b * c < 0) :
  (a / |a|) + (b / |b|) + (c / |c|) = 1 := by
  sorry

end algebraic_expression_value_l199_199408


namespace max_value_expression_l199_199021

noncomputable def target_expr (x y z : ℝ) : ℝ :=
  (x^2 - x * y + y^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)

theorem max_value_expression (x y z : ℝ) (h : x + y + z = 3) (hxy : x = y) (hxz : 0 ≤ x) (hyz : 0 ≤ y) (hzz : 0 ≤ z) :
  target_expr x y z ≤ 9 / 4 := by
  sorry

end max_value_expression_l199_199021


namespace f_double_l199_199071

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 3 ^ x - 2 else Real.log x / Real.log 3

theorem f_double (h1 : ∃ x, x = 5 / 3) (f_five_thirds : f (5 / 3) = Real.log (2 / 3) / Real.log 3) :
  f (f (5 / 3)) = -4 / 3 := sorry

end f_double_l199_199071


namespace optimal_garage_location_l199_199360

-- Define basic components of the problem.
variables (V : Type*) [Fintype V] [DecidableEq V] -- V is the set of vertices (locations)
variables (E : Type*) [Fintype E] [DecidableEq E] -- E is the set of edges (streets)
variables (G : Graph V E) -- G is the graph representing the town's street network
variable (house : V) -- 'house' is Dr. Sharadek's home location

-- Conditions
def is_eulerian_circuit (p : Path G) : Prop :=
  p.starts_ends_at_same_point ∧ p.visits_every_edge

-- Given a shortest route length property (this could be further detailed)
parameter (route_efficiency : ℝ) -- route increases by 16.7%
axiom efficiency_increase : route_efficiency = 1.167

-- The proof problem
theorem optimal_garage_location : is_eulerian_circuit (some_path G) → 
                                    reachable_from_everywhere G house →
                                    (∀ start_location, 
                                     is_no_worse_than (path_length start_location (some_path G)) 
                                                     route_efficiency) → 
                                   correct_decision :=
sorry

end optimal_garage_location_l199_199360


namespace arccos_half_eq_pi_div_three_l199_199253

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199253


namespace find_A_l199_199394

theorem find_A (a b : ℝ) (A : ℝ) : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A → A = 60 * a * b :=
begin
  sorry,
end

end find_A_l199_199394


namespace tan_theta_proof_l199_199801

noncomputable def tan_theta_value (θ : ℝ) : ℝ :=
  if θ ∈ Ioo 0 (π / 2) ∧ ((cos θ, 2) : ℝ × ℝ) ⊥ ((-1, sin θ) : ℝ × ℝ)
  then tan θ
  else 0

theorem tan_theta_proof (θ : ℝ) (h1 : θ ∈ Ioo 0 (π / 2)) (h2 : ((cos θ, 2) : ℝ × ℝ) ⊥ ((-1, sin θ) : ℝ × ℝ)) :
  tan θ = 1 / 2 :=
sorry

end tan_theta_proof_l199_199801


namespace locus_of_midpoint_of_chord_l199_199400

theorem locus_of_midpoint_of_chord
  (x y : ℝ)
  (hx : (x - 1)^2 + y^2 ≠ 0)
  : (x - 1) * (x - 1) + y * y = 1 :=
by
  sorry

end locus_of_midpoint_of_chord_l199_199400


namespace correct_condition_l199_199399

-- Definitions of lines and planes
variable {Line Plane : Type}
variable (perpendicular parallel : Line → Plane → Prop)
variable (perpendicular_line : Line → Line → Boolean)
variable (parallel_line : Line → Line → Boolean)

-- Conditions specifying lines and planes relationship
def condition_3 : Prop :=
  ∀ (x y : Line) (z : Plane), perpendicular x z → perpendicular y z → parallel_line x y

-- The theorem statement
theorem correct_condition :
  condition_3 :=
by
  sorry

end correct_condition_l199_199399


namespace min_value_expr_l199_199903

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x, (x = a^2 + b^2 + 1 / (a + b)^2 + (a^2 * b^2) / (a + b)^2) ∧ x ≥ sqrt 3 :=
begin
  sorry
end

end min_value_expr_l199_199903


namespace replace_asterisk_with_2x_l199_199541

-- Defining conditions for Lean
def expr_with_monomial (a : ℤ) : ℤ := (x : ℤ) → (x^3 - 2)^2 + (x^2 + a * x)^2

-- The statement of the proof in Lean 4
theorem replace_asterisk_with_2x : expr_with_monomial 2 = (x : ℤ) → x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_asterisk_with_2x_l199_199541


namespace equal_intercepts_imp_a_zero_or_three_line_not_passing_third_quadrant_range_l199_199514

-- Definitions based on the condition from part (1)
def line_eq (a : ℝ) (x y : ℝ) : Prop :=
  (a + 1) * x + y - 3 + a = 0

def x_intercept (a : ℝ) : ℝ :=
  (3 - a) / (a + 1)

def y_intercept (a : ℝ) : ℝ :=
  3 - a

def intercepts_equal (a : ℝ) : Prop :=
  x_intercept a = y_intercept a

-- To be proven: If intercepts are equal, a must be 0 or 3
theorem equal_intercepts_imp_a_zero_or_three (a : ℝ) :
  intercepts_equal a → a = 0 ∨ a = 3 :=
  sorry

-- Definitions based on the condition from part (2)
def does_not_pass_third_quadrant (a : ℝ) (x y : ℝ) : Prop :=
  ¬ ((x < 0) ∧ (y < 0))

-- To be proven: If the line does not pass the third quadrant, -1 ≤ a ≤ 3
theorem line_not_passing_third_quadrant_range (a : ℝ) :
  (∀ (x y : ℝ), line_eq a x y → does_not_pass_third_quadrant a x y) → -1 ≤ a ∧ a ≤ 3 :=
  sorry

end equal_intercepts_imp_a_zero_or_three_line_not_passing_third_quadrant_range_l199_199514


namespace birgit_hiking_time_l199_199055

def hiking_conditions
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ) : Prop :=
  ∃ (average_speed_time : ℝ) (birgit_speed_time : ℝ) (total_minutes_hiked : ℝ),
    total_minutes_hiked = hours_hiked * 60 ∧
    average_speed_time = total_minutes_hiked / distance_km ∧
    birgit_speed_time = average_speed_time - time_faster ∧
    (birgit_speed_time * distance_target_km = 48)

theorem birgit_hiking_time
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ)
  : hiking_conditions hours_hiked distance_km time_faster distance_target_km :=
by
  use 10, 6, 210
  sorry

end birgit_hiking_time_l199_199055


namespace train_speed_l199_199705

open_locale real

theorem train_speed (carriages : ℕ) (engine : ℕ) (length : ℝ) (bridge_length_km : ℝ) (time_min : ℝ) :
  (carriages + engine) * length / (time_min / 60) = 60 :=
by {
  -- Definitions based on the given conditions
  let number_of_carriages := 24,
  let number_of_engines := 1,
  let length_of_each := 60, -- in meters
  let length_of_bridge_km := 3.5, -- in kilometers
  let time_to_cross_min := 5, -- in minutes

  -- Prove the speed is 60 km/h
  sorry
}

end train_speed_l199_199705


namespace arccos_of_half_eq_pi_over_three_l199_199301

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199301


namespace angle_Z_130_degrees_l199_199923

theorem angle_Z_130_degrees {p q : Line} {X Y Z : Point}
  (h_parallel : p ∥ q)
  (h_angle_X : m∠X = 100)
  (h_angle_Y : m∠Y = 130) :
  m∠Z = 130 :=
sorry

end angle_Z_130_degrees_l199_199923


namespace circle_and_tangent_lines_l199_199785

noncomputable def circle_eq (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

noncomputable def tangent_line_eq (x y : ℝ) : Prop := (x = 2 ∨ 3 * x - 4 * y + 26 = 0)

theorem circle_and_tangent_lines :
  (∃ x y : ℝ, (x = 0 ∧ y = -6) ∨ (x = 1 ∧ y = -5) ∧ circle_eq x y) ∧
  (∃ x y : ℝ, x - y + 1 = 0 ∧ circle_eq x y) ∧
  (∃ x y : ℝ, (x = 2 ∧ y = 8) ∧ tangent_line_eq x y) :=
begin
  sorry
end

end circle_and_tangent_lines_l199_199785


namespace beautiful_ellipse_slopes_product_l199_199960

theorem beautiful_ellipse_slopes_product (a b : ℝ) (h : a > b ∧ b > 0)
    (e : ℝ) (he : e = (Real.sqrt 5 - 1) / 2)
    (P : ℝ × ℝ) (hp : (P.fst / a)^2 + (P.snd / b)^2 = 1 ∧ P ≠ (-a, 0) ∧ P ≠ (a, 0)) :
    let k1 := (b * P.snd) / (a * Real.cos P.fst + a)
        k2 := (b * P.snd) / (a * Real.cos P.fst - a)
    in k1 * k2 = (1 - Real.sqrt 5) / 2 :=
by sorry

end beautiful_ellipse_slopes_product_l199_199960


namespace total_snakes_l199_199525

noncomputable def total_snakes_count 
    (only_dogs: ℕ) 
    (only_cats: ℕ) 
    (only_snakes: ℕ) 
    (only_rabbits: ℕ) 
    (only_birds: ℕ) 
    (dogs_cats: ℕ) 
    (dogs_snakes: ℕ) 
    (dogs_rabbits: ℕ) 
    (dogs_birds: ℕ) 
    (cats_snakes: ℕ) 
    (cats_rabbits: ℕ) 
    (cats_birds: ℕ) 
    (snakes_rabbits: ℕ) 
    (snakes_birds: ℕ) 
    (rabbits_birds: ℕ) 
    (dogs_cats_snakes: ℕ) 
    (dogs_cats_rabbits: ℕ) 
    (dogs_cats_birds: ℕ) 
    (dogs_snakes_rabbits: ℕ) 
    (cats_snakes_rabbits: ℕ) 
    (all_pets: ℕ) 
    (total_people: ℕ) : ℕ :=
  only_snakes + dogs_snakes + cats_snakes + snakes_rabbits + snakes_birds + 
  dogs_cats_snakes + dogs_snakes_rabbits + cats_snakes_rabbits + all_pets

theorem total_snakes 
    (only_dogs: ℕ) 
    (only_cats: ℕ) 
    (only_snakes: ℕ = 8) 
    (only_rabbits: ℕ) 
    (only_birds: ℕ) 
    (dogs_cats: ℕ) 
    (dogs_snakes: ℕ = 7)
    (dogs_rabbits: ℕ) 
    (dogs_birds: ℕ) 
    (cats_snakes: ℕ = 9) 
    (cats_rabbits: ℕ) 
    (cats_birds: ℕ) 
    (snakes_rabbits: ℕ = 5) 
    (snakes_birds: ℕ = 3) 
    (rabbits_birds: ℕ) 
    (dogs_cats_snakes: ℕ = 4) 
    (dogs_cats_rabbits: ℕ) 
    (dogs_cats_birds: ℕ) 
    (dogs_snakes_rabbits: ℕ = 3) 
    (cats_snakes_rabbits: ℕ = 2) 
    (all_pets: ℕ = 1) 
    (total_people: ℕ = 125) : 
    total_snakes_count only_dogs only_cats only_snakes only_rabbits only_birds 
      dogs_cats dogs_snakes dogs_rabbits dogs_birds cats_snakes cats_rabbits 
      cats_birds snakes_rabbits snakes_birds rabbits_birds dogs_cats_snakes 
      dogs_cats_rabbits dogs_cats_birds dogs_snakes_rabbits cats_snakes_rabbits 
      all_pets total_people = 42 := by 
  sorry

end total_snakes_l199_199525


namespace translated_function_fixed_point_l199_199674

theorem translated_function_fixed_point
  (a : ℝ)
  (h1 : ∀ x, (λ x, a * x) 0 = 1) :
  ∃ x y, (x = 3) ∧ (y = 4) ∧ (λ x, a * x - 3 + 3 = y) x :=
begin
  sorry
end

end translated_function_fixed_point_l199_199674


namespace max_y_difference_l199_199378

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference : 
  ∃ x1 x2 : ℝ, 
    f x1 = g x1 ∧ f x2 = g x2 ∧ 
    (∀ x : ℝ, f x = g x → x = x1 ∨ x = x2) ∧ 
    abs ((f x1) - (f x2)) = 2 := 
by
  sorry

end max_y_difference_l199_199378


namespace parity_of_sequence_l199_199349

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 2 else
  if n = 1 then 3 else
  if n = 2 then 4 else
  sequence (n - 1) + 2 * sequence(n - 2) - sequence(n - 3)

theorem parity_of_sequence :
  (sequence 10 % 2, sequence 11 % 2, sequence 12 % 2) = (1, 0, 1) :=
by
  sorry

end parity_of_sequence_l199_199349


namespace part1_solution_part2_solution_l199_199052

def f (x : ℝ) (a : ℝ) := |x + 1| - |a * x - 1|

-- Statement for part 1
theorem part1_solution (x : ℝ) : (f x 1 > 1) ↔ (x > 1 / 2) := sorry

-- Statement for part 2
theorem part2_solution (a : ℝ) (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (f x a > x) ↔ (0 < a ∧ a ≤ 2) := sorry

end part1_solution_part2_solution_l199_199052


namespace imaginary_part_is_neg_one_l199_199733

noncomputable def complex_z : ℂ := 2 / (-1 + Complex.i)

theorem imaginary_part_is_neg_one : complex_z.im = -1 := 
by
  -- provide the proof here
  sorry

end imaginary_part_is_neg_one_l199_199733


namespace probability_heads_penny_dime_quarter_l199_199570

theorem probability_heads_penny_dime_quarter :
  let total_outcomes := 2^5 in
  let favorable_outcomes := 2^2 in
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 8 :=
by
  let total_outcomes : ℚ := 32
  let favorable_outcomes : ℚ := 4
  have probability : ℚ := favorable_outcomes / total_outcomes
  have expected : ℚ := 1 / 8
  show probability = expected
  sorry

end probability_heads_penny_dime_quarter_l199_199570


namespace smallest_piece_not_form_triangle_l199_199708

theorem smallest_piece_not_form_triangle :
  ∃ z : ℕ, z ≥ 8 ∧ 
        ∀ z' : ℕ, z' < z → (13 - z' / 2 + (22 - z')) > (25 - z') :=
begin
  sorry
end

end smallest_piece_not_form_triangle_l199_199708


namespace cos_sin_sum_eq_l199_199079

theorem cos_sin_sum_eq :
  (let α := atan 3 in cos α + sin α = 2 * sqrt 10 / 5) :=
sorry

end cos_sin_sum_eq_l199_199079


namespace negation_proposition_l199_199074

theorem negation_proposition :
  (∀ x : ℝ, 0 < x → x^2 + 1 ≥ 2 * x) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proposition_l199_199074


namespace cafe_tables_l199_199065

theorem cafe_tables (seats_base8 : ℕ) (seats_per_table : ℕ) (seats_base8 = 0o312) (seats_per_table = 3) :
  let seats_base10 := 3 * 8^2 + 1 * 8^1 + 2 * 8^0 in
  let number_of_tables := seats_base10 / seats_per_table in
  number_of_tables = 67 :=
by
  -- The proof body would go here
  sorry

end cafe_tables_l199_199065


namespace smallest_m_l199_199720

theorem smallest_m (n : ℕ) (hn : 1 < n) : ∃ (m : ℕ), (∀ (a b : ℕ), 
  a ∈ finset.range (2 * n) ∧ a ≠ 0 → b ∈ finset.range (2 * n) ∧ b ≠ 0 ∧ a ≠ b → 
  (∃ x y : ℕ, (x ≠ 0 ∨ y ≠ 0) ∧ 2 * n ∣ (a * x + b * y) ∧ x + y ≤ m)) ∧ (m = n) :=
  by
    sorry

end smallest_m_l199_199720


namespace least_pennies_l199_199656

theorem least_pennies : 
  ∃ (a : ℕ), a % 5 = 1 ∧ a % 3 = 2 ∧ a = 11 :=
by
  sorry

end least_pennies_l199_199656


namespace find_f_prime_one_l199_199421

noncomputable def f : ℝ → ℝ := λ x, 2 * x * (D f 1) + Real.log x

theorem find_f_prime_one (f' : ℝ → ℝ) (D_f : ∀ x, D f' x = 2 * D f' 1 + 1 / x) :
    D f' 1 = -1 :=
by
  sorry

end find_f_prime_one_l199_199421


namespace planes_parallel_l199_199031

variables (m n α β : Type)
variable [affine_space m]
variable [affine_space n]
variable [affine_plane α]
variable [affine_plane β]

def parallel_lines (x y : Type) : Prop := sorry
def perpendicular_lines (x y : Type) : Prop := sorry

theorem planes_parallel (h1 : perpendicular_lines m α) 
                       (h2 : perpendicular_lines n β)
                       (h3 : parallel_lines m n)
                       : parallel_planes α β := sorry

end planes_parallel_l199_199031


namespace racing_track_width_l199_199698

theorem racing_track_width (r1 r2 : ℝ) (h : 2 * real.pi * r1 - 2 * real.pi * r2 = 20 * real.pi) : r1 - r2 = 10 :=
by
  sorry

end racing_track_width_l199_199698


namespace scientific_notation_l199_199883

theorem scientific_notation (n : ℤ) (hn : n = 12910000) : ∃ a : ℝ, (a = 1.291 ∧ hn = (a * 10^7).to_int) :=
by
  sorry

end scientific_notation_l199_199883


namespace maximal_airports_travel_l199_199672

theorem maximal_airports_travel:
  ∀ (n p : ℕ), 1 < n → Prime p → p ∣ n →
  (∀ (states : Fin p → Fin n → Fin n), ∃ (air_company : Fin p) 
    (airports_chosen : Fin n → Prop), 
    (∀ (i j : Fin n), airports_chosen i ∧ airports_chosen j → 
      ∃ (path : List (Fin n)), List.Chain' (λ x y, (states x y) = air_company) (i :: path ++ [j]))) →
  ∃ (N : ℕ), N = n :=
begin
  intros n p hn hp hpn h,
  use n,
  sorry,
end

end maximal_airports_travel_l199_199672


namespace interval_of_monotonicity_bc_value_l199_199015

noncomputable theory
open Real

def f (x : ℝ) : ℝ := sin (2 * x + π / 6) - 2 * (cos x) ^ 2

theorem interval_of_monotonicity :
  ∀ k : ℤ, 
  ∀ x : ℝ, 
  -π / 6 + (k:ℝ) * π ≤ x ∧ x ≤ π / 3 + (k:ℝ) * π ↔ derivative (λ x, f x) x > 0 := sorry

theorem bc_value (A B C D : ℝ) (k : ℤ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = sin (2 * x + π / 6) - 2 * cos x ^ 2) →
  f (A / 2 - π / 6) = -5 / 4 →
  ∥C - D∥ = 2 * ∥D - A∥ →
  ∥B - D∥ = √10 →
  cos (angle B D A) = √10 / 4 →
  B C = 6 := sorry

end interval_of_monotonicity_bc_value_l199_199015


namespace geometric_sequence_general_term_sum_of_b_terms_l199_199402

variable {n : ℕ} (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Conditions: Given a geometric sequence {a_n} that satisfies a_2 = 2 a_1,
-- and a_2 + 1 is the arithmetic mean between a_1 and a_3.
-- Definitions
def is_geometric_sequence (a : ℕ → ℕ) :=
  ∃ q : ℕ, ∀ n : ℕ, a (n + 1) = q * a n

noncomputable def geometric_sequence_condition (a : ℕ → ℕ) :=
  a 2 = 2 * a 1 ∧ 2 * (a 2 + 1) = a 1 + a 3

-- Prove the general formula for the nth term of the sequence {a_n}.
theorem geometric_sequence_general_term (h : is_geometric_sequence a) (hc : geometric_sequence_condition a) :
  ∀ n : ℕ, a n = 2 ^ n := sorry

-- Given b_n = a_n - 2 * log_2 (a_n), prove the sum of the first n terms S_n.
def b (a : ℕ → ℕ) (n : ℕ) := a n - 2 * Nat.log2 (a n)

noncomputable def S (b : ℕ → ℕ) (n : ℕ) := ∑ k in Finset.range n, b k

-- Final result for S_n
theorem sum_of_b_terms (h : is_geometric_sequence a) (hc : geometric_sequence_condition a) :
  ∀ n : ℕ, S b n = 2 ^ (n + 1) - n^2 - n - 2 := sorry

end geometric_sequence_general_term_sum_of_b_terms_l199_199402


namespace find_ellipse_equation_find_constant_value_l199_199481

section Problem1

-- Define the constants and equations
variables (a b c : ℝ) (hx : a > b) (hy : b > 0)
def ellipse_eq (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
def eccentricity : ℝ := c / a

-- Given constraint condition
theorem find_ellipse_equation 
  (e : eccentricity = sqrt 6 / 3) 
  (line_perpendicular : ∀ x y, ellipse_eq x y → x = sqrt 6 / 3) 
  (chord_length : 2 * b = 2 * sqrt 6 / 3) : 
  a = sqrt 6 / 3 ∧ b = IT sqrt 6 / 3 ∧ ellipse_eq_a_b := sorry

end Problem1

section Problem2

-- Variables and equations
variables (x0 k : ℝ)
def EA_sq (x y : ℝ) (x0 : ℝ) := (x - x0)^2 + y^2
def EB_sq (x y : ℝ) (x0 : ℝ) := (x + x0)^2 + y^2

-- Define coordinates of eccentric point E
def point_E (x : ℝ) := (x = abs k ∨ x = -abs k)
def constant_value (val : ℝ) := val = 2

-- Given proof theorem
theorem find_constant_value 
  (E_condition : point_E (sqrt 3))
  (EA_val : ∀ x y, EA_sq x y (sqrt 3) = 1)
  (EB_val : ∀ x y, EB_sq x y (sqrt 3) = 1)
  : ∃ (E : ℝ √3), constant_value (2) := sorry

end Problem2

end find_ellipse_equation_find_constant_value_l199_199481


namespace convex_polygon_condition_l199_199591

variable {n : ℕ} {a b : Fin n → ℝ}

def origin_in_convex_hull (a b : Fin n → ℝ) : Prop :=
  ∃ (λ : Fin n → ℝ), (∀ i, 0 ≤ λ i) ∧ (∑ i, λ i = 1) ∧
    (∑ i, λ i * a i = 0) ∧ (∑ i, λ i * b i = 0)

theorem convex_polygon_condition (h : origin_in_convex_hull a b) : 
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ 
    (∑ i, a i * x^(a i) * y^(b i) = 0) ∧ 
    (∑ i, b i * x^(a i) * y^(b i) = 0) :=
  sorry

end convex_polygon_condition_l199_199591


namespace pascal_diff_half_l199_199792

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k       => 0
| n+1, k+1   => binom n k + binom n (k+1)

def a_i (i : ℕ) : ℕ := binom 2020 i
def b_i (i : ℕ) : ℕ := binom 2021 i
def c_i (i : ℕ) : ℕ := binom 2022 i

theorem pascal_diff_half :
  (∑ i in Finset.range (2021 + 1), b_i i / c_i i) - (∑ i in Finset.range 2021, a_i i / b_i i) = 1 / 2 := sorry

end pascal_diff_half_l199_199792


namespace arccos_one_half_l199_199230

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199230


namespace arccos_one_half_l199_199224

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199224


namespace arccos_one_half_l199_199223

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199223


namespace general_formula_for_arithmetic_sequence_l199_199403

-- Define the arithmetic sequence as a function
noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

-- Define fn(x) and the condition that f_n(1) = n^2
def f_n (a : ℕ) (seq : ℕ → ℕ) (n : ℕ) (x : ℕ) : ℕ :=
  a + ∑ i in Finset.range (n + 1), seq i * x^i

theorem general_formula_for_arithmetic_sequence 
  (a : ℕ) (seq : ℕ → ℕ) :
  (∀ n : ℕ, f_n a seq n 1 = n^2) →
  (∀ n : ℕ, seq n = 2 * n - 1) :=
  by
    intros h n
    sorry

end general_formula_for_arithmetic_sequence_l199_199403


namespace D_72_l199_199500

/-- D(n) denotes the number of ways of writing the positive integer n
    as a product n = f1 * f2 * ... * fk, where k ≥ 1, the fi are integers
    strictly greater than 1, and the order in which the factors are
    listed matters. -/
def D (n : ℕ) : ℕ := sorry

theorem D_72 : D 72 = 43 := sorry

end D_72_l199_199500


namespace divisibility_3804_l199_199047

theorem divisibility_3804 (n : ℕ) (h : 0 < n) :
    3804 ∣ ((n ^ 3 - n) * (5 ^ (8 * n + 4) + 3 ^ (4 * n + 2))) :=
sorry

end divisibility_3804_l199_199047


namespace number_of_valid_pairs_l199_199385

noncomputable def count_valid_pairs : ℕ :=
  let sqrt_2 : ℝ := Real.sqrt 2
  let sqrt_3 : ℝ := Real.sqrt 3
  let valid_pair (a b : ℕ) : Prop :=
    (a ≤ 20) ∧ (b ≤ 20) ∧
    let r1 := a * sqrt_2 + b * sqrt_3 - ⌊a * sqrt_2 + b * sqrt_3 / sqrt_2⌋ * sqrt_2
    let r2 := a * sqrt_2 + b * sqrt_3 - ⌊a * sqrt_2 + b * sqrt_3 / sqrt_3⌋ * sqrt_3
    r1 + r2 = sqrt_2
  {n : ℕ // n = 16 ∧ ∃ f : Fin n -> (ℕ × ℕ), ∀ i : Fin n, valid_pair (f i).1 (f i).2}

theorem number_of_valid_pairs : ∃ (n : ℕ), count_valid_pairs = n ∧ n = 16 := sorry

end number_of_valid_pairs_l199_199385


namespace number_of_sets_B_l199_199967

theorem number_of_sets_B :
  {B : set ℕ // {1, 2} ∪ B = {1, 2, 3, 4, 5}}.card = 4 :=
sorry

end number_of_sets_B_l199_199967


namespace arccos_one_half_l199_199237

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199237


namespace Q_investment_l199_199526

theorem Q_investment :
  ∃ (X : ℕ), (75000 / X) = 5 ∧ X = 15000 :=
begin
  use 15000,
  split,
  { norm_num,
    exact 75000 / 15000 = 5 },
  { norm_num }
end

end Q_investment_l199_199526


namespace train_cross_time_approx_l199_199848

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def bridge_length : ℝ := 350 -- meters
noncomputable def train_speed_kmph : ℝ := 25 -- kmph

noncomputable def total_distance : ℝ := train_length + bridge_length -- 600 meters
noncomputable def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600 -- 6.944 m/s

theorem train_cross_time_approx :
  (total_distance / train_speed_mps) ≈ 86.4 := by
  sorry

end train_cross_time_approx_l199_199848


namespace correct_average_of_ten_numbers_l199_199668

theorem correct_average_of_ten_numbers :
  let incorrect_average := 20 
  let num_values := 10 
  let incorrect_number := 26
  let correct_number := 86 
  let incorrect_total_sum := incorrect_average * num_values
  let correct_total_sum := incorrect_total_sum - incorrect_number + correct_number 
  (correct_total_sum / num_values) = 26 := 
by
  sorry

end correct_average_of_ten_numbers_l199_199668


namespace find_C_l199_199164

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 300) 
  (h2 : A + C = 200) 
  (h3 : B + C = 350) : 
  C = 250 := 
  by sorry

end find_C_l199_199164


namespace arccos_half_eq_pi_div_three_l199_199290

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199290


namespace sin_theta_area_l199_199160

theorem sin_theta_area (A a m : ℝ) (hA : A = 50) (ha : a = 12) (hm : m = 13) :
  ∃ θ : ℝ, real.sin θ = 25 / 39 ∧ A = 1 / 2 * a * m * real.sin θ :=
by
  use real.arcsin (25 / 39)
  constructor
  · -- Prove that real.sin θ = 25 / 39
    sorry
  · -- Prove the area equation holds
    rw [hA, ha, hm]
    norm_num
    sorry

end sin_theta_area_l199_199160


namespace no_2021_residents_possible_l199_199749

-- Definition: Each islander is either a knight or a liar
def is_knight_or_liar (i : ℕ) : Prop := true -- Placeholder definition for either being a knight or a liar

-- Definition: Knights always tell the truth
def knight_tells_truth (i : ℕ) : Prop := true -- Placeholder definition for knights telling the truth

-- Definition: Liars always lie
def liar_always_lies (i : ℕ) : Prop := true -- Placeholder definition for liars always lying

-- Definition: Even number of knights claimed by some islanders
def even_number_of_knights : Prop := true -- Placeholder definition for the claim of even number of knights

-- Definition: Odd number of liars claimed by remaining islanders
def odd_number_of_liars : Prop := true -- Placeholder definition for the claim of odd number of liars

-- Question and proof problem
theorem no_2021_residents_possible (K L : ℕ) (h1 : K + L = 2021) (h2 : ∀ i, is_knight_or_liar i) 
(h3 : ∀ k, knight_tells_truth k → even_number_of_knights) 
(h4 : ∀ l, liar_always_lies l → odd_number_of_liars) : 
  false := sorry

end no_2021_residents_possible_l199_199749


namespace arccos_half_eq_pi_over_three_l199_199208

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199208


namespace arccos_one_half_l199_199234

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199234


namespace arccos_half_eq_pi_div_three_l199_199307

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199307


namespace b_n_arithmetic_sum_c_n_l199_199788

-- Define the sequences a_n, b_n, and c_n
def a_n (n : ℕ) : ℚ := (1 / 4) ^ n
def b_n (n : ℕ) : ℚ := 3 * n - 2
def c_n (n : ℕ) : ℚ := a_n n * b_n n

-- Prove that b_n is an arithmetic sequence
theorem b_n_arithmetic : ∀ n : ℕ, b_n (n + 1) - b_n n = 3 := by
  intro n
  simp only [b_n]
  rw [sub_eq_add_neg, add_mul, mul_one, add_assoc, add_comm (3*n)]
  linarith

-- Prove the sum of the first n terms of the sequence c_n
theorem sum_c_n (n : ℕ) : ∑ i in finset.range n, c_n i = (2 / 3) - ((3 * n + 2) / 3) * (1 / 4) ^ n := by
  sorry

end b_n_arithmetic_sum_c_n_l199_199788


namespace z_in_second_quadrant_l199_199861

-- Definition of the complex number (given condition)
def z : ℂ := (3 + complex.i) * complex.i

-- Statement that needs to be proved: z is in the second quadrant
theorem z_in_second_quadrant : z.re < 0 ∧ z.im > 0 := by
  sorry

end z_in_second_quadrant_l199_199861


namespace probability_particle_at_2_3_after_5_moves_l199_199695

noncomputable def binom (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k
else 0

theorem probability_particle_at_2_3_after_5_moves :
  let p_right := 1 / 2
  let p_up := 1 / 2
  let moves := 5
  let right_moves := 2
  let up_moves := 3
  let total_probability := (binom moves right_moves) * (p_right ^ right_moves) * (p_up ^ up_moves)
  total_probability = binom moves right_moves * (1/2)^5 := by sorry

end probability_particle_at_2_3_after_5_moves_l199_199695


namespace arccos_half_is_pi_div_three_l199_199266

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199266


namespace fried_frog_probability_l199_199389

def Grid : Type :=
| Center
| CornerA
| CornerB
| CornerD
| CornerE
| EdgeF
| EdgeG
| EdgeH
| EdgeI

-- Transition probabilities for movement including wrap-around rules
def transition_prob (state : Grid) : Grid → ℚ :=
  match state with
  | Grid.Center => fun g => if g = Grid.EdgeF ∨ g = Grid.EdgeG ∨ g = Grid.EdgeH ∨ g = Grid.EdgeI then 1/4 else 0
  | Grid.EdgeF => fun g => if g = Grid.Center ∨ g = Grid.CornerA ∨ g = Grid.EdgeG ∨ g = Grid.EdgeH then 1/4 else 0
  | Grid.EdgeG => fun g => if g = Grid.Center ∨ g = Grid.CornerB ∨ g = Grid.EdgeF ∨ g = Grid.EdgeI then 1/4 else 0
  | Grid.EdgeH => fun g => if g = Grid.Center ∨ g = Grid.CornerD ∨ g = Grid.EdgeF ∨ g = Grid.EdgeI then 1/4 else 0
  | Grid.EdgeI => fun g => if g = Grid.Center ∨ g = Grid.CornerE ∨ g = Grid.EdgeG ∨ g = Grid.EdgeH then 1/4 else 0
  | _ => fun _ => 0

-- Recursive probability of reaching a corner square in n hops
def p_n (n : ℕ) : Grid → ℚ
| 0 => fun state => if state = Grid.CornerA ∨ state = Grid.CornerB ∨ state = Grid.CornerD ∨ state = Grid.CornerE then 1 else 0
| (n+1) => fun state => (∑ t : Grid, transition_prob state t * p_n n t)

theorem fried_frog_probability : p_n 4 Grid.Center = 25/32 :=
by
  sorry

end fried_frog_probability_l199_199389


namespace symmetry_axis_shifted_sine_l199_199082

theorem symmetry_axis_shifted_sine :
  let f (x : ℝ) := sin (π / 6 - 2 * x)
  let g (x : ℝ) := f (x - π / 12)
  ∃ k : ℤ, ∀ x : ℝ, g(x) = -sin (2x - π / 3) → x = k * (π / 2) + 5 * π / 12 :=
  sorry

end symmetry_axis_shifted_sine_l199_199082


namespace track_length_l199_199044

theorem track_length (h₁ : ∀ (x : ℕ), (exists y₁ y₂ : ℕ, y₁ = 120 ∧ y₂ = 180 ∧ y₁ + y₂ = x ∧ (y₂ - y₁ = 60) ∧ (y₂ = x - 120))) : 
  ∃ x : ℕ, x = 600 := by
  sorry

end track_length_l199_199044


namespace speed_of_first_car_l199_199618

-- Define the conditions
def t : ℝ := 3.5
def v : ℝ := sorry -- (To be solved in the proof)
def speed_second_car : ℝ := 58
def total_distance : ℝ := 385

-- The distance each car travels after t hours
def distance_first_car : ℝ := v * t
def distance_second_car : ℝ := speed_second_car * t

-- The equation representing the total distance between the two cars after 3.5 hours
def equation := distance_first_car + distance_second_car = total_distance

-- The main theorem stating the speed of the first car
theorem speed_of_first_car : v = 52 :=
by
  -- The important proof steps would go here solving the equation "equation".
  sorry

end speed_of_first_car_l199_199618


namespace arccos_half_eq_pi_div_three_l199_199337

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199337


namespace arccos_of_half_eq_pi_over_three_l199_199306

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199306


namespace cryptarithm_solved_l199_199369

-- Definitions for the digits A, B, C
def valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

-- Given conditions, where A, B, C are distinct non-zero digits
def conditions (A B C : ℕ) : Prop :=
  valid_digit A ∧ valid_digit B ∧ valid_digit C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C

-- Definitions of the two-digit and three-digit numbers
def two_digit (A B : ℕ) : ℕ := 10 * A + B
def three_digit_rep (C : ℕ) : ℕ := 111 * C

-- Main statement of the proof problem
theorem cryptarithm_solved (A B C : ℕ) (h : conditions A B C) :
  two_digit A B + A * three_digit_rep C = 247 → A * 100 + B * 10 + C = 251 :=
sorry -- Proof goes here

end cryptarithm_solved_l199_199369


namespace sahil_selling_price_l199_199051

noncomputable def sales_tax : ℝ := 0.10 * 18000
noncomputable def initial_cost_with_tax : ℝ := 18000 + sales_tax

noncomputable def broken_part_cost : ℝ := 3000
noncomputable def software_update_cost : ℝ := 4000
noncomputable def total_repair_cost : ℝ := broken_part_cost + software_update_cost
noncomputable def service_tax_on_repair : ℝ := 0.05 * total_repair_cost
noncomputable def total_repair_cost_with_tax : ℝ := total_repair_cost + service_tax_on_repair

noncomputable def transportation_charges : ℝ := 1500
noncomputable def total_cost_before_depreciation : ℝ := initial_cost_with_tax + total_repair_cost_with_tax + transportation_charges

noncomputable def depreciation_first_year : ℝ := 0.15 * total_cost_before_depreciation
noncomputable def value_after_first_year : ℝ := total_cost_before_depreciation - depreciation_first_year

noncomputable def depreciation_second_year : ℝ := 0.15 * value_after_first_year
noncomputable def value_after_second_year : ℝ := value_after_first_year - depreciation_second_year

noncomputable def profit : ℝ := 0.50 * value_after_second_year
noncomputable def selling_price : ℝ := value_after_second_year + profit

theorem sahil_selling_price : selling_price = 31049.44 := by
  sorry

end sahil_selling_price_l199_199051


namespace arccos_one_half_is_pi_div_three_l199_199195

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199195


namespace height_of_water_in_Tank_A_value_of_c_plus_d_l199_199610

-- Define the volumes and corresponding variables
def volume_of_cone (radius height : ℝ) : ℝ :=
  (1 / 3) * π * (radius ^ 2) * height

def height_of_water_in_cone (full_height scale_factor : ℝ) : ℝ :=
  full_height * scale_factor

-- Define the given conditions
def radius_A : ℝ := 20
def height_A : ℝ := 80
def volume_A := volume_of_cone radius_A height_A
def fill_percentage_A : ℝ := 30 / 100

-- Prove that height of water in Tank A is 80∛(3/2)
theorem height_of_water_in_Tank_A : 
  let x := (fill_percentage_A)^(1 / 3)
  in height_of_water_in_cone height_A x = 80 * (real.cbrt (3 / 2)) := by
  sorry

-- The components c and d and their addition
def c : ℝ := 80
def d : ℝ := 3

theorem value_of_c_plus_d : c + d = 83 := by 
  simp [c, d]

end height_of_water_in_Tank_A_value_of_c_plus_d_l199_199610


namespace value_of_f_l199_199740

def B : Set ℚ := {x | x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℝ := sorry

noncomputable def h (x : ℚ) : ℚ :=
  1 / (1 - x)

lemma cyclic_of_h :
  ∀ x ∈ B, h (h (h x)) = x :=
sorry

lemma functional_property (x : ℚ) (hx : x ∈ B) :
  f x + f (h x) = 2 * Real.log (|x|) :=
sorry

theorem value_of_f :
  f 2023 = Real.log 2023 :=
sorry

end value_of_f_l199_199740


namespace icosahedron_num_interior_diagonals_l199_199442

-- Define an icosahedron structure
structure Icosahedron where
  num_faces : ℕ := 20       -- 20 triangular faces
  num_vertices : ℕ := 12    -- 12 vertices
  faces_per_vertex : ℕ := 5 -- 5 faces meeting at each vertex

-- Define the concept of an interior diagonal
def is_interior_diagonal (i : Icosahedron) (v1 v2 : ℕ) : Prop :=
  ¬ {v1, v2} ∈ { {v | (v ∈ i.num_vertices) ∧ (v ∉ {v1, v2}) ∧ ∃ f, f ∈ i.num_faces ∧ v1 ∈ f ∧ v2 ∈ f} }

-- State the theorem: The number of interior diagonals of an icosahedron
theorem icosahedron_num_interior_diagonals (i : Icosahedron) : 
  ∃ n, n = 36 :=
by
  -- placeholder for the actual proof
  sorry

end icosahedron_num_interior_diagonals_l199_199442


namespace parallel_lines_eq_slope_l199_199455

theorem parallel_lines_eq_slope (a : ℝ) (l1_parallel_l2 : ∀ x y, (ax + y + 2a = 0) → (x + ay + 3 = 0)) : a = 1 ∨ a = -1 :=
by sorry

end parallel_lines_eq_slope_l199_199455


namespace base_7_perfect_square_ab2c_l199_199860

-- Define the necessary conditions
def is_base_7_representation_of (n : ℕ) (a b c : ℕ) : Prop :=
  n = a * 7^3 + b * 7^2 + 2 * 7 + c

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Lean statement for the problem
theorem base_7_perfect_square_ab2c (n a b c : ℕ) (h1 : a ≠ 0) (h2 : is_base_7_representation_of n a b c) (h3 : is_perfect_square n) :
  c = 2 ∨ c = 3 ∨ c = 6 :=
  sorry

end base_7_perfect_square_ab2c_l199_199860


namespace min_distance_of_point_P_on_parabola_l199_199807

-- Define the coordinates of point A, F (focus), and the equation of the parabola
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 4, y := 3 }
def F : Point := { x := 1, y := 0 }  -- Focus of the parabola y^2 = 4x, thus F(1, 0)

-- Define the parabola as a set of points
def parabola : set Point := { P | P.y ^ 2 = 4 * P.x }

-- Definition for distance between two points
def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- The theorem we need to prove
theorem min_distance_of_point_P_on_parabola : 
  ∃ P : Point, P ∈ parabola ∧ (∀ Q ∈ parabola, dist A P + dist P F ≤ dist A Q + dist Q F) ∧ P = { x := 9/4, y := 3 } :=
by
  sorry

end min_distance_of_point_P_on_parabola_l199_199807


namespace subtracting_is_adding_opposite_l199_199673

theorem subtracting_is_adding_opposite (a b : ℚ) : a - b = a + (-b) :=
by sorry

end subtracting_is_adding_opposite_l199_199673


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199323

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199323


namespace function_nonnegative_l199_199351

noncomputable def f (x : ℝ) := (x - 10*x^2 + 35*x^3) / (9 - x^3)

theorem function_nonnegative (x : ℝ) : 
  (f x ≥ 0) ↔ (0 ≤ x ∧ x ≤ (1 / 7)) ∨ (3 ≤ x) :=
sorry

end function_nonnegative_l199_199351


namespace sale_in_first_month_l199_199687

noncomputable def first_month_sale 
    (last_four_months : List ℝ) -- sales for the last four months
    (sixth_month_sale : ℝ)      -- sale for the sixth month
    (average_sale : ℝ)          -- average sale for six months
    (number_of_months : ℕ)      -- number of months
    (expected_first_month_sale : ℝ) -- expected sale in the first month
    : ℝ :=
let total_sales := average_sale * number_of_months
let sales_sum := last_four_months.sum + sixth_month_sale
total_sales - sales_sum

theorem sale_in_first_month
    (last_four_months : List ℝ)
    (sixth_month_sale : ℝ)
    (average_sale : ℝ)
    (number_of_months : ℕ)
    (expected_first_month_sale : ℝ)
    (h1 : last_four_months = [5660, 6200, 6350, 6500])
    (h2 : sixth_month_sale = 7070)
    (h3 : average_sale = 6200)
    (h4 : number_of_months = 6)
    : first_month_sale last_four_months sixth_month_sale average_sale number_of_months 5420 = 5420 := 
by
    unfold first_month_sale
    rw [h1, h2, h3, h4]
    norm_num
    simp
    sorry

end sale_in_first_month_l199_199687


namespace first_studio_students_l199_199091

theorem first_studio_students :
  ∀ (total_students second_studio third_studio : ℕ), 
    total_students = 376 → 
    second_studio = 135 → 
    third_studio = 131 → 
    ∃ (first_studio : ℕ), first_studio = total_students - (second_studio + third_studio) ∧ first_studio = 110 :=
by
  intros total_students second_studio third_studio h1 h2 h3
  use total_students - (second_studio + third_studio)
  split
  . exact rfl
  . rw [h1, h2, h3]
    rfl

end first_studio_students_l199_199091


namespace rectangle_painting_ways_l199_199102

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

theorem rectangle_painting_ways :
  let lines := 6
  in binom lines 2 * binom lines 2 = 225 := by
sorry


end rectangle_painting_ways_l199_199102


namespace find_n_when_a_n_is_zero_l199_199597

def sequence (a : ℕ → ℝ) :=
  (a 1 = 19) ∧
  (a 2 = 98) ∧
  (∀ n : ℕ, a (n + 2) = a n - (2 / a (n + 1)))

theorem find_n_when_a_n_is_zero :
  ∃ n : ℕ, sequence a ∧ a n = 0 ∧ n = 933 :=
sorry

end find_n_when_a_n_is_zero_l199_199597


namespace value_of_x2_plus_4y2_l199_199448

theorem value_of_x2_plus_4y2 (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : x * y = -12) : x^2 + 4*y^2 = 84 := 
  sorry

end value_of_x2_plus_4y2_l199_199448


namespace repeating_decimal_theorem_l199_199452

-- We define the repeating decimal and the given condition
def repeating_decimal (n : ℕ) : ℚ :=
  let base := 31 in
  base / (10^n - 10^(n-2))

-- Now we state the problem
theorem repeating_decimal_theorem (n : ℕ) (h : (10^5 - 10^3) * repeating_decimal n = 31) : 
  repeating_decimal n = 1 / 3168 := 
by 
  sorry

end repeating_decimal_theorem_l199_199452


namespace find_rate_of_current_l199_199978

noncomputable def rate_of_current (speed_boat : ℝ) (distance_downstream : ℝ) (time_downstream_min : ℝ) : ℝ :=
 speed_boat + 3

theorem find_rate_of_current : ∀ (speed_boat distance_downstream time_downstream_min : ℝ), 
  speed_boat = 15 → 
  distance_downstream = 3.6 → 
  time_downstream_min = 12 → 
  rate_of_current speed_boat distance_downstream time_downstream_min = 18 - speed_boat :=
by
  intros speed_boat distance_downstream time_downstream_min h_speed h_distance h_time
  rw [h_speed, h_distance, h_time]
  have time_hours : ℝ := (12 / 60 : ℝ)
  calc
    (rate_of_current 15 3.6 12) = speed_boat + 3 : rfl
    ... = 15 + 3 : by rw [h_speed]
    ... = 18 - speed_boat : by rw [h_speed]

end find_rate_of_current_l199_199978


namespace round_308607_to_nearest_l199_199165

theorem round_308607_to_nearest : Real.round 308.607 = 309 := by
  sorry

end round_308607_to_nearest_l199_199165


namespace replace_asterisk_with_2x_l199_199534

theorem replace_asterisk_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by sorry

end replace_asterisk_with_2x_l199_199534


namespace algebraic_expression_meaningful_range_l199_199859

-- Definitions based on conditions
def condition1 (x : ℝ) : Prop := x ≤ 3
def condition2 (x : ℝ) : Prop := x > 1
def condition3 (x : ℝ) : Prop := x ≠ 2

-- The theorem statement representing the proof problem
theorem algebraic_expression_meaningful_range (x : ℝ) :
  condition1 x → condition2 x → condition3 x →
  1 < x ∧ x ≤ 3 ∧ x ≠ 2 :=
begin
  intro h1,
  intro h2,
  intro h3,
  split,
  {
    exact h2,
  },
  {
    split,
    {
      exact h1,
    },
    {
      exact h3,
    }
  }
end

end algebraic_expression_meaningful_range_l199_199859


namespace number_of_incorrect_propositions_l199_199851

def line : Type := sorry -- Placeholder type for lines
def plane : Type := sorry -- Placeholder type for planes

noncomputable def is_perpendicular (m n : line) : Prop := sorry
noncomputable def is_parallel_line (m n : line) : Prop := sorry
noncomputable def is_parallel_plane (α β : plane) : Prop := sorry
noncomputable def is_perpendicular_plane (α : plane) (m : line) : Prop := sorry
noncomputable def equal_angles (m n : line) (α : plane) : Prop := sorry

theorem number_of_incorrect_propositions (m n : line) (α β γ : plane) :
    (is_perpendicular m n ∧ is_parallel_plane α β ∧ is_parallel_plane α (m : plane)).not ∧
    (is_perpendicular_plane α γ ∧ is_perpendicular_plane β γ ∧ is_perpendicular_plane α β).not ∧
    (is_perpendicular_plane α m ∧ is_perpendicular m n ∧ is_parallel_plane α (n : plane)).not ∧
    (equal_angles m n α ∧ is_parallel_line m n).not →
    4 :=
sorry

end number_of_incorrect_propositions_l199_199851


namespace max_area_14_5_l199_199516

noncomputable def rectangle_max_area (P D : ℕ) (x y : ℝ) : ℝ :=
  if (2 * x + 2 * y = P) ∧ (x^2 + y^2 = D^2) then x * y else 0

theorem max_area_14_5 :
  ∃ (x y : ℝ), (2 * x + 2 * y = 14) ∧ (x^2 + y^2 = 5^2) ∧ rectangle_max_area 14 5 x y = 12.25 :=
by
  sorry

end max_area_14_5_l199_199516


namespace compare_areas_of_rectangles_l199_199000

variable {α : Type*} [LinearOrder α] [Zero α] [AddCommMonoid α] [Mul α]

-- Definitions for the dimensions of the rectangles
variables (a1 b1 a2 b2 : α)

-- Definitions for the perimeters of the rectangles
def P1 : α := 2 * (a1 + b1)
def P2 : α := 2 * (a2 + b2)

-- Areas of the rectangles
def S1 : α := a1 * b1
def S2 : α := a2 * b2

theorem compare_areas_of_rectangles (h : P1 a1 b1 > P2 a2 b2) :
  S1 a1 b1 > S2 a2 b2 ∨ S1 a1 b1 < S2 a2 b2 ∨ S1 a1 b1 = S2 a2 b2 :=
by sorry

end compare_areas_of_rectangles_l199_199000


namespace replace_asterisk_with_2x_l199_199536

theorem replace_asterisk_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by sorry

end replace_asterisk_with_2x_l199_199536


namespace polygon_area_is_400_l199_199746

def Point : Type := (ℤ × ℤ)

def area_of_polygon (vertices : List Point) : ℤ := 
  -- Formula to calculate polygon area would go here
  -- As a placeholder, for now we return 400 since proof details aren't required
  400

theorem polygon_area_is_400 :
  area_of_polygon [(0,0), (20,0), (30,10), (20,20), (0,20), (10,10), (0,0)] = 400 := by
  -- Proof would go here
  sorry

end polygon_area_is_400_l199_199746


namespace find_radius_of_circles_l199_199620

noncomputable theory

variable (r : ℝ)

def circle_eq (x y : ℝ) : Prop := (x - r)^2 + y^2 = r^2

def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 3 * y^2 = 12

theorem find_radius_of_circles (h_externally_tangent : ∀ x y : ℝ, circle_eq r x y → ellipse_eq x y) : 
  r = Real.sqrt (4 / 3) :=
sorry

end find_radius_of_circles_l199_199620


namespace arccos_half_eq_pi_div_three_l199_199284

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199284


namespace obtuse_triangle_l199_199866

namespace TriangleProblem

theorem obtuse_triangle (A B C : ℝ)
(sin_A sin_B : ℝ)
(a b c : ℝ)
(h1 : a > 0)
(h2 : b > 0)
(h3 : c > 0)
(h4 : a = c * sin A)
(h5 : b = c * sin B)
(h6 : A + B + C = π)
(h7 : 2 * sin A * sin B < -cos (2 * B + C)):
  a^2 + b^2 < c^2 := sorry

end TriangleProblem

end obtuse_triangle_l199_199866


namespace smallest_m_last_four_digits_l199_199018

-- Conditions
def is_divisible_by_3_and_6 (m : ℕ) : Prop :=
  m % 3 = 0 ∧ m % 6 = 0

def consists_of_only_3_and_6 (m : ℕ) : Prop :=
  ∀ c : Char, c ∈ m.digits 10 → (c = '3' ∨ c = '6')

def contains_at_least_one_3_and_one_6 (m : ℕ) : Prop :=
  ∃ l : List ℕ, m.digits 10 = l ∧ ('3'.to_digit 10 ∈ l ∧ '6'.to_digit 10 ∈ l)

-- Question translated to a Lean theorem
theorem smallest_m_last_four_digits (m : ℕ) :
  is_divisible_by_3_and_6 m ∧ consists_of_only_3_and_6 m ∧ contains_at_least_one_3_and_one_6 m →
  (m % 10000 = 3630) := sorry

end smallest_m_last_four_digits_l199_199018


namespace incorrect_statement_is_C_l199_199660

theorem incorrect_statement_is_C :
  let A := abs 81 = 81 ∧ sqrt 81 ≠ 3
  let B := (∀ x, sqrt x = x → x = 0)
  let C := (∃ x y, (irrational x ∧ irrational y ∧ ¬irrational (x + y)))
  let D := (∀ x, ∃ y, (real x ↔ point_on_number_line y))
  ¬A ∧ ¬B ∧ C ∧ ¬D := sorry

end incorrect_statement_is_C_l199_199660


namespace polynomial_solution_l199_199759

def is_polynomial_of_form (f : ℝ → ℝ) (a0 r : ℝ) (n : ℕ) : Prop :=
  f = λ x, a0 * (x^2 + r^2)^n

def condition1 (f : ℝ → ℝ) (a0 : ℝ) (an : list ℝ) (n : ℕ) : Prop :=
  f = λ x, a0 * x^(2 * n) + list.foldr (+) 0 (list.map (λ (i : ℕ × ℝ), i.snd * x^(2 * (n - i.fst))) (list.enum_from 1 an))

def condition2 (a0 : ℝ) (an : list ℝ) (n : ℕ) (c2n : ℝ) : Prop :=
  ∑ j in finset.range (n + 1), ((list.nth_le an j (by linarith)) * (list.nth_le an (n - j) (by linarith))) ≤ c2n ^ n * a0 * (list.last an (by linarith))

def condition3 (roots : list ℂ) (n : ℕ) : Prop :=
  ∀ j, j < 2 * n → ∃ β : ℝ, roots.nth_le j (by linarith) = complex.I * β

theorem polynomial_solution (f : ℝ → ℝ) (a0 r : ℝ) (n : ℕ) (an : list ℝ) (c2n : ℝ) (roots : list ℂ) :
  condition1 f a0 an n → condition2 a0 an n c2n → condition3 roots n → a0 > 0 → r > 0 →
  ∃ r, is_polynomial_of_form f a0 r n :=
sorry

end polynomial_solution_l199_199759


namespace arccos_one_half_is_pi_div_three_l199_199193

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199193


namespace arccos_one_half_l199_199219

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199219


namespace solve_x_l199_199767

noncomputable def solve_quadratic (a b k : ℝ) : (ℝ × ℝ) :=
  ( (2 * a * k + real.sqrt (4 * a^2 * k^2 - 4 * (k^2 - 1) * (a^2 - 2 * b^2))) / (2 * (k^2 - 1)),
    (2 * a * k - real.sqrt (4 * a^2 * k^2 - 4 * (k^2 - 1) * (a^2 - 2 * b^2))) / (2 * (k^2 - 1)) )

theorem solve_x (x a b k : ℝ) :
  x^2 + 2 * b^2 = (a - k * x)^2 ↔
    x = (2 * a * k + real.sqrt (4 * a^2 * k^2 - 4 * (k^2 - 1) * (a^2 - 2 * b^2))) / (2 * (k^2 - 1)) ∨
    x = (2 * a * k - real.sqrt (4 * a^2 * k^2 - 4 * (k^2 - 1) * (a^2 - 2 * b^2))) / (2 * (k^2 - 1)) :=
by
  sorry

end solve_x_l199_199767


namespace sin_double_angle_sin_multiple_angle_l199_199130

-- Prove that |sin(2x)| <= 2|sin(x)| for any value of x
theorem sin_double_angle (x : ℝ) : |Real.sin (2 * x)| ≤ 2 * |Real.sin x| := 
by sorry

-- Prove that |sin(nx)| <= n|sin(x)| for any positive integer n and any value of x
theorem sin_multiple_angle (n : ℕ) (x : ℝ) (h : 0 < n) : |Real.sin (n * x)| ≤ n * |Real.sin x| :=
by sorry

end sin_double_angle_sin_multiple_angle_l199_199130


namespace arccos_half_eq_pi_div_3_l199_199278

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199278


namespace find_m_l199_199830

theorem find_m
  (α : ℝ)
  (h1 : 4 ^ α = 2)
  (m : ℝ)
  (h2 : m ^ α = 3) :
  m = 9 :=
sorry

end find_m_l199_199830


namespace edmund_earning_l199_199753

-- Definitions based on the conditions
def daily_chores := 4
def days_in_week := 7
def weeks := 2
def normal_weekly_chores := 12
def pay_per_extra_chore := 2

-- Theorem to be proven
theorem edmund_earning :
  let total_chores := daily_chores * days_in_week * weeks
      normal_chores := normal_weekly_chores * weeks
      extra_chores := total_chores - normal_chores
      earnings := extra_chores * pay_per_extra_chore
  in earnings = 64 := 
by
  sorry

end edmund_earning_l199_199753


namespace largest_integer_not_in_set_M_one_belongs_l199_199434

open Nat

def coprime (a b : ℕ) : Prop := gcd a b = 1

def in_set_M (a b n : ℤ) : Prop :=
  ∃ x y : ℕ, n = a * x + b * y

theorem largest_integer_not_in_set_M (a b : ℕ) (hc : coprime a b) :
  ∃ c, (c = a * b - a - b) ∧ (∀ n, n ≤ c → ¬ in_set_M a b n) :=
sorry

theorem one_belongs (a b : ℕ) (hc : coprime a b) :
  ∀ n : ℤ, (in_set_M (a : ℤ) (b : ℤ) n) ↔ ¬ (in_set_M (a : ℤ) (b : ℤ) (a * b - a - b - n)) :=
sorry

end largest_integer_not_in_set_M_one_belongs_l199_199434


namespace find_A_and_B_l199_199933

theorem find_A_and_B (A : ℕ) (B : ℕ) (x y : ℕ) 
  (h1 : 1000 ≤ A ∧ A ≤ 9999) 
  (h2 : B = 10^5 * x + 10 * A + y) 
  (h3 : B = 21 * A)
  (h4 : x < 10) 
  (h5 : y < 10) : 
  A = 9091 ∧ B = 190911 :=
sorry

end find_A_and_B_l199_199933


namespace combined_alloy_force_l199_199713

-- Define the masses and forces exerted by Alloy A and Alloy B
def mass_A : ℝ := 6
def force_A : ℝ := 30
def mass_B : ℝ := 3
def force_B : ℝ := 10

-- Define the combined mass and force
def combined_mass : ℝ := mass_A + mass_B
def combined_force : ℝ := force_A + force_B

-- Theorem statement
theorem combined_alloy_force :
  combined_force = 40 :=
by
  -- The proof is omitted.
  sorry

end combined_alloy_force_l199_199713


namespace range_of_a1_l199_199829

/-- Define the sequence a_n -/
noncomputable def seq (a_n : ℕ → ℝ) (n : ℕ) : Prop :=
∀ n, a_{n+1} ≥ 2 * a_n + 1 ∧ a_n < 2^(n+1) ∧ a_n > 0

theorem range_of_a1 (a : ℕ → ℝ) (h : seq a) : 
  0 < a 1 ∧ a 1 ≤ 3 := 
sorry

end range_of_a1_l199_199829


namespace cats_and_dogs_biscuits_l199_199061

theorem cats_and_dogs_biscuits 
  (d c : ℕ) 
  (h1 : d + c = 10) 
  (h2 : 6 * d + 5 * c = 56) 
  : d = 6 ∧ c = 4 := 
by 
  sorry

end cats_and_dogs_biscuits_l199_199061


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199321

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199321


namespace A_8_coords_l199_199475

-- Define point as a structure
structure Point where
  x : Int
  y : Int

-- Initial point A
def A : Point := {x := 3, y := 2}

-- Symmetric point about the y-axis
def sym_y (p : Point) : Point := {x := -p.x, y := p.y}

-- Symmetric point about the origin
def sym_origin (p : Point) : Point := {x := -p.x, y := -p.y}

-- Symmetric point about the x-axis
def sym_x (p : Point) : Point := {x := p.x, y := -p.y}

-- Function to get the n-th symmetric point in the sequence
def sym_point (n : Nat) : Point :=
  match n % 3 with
  | 0 => A
  | 1 => sym_y A
  | 2 => sym_origin (sym_y A)
  | _ => A  -- Fallback case (should not be reachable for n >= 0)

theorem A_8_coords : sym_point 8 = {x := 3, y := -2} := sorry

end A_8_coords_l199_199475


namespace arccos_half_eq_pi_div_three_l199_199289

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199289


namespace single_elimination_matches_l199_199467

theorem single_elimination_matches (n : ℕ) (h : n = 128) : ∃ m, m = n - 1 :=
by
  use 127
  rw h
  simp
  sorry

end single_elimination_matches_l199_199467


namespace arccos_of_half_eq_pi_over_three_l199_199299

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199299


namespace minimal_moves_l199_199897

-- Define state of lamps as a list of booleans, where true represents the lamp being on and false represents the lamp being off.
def lamp_state := List Bool

-- Define the function that switches the state of the first i lamps
def switch (lst : lamp_state) (i : ℕ) : lamp_state :=
  lst.take i ++ lst.drop i (λ x => !x)

-- Prove that for n lamps, the minimal number of moves to turn all lamps on is n
theorem minimal_moves (n : ℕ) (h : 1 ≤ n) : 
  ∀ initial_state : lamp_state, initial_state.length = n → 
  ∃ k : ℕ, (k = n) ∧ (∀ final_state, all (λ x => x = true) (λ e => switch initial_state e = final_state)) :=
  sorry

end minimal_moves_l199_199897


namespace max_attempts_to_match_keys_l199_199951

theorem max_attempts_to_match_keys (n : ℕ) (h : n = 10) : 
  (finset.range (n - 1)).sum id = 45 :=
by
  sorry

end max_attempts_to_match_keys_l199_199951


namespace arccos_half_eq_pi_div_3_l199_199271

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199271


namespace greatest_multiple_of_35_less_than_1000_l199_199635

theorem greatest_multiple_of_35_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 35 = 0 ∧ (∀ k : ℕ, k < 1000 → k % 35 = 0 → k ≤ n) ∧ n = 980 :=
begin
  sorry
end

end greatest_multiple_of_35_less_than_1000_l199_199635


namespace parallel_lines_slope_l199_199453

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + y + 2a = 0) ∧ (∀ x y : ℝ, x + ay + 3 = 0) →
  a = 1 ∨ a = -1 :=
by
  sorry

end parallel_lines_slope_l199_199453


namespace totalSurfaceAreaCorrect_l199_199038

noncomputable def cubeVolumes : List ℝ := [512, 343, 216, 125, 64, 27, 8, 1, 0.125]

-- A condition that the cubes are stacked vertically with volumes decreasing from bottom to top.
def stackedDecreasingVolumes (vols : List ℝ) : Prop := 
  ∀ i j, i < j → (i < vols.length ∧ j < vols.length) → vols[i] ≥ vols[j]

-- Condition that the third cube's half bottom face is visible due to slight shift
def halfFaceVisible (vol : ℝ) : Prop := 
  vol = 216

-- Calculating total surface area including the shifting condition
def totalSurfaceArea (vols : List ℝ) : ℝ :=
  let sideArea (v : ℝ) : ℝ := (6 : ℝ^ (1 / 3))^2
  5 * sideArea vols[0] + -- bottom face visible
  5 * sideArea vols[1] + -- due to third cube shift
  4.5 * sideArea vols[2] + -- half bottom visible
  vols.drop 3 |>.init.foldl (λ acc v, acc + 4 * sideArea v) 0 +
  5 * sideArea (vols.getLast? 0)

theorem totalSurfaceAreaCorrect : 
  stackedDecreasingVolumes cubeVolumes → 
  halfFaceVisible cubeVolumes[2] → 
  totalSurfaceArea cubeVolumes = 948.25 := 
by
  intros h1 h2
  sorry

end totalSurfaceAreaCorrect_l199_199038


namespace thirty_thousand_times_thirty_thousand_l199_199180

-- Define the number thirty thousand
def thirty_thousand : ℕ := 30000

-- Define the product of thirty thousand times thirty thousand
def product_thirty_thousand : ℕ := thirty_thousand * thirty_thousand

-- State the theorem that this product equals nine hundred million
theorem thirty_thousand_times_thirty_thousand :
  product_thirty_thousand = 900000000 :=
sorry -- Proof goes here

end thirty_thousand_times_thirty_thousand_l199_199180


namespace or_false_iff_not_p_l199_199128

theorem or_false_iff_not_p (p q : Prop) : (p ∨ q → false) ↔ ¬p :=
by sorry

end or_false_iff_not_p_l199_199128


namespace largest_binomial_coeff_l199_199479

theorem largest_binomial_coeff (x y : ℤ) :
  let n := 11 in
  let term_positions := nat_pred (n / 2 + 1) :: nat_pred (n / 2 + 2) :: [] in
  (x - y)^n = (x - y)^(length term_positions)
    → term_positions = [6, 7] :=
sorry

end largest_binomial_coeff_l199_199479


namespace total_highlighters_l199_199867

theorem total_highlighters (pink yellow blue : Nat) (h1 : pink = 4) (h2 : yellow = 2) (h3 : blue = 5) :
  pink + yellow + blue = 11 :=
by
  rw [h1, h2, h3]
  norm_num

end total_highlighters_l199_199867


namespace distance_between_lines_l199_199376

noncomputable def vec := ℝ × ℝ × ℝ

def A : vec := (-3, 0, 1)
def B : vec := (2, 1, -1)
def C : vec := (-2, 2, 0)
def D : vec := (1, 3, 2)

def vector_sub (p q : vec) : vec := (p.1 - q.1, p.2 - q.2, p.3 - q.3)
def dot_product (v w : vec) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def AB : vec := vector_sub B A
def CD : vec := vector_sub D C

def normal_vector : vec := (2, -8, 1)

def plane_eq (p : vec) := 2 * p.1 - 8 * p.2 + p.3 + 5

def distance {p q : vec} (x y z : ℝ) (coords : vec) : ℝ :=
  abs (x * coords.1 + y * coords.2 + z * coords.3 + p.2 * p.3) / (real.sqrt (x^2 + y^2 + z^2))

def point_to_plane_dist : ℝ :=
  distance (2, -8, 1) 2 (-8) 1 (-2, 2, 0)

theorem distance_between_lines :
  point_to_plane_dist = 5 * real.sqrt 3 / real.sqrt 23 :=
sorry

end distance_between_lines_l199_199376


namespace sandy_change_from_twenty_dollar_bill_l199_199185

theorem sandy_change_from_twenty_dollar_bill :
  let cappuccino_cost := 2
  let iced_tea_cost := 3
  let cafe_latte_cost := 1.5
  let espresso_cost := 1
  let num_cappuccinos := 3
  let num_iced_teas := 2
  let num_cafe_lattes := 2
  let num_espressos := 2
  let total_cost := num_cappuccinos * cappuccino_cost
                  + num_iced_teas * iced_tea_cost
                  + num_cafe_lattes * cafe_latte_cost
                  + num_espressos * espresso_cost
  20 - total_cost = 3 := 
by
  sorry

end sandy_change_from_twenty_dollar_bill_l199_199185


namespace winning_position_l199_199605

def takes (n m : ℕ) : Prop :=
  (m = 1 ∨ (Nat.Prime m ∧ m ∣ n))

def win (n : ℕ) : Prop :=
  ∃ m, takes n m ∧ ¬win (n - m)

theorem winning_position (n : ℕ) : win n ↔ ¬(∃ k, n = 4 * k) :=
sorry

end winning_position_l199_199605


namespace decagon_tenth_angle_l199_199857

theorem decagon_tenth_angle (h : ∀ i : Fin 9, (angles : Fin 10 → ℝ) (h_i : i.1 < 9), angles ⟨i.val, h_i⟩ = 150) :
  (angles ⟨9, by decide⟩ = 90) :=
sorry

end decagon_tenth_angle_l199_199857


namespace probability_of_matching_pair_l199_199663
-- Import the necessary library for probability and combinatorics

def probability_matching_pair (pairs : ℕ) (total_shoes : ℕ) : ℚ :=
  if total_shoes = 2 * pairs then
    (pairs : ℚ) / ((total_shoes * (total_shoes - 1) / 2) : ℚ)
  else 0

theorem probability_of_matching_pair (pairs := 6) (total_shoes := 12) : 
  probability_matching_pair pairs total_shoes = 1 / 11 := 
by
  sorry

end probability_of_matching_pair_l199_199663


namespace scientific_notation_l199_199881

theorem scientific_notation (n : ℤ) (hn : n = 12910000) : ∃ a : ℝ, (a = 1.291 ∧ hn = (a * 10^7).to_int) :=
by
  sorry

end scientific_notation_l199_199881


namespace simplify_expression_l199_199946

variable (a b : ℤ)

theorem simplify_expression : 
  (15 * a + 45 * b) + (21 * a + 32 * b) - (12 * a + 40 * b) = 24 * a + 37 * b := 
    by sorry

end simplify_expression_l199_199946


namespace divisor_of_polynomial_l199_199769

theorem divisor_of_polynomial (a : ℤ) (h : ∀ x : ℤ, (x^2 - x + a) ∣ (x^13 + x + 180)) : a = 1 :=
sorry

end divisor_of_polynomial_l199_199769


namespace initial_extra_planks_l199_199187

-- Definitions corresponding to the conditions
def charlie_planks : Nat := 10
def father_planks : Nat := 10
def total_planks : Nat := 35

-- The proof problem statement
theorem initial_extra_planks : total_planks - (charlie_planks + father_planks) = 15 := by
  sorry

end initial_extra_planks_l199_199187


namespace members_in_both_sets_are_23_l199_199606

variable (U A B : Finset ℕ)
variable (count_U count_A count_B count_neither count_both : ℕ)

theorem members_in_both_sets_are_23 (hU : count_U = 192)
    (hA : count_A = 107) (hB : count_B = 49) (hNeither : count_neither = 59) :
    count_both = 23 :=
by
  sorry

end members_in_both_sets_are_23_l199_199606


namespace ellipse_equation_max_triangle_OPQ_l199_199407

theorem ellipse_equation (a b : ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ)
    (eccentricity : ℝ) (slope_AF : ℝ)
    (a_pos : a > 0) (b_pos : b > 0)
    (A_coords : A = (0, -2)) 
    (eccentricity_def : eccentricity = (Real.sqrt 3) / 2)
    (slope_AF_def : slope_AF = (2 * Real.sqrt 3) / 3) :
    E := ( ∀ x y, (x/a)^2 + (y/b)^2 = 1 ) → 
    ( ∃ a b, a = 2 ∧ b = 1 ) := sorry

theorem max_triangle_OPQ (l : ℝ → ℝ → Prop) 
  (E : ℝ → ℝ → Prop)
  (O : ℝ × ℝ) (A : ℝ × ℝ)
  (A_coords : A = (0, -2))
  (O_coords : O = (0, 0))
  (line_condition : ∀ x1 y1 x2 y2, l x1 y1 → l x2 y2 → (x1 ≠ x2 ∨ y1 ≠ y2))
  (ellipse_condition : ∀ x y, E x y = ((x^2 / 4) + y^2 = 1))
  (line_through_A : ∑ (k:ℝ), l = (λ x y, y = k*x - 2) ) :
  ∃ (k : ℝ), (k = Real.sqrt 7 / 2 ∨ k = -Real.sqrt 7 / 2) := sorry

end ellipse_equation_max_triangle_OPQ_l199_199407


namespace three_digit_numbers_sum_l199_199772

-- Define the digits
def digits : List ℕ := [3, 4, 5, 7, 9]

-- Define the count of three-digit numbers
def count_three_digit_numbers := 60

-- Define the total sum of three-digit numbers
def total_sum_three_digit_numbers := 37296

-- Theorem statement
theorem three_digit_numbers_sum :
  (list.permutations digits).filter (λ l, l.length = 3).length = count_three_digit_numbers ∧
  ∑ i in (list.permutations digits).filter (λ l, l.length = 3), 
    (100 * i.nth_le 0 sorry + 10 * i.nth_le 1 sorry + i.nth_le 2 sorry) = total_sum_three_digit_numbers := 
sorry

end three_digit_numbers_sum_l199_199772


namespace sum_of_divisors_252_l199_199642

open BigOperators

-- Definition of the sum of divisors for a given number n
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in Nat.divisors n, d

-- Statement of the problem
theorem sum_of_divisors_252 : sum_of_divisors 252 = 728 := 
sorry

end sum_of_divisors_252_l199_199642


namespace tens_digit_of_7_pow_2011_l199_199111

-- Define the conditions for the problem
def seven_power := 7
def exponent := 2011
def modulo := 100

-- Define the target function to find the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem formally
theorem tens_digit_of_7_pow_2011 : tens_digit (seven_power ^ exponent % modulo) = 4 := by
  sorry

end tens_digit_of_7_pow_2011_l199_199111


namespace summation_indices_equal_l199_199789

theorem summation_indices_equal
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i ≤ 100)
  (h_length : ∀ i, i < 16) :
  ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l := 
by {
  sorry
}

end summation_indices_equal_l199_199789


namespace rectangle_area_l199_199988

variable (m : ℝ)

theorem rectangle_area : 
  let width := m in
  let length := (2 * m + 1) in
  let area := width * length in
  area = (2 * m^2 + m) :=
by
  sorry

end rectangle_area_l199_199988


namespace rearrange_tokens_invariant_l199_199476

theorem rearrange_tokens_invariant (chessboard : ℕ → ℕ → ℝ × ℝ)
  (initial_config final_config : ℕ → (ℝ × ℝ))
  (h_init : ∀ i, initial_config i = chessboard i)
  (h_rearrange : ∀ i j, dist (final_config i) (final_config j) ≥ dist (initial_config i) (initial_config j)) :
  ∀ i j, dist (final_config i) (final_config j) = dist (initial_config i) (initial_config j) :=
by
  sorry

end rearrange_tokens_invariant_l199_199476


namespace probability_reach_corner_within_four_hops_l199_199387

/-- 
  Frieda the frog starts from the center of a 3x3 grid. 
  She can hop up, down, left, or right with equal probability. 
  If her hop would take her off the grid, she wraps around to the opposite edge.
  She stops if she lands on a corner square. What is the probability that she reaches a corner square within four hops?
-/
theorem probability_reach_corner_within_four_hops :
  -- Define the initial state and the transition probabilities
  let grid_size := 3
  let num_hops := 4
  let corner_probability : ℚ := 11 / 16
  ∀ (initial_state : ℕ × ℕ),
    initial_state = (2, 2) →
    (∑ p in all_possible_paths initial_state grid_size num_hops, reach_corner p grid_size) / (num_possible_paths initial_state grid_size num_hops) = corner_probability :=
sorry

end probability_reach_corner_within_four_hops_l199_199387


namespace overall_average_score_l199_199462

theorem overall_average_score (students_total : ℕ) (scores_day1 : ℕ) (avg1 : ℝ)
  (scores_day2 : ℕ) (avg2 : ℝ) (scores_day3 : ℕ) (avg3 : ℝ)
  (h1 : students_total = 45)
  (h2 : scores_day1 = 35)
  (h3 : avg1 = 0.65)
  (h4 : scores_day2 = 8)
  (h5 : avg2 = 0.75)
  (h6 : scores_day3 = 2)
  (h7 : avg3 = 0.85) :
  (scores_day1 * avg1 + scores_day2 * avg2 + scores_day3 * avg3) / students_total = 0.68 :=
by
  -- Lean proof goes here
  sorry

end overall_average_score_l199_199462


namespace birgit_time_to_travel_8km_l199_199056

theorem birgit_time_to_travel_8km
  (total_hours : ℝ)
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (distance_to_travel : ℝ)
  (total_minutes := total_hours * 60)
  (average_speed := total_minutes / total_distance)
  (birgit_speed := average_speed - speed_difference) :
  total_hours = 3.5 →
  total_distance = 21 →
  speed_difference = 4 →
  distance_to_travel = 8 →
  (birgit_speed * distance_to_travel) = 48 :=
by
  sorry

end birgit_time_to_travel_8km_l199_199056


namespace largest_consecutive_integers_in_polynomial_image_l199_199496

def consecutive_integers_in_polynomial_image (P : ℤ[X]) (hP : 1 < degree P) : ℕ :=
  let n := nat_degree P
  n

theorem largest_consecutive_integers_in_polynomial_image (P : ℤ[X]) (hP : 1 < degree P) :
  consecutive_integers_in_polynomial_image P hP = P.nat_degree :=
by
  sorry

end largest_consecutive_integers_in_polynomial_image_l199_199496


namespace arccos_half_eq_pi_div_three_l199_199315

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199315


namespace num_pairs_in_arithmetic_progression_l199_199354

theorem num_pairs_in_arithmetic_progression : 
  ∃ n : ℕ, n = 2 ∧ ∀ a b : ℝ, (a = (15 + b) / 2 ∧ (a + a * b = 2 * b)) ↔ 
  (a = (9 + 3 * Real.sqrt 7) / 2 ∧ b = -6 + 3 * Real.sqrt 7)
  ∨ (a = (9 - 3 * Real.sqrt 7) / 2 ∧ b = -6 - 3 * Real.sqrt 7) :=    
  sorry

end num_pairs_in_arithmetic_progression_l199_199354


namespace greatest_multiple_of_35_less_than_1000_l199_199634

theorem greatest_multiple_of_35_less_than_1000 : 
  ∃ n : ℕ, n < 1000 ∧ n % 35 = 0 ∧ (∀ k : ℕ, k < 1000 → k % 35 = 0 → k ≤ n) ∧ n = 980 :=
begin
  sorry
end

end greatest_multiple_of_35_less_than_1000_l199_199634


namespace crossnumber_unique_solution_l199_199600

-- Definition of two-digit numbers
def two_digit_numbers (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Definition of prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Definition of square
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The given conditions reformulated
def crossnumber_problem : Prop :=
  ∃ (one_across one_down two_down three_across : ℕ),
    two_digit_numbers one_across ∧ is_prime one_across ∧
    two_digit_numbers one_down ∧ is_square one_down ∧
    two_digit_numbers two_down ∧ is_square two_down ∧
    two_digit_numbers three_across ∧ is_square three_across ∧
    one_across = 83 ∧ one_down = 81 ∧ two_down = 16 ∧ three_across = 16

theorem crossnumber_unique_solution : crossnumber_problem :=
by
  sorry

end crossnumber_unique_solution_l199_199600


namespace arccos_one_half_l199_199231

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199231


namespace arccos_half_eq_pi_div_three_l199_199282

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199282


namespace sum_of_coeff_l199_199081

theorem sum_of_coeff {x y : ℤ} : 
  let expr := (x^3 - 3*x*y^2 + y^3)^5 in
  (x = 1 ∧ y = 1) → 
  expr = -1 := by
  sorry

end sum_of_coeff_l199_199081


namespace least_number_to_add_l199_199123

theorem least_number_to_add (n : ℕ) (h : 1056 % 29 = 12) : n = 29 - 12 → (1056 + n) % 29 = 0 :=
by
  intro hn
  have hn : n = 29 - 12 := hn
  rw [hn]
  have hmod : 1056 % 29 = 12 := h
  rw Nat.add_mod
  rw hmod
  rw Nat.mod_self
  rw add_zero
  exact Nat.mod_self 29

end least_number_to_add_l199_199123


namespace wendy_packages_chocolates_l199_199628

variable (packages_per_5min : Nat := 2)
variable (dozen_size : Nat := 12)
variable (minutes_in_hour : Nat := 60)
variable (hours : Nat := 4)

theorem wendy_packages_chocolates (h1 : packages_per_5min = 2) 
                                 (h2 : dozen_size = 12) 
                                 (h3 : minutes_in_hour = 60) 
                                 (h4 : hours = 4) : 
    let chocolates_per_5min := packages_per_5min * dozen_size
    let intervals_per_hour := minutes_in_hour / 5
    let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
    let chocolates_in_4hours := chocolates_per_hour * hours
    chocolates_in_4hours = 1152 := 
by
  let chocolates_per_5min := packages_per_5min * dozen_size
  let intervals_per_hour := minutes_in_hour / 5
  let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
  let chocolates_in_4hours := chocolates_per_hour * hours
  sorry

end wendy_packages_chocolates_l199_199628


namespace builder_windows_installed_l199_199693

theorem builder_windows_installed (total_windows : ℕ) (hours_per_window : ℕ) (total_hours_left : ℕ) :
  total_windows = 14 → hours_per_window = 4 → total_hours_left = 36 → (total_windows - total_hours_left / hours_per_window) = 5 :=
by
  intros
  sorry

end builder_windows_installed_l199_199693


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199322

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199322


namespace zoo_possibility_l199_199757

theorem zoo_possibility (A B C : Prop)
  (H1 : A ∧ B → ¬ C)
  (H2 : B ∧ ¬ C → A)
  (H3 : A ∧ C → B) :
  A ∧ ¬ B ∧ ¬ C :=
begin
  sorry
end

end zoo_possibility_l199_199757


namespace simplify_f_eval_f_l199_199397

def f (α : ℝ) : ℝ := 
  (sin (α - 3 * real.pi) * cos (2 * real.pi - α) * sin (-α + 3 * real.pi / 2)) / 
  (cos (-real.pi - α) * sin (-real.pi - α))

theorem simplify_f (α : ℝ) : f α = -cos α := 
  sorry

theorem eval_f (α : ℝ) (h : α = -31 * real.pi / 3) : f α = -1/2 := 
  by 
    rw [simplify_f, h]
    norm_num       
    rw [cos_add_pi_div_two, cos_periodic]
    -- Additional steps might be needed depending on built-in trigonometric simplifications in Lean
    sorry

end simplify_f_eval_f_l199_199397


namespace hyperbola_standard_eq_l199_199580

theorem hyperbola_standard_eq (a b : ℝ) (h1 : a^2 = 1) (h2 : b^2 = 2) (x y m : ℝ)
  (asymptote_cond : (x + sqrt 2 * y = 0) ∨ (x - sqrt 2 * y = 0))
  (passing_point : ∀ (x y : ℝ), x = -2 ∧ y = sqrt 3 → x^2 - 2 * y^2 = m)
  : m = -2 → (y^2 - (x^2 / 2) = 1) :=
by
  sorry

end hyperbola_standard_eq_l199_199580


namespace harry_terry_difference_l199_199839

theorem harry_terry_difference :
  let H := 8 - (2 + 5)
  let T := 8 - 2 + 5
  H - T = -10 :=
by 
  sorry

end harry_terry_difference_l199_199839


namespace unique_positive_integer_l199_199846

-- Define an auxiliary function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Main statement
theorem unique_positive_integer (n : ℕ) (h1 : n < 1000) (h2 : n = 8 * (sum_of_digits n)) :
  n = 72 :=
begin
  sorry
end

end unique_positive_integer_l199_199846


namespace distinct_geometric_sequences_count_l199_199843

def is_geometric_sequence (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * c = b * b)

def choices : set (ℕ × ℕ × ℕ) :=
  { (a, b, c) | a ∈ finset.range 1 21 ∧ b ∈ finset.range 1 21 ∧ c ∈ finset.range 1 21 ∧ is_geometric_sequence a b c }

theorem distinct_geometric_sequences_count : (finset.filter (λ x : ℕ × ℕ × ℕ, x ∈ choices) (finset.univ : finset (ℕ × ℕ × ℕ))).card = 22 :=
by
  sorry

end distinct_geometric_sequences_count_l199_199843


namespace arccos_half_is_pi_div_three_l199_199263

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199263


namespace replace_asterisk_with_2x_l199_199533

theorem replace_asterisk_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by sorry

end replace_asterisk_with_2x_l199_199533


namespace max_chocolates_l199_199118

theorem max_chocolates (initial_money price_per_chocolate wrappers_needed : ℕ) (h_price : price_per_chocolate = 1) (h_wrappers : wrappers_needed = 3) (h_money : initial_money = 15) :
  let initial_chocolates := initial_money / price_per_chocolate,
      additional_chocolates := initial_chocolates / wrappers_needed,
      total_chocolates := initial_chocolates + additional_chocolates + (additional_chocolates / wrappers_needed) + ((additional_chocolates % wrappers_needed + additional_chocolates / wrappers_needed) / wrappers_needed)
  in total_chocolates = 22 :=
sorry

end max_chocolates_l199_199118


namespace GM_food_uncertainty_l199_199888

theorem GM_food_uncertainty (options : list string) (question : string) (answer : string) : 
  question = "It is uncertain ___ side effect the GM food will bring about, although some scientists consider it to be safe." 
  ∧ options = ["that", "what", "how", "whether"]
  ∧ answer = "what"
  → option.get_or_else (list.nth options 1) "" = answer :=
by
  intros h
  cases h with q_rest h_rest
  cases h_rest with o_rest ans_eq
  simp [o_rest, ans_eq]
  sorry

end GM_food_uncertainty_l199_199888


namespace smallest_omega_correct_l199_199094

noncomputable def smallest_omega : ℝ :=
  if ω > 0 ∧ ∀ x : ℝ, ∃ k : ℤ, 
    2 * sin (ω * (x + π/4) - π/4) = 2 * sin (ω * x + (ω - 1) * π / 4) ∧ 
    2 * sin (ω * (x - π/4) - π/4) = 2 * sin (ω * x - (ω + 1) * π / 4) ∧
    (ω * x + (ω - 1) * π / 4 = ω * x - (ω + 1) * π / 4 ∨ 
     ω * x + (ω - 1) * π / 4 = ω * x - (ω + 1) * π / 4 + k * π) then 2 else 0

-- Statement for Lean 4 theorem
theorem smallest_omega_correct : 
  ∃ ω : ℝ, ω > 0 ∧ ∀ x : ℝ, ∃ k : ℤ, 
    (2 * sin (ω * (x + π/4) - π/4) = 2 * sin (ω * x + (ω - 1) * π / 4)) ∧
    (2 * sin (ω * (x - π/4) - π/4) = 2 * sin (ω * x - (ω + 1) * π / 4)) ∧
    (ω * x + (ω - 1) * π/4 = ω * x - (ω + 1) * π/4 ∨ 
     ω * x + (ω - 1) * π/4 = ω * x - (ω + 1) * π/4 + k * π) ∧
    ω = 2 :=
sorry

end smallest_omega_correct_l199_199094


namespace greatest_divisor_lemma_l199_199669

theorem greatest_divisor_lemma : ∃ (d : ℕ), d = Nat.gcd 1636 1852 ∧ d = 4 := by
  sorry

end greatest_divisor_lemma_l199_199669


namespace graph_shifted_l199_199993

variable (f : ℝ → ℝ) (x : ℝ)

theorem graph_shifted :
  ∀ (x : ℝ), f(x - 2) + 1 = (f(x) + 1) + 1 :=
sorry

end graph_shifted_l199_199993


namespace arccos_half_is_pi_div_three_l199_199264

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199264


namespace max_true_statements_l199_199915

theorem max_true_statements (x y : ℝ) :
  let s1 := (1 / x > 1 / y)
  let s2 := (x^2 < y^2)
  let s3 := (x > y)
  let s4 := (x > 0)
  let s5 := (y > 0)
  bool := (s1.to_bool + s2.to_bool + s3.to_bool + s4.to_bool + s5.to_bool ≤ 3 : ℕ) :=
sorry

end max_true_statements_l199_199915


namespace find_omega_find_range_of_f_l199_199823

-- Definitions and conditions from the problem statement.
def f (ω : ℝ) (x : ℝ) : ℝ := (Real.sqrt 3) * (Real.sin (ω * x)) * (Real.cos (ω * x)) - (Real.cos (ω * x))^2

def period_condition (ω : ℝ) : Prop := (0 < ω) ∧ (∀ x, f ω (x + (π / 2)) = f ω x)

def triangle_condition (a b c : ℝ) : Prop := (b^2 = a * c)

def angle_condition (a b c x : ℝ) : Prop := ∃ x, triangle_condition a b c ∧ (b^2 = a * c)

-- First proof problem: determine ω given period condition
theorem find_omega (ω : ℝ) (hω: period_condition ω) : ω = 2 :=
sorry

-- Second proof problem: determine range of function given triangle conditions
theorem find_range_of_f (a b c x : ℝ) (ω : ℝ) (hω: ω = 2) (hx : angle_condition a b c x) : 
  set.range (λ x, f ω x) = set.Icc (-1 : ℝ) (1 / 2 : ℝ) :=
sorry

end find_omega_find_range_of_f_l199_199823


namespace magnitude_vector_difference_l199_199437

open Real

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 3)
variables (angle_ab : innerProductSpace.orthonormal θ 60)

theorem magnitude_vector_difference : ∥2 • a - b∥ = sqrt 13 := by
  sorry

end magnitude_vector_difference_l199_199437


namespace bicycle_distance_travel_l199_199679

def tire_wear_front (d : ℕ) : Prop :=
  d ≤ 5000

def tire_wear_rear (d : ℕ) : Prop :=
  d ≤ 3000

noncomputable def gcd (a b : ℕ) : ℕ := a.gcd b
noncomputable def lcm (a b : ℕ) : ℕ := a.lcm b

theorem bicycle_distance_travel (d : ℕ)
  (h1 : ∀ d < 5000, tire_wear_front d)
  (h2 : ∀ d < 3000, tire_wear_rear d)
  (swap_distance : ℕ) :
  (lcm 5000 3000) / (gcd 5000 3000) = 3750 := 
sorry

end bicycle_distance_travel_l199_199679


namespace arccos_half_eq_pi_over_three_l199_199211

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199211


namespace bill_earnings_l199_199727

theorem bill_earnings
  (milk_total : ℕ)
  (fraction : ℚ)
  (milk_to_butter_ratio : ℕ)
  (milk_to_sour_cream_ratio : ℕ)
  (butter_price_per_gallon : ℚ)
  (sour_cream_price_per_gallon : ℚ)
  (whole_milk_price_per_gallon : ℚ)
  (milk_for_butter : ℚ)
  (milk_for_sour_cream : ℚ)
  (remaining_milk : ℚ)
  (total_earnings : ℚ) :
  milk_total = 16 →
  fraction = 1/4 →
  milk_to_butter_ratio = 4 →
  milk_to_sour_cream_ratio = 2 →
  butter_price_per_gallon = 5 →
  sour_cream_price_per_gallon = 6 →
  whole_milk_price_per_gallon = 3 →
  milk_for_butter = milk_total * fraction / milk_to_butter_ratio →
  milk_for_sour_cream = milk_total * fraction / milk_to_sour_cream_ratio →
  remaining_milk = milk_total - 2 * (milk_total * fraction) →
  total_earnings = (remaining_milk * whole_milk_price_per_gallon) + 
                   (milk_for_sour_cream * sour_cream_price_per_gallon) + 
                   (milk_for_butter * butter_price_per_gallon) →
  total_earnings = 41 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end bill_earnings_l199_199727


namespace group_distribution_l199_199473

theorem group_distribution (m w : ℕ) (h_m : m = 4) (h_w : w = 5) :
  (∀A B C : Finset (Fin m ⊕ Fin w), 
    A.card = 2 ∧ B.card = 2 ∧ C.card = 3 ∧ 
    (∃ a1 a2 : Fin m ⊕ Fin w, a1 ∈ A ∧ a2 ∈ A ∧ ∃ b1 b2 : Fin m ⊕ Fin w, b1 ∈ B ∧ b2 ∈ B ∧ ∃ c1 c2 c3 : Fin m ⊕ Fin w, c1 ∈ C ∧ c2 ∈ C ∧ c3 ∈ C) ∧ 
    (∃ man_in_group : (Fin m ⊕ Fin w) → Prop, (man_in_group A ∧ man_in_group B ∧ man_in_group C) ∧ ¬ ∃ two_groups, two_groups ∈ A ∧ two_groups ∈ B ∨ two_groups ∈ A ∧ two_groups ∈ C ∨ two_groups ∈ B ∧ two_groups ∈ C)) := 600

end group_distribution_l199_199473


namespace cubs_cardinals_home_run_differential_is_3_l199_199480

-- Define the conditions of the game
def weather_conditions := "Wind blowing at 10 mph towards left field"
def cubs_starting_pitcher_stats := (105 : ℕ, 7 : ℕ, 2 : ℕ, 82 : ℕ) -- (total pitches, strikeouts, walks, pitch count in 7th inning)
def cardinals_starting_pitcher_stats := (112 : ℕ, 6 : ℕ, 3 : ℕ) -- (total pitches, strikeouts, walks)

def cubs_home_runs_3rd_inning := 2
def cubs_home_runs_5th_inning := 1
def cubs_home_runs_8th_inning := 2
def total_cubs_home_runs := cubs_home_runs_3rd_inning + cubs_home_runs_5th_inning + cubs_home_runs_8th_inning

def cardinals_home_run_2nd_inning := 1
def cardinals_home_run_5th_inning := 1
def total_cardinals_home_runs := cardinals_home_run_2nd_inning + cardinals_home_run_5th_inning

def home_run_differential := total_cubs_home_runs - total_cardinals_home_runs

-- The goal to prove that the home run differential is 3
theorem cubs_cardinals_home_run_differential_is_3 : home_run_differential = 3 := 
by
  sorry

end cubs_cardinals_home_run_differential_is_3_l199_199480


namespace correct_average_of_ten_numbers_l199_199575

theorem correct_average_of_ten_numbers:
  ∀ (nums : List ℕ),
  (List.length nums = 10) →
  ((nums.sum).natAbs = 10 * 16 + 55 - 25) →
  ((nums.sum + 55 - 25) / 10 = 19) :=
by sorry

end correct_average_of_ten_numbers_l199_199575


namespace cone_volume_from_truncated_cone_l199_199977

-- Define the given conditions
variables (a α : ℝ)

-- Mathematically equivalent proof problem statement
theorem cone_volume_from_truncated_cone (a α : ℝ) : 
  let R := (1 / 2) * a * Real.sin α in
  ∃ V, 
    V = (∏ * a ^ 3 / 12 * (Real.sin α) ^ 5 * (Real.cos (α / 2)) ^ 2) := 
sorry

end cone_volume_from_truncated_cone_l199_199977


namespace part1_common_root_part2_arithmetic_sequence_l199_199797

-- Definitions and conditions for part 1
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Part 1: Prove the common root
theorem part1_common_root (a : ℕ → ℝ) (d : ℝ) (h : is_arithmetic_sequence a d) (hd : d ≠ 0) (ha_nonzero : ∀ n : ℕ, a n ≠ 0) (k : ℕ) :
  ∃ x : ℝ, ∀ k : ℕ, a k * x^2 + 2 * a (k+1) * x + a (k+2) = 0 :=
begin
  use -1,
  intro k,
  have h1 : a (k+1) = a k + d := h k,
  have h2 : a (k+2) = a k + 2 * d := by simp [h, h1],
  linarith,
  sorry -- Prove that x = -1 is a root using the arithmetic sequence properties
end

-- Definitions and conditions for part 2
theorem part2_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (h : is_arithmetic_sequence a d) (hd : d ≠ 0) (ha_nonzero : ∀ n : ℕ, a n ≠ 0) :
  ∃ c : ℝ, ∀ k l : ℕ, (1 / (-a (k+2) / a k + 1)) - (1 / (-a (l+2) / a l + 1)) = c * (k - l) :=
begin
  use -1/2,
  intros k l,
  sorry -- Prove the arithmetic property by using the given conditions
end

end part1_common_root_part2_arithmetic_sequence_l199_199797


namespace replace_star_with_2x_l199_199556

theorem replace_star_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by
  sorry

end replace_star_with_2x_l199_199556


namespace find_magnitude_of_sum_l199_199799

-- Definitions
variables {x y : ℝ}

def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b (y : ℝ) : ℝ × ℝ := (1, y)
def vec_c : ℝ × ℝ := (2, -4)

-- Conditions
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def is_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Problem Statement
theorem find_magnitude_of_sum : 
  is_perpendicular (vec_a x) vec_c →
  is_parallel (vec_b y) vec_c →
  ∃ x y, (x = 2) → (y = -2) → 
  ∥(vec_a x).1 + (vec_b y).1, (vec_a x).2 + (vec_b y).2∥ = √10 :=
by sorry

end find_magnitude_of_sum_l199_199799


namespace simplify_and_rationalize_l199_199560

theorem simplify_and_rationalize :
  ( √3 / √4 ) * ( √5 / √6 ) * ( √7 / √8 ) = √70 / 16 :=
by
  sorry

end simplify_and_rationalize_l199_199560


namespace arccos_half_eq_pi_div_three_l199_199287

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199287


namespace ellipse_eccentricity_l199_199816

-- Definitions for the problem
variables {a b c : ℝ} -- a, b, and c are real numbers
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions of the problem
def conditions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) : Prop :=
  let c := sqrt (a^2 - b^2) in -- The relationship between a, b, and c
  let M := (-a, 0) in
  let N := (0, b) in
  let F := (c, 0) in
  let NM := (-a, b) in -- Vector from N to M
  let NF := (c, b) in -- Vector from N to F
  (NM.1 * NF.1 + NM.2 * NF.2 = 0) -- Dot product condition

-- The main proof statement
theorem ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h : conditions a b ha hb hab) : 
  let c := sqrt (a^2 - b^2) in -- The relationship between a, b, and c
  let e := c / a in
  e = (sqrt 5 - 1) / 2 :=
sorry

end ellipse_eccentricity_l199_199816


namespace bob_shuck_2_hours_l199_199176

def shucking_rate : ℕ := 10  -- oysters per 5 minutes
def minutes_per_hour : ℕ := 60
def hours : ℕ := 2
def minutes : ℕ := hours * minutes_per_hour
def interval : ℕ := 5  -- minutes per interval
def intervals : ℕ := minutes / interval
def num_oysters (intervals : ℕ) : ℕ := intervals * shucking_rate

theorem bob_shuck_2_hours : num_oysters intervals = 240 := by
  -- leave the proof as an exercise
  sorry

end bob_shuck_2_hours_l199_199176


namespace olivia_total_earnings_l199_199524

variable (rate : ℕ) (hours_monday : ℕ) (hours_wednesday : ℕ) (hours_friday : ℕ)

def olivia_earnings : ℕ := hours_monday * rate + hours_wednesday * rate + hours_friday * rate

theorem olivia_total_earnings :
  rate = 9 → hours_monday = 4 → hours_wednesday = 3 → hours_friday = 6 → olivia_earnings rate hours_monday hours_wednesday hours_friday = 117 :=
by
  sorry

end olivia_total_earnings_l199_199524


namespace first_three_digits_of_quotient_are_239_l199_199939

noncomputable def a : ℝ := 0.12345678910114748495051
noncomputable def b_lower_bound : ℝ := 0.515
noncomputable def b_upper_bound : ℝ := 0.516

theorem first_three_digits_of_quotient_are_239 (b : ℝ) (hb : b_lower_bound < b ∧ b < b_upper_bound) :
    0.239 * b < a ∧ a < 0.24 * b := 
sorry

end first_three_digits_of_quotient_are_239_l199_199939


namespace abs_add_conditions_l199_199450

theorem abs_add_conditions (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  a + b = 1 ∨ a + b = 7 :=
by
  sorry

end abs_add_conditions_l199_199450


namespace problem_statement_l199_199410

noncomputable def tan_plus_alpha_half_pi (α : ℝ) : ℝ := -1 / (Real.tan α)

theorem problem_statement (α : ℝ) (h : tan_plus_alpha_half_pi α = -1 / 2) :
  (2 * Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -5 := by
  sorry

end problem_statement_l199_199410


namespace arccos_half_eq_pi_div_three_l199_199310

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199310


namespace arccos_half_eq_pi_div_three_l199_199291

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199291


namespace arccos_half_eq_pi_div_three_l199_199250

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199250


namespace monotonicity_of_f_extremum_of_f_on_interval_l199_199821

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem monotonicity_of_f : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → 1 ≤ x₂ → f x₁ < f x₂ := by
  sorry

theorem extremum_of_f_on_interval : 
  f 1 = 3 / 2 ∧ f 4 = 9 / 5 := by
  sorry

end monotonicity_of_f_extremum_of_f_on_interval_l199_199821


namespace B_coordinates_when_A_is_origin_l199_199927

-- Definitions based on the conditions
def A_coordinates_when_B_is_origin := (2, 5)

-- Theorem to prove the coordinates of B when A is the origin
theorem B_coordinates_when_A_is_origin (x y : ℤ) :
    A_coordinates_when_B_is_origin = (2, 5) →
    (x, y) = (-2, -5) :=
by
  intro h
  -- skipping the proof steps
  sorry

end B_coordinates_when_A_is_origin_l199_199927


namespace frank_initial_boxes_l199_199771

theorem frank_initial_boxes (filled left : ℕ) (h_filled : filled = 8) (h_left : left = 5) : 
  filled + left = 13 := by
  sorry

end frank_initial_boxes_l199_199771


namespace simplify_expression_l199_199179

theorem simplify_expression : 
  ((Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2)) - 
  (Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2 / 3))) = -Real.sqrt 2 :=
by
  sorry

end simplify_expression_l199_199179


namespace train_crossing_time_l199_199484

-- Define the given conditions
def train_length : ℝ := 320    -- in meters
def train_speed_kmh : ℝ := 144 -- in kilometers per hour
def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600 -- converted to meters per second

-- Define the time it takes to cross the pole
def crossing_time : ℝ := train_length / train_speed_ms

-- Statement to prove
theorem train_crossing_time :
  crossing_time = 8 :=
by
  sorry

end train_crossing_time_l199_199484


namespace distance_between_lines_l199_199375

noncomputable def vec := ℝ × ℝ × ℝ

def A : vec := (-3, 0, 1)
def B : vec := (2, 1, -1)
def C : vec := (-2, 2, 0)
def D : vec := (1, 3, 2)

def vector_sub (p q : vec) : vec := (p.1 - q.1, p.2 - q.2, p.3 - q.3)
def dot_product (v w : vec) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def AB : vec := vector_sub B A
def CD : vec := vector_sub D C

def normal_vector : vec := (2, -8, 1)

def plane_eq (p : vec) := 2 * p.1 - 8 * p.2 + p.3 + 5

def distance {p q : vec} (x y z : ℝ) (coords : vec) : ℝ :=
  abs (x * coords.1 + y * coords.2 + z * coords.3 + p.2 * p.3) / (real.sqrt (x^2 + y^2 + z^2))

def point_to_plane_dist : ℝ :=
  distance (2, -8, 1) 2 (-8) 1 (-2, 2, 0)

theorem distance_between_lines :
  point_to_plane_dist = 5 * real.sqrt 3 / real.sqrt 23 :=
sorry

end distance_between_lines_l199_199375


namespace sum_of_perimeters_l199_199120

-- Definitions and Conditions
def perimeters_sum (S1 : ℝ) : ℝ :=
  let P1 := 3 * S1
  let r := 1 / 2
  P1 / (1 - r)

-- Theorem statement
theorem sum_of_perimeters (S1 : ℝ): perimeters_sum S1 = 180 :=
by
  unfold perimeters_sum
  have h1 : 3 * S1 = 90 := sorry
  have h2 : 1 - (1 / 2) = 1 / 2 := sorry
  rw [h1, h2]
  have h3 : 90 / (1 / 2) = 180 := sorry
  exact h3

end sum_of_perimeters_l199_199120


namespace scientific_notation_correct_l199_199880

def n : ℝ := 12910000

theorem scientific_notation_correct : n = 1.291 * 10^7 := 
by
  sorry

end scientific_notation_correct_l199_199880


namespace adam_apples_on_monday_l199_199166

def apples_monday (x : ℕ) (tuesday : ℕ) (wednesday : ℕ) (total : ℕ) : Prop :=
  tuesday = 3 * x ∧ wednesday = 4 * tuesday ∧ total = x + tuesday + wednesday

theorem adam_apples_on_monday : ∃ x : ℕ, apples_monday x (3 * x) (12 * x) 240 ∧ x = 15 :=
by
  exists 15
  dsimp [apples_monday]
  split
  . rfl
  split
  . rfl
  . norm_num

end adam_apples_on_monday_l199_199166


namespace Sandy_has_11_nickels_Sandy_has_0_point_77_euros_l199_199945

theorem Sandy_has_11_nickels (initial_nickels borrowed_nickels : ℕ) (h1: initial_nickels = 31) (h2: borrowed_nickels = 20) :
  initial_nickels - borrowed_nickels = 11 := by sorry

theorem Sandy_has_0_point_77_euros (pennies nickels borrowed_nickels : ℕ) (initial_nickels initial_pennies : ℕ)
  (nickel_value penny_value cents_to_dollars exchange_rate to_euros : ℚ)
  (h1: pennies = 36) (h2: nickels = 31) (h3: initial_pennies = pennies) (h4: initial_nickels = nickels)
  (h5: borrowed_nickels = 20) (h6: nickel_value = 5) (h7: penny_value = 1)
  (h8: cents_to_dollars = 100) (h9: exchange_rate = 1.18)
  (value_in_dollars : ℚ := (11 * nickel_value + 36 * penny_value) / cents_to_dollars)
  (value_in_euros : ℚ := value_in_dollars / exchange_rate) :
  value_in_euros ≈ 0.77 := by sorry

end Sandy_has_11_nickels_Sandy_has_0_point_77_euros_l199_199945


namespace least_plates_to_ensure_matching_pair_l199_199671

theorem least_plates_to_ensure_matching_pair
  (white_plates : ℕ)
  (green_plates : ℕ)
  (red_plates : ℕ)
  (pink_plates : ℕ)
  (purple_plates : ℕ)
  (h_white : white_plates = 2)
  (h_green : green_plates = 6)
  (h_red : red_plates = 8)
  (h_pink : pink_plates = 4)
  (h_purple : purple_plates = 10) :
  ∃ n, n = 6 :=
by
  sorry

end least_plates_to_ensure_matching_pair_l199_199671


namespace inequality_proof_l199_199048

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l199_199048


namespace bob_shuck_2_hours_l199_199177

def shucking_rate : ℕ := 10  -- oysters per 5 minutes
def minutes_per_hour : ℕ := 60
def hours : ℕ := 2
def minutes : ℕ := hours * minutes_per_hour
def interval : ℕ := 5  -- minutes per interval
def intervals : ℕ := minutes / interval
def num_oysters (intervals : ℕ) : ℕ := intervals * shucking_rate

theorem bob_shuck_2_hours : num_oysters intervals = 240 := by
  -- leave the proof as an exercise
  sorry

end bob_shuck_2_hours_l199_199177


namespace median_relation_l199_199012

variable {ξ : Type*} [Discrete ξ] [Measurable ξ] [ProbabilitySpace ξ]
variable {P : set ξ → ℝ} [P_is_measure : measure_space P]

noncomputable def M_a : set ℝ :=
{μ | max (P {x | x > μ}) (P {x | x < μ}) ≤ 1 / 2}

noncomputable def M_b : set ℝ :=
{μ | P {x | x < μ} ≤ 1 / 2 ∧ 1 / 2 ≤ P {x | x ≤ μ}}

noncomputable def M_c : set ℝ :=
{μ | μ = Inf {x | 1 / 2 ≤ P {x | x ≤ μ}}}

theorem median_relation : M_c ⊆ M_a ∧ M_a = M_b :=
begin
  sorry
end

end median_relation_l199_199012


namespace extreme_points_and_inequality_l199_199820

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1/2) * x^2 - a * x + real.log x

theorem extreme_points_and_inequality (a : ℝ) (m n : ℝ) (k : ℤ) 
  (h1 : a > 2)
  (h2 : m = (a - real.sqrt (a^2 - 4)) / 2)
  (h3 : n = (a + real.sqrt (a^2 - 4)) / 2)
  (h4 : m > real.sqrt 2 / 2) :
  0 < f n a + k ∧ f n a + k < f m a ∧ f m a < f n a + 3 * k + 5 * real.log 2 :=
sorry

end extreme_points_and_inequality_l199_199820


namespace determine_kopeck_coin_hand_l199_199527

theorem determine_kopeck_coin_hand (s : ℕ) (hl hr : ℕ) (cl cr : ℕ) :
  (cl = 10 ∨ cl = 15) ∧ (cr = 10 ∨ cr = 15) ∧
  (hl = 4 ∨ hl = 10 ∨ hl = 12 ∨ hl = 26) ∧
  (hr = 7 ∨ hr = 13 ∨ hr = 21 ∨ hr = 35) ∧
  s = cl * hl + cr * hr →
  ∃ (left_is_10 : bool), (if left_is_10 then cl = 10 else cr = 10) :=
by 
  sorry

end determine_kopeck_coin_hand_l199_199527


namespace sum_reciprocals_equals_p_l199_199914

open Complex Polynomial

noncomputable def sum_of_reciprocals_of_roots {p q r s : ℝ} 
  (h_roots : ∀ z : ℂ, z ∈ (roots (X ^ 4 + C p * X ^ 3 + C q * X ^ 2 + C r * X + C s))
    → abs z = 1) : ℝ := sorry

theorem sum_reciprocals_equals_p {p q r s : ℝ} 
  (h_roots : ∀ z : ℂ, z ∈ (roots (X ^ 4 + C p * X ^ 3 + C q * X ^ 2 + C r * X + C s))
    → abs z = 1) : sum_of_reciprocals_of_roots h_roots = p :=
sorry

end sum_reciprocals_equals_p_l199_199914


namespace min_value_of_function_l199_199073

theorem min_value_of_function (x : ℝ) (h : x > 3) :
  let y := (1 / (x - 3)) + x in y >= 5 ∧ (y = 5 → x = 4) :=
by
  sorry

end min_value_of_function_l199_199073


namespace wicket_keeper_older_by_3_l199_199577

theorem wicket_keeper_older_by_3 (captain_age team_avg remaining_avg total_members : ℕ)
  (h1 : captain_age = 26)
  (h2 : team_avg = 23)
  (h3 : remaining_avg = 22)
  (h4 : total_members = 11) :
  let team_total_age := team_avg * total_members,
      remaining_players := total_members - 2,
      remaining_total_age := remaining_avg * remaining_players,
      combined_age := team_total_age - remaining_total_age,
      wicket_keeper_age := combined_age - captain_age
  in wicket_keeper_age - captain_age = 3 := 
by 
  sorry

end wicket_keeper_older_by_3_l199_199577


namespace arccos_one_half_is_pi_div_three_l199_199196

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199196


namespace arccos_one_half_l199_199232

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199232


namespace sum_digits_even_l199_199900

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_17_digit_number (n : ℕ) : Prop := n >= 10^16 ∧ n < 10^17

theorem sum_digits_even 
  (M N : ℕ)
  (hM : is_17_digit_number M)
  (hN : N = (reverse_digits M)) :
  ∃ d, d ∈ digits (M + N) ∧ is_even d :=
begin
  sorry
end

end sum_digits_even_l199_199900


namespace find_f_11_11_l199_199396

noncomputable def f : ℕ+ → ℕ+ → ℕ+
| ⟨1, h1⟩, ⟨1, h2⟩ := 1
| ⟨m+1, h1⟩, ⟨n+2, h2⟩ := f ⟨m+1, h1⟩ ⟨n+1, h2⟩ + 2
| ⟨m+2, h1⟩, ⟨1, h2⟩ := 2 * f ⟨m+1, h1⟩ ⟨1, h2⟩
| ⟨_, _⟩, ⟨_, _⟩ := sorry

theorem find_f_11_11 :
  f ⟨11, by norm_num⟩ ⟨11, by norm_num⟩ = 1044 := sorry

end find_f_11_11_l199_199396


namespace arccos_half_eq_pi_over_three_l199_199215

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199215


namespace triangle_identity_l199_199530

noncomputable def altitude (CH : ℝ) : ℝ := CH
noncomputable def circumradius (R : ℝ) : ℝ := R

theorem triangle_identity (a b c CH R : ℝ)
  (h1 : CH = c * Real.cot (Real.acos ((a^2 + b^2 - c^2) / (2 * a * b))))
  (h2 : R = c / (2 * Real.sin (Real.acos ((a^2 + b^2 - c^2) / (2 * a * b)))))
  : (a^2 + b^2 - c^2) / (a * b) = CH / R := 
by 
  sorry

end triangle_identity_l199_199530


namespace sandy_change_from_twenty_dollar_bill_l199_199186

theorem sandy_change_from_twenty_dollar_bill :
  let cappuccino_cost := 2
  let iced_tea_cost := 3
  let cafe_latte_cost := 1.5
  let espresso_cost := 1
  let num_cappuccinos := 3
  let num_iced_teas := 2
  let num_cafe_lattes := 2
  let num_espressos := 2
  let total_cost := num_cappuccinos * cappuccino_cost
                  + num_iced_teas * iced_tea_cost
                  + num_cafe_lattes * cafe_latte_cost
                  + num_espressos * espresso_cost
  20 - total_cost = 3 := 
by
  sorry

end sandy_change_from_twenty_dollar_bill_l199_199186


namespace equation_has_one_negative_and_one_zero_root_l199_199943

theorem equation_has_one_negative_and_one_zero_root :
  ∃ x y : ℝ, x < 0 ∧ y = 0 ∧ 3^x + x^2 + 2 * x - 1 = 0 ∧ 3^y + y^2 + 2 * y - 1 = 0 :=
sorry

end equation_has_one_negative_and_one_zero_root_l199_199943


namespace second_machine_time_l199_199690

/-- Given:
1. A first machine can address 600 envelopes in 10 minutes.
2. Both machines together can address 600 envelopes in 4 minutes.
We aim to prove that the second machine alone would take 20/3 minutes to address 600 envelopes. -/
theorem second_machine_time (x : ℝ) 
  (first_machine_rate : ℝ := 600 / 10)
  (combined_rate_needed : ℝ := 600 / 4)
  (second_machine_rate : ℝ := combined_rate_needed - first_machine_rate) 
  (secs_envelope_rate : ℝ := second_machine_rate) 
  (envelopes : ℝ := 600) : 
  x = envelopes / secs_envelope_rate :=
sorry

end second_machine_time_l199_199690


namespace probability_heads_penny_dime_quarter_l199_199569

theorem probability_heads_penny_dime_quarter :
  let total_outcomes := 2^5
  let favorable_outcomes := 2 * 2
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 8 :=
by
  simp [total_outcomes, favorable_outcomes]
  sorry

end probability_heads_penny_dime_quarter_l199_199569


namespace inequality_solution_l199_199371

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := 
by 
  sorry

end inequality_solution_l199_199371


namespace subset_with_properties_l199_199898

theorem subset_with_properties (n : ℕ) (h : 0 < n) :
  ∃ S : set ℕ, S ⊆ {i | i ≤ 2^n} ∧ 
  S.card ≥ 2^(n-1) + n ∧ 
  (∀ x y ∈ S, x ≠ y → ¬(x + y) ∣ (x * y)) :=
sorry

end subset_with_properties_l199_199898


namespace smallest_M_inequality_l199_199743

theorem smallest_M_inequality (a b c : ℝ) :
  ∃ M : ℝ, ( ∀ a b c : ℝ, abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
                      ≤ M * (a^2 + b^2 + c^2)^2 ) ∧
           (∀ N : ℝ, (∀ a b c : ℝ, abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
                      ≤ N * (a^2 + b^2 + c^2)^2 ) → M ≤ N) ∧
           (M = 9 * real.sqrt 2 / 32) :=
sorry

end smallest_M_inequality_l199_199743


namespace correct_propositions_l199_199170

-- Proposition ①: f(x) = (1/3)^x is decreasing over (-∞, +∞)
def proposition_1 : Prop :=
  ∀ x y : ℝ, x < y → (1/3)^x > (1/3)^y

-- Proposition ②: The domain of the function f(x) = sqrt(x-1) is (1, +∞)
def proposition_2 : Prop :=
  ∃ D, D = { x : ℝ | x > 1 }

-- Proposition ③: Given the mapping f(x, y) = (x + y, x - y), the image of (3, 1) under this mapping is (4, 2)
def mapping : ℝ × ℝ → ℝ × ℝ := λ p, (p.1 + p.2, p.1 - p.2)
def proposition_3 : Prop := mapping (3, 1) = (4, 2)

-- Combining the propositions and proving the correct ones are ① and ③
theorem correct_propositions : proposition_1 ∧ ¬proposition_2 ∧ proposition_3 :=
by
  sorry

end correct_propositions_l199_199170


namespace four_lines_create_quadrilateral_l199_199046

theorem four_lines_create_quadrilateral : ∃ (regions : set (set (ℝ × ℝ))), 
  (4.lines_partition_the_plane regions) ∧ (∃ quad ∈ regions, quadrilateral quad) :=
sorry

end four_lines_create_quadrilateral_l199_199046


namespace measure_of_angle_B_l199_199885

-- Define the conditions and the goal as a theorem
theorem measure_of_angle_B (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : A = 3 * B)
  (triangle_angle_sum : A + B + C = 180) : B = 30 :=
by
  -- Substitute the conditions into Lean to express and prove the statement
  sorry

end measure_of_angle_B_l199_199885


namespace arccos_half_eq_pi_div_three_l199_199293

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199293


namespace car_stopping_distance_max_l199_199583

theorem car_stopping_distance_max :
  ∀ t : ℝ, ∃ t₀ : ℝ, (s : ℝ) = 30 * t - 5 * t^2 ∧ (∀ t', s t ≤ s t₀) ∧ s t₀ = 45 := by
  sorry

end car_stopping_distance_max_l199_199583


namespace younger_person_age_l199_199952

theorem younger_person_age 
  (y e : ℕ)
  (h1 : e = y + 20)
  (h2 : e - 4 = 5 * (y - 4)) : 
  y = 9 := 
sorry

end younger_person_age_l199_199952


namespace arccos_one_half_is_pi_div_three_l199_199201

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199201


namespace find_extrema_f_l199_199424

noncomputable def f (x : ℝ) : ℝ := 3 / (x + 1)

theorem find_extrema_f : 
  (is_max_on f (set.Icc 0 5) 0 ∧ f 0 = 3) ∧ 
  (is_min_on f (set.Icc 0 5) 5 ∧ f 5 = 1/2) :=
by
  sorry

end find_extrema_f_l199_199424


namespace arccos_one_half_l199_199220

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199220


namespace smallest_product_largest_product_l199_199097

def is_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

def uses_unique_digits_1_to_9 (a b c : ℕ) : Prop :=
  let digits := ((toDigits 10 a) ++ (toDigits 10 b) ++ (toDigits 10 c))
  digits.length = 9 ∧ ∀ d, d ∈ digits → d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem smallest_product : ∀ a b c : ℕ, 
  is_three_digit_number a → is_three_digit_number b → is_three_digit_number c → 
  uses_unique_digits_1_to_9 a b c → 
  a * b * c ≥ 147 * 258 * 369 :=
by
  sorry

theorem largest_product : ∀ a b c : ℕ, 
  is_three_digit_number a → is_three_digit_number b → is_three_digit_number c → 
  uses_unique_digits_1_to_9 a b c → 
  a * b * c ≤ 941 * 852 * 763 :=
by
  sorry


end smallest_product_largest_product_l199_199097


namespace arccos_half_eq_pi_div_three_l199_199308

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199308


namespace greatest_multiple_of_5_and_7_lt_1000_l199_199636

theorem greatest_multiple_of_5_and_7_lt_1000 : ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) := 
  ∃ n, n = 980 ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ 980) :=
    by
  sorry

end greatest_multiple_of_5_and_7_lt_1000_l199_199636


namespace arccos_one_half_l199_199221

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199221


namespace uruguayan_goals_conceded_l199_199845

theorem uruguayan_goals_conceded (x : ℕ) (h : 14 = 9 + x) : x = 5 := by
  sorry

end uruguayan_goals_conceded_l199_199845


namespace sum_of_divisors_of_252_l199_199648

theorem sum_of_divisors_of_252 :
  ∑ (d : ℕ) in (finset.filter (λ x, 252 % x = 0) (finset.range (252 + 1))), d = 728 :=
by
  sorry

end sum_of_divisors_of_252_l199_199648


namespace balls_problem_l199_199464

noncomputable def red_balls_initial := 420
noncomputable def total_balls_initial := 600
noncomputable def percent_red_required := 60 / 100

theorem balls_problem :
  ∃ (x : ℕ), 420 - x = (3 / 5) * (600 - x) :=
by
  sorry

end balls_problem_l199_199464


namespace alpha_beta_expression_l199_199776

noncomputable theory

variables (α β : ℝ)

-- Conditions
axiom root_equation : ∀ x, (x = α ∨ x = β) → x^2 - x - 1 = 0
axiom alpha_square : α^2 = α + 1
axiom alpha_beta_sum : α + β = 1

-- The statement
theorem alpha_beta_expression : α^4 + 3 * β = 5 :=
sorry

end alpha_beta_expression_l199_199776


namespace min_distance_PS_l199_199937

-- Definitions of the distances given in the problem
def PQ : ℝ := 12
def QR : ℝ := 7
def RS : ℝ := 5

-- Hypotheses for the problem
axiom h1 : PQ = 12
axiom h2 : QR = 7
axiom h3 : RS = 5

-- The goal is to prove that the minimum distance between P and S is 0.
theorem min_distance_PS : ∃ PS : ℝ, PS = 0 :=
by
  -- The proof is omitted
  sorry

end min_distance_PS_l199_199937


namespace ice_cream_sandwiches_l199_199096

theorem ice_cream_sandwiches (n : ℕ) (x : ℕ) (h1 : n = 11) (h2 : x = 13) : (n * x = 143) := 
by
  sorry

end ice_cream_sandwiches_l199_199096


namespace arccos_one_half_l199_199241

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199241


namespace sum_of_coordinates_of_intersections_l199_199744

theorem sum_of_coordinates_of_intersections :
  let circle1 := λ x y, x^2 - 6x + y^2 - 8y + 25 = 0
  let circle2 := λ x y, x^2 - 4x + y^2 - 8y + 16 = 0
  ∃ (p1 p2 : ℝ × ℝ), 
    (circle1 p1.1 p1.2 ∧ circle2 p1.1 p1.2) ∧
    (circle1 p2.1 p2.2 ∧ circle2 p2.1 p2.2) ∧
    p1 ≠ p2 ∧ 
    p1.1 + p1.2 + p2.1 + p2.2 = 13 := 
by
  sorry

end sum_of_coordinates_of_intersections_l199_199744


namespace arccos_half_is_pi_div_three_l199_199262

-- Define the key values and the condition
def arccos_half_eq_pi_div_three : Prop :=
  arccos (1 / 2) = π / 3

-- State the theorem to be proved
theorem arccos_half_is_pi_div_three : arccos_half_eq_pi_div_three :=
  sorry

end arccos_half_is_pi_div_three_l199_199262


namespace arccos_one_half_is_pi_div_three_l199_199198

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199198


namespace cosine_of_acute_angle_l199_199435

/-
We are given two lines parameterized by:
Line 1: (x, y) = (2, -1) + s * (4, -1)
Line 2: (x, y) = (-3, 5) + t * (2, 4)

We want to prove that the cosine of the acute angle \phi formed by these two lines is \frac{2\sqrt{85}}{85}.
-/

theorem cosine_of_acute_angle 
  (v1 v2 : ℝ × ℝ) 
  (hv1 : v1 = (4, -1)) 
  (hv2 : v2 = (2, 4)) 
  (phi : ℝ) 
  (hphi : cos phi = (2 * Real.sqrt 85) / 85) :
  cos (Real.angle v1 v2) = (2 * Real.sqrt 85) / 85 :=
sorry

end cosine_of_acute_angle_l199_199435


namespace find_point_N_l199_199414

theorem find_point_N 
  (M N : ℝ × ℝ) 
  (MN_length : Real.sqrt (((N.1 - M.1) ^ 2) + ((N.2 - M.2) ^ 2)) = 4)
  (MN_parallel_y_axis : N.1 = M.1)
  (M_coord : M = (-1, 2)) 
  : (N = (-1, 6)) ∨ (N = (-1, -2)) :=
sorry

end find_point_N_l199_199414


namespace round_robin_pairing_possible_l199_199168

def players : Set String := {"A", "B", "C", "D", "E", "F"}

def is_pairing (pairs : List (String × String)) : Prop :=
  ∀ (p : String × String), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ players ∧ p.2 ∈ players

def unique_pairs (rounds : List (List (String × String))) : Prop :=
  ∀ r, r ∈ rounds → is_pairing r ∧ (∀ p1 p2, p1 ∈ r → p2 ∈ r → p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)

def all_players_paired (rounds : List (List (String × String))) : Prop :=
  ∀ p, p ∈ players →
  (∀ q, q ∈ players → p ≠ q → 
    (∃ r, r ∈ rounds ∧ (p,q) ∈ r ∨ (q,p) ∈ r))

theorem round_robin_pairing_possible : 
  ∃ rounds, List.length rounds = 5 ∧ unique_pairs rounds ∧ all_players_paired rounds :=
  sorry

end round_robin_pairing_possible_l199_199168


namespace probability_all_together_l199_199724

/--
  Three people, Anna, Borya, and Vasya, decide to go to an event.
  Each one will arrive at a random time between 15:00 and 16:00.
  Vasya will wait up to 15 minutes for someone before leaving alone.
  Borya will wait up to 10 minutes for someone before leaving alone.
  Anna will not wait for anyone.
  If Borya and Vasya meet, they will wait for Anna until 16:00.
  We aim to prove that the probability they will all go to the event together is approximately 0.124.
-/
theorem probability_all_together :
  ∃ ε > 0, abs (((
    have h1 : ∀ t ∈ Icc (0 : ℝ) 60, t <= 60, from sorry, -- Vasya and Borya meet condition
    have h2 : (permutations [15, 10, 0]).count (λ l, last l = 0) = 2, from sorry, -- Anya last condition
    have prob_anya_last : ℝ := (2 / 6), -- Probability Anya is last
    have prob_meet : ℝ := 1337.5 / 3600, -- Probability Borya and Vasya meet
    prob_anya_last * prob_meet) - 0.124) < ε

end probability_all_together_l199_199724


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199326

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199326


namespace min_value_abs_x1_x2_l199_199428

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem min_value_abs_x1_x2 
  (a : ℝ) (x1 x2 : ℝ)
  (h_symm : ∃ k : ℤ, -π / 6 - (Real.arctan (Real.sqrt 3 / a)) = (k * π + π / 2))
  (h_diff : f a x1 - f a x2 = -4) :
  |x1 + x2| = (2 * π) / 3 := 
sorry

end min_value_abs_x1_x2_l199_199428


namespace correct_propositions_count_l199_199504

variable (a α b : Type) 
variable [plane a] [line α] [line b]

-- Define the conditions as hypotheses
def cond1 : Prop := ∀ (a α b : line), a ∥ α ∧ a ⊥ b → b ⊥ α
def cond2 : Prop := ∀ (a b α : line), a ∥ b ∧ a ⊥ α → b ⊥ α
def cond3 : Prop := ∀ (a α b : line), a ⊥ α ∧ a ⊥ b → b ∥ α
def cond4 : Prop := ∀ (a α b : line), a ⊥ α ∧ b ⊥ α → a ∥ b

-- Main theorem to prove
theorem correct_propositions_count : 
  (¬cond1 a α b) ∧ (cond2 a b α) ∧ (¬cond3 a α b) ∧ (cond4 a α b) :=
by 
  sorry

end correct_propositions_count_l199_199504


namespace CitadelSchoolEarnings_l199_199726

theorem CitadelSchoolEarnings :
  let apex_students : Nat := 9
  let apex_days : Nat := 5
  let beacon_students : Nat := 3
  let beacon_days : Nat := 4
  let citadel_students : Nat := 6
  let citadel_days : Nat := 7
  let total_payment : ℕ := 864
  let total_student_days : ℕ := (apex_students * apex_days) + (beacon_students * beacon_days) + (citadel_students * citadel_days)
  let daily_wage_per_student : ℚ := total_payment / total_student_days
  let citadel_student_days : ℕ := citadel_students * citadel_days
  let citadel_earnings : ℚ := daily_wage_per_student * citadel_student_days
  citadel_earnings = 366.55 := by
  sorry

end CitadelSchoolEarnings_l199_199726


namespace edmund_earning_l199_199754

-- Definitions based on the conditions
def daily_chores := 4
def days_in_week := 7
def weeks := 2
def normal_weekly_chores := 12
def pay_per_extra_chore := 2

-- Theorem to be proven
theorem edmund_earning :
  let total_chores := daily_chores * days_in_week * weeks
      normal_chores := normal_weekly_chores * weeks
      extra_chores := total_chores - normal_chores
      earnings := extra_chores * pay_per_extra_chore
  in earnings = 64 := 
by
  sorry

end edmund_earning_l199_199754


namespace at_least_two_sets_equal_two_sets_no_common_elements_possible_l199_199911

open Set

variables (S1 S2 S3 : Set ℤ)
variables (non_empty_S1 : S1.Nonempty) (non_empty_S2 : S2.Nonempty) (non_empty_S3 : S3.Nonempty)
variables (h : ∀ x ∈ S1, ∀ y ∈ S1, ∀ i j k, ({i, j, k} = {1, 2, 3}) → (x - y) ∈ S3)

theorem at_least_two_sets_equal : S1 = S2 ∨ S2 = S3 ∨ S3 = S1 :=
begin
  sorry
end

theorem two_sets_no_common_elements_possible : ∃ S1 S2 S3 : Set ℤ, (S1.Nonempty ∧ S2.Nonempty ∧ S3.Nonempty) ∧
  (∀ x ∈ S1, ∀ y ∈ S1, ∀ i j k, ({i, j, k} = {1, 2, 3}) → (x - y) ∈ S3) ∧
  (S1 ∩ S3 = ∅ ∧ S2 ∩ S3 = ∅) :=
begin
  use {x | x % 2 = 1}, -- S1: odd integers
  use {x | x % 2 = 1}, -- S2: odd integers
  use {x | x % 2 = 0}, -- S3: even integers
  split, 
  { split, 
    { use 1, intros hS1, contradiction }, -- S1 non-empty
    split,
    { use 1, intros hS2, contradiction }, -- S2 non-empty
    { use 0, intros hS3, contradiction }  -- S3 non-empty
  },
  split,
  { intros x hx y hy i j k hperm,
    rw Set.mem_set_of_eq at hx hy,
    rw hperm at *,
    have hxy : x - y ∈ S3 := Set.mem_set_of_eq.mpr (ih hx hy),
    rw Set.mem_set_of_eq at hxy,
    exact hxy },
  { split, 
    { ext, rw [Set.mem_inter_iff, not_and_distrib],
      intro H, cases H, rw Set.mem_set_of_eq at *, linarith }, -- S1 ∩ S3 = ∅
    { ext, rw [Set.mem_inter_iff, not_and_distrib],
      intro H, cases H, rw Set.mem_set_of_eq at *, linarith } -- S2 ∩ S3 = ∅
  }
end

end at_least_two_sets_equal_two_sets_no_common_elements_possible_l199_199911


namespace chord_line_eq_l199_199956

theorem chord_line_eq (x y : ℝ) (h : x^2 + 4 * y^2 = 36) (midpoint : x = 4 ∧ y = 2) :
  x + 2 * y - 8 = 0 := 
sorry

end chord_line_eq_l199_199956


namespace find_parameters_l199_199964

theorem find_parameters (s l : ℝ)
  (line_eq : ∀ x : ℝ, y = (2 / 3) * x + 3)
  (param_eq : ∀ t : ℝ, ∃ x y : ℝ, (x, y) = (-9, s) + t * (l, -7)) :
  (s, l) = (-3, -10.5) :=
sorry

end find_parameters_l199_199964


namespace screws_per_pile_l199_199924

-- Definitions based on the given conditions
def initial_screws : ℕ := 8
def multiplier : ℕ := 2
def sections : ℕ := 4

-- Derived values based on the conditions
def additional_screws : ℕ := initial_screws * multiplier
def total_screws : ℕ := initial_screws + additional_screws

-- Proposition statement
theorem screws_per_pile : total_screws / sections = 6 := by
  sorry

end screws_per_pile_l199_199924


namespace intersects_midpoint_of_segment_bisection_l199_199912

open EuclideanGeometry

noncomputable def midpoint {α : Type} [metric_space α] (A B : α) :=
    ((dist A B) / 2 : ℝ)

def symmetric_with_respect_to (O P : Point) (Q : Point) : Prop :=
  dist O P = dist O Q ∧ ∃ dir : Vector, Q = O - (O - P)
  
def projection_to_line (P : Point) (A B : Point) : Point :=
  sorry -- Proper construction of projection not included for brevity

def bisects (Q : Point) (P H : Point) : Prop :=
  dist Q P = dist Q H

theorem intersects_midpoint_of_segment_bisection 
  (circle : Circle) (O A B M K P Q H: Point) :
  is_midpoint M A B →
  symmetric_with_respect_to O M K →
  on_circle P circle →
  let perpendicular_to_AB_at_A := {l | is_perpendicular l (line_through A B) ∧ passes_through l A}
  let perpendicular_to_PK_at_P := {l | is_perpendicular l (line_through P K) ∧ passes_through l P}
  intersect_line AB PK ⟨Q, Q ∈ ∩ perpendicular_to_AB_at_A.perpendicular_to_PK_at_P⟩ →
  projection_to_line P A B = H →    
  bisects Q P H :=
sorry

end intersects_midpoint_of_segment_bisection_l199_199912


namespace range_f_on_interval_l199_199976

open Real

def f (x : ℝ) := x^2 + 2*x - 3

theorem range_f_on_interval : 
  set.range (f ∘ λ x, x) (Icc (-3 : ℝ) (0 : ℝ)) = Icc (-4 : ℝ) (0 : ℝ) := 
sorry

end range_f_on_interval_l199_199976


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199332

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199332


namespace arccos_one_half_l199_199226

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l199_199226


namespace quadratic_function_max_value_l199_199072

theorem quadratic_function_max_value (x : ℝ) : 
  let y := -x^2 - 1 in y ≤ -1 :=
begin
  let y := -x^2 - 1,
  use x,
  show y = -1,
  sorry
end

end quadratic_function_max_value_l199_199072


namespace first_player_wins_for_large_n_l199_199680

theorem first_player_wins_for_large_n (n : ℕ) (h : n > 1) : 
  ∃ strategy, 
    (∀ k ≤ 2 * n - 2, ∃ m, strategy k = m) ∧
    (∀ last_move, strategy last_move = n - 1) := 
sorry

end first_player_wins_for_large_n_l199_199680


namespace find_2a_plus_6b_l199_199023

theorem find_2a_plus_6b (a b : ℕ) (n : ℕ)
  (h1 : 3 * a + 5 * b ≡ 19 [MOD n + 1])
  (h2 : 4 * a + 2 * b ≡ 25 [MOD n + 1])
  (hn : n = 96) :
  2 * a + 6 * b = 96 :=
by
  sorry

end find_2a_plus_6b_l199_199023


namespace minimum_pieces_to_ensure_special_piece_l199_199522

theorem minimum_pieces_to_ensure_special_piece :
  let positions_fish := [(1, 2), (5, 5), (7, 2)]
  let positions_sausage := [(2, 2), (5, 6)]
  ∃ (special_piece : ℕ × ℕ),
    special_piece ∉ [(1, 2), (5, 5), (7, 2)] ∧
    special_piece ∉ [(2, 2), (5, 6)] ∧
    (special_piece.1 ≥ 1 ∧ special_piece.1 ≤ 8) ∧
    (special_piece.2 >= 1 ∧ special_piece.2 <= 8) ∧
    (∀ sqf : ℕ × ℕ → Prop, (∀ x y, 
      (x+6 <= 8 ∧ y+6 <= 8) → sqf (x, y) → 
      (∃ f1 f2 f3, sqf f1 ∧ f1 ∈ positions_fish ∧
                   sqf f2 ∧ f2 ∈ positions_fish ∧
                   f1 ≠ f2)) ∧
    (∀ sqs : ℕ × ℕ → Prop, (∀ x y, 
      (x+3 <= 8 ∧ y+3 <= 8) → sqs (x, y) → 
      (∀ s1 s2, sqs s1 ∧ s1 ∈ positions_sausage ∧
                s1 ≠ s2 ∧ sqs s2 → false))) →
    (∀ t, t ⊆ set.univ \ {positions_fish ∪ positions_sausage} ∧
       t.card >= 5 → special_piece ∈ t)

end minimum_pieces_to_ensure_special_piece_l199_199522


namespace replace_asterisk_with_2x_l199_199537

theorem replace_asterisk_with_2x (x : ℝ) :
  ((x^3 - 2)^2 + (x^2 + 2x)^2) = x^6 + x^4 + 4x^2 + 4 :=
by sorry

end replace_asterisk_with_2x_l199_199537


namespace rectangle_proof_right_triangle_proof_l199_199841

-- Definition of rectangle condition
def rectangle_condition (a b : ℕ) : Prop :=
  a * b = 2 * (a + b)

-- Definition of right triangle condition
def right_triangle_condition (a b : ℕ) : Prop :=
  a + b + Int.natAbs (Int.sqrt (a^2 + b^2)) = a * b / 2 ∧
  (∃ c : ℕ, c = Int.natAbs (Int.sqrt (a^2 + b^2)))

-- Recangle proof
theorem rectangle_proof : ∃! p : ℕ × ℕ, rectangle_condition p.1 p.2 := sorry

-- Right triangle proof
theorem right_triangle_proof : ∃! t : ℕ × ℕ, right_triangle_condition t.1 t.2 := sorry

end rectangle_proof_right_triangle_proof_l199_199841


namespace probability_n_n_plus_1_divisible_13_and_17_l199_199721

theorem probability_n_n_plus_1_divisible_13_and_17 :
  let p : ℚ := 1 / 250 in
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → Probability (n*(n+1) % 13 = 0 ∧ n*(n+1) % 17 = 0) = p :=
by
  sorry -- proof to be completed

end probability_n_n_plus_1_divisible_13_and_17_l199_199721


namespace sum_possible_m_values_l199_199161

open Real

-- Define the conditions of the problem
def triangle_vertices := (0, 0, 2, 2, 6 * m, 0)

def dividing_line (m : ℝ) : ℝ → ℝ := λ x, (m / 2) * x

def equal_area_condition (m : ℝ) : Prop :=
  let D := (6 * m + 2) / 2
  in dividing_line m D = 1

-- Sum of all possible values of m given that the line divides the triangle equally
theorem sum_possible_m_values :
  let m := (λ (x : ℝ), @eq ℝ (real.eval_divide _ _ x) (1 : ℝ)).root
  in m.sum = -1/3 :=
sorry

end sum_possible_m_values_l199_199161


namespace original_survey_customers_approx_l199_199703

theorem original_survey_customers_approx (x : ℝ) (hx : 7 / x * 100 + 4 = 14.29) : x ≈ 68 :=
by
  -- proof goes here
  sorry

end original_survey_customers_approx_l199_199703


namespace arrangement_plans_l199_199469

-- Definition of the problem conditions
def numChineseTeachers : ℕ := 2
def numMathTeachers : ℕ := 4
def numTeachersPerSchool : ℕ := 3

-- Definition of the problem statement
theorem arrangement_plans
  (c : ℕ) (m : ℕ) (s : ℕ)
  (h1 : numChineseTeachers = c)
  (h2 : numMathTeachers = m)
  (h3 : numTeachersPerSchool = s)
  (h4 : ∀ a b : ℕ, a + b = numChineseTeachers → a = 1 ∧ b = 1)
  (h5 : ∀ a b : ℕ, a + b = numMathTeachers → a = 2 ∧ b = 2) :
  (c * (1 / 2 * m * (m - 1) / 2)) = 12 :=
sorry

end arrangement_plans_l199_199469


namespace rectangle_perimeter_l199_199701

-- Definitions based on conditions
def length (w : ℝ) : ℝ := 2 * w
def width (w : ℝ) : ℝ := w
def area (w : ℝ) : ℝ := length w * width w
def perimeter (w : ℝ) : ℝ := 2 * (length w + width w)

-- Problem statement: Prove that the perimeter is 120 cm given area is 800 cm² and length is twice the width
theorem rectangle_perimeter (w : ℝ) (h : area w = 800) : perimeter w = 120 := by
  sorry

end rectangle_perimeter_l199_199701


namespace arccos_half_eq_pi_over_three_l199_199210

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199210


namespace value_of_b_l199_199627

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end value_of_b_l199_199627


namespace chefs_drop_out_l199_199955

theorem chefs_drop_out : 
  let total_chefs := 16
  let total_waiters := 16
  let waiters_drop := 3
  let total_remaining := 23
  let initial_staff := total_chefs + total_waiters
  let waiters_left := total_waiters - waiters_drop
  let chefs_left := total_remaining - waiters_left
  in 
  total_chefs - chefs_left = 6 :=
by
  sorry

end chefs_drop_out_l199_199955


namespace trigonometric_range_l199_199916

theorem trigonometric_range :
  (∀ x : ℝ, (-π / 2) ≤ x ∧ x ≤ (3 * π / 2) → (√(1 + sin (2 * x)) = sin x + cos x))
  → (∀ x : ℝ, (-π / 4) ≤ x ∧ x ≤ (3 * π / 4)) :=
begin
  sorry
end

end trigonometric_range_l199_199916


namespace length_of_AB_l199_199615

-- Define the variables and constants
variables (DE EF BC AB : ℝ)
variable (h_similar : triangle_similar DEF ABC)

-- Given conditions
def conditions : Prop := DE = 6 ∧ EF = 12 ∧ BC = 18

-- The goal to prove
theorem length_of_AB (h : conditions) : AB = 9 :=
by
  sorry

end length_of_AB_l199_199615


namespace true_propositions_l199_199958

--  Condition definitions
def complementary (a b : ℝ) : Prop := a + b = 90
def supplementary (a b : ℝ) : Prop := a + b = 180
def parallel_postulate (L1 L2 : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  (∃! L2, ∀ x, L1 x = 0 → L2 x = 0) ∧ P.1 ≠ 0
def perpendicular_distance_criterion (P : ℝ × ℝ) (L : ℝ → ℝ) : Prop :=
  ∀ Q : ℝ × ℝ, (L Q.1 = Q.2) → (P.1 = Q.1 ∧ P.2 = Q.2)

-- Proposition definitions
def prop1 (a b : ℝ) : Prop := (complementary a b) → (a = b)
def prop2 (a b : ℝ) : Prop := (supplementary a b) → (complementary (a/2) (b/2))
def prop3 (L : ℝ → ℝ) (P : ℝ × ℝ) : Prop := parallel_postulate L L P
def prop4 (P : ℝ × ℝ) (L : ℝ → ℝ) : Prop := perpendicular_distance_criterion P L

-- Proof problem statement
theorem true_propositions (a b : ℝ) (L : ℝ → ℝ) (P : ℝ × ℝ) :
  ¬ prop1 a b ∧ prop2 a b ∧ prop3 L P ∧ prop4 P L :=
by
  sorry

end true_propositions_l199_199958


namespace min_value_inequality_l199_199509

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 * x / (2 * y + z) + 3 * y / (x + 2 * z) + 9 * z / (x + y) ≥ 83 :=
sorry

end min_value_inequality_l199_199509


namespace hash_four_times_l199_199739

noncomputable def hash (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_four_times (N : ℝ) : hash (hash (hash (hash N))) = 11.8688 :=
  sorry

end hash_four_times_l199_199739


namespace combined_alloy_force_l199_199714

-- Define the masses and forces exerted by Alloy A and Alloy B
def mass_A : ℝ := 6
def force_A : ℝ := 30
def mass_B : ℝ := 3
def force_B : ℝ := 10

-- Define the combined mass and force
def combined_mass : ℝ := mass_A + mass_B
def combined_force : ℝ := force_A + force_B

-- Theorem statement
theorem combined_alloy_force :
  combined_force = 40 :=
by
  -- The proof is omitted.
  sorry

end combined_alloy_force_l199_199714


namespace initial_number_of_fruits_l199_199710

theorem initial_number_of_fruits (oranges apples limes : ℕ) (h_oranges : oranges = 50)
  (h_apples : apples = 72) (h_oranges_limes : oranges = 2 * limes) (h_apples_limes : apples = 3 * limes) :
  (oranges + apples + limes) * 2 = 288 :=
by
  sorry

end initial_number_of_fruits_l199_199710


namespace green_beads_initially_l199_199088

theorem green_beads_initially (b r remaining_beads total_taken g : ℕ) (h_b : b = 2) (h_r : r = 3) (h_total_taken : total_taken = 2) (h_remaining_beads : remaining_beads = 4) :
  g = 1 :=
by
  have h_initial_beads := h_total_taken + h_remaining_beads
  have h_total_initial := h_b + h_r
  have h := h_initial_beads - h_total_initial
  have h_g := g = h
  sorry

end green_beads_initially_l199_199088


namespace num_intersections_inside_circle_l199_199944

-- Define the problem conditions
def regular_polygons (sides : ℕ) : Prop := 
  sides ∈ {4, 7, 9, 10}

def inscribed_in_same_circle : Prop := true

def no_shared_vertices_or_three_side_intersections : Prop := true

-- Main theorem statement
theorem num_intersections_inside_circle : 
  (regular_polygons 4) → 
  (regular_polygons 7) → 
  (regular_polygons 9) → 
  (regular_polygons 10) →
  inscribed_in_same_circle →
  no_shared_vertices_or_three_side_intersections →
  ∃ (n : ℕ), n = 70 :=
by 
  sorry -- skipping proof

end num_intersections_inside_circle_l199_199944


namespace arccos_half_eq_pi_div_3_l199_199272

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199272


namespace sin_value_l199_199779

variable {α : Real}

theorem sin_value (h : cos (α - π / 3) = -1 / 2) : sin (π / 6 + α) = -1 / 2 := 
  sorry

end sin_value_l199_199779


namespace total_distance_covered_l199_199678

theorem total_distance_covered (h : ℝ) : (h > 0) → 
  ∑' n : ℕ, (h * (0.8 : ℝ) ^ n + h * (0.8 : ℝ) ^ (n + 1)) = 5 * h :=
  by
  sorry

end total_distance_covered_l199_199678


namespace patrick_purchased_pencils_l199_199043

theorem patrick_purchased_pencils (c s : ℝ) : 
  (∀ n : ℝ, n * c = 1.375 * n * s ∧ (n * c - n * s = 30 * s) → n = 80) :=
by sorry

end patrick_purchased_pencils_l199_199043


namespace five_digit_even_numbers_count_l199_199842

theorem five_digit_even_numbers_count :
  (∀ n ∈ ({1, 2, 3, 4, 5} : Finset ℕ), 
   n ∈ ({1, 2, 3, 4, 5} : Finset ℕ) ∧
   ∀ m, m ∈ n.digits 10 →
     (m ∈ ({1, 2, 3, 4, 5} : Finset ℕ) →
     m ≠ last (n.digits 10)) ∧
     list.distinct (n.digits 10) ∧
     last (n.digits 10) % 2 = 0) → 
  (card = 48) sorry

end five_digit_even_numbers_count_l199_199842


namespace range_of_k_l199_199426

def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 1)

def g (x k : ℝ) : ℝ := x^2 - 2 * k * x + 4 * k + 3

theorem range_of_k (k : ℝ) : 
  (∀ x1 ∈ set.Icc (-1 : ℝ) 1, ∃ x2 ∈ set.Ioi (1 : ℝ), f x1 = g x2 k) ↔ 
  (k < - 5 / 2 ∨ k ≥ 2 + 2 * real.sqrt 2) :=
sorry

end range_of_k_l199_199426


namespace team_winning_percentage_l199_199704

theorem team_winning_percentage (wins_first_30 : ℕ) (total_games : ℕ) (total_win_percentage : ℕ) (approx_total_games : ℕ) :
    (wins_first_30 = 12) ∧ (total_win_percentage = 60) ∧ (approx_total_games = 60) →
    let remaining_games := approx_total_games - 30 in
    let remaining_wins := total_win_percentage * approx_total_games / 100 - wins_first_30 in
    let remaining_win_percentage := remaining_wins * 100 / remaining_games in
    remaining_win_percentage = 80 :=
begin
    sorry
end

end team_winning_percentage_l199_199704


namespace minimum_k_exists_l199_199125

theorem minimum_k_exists :
  ∀ (s : Finset ℝ), s.card = 3 → (∀ (a b : ℝ), a ∈ s → b ∈ s → (|a - b| ≤ (1.5 : ℝ) ∨ |(1 / a) - (1 / b)| ≤ 1.5)) :=
by
  sorry

end minimum_k_exists_l199_199125


namespace pechkin_fraction_l199_199624

variable (P U M S : ℝ)

-- Conditions from the problem
def UncleFyodor (P : ℝ) : ℝ := P / 2
def CatMatroskin (P : ℝ) : ℝ := (1 - P) / 2
def Sharik : ℝ := 0.1

-- The claim we need to prove
theorem pechkin_fraction (h1 : U = UncleFyodor P) 
                         (h2 : M = CatMatroskin P) 
                         (h3 : S = Sharik) 
                         (h4 : P + S = 0.5) : 
  P = 0.4 :=
by
  skip -- proof body is not required
  sorry

end pechkin_fraction_l199_199624


namespace arccos_half_eq_pi_div_three_l199_199341

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199341


namespace sum_of_repeating_decimal_l199_199581

-- Definitions and conditions from (a)
def fraction := 1 / (98^2)
def n : Nat := 96 -- length of the period for the repeating decimal

-- Main theorem to be proven
theorem sum_of_repeating_decimal (d : Fin n → Nat) (h : 0 < n) :
  (1 / (98^2) = 0.\overline{d_{n-1}d_{n-2}\ldots d_2d_1d_0}) → (∑ i in Finset.range n, d i) = 1260 :=
sorry

end sum_of_repeating_decimal_l199_199581


namespace part1_part2_l199_199425

-- Define the function f(x) = ln x - kx + 1
def f (x k : ℝ) : ℝ := Real.log x - k * x + 1

-- Condition for part (1): interval of monotonic increase for k=2
theorem part1 (x : ℝ) (hx : 0 < x ∧ x < 1/2) : 
  (f x 2) > (f x 2) := by sorry

-- Define a function g(x) for part (2)
def g (x : ℝ) : ℝ := (Real.log x + 1) / x

-- Condition for part (2): range of k given that f(x) ≤ 0 always holds
theorem part2 (k : ℝ) (h : ∀ x > 0, f x k ≤ 0) : k ≥ 1 := by sorry

end part1_part2_l199_199425


namespace isosceles_triangles_area_ratio_l199_199622

open Lean

theorem isosceles_triangles_area_ratio 
  (b₁ h₁ b₂ h₂ : ℝ)
  (h_eq_vert_angles : ∃ (α : ℝ), isosceles_triangle_with_angle b₁ h₁ α ∧ isosceles_triangle_with_angle b₂ h₂ α)
  (h_height_ratio : h₁ / h₂ = 0.6) :
  (1 / 2 * b₁ * h₁) / (1 / 2 * b₂ * h₂) = 0.36 := 
sorry

-- Definitions for isosceles_triangle_with_angle are assumed to exist in the Mathlib library. 
-- If not, you have to define them based on the problem context.

end isosceles_triangles_area_ratio_l199_199622


namespace projection_of_v_onto_plane_l199_199763

noncomputable def v : ℝ^3 := ![1, 2, 3]
noncomputable def normal_vector : ℝ^3 := ![3, -1, 4]

def plane (x y z : ℝ) : Prop := 3 * x - y + 4 * z = 0

noncomputable def projection_onto_plane := ![-1/2, 5/2, 1]

theorem projection_of_v_onto_plane :
  let proj := (v - (v ⬝ normal_vector) / (normal_vector ⬝ normal_vector) • normal_vector)
  proj = projection_onto_plane :=
sorry

end projection_of_v_onto_plane_l199_199763


namespace arccos_of_half_eq_pi_over_three_l199_199297

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199297


namespace train_bus_cost_difference_l199_199706

theorem train_bus_cost_difference :
  let bus_cost := 3.75
  let total_cost := 9.85
  let train_cost := total_cost - bus_cost
  in train_cost - bus_cost = 2.35 :=
by
  let bus_cost := 3.75
  let total_cost := 9.85
  let train_cost := total_cost - bus_cost
  show train_cost - bus_cost = 2.35
  sorry

end train_bus_cost_difference_l199_199706


namespace exponent_equality_l199_199654

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality_l199_199654


namespace no_tiling_with_pentagons_and_decagons_l199_199887

theorem no_tiling_with_pentagons_and_decagons :
  (∀ A B C D E F G H I J : ℝ, A = 108 ∧ B = 108 ∧ C = 144 ∧ D = 108 ∧ E = 108 ∧ F = 144 ∧ G = 108 ∧ H = 108 ∧ I = 144 ∧ J = 108) →
  (∀ P Q R S T U V W X Y : ℝ, P = Q ∧ R = S ∧ T = U ∧ V = W ∧ X = Y) →
  False :=
begin
  intros hAng hSum,
  sorry -- Proof goes here
end

end no_tiling_with_pentagons_and_decagons_l199_199887


namespace greatest_multiple_of_5_and_7_lt_1000_l199_199637

theorem greatest_multiple_of_5_and_7_lt_1000 : ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) := 
  ∃ n, n = 980 ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ 980) :=
    by
  sorry

end greatest_multiple_of_5_and_7_lt_1000_l199_199637


namespace gf_three_l199_199907

def f (x : ℕ) : ℕ := x^3 - 4 * x + 5
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem gf_three : g (f 3) = 1222 :=
by {
  -- We would need to prove the given mathematical statement here.
  sorry
}

end gf_three_l199_199907


namespace find_omega_range_l199_199430

-- Define the function f
def f (ω x : ℝ) : ℝ := cos (ω * x) + sin (ω * x + π / 6)

-- Define the conditions
def condition (ω : ℝ) : Prop :=
  ω > 0 ∧ (∃ (a b : ℝ), a = 2 * π ∧ b = (5 * π) / 2 ∧ a ≤ ω * π + π / 3 ∧ ω * π + π / 3 < b)

-- The main statement we want to prove
theorem find_omega_range (ω : ℝ) (hω : condition ω) :
  ∃ I : Set ℝ, I = Set.Ico (5 / 3) (13 / 6) ∧ (∀ x ∈ Set.Icc 0 π, f ω x = 0) ∧ (∃ x ∈ Set.Icc 0 π, IsMax (f ω x)) := sorry

end find_omega_range_l199_199430


namespace jennifer_fruits_left_l199_199001

-- Definitions based on the conditions
def pears : ℕ := 15
def oranges : ℕ := 30
def apples : ℕ := 2 * pears
def cherries : ℕ := oranges / 2
def grapes : ℕ := 3 * apples
def pineapples : ℕ := pears + oranges + apples + cherries + grapes

-- Definitions for the number of fruits given to the sister
def pears_given : ℕ := 3
def oranges_given : ℕ := 5
def apples_given : ℕ := 5
def cherries_given : ℕ := 7
def grapes_given : ℕ := 3

-- Calculations based on the conditions for what's left after giving fruits
def pears_left : ℕ := pears - pears_given
def oranges_left : ℕ := oranges - oranges_given
def apples_left : ℕ := apples - apples_given
def cherries_left : ℕ := cherries - cherries_given
def grapes_left : ℕ := grapes - grapes_given

def remaining_pineapples : ℕ := pineapples - (pineapples / 2)

-- Total number of fruits left
def total_fruits_left : ℕ := pears_left + oranges_left + apples_left + cherries_left + grapes_left + remaining_pineapples

-- Theorem statement
theorem jennifer_fruits_left : total_fruits_left = 247 :=
by
  -- The detailed proof would go here
  sorry

end jennifer_fruits_left_l199_199001


namespace number_of_correct_propositions_l199_199800

theorem number_of_correct_propositions 
  (α β γ : Plane) (hαβ : α ≠ β) (hαγ : α ≠ γ) (hβγ : β ≠ γ)
  (m n : Line) (hmn : m ≠ n) :
  (¬ (α ⟂ γ ∧ β ⟂ γ → α ∥ β)) ∧
  (¬ (α ⟂ β ∧ β ⟂ γ → α ⟂ γ)) ∧
  (¬ (m ⟂ α ∧ α ⟂ β → m ∥ β)) ∧
  (m ⟂ α ∧ n ⟂ α → m ∥ n) →
  1 :=
by sorry

end number_of_correct_propositions_l199_199800


namespace avocado_cost_l199_199518

/-- Proof Problem Setup:
Let initial_amount be 20 (in dollars),
Let change be 14 (in dollars),
Let num_avocados be 3 (the number of avocados bought),
Prove that the cost per avocado is 2 (in dollars).
-/
theorem avocado_cost (initial_amount : ℕ) (change : ℕ) (num_avocados : ℕ) (cost_per_avocado : ℕ) :
  initial_amount = 20 → change = 14 → num_avocados = 3 → cost_per_avocado = (initial_amount - change) / num_avocados → cost_per_avocado = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact sorry

end avocado_cost_l199_199518


namespace complementary_angle_l199_199812

theorem complementary_angle (α : ℝ) (h : α = 35 + 30 / 60) : 90 - α = 54 + 30 / 60 :=
by
  sorry

end complementary_angle_l199_199812


namespace arccos_half_eq_pi_div_three_l199_199335

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199335


namespace EdProblem_l199_199362

/- Define the conditions -/
def EdConditions := 
  ∃ (m : ℕ) (N : ℕ), 
    m = 16 ∧ 
    N = Nat.choose 15 5 ∧
    N % 1000 = 3

/- The statement to be proven -/
theorem EdProblem : EdConditions :=
  sorry

end EdProblem_l199_199362


namespace max_books_l199_199361

theorem max_books (cost_per_book : ℝ) (total_money : ℝ) (h_cost : cost_per_book = 8.75) (h_money : total_money = 250.0) :
  ∃ n : ℕ, n = 28 ∧ cost_per_book * n ≤ total_money ∧ ∀ m : ℕ, cost_per_book * m ≤ total_money → m ≤ 28 :=
by
  sorry

end max_books_l199_199361


namespace charles_wage_more_than_robin_l199_199670

-- Definitions of the conditions
def erica_wage : ℝ := E
def robin_wage : ℝ := 1.3 * erica_wage
def charles_wage : ℝ := 1.7 * erica_wage

-- Statement to prove: the wage earned by Charles is 30.77% more than that earned by Robin
theorem charles_wage_more_than_robin : 
  ((charles_wage - robin_wage) / robin_wage) * 100 = 30.77 := by
  sorry

end charles_wage_more_than_robin_l199_199670


namespace arccos_half_eq_pi_div_3_l199_199279

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l199_199279


namespace closest_integer_to_harmonic_sum_l199_199377

noncomputable def harmonic_sum : ℤ :=
  2000 * (∑ n in finset.range 14997 \ finset.range 4, 1 / (n^2 - 16))

theorem closest_integer_to_harmonic_sum : harmonic_sum ≈ 458 :=
sorry

end closest_integer_to_harmonic_sum_l199_199377


namespace sum_binom_coeffs_l199_199981

theorem sum_binom_coeffs (x : ℝ) : 
  (∑ k in Finset.range (11), binomial 10 k * (x^2)^k * (-(1/√x))^(10 - k)) = 2^10 := 
sorry

end sum_binom_coeffs_l199_199981


namespace determinant_zero_implies_sum_neg_nine_l199_199908

theorem determinant_zero_implies_sum_neg_nine
  (x y : ℝ)
  (h1 : x ≠ y)
  (h2 : x * y = 1)
  (h3 : (Matrix.det ![
    ![1, 5, 8], 
    ![3, x, y], 
    ![3, y, x]
  ]) = 0) : 
  x + y = -9 := 
sorry

end determinant_zero_implies_sum_neg_nine_l199_199908


namespace prism_volume_l199_199358

theorem prism_volume (a b c : ℝ) (h1 : a * b = 12) (h2 : b * c = 8) (h3 : a * c = 4) : a * b * c = 8 * Real.sqrt 6 :=
by 
  sorry

end prism_volume_l199_199358


namespace expand_polynomials_l199_199367

variable (t : ℝ)

def poly1 := 3 * t^2 - 4 * t + 3
def poly2 := -2 * t^2 + 3 * t - 4
def expanded_poly := -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12

theorem expand_polynomials : (poly1 * poly2) = expanded_poly := 
by
  sorry

end expand_polynomials_l199_199367


namespace locus_of_M_l199_199931

theorem locus_of_M (O A B C A1 B1 C1 : ℝ^3)
  (h1 : dist O A = dist O B)
  (h2 : dist O B = dist O C)
  (h3 : ∀ l : ℝ^3, l ≠ 0 ↔ l ∈ ℝ^3)
  (h4 : symmetric_wrt_line A1 A)
  (h5 : symmetric_wrt_line B1 B)
  (h6 : symmetric_wrt_line C1 C)
  (h7 : plane_eq (perpendicular_to OA A1)
                 (perpendicular_to OB B1)
                 (perpendicular_to OC C1)) :
  { M : ℝ^3 | |M.x| ≤ a ∧ |M.y| ≤ a ∧ |M.z| ≤ a ∧ (M.x + M.y + M.z = -a) } :=
sorry

end locus_of_M_l199_199931


namespace complex_number_solution_l199_199814

-- Define the given complex number
def z : ℂ := (2 * complex.I) / (1 - complex.I)

-- State the theorem to prove
theorem complex_number_solution :
  z = -1 + complex.I :=
by
  sorry

end complex_number_solution_l199_199814


namespace gain_percent_is_10_l199_199665

-- Define the conditions
def cost_price : ℝ := 100
def selling_price : ℝ := 110

-- Define the gain calculation
def gain := selling_price - cost_price

-- Define the gain percent calculation
def gain_percent := (gain / cost_price) * 100

-- Theorem stating the gain percent is 10%
theorem gain_percent_is_10 : gain_percent = 10 := by
  sorry

end gain_percent_is_10_l199_199665


namespace sum_of_powers_2017_l199_199783

theorem sum_of_powers_2017 (n : ℕ) (x : Fin n → ℤ) (h : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -1) (h_sum : (Finset.univ : Finset (Fin n)).sum x = 1000) :
  (Finset.univ : Finset (Fin n)).sum (λ i => (x i)^2017) = 1000 :=
by
  sorry

end sum_of_powers_2017_l199_199783


namespace max_guests_theorem_l199_199007

noncomputable def maxGuests (n : ℕ) : ℕ :=
if n = 1 then 1 else n^4 - n^3

theorem max_guests_theorem (n : ℕ) (h_pos : 0 < n) :
  (∀ (guests : ℕ) (h : guests ≤ maxGuests n),
    ∃ (arrangements : Fin n → Fin n → Fin n → Fin n → Fin n),
      (∀ i j, i ≠ j → arrangements i ≠ arrangements j) ∧
      (¬ ∃ (subset : Fin n → ℕ),
        (∀ i, subset i < n) ∧ 
        (∃ (fixed_aspects : Fin 4),
          ∀ i j, i ≠ j →
           (arrangements i) (fixed_aspects i) = (arrangements j) (fixed_aspects j))) :=
sorry

end max_guests_theorem_l199_199007


namespace distance_problem_l199_199760

open Real EuclideanGeometry

def ptA : ℝ × ℝ × ℝ := (2, 0, -1)
def ptP1 : ℝ × ℝ × ℝ := (1, 3, 2)
def ptP2 : ℝ × ℝ × ℝ := (0, -1, 5)

def distance_from_point_to_line
  (a p1 p2 : ℝ × ℝ × ℝ ) : ℝ := 
  let u := (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3) in
  let v := (p1.1 - a.1, p1.2 - a.2, p1.3 - a.3) in
  let cross_prod := (v.2 * u.3 - v.3 * u.2, v.3 * u.1 - v.1 * u.3, v.1 * u.2 - v.2 * u.1) in
  (Real.sqrt (cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2)) / 
  (Real.sqrt (u.1^2 + u.2^2 + u.3^2))

theorem distance_problem : 
  distance_from_point_to_line ptA ptP1 ptP2 = 5 := sorry

end distance_problem_l199_199760


namespace circumcircles_concurrent_l199_199895

variables {A B C H1 H2 H3 T1 T2 T3 P1 P2 P3 : Type}

-- Conditions of the problem
variables (altitudes: A B C → H1 H2 H3)
variables (incircle_tangent: ∀ x y z, x ∈ [y, z] → T1 = T2 = T3)
variables (isosceles_triangle: ∀ k : ℕ, k = 1 ∨ k = 2 ∨ k = 3 → 
  (∃ P_i : Type, P_i ∈ H_iH_{i+1} ∧ H_iT_iP_i ∧ H_iT_i = H_iP_i))

theorem circumcircles_concurrent 
  (h : ∀ (k : ℕ), k = 1 ∨ k = 2 ∨ k = 3 → P_i point on H_iH_{i+1} such that H_iT_iP_i is an isosceles triangle with H_iT_i = H_iP_i) 
  : ∃ H : Point, ∀ k : ℕ, circumcircle (T_kP_kT_{k+1}) passes through the common point H :=
sorry

end circumcircles_concurrent_l199_199895


namespace max_value_of_F_when_m_eq_e_range_of_m_given_inequality_l199_199014

namespace ProofStatements

variable {x : ℝ}

/- Define the functions given in the problem -/
def f (x : ℝ) : ℝ := Real.log x
def g (x m : ℝ) : ℝ := x^2 - x - m + 2
def F (x m : ℝ) : ℝ := f x - g x m

/- Part Ⅰ - Prove the maximum value of F(x) for m = e is e - 2 -/
theorem max_value_of_F_when_m_eq_e : ∃ x : ℝ, F x Real.exp = Real.exp - 2 := 
sorry

/- Part Ⅱ - Prove the range of m given the inequality constraint  -/
theorem range_of_m_given_inequality : 
  (∀ x : ℝ, 0 < x ∧ x < 2 -> f x + g x m ≤ x^2 - (x - 2) * Real.exp x) 
  → ∀ m : ℝ, m ≥ Real.log 2 := 
sorry

end ProofStatements

end max_value_of_F_when_m_eq_e_range_of_m_given_inequality_l199_199014


namespace arccos_one_half_l199_199240

theorem arccos_one_half : arccos (1/2) = π / 3 :=
by
  sorry

end arccos_one_half_l199_199240


namespace arccos_half_eq_pi_div_three_l199_199319

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end arccos_half_eq_pi_div_three_l199_199319


namespace base_conversion_l199_199954

theorem base_conversion (b : ℝ) (h : 2 * b^2 + 3 = 51) : b = 2 * Real.sqrt 6 :=
by
  sorry

end base_conversion_l199_199954


namespace find_b_l199_199865

theorem find_b (a b : ℝ) (h : ∀ x, 2 * x^2 - a * x + 4 < 0 ↔ 1 < x ∧ x < b) : b = 2 :=
sorry

end find_b_l199_199865


namespace remainder_abc_mod9_l199_199905

open Nat

-- Define the conditions for the problem
variables (a b c : ℕ)

-- Assume conditions: a, b, c are non-negative and less than 9, and the given congruences
theorem remainder_abc_mod9 (h1 : a < 9) (h2 : b < 9) (h3 : c < 9)
  (h4 : (a + 3 * b + 2 * c) % 9 = 3)
  (h5 : (2 * a + 2 * b + 3 * c) % 9 = 6)
  (h6 : (3 * a + b + 2 * c) % 9 = 1) :
  (a * b * c) % 9 = 4 :=
sorry

end remainder_abc_mod9_l199_199905


namespace gcd_problem_l199_199585

theorem gcd_problem :
  ∃ n : ℕ, (80 ≤ n) ∧ (n ≤ 100) ∧ (n % 9 = 0) ∧ (Nat.gcd n 27 = 9) ∧ (n = 90) :=
by sorry

end gcd_problem_l199_199585


namespace sum_of_square_areas_l199_199722

variable (WX XZ : ℝ)

theorem sum_of_square_areas (hW : WX = 15) (hX : XZ = 20) : WX^2 + XZ^2 = 625 := by
  sorry

end sum_of_square_areas_l199_199722


namespace sum_of_divisors_252_l199_199647

theorem sum_of_divisors_252 :
  let n := 252
  let prime_factors := [2, 2, 3, 3, 7]
  sum_of_divisors n = 728 :=
by
  sorry

end sum_of_divisors_252_l199_199647


namespace system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l199_199764

theorem system_of_equations_solution_non_negative (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x ≥ 0) (h4 : y ≥ 0) : x = 1 ∧ y = 0 :=
sorry

theorem system_of_equations_solution_positive_sum (x y : ℝ) (h1 : x^3 + y^3 + 3 * x * y = 1) (h2 : x^2 - y^2 = 1) (h3 : x + y > 0) : x = 1 ∧ y = 0 :=
sorry

end system_of_equations_solution_non_negative_system_of_equations_solution_positive_sum_l199_199764


namespace find_beta_l199_199010

theorem find_beta (α β : ℝ)
  (h₀ : 0 < α ∧ α < π / 2)
  (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : cos α = 1 / 7)
  (h₃ : sin (α + β) = (5 * sqrt 3) / 14) :
  β = π / 3 :=
sorry

end find_beta_l199_199010


namespace three_digit_number_formed_by_second_row_is_531_l199_199368

-- Definitions for the conditions
variable (a11 a12 a13 a14 a15 : ℕ)
variable (a21 a22 a23 a24 a25 : ℕ)

// Condition 1: Each row contains numbers 1 to 5 without repetition
def row_valid : Prop :=
  {a11, a12, a13, a14, a15} = {1, 2, 3, 4, 5} ∧
  {a21, a22, a23, a24, a25} = {1, 2, 3, 4, 5}

// Condition 2: Each column contains numbers 1 to 5 without repetition
def column_valid : Prop :=
  {a11, a21} = {a11, a21} ∧
  {a12, a22} = {a12, a22} ∧
  {a13, a23} = {a13, a23} ∧
  {a14, a24} = {a14, a24} ∧
  {a15, a25} = {a15, a25}

// Condition 3: Divisibility rule
def divisible_rule : Prop :=
  a21 % a11 = 0 ∧ a22 % a12 = 0 ∧ a23 % a13 = 0 ∧
  a24 % a14 = 0 ∧ a25 % a15 = 0 ∧
  a12 % a11 = 0 ∧ a13 % a12 = 0 ∧ a14 % a13 = 0 ∧ a15 % a14 = 0 ∧
  a22 % a21 = 0 ∧ a23 % a22 = 0 ∧ a24 % a23 = 0 ∧ a25 % a24 = 0

-- The proof problem: the first three digits of the second row form the number 531.
theorem three_digit_number_formed_by_second_row_is_531 :
  row_valid a11 a12 a13 a14 a15 a21 a22 a23 a24 a25 →
  column_valid a11 a12 a13 a14 a15 a21 a22 a23 a24 a25 →
  divisible_rule a11 a12 a13 a14 a15 a21 a22 a23 a24 a25 →
  a21 * 100 + a22 * 10 + a23 = 531 :=
by
  intros _ _ _
  sorry

end three_digit_number_formed_by_second_row_is_531_l199_199368


namespace minimum_a10_l199_199902

def Γ (S : Set ℕ) : ℕ := S.sum id

def extra_conditions (A : Finset ℕ) (ΓA : ∀ n ≤ 1500, ∃ S ⊆ A, Γ S = n) :=
  (∀ (a₁ a₂ : ℕ), a₁ ∈ A → a₂ ∈ A → a₁ < a₂ → a₁ + a₂ ∈ A) ∧
  (∀ a ∈ A, ∃ S ⊆ A, Γ S = a)

theorem minimum_a10 :
  ∀ (A : Finset ℕ),
  (∀ (n : ℕ), n ≤ 1500 → ∃ S ⊆ A, Γ S = n) →
  sorted (A.sort (<)) →
  ∃ a₁ a₂ ... a₁₀ ∈ A, minimum { a₁₀ | true } = 248 :=
sorry

end minimum_a10_l199_199902


namespace purple_cars_count_l199_199491

theorem purple_cars_count
    (P R G : ℕ)
    (h1 : R = P + 6)
    (h2 : G = 4 * R)
    (h3 : P + R + G = 312) :
    P = 47 :=
by 
  sorry

end purple_cars_count_l199_199491


namespace spending_difference_l199_199737

-- Define the conditions
def spent_on_chocolate : ℤ := 7
def spent_on_candy_bar : ℤ := 2

-- The theorem to be proven
theorem spending_difference : (spent_on_chocolate - spent_on_candy_bar = 5) :=
by sorry

end spending_difference_l199_199737


namespace vacation_cost_division_l199_199603

theorem vacation_cost_division (total_cost : ℕ) (cost_per_person3 different_cost : ℤ) (n : ℕ)
  (h1 : total_cost = 375)
  (h2 : cost_per_person3 = total_cost / 3)
  (h3 : different_cost = cost_per_person3 - 50)
  (h4 : different_cost = total_cost / n) :
  n = 5 :=
  sorry

end vacation_cost_division_l199_199603


namespace decreasing_functions_l199_199584

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

def f1 (x : ℝ) : ℝ := -x + 1
def f2 (x : ℝ) : ℝ := -(3 / x)
def f3 (x : ℝ) : ℝ := x^2 + x - 2

theorem decreasing_functions : (is_decreasing f1 ∧ ¬ is_decreasing f2 ∧ ¬ is_decreasing f3) := 
by
  sorry

end decreasing_functions_l199_199584


namespace sum_of_cube_faces_consecutive_even_integers_l199_199067

theorem sum_of_cube_faces_consecutive_even_integers :
  ∃ n : ℕ, 2*n + 30 = 42 ∧ (2 divides n) := 
begin
  sorry
end

end sum_of_cube_faces_consecutive_even_integers_l199_199067


namespace largest_divisor_of_expression_of_even_x_l199_199852

theorem largest_divisor_of_expression_of_even_x (x : ℤ) (h_even : ∃ k : ℤ, x = 2 * k) :
  ∃ (d : ℤ), d = 240 ∧ d ∣ ((8 * x + 2) * (8 * x + 4) * (4 * x + 2)) :=
by
  sorry

end largest_divisor_of_expression_of_even_x_l199_199852


namespace competition_end_time_l199_199143

-- Definitions for the problem conditions
def start_time : ℕ := 15 * 60  -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 1300       -- competition duration in minutes
def end_time : ℕ := start_time + duration

-- The expected end time in minutes from midnight, where 12:40 p.m. is (12*60 + 40) = 760 + 40 = 800 minutes from midnight.
def expected_end_time : ℕ := 12 * 60 + 40 

-- The theorem to prove
theorem competition_end_time : end_time = expected_end_time := by
  sorry

end competition_end_time_l199_199143


namespace log_problem_l199_199178

noncomputable def log : ℝ → ℝ
| x := real.log x / real.log 10

theorem log_problem :
  (log 2)^2 + log 20 * log 5 = 1 :=
by
  -- Additional assumptions capturing given logarithmic properties.
  have log_10 : log 10 = 1, from sorry,
  have log_mult (a b : ℝ) : log (a * b) = log a + log b, from sorry,
  sorry

end log_problem_l199_199178


namespace range_of_g_l199_199030

open Real

noncomputable def g (x : ℝ) : ℝ := (arccos x) * (arcsin x)

theorem range_of_g :
  ∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), 0 ≤ g x ∧ g x ≤ (π ^ 2) / 8 :=
by
  intros x hx
  have h1 : arccos x + arcsin x = π / 2 :=
    arcsin_arccos_add x hx.left hx.right
  sorry

end range_of_g_l199_199030


namespace apples_difference_l199_199004

theorem apples_difference 
  (father_apples : ℕ := 8)
  (mother_apples : ℕ := 13)
  (jungkook_apples : ℕ := 7)
  (brother_apples : ℕ := 5) :
  max father_apples (max mother_apples (max jungkook_apples brother_apples)) - 
  min father_apples (min mother_apples (min jungkook_apples brother_apples)) = 8 :=
by
  sorry

end apples_difference_l199_199004


namespace greatest_multiple_of_5_and_7_lt_1000_l199_199638

theorem greatest_multiple_of_5_and_7_lt_1000 : ∃ n, n < 1000 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) := 
  ∃ n, n = 980 ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 1000 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ 980) :=
    by
  sorry

end greatest_multiple_of_5_and_7_lt_1000_l199_199638


namespace number_of_boys_l199_199609

theorem number_of_boys 
  (B G : ℕ) 
  (h1 : B + G = 650) 
  (h2 : G = B + 106) :
  B = 272 :=
sorry

end number_of_boys_l199_199609


namespace expand_polynomials_l199_199366

variable (t : ℝ)

def poly1 := 3 * t^2 - 4 * t + 3
def poly2 := -2 * t^2 + 3 * t - 4
def expanded_poly := -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12

theorem expand_polynomials : (poly1 * poly2) = expanded_poly := 
by
  sorry

end expand_polynomials_l199_199366


namespace birgit_hiking_time_l199_199054

def hiking_conditions
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ) : Prop :=
  ∃ (average_speed_time : ℝ) (birgit_speed_time : ℝ) (total_minutes_hiked : ℝ),
    total_minutes_hiked = hours_hiked * 60 ∧
    average_speed_time = total_minutes_hiked / distance_km ∧
    birgit_speed_time = average_speed_time - time_faster ∧
    (birgit_speed_time * distance_target_km = 48)

theorem birgit_hiking_time
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ)
  : hiking_conditions hours_hiked distance_km time_faster distance_target_km :=
by
  use 10, 6, 210
  sorry

end birgit_hiking_time_l199_199054


namespace arccos_one_half_is_pi_div_three_l199_199191

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199191


namespace roots_are_prime_then_a_is_five_l199_199805

theorem roots_are_prime_then_a_is_five (x1 x2 a : ℕ) (h_prime_x1 : Prime x1) (h_prime_x2 : Prime x2)
  (h_eq : x1 + x2 = a) (h_eq_mul : x1 * x2 = a + 1) : a = 5 :=
sorry

end roots_are_prime_then_a_is_five_l199_199805


namespace replace_star_with_2x_l199_199546

theorem replace_star_with_2x (c : ℤ) (x : ℤ) :
  (x^3 - 2)^2 + (x^2 + c)^2 = x^6 + x^4 + 4x^2 + 4 ↔ c = 2 * x :=
by
  sorry

end replace_star_with_2x_l199_199546


namespace area_of_ABCD_is_729_l199_199468

noncomputable def area_of_shaded_region
  (side_length : ℝ) (radius : ℝ) (center : ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (h1 : side_length = 10)
  (h2 : radius = 20)
  (h3 : A, B, C, D are intersections of the circle with center centered)
  : ℝ :=
  let hex_area := 3 * (sqrt 3 / 4 * radius^2) in
  let segment_area := 1 / 6 * real.pi * radius^2 in
  hex_area + segment_area

theorem area_of_ABCD_is_729
  (side_length : ℝ) (radius : ℝ) (center : ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (h1 : side_length = 10)
  (h2 : radius = 20)
  (h3 : A, B, C, D are intersections of the circle with grid lines)
  : area_of_shaded_region side_length radius center A B C D h1 h2 h3 = 729 := 
begin
  sorry
end

end area_of_ABCD_is_729_l199_199468


namespace tree_decomposition_l199_199017

theorem tree_decomposition (k : ℕ) (T : Tree) (X : Set T.vert) (F : Set (T.edge))
  (hk : k ≥ 2)
  (h_degree : ∀ v ∈ T.vert, T.degree v ≤ 3)
  (h_component_vertices : ∀ c ∈ T.components (T.remove_edges F), k ≤ |c ∩ X| ∧ |c ∩ X| ≤ 2 * k - 1 ∨ ∃ d ∈ T.components (T.remove_edges F), |d ∩ X| < k) :
  ∃ F, ∀ c ∈ T.components (T.remove_edges F), k ≤ |c ∩ X| ∧ |c ∩ X| ≤ 2 * k - 1 ∨ ∃ d ∈ T.components (T.remove_edges F), |d ∩ X| < k :=
sorry

end tree_decomposition_l199_199017


namespace jiajia_cost_theorem_spending_difference_theorem_l199_199992

variable (a b : ℕ)
variable (n95_jiajia regular_jiajia n95_qiqi regular_qiqi : ℕ)
variable (a_eq_b_plus_3 : a = b + 3)

-- Total cost of Jiajia's mask purchase
def jiajia_cost := 5 * a + 2 * b

-- Total cost of Qiqi's mask purchase
def qiqi_cost := 2 * a + 5 * b

-- Difference in spending
def spending_difference   := 3 * (a - b)

theorem jiajia_cost_theorem : jiajia_cost a b = 5 * a + 2 * b := by
  unfold jiajia_cost
  rfl

theorem spending_difference_theorem : spending_difference a b = 9 := by
  unfold spending_difference
  rw [a_eq_b_plus_3]
  linarith

end jiajia_cost_theorem_spending_difference_theorem_l199_199992


namespace sum_of_acute_angles_iff_l199_199796

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)

theorem sum_of_acute_angles_iff :
  α + β = π / 2 ↔ (sin α) ^ 4 / (cos β) ^ 2 + (cos α) ^ 4 / (sin β) ^ 2 = 1 :=
sorry

end sum_of_acute_angles_iff_l199_199796


namespace queens_in_each_subboard_l199_199926

-- Define the chessboard
def Chessboard := Fin 100 × Fin 100

-- Define what it means for queens to not attack each other
def noAttack (qs : Finset Chessboard) : Prop :=
  ∀ (q1 q2 : Chessboard), q1 ∈ qs → q2 ∈ qs → q1 ≠ q2 →
    q1.1 ≠ q2.1 ∧ q1.2 ≠ q2.2 ∧ 
    (q1.1 - q2.1 ≠ q1.2 - q2.2 ∧ q1.1 - q2.1 ≠ q2.2 - q1.2)

-- Define what it means for a 50x50 sub-board to contain a queen
def containsQueen (qs : Finset Chessboard) (start : Chessboard) : Prop :=
  ∃ q : Chessboard, q ∈ qs ∧ 
  start.1 ≤ q.1 ∧ q.1 < start.1 + 50 ∧
  start.2 ≤ q.2 ∧ q.2 < start.2 + 50

-- Define the hypothesis and statement
theorem queens_in_each_subboard (qs : Finset Chessboard) 
  (h_card : qs.card = 100)
  (h_noAttack : noAttack qs) :
  containsQueen qs (0, 0) ∧ 
  containsQueen qs (0, 50) ∧ 
  containsQueen qs (50, 0) ∧ 
  containsQueen qs (50, 50) :=
by
  sorry

end queens_in_each_subboard_l199_199926


namespace function_value_at_e_l199_199422

noncomputable def f (x : ℝ) : ℝ := 2 * x * (deriv f e) + real.log x

theorem function_value_at_e (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * (deriv f e) + real.log x) : f e = -1 :=
by
  sorry

end function_value_at_e_l199_199422


namespace incorrect_proposition_l199_199116

-- Variables and conditions
variable (p q : Prop)
variable (m x a b c : ℝ)
variable (hreal : 1 + 4 * m ≥ 0)

-- Theorem statement
theorem incorrect_proposition :
  ¬ (∀ m > 0, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) :=
sorry

end incorrect_proposition_l199_199116


namespace y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l199_199080

def line_equation (m x1 y1 x y : ℝ) : Prop :=
  y - y1 = m * (x - x1)

theorem y_intercept_of_line_with_slope_3_and_x_intercept_7_0 :
  ∃ b : ℝ, line_equation 3 7 0 0 b ∧ b = -21 :=
by
  sorry

end y_intercept_of_line_with_slope_3_and_x_intercept_7_0_l199_199080


namespace positive_difference_mean_median_l199_199463

theorem positive_difference_mean_median :
  let heights := [140, 150, 135, 180, 145, 170, 145],
      mean := (heights.sum : ℝ) / heights.length,
      median := heights.sorted !!!! 3,
      diff := abs (mean - median)
  in diff = 7 := by
  sorry

end positive_difference_mean_median_l199_199463


namespace vector_statements_l199_199838

def vector (α : Type*) := list α
variable (m : Real)
def a : vector Real := [4, 3 - m]
def b : vector Real := [1, m]

/-- The statements B and C given the vectors and conditions -/
theorem vector_statements :
  (m = 3 / 5 → (∃ k : Real, ∀ i, a !! i = k * (b !! i))) ∧
  (∃ m : Real, 45 + m^2 + 6 * m = 36) := 
sorry

end vector_statements_l199_199838


namespace increasing_interval_l199_199115

noncomputable def f (x : ℝ) : ℝ := x * real.sin x + real.cos x

theorem increasing_interval :
  ∀ x ∈ set.Ioo ((3:ℝ) * real.pi / 2) ((5:ℝ) * real.pi / 2), (x * real.cos x) > 0 :=
by
  sorry

end increasing_interval_l199_199115


namespace amount_given_to_second_set_of_families_l199_199557

theorem amount_given_to_second_set_of_families
  (total_spent : ℝ) (amount_first_set : ℝ) (amount_last_set : ℝ)
  (h_total_spent : total_spent = 900)
  (h_amount_first_set : amount_first_set = 325)
  (h_amount_last_set : amount_last_set = 315) :
  total_spent - amount_first_set - amount_last_set = 260 :=
by
  -- sorry is placed to skip the proof
  sorry

end amount_given_to_second_set_of_families_l199_199557


namespace arccos_half_eq_pi_div_three_l199_199342

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199342


namespace dragon_2023_first_reappearance_l199_199963

theorem dragon_2023_first_reappearance :
  let cycle_letters := 6
  let cycle_digits := 4
  Nat.lcm cycle_letters cycle_digits = 12 :=
by
  rfl -- since LCM of 6 and 4 directly calculates to 12

end dragon_2023_first_reappearance_l199_199963


namespace probability_heads_penny_dime_quarter_l199_199568

theorem probability_heads_penny_dime_quarter :
  let total_outcomes := 2^5
  let favorable_outcomes := 2 * 2
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 8 :=
by
  simp [total_outcomes, favorable_outcomes]
  sorry

end probability_heads_penny_dime_quarter_l199_199568


namespace length_of_first_train_solution_l199_199095

noncomputable def length_of_first_train (speed1_kmph speed2_kmph : ℝ) (length2_m time_s : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (5 / 18)
  let speed2_mps := speed2_kmph * (5 / 18)
  let relative_speed_mps := speed1_mps + speed2_mps
  let combined_length_m := relative_speed_mps * time_s
  combined_length_m - length2_m

theorem length_of_first_train_solution 
  (speed1_kmph : ℝ) 
  (speed2_kmph : ℝ) 
  (length2_m : ℝ) 
  (time_s : ℝ) 
  (h₁ : speed1_kmph = 42) 
  (h₂ : speed2_kmph = 30) 
  (h₃ : length2_m = 120) 
  (h₄ : time_s = 10.999120070394369) : 
  length_of_first_train speed1_kmph speed2_kmph length2_m time_s = 99.98 :=
by 
  sorry

end length_of_first_train_solution_l199_199095


namespace arccos_of_half_eq_pi_over_three_l199_199302

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199302


namespace vector_magnitude_problem_l199_199835

open_locale real_inner_product

noncomputable def magnitude_of_vector_sum (a b : ℝ) (angle: ℝ) (ha : |a| = 1) (hb : |b| = 2) (hangle: angle = real.pi / 3) : ℝ :=
sqrt (|a|^2 + |b|^2 + 2 * |a| * |b| * real.cos angle)

theorem vector_magnitude_problem :
  let a := 1,
      b := 2,
      angle := real.pi / 3 in
  |magnitude_of_vector_sum a b angle (by simp) (by simp) (by simp)| = sqrt 7 :=
sorry

end vector_magnitude_problem_l199_199835


namespace probability_fourth_shiny_after_five_l199_199136

def numShinyPennies := 5
def numDullPennies := 6
def totalPennies := numShinyPennies + numDullPennies

noncomputable def probabilityMoreThanFiveDrawsForFourthShiny : ℚ :=
  let totalCombinations := (nat.choose totalPennies numShinyPennies)
  let case1Probability := (nat.choose numShinyPennies 3 * nat.choose numDullPennies 2) / totalCombinations
  let case2Probability := (nat.choose numShinyPennies 2 * nat.choose numDullPennies 3) / totalCombinations
  let case3Probability := (nat.choose numShinyPennies 1 * nat.choose numDullPennies 4) / totalCombinations
  let case4Probability := (nat.choose numShinyPennies 0 * nat.choose numDullPennies 5) / totalCombinations
  let totalProbability := case1Probability + case2Probability + case3Probability + case4Probability
  totalProbability

theorem probability_fourth_shiny_after_five : 
  probabilityMoreThanFiveDrawsForFourthShiny = 431 / 462 := 
sorry

end probability_fourth_shiny_after_five_l199_199136


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199327

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199327


namespace arccos_half_eq_pi_div_three_l199_199345

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199345


namespace arccos_half_eq_pi_div_three_l199_199248

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l199_199248


namespace calculate_expression_l199_199731

theorem calculate_expression : 15 * 30 + 45 * 15 + 90 = 1215 := 
by 
  sorry

end calculate_expression_l199_199731


namespace area_of_FDBG_l199_199613

open_locale classical

theorem area_of_FDBG (A B C D E F G : Type*) 
  [is_triangle ABC]
  (AB AC : ℝ) (AB_eq_60 : AB = 60) (AC_eq_15 : AC = 15)
  (area_ABC : ℝ) (area_ABC_eq_180 : area_ABC = 180)
  (D_midpoint_AB : D = midpoint AB)
  (E_midpoint_AC : E = midpoint AC)
  (F_angle_bisector : F ∈ bisector_angle_BAC)
  (G_intersects_DE_BC : G ∈ (DE ∩ BC)) :
  area_quadrilateral FDBG = 108 := 
sorry

end area_of_FDBG_l199_199613


namespace oldest_child_age_l199_199064

theorem oldest_child_age (a b : ℕ) (avg : ℕ) (h_avg : avg = 7) (h_young1 : a = 4) (h_young2 : b = 7) :
  let oldest := 21 - (a + b) in oldest = 10 :=
by sorry

end oldest_child_age_l199_199064


namespace largest_multiple_5_6_lt_1000_is_990_l199_199106

theorem largest_multiple_5_6_lt_1000_is_990 : ∃ n, (n < 1000) ∧ (n % 5 = 0) ∧ (n % 6 = 0) ∧ n = 990 :=
by 
  -- Needs to follow the procedures to prove it step-by-step
  sorry

end largest_multiple_5_6_lt_1000_is_990_l199_199106


namespace max_distinct_acute_lattice_triangles_l199_199085

theorem max_distinct_acute_lattice_triangles (N : ℕ) :
  (∀ (T : fin N → (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)), 
    (∀ (i : fin N), is_acute (T i) ∧ same_area (T i) (2^2020) ∧ is_lattice_triangle (T i)) 
    → ∀ i j, i ≠ j → ¬ congruent (T i) (T j)) 
  ↔ N ≤ 2^2020 - 1 :=
sorry

end max_distinct_acute_lattice_triangles_l199_199085


namespace sum_of_divisors_of_252_l199_199650

theorem sum_of_divisors_of_252 :
  ∑ (d : ℕ) in (finset.filter (λ x, 252 % x = 0) (finset.range (252 + 1))), d = 728 :=
by
  sorry

end sum_of_divisors_of_252_l199_199650


namespace calculate_change_l199_199184

def price_cappuccino := 2
def price_iced_tea := 3
def price_cafe_latte := 1.5
def price_espresso := 1

def quantity_cappuccinos := 3
def quantity_iced_teas := 2
def quantity_cafe_lattes := 2
def quantity_espressos := 2

def amount_paid := 20

theorem calculate_change :
  let total_cost := (price_cappuccino * quantity_cappuccinos) + 
                    (price_iced_tea * quantity_iced_teas) + 
                    (price_cafe_latte * quantity_cafe_lattes) + 
                    (price_espresso * quantity_espressos) in
  amount_paid - total_cost = 3 :=
by
  sorry

end calculate_change_l199_199184


namespace min_max_expression_l199_199126

theorem min_max_expression (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 19) (h2 : b^2 + b * c + c^2 = 19) :
  ∃ (min_val max_val : ℝ), 
    min_val = 0 ∧ max_val = 57 ∧ 
    (∀ x, x = c^2 + c * a + a^2 → min_val ≤ x ∧ x ≤ max_val) :=
by sorry

end min_max_expression_l199_199126


namespace find_P_eq_30_l199_199129

theorem find_P_eq_30 (P Q R S : ℕ) :
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S ∧
  P * Q = 120 ∧ R * S = 120 ∧ P - Q = R + S → P = 30 :=
by
  sorry

end find_P_eq_30_l199_199129


namespace median_of_list_l199_199352

theorem median_of_list :
  let lst := (List.range 3031).map (λ x => x + 1) ++ (List.range 3031).map (λ x => (x + 1) * (x + 1)) in
  lst.length = 6060 →
  List.Median (lst.sort) = 2975.5 :=
by
  sorry

end median_of_list_l199_199352


namespace maximize_rectangle_area_l199_199982

theorem maximize_rectangle_area :
  ∀ (x y : ℝ), x + y = 6 → (∀ x y : ℝ, (x + y = 6) ∧ (x * y ≤ 9)) → x = 3 ∧ y = 3 :=
by
  intros x y h1 h2
  have : (x - 3)^2 ≤ 0 :=
    by
      cases h2 3 3 with h3 h4
      rw [h1] at h4
      linarith
  linarith

end maximize_rectangle_area_l199_199982


namespace nth_term_of_sequence_l199_199832

theorem nth_term_of_sequence (n : ℕ) (h : n > 0) : 
  (10 ^ (n - 1)) = nth_term_of_sequence n := 
sorry

end nth_term_of_sequence_l199_199832


namespace monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l199_199602

/- Part 1: Monthly Average Growth Rate -/
theorem monthly_average_growth_rate (m : ℝ) (sale_april sale_june : ℝ) (h_apr_val : sale_april = 256) (h_june_val : sale_june = 400) :
  256 * (1 + m) ^ 2 = 400 → m = 0.25 :=
sorry

/- Part 2: Optimal Selling Price for Desired Profit -/
theorem optimal_selling_price_for_desired_profit (y : ℝ) (initial_price selling_price : ℝ) (sale_june : ℝ) (h_june_sale : sale_june = 400) (profit : ℝ) (h_profit : profit = 8400) :
  (y - 35) * (1560 - 20 * y) = 8400 → y = 50 :=
sorry

end monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l199_199602


namespace max_horizontal_vertical_sum_l199_199752

theorem max_horizontal_vertical_sum (a b c d e f : ℕ) (h1 : {a, b, c, d, e, f} = {2, 5, 8, 11, 14, 17}) (h2 : b + e = c + f) (h3 : a + c = b + d) (h4 : e = 17) : 33 ≤ max (a + b + e) (a + c + f) :=
sorry

end max_horizontal_vertical_sum_l199_199752


namespace count_equilateral_triangles_l199_199913

-- Define the set T
def T : set (ℕ × ℕ × ℕ) := { p | let (x, y, z) := p in x ∈ {0, 1, 2, 3} ∧ y ∈ {0, 1, 2, 3} ∧ z ∈ {0, 1, 2, 3} ∧ (x + y + z) % 2 = 0 }

-- Define a function to calculate the distance between two points
def dist (p1 p2 : ℕ × ℕ × ℕ) : ℚ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

-- Define a property to check if three points form an equilateral triangle
def is_equilateral (p1 p2 p3 : ℕ × ℕ × ℕ) : Prop :=
  (dist p1 p2 = dist p2 p3) ∧ (dist p2 p3 = dist p3 p1)

-- Define the main theorem to count equilateral triangles
theorem count_equilateral_triangles : ∃ n, n = 70 ∧
  (∀ (p1 p2 p3 : ℕ × ℕ × ℕ), p1 ∈ T → p2 ∈ T → p3 ∈ T → is_equilateral p1 p2 p3 → n = 70) :=
begin
  -- Main proof steps are not required
  sorry
end

end count_equilateral_triangles_l199_199913


namespace max_value_fraction_l199_199828

theorem max_value_fraction (e a b : ℝ) (h : ∀ x : ℝ, (e - a) * Real.exp x + x + b + 1 ≤ 0) : 
  (b + 1) / a ≤ 1 / e :=
sorry

end max_value_fraction_l199_199828


namespace monoticity_f_max_min_f_l199_199735

noncomputable def f (x : ℝ) := Real.log (2 * x + 3) + x^2

theorem monoticity_f :
  (∀ x ∈ Ioo (-(3:ℝ)/2) (-1), f x > f (-1)) ∧ 
  (∀ x ∈ Ioo (-(1:ℝ)/2) (1/0), f x > f (1/0)) ∧ 
  (∀ x ∈ Ioo (-1) (-(1:ℝ)/2), f x < f (-1/2)) :=
sorry

theorem max_min_f :
  ∀ x ∈ Icc (0 : ℝ) (1), 
  f(1) = Real.log 5 + 1 ∧ 
  f(0) = Real.log 3 :=
sorry

end monoticity_f_max_min_f_l199_199735


namespace arccos_half_eq_pi_div_three_l199_199292

theorem arccos_half_eq_pi_div_three : real.arccos (1/2) = real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_three_l199_199292


namespace even_four_digit_count_correct_l199_199441

def num_even_four_digit_numbers : ℕ :=
  48

theorem even_four_digit_count_correct :
  ∃ n : ℕ, (n = num_even_four_digit_numbers) ∧ 
           (forall (a b c d : ℕ), a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3, 4, 5} ∧ c ∈ {1, 2, 3, 4, 5} ∧ d ∈ {1, 2, 3, 4, 5} ∧ 
           a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
           d % 2 = 0 → (a * 1000 + b * 100 + c * 10 + d) is_valid_even_number) :=
begin
  use 48,
  split,
  { rfl, },
  { intros a b c d h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₁₀ h₁₁ h₁₂,
    sorry
  }
end

end even_four_digit_count_correct_l199_199441


namespace roots_equation_value_l199_199777

theorem roots_equation_value (α β : ℝ) (h1 : α^2 - α - 1 = 0) (h2 : β^2 - β - 1 = 0) (h3 : α + β = 1) :
    α^4 + 3 * β = 5 := by
sorry

end roots_equation_value_l199_199777


namespace sum_of_divisors_of_252_l199_199651

theorem sum_of_divisors_of_252 :
  ∑ (d : ℕ) in (finset.filter (λ x, 252 % x = 0) (finset.range (252 + 1))), d = 728 :=
by
  sorry

end sum_of_divisors_of_252_l199_199651


namespace find_a100_l199_199810

noncomputable def S (k : ℝ) (n : ℤ) : ℝ := k * (n ^ 2) + n
noncomputable def a (k : ℝ) (n : ℤ) : ℝ := S k n - S k (n - 1)

theorem find_a100 (k : ℝ) 
  (h1 : a k 10 = 39) :
  a k 100 = 399 :=
sorry

end find_a100_l199_199810


namespace first_car_made_earlier_l199_199092

def year_first_car : ℕ := 1970
def year_third_car : ℕ := 2000
def diff_third_second : ℕ := 20

theorem first_car_made_earlier : (year_third_car - diff_third_second) - year_first_car = 10 := by
  sorry

end first_car_made_earlier_l199_199092


namespace projection_of_a_onto_b_l199_199436

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / magnitude v2

theorem projection_of_a_onto_b : projection vec_a vec_b = Real.sqrt 5 :=
by
  sorry

end projection_of_a_onto_b_l199_199436


namespace product_of_possible_values_of_b_l199_199587

theorem product_of_possible_values_of_b :
  let y₁ := -1
  let y₂ := 4
  let x₁ := 1
  let side_length := y₂ - y₁ -- Since this is 5 units
  let b₁ := x₁ - side_length -- This should be -4
  let b₂ := x₁ + side_length -- This should be 6
  let product := b₁ * b₂ -- So, (-4) * 6
  product = -24 :=
by
  sorry

end product_of_possible_values_of_b_l199_199587


namespace arccos_of_half_eq_pi_over_three_l199_199296

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l199_199296


namespace winner_percentage_l199_199472

theorem winner_percentage (W L V : ℕ) 
    (hW : W = 868) 
    (hDiff : W - L = 336)
    (hV : V = W + L) : 
    (W * 100 / V) = 62 := 
by 
    sorry

end winner_percentage_l199_199472


namespace determine_right_triangle_l199_199817

theorem determine_right_triangle :
  (∀ (A B C : ℝ),
    (A + B + C = 180) → (C = A - B) → (A = 90)) ∧
  (∀ (A B C : ℝ)(x : ℝ),
    (5 * x + 2 * x + 3 * x = 180) → (A = 5 * x) → (A = 90)) ∧
  (∀ (a b c : ℝ),
    (a = 3 / 5 * c) → (b = 4 / 5 * c) → (a * a + b * b = c * c)) ∧
  (∀ (a b c : ℝ),
    (a : b : c = 1 : 2 : √3) → (a * a + c * c ≠ b * b)) →
  (4 : ℕ) = 4 := sorry

end determine_right_triangle_l199_199817


namespace ravi_profit_l199_199122

theorem ravi_profit
  (CP_r CP_m : ℝ)
  (L_p P_p : ℝ)
  (H1 : CP_r = 15000)
  (H2 : CP_m = 8000)
  (H3 : L_p = 0.04)
  (H4 : P_p = 0.09) :
  let loss_r := L_p * CP_r in
  let SP_r := CP_r - loss_r in
  let profit_m := P_p * CP_m in
  let SP_m := CP_m + profit_m in
  let total_CP := CP_r + CP_m in
  let total_SP := SP_r + SP_m in
  total_SP - total_CP = 120 :=
by
  have loss_r := L_p * CP_r,
  have SP_r := CP_r - loss_r,
  have profit_m := P_p * CP_m,
  have SP_m := CP_m + profit_m,
  have total_CP := CP_r + CP_m,
  have total_SP := SP_r + SP_m,
  calc
    total_SP - total_CP = (SP_r + SP_m) - (CP_r + CP_m) : by sorry
                    ... = 120 : by sorry

end ravi_profit_l199_199122


namespace cos_pi_over_3_arccos_property_arccos_one_half_l199_199320

-- Define the known cosine value
theorem cos_pi_over_3 : Real.cos (π / 3) = 1 / 2 := sorry

-- Define the property of arccos
theorem arccos_property {x : Real} (h : 0 ≤ x ∧ x ≤ 1) : Real.cos (Real.arccos x) = x := Real.cos_arccos h

-- Formulate and state the main theorem
theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := 
by 
  have h_cos_value : Real.cos (π / 3) = 1 / 2 := cos_pi_over_3
  have h_range_condition : 0 ≤ (1 / 2) ∧ (1 / 2) ≤ 1 := by norm_num
  exact eq_of_cos_eq_right (by norm_num) h_cos_value (arccos_property h_range_condition)


end cos_pi_over_3_arccos_property_arccos_one_half_l199_199320


namespace sequences_count_n3_sequences_count_n6_sequences_count_n9_l199_199157

inductive Shape
  | triangle
  | square
  | rectangle (k : ℕ)

open Shape

def transition (s : Shape) : List Shape :=
  match s with
  | triangle => [triangle, square]
  | square => [rectangle 1]
  | rectangle k =>
    if k = 0 then [rectangle 1] else [rectangle (k - 1), rectangle (k + 1)]

def count_sequences (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (shapes : List Shape) : ℕ :=
    if m = 0 then shapes.length
    else
      let next_shapes := shapes.bind transition
      aux (m - 1) next_shapes
  aux n [square]

theorem sequences_count_n3 : count_sequences 3 = 5 :=
  by sorry

theorem sequences_count_n6 : count_sequences 6 = 24 :=
  by sorry

theorem sequences_count_n9 : count_sequences 9 = 149 :=
  by sorry

end sequences_count_n3_sequences_count_n6_sequences_count_n9_l199_199157


namespace arithmetic_sequence_sub_l199_199025

noncomputable def seq_a (p q : ℕ) (h : Nat.coprime p q) : ℕ → ℕ
| 1       => p
| (n + 1) => let prev_a := seq_a p q h n
             let prev_b := seq_b p q h n
             seq_a p q h n

noncomputable def seq_b (p q : ℕ) (h : Nat.coprime p q) : ℕ → ℕ
| 1       => q
| (n + 1) => let prev_a := seq_a p q h n
             let prev_b := seq_b p q h n
             Nat.succ_above (seq_b p q h n) (prev_b + 1)

theorem arithmetic_sequence_sub (p q : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : Nat.coprime p q) :
  ∃ d, ∀ n : ℕ, seq_a p q h3 n - seq_b p q h3 n = d * n :=
sorry

end arithmetic_sequence_sub_l199_199025


namespace speed_of_jakes_dad_second_half_l199_199487

theorem speed_of_jakes_dad_second_half :
  let distance_to_park := 22
  let total_time := 0.5
  let time_half_journey := total_time / 2
  let speed_first_half := 28
  let distance_first_half := speed_first_half * time_half_journey
  let remaining_distance := distance_to_park - distance_first_half
  let time_second_half := time_half_journey
  let speed_second_half := remaining_distance / time_second_half
  speed_second_half = 60 :=
by
  sorry

end speed_of_jakes_dad_second_half_l199_199487


namespace replace_star_with_2x_l199_199545

theorem replace_star_with_2x (c : ℤ) (x : ℤ) :
  (x^3 - 2)^2 + (x^2 + c)^2 = x^6 + x^4 + 4x^2 + 4 ↔ c = 2 * x :=
by
  sorry

end replace_star_with_2x_l199_199545


namespace find_a_b_slope_AB_area_ABCD_range_l199_199815

-- Define the ellipse and its properties
def ellipse (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle and its properties
def circle (h k r : ℝ) := ∀ x y : ℝ, (x + h)^2 + (y + k)^2 = r^2

-- Define the points M and N from the intersection of the ellipse and the circle
def points_M_N (x1 y1 x2 y2 : ℝ) := 
  (ellipse 2 1 (-1) (√3 / 2)) ∧ 
  (ellipse 2 1 (-1) (-√3 / 2)) ∧ 
  (circle (-3 / 2) 0 1 (-1) (√3 / 2)) ∧ 
  (circle (-3 / 2) 0 1 (-1) (-√3 / 2))

-- Define the slopes of lines through the center of the ellipse
def slopes (k1 k2 : ℝ) := k1 * k2 = 1 / 4

-- Final statements
theorem find_a_b (h : ellipse 2 1 ∧ points_M_N (-1) (√3 / 2) (-1) (-√3 / 2)) : 2 = 2 ∧ 1 = 1 := by 
  sorry

theorem slope_AB (h : ellipse 2 1 ∧ slopes 1/2 (-2)) : k = ±(1 / 2) := by 
  sorry

theorem area_ABCD_range (h : ellipse 2 1 ∧ slopes 1/2 (-2)) : S ∈ (0, 4) := by 
  sorry

end find_a_b_slope_AB_area_ABCD_range_l199_199815


namespace min_residents_own_all_appliances_l199_199870

-- Define the total population of residents in Eclectopia
def total_population : ℕ := 75000

-- Define the ownership percentages of each appliance
def percentage_refrigerator : ℝ := 0.75
def percentage_television : ℝ := 0.90
def percentage_computer : ℝ := 0.85
def percentage_air_conditioner : ℝ := 0.75
def percentage_microwave : ℝ := 0.95
def percentage_washing_machine : ℝ := 0.80
def percentage_dishwasher : ℝ := 0.70

-- Define the minimum number of residents who own all appliances
def min_own_all_appliances := (percentage_dishwasher * total_population).toInt

-- Prove that the minimum number of residents who own all the mentioned appliances is 52500
theorem min_residents_own_all_appliances : min_own_all_appliances = 52500 := by
  sorry

end min_residents_own_all_appliances_l199_199870


namespace oil_level_drop_l199_199146

noncomputable def stationary_tank_radius : ℝ := 100
noncomputable def stationary_tank_height : ℝ := 25
noncomputable def truck_tank_radius : ℝ := 7
noncomputable def truck_tank_height : ℝ := 10

noncomputable def π : ℝ := Real.pi
noncomputable def truck_tank_volume := π * truck_tank_radius^2 * truck_tank_height
noncomputable def stationary_tank_area := π * stationary_tank_radius^2

theorem oil_level_drop (volume_truck: ℝ) (area_stationary: ℝ) : volume_truck = 490 * π → area_stationary = π * 10000 → (volume_truck / area_stationary) = 0.049 :=
by
  intros h1 h2
  sorry

end oil_level_drop_l199_199146


namespace Mcdonald_needs_hens_l199_199521

noncomputable def eggs_needed_per_week (Saly Ben Ked Rachel Tom : ℕ) : ℕ :=
  Saly + Ben + Ked + Rachel + Tom

noncomputable def eggs_needed_per_month_with_increase (weekly_eggs : ℕ) : ℕ :=
  let week1 := weekly_eggs
  let week2 := weekly_eggs * 105 / 100
  let week3 := week2 * 105 / 100
  let week4 := week3 * 105 / 100
  nat.ceil (week1 + week2 + week3 + week4)

noncomputable def hens_needed (total_eggs : ℕ) (eggs_per_hen_per_week : ℕ) (weeks : ℕ) : ℕ :=
  let eggs_per_hen_per_month := eggs_per_hen_per_week * weeks
  nat.ceil (total_eggs / eggs_per_hen_per_month)

-- Define the given conditions
def Saly_needs := 10
def Ben_needs := 14
def Ked_needs := Ben_needs / 2
def Rachel_needs := Saly_needs * 3 / 2
def Tom_needs := 24
def weekly_increase := 5
def eggs_per_hen_per_week := 5
def weeks := 4

-- Calculate the weekly and monthly egg requirements
def weekly_eggs := eggs_needed_per_week Saly_needs Ben_needs Ked_needs Rachel_needs Tom_needs
def monthly_eggs := eggs_needed_per_month_with_increase weekly_eggs

-- Calculate the number of hens needed to meet the monthly egg demand
theorem Mcdonald_needs_hens : hens_needed monthly_eggs eggs_per_hen_per_week weeks = 16 := by
  sorry

end Mcdonald_needs_hens_l199_199521


namespace minimum_sides_of_polygon_l199_199994

theorem minimum_sides_of_polygon (θ : ℝ) (hθ : θ = 25.5) : ∃ n : ℕ, n = 240 ∧ ∀ k : ℕ, (k * θ) % 360 = 0 → k = n := 
by
  -- The proof goes here
  sorry

end minimum_sides_of_polygon_l199_199994


namespace arccos_one_half_is_pi_div_three_l199_199199

noncomputable def arccos_one_half_eq_pi_div_three : Prop :=
  arccos (1/2) = (π / 3)

theorem arccos_one_half_is_pi_div_three : arccos_one_half_eq_pi_div_three :=
by
  sorry

end arccos_one_half_is_pi_div_three_l199_199199


namespace arccos_half_eq_pi_over_three_l199_199212

theorem arccos_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi / 3 :=
by
  sorry

end arccos_half_eq_pi_over_three_l199_199212


namespace mary_needs_more_flour_l199_199520

theorem mary_needs_more_flour (total_flour added_flour : ℕ) 
  (h1 : total_flour = 10) 
  (h2 : added_flour = 6) : 
  total_flour - added_flour = 4 := 
by 
  rw [h1, h2]; 
  norm_num

end mary_needs_more_flour_l199_199520


namespace arccos_half_eq_pi_div_three_l199_199343

theorem arccos_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 := 
  sorry

end arccos_half_eq_pi_div_three_l199_199343
