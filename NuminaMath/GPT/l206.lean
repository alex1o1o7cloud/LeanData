import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Quad.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.MeasureTheory.Integral.Fubini
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Cycle.Type
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.GCD
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector
import Mathlib.Geometry.Circle.Incircle
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.ThreeDimensional
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import data.nat.prime

namespace average_speed_round_trip_l206_206020

noncomputable def average_speed (distance : ℝ) (speed_upstream : ℝ) (speed_downstream : ℝ) : ℝ :=
  let time_upstream := distance / speed_upstream
  let time_downstream := distance / speed_downstream
  let total_distance := 2 * distance
  let total_time := time_upstream + time_downstream
  total_distance / total_time

theorem average_speed_round_trip :
  ∀ (D : ℝ), D > 0 → 
  average_speed D 4 7 = 56 / 11 := 
by 
  intro D hD
  simp [average_speed]
  have t_up := D / 4
  have t_down := D / 7
  have total_time := t_up + t_down
  have total_distance := 2 * D
  calc
    average_speed D 4 7 = total_distance / total_time : by simp [average_speed]
                     ... = 2 * D / (D / 4 + D / 7) : by simp
                     ... = 2 * D / (7 * D / 28 + 4 * D / 28) : by simp [←div_eq_mul_one_div]
                     ... = 2 * D / (11 * D / 28) : by simp [*, add_div]
                     ... = 2 * 28 / 11 : by simp [←mul_div_cancel_left 11]
                     ... = 56 / 11 : by ring

end average_speed_round_trip_l206_206020


namespace women_ratio_l206_206178

theorem women_ratio (pop : ℕ) (w_retail : ℕ) (w_fraction : ℚ) (h_pop : pop = 6000000) (h_w_retail : w_retail = 1000000) (h_w_fraction : w_fraction = 1 / 3) : 
  (3000000 : ℚ) / (6000000 : ℚ) = 1 / 2 :=
by sorry

end women_ratio_l206_206178


namespace distance_focus_directrix_of_parabola_l206_206857

noncomputable def parabola_distance_focus_directrix (x y : ℝ) : ℝ :=
  if h : x^2 = (1/2) * y then
    let p := 1 / 4 in p
  else
    0

theorem distance_focus_directrix_of_parabola : parabola_distance_focus_directrix x y = 1 / 4 := by
  rw parabola_distance_focus_directrix
  split_ifs
  exact rfl
  sorry

end distance_focus_directrix_of_parabola_l206_206857


namespace average_marks_first_class_l206_206455

theorem average_marks_first_class (A : ℝ) :
  let students_class1 := 55
  let students_class2 := 48
  let avg_class2 := 58
  let avg_all := 59.067961165048544
  let total_students := 103
  let total_marks := avg_all * total_students
  total_marks = (A * students_class1) + (avg_class2 * students_class2) 
  → A = 60 :=
by
  sorry

end average_marks_first_class_l206_206455


namespace dice_pair_not_three_of_a_kind_probability_l206_206071

/-- Each of six standard six-sided dice is rolled once. 
  Prove that the probability that there is at least one pair but not a three-of-a-kind
  (that is, there are either two dice showing the same value, or three pairs are showing,
  but no three dice show the same value) is 25/81. -/
theorem dice_pair_not_three_of_a_kind_probability :
  let total_outcomes := (6:ℕ)^6
  let successful_outcomes := 
    6 * 15 * 120 +      -- one pair and the others different
    15 * 15 * 72 +      -- two different pairs and the others different
    20 * 90            -- three different pairs
  (successful_outcomes : ℕ) / (total_outcomes : ℤ) = 25 / 81 := 
by
  sorry

end dice_pair_not_three_of_a_kind_probability_l206_206071


namespace find_m_l206_206949

def A := {x : ℝ | x^2 + x - 6 = 0}
def B (m : ℝ) := {x : ℝ | m * x + 1 = 0}

theorem find_m (m : ℝ) : B(m) ⊆ A ↔ m ∈ {-1/2, 0, 1/3} := sorry

end find_m_l206_206949


namespace single_point_graph_l206_206318

theorem single_point_graph (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 8 * y + d = 0 → x = -1 ∧ y = 4) → d = 19 :=
by
  sorry

end single_point_graph_l206_206318


namespace helen_made_56_pies_l206_206633

theorem helen_made_56_pies (pinky_pies total_pies : ℕ) (h_pinky : pinky_pies = 147) (h_total : total_pies = 203) :
  (total_pies - pinky_pies) = 56 :=
by
  sorry

end helen_made_56_pies_l206_206633


namespace triangle_area_l206_206563

theorem triangle_area (A B C : Type) [Triangle A B C] (angle_B : Angle) (len_AC : Real) (len_AB : Real) 
  (h1 : angle_B = 30 * Real.pi / 180)
  (h2 : len_AC = 1)
  (h3 : len_AB = Real.sqrt 3):
  (area_of_triangle A B C = (Real.sqrt 3) / 2) ∨ (area_of_triangle A B C = (Real.sqrt 3) / 4) :=
by
  sorry

end triangle_area_l206_206563


namespace prove_a_gt_0_prove_a_plus_b_plus_c_gt_0_l206_206510

variable {a b c : ℝ}

-- Definition of the conditions
def solution_set_parabola (a b c : ℝ) : set ℝ :=
  {x | ax^2 + bx + c ≥ 0}

def condition_solution_set : Prop := 
  solution_set_parabola a b c = {x | x ≤ 3 ∨ x ≥ 4}

-- Statements to prove based on the given conditions
theorem prove_a_gt_0 (h : condition_solution_set) : a > 0 := sorry

theorem prove_a_plus_b_plus_c_gt_0 (h : condition_solution_set) : a + b + c > 0 := sorry

end prove_a_gt_0_prove_a_plus_b_plus_c_gt_0_l206_206510


namespace number_of_valid_integers_l206_206159

-- Definitions of the two-digit numbers
def ab (a b : ℕ) : ℕ := 10 * a + b
def bc (b c : ℕ) : ℕ := 10 * b + c
def cd (c d : ℕ) : ℕ := 10 * c + d

-- Conditions for \( ab \), \( bc \), \( cd \)
def decreasing_arithmetic (a b c d : ℕ) : Prop :=
  ab a b > bc b c ∧ bc b c > cd c d ∧ 10 * (a - 2 * b + c) = b - 2 * c + d

-- Additional constraints
def valid_digits (a b c d : ℕ) : Prop :=
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a ≠ 0

-- Proof statement
theorem number_of_valid_integers : 
  (finset.card (finset.filter 
    (λ (abcd : ℕ × ℕ × ℕ × ℕ), 
      let (a, b, c, d) := abcd in decreasing_arithmetic a b c d ∧ valid_digits a b c d)
    (finset.cross (finset.range 10) (finset.range 10) (finset.range 10) (finset.range 10)))) = 6 :=
sorry

end number_of_valid_integers_l206_206159


namespace remainder_when_squared_expression_divided_by_40_l206_206164

theorem remainder_when_squared_expression_divided_by_40
  (k : ℤ) : ((40 * k - 1)^2 - 3 * (40 * k - 1) + 5) % 40 = 9 := 
by {
  -- calculation steps are omitted
  sorry
}

end remainder_when_squared_expression_divided_by_40_l206_206164


namespace meaningful_sqrt_condition_correct_option_l206_206733

theorem meaningful_sqrt_condition (x : ℝ) : 
  (\sqrt{x - 2}).isDefined ↔ x ≥ 2 :=
by sorry

theorem correct_option : 
  (2 ≥ 2) ∧ (-3 < 2) ∧ (0 < 2) ∧ (1 < 2) := 
by sorry

end meaningful_sqrt_condition_correct_option_l206_206733


namespace not_divide_660_into_31_not_divide_660_into_31_l206_206236

theorem not_divide_660_into_31 (H : ∀ (A B : ℕ), A ∈ S ∧ B ∈ S → A < 2 * B) : 
  false :=
begin
  -- math proof steps should go here
  sorry
end

namespace my_proof

def similar (x y : ℕ) : Prop := x < 2 * y

theorem not_divide_660_into_31 : ¬ ∃ (S : set ℕ), (∀ x ∈ S, x < 2 * y) ∧ (∑ x in S, x = 660) is ∧ |S| = 31 :=
begin
  sorry
end

end my_proof

end not_divide_660_into_31_not_divide_660_into_31_l206_206236


namespace problem1_problem2_l206_206564

noncomputable def triangle_ABC_conditions_one (A B C a b c : Real) :=
  a = c * Real.sin B + b * Real.cos C

noncomputable def triangle_ABC_conditions_two (A B C a b c : Real) :=
  A + C = 3 * Real.pi / 4 ∧ b = Real.sqrt 2 ∧
  a = c * Real.sin B + b * Real.cos C

theorem problem1 (A B C a b c : Real) (h : triangle_ABC_conditions_one A B C a b c) :
  A + C = 3 * Real.pi / 4 :=
sorry

theorem problem2 (A B C a b c : Real) (h : triangle_ABC_conditions_two A B C a b c) :
  let area := (Real.sqrt 2 / 4) * (a * c) in
  area ≤ (1 + Real.sqrt 2) / 2 :=
sorry

end problem1_problem2_l206_206564


namespace distance_between_symmetric_points_l206_206578

def point := (ℝ × ℝ × ℝ)

def symmetric_with_respect_to_y (p : point) : point :=
  (-p.1, p.2, -p.3) 

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem distance_between_symmetric_points :
  let A := symmetric_with_respect_to_y (1, 2, -3)
  ∃ d, d = 4 * real.sqrt 2 ∧ d = distance A (-1, -2, -1) :=
by
  let A := symmetric_with_respect_to_y (1, 2, -3)
  exists 4 * real.sqrt 2
  split
  · refl
  · sorry

end distance_between_symmetric_points_l206_206578


namespace question_I_question_II_l206_206515

-- Definitions based on given conditions:
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x else if 1 ≤ x then 1 / x else 0

def g (a : ℝ) (x : ℝ) : ℝ :=
  a * f x - abs (x - 1)

-- Problem (I) for a = 0 and any x ∈ (0, +∞):
theorem question_I (b : ℝ) : (∀ x : ℝ, 0 < x → g 0 x ≤ abs (x - 2) + b) ↔ b ≥ -1 := by
  sorry

-- Problem (II) for a = 1 and the maximum value of g(x):
theorem question_II : ∃ x : ℝ, g 1 x = 1 ∧ (∀ y : ℝ, g 1 y ≤ 1) := by
  sorry

end question_I_question_II_l206_206515


namespace initial_oranges_l206_206622

theorem initial_oranges (x : ℝ) :
  (x / 8 + 1 / 8 = x / 4 - 3 / 4) → x = 7 := 
begin
  intro h,
  have hx : x / 8 + 1 / 8 = x / 4 - 3 / 4, from h,
  -- Simplifying the equation to solve for x
  sorry
end

end initial_oranges_l206_206622


namespace binomial_param_exact_l206_206401

variable (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ)

-- Define the conditions: expectation and variance
axiom expectation_eq : n * p = 3
axiom variance_eq : n * p * (1 - p) = 2

-- Statement to prove
theorem binomial_param_exact (h1 : n * p = 3) (h2 : n * p * (1 - p) = 2) : p = 1 / 3 :=
by
  rw [expectation_eq] at h2
  sorry

end binomial_param_exact_l206_206401


namespace find_a_l206_206692

open Real

-- Definition of regression line
def regression_line (x : ℝ) : ℝ := 12.6 * x + 0.6

-- Data points for x and y
def x_values : List ℝ := [2, 3, 3.5, 4.5, 7]
def y_values : List ℝ := [26, 38, 43, 60]

-- Proof statement
theorem find_a (a : ℝ) (hx : x_values = [2, 3, 3.5, 4.5, 7])
  (hy : y_values ++ [a] = [26, 38, 43, 60, a]) : a = 88 :=
  sorry

end find_a_l206_206692


namespace problem_statement_l206_206420

variable {R : Type*} [comm_ring R]

def is_odd_function (f : R → R) : Prop :=
∀ x : R, f (-x) = -f x

theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_periodicity : ∀ x : ℝ, f (x + 2) = -f x) :
  f 2008 + f 2009 + f 2010 + f 2011 = 0 :=
sorry

end problem_statement_l206_206420


namespace integral_f_l206_206884

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Ico (-1 : ℝ) 1 then sqrt (1 - x^2)
else if x ∈ Icc (1 : ℝ) 2 then x^2 - 1
else 0

theorem integral_f :
  ∫ x in -1..2, f x = real.pi / 2 + 4 / 3 :=
by
  sorry

end integral_f_l206_206884


namespace symmetric_polynomial_identity_l206_206844

variable (x y z : ℝ)
def σ1 : ℝ := x + y + z
def σ2 : ℝ := x * y + y * z + z * x
def σ3 : ℝ := x * y * z

theorem symmetric_polynomial_identity : 
  x^3 + y^3 + z^3 = σ1 x y z ^ 3 - 3 * σ1 x y z * σ2 x y z + 3 * σ3 x y z := by
  sorry

end symmetric_polynomial_identity_l206_206844


namespace isosceles_triangle_leg_l206_206171

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ a = c ∨ b = c)

theorem isosceles_triangle_leg
  (a b c : ℝ)
  (h1 : is_isosceles_triangle a b c)
  (h2 : a + b + c = 18)
  (h3 : a = 8 ∨ b = 8 ∨ c = 8) :
  (a = 5 ∨ b = 5 ∨ c = 5 ∨ a = 8 ∨ b = 8 ∨ c = 8) :=
sorry

end isosceles_triangle_leg_l206_206171


namespace proof1_proof2_l206_206909

open Real

noncomputable def problem1 (a b c : ℝ) (A : ℝ) (S : ℝ) :=
  ∃ (a b : ℝ), A = π / 3 ∧ c = 2 ∧ S = sqrt 3 / 2 ∧ S = 1/2 * b * 2 * sin (π / 3) ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos (π / 3) ∧ b = 1 ∧ a = sqrt 3

noncomputable def problem2 (a b c : ℝ) (A B : ℝ) :=
  c = a * cos B ∧ (a + b + c) * (a + b - c) = (2 + sqrt 2) * a * b ∧ 
  B = π / 4 ∧ A = π / 2 → 
  ∃ C, C = π / 4 ∧ C = B

theorem proof1 : problem1 (sqrt 3) 1 2 (π / 3) (sqrt 3 / 2) :=
by
  sorry

theorem proof2 : problem2 (sqrt 3) 1 2 (π / 2) (π / 4) :=
by
  sorry

end proof1_proof2_l206_206909


namespace inradius_exradius_relation_l206_206201

variable {A B C M : Type} (triangle_ABC : Triangle A B C)

-- Define the inradii
def inradius_ABC (t : Triangle A B C) : ℝ := sorry
def inradius_AMC (t : Triangle A M C) : ℝ := sorry
def inradius_BMC (t : Triangle B M C) : ℝ := sorry

-- Define the exradii
def exradius_ABC (t : Triangle A B C) : ℝ := sorry
def exradius_AMC (t : Triangle A M C) : ℝ := sorry
def exradius_BMC (t : Triangle B M C) : ℝ := sorry

-- Definitions for the inradii and exradii specific to this problem
def r  : ℝ := inradius_ABC triangle_ABC
def r1 : ℝ := inradius_AMC triangle_ABC
def r2 : ℝ := inradius_BMC triangle_ABC

def q  : ℝ := exradius_ABC triangle_ABC
def q1 : ℝ := exradius_AMC triangle_ABC
def q2 : ℝ := exradius_BMC triangle_ABC

theorem inradius_exradius_relation :
  r * r2 / (q1 * q2) = r / q :=
sorry

end inradius_exradius_relation_l206_206201


namespace count_divisibles_in_range_l206_206540

theorem count_divisibles_in_range :
  let count_div := λ (a b c d : ℕ), 
    (d / (Nat.lcm a (Nat.lcm b c))) 
  in count_div 2 3 5 299 = 9 :=
by
  -- Define the conditions
  let a := 2
  let b := 3
  let c := 5
  let d := 299
  -- Calculate the LCM of a, b, c
  let lcm := Nat.lcm a (Nat.lcm b c)
  -- Calculate the number of multiples of lcm less than d
  have count_div : ℕ := d / lcm
  -- Assert the result
  show count_div = 9 from sorry

end count_divisibles_in_range_l206_206540


namespace abs_diff_twenty_first_term_l206_206354

theorem abs_diff_twenty_first_term :
  let a_n := λ n : ℕ, 50 + 15 * (n - 1)
  let b_n := λ n : ℕ, 50 - 15 * (n - 1)
  |a_n 21 - b_n 21| = 600 :=
by {
  let a_n := λ n : ℕ, 50 + 15 * (n - 1)
  let b_n := λ n : ℕ, 50 - 15 * (n - 1)
  have h1 : a_n 21 = 50 + 15 * 20 := sorry,
  have h2 : b_n 21 = 50 - 15 * 20 := sorry,
  have h3 : a_n 21 - b_n 21 = 600 := sorry,
  show |a_n 21 - b_n 21| = 600, by rw h3,
  sorry
}

end abs_diff_twenty_first_term_l206_206354


namespace ratio_of_areas_l206_206607

noncomputable def area_of_simplex (vertices : fin 3 → ℝ^3) : ℝ :=
  1/2 -- Here, a function to compute the area of a simplex given its vertices can be defined as needed

def in_support (p : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p in
  let (a_x, a_y, a_z) := a in
  (x ≥ a_x ∧ y < a_y ∧ z < a_z) ∨
  (y ≥ a_y ∧ x < a_x ∧ z < a_z) ∨
  (z ≥ a_z ∧ x < a_x ∧ y < a_y)

def T := {p : ℝ × ℝ × ℝ | 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ 0 ≤ p.3 ∧ p.1 + p.2 + p.3 = 1}

def S : set (ℝ × ℝ × ℝ) :=
  {p ∈ T | in_support p (1/3, 1/4, 1/5)}

theorem ratio_of_areas : ∃ (a b : ℝ), (a = area_of_simplex T) ∧ (b = area_of_simplex S) ∧ (b / a = 1/4) :=
begin
  sorry
end

end ratio_of_areas_l206_206607


namespace general_equation_of_C_distance_function_and_trajectory_passing_origin_l206_206116

-- Define the parameterization of the curve C
def curve_C (t: ℝ) : ℝ × ℝ := (2 * Real.cos t, 2 * Real.sin t)

-- Points P and Q with given parameters alpha and 2*alpha
def P (α : ℝ) : ℝ × ℝ := curve_C α
def Q (α : ℝ) : ℝ × ℝ := curve_C (2 * α)

-- Midpoint M of P and Q
def M (α : ℝ) : ℝ × ℝ :=
  let (px, py) := P α
  let (qx, qy) := Q α
  ((px + qx) / 2, (py + qy) / 2)

-- Distance from M to the origin
def distance_to_origin (pt : ℝ × ℝ) : ℝ :=
  let (x, y) := pt
  Real.sqrt (x * x + y * y)

-- Statements to prove
theorem general_equation_of_C :
  ∀ t : ℝ, (0 < t ∧ t < 2 * Real.pi) → (let (x, y) := curve_C t in x^2 + y^2 = 4) := sorry

theorem distance_function_and_trajectory_passing_origin (α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) :
  distance_to_origin (M α) = Real.sqrt (2 + 2 * Real.cos α) ∧ M α = (0,0) → α = Real.pi := sorry

end general_equation_of_C_distance_function_and_trajectory_passing_origin_l206_206116


namespace construct_triangle_given_side_and_medians_l206_206737

theorem construct_triangle_given_side_and_medians
  (AB : ℝ) (m_a m_b : ℝ)
  (h1 : AB > 0) (h2 : m_a > 0) (h3 : m_b > 0) :
  ∃ (A B C : ℝ × ℝ),
    (∃ G : ℝ × ℝ, 
      dist A B = AB ∧ 
      dist A G = (2 / 3) * m_a ∧
      dist B G = (2 / 3) * m_b ∧ 
      dist G (midpoint ℝ A C) = m_b / 3 ∧ 
      dist G (midpoint ℝ B C) = m_a / 3) :=
sorry

end construct_triangle_given_side_and_medians_l206_206737


namespace strips_cannot_cover_board_l206_206472

def board_is_covered (board_size : ℕ) (remaining_squares : ℕ) (num_strips : ℕ) (strip_length : ℕ) (strip_width : ℕ) : Prop :=
  ∀(strip: ℕ → ℕ) (x : ℕ → ℕ) (y: ℕ → ℕ),  -- The precise formulation would depend on representations of moving strips around a board.
    remaining_squares ≠ num_strips * strip_length ∨ 
    board_size != 11 ∨
    strip_length != 8 ∨
    strip_width != 1

theorem strips_cannot_cover_board:
  let board := 11
  let central_square_removed := 11 * 11 - 1
  let num_strips := 15
  let strip_length := 8
  let strip_width := 1 in
  ¬ board_is_covered board central_square_removed num_strips strip_length strip_width :=
begin
  sorry  -- Proof goes here
end

end strips_cannot_cover_board_l206_206472


namespace find_square_number_divisible_by_three_between_90_and_150_l206_206454

theorem find_square_number_divisible_by_three_between_90_and_150 :
  ∃ x : ℕ, 90 < x ∧ x < 150 ∧ ∃ y : ℕ, x = y * y ∧ 3 ∣ x ∧ x = 144 := 
by 
  sorry

end find_square_number_divisible_by_three_between_90_and_150_l206_206454


namespace _l206_206873

noncomputable theorem solve_cubic : ∀ (x : ℝ), 27 * (x + 1)^3 = -64 → x = -7 / 3 :=
by
  intro x
  intro h
  -- we skip the proof with sorry
  sorry

noncomputable theorem solve_quadratic : ∀ (x : ℝ), (x + 1)^2 = 25 → x = 4 ∨ x = -6 :=
by
  intro x
  intro h
  -- we skip the proof with sorry
  sorry

end _l206_206873


namespace part1_part2_l206_206200

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def cond1 : Prop := c * Real.cos B + (Real.sqrt 3 / 3) * b * Real.sin C - a = 0
def cond2 : Prop := c = 3
def cond3 : Prop := (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4

-- First part: Proving the measure of angle C
theorem part1 (h : cond1) : C = Real.pi / 3 :=
sorry

-- Second part: Proving the value of a + b
theorem part2 (hC : C = Real.pi / 3) (hc : cond2) (harea : cond3) : a + b = 3 * Real.sqrt 2 :=
sorry

end part1_part2_l206_206200


namespace triangle_perimeter_l206_206440

-- Define points and distances specific to the problem
def Point := ℝ × ℝ

variable {P Q R S T U A B C : Point}

-- Define the radius and tangency conditions
def radius (p : Point) (r : ℝ) := ∀ q : Point, dist p q = r

def tangent (p q : Point) :=
  ∃ r : ℝ, radius p r ∧ radius q r ∧ dist p q = 2 * r

def isosceles_right_triangle (A B C : Point) :=
  dist A B = dist A C ∧ ∠ BAC = π / 2

-- Assume six circles are arranged as described
def circle_conditions := tangent P Q ∧ tangent Q R ∧ tangent R P ∧
                         tangent S T ∧ tangent U A ∧ tangent U C ∧
                         isosceles_right_triangle A B C

-- Define the problem of finding the perimeter of the triangle
def perimeter (A B C : Point) : ℝ := 
  dist A B + dist B C + dist A C

theorem triangle_perimeter (h : circle_conditions) :
  perimeter A B C = 8 + 4 * Real.sqrt 2 :=
sorry

end triangle_perimeter_l206_206440


namespace helen_oranges_l206_206533

def initial_oranges := 9
def oranges_from_ann := 29
def oranges_taken_away := 14

def final_oranges (initial : Nat) (add : Nat) (taken : Nat) : Nat :=
  initial + add - taken

theorem helen_oranges :
  final_oranges initial_oranges oranges_from_ann oranges_taken_away = 24 :=
by
  sorry

end helen_oranges_l206_206533


namespace trigonometric_equivalent_x_l206_206889

-- Define the condition
def trigonometric_condition (x : ℝ) : Prop :=
  sin x * sin (2 * x) * sin (3 * x) + cos x * cos (2 * x) * cos (3 * x) = 1

-- Define the final proof problems
theorem trigonometric_equivalent_x (x : ℝ) :
  (∃ k : ℤ, x = k * Real.pi) ↔ trigonometric_condition x :=
by
  sorry

end trigonometric_equivalent_x_l206_206889


namespace marts_income_percentage_of_juans_l206_206761

variable (T J M : Real)
variable (h1 : M = 1.60 * T)
variable (h2 : T = 0.40 * J)

theorem marts_income_percentage_of_juans : M = 0.64 * J :=
by
  sorry

end marts_income_percentage_of_juans_l206_206761


namespace infinite_two_i_l206_206768

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 > 10 ∧ ∀ n > 1, a n = a (n - 1) + gcd n (a (n - 1))

theorem infinite_two_i (a : ℕ → ℕ) (h_seq : sequence a) :
  ∃∞ i, a i = 2 * i :=
sorry

end infinite_two_i_l206_206768


namespace impossibility_of_dividing_stones_l206_206276

theorem impossibility_of_dividing_stones (stones : ℕ) (heaps : ℕ) (m : ℕ) :
  stones = 660 → heaps = 31 → (∀ (x y : ℕ), x ∈ heaps → y ∈ heaps → x < 2 * y → False) := sorry

end impossibility_of_dividing_stones_l206_206276


namespace find_xyz_l206_206230

theorem find_xyz (x y z : ℝ) (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 8/3) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  xyz = (11 + Real.sqrt 117) / 2 :=
begin
  sorry
end

end find_xyz_l206_206230


namespace impossible_division_l206_206256

theorem impossible_division (heap_size : ℕ) (heap_count : ℕ) (similar_sizes : Π (a b : ℕ), a < 2 * b) :
  ¬ (∃ (heaps : Finset ℕ), heaps.card = heap_count ∧ heaps.sum = heap_size ∧ ∀ (a b ∈ heaps), similar_sizes a b) :=
by
  let heap_size := 660
  let heap_count := 31
  sorry

end impossible_division_l206_206256


namespace option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l206_206132

noncomputable def C (n : ℕ) : Set (ℝ × ℝ) :=
  {p | let (x, y) := p in x^2 - 2 * n * x + y^2 = 0}

noncomputable def tangent_point (n : ℕ) (k_n : ℝ) (x_n y_n : ℝ) :=
  y_n = k_n * (x_n + 1) ∧ (x_n, y_n) ∈ C n

noncomputable def x_n (n : ℕ) := n / (n + 1)

noncomputable def y_n (n : ℕ) := n * Real.sqrt (2 * n + 1) / (n + 1)

noncomputable def T (n : ℕ) := ∑ i in Finset.range n, (y_n i)^2 / i^4

theorem option_A_correct (n : ℕ) : x_n n = n / (n + 1) := sorry

theorem option_B_correct (n : ℕ) : 
  T n = (n^2 + 2 * n) / (n + 1)^2 := sorry

theorem option_C_correct (n : ℕ) : 
  ∏ i in Finset.range n, x_n (2 * i) < Real.sqrt (x_n n / n) := sorry

theorem option_D_incorrect (n : ℕ) : 
  ¬ (Real.log (x_n n) - Real.log (y_n n) > 2 * (x_n n - y_n n) / (x_n n + y_n n)) := sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_incorrect_l206_206132


namespace conor_vegetables_per_week_l206_206058

theorem conor_vegetables_per_week : 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := 
by 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  show (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 
  sorry

end conor_vegetables_per_week_l206_206058


namespace max_value_of_f_l206_206864

noncomputable def f (x : Real) : Real :=
  9 * Real.sin x + 12 * Real.cos x

theorem max_value_of_f : ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) := 
  ∃ M, M = 15 ∧ (∀ x, f x ≤ 15) ∧ (f (Real.atan (4/3)) = 15) :=
sorry

end max_value_of_f_l206_206864


namespace positive_factors_count_l206_206973

theorem positive_factors_count (b n : ℕ) (hb : b = 14) (hn : n = 15) (hbn : b ≤ 15 ∧ n ≤ 15) : 
  (∀ (p q : ℕ) (heq : b = p * q) (h2 : nat.prime p) (h7 : nat.prime q) (pow_eq : b^n = (p^n) * (q^n)),
   let e1 := n
   let e2 := n
   in (e1 + 1) * (e2 + 1) = 256) := 
by
  sorry

end positive_factors_count_l206_206973


namespace prove_problem_statement_l206_206051

noncomputable def problem_statement : Prop :=
  let ε1 := Complex.exp (2 * Real.pi * Complex.I / 7) in
  (ε1^7 = 1) ∧
  (2 * Complex.cos (2 * Real.pi / 7) = ε1 + ε1⁻¹) ∧
  (2 * Complex.cos (4 * Real.pi / 7) = ε1^2 + ε1⁻¹^2) ∧
  (2 * Complex.cos (6 * Real.pi / 7) = ε1^3 + ε1⁻¹^3) ∧
  ((1 / (2 * Complex.cos (2 * Real.pi / 7)) +
    1 / (2 * Complex.cos (4 * Real.pi / 7)) +
    1 / (2 * Complex.cos (6 * Real.pi / 7))) = -2)

theorem prove_problem_statement : problem_statement :=
  by
  sorry

end prove_problem_statement_l206_206051


namespace value_log_power_function_l206_206330

noncomputable def powerFunctionPassingThroughPoint (a : ℝ) (f : ℝ → ℝ) : Prop :=
  f (1/2) = (1/2) ^ a ∧ f (1/2) = sqrt 2 / 2

theorem value_log_power_function (a : ℝ) (f : ℝ → ℝ) (h : powerFunctionPassingThroughPoint a f) :
  1 + log a (f 4) = 0 :=
  sorry

end value_log_power_function_l206_206330


namespace minimum_value_of_vector_expression_l206_206499

variables {ℝ: Type*} [has_inner ℝ] [normed_space ℝ ℝ]

variables (a b c : ℝ → ℝ) (t1 t2 : ℝ)

-- Conditions as per given problem
def is_unit_vector (v : ℝ → ℝ) : Prop := ∥v∥ = 1
def is_perpendicular (v w : ℝ → ℝ) : Prop := inner v w = 0

axiom a_unit_vector : is_unit_vector a
axiom b_unit_vector : is_unit_vector b
axiom a_perpendicular_b : is_perpendicular a b
axiom c_norm : ∥c∥ = 13
axiom c_dot_a : inner c a = 3
axiom c_dot_b : inner c b = 4

-- The theorem to state the problem
theorem minimum_value_of_vector_expression :
  ∃ t1 t2 : ℝ, ∥c - t1 • a - t2 • b∥ = 12 :=
sorry

end minimum_value_of_vector_expression_l206_206499


namespace rubies_in_treasure_l206_206803

theorem rubies_in_treasure (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) : 
  total_gems - diamonds = 5110 := by
  sorry

end rubies_in_treasure_l206_206803


namespace vector_magnitude_equiv_l206_206156

open Real

variables (m : ℝ)
variables (a b : ℝ × ℝ)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)

theorem vector_magnitude_equiv (h : dot_product (4, m) (1, -2) = 0) :
  magnitude (add (4, m) (1, -2)) = 5 :=
by
  sorry

end vector_magnitude_equiv_l206_206156


namespace even_in_third_row_onward_l206_206450

variable (rows : ℕ → List ℕ)
variable (sum_rule : ∀ (n i : ℕ), rows (n + 1) !! i = 
  (rows n !! (i - 1) + rows n !! i + rows n !! (i + 1)) % 2)

theorem even_in_third_row_onward :
  ∀ n ≥ 2, ∃ i, rows n !! i % 2 = 0 :=
sorry

end even_in_third_row_onward_l206_206450


namespace cannot_divide_660_stones_into_31_heaps_l206_206271

theorem cannot_divide_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i, 0 < heaps i ∧ heaps i < 2 * heaps (succ i % 31)) :=
sorry

end cannot_divide_660_stones_into_31_heaps_l206_206271


namespace final_amount_correct_l206_206374

def cost_bread := 5 * 6.50
def cost_orange_juice := 4 * 3.25
def cost_cookies := 7 * 2.10
def cost_apples := 8 * 1.75
def cost_soups := (9 / 3) * 4
def cost_soda := (12 / 12) * 11
def cost_chocolate_bars := (15 / 3) * 2 * 1.80
def cost_chips := 2 * 4.50 - 2.00

def total_cost_before_discount := cost_bread + cost_orange_juice + cost_cookies + cost_apples + cost_soups + cost_soda + cost_chocolate_bars + cost_chips

def discount := 0.05 * total_cost_before_discount

def final_cost := total_cost_before_discount - discount

def initial_amount := 250

def money_left := initial_amount - final_cost

theorem final_amount_correct : money_left = 133.91 :=
by
  sorry

end final_amount_correct_l206_206374


namespace certain_number_is_82_l206_206336

theorem certain_number_is_82 (n : ℕ) (h1 : n = 163) (h2 : n % 2 = 1) : 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.range (n + 1)), i) = 81 * 82 :=
by
  -- Convert the conditions into Lean definitions and use them
  have h3 : ∑ i in finset.filter (λ x, x % 2 = 0) (finset.range 164), i = 
    81 * 82 := sorry
  exact h3

end certain_number_is_82_l206_206336


namespace find_n_l206_206868

theorem find_n (n : ℕ) (h1 : 0 < n) : 
  ∃ n, n > 0 ∧ (Real.tan (Real.pi / (2 * n)) + Real.sin (Real.pi / (2 * n)) = n / 3) := 
sorry

end find_n_l206_206868


namespace squido_oysters_l206_206732

theorem squido_oysters (S C : ℕ) (h1 : C ≥ 2 * S) (h2 : S + C = 600) : S = 200 :=
sorry

end squido_oysters_l206_206732


namespace tangent_line_at_point_l206_206138

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 * real.log x - 2 * x + 1

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := 2 * x * real.log x + x - 2

-- Define the tangent line equation
def tangent_line (m x₁ y₁ x : ℝ) : ℝ := m * (x - x₁) + y₁

-- Theorem statement
theorem tangent_line_at_point :
  let m := f_prime 1
  let y₁ := f 1
  m = -1 →
  y₁ = -1 →
  ∀ (x y : ℝ), tangent_line m 1 y₁ x = y → x + y = 0 :=
by
  intros m y₁ hm hy₁ x y h
  have : f 1 = -1 := hy₁
  have : f_prime 1 = -1 := hm
  sorry

end tangent_line_at_point_l206_206138


namespace triangle_inequality_l206_206414

noncomputable def problem (A B C P Q R : Type) [triangle A B C] (circumcircle : circle A B C)
  (is_bisector_A : bisector_to_circumcircle A P circumcircle)
  (is_bisector_B : bisector_to_circumcircle B Q circumcircle)
  (is_bisector_C : bisector_to_circumcircle C R circumcircle) : Prop :=
  dist A P + dist B Q + dist C R > dist A B + dist B C + dist C A

theorem triangle_inequality (A B C P Q R : Type) [triangle A B C] (circumcircle : circle A B C)
  (is_bisector_A : bisector_to_circumcircle A P circumcircle)
  (is_bisector_B : bisector_to_circumcircle B Q circumcircle)
  (is_bisector_C : bisector_to_circumcircle C R circumcircle) : 
  dist A P + dist B Q + dist C R > dist A B + dist B C + dist C A :=
by
  sorry

end triangle_inequality_l206_206414


namespace possible_values_b_count_l206_206673

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206673


namespace range_of_m_l206_206554

noncomputable def f (x : ℝ) (m : ℝ) := 9^x - m * 3^x - 3

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f x m = 9^x - m * 3^x - 3) →
  (∀ x : ℝ, f (-x) m = - f(x) m) →
  -2 ≤ m :=
sorry

end range_of_m_l206_206554


namespace largest_prime_palindrome_factor_of_450_and_315_l206_206366

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

theorem largest_prime_palindrome_factor_of_450_and_315 : 
  ∃ p, p ∣ 450 ∧ p ∣ 315 ∧ Prime p ∧ is_palindrome p ∧ ∀ q, (q ∣ 450 ∧ q ∣ 315 ∧ Prime q ∧ is_palindrome q) → q ≤ p :=
sorry

end largest_prime_palindrome_factor_of_450_and_315_l206_206366


namespace priyas_speed_is_30_l206_206303

noncomputable def find_priyas_speed (v : ℝ) : Prop :=
  let riya_speed := 20
  let time := 0.5  -- in hours
  let distance_apart := 25
  (riya_speed + v) * time = distance_apart

theorem priyas_speed_is_30 : ∃ v : ℝ, find_priyas_speed v ∧ v = 30 :=
by
  sorry

end priyas_speed_is_30_l206_206303


namespace triangle_equilateral_if_abs_eq_zero_l206_206474

theorem triangle_equilateral_if_abs_eq_zero (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_if_abs_eq_zero_l206_206474


namespace relay_team_order_l206_206598

def number_of_orders (total_members : ℕ) (lara_position : ℕ) : ℕ :=
  if lara_position = total_members then (total_members - 1)! else 0

theorem relay_team_order : number_of_orders 5 5 = 24 :=
by
  sorry

end relay_team_order_l206_206598


namespace range_of_m_local_odd_function_l206_206555

def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

noncomputable def f (x m : ℝ) : ℝ :=
  9^x - m * 3^x - 3

theorem range_of_m_local_odd_function :
  (∀ m : ℝ, is_local_odd_function (λ x => f x m) ↔ m ∈ Set.Ici (-2)) :=
by
  sorry

end range_of_m_local_odd_function_l206_206555


namespace simplify_neg_expression_l206_206749

variable (a b c : ℝ)

theorem simplify_neg_expression : 
  - (a - (b - c)) = -a + b - c :=
sorry

end simplify_neg_expression_l206_206749


namespace solve_inequality_l206_206706

-- Define the function satisfying the given conditions
def f (x : ℝ) : ℝ := sorry

axiom f_functional_eq : ∀ (x y : ℝ), f (x / y) = f x - f y
axiom f_not_zero : ∀ x : ℝ, f x ≠ 0
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

-- Define the theorem that proves the inequality given the conditions
theorem solve_inequality (x : ℝ) :
  f x + f (x + 1/2) < 0 ↔ x ∈ (Set.Ioo ( (1 - Real.sqrt 17) / 4 ) 0) ∪ (Set.Ioo 0 ( (1 + Real.sqrt 17) / 4 )) :=
by
  sorry

end solve_inequality_l206_206706


namespace number_of_possible_values_of_b_l206_206665

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206665


namespace Kayla_total_items_l206_206213

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l206_206213


namespace probability_of_pink_l206_206206

variable (B P : ℕ) -- number of blue and pink gumballs
variable (h_total : B + P > 0) -- there is at least one gumball in the jar
variable (h_prob_two_blue : (B / (B + P)) * (B / (B + P)) = 16 / 49) -- the probability of drawing two blue gumballs in a row

theorem probability_of_pink : (P / (B + P)) = 3 / 7 :=
sorry

end probability_of_pink_l206_206206


namespace slope_product_l206_206007

theorem slope_product (k1 : ℝ) (k2 : ℝ) :
    (∃ (P1 P2 : ℝ × ℝ),
        P1 ≠ P2 ∧ 
        (∃ (px : ℝ), (∃ (M P : ℝ × ℝ), 
            M = (-1,0) ∧ 
            (∃ (l : ℝ → ℝ), 
                (l px = k1 * (px + 1)) ∧ 
                (∀ x y: ℝ, l x = y → x^2 + 2 * y^2 = 4)) ∧ 
            P = ( (fst P1 + fst P2) / 2, (snd P1 + snd P2) / 2) ∧ 
            k2 = snd P / fst P))) 
    → k1 * k2 = -1 / 2 :=
by
  sorry

end slope_product_l206_206007


namespace probability_three_primes_l206_206424

def prime_between_one_and_twelve := {2, 3, 5, 7, 11}

theorem probability_three_primes :
  (∃ (p : ℚ), p = (3500 : ℚ) / 20736) :=
begin
  -- conditions
  let faces := 12,
  let primes_count := 5,
  let dice := 4,
  let successes := 3,

  -- probabilities
  let p_prime := primes_count / faces,
  let p_not_prime := 1 - p_prime,
  let binom_coeff := nat.choose dice successes,

  -- calculation
  let probability := binom_coeff * p_prime^successes * p_not_prime^(dice - successes),
  
  -- desired outcome
  use probability,
  norm_cast,
  norm_num,
  simp,
end

end probability_three_primes_l206_206424


namespace track_laying_problem_l206_206776

noncomputable def trackLayingEquation (x : ℝ) : Prop :=
  let original_days := 6000 / x
  let revised_days := 6000 / (x + 20)
  original_days - revised_days = 15

theorem track_laying_problem (x : ℝ) (h1 : x > 0) (h2 : x ≠ -20) : trackLayingEquation x :=
  by
    let original_days := 6000 / x
    let revised_days := 6000 / (x + 20)
    have eq : original_days - revised_days = 15
    sorry

end track_laying_problem_l206_206776


namespace ella_last_roll_probability_l206_206976

theorem ella_last_roll_probability :
  let p := ((5/6)^10 * (1/6)) in
  abs (p - 0.027) < 0.001 := sorry

end ella_last_roll_probability_l206_206976


namespace circle_area_difference_l206_206435

/-- 
Prove that the area of the circle with radius r1 = 30 inches is 675π square inches greater than 
the area of the circle with radius r2 = 15 inches.
-/
theorem circle_area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15) :
  π * r1^2 - π * r2^2 = 675 * π := 
by {
  -- Placeholders to indicate where the proof would go
  sorry 
}

end circle_area_difference_l206_206435


namespace cannot_divide_660_stones_into_31_heaps_l206_206270

theorem cannot_divide_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i, 0 < heaps i ∧ heaps i < 2 * heaps (succ i % 31)) :=
sorry

end cannot_divide_660_stones_into_31_heaps_l206_206270


namespace eventually_periodic_iff_odd_l206_206767

theorem eventually_periodic_iff_odd (u : ℚ) (m : ℕ) 
  (h_positive_u : 0 < u) (h_positive_m : 0 < m) :
  (∃ c t : ℕ, ∀ n : ℕ, n ≥ c → 
    let q : ℕ → ℚ := 
      λ n, match n with
           | 0     := u
           | (n+1) := if h : n ≥ 1 then 
                        let (a, b) := num_den (q n) in
                        (a + (m : ℚ) * b) / (b + 1)
                      else 
                        sorry -- this case should not happen because n ≥ 1
      in q n = q (n + t)) ↔ odd m := sorry

/-- Function to extract numerator and denominator of a rational number in irreducible form --/
def num_den (r : ℚ) : ℚ × ℚ := 
  let ⟨a, b, h, coprime⟩ := rat.num_denom r in 
  (a, b)

end eventually_periodic_iff_odd_l206_206767


namespace same_mass_probability_l206_206770

theorem same_mass_probability : 
  let nums := [1, 2, 3, 4, 5],
      masses := λ x : ℕ, x^2 - 5 * x + 30,
      pairs := (nums.pairwise_disjoint_on (λ x y, masses x = masses y)) in
  (pairs.card : ℝ) / (finset.card (nums.product nums).filter (λ p, p.1 ≠ p.2).to_finset) = 1 / 5 :=
by sorry

end same_mass_probability_l206_206770


namespace sum_interior_angles_star_l206_206035

theorem sum_interior_angles_star {n : ℕ} (h : n ≥ 6) :
  let S := ∑ i in Finset.range n, (180 - 360 / n : ℝ)
  S = 180 * (n - 2) :=
by
  let alpha := 360 / n
  have h_alpha : alpha * n = 360 := by sorry
  let S := ∑ i in Finset.range n, (180 - alpha : ℝ)
  calc
    S = ∑ i in Finset.range n, 180 - alpha : _
    ... = 180 * n - (∑ i in Finset.range n, alpha) : by sorry
    ... = 180 * n - (alpha * n) : by sorry
    ... = 180 * n - 360 : by sorry
    ... = 180 * (n - 2) : by sorry

end sum_interior_angles_star_l206_206035


namespace correct_numbering_scheme_for_population_size_l206_206110

theorem correct_numbering_scheme_for_population_size :
  ∀ (population_size sample_size : ℕ), population_size = 106 →
  sample_size = 10 →
  ∃ numbering_scheme : string,
  numbering_scheme = "000,001,…,105" :=
by
  intros population_size sample_size pop_eq sample_eq
  have numbering_scheme := "000,001,…,105"
  use numbering_scheme
  sorry

end correct_numbering_scheme_for_population_size_l206_206110


namespace heap_division_impossible_l206_206240

def similar_sizes (a b : Nat) : Prop := a < 2 * b

theorem heap_division_impossible (n k : Nat) (h1 : n = 660) (h2 : k = 31) 
    (h3 : ∀ p1 p2 : Nat, p1 ∈ (Finset.range k).to_set → p2 ∈ (Finset.range k).to_set → similar_sizes p1 p2 → p1 < 2 * p2) : 
    False := 
by
  sorry

end heap_division_impossible_l206_206240


namespace correct_operation_l206_206750

-- Define the conditions as hypotheses
variable (a : ℝ)

-- A: \(a^2 \cdot a = a^3\)
def condition_A : Prop := a^2 * a = a^3

-- B: \((a^3)^3 = a^6\)
def condition_B : Prop := (a^3)^3 = a^6

-- C: \(a^3 + a^3 = a^5\)
def condition_C : Prop := a^3 + a^3 = a^5

-- D: \(a^6 \div a^2 = a^3\)
def condition_D : Prop := a^6 / a^2 = a^3

-- Proof that only condition A is correct:
theorem correct_operation : condition_A a ∧ ¬condition_B a ∧ ¬condition_C a ∧ ¬condition_D a :=
by
  sorry  -- Actual proofs would go here

end correct_operation_l206_206750


namespace sin_neg_1020_eq_sqrt3_div_2_sin_120_eq_sqrt3_div_2_sin_neg_1020_eq_sqrt3_div_2_combined_l206_206819

noncomputable def sin_period : ℝ := 360

theorem sin_neg_1020_eq_sqrt3_div_2 :
  Real.sin (degToRad (-1020)) = Real.sin (degToRad 120) :=
by
  have h : Real.sin (degToRad (-1020)) = Real.sin (degToRad (-1020 + 360 * 3)) := by
    rw [← Real.sin_periodic]
    norm_num
  rw [h]
  norm_num

theorem sin_120_eq_sqrt3_div_2 :
  Real.sin (degToRad 120) = sqrt(3) / 2 :=
by
  sorry

theorem sin_neg_1020_eq_sqrt3_div_2_combined :
  Real.sin (degToRad (-1020)) = sqrt(3) / 2 :=
by
  rw [sin_neg_1020_eq_sqrt3_div_2]
  rw [sin_120_eq_sqrt3_div_2]
  apply rfl

end sin_neg_1020_eq_sqrt3_div_2_sin_120_eq_sqrt3_div_2_sin_neg_1020_eq_sqrt3_div_2_combined_l206_206819


namespace quotient_when_m_divided_by_11_is_2_l206_206316

theorem quotient_when_m_divided_by_11_is_2 :
  let n_values := [1, 2, 3, 4, 5]
  let squares := n_values.map (λ n => n^2)
  let remainders := List.eraseDup (squares.map (λ x => x % 11))
  let m := remainders.sum
  m / 11 = 2 :=
by
  sorry

end quotient_when_m_divided_by_11_is_2_l206_206316


namespace polynomial_sum_l206_206001

noncomputable def cubic_q (x : ℝ) : ℝ := sorry

axiom q_condition_1 : cubic_q 3 = 2
axiom q_condition_2 : cubic_q 8 = 20
axiom q_condition_3 : cubic_q 18 = 12
axiom q_condition_4 : cubic_q 23 = 30

theorem polynomial_sum : (∑ n in finset.range 21, cubic_q (4 + n)) = 336 :=
by
  sorry

end polynomial_sum_l206_206001


namespace problem_a_problem_b_unique_solution_l206_206757

-- Problem (a)

theorem problem_a (a b c n : ℤ) (hnat : 0 ≤ n) (h : a * n^2 + b * n + c = 0) : n ∣ c :=
sorry

-- Problem (b)

theorem problem_b_unique_solution : ∀ n : ℕ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = 3 :=
sorry

end problem_a_problem_b_unique_solution_l206_206757


namespace triangular_prism_volume_l206_206798

noncomputable def volume_of_triangular_prism (r : ℝ) : ℝ :=
  let a := sqrt 3 in  -- side length of the equilateral triangle
  let h := a in  -- height of the prism
  let A := (sqrt 3 / 4) * a^2 in  -- area of the equilateral triangle base
  (A * h)  -- volume of the prism

theorem triangular_prism_volume (r : ℝ) (hr : r = 1) : volume_of_triangular_prism r = 2.25 :=
by
  -- definitions and conditions used here
  let a := sqrt 3 in
  let h := a in
  let A := (sqrt 3 / 4) * a ^ 2 in
  have ha : r = (a * sqrt 3) / 3 := by sorry,
  rw [hr, ha],
  show (A * h) = 2.25,
  sorry -- proof steps to be filled


end triangular_prism_volume_l206_206798


namespace rectangle_ratio_l206_206993

theorem rectangle_ratio (a b c d : ℝ) (CK_perp_DQ : ∀ (x : ℝ), x = 2):
  let side_length := 4
  let square_area := side_length * side_length
  let KP := side_length / 2
  let KD := KP
  let DQ := real.sqrt (KD ^ 2 + side_length ^ 2)
  let rect_height := (2 * KP * DQ) / side_length
  let rect_base := DQ
  let ratio := rect_height/rect_base in
  rect_height * rect_base = square_area ∧
  rect_base ≠ 0 ∧
  ratio = 4/5 := by
  sorry

end rectangle_ratio_l206_206993


namespace number_of_ordered_pairs_l206_206458

def has_digit_5 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * 10 + 5 + b * 10^2

def neither_has_digit_5 (a b : ℕ) : Prop :=
  ¬ has_digit_5 a ∧ ¬ has_digit_5 b

theorem number_of_ordered_pairs :
  let S := { (a, b) ∈ finset.range 500 ×ˢ finset.range 500 | a + b = 500 ∧ neither_has_digit_5 a b } in
  S.card = 449 :=
by
  sorry

end number_of_ordered_pairs_l206_206458


namespace find_k_l206_206155

-- Define the vectors and the condition of perpendicularity
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, -1)
def c (k : ℝ) : ℝ × ℝ := (3 + k, 1 - k)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The primary statement we aim to prove
theorem find_k : ∃ k : ℝ, dot_product a (c k) = 0 ∧ k = -5 :=
by
  exists -5
  sorry

end find_k_l206_206155


namespace polynomial_root_arithmetic_progression_b_l206_206853

theorem polynomial_root_arithmetic_progression_b (b : ℝ) :
  (∃ r : ℂ, ∃ d : ℂ, 
    (r - d) + r + (r + d) = 9 ∧
    (r - d) * r + (r - d) * (r + d) + r * (r + d) = 39 ∧
    (r - d) ∈ ℂ ∧ (r + d) ∈ ℂ ∧ 
    complex.is_conjugate (r - d) (r + d) ∧
    -(r - d) * r * (r + d) = b) → 
  b = 9 :=
by
  sorry

end polynomial_root_arithmetic_progression_b_l206_206853


namespace number_of_divisors_of_24_divisible_by_4_l206_206651

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206651


namespace surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l206_206855

-- First Problem:
theorem surface_area_cone_first_octant :
  ∃ (surface_area : ℝ), 
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ z^2 = 2*x*y) → surface_area = 16 :=
sorry

-- Second Problem:
theorem surface_area_sphere_inside_cylinder (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 ∧ x^2 + y^2 = R*x) → surface_area = 2 * R^2 * (π - 2) :=
sorry

-- Third Problem:
theorem surface_area_cylinder_inside_sphere (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 = R*x ∧ x^2 + y^2 + z^2 = R^2) → surface_area = 4 * R^2 :=
sorry

end surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l206_206855


namespace min_value_of_fraction_l206_206170

noncomputable def problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  problem_statement a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_fraction_l206_206170


namespace product_equals_eight_l206_206827

theorem product_equals_eight : (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7) = 8 := 
sorry

end product_equals_eight_l206_206827


namespace heap_division_impossible_l206_206264

theorem heap_division_impossible (n : ℕ) (k : ℕ) (similar : ℕ → ℕ → Prop)
  (h_similar : ∀ a b, similar a b ↔ a < 2 * b ∧ 2 * a > b) :
  n = 660 → k = 31 → ¬(∃ (heaps : Fin k → ℕ), ∑ i, heaps i = n ∧ ∀ i j, similar (heaps i) (heaps j)) :=
sorry

end heap_division_impossible_l206_206264


namespace trigonometric_identity_proof_l206_206906

theorem trigonometric_identity_proof (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : tan (π / 4 + α) = 2) :
  tan α = 1 / 3 ∧ (sin (2 * α) * cos α - sin α) / cos (2 * α) = sqrt 10 / 10 := 
sorry

end trigonometric_identity_proof_l206_206906


namespace heap_division_impossible_l206_206241

def similar_sizes (a b : Nat) : Prop := a < 2 * b

theorem heap_division_impossible (n k : Nat) (h1 : n = 660) (h2 : k = 31) 
    (h3 : ∀ p1 p2 : Nat, p1 ∈ (Finset.range k).to_set → p2 ∈ (Finset.range k).to_set → similar_sizes p1 p2 → p1 < 2 * p2) : 
    False := 
by
  sorry

end heap_division_impossible_l206_206241


namespace simplification_is_correct_l206_206644

noncomputable def simplify_expression (θ : ℝ) (h : θ = 35) : ℝ :=
  real.sqrt (1 - 2 * real.sin (θ) * real.cos (θ))

theorem simplification_is_correct : 
  simplify_expression 35 (by norm_num) = real.cos 35 - real.sin 35 :=
sorry

end simplification_is_correct_l206_206644


namespace no_unboxed_products_l206_206373

-- Definitions based on the conditions
def big_box_capacity : ℕ := 50
def small_box_capacity : ℕ := 40
def total_products : ℕ := 212

-- Theorem statement proving the least number of unboxed products
theorem no_unboxed_products (big_box_capacity small_box_capacity total_products : ℕ) : 
  (total_products - (total_products / big_box_capacity) * big_box_capacity) % small_box_capacity = 0 :=
by
  sorry

end no_unboxed_products_l206_206373


namespace f_9_eq_2_solve_f_2x_gt_2_plus_fx_minus_2_l206_206228

noncomputable def f : ℝ → ℝ := sorry

axiom f_mono_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f(x) < f(y)
axiom f_additivity : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) = f(x) + f(y)
axiom f_3_eq_1 : f(3) = 1

theorem f_9_eq_2 : f(9) = 2 :=
sorry

theorem solve_f_2x_gt_2_plus_fx_minus_2 (x : ℝ) (hx : x > 2) : 
  (f(2 * x) > 2 + f(x - 2)) ↔ (x ∈ Ioo (2:ℝ) (18/7)) :=
sorry

end f_9_eq_2_solve_f_2x_gt_2_plus_fx_minus_2_l206_206228


namespace normal_distribution_students_above_120_l206_206183

noncomputable def normal_distribution_probability_estimation (μ σ: ℝ) (n: ℕ): ℝ :=
if (μ = 110) ∧ (P 100 ≤ ℓ ≤ 110) = 0.2 ∧ (n = 800)
then 0.3 * n
else 0

theorem normal_distribution_students_above_120 :
  ∀ μ σ (X: ℝ → Prop) (n: ℕ), 
  (X ~ N(μ, σ^2)) → μ = 110 → n = 800 → 
  (∫ (100 <= ℓ and ℓ <= 110) and P(100 ≤ ℓ ≤ 110) = 0.2) → 
  (∫ (ℓ >= 120) = 0.3) → 
  normal_distribution_probability_estimation μ σ n = 240 :=
by {
    intros, sorry,
}

end normal_distribution_students_above_120_l206_206183


namespace max_distance_on_ellipse_l206_206615

noncomputable def ellipse (x y : ℝ) : Prop := (y^2 / 4) + (x^2 / 3) = 1

def A := (1 : ℝ, 1 : ℝ)
def B := (0 : ℝ, -1 : ℝ)

noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.fst - Q.fst)^2 + (P.snd - Q.snd)^2)

theorem max_distance_on_ellipse :
  ∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧ dist P A + dist P B = 5 :=
sorry

end max_distance_on_ellipse_l206_206615


namespace perp_PQ_AB_l206_206013

variables {A B C D P Q : Point}
variables [Rectangle ABCD]
variables (PC PD : Line)
variables (H_APC : Perpendicular (line_through A P) PC)
variables (H_BPD : Perpendicular (line_through B P) PD)
variables (HQ : point_of_intersection (line_through A (foot_of_perpendicular A PC)) (line_through B (foot_of_perpendicular B PD)) = Q)

theorem perp_PQ_AB : Perpendicular (line_through P Q) (line_through A B) :=
sorry

end perp_PQ_AB_l206_206013


namespace count_three_digit_values_is_38_l206_206606

def sum_of_digits (x : ℕ) : ℕ :=
  (toString x).foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0

def count_three_digit_values_with_condition : ℕ :=
  have three_digit_numbers : List ℕ := (100 : ℕ) :: List.range (999 - 100 + 1)
  three_digit_numbers.count (fun x => sum_of_digits (sum_of_digits x) = 4)

theorem count_three_digit_values_is_38 : count_three_digit_values_with_condition = 38 := by
  sorry

end count_three_digit_values_is_38_l206_206606


namespace segment_ap_length_l206_206799

theorem segment_ap_length (side : ℝ) (area_fraction : ℝ) (AP : ℝ) (AQ : ℝ) (P Q : ℝ) 
  (h_side : side = 4) (h_area_fraction : area_fraction = 4)
  (h_triangle : ∃ P Q, ∃ AP AQ, P ∈ segment(B, C) ∧ Q ∈ segment(C, D) ∧ right_triangle APQ ∧ AP = AQ) :
  AP = 2 * Real.sqrt 2 := 
sorry

end segment_ap_length_l206_206799


namespace binary_to_decimal_and_base7_l206_206822

theorem binary_to_decimal_and_base7 : 
    ∃ (d : ℕ) (b7 : ℕ), 
    nat.of_digits 2 [1, 0, 1, 1, 0, 1] = d ∧ 
    nat.of_digits 10 [4, 5] = d ∧ 
    nat.of_digits 7 [6, 3] = b7 :=
by
  sorry

end binary_to_decimal_and_base7_l206_206822


namespace correct_proposition_is_D_l206_206034

theorem correct_proposition_is_D :
  (∀ (pyramid : Type) (cut : Type), 
    ¬ (∃ (plane : Type), 
      -- condition A: unless the plane is parallel to the base, it's not a frustum
      (plane = cutting_plane ∧ ¬ (cutting_plane || base_of_pyramid))) // simplifying natural language explanation
  ) ∧
  (∀ (solid : Type), 
    ¬ (∃ (parallel_faces : Type) (quadrilaterals : Type),
      -- condition B: does not necessarily form a prism
      (solid.has_two_parallel_faces parallel_faces ∧ solid.all_other_faces quadrilaterals)
      ∧ solid.quadrilaterals.are_parallelograms quadrilaterals)))
  ) ∧
  (∀ (triangle : Type) {right_triangle : triangle}, 
    ¬ (∃ (side : Type),
      -- condition C: only forms a cone if it’s right-angle side
      (triangle.rotate side = cone ∧ ¬ (side = right_angle_side))
    )
   ∧
  (∀ (hemisphere : Type) (fixed_line : Type), 
    -- condition D: forms a sphere
    (hemisphere.rotate fixed_line = sphere)
  ) := sorry

end correct_proposition_is_D_l206_206034


namespace math_problem_l206_206612

theorem math_problem
  (x n : ℝ)
  (d e f : ℕ)
  (h1 : d + sqrt (e + sqrt f) = n)
  (h2 : 4 / (x - 4) + 6 / (x - 6) + 13 / (x - 13) + 15 / (x - 15) = x^2 - 9 * x - 5)
  (h3 : x = n)
  (h4 : n = d + sqrt (e + sqrt f))
  : d + e + f = 92 :=
by
  sorry

end math_problem_l206_206612


namespace triangle_area_is_twelve_l206_206762

-- Definitions according to the conditions
def base : ℝ := 4  -- in meters
def height : ℝ := 6  -- in meters
def area (b h : ℝ) : ℝ := (b * h) / 2

-- The proof problem statement
theorem triangle_area_is_twelve : area base height = 12 := 
by 
  sorry

end triangle_area_is_twelve_l206_206762


namespace num_children_proof_l206_206807

noncomputable def number_of_children (total_persons : ℕ) (total_revenue : ℕ) (adult_price : ℕ) (child_price : ℕ) : ℕ :=
  let adult_tickets := (child_price * total_persons - total_revenue) / (child_price - adult_price)
  let child_tickets := total_persons - adult_tickets
  child_tickets

theorem num_children_proof : number_of_children 280 14000 60 25 = 80 := 
by
  unfold number_of_children
  sorry

end num_children_proof_l206_206807


namespace find_remainder_l206_206869

def P (x : ℝ) : ℝ := x^100 - 2 * x^51 + 1
def divisor (x : ℝ) : ℝ := x^2 - 1

theorem find_remainder :
  ∃ (a b : ℝ), ∀ (x : ℝ), P(x) = (x^2 - 1) * (P(x) / (x^2 - 1)) + (a * x + b) ∧
                   (a = -2) ∧ (b = 2) := 
sorry

end find_remainder_l206_206869


namespace no_division_660_stones_into_31_heaps_l206_206251

theorem no_division_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) :=
  sorry

end no_division_660_stones_into_31_heaps_l206_206251


namespace line_equation_parallel_l206_206327

theorem line_equation_parallel (c : ℝ) (x y : ℝ): 
  (line_eq : x - 2 * y + c = 0) →
  (point_cond : (1 - 2 * 0) + (c = 0)) →
  (parallel_cond : (x - 2 * y - 2 = 0)) →
  (line_eq_final : x - 2 * y - 1 = 0) :=
begin
  sorry
end

end line_equation_parallel_l206_206327


namespace impossible_arithmetic_mean_not_between_l206_206291

theorem impossible_arithmetic_mean_not_between (arr : ℤ → ℤ) :
  (∀ x y, ∃ (aₓ aᵧ : ℤ), arr(aₓ) = x ∧ arr(aᵧ) = y ∧ (x + y) / 2 ≠ x ∧ (x + y) / 2 ≠ y
  → false) :=
sorry

end impossible_arithmetic_mean_not_between_l206_206291


namespace part_I_part_II_part_III_l206_206095

noncomputable def a_3 := 1
noncomputable def S (n : ℕ) := nat.iterate (λ k, k * 2) n (1 / 2)

theorem part_I (a₁ a₂ : ℝ) (h1 : a₃ = 1) (h2 : S 1 = a₂) : a₁ = 1 / 2 ∧ a₂ = 1 / 2 := by
  sorry

theorem part_II (n : ℕ) (h1 : a₃ = 1) (h2 : S (n + 1) = aₙ₊₁) : S n = 2 ^ (n - 2) := by
  sorry

theorem part_III (n : ℕ) : (∑ k in finset.range n, (1 / (k+1)) - (1 / (k+2))) = (n / (2n + 4)) := by
  sorry

end part_I_part_II_part_III_l206_206095


namespace smallest_solution_equation_l206_206463

noncomputable def equation (x : ℝ) : ℝ :=
  (3*x / (x-3)) + ((3*x^2 - 45) / x) + 3

theorem smallest_solution_equation : 
  ∃ x : ℝ, equation x = 14 ∧ x = (1 - Real.sqrt 649) / 12 :=
sorry

end smallest_solution_equation_l206_206463


namespace heap_division_impossible_l206_206262

theorem heap_division_impossible (n : ℕ) (k : ℕ) (similar : ℕ → ℕ → Prop)
  (h_similar : ∀ a b, similar a b ↔ a < 2 * b ∧ 2 * a > b) :
  n = 660 → k = 31 → ¬(∃ (heaps : Fin k → ℕ), ∑ i, heaps i = n ∧ ∀ i j, similar (heaps i) (heaps j)) :=
sorry

end heap_division_impossible_l206_206262


namespace children_attended_l206_206808

theorem children_attended 
  (x y : ℕ) 
  (h₁ : x + y = 280) 
  (h₂ : 0.60 * x + 0.25 * y = 140) : 
  y = 80 := 
by
  sorry

end children_attended_l206_206808


namespace no_division_660_stones_into_31_heaps_l206_206252

theorem no_division_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) :=
  sorry

end no_division_660_stones_into_31_heaps_l206_206252


namespace seven_digit_palindromes_count_l206_206960

theorem seven_digit_palindromes_count : 
  let digits := [1, 2, 2, 4, 4, 4, 4] in
  let is_palindrome (l : List ℕ) := l = l.reverse in
  let palindromes := {l : List ℕ | l.length = 7 ∧ is_palindrome l ∧ l.perm digits} in
  palindromes.to_list.length = 10 :=
by
  sorry

end seven_digit_palindromes_count_l206_206960


namespace area_of_cut_surface_l206_206805

def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10
def angle_PQ : ℝ := 90

/--
The area of one new flat surface created by the cut through points P, Q, and the central axis of the cylinder is given
in the form aπ + b√c where a, b, and c are integers, with c not divisible by the square of any prime.

Determine: a + b + c.
-/
theorem area_of_cut_surface (radius : ℝ) (height : ℝ) (angle : ℝ) (a b c : ℤ) (c_not_div_square_prime : ¬∃ p : ℕ, prime p ∧ p^2 ∣ c):
  radius = cylinder_radius →
  height = cylinder_height →
  angle = angle_PQ →
  a = 0 →
  b = 25 →
  c = 2 →
  c_not_div_square_prime →
  a + b + c = 27 := 
by
  intros hr hh ha ha_eq hb_eq hc_eq cprime
  sorry

end area_of_cut_surface_l206_206805


namespace number_of_five_digit_numbers_with_at_least_one_zero_l206_206534

-- Definitions for the conditions
def total_five_digit_numbers : ℕ := 90000
def five_digit_numbers_with_no_zeros : ℕ := 59049

-- Theorem stating that the number of 5-digit numbers with at least one zero is 30,951
theorem number_of_five_digit_numbers_with_at_least_one_zero : 
    total_five_digit_numbers - five_digit_numbers_with_no_zeros = 30951 :=
by
  sorry

end number_of_five_digit_numbers_with_at_least_one_zero_l206_206534


namespace focus_of_parabola_y2_eq_4x_l206_206324

theorem focus_of_parabola_y2_eq_4x :
  let p := 1 in (p, 0) = (1, 0) :=
by
  sorry

end focus_of_parabola_y2_eq_4x_l206_206324


namespace smallest_integer_not_prime_not_square_no_prime_lt_70_l206_206746

open Nat

theorem smallest_integer_not_prime_not_square_no_prime_lt_70 :
  ∃ n : ℕ, n = 5183 ∧
    ¬ Prime n ∧
    ∀ k : ℕ, k * k ≠ n ∧
    ∀ p : ℕ, Prime p → p ∣ n → p ≥ 70 :=
by
  sorry

end smallest_integer_not_prime_not_square_no_prime_lt_70_l206_206746


namespace angle_BTC_l206_206394

-- Definitions and assumptions based on given conditions
variables {α β γ δ θ ξ : ℝ} -- Angles
variables {A B C D E F P Q R S T : Type} -- Points

-- Conditions
def hexagon_inscribed (A B C D E F : Type) := 
  ∃ (k : Type), 
    ∃ (K : k → Prop), 
    (∀ (x : k), K x → ∃ (p : Type), is_on_circle A B K ∧ is_on_circle C D K ∧ is_on_circle E F K)

def intersection (P Q R S : Type) (p q r s : Type) := 
  (∃ P : Type, P = AB ∩ DC) ∧ 
  (∃ Q : Type, Q = BC ∩ ED) ∧ 
  (∃ R : Type, R = CD ∩ FE) ∧ 
  (∃ S : Type, S = DE ∩ AF)

def angles (f : Type → Type) :=
  (f B P C = 50 ∧ f C Q D = 45 ∧ f D R E = 40 ∧ f E S F = 35)

def find_T (f : Type → Type) :=
  ∃ T : Type, (T = BE ∩ CF)

-- Proof problem in Lean:
theorem angle_BTC (A B C D E F P Q R S T : Type) (a1 a2 a3 a4 : ℝ)
  (H1 : hexagon_inscribed A B C D E F)
  (H2 : intersection A B C D P Q R S)
  (H3 : angles (λ x, x = a1 ∨ x = a2 ∨ x = a3 ∨ x = a4))
  (H4 : find_T T)
  (H5 : a1 = 50) (H6 : a2 = 45) (H7 : a3 = 40) (H8 : a4 = 35) :
  ∃ btc, btc = 155 :=
by sorry

end angle_BTC_l206_206394


namespace minimum_value_f_minimum_value_abc_l206_206619

noncomputable def f (x : ℝ) : ℝ := abs (x - 4) + abs (x - 3)

theorem minimum_value_f : ∃ m : ℝ, m = 1 ∧ ∀ x : ℝ, f x ≥ m := 
by
  let m := 1
  existsi m
  sorry

theorem minimum_value_abc (a b c : ℝ) (h : a + 2 * b + 3 * c = 1) : ∃ n : ℝ, n = 1/14 ∧ a^2 + b^2 + c^2 ≥ n :=
by
  let n := 1 / 14
  existsi n
  sorry

end minimum_value_f_minimum_value_abc_l206_206619


namespace additional_girls_needed_l206_206341

theorem additional_girls_needed :
  ∃ g : ℕ, (2 + g) = (5 / 8 : ℝ) * (8 + g) :=
begin
  use 8, -- we choose 8 for g
  -- rest of the proof is omitted
  sorry
end

end additional_girls_needed_l206_206341


namespace graph_passing_point_l206_206709

def function_passing_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : Prop :=
  let f := λ x : ℝ, a^(x - 1) + 2
  f 1 = 3

-- Statement to prove
theorem graph_passing_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) :
  function_passing_point a h h1 :=
by
  -- Proof will be filled here
  sorry

end graph_passing_point_l206_206709


namespace area_ratio_preserved_by_affine_l206_206636

noncomputable theory

variables {P : Type} [AffineSpace ℝ P] 
variables (M N M' N' : Set P) -- Define polygons as sets of points

-- An affine transformation function
variable (T : P → P)
-- Condition that T is an affine transformation
variable (affine_T : AffineMap ℝ P P T)

-- The areas of polygons
variable (area : Set P → ℝ)
variable (h_area : ∀ A B : Set P, area (A ∪ B) = area A + area B)
variable (h_area_nonneg : ∀ A : Set P, 0 ≤ area A)

-- Affine transformation maps polygons M to M' and N to N'
variables (hM : M' = T '' M) (hN : N' = T '' N)

theorem area_ratio_preserved_by_affine :
  area M / area N = area M' / area N' := sorry

end area_ratio_preserved_by_affine_l206_206636


namespace no_division_660_stones_into_31_heaps_l206_206250

theorem no_division_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) :=
  sorry

end no_division_660_stones_into_31_heaps_l206_206250


namespace area_of_absolute_value_sum_eq_9_l206_206743

theorem area_of_absolute_value_sum_eq_9 :
  (∃ (area : ℝ), (|x| + |3 * y| = 9) → area = 54) := 
sorry

end area_of_absolute_value_sum_eq_9_l206_206743


namespace tangent_line_decreasing_interval_f_has_no_zeros_l206_206516

-- First, we define our functions and preliminary conditions.
def f (a : ℝ) (x : ℝ) := (2 - a) * (x - 1) - 2 * log x
def g (a : ℝ) (x : ℝ) := f(a, x) + x

-- Problem (I)
theorem tangent_line_decreasing_interval (a : ℝ) (h : g(a, 1) = 1) (h_tan : 1 - a = -1) : ∃ I, I = (0 : ℝ) < ∧ I < (2 : ℝ) ∧ ∀ x ∈ I, (3 - a) - 2 / x < 0 := sorry

-- Problem (II)
theorem f_has_no_zeros (a : ℝ) (h : ∀ x ∈ set.Ioo (0 : ℝ) (1/2 : ℝ), f a x > 0) : a ≥ 2 - 4 * log 2 := sorry

end tangent_line_decreasing_interval_f_has_no_zeros_l206_206516


namespace find_d_l206_206712

theorem find_d (x y t : ℝ) (d : ℝ × ℝ)
  (h1 : y = (2 * x + 1) / 3)
  (h2 : (⟨x, y⟩: ℝ × ℝ) = ⟨-1, 0⟩ + t • d)
  (h3 : x ≥ -1)
  (h4 : dist (⟨x, y⟩ : ℝ × ℝ) ⟨-1, 0⟩ = t) :
  d = ⟨3/real.sqrt 13, 2/real.sqrt 13⟩ :=
sorry

end find_d_l206_206712


namespace repeating_decimal_as_fraction_l206_206076

-- Define the repeating decimal 0.36666... as a real number
def repeating_decimal : ℝ := 0.366666666666...

-- State the theorem to express the repeating decimal as a fraction
theorem repeating_decimal_as_fraction : repeating_decimal = (11 : ℝ) / 30 := 
sorry

end repeating_decimal_as_fraction_l206_206076


namespace distinct_cubed_mod_7_units_digits_l206_206157

theorem distinct_cubed_mod_7_units_digits : 
  (∃ S : Finset ℕ, S.card = 3 ∧ ∀ n ∈ (Finset.range 7), (n^3 % 7) ∈ S) :=
  sorry

end distinct_cubed_mod_7_units_digits_l206_206157


namespace calculate_expression_l206_206045

theorem calculate_expression : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end calculate_expression_l206_206045


namespace accounting_majors_l206_206179

theorem accounting_majors (p q r s t u : ℕ) 
  (hpqt : (p * q * r * s * t * u = 51030)) 
  (hineq : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) :
  p = 2 :=
sorry

end accounting_majors_l206_206179


namespace heap_division_impossible_l206_206261

theorem heap_division_impossible (n : ℕ) (k : ℕ) (similar : ℕ → ℕ → Prop)
  (h_similar : ∀ a b, similar a b ↔ a < 2 * b ∧ 2 * a > b) :
  n = 660 → k = 31 → ¬(∃ (heaps : Fin k → ℕ), ∑ i, heaps i = n ∧ ∀ i j, similar (heaps i) (heaps j)) :=
sorry

end heap_division_impossible_l206_206261


namespace problem_statement_l206_206103

variables {m n : Type} [line m] [line n]
variables {α β γ : Type} [plane α] [plane β] [plane γ]

-- Lines and planes are distinct
axiom distinct_lines (h : m ≠ n) : Prop
axiom distinct_planes (h : α ≠ β) (h : β ≠ γ) (h : α ≠ γ) : Prop

-- Conditions

axiom parallel_planes (α β : Type) [plane α] [plane β] : Prop
axiom perpendicular_planes (α β : Type) [plane α] [plane β] : Prop
axiom line_in_plane (m : Type) [line m] (α : Type) [plane α] : Prop
axiom parallel_lines (m n : Type) [line m] [line n] : Prop
axiom perpendicular_lines (m : Type) [line m] (α : Type) [plane α] : Prop

theorem problem_statement :
  (parallel_planes α β → parallel_planes β γ → parallel_planes α γ) ∧
  ((perpendicular_planes α γ ∧ perpendicular_planes β γ) → ¬ parallel_planes α β) ∧
  (perpendicular_lines m α → perpendicular_lines n α → parallel_lines m n) ∧
  (parallel_planes α β → line_in_plane n α → parallel_lines n β) :=
by sorry

end problem_statement_l206_206103


namespace vector_angle_120_deg_l206_206544

open Real

noncomputable def angle_between_vectors (e₁ e₂ : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  let a := 2 • e₁ + e₂
  let b := -3 • e₁ + 2 • e₂
  let dot_product := (a ⬝ b)
  let norm_a := ‖a‖
  let norm_b := ‖b‖
  acos (dot_product / (norm_a * norm_b))

theorem vector_angle_120_deg {e₁ e₂ : EuclideanSpace ℝ (Fin 2)} 
  (h₁ : ‖e₁‖ = 1) (h₂ : ‖e₂‖ = 1) (h₃ : e₁ ⬝ e₂ = (1 / 2)) :
  angle_between_vectors e₁ e₂ = (2 * π / 3) :=
by sorry

end vector_angle_120_deg_l206_206544


namespace valid_pin_count_l206_206565

def total_pins : ℕ := 10^5

def restricted_pins (seq : List ℕ) : ℕ :=
  if seq = [3, 1, 4, 1] then 10 else 0

def valid_pins (seq : List ℕ) : ℕ :=
  total_pins - restricted_pins seq

theorem valid_pin_count :
  valid_pins [3, 1, 4, 1] = 99990 :=
by
  sorry

end valid_pin_count_l206_206565


namespace circle_radius_l206_206106

theorem circle_radius (m : ℝ) (h : 2 * 1 + (-m / 2) = 0) :
  let radius := 1 / 2 * Real.sqrt (4 + m ^ 2 + 16)
  radius = 3 :=
by
  sorry

end circle_radius_l206_206106


namespace second_candidate_votes_correct_l206_206997

-- Define the given condition variables
def votes_winning_candidate : ℕ := 11628
def percentage_winning_candidate : ℝ := 49.69230769230769 / 100
def votes_other_candidate : ℕ := 4136

-- Define the total votes based on the given percentage and votes of the winning candidate
noncomputable def total_votes : ℕ := (votes_winning_candidate : ℝ) / percentage_winning_candidate

-- Define the number of votes the second candidate received
noncomputable def votes_second_candidate : ℕ :=
  total_votes - (votes_winning_candidate + votes_other_candidate)

-- The goal is to prove that the second candidate received 7636 votes
theorem second_candidate_votes_correct :
  votes_second_candidate = 7636 := by
  sorry

end second_candidate_votes_correct_l206_206997


namespace scientific_notation_distance_l206_206325

theorem scientific_notation_distance (a n : ℝ) (h1 : 383900 = a * 10^n) (h2 : 1 ≤ |a| < 10) :
  a = 3.839 ∧ n = 5 := by
  sorry

end scientific_notation_distance_l206_206325


namespace unique_sequence_length_l206_206725

theorem unique_sequence_length (b : ℕ → ℕ) (m : ℕ) :
  (∀ i j, i < j → b i < b j) ∧ (∃ b_list : List ℕ, b_list.length = m ∧ 
      (∀ i, b_list.nth i = some (b i)) ∧ 
      (2^195 + 1) / (2^15 + 1) = b_list.sum (λ x, 2^x)) → m = 97 := 
sorry

end unique_sequence_length_l206_206725


namespace modeling_clay_problem_l206_206804

def volume_block (length width height : ℝ) : ℝ :=
  length * width * height

def volume_cylinder (radius height : ℝ) : ℝ :=
  Real.pi * radius ^ 2 * height

def number_of_blocks (volume_block volume_cylinder : ℝ) : ℝ :=
  (volume_cylinder / volume_block).ceil

theorem modeling_clay_problem :
  let block_volume := volume_block 8 3 2 in
  let sculpture_volume := volume_cylinder 3 9 in
  number_of_blocks block_volume sculpture_volume = 6 :=
by
  -- The proof would go here, but is omitted as per the requirements
  sorry

end modeling_clay_problem_l206_206804


namespace length_of_c_l206_206198

theorem length_of_c (A B C : ℝ) (a b c : ℝ) (h1 : (π / 3) - A = B) (h2 : a = 3) (h3 : b = 5) : c = 7 :=
sorry

end length_of_c_l206_206198


namespace community_center_table_count_l206_206390

theorem community_center_table_count : 
  let num_people := 3 * (5^2) + 2 * (5^1) + 1 * (5^0) in
  num_people = 86 →
  let tables_needed := Int.ceil ((num_people : ℤ) / 3) in
  tables_needed = 29 :=
by 
  sorry

end community_center_table_count_l206_206390


namespace sequence_properties_l206_206926

variable {n : ℕ}
variable {a : ℕ → ℤ}
variable {b : ℕ → ℤ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n, b (n + 1) = q * b n

-- Theorem statement
theorem sequence_properties (ha : arithmetic_sequence a 3) (hb : geometric_sequence b 2) :
  (∀ n, a (n + 1) + 1 - (a n + 1) = 3) ∧
  (∀ n, b (a (n + 1)) = 8 * b (a n)) :=
by
  sorry

end sequence_properties_l206_206926


namespace domain_of_function_l206_206446

open Real

noncomputable def domain_of_f : Set ℝ := { x | -9 < x ∧ x < 3 }

theorem domain_of_function :
  ∀ x : ℝ, (f : ℝ → ℝ) (x) = (log 2 (3 - x)) / sqrt (81 - x^2) → x ∈ domain_of_f :=
by
  sorry

end domain_of_function_l206_206446


namespace angle_between_foci_l206_206634

variables (P F1 F2 : ℝ × ℝ)
-- Ellipse centered at origin with semi-major axis 4 and semi-minor axis 3
def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16 + P.2^2 / 9 = 1)

-- Foci of the ellipse at (-2√7, 0) and (2√7, 0)
def is_focus (F : ℝ × ℝ) : Prop :=
  F = (-2 * Real.sqrt 7, 0) ∨ F = (2 * Real.sqrt 7, 0)

-- Given condition |PF1| * |PF2| = 12
def product_of_distances (P F1 F2 : ℝ × ℝ) : Prop :=
  (Real.dist P F1) * (Real.dist P F2) = 12

-- Proof statement
theorem angle_between_foci (h1: on_ellipse P) (h2: is_focus F1) (h3: is_focus F2) (h4: product_of_distances P F1 F2) :
  ∃ θ : ℝ, θ = 60 ∧ cos θ = (40 - 28) / (2 * 12) :=
sorry

end angle_between_foci_l206_206634


namespace exp_prop_l206_206361

theorem exp_prop : (9/11) ^ 4 * (9/11) ^ (-4) = 1 := by
  -- The properties of exponents are used here
  sorry

end exp_prop_l206_206361


namespace total_cost_is_103_l206_206389

-- Base cost of the plan is 20 dollars
def base_cost : ℝ := 20

-- Cost per text message in dollars
def cost_per_text : ℝ := 0.10

-- Cost per minute over 25 hours in dollars
def cost_per_minute_over_limit : ℝ := 0.15

-- Number of text messages sent
def text_messages : ℕ := 200

-- Total hours talked
def hours_talked : ℝ := 32

-- Free minutes (25 hours)
def free_minutes : ℝ := 25 * 60

-- Calculating the extra minutes talked
def extra_minutes : ℝ := (hours_talked * 60) - free_minutes

-- Total cost
def total_cost : ℝ :=
  base_cost +
  (text_messages * cost_per_text) +
  (extra_minutes * cost_per_minute_over_limit)

-- Proving that the total cost is 103 dollars
theorem total_cost_is_103 : total_cost = 103 := by
  sorry

end total_cost_is_103_l206_206389


namespace line_OG_independent_of_F1_F2_l206_206483

theorem line_OG_independent_of_F1_F2
  (O : Point)
  (e1 e2 f : Line)
  (h1 : O ∈ e1 ∧ O ∈ e2 ∧ O ∈ f)
  (F1 F2 : Point)
  (hf1 : F1 ∈ f ∧ F2 ∈ f)
  (ho1 : F1 ≠ O ∧ F2 ≠ O ∧ F1 ≠ F2)
  (E1 E2 : Point)
  (hperp1 : PerpendicularFrom(F1) f = e1 ∧ E1 = PerpendicularIntersection(F1) f e1)
  (hperp2 : PerpendicularFrom(F2) f = e2 ∧ E2 = PerpendicularIntersection(F2) f e2)
  (G : Point)
  (hG : G = lineIntersection (lineThrough E1 F2) (lineThrough E2 F1)) :
  ∀ (F1' F2' : Point), 
  (F1' ∈ f ∧ F2' ∈ f) ∧ (F1' ≠ O ∧ F2' ≠ O ∧ F1' ≠ F2') →
  G = lineIntersection (lineThrough (PerpendicularIntersection(F1') f e1) (PerpendicularIntersection(F2') f e2)) :=
by
  sorry

end line_OG_independent_of_F1_F2_l206_206483


namespace count_leftmost_digit_eight_l206_206443

noncomputable def U : set ℕ := {k | 0 ≤ k ∧ k ≤ 3000 ∧ (7^k).digits.head = 8}

theorem count_leftmost_digit_eight 
  (digits_3000 : (7^3000).digits.length = 2827) 
  (leftmost_3000 : (7^3000).digits.head = 8) : 
  ∃ n, n = 173 ∧ ∀ k, k ∈ U ↔ (7^k).digits.head = 8 := 
sorry

end count_leftmost_digit_eight_l206_206443


namespace woman_first_half_speed_l206_206030

noncomputable def first_half_speed (total_time : ℕ) (second_half_speed : ℕ) (total_distance : ℕ) : ℕ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem woman_first_half_speed : first_half_speed 20 24 448 = 21 := by
  sorry

end woman_first_half_speed_l206_206030


namespace endpoint_of_parallel_segment_l206_206008

theorem endpoint_of_parallel_segment (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hA : A = (2, 1)) (h_parallel : B.snd = A.snd) (h_length : abs (B.fst - A.fst) = 5) :
  B = (7, 1) ∨ B = (-3, 1) :=
by
  -- Proof goes here
  sorry

end endpoint_of_parallel_segment_l206_206008


namespace impossible_division_l206_206257

theorem impossible_division (heap_size : ℕ) (heap_count : ℕ) (similar_sizes : Π (a b : ℕ), a < 2 * b) :
  ¬ (∃ (heaps : Finset ℕ), heaps.card = heap_count ∧ heaps.sum = heap_size ∧ ∀ (a b ∈ heaps), similar_sizes a b) :=
by
  let heap_size := 660
  let heap_count := 31
  sorry

end impossible_division_l206_206257


namespace min_value_expression_l206_206866

theorem min_value_expression (x y : ℝ) : ∃ (x y : ℝ), x^2 + 2 * x * y + y^2 = 0 :=
by
  use (-y, y)
  ring
  sorry

end min_value_expression_l206_206866


namespace smallest_angle_measure_l206_206713

-- Define the conditions
def is_spherical_triangle (a b c : ℝ) : Prop :=
  a + b + c > 180 ∧ a + b + c < 540

def angles (k : ℝ) : Prop :=
  let a := 3 * k
  let b := 4 * k
  let c := 5 * k
  is_spherical_triangle a b c ∧ a + b + c = 270

-- Statement of the theorem
theorem smallest_angle_measure (k : ℝ) (h : angles k) : 3 * k = 67.5 :=
sorry

end smallest_angle_measure_l206_206713


namespace present_number_of_teachers_l206_206763

theorem present_number_of_teachers (S T : ℕ) (h1 : S = 50 * T) (h2 : S + 50 = 25 * (T + 5)) : T = 3 := 
by 
  sorry

end present_number_of_teachers_l206_206763


namespace possible_values_b_count_l206_206679

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206679


namespace number_of_arrangements_l206_206812

-- Definitions for people
inductive Person
| Jia 
| Yi 
| Bing 
| Ding 
| Wu
deriving DecidableEq, Inhabited

open Person

-- Theorem stating the conditions and the required proof
theorem number_of_arrangements (arr : List Person) :
  -- conditions:
  all_permutations arr ∈ permutations [Jia, Yi, Bing, Ding, Wu] →
  (∀ i, (arr.get? i = some Jia → (arr.get? (i+1) ≠ some Bing ∧ arr.get? (i-1) ≠ some Bing)) ∧
         (arr.get? i = some Yi → (arr.get? (i+1) ≠ some Bing ∧ arr.get? (i-1) ≠ some Bing))) →
  -- correct answer:
  count (valid_arrangements arr) = 36 :=
sorry

end number_of_arrangements_l206_206812


namespace modulus_of_complex_z_l206_206511

theorem modulus_of_complex_z (z : ℂ) (h : (⟂ * z = 1 + 2 * ⟂)) : complex.abs z = real.sqrt 5 :=
sorry

end modulus_of_complex_z_l206_206511


namespace min_frac_sum_l206_206480

open Real

noncomputable def minValue (m n : ℝ) : ℝ := 1 / m + 2 / n

theorem min_frac_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  minValue m n = 3 + 2 * sqrt 2 := by
  sorry

end min_frac_sum_l206_206480


namespace number_of_roots_in_interval_l206_206832

noncomputable def g : ℝ → ℝ := sorry

theorem number_of_roots_in_interval :
  (∀ x : ℝ, g (3 + x) = g (3 - x)) →
  (∀ x : ℝ, g (8 + x) = g (8 - x)) →
  g 0 = 0 →
  let roots := {x : ℝ | g x = 0 ∧ -1000 ≤ x ∧ x ≤ 1000} in
  set.finite roots ∧ (set.card roots = 335) :=
by 
  sorry

end number_of_roots_in_interval_l206_206832


namespace find_other_number_l206_206714

theorem find_other_number (A B : ℕ) (h1 : A = 24) (h2 : (HCF A B) = 14) (h3 : (LCM A B) = 312) : B = 182 :=
by
  sorry

end find_other_number_l206_206714


namespace repeating_decimal_fraction_l206_206082

theorem repeating_decimal_fraction : (0.366666... : ℝ) = 33 / 90 := 
sorry

end repeating_decimal_fraction_l206_206082


namespace rate_of_drawing_barbed_wire_is_correct_l206_206695

noncomputable def rate_of_drawing_barbed_wire (area cost: ℝ) (gate_width barbed_wire_extension: ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_barbed_wire := (perimeter - 2 * gate_width) + 4 * barbed_wire_extension
  cost / total_barbed_wire

theorem rate_of_drawing_barbed_wire_is_correct :
  rate_of_drawing_barbed_wire 3136 666 1 3 = 2.85 :=
by
  sorry

end rate_of_drawing_barbed_wire_is_correct_l206_206695


namespace no_55_rooms_l206_206999

open set

variable (rooms : ℕ)
variables (roses carnations chrysanthemums : ℕ)
variables (both_chrys_carnations both_chrys_roses both_carn_roses all_three : ℕ)

definition mansion_conditions (rooms : ℕ) (roses carnations chrysanthemums : ℕ) 
  (both_chrys_carnations both_chrys_roses both_carn_roses all_three : ℕ) : Prop :=
  roses = 30 ∧
  carnations = 20 ∧
  chrysanthemums = 10 ∧
  both_chrys_carnations = 2 ∧
  both_chrys_roses = 3 ∧
  both_carn_roses = 4 ∧
  rooms ≥ 1

theorem no_55_rooms (rooms roses carnations chrysanthemums : ℕ)
  (both_chrys_carnations both_chrys_roses both_carn_roses all_three : ℕ) :
  mansion_conditions rooms roses carnations chrysanthemums both_chrys_carnations both_chrys_roses both_carn_roses all_three →
  rooms ≠ 55 :=
begin
  sorry
end

end no_55_rooms_l206_206999


namespace total_wins_is_772_l206_206185

def wins_sam_first : ℕ := 50 * 200 / 100
def wins_sam_next : ℕ := 60 * 300 / 100
def wins_sam_total : ℕ := wins_sam_first + wins_sam_next

def wins_alex_first : ℕ := 65 * 150 / 100
def wins_alex_next : ℕ := 70 * 250 / 100
def wins_alex_total : ℕ := wins_alex_first + wins_alex_next

def wins_kim_first : ℕ := 70 * 100 / 100
def wins_kim_next : ℕ := 75 * 200 / 100
def wins_kim_total : ℕ := wins_kim_first + wins_kim_next

def total_wins : ℕ := wins_sam_total + wins_alex_total + wins_kim_total

theorem total_wins_is_772 : total_wins = 772 := by
  have h_sam : wins_sam_total = 280 := by sorry
  have h_alex : wins_alex_total = 272 := by sorry
  have h_kim : wins_kim_total = 220 := by sorry
  have h_total : total_wins = 280 + 272 + 220 := by sorry
  rw [h_total]
  norm_num

end total_wins_is_772_l206_206185


namespace no_sequence_exists_l206_206766

theorem no_sequence_exists :
  ¬ ∃ (a : Fin 15 → ℕ), 
    ∀ i : Fin 15, 
      (∑ j in Finset.filter (< a i) (Finset.Ico 0 15).toFinset, 1) = 
      [1, 0, 3, 6, 9, 4, 7, 2, 5, 8, 8, 5, 10, 13, 13][i] :=
by
  sorry

end no_sequence_exists_l206_206766


namespace cans_needed_for_30_rooms_l206_206294

noncomputable def cans_per_room := (40 - 30) / 6

theorem cans_needed_for_30_rooms (cans_per_room : ℝ) : 30 / cans_per_room = 18 := by
  have h : cans_per_room = (40 - 30) / 6 := rfl
  rw [h]
  calc 
    30 / ((40 - 30) / 6)
    = 30 * (6 / (40 - 30)) : by field_simp
    ... = 30 * (6 / 10) : by norm_num
    ... = 30 * (3 / 5) : by norm_num
    ... = 90 / 5 : by norm_num
    ... = 18 : by norm_num

end cans_needed_for_30_rooms_l206_206294


namespace problem1_problem2_l206_206940

-- Define the function f(x)
def f (m x : ℝ) : ℝ := |m * x + 1| + |2 * x - 3|

-- Problem 1: Prove the range of x for f(x) = 4 when m = 2
theorem problem1 (x : ℝ) : f 2 x = 4 ↔ -1 / 2 ≤ x ∧ x ≤ 3 / 2 :=
by
  sorry

-- Problem 2: Prove the range of m given f(1) ≤ (2a^2 + 8) / a for any positive a
theorem problem2 (m : ℝ) (h : ∀ a : ℝ, a > 0 → f m 1 ≤ (2 * a^2 + 8) / a) : -8 ≤ m ∧ m ≤ 6 :=
by
  sorry

end problem1_problem2_l206_206940


namespace initial_oranges_l206_206623

theorem initial_oranges (x : ℝ) :
  (x / 8 + 1 / 8 = x / 4 - 3 / 4) → x = 7 := 
begin
  intro h,
  have hx : x / 8 + 1 / 8 = x / 4 - 3 / 4, from h,
  -- Simplifying the equation to solve for x
  sorry
end

end initial_oranges_l206_206623


namespace product_of_roots_of_quartic_polynomial_l206_206613

theorem product_of_roots_of_quartic_polynomial :
  (∀ x : ℝ, (3 * x^4 - 8 * x^3 + x^2 - 10 * x - 24 = 0) → x = p ∨ x = q ∨ x = r ∨ x = s) →
  (p * q * r * s = -8) :=
by
  intros
  -- proof goes here
  sorry

end product_of_roots_of_quartic_polynomial_l206_206613


namespace right_triangle_condition_l206_206986

theorem right_triangle_condition (a b c : ℝ) (A B C : ℝ) :
  (a = 6 ∧ b = 8 ∧ c = 10) → a^2 + b^2 = c^2 :=
by
  intros h
  cases h with a_eq six_eq
  cases six_eq with b_eq c_eq
  rw [a_eq, b_eq, c_eq]
  linarith [36, 64, 100]
  sorry

end right_triangle_condition_l206_206986


namespace area_of_absolute_value_sum_eq_9_l206_206742

theorem area_of_absolute_value_sum_eq_9 :
  (∃ (area : ℝ), (|x| + |3 * y| = 9) → area = 54) := 
sorry

end area_of_absolute_value_sum_eq_9_l206_206742


namespace sum_of_first_33_terms_l206_206981

variable a d : ℝ

-- Conditions
def S3 : ℝ := 3 / 2 * (2 * a + 2 * d)
def S30 : ℝ := 30 / 2 * (2 * a + 29 * d)
def S33 : ℝ := 33 / 2 * (2 * a + 32 * d)

theorem sum_of_first_33_terms (h1 : S3 = 30) (h2 : S30 = 300) : S33 = 330 := by
  sorry

end sum_of_first_33_terms_l206_206981


namespace Kayla_total_items_l206_206214

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l206_206214


namespace dow_at_end_of_day_l206_206691

theorem dow_at_end_of_day (d0 : ℝ) (fall_percent : ℝ) (d1 : ℝ) :
  d0 = 8900 → fall_percent = 0.02 → d1 = d0 - fall_percent * d0 → d1 = 8722 :=
by
  intros h_d0 h_fall h_d1
  rw [h_d0, h_fall] at h_d1
  simp at h_d1
  exact h_d1

end dow_at_end_of_day_l206_206691


namespace part1_part2_l206_206904

-- Definitions for the conditions
def A : Set ℝ := {x : ℝ | 2 * x - 4 < 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}
def U : Set ℝ := Set.univ

-- The questions translated as Lean theorems
theorem part1 : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

theorem part2 : (U \ A) ∩ B = {x : ℝ | 2 ≤ x ∧ x < 5} := by
  sorry

end part1_part2_l206_206904


namespace johns_original_earnings_l206_206207

-- Define the conditions
def raises (original : ℝ) (percentage : ℝ) := original + original * percentage

-- The theorem stating the equivalent problem proof
theorem johns_original_earnings :
  ∃ (x : ℝ), raises x 0.375 = 55 ↔ x = 40 :=
sorry

end johns_original_earnings_l206_206207


namespace non_empty_even_subsets_count_l206_206964

theorem non_empty_even_subsets_count : 
  let s := {2, 4, 6, 8, 10}
  in (s.powerset.card - 1) = 31 :=
by
  -- defining the set s
  let s := {2, 4, 6, 8, 10}
  -- the number of non-empty subsets is the total number of subsets minus the empty set
  have hs : s.powerset.card - 1 = 31 := sorry
  exact hs

end non_empty_even_subsets_count_l206_206964


namespace number_of_divisors_of_24_divisible_by_4_l206_206649

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206649


namespace avg_age_assist_coaches_l206_206698

-- Define the conditions given in the problem

def total_members := 50
def avg_age_total := 22
def girls := 30
def boys := 15
def coaches := 5
def avg_age_girls := 18
def avg_age_boys := 20
def head_coaches := 3
def assist_coaches := 2
def avg_age_head_coaches := 30

-- Define the target theorem to prove
theorem avg_age_assist_coaches : 
  (avg_age_total * total_members - avg_age_girls * girls - avg_age_boys * boys - avg_age_head_coaches * head_coaches) / assist_coaches = 85 := 
  by
    sorry

end avg_age_assist_coaches_l206_206698


namespace num_of_possible_values_b_l206_206684

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206684


namespace no_negative_roots_but_at_least_one_positive_root_l206_206840

def f (x : ℝ) : ℝ := x^6 - 3 * x^5 - 6 * x^3 - x + 8

theorem no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → f x ≠ 0) ∧ (∃ x : ℝ, x > 0 ∧ f x = 0) :=
by {
  sorry
}

end no_negative_roots_but_at_least_one_positive_root_l206_206840


namespace max_value_achieved_l206_206493

variable (a₁ d : ℝ) (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (n : ℕ) : ℝ := n * a₁ + (n * (n - 1) / 2) * d

axiom a₀_pos : a₁ > 0
axiom sum_eq : sum_first_n_terms 3 = sum_first_n_terms 10

theorem max_value_achieved (n : ℕ) :
  S n = sum_first_n_terms n → (n = 6 ∨ n = 7) :=
sorry

end max_value_achieved_l206_206493


namespace smallest_prime_after_seven_consecutive_nonprimes_l206_206187

theorem smallest_prime_after_seven_consecutive_nonprimes :
  ∃ p, p > 96 ∧ Nat.Prime p ∧ ∀ n, 90 ≤ n ∧ n ≤ 96 → ¬Nat.Prime n :=
by
  sorry

end smallest_prime_after_seven_consecutive_nonprimes_l206_206187


namespace clotheslines_per_house_l206_206573

/-- There are a total of 11 children and 20 adults.
Each child has 4 items of clothing on the clotheslines.
Each adult has 3 items of clothing on the clotheslines.
Each clothesline can hold 2 items of clothing.
All of the clotheslines are full.
There are 26 houses on the street.
Show that the number of clotheslines per house is 2. -/
theorem clotheslines_per_house :
  (11 * 4 + 20 * 3) / 2 / 26 = 2 :=
by
  sorry

end clotheslines_per_house_l206_206573


namespace find_lambda_l206_206952

variable {R : Type*} [LinearOrderedField R]

/-- 
Given two non-zero vectors m and n with an angle π/3 between them,
and |n| = λ|m| (λ > 0).
The vector group x1, x2, x3 is composed of one m and two n's,
while the vector group y1, y2, y3 is composed of two m's and one n.
If the minimum possible value of x1 · y1 + x2 · y2 + x3 · y3 is 4m²,
then the value of λ is 8/3.
-/
theorem find_lambda 
  {m n : EuclideanSpace R (Fin 2)} (h₀ : m ≠ 0 ∧ n ≠ 0)
  (h₁ : ∠ (m: ℝ) (n: ℝ) = π/3)
  (h₂ : ∥n∥ = λ * ∥m∥ ∧ λ > 0)
  (h_min : ∀ x y z x' y' z', 
            (x, y, z) = (m, n, n)
            → (x', y', z') = (m, m, n)
            → x • x' + y • y' + z • z' = 4 * (∥m∥^2)) :
  λ = 8 / 3 :=
by
  sorry

end find_lambda_l206_206952


namespace solve_trolls_problem_l206_206072

def trolls_problem : Prop :=
  ∃ (B : ℕ), 
  8 = 2 * B - 4 ∧ 
  let plains := B / 2 in
  let mountain := plains + 3 in
  8 - mountain = 2 * B ∧
  8 + B + plains + mountain = 23

theorem solve_trolls_problem : trolls_problem :=
by {
  sorry
}

end solve_trolls_problem_l206_206072


namespace find_x_y_sum_l206_206617

variable {x y : ℝ}

theorem find_x_y_sum (h₁ : (x-1)^3 + 1997 * (x-1) = -1) (h₂ : (y-1)^3 + 1997 * (y-1) = 1) : 
  x + y = 2 := 
by
  sorry

end find_x_y_sum_l206_206617


namespace credibility_relation_l206_206469

theorem credibility_relation (X Y : Type) (K2 : Type) (k : K2) :
  (↑k : ℝ) → Prop :=
sorry

end credibility_relation_l206_206469


namespace prove_inequality_l206_206231

open Real

noncomputable def inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ 
  3 * (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1)

theorem prove_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  inequality a b c h1 h2 h3 := 
  sorry

end prove_inequality_l206_206231


namespace geometric_condition_l206_206150

def Sn (p : ℤ) (n : ℕ) : ℤ := p * 2^n + 2

def an (p : ℤ) (n : ℕ) : ℤ :=
  if n = 1 then Sn p n
  else Sn p n - Sn p (n - 1)

def is_geometric_progression (p : ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ r : ℤ, an p n = an p (n - 1) * r

theorem geometric_condition (p : ℤ) :
  is_geometric_progression p ↔ p = -2 :=
sorry

end geometric_condition_l206_206150


namespace rachel_robert_picture_probability_correct_l206_206639

noncomputable def probability_both_in_picture : ℚ :=
  have rachel_lap_time : ℚ := 75
  have robert_lap_time : ℚ := 100
  have time_after_start : ℚ := 720 -- 12 minutes in seconds
  have picture_fraction : ℚ := 1/5
  have picture_position : ℚ := 1/3
  
  -- Calculate time positions within the track
  let rachel_time_position := (time_after_start % rachel_lap_time)
  let robert_time_position := (time_after_start % robert_lap_time)
  
  -- Translate positions to within the bounds of the track
  let rachel_relative_position := rachel_time_position / rachel_lap_time
  let robert_relative_position := robert_time_position / robert_lap_time

  -- Define the range within the picture for Rachel and Robert
  let rachel_in_picture := (rachel_relative_position ≥ picture_position - picture_fraction) ∧ 
                           (rachel_relative_position ≤ picture_position + picture_fraction)
  let robert_in_picture := (robert_relative_position ≥ picture_position - picture_fraction) ∧
                           (robert_relative_position ≤ picture_position + picture_fraction)

  -- Calculate overlap
  let overlap := (if rachel_in_picture ∧ robert_in_picture then 
                    min (rachel_relative_position + picture_fraction) 
                        (robert_relative_position + picture_fraction) -
                    max (rachel_relative_position - picture_fraction)
                        (robert_relative_position - picture_fraction)
                  else 0)
                 
  -- Calculate probability
  overlap / 1 -- normalized to total track length

theorem rachel_robert_picture_probability_correct :
  probability_both_in_picture = 4/15 :=
by sorry

end rachel_robert_picture_probability_correct_l206_206639


namespace remove_min_toothpicks_no_triangles_l206_206846

/--
Fifty identical toothpicks are used to create a larger triangular grid.
The grid consists of 15 upward-pointing small triangles and 10 downward-pointing small triangles.
Prove that the fewest number of toothpicks that could be removed so that no triangles of any size remain is 15.
-/
theorem remove_min_toothpicks_no_triangles (t : ℕ) (u : ℕ) (d : ℕ) (ht : t = 50) (hu : u = 15) (hd : d = 10) :
  ∃ m : ℕ, m = 15 ∧ ∀ x : ℕ, x < 15 → triangles_remaining t u d m = 0 := 
sorry

end remove_min_toothpicks_no_triangles_l206_206846


namespace speed_of_stream_l206_206398

theorem speed_of_stream (v : ℝ) (h_still : ∀ (d : ℝ), d / (3 - v) = 2 * d / (3 + v)) : v = 1 :=
by
  sorry

end speed_of_stream_l206_206398


namespace range_of_a_l206_206934

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1 / 2) * x - 1 else 1 / x

theorem range_of_a : {a : ℝ | f a ≤ a} = {a : ℝ | -1 ≤ a ∧ a < 0} ∪ {a : ℝ | a ≥ 0} :=
by
  -- sorry is used to skip the proof
  sorry

end range_of_a_l206_206934


namespace shape_properties_l206_206406

theorem shape_properties
    (shape : Type) (line : shape -> shape -> Prop)
    (folds_and_coincides : ∀ (a b : shape), line a b → a = b) :
    (∀ x y : shape, x = y → congruent x y) ∧ 
    (∃ s : shape → Prop, ∀ x : shape, s x → symmetrical x) ∧
    (∃ l : shape → shape → Prop, ∀ x y : shape, l x y → axis_of_symmetry x y) ∧
    (∃ p : shape → shape → Prop, ∀ x y : shape, p x y → symmetrical_points x y) :=
by
  sorry

end shape_properties_l206_206406


namespace find_a_b_find_range_of_x_l206_206227

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (Real.log x / Real.log 2)^2 - 2 * a * (Real.log x / Real.log 2) + b

theorem find_a_b (a b : ℝ) :
  (f (1/4) a b = -1) → (a = -2 ∧ b = 3) :=
by
  sorry

theorem find_range_of_x (a b : ℝ) :
  a = -2 → b = 3 →
  ∀ x : ℝ, (f x a b < 0) → (1/8 < x ∧ x < 1/2) :=
by
  sorry

end find_a_b_find_range_of_x_l206_206227


namespace problem_inequality_l206_206616

variable {n : ℕ}
variables {a b c : Fin n → ℝ}

theorem problem_inequality
  (h_pos : ∀ i, a i > 0 ∧ b i > 0 ∧ c i > 0)
  (h_cond : ∀ i, a i * b i - c i ^ 2 > 0) :
  (n ^ 3) 
  / 
  (∑ i, a i * ∑ i, b i - (∑ i, c i) ^ 2)
  ≤ 
  ∑ i, 1 / (a i * b i - c i ^ 2) := 
sorry

end problem_inequality_l206_206616


namespace remainder_3_pow_89_plus_5_mod_7_l206_206369

theorem remainder_3_pow_89_plus_5_mod_7 :
  (3^1 % 7 = 3) ∧ (3^2 % 7 = 2) ∧ (3^3 % 7 = 6) ∧ (3^4 % 7 = 4) ∧ (3^5 % 7 = 5) ∧ (3^6 % 7 = 1) →
  ((3^89 + 5) % 7 = 3) :=
by
  intros h
  sorry

end remainder_3_pow_89_plus_5_mod_7_l206_206369


namespace volume_of_smaller_cone_l206_206702

-- Definitions, variables, and conditions
variables {V : ℝ} {α : ℝ}

-- Geometry setup and volume relationship
def V1 : ℝ := (V / cos(α / 2)^2)

-- Volume of the smaller cone to prove
def V2 : ℝ := V * tan(α / 2)^2

-- Lean statement to prove the volume of the smaller cone
theorem volume_of_smaller_cone (h : V1 - V = V) : V2 = V * tan(α / 2)^2 :=
by
  skip -- Proof of this theorem is skipped
  sorry

end volume_of_smaller_cone_l206_206702


namespace sequence_inequality_l206_206945

theorem sequence_inequality (a : ℕ → ℕ) 
  (h_nonneg : ∀ n, 0 ≤ a n)
  (h_additive : ∀ m n, a (n + m) ≤ a n + a m) 
  (N n : ℕ) 
  (h_N_ge_n : N ≥ n) : 
  a n + a N ≤ n * a 1 + N / n * a n :=
sorry

end sequence_inequality_l206_206945


namespace arc_length_calculation_l206_206765

noncomputable def arc_length_eq {t : ℝ} (t_lb t_ub : ℝ) : ℝ :=
  ∫ t_lb..t_ub, 4 * (Real.sqrt (2 * (1 - Real.cos t)))

theorem arc_length_calculation :
  arc_length_eq 0 (Real.pi / 3) = 8 * (2 - Real.sqrt 3) :=
by
  sorry

end arc_length_calculation_l206_206765


namespace complex_number_second_quadrant_l206_206130

theorem complex_number_second_quadrant (z : ℂ) (h : (1 - complex.I) * z = 3 + 5 * complex.I) : 
  ∃ (a b : ℝ), z = a + b * complex.I ∧ a < 0 ∧ b > 0 :=
by
  sorry

end complex_number_second_quadrant_l206_206130


namespace angle_HAD_half_diff_angle_BC_l206_206994

theorem angle_HAD_half_diff_angle_BC
  (ABC : Triangle)
  (A B C : Point)
  (H D : Point)
  (h1 : B ≠ C)
  (h2 : isAltitute A H B C)
  (h3 : isAngleBisector A D B C) :
  angle_between A H D = (angle_between B A C - angle_between C A B) / 2 :=
begin
  sorry
end

end angle_HAD_half_diff_angle_BC_l206_206994


namespace no_real_roots_of_quadratic_l206_206470

theorem no_real_roots_of_quadratic (k : ℝ) (h : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0 :=
by sorry

end no_real_roots_of_quadratic_l206_206470


namespace problem_correct_answer_l206_206418

theorem problem_correct_answer : 
  (let p3 := ∀ x : ℝ, (-0.5 * (x + 1) + 3 = (-0.5 * x + 3) - 0.5) in
  let p4 := ∀ x : ℝ, |sin (x + 1) + π| = |sin x + 1| in
  (p3 ∧ p4)) :=
by
  sorry

end problem_correct_answer_l206_206418


namespace line_intersects_xy_plane_at_point_l206_206091

/-- 
Proof that the point where the line passing through (2, 3, -1) and (4, -1, 3) 
intersects the xy-plane is (2.5, 2, 0).
-/
theorem line_intersects_xy_plane_at_point :
  let p1 := (2 : ℝ, 3 : ℝ, -1 : ℝ)
  let p2 := (4 : ℝ, -1 : ℝ, 3 : ℝ)
  let direction := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)
  let line := λ t : ℝ, (p1.1 + t * direction.1, p1.2 + t * direction.2, p1.3 + t * direction.3)
  ∃ t : ℝ, (line t).2 = 0 ∧ (line t) = (2.5, 2, 0) :=
by
  sorry

end line_intersects_xy_plane_at_point_l206_206091


namespace distance_to_other_focus_l206_206484

-- Definitions
def hyperbola_equation (x y : ℝ) : Prop := y^2 - 4 * x^2 = 16
def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Given preconditions
variables {P : ℝ × ℝ} (hP : hyperbola_equation P.1 P.2)
variable (dPF1 : ℝ) (dPF2 : ℝ)
variable (h_dist : dPF1 = 2)
def foci_distance : ℝ := real.sqrt (4 + 4 * 5)

-- Prove the distance to the other focus
theorem distance_to_other_focus (P : ℝ × ℝ)
  (hx1 : hyperbola_equation P.1 P.2)
  (dPF1 : real.sqrt ((P.1 - 2*real.sqrt(5))^2 + P.2^2) = 2) :
  real.sqrt ((P.1 + 2*real.sqrt(5))^2 + P.2^2) = 10 :=
sorry

end distance_to_other_focus_l206_206484


namespace intersection_A_B_union_A_B_complement_intersection_l206_206495

def setA : Set ℝ := { x | x + 2 ≥ 0 }
def setB : Set ℝ := { x | (x - 1) / (x + 1) ≥ 2 }
def setComplementA : Set ℝ := { x | ¬(x + 2 ≥ 0) }

theorem intersection_A_B : setA ∩ setB = Ico (-2:ℝ) (-1) :=
by
  -- proof steps
  sorry

theorem union_A_B : setA ∪ setB = Ici (-3:ℝ) :=
by
  -- proof steps
  sorry

theorem complement_intersection : setComplementA ∩ setB = Ico (-3:ℝ) (-2) :=
by
  -- proof steps
  sorry

end intersection_A_B_union_A_B_complement_intersection_l206_206495


namespace proof_1_proof_2_l206_206298

noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 0 then x^2 + (4*a - 3)*x + 3*a else log a (x + 1) + 1

def p (a : ℝ) : Prop :=
  monotone_dec (f a)

def q (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x → x ≤ real.sqrt 2 / 2 → x^2 - a ≤ 0

theorem proof_1 (a : ℝ) (h : q a) : a ≥ 1 / 2 :=
sorry

theorem proof_2 (a : ℝ) (p_true : p a) (q_false : ¬ q a ∨ q a) : 1 / 3 ≤ a ∧ a < 1 / 2 ∨ a > 3 / 4 :=
sorry

end proof_1_proof_2_l206_206298


namespace train_speed_l206_206802

theorem train_speed
    (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
    (h_train : length_train = 250)
    (h_platform : length_platform = 250.04)
    (h_time : time_seconds = 25) :
    (length_train + length_platform) / time_seconds * 3.6 = 72.006 :=
by sorry

end train_speed_l206_206802


namespace principal_amount_calculation_l206_206456

variables (P : ℝ) (r : ℝ) (t : ℝ) (n : ℕ) 
variables (compound_interest : ℝ) (final_amount : ℝ)

def principal_amount := P

def compound_interest_formula (P r t n : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def final_amount_formula := principal_amount + compound_interest

theorem principal_amount_calculation :
  r = 0.15 ∧
  t = 7 / 3 ∧
  n = 1 ∧
  compound_interest = 3886.25 ∧
  final_amount = principal_amount + compound_interest ∧
  compound_interest_formula principal_amount r t n = final_amount 
  → principal_amount = 9600.00 :=
sorry

end principal_amount_calculation_l206_206456


namespace instantaneous_velocity_at_1_l206_206980

variable (t : ℝ)

noncomputable def S (t : ℝ) : ℝ := 2 * t^2 + t

theorem instantaneous_velocity_at_1 : (derivative S 1) = 5 :=
by
  sorry

end instantaneous_velocity_at_1_l206_206980


namespace count_doublyOddlyPowerful_lt_3050_l206_206437

def doublyOddlyPowerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), b > 1 ∧ odd b ∧ odd a ∧ a^b = n

theorem count_doublyOddlyPowerful_lt_3050 : 
  (Finset.filter (λ n, doublyOddlyPowerful n) (Finset.range 3050)).card = 8 := 
by
  sorry

end count_doublyOddlyPowerful_lt_3050_l206_206437


namespace find_smallest_n_l206_206451

noncomputable def P (n : Nat) : Real :=
  (∏ k in Finset.range (n - 1), (k ^ 2 : Real) / (k ^ 2 + 1)) * (1 / (n ^ 2 + 1))

theorem find_smallest_n : ∃ (n : ℕ), P(n) < 1 / 51 ∧ ∀ m < n, P(m) ≥ 1 / 51 :=
by {
  sorry
}

end find_smallest_n_l206_206451


namespace diagonal_length_square_l206_206172

variable (ABCD : Type)
variable [metric_space : metric_space ABCD]
variable [is_square : is_square ABCD]
variable (side : ℝ)
variable (perimeter : ℝ)

noncomputable def length_of_diagonal (AC : ℝ) : Prop :=
  perimeter = 24 ∧ is_square → AC = 6 * real.sqrt 2

theorem diagonal_length_square (h : perimeter = 24) (hsq : is_square) :
  length_of_diagonal ABCD
:= sorry

end diagonal_length_square_l206_206172


namespace min_value_expression_l206_206099

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 10 + 6 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (a + 2 * y = 1) → ( (y^2 + a + 1) / (a * y)  ≥  c )) :=
sorry

end min_value_expression_l206_206099


namespace hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l206_206818

theorem hexagon_exists_equal_sides_four_equal_angles : 
  ∃ (A B C D E F : Type) (AB BC CD DE EF FA : ℝ) (angle_A angle_B angle_C angle_D angle_E angle_F : ℝ), 
  (AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB) ∧ 
  (angle_A = angle_B ∧ angle_B = angle_E ∧ angle_E = angle_F) ∧ 
  4 * angle_A + angle_C + angle_D = 720 :=
sorry

theorem hexagon_exists_equal_angles_four_equal_sides :
  ∃ (A B C D E F : Type) (AB BC CD DA : ℝ) (angle : ℝ), 
  (angle_A = angle_B ∧ angle_B = angle_C ∧ angle_C = angle_D ∧ angle_D = angle_E ∧ angle_E = angle_F ∧ angle_F = angle_A) ∧ 
  (AB = BC ∧ BC = CD ∧ CD = DA) :=
sorry

end hexagon_exists_equal_sides_four_equal_angles_hexagon_exists_equal_angles_four_equal_sides_l206_206818


namespace garden_length_l206_206016

theorem garden_length (P B : ℕ) (hP : P = 600) (hB : B = 150) : ∃ L : ℕ, 2 * (L + B) = P ∧ L = 150 :=
by
  existsi 150
  simp [hP, hB]
  sorry

end garden_length_l206_206016


namespace heap_division_impossible_l206_206243

def similar_sizes (a b : Nat) : Prop := a < 2 * b

theorem heap_division_impossible (n k : Nat) (h1 : n = 660) (h2 : k = 31) 
    (h3 : ∀ p1 p2 : Nat, p1 ∈ (Finset.range k).to_set → p2 ∈ (Finset.range k).to_set → similar_sizes p1 p2 → p1 < 2 * p2) : 
    False := 
by
  sorry

end heap_division_impossible_l206_206243


namespace unrealistic_data_l206_206031

theorem unrealistic_data :
  let A := 1000
  let A1 := 265
  let A2 := 51
  let A3 := 803
  let A1U2 := 287
  let A2U3 := 843
  let A1U3 := 919
  let A1I2 := A1 + A2 - A1U2
  let A2I3 := A2 + A3 - A2U3
  let A3I1 := A3 + A1 - A1U3
  let U := A1 + A2 + A3 - A1I2 - A2I3 - A3I1
  let A1I2I3 := A - U
  A1I2I3 > A2 :=
by
   sorry

end unrealistic_data_l206_206031


namespace prob_at_least_one_face_or_ace_l206_206775

open Classical

-- Defining the standard deck, including face cards and aces.
def standard_deck := fin 52
def is_face_or_ace (card : standard_deck) : Prop := card.val % 13 ∈ { 0, 10, 11, 12 }

-- Defining the event probability calculation.
def prob_face_or_ace_event :=
  let p_face_or_ace : ℚ := 16 / 52
  let p_not_face_or_ace : ℚ := 36 / 52
  1 - p_not_face_or_ace * p_not_face_or_ace

-- The theorem statement.
theorem prob_at_least_one_face_or_ace : prob_face_or_ace_event = 88 / 169 :=
by
  sorry

end prob_at_least_one_face_or_ace_l206_206775


namespace radius_first_circle_l206_206335

theorem radius_first_circle
  (A B C D E : Type)
  [MetricSpace B]
  (d : ℝ)
  (h1 : Segment A B = Diameter Circle1)
  (h2 : Center Circle2 = B)
  (h3 : radius Circle2 = 2)
  (h4 : intersects Circle1 Circle2 C)
  (h5 : Chord.Cons Circle2 CE: Chord CE = 3)
  (h6 : tangent_to Circle1 CE) :
  radius Circle1 = 4 / Real.sqrt 7 :=
by
  sorry

end radius_first_circle_l206_206335


namespace part1_part2_part3_l206_206149

universe u

def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12 * x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}
def CR_A : Set ℝ := {x | x < -3 ∨ x ≥ 7}

theorem part1 : A ∪ B = {x | -3 ≤ x ∧ x < 10} := by
  sorry

theorem part2 : CR_A ∩ B = {x | 7 ≤ x ∧ x < 10} := by
  sorry

theorem part3 (a : ℝ) (h : (A ∩ C a).Nonempty) : a > -3 := by
  sorry

end part1_part2_part3_l206_206149


namespace number_of_possible_values_of_b_l206_206668

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206668


namespace isosceles_triangle_bisector_intersection_l206_206693

theorem isosceles_triangle_bisector_intersection
  (α : ℝ)
  (A B C A1 B1 C1 : Type)
  [normed_add_comm_group A] [normed_add_comm_group B] [normed_add_comm_group C]
  [normed_add_comm_group A1] [normed_add_comm_group B1] [normed_add_comm_group C1]
  [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C]
  [normed_space ℝ A1] [normed_space ℝ B1] [normed_space ℝ C1]
  (angle_ratio : 1:2:4 → Type)
  (angle_BAC : ∠BAC = α)
  (angle_ABC : ∠ABC = 2 * α)
  (angle_ACB : ∠ACB = 4 * α) :
  ∃ (C1A1 = C1B1), sorry

end isosceles_triangle_bisector_intersection_l206_206693


namespace volume_ratio_l206_206043

def volume_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem volume_ratio :
  let r_b := 6  -- radius of Ben's container
      h_b := 24 -- height of Ben's container
      r_l := 12 -- radius of Lisa's container
      h_l := 24 -- height of Lisa's container
      V_b := volume_cylinder r_b h_b
      V_l := volume_cylinder r_l h_l
  in V_b / V_l = 1 / 4 :=
by
  let r_b := 6
  let h_b := 24
  let r_l := 12
  let h_l := 24
  let V_b := volume_cylinder r_b h_b
  let V_l := volume_cylinder r_l h_l
  have h1: V_b = 864 * Real.pi := by sorry
  have h2: V_l = 3456 * Real.pi := by sorry
  have h3: V_b / V_l = (864 * Real.pi) / (3456 * Real.pi) := by sorry
  have h4: (864 * Real.pi) / (3456 * Real.pi) = 1 / 4 := by sorry
  exact h4

end volume_ratio_l206_206043


namespace circle_problems_satisfy_conditions_l206_206572

noncomputable def circle1_center_x := 11
noncomputable def circle1_center_y := 8
noncomputable def circle1_radius_squared := 87

noncomputable def circle2_center_x := 14
noncomputable def circle2_center_y := -3
noncomputable def circle2_radius_squared := 168

theorem circle_problems_satisfy_conditions :
  (∀ x y, (x-11)^2 + (y-8)^2 = 87 ∨ (x-14)^2 + (y+3)^2 = 168) := sorry

end circle_problems_satisfy_conditions_l206_206572


namespace accounting_majors_count_l206_206181

theorem accounting_majors_count (p q r s t u : ℕ) 
  (h_eq : p * q * r * s * t * u = 51030)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) : 
  p = 2 :=
sorry

end accounting_majors_count_l206_206181


namespace paintbrush_ratio_l206_206789

theorem paintbrush_ratio (s w : ℝ) 
    (h_square_area : s^2 = s^2)
    (h_painted_area : w^2 + (s - w)^2 / 2 = (1 : ℝ) / 3 * s^2) :
    s / w = 2 * real.sqrt 3 - 2 :=
by
  sorry

end paintbrush_ratio_l206_206789


namespace sum_of_infinite_series_l206_206464

theorem sum_of_infinite_series :
  ∑' n, (1 : ℝ) / ((2 * n + 1)^2 - (2 * n - 1)^2) * ((1 : ℝ) / (2 * n - 1)^2 - (1 : ℝ) / (2 * n + 1)^2) = 1 :=
sorry

end sum_of_infinite_series_l206_206464


namespace impossible_division_l206_206260

theorem impossible_division (heap_size : ℕ) (heap_count : ℕ) (similar_sizes : Π (a b : ℕ), a < 2 * b) :
  ¬ (∃ (heaps : Finset ℕ), heaps.card = heap_count ∧ heaps.sum = heap_size ∧ ∀ (a b ∈ heaps), similar_sizes a b) :=
by
  let heap_size := 660
  let heap_count := 31
  sorry

end impossible_division_l206_206260


namespace paint_required_for_small_statues_l206_206165

-- Constants definition
def pint_per_8ft_statue : ℕ := 1
def height_original_statue : ℕ := 8
def height_small_statue : ℕ := 2
def number_of_small_statues : ℕ := 400

-- Theorem statement
theorem paint_required_for_small_statues :
  pint_per_8ft_statue = 1 →
  height_original_statue = 8 →
  height_small_statue = 2 →
  number_of_small_statues = 400 →
  (number_of_small_statues * (pint_per_8ft_statue * (height_small_statue / height_original_statue)^2)) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_required_for_small_statues_l206_206165


namespace student_marks_math_l206_206408

-- Conditions from part a)
def average_three_subjects := 85
def average_physics_other := 90
def average_physics_chemistry := 70
def physics_marks := 65

-- Correct answer from part b)
def mathematics_marks := 115

-- Lean theorem statement
theorem student_marks_math :
  let P := physics_marks
  let A := average_three_subjects
  let T := 3 * A
  let X := 2 * average_physics_other - P
  let C := 2 * average_physics_chemistry - P
  P + C + X = T -> X = 115 :=
by
  intros P A T X C h
  have h1 : T = 3 * 85, from rfl
  rw [←h1] at h
  simp only [physics_marks, average_physics_other, average_physics_chemistry] at h
  sorry

end student_marks_math_l206_206408


namespace range_of_m_l206_206932

variables {a c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 - 2 * a * x + c

theorem range_of_m (h : f 2017 < f (-2016)) : {m : ℝ | f m ≤ f 0} = set.Icc 0 2 :=
begin
  sorry
end

end range_of_m_l206_206932


namespace minimum_value_expr_l206_206883

theorem minimum_value_expr 
    (a b : ℝ)
    (h1 : a > b)
    (h2 : ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
    (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + b = 0) :
  (∃ c : ℝ, ∀ a b, c = min ( (a^2 + b^2)/(a - b)) (2 * real.sqrt 2)) := sorry

end minimum_value_expr_l206_206883


namespace three_terms_not_geometric_sequence_l206_206717

def a_seq (n : ℕ) : ℝ := 2 * n - 1 + Real.sqrt 2
def S_n (n : ℕ) : ℝ := n * (n + Real.sqrt 2)
def b_seq (n : ℕ) : ℝ := S_n n / n

theorem three_terms_not_geometric_sequence (p q r : ℕ) (hp : p ≠ q) (hq : q ≠ r) (hr : r ≠ p) :
  ¬ ((b_seq q) * (b_seq q) = (b_seq p) * (b_seq r)) :=
  sorry

end three_terms_not_geometric_sequence_l206_206717


namespace medians_of_groups_are_correct_l206_206953

-- Group A and Group B definitions
def groupA : List Int := [28, 31, 39, 42, 45, 55, 57, 58, 66]
def groupB : List Int := [29, 34, 35, 42, 46, 48, 53, 55, 55, 67]

-- Definition for median
def median (l : List Int) : Int :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 1 then
    sorted[sorted.length / 2]
  else
    (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2

-- Theorem statement to prove
theorem medians_of_groups_are_correct :
    median groupA = 45 ∧ median groupB = 47 := 
by
  sorry

end medians_of_groups_are_correct_l206_206953


namespace last_two_nonzero_digits_70_l206_206436

noncomputable def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let fac := nat.factorial n
  fac.div (10 ^ (fac.factors 10)).factorial % 100

theorem last_two_nonzero_digits_70 :
  last_two_nonzero_digits 70 = 12 := -- Assuming 12 is the correct answer, replace 12 with the actual.
sorry

end last_two_nonzero_digits_70_l206_206436


namespace determine_b_l206_206069

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iio 2 ∪ Set.Ioi 6 → -x^2 + b * x - 7 < 0) ∧ 
  (∀ x : ℝ, ¬(x ∈ Set.Iio 2 ∪ Set.Ioi 6) → ¬(-x^2 + b * x - 7 < 0)) → 
  b = 8 :=
sorry

end determine_b_l206_206069


namespace deg_to_rad_rad_to_deg_l206_206442

-- Theorem 1: Conversion from degrees to radians
theorem deg_to_rad (h : 180 = Real.pi) : 210 = 7 * Real.pi / 6 :=
by 
  have one_deg : 1 = Real.pi / 180 := by rw [←h]; exact Real.div_one Real.pi
  calc 
    210 
    = 210 * (Real.pi / 180) : by rw one_deg
    ... = 210 * Real.pi / 180 : by ring
    ... = 7 * Real.pi / 6 : by norm_num

-- Theorem 2: Conversion from radians to degrees
theorem rad_to_deg (h : 180 = Real.pi) : -5 * Real.pi / 2 = -450 :=
by 
  have one_rad : 1 = 180 / Real.pi := by rw [←h]; exact Real.div_one 180
  calc 
    -5 * Real.pi / 2
    = -5 / 2 * 180 : by rw one_rad
    ... = -450 : by norm_num


end deg_to_rad_rad_to_deg_l206_206442


namespace definite_integral_example_l206_206073

theorem definite_integral_example : ∫ x in (0 : ℝ)..(π/2), 2 * x = π^2 / 4 := 
by 
  sorry

end definite_integral_example_l206_206073


namespace solve_for_x_l206_206194

theorem solve_for_x (x : ℚ) (h : 1/4 + 7/x = 13/x + 1/9) : x = 216/5 :=
by
  sorry

end solve_for_x_l206_206194


namespace distance_from_home_to_high_school_l206_206600

theorem distance_from_home_to_high_school 
  (total_mileage track_distance d : ℝ)
  (h_total_mileage : total_mileage = 10)
  (h_track : track_distance = 4)
  (h_eq : d + d + track_distance = total_mileage) :
  d = 3 :=
by sorry

end distance_from_home_to_high_school_l206_206600


namespace circumcircle_tangent_incircle_l206_206421

theorem circumcircle_tangent_incircle
  {A B C D E F M N P Q X : Point}
  (h1 : incircle_triangle A B C touches_bc D)
  (h2 : incircle_triangle A B C touches_ca E)
  (h3 : incircle_triangle A B C touches_ab F)
  (h4 : midpoint M B F)
  (h5 : midpoint N B D)
  (h6 : midpoint P C E)
  (h7 : midpoint Q C D)
  (h8 : intersect MN PQ X) :
  tangent (circumcircle X B C) (incircle A B C) :=
by sorry

end circumcircle_tangent_incircle_l206_206421


namespace points_in_quadrants_l206_206716

theorem points_in_quadrants (x y : ℝ) (h₁ : y > 3 * x) (h₂ : y > 6 - x) : 
  (0 <= x ∧ 0 <= y) ∨ (x <= 0 ∧ 0 <= y) :=
by
  sorry

end points_in_quadrants_l206_206716


namespace square_area_l206_206037

theorem square_area (x : ℚ) (side_length : ℚ) 
  (h1 : side_length = 3 * x - 12) 
  (h2 : side_length = 24 - 2 * x) : 
  side_length ^ 2 = 92.16 := 
by 
  sorry

end square_area_l206_206037


namespace ratio_of_volumes_l206_206018

-- Definitions based on the conditions
def side_a : ℝ := 3
def side_b : ℝ := 6
def height : ℝ := 9
def radius : ℝ := side_a / 2
def volume_prism : ℝ := side_a * side_b * height
def volume_cone : ℝ := (1 / 3) * Real.pi * radius^2 * height

-- Proof problem statement
theorem ratio_of_volumes : volume_cone / volume_prism = Real.pi / 24 := 
  by
  -- Place proof body here
  sorry

end ratio_of_volumes_l206_206018


namespace range_of_f_l206_206935

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then x^2 else x + 4/x - 3

theorem range_of_f : Set.range f = Set.Ici 0 := sorry

end range_of_f_l206_206935


namespace area_ratio_of_extended_isosceles_triangles_l206_206603

theorem area_ratio_of_extended_isosceles_triangles
    (A B C B' C' B'' C'' : Type)
    [Triangle A B C]
    (isosceles : AB = AC)
    (extension_BB' : BB' = 2 * AB)
    (extension_CC' : CC' = 2 * AC)
    (extension_BC'' : BB'' = 2 * BC)
    (extension_CC'' : CC'' = 2 * BC)
    : area (Triangle B'' B C'') / area (Triangle A B C) = 36 := 
sorry

end area_ratio_of_extended_isosceles_triangles_l206_206603


namespace impossibility_of_dividing_stones_l206_206280

theorem impossibility_of_dividing_stones (stones : ℕ) (heaps : ℕ) (m : ℕ) :
  stones = 660 → heaps = 31 → (∀ (x y : ℕ), x ∈ heaps → y ∈ heaps → x < 2 * y → False) := sorry

end impossibility_of_dividing_stones_l206_206280


namespace inscribed_rectangle_λ_l206_206014

theorem inscribed_rectangle_λ (a b d λ m : ℝ)
  (h0 : 0 ≤ λ)
  (h1 : λ ≤ b)
  (h2 : 0 < m)
  (h3 : m < d / (a + b)) :
  λ = (a + d * m) * (a * m + b * m - d) / (2 * a * m + d * (m^2 - 1)) :=
sorry

end inscribed_rectangle_λ_l206_206014


namespace tangent_parallel_x_axis_tangent_45_degrees_x_axis_l206_206042

-- Condition: Define the curve
def curve (x : ℝ) : ℝ := x^2 - 1

-- Condition: Calculate derivative
def derivative_curve (x : ℝ) : ℝ := 2 * x

-- Part (a): Point where tangent is parallel to the x-axis
theorem tangent_parallel_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 0 ∧ x = 0 ∧ y = -1) :=
  sorry

-- Part (b): Point where tangent forms a 45 degree angle with the x-axis
theorem tangent_45_degrees_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 1 ∧ x = 1/2 ∧ y = -3/4) :=
  sorry

end tangent_parallel_x_axis_tangent_45_degrees_x_axis_l206_206042


namespace number_of_divisors_of_24_divisible_by_4_l206_206654

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206654


namespace value_of_c_l206_206312

-- Define a structure representing conditions of the problem
structure ProblemConditions where
  c : Real

-- Define the problem in terms of given conditions and required proof
theorem value_of_c (conditions : ProblemConditions) : conditions.c = 5 / 2 := by
  sorry

end value_of_c_l206_206312


namespace no_real_solutions_l206_206969

theorem no_real_solutions : ∀ x : ℝ, (3 * x - 4) ^ 2 + 3 ≠ -2 * |x - 1| :=
by
  intro x
  have h1 : (3 * x - 4) ^ 2 + 3 ≥ 3 :=
    calc
      (3 * x - 4) ^ 2 + 3 ≥ 0 + 3 : by apply add_le_add_right (pow_two_nonneg (3 * x - 4)) 3
      _ = 3 : by rw zero_add
  have h2 : -2 * |x - 1| ≤ 0 := by nlinarith [abs_nonneg (x - 1)]
  linarith

end no_real_solutions_l206_206969


namespace scientific_notation_of_trade_volume_l206_206727

-- Define the total trade volume
def total_trade_volume : ℕ := 175000000000

-- Define the expected scientific notation result
def expected_result : ℝ := 1.75 * 10^11

-- Theorem stating the problem
theorem scientific_notation_of_trade_volume :
  (total_trade_volume : ℝ) = expected_result := by
  sorry

end scientific_notation_of_trade_volume_l206_206727


namespace relationship_of_angles_l206_206824

theorem relationship_of_angles (P Q R X Y Z: Type)
  (h_isosceles: PQ = PR)
  (h_right_triangle: ∃(X Y Z: Type), ¬ (XY = XZ ∧ YZ = YZ))
  (h_on_pq: X ∈ PQ)
  (h_on_pr: Y ∈ PR)
  (h_on_qr: Z ∈ QR)
  (alpha beta gamma: ℝ)
  (h_alpha: α = ∠XZY)
  (h_beta: β = ∠ZYX)
  (h_gamma: γ = ∠YXZ):
  γ = α + β := by
  sorry

end relationship_of_angles_l206_206824


namespace unique_zero_when_m_is_neg2_l206_206931

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4^x + m * 2^x + 1

theorem unique_zero_when_m_is_neg2 (m : ℝ) :
  ∃! x : ℝ, f x (-2) = 0 :=
begin
  sorry
end

end unique_zero_when_m_is_neg2_l206_206931


namespace problem_part1_problem_part2_l206_206608

open Set Real

-- Define the sets and condition for B
def A : Set ℝ := univ
def B (a : ℝ) : Set ℝ := {x | (a-2) * x^2 + 2 * (a-2) * x - 3 < 0}

theorem problem_part1 (a : ℝ) (h : a = 3) : B a = Ioo (-3) 1 :=
by
  rw [h, B]
  have h' : (3 - 2) = 1 := by norm_num
  rw [h']
  ext x
  simp only [mem_set_of_eq, Ioo, lt_iff_le_and_ne]
  sorry

theorem problem_part2 (a : ℝ) (h : A = B a) : -1 < a ∧ a ≤ 2 :=
by
  rw [h, A, B]

  split
  {
    intro h'
    have : a - 2 = 0 := by linarith
    linarith
  }
  {
    intro h'
    apply lt_neg_one_or_one_le
    apply and.intro
    {
      intro h''
      apply lt_neg_one_of_eq_neg
      sorry
    }
    {
      intro h''
      apply one_le_of_eq
      sorry
    }
  }

end problem_part1_problem_part2_l206_206608


namespace f_is_even_g_is_odd_l206_206978

-- Definitions of the two functions f and g
def f (x : ℝ) : ℝ := 3^x + 3^(-x)
def g (x : ℝ) : ℝ := 3^x - 3^(-x)

-- Theorem stating that f is even and g is odd
theorem f_is_even_g_is_odd : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) :=
by
  sorry

end f_is_even_g_is_odd_l206_206978


namespace smallest_positive_integer_N_l206_206871

theorem smallest_positive_integer_N :
  ∃ N : ℕ, N > 0 ∧ (N % 7 = 5) ∧ (N % 8 = 6) ∧ (N % 9 = 7) ∧ (∀ M : ℕ, M > 0 ∧ (M % 7 = 5) ∧ (M % 8 = 6) ∧ (M % 9 = 7) → N ≤ M) :=
sorry

end smallest_positive_integer_N_l206_206871


namespace compute_c_minus_d_cubed_l206_206610

-- define c as the number of positive multiples of 12 less than 60
def c : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

-- define d as the number of positive integers less than 60 and a multiple of both 3 and 4
def d : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

theorem compute_c_minus_d_cubed : (c - d)^3 = 0 := by
  -- since c and d are computed the same way, (c - d) = 0
  -- hence, (c - d)^3 = 0^3 = 0
  sorry

end compute_c_minus_d_cubed_l206_206610


namespace intersection_A_B_l206_206561

open Set

-- Define the universal set as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {-1, 0, 1, 2, 3, 4, 5, 6}

-- Define set B as natural numbers less than 4
def B : Set ℕ := { x ∈ ℕ | x < 4 }

-- Theorem to prove the intersection of A and B
theorem intersection_A_B : A ∩ B = {0, 1, 2, 3} :=
by
  sorry

end intersection_A_B_l206_206561


namespace necessarily_positive_b_plus_3c_l206_206640

theorem necessarily_positive_b_plus_3c 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  b + 3 * c > 0 := 
sorry

end necessarily_positive_b_plus_3c_l206_206640


namespace sum_of_coordinates_l206_206924

def g : ℝ → ℝ := sorry

theorem sum_of_coordinates :
  g 3 = 10 →
  let y := (46 / 2) in
  let x := 1 in
  x + y = 24 :=
begin
  intro h,
  have hy : y = 23,
  { change 46 / 2 = 23, norm_num },
  change 1 + y = 24,
  rw hy,
  norm_num,
end

end sum_of_coordinates_l206_206924


namespace range_of_a_l206_206902

theorem range_of_a (a : ℝ) (x : ℝ) :
  (¬(x > a) →¬(x^2 + 2*x - 3 > 0)) → (a ≥ 1 ) :=
by
  intro h
  sorry

end range_of_a_l206_206902


namespace egyptian_fraction_l206_206302

theorem egyptian_fraction (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) : 
  (2 : ℚ) / 7 = (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c :=
by
  sorry

end egyptian_fraction_l206_206302


namespace total_amount_shared_l206_206174

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.2 * z) (h3 : z = 400) :
  x + y + z = 1480 :=
by
  sorry

end total_amount_shared_l206_206174


namespace cannot_achieve_goal_l206_206343

noncomputable def jar_capacity : ℕ := 1000 -- 1 liter in mL

inductive Jar
| first : Jar
| second : Jar
| third : Jar

structure State :=
(vol2 : ℕ) -- volume in the second jar
(sugar2 : ℕ) -- sugar in the second jar in grams
(vol3 : ℕ) -- volume in the third jar
(sugar3 : ℕ) -- sugar in the third jar in grams

def initial_state : State :=
{ vol2 := 700, sugar2 := 50, vol3 := 800, sugar3 := 60 }

def transfer (from to : Jar) (amount : ℕ) (s : State) : State :=
match from, to with
| Jar.second, Jar.third =>
  { s with vol2 := s.vol2 - amount, vol3 := s.vol3 + amount, 
           sugar2 := s.sugar2 - (amount * s.sugar2 / s.vol2), 
           sugar3 := s.sugar3 + (amount * s.sugar2 / s.vol2) }
| Jar.third, Jar.second =>
  { s with vol3 := s.vol3 - amount, vol2 := s.vol2 + amount, 
           sugar3 := s.sugar3 - (amount * s.sugar3 / s.vol3), 
           sugar2 := s.sugar2 + (amount * s.sugar3 / s.vol3) }
| _, _ => s
end

def is_goal_state (s : State) : Prop :=
s.sugar2 = s.sugar3

theorem cannot_achieve_goal : ¬ ∃ (s : State), (initial_state → s) ∧ is_goal_state s := sorry

end cannot_achieve_goal_l206_206343


namespace orlie_age_l206_206304

theorem orlie_age (O R : ℕ) (h1 : R = 9) (h2 : R = (3 * O) / 4)
  (h3 : R - 4 = ((O - 4) / 2) + 1) : O = 12 :=
by
  sorry

end orlie_age_l206_206304


namespace statement_A_statement_B_statement_C_statement_D_l206_206589

-- For all theorems, a triangle ABC with sides a, b, c and angles A, B, and C respectively.

-- Statement A
theorem statement_A (A B : ℝ) (h : A > B) : sin A > sin B :=
sorry

-- Statement B
theorem statement_B (A : ℝ) (a : ℝ) (hA : A = π / 6) (ha : a = 5) : 
  2 * a / sin A = 10 :=
by
  have hR : 2 * a / sin A = 2 * 5 / sin A := by rw [ha]
  have hπ : 2 * 5 / sin (π / 6) = 10 := by
    rw [sin_pi_div_six]
    norm_num
  rw [hA, hπ]
  sorry

-- Statement C
theorem statement_C (a b c : ℝ) (A B C : ℝ) (h : a = 2 * b * cos C) : 
  is_isosceles (A B C) :=
sorry

-- Statement D
theorem statement_D (a b c : ℝ) (A : ℝ) (hb : b = 1) (hc : c = 2) (hA : A = 2 * π / 3) : 
  area a b c A = sqrt 3 / 2 :=
by
  have h_area : 1 / 2 * b * c * sin A = 1 / 2 * 1 * 2 * sin (2 * π / 3) := by
    rw [hb, hc]
  rw [hA, sin_two_pi_div_three, mul_comm (1 / 2), mul_assoc, mul_comm (sqrt 3 / 2)] at h_area
  sorry

end statement_A_statement_B_statement_C_statement_D_l206_206589


namespace heap_division_impossible_l206_206242

def similar_sizes (a b : Nat) : Prop := a < 2 * b

theorem heap_division_impossible (n k : Nat) (h1 : n = 660) (h2 : k = 31) 
    (h3 : ∀ p1 p2 : Nat, p1 ∈ (Finset.range k).to_set → p2 ∈ (Finset.range k).to_set → similar_sizes p1 p2 → p1 < 2 * p2) : 
    False := 
by
  sorry

end heap_division_impossible_l206_206242


namespace solve_reflex_angle_at_H_l206_206881

noncomputable def reflex_angle_at_H (C D F M H : Point) (angle_CDH angle_HFM : ℝ) : ℝ :=
-- Provide or assume the context/geometry definitions if needed
if collinear C D F M ∧ angle_CDH = 130 ∧ angle_HFM = 70 then
  340  -- As derived in the equivalent problem
else
  0    -- Otherwise, it does not match the problem specific context

theorem solve_reflex_angle_at_H (C D F M H : Point) 
    (h_collinear : collinear C D F M) 
    (h_angle_CDH : ∠ CDH = 130) 
    (h_angle_HFM : ∠ HFM = 70) :
  reflex_angle_at_H C D F M H 130 70 = 340 := 
begin
  sorry
end

end solve_reflex_angle_at_H_l206_206881


namespace largest_n_l206_206833

theorem largest_n (n x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 6 * x + 6 * y + 6 * z - 18 →
  n ≤ 3 := 
by 
  sorry

end largest_n_l206_206833


namespace limit_of_sum_of_geometric_sequences_l206_206509

/-- 
Given sequences a_n and b_n are geometric with common ratios p and q respectively, 
where p > q and p ≠ 1, q ≠ 1. Define c_n = a_n + b_n and let S_n be the sum of the 
first n terms of the sequence c_n. Prove that the limit 
lim (n → ∞) (S_n / S_{n-1}) = p.
-/
theorem limit_of_sum_of_geometric_sequences 
  {a_n b_n : ℕ → ℝ} {a_1 b_1 p q : ℝ} (hpq : p > q) (hp1 : p ≠ 1) (hq1 : q ≠ 1) (hpos : ∀ n, a_n n > 0 ∧ b_n n > 0) 
  (ha : ∀ n, a_n n = a_1 * p^(n - 1)) (hb: ∀ n, b_n n = b_1 * q^(n - 1)):
  (tendsto (λ n, let S := (λ n, ∑ i in range n, a_n i + b_n i) in S n / S (n - 1)) at_top (𝓝 p)) := 
sorry

end limit_of_sum_of_geometric_sequences_l206_206509


namespace convex_polygon_obtuse_sum_l206_206541
open Int

def convex_polygon_sides (n : ℕ) (S : ℕ) : Prop :=
  180 * (n - 2) = 3000 + S ∧ (S = 60 ∨ S = 240)

theorem convex_polygon_obtuse_sum (n : ℕ) (hn : 3 ≤ n) :
  (∃ S, convex_polygon_sides n S) ↔ (n = 19 ∨ n = 20) :=
by
  sorry

end convex_polygon_obtuse_sum_l206_206541


namespace constant_term_binomial_expansion_l206_206475

noncomputable def a : ℝ := (2 / Real.pi) * (∫ x in -1..1, Real.sqrt (1 - x^2) + Real.sin x)

theorem constant_term_binomial_expansion :
  let expr : Polynomial ℝ := (Polynomial.X - Polynomial.C (a / Polynomial.X^2)) ^ 9 in
  expr.coeff 0 = -84 :=
sorry

end constant_term_binomial_expansion_l206_206475


namespace not_consecutive_l206_206724

theorem not_consecutive (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) : 
  ¬ (∃ n : ℕ, (2023 + a - b = n ∧ 2023 + b - c = n + 1 ∧ 2023 + c - a = n + 2) ∨ 
    (2023 + a - b = n ∧ 2023 + b - c = n - 1 ∧ 2023 + c - a = n - 2)) :=
by
  sorry

end not_consecutive_l206_206724


namespace function_strictly_increasing_on_intervals_l206_206457

noncomputable def f (x : ℝ) : ℝ := (Real.log (Real.cos (2 * x - Real.pi / 6))) / Real.log (1/2)

def strictly_increasing_intervals (k : ℤ) : Set ℝ :=
  Ico (↑k * Real.pi + Real.pi / 12) (↑k * Real.pi + Real.pi / 3)

theorem function_strictly_increasing_on_intervals :
  ∀ k : ℤ, StrictMonoOn f (strictly_increasing_intervals k) :=
by
  sorry

end function_strictly_increasing_on_intervals_l206_206457


namespace count_integers_with_inverse_mod_11_l206_206963

theorem count_integers_with_inverse_mod_11 : 
  (set.count (λ a, ∃ b, (a * b) % 11 = 1) (set.Icc 1 9)) = 9 := 
by
  sorry

end count_integers_with_inverse_mod_11_l206_206963


namespace converse_and_inverse_true_l206_206151

-- Define the propositions
def p : Prop := ∀ s, is_circle s → has_constant_curvature s
def q : Prop := ∀ s, has_constant_curvature s → is_circle s
def r : Prop := ∀ s, ¬ is_circle s → ¬ has_constant_curvature s

-- The original true statement
axiom original_statement : p

-- The converse and the inverse of the original statement
axiom converse : q
axiom inverse : r

-- Prove that both the converse and the inverse are true
theorem converse_and_inverse_true : q ∧ r := 
by 
  sorry

end converse_and_inverse_true_l206_206151


namespace log_equation_solution_l206_206167

theorem log_equation_solution (x y : ℝ) (h : (Real.logBase 3 x) * (Real.logBase x (2 * x)) * (Real.logBase (2 * x) y) = Real.logBase x (x^2)) : y = 9 := sorry

end log_equation_solution_l206_206167


namespace sum_squares_of_distances_constant_l206_206482

theorem sum_squares_of_distances_constant (O P: Point) (R : ℝ)
  (hO : circle_center_radius O R)
  (ins_rectangle : ∀ (A B C D: Point), inscribed_rectangle O A B C D) :
  ∀ (A B C D : Point), inscribed_rectangle O A B C D →
    dist2 P A + dist2 P B + dist2 P C + dist2 P D = 4 * (dist2 P O + R^2) := 
by
  -- Proof is skipped
  sorry

end sum_squares_of_distances_constant_l206_206482


namespace davids_test_scores_l206_206715

open Nat

theorem davids_test_scores :
  ∃ (scores : List ℕ),
    scores = [94, 92, 91, 87, 86] ∧
    scores.sum = 450 ∧
    ∀ s ∈ scores, s < 95 ∧
    scores.nodup ∧
    scores.head = 94 ∧
    (scores.tail.getD []).head = 92 ∧
    (scores.tail.getD []).tail.getD [].head = 91 ∧
    (scores.tail.getD []).tail.getD [].tail.getD [].head = 87 ∧
    (scores.tail.getD []).tail.getD [].tail.getD [].tail.getD [].head = 86 :=
by
  sorry

end davids_test_scores_l206_206715


namespace coprime_adjacent_numbers_exists_l206_206295

theorem coprime_adjacent_numbers_exists :
  ∃ (f : Fin 9 → Nat), 
    (∀ i, f i ∈ List.range 1 10) ∧
    Function.Injective f ∧
    (∀ i, Nat.gcd (f i) (f ((i + 1) % 9)) = 1) :=
by
  sorry

end coprime_adjacent_numbers_exists_l206_206295


namespace suff_not_necc_condition_l206_206769

theorem suff_not_necc_condition (x : ℝ) : (x=2) → ((x-2) * (x+5) = 0) ∧ ¬((x-2) * (x+5) = 0 → x=2) :=
by {
  sorry
}

end suff_not_necc_condition_l206_206769


namespace PP_le_n_l206_206831

def P (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 
    let factors := n.factors;
    factors.foldl (λ acc p, acc * (factors.count p) ^ p) 1

theorem PP_le_n : ∀ (n : ℕ), 0 < n → P (P n) ≤ n := by
  intros n hn
  sorry

end PP_le_n_l206_206831


namespace sum_of_solutions_eq_neg_two_l206_206314

open Complex

theorem sum_of_solutions_eq_neg_two :
  (∑ x in ({x : ℂ | 3^((x^2) + 4*x + 4) = 9^(x + 1)} : Set ℂ), x) = -2 :=
by
  sorry

end sum_of_solutions_eq_neg_two_l206_206314


namespace pyramid_prism_sum_l206_206581

-- Definitions based on conditions
structure Prism :=
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)

-- The initial cylindrical-prism object
noncomputable def initial_prism : Prism :=
  { vertices := 8,
    edges := 10,
    faces := 5 }

-- Structure for Pyramid Addition
structure PyramidAddition :=
  (new_vertices : ℕ)
  (new_edges : ℕ)
  (new_faces : ℕ)

noncomputable def pyramid_addition : PyramidAddition := 
  { new_vertices := 1,
    new_edges := 4,
    new_faces := 4 }

-- Function to add pyramid to the prism
noncomputable def add_pyramid (prism : Prism) (pyramid : PyramidAddition) : Prism :=
  { vertices := prism.vertices + pyramid.new_vertices,
    edges := prism.edges + pyramid.new_edges,
    faces := prism.faces - 1 + pyramid.new_faces }

-- The resulting prism after adding the pyramid
noncomputable def resulting_prism := add_pyramid initial_prism pyramid_addition

-- Proof problem statement
theorem pyramid_prism_sum : 
  resulting_prism.vertices + resulting_prism.edges + resulting_prism.faces = 31 :=
by sorry

end pyramid_prism_sum_l206_206581


namespace num_of_possible_values_b_l206_206686

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206686


namespace find_length_AC_l206_206587

-- Definitions for lengths BD, DE, and EC
def BD : ℝ := 1
def DE : ℝ := 3
def EC : ℝ := 5
def DC : ℝ := DE + EC -- 8
def trisectors_angle_BAC (A B C : Type) [IsTriangle A B C] : Prop :=
  ∃ D E : Type, Trisectors A B C D E

theorem find_length_AC {A B C : Type} [IsTriangle A B C]
  (D E : Type) (h1 : trisectors_angle_BAC A B C)
  (h2 : length BD = 1) (h3 : length DE = 3) (h4 : length EC = 5) :
  let AD := sqrt(19.5) in
  length (AC : Segment A C) = (5 / 3) * AD :=
sorry

end find_length_AC_l206_206587


namespace possible_values_b_count_l206_206675

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206675


namespace perfect_square_trinomial_m_value_l206_206913

theorem perfect_square_trinomial_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ y : ℤ, y^2 + my + 9 = (y + a)^2) ↔ (m = 6 ∨ m = -6) :=
by
  sorry

end perfect_square_trinomial_m_value_l206_206913


namespace max_handshakes_with_three_people_excluded_l206_206772

theorem max_handshakes_with_three_people_excluded (n : ℕ) (h: n = 27) :
  let total_possible_handshakes := nat.choose n 2,
      unmade_handshakes := nat.choose 3 2 in
  total_possible_handshakes - unmade_handshakes = 348 :=
by
  intros
  rw [h]
  have hp1: total_possible_handshakes = 351 := by decide
  have hp2: unmade_handshakes = 3 := by decide
  rw [hp1, hp2]
  decide

end max_handshakes_with_three_people_excluded_l206_206772


namespace expected_waiting_time_for_passenger_l206_206388

noncomputable def expected_waiting_time (arrival_time: ℕ) : ℝ := 
  if arrival_time < 20 * 60
    then (20 / 60) * (40 / 2) + (40 / 60) * (45) -- from 8:00 to 9:00 AM
    else 60 + 45 -- from 9:00 to 10:00 AM

theorem expected_waiting_time_for_passenger :
  expected_waiting_time (20 * 60) = 53.3 :=
sorry

end expected_waiting_time_for_passenger_l206_206388


namespace rate_of_drawing_barbed_wire_per_meter_l206_206696

-- Defining the given conditions as constants
def area : ℝ := 3136
def gates_width_total : ℝ := 2 * 1
def total_cost : ℝ := 666

-- Constants used for intermediary calculations
def side_length : ℝ := Real.sqrt area
def perimeter : ℝ := 4 * side_length
def barbed_wire_length : ℝ := perimeter - gates_width_total

-- Formulating the theorem
theorem rate_of_drawing_barbed_wire_per_meter : 
    (total_cost / barbed_wire_length) = 3 := 
by
  sorry

end rate_of_drawing_barbed_wire_per_meter_l206_206696


namespace number_of_divisors_of_24_divisible_by_4_l206_206650

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206650


namespace accounting_majors_count_l206_206182

theorem accounting_majors_count (p q r s t u : ℕ) 
  (h_eq : p * q * r * s * t * u = 51030)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) : 
  p = 2 :=
sorry

end accounting_majors_count_l206_206182


namespace triangle_ratio_AN_NC_l206_206987

theorem triangle_ratio_AN_NC (A B C N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace N]
  (AB BC AC : ℝ) (AN NC : ℝ):
  AB = 10 ∧ BC = 14 ∧ AC = 16 ∧ AN + NC = 16 ∧
  let incircle_radius (a b c : ℝ) := sorry -- Define the formula for the incenter radius
  in
  incircle_radius 10 (AC-AN) AN = incircle_radius 14 (NC-AC) NC → 
  AN / NC = 3 / 5 :=
by
  sorry

end triangle_ratio_AN_NC_l206_206987


namespace h_two_n_mul_h_2024_l206_206360

variable {h : ℕ → ℝ}
variable {k : ℝ}
variable (n : ℕ) (k_ne_zero : k ≠ 0)

-- Condition 1: h(m + n) = h(m) * h(n)
axiom h_add_mul (m n : ℕ) : h (m + n) = h m * h n

-- Condition 2: h(2) = k
axiom h_two : h 2 = k

theorem h_two_n_mul_h_2024 : h (2 * n) * h 2024 = k^(n + 1012) := 
  sorry

end h_two_n_mul_h_2024_l206_206360


namespace product_of_five_consecutive_numbers_not_square_l206_206307

theorem product_of_five_consecutive_numbers_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) :=
by
  sorry

end product_of_five_consecutive_numbers_not_square_l206_206307


namespace transformation_sequence_count_l206_206823

-- Representation of the possible transformations as a sum type
inductive Transformation
| rotate_90 : Transformation
| rotate_180 : Transformation
| rotate_270 : Transformation
| reflect_x : Transformation
| reflect_y : Transformation
| translate_23 : Transformation

-- Function to apply a transformation to a rectangle
def apply_transformation (R : set (ℝ × ℝ)) : Transformation → set (ℝ × ℝ)
| Transformation.rotate_90 := ... -- Define the effect of 90 degree rotation
| Transformation.rotate_180 := ... -- Define the effect of 180 degree rotation
| Transformation.rotate_270 := ... -- Define the effect of 270 degree rotation
| Transformation.reflect_x := ... -- Define the reflection across x-axis
| Transformation.reflect_y := ... -- Define the reflection across y-axis
| Transformation.translate_23 := ... -- Define the translation by (2,3)

-- Check if a given sequence of transformations returns the rectangle to its original position
def returns_to_original (R : set (ℝ × ℝ)) (seq : list Transformation) : Prop :=
  apply_transformations R seq = R

-- Count how many sequences of length 3 return R to original position
def valid_sequence_count (R : set (ℝ × ℝ)) : ℕ :=
  nat.card { seq : list Transformation // list.length seq = 3 ∧ returns_to_original R seq }

-- The rectangle with its vertices
def R : set (ℝ × ℝ) := {(0, 0), (5, 0), (5, 2), (0, 2)}

theorem transformation_sequence_count : valid_sequence_count R = 12 :=
by
  -- Provide a proof that there are exactly 12 valid sequences
  sorry

end transformation_sequence_count_l206_206823


namespace peter_stamp_grouping_l206_206300

theorem peter_stamp_grouping :
  let stamps := (List.range 38).tail in
  let total_value := stamps.sum in
  let factors := [1, 19, 37, 703].filter (λ d, total_value % d = 0) in
  factors.length = 2 :=
by
  let stamps := (List.range 38).tail
  let total_value := stamps.sum
  let factors := [1, 19, 37, 703].filter (λ d, total_value % d = 0)
  show factors.length = 2
  sorry

end peter_stamp_grouping_l206_206300


namespace perpendicular_lines_find_a_l206_206979

theorem perpendicular_lines_find_a (a : ℝ) :
  (∃ b : ℝ, (ax + 2y + 6 = 0) ∧ (x + a(a+1)y + a^2 - 1 = 0) ∧ 
  (ax + 2y + 6 = 0) ⬝ (x + a(a+1)y + a^2 - 1 = 0 perpendicular)) →
  (a = 0 ∨ a = -3 / 2) :=
by
  sorry

end perpendicular_lines_find_a_l206_206979


namespace proof_part_1_proof_part_2_l206_206098

variable {α : ℝ}

/-- Given tan(α) = 3, prove
  (1) (3 * sin(α) + 2 * cos(α))/(sin(α) - 4 * cos(α)) = -11 -/
theorem proof_part_1
  (h : Real.tan α = 3) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - 4 * Real.cos α) = -11 := 
by
  sorry

/-- Given tan(α) = 3, prove
  (2) (5 * cos^2(α) - 3 * sin^2(α))/(1 + sin^2(α)) = -11/5 -/
theorem proof_part_2
  (h : Real.tan α = 3) :
  (5 * (Real.cos α)^2 - 3 * (Real.sin α)^2) / (1 + (Real.sin α)^2) = -11 / 5 :=
by
  sorry

end proof_part_1_proof_part_2_l206_206098


namespace find_b_coefficients_l206_206513

theorem find_b_coefficients (x : ℝ) (b₁ b₂ b₃ b₄ : ℝ) :
  x^4 = (x + 1)^4 + b₁ * (x + 1)^3 + b₂ * (x + 1)^2 + b₃ * (x + 1) + b₄ →
  b₁ = -4 ∧ b₂ = 6 ∧ b₃ = -4 ∧ b₄ = 1 := by
  sorry

end find_b_coefficients_l206_206513


namespace additional_people_needed_to_complete_the_work_l206_206779
-- Import the entire Mathlib library for this problem

-- Define the conditions as given in the problem
variables (initial_people : ℕ) (total_days : ℕ) (days_passed : ℕ) 
          (work_complete_fraction : ℚ)
          (remaining_work : ℕ)

/-
  initial_people: 60
  total_days: 50
  days_passed: 25
  work_complete_fraction: 0.4 (40%)
  remaining_work is calculated in terms of man-days: 3000 * 0.6
-/
def remaining_work_days := total_days - days_passed
def initial_work := initial_people * total_days
def remaining_work := initial_work - (initial_work * work_complete_fraction : ℚ)
def work_by_existing_workers := initial_people * remaining_work_days
def additional_people_needed (X : ℕ) := X * remaining_work_days =
  (remaining_work - work_by_existing_workers : ℕ)

-- The theorem statement to prove
theorem additional_people_needed_to_complete_the_work :
  initial_people = 60 → total_days = 50 → days_passed = 25 → work_complete_fraction = 0.4 →
  remaining_work = 1800 →
  ∃ X, additional_people_needed initial_people total_days days_passed work_complete_fraction remaining_work X ∧ X = 12 := 
by 
  sorry

end additional_people_needed_to_complete_the_work_l206_206779


namespace count_of_digit_2_in_homes_l206_206570

theorem count_of_digit_2_in_homes (h : ∑ n in Finset.range 100, (n.digits 10).count 2 = 20) : 
  100 = Finset.card (Finset.range 100) :=
by
  sorry

end count_of_digit_2_in_homes_l206_206570


namespace max_m_le_sqrt_3_l206_206117

-- Definitions of the points and the parabola
def A := (-1 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 0 : ℝ)
def parabola_y_squared_eq_2x (P : ℝ × ℝ) := P.2 ^ 2 = 2 * P.1

-- Definition of the distance formula
def dist (X Y : ℝ × ℝ) : ℝ := Real.sqrt ((X.1 - Y.1) ^ 2 + (X.2 - Y.2) ^ 2)

-- Definition of the problem conditions
def point_on_parabola (P : ℝ × ℝ) := parabola_y_squared_eq_2x P
def m_condition (m : ℝ) (P : ℝ × ℝ) := dist P A = m * dist P B

-- The mathematical problem in Lean 4 statement
theorem max_m_le_sqrt_3 (m : ℝ) (P : ℝ × ℝ) :
  point_on_parabola P →
  m_condition m P →
  m ≤ Real.sqrt 3 :=
by
  sorry

end max_m_le_sqrt_3_l206_206117


namespace third_motorcyclist_speed_l206_206356

theorem third_motorcyclist_speed 
  (t₁ t₂ : ℝ)
  (x : ℝ)
  (h1 : t₁ - t₂ = 1.25)
  (h2 : 80 * t₁ = x * (t₁ - 0.5))
  (h3 : 60 * t₂ = x * (t₂ - 0.5))
  (h4 : x ≠ 60)
  (h5 : x ≠ 80):
  x = 100 :=
by
  sorry

end third_motorcyclist_speed_l206_206356


namespace show_coloring_possible_l206_206899

noncomputable def coloring_possible : Prop :=
  ∀ (n : ℕ), n ≥ 1 → 
  ∃ (color : ℝ × ℝ → bool),
  ∀ (p q : ℝ × ℝ), p ≠ q → 
  (∃ (circle : ℝ × ℝ × ℝ), 
    (p.1 - circle.1)^2 + (p.2 - circle.2)^2 = circle.3^2 ∧
    (q.1 - circle.1)^2 + (q.2 - circle.2)^2 = circle.3^2) →
  color p ≠ color q

theorem show_coloring_possible : coloring_possible :=
sorry

end show_coloring_possible_l206_206899


namespace impossibility_of_dividing_stones_l206_206279

theorem impossibility_of_dividing_stones (stones : ℕ) (heaps : ℕ) (m : ℕ) :
  stones = 660 → heaps = 31 → (∀ (x y : ℕ), x ∈ heaps → y ∈ heaps → x < 2 * y → False) := sorry

end impossibility_of_dividing_stones_l206_206279


namespace simplify_expression_l206_206308

theorem simplify_expression (x : ℝ) :
  (3 * x)^5 + (4 * x^2) * (3 * x^2) = 243 * x^5 + 12 * x^4 :=
by
  sorry

end simplify_expression_l206_206308


namespace number_of_true_props_l206_206504

variables {α : Type} [linear_ordered_field α]

-- Definition of an even function
def is_even (f : α → α) : Prop := ∀ x, f x = f (-x)

-- Given conditions
variables (f : α → α) (h1 : is_even f) (h2 : ∀ x, f x + f (2 - x) = 0)

-- Definitions of the propositions
def prop1 : Prop := ∀ x, f (x + 2) = f x
def prop2 : Prop := ∀ x, f (x + 4) = f x
def prop3 : Prop := ∀ x, f (x - 1) = -f (-(x - 1))
def prop4 : Prop := ∀ x, f (x - 3) = f (-(x - 3))

-- Main theorem statement
theorem number_of_true_props : 
  ∃ (n : ℕ), n = 2 ∧ 
    (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = n :=
begin
  sorry
end

end number_of_true_props_l206_206504


namespace racetrack_circumference_diff_l206_206400

theorem racetrack_circumference_diff (d_inner d_outer width : ℝ) 
(h1 : d_inner = 55) (h2 : width = 15) (h3 : d_outer = d_inner + 2 * width) : 
  (π * d_outer - π * d_inner) = 30 * π :=
by
  sorry

end racetrack_circumference_diff_l206_206400


namespace four_digit_number_exists_l206_206090

theorem four_digit_number_exists :
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 4 * n = (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000) :=
sorry

end four_digit_number_exists_l206_206090


namespace solve_quadratic_for_q_l206_206849

-- Define the quadratic equation and the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main theorem statement
theorem solve_quadratic_for_q : ∃ q : ℝ, q ≠ 0 ∧ (discriminant q (-8) 2 = 0) → q = 8 :=
by
  -- Insert the assumptions and proof context here.
  -- However, since we were instructed not to consider the solution steps
  -- the proof is skipped with a "sorry".
  sorry

end solve_quadratic_for_q_l206_206849


namespace select_sqrt_n_points_not_equilateral_l206_206104

theorem select_sqrt_n_points_not_equilateral (n : ℕ) (points : fin n → ℝ × ℝ) :
  ∃ subset : fin (⌊real.sqrt n⌋₊) → fin n, ∀ (i j k : fin (⌊real.sqrt n⌋₊)),
    (i ≠ j ∧ j ≠ k ∧ i ≠ k) → 
    let p1 := points (subset i),
        p2 := points (subset j),
        p3 := points (subset k) in
    ¬ (dist p1 p2 = dist p2 p3 ∧ dist p2 p3 = dist p3 p1) :=
by
  sorry

end select_sqrt_n_points_not_equilateral_l206_206104


namespace a_6_value_l206_206944

-- Define the sequence as a function in Lean
def a : ℕ → ℚ
| 0       := -2  -- Note that in Lean, sequences are 0-indexed by default
| (n + 1) := 2 + (2 * a n) / (1 - a n)

theorem a_6_value : a 5 = -14 / 3 :=
sorry

end a_6_value_l206_206944


namespace vikas_rank_among_boys_l206_206990

def vikas_rank_overall := 9
def tanvi_rank_overall := 17
def girls_between := 2
def vikas_rank_top_boys := 4
def vikas_rank_bottom_overall := 18

theorem vikas_rank_among_boys (vikas_rank_overall tanvi_rank_overall girls_between vikas_rank_top_boys vikas_rank_bottom_overall : ℕ) :
  vikas_rank_top_boys = 4 := by
  sorry

end vikas_rank_among_boys_l206_206990


namespace only_strictly_increasing_function_l206_206877

def mho (n : ℕ) : ℕ :=
  ∑ i in (Nat.factors n).filter (λ p, p > 10^100), Nat.factorMultiplicity p n

theorem only_strictly_increasing_function (f : ℤ → ℤ):
  (∀ a b : ℤ, a > b → mho (Int.natAbs (f a - f b)) ≤ mho (Int.natAbs (a - b))) →
  (StrictMono f) →
  ∀ x : ℤ, f x = x := sorry

end only_strictly_increasing_function_l206_206877


namespace simplify_expression_l206_206310

theorem simplify_expression :
  (27 / 125) ^ (-1 / 3 : ℝ) = 5 / 3 := 
  sorry

end simplify_expression_l206_206310


namespace sand_weight_for_sandbox_l206_206392

theorem sand_weight_for_sandbox (side_length inches_per_bag weight_per_bag : ℕ) 
    (h1 : side_length = 40) 
    (h2 : inches_per_bag = 80) 
    (h3 : weight_per_bag = 30) : 
    (side_length * side_length / inches_per_bag) * weight_per_bag = 600 :=
by {
  have : side_length * side_length = 1600, 
  { rw h1, exact (40 * 40).symm },
  have : side_length * side_length / inches_per_bag = 20, 
  { rw [h2, this], exact (1600 / 80).symm },
  rw [this, h3],
  exact (20 * 30).symm
}

end sand_weight_for_sandbox_l206_206392


namespace find_x0_l206_206620

def f (x : ℝ) : ℝ :=
  if x >= 2 then x^2 + 2 else 2 * x

theorem find_x0 (x0 : ℝ) (h1 : f x0 = 8) : x0 = Real.sqrt 6 :=
  sorry

end find_x0_l206_206620


namespace solve_for_x_l206_206377

variable (x : ℝ)

-- Define the condition: 20% of x = 300
def twenty_percent_eq_300 := (0.20 * x = 300)

-- Define the goal: 120% of x = 1800
def one_twenty_percent_eq_1800 := (1.20 * x = 1800)

theorem solve_for_x (h : twenty_percent_eq_300 x) : one_twenty_percent_eq_1800 x :=
sorry

end solve_for_x_l206_206377


namespace hexagon_ratio_identity_l206_206891

noncomputable def convex_hexagon := Type

variables (ABCDEF : convex_hexagon)
variables (A B C D E F : ABCDEF)
variables [convex_hexagon ABCDEF]

-- Angles in degrees
variables (angle_B angle_D angle_F : ℝ)
variables h_angles_sum : angle_B + angle_D + angle_F = 360

-- Side ratios
variables (AB BC CD DE EF FA : ℝ)
variables h_side_ratios : (AB / BC) * (CD / DE) * (EF / FA) = 1

-- Target proof statement
theorem hexagon_ratio_identity 
  (BC CA AE FD DB : ℝ)
  (h1 : convex_hexagon ABCDEF)
  (h_angles_sum : angle_B + angle_D + angle_F = 360)
  (h_side_ratios : (AB / BC) * (CD / DE) * (EF / FA) = 1) :
  (BC / CA) * (AE / EF) * (FD / DB) = 1 := 
by 
  sorry

end hexagon_ratio_identity_l206_206891


namespace number_of_mappings_l206_206147

open Finset

-- Definitions of the sets M and N
def M : Finset (Fin 3) := {0, 1, 2}
def N : Finset (Fin 2) := {0, 1}

-- The number of different mappings from set M to set N
theorem number_of_mappings : (M.card * N.card) = 8 := by
  sorry

end number_of_mappings_l206_206147


namespace explicit_expression_solve_inequality_l206_206911

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := (n^2 - 3*n + 3) * x^(n+1)

theorem explicit_expression (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x)) :
  (∀ n x, f n x = x^3) :=
by
  sorry

theorem solve_inequality (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x))
  (f_eq : ∀ n x, f n x = x^3) :
  ∀ x, (x + 1)^3 + (3 - 2*x)^3 > 0 → x < 4 :=
by
  sorry

end explicit_expression_solve_inequality_l206_206911


namespace log_function_diff_l206_206982

noncomputable def maxmin (a : ℝ) :=
  let max_val := a^2
  let min_val := a^0
  max_val + min_val

theorem log_function_diff (a : ℝ) (hx : maxmin a = 5) : 
  let max_val := log a 2
  let min_val := log a (1 / 4)
  max_val - min_val = 3 := by
  sorry

end log_function_diff_l206_206982


namespace triangle_area_inequality_l206_206719

theorem triangle_area_inequality
  (A B C X Y Z : Type)
  [triangle ABC]
  [points_on_segment AX AB]
  [points_on_segment AY AC]
  (BY_col : collinear BY Z)
  (CX_col : collinear CX Z) :
  area BZX + area CZY > 2 * area XYZ :=
by
  sorry

end triangle_area_inequality_l206_206719


namespace number_of_roots_of_unity_l206_206459

noncomputable def is_root_of_unity (z : ℂ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ z ^ n = 1

theorem number_of_roots_of_unity (a : ℤ) : 
  (∑ z in {z : ℂ | is_root_of_unity z ∧ (z^2 + a * z - 1 = 0)}, 1) = 2 :=
by
  -- Proof omitted
  sorry

end number_of_roots_of_unity_l206_206459


namespace tangent_line_at_point_l206_206328

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

theorem tangent_line_at_point :
  let x := -1 in let y := -1 in
  let slope := (deriv f x) in
  let tangent_eq (x y : ℝ) := y = slope * (x + 1) - 1
  tangent_eq x y = (y = 2 * x + 1) :=
by
  sorry

end tangent_line_at_point_l206_206328


namespace max_value_of_function_l206_206552

theorem max_value_of_function (a : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = x^2 - a * x - a) (H_max : ∀ x ∈ set.Icc 0 2, f x ≤ 1) :
  a = 1 :=
sorry

end max_value_of_function_l206_206552


namespace sum_of_digits_divisible_by_9_l206_206894

theorem sum_of_digits_divisible_by_9 (N : ℕ) (a b c : ℕ) (hN : N < 10^1962)
  (h1 : N % 9 = 0)
  (ha : a = (N.digits 10).sum)
  (hb : b = (a.digits 10).sum)
  (hc : c = (b.digits 10).sum) :
  c = 9 :=
sorry

end sum_of_digits_divisible_by_9_l206_206894


namespace min_value_of_f_l206_206549

noncomputable def f (x : ℝ) : ℝ := 2 + 3 * x + 4 / (x - 1)

theorem min_value_of_f :
  (∀ x : ℝ, x > 1 → f x ≥ (5 + 4 * Real.sqrt 3)) ∧
  (f (1 + 2 * Real.sqrt 3 / 3) = 5 + 4 * Real.sqrt 3) :=
by
  sorry

end min_value_of_f_l206_206549


namespace length_of_rest_of_body_l206_206629

theorem length_of_rest_of_body (h : ℝ) (legs : ℝ) (head : ℝ) (rest_of_body : ℝ) :
  h = 60 → legs = (1 / 3) * h → head = (1 / 4) * h → rest_of_body = h - (legs + head) → rest_of_body = 25 := by
  sorry

end length_of_rest_of_body_l206_206629


namespace find_initial_sum_l206_206371

noncomputable def initial_sum (r : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
    let compound_interest := λ P : ℝ, P * ((1 + r)^t - 1)
    let simple_interest := λ P : ℝ, P * r * t
    let diff := λ P : ℝ, compound_interest P - simple_interest P
    classical.some (exists_P diff d)

theorem find_initial_sum :
  ∀ r t d, r = 0.16 → t = 4 → d = 170.64 → 
  initial_sum r t d = 1565.50 := by
  intros r t d hr ht hd
  rw [hr, ht, hd]
  unfold initial_sum
  apply_fun (λ x, x)
  sorry

end find_initial_sum_l206_206371


namespace triangle_LMN_area_l206_206365

-- Define the coordinates of the vertices
def L : ℝ × ℝ := (7.5, 12.5)
def M : ℝ × ℝ := (13.5, 2.6)
def N : ℝ × ℝ := (9.4, 18.8)

-- Function to compute the area of the triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  |(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2|

-- Theorem stating the area of the triangle L, M, and N
theorem triangle_LMN_area : triangle_area L M N = 28.305 := sorry

end triangle_LMN_area_l206_206365


namespace domino_partition_exists_l206_206359

theorem domino_partition_exists (dominoes : set (set (ℕ × ℕ))) :
  (∃ vertical_line : ℕ, vertical_line ∈ {1, 2, 3, 4, 5} ∧ 
    (∀ domino ∈ dominoes, ¬ (∃ x, (x, vertical_line) ∈ domino ∧ (x, vertical_line + 1) ∈ domino))) ∨
  (∃ horizontal_line : ℕ, horizontal_line ∈ {1, 2, 3, 4, 5} ∧ 
    (∀ domino ∈ dominoes, ¬ (∃ y, (horizontal_line, y) ∈ domino ∧ (horizontal_line + 1, y) ∈ domino))) :=
sorry

end domino_partition_exists_l206_206359


namespace gcd_1617_1225_gcd_2023_111_gcd_589_6479_l206_206858

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 :=
by
  sorry

theorem gcd_2023_111 : Nat.gcd 2023 111 = 1 :=
by
  sorry

theorem gcd_589_6479 : Nat.gcd 589 6479 = 589 :=
by
  sorry

end gcd_1617_1225_gcd_2023_111_gcd_589_6479_l206_206858


namespace curve_represents_line_segment_and_semicircle_l206_206703

theorem curve_represents_line_segment_and_semicircle :
  ∃ (x y : ℝ), (3 * x - y + 1 = 0 ∨ y = sqrt (1 - x^2) ∧ -1 ≤ x ∧ x ≤ 1) →
    ((∃ y : ℝ, ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → y = sqrt (1 - x^2)) ∧
     (∃ x y : ℝ, 3 * x - y + 1 = 0)) :=
sorry

end curve_represents_line_segment_and_semicircle_l206_206703


namespace range_f_l206_206068

def f : ℝ → ℝ := λ x, log 2 (3 ^ x + 1)

theorem range_f :
  ∀ y, 0 < y ↔ ∃ x : ℝ, f x = y := 
sorry

end range_f_l206_206068


namespace polynomial_degree_at_least_n_l206_206492

noncomputable def polynomial (n : ℕ) : Type := n → ℤ

def poly_cond (P : polynomial → ℤ) (x : polynomial → bool) : Prop :=
  (∀ x: polynomial, (if list.all x true then
                     ∃ k, P(x) = k ∧ k > 0
                   else if list.count x false % 2 = 0 then P(x) > 0
                   else P(x) < 0 ))

theorem polynomial_degree_at_least_n (n : ℕ) (P : polynomial n → ℤ) :
  (poly_cond P) →
  (∃ (x : polynomial n), polynomial_degree P ≥ n) :=
sorry

end polynomial_degree_at_least_n_l206_206492


namespace num_proper_subsets_C_l206_206169

def A : Set ℤ := {-1, 1}
def B : Set ℤ := {0, 2}
def C : Set ℤ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x + y}

theorem num_proper_subsets_C : (C.toFinset.powerset.card - 1) = 7 := by
  sorry

end num_proper_subsets_C_l206_206169


namespace problem_f_neg2012_add_f_2013_l206_206918

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_neg2012_add_f_2013 :=
  (h1 : ∀ x, f (-x) = f x)
  (h2 : ∀ x ≥ 0, f (x + 2) = f x)
  (h3 : ∀ x, (0 ≤ x ∧ x < 2) → f x = Real.log2 (x + 1)) : 
  f (-2012) + f 2013 = 1 := 
by
  sorry

end problem_f_neg2012_add_f_2013_l206_206918


namespace positive_values_of_f_and_g_l206_206140

theorem positive_values_of_f_and_g (m : ℝ) (x : ℝ) (hf : ∀ x : ℝ, 0 < f(m, x) ∨ 0 < g(m, x)) :
    0 < m ∧ m < 8 :=
sorry

variables {m x : ℝ}
def f (m x : ℝ) : ℝ :=
  2 * m * x ^ 2 - 2 * (4 - m) * x + 1

def g (m x : ℝ) : ℝ :=
  m * x

end positive_values_of_f_and_g_l206_206140


namespace exists_p_q_for_integer_roots_l206_206054

theorem exists_p_q_for_integer_roots : 
  ∃ (p q : ℤ), ∀ k (hk : k ∈ (Finset.range 10)), 
    ∃ (r1 r2 : ℤ), (r1 + r2 = -(p + k)) ∧ (r1 * r2 = (q + k)) :=
sorry

end exists_p_q_for_integer_roots_l206_206054


namespace minimal_hair_loss_l206_206395

theorem minimal_hair_loss (cards : Fin 100 → ℕ)
    (sum_sage1 : ℕ)
    (communicate_card_numbers : List ℕ)
    (communicate_sum : ℕ) :
    (∀ i : Fin 100, (communicate_card_numbers.contains (cards i))) →
    communicate_sum = sum_sage1 →
    sum_sage1 = List.sum communicate_card_numbers →
    communicate_card_numbers.length = 100 →
    ∃ (minimal_loss : ℕ), minimal_loss = 101 := by
  sorry

end minimal_hair_loss_l206_206395


namespace sin_diff_range_l206_206122

theorem sin_diff_range (x y : ℝ) (h : cos x + cos y = 1) : 
  -real.sqrt 3 ≤ sin x - sin y ∧ sin x - sin y ≤ real.sqrt 3 := 
begin
  sorry
end

end sin_diff_range_l206_206122


namespace relationship_among_a_b_c_l206_206501

noncomputable def f : ℝ → ℝ := sorry

lemma even_function {x : ℝ} : f x = f |x| := sorry
lemma increasing_on_neg {x y : ℝ} (hxy : x < y) (hy : y ≤ 0) : f x ≤ f y := sorry

def a := f (Real.log 7 / Real.log 4)
def b := f (-(Real.log 3 / Real.log 2))
def c := f (0.2 ^ (-0.5))

theorem relationship_among_a_b_c : a > b ∧ b > c := 
by
  -- Proof goes here
  sorry

end relationship_among_a_b_c_l206_206501


namespace heap_division_impossible_l206_206244

def similar_sizes (a b : Nat) : Prop := a < 2 * b

theorem heap_division_impossible (n k : Nat) (h1 : n = 660) (h2 : k = 31) 
    (h3 : ∀ p1 p2 : Nat, p1 ∈ (Finset.range k).to_set → p2 ∈ (Finset.range k).to_set → similar_sizes p1 p2 → p1 < 2 * p2) : 
    False := 
by
  sorry

end heap_division_impossible_l206_206244


namespace winning_candidate_votes_l206_206345

-- Define the conditions as hypotheses in Lean.
def two_candidates (candidates : ℕ) : Prop := candidates = 2
def winner_received_62_percent (V : ℝ) (votes_winner : ℝ) : Prop := votes_winner = 0.62 * V
def winning_margin (V : ℝ) : Prop := 0.24 * V = 384

-- The main theorem to prove: the winner candidate received 992 votes.
theorem winning_candidate_votes (V votes_winner : ℝ) (candidates : ℕ) 
  (h1 : two_candidates candidates) 
  (h2 : winner_received_62_percent V votes_winner)
  (h3 : winning_margin V) : 
  votes_winner = 992 :=
by
  sorry

end winning_candidate_votes_l206_206345


namespace card_division_ways_count_l206_206720

theorem card_division_ways_count :
  let cards := {x : ℕ | 15 ≤ x ∧ x ≤ 33}
  let participants := {Vasya, Petya, Misha}
  (∃ (f : cards → participants), 
    (∀ p ∈ participants, ∃ c ∈ cards, f c = p) ∧ 
    (∀ p ∈ participants, (∀ c₁ c₂ ∈ cards, f c₁ = p ∧ f c₂ = p ∧ c₁ ≠ c₂ →
      (c₁ - c₂) % 2 = 0))) →
  -- Proven that there are exactly 4596 ways to divide the cards.
  (card {f : cards → participants | 
    (∀ p ∈ participants, ∃ c ∈ cards, f c = p) ∧ 
    (∀ p ∈ participants, (∀ c₁ c₂ ∈ cards, f c₁ = p ∧ f c₂ = p ∧ c₁ ≠ c₂ →
      (c₁ - c₂) % 2 = 0))}) = 4596 :=
sorry

end card_division_ways_count_l206_206720


namespace arithmetic_sequence_a15_l206_206508

theorem arithmetic_sequence_a15 
  (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h1 : a 3 + a 13 = 20)
  (h2 : a 2 = -2) :
  a 15 = 24 := 
by
  sorry

end arithmetic_sequence_a15_l206_206508


namespace inequality_condition_l206_206648

variables {a b c : ℝ} {x : ℝ}

theorem inequality_condition (h : a * a + b * b < c * c) : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end inequality_condition_l206_206648


namespace number_of_apples_l206_206882

def asparagus_cost : ℝ := 60 * 3
def grapes_cost : ℝ := 40 * 2.5
def total_cost : ℝ := 630
def apple_price : ℝ := 0.5

theorem number_of_apples :
  let apple_cost := total_cost - (asparagus_cost + grapes_cost)
  apple_cost / apple_price = 700 :=
by 
  let apple_cost := total_cost - (asparagus_cost + grapes_cost)
  calc
  apple_cost / apple_price = 350 / 0.5 : by sorry
                        ... = 700       : by sorry

end number_of_apples_l206_206882


namespace shortest_chord_length_l206_206011

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the circle C by its equation and converting to standard form
def C_center : ℝ × ℝ := (1, 0)
def C_radius : ℝ := 2

-- Define the function to calculate the distance between two points
def dist (A B : ℝ × ℝ) : ℝ :=
  (real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2))

-- Calculate the distance CP
def CP_distance : ℝ := dist P C_center

-- State the theorem to be proven
theorem shortest_chord_length :
  2 * real.sqrt (C_radius ^ 2 - CP_distance ^ 2) = 2 * real.sqrt 2 :=
by
  sorry

end shortest_chord_length_l206_206011


namespace problem_part1_problem_part2_l206_206097

theorem problem_part1
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (x y : ℤ)
  (hA : A x y = 2 * x ^ 2 + 4 * x * y - 2 * x - 3)
  (hB : B x y = -x^2 + x*y + 2) :
  3 * A x y - 2 * (A x y + 2 * B x y) = 6 * x ^ 2 - 2 * x - 11 := by
  sorry

theorem problem_part2
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (y : ℤ)
  (H : ∀ x, B x y + (1 / 2) * A x y = C) :
  y = 1 / 3 := by
  sorry

end problem_part1_problem_part2_l206_206097


namespace num_children_proof_l206_206806

noncomputable def number_of_children (total_persons : ℕ) (total_revenue : ℕ) (adult_price : ℕ) (child_price : ℕ) : ℕ :=
  let adult_tickets := (child_price * total_persons - total_revenue) / (child_price - adult_price)
  let child_tickets := total_persons - adult_tickets
  child_tickets

theorem num_children_proof : number_of_children 280 14000 60 25 = 80 := 
by
  unfold number_of_children
  sorry

end num_children_proof_l206_206806


namespace solve_system_infinite_solutions_l206_206983

theorem solve_system_infinite_solutions (m : ℝ) (h1 : ∀ x y : ℝ, x + m * y = 2) (h2 : ∀ x y : ℝ, m * x + 16 * y = 8) :
  m = 4 :=
sorry

end solve_system_infinite_solutions_l206_206983


namespace complex_rotation_l206_206580

noncomputable def rotate_complex (z : ℂ) (θ : ℝ) : ℂ :=
  z * complex.exp (θ * complex.I)

theorem complex_rotation :
  let z : ℂ := 3 - complex.I * real.sqrt 3
  let θ : ℝ := real.pi / 3
  rotate_complex z θ = - 2 * real.sqrt 3 * complex.I :=
by
  unfold rotate_complex
  simp [complex.exp, complex.cos, complex.sin]
  sorry

end complex_rotation_l206_206580


namespace algebraic_expression_l206_206845

-- Define a variable x
variable (x : ℝ)

-- State the theorem
theorem algebraic_expression : (5 * x - 3) = 5 * x - 3 :=
by
  sorry

end algebraic_expression_l206_206845


namespace impossible_division_l206_206259

theorem impossible_division (heap_size : ℕ) (heap_count : ℕ) (similar_sizes : Π (a b : ℕ), a < 2 * b) :
  ¬ (∃ (heaps : Finset ℕ), heaps.card = heap_count ∧ heaps.sum = heap_size ∧ ∀ (a b ∈ heaps), similar_sizes a b) :=
by
  let heap_size := 660
  let heap_count := 31
  sorry

end impossible_division_l206_206259


namespace num_of_possible_values_b_l206_206687

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206687


namespace sum_diff_reciprocals_equals_zero_l206_206527

theorem sum_diff_reciprocals_equals_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (1 / (a + 1)) + (1 / (a - 1)) + (1 / (b + 1)) + (1 / (b - 1)) = 0) :
  (a + b) - (1 / a + 1 / b) = 0 :=
by
  sorry

end sum_diff_reciprocals_equals_zero_l206_206527


namespace possible_values_b_count_l206_206677

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206677


namespace coefficient_x3_term_l206_206044

theorem coefficient_x3_term (x : ℝ) : (2 * real.sqrt x - 1 / real.sqrt x) ^ 6 = 
  64 * x ^ 3 + sorry :=
by sorry

end coefficient_x3_term_l206_206044


namespace triangle_inequality_l206_206592

theorem triangle_inequality 
  (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : c > 0) 
  (h5 : a + b > c) 
  (h6 : b + c > a) 
  (h7 : c + a > b)
  : 5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7 / 3 :=
begin
  sorry
end

end triangle_inequality_l206_206592


namespace number_of_possible_values_l206_206659

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206659


namespace circle_condition_l206_206192

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) →
  m < 1 :=
sorry

end circle_condition_l206_206192


namespace range_of_m_local_odd_function_l206_206556

def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

noncomputable def f (x m : ℝ) : ℝ :=
  9^x - m * 3^x - 3

theorem range_of_m_local_odd_function :
  (∀ m : ℝ, is_local_odd_function (λ x => f x m) ↔ m ∈ Set.Ici (-2)) :=
by
  sorry

end range_of_m_local_odd_function_l206_206556


namespace average_of_data_with_mode_of_2_l206_206022

theorem average_of_data_with_mode_of_2 (x : ℕ) (h_mode : List.mode [2, 3, x, 2, 3, 6] = 2) :
  List.sum [2, 3, x, 2, 3, 6] / 6 = 3 :=
by
  sorry

end average_of_data_with_mode_of_2_l206_206022


namespace abs_expression_evaluation_l206_206439

noncomputable def π : ℝ := Real.pi -- defining noncomputable value π

theorem abs_expression_evaluation : π < 10 → |π - |π - 10|| = 10 - 2 * π :=
by
  intro h
  have inner := abs_sub_lt _ _ h
  have outer := abs_sub_lt _ _ inner
  exact outer

end abs_expression_evaluation_l206_206439


namespace max_f_value_l206_206863

noncomputable def f (x : ℝ) : ℝ := 9 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 15 :=
by
  sorry

end max_f_value_l206_206863


namespace negation_equivalence_l206_206143

theorem negation_equivalence :
  ¬(∃ x : ℝ, tan x = 1) ↔ ∀ x : ℝ, tan x ≠ 1 :=
by
  sorry

end negation_equivalence_l206_206143


namespace determine_t_l206_206837

noncomputable def t_closest_to_vector (t : ℚ) : Prop :=
  let v := λ t : ℚ, (1, -2, 4 : ℚ) + t • (5, 3, 2)
  let a := (3, 1, -1 : ℚ) 
  let direction := (5, 3, 2 : ℚ)
  let diff := (v t) - a
  let dot := diff.1 * direction.1 + diff.2 * direction.2 + diff.3 * direction.3
  dot = 0

theorem determine_t : t_closest_to_vector (9 / 38) := 
sorry

end determine_t_l206_206837


namespace fraction_budget_paid_l206_206843

variable (B : ℝ) (b k : ℝ)

-- Conditions
def condition1 : b = 0.30 * (B - k) := by sorry
def condition2 : k = 0.10 * (B - b) := by sorry

-- Proof that Jenny paid 35% of her budget for her book and snack
theorem fraction_budget_paid :
  b + k = 0.35 * B :=
by
  -- use condition1 and condition2 to prove the theorem
  sorry

end fraction_budget_paid_l206_206843


namespace vec_a_perp_vec_diff_l206_206954

-- Define vectors a and b
def vec_a : EuclideanSpace ℝ (Fin 2) := ![-2, 1]
def vec_b : EuclideanSpace ℝ (Fin 2) := ![-1, 3]

-- Define the vector difference a - b
def vec_diff : EuclideanSpace ℝ (Fin 2) := vec_a - vec_b

-- Prove the dot product is zero, indicating orthogonality
theorem vec_a_perp_vec_diff : (vec_a ⬝ vec_diff = 0) :=
by sorry

end vec_a_perp_vec_diff_l206_206954


namespace words_per_page_l206_206387

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 120) : p = 195 := by
  sorry

end words_per_page_l206_206387


namespace max_cars_and_quotient_l206_206290

-- Definition of the problem parameters
def car_length : ℕ := 5
def speed_per_car_length : ℕ := 10
def hour_in_seconds : ℕ := 3600
def one_kilometer_in_meters : ℕ := 1000
def distance_in_meters_per_hour (n : ℕ) : ℕ := (10 * n) * one_kilometer_in_meters
def unit_distance (n : ℕ) : ℕ := car_length * (n + 1)

-- Hypotheses
axiom car_spacing : ∀ n : ℕ, unit_distance n = car_length * (n + 1)
axiom car_speed : ∀ n : ℕ, distance_in_meters_per_hour n = (10 * n) * one_kilometer_in_meters

-- Maximum whole number of cars M that can pass in one hour and the quotient when M is divided by 10
theorem max_cars_and_quotient : ∃ (M : ℕ), M = 3000 ∧ M / 10 = 300 := by
  sorry

end max_cars_and_quotient_l206_206290


namespace quadratic_inequality_l206_206583

theorem quadratic_inequality (a b c d x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 = -a) 
  (h2 : x1 * x2 = b)
  (h3 : x3 + x4 = -c)
  (h4 : x3 * x4 = d)
  (h5 : b > d)
  (h6 : b > 0)
  (h7 : d > 0) :
  a^2 - c^2 > b - d :=
by
  sorry

end quadratic_inequality_l206_206583


namespace solve_problem_l206_206047

noncomputable def problem_statement : Prop :=
  (2015 : ℝ) / (2015^2 - 2016 * 2014) = 2015

theorem solve_problem : problem_statement := by
  -- Proof steps will be filled in here.
  sorry

end solve_problem_l206_206047


namespace arctan_sum_eq_pi_div_two_l206_206438

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
sorry

end arctan_sum_eq_pi_div_two_l206_206438


namespace ice_block_original_volume_l206_206417

theorem ice_block_original_volume
  (V : ℝ)
  (h1 : V > 0) -- The original volume is positive
  (h_after_first_hour : V * (3/5)) 
  (h_after_second_hour : V * (3/5) * (4/7))
  (h_after_third_hour : V * (3/5) * (4/7) * (2/3) = 0.15): 
  V = 0.65625 := 
begin
  -- This is where the proof would go
  sorry
end

end ice_block_original_volume_l206_206417


namespace range_of_a_l206_206173

noncomputable def range_of_a_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, |x + 1| + |x - a| ≤ 2

theorem range_of_a : ∀ a : ℝ, range_of_a_condition a → (-3 : ℝ) ≤ a ∧ a ≤ 1 :=
by
  intros a h
  sorry

end range_of_a_l206_206173


namespace money_made_l206_206220

def initial_amount : ℕ := 26
def final_amount : ℕ := 52

theorem money_made : (final_amount - initial_amount) = 26 :=
by sorry

end money_made_l206_206220


namespace smallest_option_l206_206875

-- Define the problem with the given condition
def x : ℕ := 10

-- Define all the options in the problem
def option_a := 6 / x
def option_b := 6 / (x + 1)
def option_c := 6 / (x - 1)
def option_d := x / 6
def option_e := (x + 1) / 6
def option_f := (x - 2) / 6

-- The proof problem statement to show that option_b is the smallest
theorem smallest_option :
  option_b < option_a ∧ option_b < option_c ∧ option_b < option_d ∧ option_b < option_e ∧ option_b < option_f :=
by
  sorry

end smallest_option_l206_206875


namespace neg_three_is_square_mod_p_l206_206229

theorem neg_three_is_square_mod_p (q : ℤ) (p : ℕ) (prime_p : Nat.Prime p) (condition : p = 3 * q + 1) :
  ∃ x : ℤ, (x^2 ≡ -3 [ZMOD p]) :=
sorry

end neg_three_is_square_mod_p_l206_206229


namespace number_of_possible_values_l206_206662

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206662


namespace paris_hair_count_paris_hair_count_specific_paris_hair_count_increase_l206_206916

theorem paris_hair_count (num_hairs : ℕ) (num_parisians : ℕ) 
  (h_hairs : num_hairs < 300000) (h_parisians : num_parisians = 3000000) : 
  ∃ (k : ℕ), k ≥ 2 ∧ ∃ S : Finset ℕ, S.card ≥ 2 ∧ ∀ x ∈ S, x < 300000 :=
by admit

theorem paris_hair_count_specific (num_hairs : ℕ) (num_parisians : ℕ)
  (h_hairs : num_hairs < 300000) (h_parisians : num_parisians = 3000000) :
  ∃ (k : ℕ), k ≥ 10 ∧ ∃ S : Finset ℕ, S.card ≥ 10 ∧ ∀ x ∈ S, x < 300000 :=
by admit

theorem paris_hair_count_increase (num_hairs : ℕ) (num_parisians : ℕ)
  (h_hairs : num_hairs < 300000) (h_parisians : num_parisians = 3000001) :
  ∃ (k : ℕ), k ≥ 11 ∧ ∃ S : Finset ℕ, S.card ≥ 11 ∧ ∀ x ∈ S, x < 300000 :=
by admit

end paris_hair_count_paris_hair_count_specific_paris_hair_count_increase_l206_206916


namespace subset_implies_l206_206163

theorem subset_implies (A B C : Set) (h : A ∪ B = B ∩ C) : A ⊆ C := by
  sorry

end subset_implies_l206_206163


namespace birds_find_more_than_half_millet_on_thursday_l206_206286

-- Define initial conditions
def initial_total_seeds : ℕ := 2
def initial_millet_percent : ℝ := 0.3
def initial_millet_seeds : ℝ := initial_millet_percent * initial_total_seeds
def initial_other_seeds : ℝ := initial_total_seeds - initial_millet_seeds

-- Define daily patterns
def daily_total_seeds_added : ℕ := 2
def daily_millet_seeds_added : ℝ := initial_millet_percent * daily_total_seeds_added
def daily_other_seeds_percent_reduction : ℝ := 0.1
def daily_millet_consumption_rate : ℝ := 0.3

-- Recursive function to calculate remaining millet seeds on day n
def remaining_millet (n : ℕ) : ℝ :=
if n = 1 then initial_millet_seeds + daily_millet_seeds_added
else (1 - daily_millet_consumption_rate) * remaining_millet (n - 1) + daily_millet_seeds_added

-- Recursive function to calculate remaining other seeds on day n
def remaining_other_seeds (n : ℕ) : ℝ :=
if n = 1 then initial_other_seeds
else if n = 2 then 1.2
else (remaining_other_seeds (n - 1)) * (1 - daily_other_seeds_percent_reduction) + daily_total_seeds_added * (1 - initial_millet_percent)

-- Check if more than half of the seeds are millet on the given day
def is_more_than_half_millet (n : ℕ) : Prop :=
remaining_millet n > (remaining_millet n + remaining_other_seeds n) / 2

-- Main theorem to prove the condition on Thursday
theorem birds_find_more_than_half_millet_on_thursday : is_more_than_half_millet 4 :=
sorry

end birds_find_more_than_half_millet_on_thursday_l206_206286


namespace average_value_l206_206820

variable (z : ℝ)

theorem average_value : (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 :=
by
  sorry

end average_value_l206_206820


namespace evaluate_function_l206_206477

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 2) ^ x else Math.log x / Math.log 2

theorem evaluate_function :
  f 8 + f (Real.log (1 / 4) / Real.log 2) = 7 :=
by
  sorry

end evaluate_function_l206_206477


namespace grid_filling_possible_iff_even_l206_206575

theorem grid_filling_possible_iff_even (n : ℕ) (h_pos : 0 < n) :
  (∃ (grid : matrix (fin n) (fin n) ℕ), 
    (∀ i j, grid i j = 0 ∨ grid i j = 1 ∨ grid i j = 2) ∧ 
    (finset.univ.image (λ i, (finset.univ.sum (λ j, grid i j))).val = finset.range (2 * n + 1)) ∧
    (finset.univ.image (λ j, (finset.univ.sum (λ i, grid i j))).val = finset.range (2 * n + 1))) ↔ 
    n % 2 = 0 :=
begin
  sorry
end

end grid_filling_possible_iff_even_l206_206575


namespace thirtieth_term_of_arithmetic_sequence_l206_206337

theorem thirtieth_term_of_arithmetic_sequence (a3 a11 : ℝ) (d : ℝ) 
  (h1 : a3 = 12) (h2 : a11 = 32) (h3: a11 = a3 + 8 * d) : 
  let a30 := a3 + 27 * d in a30 = 79.5 := 
by
  have h4 : 8 * d = a11 - a3 := by rw [h1, h2, h3]; linarith
  have h5 : d = (a11 - a3) / 8 := by rw h4; ring
  have h6 : d = 2.5 := by rw [h1, h2, h5]; norm_num
  have a30_eq : a30 = a3 + 27 * d := by rfl
  rw [h1, h6] at a30_eq
  simp  at a30_eq
  norm_num[show 27 * 2.5 = 67.5, by norm_num] at sorry

end thirtieth_term_of_arithmetic_sequence_l206_206337


namespace evaluate_fraction_l206_206609

open Complex

theorem evaluate_fraction (a b : ℂ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a^2 - a*b + b^2 = 0) : 
  (a^6 + b^6) / (a + b)^6 = 1 / 18 := by
  sorry

end evaluate_fraction_l206_206609


namespace find_broomsticks_l206_206830

variables (skulls spiderwebs cauldrons pumpkins budget_left pending_decorations total_decorations : ℕ)
variable broomsticks

-- Conditions from a)
def conditions :=
  skulls = 12 ∧
  spiderwebs = 12 ∧
  cauldrons = 1 ∧
  pumpkins = 2 * spiderwebs ∧
  budget_left = 20 ∧
  pending_decorations = 10 ∧
  total_decorations = 83

-- The main statement to prove
theorem find_broomsticks (h : conditions): 
  (broomsticks : ℕ) = total_decorations - (skulls + spiderwebs + pumpkins + cauldrons + budget_left + pending_decorations - skulls - spiderwebs - cauldrons - pumpkins) :=
begin
  sorry
end

end find_broomsticks_l206_206830


namespace power_function_properties_l206_206920

theorem power_function_properties (m : ℤ) :
  (m^2 - 2 * m - 2 ≠ 0) ∧ (m^2 + 4 * m < 0) ∧ (m^2 + 4 * m % 2 = 1) → m = -1 := by
  intro h
  sorry

end power_function_properties_l206_206920


namespace prob_three_primes_in_four_rolls_l206_206427

-- Define the basic properties
def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def prob_prime : ℚ := 5 / 12
def prob_not_prime : ℚ := 7 / 12

def choose (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

lemma binom_coefficient : choose 4 3 = 4 := 
  by simp [choose, nat.factorial]

theorem prob_three_primes_in_four_rolls : 
  (4 * ((prob_prime ^ 3) * prob_not_prime) = (875 / 5184)) :=
  by tidy; sorry -- The actual proof is omitted

end prob_three_primes_in_four_rolls_l206_206427


namespace T_n_lt_5_div_12_l206_206488

-- Define sequence {a_n} such that S_n = n^2 + 2n
def S : ℕ → ℕ := λ n, n * n + 2 * n
def a (n : ℕ) : ℕ := if n = 0 then S 1 else (S n - S (n - 1))

-- Define the sequence {b_n}
def b (n : ℕ) : ℝ := 4 / ((a n + 1) * (a (n + 1) + 3))

-- Sum of the first n terms of the sequence {b_n}
noncomputable def T (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Theorem to prove T_n < 5 / 12
theorem T_n_lt_5_div_12 (n : ℕ) : T n < 5 / 12 := by
  sorry

end T_n_lt_5_div_12_l206_206488


namespace number_of_possible_values_of_b_l206_206670

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206670


namespace triangle_ABC_area_l206_206028

noncomputable def area_of_triangle_ABC : ℝ :=
let a : ℝ := 4, b : ℝ := 2, c : ℝ := 6,
    s : ℝ := (a + b + c) / 2 in
Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_ABC_area :
  area_of_triangle_ABC = 4 * Real.sqrt 6 :=
by sorry

end triangle_ABC_area_l206_206028


namespace slope_of_dividing_line_l206_206190

-- Define the vertices of the L-shaped region
def vertices := [(0, 0), (0, 4), (4, 4), (4, 2), (6, 2), (6, 0)]

-- Define the total area of the L-shaped region
def total_area : ℝ := 20

-- Define the statement of the theorem
theorem slope_of_dividing_line :
  ∃ m : ℝ, (total_area / 2 = 10) ∧ (m = 1 / 2) :=
begin
  sorry
end

end slope_of_dividing_line_l206_206190


namespace sum_of_distances_is_ellipse_l206_206880

theorem sum_of_distances_is_ellipse
  (A B : Point) (P : Point) (d : ℝ) (hd : d = distance A B) :
  (distance P A + distance P B = 2 * d) ->
  is_ellipse (set_of (λ P, distance P A + distance P B = 2 * d)) :=
  sorry

end sum_of_distances_is_ellipse_l206_206880


namespace countNegativeValues_l206_206467

-- Define the condition that sqrt(x + 122) is a positive integer
noncomputable def isPositiveInteger (n : ℤ) (x : ℤ) : Prop :=
  ∃ n : ℤ, (n > 0) ∧ (x + 122 = n * n)

-- Define the condition that x is negative
def isNegative (x : ℤ) : Prop :=
  x < 0

-- Prove the number of different negative values of x such that sqrt(x + 122) is a positive integer is 11
theorem countNegativeValues :
  ∃ x_set : Finset ℤ, (∀ x ∈ x_set, isNegative x ∧ isPositiveInteger x (x + 122)) ∧ x_set.card = 11 :=
sorry

end countNegativeValues_l206_206467


namespace average_of_first_17_even_numbers_l206_206378

theorem average_of_first_17_even_numbers :
  let even_numbers := (List.range 17).map (λ n => 2 * (n + 1))
  let sum_even_numbers := even_numbers.sum
  sum_even_numbers / 17 = 20 :=
by
  let even_numbers := (List.range 17).map (λ n => 2 * (n + 1))
  let sum_even_numbers := even_numbers.sum
  have h_sum : sum_even_numbers = 342 := by sorry
  rw [h_sum]
  norm_num

end average_of_first_17_even_numbers_l206_206378


namespace john_coffees_per_day_l206_206208

theorem john_coffees_per_day (x : ℕ)
  (h1 : ∀ p : ℕ, p = 2)
  (h2 : ∀ p : ℕ, p = p + p / 2)
  (h3 : ∀ n : ℕ, n = x / 2)
  (h4 : ∀ d : ℕ, 2 * x - 3 * (x / 2) = 2) :
  x = 4 :=
by
  sorry

end john_coffees_per_day_l206_206208


namespace area_of_triangle_ABO_S_domain_and_max_value_l206_206520

noncomputable def S (k : ℝ) := (4 * real.sqrt 2 * real.sqrt (k^2 * (1 - k^2))) / (1 + k^2)

theorem area_of_triangle_ABO (k : ℝ) : 
  -1 < k ∧ k ≠ 0 ∧ 1 > k → 
  S(k) = (4 * real.sqrt 2 * real.sqrt (k^2 * (1 - k^2))) / (1 + k^2) :=
sorry

theorem S_domain_and_max_value : 
  ∃ k : ℝ, (k = real.sqrt 3 / 3 ∨ k = - real.sqrt 3 / 3) ∧ S(k) = 2 :=
sorry

end area_of_triangle_ABO_S_domain_and_max_value_l206_206520


namespace super_k_teams_l206_206193

theorem super_k_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 :=
sorry

end super_k_teams_l206_206193


namespace quadratic_no_real_roots_prob_l206_206012

theorem quadratic_no_real_roots_prob :
  let a_lower := (1:ℝ) / 4
  let a := set.Icc (0:ℝ) (1:ℝ)
  let root_free_prob := measure_theory.measure_space.prob ({ x | x ≤ a_lower }ᶜ ∩ a) :=
  root_free_prob = 3 / 4 :=
by
  sorry

end quadratic_no_real_roots_prob_l206_206012


namespace original_fund_was_830_l206_206009

/- Define the number of employees as a variable -/
variables (n : ℕ)

/- Define the conditions given in the problem -/
def initial_fund := 60 * n - 10
def new_fund_after_distributing_50 := initial_fund - 50 * n
def remaining_fund := 130

/- State the proof goal -/
theorem original_fund_was_830 :
  initial_fund = 830 :=
by sorry

end original_fund_was_830_l206_206009


namespace triangle_inequality_l206_206915

variable (a b c : ℝ)

theorem triangle_inequality (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b) : 
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 := 
sorry

end triangle_inequality_l206_206915


namespace find_nonzero_q_for_quadratic_l206_206848

theorem find_nonzero_q_for_quadratic :
  ∃ (q : ℝ), q ≠ 0 ∧ (∀ (x1 x2 : ℝ), (q * x1^2 - 8 * x1 + 2 = 0 ∧ q * x2^2 - 8 * x2 + 2 = 0) → x1 = x2) ↔ q = 8 :=
by
  sorry

end find_nonzero_q_for_quadratic_l206_206848


namespace evaluate_expression_l206_206453

theorem evaluate_expression :
  (0.25 : ℝ) ^ (-0.5) + (1 / 27 : ℝ) ^ (-1 / 3) - (625 : ℝ) ^ 0.25 = 0 := by
  sorry

end evaluate_expression_l206_206453


namespace number_of_divisors_of_24_divisible_by_4_l206_206652

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206652


namespace midpoint_H_H_l206_206221

-- Define the conditions
variables (ABC : Triangle) (H : Point) (Γ : Circle) 
          (A' B' C' : Point) (l_a l_b l_c : Line) 
          (O : Point) (H' : Point)

-- State the projections and conditions
axiom cond1 : is_orthocenter H ABC
axiom cond2 : circumcircle Γ ABC
axiom cond3 : on_circle A' Γ ∧ A' ≠ A
axiom cond4 : on_circle B' Γ ∧ B' ≠ B
axiom cond5 : on_circle C' Γ ∧ C' ≠ C
axiom cond6 : passes_through_projections l_a A' AB AC
axiom cond7 : passes_through_projections l_b B' BC BA
axiom cond8 : passes_through_projections l_c C' CA CB
axiom cond9 : is_circumcenter O l_a l_b l_c
axiom cond10 : is_orthocenter H' (Triangle.mk A' B' C')

-- Prove the main statement
theorem midpoint_H_H' : is_midpoint O H H' :=
sorry

end midpoint_H_H_l206_206221


namespace exists_diff_with_four_pairs_l206_206888

theorem exists_diff_with_four_pairs
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i ≤ 2015) :
  ∃ k > 0, ∃ ss : finset (ℕ × ℕ), 
    4 ≤ ss.card ∧ ∀ (p : ℕ × ℕ) ∈ ss, a p.1 - a p.2 = k :=
sorry

end exists_diff_with_four_pairs_l206_206888


namespace lateral_area_is_six_pi_l206_206781

noncomputable def lateral_area_of_cylinder (h : ℝ) : ℝ :=
  let D : ℝ := 6 / h in
  let S : ℝ := π * D * h in
  S

theorem lateral_area_is_six_pi (h : ℝ) (h_pos : h > 0) :
  lateral_area_of_cylinder h = 6 * π :=
by
  have : lateral_area_of_cylinder h = 6 * π := sorry
  exact this

end lateral_area_is_six_pi_l206_206781


namespace expected_value_bound_l206_206618

noncomputable def xi : ℝ → ℝ := sorry -- definition of xi as a nonnegative random variable
noncomputable def zeta : ℝ → ℝ := sorry -- definition of zeta as a nonnegative random variable
def indicator (p : Prop) [Decidable p] := if p then 1 else 0

axiom cond1 (x : ℝ) (hx : x > 0) :
  ℝ := sorry -- P(xi ≥ x) ≤ x⁻¹ E[zeta * indicator(xi ≥ x)]

theorem expected_value_bound (p: ℝ) (hp: p > 1) :
  ℝ := sorry -- E[xi^p] ≤ (p/(p-1))^p E[zeta^p]

end expected_value_bound_l206_206618


namespace lawn_length_is_51_l206_206402

noncomputable def find_lawn_length (width : ℕ) (road_width : ℕ) (cost_per_sq_meter : ℝ) (total_cost : ℝ) : ℕ :=
  let area_one_road := road_width * width
  let total_area := road_width * (road_width + width)
  let total_cost_calc := total_area * cost_per_sq_meter
  if total_cost_calc = total_cost then
    (total_cost / cost_per_sq_meter - area_one_road) / road_width
  else
    0 -- default value if there is a mismatch

theorem lawn_length_is_51 :
  find_lawn_length 35 4 0.75 258 = 51 :=
by
  sorry

end lawn_length_is_51_l206_206402


namespace possible_values_b_count_l206_206678

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206678


namespace repeating_decimal_as_fraction_l206_206075

-- Define the repeating decimal 0.36666... as a real number
def repeating_decimal : ℝ := 0.366666666666...

-- State the theorem to express the repeating decimal as a fraction
theorem repeating_decimal_as_fraction : repeating_decimal = (11 : ℝ) / 30 := 
sorry

end repeating_decimal_as_fraction_l206_206075


namespace parabola_focus_and_area_l206_206128

theorem parabola_focus_and_area :
  (∃ p : ℝ, p > 0 ∧ parabola_eq : ∀ x y : ℝ, ((y ^ 2 = 2 * p * x) ↔ (y ^ 2 = 16 * x))) ∧
  (∃ A B : ℝ × ℝ, A ≠ B ∧
                  (∀ x y : ℝ, ((y = x - 4) → is_on_parabola (x, y) ↔ (y ^ 2 = 16 * x))
                  → area_ΔAOB = 32 * real.sqrt 2) :=
  
begin
  sorry
end

def is_on_parabola (point : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, point = (x, y) ∧ (y ^ 2 = 16 * x)

def area_ΔAOB (A B : ℝ × ℝ) : ℝ := 
  ((1/2) * (abs (fst A - fst B) * (dist ((0,0) : ℝ × ℝ) (fst A, snd A))))

end parabola_focus_and_area_l206_206128


namespace avg_abc_43_l206_206700

variables (A B C : ℝ)

def avg_ab (A B : ℝ) : Prop := (A + B) / 2 = 40
def avg_bc (B C : ℝ) : Prop := (B + C) / 2 = 43
def weight_b (B : ℝ) : Prop := B = 37

theorem avg_abc_43 (A B C : ℝ) (h1 : avg_ab A B) (h2 : avg_bc B C) (h3 : weight_b B) :
  (A + B + C) / 3 = 43 :=
by
  sorry

end avg_abc_43_l206_206700


namespace log_5_3000_rounded_l206_206741

theorem log_5_3000_rounded : 
  (∃ x : ℝ, 5 ^ 4 = 625 ∧ 5 ^ 5 = 3125 ∧ x = log 5 3000 ∧ abs (x - 5) < 1) → 
  (⌊log 5 3000 + 0.5⌋ = 5) :=
by
  sorry

end log_5_3000_rounded_l206_206741


namespace maximize_GDP_growth_l206_206349

def projectA_investment : ℕ := 20  -- million yuan
def projectB_investment : ℕ := 10  -- million yuan

def total_investment (a b : ℕ) : ℕ := a + b
def total_electricity (a b : ℕ) : ℕ := 20000 * a + 40000 * b
def total_jobs (a b : ℕ) : ℕ := 24 * a + 36 * b
def total_GDP_increase (a b : ℕ) : ℕ := 26 * a + 20 * b  -- scaled by 10 to avoid decimals

theorem maximize_GDP_growth : 
  total_investment projectA_investment projectB_investment ≤ 30 ∧
  total_electricity projectA_investment projectB_investment ≤ 1000000 ∧
  total_jobs projectA_investment projectB_investment ≥ 840 → 
  total_GDP_increase projectA_investment projectB_investment = 860 := 
by
  -- Proof would be provided here
  sorry

end maximize_GDP_growth_l206_206349


namespace not_divide_660_into_31_not_divide_660_into_31_l206_206237

theorem not_divide_660_into_31 (H : ∀ (A B : ℕ), A ∈ S ∧ B ∈ S → A < 2 * B) : 
  false :=
begin
  -- math proof steps should go here
  sorry
end

namespace my_proof

def similar (x y : ℕ) : Prop := x < 2 * y

theorem not_divide_660_into_31 : ¬ ∃ (S : set ℕ), (∀ x ∈ S, x < 2 * y) ∧ (∑ x in S, x = 660) is ∧ |S| = 31 :=
begin
  sorry
end

end my_proof

end not_divide_660_into_31_not_divide_660_into_31_l206_206237


namespace sum_divides_product_iff_l206_206299

theorem sum_divides_product_iff (n : ℕ) : 
  (n*(n+1)/2) ∣ n! ↔ ∃ (a b : ℕ), 1 < a ∧ 1 < b ∧ a * b = n + 1 ∧ a ≤ n ∧ b ≤ n :=
sorry

end sum_divides_product_iff_l206_206299


namespace max_fx_increasing_intervals_fx_l206_206930

noncomputable def fx (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 + sqrt 3 * Real.sin (2 * x)

theorem max_fx : ∃ x : ℝ, ∀ y : ℝ, fx y ≤ 3 ∧ (fx x = 3 ↔ ∃ k : ℤ, x = Real.pi / 6 + k * Real.pi) := by
  sorry

theorem increasing_intervals_fx : ∀ k : ℤ, ∃ I : Set ℝ, I = Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) ∧
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → fx x < fx y := by
  sorry

end max_fx_increasing_intervals_fx_l206_206930


namespace dividend_calculation_l206_206375

def total_dividend 
  (investment : ℝ)
  (face_value : ℝ)
  (premium_rate : ℝ)
  (dividend_rate : ℝ) : ℝ :=
let cost_per_share := face_value * (1 + premium_rate)
    number_of_shares := investment / cost_per_share
    dividend_per_share := face_value * dividend_rate
in number_of_shares * dividend_per_share

theorem dividend_calculation (investment face_value premium_rate dividend_rate : ℝ)
  (h_inv : investment = 14400)
  (h_face : face_value = 100)
  (h_premium : premium_rate = 0.2)
  (h_dividend : dividend_rate = 0.05) :
  total_dividend investment face_value premium_rate dividend_rate = 600 := 
sorry

end dividend_calculation_l206_206375


namespace ordered_triples_54000_l206_206539

theorem ordered_triples_54000 : 
  ∃ (count : ℕ), 
  count = 16 ∧ 
  ∀ (a b c : ℕ), 
  0 < a → 0 < b → 0 < c → a^4 * b^2 * c = 54000 → 
  count = 16 := 
sorry

end ordered_triples_54000_l206_206539


namespace find_n_l206_206859

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ 123456 [MOD 8]) : n = 0 :=
sorry

end find_n_l206_206859


namespace hip_hop_final_percentage_is_39_l206_206710

noncomputable def hip_hop_percentage (total_songs percentage_country: ℝ):
  ℝ :=
  let percentage_non_country := 1 - percentage_country
  let original_ratio_hip_hop := 0.65
  let original_ratio_pop := 0.35
  let total_non_country := original_ratio_hip_hop + original_ratio_pop
  let hip_hop_percentage := original_ratio_hip_hop / total_non_country * percentage_non_country
  hip_hop_percentage

theorem hip_hop_final_percentage_is_39 (total_songs : ℕ) :
  hip_hop_percentage total_songs 0.40 = 0.39 :=
by
  sorry

end hip_hop_final_percentage_is_39_l206_206710


namespace find_f_f_0_l206_206135

def f (x : ℝ) : ℝ :=
if x > 0 then 3 * x^2 - 4
else if x = 0 then Real.pi
else 0

theorem find_f_f_0 : f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end find_f_f_0_l206_206135


namespace vertical_asymptote_sum_l206_206449

theorem vertical_asymptote_sum :
  ∀ x y : ℝ, (4 * x^2 + 8 * x + 3 = 0) → (4 * y^2 + 8 * y + 3 = 0) → x ≠ y → x + y = -2 :=
by
  sorry

end vertical_asymptote_sum_l206_206449


namespace value_at_one_positive_l206_206505

-- Define the conditions
variable {f : ℝ → ℝ} 

-- f is a monotonically increasing function
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement: proving that f(1) > 0
theorem value_at_one_positive (h1 : monotone_increasing f) (h2 : odd_function f) : f 1 > 0 :=
sorry

end value_at_one_positive_l206_206505


namespace age_solution_l206_206730

noncomputable def age_problem : Prop :=
  ∃ (m s x : ℕ),
  (m - 3 = 2 * (s - 3)) ∧
  (m - 5 = 3 * (s - 5)) ∧
  (m + x) * 2 = 3 * (s + x) ∧
  x = 1

theorem age_solution : age_problem :=
  by
    sorry

end age_solution_l206_206730


namespace second_derivative_y_wrt_x_l206_206870

noncomputable def x (t : ℝ) := Real.log t
noncomputable def y (t : ℝ) := Real.arctan t

theorem second_derivative_y_wrt_x (t : ℝ) (ht : t ≠ 0) :
  let y'' := (t * (1 - t^2)) / ((1 + t^2)^2)
  deriv (deriv (fun t => Real.arctan t) t) ((Real.log t) t) = y'' := sorry

end second_derivative_y_wrt_x_l206_206870


namespace dot_product_value_l206_206153

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (3, 1)

theorem dot_product_value :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 :=
by
  -- Proof goes here
  sorry

end dot_product_value_l206_206153


namespace clara_hardcover_books_l206_206839

-- Define the variables and conditions
variables (h p : ℕ)

-- Conditions based on the problem statement
def volumes_total : Prop := h + p = 12
def total_cost (total : ℕ) : Prop := 28 * h + 18 * p = total

-- The theorem to prove
theorem clara_hardcover_books (h p : ℕ) (H1 : volumes_total h p) (H2 : total_cost h p 270) : h = 6 :=
by
  sorry

end clara_hardcover_books_l206_206839


namespace frequency_of_defective_parts_l206_206292

theorem frequency_of_defective_parts (n m : ℕ) (h₁ : n = 5000) (h₂ : m = 32) : (m / n : ℚ) = 0.0064 := by
  -- We need to use the rational representation of 0.0064 for exact comparison
  have h₃ : (32 : ℚ) / 5000 = 32 / 5000 := by norm_num
  rw [h₁, h₂]
  exact h₃

end frequency_of_defective_parts_l206_206292


namespace problem_1_problem_2_problem_3_problem_4_l206_206432

-- 1. Given problem and correct answer (a = 12), prove the condition holds
theorem problem_1 (a : ℕ) (h : (∑ i in range (a + 1), i^2) / (∑ i in range (a + 1)) = 25 / 3) : a = 12 :=
sorry

-- 2. Given side length of a cube a = 12 cm, prove that the volume of the pyramid b = 36 cm^3
theorem problem_2 (a : ℝ) (ha : a = 12) (b : ℝ) (h : b = 1 / 48 * a^3) : b = 36 :=
sorry

-- 3. Given x^2 + cx + 36 ≥ 0 for all x ∈ ℝ, prove the maximum value of c is 12
theorem problem_3 (c : ℝ) (h : ∀ x : ℝ, x^2 + c * x + 36 ≥ 0) : c ≤ 12 :=
sorry

-- 4. Given the unit digit of 1997^1997 is (c - d) where c = 12, prove d = 5
theorem problem_4 (c : ℕ) (d : ℕ) (h : c = 12) (h_unit : (1997^1997) % 10 = c - d) : d = 5 :=
sorry

end problem_1_problem_2_problem_3_problem_4_l206_206432


namespace friend_spent_more_than_you_l206_206751

-- Define the total amount spent by both
def total_spent : ℤ := 19

-- Define the amount spent by your friend
def friend_spent : ℤ := 11

-- Define the amount spent by you
def you_spent : ℤ := total_spent - friend_spent

-- Define the difference in spending
def difference_in_spending : ℤ := friend_spent - you_spent

-- Prove that the difference in spending is $3
theorem friend_spent_more_than_you : difference_in_spending = 3 :=
by
  sorry

end friend_spent_more_than_you_l206_206751


namespace three_digit_cube_palindromes_count_l206_206959

def is_palindrome (n : ℕ) : Bool :=
  let s := n.digits 10
  s = s.reverse

theorem three_digit_cube_palindromes_count : 
  (Finset.filter is_palindrome 
      (Finset.image (λ n : ℕ, n^3) 
         (Finset.Ico 5 10))).card = 1 :=
by
  sorry

end three_digit_cube_palindromes_count_l206_206959


namespace extra_flowers_l206_206817

-- Definitions from the conditions
def tulips : Nat := 57
def roses : Nat := 73
def daffodils : Nat := 45
def sunflowers : Nat := 35
def used_flowers : Nat := 181

-- Statement to prove
theorem extra_flowers : (tulips + roses + daffodils + sunflowers) - used_flowers = 29 := by
  sorry

end extra_flowers_l206_206817


namespace composite_addition_l206_206287

theorem composite_addition (a b c : ℕ) (hc : c ≥ 2) (h : (1 / (a : ℚ)) + (1 / (b : ℚ)) = (1 / (c : ℚ))) :
  ∃ x, x = a + c ∧ ¬ Nat.prime x ∨ x = b + c ∧ ¬ Nat.prime x :=
by
  sorry

end composite_addition_l206_206287


namespace num_of_possible_values_b_l206_206681

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206681


namespace f_monotonic_increasing_g_increasing_domain_m_range_l206_206115

-- Definitions of the given functions
def f (a : ℝ) (x : ℝ) : ℝ := ln x - a / x
def g (a : ℝ) (x : ℝ) : ℝ := ln x - a / x + a * x - 6 * ln x
def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

-- 1. Prove the monotonicity of f(x) when a = 1
theorem f_monotonic_increasing : ∀ x > 0, MonotoneOn (f 1) (Ioi 0) := sorry

-- 2. For g(x) to be increasing in its domain, find the range of a
theorem g_increasing_domain : (∀ x > 0, 0 ≤ deriv (g a) x) → a ≥ 5 / 2 := sorry

-- 3. When a = 2, find the range of m such that g(x1) ≥ h(x2) for some x1 ∈ [1, 2]
theorem m_range
  (a2 : a = 2)
  (exists_x1 : ∃ (x1 ∈ set.Icc (1 : ℝ) 2), ∀ (x2 : ℝ), g 2 x1 ≥ h m x2) :
  m ≥ 8 - 5 * ln 2 := sorry

end f_monotonic_increasing_g_increasing_domain_m_range_l206_206115


namespace possible_values_b_count_l206_206680

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206680


namespace number_of_divisors_leq_two_sqrt_l206_206638

theorem number_of_divisors_leq_two_sqrt (n : ℕ) : 
  nat.totient (n) ≤ 2 * nat.sqrt n := 
sorry

end number_of_divisors_leq_two_sqrt_l206_206638


namespace quadratic_vertex_problem_l206_206708

/-- 
    Given a quadratic equation y = ax^2 + bx + c, where (2, -3) 
    is the vertex of the parabola and it passes through (0, 1), 
    prove that a - b + c = 6. 
-/
theorem quadratic_vertex_problem 
    (a b c : ℤ)
    (h : ∀ x : ℝ, y = a * (x - 2)^2 - 3)
    (h_point : y = 1)
    (h_passes_through_origin : y = a * (0 - 2)^2 - 3) :
    a - b + c = 6 :=
sorry

end quadratic_vertex_problem_l206_206708


namespace cannot_divide_660_stones_into_31_heaps_l206_206268

theorem cannot_divide_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i, 0 < heaps i ∧ heaps i < 2 * heaps (succ i % 31)) :=
sorry

end cannot_divide_660_stones_into_31_heaps_l206_206268


namespace Kayla_total_items_l206_206217

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l206_206217


namespace nonempty_subsequence_sums_to_zero_l206_206602

-- Definitions for conditions
def sequence (A : Array Int) : Prop := 
  A.size == 2000 ∧ ∀ i, 0 ≤ i < 2000 → -1000 ≤ A[i] ∧ A[i] ≤ 1000

def sum_one (A : Array Int) : Prop := 
  (List.ofArray A).sum = 1

-- Proposition stating the proof goal
theorem nonempty_subsequence_sums_to_zero (A : Array Int) 
  (h_seq: sequence A) (h_sum: sum_one A) : 
  ∃ (S : Set ℕ), S.nonempty ∧ (S ⊆ Finset.range 2000) ∧ ((S.sum (λ i : ℕ, (A[i] : Int))) = 0) :=
sorry

end nonempty_subsequence_sums_to_zero_l206_206602


namespace tank_empty_in_12_minutes_l206_206756

noncomputable def time_to_empty_tank (init_fraction tank full : ℚ) (rate_fill rate_empty : ℚ) : ℚ :=
  init_fraction / (rate_fill - rate_empty)

theorem tank_empty_in_12_minutes :
  let rate_fill := (1 : ℚ) / 10
  let rate_empty := (1 : ℚ) / 6
  let init_fraction := (4 : ℚ) / 5
  time_to_empty_tank init_fraction 1 rate_fill rate_empty = 12 :=
by
  simp [rate_fill, rate_empty, init_fraction, time_to_empty_tank]
  simp only [div_eq_mul_inv, mul_inv_cancel, sub_div, one_mul, mul_comm, mul_assoc,
             add_comm, add_assoc, inv_mul_cancel, inv_eq_zero]
  sorry

end tank_empty_in_12_minutes_l206_206756


namespace find_magnitude_b_l206_206530

variable (a b : ℝ^3)
variable (angle_ab : Real.Angle)
variable (magnitude_a magnitude_b : ℝ)
variable (magnitude_a_2b : ℝ)

-- Given Conditions
def conditions : Prop :=
  angle_ab = Real.Angle.pi / 3 ∧
  magnitude_a = 2 ∧
  magnitude_a_2b = 2 * Real.sqrt 7 ∧
  (Real.norm (a - 2 * b)) = magnitude_a_2b

-- Problem Statement
theorem find_magnitude_b (h : conditions a b angle_ab magnitude_a magnitude_b magnitude_a_2b) :
  magnitude_b = 3 :=
sorry

end find_magnitude_b_l206_206530


namespace impossibility_of_dividing_stones_l206_206277

theorem impossibility_of_dividing_stones (stones : ℕ) (heaps : ℕ) (m : ℕ) :
  stones = 660 → heaps = 31 → (∀ (x y : ℕ), x ∈ heaps → y ∈ heaps → x < 2 * y → False) := sorry

end impossibility_of_dividing_stones_l206_206277


namespace solve_linear_system_l206_206313

theorem solve_linear_system (x y a : ℝ) (h1 : 4 * x + 3 * y = 1) (h2 : a * x + (a - 1) * y = 3) (hxy : x = y) : a = 11 :=
by
  sorry

end solve_linear_system_l206_206313


namespace correct_q_solution_l206_206646

noncomputable def solve_q (n m q : ℕ) : Prop :=
  (7 / 8 : ℚ) = (n / 96 : ℚ) ∧
  (7 / 8 : ℚ) = ((m + n) / 112 : ℚ) ∧
  (7 / 8 : ℚ) = ((q - m) / 144 : ℚ) ∧
  n = 84 ∧
  m = 14 →
  q = 140

theorem correct_q_solution : ∃ (q : ℕ), solve_q 84 14 q :=
by sorry

end correct_q_solution_l206_206646


namespace total_flour_required_l206_206285

-- Definitions specified based on the given conditions
def flour_already_put_in : ℕ := 10
def flour_needed : ℕ := 2

-- Lean 4 statement to prove the total amount of flour required by the recipe
theorem total_flour_required : (flour_already_put_in + flour_needed) = 12 :=
by
  sorry

end total_flour_required_l206_206285


namespace trader_sold_40_meters_of_cloth_l206_206027

theorem trader_sold_40_meters_of_cloth 
  (total_profit_per_meter : ℕ) 
  (total_profit : ℕ) 
  (meters_sold : ℕ) 
  (h1 : total_profit_per_meter = 30) 
  (h2 : total_profit = 1200) 
  (h3 : total_profit = total_profit_per_meter * meters_sold) : 
  meters_sold = 40 := by
  sorry

end trader_sold_40_meters_of_cloth_l206_206027


namespace interest_rate_is_11_percent_l206_206460

-- Define the principal amount, total amount after given time, and the time period
def principal : ℝ := 886.0759493670886
def total_amount : ℝ := 1120
def time_period : ℝ := 2.4

-- Define the relationship between interest, principal, rate, and time.
def interest_rate_proof : Prop :=
  let interest := total_amount - principal in
  let rate := interest / (principal * time_period) in
  rate = 0.11

-- Formalize the statement to be proved in Lean
theorem interest_rate_is_11_percent : interest_rate_proof := 
by
  unfold interest_rate_proof
  sorry

end interest_rate_is_11_percent_l206_206460


namespace triangle_area_l206_206422

variables {k : ℝ}
variables {A B C D P Q N M O : EuclideanSpace ℝ (Fin 3)}

-- Definition of a parallelogram
def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 3)) :=
  collinear ({A, B, C}) ∧ collinear ({B, C, D}) ∧ collinear ({C, D, A}) ∧ collinear ({D, A, B})

-- Area of parallelogram ABCD
def area_of_parallelogram (A B C D : EuclideanSpace ℝ (Fin 3)) : ℝ := k

-- Midpoints
def is_midpoint (N : EuclideanSpace ℝ (Fin 3)) (X Y : EuclideanSpace ℝ (Fin 3)) :=
  2 • N = X + Y

-- Conditions based on the problem
def problem_conditions (A B C D P Q N M O : EuclideanSpace ℝ (Fin 3)) : Prop :=
  is_parallelogram A B C D ∧
  is_midpoint N B C ∧
  ∃ λ : ℝ, P = A + λ • (B - A) ∧ ∃ μ : ℝ, M = 1 / 2 • (A + D) ∧
  ∃ ν : ℝ, Q = A + ν • (B - A) ∧
  ∃ ξ : ℝ, O = A + ξ • (P - A)

-- Question to prove
theorem triangle_area (conditions : problem_conditions A B C D P Q N M O) : 
  area_of_triangle Q P O = (9 / 8) * k := 
sorry

end triangle_area_l206_206422


namespace students_per_bus_l206_206289

theorem students_per_bus
  (total_students : ℕ)
  (buses : ℕ)
  (students_in_cars : ℕ)
  (h1 : total_students = 375)
  (h2 : buses = 7)
  (h3 : students_in_cars = 4) :
  (total_students - students_in_cars) / buses = 53 :=
by
  sorry

end students_per_bus_l206_206289


namespace combined_mpg_l206_206301

-- Definitions based on the conditions
def ray_miles : ℕ := 150
def tom_miles : ℕ := 100
def ray_mpg : ℕ := 30
def tom_mpg : ℕ := 20

-- Theorem statement
theorem combined_mpg : (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 25 := by
  sorry

end combined_mpg_l206_206301


namespace number_of_possible_values_of_b_l206_206669

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206669


namespace perp_vectors_l206_206531

open Real

noncomputable def a : ℝ × ℝ × ℝ := (-1, -5, -2)

noncomputable def b (x : ℝ) : ℝ × ℝ × ℝ := (x, 2, x + 2)

theorem perp_vectors (x : ℝ) (hx : (a.1 * (b x).1 + a.2 * (b x).2 + a.3 * (b x).3) = 0) : 
  x = -14 / 3 := sorry

end perp_vectors_l206_206531


namespace polynomial_inequality_l206_206601

noncomputable def polynomial_max {R : Type*} [linear_ordered_field R] (p : polynomial R) : R :=
  ((coeff p).map abs).foldr max 0

theorem polynomial_inequality
  {R : Type*} [linear_ordered_field R]
  (f g : polynomial R)
  (r : R)
  (n : ℕ)
  (h₁ : g = (X + C r) * f)
  (h₂ : f.degree = n)
  (h₃ : g.degree = n + 1) :
  polynomial_max f / polynomial_max g ≤ n + 1 :=
sorry

end polynomial_inequality_l206_206601


namespace candy_store_sales_l206_206000

def total_sales (fudge_pounds truffles_count pretzels_count : ℕ) (fudge_price truffle_price pretzel_price : ℝ) : ℝ :=
  fudge_pounds * fudge_price + truffles_count * truffle_price + pretzels_count * pretzel_price

def sales_after_discount (sales : ℝ) (discount_rate : ℝ) : ℝ :=
  sales - (sales * discount_rate)

def sales_after_tax (sales : ℝ) (tax_rate : ℝ) : ℝ :=
  sales + (sales * tax_rate)

theorem candy_store_sales
  (fudge_pounds : ℕ := 37)
  (fudge_price : ℝ := 2.50)
  (truffles_count : ℕ := 82)
  (truffle_price : ℝ := 1.50)
  (pretzels_count : ℕ := 48)
  (pretzel_price : ℝ := 2.00)
  (discount_rate : ℝ := 0.10)
  (tax_rate : ℝ := 0.05) :
  sales_after_tax
    (total_sales fudge_pounds 1 fudge_price + truffles_count * truffle_price + pretzels_count * pretzel_price - (fudge_pounds * fudge_price * discount_rate))
    tax_rate
  = 317.36 :=
by sorry

end candy_store_sales_l206_206000


namespace find_eccentricity_l206_206521

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the ellipse equation
def ellipse_eq (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the midpoint condition for points A and B
def midpoint_cond (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1

-- Define the condition for the intersection points A and B
def intersection_points (x1 y1 x2 y2 a b : ℝ) : Prop :=
  line_eq x1 y1 ∧ line_eq x2 y2 ∧ ellipse_eq x1 y1 a b ∧ ellipse_eq x2 y2 a b

-- Define the condition on the intersection points and midpoint
def intersection_midpoint_condition (x1 y1 x2 y2 a b : ℝ) : Prop :=
  intersection_points x1 y1 x2 y2 a b ∧ midpoint_cond x1 y1 x2 y2

-- Define the eccentricity formula
def eccentricity (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2) / a

-- Prove the eccentricity given the conditions
theorem find_eccentricity (a b x1 y1 x2 y2 : ℝ) (h : a > b ∧ b > 0) (h_int : intersection_midpoint_condition x1 y1 x2 y2 a b) :
  eccentricity a b = real.sqrt 2 / 2 := by
  sorry

end find_eccentricity_l206_206521


namespace solve_for_d_l206_206887

variable (n c b d : ℚ)  -- Alternatively, specify the types if they are required to be specific
variable (H : n = d * c * b / (c - d))

theorem solve_for_d :
  d = n * c / (c * b + n) :=
by
  sorry

end solve_for_d_l206_206887


namespace vec_mag_diff_eq_neg_one_l206_206154

variables (a b : ℝ × ℝ)

def vec_add_eq := a + b = (2, 3)

def vec_sub_eq := a - b = (-2, 1)

theorem vec_mag_diff_eq_neg_one (h₁ : vec_add_eq a b) (h₂ : vec_sub_eq a b) :
  (a.1 ^ 2 + a.2 ^ 2) - (b.1 ^ 2 + b.2 ^ 2) = -1 :=
  sorry

end vec_mag_diff_eq_neg_one_l206_206154


namespace enclosed_area_abs_eq_54_l206_206745

theorem enclosed_area_abs_eq_54 :
  (∃ (x y : ℝ), abs x + abs (3 * y) = 9) → True := 
by
  sorry

end enclosed_area_abs_eq_54_l206_206745


namespace intersection_m_zero_range_of_m_l206_206948

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def B (x : ℝ) (m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

theorem intersection_m_zero : 
  ∀ x : ℝ, A x → B x 0 ↔ (1 ≤ x ∧ x < 3) :=
sorry

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, A x → B x m) ∧ (∃ x : ℝ, B x m ∧ ¬A x) → (m ≤ -2 ∨ m ≥ 4) :=
sorry

end intersection_m_zero_range_of_m_l206_206948


namespace gcd_binom_gt_one_iff_prime_power_l206_206852

theorem gcd_binom_gt_one_iff_prime_power (n : ℕ) (h_pos : n > 0) :
  (∃ p k : ℕ, nat.prime p ∧ k > 0 ∧ n = p ^ k) ↔
    nat.gcd (list.map (λ i, nat.choose n i) (list.range (n - 1))) > 1 := 
begin
  sorry
end

end gcd_binom_gt_one_iff_prime_power_l206_206852


namespace lisa_needs_additional_marbles_l206_206284

theorem lisa_needs_additional_marbles :
  ∀ (n m : ℕ), (n = 11) → (m = 45) → (Σ i in finset.range(n), (i + 1)) = 66 → (66 - m) = 21 :=
begin
  intros n m h1 h2 h_sum,
  rw h1 at *,
  rw h2 at *,
  rw h_sum,
  norm_num,
  sorry
end

end lisa_needs_additional_marbles_l206_206284


namespace complement_A_eq_l206_206946

open Set

variable {U : Set ℝ} {A : Set ℝ}

def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | x ≤ -1 ∨ x > 2}

theorem complement_A_eq : compl A = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by
  ext x
  split
  · intro h
    simp at h
    constructor
    · specialize h (Or.inl (le_refl _))
      linarith
    · specialize h (Or.inr (lt_irrefl _))
      linarith
  · intro h
    simp
    intro ha
    cases ha
    · linarith
    · linarith

end complement_A_eq_l206_206946


namespace conor_vegetables_per_week_l206_206057

theorem conor_vegetables_per_week : 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := 
by 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  show (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 
  sorry

end conor_vegetables_per_week_l206_206057


namespace solve_system_l206_206900

theorem solve_system (n : ℕ) (h_n : 5 < n) :
  ∃ (x : Fin n → ℕ),
  (x 0 + x 1 + x 2 + ∑ i in Finset.range (n - 3), x (i + 3) = n + 2) ∧
  (x 0 + 2 * x 1 + 3 * x 2 + ∑ i in Finset.range (n - 3), (i + 4) * x (i + 3) = 2n + 2) ∧
  (x 0 + 4 * x 1 + 9 * x 2 + ∑ i in Finset.range (n - 3), (i + 4)^2 * x (i + 3) = n^2 + n + 4) ∧
  (x 0 + 8 * x 1 + 27 * x 2 + ∑ i in Finset.range (n - 3), (i + 4)^3 * x (i + 3) = n^3 + n + 8) ∧
  ((∀i, 0 ≤ x i) ∧ (x 0 = n) ∧ (x 1 = 1) ∧ (∀ i, 2 ≤ i ∧ i < n - 1 → x i = 0) ∧ (x (n - 1) = 1)) := sorry

end solve_system_l206_206900


namespace triangle_statements_correct_l206_206590

theorem triangle_statements_correct :
  (∀ (A B : ℝ) (a b c : ℝ),
    (A > B) → (a = b * c / sin A) → (a / sin A = b / sin B) → (sin A > sin B)) ∧
  (∀ (A : ℝ) (R : ℝ) (a b c : ℝ),
    (A = π / 6) → (a = 5) → ¬ (R = 10) → (a / sin A = R) → (a / 2R = sin A / R) → R = 5) ∧
  (∀ (a b c : ℝ),
    (a = 2 * b * cos c) → (sin a = 2 * sin b * cos c) → (sin (b + c) = 2 * sin b * cos c) → (sin (b - c) = 0) → (b = c)) ∧
  (∀ (A : ℝ) (b c : ℝ) (area : ℝ),
    (b = 1) → (c = 2) → (A = 2 * π / 3) → (area = 1 / 2 * b * c * sin A) → (area = sqrt 3 / 2) → area = sqrt 3 / 2) :=
by
  sorry

end triangle_statements_correct_l206_206590


namespace apples_distribution_l206_206957

theorem apples_distribution (total_apples : ℝ) (apples_per_person : ℝ) (number_of_people : ℝ) 
    (h1 : total_apples = 45) (h2 : apples_per_person = 15.0) : number_of_people = 3 :=
by
  sorry

end apples_distribution_l206_206957


namespace find_valid_polynomials_l206_206085

-- Definitions for the problem conditions
def polynomial (n : ℕ) (a : Fin (n+1) → ℝ) : Polynomial ℝ :=
  ∑ i in Finset.range (n+1), Polynomial.C (a ⟨i, Finset.mem_range.2 (Nat.lt_succ_self i)⟩) * (Polynomial.X ^ i)

def valid_polynomial (p : Polynomial ℝ) (n : ℕ) : Prop :=
  let coeffs := (Finset.range (n+1)).image (λ i, p.coeff i)
  coeffs = Finset.range (n+1) ∧ (∃ x : Fin (n+1) → ℝ, ∀ k : Fin (n+1), is_root p (x k))

-- The main theorem statement in Lean 4
theorem find_valid_polynomials : 
  ∀ (n : ℕ), n ∈ {1, 2, 3} →
  ∀ p : Polynomial ℝ, valid_polynomial p n →
  p = ∑ i in Finset.range (n+1), (if i = 0 then 1 else if i = 1 then 2 else if i = n then 3 else 0) * Polynomial.X ^ i ∨
  p = ∑ i in Finset.range (n+1), (if i = 0 then 2 else if i = 1 then 3 else if i = n then 1 else 0) * Polynomial.X ^ i :=
by
  sorry

end find_valid_polynomials_l206_206085


namespace repeating_decimal_as_fraction_l206_206077

-- Define the repeating decimal 0.36666... as a real number
def repeating_decimal : ℝ := 0.366666666666...

-- State the theorem to express the repeating decimal as a fraction
theorem repeating_decimal_as_fraction : repeating_decimal = (11 : ℝ) / 30 := 
sorry

end repeating_decimal_as_fraction_l206_206077


namespace card_division_count_correct_l206_206722

noncomputable def count_card_divisions (cards : Finset ℕ) (participants : ℕ) : ℕ :=
if h : participants ≥ 1 ∧ all_cards_distributed cards participants then 4596 else 0

theorem card_division_count_correct :
  count_card_divisions (Finset.range' 15 19) 3 = 4596 :=
by sorry

end card_division_count_correct_l206_206722


namespace coincide_Q_Q_l206_206491

variable {A B C D P Q Q' S : Type}
variable [coord_space ℝ A]
variable [coord_space ℝ B]
variable [coord_space ℝ C]
variable [coord_space ℝ D]
variable [point_on BC P]   -- P is a point on [BC]
variable [parallel AP CQ]
variable [parallel DP BQ']
variable [intersect_line BC AD S]
variable [is_trapezoid ABCD AB CD]  -- (AB)//(CD) and AB < CD 
variable [intersect A D Q CQ]
variable [intersect B D Q' BQ']

theorem coincide_Q_Q' 
  (parallelogram_condition1 : parallel (line_segment AP) (line_segment CQ))
  (parallelogram_condition2 : parallel (line_segment DP) (line_segment BQ'))
  (trapezoid_ABCD : is_trapezoid ABCD (line AB) (line CD))
  (intersect_C_AD_at_Q : intersect (line_segment AD) (line_parallel_through C))
  (intersect_B_AD_at_Q' : intersect (line_segment AD) (line_parallel_through B))
  : Q = Q' := 
  sorry

end coincide_Q_Q_l206_206491


namespace problem_distribution_l206_206811

theorem problem_distribution:
  let num_problems := 6
  let num_friends := 15
  (num_friends ^ num_problems) = 11390625 :=
by sorry

end problem_distribution_l206_206811


namespace card_division_count_correct_l206_206723

noncomputable def count_card_divisions (cards : Finset ℕ) (participants : ℕ) : ℕ :=
if h : participants ≥ 1 ∧ all_cards_distributed cards participants then 4596 else 0

theorem card_division_count_correct :
  count_card_divisions (Finset.range' 15 19) 3 = 4596 :=
by sorry

end card_division_count_correct_l206_206723


namespace together_finish_work_in_10_days_l206_206764

theorem together_finish_work_in_10_days (x_days y_days : ℕ) (hx : x_days = 15) (hy : y_days = 30) :
  let x_rate := 1 / (x_days : ℚ)
  let y_rate := 1 / (y_days : ℚ)
  let combined_rate := x_rate + y_rate
  let total_days := 1 / combined_rate
  total_days = 10 :=
by
  sorry

end together_finish_work_in_10_days_l206_206764


namespace compute_sum_of_cosines_l206_206049

-- Define ε1 as a 7th root of unity
def ε1 : ℂ := exp (2 * ℂ.pi * complex.I / 7)

-- State the theorem to be proven
theorem compute_sum_of_cosines : 
  (1 / (2 * complex.cos (2 * ℂ.pi / 7)) + 
   1 / (2 * complex.cos (4 * ℂ.pi / 7)) + 
   1 / (2 * complex.cos (6 * ℂ.pi / 7))) = -2 := 
by
  sorry

end compute_sum_of_cosines_l206_206049


namespace find_initial_balance_l206_206701

-- Define the initial balance
variable (X : ℝ)

-- Conditions
def balance_tripled (X : ℝ) : ℝ := 3 * X
def balance_after_withdrawal (X : ℝ) : ℝ := balance_tripled X - 250

-- The problem statement to prove
theorem find_initial_balance (h : balance_after_withdrawal X = 950) : X = 400 :=
by
  sorry

end find_initial_balance_l206_206701


namespace distance_between_foci_eq_2sqrt7_l206_206512

def ellipse_eq (x y : ℝ) := x^2 / 16 + y^2 / 9 = 1

theorem distance_between_foci_eq_2sqrt7 :
  (∀ x y : ℝ, ellipse_eq x y) →
  (forall a b : ℝ, (a = 4) → (b = 3) → (c = sqrt (a^2 - b^2)) → (|d| = 2 * c)) →
  (|d| = 2 * sqrt(7)) :=
sorry

end distance_between_foci_eq_2sqrt7_l206_206512


namespace find_digit_A_l206_206346

theorem find_digit_A : ∃ A : ℕ, A < 10 ∧ (200 + 10 * A + 4) % 13 = 0 ∧ A = 7 :=
by
  sorry

end find_digit_A_l206_206346


namespace impossible_division_l206_206255

theorem impossible_division (heap_size : ℕ) (heap_count : ℕ) (similar_sizes : Π (a b : ℕ), a < 2 * b) :
  ¬ (∃ (heaps : Finset ℕ), heaps.card = heap_count ∧ heaps.sum = heap_size ∧ ∀ (a b ∈ heaps), similar_sizes a b) :=
by
  let heap_size := 660
  let heap_count := 31
  sorry

end impossible_division_l206_206255


namespace accounting_majors_l206_206180

theorem accounting_majors (p q r s t u : ℕ) 
  (hpqt : (p * q * r * s * t * u = 51030)) 
  (hineq : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) :
  p = 2 :=
sorry

end accounting_majors_l206_206180


namespace min_value_f_l206_206118

variables {a b c x y z : ℝ}

def f (x y z : ℝ) : ℝ :=
  (x^2 / (1 + x)) + (y^2 / (1 + y)) + (z^2 / (1 + z))

theorem min_value_f :
  0 < a → 0 < b → 0 < c → 0 < x → 0 < y → 0 < z →
  c * y + b * z = a →
  a * z + c * x = b →
  b * x + a * y = c →
  f x y z = 1 / 2 :=
by
  sorry

end min_value_f_l206_206118


namespace trajectory_and_min_distance_l206_206125

theorem trajectory_and_min_distance
    (P : ℝ × ℝ) (Q : ℝ × ℝ) (M : ℝ × ℝ)
    (hP : P.1 + 2 * P.2 = 1)
    (hQ : Q.1 + 2 * Q.2 = -3)
    (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))
    (hIneq1 : M.2 ≤ M.1 / 3 + 2)
    (hIneq2 : M.2 ≤ -M.1 + 2) :
    (M.1 + 2 * M.2 + 1 = 0) ∧
    (sqrt ((M.1)^2 + (M.2)^2) = sqrt 5 / 5) :=
begin
  sorry
end

end trajectory_and_min_distance_l206_206125


namespace Conor_can_chop_116_vegetables_in_a_week_l206_206059

-- Define the conditions
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8
def work_days_per_week : ℕ := 4

-- Define the total vegetables per day
def vegetables_per_day : ℕ := eggplants_per_day + carrots_per_day + potatoes_per_day

-- Define the total vegetables per week
def vegetables_per_week : ℕ := vegetables_per_day * work_days_per_week

-- The proof statement
theorem Conor_can_chop_116_vegetables_in_a_week : vegetables_per_week = 116 :=
by
  sorry  -- The proof step is omitted with sorry

end Conor_can_chop_116_vegetables_in_a_week_l206_206059


namespace circles_concentric_l206_206133

noncomputable theory

-- Define the equation of circle C1
def equation_circle_C1 (f : ℝ → ℝ → ℝ) (x y : ℝ) := f x y = 0

-- Define the point P(x, y) being outside circle C1
def point_outside_circle_C1 (f : ℝ → ℝ → ℝ) (x y : ℝ) := f x y > 0

-- Define the equation of circle C2
def equation_circle_C2 (f : ℝ → ℝ → ℝ) (x y : ℝ) := f x y = f x y

-- The main theorem to prove: circles C1 and C2 are concentric
theorem circles_concentric (f : ℝ → ℝ → ℝ) (x y : ℝ) (h1 : equation_circle_C1 f x y) (h2 : point_outside_circle_C1 f x y) (h3 : equation_circle_C2 f x y) : 
  ∃ cx cy r1 r2, (∀ x y, equation_circle_C1 (λ x y, (x - cx) ^ 2 + (y - cy) ^ 2 - r1^2) x y) ∧ (∀ x y, equation_circle_C2 (λ x y, (x - cx) ^ 2 + (y - cy) ^ 2 - r2^2) x y) := 
begin
  sorry
end

end circles_concentric_l206_206133


namespace points_lie_on_line_l206_206468

theorem points_lie_on_line : ∀ t : ℝ, let x := (Real.cos t)^2, y := (Real.sin t)^2 in x + y = 1 := 
by 
  intro t
  let x := (Real.cos t)^2
  let y := (Real.sin t)^2
  have h : x + y = (Real.cos t)^2 + (Real.sin t)^2 := by rfl
  rw [h, Real.cos_sq_add_sin_sq]
  exact Eq.refl 1

end points_lie_on_line_l206_206468


namespace probability_eight_distinct_numbers_l206_206368

-- Given there are eight six-sided dice
def dice_sides := 6
def num_dice := 8

-- Define the probability of rolling eight distinct numbers with eight dice
theorem probability_eight_distinct_numbers : (num_dice > dice_sides) → 
  (nat.choose num_dice dice_sides) / (dice_sides ^ num_dice) = 0 := 
by sorry

end probability_eight_distinct_numbers_l206_206368


namespace shaded_region_ratio_l206_206814

noncomputable def ratio_of_areas (AB AC BC CD : ℝ) (S A : ℝ) : Prop := 
  CD^2 = AC * BC ∧ 
  S = (π / 4) * AC * BC ∧
  A = π * CD^2 ∧ 
  S / A = 1 / 4

theorem shaded_region_ratio (AB AC BC CD S A : ℝ) (h1 : CD^2 = AC * BC) (h2 : S = (π / 4) * AC * BC) (h3 : A = π * CD^2) : 
  S / A = 1 / 4 := 
by
  have h4 : S / (π * CD^2) = 1 / 4 := by 
    rw [h3]
  rw [h2]
  rw [h1]
  rw [<-mul_assoc, div_mul_eq_mul_div, mul_div_cancel_left ((π / 4) * AC * BC) π]
  -- Final steps to reach the conclusion
  simp only [mul_comm π, mul_div_cancel, ne_of_gt (pi_gt_zero)]
  rw [one_div, mul_one_div_cancel]
  rw [mul_comm, div_eq_div_iff]
  have t1 : π * 1 = π := by simp
  field_simp
  exact t1
  sorry

end shaded_region_ratio_l206_206814


namespace dense_set_count_l206_206795

-- Define the notion of a dense set based on the given conditions
def isDenseSet (A : Set ℕ) : Prop :=
  (1 ∈ A ∧ 49 ∈ A) ∧ (∀ n ∈ A, n ≥ 1 ∧ n ≤ 49) ∧ (A.card > 40) ∧ (∀ n ∈ A, (n + 1) ∈ A → (n + 2) ∈ A → (n + 3) ∈ A → (n + 4) ∈ A → (n + 5) ∈ A → (n + 6) ∉ A)

-- State the theorem indicating the number of such sets
theorem dense_set_count : ∃ S : Finset (Set ℕ), S.card = 495 ∧ ∀ A ∈ S, isDenseSet A :=
sorry

end dense_set_count_l206_206795


namespace new_commission_percentage_l206_206405

theorem new_commission_percentage
  (fixed_salary : ℝ)
  (total_sales : ℝ)
  (sales_threshold : ℝ)
  (previous_commission_rate : ℝ)
  (additional_earnings : ℝ)
  (prev_commission : ℝ)
  (extra_sales : ℝ)
  (new_commission : ℝ)
  (new_remuneration : ℝ) :
  fixed_salary = 1000 →
  total_sales = 12000 →
  sales_threshold = 4000 →
  previous_commission_rate = 0.05 →
  additional_earnings = 600 →
  prev_commission = previous_commission_rate * total_sales →
  extra_sales = total_sales - sales_threshold →
  new_remuneration = fixed_salary + new_commission * extra_sales →
  new_remuneration = prev_commission + additional_earnings →
  new_commission = 2.5 / 100 :=
by
  intros
  sorry

end new_commission_percentage_l206_206405


namespace find_x_value_l206_206942

-- Definitions based on the conditions
def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

-- Given condition that a is parallel to b.
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ λ : ℝ, u = (λ * v.1, λ * v.2)

-- The main theorem statement
theorem find_x_value (x : ℝ) (h : is_parallel a (b x)) : x = -9 :=
sorry

end find_x_value_l206_206942


namespace cartesian_to_polar_circle_l206_206126

open Real

theorem cartesian_to_polar_circle (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = ρ * cos θ) 
  (h2 : y = ρ * sin θ) 
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * cos θ :=
sorry

end cartesian_to_polar_circle_l206_206126


namespace sum_of_ns_with_2_percent_perfect_squares_l206_206872

theorem sum_of_ns_with_2_percent_perfect_squares : 
  (∑ n in {n : ℕ | ∃ m, 2 * m * n = m * m ∧ 0.02 * n = m}, n) = 4900 :=
begin
  sorry
end

end sum_of_ns_with_2_percent_perfect_squares_l206_206872


namespace error_in_area_l206_206759

theorem error_in_area (s : ℝ) (h : s > 0) :
  let s_measured := 1.02 * s
  let A_actual := s^2
  let A_measured := s_measured^2
  let error := (A_measured - A_actual) / A_actual * 100
  error = 4.04 := by
  sorry

end error_in_area_l206_206759


namespace Robinson_overtakes_Brown_in_12_minutes_l206_206429

theorem Robinson_overtakes_Brown_in_12_minutes (Brown_lap_time Robinson_lap_time : ℕ) 
  (hBrown : Brown_lap_time = 6) 
  (hRobinson : Robinson_lap_time = 4) : 
  Robinson_overtakes_Brown_in (4 * 3) :=
by
  sorry

end Robinson_overtakes_Brown_in_12_minutes_l206_206429


namespace number_of_divisors_of_24_divisible_by_4_l206_206656

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206656


namespace constant_term_expansion_eq_4351_l206_206856

theorem constant_term_expansion_eq_4351 :
  let f (x : ℤ) := (1 + x + x⁻²) ^ 10 in
  constant_term f = 4351 :=
sorry

end constant_term_expansion_eq_4351_l206_206856


namespace cylinder_radius_in_cone_l206_206019

theorem cylinder_radius_in_cone :
  ∀ (R h : ℝ), R = 7 ∧ h = 16 → (∃ (r : ℝ), r = 56 / 15 ∧ (∃ d, d = 2 * r ∧ d = 2 * r ∧ h - d = (h / R) * r)) :=
by {
  intros R h cone_props,
  obtain ⟨R_prop, h_prop⟩ := cone_props,
  use 56 / 15,
  split,
  { refl },
  { use 2 * (56 / 15),
    split,
    { ring },
    { rw [h_prop, R_prop],
      sorry } }
}

end cylinder_radius_in_cone_l206_206019


namespace find_S_coordinates_l206_206729

-- Definitions for vertices of the parallelogram
def point := (ℝ × ℝ × ℝ)

def P : point := (2, 3, 1)
def Q : point := (4, -1, -3)
def R : point := (0, 0, 1)

def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2 ∧ M.3 = (A.3 + B.3) / 2

theorem find_S_coordinates :
  ∃ S: point, is_midpoint (1, 1.5, 1) Q S ∧ S = (-2, 4, 5) :=
sorry -- proof goes here

end find_S_coordinates_l206_206729


namespace laser_travel_distance_l206_206397

noncomputable def distance : ℝ :=
  real.sqrt ((4 - (-10)) ^ 2 + (7 - (-7)) ^ 2)

theorem laser_travel_distance :
  distance = 14 * real.sqrt 2 := by
  sorry

end laser_travel_distance_l206_206397


namespace point_relationship_l206_206635

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -(x - 1) ^ 2 + c

noncomputable def y1_def (c : ℝ) : ℝ := quadratic_function (-3) c
noncomputable def y2_def (c : ℝ) : ℝ := quadratic_function (-1) c
noncomputable def y3_def (c : ℝ) : ℝ := quadratic_function 5 c

theorem point_relationship (c : ℝ) :
  y2_def c > y1_def c ∧ y1_def c = y3_def c :=
by
  sorry

end point_relationship_l206_206635


namespace not_divide_660_into_31_not_divide_660_into_31_l206_206234

theorem not_divide_660_into_31 (H : ∀ (A B : ℕ), A ∈ S ∧ B ∈ S → A < 2 * B) : 
  false :=
begin
  -- math proof steps should go here
  sorry
end

namespace my_proof

def similar (x y : ℕ) : Prop := x < 2 * y

theorem not_divide_660_into_31 : ¬ ∃ (S : set ℕ), (∀ x ∈ S, x < 2 * y) ∧ (∑ x in S, x = 660) is ∧ |S| = 31 :=
begin
  sorry
end

end my_proof

end not_divide_660_into_31_not_divide_660_into_31_l206_206234


namespace find_k_l206_206928

theorem find_k (k : ℝ) 
  (h1 : ∀ (r s : ℝ), r + s = -k ∧ r * s = 8 → (r + 3) + (s + 3) = k) : 
  k = 3 :=
by
  sorry

end find_k_l206_206928


namespace circle_equation_l206_206109

noncomputable def symmetric_point (p : Real × Real) (l : Real × Real) : Real × Real := 
  let (x, y) := p
  let (a, b) := l
  (x - 2 * (a * x + b * y) * a / (a^2 + b^2), y - 2 * (a * x + b * y) * b / (a^2 + b^2))

theorem circle_equation :
  ∃ c : Real × Real, ∃ r : Real,
    (let sym := symmetric_point (2, 3) (1, 2) in 
    (dist (2, 3) (c.1, c.2) = r ∧ dist sym (c.1, c.2) = r) ∧
    let d := abs (3 * c.1 - c.2 + 1) / (Math.sqrt 2) in
    d^2 + (Math.sqrt 2)^2 = r^2 ∧
    ((c.1 - 6)^2 + (c.2 + 3)^2 = 52 ∨ (c.1 - 14)^2 + (c.2 + 7)^2 = 244)) :=
sorry

end circle_equation_l206_206109


namespace total_population_of_city_l206_206773

theorem total_population_of_city (P : ℝ) (h : 0.85 * P = 85000) : P = 100000 :=
  by
  sorry

end total_population_of_city_l206_206773


namespace geometric_progression_twentieth_term_natural_l206_206893

def is_geometric_progression (a : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
  a * q ^ n

theorem geometric_progression_twentieth_term_natural {a₁ : ℕ} {q : ℚ}
  (h_q_pos : 0 < q)
  (h_q_rational : ∃ (m n : ℕ), n ≠ 0 ∧ (q = m / n) ∧ nat.coprime m n)
  (h_a₁_natural : ∀ n ∈ {0, 9, 29}, is_geometric_progression a₁ q n ∈ ℕ) :
  is_geometric_progression a₁ q 19 ∈ ℕ :=
by
  sorry

end geometric_progression_twentieth_term_natural_l206_206893


namespace find_a_l206_206885

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 + 2 * x + 1

theorem find_a :
  (∫ x in -1..1, f x) = 2 * f -1 ∨ (∫ x in -1..1, f x) = 2 * f (1 / 3) :=
by
  sorry

end find_a_l206_206885


namespace modulus_of_complex_number_l206_206107

noncomputable def modulus_of_z (z : ℂ) : ℝ :=
  complex.abs z

theorem modulus_of_complex_number (z : ℂ) (hi : z / (1 + 2 * complex.I) = 1) : modulus_of_z z = real.sqrt 5 :=
by
  sorry

end modulus_of_complex_number_l206_206107


namespace f_conjecture_l206_206522

-- Define the sequence a_n
def a (n : ℕ+) : ℝ := 1 / (n + 1)^2

-- Define the function f
def f : ℕ+ → ℝ 
| ⟨0, _⟩ => 1
| n      => ∏ i in finset.range n, (1 - a (⟨i+1, i.succ_pos_of_succ⟩))

-- Define the conjecture for f(n)
theorem f_conjecture (n : ℕ+) : f n = (n + 2) / (2 * (n + 1)) :=
sorry

end f_conjecture_l206_206522


namespace equal_distribution_of_cupcakes_l206_206342

theorem equal_distribution_of_cupcakes (cupcakes children : ℕ) (h1 : cupcakes = 96) (h2 : children = 8) : cupcakes / children = 12 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by simp)

end equal_distribution_of_cupcakes_l206_206342


namespace brian_tape_needed_l206_206430

-- Define lengths and number of each type of box
def long_side_15_30 := 32
def short_side_15_30 := 17
def num_15_30 := 5

def side_40_40 := 42
def num_40_40 := 2

def long_side_20_50 := 52
def short_side_20_50 := 22
def num_20_50 := 3

-- Calculate the total tape required
def total_tape : Nat :=
  (num_15_30 * (long_side_15_30 + 2 * short_side_15_30)) +
  (num_40_40 * (3 * side_40_40)) +
  (num_20_50 * (long_side_20_50 + 2 * short_side_20_50))

-- Proof statement
theorem brian_tape_needed : total_tape = 870 := by
  sorry

end brian_tape_needed_l206_206430


namespace set_union_covers_real_line_l206_206947

open Set

def M := {x : ℝ | x < 0 ∨ 2 < x}
def N := {x : ℝ | -Real.sqrt 5 < x ∧ x < Real.sqrt 5}

theorem set_union_covers_real_line : M ∪ N = univ := sorry

end set_union_covers_real_line_l206_206947


namespace compute_sum_of_cosines_l206_206048

-- Define ε1 as a 7th root of unity
def ε1 : ℂ := exp (2 * ℂ.pi * complex.I / 7)

-- State the theorem to be proven
theorem compute_sum_of_cosines : 
  (1 / (2 * complex.cos (2 * ℂ.pi / 7)) + 
   1 / (2 * complex.cos (4 * ℂ.pi / 7)) + 
   1 / (2 * complex.cos (6 * ℂ.pi / 7))) = -2 := 
by
  sorry

end compute_sum_of_cosines_l206_206048


namespace platform_length_approximation_l206_206410

noncomputable def length_of_platform (train_length : ℝ) (train_speed : ℝ) (cross_time : ℝ) : ℝ := 
  (train_speed * cross_time) - train_length

theorem platform_length_approximation :
  length_of_platform 200  (80 * (1000 / 3600)) 22 ≈ 288.84 :=
by
  sorry

end platform_length_approximation_l206_206410


namespace solve_an_l206_206897

variable {ℕ : Type} [Nat ℕ]

def S (n : ℕ) : ℕ :=
2 * a n - 4 

def a (n : ℕ) : ℕ := 
2 ^ (n + 1)

-- The theorem to prove 
theorem solve_an (n : ℕ) : S n = 2 * a n - 4 → a n = 2 ^ (n + 1) := 
by sorry

end solve_an_l206_206897


namespace impossibility_of_dividing_stones_l206_206278

theorem impossibility_of_dividing_stones (stones : ℕ) (heaps : ℕ) (m : ℕ) :
  stones = 660 → heaps = 31 → (∀ (x y : ℕ), x ∈ heaps → y ∈ heaps → x < 2 * y → False) := sorry

end impossibility_of_dividing_stones_l206_206278


namespace shots_in_last_5_l206_206810

-- Define the conditions
def initial_shots := 20
def initial_success_rate := 0.55
def additional_shots := 5
def total_shots := initial_shots + additional_shots
def final_success_rate := 0.56

-- Translate conditions to mathematical expressions
def initial_made_shots := initial_success_rate * initial_shots
def final_made_shots := final_success_rate * total_shots

-- The statement to be proved
theorem shots_in_last_5 : final_made_shots - initial_made_shots = 3 :=
by
  sorry

end shots_in_last_5_l206_206810


namespace sum_of_b_factors_l206_206605

theorem sum_of_b_factors (T : ℤ) : 
  (T = ∑ b in { b : ℤ | ∃ r s : ℤ, r + s = -b ∧ r * s = 2023 * b ∧ r ≠ 0 ∧ s ≠ 0 }, b) → 
  |T| = 121380 :=
sorry

end sum_of_b_factors_l206_206605


namespace area_quad_abc_d_is_eight_l206_206632

/-- Proof that the area of quadrilateral ABCD is 8 square units -/
theorem area_quad_abc_d_is_eight : 
  let A := (0, 0)
  let B := (2, 2)
  let C := (4, 0)
  let D := (2, 0)
  area_of_quadrilateral A B C D = 8 := 
sorry

end area_quad_abc_d_is_eight_l206_206632


namespace mul97_eq_9409_l206_206452

theorem mul97_eq_9409 : (97 * 97 = 9409) :=
by
  calc
    97 * 97 = (100 - 3)^2   : by sorry
          ... = 100^2 - 2 * 3 * 100 + 3^2 : by sorry
          ... = 10000 - 600 + 9 : by sorry
          ... = 9409 : by sorry

end mul97_eq_9409_l206_206452


namespace monotonic_increase_intervals_range_of_f2A_l206_206951

noncomputable theory

open Real

variables {a b c : ℝ} {A B C x : ℝ}

def m_vec (x : ℝ) := (√3 * sin (x / 4), 1)
def n_vec (x : ℝ) := (cos (x / 4), cos (x / 4) ^ 2)
def f (x : ℝ) := (m_vec x).1 * (n_vec x).1 + (m_vec x).2 * (n_vec x).2

theorem monotonic_increase_intervals :
  {x : ℝ | ∃ k : ℤ, 4 * k * π - 4 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 2 * π / 3} =
  {x : ℝ | ∃ k : ℤ, f (x) = sin (x / 2 + π / 6) + 1 / 2} :=
sorry

theorem range_of_f2A
  (h1 : (2 * a - c) * cos B = b * cos C)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : B = π / 3)
  : {y : ℝ | ∃ A : ℝ, A == 2 * A ∧ f (2 * A) = y} ⊆ (√3 / 2, 3 / 2] :=
sorry

end monotonic_increase_intervals_range_of_f2A_l206_206951


namespace part_I_part_II_l206_206191

-- Part I
theorem part_I :
  ∀ (x_0 y_0 : ℝ),
  (x_0 ^ 2 + y_0 ^ 2 = 8) ∧
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) →
  ∃ a b : ℝ, (a = 2 ∧ b = 2) →
  (∀ x y : ℝ, (x - 2) ^ 2 + (y - 2) ^ 2 = 8) :=
by 
sorry

-- Part II
theorem part_II :
  ¬ ∃ (x_0 y_0 k_1 k_2 : ℝ),
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) ∧
  (k_1k_2 = (y_0^2 - 4) / (x_0^2 - 4)) ∧
  (k_1 + k_2 = 2 * x_0 * y_0 / (x_0^2 - 4)) ∧
  (k_1k_2 - (k_1 + k_2) / (x_0 * y_0) + 1 = 0) :=
by 
sorry

end part_I_part_II_l206_206191


namespace work_day_percentage_l206_206626

theorem work_day_percentage 
  (work_day_hours : ℕ) 
  (first_meeting_minutes : ℕ) 
  (second_meeting_factor : ℕ) 
  (h_work_day : work_day_hours = 10) 
  (h_first_meeting : first_meeting_minutes = 60) 
  (h_second_meeting_factor : second_meeting_factor = 2) :
  ((first_meeting_minutes + second_meeting_factor * first_meeting_minutes) / (work_day_hours * 60) : ℚ) * 100 = 30 :=
sorry

end work_day_percentage_l206_206626


namespace sum_xyz_eq_10_l206_206901

theorem sum_xyz_eq_10 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * x * y + 3 * x * y * z = 115) : 
  x + y + z = 10 :=
sorry

end sum_xyz_eq_10_l206_206901


namespace positive_diff_between_solutions_l206_206835

theorem positive_diff_between_solutions : 
  let eq := (x : ℝ) → (7 - x^2 / 4) ^ (1/3) = -3
  ∃ (x₁ x₂ : ℝ), eq x₁ ∧ eq x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 136 := 
by
  sorry

end positive_diff_between_solutions_l206_206835


namespace number_of_possible_values_l206_206663

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206663


namespace hyperbola_eccentricity_l206_206123

noncomputable def hyperbola_data : Type :=
ℝ × ℝ

def F1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ × ℝ := (-c, 0)
def F2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ × ℝ := (c, 0)
def P (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (y : ℝ) : ℝ × ℝ :=
⟨-2a, y⟩ -- P is on the left branch

def is_symmetric (P F2 : ℝ × ℝ) (L : ℝ -> ℝ): Prop :=
L (fst P) = snd P

def eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (P F2 : ℝ × ℝ) :
  is_symmetric P (F2 a b h1 h2) (λ x, (b / a) * x) →
  eccentricity a b h1 h2 = sqrt 5 :=
by sorry

end hyperbola_eccentricity_l206_206123


namespace intersection_of_A_and_B_l206_206146

open Set

variable {α : Type} [PartialOrder α]

noncomputable def A := { x : ℝ | -1 < x ∧ x < 1 }
noncomputable def B := { x : ℝ | 0 < x }

theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 0 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_l206_206146


namespace dense_set_count_l206_206794

-- Define the notion of a dense set based on the given conditions
def isDenseSet (A : Set ℕ) : Prop :=
  (1 ∈ A ∧ 49 ∈ A) ∧ (∀ n ∈ A, n ≥ 1 ∧ n ≤ 49) ∧ (A.card > 40) ∧ (∀ n ∈ A, (n + 1) ∈ A → (n + 2) ∈ A → (n + 3) ∈ A → (n + 4) ∈ A → (n + 5) ∈ A → (n + 6) ∉ A)

-- State the theorem indicating the number of such sets
theorem dense_set_count : ∃ S : Finset (Set ℕ), S.card = 495 ∧ ∀ A ∈ S, isDenseSet A :=
sorry

end dense_set_count_l206_206794


namespace triangle_statements_correct_l206_206591

theorem triangle_statements_correct :
  (∀ (A B : ℝ) (a b c : ℝ),
    (A > B) → (a = b * c / sin A) → (a / sin A = b / sin B) → (sin A > sin B)) ∧
  (∀ (A : ℝ) (R : ℝ) (a b c : ℝ),
    (A = π / 6) → (a = 5) → ¬ (R = 10) → (a / sin A = R) → (a / 2R = sin A / R) → R = 5) ∧
  (∀ (a b c : ℝ),
    (a = 2 * b * cos c) → (sin a = 2 * sin b * cos c) → (sin (b + c) = 2 * sin b * cos c) → (sin (b - c) = 0) → (b = c)) ∧
  (∀ (A : ℝ) (b c : ℝ) (area : ℝ),
    (b = 1) → (c = 2) → (A = 2 * π / 3) → (area = 1 / 2 * b * c * sin A) → (area = sqrt 3 / 2) → area = sqrt 3 / 2) :=
by
  sorry

end triangle_statements_correct_l206_206591


namespace average_salary_rest_workers_l206_206995

-- Define the conditions
def total_workers : Nat := 21
def average_salary_all_workers : ℝ := 8000
def number_of_technicians : Nat := 7
def average_salary_technicians : ℝ := 12000

-- Define the task
theorem average_salary_rest_workers :
  let number_of_rest := total_workers - number_of_technicians
  let total_salary_all := average_salary_all_workers * total_workers
  let total_salary_technicians := average_salary_technicians * number_of_technicians
  let total_salary_rest := total_salary_all - total_salary_technicians
  let average_salary_rest := total_salary_rest / number_of_rest
  average_salary_rest = 6000 :=
by
  sorry

end average_salary_rest_workers_l206_206995


namespace largest_prime_divisor_1100_to_1150_l206_206734

theorem largest_prime_divisor_1100_to_1150 : 
  (∀ n, 1100 ≤ n ∧ n ≤ 1150 → (n > 1 ∧ (∀ p : ℕ, nat.prime p ∧ p ≤ ⌊sqrt 1150⌋ → ¬ (p ∣ n)) → nat.prime n)) :=
by sorry

end largest_prime_divisor_1100_to_1150_l206_206734


namespace harmonic_sum_bound_l206_206357

theorem harmonic_sum_bound (n : ℕ) (h : n ≥ 2) :
  (∑ i in Finset.range (2^n), 1 / (i + 1) : ℝ) < n := 
sorry

end harmonic_sum_bound_l206_206357


namespace function_is_odd_then_a_is_zero_function_monotonicity_on_interval_l206_206919

theorem function_is_odd_then_a_is_zero (a : ℝ) :
  (∀ x ∈ Ioc (-1 : ℝ) 1, (f : ℝ → ℝ) = fun x ↦ (x + a) / (x^2 + 1) ∧ f (-x) = -f x) → a = 0 :=
sorry

theorem function_monotonicity_on_interval :
  (∀ x ∈ Ioc (-1 : ℝ) 1, (f : ℝ → ℝ) = fun x ↦ x / (x^2 + 1)) →
  (∀ x1 x2 ∈ Ioc (-1 : ℝ) 1, x1 < x2 → f x1 < f x2) :=
sorry

end function_is_odd_then_a_is_zero_function_monotonicity_on_interval_l206_206919


namespace monochromatic_shapes_exists_l206_206550

theorem monochromatic_shapes_exists (P : ℝ × ℝ → Prop) (hP : ∀ x : ℝ × ℝ, P x ∨ ¬P x) :
  ∃ (A B C : ℝ × ℝ), 
    (distance A B = 2 ∧ distance B C = 2 ∧ distance C A = 2 ∨
    distance A B = sqrt 3 ∧ distance B C = sqrt 3 ∧ distance C A = sqrt 3 ∨
    distance A B = 1 ∧ distance A C = 1 ∧ (distance B C = distance A C ∨ distance B C = distance A B)) ∧
    (P A ∧ P B ∧ P C ∨ ¬P A ∧ ¬P B ∧ ¬P C) :=
by
  sorry

end monochromatic_shapes_exists_l206_206550


namespace min_m_n_sum_value_l206_206611

noncomputable def min_m_n_sum {m n : ℝ} (h_m : 0 < m) (h_n : 0 < n) 
  (tangent_condition : (m+1)^2 + (n+1)^2 = 4) : ℝ :=
m + n

theorem min_m_n_sum_value 
  {m n : ℝ} 
  (h_m : 0 < m) 
  (h_n : 0 < n) 
  (tangent_condition : (m+1)^2 + (n+1)^2 = 4) :
  min_m_n_sum h_m h_n tangent_condition = 2 + 2 * real.sqrt 2 :=
sorry

end min_m_n_sum_value_l206_206611


namespace Kayla_total_items_l206_206218

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l206_206218


namespace grid_coloring_count_l206_206062

theorem grid_coloring_count (n : ℕ) : 
  let grid := (n-1) * (n-1)
  in ∃ coloring_schemes : ℕ, 
       (∀ (scheme : Fin (2 ^ (n * n))), 
          (∀ (i j : Fin n, i < n - 1 ∧ j < n - 1), 
             exactly_two_red_vertices i j scheme) →
          coloring_count grid coloring_schemes) = 2 ^ (n + 1) - 2 := 
sorry

end grid_coloring_count_l206_206062


namespace repeating_decimal_fraction_l206_206083

theorem repeating_decimal_fraction : (0.366666... : ℝ) = 33 / 90 := 
sorry

end repeating_decimal_fraction_l206_206083


namespace num_five_digit_integers_l206_206961

theorem num_five_digit_integers :
  ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, (x >= 10^4 ∧ x < 10^5 ∧ (∀ d ∈ (digits 10 x), d = 3 ∨ d = 6) ∧ (digits 10 x).sum % 9 = 0) → ∃ is_valid : ℕ, is_valid = x) :=
by {
  sorry
}

end num_five_digit_integers_l206_206961


namespace cannot_divide_660_stones_into_31_heaps_l206_206274

theorem cannot_divide_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i, 0 < heaps i ∧ heaps i < 2 * heaps (succ i % 31)) :=
sorry

end cannot_divide_660_stones_into_31_heaps_l206_206274


namespace max_min_difference_f_l206_206705

noncomputable def f (x : ℝ) : ℝ := abs (Real.sin x) + Real.sin (2 * x) + abs (Real.cos x)

theorem max_min_difference_f : 
  let max_val := Real.sqrt 2 + 1,
      min_val := 1 in
  max_val - min_val = Real.sqrt 2 :=
by
  sorry

end max_min_difference_f_l206_206705


namespace arithmetic_progression_sum_l206_206579

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_sum
    (a1 d : ℤ)
    (h : a 9 a1 d = a 12 a1 d / 2 + 3) :
  S 11 a1 d = 66 := 
by 
  sorry

end arithmetic_progression_sum_l206_206579


namespace number_of_possible_values_l206_206660

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206660


namespace heap_division_impossible_l206_206263

theorem heap_division_impossible (n : ℕ) (k : ℕ) (similar : ℕ → ℕ → Prop)
  (h_similar : ∀ a b, similar a b ↔ a < 2 * b ∧ 2 * a > b) :
  n = 660 → k = 31 → ¬(∃ (heaps : Fin k → ℕ), ∑ i, heaps i = n ∧ ∀ i j, similar (heaps i) (heaps j)) :=
sorry

end heap_division_impossible_l206_206263


namespace solve_quadratic_for_q_l206_206850

-- Define the quadratic equation and the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main theorem statement
theorem solve_quadratic_for_q : ∃ q : ℝ, q ≠ 0 ∧ (discriminant q (-8) 2 = 0) → q = 8 :=
by
  -- Insert the assumptions and proof context here.
  -- However, since we were instructed not to consider the solution steps
  -- the proof is skipped with a "sorry".
  sorry

end solve_quadratic_for_q_l206_206850


namespace evaluate_A_l206_206842

noncomputable def A : ℝ := 
  cos (5 * Real.pi / 2 + 1 / 2 * Real.arcsin (3 / 5)) ^ 6 + 
  cos (7 * Real.pi / 2 - 1 / 2 * Real.arcsin (4 / 5)) ^ 6

theorem evaluate_A : A = 0.009 :=
by {
  let α := Real.arcsin (3 / 5),
  let β := Real.arcsin (4 / 5),
  have h_cos_α : Real.cos α = 4 / 5 := by sorry,
  have h_cos_β : Real.cos β = 3 / 5 := by sorry,
  have h_cos5π2α : Real.cos (5 * Real.pi / 2 + 1 / 2 * α) = Real.sin (1 / 2 * α) := by sorry,
  have h_cos7π2β : Real.cos (7 * Real.pi / 2 - 1 / 2 * β) = Real.sin (1 / 2 * β) := by sorry,
  have h_sin_half_α : Real.sin (1 / 2 * α) = 1 / (Real.sqrt 10) := by sorry,
  have h_sin_half_β : Real.sin (1 / 2 * β) = 1 / (Real.sqrt 5) := by sorry,
  calc 
    A = (Real.sin (1 / 2 * α))^6 + (Real.sin (1 / 2 * β))^6 : by rw [h_cos5π2α, h_cos7π2β]
    ... = (1 / (Real.sqrt 10))^6 + (1 / (Real.sqrt 5))^6 : by rw [h_sin_half_α, h_sin_half_β]
    ... = 0.009 : by sorry
}

end evaluate_A_l206_206842


namespace average_of_11_numbers_l206_206321

theorem average_of_11_numbers (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ)
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = 58)
  (h2 : (a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 6 = 65)
  (h3 : a₆ = 78) : 
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 11 = 60 := 
by 
  sorry 

end average_of_11_numbers_l206_206321


namespace impossible_division_l206_206258

theorem impossible_division (heap_size : ℕ) (heap_count : ℕ) (similar_sizes : Π (a b : ℕ), a < 2 * b) :
  ¬ (∃ (heaps : Finset ℕ), heaps.card = heap_count ∧ heaps.sum = heap_size ∧ ∀ (a b ∈ heaps), similar_sizes a b) :=
by
  let heap_size := 660
  let heap_count := 31
  sorry

end impossible_division_l206_206258


namespace third_term_of_geometric_sequence_l206_206783

theorem third_term_of_geometric_sequence
  (a₁ : ℕ) (a₄ : ℕ)
  (h1 : a₁ = 5)
  (h4 : a₄ = 320) :
  ∃ a₃ : ℕ, a₃ = 80 :=
by
  sorry

end third_term_of_geometric_sequence_l206_206783


namespace angle_PB1Q_lt_angle_PRQ_l206_206892

theorem angle_PB1Q_lt_angle_PRQ
  (a : ℝ)
  (P Q R B1 : ℝ × ℝ × ℝ)
  (cube : Set (ℝ × ℝ × ℝ))
  (cube_eq : cube = {p | ∃ x y z, p = (x, y, z) ∧ 0 ≤ x ∧ x ≤ 2*a ∧ 0 ≤ y ∧ y ≤ 2*a ∧ 0 ≤ z ∧ z ≤ 2*a})
  (P_midpoint_AA1 : P = (a, 0, 0))
  (Q_midpoint_CD : Q = (a, a, 2*a))
  (R_midpoint_B1C1 : R = (a, 2*a, a))
  (B1_position : B1 = (2*a, a, 0))
  : ∠ (P, B1, Q) < ∠ (P, R, Q) := sorry

end angle_PB1Q_lt_angle_PRQ_l206_206892


namespace find_a_l206_206476

noncomputable def problem_statement (a : ℝ) : Prop :=
(1 + a * complex.I) * complex.I = 3 + complex.I

theorem find_a (a : ℝ) (h : problem_statement a) : a = -3 :=
sorry

end find_a_l206_206476


namespace students_with_uncool_parents_l206_206989

def total_students : ℕ := 40
def cool_dads_count : ℕ := 18
def cool_moms_count : ℕ := 20
def both_cool_count : ℕ := 10

theorem students_with_uncool_parents :
  total_students - (cool_dads_count + cool_moms_count - both_cool_count) = 12 :=
by sorry

end students_with_uncool_parents_l206_206989


namespace decreasing_interval_find_a_min_value_l206_206137

noncomputable def f (x a : ℝ) : ℝ := Real.log x - (x - a) / x

theorem decreasing_interval (a : ℝ) (ha : a > 0) (x : ℝ) :
  (∃ x : ℝ, f' (1, a) = -1 ∧ a = 2) → 
  ∃ I: Set ℝ, x ∈ I ∧ I = Set.Ioo 0 2 ∧ @MonotoneOn ℝ ℝ _ _ (λ x, f x 2) I :=
begin
  sorry
end

theorem find_a_min_value (a : ℝ) (ha : a > 0) :
  (∃ fmin : ℝ, ∀ x ∈ Set.Icc (1:ℝ) 3, f x a ≥ fmin ∧ fmin = 1 / 3) →
  a = Real.exp (1 / 3) :=
begin
  sorry
end

end decreasing_interval_find_a_min_value_l206_206137


namespace intercepts_sum_l206_206785

theorem intercepts_sum : 
  let xi := (λ (x y: ℝ), y - 5 = 3 * (x - 9)) 0
  let yi := (λ (x y: ℝ), y - 5 = 3 * (x - 9)) 0
  xi + yi = -44 / 3 :=
by 
  -- Definitions of xi and yi
  let xi := (9 - (5 / 3))
  let yi := (-27 + 5)
  -- Aggregate the results
  have sum_x_y_intercepts := (xi) + (yi)
  -- Process the calculations
  have hx := (9 -  (5/3) : ℝ)
  have hy := (-22 : ℝ)
  have hsum := hx + hy
  -- Final assertion
  rw[hx, hy]
  calc ((9 - (5/3))) + (-22) = -44 / 3
    sorry

end intercepts_sum_l206_206785


namespace probability_three_primes_l206_206425

def prime_between_one_and_twelve := {2, 3, 5, 7, 11}

theorem probability_three_primes :
  (∃ (p : ℚ), p = (3500 : ℚ) / 20736) :=
begin
  -- conditions
  let faces := 12,
  let primes_count := 5,
  let dice := 4,
  let successes := 3,

  -- probabilities
  let p_prime := primes_count / faces,
  let p_not_prime := 1 - p_prime,
  let binom_coeff := nat.choose dice successes,

  -- calculation
  let probability := binom_coeff * p_prime^successes * p_not_prime^(dice - successes),
  
  -- desired outcome
  use probability,
  norm_cast,
  norm_num,
  simp,
end

end probability_three_primes_l206_206425


namespace determine_sum_l206_206800

theorem determine_sum (P R : ℝ) (h : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 78) : 
  P = 2600 :=
sorry

end determine_sum_l206_206800


namespace impossibility_of_dividing_stones_l206_206275

theorem impossibility_of_dividing_stones (stones : ℕ) (heaps : ℕ) (m : ℕ) :
  stones = 660 → heaps = 31 → (∀ (x y : ℕ), x ∈ heaps → y ∈ heaps → x < 2 * y → False) := sorry

end impossibility_of_dividing_stones_l206_206275


namespace flippy_number_divisible_by_6_count_l206_206158

-- Definitions from conditions
def is_flippy_number (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  (List.length digits = 5) ∧
  digits.nth 0 = digits.nth 2 ∧
  digits.nth 2 = digits.nth 4 ∧
  digits.nth 1 = digits.nth 3 ∧
  digits.nth 0 ≠ digits.nth 1

def is_divisible_by_2 (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop :=
  n.digits 10.sum % 3 = 0

def is_divisible_by_6 (n : ℕ) : Prop :=
  is_divisible_by_2 n ∧ is_divisible_by_3 n

def valid_flippy_number (n : ℕ) : Prop :=
  is_flippy_number n ∧ is_divisible_by_6 n

-- Main theorem statement
theorem flippy_number_divisible_by_6_count : 
  {n : ℕ | valid_flippy_number n}.to_finset.card = 12 :=
sorry

end flippy_number_divisible_by_6_count_l206_206158


namespace midpoint_distance_inequality_equality_case_1_equality_case_2_l206_206189

-- Definition of points and segments in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Distance function between two points in 3D space
def distance (p1 p2 : Point3D) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2).sqrt

-- Midpoint function between two points
def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2, z := (p1.z + p2.z) / 2 }

-- Definitions for points
variables (A B A1 B1 : Point3D)
local notation "M" => midpoint A B
local notation "M1" => midpoint A1 B1

-- Main theorem statement: Prove the distance inequality for midpoints M and M1
theorem midpoint_distance_inequality :
  let AA1 := distance A A1
  let BB1 := distance B B1
  let MM1 := distance M M1
  1 / 2 * |AA1 - BB1| ≤ MM1 ∧ MM1 ≤ 1 / 2 * (AA1 + BB1) :=
sorry

-- Case 1: MM1 = 1/2 (AA1 + BB1) holds when AA1 and BB1 are parallel and have the same direction
theorem equality_case_1 :
  let AA1 := distance A A1
  let BB1 := distance B B1
  let MM1 := distance M M1
  -- Assume condition for AA1 parallel to BB1 and same direction
  (AA1//BB1 : Bool) -> (MM1 = 1 / 2 * (AA1 + BB1)) :=
sorry

-- Case 2: MM1 = 1/2 |AA1 - BB1| holds when AA1 and BB1 are parallel but have opposite directions
theorem equality_case_2 :
  let AA1 := distance A A1
  let BB1 := distance B B1
  let MM1 := distance M M1
  -- Assume condition for AA1 parallel to BB1 but opposite direction
  (AA1 ≠ BB1 : Bool) -> (MM1 = 1 / 2 * |AA1 - BB1|) :=
sorry

end midpoint_distance_inequality_equality_case_1_equality_case_2_l206_206189


namespace avg_visitors_sundays_l206_206006

-- Definitions
def days_in_month := 30
def avg_visitors_per_day_month := 750
def avg_visitors_other_days := 700
def sundays_in_month := 5
def other_days := days_in_month - sundays_in_month

-- Main statement to prove
theorem avg_visitors_sundays (S : ℕ) 
  (H1 : days_in_month = 30) 
  (H2 : avg_visitors_per_day_month = 750) 
  (H3 : avg_visitors_other_days = 700) 
  (H4 : sundays_in_month = 5) 
  (H5 : other_days = days_in_month - sundays_in_month) 
  :
  (sundays_in_month * S + other_days * avg_visitors_other_days) = avg_visitors_per_day_month * days_in_month 
  → S = 1000 :=
by 
  sorry

end avg_visitors_sundays_l206_206006


namespace age_difference_l206_206718
noncomputable theory
open_locale classical

variable {A B C : ℕ}

-- Condition: The total age of A and B is 12 years more than the total age of B and C.
axiom condition : A + B = B + C + 12

-- Theorem: How many years younger is C than A?
theorem age_difference : A - C = 12 := by
  -- proof goes here
  sorry

end age_difference_l206_206718


namespace transformed_function_is_correct_l206_206352

-- Define the original function f(x)
def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 6)

-- Define the transformation applied to x as per the given conditions
def transform (x : ℝ) : ℝ := 2 * x

-- Define the resulting function g(x) after applying the transformation
def g (x : ℝ) : ℝ := Real.cos (transform x + Real.pi / 3)

-- Proof statement in Lean
theorem transformed_function_is_correct : 
  ∀ x : ℝ, g x = Real.cos (2 * x + Real.pi / 3) :=
by
  intro x
  unfold g
  unfold transform
  sorry

end transformed_function_is_correct_l206_206352


namespace cupcakes_sold_l206_206878

theorem cupcakes_sold (initial additional final sold : ℕ) (h1 : initial = 14) (h2 : additional = 17) (h3 : final = 25) :
  initial + additional - final = sold :=
by
  sorry

end cupcakes_sold_l206_206878


namespace no_division_660_stones_into_31_heaps_l206_206247

theorem no_division_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) :=
  sorry

end no_division_660_stones_into_31_heaps_l206_206247


namespace not_divide_660_into_31_not_divide_660_into_31_l206_206239

theorem not_divide_660_into_31 (H : ∀ (A B : ℕ), A ∈ S ∧ B ∈ S → A < 2 * B) : 
  false :=
begin
  -- math proof steps should go here
  sorry
end

namespace my_proof

def similar (x y : ℕ) : Prop := x < 2 * y

theorem not_divide_660_into_31 : ¬ ∃ (S : set ℕ), (∀ x ∈ S, x < 2 * y) ∧ (∑ x in S, x = 660) is ∧ |S| = 31 :=
begin
  sorry
end

end my_proof

end not_divide_660_into_31_not_divide_660_into_31_l206_206239


namespace number_of_chess_games_l206_206567

theorem number_of_chess_games (A B : ℕ) (total_players : ℕ) (no_games_between_A : true) (no_games_between_B : true) :
  total_players = 6 ∧ A = 3 ∧ B = 3 ∧ no_games_between_A ∧ no_games_between_B →
  A * B = 9 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  rw [h2, h3]
  exact mul_eq_mul_right_iff.mpr (or.intro_left _ rfl)

end number_of_chess_games_l206_206567


namespace oldest_child_age_l206_206784

theorem oldest_child_age (number_of_children : ℕ) (average_age : ℕ) 
  (ages : Fin number_of_children → ℕ)
  (distinct_ages: ∀ i j : Fin number_of_children, i ≠ j → ages i ≠ ages j)
  (diff_one_year : ∀ i j : Fin number_of_children, (i < j) → (ages j = ages i + (j.1 - i.1)))
  (h_number_of_children : number_of_children = 7)
  (h_average_age : average_age = 8)
  (h_avg : (∑ i, ages ⟨i, by linarith [h_number_of_children.symm]⟩) / number_of_children = average_age) :
  (ages ⟨number_of_children - 1, by linarith [h_number_of_children.symm]⟩) = 11 := by
  sorry

end oldest_child_age_l206_206784


namespace part_I_part_II_l206_206108

variable {R : Type} [LinearOrderedField R]

theorem part_I (a b : R) (h1 : a^2 + b^2 = a + b) (h2 : 0 < a) (h3 : 0 < b) : 
  (1 / a) + (1 / b) ≥ 2 := sorry

theorem part_II : 
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ (a^2 + b^2 = a + b) ∧ ((a + 1) * (b + 1) = 4) :=
begin
  use [1, 1],
  split, exact zero_lt_one,
  split, exact zero_lt_one,
  split, 
  { simp },
  { simp }
end

end part_I_part_II_l206_206108


namespace ellipse_eccentricity_range_l206_206494

/-- Given an ellipse with the equation 
      x^2 / a^2 + y^2 / b^2 = 1, a > b > 0, which always passes through the fixed point (sqrt(2)/2, sqrt(2)/2)
      and the range of lengths for its major axis is [sqrt 5, sqrt 6]. Prove that the eccentricity e lies in 
      [sqrt 3 / 3, sqrt 2 / 2].
-/
theorem ellipse_eccentricity_range :
  ∀ (a b e : ℝ),
    (∀ (x y : ℝ), 
      x^2 / a^2 + y^2 / b^2 = 1) ∧ 
    (a > b ∧ b > 0) ∧ 
    (∃ (a' : ℝ), sqrt 5 ≤ 2 * a' ∧ 2 * a' ≤ sqrt 6) ∧ 
    (∃ (x y : ℝ), x = sqrt 2 / 2 ∧ y = sqrt 2 / 2 ∧ (x^2 / a^2 + y^2 / b^2 = 1)) 
    → (∃ (e' : ℝ), e' = sqrt (1 - (b^2 / a^2)) ∧ (sqrt 3 / 3 ≤ e' ∧ e' ≤ sqrt 2 / 2)) :=
sorry

end ellipse_eccentricity_range_l206_206494


namespace pulley_weight_l206_206399

theorem pulley_weight (M g : ℝ) (hM_pos : 0 < M) (F : ℝ := 50) :
  (g ≠ 0) → (M * g = 100) :=
by
  sorry

end pulley_weight_l206_206399


namespace set_listing_method_l206_206523

theorem set_listing_method :
  {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 5} = {0, 1, 2} :=
by
  sorry

end set_listing_method_l206_206523


namespace constant_term_in_binomial_expansion_l206_206101

theorem constant_term_in_binomial_expansion :
  let a := ∫ x in (0 : ℝ)..π, Real.sin x
  a = 2 → 
  (a * Real.sqrt x - 1 / Real.sqrt x) ^ 6 = 
  ∑ k in Finset.range (6 + 1), 
    (↑((Nat.choose 6 k) * (-1 : ℝ) ^ k) * 
      (2 : ℝ) ^ (6 - k) * 
      x ^ (3 - k / 2)) →
  ∑ k in Finset.range (6 + 1) | 3 - k / 2 = 0 | 
    (↑((Nat.choose 6 k) * (-1 : ℝ) ^ k) * 
      (2 : ℝ) ^ (6 - k)) = 1 :=
by
  sorry

end constant_term_in_binomial_expansion_l206_206101


namespace modulus_div_conj_eq_one_l206_206131

noncomputable def z : ℂ := 1 - 3 * complex.i
def z_conj : ℂ := complex.conj z

theorem modulus_div_conj_eq_one : complex.abs (z / z_conj) = 1 := by {
  sorry
}

end modulus_div_conj_eq_one_l206_206131


namespace repeat_decimal_to_fraction_l206_206080

theorem repeat_decimal_to_fraction : 0.36666 = 11 / 30 :=
by {
    sorry
}

end repeat_decimal_to_fraction_l206_206080


namespace totient_ratio_l206_206066

theorem totient_ratio :
  let phi_n (x : ℕ) := λ (n x : ℕ), (list.range n).filter (λ y, gcd x y = 1) .length,
    m := Nat.lcm 2016 6102,
    m_m := m^m in
  (φ_{m_m}(2016) /φ_{m_m}(6102) = 339 / 392) := by
  sorry

end totient_ratio_l206_206066


namespace math_problem_solution_l206_206441

noncomputable def common_roots_sum : Nat :=
  let C : ℤ := 0 -- C is the given integer in p(x)
  let D : ℤ := 0 -- D is the given integer in q(x)
  let a : ℕ := 20
  let b : ℕ := 3
  let c : ℕ := 2
  a + b + c

theorem math_problem_solution : common_roots_sum = 25 :=
  by
  let C := 0
  let D := 0
  let a := 20
  let b := 3
  let c := 2
  show 20 + 3 + 2 = 25 by
    calc 20 + 3 + 2 = 25 : by rfl


end math_problem_solution_l206_206441


namespace Kayla_total_items_l206_206216

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l206_206216


namespace words_on_each_page_l206_206385

theorem words_on_each_page (p : ℕ) (h1 : p ≤ 120) (h2 : 150 * p % 221 = 210) : p = 48 :=
sorry

end words_on_each_page_l206_206385


namespace equivalence_l206_206637

-- Non-computable declaration to avoid the computational complexity.
noncomputable def is_isosceles_right_triangle (x₁ x₂ : Complex) : Prop :=
  x₂ = x₁ * Complex.I ∨ x₁ = x₂ * Complex.I

-- Definition of the polynomial roots condition.
def roots_form_isosceles_right_triangle (a b : Complex) : Prop :=
  ∃ x₁ x₂ : Complex,
    x₁ + x₂ = -a ∧
    x₁ * x₂ = b ∧
    is_isosceles_right_triangle x₁ x₂

-- Main theorem statement that matches the mathematical equivalency.
theorem equivalence (a b : Complex) : a^2 = 2*b ∧ b ≠ 0 ↔ roots_form_isosceles_right_triangle a b :=
sorry

end equivalence_l206_206637


namespace minimize_energy_l206_206204

-- Definition of constants and variables
variables (k : ℝ) (v : ℝ) (t : ℝ)
-- Given conditions
def time_taken (v : ℝ) : ℝ := 100 / (v - 3)
def energy (k : ℝ) (v : ℝ) : ℝ := k * v^3 * (100 / (v - 3))

-- Proof statement
theorem minimize_energy (k_pos : k > 0) : 
  (∃ v > 3, ∀ u > 3, (energy k v <= energy k u)) ↔ v = 4.5 := sorry

end minimize_energy_l206_206204


namespace projection_midpoint_l206_206358

open EuclideanGeometry

variables {A B C C₀ A₁ B₁ C₀' A₁' B₁' : Point}

def midpoint (AB : Line) (C₀ : Point) : Prop := C₀ = (midpoint A B)

def altitude_foot (A BC A₁ : Point) : Prop := is_foot_of_perpendicular A BC A₁ 

def altitude_foot (B CA B₁ : Point) : Prop := is_foot_of_perpendicular B CA B₁

def circumcircle (ABC : Triangle) : Circle := Circle ⟨A, B, C⟩ (radius_of_triangle ABC)

def tangent_at_C (circ : Circle) (C : Point) : Line := tangent_line circ C

def projection (P t P' : Point) : Prop := same_projection P t P'

theorem projection_midpoint 
  (triangle_ABC : Triangle)
  (circumcircle_ABC : Circle := circumcircle triangle_ABC)
  (tangent_at_C : Line := tangent_at_C circumcircle_ABC C)
  (mid : midpoint AB C₀)
  (foot_A : altitude_foot A BC A₁)
  (foot_B : altitude_foot B CA B₁)
  (proj_C₀ : projection C₀ tangent_at_C C₀')
  (proj_A₁ : projection A₁ tangent_at_C A₁')
  (proj_B₁ : projection B₁ tangent_at_C B₁') :
  C₀' = midpoint A₁' B₁' :=
begin
  sorry
end

end projection_midpoint_l206_206358


namespace max_value_of_f_l206_206865

noncomputable def f (x : Real) : Real :=
  9 * Real.sin x + 12 * Real.cos x

theorem max_value_of_f : ∃ M, ∀ x, f x ≤ M ∧ (∃ y, f y = M) := 
  ∃ M, M = 15 ∧ (∀ x, f x ≤ 15) ∧ (f (Real.atan (4/3)) = 15) :=
sorry

end max_value_of_f_l206_206865


namespace find_m3_minus_2mn_plus_n3_l206_206956

theorem find_m3_minus_2mn_plus_n3 (m n : ℝ) (h1 : m^2 = n + 2) (h2 : n^2 = m + 2) (h3 : m ≠ n) : m^3 - 2 * m * n + n^3 = -2 := by
  sorry

end find_m3_minus_2mn_plus_n3_l206_206956


namespace f_leq_2x_l206_206481

variables (f : ℝ → ℝ)

axiom f_domain : ∀ x, 0 ≤ x → x ≤ 1 → ∃ y, f y = x ∧ f(y) ∈ (0 : set ℝ)

axiom f_1 : f 1 = 1
axiom f_nonneg : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x
axiom f_superadditive : ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y ≤ 1 → f (x + y) ≥ f x + f y

theorem f_leq_2x : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x :=
begin
  sorry
end

end f_leq_2x_l206_206481


namespace number_of_possible_values_of_b_l206_206667

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206667


namespace number_of_squares_l206_206966

open Int

theorem number_of_squares (n : ℕ) (h : n < 10^7) : 
  (∃ n, 36 ∣ n ∧ n^2 < 10^07) ↔ (n = 87) :=
by sorry

end number_of_squares_l206_206966


namespace balloon_height_is_correct_l206_206874

/-- Assume five points A, B, C, D, O positioned on a flat field such that:
  - A is directly north of O.
  - B is directly west of O.
  - C is directly south of O.
  - D is directly east of O.
  - The distance between C and D is 160 m.
  A hot-air balloon is located directly above O at point H.
  The balloon is anchored by four ropes HA, HB, HC, and HD such that:
  - Rope HC has a length of 160 m.
  - Rope HD has a length of 140 m.
  Prove that the height of the balloon above the field (OH) is 80 * (real.sqrt 2). -/
theorem balloon_height_is_correct (c d h : ℝ) (h₁ : h ^ 2 + c ^ 2 = 160 ^ 2) (h₂ : h ^ 2 + d ^ 2 = 140 ^ 2)
  (h₃ : c ^ 2 + d ^ 2 = 160 ^ 2) (h₄ : c = d) : h = 80 * real.sqrt 2 :=
by
  sorry

end balloon_height_is_correct_l206_206874


namespace that_remaining_money_l206_206344

section
/-- Initial money in Olivia's wallet --/
def initial_money : ℕ := 53

/-- Money collected from ATM --/
def collected_money : ℕ := 91

/-- Money spent at the supermarket --/
def spent_money : ℕ := collected_money + 39

/-- Remaining money after visiting the supermarket --
Theorem that proves Olivia's remaining money is 14 dollars.
-/
theorem remaining_money : initial_money + collected_money - spent_money = 14 := 
by
  unfold initial_money collected_money spent_money
  simp
  sorry
end

end that_remaining_money_l206_206344


namespace evening_minivans_l206_206219

theorem evening_minivans (total_minivans afternoon_minivans : ℕ) (h_total : total_minivans = 5) 
(h_afternoon : afternoon_minivans = 4) : total_minivans - afternoon_minivans = 1 := 
by
  sorry

end evening_minivans_l206_206219


namespace sum_lt_or_eq_1_1_l206_206985

theorem sum_lt_or_eq_1_1 :
  (∀ l : List ℝ, l = [1.4, 9/10, 1.2, 0.5, 13/10] → 
  l.filter (λ x, x ≤ 1.1) = [9/10, 0.5] → 
  l.filter (λ x, x ≤ 1.1).sum = 1.4) :=
by
  intro l h1 h2
  rw [h2]
  norm_num
  sorry

end sum_lt_or_eq_1_1_l206_206985


namespace right_isosceles_triangle_areas_l206_206641

theorem right_isosceles_triangle_areas :
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  A + B = C :=
by
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  sorry

end right_isosceles_triangle_areas_l206_206641


namespace fifth_stack_33_l206_206647

def cups_in_fifth_stack (a d : ℕ) : ℕ :=
a + 4 * d

theorem fifth_stack_33 
  (a : ℕ) 
  (d : ℕ) 
  (h_first_stack : a = 17) 
  (h_pattern : d = 4) : 
  cups_in_fifth_stack a d = 33 := by
  sorry

end fifth_stack_33_l206_206647


namespace cube_root_simplification_l206_206309

-- Define the given condition
def factorized_450 : Prop := 450 = 2 * 3^2 * 5^2

-- Prove the required simplification
theorem cube_root_simplification (h : factorized_450) : 
  Real.cbrt 450 = Real.cbrt 2 * (3^(2/3)) * (5^(2/3)) := 
by
  sorry

end cube_root_simplification_l206_206309


namespace find_angle_A_find_tan_C_l206_206498

-- Import necessary trigonometric identities and basic Lean setup
open Real

-- First statement: Given the dot product condition, find angle A
theorem find_angle_A (A : ℝ) (h1 : cos A + sqrt 3 * sin A = 1) :
  A = 2 * π / 3 := 
sorry

-- Second statement: Given the trigonometric condition, find tan C
theorem find_tan_C (B C : ℝ)
  (h1 : 1 + sin (2 * B) = 2 * (cos B ^ 2 - sin B ^ 2))
  (h2 : B + C = π) :
  tan C = (5 * sqrt 3 - 6) / 3 := 
sorry

end find_angle_A_find_tan_C_l206_206498


namespace gibraltar_initial_population_stable_l206_206711
-- Import necessary libraries

-- Define constants based on conditions
def full_capacity := 300 * 4
def initial_population := (full_capacity / 3) - 100
def population := 300 -- This is the final answer we need to validate

-- The main theorem to prove
theorem gibraltar_initial_population_stable : initial_population = population :=
by 
  -- Proof is skipped as requested
  sorry

end gibraltar_initial_population_stable_l206_206711


namespace part1_equiv_points_part2_b_value_part3_m_range_l206_206444

-- Part 1
theorem part1_equiv_points (f g : ℝ → ℝ) :
  (f = λ x, 2 * x + 1 → ∃ x, f x = x ∧ x = -1) ∧
  (g = λ x, x^2 - x + 2 → ¬ ∃ x, g x = x) := 
by sorry

-- Part 2
theorem part2_b_value (A B C : ℝ × ℝ) (b : ℝ) :
  (A = (3, 3) ∧ B = (b / 2, b / 2) ∧ C = (b / 2, 0) ∧ (3 * (3 - b / 2)) / 2 = 3 → 
    b = 3 + Real.sqrt 33 ∨ b = 3 - Real.sqrt 33) := 
by sorry

-- Part 3
theorem part3_m_range (m : ℝ) :
  ((∀ x, x < m → x^2 - 4 = x ∧ x = (1 - Real.sqrt 17) / 2 ∧ x = (1 + Real.sqrt 17) / 2) ∧ 
    (¬ ∃ x, (x^2-4=x ∧ x< (-17/8)) ∨ ((1-Real.sqrt 17)/2 < x ∧ x < (1+Real.sqrt 17)/2)) → 
    m < -17 / 8 ∨ (1 - Real.sqrt 17) / 2 < m ∧ m < (1 + Real.sqrt 17) / 2) := 
by sorry

end part1_equiv_points_part2_b_value_part3_m_range_l206_206444


namespace cyclic_quadrilateral_properties_l206_206379

open Real

-- Definitions based on given conditions
def radius : ℝ := 4 
def similar_triangle_ADP_QAB (ADP QAB : Triangle) : Prop := 
  ∀ ⦃A B D P Q⦄, is_cyclic_quadrilateral A B C D ∧ point_exists P ∧ point_exists Q ∧ triangle_similar ADP QAB

def is_cyclic_quadrilateral (A B C D : Point) : Prop := 
  ∃ (O : Point) (r : ℝ), is_inscribed O r A B C D ∧ r = 4

def is_inscribed (O : Point) (r : ℝ) (A B C D : Point) : Prop :=
  distance O A = r ∧ distance O B = r ∧ distance O C = r ∧ distance O D = r

-- Value from problem part (a)
def AC := 8

-- Ratio given in problem part (b)
def ratio_CK_KT_TA := 3:1:4 

-- Statement in Lean
theorem cyclic_quadrilateral_properties (A B C D P Q : Point) 
(hc : is_cyclic_quadrilateral A B C D) 
(hi : point_exists P) 
(hj : point_exists Q) 
(hs : similar_triangle_ADP_QAB ⟨A D P⟩ ⟨Q A B⟩) :
AC = 4 + 4 ∧ ∠DAC = 45 ∧ quadrilateral_area ABCD = 31 := 
by
  -- Math proof steps go here
  sorry

end cyclic_quadrilateral_properties_l206_206379


namespace matrix_identity_l206_206826

noncomputable def M (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 3*y, z; 2*x, y, -z; 2*x, -y, z]

theorem matrix_identity (x y z : ℝ) (h : M x y zᵀ ⬝ M x y z = 1) :
  x^2 + y^2 + z^2 = 145 / 264 := sorry

end matrix_identity_l206_206826


namespace triangles_from_sticks_l206_206645

theorem triangles_from_sticks (a1 a2 a3 a4 a5 a6 : ℕ) (h_diff: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 
∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 
∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 
∧ a4 ≠ a5 ∧ a4 ≠ a6 
∧ a5 ≠ a6) (h_order: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) : 
  (a1 + a3 > a5 ∧ a1 + a5 > a3 ∧ a3 + a5 > a1) ∧ 
  (a2 + a4 > a6 ∧ a2 + a6 > a4 ∧ a4 + a6 > a2) :=
by
  sorry

end triangles_from_sticks_l206_206645


namespace both_players_score_same_points_l206_206593

theorem both_players_score_same_points :
  let P_A_score := 0.5 
  let P_B_score := 0.8 
  let P_A_miss := 1 - P_A_score
  let P_B_miss := 1 - P_B_score
  let P_both_miss := P_A_miss * P_B_miss
  let P_both_score := P_A_score * P_B_score
  let P_same_points := P_both_miss + P_both_score
  P_same_points = 0.5 := 
by {
  -- Actual proof should be here
  sorry
}

end both_players_score_same_points_l206_206593


namespace Kayla_total_items_l206_206215

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l206_206215


namespace cricket_game_initial_overs_l206_206568

theorem cricket_game_initial_overs
    (run_rate_initial : ℝ)
    (run_rate_remaining : ℝ)
    (remaining_overs : ℕ)
    (target_score : ℝ)
    (initial_overs : ℕ) :
    run_rate_initial = 3.2 →
    run_rate_remaining = 5.25 →
    remaining_overs = 40 →
    target_score = 242 →
    initial_overs = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_game_initial_overs_l206_206568


namespace problem_statement_l206_206121

theorem problem_statement (a b : ℝ) (h : sqrt (a + 2) + abs (b - 1) = 0) : (a + b)^2017 = -1 :=
sorry

end problem_statement_l206_206121


namespace problem_1_problem_2_l206_206134

noncomputable def f (x : ℝ) : ℝ := 
  sin x * cos x + sqrt 3 * (sin x) ^ 2 - sqrt 3 / 2

theorem problem_1 (hx : ∀ x ∈ Icc (0:ℝ) ((π / 2):ℝ), 
  - sqrt 3 / 2 ≤ f x ∧ f x ≤ 1) :
  ∀ x ∈ Icc (0:ℝ) ((π / 2):ℝ), 
  f x ∈ Icc (- sqrt 3 / 2) 1 :=
by
  sorry

theorem problem_2 (C : ℝ) (AB : ℝ) (A : ℝ) 
  (hC : 0 < C ∧ C < π / 2)
  (hf : f (C / 2) = -1/2)
  (hAB : AB = 3)
  (hA : A = π / 4) :
  ∃ BC : ℝ, BC = 3 * sqrt 2 :=
by
  sorry

end problem_1_problem_2_l206_206134


namespace count_integers_with_remainders_l206_206536

theorem count_integers_with_remainders : 
  {n : ℤ | 100 < n ∧ n < 200 ∧ (n % 6 = n % 8)}.finite.to_finset.card = 25 :=
by sorry

end count_integers_with_remainders_l206_206536


namespace not_divide_660_into_31_not_divide_660_into_31_l206_206238

theorem not_divide_660_into_31 (H : ∀ (A B : ℕ), A ∈ S ∧ B ∈ S → A < 2 * B) : 
  false :=
begin
  -- math proof steps should go here
  sorry
end

namespace my_proof

def similar (x y : ℕ) : Prop := x < 2 * y

theorem not_divide_660_into_31 : ¬ ∃ (S : set ℕ), (∀ x ∈ S, x < 2 * y) ∧ (∑ x in S, x = 660) is ∧ |S| = 31 :=
begin
  sorry
end

end my_proof

end not_divide_660_into_31_not_divide_660_into_31_l206_206238


namespace part1_part2_l206_206886

-- Part (I)
theorem part1 (x : ℝ) : 
  let f := λ x : ℝ, abs (2 * x + 2) + abs (x - 3) in 
  (f x <= 5) ↔ (x ≥ -4/3 ∧ x <= 0) := 
by sorry

-- Part (II)
theorem part2 (a b c : ℝ) (m : ℝ) (h1 : a + b + c = m) (h2 : m = 4) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) : 
  1 / (a + b) + 1 / (b + c) + 1 / (a + c) >= 9 / (2 * m) :=
by sorry

end part1_part2_l206_206886


namespace set_equivalence_l206_206525

-- Definitions
def U := {1, 2, 3, 4}
def M := {1, 3, 4}
def N := {1, 2}
def C (U: Set ℕ) (A: Set ℕ) := U \ A

-- Proof statement
theorem set_equivalence : C U M ∪ C U N = {2, 3, 4} :=
by
  sorry

end set_equivalence_l206_206525


namespace range_of_m_in_triangle_condition_l206_206479

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3*x + m

theorem range_of_m_in_triangle_condition : 
  (∀ a b c ∈ set.Icc 0 2, a ≠ b → b ≠ c → a ≠ c → 
     (let fa := f a m in let fb := f b m in let fc := f c m in
     fa + fb > fc ∧ fb + fc > fa ∧ fc + fa > fb)) → 
  m > 2 :=
sorry

end range_of_m_in_triangle_condition_l206_206479


namespace sum_of_coefficients_evaluated_l206_206448

theorem sum_of_coefficients_evaluated 
  (x y : ℤ) (h1 : x = 2) (h2 : y = -1)
  : (3 * x + 4 * y)^9 + (2 * x - 5 * y)^9 = 387420501 := 
by
  rw [h1, h2]
  sorry

end sum_of_coefficients_evaluated_l206_206448


namespace paint_board_unique_pair_l206_206958

theorem paint_board_unique_pair :
  ∃! (a b : ℕ), 
    (∀ (i j : ℕ), i < 8 ∧ j < 8 → 
       (∀ (k l : ℕ), k ≤ 3 ∧ l ≤ 3 → (∑ i in finset.range 3, ∑ j in finset.range 3, board (i + k) (j + l) = a)) 
     ∧ (∀ (m n : ℕ), m ≤ 4 ∧ n ≤ 2 → (∑ i in finset.range 2, ∑ j in finset.range 4, board (m + i) (n + j) = b))) 
  := (a, b) = (9, 8) :=
by
  sorry

end paint_board_unique_pair_l206_206958


namespace problem_statement_l206_206905

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℤ := {-1, 1}

theorem problem_statement : M ∩ N = {1} := 
by sorry

end problem_statement_l206_206905


namespace lim_one_eq_exp_lim_two_eq_exp_neg_two_lim_three_eq_exp_neg_ten_lim_four_eq_exp_neg_one_l206_206382

-- 1. \( \lim_{n \rightarrow \infty} \left(1 + \frac{a}{n}\right)^{n} = e^{a} \)
noncomputable def limit_one : ℝ → ℝ
| a := (exp(a))

theorem lim_one_eq_exp {a : ℝ} :
  (filter.at_top.map (λ n:ℕ, (1 + a / (n:ℝ)) ^ n)).lim = limit_one a := 
sorry

-- 2. \( \lim_{x \rightarrow 0} \sqrt[x]{1 - 2x} = e^{-2} \)
noncomputable def limit_two : ℝ :=
  (exp (-2))

theorem lim_two_eq_exp_neg_two :
  (filter.tendsto (λ x:ℝ, (1 - 2*x) ^ (1 / x)) filter.lo_at_zero filter.at_bot) : Prop :=
sorry

-- 3. \( \lim_{t \rightarrow \infty} \left(\frac{t - 3}{t + 2}\right)^{2t + 1} = e^{-10} \)
noncomputable def limit_three : ℝ :=
  (exp (-10))

theorem lim_three_eq_exp_neg_ten :
  (filter.at_top.map (λ t:ℕ, ((t:ℝ - 3) / (t + 2)) ^ (2 * t + 1))).lim = limit_three := 
  sorry

-- 4. \( \lim_{x \rightarrow \frac{\pi}{4}} (\tan x)^{\tan(2x)} = e^{-1} \)
noncomputable def limit_four : ℝ :=
  (exp (-1))

theorem lim_four_eq_exp_neg_one :
  (filter.tendsto (λ x:ℝ, (Real.tan x) ^ Real.tan (2 * x)) 
  (filter.nhds (Real.pi / 4)) filter.at_bot) : Prop :=
  sorry

end lim_one_eq_exp_lim_two_eq_exp_neg_two_lim_three_eq_exp_neg_ten_lim_four_eq_exp_neg_one_l206_206382


namespace volume_of_solid_of_revolution_l206_206046

theorem volume_of_solid_of_revolution (a b : ℝ) (h_a : 0 ≤ a) :
  (π * b^2 * ∫ x in 0..a, (1 - (x^2 / a^2))) = (2 * π * a * b^2 / 3) :=
by
  sorry

end volume_of_solid_of_revolution_l206_206046


namespace expected_value_calc_l206_206752

-- Define the length of the sequence we are looking for
def sequence_length : ℕ := 2012

-- Define the string S of length 2012, which is 'HTHT...HT'
noncomputable def S : string := (List.replicate (sequence_length / 2) "HT").asString

-- Define t as the first time the sequence S appears
def t (n : ℕ) : ℕ := -- function that returns the first time S appears (details skipped)

-- Define the expected value of t
noncomputable def E_t : ℕ := ∑ n in (Finset.range (sequence_length + 1)), n * probability (t n = n)

-- The expected value E(t) should match the calculated answer
theorem expected_value_calc : E_t = (2 ^ (sequence_length + 2) - 4) / 3 := 
by sorry

end expected_value_calc_l206_206752


namespace decreasing_interval_value_range_a_intersection_product_l206_206936

noncomputable def f (x : ℝ) : ℝ := Real.log x - (3 / 2) * x^2 - 2 * x
def critical_points (x : ℝ) : Bool := f x = (@top ℝ)

-- Define other necessary constants and functions
def F (x : ℝ) (a b : ℝ) : ℝ := ln x + a / x

def H (x1 x2 : ℝ) : Prop := x1 * x2 > 2 * Real.exp 2

theorem decreasing_interval (x : ℝ) : x ∈ Ioo 0 (1/3) :=
begin
  sorry
end

theorem value_range_a (a : ℝ) : a ≥ 15 / 8 :=
begin
  sorry
end

theorem intersection_product (x1 x2 : ℝ) : H x1 x2 :=
begin
  sorry
end

end decreasing_interval_value_range_a_intersection_product_l206_206936


namespace count_ways_to_express_1523_l206_206225

theorem count_ways_to_express_1523 :
  let count_combinations (count : ℕ) :
    ∃ (M : ℕ), M = count :=
  sorry

end count_ways_to_express_1523_l206_206225


namespace impossible_division_l206_206254

theorem impossible_division (heap_size : ℕ) (heap_count : ℕ) (similar_sizes : Π (a b : ℕ), a < 2 * b) :
  ¬ (∃ (heaps : Finset ℕ), heaps.card = heap_count ∧ heaps.sum = heap_size ∧ ∀ (a b ∈ heaps), similar_sizes a b) :=
by
  let heap_size := 660
  let heap_count := 31
  sorry

end impossible_division_l206_206254


namespace kayla_total_items_l206_206211

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l206_206211


namespace frog_jumping_sequences_l206_206111

def regular_hexagon : Type :=
  { verts : List ℝ × ℝ // verts.length = 6 }

def adjacent (v1 v2 : ℝ × ℝ) : Prop := 
  (v1 = (0, 0) ∧ v2 = (1, 0)) ∨ 
  (v1 = (1, 0) ∧ v2 = (1/2, √3/2)) ∨ 
  (v1 = (1/2, √3/2) ∧ v2 = (-1/2, √3/2)) ∨ 
  (v1 = (-1/2, √3/2) ∧ v2 = (-1,0)) ∨ 
  (v1 = (-1,0) ∧ v2 = (-1/2, -√3/2)) ∨ 
  (v1 = (-1/2, -√3/2) ∧ v2 = (0,0)) ∨ 
  (v2 = (0, 0) ∧ v1 = (1, 0)) ∨ 
  (v2 = (1, 0) ∧ v1 = (1/2, √3/2)) ∨ 
  (v2 = (1/2, √3/2) ∧ v1 = (-1/2, √3/2)) ∨ 
  (v2 = (-1/2, √3/2) ∧ v1 = (-1,0)) ∨ 
  (v2 = (-1,0) ∧ v1 = (-1/2, -√3/2)) ∨ 
  (v2 = (-1/2, -√3/2) ∧ v1 = (0,0))

def frog (start : ℝ × ℝ) (end_ : ℝ × ℝ) (jumps : ℕ) : List (ℝ × ℝ) → Prop
| []            := false
| [v]           := v = end_
| v1 :: v2 :: l := adjacent v1 v2 ∧ frog v2 end_ l

-- Question rewritten to a Lean 4 Statement
theorem frog_jumping_sequences (A B C D E F : ℝ × ℝ) 
  (hex : regular_hexagon) 
  (start := A) 
  (stop := D) 
  (jumps : ℕ) :
  count (list (ℝ × ℝ))
    (· = (start :: List.repeat stop jumps) ∨ exists_list jumps 
    (λ x, x.head = start ∧ x.last = stop ∧ List.all x 
    (λ (x1 x2), adjacent x1 x2))) = 26 :=
sorry

end frog_jumping_sequences_l206_206111


namespace escher_consecutive_probability_l206_206630

open Classical

noncomputable def probability_Escher_consecutive (total_pieces escher_pieces: ℕ): ℚ :=
  if total_pieces < escher_pieces then 0 else (Nat.factorial (total_pieces - escher_pieces) * Nat.factorial escher_pieces) / Nat.factorial (total_pieces - 1)

theorem escher_consecutive_probability :
  probability_Escher_consecutive 12 4 = 1 / 41 :=
by
  sorry

end escher_consecutive_probability_l206_206630


namespace shaded_area_l206_206704

theorem shaded_area (r1 r2 : ℝ) 
  (h_r1 : r1 = 2) 
  (h_r2 : r2 = 1) 
  (h_touch : ∀ (i : ℕ), i < 3 → True) 
  :
  let large_semicircle_area := (1/2) * π * r1^2,
      small_semicircle_area := (1/2) * π * r2^2,
      rectangle_area := 2 * r2^2,
      unshaded_area := rectangle_area + π * r2^2
  in large_semicircle_area - unshaded_area = π - 2 :=
by
  sorry

end shaded_area_l206_206704


namespace trig_expression_evaluation_l206_206205

theorem trig_expression_evaluation 
  {α : ℝ} 
  (h_eq_root : ∃ x, 5 * x^2 - 7 * x - 6 = 0 ∧ x = sin α) 
  (h_third_quadrant : π < α ∧ α < 3 * π / 2) : 
  (sin (-α - 3 * π / 2) * cos (3 * π / 2 - α) * tan^2 (π - α)) / (cos (π / 2 - α) * sin (π / 2 + α)) = -9 / 16 :=
by
  sorry

end trig_expression_evaluation_l206_206205


namespace oranges_distribution_l206_206625

def initialOranges (x : ℝ) : ℝ := x
def firstSon (x : ℝ) : ℝ := (x / 2) + (1 / 2)
def afterFirstSon (x : ℝ) : ℝ := x - (firstSon x) -- = (x / 2) - (1 / 2)
def secondSon (x : ℝ) : ℝ := (1 / 2) * (afterFirstSon x) + (1 / 2)
def afterSecondSon (x : ℝ) : ℝ := afterFirstSon x - (secondSon x) -- = (x / 4) - (3 / 4)
def thirdSon (x : ℝ) : ℝ := (1 / 2) * (afterSecondSon x) + (1 / 2)

theorem oranges_distribution (x : ℝ) (h : thirdSon x = initialOranges x - (firstSon x + secondSon x)) : 
  x = 7 := 
sorry

end oranges_distribution_l206_206625


namespace selection_of_village_assistants_l206_206096

theorem selection_of_village_assistants (A B C : ℕ) (n : ℕ) (total_candidates : ℕ) (r : ℕ) :
  (total_candidates = 10) → (r = 3) → (C ≠ total_candidates) → 
  (A ≠ total_candidates) → (B ≠ total_candidates) → 
  ∑ i in (Finset.Ico 1 (total_candidates - 2)), 
    choose (total_candidates - 1) r - choose (total_candidates - 3) r = 49 := by
  intro h1 h2 h3 h4 h5
  sorry

end selection_of_village_assistants_l206_206096


namespace ceil_eq_intervals_l206_206084

theorem ceil_eq_intervals (x : ℝ) :
  (⌈⌈ 3 * x ⌉ + 1 / 2⌉ = ⌈ x - 2 ⌉) ↔ (-1 : ℝ) ≤ x ∧ x < -2 / 3 := 
by
  sorry

end ceil_eq_intervals_l206_206084


namespace radii_of_inscribed_circles_equal_l206_206297

-- Define the arbelos, circles, and relevant geometry
noncomputable def arbelos (B : Type) [metric_space B] [proper_space B] :=
  sorry -- Details of arbelos can be formally defined if needed

variables {B : Type} [metric_space B] [proper_space B]
variables {D B' : B} {C₁ C₂ : set B} -- C₁ and C₂ represent the inscribed circles

def perpendicular (D B' : B) : Prop := sorry -- Define perpendicularity

def inscribed (C₁ C₂ : set B) (D B' : B) : Prop :=
  -- Circles C₁ and C₂ are inscribed on opposite sides of the perpendicular DB
  sorry

def radius_eq (C₁ C₂ : set B) : Prop :=
  -- The radii of circles C₁ and C₂ are equal
  sorry

theorem radii_of_inscribed_circles_equal (D B' : B) (C₁ C₂ : set B) :
  perpendicular D B' → inscribed C₁ C₂ D B' → radius_eq C₁ C₂ :=
begin
  sorry -- The proof of the theorem
end

end radii_of_inscribed_circles_equal_l206_206297


namespace parabolas_intersect_at_points_l206_206834

theorem parabolas_intersect_at_points :
  ∃ (x y : ℝ), (y = 3 * x^2 - 5 * x + 1 ∧ y = 4 * x^2 + 3 * x + 1) ↔ ((x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233)) := 
sorry

end parabolas_intersect_at_points_l206_206834


namespace ratio_Raphael_to_Manny_l206_206627

-- Define the pieces of lasagna each person will eat
def Manny_pieces : ℕ := 1
def Kai_pieces : ℕ := 2
def Lisa_pieces : ℕ := 2
def Aaron_pieces : ℕ := 0
def Total_pieces : ℕ := 6

-- Calculate the remaining pieces for Raphael
def Raphael_pieces : ℕ := Total_pieces - (Manny_pieces + Kai_pieces + Lisa_pieces + Aaron_pieces)

-- Prove that the ratio of Raphael's pieces to Manny's pieces is 1:1
theorem ratio_Raphael_to_Manny : Raphael_pieces = Manny_pieces :=
by
  -- Provide the actual proof logic, but currently leaving it as a placeholder
  sorry

end ratio_Raphael_to_Manny_l206_206627


namespace employee_new_annual_salary_l206_206038

noncomputable def new_salary (increase : ℝ) (percent_increase : ℝ) : ℝ :=
  let original_salary := increase / percent_increase in
  original_salary + increase

theorem employee_new_annual_salary :
  new_salary 25000 0.3846153846153846 = 90000 :=
by
  -- Sorry, to complete the proof.
  sorry

end employee_new_annual_salary_l206_206038


namespace skew_and_perpendicular_l206_206466

/-- Given an isosceles right triangle ABC, after folding along the median of the hypotenuse BC to form tetrahedron ABCD, the lines AD and BC are skew and perpendicular. -/
theorem skew_and_perpendicular (A B C D : Point) (h_triangle : isosceles_right_triangle A B C)
  (h_fold : fold_along_median_to_tetrahedron A B C D) :
  skew_lines AD BC ∧ is_perpendicular AD BC :=
sorry

end skew_and_perpendicular_l206_206466


namespace greatest_distance_inner_outer_square_l206_206025

noncomputable def side_length_of_square_perimeter (p : ℝ) : ℝ := p / 4

theorem greatest_distance_inner_outer_square
    (perimeter_inner perimeter_outer : ℝ)
    (h_inner : perimeter_inner = 24)
    (h_outer: perimeter_outer = 32) :
  let s_i := side_length_of_square_perimeter perimeter_inner,
      s_o := side_length_of_square_perimeter perimeter_outer
  in s_o - s_i = 2
  ∧ let distance := Real.sqrt ((s_o - s_i) / 2 ^ 2 + (s_o - s_i) / 2 ^ 2)
  in distance = Real.sqrt 2 := 
by {
  sorry
}

end greatest_distance_inner_outer_square_l206_206025


namespace sum_of_digits_of_R_l206_206175

def vehicles := { k : ℕ | 1 ≤ k ∧ k ≤ 12 }
def refuels_at (t : ℕ) (k : ℕ) : Prop := t % k = 0
def is_refueling_simultaneously (t : ℕ) : Prop := ∃ S ⊆ vehicles, S.card ≥ 7 ∧ ∀ k ∈ S, refuels_at t k

noncomputable def R : ℕ := Nat.find (λ t, is_refueling_simultaneously t)

def digit_sum (n : ℕ) : ℕ := Nat.digits 10 n |>.sum

theorem sum_of_digits_of_R : digit_sum R = 6 := by
  sorry

end sum_of_digits_of_R_l206_206175


namespace words_on_each_page_l206_206384

theorem words_on_each_page (p : ℕ) (h1 : p ≤ 120) (h2 : 150 * p % 221 = 210) : p = 48 :=
sorry

end words_on_each_page_l206_206384


namespace problem_statement_l206_206232

-- Define the conditions:
def f (x : ℚ) : ℚ := sorry

axiom f_mul (a b : ℚ) : f (a * b) = f a + f b
axiom f_int (n : ℤ) : f (n : ℚ) = (n : ℚ)

-- The problem statement:
theorem problem_statement : f (8/13) < 0 :=
sorry

end problem_statement_l206_206232


namespace exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l206_206758

theorem exists_five_integers_sum_fifth_powers (A B C D E : ℤ) : 
  ∃ (A B C D E : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 + E^5 :=
  by
    sorry

theorem no_four_integers_sum_fifth_powers (A B C D : ℤ) : 
  ¬ ∃ (A B C D : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 :=
  by
    sorry

end exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l206_206758


namespace compute_expr_l206_206055

noncomputable def expr : ℝ :=
  (1 / 2) ^ (-1) - 27 ^ (-1 / 3) - (Real.log 4) / (Real.log 8)

theorem compute_expr : expr = 1 := by
  sorry

end compute_expr_l206_206055


namespace cannot_divide_660_stones_into_31_heaps_l206_206272

theorem cannot_divide_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i, 0 < heaps i ∧ heaps i < 2 * heaps (succ i % 31)) :=
sorry

end cannot_divide_660_stones_into_31_heaps_l206_206272


namespace probability_of_drawing_two_white_balls_l206_206574

noncomputable def probability_two_white_balls : ℚ :=
  let total_balls := 3 in   -- Total number of balls
  let white_ball_count := 1 in  -- Number of white balls
  (white_ball_count / total_balls) * (white_ball_count / total_balls)

theorem probability_of_drawing_two_white_balls :
  probability_two_white_balls = 1 / 9 :=
sorry

end probability_of_drawing_two_white_balls_l206_206574


namespace ellipse_equation_isosceles_triangle_l206_206898

theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : 2 * a = 6)
  (M_eq : ∀x y : ℝ, ((x - 2)^2 + y^2 = 40 / 9) → ((x - 2)^2 + y^2 = 40 / 9)) 
  (common_chord_len : 4 * real.sqrt (10)/3 = 4 * real.sqrt (10)/3) :
  ∃ C, C = ∀x y : ℝ, (x^2 / 9 + y^2 / 8 = 1) :=
sorry

theorem isosceles_triangle (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : 2 * a = 6)
  (h5 : k > 0) (h6 : ∀x y : ℝ, ((x - 2)^2 + y^2 = 40 / 9) → ((x - 2)^2 + y^2 = 40 / 9)) 
  (common_chord_len : 4 * real.sqrt (10)/3 = 4 * real.sqrt (10)/3) 
  (line_through_P : ∀x : ℝ, (2 + k * x) = 2 + k * x)
  (h7 : ∀x y : ℝ, (y = k * x + 2 → x^2 / 9 + y^2 / 8 = 1)) :
  ∃ (m : ℝ), - real.sqrt(2) / 12 ≤ m ∧ m < 0 :=
sorry

end ellipse_equation_isosceles_triangle_l206_206898


namespace solve_system_l206_206950

-- Define the conditions
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 8) ∧ (2 * x - y = 7)

-- Define the proof problem statement
theorem solve_system : 
  system_of_equations 5 3 :=
by
  -- Proof will be filled in here
  sorry

end solve_system_l206_206950


namespace incorrect_statement_A_l206_206288

theorem incorrect_statement_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) :
  ¬ (a - a^2 > b - b^2) := sorry

end incorrect_statement_A_l206_206288


namespace count_mod_9_eq_1_l206_206538

theorem count_mod_9_eq_1 : (Finset.filter (λ x : ℕ, x % 9 = 1) (Finset.range 201)).card = 23 :=
by
  sorry

end count_mod_9_eq_1_l206_206538


namespace sum_of_smallest_solutions_l206_206331

noncomputable def floor : ℝ → ℤ := λ (x : ℝ), Int.floor x

theorem sum_of_smallest_solutions :
  let x₁ := 2 + 1
  let x₂ := 3 + 2/3
  let x₃ := 4 + 1/2
  x₁ + x₂ + x₃ = (11 : ℚ) + (1 : ℚ)/6 :=
by
  sorry

end sum_of_smallest_solutions_l206_206331


namespace count_valid_n_l206_206160

theorem count_valid_n : (finset.filter (λ n : ℤ, 200 < n ∧ n < 300 ∧ (∃ r : ℤ, 0 ≤ r ∧ r < 7 ∧ n % 7 = r ∧ n % 9 = r)) (finset.Ico 201 300)).card = 7 :=
begin
  sorry
end

end count_valid_n_l206_206160


namespace fraction_of_males_on_time_l206_206041

theorem fraction_of_males_on_time (A : ℕ) :
  (2 / 9 : ℚ) * A = (2 / 9 : ℚ) * A → 
  (2 / 3 : ℚ) * A = (2 / 3 : ℚ) * A → 
  (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) = (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) → 
  ((7 / 9 : ℚ) * A - (5 / 18 : ℚ) * A) / ((2 / 3 : ℚ) * A) = (1 / 2 : ℚ) :=
by
  intros h1 h2 h3
  sorry

end fraction_of_males_on_time_l206_206041


namespace area_within_circle_l206_206320

theorem area_within_circle (A B C K M N : Point) (a : ℝ) (h_area : area A B C = 1)
  (h_AC : dist A C = 2 * dist B C) (h_mid_K : midpoint A C K)
  (h_AB_mid : AM = MN ∧ MN = NB ∧ circle_mid K AB AM MN NB) :
  enclosed_area A B C K M N = (2 * π * sqrt 3 / 27) + (1 / 6) :=
by sorry

end area_within_circle_l206_206320


namespace time_to_pass_man_is_14_secs_l206_206411

-- Definitions for the conditions in the problem

def length_of_train : ℝ := 350 -- in meters
def speed_of_train : ℝ := 80 * 1000 / 3600 -- converting kmph to m/s
def speed_of_man : ℝ := 10 * 1000 / 3600 -- converting kmph to m/s
def relative_speed : ℝ := speed_of_train + speed_of_man
def time_to_pass_man : ℝ := length_of_train / relative_speed

-- The theorem to prove
theorem time_to_pass_man_is_14_secs : time_to_pass_man = 14 := by
  sorry

end time_to_pass_man_is_14_secs_l206_206411


namespace polygon_area_is_odd_l206_206890

theorem polygon_area_is_odd {P : Polygon}
  (h_sides : P.sides = 100)
  (h_vertices : ∀ v ∈ P.vertices, v ∈ ℤ × ℤ)
  (h_parallel : ∀ e ∈ P.edges, e.parallel_to_axes)
  (h_odd_lengths : ∀ e ∈ P.edges, e.length.odd) :
  P.area.odd :=
sorry

end polygon_area_is_odd_l206_206890


namespace exists_irrational_irrational_exponentiation_rational_l206_206070

theorem exists_irrational_irrational_exponentiation_rational :
  ∃ a b : ℝ, irrational a ∧ irrational b ∧ ∃ q : ℚ, a^b = q := sorry

end exists_irrational_irrational_exponentiation_rational_l206_206070


namespace exists_infinitely_many_in_S_inter_l206_206604

noncomputable theory

def f (x : ℝ) : ℝ := x^3 - 10 * x^2 + 29 * x - 25

theorem exists_infinitely_many_in_S_inter (S : ℝ → set ℕ) :
  (∀ x, S(x) = {n : ℕ | floor(n * x)}) →
  (∃ α β : ℝ, α ≠ β ∧ f(α) = 0 ∧ f(β) = 0 ∧ 
   ∀ M : ℕ, ∃ N : ℕ, N > M ∧ N ∈ S(α) ∩ S(β)) :=
by
  sorry

end exists_infinitely_many_in_S_inter_l206_206604


namespace part1_part2_l206_206141

theorem part1 (x : ℝ) (h : x^2 - x - 3 + 1 > 0) : 
  x ∈ (-∞ : Set ℝ) ∪ Set.Ioo (-1 : ℝ) 2 ∪ (2 : ℝ) ∪ Set.Ico (2 : ℝ) ∞ :=
by sorry

theorem part2 (m : ℝ) (h : ∀ x : ℝ, x^2 - x - m + 1 > 0) : 
  m ∈ Ioo (-∞ : ℝ) (3/4 : ℝ) :=
by sorry

end part1_part2_l206_206141


namespace rectangle_inscribed_circle_l206_206015

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem rectangle_inscribed_circle (a b : ℝ) (Ha : a = 7) (Hb : b = 24) :
  let d := rectangle_diagonal a b in
  let circumference := real.pi * d in
  let area := a * b in
  d = 25 ∧ circumference = 25 * real.pi ∧ area = 168 :=
by
  sorry

end rectangle_inscribed_circle_l206_206015


namespace sum_first_7_terms_is_105_l206_206113

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables {a d : ℕ}
axiom a4_is_15 : arithmetic_seq a d 4 = 15

-- Goal/theorem to be proven
theorem sum_first_7_terms_is_105 : sum_arithmetic_seq a d 7 = 105 :=
sorry

end sum_first_7_terms_is_105_l206_206113


namespace problem_statement_l206_206434

theorem problem_statement : (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end problem_statement_l206_206434


namespace quadratic_root_value_l206_206551

theorem quadratic_root_value (m : ℝ) :
  ∃ m, (∀ x, x^2 - m * x - 3 = 0 → x = -2) → m = -1/2 :=
by
  sorry

end quadratic_root_value_l206_206551


namespace cylinder_volume_l206_206507

theorem cylinder_volume (r h : ℝ) (hrh : 2 * Real.pi * r * h = 100 * Real.pi) (h_diag : 4 * r^2 + h^2 = 200) :
  Real.pi * r^2 * h = 250 * Real.pi :=
sorry

end cylinder_volume_l206_206507


namespace two_a_plus_two_b_plus_two_c_l206_206545

variable (a b c : ℝ)

-- Defining the conditions as the hypotheses
def condition1 : Prop := b + c = 15 - 4 * a
def condition2 : Prop := a + c = -18 - 4 * b
def condition3 : Prop := a + b = 10 - 4 * c

-- The theorem to prove
theorem two_a_plus_two_b_plus_two_c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  2 * a + 2 * b + 2 * c = 7 / 3 :=
by
  sorry

end two_a_plus_two_b_plus_two_c_l206_206545


namespace kite_initial_gain_percentage_l206_206023

noncomputable def initial_gain_percentage (MP CP : ℝ) : ℝ :=
  ((MP - CP) / CP) * 100

theorem kite_initial_gain_percentage :
  ∃ MP CP : ℝ,
    SP = 30 ∧
    SP = MP * 0.9 ∧
    1.035 * CP = SP ∧
    initial_gain_percentage MP CP = 15 :=
sorry

end kite_initial_gain_percentage_l206_206023


namespace correct_propositions_l206_206813

-- Definitions for Conditions
def condition_1 {α β : Type} [Aff_dim α 3] [Aff_dim β 3] (AB CD : α → β) (p1 p2 : β) :=
  parallel α β ∧ (AB p1 = CD p2 → AB = CD)

def condition_2 {a b c : Type} [Line a] [Line b] [Line c] :=
  skew_lines a b ∧ skew_lines b c → skew_lines a c

def condition_3 {α : Type} [Plane α] : 
  ∀ p : α, ∃ l1 l2 : α, perpendicular l1 α ∧ perpendicular l2 α

def condition_4 {α β : Type} [Aff_dim α 3] [Aff_dim β 3] (P Q : α) _:=
  parallel α β ∧ (P ∈ α) ∧ parallel (Q : β) β → Q ∈ α

def condition_5 {tri : Type} [Triangle tri] (P : tri → Type) :=
  ∀ (A B C : tri → tri), equidistant P A B C → projection P (triangle_plane tri) = circumcenter tri A B C

def condition_6 {a b P : Type} [Line a] [Line b] :=
  skew_lines a b ∧ ∃ l : Plane P, perpendicular l a ∧ parallel l b

-- Proof of correct propositions
theorem correct_propositions : 
  condition_1 ∧ ¬ condition_2 ∧ ¬ condition_3 ∧ condition_4 ∧ condition_5 ∧ ¬ condition_6 :=
  by
    -- Assuming the conditions hold, we need to prove,
    -- (1) Proposition 1 is correct (AB = CD when AB ∥ CD and both lie between parallel planes \(\alpha\) and \(\beta\))
    assume c1 c4 c5, sorry
    -- (2) Proposition 2 is incorrect
    assume ¬c2, sorry
    -- (3) Proposition 3 is incorrect
    assume ¬c3, sorry
    -- (4) Proposition 4 is correct (PQ ∈ \(\alpha\) when \(P ∈ \alpha\) and \(PQ ∥ \beta\))
    assume c4, sorry
    -- (5) Proposition 5 is correct (Projection of \(P\)  produces \(O\) on the triangle plane such that \(O\) is the circumcenter)
    assume c5, sorry
    -- (6) Proposition 6 is incorrect
    assume ¬c6, sorry

end correct_propositions_l206_206813


namespace part1_part2_part3_l206_206955

-- First problem: Proving f(x) is as given
theorem part1 (x : ℝ) : 
             let a := (cos x, cos x ^ 2), b := (sin (x + π / 6), -1) in
             let f := λ x, 2 * (a.1 * b.1 + a.2 * b.2) + 1/2 in
             f x = sin (2 * x - π / 6) :=
by sorry

-- Second problem: Proving the range of m and tan(x₁ + x₂)
theorem part2 (x : ℝ) (m : ℝ) (x1 x2 : ℝ) (h_x1 : x1 ∈ [0, π / 2]) (h_x2 : x2 ∈ [0, π / 2]) :
             let g := λ x, sin (2 * (x + π / 4) - π / 6) in
             (∀ (m : ℝ), (∃ x1 x2, 2*g x - m = 1) → m ∈ [sqrt 3 - 1, 1)) ∧
             (tan (x1 + x2) = sqrt 3 / 3) :=
by sorry

-- Third problem: Proving the analytical expression for ϕ(m)
theorem part3 (m : ℝ) (h_m : m ∈ [0, π]) :
             let h := λ x, sin (x + π / 3),
                 ϕ := λ m, h(m + π / 2) - h(m) in
             (m ∈ [0, π / 6] → ϕ m = 1 - sin (m + 5 * π / 6)) ∧
             (m ∈ [π / 6, 2 * π / 3] → ϕ m = sin (m + π / 3) - sin (m + 5 * π / 6)) ∧
             (m ∈ [2 * π / 3, 11 * π / 12] → ϕ m = sin (m + π / 3) + 1) ∧
             (m ∈ [11 * π / 12, π] → ϕ m = sin (m + 5 * π / 6) + 1) :=
by sorry

end part1_part2_part3_l206_206955


namespace percent_students_with_pets_l206_206815

theorem percent_students_with_pets 
  (total_students : ℕ) (students_with_cats : ℕ) (students_with_dogs : ℕ) (students_with_both : ℕ) (h_total : total_students = 500)
  (h_cats : students_with_cats = 150) (h_dogs : students_with_dogs = 100) (h_both : students_with_both = 40) :
  (students_with_cats + students_with_dogs - students_with_both) * 100 / total_students = 42 := 
by
  sorry

end percent_students_with_pets_l206_206815


namespace amin_statement_true_l206_206340

theorem amin_statement_true :
  ∃ k : ℕ, 60 ≤ k ∧ k ≤ 89 ∧ (∃ l : ℕ, l ≥ 4 ∧ (∃ S : set (fin 90), S.card = l ∧ ∀ s ∈ S, num_acquaintances s = k))
:= 
sorry

end amin_statement_true_l206_206340


namespace calculate_probability_two_cards_sum_to_15_l206_206355

-- Define the probability calculation as per the problem statement
noncomputable def probability_two_cards_sum_to_15 : ℚ :=
  let total_cards := 52
  let number_cards := 36 -- 9 values (2 through 10) each with 4 cards
  let card_combinations := (number_cards * (number_cards - 1)) / 2 -- Total pairs to choose from
  let favourable_combinations := 144 -- Manually calculated from cases in the solution
  favourable_combinations / card_combinations

theorem calculate_probability_two_cards_sum_to_15 :
  probability_two_cards_sum_to_15 = 8 / 221 :=
by
  -- Here we ignore the proof steps and directly state it assuming the provided assumption
  admit

end calculate_probability_two_cards_sum_to_15_l206_206355


namespace rate_of_drawing_barbed_wire_per_meter_l206_206697

-- Defining the given conditions as constants
def area : ℝ := 3136
def gates_width_total : ℝ := 2 * 1
def total_cost : ℝ := 666

-- Constants used for intermediary calculations
def side_length : ℝ := Real.sqrt area
def perimeter : ℝ := 4 * side_length
def barbed_wire_length : ℝ := perimeter - gates_width_total

-- Formulating the theorem
theorem rate_of_drawing_barbed_wire_per_meter : 
    (total_cost / barbed_wire_length) = 3 := 
by
  sorry

end rate_of_drawing_barbed_wire_per_meter_l206_206697


namespace balloons_total_l206_206315

theorem balloons_total (number_of_groups balloons_per_group : ℕ)
  (h1 : number_of_groups = 7) (h2 : balloons_per_group = 5) : 
  number_of_groups * balloons_per_group = 35 := by
  sorry

end balloons_total_l206_206315


namespace perimeter_inner_pentagon_l206_206778

noncomputable theory

-- Definitions and conditions for the problem
def equiangular_star_link_length : ℝ := 1

-- Theorem statement for the perimeter of the inner pentagon
theorem perimeter_inner_pentagon :
  ∃ P : ℝ, (P = sqrt 5 - 2) ∧
           (∀ (ABCDE : ℝ), ABCDE = P) :=
begin
  sorry
end

end perimeter_inner_pentagon_l206_206778


namespace angle_relationship_diameter_inscribed_angle_angle_equality_in_triangle_l206_206064

variables {α : Type*} [linear_ordered_field α]

-- Define the points and the circle with center O
variables (O A B C : Point)
variables (α β : ℝ)

-- Angle measures
variable (θ : ℝ)

-- Condition definitions
def is_on_circle (O A B : Point) : Prop := -- fill in definition
def minor_arc_angle (O A B : Point) (θ : ℝ) : Prop := -- fill in definition
def inscribed_angle_condition (O A B C : Point) (θ : ℝ) : Prop :=
  ∠ AOB = 2 * ∠ ACB

-- Theorem statements
theorem angle_relationship 
  (h1: is_on_circle O A B)
  (h2: minor_arc_angle O A B θ)
  (h3: inscribed_angle_condition O A B C θ) :
  ∠ AOB = 2 * ∠ ACB :=
sorry

theorem diameter_inscribed_angle 
  (h1: is_on_circle O C2 B)
  (h2: ∠ OAB = 90) :
   ∠ C2AB = 90 :=
sorry

theorem angle_equality_in_triangle 
  (A B C D O : Point)
  (h1: altitude AD BC)
  (h2: center_of_circumcircle O A B C) :
  ∠ CAO = ∠ DAB :=
sorry


end angle_relationship_diameter_inscribed_angle_angle_equality_in_triangle_l206_206064


namespace find_multiple_l206_206419

-- Definitions and given conditions
def total_seats : ℤ := 387
def first_class_seats : ℤ := 77

-- The statement we need to prove
theorem find_multiple (m : ℤ) :
  (total_seats = first_class_seats + (m * first_class_seats + 2)) → m = 4 :=
by
  sorry

end find_multiple_l206_206419


namespace initial_fraction_spent_on_clothes_l206_206790

-- Define the conditions and the theorem to be proved
theorem initial_fraction_spent_on_clothes 
  (M : ℝ) (F : ℝ)
  (h1 : M = 249.99999999999994)
  (h2 : (3 / 4) * (4 / 5) * (1 - F) * M = 100) :
  F = 11 / 15 :=
sorry

end initial_fraction_spent_on_clothes_l206_206790


namespace intersection_point_on_BC_l206_206177

-- Conditions Definitions
variables (A B C M N X : Point)
variables (AX : Line)
variables (circumcircle : Circle A B C) (omega_B omega_C : Circle)

-- Assumptions
-- 1. \(M\) and \(N\) are the midpoints of \(AB\) and \(AC\) respectively.
variable (is_midpoint_M : midpoint M A B)
variable (is_midpoint_N : midpoint N A C)

-- 2. \(A\) is the point where the tangent \(AX\) meets the circumcircle of \(\triangle ABC\).
variable (tangent_AX : tangent AX circumcircle A)

-- 3. \(\omega_B\) is the circle passing through \(M\), \(B\), and tangent to \(MX\).
variable (omega_B_passes_MB : passes_through omega_B M B)
variable (tangent_omega_B : tangent (Line.mk M X) omega_B)

-- 4. \(\omega_C\) is the circle passing through \(N\), \(C\), and tangent to \(NX\).
variable (omega_C_passes_NC : passes_through omega_C N C)
variable (tangent_omega_C : tangent (Line.mk N X) omega_C)

-- Lean 4 statement to prove equivalence
theorem intersection_point_on_BC :
  ∃ D : Point, intersects_on_line omega_B omega_C D (Line.mk B C) :=
  sorry

end intersection_point_on_BC_l206_206177


namespace pow_mul_inv_eq_one_problem_theorem_l206_206363

theorem pow_mul_inv_eq_one (a : ℚ) (n : ℤ) (h_a : a ≠ 0) : 
  (a ^ (n : ℚ)) * (a ^ (-n : ℚ)) = 1 :=
by
  have h1 : a ^ (n : ℚ) * a ^ (-n : ℚ) = a ^ (n + -n : ℚ), by sorry
  have h2 : (n + -n : ℚ) = 0, by sorry
  have h3 : a ^ 0 = 1, by sorry
  rw [h1, h2, h3]

-- Constants specific to the problem
def rational_a : ℚ := 9 / 11
def int_n : ℤ := 4
def int_neg_n : ℤ := -4

-- Ensure a is nonzero
axiom a_nonzero : rational_a ≠ 0

-- The main theorem specific to the problem
theorem problem_theorem : 
  (rational_a ^ (int_n : ℚ)) * (rational_a ^ (int_neg_n : ℚ)) = 1 :=
pow_mul_inv_eq_one rational_a int_n a_nonzero

end pow_mul_inv_eq_one_problem_theorem_l206_206363


namespace arithmetic_sequence_terms_sum_l206_206996

theorem arithmetic_sequence_terms_sum
  (a : ℕ → ℝ)
  (h₁ : ∀ n, a (n+1) = a n + d)
  (h₂ : a 2 = 1 - a 1)
  (h₃ : a 4 = 9 - a 3)
  (h₄ : ∀ n, a n > 0):
  a 4 + a 5 = 27 :=
sorry

end arithmetic_sequence_terms_sum_l206_206996


namespace james_out_of_pocket_l206_206594

-- Definitions based on conditions
def old_car_value : ℝ := 20000
def old_car_sold_for : ℝ := 0.80 * old_car_value
def new_car_sticker_price : ℝ := 30000
def new_car_bought_for : ℝ := 0.90 * new_car_sticker_price

-- Question and proof statement
def amount_out_of_pocket : ℝ := new_car_bought_for - old_car_sold_for

theorem james_out_of_pocket : amount_out_of_pocket = 11000 := by
  sorry

end james_out_of_pocket_l206_206594


namespace function_satisfies_condition_l206_206851

noncomputable def f : ℕ → ℕ := sorry

theorem function_satisfies_condition (f : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → f (n + 1) > (f n + f (f n)) / 2) :
  (∃ b : ℕ, ∀ n : ℕ, (n < b → f n = n) ∧ (n ≥ b → f n = n + 1)) :=
sorry

end function_satisfies_condition_l206_206851


namespace range_f_l206_206447

noncomputable def f (x : ℝ) : ℝ := (x^3 - (1/2) * x^2) / (x^2 + 2 * x + 2)

theorem range_f : set.range f = set.univ :=
by
  sorry

end range_f_l206_206447


namespace triangle_area_l206_206197

theorem triangle_area (A B C : ℝ) (AB AC : ℝ) (A_angle : ℝ) (h1 : A_angle = π / 6)
  (h2 : AB * AC * Real.cos A_angle = Real.tan A_angle) :
  1 / 2 * AB * AC * Real.sin A_angle = 1 / 6 :=
by
  sorry

end triangle_area_l206_206197


namespace angle_in_gradians_l206_206186

noncomputable def gradians_in_full_circle : ℝ := 600
noncomputable def degrees_in_full_circle : ℝ := 360
noncomputable def angle_in_degrees : ℝ := 45

theorem angle_in_gradians :
  angle_in_degrees / degrees_in_full_circle * gradians_in_full_circle = 75 := 
by
  sorry

end angle_in_gradians_l206_206186


namespace Milly_science_homework_time_l206_206628

theorem Milly_science_homework_time
  (math_hw_time : ℕ)
  (geo_hw_time : ℕ)
  (total_study_time : ℕ)
  (math_hw_time_eq : math_hw_time = 60)
  (geo_hw_time_eq : geo_hw_time = math_hw_time / 2)
  (total_study_time_eq : total_study_time = 135) :
  total_study_time - math_hw_time - geo_hw_time = 45 :=
by
  rw [math_hw_time_eq, geo_hw_time_eq, total_study_time_eq]
  norm_num

end Milly_science_homework_time_l206_206628


namespace complement_union_intersection_l206_206497

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem complement_union_intersection :
  (compl (A ∪ B) = {x | x ≤ 2 ∨ 9 ≤ x}) ∧
  (compl (A ∩ B) = {x | x < 3 ∨ 5 ≤ x}) :=
by
  sorry

end complement_union_intersection_l206_206497


namespace intersection_eq_l206_206496

open Set

def A : Set ℝ := {x | -1 < x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_eq : A ∩ B = {0, 1} :=
by {
  sorry
}

end intersection_eq_l206_206496


namespace find_c_l206_206546

theorem find_c (a c : ℝ) (h1 : x^2 + 80 * x + c = (x + a)^2) (h2 : 2 * a = 80) : c = 1600 :=
sorry

end find_c_l206_206546


namespace exp_prop_l206_206362

theorem exp_prop : (9/11) ^ 4 * (9/11) ^ (-4) = 1 := by
  -- The properties of exponents are used here
  sorry

end exp_prop_l206_206362


namespace number_of_divisors_of_24_divisible_by_4_l206_206653

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206653


namespace top_square_after_folds_l206_206061

theorem top_square_after_folds : 
  let initial_grid := (fun (i : ℕ) (j : ℕ) => (i * 5 + j + 1)) 
    (/\ 0 <= i < 5)
    (/\ 0 <= j < 5) 
  in 
  let fold_1 := fun (grid : grid_type) => (fun (i j : ℕ) => if j < 2.5 then grid i (4 - j) else grid i j) initial_grid in
  let fold_2 := fun (grid : grid_type) => (fun (i j : ℕ) => if j >= 2.5 then grid i (4 - j) else grid i j) fold_1 in
  let fold_3 := fun (grid : grid_type) => (fun (i j : ℕ) => if i < 2.5 then grid (4 - i) j else grid i j) fold_2 in
  let fold_4 := fun (grid : grid_type) => (fun (i j : ℕ) => if i >= 2.5 then grid (4 - i) j else grid i j) fold_3 in
  fold_4 0 0 = 3 :=
by sorry

end top_square_after_folds_l206_206061


namespace percentage_increase_l206_206528

theorem percentage_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) : ((M - N) / N) * 100 = P :=
by
  sorry

end percentage_increase_l206_206528


namespace sin_neg_pi_l206_206339

theorem sin_neg_pi : Real.sin (-Real.pi) = 0 := by
  sorry

end sin_neg_pi_l206_206339


namespace claire_balloon_count_l206_206053

variable (start_balloons lost_balloons initial_give_away more_give_away final_balloons grabbed_from_coworker : ℕ)

theorem claire_balloon_count (h1 : start_balloons = 50)
                           (h2 : lost_balloons = 12)
                           (h3 : initial_give_away = 1)
                           (h4 : more_give_away = 9)
                           (h5 : final_balloons = 39)
                           (h6 : start_balloons - initial_give_away - lost_balloons - more_give_away + grabbed_from_coworker = final_balloons) :
                           grabbed_from_coworker = 11 :=
by
  sorry

end claire_balloon_count_l206_206053


namespace part_a_moments_of_all_orders_part_b_moments_of_all_orders_l206_206380

variables {X Y : ℝ → ℝ} (P Q : ℝ → ℝ → ℝ)
variables (n m : ℕ) (p : ℝ)
variables [admissible_P : ∀ x y, P x y ≠ 0 ∧ degree P = n]
variables [admissible_Q : ∀ x y, Q x y ≠ 0 ∧ degree Q = m]
variables [independent_XY : ∀ x y, X x * Y y = 0]

-- Part (a)
theorem part_a_moments_of_all_orders
  (hX : ∀ p > 0, ∃ p > 0, 𝔼|X p| < ∞)
  (hY : ∀ p > 0, ∃ p > 0, 𝔼|Y p| < ∞) :
  (∀ k > 0, 𝔼|X k| < ∞ ∧ ∀ k > 0, 𝔼|Y k| < ∞) :=
sorry

-- Part (b)
theorem part_b_moments_of_all_orders
  (hX : ∀ x, continuous X ∧ ∀ k, (X k) = 0)
  (hY : ∀ y, continuous Y ∧ ∀ k, (Y k) = 0)
  (Theorem_II_641 : ∃ H, (Theorem_II_641 H)) :
  (∀ k > 0, 𝔼|X k| < ∞ ∧ ∀ k > 0, 𝔼|Y k| < ∞) :=
sorry

end part_a_moments_of_all_orders_part_b_moments_of_all_orders_l206_206380


namespace domain_of_f_l206_206326

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3

theorem domain_of_f : ∀ x : ℝ, ∃ y : ℝ, y = f x :=
by
  intro x
  exists f x
  rfl

end domain_of_f_l206_206326


namespace heap_division_impossible_l206_206267

theorem heap_division_impossible (n : ℕ) (k : ℕ) (similar : ℕ → ℕ → Prop)
  (h_similar : ∀ a b, similar a b ↔ a < 2 * b ∧ 2 * a > b) :
  n = 660 → k = 31 → ¬(∃ (heaps : Fin k → ℕ), ∑ i, heaps i = n ∧ ∀ i j, similar (heaps i) (heaps j)) :=
sorry

end heap_division_impossible_l206_206267


namespace repeat_decimal_to_fraction_l206_206078

theorem repeat_decimal_to_fraction : 0.36666 = 11 / 30 :=
by {
    sorry
}

end repeat_decimal_to_fraction_l206_206078


namespace sam_speed_correct_l206_206032

-- Define the speeds of Alex, Jamie, and Sam
def alex_speed : ℝ := 6
def jamie_speed : ℝ := (4 / 5) * alex_speed
def sam_speed : ℝ := (3 / 4) * jamie_speed

-- The theorem to prove Sam's running speed
theorem sam_speed_correct : sam_speed = 18 / 5 :=
by 
  sorry

end sam_speed_correct_l206_206032


namespace solve_quadratic_l206_206689

theorem solve_quadratic (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = c ∨ x = d) : (c, d) = (1, -2) :=
sorry

end solve_quadratic_l206_206689


namespace kayla_total_items_l206_206210

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l206_206210


namespace no_common_root_l206_206094

theorem no_common_root (a b c d : ℝ) (ha : 0 < a) (hb : a < b) (hc : b < c) (hd : c < d) :
  ¬ ∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) :=
by
  sorry

end no_common_root_l206_206094


namespace price_per_unit_l206_206777

theorem price_per_unit (x y : ℝ) 
    (h1 : 2 * x + 3 * y = 690) 
    (h2 : x + 4 * y = 720) : 
    x = 120 ∧ y = 150 := 
by 
    sorry

end price_per_unit_l206_206777


namespace initial_number_of_numbers_is_five_l206_206322

-- Define the conditions and the given problem
theorem initial_number_of_numbers_is_five
  (n : ℕ) (S : ℕ)
  (h1 : S / n = 27)
  (h2 : (S - 35) / (n - 1) = 25) : n = 5 :=
by
  sorry

end initial_number_of_numbers_is_five_l206_206322


namespace not_divide_660_into_31_not_divide_660_into_31_l206_206233

theorem not_divide_660_into_31 (H : ∀ (A B : ℕ), A ∈ S ∧ B ∈ S → A < 2 * B) : 
  false :=
begin
  -- math proof steps should go here
  sorry
end

namespace my_proof

def similar (x y : ℕ) : Prop := x < 2 * y

theorem not_divide_660_into_31 : ¬ ∃ (S : set ℕ), (∀ x ∈ S, x < 2 * y) ∧ (∑ x in S, x = 660) is ∧ |S| = 31 :=
begin
  sorry
end

end my_proof

end not_divide_660_into_31_not_divide_660_into_31_l206_206233


namespace smallest_K_l206_206462

theorem smallest_K : ∃ K, (K = 39) ∧ (∀ S : Finset ℕ, (S ⊆ Finset.range 51) ∧ (S.card = K) → ∃ a b ∈ S, a ≠ b ∧ (a + b) ∣ (a * b)) :=
by {
  use 39,
  split,
  { refl },
  { intros S hS,
    sorry
  }
}

end smallest_K_l206_206462


namespace more_calories_per_dollar_l206_206596

-- The conditions given in the problem as definitions
def price_burritos : ℕ := 6
def price_burgers : ℕ := 8
def calories_per_burrito : ℕ := 120
def calories_per_burger : ℕ := 400
def num_burritos : ℕ := 10
def num_burgers : ℕ := 5

-- The theorem stating the mathematically equivalent proof problem
theorem more_calories_per_dollar : 
  (num_burgers * calories_per_burger / price_burgers) - (num_burritos * calories_per_burrito / price_burritos) = 50 :=
by
  sorry

end more_calories_per_dollar_l206_206596


namespace pyramid_blocks_total_l206_206797

theorem pyramid_blocks_total :
  let total_blocks : ℕ :=
    (let layer1 := 2 in
    let layer2 := 4 * layer1 in
    let layer3 := 4 * layer2 in
    let layer4 := 4 * layer3 in
    let layer5 := 4 * layer4 in
    let layer6 := 4 * layer5 in
    layer1 + layer2 + layer3 + layer4 + layer5 + layer6)
  in total_blocks = 2730 :=
by
  -- Calculation and proof step, skipped for now
  sorry

end pyramid_blocks_total_l206_206797


namespace number_of_possible_values_of_b_l206_206672

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206672


namespace minimize_sum_at_centroid_minimize_product_at_centroid_l206_206203

open_locale real

variable (A B C M A1 B1 C1 : Type) [has_dist A] [has_dist B] [has_dist C]

-- Hypotheses
-- Point M is inside the triangle ABC and lines AM, BM, and CM intersect sides at A1, B1, and C1.
variable (hM_in_triangle : M ∈ interior (triangle A B C))
variable (hA1_intersection : lies_on_segment (line_through A M) (segment B C) A1)
variable (hB1_intersection : lies_on_segment (line_through B M) (segment C A) B1)
variable (hC1_intersection : lies_on_segment (line_through C M) (segment A B) C1)

-- Define ratio expressions
noncomputable def ratio_AM (A M A1 : Type) [has_dist A] [has_dist M] [has_dist A1] : ℝ :=
  (dist A M) / (dist M A1)

noncomputable def ratio_BM (B M B1 : Type) [has_dist B] [has_dist M] [has_dist B1] : ℝ :=
  (dist B M) / (dist M B1)

noncomputable def ratio_CM (C M C1 : Type) [has_dist C] [has_dist M] [has_dist C1] : ℝ :=
  (dist C M) / (dist M C1)

-- Prove that the sum achieves its minimum at the centroid
theorem minimize_sum_at_centroid :
  (∃ (G : Type) [has_dist G], G = centroid △ A B C ∧
  (ratio_AM A G A1 + ratio_BM B G B1 + ratio_CM C G C1) =
  min { ratio_AM A m A1 + ratio_BM B m B1 + ratio_CM C m C1 | M ∈ interior (triangle A B C) }) :=
sorry

-- Prove that the product achieves its minimum at the centroid
theorem minimize_product_at_centroid :
  (∃ (G : Type) [has_dist G], G = centroid △ A B C ∧
  (ratio_AM A G A1 * ratio_BM B G B1 * ratio_CM C G C1) =
  min { ratio_AM A m A1 * ratio_BM B m B1 * ratio_CM C m C1 | M ∈ interior (triangle A B C) }) :=
sorry

end minimize_sum_at_centroid_minimize_product_at_centroid_l206_206203


namespace area_of_given_triangle_l206_206088

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

def vertex_A : ℝ × ℝ := (-1, 4)
def vertex_B : ℝ × ℝ := (7, 0)
def vertex_C : ℝ × ℝ := (11, 5)

theorem area_of_given_triangle : area_of_triangle vertex_A vertex_B vertex_C = 28 :=
by
  show 1 / 2 * |(-1) * (0 - 5) + 7 * (5 - 4) + 11 * (4 - 0)| = 28
  sorry

end area_of_given_triangle_l206_206088


namespace area_of_subtriangle_l206_206754

theorem area_of_subtriangle (A B C D E F N1 N2 N3 : Point)
  (hBCD : midpoint D B C)
  (hACE : midpoint E A C)
  (hABF : midpoint F A B)
  (hN1_intrsec : intersection_point A D C F = N1)
  (hN2_intrsec : intersection_point B E A D = N2)
  (hN3_intrsec : intersection_point B E C F = N3) :
  area (triangle N1 N2 N3) = (1/8) * area (triangle A B C) :=
sorry

end area_of_subtriangle_l206_206754


namespace necessary_but_not_sufficient_for_inequality_l206_206100

variables (a b : ℝ)

theorem necessary_but_not_sufficient_for_inequality (h : a ≠ b) (hab_pos : a * b > 0) :
  (b/a + a/b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequality_l206_206100


namespace polynomial_relation_l206_206971

theorem polynomial_relation (x y : ℕ) :
  (x = 1 ∧ y = 1) ∨ 
  (x = 2 ∧ y = 4) ∨ 
  (x = 3 ∧ y = 9) ∨ 
  (x = 4 ∧ y = 16) ∨ 
  (x = 5 ∧ y = 25) → 
  y = x^2 := 
by
  sorry

end polynomial_relation_l206_206971


namespace part1_part2_part3_l206_206102

noncomputable def f (x : ℝ) : ℝ :=
  ([sin((π / 2) - x) * tan(π - x)]^2 - 1) / (4 * sin((3 * π / 2) + x) + cos(π - x) + cos((2 * π) - x))

theorem part1 : f (-1860 * π / 180) = sqrt 3 / 4 :=
sorry

theorem part2 (a : ℝ) : 
  ∀ x ∈ Icc (π / 6) (3 * π / 4), (f x)^2 + (1 + (1 / 2) * a) * sin x + 2 * a = 0 → (-1 / 2 < a ∧ a ≤ -sqrt 2 / 4) :=
sorry

theorem part3 (a : ℝ) : 
  ∃ z, ∀ x, -1 ≤ cos x ∧ cos x ≤ 1 →
  y = 4 * a * (f x)^2 + 2 * cos x → 
  (a = 0 → y = 2) ∧ (0 < a → (a < 1 → y = 2) ∧ (a ≥ 1 → y = a + 1 / a)) ∧ (a < 0 → y = 2) :=
sorry

end part1_part2_part3_l206_206102


namespace triangle_pqr_PQ_6_sqrt_6_l206_206199

theorem triangle_pqr_PQ_6_sqrt_6
  (P Q R : Type) [EuclideanGeometry] (h_1 : angle PQR = 90)
  (h_2 : QR = 15) (h_3 : tan R = 5 * cos Q) : PQ = 6 * sqrt 6 := 
  by
  sorry

end triangle_pqr_PQ_6_sqrt_6_l206_206199


namespace length_of_PB_equals_21_l206_206998

variables (A B C D P : Type*) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space D] [metric_space P] 
  (CD BC : ℝ)
  (AP PB : ℝ)

-- Conditions
def quadrilateral_ABCD (AB CD AD BD : ℝ) : Prop := 
  (CD = 48) ∧ 
  (BC = 36) ∧ 
  ((∃ θ φ : ℝ, 
    ∠ABD = θ ∧ 
    ∠BDA = φ ∧ 
    CD ⟂ AB ∧ 
    BC ⟂ AD ∧ 
    ∠ACP = φ ∧ 
    CP ⟂ BD ∧ 
    AP = 20 ∧ 
    PB ≈ 21))

theorem length_of_PB_equals_21 (A B C D P : Type*) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space P] 
  (AB CD AD BD : ℝ) (AP : ℝ) (h : quadrilateral_ABCD CD BC AB AD BD):
  PB = 21 :=
by
  sorry

end length_of_PB_equals_21_l206_206998


namespace rectangle_length_l206_206333

variable (L W : ℕ)

theorem rectangle_length (h1 : 6 * W = 5 * L) (h2 : W = 20) : L = 24 := by
  sorry

end rectangle_length_l206_206333


namespace numSelectionSchemes_l206_206791

-- Define the courses
def courses := ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

-- Define the main theorem
theorem numSelectionSchemes : 
  let courses_set := { "A", "B", "C", "D", "E", "F", "G", "H", "I" }.to_finset in
  let ABC := { "A", "B", "C" }.to_finset in
  let remaining := courses_set \ ABC in
  let choose (n m : ℕ) := nat.choose n m in
  let category1 := choose 3 1 * choose 6 3 in
  let category2 := choose 6 4 in
  category1 + category2 = 75 :=
by
  sorry

end numSelectionSchemes_l206_206791


namespace solution_set_inequality_l206_206067

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_deriv : ∀ x : ℝ, has_deriv_at f (f x) x
axiom f_plus_deriv_gt_one : ∀ x : ℝ, f(x) + deriv f(x) > 1
axiom f_at_zero : f 0 = 4

theorem solution_set_inequality : ∀ x : ℝ, x > 0 → e^x * f x > e^x + 3 := sorry

end solution_set_inequality_l206_206067


namespace angle_equality_or_complementarity_l206_206471

noncomputable def dihedral_angle (α : Type*) [linear_ordered_field α] : Prop :=
∃ (faces : set (set (α × α × α))), 
  ∃ (point : α × α × α), 
  ∀ face ∈ faces, 
  ∃ (line : set (α × α × α)), 
  line ⊥ face ∧ point ∈ line

theorem angle_equality_or_complementarity
  {α : Type*} [linear_ordered_field α]
  (point : α × α × α)
  (angle : dihedral_angle α)
  (line1 line2 : set (α × α × α))
  (perpendicular1 : ∀ face ∈ angle.faces, line1 ⊥ face ∧ point ∈ line1)
  (perpendicular2 : ∀ face ∈ angle.faces, line2 ⊥ face ∧ point ∈ line2)
  (plane_angle : α) :
  (∃ angle_result : α, (angle_result = plane_angle) ∨ (angle_result + plane_angle = π)) :=
sorry

end angle_equality_or_complementarity_l206_206471


namespace geometric_series_sum_l206_206747

theorem geometric_series_sum :
  ∃ (n : ℕ), (3 * 2^(n-1) = 3072) ∧
  (∑ i in finset.range n, 3 * 2^i = 6141) :=
by
  sorry

end geometric_series_sum_l206_206747


namespace no_division_660_stones_into_31_heaps_l206_206253

theorem no_division_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) :=
  sorry

end no_division_660_stones_into_31_heaps_l206_206253


namespace heap_division_impossible_l206_206265

theorem heap_division_impossible (n : ℕ) (k : ℕ) (similar : ℕ → ℕ → Prop)
  (h_similar : ∀ a b, similar a b ↔ a < 2 * b ∧ 2 * a > b) :
  n = 660 → k = 31 → ¬(∃ (heaps : Fin k → ℕ), ∑ i, heaps i = n ∧ ∀ i j, similar (heaps i) (heaps j)) :=
sorry

end heap_division_impossible_l206_206265


namespace possible_values_b_count_l206_206676

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206676


namespace locus_of_point_M_l206_206056

theorem locus_of_point_M 
  (O A : EuclideanSpace ℝ (Fin 2)) 
  (r m n : ℝ) 
  (h_r : r > 0) 
  (h_mn : m > 0 ∧ n > 0)
  (P : EuclideanSpace ℝ (Fin 2))
  (hP : dist O P = r) 
  (M : EuclideanSpace ℝ (Fin 2))
  (hM : ∃ k : ℝ, k = m / (m + n) ∧ M = k • P + (1 - k) • A)
  (hA_inside : dist O A < r): 
  ∃ O' : EuclideanSpace ℝ (Fin 2), 
    (∃ l : ℝ, l = n / (m + n) ∧ O' = l • A + (1 - l) • O) ∧
    (∃ R : ℝ, R = (n / (m + n)) * r ∧ ∀ P : EuclideanSpace ℝ (Fin 2), dist P O = r → dist M O' = R) := 
sorry

end locus_of_point_M_l206_206056


namespace constant_term_expansion_l206_206195

theorem constant_term_expansion : 
  let x := arbitrary_real 
  is_zero (\Summation_{n = 0}^{5} (x^(1 - n/2) * (-1)^n * \binom 5 n ) * x) = 10 := 
  by
  sorry

end constant_term_expansion_l206_206195


namespace number_of_divisors_of_24_divisible_by_4_l206_206655

theorem number_of_divisors_of_24_divisible_by_4 :
  (∃ b, 4 ∣ b ∧ b ∣ 24 ∧ 0 < b) → (finset.card (finset.filter (λ b, 4 ∣ b) (finset.filter (λ b, b ∣ 24) (finset.range 25))) = 4) :=
by
  sorry

end number_of_divisors_of_24_divisible_by_4_l206_206655


namespace min_distance_in_prism_l206_206487

theorem min_distance_in_prism :
  let A := (0, 0, 0) in
  let B := (6, 0, 0) in
  let C := (0, \sqrt{2}, 0) in
  let C₁ := (0, \sqrt{2}, \sqrt{2}) in
  let A₁ := (0, 0, \sqrt{2}) in
  let P (t : ℝ) := (0, t * \sqrt{2}, t * \sqrt{2}) in -- P is a parameterized point on BC₁
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → CP t + PA₁ t ≥ 5 * \sqrt{2}

:= sorry

end min_distance_in_prism_l206_206487


namespace lambda_range_l206_206559

noncomputable def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → a n > a (n-1)

theorem lambda_range (f : ℕ → ℝ) (λ : ℝ) : 
  (∀ n : ℕ, 1 ≤ n → f n = n^2 + λ * ↑n) →
  increasing_sequence f →
  λ > -3 :=
by
  sorry

end lambda_range_l206_206559


namespace lcm_of_1_to_5_l206_206771

theorem lcm_of_1_to_5 : ∃ n : ℕ, n > 0 ∧ (∀ k ∈ {1, 2, 3, 4, 5}, k ∣ n) ∧ n = 60 :=
by
  use 60
  split
  · exact Nat.pos_of_ne_zero (by norm_num) 
  split
  · intro k h
    fin_cases h
    norm_num
  · rfl

end lcm_of_1_to_5_l206_206771


namespace intersection_and_perpendicular_line_l206_206860

theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ), (3 * x + y - 1 = 0) ∧ (x + 2 * y - 7 = 0) ∧ (2 * x - y + 6 = 0) :=
by
  sorry

end intersection_and_perpendicular_line_l206_206860


namespace find_k_l206_206943

-- Define the vectors
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 1)
def c : ℝ × ℝ := (-5, 1)

-- Define the condition for parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define the statement to prove
theorem find_k : parallel (a.1 + k * b.1, a.2 + k * b.2) c → k = 1/2 :=
by
  sorry

end find_k_l206_206943


namespace Conor_can_chop_116_vegetables_in_a_week_l206_206060

-- Define the conditions
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8
def work_days_per_week : ℕ := 4

-- Define the total vegetables per day
def vegetables_per_day : ℕ := eggplants_per_day + carrots_per_day + potatoes_per_day

-- Define the total vegetables per week
def vegetables_per_week : ℕ := vegetables_per_day * work_days_per_week

-- The proof statement
theorem Conor_can_chop_116_vegetables_in_a_week : vegetables_per_week = 116 :=
by
  sorry  -- The proof step is omitted with sorry

end Conor_can_chop_116_vegetables_in_a_week_l206_206060


namespace repeat_decimal_to_fraction_l206_206079

theorem repeat_decimal_to_fraction : 0.36666 = 11 / 30 :=
by {
    sorry
}

end repeat_decimal_to_fraction_l206_206079


namespace hyperbola_equation_through_point_l206_206917

theorem hyperbola_equation_through_point
  (hyp_passes_through : ∀ (x y : ℝ), (x, y) = (1, 1) → ∃ (a b t : ℝ), (y^2 / a^2 - x^2 / b^2 = t))
  (asymptotes : ∀ (x y : ℝ), (y / x = Real.sqrt 2 ∨ y / x = -Real.sqrt 2) → ∃ (a b t : ℝ), (a = b * Real.sqrt 2)) :
  ∃ (a b t : ℝ), (2 * (1:ℝ)^2 - (1:ℝ)^2 = 1) :=
by
  sorry

end hyperbola_equation_through_point_l206_206917


namespace number_of_possible_values_of_b_l206_206666

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206666


namespace gcd_polynomial_l206_206500

theorem gcd_polynomial (b : ℤ) (h : 1820 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_polynomial_l206_206500


namespace covered_digits_l206_206393

def four_digit_int (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

theorem covered_digits (a b c : ℕ) (n1 n2 n3 : ℕ) :
  four_digit_int n1 → four_digit_int n2 → four_digit_int n3 →
  n1 + n2 + n3 = 10126 →
  (n1 % 10 = 3 ∧ n2 % 10 = 7 ∧ n3 % 10 = 6) →
  (n1 / 10 % 10 = 4 ∧ n2 / 10 % 10 = a ∧ n3 / 10 % 10 = 2) →
  (n1 / 100 % 10 = 2 ∧ n2 / 100 % 10 = 1 ∧ n3 / 100 % 10 = c) →
  (n1 / 1000 = 1 ∧ n2 / 1000 = 2 ∧ n3 / 1000 = b) →
  (a = 5 ∧ b = 6 ∧ c = 7) := 
sorry

end covered_digits_l206_206393


namespace problem_a1_value_l206_206975

theorem problem_a1_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (h : ∀ x : ℝ, x^10 = a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7 + a₈ * (x - 1)^8 + a₉ * (x - 1)^9 + a₁₀ * (x - 1)^10) :
  a₁ = 10 :=
sorry

end problem_a1_value_l206_206975


namespace range_of_a_l206_206490

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a*x^2 - 3*x + 2 = 0) ∧ 
  (∀ x y : ℝ, a*x^2 - 3*x + 2 = 0 ∧ a*y^2 - 3*y + 2 = 0 → x = y) 
  ↔ (a = 0 ∨ a = 9 / 8) := by sorry

end range_of_a_l206_206490


namespace samantha_num_trips_l206_206642

def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * π * r^3
def num_trips_to_fill (container_volume bucket_volume : ℝ) : ℤ := 
  if container_volume / bucket_volume % 1 = 0 then
    container_volume / bucket_volume
  else
    (container_volume / bucket_volume).ceil

theorem samantha_num_trips :
  let r_cylinder := 8
  let h_cylinder := 20
  let r_hemisphere := 5
  let V_cylinder := volume_cylinder r_cylinder h_cylinder
  let V_hemisphere := volume_hemisphere r_hemisphere
  num_trips_to_fill V_cylinder V_hemisphere = 16 :=
by
  sorry

end samantha_num_trips_l206_206642


namespace numbers_not_all_less_than_six_l206_206908

theorem numbers_not_all_less_than_six (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬ (a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) :=
sorry

end numbers_not_all_less_than_six_l206_206908


namespace total_fish_count_l206_206052

theorem total_fish_count (kyle_caught_same_as_tasha : ∀ kyle tasha : ℕ, kyle = tasha) 
  (carla_caught : ℕ) (kyle_caught : ℕ) (tasha_caught : ℕ)
  (h0 : carla_caught = 8) (h1 : kyle_caught = 14) (h2 : tasha_caught = kyle_caught) : 
  8 + 14 + 14 = 36 :=
by sorry

end total_fish_count_l206_206052


namespace min_value_of_expression_l206_206367

noncomputable def find_min_value : Real :=
  let expression (x y : ℝ) := (y * cos x - 2) ^ 2 + (y + sin x + 1) ^ 2
  let min_value := (λ x y, expression x y) π (-1.5)
  min_value

theorem min_value_of_expression : find_min_value = 1 / 2 := 
  by
    sorry

end min_value_of_expression_l206_206367


namespace true_proposition_l206_206903

def p (A B a b : ℝ) : Prop := 
  A > B ↔ a > b ∧ (a / Real.sin A) = (b / Real.sin B)

def q (a_n : ℕ → ℝ) : Prop := 
  ∀ (m : ℕ), m > 0 →
    let S := fun n => (n * (n + 1)) / 2
    in  (S m, S (2 * m), S (3 * m)) do not form an arithmetic sequence

theorem true_proposition (A B a b : ℝ) (a_n : ℕ → ℝ) (m : ℕ) (h1 : p A B a b) (h2 : q a_n → False) :
  p A B a b ∨ ¬ q a_n :=
by 
  sorry

end true_proposition_l206_206903


namespace max_value_tan_sec_l206_206861

theorem max_value_tan_sec : ∀ x : ℝ, ∃ y : ℝ, y ≤ tan x ∧ y ≤ sec x → ∀ n : ℕ, ∃ M : ℝ, tan^6 x + sec^4 x = M ∧ M = ∞ := by sorry

end max_value_tan_sec_l206_206861


namespace smallest_number_of_marbles_l206_206040

theorem smallest_number_of_marbles 
  (r w b bl n : ℕ) 
  (h : r + w + b + bl = n)
  (h1 : r * (r - 1) * (r - 2) * (r - 3) = 24 * w * b * (r * (r - 1) / 2))
  (h2 : r * (r - 1) * (r - 2) * (r - 3) = 24 * bl * b * (r * (r - 1) / 2))
  (h_no_neg : 4 ≤ r):
  n = 18 :=
sorry

end smallest_number_of_marbles_l206_206040


namespace breadth_of_rectangular_prism_l206_206403

theorem breadth_of_rectangular_prism (b l h : ℝ) (h_l : l = 3 * b) (h_h : h = 2 * b) (h_v : l * b * h = 3294) : 
  b ≈ 8.2 :=
by
  sorry

end breadth_of_rectangular_prism_l206_206403


namespace heap_division_impossible_l206_206246

def similar_sizes (a b : Nat) : Prop := a < 2 * b

theorem heap_division_impossible (n k : Nat) (h1 : n = 660) (h2 : k = 31) 
    (h3 : ∀ p1 p2 : Nat, p1 ∈ (Finset.range k).to_set → p2 ∈ (Finset.range k).to_set → similar_sizes p1 p2 → p1 < 2 * p2) : 
    False := 
by
  sorry

end heap_division_impossible_l206_206246


namespace possible_values_b_count_l206_206674

theorem possible_values_b_count :
    {b : ℤ | b > 0 ∧ 4 ∣ b ∧ b ∣ 24}.to_finset.card = 4 :=
sorry

end possible_values_b_count_l206_206674


namespace find_k_value_l206_206577

structure Quadrilateral :=
(A B C D : Point)

structure Midpoint (P Q R : Point) :=
(belongs_to_diagonal : Line P Q | Line Q R)
(mid : ∃ E : Point, E = ((P + Q) / 2) <|> E = ((Q + R) / 2))

noncomputable def calculate_k (quad : Quadrilateral) (M N E : Point) (area_ABC : ℝ) (area_definitions : Bool) : ℝ :=
if area_definitions then sorry else
  let total_area := 4 * area_ABC in
  let tri_MNE_area := total_area / 8 in
  tri_MNE_area / total_area

theorem find_k_value (quad : Quadrilateral) (M N E : Point) (area_ABC : ℝ) 
    (hE : Midpoint quad.A quad.C E)
    (hN : Midpoint quad.B quad.D N) 
    (hM_int : ∃ P: Point, P = Midpoint_between_intersection quad.A quad.C quad.B quad.D M) :
    calculate_k quad M N E area_ABC true = 1 / 8 :=
sorry

end find_k_value_l206_206577


namespace numberOfTruePropositions_l206_206929

def coinTossEventA (toss1 toss2 : Bool) : Bool := toss1 = true ∧ toss2 = true 
def coinTossEventB (toss1 toss2 : Bool) : Bool := toss1 = false ∧ toss2 = false 

def coinEventsComplementary : Prop := ¬ (∀ toss1 toss2, coinTossEventA toss1 toss2 ∨ coinTossEventB toss1 toss2)
def coinEventsMutuallyExclusive : Prop := ∀ toss1 toss2, ¬ (coinTossEventA toss1 toss2 ∧ coinTossEventB toss1 toss2)

def productEventA (selected : Finset Nat) : Prop := selected.card <= 2 
def productEventB (selected : Finset Nat) : Prop := selected.card >= 2 

def productEventsMutuallyExclusive (selected : Finset Nat) : Prop := ¬ (productEventA selected ∧ productEventB selected)

theorem numberOfTruePropositions : ((coinEventsComplementary = False) 
                                ∧ (coinEventsMutuallyExclusive = True) 
                                ∧ (productEventsMutuallyExclusive = False)) → 
                                1 = 1 := 
by
  intro h
  cases h
  exact rfl

end numberOfTruePropositions_l206_206929


namespace f_monotonic_m_range_l206_206283

-- Definitions and conditions for the function f
def f (m x : ℝ) : ℝ := Real.exp(m*x) + x^2 - m*x

-- Proof 1: Monotonicity
theorem f_monotonic (m : ℝ) :
  (∀ x : ℝ, x < 0 → (f m x) > (f m (x+ε))) ∧
  (∀ x : ℝ, x > 0 → (f m x) < (f m (x-ε))) :=
sorry

-- Proof 2: Range of m given the condition
theorem m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Icc (-1 : ℝ) (1 : ℝ) → x₂ ∈ Icc (-1 : ℝ) (1 : ℝ) → |f m x₁ - f m x₂| ≤ Real.exp(1) - 1) →
  m ∈ Icc (-1) (1) :=
sorry

end f_monotonic_m_range_l206_206283


namespace max_area_of_sector_l206_206323

theorem max_area_of_sector (r l : ℝ) (h : 2 * r + l = 4) : 
  (1/2) * l * r ≤ 1 :=
begin
  sorry
end

end max_area_of_sector_l206_206323


namespace S_A1_card_eq_5_S_A2_card_eq_2n_sub_3_l206_206879

-- Condition 1
def A1 := {1, 2, 3, 4}

-- Question 1: Prove S(A1) = 5
theorem S_A1_card_eq_5 : 
  let S := {x | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ x = A1.toNatSet i + A1.toNatSet j} in
  S.card = 5 :=
by sorry

-- Condition 2
variables {n : ℕ} (A2 : Finset ℕ) (d : ℕ)
  (h1 : 0 < d)
  (h2 : ∀ i j, i < j → A2.toNatSet j = A2.toNatSet i + d)
  (h3 : A2.card = n)
  (h4 : n ≥ 3)

-- Question 2: Prove S(A2) = 2n - 3
theorem S_A2_card_eq_2n_sub_3 : 
  let S := {x | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ A2.card ∧ x = A2.toNatSet i + A2.toNatSet j} in
  S.card = 2 * n - 3 :=
by sorry

end S_A1_card_eq_5_S_A2_card_eq_2n_sub_3_l206_206879


namespace minimum_sum_distance_l206_206124

noncomputable def minimum_distance (P A : ℝ × ℝ) : ℝ :=
  let focus : ℝ × ℝ := (1 / 2, 0)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  in distance P focus + distance P A

def parabola (x y : ℝ) : Prop :=
  y^2 = 2 * x

def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

def directrix_distance (P : ℝ × ℝ) : ℝ :=
  P.1 - 1 / 2

-- The central statement we need to prove
theorem minimum_sum_distance {P : ℝ × ℝ} (hP : point_on_parabola P) :
  minimum_distance P (0, 2) = sqrt (17) / 2 :=
sorry

end minimum_sum_distance_l206_206124


namespace part_I_part_II_1_part_II_2_part_III_l206_206896

-- Define the sequence {a_n}
def a_n1 (n : ℕ) : ℕ := 2 * n - 1
def a_n2 (n : ℕ) : ℕ := 10^(n-1)

-- Define the set A_k
def A_k (k : ℕ) (a : ℕ → ℕ) : set ℤ :=
  {x | ∃ (λ : ℕ → ℤ), x = ∑ i in finset.range (k+1), λ i * a (i + 1) ∧ ∀ i, λ i ∈ {-1, 0, 1}}

-- Define properties of the sequence
def complete_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ x ∈ A_k k a, ∃! λ : ℕ → ℤ, x = ∑ i in finset.range (k+1), λ i * a (i + 1)

def full_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  let m_k := ∑ i in finset.range (k+1), a (i + 1)
  in ∀ x : ℤ, |x| ≤ m_k → x ∈ A_k k a

def perfect_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  complete_sequence a k ∧ full_sequence a k

-- Part I
theorem part_I : A_k 2 a_n1 = {x | x ∈ [-4,-3,-2,-1,0,1,2,3,4]} ∧
  complete_sequence a_n1 2 ∧ full_sequence a_n1 2 ∧ perfect_sequence a_n1 2 :=
sorry

-- Part II
theorem part_II_1 : complete_sequence a_n2 n := sorry

theorem part_II_2 : ∑ x in (A_k n a_n2).to_finset, x = 0 := sorry

-- Part III
theorem part_III : ∃ a : ℕ → ℕ, perfect_sequence a n ∧
  A_k n a = {x | x ∈ [-(3^n - 1)/2, -(3^n - 3)/2, ... , 0, ... , (3^n - 3)/2, (3^n - 1)/2]} ∧
  a n = 3^(n-1) :=
sorry

end part_I_part_II_1_part_II_2_part_III_l206_206896


namespace triangular_pyramid_volume_l206_206029

theorem triangular_pyramid_volume (a b c : ℝ)
  (h1 : 1/2 * a * b = 1.5)
  (h2 : 1/2 * b * c = 2)
  (h3 : 1/2 * a * c = 6) :
  (1/6 * a * b * c = 2) :=
by {
  -- Here, we would provide the proof steps, but for now we leave it as sorry
  sorry
}

end triangular_pyramid_volume_l206_206029


namespace berengere_contribution_l206_206428

noncomputable def usd_to_euro (usd : ℝ) : ℝ := usd / 1.10

def pastry_cost_euro : ℝ := 8

def liam_money_usd : ℝ := 10

theorem berengere_contribution : 
  let liam_money_euro := usd_to_euro liam_money_usd in
  liam_money_euro >= pastry_cost_euro -> 
  ∃ (berengere_contribution : ℝ), berengere_contribution = 0 := 
by
  sorry

end berengere_contribution_l206_206428


namespace sequences_conditions_l206_206086

theorem sequences_conditions (b_n c_n : ℕ → ℝ) :
  (∀ n : ℕ, 0 < n → b_n ≤ c_n) →
  (∀ n : ℕ, 0 < n → (∃ b_n1 c_n1 : ℝ, b_n1 + c_n1 = -b_n ∧ b_n1 * c_n1 = c_n 
    ∧ b_n1 = b_{n+1} ∧ c_n1 = c_{n+1})) →
  (∀ n : ℕ, b_n = 0 ∧ c_n = 0) :=
by
  intros h1 h2
  exists [b_n, c_n]
  apply sorry

end sequences_conditions_l206_206086


namespace domain_g_l206_206127

-- Define the function f and its domain
variable {f : ℝ → ℝ}
axiom domain_f : ∀ x, 0 ≤ x → x ≤ 8

-- Define the function g
def g (x : ℝ) := f (2 * x) / (3 - x)

-- Prove the domain of g
theorem domain_g : ∀ x, (0 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4) :=
by {
  intros x,
  split,
  {
    assume h,
    cases h,
    have : 0 ≤ 2 * x ∧ 2 * x ≤ 8, sorry,
    have : 3 - x ≠ 0, sorry,
    sorry
  },
  {
    assume h,
    cases h,
    have : 0 ≤ 2 * x ∧ 2 * x ≤ 8, sorry,
    have : 3 - x ≠ 0, sorry,
    sorry
  }
}

end domain_g_l206_206127


namespace juan_amal_probability_l206_206209

theorem juan_amal_probability :
  let prob := ∑ i in finset.range 1 11, ∑ j in finset.range 1 7, 
             if (i * j) % 4 = 0 then (1 / 10) * (1 / 6) else 0
  in prob = 1 / 3 :=
by {
  let prob := ∑ i in finset.range 1 11, ∑ j in finset.range 1 7, 
             if (i * j) % 4 = 0 then (1 / 10) * (1 / 6) else 0,
  exact eq.trans (prob) 1 / 3 sorry
}

end juan_amal_probability_l206_206209


namespace new_length_of_rope_l206_206021

noncomputable def new_rope_length (r1 : ℝ) (A_add : ℝ) : ℝ :=
  real.sqrt (r1^2 + A_add / real.pi)

theorem new_length_of_rope : new_rope_length 12 1210 ≈ 23 :=
by
  sorry

end new_length_of_rope_l206_206021


namespace trailing_zeros_28_factorial_l206_206548

theorem trailing_zeros_28_factorial : ∀ n : ℕ, 
  n = 28 → (nat.factorial n).trailing_zeros = 6 :=
by
  intros n hn
  rw hn
  simp [factorial]
  sorry

end trailing_zeros_28_factorial_l206_206548


namespace number_of_pairs_l206_206161

theorem number_of_pairs (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (m^2 + 2 * n < 30) ↔ (|{p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + 2 * p.2 < 30}| = 42) :=
sorry

end number_of_pairs_l206_206161


namespace stacy_days_to_complete_paper_l206_206317

variable (total_pages pages_per_day : ℕ)
variable (d : ℕ)

theorem stacy_days_to_complete_paper 
  (h1 : total_pages = 63) 
  (h2 : pages_per_day = 21) 
  (h3 : total_pages = pages_per_day * d) : 
  d = 3 := 
sorry

end stacy_days_to_complete_paper_l206_206317


namespace number_of_possible_values_l206_206661

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206661


namespace trig_identity_solution_l206_206105

theorem trig_identity_solution
  (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trig_identity_solution_l206_206105


namespace fraction_to_decimal_17_625_l206_206465

def fraction_to_decimal (num : ℕ) (den : ℕ) : ℚ := num / den

theorem fraction_to_decimal_17_625 : fraction_to_decimal 17 625 = 272 / 10000 := by
  sorry

end fraction_to_decimal_17_625_l206_206465


namespace range_of_a_l206_206144

theorem range_of_a {a : ℝ} 
  (hA : ∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))
  (hB : ∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))
  (hC : (a^2 + 1) / (2 * a) > 0)
  (hOnlyOneFalse : (¬(∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬(∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬((a^2 + 1) / (2 * a) > 0))):
  0 < a ∧ a < 1 := 
sorry

end range_of_a_l206_206144


namespace number_of_possible_values_of_b_l206_206671

theorem number_of_possible_values_of_b
  (b : ℕ) 
  (h₁ : 4 ∣ b) 
  (h₂ : b ∣ 24)
  (h₃ : 0 < b) :
  { n : ℕ | 4 ∣ n ∧ n ∣ 24 ∧ 0 < n }.card = 4 :=
sorry

end number_of_possible_values_of_b_l206_206671


namespace car_speed_l206_206774

theorem car_speed (distance_meters : ℝ) (time_seconds : ℝ) (conversion_factor : ℝ) (speed_kmph : ℝ) : 
  distance_meters = 375 ∧ time_seconds = 15 ∧ conversion_factor = 3.6 ∧ 
  speed_kmph = (distance_meters / time_seconds) * conversion_factor → speed_kmph = 90 :=
begin
  intros h,
  cases h with d h1,
  cases h1 with t h2,
  cases h2 with cf h3,
  cases h3 with skmph_eq,
  rw [d, t, cf] at skmph_eq,
  norm_num at skmph_eq,
  rw skmph_eq,
  refl,
end

end car_speed_l206_206774


namespace max_min_sine_function_find_a_l206_206381

-- Problem 1: Maximum and Minimum of sine-based function
theorem max_min_sine_function :
  let y := λ x => 1 - 2 * sin (x + π / 6) in
  (∀ x, y x ≤ 3) ∧ (∃ x, y x = 3) ∧ 
  (∀ x, -1 ≤ y x) ∧ (∃ x, y x = -1) :=
by
  let y := λ x => 1 - 2 * sin (x + π / 6)
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

-- Problem 2: Find the value of a
theorem find_a (a : ℝ) :
  (∀ x ∈ Icc 0 (π / 2), cos (2 * x + π / 3) ∈ Icc (-1 : ℝ) (1 / 2 : ℝ)) →
  has_max_value (λ x : ℝ, a * cos (2 * x + π / 3) + 3) 4 →
  a = 2 ∨ a = -1 :=
by
  sorry

end max_min_sine_function_find_a_l206_206381


namespace distinct_3x3x3_cube_configurations_l206_206396

noncomputable def number_of_distinct_3x3x3_cube_configurations : ℕ :=
  89754

theorem distinct_3x3x3_cube_configurations :
  (∃ G : Type, (order_G : G) -> order_G = 26) ->
  (white_cubes : ℕ) -> (blue_cubes : ℕ) ->
  white_cubes = 13 -> blue_cubes = 14 ->
  ∃ n, n = number_of_distinct_3x3x3_cube_configurations := by
  sorry

end distinct_3x3x3_cube_configurations_l206_206396


namespace smallest_positive_period_f_max_min_values_f_shifted_l206_206933

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * sin (x + π / 8) * (sin (x + π / 8) - cos (x + π / 8))

theorem smallest_positive_period_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π := sorry

noncomputable def f_shifted (x : ℝ) : ℝ := f (x + π / 8)

theorem max_min_values_f_shifted :
  (∀ x ∈ Icc (-π / 2) 0, f_shifted x ≤ √2) ∧
  (∀ x ∈ Icc (-π / 2) 0, f_shifted x ≥ -1) := sorry

end smallest_positive_period_f_max_min_values_f_shifted_l206_206933


namespace michaels_estimate_l206_206731

-- Define the variables as positive real numbers.
variables {x y z : ℝ}

-- Michael's estimation is greater than 3 times the difference of the original numbers.
theorem michaels_estimate (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) :
  3 * ((x + z) - (y - 2z)) > 3 * (x - y) :=
by
  sorry

end michaels_estimate_l206_206731


namespace find_k_l206_206353

-- Define a point and its translation
structure Point where
  x : ℕ
  y : ℕ

-- Original and translated points
def P : Point := { x := 5, y := 3 }
def P' : Point := { x := P.x - 4, y := P.y - 1 }

-- Given function with parameter k
def line (k : ℕ) (p : Point) : ℕ := (k * p.x) - 2

-- Prove the value of k
theorem find_k (k : ℕ) (h : line k P' = P'.y) : k = 4 :=
by
  sorry

end find_k_l206_206353


namespace shortest_side_in_triangle_ABC_l206_206176

theorem shortest_side_in_triangle_ABC :
  (∀ A B C : ℝ, B = 45 ∧ C = 60 ∧ ∃ c = 1, (angle_sum : A + B + C = 180) → 
  (let sin (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180),
       a := sin A * c / sin C,
       b := sin B * c / sin C,
       lengths := [a, b, c]) in
   ∃ (shortest : ℝ), shortest = b ∧ shortest = sqrt 6 / 3))
:= by
  intros A B C h c hc angle_sum
  let sin (x : ℝ) := Real.sin (x * Real.pi / 180)
  let c := 1
  let a := sin A * c / sin C
  let b := sin B * c / sin C
  let lengths := [a, b, c]
  exists b
  sorry

end shortest_side_in_triangle_ABC_l206_206176


namespace heap_division_impossible_l206_206266

theorem heap_division_impossible (n : ℕ) (k : ℕ) (similar : ℕ → ℕ → Prop)
  (h_similar : ∀ a b, similar a b ↔ a < 2 * b ∧ 2 * a > b) :
  n = 660 → k = 31 → ¬(∃ (heaps : Fin k → ℕ), ∑ i, heaps i = n ∧ ∀ i j, similar (heaps i) (heaps j)) :=
sorry

end heap_division_impossible_l206_206266


namespace area_ratios_l206_206829

noncomputable def meet_ratios : Prop :=
  let S (A B C : Point) := area A B C in
  ∃ (A B C M A1 B1 : Point),
    collinear B C A1 ∧ collinear A C B1 ∧
    line_through A A1 ∩ line_through B B1 = {M} ∧
    segment_ratio B A1 A1 C = 1 / 3 ∧
    segment_ratio A B1 B1 C = 1 / 2 ∧
    S A B M / S B C M = 1 / 2 ∧
    S A B M / S A C M = 1 / 3

theorem area_ratios : meet_ratios :=
sorry

end area_ratios_l206_206829


namespace sample_size_ratio_l206_206782

theorem sample_size_ratio (units_A model_A : ℕ) (ratio_A_B_C : ℕ × ℕ × ℕ) (sample_units : units_A) :
  ratio_A_B_C = (3, 4, 7) ∧ model_A = 15 → units_A = 70 :=
by
  intro h
  cases h with ratio_eq model_eq
  cases ratio_eq
  rw model_eq
  sorry

end sample_size_ratio_l206_206782


namespace dense_sets_count_l206_206793

open Finset

def is_dense (A : Finset ℕ) : Prop :=
  (A ⊆ range 50) ∧ (card A > 40) ∧
  (∀ x ∈ range 45, range' x 6 \subseteq A → false)

theorem dense_sets_count :
  {A : Finset ℕ | is_dense A}.card = 495 :=
sorry

end dense_sets_count_l206_206793


namespace total_pages_written_l206_206631

-- Define the conditions
def timeMon : ℕ := 60  -- Minutes on Monday
def rateMon : ℕ := 30  -- Minutes per page on Monday

def timeTue : ℕ := 45  -- Minutes on Tuesday
def rateTue : ℕ := 15  -- Minutes per page on Tuesday

def pagesWed : ℕ := 5  -- Pages written on Wednesday

-- Function to compute pages written based on time and rate
def pages_written (time rate : ℕ) : ℕ := time / rate

-- Define the theorem to be proved
theorem total_pages_written :
  pages_written timeMon rateMon + pages_written timeTue rateTue + pagesWed = 10 :=
sorry

end total_pages_written_l206_206631


namespace rhombus_diagonals_perpendicular_l206_206202

section circumscribed_quadrilateral

variables {a b c d : ℝ}

-- Definition of a tangential quadrilateral satisfying Pitot's theorem.
def tangential_quadrilateral (a b c d : ℝ) :=
  a + c = b + d

-- Defining a rhombus in terms of its sides
def rhombus (a b c d : ℝ) :=
  a = b ∧ b = c ∧ c = d

-- The theorem we want to prove
theorem rhombus_diagonals_perpendicular
  (h : tangential_quadrilateral a b c d)
  (hr : rhombus a b c d) : 
  true := sorry

end circumscribed_quadrilateral

end rhombus_diagonals_perpendicular_l206_206202


namespace geometric_seq_min_value_l206_206226

theorem geometric_seq_min_value (b : ℕ → ℝ) (s : ℝ) (h1 : b 1 = 1) (h2 : ∀ n : ℕ, b (n + 1) = s * b n) : 
  ∃ s : ℝ, 3 * b 1 + 4 * b 2 = -9 / 16 :=
by
  sorry

end geometric_seq_min_value_l206_206226


namespace product_common_divisors_180_36_l206_206092

theorem product_common_divisors_180_36 : 
  (∏ i in (finset.filter (λ x, 36 ∣ x) (finset.attach (finset.univ : finset (integer.divisors 180)))), i) = 142884480 := 
by
  sorry

end product_common_divisors_180_36_l206_206092


namespace find_a_l206_206922

   theorem find_a (e : ℝ) (f : ℝ → ℝ) (a : ℝ) (x₀ : ℝ) (h₀ : e = 2.71828) (h₁ : f x = (a * x) / e ^ x) (h₂ : ∀ x, f' x = (a * (1 - x)) / (e ^ x)) (hx₀ : x₀ = 1) (h₃ : f x₀ = 1 / e) :
   a = 1 := sorry
   
end find_a_l206_206922


namespace cosine_theorem_oblique_prism_space_l206_206431

variables {R : Type*} [CommRing R]
variables {A_1 C_1 C B B_1 A_ B_ : R}
variables {S_ΔA1C1C S_ΔBB1A1 S_qBCC1B1 : R}
variables {θ : R}

theorem cosine_theorem_oblique_prism_space :
  S_ΔA1C1C^2 = S_ΔBB1A1^2 + S_qBCC1B1^2 - 2 * S_ΔBB1A1 * S_qBCC1B1 * (cos θ) := 
sorry

end cosine_theorem_oblique_prism_space_l206_206431


namespace ant_back_to_A_after_4_steps_l206_206895

-- Definitions of vertices and rules for the ant's movement.
inductive Vertex
| A | B | C | D

open Vertex

def edges : Vertex → list Vertex
| A := [B, C, D]
| B := [A, C, D]
| C := [A, B, D]
| D := [A, B, C]

def step_probability : Vertex → Vertex → ℚ
| v u := if u ∈ edges v then 1 / 3 else 0

def crawl_probability : ℕ → Vertex → Vertex → ℚ
| 0, v, u := if v = u then 1 else 0
| (n + 1), v, u := ∑ w in edges v, (step_probability v w) * (crawl_probability n w u)

-- Probability calculation for exactly 4 steps back to A
theorem ant_back_to_A_after_4_steps : crawl_probability 4 A A = 7 / 27 := 
by
  sorry

end ant_back_to_A_after_4_steps_l206_206895


namespace asymptote_problem_l206_206584

def polynomial_fraction : RatFunc :=
  { num := Polynomial.mk [6, 5, 1], den := Polynomial.mk [-6, 1, 1, 1] }

def p := 1  -- number of holes
def q := 2  -- number of vertical asymptotes
def r := 1  -- number of horizontal asymptotes
def s := 0  -- number of oblique asymptotes

theorem asymptote_problem :
  p + 2 * q + 3 * r + 4 * s = 8 :=
by
  sorry

end asymptote_problem_l206_206584


namespace find_primes_2004_l206_206854

theorem find_primes_2004 (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) (h1 : p < q) (h2 : q < r) 
  (h3 : 25 * p * q + r = 2004) (h4 : ∃ k : ℕ, p * q * r + 1 = k * k) : p = 7 ∧ q = 11 ∧ r = 79 :=
by
  sorry

end find_primes_2004_l206_206854


namespace number_of_possible_values_l206_206664

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206664


namespace money_distribution_l206_206413

variable (A B C : ℕ)

theorem money_distribution :
  A + B + C = 500 →
  B + C = 360 →
  C = 60 →
  A + C = 200 :=
by
  intros h1 h2 h3
  sorry

end money_distribution_l206_206413


namespace inequality_integer_solutions_l206_206557

theorem inequality_integer_solutions (a : ℝ) :
  (∃ x : ℤ, 4 < x ∧ x ≤ a) → 
  (∀ n : ℤ, 4 < n → n ≤ a → (n = 5 ∨ n = 6 ∨ n = 7)) → 
  7 ≤ a ∧ a < 8 :=
begin
  sorry
end

end inequality_integer_solutions_l206_206557


namespace pure_imaginary_b_eq_two_l206_206910

theorem pure_imaginary_b_eq_two (b : ℝ) : (∃ (im_part : ℝ), (1 + b * Complex.I) / (2 - Complex.I) = im_part * Complex.I) ↔ b = 2 :=
by
  sorry

end pure_imaginary_b_eq_two_l206_206910


namespace cupcakes_frosted_in_10_minutes_l206_206433

-- Conditions as Lean definitions
def cagney_rate : ℝ := 1 / 25
def lacey_rate : ℝ := 1 / 35
def casey_rate : ℝ := 1 / 45
def time_in_seconds : ℕ := 600

-- The proof statement
theorem cupcakes_frosted_in_10_minutes : 
  (cagney_rate + lacey_rate + casey_rate) * time_in_seconds = 54 :=
by
  sorry

end cupcakes_frosted_in_10_minutes_l206_206433


namespace repeating_decimal_fraction_l206_206081

theorem repeating_decimal_fraction : (0.366666... : ℝ) = 33 / 90 := 
sorry

end repeating_decimal_fraction_l206_206081


namespace complement_intersection_eq_l206_206148

open Set

def P : Set ℝ := { x | x^2 - 2 * x ≥ 0 }
def Q : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

theorem complement_intersection_eq :
  (compl P) ∩ Q = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end complement_intersection_eq_l206_206148


namespace sum_abc_is_6sqrt3_l206_206560

noncomputable def sum_of_abc (a b c : ℝ) (h₁ : a^2 + b^2 + c^2 = 52) (h₂ : a * b + b * c + c * a = 28) : ℝ :=
a + b + c

theorem sum_abc_is_6sqrt3 (a b c : ℝ) (h₁ : a^2 + b^2 + c^2 = 52) (h₂ : a * b + b * c + c * a = 28) :
  sum_of_abc a b c h₁ h₂ = 6 * real.sqrt 3 :=
sorry

end sum_abc_is_6sqrt3_l206_206560


namespace binary_addition_is_correct_l206_206416

theorem binary_addition_is_correct :
  (0b101101 + 0b1011 + 0b11001 + 0b1110101 + 0b1111) = 0b10010001 :=
by sorry

end binary_addition_is_correct_l206_206416


namespace range_of_a_l206_206282

noncomputable def f (x a : ℝ) := 3 ^ abs (x - 1) - 2 * x + a
def g (x : ℝ) := 2 - x ^ 2
def h (x : ℝ) := 2 - x ^ 2 + 2 * x - 3 ^ abs (x - 1)

theorem range_of_a : ∀ (a : ℝ), 
  (∀ (x : ℝ), 0 < x ∧ x < 3 → f x a > g x) ↔ 2 < a :=
by
  intros
  sorry

end range_of_a_l206_206282


namespace find_alpha_l206_206585

/-- Given parametric equations of curve C1 and polar equation of curve C2.
     Also, curves intersect at points A and B on curve C3, such that |AB| = 4√2,
     prove that α = 3π / 4. --/

open Real

noncomputable def parametric_to_cartesian_equation_of_C1 (x y φ : ℝ) : Prop :=
  x = 2 + 2 * cos φ ∧ y = 2 * sin φ →
  (x - 2)^2 + y^2 = 4

noncomputable def polar_to_cartesian_equation_of_C2 (x y θ : ℝ) : Prop :=
  ρ = 4 * sin θ → x^2 + (y - 2)^2 = 4

theorem find_alpha (φ θ α ρ1 ρ2 : ℝ)
  (hC1 : parametric_to_cartesian_equation_of_C1 x y φ)
  (hC2 : polar_to_cartesian_equation_of_C2 x y θ)
  (hC3 : θ = α ∧ 0 < α ∧ α < π)
  (hA_B_distinct : x ≠ 0 ∨ y ≠ 0)
  (h_AB_distance : abs (4 * (sin α - cos α)) = 4 * sqrt 2) :
  α = 3 * π / 4 := by
  sorry

end find_alpha_l206_206585


namespace num_of_possible_values_b_l206_206688

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206688


namespace max_plus_min_eq_four_l206_206136

def f (x : ℝ) : ℝ := (2 ^ (|x| + 1) + x^3 + 2) / (2 ^ |x| + 1)

def g (x : ℝ) : ℝ := x^3 / (2 ^ |x| + 1)

lemma g_odd (x : ℝ) : g (-x) = -g x := sorry

lemma g_max_min_sum_zero : (sup (g '' set.univ)) + (inf (g '' set.univ)) = 0 := sorry

theorem max_plus_min_eq_four : 
  (sup (f '' set.univ)) + (inf (f '' set.univ)) = 4 := sorry

end max_plus_min_eq_four_l206_206136


namespace only_statement_1_true_l206_206063

theorem only_statement_1_true 
  (a b x y : ℝ) :
  (2 * (a + b) = 2 * a + 2 * b) ∧ ¬(5^(a + b) = 5^a + 5^b) ∧ ¬(log (x + y) = log x + log y) ∧ ¬(sqrt (a^2 + b^2) = a + b) ∧ ¬((x + y)^2 = x^2 + y^2) :=
by
  sorry

end only_statement_1_true_l206_206063


namespace greatest_prime_divisor_sum_of_digits_l206_206370

/-- The sum of the digits of the greatest prime number that is a divisor of 16385 is 13. -/
theorem greatest_prime_divisor_sum_of_digits :
  let n := 16385
  let p := Nat.greatest_prime_divisor n
  p = 1093 ∧ Nat.sum_of_digits p = 13 := by
  sorry

end greatest_prime_divisor_sum_of_digits_l206_206370


namespace victor_decks_l206_206740

theorem victor_decks (V : ℕ) (cost_per_deck total_cost : ℕ) 
  (friend_decks friend_cost : ℕ)
  (cost_per_deck = 8)
  (friend_decks = 2)
  (total_cost = 64)
  (friend_cost = friend_decks * cost_per_deck)
  (total_cost = 8 * V + friend_cost) :
  V = 6 :=
by
  sorry

end victor_decks_l206_206740


namespace sampling_method_systematic_l206_206566

/-- 
Given that there are 35 classes, each with 56 students numbered from 1 to 56, 
and the student numbered 14 is required to stay for the exchange, 
prove that the sampling method used is 'Systematic Sampling'.
--/

theorem sampling_method_systematic :
  ∃ (n_classes n_students : ℕ) (chosen_student : ℕ),
    n_classes = 35 ∧
    n_students = 56 ∧
    chosen_student = 14 ∧
    sampling_method = "Systematic Sampling" 
sorry

end sampling_method_systematic_l206_206566


namespace find_trajectory_eq_slopes_sum_eq_zero_l206_206923

-- 1. Proving the trajectory equation of curve C
theorem find_trajectory_eq (M F : ℝ × ℝ) (x_axis_fixed : ℝ → bool) (h1 : F = (1, 0)) (h2 : x_axis_fixed 4 = true)
                           (h_ratio : ∀ M, (real.sqrt ((M.1 - F.1)^2 + M.2^2) / (|M.1 - 4|)) = (1/2)) :
  ∀ M, (\frac{M.1^2}{4} + \frac{M.2^2}{3} = 1) :=
sorry

-- 2. Proving the sum of slopes k_1 and k_2 is 0
theorem slopes_sum_eq_zero (F P : ℝ × ℝ) (curve_eq : ℝ × ℝ → bool) (h1 : F = (1, 0)) (h2 : P = (4, 0))
                            (h_curve_eq : ∀ M, curve_eq(M) = ((M.1^2)/4 + (M.2^2)/3 = 1))
                            (A B : ℝ × ℝ) (l : line) (h_non_zero_slope : l.slope ≠ 0) (intersects_curve : l.pass_through(F) ∧ curve_eq(A)
                            ∧ curve_eq(B))
                            (k1 k2 : ℝ) (slope_PA : k1 = (P.2 - A.2) / (P.1 - A.1)) (slope_PB : k2 = (P.2 - B.2) / (P.1 - B.1)) :
  k1 + k2 = 0 :=
sorry

end find_trajectory_eq_slopes_sum_eq_zero_l206_206923


namespace area_of_trapezoid_l206_206196

-- Definitions of geometric properties and conditions
def is_perpendicular (a b c : ℝ) : Prop := a + b = 90 -- representing ∠ABC = 90°
def tangent_length (bc ad : ℝ) (O : ℝ) : Prop := bc * ad = O -- representing BC tangent to O with diameter AD
def is_diameter (ad r : ℝ) : Prop := ad = 2 * r -- AD being the diameter of the circle with radius r

-- Given conditions in the problem
variables (AB BC CD AD r O : ℝ) (h1 : is_perpendicular AB BC 90) (h2 : is_perpendicular BC CD 90)
          (h3 : tangent_length BC AD O) (h4 : is_diameter AD r) (h5 : BC = 2 * CD)
          (h6 : AB = 9) (h7 : CD = 3)

-- Statement to prove the area is 36
theorem area_of_trapezoid : (AB + CD) * CD = 36 := by
  sorry

end area_of_trapezoid_l206_206196


namespace new_avg_weight_l206_206699

theorem new_avg_weight (A B C D E : ℝ) (h1 : (A + B + C) / 3 = 84) (h2 : A = 78) 
(h3 : (B + C + D + E) / 4 = 79) (h4 : E = D + 6) : 
(A + B + C + D) / 4 = 80 :=
by
  sorry

end new_avg_weight_l206_206699


namespace a9_value_l206_206142

theorem a9_value:
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ), 
  (∀ (x : ℝ), x^2 + x^(10) = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^(2) + a_3 * (x + 1)^(3) + a_4 * (x + 1)^(4) + a_5 * (x + 1)^(5) + a_6 * (x + 1)^(6) + a_7 * (x + 1)^(7) + a_8 * (x + 1)^(8) + a_9 * (x + 1)^(9) + a_{10} * (x + 1)^(10)) → 
  a_9 = -10 :=
by 
  intros a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10}
  intro h
  sorry

end a9_value_l206_206142


namespace statement_A_statement_B_statement_C_statement_D_l206_206588

-- For all theorems, a triangle ABC with sides a, b, c and angles A, B, and C respectively.

-- Statement A
theorem statement_A (A B : ℝ) (h : A > B) : sin A > sin B :=
sorry

-- Statement B
theorem statement_B (A : ℝ) (a : ℝ) (hA : A = π / 6) (ha : a = 5) : 
  2 * a / sin A = 10 :=
by
  have hR : 2 * a / sin A = 2 * 5 / sin A := by rw [ha]
  have hπ : 2 * 5 / sin (π / 6) = 10 := by
    rw [sin_pi_div_six]
    norm_num
  rw [hA, hπ]
  sorry

-- Statement C
theorem statement_C (a b c : ℝ) (A B C : ℝ) (h : a = 2 * b * cos C) : 
  is_isosceles (A B C) :=
sorry

-- Statement D
theorem statement_D (a b c : ℝ) (A : ℝ) (hb : b = 1) (hc : c = 2) (hA : A = 2 * π / 3) : 
  area a b c A = sqrt 3 / 2 :=
by
  have h_area : 1 / 2 * b * c * sin A = 1 / 2 * 1 * 2 * sin (2 * π / 3) := by
    rw [hb, hc]
  rw [hA, sin_two_pi_div_three, mul_comm (1 / 2), mul_assoc, mul_comm (sqrt 3 / 2)] at h_area
  sorry

end statement_A_statement_B_statement_C_statement_D_l206_206588


namespace no_valid_pairs_exist_l206_206867

open Nat

theorem no_valid_pairs_exist :
  ∀ (a b : ℕ), 0 < a → 0 < b → a * b + 52 = 20 * lcm a b + 15 * gcd a b → False :=
by
  intros a b hapos hbpos hab_eq
  sorry

end no_valid_pairs_exist_l206_206867


namespace minimum_S_n_value_l206_206489

-- Definitions based on conditions
def H (a : ℕ → ℤ) (n : ℕ) : ℤ := (∑ i in finset.range n, (2^i) * a (i + 1)) / n

axiom H_value (a : ℕ → ℤ) (n : ℕ) : H a n = 2^(n + 1)

-- The goal is to find the minimum value of S_n, where S_n denotes the sum of the first n terms of the sequence (a_n - 20)
def S (a : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in finset.range n, a (i + 1) - 20

theorem minimum_S_n_value : ∃ n, S (λ n, 2 * (n + 1)) n = -72 :=
by {
  -- Proof omitted
  sorry 
}

end minimum_S_n_value_l206_206489


namespace hyperbola_same_foci_l206_206921

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

noncomputable def ellipse_equation :=
  ∀ x y : ℝ, x^2 / 16 + y^2 / 9 = 1

theorem hyperbola_same_foci (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h_focuses : ∃ (c : ℝ), c = Real.sqrt 7 ∧ a^2 + b^2 = c^2)
  (h_eccentricity : ∃ (e_e e_h: ℝ), e_e = (Real.sqrt 7) / 4 ∧ e_h = 2 * e_e ∧ e_h = (Real.sqrt 7) / a) :
  hyperbola_equation 2 (Real.sqrt 3) (by norm_num1) (by norm_num1) :=
sorry

end hyperbola_same_foci_l206_206921


namespace impossibility_of_dividing_stones_l206_206281

theorem impossibility_of_dividing_stones (stones : ℕ) (heaps : ℕ) (m : ℕ) :
  stones = 660 → heaps = 31 → (∀ (x y : ℕ), x ∈ heaps → y ∈ heaps → x < 2 * y → False) := sorry

end impossibility_of_dividing_stones_l206_206281


namespace number_of_possible_values_l206_206658

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206658


namespace unit_digit_of_square_l206_206166

theorem unit_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := sorry

end unit_digit_of_square_l206_206166


namespace polynomial_roots_l206_206461

theorem polynomial_roots : (∃ x : ℝ, (4 * x ^ 4 + 11 * x ^ 3 - 37 * x ^ 2 + 18 * x = 0) ↔ (x = 0 ∨ x = 1 / 2 ∨ x = 3 / 2 ∨ x = -6)) :=
by 
  sorry

end polynomial_roots_l206_206461


namespace brick_width_l206_206780

theorem brick_width (length_courtyard : ℕ) (width_courtyard : ℕ) (num_bricks : ℕ) (brick_length : ℕ) (total_area : ℕ) (brick_area : ℕ) (w : ℕ)
  (h1 : length_courtyard = 1800)
  (h2 : width_courtyard = 1200)
  (h3 : num_bricks = 30000)
  (h4 : brick_length = 12)
  (h5 : total_area = length_courtyard * width_courtyard)
  (h6 : total_area = num_bricks * brick_area)
  (h7 : brick_area = brick_length * w) :
  w = 6 :=
by
  sorry

end brick_width_l206_206780


namespace words_per_page_l206_206386

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 120) : p = 195 := by
  sorry

end words_per_page_l206_206386


namespace binom_problem_l206_206977

noncomputable def binom (x : ℝ) (k : ℕ) : ℝ := x * (x - 1) * (x - 2) * ... * (x - k + 1) / k.fact

theorem binom_problem : 
  (binom (1/2) 1007 * (8 : ℝ) ^ 1007 / binom 2014 1007) = -((8 : ℝ) * (2 : ℝ) ^ 2014 / 2014.fact) := by
    sorry

end binom_problem_l206_206977


namespace value_b15_l206_206925

def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d
def geometric_sequence (b : ℕ → ℤ) := ∃ q : ℤ, ∀ n : ℕ, b (n+1) = q * b n

theorem value_b15 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : ∀ n : ℕ, S n = (n * (a 0 + a (n-1)) / 2))
  (h3 : S 9 = -18)
  (h4 : S 13 = -52)
  (h5 : geometric_sequence b)
  (h6 : b 5 = a 5)
  (h7 : b 7 = a 7) : 
  b 15 = -64 :=
sorry

end value_b15_l206_206925


namespace num_of_possible_values_b_l206_206682

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206682


namespace number_of_possible_values_l206_206657

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l206_206657


namespace most_profit_increase_after_1998_is_2002_l206_206707

noncomputable def profit (year : ℕ) : ℕ :=
  if year = 1998 then 20
  else if year = 2000 then 40
  else if year = 2002 then 70
  else if year = 2004 then 80
  else if year = 2006 then 100
  else 0

def profit_increase (year1 year2 : ℕ) : ℕ :=
  profit year2 - profit year1

theorem most_profit_increase_after_1998_is_2002 :
  (∀ year, profit_increase 1998 year ≤ profit_increase 2000 2002) →
  ∃ year = 2002, profit_increase 1998 year = profit_increase 2000 2002 :=
  sorry

end most_profit_increase_after_1998_is_2002_l206_206707


namespace length_of_bridge_is_correct_l206_206787

noncomputable def length_of_inclined_bridge (initial_speed : ℕ) (time : ℕ) (acceleration : ℕ) : ℚ :=
  (1 / 60) * (time * initial_speed + (time * (time - 1)) / 2)

theorem length_of_bridge_is_correct : 
  length_of_inclined_bridge 10 18 1 = 5.55 := 
by
  sorry

end length_of_bridge_is_correct_l206_206787


namespace toys_complete_on_time_l206_206728

theorem toys_complete_on_time : 
  ∀ (start_day end_day total_toys avg_first_three avg_remaining) 
    (days_first : ℕ) 
    (days_remaining : ℕ), 
  start_day = 21 → 
  end_day = 31 → 
  total_toys = 3000 → 
  avg_first_three = 250 → 
  avg_remaining = 375 → 
  days_first = 3 → 
  days_remaining = 6 →
  days_first + days_remaining < end_day - start_day + 1 → 
  "They can complete the production task on time." := 
by
  intros
  sorry

end toys_complete_on_time_l206_206728


namespace domain_of_sqrt_function_l206_206089

theorem domain_of_sqrt_function :
  ∀ x : ℝ, (∃ y : ℝ, y = sqrt (18 + 3 * x - x^2)) ↔ (-3 ≤ x ∧ x ≤ 6) :=
by
  sorry

end domain_of_sqrt_function_l206_206089


namespace prob_B_wins_first_game_l206_206690

noncomputable def sequences (A_wins_series: Bool) (B_wins_second_game: Bool): List (List Char) :=
if A_wins_series && B_wins_second_game
then [['B', 'B', 'A', 'A', 'A'],
      ['A', 'B', 'B', 'A', 'A'],
      ['A', 'B', 'A', 'B', 'A'],
      ['A', 'B', 'A', 'A', 'B']]
else []

def favorable_outcomes (all_seqs: List (List Char)): Nat :=
(all_seqs.filter (λ seq => seq.headD '' == 'B')).length

def total_seqs (all_seqs: List (List Char)): Nat :=
all_seqs.length

theorem prob_B_wins_first_game :
  (favorable_outcomes (sequences True True)).toFloat / 
  (total_seqs (sequences True True)).toFloat = 1.0 / 4 := by
sorry

end prob_B_wins_first_game_l206_206690


namespace weight_small_cube_is_7_l206_206002

noncomputable def weight_small_cube (s : ℝ) : ℝ :=
  let V_small := s ^ 3
  let V_large := (2 * s) ^ 3
  let W_large := 56
  by
    let proportion := V_small / V_large = W_small / W_large
    have h1 : V_small = s ^ 3 := rfl
    have h2 : V_large = 8 * (s ^ 3) := by norm_num
    have h3 : proportion = (s ^ 3) / (8 * (s ^ 3)) = W_small / 56 := by simp [h1, h2]
    have h4 : (s ^ 3) / (8 * (s ^ 3)) = 1 / 8 := by field_simp; norm_num
    have h5 : 1 / 8 = W_small / 56 := by rwa h4 at h3
    exact (W_small : ℝ) = (1 / 8) * 56 sorry

theorem weight_small_cube_is_7 (s : ℝ) : weight_small_cube s = 7 :=
  sorry

end weight_small_cube_is_7_l206_206002


namespace no_division_660_stones_into_31_heaps_l206_206248

theorem no_division_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) :=
  sorry

end no_division_660_stones_into_31_heaps_l206_206248


namespace longest_side_of_polygonal_region_l206_206828

theorem longest_side_of_polygonal_region :
  ∃ A B : ℝ × ℝ, (A = (1, 0) ∧ B = (0, 5)) ∨ 
                 (A = (0, 3) ∧ B = (1, 0)) ∨
                 (A = (0, 3) ∧ B = (0, 5)) 
                 ∧
                 (x y : ℝ), 
                 x + y ≤ 5 ∧ 3 * x + y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0 
                → dist A B = real.sqrt 26 := 
sorry

end longest_side_of_polygonal_region_l206_206828


namespace avg_people_moving_per_hour_l206_206293

theorem avg_people_moving_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) : 
  total_people = 3500 → days = 5 → hours_per_day = 24 → 
  let total_hours := days * hours_per_day in 
  let avg_people_per_hour := total_people / total_hours in
  avg_people_per_hour ≈ 29 :=
by
  intros h1 h2 h3
  let total_hours := days * hours_per_day
  have h_total_hours : total_hours = 120 := by
    calc
      total_hours = days * hours_per_day : rfl
      ... = 5 * 24 : by rw [h2, h3]
      ... = 120 : rfl
  let avg_people_per_hour := total_people / total_hours
  have h_avg_people_per_hour : avg_people_per_hour = 29 := by
    calc
      avg_people_per_hour = total_people / total_hours : rfl
      ... = 3500 / 120 : by rw [h1, h_total_hours]
      ... = 29 : by norm_num
  exact h_avg_people_per_hour


end avg_people_moving_per_hour_l206_206293


namespace num_of_possible_values_b_l206_206685

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206685


namespace price_25_bag_l206_206003

noncomputable def price_per_bag_25 : ℝ := 28.97

def price_per_bag_5 : ℝ := 13.85
def price_per_bag_10 : ℝ := 20.42

def total_cost (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ) : ℝ :=
  n5 * p5 + n10 * p10 + n25 * p25

theorem price_25_bag :
  ∃ (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ),
    p5 = price_per_bag_5 ∧
    p10 = price_per_bag_10 ∧
    p25 = price_per_bag_25 ∧
    65 ≤ 5 * n5 + 10 * n10 + 25 * n25 ∧
    5 * n5 + 10 * n10 + 25 * n25 ≤ 80 ∧
    total_cost p5 p10 p25 n5 n10 n25 = 98.77 :=
by
  sorry

end price_25_bag_l206_206003


namespace unique_positive_solution_l206_206643

open Classical

noncomputable def polynomial_2020(x : ℝ) : ℝ :=
  x * (x + 1) * (x + 2) * ... * (x + 2020) - 1

theorem unique_positive_solution (x0 : ℝ) :
  (∃! x, x > 0 ∧ polynomial_2020 x = 0) ∧
  (polynomial_2020 x0 = 0 →
   x0 > 0 ∧ (1/(2020.factorial + 10) < x0 ∧ x0 < 1/(2020.factorial + 6))) :=
sorry

end unique_positive_solution_l206_206643


namespace intersect_x_axis_iff_k_le_4_l206_206518

theorem intersect_x_axis_iff_k_le_4 (k : ℝ) :
  (∃ x : ℝ, (k-3) * x^2 + 2 * x + 1 = 0) ↔ k ≤ 4 :=
sorry

end intersect_x_axis_iff_k_le_4_l206_206518


namespace cos_shift_right_l206_206348

theorem cos_shift_right (x : ℝ) : 
  ∃ c : ℝ, (∀ x, cos (2 * x) = cos (2 * (x + c + π/12))) ∧ (c = -π/12) :=
by
  use -π/12
  intros x
  have h1 : cos (2 * x) = cos (2 * (x - π/12 + π/12)) := by rw [add_sub_cancel]
  rw [sub_eq_add_neg, h1, cos_add]
  sorry

end cos_shift_right_l206_206348


namespace kayla_total_items_l206_206212

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l206_206212


namespace expense_recording_l206_206168

-- Define the recording of income and expenses
def record_income (amount : Int) : Int := amount
def record_expense (amount : Int) : Int := -amount

-- Given conditions
def income_example := record_income 500
def expense_example := record_expense 400

-- Prove that an expense of 400 yuan is recorded as -400 yuan
theorem expense_recording : record_expense 400 = -400 :=
  by sorry

end expense_recording_l206_206168


namespace arithmetic_sequence_sum_l206_206821

theorem arithmetic_sequence_sum :
  let first_term := 1
  let common_diff := 2
  let last_term := 33
  let n := (last_term + 1) / common_diff
  (n * (first_term + last_term)) / 2 = 289 :=
by
  sorry

end arithmetic_sequence_sum_l206_206821


namespace total_revenue_correct_l206_206726

def sections := 5
def seats_per_section_1_4 := 246
def seats_section_5 := 314
def ticket_price_1_4 := 15
def ticket_price_5 := 20

theorem total_revenue_correct :
  4 * seats_per_section_1_4 * ticket_price_1_4 + seats_section_5 * ticket_price_5 = 21040 :=
by
  sorry

end total_revenue_correct_l206_206726


namespace dr_reeds_statement_l206_206838

variables (P Q : Prop)

theorem dr_reeds_statement (h : P → Q) : ¬Q → ¬P :=
by sorry

end dr_reeds_statement_l206_206838


namespace ratio_XY_BC_l206_206569

variables (a b c : ℝ) -- angles of the acute triangle ABC
variables (ABC : Triangle) -- the triangle
variables [is_acute ABC] -- ABC is an acute triangle

-- points of tangency of the inscribed circle with sides BC, AB, and AC
variables (D E F : Point)
variables [tangency D ABC.BC]
variables [tangency E ABC.AB]
variables [tangency F ABC.AC]

-- circle gamma with diameter BC and its intersections with EF at X and Y
variables (γ : Circle ABC.BC)
variables (X Y : Point)
variables [intersects γ (Line_segment E F) X Y]

-- proof statement
theorem ratio_XY_BC (h : ∠ABC = a ∧ ∠BCA = b ∧ ∠CAB = c) :
  XY.length / ABC.BC.length = real.sin (a / 2) :=
sorry

end ratio_XY_BC_l206_206569


namespace min_value_trig_expr_l206_206074

theorem min_value_trig_expr : 
  ∃ A : ℝ, A ∈ set.Icc 0 (2 * Real.pi) ∧
  2 * Real.sin (A / 2) + Real.sin A = -4 :=
sorry

end min_value_trig_expr_l206_206074


namespace students_neither_play_l206_206991

theorem students_neither_play (total_students football_players cricket_players both_play : ℕ)
  (h_total : total_students = 250)
  (h_football : football_players = 160)
  (h_cricket : cricket_players = 90)
  (h_both : both_play = 50) :
  total_students - (football_players + cricket_players - both_play) = 50 :=
by {
  rw [h_total, h_football, h_cricket, h_both],
  norm_num,
}

end students_neither_play_l206_206991


namespace tangent_line_of_f_eq_kx_l206_206912

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

theorem tangent_line_of_f_eq_kx (k : ℝ) : 
    (∃ x₀, tangent_line k x₀ = f x₀ ∧ deriv f x₀ = k) → 
    (k = 0 ∨ k = 1 ∨ k = -1) := 
  sorry

end tangent_line_of_f_eq_kx_l206_206912


namespace correct_statements_l206_206372

section
  -- Define the events as predicates on the results of the two throws
  variables {Ω : Type} [ProbabilitySpace Ω]
  variables (throw1 throw2 : Ω → ℕ)
  
  -- Define event A: the first time 2 points appear
  def A (ω : Ω) : Prop := throw1 ω = 2
  
  -- Define event B: the second time the points are less than 5
  def B (ω : Ω) : Prop := throw2 ω < 5
  
  -- Define event C: the sum of the two points is odd
  def C (ω : Ω) : Prop := (throw1 ω + throw2 ω) % 2 = 1
  
  -- Define event D: the sum of the two points is 9
  def D (ω : Ω) : Prop := throw1 ω + throw2 ω = 9
  
  -- Define the probability space (replace this with the actual distribution if necessary)
  variable [FairDie : ∀ ω : Ω, throw1 ω ∈ Finset.range 1 7 ∧ throw2 ω ∈ Finset.range 1 7]
  
  -- Prove the statements about the events
  theorem correct_statements : 
    (
      (¬ Disjoint A B) ∧ Independent A B ∧ 
      Disjoint A D ∧ ¬ Independent A D ∧ 
      ¬ Disjoint B D ∧ -- corrected for C incorrect statement
      Independent A C ∧ 
      (¬ Disjoint A C)
    ) 
  := sorry

end

end correct_statements_l206_206372


namespace monthly_manufacturing_expenses_l206_206026

theorem monthly_manufacturing_expenses 
  (num_looms : ℕ) (total_sales_value : ℚ) 
  (monthly_establishment_charges : ℚ) 
  (decrease_in_profit : ℚ) 
  (sales_per_loom : ℚ) 
  (manufacturing_expenses_per_loom : ℚ) 
  (total_manufacturing_expenses : ℚ) : 
  num_looms = 80 → 
  total_sales_value = 500000 → 
  monthly_establishment_charges = 75000 → 
  decrease_in_profit = 4375 → 
  sales_per_loom = total_sales_value / num_looms → 
  manufacturing_expenses_per_loom = sales_per_loom - decrease_in_profit → 
  total_manufacturing_expenses = manufacturing_expenses_per_loom * num_looms →
  total_manufacturing_expenses = 150000 :=
by
  intros h_num_looms h_total_sales h_monthly_est_charges h_decrease_in_profit h_sales_per_loom h_manufacturing_expenses_per_loom h_total_manufacturing_expenses
  sorry

end monthly_manufacturing_expenses_l206_206026


namespace range_of_m_l206_206553

noncomputable def f (x : ℝ) (m : ℝ) := 9^x - m * 3^x - 3

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f x m = 9^x - m * 3^x - 3) →
  (∀ x : ℝ, f (-x) m = - f(x) m) →
  -2 ≤ m :=
sorry

end range_of_m_l206_206553


namespace card_division_ways_count_l206_206721

theorem card_division_ways_count :
  let cards := {x : ℕ | 15 ≤ x ∧ x ≤ 33}
  let participants := {Vasya, Petya, Misha}
  (∃ (f : cards → participants), 
    (∀ p ∈ participants, ∃ c ∈ cards, f c = p) ∧ 
    (∀ p ∈ participants, (∀ c₁ c₂ ∈ cards, f c₁ = p ∧ f c₂ = p ∧ c₁ ≠ c₂ →
      (c₁ - c₂) % 2 = 0))) →
  -- Proven that there are exactly 4596 ways to divide the cards.
  (card {f : cards → participants | 
    (∀ p ∈ participants, ∃ c ∈ cards, f c = p) ∧ 
    (∀ p ∈ participants, (∀ c₁ c₂ ∈ cards, f c₁ = p ∧ f c₂ = p ∧ c₁ ≠ c₂ →
      (c₁ - c₂) % 2 = 0))}) = 4596 :=
sorry

end card_division_ways_count_l206_206721


namespace trig_function_max_value_l206_206332

-- Define the function f(x) as given in the problem
def f (x : ℝ) : ℝ := (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- Define the interval [π/4, π/2]
def interval (x : ℝ) : Prop := (Real.pi / 4) ≤ x ∧ x ≤ (Real.pi / 2)

-- Statement of the problem, proving the maximum value over the given interval
theorem trig_function_max_value : 
  ∃ x ∈ set.Icc (Real.pi / 4) (Real.pi / 2), ∀ y ∈ set.Icc (Real.pi / 4) (Real.pi / 2), f y ≤ f x :=
sorry

end trig_function_max_value_l206_206332


namespace part1_part2_l206_206526

variable {U : Set ℝ} (A B M : Set ℝ)

def A := {x : ℝ | x < -4 ∨ x > 1}
def B := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def M (k : ℝ) := {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

theorem part1 (U : Set ℝ) (A B : Set ℝ) :
  (A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 3}) ∧ (compl A ∪ compl B = compl (A ∩ B)) :=
by
  sorry

theorem part2 (A : Set ℝ) :
  (∀ k : ℝ, M k ⊆ A → k < -5/2 ∨ k > 1) :=
by
  sorry

end part1_part2_l206_206526


namespace evaluate_powers_of_i_l206_206841

-- Define complex number "i"
def i := Complex.I

-- Define the theorem to prove
theorem evaluate_powers_of_i : i^44 + i^444 + 3 = 5 := by
  -- use the cyclic property of i to simplify expressions
  sorry

end evaluate_powers_of_i_l206_206841


namespace max_product_from_digits_l206_206735

theorem max_product_from_digits :
  ∃ (a b c d e : ℕ), 
    {a, b, c, d, e} = {2, 4, 6, 7, 9} ∧ 
    (100 * a + 10 * b + c) * (10 * d + e) = 762 * 94 :=
begin
  sorry
end

end max_product_from_digits_l206_206735


namespace max_f_value_l206_206862

noncomputable def f (x : ℝ) : ℝ := 9 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 15 :=
by
  sorry

end max_f_value_l206_206862


namespace real_root_of_system_l206_206836

theorem real_root_of_system :
  (∃ x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0) ↔ x = -3 := 
by 
  sorry

end real_root_of_system_l206_206836


namespace slope_gt_sqrt3_l206_206927

variable {a b : ℝ} (h₀ : a > b > 0)
variable (A B P: ℝ × ℝ)
hypothesis (h₁ : A = (-a, 0))
hypothesis (h₂ : B = (a, 0))
hypothesis (P_on_ellipse : (P.1)^2 / a^2 + (P.2)^2 / b^2 = 1)
hypothesis (P_not_AB : P ≠ A ∧ P ≠ B)
hypothesis (AP_OA_equal: dist A P = a)

theorem slope_gt_sqrt3 (k : ℝ) (slope_def : k = P.2 / P.1) : |k| > √3 := 
by 
  have hP : P ∈ set_of (λ p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1) :=
    by exact P_on_ellipse
  sorry

end slope_gt_sqrt3_l206_206927


namespace pow_mul_inv_eq_one_problem_theorem_l206_206364

theorem pow_mul_inv_eq_one (a : ℚ) (n : ℤ) (h_a : a ≠ 0) : 
  (a ^ (n : ℚ)) * (a ^ (-n : ℚ)) = 1 :=
by
  have h1 : a ^ (n : ℚ) * a ^ (-n : ℚ) = a ^ (n + -n : ℚ), by sorry
  have h2 : (n + -n : ℚ) = 0, by sorry
  have h3 : a ^ 0 = 1, by sorry
  rw [h1, h2, h3]

-- Constants specific to the problem
def rational_a : ℚ := 9 / 11
def int_n : ℤ := 4
def int_neg_n : ℤ := -4

-- Ensure a is nonzero
axiom a_nonzero : rational_a ≠ 0

-- The main theorem specific to the problem
theorem problem_theorem : 
  (rational_a ^ (int_n : ℚ)) * (rational_a ^ (int_neg_n : ℚ)) = 1 :=
pow_mul_inv_eq_one rational_a int_n a_nonzero

end pow_mul_inv_eq_one_problem_theorem_l206_206364


namespace heap_division_impossible_l206_206245

def similar_sizes (a b : Nat) : Prop := a < 2 * b

theorem heap_division_impossible (n k : Nat) (h1 : n = 660) (h2 : k = 31) 
    (h3 : ∀ p1 p2 : Nat, p1 ∈ (Finset.range k).to_set → p2 ∈ (Finset.range k).to_set → similar_sizes p1 p2 → p1 < 2 * p2) : 
    False := 
by
  sorry

end heap_division_impossible_l206_206245


namespace alice_next_birthday_age_l206_206033

theorem alice_next_birthday_age (a b c : ℝ) 
  (h1 : a = 1.25 * b)
  (h2 : b = 0.7 * c)
  (h3 : a + b + c = 30) : a + 1 = 11 :=
by {
  sorry
}

end alice_next_birthday_age_l206_206033


namespace max_height_of_object_l206_206039

theorem max_height_of_object :
  (∀ t : ℝ, h t = -15 * (t - 2) ^ 2 + 150) →
  h 2 = 150 :=
by
  sorry

end max_height_of_object_l206_206039


namespace dispatch_methods_correct_l206_206305

noncomputable def comb (n k : ℕ) : ℕ := nat.choose n k
noncomputable def perm (n k : ℕ) : ℕ := nat.factorial n / nat.factorial (n - k)

open nat 

theorem dispatch_methods_correct :
  let num_male := 5 in
  let num_female := 4 in
  let num_areas := 3 in
  comb (num_male + num_female) 3 - comb num_male 3 - comb num_female 3 * perm num_areas 3 = 420 :=
by
  intros,
  sorry

end dispatch_methods_correct_l206_206305


namespace area_of_ABCD_is_correct_l206_206404

-- Definitions
open Float
structure RectangularPrism where
  (length width height : ℝ)

structure Point where
  x y z : ℝ

-- Given conditions
def A : Point := ⟨0, 0, 0⟩
def C (p : RectangularPrism) : Point := ⟨p.length, p.width, p.height⟩
def midpoint (p1 p2 : Point) : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2⟩
def B (p : RectangularPrism) : Point := midpoint ⟨0, 0, p.height⟩ ⟨p.length, 0, p.height⟩ -- midpoint of top edge
def D (p : RectangularPrism) : Point := midpoint ⟨0, p.width, 0⟩ ⟨p.length, p.width, 0⟩ -- midpoint of bottom edge

def area_quadrilateral (A B C D : Point) : ℝ :=
  let diag1 := (AC : ℝ) := sqrt ((C.x - A.x)^2 + (C.y - A.y)^2 + (C.z - A.z)^2)
  let diag2 := (BD : ℝ) := sqrt ((D.x - B.x)^2 + (D.y - B.y)^2 + (D.z - B.z)^2)
  (diag1 * diag2) / 2

-- Proof statement
theorem area_of_ABCD_is_correct (p : RectangularPrism) (h : p.length = 2 ∧ p.width = 3 ∧ p.height = 5) :
  area_quadrilateral A (B p) (C p) (D p) = (7*sqrt 26)/2 := by
sorry

end area_of_ABCD_is_correct_l206_206404


namespace part1_part2_l206_206938

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem part1 (x : ℝ) : |f (-x)| + |f x| ≥ 4 * |x| := 
by
  sorry

theorem part2 (x a : ℝ) (h : |x - a| < 1 / 2) : |f x - f a| < |a| + 5 / 4 := 
by
  sorry

end part1_part2_l206_206938


namespace exists_zero_in_interval_l206_206319

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 1 / x

theorem exists_zero_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- This is just the Lean statement, no proof is provided
  sorry

end exists_zero_in_interval_l206_206319


namespace find_nonzero_q_for_quadratic_l206_206847

theorem find_nonzero_q_for_quadratic :
  ∃ (q : ℝ), q ≠ 0 ∧ (∀ (x1 x2 : ℝ), (q * x1^2 - 8 * x1 + 2 = 0 ∧ q * x2^2 - 8 * x2 + 2 = 0) → x1 = x2) ↔ q = 8 :=
by
  sorry

end find_nonzero_q_for_quadratic_l206_206847


namespace find_point_Q_l206_206139

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1 / 3) * x^3 + 2*x - m + m / x

theorem find_point_Q :
  ∀ (m : ℝ), (0 < m ∧ m ≤ 3 ∧ ∀ x: ℝ, 1 ≤ x → g'(x) ≥ 0) →
  ∃ Q : ℝ × ℝ, Q = (0, -3) ∧ (∀ line, encloses_two_equal_areas(line, Q, g)) :=
begin
  sorry
end

end find_point_Q_l206_206139


namespace total_money_l206_206786

theorem total_money (total_coins nickels dimes : ℕ) (val_nickel val_dime : ℕ)
  (h1 : total_coins = 8)
  (h2 : nickels = 2)
  (h3 : total_coins = nickels + dimes)
  (h4 : val_nickel = 5)
  (h5 : val_dime = 10) :
  (nickels * val_nickel + dimes * val_dime) = 70 :=
by
  sorry

end total_money_l206_206786


namespace smallest_period_of_f_range_of_f_interval_strictly_decreasing_sin_2_alpha_eq_l206_206517

def f (x : ℝ) : ℝ := cos (x / 2) ^ 2 - sin (x / 2) * cos (x / 2) - 1 / 2

-- Problem 1: Period of the function
theorem smallest_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x := 
  sorry

-- Problem 2: Range of the function
theorem range_of_f : ∃ y, f y ∈ {z | -√2 / 2 ≤ z ∧ z ≤ √2 / 2} := 
  sorry

-- Problem 3: Interval of decreasing
theorem interval_strictly_decreasing (k : ℤ) : 
  ∃ I, I = set.Icc (2 * k * π - π / 4) (2 * k * π + 3 * π / 4) ∧ 
       ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y := 
  sorry

-- Problem 4: Value of sin 2α
theorem sin_2_alpha_eq (α : ℝ) (h : f α = 3 * √2 / 10) : sin (2 * α) = 16 / 25 := 
  sorry

end smallest_period_of_f_range_of_f_interval_strictly_decreasing_sin_2_alpha_eq_l206_206517


namespace sum_inverse_squares_accuracy_sum_inverse_factorials_accuracy_l206_206376

-- Part (a) proof problem statement
theorem sum_inverse_squares_accuracy :
  abs ((∑ k in finset.range 991 (λ n, 1 / ((n + 10)^2))) - 0.105) ≤ 0.006 :=
begin
  sorry
end

-- Part (b) proof problem statement
theorem sum_inverse_factorials_accuracy :
  abs ((∑ k in finset.range 991 (λ n, 1 / (nat.factorial (n + 10)))) - 0.00000029) ≤ 0.000000015 :=
begin
  sorry
end

end sum_inverse_squares_accuracy_sum_inverse_factorials_accuracy_l206_206376


namespace discount_profit_percentage_l206_206796

theorem discount_profit_percentage
  (CP : ℝ := 100) -- Cost Price
  (percentage_profit_no_discount : ℝ := 50) -- 50% profit without discount
  (discount_rate : ℝ := 5) -- 5% discount rate
  : 
  let SP_no_discount := CP * (1 + percentage_profit_no_discount / 100) in
  let SP_discounted := SP_no_discount * (1 - discount_rate / 100) in
  let profit_with_discount := SP_discounted - CP in 
  let profit_percentage_with_discount := (profit_with_discount / CP) * 100 in 
  profit_percentage_with_discount = 42.5 :=
by
  sorry

end discount_profit_percentage_l206_206796


namespace square_area_l206_206036

theorem square_area (x : ℚ) :
  (5 * x - 22 = 34 - 4 * x) →
  let side := (5 * x - 22) in
  side^2 = 6724 / 81 :=
by sorry

end square_area_l206_206036


namespace slope_tangent_roots_derivative_eq_zero_extreme_point_one_l206_206937

noncomputable def f (x a : ℝ) : ℝ := x * Real.cos x + a
noncomputable def g (x : ℝ) : ℝ := Real.cos x - x * Real.sin x
noncomputable def F (x a : ℝ) : ℝ := x * Real.sin x + Real.cos x + a * x

-- part (I)
theorem slope_tangent (a : ℝ) :
  deriv (λ x => f x a) (Real.pi / 2) = - (Real.pi / 2) := by
  sorry

-- part (II)
theorem roots_derivative_eq_zero :
  ∃! x ∈ Ioo 0 1, g x = 0 := by
  sorry

-- part (III)
theorem extreme_point_one (a : ℝ) :
  (∃ (c ∈ Ioo 0 1), deriv (λ x => F x a) c = 0) ∧ 
  (∀ (x ∈ Ioo 0 c), deriv (λ x => F x a) x > 0) ∧ 
  (∀ (x ∈ Ioc c 1), deriv (λ x => F x a) x < 0) ↔ 
  (-Real.cos 1 ≤ a ∧ a < 0) := by
  sorry

end slope_tangent_roots_derivative_eq_zero_extreme_point_one_l206_206937


namespace average_age_of_population_l206_206184

-- Define the conditions and the target statement to prove in Lean 4
theorem average_age_of_population (k : ℕ) : 
  let women := 10 * k,
      men := 9 * k,
      total_population := women + men,
      total_age := women * 36 + men * 33,
      average_age := (total_age : ℚ) / total_population
  in average_age = 34 + 13 / 19 := 
by
  sorry

end average_age_of_population_l206_206184


namespace analytical_method_l206_206736

theorem analytical_method (a : ℝ) (h : 1 < a) : 
    sqrt (a + 1) + sqrt (a - 1) < 2 * sqrt a := 
by 
  sorry

end analytical_method_l206_206736


namespace couriers_travel_times_l206_206415

theorem couriers_travel_times (a : ℝ) (x y z : ℝ) 
  (h1 : a < 30) 
  (h2 : z * (x + a) = 180) 
  (h3 : (z - 3) * y = 180) 
  (h4 : 180 / x - 180 / y = 6) :
  z = (-3 * a + 3 * real.sqrt (a ^ 2 + 240 * a)) / (2 * a) ∧ 
  z - 3 = (-9 * a + 3 * real.sqrt (a ^ 2 + 240 * a)) / (2 * a) :=
  by sorry

end couriers_travel_times_l206_206415


namespace kayak_rental_cost_l206_206350

theorem kayak_rental_cost (F : ℝ) (C : ℝ) (h1 : ∀ t : ℝ, C = F + 5 * t)
  (h2 : C = 30) : C = 45 :=
sorry

end kayak_rental_cost_l206_206350


namespace triangle_ABC_angle_AFE_l206_206586

theorem triangle_ABC_angle_AFE 
  (A B C D E F : Type) 
  [EuclideanGeometry A B C] 
  (h1 : ∠ABC = 36) 
  (h2 : D ∈ line AC) 
  (h3 : E ∈ line AB) 
  (h4 : F ∈ line BC) 
  (h5 : dist A D = dist F D)
  (h6 : dist F E = dist E C) 
  (h7 : dist A D = dist E C) : 
  ∠AFE = 126 :=
sorry

end triangle_ABC_angle_AFE_l206_206586


namespace train_length_l206_206412

theorem train_length (L : ℕ) 
  (h_tree : L / 120 = L / 200 * 200) 
  (h_platform : (L + 800) / 200 = L / 120) : 
  L = 1200 :=
by
  sorry

end train_length_l206_206412


namespace enclosed_area_abs_eq_54_l206_206744

theorem enclosed_area_abs_eq_54 :
  (∃ (x y : ℝ), abs x + abs (3 * y) = 9) → True := 
by
  sorry

end enclosed_area_abs_eq_54_l206_206744


namespace three_roots_exactly_l206_206087

theorem three_roots_exactly (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = sqrt (x^2 - 2 * x - 3) * (x^2 - 3 * a * x + 2 * a^2)) ∧
   (Set.Countable {x | f x = 0} ∧ 
    (∀ x, x^2 - 2 * x - 3 ≥ 0 → f x = 0 → (x = -1 ∨ x = 3 ∨ (x ≠ -1 ∧ x ≠ 3))))) ↔ 
  (-1 ≤ a ∧ a < -1/2) ∨ (3/2 < a ∧ a ≤ 3) :=
by
  sorry

end three_roots_exactly_l206_206087


namespace tim_surprises_combinations_l206_206347

theorem tim_surprises_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 120 :=
by
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  sorry

end tim_surprises_combinations_l206_206347


namespace number_of_valid_paintings_l206_206383

-- Definition of the grid and the conditions
def is_valid_painting (grid : Array (Array Bool)) : Prop :=
  let n := 3
  Array.size grid = n ∧ (∀ i j, i < n → j < n →
    grid.get! i |>.get! j = true →
    (i + 1 < n → grid.get! (i + 1) |>.get! j = true) ∨
    (j + 1 < n → grid.get! i |>.get! (j + 1) = true))

-- The number of valid paintings
noncomputable def count_valid_paintings : Nat :=
  (Matrix fin 3).toArray.count is_valid_painting

-- Stating the theorem
theorem number_of_valid_paintings : count_valid_paintings = 16 := by
  sorry

end number_of_valid_paintings_l206_206383


namespace distance_from_home_to_high_school_l206_206599

theorem distance_from_home_to_high_school 
  (total_mileage track_distance d : ℝ)
  (h_total_mileage : total_mileage = 10)
  (h_track : track_distance = 4)
  (h_eq : d + d + track_distance = total_mileage) :
  d = 3 :=
by sorry

end distance_from_home_to_high_school_l206_206599


namespace a_5_eq_16_S_8_eq_255_l206_206145

open Nat

-- Definitions from the conditions
def a : ℕ → ℕ
| 0     => 1
| (n+1) => 2 * a n

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Proof problem statements
theorem a_5_eq_16 : a 4 = 16 := sorry

theorem S_8_eq_255 : S 8 = 255 := sorry

end a_5_eq_16_S_8_eq_255_l206_206145


namespace net_sale_value_l206_206558

theorem net_sale_value (P : ℝ) : 
  let day1_price := 0.90 * P,
      day1_sales := day1_price * 1.85,
      day2_price := 0.85 * day1_price,
      day2_price_tax := day2_price * 1.08,
      day2_sales := day2_price_tax * 1.50,
      day3_price := 0.80 * day2_price,
      day3_price_discount := day3_price * 0.95,
      day3_sales := day3_price_discount * 1.30,
      net_sale_value := day1_sales + day2_sales + day3_sales in
    net_sale_value = 3.66012 * P := by
  sorry

end net_sale_value_l206_206558


namespace binomial_identity_l206_206907

theorem binomial_identity :
  (Nat.choose 16 6 = 8008) → (Nat.choose 16 7 = 11440) → (Nat.choose 16 8 = 12870) →
  Nat.choose 18 8 = 43758 :=
by
  intros h1 h2 h3
  sorry

end binomial_identity_l206_206907


namespace count_elements_S_l206_206223

def sum_of_squares (n : ℕ) : Prop := ∃ a b : ℕ, a * a + b * b = n

def set_S : set ℕ := {n | sum_of_squares n ∧ n < 1000000}

theorem count_elements_S :
  set_S.to_finset.card = 215907 := 
sorry

end count_elements_S_l206_206223


namespace jim_age_in_2_years_l206_206351

theorem jim_age_in_2_years (c1 : ∀ t : ℕ, t = 37) (c2 : ∀ j : ℕ, j = 27) : ∀ j2 : ℕ, j2 = 29 :=
by
  sorry

end jim_age_in_2_years_l206_206351


namespace find_x_for_parallel_l206_206621

-- Definitions for vector components and parallel condition.
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_x_for_parallel :
  ∃ x : ℝ, parallel a (b x) ∧ x = -3 / 2 :=
by
  -- The statement to be proven
  sorry

end find_x_for_parallel_l206_206621


namespace total_cost_after_discount_is_correct_l206_206595

variables (num_bedroom_doors num_outside_doors num_bathroom_doors : ℕ)
variables (cost_outside_door cost_bedroom_door cost_bathroom_door total_cost_after_discount : ℝ)

-- Given John's conditions
def initial_conditions : Prop :=
  num_bedroom_doors = 3 ∧ 
  num_outside_doors = 2 ∧ 
  num_bathroom_doors = 1 ∧ 
  cost_outside_door = 20 ∧ 
  cost_bedroom_door = cost_outside_door / 2 ∧ 
  cost_bathroom_door = 2 * cost_outside_door ∧ 
  total_cost_after_discount = 106

-- We need to prove that under these conditions
theorem total_cost_after_discount_is_correct : initial_conditions → 
  let total_cost := num_outside_doors * cost_outside_door + 
                    num_bedroom_doors * cost_bedroom_door + 
                    num_bathroom_doors * cost_bathroom_door 
  in total_cost - 0.1 * cost_bathroom_door = total_cost_after_discount :=
by
  intros,
  sorry

end total_cost_after_discount_is_correct_l206_206595


namespace fraction_shaded_rectangle_l206_206992

theorem fraction_shaded_rectangle :
  let rectangles := [((1, 1), (1, 3)), ((2, 1), (2, 3))] -- indexes for squares
  let shaded_triangles := [((1, 1), (1, 3)), ((2, 2), (2, 2))]
  (shaded_area rectangles shaded_triangles) = 1 / 4 :=
sorry

end fraction_shaded_rectangle_l206_206992


namespace oranges_distribution_l206_206624

def initialOranges (x : ℝ) : ℝ := x
def firstSon (x : ℝ) : ℝ := (x / 2) + (1 / 2)
def afterFirstSon (x : ℝ) : ℝ := x - (firstSon x) -- = (x / 2) - (1 / 2)
def secondSon (x : ℝ) : ℝ := (1 / 2) * (afterFirstSon x) + (1 / 2)
def afterSecondSon (x : ℝ) : ℝ := afterFirstSon x - (secondSon x) -- = (x / 4) - (3 / 4)
def thirdSon (x : ℝ) : ℝ := (1 / 2) * (afterSecondSon x) + (1 / 2)

theorem oranges_distribution (x : ℝ) (h : thirdSon x = initialOranges x - (firstSon x + secondSon x)) : 
  x = 7 := 
sorry

end oranges_distribution_l206_206624


namespace collinear_DE_F_l206_206338

open EuclideanGeometry

-- Define the problem conditions such that triangle ABC is inscribed in circle O,
-- I is the incenter of this triangle, AI and BI intersect the circle at D and E respectively,
-- line l_I is drawn through I and parallel to AB, and a tangent line l_C through C
-- intersects l_I at F.

noncomputable def is_collinear {P Q R : Point} : Prop :=
  Collinear P Q R

theorem collinear_DE_F {A B C I D E F O : Point} (hO : Circle O) 
  (hIncenter : Incenter I A B C) (hAID : [Line A I] and InCircle (circle O A D white))
  (hBIE : [Line B I] and InCircle (circle O B E white))
  (hParallel : [Line l_I] and Through l_I I and Parallel l_I AB)
  (hTangent : Tangent l_C (circle O) C and Intersect l_C l_I = F) :
  is_collinear D E F :=
by sorry

end collinear_DE_F_l206_206338


namespace max_area_cyclic_polygon_l206_206119

-- Definitions for the conditions
def positive_real (x : ℝ) : Prop := x > 0

def side_lengths_satisfy_triangle_inequality (sides : list ℝ) : Prop :=
  ∀ i < sides.length, 2 * sides.nth_le i sorry < sides.sum

-- Main theorem statement
theorem max_area_cyclic_polygon (sides : list ℝ)
  (h₀ : ∀ x ∈ sides, positive_real x) 
  (h₁ : side_lengths_satisfy_triangle_inequality sides) :
  ∃ (polygon : set (set ℝ)),
    (∀ x ∈ polygon, positive_real x) ∧ 
    side_lengths_satisfy_triangle_inequality sides ∧ 
    is_cyclic polygon ∧ 
    maximum_area polygon := 
sorry

end max_area_cyclic_polygon_l206_206119


namespace perfect_squares_lt_10_pow_7_and_multiple_of_36_l206_206967

theorem perfect_squares_lt_10_pow_7_and_multiple_of_36 :
  ∃ (n : ℕ), card { m : ℕ | m > 0 ∧ m ^ 2 < 10^7 ∧ 36 ∣ m ^ 2 } = 87 :=
by
  sorry

end perfect_squares_lt_10_pow_7_and_multiple_of_36_l206_206967


namespace cannot_divide_660_stones_into_31_heaps_l206_206269

theorem cannot_divide_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i, 0 < heaps i ∧ heaps i < 2 * heaps (succ i % 31)) :=
sorry

end cannot_divide_660_stones_into_31_heaps_l206_206269


namespace complex_expression_distinct_values_l206_206825

theorem complex_expression_distinct_values : 
  let i := Complex.I
  ∃ T_values : Finset ℝ, ∀ n : ℤ, 
    let T := i^(2*n) + i^(-2*n) + Real.cos (n * π)
    T ∈ T_values ∧ T_values.card = 2 :=
by
  /- Insert the detailed proof here -/
  sorry

end complex_expression_distinct_values_l206_206825


namespace num_satisfying_numbers_l206_206876

def digit_sum (n : ℕ) : ℕ :=
  n.digitSum

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def satisfies_condition (a : ℕ) : Prop :=
  digit_sum a = digit_sum (2 * a)

theorem num_satisfying_numbers : 
  ∃! (n : ℕ), (n = 80) ∧ (∀ a, is_three_digit a → satisfies_condition a) :=
sorry

end num_satisfying_numbers_l206_206876


namespace determine_cost_of_fabric_l206_206065

def cost_of_fabric (F : ℝ) : Prop :=
  let monday_earnings := 20 * F in
  let tuesday_earnings := 40 * F in
  let wednesday_earnings := 10 * F in
  let total_earnings := monday_earnings + tuesday_earnings + wednesday_earnings in
  total_earnings = 140 → F = 2

theorem determine_cost_of_fabric : ∃ F : ℝ, cost_of_fabric F :=
begin
  use 2,
  unfold cost_of_fabric,
  simp,
  intros,
  assumption,
end

end determine_cost_of_fabric_l206_206065


namespace number_of_squares_l206_206965

open Int

theorem number_of_squares (n : ℕ) (h : n < 10^7) : 
  (∃ n, 36 ∣ n ∧ n^2 < 10^07) ↔ (n = 87) :=
by sorry

end number_of_squares_l206_206965


namespace find_length_GH_l206_206188

variables {Point : Type} [InnerProductGeometry Point]
open InnerProductGeometry

structure Tetrahedron :=
(A B C D : Point)

def dihedralAngle (ABC BCD : {Tetrahedron} → ℝ) : Prop := sorry

def hasOrthocenter (H B : Point) (BCD : {Tetrahedron} → ℝ) : Prop := sorry

def centroid (G A B C : Point) : Prop := sorry

theorem find_length_GH (T : Tetrahedron)
    (angle : dihedralAngle T ABC T BCD = π/3)
    (orthocenter : hasOrthocenter H A T BCD)
    (centroid : centroid G A T B C)
    (length_AH : dist A H = 4)
    (equal_sides : dist A B = dist A C) :
    dist G H = (4 / 9) * sqrt 21 :=
begin
  sorry
end

end find_length_GH_l206_206188


namespace part1_part2_l206_206152

def universal_set (U : Set ℝ) := ∀ (x : ℝ), x ∈ U

def A (x : ℝ) : Prop := -2 + 3 * x - x^2 ≤ 0

def complement_U (U A : Set ℝ) (x : ℝ) := x ∈ U ∧ ¬A x

def B (a : ℝ) (x : ℝ) : Prop := a < x ∧ x < a + 2

theorem part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | complement_U Set.univ (A x) x} ∩ {x : ℝ | B a x} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

theorem part2 (hu : {x : ℝ | complement_U Set.univ (A x) x} ∩ {x : ℝ | B a x} = ∅) :
  a ≤ -1 ∨ 2 ≤ a :=
sorry

end part1_part2_l206_206152


namespace abs_simplification_l206_206547

theorem abs_simplification (x : ℝ) (h : x < -1) : 
  abs (x - real.sqrt ((x + 2)^2)) = -2*x - 2 :=
by
  sorry

end abs_simplification_l206_206547


namespace least_number_divisible_by_23_l206_206748

theorem least_number_divisible_by_23 (n d : ℕ) (h_n : n = 1053) (h_d : d = 23) : ∃ x : ℕ, (n + x) % d = 0 ∧ x = 5 := by
  sorry

end least_number_divisible_by_23_l206_206748


namespace book_exchange_ways_l206_206311

-- Conditions: 6 friends should exchange books without direct trading
-- Problem Statement: Prove that there are 160 ways for this exchange to happen
theorem book_exchange_ways :
  fintype.card (equiv.perm (fin 6)) = 160 :=
by
  sorry

end book_exchange_ways_l206_206311


namespace power_function_through_point_l206_206941

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_through_point (f : ℝ → ℝ)
  (h : ∀ x, f x = x ^ (1 / 2))
  (point : f 2 = Real.sqrt 2) :
  f = (λ x, x ^ (1 / 2)) :=
by sorry

end power_function_through_point_l206_206941


namespace find_x_l206_206532

theorem find_x (x : ℝ) : 
  let a := (x - 5 : ℝ, 3 : ℝ)
      b := (2 : ℝ, x : ℝ) in
      (a.1 * b.1 + a.2 * b.2 = 0) -> x = 2 :=
by 
  intro h,
  sorry

end find_x_l206_206532


namespace tangent_line_P_through_point_P_tangent_line_Q_through_point_Q_l206_206514

noncomputable def point_P := (2: ℝ, -Real.sqrt 5 : ℝ)
noncomputable def point_Q := (3: ℝ, 5 : ℝ)

def is_tangent (circle_eq line_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), circle_eq x y → line_eq x y

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

def tangent_line_P_eq (x y : ℝ) : Prop := 2 * x - Real.sqrt 5 * y = 9

def tangent_line_Q1_eq (x y : ℝ) : Prop := 3 * x + 5 * y = 30
def tangent_line_Q2_eq (x y : ℝ) : Prop := x = 3

theorem tangent_line_P_through_point_P :
  is_tangent circle_eq tangent_line_P_eq :=
sorry

theorem tangent_line_Q_through_point_Q :
  is_tangent circle_eq tangent_line_Q1_eq ∧ is_tangent circle_eq tangent_line_Q2_eq :=
sorry

end tangent_line_P_through_point_P_tangent_line_Q_through_point_Q_l206_206514


namespace minor_premise_of_syllogism_l206_206738

-- Definitions based on problem conditions
def f (x : ℝ) : ℝ := log (((1 - x) / (1 + x) : ℝ))

-- Lean statement for the proof problem
theorem minor_premise_of_syllogism : 
    (∀ x : ℝ, f (-x) = -f x) → (∃ (p : Prop), p ∧ (p ↔ (∀ x : ℝ, f (-x) = -f x))) :=
by
  sorry

end minor_premise_of_syllogism_l206_206738


namespace find_a_l206_206503

theorem find_a : 
  (∃ (a : ℝ), ∀ x : ℝ, f x = x^3 - a * x^2 + 2 → 
  (∃ m : ℝ, m = -1 ∧ deriv f 1 = m)) → a = 2 :=
begin
  sorry

end find_a_l206_206503


namespace probability_of_double_l206_206024

open Set

-- Condition: A smaller domino set is comprised only of integers 0 through 6.
def dominos_set := {(i, j) | 0 ≤ i ∧ i ≤ 6 ∧ 0 ≤ j ∧ j ≤ 6 ∧ i ≤ j}

-- Condition: Each integer pairs with every other within the same range.
def total_pairings := dominos_set.card

-- Number of doubles (pairs where i = j)
def double_dominos_set := {(i, i) | 0 ≤ i ∧ i ≤ 6}
def number_of_doubles := double_dominos_set.card

-- Question: What is the probability that a domino randomly selected is a double?
theorem probability_of_double : (number_of_doubles : ℚ) / total_pairings = 1 / 3 :=
by sorry

end probability_of_double_l206_206024


namespace isosceles_triangle_roots_l206_206576

theorem isosceles_triangle_roots (BC AB AC : ℝ) (m : ℝ) (h_triangle : isosceles_triangle ABC) 
  (h_BC : BC = 8)
  (h_roots : polynomial.is_root (polynomial.X^2 - polynomial.C 10 * polynomial.X + polynomial.C m) (AB) ∧ 
             polynomial.is_root (polynomial.X^2 - polynomial.C 10 * polynomial.X + polynomial.C m) (AC)) :
  truth :=
  ∃ AB AC, (AB = AC) ∨ (AB = 8) ∨ (AC = 8) := sorry

end isosceles_triangle_roots_l206_206576


namespace parabola_focus_sum_l206_206974

theorem parabola_focus_sum (n : ℕ) (P : Fin n → ℝ × ℝ) (x : Fin n → ℝ) (F : ℝ × ℝ)
  (h_parabola : ∀ i, P i = (x i, Real.sqrt (4 * (x i))))
  (h_focus : F = (1, 0))
  (h_sum : ∑ i, (x i) = 10) :
  ∑ i, Real.dist (P i) F = n + 10 := 
sorry

end parabola_focus_sum_l206_206974


namespace concert_parking_fee_l206_206306

theorem concert_parking_fee :
  let ticket_cost := 50 
  let processing_fee_percentage := 0.15 
  let entrance_fee_per_person := 5 
  let total_cost_concert := 135
  let num_people := 2 

  let total_ticket_cost := ticket_cost * num_people
  let processing_fee := total_ticket_cost * processing_fee_percentage
  let total_ticktet_cost_with_fee := total_ticket_cost + processing_fee
  let total_entrance_fee := entrance_fee_per_person * num_people
  let total_cost_without_parking := total_ticktet_cost_with_fee + total_entrance_fee
  total_cost_concert - total_cost_without_parking = 10 := by 
  sorry

end concert_parking_fee_l206_206306


namespace prob_three_primes_in_four_rolls_l206_206426

-- Define the basic properties
def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

def prob_prime : ℚ := 5 / 12
def prob_not_prime : ℚ := 7 / 12

def choose (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

lemma binom_coefficient : choose 4 3 = 4 := 
  by simp [choose, nat.factorial]

theorem prob_three_primes_in_four_rolls : 
  (4 * ((prob_prime ^ 3) * prob_not_prime) = (875 / 5184)) :=
  by tidy; sorry -- The actual proof is omitted

end prob_three_primes_in_four_rolls_l206_206426


namespace problem_l206_206162

theorem problem (a b : ℝ) (i : ℂ) (hi : i = complex.I) 
  (h : (a - 2*i) * i = b + i) : a + b = 3 := sorry

end problem_l206_206162


namespace no_division_660_stones_into_31_heaps_l206_206249

theorem no_division_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ), (∑ i, heaps i = 660) ∧ (∀ i j, heaps i < 2 * heaps j) :=
  sorry

end no_division_660_stones_into_31_heaps_l206_206249


namespace baker_cakes_total_l206_206423

-- Define the variables corresponding to the conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- State the theorem to prove that the total number of cakes made is 217
theorem baker_cakes_total : cakes_sold + cakes_left = 217 := 
by 
-- The proof is omitted according to the instructions
sorry

end baker_cakes_total_l206_206423


namespace rate_of_drawing_barbed_wire_is_correct_l206_206694

noncomputable def rate_of_drawing_barbed_wire (area cost: ℝ) (gate_width barbed_wire_extension: ℝ) : ℝ :=
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_barbed_wire := (perimeter - 2 * gate_width) + 4 * barbed_wire_extension
  cost / total_barbed_wire

theorem rate_of_drawing_barbed_wire_is_correct :
  rate_of_drawing_barbed_wire 3136 666 1 3 = 2.85 :=
by
  sorry

end rate_of_drawing_barbed_wire_is_correct_l206_206694


namespace distance_between_J_and_K_l206_206597

-- Definitions based on the conditions
def side_length : ℝ := 1
def height_of_cube : ℝ := side_length
def diagonal_of_square : ℝ := real.sqrt (side_length^2 + side_length^2)
def half_diagonal_of_square : ℝ := diagonal_of_square / 2
def height_of_pyramid : ℝ := real.sqrt (side_length^2 - (half_diagonal_of_square)^2)

-- The main statement to prove
theorem distance_between_J_and_K : 
  (height_of_cube + 2 * height_of_pyramid = 1 + real.sqrt 2) :=
by
  -- Placeholder for the proof
  sorry

end distance_between_J_and_K_l206_206597


namespace geoff_total_spending_l206_206473

def price_day1 : ℕ := 60
def pairs_day1 : ℕ := 2
def price_per_pair_day1 : ℕ := price_day1 / pairs_day1

def multiplier_day2 : ℕ := 3
def price_per_pair_day2 : ℕ := price_per_pair_day1 * 3 / 2
def discount_day2 : Real := 0.10
def cost_before_discount_day2 : ℕ := multiplier_day2 * price_per_pair_day2
def cost_after_discount_day2 : Real := cost_before_discount_day2 * (1 - discount_day2)

def multiplier_day3 : ℕ := 5
def price_per_pair_day3 : ℕ := price_per_pair_day1 * 2
def sales_tax_day3 : Real := 0.08
def cost_before_tax_day3 : ℕ := multiplier_day3 * price_per_pair_day3
def cost_after_tax_day3 : Real := cost_before_tax_day3 * (1 + sales_tax_day3)

def total_cost : Real := price_day1 + cost_after_discount_day2 + cost_after_tax_day3

theorem geoff_total_spending : total_cost = 505.50 := by
  sorry

end geoff_total_spending_l206_206473


namespace prob_A_drinks_two_qualified_prob_one_person_drinks_unqualified_l206_206988

-- Definition for the qualification rate condition
def qualification_rate := 0.8

-- (Ⅰ) The probability that A drinks two bottles of the X beverage and both are qualified
theorem prob_A_drinks_two_qualified 
  (p : ℚ) 
  (h : p = qualification_rate) :
  (p * p) = 0.64 :=
by sorry

-- (Ⅱ) The probability that A, B, and C each drink two bottles, and exactly one person drinks unqualified beverages (rounded to 0.01)
theorem prob_one_person_drinks_unqualified 
  (p : ℚ) 
  (h : p = qualification_rate) :
  (3 * (p * p) * (1 - p)) ≈ 0.44 :=
by sorry

end prob_A_drinks_two_qualified_prob_one_person_drinks_unqualified_l206_206988


namespace sequence_condition_l206_206222

noncomputable def f (A : List ℝ) : List ℝ :=
match A with
| [] => []
| [x] => [x]
| _ => List.zipWith (fun x y => (x + y) / 2) A (A.tail ++ [A.head])

noncomputable def f_iter (A : List ℝ) (k : ℕ) : List ℝ :=
Nat.recOn k A (fun _ Ak => f Ak)

theorem sequence_condition (A : List ℝ) (N : ℕ) (hN : 0 < N) (hA_int : ∀ i, A.nth i ∈ some ℤ) :
  (∀ k, ∀ i, f_iter A k .nth i ∈ some ℤ) ↔ 
    (∃ c : ℤ, (N % 2 = 1 ∧ ∀ i, f A .nth i = some c) ∨ 
               (N % 2 = 0 ∧ ∃ e f : ℤ, ∀ i, f A .nth i ∈ {some e, some f} ∧ e + f = 2 * c)) :=
sorry

end sequence_condition_l206_206222


namespace prove_f_neg_a_l206_206478

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 1

theorem prove_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -4 :=
by
  sorry

end prove_f_neg_a_l206_206478


namespace rhombus_area_of_square_l206_206542

theorem rhombus_area_of_square (h : ∀ (c : ℝ), c = 96) : ∃ (a : ℝ), a = 288 := 
by
  sorry

end rhombus_area_of_square_l206_206542


namespace find_hyperbola_eq_l206_206506

open Real

noncomputable def hyperbola_eq : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (c = sqrt (a^2 + b^2)) ∧ c = sqrt 5 ∧
  (∀ x : ℝ, y = 2 * x → y = 1 / 16 * x^2 + 1 = false)

noncomputable def problem_equiv : Prop :=
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (c = sqrt (a^2 + b^2)) ∧ c = sqrt 5 ∧
  (∃ x : ℝ, y = x / 2 → y = 1 / 16 * x^2 + 1 = true)) ∧ 
  (∃ a b : ℝ, a = 2 ∧ b = 1)

theorem find_hyperbola_eq : hyperbola_eq ↔ problem_equiv := sorry

end find_hyperbola_eq_l206_206506


namespace children_attended_l206_206809

theorem children_attended 
  (x y : ℕ) 
  (h₁ : x + y = 280) 
  (h₂ : 0.60 * x + 0.25 * y = 140) : 
  y = 80 := 
by
  sorry

end children_attended_l206_206809


namespace no_value_of_n_l206_206614

noncomputable def t1 (n : ℕ) : ℚ :=
3 * n * (n + 2)

noncomputable def t2 (n : ℕ) : ℚ :=
(3 * n^2 + 19 * n) / 2

theorem no_value_of_n (n : ℕ) (h : n > 0) : t1 n ≠ t2 n :=
by {
  sorry
}

end no_value_of_n_l206_206614


namespace compound_interest_example_l206_206760

/-- 
Theorem: Given the conditions on principal amount, interest rate, compounding frequency, 
and investment time, the amount in the savings account after one year is $882.
-/
theorem compound_interest_example 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ)
  (hP : P = 800) (hr : r = 0.10) (hn : n = 2) (ht : t = 1) :
  P * (1 + r / n)^(n * t) = 882 :=
by
  rw [hP, hr, hn, ht]
  simp
  norm_num
  sorry

end compound_interest_example_l206_206760


namespace problem1_problem2_l206_206524

noncomputable def setA (x : ℝ) : ℝ := real.sqrt(3 - 2 * x)
def setB (m : ℝ) : set ℝ := { x | (1 - m ≤ x) ∧ (x ≤ m + 1) }

theorem problem1 (m : ℝ) (h : m = 2) : 
  (∀ y, (y ∈ {y | ∃ x, setA x = y ∧ -13 / 2 ≤ x ∧ x ≤ 3 / 2}) → (y ∈ {x | (1 - 2 ≤ x ∧ x ≤ 2 + 1)})) → 
  ∃! z, z = 0 ∨ z = 3 := 
sorry

theorem problem2 (m : ℝ) : 
  (∀ x, (x ∈ { x | (1 - m ≤ x) ∧ (x ≤ m + 1) }) → (∃ y, y ∈ {y | setA y = x ∧ -13 / 2 ≤ y ∧ y ≤ 3 / 2})) → 
  m ≤ 1 := 
sorry

end problem1_problem2_l206_206524


namespace ratio_proof_l206_206543

theorem ratio_proof (a b c d : ℝ) (h1 : a / b = 20) (h2 : c / b = 5) (h3 : c / d = 1 / 8) : 
  a / d = 1 / 2 :=
by
  sorry

end ratio_proof_l206_206543


namespace polynomial_expansion_sum_l206_206224

theorem polynomial_expansion_sum (a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a_6 + a_5 + a_4 + a_3 + a_2 + a_1 + a = 64 :=
by
  -- Proof is not needed, placeholder here.
  sorry

end polynomial_expansion_sum_l206_206224


namespace total_number_of_employees_l206_206788
  
def part_time_employees : ℕ := 2041
def full_time_employees : ℕ := 63093
def total_employees : ℕ := part_time_employees + full_time_employees

theorem total_number_of_employees : total_employees = 65134 := by
  sorry

end total_number_of_employees_l206_206788


namespace math_problem_l206_206114

noncomputable def ellipse_standard_equation (a b : ℝ) : Prop :=
  a = 2 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def constant_slope_sum (T R S : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  T = (4, 0) ∧ l (1, 0) ∧ 
  (∀ TR TS : ℝ, (TR = (R.2 / (R.1 - 4)) ∧ TS = (S.2 / (S.1 - 4)) ∧ 
  (TR + TS = 0)))

theorem math_problem 
  {a b : ℝ} {T R S : ℝ × ℝ} {l : ℝ × ℝ → Prop} : 
  ellipse_standard_equation a b ∧ constant_slope_sum T R S l :=
by
  sorry

end math_problem_l206_206114


namespace parallel_line_through_P_perpendicular_line_through_P_l206_206485

-- Define point P
def P := (-4, 2)

-- Define line l
def l (x y : ℝ) := 3 * x - 2 * y - 7 = 0

-- Define the equation of the line parallel to l that passes through P
def parallel_line (x y : ℝ) := 3 * x - 2 * y + 16 = 0

-- Define the equation of the line perpendicular to l that passes through P
def perpendicular_line (x y : ℝ) := 2 * x + 3 * y + 2 = 0

-- Theorem 1: Prove that parallel_line is the equation of the line passing through P and parallel to l
theorem parallel_line_through_P :
  ∀ (x y : ℝ), 
    (parallel_line x y → x = -4 ∧ y = 2) :=
sorry

-- Theorem 2: Prove that perpendicular_line is the equation of the line passing through P and perpendicular to l
theorem perpendicular_line_through_P :
  ∀ (x y : ℝ), 
    (perpendicular_line x y → x = -4 ∧ y = 2) :=
sorry

end parallel_line_through_P_perpendicular_line_through_P_l206_206485


namespace circle_eq_of_diameter_l206_206529

theorem circle_eq_of_diameter (A B : ℝ × ℝ) (hA : A = (2, 0)) (hB : B = (0, 2)) :
  ∃ (c : ℝ × ℝ) (r : ℝ), (c = (1, 1) ∧ r = real.sqrt 2 ∧ ∀ x y : ℝ, (x - c.1)^2 + (y - c.2)^2 = r^2 → (x - 1)^2 + (y - 1)^2 = 2) :=
sorry

end circle_eq_of_diameter_l206_206529


namespace base_of_first_term_l206_206582

-- Define the necessary conditions
def equation (x s : ℝ) : Prop :=
  x^16 * 25^s = 5 * 10^16

-- The proof goal
theorem base_of_first_term (x s : ℝ) (h : equation x s) : x = 2 / 5 :=
by
  sorry

end base_of_first_term_l206_206582


namespace alpha_not_rational_l206_206120

open Real

theorem alpha_not_rational (cos_alpha : ℝ) (h : cos_alpha = 1/3) : 
  ¬ ∃ (m n : ℤ), (360 * n : ℝ) ≠ 0 ∧ α = m / n :=
begin
  sorry -- Proof goes here
end

end alpha_not_rational_l206_206120


namespace hexagon_angle_sum_l206_206535

theorem hexagon_angle_sum 
  (angles : Fin 6 → ℝ)
  (h_sum : ∑ i, angles i = 720)
  (h_angles : angles 1 = 108 ∧ angles 2 = 130 ∧ angles 3 = 142 ∧ angles 4 = 105 ∧ angles 5 = 120) :
  angles 0 = 115 :=
by {
  -- Use the sum of the angles and known values to derive the measure of angle Q
  sorry
}

end hexagon_angle_sum_l206_206535


namespace cannot_divide_660_stones_into_31_heaps_l206_206273

theorem cannot_divide_660_stones_into_31_heaps :
  ¬ ∃ (heaps : Fin 31 → ℕ),
    (∑ i, heaps i = 660) ∧
    (∀ i, 0 < heaps i ∧ heaps i < 2 * heaps (succ i % 31)) :=
sorry

end cannot_divide_660_stones_into_31_heaps_l206_206273


namespace transformed_function_expression_l206_206129

def transformations_stretch_shift (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (f (x / 3 + π / 12))

def f_orig (x : ℝ) : ℝ := Real.sin x

def f_transformed (x : ℝ) : ℝ := Real.sin (x / 3 + π / 12)

theorem transformed_function_expression :
  ∀ x, transformations_stretch_shift f_orig x = f_transformed x :=
by
  sorry

end transformed_function_expression_l206_206129


namespace ninety_percent_greater_than_eighty_percent_l206_206970

-- Define the constants involved in the problem
def ninety_percent (n : ℕ) : ℝ := 0.90 * n
def eighty_percent (n : ℕ) : ℝ := 0.80 * n

-- Define the problem statement
theorem ninety_percent_greater_than_eighty_percent :
  ninety_percent 40 - eighty_percent 30 = 12 :=
by
  sorry

end ninety_percent_greater_than_eighty_percent_l206_206970


namespace fence_cost_l206_206445

variable (lengthA lengthB lengthC lengthD : ℕ)
variable (costA costB costC costD : ℕ)

theorem fence_cost (hA : lengthA = 8) (hB : lengthB = 5) (hC : lengthC = 6) (hD : lengthD = 7)
  (cost_per_foot_A : costA = 58) (cost_per_foot_B : costB = 62) 
  (cost_per_foot_C : costC = 64) (cost_per_foot_D : costD = 68) :
  lengthA * costA + lengthB * costB + lengthC * costC + lengthD * costD = 1634 := by
  rw [hA, hB, hC, hD, cost_per_foot_A, cost_per_foot_B, cost_per_foot_C, cost_per_foot_D]
  sorry

end fence_cost_l206_206445


namespace selection_of_representatives_l206_206004

theorem selection_of_representatives :
  ∃ (group : Finset ℕ) (female_students male_students : Finset ℕ),
    female_students.card = 4 ∧
    male_students.card = 6 ∧
    female_students ∪ male_students = group ∧
    disjoint female_students male_students ∧
    ∑ (f in (finset.range 4)), ∑ (m in (finset.range 7)),
        if f + m = 3 then (f.choose 1) * (m.choose 2) + (f.choose 2) * (m.choose 1) + (f.choose 3) else 0 = 100 :=
sorry

end selection_of_representatives_l206_206004


namespace find_g_of_condition_l206_206329

theorem find_g_of_condition (g : ℝ → ℝ)
  (h : ∀ x : ℝ, g(x) + 2 * g(2 - x) = 4 * x^2 + 1) :
  g 4 = -31 / 3 :=
sorry

end find_g_of_condition_l206_206329


namespace thousands_digit_is_0_or_5_l206_206010

theorem thousands_digit_is_0_or_5 (n t : ℕ) (h₁ : n > 1000000) (h₂ : n % 40 = t) (h₃ : n % 625 = t) : 
  ((n / 1000) % 10 = 0) ∨ ((n / 1000) % 10 = 5) :=
sorry

end thousands_digit_is_0_or_5_l206_206010


namespace convex_quadrilateral_parallel_sides_l206_206486

-- Define a regular 18-gon as a set of vertices
def regular_18_gon := { vertices : Finset Point // vertices.card = 18 ∧ is_regular_polygon vertices }

-- Number of sets of four vertices that form convex quadrilaterals with at least one pair of parallel sides
theorem convex_quadrilateral_parallel_sides (M : regular_18_gon) :
  ∃ (N : ℕ), N = 540 :=
by
  sorry

end convex_quadrilateral_parallel_sides_l206_206486


namespace different_radii_no_four_tangents_l206_206571

open Set Real

-- Definitions and conditions
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (radius_pos : radius > 0)

def commonTangents (c1 c2 : Circle) : ℕ :=
  if c1.radius ≠ c2.radius then
    -- Different radii
    if ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > (c1.radius + c2.radius)^2) then 2 -- Separate
    else if ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 < (c1.radius - c2.radius)^2) then 0 -- One inside another
    else if ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2) then 3 -- Externally tangent
    else 2 -- Intersecting at two points
  else 4 -- Assume 4 if they have the same radius (not applicable here)

theorem different_radii_no_four_tangents (c1 c2 : Circle) :
  c1.radius ≠ c2.radius → commonTangents c1 c2 ≠ 4 :=
by
  intros h
  unfold commonTangents
  split_ifs
  -- Conditions about the circle configurations are handled here
  assumption
  -- Proof steps are omitted
  sorry

end different_radii_no_four_tangents_l206_206571


namespace sum_of_angles_subtended_by_arcs_l206_206296

theorem sum_of_angles_subtended_by_arcs
  (A B X Y C : Type)
  (arc_AX arc_XC : ℝ)
  (h1 : arc_AX = 58)
  (h2 : arc_XC = 62)
  (R S : ℝ)
  (hR : R = arc_AX / 2)
  (hS : S = arc_XC / 2) :
  R + S = 60 :=
by
  rw [hR, hS, h1, h2]
  norm_num

end sum_of_angles_subtended_by_arcs_l206_206296


namespace not_divide_660_into_31_not_divide_660_into_31_l206_206235

theorem not_divide_660_into_31 (H : ∀ (A B : ℕ), A ∈ S ∧ B ∈ S → A < 2 * B) : 
  false :=
begin
  -- math proof steps should go here
  sorry
end

namespace my_proof

def similar (x y : ℕ) : Prop := x < 2 * y

theorem not_divide_660_into_31 : ¬ ∃ (S : set ℕ), (∀ x ∈ S, x < 2 * y) ∧ (∑ x in S, x = 660) is ∧ |S| = 31 :=
begin
  sorry
end

end my_proof

end not_divide_660_into_31_not_divide_660_into_31_l206_206235


namespace surface_area_of_each_smaller_cube_l206_206562

theorem surface_area_of_each_smaller_cube
  (L : ℝ) (l : ℝ)
  (h1 : 6 * L^2 = 600)
  (h2 : 125 * l^3 = L^3) :
  6 * l^2 = 24 := by
  sorry

end surface_area_of_each_smaller_cube_l206_206562


namespace tan_of_cos_first_quadrant_l206_206914

-- Define the angle α in the first quadrant and its cosine value
variable (α : ℝ) (h1 : 0 < α ∧ α < π/2) (hcos : Real.cos α = 2 / 3)

-- State the theorem
theorem tan_of_cos_first_quadrant : Real.tan α = Real.sqrt 5 / 2 := 
by
  sorry

end tan_of_cos_first_quadrant_l206_206914


namespace prove_problem_statement_l206_206050

noncomputable def problem_statement : Prop :=
  let ε1 := Complex.exp (2 * Real.pi * Complex.I / 7) in
  (ε1^7 = 1) ∧
  (2 * Complex.cos (2 * Real.pi / 7) = ε1 + ε1⁻¹) ∧
  (2 * Complex.cos (4 * Real.pi / 7) = ε1^2 + ε1⁻¹^2) ∧
  (2 * Complex.cos (6 * Real.pi / 7) = ε1^3 + ε1⁻¹^3) ∧
  ((1 / (2 * Complex.cos (2 * Real.pi / 7)) +
    1 / (2 * Complex.cos (4 * Real.pi / 7)) +
    1 / (2 * Complex.cos (6 * Real.pi / 7))) = -2)

theorem prove_problem_statement : problem_statement :=
  by
  sorry

end prove_problem_statement_l206_206050


namespace sum_of_angles_l206_206334

open Complex

theorem sum_of_angles :
  let z : ℂ := -(1/2) - (Real.sqrt 3)/(2:ℂ) * Complex.I
  let roots := λ φ, Complex.ofRealAngle φ
  let φs := [60, 120, 180, 240, 300, 360] : List Real
  (∑ i in φs.indexes, φs.nthLe i H) = 1080 :=
by
  sorry

end sum_of_angles_l206_206334


namespace num_of_possible_values_b_l206_206683

-- Define the positive divisors set of 24
def divisors_of_24 := {n : ℕ | n > 0 ∧ 24 % n = 0}

-- Define the subset where 4 is a factor of each element
def divisors_of_24_div_by_4 := {b : ℕ | b ∈ divisors_of_24 ∧ b % 4 = 0}

-- Prove the number of positive values b where 4 is a factor of b and b is a divisor of 24 is 4
theorem num_of_possible_values_b : finset.card (finset.filter (λ b, b % 4 = 0) (finset.filter (λ n, 24 % n = 0) (finset.range 25))) = 4 :=
by
  sorry

end num_of_possible_values_b_l206_206683


namespace two_digit_diff_l206_206753

theorem two_digit_diff (s : Finset ℕ) (h : s = {1, 4, 7, 9}) :
  (let largest := max (s.erase 9).max' (by simp [Finset.erase_ne_of_mem 9 (by norm_num)]) * 10 + 9,
       smallest := min (s.erase 1).min' (by simp [Finset.erase_ne_of_mem 1 (by norm_num)]) * 10 + 1 in
   largest - smallest = 83) :=
by sorry

end two_digit_diff_l206_206753


namespace select_ways_l206_206984

-- Define the problem in Lean 4
theorem select_ways (a1 a2 a3 : ℕ) :
  a1 < a2 → a2 < a3 → a2 - a1 ≥ 3 → a3 - a2 ≥ 3 →
  a1 ∈ finset.range 15 → a2 ∈ finset.range 15 → a3 ∈ finset.range 15 →
  (finset.card {x : finset (finset ℕ) | ∃ (a1 a2 a3 : ℕ), 
    a1 < a2 ∧ a2 < a3 ∧ a2 - a1 ≥ 3 ∧ a3 - a2 ≥ 3 ∧
    a1 ∈ finset.range 15 ∧ a2 ∈ finset.range 15 ∧ a3 ∈ finset.range 15 = 120}) := 
sorry

end select_ways_l206_206984


namespace initial_quantity_liquid_A_l206_206005

theorem initial_quantity_liquid_A (x : ℝ) :
  let seven_ninths := 7 / 9
  let two_ninths := 2 / 9
  let initial_quantity_A := 7 * x
  let initial_quantity_B := 2 * x
  let initial_quantity := initial_quantity_A + initial_quantity_B
  let removed_quantity := 36
  let liquid_A_removed := seven_ninths * removed_quantity
  let liquid_B_removed := two_ninths * removed_quantity
  let remaining_liquid_A := initial_quantity_A - liquid_A_removed
  let remaining_liquid_B := initial_quantity_B - liquid_B_removed + removed_quantity
  let new_ratio := remaining_liquid_A / remaining_liquid_B = 5 / 8
  7 * x = 55.39 := by
calc
  initial_quantity_A = 55.39 : sorry

end initial_quantity_liquid_A_l206_206005


namespace ellipse_and_tangent_existence_l206_206502

-- Definitions of the given conditions
def is_ellipse_center_origin (C : ℝ × ℝ → Prop) := ∀ (x y : ℝ), C (0, 0)
def is_focus_on_x_axis (C : ℝ × ℝ → Prop) (f : ℝ) := ∃ (c : ℝ), f = c ∧ C (c, 0)
def eccentricity (e : ℝ) := e = 1 / 2
def is_vertex_of_parabola (v : ℝ × ℝ) := v = (0, -√3)
def parabola (p : ℝ × ℝ → Prop) := p = λ (x y : ℝ), x^2 + 4*√3*y = 0

-- Standard equation of the ellipse
def standard_equation_ellipse (C : ℝ × ℝ → Prop) := ∀ (x y : ℝ), C (x, y) ↔ (x^2 / 4 + y^2 / 3 = 1)

-- Equation of the tangent line and coordinates of the tangent point
def tangent_line_and_point (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) :=
  ∃ (k : ℝ) (x y : ℝ), k = -1 / 2 ∧ l = λ (x y : ℝ), y = -1 / 2 * (x - 2) + 1 ∧ (x, y) = (1, 3 / 2)

-- Main theorem statement
theorem ellipse_and_tangent_existence :
  ∃ (C l: ℝ × ℝ → Prop), is_ellipse_center_origin C ∧
    is_focus_on_x_axis C 1 ∧
    eccentricity (1 / 2) ∧
    is_vertex_of_parabola (0, -√3) ∧
    parabola (λ (x y : ℝ), x^2 = -4*√3*y) ∧
    standard_equation_ellipse C ∧
    tangent_line_and_point l C :=
by sorry

end ellipse_and_tangent_existence_l206_206502


namespace number_of_integers_between_cubes_l206_206962

theorem number_of_integers_between_cubes : 
  let x : ℝ := (9.4)^3
  let y : ℝ := (9.5)^3
  let a : ℕ := ⌈x⌉.to_nat
  let b : ℕ := ⌊y⌋.to_nat
  b - a + 1 = 27 := by
  -- Definitions from conditions
  let x := (9.4)^3
  let y := (9.5)^3
  let a := ⌈x⌉.to_nat
  let b := ⌊y⌋.to_nat
  -- No proof required here
  sorry

end number_of_integers_between_cubes_l206_206962


namespace problem1_solution_problem2_solution_l206_206519

def f (c a b : ℝ) (x : ℝ) : ℝ := |(c * x + a)| + |(c * x - b)|
def g (c : ℝ) (x : ℝ) : ℝ := |(x - 2)| + c

noncomputable def sol_set_eq1 := {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 2}
noncomputable def range_a_eq2 := {a : ℝ | a ≤ -2 ∨ a ≥ 0}

-- Problem (1)
theorem problem1_solution : ∀ (x : ℝ), f 2 1 3 x - 4 = 0 ↔ x ∈ sol_set_eq1 := 
by
  intro x
  sorry -- Proof to be filled in

-- Problem (2)
theorem problem2_solution : 
  ∀ x_1 : ℝ, ∃ x_2 : ℝ, g 1 x_2 = f 1 0 1 x_1 ↔ a ∈ range_a_eq2 :=
by
  intro x_1
  sorry -- Proof to be filled in

end problem1_solution_problem2_solution_l206_206519


namespace train_pass_time_l206_206409

-- Definitions
def length_of_train : ℝ := 200 -- in meters
def speed_of_train_kmh : ℝ := 120 -- in kilometers per hour
def speed_of_man_kmh : ℝ := 15 -- in kilometers per hour

-- Conversion functions
def kmh_to_mps (kmh : ℝ) : ℝ := (kmh * 1000) / 3600

-- Speeds in meters per second
def speed_of_train_mps : ℝ := kmh_to_mps speed_of_train_kmh
def speed_of_man_mps : ℝ := kmh_to_mps speed_of_man_kmh

-- Relative speed (since they are running in opposite directions)
def relative_speed_mps : ℝ := speed_of_train_mps + speed_of_man_mps

-- Time to pass the man
def time_to_pass : ℝ := length_of_train / relative_speed_mps

-- The main theorem to prove
theorem train_pass_time : abs (time_to_pass - 5.33) < 0.01 := by
  sorry

end train_pass_time_l206_206409


namespace friend_decks_l206_206739

noncomputable def cost_per_deck : ℕ := 8
noncomputable def victor_decks : ℕ := 6
noncomputable def total_amount_spent : ℕ := 64

theorem friend_decks :
  ∃ x : ℕ, (victor_decks * cost_per_deck) + (x * cost_per_deck) = total_amount_spent ∧ x = 2 :=
by
  sorry

end friend_decks_l206_206739


namespace vertex_angle_new_figure_l206_206017

theorem vertex_angle_new_figure (n : ℕ) (h₀ : n > 4) (h₁ : ∃ (polygon : Type) (sides : polygon → ℝ), sides.count = n) :
  ∀ (new_figure : Type), angle_at_each_vertex new_figure = 360 - 1080 / n :=
by
  sorry

end vertex_angle_new_figure_l206_206017


namespace total_bottles_consumed_from_initial_initial_bottles_from_total_consumed_l206_206093

-- Definitions based on conditions
def initial_bottles : ℕ := 1999
def consumed_bottles_final : ℕ := 2665
def total_consumed : ℕ := 3126
def originally_bought : ℕ := 2345

-- Recycling process: 4 empty bottles for 1 new bottle
def recycle_bottles (empty_bottles : ℕ) : ℕ := empty_bottles / 4

-- First theorem: Calculate total bottles consumed starting from 1999 bottles
theorem total_bottles_consumed_from_initial :
  (let rec consume (empty_bottles remaining : ℕ) : ℕ :=
    let new_bottles := recycle_bottles empty_bottles,
    if new_bottles = 0 then remaining + empty_bottles
    else consume (empty_bottles % 4 + new_bottles) (remaining + new_bottles)
   in consume initial_bottles initial_bottles) = consumed_bottles_final :=
by sorry

-- Second theorem: Calculate the initial number of bottles if total consumed is 3126
theorem initial_bottles_from_total_consumed :
  (∃ n : ℕ, let rec consume (empty_bottles remaining : ℕ) : ℕ :=
    let new_bottles := recycle_bottles empty_bottles,
    if new_bottles = 0 then remaining + empty_bottles
    else consume (empty_bottles % 4 + new_bottles) (remaining + new_bottles)
   in consume n n = total_consumed) :=
by sorry

end total_bottles_consumed_from_initial_initial_bottles_from_total_consumed_l206_206093


namespace student_solved_correctly_l206_206407

-- Problem conditions as definitions
def sums_attempted : Nat := 96

def sums_correct (x : Nat) : Prop :=
  let sums_wrong := 3 * x
  x + sums_wrong = sums_attempted

-- Lean statement to prove
theorem student_solved_correctly (x : Nat) (h : sums_correct x) : x = 24 :=
  sorry

end student_solved_correctly_l206_206407


namespace polynomial_factor_common_l206_206972

theorem polynomial_factor_common :
  ∃ (c : ℝ), (4 * 4^3 - 16 * 4^2 + c * 4 - 20 = 0) →  ∀ (p : ℝ[X]), (p = 4 * X^3 - 16 * X^2 + c * X - 20) → (X - 4) ∣ p → (4 * X^2 + 5) ∣ p :=
by
  sorry

end polynomial_factor_common_l206_206972


namespace sphere_surface_area_l206_206112

open Real

variables (O A B C : Point) -- Declare O, A, B, C as points
variables (S : Sphere) -- Declare S as a sphere

-- Conditions
axiom angle_BOC : ∠ B O C = π / 2
axiom OA_perpendicular_BOC_plane : ∀ (P: Point), P ∈ {B, O, C} \to perpendicular OA (Line.from_points B C)
axiom AB_length : dist A B = sqrt 10
axiom BC_length : dist B C = sqrt 13
axiom AC_length : dist A C = sqrt 5
axiom points_on_sphere : ∀ {P : Point}, P ∈ {O, A, B, C} → S.contains P

-- Question to prove the surface area of sphere S
theorem sphere_surface_area : 4 * π * (radius S)^2 = 14 * π :=
by sorry

end sphere_surface_area_l206_206112


namespace dense_sets_count_l206_206792

open Finset

def is_dense (A : Finset ℕ) : Prop :=
  (A ⊆ range 50) ∧ (card A > 40) ∧
  (∀ x ∈ range 45, range' x 6 \subseteq A → false)

theorem dense_sets_count :
  {A : Finset ℕ | is_dense A}.card = 495 :=
sorry

end dense_sets_count_l206_206792


namespace solveProblem_l206_206939

section ProofProblem

-- Define the curve G based on given function
def curveG (x y : ℝ) := y = real.sqrt(2 * x ^ 2 + 2)

-- Define the equation of curve G (hyperbola) based on the given problem.
def hyperbolaG (x y : ℝ) := (y^2 / 2) - x^2 = 1 ∧ y ≥ real.sqrt 2

-- The coordinates of the focus
def focusF := (0, real.sqrt 3)

-- Define lines passing through the focus F intersecting the curve G
variable {F : ℝ × ℝ} (x y k : ℝ)
def lineL1 := y = k * x + F.2
def lineL2 := y = -(1 / k) * x + F.2

-- Define the quadrilateral area conditions related to curves intersecting at points and their vectors
def areaMinimal (k A B C D S : ℝ) := 
  (curveG A (k * A + focusF.2) ∧ curveG C (-k * C + focusF.2) ∧
   curveG B (-1/k * B + focusF.2) ∧ curveG D (1/k * D + focusF.2) ∧ 
   (k^2 > 0) ∧ (k^2 < 2) ∧ (over.vec (A, 0) C = D ∧ over.vec (B, 0) ≠ C) 
   → (S = 16))

-- The theorem to be proven
theorem solveProblem (x y A B C D S : ℝ) : focusF = (0, real.sqrt 3) ∧
  (∀ x y, curveG x y → hyperbolaG x y) ∧
  (∃ k, areaMinimal k A B C D S) :=
sorry

end ProofProblem

end solveProblem_l206_206939


namespace sample_size_is_correct_l206_206391

variable (A B C : ℝ)
variable (rA rB rC : ℝ)
variable (n : ℝ)
variable (units_of_A_in_sample : ℝ)

-- Conditions as definitions
def product_ratios := (rA = 2) ∧ (rB = 3) ∧ (rC = 5) ∧ (rA + rB + rC = 10)
def sample_contains_units_of_A := units_of_A_in_sample = 16

-- Theorem to prove
theorem sample_size_is_correct (h1 : product_ratios) (h2 : sample_contains_units_of_A) : n = 80 :=
sorry

end sample_size_is_correct_l206_206391


namespace perfect_squares_lt_10_pow_7_and_multiple_of_36_l206_206968

theorem perfect_squares_lt_10_pow_7_and_multiple_of_36 :
  ∃ (n : ℕ), card { m : ℕ | m > 0 ∧ m ^ 2 < 10^7 ∧ 36 ∣ m ^ 2 } = 87 :=
by
  sorry

end perfect_squares_lt_10_pow_7_and_multiple_of_36_l206_206968


namespace nonzero_terms_count_l206_206537

-- Define the given polynomials
def P1 : ℚ[X] := X^2 + 2
def P2 : ℚ[X] := 3 * X^3 + 2 * X^2 + 4
def P3 : ℚ[X] := X^4 + X^3 - 3 * X

-- Define the expression we need to expand and simplify
def expression : ℚ[X] := P1 * P2 - 4 * P3

-- Statement to prove the number of nonzero terms in the expanded expression is 6
theorem nonzero_terms_count : (expression.coeffs.filter (λ c, c ≠ 0)).length = 6 := 
sorry

end nonzero_terms_count_l206_206537


namespace number_of_convex_numbers_l206_206801

theorem number_of_convex_numbers :
  (finset.filter 
    (λ (n : ℕ), let (a_1, rest) := n.div_mod 100 in
                let (a_2, a_3) := rest.div_mod 10 in
                100 ≤ n ∧ n ≤ 999 ∧ a_1 < a_2 ∧ a_2 > a_3) 
    (finset.range 900)).card = 240 := sorry

end number_of_convex_numbers_l206_206801


namespace fraction_of_gasoline_used_l206_206816

-- Define the conditions
def gasoline_per_mile := 1 / 30  -- Gallons per mile
def full_tank := 12  -- Gallons
def speed := 60  -- Miles per hour
def travel_time := 5  -- Hours

-- Total distance traveled
def distance := speed * travel_time  -- Miles

-- Gasoline used
def gasoline_used := distance * gasoline_per_mile  -- Gallons

-- Fraction of the full tank used
def fraction_used := gasoline_used / full_tank

-- The theorem to be proved
theorem fraction_of_gasoline_used :
  fraction_used = 5 / 6 :=
by sorry

end fraction_of_gasoline_used_l206_206816


namespace cost_price_for_one_meter_l206_206755

variable (meters_sold : Nat) (selling_price : Nat) (loss_per_meter : Nat) (total_cost_price : Nat)
variable (cost_price_per_meter : Rat)

theorem cost_price_for_one_meter (h1 : meters_sold = 200)
                                  (h2 : selling_price = 12000)
                                  (h3 : loss_per_meter = 12)
                                  (h4 : total_cost_price = selling_price + loss_per_meter * meters_sold)
                                  (h5 : cost_price_per_meter = total_cost_price / meters_sold) :
  cost_price_per_meter = 72 := by
  sorry

end cost_price_for_one_meter_l206_206755
