import Complex.Real
import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Exponent
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.Inverse
import Mathlib.Analysis.Convex.Hull.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Irrational
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.ConicSection.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.Basic
import Mathlib.Logic.Basic
import Mathlib.Tactic

namespace tan_of_triangle_eqn_l332_332987

noncomputable def sides (a b c : ℝ) := True
noncomputable def equation (a b c : ℝ) := 3 * a ^ 2 + 3 * b ^ 2 - 3 * c ^ 2 + 2 * a * b = 0

theorem tan_of_triangle_eqn
  (a b c : ℝ)
  (h_sides : sides a b c)
  (h_eq : equation a b c) :
  Real.tan (angle C) = -2 * Real.sqrt 2 :=
by
  sorry

end tan_of_triangle_eqn_l332_332987


namespace intersection_complement_l332_332466

open Set

variable (U : Set ℝ)
variable (M : Set ℝ)
variable (N : Set ℝ)

def setU : Set ℝ := univ
def setM : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def setN : Set ℝ := {x | x ≤ 0}

theorem intersection_complement :
  (M ∩ (U \ N) = {x | 0 < x ∧ x ≤ 1}) :=
by
  unfold setU setM setN
  ext x
  simp
  sorry

end intersection_complement_l332_332466


namespace sum_of_solutions_equation_l332_332939

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332939


namespace family_reunion_handshakes_l332_332351

theorem family_reunion_handshakes (married_couples : ℕ) (participants : ℕ) (allowed_handshakes : ℕ) (total_handshakes : ℕ) :
  married_couples = 8 →
  participants = married_couples * 2 →
  allowed_handshakes = participants - 1 - 1 - 6 →
  total_handshakes = (participants * allowed_handshakes) / 2 →
  total_handshakes = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end family_reunion_handshakes_l332_332351


namespace circle_distance_problem_l332_332654

theorem circle_distance_problem (r₁ r₂ : ℝ) (A B : Type*) [MetricSpace A] [MetricSpace B]
    (RA : MetricSpace.toRadius r₁ A) (RB : MetricSpace.toRadius r₂ B) :
    (∀ (d : ℝ), d = dist A B → (d = r₁ - r₂ ∨ d = r₁ + r₂ ∨ d > r₁ + r₂ ∨ d > r₁ - r₂))
    → false := by
  sorry

end circle_distance_problem_l332_332654


namespace clock_angle_3_40_l332_332675

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332675


namespace option_A_correct_l332_332994

open Set

variables {α β : Type} [AffinePlane α] [AffinePlane β]
variables (l : Line) (a b : Plane)

-- Assume the required conditions
def parallel_line_plane (l : Line) (p : Plane) : Prop :=
  ∃ m : Line, m ⊆ p ∧ l ∥ m

def perpendicular_line_plane (l : Line) (p : Plane) : Prop :=
  ∃ m : Line, m ⊆ p ∧ l ⊥ m

def perpendicular_plane_plane (p q : Plane) : Prop :=
  ∀ (m : Line), m ⊆ p → ∃ n : Line, n ⊆ q ∧ m ⊥ n

-- Mathematically equivalent proof problem
theorem option_A_correct (h1 : parallel_line_plane l a) (h2 : perpendicular_line_plane l b) :
  perpendicular_plane_plane a b := sorry

end option_A_correct_l332_332994


namespace first_hour_rain_l332_332523

variable (x : ℝ)
variable (rain_1st_hour : ℝ) (rain_2nd_hour : ℝ)
variable (total_rain : ℝ)

-- Define the conditions
def condition_1 (x rain_2nd_hour : ℝ) : Prop :=
  rain_2nd_hour = 2 * x + 7

def condition_2 (x rain_2nd_hour total_rain : ℝ) : Prop :=
  x + rain_2nd_hour = total_rain

-- Prove the amount of rain in the first hour
theorem first_hour_rain (h1 : condition_1 x rain_2nd_hour)
                         (h2 : condition_2 x rain_2nd_hour total_rain)
                         (total_rain_is_22 : total_rain = 22) :
  x = 5 :=
by
  -- Proof steps go here
  sorry

end first_hour_rain_l332_332523


namespace domain_of_f_l332_332368

def f (x : ℝ) : ℝ := Real.log (2 ^ x - 1)

theorem domain_of_f :
  ∀ x : ℝ, (0 < x) ↔ (∃ y : ℝ, y = f x) :=
by
  sorry

end domain_of_f_l332_332368


namespace sum_of_solutions_eqn_l332_332943

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332943


namespace math_problem_l332_332430

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem math_problem :
  (a + b = ab = c) →
  (c ≠ 0 → (2*a - 3*ab + 2*b) / (5*a + 7*ab + 5*b) = -1/12) →
  (a = 3 → b + c = 6) →
  (c ≠ 0 → (1 - a) * (1 - b) = 1/a + 1/b) →
  (c = 4 → a^2 + b^2 = 8) →
  (4) := by
  intros h1 h2 h3 h4 h5
  sorry

end math_problem_l332_332430


namespace total_cost_of_goods_l332_332206

theorem total_cost_of_goods :
  ∃ (M R F : ℝ),
    (10 * M = 24 * R) ∧
    (6 * F = 2 * R) ∧
    (F = 20.50) ∧
    (4 * M + 3 * R + 5 * F = 877.40) :=
by {
  sorry
}

end total_cost_of_goods_l332_332206


namespace solve_quadratic_eq_l332_332173

theorem solve_quadratic_eq (x : ℝ) : (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l332_332173


namespace sum_of_solutions_sum_of_all_solutions_l332_332907

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332907


namespace solution_interval_l332_332873

noncomputable def set_of_solutions : Set ℝ :=
  {x : ℝ | 4 * x - 3 < (x - 2) ^ 2 ∧ (x - 2) ^ 2 < 6 * x - 5}

theorem solution_interval :
  set_of_solutions = {x : ℝ | 7 < x ∧ x < 9} := by
  sorry

end solution_interval_l332_332873


namespace num_distinct_positions_R_l332_332586

-- Definitions of Points P and Q
def P : (ℝ × ℝ) := (-3, 0)
def Q : (ℝ × ℝ) := (3, 0)

-- Distance between P and Q
def PQ : ℝ := 6

-- Given area of the triangle PQR
def area_PQR : ℝ := 18

-- Prove that there are exactly 6 distinct positions for R such that the triangle PQR forms a right triangle with the given area
theorem num_distinct_positions_R : ∃ R : set (ℝ × ℝ), R.finite ∧ R.card = 6 ∧
  ∀ r ∈ R, ∃ x y : ℝ, (P = (-3, 0) → Q = (3, 0) → PQ = 6 → [P, Q, (x, y)].area = 18) ∧
  ((P, Q, (x, y)).forms_right_triangle) :=
sorry

end num_distinct_positions_R_l332_332586


namespace castor_chess_lost_to_AI_l332_332582

-- Conditions from part a
def total_players : ℕ := 120
def never_lost_proportion : ℚ := 2 / 5

-- Definitions derived from the conditions
def lost_proportion : ℚ := 1 - never_lost_proportion
def never_lost_players : ℕ := (never_lost_proportion * total_players).natAbs
def lost_players : ℕ := (lost_proportion * total_players).natAbs

-- Theorem to prove the correct answer
theorem castor_chess_lost_to_AI :
  lost_players = 72 :=
by
  sorry

end castor_chess_lost_to_AI_l332_332582


namespace day12_is_monday_l332_332091

-- Define the days of the week
inductive WeekDay
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open WeekDay

-- Define the problem using the conditions
def five_fridays_in_month (first_day : WeekDay) (last_day : WeekDay) : Prop :=
  first_day ≠ Friday ∧ last_day ≠ Friday ∧
  ((first_day = Monday ∨ first_day = Tuesday ∨ first_day = Wednesday ∨ first_day = Thursday ∨
    first_day = Saturday ∨ first_day = Sunday) ∧
  ∃ fridays : Finset ℕ,
  fridays.card = 5 ∧
  ∀ n ∈ fridays, (n % 7 = (5 - WeekDay.recOn first_day 6 0 1 2 3 4 5)) ∧
  fridays ⊆ Finset.range 31 ∧
  1 ∉ fridays ∧ (31 - Finset.max' fridays sorry) % 7 ≠ 0 )

-- Given the problem, prove that the 12th day is a Monday
theorem day12_is_monday (first_day last_day : WeekDay)
  (h : five_fridays_in_month first_day last_day) : 
  (12 % 7 + WeekDay.recOn first_day 6 0 1 2 3 4 5) % 7 = 0 :=
sorry

end day12_is_monday_l332_332091


namespace inverse_proposition_of_parallel_lines_l332_332223

theorem inverse_proposition_of_parallel_lines 
  (P : Prop) (Q : Prop) 
  (h : P ↔ Q) : 
  (Q ↔ P) :=
by 
  sorry

end inverse_proposition_of_parallel_lines_l332_332223


namespace smaller_angle_at_3_40_l332_332733

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332733


namespace angle_at_3_40_pm_is_130_degrees_l332_332663

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332663


namespace sum_of_roots_eq_seventeen_l332_332914

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332914


namespace sum_inequality_l332_332149

theorem sum_inequality
  (n : ℕ)
  (a : ℕ → ℕ)
  (h : ∀ i j : ℕ, i < j → a i < a j)
  (h_pos : ∀ i, 1 ≤ a i) :
  ∑ k in Finset.range n, (a k) / k^2 ≥ ∑ k in Finset.range n, 1 / k := sorry

end sum_inequality_l332_332149


namespace set_of_n_with_s_eq_5_l332_332396

def s (n : ℕ) := ∑ d in (Finset.divisors (n^2)), 1

theorem set_of_n_with_s_eq_5 : {n : ℕ | s n = 5} = 
  {k : ℕ | ∃ p : ℕ, p.Prime ∧ k = p^2} := 
begin
  sorry
end

end set_of_n_with_s_eq_5_l332_332396


namespace BM_eq_CM_l332_332370

theorem BM_eq_CM
  (A B C D P X Y M : Type)
  [Trapezoid ABCD]
  (h_diag_AC : Diagonal AC P)
  (h_diag_BD : Diagonal BD P)
  (h_circum_ABP : Circumcircle ABP X)
  (h_second_X : SecondIntersection X A D)
  (h_circum_CDP : Circumcircle CDP Y)
  (h_second_Y : SecondIntersection Y C D)
  (h_midpoint_M : Midpoint M X Y) :
  Distance B M = Distance C M := sorry

end BM_eq_CM_l332_332370


namespace largest_k_2520_l332_332828

def highest_power_of_prime_factor (n : ℕ) (p : ℕ) : ℕ :=
  ∑ i in finset.Ico 1 (n+1), n / p^i

theorem largest_k_2520 :
  let n := 2520
  let k_max := 629
  let factors := (2^3) * (3^2) * 5 * 7
  let power_2 := highest_power_of_prime_factor n 2
  let power_3 := highest_power_of_prime_factor n 3
  let power_5 := highest_power_of_prime_factor n 5
  let power_7 := highest_power_of_prime_factor n 7
  shows (2520^k_max) ∣ nat.factorial n :=
by
  let n := 2520
  let k_max := 629
  let power_2 := highest_power_of_prime_factor n 2
  let power_3 := highest_power_of_prime_factor n 3
  let power_5 := highest_power_of_prime_factor n 5
  let power_7 := highest_power_of_prime_factor n 7
  have power_2_value : power_2 = 2514 := by sorry
  have power_3_value : power_3 = 1258 := by sorry
  have power_5_value : power_5 = 628 := by sorry
  have power_7_value : power_7 = 419 := by sorry
  have key : k_max = 629 := by sorry
  have correct_division : (2520 ^ k_max) ∣ (nat.factorial n) := by sorry
  exact correct_division

end largest_k_2520_l332_332828


namespace cevian_product_one_l332_332520

-- Define the geometric setup
variables {A B C P A₁ B₁ C₁ : Point}
variable (triangleABC : PlaneTriangle A B C)
variable (pointP : PointOnPlane P)
variable (intA₁ : SegmentIntersection (LineThrough P A) (LineSeg B C))
variable (intB₁ : SegmentIntersection (LineThrough P B) (LineSeg A C))
variable (intC₁ : SegmentIntersection (LineThrough P C) (LineSeg A B))

-- Prove the required relation
theorem cevian_product_one (hA₁ : intA₁ = A₁) (hB₁ : intB₁ = B₁) (hC₁ : intC₁ = C₁) :
  (segmentRatio A C₁ (B C₁)) * (segmentRatio B A₁ (C A₁)) * (segmentRatio C B₁ (A B₁)) = -1 :=
sorry

end cevian_product_one_l332_332520


namespace minimum_blue_cells_l332_332554

def is_related (n : ℕ) (cell1 cell2 : ℕ × ℕ) : Prop := 
  (cell1.1 = cell2.1 ∨ cell1.2 = cell2.2) ∧ cell1 ≠ cell2

def at_least_two_related_blue (n : ℕ) (grid : ℕ × ℕ → Prop) : Prop := 
  ∀ cell : ℕ × ℕ, ∃ blue_cells : List (ℕ × ℕ), 
    (∀ c ∈ blue_cells, grid c) ∧
    blue_cells.length ≥ 2 ∧
    ∀ c ∈ blue_cells, is_related n cell c

theorem minimum_blue_cells {m : ℕ} (hm : m > 0) :
  ∃ grid : (ℕ × ℕ) → Prop,
    (at_least_two_related_blue (4 * m) grid) ∧
    (∑ i in Finset.range (4 * m), ∑ j in Finset.range (4 * m), if grid (i, j) then 1 else 0) = 6 * m := 
sorry

end minimum_blue_cells_l332_332554


namespace focus_of_ellipse_with_lesser_y_coordinate_l332_332355

theorem focus_of_ellipse_with_lesser_y_coordinate :
  let major_axis := ((1, -2), (7, -2)) in
  let minor_axis := ((4, 1), (4, -5)) in
  let center := (4, -2) in
  (major_axis.fst.1 - major_axis.snd.1)^2 + (minor_axis.fst.2 - minor_axis.snd.2)^2 = 0 →
  center = (4, -2) :=
by
  let major_axis := ((1, -2), (7, -2))
  let minor_axis := ((4, 1), (4, -5))
  let center := (4, -2)
  have h : (major_axis.fst.1 - major_axis.snd.1)^2 - (minor_axis.fst.2 - minor_axis.snd.2)^2 = 0,
  sorry
  exact (calc center : (4, -2) 
           ... = (4 - 0, -2 + 0) : by sorry)

end focus_of_ellipse_with_lesser_y_coordinate_l332_332355


namespace coefficient_x2_in_expansion_l332_332874

theorem coefficient_x2_in_expansion : 
  let T (n : ℕ) (r : ℕ) := (-3)^r * Nat.choose 6 r * x^r in
  T 6 2 = 135 :=
by
  sorry

end coefficient_x2_in_expansion_l332_332874


namespace sum_of_solutions_sum_of_all_solutions_l332_332901

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332901


namespace prime_factors_count_l332_332481

theorem prime_factors_count (n : ℕ) (h : n = 75) : (nat.factors n).to_finset.card = 2 :=
by
  rw h
  -- The proof part is omitted as instructed
  sorry

end prime_factors_count_l332_332481


namespace sin_lower_bound_lt_l332_332040

theorem sin_lower_bound_lt (a : ℝ) (h : ∃ x : ℝ, Real.sin x < a) : a > -1 :=
sorry

end sin_lower_bound_lt_l332_332040


namespace equivalent_proof_problem_l332_332022

open Set

variable (m : ℝ)

def M := {1, 6, 3}
def N := {x | x ^ 2 - 2 * x - 3 = 0}

theorem equivalent_proof_problem (M ∩ N = {3}) :
  (M ∪ N = {-1, 1, 3, 6} ∨ M ∪ N = {-1, 1, 3, 3 - 12 * I}) :=
by sorry

end equivalent_proof_problem_l332_332022


namespace circular_field_area_correct_l332_332607

open Real

noncomputable def area_of_circular_field_in_hectares (cost_per_meter total_cost : Real) : Real :=
  let circumference := total_cost / cost_per_meter
  let radius := circumference / (2 * pi)
  let area_square_meters := pi * radius ^ 2
  area_square_meters / 10000

theorem circular_field_area_correct :
  area_of_circular_field_in_hectares 4.60 6070.778380479544 = 13.8785098 :=
by
  sorry

end circular_field_area_correct_l332_332607


namespace new_solid_edges_l332_332834

-- Definitions based on conditions
def original_vertices : ℕ := 8
def original_edges : ℕ := 12
def new_edges_per_vertex : ℕ := 3
def number_of_vertices : ℕ := original_vertices

-- Conclusion to prove
theorem new_solid_edges : 
  (original_edges + new_edges_per_vertex * number_of_vertices) = 36 := 
by
  sorry

end new_solid_edges_l332_332834


namespace angle_terminal_side_eq_l332_332521

theorem angle_terminal_side_eq (α : ℝ) : 
  (α = -4 * Real.pi / 3 + 2 * Real.pi) → (0 ≤ α ∧ α < 2 * Real.pi) → α = 2 * Real.pi / 3 := 
by 
  sorry

end angle_terminal_side_eq_l332_332521


namespace alex_ate_jelly_beans_l332_332341

theorem alex_ate_jelly_beans :
  ∀ (x : ℕ), x = (36 - 3 * 10) → x = 6 :=
begin
  intros x h,
  rw h,
  norm_num,
end

end alex_ate_jelly_beans_l332_332341


namespace ice_cream_volume_l332_332617

-- Define the data given in the problem
def radius (r : ℝ) : Prop := r = 3
def height (h : ℝ) : Prop := h = 10

-- Define the formulas for volume
def volumeCone (r h : ℝ) := (1 / 3) * π * r^2 * h
def volumeHemisphere (r : ℝ) := (2 / 3) * π * r^3

-- Define the total volume as a sum of the cone's volume and the hemisphere's volume
def totalVolume (r h : ℝ) := volumeCone r h + volumeHemisphere r

-- The main theorem to prove
theorem ice_cream_volume : totalVolume 3 10 = 48 * π :=
by
  sorry

end ice_cream_volume_l332_332617


namespace average_check_l332_332609

variable (a b c d e f g x : ℕ)

def sum_natural (l : List ℕ) : ℕ := l.foldr (λ x y => x + y) 0

theorem average_check (h1 : a = 54) (h2 : b = 55) (h3 : c = 57) (h4 : d = 58) (h5 : e = 59) (h6 : f = 63) (h7 : g = 65) (h8 : x = 65) (avg : 60 * 8 = 480) :
    sum_natural [a, b, c, d, e, f, g, x] = 480 :=
by
  sorry

end average_check_l332_332609


namespace number_of_ways_to_place_balls_l332_332242

theorem number_of_ways_to_place_balls (balls : Fin 5) (boxes : Fin 3)
  (choose : ∀ {n k : ℕ}, n.choose k) 
  (permute : ∀ {n : ℕ}, n.factorial) :
  choose 5 4 * choose 4 2 * permute 3 = 180 :=
by
  sorry

end number_of_ways_to_place_balls_l332_332242


namespace shaded_area_fraction_l332_332587

-- Define the points O, A, B, C, D, E, F, G, H, I, J, Y
variables (O A B C D E F G H I J Y : Type)

-- Constants and assumptions for the regular decagon
constants (regular_decagon : set Type)

-- Define that O is the center of the regular decagon
constant decagon_center : O ∈ regular_decagon

-- Define that Y is the midpoint of side CD
constant midpoint_CD : Y ∈ segment C D

-- Theorem to be proven: the fraction of the shaded area is 7/20
theorem shaded_area_fraction : 
  (fraction_of_shaded_area regular_decagon O Y = 7 / 20) := 
sorry

end shaded_area_fraction_l332_332587


namespace nonnegative_sequence_inequality_l332_332538

theorem nonnegative_sequence_inequality (n : ℕ) (A : Fin (n + 1) → ℝ)
  (h_nonneg : ∀ i : Fin (n+1), 0 ≤ A i)
  (h_increasing : ∀ i j : Fin (n+1), i ≤ j → A i ≤ A j) :
  ∣∑ i in Finset.range ((n/2) + 1), A ⟨2*i, Nat.lt_succ_of_le (Nat.le_of_lt (Nat.mul_lt_mul_of_pos_left (by norm_num) (by norm_num)))⟩ - 1/2 * ∑ i in Finset.range (n+1), A ⟨i, Nat.lt_succ_of_le (Nat.le_int_of_range_succ (by norm_num))⟩∣ ≤ 1/2 * A ⟨n, Nat.lt_succ_of_le (Nat.le_refl n)⟩ :=
by
  sorry

end nonnegative_sequence_inequality_l332_332538


namespace number_of_integers_satisfying_inequality_l332_332031

theorem number_of_integers_satisfying_inequality (S : set ℤ) :
  (S = {x : ℤ | |7 * x - 5| ≤ 9}) →
  S.card = 3 :=
by
  intro hS
  sorry

end number_of_integers_satisfying_inequality_l332_332031


namespace circle_arrangement_division_l332_332576

theorem circle_arrangement_division :
  let d := 1 in
  let regions := (3, 3) in -- 3 rows and 3 columns of circles
  let slope := 2 in
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c ≥ 0 ∧ Nat.gcd a (Nat.gcd b c) = 1 ∧ a = 2 ∧ b = 1 ∧ c = 0 ∧ a^2 + b^2 + c^2 = 5 :=
begin
  sorry
end

end circle_arrangement_division_l332_332576


namespace ellipse_equation_lambda_exists_l332_332426

/-- Problem (I): find the equation of the ellipse C given the conditions. -/
theorem ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0)
  (c : ℝ) (hc : c = sqrt 5)
  (B : ℝ × ℝ) (hB1 : B = (0, 2))
  (hB2 : B.fst^2 / a^2 + B.snd^2 / b^2 = 1) :
  a = 3 ∧ b = 2 ∧ (a^2 = b^2 + c^2) ∧ (∀ x y, x^2 / 9 + y^2 / 4 = 1) :=
sorry

/-- Problem (II): determine if there exists a constant λ such that k₁ = λ k₂. -/
theorem lambda_exists (a b : ℝ) (h : a > b ∧ b > 0)
  (A1 A2 P : ℝ × ℝ) (k1 k2 : ℝ)
  (hne : A1 ≠ A2)
  (hP : P = (1, 0))
  (hA1 : A1 = (-3, 0))
  (hA2 : A2 = (3, 0))
  (hM N : ℝ × ℝ)
  (hMN : M ≠ N)
  (hslope1 : k1 = (M.snd - A1.snd) / (M.fst - A1.fst))
  (hslope2 : k2 = (N.snd - A2.snd) / (N.fst - A2.fst))
  (heq : ∃ λ : ℝ, k1 = λ * k2) :
  ∃ λ, λ = (1/2) :=
sorry

end ellipse_equation_lambda_exists_l332_332426


namespace transfer_deck_l332_332645

-- Define the conditions
variables {k n : ℕ}

-- Assume conditions explicitly
axiom k_gt_1 : k > 1
axiom cards_deck : 2*n = 2*n -- Implicitly states that we have 2n cards

-- Define the problem statement
theorem transfer_deck (k_gt_1 : k > 1) (cards_deck : 2*n = 2*n) : n = k - 1 :=
sorry

end transfer_deck_l332_332645


namespace tank_depth_l332_332805

open Real

theorem tank_depth :
  ∃ d : ℝ, (0.75 * (2 * 25 * d + 2 * 12 * d + 25 * 12) = 558) ∧ d = 6 :=
sorry

end tank_depth_l332_332805


namespace part_I_part_II_l332_332456

-- Define the function f
def f (x : ℕ) : ℤ := x*x - 4*x

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := f n

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 1 then -3
  else 2 * n - 5

-- Define c_n = 2^(a_n)
def c (n : ℕ) : ℝ := real.exp (a n * real.log 2)

-- Sum of first n terms of sequence c_n
def T (n : ℕ) : ℝ := ((c 1) * (1 - real.exp ((n : ℝ) * (real.log 4)))) / (1 - real.log 4)

-- The first part of proof: proving the general term formula for a_n
theorem part_I (n : ℕ) : a n = 2 * n - 5 := sorry

-- The second part of proof: sum T_n of the first n terms of sequence {c_n}
theorem part_II (n : ℕ) : T n = (real.exp (n * (real.log 4)) - 1) / 24 := sorry

end part_I_part_II_l332_332456


namespace ticket_cost_per_ride_l332_332372

theorem ticket_cost_per_ride
  (total_tickets: ℕ) 
  (spent_tickets: ℕ)
  (rides: ℕ)
  (remaining_tickets: ℕ)
  (ride_cost: ℕ)
  (h1: total_tickets = 79)
  (h2: spent_tickets = 23)
  (h3: rides = 8)
  (h4: remaining_tickets = total_tickets - spent_tickets)
  (h5: remaining_tickets = ride_cost * rides):
  ride_cost = 7 :=
by
  sorry

end ticket_cost_per_ride_l332_332372


namespace systematic_sampling_l332_332649

-- Define the conditions given in the problem
def total_bags : ℕ := 50
def bags_to_select : ℕ := 5
def first_bag : ℕ := 6
def step : ℕ := 10

-- Define the sequence as part of the systematic sampling method
def selected_bags : list ℕ :=
  list.range bags_to_select.map (λ i, first_bag + i * step)

-- The final theorem to check the systematic sampling result
theorem systematic_sampling :
  selected_bags = [6, 16, 26, 36, 46] :=
by
sorry

end systematic_sampling_l332_332649


namespace find_lambda_l332_332468

-- Define the vectors a and b
def a := (1 : ℝ, 0 : ℝ)
def b := (2 : ℝ, 1 : ℝ)

-- Define the condition that b - λa is perpendicular to a
def perpendicular_condition (λ : ℝ) : Prop :=
  let diff := (b.1 - λ * a.1, b.2 - λ * a.2)
  diff.1 * a.1 + diff.2 * a.2 = 0

-- State the proof problem
theorem find_lambda : ∃ λ : ℝ, (perpendicular_condition λ) ∧ λ = 2 :=
by {
  use 2,
  simp [perpendicular_condition, a, b],
  sorry
}

end find_lambda_l332_332468


namespace angle_between_vectors_l332_332273

open Real
open InnerProductSpace

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors (a b : V)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = sqrt 2)
  (hperp : ⟪a + b, 2 • a - b⟫ = 0) : 
  real.angle a b = real.pi / 2 :=
by 
  sorry

end angle_between_vectors_l332_332273


namespace incorrect_B_statement_l332_332107

def residual_analysis (n : ℕ) (y : ℕ → ℝ) (y_bar : ℝ) : ℝ :=
  ∑ i in finset.range n, (y i - y_bar) ^ 2

def regression_analysis_statement_B_incorrect (n : ℕ) (y : ℕ → ℝ) (y_hat : ℕ → ℝ) : Prop :=
  ∑ i in finset.range n, (y i - (y_hat i)) ^ 2 < ∑ i in finset.range n, (y i - (y i).mean) ^ 2

theorem incorrect_B_statement 
  (n : ℕ) (y : ℕ → ℝ) (y_bar : ℝ) (y_hat : ℕ → ℝ) :
  regression_analysis_statement_B_incorrect n y y_hat :=
begin
  sorry
end

end incorrect_B_statement_l332_332107


namespace second_antifreeze_percentage_l332_332781

theorem second_antifreeze_percentage :
  ∃ x : ℝ, 
    let pct1 := 0.60
    let vol1 := 26
    let vol_total := 39
    let pct_final := 0.58
    let vol2 := vol_total - vol1
    let pure1 := pct1 * vol1
    let pure2 := x / 100 * vol2
    let pure_final := pct_final * vol_total
    pure_final = pure1 + pure2 ∧ x = 54 :=
begin
  use 54,
  let pct1 := 0.60,
  let vol1 := 26,
  let vol_total := 39,
  let pct_final := 0.58,
  let vol2 := vol_total - vol1,
  let pure1 := pct1 * vol1,
  let pure2 := 54 / 100 * vol2,
  let pure_final := pct_final * vol_total,
  split,
  {
    calc
      pct_final * vol_total = pure1 + pure2 : sorry,
  },
  {
    refl
  }
end

end second_antifreeze_percentage_l332_332781


namespace range_of_a_l332_332018

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 4) (h3 : a > b) (h4 : b > c) :
  a ∈ Set.Ioo (2 / 3) 2 :=
sorry

end range_of_a_l332_332018


namespace sum_of_solutions_l332_332891

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332891


namespace sphere_volume_proof_l332_332007

noncomputable def sphere_volume (R : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (R^3)

theorem sphere_volume_proof :
  ∀ (O A B C : Point) (R r : ℝ),
  distance O (Plane A B C) = R / 2 →
  dist A B = 3 →
  Real.tan (angle A C B) = -Real.sqrt 3 →
  volume (sphere_volume 2) = (32 / 3) * Real.pi :=
by
  intros
  sorry

end sphere_volume_proof_l332_332007


namespace integral_sin2_sin_cos2_cos_l332_332827

open Real

noncomputable def f (x : ℝ) : ℝ := sin (sin x) ^ 2 + cos (cos x) ^ 2

theorem integral_sin2_sin_cos2_cos :
  ∫ x in 0..(π / 2), f x = π / 4 :=
by
  sorry

end integral_sin2_sin_cos2_cos_l332_332827


namespace sum_of_roots_eq_seventeen_l332_332912

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332912


namespace line_m_passes_through_fixed_point_find_line_n_l332_332461

noncomputable def line_m (a : ℝ) : ℝ × ℝ → Prop := 
  λ (p : ℝ × ℝ), (a + 2) * p.1 + (1 - 2a) * p.2 + 4 - 3a = 0

def fixed_point : ℝ × ℝ := (-1, -2)

theorem line_m_passes_through_fixed_point (a : ℝ) : line_m a fixed_point :=
by sorry

noncomputable def line_n (a b : ℝ) : ℝ × ℝ → Prop := 
  λ (p : ℝ × ℝ), p.1 / a + p.2 / b = 1

theorem find_line_n :
  ∃ (a b : ℝ), (line_n a b fixed_point ∧ 
  ((-1 / a + -2 / b = 1) ∧ (1 / 2 * a * b = 4)) ∧
  ((2 * a + b + 4 = 1) → (b = -4) ∧ (a = -2))) :=
by sorry

end line_m_passes_through_fixed_point_find_line_n_l332_332461


namespace circumcenter_AIC_on_circumcircle_ABC_l332_332145

open EuclideanGeometry

variables {A B C I K O : Type} -- Represent points in the Euclidean plane

-- Define the conditions
variable (hI : is_incenter A B C I)
variable (hCircumcircleABC : is_circumcircle A B C O)
variable (hBisectorBI : is_angle_bisector B I K)
variable (hKOnCircumcircle : lies_on_circumcircle K A B C)

-- Define the goal
theorem circumcenter_AIC_on_circumcircle_ABC :
  lies_on_circumcircle (circumcenter A I C) A B C :=
begin
  sorry
end

end circumcenter_AIC_on_circumcircle_ABC_l332_332145


namespace distance_at_least_root_two_l332_332400

-- Define the conditions of the problem
variable {α : Type} [NormedAddCommGroup α] [NormedSpace ℝ α]

/-- Four points in a normed space form a convex quadrilateral -/
def convex_quadrilateral (A B C D : α) : Prop :=
  -- We assume the property that the quadrilateral is convex.
  true -- This is just a placeholder for convexity condition in a real scenario.

theorem distance_at_least_root_two {A B C D : α} (h: convex_quadrilateral A B C D)
  (h2 : ∀ P Q ∈ {A, B, C, D}, P ≠ Q → dist P Q ≥ d) :
  ∃ P Q ∈ {A, B, C, D}, dist P Q ≥ d * Real.sqrt 2 := sorry

end distance_at_least_root_two_l332_332400


namespace three_letter_sets_initials_eq_1000_l332_332474

theorem three_letter_sets_initials_eq_1000 :
  (∃ (A B C : Fin 10), true) = 1000 := 
sorry

end three_letter_sets_initials_eq_1000_l332_332474


namespace candy_pencils_l332_332358

theorem candy_pencils (C Caleb Calen : ℕ) 
  (h1 : Caleb = 2 * C - 3) 
  (h2 : Calen = Caleb + 5) 
  (h3 : Calen - 10 = 10) : 
  C = 9 :=
begin
  sorry
end

end candy_pencils_l332_332358


namespace probability_calculation_l332_332038

def biased_coin_prob_head : ℝ := 0.3
def probability_no_heads_in_two_flips : ℝ := (1 - biased_coin_prob_head) * (1 - biased_coin_prob_head)
def probability_at_least_one_head_in_two_flips : ℝ := 1 - probability_no_heads_in_two_flips
def probability_die_is_six : ℝ := 1 / 6
def combined_probability : ℝ := probability_at_least_one_head_in_two_flips * probability_die_is_six

theorem probability_calculation :
  combined_probability = 0.085 :=
by
  unfold biased_coin_prob_head
  unfold probability_no_heads_in_two_flips
  unfold probability_at_least_one_head_in_two_flips
  unfold probability_die_is_six
  unfold combined_probability
  sorry

end probability_calculation_l332_332038


namespace arithmetic_progression_of_squares_l332_332226

theorem arithmetic_progression_of_squares 
  (a b c : ℝ)
  (h : 1 / (a + b) - 1 / (a + c) = 1 / (b + c) - 1 / (a + c)) :
  2 * b^2 = a^2 + c^2 :=
by
  sorry

end arithmetic_progression_of_squares_l332_332226


namespace geometric_sum_4500_l332_332240

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l332_332240


namespace sum_of_solutions_equation_l332_332938

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332938


namespace convex_functions_on_0_pi_div_2_l332_332427

open Real

noncomputable def f1 (x : ℝ) : ℝ := sin x + cos x
noncomputable def f2 (x : ℝ) : ℝ := log x - 2 * x
noncomputable def f3 (x : ℝ) : ℝ := - x^3 + x
noncomputable def f4 (x : ℝ) : ℝ := x * exp x

def is_convex (f : ℝ → ℝ) (D : set ℝ) : Prop :=
∀ x ∈ D, diffable ℝ f x ∧ diffable ℝ (deriv f) x ∧ (deriv (deriv f) x < 0)

theorem convex_functions_on_0_pi_div_2 :
  is_convex f1 (set.Ioo 0 (pi/2)) ∧
  is_convex f2 (set.Ioo 0 (pi/2)) ∧
  is_convex f3 (set.Ioo 0 (pi/2)) ∧
  ¬ is_convex f4 (set.Ioo 0 (pi/2)) := 
begin
  sorry
end

end convex_functions_on_0_pi_div_2_l332_332427


namespace bills_difference_l332_332298

variable (m j : ℝ)

theorem bills_difference :
  (0.10 * m = 2) → (0.20 * j = 2) → (m - j = 10) :=
by
  intros h1 h2
  sorry

end bills_difference_l332_332298


namespace sum_of_solutions_eq_16_l332_332976

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332976


namespace function_has_property_T_l332_332057

noncomputable def property_T (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ (f a ≠ 0) ∧ (f b ≠ 0) ∧ (f a * f b = -1)

theorem function_has_property_T : property_T (fun x => 1 + x * Real.log x) :=
sorry

end function_has_property_T_l332_332057


namespace product_of_box_areas_l332_332836

theorem product_of_box_areas (a r : ℝ) : 
  let bottom_area := a * (a * r),
      side_area := (a * r) * (a * r^2),
      front_area := (a * r^2) * a in
  bottom_area * side_area * front_area = a^3 * r^8 :=
by
  sorry

end product_of_box_areas_l332_332836


namespace my_current_age_l332_332315

-- Definitions based on the conditions
def bro_age (x : ℕ) : ℕ := 2 * x - 5

-- Main theorem to prove that my current age is 13 given the conditions
theorem my_current_age 
  (x y : ℕ)
  (h1 : y - 5 = 2 * (x - 5))
  (h2 : (x + 8) + (y + 8) = 50) :
  x = 13 :=
sorry

end my_current_age_l332_332315


namespace infinite_multiples_of_2005_with_equal_digits_l332_332185

theorem infinite_multiples_of_2005_with_equal_digits :
  ∃ (B : ℕ) (x_n : ℕ → ℕ), (B = 1023467895) ∧ 
  (∀ n : ℕ, x_n n = nat.of_digits 10 (list.replicate n (nat.digits 10 B))) ∧
  (∀ k n : ℕ, ∃ y_n U_n a_n : ℕ, 
    y_n = x_n (k + n) - x_n k ∧ 
    y_n = U_n * (10 ^ a_n) ∧ 
    U_n % 2005 = 0 ∧ (∀ m, 0 ≤ m ∧ m < 10 → list.count m (nat.digits 10 y_n) = n)) :=
begin
  sorry,
end

end infinite_multiples_of_2005_with_equal_digits_l332_332185


namespace initial_number_of_men_l332_332192

theorem initial_number_of_men (M : ℝ) (P : ℝ) (h1 : P = M * 20) (h2 : P = (M + 200) * 16.67) : M = 1000 :=
by
  sorry

end initial_number_of_men_l332_332192


namespace leftover_balls_when_placing_60_in_tetrahedral_stack_l332_332181

def tetrahedral_number (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) / 6

/--
  When placing 60 balls in a tetrahedral stack, the number of leftover balls is 4.
-/
theorem leftover_balls_when_placing_60_in_tetrahedral_stack :
  ∃ n, tetrahedral_number n ≤ 60 ∧ 60 - tetrahedral_number n = 4 := by
  sorry

end leftover_balls_when_placing_60_in_tetrahedral_stack_l332_332181


namespace intersecting_lines_l332_332844

def diamondsuit (a b : ℝ) : ℝ := a^2 + a * b - b^2

theorem intersecting_lines (x y : ℝ) : 
  (diamondsuit x y = diamondsuit y x) ↔ (y = x ∨ y = -x) := by
  sorry

end intersecting_lines_l332_332844


namespace find_grey_stones_l332_332357

-- Definitions and given conditions
def W : ℕ := 60
def Gr : ℕ := 60
def TotalStones : ℕ := 100
def B : ℕ := TotalStones - W
def StonesRatio : Prop := W / B = G / Gr

-- Goal
theorem find_grey_stones (H1 : W = 60) (H2 : Gr = 60) (H3 : TotalStones = 100) (H4 : W > B) (H5 : StonesRatio) :
  G = 90 :=
sorry

end find_grey_stones_l332_332357


namespace circumcenter_AIC_on_circumcircle_ABC_l332_332148

-- Define points A, B, C, I, O, K
variables (A B C I O K : Type*) [point A] [point B] [point C] [point I] [point O] [point K]
-- Assuming I is the incenter of triangle ABC
variable (incenter_ABC : is_incenter I A B C)
-- Assuming O is the circumcenter of triangle ABC
variable (circumcenter_ABC : is_circumcenter O A B C)
-- Assuming K is the intersection of BI with the circumcircle of ABC
variable (K_on_circumcircle_ABC : is_on_circumcircle K A B C)
variable (BI_bisects_ABC : is_angle_bisector I B K)

-- Define the theorem to be proven in Lean
theorem circumcenter_AIC_on_circumcircle_ABC 
  (h1 : incenter_ABC) 
  (h2 : circumcenter_ABC) 
  (h3 : K_on_circumcircle_ABC)
  (h4 : BI_bisects_ABC) :
  is_circumcenter K A I C :=
by sorry

end circumcenter_AIC_on_circumcircle_ABC_l332_332148


namespace sin_angle_l332_332434

variables (e1 e2 : EuclideanSpace ℝ (Fin 2))
variable (h_norm_e1 : ∥e1∥ = 1)
variable (h_norm_e2 : ∥e2∥ = 1)
variable (h_angle : ∥e1 - e2∥ = sqrt 3)

noncomputable def a : EuclideanSpace ℝ (Fin 2) := 2 • e1 + e2
noncomputable def b : EuclideanSpace ℝ (Fin 2) := -3 • e1 + 2 • e2

theorem sin_angle (h_angle_60 : (1 / 2 : ℝ) = Real.cos π / 3) :
  Real.sin (Real.arccos ((inner (a e1 e2 h_norm_e1 h_norm_e2) (b e1 e2 h_norm_e1 h_norm_e2)) / (∥a e1 e2 h_norm_e1 h_norm_e2∥ * ∥b e1 e2 h_norm_e1 h_norm_e2∥))) = sqrt 3 / 2 :=
sorry

end sin_angle_l332_332434


namespace sum_of_solutions_equation_l332_332940

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332940


namespace perfect_square_trinomial_l332_332489

theorem perfect_square_trinomial (k : ℝ) :
  ∃ k, (∀ x, (4 * x^2 - 2 * k * x + 1) = (2 * x + 1)^2 ∨ (4 * x^2 - 2 * k * x + 1) = (2 * x - 1)^2) → 
  (k = 2 ∨ k = -2) := by
  sorry

end perfect_square_trinomial_l332_332489


namespace range_of_f_on_interval_l332_332860

def f (x : ℝ) : ℝ := 3 / (x + 2)

theorem range_of_f_on_interval :
  ∀ x ∈ set.Icc (-5 : ℝ) (-4 : ℝ), f x ∈ set.Icc (-3/2 : ℝ) (-1 : ℝ) :=
sorry

end range_of_f_on_interval_l332_332860


namespace sum_values_frac_l332_332156

theorem sum_values_frac (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = -3 / 7) : 
  let frac := (p - r) * (q - s) / ((p - q) * (r - s)) in 
  frac = 1 := 
sorry

end sum_values_frac_l332_332156


namespace range_of_m_l332_332560

def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_m (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) :
  (∀ (m : ℝ), f (m * Real.sin θ) + f (1 - m) > 0) → ∃ m ≤ 1, True :=
by
  sorry

end range_of_m_l332_332560


namespace smaller_angle_3_40_pm_l332_332741

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332741


namespace range_of_a_l332_332056

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 
    (x^2 + (2*a^2 + 2)*x - a^2 + 4*a - 7) / (x^2 + (a^2 + 4*a - 5)*x - a^2 + 4*a - 7) < 0) 
    ∧ 
    (∑ (lo hi in set.Ioi (-∞)) (lo < hi), hi - lo < 4)
    → 
  a ∈ set.Iic 1 ∪ set.Ici 3 :=
by
  sorry

end range_of_a_l332_332056


namespace day12_is_monday_l332_332093

-- Define the days of the week
inductive WeekDay
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open WeekDay

-- Define the problem using the conditions
def five_fridays_in_month (first_day : WeekDay) (last_day : WeekDay) : Prop :=
  first_day ≠ Friday ∧ last_day ≠ Friday ∧
  ((first_day = Monday ∨ first_day = Tuesday ∨ first_day = Wednesday ∨ first_day = Thursday ∨
    first_day = Saturday ∨ first_day = Sunday) ∧
  ∃ fridays : Finset ℕ,
  fridays.card = 5 ∧
  ∀ n ∈ fridays, (n % 7 = (5 - WeekDay.recOn first_day 6 0 1 2 3 4 5)) ∧
  fridays ⊆ Finset.range 31 ∧
  1 ∉ fridays ∧ (31 - Finset.max' fridays sorry) % 7 ≠ 0 )

-- Given the problem, prove that the 12th day is a Monday
theorem day12_is_monday (first_day last_day : WeekDay)
  (h : five_fridays_in_month first_day last_day) : 
  (12 % 7 + WeekDay.recOn first_day 6 0 1 2 3 4 5) % 7 = 0 :=
sorry

end day12_is_monday_l332_332093


namespace clock_angle_l332_332707

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332707


namespace solve_for_x_l332_332601

theorem solve_for_x (x : ℚ) :
  (4 * x - 12) / 3 = (3 * x + 6) / 5 → 
  x = 78 / 11 :=
sorry

end solve_for_x_l332_332601


namespace max_people_stand_up_l332_332349

noncomputable def max_people_stand (n : ℕ) : ℕ := 6 * n - 4

theorem max_people_stand_up (n : ℕ) :
  ∀ (P B E : ℕ) (seating : List ℕ), 
  P = 2 * n ∧ B = 2 * n ∧ E = 2 * n ∧
  All Equiv. seating (replicate P 0 ++ replicate B 1 ++ replicate E 2) →
  (∀ i, seating.get cyclic (i % length seating) = seating.get cyclic ((i + 2) % length seating) →
  seating.get cyclic ((i + 1) % length seating) ≠ seating.get cyclic ((i + 2) % length seating)) →
  ∃ stand_up : ℕ, stand_up = max_people_stand n :=
begin
  sorry
end

end max_people_stand_up_l332_332349


namespace circle_equation_reflected_ray_equation_l332_332004

-- Problem 1: Proving the equation of the circle
theorem circle_equation : 
  (∃ a b r : ℝ, a = 3 ∧ b = 2 ∧ r = 2 ∧ 
    (3 - a)^2 + (0 - b)^2 = r^2 ∧
    (5 - a)^2 + (2 - b)^2 = r^2 ∧
    2 * a - b - 4 = 0) → 
  ∀ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 4 :=
sorry

-- Problem 2: Proving the line equation of the reflected ray
theorem reflected_ray_equation :
  (∃ x y : ℝ, (x = 1 ∨ 12 * x - 5 * y - 52 = 0) ∧ 
    (∃ m_x m_y : ℝ, m_x = -4 ∧ m_y = -3 ∧ ray_reflects m_x m_y x y) ∧
    circle_eq (x - 3)^2 + (y - 2)^2 = 4) :=
sorry

end circle_equation_reflected_ray_equation_l332_332004


namespace rain_in_first_hour_l332_332129

theorem rain_in_first_hour :
  ∃ x : ℕ, (let rain_second_hour := 2 * x + 7 in x + rain_second_hour = 22) ∧ x = 5 :=
by
  sorry

end rain_in_first_hour_l332_332129


namespace twelve_is_monday_l332_332089

def Weekday := {d : String // d ∈ ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

def not_friday (d: Weekday) : Prop := d.val ≠ "Friday"

def has_exactly_five_fridays (first_friday: nat) (days_in_month: nat) : Prop :=
  first_friday + 28 <= days_in_month ∧
  first_friday + 21 <= days_in_month ∧
  first_friday + 14 <= days_in_month ∧
  first_friday + 7 <= days_in_month ∧
  first_friday > 0 ∧ days_in_month <= 31

noncomputable def compute_day_of_week (start_day: String) (n: nat) : Weekday :=
  sorry

theorem twelve_is_monday (start_day: Weekday) (days_in_month: nat) :
    has_exactly_five_fridays 2 days_in_month
  → not_friday start_day
  → not_friday (compute_day_of_week start_day.val days_in_month)
  → compute_day_of_week start_day.val 12 = ⟨"Monday", by sorry⟩ :=
begin
  sorry
end

end twelve_is_monday_l332_332089


namespace pond_eye_count_l332_332098

def total_animal_eyes (snakes alligators spiders snails : ℕ) 
    (snake_eyes alligator_eyes spider_eyes snail_eyes: ℕ) : ℕ :=
  snakes * snake_eyes + alligators * alligator_eyes + spiders * spider_eyes + snails * snail_eyes

theorem pond_eye_count : total_animal_eyes 18 10 5 15 2 2 8 2 = 126 := 
by
  sorry

end pond_eye_count_l332_332098


namespace cars_rented_when_x_135_functional_relationship_max_net_income_when_x_l332_332323

-- Definitions for the conditions provided
def total_cars := 50
def management_fee := 1100

def rental_income (x : ℕ) : ℕ :=
if x ≤ 100 then x * total_cars
else x * (total_cars - (x - 100) / 5)

def net_income (x : ℕ) : ℕ :=
rental_income x - management_fee

-- Proof problem 1: Number of cars rented when x = 135
theorem cars_rented_when_x_135 : rental_income 135 = 135 * 43 := by
  sorry

-- Proof problem 2: Functional relationship between y and x
theorem functional_relationship (x : ℕ) : net_income x = 
  if x ≤ 100 then 50 * x - 1100
  else -(1/5 : ℚ) * x^2 + 70 * x - 1100 := by
  sorry

-- Proof problem 3: Maximum net income when x = 175
theorem max_net_income_when_x : (∀ x > 100, net_income x ≤ net_income 175) ∧
(∀ x≤ 100, net_income x ≤ net_income 175) := by
  sorry

end cars_rented_when_x_135_functional_relationship_max_net_income_when_x_l332_332323


namespace green_pill_cost_l332_332356

-- Given conditions
def days := 21
def total_cost := 903
def cost_difference := 2
def daily_cost := total_cost / days

-- Statement to prove
theorem green_pill_cost : (∃ (y : ℝ), y + (y - cost_difference) = daily_cost ∧ y = 22.5) :=
by
  sorry

end green_pill_cost_l332_332356


namespace nathan_blankets_l332_332763

theorem nathan_blankets (b : ℕ) (hb : 21 = (b / 2) * 3) : b = 14 :=
by sorry

end nathan_blankets_l332_332763


namespace hilltop_high_students_l332_332812

theorem hilltop_high_students : 
  ∀ (n_sophomore n_freshman n_junior : ℕ), 
  (n_sophomore : ℚ) / n_freshman = 7 / 4 ∧ (n_junior : ℚ) / n_sophomore = 6 / 7 → 
  n_sophomore + n_freshman + n_junior = 17 :=
by
  sorry

end hilltop_high_students_l332_332812


namespace twelfth_day_is_monday_l332_332064

def Month := ℕ
def Day := ℕ

-- Definitions for days of the week, where 0 represents Monday, 1 represents Tuesday, etc.
inductive Weekday : Type
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
| Sunday : Weekday

open Weekday

-- A month has exactly 5 Fridays
def has_five_fridays (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday ∧
    (∃ last_day: Weekday, last_day ≠ Friday ∧ 
      let fridays := List.filter (λ d, d = Friday) (List.range 31) in
      fridays.length = 5)

-- The first day of the month is not a Friday
def first_day_not_friday (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday

-- The last day of the month is not a Friday
def last_day_not_friday (month: Month): Prop :=
  ∀ last_day: Weekday, last_day = (29 % 7) → last_day ≠ Friday

-- Combining the conditions for the problem
def valid_month (month: Month): Prop :=
  has_five_fridays(month) ∧ first_day_not_friday(month) ∧ last_day_not_friday(month)

-- Prove that the 12th day of the month is a Monday given the conditions
theorem twelfth_day_is_monday (month: Month) (h: valid_month(month)): (∃ starts_on: Weekday, starts_on + 11 = Monday) :=
sorry

end twelfth_day_is_monday_l332_332064


namespace smaller_angle_at_3_40_l332_332731

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332731


namespace sum_inverses_of_distances_eq_2_div_radius_l332_332421

noncomputable theory
open_locale classical

variables {Triangle : Type*} [is_triangle Triangle]
variables (A B C O : Triangle) (R : ℝ) (L M N AL BM CN : ℝ)

-- Assume A, B, C form an acute triangle
def IsAcuteTriangle (A B C : Triangle) : Prop :=
triangle.is_acute A B C

-- Assume O is the circumcenter of the triangle ABC
def IsCircumcenter (O A B C : Triangle) : Prop :=
triangle.is_circumcenter O A B C

-- Assume AO intersects BC at L, BO intersects CA at M, and CO intersects AB at N
def IntersectSides (A B C O L M N : Triangle) : Prop :=
(intersects AO BC L) ∧ (intersects BO CA M) ∧ (intersects CO AB N)

-- Assume R is the radius of the circumcircle of triangle ABC
def Circumradius (O A B C : Triangle) (R : ℝ) : Prop :=
triangle.circumradius O A B C R

-- The theorem statement
theorem sum_inverses_of_distances_eq_2_div_radius
  (h1 : IsAcuteTriangle A B C)
  (h2 : IsCircumcenter O A B C)
  (h3 : IntersectSides A B C O L M N)
  (h4 : Circumradius O A B C R) :
  \(\frac{1}{AL} + \frac{1}{BM} + \frac{1}{CN} = \frac{2}{R}\) := sorry

end sum_inverses_of_distances_eq_2_div_radius_l332_332421


namespace smaller_angle_3_40_l332_332693

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332693


namespace sum_of_solutions_l332_332895

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332895


namespace a_eq_b_l332_332311

def is_power_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2^k

def a_partitions (n : ℕ) : set (list ℕ) :=
  { l | l.sum = n ∧ sorted l ∧ ∀ x ∈ l, is_power_of_two (x + 1) }

def b_partitions (n : ℕ) : set (list ℕ) :=
  { l | l.sum = n ∧ ∀ (x y ∈ l), x ≤ y → 2 * x ≤ y ∨ y = l.last (by sorry) }

theorem a_eq_b (n : ℕ) (hn : 0 < n) :
  (a_partitions n).finite.card = (b_partitions n).finite.card :=
sorry

end a_eq_b_l332_332311


namespace number_of_silver_tokens_l332_332584

/--
Pablo starts with 80 red tokens and 60 blue tokens. In the first booth, 
he can exchange three red tokens for one silver token and two blue tokens.
In the second booth, he can exchange two blue tokens for one silver token 
and one red token. Exchanges continue until no further exchanges are 
possible. How many silver tokens will Pablo have at the end? 
-/
theorem number_of_silver_tokens :
  (∃ x y : ℕ, 
    let R := 80 - 3 * x + y,
    let B := 60 + 2 * x - 2 * y,
    R < 3 ∧ B < 2 ∧ x + y = 134) :=
begin
  -- Proof outline or actual proof steps go here
  sorry
end

end number_of_silver_tokens_l332_332584


namespace youngest_child_age_l332_332634

variables (child_ages : Fin 5 → ℕ)

def child_ages_eq_intervals (x : ℕ) : Prop :=
  child_ages 0 = x ∧ child_ages 1 = x + 8 ∧ child_ages 2 = x + 16 ∧ child_ages 3 = x + 24 ∧ child_ages 4 = x + 32

def sum_of_ages_eq (child_ages : Fin 5 → ℕ) (sum : ℕ) : Prop :=
  (Finset.univ : Finset (Fin 5)).sum child_ages = sum

theorem youngest_child_age (child_ages : Fin 5 → ℕ) (h1 : ∃ x, child_ages_eq_intervals child_ages x) (h2 : sum_of_ages_eq child_ages 90) :
  ∃ x, x = 2 ∧ child_ages 0 = x :=
sorry

end youngest_child_age_l332_332634


namespace harkamal_total_payment_l332_332026

theorem harkamal_total_payment :
  let cost_grapes := 8 * 70 in
  let cost_mangoes := 9 * 45 in
  let cost_apples := 5 * 30 in
  let cost_strawberries := 3 * 100 in
  cost_grapes + cost_mangoes + cost_apples + cost_strawberries = 1415 := by
  sorry

end harkamal_total_payment_l332_332026


namespace sum_of_divisors_154_l332_332762

theorem sum_of_divisors_154 : ∑ d in (finset.filter (λ d, 154 % d = 0) (finset.range 155)), d = 288 :=
by
  sorry

end sum_of_divisors_154_l332_332762


namespace problem1_problem2_l332_332816

-- Problem 1
theorem problem1 : sqrt 36 - 3 * (-1 : ℤ) ^ 2023 + real.cbrt (-8) = 7 := 
sorry

-- Problem 2
theorem problem2 :
  (3 * real.sqrt 3 - 2 * real.sqrt 2) + real.sqrt 2 + abs (1 - real.sqrt 3) = 
  4 * real.sqrt 3 - real.sqrt 2 - 1 := 
sorry

end problem1_problem2_l332_332816


namespace necessary_but_not_sufficient_condition_l332_332435

variable {a : Nat → Real} -- Sequence a_n
variable {q : Real} -- Common ratio
variable (a1_pos : a 1 > 0) -- Condition a1 > 0

-- Definition of geometric sequence
def is_geometric_sequence (a : Nat → Real) (q : Real) : Prop :=
  ∀ n : Nat, a (n + 1) = a n * q

-- Definition of increasing sequence
def is_increasing_sequence (a : Nat → Real) : Prop :=
  ∀ n : Nat, a n < a (n + 1)

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : Nat → Real) (q : Real) (a1_pos : a 1 > 0) :
  is_geometric_sequence a q →
  is_increasing_sequence a →
  q > 0 ∧ ¬(q > 0 → is_increasing_sequence a) := by
  sorry

end necessary_but_not_sufficient_condition_l332_332435


namespace solution_l332_332870

open Set

theorem solution (A B : Set ℤ) :
  (∀ x, x ∈ A ∨ x ∈ B) →
  (∀ x, x ∈ A → (x - 1) ∈ B) →
  (∀ x y, x ∈ B ∧ y ∈ B → (x + y) ∈ A) →
  A = { z | ∃ n, z = 2 * n } ∧ B = { z | ∃ n, z = 2 * n + 1 } :=
by
  sorry

end solution_l332_332870


namespace boys_neither_happy_nor_sad_l332_332306

theorem boys_neither_happy_nor_sad (total_children : ℕ)
  (happy_children sad_children neither_happy_nor_sad total_boys total_girls : ℕ)
  (happy_boys sad_girls : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_sad : sad_children = 10)
  (h_neither : neither_happy_nor_sad = 20)
  (h_boys : total_boys = 17)
  (h_girls : total_girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4) :
  ∃ (boys_neither_happy_nor_sad : ℕ), boys_neither_happy_nor_sad = 5 := by
  sorry

end boys_neither_happy_nor_sad_l332_332306


namespace sum_of_solutions_l332_332966

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332966


namespace small_angle_at_3_40_is_130_degrees_l332_332760

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332760


namespace side_length_of_square_l332_332302

theorem side_length_of_square (s : ℝ) (h : s^2 = 100) : s = 10 := 
sorry

end side_length_of_square_l332_332302


namespace sum_of_solutions_equation_l332_332936

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332936


namespace sum_of_diameters_l332_332652

-- Define the triangle and its sides
def Triangle (P Q R : Type) (PQ : ℝ) (QR : ℝ) (PR : ℝ) :=
  PQ = 15 ∧ QR = 16 ∧ PR = 17

-- Define midpoints
def Midpoint (A B M : Type) := M = (A + B) / 2

-- Main theorem (proof problem)
theorem sum_of_diameters (P Q R J K L Y : Type) (PQ QR PR : ℝ) :
  Triangle P Q R PQ QR PR →
  Midpoint P Q J →
  Midpoint Q R K →
  Midpoint P R L →
  (Y ≠ K) →
  Y ∈ (circumcircle QJK ∩ circumcircle RKL) →
  YP + YQ + YR = \(\frac{6120}{\sqrt{465}}\) :=
sorry

end sum_of_diameters_l332_332652


namespace mark_has_24_dollars_l332_332171

theorem mark_has_24_dollars
  (small_bag_cost : ℕ := 4)
  (small_bag_balloons : ℕ := 50)
  (medium_bag_cost : ℕ := 6)
  (medium_bag_balloons : ℕ := 75)
  (large_bag_cost : ℕ := 12)
  (large_bag_balloons : ℕ := 200)
  (total_balloons : ℕ := 400) :
  total_balloons / large_bag_balloons = 2 ∧ 2 * large_bag_cost = 24 := by
  sorry

end mark_has_24_dollars_l332_332171


namespace smaller_angle_between_hands_at_3_40_l332_332684

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332684


namespace units_digit_a2017_l332_332408

-- Definitions based on conditions
def a1 := 1
def a_seq : ℕ → ℕ
| 1       := a1
| (n + 2) := begin
  sorry -- This denotes the definition deriving from the given recurrence relation
end

-- Definition of the units digit function
def M (x : ℕ) : ℕ := x % 10

-- The main proof problem
theorem units_digit_a2017 : M (a_seq 2017) = 1 :=
sorry

end units_digit_a2017_l332_332408


namespace sum_of_solutions_eq_16_l332_332923

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332923


namespace shirts_washed_and_dried_l332_332842

theorem shirts_washed_and_dried (S : ℕ) : 
  let total_clothing := S + 8 in
  let folded_clothing := 12 + 5 in
  let remaining_clothing := 11 in
  total_clothing - folded_clothing = remaining_clothing → 
  S = 20 :=
by
  intros
  sorry

end shirts_washed_and_dried_l332_332842


namespace percentage_increase_twice_l332_332626

theorem percentage_increase_twice (P : ℝ) (x : ℝ) (hx: P * (1 + x / 100)^2 = P * 1.44) : 
  x = 20 :=
begin
  sorry
end

end percentage_increase_twice_l332_332626


namespace fraction_not_on_time_l332_332352

theorem fraction_not_on_time (n : ℕ) (h1 : ∃ (k : ℕ), 3 * k = 5 * n) 
(h2 : ∃ (k : ℕ), 4 * k = 5 * m) 
(h3 : ∃ (k : ℕ), 5 * k = 6 * f) 
(h4 : m + f = n) 
(h5 : r = rm + rf) 
(h6 : rm = 4/5 * m) 
(h7 : rf = 5/6 * f) :
  (not_on_time : ℚ) = 1/5 := 
by
  sorry

end fraction_not_on_time_l332_332352


namespace total_profit_calculation_l332_332650

variable (investment_Tom : ℝ) (investment_Jose : ℝ) (time_Jose : ℝ) (share_Jose : ℝ) (total_time : ℝ) 
variable (total_profit : ℝ)

theorem total_profit_calculation 
  (h1 : investment_Tom = 30000) 
  (h2 : investment_Jose = 45000) 
  (h3 : time_Jose = 10) -- Jose joined 2 months later, so he invested for 10 months out of 12
  (h4 : share_Jose = 30000) 
  (h5 : total_time = 12) 
  : total_profit = 54000 :=
sorry

end total_profit_calculation_l332_332650


namespace probability_X_3_l332_332060

noncomputable def probability_of_two_reds_in_three_draws (white_balls red_balls : ℕ) : ℚ :=
  let total_balls := white_balls + red_balls in
  (2 * ((white_balls / total_balls) * (red_balls / total_balls) * (red_balls / total_balls))).to_rat

theorem probability_X_3 (white_balls red_balls : ℕ) (h_white : white_balls = 5) (h_red : red_balls = 3) :
  probability_of_two_reds_in_three_draws white_balls red_balls = (45 / 256 : ℚ) :=
by
  rw [h_white, h_red]
  -- Skip the proof
  sorry

end probability_X_3_l332_332060


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332721

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332721


namespace quadrilateral_area_l332_332593

-- Define the problem in Lean 4
def quadrilateral_area_proof (A B C D : ℝ) (AB BC AD DC : ℤ) : Prop :=
∃ (AB BC AD DC : ℝ),
  AB^2 + BC^2 = 25 ∧
  AD^2 + DC^2 = 25 ∧
  AB ≠ AD ∧
  BC ≠ DC ∧
  ∃ S1 S2 : ℝ,
    S1 = 0.5 * (AB * BC) ∧
    S2 = 0.5 * (AD * DC) ∧
    S1 + S2 = 12

-- Lean 4 statement of proof problem
theorem quadrilateral_area (A B C D : ℝ) (h1 : ∠ABC = 90) (h2 : ∠ADC = 90) (AC : ℝ) (H : AC = 5) : 
  quadrilateral_area_proof A B C D :=
sorry

end quadrilateral_area_l332_332593


namespace integer_nearest_T_l332_332788

theorem integer_nearest_T (g : ℝ → ℝ)
  (h : ∀ (x : ℝ), x ≠ 0 → 3 * g x + g (1 / x) = 6 * x + 9) :
  let T := ∑ x in {x : ℝ | g x = 3006}, x in
  T = 667 := by
sorry

end integer_nearest_T_l332_332788


namespace sum_of_multiples_of_11_l332_332287

theorem sum_of_multiples_of_11 (a b : ℤ) (H1 : a = -29) (H2 : b = 79) : 
  (∑ k in (finset.filter (λ x, x % 11 = 0) (finset.range (b - a + 1)).map (λ n, a + n)), k) = 275 :=
by
  -- Lean throws an error if the following sorry statement is missing.
  sorry

end sum_of_multiples_of_11_l332_332287


namespace angle_at_3_40_pm_is_130_degrees_l332_332662

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332662


namespace triangle_construction_possible_l332_332839

theorem triangle_construction_possible (a m_a e : ℝ) 
  (h_positive_a : 0 < a) (h_positive_ma : 0 < m_a) (h_positive_e : 0 < e) :
  a^2 + 4 * m_a^2 ≤ e^2 → 
  ∃ A B C : EuclideanGeometry.Point, 
    EuclideanGeometry.distance B C = a ∧ 
    EuclideanGeometry.perpendicular_distance A B C = m_a ∧ 
    EuclideanGeometry.distance A B + EuclideanGeometry.distance A C = e :=
begin
  intro h,
  sorry
end

end triangle_construction_possible_l332_332839


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332888

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332888


namespace mean_vehicles_l332_332297

theorem mean_vehicles :
  let sunny_cars := [30, 14, 14, 21, 25]
  let sunny_motorcycles := [5, 2, 4, 1, 3]
  let rainy_cars := [40, 20, 17, 31, 30]
  let rainy_motorcycles := [2, 1, 1, 0, 2]
  (sum sunny_cars / 5 = 20.8) ∧
  (sum sunny_motorcycles / 5 = 3) ∧
  (sum rainy_cars / 5 = 27.6) ∧
  (sum rainy_motorcycles / 5 = 1.2) :=
by
  sorry

end mean_vehicles_l332_332297


namespace meeting_probability_l332_332655

theorem meeting_probability (m1 m2 : ℝ) (F1 F2 : ℝ) (a b c : ℤ) : 
  0 ≤ F1 ∧ F1 ≤ 120 ∧ 0 ≤ F2 ∧ F2 ≤ 120 ∧ m1 = m2 ∧
  (m1 + m2 : ℝ) = (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) ∧ 
  (a : ℤ).natAbs ∧ (b : ℤ).natAbs ∧ (c : ℤ).natAbs ∧ 
  ¬ ∃ p : ℤ, p ^ 2 ∣ c ∧ p > 1 →
  (14400 - (120 - m1)^2) / 14400 = 0.5 →
  a + b + c = 362 :=
by
  sorry

end meeting_probability_l332_332655


namespace sum_of_solutions_sum_of_all_solutions_l332_332902

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332902


namespace rain_in_first_hour_l332_332127

theorem rain_in_first_hour :
  ∃ x : ℕ, (let rain_second_hour := 2 * x + 7 in x + rain_second_hour = 22) ∧ x = 5 :=
by
  sorry

end rain_in_first_hour_l332_332127


namespace speed_conversion_l332_332338

theorem speed_conversion (speed_mps: ℝ) (conversion_factor: ℝ) (expected_speed_kmph: ℝ):
  speed_mps * conversion_factor = expected_speed_kmph :=
by
  let speed_mps := 115.00919999999999
  let conversion_factor := 3.6
  let expected_speed_kmph := 414.03312
  sorry

end speed_conversion_l332_332338


namespace dot_product_value_l332_332159

variables {O A B P : Type} [normed_group P] [inner_product_space ℝ P]
variables (p a b : P)

def midpoint (a b : P) : P := (1 / 2 • a + 1 / 2 • b)

def is_perpendicular (u v : P) : Prop := ⟪u, v⟫ = 0

def is_on_perpendicular_bisector (p a b : P) : Prop :=
  ∃ q : P, q = midpoint a b ∧ is_perpendicular (p - q) (b - a)

variables (ha : ∥a∥ = 5) (hb : ∥b∥ = 3) (hp : is_on_perpendicular_bisector p a b)

theorem dot_product_value : ⟪p, a - b⟫ = 8 :=
sorry

end dot_product_value_l332_332159


namespace proof_l332_332867

noncomputable def proof_problem (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem proof (
  a b c : ℝ
) (h1 : (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 = 3 * (a^3 - b^3) * (b^3 - c^3) * (c^3 - a^3))
  (h2 : (a - b)^3 + (b - c)^3 + (c - a)^3 = 3 * (a - b) * (b - c) * (c - a)) :
  proof_problem a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end proof_l332_332867


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332724

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332724


namespace simplify_and_evaluate_expression_l332_332597

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  ((x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l332_332597


namespace twelfth_day_is_monday_l332_332062

def Month := ℕ
def Day := ℕ

-- Definitions for days of the week, where 0 represents Monday, 1 represents Tuesday, etc.
inductive Weekday : Type
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
| Sunday : Weekday

open Weekday

-- A month has exactly 5 Fridays
def has_five_fridays (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday ∧
    (∃ last_day: Weekday, last_day ≠ Friday ∧ 
      let fridays := List.filter (λ d, d = Friday) (List.range 31) in
      fridays.length = 5)

-- The first day of the month is not a Friday
def first_day_not_friday (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday

-- The last day of the month is not a Friday
def last_day_not_friday (month: Month): Prop :=
  ∀ last_day: Weekday, last_day = (29 % 7) → last_day ≠ Friday

-- Combining the conditions for the problem
def valid_month (month: Month): Prop :=
  has_five_fridays(month) ∧ first_day_not_friday(month) ∧ last_day_not_friday(month)

-- Prove that the 12th day of the month is a Monday given the conditions
theorem twelfth_day_is_monday (month: Month) (h: valid_month(month)): (∃ starts_on: Weekday, starts_on + 11 = Monday) :=
sorry

end twelfth_day_is_monday_l332_332062


namespace min_elements_in_sum_set_l332_332999

theorem min_elements_in_sum_set {n : ℕ} (h1 : n ≥ 2)
  (a : Finₓ (n + 1) → ℕ)
  (h2 : a 0 = 0)
  (h3 : a n = 2 * n - 1)
  (h4 : ∀ i j, i < j → a i < a j) :
  ∃ S : Set ℕ, (∀ (i j : Finₓ (n + 1)), i ≤ j → a i + a j ∈ S) ∧ S.card ≥ 3 * n := 
sorry

end min_elements_in_sum_set_l332_332999


namespace inequality_cube_l332_332558

theorem inequality_cube (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_l332_332558


namespace isogonal_conjugate_foci_of_inscribed_ellipse_l332_332347

-- Defining the geometric environment
variables {A B C P Q R F1 F2 : Type*}

-- Conditions given as hypotheses
def is_inscribed (ellipse : Type*) (triangle: Type*) (P Q R : Type*) : Prop := 
  -- The ellipse is tangent to the sides of the triangle at points P, Q, and R
  sorry

def focal_property (X F1 F2 : Type*) : Prop := 
  -- For any point on the ellipse, the sum of distances to the foci is constant
  sorry

def is_isogonal_conjugate (F1 F2 : Type*) (triangle: Type*) : Prop :=
  -- Definition of being isogonally conjugate with respect to a triangle
  sorry

-- The main theorem statement
theorem isogonal_conjugate_foci_of_inscribed_ellipse :
  ∀ (triangle ellipse : Type*) (A B C P Q R F1 F2 : Type*),
  is_inscribed ellipse triangle P Q R →
  focal_property A F1 F2 → 
  is_isogonal_conjugate F1 F2 triangle :=
begin
  intros,
  sorry
end

end isogonal_conjugate_foci_of_inscribed_ellipse_l332_332347


namespace twelfth_day_is_monday_l332_332066

def Month := ℕ
def Day := ℕ

-- Definitions for days of the week, where 0 represents Monday, 1 represents Tuesday, etc.
inductive Weekday : Type
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
| Sunday : Weekday

open Weekday

-- A month has exactly 5 Fridays
def has_five_fridays (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday ∧
    (∃ last_day: Weekday, last_day ≠ Friday ∧ 
      let fridays := List.filter (λ d, d = Friday) (List.range 31) in
      fridays.length = 5)

-- The first day of the month is not a Friday
def first_day_not_friday (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday

-- The last day of the month is not a Friday
def last_day_not_friday (month: Month): Prop :=
  ∀ last_day: Weekday, last_day = (29 % 7) → last_day ≠ Friday

-- Combining the conditions for the problem
def valid_month (month: Month): Prop :=
  has_five_fridays(month) ∧ first_day_not_friday(month) ∧ last_day_not_friday(month)

-- Prove that the 12th day of the month is a Monday given the conditions
theorem twelfth_day_is_monday (month: Month) (h: valid_month(month)): (∃ starts_on: Weekday, starts_on + 11 = Monday) :=
sorry

end twelfth_day_is_monday_l332_332066


namespace second_course_cost_difference_l332_332808

-- Define the conditions
def initial_amount : ℤ := 60
def first_course_cost : ℤ := 15
def remaining_amount : ℤ := 20
def total_spent : ℤ := initial_amount - remaining_amount

-- Define the cost of the second course
variable (second_course_cost : ℤ)

-- Define the dessert cost based on the second course cost
noncomputable def dessert_cost : ℤ := (second_course_cost * 25) / 100

-- Define the total cost equation
def total_cost := first_course_cost + second_course_cost + dessert_cost second_course_cost

-- The proof problem
theorem second_course_cost_difference :
  (total_cost second_course_cost = total_spent) →
  (second_course_cost - first_course_cost = 5) :=
by
  intro h
  sorry

end second_course_cost_difference_l332_332808


namespace pizza_eaten_after_six_trips_l332_332303

theorem pizza_eaten_after_six_trips :
  (1 / 3) + (1 / 3) / 2 + (1 / 3) / 2 / 2 + (1 / 3) / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 / 2 = 21 / 32 :=
by
  sorry

end pizza_eaten_after_six_trips_l332_332303


namespace relation_a_b_range_of_a_l332_332012

-- Definitions and conditions
def f (a b x : ℝ) : ℝ := (a * log x + b) / x
def g (a x : ℝ) : ℝ := x + (2 / x) - a - 2
def F (a b x : ℝ) : ℝ := f a b x + g a x

-- Proof of the relation between a and b
theorem relation_a_b
  (a b : ℝ)
  (h_cond1 : a ≤ 2)
  (h_cond2 : a ≠ 0)
  (h_tangent : let f_val := f a b 1 in let f_prime := deriv (f a b) 1 in (f_prime = a - b) ∧ (0 = (a - b) * (3 - 1) + b))
  : b = 2 * a :=
sorry

-- Proof of the range of a
theorem range_of_a
  (a : ℝ)
  (h_cond : let F_val := λ x => F a (2*a) x in has_one_zero F_val (Ioc 0 2))
  : a = -1 ∨ a < (-(2/(log 2))) ∨ (0 < a ∧ a ≤ 2) :=
sorry

end relation_a_b_range_of_a_l332_332012


namespace minimum_ω_l332_332651

variable (ω : ℝ)
axiom ω_pos : ω > 0

def f (x : ℝ) : ℝ := Real.sin (ω * x)
def g (x : ℝ) : ℝ := Real.sin (ω * (x - (Real.pi / 4)))

theorem minimum_ω : (ω > 0) ∧ (∀ x : ℝ, g x = g (3 * Real.pi / 4 - x)) → ω = 2 := 
by
  intro h
  sorry

end minimum_ω_l332_332651


namespace dennis_total_cost_l332_332849

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end dennis_total_cost_l332_332849


namespace area_triangle_ABP_l332_332791

-- Definitions for the parabola and related geometry
variables
  (C : Parabola) -- The parabola
  (F : Point) -- Focus of the parabola
  (D : Line) -- Directrix of the parabola
  (l : Line) -- Line l passing through F and perpendicular to the axis of symmetry of C
  (A B P : Point) -- Points of intersection and a point on the directrix

-- Conditions
axiom parabola_contains_focus : C.contains F
axiom directrix_property : on_line P D
axiom line_l_property : l.passes_through F ∧ l.perpendicular (axis_of_symmetry C)
axiom intersects_at_A_and_B : C.intersects l A ∧ C.intersects l B
axiom distance_AB : distance A B = 12

-- Theorem: The area of triangle ABP is 36
theorem area_triangle_ABP :
  (area_of_triangle A B P = 36) :=
sorry

end area_triangle_ABP_l332_332791


namespace chord_segments_division_l332_332838

theorem chord_segments_division (O : Point) (r r0 : ℝ) (h : r0 < r) : 
  3 * r0 ≥ r :=
sorry

end chord_segments_division_l332_332838


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332890

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332890


namespace measure_angle_BDC_proof_l332_332441

-- Define the setup of the problem
variables (A B C D : Point) -- vertices of the triangle and the intersection point of external angle bisectors
variable (triangle_ABC : Triangle A B C) -- Triangle ∆ABC
variable (ext_angle_bisectors_intersect : ExternalAngleBisectorsIntersectAt D B C A) -- External angle bisectors intersect at point D

noncomputable def measure_angle_BDC (α β γ δ : Angle) : Angle :=
  180° - (α / 2 + β / 2)

-- The statement to prove
theorem measure_angle_BDC_proof :
  ∠ BDC = 1 / 2 * (180° - ∠ A) :=
sorry

end measure_angle_BDC_proof_l332_332441


namespace rain_first_hour_l332_332121

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end rain_first_hour_l332_332121


namespace projection_correct_l332_332381

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P : Point3D := ⟨-1, 3, -4⟩

def projection_yOz_plane (P : Point3D) : Point3D :=
  ⟨0, P.y, P.z⟩

theorem projection_correct :
  projection_yOz_plane P = ⟨0, 3, -4⟩ :=
by
  -- The theorem proof is omitted.
  sorry

end projection_correct_l332_332381


namespace number_of_elements_in_Y_divisible_by_prime_l332_332137

open Function

variable {X : Type} [Fintype X] {f : X → X} {p : ℕ}

-- Conditions
noncomputable def f_iterate (x : X) : X := f^[p] x

theorem number_of_elements_in_Y_divisible_by_prime 
  (h_nonempty : ∃ x : X, True) 
  (h_fpk : ∀ x : X, f_iterate x = x) 
  (prime_p : Nat.Prime p) :
  let Y := {x : X | f x ≠ x}
  in Fintype.card Y % p = 0 := 
by
  -- Proof would go here
  sorry

end number_of_elements_in_Y_divisible_by_prime_l332_332137


namespace area_of_quadrilateral_ABCD_l332_332590

noncomputable def area_quadrilateral_ABCDE
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC AD DC : ℝ)
  (h_A : AB > 0) (h_B : BC > 0) (h_C : AD > 0) (h_D : DC > 0)
  (h_right_angle_B : ∃ θ : ℝ, θ = π / 2)
  (h_right_angle_D : ∃ θ : ℝ, θ = π / 2)
  (h_AC : ∃ AC : ℝ, AC = 5) :
  ℝ :=
1/2 * AB * BC + 1/2 * AD * DC

theorem area_of_quadrilateral_ABCD
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC AD DC : ℝ)
  (h1 : AB = 3) (h2 : BC = 4) (h3 : AD = 4) (h4 : DC = 3)
  (h_right_angle_B : ∃ θ : ℝ, θ = π / 2)
  (h_right_angle_D : ∃ θ : ℝ, θ = π / 2)
  (h_AC : ∃ AC : ℝ, AC = 5) :
  area_quadrilateral_ABCDE A B C D AB BC AD DC 0 0 0 0 h_right_angle_B h_right_angle_D h_AC = 12 :=
sorry

end area_of_quadrilateral_ABCD_l332_332590


namespace find_other_endpoint_l332_332621

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_other_endpoint
  (m : ℝ × ℝ) (e1 e2 : ℝ × ℝ)
  (hm : m = (3, -3))
  (he1 : e1 = (7, 4))
  (hm_eq : midpoint e1 e2 = m) :
  e2 = (-1, -10) :=
sorry

end find_other_endpoint_l332_332621


namespace problem_A_correct_problem_B_correct_problem_C_incorrect_problem_D_incorrect_l332_332769

variables (α : ℝ)

theorem problem_A_correct : tan (Real.pi + 1) = tan 1 :=
by 
  -- Using the periodicity of tangent
  exact Real.tan_add_pi 1

theorem problem_B_correct (α : ℝ) : sin (-α) / tan (2 * Real.pi - α) = cos α :=
by 
  -- Simplify using trigonometric identities
  have h1 : sin (-α) = -sin α := Real.sin_neg α,
  have h2 : tan (2 * Real.pi - α) = -tan α := by rw Real.tan_sub (2 * Real.pi) α; exact Real.tan_2pi_sub α,
  rw [h1, h2, neg_div_neg_eq],
  exact (cos_div_sin α).symm

/-
The following theorems are included here for completeness but are shown to be incorrect according to the given solution.
-/

theorem problem_C_incorrect (α : ℝ) : ¬ (sin (Real.pi - α) / cos (Real.pi + α) = tan α) :=
by 
  -- Simplify using trigonometric identities
  have h1 : sin (Real.pi - α) = sin α := Real.sin_pi_sub α,
  have h2 : cos (Real.pi + α) = -cos α := Real.cos_add_pi α,
  rw [h1, h2, neg_div],
  exact (neg_ne_self (tan α)).symm

theorem problem_D_incorrect (α : ℝ) : ¬ (cos (Real.pi - α) * tan(-Real.pi - α) / sin (2 * Real.pi - α) = 1) :=
by 
  -- Simplify using trigonometric identities
  have h1 : cos (Real.pi - α) = -cos α := Real.cos_pi_sub α,
  have h2 : tan (-Real.pi - α) = tan (α - 2 * Real.pi) := Real.tan_sub_pi α,
  have h3 : sin (2 * Real.pi - α) = -sin α := Real.sin_sub_pi α,
  rw [h1, h2, h3],
  -- The simplified form will show it differs from 1
  exact (neg_ne_self (1 : ℝ)).symm

end problem_A_correct_problem_B_correct_problem_C_incorrect_problem_D_incorrect_l332_332769


namespace simplify_expr_l332_332596

theorem simplify_expr (x : ℝ) : (4 / (3 * x^(-3))) * (3 * x^2 / 2) = 2 * x^5 :=
by
  sorry

end simplify_expr_l332_332596


namespace ab_ac_product_l332_332105

-- Definition of the problem
theorem ab_ac_product (ABC : Triangle)
  (ha : is_acute ABC)
  (R S : Point)
  (hR : foot_of_perpendicular C AB = R)
  (hS : foot_of_perpendicular B AC = S)
  (Z W : Point)
  (hZW : line_through R S ∩ circumcircle ABC = {Z, W})
  (hZR : distance Z R = 12)
  (hRS : distance R S = 30)
  (hSW : distance S W = 18) :
  AB * AC = 720 * sqrt 15 := 
sorry

end ab_ac_product_l332_332105


namespace find_a_b_decreasing_intervals_max_value_l332_332452

open Real
open Set

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  a * sin x + b * cos x

theorem find_a_b_decreasing_intervals_max_value :
  ∃ (a b : ℝ), 
  (f (π / 3) a b = 0 ∧ f (π / 2) a b = 1 ∧
  a = 1 ∧ b = -√3 ∧
  (∀ k : ℤ, ∃ I : Set ℝ, (I = Icc (2 * k * π + 5 * π / 6) (2 * k * π + 11 * π / 6) ∧
  ∀ x ∈ I, f x a b < f (x + ε) a b = false ) ∧
  ∀ k : ℤ, f (2 * k * π + 5 * π / 6) a b = 2)) :=
by
  sorry

end find_a_b_decreasing_intervals_max_value_l332_332452


namespace last_opened_locker_is_342_l332_332320

def lockers := List.range' 1 1025

def open_lockers : List Bool := List.replicate 1024 false

def flip_lockers (lockers : List Bool) (positions : List Nat) : List Bool :=
positions.foldl (fun acc pos => acc.set! pos (not (acc.get! pos))) lockers

def student_walk (cycle : Nat) (lockers : List Bool) : List Bool :=
let pos_list := List.range’ 0 1024
let positions_to_flip := pos_list.filter (fun pos => (pos / cycle) % 2 == 0)
flip_lockers lockers positions_to_flip

def final_state : List Bool :=
let cycles := List.range’ 1 11
cycles.foldl (fun acc cycle => student_walk cycle acc) open_lockers

def last_opened_locker : Nat :=
let final_lockers := final_state
final_lockers.reverse.indexOf true + 1 -- +1 because we need to translate back to 1-based index

theorem last_opened_locker_is_342 :
  last_opened_locker = 342 :=
by
  sorry

end last_opened_locker_is_342_l332_332320


namespace order_of_magnitude_l332_332858

-- Define the numbers involved
def num1 := 50.6
def num2 := 0.65
def num3 := Real.log 0.65

-- State the proposition to prove
theorem order_of_magnitude (h1 : Real.log 0.65 < 0.65) (h2 : 0.65 < 50.6) : 
  Real.log 0.65 < 0.65 ∧ 0.65 < 50.6 :=
by
  exact ⟨h1, h2⟩

end order_of_magnitude_l332_332858


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332730

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332730


namespace distinct_distances_at_least_15_l332_332411

theorem distinct_distances_at_least_15 (points : Fin 400 → ℝ × ℝ) : 
    ∃ k, k ≥ 15 ∧ k = (set.univ.image (λ (i j : Fin 400), dist (points i) (points j))).card := by
  sorry

end distinct_distances_at_least_15_l332_332411


namespace simplify_expression_l332_332488

variables (b : ℝ)

def computed_expression := (3 * b + 10 - 5 * b^2) / 5

theorem simplify_expression : computed_expression b = -b^2 + 3 * b / 5 + 2 :=
by
  unfold computed_expression
  sorry

end simplify_expression_l332_332488


namespace StatementA_StatementB_StatementC_StatementD_l332_332770

theorem StatementA (x : ℝ) : ¬minimum_value (λ x, sqrt (x^2 + 16) + 9 / sqrt (x^2 + 16)) 6 :=
sorry

theorem StatementB (a c : ℝ) (h : ∀ x, (a * x^2 + 2 * x + c < 0 ↔ x < -1 ∨ x > 2)) : a + c = 2 :=
sorry

theorem StatementC (m : ℝ) (h : decreasing (λ x, (m^2 - 3 * m + 3) * x^(3 * m - 4)) (Ioi 0)) : m = 1 :=
sorry

theorem StatementD (f : ℝ → ℝ) (h : ∀ x ∈ Icc 0 2, domain f x) : domain (λ x, f (2 * x)) (Icc 0 1) :=
sorry

end StatementA_StatementB_StatementC_StatementD_l332_332770


namespace stuart_segments_returning_to_A_l332_332603

-- Define the conditions from the problem
def m_angle_ABC : ℝ := 60

-- Define the derived quantities
def minor_arc_AC := 2 * m_angle_ABC
def minor_arc_AB := (360 - minor_arc_AC) / 2
def minor_arc_BC := minor_arc_AB

-- Define the problem goal in terms of the number of segments drawn
def num_segments (n : ℕ) (m : ℕ) : Prop := 120 * n = 360 * m

-- Theorem statement
theorem stuart_segments_returning_to_A : num_segments 3 1 :=
by
  -- The proof should go here, but we use 'sorry' for now
  sorry

end stuart_segments_returning_to_A_l332_332603


namespace least_positive_multiple_of_15_with_digit_product_multiple_of_15_l332_332284

theorem least_positive_multiple_of_15_with_digit_product_multiple_of_15 : 
  ∃ (n : ℕ), 
    n % 15 = 0 ∧ 
    (∀ k, k % 15 = 0 ∧ (∃ m : ℕ, m < n ∧ m % 15 = 0 ∧ 
    list.prod (nat.digits 10 m) % 15 == 0) 
    → list.prod (nat.digits 10 k) % 15 == 0) 
    ∧ list.prod (nat.digits 10 n) % 15 = 0 
    ∧ n = 315 :=
sorry

end least_positive_multiple_of_15_with_digit_product_multiple_of_15_l332_332284


namespace slope_angle_tangent_line_l332_332009

-- Condition: Curve equation
def curve_equation (x : ℝ) : ℝ := (1 / 2) * x^2 - 2

-- Condition: Point on the curve
def point_P : ℝ × ℝ := (Real.sqrt 3, -(1 / 2))

-- Condition: Derivative of the curve equation
def curve_derivative (x : ℝ) : ℝ := x

-- Prove: The slope angle of the tangent line passing through point P is 60 degrees
theorem slope_angle_tangent_line :
  let P := point_P
  let m := curve_derivative P.1
  θ := Real.atan m * 180 / Real.pi in
  P.1 = Real.sqrt 3 →
  P.2 = -(1 / 2) →
  m = Real.sqrt 3 →
  θ = 60 :=
by
  sorry

end slope_angle_tangent_line_l332_332009


namespace determine_x_l332_332195

def f (x : ℝ) : ℝ := 30 / (x + 2)

def g (x : ℝ) (f_inv : ℝ → ℝ) : ℝ := 4 * f_inv x

theorem determine_x (x : ℝ) (f_inv : ℝ → ℝ) 
    (h₁ : ∀ y, f (f_inv y) = y) 
    (h₂ : ∀ y, f_inv (f y) = y) 
    (hg : g x f_inv = 20) : 
    x = 30 / 7 :=
by
    unfold g at hg
    have hfi : f_inv x = 5 := by linarith
    have hfx : x = f 5 := by rw [←h₁ 5, hfi]
    simp [f] at hfx
    linarith

end determine_x_l332_332195


namespace sum_of_solutions_eqn_l332_332946

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332946


namespace red_block_exists_l332_332495

theorem red_block_exists (board : ℕ × ℕ → Prop) (h_board : ∀ x y, x < 9 ∧ y < 9 → board (x, y) = true ∨ board (x, y) = false) 
(h_red_count : (Finset.univ.filter (λ p : ℕ × ℕ, board p = true)).card = 46) :
  ∃ x y, x < 8 ∧ y < 8 ∧ (Finset.card ((Finset.range 2).product (Finset.range 2)).filter (λ p, board (x + p.fst, y + p.snd) = true)) ≥ 3 :=
by
  sorry

end red_block_exists_l332_332495


namespace prime_factors_count_l332_332482

theorem prime_factors_count (n : ℕ) (h : n = 75) : (nat.factors n).to_finset.card = 2 :=
by
  rw h
  -- The proof part is omitted as instructed
  sorry

end prime_factors_count_l332_332482


namespace minimum_peanuts_l332_332647

noncomputable def original_peanuts (a : ℕ) : ℕ :=
  9 * a + 1

theorem minimum_peanuts : ∃ (a : ℕ), original_peanuts a = 25 :=
by
  use 2
  dsimp [original_peanuts]
  norm_num

end minimum_peanuts_l332_332647


namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l332_332264

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l332_332264


namespace g_is_zero_l332_332152

def g (x : ℝ) : ℝ := sqrt (cos x ^ 4 + 9 * sin x ^ 2) - sqrt (sin x ^ 4 + 9 * cos x ^ 2)

theorem g_is_zero (x : ℝ) : g x = 0 := by
  sorry

end g_is_zero_l332_332152


namespace sum_of_roots_eq_seventeen_l332_332916

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332916


namespace determinant_of_trig_matrix_l332_332540

theorem determinant_of_trig_matrix (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
(h_sum : A + B + C = π) : 
  matrix.det ![![sin A ^ 2, cot A, 1], ![sin B ^ 2, cot B, 1], ![sin C ^ 2, cot C, 1]] = 0 := 
sorry

end determinant_of_trig_matrix_l332_332540


namespace sum_of_solutions_l332_332900

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332900


namespace inv_point_zero_l332_332151

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 1 / (ax + b) ^ (1 / 3)

theorem inv_point_zero (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, (inv_fun (f a b)) 1 = 0 → x = (1 - b) / a :=
sorry

end inv_point_zero_l332_332151


namespace sum_of_solutions_eqn_l332_332950

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332950


namespace motorboat_max_distance_l332_332764

/-- Given a motorboat which, when fully fueled, can travel exactly 40 km against the current 
    or 60 km with the current, proves that the maximum distance it can travel up the river and 
    return to the starting point with the available fuel is 24 km. -/
theorem motorboat_max_distance (upstream_dist : ℕ) (downstream_dist : ℕ) : 
  upstream_dist = 40 → downstream_dist = 60 → 
  ∃ max_round_trip_dist : ℕ, max_round_trip_dist = 24 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end motorboat_max_distance_l332_332764


namespace mary_earnings_max_hours_l332_332573

noncomputable def earnings (hours : ℕ) : ℝ :=
  if hours <= 40 then 
    hours * 10
  else if hours <= 60 then 
    (40 * 10) + ((hours - 40) * 13)
  else 
    (40 * 10) + (20 * 13) + ((hours - 60) * 16)

theorem mary_earnings_max_hours : 
  earnings 70 = 820 :=
by
  sorry

end mary_earnings_max_hours_l332_332573


namespace range_of_a_l332_332041

theorem range_of_a (a : ℝ) : (∃ x : ℝ, real.sin x < a) → a > -1 :=
sorry

end range_of_a_l332_332041


namespace tan_value_of_point_on_exp_graph_l332_332053

theorem tan_value_of_point_on_exp_graph (a : ℝ) (h1 : (a, 9) ∈ {p : ℝ × ℝ | ∃ x, p = (x, 3^x)}) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := by
  sorry

end tan_value_of_point_on_exp_graph_l332_332053


namespace central_square_intersects_all_l332_332835

structure Square (center : ℝ × ℝ) :=
  (sides_parallel_to_axes : True)
  (unit_length : True)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

variable (M : Finset (Square (0, 0)))

theorem central_square_intersects_all (hM : ∀ s1 s2 ∈ M, distance s1.center s2.center ≤ 2) :
  ∃ sq : Square (1, 1), ∀ s ∈ M, (∃ p : ℝ × ℝ, p ∈ s ∧ p ∈ sq) :=
begin
  -- Placeholder to indicate the proof is noncomputable or incomplete.
  sorry
end

end central_square_intersects_all_l332_332835


namespace Calvin_insect_collection_l332_332820

def Calvin_has_insects (num_roaches num_scorpions num_crickets num_caterpillars total_insects : ℕ) : Prop :=
  total_insects = num_roaches + num_scorpions + num_crickets + num_caterpillars

theorem Calvin_insect_collection
  (roach_count : ℕ)
  (scorpion_count : ℕ)
  (cricket_count : ℕ)
  (caterpillar_count : ℕ)
  (total_count : ℕ)
  (h1 : roach_count = 12)
  (h2 : scorpion_count = 3)
  (h3 : cricket_count = roach_count / 2)
  (h4 : caterpillar_count = scorpion_count * 2)
  (h5 : total_count = roach_count + scorpion_count + cricket_count + caterpillar_count) :
  Calvin_has_insects roach_count scorpion_count cricket_count caterpillar_count total_count :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact h5

end Calvin_insect_collection_l332_332820


namespace original_profit_percentage_is_10_l332_332813

-- Define the conditions and the theorem
theorem original_profit_percentage_is_10
  (original_selling_price : ℝ)
  (price_reduction: ℝ)
  (additional_profit: ℝ)
  (profit_percentage: ℝ)
  (new_profit_percentage: ℝ)
  (new_selling_price: ℝ) :
  original_selling_price = 659.9999999999994 →
  price_reduction = 0.10 →
  additional_profit = 42 →
  profit_percentage = 30 →
  new_profit_percentage = 1.30 →
  new_selling_price = original_selling_price + additional_profit →
  ((original_selling_price / (original_selling_price / (new_profit_percentage * (1 - price_reduction)))) - 1) * 100 = 10 :=
by
  sorry

end original_profit_percentage_is_10_l332_332813


namespace vector_sum_l332_332361

theorem vector_sum :
  let a : ℝ^3 := ![3, -2, 7]
  let b : ℝ^3 := ![-1, 5, -3]
  a + b = ![2, 3, 4] :=
by sorry

end vector_sum_l332_332361


namespace sum_of_a_and_b_l332_332014

noncomputable def log_function (a b x : ℝ) : ℝ := Real.log (x + b) / Real.log a

theorem sum_of_a_and_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log_function a b 2 = 1)
                      (h4 : ∃ x : ℝ, log_function a b x = 8 ∧ log_function a b x = 2) :
  a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l332_332014


namespace hyperbola_eccentricity_l332_332444

theorem hyperbola_eccentricity (m : ℝ) (h_eq : ∀ x y, x^2 / 16 - y^2 / m = 1) (h_e : ∀ c a, sqrt (16 + m) / 4 = 5 / 4) : m = 9 :=
sorry

end hyperbola_eccentricity_l332_332444


namespace sum_of_solutions_eqn_l332_332947

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332947


namespace small_angle_at_3_40_is_130_degrees_l332_332753

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332753


namespace largest_integer_k_divides_factorial_l332_332830

open BigOperators

def largest_power_dividing_factorial (n p : ℕ) : ℕ :=
  (∑ i in finset.range (n.log p + 1), n / p^i)

theorem largest_integer_k_divides_factorial (k : ℕ) :
  2520 = 2^3 * 3^2 * 5 * 7 →
  k = largest_power_dividing_factorial 2520 7 →
  k = 418 :=
by
  intros h_fac h_k
  -- proof goes here
  sorry

end largest_integer_k_divides_factorial_l332_332830


namespace largest_divisor_of_m_l332_332499

theorem largest_divisor_of_m (m : ℕ) (h1 : 0 < m) (h2 : 39 ∣ m^2) : 39 ∣ m := sorry

end largest_divisor_of_m_l332_332499


namespace odd_function_for_f_l332_332443

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2 * Real.sin x else -x^2 - 2 * Real.sin x

theorem odd_function_for_f (x : ℝ) : 
  (∀ x, f(-x) = -f(x)) ∧ (∀ x ≥ 0, f(x) = x^2 - 2 * Real.sin x) → ∀ x < 0, f(x) = -x^2 - 2 * Real.sin x :=
by
  sorry

end odd_function_for_f_l332_332443


namespace inequality_proof_l332_332407

theorem inequality_proof (a b c : ℝ) (ha : a = 2 / 21) (hb : b = Real.log 1.1) (hc : c = 21 / 220) : a < b ∧ b < c :=
by
  sorry

end inequality_proof_l332_332407


namespace sum_of_solutions_equation_l332_332937

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332937


namespace tank_fish_count_l332_332254

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l332_332254


namespace smaller_angle_between_hands_at_3_40_l332_332687

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332687


namespace acute_triangle_inequality_l332_332511

theorem acute_triangle_inequality (A B C : ℝ) (hA : 0 < A ∧ A < π/2)
  (hB : 0 < B ∧ B < π/2) (hC : 0 < C ∧ C < π/2)
  (h_sum : A + B + C = π) :
  (cos A)^2 / (sin B * sin C) + (cos B)^2 / (sin C * sin A) + (cos C)^2 / (sin A * sin B) ≥ 1 :=
sorry

end acute_triangle_inequality_l332_332511


namespace proof_problem_statement_l332_332429

noncomputable def ellipse := { P : ℝ × ℝ // P.1^2 / 2 + P.2^2 = 1 }

variables {P Q : ellipse} {F1 F2 O : (ℝ × ℝ)}
variables {k : ℝ}

-- Conditions
def on_ellipse (P : ellipse) : Prop :=
  P.1 * P.1 / 2 + P.2 * P.2 = 1

def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

-- Given conditions
def given_conditions (P Q F1 F2 : ℝ × ℝ) : Prop :=
  on_ellipse P ∧ on_ellipse Q ∧ perpendicular (P - F2) (Q - F2)

def condition1 (P F1 F2 O : ℝ × ℝ) : Prop :=
  min (complex.norm (P - F1 + P - F2)) = 2

def condition2 (P Q F1 F2 : ℝ × ℝ) (k : ℝ) : Prop :=
  perpendicular (P - F1 + P - F2) (Q - F1 + Q - F2) →
  k = slope_of_PQ P Q → k^2 = (2 * sqrt 10 / 10) - 5 / 10

theorem proof_problem_statement (P Q F1 F2 : ℝ × ℝ) (k : ℝ) (h : given_conditions P Q F1 F2) :
  condition1 P F1 F2 0 ∧ condition2 P Q F1 F2 k :=
by
  sorry

end proof_problem_statement_l332_332429


namespace three_letter_sets_initials_eq_1000_l332_332476

theorem three_letter_sets_initials_eq_1000 :
  (∃ (A B C : Fin 10), true) = 1000 := 
sorry

end three_letter_sets_initials_eq_1000_l332_332476


namespace percentage_decrease_l332_332228

theorem percentage_decrease (original_price new_price decrease: ℝ) (h₁: original_price = 2400) (h₂: new_price = 1200) (h₃: decrease = original_price - new_price): 
  decrease / original_price * 100 = 50 :=
by
  rw [h₁, h₂] at h₃ -- Update the decrease according to given prices
  sorry -- Left as a placeholder for the actual proof

end percentage_decrease_l332_332228


namespace rectangular_prism_volume_l332_332008

theorem rectangular_prism_volume
    (height : ℝ) 
    (A'B' A'D' : ℝ) 
    (angle_D_A_B : ℝ) 
    (h_height : height = 1)
    (h_AB_ratio : A'B' = 2 * A'D')
    (h_AB_length : A'B' = 2)
    (h_angle : angle_D_A_B = real.pi / 4) : 
  2 * 2 * 1 = 4 :=
by sorry

end rectangular_prism_volume_l332_332008


namespace pizza_consumption_order_l332_332401

theorem pizza_consumption_order :
  let total_slices := 60
  let alex_slices := total_slices * (1 / 5)
  let beth_slices := total_slices * (1 / 3)
  let cyril_slices := total_slices * (1 / 4)
  let dan_slices := total_slices - (alex_slices + beth_slices + cyril_slices)
  let consumption_list := [(beth_slices, "Beth"), (cyril_slices, "Cyril"), (dan_slices, "Dan"), (alex_slices, "Alex")]
  (consumption_list.sort (fun (a b : (ℝ × String)) => a.fst > b.fst)).map Prod.snd =
  ["Beth", "Cyril", "Dan", "Alex"] :=
by
  sorry

end pizza_consumption_order_l332_332401


namespace parabola_y_intersection_l332_332201

theorem parabola_y_intersection : intersects (x^2 - 4) (0, -4) :=
by
  sorry

end parabola_y_intersection_l332_332201


namespace problem_l332_332624

theorem problem (q r : ℕ) (hq : 1259 = 23 * q + r) (hq_pos : 0 < q) (hr_pos : 0 < r) :
  q - r ≤ 37 :=
sorry

end problem_l332_332624


namespace part_a_part_b_l332_332774

variables (a b c m_a m_b: ℝ)

-- Conditions: lengths of the sides of a triangle
axiom triangle_inequality : ∀ {a b c : ℝ}, a > 0 → b > 0 → c > 0 → a + b > c

-- Problem (a)
theorem part_a (h: triangle_inequality a b c) : a^2 + b^2 ≥ c^2 / 2 := sorry

-- Definitions for medians
def median_a (a b c: ℝ) := (sqrt (2 * b^2 + 2 * c^2 - a^2)) / 2
def median_b (a b c: ℝ) := (sqrt (2 * a^2 + 2 * c^2 - b^2)) / 2

-- Problem (b)
theorem part_b (h: triangle_inequality a b c)
              (h_ma: m_a = median_a a b c)
              (h_mb: m_b = median_b a b c) : m_a^2 + m_b^2 ≥ 9 * c^2 / 8 := sorry

end part_a_part_b_l332_332774


namespace twelve_is_monday_l332_332087

def Weekday := {d : String // d ∈ ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

def not_friday (d: Weekday) : Prop := d.val ≠ "Friday"

def has_exactly_five_fridays (first_friday: nat) (days_in_month: nat) : Prop :=
  first_friday + 28 <= days_in_month ∧
  first_friday + 21 <= days_in_month ∧
  first_friday + 14 <= days_in_month ∧
  first_friday + 7 <= days_in_month ∧
  first_friday > 0 ∧ days_in_month <= 31

noncomputable def compute_day_of_week (start_day: String) (n: nat) : Weekday :=
  sorry

theorem twelve_is_monday (start_day: Weekday) (days_in_month: nat) :
    has_exactly_five_fridays 2 days_in_month
  → not_friday start_day
  → not_friday (compute_day_of_week start_day.val days_in_month)
  → compute_day_of_week start_day.val 12 = ⟨"Monday", by sorry⟩ :=
begin
  sorry
end

end twelve_is_monday_l332_332087


namespace point_on_hyperbola_l332_332459

theorem point_on_hyperbola : 
  (∃ x y : ℝ, (x, y) = (3, -2) ∧ y = -6 / x) :=
by
  sorry

end point_on_hyperbola_l332_332459


namespace sum_of_solutions_eq_16_l332_332974

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332974


namespace line_passes_through_fixed_point_circle_radius_line_circle_intersection_max_distance_from_center_to_line_l332_332460

noncomputable def line (m : ℝ) : ℝ × ℝ → Prop := λ (xy : ℝ × ℝ), (m + 1) * xy.1 + 2 * xy.2 - m - 3 = 0

def circle (xy : ℝ × ℝ) : Prop := xy.1^2 + xy.2^2 - 4 * xy.1 - 4 * xy.2 + 4 = 0

theorem line_passes_through_fixed_point (m : ℝ) :
  line m (1, 1) :=
begin
  dsimp [line],
  ring,
end

theorem circle_radius :
  ∀ (xy : ℝ × ℝ), circle xy → ∃ r : ℝ, r = 2 :=
sorry

theorem line_circle_intersection (m : ℝ) :
  ∃ (xy : ℝ × ℝ), line m xy ∧ circle xy :=
sorry

theorem max_distance_from_center_to_line (m : ℝ) :
  ∃ d : ℝ, d = real.sqrt 2 :=
sorry

end line_passes_through_fixed_point_circle_radius_line_circle_intersection_max_distance_from_center_to_line_l332_332460


namespace min_people_liking_both_l332_332507

theorem min_people_liking_both (total : ℕ) (Beethoven : ℕ) (Chopin : ℕ) 
    (total_eq : total = 150) (Beethoven_eq : Beethoven = 120) (Chopin_eq : Chopin = 95) : 
    ∃ (both : ℕ), both = 65 := 
by 
  have H := Beethoven + Chopin - total
  sorry

end min_people_liking_both_l332_332507


namespace correct_conclusion_l332_332563

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else n * 2^n

theorem correct_conclusion (n : ℕ) (h₁ : ∀ k : ℕ, k > 0 → a_n (k + 1) - 2 * a_n k = 2^(k + 1)) :
  a_n n = n * 2 ^ n :=
by
  sorry

end correct_conclusion_l332_332563


namespace shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l332_332633

noncomputable def k : ℝ := (1 / 20) * Real.log (1 / 4)
noncomputable def b : ℝ := Real.log 160
noncomputable def y (x : ℝ) : ℝ := Real.exp (k * x + b)

theorem shelf_life_at_30_degrees : y 30 = 20 := sorry

theorem temperature_condition_for_shelf_life (x : ℝ) : y x ≥ 80 → x ≤ 10 := sorry

end shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l332_332633


namespace smaller_angle_3_40_pm_l332_332745

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332745


namespace sum_of_roots_eq_seventeen_l332_332919

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332919


namespace day_of_12th_l332_332076

theorem day_of_12th (month : Type) [decidable_eq month] [fintype month] 
  (is_friday : month → Prop) (is_first_day : month → Prop) (is_last_day : month → Prop)
  (days : list month) (nth_day : Π (n : ℕ), month)
  (five_fridays : ∃ (n : ℕ), (is_friday ∘ nth_day) '' (finset.range (min n 7)) = {5})
  (not_first_day_friday : ∀ d, is_first_day d → ¬ is_friday d)
  (not_last_day_friday : ∀ d, is_last_day d → ¬ is_friday d) : 
  nth_day 12 = "Monday" := 
sorry

end day_of_12th_l332_332076


namespace range_of_f_eq_l332_332561

def f (x : ℝ) : ℝ :=
  if x >= 0 then (2 / 3) * x - 1 else 1 / x

theorem range_of_f_eq (a : ℝ) : f(a) < a ↔ -1 < a :=
by
  sorry

end range_of_f_eq_l332_332561


namespace Missouri_to_NewYork_by_car_l332_332213

def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def total_distance_car : ℝ := distance_plane * (1 + increase_percentage)
def distance_midway : ℝ := total_distance_car / 2

theorem Missouri_to_NewYork_by_car : distance_midway = 1400 := by
  sorry

end Missouri_to_NewYork_by_car_l332_332213


namespace count_divisible_by_4_or_7_l332_332487

theorem count_divisible_by_4_or_7 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 60) : 
  (∃ k : ℕ, 4 * k = n) ∨ (∃ k : ℕ, 7 * k = n) ∨ 
  ((∃ k4 : ℕ, 4 * k4 = n) ∧ (∃ k7 : ℕ, 7 * k7 = n)) →
  (finset.card (finset.filter (λ n, (n % 4 = 0) ∨ (n % 7 = 0)) (finset.range 61))) = 21 := 
by
  sorry

end count_divisible_by_4_or_7_l332_332487


namespace simplify_expression_l332_332595

theorem simplify_expression :
  (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 :=
by
  sorry

end simplify_expression_l332_332595


namespace negation_of_prop_l332_332224

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
sorry

end negation_of_prop_l332_332224


namespace find_a10_l332_332232

variable {a : ℕ → ℝ} (d a1 : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

theorem find_a10 (h1 : a 7 + a 9 = 10) 
                (h2 : sum_of_arithmetic_sequence a S)
                (h3 : S 11 = 11) : a 10 = 9 :=
sorry

end find_a10_l332_332232


namespace an_formula_bn_formula_Tn_formula_l332_332445

-- Definitions based on given conditions
def Sn (n : ℕ) : ℕ := n^2
def bn (n : ℕ) : ℕ := 2^(n-1)

-- Questions to prove in Lean 4
theorem an_formula (n : ℕ) : (n > 0) → ∀ (a_n : ℕ), Sn n - Sn (n-1) = a_n → a_n = 2 * n - 1 := sorry

theorem bn_formula (n : ℕ) : (n > 1) → bn 2 = 2 ∧ bn 5 = 16 → ∀ (b_n : ℕ), bn n = b_n := sorry

theorem Tn_formula (n : ℕ) : (n > 0) → ∀ (T_n : ℕ), 
  T_n = (Σ i in range (n + 1), (2 * i - 1) * 2^(i-1)) → T_n = (2 * n - 3) * 2^n + 3 := sorry

end an_formula_bn_formula_Tn_formula_l332_332445


namespace angle_at_3_40_pm_is_130_degrees_l332_332661

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332661


namespace unique_solution_range_l332_332016

theorem unique_solution_range (a : ℝ) : (∀ x : ℝ, (x > 0 → x * log x - a * x + a < 0) → ∃! x : ℤ, x > 0 ∧ x * log x - a * x + a < 0) ↔ a ∈ (2 * log 2, 3/2 * log 3] :=
sorry

end unique_solution_range_l332_332016


namespace pupils_sent_up_exam_l332_332608

theorem pupils_sent_up_exam (average_marks : ℕ) (specific_scores : List ℕ) (new_average : ℕ) : 
  (average_marks = 39) → 
  (specific_scores = [25, 12, 15, 19]) → 
  (new_average = 44) → 
  ∃ n : ℕ, (n > 4) ∧ (average_marks * n) = 39 * n ∧ ((39 * n - specific_scores.sum) / (n - specific_scores.length)) = new_average →
  n = 21 :=
by
  intros h_avg h_scores h_new_avg
  sorry

end pupils_sent_up_exam_l332_332608


namespace min_value_2_when_x_positive_l332_332369

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the problem in Lean
theorem min_value_2_when_x_positive : ∀ x : ℝ, x > 0 → ∃ y, f(x) = y ∧ y ≥ 2 :=
by
  sorry

end min_value_2_when_x_positive_l332_332369


namespace selection_of_students_l332_332984

theorem selection_of_students (A B : Type) (students : Set Type) (h_students : students.card = 10) 
  (h_A : A ∈ students) (h_B : B ∈ students) :
  ∃ ways : ℕ, ways = 140 :=
by
  sorry

end selection_of_students_l332_332984


namespace intersection_M_complement_N_l332_332465

noncomputable def M := {y : ℝ | 1 ≤ y ∧ y ≤ 2}
noncomputable def N_complement := {x : ℝ | 1 ≤ x}

theorem intersection_M_complement_N : M ∩ N_complement = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_M_complement_N_l332_332465


namespace day_of_12th_l332_332078

theorem day_of_12th (month : Type) [decidable_eq month] [fintype month] 
  (is_friday : month → Prop) (is_first_day : month → Prop) (is_last_day : month → Prop)
  (days : list month) (nth_day : Π (n : ℕ), month)
  (five_fridays : ∃ (n : ℕ), (is_friday ∘ nth_day) '' (finset.range (min n 7)) = {5})
  (not_first_day_friday : ∀ d, is_first_day d → ¬ is_friday d)
  (not_last_day_friday : ∀ d, is_last_day d → ¬ is_friday d) : 
  nth_day 12 = "Monday" := 
sorry

end day_of_12th_l332_332078


namespace intersection_y_axis_parabola_l332_332203

theorem intersection_y_axis_parabola : (0, -4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 - 4) ∧ x = 0 } :=
by
  sorry

end intersection_y_axis_parabola_l332_332203


namespace image_of_hom_is_subgroup_l332_332141

variables {G H : Type*} [group G] [group H] (φ : G →* H)

theorem image_of_hom_is_subgroup : subgroup (set.image φ (set.univ : set G)) :=
begin
  sorry
end

end image_of_hom_is_subgroup_l332_332141


namespace small_angle_at_3_40_is_130_degrees_l332_332757

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332757


namespace clock_angle_3_40_l332_332718

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332718


namespace clock_angle_l332_332710

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332710


namespace part_I_monotonic_intervals_part_II_max_integer_k_l332_332453

-- Define the function based on conditions
def f (x : ℝ) (k l : ℝ) : ℝ := x * (Real.log x) + (l - k) * x + k

-- Part (I) when k = l
theorem part_I_monotonic_intervals (k l : ℝ) :
  l = k →
  ∀ x : ℝ, 
    ((1 / Real.exp 1 < x → deriv (λ x, f x l l) x > 0) ∧ 
    (0 < x ∧ x < 1 / Real.exp 1 → deriv (λ x, f x l l) x < 0)) := 
by 
  intros h x
  -- Apply the hypothesis l = k
  rw [h]
  sorry

-- Part (II) find the maximum integer k such that f(x) > 0 for all x > 1
theorem part_II_max_integer_k (l : ℝ) : 
  (∃ k : ℤ, ∀ x : ℝ, 1 < x → f x (k : ℝ) l > 0) → 
  ∃ x₀ : ℝ, 3 < x₀ ∧ x₀ < 4 ∧ (k : ℝ) < x₀ := 
by 
  intros h
  use 3 
  use 4 
  -- Continue the proof steps 
  sorry

end part_I_monotonic_intervals_part_II_max_integer_k_l332_332453


namespace smaller_angle_3_40_pm_l332_332750

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332750


namespace fish_count_l332_332247

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l332_332247


namespace min_value_expression_l332_332387

theorem min_value_expression :
  ∃ x y : ℝ, (sqrt (2 * (1 + cos (2 * x))) - sqrt (9 - sqrt 7) * sin x + 1) * (3 + 2 * sqrt (13 - sqrt 7) * cos y - cos (2 * y)) = -19 :=
by sorry

end min_value_expression_l332_332387


namespace monotonic_decreasing_interval_l332_332623

open Real

noncomputable def f (x : ℝ) := log 0.5 (x^2 - 4)

theorem monotonic_decreasing_interval :
  (∀ x y : ℝ, 2 < x ∨ x < -2 → 2 < y ∨ y < -2 → x < y → f y < f x) :=
by
  sorry

end monotonic_decreasing_interval_l332_332623


namespace range_of_a_l332_332463

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, (a * x^2 - 3 * x - 4 = 0) ∧ (a * y^2 - 3 * y - 4 = 0) → x = y) ↔ (a ≤ -9 / 16 ∨ a = 0) := 
by
  sorry

end range_of_a_l332_332463


namespace total_lives_l332_332256

-- Definitions of given conditions
def original_friends : Nat := 2
def lives_per_player : Nat := 6
def additional_players : Nat := 2

-- Proof statement to show the total number of lives
theorem total_lives :
  (original_friends * lives_per_player) + (additional_players * lives_per_player) = 24 := by
  sorry

end total_lives_l332_332256


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l332_332262

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l332_332262


namespace sum_of_solutions_l332_332968

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332968


namespace f_value_correct_l332_332993

def f : ℝ → ℝ := 
  λ x, if x ≥ 4 then (1 / 2) ^ x else f (x + 1)

noncomputable def f_value : ℝ := 1 + Real.log 5 / Real.log 2

theorem f_value_correct :
  f f_value = 1 / 20 := sorry

end f_value_correct_l332_332993


namespace volume_of_sphere_with_radius_three_l332_332419

theorem volume_of_sphere_with_radius_three : 
  let R := 3 in
  ∀ V, V = (4 / 3) * Real.pi * R^3 → V = 36 * Real.pi :=
by
  intros R V h
  sorry

end volume_of_sphere_with_radius_three_l332_332419


namespace correct_propositions_l332_332343

open_locale classical
noncomputable theory

variables {v1 v2 v3 : ℝ × ℝ}

def is_basis (v1 v2 : ℝ × ℝ) : Prop :=
  v1 ≠ 0 ∧ v2 ≠ 0 ∧ v1 ≠ v2 ∧ ¬ collinear v1 v2

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1.1 = k * v2.1 ∧ v1.2 = k * v2.2

def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0
  
def nonzero (v : ℝ × ℝ) : Prop :=
  v ≠ 0

def two_vectors_bring_infinite_bases (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ v3 v4 : ℝ × ℝ, is_basis v3 v4

theorem correct_propositions (v1 v2 v3 : ℝ × ℝ) :
  (¬ is_basis v1 v2) ∧ is_basis v3 v1 ∧ perpendicular v1 v2 ∧
  ¬ (∀ v: ℝ × ℝ, nonzero v → (∃ a b c : ℝ, v = a • v1 + b • v2 + c • v3)) := by
  sorry

end correct_propositions_l332_332343


namespace canFormTriangle_cannotFormIsoscelesTriangle_l332_332246

section TriangleSticks

noncomputable def stickLengths : List ℝ := 
  List.range 10 |>.map (λ n => 1.9 ^ n)

def satisfiesTriangleInequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem canFormTriangle : ∃ (a b c : ℝ), a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

theorem cannotFormIsoscelesTriangle : ¬∃ (a b c : ℝ), a = b ∧ a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

end TriangleSticks

end canFormTriangle_cannotFormIsoscelesTriangle_l332_332246


namespace parallel_lines_a_l332_332498

-- Definitions of the lines
def l1 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 6 * x + a * y + 2 = 0

-- The main theorem to prove
theorem parallel_lines_a (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = 3) := 
sorry

end parallel_lines_a_l332_332498


namespace number_of_positive_divisors_of_12m_squared_l332_332545

theorem number_of_positive_divisors_of_12m_squared (m : ℤ) (hm1 : m % 2 ≠ 0) (hm2 : (m.factors.length.succ = 13)) : 
  ∃ d, d = ∏ (p, k) in (12 * m^2).factors.pmap (λ p n, (p, n.succ)) (λ p hp, prod.fst hp) := 
  d = 150 :=
sorry

end number_of_positive_divisors_of_12m_squared_l332_332545


namespace incorrect_expression_d_l332_332362

variables {u v : ℕ} {X Y Z : ℝ}

-- Z is expressed as 0.XY... with X non-repeating and Y repeating.
def repeating_decimal (u v : ℕ) (X Y : ℝ) : ℝ := 
  let non_repeating_part := X * 10^(-u)
  let repeating_part := Y / (10^v - 1) * 10^(-u)
  non_repeating_part + repeating_part

def Z := repeating_decimal u v X Y

theorem incorrect_expression_d :
  10^{2*u} * (10^v - 1) * Z ≠ Y * (X^2 - 1) :=
sorry

end incorrect_expression_d_l332_332362


namespace sum_of_solutions_equation_l332_332933

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332933


namespace followers_after_one_year_l332_332852

theorem followers_after_one_year :
  let initial_followers := 100000
  let daily_new_followers := 1000
  let unfollowers_per_year := 20000
  let days_per_year := 365
  initial_followers + (daily_new_followers * days_per_year - unfollowers_per_year) = 445000 :=
by
  sorry

end followers_after_one_year_l332_332852


namespace find_alpha_l332_332161

-- Define that y = x^alpha is an odd function
def is_odd_function (f: ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f x

-- y = x^alpha being above the line y = x in the interval (0,1)
def above_line (alpha : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 1 → x^alpha > x

-- Main theorem statement
theorem find_alpha : 
  ∃ α ∈ ({1, 2, 3, 1/2, -1} : set ℝ), 
    (is_odd_function (λ x, x^α)) ∧ (above_line α) := 
begin
  use -1,
  simp,
  split,
  { intros x,
    simp,
    exact sorry,
  },
  { intros x hx1 hx2,
    norm_num at hx1 hx2,
    exact sorry,
  }
end

end find_alpha_l332_332161


namespace percentage_needed_to_pass_l332_332575

def MikeScore : ℕ := 212
def Shortfall : ℕ := 19
def MaxMarks : ℕ := 770

theorem percentage_needed_to_pass :
  (231.0 / (770.0 : ℝ)) * 100 = 30 := by
  -- placeholder for proof
  sorry

end percentage_needed_to_pass_l332_332575


namespace train_length_l332_332658

noncomputable theory

variables 
  (L : ℝ) -- Length of each train in meters.
  (v_fast_kmhr : ℝ := 46) -- Speed of the faster train in km/hr.
  (v_slow_kmhr : ℝ := 36) -- Speed of the slower train in km/hr.
  (t_sec : ℝ := 27) -- Time to pass in seconds.

-- Convert km/hr to m/s using the conversion factor 1 km/hr = 5/18 m/s.
def kmhr_to_ms (v : ℝ) : ℝ := v * (5/18)

-- Define the speeds in m/s.
def v_fast := kmhr_to_ms v_fast_kmhr
def v_slow := kmhr_to_ms v_slow_kmhr

-- Calculate the relative speed in m/s.
def relative_speed : ℝ := v_fast - v_slow

-- Distance covered in the given time.
def distance_covered : ℝ := relative_speed * t_sec

-- The length of each train is half the distance covered.
def length_of_each_train : ℝ := distance_covered / 2

-- The main statement to be proven.
theorem train_length : length_of_each_train = 37.5 :=
sorry

end train_length_l332_332658


namespace simple_interest_time_period_l332_332354

theorem simple_interest_time_period 
  (P : ℝ) (R : ℝ := 4) (T : ℝ) (SI : ℝ := (2 / 5) * P) :
  SI = P * R * T / 100 → T = 10 :=
by {
  sorry
}

end simple_interest_time_period_l332_332354


namespace divide_select_fairness_l332_332656

-- Define the conditions
def housewives (H1 H2 : Type) := ∃ m : Type, 
  (m ∈ H1) ∧ 
  (m ∈ H2) ∧ 
  (∃ k a : Prop, k ∧ a)

-- The main theorem statement
theorem divide_select_fairness (H1 H2 : Type) (m : Type)
    (condition1 : m ∈ H1 ∧ m ∈ H2)
    (condition2 : ∃ (k a : Prop), k ∧ a) :
  ∃ (Part1 Part2 : m), 
    (H1-> Part1 \approx Part2) ∧ 
    (H2-> (selects Part1 ∨ selects Part2)) ∧ 
    (fairness Part1 Part2)
sorry

end divide_select_fairness_l332_332656


namespace find_alpha_find_angle_l332_332118

noncomputable def pointA : (ℝ × ℝ) := (2, 0)
noncomputable def pointC (α : ℝ) : (ℝ × ℝ) := (Real.cos α, Real.sin α)
noncomputable def vectorOA : (ℝ × ℝ) := pointA
noncomputable def vectorOC (α : ℝ) : (ℝ × ℝ) := pointC α
noncomputable def magnitude (x : ℝ) (y : ℝ) : ℝ := Real.sqrt (x * x + y * y)

-- Part 1: Proving the value of α
theorem find_alpha (α : ℝ) (hα : α ∈ Set.Ioo 0 Real.pi) 
  (hmagnitude : magnitude (2 + Real.cos α) (Real.sin α) = Real.sqrt 7) : α = Real.pi / 3 :=
sorry

-- Part 2: Find the angle between ∠(OA, AC)
noncomputable def vectorAC (α : ℝ) : (ℝ × ℝ) := 
  (Real.cos α - 2, Real.sin α)

theorem find_angle (α : ℝ) (hα : α = Real.pi / 3) 
  : ∃ θ : ℝ, θ = Real.acos ((2*(Real.cos α - 2)) / (magnitude 2 0 * magnitude (Real.cos α - 2) (Real.sin α))) :=
sorry

end find_alpha_find_angle_l332_332118


namespace socks_total_l332_332169

def socks_lisa (initial: Nat) := initial + 0

def socks_sandra := 20

def socks_cousin (sandra: Nat) := sandra / 5

def socks_mom (initial: Nat) := 8 + 3 * initial

theorem socks_total (initial: Nat) (sandra: Nat) :
  initial = 12 → sandra = 20 → 
  socks_lisa initial + socks_sandra + socks_cousin sandra + socks_mom initial = 80 :=
by
  intros h_initial h_sandra
  rw [h_initial, h_sandra]
  sorry

end socks_total_l332_332169


namespace symmetric_point_origin_l332_332205

def symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, -p.2, -p.3)

theorem symmetric_point_origin :
  symmetric_point (1, 2, 3) = (-1, -2, -3) :=
by
  sorry

end symmetric_point_origin_l332_332205


namespace shaded_area_l332_332106

-- Definitions for the conditions
variables (ABCD : Type) [parallelogram ABCD]
variable (O E F : ABCD)
variable (OE_eq_EF_eq_FD : distance OE = distance EF ∧ distance EF = distance FD)
variable (area_ABCD : area ABCD = 240)

-- The proof statement
theorem shaded_area (h : distance OE = distance EF ∧ distance EF = distance FD) (h_area : area ABCD = 240) :
  shaded_region(O E F) = 20 := sorry

end shaded_area_l332_332106


namespace twelfth_is_monday_l332_332075

def days_of_week : Type := {d // d < 7}

def starts_on_friday (d : ℕ) : days_of_week := ⟨d % 7, by linarith [Nat.mod_lt d (by norm_num)]⟩

-- Condition 1: There are exactly 5 Fridays in the month (which has at least 30 days)
def has_five_fridays (days_in_month : ℕ) : Prop := 
  ∃ (start : ℕ), 
    (start % 7 ≠ 5) ∧ 
    (days_in_month > 28 ∧ days_in_month < 32) ∧ -- At least 30 days to have 5 Fridays
    ∃ f, ∀ i, starts_on_friday(start + 7*i) = f → (i < 5)

-- Condition 2: The first day of the month is not a Friday
def first_not_friday (start : ℕ) : Prop := start % 7 ≠ 5

-- Condition 3: The last day of the month is not a Friday
def last_not_friday (start days_in_month : ℕ) : Prop := (start + days_in_month - 1) % 7 ≠ 5
    
theorem twelfth_is_monday (days_in_month : ℕ) (start : ℕ) 
  (h_five_fridays : has_five_fridays days_in_month)
  (h_first_not_friday : first_not_friday start)
  (h_last_not_friday : last_not_friday start days_in_month) : 
  (start + 11) % 7 = 1 :=
sorry

end twelfth_is_monday_l332_332075


namespace hotel_room_friends_distribution_l332_332330

theorem hotel_room_friends_distribution 
    (rooms : ℕ)
    (friends : ℕ)
    (min_friends_per_room : ℕ)
    (max_friends_per_room : ℕ)
    (unique_ways : ℕ) :
    rooms = 6 →
    friends = 10 →
    min_friends_per_room = 1 →
    max_friends_per_room = 3 →
    unique_ways = 1058400 :=
by
  intros h_rooms h_friends h_min_friends h_max_friends
  sorry

end hotel_room_friends_distribution_l332_332330


namespace smallest_k_for_angle_bisectors_l332_332392

theorem smallest_k_for_angle_bisectors (a b t_a t_b : ℝ) (h_a_b_triangle : a > 0 ∧ b > 0) 
                                       (h_ta_angle_bisector : is_angle_bisector a t_a)
                                       (h_tb_angle_bisector : is_angle_bisector b t_b) : 
  (∃ k : ℝ, (∀ (a b t_a t_b : ℝ), (a > 0 ∧ b > 0) → is_angle_bisector a t_a → is_angle_bisector b t_b → 
            (t_a + t_b) / (a + b) < k) ∧ k = 4 / 3) :=
begin
  sorry
end

end smallest_k_for_angle_bisectors_l332_332392


namespace num_nonneg_int_solutions_l332_332843

def op (a b : ℤ) : ℤ := 1 - a * b

theorem num_nonneg_int_solutions : 
  let num_nonneg_solutions := {x : ℤ | x ≥ 0 ∧ op x 2 ≥ -3}.to_finset.card
  num_nonneg_solutions = 3 :=
sorry

end num_nonneg_int_solutions_l332_332843


namespace households_using_both_brands_l332_332794

variable (x : ℕ) -- the number of households using both brands of soap

-- Given conditions
variable (total_households : ℕ) (neither_E_nor_B : ℕ) (only_E : ℕ) (ratio : ℕ)

axiom total_households_eq_500 : total_households = 500
axiom neither_E_nor_B_eq_150 : neither_E_nor_B = 150
axiom only_E_eq_140 : only_E = 140
axiom ratio_eq_5 : ratio = 5

theorem households_using_both_brands :
  total_households - (neither_E_nor_B + only_E + x + (ratio * x / 2)) = 0 -> x = 60 :=
by
  intros h
  rw [total_households_eq_500, neither_E_nor_B_eq_150, only_E_eq_140, ratio_eq_5] at h
  -- The proof follows from the conditions and the equation
  sorry

end households_using_both_brands_l332_332794


namespace leo_current_weight_l332_332496

def leo_sisters_weights (L K T : ℝ) : Prop :=
  L + 15 = 1.75 * K ∧
  T = 0.80 * K ∧
  L + K + T = 350

theorem leo_current_weight : ∃ L : ℝ, leo_sisters_weights L 102.82 (0.80 * 102.82) ∧ L ≈ 164.94 :=
by
  sorry

end leo_current_weight_l332_332496


namespace triangle_identity_proof_l332_332182

variables (r r_a r_b r_c R S p : ℝ)
-- assume necessary properties for valid triangle (not explicitly given in problem but implied)
-- nonnegativity, relations between inradius, exradii and circumradius, etc.

theorem triangle_identity_proof
  (h_r_pos : 0 < r)
  (h_ra_pos : 0 < r_a)
  (h_rb_pos : 0 < r_b)
  (h_rc_pos : 0 < r_c)
  (h_R_pos : 0 < R)
  (h_S_pos : 0 < S)
  (h_p_pos : 0 < p)
  (h_area : S = r * p) :
  (1 / r^3) - (1 / r_a^3) - (1 / r_b^3) - (1 / r_c^3) = (12 * R) / (S^2) :=
sorry

end triangle_identity_proof_l332_332182


namespace least_multiple_of_15_with_product_multiple_of_15_is_315_l332_332281

-- Let n be a positive integer
def is_multiple_of (n m : ℕ) := ∃ k, n = m * k

def is_product_multiple_of (digits : ℕ) (m : ℕ) := ∃ k, digits = m * k

-- The main theorem we want to state and prove
theorem least_multiple_of_15_with_product_multiple_of_15_is_315 (n : ℕ) 
  (h1 : is_multiple_of n 15) 
  (h2 : n > 0) 
  (h3 : is_product_multiple_of (n.digits.prod) 15) 
  : n = 315 := 
sorry

end least_multiple_of_15_with_product_multiple_of_15_is_315_l332_332281


namespace x_coordinate_of_M_l332_332329

open Real

theorem x_coordinate_of_M (x y : ℝ) :
  (∃ M : ℝ × ℝ, M.1 = x ∧ M.2 = y ∧ 2 * x - y + 3 = 0
    ∧ ((x - 2)^2 + y^2 = 5) ∧ ((∃ P Q : ℝ × ℝ, P ≠ Q 
    ∧ dist P M = dist Q M = sqrt 5 
    ∧ ∠ (P.1, P.2) (2, 0) (Q.1, Q.2) = π / 2)))
  → x = -1 ∨ x = -3/5 := by
  sorry

end x_coordinate_of_M_l332_332329


namespace twelfth_is_monday_l332_332073

def days_of_week : Type := {d // d < 7}

def starts_on_friday (d : ℕ) : days_of_week := ⟨d % 7, by linarith [Nat.mod_lt d (by norm_num)]⟩

-- Condition 1: There are exactly 5 Fridays in the month (which has at least 30 days)
def has_five_fridays (days_in_month : ℕ) : Prop := 
  ∃ (start : ℕ), 
    (start % 7 ≠ 5) ∧ 
    (days_in_month > 28 ∧ days_in_month < 32) ∧ -- At least 30 days to have 5 Fridays
    ∃ f, ∀ i, starts_on_friday(start + 7*i) = f → (i < 5)

-- Condition 2: The first day of the month is not a Friday
def first_not_friday (start : ℕ) : Prop := start % 7 ≠ 5

-- Condition 3: The last day of the month is not a Friday
def last_not_friday (start days_in_month : ℕ) : Prop := (start + days_in_month - 1) % 7 ≠ 5
    
theorem twelfth_is_monday (days_in_month : ℕ) (start : ℕ) 
  (h_five_fridays : has_five_fridays days_in_month)
  (h_first_not_friday : first_not_friday start)
  (h_last_not_friday : last_not_friday start days_in_month) : 
  (start + 11) % 7 = 1 :=
sorry

end twelfth_is_monday_l332_332073


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332729

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332729


namespace rook_path_count_l332_332184

theorem rook_path_count (n : ℕ) :
  ∀ (board : Fin n × Fin n), (n > 0) → (¬(board = (⟨0, λ h,0⟩, ⟨0, λ h,0⟩)) → 
  ∃ (f : (Fin n × Fin n) × (Fin n × Fin n) → bool), ((∀ (x y : Fin n × Fin n), 
  f (x, y) → (y.1 = x.1 + 1 ∨ y.2 = x.2 + 1)) ∧ ∃ (p : list ((Fin n × Fin n))), 
  (head p = (⟨0, λ h,0⟩, ⟨0, λ h,0⟩)) ∧ (last p = board) ∧ (∀ (i < p.length - 1), f (p.nth_le i sorry, p.nth_le (i + 1) sorry)))).

#check rook_path_count

end rook_path_count_l332_332184


namespace find_omega_value_l332_332050

theorem find_omega_value (ω : ℝ) (h : ω > 0) (h_dist : (1/2) * (2 * π / ω) = π / 6) : ω = 6 :=
by
  sorry

end find_omega_value_l332_332050


namespace sum_of_solutions_l332_332964

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332964


namespace perpendicular_planes_necessary_not_sufficient_l332_332985

-- Definitions of the planes and line
variable (α β : Plane) (m : Line)

-- Conditions
axiom (h_m_in_α : m ∈ α)

-- Statement to prove
theorem perpendicular_planes_necessary_not_sufficient : (α ⊥ β) → (m ⊥ β) ∧ ¬(m ⊥ β → α ⊥ β) :=
sorry

end perpendicular_planes_necessary_not_sufficient_l332_332985


namespace gift_contributors_l332_332322

theorem gift_contributors :
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ 20 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → (9 : ℕ) ≤ 20) →
  (∃ (n : ℕ), n = 12) :=
by
  sorry

end gift_contributors_l332_332322


namespace largest_x_to_floor_ratio_l332_332383

theorem largest_x_to_floor_ratio : ∃ x : ℝ, (⌊x⌋ / x = 9 / 10 ∧ ∀ y : ℝ, (⌊y⌋ / y = 9 / 10 → y ≤ x)) :=
sorry

end largest_x_to_floor_ratio_l332_332383


namespace binomial_expansion_third_term_l332_332110

theorem binomial_expansion_third_term (a b : ℝ) :
  let term := (Nat.choose 10 2) * (a ^ 8) * (b ^ 2) in
  term = (Nat.choose 10 2) * (a ^ 8) * (b ^ 2) := by
  sorry

end binomial_expansion_third_term_l332_332110


namespace consistent_coloring_of_hexagonal_board_l332_332577

-- Assuming the existence of a function 'traverse' that produces
-- the numbering based on camel traversal and a function 'coloring'
-- that assigns colors based on the given conditions.

def camel_traversal (n : ℕ) : List ℕ :=
  sorry  -- Define the specific traversal sequence of the camel.

def color_field (x : ℕ) : String :=
  if x % 3 == 0 then "Black"
  else if x % 3 == 1 then "Red"
  else "Other"

theorem consistent_coloring_of_hexagonal_board :
  ∀ (board_size : ℕ), 
    let board := camel_traversal board_size in 
    let colored_board := board.map color_field in
    -- Assuming we want to check equivalence of coloring patterns (e.g., the positions of “Black” or “Red” coincide correctly)
    ∀ x y, color_field x = color_field y → (x = y ∨ x ≠ y) :=
by
  sorry -- Proof provided here will establish the consistency of the coloring pattern.

end consistent_coloring_of_hexagonal_board_l332_332577


namespace opposite_of_neg_third_l332_332227

theorem opposite_of_neg_third : (-(-1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_neg_third_l332_332227


namespace sum_of_solutions_l332_332959
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332959


namespace geometric_sequence_sum_l332_332518

variable {a b : ℝ} -- Parameters for real numbers a and b
variable (a_ne_zero : a ≠ 0) -- condition a ≠ 0

/-- Proof that in the geometric sequence {a_n}, given a_5 + a_6 = a and a_15 + a_16 = b, 
    a_25 + a_26 = b^2 / a --/
theorem geometric_sequence_sum (a5_plus_a6 : ℕ → ℝ) (a15_plus_a16 : ℕ → ℝ) (a25_plus_a26 : ℕ → ℝ)
  (h1 : a5_plus_a6 5 + a5_plus_a6 6 = a)
  (h2 : a15_plus_a16 15 + a15_plus_a16 16 = b) :
  a25_plus_a26 25 + a25_plus_a26 26 = b^2 / a :=
  sorry

end geometric_sequence_sum_l332_332518


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332889

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332889


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332725

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332725


namespace number_of_powers_of_2_not_powers_of_4_below_500000_l332_332034

theorem number_of_powers_of_2_not_powers_of_4_below_500000 : 
  (Set.filter (λ n, ∃ k, n = 2^k ∧ n < 500000 ∧ ¬ ∃ m, n = 4^m) {n : ℕ | n > 0}).card = 9 :=
sorry

end number_of_powers_of_2_not_powers_of_4_below_500000_l332_332034


namespace intersection_y_axis_parabola_l332_332204

theorem intersection_y_axis_parabola : (0, -4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 - 4) ∧ x = 0 } :=
by
  sorry

end intersection_y_axis_parabola_l332_332204


namespace clock_angle_l332_332703

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332703


namespace remainder_of_product_div_1000_l332_332879

noncomputable def sequence : ℕ → ℤ
| 1     := 9
| k + 2 := 10^(k + 2) - 1

noncomputable def product_of_sequence : ℕ → ℤ
| 0     := 1
| k + 1 := (sequence (k + 1)) * product_of_sequence k

theorem remainder_of_product_div_1000 :
  (product_of_sequence 999) % 1000 = 109 :=
sorry

end remainder_of_product_div_1000_l332_332879


namespace smaller_angle_between_hands_at_3_40_l332_332690

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332690


namespace steves_initial_emails_l332_332193

theorem steves_initial_emails (E : ℝ) (ht : E / 2 = (0.6 * E) + 120) : E = 400 :=
  by sorry

end steves_initial_emails_l332_332193


namespace find_x_value_l332_332492

theorem find_x_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 :=
sorry

end find_x_value_l332_332492


namespace carlas_overall_score_percentage_l332_332825

theorem carlas_overall_score_percentage :
  let score1 := 0.85 * 15,
      score2 := 0.75 * 25,
      score3 := 0.80 * 20,
      total_correct := score1 + score2 + score3,
      total_problems := 15 + 25 + 20,
      overall_percentage := (total_correct / total_problems) * 100
  in
  overall_percentage.round = 80 :=
by 
  let score1 := 0.85 * 15;
  let score2 := 0.75 * 25;
  let score3 := 0.80 * 20;
  let total_correct := score1 + score2 + score3;
  let total_problems := 15 + 25 + 20;
  let overall_percentage := (total_correct / total_problems) * 100;
  have h: overall_percentage.round = 80 := by sorry;
  exact h

end carlas_overall_score_percentage_l332_332825


namespace three_digit_multiples_of_15_not_70_l332_332483

theorem three_digit_multiples_of_15_not_70 : 
  let is_three_digit := λ n : ℕ, 100 ≤ n ∧ n < 1000
  ∧ is_multiple_of_15 := λ n : ℕ, n % 15 = 0
  ∧ is_multiple_of_70 := λ n : ℕ, n % 70 = 0
  ∧ valid_num := λ n : ℕ, is_three_digit n ∧ is_multiple_of_15 n ∧ ¬is_multiple_of_70 n in
  (count (λ n, valid_num n) (list.range' 100 900) = 56) :=
by sorry

end three_digit_multiples_of_15_not_70_l332_332483


namespace perimeter_of_shaded_region_l332_332112

theorem perimeter_of_shaded_region 
  (C : ℝ) (circumference_eq : C = 48)
  (equilateral_triangle : ∀ (A B C : Type) 
                                    (d : A → B → ℝ), 
                  d A B = d B C ∧ d A B = d A C)
  (arc_angle : ∀ (r : ℝ), arc_length (r : ℝ) → angle = 2 * pi / 3) :
  let r := C / (2 * Real.pi),
        arc_length := (120 / 360) * C in
  3 * arc_length = 48 := 
by 
  sorry

end perimeter_of_shaded_region_l332_332112


namespace quadratic_linear_term_l332_332500

theorem quadratic_linear_term (m : ℝ) 
  (h : 2 * m = 6) : -4 * (x : ℝ) + m * x = -x := by 
  sorry

end quadratic_linear_term_l332_332500


namespace AM_QM_Muirhead_Inequality_l332_332605

open Real

theorem AM_QM_Muirhead_Inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  ((a + b + c) / 3 = sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) ∧
  (sqrt ((a^2 + b^2 + c^2) / 3) = ((ab / c) + (bc / a) + (ca / b)) / 3 ↔ a = b ∧ b = c) :=
by sorry

end AM_QM_Muirhead_Inequality_l332_332605


namespace smaller_angle_at_3_40_l332_332739

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332739


namespace simplify_and_evaluate_expression_l332_332598

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  ((x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l332_332598


namespace smallest_cube_volume_l332_332134

noncomputable def pyramid_height := 15
noncomputable def base_side1 := 9
noncomputable def base_side2 := 12

theorem smallest_cube_volume : ∃ (s : ℕ), s = 15 ∧ s^3 = 3375 :=
by
  use 15
  split
  . rfl
  . exact (by norm_num : 15^3 = 3375)

end smallest_cube_volume_l332_332134


namespace quadrilateral_area_l332_332592

-- Define the problem in Lean 4
def quadrilateral_area_proof (A B C D : ℝ) (AB BC AD DC : ℤ) : Prop :=
∃ (AB BC AD DC : ℝ),
  AB^2 + BC^2 = 25 ∧
  AD^2 + DC^2 = 25 ∧
  AB ≠ AD ∧
  BC ≠ DC ∧
  ∃ S1 S2 : ℝ,
    S1 = 0.5 * (AB * BC) ∧
    S2 = 0.5 * (AD * DC) ∧
    S1 + S2 = 12

-- Lean 4 statement of proof problem
theorem quadrilateral_area (A B C D : ℝ) (h1 : ∠ABC = 90) (h2 : ∠ADC = 90) (AC : ℝ) (H : AC = 5) : 
  quadrilateral_area_proof A B C D :=
sorry

end quadrilateral_area_l332_332592


namespace dot_product_eq_quarter_norm_sum_eq_sqrt10_div2_l332_332163

variables (a b : ℝ^2) -- assuming 2D vectors for simplicity
open Real

-- Assumptions
variables (ha : ‖a‖ = 1)
          (hb : ‖b‖ = 1)
          (h : ‖2 • a + b‖ = sqrt 6)

-- Theorem statements
theorem dot_product_eq_quarter (h : ‖2 • a + b‖ = sqrt 6) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  a.dot b = 1 / 4 :=
sorry

theorem norm_sum_eq_sqrt10_div2 (h : ‖2 • a + b‖ = sqrt 6) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  ‖a + b‖ = sqrt 10 / 2 :=
sorry

end dot_product_eq_quarter_norm_sum_eq_sqrt10_div2_l332_332163


namespace find_x_for_g_inv_l332_332454

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 4 * x + 1

theorem find_x_for_g_inv (x : ℝ) (h : g 3 = x) : g⁻¹ 3 = 3 :=
by
  sorry

end find_x_for_g_inv_l332_332454


namespace measure_angle_EUT_perimeter_PQRSTU_measure_ST_given_PQ_l332_332113

variables {A B C D E F P Q R S T U : Point} 
variables (BC EF : Line)
variables (PQ : ℝ) (PQ_measure : PQ = 6)
variables (side_ABC : ℝ) (side_DEF : ℝ) (side_ABC_measure : side_ABC = 14) (side_DEF_measure : side_DEF = 13)
variables (equilateral_ABC : EquilateralTriangle ABC)
variables (equilateral_DEF : EquilateralTriangle DEF)
variables (parallel_lines : Parallel BC EF)

-- 1. Prove that the measure of angle EUT is 60 degrees
theorem measure_angle_EUT : measure_angle E U T = 60 := sorry

-- 2. Prove that the perimeter of the polygon PQRSTU is 27 cm
theorem perimeter_PQRSTU : perimeter P Q R S T U = 27 := sorry

-- 3. Given PQ = 6 cm, prove that the measure of segment ST is 7 cm
theorem measure_ST_given_PQ : ST = 7 := sorry

end measure_angle_EUT_perimeter_PQRSTU_measure_ST_given_PQ_l332_332113


namespace max_digit_change_l332_332292

def digit_at_pos (n : ℕ) (x : ℕ) (y : ℝ) (digit : ℕ) : ℝ :=
if x = n then 9 / 10 ^ n else y else

theorem max_digit_change {x: ℝ} (h : x = 0.1234567) : 
  (digit_at_pos 1 0 1234567 1 9 = 0.9234567) :=
sorry

end max_digit_change_l332_332292


namespace part_I_part_II_l332_332415

-- Part (I)
theorem part_I (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = - f x) (h_domain : ∀ x, x ∈ ℝ) : 
  (∃ m n : ℝ, ∀ x, f(x) = (n - 2^x) / (2^(x + 1) + m) ∧ f 0 = 0) → 
  (∃ (m n : ℝ), n = 1 ∧ m = 2) := 
sorry

-- Part (II)
theorem part_II (f : ℝ → ℝ) (h_f : ∀ x, f(x) = - (1/2) + 1 / (2^x + 1)) : 
  (∀ x ∈ set.Icc (1 / 2) 3, f(k * x^2) + f (2 * x - 1) > 0) →
  k < -1 :=
sorry

end part_I_part_II_l332_332415


namespace isosceles_trapezoids_diagonals_equal_l332_332657

noncomputable theory

open_locale classical

theorem isosceles_trapezoids_diagonals_equal
  (circle : Type)
  [EuclideanGeometry circle]
  (T1 T2 : Trapezoid circle)
  (h1 : is_isosceles T1)
  (h2 : is_isosceles T2)
  (h_parallel_sides : parallel_sides T1 T2)
  (h_inscribed : inscribed T1 circle ∧ inscribed T2 circle):
  diagonal T1 = diagonal T2 :=
sorry

end isosceles_trapezoids_diagonals_equal_l332_332657


namespace separating_covering_sets_minimal_l332_332551

def isSeparating {α : Type*} (F : set (set α)) (N : set α) : Prop :=
  ∀ x y ∈ N, x ≠ y → ∃ A ∈ F, (A ∩ {x, y}).card = 1

def isCovering {α : Type*} (F : set (set α)) (N : set α) : Prop :=
  ∀ x ∈ N, ∃ A ∈ F, x ∈ A

noncomputable def smallestSeparatingCovering {α : Type*} [decidable_eq α] (N : set α) : ℕ :=
  ⌈nat.log2 N.to_finset.card⌉ + 1

theorem separating_covering_sets_minimal {α : Type*} [decidable_eq α] (n : ℕ) (h : 2 ≤ n) :
  ∃ F : set (set (fin n)), 
    isSeparating F (finset.range n).to_set ∧ 
    isCovering F (finset.range n).to_set ∧ 
    F.to_finset.card = smallestSeparatingCovering (finset.range n).to_set :=
sorry

end separating_covering_sets_minimal_l332_332551


namespace sum_of_solutions_l332_332965

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332965


namespace sqrt_a_add_4b_eq_pm3_l332_332289

theorem sqrt_a_add_4b_eq_pm3
  (a b : ℝ)
  (A_sol : a * (-1) + 5 * (-1) = 15)
  (B_sol : 4 * 5 - b * 2 = -2) :
  (a + 4 * b)^(1/2) = 3 ∨ (a + 4 * b)^(1/2) = -3 := by
  sorry

end sqrt_a_add_4b_eq_pm3_l332_332289


namespace twelfth_is_monday_l332_332070

def days_of_week : Type := {d // d < 7}

def starts_on_friday (d : ℕ) : days_of_week := ⟨d % 7, by linarith [Nat.mod_lt d (by norm_num)]⟩

-- Condition 1: There are exactly 5 Fridays in the month (which has at least 30 days)
def has_five_fridays (days_in_month : ℕ) : Prop := 
  ∃ (start : ℕ), 
    (start % 7 ≠ 5) ∧ 
    (days_in_month > 28 ∧ days_in_month < 32) ∧ -- At least 30 days to have 5 Fridays
    ∃ f, ∀ i, starts_on_friday(start + 7*i) = f → (i < 5)

-- Condition 2: The first day of the month is not a Friday
def first_not_friday (start : ℕ) : Prop := start % 7 ≠ 5

-- Condition 3: The last day of the month is not a Friday
def last_not_friday (start days_in_month : ℕ) : Prop := (start + days_in_month - 1) % 7 ≠ 5
    
theorem twelfth_is_monday (days_in_month : ℕ) (start : ℕ) 
  (h_five_fridays : has_five_fridays days_in_month)
  (h_first_not_friday : first_not_friday start)
  (h_last_not_friday : last_not_friday start days_in_month) : 
  (start + 11) % 7 = 1 :=
sorry

end twelfth_is_monday_l332_332070


namespace find_d_minus_r_l332_332194

theorem find_d_minus_r :
  ∃ d r : ℕ, 1 < d ∧ 1223 % d = r ∧ 1625 % d = r ∧ 2513 % d = r ∧ d - r = 1 :=
by
  sorry

end find_d_minus_r_l332_332194


namespace necklace_bead_condition_l332_332153

theorem necklace_bead_condition (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) :
  (∀ beads : fin n → bool,
    ∀ i j : fin m,
      i ≠ j → 
      (∑ k in finset.range n, if beads (fin.cast_add ((i * n + k) % (m * n))) then 1 else 0) ≠ 
      (∑ k in finset.range n, if beads (fin.cast_add ((j * n + k) % (m * n))) then 1 else 0)) →
  m ≤ n + 1 := by
  sorry

end necklace_bead_condition_l332_332153


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332723

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332723


namespace day_of_12th_l332_332077

theorem day_of_12th (month : Type) [decidable_eq month] [fintype month] 
  (is_friday : month → Prop) (is_first_day : month → Prop) (is_last_day : month → Prop)
  (days : list month) (nth_day : Π (n : ℕ), month)
  (five_fridays : ∃ (n : ℕ), (is_friday ∘ nth_day) '' (finset.range (min n 7)) = {5})
  (not_first_day_friday : ∀ d, is_first_day d → ¬ is_friday d)
  (not_last_day_friday : ∀ d, is_last_day d → ¬ is_friday d) : 
  nth_day 12 = "Monday" := 
sorry

end day_of_12th_l332_332077


namespace angle_between_vectors_l332_332469

variables (a b : EuclideanSpace ℝ (Fin 3))

theorem angle_between_vectors (h₁ : inner a (a - b) = 2) (h₂ : ∥a∥ = 1) (h₃ : ∥b∥ = 2) :
  real.angle_between a b = (2 * real.pi) / 3 :=
by
  sorry

end angle_between_vectors_l332_332469


namespace probability_f_ge_0_l332_332450

noncomputable def f (k x : ℝ) : ℝ := k * x + 1

theorem probability_f_ge_0 (k : ℝ) (k_in_range : k ∈ Set.Icc (-2 : ℝ) 1): 
  (Set.Probability (λ k, ∀ x ∈ Set.Icc (0 : ℝ) 1, f k x ≥ 0) {k | k ∈ Set.Icc (-2 : ℝ) 1}) = 2 / 3 :=
by
  sorry

end probability_f_ge_0_l332_332450


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332883

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332883


namespace abs_diff_probs_l332_332505

def numRedMarbles := 1000
def numBlackMarbles := 1002
def totalMarbles := numRedMarbles + numBlackMarbles

def probSame : ℚ := 
  ((numRedMarbles * (numRedMarbles - 1)) / 2 + (numBlackMarbles * (numBlackMarbles - 1)) / 2) / (totalMarbles * (totalMarbles - 1) / 2)

def probDiff : ℚ :=
  (numRedMarbles * numBlackMarbles) / (totalMarbles * (totalMarbles - 1) / 2)

theorem abs_diff_probs : |probSame - probDiff| = 999 / 2003001 := 
by {
  sorry
}

end abs_diff_probs_l332_332505


namespace recolor_all_black_l332_332154

theorem recolor_all_black (n : ℕ) : 
  ∃ (c : ℕ → bool), (∀ i : ℕ, i ∈ (finset.range (n+1)) → c i = tt) :=
by
  sorry

end recolor_all_black_l332_332154


namespace P_sufficient_but_not_necessary_for_Q_l332_332549

variables {x y : ℝ}

def P : Prop := x + y ≠ 5
def Q : Prop := x ≠ 2 ∨ y ≠ 3

theorem P_sufficient_but_not_necessary_for_Q : (P → Q) ∧ (∃ x y : ℝ, P ∧ ¬Q) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l332_332549


namespace max_area_garden_side_l332_332798

open Real

theorem max_area_garden_side (x : ℝ) :
  (x : ℝ ≥ 0) →
  (10 * (2 * x + (200 - 2 * x)) = 2000) →
  let area : ℝ := x * (200 - 2 * x) in
  ∃ (s : ℝ), (s = 200 - 2 * x) ∧ (x = 50) ∧ (s = 100) :=
by
  intros hx hcost
  let area : ℝ := x * (200 - 2 * x)
  have h_area : area = 200 * x - 2 * x^2 := by sorry
  have quad_form : ∃! (x : ℝ), (200 * x - 2 * x^2) = area := by sorry
  have max_x : x = 50 := by sorry
  use 200 - 2 * x
  split
  case left => exact rfl
  case right =>
    split
    exact max_x
    rw [max_x]
    norm_num
    exact rfl

end max_area_garden_side_l332_332798


namespace solve_for_x_l332_332046

variables (x y z : ℝ)

def condition : Prop :=
  1 / (x + y) + 1 / (x - y) = z / (x - y)

theorem solve_for_x (h : condition x y z) : x = z / 2 :=
by
  sorry

end solve_for_x_l332_332046


namespace teacher_total_score_l332_332336

/-- Given the teacher's scores in the written test and interview, and the respective weights of these components, the total score is 72 points. -/
theorem teacher_total_score :
  ∀ (w_score i_score : ℕ) (w_weight i_weight : ℝ),
  w_score = 80 →
  w_weight = 0.6 →
  i_score = 60 →
  i_weight = 0.4 →
  (w_score * w_weight + i_score * i_weight) = 72 :=
by
  intros w_score i_score w_weight i_weight h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end teacher_total_score_l332_332336


namespace sum_of_solutions_eqn_l332_332945

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332945


namespace number_of_scenarios_correct_l332_332514

noncomputable def number_of_scenarios : ℕ := 
  let case1 := (nat.choose 8 2) * (nat.choose 6 2) * (nat.choose 4 4) / 2 * 6
  let case2 := (nat.choose 8 2) * (nat.choose 6 3) * (nat.choose 3 3) / 2 * 6
  case1 + case2

theorem number_of_scenarios_correct : 
  number_of_scenarios = 2940 := 
by
  sorry

end number_of_scenarios_correct_l332_332514


namespace fish_count_l332_332248

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l332_332248


namespace sum_of_solutions_l332_332955
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332955


namespace range_of_a_l332_332021

noncomputable def proof_problem (x : ℝ) (a : ℝ) : Prop :=
  (x^2 - 4*x + 3 < 0) ∧ (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + a < 0)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, proof_problem x a) ↔ a ≤ 9 :=
by
  sorry

end range_of_a_l332_332021


namespace greatest_ratio_is_2_l332_332371

/-- Define the circle of radius 4 -/
def circle (x y : ℤ) : Prop := x^2 + y^2 = 16

/-- Predicate for points on the circle -/
def on_circle (p : ℤ × ℤ) : Prop := circle p.1 p.2

/-- Predicate for an irrational distance between two points -/
def irr_dist (p1 p2 : ℤ × ℤ) : Prop :=
  ¬ ∃ (k : ℤ), (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 = k^2

/-- The problem statement -/
theorem greatest_ratio_is_2 :
  ∃ (A B C D : ℤ × ℤ),
  on_circle A ∧ on_circle B ∧ on_circle C ∧ on_circle D ∧
  irr_dist A B ∧ irr_dist C D ∧
  (∀ R, (∃ A' B' C' D' : ℤ × ℤ, on_circle A' ∧ on_circle B' ∧ on_circle C' ∧ on_circle D' ∧ irr_dist A' B' ∧ irr_dist C' D') → 
        R = dist A B / dist C D) ∧
  dist A B / dist C D = 2 := sorry

open Real

/-- Define the distance formula -/
def dist (p1 p2 : ℤ × ℤ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

end greatest_ratio_is_2_l332_332371


namespace find_x_l332_332393

noncomputable def x : ℝ := 74

theorem find_x (h : sqrt (x + 7) + 5 = 14) : x = 74 :=
by
  sorry

end find_x_l332_332393


namespace rectangle_y_value_l332_332630

theorem rectangle_y_value 
  (y : ℝ)
  (A : (0, 0) = E ∧ (0, 5) = F ∧ (y, 5) = G ∧ (y, 0) = H)
  (area : 5 * y = 35)
  (y_pos : y > 0) :
  y = 7 :=
sorry

end rectangle_y_value_l332_332630


namespace sum_of_solutions_l332_332969

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332969


namespace appropriate_sampling_methods_l332_332260
-- Import the entire Mathlib library for broader functionality

-- Define the conditions
def community_high_income_families : ℕ := 125
def community_middle_income_families : ℕ := 280
def community_low_income_families : ℕ := 95
def community_total_households : ℕ := community_high_income_families + community_middle_income_families + community_low_income_families

def student_count : ℕ := 12

-- Define the theorem to be proven
theorem appropriate_sampling_methods :
  (community_total_households = 500 → stratified_sampling) ∧
  (student_count = 12 → random_sampling) :=
by sorry

end appropriate_sampling_methods_l332_332260


namespace twelve_is_monday_l332_332088

def Weekday := {d : String // d ∈ ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

def not_friday (d: Weekday) : Prop := d.val ≠ "Friday"

def has_exactly_five_fridays (first_friday: nat) (days_in_month: nat) : Prop :=
  first_friday + 28 <= days_in_month ∧
  first_friday + 21 <= days_in_month ∧
  first_friday + 14 <= days_in_month ∧
  first_friday + 7 <= days_in_month ∧
  first_friday > 0 ∧ days_in_month <= 31

noncomputable def compute_day_of_week (start_day: String) (n: nat) : Weekday :=
  sorry

theorem twelve_is_monday (start_day: Weekday) (days_in_month: nat) :
    has_exactly_five_fridays 2 days_in_month
  → not_friday start_day
  → not_friday (compute_day_of_week start_day.val days_in_month)
  → compute_day_of_week start_day.val 12 = ⟨"Monday", by sorry⟩ :=
begin
  sorry
end

end twelve_is_monday_l332_332088


namespace find_z_l332_332447

-- Define the complex number z and its condition
def satisfies_condition (z : ℂ) : Prop :=
  z * (1 - complex.I) = 2 * complex.I

-- Define the correct answer
def correct_answer (z : ℂ) : Prop :=
  z = -1 + complex.I

-- The theorem states that the value of z satisfying the condition is indeed the correct answer
theorem find_z (z : ℂ) (h : satisfies_condition z) : correct_answer z :=
sorry

end find_z_l332_332447


namespace followers_after_one_year_l332_332851

theorem followers_after_one_year :
  let initial_followers := 100000
  let daily_new_followers := 1000
  let unfollowers_per_year := 20000
  let days_per_year := 365
  initial_followers + (daily_new_followers * days_per_year - unfollowers_per_year) = 445000 :=
by
  sorry

end followers_after_one_year_l332_332851


namespace small_angle_at_3_40_is_130_degrees_l332_332758

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332758


namespace conditional_probability_l332_332776

variable (Ω : Type) [ProbabilitySpace Ω]
variable (g h : Event Ω)

theorem conditional_probability (qg qh qgh : ℝ) (h0 : 0 ≤ qg) (h1 : qg ≤ 1) (h2 : 0 ≤ qh) (h3 : qh ≤ 1) 
  (h4 : 0 ≤ qgh) (h5 : qgh ≤ 1) (Hqg : q(g) = qg) (Hqh : q(h) = qh) (Hqgh : q(g ∩ h) = qgh) :
  qh ≠ 0 → q(g | h) = qgh / qh ∧ q(¬ g | h) = 1 - (qgh / qh) := by
  sorry

-- Specific instance
example : conditional_probability Ω g h 0.30 0.9 0.9 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) 
  (by rfl) (by rfl) (by rfl) (by norm_num) := by
  sorry

end conditional_probability_l332_332776


namespace smaller_angle_3_40_pm_l332_332746

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332746


namespace twelfth_day_is_monday_l332_332063

def Month := ℕ
def Day := ℕ

-- Definitions for days of the week, where 0 represents Monday, 1 represents Tuesday, etc.
inductive Weekday : Type
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
| Sunday : Weekday

open Weekday

-- A month has exactly 5 Fridays
def has_five_fridays (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday ∧
    (∃ last_day: Weekday, last_day ≠ Friday ∧ 
      let fridays := List.filter (λ d, d = Friday) (List.range 31) in
      fridays.length = 5)

-- The first day of the month is not a Friday
def first_day_not_friday (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday

-- The last day of the month is not a Friday
def last_day_not_friday (month: Month): Prop :=
  ∀ last_day: Weekday, last_day = (29 % 7) → last_day ≠ Friday

-- Combining the conditions for the problem
def valid_month (month: Month): Prop :=
  has_five_fridays(month) ∧ first_day_not_friday(month) ∧ last_day_not_friday(month)

-- Prove that the 12th day of the month is a Monday given the conditions
theorem twelfth_day_is_monday (month: Month) (h: valid_month(month)): (∃ starts_on: Weekday, starts_on + 11 = Monday) :=
sorry

end twelfth_day_is_monday_l332_332063


namespace exactly_one_is_multiple_of_5_l332_332048

theorem exactly_one_is_multiple_of_5 (a b : ℤ) (h: 24 * a^2 + 1 = b^2) : 
  (∃ k : ℤ, a = 5 * k) ∧ (∀ l : ℤ, b ≠ 5 * l) ∨ (∃ m : ℤ, b = 5 * m) ∧ (∀ n : ℤ, a ≠ 5 * n) :=
sorry

end exactly_one_is_multiple_of_5_l332_332048


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332722

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332722


namespace sum_of_solutions_l332_332970

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332970


namespace tangent_line_equation_l332_332448

noncomputable def g : ℝ → ℝ := sorry
def g' (x : ℝ) : ℝ := sorry

theorem tangent_line_equation (g' : ℝ → ℝ) (g'1_eq_2 : g'(1) = 2) (g1_eq_3 : g(1) = 3) :
  let f (x : ℝ) := g(x) + x^2
  ∧ f'(x : ℝ) := (g' x) + 2 * x
  ∧ tangent_line (f : ℝ → ℝ) := (λ x, 4 * x) in
    tangent_line f = (λ x, 4 * x) :=
by
  intro f f' tangent_line
  have f := (λ x, g(x) + x^2)
  have f' := (λ x, (g' x) + 2 * x)
  have tangent_line := (λ x, 4 * x)
  sorry

end tangent_line_equation_l332_332448


namespace absolute_difference_center_l332_332628

theorem absolute_difference_center (x1 y1 x2 y2 : ℝ) 
 (h1: x1 = 8) (h2: y1 = -7) (h3: x2 = -4) (h4: y2 = 5) : 
|((x1 + x2) / 2 - (y1 + y2) / 2)| = 3 :=
by
  sorry

end absolute_difference_center_l332_332628


namespace initials_count_l332_332471

-- Let L be the set of letters {A, B, C, D, E, F, G, H, I, J}
def L : finset char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Each initial can be any element in L
def initials : Type := L × L × L

-- We need to prove that the number of different three-letter sets of initials is equal to 1000
theorem initials_count : finset.card (finset.product L (finset.product L L)) = 1000 := 
sorry

end initials_count_l332_332471


namespace optimal_screen_placement_area_l332_332179

theorem optimal_screen_placement_area :
  let l := 4 in
  let angle := 45 in
  let enclosed_area := 8 * (Real.sqrt 2 + 1) in
  (PartitionedCornerArea l angle = enclosed_area) :=
sorry

end optimal_screen_placement_area_l332_332179


namespace total_animal_eyes_l332_332097

def frogs_in_pond := 20
def crocodiles_in_pond := 6
def eyes_per_frog := 2
def eyes_per_crocodile := 2

theorem total_animal_eyes : (frogs_in_pond * eyes_per_frog + crocodiles_in_pond * eyes_per_crocodile) = 52 := by
  sorry

end total_animal_eyes_l332_332097


namespace sum_of_solutions_sum_of_all_solutions_l332_332909

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332909


namespace length_of_BD_l332_332176

theorem length_of_BD (c : ℝ) (BC : ℝ) (AC : ℝ) (AD : ℝ) (AB : ℝ) (BD : ℝ) :
  BC = 3 → AC = c → AD = c - 1 → (AB = real.sqrt (AC^2 + BC^2))
  → (AB = real.sqrt (AD^2 + BD^2)) → BD = real.sqrt (2*c + 8) :=
by
  intros hBC hAC hAD hAB1 hAB2
  have h1 : AB^2 = AC^2 + BC^2, from calc
    AB^2 = (real.sqrt (AC^2 + BC^2))^2 : by rw hAB1
    ...   = AC^2 + BC^2               : by rw real.sq_sqrt (add_nonneg (sq_nonneg AC) (sq_nonneg BC))
  have h2 : AB^2 = (c - 1)^2 + BD^2, from calc
    AB^2 = (real.sqrt (AD^2 + BD^2))^2 : by rw hAB2
    ...   = AD^2 + BD^2               : by rw real.sq_sqrt (add_nonneg (sq_nonneg AD) (sq_nonneg BD))
  rw [hAC, hBC, hAD] at h1,
  rw [h1, h2] at hAB1,
  sorry

end length_of_BD_l332_332176


namespace incorrect_score_modulo_l332_332826

theorem incorrect_score_modulo (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9) 
  (hb : 0 ≤ b ∧ b ≤ 9) 
  (hc : 0 ≤ c ∧ c ≤ 9) : 
  ∃ remainder : ℕ, remainder = (90 * a + 9 * b + c) % 9 ∧ 0 ≤ remainder ∧ remainder ≤ 9 := 
by
  sorry

end incorrect_score_modulo_l332_332826


namespace q_multiplied_factor_l332_332052

variable (e x z : ℝ)

def q (e x z : ℝ) : ℝ := (5 * e) / (4 * x * (z^2))

theorem q_multiplied_factor (h1 : e' = 4 * e) (h2 : x' = 2 * x) (h3 : z' = 3 * z) :
  q e' x' z' = (4 / 9) * (q e x z) :=
by
  sorry

end q_multiplied_factor_l332_332052


namespace rectangle_coord_eq_trajectory_and_max_area_l332_332629

-- Definition of the given conditions and problem
theorem rectangle_coord_eq_trajectory_and_max_area (x y : ℝ) (A B : ℝ × ℝ) :
  (let O := (0, 0)
   let C1 := {p : ℝ × ℝ | p.1 = 4}
   let OM := dist O M
   let OP := dist O P
   let rho_cos_theta := dist O (4, y)
    ρ * cos θ = 4 ∧
    OM * OP = 16 ∧
    A = (1, √3) ∧
    B ∈ C2) →
  (x - 2)^2 + y^2 = 4 ∧ 
  (d = sqrt(4 - 1) → max_area_triangle AOB = 2 + sqrt(3)) :=
by
  sorry

--Note: this is a single theorem combining both parts 1 and 2 

end rectangle_coord_eq_trajectory_and_max_area_l332_332629


namespace sum_of_solutions_equation_l332_332934

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332934


namespace probability_first_three_same_color_l332_332801

/-- A standard deck of 52 cards is divided into 4 suits, with each suit containing 13 cards.
    Two of these suits are red, and the other two are black. The deck is shuffled, placing the cards
    in random order. Prove that the probability that the first three cards drawn from the deck are 
    all the same color is 40/85. -/
theorem probability_first_three_same_color :
  let total_cards := 52
  let cards_per_suit := 13
  let red_suits := 2
  let black_suits := 2
  let red_cards := red_suits * cards_per_suit
  let black_cards := black_suits * cards_per_suit
  in (red_cards / total_cards) * ((red_cards - 1) / (total_cards - 1)) * ((red_cards - 2) / (total_cards - 2)) +
     (black_cards / total_cards) * ((black_cards - 1) / (total_cards - 1)) * ((black_cards - 2) / (total_cards - 2)) = 40 / 85 := 
by
  sorry

end probability_first_three_same_color_l332_332801


namespace floor_sum_product_l332_332863

theorem floor_sum_product : 
  Int.floor 13.7 + Int.floor (-13.7) + (Int.floor 2.5 * Int.floor (-4.3)) = -11 := by
  -- Given conditions
  have h1 : Int.floor 13.7 = 13 := by sorry
  have h2 : Int.floor (-13.7) = -14 := by sorry
  have h3 : Int.floor 2.5 = 2 := by sorry
  have h4 : Int.floor (-4.3) = -5 := by sorry
  
  -- Applying conditions to the proof
  rw [h1, h2, h3, h4]
  norm_num
  trivial

end floor_sum_product_l332_332863


namespace points_labeling_l332_332990

theorem points_labeling (n : ℕ) (points : Fin n → Point) :
  (∀ (i j k : Fin n), i ≠ j ∧ i ≠ k ∧ j ≠ k → ∃ α β γ : ℝ, α + β + γ = 180 ∧ 
  (α > 120 ∨ β > 120 ∨ γ > 120)) →
  ∃ (label : Fin n → Fin n), ∀ (i j k : Fin n), i < j ∧ j < k → angle label i label j label k > 120 :=
sorry

end points_labeling_l332_332990


namespace swimmer_path_min_time_l332_332804

theorem swimmer_path_min_time (k : ℝ) :
  (k > Real.sqrt 2 → ∀ x y : ℝ, x = 0 ∧ y = 0 ∧ t = 2/k) ∧
  (k < Real.sqrt 2 → x = 1 ∧ y = 1 ∧ t = Real.sqrt 2) ∧
  (k = Real.sqrt 2 → ∀ x y : ℝ, x = y ∧ t = (1 / Real.sqrt 2) + Real.sqrt 2 + (1 / Real.sqrt 2)) :=
by sorry

end swimmer_path_min_time_l332_332804


namespace initial_stock_is_1200_l332_332135

def initial_stock_of_books (X : ℝ) :=
  let sold_total := 75 + 50 + 64 + 78 + 135
  let percent_sold := 33.5 / 100
  let eq := percent_sold * X = sold_total
  eq

theorem initial_stock_is_1200 : initial_stock_of_books 1200 :=
by
  unfold initial_stock_of_books
  have sold_total := (75:ℝ) + 50 + 64 + 78 + 135
  have percent_sold := 33.5 / 100
  have eq : percent_sold * 1200 = sold_total := sorry
  exact eq

end initial_stock_is_1200_l332_332135


namespace only_setC_can_form_triangle_l332_332296

-- Define a predicate to check if three given lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the given sets of line segments
def setA := (2 : ℝ, 2 : ℝ, 5 : ℝ)
def setB := (3 : ℝ, 5 : ℝ, 8 : ℝ)
def setC := (4 : ℝ, 4 : ℝ, 7 : ℝ)
def setD := (4 : ℝ, 5 : ℝ, 10 : ℝ)

-- Define the theorem stating which set can form a triangle
theorem only_setC_can_form_triangle :
  can_form_triangle setC.1 setC.2 setC.2.2 ∧
  ¬can_form_triangle setA.1 setA.2 setA.2.2 ∧
  ¬can_form_triangle setB.1 setB.2 setB.2.2 ∧
  ¬can_form_triangle setD.1 setD.2 setD.2.2 :=
by
  sorry

end only_setC_can_form_triangle_l332_332296


namespace polar_eq_of_parametric_curve_l332_332515

theorem polar_eq_of_parametric_curve 
    (x y : ℝ)
    (α : ℝ)
    (h1 : x = cos α)
    (h2 : y = sin α + 1) : 
    ∃ θ ρ, (x = ρ * cos θ ∧ y = ρ * sin θ) ∧ ρ = 2 * sin θ := 
by 
  sorry

end polar_eq_of_parametric_curve_l332_332515


namespace distinct_digit_product_inequality_l332_332824

open Finset

theorem distinct_digit_product_inequality :
  ∀ (K O T U CH E H Υ J : ℕ),
    {K, O, T, U, CH, E, H, Υ, J} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} →
    {K, O, T, U, CH, E, H, Υ, J}.card = 9 →
    K * O * T ≠ U * CH * E * H * Υ * J :=
by
  intros K O T U CH E H Υ J hsubset hcard
  -- Maximum possible value for K * O * T
  have hleft_max : K * O * T ≤ 7 * 8 * 9 := sorry
  -- Minimum possible value for U * CH * E * H * Υ * J
  have hright_min : U * CH * E * H * Υ * J ≥ 1 * 2 * 3 * 4 * 5 * 6 := sorry
  -- Conclude the inequality by comparing max of the left and min of the right
  exact sorry

end distinct_digit_product_inequality_l332_332824


namespace original_sheets_count_is_115_l332_332810

def find_sheets_count (S P : ℕ) : Prop :=
  -- Ann's condition: all papers are used leaving 100 flyers
  S - P = 100 ∧
  -- Bob's condition: all bindings used leaving 35 sheets of paper
  5 * P = S - 35

theorem original_sheets_count_is_115 (S P : ℕ) (h : find_sheets_count S P) : S = 115 :=
by
  sorry

end original_sheets_count_is_115_l332_332810


namespace find_equation_of_ellipse_intercept_product_constant_l332_332998

constant a b x y : ℝ
constant A B P M N : ℝ → ℝ → Prop
constant O : ℝ → ℝ

-- Given conditions
axiom ellipse_def : ∀ (x y a b : ℝ), (a > b) → (0 < b) → ((x/a)^2 + (y/b)^2 = 1)
axiom eccentricity_def : ∀ (a b c : ℝ), (c = a/2) → (a > 0) → (b > 0)
axiom max_distance : ∀ (a c : ℝ), (c = a/2) → (a + c = 3)

-- Prove that
theorem find_equation_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : 0 < b) (h3 : c = a/2) (h4 : a + c = 3) : 
  (a = 2) ∧ (b = √3) → (∀ x y : ℝ, (x^2)/4 + (y^2)/3 = 1) :=
begin
  sorry
end

-- Definitions for Part (2)
def symmetric_about_x (A B : ℝ → ℝ → Prop) : Prop := 
  ∀ x y : ℝ, A x y → B x (-y)

def intersect_x_axis (P : ℝ → ℝ → Prop) (x_intercept : ℝ) : Prop :=
  ∃ x y : ℝ, P x y ∧ y = 0 ∧ x = x_intercept

def ellipse_points (ellipse : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, ellipse x y → (x^2)/4 + (y^2)/3 = 1

-- Prove that the product of the intercepts is constant
theorem intercept_product_constant (ellipse : ℝ → ℝ → Prop) (A B : ℝ → ℝ → Prop) 
  (P : ℝ → ℝ → Prop) (xM xN : ℝ) (hA : ellipse_points A) (hB : symmetric_about_x A B)
  (hP : ellipse_points P) (hM : intersect_x_axis (λ x y, A x y ∧ P x y) xM)
  (hN : intersect_x_axis (λ x y, B x y ∧ P x y) xN) :
  ∀ x1 y1 x2 y2 : ℝ, (xM * xN = 4) :=
begin
  sorry
end

end find_equation_of_ellipse_intercept_product_constant_l332_332998


namespace polynomial_binom_expansion_evaluation_l332_332412

notation "^" => Monoid.npow

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem polynomial_binom_expansion_evaluation :
  let a := 1
  let b := -10
  let c := 40
  let d := -80
  let e := 80
  let f := -32
  16 * (a + b) + 4 * (c + d) + (e + f) = -256 :=
by
  -- Let x be a real number
  let x := (λ n : ℕ, (x-2)^n)
  
  -- Define the expansion of (x-2)^5 using binomial theorem
  let poly_exp := ∑ k in Finset.range 6, binom 5 k * (x)^(5-k) * (-2)^k

  -- Prove our expression given a, b, c, d, e, and f
  have expand_eq : poly_exp = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f, by
    sorry -- Expansion using binomial theorem

  -- Use the coefficients to prove our final statement
  have coefficients : a = 1 ∧ b = -10 ∧ c = 40 ∧ d = -80 ∧ e = 80 ∧ f = -32, by
    sorry -- Coefficients from expanded form

  -- Simplify final expression
  have result : 16 * (a + b) + 4 * (c + d) + (e + f) = -256, by
    sorry -- Combine and compute the simplified expression

  exact result

end polynomial_binom_expansion_evaluation_l332_332412


namespace clock_angle_3_40_l332_332717

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332717


namespace part1_part2_l332_332425

-- Arithmetic sequence with first term a_1 = 3 and a positive common ratio
variables {α : Type*} [linear_ordered_field α]

def a1 : α := 3
def q : α := 1 / 2
def an (n : ℕ) : α := if n = 0 then a1 else 6 * q^n
def Sn (n : ℕ) : α := ∑ i in finset.range n, an i

-- We are given that S3 + a3, S5 + a5, S4 + a4 form an arithmetic sequence.
def condition (S3 S4 S5 a3 a4 a5 : α) : Prop :=
  2 * (S5 + a5) = S3 + a3 + S4 + a4

-- Define b_n as given in the problem
def bn (n : ℕ) : α := n * an n / 6

-- Define the sum of the first n terms Tn of the sequence {b_n}
def Tn (n : ℕ) : α := ∑ i in finset.range n, bn i

theorem part1 (n : ℕ) : an n = 6 * (1 / 2)^n := sorry

theorem part2 (n : ℕ) : Tn n = 2 - (n + 2) * (1 / 2)^n := sorry

end part1_part2_l332_332425


namespace monotonic_decreasing_interval_l332_332622

theorem monotonic_decreasing_interval:
  (∀ x : ℝ, x ≤ 3 → (x - 3)^2 - 4 = x^2 - 6x + 5) →
  (∀ x y : ℝ, x ≤ y → y ≤ 3 → (x - 3)^2 - 4 ≥ (y - 3)^2 - 4) :=
begin
  sorry
end

end monotonic_decreasing_interval_l332_332622


namespace parallel_lines_m_value_l332_332467

/-- Given two lines x + m * y + 6 = 0 and (m - 2) * x + 3 * y + 2 * m = 0 are parallel,
    prove that the value of the real number m that makes the lines parallel is -1. -/
theorem parallel_lines_m_value (m : ℝ) : 
  (x + m * y + 6 = 0 ∧ (m - 2) * x + 3 * y + 2 * m = 0 → 
  (m = -1)) :=
by
  sorry

end parallel_lines_m_value_l332_332467


namespace angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l332_332059

-- Definitions of the sides and conditions in triangle
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: a + b = 6
axiom sum_of_sides : a + b = 6

-- Condition: Area of triangle ABC is 2 * sqrt(3)
axiom area_of_triangle : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3

-- Condition: a cos B + b cos A = 2c cos C
axiom cos_condition : (a * Real.cos B + b * Real.cos A) / c = 2 * Real.cos C

-- Proof problem 1: Prove that C = π/3
theorem angle_C_is_pi_div_3 (h_cos : Real.cos C = 1/2) : C = Real.pi / 3 :=
sorry

-- Proof problem 2: Prove that c = 2 sqrt(3)
theorem side_c_is_2_sqrt_3 (h_sin : Real.sin C = Real.sqrt 3 / 2) : c = 2 * Real.sqrt 3 :=
sorry

end angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l332_332059


namespace equal_central_angles_equal_arcs_three_points_determine_circle_l332_332771

theorem equal_central_angles_equal_arcs 
  (c : Circle) (θ₁ θ₂ : Angle) (h : θ₁ = θ₂) :
  arc_of c θ₁ = arc_of c θ₂ := 
sorry

theorem three_points_determine_circle 
  (A B C: Point) 
  (h1 : A ≠ B) 
  (h2 : B ≠ C) 
  (h3 : A ≠ C) 
  (h4 : ¬ collinear A B C) : 
  ∃! c : Circle, passes_through A c ∧ passes_through B c ∧ passes_through C c := 
sorry

end equal_central_angles_equal_arcs_three_points_determine_circle_l332_332771


namespace rectangle_dimension_area_l332_332274

theorem rectangle_dimension_area (x : ℝ) 
  (h_dim : (3 * x - 5) * (x + 7) = 14 * x - 35) : 
  x = 0 :=
by
  sorry

end rectangle_dimension_area_l332_332274


namespace ellipse_problem_l332_332449

noncomputable def ellipse_equation (a b x y : ℝ) : Prop :=
  (a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1))

noncomputable def point_on_ellipse (a x y : ℝ) : Prop :=
  (x = √3 ∧ y = 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1))

noncomputable def ellipse_eccentricity (a c : ℝ) : Prop :=
  (c = √2 ∧ (c / a = √6 / 3))

noncomputable def line_intersects_ellipse (k m : ℝ) : Prop :=
  ((3*k^2 + 1)*x^2 + 6*k*m*x + 3*(m^2 - 1) = 0)

noncomputable def midpoint_on_line (k x1 x2 y1 y2 m : ℝ) : Prop :=
  (x1 + x2 = 2 ∨ x1 + x2 = -2)

noncomputable def range_of_k : Set ℝ :=
  (-∞, -√6/6) ∪ (√6/6, +∞)

theorem ellipse_problem
  (a b c x y k m x1 x2 y1 y2 : ℝ) :
  ellipse_equation a b x y →
  point_on_ellipse a x y →
  ellipse_eccentricity a c →
  line_intersects_ellipse k m →
  midpoint_on_line k x1 x2 y1 y2 m →
  (b^2 = 1 ∧ k ∈ range_of_k) := 
sorry

end ellipse_problem_l332_332449


namespace dihedral_angle_D_AC_B_90_l332_332800

variables (A B C D : ℝ) -- assuming we are working in a coordinate space

-- Given conditions
def is_square_ABC : Prop := (B - A) = (C - B) ∧ (D - C) = (A - D) ∧ (B - A) ^ 2 = a ^ 2
def fold_along_AC : Prop := ∀ D', D' = D / 2
def DAB_angle_60 : Prop := angle D A B = 60

-- The main theorem to be proven
theorem dihedral_angle_D_AC_B_90 (a : ℝ) (A B C D : ℝ) (h1 : is_square_ABC A B C D) 
(h2 : fold_along_AC D) (h3 : DAB_angle_60 A B D) : 
dihedral_angle D A C B = 90 :=
sorry

end dihedral_angle_D_AC_B_90_l332_332800


namespace sum_of_solutions_l332_332892

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332892


namespace number_of_adults_had_meal_l332_332326

-- Definitions based on conditions
def total_adults : ℕ := 55
def total_children : ℕ := 70
def meal_for_adults : ℕ := 70
def meal_for_children : ℕ := 90
def remaining_children_meal : ℕ := 45

-- The ratio based on the problem conditions
def adults_children_ratio : ℕ → ℚ := λ n, (9 / 7) * n

-- The proof statement
theorem number_of_adults_had_meal :
  ∃ (a : ℕ), adults_children_ratio a = 45 ∧ a = 35 := by
  -- Proof to be filled in
  sorry

end number_of_adults_had_meal_l332_332326


namespace initial_number_of_earning_members_l332_332198

theorem initial_number_of_earning_members (n : ℕ) 
  (h1 : 840 * n - 650 * (n - 1) = 1410) : n = 4 :=
by {
  -- Proof omitted
  sorry
}

end initial_number_of_earning_members_l332_332198


namespace count_valid_sequences_l332_332485

def even_digits_set : Finset ℕ := {0, 2, 4, 6, 8}
def odd_digits_set : Finset ℕ := {1, 3, 5, 7, 9}

theorem count_valid_sequences :
  let count := (∏ i in Finset.range 7, if i % 2 = 0 then 5 else 5) * 5;
  count = 390625 :=
by
  let count := (∏ i in Finset.range 7, if i % 2 = 0 then 5 else 5) * 5;
  exact sorry

end count_valid_sequences_l332_332485


namespace cube_vertices_l332_332773

theorem cube_vertices : ∀ (s : Type) [cube : Cube s], number_of_vertices s = 8 := by
  sorry

end cube_vertices_l332_332773


namespace minimum_equilateral_triangles_l332_332285

theorem minimum_equilateral_triangles (side_small : ℝ) (side_large : ℝ)
  (h_small : side_small = 1) (h_large : side_large = 15) :
  225 = (side_large / side_small)^2 :=
by
  -- Proof is skipped.
  sorry

end minimum_equilateral_triangles_l332_332285


namespace smaller_angle_3_40_l332_332694

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332694


namespace sum_of_solutions_eq_16_l332_332928

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332928


namespace smaller_angle_between_hands_at_3_40_l332_332686

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332686


namespace third_term_binomial_expansion_l332_332765

-- Let a, x be real numbers
variables (a x : ℝ)

-- Binomial theorem term for k = 2
def binomial_term (n k : ℕ) (x y : ℝ) : ℝ :=
  (Nat.choose n k) * x^(n-k) * y^k

theorem third_term_binomial_expansion :
  binomial_term 6 2 (a / Real.sqrt x) (-Real.sqrt x / a^2) = 15 / x :=
by
  sorry

end third_term_binomial_expansion_l332_332765


namespace sum_of_solutions_eq_16_l332_332972

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332972


namespace max_N_when_k_is_4_l332_332140

variables (n : ℕ) (S : Finset (ℝ × ℝ))

def inner_product (v v' : ℝ × ℝ) :=
  v.1 * v'.1 + v.2 * v'.2

def length_squared (v : ℝ × ℝ) :=
  v.1^2 + v.2^2

def condition (v v' : ℝ × ℝ) : Prop :=
  4 * (inner_product v v') + (length_squared v - 1) * (length_squared v' - 1) < 0

def N (S : Finset (ℝ × ℝ)) : ℕ :=
  ((S.subsets 2).filter (λ t, match t.to_list with
                              | [v, v'] := condition v v'
                              | _        := false
                            end)).card

theorem max_N_when_k_is_4 : ∀ (S : Finset (ℝ × ℝ)) (h : S.card = n), N S ≤ (Finset.univ : Finset (Finset (ℝ × ℝ))) = 6 :=
sorry

end max_N_when_k_is_4_l332_332140


namespace number_of_integers_satisfying_inequality_l332_332030

theorem number_of_integers_satisfying_inequality (S : set ℤ) :
  (S = {x : ℤ | |7 * x - 5| ≤ 9}) →
  S.card = 3 :=
by
  intro hS
  sorry

end number_of_integers_satisfying_inequality_l332_332030


namespace twelfth_is_monday_l332_332069

def days_of_week : Type := {d // d < 7}

def starts_on_friday (d : ℕ) : days_of_week := ⟨d % 7, by linarith [Nat.mod_lt d (by norm_num)]⟩

-- Condition 1: There are exactly 5 Fridays in the month (which has at least 30 days)
def has_five_fridays (days_in_month : ℕ) : Prop := 
  ∃ (start : ℕ), 
    (start % 7 ≠ 5) ∧ 
    (days_in_month > 28 ∧ days_in_month < 32) ∧ -- At least 30 days to have 5 Fridays
    ∃ f, ∀ i, starts_on_friday(start + 7*i) = f → (i < 5)

-- Condition 2: The first day of the month is not a Friday
def first_not_friday (start : ℕ) : Prop := start % 7 ≠ 5

-- Condition 3: The last day of the month is not a Friday
def last_not_friday (start days_in_month : ℕ) : Prop := (start + days_in_month - 1) % 7 ≠ 5
    
theorem twelfth_is_monday (days_in_month : ℕ) (start : ℕ) 
  (h_five_fridays : has_five_fridays days_in_month)
  (h_first_not_friday : first_not_friday start)
  (h_last_not_friday : last_not_friday start days_in_month) : 
  (start + 11) % 7 = 1 :=
sorry

end twelfth_is_monday_l332_332069


namespace first_player_wins_l332_332640

theorem first_player_wins (M : ℕ) (H : M = 2014) : 
  ∃ (strategy : Π (remaining_matches : ℕ), 1 ≤ remaining_matches ∧ remaining_matches ≤ 7), 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 7 → strategy (M - n) = 6 - n) ∧ 
  ∀ (remaining_matches : ℕ), remaining_matches % 8 = 0 → 
  (∃ (k : ℕ), k = 8 - remaining_matches):
    (player_takes_last_match strategy remaining_matches) :=
by 
  sorry

end first_player_wins_l332_332640


namespace no_five_circles_with_given_property_l332_332304

theorem no_five_circles_with_given_property :
  ¬∃ (centers : Fin 5 → ℝ × ℝ) (radii : Fin 5 → ℝ), 
    ∀ i, ∃ center_indices : Finset (Fin 5), 
    center_indices.card = 3 ∧ (∀ j ∈ center_indices, (centers i) ∈ (circumcircle (centers j) (radii j))) :=
by
  sorry

end no_five_circles_with_given_property_l332_332304


namespace angle_at_3_40_pm_is_130_degrees_l332_332667

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332667


namespace exterior_angle_theorem_contradicts_given_statement_l332_332217

theorem exterior_angle_theorem_contradicts_given_statement
    (A B C A_ext : ℝ)
    (h1: A + B + C = π)
    (h2: A_ext = B + C) :
    ¬ (A_ext = A + B) :=
by
    intro h3
    rw [h2] at h3
    have h4 : A + B = B + C := h3
    exact false_of_ne h1 (ne_of_lt (lt_of_add_lt_add_right (lt_add_of_lt_of_nonneg (lt_of_lt_of_le (add_pos (lt_add_of_pos_left A (lt_add_of_pos_left B (lt_irrefl _ _)))) (le_refl π)) (le_refl (π - (A + B)))))

end exterior_angle_theorem_contradicts_given_statement_l332_332217


namespace number_of_students_l332_332180

theorem number_of_students (Pedro_best Pedro_worst : ℕ) (h_best : Pedro_best = 30) (h_worst : Pedro_worst = 30) : 
∃ n : ℕ, n = 59 :=
by
  let num_students := 29 + 29 + 1
  have h : num_students = 59 := rfl
  use num_students
  exact h

end number_of_students_l332_332180


namespace fish_count_l332_332249

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l332_332249


namespace final_student_count_is_correct_l332_332353

-- Define the initial conditions
def initial_students : ℕ := 11
def students_left_first_semester : ℕ := 6
def students_joined_first_semester : ℕ := 25
def additional_students_second_semester : ℕ := 15
def students_transferred_second_semester : ℕ := 3
def students_switched_class_second_semester : ℕ := 2

-- Define the final number of students to be proven
def final_number_of_students : ℕ := 
  initial_students - students_left_first_semester + students_joined_first_semester + 
  additional_students_second_semester - students_transferred_second_semester - students_switched_class_second_semester

-- The theorem we need to prove
theorem final_student_count_is_correct : final_number_of_students = 40 := by
  sorry

end final_student_count_is_correct_l332_332353


namespace expand_polynomials_l332_332866

theorem expand_polynomials : 
  (λ x : ℝ, 7 * x^2 + 3) * (λ x : ℝ, 5 * x^3 + 2 * x + 1) = 
  (λ x : ℝ, 35 * x^5 + 29 * x^3 + 7 * x^2 + 6 * x + 3) :=
sorry

end expand_polynomials_l332_332866


namespace total_pink_crayons_l332_332571

def mara_crayons := 40
def mara_pink_percent := 10
def luna_crayons := 50
def luna_pink_percent := 20

def pink_crayons (total_crayons : ℕ) (percent_pink : ℕ) : ℕ :=
  (percent_pink * total_crayons) / 100

def mara_pink_crayons := pink_crayons mara_crayons mara_pink_percent
def luna_pink_crayons := pink_crayons luna_crayons luna_pink_percent

theorem total_pink_crayons : mara_pink_crayons + luna_pink_crayons = 14 :=
by
  -- Proof can be written here.
  sorry

end total_pink_crayons_l332_332571


namespace prove_chord_conditions_l332_332130

theorem prove_chord_conditions
  (a c : ℝ)
  (h_pr_perpendicular : PR ⟂ MN)
  (h_PR_diameter : PR = 2)
  (h_unit_circle : OR = 1)
  (h_eq1 : c - a = 1)
  (h_eq2 : ca = 1)
  (h_eq3 : c^2 - a^2 = 2) :
  (c = 1.5 ∧ a = 0.5) ∧ 
  ¬ (ca = 1) ∧ 
  (c^2 - a^2 = 2) := 
by
  sorry

end prove_chord_conditions_l332_332130


namespace sum_of_solutions_l332_332951
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332951


namespace R_l332_332548

variables (m b d : ℝ)

def t1 : ℝ := (m / 2) * (2 * (b + 2) + (m - 1) * (d + 1))
def t2 : ℝ := (3 * m / 2) * (2 * (b + 2) + (3 * m - 1) * (d + 1))
def t3 : ℝ := (5 * m / 2) * (2 * (b + 2) + (5 * m - 1) * (d + 1))
def R' : ℝ := t3 - t2 - t1

theorem R'_dependency : ∀ {b : ℝ}, R' m b d = sorry

end R_l332_332548


namespace snail_total_distance_l332_332799

-- Conditions
def initial_pos : ℤ := 0
def pos1 : ℤ := 4
def pos2 : ℤ := -3
def pos3 : ℤ := 6

-- Total distance traveled by the snail
def distance_traveled : ℤ :=
  abs (pos1 - initial_pos) +
  abs (pos2 - pos1) +
  abs (pos3 - pos2)

-- Theorem statement
theorem snail_total_distance : distance_traveled = 20 :=
by
  -- Proof is omitted, as per request
  sorry

end snail_total_distance_l332_332799


namespace a_2011_eq_2012_l332_332614

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) : ℝ := f (n : ℝ)

axiom f_x_plus_3_leq_f_x_plus_3 (x : ℝ) : f(x + 3) ≤ f(x) + 3
axiom f_x_plus_2_geq_f_x_plus_2 (x : ℝ) : f(x + 2) ≥ f(x) + 2
axiom f_1_eq_2 : f(1) = 2
axiom a_n_eq_f_n (n : ℕ) : a n = f n

theorem a_2011_eq_2012 : a 2011 = 2012 := by sorry

end a_2011_eq_2012_l332_332614


namespace clock_angle_l332_332705

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332705


namespace solve_for_c_l332_332199

theorem solve_for_c (a b c d e : ℝ) 
  (h1 : a + b + c = 48)
  (h2 : c + d + e = 78)
  (h3 : a + b + c + d + e = 100) :
  c = 26 :=
by
sorry

end solve_for_c_l332_332199


namespace angle_range_l332_332490

noncomputable def angle_between_vectors (a b : ℝ^3) : ℝ :=
Real.acos ((a ⬝ b) / (∥a∥ * ∥b∥))

theorem angle_range (a b : ℝ^3) (λ : ℝ) :
  (∥a∥ ≠ 0) →
  (∥b∥ ≠ 0) →
  (λ ∈ Icc (Real.sqrt 3 / 3) 1) →
  (∥a∥ = ∥b∥) →
  (∥a∥ = λ * ∥a + b∥) →
  let α := angle_between_vectors b (a - b) in
  α ∈ Icc (2 * Real.pi / 3) (5 * Real.pi / 6) :=
by
  intros _ _ _ _ _
  sorry

end angle_range_l332_332490


namespace area_of_rectangle_formed_by_roots_l332_332219

noncomputable def roots : set ℂ := {z : ℂ | (3*z^4 + 6*complex.I*z^3 - 8*z^2 - 2*complex.I*z + (8 - 24*ℂ.I)) = 0}

def forms_rectangle (points : set ℂ) : Prop :=
  ∃ a b c d : ℂ, points = {a, b, c, d} ∧ (a + b + c + d) / 4 = -complex.I / 2

theorem area_of_rectangle_formed_by_roots :
  forms_rectangle roots →
  ∃ (k j : ℝ), k * j = real.sqrt 2.687 :=
by
  intro h
  sorry

end area_of_rectangle_formed_by_roots_l332_332219


namespace card_difference_l332_332811

-- Definitions for the number of cards each individual has
def Heike := 60 / 6
def Ann := 60
def Anton := Heike * 3
def Bertrand := Heike * 2
def Carla := Heike * 4
def Desmond := Heike * 8

-- Statement to prove the difference between the highest and lowest number of cards
theorem card_difference : Desmond - Heike = 70 :=
by
  -- Ensure all definitions are correctly evaluated
  have hHeike : Heike = 10 := by norm_num
  have hAnton : Anton = 30 := by norm_num [Heike, hHeike]
  have hBertrand : Bertrand = 20 := by norm_num [Heike, hHeike]
  have hCarla : Carla = 40 := by norm_num [Heike, hHeike]
  have hDesmond : Desmond = 80 := by norm_num [Heike, hHeike]
  -- Simplify and calculate the difference
  simpa [Desmond, Heike, hDesmond, hHeike]

end card_difference_l332_332811


namespace triangle_PXY_perimeter_l332_332269

/-- Triangle PQR has side-lengths PQ = 10, QR = 20, PR = 16.
The line through the incenter of triangle PQR, parallel to line QR, intersects line PQ at X and line PR at Y.
The perimeter of triangle PXY is 26. -/
theorem triangle_PXY_perimeter : 
  ∀ (P Q R X Y : Type) (PQ QR PR : ℝ) (hPQ : PQ = 10) (hQR : QR = 20) (hPR : PR = 16)
    (h_parallel : (XY ∥ QR)) (h_intersect_X : XY.intersects PQ at X) (h_intersect_Y : XY.intersects PR at Y), 
  perimeter (triangle P X Y) = 26 :=
by 
  sorry

end triangle_PXY_perimeter_l332_332269


namespace problem1_problem2_l332_332819

-- Problem 1 statement in Lean 4
theorem problem1 : (Real.sqrt 36 - 3 * (-1)^2023 + Real.cbrt (-8) = 7) :=
by
  sorry

-- Problem 2 statement in Lean 4
theorem problem2 : ((3 * Real.sqrt 3 - 2 * Real.sqrt 2) + Real.sqrt 2 + abs (1 - Real.sqrt 3) = 4 * Real.sqrt 3 - Real.sqrt 2 - 1) :=
by
  sorry

end problem1_problem2_l332_332819


namespace distinct_possible_collections_l332_332580

-- Definitions of letters and the problem
def letters : set Char := {'P', 'H', 'Y', 'C', 'S', 'S', 'I'}
def vowels : set Char := {'I'}
def consonants : set Char := {'P', 'H', 'Y', 'C', 'S', 'S'}

-- We assume all vowels and semiconditional are counted and indistinguishable as stated
theorem distinct_possible_collections : 
    (3 ∈ vowels ∧ 3 ∈ consonants) ∧ (∀ s ∈ consonants, s ≠ 'S') →
    ∃ n : ℕ, n = 14 :=
sorry

end distinct_possible_collections_l332_332580


namespace age_of_oldest_child_l332_332197

def average_age_of_children (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem age_of_oldest_child :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → average_age_of_children a b c d = 9 → d = 9 :=
by
  intros a b c d h_a h_b h_c h_avg
  sorry

end age_of_oldest_child_l332_332197


namespace twelve_is_monday_l332_332086

def Weekday := {d : String // d ∈ ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

def not_friday (d: Weekday) : Prop := d.val ≠ "Friday"

def has_exactly_five_fridays (first_friday: nat) (days_in_month: nat) : Prop :=
  first_friday + 28 <= days_in_month ∧
  first_friday + 21 <= days_in_month ∧
  first_friday + 14 <= days_in_month ∧
  first_friday + 7 <= days_in_month ∧
  first_friday > 0 ∧ days_in_month <= 31

noncomputable def compute_day_of_week (start_day: String) (n: nat) : Weekday :=
  sorry

theorem twelve_is_monday (start_day: Weekday) (days_in_month: nat) :
    has_exactly_five_fridays 2 days_in_month
  → not_friday start_day
  → not_friday (compute_day_of_week start_day.val days_in_month)
  → compute_day_of_week start_day.val 12 = ⟨"Monday", by sorry⟩ :=
begin
  sorry
end

end twelve_is_monday_l332_332086


namespace sum_of_solutions_eq_16_l332_332930

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332930


namespace question1_question2_l332_332446

noncomputable section 

variable {α : ℝ}

-- Definitions from conditions
def tanα : ℝ := 2
def sinα : ℝ := 2 / real.sqrt 5
def cosα : ℝ := 1 / real.sqrt 5

-- Proof problem statements
theorem question1 : (2 * sinα - cosα) / (sinα + 2 * cosα) = 3 / 4 := by
  sorry

theorem question2 : sinα^2 + sinα * cosα - 2 * cosα^2 = 4 / 5 := by
  sorry

end question1_question2_l332_332446


namespace rationalize_denominator_l332_332594

theorem rationalize_denominator : 
  ∀ (a b d : ℝ), 
  a = 7 → 
  b = 2 * sqrt 50 → 
  d = 20 →
  (a / b) = (7 * sqrt 2 / d) :=
by
  intros a b d ha hb hd
  -- Proof omitted for brevity.
  sorry

end rationalize_denominator_l332_332594


namespace geometric_sum_4500_l332_332239

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l332_332239


namespace percentage_cut_l332_332772

theorem percentage_cut (S C : ℝ) (hS : S = 940) (hC : C = 611) :
  (C / S) * 100 = 65 := 
by
  rw [hS, hC]
  norm_num

end percentage_cut_l332_332772


namespace sum_of_solutions_eq_16_l332_332921

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332921


namespace square_area_le_half_triangle_area_l332_332537

-- Define the problem
variable {A B C M N P Q : Point}
variable {a h : ℝ}
variable {area_triangle area_square : ℝ}
variable (acute_triangle : Triangle A B C)
variable (inscribed_square : Square M N P Q)

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop := ∀ angle, t.interiorAngle angle < 90

def side_length (a : ℝ) := a
def height (h : ℝ) := h

def area_of_triangle (a h : ℝ) := (1 / 2) * a * h
def area_of_square (s : ℝ) := s ^ 2

axiom triangle_is_acute : is_acute_triangle acute_triangle
axiom square_is_inscribed : inscribed_square M N P Q

noncomputable def triangle_area := area_of_triangle (side_length a) (height h)
noncomputable def square_area := area_of_square (a * h / (a + h))

-- Theorem statement
theorem square_area_le_half_triangle_area :
  square_area inscribed_square M N P Q ≤ (1 / 2) * triangle_area acute_triangle :=
sorry

end square_area_le_half_triangle_area_l332_332537


namespace arithmetic_mean_of_4_and_16_l332_332440

theorem arithmetic_mean_of_4_and_16 : 
  (x : ℝ) (h : x = (4 + 16) / 2) → x = 10 := 
by
  intro x h
  rw h
  norm_num
  sorry

end arithmetic_mean_of_4_and_16_l332_332440


namespace sum_of_solutions_eq_16_l332_332927

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332927


namespace electric_car_charging_cost_l332_332503

/-- The fractional equation for the given problem,
    along with the correct solution for the average charging cost per kilometer. -/
theorem electric_car_charging_cost (
    x : ℝ
) : 
    (200 / x = 4 * (200 / (x + 0.6))) → x = 0.2 :=
by
  intros h_eq
  sorry

end electric_car_charging_cost_l332_332503


namespace area_triangle_ABC_is_10_l332_332627

-- Define points A, B, and C with the given transformations
def pointA : ℝ × ℝ := (2, 3)
def pointB : ℝ × ℝ := (-pointA.1, pointA.2)  -- Reflect over y-axis
def pointC : ℝ × ℝ := (-pointB.2, -pointB.1)  -- Reflect over y = -x

-- Define the function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the area of the triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * distance A B * abs (C.1 - A.1)

-- The main statement to prove
theorem area_triangle_ABC_is_10 : triangle_area pointA pointB pointC = 10 := by
  sorry

end area_triangle_ABC_is_10_l332_332627


namespace wall_building_time_l332_332132

theorem wall_building_time (m1 m2 d1 d2 k : ℕ) (h1 : m1 = 12) (h2 : d1 = 6) (h3 : m2 = 18) (h4 : k = 72) 
  (condition : m1 * d1 = k) (rate_const : m2 * d2 = k) : d2 = 4 := by
  sorry

end wall_building_time_l332_332132


namespace domain_range_e_ln_eq_1_over_sqrt_l332_332345

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}
def range (f : ℝ → ℝ) : Set ℝ := {y | ∃ x, f x = y}

theorem domain_range_e_ln_eq_1_over_sqrt :
  (domain (λ x : ℝ, Real.exp (Real.log x)) = {x | x > 0}) ∧ 
  (range (λ x : ℝ, Real.exp (Real.log x)) = Set.Ioi 0) →
  (domain (λ x : ℝ, 1 / Real.sqrt x) = {x | x > 0}) ∧ 
  (range (λ x : ℝ, 1 / Real.sqrt x) = Set.Ioi 0) := by
  intros
  sorry

end domain_range_e_ln_eq_1_over_sqrt_l332_332345


namespace BurjKhalifaHeight_l332_332189

def SearsTowerHeight : ℕ := 527
def AdditionalHeight : ℕ := 303

theorem BurjKhalifaHeight : (SearsTowerHeight + AdditionalHeight) = 830 :=
by
  sorry

end BurjKhalifaHeight_l332_332189


namespace cube_surface_area_l332_332638

theorem cube_surface_area (v : ℝ) (h : v = 1000) : ∃ (s : ℝ), s^3 = v ∧ 6 * s^2 = 600 :=
by
  sorry

end cube_surface_area_l332_332638


namespace find_y_l332_332157

theorem find_y (x y : ℝ)
  (h : y = sqrt ((2008 * x + 2009) / (2010 * x - 2011)) + sqrt ((2008 * x + 2009) / (2011 - 2010 * x)) + 2010) :
  y = 2010 :=
sorry

end find_y_l332_332157


namespace sum_of_solutions_l332_332897

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332897


namespace geometric_series_sum_l332_332815

theorem geometric_series_sum (a r : ℚ) (ha : a = 1) (hr : r = 1/4) : 
  (∑' n:ℕ, a * r^n) = 4/3 :=
by
  rw [ha, hr]
  sorry

end geometric_series_sum_l332_332815


namespace coeff_x5_in_expansion_is_neg_6_l332_332875

noncomputable def coefficient_x5_term_expansion : ℤ :=
let term1 := (6.choose 1) * (5.choose 0) * (-1)^1 in
- term1

theorem coeff_x5_in_expansion_is_neg_6 :
  coefficient_x5_term_expansion = -6 :=
by
simp only [coefficient_x5_term_expansion, int.of_nat_eq_coe, int.coe_nat_choose, nat.choose_zero_right, nat.choose_one_right]
sorry

end coeff_x5_in_expansion_is_neg_6_l332_332875


namespace seq_sum_inequality_l332_332019

theorem seq_sum_inequality (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, 0 < n → a (n + 1) - a n ≥ 2) →
  (∀ n : ℕ, S n = ∑ i in (finset.range n).image (λ i, i + 1), a i) →
  ∀ n : ℕ, S n ≥ n^2 :=
by sorry

end seq_sum_inequality_l332_332019


namespace minimum_score_for_algebra_l332_332648

theorem minimum_score_for_algebra :
  ∀ (score1 score2 score3 min_avg : ℕ), 
  (score1 = 82) → (score2 = 77) → (score3 = 75) → (min_avg = 83) →
  ∑ i in [score1, score2, score3, 98], i / 4 ≥ min_avg := sorry

end minimum_score_for_algebra_l332_332648


namespace cannot_invert_all_signs_l332_332099

structure RegularDecagon :=
  (vertices : Fin 10 → ℤ)
  (diagonals : Fin 45 → ℤ) -- Assume we encode the intersections as unique indices for simplicity.
  (all_positives : ∀ v, vertices v = 1 ∧ ∀ d, diagonals d = 1)

def isValidSignChange (t : List ℤ) : Prop :=
  t.length % 2 = 0

theorem cannot_invert_all_signs (D : RegularDecagon) :
  ¬ (∃ f : Fin 10 → ℤ → ℤ, ∀ (side : Fin 10) (val : ℤ), f side val = -val) :=
sorry

end cannot_invert_all_signs_l332_332099


namespace stopped_clock_more_accurate_l332_332291

theorem stopped_clock_more_accurate (slow_correct_time_frequency : ℕ)
  (stopped_correct_time_frequency : ℕ)
  (h1 : slow_correct_time_frequency = 720)
  (h2 : stopped_correct_time_frequency = 2) :
  stopped_correct_time_frequency > slow_correct_time_frequency / 720 :=
by
  sorry

end stopped_clock_more_accurate_l332_332291


namespace perfect_squares_ending_in_5_or_6_lt_2000_l332_332032

theorem perfect_squares_ending_in_5_or_6_lt_2000 :
  ∃ (n : ℕ), n = 9 ∧ ∀ k, 1 ≤ k ∧ k ≤ 44 → 
  (∃ m, m * m < 2000 ∧ (m % 10 = 5 ∨ m % 10 = 6)) :=
by
  sorry

end perfect_squares_ending_in_5_or_6_lt_2000_l332_332032


namespace total_pink_crayons_l332_332568

-- Define the conditions
def Mara_crayons : ℕ := 40
def Mara_pink_percent : ℕ := 10
def Luna_crayons : ℕ := 50
def Luna_pink_percent : ℕ := 20

-- Define the proof problem statement
theorem total_pink_crayons : 
  (Mara_crayons * Mara_pink_percent / 100) + (Luna_crayons * Luna_pink_percent / 100) = 14 := 
by sorry

end total_pink_crayons_l332_332568


namespace final_result_l332_332364

-- Define the double factorial for odd and even n
def double_factorial (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    (List.range' 1 (n // 2 + 1)).map (λ k, 2 * k + 1).foldr (*) 1
  else
    (List.range' 1 (n // 2 + 1)).map (λ k, 2 * k).foldr (*) 1

-- Define the sum S
def sum_S : ℕ :=
  (List.range' 1 2011).map (λ i, double_factorial (2 * i) / double_factorial (2 * i + 1)).sum

-- Prove the final objective
theorem final_result : ∑ i in Finset.range 2010, (double_factorial (2 * i + 1) / double_factorial (2 * i + 2)) = 0 :=
  by sorry

end final_result_l332_332364


namespace percentage_third_year_students_l332_332101

-- Define the conditions as given in the problem
variables (T : ℝ) (T_3 : ℝ) (S_2 : ℝ)

-- Conditions
def cond1 : Prop := S_2 = 0.10 * T
def cond2 : Prop := (0.10 * T) / (T - T_3) = 1 / 7

-- Define the proof goal
theorem percentage_third_year_students (h1 : cond1 T S_2) (h2 : cond2 T T_3) : T_3 = 0.30 * T :=
sorry

end percentage_third_year_students_l332_332101


namespace range_of_a_l332_332042

theorem range_of_a (a : ℝ) : (∃ x : ℝ, real.sin x < a) → a > -1 :=
sorry

end range_of_a_l332_332042


namespace sum_of_solutions_eq_16_l332_332977

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332977


namespace inverse_of_function_l332_332222

theorem inverse_of_function (x : ℝ) (hx : x > 1) : 
  ∃ (y : ℝ), 5^y + 1 = x ∧ y = log 5 (x - 1) :=
by
  sorry

end inverse_of_function_l332_332222


namespace zero_in_interval_l332_332221

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x - 5

theorem zero_in_interval : ∃ c ∈ set.Ioo 1 2, f c = 0 :=
by
  sorry

end zero_in_interval_l332_332221


namespace day_of_12th_l332_332082

theorem day_of_12th (month : Type) [decidable_eq month] [fintype month] 
  (is_friday : month → Prop) (is_first_day : month → Prop) (is_last_day : month → Prop)
  (days : list month) (nth_day : Π (n : ℕ), month)
  (five_fridays : ∃ (n : ℕ), (is_friday ∘ nth_day) '' (finset.range (min n 7)) = {5})
  (not_first_day_friday : ∀ d, is_first_day d → ¬ is_friday d)
  (not_last_day_friday : ∀ d, is_last_day d → ¬ is_friday d) : 
  nth_day 12 = "Monday" := 
sorry

end day_of_12th_l332_332082


namespace day_of_12th_l332_332081

theorem day_of_12th (month : Type) [decidable_eq month] [fintype month] 
  (is_friday : month → Prop) (is_first_day : month → Prop) (is_last_day : month → Prop)
  (days : list month) (nth_day : Π (n : ℕ), month)
  (five_fridays : ∃ (n : ℕ), (is_friday ∘ nth_day) '' (finset.range (min n 7)) = {5})
  (not_first_day_friday : ∀ d, is_first_day d → ¬ is_friday d)
  (not_last_day_friday : ∀ d, is_last_day d → ¬ is_friday d) : 
  nth_day 12 = "Monday" := 
sorry

end day_of_12th_l332_332081


namespace alice_card_value_l332_332470

theorem alice_card_value (x : ℝ) (hx : x ∈ Ioo (π / 2) π) 
  (h1 : ∃ a b c : ℝ, set.eq_on (λ y, y) (λ y, sin x) {a} ∧ set.eq_on (λ y, y) (λ y, cos x) {b} ∧ set.eq_on (λ y, y) (λ y, tan x) {c} ∧ ∀ y ∈ {a}, sin x ≠ y → ∀ y ∈ {b}, cos x ≠ y → ∀ y ∈ {c}, tan x ≠ y) :
  sin x = (-1 + real.sqrt 5) / 2 :=
sorry

end alice_card_value_l332_332470


namespace sum_of_solutions_sum_of_all_solutions_l332_332903

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332903


namespace sum_geometric_sequence_terms_l332_332236

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l332_332236


namespace combine_largest_and_second_smallest_l332_332644

theorem combine_largest_and_second_smallest
  (S : set ℤ)
  (h : S = {10, 11, 12, 13}) :
  let largest := max (max 10 11) (max 12 13),
      second_smallest := max (min 10 11) (min 11 12) in
  largest + second_smallest = 24 := 
by
  sorry

end combine_largest_and_second_smallest_l332_332644


namespace wave_number_count_l332_332172

def wave_number (n : ℕ) : Prop :=
  ∀ i, (i = 1 ∨ i = 3) → (n.digit i > n.digit (i - 1) ∧ n.digit i > n.digit (i + 1))

theorem wave_number_count :
  let digits := [2, 3, 4, 5, 6] in
  let count := (digits.numbers(5).filter wave_number).length in
  count = 32 := by
  sorry

end wave_number_count_l332_332172


namespace least_multiple_15_product_15_l332_332278

def digits_product (n : ℕ) : ℕ :=
  -- Function to compute the product of the digits of a number
  (n.digits 10).product

def is_multiple_of_15 (n : ℕ) : Prop :=
  15 ∣ n

def ends_with_5 (n : ℕ) : Prop :=
  n % 10 = 5

theorem least_multiple_15_product_15 :
  ∃ n : ℕ, is_multiple_of_15 n ∧ ends_with_5 n ∧ digits_product n = 15 ∧ (∀ m : ℕ, is_multiple_of_15 m ∧ ends_with_5 m ∧ digits_product m = 15 → n ≤ m) :=
sorry

end least_multiple_15_product_15_l332_332278


namespace find_x_solutions_l332_332379

theorem find_x_solutions :
  ∀ x : ℝ, (Real.sqrt (Real.sqrt (57 - 2 * x)) + Real.sqrt (Real.sqrt (45 + 2 * x)) = 4) ↔ (x = 27 ∨ x = -17) :=
by
  intro x
  split
  sorry
  sorry

end find_x_solutions_l332_332379


namespace distinct_real_numbers_condition_l332_332398

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)

theorem distinct_real_numbers_condition (a b x1 x2 x3 : ℝ) :
  f a b x1 = x2 → f a b x2 = x3 → f a b x3 = x1 → x1 ≠ x2 → x2 ≠ x3 → x1 ≠ x3 → a = -b^2 :=
by
  sorry

end distinct_real_numbers_condition_l332_332398


namespace student_marks_proportion_l332_332339

theorem student_marks_proportion (x : ℝ) (p1 p2 p3 p4 p5 : ℝ)
  (hx_pos : x > 0)
  (h_sum : p1 + p2 + p3 + p4 + p5 = 3 * x)
  (h_greater_half_x : ∀ i, [i ∈ {p1, p2, p3, p4, p5}] → i > 0.5 * x) :
  ∀ i ∈ {p1, p2, p3, p4, p5}, i / x = 0.6 :=
by
  sorry

end student_marks_proportion_l332_332339


namespace total_spending_l332_332845

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end total_spending_l332_332845


namespace DQ_eq_2PE_l332_332420

variables {A B C D F Q P K E : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited F] [inhabited Q] [inhabited P] [inhabited K] [inhabited E]
variables [square A B C D] [bisector F A C D] [on_side Q C D] [perpendicular B Q F] [intersection P A C B Q]
variables [midpoint K D Q] [midpoint E D B]

theorem DQ_eq_2PE : (distance D Q) = 2 * (distance P E) :=
sorry

end DQ_eq_2PE_l332_332420


namespace general_equation_M_range_distance_D_to_l_l332_332109

noncomputable def parametric_to_general (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  x^2 + y^2 / 4 = 1

noncomputable def distance_range (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  let l := x + y - 4
  let d := |x + 2 * y - 4| / Real.sqrt 2
  let min_dist := (4 * Real.sqrt 2 - Real.sqrt 10) / 2
  let max_dist := (4 * Real.sqrt 2 + Real.sqrt 10) / 2
  min_dist ≤ d ∧ d ≤ max_dist

theorem general_equation_M (θ : ℝ) : parametric_to_general θ := sorry

theorem range_distance_D_to_l (θ : ℝ) : distance_range θ := sorry

end general_equation_M_range_distance_D_to_l_l332_332109


namespace problem1_problem2_l332_332818

-- Problem 1 statement in Lean 4
theorem problem1 : (Real.sqrt 36 - 3 * (-1)^2023 + Real.cbrt (-8) = 7) :=
by
  sorry

-- Problem 2 statement in Lean 4
theorem problem2 : ((3 * Real.sqrt 3 - 2 * Real.sqrt 2) + Real.sqrt 2 + abs (1 - Real.sqrt 3) = 4 * Real.sqrt 3 - Real.sqrt 2 - 1) :=
by
  sorry

end problem1_problem2_l332_332818


namespace initials_count_l332_332473

-- Let L be the set of letters {A, B, C, D, E, F, G, H, I, J}
def L : finset char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Each initial can be any element in L
def initials : Type := L × L × L

-- We need to prove that the number of different three-letter sets of initials is equal to 1000
theorem initials_count : finset.card (finset.product L (finset.product L L)) = 1000 := 
sorry

end initials_count_l332_332473


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332727

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332727


namespace least_value_of_abc_l332_332604

theorem least_value_of_abc :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (∃ k : ℕ, k = √(a * √(b * √(c)))) → a + b + c ≤ x + y + z → a + b + c = 7 :=
by
  sorry

end least_value_of_abc_l332_332604


namespace lisa_socks_total_l332_332166

def total_socks (initial : ℕ) (sandra : ℕ) (cousin_ratio : ℕ → ℕ) (mom_extra : ℕ → ℕ) : ℕ :=
  initial + sandra + cousin_ratio sandra + mom_extra initial

def cousin_ratio (sandra : ℕ) : ℕ := sandra / 5
def mom_extra (initial : ℕ) : ℕ := 3 * initial + 8

theorem lisa_socks_total :
  total_socks 12 20 cousin_ratio mom_extra = 80 := by
  sorry

end lisa_socks_total_l332_332166


namespace sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l332_332778

theorem sufficient_but_not_necessary_condition_for_x_lt_3 (x : ℝ) : |x - 1| < 2 → x < 3 :=
by {
  sorry
}

theorem not_necessary_condition_for_x_lt_3 (x : ℝ) : (x < 3) → ¬(-1 < x ∧ x < 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l332_332778


namespace orthocenter_closest_to_B_l332_332524

theorem orthocenter_closest_to_B (A B C : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C]
  (triangle : A × B × C) 
  (AB BC CA : ℝ) 
  (h : AB < BC ∧ BC < CA) :
  closest_vertex_orthocenter triangle B :=
sorry

end orthocenter_closest_to_B_l332_332524


namespace parabola_y_intersection_l332_332202

theorem parabola_y_intersection : intersects (x^2 - 4) (0, -4) :=
by
  sorry

end parabola_y_intersection_l332_332202


namespace num_integer_roots_P_P_x_l332_332142

open Polynomial

noncomputable def P : Polynomial ℤ := sorry
variable {n : ℕ}
variable (hdeg : P.degree = n)
variable (hn_geq_5 : n ≥ 5)
variable (hroots : ∀ {x : ℤ}, P x = 0 → x ∈ finset.range n)
variable (hP_zero : P 0 = 0)

theorem num_integer_roots_P_P_x : 
  finset.card (finset.filter (λ x : ℤ, (P.comp P) x = 0) finset.univ) = n :=
begin
  sorry
end

end num_integer_roots_P_P_x_l332_332142


namespace magnitude_fraction_complex_l332_332864

theorem magnitude_fraction_complex :
  abs ((7 : ℂ) / 4 - 3 * I) = real.sqrt 193 / 4 :=
begin 
  sorry 
end

end magnitude_fraction_complex_l332_332864


namespace calvin_total_insects_l332_332822

-- Definitions based on the conditions
def roaches := 12
def scorpions := 3
def crickets := roaches / 2
def caterpillars := scorpions * 2

-- Statement of the problem
theorem calvin_total_insects : 
  roaches + scorpions + crickets + caterpillars = 27 :=
  by
    sorry

end calvin_total_insects_l332_332822


namespace rain_first_hour_l332_332122

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end rain_first_hour_l332_332122


namespace rain_in_first_hour_l332_332124

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end rain_in_first_hour_l332_332124


namespace clock_angle_3_40_l332_332716

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332716


namespace angle_at_3_40_pm_is_130_degrees_l332_332670

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332670


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l332_332261

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l332_332261


namespace average_weight_of_children_l332_332610

theorem average_weight_of_children
  (S_B S_G : ℕ)
  (avg_boys_weight : S_B = 8 * 160)
  (avg_girls_weight : S_G = 5 * 110) :
  (S_B + S_G) / 13 = 141 := 
by
  sorry

end average_weight_of_children_l332_332610


namespace not_perfect_squares_l332_332767

-- Definitions of the numbers as per conditions
def n1 : ℕ := 6^3032
def n2 : ℕ := 7^3033
def n3 : ℕ := 8^3034
def n4 : ℕ := 9^3035
def n5 : ℕ := 10^3036

-- Proof statement asserting which of these numbers are not perfect squares.
theorem not_perfect_squares : ¬ (∃ x : ℕ, x^2 = n2) ∧ ¬ (∃ x : ℕ, x^2 = n4) :=
by
  -- Proof is omitted
  sorry

end not_perfect_squares_l332_332767


namespace problem_2021_a1_l332_332539

-- Definitions based on the problem conditions
-- 1. Define the scenario of vertices arranged in an n-gon with n = 3^7 + 1
def n : ℕ := 2188
def O : ℂ := 0 -- center of the n-gon is the origin in the complex plane

-- 2. Define the vertices A_j^{(0)} in the complex plane
def cis (θ : ℝ) : ℂ := complex.exp (θ * complex.I)
def ω := cis (2 * real.pi / n)
def A (j : ℕ) : ℕ -> ℂ | 0 := ω^j

-- 3. Define the centroid transformation for i = 1 to 7
def centroid (a b c : ℂ) : ℂ := (a + b + c) / 3

-- Recursive definition of A_j^{(i)}
def A' (i j : ℕ) : ℂ :=
  if i = 0 then A j
  else centroid (A' (i - 1) j) (A' (i - 1) (j + 3^(7 - i) % n)) (A' (i - 1) (j + 2 * 3^(7 - i) % n))

-- 4. Prove the ratio condition leading to p + q
theorem problem_2021_a1 :
  let p := 1
  let q := 2187
  p + q = 2188 :=
by {
  sorry
}

end problem_2021_a1_l332_332539


namespace function_is_decreasing_l332_332856

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)

theorem function_is_decreasing : ∀ x : ℝ, x ≠ -1 → f' x < 0 := by
  have domain : ∀ x : ℝ, x ≠ -1 → f x = 1 / (x + 1) := by
    intros
    exact rfl
  have derivative : ∀ x : ℝ, x ≠ -1 → deriv f x = -1 / (x + 1) ^ 2 := by
    sorry  -- Proof of the derivative computation is omitted
  intros x hx
  rw [derivative x hx]
  rw [domain x hx]
  have hpos : (x + 1) ^ 2 > 0 := by
    have h := pow_two_nonneg (x + 1)
    have h_eq : (x + 1) ^ 2 = 0 ↔ x = -1 := by
      split
      intro heq
      exact eq_zero_of_pow_two_eq_zero heq
      intro heq
      rw [heq]
      ring
    cases lt_or_eq_of_le h
    exact h
    exfalso
    exact hx h_1
  have hneg : - (x + 1) ^ 2 < 0
    exact neg_pos_of_pos hpos
  exact hneg

end function_is_decreasing_l332_332856


namespace no_super_squarish_numbers_l332_332337

def is_super_squarish (M : ℕ) : Prop :=
  let a := M / 100000 % 100
  let b := M / 1000 % 1000
  let c := M % 100
  (M ≥ 1000000 ∧ M < 10000000) ∧
  (M % 10 ≠ 0 ∧ (M / 10) % 10 ≠ 0 ∧ (M / 100) % 10 ≠ 0 ∧ (M / 1000) % 10 ≠ 0 ∧
    (M / 10000) % 10 ≠ 0 ∧ (M / 100000) % 10 ≠ 0 ∧ (M / 1000000) % 10 ≠ 0) ∧
  (∃ y : ℕ, y * y = M) ∧
  (∃ f g : ℕ, f * f = a ∧ 2 * f * g = b ∧ g * g = c) ∧
  (10 ≤ a ∧ a ≤ 99) ∧
  (100 ≤ b ∧ b ≤ 999) ∧
  (10 ≤ c ∧ c ≤ 99)

theorem no_super_squarish_numbers : ∀ M : ℕ, is_super_squarish M → false :=
sorry

end no_super_squarish_numbers_l332_332337


namespace prove_DE_equals_diameter_of_inscribed_circle_l332_332508

-- Define the given conditions of the problem in Lean
variables {a b c : ℕ}
variables (ABC : Triangle) 
variables (AD BE DE r : ℕ)

-- Assume the necessary properties
axiom RightTriABC : ABC.is_right (angle B) -- ABC is a right triangle at B
axiom Hypotenuse : ABC.hypotenuse = c -- Hypotenuse AB
axiom LegBC : ABC.side B C = a -- Leg BC
axiom LegAC : ABC.side A C = b -- Leg AC
axiom SegmentAD : AD = b -- Segment AD = AC = b
axiom SegmentBE : BE = a -- Segment BE = BC = a
axiom Radius : r = (a + b - c) / 2 -- Radius of the inscribed circle

-- The statement to prove
theorem prove_DE_equals_diameter_of_inscribed_circle 
  (Triangle.is_right ABC B) (Hypotenuse ABC c) (LegBC ABC a) (LegAC ABC b)
  (SegmentAD AD b) (SegmentBE BE a) (Radius r ((a + b - c) / 2)) :
  DE = a + b - c := 
by { sorry }

end prove_DE_equals_diameter_of_inscribed_circle_l332_332508


namespace distributor_profit_percentage_l332_332324

theorem distributor_profit_percentage 
    (commission_rate : ℝ) (cost_price : ℝ) (final_price : ℝ) (P : ℝ) (profit : ℝ) 
    (profit_percentage: ℝ) :
  commission_rate = 0.20 →
  cost_price = 15 →
  final_price = 19.8 →
  0.80 * P = final_price →
  P = cost_price + profit →
  profit_percentage = (profit / cost_price) * 100 →
  profit_percentage = 65 :=
by
  intros h_commission_rate h_cost_price h_final_price h_equation h_profit_eq h_percent_eq
  sorry

end distributor_profit_percentage_l332_332324


namespace central_angle_sector_l332_332006

theorem central_angle_sector (l r : ℝ) (h_l : l = 2 * Real.pi) (h_r : r = 2) : 
  (∃ θ : ℝ, l = r * θ) → θ = Real.pi := 
by 
  intro h 
  obtain ⟨θ, hθ⟩ := h
  have hθ' : 2 * Real.pi = 2 * θ := by rw [h_l, h_r]; assumption
  have hθ'' : Real.pi = θ := by linarith 
  exact hθ''.symm 
  sorry

end central_angle_sector_l332_332006


namespace rhombus_properties_l332_332100

-- Define the conditions
def rhombus (a d₁ θ : ℝ) : Prop :=
  a = 25 ∧ d₁ = 10 ∧ θ = real.to_nnreal(60)

-- Define the length of the other diagonal
def length_other_diagonal (a d₁ θ : ℝ) (d₂ : ℝ) : Prop :=
  rhombus a d₁ θ → d₂ = 20

-- Define the height of the rhombus
def height_of_rhombus (a d₁ θ : ℝ) (h : ℝ) : Prop :=
  rhombus a d₁ θ → h = approx 21.65 1e-2

-- Define the area of the rhombus
def area_of_rhombus (a d₁ θ : ℝ) (area : ℝ) : Prop :=
  rhombus a d₁ θ → area = 100

-- Combined theorem statement without proving the steps
theorem rhombus_properties : 
  ∃ d₂ h area, 
    length_other_diagonal 25 10 (real.to_nnreal(60)) d₂ ∧ 
    height_of_rhombus 25 10 (real.to_nnreal(60)) h ∧ 
    area_of_rhombus 25 10 (real.to_nnreal(60)) area := by
  exists 20, approx 21.65 1e-2, 100
  simp [length_other_diagonal, height_of_rhombus, area_of_rhombus, rhombus, real.to_nnreal]
  sorry

end rhombus_properties_l332_332100


namespace triangle_similarity_l332_332177

   theorem triangle_similarity {A B C A₁ B₁ C₁ H : Type*} [triangle A B C] 
     (hA₁ : ratio_fac (altitude A) A₁ 2 1) 
     (hB₁ : ratio_fac (altitude B) B₁ 2 1) 
     (hC₁ : ratio_fac (altitude C) C₁ 2 1) :
     similar (triangle A₁ B₁ C₁) (triangle A B C) := 
   by 
     sorry
   
end triangle_similarity_l332_332177


namespace no_infinite_prime_sequence_l332_332531

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∈ Icc 2 (n - 1), ¬(m ∣ n)

def sequence_satisfies_condition (seq : ℕ → ℕ) : Prop :=
∀ n : ℕ, |seq n.succ - 2 * seq n| = 1

theorem no_infinite_prime_sequence :
  ¬ ∃ seq : ℕ → ℕ, (∀ n, is_prime (seq n)) ∧ sequence_satisfies_condition seq :=
sorry

end no_infinite_prime_sequence_l332_332531


namespace least_multiple_of_15_with_product_multiple_of_15_is_315_l332_332280

-- Let n be a positive integer
def is_multiple_of (n m : ℕ) := ∃ k, n = m * k

def is_product_multiple_of (digits : ℕ) (m : ℕ) := ∃ k, digits = m * k

-- The main theorem we want to state and prove
theorem least_multiple_of_15_with_product_multiple_of_15_is_315 (n : ℕ) 
  (h1 : is_multiple_of n 15) 
  (h2 : n > 0) 
  (h3 : is_product_multiple_of (n.digits.prod) 15) 
  : n = 315 := 
sorry

end least_multiple_of_15_with_product_multiple_of_15_is_315_l332_332280


namespace find_years_ago_twice_age_l332_332636

-- Definitions of given conditions
def age_sum (H J : ℕ) : Prop := H + J = 43
def henry_age : ℕ := 27
def jill_age : ℕ := 16

-- Definition of the problem to be proved
theorem find_years_ago_twice_age (X : ℕ) 
  (h1 : age_sum henry_age jill_age) 
  (h2 : henry_age = 27) 
  (h3 : jill_age = 16) : (27 - X = 2 * (16 - X)) → X = 5 := 
by 
  sorry

end find_years_ago_twice_age_l332_332636


namespace final_state_probability_l332_332186

-- Defining the initial state and process
def initial_state := (1, 1, 1, 1)

def process (state: (ℕ, ℕ, ℕ, ℕ)) : (ℕ, ℕ, ℕ, ℕ) :=
  -- Define the transition function according to the problem conditions.
  sorry

theorem final_state_probability :
  let final_state := process^2020 initial_state in
  probability(final_state = (1, 1, 1, 1)) = 8 / 27 := sorry

end final_state_probability_l332_332186


namespace distance_from_fourth_friends_house_to_work_l332_332162

theorem distance_from_fourth_friends_house_to_work (x : ℝ) :
  let d_1 := x
  let d_2 := x^2
  let d_3 := real.sqrt (x + d_2)
  let d_4 := d_3 + 5
  let total_distance := d_1 + d_2 + 2 * d_3 + 5
  let work_distance := 2 * total_distance
  work_distance - total_distance = d_1 + d_2 + 2 * d_3 + 5 :=
by {
  sorry
}

end distance_from_fourth_friends_house_to_work_l332_332162


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332882

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332882


namespace reciprocals_expression_value_l332_332491

theorem reciprocals_expression_value (a b : ℝ) (h : a * b = 1) : a^2 * b - (a - 2023) = 2023 := 
by 
  sorry

end reciprocals_expression_value_l332_332491


namespace symmetry_axis_of_g_l332_332602

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + sqrt 3 * cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 6)

theorem symmetry_axis_of_g : ∃ k : ℤ, k * π + π / 3 = π / 3 :=
by
  use 0
  sorry

end symmetry_axis_of_g_l332_332602


namespace sum_of_solutions_l332_332962

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332962


namespace roots_difference_l332_332618

theorem roots_difference :
  let r := (1 : ℝ),
      s := (1 / 1983 : ℝ) in
  (exists x1, (1984 * x1)^2 - 1983 * 1985 * (1984 * x1) - 1 = 0 ∧ x1 = r) ∧
  (exists x2, 1983 * x2^2 - 1984 * x2 + 1 = 0 ∧ x2 = s) →
  r - s = 1982 / 1983 :=
by
  -- Proof goes here
  sorry

end roots_difference_l332_332618


namespace range_of_y0_l332_332417

theorem range_of_y0 (x0 y0 : ℝ)
  (h1 : x0^2 / 2 - y0^2 = 1)
  (h2 : (sqrt 3 - x0) * -sqrt 3 + (-y0) * (-y0) < 0) : 
  - (sqrt 3) / 3 < y0 ∧ y0 < (sqrt 3) / 3 :=
sorry

end range_of_y0_l332_332417


namespace angle_at_3_40_pm_is_130_degrees_l332_332664

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332664


namespace distance_missouri_to_new_york_by_car_l332_332211

-- Define the given conditions
def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def midway_factor : ℝ := 0.5

-- Define the problem to be proven
theorem distance_missouri_to_new_york_by_car :
  let total_distance : ℝ := distance_plane + (distance_plane * increase_percentage)
  let missouri_to_new_york_distance : ℝ := total_distance * midway_factor
  missouri_to_new_york_distance = 1400 :=
by
  sorry

end distance_missouri_to_new_york_by_car_l332_332211


namespace smaller_angle_3_40_pm_l332_332743

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332743


namespace OQ_parallel_KI_l332_332528

theorem OQ_parallel_KI
  (K I A O N H Q : Type)
  [triangle : EuclideanGeometry K I A]
  (KI KA : LineSegment K I)
  (O_is_angle_bisector : angle_bisector_of (angle K I A) O)
  (N_is_midpoint : midpoint I A N)
  (H_is_perpendicular : perpendicular_from I K O H)
  (IH_intersects_KN_at_Q : intersects IH KN Q)
  (KI_lt_KA : KI < KA) :
  parallel OQ KI :=
begin
  sorry
end

end OQ_parallel_KI_l332_332528


namespace geometric_sum_4500_l332_332238

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l332_332238


namespace distance_between_centers_of_inscribed_circles_l332_332144

theorem distance_between_centers_of_inscribed_circles (X Y Z : ℝ × ℝ) (XY XZ YZ : ℝ)
  (C1 C2 C3 : circle) (PQ RS : line) (m : ℝ) :
  X = (0,0) →
  Y = (60,0) →
  Z = (0,80) →
  XY = 60 →
  XZ = 80 →
  YZ = 100 →
  C1.radius = 20 →
  -- PQ is perpendicular to XZ and tangent to C1
  PQ.perpendicular_to XZ ∧ PQ.tangent_to C1 →
  -- RS is perpendicular to XY and tangent to C1
  RS.perpendicular_to XY ∧ RS.tangent_to C1 →
  -- C2 and C3 are inscribed circles of triangles PQR and QYS respectively
  ∃ P R : ℝ × ℝ, ∃ Q S : ℝ × ℝ,
    P ∈ XZ ∧ Q ∈ YZ ∧ R ∈ XY ∧ S ∈ YZ ∧
    C2.radius = 15 ∧ C3.radius = 100/3 ∧
    C2.center = (15, 55) ∧ C3.center = (50, 100 / 3 + 10) →
  m = 16050 :=
by
  sorry

end distance_between_centers_of_inscribed_circles_l332_332144


namespace part1_part2_smallest_positive_period_part2_monotonic_increase_intervals_l332_332013

def f (x : ℝ) : ℝ := 2 * cos x * (sin x + cos x)

theorem part1 : f (5 * π / 4) = 2 := 
by 
  sorry

theorem part2_smallest_positive_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π :=
by 
  sorry

theorem part2_monotonic_increase_intervals : ∀ k : ℤ, ∃ x : ℝ, f x = f (k * π - 3 * π / 8) ∧ f (x + π/8) = f (k * π + π / 8) := 
by 
  sorry

end part1_part2_smallest_positive_period_part2_monotonic_increase_intervals_l332_332013


namespace min_value_abs_sum_arithmetic_seq_l332_332422

open Real

noncomputable def min_absolute_sum (a : ℕ → ℝ) (n : ℕ) :=
  (finset.range n).sum (λ i, abs (a i))

theorem min_value_abs_sum_arithmetic_seq {a : ℕ → ℝ} :
  (∀ i j, a i = a j + (i - j) * (a 1 - a 0)) ∧ ((finset.range 12).sum a = 60)
  → min_absolute_sum a 12 = 60 :=
  by
  intros h
  sorry

end min_value_abs_sum_arithmetic_seq_l332_332422


namespace constant_value_l332_332307

theorem constant_value (x y z C : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : x > y) (h4 : y > z) (h5 : z = 2) (h6 : 2 * x + 3 * y + 3 * z = 5 * y + C) : C = 8 :=
by
  sorry

end constant_value_l332_332307


namespace perpendicular_AX_HM_l332_332557

open EuclideanGeometry

theorem perpendicular_AX_HM
  (ABC : Triangle)
  (hAB_neq_AC : ABC.A ≠ ABC.C)
  (H : Point)
  (horthocenter : is_orthocenter H ABC)
  (M : Point)
  (hmidpoint : is_midpoint M ABC.B ABC.C)
  (D E : Point)
  (hD_on_AB : on_segment D ABC.A ABC.B)
  (hE_on_AC : on_segment E ABC.A ABC.C)
  (hAD_eq_AE : segment_length ABC.A D = segment_length ABC.A E)
  (hcollinear_DHE : collinear [D, H, E])
  (X : Point)
  (hintersect_circumcircles : on_circumcircle X ABC ∧ on_circumcircle X (Triangle.mk ABC.A D E)) :
  is_perpendicular (line_through ABC.A X) (line_through H M) := 
sorry

end perpendicular_AX_HM_l332_332557


namespace evaluation_result_l332_332374

noncomputable def evaluate_expression : ℝ :=
  let a := 210
  let b := 206
  let numerator := 980 ^ 2
  let denominator := a^2 - b^2
  numerator / denominator

theorem evaluation_result : evaluate_expression = 577.5 := 
  sorry  -- Placeholder for the proof

end evaluation_result_l332_332374


namespace sum_of_solutions_equation_l332_332935

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332935


namespace license_plate_count_l332_332328

noncomputable def num_license_plates : Nat :=
  let num_digit_possibilities := 10
  let num_letter_possibilities := 26
  let num_letter_pairs := num_letter_possibilities * num_letter_possibilities
  let num_positions_for_block := 6
  num_positions_for_block * (num_digit_possibilities ^ 5) * num_letter_pairs

theorem license_plate_count :
  num_license_plates = 40560000 :=
by
  sorry

end license_plate_count_l332_332328


namespace sum_of_solutions_sum_of_all_solutions_l332_332906

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332906


namespace dennis_total_cost_l332_332848

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end dennis_total_cost_l332_332848


namespace clock_angle_3_40_l332_332711

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332711


namespace sum_of_solutions_l332_332961

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332961


namespace smaller_angle_between_hands_at_3_40_l332_332688

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332688


namespace problem1_problem2_l332_332817

-- Problem 1
theorem problem1 : sqrt 36 - 3 * (-1 : ℤ) ^ 2023 + real.cbrt (-8) = 7 := 
sorry

-- Problem 2
theorem problem2 :
  (3 * real.sqrt 3 - 2 * real.sqrt 2) + real.sqrt 2 + abs (1 - real.sqrt 3) = 
  4 * real.sqrt 3 - real.sqrt 2 - 1 := 
sorry

end problem1_problem2_l332_332817


namespace find_m_perpendicular_l332_332024

-- Define the two vectors
def a (m : ℝ) : ℝ × ℝ := (m, -1)
def b : ℝ × ℝ := (1, 2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Theorem stating the mathematically equivalent proof problem
theorem find_m_perpendicular (m : ℝ) (h : dot_product (a m) b = 0) : m = 2 :=
by sorry

end find_m_perpendicular_l332_332024


namespace distance_BC_is_15_l332_332218

theorem distance_BC_is_15
  (AC BD AD : ℝ)
  (h1 : AC = 50)
  (h2 : BD = 45)
  (h3 : AD = 80) :
  let CD := AD - AC in
  let BC := BD - CD in
  BC = 15 := by
{
  sorry
}

end distance_BC_is_15_l332_332218


namespace prime_factors_count_l332_332480

theorem prime_factors_count (n : ℕ) (h : n = 75) : (nat.factors n).to_finset.card = 2 :=
by
  rw h
  -- The proof part is omitted as instructed
  sorry

end prime_factors_count_l332_332480


namespace zero_in_interval_l332_332639

noncomputable def f (x : ℝ) : ℝ := 2 * x - 8 + Real.logb 3 x

-- The function f is continuous
axiom f_cont : Continuous f

-- The zero of the function f must be located in the interval (3, 4)
theorem zero_in_interval : ∃ c ∈ Ioo 3 4, f c = 0 :=
by
  sorry

end zero_in_interval_l332_332639


namespace twelfth_is_monday_l332_332072

def days_of_week : Type := {d // d < 7}

def starts_on_friday (d : ℕ) : days_of_week := ⟨d % 7, by linarith [Nat.mod_lt d (by norm_num)]⟩

-- Condition 1: There are exactly 5 Fridays in the month (which has at least 30 days)
def has_five_fridays (days_in_month : ℕ) : Prop := 
  ∃ (start : ℕ), 
    (start % 7 ≠ 5) ∧ 
    (days_in_month > 28 ∧ days_in_month < 32) ∧ -- At least 30 days to have 5 Fridays
    ∃ f, ∀ i, starts_on_friday(start + 7*i) = f → (i < 5)

-- Condition 2: The first day of the month is not a Friday
def first_not_friday (start : ℕ) : Prop := start % 7 ≠ 5

-- Condition 3: The last day of the month is not a Friday
def last_not_friday (start days_in_month : ℕ) : Prop := (start + days_in_month - 1) % 7 ≠ 5
    
theorem twelfth_is_monday (days_in_month : ℕ) (start : ℕ) 
  (h_five_fridays : has_five_fridays days_in_month)
  (h_first_not_friday : first_not_friday start)
  (h_last_not_friday : last_not_friday start days_in_month) : 
  (start + 11) % 7 = 1 :=
sorry

end twelfth_is_monday_l332_332072


namespace smaller_angle_3_40_pm_l332_332744

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332744


namespace solve_for_x_l332_332366

noncomputable def log := Real.log

theorem solve_for_x (k l m : ℝ) (a p q r : ℝ) (hk : k ≠ 0) (hl : l ≠ 0) (hm : m ≠ 0) (ha_pos : a > 0)
  (hp_pos : p > 0) (hq_pos : q > 0) (hr_pos : r > 0) : 
  let A := k + l + m 
  let B := k * (log q + log r) + l * (log p + log r) + m * (log p + log q)
  let C := k * (log q * log r) + l * (log p * log r) + m * (log p * log q)
  let Δ := B^2 - 4 * A * C 
  in 
  (A ≠ 0) →
  k * log a + l * log (a^q) * x + m * log a = 0 → 
  (x = 10^(((-B) + Real.sqrt Δ) / (2 * A)) ∨ x = 10^(((-B) - Real.sqrt Δ) / (2 * A))) := 
  sorry

end solve_for_x_l332_332366


namespace most_likely_outcome_l332_332394

-- Define the conditions of the problem
def num_children : ℕ := 5

-- Define probability for a child being a boy or a girl
def prob_child_boy : ℚ := 1 / 2
def prob_child_girl : ℚ := 1 / 2

-- Define the probabilities for each outcome given the conditions
def prob_all_boys : ℚ := (prob_child_boy ^ num_children)
def prob_all_girls : ℚ := (prob_child_girl ^ num_children)
def prob_3_girls_2_boys : ℚ := (binomial num_children 3) * (prob_child_boy ^ 2) * (prob_child_girl ^ 3)
def prob_4_one_gender_1_other_gender : ℚ := (binomial num_children 4) * (prob_child_boy ^ 4) * (prob_child_girl) + (binomial num_children 4) * (prob_child_girl ^ 4) * (prob_child_boy)

-- The proof problem: Prove that prob_3_girls_2_boys and prob_4_one_gender_1_other_gender are the highest probabilities
theorem most_likely_outcome : 
  (prob_3_girls_2_boys = prob_4_one_gender_1_other_gender ∧ 
  prob_3_girls_2_boys > prob_all_boys ∧ 
  prob_3_girls_2_boys > prob_all_girls) := 
sorry

end most_likely_outcome_l332_332394


namespace sum_square_lt_max_min_diff_squared_l332_332555

theorem sum_square_lt_max_min_diff_squared (n : ℕ) (x : Fin n → ℝ) 
  (h1 : n ≥ 2) 
  (h2 : ∑ i, x i = 0) 
  (h3 : ∀ t > 0, ∃ at_most, at_most ≤ 1 / t ∧ ∀ (i j : Fin n), |x i - x j| ≥ t → (i,j) ≠ ⟨0,0⟩ → (i,j).pairwise at_most) :
  ∑ i, (x i) ^ 2 < (1 / n) * (Finset.max' (Finset.univ.image x) (by simp [←Multiset.coe_eq_coe.mpr (Multiset.eq_coe_max_of_fin_nonempty hien1 n x h1)]) - 
  Finset.min' (Finset.univ.image x) (by simp [←Multiset.coe_eq_coe.mpr (Multiset.eq_coe_min_of_fin_nonempty hien1 n x h1)] )) ^ 2 := sorry

end sum_square_lt_max_min_diff_squared_l332_332555


namespace seating_arrangements_correct_l332_332257

theorem seating_arrangements_correct :
  let brothers := [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)] in
  let first_row := [(1,1), (2,1), (3,1)] in
  let second_row := [(1,2), (2,2), (3,2)] in
  let no_adjacent (a b : (ℕ × ℕ)) := a.1 ≠ b.1 ∨ a.2 = b.2 in
  let columns_diff (a b : (ℕ × ℕ)) := a.2 ≠ b.2 in
  let is_valid_arrangement (arr : list (ℕ × ℕ)) :=
    (∀ i ∈ first_row, ∀ j ∈ second_row, no_adjacent i j) ∧
    (∀ i ∈ first_row, ∀ j ∈ second_row, columns_diff i j) in
  let arrangements := list.permutations brothers in
  (∃ arr ∈ arrangements, is_valid_arrangement arr) → arrangements.length = 12 :=
sorry

end seating_arrangements_correct_l332_332257


namespace parabola_equation_proposition_P_l332_332637

/-- 
The vertex of parabola C is at the origin O, 
its focus F is on the positive half of the y-axis, 
and its directrix l is tangent to the circle x^2 + y^2 = 4.
 -/
def parabola_vertex := (0, 0 : ℝ)
def parabola_focus (f : ℝ) (hf : f > 0) := (0, f)
def circle := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4 }

/-- 
Question (Ⅰ): Find the equation of parabola C; 
translate to Lean 
-/
theorem parabola_equation : 
  ∃ p > 0, ∀ x y : ℝ, (x, y) ∈ parabola_vertex ∪ parabola_focus p (by linarith) ∩ circle → x^2 = 2 * p * y :=
sorry

/--
Given that the line l and parabola C intersect at points A and B, 
and proposition P: "If line l passes through the fixed point (0, 1), 
then ⟪ ⟪OA⟫, ⟪OB⟫ ⟧ = -7";
translate to Lean
 -/
def line_passing_through_fixed_point (k : ℝ) := { p : ℝ × ℝ | p.2 = k * p.1 + 1 }
def parabola_C := { p : ℝ × ℝ | p.1 ^ 2 = 8 * p.2 }
def OA (A : ℝ × ℝ) := A
def OB (B : ℝ × ℝ) := B
def dot_product (A B : ℝ × ℝ) := A.1 * B.1 + A.2 * B.2

theorem proposition_P (A B : ℝ × ℝ) (hA : A ∈ line_passing_through_fixed_point 0) (hB : B ∈ parabola_C) :
  dot_product (OA A) (OB B) = -7 :=
sorry

end parabola_equation_proposition_P_l332_332637


namespace sum_of_solutions_l332_332967

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332967


namespace smallest_k_for_a_l332_332553

theorem smallest_k_for_a (a n : ℕ) (h : 10 ^ 2013 ≤ a^n ∧ a^n < 10 ^ 2014) : ∀ k : ℕ, k < 46 → ∃ n : ℕ, (10 ^ (k - 1)) ≤ a ∧ a < 10 ^ k :=
by sorry

end smallest_k_for_a_l332_332553


namespace total_spending_l332_332846

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end total_spending_l332_332846


namespace lisa_socks_total_l332_332167

def total_socks (initial : ℕ) (sandra : ℕ) (cousin_ratio : ℕ → ℕ) (mom_extra : ℕ → ℕ) : ℕ :=
  initial + sandra + cousin_ratio sandra + mom_extra initial

def cousin_ratio (sandra : ℕ) : ℕ := sandra / 5
def mom_extra (initial : ℕ) : ℕ := 3 * initial + 8

theorem lisa_socks_total :
  total_socks 12 20 cousin_ratio mom_extra = 80 := by
  sorry

end lisa_socks_total_l332_332167


namespace min_expression_value_l332_332386

theorem min_expression_value : ∀ x y : ℝ, 
  (sqrt (2 * (1 + cos (2 * x))) - sqrt (9 - sqrt 7) * sin x + 1) * 
  (3 + 2 * sqrt (13 - sqrt 7) * cos y - cos (2 * y)) ≥ -19 := 
by
  sorry

end min_expression_value_l332_332386


namespace smaller_angle_3_40_l332_332696

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332696


namespace max_unique_sums_l332_332321

def coin_values : Finset ℕ := {1, 5, 10, 25}

theorem max_unique_sums : 
  finset.card (finset.image (λ (p : ℕ × ℕ), p.1 + p.2) (coin_values ×ˢ coin_values)) = 9 :=
sorry

end max_unique_sums_l332_332321


namespace stratified_sampling_l332_332783

theorem stratified_sampling (E M Y T : ℕ) (e m y : ℕ) :
  E = 25 → M = 35 → Y = 40 → T = 40 → e = 10 → m = 14 → y = 16 → e + m + y = T :=
by
  intros hE hM hY hT he hm hy
  rw [he, hm, hy]
  exact hT


end stratified_sampling_l332_332783


namespace find_expression_l332_332616

-- Define p and q according to the given conditions
variables {a c : ℝ}

def p (x : ℝ) := a * x + c
def q (x : ℝ) := (x - 3) ^ 2

theorem find_expression : p (1) = 4 ∧ q (3) = 0 ∧ (∀ x, q x = (x - 3) ^ 2) →
  p (x) + q (x) = x^2 + (a - 6) * x + 13 :=
begin
  intros h,
  sorry
end

end find_expression_l332_332616


namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l332_332265

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l332_332265


namespace cost_of_monogramming_each_backpack_l332_332025

def number_of_backpacks : ℕ := 5
def original_price_per_backpack : ℝ := 20.00
def discount_rate : ℝ := 0.20
def total_cost : ℝ := 140.00

theorem cost_of_monogramming_each_backpack : 
  (total_cost - (number_of_backpacks * (original_price_per_backpack * (1 - discount_rate)))) / number_of_backpacks = 12.00 :=
by
  sorry 

end cost_of_monogramming_each_backpack_l332_332025


namespace range_of_m_l332_332988

variable {x m : ℝ}

def condition_p (x : ℝ) : Prop := |x - 3| ≤ 2
def condition_q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, ¬(condition_p x) → ¬(condition_q x m)) ∧ ¬(∀ x, ¬(condition_q x m) → ¬(condition_p x)) →
  2 < m ∧ m < 4 := 
sorry

end range_of_m_l332_332988


namespace evaluate_expression_l332_332373

-- Definition of the imaginary unit and its properties
def i := Complex.I

-- Given the properties of i
lemma i_squared: i^2 = -1 := by 
  rw [Complex.I_sq, Complex.neg_one_re_add_im]; simp

lemma i_fourth_power: i^4 = 1 := by
  rw [pow_two, pow_two, i_squared]; ring

-- Statement to be proven
theorem evaluate_expression : i^(-5) + i^(8) + i^(14) - i^(22) = 1 - i := 
  by
    -- We can use the conditions (i^2 = -1 and i^4 = 1) in this proof
    sorry -- Proof details are omitted as per instruction

end evaluate_expression_l332_332373


namespace problem_statement_l332_332395

def units_digit (n : ℕ) : ℕ := ((n * (n + 1)) / 2) % 10

def sum_units_digits (n : ℕ) : ℕ :=
  nat.rec_on n 0 (λ n sum, sum + units_digit (n+1))

theorem problem_statement : (sum_units_digits 2017) % 1000 = 69 :=
by sorry

end problem_statement_l332_332395


namespace henry_final_price_l332_332780

-- Definitions based on the conditions in the problem
def price_socks : ℝ := 5
def price_tshirt : ℝ := price_socks + 10
def price_jeans : ℝ := 2 * price_tshirt
def discount_jeans : ℝ := 0.15 * price_jeans
def discounted_price_jeans : ℝ := price_jeans - discount_jeans
def sales_tax_jeans : ℝ := 0.08 * discounted_price_jeans
def final_price_jeans : ℝ := discounted_price_jeans + sales_tax_jeans

-- Statement to prove
theorem henry_final_price : final_price_jeans = 27.54 := by
  sorry

end henry_final_price_l332_332780


namespace condition_necessary_but_not_sufficient_l332_332308

theorem condition_necessary_but_not_sufficient :
  (∀ x : ℝ, ((1 / 3) ^ x < 1) → (x > 0)) ∧ 
  (∃ x : ℝ, (0 < x ∧ x < 1) → (1 / x > 1)) ∧ 
  (∀ x : ℝ, (1 / x > 1) → ((1 / 3) ^ x < 1)) ∧ ¬(∀ x : ℝ, (((1 / 3) ^ x < 1) → (0 < x ∧ x < 1))) :=
by
  sorry

end condition_necessary_but_not_sufficient_l332_332308


namespace clock_angle_3_40_l332_332673

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332673


namespace remainder_sum_mod9_l332_332390

def a1 := 8243
def a2 := 8244
def a3 := 8245
def a4 := 8246

theorem remainder_sum_mod9 : ((a1 + a2 + a3 + a4) % 9) = 7 :=
by
  sorry

end remainder_sum_mod9_l332_332390


namespace exists_n_with_70_pairs_l332_332803

/-- Define our subset S and the properties. --/
def S : Finset ℕ := sorry -- S is a finite subset of ℕ
def U : Finset ℕ := Finset.range 2001

/-- Assumptions --/
axiom h_subset : S ⊆ U
axiom h_card : S.card = 401

/-- The Statement to Prove --/
theorem exists_n_with_70_pairs :
  ∃ n : ℕ, n > 0 ∧ (Finset.filter (λ x, x + n ∈ S) S).card ≥ 70 :=
by
  sorry

end exists_n_with_70_pairs_l332_332803


namespace twelfth_is_monday_l332_332074

def days_of_week : Type := {d // d < 7}

def starts_on_friday (d : ℕ) : days_of_week := ⟨d % 7, by linarith [Nat.mod_lt d (by norm_num)]⟩

-- Condition 1: There are exactly 5 Fridays in the month (which has at least 30 days)
def has_five_fridays (days_in_month : ℕ) : Prop := 
  ∃ (start : ℕ), 
    (start % 7 ≠ 5) ∧ 
    (days_in_month > 28 ∧ days_in_month < 32) ∧ -- At least 30 days to have 5 Fridays
    ∃ f, ∀ i, starts_on_friday(start + 7*i) = f → (i < 5)

-- Condition 2: The first day of the month is not a Friday
def first_not_friday (start : ℕ) : Prop := start % 7 ≠ 5

-- Condition 3: The last day of the month is not a Friday
def last_not_friday (start days_in_month : ℕ) : Prop := (start + days_in_month - 1) % 7 ≠ 5
    
theorem twelfth_is_monday (days_in_month : ℕ) (start : ℕ) 
  (h_five_fridays : has_five_fridays days_in_month)
  (h_first_not_friday : first_not_friday start)
  (h_last_not_friday : last_not_friday start days_in_month) : 
  (start + 11) % 7 = 1 :=
sorry

end twelfth_is_monday_l332_332074


namespace rain_in_first_hour_l332_332128

theorem rain_in_first_hour :
  ∃ x : ℕ, (let rain_second_hour := 2 * x + 7 in x + rain_second_hour = 22) ∧ x = 5 :=
by
  sorry

end rain_in_first_hour_l332_332128


namespace third_median_length_is_9_l332_332509

noncomputable def length_of_third_median_of_triangle (m₁ m₂ m₃ area : ℝ) : Prop :=
  ∃ median : ℝ, median = m₃

theorem third_median_length_is_9 :
  length_of_third_median_of_triangle 5 7 9 (6 * Real.sqrt 10) :=
by
  sorry

end third_median_length_is_9_l332_332509


namespace count_powers_of_2_not_powers_of_4_less_than_500000_l332_332035

theorem count_powers_of_2_not_powers_of_4_less_than_500000 :
  (finset.card (finset.filter (λ n, (∃ k, n = 2^k) ∧ ¬ (∃ m, n = 4^m))
                              (finset.range 500000))) = 9 :=
by {
  sorry
}

end count_powers_of_2_not_powers_of_4_less_than_500000_l332_332035


namespace children_coins_distribution_l332_332139

theorem children_coins_distribution (n : ℕ) (a : Fin n → ℕ) :
  3 < n →
  (\sum i, i * a i) % n = (n * (n + 1) / 2) % n →
  (∃ k : ℕ, (∀ i : Fin n, a i + k = 1) ∧ k ≥ 0) :=
by sorry

end children_coins_distribution_l332_332139


namespace total_shaded_cubes_4x4x4_l332_332790

theorem total_shaded_cubes_4x4x4 :
  let large_cube := { x : ℕ // x < 64 }
  let face_shaded_pattern (face : ℕ) (x y : ℕ) : Prop := (x < 2 ∧ y < 2) ∨ (x > 1 ∧ y < 2) ∨ (x < 2 ∧ y > 1) ∨ (x > 1 ∧ y > 1)
  let cubes_shaded (x y z : ℕ) : Prop := 
    (face_shaded_pattern 0 x y ∨ face_shaded_pattern 1 x y) ∧
    (face_shaded_pattern 2 x z ∨ face_shaded_pattern 3 x z) ∧
    (face_shaded_pattern 4 y z ∨ face_shaded_pattern 5 y z)
  in ∑ x in range 4, ∑ y in range 4, ∑ z in range 4, if cubes_shaded x y z then 1 else 0 = 44 := sorry

end total_shaded_cubes_4x4x4_l332_332790


namespace circumcenter_AIC_on_circumcircle_ABC_l332_332146

open EuclideanGeometry

variables {A B C I K O : Type} -- Represent points in the Euclidean plane

-- Define the conditions
variable (hI : is_incenter A B C I)
variable (hCircumcircleABC : is_circumcircle A B C O)
variable (hBisectorBI : is_angle_bisector B I K)
variable (hKOnCircumcircle : lies_on_circumcircle K A B C)

-- Define the goal
theorem circumcenter_AIC_on_circumcircle_ABC :
  lies_on_circumcircle (circumcenter A I C) A B C :=
begin
  sorry
end

end circumcenter_AIC_on_circumcircle_ABC_l332_332146


namespace shifted_parabola_coefficients_l332_332290

theorem shifted_parabola_coefficients:
  let f (x : ℝ) := 2 * x^2 - x + 7
  let g (x : ℝ) := 2 * x^2 - 25 * x + 85
  g = (λ x, f (x - 6)) → 
  (2 - 25 + 85) = 62 := 
by {
  intros,
  sorry
}

end shifted_parabola_coefficients_l332_332290


namespace compare_squares_l332_332002

theorem compare_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end compare_squares_l332_332002


namespace sum_of_solutions_l332_332963

-- Define the initial condition
def initial_equation (x : ℝ) : Prop := (x - 8) ^ 2 = 49

-- Define the conclusion we want to reach
def sum_of_solutions_is_16 : Prop :=
  ∃ x1 x2 : ℝ, initial_equation x1 ∧ initial_equation x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 16

theorem sum_of_solutions :
  sum_of_solutions_is_16 :=
by
  sorry

end sum_of_solutions_l332_332963


namespace smaller_angle_at_3_40_l332_332732

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332732


namespace small_angle_at_3_40_is_130_degrees_l332_332755

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332755


namespace twelve_is_monday_l332_332085

def Weekday := {d : String // d ∈ ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

def not_friday (d: Weekday) : Prop := d.val ≠ "Friday"

def has_exactly_five_fridays (first_friday: nat) (days_in_month: nat) : Prop :=
  first_friday + 28 <= days_in_month ∧
  first_friday + 21 <= days_in_month ∧
  first_friday + 14 <= days_in_month ∧
  first_friday + 7 <= days_in_month ∧
  first_friday > 0 ∧ days_in_month <= 31

noncomputable def compute_day_of_week (start_day: String) (n: nat) : Weekday :=
  sorry

theorem twelve_is_monday (start_day: Weekday) (days_in_month: nat) :
    has_exactly_five_fridays 2 days_in_month
  → not_friday start_day
  → not_friday (compute_day_of_week start_day.val days_in_month)
  → compute_day_of_week start_day.val 12 = ⟨"Monday", by sorry⟩ :=
begin
  sorry
end

end twelve_is_monday_l332_332085


namespace range_of_f_on_interval_l332_332003

noncomputable def f (x : ℝ) : ℝ := 1 / x^2

theorem range_of_f_on_interval :
  set.range (λ x, f x) ∩ set.Icc 1 (3 : ℝ) = set.Icc (1 / 9) 1 :=
sorry

end range_of_f_on_interval_l332_332003


namespace smaller_angle_3_40_pm_l332_332747

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332747


namespace log2_T_approx_l332_332143

noncomputable def T : ℝ :=
  ( (2 + 1) ^ 2011 + (2 - 1) ^ 2011 ) / 2

theorem log2_T_approx : 
  log2 T ≈ 3184.55796 := 
sorry

end log2_T_approx_l332_332143


namespace cost_price_of_product_is_100_l332_332797

theorem cost_price_of_product_is_100 
  (x : ℝ) 
  (h : x * 1.2 * 0.9 - x = 8) : 
  x = 100 := 
sorry

end cost_price_of_product_is_100_l332_332797


namespace find_m_l332_332501

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem find_m (a b m : ℝ) (h1 : f m a b = 0) (h2 : 3 * m^2 + 2 * a * m + b = 0)
  (h3 : f (m / 3) a b = 1 / 2) (h4 : m ≠ 0) : m = 3 / 2 :=
  sorry

end find_m_l332_332501


namespace ratio_of_root_differences_l332_332462

theorem ratio_of_root_differences (a b A B C D : ℝ) (hA : A = real.sqrt (4 - 4 * a))
  (hB : B = real.sqrt (b^2 + 4)) (hC : C = 1 / 2 * real.sqrt (b^2 - 12 * b - 24 * a + 28))
  (hD : D = 1 / 2 * real.sqrt (9 * b^2 - 12 * b + 8 * a + 28)) (hne : |A| ≠ |B|) :
  (C^2 - D^2) / (A^2 - B^2) = 2 :=
by
  sorry

end ratio_of_root_differences_l332_332462


namespace clock_angle_l332_332708

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332708


namespace face_opposite_P_is_D_l332_332225

-- Definitions based on conditions in a)
def is_rhombicuboctahedron (net : Type) : Prop := sorry
def face (net : Type) := Type
constant P : face nat
constant D : face nat

-- Theorem statement based on b) and c)
theorem face_opposite_P_is_D (net : Type) [is_rhombicuboctahedron net] (opposite : face net → face net) :
  opposite P = D :=
sorry

end face_opposite_P_is_D_l332_332225


namespace sum_of_roots_eq_seventeen_l332_332917

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332917


namespace clock_angle_3_40_l332_332715

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332715


namespace small_angle_at_3_40_is_130_degrees_l332_332756

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332756


namespace rate_per_kg_first_batch_l332_332350

/-- This theorem proves the rate per kg of the first batch of wheat. -/
theorem rate_per_kg_first_batch (x : ℝ) 
  (h1 : 30 * x + 20 * 14.25 = 285 + 30 * x) 
  (h2 : (30 * x + 285) * 1.3 = 819) : 
  x = 11.5 := 
sorry

end rate_per_kg_first_batch_l332_332350


namespace day_of_12th_l332_332080

theorem day_of_12th (month : Type) [decidable_eq month] [fintype month] 
  (is_friday : month → Prop) (is_first_day : month → Prop) (is_last_day : month → Prop)
  (days : list month) (nth_day : Π (n : ℕ), month)
  (five_fridays : ∃ (n : ℕ), (is_friday ∘ nth_day) '' (finset.range (min n 7)) = {5})
  (not_first_day_friday : ∀ d, is_first_day d → ¬ is_friday d)
  (not_last_day_friday : ∀ d, is_last_day d → ¬ is_friday d) : 
  nth_day 12 = "Monday" := 
sorry

end day_of_12th_l332_332080


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332886

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332886


namespace vector_b_satisfies_conditions_l332_332542

noncomputable def a : ℝ × ℝ × ℝ := (3, 2, 4)
noncomputable def b : ℝ × ℝ × ℝ := (7, -2, 3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem vector_b_satisfies_conditions :
  dot_product a b = 20 ∧ cross_product a b = (1, -15, 5) :=
by
  sorry

end vector_b_satisfies_conditions_l332_332542


namespace curve_and_line_intersect_distance_l332_332522

noncomputable def curve_eq (x y : ℝ) : Prop := (x - 1)^2 / 4 + y^2 / 3 = 1
noncomputable def line_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0

theorem curve_and_line_intersect_distance :
  (∀ x y : ℝ, (curve_eq x y ↔ (∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ ρ * (cos θ) = x ∧ ρ * (sin θ) = y ∧ ρ = 3 / (2 - cos θ)))
  ∧ (line_eq = λ x y, ∃ t : ℝ, x = 3 + t ∧ y = 2 + 2 * t)
  ∧ ({x y | curve_eq x y}.inter {x y | line_eq x y}).card = 2
  ∧ |(λ (x1 y1 x2 y2 : ℝ), dist (⟨x1, y1⟩:ℝ × ℝ) (⟨x2, y2⟩:ℝ × ℝ)) A B| = 60/19) :=
by
  sorry

end curve_and_line_intersect_distance_l332_332522


namespace solve_problem_l332_332028

noncomputable def solution_set : Set ℤ := {x | abs (7 * x - 5) ≤ 9}

theorem solve_problem : solution_set = {0, 1, 2} := by
  sorry

end solve_problem_l332_332028


namespace find_ab_l332_332779

theorem find_ab 
(a b : ℝ) 
(h1 : a + b = 2) 
(h2 : a * b = 1 ∨ a * b = -1) :
(a = 1 ∧ b = 1) ∨
(a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
(a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
sorry

end find_ab_l332_332779


namespace smallest_number_condition_l332_332286

theorem smallest_number_condition
  (x : ℕ)
  (h1 : (x - 24) % 5 = 0)
  (h2 : (x - 24) % 10 = 0)
  (h3 : (x - 24) % 15 = 0)
  (h4 : (x - 24) / 30 = 84)
  : x = 2544 := 
sorry

end smallest_number_condition_l332_332286


namespace case1_case2_case3_l332_332259

-- Definitions from conditions
def tens_digit_one : ℕ := sorry
def units_digit_one : ℕ := sorry
def units_digit_two : ℕ := sorry
def tens_digit_two : ℕ := sorry
def sum_units_digits_ten : Prop := units_digit_one + units_digit_two = 10
def same_digit : ℕ := sorry
def sum_tens_digits_ten : Prop := tens_digit_one + tens_digit_two = 10

-- The proof problems
theorem case1 (A B D : ℕ) (hBplusD : B + D = 10) :
  (10 * A + B) * (10 * A + D) = 100 * (A^2 + A) + B * D :=
sorry

theorem case2 (A B C : ℕ) (hAplusC : A + C = 10) :
  (10 * A + B) * (10 * C + B) = 100 * A * C + 100 * B + B^2 :=
sorry

theorem case3 (A B C : ℕ) (hAplusB : A + B = 10) :
  (10 * A + B) * (10 * C + C) = 100 * A * C + 100 * C + B * C :=
sorry

end case1_case2_case3_l332_332259


namespace three_digit_numbers_with_234_count_l332_332037

theorem three_digit_numbers_with_234_count :
  let numbers := range 100 1000 in
  let has_two_three_four :=
    λ n : ℕ, (n / 100 = 2 ∨ (n / 10) % 10 = 2 ∨ n % 10 = 2) ∧
             (n / 100 = 3 ∨ (n / 10) % 10 = 3 ∨ n % 10 = 3) ∧
             (n / 100 = 4 ∨ (n / 10) % 10 = 4 ∨ n % 10 = 4) in
  (numbers.toList.filter has_two_three_four).length = 126 :=
by
  sorry

end three_digit_numbers_with_234_count_l332_332037


namespace three_letter_sets_initials_eq_1000_l332_332475

theorem three_letter_sets_initials_eq_1000 :
  (∃ (A B C : Fin 10), true) = 1000 := 
sorry

end three_letter_sets_initials_eq_1000_l332_332475


namespace least_positive_multiple_of_15_with_digit_product_multiple_of_15_l332_332282

theorem least_positive_multiple_of_15_with_digit_product_multiple_of_15 : 
  ∃ (n : ℕ), 
    n % 15 = 0 ∧ 
    (∀ k, k % 15 = 0 ∧ (∃ m : ℕ, m < n ∧ m % 15 = 0 ∧ 
    list.prod (nat.digits 10 m) % 15 == 0) 
    → list.prod (nat.digits 10 k) % 15 == 0) 
    ∧ list.prod (nat.digits 10 n) % 15 = 0 
    ∧ n = 315 :=
sorry

end least_positive_multiple_of_15_with_digit_product_multiple_of_15_l332_332282


namespace geometry_problem_l332_332196

theorem geometry_problem
  (A B C D E M O : Point)
  (circle : Circle)
  (h1 : Tangent A B circle)
  (h2 : Tangent A C circle)
  (secant : Secant D E circle)
  (h3 : Midpoint M B C)
  : 
  -- First claim: BM^2 = DM * ME
  (dist B M * dist B M = dist D M * dist M E)
  -- Second claim: ∠DME = 2 * ∠DCE
  ∧ 
  (angle D M E = 2 * angle D C E)
  -- Third claim: ∠BEM = ∠DEC
  ∧ 
  (angle B E M = angle D E C) :=
sorry

end geometry_problem_l332_332196


namespace number_of_powers_of_2_not_powers_of_4_below_500000_l332_332033

theorem number_of_powers_of_2_not_powers_of_4_below_500000 : 
  (Set.filter (λ n, ∃ k, n = 2^k ∧ n < 500000 ∧ ¬ ∃ m, n = 4^m) {n : ℕ | n > 0}).card = 9 :=
sorry

end number_of_powers_of_2_not_powers_of_4_below_500000_l332_332033


namespace Rachel_money_left_l332_332187

theorem Rachel_money_left 
  (money_earned : ℕ)
  (lunch_fraction : ℚ)
  (clothes_percentage : ℚ)
  (dvd_cost : ℚ)
  (supplies_percentage : ℚ)
  (money_left : ℚ) :
  money_earned = 200 →
  lunch_fraction = 1 / 4 →
  clothes_percentage = 15 / 100 →
  dvd_cost = 24.50 →
  supplies_percentage = 10.5 / 100 →
  money_left = 74.50 :=
by
  intros h_money h_lunch h_clothes h_dvd h_supplies
  sorry

end Rachel_money_left_l332_332187


namespace angle_bisector_length_l332_332527

-- Definitions for the given conditions
def PQ : ℝ := 4
def PR : ℝ := 8
def cos_angle_P : ℝ := 1 / 10

-- Statement asserting the proof problem
theorem angle_bisector_length :
  ∃ (PS : ℝ), PS = Real.sqrt 22.755 :=
sorry

end angle_bisector_length_l332_332527


namespace intersection_complement_M_N_l332_332431

def M := { x : ℝ | x ≤ 1 / 2 }
def N := { x : ℝ | x^2 ≤ 1 }
def complement_M := { x : ℝ | x > 1 / 2 }

theorem intersection_complement_M_N :
  (complement_M ∩ N = { x : ℝ | 1 / 2 < x ∧ x ≤ 1 }) :=
by
  sorry

end intersection_complement_M_N_l332_332431


namespace increasing_range_of_a_l332_332011

-- Define the function f piecewise
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 * (a - 2) * x - 1 else a^x

-- Define conditions
variables (a : ℝ)
noncomputable theory

-- Claim to prove
theorem increasing_range_of_a (h0 : 0 < a) (h1 : a ≠ 1) : 
  (∀ x y : ℝ, 0 < x → x < y → f a x < f a y) ↔ (4 / 3 ≤ a ∧ a ≤ 2) := sorry

end increasing_range_of_a_l332_332011


namespace sarith_laps_l332_332535

theorem sarith_laps 
  (k_speed : ℝ) (s_speed : ℝ) (k_laps : ℝ) (s_laps : ℝ) (distance_ratio : ℝ) :
  k_speed = 3 * s_speed →
  distance_ratio = 1 / 2 →
  k_laps = 12 →
  s_laps = (k_laps * 2 / 3) →
  s_laps = 8 :=
by
  intros
  sorry

end sarith_laps_l332_332535


namespace cats_favorite_number_l332_332359

/-- Definitions and conditions for the problem --/
def is_two_digit_perfect_square (n : ℕ) : Prop :=
  n ∈ {16, 25, 36, 49, 64, 81}

def digit_set (n : ℕ) : set ℕ :=
  {n / 10, n % 10}

def unique_digit_property (n : ℕ) : Prop :=
  ∃ d ∈ digit_set(n), ∀ m, is_two_digit_perfect_square m ∧ d ∈ digit_set(m) → m = n

def ambiguous_sum_difference (n : ℕ) : Prop :=
  let s := (n / 10) + (n % 10) in
  let d := abs ((n / 10) - (n % 10)) in
  ∀ m, is_two_digit_perfect_square m ∧
    ((digit_set(m).sum = s) ∨ (abs ((m / 10) - (m % 10)) = d)) →
    m = n → m = 25

/-- Prove that Cat's favorite number is 25 given the conditions. --/
theorem cats_favorite_number : ∃ n, is_two_digit_perfect_square n ∧ unique_digit_property n ∧ ambiguous_sum_difference n :=
by
  use 25
  split
  . exact dec_trivial -- 25 is a two-digit perfect square.
  split
  . sorry -- Need to show the unique digit property for 25.
  . sorry -- Need to show the ambiguous sum and difference conditions for 25.

end cats_favorite_number_l332_332359


namespace area_of_triangle_QRS_l332_332588

-- Definitions for points in 3-dimensional space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definitions of distances and angles
def dist (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2 + (B.z - A.z)^2)

def angle (A B C : Point) : ℝ := sorry  -- Assuming there is a function to calculate angle in 3D

-- The conditions from the problem
variables (P Q R S T : Point)
variable (h1 : dist P Q = 3)
variable (h2 : dist Q R = 3)
variable (h3 : dist R S = 3)
variable (h4 : dist S T = 3)
variable (h5 : dist T P = 3)
variable (h6 : angle P Q R = 120)
variable (h7 : angle R S T = 120)
variable (h8 : angle S T P = 120)
variable (h9 : ∃ (plane1 plane2 : Point → Prop), 
  (plane1 P) ∧ (plane1 Q) ∧ (plane1 R) ∧ (plane2 S) ∧ (plane2 T) ∧ 
  (∀ (A B : Point), plane1 A ∧ plane2 B → (A.z = B.z)))

-- Lean statement to prove the area of triangle QRS
theorem area_of_triangle_QRS : 
  ∃ (A : ℝ), A = (9 * real.sqrt 3) / 4 :=
by
  sorry

end area_of_triangle_QRS_l332_332588


namespace sum_of_solutions_eqn_l332_332941

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332941


namespace find_a6_l332_332519

variable {a : ℕ → ℝ} -- Sequence a is indexed by natural numbers and the terms are real numbers.

-- Conditions
def a_is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)
def a1_eq_4 (a : ℕ → ℝ) := a 1 = 4
def a3_eq_a2_mul_a4 (a : ℕ → ℝ) := a 3 = a 2 * a 4

theorem find_a6 (a : ℕ → ℝ) 
  (h1 : a_is_geom_seq a)
  (h2 : a1_eq_4 a)
  (h3 : a3_eq_a2_mul_a4 a) : 
  a 6 = 1 / 8 ∨ a 6 = - (1 / 8) := 
by 
  sorry

end find_a6_l332_332519


namespace largest_k_2520_l332_332829

def highest_power_of_prime_factor (n : ℕ) (p : ℕ) : ℕ :=
  ∑ i in finset.Ico 1 (n+1), n / p^i

theorem largest_k_2520 :
  let n := 2520
  let k_max := 629
  let factors := (2^3) * (3^2) * 5 * 7
  let power_2 := highest_power_of_prime_factor n 2
  let power_3 := highest_power_of_prime_factor n 3
  let power_5 := highest_power_of_prime_factor n 5
  let power_7 := highest_power_of_prime_factor n 7
  shows (2520^k_max) ∣ nat.factorial n :=
by
  let n := 2520
  let k_max := 629
  let power_2 := highest_power_of_prime_factor n 2
  let power_3 := highest_power_of_prime_factor n 3
  let power_5 := highest_power_of_prime_factor n 5
  let power_7 := highest_power_of_prime_factor n 7
  have power_2_value : power_2 = 2514 := by sorry
  have power_3_value : power_3 = 1258 := by sorry
  have power_5_value : power_5 = 628 := by sorry
  have power_7_value : power_7 = 419 := by sorry
  have key : k_max = 629 := by sorry
  have correct_division : (2520 ^ k_max) ∣ (nat.factorial n) := by sorry
  exact correct_division

end largest_k_2520_l332_332829


namespace day12_is_monday_l332_332095

-- Define the days of the week
inductive WeekDay
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open WeekDay

-- Define the problem using the conditions
def five_fridays_in_month (first_day : WeekDay) (last_day : WeekDay) : Prop :=
  first_day ≠ Friday ∧ last_day ≠ Friday ∧
  ((first_day = Monday ∨ first_day = Tuesday ∨ first_day = Wednesday ∨ first_day = Thursday ∨
    first_day = Saturday ∨ first_day = Sunday) ∧
  ∃ fridays : Finset ℕ,
  fridays.card = 5 ∧
  ∀ n ∈ fridays, (n % 7 = (5 - WeekDay.recOn first_day 6 0 1 2 3 4 5)) ∧
  fridays ⊆ Finset.range 31 ∧
  1 ∉ fridays ∧ (31 - Finset.max' fridays sorry) % 7 ≠ 0 )

-- Given the problem, prove that the 12th day is a Monday
theorem day12_is_monday (first_day last_day : WeekDay)
  (h : five_fridays_in_month first_day last_day) : 
  (12 % 7 + WeekDay.recOn first_day 6 0 1 2 3 4 5) % 7 = 0 :=
sorry

end day12_is_monday_l332_332095


namespace subtract_vectors_l332_332986

def vec_a : ℤ × ℤ × ℤ := (5, -3, 2)
def vec_b : ℤ × ℤ × ℤ := (-2, 4, 1)
def vec_result : ℤ × ℤ × ℤ := (9, -11, 0)

theorem subtract_vectors :
  vec_a - 2 • vec_b = vec_result :=
by sorry

end subtract_vectors_l332_332986


namespace smallest_root_of_g_l332_332880

def g (x : ℝ) : ℝ := 10 * x^4 - 14 * x^2 + 4

theorem smallest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → x ≤ y :=
by
  use -1
  split
  sorry -- Proof that g(-1) = 0
  intro y
  intro hy
  sorry -- Proof that -1 is the smallest root

end smallest_root_of_g_l332_332880


namespace sum_of_roots_eq_seventeen_l332_332918

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332918


namespace smaller_angle_3_40_l332_332691

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332691


namespace exists_sequence_epsilon_l332_332138

theorem exists_sequence_epsilon
  (α : ℝ) (hα : 0 < α ∧ α < 1) :
  ∃ (ε : ℕ → ℕ), (∀ n, ε n = 0 ∨ ε n = 1) ∧
    ∀ (n : ℕ), 2 ≤ n →
      (let s_n := ∑ i in (Finset.range n).map (λ i, i + 1),
                   (ε i) / (↑i + n - 1) / (↑(i + n)) in
        0 ≤ α - 2 * n * s_n ∧ α - 2 * n * s_n ≤ 2 / (n + 1)) :=
sorry

end exists_sequence_epsilon_l332_332138


namespace math_problem_l332_332045

theorem math_problem
  (x y : ℚ)
  (h1 : x + y = 11 / 17)
  (h2 : x - y = 1 / 143) :
  x^2 - y^2 = 11 / 2431 :=
by
  sorry

end math_problem_l332_332045


namespace company_employees_january_l332_332360

def employees_in_january (d: ℕ): ℕ := (d: ℚ * 100 / 115).toInt

theorem company_employees_january
  (employees_december: ℕ)
  (hpec: employees_december = 500)
  (hperc_inc: ∀ j, employees_december = j * 115 / 100): 
  employees_january employees_december = 435 := by
  sorry

end company_employees_january_l332_332360


namespace sum_of_solutions_eq_16_l332_332978

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332978


namespace circumcircle_ASG_tangent_J_l332_332010

-- Given conditions:
variables {O J A B C D E F G S : Point}
variables {circumcircle : Triangle → Circle}
variables {inscribed_circle : Circle → Circle}
variables {tangent : Point → Circle → Prop}
variables {line_segment : Point → Point → Line}

-- Conditions
axiom h1 : circumcircle(𝛥ABC) = O
axiom h2 : inscribed_circle(O) = J
axiom h3 : tangent D J ∧ tangent E J
axiom h4 : line_segment(F, G) ∧ tangent A O ∧ (AF = AG = AD)
axiom h5 : circumcircle(𝛥AFB) ∩ J = S

-- Conclusion to prove
theorem circumcircle_ASG_tangent_J : tangent S (circumcircle(𝛥ASG)) := 
sorry

end circumcircle_ASG_tangent_J_l332_332010


namespace product_of_three_smallest_prime_factors_of_180_l332_332761

/-- A helper predicate to define prime numbers -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Prime factorizes the number 180 into a list of its prime factors -/
def prime_factors (n : ℕ) : List ℕ :=
  match n with
  | 180 => [2, 2, 3, 3, 5]
  | _ => []

/-- Extracts the smallest unique prime factors from a list of prime factors -/
def smallest_primes(factors : List ℕ): List ℕ :=
  factors.eraseDup.filter is_prime

/-- Returns the product of elements in a list of natural numbers -/
def product (lst : List ℕ) : ℕ :=
  lst.foldr (· * ·) 1

theorem product_of_three_smallest_prime_factors_of_180 : product (smallest_primes (prime_factors 180)) = 30 := by
  sorry

end product_of_three_smallest_prime_factors_of_180_l332_332761


namespace Mabel_gave_away_daisies_l332_332566

-- Setting up the conditions
variables (d_total : ℕ) (p_per_daisy : ℕ) (p_remaining : ℕ)

-- stating the assumptions
def initial_petals (d_total p_per_daisy : ℕ) := d_total * p_per_daisy
def petals_given_away (d_total p_per_daisy p_remaining : ℕ) := initial_petals d_total p_per_daisy - p_remaining
def daisies_given_away (d_total p_per_daisy p_remaining : ℕ) := petals_given_away d_total p_per_daisy p_remaining / p_per_daisy

-- The main theorem
theorem Mabel_gave_away_daisies 
  (h1 : d_total = 5)
  (h2 : p_per_daisy = 8)
  (h3 : p_remaining = 24) :
  daisies_given_away d_total p_per_daisy p_remaining = 2 :=
sorry

end Mabel_gave_away_daisies_l332_332566


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332881

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332881


namespace sum_of_floor_sqrt_up_to_25_l332_332832

theorem sum_of_floor_sqrt_up_to_25 : (∑ n in Finset.range 25, nat.floor (Real.sqrt (n + 1))) = 75 :=
by
  sorry

end sum_of_floor_sqrt_up_to_25_l332_332832


namespace midpoint_coordinates_sum_l332_332000

theorem midpoint_coordinates_sum (M A B : ℝ × ℝ) 
  (hM : M = (1.5, 4.5)) 
  (hA : A = (1, 4)) 
  (hMidpoint : ∀ (x1 y1 x2 y2 : ℝ), M = ((x1 + x2) / 2, (y1 + y2) / 2) → A = (x1, y1) → B = (x2, y2)) :
  ∃ Bx By, B = (Bx, By) ∧ Bx + By = 7 :=
by
  intros
  use 2, 5
  split
  { unfold_projs
    rw [← hMidpoint, hA, hM]
    norm_num }
  { norm_num }
  sorry

end midpoint_coordinates_sum_l332_332000


namespace least_multiple_of_15_with_product_multiple_of_15_is_315_l332_332279

-- Let n be a positive integer
def is_multiple_of (n m : ℕ) := ∃ k, n = m * k

def is_product_multiple_of (digits : ℕ) (m : ℕ) := ∃ k, digits = m * k

-- The main theorem we want to state and prove
theorem least_multiple_of_15_with_product_multiple_of_15_is_315 (n : ℕ) 
  (h1 : is_multiple_of n 15) 
  (h2 : n > 0) 
  (h3 : is_product_multiple_of (n.digits.prod) 15) 
  : n = 315 := 
sorry

end least_multiple_of_15_with_product_multiple_of_15_is_315_l332_332279


namespace find_a_l332_332058

variable (y : ℝ) (a : ℝ)

theorem find_a (hy : y > 0) (h_expr : (a * y / 20) + (3 * y / 10) = 0.7 * y) : a = 8 :=
by
  sorry

end find_a_l332_332058


namespace trains_clear_each_other_time_l332_332272

theorem trains_clear_each_other_time ({L1 L2 S1 S2 : ℝ}) :
  L1 = 100 ∧ L2 = 220 ∧ S1 = 42 ∧ S2 = 30 →
  (1000 / 3600) * (S1 + S2) ≠ 0 →
  (L1 + L2) / (1000 / 3600 * (S1 + S2)) = 16 :=
by
  sorry

end trains_clear_each_other_time_l332_332272


namespace problem_statement_l332_332438

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (x : ℝ) (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f (x + π) = f x) 
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x = 1 - sin x) 
  (hx : π * 5 / 2 ≤ x ∧ x ≤ 3 * π) :
  f x = 1 - sin x :=
sorry

end problem_statement_l332_332438


namespace small_angle_at_3_40_is_130_degrees_l332_332752

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332752


namespace spadesuit_problem_l332_332397

def spadesuit (x y : ℝ) : ℝ := x - 1 / y

theorem spadesuit_problem : spadesuit 3 (spadesuit 3 3) = 21 / 8 := by
  sorry

end spadesuit_problem_l332_332397


namespace pascal_triangle_sum_squares_l332_332191

theorem pascal_triangle_sum_squares :
  (∑ i in Finset.range (1004), (Nat.choose 1003 i / Nat.choose 1004 i) ^ 2) -
  (∑ i in Finset.range (1003), (Nat.choose 1002 i / Nat.choose 1003 i) ^ 2) = 2 / 3 :=
by 
  sorry

end pascal_triangle_sum_squares_l332_332191


namespace max_sum_of_products_l332_332635

theorem max_sum_of_products (x : Fin 10 → ℝ) (h_nonneg : ∀ i, 0 ≤ x i) (h_sum : ∑ i, x i = 1) :
  (∑ i in Finset.range 9, x i * x (i + 1)) ≤ 0.25 :=
sorry

end max_sum_of_products_l332_332635


namespace number_of_three_letter_initials_l332_332477

theorem number_of_three_letter_initials : 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] in
  (length letters) ^ 3 = 1000 := 
by
  -- Definitions:
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  -- The problem reduces to computing 10^3 and showing it equals 1000.
  have length_letters : length letters = 10 := by sorry
  calc 
    (length letters) ^ 3
      = 10 ^ 3 : by rw length_letters
  ... = 1000 : by norm_num

end number_of_three_letter_initials_l332_332477


namespace distance_from_missouri_l332_332207

-- Open a namespace for our problem context
namespace DrivingDistance

-- Define the conditions
def distance_by_plane := 2000 -- Distance between Arizona and New York by plane in miles
def increase_rate := 0.40 -- Increase in distance by driving

def total_driving_distance : ℝ :=
  distance_by_plane * (1 + increase_rate)

def midway_distance : ℝ :=
  total_driving_distance / 2

-- Theorem to prove the distance from Missouri to New York by car
theorem distance_from_missouri :
  midway_distance = 1400 :=
by
  sorry

end DrivingDistance

end distance_from_missouri_l332_332207


namespace zero_of_function_l332_332244

theorem zero_of_function (x : ℝ) : (4 * x - 2 = 0) ↔ (x = 1 / 2) :=
begin
  sorry
end

end zero_of_function_l332_332244


namespace denny_followers_l332_332853

theorem denny_followers (initial_followers: ℕ) (new_followers_per_day: ℕ) (unfollowers_in_year: ℕ) (days_in_year: ℕ)
  (h_initial: initial_followers = 100000)
  (h_new_per_day: new_followers_per_day = 1000)
  (h_unfollowers: unfollowers_in_year = 20000)
  (h_days: days_in_year = 365):
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers_in_year = 445000 :=
by
  sorry

end denny_followers_l332_332853


namespace day12_is_monday_l332_332096

-- Define the days of the week
inductive WeekDay
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open WeekDay

-- Define the problem using the conditions
def five_fridays_in_month (first_day : WeekDay) (last_day : WeekDay) : Prop :=
  first_day ≠ Friday ∧ last_day ≠ Friday ∧
  ((first_day = Monday ∨ first_day = Tuesday ∨ first_day = Wednesday ∨ first_day = Thursday ∨
    first_day = Saturday ∨ first_day = Sunday) ∧
  ∃ fridays : Finset ℕ,
  fridays.card = 5 ∧
  ∀ n ∈ fridays, (n % 7 = (5 - WeekDay.recOn first_day 6 0 1 2 3 4 5)) ∧
  fridays ⊆ Finset.range 31 ∧
  1 ∉ fridays ∧ (31 - Finset.max' fridays sorry) % 7 ≠ 0 )

-- Given the problem, prove that the 12th day is a Monday
theorem day12_is_monday (first_day last_day : WeekDay)
  (h : five_fridays_in_month first_day last_day) : 
  (12 % 7 + WeekDay.recOn first_day 6 0 1 2 3 4 5) % 7 = 0 :=
sorry

end day12_is_monday_l332_332096


namespace smaller_angle_between_hands_at_3_40_l332_332683

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332683


namespace count_powers_of_2_not_powers_of_4_less_than_500000_l332_332036

theorem count_powers_of_2_not_powers_of_4_less_than_500000 :
  (finset.card (finset.filter (λ n, (∃ k, n = 2^k) ∧ ¬ (∃ m, n = 4^m))
                              (finset.range 500000))) = 9 :=
by {
  sorry
}

end count_powers_of_2_not_powers_of_4_less_than_500000_l332_332036


namespace rosencrans_wins_for_all_odd_l332_332653

-- Definitions for the game.
def game (n : ℕ) : Type := {p : ℕ // 5 ≤ n}

def valid_move (m1 m2 : game n) : Prop :=
  ∃ x y : game n, (m1 = x ∨ m1 = y) ∧ (m2 = x ∨ m2 = y)

def rosencrans_wins (n : ℕ) : Prop :=
  (odd n) → (∃ m : ℕ, valid_move m m)

-- Statement to be proved:
theorem rosencrans_wins_for_all_odd (n : ℕ) : 
  5 ≤ n → odd n → rosencrans_wins n :=
sorry

end rosencrans_wins_for_all_odd_l332_332653


namespace bc_work_in_15_days_l332_332258

noncomputable def work_rate_a : ℝ := 1 / 24
def work_rate_sum_abc : ℝ := 1 / 6
def work_rate_sum_ab : ℝ := 1 / 10
def work_rate_sum_ca : ℝ := 1 / 20

theorem bc_work_in_15_days :
  let A := work_rate_a in
  let B := work_rate_sum_ab - A in
  let C := work_rate_sum_ca - A in
  B + C = 1 / 15 :=
by
  let A := work_rate_a
  let B := work_rate_sum_ab - A
  let C := work_rate_sum_ca - A
  have hB : B = 7 / 120 := by sorry
  have hC : C = 1 / 120 := by sorry
  calc
    B + C = 7 / 120 + 1 / 120 : by rw [hB, hC]
    ... = 8 / 120 : by ring
    ... = 1 / 15 : by norm_num

end bc_work_in_15_days_l332_332258


namespace range_of_t_l332_332233

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 1 else (1 / 5) ^ n

def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a_sequence (i + 1)

theorem range_of_t (t : ℝ) (h : ∀ n : ℕ, S n < t) : t ≥ 1 / 4 :=
by
  sorry

end range_of_t_l332_332233


namespace angle_at_3_40_pm_is_130_degrees_l332_332665

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332665


namespace log_a_abs_mono_increasing_l332_332442

theorem log_a_abs_mono_increasing (a : ℝ) (h : 1 < a) :
  ∀ x : ℝ, f(x) = log a (abs x) → f 1 < f (-2) ∧ f (-2) < f 3 :=
by 
  sorry  -- Proof can be filled in later

end log_a_abs_mono_increasing_l332_332442


namespace unique_tournament_sequences_l332_332312

def league_points := (win_points : ℕ := 2) (draw_points : ℕ := 1) (loss_points : ℕ := 0) : Type :=
  { team_points : ℕ → ℕ // ∀ i j, teams_played_points i j = if i = j then 0 else (win_points + draw_points) }

def team_participation (N : ℕ) := { team_number : ℕ // team_number < N }

def point_distribution (N : ℕ) := (sequence : fin N → ℕ) (descending_order : ∀ i j, i ≤ j → sequence i ≥ sequence j)

def is_good_tournament (N : ℕ) (A : point_distribution N) :=
  ∀ T' ≠ T, d(T) ≠ d(T')

def unique_outcome (N : ℕ) (A: point_distribution N) : Prop :=
  ∃! T, d(T) = A

theorem unique_tournament_sequences (N : ℕ) (A : point_distribution N) :
  unique_outcome N A → ∃ k, A = Fibonacci.sequence k :=
sorry

end unique_tournament_sequences_l332_332312


namespace triangle_exists_l332_332581

variables {Point Line : Type} [origin : Point]

-- Given the three rays R1, R2, R3 with a common origin O
variables (O : Point) (R1 R2 R3 : Line)
variables (in_r1 : Point → Prop) (in_r2 : Point → Prop) (in_r3 : Point → Prop)
variables (on_ray : ∀ {p : Point}, in_r1 p → in_r2 p → in_r3 p → p = O)

-- Given the points A1, A2, A3 inside the angles formed by the rays
variables (A1 A2 A3 : Point)
variables (A1_in_R1 : in_r1 A1) (A2_in_R2 : in_r2 A2) (A3_in_R3 : in_r3 A3)

-- Define the Triangle
def construct_triangle : Prop :=
  ∃ (P1 P2 P3 : Point), in_r1 P1 ∧ in_r2 P2 ∧ in_r3 P3 ∧
  (∃ line1, ∀ P, on_ray (line1 P1 P2) → A1 = P) ∧
  (∃ line2, ∀ P, on_ray (line2 P2 P3) → A2 = P) ∧
  (∃ line3, ∀ P, on_ray (line3 P1 P3) → A3 = P)

-- Lean statement to prove the existence of such a triangle
theorem triangle_exists : construct_triangle :=
sorry

end triangle_exists_l332_332581


namespace sum_of_solutions_sum_of_all_solutions_l332_332904

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332904


namespace fair_coin_run_prob_l332_332547

noncomputable def q : ℚ :=
  let q := 2/11 in q

theorem fair_coin_run_prob (m n : ℕ) (hmn_coprime : Nat.coprime m n) (hq : q = m / n) :
  m + n = 13 :=
by
  sorry

end fair_coin_run_prob_l332_332547


namespace sum_of_solutions_eqn_l332_332944

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332944


namespace exists_strictly_increasing_function_l332_332530

def is_strictly_increasing {X : Type*} [preorder X] (f : X → X) : Prop :=
  ∀ x y, x < y → f x < f y

theorem exists_strictly_increasing_function :
  ∃ f : ℕ+ → ℕ+,
    (f 1 = 2) ∧
    is_strictly_increasing f ∧
    (∀ n : ℕ+, f (f n) = f n + n) :=
by
  sorry

end exists_strictly_increasing_function_l332_332530


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332726

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332726


namespace no_int_solutions_for_cubic_eqn_l332_332862

theorem no_int_solutions_for_cubic_eqn :
  ¬ ∃ (m n : ℤ), m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_int_solutions_for_cubic_eqn_l332_332862


namespace circumcenter_AIC_on_circumcircle_ABC_l332_332147

-- Define points A, B, C, I, O, K
variables (A B C I O K : Type*) [point A] [point B] [point C] [point I] [point O] [point K]
-- Assuming I is the incenter of triangle ABC
variable (incenter_ABC : is_incenter I A B C)
-- Assuming O is the circumcenter of triangle ABC
variable (circumcenter_ABC : is_circumcenter O A B C)
-- Assuming K is the intersection of BI with the circumcircle of ABC
variable (K_on_circumcircle_ABC : is_on_circumcircle K A B C)
variable (BI_bisects_ABC : is_angle_bisector I B K)

-- Define the theorem to be proven in Lean
theorem circumcenter_AIC_on_circumcircle_ABC 
  (h1 : incenter_ABC) 
  (h2 : circumcenter_ABC) 
  (h3 : K_on_circumcircle_ABC)
  (h4 : BI_bisects_ABC) :
  is_circumcenter K A I C :=
by sorry

end circumcenter_AIC_on_circumcircle_ABC_l332_332147


namespace cricket_run_rate_l332_332114

theorem cricket_run_rate
  (T : ℕ)  -- Target score
  (initial_runs : ℕ := 32)  -- Runs scored in the first 10 overs
  (required_rate : ℕ := 6)  -- Required rate in the remaining 40 overs
  (remaining_overs : ℕ := 40)  -- Remaining overs
  (total_target : ℕ := 272) : T = 272 :=
by
  -- Given initial_runs and required_rate, total_target should be 272
  have h : T = initial_runs + required_rate * remaining_overs := sorry
  exact h

end cricket_run_rate_l332_332114


namespace count_p_values_l332_332049

theorem count_p_values (p : ℤ) (n : ℝ) :
  (n = 16 * 10^(-p)) →
  (-4 < p ∧ p < 4) →
  ∃ m, p ∈ m ∧ (m.count = 3 ∧ m = [-2, 0, 2]) :=
by 
  sorry

end count_p_values_l332_332049


namespace drama_club_exactly_two_skills_l332_332506

theorem drama_club_exactly_two_skills (total_members cannot_paint cannot_write cannot_direct: ℕ)
    (h1 : total_members = 120)
    (h2 : cannot_paint = 50)
    (h3 : cannot_write = 75)
    (h4 : cannot_direct = 40)
    (h5 : ∀ (paint write direct : Prop) (h : paint ∧ write ∧ direct), false) :
    ∃ (number_with_two_skills : ℕ), number_with_two_skills = 75 :=
by
  -- Definitions based on the given conditions
  let can_paint := total_members - cannot_paint
  let can_write := total_members - cannot_write
  let can_direct := total_members - cannot_direct
  -- The total members ignoring overlaps
  let total_no_overlaps := can_paint + can_write + can_direct

  -- The members with exactly two talents
  let number_with_two_skills := total_no_overlaps - total_members

  use number_with_two_skills
  -- Calculate and verify
  calc number_with_two_skills = total_no_overlaps - total_members : by rfl
                        ...   = 195 - 120                    : by
                                rw [total_no_overlaps, h1, h2, h3, h4]
                                simp
                        ...   = 75                          : by norm_num

end drama_club_exactly_two_skills_l332_332506


namespace smaller_angle_between_hands_at_3_40_l332_332681

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332681


namespace least_number_of_crayons_l332_332782

theorem least_number_of_crayons :
  ∃ n : ℕ, ∀ d ∈ ({4, 6, 8, 9, 10} : finset ℕ), d ∣ n ∧ n = 360 :=
begin
  sorry
end

end least_number_of_crayons_l332_332782


namespace range_of_g_l332_332878

open Real

noncomputable def g (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_g : range g = {y : ℝ | y ≠ 3} :=
by 
  sorry

end range_of_g_l332_332878


namespace distance_from_missouri_l332_332208

-- Open a namespace for our problem context
namespace DrivingDistance

-- Define the conditions
def distance_by_plane := 2000 -- Distance between Arizona and New York by plane in miles
def increase_rate := 0.40 -- Increase in distance by driving

def total_driving_distance : ℝ :=
  distance_by_plane * (1 + increase_rate)

def midway_distance : ℝ :=
  total_driving_distance / 2

-- Theorem to prove the distance from Missouri to New York by car
theorem distance_from_missouri :
  midway_distance = 1400 :=
by
  sorry

end DrivingDistance

end distance_from_missouri_l332_332208


namespace twelve_is_monday_l332_332084

def Weekday := {d : String // d ∈ ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

def not_friday (d: Weekday) : Prop := d.val ≠ "Friday"

def has_exactly_five_fridays (first_friday: nat) (days_in_month: nat) : Prop :=
  first_friday + 28 <= days_in_month ∧
  first_friday + 21 <= days_in_month ∧
  first_friday + 14 <= days_in_month ∧
  first_friday + 7 <= days_in_month ∧
  first_friday > 0 ∧ days_in_month <= 31

noncomputable def compute_day_of_week (start_day: String) (n: nat) : Weekday :=
  sorry

theorem twelve_is_monday (start_day: Weekday) (days_in_month: nat) :
    has_exactly_five_fridays 2 days_in_month
  → not_friday start_day
  → not_friday (compute_day_of_week start_day.val days_in_month)
  → compute_day_of_week start_day.val 12 = ⟨"Monday", by sorry⟩ :=
begin
  sorry
end

end twelve_is_monday_l332_332084


namespace Juan_has_498_marbles_l332_332833

def ConnieMarbles : Nat := 323
def JuanMoreMarbles : Nat := 175
def JuanMarbles : Nat := ConnieMarbles + JuanMoreMarbles

theorem Juan_has_498_marbles : JuanMarbles = 498 := by
  sorry

end Juan_has_498_marbles_l332_332833


namespace distance_missouri_to_new_york_by_car_l332_332210

-- Define the given conditions
def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def midway_factor : ℝ := 0.5

-- Define the problem to be proven
theorem distance_missouri_to_new_york_by_car :
  let total_distance : ℝ := distance_plane + (distance_plane * increase_percentage)
  let missouri_to_new_york_distance : ℝ := total_distance * midway_factor
  missouri_to_new_york_distance = 1400 :=
by
  sorry

end distance_missouri_to_new_york_by_car_l332_332210


namespace axis_of_symmetry_cos_l332_332615

/-- Given the function y = cos(ωx + φ) with ω > 0 and 0 < φ < π,
if A and B are the highest and lowest points of this function with 
a distance of 2 between them, then one of the axes of symmetry
is x = 1. -/
theorem axis_of_symmetry_cos (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π)
  (A B : ℝ) (hA : A = cos((ω * 0) + φ)) (hB : B = cos((ω * 1) + φ)) (h : B - A = 2) :
  ∃ x : ℝ, x = 1 :=
by
  sorry

end axis_of_symmetry_cos_l332_332615


namespace cube_single_cut_no_eight_face_l332_332288

theorem cube_single_cut_no_eight_face :
  let initial_faces := 6 in
  let new_face := 1 in
  let possible_faces := [4, 5, 6, 7, 8] in
  ∀ f ∈ possible_faces,
    if f = 8 then ∃ (p : ℕ), p = initial_faces + new_face - 1 → False
    else True :=
by
  -- proof here
  sorry

end cube_single_cut_no_eight_face_l332_332288


namespace perfect_cube_probability_l332_332802

theorem perfect_cube_probability
  (faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8})
  (rolls : ℕ := 5)
  (rel_prime : forall p q : ℕ, rel_prime p q → gcd p q = 1) :
  (∃ (p q : ℕ), p + q = 16487 ∧ (rel_prime p q) ∧ (prob := p / q) ∧
      probability (product_is_cube rolls [1, 2, 3, 4, 5, 6, 7, 8]) = prob) :=
   sorry

end perfect_cube_probability_l332_332802


namespace distance_from_missouri_l332_332209

-- Open a namespace for our problem context
namespace DrivingDistance

-- Define the conditions
def distance_by_plane := 2000 -- Distance between Arizona and New York by plane in miles
def increase_rate := 0.40 -- Increase in distance by driving

def total_driving_distance : ℝ :=
  distance_by_plane * (1 + increase_rate)

def midway_distance : ℝ :=
  total_driving_distance / 2

-- Theorem to prove the distance from Missouri to New York by car
theorem distance_from_missouri :
  midway_distance = 1400 :=
by
  sorry

end DrivingDistance

end distance_from_missouri_l332_332209


namespace f_eq_n_l332_332150

noncomputable def f : ℕ → ℕ := sorry

theorem f_eq_n (h : ∀ n : ℕ, f(n+1) > f(f(n))) : ∀ n : ℕ, f(n) = n :=
sorry

end f_eq_n_l332_332150


namespace clock_angle_3_40_l332_332678

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332678


namespace range_of_k_l332_332451

   noncomputable def f (x : ℝ) : ℝ := x + sin x

   theorem range_of_k (k : ℝ) : 
     (∃ x ∈ Icc (-2 : ℝ) (1 : ℝ), f (x^2 + x) + f (x - k) = 0) → k ∈ Icc (-1 : ℝ) (3 : ℝ) :=
   by
     sorry
   
end range_of_k_l332_332451


namespace number_of_three_letter_initials_l332_332478

theorem number_of_three_letter_initials : 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] in
  (length letters) ^ 3 = 1000 := 
by
  -- Definitions:
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  -- The problem reduces to computing 10^3 and showing it equals 1000.
  have length_letters : length letters = 10 := by sorry
  calc 
    (length letters) ^ 3
      = 10 ^ 3 : by rw length_letters
  ... = 1000 : by norm_num

end number_of_three_letter_initials_l332_332478


namespace there_exists_student_with_odd_number_of_friends_l332_332641

variables (students : Finset ℕ)
variables (friends relation : ℕ → ℕ → Prop)

-- Condition 1: There are 36 students in a mathematics class 
axiom (h1 : students.card = 36)

-- Condition 2: Exactly one of them recently won a mathematics competition
axiom (winner : ℕ)
axiom (h2 : winner ∈ students )

-- Condition 3: Each of his classmates has exactly five mutual friends with him
axiom (h3 : ∀ (s ∈ students), s ≠ winner → (students.filter (λ t, relation s t)).card = 5)

theorem there_exists_student_with_odd_number_of_friends :
  ∃ (s ∈ students), (students.filter (λ t, relation s t)).card % 2 = 1 :=
sorry

end there_exists_student_with_odd_number_of_friends_l332_332641


namespace Calvin_insect_collection_l332_332821

def Calvin_has_insects (num_roaches num_scorpions num_crickets num_caterpillars total_insects : ℕ) : Prop :=
  total_insects = num_roaches + num_scorpions + num_crickets + num_caterpillars

theorem Calvin_insect_collection
  (roach_count : ℕ)
  (scorpion_count : ℕ)
  (cricket_count : ℕ)
  (caterpillar_count : ℕ)
  (total_count : ℕ)
  (h1 : roach_count = 12)
  (h2 : scorpion_count = 3)
  (h3 : cricket_count = roach_count / 2)
  (h4 : caterpillar_count = scorpion_count * 2)
  (h5 : total_count = roach_count + scorpion_count + cricket_count + caterpillar_count) :
  Calvin_has_insects roach_count scorpion_count cricket_count caterpillar_count total_count :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact h5

end Calvin_insect_collection_l332_332821


namespace cubes_sum_equiv_l332_332044

theorem cubes_sum_equiv (h : 2^3 + 4^3 + 6^3 + 8^3 + 10^3 + 12^3 + 14^3 + 16^3 + 18^3 = 16200) :
  3^3 + 6^3 + 9^3 + 12^3 + 15^3 + 18^3 + 21^3 + 24^3 + 27^3 = 54675 := 
  sorry

end cubes_sum_equiv_l332_332044


namespace double_integral_value_l332_332814

def region_Ω : set (ℝ × ℝ) := 
  {p | let (x, y) := p in 
       ((y ≥ -4) ∧ (y ≤ -1) ∧ (x ≥ -y-2) ∧ (x ≤ (10 - y) / 7)) ∨ 
       ((y > -1) ∧ (y ≤ 3) ∧ (x ≥ (y / 2) - 0.5) ∧ (x ≤ (10 - y) / 7))}

theorem double_integral_value :
  ∫∫ (xy: ℝ × ℝ) in region_Ω, (2 * xy.1 + 3 * xy.2 + 1) ∂xy = -83 / 294 := by
  sorry

end double_integral_value_l332_332814


namespace fred_found_43_seashells_l332_332267

-- Define the conditions
def tom_seashells : ℕ := 15
def additional_seashells : ℕ := 28

-- Define Fred's total seashells based on the conditions
def fred_seashells : ℕ := tom_seashells + additional_seashells

-- The theorem to prove that Fred found 43 seashells
theorem fred_found_43_seashells : fred_seashells = 43 :=
by
  -- Proof goes here
  sorry

end fred_found_43_seashells_l332_332267


namespace prime_power_sum_l332_332155

theorem prime_power_sum (a b p : ℕ) (hp : p = a ^ b + b ^ a) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (hp_prime : Nat.Prime p) : 
  p = 17 := 
sorry

end prime_power_sum_l332_332155


namespace determine_y_l332_332111

-- Definitions derived from the conditions
def EF_and_GH_are_straight_lines (EF GH : Set Point) : Prop := 
  straight_line EF ∧ straight_line GH

def angle_EPF (P : Point) : ℝ := 70
def angle_FPR (P : Point) : ℝ := 40
def angle_GQR (P : Point) : ℝ := 110

def angle_sum_of_triangle (triangle : Triangle) : Prop := 
  triangle.angles.sum = 180

-- Statement proving the value of y
theorem determine_y (P Q R : Point) (EF GH : Set Point) (EPF FPR : ℝ) : 
  EF_and_GH_are_straight_lines EF GH →
  angle_EPF P = 70 →
  angle_FPR P = 40 →
  angle_GQR P = 110 →
  angle_sum_of_triangle ⟨Q, P, R⟩ →
  let y := 180 - angle_EPF P - angle_FPR P in
  y = 40 :=
sorry

end determine_y_l332_332111


namespace write_as_sum_1800_l332_332486

/-- The number of ways to write 1800 as the sum of 1s, 2s, and 3s, ignoring order, is 4^300. -/
theorem write_as_sum_1800 : 
  (∑ (n : ℕ) in finset.range 1801, if ∃ (s₁ s₂ s₃ : ℕ), s₁ + s₂ + s₃ = n ∧ s₁ + 2 * s₂ + 3 * s₃ = 1800 then 1 else 0) = 4^300 :=
sorry

end write_as_sum_1800_l332_332486


namespace find_p_l332_332458

theorem find_p (p : ℝ) (h₁ : p > 0)
  (h₂ : ∃ x y : ℝ, x^2 / 3 - 16 * y^2 / p^2 = 1 ∧ x = - (sqrt (3 + p^2 / 16)) ∧ y = 0 
        ∧ (y^2 = 2 * p * x → x = p / 2)) : 
  p = 4 := sorry

end find_p_l332_332458


namespace coordinates_of_point_P_l332_332432

theorem coordinates_of_point_P
  (O : Point)
  (hO : O = ⟨0, 0⟩)
  (P : Point)
  (hP : ∃ α : Real, P = ⟨4 * Real.cos α, 2 * Real.sqrt 3 * Real.sin α⟩ ∧
              0 ≤ 4 * Real.cos α ∧ 0 ≤ 2 * Real.sqrt 3 * Real.sin α)
  (slope : Real)
  (hslope : slope = Real.tan (Real.pi / 3)) :
  P = ⟨4 * Real.sqrt 5 / 5, 4 * Real.sqrt 15 / 5⟩ :=
by
  sorry

end coordinates_of_point_P_l332_332432


namespace paint_mixer_days_l332_332567

/-- Making an equal number of drums of paint each day, a paint mixer takes three days to make 18 drums of paint.
    We want to determine how many days it will take for him to make 360 drums of paint. -/
theorem paint_mixer_days (n : ℕ) (h1 : n > 0) 
  (h2 : 3 * n = 18) : 
  360 / n = 60 := by
  sorry

end paint_mixer_days_l332_332567


namespace least_multiple_15_product_15_l332_332277

def digits_product (n : ℕ) : ℕ :=
  -- Function to compute the product of the digits of a number
  (n.digits 10).product

def is_multiple_of_15 (n : ℕ) : Prop :=
  15 ∣ n

def ends_with_5 (n : ℕ) : Prop :=
  n % 10 = 5

theorem least_multiple_15_product_15 :
  ∃ n : ℕ, is_multiple_of_15 n ∧ ends_with_5 n ∧ digits_product n = 15 ∧ (∀ m : ℕ, is_multiple_of_15 m ∧ ends_with_5 m ∧ digits_product m = 15 → n ≤ m) :=
sorry

end least_multiple_15_product_15_l332_332277


namespace triangle_ratio_area_l332_332585

-- Define the conditions and structures
structure Triangle := 
  (A B C : Point)
  (side_eq : dist A B = dist A C)
  (D_on_AC : lies_on D AC)
  (angle_DBC : angle D B C = 30)

-- Define the result to be proved
theorem triangle_ratio_area (T : Triangle) : 
  ratio (area (triangle A D B)) (area (triangle C D B)) = (2 * sqrt 3 - 3) / 3 := 
sorry

end triangle_ratio_area_l332_332585


namespace sum_of_solutions_eq_16_l332_332926

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332926


namespace area_of_quadrilateral_AGHB_l332_332334

theorem area_of_quadrilateral_AGHB
  (AB CD : ℝ) (AD BC : ℝ)
  (AB_len : AB = 40) (AD_len : AD = 24)
  (G : ℝ) (H : ℝ)
  (G_mid_AD : G = AD / 2) (H_mid_AB : H = AB / 2) :
  let A_AGHB := (AB * AD) - (1 / 2 * G * AB) - (1 / 2 * H * AD) in
  A_AGHB = 480 := 
by {
  have A_ABCD := AB * AD,
  have A_AGD := 1 / 2 * G * AB,
  have A_BHC := 1 / 2 * H * AD,
  have A_AGHB := A_ABCD - (A_AGD + A_BHC),
  rw [AB_len, AD_len, G_mid_AD, H_mid_AB],
  simp,
  exact sorry
}

end area_of_quadrilateral_AGHB_l332_332334


namespace find_s_l332_332806

/-
The problem is to prove that for a triangle with one vertex at the vertex of the parabola 
y = x^2 + 3x - 4 and the other two vertices at the intersections of the line y = s with the 
parabola, if the area of the triangle is exactly 24, then the possible values of s are identified.
-/

/-- Define the parabola equation y = x^2 + 3x - 4 -/
def parabola (x : ℝ) : ℝ := x^2 + 3*x - 4

/-- Define the line equation y = s -/
def line (s : ℝ) (x : ℝ) : ℝ := s

/-- Define the vertex of the parabola y = x^2 + 3x - 4 -/
def vertex_of_parabola : ℝ × ℝ := (-3 / 2, parabola (-3 / 2))

/-- Define the intersections of the line y = s with the parabola y = x^2 + 3x - 4 -/
def intersections (s : ℝ) : ℝ × ℝ :=
  let disc := real.sqrt (4 * s + 25) in
  ( (-3 - disc) / 2, s ),
  ( (-3 + disc) / 2, s )

/-- Define the area of the triangle given the vertices -/
def area_of_triangle (s : ℝ) : ℝ :=
  let base := real.sqrt (4 * s + 25) in
  let height := abs (s + 25 / 4) in
  (1 / 2) * base * height

/-- The theorem to prove that if the area of the triangle is exactly 24, then the possible value of s is 7 -/
theorem find_s (s : ℝ) (h : area_of_triangle s = 24) : s = 7 :=
  sorry

end find_s_l332_332806


namespace smaller_angle_3_40_l332_332698

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332698


namespace frog_probability_l332_332325

-- Define the probability function P
noncomputable def P : (ℤ × ℤ) → ℚ
| (0, _) => 1
| (6, _) => 1
| (_, 0) => 0
| (_, 6) => 0
| (3, 2) => (P (2, 2) + P (4, 2) + P (3, 1) + P (3, 3)) / 4
| (2, 2) => (1 + 1 + P (3, 2) + P (2, 3)) / 4
| (4, 2) => (1 + 1 + P (3, 2) + P (4, 3)) / 4
| (3, 1) => (P (2, 1) + P (4, 1) + 0 + P (3, 2)) / 4
| _ => 0

-- State the theorem
theorem frog_probability :
  P (3, 2) = 11 / 18 :=
sorry

end frog_probability_l332_332325


namespace tan_alpha_value_complicated_expression_value_l332_332405

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) : 
  Real.tan α = -2 := by 
  sorry

theorem complicated_expression_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) (h3 : Real.tan α = -2) :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) / 
  (Real.cos (α - Real.pi / 2) - Real.sin (2 * Real.pi / 2 + α)) = -5 := by 
  sorry

end tan_alpha_value_complicated_expression_value_l332_332405


namespace probability_first_or_second_l332_332809

/-- Define the events and their probabilities --/
def prob_hit_first_sector : ℝ := 0.4
def prob_hit_second_sector : ℝ := 0.3
def prob_hit_first_or_second : ℝ := 0.7

/-- The proof that these probabilities add up as mutually exclusive events --/
theorem probability_first_or_second (P_A : ℝ) (P_B : ℝ) (P_A_or_B : ℝ) (hP_A : P_A = prob_hit_first_sector) (hP_B : P_B = prob_hit_second_sector) (hP_A_or_B : P_A_or_B = prob_hit_first_or_second) :
  P_A_or_B = P_A + P_B := 
  by
    rw [hP_A, hP_B, hP_A_or_B]
    sorry

end probability_first_or_second_l332_332809


namespace dividend_calculation_l332_332793

noncomputable def investment : ℝ := 14400
noncomputable def share_face_value : ℝ := 100
noncomputable def premium_rate : ℝ := 0.20
noncomputable def dividend_rate : ℝ := 0.07

def cost_per_share (face_value : ℝ) (premium : ℝ) : ℝ :=
  face_value * (1 + premium)

def number_of_shares (investment : ℝ) (cost_per_share : ℝ) : ℝ :=
  investment / cost_per_share

def dividend_per_share (face_value : ℝ) (rate : ℝ) : ℝ :=
  face_value * rate

def total_dividend (shares : ℝ) (dividend_per_share : ℝ) : ℝ :=
  shares * dividend_per_share

theorem dividend_calculation :
  let cost_share := cost_per_share share_face_value premium_rate in
  let total_shares := number_of_shares investment cost_share in
  let per_share_dividend := dividend_per_share share_face_value dividend_rate in
  total_dividend total_shares per_share_dividend = 840 := by
  sorry

end dividend_calculation_l332_332793


namespace integral_value_l332_332414

theorem integral_value (a : ℝ) (h : z = a + (a - 1) * Complex.I) (ha : a = 1) :
  ∫ x in 0..a, sqrt (1 - x^2) + x = (Real.pi / 4) + 1 / 2 :=
by 
  sorry

end integral_value_l332_332414


namespace product_of_nonzero_elements_eq_neg_one_wilsons_theorem_l332_332309

theorem product_of_nonzero_elements_eq_neg_one (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (p-1)! % p = p - 1 % p :=
sorry

theorem wilsons_theorem (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (p - 1)! + 1 % p^2 = 0 :=
sorry

end product_of_nonzero_elements_eq_neg_one_wilsons_theorem_l332_332309


namespace range_of_m_l332_332992

theorem range_of_m (f : ℝ → ℝ) (h1 : ∀ x ∈ set.Icc (-2 : ℝ) 2, f (-x) = f x)
  (h2 : ∀ a b ∈ set.Icc (0 : ℝ) 2, a ≠ b → (f a - f b) / (a - b) < 0) 
  (h3 : ∀ m ∈ set.Icc (-2 : ℝ) 2, f (1 - m) < f m) : 
  ∀ m, m ≥ -1 ∧ m < 0.5 :=
by 
  sorry

end range_of_m_l332_332992


namespace total_cost_is_15_l332_332840

def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

def dale_breakfast_cost := dale_toast * toast_cost + dale_eggs * egg_cost
def andrew_breakfast_cost := andrew_toast * toast_cost + andrew_eggs * egg_cost

def total_breakfast_cost := dale_breakfast_cost + andrew_breakfast_cost

theorem total_cost_is_15 : total_breakfast_cost = 15 := by
  sorry

end total_cost_is_15_l332_332840


namespace prob_diff_suits_at_least_one_heart_l332_332574

theorem prob_diff_suits_at_least_one_heart :
  let total_cards := 104
  let hearts := 52
  let non_hearts := total_cards - hearts
  let total_combinations := nat.choose total_cards 2
  let no_hearts_combinations := nat.choose non_hearts 2
  let at_least_one_heart_combinations := total_combinations - no_hearts_combinations
  let probability_diff_suits := 78 / 103
  (probability_diff_suits * (at_least_one_heart_combinations / total_combinations)).approx = 0.331 :=
by
  sorry

end prob_diff_suits_at_least_one_heart_l332_332574


namespace tangent_line_properties_l332_332230

-- Define the given parabola
def parabola (x : ℝ) : ℝ := x^2 + x + 2

-- Define a point on the parabola
def point_on_parabola : ℝ × ℝ := (1, 4)

-- Define the derivative of the parabola
def parabola_derivative (x : ℝ) : ℝ := 2 * x + 1

-- Define the slope at a given point
def slope_at_point (p : ℝ × ℝ) : ℝ := parabola_derivative p.1

-- Define the equation of the tangent line using the point-slope form
def tangent_line_equation (x y slope x1 y1 : ℝ) : Prop := y - y1 = slope * (x - x1)

-- Define that the slope is 3 at point (1,4)
def slope_is_3_at_1_4 : Prop := slope_at_point point_on_parabola = 3

-- Define that the equation of the tangent line is 3x - y + 1 = 0
def tangent_line_at_1_4 : Prop := ∀ x y, tangent_line_equation x y 3 1 4 ↔ 3 * x - y + 1 = 0

theorem tangent_line_properties :
  slope_is_3_at_1_4 ∧ tangent_line_at_1_4 :=
by
  split
  -- Prove the slope at point (1,4) is 3
  sorry
  -- Prove the equation of the tangent line is 3x - y + 1 = 0
  sorry

end tangent_line_properties_l332_332230


namespace value_of_expression_l332_332410

theorem value_of_expression (x y : ℝ) (h₀ : x = Real.sqrt 2 + 1) (h₁ : y = Real.sqrt 2 - 1) : 
  (x + y) * (x - y) = 4 * Real.sqrt 2 :=
by
  sorry

end value_of_expression_l332_332410


namespace sum_of_solutions_l332_332960
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332960


namespace intersection_distance_eq_two_l332_332119

theorem intersection_distance_eq_two :
  let C1 := λ ρ θ, ρ * Math.sin (θ + Real.pi / 4) = 1
  let C2 := λ ρ, ρ = Real.sqrt 2
  ∀ (A B : ℝ × ℝ), 
    (∃ (ρ θ : ℝ), C1 ρ θ ∧ C2 ρ ∧ (A = (ρ * Math.cos θ, ρ * Math.sin θ))) ∧
    (∃ (ρ' θ' : ℝ), C1 ρ' θ' ∧ C2 ρ' ∧ (B = (ρ' * Math.cos θ', ρ' * Math.sin θ'))) →
    Real.abs (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) = 2 :=
by
  intros
  sorry

end intersection_distance_eq_two_l332_332119


namespace smaller_angle_between_clock_hands_3_40_pm_l332_332728

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l332_332728


namespace find_a_and_b_l332_332413

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

theorem find_a_and_b :
  (∃ a b : ℝ, a ≠ 0 ∧
   (∀ x, -1 ≤ x ∧ x ≤ 2 → f a b x ≤ 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = -29)
  ) → ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
sorry

end find_a_and_b_l332_332413


namespace hyperbola_distance_from_focus_to_asymptote_l332_332382

theorem hyperbola_distance_from_focus_to_asymptote :
  (distance_from_focus_to_asymptote (4 : ℝ) (12 : ℝ) = 2 * real.sqrt 3) :=
by sorry

end hyperbola_distance_from_focus_to_asymptote_l332_332382


namespace sum_of_solutions_l332_332957
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332957


namespace rain_in_first_hour_l332_332125

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end rain_in_first_hour_l332_332125


namespace correct_decimal_multiplication_l332_332342

theorem correct_decimal_multiplication : 0.085 * 3.45 = 0.29325 := 
by 
  sorry

end correct_decimal_multiplication_l332_332342


namespace exists_real_root_for_interval_l332_332871

theorem exists_real_root_for_interval (a : ℝ) :
  (0.5925003438768 ≤ a ∧ a ≤ 17/3) →
  ∃ x : ℝ, 3^(cos (2 * x) + 1) - (a - 5) * 3^(cos (2 * x)^2) = 7 :=
by
  sorry

end exists_real_root_for_interval_l332_332871


namespace prove_a_ge_1_div_4_l332_332457

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - Real.log (x + 1)
noncomputable def g (x a : ℝ) : ℝ := x^2 - 2 * a * x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := x^2 - 1 / (x + 1)

-- Define the theorem
theorem prove_a_ge_1_div_4 (a : ℝ) :
  (∃ x₁ ∈ Icc (0 : ℝ) 1, ∃ x₂ ∈ Icc (1 : ℝ) 2, f' x₁ ≥ g x₂ a) → 
  a ≥ 1 / 4 :=
begin
  sorry
end

end prove_a_ge_1_div_4_l332_332457


namespace intersection_nonempty_implies_a_lt_one_l332_332464

variable {α : Type*} [LinearOrder α]

def M (x : α) : Prop := x ≤ (1 : α)
def P (x : α) (a : α) : Prop := x > a

theorem intersection_nonempty_implies_a_lt_one {a : α} (h : ∃ x, M x ∧ P x a) : a < 1 := sorry

end intersection_nonempty_implies_a_lt_one_l332_332464


namespace distance_missouri_to_new_york_by_car_l332_332212

-- Define the given conditions
def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def midway_factor : ℝ := 0.5

-- Define the problem to be proven
theorem distance_missouri_to_new_york_by_car :
  let total_distance : ℝ := distance_plane + (distance_plane * increase_percentage)
  let missouri_to_new_york_distance : ℝ := total_distance * midway_factor
  missouri_to_new_york_distance = 1400 :=
by
  sorry

end distance_missouri_to_new_york_by_car_l332_332212


namespace rain_in_first_hour_l332_332126

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end rain_in_first_hour_l332_332126


namespace findSolutions_l332_332872

-- Define the given mathematical problem
def originalEquation (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3)) / ((x - 4) * (x - 6) * (x - 4)) = 1

-- Define the conditions where the equation is valid
def validCondition (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6

-- Define the set of solutions
def solutions (x : ℝ) : Prop :=
  x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2

-- The theorem stating the correct set of solutions
theorem findSolutions (x : ℝ) : originalEquation x ∧ validCondition x ↔ solutions x :=
by sorry

end findSolutions_l332_332872


namespace sum_of_solutions_eq_16_l332_332980

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332980


namespace proof_not_necessarily_15_points_l332_332327

-- Define the number of teams
def teams := 14

-- Define a tournament where each team plays every other exactly once
def games := (teams * (teams - 1)) / 2

-- Define a function calculating the total points by summing points for each game
def total_points (wins draws : ℕ) := (3 * wins) + (1 * draws)

-- Define a statement that total points is at least 150
def scores_sum_at_least_150 (wins draws : ℕ) : Prop :=
  total_points wins draws ≥ 150

-- Define a condition that a score could be less than 15
def highest_score_not_necessarily_15 : Prop :=
  ∃ (scores : Finset ℕ), scores.card = teams ∧ ∀ score ∈ scores, score < 15

theorem proof_not_necessarily_15_points :
  ∃ (wins draws : ℕ), wins + draws = games ∧ scores_sum_at_least_150 wins draws ∧ highest_score_not_necessarily_15 :=
by
  sorry

end proof_not_necessarily_15_points_l332_332327


namespace function_B_is_odd_and_decreasing_l332_332344

-- Definitions of the functions
def f_A (x : ℝ) : ℝ := x * log 2
def f_B (x : ℝ) : ℝ := -x * abs x
def f_C (x : ℝ) : ℝ := sin x
def f_D (x : ℝ) : ℝ := (log x) / x

-- Statement of the proof problem
theorem function_B_is_odd_and_decreasing :
  (∀ x : ℝ, f_B (-x) = -f_B x) ∧ (∀ x y : ℝ, x < y → f_B x > f_B y) :=
by
  sorry

end function_B_is_odd_and_decreasing_l332_332344


namespace sea_horses_count_l332_332775

theorem sea_horses_count (S P : ℕ) (h1 : 11 * S = 5 * P) (h2 : P = S + 85) : S = 70 :=
by
  sorry

end sea_horses_count_l332_332775


namespace blood_expiration_l332_332300

noncomputable def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def seconds_in_a_day := 60 * 60 * 24
def days_to_expire := factorial 8 / seconds_in_a_day

theorem blood_expiration :
  ∀ (donation1 donation2 : ℕ) (expiration_time : ℕ),
    donation1 = 5 →
    donation2 = 7 →
    expiration_time = factorial 8 →
    (donation1 + days_to_expire = 5) ∧ (donation2 + days_to_expire = 7) :=
begin
  intros donation1 donation2 expiration_time h1 h2 h3,
  rw [h1, h2, h3],
  split;
  -- lean proof steps can be filled here
  sorry
end

end blood_expiration_l332_332300


namespace log_abs_monotone_decreasing_l332_332502

open Real

theorem log_abs_monotone_decreasing {a : ℝ} (h : ∀ x y, 0 < x ∧ x < y ∧ y ≤ a → |log x| ≥ |log y|) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end log_abs_monotone_decreasing_l332_332502


namespace basketball_scores_l332_332318

theorem basketball_scores :
  ∃ (S : set ℕ), (∀ s ∈ S, ∃ x : ℕ, 0 ≤ x ∧ x ≤ 7 ∧ s = 2 * (7 - x) + 3 * x) ∧ S.card = 8 :=
by
  sorry

end basketball_scores_l332_332318


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332884

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332884


namespace magnitude_difference_l332_332005

-- Define the vectors and given conditions
def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (1 * real.cos(π / 3), 1 * real.sin(π / 3))

-- Calculate the magnitude of a vector
noncomputable def vector_magnitude (v : ℝ × ℝ) := real.sqrt (v.fst^2 + v.snd^2)

-- Define the given angle and magnitudes
def angle_ab := real.pi / 3
def mag_b := 1

-- Prove the statement
theorem magnitude_difference : vector_magnitude (a.1 - 2 * b.1, a.2 - 2 * b.2) = 2 :=
by
  sorry

end magnitude_difference_l332_332005


namespace sum_of_first_40_terms_l332_332632

def a : ℕ → ℤ := sorry

def S (n : ℕ) : ℤ := (Finset.range n).sum a

theorem sum_of_first_40_terms :
  (∀ n : ℕ, a (n + 1) + (-1) ^ n * a n = n) →
  S 40 = 420 := 
sorry

end sum_of_first_40_terms_l332_332632


namespace sum_geometric_sequence_terms_l332_332237

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l332_332237


namespace ana_wins_probability_l332_332533

noncomputable def probability_ana_wins : ℚ := 
  let a := (1 / 2)^5
  let r := (1 / 2)^4
  a / (1 - r)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 30 :=
by
  sorry

end ana_wins_probability_l332_332533


namespace clock_angle_3_40_l332_332674

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332674


namespace min_expression_value_l332_332385

theorem min_expression_value : ∀ x y : ℝ, 
  (sqrt (2 * (1 + cos (2 * x))) - sqrt (9 - sqrt 7) * sin x + 1) * 
  (3 + 2 * sqrt (13 - sqrt 7) * cos y - cos (2 * y)) ≥ -19 := 
by
  sorry

end min_expression_value_l332_332385


namespace clock_angle_3_40_l332_332720

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332720


namespace sum_of_solutions_eq_16_l332_332979

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332979


namespace simplify_and_evaluate_expression_l332_332600

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) : 
  ( ( x + 3 ) / ( x^2 - 2*x + 1 ) * ( x - 1 ) / ( x^2 + 3*x ) + 1 / x ) = Real.sqrt 3 / 3 :=
by
  rw h
  sorry

end simplify_and_evaluate_expression_l332_332600


namespace exists_partition_l332_332536

def isPartitioned {α : Type} (S : set (fin n → bool)) (partition : set (set (fin n → bool))) :=
  (∀ A ∈ partition, A ⊆ S ∧ Fintype.card A = 3) ∧
  (⋃₀ partition = S) ∧
  (∀ A B ∈ partition, A ≠ B → A ∩ B = ∅)

def condition (a b c : fin n → bool) : Prop :=
  ∀ i : fin n, (a i + b i + c i : nat) % 2 = 0

noncomputable def S (n : ℕ) : set (fin n → bool) :=
  {f | ∀ i, f i = 0 ∨ f i = 1}

theorem exists_partition (n : ℕ) (h_even : n % 2 = 0) :
  ∃ partition : set (set (fin n → bool)),
    isPartitioned (S n) partition ∧
    (∀ A ∈ partition, ∃ a b c ∈ A, condition a b c) := sorry

end exists_partition_l332_332536


namespace distance_P1P2_l332_332516

def P1 : ℝ × ℝ × ℝ := (-1, 3, 5)
def P2 : ℝ × ℝ × ℝ := (2, 4, -3)

theorem distance_P1P2 : 
  let d := real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2 + (P2.3 - P1.3)^2) in
  d = real.sqrt 74 :=
by
  sorry

end distance_P1P2_l332_332516


namespace sum_of_solutions_eq_16_l332_332973

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332973


namespace smaller_angle_between_hands_at_3_40_l332_332689

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332689


namespace sum_of_solutions_l332_332893

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332893


namespace color_table_exists_diff_colors_l332_332612

theorem color_table_exists_diff_colors :
  (∃ (rows1 rows2 cols1 cols2 : fin 100),
    rows1 ≠ rows2 ∧ cols1 ≠ cols2 ∧
    let color (r : fin 100) (c : fin 100) : fin 4 in
    color rows1 cols1 ≠ color rows1 cols2 ∧
    color rows1 cols1 ≠ color rows2 cols1 ∧
    color rows2 cols1 ≠ color rows2 cols2 ∧
    color rows1 cols2 ≠ color rows2 cols2) :=
sorry

end color_table_exists_diff_colors_l332_332612


namespace solve_problem_l332_332029

noncomputable def solution_set : Set ℤ := {x | abs (7 * x - 5) ≤ 9}

theorem solve_problem : solution_set = {0, 1, 2} := by
  sorry

end solve_problem_l332_332029


namespace clock_angle_3_40_l332_332714

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332714


namespace sum_of_solutions_l332_332898

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332898


namespace max_value_unbounded_l332_332384

noncomputable def distAP (x : ℝ) : ℝ := real.sqrt (x^2 + (2 - x)^2)
noncomputable def distBP (x : ℝ) : ℝ :=  real.sqrt ((2 - x)^2 + (2 + x)^2)

theorem max_value_unbounded :
  ∀ x : ℝ, (distAP x + distBP x) → ∞ :=
sorry

end max_value_unbounded_l332_332384


namespace tan_of_alpha_eq_sqrt_2_l332_332439

noncomputable def α : ℝ := sorry

theorem tan_of_alpha_eq_sqrt_2 (h1 : α ∈ (0 : ℝ, Real.pi / 2))
    (h2 : Real.sin α = (Real.sqrt 6) / 3) : 
    Real.tan α = Real.sqrt 2 := 
sorry

end tan_of_alpha_eq_sqrt_2_l332_332439


namespace general_formula_a_seq_sum_b_seq_l332_332164

-- Definitions of sequences
def a_seq (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ ∀ n : ℕ, 2 * a (n + 1) + (finset.sum (finset.range n) (λ i, a (i + 1))) - 2 = 0

def b_seq (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b n = n * (a n)^2

-- Proving statements
theorem general_formula_a_seq (a : ℕ → ℝ) (h : a_seq a) :
  ∀ n : ℕ, a n = (1 / 2) ^ (n - 1) :=
sorry

theorem sum_b_seq (a b : ℕ → ℝ) (h₁ : a_seq a) (h₂ : b_seq a b) :
  ∀ n : ℕ, (finset.sum (finset.range n) b) = (16 / 9) - (3 * n + 4) / (9 * 4 ^ (n - 1)) :=
sorry

end general_formula_a_seq_sum_b_seq_l332_332164


namespace peaches_left_in_baskets_l332_332642

theorem peaches_left_in_baskets :
  let initial_baskets := 5
  let initial_peaches_per_basket := 20
  let new_baskets := 4
  let new_peaches_per_basket := 25
  let peaches_removed_per_basket := 10

  let total_initial_peaches := initial_baskets * initial_peaches_per_basket
  let total_new_peaches := new_baskets * new_peaches_per_basket
  let total_peaches_before_removal := total_initial_peaches + total_new_peaches

  let total_baskets := initial_baskets + new_baskets
  let total_peaches_removed := total_baskets * peaches_removed_per_basket
  let peaches_left := total_peaches_before_removal - total_peaches_removed

  peaches_left = 110 := by
  sorry

end peaches_left_in_baskets_l332_332642


namespace clock_angle_l332_332701

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332701


namespace smaller_angle_3_40_l332_332697

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332697


namespace least_multiple_15_product_15_l332_332276

def digits_product (n : ℕ) : ℕ :=
  -- Function to compute the product of the digits of a number
  (n.digits 10).product

def is_multiple_of_15 (n : ℕ) : Prop :=
  15 ∣ n

def ends_with_5 (n : ℕ) : Prop :=
  n % 10 = 5

theorem least_multiple_15_product_15 :
  ∃ n : ℕ, is_multiple_of_15 n ∧ ends_with_5 n ∧ digits_product n = 15 ∧ (∀ m : ℕ, is_multiple_of_15 m ∧ ends_with_5 m ∧ digits_product m = 15 → n ≤ m) :=
sorry

end least_multiple_15_product_15_l332_332276


namespace hyperbola_standard_equation_ellipse_standard_equation_l332_332314

-- Problem 1: Hyperbola
theorem hyperbola_standard_equation :
  let e := 5 / 4
  let foci_ellipse := (x^2 / 40) + (y^2 / 15) = 1 in
  let hyperbola_eq := (x^2 / 16) - (y^2 / 9) = 1 in
  ∃ m, hyperbola_eq ∧ m > 0 ∧ (25 - m) > 0 ∧ e = sqrt(1 + (25 - m) / m) := by {
  sorry
}

-- Problem 2: Ellipse
theorem ellipse_standard_equation :
  let P := (3, 2)
  let major_axis_length := 3 * minor_axis_length
  let foci := (a > b > 0) ∧ (2 * a = 3 * 2 * b) ∧ (3^2 / a^2) + (2^2 / b^2) = 1 in
  let ellipse_eq := (x^2 / 45) + (y^2 / 5) = 1 in
  ellipse_eq := by {
  sorry
}

end hyperbola_standard_equation_ellipse_standard_equation_l332_332314


namespace f_f_neg1_l332_332313

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_f_neg1 : f (f (-1)) = 5 :=
  by
    sorry

end f_f_neg1_l332_332313


namespace compare_expressions_l332_332293

theorem compare_expressions (n : ℕ) (hn : 0 < n):
  (n ≤ 48 ∧ 99^n + 100^n > 101^n) ∨ (n > 48 ∧ 99^n + 100^n < 101^n) :=
sorry  -- Proof is omitted.

end compare_expressions_l332_332293


namespace area_of_region_l332_332275

-- Define the conditions given in the problem.
def equation (x y : ℝ) := x^2 + y^2 = |x| + |y|

-- Define the theorem to prove the area enclosed by the graph of the equation.
theorem area_of_region : 
  let region := {p : ℝ × ℝ | equation p.1 p.2} in
  ∫ (p : ℝ × ℝ) in region, 1 = π + 2 :=
sorry

end area_of_region_l332_332275


namespace victor_lives_at_end_l332_332660

theorem victor_lives_at_end 
  (initial_lives : ℕ := 320)
  (level1_minutes : ℕ := 50)
  (level1_lost_rate : ℕ := 7)
  (level1_lost_interval : ℕ := 10)
  (level1_gain_rate : ℕ := 2)
  (level1_gain_interval : ℕ := 25)
  (level2_minutes : ℕ := 210)
  (level2_gain_rate : ℕ := 4)
  (level2_gain_interval : ℕ := 35)
  (level2_lost_rate : ℕ := 1)
  (level2_lost_interval : ℕ := 8)
  (level3_minutes : ℕ := 120)
  (level3_lost_rate : ℕ := 8)
  (level3_lost_interval : ℕ := 30)
  (level3_gain_rate : ℕ := 5)
  (level3_gain_interval : ℕ := 40) :
  let level1_lives_lost := (level1_minutes / level1_lost_interval) * level1_lost_rate,
      level1_lives_gained := (level1_minutes / level1_gain_interval) * level1_gain_rate,
      level1_net := level1_lives_gained - level1_lives_lost,
      level2_lives_gained := (level2_minutes / level2_gain_interval) * level2_gain_rate,
      level2_lives_lost := (level2_minutes / level2_lost_interval),
      level2_net := level2_lives_gained - level2_lives_lost,
      level3_lives_lost := (level3_minutes / level3_lost_interval) * level3_lost_rate,
      level3_lives_gained := (level3_minutes / level3_gain_interval) * level3_gain_rate,
      level3_net := level3_lives_gained - level3_lives_lost,
      final_lives := initial_lives + level1_net + level2_net + level3_net 
  in final_lives = 270 :=
by
  sorry

end victor_lives_at_end_l332_332660


namespace geometric_sequence_log_sum_l332_332436

/-
Given a geometric sequence {a_n} with a_n > 0 and a_1 * a_{2009} = 2^{2010},
prove that the sum of logarithms at base 2 of every other term from a_1 to a_{2009} equals 1005^2.
-/

theorem geometric_sequence_log_sum :
  (∃ (a : ℕ → ℝ), (∀ n, a n > 0) ∧ (∀ n, a (n+1) = a 1 * (2 : ℝ) ^ (2 * n)) ∧ a 1 * a 2009 = 2 ^ 2010) →
    ∑ i in (finset.range 1005).image (λ i, 2 * i + 1), real.log (a i) / real.log 2 = 1005 ^ 2 :=
begin
  sorry
end

end geometric_sequence_log_sum_l332_332436


namespace circus_illumination_l332_332786

theorem circus_illumination (n : ℕ) (hl_cond : n >= 2) :
  (∃ (spotlights : Fin n → Set ℝ) (illuminates_convex : ∀ i, Convex (spotlights i)),
   (∀ i, (⋂ (j : Fin n), spotlights j) = (⋂ (j : Fin n | j ≠ i), spotlights j)) ∧
   (∀ i j, i ≠ j → (⋂ (k : Fin n), spotlights k) ≠ (⋂ (k : Fin n | k ≠ i ∧ k ≠ j), spotlights k))) := 
sorry

end circus_illumination_l332_332786


namespace impossible_to_have_50_and_51_l332_332625

/-- 
Given the sequence of numbers from 1 to 100 and the allowed operation (erasing several consecutive 
numbers and replacing them with the number of elements erased), prove that it is impossible to 
have only the numbers 50 and 51 remaining on the board.
-/
theorem impossible_to_have_50_and_51 : 
  (1 ≤ 100) ∧ (∀ k : ℕ, 1 ≤ k → k ≤ 100 → 
   (∃ l : ℕ, l < k → (erased : list ℕ → erased.length = l) → true)) → false :=
sorry

end impossible_to_have_50_and_51_l332_332625


namespace find_x_l332_332982

theorem find_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) → x = 1.5 := 
by 
  sorry

end find_x_l332_332982


namespace sum_of_solutions_equation_l332_332931

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332931


namespace number_of_three_letter_initials_l332_332479

theorem number_of_three_letter_initials : 
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] in
  (length letters) ^ 3 = 1000 := 
by
  -- Definitions:
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  -- The problem reduces to computing 10^3 and showing it equals 1000.
  have length_letters : length letters = 10 := by sorry
  calc 
    (length letters) ^ 3
      = 10 ^ 3 : by rw length_letters
  ... = 1000 : by norm_num

end number_of_three_letter_initials_l332_332479


namespace exists_bag_with_7_candies_l332_332643

theorem exists_bag_with_7_candies
  (candies : Fin 5 → ℕ)
  (distinct : Function.Injective candies)
  (sum_candies : Finset.univ.sum candies = 21)
  (redistribute : ∀ i j : Fin 5, ∃ x : ℕ, ∑ k in (Finset.univ \ {i, j} : Finset (Fin 5)), candies k = 3 * x) :
  ∃ i : Fin 5, candies i = 7 :=
by
  sorry

end exists_bag_with_7_candies_l332_332643


namespace smaller_angle_at_3_40_l332_332736

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332736


namespace sum_of_solutions_eq_16_l332_332924

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332924


namespace tangent_line_ln_at_e_l332_332877

noncomputable def tangent_line_eqn (x y : ℝ) : Prop :=
  x - real.exp(1) * y = 0

theorem tangent_line_ln_at_e :
  ∃ x y : ℝ, y = real.log x ∧ x = real.exp(1) → tangent_line_eqn x y :=
by
  sorry

end tangent_line_ln_at_e_l332_332877


namespace geometric_progression_b_formula_a_n_range_of_a_l332_332418

noncomputable def sequence_a (a : ℝ) (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then a
  else 2 * S (n - 1) + 4 ^ (n - 1)

noncomputable def sequence_S (a : ℝ) (S : ℕ → ℝ) : ℕ → ℝ
| 0 := 0
| 1 := a
| (n + 1) := 3 * S n + 4 ^ n

noncomputable def sequence_b (a : ℝ) (n : ℕ) (S : ℕ → ℝ) : ℝ :=
  (sequence_S a S n) - 4 ^ n

theorem geometric_progression_b (a : ℝ) (S : ℕ → ℝ) (n : ℕ) (h₀ : n ≠ 0) : 
  sequence_b a (n + 1) S = 3 * (sequence_b a n S) := 
sorry

theorem formula_a_n (a : ℝ) (S : ℕ → ℝ) (h₀ : ∀ n, S n = sequence_S a S n) (n : ℕ) : 
  (sequence_a a S n) = 
    if n = 0 then 0 else if n = 1 then a
    else 3 * 4 ^ (n - 1) + 2 * (a - 4) * 3 ^ (n - 2) :=
sorry

theorem range_of_a (a : ℝ) (h₀ : a ≠ 4) (h₁ : ∀ n : ℕ, n ≠ 0 → (sequence_a a (sequence_S a (λ n, sequence_S a (λ n, 0))) (n + 1)) ≥ (sequence_a a (sequence_S a (λ n, sequence_S a (λ n, 0))) n)) :
  a ∈ set.Icc (-4 : ℝ) 4 := 
sorry

end geometric_progression_b_formula_a_n_range_of_a_l332_332418


namespace rectangle_perimeter_l332_332268

-- Define the sides of triangle DEF
def a := 10
def b := 24
def c := 26

-- Define the width of the rectangle
def width := 8

-- Prove that the area of triangle DEF is equal to 120 square units
lemma triangle_area :
  (∃ (area : ℝ), area = 0.5 * a * b) := by
  use 0.5 * a * b
  norm_num

-- Prove that the perimeter of the rectangle is 46 units
theorem rectangle_perimeter :
  (∃ (perimeter : ℝ), perimeter = 2 * (width + 120 / width)) := by
  use 2 * (width + 120 / width)
  norm_num

end rectangle_perimeter_l332_332268


namespace solve_modular_expression_l332_332868

theorem solve_modular_expression :
  ∃ (x y : ℤ), (7 * x ≡ 1 [MOD 60]) ∧ (13 * y ≡ 1 [MOD 60]) ∧
  (3 * x + 9 * y ≡ 42 [MOD 60]) := by
  sorry

end solve_modular_expression_l332_332868


namespace general_term_formula_sum_first_n_of_b_l332_332423

-- The arithmetic sequence a_n with first term a_1 and common difference d
def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
  ∀ n, a n = a₁ + n * d

-- The sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (2 * a 0 + (n - 1) * d) / 2

-- Given conditions
variable (a : ℕ → ℝ)
variables {a₁ d S : ℕ → ℝ}

axiom S_10 : S 10 = 110
axiom geo_seq : (a 1) ^ 2 = a 0 * (a 0 + 3 * d)

-- Prove the general term of the sequence
theorem general_term_formula (ha : arithmetic_sequence a a₁ d)
  (hs : sum_first_n_terms a S) : (a₁ = 2 ∧ d = 2) → ∀ n, a n = 2 * n := sorry

-- Define the new sequence b_n
def b (n : ℕ) : ℝ := 1 / ((a n - 1) * (a n + 1))

-- Sum of the first n terms of the sequence b_n
def T (n : ℕ) := ∑ i in finset.range n, b i

-- Prove the sum of the first n terms of the sequence b_n
theorem sum_first_n_of_b (hb : ∀ n, a n = 2 * n) (ht : T = λ n, ∑ i in finset.range n, b i)
  : ∀ n, T n = n / (2 * n + 1) := sorry

end general_term_formula_sum_first_n_of_b_l332_332423


namespace natural_number_pairs_l332_332378

theorem natural_number_pairs (a b : ℕ) (p q : ℕ) :
  a ≠ b →
  (∃ p, a + b = 2^p) →
  (∃ q, ab + 1 = 2^q) →
  (a = 1 ∧ b = 2^p - 1 ∨ a = 2^q - 1 ∧ b = 2^q + 1) :=
by intro hne hp hq; sorry

end natural_number_pairs_l332_332378


namespace find_value_of_x_l332_332316

theorem find_value_of_x :
  ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ x = 230 :=
by
  sorry

end find_value_of_x_l332_332316


namespace day12_is_monday_l332_332092

-- Define the days of the week
inductive WeekDay
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open WeekDay

-- Define the problem using the conditions
def five_fridays_in_month (first_day : WeekDay) (last_day : WeekDay) : Prop :=
  first_day ≠ Friday ∧ last_day ≠ Friday ∧
  ((first_day = Monday ∨ first_day = Tuesday ∨ first_day = Wednesday ∨ first_day = Thursday ∨
    first_day = Saturday ∨ first_day = Sunday) ∧
  ∃ fridays : Finset ℕ,
  fridays.card = 5 ∧
  ∀ n ∈ fridays, (n % 7 = (5 - WeekDay.recOn first_day 6 0 1 2 3 4 5)) ∧
  fridays ⊆ Finset.range 31 ∧
  1 ∉ fridays ∧ (31 - Finset.max' fridays sorry) % 7 ≠ 0 )

-- Given the problem, prove that the 12th day is a Monday
theorem day12_is_monday (first_day last_day : WeekDay)
  (h : five_fridays_in_month first_day last_day) : 
  (12 % 7 + WeekDay.recOn first_day 6 0 1 2 3 4 5) % 7 = 0 :=
sorry

end day12_is_monday_l332_332092


namespace angle_at_3_40_pm_is_130_degrees_l332_332669

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332669


namespace binary_101101_is_45_l332_332363

def bin_to_dec (n : List ℕ) : ℕ :=
  n.foldr (λ (bit : ℕ) (acc : ℕ), acc * 2 + bit) 0

theorem binary_101101_is_45 : bin_to_dec [1, 0, 1, 1, 0, 1] = 45 := by
  sorry

end binary_101101_is_45_l332_332363


namespace total_number_of_fish_l332_332251

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l332_332251


namespace tank_fish_count_l332_332255

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l332_332255


namespace AN_parallel_CK_l332_332784

open EuclideanGeometry

variables (ω : Circle) 
          (A B C K M L N : Point) 
          (h1 : Triangle A B C) 
          (h2 : Circle.CircumCircle ω A B C)
          (h3 : TangentAt ω C K)
          (h4 : Midpoint M C K)
          (h5 : Line.Intersects BM L ω)
          (h6 : Line.Intersects KL N ω)
          (h7 : A, B, C, K, M, L, N : Point)

theorem AN_parallel_CK : Parallel AN CK := by
  -- Proof will be filled in here
  sorry

end AN_parallel_CK_l332_332784


namespace median_not_3_98_l332_332299

noncomputable def data_set : List ℝ := [3.9, 4.1, 3.9, 3.8, 4.2]

def median (l : List ℝ) : ℝ :=
  let sorted := l.sort
  sorted.sorted.nth! (sorted.length / 2)

theorem median_not_3_98 : median data_set ≠ 3.98 := by
  sorry

end median_not_3_98_l332_332299


namespace twelfth_day_is_monday_l332_332065

def Month := ℕ
def Day := ℕ

-- Definitions for days of the week, where 0 represents Monday, 1 represents Tuesday, etc.
inductive Weekday : Type
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
| Sunday : Weekday

open Weekday

-- A month has exactly 5 Fridays
def has_five_fridays (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday ∧
    (∃ last_day: Weekday, last_day ≠ Friday ∧ 
      let fridays := List.filter (λ d, d = Friday) (List.range 31) in
      fridays.length = 5)

-- The first day of the month is not a Friday
def first_day_not_friday (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday

-- The last day of the month is not a Friday
def last_day_not_friday (month: Month): Prop :=
  ∀ last_day: Weekday, last_day = (29 % 7) → last_day ≠ Friday

-- Combining the conditions for the problem
def valid_month (month: Month): Prop :=
  has_five_fridays(month) ∧ first_day_not_friday(month) ∧ last_day_not_friday(month)

-- Prove that the 12th day of the month is a Monday given the conditions
theorem twelfth_day_is_monday (month: Month) (h: valid_month(month)): (∃ starts_on: Weekday, starts_on + 11 = Monday) :=
sorry

end twelfth_day_is_monday_l332_332065


namespace total_pink_crayons_l332_332569

-- Define the conditions
def Mara_crayons : ℕ := 40
def Mara_pink_percent : ℕ := 10
def Luna_crayons : ℕ := 50
def Luna_pink_percent : ℕ := 20

-- Define the proof problem statement
theorem total_pink_crayons : 
  (Mara_crayons * Mara_pink_percent / 100) + (Luna_crayons * Luna_pink_percent / 100) = 14 := 
by sorry

end total_pink_crayons_l332_332569


namespace lattice_points_count_l332_332104

open Int

theorem lattice_points_count : 
  {p : ℤ × ℤ × ℤ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.toFinset.card = 30 :=
by
  sorry

end lattice_points_count_l332_332104


namespace clock_angle_l332_332702

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332702


namespace question_answer_l332_332543

def polynomial (x : ℝ) : ℝ := 10 * x^3 + 101 * x + 210

noncomputable def roots_satisfy : ℝ × ℝ × ℝ :=
  Classical.some (exists_root_polynomial polynomial)

def a := (roots_satisfy).1
def b := (roots_satisfy).2
def c := (roots_satisfy).3

theorem question_answer :
  polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0 ∧ 
  (a + b + c = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 63 := 
begin
  sorry
end

end question_answer_l332_332543


namespace nth_term_sequence_l332_332579

theorem nth_term_sequence (n : ℕ) (h : n > 0) : 
  let a : ℕ → ℝ := λ n, 
    match n with
    | 1 => √3
    | 2 => 3
    | 3 => √15
    | 4 => √21
    | 5 => 3*√3
    | _ => √(6 * n - 3)
  in a n = √(6 * n - 3) := 
by
  sorry

end nth_term_sequence_l332_332579


namespace twelfth_day_is_monday_l332_332067

def Month := ℕ
def Day := ℕ

-- Definitions for days of the week, where 0 represents Monday, 1 represents Tuesday, etc.
inductive Weekday : Type
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
| Sunday : Weekday

open Weekday

-- A month has exactly 5 Fridays
def has_five_fridays (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday ∧
    (∃ last_day: Weekday, last_day ≠ Friday ∧ 
      let fridays := List.filter (λ d, d = Friday) (List.range 31) in
      fridays.length = 5)

-- The first day of the month is not a Friday
def first_day_not_friday (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday

-- The last day of the month is not a Friday
def last_day_not_friday (month: Month): Prop :=
  ∀ last_day: Weekday, last_day = (29 % 7) → last_day ≠ Friday

-- Combining the conditions for the problem
def valid_month (month: Month): Prop :=
  has_five_fridays(month) ∧ first_day_not_friday(month) ∧ last_day_not_friday(month)

-- Prove that the 12th day of the month is a Monday given the conditions
theorem twelfth_day_is_monday (month: Month) (h: valid_month(month)): (∃ starts_on: Weekday, starts_on + 11 = Monday) :=
sorry

end twelfth_day_is_monday_l332_332067


namespace irrational_of_pi_over_3_l332_332768

theorem irrational_of_pi_over_3 :
  let A := (Real.pi - 1) ^ 0
  let B := Real.pi / 3
  let C := 5.010010001
  let D := 3.14
  Irrational B ∧ ¬Irrational A ∧ ¬Irrational C ∧ ¬Irrational D :=
by 
  sorry

end irrational_of_pi_over_3_l332_332768


namespace inequality_of_triangle_tangents_l332_332562

theorem inequality_of_triangle_tangents
  (a b c x y z : ℝ)
  (h1 : a = y + z)
  (h2 : b = x + z)
  (h3 : c = x + y)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_tangents : z ≥ y ∧ y ≥ x) :
  (a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2) ∧
  ((a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z) :=
sorry

end inequality_of_triangle_tangents_l332_332562


namespace altered_solution_detergent_volume_l332_332777

theorem altered_solution_detergent_volume 
  (bleach : ℕ)
  (detergent : ℕ)
  (water : ℕ)
  (h1 : bleach / detergent = 4 / 40)
  (h2 : detergent / water = 40 / 100)
  (ratio_tripled : 3 * (bleach / detergent) = bleach / detergent)
  (ratio_halved : (detergent / water) / 2 = (detergent / water))
  (altered_water : water = 300) : 
  detergent = 60 := 
  sorry

end altered_solution_detergent_volume_l332_332777


namespace option_C_equiv_option_D_equiv_l332_332294

-- Define the conditions for Option C
def func_c1 (x : ℝ) (hx : x ≠ 0) : ℝ := x^0
def func_c2 (x : ℝ) : ℝ := 1

-- Define the conditions for Option D
def func_d1 (x : ℝ) : ℝ := x^2
def func_d2 (t : ℝ) : ℝ := t^2

-- Statements to prove
theorem option_C_equiv : ∀ x : ℝ, x ≠ 0 → func_c1 x ‹x ≠ 0› = func_c2 x := by
  sorry

theorem option_D_equiv : ∀ x : ℝ, func_d1 x = func_d2 x := by
  sorry

end option_C_equiv_option_D_equiv_l332_332294


namespace tan_double_angle_l332_332243

-- Define the conditions
def angle_vertex_at_origin (θ : ℝ) : Prop := true -- Placeholder condition for the vertex at origin
def initial_side_nonneg_x_axis (θ : ℝ) : Prop := true -- Placeholder condition for initial side on non-negative x-axis
def terminal_side_on_line_y_eq_2x (θ : ℝ) : Prop := tan θ = 2

-- The main theorem to prove
theorem tan_double_angle (θ : ℝ) 
  (h1 : angle_vertex_at_origin θ) 
  (h2 : initial_side_nonneg_x_axis θ) 
  (h3 : terminal_side_on_line_y_eq_2x θ) : 
  tan (2 * θ) = -4 / 3 :=
by
  sorry

end tan_double_angle_l332_332243


namespace sum_of_first_9_terms_arithmetic_sequence_l332_332424

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem sum_of_first_9_terms_arithmetic_sequence
  (h_arith_seq : is_arithmetic_sequence a)
  (h_condition : a 2 + a 8 = 8) :
  (Finset.range 9).sum a = 36 :=
sorry

end sum_of_first_9_terms_arithmetic_sequence_l332_332424


namespace tank_fill_time_l332_332047

theorem tank_fill_time (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1/30) (hB : B_rate = 1/20) (hC : C_rate = -1/40) : 
  1 / (A_rate + B_rate + C_rate) = 120 / 7 :=
by
  -- proof goes here
  sorry

end tank_fill_time_l332_332047


namespace value_of_m_l332_332494

theorem value_of_m (m : ℝ) (h1 : m^2 - 2 * m - 1 = 2) (h2 : m ≠ 3) : m = -1 :=
sorry

end value_of_m_l332_332494


namespace clock_strikes_times_l332_332787

theorem clock_strikes_times (T : ℕ → ℚ) (U : ℕ → ℚ) :
  (∀ n, T n = (n-1) * (10 / 9)) ∧
  U 8 = T 8 ∧ U 15 = T 15 :=
by
  -- Define the time function
  have T_is_defined : ∀ n, T n = (n-1) * (10 / 9),
  from sorry,
  have U_8_is_correct : U 8 = (8-1) * (10 / 9),
  from sorry,
  have U_15_is_correct : U 15 = (15-1) * (10 / 9),
  from sorry,
  exact ⟨T_is_defined, U_8_is_correct, U_15_is_correct⟩

end clock_strikes_times_l332_332787


namespace Missouri_to_NewYork_by_car_l332_332215

def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def total_distance_car : ℝ := distance_plane * (1 + increase_percentage)
def distance_midway : ℝ := total_distance_car / 2

theorem Missouri_to_NewYork_by_car : distance_midway = 1400 := by
  sorry

end Missouri_to_NewYork_by_car_l332_332215


namespace self_employed_sample_size_l332_332061

noncomputable def population : ℕ := 3200
noncomputable def unemployed : ℕ := 1000
noncomputable def selfEmployed : ℕ := 1160
noncomputable def salariedWorkers : ℕ := 1040
noncomputable def sampleSize : ℕ := 160
noncomputable def samplingProportion : ℚ := 160 / 3200

theorem self_employed_sample_size :
  let unemployedSampleSize := unemployed * samplingProportion in
  let selfEmployedSampleSize := selfEmployed * samplingProportion in
  let salariedWorkersSampleSize := salariedWorkers * samplingProportion in
  selfEmployedSampleSize = 58 := 
by
  sorry

end self_employed_sample_size_l332_332061


namespace sum_of_solutions_sum_of_all_solutions_l332_332910

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332910


namespace smaller_angle_3_40_l332_332699

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332699


namespace area_of_circle_irrational_l332_332785

theorem area_of_circle_irrational (r : ℝ) (π_ne_zero : π ≠ 0) :
  (∃ (r : ℝ), 2 * π * r = 4 * 2) ∧ r = 4 / π ∧ ∃ (A : ℝ), A = π * (r ^ 2) → irrational (π * (4 / π) ^ 2) :=
by
  -- Given conditions
  have condition1 : 2 * π * (4 / π) = 8 := by sorry
  have area_of_circle : π * (4 / π) ^ 2 = 16 / π := by sorry
  have irrational_16_over_pi : irrational (16 / π) := by sorry
  -- Conclude that the area of the circle is irrational
  exact irrational_16_over_pi

#check area_of_circle_irrational

end area_of_circle_irrational_l332_332785


namespace non_talking_birds_count_l332_332795

def total_birds : ℕ := 77
def talking_birds : ℕ := 64

theorem non_talking_birds_count : total_birds - talking_birds = 13 := by
  sorry

end non_talking_birds_count_l332_332795


namespace complex_numbers_on_circle_l332_332160

noncomputable def S : ℝ := sorry -- S is a real number ∣S∣ ≤ 2

theorem complex_numbers_on_circle
  (a1 a2 a3 a4 a5 : ℂ)
  (q : ℂ)
  (h1 : q ≠ 0)
  (condition1 : a2 = q * a1)
  (condition2 : a3 = q * a2)
  (condition3 : a4 = q * a3)
  (condition4 : a5 = q * a4)
  (condition5 : (a1 + a2 + a3 + a4 + a5) = 4 * (1/a1 + 1/a2 + 1/a3 + 1/a4 + 1/a5))
  (condition6 : |S| ≤ 2) :
  ∃ r : ℝ, ∀ i, ∃ z : ℂ, z = a1 ∨ z = a2 ∨ z = a3 ∨ z = a4 ∨ z = a5 ∧ |z| = r :=
begin
  sorry
end

end complex_numbers_on_circle_l332_332160


namespace clock_angle_3_40_l332_332679

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332679


namespace problem1_problem2_l332_332108

-- Problem 1
theorem problem1 (t α : ℝ) (t1 t2 : ℝ) (cosα sinα : ℝ) :
  (x y : ℝ) (h : x = 1 + t * cosα ∧ y = 1 + t * sinα)
  (h1 : (1 + t1 * cosα)^2 + (1 + t1 * sinα)^2 = 1)
  (h2 : (1 + t2 * cosα)^2 + (1 + t2 * sinα)^2 = 1)
  (M : ℝ × ℝ)
  (m : M = (1, 1)) :
  |t1 * t2| = 1 :=
sorry

-- Problem 2
theorem problem2 :
  ∀ (x y x' y' : ℝ) (h : x' = √3 * x ∧ y' = y)
  (hC1 : x^2 + y^2 = 1)
  (hxhy : x = √3 * cos θ ∧ y = sin θ)
  (θ : ℝ) :
  let perimeter := 4 * (√3 * cos θ + sin θ)
  in perimeter ≤ 8 :=
sorry

end problem1_problem2_l332_332108


namespace small_angle_at_3_40_is_130_degrees_l332_332751

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332751


namespace product_geometric_sequence_l332_332529

theorem product_geometric_sequence (n : ℕ) (hn : 0 < n) :
  ∃ r, (∀ i : ℕ, i = 1 ∨ (i ≤ n → r^i = 100) ) → ∏ i in finset.range n, r^(i+1) = 10^n := 
sorry

end product_geometric_sequence_l332_332529


namespace smaller_angle_3_40_pm_l332_332749

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332749


namespace max_value_of_a_ln_b_l332_332995

theorem max_value_of_a_ln_b (a b : ℝ) (h1 : a > 1) (h2 : b > 1) (hP : b = (real.exp 2) / a) : 
  a ^ real.log b ≤ real.exp 1 :=
by
  sorry

end max_value_of_a_ln_b_l332_332995


namespace A_n_eq_B_n_l332_332556

-- Definitions of A_n and B_n
def A_n (n : ℕ) : ℕ :=
  number_of_ways n (λ (a : ℕ) (k : ℕ) (seq : Fin k → ℕ), increasing seq ∧ no_repeat_even seq)

def B_n (n : ℕ) : ℕ :=
  number_of_ways n (λ (a : ℕ) (k : ℕ) (seq : Fin k → ℕ), increasing seq ∧ no_more_than_three seq)

-- Conditions for sequences
def increasing (seq : Fin k → ℕ) : Prop :=
  ∀ i j, i ≤ j → seq i ≤ seq j

def no_repeat_even (seq : Fin k → ℕ) : Prop :=
  ∀ i j, i ≠ j → even (seq i) → seq i ≠ seq j

def no_more_than_three (seq : Fin k → ℕ) : Prop :=
  ∀ i x, (count_occurrences seq x) ≤ 3

-- Main theorem
theorem A_n_eq_B_n (n : ℕ) : A_n n = B_n n := by
  sorry

end A_n_eq_B_n_l332_332556


namespace denis_sum_of_numbers_l332_332365

theorem denis_sum_of_numbers :
  ∃ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ a*d = 32 ∧ b*c = 14 ∧ a + b + c + d = 42 :=
sorry

end denis_sum_of_numbers_l332_332365


namespace sum_of_solutions_eq_16_l332_332922

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332922


namespace ratio_boys_girls_l332_332102

theorem ratio_boys_girls
  (B G : ℕ)  -- Number of boys and girls
  (h_ratio : 75 * G = 80 * B)
  (h_total_no_scholarship : 100 * (3 * B + 4 * G) = 7772727272727272 * (B + G)) :
  B = 5 * G := sorry

end ratio_boys_girls_l332_332102


namespace min_value_expression_l332_332437

noncomputable def log (base : ℝ) (num : ℝ) := Real.log num / Real.log base

theorem min_value_expression (a b : ℝ) (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * log a b + 6 * log b a = 11) : 
  a^3 + (2 / (b - 1)) ≥ 2 * Real.sqrt 2 + 1 :=
by
  sorry

end min_value_expression_l332_332437


namespace sum_of_roots_eq_seventeen_l332_332913

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332913


namespace lampshire_parade_group_max_members_l332_332606

theorem lampshire_parade_group_max_members 
  (n : ℕ) 
  (h1 : 30 * n % 31 = 7)
  (h2 : 30 * n % 17 = 0)
  (h3 : 30 * n < 1500) :
  30 * n = 1020 :=
sorry

end lampshire_parade_group_max_members_l332_332606


namespace num_divisible_3_5_11_is_6_l332_332484

noncomputable def num_divisible_by_3_5_11 : ℕ := ∃ n : ℕ, ∀ k : ℕ, (100 ≤ k) ∧ (k ≤ 999) ∧ (k % 3 = 0) ∧ (k % 5 = 0) ∧ (k % 11 = 0) ↔ k ∈ (list.range' 165 6.map (λ i, 165 + 165 * i))

theorem num_divisible_3_5_11_is_6 : num_divisible_by_3_5_11 = 6 := 
by 
  -- Prove the correct count of numbers 
  sorry 

end num_divisible_3_5_11_is_6_l332_332484


namespace sum_of_solutions_l332_332894

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332894


namespace polynomial_has_integer_root_l332_332552

noncomputable def P : Polynomial ℤ := sorry

theorem polynomial_has_integer_root
  (P : Polynomial ℤ)
  (h_deg : P.degree = 3)
  (h_infinite_sol : ∀ (x y : ℤ), x ≠ y → x * P.eval x = y * P.eval y → 
  ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y) :
  ∃ k : ℤ, P.eval k = 0 :=
sorry

end polynomial_has_integer_root_l332_332552


namespace sequence_10_eq_123_l332_332578

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 3
  | n + 2 => sequence (n + 1) + sequence n

theorem sequence_10_eq_123 : sequence 10 = 123 :=
by
  sorry

end sequence_10_eq_123_l332_332578


namespace stairs_ways_l332_332865

def f : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 2
| n := f (n - 1) + f (n - 2) + if n ≥ 4 then f (n - 4) else 0
                      + if n ≥ 5 then f (n - 5) else 0

theorem stairs_ways : f 8 = 52 := 
by
  sorry

end stairs_ways_l332_332865


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332885

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332885


namespace sum_of_solutions_eq_16_l332_332975

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332975


namespace check_blank_value_l332_332115

/-- Define required constants and terms. -/
def six_point_five : ℚ := 6 + 1/2
def two_thirds : ℚ := 2/3
def three_point_five : ℚ := 3 + 1/2
def one_and_eight_fifteenths : ℚ := 1 + 8/15
def blank : ℚ := 3 + 1/20
def seventy_one_point_ninety_five : ℚ := 71 + 95/100

/-- The translated assumption and statement to be proved: -/
theorem check_blank_value :
  (six_point_five - two_thirds) / three_point_five - one_and_eight_fifteenths * (blank + seventy_one_point_ninety_five) = 1 :=
sorry

end check_blank_value_l332_332115


namespace sum_of_roots_eq_seventeen_l332_332920

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332920


namespace smaller_angle_3_40_l332_332692

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332692


namespace max_students_l332_332310

-- Definitions based on the conditions
variable (Students Clubs : Type)
variable [Fintype Students] [Fintype Clubs]
variable [DecidableEq Students]

-- Assume there are exactly 10 clubs
axiom club_count : Fintype.card Clubs = 10

-- For any two distinct students, there is a club such that exactly one belongs to that club
axiom club_condition_pair : ∀ (s1 s2 : Students) (h : s1 ≠ s2), ∃ c : Clubs, (c ∈ s1.clubs ∧ c ∉ s2.clubs) ∨ (c ∉ s1.clubs ∧ c ∈ s2.clubs)

-- For any three distinct students, there is a club such that either exactly one or all three belong to that club
axiom club_condition_triplet : ∀ (s1 s2 s3 : Students) (h1 : s1 ≠ s2) (h2 : s2 ≠ s3) (h3 : s1 ≠ s3), 
  ∃ c : Clubs, (c ∈ s1.clubs ∧ c ∉ s2.clubs ∧ c ∉ s3.clubs) ∨ 
               (c ∉ s1.clubs ∧ c ∈ s2.clubs ∧ c ∉ s3.clubs) ∨ 
               (c ∉ s1.clubs ∧ c ∉ s2.clubs ∧ c ∈ s3.clubs) ∨ 
               (c ∈ s1.clubs ∧ c ∈ s2.clubs ∧ c ∈ s3.clubs)

-- Prove that the largest number of students under these conditions is 513
theorem max_students : Fintype.card Students ≤ 513 := 
sorry

end max_students_l332_332310


namespace clock_angle_3_40_l332_332712

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332712


namespace twelve_is_monday_l332_332083

def Weekday := {d : String // d ∈ ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}

def not_friday (d: Weekday) : Prop := d.val ≠ "Friday"

def has_exactly_five_fridays (first_friday: nat) (days_in_month: nat) : Prop :=
  first_friday + 28 <= days_in_month ∧
  first_friday + 21 <= days_in_month ∧
  first_friday + 14 <= days_in_month ∧
  first_friday + 7 <= days_in_month ∧
  first_friday > 0 ∧ days_in_month <= 31

noncomputable def compute_day_of_week (start_day: String) (n: nat) : Weekday :=
  sorry

theorem twelve_is_monday (start_day: Weekday) (days_in_month: nat) :
    has_exactly_five_fridays 2 days_in_month
  → not_friday start_day
  → not_friday (compute_day_of_week start_day.val days_in_month)
  → compute_day_of_week start_day.val 12 = ⟨"Monday", by sorry⟩ :=
begin
  sorry
end

end twelve_is_monday_l332_332083


namespace sum_of_solutions_eqn_l332_332942

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332942


namespace rain_first_hour_l332_332123

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end rain_first_hour_l332_332123


namespace truncated_pyramid_volume_correct_l332_332510

noncomputable def herons_formula_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def truncated_pyramid_volume (H S1 S2 : ℝ) : ℝ :=
  (1 / 3) * H * (S1 + S2 + Real.sqrt (S1 * S2))

theorem truncated_pyramid_volume_correct :
  let H := 10 
  let a := 27 
  let b := 29 
  let c := 25 
  let P1 := a + b + c
  let P2 := 72 
  let S1 := herons_formula_area a b c
  let S2 := (S1 * (P2 / P1)^2)
  truncated_pyramid_volume H S1 S2 ≈ 3265.3 :=
by
  sorry

end truncated_pyramid_volume_correct_l332_332510


namespace complement_A_correct_l332_332023

def U := set ℝ
def A : set ℝ := {y | ∃ x, x > 1 ∧ y = real.log x / real.log 2}
def complement_A : set ℝ := {y | y ≤ 0 }

theorem complement_A_correct : (U \ A) = complement_A := 
by sorry

end complement_A_correct_l332_332023


namespace probability_B_in_A_is_17_over_24_l332_332541

open Set

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs p.1 + abs p.2 <= 2}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ∈ set_A ∧ p.2 <= p.1 ^ 2}

noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry -- Assume we have means to compute the area of a set

theorem probability_B_in_A_is_17_over_24 :
  (area set_B / area set_A) = 17 / 24 :=
sorry

end probability_B_in_A_is_17_over_24_l332_332541


namespace general_term_of_sequence_l332_332433

-- Define the sequence S_n and the given condition log2(S_n + 1) = n + 1
def S (n : ℕ) := (2^(n + 1)) - 1

-- Definition of the sequence a_n we need to prove
def a (n : ℕ) : ℕ := if n = 1 then 3 else 2^n

-- The theorem stating the desired property
theorem general_term_of_sequence (n : ℕ) : 
  (S n = S n) → 
  (∀ n : ℕ, log 2 (S n + 1) = n + 1) →
  (∀ n : ℕ, S n = (a n + S (n - 1)) → 
  (a n = if n = 1 then 3 else 2^n)) := 
  by
    sorry

end general_term_of_sequence_l332_332433


namespace tank_fish_count_l332_332253

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l332_332253


namespace clock_angle_l332_332704

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332704


namespace angle_relationship_l332_332158

variables {AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1}
variables {angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ}

-- Define the conditions
def conditions (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 : ℝ) : Prop :=
  AB = A_1B_1 ∧ BC = B_1C_1 ∧ CD = C_1D_1 ∧ DA = D_1A_1 ∧ angleA > angleA1

theorem angle_relationship (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ)
  (h : conditions AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 angleA angleA1) :
  angleB < angleB1 ∧ angleC > angleC1 ∧ angleD < angleD1 :=
by {
  sorry
}

end angle_relationship_l332_332158


namespace day12_is_monday_l332_332090

-- Define the days of the week
inductive WeekDay
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open WeekDay

-- Define the problem using the conditions
def five_fridays_in_month (first_day : WeekDay) (last_day : WeekDay) : Prop :=
  first_day ≠ Friday ∧ last_day ≠ Friday ∧
  ((first_day = Monday ∨ first_day = Tuesday ∨ first_day = Wednesday ∨ first_day = Thursday ∨
    first_day = Saturday ∨ first_day = Sunday) ∧
  ∃ fridays : Finset ℕ,
  fridays.card = 5 ∧
  ∀ n ∈ fridays, (n % 7 = (5 - WeekDay.recOn first_day 6 0 1 2 3 4 5)) ∧
  fridays ⊆ Finset.range 31 ∧
  1 ∉ fridays ∧ (31 - Finset.max' fridays sorry) % 7 ≠ 0 )

-- Given the problem, prove that the 12th day is a Monday
theorem day12_is_monday (first_day last_day : WeekDay)
  (h : five_fridays_in_month first_day last_day) : 
  (12 % 7 + WeekDay.recOn first_day 6 0 1 2 3 4 5) % 7 = 0 :=
sorry

end day12_is_monday_l332_332090


namespace day_of_12th_l332_332079

theorem day_of_12th (month : Type) [decidable_eq month] [fintype month] 
  (is_friday : month → Prop) (is_first_day : month → Prop) (is_last_day : month → Prop)
  (days : list month) (nth_day : Π (n : ℕ), month)
  (five_fridays : ∃ (n : ℕ), (is_friday ∘ nth_day) '' (finset.range (min n 7)) = {5})
  (not_first_day_friday : ∀ d, is_first_day d → ¬ is_friday d)
  (not_last_day_friday : ∀ d, is_last_day d → ¬ is_friday d) : 
  nth_day 12 = "Monday" := 
sorry

end day_of_12th_l332_332079


namespace correct_operation_l332_332295

theorem correct_operation :
  let m : ℝ := sorry in
  (2 * m^2 + m^2 ≠ 3 * m^4) ∧
  (m^2 * m^4 ≠ m^8) ∧
  (m^4 / m^2 = m^2) ∧
  ((m^2) ^ 4 ≠ m^6) :=
by
  intros m
  split
  { intro h
    calc (2 * m^2 + m^2) = (2 + 1) * m^2 : by ring
                      ... = 3 * m^2 : by norm_num
                      ... ≠ 3 * m^4 : by sorry },
  split
  { intro h
    calc (m^2 * m^4) = m^(2 + 4) : by sorry
                  ... = m^6 : by norm_num
                  ... ≠ m^8 : by sorry },
  split
  { intro h
    calc (m^4 / m^2) = m^(4 - 2) : by sorry
                  ... = m^2 : by norm_num },
  { intro h
    calc ((m^2)^4) = m^(2 * 4) : by sorry
                  ... = m^8 : by norm_num
                  ... ≠ m^6 : by sorry }

end correct_operation_l332_332295


namespace tax_per_pound_is_one_l332_332117

-- Define the conditions
def bulk_price_per_pound : ℝ := 5          -- Condition 1
def minimum_spend : ℝ := 40               -- Condition 2
def total_paid : ℝ := 240                 -- Condition 4
def excess_pounds : ℝ := 32               -- Condition 5

-- Define the proof problem statement
theorem tax_per_pound_is_one :
  ∃ (T : ℝ), total_paid = (minimum_spend / bulk_price_per_pound + excess_pounds) * bulk_price_per_pound + 
  (minimum_spend / bulk_price_per_pound + excess_pounds) * T ∧ 
  T = 1 :=
by 
  sorry

end tax_per_pound_is_one_l332_332117


namespace inclination_of_line_through_origin_and_neg1_neg1_l332_332619

noncomputable def angle_of_inclination (p1 p2 : ℝ × ℝ) : ℝ :=
if h : p1.1 ≠ p2.1 then real.arctan ((p2.2 - p1.2) / (p2.1 - p1.1)) else 90

theorem inclination_of_line_through_origin_and_neg1_neg1 :
  angle_of_inclination (0, 0) (-1, -1) = 45 :=
by
  sorry

end inclination_of_line_through_origin_and_neg1_neg1_l332_332619


namespace problem_inequality_problem_range_of_a_l332_332220

noncomputable def f (x : ℝ) : ℝ :=
  (x^(-3) * Real.exp(x) - x - 1) / Real.log(x)

theorem problem_inequality (x : ℝ) (hx : 1 < x) : 
  (∀ a, (x^(-3) * Real.exp(x) - Real.log(x) ≥ x + 1) → a ≤ f(x)) :=
sorry

theorem problem_range_of_a :
  ∀ a, (∀ x > 1, x^(-3) * Real.exp(x) - Real.log(x) ≥ x + 1 → a ≤ -3) :=
sorry

end problem_inequality_problem_range_of_a_l332_332220


namespace angle_at_3_40_pm_is_130_degrees_l332_332666

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332666


namespace m_plus_n_eq_e_l332_332546

noncomputable def m : ℝ := ∫ x in 0..1, Real.exp x
noncomputable def n : ℝ := ∫ x in 1..Real.exp 1, x⁻¹

theorem m_plus_n_eq_e : m + n = Real.exp 1 := by
  sorry

end m_plus_n_eq_e_l332_332546


namespace jack_to_jill_valid_paths_l332_332133

theorem jack_to_jill_valid_paths : 
  let total_paths := binomial 8 5,
      paths_through_21 := binomial 3 2 * binomial 5 3,
      paths_through_12 := binomial 3 1 * binomial 5 1,
      double_counted_paths := binomial 2 1 * binomial 4 2
  in total_paths - (paths_through_21 + paths_through_12 - double_counted_paths) = 23 := 
by
  sorry

end jack_to_jill_valid_paths_l332_332133


namespace smaller_angle_3_40_l332_332700

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332700


namespace day12_is_monday_l332_332094

-- Define the days of the week
inductive WeekDay
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

open WeekDay

-- Define the problem using the conditions
def five_fridays_in_month (first_day : WeekDay) (last_day : WeekDay) : Prop :=
  first_day ≠ Friday ∧ last_day ≠ Friday ∧
  ((first_day = Monday ∨ first_day = Tuesday ∨ first_day = Wednesday ∨ first_day = Thursday ∨
    first_day = Saturday ∨ first_day = Sunday) ∧
  ∃ fridays : Finset ℕ,
  fridays.card = 5 ∧
  ∀ n ∈ fridays, (n % 7 = (5 - WeekDay.recOn first_day 6 0 1 2 3 4 5)) ∧
  fridays ⊆ Finset.range 31 ∧
  1 ∉ fridays ∧ (31 - Finset.max' fridays sorry) % 7 ≠ 0 )

-- Given the problem, prove that the 12th day is a Monday
theorem day12_is_monday (first_day last_day : WeekDay)
  (h : five_fridays_in_month first_day last_day) : 
  (12 % 7 + WeekDay.recOn first_day 6 0 1 2 3 4 5) % 7 = 0 :=
sorry

end day12_is_monday_l332_332094


namespace LTF_prevents_Sunny_iff_even_l332_332245

def LTF_prevents_Sunny_winning (n : ℕ) : Prop :=
  ∀ (cards : Fin n → ℝ), -- the list of positive real numbers on the cards
  ((∀ i, 0 < cards i) ∧ -- condition that all cards have positive real numbers
  (n % 2 = 0 →        -- if n is even
     ∃ strategy_for_LTF : Fin n → Bool,  -- LTF's strategy, Bool indicates if taking the card or not
       strategy_for_Sunny : Fin n → Bool, -- Sunny's strategy
       (∑ i, if strategy_for_LTF i then cards i else 0) ≥ (∑ i, if strategy_for_Sunny i then cards i else 0)) ∧
  (n % 2 = 1 →        -- if n is odd
      ∃ cards : Fin n → ℝ, -- Example sequence of cards
        ¬ (∃ strategy_for_LTF : Fin n → Bool, -- there does not exist a strategy for LTF
            (∑ i, if strategy_for_LTF i then cards i else 0) ≥ (∑ i, if strategy_for_Sunny i then cards i else 0))))

theorem LTF_prevents_Sunny_iff_even (n : ℕ) :
  LTF_prevents_Sunny_winning n ↔ n % 2 = 0 :=
sorry -- proof will be filled in later

end LTF_prevents_Sunny_iff_even_l332_332245


namespace midpoints_intersect_at_centroids_l332_332517

-- Define the points and properties of the convex quadrilateral ABCD
variables {A B C D G1 G2 G3 G4 : Type} [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
variables (A B C D G1 G2 G3 G4 : ℝ × ℝ)

-- Define a convex quadrilateral ABCD
def convex_quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), is_convex P (insert A (insert B (insert C {D})))

-- Define the centroids of triangles ABC, BCD, CDA, and DAB
def centroid (X Y Z : ℝ × ℝ) : ℝ × ℝ :=
  ((X.1 + Y.1 + Z.1) / 3, (X.2 + Y.2 + Z.2) / 3)

-- Define quadrilateral KLMN
def quadrilateral_KLMN (A B C D : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let K := centroid A B C,
      L := centroid B C D,
      M := centroid D A B,
      N := centroid C D A in
  (K, L, M, N)

-- Define the midpoints of the sides of a quadrilateral
def midpoint (X Y : ℝ × ℝ) : ℝ × ℝ :=
  ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- The lines connecting the midpoints of opposite sides of ABCD and KLMN intersect at the same point
theorem midpoints_intersect_at_centroids (A B C D : ℝ × ℝ) :
  convex_quadrilateral A B C D →
  let G1 := centroid A B C,
      G2 := centroid B C D,
      G3 := centroid D A B,
      G4 := centroid C D A,
      (K, L, M, N) := quadrilateral_KLMN A B C D in
  ∃ P, P = midpoint (midpoint A B) (midpoint C D) ∧ P = midpoint (midpoint B C) (midpoint D A) ∧
       P = midpoint (midpoint G1 G2) (midpoint G3 G4) ∧ P = midpoint (midpoint G2 G3) (midpoint G4 G1) :=
by
  sorry

end midpoints_intersect_at_centroids_l332_332517


namespace set_intersection_l332_332564

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def complement (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem set_intersection (hU : U = univ)
                         (hA : A = {x : ℝ | x > 0})
                         (hB : B = {x : ℝ | x > 1}) :
  A ∩ complement B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end set_intersection_l332_332564


namespace sum_of_solutions_l332_332952
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332952


namespace part_1_solution_set_part_2_range_of_a_l332_332015

theorem part_1_solution_set (x : ℝ) :
  let f (x : ℝ) := |2 * x + 1| + |2 * x - 3|
  in f x ≤ 6 ↔ -1 ≤ x ∧ x ≤ 2 :=
by
  sorry

theorem part_2_range_of_a (a : ℝ) :
  let f (x : ℝ) := |2 * x + 1| + |2 * x - 3|
  in (∃ x : ℝ, f x < |a - 1|) ↔ a > 5 ∨ a < -3 :=
by
  sorry

end part_1_solution_set_part_2_range_of_a_l332_332015


namespace smaller_angle_3_40_l332_332695

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l332_332695


namespace clock_angle_3_40_l332_332713

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332713


namespace percentage_of_b_giving_6_l332_332317

-- Define the conditions
variables (a b c : ℝ)

-- 8 is 6% of a
def condition1 := (8 = (6 / 100) * a)

-- c equals b / a
def condition2 := (c = b / a)

-- The question can now be formulated: What is the percentage of b that gives 6?
theorem percentage_of_b_giving_6 (h1 : condition1) (h2 : condition2) : (6 = (1 / 100) * b) :=
sorry

end percentage_of_b_giving_6_l332_332317


namespace clock_angle_3_40_l332_332672

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332672


namespace sum_partial_fractions_l332_332981

theorem sum_partial_fractions :
  (∑ n in Finset.range 12, (1 / ((n + 1) * (n + 2)))) = 12 / 13 :=
by
  sorry

end sum_partial_fractions_l332_332981


namespace small_angle_at_3_40_is_130_degrees_l332_332759

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332759


namespace sum_of_solutions_l332_332953
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332953


namespace solution_set_of_inequality_l332_332231

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 2 * x > 0} = {x : ℝ | x ∈ (-∞, 0) ∪ (2, +∞)} :=
sorry

end solution_set_of_inequality_l332_332231


namespace sum_of_solutions_sum_of_solutions_is_16_l332_332887

theorem sum_of_solutions (x : ℝ) (h : (x - 8)^2 = 49) : x = 15 ∨ x = 1 := sorry

theorem sum_of_solutions_is_16 : ∑ (x : ℝ) in { x : ℝ | (x - 8)^2 = 49 }.to_finset, x = 16 := sorry

end sum_of_solutions_sum_of_solutions_is_16_l332_332887


namespace g_evaluation_l332_332136

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then (2 * a * b - a + 3) / (3 * a)
  else (a * b - b - 1) / (-3 * b)

theorem g_evaluation : g 2 1 + g 2 4 = 7 / 12 := 
by {
  sorry
}

end g_evaluation_l332_332136


namespace middle_number_is_9_point_5_l332_332241

theorem middle_number_is_9_point_5 (x y z : ℝ) 
  (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 22) : y = 9.5 := 
by {
  sorry
}

end middle_number_is_9_point_5_l332_332241


namespace position_of_YZPRQ_l332_332659

def alphabet_order_position : list char := ['P', 'Q', 'R', 'Y', 'Z']
def target_word : list char := ['Y', 'Z', 'P', 'R', 'Q']

theorem position_of_YZPRQ : 
  ∀ (L : list char) (target : list char), 
  L = alphabet_order_position → 
  target = target_word → 
  list.position (list.permutations L) target = 67 :=
by
  intros,
  rw [a_1, a_2],
  sorry

end position_of_YZPRQ_l332_332659


namespace congruent_incenter_circumcenter_triangles_l332_332229

theorem congruent_incenter_circumcenter_triangles 
  (A B C A₁ B₁ C₁ I_AB₁C₁ I_BC₁A₁ I_CA₁B₁ O_AB₁C₁ O_BC₁A₁ O_CA₁B₁ : Point)
  (h1 : mid_point A₁ B C)
  (h2 : mid_point B₁ A C)
  (h3 : mid_point C₁ A B)
  (h_inc1 : incenter I_AB₁C₁ (triangle A B₁ C₁))
  (h_inc2 : incenter I_BC₁A₁ (triangle B C₁ A₁))
  (h_inc3 : incenter I_CA₁B₁ (triangle C A₁ B₁))
  (h_circ1 : circumcenter O_AB₁C₁ (triangle A B₁ C₁))
  (h_circ2 : circumcenter O_BC₁A₁ (triangle B C₁ A₁))
  (h_circ3 : circumcenter O_CA₁B₁ (triangle C A₁ B₁)) :
  congruent (triangle I_AB₁C₁ I_BC₁A₁ I_CA₁B₁) (triangle O_AB₁C₁ O_BC₁A₁ O_CA₁B₁) :=
sorry

end congruent_incenter_circumcenter_triangles_l332_332229


namespace milk_butterfat_percentage_l332_332027

theorem milk_butterfat_percentage :
  ∃ V x, 8 * 0.45 + V * (x / 100) = (8 + V) * 0.20 ∧ 8 + V = 20 ∧ x = 3.33 := 
by
  use 12, 3.33
  split
  { norm_num }
  split 
  { norm_num }
  { norm_num }

end milk_butterfat_percentage_l332_332027


namespace intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l332_332620

variable (a b : ℝ)

-- Define the equations given in the problem
def line1 := ∀ y : ℝ, 3 = (1/3) * y + a
def line2 := ∀ x : ℝ, 3 = (1/3) * x + b

-- The Lean statement for the proof
theorem intersecting_lines_at_3_3_implies_a_plus_b_eq_4 :
  (line1 3) ∧ (line2 3) → a + b = 4 :=
by 
  sorry

end intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l332_332620


namespace area_hexagon_half_triangle_l332_332997

-- Define the points in the acute triangle and the midpoints
variables {A B C A1 B1 C1 A2 B2 C2 : Type*}

-- Assume points form an acute triangle
axiom acute_triangle (h : ∃ (A B C : Type*), true) : ∃ (A B C : Type*), true

-- Assume A1, B1, C1 are the midpoints of sides of triangle ABC
axiom midpoint_def : ∃ (A1 B1 C1 : Type*), 
  (true)

-- Assume perpendiculars from A1, B1, C1 to the other two sides of triangle meet the sides at A2, B2, C2
axiom perpendiculars_def : ∃ (A2 B2 C2 : Type*), 
  true

-- The theorem to prove
theorem area_hexagon_half_triangle (h₁ : ∃ (A B C : Type*), true) 
  (h₂ : ∃ (A1 B1 C1 : Type*), true)
  (h₃ : ∃ (A2 B2 C2 : Type*), true) : 
  ∃ (hexagon_area triangle_area : Type*), hexagon_area = (1 / 2) * triangle_area :=
by sorry

end area_hexagon_half_triangle_l332_332997


namespace clock_angle_3_40_l332_332719

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l332_332719


namespace unique_functional_equation_solution_l332_332855

theorem unique_functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end unique_functional_equation_solution_l332_332855


namespace clock_angle_3_40_l332_332677

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332677


namespace sum_of_solutions_eq_16_l332_332925

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332925


namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l332_332266

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l332_332266


namespace triangle_ABC_area_l332_332513

-- Define the conditions
def right_triangle_ABC (A B C : Type) [AddGroup A] [AddGroup B] [AddGroup C] 
  (a b c : ℝ) :=
  (angle B = 90) ∧  -- angle B is a right angle
  (AD = 5) ∧        -- AD = 5 cm
  (DC = 3) ∧        -- DC = 3 cm
  (D is foot of altitude from B onto AC)

-- The theorem to prove
theorem triangle_ABC_area (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] 
  (a b c ac ad bd : ℝ) 
  (h₁ : right_triangle_ABC A B C a b c) 
  (h₂ : AC = ad + dc) 
  (h₃ : BD = sqrt (ad * dc)) 
  : Area ABC = 4 * sqrt 15 := 
sorry

end triangle_ABC_area_l332_332513


namespace ratio_of_areas_pyramid_l332_332120

theorem ratio_of_areas_pyramid (A B C D M L K P Q : Type) 
  (area_tr_ABD area_tr_ABC : ℝ) 
  (h1 : area_tr_ABC = 4 * area_tr_ABD)
  (h2 : (CM CD : ℝ), CM / CD = 2 / 3) 
  (S : ℝ) :
  let area_tr_LKM := (1 / 3) ^ 2 * area_tr_ABC,
      area_tr_QPM := (2 / 3) ^ 2 * area_tr_ABD in
  area_tr_LKM = area_tr_QPM :=
by
  sorry

end ratio_of_areas_pyramid_l332_332120


namespace proof_vectors_equal_norms_l332_332428

def vectors_a_b (a b : ℝ^3) : Prop :=
  ∃ a b : ℝ^3, (a ≠ 0 ∧ b ≠ 0) ∧ (inner (a + b) (a - b) = 0 → norm a = norm b)

theorem proof_vectors_equal_norms (a b : V)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : inner (a + b) (a - b) = 0) :
  norm a = norm b := 
sorry

end proof_vectors_equal_norms_l332_332428


namespace roots_of_polynomial_l332_332391

theorem roots_of_polynomial :
  (3 * (2 + Real.sqrt 3)^4 - 19 * (2 + Real.sqrt 3)^3 + 34 * (2 + Real.sqrt 3)^2 - 19 * (2 + Real.sqrt 3) + 3 = 0) ∧ 
  (3 * (2 - Real.sqrt 3)^4 - 19 * (2 - Real.sqrt 3)^3 + 34 * (2 - Real.sqrt 3)^2 - 19 * (2 - Real.sqrt 3) + 3 = 0) ∧
  (3 * ((7 + Real.sqrt 13) / 6)^4 - 19 * ((7 + Real.sqrt 13) / 6)^3 + 34 * ((7 + Real.sqrt 13) / 6)^2 - 19 * ((7 + Real.sqrt 13) / 6) + 3 = 0) ∧
  (3 * ((7 - Real.sqrt 13) / 6)^4 - 19 * ((7 - Real.sqrt 13) / 6)^3 + 34 * ((7 - Real.sqrt 13) / 6)^2 - 19 * ((7 - Real.sqrt 13) / 6) + 3 = 0) :=
by sorry

end roots_of_polynomial_l332_332391


namespace clock_angle_3_40_l332_332680

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332680


namespace equal_real_roots_implies_k_zero_l332_332051

-- Define the quadratic equation
def quadratic_eq (k : ℝ) : ℝ → ℝ := λ x, x^2 - 2 * x - k + 1

-- The theorem statement
theorem equal_real_roots_implies_k_zero (k : ℝ) : 
  (∃ x : ℝ, quadratic_eq k x = 0 ∧ ∀ y : ℝ, quadratic_eq k y = 0 → y = x) → k = 0 :=
by
  -- Proof is omitted.
  sorry

end equal_real_roots_implies_k_zero_l332_332051


namespace twelfth_day_is_monday_l332_332068

def Month := ℕ
def Day := ℕ

-- Definitions for days of the week, where 0 represents Monday, 1 represents Tuesday, etc.
inductive Weekday : Type
| Monday : Weekday
| Tuesday : Weekday
| Wednesday : Weekday
| Thursday : Weekday
| Friday : Weekday
| Saturday : Weekday
| Sunday : Weekday

open Weekday

-- A month has exactly 5 Fridays
def has_five_fridays (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday ∧
    (∃ last_day: Weekday, last_day ≠ Friday ∧ 
      let fridays := List.filter (λ d, d = Friday) (List.range 31) in
      fridays.length = 5)

-- The first day of the month is not a Friday
def first_day_not_friday (month: Month): Prop :=
  ∃ starts_on: Weekday, starts_on ≠ Friday

-- The last day of the month is not a Friday
def last_day_not_friday (month: Month): Prop :=
  ∀ last_day: Weekday, last_day = (29 % 7) → last_day ≠ Friday

-- Combining the conditions for the problem
def valid_month (month: Month): Prop :=
  has_five_fridays(month) ∧ first_day_not_friday(month) ∧ last_day_not_friday(month)

-- Prove that the 12th day of the month is a Monday given the conditions
theorem twelfth_day_is_monday (month: Month) (h: valid_month(month)): (∃ starts_on: Weekday, starts_on + 11 = Monday) :=
sorry

end twelfth_day_is_monday_l332_332068


namespace allan_balloons_l332_332807

theorem allan_balloons :
  ∀ (initial_allan initial_jake final_total balloons_bought : ℕ),
  initial_allan = 3 →
  initial_jake = 5 →
  final_total = 10 →
  balloons_bought = final_total - (initial_allan + initial_jake) →
  balloons_bought = 2 :=
by
  intros initial_allan initial_jake final_total balloons_bought
  assume h1 : initial_allan = 3
  assume h2 : initial_jake = 5
  assume h3 : final_total = 10
  assume h4 : balloons_bought = final_total - (initial_allan + initial_jake)
  rw [h1, h2, h3] at h4 
  norm_num at h4
  exact h4

end allan_balloons_l332_332807


namespace simplest_quadratic_surds_count_l332_332346

theorem simplest_quadratic_surds_count (x y : ℝ) : 
  (set.count { (sqrt (1/2)), (sqrt 12), (sqrt 30), (sqrt (x + 2)), (sqrt (40 * x^2)), (sqrt (x^2 + y^2)) }.filter (λ s, ¬ ∃ a b, s = a * sqrt b ∧ a ≠ 1 ∧ ∃ c d, b = c * d ∧ c = sqrt d ∧ d ≠ 1)) = 3 :=
sorry

end simplest_quadratic_surds_count_l332_332346


namespace calvin_total_insects_l332_332823

-- Definitions based on the conditions
def roaches := 12
def scorpions := 3
def crickets := roaches / 2
def caterpillars := scorpions * 2

-- Statement of the problem
theorem calvin_total_insects : 
  roaches + scorpions + crickets + caterpillars = 27 :=
  by
    sorry

end calvin_total_insects_l332_332823


namespace sum_of_solutions_l332_332958
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332958


namespace no_valid_starting_day_for_equal_tuesdays_and_fridays_l332_332332

theorem no_valid_starting_day_for_equal_tuesdays_and_fridays (d : ℕ) (h : 1 ≤ d ∧ d ≤ 7) :
  ∃ k : ℕ, k = 0 ↔ ∀ i : ℕ, i ∈ {0, 1, 2, 3, 4, 5, 6} → ∀ tuesdays fridays : ℕ, 
  tuesdays ≠ fridays := sorry

end no_valid_starting_day_for_equal_tuesdays_and_fridays_l332_332332


namespace simplify_and_evaluate_expression_l332_332599

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) : 
  ( ( x + 3 ) / ( x^2 - 2*x + 1 ) * ( x - 1 ) / ( x^2 + 3*x ) + 1 / x ) = Real.sqrt 3 / 3 :=
by
  rw h
  sorry

end simplify_and_evaluate_expression_l332_332599


namespace sum_of_first_10_terms_l332_332234

def sequence_a (n : ℕ) : ℕ
| 0       => 0
| (n + 1) => if n = 0 then 2 else 2 * sequence_a n - 1

def S_n (n : ℕ) : ℕ :=
(n + 1) * (sequence_a n + 1)

theorem sum_of_first_10_terms : S_n 10 = 1033 := by
  sorry

end sum_of_first_10_terms_l332_332234


namespace twelfth_is_monday_l332_332071

def days_of_week : Type := {d // d < 7}

def starts_on_friday (d : ℕ) : days_of_week := ⟨d % 7, by linarith [Nat.mod_lt d (by norm_num)]⟩

-- Condition 1: There are exactly 5 Fridays in the month (which has at least 30 days)
def has_five_fridays (days_in_month : ℕ) : Prop := 
  ∃ (start : ℕ), 
    (start % 7 ≠ 5) ∧ 
    (days_in_month > 28 ∧ days_in_month < 32) ∧ -- At least 30 days to have 5 Fridays
    ∃ f, ∀ i, starts_on_friday(start + 7*i) = f → (i < 5)

-- Condition 2: The first day of the month is not a Friday
def first_not_friday (start : ℕ) : Prop := start % 7 ≠ 5

-- Condition 3: The last day of the month is not a Friday
def last_not_friday (start days_in_month : ℕ) : Prop := (start + days_in_month - 1) % 7 ≠ 5
    
theorem twelfth_is_monday (days_in_month : ℕ) (start : ℕ) 
  (h_five_fridays : has_five_fridays days_in_month)
  (h_first_not_friday : first_not_friday start)
  (h_last_not_friday : last_not_friday start days_in_month) : 
  (start + 11) % 7 = 1 :=
sorry

end twelfth_is_monday_l332_332071


namespace chickens_bushels_per_day_l332_332841

def totalBushelsPerDay (cows sheep chickens : ℕ) (bushelsPerCowSheep : ℕ) (totalBushels : ℕ) : ℕ :=
  totalBushels - ((cows + sheep) * bushelsPerCowSheep)

theorem chickens_bushels_per_day :
  ∀ (cows sheep chickens bushelsPerCowSheep totalBushels : ℕ), 
    cows = 4 → sheep = 3 → bushelsPerCowSheep = 2 →
    chickens = 7 → totalBushels = 35 →
    totalBushelsPerDay cows sheep chickens bushelsPerCowSheep totalBushels / chickens = 3 :=
by
  intros
  rw [totalBushelsPerDay, Nat.sub, Nat.add, Nat.mul, Nat.div]
  sorry

end chickens_bushels_per_day_l332_332841


namespace dennis_total_cost_l332_332850

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end dennis_total_cost_l332_332850


namespace smaller_angle_at_3_40_l332_332735

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332735


namespace min_value_expression_l332_332388

theorem min_value_expression :
  ∃ x y : ℝ, (sqrt (2 * (1 + cos (2 * x))) - sqrt (9 - sqrt 7) * sin x + 1) * (3 + 2 * sqrt (13 - sqrt 7) * cos y - cos (2 * y)) = -19 :=
by sorry

end min_value_expression_l332_332388


namespace smaller_angle_3_40_pm_l332_332748

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332748


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l332_332263

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l332_332263


namespace sin_lower_bound_lt_l332_332039

theorem sin_lower_bound_lt (a : ℝ) (h : ∃ x : ℝ, Real.sin x < a) : a > -1 :=
sorry

end sin_lower_bound_lt_l332_332039


namespace chess_games_points_l332_332270

theorem chess_games_points (x : ℕ) (n : ℕ) (total_games : ℕ) (points_for_win : ℕ) (total_points : ℕ) :
  total_games = 100 ∧ points_for_win = 11 ∧ total_points = 800 ∧  (11 - x) * n = 800 - 100 * x 
  → x = 3 ∨ x = 4 :=
begin
  -- Proof is omitted with 'sorry'
  sorry
end

end chess_games_points_l332_332270


namespace option_d_is_correct_l332_332766

theorem option_d_is_correct : sqrt 32 / sqrt 2 = 4 :=
by
  sorry

end option_d_is_correct_l332_332766


namespace smaller_angle_between_hands_at_3_40_l332_332682

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332682


namespace Faye_total_pencils_l332_332376

def pencils_per_row : ℕ := 8
def number_of_rows : ℕ := 4
def total_pencils : ℕ := pencils_per_row * number_of_rows

theorem Faye_total_pencils : total_pencils = 32 := by
  sorry

end Faye_total_pencils_l332_332376


namespace sum_of_solutions_eq_16_l332_332971

theorem sum_of_solutions_eq_16 :
  let solutions := {x : ℝ | (x - 8) ^ 2 = 49} in
  ∑ x in solutions, x = 16 := by
  sorry

end sum_of_solutions_eq_16_l332_332971


namespace sum_of_solutions_l332_332896

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332896


namespace probability_is_one_third_l332_332333

noncomputable def probabilityRedGreaterThanBlueButLessThanThreeTimes : ℝ :=
  let areaRegion := (∫ x in 0..2, (min (3 * x) 2) - x) -- Region area under constraints
  let totalArea := 4                                      -- Total possible area as square is 2x2
  areaRegion / totalArea

theorem probability_is_one_third :
  probabilityRedGreaterThanBlueButLessThanThreeTimes = 1 / 3 :=
by
  sorry

end probability_is_one_third_l332_332333


namespace average_weight_increase_l332_332611

theorem average_weight_increase (A : ℝ) :
  let total_weight_initial := 8 * A in
  let weight_removed := 55 in
  let weight_added := 75 in
  let increase_in_weight := (weight_added - weight_removed) in
  let total_weight_new := total_weight_initial + increase_in_weight in
  let average_weight_new := total_weight_new / 8 in
  let increase_in_average_weight := average_weight_new - A in
  increase_in_average_weight = 2.5 :=
by
  let total_weight_initial := 8 * A
  let weight_removed := 55
  let weight_added := 75
  let increase_in_weight := (weight_added - weight_removed)
  let total_weight_new := total_weight_initial + increase_in_weight
  let average_weight_new := total_weight_new / 8
  let increase_in_average_weight := average_weight_new - A
  sorry

end average_weight_increase_l332_332611


namespace paint_mixture_replacement_l332_332646

theorem paint_mixture_replacement :
  ∃ x y : ℝ,
    (0.5 * (1 - x) + 0.35 * x = 0.45) ∧
    (0.6 * (1 - y) + 0.45 * y = 0.55) ∧
    (x = 1 / 3) ∧
    (y = 1 / 3) :=
sorry

end paint_mixture_replacement_l332_332646


namespace least_positive_multiple_of_15_with_digit_product_multiple_of_15_l332_332283

theorem least_positive_multiple_of_15_with_digit_product_multiple_of_15 : 
  ∃ (n : ℕ), 
    n % 15 = 0 ∧ 
    (∀ k, k % 15 = 0 ∧ (∃ m : ℕ, m < n ∧ m % 15 = 0 ∧ 
    list.prod (nat.digits 10 m) % 15 == 0) 
    → list.prod (nat.digits 10 k) % 15 == 0) 
    ∧ list.prod (nat.digits 10 n) % 15 = 0 
    ∧ n = 315 :=
sorry

end least_positive_multiple_of_15_with_digit_product_multiple_of_15_l332_332283


namespace probability_line_not_passing_through_third_quadrant_l332_332403

def line_does_not_pass_through_third_quadrant (k b : ℤ) : Prop :=
  k < 0 ∧ b > 0

theorem probability_line_not_passing_through_third_quadrant :
  let A := {-1, 1, 2}
  let B := {-2, 1, 2}
  (∑ (k : ℤ) in A, ∑ (b : ℤ) in B, if line_does_not_pass_through_third_quadrant k b then 1 else 0)
  / (A.card * B.card) = 2 / 9 :=
by
  -- Proof goes here
  sorry

end probability_line_not_passing_through_third_quadrant_l332_332403


namespace at_least_one_not_solved_l332_332331

theorem at_least_one_not_solved (p q : Prop) : (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by sorry

end at_least_one_not_solved_l332_332331


namespace derivative_at_e_ln_l332_332409

-- Definition of the function
def f (x : ℝ) := Real.log x

-- Theorem statement
theorem derivative_at_e_ln (x : ℝ) (h : x = Real.exp 1) : Real.deriv f x = 1 / x :=
by
  sorry

end derivative_at_e_ln_l332_332409


namespace cubic_polynomial_real_root_l332_332389

theorem cubic_polynomial_real_root (a b : ℝ) 
  (h₁ : -13 * a - 4 * b = 15) 
  (h₂ : -2 * a - 2 * b = 0) :
  let x := (21 : ℝ) / 5 in 
  a = -5 / 3 → b = 5 / 3 → 
  a * x^3 + 4 * x^2 + b * x - 35 = 0 :=
by 
  intros ha hb
  simp [ha, hb]
  sorry

end cubic_polynomial_real_root_l332_332389


namespace largest_integer_k_divides_factorial_l332_332831

open BigOperators

def largest_power_dividing_factorial (n p : ℕ) : ℕ :=
  (∑ i in finset.range (n.log p + 1), n / p^i)

theorem largest_integer_k_divides_factorial (k : ℕ) :
  2520 = 2^3 * 3^2 * 5 * 7 →
  k = largest_power_dividing_factorial 2520 7 →
  k = 418 :=
by
  intros h_fac h_k
  -- proof goes here
  sorry

end largest_integer_k_divides_factorial_l332_332831


namespace find_slope_l332_332165

theorem find_slope (m : ℝ) : 
    (∀ x : ℝ, (2, 13) = (x, 5 * x + 3)) → 
    (∀ x : ℝ, (2, 13) = (x, m * x + 1)) → 
    m = 6 :=
by 
  intros hP hQ
  have h_inter_p := hP 2
  have h_inter_q := hQ 2
  simp at h_inter_p h_inter_q
  have : 13 = 5 * 2 + 3 := h_inter_p
  have : 13 = m * 2 + 1 := h_inter_q
  linarith

end find_slope_l332_332165


namespace basketball_team_combinations_l332_332319

/-
  Problem:
  A basketball team has 12 players. The coach needs to select a team captain and then choose 5 players for the starting lineup (excluding the captain). Prove that the number of different combinations the coach can form is 5544.
-/

theorem basketball_team_combinations : ∃ (n : ℕ), n = 12 * (Nat.choose 11 5) ∧ n = 5544 :=
by
  sorry

end basketball_team_combinations_l332_332319


namespace smaller_angle_at_3_40_l332_332740

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332740


namespace clock_angle_l332_332709

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332709


namespace imaginary_part_z1_mul_z2_l332_332559

noncomputable def z1 : ℂ := 2 - 1 * complex.I
noncomputable def z2 : ℂ := 1 - 2 * complex.I

theorem imaginary_part_z1_mul_z2 : complex.im (z1 * z2) = -5 := by
  sorry

end imaginary_part_z1_mul_z2_l332_332559


namespace sum_of_roots_eq_seventeen_l332_332915

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332915


namespace expression_multiple_l332_332989

theorem expression_multiple :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  (a - b) / (1/78) = 13 :=
by
  sorry

end expression_multiple_l332_332989


namespace p_eq_2q_l332_332613

-- Definitions based on conditions
variables (A B C D E : Type)
variables (p q : ℝ) -- Angles in degrees

-- Assume the conditions
variable [triangle_ABC : Triangle ABC]  -- Triangle ABC with side BA extended to point E
variable [A_Bisector_BC : AngleBisector (Angle ABC) D] -- The bisector of angle ABC meets the bisector of angle EAC at D
variable [E_Bisector_AC : AngleBisector (Angle EAC) D]
variable (∠BCA : Angle) (∠BCA = p) -- Angle BCA equals p degrees
variable (∠BDA : Angle) (∠BDA = q) -- Angle BDA equals q degrees

-- Statement to prove
theorem p_eq_2q : p = 2 * q :=
by sorry

end p_eq_2q_l332_332613


namespace period_cosine_l332_332859

noncomputable def period_of_cosine_function : ℝ := 2 * Real.pi / 3

theorem period_cosine (x : ℝ) : ∃ T, ∀ x, Real.cos (3 * x - Real.pi) = Real.cos (3 * (x + T) - Real.pi) :=
  ⟨period_of_cosine_function, by sorry⟩

end period_cosine_l332_332859


namespace C1_C2_properties_l332_332017

-- Definitions based on the conditions
def C1_parametric (a b θ : ℝ) : ℝ × ℝ := (a * Real.cos θ, b * Real.sin θ)
def C1_equation (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def C2_equation (r x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Main theorem 
theorem C1_C2_properties (a b r x y θ : ℝ) (ha_gt_hb : a > b) (hb_gt_zero : b > 0) (hr_gt_zero : r > 0) :
  (C1_equation a b (a * Real.cos θ) (b * Real.sin θ)) ∧
  (C2_equation r x y) ∧
  ((r = a ∨ r = b) → ∃ p1 p2 : ℝ × ℝ, C1_equation a b p1.1 p1.2 ∧ C2_equation r p1.1 p1.2 ∧ C1_equation a b p2.1 p2.2 ∧ C2_equation r p2.1 p2.2) ∧
  ((b < r ∧ r < a) → ∃ p1 p2 p3 p4 : ℝ × ℝ, C1_equation a b p1.1 p1.2 ∧ C2_equation r p1.1 p1.2 ∧ 
                                                C1_equation a b p2.1 p2.2 ∧ C2_equation r p2.1 p2.2 ∧ 
                                                C1_equation a b p3.1 p3.2 ∧ C2_equation r p3.1 p3.2 ∧ 
                                                C1_equation a b p4.1 p4.2 ∧ C2_equation r p4.1 p4.2) 
∧ ((0 < r ∧ r < b) ∨ (r > a) → ∀ (p : ℝ × ℝ), ¬ (C1_equation a b p.1 p.2 ∧ C2_equation r p.1 p.2)) ∧
  ((b < r) ∧ (r < a) → ∃ (θ : ℝ), 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ (2 * a * b * Real.sin (2 * θ) = 2 * a * b)) :=
by
  sorry

end C1_C2_properties_l332_332017


namespace total_number_of_fish_l332_332250

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l332_332250


namespace num_real_a_with_even_root_and_abs_x0_lt_1000_l332_332857

theorem num_real_a_with_even_root_and_abs_x0_lt_1000 :
  let evens := {n : ℤ | n % 2 = 0 ∧ |n| < 1000},
  {a : ℝ | ∃ x0 ∈ evens, x0^3 = a * x0 + a + 1}.infinite :=
by sorry

end num_real_a_with_even_root_and_abs_x0_lt_1000_l332_332857


namespace shortest_chord_line_through_point_l332_332792

theorem shortest_chord_line_through_point (l : ℝ → ℝ → Prop)
    (passes_through : l 1 1)
    (intersects_circle : ∀ x y, l x y → x^2 + (y-2)^2 = 5)
    (shortest_chord : ∀ k, l = λ x y, y = k * x + (1 - k))
  : ∀ x y, l x y ↔ x = y :=
by
  sorry

end shortest_chord_line_through_point_l332_332792


namespace clock_angle_3_40_l332_332671

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332671


namespace equality_of_expressions_l332_332497

theorem equality_of_expressions (a b c : ℝ) (h : a = b + c + 2) : 
  a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1 :=
by sorry

end equality_of_expressions_l332_332497


namespace segment_radius_with_inscribed_equilateral_triangle_l332_332103

theorem segment_radius_with_inscribed_equilateral_triangle (α h : ℝ) : 
  ∃ x : ℝ, x = (h / (Real.sin (α / 2))^2) * (Real.cos (α / 2) + Real.sqrt (1 + (1 / 3) * (Real.sin (α / 2))^2)) :=
sorry

end segment_radius_with_inscribed_equilateral_triangle_l332_332103


namespace maximal_matching_iff_no_augmenting_path_l332_332183

variable {V : Type} [DecidableEq V]
variable {G : Type} [Graph G] {E : Type} [Edge E V]

-- Define a matching
def is_matching (M : set E) : Prop := 
  ∀ e1 e2 : E, e1 ≠ e2 ∧ e1 ∈ M ∧ e2 ∈ M → ¬ (incident e1 e2)

-- Define a maximal matching
def is_maximal_matching (M : set E) : Prop :=
  is_matching M ∧ ∀ (M' : set E), M ⊆ M' ∧ is_matching M' → M = M'

-- Define the path conditions
def augmenting_path (G : Graph G) (M : set E) (p : list V) : Prop :=
  (∀ i : ℕ, i < p.length - 1 →
    (i % 2 = 0 ∧ ∃ e : E, incident e (p.nth i) (p.nth (i + 1)) ∧ e ∉ M) ∨
    (i % 2 = 1 ∧ ∃ e : E, incident e (p.nth i) (p.nth (i + 1)) ∧ e ∈ M)) ∧
  ¬ incident (p.nth 0) M ∧ ¬ incident (p.nth (p.length - 1)) M ∧
  p.head ≠ p.last

theorem maximal_matching_iff_no_augmenting_path (M : set E) : 
  is_maximal_matching M ↔
    ∀ (p : list V), ¬ augmenting_path G M p :=
sorry

end maximal_matching_iff_no_augmenting_path_l332_332183


namespace sum_of_solutions_l332_332899

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l332_332899


namespace waiter_serves_meals_l332_332983

theorem waiter_serves_meals :
  let friends := ["F1", "F2", "F3", "F4"]
  let orders := ["P", "P", "B", "B"]
  let serving_ways (l : List String) := l.permutations
  let correct_serving (l : List String) := (List.zip friends l).filter (λ (x : String × String), x.1 == x.2)
  let valid_serving_count := serving_ways orders |>.count (λ l, correct_serving l |> List.length = 2)
  valid_serving_count = 2 :=
by
  -- The proof is omitted
  sorry

end waiter_serves_meals_l332_332983


namespace small_angle_at_3_40_is_130_degrees_l332_332754

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l332_332754


namespace sum_of_solutions_l332_332956
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332956


namespace denny_followers_l332_332854

theorem denny_followers (initial_followers: ℕ) (new_followers_per_day: ℕ) (unfollowers_in_year: ℕ) (days_in_year: ℕ)
  (h_initial: initial_followers = 100000)
  (h_new_per_day: new_followers_per_day = 1000)
  (h_unfollowers: unfollowers_in_year = 20000)
  (h_days: days_in_year = 365):
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers_in_year = 445000 :=
by
  sorry

end denny_followers_l332_332854


namespace proof_problem_l332_332131

-- Define the conditions and statements as Lean definitions and theorems

structure Triangle (V : Type) [InnerProductSpace ℝ V] := 
(A B C : V)

variables {V : Type} [InnerProductSpace ℝ V] (T : Triangle V)

noncomputable def AC := T.C - T.A
noncomputable def AB := T.B - T.A
noncomputable def BC := T.C - T.B

def given_conditions : Prop :=
  (∥AC T∥ = 2 * sqrt 3) ∧
  (⟪BC T, ⟪AB T, BC T⟫⟩ / ∥BC T∥ + ⟪AB T, ⟪BC T, AB T⟫⟩ / ∥AB T∥ = ∥AC T∥ * (sqrt (1 - (⟪AB T, AC T⟫ ^ 2 / ∥AC T∥ ^ 2) / ∥AB T∥ ^ 2)))

noncomputable def angle_B : ℝ :=
  real.arccos (⟪AB T, BC T⟫ / (∥AB T∥ * ∥BC T∥))

noncomputable def area_ABC : ℝ :=
  1/2 * ∥AC T∥ * ∥BC T∥ * real.sin (angle_B T)

theorem proof_problem :
  given_conditions T →
  (angle_B T = 2 * π / 3) ∧ (area_ABC T = sqrt 3) :=
by
  -- Inserting proof steps here will be required to complete this theorem
  sorry

end proof_problem_l332_332131


namespace find_x2_y2_l332_332043

theorem find_x2_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (10344 / 169) := by
  sorry

end find_x2_y2_l332_332043


namespace sum_of_solutions_eq_16_l332_332929

theorem sum_of_solutions_eq_16 : ∀ x : ℝ, (x - 8) ^ 2 = 49 → x ∈ {15, 1} → 15 + 1 = 16 :=
by {
  intro x,
  intro h,
  intro hx,
  sorry
}

end sum_of_solutions_eq_16_l332_332929


namespace lattice_lines_l332_332174

noncomputable def M: set ℝ → Prop := λ l, ∃ p ∈ ℤ^2, ∀ p' ∈ ℤ^2, p ≠ p' → ¬ l passesThrough(p')
noncomputable def N: set ℝ → Prop := λ l, ∀ p ∈ ℤ^2, ¬ l passesThrough(p)
noncomputable def P: set ℝ → Prop := λ l, ∀ p ∈ ℤ^2, ∃ p' ∈ ℤ^2, p ≠ p' ∧ l passesThrough(p')

theorem lattice_lines:
  (∃ l, M l) ∧
  (∃ l, N l) ∧
  (∃ l, P l) ∧
  (∀ l, l passesThrough at least one <=> l ∈ M ∪ N ∪ P) :=
  sorry

end lattice_lines_l332_332174


namespace pure_imaginary_when_m_eq_3_eq_3_plus_6i_when_m_eq_6_in_fourth_quadrant_when_0_lt_m_lt_3_l332_332399

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

noncomputable def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem pure_imaginary_when_m_eq_3 (m : ℝ) :
  (m^2 - 8 * m + 15 = 0) → 
  (m^2 - 5 * m ≠ 0) →
  m = 3 →
  let z := complex.mk 0 (m^2 - 5 * m) in 
  is_pure_imaginary z :=
by
  sorry

theorem eq_3_plus_6i_when_m_eq_6 (m : ℝ) :
  (m^2 - 8 * m + 15 = 3) →
  (m^2 - 5 * m = 6) →
  m = 6 →
  let z := complex.mk 3 6 in 
  z = (complex.mk (m^2 - 8 * m + 15) (m^2 - 5 * m)) :=
by
  sorry

theorem in_fourth_quadrant_when_0_lt_m_lt_3 (m : ℝ) :
  (m^2 - 8 * m + 15 > 0) →
  (m^2 - 5 * m < 0) →
  (0 < m ∧ m < 3) →
  is_in_fourth_quadrant (m^2 - 8 * m + 15) (m^2 - 5 * m) :=
by
  sorry

end pure_imaginary_when_m_eq_3_eq_3_plus_6i_when_m_eq_6_in_fourth_quadrant_when_0_lt_m_lt_3_l332_332399


namespace sum_of_solutions_l332_332954
-- Import the necessary library

-- Define the problem conditions in Lean 4
def equation_condition (x : ℝ) : Prop :=
  (x - 8)^2 = 49

-- State the problem as a theorem in Lean 4
theorem sum_of_solutions :
  (∑ x in { x : ℝ | equation_condition x }, id x) = 16 :=
by
  sorry

end sum_of_solutions_l332_332954


namespace sum_of_solutions_eqn_l332_332948

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332948


namespace yoojung_notebooks_l332_332301

theorem yoojung_notebooks (N : ℕ) (h : (N - 5) / 2 = 4) : N = 13 :=
by
  sorry

end yoojung_notebooks_l332_332301


namespace least_number_exists_least_number_is_2187_l332_332305

-- Definition of conditions
def n := 2187

-- Lean statements formulating the problem

theorem least_number_exists :
  ∃ n, (n % 56 = 3) ∧ (n % 78 = 3) ∧ (n % 9 = 0) :=
by {
  use 2187,
  split,
  exact Nat.mod_eq_of_lt (by norm_num),
  split,
  exact Nat.mod_eq_of_lt (by norm_num),
  exact Nat.mod_eq_of_lt (by norm_num)
}

-- Proving the exact least number
theorem least_number_is_2187 :
  (∃ m, (m % 56 = 3) ∧ (m % 78 = 3) ∧ (m % 9 = 0)) ∧ (∀n, (n % 56 = 3) ∧ (n % 78 = 3) ∧ (n % 9 = 0) → n ≥ 2187) :=
by {
  split,
  use 2187,
  split,
  exact Nat.mod_eq_of_lt (by norm_num),
  split,
  exact Nat.mod_eq_of_lt (by norm_num),
  exact Nat.mod_eq_of_lt (by norm_num),
  introv h,
  obtain ⟨ h56, h78, h9 ⟩ := h,
  sorry
}

end least_number_exists_least_number_is_2187_l332_332305


namespace geometric_sequence_a3_q_l332_332116

theorem geometric_sequence_a3_q (a_5 a_4 a_3 a_2 a_1 : ℝ) (q : ℝ) :
  a_5 - a_1 = 15 →
  a_4 - a_2 = 6 →
  (q = 2 ∧ a_3 = 4) ∨ (q = 1/2 ∧ a_3 = -4) :=
by
  sorry

end geometric_sequence_a3_q_l332_332116


namespace maximum_value_in_set_B_l332_332631

-- Define the conditions
def is_permutation (a : ℕ → ℕ) (N : ℕ) : Prop := ∀ k : ℕ, (1 ≤ k ∧ k ≤ N) → ∃ n : ℕ, a n = k

def set_B (a : ℕ → ℕ) (m : ℕ) : set ℕ := {x | ∃ n : ℕ, n ≤ 2016 - m ∧ x = (finset.range m).sum (λ i, a (n + i + 1)) }

-- Statement to prove
theorem maximum_value_in_set_B :
  ∀ (a : ℕ → ℕ), is_permutation a 2016 → ∃ x ∈ set_B a 8, x = 16100 := 
sorry

end maximum_value_in_set_B_l332_332631


namespace reduced_price_per_kg_l332_332335

theorem reduced_price_per_kg (P R : ℝ) (Q : ℝ)
  (h1 : R = 0.80 * P)
  (h2 : Q * P = 1500)
  (h3 : (Q + 10) * R = 1500) : R = 30 :=
by
  sorry

end reduced_price_per_kg_l332_332335


namespace find_integer_solutions_l332_332377

theorem find_integer_solutions (x y : ℤ) :
  8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y ↔
  (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
by 
  sorry

end find_integer_solutions_l332_332377


namespace dave_paid_more_than_doug_l332_332796

theorem dave_paid_more_than_doug :
  let base_price := 10
  let garlic_slices := 6
  let additional_garlic_cost := 3
  let total_slices := 10
  let dave_slices := garlic_slices + 1
  let doug_slices := total_slices - dave_slices
  let regular_drink_cost := 2
  let premium_drink_cost := 4
  let total_pizza_cost := base_price + additional_garlic_cost
  let cost_per_slice := total_pizza_cost / total_slices
  let dave_total_cost := dave_slices * cost_per_slice + premium_drink_cost
  let doug_total_cost := doug_slices * cost_per_slice + regular_drink_cost
  dave_total_cost - doug_total_cost = 7.2 := 
by
  have base_price := 10
  have garlic_slices := 6
  have additional_garlic_cost := 3
  have total_slices := 10
  have dave_slices := garlic_slices + 1
  have doug_slices := total_slices - dave_slices
  have regular_drink_cost := 2
  have premium_drink_cost := 4
  have total_pizza_cost := base_price + additional_garlic_cost
  have cost_per_slice := total_pizza_cost / total_slices
  have dave_total_cost := dave_slices * cost_per_slice + premium_drink_cost
  have doug_total_cost := doug_slices * cost_per_slice + regular_drink_cost
  show dave_total_cost - doug_total_cost = 7.2,
  from sorry

end dave_paid_more_than_doug_l332_332796


namespace line_L_correct_l332_332837

-- Definitions
def line1 (x : ℝ) : ℝ := (1/2) * x + 3

def is_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def distance_between_parallel_lines (m : ℝ) (c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / real.sqrt (m^2 + 1)

-- Conditions
def slope_line1 := 1/2
def c1 := 3
def d := 5

-- Question - theorem to prove
theorem line_L_correct :
  ∃ c2, distance_between_parallel_lines slope_line1 c1 c2 = d ∧
        (c2 = 3 + (5 * real.sqrt 5) / 2 ∨ c2 = 3 - (5 * real.sqrt 5) / 2) :=
sorry

end line_L_correct_l332_332837


namespace total_number_of_fish_l332_332252

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l332_332252


namespace total_cost_with_discount_l332_332271

theorem total_cost_with_discount (total_cost_two_books : ℝ) (discount_rate : ℝ) (total_books : ℕ) :
  total_cost_two_books = 36 → discount_rate = 0.10 → total_books = 3 →
  (total_cost_two_books / 2 * (1 - discount_rate) * total_books) = 48.60 :=
begin
  intros h1 h2 h3,
  sorry
end

end total_cost_with_discount_l332_332271


namespace external_angle_bisector_C_property_l332_332526

-- Definition of the points and main entities.
variables {A B C A₁ B₁ : Type}

-- Conditions of the problem
def is_triangle (A B C : Type) : Prop := true -- Placeholder, needs proper definition
def is_angle_bisector (X Y Z P : Type) : Prop := true -- Placeholder, needs proper definition
def is_external_angle_bisector (X Y Z : Type) : Prop := true -- Placeholder, needs proper definition
def is_on_line (P Q R : Type) : Prop := true -- Placeholder, needs proper definition
def are_parallel (L M : Type) : Prop := true -- Placeholder, needs proper definition

-- Conditions specific to the problem.
axiom h1 : is_triangle A B C
axiom h2 : is_angle_bisector A B C A₁
axiom h3 : is_angle_bisector B A C B₁

-- Statement of the theorem to be proven.
theorem external_angle_bisector_C_property :
  is_external_angle_bisector A B C → 
  (∃ C₁, is_on_line C₁ A B ∧ is_on_line A₁ B₁ C₁) ∨ 
  are_parallel (line A B) (line A₁ B₁) :=
sorry -- Proof is to be completed.

end external_angle_bisector_C_property_l332_332526


namespace Jill_age_l332_332404

theorem Jill_age 
  (G H I J : ℕ)
  (h1 : G = H - 4)
  (h2 : H = I + 5)
  (h3 : I + 2 = J)
  (h4 : G = 18) : 
  J = 19 := 
sorry

end Jill_age_l332_332404


namespace derivative_correct_l332_332455

-- Define the function y = x^2 * sin x
def func (x : ℝ) : ℝ := x^2 * Real.sin x

-- State that the derivative of func is 2x * sin x + x^2 * cos x
theorem derivative_correct : (fun x => deriv func x) = (fun x => 2 * x * Real.sin x + x^2 * Real.cos x) :=
by sorry

end derivative_correct_l332_332455


namespace watering_area_is_10000_l332_332340

-- Define the conditions
def rhinoceroses := 8000
def grazing_per_rhinoceros := 100
def population_increase_rate := 0.10
def total_area_after_increase := 890000

-- Define the future rhinoceros population
def future_rhinoceroses := rhinoceroses + (population_increase_rate * rhinoceroses).to_nat

-- Define the total grazing area required for the increased population
def grazing_area (r : ℕ) (g : ℕ) : ℕ := r * g
def total_grazing_area_for_future := grazing_area future_rhinoceroses grazing_per_rhinoceros

-- Define the problem: Prove that the watering area is 10000 acres
theorem watering_area_is_10000 : 
  total_area_after_increase - total_grazing_area_for_future = 10000 := 
sorry

end watering_area_is_10000_l332_332340


namespace total_spending_l332_332847

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end total_spending_l332_332847


namespace socks_total_l332_332168

def socks_lisa (initial: Nat) := initial + 0

def socks_sandra := 20

def socks_cousin (sandra: Nat) := sandra / 5

def socks_mom (initial: Nat) := 8 + 3 * initial

theorem socks_total (initial: Nat) (sandra: Nat) :
  initial = 12 → sandra = 20 → 
  socks_lisa initial + socks_sandra + socks_cousin sandra + socks_mom initial = 80 :=
by
  intros h_initial h_sandra
  rw [h_initial, h_sandra]
  sorry

end socks_total_l332_332168


namespace area_of_quadrilateral_ABCD_l332_332591

noncomputable def area_quadrilateral_ABCDE
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC AD DC : ℝ)
  (h_A : AB > 0) (h_B : BC > 0) (h_C : AD > 0) (h_D : DC > 0)
  (h_right_angle_B : ∃ θ : ℝ, θ = π / 2)
  (h_right_angle_D : ∃ θ : ℝ, θ = π / 2)
  (h_AC : ∃ AC : ℝ, AC = 5) :
  ℝ :=
1/2 * AB * BC + 1/2 * AD * DC

theorem area_of_quadrilateral_ABCD
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC AD DC : ℝ)
  (h1 : AB = 3) (h2 : BC = 4) (h3 : AD = 4) (h4 : DC = 3)
  (h_right_angle_B : ∃ θ : ℝ, θ = π / 2)
  (h_right_angle_D : ∃ θ : ℝ, θ = π / 2)
  (h_AC : ∃ AC : ℝ, AC = 5) :
  area_quadrilateral_ABCDE A B C D AB BC AD DC 0 0 0 0 h_right_angle_B h_right_angle_D h_AC = 12 :=
sorry

end area_of_quadrilateral_ABCD_l332_332591


namespace triangle_area_is_correct_l332_332380

-- Defining the points
structure Point where
  x : ℝ
  y : ℝ

-- Defining vertices A, B, C
def A : Point := { x := 2, y := -3 }
def B : Point := { x := 0, y := 4 }
def C : Point := { x := 3, y := -1 }

-- Vector from C to A
def v : Point := { x := A.x - C.x, y := A.y - C.y }

-- Vector from C to B
def w : Point := { x := B.x - C.x, y := B.y - C.y }

-- Cross product of vectors v and w in 2D
noncomputable def cross_product (v w : Point) : ℝ :=
  v.x * w.y - v.y * w.x

-- Absolute value of the cross product
noncomputable def abs_cross_product (v w : Point) : ℝ :=
  |cross_product v w|

-- Area of the triangle
noncomputable def area_of_triangle (v w : Point) : ℝ :=
  (1 / 2) * abs_cross_product v w

-- Prove the area of the triangle is 5.5
theorem triangle_area_is_correct : area_of_triangle v w = 5.5 :=
  sorry

end triangle_area_is_correct_l332_332380


namespace sum_of_all_possible_values_of_f1_l332_332544

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem sum_of_all_possible_values_of_f1 :
    (∀ x : ℝ, x ≠ 0 → f a b c (x-1) + f a b c x + f a b c (x+1) = (f a b c x)^2 / (2027 * x)) →
    (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) →
    ∑ x in {f a b c 1}, Real x = 6081 :=
by
  sorry

end sum_of_all_possible_values_of_f1_l332_332544


namespace initials_count_l332_332472

-- Let L be the set of letters {A, B, C, D, E, F, G, H, I, J}
def L : finset char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Each initial can be any element in L
def initials : Type := L × L × L

-- We need to prove that the number of different three-letter sets of initials is equal to 1000
theorem initials_count : finset.card (finset.product L (finset.product L L)) = 1000 := 
sorry

end initials_count_l332_332472


namespace expand_and_simplify_fraction_l332_332375

theorem expand_and_simplify_fraction (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * ((7 / (x^2)) + 15 * (x^3) - 4 * x) = (3 / (x^2)) + (45 * (x^3) / 7) - (12 * x / 7) :=
by
  sorry

end expand_and_simplify_fraction_l332_332375


namespace sum_sequence_l332_332055

theorem sum_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, (∑ i in finset.range n, real.sqrt (a (i + 1))) = n^2 + n) :
  (∑ i in finset.range n, (a (i + 1) / (i + 1))) = 2 * n^2 + 2 * n := by
sorry

end sum_sequence_l332_332055


namespace sampling_interval_l332_332402

theorem sampling_interval (N n : ℕ) (hN : N = 2014) (hn : n = 100) : 
  ∃ I, I = 20 ∧ N % n ≠ 0 :=
by
  use 20
  split
  · rfl
  · norm_num
    sorry

end sampling_interval_l332_332402


namespace max_sum_three_with_at_least_one_neg_l332_332020

def max_sum_three_with_neg (s : Set ℤ) : ℤ :=
  if h : s = { -7, -5, -3, 0, 2, 4, 6 }
  then 7
  else 0 -- Default, should not reach here if conditions are met

theorem max_sum_three_with_at_least_one_neg : 
  max_sum_three_with_neg { -7, -5, -3, 0, 2, 4, 6 } = 7 :=
by
  -- The detailed proof will follow here
  sorry

end max_sum_three_with_at_least_one_neg_l332_332020


namespace imaginary_part_of_z_l332_332001

theorem imaginary_part_of_z (θ : ℝ) (h : sin (2 * θ) - 1 = 0): (sqrt 2 * cos θ - 1) ≠ 0 → (0 - (1 + sqrt 2)) = -2 :=
by sorry

end imaginary_part_of_z_l332_332001


namespace jake_has_fewer_peaches_than_steven_l332_332532

def steven_peaches : ℕ := 19
def jill_peaches : ℕ := 6
def jake_peaches : ℕ

def condition (s j : ℕ) : Prop :=
  s = 19 ∧ j = 6

theorem jake_has_fewer_peaches_than_steven (s j : ℕ) (cond : condition s j) : jake_peaches < s :=
by
  cases cond
  construct sorry -- the proof goes here

#check jake_has_fewer_peaches_than_steven

end jake_has_fewer_peaches_than_steven_l332_332532


namespace smaller_angle_at_3_40_l332_332738

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332738


namespace range_of_g_l332_332406

noncomputable def f (a x : ℝ) : ℝ := (Real.log ((x / a) - 1)) + (1 / x) + (a / 2)

noncomputable def g (a : ℝ) : ℝ := f a ((1 - Real.sqrt(1 - 4 * a)) / 2) + f a ((1 + Real.sqrt(1 - 4 * a)) / 2)

theorem range_of_g :
  (∀ a : ℝ, 0 < a ∧ a < 1/4 → g(a) = 1/a + a) → (∀ a : ℝ, 0 < a ∧ a < 1/4 → g(a) ∈ Ioo (17 / 4) (⊤)) := 
sorry

end range_of_g_l332_332406


namespace clock_angle_3_40_l332_332676

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l332_332676


namespace AC_perp_A1L_l332_332200

variable {A B C I A1 B1 K L : Point}

-- Conditions
def inscribed_circle_center (I : Point) (A B C : Point) : Prop := sorry
def touch_BC_at_A1 (I A1 B C : Point) : Prop := sorry
def touch_AC_at_B1 (I B1 A C : Point) : Prop := sorry
def perpendicular_bisector_CI_intersects_BC_at_K (I K C B : Point) : Prop := sorry
def perpendicular_to_KB1_through_I_intersects_AC_at_L (I L K B1 A C : Point) : Prop := sorry

-- Prove AC and A1L are perpendicular
theorem AC_perp_A1L 
  (h1 : inscribed_circle_center I A B C)
  (h2 : touch_BC_at_A1 I A1 B C)
  (h3 : touch_AC_at_B1 I B1 A C)
  (h4 : perpendicular_bisector_CI_intersects_BC_at_K I K C B)
  (h5 : perpendicular_to_KB1_through_I_intersects_AC_at_L I L K B1 A C) :
  is_perpendicular AC A1L :=
sorry

end AC_perp_A1L_l332_332200


namespace sum_geometric_sequence_terms_l332_332235

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l332_332235


namespace triangle_equi_if_sides_eq_sum_of_products_l332_332996

theorem triangle_equi_if_sides_eq_sum_of_products (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ac) : a = b ∧ b = c :=
by sorry

end triangle_equi_if_sides_eq_sum_of_products_l332_332996


namespace find_N_l332_332348

theorem find_N :
  (∃ (N : ℕ), 
    (5 / 10) * (20 / (20 + N)) + (5 / 10) * (N / (20 + N)) = 0.60) → N = 5 :=
by
  sorry

end find_N_l332_332348


namespace polynomial_has_root_of_multiplicity_2_l332_332216

theorem polynomial_has_root_of_multiplicity_2 (r s k : ℝ)
  (h1 : x^3 + k * x - 128 = (x - r)^2 * (x - s)) -- polynomial has a root of multiplicity 2
  (h2 : -2 * r - s = 0)                         -- relationship from coefficient of x²
  (h3 : r^2 + 2 * r * s = k)                    -- relationship from coefficient of x
  (h4 : r^2 * s = 128)                          -- relationship from constant term
  : k = -48 := 
sorry

end polynomial_has_root_of_multiplicity_2_l332_332216


namespace sum_of_solutions_eqn_l332_332949

theorem sum_of_solutions_eqn :
  (∑ x in {x | (x - 8) ^ 2 = 49}, x) = 16 :=
sorry

end sum_of_solutions_eqn_l332_332949


namespace find_b_l332_332170

theorem find_b (b : ℤ) (h₁ : b < 0) : (∃ n : ℤ, (x : ℤ) * x + b * x - 36 = (x + n) * (x + n) - 20) → b = -8 :=
by
  intro hX
  sorry

end find_b_l332_332170


namespace smaller_angle_at_3_40_l332_332737

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332737


namespace diana_probability_l332_332861

open Classical

noncomputable def probability_diana_greater_or_double_apollo : ℚ :=
  let outcomes_diana := {1, 2, 3, 4, 5, 6, 7, 8}
  let outcomes_apollo := {1, 2, 3, 4}
  let successful_pairs := 
    { (a, b) | a ∈ outcomes_diana ∧ b ∈ outcomes_apollo ∧ 
               (a > b ∨ a = 2 * b) }
  let successful_outcomes := successful_pairs.card
  let total_outcomes := outcomes_diana.card * outcomes_apollo.card
  (successful_outcomes : ℚ) / (total_outcomes : ℚ)

theorem diana_probability : probability_diana_greater_or_double_apollo = 13 / 16 := by
  sorry

end diana_probability_l332_332861


namespace condition_sufficient_but_not_necessary_l332_332493

theorem condition_sufficient_but_not_necessary (x : ℝ) :
  (\frac{x-5}{2-x} > 0 → |x-1| < 4) ∧ ¬ (|x-1| < 4 → \frac{x-5}{2-x} > 0) :=
sorry

end condition_sufficient_but_not_necessary_l332_332493


namespace sum_of_solutions_sum_of_all_solutions_l332_332905

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332905


namespace sabrina_remaining_cookies_l332_332190

-- Definitions based on the conditions
def initial_cookies : ℕ := 20
def cookies_given_to_brother : ℕ := 10
def cookies_given_to_sabrina_by_mother := cookies_given_to_brother / 2
def cookies_after_giving_to_brother := initial_cookies - cookies_given_to_brother
def total_cookies_after_receiving_from_mother := cookies_after_giving_to_brother + cookies_given_to_sabrina_by_mother
def cookies_given_to_sister := (2 / 3 : ℚ) * total_cookies_after_receiving_from_mother

-- The statement to prove
theorem sabrina_remaining_cookies : total_cookies_after_receiving_from_mother - cookies_given_to_sister = 5 :=
by
  have h1 : initial_cookies = 20 := rfl
  have h2 : cookies_given_to_brother = 10 := rfl
  have h3 : cookies_given_to_sabrina_by_mother = 5 := rfl
  have h4 : cookies_after_giving_to_brother = 10 := by rw [h1, h2]; norm_num
  have h5 : total_cookies_after_receiving_from_mother = 15 := by rw [h4, h3]; norm_num
  have h6 : cookies_given_to_sister = 10 := by rw [h5]; norm_num
  have h7 : total_cookies_after_receiving_from_mother - cookies_given_to_sister = 5 := by rw [h5, h6]; norm_num
  exact h7

end sabrina_remaining_cookies_l332_332190


namespace base_b_expression_not_divisible_l332_332367

theorem base_b_expression_not_divisible 
  (b : ℕ) : 
  (b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 7 ∨ b = 8) →
  (2 * b^3 - 2 * b^2 + b - 1) % 5 ≠ 0 ↔ (b ≠ 6) :=
by
  sorry

end base_b_expression_not_divisible_l332_332367


namespace sum_of_solutions_equation_l332_332932

def sum_of_solutions : ℤ :=
  let solutions := {x : ℤ | (x - 8) ^ 2 = 49}
  solutions.sum

theorem sum_of_solutions_equation : sum_of_solutions = 16 := by
  sorry

end sum_of_solutions_equation_l332_332932


namespace complex_magnitude_l332_332991

theorem complex_magnitude (z : ℂ) (h : complex.I * z = 4 - 2 * complex.I) : complex.abs z = 2 * real.sqrt 5 :=
sorry

end complex_magnitude_l332_332991


namespace volume_tetrahedron_NMLB_l332_332178

noncomputable def volume_of_tetrahedron (V : ℝ) (BL BC : ℝ) (CM CD : ℝ) (DN AD : ℝ) : ℝ :=
  V / 60

theorem volume_tetrahedron_NMLB (V BC CD AD : ℝ)
  (hV : V > 0)
  (hBL : 3 * BL = BC)
  (hCM : 4 * CM = CD)
  (hDN : 5 * DN = AD) :
  volume_of_tetrahedron V BL CM DN = V / 60 :=
by
  -- proof goes here
  sorry

end volume_tetrahedron_NMLB_l332_332178


namespace minLengthUniversal_l332_332416

/-- A sequence is universal for a given n if any permutation of 1, 2, ..., n can be obtained by removing some elements from it -/
def isUniversalSequence (n : ℕ) (seq : List ℕ) : Prop :=
  ∀ (perm : List ℕ), (perm.length = n) → (perm.nodup) → (∀ m ∈ perm, m ≤ n) → ∃ (subseq : List ℕ), subseq ⊆ seq ∧ perm ~ subseq

/-- Prove that any universal sequence consists of at least (n * (n + 1)) // 2 members -/
theorem minLengthUniversal (n : ℕ) (seq : List ℕ) :
  isUniversalSequence n seq → seq.length ≥ (n * (n + 1)) // 2 :=
sorry

end minLengthUniversal_l332_332416


namespace range_of_a_l332_332550

theorem range_of_a (a : ℝ) (x1 x2 : ℝ)
  (h_poly: ∀ x, x * x + (a * a - 1) * x + (a - 2) = 0 → x = x1 ∨ x = x2)
  (h_order: x1 < 1 ∧ 1 < x2) : 
  -2 < a ∧ a < 1 := 
sorry

end range_of_a_l332_332550


namespace sum_of_solutions_sum_of_all_solutions_l332_332908

theorem sum_of_solutions (x : ℝ) (h : (x - 8) ^ 2 = 49) : x = 15 ∨ x = 1 := by
  sorry

lemma sum_x_values : 15 + 1 = 16 := by
  norm_num

theorem sum_of_all_solutions : 16 := by
  exact sum_x_values

end sum_of_solutions_sum_of_all_solutions_l332_332908


namespace find_s_l332_332512

variables (PQRS : Type*) [parallelogram PQRS] 
          (X PQ RS P Q R S : PQRS)
          (φ : ℝ) 
          (angle_QPS angle_PSR angle_PRQ angle_PQR angle_PXQ : ℝ)
          (s : ℝ)

-- Conditions
#check ∀  (h₁ : X = intersection (diagonal PQ RS)),
        (h₂ : angle_QPS = 3 * φ),
        (h₃ : angle_PSR = 3 * φ),
        (h₄ : angle_PRQ = φ),
        (h₅ : angle_PQR = s * angle_PXQ)
-- Proof of the specific problem
theorem find_s 
        (h₁ : X = intersection (diagonal PQ RS))
        (h₂ : angle_QPS = 3 * φ)
        (h₃ : angle_PSR = 3 * φ)
        (h₄ : angle_PRQ = φ)
        (h₅ : angle_PQR = s * angle_PXQ) : 
        s = 7 / 8 := 
    sorry

end find_s_l332_332512


namespace Missouri_to_NewYork_by_car_l332_332214

def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def total_distance_car : ℝ := distance_plane * (1 + increase_percentage)
def distance_midway : ℝ := total_distance_car / 2

theorem Missouri_to_NewYork_by_car : distance_midway = 1400 := by
  sorry

end Missouri_to_NewYork_by_car_l332_332214


namespace tan_of_angle_B_l332_332565

theorem tan_of_angle_B {A B C F G : Type} 
  [triangle : RightTriangle A B C C F G] 
  (h1 : hasRightAngle C)
  (h2 : Between A F G)
  (h3 : TrisectAngle (Angle C A F) (Angle C F G))
  (h4 : FG_to_BG_ratio : FG / BG = 3 / 7) :
  tan B = 7 * sqrt (49 * x^2 - 16) / (196 * x) :=
sorry

end tan_of_angle_B_l332_332565


namespace maria_visits_per_day_l332_332572

-- Definitions based on the conditions
def cups_per_visit : ℕ := 3
def cups_per_day : ℕ := 6

-- Statement: Prove that Maria goes to the coffee shop 2 times per day.
theorem maria_visits_per_day : ∃ n : ℕ, cups_per_day = n * cups_per_visit ∧ n = 2 :=
by {
  use 2,
  split,
  {
    -- Proof for cups_per_day = 2 * cups_per_visit
    sorry
  },
  {
    -- Proof that n = 2
    sorry
  }
}

end maria_visits_per_day_l332_332572


namespace find_x_l332_332869

theorem find_x (x : ℝ) : log 49 (3 * x - 1) = -1 / 2 → x = 8 / 21 :=
by
  intro h
  sorry

end find_x_l332_332869


namespace clock_angle_l332_332706

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l332_332706


namespace work_done_together_in_six_days_l332_332789

theorem work_done_together_in_six_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
  1 / (A + B) = 6 :=
by
  sorry

end work_done_together_in_six_days_l332_332789


namespace min_questions_30_l332_332175

theorem min_questions_30 : ∀ (cards : Fin 30 → ℤ), 
    (∀ i, cards i = 1 ∨ cards i = -1) →
    ∃ (questions : Fin 30 → Fin 30 × Fin 30 × Fin 30), 
    (∀ (q : Fin 30), (cards (questions q).1) * (cards (questions q).2) * (cards (questions q).3) ∈ {1, -1}) ∧
    ∀ (q : Fin 30), q.cardinality = 10 := 
sorry

end min_questions_30_l332_332175


namespace sum_of_roots_eq_seventeen_l332_332911

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l332_332911


namespace inequality_a3_b3_c3_l332_332589

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3 * a * b * c > a * b * (a + b) + b * c * (b + c) + a * c * (a + c) :=
by
  sorry

end inequality_a3_b3_c3_l332_332589


namespace isosceles_triangle_count_correct_l332_332525

variables (P : Type) [EuclideanGeometry P]

def isosceles_triangle_count (ABC : Triangle P) (F : P)
  (h1 : ABC.isEquilateral)
  (D : P) (h2 : D ∈ segment ABC.ABC.A ABC.ABC.C)
  (h3 : segment ABC.ABC.A D = segment D ABC.ABC.C)
  (E : P) (h4 : E ∈ segment ABC.ABC.B ABC.ABC.C)
  (h5 : parallel (segment D E) (segment ABC.ABC.A ABC.ABC.B))
  (h6 : midpoint F (segment ABC.ABC.A ABC.ABC.C)) : Prop :=
  ∃ n, n = 6

theorem isosceles_triangle_count_correct (ABC : Triangle P) (F : P)
  (h1 : ABC.isEquilateral)
  (D : P) (h2 : D ∈ segment ABC.ABC.A ABC.ABC.C)
  (h3 : segment ABC.ABC.A D = segment D ABC.ABC.C)
  (E : P) (h4 : E ∈ segment ABC.ABC.B ABC.ABC.C)
  (h5 : parallel (segment D E) (segment ABC.ABC.A ABC.ABC.B))
  (h6 : midpoint F (segment ABC.ABC.A ABC.ABC.C)) :
  isosceles_triangle_count ABC F h1 D h2 h3 E h4 h5 h6 :=
by 
  sorry

end isosceles_triangle_count_correct_l332_332525


namespace R_has_3200_l332_332583

-- Definitions used in the problem
def total_amount : ℝ := 8000
def pq_amount (R_amount : ℝ) : ℝ := (3/5) * total_amount
def R_amount : ℝ := (2/5) * total_amount

-- The statement to prove
theorem R_has_3200 (total_amount : ℝ) (R_amount : ℝ) (pq_amount : ℝ) :
    total_amount = 8000 ∧
    pq_amount = (3/5) * total_amount ∧
    R_amount = (2/3) * pq_amount →
    R_amount = 3200 := by
  sorry

end R_has_3200_l332_332583


namespace combined_mpg_proof_l332_332188

noncomputable def combined_mpg (d : ℝ) : ℝ :=
  let ray_mpg := 50
  let tom_mpg := 20
  let alice_mpg := 25
  let total_fuel := (d / ray_mpg) + (d / tom_mpg) + (d / alice_mpg)
  let total_distance := 3 * d
  total_distance / total_fuel

theorem combined_mpg_proof :
  ∀ d : ℝ, d > 0 → combined_mpg d = 300 / 11 :=
by
  intros d hd
  rw [combined_mpg]
  simp only [div_eq_inv_mul, mul_inv, inv_inv]
  sorry

end combined_mpg_proof_l332_332188


namespace angle_at_3_40_pm_is_130_degrees_l332_332668

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l332_332668


namespace total_pink_crayons_l332_332570

def mara_crayons := 40
def mara_pink_percent := 10
def luna_crayons := 50
def luna_pink_percent := 20

def pink_crayons (total_crayons : ℕ) (percent_pink : ℕ) : ℕ :=
  (percent_pink * total_crayons) / 100

def mara_pink_crayons := pink_crayons mara_crayons mara_pink_percent
def luna_pink_crayons := pink_crayons luna_crayons luna_pink_percent

theorem total_pink_crayons : mara_pink_crayons + luna_pink_crayons = 14 :=
by
  -- Proof can be written here.
  sorry

end total_pink_crayons_l332_332570


namespace sin_of_tan_l332_332504

theorem sin_of_tan (A B C : Type*) [Triangle ABC] 
  (h1 : angle A = 90) (h2 : tan C = 4/3) : sin C = 3/5 :=
sorry

end sin_of_tan_l332_332504


namespace dividend_calculation_l332_332876

theorem dividend_calculation :
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  dividend = 10917708 :=
by
  let divisor := 12
  let quotient := 909809
  let dividend := divisor * quotient
  show dividend = 10917708
  sorry

end dividend_calculation_l332_332876


namespace cube_root_of_minus_eighth_l332_332054

theorem cube_root_of_minus_eighth : 
  ∃ (y : ℝ), x = -1/8 → y^3 = x → y = -1/2 :=
begin
  sorry
end

end cube_root_of_minus_eighth_l332_332054


namespace smaller_angle_3_40_pm_l332_332742

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l332_332742


namespace smaller_angle_at_3_40_l332_332734

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l332_332734


namespace smaller_angle_between_hands_at_3_40_l332_332685

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l332_332685


namespace kanul_spent_on_raw_materials_l332_332534

theorem kanul_spent_on_raw_materials 
    (total_amount : ℝ)
    (spent_machinery : ℝ)
    (spent_cash_percent : ℝ)
    (spent_cash : ℝ)
    (amount_raw_materials : ℝ)
    (h_total : total_amount = 93750)
    (h_machinery : spent_machinery = 40000)
    (h_percent : spent_cash_percent = 20 / 100)
    (h_cash : spent_cash = spent_cash_percent * total_amount)
    (h_sum : total_amount = amount_raw_materials + spent_machinery + spent_cash) : 
    amount_raw_materials = 35000 :=
sorry

end kanul_spent_on_raw_materials_l332_332534
