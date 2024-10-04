import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GCD
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Path
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Polygon
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set
import Mathlib.Data.Set.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean
import Mathlib.Geometry.euclidean.basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Logic.Basic
import Mathlib.SetTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace mary_sheep_problem_l643_643284

theorem mary_sheep_problem :
  let initial_sheep := 400
  in let sheep_given_to_sister := initial_sheep / 4
  in let remaining_after_sister := initial_sheep - sheep_given_to_sister
  in let sheep_given_to_brother := remaining_after_sister / 2
  in let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  in sheep_remaining = 150 :=
by
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let remaining_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := remaining_after_sister / 2
  let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  show sheep_remaining = 150 from sorry

end mary_sheep_problem_l643_643284


namespace right_triangle_area_l643_643678

theorem right_triangle_area
  (hypotenuse : ℝ) (angle : ℝ) (hyp_eq : hypotenuse = 12) (angle_eq : angle = 30) :
  ∃ area : ℝ, area = 18 * Real.sqrt 3 :=
by
  have side1 := hypotenuse / 2  -- Shorter leg = hypotenuse / 2
  have side2 := side1 * Real.sqrt 3  -- Longer leg = shorter leg * sqrt 3
  let area := (side1 * side2) / 2  -- Area calculation
  use area
  sorry

end right_triangle_area_l643_643678


namespace find_speed_l643_643620

-- Define relevant constants and conditions
def circumference_in_feet : ℝ := 15
def feet_per_mile : ℝ := 5280
def speed_increase_mph : ℝ := 8
def time_reduction_seconds : ℝ := 1 / 6
def seconds_per_hour : ℝ := 3600

-- Define the equivalent Lean 4 problem statement
theorem find_speed (r : ℝ) (t : ℝ) (h1 : r * t = circumference_in_feet / feet_per_mile * seconds_per_hour) 
    (h2 : (r + speed_increase_mph) * (t - time_reduction_seconds / seconds_per_hour) = circumference_in_feet / feet_per_mile * seconds_per_hour) : 
    r = 12 :=
sorry

end find_speed_l643_643620


namespace chord_AB_length_line_PQ_eqn_l643_643844

/-- Problem conditions: definitions and statements -/
def circle_O_param (θ: ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

def line_l_param (t: ℝ) : ℝ × ℝ := (2 + t, 4 + t)

def polar_circle_C (θ: ℝ) : ℝ := 2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

/-- Questions: proving the given results -/
theorem chord_AB_length :
  let O := (0, 0)
  let intersection_points := {p : ℝ × ℝ | ∃ θ t, circle_O_param θ = p ∧ line_l_param t = p}
  ∃ A B, A ≠ B ∧ A ∈ intersection_points ∧ B ∈ intersection_points ∧ 
  Real.dist A B = 2 * Real.sqrt 2 :=
sorry

theorem line_PQ_eqn :
  let circle_O := {p: ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4}
  let circle_C := {p: ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 2 * p.1 + 2 * Real.sqrt 3 * p.2}
  ∃ x y, (x, y) ∈ circle_O ∧ (x, y) ∈ circle_C ∧
  ∀ (p: ℝ × ℝ), p ∈ circle_O ∧ p ∈ circle_C → p.1 + Real.sqrt 3 * p.2 - 2 = 0 :=
sorry

end chord_AB_length_line_PQ_eqn_l643_643844


namespace third_side_range_l643_643189

theorem third_side_range (a : ℝ) (h₃ : 0 < a ∧ a ≠ 0) (h₅ : 0 < a ∧ a ≠ 0): 
  (2 < a ∧ a < 8) ↔ (3 - 5 < a ∧ a < 3 + 5) :=
by
  sorry

end third_side_range_l643_643189


namespace find_AB_length_l643_643598

noncomputable def is_rectangle (A B C D : Type) := sorry

theorem find_AB_length (A B C D P Q : Type) (BP CP AB x : ℝ) (h1 : is_rectangle A B C D) 
  (h2 : P ∈ line_segment B C) 
  (h3 : BP = 20) 
  (h4 : CP = 5) 
  (h5 : Q = projection P (segment A D)) 
  (h6 : AB = PQ) 
  (h7 : tan (angle A P D) = 2) :
  AB = 20 := 
sorry

end find_AB_length_l643_643598


namespace conjugate_of_Z_l643_643172

noncomputable def Z : ℂ := (i^2018) / (1 - i)

theorem conjugate_of_Z : conj Z = (-1 + i) / 2 :=
  sorry

end conjugate_of_Z_l643_643172


namespace dice_probability_l643_643501

theorem dice_probability :
  let total_outcomes := 6^4,
      successful_outcomes := 6 * 4 * 5
  in
  (successful_outcomes / total_outcomes : ℚ) = 5 / 54 :=
by
  sorry

end dice_probability_l643_643501


namespace solve_for_s_l643_643218

theorem solve_for_s : ∃ (s t : ℚ), (8 * s + 7 * t = 160) ∧ (s = t - 3) ∧ (s = 139 / 15) := by
  sorry

end solve_for_s_l643_643218


namespace trains_meet_1050_km_from_delhi_l643_643738

def distance_train_meet (t1_departure t2_departure : ℕ) (s1 s2 : ℕ) : ℕ :=
  let t_gap := t2_departure - t1_departure      -- Time difference between the departures in hours
  let d1 := s1 * t_gap                          -- Distance covered by the first train until the second train starts
  let relative_speed := s2 - s1                 -- Relative speed of the second train with respect to the first train
  d1 + s2 * (d1 / relative_speed)               -- Distance from Delhi where they meet

theorem trains_meet_1050_km_from_delhi :
  distance_train_meet 9 14 30 35 = 1050 := by
  -- Definitions based on the problem's conditions
  let t1 := 9          -- First train departs at 9 a.m.
  let t2 := 14         -- Second train departs at 2 p.m. (14:00 in 24-hour format)
  let s1 := 30         -- Speed of the first train in km/h
  let s2 := 35         -- Speed of the second train in km/h
  sorry -- proof to be filled in

end trains_meet_1050_km_from_delhi_l643_643738


namespace terminal_side_quadrant_l643_643576

theorem terminal_side_quadrant (α : ℝ) (k : ℤ) (hk : α = 45 + k * 180) :
  (∃ n : ℕ, k = 2 * n ∧ α = 45) ∨ (∃ n : ℕ, k = 2 * n + 1 ∧ α = 225) :=
sorry

end terminal_side_quadrant_l643_643576


namespace length_of_scale_parts_l643_643736

theorem length_of_scale_parts (total_length_ft : ℕ) (remaining_inches : ℕ) (parts : ℕ) : 
  total_length_ft = 6 ∧ remaining_inches = 8 ∧ parts = 2 →
  ∃ ft inches, ft = 3 ∧ inches = 4 :=
by
  sorry

end length_of_scale_parts_l643_643736


namespace mutually_exclusive_not_opposite_l643_643472

noncomputable def distribute_cards (cards : Fin 4) (people : Fin 4) : Fin 4 → Fin 4 :=
  λ i, if i < fin_zero 1 then cards else people

theorem mutually_exclusive_not_opposite :
  ∀ (cards: Fin 4) (people: Fin 4), 
  let A_gets_red := distribute_cards cards people 0 = 0,
      B_gets_red := distribute_cards cards people 1 = 0 in
  (A_gets_red ∧ B_gets_red) = False ∧
  (¬(A_gets_red ∧ ¬B_gets_red) ∧ ¬(¬A_gets_red ∧ B_gets_red)) = False :=
by
  sorry

end mutually_exclusive_not_opposite_l643_643472


namespace least_number_least_number_divisible_l643_643739

noncomputable def least_number : ℕ :=
  let lcm := Nat.lcm 33 8
  in lcm + 2

theorem least_number_least_number_divisible (n : ℕ) : 
  (n % 33 = 2) ∧ (n % 8 = 2) ↔ n = least_number :=
by
  sorry

end least_number_least_number_divisible_l643_643739


namespace total_pencils_sold_l643_643688

theorem total_pencils_sold:
  let first_two := 2 * 2 in
  let next_six := 6 * 3 in
  let last_two := 2 * 1 in
  first_two + next_six + last_two = 24 :=
by
  let first_two := 2 * 2
  let next_six := 6 * 3
  let last_two := 2 * 1
  calc
    first_two + next_six + last_two = 4 + 18 + 2 := by sorry
    ... = 24 := by sorry

end total_pencils_sold_l643_643688


namespace collinearity_MT_midpoint_l643_643968

open EuclideanGeometry

variables {A B C A1 B1 C1 M P T S : Point}
variables [CircumcircleABC ω] [CircumcircleAB1C1 ωAB1C1]
variables (hABCacute : acuteTriangle A B C)
variables (hScaleneABC : scalene A B C)
variables (hHeights : TriangleHeight A B C A1 B1 C1)
variables (hCircumcircle : CircumcircleABC A B C ω)
variables (hMidM : Midpoint M B C)
variables (hIntersectionP : SecondIntersection P ωAB1C1 ω)
variables (hTangentsT : TangentIntersection T ω B C)
variables (hIntersectionS : IntersectionAt AT S ω)

theorem collinearity_MT_midpoint 
  (hCollinearityPA1S : Collinear P A1 S) :
  Collinear P A1 S (MidpointSegment M T) :=
sorry

end collinearity_MT_midpoint_l643_643968


namespace grayson_travels_further_l643_643562

noncomputable def grayson_first_part_distance : ℝ := 25 * 1
noncomputable def grayson_second_part_distance : ℝ := 20 * 0.5
noncomputable def total_distance_grayson : ℝ := grayson_first_part_distance + grayson_second_part_distance

noncomputable def total_distance_rudy : ℝ := 10 * 3

theorem grayson_travels_further : (total_distance_grayson - total_distance_rudy) = 5 := by
  sorry

end grayson_travels_further_l643_643562


namespace sum_of_coordinates_l643_643944

-- Definitions based on conditions
variable (f k : ℝ → ℝ)
variable (h₁ : f 4 = 8)
variable (h₂ : ∀ x, k x = (f x) ^ 3)

-- Statement of the theorem
theorem sum_of_coordinates : 4 + k 4 = 516 := by
  -- Proof would go here
  sorry

end sum_of_coordinates_l643_643944


namespace find_number_l643_643407

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 :=
sorry

end find_number_l643_643407


namespace count_integers_in_range_l643_643863

theorem count_integers_in_range : 
  {n : ℤ | 20 < n^2 ∧ n^2 < 200}.to_finset.card = 20 :=
sorry

end count_integers_in_range_l643_643863


namespace ratio_of_areas_l643_643971

variables {s : ℝ} (A B C D P Q : ℝ × ℝ)
  (h_square : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = s ∧ B.2 = 0 ∧ C.1 = s ∧ C.2 = s ∧ D.1 = 0 ∧ D.2 = s)
  (h_P : P.1 = s / 3 ∧ P.2 = 0)
  (h_Q : Q.1 = s ∧ Q.2 = s / 3)

theorem ratio_of_areas (h_nonzero : s ≠ 0) :
  let area_triangle := (1 / 2) * (s / 3) * (2 * s / 3) in
  let area_square := s^2 in
  area_triangle / area_square = 1 / 9 :=
by
  sorry

end ratio_of_areas_l643_643971


namespace mixture_problem_l643_643572

theorem mixture_problem :
  ∀ (x P : ℝ), 
    let initial_solution := 70
    let initial_percentage := 0.20
    let final_percentage := 0.40
    let final_amount := 70
    (x = 70) →
    (initial_percentage * initial_solution + P * x = final_percentage * (initial_solution + x)) →
    (P = 0.60) :=
by
  intros x P initial_solution initial_percentage final_percentage final_amount hx h_eq
  sorry

end mixture_problem_l643_643572


namespace slope_eq_4_l643_643516

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a1 d : ℕ → ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d
def sum_of_terms (a1 d : ℕ → ℕ) (n : ℕ) : ℕ := n * a1 + (n * (n - 1) * d) / 2

-- Given conditions
axiom S2_eq_10 : sum_of_terms a1 d 2 = 10
axiom S5_eq_55 : sum_of_terms a1 d 5 = 55

-- The slope of the line passing through points P(n, a_n) and Q(n+2, a_{n+2})
noncomputable def slope (a : ℕ → ℕ) (n : ℕ) : ℝ := 
  (a (n + 2) - a n) / (2 : ℕ)

-- The required proof
theorem slope_eq_4 {a1 d : ℕ → ℕ} {n : ℕ} 
  (h1 : sum_of_terms a1 d 2 = 10)
  (h2 : sum_of_terms a1 d 5 = 55) : slope (arithmetic_sequence a1 d) n = 4 := 
sorry

end slope_eq_4_l643_643516


namespace determine_s_l643_643951

-- Definitions of triangle, points, and lines
def Triangle (A B C : Type) : Prop := 
sorry 

-- Conditions about the ratios
def cond_CF_FB (CF FB : ℝ) : Prop := CF / FB = 1 / 2
def cond_AG_GB (AG GB : ℝ) : Prop := AG / GB = 2 / 1

-- Definition of s in terms of CQ and QF
def s (CQ QF : ℝ) : ℝ := CQ / QF

-- The main statement to prove
theorem determine_s (A B C F G Q : Type) (CF FB AG GB CQ QF : ℝ) 
  (h1 : Triangle A B C) 
  (h2 : cond_CF_FB CF FB)
  (h3 : cond_AG_GB AG GB)
  (hQ : Q is the intersection of CF and AG) :
  s CQ QF = 2 := 
sorry

end determine_s_l643_643951


namespace hyperbola_problem_l643_643995

noncomputable def value_of_pf2 (P F1 F2 : ℝ × ℝ) : ℝ :=
  let xP := P.1,
      yP := P.2,
      xF1 := F1.1,
      yF1 := F1.2 in
  real.sqrt ((xP - xF1)^2 + (yP - yF1)^2)

theorem hyperbola_problem
  (P F1 F2 : ℝ × ℝ)
  (a b : ℝ)
  (H : a = 8)
  (H2 : b = 6)
  (H3 : ∀ x y, (x, y) = P → x^2 / 64 - y^2 / 36 = 1)
  (HF1 : value_of_pf2 P F1 = 17)
  (HPF2 : ∀ c, c = real.sqrt (a^2 + b^2) → value_of_pf2 P F2 - value_of_pf2 P F1 = 2 * a)
  : value_of_pf2 P F2 = 33 :=
begin
  sorry
end

end hyperbola_problem_l643_643995


namespace price_per_pound_of_rocks_l643_643253

def number_of_rocks : ℕ := 10
def average_weight_per_rock : ℝ := 1.5
def total_amount_made : ℝ := 60

theorem price_per_pound_of_rocks:
  (total_amount_made / (number_of_rocks * average_weight_per_rock)) = 4 := 
by
  sorry

end price_per_pound_of_rocks_l643_643253


namespace repeating_decimal_sum_numerator_denominator_l643_643727

open Rat

theorem repeating_decimal_sum_numerator_denominator : 
  let x := 0.\overline{134} in
  let f := { ratNum := 134, ratDenom := 999, H := by norm_num } in -- Define the fraction 134/999
  Rat.add f.num f.denom = 1133 := 
by sorry

end repeating_decimal_sum_numerator_denominator_l643_643727


namespace pencil_sharpening_time_l643_643775

theorem pencil_sharpening_time (t : ℕ) :
  let hand_crank_rate := 45
  let electric_rate := 20
  let sharpened_by_hand := (60 * t) / hand_crank_rate
  let sharpened_by_electric := (60 * t) / electric_rate
  (sharpened_by_electric = sharpened_by_hand + 10) → 
  t = 6 :=
by
  intros hand_crank_rate electric_rate sharpened_by_hand sharpened_by_electric h
  sorry

end pencil_sharpening_time_l643_643775


namespace num_ways_to_distribute_balls_l643_643212

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end num_ways_to_distribute_balls_l643_643212


namespace units_digit_of_expression_l643_643491

noncomputable def C : ℝ := 7 + Real.sqrt 50
noncomputable def D : ℝ := 7 - Real.sqrt 50

theorem units_digit_of_expression (C D : ℝ) (hC : C = 7 + Real.sqrt 50) (hD : D = 7 - Real.sqrt 50) : 
  ((C ^ 21 + D ^ 21) % 10) = 4 :=
  sorry

end units_digit_of_expression_l643_643491


namespace sum_of_intersection_coordinates_l643_643812

noncomputable def f : ℝ → ℝ := sorry
def intersect_points : { p : ℝ × ℝ // f p.1 = f (p.1 - 4) } :=
  sorry

theorem sum_of_intersection_coordinates :
  let (a, b) := intersect_points.val in a + b = 6 :=
begin
  sorry
end

end sum_of_intersection_coordinates_l643_643812


namespace max_h_value_on_01_range_b_two_roots_max_a_graph_above_l643_643188

/-- Proof Problem (1) -/
theorem max_h_value_on_01 (a : ℝ) (b : ℝ) (h : ℝ → ℝ) :
  a = -4 ∧ b = 1 - (a / 2) ∧ (∀ x, h x = exp x * ((-2) * x + 1)) → 
  ∃ x : ℝ, h x = 2 * exp (1 / 2) ∧ x ∈ set.Icc (0 : ℝ) (1 : ℝ) :=
sorry

/-- Proof Problem (2) -/
theorem range_b_two_roots (a b : ℝ) :
  a = 4 →
  (∃ b : ℝ, (2 - 2 * real.log 2 < b ∧ b ≤ 1) ∧ 
  (∀ x, exp x = 2 * x + b → x ∈ set.Icc (0 : ℝ) (2 : ℝ))) :=
sorry

/-- Proof Problem (3) -/
theorem max_a_graph_above (a : ℕ) (h : ℝ → ℝ) :
  (∀ x, f x > g x) ∧ b = -15 / 2 ∧ 2.71 < real.exp (1 : ℝ) ∧ real.exp (1 : ℝ) < 2.72 →
  ∃ a : ℕ, a = 14 :=
sorry

end max_h_value_on_01_range_b_two_roots_max_a_graph_above_l643_643188


namespace residue_S_mod_2010_l643_643617

theorem residue_S_mod_2010 :
  let S := ((list.range(2010)).map (λ n, if n % 2 = 0 then -n else n)).sum in
  S % 2010 = 1005 :=
by
  sorry

end residue_S_mod_2010_l643_643617


namespace maximum_expr_value_l643_643993

noncomputable def maximum_expr (a b c : ℝ) : ℝ :=
  a + 3 * b + 9 * c

theorem maximum_expr_value :
  ∃ Γ : ℚ, (Γ = (13 / 3)) ∧
  (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
  (Real.log (a + b + c) / Real.log 30 = 
   Real.log (3 * a) / Real.log 8 ∧
   Real.log (3 * b) / Real.log 27 ∧
   Real.log (3 * c) / Real.log 125) →
  maximum_expr a b c ≤ Γ) :=
sorry

end maximum_expr_value_l643_643993


namespace find_value_of_f_l643_643338

noncomputable def f : ℝ → ℝ := λ x, if x ≤ 1 then 2^x - 2 else -Real.log (x + 1) / Real.log 2

theorem find_value_of_f (a : ℝ) (h : f a = -3) : f (5 - a) = -7 / 4 :=
  sorry

end find_value_of_f_l643_643338


namespace part_one_part_two_l643_643191

noncomputable theory

/-
Given the parabola y^2 = 2*p*x that passes through the fixed point C(1,2),
and considering any point A on the parabola different from point C,
the line AC intersects the line y = x + 3 at point P,
and drawing a line parallel to the x-axis through point P that intersects the parabola at point B:
- Prove that the line AB passes through a fixed point
- Find the minimum value of the area of triangle ABC.
-/

variables (p : ℝ) (A P B : ℝ × ℝ) (A_ne_C : A ≠ (1,2))

def parabola_eq (x y : ℝ) : Prop := y^2 = 2 * p * x

def passes_through_C (x y : ℝ) : Prop := parabola_eq p x y ∧ (x = 1 → y = 2)

def A_on_parabola (A : ℝ × ℝ) : Prop := parabola_eq p A.1 A.2

def P_on_line_AC (A P : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, P.2 = 2 + k * (P.1 - 1) ∧ P.2 = P.1 + 3

def B_on_parabola (B : ℝ × ℝ) (P : ℝ × ℝ) : Prop := 
  B.1 = P.1 ∧ parabola_eq p B.1 B.2

theorem part_one (hA : A_on_parabola p A) (hP : P_on_line_AC A P) (hB : B_on_parabola B P) : 
  ∃ Q : ℝ × ℝ, Q = (3,2) ∧ passes_through_C p Q.1 Q.2 := 
sorry

theorem part_two (hA : A_on_parabola p A) (hP : P_on_line_AC A P) (hB : B_on_parabola B P) : 
  ∃ area : ℝ, area = 4 * real.sqrt 2 := 
sorry

end part_one_part_two_l643_643191


namespace min_xyz_value_l643_643518

-- Define the equilateral triangle ABC
def equilateral_triangle :=
  ∃ (A B C : Point) (AB BC CA : ℝ), AB = BC ∧ BC = CA ∧ CA = 4 ∧ ∠A = 60° ∧ ∠B = 60° ∧ ∠C = 60°

-- Define points D on BC, E on CA, F on AB with given distances
def points_def (A B C D E F : Point) :=
  (D ∈ Line BC ∧ |CD| = 1) ∧ (E ∈ Line CA ∧ |AE| = 1) ∧ (F ∈ Line AB ∧ |BF| = 1)

-- Define triangle RQS formed by intersection of AD, BE, CF
def triangle_intersection (A B C D E F R Q S : Point) :=
  ∃ (AD BE CF : Line), AD ∩ BE = R ∧ BE ∩ CF = Q ∧ CF ∩ AD = S

-- Define the distances x, y, z, from P to the sides of ABC
def distances (P : Point) (x y z : ℝ) :=
  ∃ P ∈ triangle RQS ∧ x = distance P BC ∧ y = distance P CA ∧ z = distance P AB ∧ x + y + z = 2 * sqrt(3)

-- State the proof to be with Equalities and the minimum value of xyz
theorem min_xyz_value : ∀ (A B C D E F R Q S P : Point) (x y z : ℝ),
  equilateral_triangle A B C →
  points_def A B C D E F →
  triangle_intersection A B C D E F R Q S →
  distances P x y z →
  xyz = min_xyz → xyz = (648 * sqrt(3)) / 2197 := by 
  sorry

end min_xyz_value_l643_643518


namespace complex_number_in_first_quadrant_l643_643509

open Complex

theorem complex_number_in_first_quadrant 
    (z : ℂ) 
    (h : z = (3 + I) / (1 - 3 * I) + 2) : 
    0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_number_in_first_quadrant_l643_643509


namespace bisectors_of_adjacent_supplementary_angles_perpendicular_l643_643666

theorem bisectors_of_adjacent_supplementary_angles_perpendicular
  {A B : Type} [linear_ordered_field A] [linear_ordered_field B]
  (α β : A) (shared_vertex : B)
  (h_adjacent : α + β = 180) (h_supplementary : α + β = 180)
  (h_div1 : bisector α = bisector β)
  (h_div2 : α = 90) (h_div3 : β = 90) :
  bisector α ⊥ bisector β :=
sorry

end bisectors_of_adjacent_supplementary_angles_perpendicular_l643_643666


namespace raft_capacity_l643_643706

theorem raft_capacity (total_without_life_jackets : ℕ) (reduction_with_life_jackets : ℕ)
  (people_needing_life_jackets : ℕ) (total_capacity_with_life_jackets : ℕ)
  (no_life_jackets_capacity : total_without_life_jackets = 21)
  (life_jackets_reduction : reduction_with_life_jackets = 7)
  (life_jackets_needed : people_needing_life_jackets = 8) :
  total_capacity_with_life_jackets = 14 :=
by
  -- Proof should be here
  sorry

end raft_capacity_l643_643706


namespace sum_of_fraction_numerator_denominator_l643_643837

-- Define the repeating decimal 0.474747... in terms of a fraction.
def repeating_decimal_fraction (x : ℚ) : Prop :=
  x = 47 / 99

-- Theorem to prove the sum of the numerator and denominator
theorem sum_of_fraction_numerator_denominator :
  (repeating_decimal_fraction (47 / 99)) → (47 + 99 = 146) := by
  intros h,
  sorry

end sum_of_fraction_numerator_denominator_l643_643837


namespace sequence_properties_l643_643925

noncomputable def Sn : ℕ → ℝ
| 0     := S1
| (n+1) := 4 * Sn n

theorem sequence_properties (S1 : ℝ) :
  (∀ n, Sn (n + 1) = 4 * Sn n) →
  (∃ (a : ℕ → ℝ), 
    ((∀ n, a (n + 1) = 3 * Sn n) ∧
     (is_arithmetic a ∨ (¬ is_geometric a))) := 
begin
  sorry
end

end sequence_properties_l643_643925


namespace part1_part2_l643_643506

-- Define the conditions for p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 6) <= 0
def q (x m : ℝ) : Prop := (2 - m <= x) ∧ (x <= 2 + m)

-- Proof statement for part (1)
theorem part1 (m: ℝ) : 
  (∀ x : ℝ, p x → q x m) → 4 <= m :=
sorry

-- Proof statement for part (2)
theorem part2 (x : ℝ) (m : ℝ) : 
  (m = 5) → (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → x ∈ Set.Ico (-3) (-2) ∪ Set.Ioc 6 7 :=
sorry

end part1_part2_l643_643506


namespace prob_select_n_from_polynomial_l643_643974

theorem prob_select_n_from_polynomial : 
  let word := "polynomial".toList in
  let total_letters := 10 in
  let count_n := word.count (λ c, c = 'n') in
  let probability := (count_n : ℝ) / (total_letters : ℝ) in
  probability = 1 / 10 :=
by
  sorry

end prob_select_n_from_polynomial_l643_643974


namespace sheep_count_l643_643592

-- Define the conditions
def TotalAnimals : ℕ := 200
def NumberCows : ℕ := 40
def NumberGoats : ℕ := 104

-- Define the question and its corresponding answer
def NumberSheep : ℕ := TotalAnimals - (NumberCows + NumberGoats)

-- State the theorem
theorem sheep_count : NumberSheep = 56 := by
  -- Skipping the proof
  sorry

end sheep_count_l643_643592


namespace find_x_l643_643481

theorem find_x (x : ℝ) (h : 9 ^ (Real.log x / Real.log 5) = 81) : x = 25 :=
sorry

end find_x_l643_643481


namespace arithmetic_sequence_fifth_term_l643_643148

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (a6 : a 6 = -3) 
  (S6 : S 6 = 12)
  (h_sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 1 - a 0)) / 2)
  : a 5 = -1 :=
sorry

end arithmetic_sequence_fifth_term_l643_643148


namespace range_of_a_l643_643886

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x : ℝ, if x ≤ 1 then (2*a - 1) * x + 4 else a^x

theorem range_of_a (a : ℝ) (ha : ∀ n : ℕ, 0 < n → f a n < f a (n + 1)) : 
  3 < a :=
by {
  sorry
}

end range_of_a_l643_643886


namespace total_handshakes_l643_643708

def gremlins := 30
def pixies := 12
def unfriendly_gremlins := 15
def friendly_gremlins := 15

def handshake_count : Nat :=
  let handshakes_friendly_gremlins := friendly_gremlins * (friendly_gremlins - 1) / 2
  let handshakes_friendly_unfriendly := friendly_gremlins * unfriendly_gremlins
  let handshakes_gremlins_pixies := gremlins * pixies
  handshakes_friendly_gremlins + handshakes_friendly_unfriendly + handshakes_gremlins_pixies

theorem total_handshakes : handshake_count = 690 := by
  sorry

end total_handshakes_l643_643708


namespace number_of_integers_l643_643867

theorem number_of_integers (n : ℤ) : 20 < n^2 → n^2 < 200 → (finset.Icc 5 14).card + (finset.Icc -14 -5).card = 20 := by
  sorry

end number_of_integers_l643_643867


namespace claudia_weekend_earnings_l643_643822

theorem claudia_weekend_earnings :
  (let charge := 10 in 
   let sat_kids := 20 in 
   let sun_kids := sat_kids / 2 in
   (sat_kids * charge) + (sun_kids * charge) = 300) :=
by
  sorry

end claudia_weekend_earnings_l643_643822


namespace ages_correct_l643_643638

-- Let A be Anya's age and P be Petya's age
def anya_age : ℕ := 4
def petya_age : ℕ := 12

-- The conditions
def condition1 (A P : ℕ) : Prop := P = 3 * A
def condition2 (A P : ℕ) : Prop := P - A = 8

-- The statement to be proven
theorem ages_correct : condition1 anya_age petya_age ∧ condition2 anya_age petya_age :=
by
  unfold condition1 condition2 anya_age petya_age -- Reveal the definitions
  have h1 : petya_age = 3 * anya_age := by
    sorry
  have h2 : petya_age - anya_age = 8 := by
    sorry
  exact ⟨h1, h2⟩ -- Combine both conditions into a single conjunction

end ages_correct_l643_643638


namespace trigon_identity_l643_643161

theorem trigon_identity (A B C : ℝ) (h : A + B + C = π) : 
  sin A ^ 2 + sin B ^ 2 + sin C ^ 2 - 2 * cos A * cos B * cos C = 2 := by
  sorry

end trigon_identity_l643_643161


namespace time_in_motion_is_correct_l643_643046

-- Define the variables and their properties
variable (a : ℝ) -- time in hours at which they met
variable (x y : ℝ) -- x is the boat speed, y is the river flow speed

-- Define the conditions from the problem
def boat_speed_positive : Prop := 0 < x
def river_speed_positive : Prop := 0 < y
def meeting_time_positive : Prop := 0 < a

-- Define the time they were in motion
def total_time_in_motion := a * (1 + Real.sqrt 2)

-- The target theorem we need to prove
theorem time_in_motion_is_correct
  (h1 : boat_speed_positive x)
  (h2 : river_speed_positive y)
  (h3 : meeting_time_positive a) :
  total_time_in_motion a = a * (1 + Real.sqrt 2) :=
sorry

end time_in_motion_is_correct_l643_643046


namespace relationship_abc_l643_643136

noncomputable def a := (1 / 3) ^ (1 / 3)
noncomputable def b := 3 ^ (1 / 5)
noncomputable def c := Real.log 3 (1 / 5)

theorem relationship_abc : c < a ∧ a < b := 
by 
  sorry

end relationship_abc_l643_643136


namespace curve_crosses_itself_at_point_l643_643070

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  (2 * t₁^2 + 1 = 2 * t₂^2 + 1) ∧ 
  (2 * t₁^3 - 6 * t₁^2 + 8 = 2 * t₂^3 - 6 * t₂^2 + 8) ∧ 
  2 * t₁^2 + 1 = 1 ∧ 2 * t₁^3 - 6 * t₁^2 + 8 = 8 :=
by
  sorry

end curve_crosses_itself_at_point_l643_643070


namespace solve_for_x_l643_643122

theorem solve_for_x (x : ℚ) (h : (sqrt (8 * x)) / (sqrt (4 * (x - 2))) = 3) : x = 18 / 7 :=
sorry

end solve_for_x_l643_643122


namespace group_members_before_removal_l643_643798

def original_member_count :=
  ∃ x : ℕ, x = 150 ∧
    (∃ (removed_members : ℕ) (daily_messages_per_member : ℕ) (weekly_messages : ℕ),
      removed_members = 20 ∧ 
      daily_messages_per_member = 50 ∧ 
      weekly_messages = 45500 ∧
      (let remaining_members := x - removed_members in
       let messages_per_week := daily_messages_per_member * 7 in
       remaining_members * messages_per_week = weekly_messages))

theorem group_members_before_removal :
  original_member_count :=
begin
  unfold original_member_count,
  existsi (150 : ℕ),
  split,
  { refl },
  existsi (20 : ℕ),
  existsi (50 : ℕ),
  existsi (45500 : ℕ),
  split, refl,
  split, refl,
  split, refl,
  let remaining_members := 150 - 20,
  let messages_per_week := 50 * 7,
  sorry,
end

end group_members_before_removal_l643_643798


namespace smallest_ellipse_area_l643_643065

theorem smallest_ellipse_area (a b : ℝ) :
  (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1) →
  (∀ x y, ((x - 2)^2 + y^2 = 4) ∨ ((x + 2)^2 + y^2 = 4)) →
  π * a * b = 4 * real.sqrt 2 * π :=
by
  sorry

end smallest_ellipse_area_l643_643065


namespace raft_people_with_life_jackets_l643_643703

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end raft_people_with_life_jackets_l643_643703


namespace intersection_of_M_and_N_l643_643196

def M : Set ℕ := {0, 2, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N :
  {x | x ∈ M ∧ x ∈ N} = {0, 4} := by
  sorry

end intersection_of_M_and_N_l643_643196


namespace opposite_of_neg_six_is_six_l643_643684

theorem opposite_of_neg_six_is_six : 
  ∃ (x : ℝ), (-6 + x = 0) ∧ x = 6 := by
  sorry

end opposite_of_neg_six_is_six_l643_643684


namespace AB_value_l643_643976

noncomputable def find_AB
  (AE EF FC BD : ℝ)
  (H_perpendicular : ∡A D B = 90) : ℝ :=
if h : AE = 8 ∧ EF = 4 ∧ FC = 8 ∧ BD = 15 then 17 else sorry

-- Let's state the theorem expressing the problem statement:
theorem AB_value :
  find_AB 8 4 8 15 (by {simp [Angle.eq_zero_iff_eq π, Angle.pi_eq_two_pi_div_two, Angle.pi_half]; ring}) = 17 :=
begin
  sorry
end

end AB_value_l643_643976


namespace equal_roots_implies_c_value_l643_643227

theorem equal_roots_implies_c_value (c : ℝ) 
  (h : ∃ x : ℝ, (x^2 + 6 * x - c = 0) ∧ (2 * x + 6 = 0)) :
  c = -9 :=
sorry

end equal_roots_implies_c_value_l643_643227


namespace remainder_T_mod_500_l643_643996

def digit_sum (n : ℕ) : ℕ := (n % 10) + ((n / 10) % 10) + (n / 100)

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 100) ≠ ((n / 10) % 10) ∧ 
  (n / 100) ≠ (n % 10) ∧ 
  ((n / 10) % 10) ≠ (n % 10) ∧
  2 ≤ (n / 100) ∧ (n / 100) ≤ 9

def T : ℕ := 
∑ n in {n | 100 ≤ n ∧ n < 1000 ∧ is_valid_number n}, n

theorem remainder_T_mod_500 : T % 500 = 228 :=
by
  sorry

end remainder_T_mod_500_l643_643996


namespace height_of_seventh_student_in_geometric_sequence_l643_643101

theorem height_of_seventh_student_in_geometric_sequence
    (a : ℕ → ℝ)
    (h_geom : ∀ n, a (n + 1) = a n * (a 1 / a 0))
    (h_a4 : a 4 = 1.5)
    (h_a10 : a 10 = 1.62) :
    a 7 = sqrt 2.43 :=
by
  sorry

end height_of_seventh_student_in_geometric_sequence_l643_643101


namespace lottery_buying_100_may_not_win_l643_643029

-- Definitions corresponding to conditions in a)
def total_tickets : ℕ := 100000
def win_probability : ℝ := 0.01
def tickets_bought : ℕ := 100

-- The main statement proving the question in c)
theorem lottery_buying_100_may_not_win : 
  (∃ (n : ℕ), (n = tickets_bought) ∧ (prob_eventually_does_not_win n = (tickets_bought ≤ total_tickets * win_probability))) :=
by 
  sorry

end lottery_buying_100_may_not_win_l643_643029


namespace trajectory_of_P_is_right_branch_of_hyperbola_l643_643973

-- Definitions of the given points F1 and F2
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Definition of the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Definition of point P satisfying the condition
def P (x y : ℝ) : Prop :=
  abs (distance (x, y) F1 - distance (x, y) F2) = 8

-- Trajectory of point P is the right branch of the hyperbola
theorem trajectory_of_P_is_right_branch_of_hyperbola :
  ∀ (x y : ℝ), P x y → True := -- Trajectory is hyperbola (right branch)
by
  sorry

end trajectory_of_P_is_right_branch_of_hyperbola_l643_643973


namespace average_age_of_community_l643_643232

theorem average_age_of_community:
  ∀ (k : ℕ) (w m : ℕ),
    w = 3 * k →
    m = 2 * k →
    (30 * w + 35 * m) / (w + m) = 32 :=
by
  intros k w m hw hm
  rw [hw, hm]
  rw [← nat.cast_mul, ← nat.cast_add, ← nat.cast_add, ← nat.cast_mul]
  norm_cast
  ring
  norm_num

end average_age_of_community_l643_643232


namespace point_translation_l643_643292

theorem point_translation :
  ∃ (x y : ℤ), x = -1 ∧ y = -2 ↔ 
  ∃ (x₀ y₀ : ℤ), 
    x₀ = -3 ∧ y₀ = 2 ∧ 
    x = x₀ + 2 ∧ 
    y = y₀ - 4 := by
  sorry

end point_translation_l643_643292


namespace cos_double_angle_l643_643895

variable (α : ℝ)

-- Define the given condition
def condition : Prop := Real.cos (π / 2 - α) = 1 / 5

-- State the theorem to prove
theorem cos_double_angle (h : condition α) : Real.cos (2 * α) = 23 / 25 := 
by
  -- We will fill in the proof steps here.
  sorry

end cos_double_angle_l643_643895


namespace initial_bottles_l643_643396

theorem initial_bottles (X : ℕ) 
  (drank : X - 25) 
  (bought : drank + 30) 
  (total : bought = 47) : 
  X = 42 :=
by
  sorry

end initial_bottles_l643_643396


namespace monotonic_intervals_magnitude_comparison_range_of_b_l643_643920

noncomputable def f (x: ℝ) : ℝ := Real.log x
noncomputable def g (a b x: ℝ) : ℝ := a * Real.sqrt x + b / Real.sqrt x

-- Problem 1: Monotonic Intervals
theorem monotonic_intervals (h : ℝ → ℝ) (a b : ℝ) (x : ℝ) : 
  a = 1 ∧ b = 1 → 
  h(x) = f(x) + g(a, b, x) →
  (∀ (x > 0), 0 < x ∧ x < 3 - 2 * Real.sqrt 2 → (∃ x, h x < 0)) ∧ 
  (∀ (x > 0), x > 3 - 2 * Real.sqrt 2 → (∃ x, h x > 0)) := 
  by sorry

-- Problem 2: Magnitude Comparison
theorem magnitude_comparison (a b : ℝ) (x : ℝ) :
  a = 1 ∧ b = -1 → 
  (f(x) > g(a, b, x) ∧ 0 < x ∧ x < 1) ∨
  (f(x) = g(a, b, x) ∧ x = 1) ∨
  (f(x) < g(a, b, x) ∧ x > 1) := 
  by sorry

-- Problem 3: Range of b
theorem range_of_b (a : ℝ) (x_0 : ℝ) :
  (∀ a > 0, ∃ x_0 > 0, f(x_0) > g(a, b, x_0)) ↔ b < 0 := 
  by sorry

end monotonic_intervals_magnitude_comparison_range_of_b_l643_643920


namespace cos_is_periodic_l643_643750

theorem cos_is_periodic : (∀ f, (f = cos → is_trigonometric f) → is_periodic f) → is_trigonometric cos → is_periodic cos :=
by
  intro ht hf
  apply ht
  intro x hx
  rw hx at hf
  exact hf
  sorry

end cos_is_periodic_l643_643750


namespace profit_percent_l643_643299

def car_cost : ℕ := 42000
def repair_cost : ℕ := 13000
def selling_price : ℕ := 64500

theorem profit_percent : 
  let total_cost := car_cost + repair_cost in
  let profit := selling_price - total_cost in
  (profit.toRat / total_cost.toRat) * 100 = 17.27 :=
by
  sorry

end profit_percent_l643_643299


namespace constant_function_on_chessboard_l643_643380

theorem constant_function_on_chessboard
  (f : ℤ × ℤ → ℝ)
  (h_nonneg : ∀ (m n : ℤ), 0 ≤ f (m, n))
  (h_mean : ∀ (m n : ℤ), f (m, n) = (f (m + 1, n) + f (m - 1, n) + f (m, n + 1) + f (m, n - 1)) / 4) :
  ∃ c : ℝ, ∀ (m n : ℤ), f (m, n) = c :=
sorry

end constant_function_on_chessboard_l643_643380


namespace exists_x_f_lt_g_l643_643187

noncomputable def f (x : ℝ) := (2 / Real.exp 1) ^ x

noncomputable def g (x : ℝ) := (Real.exp 1 / 3) ^ x

theorem exists_x_f_lt_g : ∃ x : ℝ, f x < g x := by
  sorry

end exists_x_f_lt_g_l643_643187


namespace equation_of_line_passing_through_center_and_perpendicular_l643_643530

theorem equation_of_line_passing_through_center_and_perpendicular :
  ∃ l : ℝ × ℝ × ℝ, (l.1 * x + l.2 * y + l.3 = 0) ∧
    (∃ center_x center_y r, (x^2 + y^2 - 6*y + 5 = 0) ∧
      l.1 * x + l.2 * y + l.3 = 0 ∧
      l.1 + l.2 + 1 = 0 ∧
      center_x = 0 ∧ center_y = 3 ∧ r = 2 ∧
      l = (1, -1, 3)) := by
sorry

end equation_of_line_passing_through_center_and_perpendicular_l643_643530


namespace divide_cookie_into_16_equal_parts_l643_643095

def Cookie (n : ℕ) : Type := sorry

theorem divide_cookie_into_16_equal_parts (cookie : Cookie 64) :
  ∃ (slices : List (Cookie 4)), slices.length = 16 ∧ 
  (∀ (slice : Cookie 4), slice ≠ cookie) := 
sorry

end divide_cookie_into_16_equal_parts_l643_643095


namespace gumballs_multiple_purchased_l643_643254

-- Definitions
def joanna_initial : ℕ := 40
def jacques_initial : ℕ := 60
def final_each : ℕ := 250

-- Proof statement
theorem gumballs_multiple_purchased (m : ℕ) :
  (joanna_initial + joanna_initial * m) + (jacques_initial + jacques_initial * m) = 2 * final_each →
  m = 4 :=
by 
  sorry

end gumballs_multiple_purchased_l643_643254


namespace mary_sheep_problem_l643_643286

theorem mary_sheep_problem :
  let initial_sheep := 400
  in let sheep_given_to_sister := initial_sheep / 4
  in let remaining_after_sister := initial_sheep - sheep_given_to_sister
  in let sheep_given_to_brother := remaining_after_sister / 2
  in let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  in sheep_remaining = 150 :=
by
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let remaining_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := remaining_after_sister / 2
  let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  show sheep_remaining = 150 from sorry

end mary_sheep_problem_l643_643286


namespace min_colors_for_hex_tessellation_l643_643722

-- Define the basic conditions for hexagonal tessellation coloring
def hexagon_adjacent {α : Type*} (G : SimpleGraph α) (x y : α) : Prop :=
  G.adj x y

-- The main theorem statement
theorem min_colors_for_hex_tessellation {α : Type*} (G : SimpleGraph α) :
  (∀ (x : α), ∃ (y : α), hexagon_adjacent G x y ∧ ss.card (G.neighborSet x) = 3) →
  ∃ (n : ℕ),  (n = 3) ∧ (∀ (f : α → ℕ), coloring G f n → (∀ (x y : α), hexagon_adjacent G x y → f x ≠ f y)) :=
sorry

end min_colors_for_hex_tessellation_l643_643722


namespace paint_replacement_fractions_l643_643587

variables {r b g : ℚ}

/-- Given the initial and replacement intensities and the final intensities of red, blue,
and green paints respectively, prove the fractions of the original amounts of each paint color
that were replaced. -/
theorem paint_replacement_fractions :
  (0.6 * (1 - r) + 0.3 * r = 0.4) ∧
  (0.4 * (1 - b) + 0.15 * b = 0.25) ∧
  (0.25 * (1 - g) + 0.1 * g = 0.18) →
  (r = 2/3) ∧ (b = 3/5) ∧ (g = 7/15) :=
by
  sorry

end paint_replacement_fractions_l643_643587


namespace union_area_of_reflected_triangles_l643_643432

def point : Type := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

def reflect (p : point) (x : ℝ) : point :=
  (2 * x - p.1, p.2)

theorem union_area_of_reflected_triangles :
  let A := (4, 3) in
  let B := (6, -2) in
  let C := (7, 1) in
  let A' := reflect A 6 in
  let B' := reflect B 6 in
  let C' := reflect C 6 in
  (triangle_area A B C) + (triangle_area A' B' C') = 10 :=
by
  sorry

end union_area_of_reflected_triangles_l643_643432


namespace exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l643_643782

-- Definition: A positive integer n is a perfect power if n = a ^ b for some integers a, b with b > 1.
def isPerfectPower (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ n = a^b

-- Part (a): Prove the existence of an arithmetic progression of 2004 perfect powers.
theorem exists_arithmetic_progression_2004_perfect_powers :
  ∃ (x r : ℕ), (∀ n : ℕ, n < 2004 → ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

-- Part (b): Prove that perfect powers cannot form an infinite arithmetic progression.
theorem perfect_powers_not_infinite_arithmetic_progression :
  ¬ ∃ (x r : ℕ), (∀ n : ℕ, ∃ a b : ℕ, b > 1 ∧ (x + n * r) = a^b) :=
sorry

end exists_arithmetic_progression_2004_perfect_powers_perfect_powers_not_infinite_arithmetic_progression_l643_643782


namespace boxes_filled_l643_643878

theorem boxes_filled (initial_boxes : ℕ) (unfilled_boxes : ℕ) : initial_boxes = 13 → unfilled_boxes = 5 → (initial_boxes - unfilled_boxes) = 8 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end boxes_filled_l643_643878


namespace cover_points_with_two_circles_l643_643139

theorem cover_points_with_two_circles {α : Type*} [metric_space α] {p : ℕ} (points : fin p → α)
  (h : ∀ (a b c : fin p), dist (points a) (points b) ≤ 1 ∨ dist (points b) (points c) ≤ 1 ∨ dist (points c) (points a) ≤ 1) :
  ∃ (A B : α), ∀ x, (∃ C, dist (points x) C ≤ 1) :=
sorry

end cover_points_with_two_circles_l643_643139


namespace minimum_value_y_min_value_of_function_l643_643680

theorem minimum_value_y (x : ℝ) (hx : 0 < x) : (x + 1/x) ≥ 2 :=
by
  sorry

theorem min_value_of_function (x : ℝ) (hx : 0 < x) : 
  let y := (x^2 + 1) / x in 
  y ≥ 2 :=
by
  let t := minimum_value_y x hx
  show y ≥ 2 from by { dsimp [y], exact t }

end minimum_value_y_min_value_of_function_l643_643680


namespace determinant_of_matrix_M_l643_643459

def matrix_M : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, -2], ![8, 5, -3], ![3, 3, 6]]

theorem determinant_of_matrix_M : matrix.det matrix_M = 99 :=
by
  sorry

end determinant_of_matrix_M_l643_643459


namespace ellipse_solution_l643_643176

def ellipse_eq (a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_solution :
  ∃(a b: ℝ), 
    a > 0 ∧ 
    a > b ∧ 
    b > 0 ∧
    ( ∃ e : ℝ, e = 1 / 2 ) ∧
    ( ∀x y: ℝ, (x + y - real.sqrt 6 = 0) → (x^2 + y^2 = b^2) ) →
    ( ∀l x1 y1 x2 y2 : ℝ, (x1, y1) ≠ (x2, y2) ∧ ( ∀N : ℝ * ℝ, N = (4, 0) ) → 
         ∃ m, ellipse_eq 2 (real.sqrt 3) x1 y1 ∧ ellipse_eq 2 (real.sqrt 3) x2 y2 ∧
               m ∈ (Ioo (-1/2) (0)) ∪ (Ioo (0) (1/2))) :=
by sorry

end ellipse_solution_l643_643176


namespace function_increasing_l643_643676

variable {f : ℝ → ℝ}

-- Given conditions
def is_increasing (g : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → 0 < y → x < y → g x < g y

axiom cond1 : is_increasing (λ x : ℝ, f x - x)
axiom cond2 : is_increasing (λ x : ℝ, f (x^2) - x^6)

-- Goal to prove
theorem function_increasing : is_increasing (λ x : ℝ, f (x^3) - (↑3)^(1/2) / 2 * x^6) :=
sorry

end function_increasing_l643_643676


namespace exists_triangle_l643_643614

variables {α : Type*} [affine_space α] [decidable_eq α]

open_locale affine

-- Given Sets and predicate conditions
def finite_set_points (s : set α) : Prop := set.finite s
def disjoint_sets (A B : finset α) : Prop := A ∩ B = ∅
def non_collinear_sets (S : finset α) : Prop := ∀ (x y z : α), x ≠ y → y ≠ z → x ≠ z → set.collinear ↥S {x, y, z} = false
def at_least_five_points_set (S : finset α) : Prop := S.card ≥ 5

theorem exists_triangle {A B : finset α} (h1 : disjoint_sets A B) (h2 : non_collinear_sets (A ∪ B)) 
(h3: at_least_five_points_set A ∨ at_least_five_points_set B) :
∃ T, (T ⊆ A ∨ T ⊆ B) ∧ T.card = 3 ∧ ∀ p ∈ (A ∪ B).filter (λ x, x ∉ T), ¬point_inside_triangle p T := 
sorry

end exists_triangle_l643_643614


namespace prod_four_triangle_areas_not_end_in_1988_l643_643417

theorem prod_four_triangle_areas_not_end_in_1988
  (ABCD : Type *)
  [convex A B C D]
  (S1 S2 S3 S4 : ℤ)
  (h1 : S1 > 0) (h2 : S2 > 0) (h3 : S3 > 0) (h4 : S4 > 0)
  (h5 : ∃ AO OC : ℤ, AO > 0 ∧ OC > 0 ∧ S1 * OC = S2 * AO ∧ S4 * AO = S3 * OC) :
  (S1 * S2 * S3 * S4) % 10000 ≠ 1988 :=
sorry

end prod_four_triangle_areas_not_end_in_1988_l643_643417


namespace work_done_l643_643406

theorem work_done (m : ℕ) : 18 * 30 = m * 36 → m = 15 :=
by
  intro h  -- assume the equality condition
  have h1 : m = 15 := by
    -- We would solve for m here similarly to the solution given to derive 15
    sorry
  exact h1

end work_done_l643_643406


namespace largest_angle_in_isosceles_triangle_l643_643595

variable (T : Type) [EuclideanGeometry T]

-- Define isosceles triangle and angles
def is_isosceles_triangle (A B C : T) := (∠ B A C = ∠ C B A)

-- Define the specific triangle with one angle given
def specific_isosceles_triangle (A B C : T) (angle_BAC : ℝ) (angle_CBA : ℝ) :=
  is_isosceles_triangle A B C ∧ ∠ B A C = angle_BAC ∧ ∠ C B A = angle_CBA

-- Given one angle is 50 degrees
def angle_50_degrees := 50

-- Define the hypothesis that one angle is 50 degrees
def isosceles_triangle_with_50 (A B C : T) :=
  specific_isosceles_triangle A B C angle_50_degrees angle_50_degrees

-- The theorem to prove
theorem largest_angle_in_isosceles_triangle (A B C : T) (h : isosceles_triangle_with_50 A B C) :
  ∠ A C B = 80 :=
by sorry

end largest_angle_in_isosceles_triangle_l643_643595


namespace problem_statement_l643_643327

variable {x y : ℤ}

def is_multiple_of_5 (n : ℤ) : Prop := ∃ m : ℤ, n = 5 * m
def is_multiple_of_10 (n : ℤ) : Prop := ∃ m : ℤ, n = 10 * m

theorem problem_statement (hx : is_multiple_of_5 x) (hy : is_multiple_of_10 y) :
  (is_multiple_of_5 (x + y)) ∧ (x + y ≥ 15) :=
sorry

end problem_statement_l643_643327


namespace sufficient_but_not_necessary_condition_l643_643696
-- Import the necessary library

-- Define the conditions
variables (a b : ℝ)

-- State the proof problem
theorem sufficient_but_not_necessary_condition (h1 : a < 0) (h2 : -1 < b) (h3 : b < 0) :
  (a + a * b < 0) ∧ ¬((a + a * b < 0) → (a < 0 ∧ -1 < b ∧ b < 0)) :=
by
  split
  sorry

end sufficient_but_not_necessary_condition_l643_643696


namespace count_integers_satisfying_inequality_l643_643860

theorem count_integers_satisfying_inequality : 
  (∃ n : ℤ, (20 < n^2) ∧ (n^2 < 200)) → (finset.Icc (-14 : ℤ) 14).filter (λ n, 20 < n^2 ∧ n^2 < 200).card = 20 := 
by
  sorry

end count_integers_satisfying_inequality_l643_643860


namespace central_angle_in_unit_circle_l643_643234

-- Define the conditions given in the problem
variables (R : ℝ) (AB : ℝ) (α : ℝ)
-- Condition: The circle is a unit circle
def unit_circle := R = 1
-- Condition: The length of chord AB is √3
def chord_length := AB = real.sqrt 3

-- Define the central angle α given the above conditions
theorem central_angle_in_unit_circle (h1 : unit_circle R) (h2 : chord_length AB) :
  α = 2 * real.pi / 3 :=
sorry

end central_angle_in_unit_circle_l643_643234


namespace raft_people_with_life_jackets_l643_643702

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end raft_people_with_life_jackets_l643_643702


namespace folding_paper_holes_l643_643427

/-
Conditions:
1. A rectangular sheet of paper is folded three times:
   * First from bottom to top (height halved).
   * Second from right to left (width halved).
   * Third from top to bottom (height halved again).
2. A hole is punched slightly towards the top right of the folded rectangle.

Question:
Prove that after unfolding the paper completely, there will be eight holes symmetrically distributed.
-/

theorem folding_paper_holes :
  ∀ (paper : Type) (hole_position : Type),
    let folded_once_h := paper → paper
    let folded_once_w := paper → paper
    let folded_twice_h := paper → paper
    let punch_hole := hole_position → hole_position in
    ∃ (num_holes : ℕ), num_holes = 8 ∧ symmetrical_hole_distribution hole_position num_holes :=
begin
  sorry
end

end folding_paper_holes_l643_643427


namespace population_triples_in_approximately_55_years_l643_643346

noncomputable def annual_growth_rate : ℝ := 51 / 50
noncomputable def target_population_ratio : ℝ := 3

theorem population_triples_in_approximately_55_years :
  ∃ n : ℕ, n ≈ 55 ∧ (annual_growth_rate ^ n) = target_population_ratio :=
by
  sorry

end population_triples_in_approximately_55_years_l643_643346


namespace people_in_room_l643_643709

theorem people_in_room (P C : ℚ) (H1 : (3 / 5) * P = (2 / 3) * C) (H2 : C / 3 = 5) : 
  P = 50 / 3 :=
by
  -- The proof would go here
  sorry

end people_in_room_l643_643709


namespace num_articles_produced_l643_643938

-- Conditions
def production_rate (x : ℕ) : ℕ := 2 * x^3 / (x * x * 2 * x)
def articles_produced (y : ℕ) : ℕ := y * 2 * y * y * production_rate y

-- Proof: Given the production rate, prove the number of articles produced.
theorem num_articles_produced (y : ℕ) : articles_produced y = 2 * y^3 := by sorry

end num_articles_produced_l643_643938


namespace problem_1_problem_2_l643_643178

noncomputable def f (x : ℝ) (a : ℝ) := Real.sqrt (a - x^2)

-- First proof problem statement: 
theorem problem_1 (a : ℝ) (x : ℝ) (A B : Set ℝ) (h1 : a = 4) (h2 : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) (h3 : B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) : 
  (A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) :=
sorry

-- Second proof problem statement:
theorem problem_2 (a : ℝ) (h : 1 ∈ {y : ℝ | 0 ≤ y ∧ y ≤ Real.sqrt a}) : a ≥ 1 :=
sorry

end problem_1_problem_2_l643_643178


namespace cos_x_sqrt_3_div_3_l643_643502

theorem cos_x_sqrt_3_div_3 (x : ℝ) (h : cos^2 (x / 2 + π / 4) = cos (x + π / 6)) : cos x = sqrt 3 / 3 :=
by
  sorry

end cos_x_sqrt_3_div_3_l643_643502


namespace log_sum_of_geometric_sequence_l643_643887

noncomputable def geometric_sequence_log_sum : Prop :=
  ∃ (a : ℕ → ℝ) (r : ℝ), 
    (∀ n, a n = a 0 * r ^ n) ∧
    (a 5 * a 6 + a 2 * a 9 = 18) ∧
    (∑ i in finset.range 10, real.log_base 3 (a i) = 10)

theorem log_sum_of_geometric_sequence : geometric_sequence_log_sum := sorry

end log_sum_of_geometric_sequence_l643_643887


namespace cd_length_is_19_l643_643597

theorem cd_length_is_19
  (AB BD BC : ℝ) (h1 : AB = 7) (h2 : BD = 15) (h3 : BC = 9)
  (alpha beta : ℝ) (h4 : ∠BAD = alpha) (h5 : ∠ADC = alpha)
  (h6 : ∠ABD = beta) (h7 : ∠BCD = beta) :
  ∃ m n : ℕ, (CD = m / n ∧ (m + n = 19)) :=
begin
  -- Mathematical proof here
  sorry
end

end cd_length_is_19_l643_643597


namespace smallest_n_l643_643575

-- Define the conditions
def is_sum_of_consecutive_numbers (n k : Nat) : Prop :=
  ∃ (a : Nat), n = k * (2 * a + k - 1) / 2

def is_natural_number (n : Nat) : Prop :=
  n ≥ 0

-- Lean statement for the smallest N 
theorem smallest_n (N : Nat) :
  is_natural_number N →
  is_sum_of_consecutive_numbers N 3 →
  is_sum_of_consecutive_numbers N 11 →
  is_sum_of_consecutive_numbers N 12 →
  N = 66 :=
begin
  sorry
end

end smallest_n_l643_643575


namespace right_angled_triangle_inradius_one_l643_643679

theorem right_angled_triangle_inradius_one (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b > c) (h₅ : a + c > b) (h₆ : b + c > a) (hinrange : inradius a b c = 1) :
  is_right_angle a b c :=
sorry

noncomputable def inradius (a b c : ℕ) : ℕ := 
  let s := (a + b + c) / 2
  let area := Math.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s

-- Definition of right-angle
def is_right_angle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

end right_angled_triangle_inradius_one_l643_643679


namespace intersection_of_A_and_B_l643_643194

def A : Set ℝ := {x | x ∈ {-3, -1, 0, 1, 2, 3, 4}}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := sorry

end intersection_of_A_and_B_l643_643194


namespace rectangle_area_increase_l643_643426

theorem rectangle_area_increase
  (l w : ℝ)
  (h₀ : l > 0) -- original length is positive
  (h₁ : w > 0) -- original width is positive
  (length_increase : l' = 1.3 * l) -- new length after increase
  (width_increase : w' = 1.15 * w) -- new width after increase
  (new_area : A' = l' * w') -- new area after increase
  (original_area : A = l * w) -- original area
  :
  ((A' / A) * 100 - 100) = 49.5 := by
  sorry

end rectangle_area_increase_l643_643426


namespace area_G1_G2_G3_is_4_l643_643615

open Classical

noncomputable def incenter := { A B C : Type } ->
  let I := sorry in
  I

noncomputable def centroid (A B C : Type) := { I : Type } ->
  let G := sorry in
  G

noncomputable def area (ABC : Type) := 36

theorem area_G1_G2_G3_is_4 :
  let A B C I G1 G2 G3 G_1_G2_G_3 := sorry in
  area AABC = 36 →
  G1 = centroid I B C →
  G2 = centroid I C A →
  G3 = centroid I A B →
  area G_1_G2_G_3 = 4 :=
by
  intros
  sorry

end area_G1_G2_G3_is_4_l643_643615


namespace choose_five_out_of_ten_l643_643935

theorem choose_five_out_of_ten : (nat.choose 10 5) = 252 := 
by {
  -- The proof would go here, but we will use sorry to skip the actual proof steps.
  sorry
}

end choose_five_out_of_ten_l643_643935


namespace possible_values_a_l643_643550

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 - a * x - 7 else a / x

theorem possible_values_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → -2 * x - a ≥ 0) ∧
  (∀ x : ℝ, x > 1 → -a / (x^2) ≥ 0) ∧
  (-8 - a ≤ a) →
  a = -2 ∨ a = -3 ∨ a = -4 :=
sorry

end possible_values_a_l643_643550


namespace min_value_expression_l643_643170

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = ( (x + 1) * (2 * y + 1) ) / (Real.sqrt (x * y)) ∧ min_val = 4 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_l643_643170


namespace num_n_for_g_multiple_of_8_l643_643873

def g (n : ℕ) : ℕ := 7 + 3*n + 2*n^2 + 5*n^3 + 3*n^4 + 2*n^5

theorem num_n_for_g_multiple_of_8 : 
  (∃ (k : ℕ), 2 ≤ k ∧ k ≤ 100 ∧ ∑ (n ∈ finset.range 101), if (g n % 8 = 0) then 1 else 0 = 12) :=
by
  sorry

end num_n_for_g_multiple_of_8_l643_643873


namespace smallest_positive_period_of_f_max_min_values_of_f_l643_643185

noncomputable def f (x : ℝ) : ℝ := sin^2 (x / 2) + sqrt 3 * sin (x / 2) * cos (x / 2)

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + 2 * π) = f x :=
by sorry

theorem max_min_values_of_f (x : ℝ) (hx : x ∈ set.Icc (π / 2) π) :
  1 ≤ f x ∧ f x ≤ 3 / 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_l643_643185


namespace BE_eq_CF_then_AB_eq_AC_BE_gt_CF_then_AB_gt_AC_BE_lt_CF_then_AB_lt_AC_l643_643243

-- Define the triangle with angle bisectors and the given conditions
variable (A B C E F : Type) [IsTriangle A B C] 
  (BE : AngleBisector A B C E) (CF : AngleBisector A C B F)

-- First statement: If BE = CF, then AB = AC
theorem BE_eq_CF_then_AB_eq_AC (h1 : BE = CF) : AB = AC := sorry

-- Second statement: If BE > CF, then AB > AC
theorem BE_gt_CF_then_AB_gt_AC (h2 : BE > CF) : AB > AC := sorry

-- Third statement: If BE < CF, then AB < AC
theorem BE_lt_CF_then_AB_lt_AC (h3 : BE < CF) : AB < AC := sorry

end BE_eq_CF_then_AB_eq_AC_BE_gt_CF_then_AB_gt_AC_BE_lt_CF_then_AB_lt_AC_l643_643243


namespace lambda_ge_one_l643_643520

variable {a1 a2 : ℝ} (ha1 : a1 > 0) (ha2 : a2 > 0)
variable {λ : ℝ} (hλ : λ > 0)

def is_solution (b1 b2 : ℝ) :=
  b1 + b2 = λ * (a1 + a2) ∧ b1 * b2 = λ * (a1 * a2)

theorem lambda_ge_one (hB : ∀ b1 b2 : ℝ, is_solution b1 b2 → b1 > 0 ∧ b2 > 0) :
  λ ≥ 1 :=
sorry

end lambda_ge_one_l643_643520


namespace rectangular_x_value_l643_643841

theorem rectangular_x_value (x : ℝ)
  (h1 : ∀ (length : ℝ), length = 4 * x)
  (h2 : ∀ (width : ℝ), width = x + 10)
  (h3 : ∀ (length width : ℝ), length * width = 2 * (2 * length + 2 * width))
  : x = (Real.sqrt 41 - 1) / 2 :=
by
  sorry

end rectangular_x_value_l643_643841


namespace find_angleA_l643_643230

variables {A B C : Type}
variables [C : ℝ] [b : ℝ] [angleB : ℝ] [angleA : ℝ]

-- Given Conditions
def given_condition_1 : C = real.sqrt 2 := sorry
def given_condition_2 : angleB = real.pi / 4 := sorry
def given_condition_3 : b = 2 := sorry

-- Question: prove angle A equals 7π/12
theorem find_angleA (h1 : given_condition_1) (h2 : given_condition_2) (h3 : given_condition_3) :
  angleA = 7 * real.pi / 12 :=
sorry

end find_angleA_l643_643230


namespace final_value_even_numbers_0_to_1014_l643_643344

theorem final_value_even_numbers_0_to_1014 :
  ∀ S : Finset ℕ,
  (S = Finset.range 1 1025) →
  (∀ i, ∃ t : Finset ℕ, (t.card = S.card / 2) ∧
    (∀ x ∈ t, ∃ a b ∈ S, x = abs (a - b))) →
  ∃ a, a ∈ S ∧ (a % 2 = 0) ∧ (a <= 1014) :=
begin
  sorry
end

end final_value_even_numbers_0_to_1014_l643_643344


namespace train_length_l643_643793

theorem train_length (speed_kmph : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmph = 60 →
  time_s = 3 →
  length_m = 50.01 :=
by
  sorry

end train_length_l643_643793


namespace arithmetic_sequence_first_term_l643_643237

theorem arithmetic_sequence_first_term (d : ℤ) (a_n a_2 a_9 a_11 : ℤ) 
  (h1 : a_2 = 7) 
  (h2 : a_11 = a_9 + 6)
  (h3 : a_11 = a_n + 10 * d)
  (h4 : a_9 = a_n + 8 * d)
  (h5 : a_2 = a_n + d) :
  a_n = 4 := by
  sorry

end arithmetic_sequence_first_term_l643_643237


namespace problem_statement_l643_643260

-- Defining the conditions of the problem
def is_coprime_with_6 (a : ℕ) : Prop := Nat.gcd a 6 = 1

noncomputable def proof_problem (n : ℕ) (A : set ℕ) : Prop :=
  (0 < n) ∧
  A.card = 8 * n + 1 ∧
  ∀ a ∈ A, is_coprime_with_6 a ∧ a < 30 * n ∧
  ∃ a b ∈ A, a ≠ b ∧ a ∣ b

-- The main theorem statement
theorem problem_statement 
  {n : ℕ} {A : set ℕ} (hn : 0 < n) 
  (hA_card : A.card = 8 * n + 1) 
  (hA_prop : ∀ a ∈ A, is_coprime_with_6 a ∧ a < 30 * n)
  : ∃ a b ∈ A, a ≠ b ∧ a ∣ b := sorry

end problem_statement_l643_643260


namespace simplify_expression1_simplify_expression2_l643_643817

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l643_643817


namespace rhombus_condition_perimeter_rhombus_given_ab_l643_643201

noncomputable def roots_of_quadratic (m : ℝ) : Set ℝ :=
{ x : ℝ | x^2 - m * x + m / 2 - 1 / 4 = 0 }

theorem rhombus_condition (m : ℝ) : 
  (∃ ab ad : ℝ, ab ∈ roots_of_quadratic m ∧ ad ∈ roots_of_quadratic m ∧ ab = ad) ↔ m = 1 :=
by
  sorry

theorem perimeter_rhombus_given_ab (m : ℝ) (ab : ℝ) (ad : ℝ) : 
  ab = 2 →
  (ab ∈ roots_of_quadratic m) →
  (ad ∈ roots_of_quadratic m) →
  ab ≠ ad →
  m = 5 / 2 →
  2 * (ab + ad) = 5 :=
by
  sorry

end rhombus_condition_perimeter_rhombus_given_ab_l643_643201


namespace incorrect_statement_is_B_l643_643733

theorem incorrect_statement_is_B :
  ((0 : polynomial ℚ).degree = 0) ∧
  ¬((8 : polynomial ℚ).degree = 5) ∧
  (coeff (-(1/5 : ℚ) * X * Y) 0 1 = -1/5) ∧
  ((X * (Y^2) - 2 * (X^2) + X - 1).degree = 3) →
  B = (2^3 * X^2) ↔ false :=
by { sorry }

end incorrect_statement_is_B_l643_643733


namespace fill_tank_time_l643_643363

theorem fill_tank_time : 
  let rate_A := 1 / 3
      rate_B := 1 / 4
      rate_C := 1 / 5
      rate_X := 1 / 8
      rate_Y := 1 / 4 in
  let combined_rate := rate_A + rate_B + rate_C - rate_X - rate_Y in
  combined_rate ≠ 0 →
  1 / combined_rate = 120 / 49 :=
by
  sorry

end fill_tank_time_l643_643363


namespace length_O_D1_l643_643246

-- Definitions for the setup of the cube and its faces, the center of the sphere, and the intersecting circles
def O : Point := sorry -- Center of the sphere and cube
def radius : ℝ := 10 -- Radius of the sphere

-- Intersection circles with given radii on specific faces of the cube
def r_ADA1D1 : ℝ := 1 -- Radius of the intersection circle on face ADA1D1
def r_A1B1C1D1 : ℝ := 1 -- Radius of the intersection circle on face A1B1C1D1
def r_CDD1C1 : ℝ := 3 -- Radius of the intersection circle on face CDD1C1

-- Distances derived from the problem
def OX1_sq : ℝ := radius^2 - r_ADA1D1^2
def OX2_sq : ℝ := radius^2 - r_A1B1C1D1^2
def OX_sq : ℝ := radius^2 - r_CDD1C1^2

-- To simplify, replace OX1, OX2, and OX with their squared values directly
def OX1_sq_calc : ℝ := 99
def OX2_sq_calc : ℝ := 99
def OX_sq_calc : ℝ := 91

theorem length_O_D1 : (OX1_sq_calc + OX2_sq_calc + OX_sq_calc) = 289 ↔ OD1 = 17 := by
  sorry

end length_O_D1_l643_643246


namespace calculate_average_probability_l643_643028

theorem calculate_average_probability :
  let frequencies := [0.957, 0.963, 0.956, 0.961, 0.962] in
  let avg := (frequencies.sum / frequencies.length) in
  Float.round avg 2 = 0.96 :=
by 
  let frequencies := [0.957, 0.963, 0.956, 0.961, 0.962]
  let avg := (frequencies.sum / frequencies.length)
  have : Float.round avg 2 = 0.96 := sorry
  exact this

end calculate_average_probability_l643_643028


namespace regular_hexagon_area_l643_643647

-- Define the coordinates of points A and C
def A : (ℝ × ℝ) := (0, 0)
def C : (ℝ × ℝ) := (8, 2)

-- Function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2)

-- Define the length of AC
def AC : ℝ := distance A C

-- Define the side length of the regular hexagon
def side_length (AC : ℝ) : ℝ :=
  AC / 2

-- Define the area of an equilateral triangle given its side length
def area_equilateral_triangle (s : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * s^2

-- Define the area of the regular hexagon
def area_hexagon (s : ℝ) : ℝ :=
  6 * area_equilateral_triangle s

-- Main theorem statement
theorem regular_hexagon_area : 
  area_hexagon (side_length AC) = (51 * real.sqrt 3) / 2 :=
by
  let A := (0, 0)
  let C := (8, 2)
  let AC := distance A C
  let s := side_length AC
  let area := area_hexagon s
  have h1 : distance A C = real.sqrt 68 := by sorry
  have h2 : side_length (real.sqrt 68) = real.sqrt 17 := by sorry
  have h3 : area_hexagon (real.sqrt 17) = (51 * real.sqrt 3) / 2 := by sorry
  show area = (51 * real.sqrt 3) / 2 from sorry

end regular_hexagon_area_l643_643647


namespace find_set_A_l643_643261

theorem find_set_A :
  ∃ (a1 a2 a3 a4 : ℤ), 
    ({a1 + a2 + a3, a1 + a2 + a4, a1 + a3 + a4, a2 + a3 + a4} = {-1, 3, 5, 8} : set ℤ) ∧ 
    ({a1, a2, a3, a4} = {-3, 0, 2, 6} : set ℤ) :=
sorry

end find_set_A_l643_643261


namespace max_apartments_in_building_no_13_apartments_in_building_apartments_with_two_lights_on_first_floor_possible_arrangements_of_9_apartments_l643_643345

section BuildingApartments

-- Define the basic building structure and conditions
def is_apartment {n : ℕ} (window_config : Fin n → Bool) : Prop :=
  ∀ i, window_config i → (i = 0 ∨ window_config (i - 1)) ∧ (i = n - 1 ∨ window_config (i + 1))

-- Conditionally defining the building characteristics
def building_has_properties : Prop :=
  (∀ f : Fin 5, ∃ win : Fin 5 → Bool, is_apartment win) ∧
  (∀ (a : ℕ), a > 0 → a ≤ 25) ∧
  (∀ (a : ℕ), (∃ (m : ℕ), m * a = 25) → ¬ (a = 13))

-- Proving the maximum number of apartments is 25
theorem max_apartments_in_building : building_has_properties → (∃ a, a = 25) :=
by
  intros h
  sorry

-- Proving that having 13 apartments is not possible
theorem no_13_apartments_in_building : building_has_properties → ¬ (∃ a, a = 13) :=
by
  intros h
  sorry

-- Proving that with two apartments with lights on the first floor, the total is 15
theorem apartments_with_two_lights_on_first_floor : 
  (∀ win : Fin 5 → Bool, (win 0 = true ∧ win 1 = true)) → (∃ a, a = 15) :=
by
  intros h
  sorry

-- Enumerating all possible arrangements of 9 apartments with lights on
theorem possible_arrangements_of_9_apartments : 
  (∀ win : Fin 5 → Bool, (count_ones win = 9)) → (∃ configs : Fin 16, is_valid_config configs) :=
by
  intros h
  sorry

end BuildingApartments

end max_apartments_in_building_no_13_apartments_in_building_apartments_with_two_lights_on_first_floor_possible_arrangements_of_9_apartments_l643_643345


namespace contractor_engaged_days_l643_643040

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l643_643040


namespace correct_equation_l643_643393

theorem correct_equation : -(-5) = |-5| :=
by
  -- sorry is used here to skip the actual proof steps which are not required.
  sorry

end correct_equation_l643_643393


namespace length_of_second_train_l643_643397

def first_train_length : ℝ := 290
def first_train_speed_kmph : ℝ := 120
def second_train_speed_kmph : ℝ := 80
def cross_time : ℝ := 9

noncomputable def first_train_speed_mps := (first_train_speed_kmph * 1000) / 3600
noncomputable def second_train_speed_mps := (second_train_speed_kmph * 1000) / 3600
noncomputable def relative_speed := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance_covered := relative_speed * cross_time
noncomputable def second_train_length := total_distance_covered - first_train_length

theorem length_of_second_train : second_train_length = 209.95 := by
  sorry

end length_of_second_train_l643_643397


namespace lineup_choices_count_l643_643632

def football_team := {member : Type} -- 12 members
def strong_members := {strong_member : Type} -- 4 strong enough members
def positions := {quarterback : Type, running_back : Type, offensive_lineman : strong_members, wide_receiver : Type, tight_end : Type }

theorem lineup_choices_count (strong_count : ℕ) (total_count : ℕ) : 
  strong_count = 4 → total_count = 12 → 
  ∃ choices : ℕ, choices = 4 * 11 * 10 * 9 * 8 ∧ choices = 31680 :=
by
  intros h1 h2,
  use (4 * 11 * 10 * 9 * 8),
  split,
  { refl },
  { norm_num }

end lineup_choices_count_l643_643632


namespace katie_sold_4_bead_necklaces_l643_643256

theorem katie_sold_4_bead_necklaces :
  ∃ (B : ℕ), 
    (∃ (G : ℕ), G = 3) ∧ 
    (∃ (C : ℕ), C = 3) ∧ 
    (∃ (T : ℕ), T = 21) ∧ 
    B * 3 + 3 * 3 = 21 :=
sorry

end katie_sold_4_bead_necklaces_l643_643256


namespace num_cars_in_parking_lot_l643_643594

-- Define the conditions
variable (C : ℕ) -- Number of cars
def number_of_bikes := 5 -- Number of bikes given
def total_wheels := 66 -- Total number of wheels given
def wheels_per_bike := 2 -- Number of wheels per bike
def wheels_per_car := 4 -- Number of wheels per car

-- Define the proof statement
theorem num_cars_in_parking_lot 
  (h1 : total_wheels = 66) 
  (h2 : number_of_bikes = 5) 
  (h3 : wheels_per_bike = 2)
  (h4 : wheels_per_car = 4) 
  (h5 : C * wheels_per_car + number_of_bikes * wheels_per_bike = total_wheels) :
  C = 14 :=
by
  sorry

end num_cars_in_parking_lot_l643_643594


namespace problem_1_problem_2_l643_643854

-- Proof problem statement for Problem 1
theorem problem_1 (x y : ℝ) : 
  (∃ P : ℝ × ℝ, P = ⟨1, 2⟩ ∧ l x y P.1 P.2 2 1 4) :=
begin
  sorry
end

-- Conditions for Problem 1
def l1 (x y : ℝ) := x + 3 * y - 3 = 0
def l2 (x y : ℝ) := x - y + 1 = 0
def l_parallel (x y : ℝ) := 2 * x + y - 3 = 0

-- Proof problem statement for Problem 2
theorem problem_2 (x y : ℝ) : 
  (∃ A : ℝ × ℝ, A = ⟨1, -1⟩ ∧ l A.1 A.2 0 1 1 ∨ l A.1 A.2 3 4 1) :=
begin
  sorry
end

-- Conditions for Problem 2
def l3 (x y : ℝ) := 2 * x + y - 6 = 0
def A : ℝ × ℝ := (1, -1)
def distance (A B : ℝ × ℝ) := 
  sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 5

end problem_1_problem_2_l643_643854


namespace max_possible_value_l643_643304

noncomputable def y (x : ℕ → ℝ) (k : ℕ) : ℝ :=
(1 / k) * (∑ i in Finset.range k, x i)

theorem max_possible_value (x : ℕ → ℝ) (h_condition : ∑ k in Finset.range 2000, |x k - x (k + 1)| = 2001) :
  ∑ k in Finset.range 2000, |y x (k + 1) - y x k| ≤ 2000 :=
sorry

end max_possible_value_l643_643304


namespace triangle_sine_roots_l643_643929

noncomputable def triangle_condition (p q : ℝ) : Prop :=
  (p^2 - 2*q = 1) ∧ (-real.sqrt 2 ≤ p ∧ p < -1) ∧ (0 < q ∧ q ≤ 1/2)

theorem triangle_sine_roots (p q : ℝ):
  (triangle_condition p q) →
  ∃ A B : ℝ, (0 < A ∧ A < π/2) ∧ (0 < B ∧ B < π/2) ∧ 
  (A + B = π/2) ∧
  (sin A) * (sin B) = q ∧ 
  (sin A + sin B = -p) ∧ 
  (∀ x : ℝ, x^2 + p*x + q = 0 ↔ x = sin A ∨ x = sin B) := by
  sorry

end triangle_sine_roots_l643_643929


namespace work_together_time_l643_643776

-- Define the work rate of A and B
def A_rate : ℝ := 1 / 24
def B_rate : ℝ := A_rate / 3
def Combined_rate : ℝ := A_rate + B_rate

-- Prove the combined time taken for A and B to complete the work together
theorem work_together_time :
  1 / Combined_rate = 18 :=
by
  -- Steps of the proof to be filled in
  sorry

end work_together_time_l643_643776


namespace jerry_age_proof_l643_643626

variable (J : ℝ)

/-- Mickey's age is 4 years less than 400% of Jerry's age. Mickey is 18 years old. Prove that Jerry is 5.5 years old. -/
theorem jerry_age_proof (h : 18 = 4 * J - 4) : J = 5.5 :=
by
  sorry

end jerry_age_proof_l643_643626


namespace correct_proposition_is_D_l643_643799

noncomputable def proposition_A : Prop := ∀ (l₁ l₂ : ℝ → ℝ³), (∀ t, l₁ t ∥ l₂ t) → (parallel_projection l₁ ∥ parallel_projection l₂)
noncomputable def proposition_B : Prop := ∀ (P₁ P₂ : ℝ³ → Prop), (∃ l : ℝ → ℝ³, (∀ x, P₁ (l x)) ∧ (∀ y, P₂ (l y))) → P₁ = P₂
noncomputable def proposition_C : Prop := ∀ (P₁ P₂ : ℝ³ → Prop), (∃ p : ℝ³ → ℝ, (∀ x, ⟂ P₁ (p x)) ∧ (∀ y, ⟂ P₂ (p y))) → P₁ = P₂
noncomputable def proposition_D : Prop := ∀ (l₁ l₂ : ℝ → ℝ³), (∃ P : ℝ³ → Prop, (∀ t, ⟂ l₁ t P) ∧ (∀ t, ⟂ l₂ t P)) → l₁ ∥ l₂

theorem correct_proposition_is_D : proposition_D :=
by
  intros
  -- Proof would go here
  sorry

end correct_proposition_is_D_l643_643799


namespace trig_identity_l643_643179

theorem trig_identity (f : ℝ → ℝ) (ϕ : ℝ) (h₁ : ∀ x, f x = 2 * Real.sin (2 * x + ϕ)) (h₂ : 0 < ϕ) (h₃ : ϕ < π) (h₄ : f 0 = 1) :
  f ϕ = 2 :=
sorry

end trig_identity_l643_643179


namespace common_ratio_geo_seq_l643_643511

def geo_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a i

theorem common_ratio_geo_seq 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (S : ℕ → ℝ := geo_sum a) 
  (h : 2 * S 4 = S 5 + S 6) : 
  q = -2 := sorry

end common_ratio_geo_seq_l643_643511


namespace equilateral_triangle_of_given_conditions_l643_643969

open EuclideanGeometry

-- We start by defining an acute-angled triangle and required points
variable (A B C F M : Point)
variable {ABC : Triangle A B C}
variable {acute : acute_triangle A B C}

-- Given Conditions
variable (h_CF_altitude : altitude A B C F)
variable (h_BM_median : median A B C B M)
variable (h_BM_eq_CF : BM.distance = CF.distance)
variable (h_MBC_eq_FCA : ∠(M, B, C) = ∠(F, C, A))

-- Prove that triangle ABC is equilateral
theorem equilateral_triangle_of_given_conditions :
  equilateral_triangle A B C :=
by
  sorry

end equilateral_triangle_of_given_conditions_l643_643969


namespace at_least_one_nonnegative_l643_643138

-- Define the eight real numbers
variables {a b c d e f g h : ℝ}

theorem at_least_one_nonnegative :
  ∃ (i : Fin 6), 
    match i with
    | ⟨0, _⟩ => a * c + b * d
    | ⟨1, _⟩ => a * e + b * f
    | ⟨2, _⟩ => a * g + b * h
    | ⟨3, _⟩ => c * e + d * f
    | ⟨4, _⟩ => c * g + d * h
    | ⟨5, _⟩ => e * g + f * h
    end ≥ 0 := by sorry

end at_least_one_nonnegative_l643_643138


namespace claudia_weekend_earnings_l643_643823

theorem claudia_weekend_earnings :
  (let charge := 10 in 
   let sat_kids := 20 in 
   let sun_kids := sat_kids / 2 in
   (sat_kids * charge) + (sun_kids * charge) = 300) :=
by
  sorry

end claudia_weekend_earnings_l643_643823


namespace series_sum_equals_9_over_4_l643_643460

noncomputable def series_sum : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

theorem series_sum_equals_9_over_4 :
  series_sum = 9 / 4 :=
sorry

end series_sum_equals_9_over_4_l643_643460


namespace region_covered_by_P_area_l643_643621

theorem region_covered_by_P_area :
  let O : Point
      A : Point
      B : Point
      C : Point
      P : Point
      AB := 3
      AC := 4
      BC := 5
      incenter (ΔABC) O
      position_vector (P) := x • (O - A) + y • (O - B) + z • (O - C)
      (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) 
   in
  area (trajectory P) = 12 := sorry

end region_covered_by_P_area_l643_643621


namespace line_slope_product_l643_643399

theorem line_slope_product (x y : ℝ) (h1 : (x, 6) = (x, 6)) (h2 : (10, y) = (10, y)) (h3 : ∀ x, y = (1 / 2) * x) : x * y = 60 :=
sorry

end line_slope_product_l643_643399


namespace ratio_HC_JE_l643_643294

-- Definitions and Theorems based on the given conditions
axiom points {A B C D E F G H J : Type}
  (on_AF : ∃ {P Q R S T U}, P = A ∧ Q = B ∧ R = C ∧ S = D ∧ T = E ∧ U = F)
  (divided_segments : ∀ {x y : ∀ {P : Prop} → X P → X P}, let seg_length = 1 in
    distance x y = seg_length)
  (G_not_on_AF : ¬ (G = A ∨ G = B ∨ G = C ∨ G = D ∨ G = E ∨ G = F))
  (H_on_GD : H ∈ line G D)
  (J_on_GF : J ∈ line G F)
  (parallel_HC_JE_AG : parallel (line H C) (line J E) ∧ parallel (line J E) (line A G))

-- The proof statement
theorem ratio_HC_JE : ∀ {HC JE : ℝ},
  parallel (line A G) (line H C) → parallel (line A G) (line J E) →
  ratio_of_distances H C J E = 7 / 4 := by
  sorry

end ratio_HC_JE_l643_643294


namespace trigonometric_expression_value_l643_643842

theorem trigonometric_expression_value : 
  cos (-25/3 * Real.pi) + sin (25/6 * Real.pi) - tan (-25/4 * Real.pi) = 2 := 
by sorry

end trigonometric_expression_value_l643_643842


namespace max_S_n_value_l643_643355

variable {α : Type*} [LinearOrderedField α]

def S_n (a : ℕ → α) (n : ℕ) : α := (finset.range n).sum a

theorem max_S_n_value (a : ℕ → α) (h1 : a 1 = 13) (h2 : S_n a 3 = S_n a 11) :
  ∃ (n : ℕ), S_n a n = S_n a 7 :=
sorry

end max_S_n_value_l643_643355


namespace divisor_function_identity_l643_643870

-- Definitions based on conditions
def divisors (n : ℕ) : Finset ℕ := (Finset.range (n + 1)).filter (λ x, x > 0 ∧ n % x = 0)

def num_divisors (n : ℕ) : ℕ := (divisors n).card

noncomputable def sum_divisors_func (n : ℕ) (f : ℕ → ℕ) : ℕ := (divisors n).sum f

theorem divisor_function_identity {n : ℕ} : 
  let d := num_divisors in sum_divisors_func n (λ k, d k ^ 3) = sum_divisors_func n d ^ 2 := 
by
  sorry

end divisor_function_identity_l643_643870


namespace find_pairs_l643_643715

/-
Define the conditions:
1. The number of three-digit phone numbers consisting of only odd digits.
2. The number of three-digit phone numbers consisting of only even digits excluding 0.
3. Revenue difference is given by a specific equation.
4. \(X\) and \(Y\) are integers less than 250.
-/
def N₁ : ℕ := 5 * 5 * 5  -- Number of combinations with odd digits (1, 3, 5, 7, 9)
def N₂ : ℕ := 4 * 4 * 4  -- Number of combinations with even digits (2, 4, 6, 8)

-- Main theorem: finding pairs (X, Y) that satisfy the given conditions.
theorem find_pairs (X Y : ℕ) (hX : X < 250) (hY : Y < 250) :
  N₁ * X - N₂ * Y = 5 ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) := 
by {
  sorry
}

end find_pairs_l643_643715


namespace number_in_center_is_seven_l643_643059

theorem number_in_center_is_seven 
  (A1 A2 A3 A4 A5 A6 A7 A8 A9 : ℕ)
  (positions : (A1, A2, A3, A4, A5, A6, A7, A8, A9) ∈ ({(n₁, n₂, n₃, n₄, n₅, n₆, n₇, n₈, n₉) | 
    (Set.of_finset {1, 2, 3, 4, 5, 6, 7, 8, 9}) ⊆ {n₁, n₂, n₃, n₄, n₅, n₆, n₇, n₈, n₉} ∧ 
    (∀ i j, n₁ ≤ n₂ → abs(i - j) = 1 → abs(n₁ - n₂) = 1) ∧ 
    (A1 + A3 + A7 + A9 = 18)
  } )) 
  : A5 = 7 := 
  sorry

end number_in_center_is_seven_l643_643059


namespace bryan_books_total_l643_643449

theorem bryan_books_total (books_per_shelf : ℕ) (num_shelves : ℕ) 
  (h : books_per_shelf = 56) (h2 : num_shelves = 9) : 
  (books_per_shelf * num_shelves = 504) :=
by
  rw [h, h2]
  norm_num

end bryan_books_total_l643_643449


namespace find_f_2_l643_643908

-- Define the function f and the condition it satisfies
def f (x : ℝ) : ℝ := 1 + f (1/2) * Real.logBase 2 x

-- State the main theorem to be proved
theorem find_f_2 : f 2 = 3 / 2 := 
  sorry  -- Proof is omitted

end find_f_2_l643_643908


namespace vector_magnitude_l643_643926

def a : ℝ × ℝ := (1, 2)

theorem vector_magnitude : |(1 : ℝ, 2 : ℝ)| = Real.sqrt 5 := by
  sorry

end vector_magnitude_l643_643926


namespace points_lost_calculation_l643_643457

variable (firstRound secondRound finalScore : ℕ)
variable (pointsLost : ℕ)

theorem points_lost_calculation 
  (h1 : firstRound = 40) 
  (h2 : secondRound = 50) 
  (h3 : finalScore = 86) 
  (h4 : pointsLost = firstRound + secondRound - finalScore) :
  pointsLost = 4 := 
sorry

end points_lost_calculation_l643_643457


namespace area_of_triangle_l643_643911

noncomputable def point := (ℝ × ℝ)

def circle_eq (M : point) : Prop :=
  let (x, y) := M in x^2 + y^2 = 100

def M1 : point := (6, 8)

theorem area_of_triangle
  (on_circle : circle_eq M1) :
  let A := (0, 0)
  let B := (100 / 3, 0)
  let O := (0, 0)
  the area (triangle O A B) = 656 / 3 :=
sorry

end area_of_triangle_l643_643911


namespace f_analytical_expression_l643_643529

-- Define f function on R given conditions
def f : ℝ → ℝ
| x if x > 0 := 3^x + 1
| 0 := 0
| x := -3^(-x) - 1 -- Note: implicitly x < 0

-- Theorem stating the equivalence of the given function to its analytical expression
theorem f_analytical_expression :
  ∀ x: ℝ, f(x) = 
    if x > 0 then 3^x + 1 
    else if x = 0 then 0 
    else -3^x - 1 :=
by 
  intro x
  sorry

end f_analytical_expression_l643_643529


namespace sum_of_digits_of_number_of_rows_l643_643433

theorem sum_of_digits_of_number_of_rows :
  ∃ N, (3 * (N * (N + 1) / 2) = 1575) ∧ (Nat.digits 10 N).sum = 8 :=
by
  sorry

end sum_of_digits_of_number_of_rows_l643_643433


namespace find_integers_l643_643113

-- Problem statement rewritten as a Lean 4 definition
theorem find_integers (a b c : ℤ) (H1 : a = 1) (H2 : b = 2) (H3 : c = 1) : 
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c :=
by
  -- The proof will be presented here
  sorry

end find_integers_l643_643113


namespace problem1_problem2_l643_643818

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l643_643818


namespace line_equation_l643_643097

theorem line_equation (t : ℝ) : 
  ∃ (x y : ℝ), (x = 3 * t + 6) ∧ (y = 6 * t - 7) → (y = 2 * x - 20) := by
  intro x y h
  cases h
  use [x, y]
  sorry

end line_equation_l643_643097


namespace certain_number_is_45_l643_643767

-- Define the variables and condition
def x : ℝ := 45
axiom h : x * 7 = 0.35 * 900

-- The statement we need to prove
theorem certain_number_is_45 : x = 45 :=
by
  sorry

end certain_number_is_45_l643_643767


namespace indoor_players_count_l643_643961

theorem indoor_players_count (T O B I : ℕ) 
  (hT : T = 400) 
  (hO : O = 350) 
  (hB : B = 60) 
  (hEq : T = (O - B) + (I - B) + B) : 
  I = 110 := 
by sorry

end indoor_players_count_l643_643961


namespace triangle_area_proof_l643_643982

noncomputable def area_of_triangle_ABC (x : ℝ) (CF BE : ℝ) : ℝ :=
  if AB = x ∧ BC = x ∧ CF = 15 ∧ BE = 15 ∧ 
     (sin (angle C B E), sin (angle D B E), sin (angle A B E)) form_arithmetic_progression ∧
     (cos (angle D B E), cos (angle C B E), cos (angle D B C)) form_geometric_progression
  then (1 / 2) * (x * sqrt 2) * (2 * x) -- Calculation of area based on x
  else 0

theorem triangle_area_proof (x : ℝ) (CF BE : ℝ) (AB BC : ℝ) :
  AB = x → BC = x → CF = 15 → BE = 15 →
  (sin (angle C B E), sin (angle D B E), sin (angle A B E)) form_arithmetic_progression →
  (cos (angle D B E), cos (angle C B E), cos (angle D B C)) form_geometric_progression →
  area_of_triangle_ABC x CF BE = 225 * sqrt 2 :=
by
  intros hAB hBC hCF hBE h_sin_prog h_cos_prog
  unfold area_of_triangle_ABC
  simp only [hAB, hBC, hCF, hBE, h_sin_prog, h_cos_prog],
  exact sorry -- Skipping the proof step

end triangle_area_proof_l643_643982


namespace remaining_pictures_l643_643297

theorem remaining_pictures (first_book : ℕ) (second_book : ℕ) (third_book : ℕ) (colored_pictures : ℕ) :
  first_book = 23 → second_book = 32 → third_book = 45 → colored_pictures = 44 →
  (first_book + second_book + third_book - colored_pictures) = 56 :=
by
  sorry

end remaining_pictures_l643_643297


namespace sum_possibilities_count_l643_643446

-- Definitions of Bag A and Bag B
def BagA := {1, 3, 5, 7}
def BagB := {2, 4, 6, 8}

-- The theorem statement
theorem sum_possibilities_count : 
  (set.univ : set ℕ) ∩ (set.image (λ (x : ℕ × ℕ), x.1 + x.2) 
  ((BagA ×ˢ BagB))) = {3, 5, 7, 9, 11, 13, 15} → 
  set.finite (set.image (λ (x : ℕ × ℕ), x.1 + x.2) 
  ((BagA ×ˢ BagB))) ∧
  set.card (set.image (λ (x : ℕ × ℕ), x.1 + x.2) 
  ((BagA ×ˢ BagB))) = 7 :=
 by 
   sorry

end sum_possibilities_count_l643_643446


namespace projection_eq_sqrt3_l643_643528

variable (a e : ℝ^3)   -- Declaring vectors a and e
variable (h_a : ‖a‖ = 2)  -- Condition 1: |a| = 2
variable (h_e : ‖e‖ = 1)  -- Condition 2: e is a unit vector
variable (h_angle : real.angle.cos (real.angle.of_real (π / 3)) = (a ⬝ e) / (‖a‖ * ‖e‖))  -- Condition 3: angle between a and e is π/3

theorem projection_eq_sqrt3 : 
  (a + e) ⬝ (a - e) / ‖a - e‖ = √3 :=
sorry

end projection_eq_sqrt3_l643_643528


namespace find_number_of_pounds_of_tomatoes_l643_643625

-- Defining the conditions as variables
variables (T : ℕ) -- the number of pounds of tomatoes

-- Given conditions
axiom price_of_tomatoes_per_pound : ℕ := 5
axiom pounds_of_apples : ℕ := 5
axiom price_of_apples_per_pound : ℕ := 6
axiom total_amount_spent : ℕ := 40

-- Total cost of apples
def cost_of_apples := pounds_of_apples * price_of_apples_per_pound

-- Total cost of tomatoes
def cost_of_tomatoes := T * price_of_tomatoes_per_pound

-- Given equation based on total amount spent
axiom total_spent_eq : cost_of_tomatoes + cost_of_apples = total_amount_spent

-- The proof statement we want to achieve
theorem find_number_of_pounds_of_tomatoes : T = 2 :=
by
  -- insert detailed steps of proof (omitted here)
  sorry

end find_number_of_pounds_of_tomatoes_l643_643625


namespace solve_y_l643_643311

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l643_643311


namespace vanya_cannot_score_100_in_multiple_exams_l643_643016

variable {r p m : ℕ}
variable (r_initial p_initial m_initial: ℕ)
variable (op_a op_b_r op_b_p op_b_m : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ))

def score_is_valid (scores : ℕ × ℕ × ℕ) : Prop := 
  scores.1 ≤ 100 ∧ scores.2 ≤ 100 ∧ scores.3 ≤ 100

def operation_a : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r + 1, p + 1, m + 1)

def operation_b_r : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r - 3, p + 1, m + 1)

def operation_b_p : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r + 1, p - 3, m + 1)

def operation_b_m : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r + 1, p + 1, m - 3)

theorem vanya_cannot_score_100_in_multiple_exams 
  (r_initial p_initial m_initial : ℕ) 
  (h1 : r_initial = m_initial - 14) 
  (h2 : p_initial = m_initial - 9) :
  ¬ ∃ (ops : List ((ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ))),
    (let final_scores := ops.foldl (λ (scores : ℕ × ℕ × ℕ) (op: ((ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ))), op scores) (r_initial, p_initial, m_initial) in
    score_is_valid final_scores ∧ (final_scores.1 = 100 ∨ final_scores.2 = 100 ∨ final_scores.3 = 100) ∧
    ((final_scores.1 = 100 ∧ final_scores.2 = 100) ∨ (final_scores.1 = 100 ∧ final_scores.3 = 100) ∨ (final_scores.2 = 100 ∧ final_scores.3 = 100))) :=
by 
  sorry

end vanya_cannot_score_100_in_multiple_exams_l643_643016


namespace max_workers_l643_643053

theorem max_workers (S a n : ℕ) (h1 : n > 0) (h2 : S > 0) (h3 : a > 0)
  (h4 : (S:ℚ) / (a * n) > (3 * S:ℚ) / (a * (n + 5))) :
  2 * n + 5 = 9 := 
by
  sorry

end max_workers_l643_643053


namespace smallest_prime_factor_in_setC_is_66_l643_643650

def setC : Set ℕ := {66, 73, 75, 79, 81}

def smallestPrimeFactor (n : ℕ) : ℕ :=
  if h : ∃ p : ℕ, p.Prime ∧ p ∣ n then
    Classical.choose h
  else
    n

theorem smallest_prime_factor_in_setC_is_66 :
  ∃ n ∈ setC, smallestPrimeFactor n = 2 ∧ 
              ∀ m ∈ setC, smallestPrimeFactor m ≥ 2 :=
by {
  sorry
}

end smallest_prime_factor_in_setC_is_66_l643_643650


namespace total_price_of_shoes_l643_643248

theorem total_price_of_shoes
  (S J : ℝ) 
  (h1 : 6 * S + 4 * J = 560) 
  (h2 : J = S / 4) :
  6 * S = 480 :=
by 
  -- Begin the proof environment
  sorry -- Placeholder for the actual proof

end total_price_of_shoes_l643_643248


namespace Heather_heavier_than_Emily_l643_643566

def Heather_weight := 87
def Emily_weight := 9

theorem Heather_heavier_than_Emily : (Heather_weight - Emily_weight = 78) :=
by sorry

end Heather_heavier_than_Emily_l643_643566


namespace numOf4DigitOddNumbers_l643_643814

/-- The set of digits available to form the numbers. -/
def digits : List ℕ := [1, 1, 2, 2, 3, 4, 5]

/-- Check if the given number is a 4-digit odd number formed by the given digits. -/
def isValid4DigitOddNumber (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000 ∧
  (n % 10 = 1 ∨ n % 10 = 3 ∨ n % 10 = 5) ∧
  List.sort (digits) = List.sort (n.digits 10)

/-- The main theorem stating the relationship. -/
theorem numOf4DigitOddNumbers : 
  Nat.card {n : ℕ // isValid4DigitOddNumber n} = 156 :=
sorry

end numOf4DigitOddNumbers_l643_643814


namespace probability_not_below_x_axis_l643_643637

open Real

def P : Point := (-1, 1)
def Q : Point := (3, -5)
def R : Point := (1, -3)
def S : Point := (-3, 3)

theorem probability_not_below_x_axis : 
  (probability (random_point_in_parallelogram P Q R S) (λ p, p.y ≥ 0)) = 1 / 4 := 
sorry

end probability_not_below_x_axis_l643_643637


namespace last_two_digits_sum_factorials_l643_643856

-- Definitions related to factorial and last two digits condition
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem last_two_digits_sum_factorials :
  last_two_digits (∑ k in (finset.range 20).filter (λ k, k ≠ 0), factorial (5 * k)) = 20 := by
  sorry

end last_two_digits_sum_factorials_l643_643856


namespace total_weight_of_parcels_l643_643288

-- Defining the weights of the parcel pairs
def weight1 := 132
def weight2 := 145
def weight3 := 150

-- Statement to prove
theorem total_weight_of_parcels : (weight1 + weight2 + weight3) / 2 = 213.5 :=
by
  sorry

end total_weight_of_parcels_l643_643288


namespace slices_with_all_toppings_l643_643758

theorem slices_with_all_toppings (p m o a b c x total : ℕ) 
  (pepperoni_slices : p = 8)
  (mushrooms_slices : m = 12)
  (olives_slices : o = 14)
  (total_slices : total = 16)
  (inclusion_exclusion : p + m + o - a - b - c - 2 * x = total) :
  x = 4 := 
by
  rw [pepperoni_slices, mushrooms_slices, olives_slices, total_slices] at inclusion_exclusion
  sorry

end slices_with_all_toppings_l643_643758


namespace angle_ADB_is_right_angle_l643_643413

open EuclideanGeometry

noncomputable def problem := sorry

theorem angle_ADB_is_right_angle {
  (r : ℝ) (A B C D : Point) (circle : Circle) (ABC : Triangle) 
  [circle.has_center C] [circle.has_radius r]
  [RightAngle (angle B C A)] [C_points : CoCircular { circle with Point}]
  [line_segment_AC_extended : ∃ D, D ∈ line_through A C] :
  angle (A D B) = 90 :=
  sorry

end angle_ADB_is_right_angle_l643_643413


namespace number_of_paths_grid_l643_643933

def paths_from_A_to_C (h v : Nat) : Nat :=
  Nat.choose (h + v) v

#eval paths_from_A_to_C 7 6 -- expected result: 1716

theorem number_of_paths_grid :
  paths_from_A_to_C 7 6 = 1716 := by
  sorry

end number_of_paths_grid_l643_643933


namespace line_equation_general_form_l643_643167

def slope : ℝ := -3
def P : ℝ × ℝ := (1, 2)

theorem line_equation_general_form :
  ∃ (a b c : ℝ), a * fst P + b * snd P + c = 0 ∧ (a = 3) ∧ (b = 1) ∧ (c = -5) :=
by
  sorry

end line_equation_general_form_l643_643167


namespace distribution_of_balls_l643_643213

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end distribution_of_balls_l643_643213


namespace figure_cyclic_permutations_l643_643435

-- Definitions of figures and initial placements
inductive Figure : Type
| red
| yellow
| green
| blue

open Figure

def initial_positions : list (ℕ × Figure) :=
[(0, red), (1, yellow), (2, green), (3, blue)]

-- Define movements, function cycles, and check the valid ending condition
def valid_move (current_pos : ℕ) (target_pos : ℕ) : bool :=
(mod (abs (target_pos - current_pos)) 12) == 4

def moves (pos : ℕ) : list ℕ :=
[pos + 4, pos - 4] |>.map (λ x, (x % 12))

def permutation_count : ℕ := 6  -- This comes from combinatorial calculation (4-1)!

-- Creating a theorem statement to show the unique cyclic permutations possible
theorem figure_cyclic_permutations : permutation_count = 6 :=
sorry

end figure_cyclic_permutations_l643_643435


namespace slopes_product_constant_l643_643171

theorem slopes_product_constant (x y : ℝ) (k1 k2 : ℝ)
  (h_circle : x^2 + y^2 = 4) 
  (h_curve : x^2 / 4 + y^2 = 1) 
  (h_A_on_C : (2 : ℝ, 0 : ℝ) ∈ C)  -- Point A lies on curve C
  (h_l_intersects_C : ∃ (m : ℝ), ∀ (B D : ℝ × ℝ), B ∈ C ∧ D ∈ C ∧ B.1 ≠ D.1 → k1 * k2 = -3 / 4) 
  : k1 * k2 = -3 / 4 :=
sorry

end slopes_product_constant_l643_643171


namespace factors_count_l643_643133

theorem factors_count (a b c d : ℕ) (h1 : prime a ∧ a = 2^3)
    (h2 : prime b ∧ b = 5^3)
    (h3 : prime c ∧ c = 3^3)
    (h4 : prime d ∧ d = 7^3):
  nat.factors (a^2 * b^3 * c^4 * d^5) = 14560 := 
by {
  sorry 
}

end factors_count_l643_643133


namespace sequence_convergence_l643_643193

variable {a : ℕ → ℝ}

def periodic_pos (n : ℕ) : Prop := (a (n+8) > 0) = (a n > 0)

def positive_count : Prop := (∃! k, (k ∈ (finset.range 8).map (λ i, if a i > 0 then 1 else 0)) ∧ k = 4)

def sequence_def (n : ℕ) : Prop := a n = 1 / n ∨ a n = -1 / n

theorem sequence_convergence (h1 : ∀ n, periodic_pos n) (h2 : positive_count) :  
  summable a ∧ (summable a → positive_count) := 
sorry

end sequence_convergence_l643_643193


namespace evaluate_f_1996_l643_643545

def f : ℕ → ℕ
| x :=
  if h : x < 2000 then
     -- Use well-founded recursion for the nested call
     have x + 8 < 2000 :=
       -- Prove that x + 8 < 2000 if x < 2000
       Nat.lt_of_lt_add' h (Nat.lt_succ_self 7),
     have f(f(x + 8))
  else x - 5

theorem evaluate_f_1996 : f 1996 = 2002 :=
by sorry

end evaluate_f_1996_l643_643545


namespace jon_awake_hours_per_day_l643_643609

def regular_bottle_size : ℕ := 16
def larger_bottle_size : ℕ := 20
def weekly_fluid_intake : ℕ := 728
def larger_bottle_daily_intake : ℕ := 40
def larger_bottle_weekly_intake : ℕ := 280
def regular_bottle_weekly_intake : ℕ := 448
def regular_bottles_per_week : ℕ := 28
def regular_bottles_per_day : ℕ := 4
def hours_per_bottle : ℕ := 4

theorem jon_awake_hours_per_day
  (h1 : jon_drinks_regular_bottle_every_4_hours)
  (h2 : jon_drinks_two_larger_bottles_daily)
  (h3 : jon_drinks_728_ounces_per_week) :
  jon_is_awake_hours_per_day = 16 :=
by
  sorry

def jon_drinks_regular_bottle_every_4_hours : Prop :=
  ∀ hours : ℕ, hours * regular_bottle_size / hours_per_bottle = 1

def jon_drinks_two_larger_bottles_daily : Prop :=
  larger_bottle_size = (regular_bottle_size * 5) / 4 ∧ 
  larger_bottle_daily_intake = 2 * larger_bottle_size

def jon_drinks_728_ounces_per_week : Prop :=
  weekly_fluid_intake = 728

def jon_is_awake_hours_per_day : ℕ :=
  regular_bottles_per_day * hours_per_bottle

end jon_awake_hours_per_day_l643_643609


namespace length_XY_l643_643602

open real

-- Let O, A, B, Y, X be points in a plane where
def O := (0, 0)
def A := (10.606, 10.606) -- arbitrary coordinate placeholders consistent with diagram properties
def B := (10.606, -10.606)
def Y := (15, 0)
def X := (10.606, 0)

-- Constants
noncomputable def radius : ℝ := 15

-- Conditions
def angle_AOB : ℝ := 90
def is_perpendicular (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p3.1 - p2.1) + (p2.2 - p1.2) * (p3.2 - p2.2) = 0
def length (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem to prove
theorem length_XY :
  is_perpendicular O Y B →
  length O Y = radius →
  angle_AOB = 90 →
  length X Y = (30 - 15 * sqrt 2) / 2 :=
by
  intros hp hlength hangle
  sorry

end length_XY_l643_643602


namespace negative_exponent_example_l643_643404

theorem negative_exponent_example : 3^(-2 : ℤ) = (1 : ℚ) / (3^2) :=
by sorry

end negative_exponent_example_l643_643404


namespace minimize_sum_find_c_l643_643090

theorem minimize_sum_find_c (a b c d e f : ℕ) (h : a + 2 * b + 6 * c + 30 * d + 210 * e + 2310 * f = 2 ^ 15) 
  (h_min : ∀ a' b' c' d' e' f' : ℕ, a' + 2 * b' + 6 * c' + 30 * d' + 210 * e' + 2310 * f' = 2 ^ 15 → 
  a' + b' + c' + d' + e' + f' ≥ a + b + c + d + e + f) :
  c = 1 :=
sorry

end minimize_sum_find_c_l643_643090


namespace max_servings_hot_chocolate_l643_643255

def servings_per_ingredient (total : ℕ) (per_serving : ℚ) : ℚ := total / per_serving * 6

theorem max_servings_hot_chocolate :
  let chocolate_servings := servings_per_ingredient 8 3
  let sugar_servings := servings_per_ingredient 3 (1/2)
  let milk_servings := servings_per_ingredient 15 6
  let vanilla_servings := servings_per_ingredient 5 (3/2)
  in
  chocolate_servings = 16 ∧ sugar_servings = 36 ∧ milk_servings = 15 ∧ vanilla_servings = 20 ∧ 
  let limiting_servings := min (min chocolate_servings sugar_servings) (min milk_servings vanilla_servings)
  in limiting_servings = 15 :=
sorry

end max_servings_hot_chocolate_l643_643255


namespace no_solution_exists_l643_643651

theorem no_solution_exists (a b : ℤ) : ∃ c : ℤ, ∀ m n : ℤ, m^2 + a * m + b ≠ 2 * n^2 + 2 * n + c :=
by {
  -- Insert correct proof here
  sorry
}

end no_solution_exists_l643_643651


namespace divisors_289n5_l643_643874

theorem divisors_289n5 (n : ℕ) (h : 210 ≤ n^3 ∧ ∃ k, k = 210n^3) : 
  ∃ d, d = 108 ∧ has_divisors (289 * n^5) d :=
by
  sorry

end divisors_289n5_l643_643874


namespace gcd_12a_20b_min_value_l643_643219

-- Define the conditions
def is_positive_integer (x : ℕ) : Prop := x > 0

def gcd_condition (a b d : ℕ) : Prop := gcd a b = d

-- State the problem
theorem gcd_12a_20b_min_value (a b : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_gcd_ab : gcd_condition a b 10) :
  ∃ (k : ℕ), k = gcd (12 * a) (20 * b) ∧ k = 40 :=
by
  sorry

end gcd_12a_20b_min_value_l643_643219


namespace spring_length_no_object_spring_length_with_5kg_spring_length_relationship_max_mass_spring_support_l643_643390

theorem spring_length_no_object : 
  (length_of_spring_with_mass 0 = 12) :=
sorry

theorem spring_length_with_5kg :
  (length_of_spring_with_mass 5 = 14.5) :=
sorry

theorem spring_length_relationship (x : ℝ) : 
  (length_of_spring_with_mass x = 0.5 * x + 12) :=
sorry

theorem max_mass_spring_support : 
  (maximum_supportable_mass = 16) :=
sorry

end spring_length_no_object_spring_length_with_5kg_spring_length_relationship_max_mass_spring_support_l643_643390


namespace problem_l643_643939

-- Define the conditions in Lean
variables {x y : ℚ}

-- Define the given conditions
def cond1 : Prop := x + y = 7 / 13
def cond2 : Prop := x - y = 1 / 91

-- Define the target theorem to be proved
theorem problem (h₁ : cond1) (h₂ : cond2) : x^2 - y^2 = 1 / 169 :=
by
  sorry

end problem_l643_643939


namespace first_pack_weight_l643_643044

variable (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
variable (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ)

theorem first_pack_weight (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
    (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ) :
    hiking_rate = 2.5 →
    hours_per_day = 9 →
    days = 7 →
    pounds_per_mile = 0.6 →
    first_resupply_percentage = 0.30 →
    second_resupply_percentage = 0.20 →
    ∃ first_pack : ℝ, first_pack = 47.25 :=
by
  intro h1 h2 h3 h4 h5 h6
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := pounds_per_mile * total_distance
  let first_resupply := total_supplies * first_resupply_percentage
  let second_resupply := total_supplies * second_resupply_percentage
  let first_pack := total_supplies - (first_resupply + second_resupply)
  use first_pack
  sorry

end first_pack_weight_l643_643044


namespace probability_passing_through_C_l643_643663

theorem probability_passing_through_C :
  ∀ (A B C : Finset (Nat × Nat)) (paths_A_to_B via paths_A_to_C plus paths_C_to_B : ℕ),
  paths_A_to_B = Nat.choose 6 3 ∧
  paths_A_to_C = Nat.choose 3 3 2 ∧
  paths_C_to_B = Nat.choose 3 3 1 →
  (paths_A_to_C * paths_C_to_B / paths_A_to_B) = 21 / 32 :=
by
  intros
  sorry

end probability_passing_through_C_l643_643663


namespace train_speed_l643_643737

theorem train_speed (train_length time_man : ℝ) (man_speed_kmh : ℝ) (train_speed_kmh : ℝ) :
  train_length = 130 →
  time_man = 6 →
  man_speed_kmh = 6 →
  train_speed_kmh = 72 :=
by
  intros h_train_length h_time_man h_man_speed_kmh
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  have h_man_speed_ms : man_speed_ms = 6 * (1000 / 3600) := by simp [h_man_speed_kmh]
  let relative_speed := train_length / time_man
  have h_relative_speed : relative_speed = 130 / 6 := by simp [h_train_length, h_time_man]
  let train_speed_ms := relative_speed - man_speed_ms
  have h_train_speed_ms : train_speed_ms = (130 / 6) - (6 * (1000 / 3600)) := by simp [h_relative_speed, h_man_speed_ms]
  let desired_train_speed_kmh := train_speed_ms * 3.6
  have h_desired_train_speed_kmh : desired_train_speed_kmh = ((130 / 6) - (6 * (1000 / 3600))) * 3.6 := by simp [h_train_speed_ms]
  have h_train_speed_kmh_calc : ((130 / 6) - (6 * (1000 / 3600))) * 3.6 = 72 := by norm_num
  exact h_train_speed_kmh_calc ▸ h_desired_train_speed_kmh.symm

end train_speed_l643_643737


namespace solve_y_l643_643312

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l643_643312


namespace distance_between_A_and_O1_is_correct_l643_643258

open EuclideanGeometry

/-- Define the length of sides of triangle ABC -/
variables (A B C : Point) (AB_length BC_length AC_length : ℝ)
  (hAB : dist A B = 5) (hBC : dist B C = 6) (hAC : dist A C = 7)

/-- Define the circumcenter O of triangle ABC -/
noncomputable def O := circumcenter A B C

/-- Define the reflections A1, B1, and C1 -/
def A1 := reflection (line_through B C) O
def B1 := reflection (line_through A C) O
def C1 := reflection (line_through A B) O

/-- Define the circumcenter O1 of triangle A1B1C1 -/
noncomputable def O1 := circumcenter A1 B1 C1

/-- Verify the distance between A and O1 -/
theorem distance_between_A_and_O1_is_correct :
  dist A O1 = 35 / (4 * real.sqrt 6) :=
sorry

end distance_between_A_and_O1_is_correct_l643_643258


namespace optimal_play_first_player_wins_l643_643714

theorem optimal_play_first_player_wins :
  ∀ (grid : ℕ × ℕ),
    grid = (19, 94) →
    (∀ turn : ℕ, ∃ size : ℕ × ℕ, size.fst = 19 ∧ size.snd ≤ 94 ∧ size.fst ≤ 19 ∧ size.snd = 94) →
    ∃ winning_strategy : (nat → (ℕ × ℕ)),
      (∀ turn : ℕ, winning_strategy turn = (18, 18) ∨
      (winning_strategy (turn + 1) = (size.snd - size.fst, size.fst - size.snd))) →
      player_wins := 1
  :=
begin
  sorry
end

end optimal_play_first_player_wins_l643_643714


namespace least_number_to_yield_multiple_of_5_l643_643721

theorem least_number_to_yield_multiple_of_5 : ∃ n : ℕ, n > 0 ∧ (879 + n) % 5 = 0 ∧ n = 1 :=
by
  use 1
  split
  · exact Nat.one_pos
  split
  · norm_num
  trivial

end least_number_to_yield_multiple_of_5_l643_643721


namespace problem_statement_l643_643903

theorem problem_statement (n : ℤ) (h : n ∈ [-1, 0, 1, 2, 3]) :
  (-(1/2)) ^ n > (-(1/5)) ^ n ↔ (n = -1) ∨ (n = 2) :=
by
  sorry

end problem_statement_l643_643903


namespace find_n_modulo_10_l643_643855

theorem find_n_modulo_10 : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [MOD 10] ∧ n = 2 := by
  sorry

end find_n_modulo_10_l643_643855


namespace possible_values_of_p_l643_643132

theorem possible_values_of_p (a b c : ℝ) (h₁ : (-a + b + c) / a = (a - b + c) / b)
  (h₂ : (a - b + c) / b = (a + b - c) / c) :
  ∃ p ∈ ({-1, 8} : Set ℝ), p = (a + b) * (b + c) * (c + a) / (a * b * c) :=
by sorry

end possible_values_of_p_l643_643132


namespace factorizable_trinomial_l643_643583

theorem factorizable_trinomial (k : ℤ) : (∃ a b : ℤ, a + b = k ∧ a * b = 5) ↔ (k = 6 ∨ k = -6) :=
by
  sorry

end factorizable_trinomial_l643_643583


namespace total_collection_amount_l643_643774

theorem total_collection_amount (n : ℕ) (member_contribution_paise : ℕ) (h₁ : n = 45) (h₂ : member_contribution_paise = 45) : (n * member_contribution_paise) / 100 = 20.25 :=
by
  sorry

end total_collection_amount_l643_643774


namespace fixed_point_l643_643872

theorem fixed_point (k : ℝ) : ∃ (a b : ℝ), (a, b) = (2, 12) ∧ ∀ k, b = 3 * a ^ 2 + k * a - 2 * k :=
by
  use [2, 12]
  sorry

end fixed_point_l643_643872


namespace rationalize_denominator_correct_l643_643301

noncomputable def rationalize_denominator (x : ℝ) := 
  (5 : ℝ) / (3 * real.cbrt 7) * (real.cbrt 49) / (real.cbrt 49)

theorem rationalize_denominator_correct : 
  rationalize_denominator (5 / (3 * real.cbrt 7)) = (5 * real.cbrt 49) / (21 : ℝ) 
  ∧ 5 + 49 + 21 = 75 := 
by
  sorry

end rationalize_denominator_correct_l643_643301


namespace mink_babies_l643_643988

theorem mink_babies (B : ℕ) (h_coats : 7 * 15 = 105)
    (h_minks: 30 + 30 * B = 210) :
  B = 6 :=
by
  sorry

end mink_babies_l643_643988


namespace pencils_sold_is_correct_l643_643690

-- Define the conditions
def first_two_students_pencils : Nat := 2 * 2
def next_six_students_pencils : Nat := 6 * 3
def last_two_students_pencils : Nat := 2 * 1
def total_pencils_sold : Nat := first_two_students_pencils + next_six_students_pencils + last_two_students_pencils

-- Prove that all pencils sold equals 24
theorem pencils_sold_is_correct : total_pencils_sold = 24 :=
by 
  -- Add the statement to be proved here
  sorry

end pencils_sold_is_correct_l643_643690


namespace proof_problem_l643_643348

def sequence_b : ℕ → ℕ
| 1 := 1
| 2 := 2
| (n+2) := sequence_b (n + 1) + sequence_b n

theorem proof_problem (n k : ℕ) (h1 : n ≥ 4) (h2 : 2 ≤ k) (h3 : k ≤ n-2):
  sequence_b n = sequence_b k * sequence_b (n - k) + sequence_b (k - 1) * sequence_b (n - k - 1) :=
sorry

end proof_problem_l643_643348


namespace circle_radius_proof_l643_643543

def circle_radius : Prop :=
  let D := -2
  let E := 3
  let F := -3 / 4
  let r := 1 / 2 * Real.sqrt (D^2 + E^2 - 4 * F)
  r = 2

theorem circle_radius_proof : circle_radius :=
  sorry

end circle_radius_proof_l643_643543


namespace total_days_to_finish_job_l643_643756

noncomputable def workers_job_completion
  (initial_workers : ℕ)
  (additional_workers : ℕ)
  (initial_days : ℕ)
  (total_days : ℕ)
  (work_completion_days : ℕ)
  (remaining_work : ℝ)
  (additional_days_needed : ℝ)
  : ℝ :=
  initial_days + additional_days_needed

theorem total_days_to_finish_job
  (initial_workers : ℕ := 6)
  (additional_workers : ℕ := 4)
  (initial_days : ℕ := 3)
  (total_days : ℕ := 8)
  (work_completion_days : ℕ := 8)
  : workers_job_completion initial_workers additional_workers initial_days total_days work_completion_days (1 - (initial_days : ℝ) / work_completion_days) (remaining_work / (((initial_workers + additional_workers) : ℝ) / work_completion_days)) = 3.5 :=
  sorry

end total_days_to_finish_job_l643_643756


namespace solve_for_y_l643_643314

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l643_643314


namespace range_of_a_l643_643547

noncomputable def f (a x : ℝ) : ℝ := a * |x| - 3 * a - 1

theorem range_of_a (a : ℝ) : 
  (∃ x₀ ∈ set.Icc (-(1 : ℝ)) (1 : ℝ), f a x₀ = 0) ↔ -1/2 ≤ a ∧ a ≤ -1/3 := 
sorry

end range_of_a_l643_643547


namespace distribute_balls_into_boxes_l643_643207

theorem distribute_balls_into_boxes : 
  ∀ (balls : ℕ) (boxes : ℕ), balls = 6 ∧ boxes = 3 → boxes^balls = 729 :=
by
  intros balls boxes h
  have hb : balls = 6 := h.1
  have hbox : boxes = 3 := h.2
  rw [hb, hbox]
  show 3^6 = 729
  exact Nat.pow 3 6 -- this would expand to the actual computation
  sorry

end distribute_balls_into_boxes_l643_643207


namespace grayson_vs_rudy_distance_l643_643565

-- Definitions based on the conditions
def grayson_first_part_distance : Real := 25 * 1
def grayson_second_part_distance : Real := 20 * 0.5
def total_grayson_distance : Real := grayson_first_part_distance + grayson_second_part_distance
def rudy_distance : Real := 10 * 3

-- Theorem stating the problem to be proved
theorem grayson_vs_rudy_distance : total_grayson_distance - rudy_distance = 5 := by
  -- Proof would go here
  sorry

end grayson_vs_rudy_distance_l643_643565


namespace area_LOM_eq_3_l643_643964

open Real

-- Defining the given problem in Lean
variable (A B C L O M : Type) [triangle : Triangle A B C]
variable (α β γ : ℝ)
variable (circumcircle : Circumcircle A B C)
variable (is_scalene : scalene_triangle A B C) 
variable (angle_condition_1 : α = β - γ) 
variable (angle_condition_2 : β = 2 * γ)
variable (angle_bisectors : PointOnCircumcircle L (AngleBisector A circumcircle) ∧ 
                                              PointOnCircumcircle O (AngleBisector B circumcircle) ∧ 
                                              PointOnCircumcircle M (AngleBisector C circumcircle))
variable (area_ABC_eq_2 : TriangleArea A B C = 2)

theorem area_LOM_eq_3 :
  TriangleArea L O M = 3 := by
  sorry

end area_LOM_eq_3_l643_643964


namespace maximum_value_of_c_l643_643144

theorem maximum_value_of_c :
  ∀ (c : ℝ) (points : ℕ → ℝ × ℝ × ℝ),
    (0 < c) →
    (∀ i : ℕ, i < 99 →
      let dist (p1 p2 : ℝ × ℝ × ℝ) := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2
      in (∃ j k : ℕ, j ≠ k ∧ j < 99 ∧ k < 99 ∧
          dist (points i) (points j) + dist (points i) (points k) ≥ (dist (points j) (points k) * c)) ) →
    c ≤ (1 + Real.sqrt 5) / 2 := 
sorry

end maximum_value_of_c_l643_643144


namespace range_and_min_value_of_function_l643_643913

noncomputable def f (a x : ℝ) := -a^(2 * x) - 2 * a^x + 1

theorem range_and_min_value_of_function (a : ℝ) (h : 1 < a) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ∧ (∃ x ∈ Icc (-2 : ℝ) 1, f a x = -7) → a = 2 :=
by sorry

end range_and_min_value_of_function_l643_643913


namespace geometric_series_first_term_l643_643802

theorem geometric_series_first_term :
  ∃ (a : ℝ), 
  let r : ℝ := -1/3 in
  let S : ℝ := 18 in
  S = a / (1 - r) ∧ a = 24 :=
by
  sorry

end geometric_series_first_term_l643_643802


namespace rectangular_prism_dimensions_l643_643382

theorem rectangular_prism_dimensions 
    (a b c : ℝ) -- edges of the rectangular prism
    (h_increase_volume : (2 * a * b = 90)) -- condition 2: increasing height increases volume by 90 cm³ 
    (h_volume_proportion : (a * (c + 2)) / 2 = (3 / 5) * (a * b * c)) -- condition 3: height change results in 3/5 of original volume
    (h_edge_relation : (a = 5 * b ∨ b = 5 * a ∨ a * b = 45)) -- condition 1: one edge 5 times longer
    : 
    (a = 0.9 ∧ b = 50 ∧ c = 10) ∨ (a = 2 ∧ b = 22.5 ∧ c = 10) ∨ (a = 3 ∧ b = 15 ∧ c = 10) :=
sorry

end rectangular_prism_dimensions_l643_643382


namespace polka_dot_blankets_given_l643_643103

variable (total_blankets : ℕ)
variable (polka_dot_fraction : ℚ)
variable (total_polka_dot_blankets_now : ℕ)

theorem polka_dot_blankets_given (h1 : total_blankets = 24) 
                                 (h2 : polka_dot_fraction = 1/3) 
                                 (h3 : total_polka_dot_blankets_now = 10) : 
  (total_polka_dot_blankets_now - nat.floor (polka_dot_fraction * total_blankets)) = 2 :=
by 
  sorry

end polka_dot_blankets_given_l643_643103


namespace triangle_area_l643_643949

theorem triangle_area {DE EF DF : ℝ} (h1 : DE = 19) (h2 : EF = 19) (h3 : DF = 30) :
  let s := (DE + EF + DF) / 2 in
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF)) in
  area = 175 :=
by
  sorry

end triangle_area_l643_643949


namespace div_add_fraction_l643_643461

theorem div_add_fraction :
  (-75) / (-25) + 1/2 = 7/2 := by
  sorry

end div_add_fraction_l643_643461


namespace ordered_pairs_count_l643_643117

noncomputable
def number_of_ordered_pairs : ℕ :=
  let pairs := {(a, b) | a^4 * b^7 = 1 ∧ a^8 * b^3 = 1}
  in pairs.to_finset.card

theorem ordered_pairs_count : number_of_ordered_pairs = 64 :=
  sorry

end ordered_pairs_count_l643_643117


namespace m_plus_n_eq_180_l643_643378

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n, n ∣ p → n = 1 ∨ n = p

def has_exactly_three_divisors (n : ℕ) : Prop :=
  ∃ p, is_prime p ∧ n = p * p

noncomputable def m : ℕ := if h : 11 > 10 ∧ is_prime 11 then 11 else 0

noncomputable def n : ℕ :=
  if h : has_exactly_three_divisors 169 ∧ 169 < 200 then 169 else 0

theorem m_plus_n_eq_180 : m + n = 180 :=
by
  -- Add the necessary statements ensuring our definitions fulfil the initial conditions
  have hm : m = 11 := by decide
  have hn : n = 169 := by decide
  rw [hm, hn]
  exact rfl

end m_plus_n_eq_180_l643_643378


namespace count_special_numbers_l643_643451

theorem count_special_numbers : 
  ∃ n, n = 56 ∧ ∀ x, 100 ≤ x ∧ x < 300 → (x.digits.length = 3 ∧ 
  ((x.digits.nth 0 = x.digits.nth 1) ∨ (x.digits.nth 1 = x.digits.nth 2) ∨ (x.digits.nth 0 = x.digits.nth 2))) :=
by {
  sorry
}

end count_special_numbers_l643_643451


namespace point_distance_inequality_l643_643014

variables (A B P Q R : Point) (d : Point → Point → ℝ)

axiom dist_nonneg : ∀ {X Y : Point}, 0 ≤ d X Y
axiom dist_comm : ∀ {X Y : Point}, d X Y = d Y X
axiom triangle_ineq : ∀ {X Y Z : Point}, d X Z ≤ d X Y + d Y Z

theorem point_distance_inequality :
  d A B + d P Q + d Q R + d R P ≤ d A P + d A Q + d A R + d B P + d B Q + d B R :=
by
  sorry

end point_distance_inequality_l643_643014


namespace average_speed_remaining_l643_643765

theorem average_speed_remaining 
  (total_distance : ℕ) (partial_distance : ℕ) (partial_speed : ℕ) 
  (total_time : ℕ) (remaining_distance : ℕ) (remaining_time : ℕ) 
  (remaining_speed : ℕ) :
  total_distance = 250 ∧ partial_distance = 220 ∧ partial_speed = 40 ∧ 
  total_time = 6 ∧ remaining_distance = total_distance - partial_distance ∧ 
  remaining_time = total_time - partial_distance / partial_speed ∧ 
  remaining_speed = remaining_distance / remaining_time → 
  remaining_speed = 60 := 
by
  intros h,
  sorry

end average_speed_remaining_l643_643765


namespace chair_and_desk_prices_l643_643687

theorem chair_and_desk_prices (c d : ℕ) 
  (h1 : c + d = 115)
  (h2 : d - c = 45) :
  c = 35 ∧ d = 80 := 
by
  sorry

end chair_and_desk_prices_l643_643687


namespace combination_10_4_eq_210_l643_643831

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the combination function using the factorial
def combination (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- State the main theorem
theorem combination_10_4_eq_210 : (combination 10 4) = 210 :=
  sorry

end combination_10_4_eq_210_l643_643831


namespace proof_cosine_commute_vectors_l643_643882

noncomputable def vector_cosine (u v : ℝ × ℝ) : ℝ :=
(u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1 ^ 2 + u.2 ^ 2) * Real.sqrt (v.1 ^ 2 + v.2 ^ 2))

theorem proof_cosine_commute_vectors : 
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (x, 1),
      b : ℝ × ℝ := (1, -2) in
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  vector_cosine (a.1 + b.1, a.2 + b.2) b = (Real.sqrt 2 / 2) := 
by
  sorry

end proof_cosine_commute_vectors_l643_643882


namespace hyperbola_equation_l643_643553

theorem hyperbola_equation 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_asymptote : (b / a) = (Real.sqrt 3 / 2))
  (c : ℝ) (hc : c = Real.sqrt 7)
  (foci_directrix_condition : a^2 + b^2 = c^2) :
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 3 = 1)) :=
by
  -- We do not provide the proof as per instructions
  sorry

end hyperbola_equation_l643_643553


namespace b_earns_more_than_a_l643_643010

-- Definitions for the conditions
def investments_ratio := (3, 4, 5)
def returns_ratio := (6, 5, 4)
def total_earnings := 10150

-- We need to prove the statement
theorem b_earns_more_than_a (x y : ℕ) (hx : 58 * x * y = 10150) : 2 * x * y = 350 := by
  -- Conditions based on ratios
  let earnings_a := 3 * x * 6 * y
  let earnings_b := 4 * x * 5 * y
  let difference := earnings_b - earnings_a
  
  -- To complete the proof, sorry is used
  sorry

end b_earns_more_than_a_l643_643010


namespace closed_form_F_l643_643405

noncomputable def f : ℝ → ℝ := sorry
noncomputable def a_kn (n k : ℕ) : ℝ := sorry
def P (n : ℕ) (x : ℝ) : ℝ := ∑ k in Finset.range (n + 1), (a_kn n k) * (x ^ k)
def F (t x : ℝ) : ℝ := ∑' n, (P n x) * (t ^ n) / (n!)

theorem closed_form_F (t x : ℝ) : F(t, x) = real.exp (x * (real.exp t - 1)) :=
sorry

end closed_form_F_l643_643405


namespace equal_angles_l643_643134

variables {P A B C D Q : Point}
variables [circle : Circle]
variables (PA tangent_to circle) (PB tangent_to circle)
variables (PCD secant_to circle)
variables (P_outside_circle : ¬(P ∈ circle))
variables (C_between_P_and_D : between P C D)
variables (on_chord_CD : Q ∈ segment C D)
variables (angle_DAQ_eq_angle_PBC : ∠ D A Q = ∠ P B C)

theorem equal_angles (h : ∠ D A Q = ∠ P B C) : ∠ D B Q = ∠ P A C :=
by sorry

end equal_angles_l643_643134


namespace square_cut_and_reassemble_l643_643381

theorem square_cut_and_reassemble (m n : ℕ) : 
  let s₁ := m^2 - n^2,
      s₂ := 2 * m * n,
      s₃ := m^2 + n^2 in
  s₁^2 + s₂^2 = s₃^2 :=
by
  sorry

end square_cut_and_reassemble_l643_643381


namespace range_AD_l643_643555

-- Definitions from the problem conditions
def parabola (x : ℝ) : ℝ := -x^2 + 2*x + 8

def B : ℝ×ℝ := (-2, 0)
def C : ℝ×ℝ := (4, 0)
def D : ℝ×ℝ := ((-2 + 4) / 2, 0)

def is_on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 = A.2
def is_above_x_axis (A : ℝ × ℝ) : Prop := A.2 > 0

-- Main theorem statement: the range of length AD
theorem range_AD (A : ℝ × ℝ) (hA1 : is_on_parabola A) (hA2 : is_above_x_axis A) (hA3 : ∠BAC < π / 2) :
  3 < dist (A, D) ∧ dist (A, D) ≤ 9 :=
sorry

end range_AD_l643_643555


namespace correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l643_643003

theorem correct_inequality:
    (-21 : ℤ) > (-21 : ℤ) := by sorry

theorem incorrect_inequality1 :
    -abs (10 + 1 / 2) < (8 + 2 / 3) := by sorry

theorem incorrect_inequality2 :
    (-abs (7 + 2 / 3)) ≠ (- (- (7 + 2 / 3))) := by sorry

theorem correct_option_d :
    (-5 / 6 : ℚ) < (-4 / 5 : ℚ) := by sorry

end correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l643_643003


namespace find_y_l643_643048

noncomputable def y_equals (x : ℝ) : ℝ := x * 5 / x

theorem find_y :
  ∃ y : ℝ, ( x ≠ 0 ∧ x = 1.4 ∧ (x * y / 5 = x^2) ) ↔ (y = 7) := 
by
  let x := 1.4
  existsi (7:ℝ)
  constructor
  intro h
  cases h with hx rest
  cases rest with hxeq hx2eq
  rw hxeq at hx2eq
  field_simp at hx2eq
  have equality := calc
    x * 7 / 5 = 1.4 * 7 / 5 : by rw hxeq
    ... = 7 * 1.4 / 5 : by ring
    ... = 1.96 : by norm_num
  have hx2 := 1.4 * 1.4 = 1.96
  linarith
  intro h
  existsi (x:ℝ)
  split
  exact ha
  repeat{ split }
  exact by sorry
  exact by sorry

end find_y_l643_643048


namespace constant_term_expansion_l643_643937

theorem constant_term_expansion :
  let n := ∫ x in 0..2, (2 * x)
  \((x - ½ / x)^n) is 3 / 2 :=
by
  let n : ℝ := ∫ x in 0..2, (2 * x)
  have h1 : n = 4 := by sorry
  have h2 : (x - ½ / x) ^ n = (x - ½ / x) ^ 4 := by sorry
  have h3 : ∀ r, (4 - 2 * r = 0) → r = 2 := by sorry
  have h4 : binomial_coeff 4 2 = 6 := by sorry
  have h5 : ((-1 / 2) ^ 2 * binomial_coeff 4 2 ) = 3 / 2 := by sorry
  show constant_term_expansion = 3 / 2 from by sorry

end constant_term_expansion_l643_643937


namespace couple_ticket_cost_l643_643370

variable (x : ℝ)

def single_ticket_cost : ℝ := 20
def total_sales : ℝ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16

theorem couple_ticket_cost :
  96 * single_ticket_cost + 16 * x = total_sales →
  x = 22.5 :=
by
  sorry

end couple_ticket_cost_l643_643370


namespace find_a_l643_643624

def points := ℝ × ℝ

def slope (p1 p2 : points) : ℝ :=
(p2.snd - p1.snd) / (p2.fst - p1.fst)

def line_l (a : ℝ) : ℝ :=
  slope (-2, 0) (0, a)

def line_ll : ℝ :=
  slope (4, 0) (6, 2)

theorem find_a : ∃ a : ℝ, line_l a = line_ll ∧ a = 2 :=
by
  have h : line_l 2 = line_ll
  { -- Calculate the slopes if necessary for proving the theorem
    calc
      line_l 2 = (2 - 0) / (0 - (-2)) : by simp [slope]
            ... = 2 / 2               : by ring
            ... = 1                   : by norm_num,
    have hll : line_ll = 1
    { calc
        line_ll = (2 - 0) / (6 - 4) : by simp [slope]
              ... = 2 / 2           : by ring
              ... = 1               : by norm_num },
    exact ⟨2, ⟨h, hll⟩⟩ }


end find_a_l643_643624


namespace sequence_a1000_l643_643590

theorem sequence_a1000 :
  (∀ n ≥ 1, (a : ℕ → ℤ), a 1 = 2010 → a 2 = 2011 → (a n + a (n + 1) + a (n + 2) = n + 3) → a 1000 = 2343) :=
by 
  sorry

end sequence_a1000_l643_643590


namespace find_f_240_l643_643692

noncomputable theory

-- Definitions
variable (f g : ℕ → ℕ)
variable F : set ℕ := { n | ∃ k, f k = n }
variable G : set ℕ := { n | ∃ k, g k = n }

axiom partition_pos_integers : ∀ n, (n ∈ F) ∨ (n ∈ G) ∧ ¬((n ∈ F) ∧ (n ∈ G))

axiom f_increasing : ∀ n m, n < m → f n < f m
axiom g_increasing : ∀ n m, n < m → g n < g m

axiom g_relation : ∀ n, g n = f (f n) + 1

-- Theorem statement
theorem find_f_240 : f 240 = 388 :=
sorry

end find_f_240_l643_643692


namespace variance_scaled_data_l643_643169

variable {α : Type} [AddCommGroup α] [Module ℝ α]

-- Assuming variances
def variance (data : List ℝ) : ℝ := sorry

theorem variance_scaled_data (a : List ℝ) (h : variance a = 1) : variance (a.map (λ x, 2 * x)) = 4 :=
sorry

end variance_scaled_data_l643_643169


namespace polynomial_satisfies_conditions_l643_643486

def fibonacci_sequence : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := fibonacci_sequence (n + 1) + fibonacci_sequence n

def P (x y : ℤ) : ℤ :=
x * (2 - ((y - x) * y - x^2)^2)

theorem polynomial_satisfies_conditions :
  (∀ n : ℤ, ∃ a b : ℤ, (P a b) = n)  → 
  (∀ x y : ℤ, 0 < P x y →  ∃ n : ℕ, P x y = fibonacci_sequence n) :=
sorry

end polynomial_satisfies_conditions_l643_643486


namespace claudia_earnings_over_weekend_l643_643827

theorem claudia_earnings_over_weekend :
  ∀ (cost_per_kid : ℕ) (saturday_kids : ℕ) (sunday_kids : ℕ),
  cost_per_kid = 10 →
  saturday_kids = 20 →
  sunday_kids = saturday_kids / 2 →
  (saturday_kids + sunday_kids) * cost_per_kid = 300 :=
by
  intros cost_per_kid saturday_kids sunday_kids
  assume h1 : cost_per_kid = 10
  assume h2 : saturday_kids = 20
  assume h3 : sunday_kids = saturday_kids / 2
  sorry

end claudia_earnings_over_weekend_l643_643827


namespace arithmetic_sequence_sum_l643_643356

theorem arithmetic_sequence_sum (a1 d : ℝ) (h1 : 3 * (a1 + d) = 15)
  (h2 : (a1 + d - 1) ^ 2 = (a1 - 1) * (a1 + 2d + 1)) : 
  let s := (10 / 2) * (2 * a1 + 9 * d) in s = 120 :=
by
  sorry

end arithmetic_sequence_sum_l643_643356


namespace sum_of_k_for_quadratic_integer_solutions_l643_643385

theorem sum_of_k_for_quadratic_integer_solutions :
  (∑ k in {k | ∃ p q : ℤ, p ≠ q ∧ 3 * p * p - k * p + 12 = 0 ∧ 3 * q * q - k * q + 12 = 0}, k) = 0 :=
by
  sorry

end sum_of_k_for_quadratic_integer_solutions_l643_643385


namespace problem_statement_l643_643544

def f (x : ℝ) : ℝ :=
  if x < 4 then log (4 - x) / log 2 else 1 + (2 ^ (x - 1))

theorem problem_statement : f 0 + f (Real.log 32 / Real.log 2) = 19 := by
  sorry

end problem_statement_l643_643544


namespace systematic_sampling_example_l643_643967

-- Definitions and conditions
def number_of_students : ℕ := 36
def sample_size : ℕ := 4
def sample_interval : ℕ := number_of_students / sample_size
def student_numbers_in_sample : set ℕ := {5, 23, 32}
def correct_answer : ℕ := 14

-- Theorem statement
theorem systematic_sampling_example (H : student_numbers_in_sample = {5, 23, 32}) :
  ∃ n, n ∉ student_numbers_in_sample ∧ n = correct_answer := by
  sorry

end systematic_sampling_example_l643_643967


namespace eval_expr_correct_l643_643846

noncomputable def eval_expr : ℝ :=
  (0.5)^(3.5) / (0.05)^3

theorem eval_expr_correct : eval_expr = 316.23 * Real.sqrt 5 :=
  sorry

end eval_expr_correct_l643_643846


namespace fourth_person_height_is_correct_l643_643362

-- Let h be the height of the first person
variable (h : ℕ)

-- Definitions based on the conditions
def second_person_height := h + 2
def third_person_height := h + 4
def fourth_person_height := h + 10

-- Condition on the average height
def height_condition := (h + second_person_height h + third_person_height h + fourth_person_height h) / 4 = 78

-- The target statement to prove
theorem fourth_person_height_is_correct
  (h : ℕ)
  (height_condition : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 78) :
  h + 10 = 84 :=
begin
  -- The proof will go here
  sorry
end

end fourth_person_height_is_correct_l643_643362


namespace value_of_expression_l643_643527

variables (m n c d : ℝ)
variables (h1 : m = -n) (h2 : c * d = 1)

theorem value_of_expression : m + n + 3 * c * d - 10 = -7 :=
by sorry

end value_of_expression_l643_643527


namespace circle_area_with_inscribed_triangle_l643_643801

theorem circle_area_with_inscribed_triangle :
  let side_length := 4
  let circumradius := (2 * real.sqrt 3 * side_length) / 3
  let area := π * circumradius^2
  area = (16 * π) / 3 :=
by
  let side_length := 4
  let circumradius := (2 * real.sqrt 3 * side_length) / 3
  let area := π * circumradius^2
  sorry

end circle_area_with_inscribed_triangle_l643_643801


namespace probability_m_ge_6_probability_mod_odd_even_not_equal_l643_643754

-- Define the faces of the tetrahedral toys
def faces : List ℕ := [1, 2, 3, 5]

-- Define all possible outcomes as the sum of pairs of faces.
def outcomes : List ℕ := List.bind faces (λ x => List.map (λ y => x + y) faces)

-- The probability calculation for the event "m >= 6".
theorem probability_m_ge_6 : (List.count (λ m => m ≥ 6) outcomes).toReal / outcomes.length.toReal = 1 / 2 := sorry

-- Check if the probabilities of "m is an odd number" and "m is an even number" are equal.
theorem probability_mod_odd_even_not_equal :
  (List.count (λ m => m % 2 = 1) outcomes).toReal / outcomes.length.toReal ≠ 
  (List.count (λ m => m % 2 = 0) outcomes).toReal / outcomes.length.toReal := sorry

end probability_m_ge_6_probability_mod_odd_even_not_equal_l643_643754


namespace simplify_expression1_simplify_expression2_l643_643815

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l643_643815


namespace count_integers_satisfying_inequality_l643_643861

theorem count_integers_satisfying_inequality : 
  (∃ n : ℤ, (20 < n^2) ∧ (n^2 < 200)) → (finset.Icc (-14 : ℤ) 14).filter (λ n, 20 < n^2 ∧ n^2 < 200).card = 20 := 
by
  sorry

end count_integers_satisfying_inequality_l643_643861


namespace vertex_in_second_quadrant_l643_643986

theorem vertex_in_second_quadrant :
  let f := λ x : ℝ, -(x + 1)^2 + 2
  let vertex_x := -1
  let vertex_y := 2
  vertex_x < 0 ∧ vertex_y > 0 :=
by
  let f := λ x : ℝ, -(x + 1)^2 + 2
  let vertex_x := -1
  let vertex_y := 2
  have h₁: vertex_x < 0 := by sorry
  have h₂: vertex_y > 0 := by sorry
  exact ⟨h₁, h₂⟩

end vertex_in_second_quadrant_l643_643986


namespace area_of_G1_G2_G3_l643_643264

theorem area_of_G1_G2_G3 (A B C P G1 G2 G3 : Point)
  (h1 : divides_into_three_equal_areas P A B C)
  (h2 : area A B C = 27)
  (g1_centroid : is_centroid G1 P B C)
  (g2_centroid : is_centroid G2 P C A)
  (g3_centroid : is_centroid G3 P A B) :
  area G1 G2 G3 = 3 :=
sorry

end area_of_G1_G2_G3_l643_643264


namespace S_PQR_equals_100_l643_643927

variables {A B C Q P R : Type} [triangle : Triangle A B C]
variables {Q_is_midpoint : Midpoint Q B C} {P_on_AC : OnLine P A C} {P_divides_AC : DividesLine P A C (3 : ℝ) (1 : ℝ)}
variables {R_on_AB : OnLine R A B}
variables {S_triangle_ABC : ℝ} (h_ABQ : S_triangle_ABC = 300)
variables {S_PQR S_RBQ : ℝ}

axiom S_PQR_equals_2_S_RBQ : S_triangle P Q R = 2 * S_triangle R B Q

theorem S_PQR_equals_100 :
  S_triangle P Q R = 100 :=
sorry

end S_PQR_equals_100_l643_643927


namespace fifth_row_first_five_positions_l643_643851

-- Define a type representing the digits including the empty space represented by 9
inductive Digit
| D2 : Digit
| D0 : Digit
| D1 : Digit
| D5 : Digit
| E9 : Digit -- empty space

-- Define the 5x5 grid with constraints
def grid : list (list Digit) :=
[
  [_, _, _, _, _],  -- Row 1
  [_, _, _, _, _],  -- Row 2
  [_, _, _, _, _],  -- Row 3
  [_, _, _, _, _],  -- Row 4
  [_, _, _, _, _]   -- Row 5
]

-- Conditions that need to be satisfied
def rowCondition (row : list Digit) : Prop :=
  -- Each row must contain exactly one of each digit 2, 0, 1, 5 and possibly empty spaces
  ∃ d2 d0 d1 d5 : list Digit, row = d2 ++ d0 ++ d1 ++ d5 ++ [Digit.E9] ∧
    d2.count (λ x, x = Digit.D2) = 1 ∧
    d0.count (λ x, x = Digit.D0) = 1 ∧
    d1.count (λ x, x = Digit.D1) = 1 ∧
    d5.count (λ x, x = Digit.D5) = 1 ∧
    (∀ x, x ∈ d2 ∨ x ∈ d0 ∨ x ∈ d1 ∨ x ∈ d5 ∨ x = Digit.E9)

def columnCondition (grid : list (list Digit)) : Prop :=
  -- Each column must contain exactly one of each digit 2, 0, 1, 5 and possibly empty spaces
  ∀ i, ∃ d2 d0 d1 d5 : list Digit,
    (grid.map (λ row, row.nth i)).filter_map id = d2 ++ d0 ++ d1 ++ d5 ++ [Digit.E9] ∧
    d2.count (λ x, x = Digit.D2) = 1 ∧
    d0.count (λ x, x = Digit.D0) = 1 ∧
    d1.count (λ x, x = Digit.D1) = 1 ∧
    d5.count (λ x, x = Digit.D5) = 1 ∧
    (∀ x, x ∈ d2 ∨ x ∈ d0 ∨ x ∈ d1 ∨ x ∈ d5 ∨ x = Digit.E9)

-- Ensure no digit is diagonally adjacent
def noDiagonalAdjacent (grid : list (list Digit)) : Prop :=
  ∀ r c, 
    (∀ d, 
      grid.nth r >>= λ row, row.nth c = some d →
      (r > 0 → c > 0 → grid.nth (r-1) >>= λ row, row.nth (c-1) ≠ some d) ∧
      (r > 0 → c < 4 → grid.nth (r-1) >>= λ row, row.nth (c+1) ≠ some d) ∧
      (r < 4 → c > 0 → grid.nth (r+1) >>= λ row, row.nth (c-1) ≠ some d) ∧
      (r < 4 → c < 4 → grid.nth (r+1) >>= λ row, row.nth (c+1) ≠ some d)
    )

-- Combine all conditions together
def validGrid (grid : list (list Digit)) : Prop :=
  (∀ row, row ∈ grid → rowCondition row) ∧
  columnCondition grid ∧
  noDiagonalAdjacent grid

-- The fifth row
def fifthRow (grid : list (list Digit)) : list Digit :=
  grid.nth 4 |>.get_or_else []

-- Problem's statement in Lean 4
theorem fifth_row_first_five_positions :
  validGrid grid →
  (fifthRow grid).take 5 = [Digit.D1, Digit.D5, Digit.E9, Digit.E9, Digit.D2] :=
sorry

end fifth_row_first_five_positions_l643_643851


namespace max_X_leq_ratio_XY_l643_643838

theorem max_X_leq_ratio_XY (x y z u : ℕ) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y z u : ℕ), (x + y = z + u) → (2 * x *y = z * u) → (x ≥ y) → m ≤ x / y :=
sorry

end max_X_leq_ratio_XY_l643_643838


namespace line_tangent_to_circle_l643_643888

noncomputable def distance_from_center_to_line (a b r : ℝ) : ℝ :=
  abs (r^2) / real.sqrt (a^2 + b^2)

theorem line_tangent_to_circle (a b r : ℝ) (h : a^2 + b^2 = r^2) (h₀ : r > 0) :
    distance_from_center_to_line a b r = r :=
by
  sorry

end line_tangent_to_circle_l643_643888


namespace max_children_l643_643514

theorem max_children (k : ℕ) : 
  (∀ (children clubs : Set (Set ℕ)), 
    (∀ c ∈ clubs, c.card ≤ 3 * k) ∧
    (∀ child ∈ children, ∃ club1 ∈ clubs, ∃ club2 ∈ clubs, ∃ club3 ∈ clubs, child ∈ club1 ∧ child ∈ club2 ∧ child ∈ club3) ∧
    (∀ child1 child2 ∈ children, ∃ club ∈ clubs, child1 ∈ club ∧ child2 ∈ club)) →
  children.card ≤ 7 * k :=
by sorry

end max_children_l643_643514


namespace sector_perimeter_l643_643532

-- Define the given conditions
def central_angle := 60 * (Real.pi / 180) -- converting 60 degrees to radians
def radius := 3

-- Define the circumference segment of the sector (arc length)
def arc_length := (central_angle / (2 * Real.pi)) * (2 * Real.pi * radius)

-- The Lean 4 theorem statement to be proven (proving the perimeter of the sector)
theorem sector_perimeter :
  (arc_length + 2 * radius) = (Real.pi + 6) :=
by
  sorry

end sector_perimeter_l643_643532


namespace maximum_value_of_product_l643_643522

-- Definitions of the foci and the ellipse.
variable (F1 F2 : ℝ × ℝ)

-- Definition of the ellipse.
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 3 = 1

-- Main statement.
theorem maximum_value_of_product (P : ℝ × ℝ) (hP : is_ellipse P.1 P.2) :
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  let dPF1 := real.sqrt (PF1.1^2 + PF1.2^2)
  let dPF2 := real.sqrt (PF2.1^2 + PF2.2^2)
  F1 = (-4, 0) ∧ F2 = (4, 0) →
  dPF1 + dPF2 = 8 →
  ∃ P, dPF1 * dPF2 = 16 := sorry

end maximum_value_of_product_l643_643522


namespace total_spent_is_64_l643_643403

/-- Condition 1: The cost of each deck is 8 dollars -/
def deck_cost : ℕ := 8

/-- Condition 2: Tom bought 3 decks -/
def tom_decks : ℕ := 3

/-- Condition 3: Tom's friend bought 5 decks -/
def friend_decks : ℕ := 5

/-- Total amount spent by Tom and his friend -/
def total_amount_spent : ℕ := (tom_decks * deck_cost) + (friend_decks * deck_cost)

/-- Proof statement: Prove that total amount spent is 64 -/
theorem total_spent_is_64 : total_amount_spent = 64 := by
  sorry

end total_spent_is_64_l643_643403


namespace solution_set_of_inequality_system_l643_643694

theorem solution_set_of_inequality_system (x : ℝ) :
  (4 * x + 1 > 2) ∧ (1 - 2 * x < 7) ↔ (x > 1 / 4) :=
by
  sorry

end solution_set_of_inequality_system_l643_643694


namespace person_age_l643_643049

variable (x : ℕ) -- Define the variable for age

-- State the condition as a hypothesis
def condition (x : ℕ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

-- State the theorem to be proved
theorem person_age (x : ℕ) (h : condition x) : x = 18 := 
sorry

end person_age_l643_643049


namespace mutual_fund_percent_change_l643_643810

theorem mutual_fund_percent_change (P : ℝ) :
  let P1 := 1.30 * P,
      P2 := P1 * 0.80,
      P3 := P2 * 1.40,
      P4 := P3 * 0.90 in
  ((P4 - P1) / P1) * 100 ≈ 0.8 :=
by
  sorry

end mutual_fund_percent_change_l643_643810


namespace inequality_holds_for_a_l643_643137

theorem inequality_holds_for_a (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + 1)^2 < Real.logb a (|x|)) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end inequality_holds_for_a_l643_643137


namespace contractor_engagement_days_l643_643037

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l643_643037


namespace sheep_remain_l643_643278

theorem sheep_remain : ∀ (total_sheep sister_share brother_share : ℕ),
  total_sheep = 400 →
  sister_share = total_sheep / 4 →
  brother_share = (total_sheep - sister_share) / 2 →
  (total_sheep - sister_share - brother_share) = 150 :=
by
  intros total_sheep sister_share brother_share h_total h_sister h_brother
  rw [h_total, h_sister, h_brother]
  sorry

end sheep_remain_l643_643278


namespace smallest_b_for_factoring_l643_643868

theorem smallest_b_for_factoring (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + (1200 : ℤ) = (x + r)*(x + s) ∧ b = r + s ∧ r * s = 1200) →
  b = 70 := 
sorry

end smallest_b_for_factoring_l643_643868


namespace value_of_b_l643_643149

theorem value_of_b (x y b : ℝ) (h1: 7^(3 * x - 1) * b^(4 * y - 3) = 49^x * 27^y) (h2: x + y = 4) : b = 3 :=
by
  sorry

end value_of_b_l643_643149


namespace straight_flush_probability_l643_643718

open Classical

noncomputable def number_of_possible_hands : ℕ := Nat.choose 52 5

noncomputable def number_of_straight_flushes : ℕ := 40 

noncomputable def probability_of_straight_flush : ℚ := number_of_straight_flushes / number_of_possible_hands

theorem straight_flush_probability :
  probability_of_straight_flush = 1 / 64974 := by
  sorry

end straight_flush_probability_l643_643718


namespace arcsin_arccos_eq_l643_643309

theorem arcsin_arccos_eq (x : ℝ) (h : Real.arcsin x + Real.arcsin (2 * x - 1) = Real.arccos x) : x = 1 := by
  sorry

end arcsin_arccos_eq_l643_643309


namespace possibleValues_set_l643_643622

noncomputable def possibleValues (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 3) : Set ℝ :=
  {x | x = 1/a + 1/b}

theorem possibleValues_set :
  ∀ a b : ℝ, (0 < a ∧ 0 < b) → (a + b = 3) → possibleValues a b (by sorry) (by sorry) = {x | ∃ y, y ≥ 4/3 ∧ x = y} :=
by
  sorry

end possibleValues_set_l643_643622


namespace fence_pole_distance_approx_equal_6_l643_643031

def path_length : ℝ := 900
def bridge_length : ℝ := 42
def total_poles : ℕ := 286

def length_lined_with_fence : ℝ := path_length - bridge_length
def poles_on_one_side : ℕ := total_poles / 2
def intervals_on_one_side : ℕ := poles_on_one_side - 1
def distance_between_poles : ℝ := length_lined_with_fence / intervals_on_one_side

theorem fence_pole_distance_approx_equal_6:
  abs (distance_between_poles - 6) < 1 :=
by
  sorry

end fence_pole_distance_approx_equal_6_l643_643031


namespace correct_statements_l643_643835

def closed_set (A : Set ℝ) : Prop :=
  ∀ a b ∈ A, a + b ∈ A ∧ a - b ∈ A

-- Given sets
def A1 : Set ℝ := {-4, -2, 0, 2, 4}
def A2 : Set ℝ := {n | ∃ k : ℤ, n = 3 * k}
def A3 : Set ℝ := {n | ∃ k : ℤ, n = 5 * k}
def A4 : Set ℝ := {n | ∃ k : ℤ, n = 2 * k}

-- Statements
def statement1 : Prop := ¬ closed_set A1
def statement2 : Prop := closed_set A2
def statement3 : Prop := ¬ closed_set (A2 ∪ A3)
def statement4 : Prop := ∃ c : ℝ, c ∉ (A2 ∪ A4)

theorem correct_statements :
  statement1 ∧ statement2 ∧ statement3 ∧ statement4 :=
by sorry

end correct_statements_l643_643835


namespace circumcircle_tangent_to_ω_l643_643639

open EuclideanGeometry
open Classical

noncomputable def problem_statement : Prop :=
  ∀ (A B C M D E X Y : Point)
  (ω : Circle)
  (h1: midpoint A B = M)
  (h2: midpoint B C = M) 
  (h3: ω.passes_through A) 
  (h4: ω.is_tangent_line BC M) 
  (h5: ω.intersects AB D)
  (h6: ω.intersects AC E) 
  (h7: midpoint B E = X) 
  (h8: midpoint C D = Y),
  circle (triangle_circumcenter M X Y) (triangle_circumradius M X Y) ∃= ω.tangent_circle_to (triangle_circumcircle M X Y)

-- Here's the statement
theorem circumcircle_tangent_to_ω (h: problem_statement) : sorry

end circumcircle_tangent_to_ω_l643_643639


namespace tangent_line_at_origin_l643_643546

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem tangent_line_at_origin :
  ∀ (x y : ℝ), (x, y) = (0, 0) → ∃ m b, y = m * x + b ∧ m = 1 ∧ b = 0 ∧ m * x + b = x := 
begin
  intros x y hxy,
  use [1, 0],
  split,
  { rw hxy,
    exact rfl },
  split,
  { exact rfl },
  split,
  { exact rfl },
  { rw hxy,
    exact rfl },
  sorry
end

end tangent_line_at_origin_l643_643546


namespace lake_area_proof_l643_643241

def satisfies_conditions (a b c d : ℕ) :=
  a^2 + c^2 = 74 ∧ b^2 + d^2 = 116 ∧ (a + b)^2 + (c + d)^2 = 370

def area_of_lake (a b c d : ℕ) :=
  let total_area := (a + b) * (c + d) / 2 in
  total_area - (7 * 5 / 2) - (10 * 4 / 2) - (7 * 4)

theorem lake_area_proof : 
  satisfies_conditions 5 4 7 10 ∧ area_of_lake 5 4 7 10 = 11 :=
by
  sorry

end lake_area_proof_l643_643241


namespace sheep_remain_l643_643280

theorem sheep_remain : ∀ (total_sheep sister_share brother_share : ℕ),
  total_sheep = 400 →
  sister_share = total_sheep / 4 →
  brother_share = (total_sheep - sister_share) / 2 →
  (total_sheep - sister_share - brother_share) = 150 :=
by
  intros total_sheep sister_share brother_share h_total h_sister h_brother
  rw [h_total, h_sister, h_brother]
  sorry

end sheep_remain_l643_643280


namespace isosceles_triangle_base_l643_643970

variable (a b : ℕ)

theorem isosceles_triangle_base 
  (h_isosceles : a = 7 ∧ b = 3)
  (triangle_inequality : 7 + 7 > 3) : b = 3 := by
-- Begin of the proof
sorry
-- End of the proof

end isosceles_triangle_base_l643_643970


namespace m_in_A_l643_643274

noncomputable def e : ℝ := Real.exp 1

def A : Set ℝ := { x | x > 2 }

def m : ℝ := Real.log (Real.exp e)

theorem m_in_A : m ∈ A := 
by
  sorry

end m_in_A_l643_643274


namespace sum_of_solutions_l643_643386

theorem sum_of_solutions (a b : ℤ) (h₁ : a = -1) (h₂ : b = -4) (h₃ : ∀ x : ℝ, (16 - 4 * x - x^2 = 0 ↔ -x^2 - 4 * x + 16 = 0)) : 
  (-b / a) = 4 := 
by 
  rw [h₁, h₂]
  norm_num
  sorry

end sum_of_solutions_l643_643386


namespace line_intersects_ellipse_l643_643353

theorem line_intersects_ellipse (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → (x^2 / 5) + (y^2 / m) = 1 → True) ↔ (1 < m ∧ m < 5) ∨ (5 < m) :=
by
  sorry

end line_intersects_ellipse_l643_643353


namespace phi_range_cos_translated_l643_643677

theorem phi_range_cos_translated (ϕ : ℝ) :
    0 < ϕ ∧ ϕ < π / 2 ∧
    (∀ x, -π / 6 ≤ x ∧ x ≤ π / 6 → ∀ x1, -π / 6 ≤ x1 ∧ x1 < x → cos (2 * x + 2 * ϕ) > cos (2 * x1 + 2 * ϕ)) ∧
    -π / 6 < (2 * x + 2 * ϕ) / 2 - k * π / 2 - π / 4 ∧
    (2 * x + 2 * ϕ) / 2 - k * π / 2 - π / 4 < 0 
    ↔ ϕ ∈ Ioc (π / 4) (π / 3) :=
    sorry

end phi_range_cos_translated_l643_643677


namespace days_worked_per_week_l643_643418

theorem days_worked_per_week (toys_per_week toys_per_day : ℕ) (h1 : toys_per_week = 5500) (h2 : toys_per_day = 1375) : toys_per_week / toys_per_day = 4 := by
  sorry

end days_worked_per_week_l643_643418


namespace volume_problem_l643_643091

-- Define the basic dimensions
def length := 6
def width := 2
def height := 3

-- Define the radius for the extensions
def radius := 1

-- Volumes calculation functions
noncomputable def parallelepipedVolume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def extendedVolume (l w h r : ℝ) : ℝ :=
  2 * (r * l * w + r * l * h + r * w * h)

noncomputable def halfCylinderVolume (l r : ℝ) : ℝ :=
  4 * (1 / 2 * π * r^2 * l)

noncomputable def sphereOctantVolume (r : ℝ) : ℝ :=
  8 * (1 / 8 * (4 / 3 * π * r^3))

noncomputable def totalVolume (l w h r : ℝ) : ℝ :=
  parallelepipedVolume l w h + extendedVolume l w h r + 
  halfCylinderVolume l r + halfCylinderVolume w r + halfCylinderVolume h r +
  sphereOctantVolume r

-- Define the given problem and the solution to be proved
theorem volume_problem :
  totalVolume length width height radius = (324 + 70 * π) / 3 :=
by
  sorry

end volume_problem_l643_643091


namespace star_vertex_angle_l643_643416

-- Defining a function that calculates the star vertex angle for odd n-sided concave regular polygon
theorem star_vertex_angle (n : ℕ) (hn_odd : n % 2 = 1) (hn_gt3 : 3 < n) : 
  (180 - 360 / n) = (n - 2) * 180 / n := 
sorry

end star_vertex_angle_l643_643416


namespace triangle_ratios_l643_643245

/-
Given a triangle ABC with points N on side AB and M on side AC,
and segments CN and BM intersecting at point O, as well as the ratios
AN:NB = 2:3 and BO:OM = 5:2, prove that the ratio CO:ON = 5:2.
-/
theorem triangle_ratios
  (A B C N M O : Type)
  [is_triangle A B C]
  (AN NB BO OM CO ON : ℝ)
  (h1 : AN / NB = 2 / 3)
  (h2 : BO / OM = 5 / 2)
  (h3 : CN (A B C N : Type) := is_side A B, ON (A B C N : Type) := is_point_on_side A B N,
  M (A B C M : Type) := is_side A C, CO (A B C O : Type) := is_point_on_side A C O)
: CO / ON = 5 / 2 := by
  sorry

end triangle_ratios_l643_643245


namespace find_p_l643_643534

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {y : ℝ × ℝ | y.2 ^ 2 = 2 * p * y.1}

noncomputable def circle : set (ℝ × ℝ) := {z : ℝ × ℝ | (z.1 - 3) ^ 2 + z.2 ^ 2 = 16}

theorem find_p (p : ℝ) (h_directrix : ∀ (x y : ℝ), y ∈ parabola p → x = -p / 2)
  (h_circle : ∀ (x y : ℝ), y ∈ circle → (y.1 - 3) ^ 2 + y.2 ^ 2 = 16) :
  ∃ p, p > 0 ∧ abs (3 + p / 2) = 4 :=
sorry

end find_p_l643_643534


namespace find_initial_quarters_l643_643630

variables {Q : ℕ} -- Initial number of quarters

def quarters_to_dollars (q : ℕ) : ℝ := q * 0.25

noncomputable def initial_cash : ℝ := 40
noncomputable def cash_given_to_sister : ℝ := 5
noncomputable def quarters_given_to_sister : ℕ := 120
noncomputable def remaining_total : ℝ := 55

theorem find_initial_quarters (Q : ℕ) (h1 : quarters_to_dollars Q + 40 = 90) : Q = 200 :=
by { sorry }

end find_initial_quarters_l643_643630


namespace log_max_value_le_neg2_l643_643898

noncomputable def log_max_value (a b c : ℝ) : ℝ :=
  if 4 * a - 2 * b + 25 * c = 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 then
    log a + log c - 2 * log b
  else
    0

theorem log_max_value_le_neg2 (a b c : ℝ) (h1 : 4 * a - 2 * b + 25 * c = 0) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  log_max_value a b c ≤ -2 := by
  sorry

end log_max_value_le_neg2_l643_643898


namespace percentage_increase_l643_643290

theorem percentage_increase (d : ℝ) (v_current v_reduce v_increase t_reduce t_increase : ℝ) (h1 : d = 96)
  (h2 : v_current = 8) (h3 : v_reduce = v_current - 4) (h4 : t_reduce = d / v_reduce) 
  (h5 : t_increase = d / v_increase) (h6 : t_reduce = t_current + 16) (h7 : t_increase = t_current - 16) :
  (v_increase - v_current) / v_current * 100 = 50 := 
sorry

end percentage_increase_l643_643290


namespace measure_of_smallest_angle_l643_643803

-- Definitions based on conditions
def isosceles_triangle (A B C : ℝ) := A = B ∨ B = C ∨ C = A
def larger_angle := 90 + (0.4 * 90)

-- Statement of the problem
theorem measure_of_smallest_angle (A B C : ℝ) (h1 : isosceles_triangle A B C) (h2 : C = larger_angle) : A = 27.0 ∨ B = 27.0 ∨ A = B :=
by
  -- provide the proof here
  sorry

end measure_of_smallest_angle_l643_643803


namespace bottles_of_soda_l643_643229

theorem bottles_of_soda (remaining_coupons blown_coupons : ℕ) (soda_per_coupon : ℕ) :
  remaining_coupons = 4 →
  blown_coupons = 3 →
  soda_per_coupon = 3 →
  (remaining_coupons + blown_coupons) * soda_per_coupon = 21 :=
by
  intros h_remaining h_blown h_soda
  rw [h_remaining, h_blown, h_soda]
  sorry

end bottles_of_soda_l643_643229


namespace edge_length_CD_is_8_l643_643350

variable (AB AC AD BC BD CD : ℕ)
variable (h_lengths : {AB, AC, AD, BC, BD, CD} = {8, 15, 17, 29, 34, 40})
variable (h_AB : AB = 40)

theorem edge_length_CD_is_8 : CD = 8 := by
  sorry

end edge_length_CD_is_8_l643_643350


namespace ladder_distance_l643_643757

theorem ladder_distance (x : ℝ) (h1 : (13:ℝ) = Real.sqrt (x ^ 2 + 12 ^ 2)) : 
  x = 5 :=
by 
  sorry

end ladder_distance_l643_643757


namespace find_b_l643_643269

noncomputable def p (x : ℝ) : ℝ := 2 * x^2 - 7
noncomputable def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

theorem find_b (b : ℝ) (h : p (q 3 b) = 31) : b = 12 + real.sqrt 19 ∨ b = 12 - real.sqrt 19 := 
  by
  sorry

end find_b_l643_643269


namespace star_points_number_l643_643766

-- Let n be the number of points in the star
def n : ℕ := sorry

-- Let A and B be the angles at the star points, with the condition that A_i = B_i - 20
def A (i : ℕ) : ℝ := sorry
def B (i : ℕ) : ℝ := sorry

-- Condition: For all i, A_i = B_i - 20
axiom angle_condition : ∀ i, A i = B i - 20

-- Total sum of angle differences equal to 360 degrees
axiom angle_sum_condition : n * 20 = 360

-- Theorem to prove
theorem star_points_number : n = 18 := by
  sorry

end star_points_number_l643_643766


namespace trajectory_of_Q_is_circle_l643_643894

noncomputable def ellipse (F1 F2 : ℝ × ℝ) (a : ℝ) : set (ℝ × ℝ) :=
  {P | dist P F1 + dist P F2 = 2 * a}

variables (F1 F2 : ℝ × ℝ) (a : ℝ)
variable {P : ℝ × ℝ}

noncomputable def point_Q (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : ℝ × ℝ :=
  let Q := (2 * (P - F1) + F1).to_prod in
  if dist P F2 = dist P Q then Q else (0, 0)

theorem trajectory_of_Q_is_circle :
  ∀ (P : ℝ × ℝ), P ∈ ellipse F1 F2 a →
  let Q := point_Q P F1 F2 in
  dist Q F1 = 2 * a :=
by
  intro P hP
  let Q := point_Q P F1 F2
  have h_dist_QF1 : dist Q F1 = 2 * a := sorry
  exact h_dist_QF1

end trajectory_of_Q_is_circle_l643_643894


namespace Vanya_can_not_score_100_in_more_than_one_exam_l643_643019

def Vanya_exams (r p m : ℕ) : Prop :=
  r = p - 5 ∧ p = m - 9 ∧ ∀ k : ℕ, ∀ op : ℕ → ℕ → ℕ → (ℕ × ℕ × ℕ),
    (op = λ r p m => (r + 1, p + 1, m + 1) ∨ 
    op = λ r p m => (r - 3, p + 1, m + 1) ∨ 
    op = λ r p m => (r + 1, p - 3, m + 1) ∨ 
    op = λ r p m => (r + 1, p + 1, m - 3)) →
    let (r', p', m') := op r p m in
    r' <= 100 ∧ p' <= 100 ∧ m' <= 100 →
    (r' = 100 ∨ p' = 100 ∨ m' = 100) → ¬(r' = 100 ∧ p' = 100 ∧ m' = 100)

theorem Vanya_can_not_score_100_in_more_than_one_exam (r p m : ℕ) : 
  Vanya_exams r p m → 
  (¬(∃ r' p' m', r' = 100 ∧ p' = 100 ∧ m' = 100)) := 
by
  sorry

end Vanya_can_not_score_100_in_more_than_one_exam_l643_643019


namespace sum_series_eq_11_div_18_l643_643458

theorem sum_series_eq_11_div_18 :
  (∑' n : ℕ, if n = 0 then 0 else 1 / (n * (n + 3))) = 11 / 18 :=
by
  sorry

end sum_series_eq_11_div_18_l643_643458


namespace sqrt_concat_digits_irrational_l643_643641

noncomputable def concat_digits (n : ℕ) : ℕ :=
  let ones := (10^n - 1) / 9
  let fours := (4 * (10^(2*n) - 1)) / 9
  ones * 10^(2*n) + fours

theorem sqrt_concat_digits_irrational (n : ℕ) (h : n > 1) : irrational (Real.sqrt (concat_digits n)) :=
sorry

end sqrt_concat_digits_irrational_l643_643641


namespace area_of_concentric_ring_l643_643375

theorem area_of_concentric_ring (r_large : ℝ) (r_small : ℝ) 
  (h1 : r_large = 10) 
  (h2 : r_small = 6) : 
  (π * r_large^2 - π * r_small^2) = 64 * π :=
by {
  sorry
}

end area_of_concentric_ring_l643_643375


namespace range_of_a_for_symmetric_graphs_l643_643918

noncomputable def f : ℝ → ℝ
| x => if x < 0 then x^2 + Real.exp x - (1 / 2) else x^2 + Real.exp x

noncomputable def g (a : ℝ) : ℝ → ℝ
| x => x^2 + Real.log (x + a)

theorem range_of_a_for_symmetric_graphs :
  ∃ a : ℝ, (∀ x : ℝ, x < 0 → f(x) = g a (-x)) ↔ a ∈ Set.Iio (Real.sqrt Real.exp 1) :=
begin
  sorry
end

end range_of_a_for_symmetric_graphs_l643_643918


namespace clara_boxes_l643_643080

theorem clara_boxes (x : ℕ)
  (h1 : 12 * x + 20 * 80 + 16 * 70 = 3320) : x = 50 := by
  sorry

end clara_boxes_l643_643080


namespace exists_nat_k_l643_643190

noncomputable def A (M : ℕ) : ℝ :=
  (M + real.sqrt (M^2 - 4)) / 2

theorem exists_nat_k (M : ℕ) (hM : M > 2) :
  ∃ k : ℕ, A M ^ 5 = (k + real.sqrt (k^2 - 4)) / 2 :=
sorry

end exists_nat_k_l643_643190


namespace monotonic_increase_interval_sin_2theta_value_l643_643182

noncomputable def f (x : Real) : Real := cos x ^ 2 - sqrt 3 * sin x * cos x + 1

-- 1. Prove the interval of monotonic increase
theorem monotonic_increase_interval (k : ℤ) :
  (f x = y) →
  (y = cos (2 * x + π / 3) + 3 / 2) →
  k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 :=
sorry

-- 2. Prove the value of sin 2θ given conditions
theorem sin_2theta_value (θ : Real) :
  (f θ = 5 / 6) →
  (θ > π / 3 ∧ θ < 2 * π / 3) →
  sin (2 * θ) = (2 * sqrt 3 - sqrt 5) / 6 :=
sorry

end monotonic_increase_interval_sin_2theta_value_l643_643182


namespace sweaters_to_wash_l643_643649

theorem sweaters_to_wash (pieces_per_load : ℕ) (total_loads : ℕ) (shirts_to_wash : ℕ) 
  (h1 : pieces_per_load = 5) (h2 : total_loads = 9) (h3 : shirts_to_wash = 43) : ℕ :=
  if total_loads * pieces_per_load - shirts_to_wash = 2 then 2 else 0

end sweaters_to_wash_l643_643649


namespace determine_a_l643_643901

theorem determine_a (a : ℕ) : 
  (2 * 10^10 + a ) % 11 = 0 ∧ 0 ≤ a ∧ a < 11 → a = 9 :=
by
  sorry

end determine_a_l643_643901


namespace solve_for_y_l643_643317

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l643_643317


namespace citizen_tax_amount_l643_643586

def tax (income : ℕ) : ℕ :=
  let base_tax := 0.11 * 40000
  let excess_income := max (income - 40000) 0
  let excess_tax := 0.20 * excess_income
  base_tax + excess_tax

theorem citizen_tax_amount (income : ℕ) (h : income = 58000) : 
  tax income = 8000 := by
  sorry

end citizen_tax_amount_l643_643586


namespace smallest_advantageous_discount_l643_643121

theorem smallest_advantageous_discount (n : ℕ) : 
  n > 31.8528 ∧ n > 36 ∧ n > 28.360704 ↔ n = 37 :=
by {
  sorry
}

end smallest_advantageous_discount_l643_643121


namespace greatest_power_of_2_factor_l643_643719

theorem greatest_power_of_2_factor
    : ∃ k : ℕ, (2^k) ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, (2^(m+1)) ∣ (10^1503 - 4^752) → m < k :=
by
    sorry

end greatest_power_of_2_factor_l643_643719


namespace geometric_series_sum_l643_643453

theorem geometric_series_sum : 
  let a := 1
  let r := 3
  let n := 6
  S = (1 + 3 + 9 + 27 + 81 + 243) :=
  (S = 364) :=
  sorry

end geometric_series_sum_l643_643453


namespace smallest_possible_a_l643_643657

theorem smallest_possible_a 
  (a b c : ℚ)
  (h_vertex : ∃ (a : ℚ), a > 0 ∧ 
  ∀ x y : ℚ, y = a * (x - 1/2)^2 - 1/2 ↔ y = a * x^2 + b * x + c)
  (h_integer : a + b + c ∈ ℤ) :
  ∃ (a_min : ℚ), a_min = 2 ∧ a_min > 0 ∧ ∀ (a' : ℚ), a' > 0 → (a' + (-a') + (a'/4 - 1/2) ∈ ℤ → a' ≥ a_min) :=
sorry

end smallest_possible_a_l643_643657


namespace problem1_problem2_l643_643922

theorem problem1 (a b : ℝ) (h1 : -1/2 = -1/2) (h2 : -1/3 = -1/3) 
    (h3 : -1/2 + (-1/3) = b/a) (h4 : (-1/2) * (-1/3) = -1/a) : a = -6 ∧ b = 5 :=
begin
  sorry
end

theorem problem2 (a b : ℝ) (h1 : a = -6) (h2 : b = 5) : ∀ x : ℝ, (x^2 - b*x + 6 < 0) ↔ (2 < x ∧ x < 3) :=
begin
  sorry
end

end problem1_problem2_l643_643922


namespace upstream_distance_l643_643777

-- Define the conditions
def velocity_current : ℝ := 1.5
def distance_downstream : ℝ := 32
def time : ℝ := 6

-- Define the speed of the man in still water
noncomputable def speed_in_still_water : ℝ := (distance_downstream / time) - velocity_current

-- Define the distance rowed upstream
noncomputable def distance_upstream : ℝ := (speed_in_still_water - velocity_current) * time

-- The theorem statement to be proved
theorem upstream_distance (v c d : ℝ) (h1 : c = 1.5) (h2 : (v + c) * 6 = 32) (h3 : (v - c) * 6 = d) : d = 14 :=
by
  -- Insert the proof here
  sorry

end upstream_distance_l643_643777


namespace is_right_triangle_right_triangle_identity_l643_643642

universe u

variables {α : Type u} [Real_θ α] (A B C a c : α)

noncomputable 
def triangle_pred_1 := Math.sin C - Math.cos B = Math.cos A

noncomputable 
def triangle_pred_2 := a + Real.cot (45 - B) = 2 / (1 - Real.cot C)

noncomputable 
def right_triangle_condition := (2 * Real.cot C - Real.cot B) = c / a^3 * (3*a^2 - 4*c^2)

theorem is_right_triangle (h1 : triangle_pred_1 ∨ triangle_pred_2) : Math.deg A = 90 := sorry

theorem right_triangle_identity 
    (h2 : Math.deg A = 90)
    : right_triangle_condition := sorry

end is_right_triangle_right_triangle_identity_l643_643642


namespace center_number_in_grid_l643_643062

theorem center_number_in_grid:
    (∀ i j : Fin 3, ∃ n : Fin 9, grid i j = n) →
    (∀ n : Fin 8, ∃ i1 j1 i2 j2 : Fin 3, grid i1 j1 = n ∧ grid i2 j2 = n + 1 ∧
        ((i1 = i2 ∧ (j1 = j2 + 1 ∨ j2 = j1 + 1)) ∨ (j1 = j2 ∧ (i1 = i2 + 1 ∨ i2 = i1 + 1)))) →
    (grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 18) →
    grid 1 1 = 7 :=
by
  intro h_nums h_adj h_sum
  sorry

end center_number_in_grid_l643_643062


namespace largest_stamps_per_page_l643_643250

theorem largest_stamps_per_page (h1 : Nat := 1050) (h2 : Nat := 1260) (h3 : Nat := 1470) :
  Nat.gcd h1 (Nat.gcd h2 h3) = 210 :=
by
  sorry

end largest_stamps_per_page_l643_643250


namespace a1964_eq_neg1_l643_643556

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ ∀ n ≥ 4, a n = a (n-1) * a (n-3)

theorem a1964_eq_neg1 (a : ℕ → ℤ) (h : seq a) : a 1964 = -1 :=
  by sorry

end a1964_eq_neg1_l643_643556


namespace simplify_expression_l643_643454
open Real

theorem simplify_expression (x y : ℝ) : -x + y - 2 * x - 3 * y = -3 * x - 2 * y :=
by
  sorry

end simplify_expression_l643_643454


namespace rationalize_denominator_correct_l643_643300

noncomputable def rationalize_denominator (x : ℝ) := 
  (5 : ℝ) / (3 * real.cbrt 7) * (real.cbrt 49) / (real.cbrt 49)

theorem rationalize_denominator_correct : 
  rationalize_denominator (5 / (3 * real.cbrt 7)) = (5 * real.cbrt 49) / (21 : ℝ) 
  ∧ 5 + 49 + 21 = 75 := 
by
  sorry

end rationalize_denominator_correct_l643_643300


namespace inequality_solutions_l643_643910

theorem inequality_solutions (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + bx + c < 0 ↔ x < -4 ∨ x > 3)
  (h2 : ∀ x, (ax - b) / (ax - c) ≤ 0 ↔ -12 < x ∧ x ≤ 1 ∨ false) :
  a + b + c > 0 ∧ b = a ∧ c = -12a :=
by sorry

end inequality_solutions_l643_643910


namespace chandler_bike_purchase_l643_643456

theorem chandler_bike_purchase :
  let bike_cost := 600
  let gift_money := 60 + 40 + 20
  let weekly_earnings := 20
  let weekly_expenses := 4
  let weekly_net_savings := weekly_earnings - weekly_expenses
  let total_savings (weeks : ℕ) := gift_money + weeks * weekly_net_savings
  ∀ (weeks : ℕ), total_savings weeks = bike_cost ↔ weeks = 30 :=
by
  unfold total_savings
  sorry

end chandler_bike_purchase_l643_643456


namespace algebra_expression_opposite_l643_643947

theorem algebra_expression_opposite (a : ℚ) :
  3 * a + 1 = -(3 * (a - 1)) → a = 1 / 3 :=
by
  intro h
  sorry

end algebra_expression_opposite_l643_643947


namespace ordered_pairs_count_l643_643118

noncomputable
def number_of_ordered_pairs : ℕ :=
  let pairs := {(a, b) | a^4 * b^7 = 1 ∧ a^8 * b^3 = 1}
  in pairs.to_finset.card

theorem ordered_pairs_count : number_of_ordered_pairs = 64 :=
  sorry

end ordered_pairs_count_l643_643118


namespace not_decreasing_in_interval_l643_643915

def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x - (π / 3))

theorem not_decreasing_in_interval :
  ¬ (∀ x ∈ Icc (0 : ℝ) (π / 2), ∀ y ∈ Icc (0 : ℝ) (π / 2), x ≤ y → f x ≥ f y) :=
sorry

end not_decreasing_in_interval_l643_643915


namespace factor_is_three_l643_643047

theorem factor_is_three (x f : ℝ) (h1 : 2 * x + 5 = y) (h2 : f * y = 111) (h3 : x = 16):
  f = 3 :=
by
  sorry

end factor_is_three_l643_643047


namespace find_x_value_l643_643450

def average_eq_condition (x : ℝ) : Prop :=
  (5050 + x) / 101 = 50 * (x + 1)

theorem find_x_value : ∃ x : ℝ, average_eq_condition x ∧ x = 0 :=
by
  use 0
  sorry

end find_x_value_l643_643450


namespace range_of_a_using_law_of_sines_l643_643585

-- Definitions of conditions
def b : ℝ := 2
def B : ℝ := 45 * real.pi / 180  -- Angle in radians

-- Problem statement
theorem range_of_a_using_law_of_sines (A C : ℝ) (h1 : B = real.pi / 4)
(h2 : A + C = real.pi * 3 / 4)
(h3 : (A > real.pi / 4 ∧ A < real.pi * 3 / 4))
(h4 : ∃ a1 a2 : ℝ, a1 ≠ a2 ∧ a1 = 2 * sqrt 2 * real.sin A ∧ a2 = 2 * sqrt 2 * real.sin (real.pi - A)) :
  (2 : ℝ) < (2 * sqrt 2 * real.sin A) ∧ (2 * sqrt 2 * real.sin A) < (2 * sqrt 2) :=
by sorry

end range_of_a_using_law_of_sines_l643_643585


namespace selling_price_l643_643787

variable (purchasePrice : ℝ) (overheadExpenses : ℝ) (profitPercent : ℝ)

theorem selling_price
  (h1 : purchasePrice = 225)
  (h2 : overheadExpenses = 15)
  (h3 : profitPercent = 25) :
  let costPrice := purchasePrice + overheadExpenses in
  let profit := (profitPercent / 100) * costPrice in
  let sellingPrice := costPrice + profit in
  sellingPrice = 300 :=
by
  sorry

end selling_price_l643_643787


namespace tan_cot_solutions_θ_l643_643469

theorem tan_cot_solutions_θ (θ : ℝ) (hθ : θ ∈ Ioo 0 (2 * π)) :
  ∃ n : ℕ, (1 : ℝ) ≤ n ∧ n ≤ 20 ∧
  (tan (4 * π * sin θ) = cot (4 * π * cos θ)) :=
by sorry

end tan_cot_solutions_θ_l643_643469


namespace number_of_ways_to_choose_starters_l643_643636

-- Define the problem parameters
def players : Finset String := 
  {"Alicia", "Amanda", "Anna", "Alice"} ∪ Finset.range 14

def quadruplets : Finset String := {"Alicia", "Amanda", "Anna", "Alice"}

-- Given conditions
def totalPlayers : ℕ := 18
def starters : ℕ := 8

-- The question to prove
theorem number_of_ways_to_choose_starters (h1 : totalPlayers = 18)
                                         (h2 : quadruplets ⊆ players)
                                         (h3 : starters = 8)
  : ∃ n : ℕ, (n = 27027) ∧ 
    (∃ (two_quadruplets : Finset (Finset String)) 
       (h_two : two_quadruplets.card = 2) 
       (six_other_players : Finset (Finset String)) 
       (h_six : six_other_players.card = 6),
       ∀ (pair : Finset String) 
         (h_pair : pair ∈ two_quadruplets)
         (elems : Finset String)
         (h_elems : elems ∈ six_other_players),
         (pair ∪ elems).card = 8)
∧ (∃ (three_quadruplets : Finset (Finset String))
       (h_three : three_quadruplets.card = 3)
       (five_other_players : Finset (Finset String))
       (h_five : five_other_players.card = 5),
        ∀ (triple : Finset String)
          (h_triple : triple ∈ three_quadruplets)
          (elems : Finset String)
          (h_elems : elems ∈ five_other_players),
          (triple ∪ elems).card = 8)
∧ (∃ (four_quadruplets : Finset (Finset String))
       (h_four : four_quadruplets.card = 4)
       (four_other_players : Finset (Finset String))
       (h_four : four_other_players.card = 4),
        ∀ (quad : Finset String)
          (h_quad : quad ⊆ quadruplets)
          (elems : Finset String)
          (h_elems : elems ∈ four_other_players),
          (quad ∪ elems).card = 8) := sorry

end number_of_ways_to_choose_starters_l643_643636


namespace determine_n_l643_643270

noncomputable def P : Polynomial ℝ := sorry  -- Definition of P, given the degree and constraints

theorem determine_n :
  (∀ k : ℕ, (0 ≤ k ∧ k ≤ 3 * 4 ∧ k % 3 = 0 → P k = 2) ∧
            (0 ≤ k ∧ k ≤ 3 * 4 - 2 ∧ k % 3 = 1 → P k = 1) ∧
            (0 ≤ k ∧ k ≤ 3 * 4 - 1 ∧ k % 3 = 2 → P k = 0) ∧
            (P (3 * 4 + 1) = 730)) → 3n = 12 :=
begin
  sorry -- The proof steps need to be filled in
end

end determine_n_l643_643270


namespace solve_for_x_y_l643_643331

theorem solve_for_x_y (x y : ℚ) 
  (h1 : (3 * x + 12 + 2 * y + 18 + 5 * x + 6 * y + (3 * x + y + 16)) / 5 = 60) 
  (h2 : x = 2 * y) : 
  x = 254 / 15 ∧ y = 127 / 15 :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_y_l643_643331


namespace election_winner_votes_l643_643366

variable (V : ℕ)
variable (h1 : 0.62 * V = winning_votes)
variable (h2 : 0.38 * V = losing_votes)
variable (h3 : winning_votes - losing_votes = 348)

theorem election_winner_votes : ∃ V, winning_votes = 899 := by
  unfold winning_votes losing_votes
  sorry

end election_winner_votes_l643_643366


namespace num_ways_to_distribute_balls_l643_643210

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end num_ways_to_distribute_balls_l643_643210


namespace relationship_among_abc_l643_643166

noncomputable def f : ℝ → ℝ := sorry

-- f(x) is an odd function.
axiom odd_function : ∀ x : ℝ, f(x) + f(-x) = 0

-- For x > 0, (f(x) / x) + f'(x) > 0.
axiom derivative_condition : ∀ x : ℝ, x > 0 → (f(x) / x) + (deriv f x) > 0

-- Definitions of a, b, and c
def a := f 1
def b := Real.log(2) * f (Real.log 2)
def c := (Real.log 2 / Real.log 3) * f (Real.log 2 / Real.log 3)

-- Final theorem statement
theorem relationship_among_abc : c > a ∧ a > b := sorry

end relationship_among_abc_l643_643166


namespace david_marks_in_math_l643_643463

/-
David obtained the following marks:
- 51 marks in English,
- some marks in Mathematics (unknown),
- 82 marks in Physics,
- 67 marks in Chemistry,
- and 85 marks in Biology.

His average marks are 70.
We aim to prove that David obtained 65 marks in Mathematics.

Proof statement:
-/

theorem david_marks_in_math (marks_in_math : ℕ) 
  (h_avg : 70) 
  (h_num_subjects : 5) 
  (h_english : 51)
  (h_physics : 82) 
  (h_chemistry : 67) 
  (h_biology : 85)
  (h_total : h_avg * h_num_subjects = 350) 
  (h_known_total : h_english + h_physics + h_chemistry + h_biology = 285) : 
  marks_in_math = 65 :=
by 
  /-
  We have the following conditions:
  - Total marks for all subjects = 350
  - Total marks in known subjects = 285
  - average marks = 70
  - Number of subjects = 5
  
  We need to show that marks in mathematics = 65.
  -/
  sorry

end david_marks_in_math_l643_643463


namespace incorrect_statement_C_l643_643005

-- conditions
def is_stratified_sampling_suitable (high_income middle_income low_income total: ℕ) : Prop := 
  total = 100 ∧ high_income = 65 ∧ middle_income = 28 ∧ low_income = 105

def regression_line_property (b : ℝ) (a : ℝ) (x̄ : ℝ) (ȳ: ℝ) : Prop :=
  ∀ x y, y = b * x + a → (x, y) = (x̄, ȳ)

def linear_correlation_definition (r : ℝ) : Prop := 
  r ∈ set.Icc (-1) 1

def median_of_data_set (a : ℝ) : ℝ :=
  if a = 2 then 2 else (if a = 1 then 2 else (if a > 3 then a else 2))

-- proof problem
theorem incorrect_statement_C (high_income middle_income low_income : ℕ) 
  (b a x̄ ȳ r a_data : ℝ) : 
  is_stratified_sampling_suitable high_income middle_income low_income 100 →
  regression_line_property b a x̄ ȳ →
  linear_correlation_definition r →
  median_of_data_set a_data = 2 →
  r ≠ 1 ∧ r ≠ -1 → False :=
sorry

end incorrect_statement_C_l643_643005


namespace sum_outside_layers_l643_643401

noncomputable def sumNumbersOutsideLayers : ℕ := sorry

theorem sum_outside_layers (n : ℕ) (K_value : ℕ) (layer_sum : ℕ)
  (column_sum : Fin n → (Fin n × Fin n → ℕ) → ℕ)
  (h_cube_size : n = 20)
  (h_total_sum : column_sum = λ _, 1)
  (h_K_value : K_value = 10)
  (h_layer_sum : layer_sum = 20) :
  sumNumbersOutsideLayers = 20 := 
sorry


end sum_outside_layers_l643_643401


namespace math_problem_l643_643533

-- Define the circle equation and function conditions
variables {r α : ℝ}
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = r^2
def sine_fn (x : ℝ) : ℝ := Real.sin x

-- Condition: unique intersection point
def unique_intersection (α : ℝ) : Prop := ∀ (x y : ℝ), circle_eq x y → sine_fn x = y → x = α ∧ y = Real.sin α

-- The expression to evaluate
def expression_val (α : ℝ) : ℝ := (2 * Real.sin (2 * α) * Real.cos α - 4 * (Real.cos α)^2) / (α * Real.cos α)

theorem math_problem
  (h1 : unique_intersection α) :
  expression_val α = -4 := 
sorry

end math_problem_l643_643533


namespace imaginary_part_of_conjugate_l643_643537

def z : Complex := Complex.mk 1 2

def z_conj : Complex := Complex.mk 1 (-2)

theorem imaginary_part_of_conjugate :
  z_conj.im = -2 := by
  sorry

end imaginary_part_of_conjugate_l643_643537


namespace contractor_engaged_days_l643_643032

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l643_643032


namespace race_outcomes_l643_643058

universe u

def participants : List String := ["Alice", "Ben", "Carla", "David", "Emily", "Frank"]

-- Condition: Alice cannot finish in 2nd place
def alice_not_second (placement: List String) : Prop := placement.nth 1 ≠ some "Alice"

-- The main theorem statement
theorem race_outcomes : ∃ outcomes : Nat, 
  outcomes = 300 ∧ 
  ∀ placement : List String, 
    placement.length = 4 ∧ 
    (∀ i j, i ≠ j → placement.nth i ≠ placement.nth j) → 
    alice_not_second placement :=
sorry

end race_outcomes_l643_643058


namespace pesto_calculation_l643_643079

def basil_needed_per_pesto : ℕ := 4
def basil_harvest_per_week : ℕ := 16
def weeks : ℕ := 8
def total_basil_harvested : ℕ := basil_harvest_per_week * weeks
def total_pesto_possible : ℕ := total_basil_harvested / basil_needed_per_pesto

theorem pesto_calculation :
  total_pesto_possible = 32 :=
by
  sorry

end pesto_calculation_l643_643079


namespace find_x_from_exp_log_condition_l643_643484

theorem find_x_from_exp_log_condition (x : ℝ) (h : 9 ^ real.log x / real.log 5 = 81) : x = 25 :=
sorry

end find_x_from_exp_log_condition_l643_643484


namespace reciprocal_of_neg_nine_l643_643347

-- Define the condition: The reciprocal of a number x is defined as 1/x
def reciprocal (x : ℝ) : ℝ := 1 / x

-- The theorem statement: Prove that the reciprocal of -9 is -1/9
theorem reciprocal_of_neg_nine : reciprocal (-9) = -1 / 9 := 
by 
  sorry

end reciprocal_of_neg_nine_l643_643347


namespace infinite_seq_perfect_squares_exists_l643_643474

theorem infinite_seq_perfect_squares_exists :
  ∃ (a : ℕ → ℕ), (∀ (n : ℕ), ∃ k : ℕ, (∑ i in finset.range n, a i ^ 2) = k^2) :=
sorry

end infinite_seq_perfect_squares_exists_l643_643474


namespace real_solution_count_of_polynomial_l643_643490

theorem real_solution_count_of_polynomial :
  (∃! x ∈ ℝ, (x^2010 + 1) * (∑ i in (list.range 1005).map (λ n, x^(2008 - 2*n)) - (∑ i in (list.range 1005).map (λ n, x^(2*n)))) = 2010 * x^2009) :=
sorry

end real_solution_count_of_polynomial_l643_643490


namespace hexagon_area_sum_l643_643771

theorem hexagon_area_sum (a b : ℤ) :
  (∃ (ABCDEF : Type) (lines : set (ABCDEF → ABCDEF)) (side_length : ℤ),
    ABCDEF.hexagon_with_segments lines ∧
    side_length = 3 ∧
    hexagon_area ABCDEF = a * sqrt b ∧
    12 = number_of_segments lines) →
  a + b = 30 :=
sorry

end hexagon_area_sum_l643_643771


namespace integral_eval_l643_643848

theorem integral_eval : 
  ∫ x in -1..1, (Real.sin x + Real.sqrt (1 - x^2)) = Real.pi / 2 :=
by sorry

end integral_eval_l643_643848


namespace no_solution_exists_l643_643125

theorem no_solution_exists : ¬ ∃ n : ℕ, (n^2 ≡ 1 [MOD 5]) ∧ (n^3 ≡ 3 [MOD 5]) := 
sorry

end no_solution_exists_l643_643125


namespace significant_difference_in_hygiene_habits_risk_level_relationship_risk_level_estimation_l643_643779

-- Definitions for data in the contingency table.
def CaseGroup.NotGoodEnough : ℕ := 40
def CaseGroup.Good : ℕ := 60
def ControlGroup.NotGoodEnough : ℕ := 10
def ControlGroup.Good : ℕ := 90

-- Calculate N = Total number of individuals surveyed.
def N : ℕ := 200 

-- Calculate K^2 using the data given.
def K_square : ℝ := (N * (CaseGroup.NotGoodEnough * ControlGroup.Good - ControlGroup.NotGoodEnough * CaseGroup.Good)^2) / 
                    ((CaseGroup.NotGoodEnough + CaseGroup.Good) * (ControlGroup.NotGoodEnough + ControlGroup.Good) * 
                     (CaseGroup.NotGoodEnough + ControlGroup.NotGoodEnough) * (CaseGroup.Good + ControlGroup.Good))

theorem significant_difference_in_hygiene_habits (hK : K_square > 6.635) : 
  ∀ x, True := 
by
  -- Skipping proof
  sorry

-- Definitions for given probabilities
def P_A_given_B : ℝ := 2/5
def P_A_given_not_B : ℝ := 1/10
def P_not_A_given_B : ℝ := 3/5
def P_not_A_given_not_B : ℝ := 9/10

-- Definition for R
def R : ℝ := (P_A_given_B / P_not_A_given_B) * (P_not_A_given_not_B / P_A_given_not_B)

-- Statement to prove the relationship for R
theorem risk_level_relationship : 
  R = (P_A_given_B / P_not_A_given_B) * (P_not_A_given_not_B / P_A_given_not_B) := 
by 
  -- Skipping proof
  sorry

-- Estimate risk level based on provided data.
def estimated_R : ℝ := 6

theorem risk_level_estimation (hR: R = 6) : 
  ∀ x, True := 
by
  -- Skipping proof
  sorry

end significant_difference_in_hygiene_habits_risk_level_relationship_risk_level_estimation_l643_643779


namespace correct_sequence_l643_643891

def a : ℕ → ℕ
| 0 := 2
| (n + 1) := if n % 2 = 0 then a n + 1 else a n + 3

def b (n : ℕ) : ℕ := a (2 * n)

theorem correct_sequence :
  ∀ n : ℕ, b n = 4 * n - 1 :=
by
  sorry

end correct_sequence_l643_643891


namespace range_of_a_l643_643557

open Set

variable {a : ℝ}

theorem range_of_a :
  (A = {a}) ∧ (B = {x : ℝ | x^2 - x > 0}) ∧ ¬ (A ⊆ B) → (0 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l643_643557


namespace soccer_player_rearrangement_l643_643089

theorem soccer_player_rearrangement (N : ℕ) (hN : N ≥ 2) : 
  ∃ (remaining : list ℕ), 
  length remaining = 2 * N ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ N → 
    no_one_between remaining (2 * i - 1) (2 * i)
  :=
sorry

end soccer_player_rearrangement_l643_643089


namespace first_five_valid_numbers_l643_643806

noncomputable def selected_bags_of_milk : list ℕ :=
[785, 667, 199, 507, 175]

theorem first_five_valid_numbers :
  let sequence := [578, 785, 916, 955, 667, 199, 507, 175, 764, 320, 24]
  in filter (λ x, x < 800) (sequence.drop 6) = selected_bags_of_milk :=
by {
  let sequence := [578, 785, 916, 955, 667, 199, 507, 175, 764, 320, 24],
  let valid_numbers := filter (λ x, x < 800) (sequence.drop 6),
  exact rfl
}

end first_five_valid_numbers_l643_643806


namespace contractor_engagement_days_l643_643036

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l643_643036


namespace quadratic_general_form_l643_643093

theorem quadratic_general_form :
  ∀ x : ℝ, (x - 2) * (x + 3) = 1 → x^2 + x - 7 = 0 :=
by
  intros x h
  sorry

end quadratic_general_form_l643_643093


namespace corn_pig_water_ratio_l643_643455

-- Define the conditions 

def pump_rate := 3 -- gallons per minute
def pump_time := 25 -- minutes

def pigs := 10
def pig_water_needs := 4 -- gallons per pig

def ducks := 20
def duck_water_needs := 0.25 -- gallons per duck

def corn_rows := 4
def plants_per_row := 15 -- corn plants per row

-- The main theorem to prove

theorem corn_pig_water_ratio : 
  let total_pumped_water := pump_rate * pump_time in
  let total_pig_water := pigs * pig_water_needs in
  let total_duck_water := ducks * duck_water_needs in
  let total_corn_water := total_pumped_water - total_pig_water - total_duck_water in
  let total_corn_plants := corn_rows * plants_per_row in
  let water_per_plant := total_corn_water / total_corn_plants in
  water_per_plant / pig_water_needs = 0.125 := 
by
  sorry

end corn_pig_water_ratio_l643_643455


namespace solve_for_y_l643_643315

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l643_643315


namespace simplify_and_evaluate_expression_l643_643653

theorem simplify_and_evaluate_expression :
  ∀ (x y : ℚ), x = -1/2 ∧ y = -2 →
  (3 * x - 2 * y) ^ 2 - (2 * y + x) * (2 * y - x) - 2 * x * (5 * x - 6 * y + x * y) = 1 :=
by
  intro x y
  intros h
  cases h with hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l643_643653


namespace solve_for_a_b_period_l643_643551

noncomputable def sine_function_max_min (a b : ℝ) (h : a > 0) 
  (max_value : a * real.sin x + 2 * b = 4)
  (min_value : 2 * b - a = 0) : Prop :=
  a + b = 3 ∧ (∀ (x : ℝ), periodic (λ x, b * real.sin (a * x)) (2 * π / a) ∧ (2 * π / a > 0))

theorem solve_for_a_b_period (a b : ℝ) (h : a > 0) 
  (max_cond : ∀ x, a * real.sin x + 2 * b = 4)
  (min_cond : 2 * b - a = 0) : 
  (a + b = 3) ∧ (2 * π / a = π) :=
by {
  sorry
}

end solve_for_a_b_period_l643_643551


namespace proof_correct_area_enclosed_by_semicircles_l643_643790

-- Define points, lines, and other necessary geometrical entities
noncomputable def semicircle_area_problem : Prop :=
  let R := (0, 0)  -- Origin for simplicity, center R 
  let P := (-2, 0)  -- Defined for semicircle PQ, with radius 2
  let Q := (2, 0)  -- Defined to maintain radius 2 with R
  let S := (0, 2)  -- Defined such that RS is perpendicular to PQ
  let T := (4, 8)  -- Dummy point far enough, extended from PS
  let U := (-4, 8)  -- Dummy point far enough, extended from QS

  -- Regions computation
  let area_PT := (1/2) * (4^2) * (Real.pi)
  let area_QU := (1/2) * (4^2) * (Real.pi)
  let area_TU := (1/2) * (Math.sqrt(2)^2) * (Real.pi)
  let area_PQS := 2  -- Area of triangle PQS

  -- Total enclosed area
  (area_PT + area_QU + area_TU - area_PQS) = 9 * Real.pi - 2

theorem proof_correct_area_enclosed_by_semicircles :
  semicircle_area_problem := by
  -- "sorry" is used to skip the proof.
  sorry

end proof_correct_area_enclosed_by_semicircles_l643_643790


namespace claudia_earnings_over_weekend_l643_643828

theorem claudia_earnings_over_weekend :
  ∀ (cost_per_kid : ℕ) (saturday_kids : ℕ) (sunday_kids : ℕ),
  cost_per_kid = 10 →
  saturday_kids = 20 →
  sunday_kids = saturday_kids / 2 →
  (saturday_kids + sunday_kids) * cost_per_kid = 300 :=
by
  intros cost_per_kid saturday_kids sunday_kids
  assume h1 : cost_per_kid = 10
  assume h2 : saturday_kids = 20
  assume h3 : sunday_kids = saturday_kids / 2
  sorry

end claudia_earnings_over_weekend_l643_643828


namespace change_combinations_12_dollars_l643_643840

theorem change_combinations_12_dollars :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), 
  (∀ (n d q : ℕ), (n, d, q) ∈ solutions ↔ 5 * n + 10 * d + 25 * q = 1200 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1) ∧ solutions.card = 61 :=
sorry

end change_combinations_12_dollars_l643_643840


namespace circumscribable_iff_cyclic_l643_643611

-- Definitions based on the given conditions
variables {A B C I D E F : Type*}

-- Incenter I, points D, E, F, and segment conditions CD = CE and F on CD
variables (triangle_ABC : ∀ (A B C I : Type*), ∃ (I : Type*), I ∈ incenter (A, B, C))
variables (on_segment_CA : ∀ (D : Type*), D ∈ segment (C, A))
variables (on_segment_BC : ∀ (E : Type*), E ∈ segment (B, C))
variables (CD_eq_CE : ∀ (D E : Type*), segment_length (C, D) = segment_length (C, E))
variables (on_segment_CD : ∀ (F : Type*), F ∈ segment (C, D))

-- Proposition statement based on the problem
theorem circumscribable_iff_cyclic :
  (circumscribable_quadrilateral (A, B, E, F)) ↔ (cyclic_quadrilateral (D, I, E, F)) :=
sorry

end circumscribable_iff_cyclic_l643_643611


namespace claudia_weekend_earnings_l643_643826

/-- Claudia charges $10.00 for her one-hour class. 20 kids attend the Saturday’s class.
    Half that many attend the Sunday’s class. Prove Claudia makes $300.00 over the weekend. -/
theorem claudia_weekend_earnings : 
  let charge_per_kid := 10
  let saturday_attendees := 20
  let sunday_attendees := saturday_attendees / 2
  let total_attendees := saturday_attendees + sunday_attendees
  let total_earnings := charge_per_kid * total_attendees
  total_earnings = 300 := 
by 
  simp
  sorry

end claudia_weekend_earnings_l643_643826


namespace percentage_calculation_l643_643858

/-- If x % of 375 equals 5.4375, then x % equals 1.45 %. -/
theorem percentage_calculation (x : ℝ) (h : x / 100 * 375 = 5.4375) : x = 1.45 := 
sorry

end percentage_calculation_l643_643858


namespace triangle_ext_angle_l643_643242

variables {A B C : Type} [angle A] [angle B] [angle C] {a b c : ℝ}

section
  variables (A B C : ℝ) (a b c : ℝ) 
  variables (alpha beta : ℝ)

  -- Conditions
  hypothesis h1 : a < b + c
  hypothesis h2 : b < a + c
  hypothesis h3 : c < a + b
  hypothesis h4 : alpha = 180 - A
  hypothesis h5 : beta = 180 - B

  -- Correct Statement
  theorem triangle_ext_angle (hA : A + B + C = 180) : B + C = alpha :=
  sorry
end

end triangle_ext_angle_l643_643242


namespace toothpicks_in_250th_stage_l643_643674

theorem toothpicks_in_250th_stage :
  let f : ℕ → ℕ :=
    λ n, if n = 0 then 5 else if n % 50 = 0 then 2 * (f (n - 1)) else f (n - 1) + 5
  in f 250 = 15350 :=
by
  sorry

end toothpicks_in_250th_stage_l643_643674


namespace four_pow_a_eq_nine_l643_643504

theorem four_pow_a_eq_nine (a : ℝ) (h : log 2 3 = a) : 4^a = 9 :=
sorry

end four_pow_a_eq_nine_l643_643504


namespace inequality_for_an_l643_643150

theorem inequality_for_an (n : ℕ) (h : n > 1) :
  let a_n := ∑ k in Finset.range (n + 1), 1 / Real.sqrt (k + 1)
  in 2 * Real.sqrt n - 3 / 2 < a_n ∧ a_n < 2 * Real.sqrt n - 1 :=
by
  sorry

end inequality_for_an_l643_643150


namespace find_value_of_reciprocal_cubic_sum_l643_643936

theorem find_value_of_reciprocal_cubic_sum
  (a b c r s : ℝ)
  (h₁ : a + b + c = 0)
  (h₂ : a ≠ 0)
  (h₃ : b^2 - 4 * a * c ≥ 0)
  (h₄ : r ≠ 0)
  (h₅ : s ≠ 0)
  (h₆ : a * r^2 + b * r + c = 0)
  (h₇ : a * s^2 + b * s + c = 0)
  (h₈ : r + s = -b / a)
  (h₉ : r * s = -c / a) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3 * a^2 + 3 * a * b) / (a + b)^3 :=
by
  sorry

end find_value_of_reciprocal_cubic_sum_l643_643936


namespace candy_price_difference_l643_643042

theorem candy_price_difference (x : ℕ) (h1 : ∀ price_diff : ℕ, price_diff = 80) : 
  let price_cheaper := x in
  let price_expensive := x + 80 in
  let andrey_grams := [(100, price_expensive), (50, price_cheaper)] in
  let yura_grams := [(75, price_cheaper), (75, price_expensive)] in
  let cost (grams: (ℕ × ℕ) list) := grams.foldl (λ acc (g, p), acc + g * p / 1000) 0 in
  cost andrey_grams = cost yura_grams + 2 := by
  sorry

end candy_price_difference_l643_643042


namespace sum_of_remainders_l643_643066

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) :
  (n % 2 + n % 9) = 3 :=
sorry

end sum_of_remainders_l643_643066


namespace smallest_b_is_4_l643_643326

noncomputable def smallest_possible_b : ℕ :=
  Inf {b | ∀ a b : ℕ, a - b = 8 ∧ a > 0 ∧ b > 0 ∧ gcd ((a^4 + b^4) / (a + b)) (a * b) = 16}

theorem smallest_b_is_4 :
  smallest_possible_b = 4 :=
by sorry

end smallest_b_is_4_l643_643326


namespace spaceship_speed_400_l643_643127

-- The function spaceship_speed takes the number of people on board and returns the speed of the spaceship.
noncomputable def spaceship_speed : ℕ → ℝ
| 0     := sorry  -- base speed without people is unknown and depends on the context
| 200   := 500
| n + 100 := (spaceship_speed n) / 2

theorem spaceship_speed_400 : spaceship_speed 400 = 125 := by sorry

end spaceship_speed_400_l643_643127


namespace dirichlet_properties_l643_643337

-- Definition of the Dirichlet function
def dirichlet (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

-- Define the properties of the Dirichlet function
theorem dirichlet_properties :
  (dirichlet (real.sqrt 2) = 0) ∧
  (∀ x : ℝ, x ∈ set.univ) ∧
  (∀ x : ℝ, dirichlet (dirichlet x) = 1) ∧
  (∀ x : ℝ, dirichlet x = dirichlet (-x)) :=
by
  sorry

end dirichlet_properties_l643_643337


namespace jerrys_current_average_score_l643_643252

theorem jerrys_current_average_score (A : ℝ) (h1 : 3 * A + 98 = 4 * (A + 2)) : A = 90 :=
by
  sorry

end jerrys_current_average_score_l643_643252


namespace complex_problem_l643_643174

-- Define the given complex numbers
def z1 : ℂ := 3 - 2 * complex.I
def z2 : ℂ := -2 + 3 * complex.I

-- Lean statement for the problem
theorem complex_problem :
  (z1 * z2 = -12 + complex.I) ∧
  (∃ z : ℂ, (1 / z = 1 / z1 + 1 / z2) ∧ |z| = 13 * real.sqrt 2 / 2) :=
by {
  -- proof placeholder
  sorry
}

end complex_problem_l643_643174


namespace definite_integral_ln_x_plus_x_l643_643452

theorem definite_integral_ln_x_plus_x :
  ∫ x in 1..2, (1/x + x) = Real.log 2 + 3/2 := 
by
  sorry

end definite_integral_ln_x_plus_x_l643_643452


namespace triangle_apb_area_eq_zero_l643_643425

noncomputable def triangle_area (A B P : ℝ × ℝ) : ℝ := 
  1 / 2 * abs ((B.1 - A.1) * (P.2 - A.2) - (P.1 - A.1) * (B.2 - A.2))

theorem triangle_apb_area_eq_zero :
  let A := (0, 0)
  let B := (8, 0)
  let C := (8, 4)
  let D := (0, 4)
  ∃ P : ℝ × ℝ, 
    (∥P - A∥ = ∥P - B∥ ∧ ∥P - B∥ = ∥P - D∥) ∧
    (P.2 = 0) →
    triangle_area A B P = 0 :=
by 
  sorry

end triangle_apb_area_eq_zero_l643_643425


namespace line_with_equal_intercepts_l643_643007

theorem line_with_equal_intercepts (a b : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) (h2 : equal_intercept : (x/a) + (y/a) = 1)
    (h3 : passes_through : (1 / a) + (2 / a) = 1) : (2 * a - b = 0) ∨ (a + b - 3 = 0) :=
by
  sorry

end line_with_equal_intercepts_l643_643007


namespace statement1_statement2_statement3_l643_643268

-- Define the assumptions about the function f
variables {f : ℝ → ℝ}
variables (h1 : ∀ x, f(-x) = -f(x))      -- f(x) is odd
variables (h2 : ∀ x, f(x+2) = -f(x))     -- f(x+2) = -f(x)

-- Define the three statements as propositions to prove
theorem statement1 : f 4 = 0 := sorry
theorem statement2 : ∀ x, f (x + 4) = f x := sorry
theorem statement3 : ∀ x, f (1 + x) = f (1 - x) := sorry

end statement1_statement2_statement3_l643_643268


namespace two_planes_perpendicular_to_same_plane_are_not_parallel_l643_643713

theorem two_planes_perpendicular_to_same_plane_are_not_parallel 
  (P1 P2 P3 : Plane) 
  (h1 : P1 ⊥ P3) 
  (h2 : P2 ⊥ P3) :
  ¬ parallel P1 P2 :=
sorry

end two_planes_perpendicular_to_same_plane_are_not_parallel_l643_643713


namespace find_number_l643_643707

theorem find_number (x: ℝ) (h: (6 * x) / 2 - 5 = 25) : x = 10 :=
by
  sorry

end find_number_l643_643707


namespace problem_1_problem_2_l643_643195

def setA (x : ℝ) : Prop := 2 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 3 < x ∧ x ≤ 10
def setC (a : ℝ) (x : ℝ) : Prop := a - 5 < x ∧ x < a

theorem problem_1 (x : ℝ) :
  (setA x ∧ setB x ↔ 3 < x ∧ x < 7) ∧
  (setA x ∨ setB x ↔ 2 ≤ x ∧ x ≤ 10) := 
by sorry

theorem problem_2 (a : ℝ) :
  (∀ x, setC a x → (2 ≤ x ∧ x ≤ 10)) ↔ (7 ≤ a ∧ a ≤ 10) :=
by sorry

end problem_1_problem_2_l643_643195


namespace groceries_spent_l643_643797

-- Definitions of conditions
def rent : ℕ := 5000
def milk : ℕ := 1500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 5650
def savings : ℕ := 2350
def total_salary : ℕ := 23500

-- Statement of the theorem
theorem groceries_spent : 
  let known_expenses := rent + milk + education + petrol + miscellaneous in
  let total_expenses := total_salary - savings in
  let groceries := total_expenses - known_expenses in
  groceries = 4500 :=
by
  sorry

end groceries_spent_l643_643797


namespace marathon_laps_l643_643068

theorem marathon_laps (r : ℝ) (distance_km : ℝ) (C : ℝ) (N : ℝ) (π_lower : ℝ) (π_upper : ℝ) 
  (hr : r = 100) (hd : distance_km = 42) (hC : C = 2 * π * r) (hN : N = distance_km * 1000 / C) 
  (hπ_bounds : π_lower < real.pi ∧ real.pi < π_upper) 
  (πl : π_lower = 3) (πu : π_upper = 4) : 
  52.5 < N ∧ N < 70 :=
by
  sorry

end marathon_laps_l643_643068


namespace equation_of_circle_C_max_area_PMQL_l643_643140

noncomputable def circle_C := { radius := 2, center := (0, 0) }
def point_A := (-2 : ℝ, 0 : ℝ)
def point_B := (0 : ℝ, 2 : ℝ)
def point_O := (0 : ℝ, 0 : ℝ)
def line_l (k : ℝ) := (λ x : ℝ, k * x + 1)
def line_l1 (k : ℝ) := (λ x : ℝ, (-1 / k) * x + 1)

-- 1) Find the equation of the circle C
theorem equation_of_circle_C : ∀ (x y : ℝ), (x, y) ∈ circle_C → x^2 + y^2 = 4 :=
by sorry

-- 2) Find the maximum area of the quadrilateral PMQN
theorem max_area_PMQL {P Q M N: (ℝ × ℝ)} :
  (P ∈ circle_C ∧ Q ∈ circle_C) ∧ (M ∈ circle_C ∧ N ∈ circle_C) →
  (P ∈ line_l 1 ∧ Q ∈ line_l 1) ∧ (M ∈ line_l1 1 ∧ N ∈ line_l1 1) →
  max_area_of_quadrilateral PMQN = 7 :=
by sorry

end equation_of_circle_C_max_area_PMQL_l643_643140


namespace seating_arrangements_count_l643_643877

-- Define the data type for seats
inductive Seat
| front_left : Seat
| front_middle : Seat
| front_right : Seat
| back_left : Seat
| back_middle : Seat
| back_right : Seat

-- Define the data type for family member
inductive FamilyMember
| guardian : FamilyMember
| a : FamilyMember
| b : FamilyMember
| c : FamilyMember
| d : FamilyMember

-- Define conditions
def valid_seating (arrangement : Seat → FamilyMember) : Prop :=
  arrangement Seat.front_left = FamilyMember.guardian ∧
  ¬ ((arrangement Seat.front_middle = FamilyMember.a ∧ arrangement Seat.back_middle = FamilyMember.b) ∨
     (arrangement Seat.front_right = FamilyMember.a ∧ arrangement Seat.back_right = FamilyMember.b) ∨
     (arrangement Seat.front_middle = FamilyMember.c ∧ arrangement Seat.back_middle = FamilyMember.d) ∨
     (arrangement Seat.front_right = FamilyMember.c ∧ arrangement Seat.back_right = FamilyMember.d)) ∧
  ¬ ((arrangement Seat.front_left = FamilyMember.a ∧ arrangement Seat.front_middle = FamilyMember.b) ∨
     (arrangement Seat.front_middle = FamilyMember.a ∧ arrangement Seat.front_right = FamilyMember.b) ∨
     (arrangement Seat.back_left = FamilyMember.c ∧ arrangement Seat.back_middle = FamilyMember.d) ∨
     (arrangement Seat.back_middle = FamilyMember.c ∧ arrangement Seat.back_right = FamilyMember.d)) ∧
  ¬ ((arrangement Seat.front_left = FamilyMember.c ∧ arrangement Seat.front_middle = FamilyMember.d) ∨
     (arrangement Seat.front_middle = FamilyMember.c ∧ arrangement Seat.front_right = FamilyMember.d) ∨
     (arrangement Seat.back_left = FamilyMember.a ∧ arrangement Seat.back_middle = FamilyMember.b) ∨
     (arrangement Seat.back_middle = FamilyMember.a ∧ arrangement Seat.back_right = FamilyMember.b))

-- Prove the total number of valid seating arrangements
theorem seating_arrangements_count : (finset.filter valid_seating
  (finset.univ : finset (Seat → FamilyMember))).card = 16 :=
by
  sorry

end seating_arrangements_count_l643_643877


namespace triangle_square_equality_or_right_angle_l643_643071

theorem triangle_square_equality_or_right_angle
  (A B C D E F G P Q : Point)
  (square_BAED : square BAED)
  (square_ACFG : square ACFG)
  (DC_intersects_AB : line_intersects DC AB P)
  (BF_intersects_AC : line_intersects BF AC Q)
  (AP_eq_AQ : dist A P = dist A Q) :
  dist A B = dist A C ∨ angle A B C = 90 := sorry

end triangle_square_equality_or_right_angle_l643_643071


namespace correct_propositions_l643_643439

variables (m n : Type) (α β γ : Type)

-- m, n are lines and α, β, γ are planes
variable [Line m] [Line n]
variable [Plane α] [Plane β] [Plane γ]

-- Propositions
def prop_1  : Prop := ∀ (m_perp_α : IsPerpendicular m α) (n_parallel_α : IsParallel n α), IsPerpendicular m n
def prop_4  : Prop := ∀ (α_parallel_β : IsParallel α β) (β_parallel_γ : IsParallel β γ) (m_perp_α : IsPerpendicular m α), IsPerpendicular m γ

theorem correct_propositions : prop_1 ∧ prop_4 :=
by
  sorry

end correct_propositions_l643_643439


namespace critical_point_properties_l643_643548

noncomputable theory
open Real

-- Define the function f
def f (x : ℝ) : ℝ := x * log x + x^2

-- Define the condition that x0 is a critical point of f
def is_critical_point (x0 : ℝ) : Prop := 
  deriv f x0 = 0

-- The statement we want to prove
theorem critical_point_properties (x0 : ℝ) (hx0 : is_critical_point x0) :
  0 < x0 ∧ x0 < 1 / exp 1 ∧ f x0 + 2 * x0 > 0 :=
sorry

end critical_point_properties_l643_643548


namespace semicircle_parametric_find_point_D_l643_643600

theorem semicircle_parametric (rho theta : ℝ) (h1 : 0 < theta ∧ theta < ιπ):
  (cos theta)^2 + (1 + sin theta)^2 = 1 :=
begin
  sorry
end

theorem find_point_D (A B C D : Point) (h1 : A = ⟨0, -2⟩) 
  (h2 : B ∋ line intersects x-axis and y-axis) 
  (h3 : C ∋ semicircle center) 
  (h4 : D ∋ semicircle and slope_angle(C, D) = 2 * slope_angle(l)) 
  (h5 : area_triangle(A, B, D) = 4) : 
  D = ⟨0, 2⟩ :=
begin
  sorry
end

end semicircle_parametric_find_point_D_l643_643600


namespace volume_of_vectors_satisfying_condition_l643_643098

open Real

def volume_of_solid_formed_by_vectors (v : ℝ × ℝ × ℝ) : Prop :=
  let v_dot_v := v.1 * v.1 + v.2 * v.2 + v.3 * v.3
  let v_dot_c := v.1 * 12 + v.2 * (-34) + v.3 * 6
  v_dot_v = v_dot_c

theorem volume_of_vectors_satisfying_condition :
  ∃ (V : ℝ), (volume_of_solid_formed_by_vectors v) → 
  V = (4 / 3) * π * (334)^(3/2) :=
by
  sorry

end volume_of_vectors_satisfying_condition_l643_643098


namespace sum_of_numbers_in_row_10_of_pascals_triangle_l643_643954

theorem sum_of_numbers_in_row_10_of_pascals_triangle : 
  (∑ k in finset.range 11, nat.choose 10 k) = 1024 := 
by {
  sorry
}

end sum_of_numbers_in_row_10_of_pascals_triangle_l643_643954


namespace no_polynomials_exist_l643_643473

theorem no_polynomials_exist :
  ¬ ∃ (P₁ P₂ P₃ P₄ : polynomial ℝ), 
    (∀ x, ∃ r, (P₁ + P₂ + P₃).eval r = 0 ∧ (P₁ + P₂ + P₄).eval r = 0 ∧ 
              (P₁ + P₃ + P₄).eval r = 0 ∧ (P₂ + P₃ + P₄).eval r = 0) ∧
    (∀ x, ∀ r, (P₁ + P₂).eval r ≠ 0 ∧ (P₁ + P₃).eval r ≠ 0 ∧ 
               (P₁ + P₄).eval r ≠ 0 ∧ (P₂ + P₃).eval r ≠ 0 ∧ 
               (P₂ + P₄).eval r ≠ 0 ∧ (P₃ + P₄).eval r ≠ 0) :=
by
  sorry

end no_polynomials_exist_l643_643473


namespace geometric_product_equality_l643_643009

open Real

noncomputable def tangent_point_D (BC : Line) (incircle : Circle) : Point :=
  sorry -- Given that D is the point of tangency of the incircle with BC

noncomputable def arbitrary_point_N (I D : Point) : Point :=
  sorry -- Given that N is an arbitrary point on segment ID

noncomputable def perpendicular_intersection (ID : Line) (N : Point) (circumcircle : Circle) : Point :=
  sorry -- The perpendicular to ID at N intersects the circumcircle

noncomputable def circumcenter_O (ABC : Triangle) : Point :=
  sorry -- Let O be the center of the circumcircle of triangle ABC

noncomputable def circumcenter_O1 (XIY : Triangle) : Point :=
  sorry -- Let O1 be the center of the circumcircle of triangle XIY

theorem geometric_product_equality 
  (BC : Line) (incircle : Circle) (I X Y : Point) (ABC XIY : Triangle) 
  (D : Point := tangent_point_D BC incircle)
  (N : Point := arbitrary_point_N I D)
  (O : Point := circumcenter_O ABC)
  (O1 : Point := circumcenter_O1 XIY) :
  OO1 * IN = Rr :=
sorry -- The product OO1 * IN equals Rr

end geometric_product_equality_l643_643009


namespace b_2022_eq_0_l643_643662

def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

def b (n : ℕ) : ℕ := fibonacci n % 4

theorem b_2022_eq_0 : b 2022 = 0 := 
by {
    -- This is where the proof would go
    sorry
}

end b_2022_eq_0_l643_643662


namespace EG_div_BC_eq_3_l643_643710

-- Definitions of the conditions
variables {A B C D E F G : Type*}

-- Assume triangles ABC and BCD are equilateral
axiom triangles_equilateral_ABC_BCD : equilateral_triangle A B C ∧ equilateral_triangle B C D

-- Assume squares ABEF and CDGH are constructed externally on sides AB and CD
axiom squares_constructed_ABEF_CDGH : external_square A B E F ∧ external_square C D G H

-- Prove the result
theorem EG_div_BC_eq_3 : EG / BC = 3 :=
by {
    have h_tri_eq : equilateral_triangle A B C ∧ equilateral_triangle B C D := triangles_equilateral_ABC_BCD,
    have h_sqr_ext : external_square A B E F ∧ external_square C D G H := squares_constructed_ABEF_CDGH,
    sorry
}

end EG_div_BC_eq_3_l643_643710


namespace vertex_in_second_quadrant_l643_643985

theorem vertex_in_second_quadrant :
  let f := λ x : ℝ, -(x + 1)^2 + 2
  let vertex_x := -1
  let vertex_y := 2
  vertex_x < 0 ∧ vertex_y > 0 :=
by
  let f := λ x : ℝ, -(x + 1)^2 + 2
  let vertex_x := -1
  let vertex_y := 2
  have h₁: vertex_x < 0 := by sorry
  have h₂: vertex_y > 0 := by sorry
  exact ⟨h₁, h₂⟩

end vertex_in_second_quadrant_l643_643985


namespace union_of_M_and_N_l643_643197

open Set

noncomputable def M : Set ℝ := {x | x < 1}
noncomputable def N : Set ℝ := {x | 2^x > 1}

theorem union_of_M_and_N : M ∪ N = univ :=
by
  sorry

end union_of_M_and_N_l643_643197


namespace find_c_minus_2d_l643_643336

theorem find_c_minus_2d :
  ∃ (c d : ℕ), (c > d) ∧ (c - 2 * d = 0) ∧ (∀ x : ℕ, (x^2 - 18 * x + 72 = (x - c) * (x - d))) :=
by
  sorry

end find_c_minus_2d_l643_643336


namespace smallest_yellow_marbles_l643_643277

theorem smallest_yellow_marbles (n : ℕ) (h_blues : n / 2 ∈ ℕ) (h_reds : n / 5 ∈ ℕ) (h_greens : 7 ∈ ℕ) :
  n % 10 = 0 → n > 0 → ∃ k : ℕ, n = k * 10 ∧ (n - ((n / 2) + (n / 5) + 7)) = 2 :=
by
  intros h1 h2
  have h3 : n = k * 10 := sorry
  sorry

end smallest_yellow_marbles_l643_643277


namespace problem_solution_l643_643517

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
noncomputable def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) := ∀ n, a_n (n + 1) = a_n n + d
noncomputable def arithmetic_sum (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) := ∀ n, S_n n = n * a_n 0 + (n * (n - 1) * d) / 2
noncomputable def condition_1 := S_n 3 = 9
noncomputable def geometric_sequence (a b c : ℝ) := b^2 = a * c
noncomputable def condition_2 := geometric_sequence (a_n 2 - 1) (a_n 3 - 1) (a_n 5 - 1)

-- Goal
theorem problem_solution (h_seq : arithmetic_sequence a_n d) (h_sum : arithmetic_sum S_n a_n)
  (h1 : condition_1) (h2 : condition_2) (h_d_nonzero : d ≠ 0) : S_n 5 = 25 := 
by
  sorry

end problem_solution_l643_643517


namespace father_and_daughter_age_l643_643421

-- A father's age is 5 times the daughter's age.
-- In 30 years, the father will be 3 times as old as the daughter.
-- Prove that the daughter's current age is 30 and the father's current age is 150.

theorem father_and_daughter_age :
  ∃ (d f : ℤ), (f = 5 * d) ∧ (f + 30 = 3 * (d + 30)) ∧ (d = 30 ∧ f = 150) :=
by
  sorry

end father_and_daughter_age_l643_643421


namespace platform_protection_l643_643498

noncomputable def max_distance (r : ℝ) (n : ℕ) : ℝ :=
  if n > 2 then r / (Real.sin (180.0 / n)) else 0

noncomputable def coverage_ring_area (r : ℝ) (w : ℝ) : ℝ :=
  let inner_radius := r * (Real.sin 20.0)
  let outer_radius := inner_radius + w
  Real.pi * (outer_radius^2 - inner_radius^2)

theorem platform_protection :
  let r := 61
  let w := 22
  let n := 9
  max_distance r n = 60 / Real.sin 20.0 ∧
  coverage_ring_area r w = 2640 * Real.pi / Real.tan 20.0 := by
  sorry

end platform_protection_l643_643498


namespace correct_statements_count_l643_643731

theorem correct_statements_count : 
    let s1 := ∀ (a: ℚ), abs a > 0
    let s2 := ∀ (a b: ℚ), abs a = abs b → a = b
    let s3 := ∀ (a: ℚ), abs a = abs (-a)
    let s4 := (¬ ∃ x: ℚ, ∀ y: ℚ, y ≤ x) ∧ (¬ ∃ z: ℚ, ∀ w: ℚ, w ≠ 0 → abs w < abs z)
    let s5 := ∀ (a : ℚ), ∃ (p q : ℤ), q ≠ 0 ∧ a = p / q
    let s6 := ∀ (a b: ℚ), (a * b < 0 → a = -b)
 in
    (¬ s1) ∧
    (¬ s2) ∧
    s3 ∧
    (¬ s4) ∧
    s5 ∧
    (¬ s6) →
    (3: ℕ) - (3 - 2: ℕ) = (2: ℕ) := 
by
    sorry

end correct_statements_count_l643_643731


namespace part_I_part_II_l643_643549

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

theorem part_I (m : ℝ) (h : ∀ x : ℝ, f x ≠ m) : m < -2 ∨ m > 2 :=
sorry

theorem part_II (P : ℝ × ℝ) (hP : P = (2, -6)) :
  (∃ m b : ℝ, ∀ x : ℝ, (m * x + b = 0 ∧ m = 3 ∧ b = 0) ∨ 
                 (m * x + b = 24 * x - 54 ∧ P.2 = 24 * P.1 - 54)) :=
sorry

end part_I_part_II_l643_643549


namespace compare_M_N_l643_643616

open Real

theorem compare_M_N (a : ℝ) : 
  let M := 2 * a * (a - 2)
  let N := (a + 1) * (a - 3)
  in M > N :=
by
  let M := 2 * a * (a - 2)
  let N := (a + 1) * (a - 3)
  sorry

end compare_M_N_l643_643616


namespace tangent_curves_alpha_l643_643853

theorem tangent_curves_alpha (α : ℝ) :
    (∃ p q : ℝ, q = α * p^2 + α * p + (1 / 24) ∧ p = α * q^2 + α * q + (1 / 24) ∧
       derivative (λ x, α * x^2 + α * x + (1 / 24)) p = derivative (λ y, α * y^2 + α * y + (1 / 24)) q)
    ↔ (α = (13 + Real.sqrt 601) / 12 ∨ α = (13 - Real.sqrt 601) / 12 ∨ α = 2 / 3 ∨ α = 3 / 2) :=
by
  sorry

end tangent_curves_alpha_l643_643853


namespace circumference_of_jogging_track_l643_643340

-- Definitions for the given conditions
def speed_deepak : ℝ := 4.5
def speed_wife : ℝ := 3.75
def meet_time : ℝ := 4.32

-- The theorem stating the problem
theorem circumference_of_jogging_track : 
  (speed_deepak + speed_wife) * meet_time = 35.64 :=
by
  sorry

end circumference_of_jogging_track_l643_643340


namespace temperature_or_daytime_not_sufficiently_high_l643_643808

variable (T : ℝ) (Daytime Lively : Prop)
axiom h1 : (T ≥ 75 ∧ Daytime) → Lively
axiom h2 : ¬ Lively

theorem temperature_or_daytime_not_sufficiently_high : T < 75 ∨ ¬ Daytime :=
by
  -- proof steps
  sorry

end temperature_or_daytime_not_sufficiently_high_l643_643808


namespace number_of_subsets_upper_bound_l643_643271

noncomputable def number_of_subsets_leq (n : ℕ) (λ : ℝ) (x : Fin n → ℝ) :=
  { A : Finset (Fin n) | ∑ i in A, x i ≥ λ }.card

theorem number_of_subsets_upper_bound (n : ℕ) (λ : ℝ) (x : Fin n → ℝ)
  (sum_eq_zero : ∑ i : Fin n, x i = 0)
  (sum_sq_eq_one : ∑ i : Fin n, x i ^ 2 = 1)
  (λ_pos : λ > 0) :
  number_of_subsets_leq n λ x ≤ 2^ (n-3) / λ^2 := by
  sorry

end number_of_subsets_upper_bound_l643_643271


namespace stddev_B_eq_stddev_A_l643_643956

universe u

-- Define the sample data for populations A and B
def A : List ℝ := [42, 4, 6, 52, 42, 50]
def B : List ℝ := A.map (λ x, x - 5)

-- Function to compute the standard deviation of a list of real numbers
noncomputable def stddev (l : List ℝ) : ℝ :=
  let mean := (l.sum / l.length : ℝ)
  let variance := (l.map (λ x, (x - mean)^2)).sum / l.length
  real.sqrt variance

-- The theorem to prove: the standard deviation of B is equal to the standard deviation of A
theorem stddev_B_eq_stddev_A : stddev B = stddev A := by
  sorry

end stddev_B_eq_stddev_A_l643_643956


namespace arithmetic_sequence_general_formula_inequality_satisfaction_l643_643146

namespace Problem

-- Definitions for the sequences and the sum of terms
def a (n : ℕ) : ℕ := sorry -- define based on conditions
def S (n : ℕ) : ℕ := sorry -- sum of first n terms of {a_n}
def b (n : ℕ) : ℕ := 2 * (S (n + 1) - S n) * S n - n * (S (n + 1) + S n)

-- Part 1: Prove the general formula for the arithmetic sequence
theorem arithmetic_sequence_general_formula :
  (∀ n : ℕ, b n = 0) → (∀ n : ℕ, a n = 0 ∨ a n = n) :=
sorry

-- Part 2: Conditions for geometric sequences and inequality
def a_2n_minus_1 (n : ℕ) : ℕ := 2 ^ n
def a_2n (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)
def b_2n (n : ℕ) : ℕ := sorry -- compute based on conditions
def b_2n_minus_1 (n : ℕ) : ℕ := sorry -- compute based on conditions

def b_condition (n : ℕ) : Prop := b_2n n < b_2n_minus_1 n

-- Prove the set of all positive integers n that satisfy the inequality
theorem inequality_satisfaction :
  { n : ℕ | b_condition n } = {1, 2, 3, 4, 5, 6} :=
sorry

end Problem

end arithmetic_sequence_general_formula_inequality_satisfaction_l643_643146


namespace complex_ordered_pairs_l643_643115

theorem complex_ordered_pairs :
  { (a, b : ℂ) // a^4 * b^7 = 1 ∧ a^8 * b^3 = 1 }.to_finset.card = 64 := 
sorry

end complex_ordered_pairs_l643_643115


namespace partition_Z_l643_643513

-- Define the set Z as points on the line whose distance from a given point is an integer.
def Z (p : ℝ) : Set ℝ := {x | ∃ n : ℤ, x = p + n}

-- Define a three-element subset H of Z.
def H (p : ℝ) (a b : ℤ) (h : 0 < a ∧ a ≤ b) : Set ℝ := {p, p + a, p + (a + b)}

-- The main theorem statement.
theorem partition_Z (p : ℝ) (a b : ℤ) (h : 0 < a ∧ a ≤ b) : 
  ∃ (S : Set ℝ) (T : Set ℝ → Set ℝ) (Hs : ∀ s ∈ S, ∃ (x : ℝ), T s = {x, x + a, x + (a + b)}), 
  ∀ (x : ℝ), x ∈ Z p → ∃ (t ∈ S), x ∈ T t :=
sorry

end partition_Z_l643_643513


namespace power_mean_bounds_neg_power_mean_bounds_l643_643265

variables {n : ℕ} {a : Fin n → ℝ} {q : Fin n → ℝ}
variable {r : ℚ}

noncomputable def power_mean (a : Fin n → ℝ) (q : Fin n → ℝ) (r : ℚ) : ℝ :=
  (∑ i, q i * (a i)^r)^(1/r)

theorem power_mean_bounds (ha_pos : ∀ i, 0 < a i)
    (ha_sorted : ∀ i j, i ≤ j → a i ≤ a j)
    (hq_pos : ∀ i, 0 < q i)
    (hq_sum : ∑ i, q i = 1)
    (hr_pos : 0 < r) :
    q (n-1)^(1/r) * a (n-1) ≤ power_mean a q r ∧ power_mean a q r ≤ a (n-1) := sorry

theorem neg_power_mean_bounds (ha_pos : ∀ i, 0 < a i)
    (ha_sorted : ∀ i j, i ≤ j → a i ≤ a j)
    (hq_pos : ∀ i, 0 < q i)
    (hq_sum : ∑ i, q i = 1)
    {s : ℚ} (hs_neg : s < 0) :
    a 0 ≤ power_mean a q s ∧ power_mean a q s ≤ q 0^(1/s) * a 0 := sorry

end power_mean_bounds_neg_power_mean_bounds_l643_643265


namespace function_fixed_point_l643_643942

noncomputable def fixed_point_of_function : Prop :=
  ∃ P : ℝ × ℝ, let x := P.1 in let y := P.2 in y = 3 + log a (x + 5) ∧ P = (-4, 3)

theorem function_fixed_point (a : ℝ) (ha : 1 < a) : fixed_point_of_function := sorry

end function_fixed_point_l643_643942


namespace eqn_y_value_l643_643471

theorem eqn_y_value (y : ℝ) (h : (2 / y) + ((3 / y) / (6 / y)) = 1.5) : y = 2 :=
sorry

end eqn_y_value_l643_643471


namespace simplify_expression_l643_643383

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (18 * x^3) * (4 * x^2) * (1 / (2 * x)^3) = 9 * x^2 :=
by
  sorry

end simplify_expression_l643_643383


namespace tile_plane_with_pentagons_or_heptagons_l643_643371

-- Definition of congruent shapes in Euclidean space
def congruent (s1 s2 : set (ℝ × ℝ)) : Prop :=
  ∃ f : ℝ × ℝ → ℝ × ℝ, isometry f ∧ f '' s1 = s2

-- Definitions of pentagon and heptagon tiling in the plane
def pentagon_tiling (S : set (set (ℝ × ℝ))) : Prop :=
  ∀ s ∈ S, congruent s (unit_square) ∧ ... -- more specific conditions to define pentagon tiling

def heptagon_tiling (S : set (set (ℝ × ℝ))) : Prop :=
  ∀ s ∈ S, congruent s (unit_square) ∧ ... -- more specific conditions to define heptagon tiling

-- The problem statement in Lean 4
theorem tile_plane_with_pentagons_or_heptagons :
  ∃ S : set (set (ℝ × ℝ)), (pentagon_tiling S ∨ heptagon_tiling S) :=
sorry

end tile_plane_with_pentagons_or_heptagons_l643_643371


namespace slope_probability_l643_643542

def line_equation (a x y : ℝ) : Prop := a * x + 2 * y - 3 = 0

def in_interval (a : ℝ) : Prop := -5 ≤ a ∧ a ≤ 4

def slope_not_less_than_1 (a : ℝ) : Prop := - a / 2 ≥ 1

noncomputable def probability_slope_not_less_than_1 : ℝ :=
  (2 - (-5)) / (4 - (-5))

theorem slope_probability :
  ∀ (a : ℝ), in_interval a → slope_not_less_than_1 a → probability_slope_not_less_than_1 = 1 / 3 :=
by
  intros a h_in h_slope
  sorry

end slope_probability_l643_643542


namespace max_a_l643_643157

open Real

theorem max_a (e : ℝ) (he : e = exp 1) :
  (∃ (a : ℝ), (∀ (x : ℝ), x ∈ Icc (1 / e) 2 → (a + e) * x - 1 - log x ≤ 0) ∧ 
  (∀ (a' : ℝ), (∀ (x : ℝ), x ∈ Icc (1 / e) 2 → (a' + e) * x - 1 - log x ≤ 0) → a' ≤ a)) :=
begin
  use -e,
  split,
  {
    intros x hx,
    sorry -- proof goes here
  },
  {
    intros a' ha',
    sorry -- proof goes here
  }
end

end max_a_l643_643157


namespace trig_comparison_inequality_l643_643907

theorem trig_comparison_inequality {f : ℝ → ℝ}
  (h_even : ∀ x, f x = f (-x))
  (h_increasing : ∀ a b, 0 ≤ a → a ≤ b → f a ≤ f b) :
  let a := f (Real.sin (50 * Real.pi / 180))
      b := f (Real.cos (50 * Real.pi / 180))
      c := f (Real.tan (50 * Real.pi / 180))
  in b < a ∧ a < c :=
by
  sorry

end trig_comparison_inequality_l643_643907


namespace midpoint_vector_sum_l643_643748

variables {V : Type*} [inner_product_space ℝ V]

variables (A B C A1 B1 C1 O : V)
variables (midpoint_AC1_A1 : (A1 = ((B + C) / 2)))
variables (midpoint_AB1_B1 : (B1 = ((A + C) / 2)))
variables (midpoint_BC1_C1 : (C1 = ((A + B) / 2)))

theorem midpoint_vector_sum :
  (\vec{O} + A1 + \vec{O} + B1 + \vec{O} + C1) =
  (\vec{O} + A + \vec{O} + B + \vec{O} + C) :=
by
  sorry

end midpoint_vector_sum_l643_643748


namespace plane_line_perpendicular_l643_643164

variables {α β : Type*} [plane α] [plane β]
variables {a b : Type*} [line a] [line b]

noncomputable def is_perpendicular (p : Type*) (q : Type*) : Prop :=
sorry -- Replace this with the actual definition of orthogonality/perpendicular if necessary

theorem plane_line_perpendicular
  (h_perp_planes : is_perpendicular α β)
  (h_line_in_plane_a : line_lies_in_plane a α)
  (h_line_in_plane_b : line_lies_in_plane b β)
  (h_perpendicular_lines : is_perpendicular a b) :
  (is_perpendicular a β) ∨ (is_perpendicular b α) :=
sorry

end plane_line_perpendicular_l643_643164


namespace contractor_engaged_days_l643_643039

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l643_643039


namespace angle_A_is_45_degrees_l643_643752

-- Define the right triangle and points P and Q based on the conditions
def triangle_ABC (A B C P Q : Type) [NoncomputableReal3] :=
  -- 1. ∆ABC is a right triangle at angle B
  ∃ A B C P Q : NoncomputableReal3,
    ∃ (angle_B : ℝ) (AC AP PQ BQ : ℝ),
    -- 2. Define the angles and side lengths
      angle_B = 90 ∧
      AC = 1 ∧  -- Assume unit length for simplicity
      AP = 1 ∧
      PQ = 1 ∧
      BQ = 1 ∧
    -- 3. Define that points P and Q are on the specified sides
      P ∈ line_segment_BC ∧
      Q ∈ line_segment_AB

-- Define the main theorem to prove the angle A is 45 degrees
theorem angle_A_is_45_degrees (A B C P Q : Type) [NoncomputableReal3] :
  triangle_ABC A B C P Q →
  ∠A = 45 :=
by
  sorry

end angle_A_is_45_degrees_l643_643752


namespace set_properties_proof_l643_643198

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := Icc (-2 : ℝ) 2)
variable (N : Set ℝ := Iic (1 : ℝ))

theorem set_properties_proof :
  (M ∪ N = Iic (2 : ℝ)) ∧
  (M ∩ N = Icc (-2 : ℝ) 1) ∧
  (U \ N = Ioi (1 : ℝ)) := by
  sorry

end set_properties_proof_l643_643198


namespace find_number_l643_643323

theorem find_number (x : ℕ) (h : x - 263 + 419 = 725) : x = 569 :=
sorry

end find_number_l643_643323


namespace valid_three_digit_numbers_count_correct_l643_643934

/-- 
  A valid three-digit number has the form ABC, 
  where 100 ≤ ABC ≤ 999 and A ≠ 0. 
  We exclude numbers where A + B = C.
 -/
def valid_three_digit_numbers_count : ℕ :=
  855

theorem valid_three_digit_numbers_count_correct :
  (count (λ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ let (A, B, C) := (n / 100, (n % 100) / 10, n % 10) in A ≠ 0 ∧ A + B ≠ C) (list.range 1000)) = valid_three_digit_numbers_count :=
by
  sorry

end valid_three_digit_numbers_count_correct_l643_643934


namespace ounces_per_jar_l643_643329

-- Define the necessary conditions
def food_per_half_pound : ℝ := 1 -- ounces
def half_pound : ℝ := 0.5 -- pounds
def total_turtle_weight : ℝ := 30 -- pounds
def jar_cost : ℝ := 2 -- dollars per jar
def total_feeding_cost : ℝ := 8 -- dollars

-- Prove how many ounces each jar contains
theorem ounces_per_jar:
  let total_food_needed := total_turtle_weight * (food_per_half_pound / half_pound) in
  let number_of_jars := total_feeding_cost / jar_cost in
  let ounces_per_jar := total_food_needed / number_of_jars in
  ounces_per_jar = 15 :=
by
  -- You would perform proof steps here, but we'll use sorry for now.
  sorry

end ounces_per_jar_l643_643329


namespace distribute_balls_into_boxes_l643_643208

theorem distribute_balls_into_boxes : 
  ∀ (balls : ℕ) (boxes : ℕ), balls = 6 ∧ boxes = 3 → boxes^balls = 729 :=
by
  intros balls boxes h
  have hb : balls = 6 := h.1
  have hbox : boxes = 3 := h.2
  rw [hb, hbox]
  show 3^6 = 729
  exact Nat.pow 3 6 -- this would expand to the actual computation
  sorry

end distribute_balls_into_boxes_l643_643208


namespace find_a_l643_643656

-- Define the functions f and g
def f (x : ℝ) : ℝ := x / 5 + 3
def g (x : ℝ) : ℝ := 4 - x

-- State the theorem that a = -6 satisfies f(g(a)) = 5
theorem find_a : ∃ a : ℝ, f (g a) = 5 ∧ a = -6 :=
by
  use -6
  simp [f, g]
  sorry

end find_a_l643_643656


namespace points_in_quadrant_I_l643_643470

def point_in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem points_in_quadrant_I (x y : ℝ) : (y > -x + 6 ∧ y > 3x - 2) → point_in_quadrant_I x y :=
by
  sorry

end points_in_quadrant_I_l643_643470


namespace find_c_l643_643914

def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c (c : ℝ) :
  (∀ x, f x c ≤ f 2 c) → c = 6 :=
sorry

end find_c_l643_643914


namespace max_value_isosceles_triangle_l643_643804

theorem max_value_isosceles_triangle (a b c : ℝ) (h_isosceles : b = c) :
  ∃ B, (∀ (a b c : ℝ), b = c → (b + c) / a ≤ B) ∧ B = 2 :=
by
  sorry

end max_value_isosceles_triangle_l643_643804


namespace fifth_derivative_l643_643111

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 - 7) * Real.log (x - 1)

theorem fifth_derivative :
  ∀ x, (deriv^[5] f) x = 8 * (x ^ 2 - 5 * x - 11) / ((x - 1) ^ 5) :=
by
  sorry

end fifth_derivative_l643_643111


namespace find_abc_l643_643358

def rearrangements (a b c : ℕ) : List ℕ :=
  [100 * a + 10 * b + c, 100 * a + 10 * c + b, 100 * b + 10 * a + c,
   100 * b + 10 * c + a, 100 * c + 10 * a + b, 100 * c + 10 * b + a]

theorem find_abc (a b c : ℕ) (habc : ℕ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (rearrangements a b c).sum = 2017 + habc →
  habc = 425 :=
by
  sorry

end find_abc_l643_643358


namespace number_of_five_digit_magical_numbers_l643_643735

def is_magical_number (n : ℕ) : Prop :=
  let digits := [0, 1, 6, 8, 9] in
  let digit_rotations := [0, 1, 9, 8, 6] in -- Rotations of 0, 1, 6, 8, 9 respectively
  let as_str := n.toString in
  as_str.length = 5 ∧
  all (fun d => d ∈ digits) as_str ∧
  as_str = (as_str.reverse.map (fun d =>
    digit_rotations[d] -- Map each digit to its rotated counterpart
  )).mkString

theorem number_of_five_digit_magical_numbers : ∀ (n : ℕ),
  (is_magical_number n) ↔ (n = 60) :=
begin
  sorry
end

end number_of_five_digit_magical_numbers_l643_643735


namespace sets_with_condition_l643_643462

noncomputable def point : Type := ℝ × ℝ

def is_circle_with_diameter (O A : point) (C : point) : Prop :=
  let d := (fst O - fst A) ^ 2 + (snd O - snd A) ^ 2
  (fst C - (fst O + fst A) / 2) ^ 2 + (snd C - (snd O + snd A) / 2) ^ 2 = d / 4

theorem sets_with_condition (O : point) (S : set point) :
  (∃ A B : point, A ∈ S ∧ B ∈ S ∧ A ≠ B ∧ A ≠ O ∧ B ≠ O ∧
  (∀ A ∈ S, ∀ (A ≠ O), ∀ C, is_circle_with_diameter O A C → C ∈ S)) →
  (S = set.univ ∨ ∃ T U, T ⊆ {P : point | (fst O - fst P) ^ 2 + (snd O - snd P) ^ 2 = 1} ∧ 
  (U = {P : point | (fst O - fst P) ^ 2 + (snd O - snd P) ^ 2 < 1}) ∧ (S = T ∪ U)) :=
sorry

end sets_with_condition_l643_643462


namespace correct_option_l643_643730

-- Define the various expressions as propositions
def optionA (a : ℕ) : Prop := a^2 * a^5 = a^10
def optionB (a : ℕ) : Prop := (3 * a^3)^2 = 6 * a^6
def optionC (a : ℕ) : Prop := 4 * a - 3 * a = a
def optionD (a : ℕ) : Prop := (a + 1)^2 = a^2 + 1

-- Prove that optionC is correct
theorem correct_option (a : ℕ) : optionC a :=
by {
  rw optionC, -- Expose the content of optionC
  sorry -- Proof of 4 * a - 3 * a = a
}

end correct_option_l643_643730


namespace bob_catches_up_l643_643990

-- Define speeds and initial conditions
def john_speed : ℝ := 2  -- miles per hour
def bob_speed : ℝ := 6   -- miles per hour
def bob_initial_distance : ℝ := 2  -- miles west of John

-- Define the relative speed
def relative_speed : ℝ := bob_speed - john_speed

-- Define the time it takes for Bob to catch up to John
def time_to_catch_up : ℝ := bob_initial_distance / relative_speed

-- Define the conversion from hours to minutes
def time_in_minutes : ℝ := time_to_catch_up * 60

-- State the theorem
theorem bob_catches_up (h : john_speed = 2 ∧ bob_speed = 6 ∧ bob_initial_distance = 2) : time_in_minutes = 30 := by
  sorry

end bob_catches_up_l643_643990


namespace car_fuel_efficiency_l643_643026

theorem car_fuel_efficiency (distance gallons fuel_efficiency D : ℝ)
  (h₀ : fuel_efficiency = 40)
  (h₁ : gallons = 3.75)
  (h₂ : distance = 150)
  (h_eff : fuel_efficiency = distance / gallons) :
  fuel_efficiency = 40 ∧ (D / fuel_efficiency) = (D / 40) :=
by
  sorry

end car_fuel_efficiency_l643_643026


namespace claudia_earnings_over_weekend_l643_643829

theorem claudia_earnings_over_weekend :
  ∀ (cost_per_kid : ℕ) (saturday_kids : ℕ) (sunday_kids : ℕ),
  cost_per_kid = 10 →
  saturday_kids = 20 →
  sunday_kids = saturday_kids / 2 →
  (saturday_kids + sunday_kids) * cost_per_kid = 300 :=
by
  intros cost_per_kid saturday_kids sunday_kids
  assume h1 : cost_per_kid = 10
  assume h2 : saturday_kids = 20
  assume h3 : sunday_kids = saturday_kids / 2
  sorry

end claudia_earnings_over_weekend_l643_643829


namespace tire_circumference_4_meters_l643_643740

theorem tire_circumference_4_meters :
  (∀ (rpm : ℕ) (speed_km_h : ℕ), rpm = 400 → speed_km_h = 96 → ((speed_km_h * 1000 / 60) / rpm) = 4) :=
by {
  intros rpm speed_km_h rpm_400 speed_96,
  rw [rpm_400, speed_96],
  norm_num,
  sorry
}

end tire_circumference_4_meters_l643_643740


namespace complement_intersection_l643_643558

open Set

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection :
  (U \ M) ∩ N = {3} :=
sorry

end complement_intersection_l643_643558


namespace max_exchanges_l643_643360

theorem max_exchanges (n : ℕ) (h : Fin n → ℕ) (h_increasing : ∀ i j, i < j → h i < h j)
  (allowed_switch : ∀ i j, i < j → h i ≤ h (j - 2) → True) :
  ∃ exchanges, exchanges ≤ Nat.choose n 3 :=
sorry

end max_exchanges_l643_643360


namespace g_96_is_1496_l643_643465

noncomputable def g : ℤ → ℤ
| n => if n >= 1500 then n - 4 else g (g (n + 6))

theorem g_96_is_1496 : g 96 = 1496 :=
by
  sorry

end g_96_is_1496_l643_643465


namespace num_partitions_A_eq_9_l643_643580

def partitions (A : Set ℕ) : Set (Set ℕ × Set ℕ) := 
  {p | (p.1 ∪ p.2 = A) ∧ (∀ (q : Set ℕ × Set ℕ), (p.1 = q.2 ∧ p.2 = q.1 → p.1 = p.2) )}

def A := {1, 2}

theorem num_partitions_A_eq_9 : 
  (partitions A).card = 9 :=
sorry

end num_partitions_A_eq_9_l643_643580


namespace small_triangles_perimeter_l643_643607

-- Given conditions:
variables (T : Triangle) (nine_smaller_triangles: List Triangle)
variable (H1 : T.perimeter = 120)
variable (H2 : nine_smaller_triangles.length = 9)
variable (H3 : ∀ t ∈ nine_smaller_triangles, t.perimeter = (nine_smaller_triangles.head).perimeter)

-- Desired proof statement:
theorem small_triangles_perimeter (T : Triangle) (nine_smaller_triangles : List Triangle)
  (H1 : T.perimeter = 120)
  (H2 : nine_smaller_triangles.length = 9)
  (H3 : ∀ t ∈ nine_smaller_triangles, t.perimeter = (nine_smaller_triangles.head).perimeter) :
  (nine_smaller_triangles.head).perimeter = 40 := 
sorry

end small_triangles_perimeter_l643_643607


namespace alpha_sin_beta_lt_beta_sin_alpha_l643_643296

variable {α β : ℝ}

theorem alpha_sin_beta_lt_beta_sin_alpha (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi / 2) : 
  α * Real.sin β < β * Real.sin α := 
by
  sorry

end alpha_sin_beta_lt_beta_sin_alpha_l643_643296


namespace angles_equal_then_sides_equal_l643_643305

theorem angles_equal_then_sides_equal
  {A B C : Type}
  {triangle : A → B → C → Prop}
  {angle : B → B → B → A}
  {side : A → C}
  (a1 a2 a3 : B)
  (s1 s2 s3 : A)
  (h1 : triangle s1 s2 s3)
  (h2 : angle a1 a2 a3 = angle a1 a3 a2)
  : side s1 = side s3 := 
sorry

end angles_equal_then_sides_equal_l643_643305


namespace candidate_votes_l643_643977

variable (E S C : ℕ)

-- Definitions based on given conditions
def eliot_votes_eq : E = 160 := sorry
def eliot_votes_double_shaun : E = 2 * S := sorry
def shaun_votes_five_times_candidate : S = 5 * C := sorry

-- The final theorem to prove
theorem candidate_votes : C = 16 :=
  by
  -- Definitions
  have h1 : E = 160 := eliot_votes_eq
  have h2 : E = 2 * S := eliot_votes_double_shaun
  have h3 : S = 5 * C := shaun_votes_five_times_candidate
  -- Start the proof
  sorry

end candidate_votes_l643_643977


namespace Vanya_can_not_score_100_in_more_than_one_exam_l643_643018

def Vanya_exams (r p m : ℕ) : Prop :=
  r = p - 5 ∧ p = m - 9 ∧ ∀ k : ℕ, ∀ op : ℕ → ℕ → ℕ → (ℕ × ℕ × ℕ),
    (op = λ r p m => (r + 1, p + 1, m + 1) ∨ 
    op = λ r p m => (r - 3, p + 1, m + 1) ∨ 
    op = λ r p m => (r + 1, p - 3, m + 1) ∨ 
    op = λ r p m => (r + 1, p + 1, m - 3)) →
    let (r', p', m') := op r p m in
    r' <= 100 ∧ p' <= 100 ∧ m' <= 100 →
    (r' = 100 ∨ p' = 100 ∨ m' = 100) → ¬(r' = 100 ∧ p' = 100 ∧ m' = 100)

theorem Vanya_can_not_score_100_in_more_than_one_exam (r p m : ℕ) : 
  Vanya_exams r p m → 
  (¬(∃ r' p' m', r' = 100 ∧ p' = 100 ∧ m' = 100)) := 
by
  sorry

end Vanya_can_not_score_100_in_more_than_one_exam_l643_643018


namespace jerry_boxes_l643_643251

theorem jerry_boxes (boxes_sold boxes_left : ℕ) (h₁ : boxes_sold = 5) (h₂ : boxes_left = 5) : (boxes_sold + boxes_left = 10) :=
by
  sorry

end jerry_boxes_l643_643251


namespace num_ways_to_distribute_balls_l643_643211

-- Define the conditions
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- The statement to prove
theorem num_ways_to_distribute_balls : num_boxes ^ num_balls = 729 :=
by {
  -- Proof steps would go here
  sorry
}

end num_ways_to_distribute_balls_l643_643211


namespace infinite_set_of_symmetric_points_l643_643612

theorem infinite_set_of_symmetric_points 
  (M : Set (EuclideanPlane)) 
  (h1 : 2 ≤ M.card) 
  (g1 g2 : Line) 
  (h2 : (∃ O : Point, O = g1 ∩ g2) ∧ (∃ q : ℝ, irrational q ∧ α = q * π)) : 
  M.card = ⊤ := 
sorry

end infinite_set_of_symmetric_points_l643_643612


namespace students_in_class_l643_643233

theorem students_in_class
  (B : ℕ) (E : ℕ) (G : ℕ)
  (h1 : B = 12)
  (h2 : G + B = 22)
  (h3 : E = 10) :
  G + E + B = 32 :=
by
  sorry

end students_in_class_l643_643233


namespace number_of_true_propositions_is_one_l643_643024

-- Define each proposition as a Boolean in Lean
def Proposition1 : Prop :=
  ∀ solid, (solid.has_identical_views → solid.is_cube)

def Proposition2 : Prop :=
  ∀ solid, (solid.front_view_is_rectangle ∧ solid.top_view_is_rectangle → solid.is_cuboid)

def Proposition3 : Prop :=
  ∀ solid, (solid.all_views_are_rectangles → solid.is_cuboid)

def Proposition4 : Prop :=
  ∀ solid, (solid.front_view_is_isosceles_trapezoid ∧ solid.side_view_is_isosceles_trapezoid → solid.is_frustum)

-- Prove that exactly one of the propositions is true
theorem number_of_true_propositions_is_one (solid : Solid) :
  (¬ Proposition1 solid ∧ ¬ Proposition2 solid ∧ Proposition3 solid ∧ ¬ Proposition4 solid) :=
begin
  -- Proof is omitted
  sorry
end

end number_of_true_propositions_is_one_l643_643024


namespace equation_of_line_BC_l643_643559

/-
Given:
1. Point A(3, -1)
2. The line containing the median from A to side BC: 6x + 10y - 59 = 0
3. The line containing the angle bisector of ∠B: x - 4y + 10 = 0

Prove:
The equation of the line containing side BC is 2x + 9y - 65 = 0.
-/

noncomputable def point_A : (ℝ × ℝ) := (3, -1)

noncomputable def median_line (x y : ℝ) : Prop := 6 * x + 10 * y - 59 = 0

noncomputable def angle_bisector_line_B (x y : ℝ) : Prop := x - 4 * y + 10 = 0

theorem equation_of_line_BC :
  ∃ (x y : ℝ), 2 * x + 9 * y - 65 = 0 :=
sorry

end equation_of_line_BC_l643_643559


namespace green_beans_count_l643_643328

def total_beans := 572
def red_beans := (1 / 4) * total_beans
def remaining_after_red := total_beans - red_beans
def white_beans := (1 / 3) * remaining_after_red
def remaining_after_white := remaining_after_red - white_beans
def green_beans := (1 / 2) * remaining_after_white

theorem green_beans_count : green_beans = 143 := by
  sorry

end green_beans_count_l643_643328


namespace remainder_mod_7_l643_643742

theorem remainder_mod_7 (n m p : ℕ) 
  (h₁ : n % 4 = 3)
  (h₂ : m % 7 = 5)
  (h₃ : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 :=
by
  sorry

end remainder_mod_7_l643_643742


namespace andy_bethany_sum_l643_643443

theorem andy_bethany_sum (a : ℕ → ℕ → ℝ)
  (A : ℝ) (B : ℝ)
  (h1 : A = (∑ j in Finset.range 100, ∑ i in Finset.range 50, a i j) / 100)
  (h2 : B = (∑ i in Finset.range 50, ∑ j in Finset.range 100, a i j) / 50)
  (h3 : ∑ j in Finset.range 100, ∑ i in Finset.range 50, a i j = ∑ i in Finset.range 50, ∑ j in Finset.range 100, a i j) :
  A / B = 1 / 2 :=
by
  sorry

end andy_bethany_sum_l643_643443


namespace smallest_n_such_that_f_n_gt_24_l643_643619

-- Define the function f(n) as described
def f(n : ℕ) : ℕ :=
  if n = 0 then 0
  else Nat.find_greatest (λ k, n ∣ (Nat.factorial k))

-- Prove that the smallest n = 696 for which f(n) > 24
theorem smallest_n_such_that_f_n_gt_24 :
  ∃ n : ℕ, f(n) > 24 ∧ ∀ m : ℕ, f(m) > 24 → m ≥ n :=
begin
  use 696,
  split,
  { -- Prove f(696) > 24
    sorry },
  { -- Prove that 696 is the smallest such n
    intros m h,
    sorry }
end

end smallest_n_such_that_f_n_gt_24_l643_643619


namespace math_problem_l643_643477

theorem math_problem :
  (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 :=
by
  sorry

end math_problem_l643_643477


namespace part1_part2_part3_l643_643912

noncomputable def g : ℝ → ℝ := λ x, 2^x

def f (x : ℝ) (a b : ℝ) : ℝ := (-g x + a) / (2 * g x + b)

theorem part1 (a b : ℝ) (h1 : g 2 = 4) (h2 : ∀ x, f (-x) a b = -f x a b) : a = 1 ∧ b = 2 :=
sorry

theorem part2 (a b : ℝ) (h1 : g 2 = 4) (h2 : ∀ x, f (-x) a b = -f x a b) (h3 : a = 1) (h4 : b = 2) :
  (∀ x y : ℝ, x < y → f x a b > f y a b) :=
sorry

theorem part3 (a b : ℝ) (k : ℝ) (h1 : g 2 = 4) (h2 : ∀ x, f (-x) a b = -f x a b)
  (h3 : a = 1) (h4 : b = 2) (h_decreasing : ∀ x y : ℝ, x < y → f x a b > f y a b)
  (h5 : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → f(t^2 - 2 * t) + f(2 * t^2 - k) > 0) : k > 1 :=
sorry

end part1_part2_part3_l643_643912


namespace slip_5_5_in_cup_D_l643_643057

def paper_slips := [1.0, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 3.5, 4.0, 4.0, 4.5, 5.0, 5.5, 6.0]
def cups := ['A', 'B', 'C', 'D', 'E', 'F']
def sum_seq := list.range (list.length cups)

axiom paper_sum : (paper_slips.sum = 49.5)
axiom organized_sums : (∀ (i j : ℕ), i < j → sum_seq[i] < sum_seq[j])
axiom cup_F_holds_two : (2 ∈ slips_in_cup cups[5])
axiom cup_C_holds_four : (4 ∈ slips_in_cup cups[2])

theorem slip_5_5_in_cup_D :
  (5.5 ∈ slips_in_cup cups[3]) :=
sorry

end slip_5_5_in_cup_D_l643_643057


namespace trigonometric_identity_l643_643652

theorem trigonometric_identity 
  (x : ℝ)
  (h1 : ∀ θ : ℝ, cot θ - 2 * cot (2 * θ) = tan θ) :
  cot x + 2 * cot (2 * x) + 4 * cot (4 * x) + 8 * tan (8 * x) = cot x := 
sorry

end trigonometric_identity_l643_643652


namespace number_of_integers_l643_643489

theorem number_of_integers (n: ℕ) (h1 : n > 1) (h2 : ∀ a: ℤ, n ∣ (a ^ 25 - a)):
  n = 2730 :=
sorry

end number_of_integers_l643_643489


namespace exchange_rate_lire_l643_643441

theorem exchange_rate_lire (x : ℕ) (h : 2500 / 2 = x / 5) : x = 6250 :=
by
  sorry

end exchange_rate_lire_l643_643441


namespace range_of_a_l643_643943

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) → -2 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l643_643943


namespace even_number_of_participants_l643_643811

-- Definition of the problem condition as a predicate
def odd_draw_exists (P : Finset ℕ) (draws : ℕ → ℕ) : Prop :=
  ∀ S ⊆ P, ∃ p ∈ S, draws p % 2 = 1

-- The main theorem to prove
theorem even_number_of_participants 
  (n : ℕ) (P : Finset ℕ) (hP : card P = n)
  (draws : ℕ → ℕ) (h : odd_draw_exists P draws) :
  n % 2 = 0 :=
by 
  sorry

end even_number_of_participants_l643_643811


namespace center_number_in_grid_l643_643061

theorem center_number_in_grid:
    (∀ i j : Fin 3, ∃ n : Fin 9, grid i j = n) →
    (∀ n : Fin 8, ∃ i1 j1 i2 j2 : Fin 3, grid i1 j1 = n ∧ grid i2 j2 = n + 1 ∧
        ((i1 = i2 ∧ (j1 = j2 + 1 ∨ j2 = j1 + 1)) ∨ (j1 = j2 ∧ (i1 = i2 + 1 ∨ i2 = i1 + 1)))) →
    (grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 18) →
    grid 1 1 = 7 :=
by
  intro h_nums h_adj h_sum
  sorry

end center_number_in_grid_l643_643061


namespace spaceship_speed_400_l643_643130

def spaceship_speed (n : ℕ) : ℝ :=
  if n = 200 then 500 else
  if n < 200 then 0 else
  spaceship_speed (n - 100) / 2

theorem spaceship_speed_400 : spaceship_speed 400 = 125 :=
by {
  -- Given conditions
  have h1 : spaceship_speed 200 = 500 := rfl,
  have h2 : spaceship_speed (200 + 100) = spaceship_speed 200 / 2,
  have h3 : spaceship_speed (200 + 200) = spaceship_speed (200 + 100) / 2,
  -- Calculations based on conditions
  simp [spaceship_speed] at *,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end spaceship_speed_400_l643_643130


namespace car_turns_proof_l643_643056

def turns_opposite_direction (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 180

theorem car_turns_proof
  (angle1 angle2 : ℝ)
  (h1 : (angle1 = 50 ∧ angle2 = 130) ∨ (angle1 = -50 ∧ angle2 = 130) ∨ 
       (angle1 = 50 ∧ angle2 = -130) ∨ (angle1 = 30 ∧ angle2 = -30)) :
  turns_opposite_direction angle1 angle2 ↔ (angle1 = 50 ∧ angle2 = 130) :=
by
  sorry

end car_turns_proof_l643_643056


namespace bus_speed_including_stoppages_l643_643849

theorem bus_speed_including_stoppages 
  (speed_excl_stoppages : ℚ) 
  (ten_minutes_per_hour : ℚ) 
  (bus_stops_for_10_minutes : ten_minutes_per_hour = 10/60) 
  (speed_is_54_kmph : speed_excl_stoppages = 54) : 
  (speed_excl_stoppages * (1 - ten_minutes_per_hour)) = 45 := 
by 
  sorry

end bus_speed_including_stoppages_l643_643849


namespace pete_raymond_money_received_l643_643291

theorem pete_raymond_money_received
  (P R : ℕ)  -- P and R are natural numbers representing the money received by Pete and Raymond respectively
  (H_pete_spent : P - 20)  -- Pete spends 20 cents (4 nickels)
  (H_raymond_left : 70) -- Raymond has 70 cents left (7 dimes)
  (H_total_spent : ((P - 20) + (R - 70)) = 200) : -- Together they spent 200 cents
  P + R = 290 := -- The sum of the money they received is 290 cents
by
  sorry

end pete_raymond_money_received_l643_643291


namespace range_of_k_l643_643906

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 ≤ k * x^2 + k * x + 3) :
  0 ≤ k ∧ k ≤ 12 :=
sorry

end range_of_k_l643_643906


namespace min_sum_terms_of_arithmetic_sequence_l643_643494

variable {α : Type*} [linear_ordered_add_comm_group α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem min_sum_terms_of_arithmetic_sequence 
  (a : ℕ → α) 
  (S : ℕ → α) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : ∀ n : ℕ, S n = n * (a 1 + a n) / 2) 
  (h_a1_neg : a 1 < 0) 
  (h_S4_eq_S11 : S 4 = S 11) :
  (∃ n, S n = min (S n)) → (n = 7 ∨ n = 8) :=
sorry

end min_sum_terms_of_arithmetic_sequence_l643_643494


namespace max_value_f_l643_643505

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

theorem max_value_f : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 9 → f x ≤ 4 :=
by
  intros x hx
  have h1 : 1 ≤ x, from hx.1,
  have h2 : x ≤ 9, from hx.2,
  have f_monotonic : ∀ {a b : ℝ}, 1 ≤ a → a ≤ b → b ≤ 9 → f a ≤ f b, sorry,
  have f_9 : f 9 = 4, sorry,
  calc
    f x ≤ f 9     : f_monotonic h1 (le_refl x) h2
    ... = 4       : f_9

end max_value_f_l643_643505


namespace ABCD_angle_bisectors_meet_at_P_l643_643257

variable (A B C D P : Type) [OrderedGeometry A B C D]
variable (α β γ δ : ℝ)
variable [Plane A]

noncomputable def convex_quadrilateral (A B C D : Type) : Prop :=
∀ {X : Type}, {x : A → ℝ // geom.affine_quadrilateral A B C D}

theorem ABCD_angle_bisectors_meet_at_P 
  (h1 : convex_quadrilateral A B C D)
  (h2 : ∃ P, bisects (angle B A C) P ∧ bisects (angle B D C) P)
  (h3 : angle A P B = angle C P D)
  : length A B + length B D = length A C + length C D := 
sorry

end ABCD_angle_bisectors_meet_at_P_l643_643257


namespace telephone_pole_height_proof_l643_643792

theorem telephone_pole_height_proof (
  (AC := 6 : ℝ)
  (AD := 4 : ℝ)
  (DE := 1.8 : ℝ)
) : 
   let DC := AC - AD in
   let ratio := DE / DC in
   let AB := ratio * AC in
   AB = 5.4 :=
by
  let DC := AC - AD
  let ratio := DE / DC
  let AB := ratio * AC
  have h : AB = 5.4 := sorry
  exact h

end telephone_pole_height_proof_l643_643792


namespace contrapositive_example_l643_643335

theorem contrapositive_example (x : ℝ) : (x > 2 → x > 0) ↔ (x ≤ 2 → x ≤ 0) :=
by
  sorry

end contrapositive_example_l643_643335


namespace cars_sold_on_tuesday_l643_643027

theorem cars_sold_on_tuesday (x : ℕ) :
  let monday := 8,
      wednesday := 10,
      thursday := 4,
      friday := 4,
      saturday := 4,
      mean := 5.5,
      days := 6 in
  let total_cars := mean * days in
  let total_without_tuesday := monday + wednesday + thursday + friday + saturday in
  total_cars = 33 ∧ total_without_tuesday = 30 → x = 3 :=
by
  sorry

end cars_sold_on_tuesday_l643_643027


namespace part1_part2_l643_643623

variable (c : ℝ) (x : ℝ)
def f (x : ℝ) (c : ℝ) := |x - c|

theorem part1 (c : ℝ) (x : ℝ) : f x c + f (-x⁻¹) c ≥ 2 :=
  sorry

theorem part2 (h : c > 2) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |f (1 / 2 * x + c) c - 1 / 2 * f x c | ≤ 1) →
  c = 4 :=
  sorry

end part1_part2_l643_643623


namespace length_of_arc_RP_l643_643238

variables {O R P : Type*} {OR : ℝ} (angle_RIP : ℝ)

def is_center (O : Type*) := true

def is_radius (O R : Type*) (OR : ℝ) := OR = 10

def angle_measure_RIP (angle_RIP : ℝ) := angle_RIP = 36

theorem length_of_arc_RP :
    is_center O →
    is_radius O R OR →
    angle_measure_RIP angle_RIP →
    let arc_length := (angle_RIP / 360) * (2 * Real.pi * OR)
    in arc_length = 4 * Real.pi :=
by
  intros hO hOR hangle
  have hyp1: OR = 10 := hOR
  have hyp2: angle_RIP = 36 := hangle
  calc
    let arc_length := (angle_RIP / 360) * (2 * Real.pi * OR)
    arc_length = (36 / 360) * (2 * Real.pi * 10) : by sorry
  ...
  sorry

end length_of_arc_RP_l643_643238


namespace raft_people_with_life_jackets_l643_643701

theorem raft_people_with_life_jackets (n m k : ℕ) (h1 : n = 21) (h2 : m = n - 7) (h3 : k = 8) :
  n - (k / (m / (n - m))) = 17 := 
by sorry

end raft_people_with_life_jackets_l643_643701


namespace number_of_integers_l643_643866

theorem number_of_integers (n : ℤ) : 20 < n^2 → n^2 < 200 → (finset.Icc 5 14).card + (finset.Icc -14 -5).card = 20 := by
  sorry

end number_of_integers_l643_643866


namespace cyclic_quadrilateral_DLCP_l643_643521

open EuclideanGeometry

variables {A B C L D K P : Point}
variables (ABC : Triangle A B C)
variables (L CL : Line)
variables (CL_angle_bisector : angleBisectorLine ABC CL)
variables (Γ : Circle)
variables (C_median_D : D pointsOnCircle Γ ∧ medianIntersectionPoint ABC C D)
variables (K_midpoint_arc : arcMidpointOfCircumcircle Γ K A C B)
variables (P_symmetric_L : symmetricPoint P L (tangentAtPoint Γ K))

theorem cyclic_quadrilateral_DLCP :
  cyclicQuadrilateral D L C P :=
sorry

end cyclic_quadrilateral_DLCP_l643_643521


namespace inverse_of_3_mod_185_l643_643852

theorem inverse_of_3_mod_185 : ∃ x : ℕ, 0 ≤ x ∧ x < 185 ∧ 3 * x ≡ 1 [MOD 185] :=
by
  use 62
  sorry

end inverse_of_3_mod_185_l643_643852


namespace right_triangles_sides_l643_643342

theorem right_triangles_sides (a b c p S r DH FC FH: ℝ)
  (h₁ : a = 10)
  (h₂ : b = 10)
  (h₃ : c = 12)
  (h₄ : p = (a + b + c) / 2)
  (h₅ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₆ : r = S / p)
  (h₇ : DH = (c / 2) - r)
  (h₈ : FC = (a * r) / DH)
  (h₉ : FH = Real.sqrt (FC^2 - DH^2))
: FC = 3 ∧ DH = 4 ∧ FH = 5 := by
  sorry

end right_triangles_sides_l643_643342


namespace find_a_range_l643_643900

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- The main theorem stating the range of a
theorem find_a_range (a : ℝ) (h : ¬(∃ x : ℝ, p a x) → ¬(∃ x : ℝ, q x) ∧ ¬(¬(∃ x : ℝ, q x) → ¬(∃ x : ℝ, p a x))) : 1 < a ∧ a ≤ 2 := sorry

end find_a_range_l643_643900


namespace triangle_abc_bc_10_median_ad_6_altitude_5_l643_643605

theorem triangle_abc_bc_10_median_ad_6_altitude_5 :
  ∀ (A B C P: Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P],
  let BC := dist B C,
      AD := dist A D,
      h := dist A P
  in
  BC = 10 ∧ AD = 6 ∧ h = 5 →
  let AB_sq := dist A B ^ 2,
      AC_sq := dist A C ^ 2,
      AB_AC_sq_sum := AB_sq + AC_sq,
      N := max AB_AC_sq_sum,
      n := min AB_AC_sq_sum
  in
  N - n = 0 := by
  intros A B C P _ _ _ _ BC AD h BC_eq AD_eq h_eq AB_sq AC_sq AB_AC_sq_sum N n,
  rw [BC_eq, AD_eq, h_eq],
  sorry

end triangle_abc_bc_10_median_ad_6_altitude_5_l643_643605


namespace cannot_determine_log_20_div_27_l643_643503

noncomputable def log5 : ℝ := 0.6990
noncomputable def log3 : ℝ := 0.4771

theorem cannot_determine_log_20_div_27 : 
  ¬ ∃ log2 : ℝ, 
    log2 + (2 * log5) - (3 * log3) = (log 20 - log 27) :=
sorry

end cannot_determine_log_20_div_27_l643_643503


namespace cosine_of_angle_is_correct_l643_643110

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D :=
{ x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

def dot_product (v1 v2 : Point3D) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

noncomputable def cosine_angle (A B C : Point3D) : ℝ :=
let AB := vector_sub A B in
let AC := vector_sub A C in
(dot_product AB AC) / (magnitude AB * magnitude AC)

theorem cosine_of_angle_is_correct :
  let A := Point3D.mk 3 3 (-1) in
  let B := Point3D.mk 5 5 (-2) in
  let C := Point3D.mk 4 1 1 in
  cosine_angle A B C = -4 / 9 :=
by
  let A := Point3D.mk 3 3 (-1) in
  let B := Point3D.mk 5 5 (-2) in
  let C := Point3D.mk 4 1 1 in
  have h1 : vector_sub A B = {x := 2, y := 2, z := -1} := by sorry,
  have h2 : vector_sub A C = {x := 1, y := -2, z := 2} := by sorry,
  have dot_product_AB_AC : dot_product (vector_sub A B) (vector_sub A C) = -4 := by sorry,
  have magnitude_AB : magnitude (vector_sub A B) = 3 := by sorry,
  have magnitude_AC : magnitude (vector_sub A C) = 3 := by sorry,
  show cosine_angle A B C = -4 / 9, from sorry

end cosine_of_angle_is_correct_l643_643110


namespace find_radius_of_inscribed_circle_l643_643289

-- Given conditions
def A := (0 : ℝ)
def B := (14 : ℝ)
def C := (42 : ℝ) -- Since BC = 28, C = B + 28 = 14 + 28 = 42
def R := 6 -- The radius of the circle that touches all three semicircles

-- Definition of the semicircles
def semi_circle_radius_AB := B / 2
def semi_circle_radius_BC := (C - B) / 2
def semi_circle_radius_AC := (C - A) / 2

-- Intermediate distances based on the centers of semicircles and the circle that touches them
def O1O3 := semi_circle_radius_AB + R -- 7 + R
def OO3 := semi_circle_radius_AC - R -- 21 - R
def O2O3 := semi_circle_radius_BC + R -- 14 + R

-- We aim to prove the equivalent Lean statement encapsulating the given conditions and correct answer
theorem find_radius_of_inscribed_circle :
  ∃ (R : ℝ), (14 : ℝ = 2 * 7) ∧ (2 * (21 - R) = 14 + R) ∧ (14 + R = 2 * (7 + R)) ∧ R = 6 :=
by
  sorry

end find_radius_of_inscribed_circle_l643_643289


namespace solve_for_y_l643_643319

theorem solve_for_y (y : ℝ) (h : ∛(5 - 2 / y) = -3) : y = 1/16 :=
sorry

end solve_for_y_l643_643319


namespace pump_time_643_minutes_l643_643761

theorem pump_time_643_minutes
  (length : ℝ) (length_eq : length = 30)
  (width : ℝ) (width_eq : width = 40)
  (depth : ℝ) (depth_eq : depth = 24 / 12)
  (gallons_per_cubic_foot : ℝ) (gallons_per_cubic_foot_eq : gallons_per_cubic_foot = 7.5)
  (rate_pump1 : ℝ) (rate_pump1_eq : rate_pump1 = 8)
  (rate_pump2 : ℝ) (rate_pump2_eq : rate_pump2 = 8)
  (rate_pump3 : ℝ) (rate_pump3_eq : rate_pump3 = 12)
  : (length * width * depth * gallons_per_cubic_foot / (rate_pump1 + rate_pump2 + rate_pump3)).ceil = 643 :=
by {
  -- skip the proof
  sorry
}

end pump_time_643_minutes_l643_643761


namespace data_set_is_1_1_3_3_l643_643429

open Real

def is_positive_integer (x : ℝ) : Prop := x > 0 ∧ x = floor x

def avg (s: List ℝ) : ℝ := (s.sum) / (s.length)

def median (s: List ℝ) : ℝ :=
  let sorted := list.qsort (≤) s
  if sorted.length % 2 = 1 then sorted.nth_le (sorted.length / 2) (by linarith)
  else (sorted.nth_le (sorted.length / 2 - 1) (by linarith) + sorted.nth_le (sorted.length / 2) (by linarith)) / 2

def std_dev (s: List ℝ) : ℝ :=
  let m := avg s
  sqrt ((s.map (λ x, (x - m)^2)).sum / s.length)

theorem data_set_is_1_1_3_3 {x1 x2 x3 x4 : ℝ} :
  (is_positive_integer x1) ∧ (is_positive_integer x2) ∧ (is_positive_integer x3) ∧ (is_positive_integer x4) →
  avg [x1, x2, x3, x4] = 2 →
  median [x1, x2, x3, x4] = 2 →
  std_dev [x1, x2, x3, x4] = 1 →
  multiset.of_list [x1, x2, x3, x4] = multiset.of_list [1, 1, 3, 3] :=
by
  sorry

end data_set_is_1_1_3_3_l643_643429


namespace claudia_weekend_earnings_l643_643825

/-- Claudia charges $10.00 for her one-hour class. 20 kids attend the Saturday’s class.
    Half that many attend the Sunday’s class. Prove Claudia makes $300.00 over the weekend. -/
theorem claudia_weekend_earnings : 
  let charge_per_kid := 10
  let saturday_attendees := 20
  let sunday_attendees := saturday_attendees / 2
  let total_attendees := saturday_attendees + sunday_attendees
  let total_earnings := charge_per_kid * total_attendees
  total_earnings = 300 := 
by 
  simp
  sorry

end claudia_weekend_earnings_l643_643825


namespace evaluate_i_powers_l643_643476

-- Define the cyclic pattern of powers of i (imaginary unit).
def cyclic_powers (n : ℕ) : ℂ :=
  match n % 4 with
  | 0 => 1
  | 1 => complex.I
  | 2 => -1
  | 3 => -complex.I
  | _ => 0  -- This should never happen because 'n % 4' is in {0, 1, 2, 3}

theorem evaluate_i_powers : cyclic_powers 25 + cyclic_powers 125 = 2 * complex.I :=
  by {
    sorry
  }

end evaluate_i_powers_l643_643476


namespace math_problem_proof_l643_643155

-- Definitions and Conditions
def E := (1, 0)
def K := (-1, 0)

-- Question 1: Trajectory of the moving point P
noncomputable def trajectory_eqn : Prop :=
  ∀ P : ℝ × ℝ, (let (x, y) := P in dist (1 - x, -y) * dist (x + 1, y) = 2 * dist (2, 0) * (x + 1)) →
    y^2 = 4 * x

-- Question 2: Equation of the circumcircle of triangle ABD
noncomputable def circumcircle_eqn : Prop :=
  ∀ A B D : ℝ × ℝ,
    let l := λ y : ℝ, 2 * y - 1 in
    let E_A := (fst A - 1, snd A) in
    let E_B := (fst B - 1, snd B) in
    let symm_A := (fst A, -snd A) in
    collinear3 K A B ∧
    inner_product E_A E_B = -8 →
    circle_eq (9, 0) A B symm_A = ((x - 9)^2 + y^2 = 40)

-- Formal statement of the problem
theorem math_problem_proof : trajectory_eqn ∧ circumcircle_eqn :=
by
  split
  -- Prove trajectory_eqn
  · sorry
  -- Prove circumcircle_eqn
  · sorry

end math_problem_proof_l643_643155


namespace count_valid_n_l643_643466

-- Define the conditions
def condition1 (n : ℕ) : Prop := (130 * n)^50 > n^100
def condition2 (n : ℕ) : Prop := n^100 > 5^250

-- Define the problem statement
theorem count_valid_n : (finset.filter (λ n, condition1 n ∧ condition2 n) (finset.range 131)).card = 74 :=
by
  -- The steps and actual proof would go here
  sorry

end count_valid_n_l643_643466


namespace range_of_n_l643_643226

theorem range_of_n (x : ℕ) (n : ℝ) : 
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 5 → x - 2 < n + 3) → ∃ n, 0 < n ∧ n ≤ 1 :=
by
  sorry

end range_of_n_l643_643226


namespace transformed_sin_function_l643_643672

theorem transformed_sin_function :
  (∀ x : ℝ, sin 2 x ) = (λ x : ℝ, sin (2 * (x - π / 12))) → (λ x : ℝ, sin (x - π / 6)) :=
by
  sorry

end transformed_sin_function_l643_643672


namespace percentage_increase_is_10_l643_643052

-- Define the original price and the increased percentage
def original_price : ℝ := 200
def increased_percentage : ℝ := 0.10

-- Calculate the new price based on the original price and increased percentage
def new_price : ℝ := original_price + (original_price * increased_percentage)

-- Calculate the increase in price
def increase_in_price : ℝ := new_price - original_price

-- Calculate the percentage increase
def percentage_increase : ℝ := (increase_in_price / original_price) * 100

-- Theorem statement: percentage increase is 10%
theorem percentage_increase_is_10 : percentage_increase = 10 := 
by sorry

end percentage_increase_is_10_l643_643052


namespace prism_volume_l643_643665

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : x * z = 32) 
  (h3 : y * z = 48) : 
  x * y * z = 192 :=
by
  sorry

end prism_volume_l643_643665


namespace relationship_among_abc_l643_643998

noncomputable def a := Real.tan (3 * Real.pi / 4)
noncomputable def b := Real.cos (2 * Real.pi / 5)
def c := (1 + Real.sin (6 * Real.pi / 5)) ^ 0

theorem relationship_among_abc : c > b ∧ b > a := by
  sorry

end relationship_among_abc_l643_643998


namespace consecutive_integers_l643_643259

theorem consecutive_integers (a b c : ℝ)
  (h1 : ∃ k : ℤ, a + b = k ∧ b + c = k + 1 ∧ c + a = k + 2)
  (h2 : ∃ k : ℤ, b + c = 2 * k + 1) :
  ∃ n : ℤ, a = n + 2 ∧ b = n + 1 ∧ c = n := 
sorry

end consecutive_integers_l643_643259


namespace min_elements_sum_set_l643_643889

theorem min_elements_sum_set (n : ℕ) (h1 : n ≥ 2) (a : Fin (n + 1) → ℕ) 
  (h2 : a 0 = 0) (h3 : a n = 2 * n - 1) (h4 : ∀ i j, i < j → a i < a j) :
  Nat.card (Finset.image (λ i : Fin (n + 1) × Fin (n + 1), a i.1 + a i.2) (Finset.univ.product Finset.univ)) = 3 * n := 
sorry

end min_elements_sum_set_l643_643889


namespace find_y_at_neg3_l643_643507

noncomputable def quadratic_solution (y x a b : ℝ) : Prop :=
  y = x ^ 2 + a * x + b

theorem find_y_at_neg3
    (a b : ℝ)
    (h1 : 1 + a + b = 2)
    (h2 : 4 - 2 * a + b = -1)
    : quadratic_solution 2 (-3) a b :=
by
  sorry

end find_y_at_neg3_l643_643507


namespace no_solution_for_equation_l643_643655

theorem no_solution_for_equation (x : ℝ) (hx : x ≠ -1) :
  (5 * x + 2) / (x^2 + x) ≠ 3 / (x + 1) := 
sorry

end no_solution_for_equation_l643_643655


namespace harrison_grade_levels_l643_643930

theorem harrison_grade_levels
  (total_students : ℕ)
  (percent_moving : ℚ)
  (advanced_class_size : ℕ)
  (num_normal_classes : ℕ)
  (normal_class_size : ℕ)
  (students_moving : ℕ)
  (students_per_grade_level : ℕ)
  (grade_levels : ℕ) :
  total_students = 1590 →
  percent_moving = 40 / 100 →
  advanced_class_size = 20 →
  num_normal_classes = 6 →
  normal_class_size = 32 →
  students_moving = total_students * percent_moving →
  students_per_grade_level = advanced_class_size + num_normal_classes * normal_class_size →
  grade_levels = students_moving / students_per_grade_level →
  grade_levels = 3 :=
by
  intros
  sorry

end harrison_grade_levels_l643_643930


namespace max_profit_achieved_at_three_l643_643786

/-- The yield function of a peach tree given the cost of fertilizer. -/
def yield (x : ℝ) : ℝ := 4 - (3 / (x + 1))

/-- The profit function given x (cost of fertilizer in hundreds of yuan). -/
def L (x : ℝ) : ℝ := 64 - (48 / (x + 1)) - 3 * x

/-- The domain constraint for the cost of fertilizer. -/
def in_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 5

/-- The profit maximization theorem for the peach tree yield and cost. -/
theorem max_profit_achieved_at_three : 
  ∃ x : ℝ, in_domain x ∧ L x = 43 ∧ (∀ y : ℝ, in_domain y → L y ≤ 43) :=
sorry

end max_profit_achieved_at_three_l643_643786


namespace area_of_LOM_is_3_l643_643966

-- Definition of a scalene triangle with given angle properties and area
structure ScaleneTriangle := 
  (A B C : Point)
  (angle_A angle_B angle_C : ℝ)
  (area : ℝ)
  (is_scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (angles_sum : angle_A + angle_B + angle_C = 180)
  (angle_relation_1 : angle_A = angle_B - angle_C)
  (angle_relation_2 : angle_B = 2 * angle_C)
  (area_value : area = 2)

-- Definition of the triangle formed by angle bisectors intersecting the circumcircle
structure BisectorIntersectionTriangle := 
  (L O M : Point)
  (original_triangle : ScaleneTriangle)
  (angle_bisector_intersection : 
    -- Assume function bisect_angle which defines that L, O, M are points on the circumcircle intersecting the angle bisectors
    bisect_angle A = L ∧ bisect_angle B = O ∧ bisect_angle C = M)

-- Problem Statement: Prove that the area of triangle LOM is approximately 3, rounded to the nearest whole number
theorem area_of_LOM_is_3 (T : ScaleneTriangle) (BIT : BisectorIntersectionTriangle)
  (H : BIT.original_triangle = T) : 
  abs (area (triangle BIT.L BIT.O BIT.M) - 3) < 1 :=
begin
  -- Definitions and conditions logic implementation would go here
  sorry
end

end area_of_LOM_is_3_l643_643966


namespace find_m_l643_643131

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 2 → (m - 1) * x < real.sqrt (4 * x - x^2)) ↔ m = 2 := by
  sorry

end find_m_l643_643131


namespace number_of_nonnegative_solutions_l643_643571

theorem number_of_nonnegative_solutions : ∃ (count : ℕ), count = 1 ∧ ∀ x : ℝ, x^2 + 9 * x = 0 → x ≥ 0 → x = 0 := by
  sorry

end number_of_nonnegative_solutions_l643_643571


namespace inscribed_square_side_length_l643_643306

variables (PQ QR PR EG : ℝ)

def right_triangle := PQ = 5 ∧ QR = 12 ∧ PR = 13 ∧ EG = (12 / 5)

theorem inscribed_square_side_length (H : right_triangle PQ QR PR EG) :
  let s := EG in s = 12 / 5 := 
by
  sorry

end inscribed_square_side_length_l643_643306


namespace no_integer_solution_xyz_l643_643753

theorem no_integer_solution_xyz : ¬ ∃ (x y z : ℤ),
  x^6 + x^3 + x^3 * y + y = 147^157 ∧
  x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by
  sorry

end no_integer_solution_xyz_l643_643753


namespace lemonade_percentage_l643_643791

theorem lemonade_percentage (L : ℝ) : 
  (0.4 * (1 - L / 100) + 0.6 * 0.55 = 0.65) → L = 20 :=
by
  sorry

end lemonade_percentage_l643_643791


namespace chord_line_eq_l643_643540

open Real

def ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

def bisecting_point (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2

theorem chord_line_eq :
  (∃ (k : ℝ), ∀ (x y : ℝ), ellipse x y → bisecting_point ((x + y) / 2) ((x + y) / 2) → y - 2 = k * (x - 4)) →
  (∃ (x y : ℝ), ellipse x y ∧ x + 2 * y - 8 = 0) :=
by
  sorry

end chord_line_eq_l643_643540


namespace max_band_members_l643_643671

theorem max_band_members (k n m : ℕ) : m = k^2 + 11 → m = n * (n + 9) → m ≤ 112 :=
by
  sorry

end max_band_members_l643_643671


namespace find_f_of_3_l643_643884

-- Define the function f and its properties
variable {f : ℝ → ℝ}

-- Define the properties given in the problem
axiom f_mono_increasing : ∀ x y : ℝ, x ≤ y → f x ≤ f y
axiom f_of_f_minus_exp : ∀ x : ℝ, f (f x - 2^x) = 3

-- The main theorem to prove
theorem find_f_of_3 : f 3 = 9 := 
sorry

end find_f_of_3_l643_643884


namespace distance_MN_l643_643603

-- defining problem conditions
def line_l_inclination := real.pi / 4

def C1_parametric (α : ℝ) : ℝ × ℝ :=
( (real.sqrt 3 / 3) * real.cos α, real.sin α )

def C2_parametric (α : ℝ) : ℝ × ℝ :=
( 3 + real.sqrt 13 * real.cos α, 2 + real.sqrt 13 * real.sin α )

-- stating the theorem
theorem distance_MN : (∀ M N,
( line_l_inclination = real.pi / 4 ) ∧ 
( C1_parametric α = M ) ∧ 
( C2_parametric α = N ) ∧ 
( M.1 > 0 ∧ M.2 > 0 ) ∧
( N.1 > 0 ∧ N.2 > 0 )
→ ( real.dist M N = (9 * real.sqrt 2) / 2 ) ) :=
by sorry

end distance_MN_l643_643603


namespace product_of_primes_l643_643883

def a (n : ℕ) : ℕ := 4^(2 * n - 1) + 3^(n - 2) 

def smallest_prime_dividing_infinitely_many_terms (a : ℕ → ℕ) : ℕ := -- implementation details
sorry

def smallest_prime_dividing_every_term (a : ℕ → ℕ) : ℕ := -- implementation details
sorry

theorem product_of_primes :
  let p := smallest_prime_dividing_infinitely_many_terms a,
      q := smallest_prime_dividing_every_term a in
  p * q = 65 :=
by
  sorry

end product_of_primes_l643_643883


namespace total_biking_distance_l643_643492

theorem total_biking_distance (onur_daily : ℕ) (days_per_week : ℕ) (additional_distance : ℕ)
  (onur_distance_per_day : onur_daily = 250)
  (days_per_week_value : days_per_week = 5)
  (hanil_additional_distance : additional_distance = 40) :
  (onur_daily * days_per_week + (onur_daily + additional_distance) * days_per_week) = 2700 := by
  -- Unfolding the given definitions
  unfold onur_daily at onur_distance_per_day,
  unfold days_per_week at days_per_week_value,
  unfold additional_distance at hanil_additional_distance,
  -- Applying the given equalities
  rw [onur_distance_per_day, days_per_week_value, hanil_additional_distance],
  -- Simplifying the arithmetic
  calc
    (250 * 5 + (250 + 40) * 5)
      = (250 * 5 + 290 * 5) : by rw add_assoc
  ... = 1250 + 1450 : by rw [mul_add, mul_add, mul_comm 250 5, mul_comm 290 5]
  ... = 2700 : by norm_num

end total_biking_distance_l643_643492


namespace total_wheels_in_parking_lot_l643_643593

theorem total_wheels_in_parking_lot :
  (∀ (cars motorcycles : ℕ), cars = 19 → motorcycles = 11 → 
  ((5 * cars) + (2 * motorcycles) = 117)) :=
by { intros cars motorcycles hCars hMotorcycles,
     sorry }

end total_wheels_in_parking_lot_l643_643593


namespace circle_equation_l643_643897

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem circle_equation (M N : ℝ × ℝ)
  (hM : M = (-2, 0))  (hN : N = (2, 0)) : 
  let center := midpoint M N in
  let radius := distance M N / 2 in
  (center = (0, 0)) ∧ (radius = 2) ∧ (x - center.1)^2 + (y - center.2)^2 = radius^2 := 
by 
  have h_mid : midpoint M N = (0, 0), sorry
  have h_dist : distance M N = 4, sorry
  have h_rad : radius = 2, sorry
  have h_eq : (x - 0)^2 + (y - 0)^2 = 2^2, sorry
  exact ⟨h_mid, h_rad, h_eq⟩

end circle_equation_l643_643897


namespace sin_2x_value_l643_643216

theorem sin_2x_value (x : ℝ) (hx : Real.sin x + Real.cos x + Real.tan x + Real.cot x + Real.sec x + Real.csc x = 9) :
  Real.sin (2 * x) = 40 - 20 * Real.sqrt 3 :=
by
  sorry

end sin_2x_value_l643_643216


namespace minimum_green_chips_l643_643762

variable (y v g : ℕ)

-- Define the conditions
axiom condition1 : v ≥ 2 * y / 3
axiom condition2 : v ≤ g / 4
axiom condition3 : y + v ≥ 75

-- Define the target statement
theorem minimum_green_chips : ∃ (g : ℕ), (condition1 y v) ∧ (condition2 v g) ∧ (condition3 y v) ∧ (∀ (g' : ℕ), ((condition1 y v) ∧ (condition2 v g') ∧ (condition3 y v)) → g ≤ g') → g = 120 := sorry

end minimum_green_chips_l643_643762


namespace train_length_l643_643794

theorem train_length (speed : ℝ) (tunnel_length : ℝ) (time_min : ℝ) (length_of_train : ℝ) 
  (h1 : speed = 75) 
  (h2 : tunnel_length = 3.5) 
  (h3 : time_min = 3) 
  (h4 : length_of_train = 3.75 - tunnel_length) : 
  length_of_train = 0.25 :=
by
  -- Given conditions
  have h1 : speed = 75 := by assumption
  have h2 : tunnel_length = 3.5 := by assumption
  have h3 : time_min = 3 := by assumption
  -- Calculated condition
  have h4 : length_of_train = 3.75 - tunnel_length := by assumption
  -- Show that length_of_train is 0.25
  have : length_of_train = 0.25 := by 
    rw [h2, h4]
    norm_num
  exact this

end train_length_l643_643794


namespace estimate_pi_correct_l643_643646

noncomputable def estimate_pi (m : ℝ) : ℝ :=
  (4 * ((m / 200) + 0.5))

theorem estimate_pi_correct :
  estimate_pi 56 = 78 / 25 :=
by
  unfold estimate_pi
  norm_num
  rw [← div_eq_mul_one_div]
  norm_num
  rw [eq_div_iff]
  norm_num
  sorry

end estimate_pi_correct_l643_643646


namespace characterize_functions_l643_643107

noncomputable def specific_function {f : ℝ → ℝ} : Prop :=
∀ x y : ℝ, x ≠ y → f'((x + y) / 2) = (f y - f x) / (y - x)

theorem characterize_functions (f : ℝ → ℝ) :
  specific_function f ↔ ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c := 
sorry

end characterize_functions_l643_643107


namespace solve_problem_l643_643104

noncomputable def problem_statement : Prop :=
  ∫ x in 0..1, (sqrt (1 - (x - 1)^2) - 2*x) = (real.pi / 4 - 1)

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l643_643104


namespace claudia_weekend_earnings_l643_643821

theorem claudia_weekend_earnings :
  (let charge := 10 in 
   let sat_kids := 20 in 
   let sun_kids := sat_kids / 2 in
   (sat_kids * charge) + (sun_kids * charge) = 300) :=
by
  sorry

end claudia_weekend_earnings_l643_643821


namespace pages_fell_out_l643_643788

theorem pages_fell_out (first_page last_page : ℕ) (h₁ : first_page = 143) (h₂ : last_page = 314) : 
  (last_page - first_page + 1) = 172 :=
by
  rw [h₁, h₂]
  simp
  rfl
  sorry

end pages_fell_out_l643_643788


namespace exists_positive_x_for_inequality_l643_643876

-- Define the problem conditions and the final proof goal.
theorem exists_positive_x_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Ico (-9/4 : ℝ) (2 : ℝ) :=
by
  sorry

end exists_positive_x_for_inequality_l643_643876


namespace eccentricity_of_hyperbola_l643_643552

noncomputable def eccentricity_problem (a b : ℝ) (h_ax : a > 0) (h_bx : b > 0) : ℝ :=
  let P := (1 : ℝ, real.sqrt 1)
  let F := (-1 : ℝ, 0)
  let tangent_slope_P := 1 / (2 * real.sqrt 1)
  let tangent := (real.sqrt 1) / (1 + 1)
  have h_intersect : sorry := sorry
  have h_tangent_focus : tangent_slope_P = tangent, from sorry
  let a := (real.sqrt 5 - 1) / 2
  let e := (real.sqrt 5 + 1) / 2
  have h_hyperbola: (1 / a^2) - (1 / b^2) = 1, from sorry 
  have h_focus: (a^2 + b^2) = 1, from sorry 
  show ℝ, from e

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a_gt : a > 0) (h_b_gt : b > 0) (x0 : ℝ) (hx0_eq : x0 = 1) (hx0_sqrt : real.sqrt x0 = 1) :
  let e := eccentricity_problem a b h_a_gt h_b_gt
  e = (real.sqrt 5 + 1) / 2 :=
sorry

end eccentricity_of_hyperbola_l643_643552


namespace total_surface_area_288π_l643_643043

noncomputable def radius_of_sphere (diameter : ℝ) : ℝ :=
  diameter / 2

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ :=
  4 * π * r^2

noncomputable def lateral_surface_area_of_cylinder (r h : ℝ) : ℝ :=
  2 * π * r * h

noncomputable def total_surface_area_decorative_ball (d : ℝ) : ℝ :=
  let r := radius_of_sphere d
  let h := d
  surface_area_of_sphere r + lateral_surface_area_of_cylinder r h

theorem total_surface_area_288π :
  total_surface_area_decorative_ball 12 = 288 * π :=
by
  sorry

end total_surface_area_288π_l643_643043


namespace ashley_cocktail_calories_l643_643074

theorem ashley_cocktail_calories:
  let mango_grams := 150
  let honey_grams := 200
  let water_grams := 300
  let vodka_grams := 100

  let mango_cal_per_100g := 60
  let honey_cal_per_100g := 640
  let vodka_cal_per_100g := 70
  let water_cal_per_100g := 0

  let total_cocktail_grams := mango_grams + honey_grams + water_grams + vodka_grams
  let total_cocktail_calories := (mango_grams * mango_cal_per_100g / 100) +
                                 (honey_grams * honey_cal_per_100g / 100) +
                                 (vodka_grams * vodka_cal_per_100g / 100) +
                                 (water_grams * water_cal_per_100g / 100)
  let caloric_density := total_cocktail_calories / total_cocktail_grams
  let result := 300 * caloric_density
  result = 576 := by
  sorry

end ashley_cocktail_calories_l643_643074


namespace problem_statement_l643_643119

theorem problem_statement : 
  (finset.filter (λ n, ∃ x : ℝ, floor x + floor (2 * x) + floor (3 * x) + floor (4 * x) = n) 
  (finset.range 1001)).card = 400 :=
by
  sorry

end problem_statement_l643_643119


namespace tangent_slope_coordinates_l643_643352

-- Define the given function
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of the given function
def f' (x : ℝ) : ℝ := 3 * x^2

-- The main statement where we prove the required condition
theorem tangent_slope_coordinates (P : ℝ × ℝ) : (f' P.1 = 3 → P = (1, f 1) ∨ P = (-1, f (-1))) :=
begin
  sorry
end

end tangent_slope_coordinates_l643_643352


namespace log_base_10_of_2_correct_l643_643539

-- Definitions of given data
def ten_pow4 : ℤ := 10^4
def ten_pow5 : ℤ := 10^5 
def two_pow12 : ℤ := 2^12
def two_pow15 : ℤ := 2^15  

theorem log_base_10_of_2_correct (a b c d : ℤ) :
    a = 10000 → b = 100000 → c = 4096 → d = 32768 →
    12 * real.log10(2) < real.log10(10^4) → 
    15 * real.log10(2) < real.log10(10^5) → 
    real.log10 2 = 0.3010 :=
by
  -- Given data values
  assume ha hb hc hd h1 h2
  -- Proof here
  -- Proof is omitted
  sorry

end log_base_10_of_2_correct_l643_643539


namespace problem_statement_l643_643423

-- Define initial conditions
variables {d a1 : ℝ} (a : ℕ → ℝ)
def arithmetic_sequence : Prop := ∀ n : ℕ, a n = a1 + (n-1)*d
def geometric_sequence (k : ℕ → ℕ) : Prop := ∀ (n : ℕ), ∃ m : ℕ, a (k m) = a1 * 4^(m-1)

-- Specific values given in the problem
def k1 := 1
def k2 := 2
def k3 := 6
def an (n : ℕ) := a1 + (n-1)*d

-- Prove that a_86 is also in the geometric sequence {a_k_n}
theorem problem_statement (h_arith_seq : arithmetic_sequence a) (h_geom_seq : geometric_sequence (λ n, n)) (hd : d ≠ 0) :
  ∃ m : ℕ, an 86 = a1 * 4^(m-1) :=
sorry

end problem_statement_l643_643423


namespace football_lineup_l643_643635

theorem football_lineup (team_size : ℕ)
  (strong_offensive : ℕ)
  (quarterback running_back wide_receiver tight_end : ℕ)
  (H1 : team_size = 12)
  (H2 : strong_offensive = 4) 
  (H3 : quarterback + running_back + wide_receiver + tight_end = team_size - 1):
  (strong_offensive) * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) = 31680 := 
by calc
  (strong_offensive) * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)
      = 4 * 11 * 10 * 9 * 8 : by { rw [H1, H2], norm_num }
  ... = 31680 : by norm_num

end football_lineup_l643_643635


namespace intersection_of_A_and_B_solve_inequality_l643_643022

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | x^2 - 16 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 ≥ 0}

-- Proof problem 1: Find A ∩ B
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} :=
sorry

-- Proof problem 2: Solve the inequality with respect to x
theorem solve_inequality (a : ℝ) :
  if a = 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = ∅
  else if a > 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | 1 < x ∧ x < a}
  else
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | a < x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_solve_inequality_l643_643022


namespace power_of_5_in_8_factorial_l643_643948

theorem power_of_5_in_8_factorial :
  let x := Nat.factorial 8
  ∃ (i k m p : ℕ), 0 < i ∧ 0 < k ∧ 0 < m ∧ 0 < p ∧ x = 2^i * 3^k * 5^m * 7^p ∧ m = 1 :=
by
  sorry

end power_of_5_in_8_factorial_l643_643948


namespace possible_integer_roots_of_polynomial_l643_643088

noncomputable def polynomial : ℤ[X] := X^4 + 4 * X^3 + (a_2 : ℤ) * X^2 + (a_1 : ℤ) * X - 60

theorem possible_integer_roots_of_polynomial : 
  (∀ (x : ℤ), polynomial.eval x polynomial = 0 → x ∈ {± 1, ± 2, ± 3, ± 4, ± 5, ± 6, ± 10, ± 12, ± 15, ± 20, ± 30, ± 60}) :=
by
  sorry

end possible_integer_roots_of_polynomial_l643_643088


namespace find_triples_l643_643108

theorem find_triples (p : ℤ) (m : ℤ) (z : ℤ) : 
  p.prime ∧ 0 < m ∧ z < 0 ∧ (p^3 + p * m + 2 * z * m = m^2 + p * z + z^2) ↔ 
  (p = 2 ∧ ∃ (z : ℤ), z ∈ {-1, -2, -3} ∧ m = 4 + z) :=
by
  sorry

end find_triples_l643_643108


namespace concession_stand_total_revenue_l643_643395

theorem concession_stand_total_revenue :
  let hot_dog_price : ℝ := 1.50
  let soda_price : ℝ := 0.50
  let total_items_sold : ℕ := 87
  let hot_dogs_sold : ℕ := 35
  let sodas_sold := total_items_sold - hot_dogs_sold
  let revenue_from_hot_dogs := hot_dogs_sold * hot_dog_price
  let revenue_from_sodas := sodas_sold * soda_price
  revenue_from_hot_dogs + revenue_from_sodas = 78.50 :=
by {
  -- Proof will go here
  sorry
}

end concession_stand_total_revenue_l643_643395


namespace tan_theta_imaginary_l643_643173

theorem tan_theta_imaginary
  (θ : ℝ)
  (h1 : ∀ (z : ℂ), z = (sin θ - (3/5)) + complex.I * (cos θ - (4/5)) → (z.re = 0))
  (h2 : sin θ = 3/5)
  (h3 : cos θ = -(4/5)) :
  tan θ = -(3/4) :=
by
  sorry

end tan_theta_imaginary_l643_643173


namespace max_value_of_ab_bc_cd_l643_643272

noncomputable def max_ab_bc_cd (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (h_sum : a + b + c + d = 200) : ℝ :=
  ab + bc + cd

theorem max_value_of_ab_bc_cd : ∃ a b c d : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ (a + b + c + d = 200) ∧ max_ab_bc_cd a b c d = 10000 :=
begin
  use [100, 100, 0, 0],
  split; try { split },
  all_goals { linarith },
end

end max_value_of_ab_bc_cd_l643_643272


namespace alpha_beta_purchase_ways_l643_643769

-- Definitions for the problem
def number_of_flavors : ℕ := 7
def number_of_milk_types : ℕ := 4
def total_products_to_purchase : ℕ := 5

-- Conditions
def alpha_max_per_flavor : ℕ := 2
def beta_only_cookies (x : ℕ) : Prop := x = number_of_flavors

-- Main theorem (statement only)
theorem alpha_beta_purchase_ways : 
  ∃ (ways : ℕ), 
    ways = 17922 ∧
    ∀ (alpha beta : ℕ), 
      alpha + beta = total_products_to_purchase →
      (alpha <= alpha_max_per_flavor * number_of_flavors ∧ beta <= total_products_to_purchase - alpha) :=
sorry

end alpha_beta_purchase_ways_l643_643769


namespace no_such_n_exists_l643_643204

theorem no_such_n_exists : ¬ ∃ n : ℕ, n < 100000 ∧ (∃ k : ℕ, n = k^6) ∧ (∃ p : ℕ, nat.prime p ∧ ∃ e : ℕ, n = p^e ∧ e % 2 = 1) :=
sorry

end no_such_n_exists_l643_643204


namespace chord_length_eq_l643_643030

noncomputable def length_of_chord (radius : ℝ) (distance_to_chord : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - distance_to_chord^2)

theorem chord_length_eq {radius distance_to_chord : ℝ} (h_radius : radius = 5) (h_distance : distance_to_chord = 4) :
  length_of_chord radius distance_to_chord = 6 :=
by
  sorry

end chord_length_eq_l643_643030


namespace drink_all_cups_l643_643445

-- Define initial conditions and the proposition
theorem drink_all_cups (M D : ℕ) (hM : M < 30) (hD : D < 30) (hDiff : M ≠ D) : 
  ∃ (rotations : ℕ → ℕ × ℕ), (∀ n, (rotations n).1 < 30 ∧ (rotations n).2 < 30) ∧ 
  (∀ m n, m ≠ n → (rotations m).1 ≠ (rotations n).1 ∧ (rotations m).2 ≠ (rotations n).2) ∧ 
  (∀ n, rotations (n + 1) = ((rotations n).1 + 2) % 30, (rotations n).2 + 2) % 30) := 
sorry

end drink_all_cups_l643_643445


namespace monotone_decreasing_interval_l643_643681

noncomputable def f (x : ℝ) := log 2 (x^2 - 1)

theorem monotone_decreasing_interval : 
  ∀ x y : ℝ, x ∈ (-∞, -1) ∧ y ∈ (-∞, -1) ∧ x < y → f(y) < f(x) :=
by
  intro x y,
  assume h : x ∈ (-∞, -1) ∧ y ∈ (-∞, -1) ∧ x < y,
  sorry

end monotone_decreasing_interval_l643_643681


namespace incorrect_statements_l643_643734

theorem incorrect_statements
  (P1 P2 : ℝ × ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h₁ : P1 = (x1, y1))
  (h₂ : P2 = (x2, y2))
  (h3 : P1 ≠ P2) :
  ¬ (∀ θ, (0 ≤ θ ∧ θ < π/2 → ∀ θ' (θ < θ' → θ' < π/2 → (tan θ < tan θ'))) ∨
         (∀ α β, tan α = tan β → α = β) ∨
         (∀ m m', (m = m' → (m ≠ m'))) ∨
         (∀ x y, ((y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)))) :=
sorry

end incorrect_statements_l643_643734


namespace cannot_form_right_triangle_l643_643440

theorem cannot_form_right_triangle (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 2) (h₃ : c = 3) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h₁, h₂, h₃]
  -- Next step would be to simplify and show the inequality, but we skip the proof
  -- 2^2 + 2^2 = 4 + 4 = 8 
  -- 3^2 = 9 
  -- 8 ≠ 9
  sorry

end cannot_form_right_triangle_l643_643440


namespace susan_missed_pay_l643_643659

noncomputable def daily_pay : ℕ := 15 * 8

noncomputable def total_work_days (weeks: ℕ) : ℕ := weeks * 5

noncomputable def unpaid_vacation_days (total_work_days paid_vacation_days: ℕ) : ℕ := 
  total_work_days - paid_vacation_days

noncomputable def missed_pay (unpaid_days daily_pay: ℕ) : ℕ := 
  unpaid_days * daily_pay

theorem susan_missed_pay :
  let weeks := 2,
      paid_vacation_days := 6 in
  missed_pay (unpaid_vacation_days (total_work_days weeks) paid_vacation_days) daily_pay = 480 :=
by
  sorry

end susan_missed_pay_l643_643659


namespace exists_x0_range_of_m_for_condition_l643_643180

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin ((Real.pi / 4) + x))^2 + (Real.sqrt 3) * (Real.cos (2 * x)) - 1

-- Problem (1):
theorem exists_x0 (x0 : ℝ) (h0 : x0 ∈ set.Ioo 0 (Real.pi / 3)) (h1 : f x0 = 1) : x0 = Real.pi / 4 :=
sorry

-- Problem (2):
theorem range_of_m_for_condition (m : ℝ)
  (h2 : ∀ x : ℝ, x ∈ set.Icc (Real.pi / 6) (5 * Real.pi / 6) → -3 < f x - m ∧ f x - m < Real.sqrt 3) :
  0 < m ∧ m < 1 :=
sorry

end exists_x0_range_of_m_for_condition_l643_643180


namespace bus_dispatch_interval_l643_643008

/--
Xiao Hua walks at a constant speed along the route of the "Chunlei Cup" bus.
He encounters a "Chunlei Cup" bus every 6 minutes head-on and is overtaken by a "Chunlei Cup" bus every 12 minutes.
Assume "Chunlei Cup" buses are dispatched at regular intervals, travel at a constant speed, and do not stop at any stations along the way.
Prove that the time interval between bus departures is 8 minutes.
-/
theorem bus_dispatch_interval
  (encounters_opposite_direction: ℕ)
  (overtakes_same_direction: ℕ)
  (constant_speed: Prop)
  (regular_intervals: Prop)
  (no_stops: Prop)
  (h1: encounters_opposite_direction = 6)
  (h2: overtakes_same_direction = 12)
  (h3: constant_speed)
  (h4: regular_intervals)
  (h5: no_stops) :
  True := 
sorry

end bus_dispatch_interval_l643_643008


namespace trains_clear_time_l643_643369

theorem trains_clear_time :
  ∀ (length_A length_B length_C : ℕ)
    (speed_A_kmph speed_B_kmph speed_C_kmph : ℕ)
    (distance_AB distance_BC : ℕ),
  length_A = 160 ∧ length_B = 320 ∧ length_C = 480 ∧
  speed_A_kmph = 42 ∧ speed_B_kmph = 30 ∧ speed_C_kmph = 48 ∧
  distance_AB = 200 ∧ distance_BC = 300 →
  ∃ (time_clear : ℚ), time_clear = 50.78 :=
by
  intros length_A length_B length_C
         speed_A_kmph speed_B_kmph speed_C_kmph
         distance_AB distance_BC h
  sorry

end trains_clear_time_l643_643369


namespace correct_bike_lock_code_l643_643764

def rotate_digit (d : ℕ) : ℕ :=
  (d + 5) % 10

theorem correct_bike_lock_code
  (initial_code : list ℕ) 
  (h : initial_code = [6, 3, 4, 8]) :
  map rotate_digit initial_code = [1, 8, 9, 3] :=
by
  sorry

end correct_bike_lock_code_l643_643764


namespace interval_contains_root_l643_643339

def f (x : ℝ) : ℝ := 2^x + x - 2

theorem interval_contains_root : ∃ c ∈ set.Ioo 0 1, f c = 0 :=
by
  have h1 : f 0 < 0,
  { simp [f],
    linarith },
  
  have h2 : f 1 > 0,
  { simp [f],
    linarith },
  
  apply exists_between,
  exact ⟨h1, h2⟩,
sorry

end interval_contains_root_l643_643339


namespace cube_root_floor_prod_l643_643083

theorem cube_root_floor_prod :
  (∏ n in (finset.range 2032).filter (λ n, odd n + 1), ⌊(n : ℝ)^(1/3)⌋ : ℝ) /
  (∏ n in (finset.range 2033).filter even, ⌊(n : ℝ)^(1/3) ⌋) = (1/4 : ℝ) := 
by sorry

end cube_root_floor_prod_l643_643083


namespace hexagonal_prism_surface_area_l643_643697

theorem hexagonal_prism_surface_area (h : ℝ) (a : ℝ) (H_h : h = 6) (H_a : a = 4) : 
  let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  let lateral_area := 6 * a * h
  let total_area := lateral_area + base_area
  total_area = 48 * (3 + Real.sqrt 3) :=
by
  -- let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  -- let lateral_area := 6 * a * h
  -- let total_area := lateral_area + base_area
  -- total_area = 48 * (3 + Real.sqrt 3)
  sorry

end hexagonal_prism_surface_area_l643_643697


namespace pos_divisors_180_l643_643573

theorem pos_divisors_180 : 
  (∃ a b c : ℕ, 180 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 1) →
  (∃ n : ℕ, n = 18 ∧ n = (a + 1) * (b + 1) * (c + 1)) := by
  sorry

end pos_divisors_180_l643_643573


namespace symmetric_point_y_axis_l643_643235

theorem symmetric_point_y_axis (A B : ℝ × ℝ) (hA : A = (2, -8)) (hSym : B.1 = -A.1 ∧ B.2 = A.2) : B = (-2, -8) := 
by 
  rw [hA] at hSym
  rw [Prod.mk.eta] at hSym
  have : B = (-2, -8) := by sorry
  exact this

end symmetric_point_y_axis_l643_643235


namespace mary_sheep_remaining_l643_643282

theorem mary_sheep_remaining : 
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let sheep_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := sheep_after_sister / 2
  let remaining_sheep := sheep_after_sister - sheep_given_to_brother
  remaining_sheep = 150 :=
by
  assume initial_sheep := 400
  have sheep_given_to_sister := initial_sheep / 4
  have sheep_after_sister := initial_sheep - sheep_given_to_sister
  have sheep_given_to_brother := sheep_after_sister / 2
  have remaining_sheep := sheep_after_sister - sheep_given_to_brother
  show remaining_sheep = 150
  sorry

end mary_sheep_remaining_l643_643282


namespace num_even_divisors_factorial_7_l643_643202

theorem num_even_divisors_factorial_7 : 
  let fact_7 := 7! in 
  let prime_factorization_7_fact := 2^4 * 3^2 * 5 * 7 in
  (fact_7 = prime_factorization_7_fact) → 
  ∃ ev_divs, (ev_divs = 48) ∧ 
    -- Conditions on the exponents for forming even divisors
    (∀ r, (r ∣ fact_7) → (∃ a b c d, (1 ≤ a ∧ a ≤ 4) ∧ (0 ≤ b ∧ b ≤ 2) ∧ 
                         (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1) ∧ 
                         r = 2^a * 3^b * 5^c * 7^d)) :=
begin
  sorry
end

end num_even_divisors_factorial_7_l643_643202


namespace animal_survival_months_l643_643589

theorem animal_survival_months : 
  ∃ (n : ℕ), 
  let p := 1 - (1 / 10 : ℝ), 
      initial_population := 300,
      expected_survival := 218.7 in
  abs ((initial_population : ℝ) * p^n - expected_survival) < 1 → 
  n = 3 :=
sorry

end animal_survival_months_l643_643589


namespace smallest_p_q_good_l643_643832

def is_p_good (p n : ℕ) : Prop :=
  ∑ k in Finset.range (n + 1), (-1) ^ (Nat.factorial.orderOf p k) < 0

def exists_p_q_good (n : ℕ) : Prop :=
  ∃ (p q : ℕ), p ≠ q ∧ Prime p ∧ Prime q ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ is_p_good p n ∧ is_p_good q n

theorem smallest_p_q_good :
  ∃ n : ℕ, exists_p_q_good n ∧ ∀ m : ℕ, m < n → ¬ exists_p_q_good m :=
  ∃ (n : ℕ), exists_p_q_good n ∧ (∀ m : ℕ, m < n → ¬ exists_p_q_good m) ∧ n = 229 := 
by
  sorry

end smallest_p_q_good_l643_643832


namespace TH_passes_through_midpoint_l643_643664

variables {A B C H Q P K T : Type}
variables [triangle ABC A H Q P K T]

-- Condition 1: The altitudes of an acute triangle $ABC$ intersect at $H$.
def altitudes_of_acute_triangle_intersect_at 
  (H : Type) (ABC : triangle) : Prop := 
  ∃ H : Point, is_altitude H ABC ∧ acute_triangle ABC

-- Condition 2: The tangent line at $H$ to the circumcircle of triangle $BHC$ intersects the lines $AB$ and $AC$ at points $Q$ and $P$.
def tangent_line_at_H_intersects_lines_AB_AC 
  (H : Type) (Q P : Point) (ABC : triangle) : Prop :=
  ∃ Q P : Point, is_tangent_line H Q P ABC

-- Condition 3: The circumcircles of triangles $ABC$ and $APQ$ intersect at point $K$ ($K \neq A$).
def circles_of_ABC_and_APQ_intersect_at_K 
  (K : Type) (ABC APQ : triangle) : Prop := 
  ∃ K : Point, is_circumcircle K ABC APQ ∧ K ≠ A

-- Condition 4: The tangent lines at points $A$ and $K$ to the circumcircle of triangle $APQ$ intersect at $T$.
def tangent_lines_at_A_and_K_intersect_at_T
  (A K T : Type) (APQ : triangle) : Prop :=
  ∃ T : Point, is_intersection_of_tangents A K T APQ

-- Conclusion: Prove that $TH$ passes through the midpoint of segment $BC$.
theorem TH_passes_through_midpoint 
  (H Q P K T : Point) (ABC APQ : triangle) :
  (altitudes_of_acute_triangle_intersect_at H ABC) →
  (tangent_line_at_H_intersects_lines_AB_AC H Q P ABC) →
  (circles_of_ABC_and_APQ_intersect_at_K K ABC APQ) →
  (tangent_lines_at_A_and_K_intersect_at_T A K T APQ) →
  passes_through_midpoint TH BC :=
sorry

end TH_passes_through_midpoint_l643_643664


namespace students_sampled_in_camps_l643_643408

-- Definitions to represent the camps and students
structure Camp :=
  (start_num : ℕ)
  (end_num : ℕ)

-- Define the three camps
def camp1 : Camp := {start_num := 1, end_num := 200}
def camp2 : Camp := {start_num := 201, end_num := 355}
def camp3 : Camp := {start_num := 356, end_num := 500}

-- Given conditions
def total_students : ℕ := 500
def sample_size : ℕ := 50
def random_selected_num : ℕ := 3

-- Define the systematic sampling formula
def systematic_sample (n : ℕ) : set ℕ := {l | ∃ k, l = 10 * k + n}

-- Main theorem to prove
theorem students_sampled_in_camps : 
  ∃ (students_camp1 students_camp2 students_camp3 : ℕ), 
    students_camp1 = 20 ∧ 
    students_camp2 = 16 ∧ 
    students_camp3 = 14 ∧ 
    set.to_finset (systematic_sample random_selected_num) ∩ finset.range 201 = finset.range (students_camp1 + 1) ∧
    set.to_finset (systematic_sample random_selected_num) ∩ finset.Ico 201 356 = finset.range (students_camp2 + 1) ∧
    set.to_finset (systematic_sample random_selected_num) ∩ finset.Ico 356 501 = finset.range (students_camp3 + 1) :=
sorry

end students_sampled_in_camps_l643_643408


namespace positive_difference_largest_smallest_l643_643645

def Varsity : ℕ := 780
def Northwest : ℕ := 1200
def Central : ℕ := 2280
def Greenbriar : ℕ := 1800
def SouthBeach : ℕ := 1620

theorem positive_difference_largest_smallest :
  abs (Central - Varsity) = 1500 := 
by 
  sorry

end positive_difference_largest_smallest_l643_643645


namespace coeff_x24_in_expansion_l643_643109

theorem coeff_x24_in_expansion :
  -- Let the polynomial be as defined
  let p := (λ (n : ℕ), (x ^ n - n)) 
  -- Define the product of these polynomials from 1 to 8
  let poly := p 1 * p 2 * p 3 * p 4 * p 5 * p 6 * p 7 * p 8 
  -- Coefficient of x^24 in the product expansion is 68
  coeff poly 24 = 68 := sorry

end coeff_x24_in_expansion_l643_643109


namespace bus_people_count_l643_643365

-- Define the initial number of people on the bus
def initial_people_on_bus : ℕ := 34

-- Define the number of people who got off the bus
def people_got_off : ℕ := 11

-- Define the number of people who got on the bus
def people_got_on : ℕ := 24

-- Define the final number of people on the bus
def final_people_on_bus : ℕ := (initial_people_on_bus - people_got_off) + people_got_on

-- Theorem: The final number of people on the bus is 47.
theorem bus_people_count : final_people_on_bus = 47 := by
  sorry

end bus_people_count_l643_643365


namespace find_side_length_of_left_square_l643_643239

noncomputable def left_square_side_length {x : ℝ} : Prop := 
  let middle_square := x + 17
  let right_square := x + 11
  (x + middle_square + right_square = 52) → (x = 8)

-- This is only the statement of the theorem, without the proof
theorem find_side_length_of_left_square {x : ℝ} :
  left_square_side_length :=
sorry

end find_side_length_of_left_square_l643_643239


namespace test_schemes_count_l643_643785

-- Defining the anti-inflammatory and antipyretic drugs
inductive AntiInflammatory
| X1 | X2 | X3 | X4 | X5

inductive Antipyretic
| T1 | T2 | T3 | T4

open AntiInflammatory Antipyretic

-- Define conditions
def is_valid (xi : Finset AntiInflammatory) (ti : Antipyretic) : Prop :=
  (xi = {X1, X2} ∧ (ti ∈ {T1, T2, T3, T4})) ∨
  (X3 ∉ xi ∨ (ti ∉ {T4})) ∨ 
  (∀ (x ∈ xi), x = X4 ∨ x = X5)

-- Define the function to count valid schemes
def count_valid_schemes : ℕ :=
  ([(X1, X2)]).sum (λ pair,
    [T1, T2, T3, T4].count (is_valid {pair.1, pair.2}))
  +
  [(X3, X4), (X3, X5)].sum (λ pair,
    [T1, T2, T3].count (is_valid {pair.1, pair.2})) 
  +
  [(X4, X5)].sum (λ pair,
    [T1, T2, T3, T4].count (is_valid {pair.1, pair.2}))

theorem test_schemes_count : count_valid_schemes = 14 := by
  sorry

end test_schemes_count_l643_643785


namespace increasing_intervals_side_a_length_l643_643916

section
variables (x A k : ℝ) (ABC : Type) [EuclideanGeometry ABC]

variables (a b c : ℝ) (angleA angleB angleC : ℝ)
variables (AD BD : ℝ) (D : ℝ)

-- Condition 1: Function definition
def f (x : ℝ) : ℝ := 2 * (sin x) * (sin (x + π / 3))

-- Condition 2: Acute triangle ABC, sides a, b, c
variables (triangle_acute : 0 < angleA ∧ angleA < π / 2)

-- Condition 3: The angle bisector of A intersects BC at D
variables (angle_bisector_AD : true)  -- Abstract condition

-- Condition 4: The line x = A is a symmetry axis of the graph of f(x)
variables (symmetry_axis : ∀ x, f (2*A - x) = f x)

-- Condition 5: given AD = √2 BD = 2
variables (AD_spec : AD = 2) (BD_spec : BD = sqrt 2)

-- Proof of intervals (I)
theorem increasing_intervals : ∀ k : ℤ, [k*π - π/6, k*π + π/3] = ℝ
 := sorry

-- Condition 6: 0 < A < π / 2
variables (A_bound : 0 < A ∧ A < π / 2)

-- Proof of side length (II)
theorem side_a_length (h_AC : a = sqrt 6) : a = sqrt 6
 := sorry
end

end increasing_intervals_side_a_length_l643_643916


namespace tablet_value_is_2100_compensation_for_m_days_l643_643081

-- Define the given conditions
def monthly_compensation: ℕ := 30
def monthly_tablet_value (x: ℕ) (cash: ℕ): ℕ := x + cash

def daily_compensation (days: ℕ) (x: ℕ) (cash: ℕ): ℕ :=
  days * (x / monthly_compensation + cash / monthly_compensation)

def received_compensation (tablet_value: ℕ) (cash: ℕ): ℕ :=
  tablet_value + cash

-- The proofs we need:
-- Proof that the tablet value is 2100 yuan
theorem tablet_value_is_2100:
  ∀ (x: ℕ) (cash_1 cash_2: ℕ), 
  ((20 * (x / monthly_compensation + 1500 / monthly_compensation)) = (x + 300)) → 
  x = 2100 := sorry

-- Proof that compensation for m days is 120m yuan
theorem compensation_for_m_days (m: ℕ):
  ∀ (x: ℕ), 
  ((x + 1500) / monthly_compensation) = 120 → 
  x = 2100 → 
  m * 120 = 120 * m := sorry

end tablet_value_is_2100_compensation_for_m_days_l643_643081


namespace ball_returns_velocity_required_initial_velocity_to_stop_l643_643410

-- Define the conditions.
def distance_A_to_wall : ℝ := 5
def distance_wall_to_B : ℝ := 2
def distance_AB : ℝ := 9
def initial_velocity_v0 : ℝ := 5
def acceleration_a : ℝ := -0.4

-- Hypothesize that the velocity when the ball returns to A is 3 m/s.
theorem ball_returns_velocity (t : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  initial_velocity_v0 * t + (1 / 2) * acceleration_a * t^2 = distance_AB + distance_A_to_wall →
  initial_velocity_v0 + acceleration_a * t = 3 := sorry

-- Hypothesize that to stop exactly at A, the initial speed should be 4 m/s.
theorem required_initial_velocity_to_stop (t' : ℝ) :
  distance_A_to_wall + distance_wall_to_B + distance_AB = 2 * (distance_A_to_wall + distance_wall_to_B) ∧
  (0.4 * t') * t' + (1 / 2) * acceleration_a * t'^2 = distance_AB + distance_A_to_wall →
  0.4 * t' = 4 := sorry

end ball_returns_velocity_required_initial_velocity_to_stop_l643_643410


namespace newton_discovery_l643_643394

theorem newton_discovery :
  (∃ x y, (x = "a" ∧ y = "the")
  ∧ (("While he was investigating ways to improve the telescope, Newton made " ++ x ++ " discover which completely changed " ++ y ++ "man’s understanding of colour")
     = "While he was investigating ways to improve the telescope, Newton made a discover which completely changed the man’s understanding of colour")) :=
begin
  use ["a", "the"],
  split,
  { split, refl, refl },
  { refl }
end

end newton_discovery_l643_643394


namespace initial_men_count_l643_643364

-- Define initial conditions and parameters
def initial_men (M : ℝ) := (∃ food_duration : ℝ, food_duration = 22 ∧ M * 22 = food_duration * M)
def additional_men := 134.11764705882354
def remaining_days := 17
def days_elapsed := 2
def total_food (M : ℝ) := M * 22

-- Proof Problem Statement
theorem initial_men_count : ∀ (M : ℝ), initial_men M →
  (total_food M - M * days_elapsed = (M + additional_men) * remaining_days) →
  M = 760 :=
by { intros, sorry }

end initial_men_count_l643_643364


namespace converse_proof_contrapositive_proof_l643_643163

variable {f : ℝ → ℝ} 
variable {a b : ℝ}

-- Assuming f is an increasing function
axiom increasing_f (x y : ℝ) : x ≤ y → f(x) ≤ f(y)

-- Original Proposition
def original_proposition : Prop := a + b ≥ 0 → f(a) + f(b) ≥ f(-a) + f(-b)

-- Converse Proposition
def converse_proposition : Prop := f(a) + f(b) ≥ f(-a) + f(-b) → a + b ≥ 0

-- Contrapositive Proposition
def contrapositive_proposition : Prop := f(a) + f(b) < f(-a) + f(-b) → a + b < 0

-- Proof of the truth of converse proposition
theorem converse_proof : converse_proposition := by
  sorry

-- Proof of the truth of contrapositive proposition
theorem contrapositive_proof : contrapositive_proposition := by
  sorry

end converse_proof_contrapositive_proof_l643_643163


namespace q_is_false_l643_643222

theorem q_is_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end q_is_false_l643_643222


namespace triangle_area_is_24_l643_643745

def area_of_triangle (a b C : ℝ) : ℝ :=
  1/2 * a * b * Real.sin (C * Real.pi / 180)

theorem triangle_area_is_24 : area_of_triangle 8 12 150 = 24 := by
  sorry

end triangle_area_is_24_l643_643745


namespace N_even_l643_643768

theorem N_even (N : ℕ) (h : ∀ (n : ℕ), n < 2 * N → ∃ (i j : fin (2 * N)), i ≠ j ∧ (i.val + j.val) % 2 = 0) : N % 2 = 0 :=
sorry

end N_even_l643_643768


namespace line_inclination_angle_l643_643541

theorem line_inclination_angle (x y : ℝ) :
  (y = sqrt 3 * x + 2) → (tan (π / 3) = sqrt 3) := by
  intros h
  sorry

end line_inclination_angle_l643_643541


namespace count_words_with_A_at_least_once_l643_643568

theorem count_words_with_A_at_least_once :
  let letters := {'A', 'B', 'C', 'D', 'E'}
  let total_words := 5^3
  let words_without_A := 4^3
  total_words - words_without_A = 61 := 
by
  let letters := {'A', 'B', 'C', 'D', 'E'}
  let total_words := Int.ofNat (5^3)
  let words_without_A := Int.ofNat (4^3)
  show total_words - words_without_A = 61
  sorry

end count_words_with_A_at_least_once_l643_643568


namespace find_a_l643_643917

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 2

theorem find_a (h : ∀ x1 ∈ Set.Icc (1:ℝ) (Real.exp 1), ∃ x2 ∈ Set.Icc (1:ℝ) (Real.exp 1), f a x1 + f a x2 = 4) :
  a = Real.exp 1 + 1 :=
begin
  sorry
end

end find_a_l643_643917


namespace imaginary_part_of_z_l643_643902

theorem imaginary_part_of_z (z : ℂ) (h1 : ∃ x y : ℝ, z = x + y * complex.I) 
  (h2 : z.conj + complex.abs z * complex.I = 1 + 2 * complex.I) : 
  z.im = -3 / 4 :=
by
  sorry

end imaginary_part_of_z_l643_643902


namespace largest_triangle_area_l643_643783

theorem largest_triangle_area {a : ℝ} 
  (h1 : ∃ (a : ℝ), 0 < a)
  (h2 : 3 + 4 * a)
  (h3 : 3 + 8 * a = 4 + real.sqrt 13) :
  3 ≤ 4 + real.sqrt 13 :=
by
  sorry

end largest_triangle_area_l643_643783


namespace four_digit_integer_unique_l643_643041

theorem four_digit_integer_unique (a b c d : ℕ) (h1 : a + b + c + d = 16) (h2 : b + c = 10) (h3 : a - d = 2)
    (h4 : (a - b + c - d) % 11 = 0) : a = 4 ∧ b = 6 ∧ c = 4 ∧ d = 2 := 
  by 
    sorry

end four_digit_integer_unique_l643_643041


namespace cost_milk_is_5_l643_643760

-- Define the total cost the baker paid
def total_cost : ℕ := 80

-- Define the cost components
def cost_flour : ℕ := 3 * 3
def cost_eggs : ℕ := 3 * 10
def cost_baking_soda : ℕ := 2 * 3

-- Define the number of liters of milk
def liters_milk : ℕ := 7

-- Define the unknown cost per liter of milk
noncomputable def cost_per_liter_milk (c : ℕ) : Prop :=
  c * liters_milk = total_cost - (cost_flour + cost_eggs + cost_baking_soda)

-- State the theorem we want to prove
theorem cost_milk_is_5 : cost_per_liter_milk 5 := 
by
  sorry

end cost_milk_is_5_l643_643760


namespace final_tomatoes_count_l643_643361

theorem final_tomatoes_count (initial_tomatoes birds_one_eat_fraction birds_two_eat_fraction plant_growth_rate : ℝ) 
  (h_initial : initial_tomatoes = 21)
  (h_birds_one : birds_one_eat_fraction = 1/3)
  (h_birds_two : birds_two_eat_fraction = 1/2)
  (h_growth : plant_growth_rate = 0.5) :
  let remaining_after_first_birds := initial_tomatoes * (1 - birds_one_eat_fraction),
      remaining_after_second_birds := remaining_after_first_birds * (1 - birds_two_eat_fraction),
      final_tomatoes := remaining_after_second_birds * (1 + plant_growth_rate) in
  final_tomatoes = 11 := 
by
  sorry

end final_tomatoes_count_l643_643361


namespace min_people_like_Mozart_and_Bach_not_Beethoven_l643_643376

/-- Two hundred people were surveyed. --/
constant total_people : ℕ := 200

/-- 160 people indicated they liked Mozart. --/
constant people_like_Mozart : ℕ := 160

/-- 120 people indicated they liked Bach. --/
constant people_like_Bach : ℕ := 120

/-- 90 people indicated they liked Beethoven. --/
constant people_like_Beethoven : ℕ := 90

theorem min_people_like_Mozart_and_Bach_not_Beethoven :
  ∃ min_people, min_people = 10 ∧ 
    (min_people = total_people - people_like_Beethoven - 
     (people_like_Mozart - people_like_Beethoven) - 
     (people_like_Bach - people_like_Beethoven)) :=
by
  sorry

end min_people_like_Mozart_and_Bach_not_Beethoven_l643_643376


namespace max_knights_on_chessboard_l643_643833

theorem max_knights_on_chessboard (m n : ℕ) (hm : m = 2013) (hn : n = 2013) :
  ∃ k : ℕ, k = ⌈(m * n) / 2⌉ ∧
    ∀ knights : ℕ, (knights ≤ k) → 
      (∀ (i j : ℕ) (i' j' : ℕ), 
        knight_move (i, j) (i', j') → 
        (i, j) ≠ (i', j') ) :=
  sorry

def knight_move (p1 p2 : ℕ × ℕ) : Prop :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  (abs (x1 - x2) = 2 ∧ abs (y1 - y2) = 1) ∨ 
  (abs (x1 - x2) = 1 ∧ abs (y1 - y2) = 2)

end max_knights_on_chessboard_l643_643833


namespace min_k_value_l643_643683

variable (p q r s k : ℕ)

/-- Prove the smallest value of k for which p, q, r, and s are positive integers and 
    satisfy the given equations is 77
-/
theorem min_k_value (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (eq1 : p + 2 * q + 3 * r + 4 * s = k)
  (eq2 : 4 * p = 3 * q)
  (eq3 : 4 * p = 2 * r)
  (eq4 : 4 * p = s) : k = 77 :=
sorry

end min_k_value_l643_643683


namespace find_circle_equation_l643_643112

noncomputable def circle_centered_on_line (A B : ℝ × ℝ) (L : ℝ × ℝ → Prop) (C_eqn : ℝ × ℝ → Prop) :=
  let center := (1, 1) in
  L center ∧ C_eqn A ∧ C_eqn B

theorem find_circle_equation (A B : ℝ × ℝ) (L : ℝ × ℝ → Prop) (C_eqn : ℝ × ℝ → Prop) :
  circle_centered_on_line (1, -1) (-1, 1) (λ p, p.1 + p.2 - 2 = 0) (λ p, (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 4) :=
by
  -- proof goes here
  sorry

end find_circle_equation_l643_643112


namespace sum_of_first_1234_terms_l643_643192

def sequence (n : ℕ) : ℕ :=
  if n % 2 = 1 then 1 else
    let k := n / 2
    in if k % (k + 1) = 0 then 2 else sorry

theorem sum_of_first_1234_terms :
  (Finset.range 1234).sum sequence = 2419 :=
sorry

end sum_of_first_1234_terms_l643_643192


namespace breakfast_time_correct_l643_643105

noncomputable def breakfast_time_calc (x : ℚ) : ℚ :=
  (7 * 60) + (300 / 13)

noncomputable def coffee_time_calc (y : ℚ) : ℚ :=
  (7 * 60) + (420 / 11)

noncomputable def total_breakfast_time : ℚ :=
  coffee_time_calc ((420 : ℚ) / 11) - breakfast_time_calc ((300 : ℚ) / 13)

theorem breakfast_time_correct :
  total_breakfast_time = 15 + (6 / 60) :=
by
  sorry

end breakfast_time_correct_l643_643105


namespace find_central_angle_l643_643165

variable (L : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions
def arc_length_condition : Prop := L = 200
def radius_condition : Prop := r = 2
def arc_length_formula : Prop := L = r * α

-- Theorem statement
theorem find_central_angle 
  (hL : arc_length_condition L) 
  (hr : radius_condition r) 
  (hf : arc_length_formula L r α) : 
  α = 100 := by
  -- Proof goes here
  sorry

end find_central_angle_l643_643165


namespace lineup_choices_count_l643_643633

def football_team := {member : Type} -- 12 members
def strong_members := {strong_member : Type} -- 4 strong enough members
def positions := {quarterback : Type, running_back : Type, offensive_lineman : strong_members, wide_receiver : Type, tight_end : Type }

theorem lineup_choices_count (strong_count : ℕ) (total_count : ℕ) : 
  strong_count = 4 → total_count = 12 → 
  ∃ choices : ℕ, choices = 4 * 11 * 10 * 9 * 8 ∧ choices = 31680 :=
by
  intros h1 h2,
  use (4 * 11 * 10 * 9 * 8),
  split,
  { refl },
  { norm_num }

end lineup_choices_count_l643_643633


namespace tangents_equal_intersection_l643_643293

open Classical

universe u

noncomputable def circles_tangent (X : Point) (C1 C2 : Circle) :=
∃ (A B : Point), tangent X C1 A ∧ tangent X C2 B ∧ tangent_len X A = tangent_len X B

theorem tangents_equal_intersection
  (X : Point) (ω1 ω2 : Circle) (A B C D : Point) :
  circles_tangent X ω1 ω2 →
  tangent X ω1 A →
  tangent X ω1 D →
  tangent X ω2 C →
  tangent X ω2 B →
  ∃ P, intersection_of_diagonals ACBD = P ∧ common_internal_tangents_intersection ω1 ω2 = P := 
by
  sorry

end tangents_equal_intersection_l643_643293


namespace exists_irreducible_poly_2_complex_zeros_l643_643400

theorem exists_irreducible_poly_2_complex_zeros (n : ℕ) :
  ∃ P : Polynomial ℚ, P.degree = 2 * n + 1 ∧ 
  (∃ z₁ z₂ : ℂ, z₁ ≠ z₂ ∧ P.eval z₁ = 0 ∧ P.eval z₂ = 0) ∧
  Irreducible P :=
sorry

end exists_irreducible_poly_2_complex_zeros_l643_643400


namespace radius_of_third_circle_l643_643411

theorem radius_of_third_circle 
  {P Q : ℝ → EuclideanSpace 2 ℝ} 
  (r1 r2 r3 : ℝ) 
  (hP : r1 = 2) 
  (hQ : r2 = 5) 
  (h_tangent : is_tangent_externally P r1 Q r2)
  (h_third_tangent : is_tangent_third_circle P r1 Q r2 r3) 
  (dPQ : distance P Q = 3) : 
  r3 = 5 / 18 := 
sorry

end radius_of_third_circle_l643_643411


namespace raft_capacity_l643_643705

theorem raft_capacity (total_without_life_jackets : ℕ) (reduction_with_life_jackets : ℕ)
  (people_needing_life_jackets : ℕ) (total_capacity_with_life_jackets : ℕ)
  (no_life_jackets_capacity : total_without_life_jackets = 21)
  (life_jackets_reduction : reduction_with_life_jackets = 7)
  (life_jackets_needed : people_needing_life_jackets = 8) :
  total_capacity_with_life_jackets = 14 :=
by
  -- Proof should be here
  sorry

end raft_capacity_l643_643705


namespace arithmetic_sequence_sum_l643_643893

theorem arithmetic_sequence_sum {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 6) 
  (h2 : a 2 + a 14 = 26) :
  (10 / 2) * (a 1 + a 10) = 80 :=
by sorry

end arithmetic_sequence_sum_l643_643893


namespace total_ninja_stars_l643_643475

variable (e c j : ℕ)
variable (H1 : e = 4) -- Eric has 4 ninja throwing stars
variable (H2 : c = 2 * e) -- Chad has twice as many ninja throwing stars as Eric
variable (H3 : j = c - 2) -- Chad sells 2 ninja stars to Jeff
variable (H4 : j = 6) -- Jeff now has 6 ninja throwing stars

theorem total_ninja_stars :
  e + (c - 2) + 6 = 16 :=
by
  sorry

end total_ninja_stars_l643_643475


namespace driving_time_equation_l643_643069

theorem driving_time_equation :
  ∀ (t : ℝ), (60 * t + 90 * (3.5 - t) = 300) :=
by
  intro t
  sorry

end driving_time_equation_l643_643069


namespace print_time_l643_643051

theorem print_time (pages_per_minute : ℕ) (total_pages : ℕ) (rounded_time : ℕ) (H1 : pages_per_minute = 24) (H2 : total_pages = 300) (H3 : rounded_time = 13) : 
  rounded_time = Int.toNat (Real.toRound (total_pages / pages_per_minute : ℝ)) := 
by
  sorry

end print_time_l643_643051


namespace tan_theta_plus_pi_over_four_l643_643160

variable (θ : ℝ)
variable (h₁ : θ ∈ (3 * Real.pi / 2, 2 * Real.pi))
variable (h₂ : Real.cos (θ - Real.pi / 4) = 3 / 5)

theorem tan_theta_plus_pi_over_four : 
  Real.tan (θ + Real.pi / 4) = 3 / 4 :=
by
  sorry

end tan_theta_plus_pi_over_four_l643_643160


namespace f_is_periodic_l643_643266

noncomputable def f : ℝ → ℝ := sorry

axiom f_abs_le_one : ∀ x : ℝ, |f x| ≤ 1

axiom functional_equation : ∀ x : ℝ, f(x + 13 / 42) + f x = f(x + 1 / 6) + f(x + 1 / 7)

theorem f_is_periodic : ∃ c ≠ 0, ∀ x : ℝ, f (x + c) = f x :=
begin
  use 1,
  have periodicity : ∀ x : ℝ, f (x + 1) = f x,
  { -- proof omitted as per instructions
    sorry },
  exact ⟨1, by norm_num, periodicity⟩
end

end f_is_periodic_l643_643266


namespace probability_sum_even_l643_643415

-- Definitions of probabilities for even integers generation by the computers
def p_ai_even : ℝ := 1 / 2
def p_bi_even : ℝ := 1 / 3

/-- Given the probabilities that each computer generates even integers, prove that
the probability that the sum a_1b_1 + a_2b_2 + ... + a_kb_k is even is
p_k = 1/6 * (1/3)^(k-1) + 1/2 -/
theorem probability_sum_even (k : ℕ) : 
  let p_ai_even := 1 / 2 in
  let p_bi_even := 1 / 3 in
  let p_k : ℕ → ℝ := λ k, 1 / 6 * (1 / 3) ^ (k - 1) + 1 / 2 in
  -- the desired probability holds
  True := sorry

end probability_sum_even_l643_643415


namespace mila_total_coin_value_l643_643627

theorem mila_total_coin_value :
  let gold_value := 2.5
  let silver_value := 3
  let total_gold_value := 20 * gold_value
  let total_silver_value := 15 * silver_value
  total_gold_value + total_silver_value = 95 :=
by
  let gold_value := 10 / 4
  let silver_value := 15 / 5
  have h1 : total_gold_value = 20 * gold_value := rfl
  have h2 : total_silver_value = 15 * silver_value := rfl
  have h3 : gold_value = 2.5 := by norm_num
  have h4 : silver_value = 3 := by norm_num
  rw [h3, h4]
  norm_num
  sorry

end mila_total_coin_value_l643_643627


namespace max_discount_l643_643430

theorem max_discount (C : ℝ) (x : ℝ) (h1 : 1.8 * C = 360) (h2 : ∀ y, y ≥ 1.3 * C → 360 - x ≥ y) : x ≤ 100 :=
by
  have hC : C = 360 / 1.8 := by sorry
  have hMinPrice : 1.3 * C = 1.3 * (360 / 1.8) := by sorry
  have hDiscount : 360 - x ≥ 1.3 * (360 / 1.8) := by sorry
  sorry

end max_discount_l643_643430


namespace oldest_child_age_correct_l643_643989

-- Defining the conditions
def jane_start_age := 16
def jane_current_age := 32
def jane_stopped_babysitting_years_ago := 10
def half (x : ℕ) := x / 2

-- Expressing the conditions
def jane_last_babysitting_age := jane_current_age - jane_stopped_babysitting_years_ago
def max_child_age_when_jane_stopped := half jane_last_babysitting_age
def years_since_jane_stopped := jane_stopped_babysitting_years_ago

def calculate_oldest_child_current_age (age : ℕ) : ℕ :=
  age + years_since_jane_stopped

def child_age_when_stopped := max_child_age_when_jane_stopped
def expected_oldest_child_current_age := 21

-- The theorem stating the equivalence
theorem oldest_child_age_correct : 
  calculate_oldest_child_current_age child_age_when_stopped = expected_oldest_child_current_age :=
by
  -- Proof here
  sorry

end oldest_child_age_correct_l643_643989


namespace probability_of_triangle_l643_643377

/-- 
Prove that the probability that three segments formed by two randomly 
chosen points on a unit line segment can form a triangle is 1/4.
-/
theorem probability_of_triangle (x y : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 1) (h₂ : 0 ≤ y) (h₃ : y ≤ 1) :
  (∀ x y, ((x ≤ y ∧ x < 0.5 ∧ y < x + 0.5 ∧ y > 0.5) ∨ 
          (y ≤ x ∧ y < 0.5 ∧ x < y + 0.5 ∧ x > 0.5))) ↔ (1/4 : ℝ) :=
sorry

end probability_of_triangle_l643_643377


namespace impossible_result_l643_643497

noncomputable def f (a b : ℝ) (c : ℤ) (x : ℝ) : ℝ :=
  a * Real.sin x + b * x + c

theorem impossible_result (a b : ℝ) (c : ℤ) :
  ¬(f a b c 1 = 1 ∧ f a b c (-1) = 2) :=
by {
  sorry
}

end impossible_result_l643_643497


namespace modulus_solution_l643_643499

noncomputable def complex_modulus (x : ℝ) : ℝ :=
  real.sqrt (6^2 + x^2)

theorem modulus_solution (x : ℝ) (h : complex_modulus x = 15 * real.sqrt 2) :
  x = real.sqrt 414 :=
by
  sorry

end modulus_solution_l643_643499


namespace max_value_of_product_l643_643899

theorem max_value_of_product (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1):
  x * y ≤ 1 / 8 :=
begin
  sorry
end

end max_value_of_product_l643_643899


namespace bob_expected_rolls_in_leap_year_l643_643077

/-- Bob rolls a fair six-sided die each morning. If he rolls an even number greater than 2, 
    he eats sweetened cereal. If he rolls an odd number greater than 2, he eats unsweetened cereal.
    If he rolls a 1 or a 2, then he rolls again. Prove that the expected number of times Bob 
    will roll his die in a leap year of 366 days is 549. -/
theorem bob_expected_rolls_in_leap_year :
  let E := 3 / 2 in
  let days_in_leap_year := 366 in
  E * days_in_leap_year = 549 :=
by
  let E := 3 / 2
  let days_in_leap_year := 366
  show E * days_in_leap_year = 549
  calc 
    E * days_in_leap_year = (3 / 2) * 366    : by rfl
    ...                  = 549                : by norm_num

end bob_expected_rolls_in_leap_year_l643_643077


namespace connie_total_markers_l643_643086

theorem connie_total_markers : 2315 + 1028 = 3343 :=
by
  sorry

end connie_total_markers_l643_643086


namespace TN_eq_DT_l643_643890

noncomputable def rhombus_side_length {A B C D : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [Finite A] [Finite B] [Finite C] [Finite D] (AB : A) (BD : B) (CD : C) (DA : D) : ℝ := sorry

noncomputable def angle_BAD {A B : Type} [DecidableEq A] [DecidableEq B] [Finite A] [Finite B] (angle: A → B → ℝ) : Prop := angle AB B = 60

noncomputable def AT_equals_BN {A B : Type} [DecidableEq A] [DecidableEq B] [Finite A] [Finite B] (AT BN: A → ℝ) : Prop := AT AB = BN B

noncomputable def points_on_sides {A B C D : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [Finite A] [Finite B] [Finite C] [Finite D] (T N: A → D) (AB BC: B) : Prop := (T AB) = (N BC)

theorem TN_eq_DT {A B C D : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [Finite A] [Finite B] [Finite C] [Finite D]
  (AB : A) (BD : B) (CD : C) (DA : D)
  (angle : A → B → ℝ) (AT BN: A → ℝ) (T N: A → D) 
  (AB_rhombus : Prop) (AB_angle : angle BAD) (eq_sides: AT_equals_BN)
  (points_position : points_on_sides T N AB BC) : 
  T N = D T := 
sorry

end TN_eq_DT_l643_643890


namespace part1_part2_part3_l643_643896

noncomputable theory

open Real

-- Conditions and definitions
def f (x : ℝ) : ℝ := sin x / x
def g (x : ℝ) (a : ℝ) : ℝ := a * cos x

-- Prove that for \( \forall x \in (0, \frac{\pi}{2}) \), \( f(x) < 1 \).
theorem part1 (x : ℝ) (h1 : x ∈ Ioo 0 (π / 2)) : f x < 1 := 
sorry

-- Prove that for \( \forall x \in (-\frac{\pi}{2}, 0) \cup (0, \frac{\pi}{2}) \), \( f(x) > g(x) \) and find the range of \( a \).
theorem part2 (x : ℝ) (a : ℝ) 
    (h1 : x ∈ (Ioo (-π / 2) 0) ∪ Ioo 0 (π / 2)) 
    (h2 : a ≤ 1) : f x > g x a :=
sorry

-- Prove that for \( \forall x \in (-\frac{\pi}{2}, 0) \cup (0, \frac{\pi}{2}) \), \( [f(x)]^2 > g(x) \) and find the range of \( a \).
theorem part3 (x : ℝ) (a : ℝ)
    (h1 : x ∈ (Ioo (-π / 2) 0) ∪ Ioo 0 (π / 2))
    (h2 : a ≤ 1) : (f x)^2 > g x a :=
sorry

end part1_part2_part3_l643_643896


namespace constant_sequence_a_eq_neg2_l643_643275

theorem constant_sequence_a_eq_neg2
  (a : ℤ)
  (h1 : ∀ n : ℕ, ∀ m : ℕ, n > 0 → m > 0 → a = (a^2 - 2) / (a + 1))
  (h2 : a_n 1 = a)
  (h3 : ∀ n : ℕ, a_n (n + 1) = (a_n n)^2 - 2 / (a_n n + 1))
  (h4 : ∀ n : ℕ, a_n n = a) :
  a = -2 :=
sorry

end constant_sequence_a_eq_neg2_l643_643275


namespace budget_research_and_development_l643_643412

-- We will define what we know from the problem as constants or definitions.
def salaries : ℕ := 60
def utilities : ℕ := 5
def equipment : ℕ := 4
def supplies : ℕ := 2
def degrees_transportation : ℕ := 72
def total_degrees : ℕ := 360
def total_budget_percentage : ℕ := 100

-- We need to prove that the percentage for research and development is 9%.
theorem budget_research_and_development :
  let transportation_percentage := (degrees_transportation * total_budget_percentage) / total_degrees in
  let known_percentages := salaries + utilities + equipment + supplies + transportation_percentage in
  total_budget_percentage - known_percentages = 9 :=
by
  -- Here we ignore the proof details by inserting sorry.
  sorry

end budget_research_and_development_l643_643412


namespace area_of_reflected_arcs_l643_643784

-- Define a regular hexagon with side length 2
def side_length := 2

-- Define the conditions under which we are working
def radius := 2 / (2 * sin (Real.pi * 35 / 180))
def subtended_angle_degrees := 70

-- Define the correct answer we need to prove
def correct_answer := 12 * Real.sqrt 3 - 3.258 * Real.pi

-- State the theorem we need to prove
theorem area_of_reflected_arcs :
  (6 * (0.543 * Real.pi * (radius ∧ 2) - Real.sqrt 3 * side_length^2)) = correct_answer := by
    sorry

end area_of_reflected_arcs_l643_643784


namespace triangle_inequality_l643_643244

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 (-a + b + c) + b^2 (a - b + c) + c^2 (a + b - c) ≤ 3 * a * b * c := 
by
  sorry

end triangle_inequality_l643_643244


namespace fido_reach_fraction_simplified_l643_643850

noncomputable def fidoReach (s r : ℝ) : ℝ :=
  let octagonArea := 2 * (1 + Real.sqrt 2) * s^2
  let circleArea := Real.pi * (s / Real.sqrt (2 + Real.sqrt 2))^2
  circleArea / octagonArea

theorem fido_reach_fraction_simplified (s : ℝ) :
  (∃ a b : ℕ, fidoReach s (s / Real.sqrt (2 + Real.sqrt 2)) = (Real.sqrt a / b) * Real.pi ∧ a * b = 16) :=
  sorry

end fido_reach_fraction_simplified_l643_643850


namespace both_selected_prob_l643_643744

-- Given conditions
def prob_Ram := 6 / 7
def prob_Ravi := 1 / 5

-- The mathematically equivalent proof problem statement
theorem both_selected_prob : (prob_Ram * prob_Ravi) = 6 / 35 := by
  sorry

end both_selected_prob_l643_643744


namespace sum_of_distances_necessary_but_not_sufficient_l643_643535

-- Definitions and conditions
variables {P A B : Type*} [metric_space P]

def foci_distance_sum_constant (P A B : P) (a : ℝ) (h : a > 0) : Prop :=
  dist P A + dist P B = 2 * a

def trajectory_is_ellipse (P A B : P) : Prop :=
  ∃ (el : set P), ∀ (p ∈ el), dist p A + dist p B = dist P A + dist P B

-- Statement 
theorem sum_of_distances_necessary_but_not_sufficient
  (P A B : P) (a : ℝ) (h : a > 0) :
  foci_distance_sum_constant P A B a h → ¬trajectory_is_ellipse P A B :=
sorry

end sum_of_distances_necessary_but_not_sufficient_l643_643535


namespace cos_theta_value_l643_643905

open Real

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

noncomputable def cos_theta (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1 ^ 2 + u.2 ^ 2) * Real.sqrt (v.1 ^ 2 + v.2 ^ 2))

theorem cos_theta_value :
  cos_theta a b = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end cos_theta_value_l643_643905


namespace tan_alpha_value_l643_643135

theorem tan_alpha_value (α : ℝ) (h₀ : sin α + cos α = 1 / 5) (h₁ : - pi / 2 ≤ α ∧ α ≤ pi / 2) : 
  tan α = - 3 / 4 := 
sorry

end tan_alpha_value_l643_643135


namespace students_liked_both_l643_643957

theorem students_liked_both (total : ℕ) (apple_pie : ℕ) (chocolate_cake : ℕ) (neither : ℕ) 
    (h_total : total = 50) (h_apple_pie : apple_pie = 22) (h_chocolate_cake : chocolate_cake = 20) (h_neither : neither = 15) :
    let liked_both := apple_pie + chocolate_cake - (total - neither) in
    liked_both = 7 :=
by
  sorry

end students_liked_both_l643_643957


namespace aubrey_travel_time_l643_643076

def aubrey_time_to_school (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem aubrey_travel_time :
  aubrey_time_to_school 88 22 = 4 := by
  sorry

end aubrey_travel_time_l643_643076


namespace number_of_shirts_that_weigh_1_pound_l643_643991

/-- 
Jon's laundry machine can do 5 pounds of laundry at a time. 
Some number of shirts weigh 1 pound. 
2 pairs of pants weigh 1 pound. 
Jon needs to wash 20 shirts and 20 pants. 
Jon has to do 3 loads of laundry. 
-/
theorem number_of_shirts_that_weigh_1_pound
    (machine_capacity : ℕ)
    (num_shirts : ℕ)
    (shirts_per_pound : ℕ)
    (pairs_of_pants_per_pound : ℕ)
    (num_pants : ℕ)
    (loads : ℕ)
    (weight_per_load : ℕ)
    (total_pants_weight : ℕ)
    (total_weight : ℕ)
    (shirt_weight_per_pound : ℕ)
    (shirts_weighing_one_pound : ℕ) :
  machine_capacity = 5 → 
  num_shirts = 20 → 
  pairs_of_pants_per_pound = 2 →
  num_pants = 20 →
  loads = 3 →
  weight_per_load = 5 → 
  total_pants_weight = (num_pants / pairs_of_pants_per_pound) →
  total_weight = (loads * weight_per_load) →
  shirts_weighing_one_pound = (total_weight - total_pants_weight) / num_shirts → 
  shirts_weighing_one_pound = 4 :=
by sorry

end number_of_shirts_that_weigh_1_pound_l643_643991


namespace cannot_form_set_l643_643002

-- Definitions based on conditions
def is_equilateral_triangle (x : Type) : Prop := sorry -- Placeholder definition
def is_student_in_first_year_high_school (x : Type) : Prop := sorry -- Placeholder definition
def is_overweight (x : Type) : Prop := sorry -- Placeholder definition
def is_irrational_number (x : Type) : Prop := sorry -- Placeholder definition

-- Statement to prove: the set of all overweight students in the first year cannot form a set
theorem cannot_form_set : ¬ (∃ S : set Type, ∀ x, (is_student_in_first_year_high_school x ∧ is_overweight x) ↔ x ∈ S) :=
sorry

end cannot_form_set_l643_643002


namespace susan_pay_missed_l643_643660

noncomputable def daily_pay : ℝ := 15 * 8

def total_workdays_in_two_weeks : ℕ := 2 * 5
def paid_vacation_days : ℕ := 6
def unpaid_vacation_days : ℕ := total_workdays_in_two_weeks - paid_vacation_days

noncomputable def pay_missed : ℝ := daily_pay * unpaid_vacation_days

theorem susan_pay_missed : pay_missed = 480 := by
  have calc1 : total_workdays_in_two_weeks = 10 := by norm_num
  have calc2 : unpaid_vacation_days = 4 := by norm_num
  have calc3 : daily_pay = 120 := by norm_num
  have calc4 : pay_missed = daily_pay * unpaid_vacation_days := rfl
  rw [calc3, calc2]
  norm_num
  sorry

end susan_pay_missed_l643_643660


namespace detail_understanding_word_meaning_guessing_logical_reasoning_l643_643389

-- Detail Understanding Question
theorem detail_understanding (sentence: String) (s: ∀ x : String, x ∈ ["He hardly watered his new trees,..."] → x = sentence) :
  sentence = "He hardly watered his new trees,..." :=
sorry

-- Word Meaning Guessing Question
theorem word_meaning_guessing (adversity_meaning: String) (meanings: ∀ y : String, y ∈ ["adversity means misfortune or disaster", "lack of water", "sufficient care/attention", "bad weather"] → y = adversity_meaning) :
  adversity_meaning = "adversity means misfortune or disaster" :=
sorry

-- Logical Reasoning Question
theorem logical_reasoning (hope: String) (sentences: ∀ z : String, z ∈ ["The author hopes his sons can withstand the tests of wind and rain in their life journey"] → z = hope) :
  hope = "The author hopes his sons can withstand the tests of wind and rain in their life journey" :=
sorry

end detail_understanding_word_meaning_guessing_logical_reasoning_l643_643389


namespace sheep_remain_l643_643279

theorem sheep_remain : ∀ (total_sheep sister_share brother_share : ℕ),
  total_sheep = 400 →
  sister_share = total_sheep / 4 →
  brother_share = (total_sheep - sister_share) / 2 →
  (total_sheep - sister_share - brother_share) = 150 :=
by
  intros total_sheep sister_share brother_share h_total h_sister h_brother
  rw [h_total, h_sister, h_brother]
  sorry

end sheep_remain_l643_643279


namespace contractor_engaged_days_l643_643034

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l643_643034


namespace possible_values_of_AD_l643_643020

variables (AB BC CA AC BD AD BE ED CD : ℝ)
variables (E : Type) [Euclidean E] -- Type for points in Euclidean space
variables [decidable_eq E] -- Rational decision functionality
variables (A B C D E : E) -- Points of the cyclic quadrilateral
hypothesis h1 : quadratic A B C D
hypothesis h2 : AB = BC ∧ BC = CA  -- Equilateral triangle side conditions
hypothesis h3 : BE = 19 ∧ ED = 6   -- Given segment lengths
hypothesis h4 : BD = BE + ED        -- Intersection of diagonals property
hypothesis h5 : ∀ t : E, diagonal_intersect AC BD E  -- Diagonal intersection at E

theorem possible_values_of_AD :
  ∃ AD : ℝ, (AD = 10 ∨ AD = 15) :=
by sorry

end possible_values_of_AD_l643_643020


namespace F_is_odd_l643_643267

variable (f : ℝ → ℝ)

def F (x : ℝ) : ℝ := f x - f (-x)

theorem F_is_odd : ∀ x : ℝ, F f (-x) = -F f x :=
by
  intro x
  simp [F]
  unfold F
  sorry

end F_is_odd_l643_643267


namespace product_ratio_geq_one_third_l643_643354

noncomputable theory

open_locale big_operators

variable {n : ℕ}

theorem product_ratio_geq_one_third (x : fin n → ℝ) (hsum : ∑ i, x i = 1 / 2) (hpos : ∀ i, x i > 0) :
  (∏ i, (1 - x i) / (1 + x i)) ≥ 1 / 3 :=
sorry

end product_ratio_geq_one_third_l643_643354


namespace actual_average_height_calculation_l643_643332

theorem actual_average_height_calculation :
  ∀ (n : ℕ) (avg_height incorrect_height correct_height : ℝ),
  n = 35 →
  avg_height = 180 →
  incorrect_height = 156 →
  correct_height = 106 →
  (real.round ((avg_height * n - incorrect_height + correct_height) / n * 100) / 100) = 178.57 :=
by
  intros n avg_height incorrect_height correct_height h1 h2 h3 h4
  sorry

end actual_average_height_calculation_l643_643332


namespace range_of_b_l643_643224

-- Given a function f(x)
def f (b x : ℝ) : ℝ := x^3 - 3 * b * x + 3 * b

-- Derivative of the function f(x)
def f' (b x : ℝ) : ℝ := 3 * x^2 - 3 * b

-- The theorem to prove the range of b
theorem range_of_b (b : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f' b x = 0) → (0 < b ∧ b < 1) := by
  sorry

end range_of_b_l643_643224


namespace solve_for_y_l643_643318

theorem solve_for_y (y : ℝ) (h : ∛(5 - 2 / y) = -3) : y = 1/16 :=
sorry

end solve_for_y_l643_643318


namespace number_of_symmetric_shapes_l643_643438

-- Definitions
def CentrallySymmetric (shape : Type) : Prop := 
  sorry -- Define centrally symmetric property

def AxiallySymmetric (shape : Type) : Prop :=
  sorry -- Define axially symmetric property

def Parallelogram : Type := sorry
def Rectangle : Type := sorry
def Rhombus : Type := sorry
def Square : Type := sorry
def EquilateralTriangle : Type := sorry

-- Given properties
axiom h1 : CentrallySymmetric Parallelogram ∧ ¬AxiallySymmetric Parallelogram
axiom h2 : CentrallySymmetric Rectangle ∧ AxiallySymmetric Rectangle
axiom h3 : CentrallySymmetric Rhombus ∧ AxiallySymmetric Rhombus
axiom h4 : CentrallySymmetric Square ∧ AxiallySymmetric Square
axiom h5 : ¬CentrallySymmetric EquilateralTriangle ∧ AxiallySymmetric EquilateralTriangle

theorem number_of_symmetric_shapes :
  (CentrallySymmetric Rectangle ∧ AxiallySymmetric Rectangle) ∧
  (CentrallySymmetric Rhombus ∧ AxiallySymmetric Rhombus) ∧
  (CentrallySymmetric Square ∧ AxiallySymmetric Square) ∧
  ¬(CentrallySymmetric Parallelogram ∧ AxiallySymmetric Parallelogram) ∧
  ¬(CentrallySymmetric EquilateralTriangle ∧ AxiallySymmetric EquilateralTriangle) →
  3 :=
by sorry

end number_of_symmetric_shapes_l643_643438


namespace probability_distance_exceeds_8_km_is_0_5_l643_643975

-- Define the basic setup of the problem
def central_house := Prop
def select_road (g : ℕ) : Fin 6 := sorry
def travel_speed := 5  -- in km/h
def travel_time := 1   -- in hours
def travel_distance := travel_speed * travel_time  -- in km
def angle_between_roads := 360 / 6  -- in degrees
def distance_geologists (g1 g2 : ℕ) : ℝ := sorry  -- Define the function to compute distance based on chosen roads

-- Define the probability calculation
def total_combinations := 6 * 6
def favorable_combinations := 18
def probability_distance_exceeds_8_km := favorable_combinations / total_combinations

theorem probability_distance_exceeds_8_km_is_0_5 : probability_distance_exceeds_8_km = 0.5 := by
  sorry

end probability_distance_exceeds_8_km_is_0_5_l643_643975


namespace parallel_vectors_eq_l643_643536

theorem parallel_vectors_eq (t : ℝ) : ∀ (m n : ℝ × ℝ), m = (2, 8) → n = (-4, t) → (∃ k : ℝ, n = k • m) → t = -16 :=
by 
  intros m n hm hn h_parallel
  -- proof goes here
  sorry

end parallel_vectors_eq_l643_643536


namespace intersection_with_xz_plane_l643_643120

open Real

def point1 := (1, 2, 3)
def point2 := (4, 0, 5)
def direction_vector := (point2.1 - point1.1, point2.2 - point1.2, point2.3 - point1.3)
def line_param (t : ℝ) := (point1.1 + t * direction_vector.1, point1.2 + t * direction_vector.2, point1.3 + t * direction_vector.3)

theorem intersection_with_xz_plane :
  ∃ t, line_param t = (4, 0, 5) := sorry

end intersection_with_xz_plane_l643_643120


namespace solve_polynomial_l643_643485

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem solve_polynomial : Q (real.cbrt 3 + 2) = 0 :=
by {
  -- The polynomial is monic hence the first coefficient is 1
  -- The polynomial has integer coefficients
  -- Substitute y = cbrt(3) + 2 into Q(y) and show it equals 0
  sorry
}

end solve_polynomial_l643_643485


namespace trapezoid_area_is_59_l643_643981

def is_trapezoid (A B C D : Point) : Prop :=
  -- Define the trapezoid property
  sorry

def distance (P Q : Point) : Real :=
  -- Distance formula
  sorry 

theorem trapezoid_area_is_59
  (A B C D : Point)
  (h_trapezoid : is_trapezoid A B C D)
  (h_BC : distance B C = 9.5)
  (h_AD : distance A D = 20)
  (h_AB : distance A B = 5)
  (h_CD : distance C D = 8.5) :
  let height := sqrt (max 0 (25 - (3^2)) / 2) in
  let area := 0.5 * (9.5 + 20) * height in
  area = 59 :=
sorry

end trapezoid_area_is_59_l643_643981


namespace expression_evaluation_l643_643847

theorem expression_evaluation : 3 - (-3) ^ (3 - (-1)) = -78 := by
  sorry

end expression_evaluation_l643_643847


namespace number_of_proper_subsets_of_A_l643_643682

-- Define the set A = {1, 2, 3}
def A : set ℕ := {1, 2, 3}

-- Define what a proper subset is in Lean
def proper_subset (s t : set ℕ) : Prop := s ⊂ t

-- The theorem to prove:
theorem number_of_proper_subsets_of_A : finset.card (finset.powerset A \ {A}) = 7 :=
by sorry

end number_of_proper_subsets_of_A_l643_643682


namespace number_of_true_statements_is_one_l643_643273

theorem number_of_true_statements_is_one (i : ℂ) (a b : ℝ) (hi : i * i = -1) :
  let s1 := (a + 1) * i
  let s2 := if a > b then (a + i) > (b + i) else false
  let s3 := 
    let p := (a^2 - 1) + (a^2 + 3 * a + 2) * i
    if p.im ≠ 0 && p.re = 0 then a = 1 ∨ a = -1 else false
  let s4 := 2 * i^2 > 3 * i^2
  (if s1.im ≠ 0 then 1 else 0) + (if s2 then 1 else 0) + (if s3 then 1 else 0) + (if s4 then 1 else 0) = 1 :=
sorry

end number_of_true_statements_is_one_l643_643273


namespace multiples_of_6_with_units_digit_2_less_than_200_l643_643205

theorem multiples_of_6_with_units_digit_2_less_than_200:
  (∃ (k : ℕ), k > 0 ∧ 6 * k < 200 ∧ (6 * k) % 10 = 2) → Finset.card
    (Finset.filter (λ k, (6 * k) % 10 = 2) (Finset.range 34)) = 5 :=
by
  sorry

end multiples_of_6_with_units_digit_2_less_than_200_l643_643205


namespace number_of_lines_passing_through_and_intersecting_parabola_l643_643932

theorem number_of_lines_passing_through_and_intersecting_parabola
  (point : ℤ × ℤ)
  (parabola : ℤ → ℤ)
  (k : ℤ)
  (a b : ℤ) :
  point = (0, 2019) ∧
  (∀ x, parabola x = x^2) ∧
  k = a + b ∧
  (a * b = -2019) ∧
  (a ≠ b) ∧
  (a + b = k) →
  (9 = { (a, b) | a * b = -2019 ∧ a ≠ b }.to_finset.card) :=
by {
  sorry
}

end number_of_lines_passing_through_and_intersecting_parabola_l643_643932


namespace max_ab_is_half_e_cubed_l643_643152

noncomputable def max_value_ab (a b : ℝ) : ℝ :=
  if x - a * Real.log x + a - b < 0 ∀ (x : ℝ) (hx : 0 < x) then ∅ else 
    let f := λ x : ℝ, x^2 * (2 - Real.log x) in
    let x_max := Real.exp (3 / 2) in
    let f_max := f x_max in
    let a_max := Real.sqrt (f_max / (2 - Real.log x_max)) in
    a_max * (2 - Real.log a_max)

theorem max_ab_is_half_e_cubed (a b : ℝ) (h : ∀ x : ℝ, 0 < x → ¬ (x - a * Real.log x + a - b < 0)) :
  max_value_ab a b = (1 / 2) * Real.exp 3 := 
sorry

end max_ab_is_half_e_cubed_l643_643152


namespace geom_seq_fraction_l643_643960

theorem geom_seq_fraction (a_1 a_2 a_3 a_4 a_5 q : ℝ)
  (h1 : q > 0)
  (h2 : a_2 = q * a_1)
  (h3 : a_3 = q^2 * a_1)
  (h4 : a_4 = q^3 * a_1)
  (h5 : a_5 = q^4 * a_1)
  (h_arith : a_2 - (1/2) * a_3 = (1/2) * a_3 - a_1) :
  (a_3 + a_4) / (a_4 + a_5) = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end geom_seq_fraction_l643_643960


namespace f_six_value_l643_643741

def f : ℤ → ℤ
| n := if n = 4 then 20 else f(n-1) - n

theorem f_six_value : f 6 = 9 :=
by {
  -- The proof is skipped, but the theorem statement is thus given
  sorry
}

end f_six_value_l643_643741


namespace claire_cupcakes_sum_l643_643726

theorem claire_cupcakes_sum :
  (∑ n in {n : ℕ | n < 100 ∧ n % 6 = 3 ∧ n % 8 = 2}.toFinset , n) = 150 :=
by
  sorry

end claire_cupcakes_sum_l643_643726


namespace arg_z1_div_z2_l643_643830

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry

theorem arg_z1_div_z2 :
  abs z1 = 1 ∧ abs z2 = 1 ∧ z2 - z1 = -1 →
  (complex.arg (z1 / z2) = π / 3 ∨ complex.arg (z1 / z2) = 5 * π / 3) :=
by
  sorry

end arg_z1_div_z2_l643_643830


namespace average_of_eight_consecutive_integers_l643_643843

theorem average_of_eight_consecutive_integers (c d : ℝ) (h : (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6) + (c + 7)) / 8 = d) :
  ((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6) + (d + 7) + (d + 8)) / 8 = c + 8 := by 
  sorry

end average_of_eight_consecutive_integers_l643_643843


namespace symmetric_line_eq_ln_l643_643142

theorem symmetric_line_eq_ln
  (l : ℝ → ℝ → Prop)
  (x y : ℝ)
  (line1 : ℝ → ℝ → Prop := λ x y, 2 * x - 3 * y + 4 = 0)
  (reflection_line : ℝ := 1)
  (is_symmetric : (∀ x y, l x y ↔ line1 (2 - x) y))
  :
  (∀ x y, l x y ↔ 2 * x + 3 * y - 8 = 0) :=
sorry

end symmetric_line_eq_ln_l643_643142


namespace grayson_travels_further_l643_643563

noncomputable def grayson_first_part_distance : ℝ := 25 * 1
noncomputable def grayson_second_part_distance : ℝ := 20 * 0.5
noncomputable def total_distance_grayson : ℝ := grayson_first_part_distance + grayson_second_part_distance

noncomputable def total_distance_rudy : ℝ := 10 * 3

theorem grayson_travels_further : (total_distance_grayson - total_distance_rudy) = 5 := by
  sorry

end grayson_travels_further_l643_643563


namespace num_integer_solutions_l643_643569

theorem num_integer_solutions : 
  (∃ (x1 x2 x3 x4: ℤ), 1 ≤ x1 ∧ x1 ≤ 5 ∧ -2 ≤ x2 ∧ x2 ≤ 4 ∧ 0 ≤ x3 ∧ x3 ≤ 5 ∧ 3 ≤ x4 ∧ x4 ≤ 9 ∧ x1 + x2 + x3 + x4 = 18) →
  (finset.card 
    (finset.filter 
      (λ (t : fin 19 × fin 19 × fin 19 × fin 19), 
        1 ≤ t.1.1 ∧ t.1.1 ≤ 5 ∧ -2 ≤ t.1.2 ∧ t.1.2 ≤ 4 ∧ 
        0 ≤ t.2.1 ∧ t.2.1 ≤ 5 ∧ 3 ≤ t.2.2 ∧ t.2.2 ≤ 9 ∧ 
        t.1.1 + t.1.2 + t.2.1 + t.2.2 = 18)
      (finset.range 6 ×ˢ finset.range 7 ×ˢ finset.range 6 ×ˢ finset.range (10 - 2 + 1))
    ) = 121) := 
begin
  sorry
end

end num_integer_solutions_l643_643569


namespace HamiltonianCircuitLength_l643_643749

theorem HamiltonianCircuitLength (m n : ℕ) :
  (∃ P : list (ℕ × ℕ),
   P.nodup ∧
   ∀ i ∈ P, i.1 ≤ n ∧ i.2 ≤ m ∧ -- within the boundary
   -- adjacency check for edges in the grid is omitted in the statement for simplicity
   (P.head = (0, 0) ∧ P.last = (n, m))) ↔ (m % 2 = 1 ∨ n % 2 = 1) ∧ (n + 1) * (m + 1) :=
by sorry

end HamiltonianCircuitLength_l643_643749


namespace jiho_initial_money_l643_643608

theorem jiho_initial_money (M : ℝ) 
  (fish_shop_expense : M/2) 
  (fruit_shop_expense : 1/3 * (M/2) + 5000)
  (remaining_money : M/2 - (1/3 * (M/2) + 5000) = 5000) :
  M = 30000 := 
sorry

end jiho_initial_money_l643_643608


namespace locus_of_intersection_is_half_line_l643_643307

-- Define points and segments
variables {Point : Type}
variables (A B C P Q : Point)

-- Define the proportion condition
axiom proportion_condition : ∀ {CA AQ CP PB : ℝ}, CA / AQ = CP / PB

-- Define the external and internal division condition
axiom division_condition : (∃ P : Point, ∀ [internally_divided_by P A B, externally_divided_by Q C P])

-- Define the side condition for A, Q, and C
axiom opposite_sides : ∀ {A Q C : Point}, on_opposite_sides A Q C

-- Theorem statement
theorem locus_of_intersection_is_half_line (A B C P Q : Point)
    (proportion : proportion_condition)
    (division : division_condition)
    (side_cond : opposite_sides A Q C) : 
    ∃ (L : Line), is_half_line L ∧ intersects (line_through A P) (line_through B Q) L := 
sorry

end locus_of_intersection_is_half_line_l643_643307


namespace trapezoid_isosceles_l643_643147

noncomputable def is_perpendicular (A B: Point) (line: Line) : Prop := sorry
noncomputable def midpoint (A B: Point) : Point := sorry

theorem trapezoid_isosceles 
    (ABCD : Quadrilateral) (A B C D : Point) 
    (h_trapezoid : ABCD.is_trapezoid AD BC)
    (K : Point) (hK : K = midpoint A C)
    (L : Point) (hL : L = midpoint B D)
    (h_perp_A : is_perpendicular A (Line.mk K L) (Line.mk C D))
    (h_perp_D : is_perpendicular D (Line.mk K L) (Line.mk A B)) :
    ABCD.is_isosceles_trapezoid :=
begin
    sorry
end

end trapezoid_isosceles_l643_643147


namespace cyclists_meeting_time_l643_643567

theorem cyclists_meeting_time :
  ∃ t : ℕ, t = Nat.lcm 7 (Nat.lcm 12 9) ∧ t = 252 :=
by
  use 252
  have h1 : Nat.lcm 7 (Nat.lcm 12 9) = 252 := sorry
  exact ⟨rfl, h1⟩

end cyclists_meeting_time_l643_643567


namespace magnitude_of_product_of_c_l643_643857

def recurrence_relation (c : ℂ) (x : ℕ → ℂ) : Prop :=
  ∀ n : ℕ, x 1 = 1 ∧ x 2 = c^2 - 4c + 7 ∧
    x (n + 1) = (c^2 - 2c)^2 * x n * x (n - 1) + 2 * x n - x (n - 1)

theorem magnitude_of_product_of_c : (∃ (c : ℂ), recurrence_relation c (λ n, if n = 0 then 0 else if n = 1 then 1 else if n = 2 then (c^2 - 4c + 7) else (c^2 - 2c)^2 * (n - 1) * (n - 2) + 2 * (n - 1) - (n - 2)) ∧ x 1006 = 2011) →
  ∃ (c : ℂ), abs c = 2 :=
by
  sorry

end magnitude_of_product_of_c_l643_643857


namespace ratio_of_times_l643_643359

theorem ratio_of_times (V_b V_s : ℝ) (h1 : V_b = 78) (h2 : V_s = 26) :
  let V_up := V_b - V_s in
  let V_down := V_b + V_s in
  V_up ≠ 0 ∧ V_down ≠ 0 ∧ (V_down / V_up = 2) :=
by
  let V_up := V_b - V_s
  let V_down := V_b + V_s
  have hV_up : V_up = 52, from by rw [V_up, h1, h2]; norm_num,
  have hV_down : V_down = 104, from by rw [V_down, h1, h2]; norm_num,
  have hV_up_nonzero : V_up ≠ 0, from by rw [hV_up]; norm_num,
  have hV_down_nonzero : V_down ≠ 0, from by rw [hV_down]; norm_num,
  have h_ratio : V_down / V_up = 2, from by rw [hV_down, hV_up]; norm_num,
  exact ⟨hV_up_nonzero, hV_down_nonzero, h_ratio⟩

end ratio_of_times_l643_643359


namespace find_y_l643_643220

theorem find_y (x y : ℝ) (h1 : x = 100) (h2 : x^3 * y - 3 * x^2 * y + 3 * x * y = 3000000) : 
  y = 3000000 / (100^3 - 3 * 100^2 + 3 * 100 * 1) :=
by sorry

end find_y_l643_643220


namespace rationalization_sum_l643_643302

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalization_sum : rationalize_denominator = 75 := by
  sorry

end rationalization_sum_l643_643302


namespace min_value_in_inf_l643_643524

noncomputable def min_value (a b : ℝ) : ℝ := 
  if h : a > 0 ∧ b > 0 ∧ a + 2 * b = 2 then (1/a + 1/b) else 0

theorem min_value_in_inf {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 2) :
  ∃ a b, min_value a b = (3/2 + Real.sqrt 2) :=
begin
  use (2 * Real.sqrt 2 - 2),
  use (2 - Real.sqrt 2),
  have ha' : 2 * Real.sqrt 2 - 2 > 0,
  { sorry }, -- proof that 2 * sqrt(2) - 2 > 0 is left,
  have hb' : 2 - Real.sqrt 2 > 0,
  { sorry }, -- proof that 2 - sqrt(2) > 0 is left,
  have hab' : (2 * Real.sqrt 2 - 2) + 2 * (2 - Real.sqrt 2) = 2,
  { sorry }, -- proof that (2 * sqrt(2) - 2) + 2 * (2 - sqrt(2)) = 2 is left,
  rw [min_value, dif_pos (and.intro ha' (and.intro hb' hab'))],
  norm_num,
end

end min_value_in_inf_l643_643524


namespace exists_four_acute_angles_among_five_rays_l643_643987

theorem exists_four_acute_angles_among_five_rays : 
  ∃ (angles : Fin 10 → ℝ), 
    (∀ (i : Fin 10), 0 < angles i ∧ angles i < 360) ∧
    (finset.univ.sum angles = 360) ∧
    (finset.filter (λ i, angles i < 90) finset.univ).card = 4 :=
sorry

end exists_four_acute_angles_among_five_rays_l643_643987


namespace collinear_A_F_C_l643_643668

noncomputable def geometry_problem : Prop :=
  ∃ (S₁ S₂ : Type) (F A B C : Point) (l parallel_line : Line),
    touches_externally_at S₁ S₂ F ∧
    is_tangent_to_line S₁ A l ∧
    is_tangent_to_line S₂ B l ∧
    is_parallel parallel_line l ∧
    is_tangent_to_line S₂ C parallel_line ∧
    intersects_at_two_points parallel_line S₁ ∧
    collinear A F C

theorem collinear_A_F_C : geometry_problem :=
sorry

end collinear_A_F_C_l643_643668


namespace trig_identity_for_15_degrees_l643_643698

theorem trig_identity_for_15_degrees :
  cos 15 ^ 2 - sin 15 ^ 2 + sin 15 * cos 15 = (1 + 2 * sqrt 3) / 4 :=
by sorry

end trig_identity_for_15_degrees_l643_643698


namespace vertex_quadrant_l643_643983

noncomputable def vertex_in_second_quadrant (f : ℝ → ℝ) : Prop :=
∃ h k : ℝ, f = λ x, -(x + h)^2 + k ∧ h = -1 ∧ k = 2 ∧ h < 0 ∧ k > 0

theorem vertex_quadrant : vertex_in_second_quadrant (λ x, -(x + 1)^2 + 2) := by
  sorry

end vertex_quadrant_l643_643983


namespace equilateral_triangle_sum_l643_643519

noncomputable
def equilateral_triangle (ABC : Triangle) : Prop :=
  ABC.is_equilateral ∧ ABC.side_length = 1

def points_on_segment (B C : Point) (n : ℕ) : list Point :=
  (list.range (n-1)).map (λ k, segment_point B C n k)

def S_n (A B C : Point) (points : list Point) : ℝ :=
  (list.zip_with (λ P Q, vector_dot (vector_from A P) (vector_from A Q))
    (A :: points) (points ++ [C])).sum

theorem equilateral_triangle_sum (n : ℕ) (ABC : Triangle)
  (h_eq_tri : equilateral_triangle ABC)
  (A B C : Point) (points : list Point)
  (h_points_eq : points = points_on_segment B C n) :
  S_n A B C points = (5 * n^2 - 2) / (6 * n) :=
sorry

end equilateral_triangle_sum_l643_643519


namespace minimum_hotels_l643_643980

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (a b c d : ℝ)
variable (M_eq : M = ![![a, b], ![c, d]])
variable (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
variable (tourist : ∀ x ∈ {a, b, c, d}, x ∈ {1, 2, 4, 8, 16})

theorem minimum_hotels : 
  ∃ hotels : Finset (Matrix (Fin 2) (Fin 2) ℝ), 
    (∀ attraction : Matrix (Fin 2) (Fin 2) ℝ, 
      (∀ x ∈ attraction, x ∈ {1, 2, 4, 8, 16}) → 
      ∃ hotel ∈ hotels, acc_slash teleport attraction hotel) 
    → hotels.card = 17 := 
sorry

end minimum_hotels_l643_643980


namespace count_sequences_l643_643206

theorem count_sequences :
  ∃ (seqs : Finset (List ℕ)), 
    (∀ (s ∈ seqs), s.length = 2008 ∧ (∀ i, 1 ≤ i ∧ i ≤ 2008 → i ∈ s.take i))
    ∧ seqs.card = 2^2007 :=
sorry

end count_sequences_l643_643206


namespace fraction_second_year_students_l643_643012

theorem fraction_second_year_students
  (total_students : ℕ)
  (third_year_students : ℕ)
  (second_year_students : ℕ)
  (h1 : third_year_students = total_students * 30 / 100)
  (h2 : second_year_students = total_students * 10 / 100) :
  (second_year_students : ℚ) / (total_students - third_year_students) = 1 / 7 := by
  sorry

end fraction_second_year_students_l643_643012


namespace base_area_of_rect_prism_l643_643795

theorem base_area_of_rect_prism (r : ℝ) (h : ℝ) (V : ℝ) (h_rate : ℝ) (V_rate : ℝ) (conversion : ℝ) :
  V_rate = conversion * V ∧ h_rate = h → ∃ A : ℝ, A = V / h ∧ A = 100 :=
by
  sorry

end base_area_of_rect_prism_l643_643795


namespace total_ways_select_representatives_l643_643308

theorem total_ways_select_representatives :
  (nat.choose 5 2 * nat.choose 4 2) + (nat.choose 5 3 * nat.choose 4 1) = 100 :=
by sorry

end total_ways_select_representatives_l643_643308


namespace infinite_series_equality_l643_643495

noncomputable def closest_integer (n : ℕ) : ℕ :=
  if (real.to_nnreal (real.sqrt n).ceil - (real.sqrt n)).abs < (real.sqrt n - real.to_nnreal (real.sqrt n).floor).abs then
    real.to_nnreal (real.sqrt n).ceil.nat_abs
  else
    real.to_nnreal (real.sqrt n).floor.nat_abs

lemma closest_integer_properties (n k : ℕ) (h : closest_integer n = k) :
  k - 1/2 < real.sqrt n ∧ real.sqrt n < k + 1/2 :=
sorry

theorem infinite_series_equality :
  ∑' n : ℕ, (3 ^ closest_integer n + 3 ^ (- closest_integer n)) / 2 ^ n =
  ∑' k : ℕ, (3 ^ k * 2 ^ (- k ^ 2) - 3 ^ (- k) * 2 ^ (- k ^ 2)) :=
sorry

end infinite_series_equality_l643_643495


namespace cathy_initial_money_l643_643078

-- Definitions of the conditions
def moneyFromDad : Int := 25
def moneyFromMom : Int := 2 * moneyFromDad
def totalMoneyReceived : Int := moneyFromDad + moneyFromMom
def currentMoney : Int := 87

-- Theorem stating the proof problem
theorem cathy_initial_money (initialMoney : Int) :
  initialMoney + totalMoneyReceived = currentMoney → initialMoney = 12 :=
by
  sorry

end cathy_initial_money_l643_643078


namespace total_cost_of_toppings_is_47_5_l643_643610

def sundaes_monday : Nat := 40
def sundaes_tuesday : Nat := 20
def mms_per_monday_sundae : Nat := 6
def gummy_bears_per_monday_sundae : Nat := 4
def mini_marshmallows_per_monday_sundae : Nat := 8
def mms_per_tuesday_sundae : Nat := 10
def gummy_bears_per_tuesday_sundae : Nat := 5
def mini_marshmallows_per_tuesday_sundae : Nat := 12
def mm_pack_quantity : Nat := 40
def mm_pack_cost : Float := 2
def gummy_bear_pack_quantity : Nat := 30
def gummy_bear_pack_cost : Float := 1.5
def mini_marshmallow_pack_quantity : Nat := 50
def mini_marshmallow_pack_cost : Float := 1

theorem total_cost_of_toppings_is_47_5 : 
  let total_mms := (sundaes_monday * mms_per_monday_sundae) + (sundaes_tuesday * mms_per_tuesday_sundae),
      total_gummy_bears := (sundaes_monday * gummy_bears_per_monday_sundae) + (sundaes_tuesday * gummy_bears_per_tuesday_sundae),
      total_mini_marshmallows := (sundaes_monday * mini_marshmallows_per_monday_sundae) + (sundaes_tuesday * mini_marshmallows_per_tuesday_sundae),
      mm_packs_needed := Float.ofNat total_mms / Float.ofNat mm_pack_quantity,
      gummy_bear_packs_needed := Float.ofNat total_gummy_bears / Float.ofNat gummy_bear_pack_quantity,
      mini_marshmallow_packs_needed := Float.ofNat total_mini_marshmallows / Float.ofNat mini_marshmallow_pack_quantity,
      mm_cost := mm_pack_cost * Float.ceil mm_packs_needed,
      gummy_bear_cost := gummy_bear_pack_cost * Float.ceil gummy_bear_packs_needed,
      mini_marshmallow_cost := mini_marshmallow_pack_cost * Float.ceil mini_marshmallow_packs_needed,
      total_cost := mm_cost + gummy_bear_cost + mini_marshmallow_cost
  in total_cost = 47.5 := 
  sorry

end total_cost_of_toppings_is_47_5_l643_643610


namespace fifth_term_geometric_sequence_l643_643221

-- Define the geometric sequence with the given conditions
theorem fifth_term_geometric_sequence :
  ∃ (a_1 q : ℝ), a_1 = 8 ∧ (a_1 * q ^ 6 = 5832) ∧ (8 * q ^ 4 = 648) :=
by
  let a_1 : ℝ := 8
  let q : ℝ := (5832 / 8)^(1/6)
  have h1 : a_1 = 8 := rfl
  have h2 : a_1 * q ^ 6 = 5832 := by
    calc
      a_1 * q ^ 6 = 8 * ((5832 / 8)^(1/6)) ^ 6 : by rw[h1]
          ... = 5832 : by simp [div_mul_cancel, (5832 : ℝ)]
  have h3 : 8 * q ^ 4 = 648 := by
    calc
      8 * q ^ 4 = 8 * (5832/8)^(4/6) : by rw[h1]
          ... = 8 * (5832/8)^(2/3) : by simp
          ... = 8 * 81 : by norm_num
          ... = 648 : by norm_num
  exact ⟨a_1, q, h1, h2, h3⟩

end fifth_term_geometric_sequence_l643_643221


namespace new_function_expression_l643_643924

def initial_function (x : ℝ) : ℝ := -2 * x ^ 2

def shifted_function (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 3

theorem new_function_expression :
  (∀ x : ℝ, (initial_function (x + 1) - 3) = shifted_function x) :=
by
  sorry

end new_function_expression_l643_643924


namespace computer_price_after_15_years_l643_643100

-- The conditions of the problem
def initial_price : ℝ := 8100
def yearly_decrease_factor : ℝ := 2 / 3
def years_elapsed : ℝ := 15
def periods : ℝ := 5

-- The statement to be proved
theorem computer_price_after_15_years :
  initial_price * (yearly_decrease_factor ^ (years_elapsed / periods)) = 2400 := by
  sorry

end computer_price_after_15_years_l643_643100


namespace correct_calculation_l643_643001

-- Definition of the conditions
def condition1 (a : ℕ) : Prop := a^2 * a^3 = a^6
def condition2 (a : ℕ) : Prop := (a^2)^10 = a^20
def condition3 (a : ℕ) : Prop := (2 * a) * (3 * a) = 6 * a
def condition4 (a : ℕ) : Prop := a^12 / a^2 = a^6

-- The main theorem to state that condition2 is the correct calculation
theorem correct_calculation (a : ℕ) : condition2 a :=
sorry

end correct_calculation_l643_643001


namespace total_oranges_picked_l643_643372

/-- Michaela needs 20 oranges to get full --/
def oranges_michaela_needs : ℕ := 20

/-- Cassandra needs twice as many oranges as Michaela to get full --/
def oranges_cassandra_needs : ℕ := 2 * oranges_michaela_needs

/-- After both have eaten until they are full, 30 oranges remain --/
def oranges_remaining : ℕ := 30

/-- The total number of oranges eaten by both Michaela and Cassandra --/
def oranges_eaten : ℕ := oranges_michaela_needs + oranges_cassandra_needs

/-- Prove that the total number of oranges picked from the farm is 90 --/
theorem total_oranges_picked : oranges_eaten + oranges_remaining = 90 := by
  sorry

end total_oranges_picked_l643_643372


namespace julian_total_friends_l643_643992

def boyd_total_friends : ℕ := 100
def boyd_percentage_boys : ℚ := 36 / 100
def julian_percentage_girls : ℚ := 2 / 5

theorem julian_total_friends :
  ∀ (boyd_total_friends : ℕ) 
    (boyd_percentage_boys : ℚ) 
    (julian_percentage_girls : ℚ),
  boyd_total_friends = 100 →
  boyd_percentage_boys = 36 / 100 →
  julian_percentage_girls = 2 / 5 →
  let boyd_girls := (1 - boyd_percentage_boys) * boyd_total_friends in
  let julian_girls := boyd_girls / 2 in
  let total_julian_friends := julian_girls / julian_percentage_girls in
  total_julian_friends = 80 :=
by
  intros
  simp [boyd_total_friends, boyd_percentage_boys, julian_percentage_girls]
  let boyd_girls := (1 - 36 / 100 : ℚ) * 100
  let julian_girls := boyd_girls / 2
  let total_julian_friends := julian_girls / (2 / 5 : ℚ)
  have : total_julian_friends = 80 := by norm_num
  exact this

end julian_total_friends_l643_643992


namespace percentage_william_land_l643_643398

-- Definitions of the given conditions
def total_tax_collected : ℝ := 3840
def william_tax : ℝ := 480

-- Proof statement
theorem percentage_william_land :
  ((william_tax / total_tax_collected) * 100) = 12.5 :=
by
  sorry

end percentage_william_land_l643_643398


namespace a_n_formula_b_n_inequality_l643_643928

variable (a b: ℕ → ℝ)

axiom a1 : a 1 = 1

axiom rec : ∀ n : ℕ, a (n + 1) + 2 * a n * a (n + 1) - a n = 0

axiom def_b : ∀ n : ℕ, 2 * a n + b n = 1 

theorem a_n_formula (n : ℕ) : a n = 1 / (2 * n - 1) := sorry

theorem b_n_inequality (n : ℕ) : 
  (∑ i in Finset.range n, ((1 - b i) * (1 - b (i + 1)))) < 2 :=
sorry

end a_n_formula_b_n_inequality_l643_643928


namespace numbers_with_digit_one_are_more_numerous_l643_643437

theorem numbers_with_digit_one_are_more_numerous :
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  total_numbers - numbers_without_one > numbers_without_one :=
by
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  show total_numbers - numbers_without_one > numbers_without_one
  sorry

end numbers_with_digit_one_are_more_numerous_l643_643437


namespace flagpole_proof_l643_643422

noncomputable def flagpole_height (AC AD DE : ℝ) (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) : ℝ :=
  let DC := AC - AD
  let h_ratio := DE / DC
  h_ratio * AC

theorem flagpole_proof (AC AD DE : ℝ) (h_AC : AC = 4) (h_AD : AD = 3) (h_DE : DE = 1.8) 
  (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) :
  flagpole_height AC AD DE h_ABC_DEC = 7.2 := by
  sorry

end flagpole_proof_l643_643422


namespace total_customers_proof_l643_643809

variable (total_tip tip_per_customer non_tipping_customers : ℕ)
variable (total_customers : ℕ)
variable (a : ℕ)

-- Given conditions
def conditions := 
  total_tip = 15 ∧ 
  tip_per_customer = 3 ∧ 
  non_tipping_customers = 5 ∧ 
  a = total_tip / tip_per_customer

-- Total number of customers calculated based on the conditions
def calculate_total_customers (total_tip tip_per_customer non_tipping_customers : ℕ) : ℕ :=
  let a := total_tip / tip_per_customer in
  a + non_tipping_customers

-- Proof statement
theorem total_customers_proof : 
  calculate_total_customers 15 3 5 = 10 :=
by
  unfold calculate_total_customers
  rw [Nat.div_eq_of_lt]
  swap, exact non_tipping_customers  -- assuming non_tipping_customers is calculated correctly
  sorry

end total_customers_proof_l643_643809


namespace pasture_monthly_cost_l643_643442

-- Definitions
def yearly_total_expense := 15890
def daily_food_cost := 10
def food_days_in_year := 365
def lesson_cost := 60
def lessons_per_week := 2
def weeks_per_year := 52
def months_in_year := 12

-- Theorem
theorem pasture_monthly_cost : 
  let yearly_food_expense := daily_food_cost * food_days_in_year in
  let yearly_lesson_expense := lesson_cost * lessons_per_week * weeks_per_year in
  let known_annual_expenses := yearly_food_expense + yearly_lesson_expense in
  let yearly_pasture_expense := yearly_total_expense - known_annual_expenses in
  yearly_pasture_expense / months_in_year = 500 :=
by
  -- Proof goes here
  sorry

end pasture_monthly_cost_l643_643442


namespace runner_injury_point_l643_643428

-- Define the initial setup conditions
def total_distance := 40
def second_half_time := 10
def first_half_additional_time := 5

-- Prove that given the conditions, the runner injured her foot at 20 miles.
theorem runner_injury_point : 
  ∃ (d v : ℝ), (d = 5 * v) ∧ (total_distance - d = 5 * v) ∧ (10 = second_half_time) ∧ (first_half_additional_time = 5) ∧ (d = 20) :=
by
  sorry

end runner_injury_point_l643_643428


namespace find_x_from_exp_log_condition_l643_643483

theorem find_x_from_exp_log_condition (x : ℝ) (h : 9 ^ real.log x / real.log 5 = 81) : x = 25 :=
sorry

end find_x_from_exp_log_condition_l643_643483


namespace solve_for_y_l643_643321

theorem solve_for_y (y : ℝ) (h : ∛(5 - 2 / y) = -3) : y = 1/16 :=
sorry

end solve_for_y_l643_643321


namespace tangent_parallel_BC_l643_643262

variables {A B C S : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited S]
variables (triangle : Triangle A B C)
variables (Γ : Circumcircle triangle)
variables (S : PointOnCircumcircleMidpointBCNotContainA triangle Γ)

theorem tangent_parallel_BC (h₁ : midpoint_of_arc S B C)
  (h₂ : is_circumcircle Γ triangle)
  (h₃ : S = midpoint_of_arc_BC_not_contain_A) :
  is_parallel (tangent_at Γ S) (line B C) :=
sorry

end tangent_parallel_BC_l643_643262


namespace find_vector_b_l643_643997

def vector3 (α : Type*) := (α × α × α)

open_locale real_inner_product_space

variables (a c : vector3 ℝ)
def collinear (a b c : vector3 ℝ) : Prop :=
∃ t : ℝ, b = (1 - t) • a + t • c

def angle_bisector (a b c : vector3 ℝ) : Prop :=
real.angle (b - a) (c - b) = real.angle a c / 2

theorem find_vector_b (a c : vector3 ℝ)
  (h₁ : a = (3, -2, 6))
  (h₂ : c = (1, 4, -2)) :
  ∃ b : vector3 ℝ,
  collinear a b c ∧ angle_bisector a b c ∧
  b = (141 / 52, 94 / 13, 15 / 13) :=
sorry

end find_vector_b_l643_643997


namespace fourth_grade_planted_89_l643_643099

-- Define the number of trees planted by the fifth grade
def fifth_grade_trees : Nat := 114

-- Define the condition that the fifth grade planted twice as many trees as the third grade
def third_grade_trees : Nat := fifth_grade_trees / 2

-- Define the condition that the fourth grade planted 32 more trees than the third grade
def fourth_grade_trees : Nat := third_grade_trees + 32

-- Theorem to prove the number of trees planted by the fourth grade is 89
theorem fourth_grade_planted_89 : fourth_grade_trees = 89 := by
  sorry

end fourth_grade_planted_89_l643_643099


namespace contractor_engaged_days_l643_643033

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l643_643033


namespace carpet_area_l643_643628

theorem carpet_area (base_feet : ℕ) (height_feet : ℕ) (feet_per_yard : ℕ)
  (h_base : base_feet = 15) (h_height : height_feet = 10) (h_conv : feet_per_yard = 3) :
  let base_yards := base_feet / feet_per_yard
  let height_yards := height_feet / feet_per_yard in
  (1 / 2) * base_yards * height_yards = 25 / 3 := 
by {
  sorry
}

end carpet_area_l643_643628


namespace perfect_square_in_set_or_subset_l643_643228

theorem perfect_square_in_set_or_subset (A : Finset ℕ) (hA : A.card = 1986)
  (hProd : ∃ d : ℕ, d.prime_factors.card = 1985 ∧ (A.prod id) = d) :
  (∃ x ∈ A, is_square x) ∨ (∃ B ⊆ A, is_square (B.prod id)) :=
sorry

end perfect_square_in_set_or_subset_l643_643228


namespace spaceship_speed_400_l643_643129

def spaceship_speed (n : ℕ) : ℝ :=
  if n = 200 then 500 else
  if n < 200 then 0 else
  spaceship_speed (n - 100) / 2

theorem spaceship_speed_400 : spaceship_speed 400 = 125 :=
by {
  -- Given conditions
  have h1 : spaceship_speed 200 = 500 := rfl,
  have h2 : spaceship_speed (200 + 100) = spaceship_speed 200 / 2,
  have h3 : spaceship_speed (200 + 200) = spaceship_speed (200 + 100) / 2,
  -- Calculations based on conditions
  simp [spaceship_speed] at *,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end spaceship_speed_400_l643_643129


namespace correct_option_l643_643729

-- Define the various expressions as propositions
def optionA (a : ℕ) : Prop := a^2 * a^5 = a^10
def optionB (a : ℕ) : Prop := (3 * a^3)^2 = 6 * a^6
def optionC (a : ℕ) : Prop := 4 * a - 3 * a = a
def optionD (a : ℕ) : Prop := (a + 1)^2 = a^2 + 1

-- Prove that optionC is correct
theorem correct_option (a : ℕ) : optionC a :=
by {
  rw optionC, -- Expose the content of optionC
  sorry -- Proof of 4 * a - 3 * a = a
}

end correct_option_l643_643729


namespace average_reciprocal_summation_l643_643096

theorem average_reciprocal_summation
  (a : Nat → ℝ) (b : Nat → ℝ)
  (h_avg_reciprocal : ∀ n, (↑n / (∑ i in Finset.range n, a (i + 1)) = 1 / (2 * ↑n + 1)))
  (h_b_def : ∀ n, b n = (a n + 1) / 4) :
  (∑ i in Finset.range 10, (1 / (b (i + 1) * b (i + 2)))) = 10 / 11 :=
by
  sorry

end average_reciprocal_summation_l643_643096


namespace probability_A_at_edge_l643_643368

theorem probability_A_at_edge (n m : ℕ) (h₀ : n = 6) (h₁ : m = 4) : 
  (m : ℚ) / n = 2 / 3 :=
by
  rw [h₀, h₁]
  norm_num
  sorry

end probability_A_at_edge_l643_643368


namespace max_min_difference_of_m_l643_643151

theorem max_min_difference_of_m (m : Real) (A B P : Real × Real) 
  (h_m_positive : m > 0)
  (h_Circle : ∀ P, P ∈ circle (0, 0) 1) 
  (h_dot_product : (A.1 + P.1) * (P.1 - B.1) + (P.2 - 2) * (P.2 - 2) = 0) 
  : (3 - 1 = 2) :=
sorry

end max_min_difference_of_m_l643_643151


namespace dot_path_length_l643_643770

noncomputable def cube_edge_length : ℝ := 2
noncomputable def face_diagonal : ℝ := real.sqrt (cube_edge_length ^ 2 + cube_edge_length ^ 2)
noncomputable def dot_radius : ℝ := face_diagonal / 2 
noncomputable def quarter_turn_arc : ℝ := (dot_radius * real.pi) / 2

theorem dot_path_length :
  (4 * quarter_turn_arc) = 2 * real.sqrt 2 * real.pi :=
by
  rw [← mul_assoc, mul_div_cancel_left, ← mul_assoc]
  sorry

end dot_path_length_l643_643770


namespace first_division_meiosis_l643_643667

/-- In the first division of meiosis, homologous chromosomes separate,
and centromeres do not split. --/
theorem first_division_meiosis :
  (∀ (homologous_chromosomes separate : Prop)
     (centromeres split : Prop), homologous_chromosomes → ¬split → 
     (homologous_chromosomes separate ∧ ¬centromeres split))
:=
begin
  intros, -- assumptions key here
  split,
  { exact homologous_chromosomes },
  { exact λ h, split h }
end

end first_division_meiosis_l643_643667


namespace paths_from_A_to_D_l643_643493

theorem paths_from_A_to_D : 
  let P_AB := 2 and P_BC := 2 and P_CD := 3 in
  let P_ACD := (1 * P_CD) in
  let P_total := (P_AB * P_BC * P_CD) + P_ACD in
  P_total = 15 :=
by
  let P_AB := 2
  let P_BC := 2
  let P_CD := 3
  let P_ACD := 1 * P_CD
  let P_total := P_AB * P_BC * P_CD + P_ACD
  have h : P_total = 15 := sorry
  exact h

end paths_from_A_to_D_l643_643493


namespace solve_y_l643_643310

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l643_643310


namespace quadratic_eq_two_distinct_real_roots_l643_643581

theorem quadratic_eq_two_distinct_real_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 2 * x₁ + a = 0) ∧ (x₂^2 - 2 * x₂ + a = 0)) ↔ a < 1 :=
by
  sorry

end quadratic_eq_two_distinct_real_roots_l643_643581


namespace moles_NaNO3_formed_l643_643114

-- Let's define our basic entities in Lean for the given problem
def AgNO3 := "AgNO3"
def NaOH := "NaOH"
def AgOH := "AgOH"
def NaNO3 := "NaNO3"

-- Condition: balanced chemical equation
def balanced_eq (x y z w : String) := x = AgNO3 ∧ y = NaOH ∧ z = AgOH ∧ w = NaNO3

-- Condition: 1 mole of reactants given
def initial_moles := (agno3 : ℕ) = 1 ∧ (naoh : ℕ) = 1

-- Condition: reaction goes to completion, no side reactions
def reaction_complete := true

-- The Lean 4 statement to prove the number of moles of NaNO3 formed
theorem moles_NaNO3_formed (agno3_moles naoh_moles : ℕ) 
  (balanced : balanced_eq AgNO3 NaOH AgOH NaNO3) 
  (initial : initial_moles agno3_moles naoh_moles) 
  (complete : reaction_complete) : 
  (n_NaNO3 : ℕ) = 1 := sorry

end moles_NaNO3_formed_l643_643114


namespace fraction_students_say_dislike_but_actually_like_is_25_percent_l643_643807

variable (total_students : Nat) (students_like_dancing : Nat) (students_dislike_dancing : Nat) 
         (students_like_dancing_but_say_dislike : Nat) (students_dislike_dancing_and_say_dislike : Nat) 
         (total_say_dislike : Nat)

def fraction_of_students_who_say_dislike_but_actually_like (total_students students_like_dancing students_dislike_dancing 
         students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike : Nat) : Nat :=
    (students_like_dancing_but_say_dislike * 100) / total_say_dislike

theorem fraction_students_say_dislike_but_actually_like_is_25_percent
  (h1 : total_students = 100)
  (h2 : students_like_dancing = 60)
  (h3 : students_dislike_dancing = 40)
  (h4 : students_like_dancing_but_say_dislike = 12)
  (h5 : students_dislike_dancing_and_say_dislike = 36)
  (h6 : total_say_dislike = 48) :
  fraction_of_students_who_say_dislike_but_actually_like total_students students_like_dancing students_dislike_dancing 
    students_like_dancing_but_say_dislike students_dislike_dancing_and_say_dislike total_say_dislike = 25 :=
by sorry

end fraction_students_say_dislike_but_actually_like_is_25_percent_l643_643807


namespace problem_l643_643869

variable {R : Type*} [OrderedRing R]

-- Define the necessary variables and assumptions
variables (f : R → R) (f' : R → R)
variables (x x₁ x₂ : R)

-- Assume the conditions of the problem
axiom f_deriv : ∀ x, has_deriv_at f (f' x) x
axiom x_in_R : x ∈ (set.univ : set R)
axiom x1_in_R : x₁ ∈ (set.univ : set R)
axiom x2_in_R : x₂ ∈ (set.univ : set R)
axiom x1_lt_x2 : x₁ < x₂
axiom ineq : ∀ x, x * f' x > -f x

-- Define the goal
theorem problem : x₁ * f x₁ < x₂ * f x₂ :=
sorry

end problem_l643_643869


namespace sum_c_d_l643_643945

theorem sum_c_d (c d : ℕ) (h : (∏ n in Finset.range (c - 5), (n + 5 + 1) / (n + 5)) = 15) : c + d = 89 :=
sorry

end sum_c_d_l643_643945


namespace area_of_LOM_is_3_l643_643965

-- Definition of a scalene triangle with given angle properties and area
structure ScaleneTriangle := 
  (A B C : Point)
  (angle_A angle_B angle_C : ℝ)
  (area : ℝ)
  (is_scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (angles_sum : angle_A + angle_B + angle_C = 180)
  (angle_relation_1 : angle_A = angle_B - angle_C)
  (angle_relation_2 : angle_B = 2 * angle_C)
  (area_value : area = 2)

-- Definition of the triangle formed by angle bisectors intersecting the circumcircle
structure BisectorIntersectionTriangle := 
  (L O M : Point)
  (original_triangle : ScaleneTriangle)
  (angle_bisector_intersection : 
    -- Assume function bisect_angle which defines that L, O, M are points on the circumcircle intersecting the angle bisectors
    bisect_angle A = L ∧ bisect_angle B = O ∧ bisect_angle C = M)

-- Problem Statement: Prove that the area of triangle LOM is approximately 3, rounded to the nearest whole number
theorem area_of_LOM_is_3 (T : ScaleneTriangle) (BIT : BisectorIntersectionTriangle)
  (H : BIT.original_triangle = T) : 
  abs (area (triangle BIT.L BIT.O BIT.M) - 3) < 1 :=
begin
  -- Definitions and conditions logic implementation would go here
  sorry
end

end area_of_LOM_is_3_l643_643965


namespace box_volume_correct_l643_643045

-- Define the dimensions of the original sheet
def length_original : ℝ := 48
def width_original : ℝ := 36

-- Define the side length of the squares cut from each corner
def side_length_cut : ℝ := 4

-- Define the new dimensions after cutting the squares
def new_length : ℝ := length_original - 2 * side_length_cut
def new_width : ℝ := width_original - 2 * side_length_cut

-- Define the height of the box
def height_box : ℝ := side_length_cut

-- Define the expected volume of the box
def volume_box_expected : ℝ := 4480

-- Prove that the calculated volume is equal to the expected volume
theorem box_volume_correct :
  new_length * new_width * height_box = volume_box_expected := by
  sorry

end box_volume_correct_l643_643045


namespace distribution_of_balls_l643_643215

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end distribution_of_balls_l643_643215


namespace minimize_expression_l643_643813

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 :=
sorry

end minimize_expression_l643_643813


namespace volume_ratio_l643_643725

noncomputable def cube_volume (s : ℝ) : ℝ := s^3

noncomputable def sphere_volume (d : ℝ) : ℝ := (4/3) * Real.pi * (d/2)^3

theorem volume_ratio (s1 s2 : ℝ) (h1 : s1 = 4) (h2 : s2 = 3) :
  (cube_volume s1) / (sphere_volume (2 * s2)) = 16 / (9 * Real.pi) :=
by
  have hs1 : cube_volume s1 = 64 := by
    rw [h1, cube_volume]
    norm_num
  have hd : 2 * s2 = 6 := by
    rw [h2]
    norm_num
  have hs2 : sphere_volume 6 = 36 * Real.pi := by
    rw [sphere_volume]
    norm_num
    ring
  rw [hs1, hs2]
  field_simp
  norm_num
  ring
  sorry

end volume_ratio_l643_643725


namespace correct_propositions_count_l643_643700

-- Define the propositions
def prop1 (m : ℝ) : Prop :=
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0

def prop2 (E F G H : Type) : Prop :=
  ¬(∃ P : Type, (∀ e f g h : P, e = E → f = F → g = G → h = H → (∀ a b : P, a ≠ b → ¬∃ q : P, q ≠ a ∧ q ≠ b)) ∧
     (∀ x y z w : P, x = E → y = F → z = G → w = H → ¬(∃ e f : P, e = x → f = y → e ≠ f ∧ (¬∃ g h : P, g ≠ e ∧ g ≠ f ∧ h ≠ g ∧ h ≠ e) ∧ ¬(e = g ∧ f = h)))))

def prop3 (a : ℝ) : Prop :=
  (∀ x : ℝ, |x + 1| + |x - 1| ≥ a) → a < 2

def prop4 (m : ℝ) : Prop :=
  (0 < m ∧ m < 1) ↔ (∃ x y : ℝ, mx^2 + (m - 1)y^2 = 1)

-- Count the number of correct propositions
def count_correct_props (m : ℝ) (E F G H : Type) (a : ℝ) : ℕ :=
  (if prop1 m then 1 else 0) +
  (if prop2 E F G H then 1 else 0) +
  (if prop3 a then 1 else 0) +
  (if prop4 m then 1 else 0)

-- The theorem stating the number of correct propositions
theorem correct_propositions_count (m : ℝ) (E F G H : Type) (a : ℝ) :
  count_correct_props m E F G H a = 2 :=
sorry

end correct_propositions_count_l643_643700


namespace orthocenter_properties_l643_643263

open euclidean_geometry

variables {V : Type*} [inner_product_space ℝ V]

def orthocenter (A B C : V) : V := 
  sorry -- Details of computing the orthocenter.

def foot_of_altitude (A B C : V) : V := 
  sorry -- Details of computing the foot of the altitude from vertex A to BC.

theorem orthocenter_properties (A B C : V) (H_A H_B H_C : V)
  (H := orthocenter A B C) 
  (P := orthocenter A H_B H_C) 
  (Q := orthocenter C H_A H_B)
  (H_A = foot_of_altitude A B C)
  (H_B = foot_of_altitude B A C)
  (H_C = foot_of_altitude C A B) :
  dist P Q = dist H_C H_A :=
begin
  sorry
end

end orthocenter_properties_l643_643263


namespace triangle_area_l643_643952

section
variables {A B C D E F : Type} [AffineSpace ℝ A]

-- Conditions
variables (midpoint_D : D = midpoint B C) -- Point D is the midpoint of side BC
variables (ratio_E : ∃ t : ℝ, E = affineCombination A C t ∧ t = 1 / 4) -- AE : EC = 1 : 3
variables (ratio_F : ∃ s : ℝ, F = affineCombination A D s ∧ s = 2 / 3) -- AF : FD = 2 : 1
variables (area_DEF : area DEF = 25) -- Area of triangle DEF is 25

-- Statement to prove
theorem triangle_area :
  ∀ A B C D E F,
    (D = midpoint B C) →
    (∃ t : ℝ, E = affineCombination A C t ∧ t = 1 / 4) →
    (∃ s : ℝ, F = affineCombination A D s ∧ s = 2 / 3) →
    (area DEF = 25) →
    (area ABC = 600) :=
by
  intros
  sorry
end

end triangle_area_l643_643952


namespace square_area_from_triangle_l643_643695

theorem square_area_from_triangle
  (ABCD : Square) 
  (PQR : Triangle) 
  (h1 : is_divided_into_6_isosceles_right_triangles ABCD)
  (h2 : is_right_isosceles_triangle PQR)
  (h3 : area PQR = 2) :
  area ABCD = 64 :=
sorry

end square_area_from_triangle_l643_643695


namespace divide_money_equally_l643_643015

-- Length of the road built by companies A, B, and total length of the road
def length_A : ℕ := 6
def length_B : ℕ := 10
def total_length : ℕ := 16

-- Money contributed by company C
def money_C : ℕ := 16 * 10^6

-- The equal contribution each company should finance
def equal_contribution := total_length / 3

-- Deviations from the expected length for firms A and B
def deviation_A := length_A - (total_length / 3)
def deviation_B := length_B - (total_length / 3)

-- The ratio based on the deviations to divide the money
def ratio_A := deviation_A * (total_length / (deviation_A + deviation_B))
def ratio_B := deviation_B * (total_length / (deviation_A + deviation_B))

-- The amount of money firms A and B should receive, respectively
def money_A := money_C * ratio_A / total_length
def money_B := money_C * ratio_B / total_length

-- Theorem statement
theorem divide_money_equally : money_A = 2 * 10^6 ∧ money_B = 14 * 10^6 :=
by 
  sorry

end divide_money_equally_l643_643015


namespace max_distance_to_line_l643_643236

-- Define the original curve C1 
def C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the transformed curve C2 by stretching x and y coordinates
def C2 (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the line l given by x + y = 4√5
def l (x y : ℝ) : Prop := x + y = 4 * Real.sqrt 5

-- Define the distance of a point on C2 to the line l
def distance_to_line (θ : ℝ) : ℝ :=
  let (x', y') := C2 θ
  Real.abs (x' + y' - 4 * Real.sqrt 5) / Real.sqrt 2

-- State the claim about maximum distance from any point on C2 to the line l
theorem max_distance_to_line : ∃ θ : ℝ, distance_to_line θ = 5 * Real.sqrt 10 / 2 :=
  sorry

end max_distance_to_line_l643_643236


namespace lemons_to_make_profit_l643_643773

theorem lemons_to_make_profit:
  ∃ n : ℕ, n = 286 ∧ 
  let cost_price_per_lemon := 15 / 4,
      selling_price_per_lemon := 25 / 6,
      profit_per_lemon := selling_price_per_lemon - cost_price_per_lemon in
  (120 / profit_per_lemon) + 1 ≤ n ∧ n ≤ (120 / profit_per_lemon) + 1 :=
by
  let cost_price_per_lemon := 15 / 4
  let selling_price_per_lemon := 25 / 6
  let profit_per_lemon := selling_price_per_lemon - cost_price_per_lemon
  have h : Real.toRat (120 : ℝ) / profit_per_lemon ≈ 285.71
  { sorry }
  exact ⟨286, by norm_num⟩, by norm_num⟩

end lemons_to_make_profit_l643_643773


namespace sum_of_real_and_imag_l643_643885

theorem sum_of_real_and_imag (z : ℂ) (h : z = (2 + ↑ℂ.i) / (2 - ↑ℂ.i) - ↑ℂ.i) : 
  complex.re z + complex.im z = 2 / 5 :=
begin
  sorry -- Proof to be filled in
end

end sum_of_real_and_imag_l643_643885


namespace probability_white_first_red_second_l643_643759

theorem probability_white_first_red_second :
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let prob_white_first := white_marbles / total_marbles
  let prob_red_second_given_white_first := red_marbles / (total_marbles - 1)
  let prob_combined := prob_white_first * prob_red_second_given_white_first
  prob_combined = 4 / 15 :=
by
  sorry

end probability_white_first_red_second_l643_643759


namespace factors_of_polynomial_l643_643000

theorem factors_of_polynomial :
  ∃ factors : Finset (Polynomial ℤ), 
    ( (Polynomial.prod factors) = (Polynomial.prod {X} * Polynomial.prod {X - 1} * Polynomial.prod {X^2 + X + 1} * Polynomial.prod {X^6 + X^3 + 1}) 
    ∧ factors.card = 4 ) :=
sorry

end factors_of_polynomial_l643_643000


namespace point_not_in_plane_l643_643143

def is_in_plane (p0 : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (x0, y0, z0) := p0
  let (nx, ny, nz) := n
  let (x, y, z) := p
  (nx * (x - x0) + ny * (y - y0) + nz * (z - z0)) = 0

theorem point_not_in_plane :
  ¬ is_in_plane (1, 2, 3) (1, 1, 1) (-2, 5, 4) :=
by
  sorry

end point_not_in_plane_l643_643143


namespace find_a3_l643_643512

variable {α : Type} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → α) (h : geometric_sequence a) (h1 : a 0 * a 4 = 16) :
  a 2 = 4 ∨ a 2 = -4 :=
by
  sorry

end find_a3_l643_643512


namespace problem1_problem2_l643_643820

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l643_643820


namespace determine_P_l643_643349

-- Given conditions:
variables (P T R : ℝ)
axiom simple_interest : 88 = (P * T * R) / 100
axiom true_discount : 80 = (88 * P) / (P + 88 * T)

-- Problem statement: Determine that we need additional information to solve for P
theorem determine_P (P T R : ℝ) 
  (h1 : 88 = (P * T * R) / 100)
  (h2 : 80 = (88 * P) / (P + 88 * T)) : 
  ∃ T R, additional_information_required_to_solve_for_P :=
sorry

end determine_P_l643_643349


namespace susan_pay_missed_l643_643661

noncomputable def daily_pay : ℝ := 15 * 8

def total_workdays_in_two_weeks : ℕ := 2 * 5
def paid_vacation_days : ℕ := 6
def unpaid_vacation_days : ℕ := total_workdays_in_two_weeks - paid_vacation_days

noncomputable def pay_missed : ℝ := daily_pay * unpaid_vacation_days

theorem susan_pay_missed : pay_missed = 480 := by
  have calc1 : total_workdays_in_two_weeks = 10 := by norm_num
  have calc2 : unpaid_vacation_days = 4 := by norm_num
  have calc3 : daily_pay = 120 := by norm_num
  have calc4 : pay_missed = daily_pay * unpaid_vacation_days := rfl
  rw [calc3, calc2]
  norm_num
  sorry

end susan_pay_missed_l643_643661


namespace solve_for_y_l643_643320

theorem solve_for_y (y : ℝ) (h : ∛(5 - 2 / y) = -3) : y = 1/16 :=
sorry

end solve_for_y_l643_643320


namespace candy_left_proof_l643_643402

def candy_left (d_candy : ℕ) (s_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  d_candy + s_candy - eaten_candy

theorem candy_left_proof :
  candy_left 32 42 35 = 39 :=
by
  sorry

end candy_left_proof_l643_643402


namespace total_steps_five_days_l643_643644

def steps_monday : ℕ := 150 + 170
def steps_tuesday : ℕ := 140 + 170
def steps_wednesday : ℕ := 160 + 210 + 25
def steps_thursday : ℕ := 150 + 140 + 30 + 15
def steps_friday : ℕ := 180 + 200 + 20

theorem total_steps_five_days :
  steps_monday + steps_tuesday + steps_wednesday + steps_thursday + steps_friday = 1760 :=
by
  have h1 : steps_monday = 320 := rfl
  have h2 : steps_tuesday = 310 := rfl
  have h3 : steps_wednesday = 395 := rfl
  have h4 : steps_thursday = 335 := rfl
  have h5 : steps_friday = 400 := rfl
  show 320 + 310 + 395 + 335 + 400 = 1760
  sorry

end total_steps_five_days_l643_643644


namespace M_eq_N_l643_643746

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_eq_N : M = N := 
by 
  sorry

end M_eq_N_l643_643746


namespace second_term_of_new_ratio_l643_643781

theorem second_term_of_new_ratio (x : ℕ) (x_eq_5 : x = 5) (first_term : ℕ) (first_term_eq_3 : first_term = 3) : 
  let second_term := 11 + x 
  in second_term = 16 :=
by
  have h1 : x = 5 := x_eq_5,
  have h2 : second_term = 11 + x := rfl,
  rw [h1] at h2,
  exact h2

end second_term_of_new_ratio_l643_643781


namespace air_conditioner_usage_l643_643800

-- Define the given data and the theorem to be proven
theorem air_conditioner_usage (h : ℝ) (rate : ℝ) (days : ℝ) (total_consumption : ℝ) :
  rate = 0.9 → days = 5 → total_consumption = 27 → (days * h * rate = total_consumption) → h = 6 :=
by
  intros hr dr tc h_eq
  sorry

end air_conditioner_usage_l643_643800


namespace x_plus_p_eq_2p_plus_3_l643_643940

theorem x_plus_p_eq_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 := by
  sorry

end x_plus_p_eq_2p_plus_3_l643_643940


namespace sum_of_roots_eq_zero_l643_643322

theorem sum_of_roots_eq_zero {x : ℝ} (h : |x|^2 + |x| - 12 = 0) : x = 3 ∨ x = -3 :=
begin
  have h1 : ∃ y : ℝ, |x| = y ∧ y^2 + y - 12 = 0, from sorry,
  cases h1 with y hy,
  have := λ h, y^2 + y - 12 = (y - 3) * (y + 4), from sorry,
  cases hy.right with h2 h3,
  use [],
  split,
  { sorry, }
end

end sum_of_roots_eq_zero_l643_643322


namespace min_cos_C_l643_643225

theorem min_cos_C (A B C : ℝ) (h : sin A + sqrt 2 * sin B = 2 * sin C) : 
  cos C ≥ (sqrt 6 - sqrt 2) / 4 :=
sorry

end min_cos_C_l643_643225


namespace prove_question_1_prove_question_2_l643_643836

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  x^2 / 4 + y^2 / 3 = 1
  
noncomputable def line_eq (x y : ℝ) : Prop := 
  3 * x + 2 * y + 2 * real.sqrt 7 - 2 = 0

def question_1 (a b : ℝ) (h : a > b ∧ b > 0) : Prop := 
  ellipse_eq = (λ x y, x^2 / 4 + y^2 / 3 = 1)

def question_2 (u v : ℝ) (h : a > b ∧ b > 0) : Prop :=
  line_eq = (λ x y, 3 * x + 2 * y + 2 * real.sqrt 7 - 2 = 0)

theorem prove_question_1 (a b : ℝ) (h : a > b ∧ b > 0) : question_1 a b h := 
  sorry

theorem prove_question_2 (a b u v : ℝ) (h : a > b ∧ b > 0) : question_2 u v h := 
  sorry

end prove_question_1_prove_question_2_l643_643836


namespace perpendicular_lines_give_a_is_half_l643_643923

-- Define the lines l₁ and l₂ with the given conditions
def line_l1 (a : ℝ) : ℝ × ℝ → ℝ := λ p, 4 * a * p.1 + p.2 - 1
def line_l2 (a : ℝ) : ℝ × ℝ → ℝ := λ p, (a - 1) * p.1 + p.2 + 1

-- Define the perpendicular condition
def lines_perpendicular (a : ℝ) : Prop :=
  (-4 * a) * (-(a - 1)) = -1

-- The theorem stating the required proof
theorem perpendicular_lines_give_a_is_half :
  ∀ (a : ℝ), lines_perpendicular a → a = 1/2 := 
by {
  intro a,
  intro h,
  sorry
}

end perpendicular_lines_give_a_is_half_l643_643923


namespace range_of_m_l643_643156

variable (m : ℝ)

def p : Prop := m + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬ (p m ∧ q m)) : m ≤ -2 ∨ m > -1 := 
by
  sorry

end range_of_m_l643_643156


namespace sqrt_sum_inequality_l643_643921

variable (a b c d : ℝ)

theorem sqrt_sum_inequality
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : a + d = b + c) :
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c :=
by
  sorry

end sqrt_sum_inequality_l643_643921


namespace total_students_correct_l643_643072

noncomputable def num_roman_numerals : ℕ := 7
noncomputable def sketches_per_numeral : ℕ := 5
noncomputable def total_students : ℕ := 35

theorem total_students_correct : num_roman_numerals * sketches_per_numeral = total_students := by
  sorry

end total_students_correct_l643_643072


namespace decreasing_interval_l643_643186

noncomputable def f (x a b : ℝ) : ℝ := x^2 * (a * x + b)

theorem decreasing_interval (a b : ℝ) (h1 : f x a b = x^2 * (a * x + b)) 
  (h2 : ∃ x : ℝ, x = 2 ∧ (derivative (f x a b)).eval 2 = 0) 
  (h3 : tangent_line_parallel : (derivative (f x a b)).eval 1 = -3) : 
  ∃ a b : ℝ, a = 1 ∧ b = -3 → ∀ x : ℝ, 0 < x ∧ x < 2 → (derivative (f x a b)).eval x < 0 := 
by
  sorry

end decreasing_interval_l643_643186


namespace largest_value_l643_643217

-- Definition: Given the condition of a quadratic equation
def equation (a : ℚ) : Prop :=
  8 * a^2 + 6 * a + 2 = 0

-- Theorem: Prove the largest value of 3a + 2 is 5/4 given the condition
theorem largest_value (a : ℚ) (h : equation a) : 
  ∃ m, ∀ b, equation b → (3 * b + 2 ≤ m) ∧ (m = 5 / 4) :=
by
  sorry

end largest_value_l643_643217


namespace rainbow_nerds_total_l643_643231

theorem rainbow_nerds_total
  (purple yellow green red blue : ℕ)
  (h1 : purple = 10)
  (h2 : yellow = purple + 4)
  (h3 : green = yellow - 2)
  (h4 : red = 3 * green)
  (h5 : blue = red / 2) :
  (purple + yellow + green + red + blue = 90) :=
by
  sorry

end rainbow_nerds_total_l643_643231


namespace chemistry_club_officer_selection_l643_643629

-- Defining the main theorem to prove the number of ways to select officers
theorem chemistry_club_officer_selection :
  let members := 25
  let officer_positions := ["president", "secretary", "treasurer"]
  let alice := "Alice"
  let bob := "Bob"
  let ronald := "Ronald"
  -- Counting the number of valid ways to select officers under given conditions
  -- 1. Alice and Bob are either both officers or none is an officer
  -- 2. Ronald is an officer only if either Alice or Bob (or both) are officers
  counts : 
    -- Scenario 1: Neither Alice nor Bob are officers, thus Ronald is not an officer
    let remaining_members_scenario1 := members - 3  -- exclude Alice, Bob, Ronald
    let choices_scenario1 := remaining_members_scenario1 * (remaining_members_scenario1 - 1) * (remaining_members_scenario1 - 2)
    -- Scenario 2: Both Alice and Bob are officers, Ronald can be the third officer
    let choices_scenario2 := 3 * 2 * 1 -- choosing positions for Alice, Bob, and Ronald
    choices_scenario1 + choices_scenario2 = 9246 := 
begin
  -- Proof goes here
  sorry -- Placeholder for the actual proof
end

end chemistry_club_officer_selection_l643_643629


namespace standard_01_sequences_count_m4_l643_643023

-- Definition of a standard 01 sequence problem for m = 4
def standard_01_sequence (seq : List Bool) (m : ℕ) : Prop :=
  seq.length = 2 * m ∧
  seq.count (λ b => b = false) = m ∧ 
  seq.count (λ b => b = true) = m ∧
  (∀ k, k ≤ 2 * m → (seq.take k).count (λ b => b = false) ≥ (seq.take k).count (λ b => b = true))

-- Prove that the number of standard 01 sequences for m = 4 is 14
theorem standard_01_sequences_count_m4 : 
  (∃ seqs, seqs.length = 14 ∧ ∀ seq ∈ seqs, standard_01_sequence seq 4) := 
sorry

end standard_01_sequences_count_m4_l643_643023


namespace sum_of_absolute_values_of_coeffs_l643_643613

theorem sum_of_absolute_values_of_coeffs :
  let f := (2 * x - 1) ^ 6
  let a := [64, -192, 240, -160, 60, -12, 1]
  ∑ k in (Finset.range 7), |a[k]| = 729 :=
by
  sorry

end sum_of_absolute_values_of_coeffs_l643_643613


namespace find_k_values_l643_643601

-- Define the conditions as given in the mathematical problem
def is_circle_intersection_one_point (k : ℝ) : Prop :=
  ∀ (z : ℂ), abs (z - 3) = 2 * abs (z + 3) → abs z = k

-- The theorem stating the possible values of k
theorem find_k_values : {k : ℝ | is_circle_intersection_one_point k} = {1, 9} :=
by
  sorry  -- Proof is omitted

end find_k_values_l643_643601


namespace can_transform_2020_2021_can_transform_1000_2021_l643_643755

-- Define the condition of adding or subtracting one of the digits of a number
def can_transform (start finish : ℕ) : Prop :=
  ∃ seq : List ℕ, seq.head = start ∧ seq.reverse.head = finish ∧ (∀ n ∈ seq, 
    (∃ d ∈ (n.digits), n + d ∈ seq ∨ n - d ∈ seq))

theorem can_transform_2020_2021: can_transform 2020 2021 := 
sorry

theorem can_transform_1000_2021: can_transform 1000 2021 := 
sorry

end can_transform_2020_2021_can_transform_1000_2021_l643_643755


namespace intersection_point_B_altitude_from_A_l643_643177

noncomputable def intersection_point (line1 line2 : ℝ → ℝ → Prop) : ℝ × ℝ :=
  Classical.choose (Exists.intro (λ (p : ℝ × ℝ), line1 p.1 p.2 ∧ line2 p.1 p.2) sorry)

def line_AB (x y : ℝ) : Prop := 3 * x + 4 * y + 12 = 0
def line_BC (x y : ℝ) : Prop := 4 * x - 3 * y + 16 = 0
def line_CA (x y : ℝ) : Prop := 2 * x + y - 2 = 0

theorem intersection_point_B : intersection_point line_AB line_BC = (-4, 0) :=
sorry

def line_altitude_from_A (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem altitude_from_A : ∀ (x y : ℝ), line_altitude_from_A x y ↔ ((2 * x + y - 2 = 0) ∧ (3 * x + 4 * y + 12 ≠ 0)) :=
sorry

end intersection_point_B_altitude_from_A_l643_643177


namespace f_is_odd_part1_part2_l643_643183

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x + 1) / (4^x + a)

axiom odd_f_condition (a x : ℝ) : f a (-x) = - f a x

theorem f_is_odd : ∀ a, f a = (λ x, (4^x + 1) / (4^x + a)) →
  odd_f_condition a x → a = -1 :=
begin
  intros a h odd_cond,
  sorry
end

theorem part1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : f (-1) x > 5/3 :=
begin
  sorry
end

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (Real.log 2 (x / 2)) * (Real.log 2 (x / 4)) + m

theorem part2 (m : ℝ) (x1 : ℝ) (h1 : 2 ≤ x1) (h2 : x1 ≤ 8) :
  ∃ x2 ∈ Icc (0 : ℝ) 1, g m x1 = f (-1) x2 →
  m ≥ 23/12 :=
begin
  sorry
end

end f_is_odd_part1_part2_l643_643183


namespace part_a_part_b_l643_643162

variable (p : ℕ)
variable (h1 : prime p)
variable (h2 : p > 3)

theorem part_a : (p + 1) % 4 = 0 ∨ (p - 1) % 4 = 0 :=
sorry

theorem part_b : ¬ ((p + 1) % 5 = 0 ∨ (p - 1) % 5 = 0) :=
sorry

end part_a_part_b_l643_643162


namespace solve_trig_equations_l643_643106

theorem solve_trig_equations
  (α β : ℝ)
  (h1 : sin α + sin β = - (21 / 65))
  (h2 : cos α + cos β = - (27 / 65))
  (h3 : (5 * π / 2) < α ∧ α < 3 * π)
  (h4 : -(π / 2) < β ∧ β < 0) :
  sin ((α + β) / 2) = - (7 / sqrt 130) ∧ cos ((α + β) / 2) = - (9 / sqrt 130) :=
  sorry

end solve_trig_equations_l643_643106


namespace problem_statement_l643_643276

theorem problem_statement (x : ℝ)
    (h : sqrt (8 + x) + sqrt (25 - x^2) = 9) :
    (8 + x) * (25 - x^2) = 576 :=
by
  sorry

end problem_statement_l643_643276


namespace no_real_solutions_l643_643200

theorem no_real_solutions (a b c : ℝ) :
  ∀ (x y z : ℝ), (a^2 + b^2 + c^2 + 3 * (x^2 + y^2 + z^2) = 6 ∧ a * x + b * y + c * z = 2) → False :=
by
  intros x y z h,
  let s := a^2 + b^2 + c^2,
  have key := calc
    s * (6 - s) / 3 : sorry, -- Derive this step from the given conditions and inequality
    s * (2 - s) * (6 - s) / 3 >= 12 : sorry, -- From the calculation s (6 - s) ≥ 12
    (-s^2 + 6 * s - 12 : ℝ) ≥ 0 : sorry, -- Simplification
  sorry -- Concluding that there are no real values for a, b, c, x, y, z that satisfy this.

end no_real_solutions_l643_643200


namespace modulus_of_complex_l643_643508

theorem modulus_of_complex (z : ℂ) (h : (1 + complex.I) * z = 2 - complex.I) : 
  complex.abs z = (real.sqrt 10) / 2 :=
by
  sorry

end modulus_of_complex_l643_643508


namespace coeff_of_quadratic_term_eq_neg5_l643_643669

theorem coeff_of_quadratic_term_eq_neg5 (a b c : ℝ) (h_eq : -5 * x^2 + 5 * x + 6 = a * x^2 + b * x + c) :
  a = -5 :=
by
  sorry

end coeff_of_quadratic_term_eq_neg5_l643_643669


namespace volume_equality_height_increase_ratio_l643_643092

noncomputable def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

-- Given conditions
def r1 : ℝ := 4
def r2 : ℝ := 8
def h1 (h2 : ℝ) : ℝ := 4 * h2
def marble_radius : ℝ := 2

-- Initial volumes
def V1 (h1 : ℝ) : ℝ := volume_cone r1 h1
def V2 (h2 : ℝ) : ℝ := volume_cone r2 h2

-- Ensure the volumes are the same initially
theorem volume_equality (h2 : ℝ) : V1 (h1 h2) = V2 h2 :=
  by simp [V1, V2, volume_cone, r1, r2, h1, h2]; ring_nf

-- Volume of the submerged marble
def marble_volume : ℝ := volume_sphere marble_radius

-- New heights after submerging the marble
def h1' (h1 : ℝ) : ℝ := h1 + 2
def h2' (h2 : ℝ) : ℝ := h2 + 0.5

-- Prove the ratio of the rise in water levels
theorem height_increase_ratio (h2 : ℝ) : (2 : ℝ) / (0.5 : ℝ) = 4 :=
  by norm_num

example (h2 : ℝ) : (h1' (h1 h2) - h1 h2) / (h2' h2 - h2) = 4 :=
  by simp [h1', h1, h2', h2, height_increase_ratio]; norm_num


end volume_equality_height_increase_ratio_l643_643092


namespace count_integers_in_range_l643_643864

theorem count_integers_in_range : 
  {n : ℤ | 20 < n^2 ∧ n^2 < 200}.to_finset.card = 20 :=
sorry

end count_integers_in_range_l643_643864


namespace problem1_problem2_l643_643819

-- Define the variables for the two problems
variables (a b x : ℝ)

-- The first problem
theorem problem1 :
  (a + 2 * b) ^ 2 - 4 * b * (a + b) = a ^ 2 :=
by 
  -- Proof goes here
  sorry

-- The second problem
theorem problem2 :
  ((x ^ 2 - 2 * x) / (x ^ 2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x ^ 2 - 4)) = x + 2 :=
by 
  -- Proof goes here
  sorry

end problem1_problem2_l643_643819


namespace range_of_f_in_interval_area_of_triangle_l643_643953

variable {A B C : ℝ}
variable {a b c : ℝ} (triangle : Triangle)
variable {f : ℝ → ℝ} (f_def : ∀ x, f x = 2 * sin (x - A) * cos x + sin (B + C))
variable {a_eq : a = 7}
variable {sin_sum : sin B + sin C = (13 * Real.sqrt 3) / 14}

-- The vertices of the triangle and sides opposite to them
variable (A B C : ℝ)
variable (a b c : ℝ)

-- Equivalent Proof Problem
theorem range_of_f_in_interval (hA : A = Real.pi / 3) :
  Set.range (λ x : ℝ, f x) \subseteq Set.Icc (-Real.sqrt 3 / 2) 1 := sorry

theorem area_of_triangle (hA : A = Real.pi / 3) :
  area_triangle a b c = 15 * Real.sqrt 3 := sorry

end range_of_f_in_interval_area_of_triangle_l643_643953


namespace minimum_lace_length_l643_643330

-- Conditions
def width : ℝ := 50 -- in mm
def length : ℝ := 80 -- in mm
def num_eyelets : ℕ := 4
def extra_length_knot : ℝ := 200 -- in mm

-- Calculations based on conditions
noncomputable def segment_length : ℝ := length / (num_eyelets - 1)
noncomputable def diagonal_length (width length_segment : ℝ) : ℝ :=
  real.sqrt (width^2 + length_segment^2)
noncomputable def total_diagonal_length (diag_length : ℝ) : ℝ :=
  6 * diag_length
noncomputable def total_length (total_diag_length width extra_length : ℝ) : ℝ :=
  total_diag_length + width + 2 * extra_length

-- Statement to be proved
theorem minimum_lace_length : total_length (total_diagonal_length (diagonal_length width segment_length)) width extra_length_knot = 790 :=
by
  sorry

end minimum_lace_length_l643_643330


namespace find_speed_first_car_l643_643711

noncomputable def speed_first_car (v : ℝ) : Prop :=
  let t := (14 : ℝ) / 3
  let d_total := 490
  let d_second_car := 60 * t
  let d_first_car := v * t
  d_second_car + d_first_car = d_total

theorem find_speed_first_car : ∃ v : ℝ, speed_first_car v ∧ v = 45 :=
by
  sorry

end find_speed_first_car_l643_643711


namespace sum_of_products_le_square_l643_643021

theorem sum_of_products_le_square (n : ℕ) (s : ℝ) (x : ℕ → ℝ) (h : ∀ i, 0 ≤ x i)
  (h_sum : (finset.range n).sum x = s) : 
  (finset.range (n - 1)).sum (λ i, x i * x (i + 1)) ≤ s^2 / 4 :=
sorry

end sum_of_products_le_square_l643_643021


namespace total_price_of_shoes_l643_643249

theorem total_price_of_shoes
  (S J : ℝ) 
  (h1 : 6 * S + 4 * J = 560) 
  (h2 : J = S / 4) :
  6 * S = 480 :=
by 
  -- Begin the proof environment
  sorry -- Placeholder for the actual proof

end total_price_of_shoes_l643_643249


namespace divisible_by_9_l643_643728

theorem divisible_by_9 (k : ℕ) (h : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
sorry

end divisible_by_9_l643_643728


namespace max_value_expression_l643_643357

theorem max_value_expression (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) (h₄ : a + b + c + d ≤ 4) :
  sqrt (4 * sqrt 3) = 4 * real.sqrt cbrt 3 :=
sorry

end max_value_expression_l643_643357


namespace C1_general_form_C2_cartesian_form_find_k_value_l643_643599

-- Given parametric equations of C1 and target general equation
def C1_parametric (θ : ℝ) : ℝ × ℝ := (3 + cos θ, 4 + sin θ)
def C1_general (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Given polar equation of C2 and target Cartesian equation
def C2_polar (ρ θ k : ℝ) : Prop := ρ * (sin θ - k * cos θ) = 3
def C2_cartesian (x y k : ℝ) : Prop := y = k * x + 3

-- Minimum length of the tangent line and target value of k
def min_tangent_length : ℝ := 2 * Real.sqrt 2
def k_value : ℝ := -4 / 3

-- Proof statements
theorem C1_general_form : ∀ θ, ∃ x y, C1_parametric θ = (x, y) → C1_general x y := by
  sorry

theorem C2_cartesian_form : ∀ ρ θ k, ∃ x y, C2_polar ρ θ k → C2_cartesian x y k := by
  sorry

theorem find_k_value : ∀ k, (∃ x y, C2_cartesian x y k) → 
  (∃ (P : ℝ × ℝ), (∀ tangent_length, tangent_length = min_tangent_length) → k = k_value) := by
  sorry

end C1_general_form_C2_cartesian_form_find_k_value_l643_643599


namespace number_of_pens_bought_l643_643955

theorem number_of_pens_bought
  (total_pens : ℕ)
  (defective_pens : ℕ)
  (non_defective_probability : ℚ)
  (h1 : total_pens = 10)
  (h2 : defective_pens = 2)
  (h3 : non_defective_probability = 0.6222222222222222) :
  ∃ n : ℕ, n = 2 ∧
    (let non_defective_pens := total_pens - defective_pens in
     (non_defective_pens / total_pens) * ((non_defective_pens - 1) / (total_pens - 1)) = non_defective_probability) :=
by
  sorry

end number_of_pens_bought_l643_643955


namespace seeds_total_l643_643420

noncomputable def seeds_planted (x : ℕ) (y : ℕ) (z : ℕ) : ℕ :=
x + y + z

theorem seeds_total (x : ℕ) (H1 :  y = 5 * x) (H2 : x + y = 156) (z : ℕ) 
(H3 : z = 4) : seeds_planted x y z = 160 :=
by
  sorry

end seeds_total_l643_643420


namespace shortest_path_pyramid_correct_l643_643693

noncomputable def shortest_path_on_pyramid (b : ℝ) (α : ℝ) : ℝ :=
  if α ≤ 45 then 2 * b * Real.sin (2 * α)
  else 2 * b

theorem shortest_path_pyramid_correct (b : ℝ) (α : ℝ) :
  (α ≤ 45 → shortest_path_on_pyramid b α = 2 * b * Real.sin (2 * α)) ∧
  (α > 45 → shortest_path_on_pyramid b α = 2 * b) :=
by
  split
  case inl =>
    intro h
    unfold shortest_path_on_pyramid
    simp [if_pos h]
    sorry
  case inr =>
    intro h
    unfold shortest_path_on_pyramid
    simp [if_neg h]
    sorry

end shortest_path_pyramid_correct_l643_643693


namespace a_2023_eq_17_l643_643604

def a : ℕ → ℤ
| 1 := 1
| 2 := 4
| 3 := 9
| 4 := 16
| 5 := 25
| n := if (n - 1) % 8 + 1 ≤ 5 then a ((n - 1) % 8 + 1) else
       if (n - 1) % 8 + 1 = 6 then 22 else 
       if (n - 1) % 8 + 1 = 7 then 17 else 10

theorem a_2023_eq_17 : a 2023 = 17 :=
by {
  sorry
}

end a_2023_eq_17_l643_643604


namespace intersect_product_l643_643447

noncomputable def h : ℝ → ℝ :=
  λ x, if x = 0 then 4 else if x = -2 then 4 else 0

theorem intersect_product : ∃ a b : ℝ, h a = h (a + 2) ∧ b = h a ∧ a * b = -8 :=
  by
    use -2, 4
    split
    · simp [h]
    split
    · simp [h]
    · norm_num
    sorry

end intersect_product_l643_643447


namespace min_max_cos_sin_l643_643488

open Real

theorem min_max_cos_sin (p q : ℚ) (hp : 0 < p) (hq : 0 < q) :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 →
    (cos x)^p * (sin x)^q ≥ 0 ∧
    ((cos x)^p * (sin x)^q = 0 ↔ (x = 0 ∨ x = π / 2)) ∧
    ((∀ y, 0 ≤ y ∧ y ≤ π / 2 → (cos y)^p * (sin y)^q ≤ (cos (arctan (sqrt (q / p))))^p * (sin (arctan (sqrt (q / p))))^q) ↔ x = arctan (sqrt (q / p))) := 
sorry

end min_max_cos_sin_l643_643488


namespace vanya_cannot_score_100_in_multiple_exams_l643_643017

variable {r p m : ℕ}
variable (r_initial p_initial m_initial: ℕ)
variable (op_a op_b_r op_b_p op_b_m : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ))

def score_is_valid (scores : ℕ × ℕ × ℕ) : Prop := 
  scores.1 ≤ 100 ∧ scores.2 ≤ 100 ∧ scores.3 ≤ 100

def operation_a : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r + 1, p + 1, m + 1)

def operation_b_r : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r - 3, p + 1, m + 1)

def operation_b_p : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r + 1, p - 3, m + 1)

def operation_b_m : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ)
| (r, p, m) => (r + 1, p + 1, m - 3)

theorem vanya_cannot_score_100_in_multiple_exams 
  (r_initial p_initial m_initial : ℕ) 
  (h1 : r_initial = m_initial - 14) 
  (h2 : p_initial = m_initial - 9) :
  ¬ ∃ (ops : List ((ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ))),
    (let final_scores := ops.foldl (λ (scores : ℕ × ℕ × ℕ) (op: ((ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ))), op scores) (r_initial, p_initial, m_initial) in
    score_is_valid final_scores ∧ (final_scores.1 = 100 ∨ final_scores.2 = 100 ∨ final_scores.3 = 100) ∧
    ((final_scores.1 = 100 ∧ final_scores.2 = 100) ∨ (final_scores.1 = 100 ∧ final_scores.3 = 100) ∨ (final_scores.2 = 100 ∧ final_scores.3 = 100))) :=
by 
  sorry

end vanya_cannot_score_100_in_multiple_exams_l643_643017


namespace football_lineup_l643_643634

theorem football_lineup (team_size : ℕ)
  (strong_offensive : ℕ)
  (quarterback running_back wide_receiver tight_end : ℕ)
  (H1 : team_size = 12)
  (H2 : strong_offensive = 4) 
  (H3 : quarterback + running_back + wide_receiver + tight_end = team_size - 1):
  (strong_offensive) * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4) = 31680 := 
by calc
  (strong_offensive) * (team_size - 1) * (team_size - 2) * (team_size - 3) * (team_size - 4)
      = 4 * 11 * 10 * 9 * 8 : by { rw [H1, H2], norm_num }
  ... = 31680 : by norm_num

end football_lineup_l643_643634


namespace min_vitamins_sold_l643_643287

theorem min_vitamins_sold (n : ℕ) (h1 : n % 11 = 0) (h2 : n % 23 = 0) (h3 : n % 37 = 0) : n = 9361 :=
by
  sorry

end min_vitamins_sold_l643_643287


namespace cousin_age_result_l643_643648

-- Let define the ages
def rick_age : ℕ := 15
def oldest_brother_age : ℕ := 2 * rick_age
def middle_brother_age : ℕ := oldest_brother_age / 3
def smallest_brother_age : ℕ := middle_brother_age / 2
def youngest_brother_age : ℕ := smallest_brother_age - 2
def cousin_age : ℕ := 5 * youngest_brother_age

-- The theorem stating the cousin's age.
theorem cousin_age_result : cousin_age = 15 := by
  sorry

end cousin_age_result_l643_643648


namespace find_phi_l643_643184

-- Define the problem conditions and the correct answer
theorem find_phi (ω : ℝ) (φ : ℝ) (k : ℤ) 
  (hω_pos : ω > 0)
  (hφ_bound : |φ| < (Real.pi / 2))
  (h_period : ∀ x, sin (ω * x + φ) = sin (ω * (x + 4 * Real.pi) + φ))
  (h_symmetry : ∀ x, sin (ω * (x - (2 * Real.pi / 3)) + φ) = sin (ω * (-(x - (2 * Real.pi / 3))) + φ)) :
  φ = (- Real.pi / 6) :=
sorry

end find_phi_l643_643184


namespace point_in_fourth_quadrant_point_bottom_right_of_line_l643_643875

-- let's define the conditions first
def real_part (m : ℝ) : ℝ := m^2 - 8 * m + 15
def imag_part (m : ℝ) : ℝ := m^2 + 3 * m - 28

-- question (1) about the fourth quadrant
theorem point_in_fourth_quadrant (m : ℝ) :
  0 < real_part m ∧ imag_part m < 0 ↔ m ∈ Set.Ioo (-7 : ℝ) 3 :=
sorry

-- question (2) about the bottom-right of the line y = 2x - 40
theorem point_bottom_right_of_line (m : ℝ) :
  imag_part m < 2 * real_part m - 40 ↔ m ∈ Set.Ioo (-∞ : ℝ) 1 ∪ Set.Ioo 18 (∞ : ℝ) :=
sorry

end point_in_fourth_quadrant_point_bottom_right_of_line_l643_643875


namespace susan_missed_pay_l643_643658

noncomputable def daily_pay : ℕ := 15 * 8

noncomputable def total_work_days (weeks: ℕ) : ℕ := weeks * 5

noncomputable def unpaid_vacation_days (total_work_days paid_vacation_days: ℕ) : ℕ := 
  total_work_days - paid_vacation_days

noncomputable def missed_pay (unpaid_days daily_pay: ℕ) : ℕ := 
  unpaid_days * daily_pay

theorem susan_missed_pay :
  let weeks := 2,
      paid_vacation_days := 6 in
  missed_pay (unpaid_vacation_days (total_work_days weeks) paid_vacation_days) daily_pay = 480 :=
by
  sorry

end susan_missed_pay_l643_643658


namespace vertex_quadrant_l643_643984

noncomputable def vertex_in_second_quadrant (f : ℝ → ℝ) : Prop :=
∃ h k : ℝ, f = λ x, -(x + h)^2 + k ∧ h = -1 ∧ k = 2 ∧ h < 0 ∧ k > 0

theorem vertex_quadrant : vertex_in_second_quadrant (λ x, -(x + 1)^2 + 2) := by
  sorry

end vertex_quadrant_l643_643984


namespace find_theta_l643_643640

noncomputable def theta (k n : ℤ) : ℝ := (n / 7) * Real.pi

theorem find_theta
(a1: PointMovesUniformly : ∀ t : ℕ, θ : ℝ, Real.radian)
(a2: StartsFromPositiveHalfXAxis : ∀ (t : ℝ), (A t).x > 0 → A t = origin)
(a3: TurnsThroughAngleθOneMinute : ∀ t : ℝ, (0 < θ ∧ θ < Real.pi))
(a4: ReachesThirdQuadrantTwoMinutes : Point A (2 min) → Quadrant = 3)
(a5: ReturnsOriginalFourteenMinutes : Point A 14 minute = origin) 
: θ = (4 / 7) * Real.pi ∨ θ = (5 / 7) * Real.pi :=
sorry

end find_theta_l643_643640


namespace count_integer_pairs_l643_643203

theorem count_integer_pairs :
  {p : ℤ × ℤ | ∃ a b : ℤ, p = (a, b) ∧ a^2 + b^2 < 9 ∧ a^2 + b^2 < 8 * (a - 2) ∧ a^2 + b^2 < 8 * (b - 2)}.card = 2 :=
sorry

end count_integer_pairs_l643_643203


namespace product_of_distances_on_hyperbola_constant_minimum_distance_to_point_on_hyperbola_l643_643531

-- Part 1
theorem product_of_distances_on_hyperbola_constant
  (P : ℝ × ℝ)
  (hP : P.1 ^ 2 / 5 - P.2 ^ 2 = 1) :
  let d1 := abs (P.1 - sqrt 5 * P.2) / sqrt 6,
      d2 := abs (P.1 + sqrt 5 * P.2) / sqrt 6 in
  d1 * d2 = 5 / 6 :=
sorry

-- Part 2
theorem minimum_distance_to_point_on_hyperbola
  (P : ℝ × ℝ)
  (A : ℝ × ℝ)
  (hA : A = (4, 0))
  (hP : P.1 ^ 2 / 5 - P.2 ^ 2 = 1) :
  let PA := sqrt ((P.1 - A.1) ^ 2 + P.2 ^ 2) in
  PA = sqrt 15 / 3 :=
sorry

end product_of_distances_on_hyperbola_constant_minimum_distance_to_point_on_hyperbola_l643_643531


namespace algebraic_identity_l643_643085

variables {R : Type*} [CommRing R] (a b : R)

theorem algebraic_identity : 2 * (a - b) + 3 * b = 2 * a + b :=
by
  sorry

end algebraic_identity_l643_643085


namespace sine_of_angle_l643_643560

variables (a b : ℝ × ℝ)
variables (θ : ℝ)
-- Given conditions
def given_conditions : Prop := a = (2, 1) ∧ 3 • b + a = (5, 4)

-- Helper definitions for readability
def dot_product := a.1 * b.1 + a.2 * b.2
def norm (v : ℝ × ℝ) := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def cos_theta := dot_product a b / (norm a * norm b)
def sin_theta := real.sqrt (1 - cos_theta a b ^ 2)

theorem sine_of_angle (h : given_conditions a b) : sin_theta a b = real.sqrt 10 / 10 :=
by sorry

end sine_of_angle_l643_643560


namespace tetrahedron_surface_area_l643_643962

theorem tetrahedron_surface_area (S : ℝ) :
  ∀ (T : Type) [has_mem ℝ T] [has_add S T] [has_mul S.sqrt 2 T]
  (midpoint_opposite_edges : ∀ (A B C D M N: T),
    M = (A + B) / 2 ∧ N = (C + D) / 2)
  (orthogonal_projection_area : ∀ (A₁ B₁ C₁ D₁: T),
    (∃ (P : ℝ), 
      A₁ B₁ C₁ D₁ form a quadrilateral ∧ quadrilateral_area = S ∧ quadrilateral_angle = 60)):
  surface_area T = 3 * S * (√2) :=
sorry

end tetrahedron_surface_area_l643_643962


namespace simplify_expression1_simplify_expression2_l643_643816

/-- Proof Problem 1: Simplify the expression (a+2b)^2 - 4b(a+b) -/
theorem simplify_expression1 (a b : ℝ) : 
  (a + 2 * b)^2 - 4 * b * (a + b) = a^2 :=
sorry

/-- Proof Problem 2: Simplify the expression ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) ÷ (x - 1) / (x^2 - 4) -/
theorem simplify_expression2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) : 
  ((x^2 - 2 * x) / (x^2 - 4 * x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 :=
sorry

end simplify_expression1_simplify_expression2_l643_643816


namespace scientific_notation_l643_643373

theorem scientific_notation (n : ℕ) (h : n = 57277000) : n = 5.7277e7 := by
  rw [h]
  norm_num
  exact eq.refl _

end scientific_notation_l643_643373


namespace greatest_number_in_set_l643_643743

theorem greatest_number_in_set (s : Set ℕ) (h1: s = {x : ℕ | ∃ n, x = 55 + 5 * n ∧ 0 ≤ n ∧ n < 45}) : 
  ∃ x ∈ s, ∀ y ∈ s, y ≤ x :=
begin
  use 275,
  split,
  { 
    -- Show that 275 is in the set s
    apply set.mem_def.mpr,
    use 44,
    split,
    { 
      refl, 
    },
    { 
      split,
      {
        linarith,
      },
      {
        linarith,
      }
    }
  },
  { 
    -- Show that for all y in s, y ≤ 275
    intros y hy,
    apply set.mem_def.mp hy,
    rintro ⟨n, rfl⟩,
    linarith,
  }
end

end greatest_number_in_set_l643_643743


namespace probability_heads_twice_in_three_flips_l643_643011

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_heads_twice_in_three_flips :
  let p := 0.5
  let n := 3
  let k := 2
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end probability_heads_twice_in_three_flips_l643_643011


namespace train_crossing_time_l643_643431

def train_length: ℝ := 120
def bridge_length: ℝ := 480
def train_speed: ℝ := 39.27272727272727
def total_distance: ℝ := train_length + bridge_length
def expected_time: ℝ := 600 / 39.27272727272727 -- approximately 15.27

theorem train_crossing_time :
  (train_length + bridge_length) / train_speed ≈ 15.27 :=
by
  sorry

end train_crossing_time_l643_643431


namespace correct_value_of_reema_marks_l643_643778

theorem correct_value_of_reema_marks :
  (∃ x : ℝ,
    let num_students := 35,
        incorrect_avg := 72,
        incorrect_marks_of_reema := 46,
        correct_avg := 71.71,
        incorrect_total_marks := incorrect_avg * num_students,
        correct_total_marks := correct_avg * num_students in
    incorrect_total_marks - incorrect_marks_of_reema + x = correct_total_marks) →
  ∃ x : ℝ, x = 36.85 :=
begin
  assume h,
  cases h with x hx,
  use x,
  exact sorry,
end

end correct_value_of_reema_marks_l643_643778


namespace coordinates_of_2006th_point_l643_643588

def spiral_coords : ℕ → ℤ × ℤ
| 1         => (0, 0)
| 2         => (1, 0)
| 3         => (1, 1)
| 4         => (0, 1)
| 5         => (0, 2)
| 6         => (1, 2)
| 7         => (2, 2)
| 8         => (2, 1)
| 9         => (2, 0)
| (n + 1)   => sorry -- This is just a placeholder; the pattern determination is complex.

theorem coordinates_of_2006th_point : spiral_coords 2006 = (44, 19) :=
by 
  sorry

end coordinates_of_2006th_point_l643_643588


namespace number_of_players_taking_mathematics_l643_643075

theorem number_of_players_taking_mathematics
  (total_players : ℕ)
  (players_physics : ℕ)
  (players_both : ℕ)
  (H1 : total_players = 30)
  (H2 : players_physics = 15)
  (H3 : players_both = 7) :
  ∃ players_math : ℕ, players_math = 22 :=
begin
  sorry
end

end number_of_players_taking_mathematics_l643_643075


namespace previous_year_profit_percentage_l643_643591

variables {R P: ℝ}

theorem previous_year_profit_percentage (h1: R > 0)
    (h2: P = 0.1 * R)
    (h3: 0.7 * P = 0.07 * R) :
    (P / R) * 100 = 10 :=
by
  -- Since we have P = 0.1 * R from the conditions and definitions,
  -- it follows straightforwardly that (P / R) * 100 = 10.
  -- We'll continue the proof from here.
  sorry

end previous_year_profit_percentage_l643_643591


namespace percentage_of_360_equals_126_l643_643388

/-- 
  Prove that (126 / 360) * 100 equals 35.
-/
theorem percentage_of_360_equals_126 : (126 / 360 : ℝ) * 100 = 35 := by
  sorry

end percentage_of_360_equals_126_l643_643388


namespace andrew_number_count_correct_l643_643067

def four_digit_numbers_ending_in_2_divisible_by_9_count : ℕ :=
  489

theorem andrew_number_count_correct :
  ∃ n : ℕ, n = four_digit_numbers_ending_in_2_divisible_by_9_count ∧ n = 489 :=
by {
  use 489,
  split,
  { refl, },
  { refl, }
}

end andrew_number_count_correct_l643_643067


namespace difference_not_divisible_by_1976_l643_643025

theorem difference_not_divisible_by_1976 (A B : ℕ) (hA : 100 ≤ A) (hA' : A < 1000) (hB : 100 ≤ B) (hB' : B < 1000) (h : A ≠ B) :
  ¬ (1976 ∣ (1000 * A + B - (1000 * B + A))) :=
by
  sorry

end difference_not_divisible_by_1976_l643_643025


namespace rays_sine_inequality_l643_643367

theorem rays_sine_inequality
  (α β γ : ℝ)
  (hα : 0 < α)
  (hβ : 0 < β)
  (hγ : 0 < γ)
  (hαβγ1 : α ≤ β)
  (hαβγ2 : β ≤ γ)
  (h_bound1 : γ < 180) :
  sin (α / 2) + sin (β / 2) > sin (γ / 2) :=
sorry

end rays_sine_inequality_l643_643367


namespace smallest_term_4_in_c_seq_l643_643515

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

noncomputable def b_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n - 1) + 15

noncomputable def c_seq (n : ℕ) : ℚ :=
  if n = 0 then 0 else (b_seq n) / (a_seq n)

theorem smallest_term_4_in_c_seq : 
  ∀ n : ℕ, n > 0 → c_seq 4 ≤ c_seq n :=
sorry

end smallest_term_4_in_c_seq_l643_643515


namespace distribution_of_balls_l643_643214

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end distribution_of_balls_l643_643214


namespace investment_initial_amount_l643_643124

noncomputable def initialInvestment (final_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  final_amount / interest_rate^years

theorem investment_initial_amount :
  initialInvestment 705.73 1.12 5 = 400.52 := by
  sorry

end investment_initial_amount_l643_643124


namespace proposition_iv_l643_643153

variables (a b c : Line) (α β γ : Plane)

theorem proposition_iv
  (h1 : a ⟂ α)
  (h2 : b ⟂ α)
  : a ∥ b := sorry

end proposition_iv_l643_643153


namespace largest_real_number_d_l643_643487

theorem largest_real_number_d (x : Fin 51 → ℝ) (M : ℝ) (d : ℝ) 
  (hM : M = x 25) 
  (hsum : ∑ i in Finset.univ, x i = M) : 
  (∑ i in Finset.univ, (x i)^2) ≥ d * (M / 51)^2 → d ≤ 51 :=
by
  sorry

end largest_real_number_d_l643_643487


namespace correct_answer_is_B_l643_643732

-- Definitions corresponding to the conditions given in the problem
def ethanol_reacts_with_sodium : Prop := 4.6 = 4.6 ∧ 1.12 ≠ 1.12

def bromine_extraction (benzene soybean_oil water bromine bromine_water : Type) [noncomputable] : Prop :=
(solubility bromine benzene > solubility bromine water) ∧ 
(solubility bromine soybean_oil > solubility bromine water) ∧ 
(immiscible benzene water) ∧ 
(immiscible soybean_oil water)

def covalent_bonds_propane (moles_of_propane moles_of_bonds : ℝ) : Prop :=
moles_of_bonds ≠ 0.7 * 6.02 * 10^23

def organic_compounds_c4h9cl : Prop :=
∃ (a b c d e : Type) [organic a] [organic b] [organic c] [organic d] [organic e],
organic_compounds_c4h9cl ≠ 5

-- The main statement
theorem correct_answer_is_B (benzene soybean_oil water bromine bromine_water : Type) :
  ¬ ethanol_reacts_with_sodium →
  bromine_extraction benzene soybean_oil water bromine bromine_water →
  ∀ (moles_of_propane moles_of_bonds : ℝ), covalent_bonds_propane moles_of_propane moles_of_bonds →
  ¬ organic_compounds_c4h9cl →
  correct_option = 'B' :=
by
  sorry

end correct_answer_is_B_l643_643732


namespace eval_expression_l643_643478

theorem eval_expression :
  (0.002 : ℝ)^(-1 / 2) - 10 * ((sqrt 5 - 2)⁻¹) + (sqrt 2 - sqrt 3)^0 = 1 :=
by sorry

end eval_expression_l643_643478


namespace least_number_to_subtract_l643_643387

theorem least_number_to_subtract (n : ℕ) : 
  ∃ k : ℕ, k = 762429836 % 17 ∧ k = 15 := 
by sorry

end least_number_to_subtract_l643_643387


namespace percentage_difference_l643_643584

noncomputable def P : ℝ := 40
variables {w x y z : ℝ}
variables (H1 : w = x * (1 - P / 100))
variables (H2 : x = 0.6 * y)
variables (H3 : z = 0.54 * y)
variables (H4 : z = 1.5 * w)

-- Goal
theorem percentage_difference : P = 40 :=
by sorry -- Proof omitted

end percentage_difference_l643_643584


namespace mary_sheep_remaining_l643_643281

theorem mary_sheep_remaining : 
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let sheep_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := sheep_after_sister / 2
  let remaining_sheep := sheep_after_sister - sheep_given_to_brother
  remaining_sheep = 150 :=
by
  assume initial_sheep := 400
  have sheep_given_to_sister := initial_sheep / 4
  have sheep_after_sister := initial_sheep - sheep_given_to_sister
  have sheep_given_to_brother := sheep_after_sister / 2
  have remaining_sheep := sheep_after_sister - sheep_given_to_brother
  show remaining_sheep = 150
  sorry

end mary_sheep_remaining_l643_643281


namespace max_min_ab_bc_ca_l643_643618

theorem max_min_ab_bc_ca (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 12) (h_ab_bc_ca : a * b + b * c + c * a = 30) :
  max (min (a * b) (min (b * c) (c * a))) = 9 :=
sorry

end max_min_ab_bc_ca_l643_643618


namespace expansion_zeros_of_999999999_cubed_zeros_in_expansion_l643_643324

theorem expansion_zeros_of_999999999_cubed :
  (999999999 : ℕ)^3 = (10^9 - 1)^3 :=
by sorry

theorem zeros_in_expansion :
  let n := 9 in
  let x := 999999999 in
  let pattern_squares (m : ℕ) : ℕ := m - 1 in
  let num_zeros_in_cubic (y : ℕ) : ℕ := 9 in
  num_zeros_in_cubic (x^3) = 9 :=
by sorry

end expansion_zeros_of_999999999_cubed_zeros_in_expansion_l643_643324


namespace find_angle_x_l643_643724

theorem find_angle_x (x : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ)
  (h₁ : α = 45)
  (h₂ : β = 3 * x)
  (h₃ : γ = x)
  (h₄ : α + β + γ = 180) :
  x = 33.75 :=
sorry

end find_angle_x_l643_643724


namespace fraction_expression_l643_643479

theorem fraction_expression :
  ((3 / 7) + (5 / 8)) / ((5 / 12) + (2 / 9)) = (531 / 322) :=
by
  sorry

end fraction_expression_l643_643479


namespace range_of_m_min_value_a2_2b2_3c2_l643_643554

theorem range_of_m (x m : ℝ) (h : ∀ x : ℝ, abs (x + 3) + abs (x + m) ≥ 2 * m) : m ≤ 1 :=
sorry

theorem min_value_a2_2b2_3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  ∃ (a b c : ℝ), a = 6/11 ∧ b = 3/11 ∧ c = 2/11 ∧ a^2 + 2 * b^2 + 3 * c^2 = 6/11 :=
sorry

end range_of_m_min_value_a2_2b2_3c2_l643_643554


namespace smallest_value_A_B_C_D_l643_643686

theorem smallest_value_A_B_C_D :
  ∃ (A B C D : ℕ), 
  (A < B) ∧ (B < C) ∧ (C < D) ∧ -- A, B, C are in arithmetic sequence and B, C, D in geometric sequence
  (C = B + (B - A)) ∧  -- A, B, C form an arithmetic sequence with common difference d = B - A
  (C = (4 * B) / 3) ∧  -- Given condition
  (D = (4 * C) / 3) ∧ -- B, C, D form geometric sequence with common ratio 4/3
  ((∃ k, D = k * 9) ∧ -- D must be an integer, ensuring B must be divisible by 9
   A + B + C + D = 43) := 
sorry

end smallest_value_A_B_C_D_l643_643686


namespace factor_expression_l643_643480

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 2 * x * (x + 3) + (x + 3) = (x + 1)^2 * (x + 3) := by
  sorry

end factor_expression_l643_643480


namespace price_of_the_car_l643_643643

noncomputable def down_payment : ℝ := 5000
noncomputable def monthly_payment : ℝ := 250
noncomputable def interest_rates : List ℝ := [0.02, 0.03, 0.04, 0.03, 0.05]
noncomputable def months_in_a_year : ℝ := 12

theorem price_of_the_car :
  (down_payment + (monthly_payment * months_in_a_year * 5) = 20000) :=
begin
  sorry
end

end price_of_the_car_l643_643643


namespace lines_parallel_lines_perpendicular_l643_643154

-- Definition of lines
def l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a ^ 2 - 1 = 0

-- Parallel condition proof problem
theorem lines_parallel (a : ℝ) : (a = -1) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y →  
        (-(a / 2) = (1 / (1 - a))) ∧ (-3 ≠ -a - 1) :=
by
  intros
  sorry

-- Perpendicular condition proof problem
theorem lines_perpendicular (a : ℝ) : (a = 2 / 3) → ∀ x y : ℝ, l1 a x y ∧ l2 a x y → 
        (- (a / 2) * (1 / (1 - a)) = -1) :=
by
  intros
  sorry

end lines_parallel_lines_perpendicular_l643_643154


namespace octagon_area_l643_643510

theorem octagon_area (AB BC : ℝ) (BDEF_is_square : Prop) (equilateral_triangles : Prop) :
  AB = 1 ∧ BC = 1 ∧ BDEF_is_square ∧ equilateral_triangles →
  area_of_octagon = 4 + sqrt 3 :=
sorry

end octagon_area_l643_643510


namespace q_is_false_l643_643582

-- Given conditions
variables (p q : Prop)
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ ¬ p

-- Proof that q is false
theorem q_is_false : q = False :=
by
  sorry

end q_is_false_l643_643582


namespace tangency_property_Pn_preserved_l643_643747

noncomputable theory

-- Define circles and tangency properties
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

def is_tangent_internally (ω ω₀ : Circle) : Prop :=
  Real.dist ω.center ω₀.center = ω₀.radius - ω.radius

def is_tangent_externally (ω₀ Ω : Circle) : Prop :=
  Real.dist ω₀.center Ω.center = ω.radius + Ω.radius

def property_Pn (n : ℕ) (ω ω₀ Ω : Circle) : Prop :=
  ∃ (ωi : Fin n.succ → Circle), 
    (∀ i : Fin n, Real.dist (ωi i).center (ωi (i + 1)).center = (ωi i).radius + (ωi (i + 1)).radius) ∧
    (∀ i : Fin n.succ, Real.dist (ωi i).center ω.center = ω.radius + (ωi i).radius) ∧ 
    (∀ i : Fin n.succ, Real.dist (ωi i).center Ω.center = ω.radius + Ω.radius)

-- Main proof problem statement
theorem tangency_property_Pn_preserved (ω Ω ω₀ ω₀' : Circle) (n : ℕ) :
  ω.radius < Ω.radius →
  is_tangent_internally ω ω₀ →
  is_tangent_externally ω₀ Ω → 
  property_Pn n ω ω₀ Ω → 
  is_tangent_externally ω₀' Ω →
  is_tangent_internally ω₀' ω →
  property_Pn n ω ω₀' Ω := 
sorry

end tangency_property_Pn_preserved_l643_643747


namespace range_of_a_l643_643175

noncomputable def curve_y (a : ℝ) (x : ℝ) : ℝ := (a - 3) * x^3 + Real.log x
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ := x^3 - a * x^2 - 3 * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ deriv (curve_y a) x = 0) ∧
  (∀ x ∈ Set.Icc (1 : ℝ) 2, 0 ≤ deriv (function_f a) x) → a ≤ 0 :=
by sorry

end range_of_a_l643_643175


namespace max_sum_prod_48_l643_643751

theorem max_sum_prod_48 (spadesuit heartsuit : Nat) (h: spadesuit * heartsuit = 48) : spadesuit + heartsuit ≤ 49 :=
sorry

end max_sum_prod_48_l643_643751


namespace find_x_l643_643482

theorem find_x (x : ℝ) (h : 9 ^ (Real.log x / Real.log 5) = 81) : x = 25 :=
sorry

end find_x_l643_643482


namespace bird_families_flew_away_to_Africa_l643_643006

theorem bird_families_flew_away_to_Africa 
  (B : ℕ) (n : ℕ) (hB94 : B = 94) (hB_A_plus_n : B = n + 47) : n = 47 :=
by
  sorry

end bird_families_flew_away_to_Africa_l643_643006


namespace correctly_delivered_probability_l643_643123

noncomputable def probability_three_correct_deliveries : ℚ := 1 / 12

theorem correctly_delivered_probability :
  let num_packages := 5
  let total_ways := Nat.factorial num_packages
  let favorable_ways := Nat.choose num_packages 3 * Nat.factorial 2
  let correct_prob := favorable_ways / total_ways
  correct_prob = probability_three_correct_deliveries :=
  by
  let num_packages := 5
  let total_ways := Nat.factorial num_packages
  let favorable_ways := Nat.choose num_packages 3 * Nat.factorial 2
  let correct_prob := favorable_ways / total_ways
  show correct_prob = probability_three_correct_deliveries from sorry

end correctly_delivered_probability_l643_643123


namespace smallest_k_value_l643_643712

def minimal_blocks_k (N : ℕ) (a : ℕ → ℕ) (pos : ℕ) : Prop :=
  ∀ (i j : ℕ), i ≤ j → a i ≤ 2^(j-3) + i

theorem smallest_k_value : ∃ (k : ℕ), k = 9 ∧ minimal_blocks_k N (λ n, 2^(n-1)) 0 :=
by
  sorry

end smallest_k_value_l643_643712


namespace bicycle_cost_price_l643_643789

theorem bicycle_cost_price (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ)
    (h1 : CP_B = 1.60 * CP_A)
    (h2 : SP_C = 1.25 * CP_B)
    (h3 : SP_C = 225) :
    CP_A = 225 / 2.00 :=
by
  sorry -- the proof steps will follow here

end bicycle_cost_price_l643_643789


namespace range_of_a_l643_643919

-- Define the function f as given in the problem
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ a then |Real.log x|
  else -(x - 3 * a + 1) ^ 2 + (2 * a - 1) ^ 2 + a

-- Define the function g as f(x) - b
def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := f x a - b

-- The statement about the range of a
theorem range_of_a (a b : ℝ) (h_pos_b : b > 0) :
  (0 < a ∧ a < 1/2) ↔ (g x a b has_four_distinct_zeros) :=
  sorry

end range_of_a_l643_643919


namespace area_LOM_eq_3_l643_643963

open Real

-- Defining the given problem in Lean
variable (A B C L O M : Type) [triangle : Triangle A B C]
variable (α β γ : ℝ)
variable (circumcircle : Circumcircle A B C)
variable (is_scalene : scalene_triangle A B C) 
variable (angle_condition_1 : α = β - γ) 
variable (angle_condition_2 : β = 2 * γ)
variable (angle_bisectors : PointOnCircumcircle L (AngleBisector A circumcircle) ∧ 
                                              PointOnCircumcircle O (AngleBisector B circumcircle) ∧ 
                                              PointOnCircumcircle M (AngleBisector C circumcircle))
variable (area_ABC_eq_2 : TriangleArea A B C = 2)

theorem area_LOM_eq_3 :
  TriangleArea L O M = 3 := by
  sorry

end area_LOM_eq_3_l643_643963


namespace problem_proof_l643_643950

   variable {A B C a b c : ℝ}
   variable (h₁ : b^2 = a * c)
   variable (h₂ : a^2 - c^2 = a * c - b * c)

   theorem problem_proof : ∀ {b : ℝ}, ∀ {sin_B : ℝ},
     b^2 = a*c → a^2 − c^2 = a*c − b*c → 
     sin_B = 2 / 3 → 
     c / (b * sin_B) = 2 / sqrt 3 :=
   by
     intro b sin_B h₁ h₂
     sorry
   
end problem_proof_l643_643950


namespace correct_article_usage_l643_643082

def sentence : String :=
  "While he was at ____ college, he took part in the march, and was soon thrown into ____ prison."

def rules_for_articles (context : String) (noun : String) : String → Bool
| "the" => noun ≠ "college" ∨ context = "specific"
| ""    => noun = "college" ∨ noun = "prison"
| _     => false

theorem correct_article_usage : 
  rules_for_articles "general" "college" "" ∧ 
  rules_for_articles "general" "prison" "" :=
by
  sorry

end correct_article_usage_l643_643082


namespace type_of_numbers_l643_643333

theorem type_of_numbers (average : ℕ → ℕ) (h: (average 10) = 11) :
  (∀ n, n < 10 → even (2 * n + 2)) :=
by
  sorry

end type_of_numbers_l643_643333


namespace connie_total_markers_l643_643087

theorem connie_total_markers : 2315 + 1028 = 3343 :=
by
  sorry

end connie_total_markers_l643_643087


namespace train_length_l643_643054

noncomputable theory

def speed_km_per_hour := 21
def time_seconds := 142.2857142857143
def length_bridge_meters := 130
def speed_m_per_s := (speed_km_per_hour * 1000) / 3600
def total_distance := speed_m_per_s * time_seconds
def length_train := total_distance - length_bridge_meters

theorem train_length : length_train = 700 := 
by sorry

end train_length_l643_643054


namespace number_of_blocks_needed_l643_643931

theorem number_of_blocks_needed 
  (length width height : ℕ)
  (h_length : length = 4)
  (h_width : width = 3)
  (h_height : height = 3) :
  (length * width * height = 36) :=
by {
  rw [h_length, h_width, h_height],
  norm_num,
  sorry
}

end number_of_blocks_needed_l643_643931


namespace problem_l643_643979

-- Define the sequence a_n
def a : ℕ → ℝ
| 0     := sqrt 2
| (n+1) := sqrt (a n ^ 2 + 2)

-- Define the sequence b_n
def b (n : ℕ) : ℝ :=
  (n + 1) / (a n ^ 4 * (n + 2) ^ 2)

-- Define the sum S_n of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, b k)

-- The main theorem
theorem problem : ∀ (n : ℕ), 16 * S n + 1 / (n+1)^2 + 1 / (n+2)^2 = 5 / 4 :=
by
  intro n
  sorry

end problem_l643_643979


namespace monomial_sum_exponents_l643_643946

theorem monomial_sum_exponents (m n : ℤ)
  (h1 : m - 2 = n)
  (h2 : 2 * m - 3 * n = 3) :
  m ^ -n = 1 / 3 :=
by
  sorry

end monomial_sum_exponents_l643_643946


namespace complex_modulus_sqrt_l643_643526

open Complex

theorem complex_modulus_sqrt :
  abs ((⟨1, 3⟩ * ⟨1, 1⟩) / (⟨1, -1⟩ * ⟨1, 1⟩)) = Real.sqrt 5 := 
by
  sorry

end complex_modulus_sqrt_l643_643526


namespace area_triangle_ADE_l643_643606

variable {Point : Type} [EuclideanGeometry.PointSpace Point]

-- Definitions of points A, B, C, D, E
variables (A B C D E : Point)

-- Condition 1: D is the midpoint of BC
def is_midpoint_BC (D B C : Point) : Prop := 
  ∃ M, segment(M, B) + segment(M, C) = segment(B, C)  ∧ B + C = 2 * M

-- Condition 2: E is the midpoint of AC
def is_midpoint_AC (E A C : Point) : Prop := 
  ∃ N, segment(N, A) + segment(N, C) = segment(A, C) ∧ A + C = 2 * N

-- Condition 3: The area of triangle ABC is 36 square units
def area_ABC_36 (A B C : Point) : Prop := 
  ∃ area, area = area_triangle(A, B, C) ∧ area = 36

-- The proof statement
theorem area_triangle_ADE (A B C D E : Point) 
  (h1 : is_midpoint_BC D B C) 
  (h2 : is_midpoint_AC E A C) 
  (h3 : area_ABC_36 A B C) : 
  area_triangle(A, D, E) = 9 := 
sorry

end area_triangle_ADE_l643_643606


namespace probability_of_multiple_of_seven_difference_l643_643102

theorem probability_of_multiple_of_seven_difference (S : Finset ℕ) (hS : S.card = 8) (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 3000) :
  ∃ a b ∈ S, a ≠ b ∧ (a - b) % 7 = 0 := 
by
  sorry

end probability_of_multiple_of_seven_difference_l643_643102


namespace right_triangle_incircle_excircle_condition_l643_643839

theorem right_triangle_incircle_excircle_condition
  (r R : ℝ) 
  (hr_pos : 0 < r) 
  (hR_pos : 0 < R) :
  R ≥ r * (3 + 2 * Real.sqrt 2) := sorry

end right_triangle_incircle_excircle_condition_l643_643839


namespace revenue_after_decrease_l643_643796

theorem revenue_after_decrease (original_revenue : ℝ) (percentage_decrease : ℝ) (final_revenue : ℝ) 
  (h1 : original_revenue = 69.0) 
  (h2 : percentage_decrease = 24.637681159420293) 
  (h3 : final_revenue = original_revenue - (original_revenue * (percentage_decrease / 100))) 
  : final_revenue = 52.0 :=
by
  sorry

end revenue_after_decrease_l643_643796


namespace total_tickets_sold_equals_522_l643_643685

def adult_ticket_cost : ℕ := 15
def child_ticket_cost : ℕ := 8
def total_receipts : ℕ := 5086
def adult_tickets_sold : ℕ := 130

theorem total_tickets_sold_equals_522 
  (C : ℕ) 
  (H1 : adult_tickets_sold * adult_ticket_cost + C * child_ticket_cost = total_receipts) 
  (H2 : adult_tickets_sold + C = 522) :
  130 + C = 522 :=
by {
  have total_adults_cost : adult_ticket_cost * adult_tickets_sold = 1950, 
  -- proof skipped for the sake of the demonstration
  sorry,

  have total_children_cost : total_receipts - (adult_ticket_cost * adult_tickets_sold) = 3136, 
  -- proof skipped for the sake of the demonstration
  sorry,

  have children_sold : C = 392, 
  -- proof skipped for the sake of the demonstration
  sorry,

  have total_sold_is_sum : adult_tickets_sold + 392 = 522, 
  -- proof skipped for the sake of the demonstration
  sorry,
  
  exact sorry
}

end total_tickets_sold_equals_522_l643_643685


namespace find_general_term_sum_of_squares_lt_l643_643978

open Nat

def a : ℕ → ℚ
| 1     := 1
| 2     := 1 / 4
| (n+1) := (n-1) * a n / (n - a n)
| _     := 0

theorem find_general_term {n : ℕ} (h_pos : 0 < n) : a n = 1 / (3 * n - 2) :=
sorry

theorem sum_of_squares_lt (n : ℕ) : ∑ k in range (n + 1), (a k) ^ 2 < 7 / 6 :=
sorry

end find_general_term_sum_of_squares_lt_l643_643978


namespace arith_seq_relationship_l643_643578

noncomputable def sum_first_n_terms (n : ℕ) (a : ℕ → ℕ) : ℕ := 
  (finset.range n).sum a

theorem arith_seq_relationship (S : ℕ → ℕ) (n : ℕ) : 
  (∀ n, S n = sum_first_n_terms n (λ i, i + 1)) → 
  S (3 * n) = 3 * (S (2 * n) - S n) :=
begin
  sorry
end

end arith_seq_relationship_l643_643578


namespace number_in_center_is_seven_l643_643060

theorem number_in_center_is_seven 
  (A1 A2 A3 A4 A5 A6 A7 A8 A9 : ℕ)
  (positions : (A1, A2, A3, A4, A5, A6, A7, A8, A9) ∈ ({(n₁, n₂, n₃, n₄, n₅, n₆, n₇, n₈, n₉) | 
    (Set.of_finset {1, 2, 3, 4, 5, 6, 7, 8, 9}) ⊆ {n₁, n₂, n₃, n₄, n₅, n₆, n₇, n₈, n₉} ∧ 
    (∀ i j, n₁ ≤ n₂ → abs(i - j) = 1 → abs(n₁ - n₂) = 1) ∧ 
    (A1 + A3 + A7 + A9 = 18)
  } )) 
  : A5 = 7 := 
  sorry

end number_in_center_is_seven_l643_643060


namespace number_value_l643_643780

theorem number_value (x : ℝ) (h : x = 3 * (1/x * -x) + 5) : x = 2 :=
by
  sorry

end number_value_l643_643780


namespace number_of_integers_l643_643865

theorem number_of_integers (n : ℤ) : 20 < n^2 → n^2 < 200 → (finset.Icc 5 14).card + (finset.Icc -14 -5).card = 20 := by
  sorry

end number_of_integers_l643_643865


namespace grayson_vs_rudy_distance_l643_643564

-- Definitions based on the conditions
def grayson_first_part_distance : Real := 25 * 1
def grayson_second_part_distance : Real := 20 * 0.5
def total_grayson_distance : Real := grayson_first_part_distance + grayson_second_part_distance
def rudy_distance : Real := 10 * 3

-- Theorem stating the problem to be proved
theorem grayson_vs_rudy_distance : total_grayson_distance - rudy_distance = 5 := by
  -- Proof would go here
  sorry

end grayson_vs_rudy_distance_l643_643564


namespace max_sin_angle_F1PF2_on_ellipse_l643_643904

theorem max_sin_angle_F1PF2_on_ellipse
  (x y : ℝ)
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (h : P ∈ {Q | Q.1^2 / 9 + Q.2^2 / 5 = 1})
  (F1_is_focus : F1 = (-2, 0))
  (F2_is_focus : F2 = (2, 0)) :
  ∃ sin_max, sin_max = 4 * Real.sqrt 5 / 9 := 
sorry

end max_sin_angle_F1PF2_on_ellipse_l643_643904


namespace initial_sales_quota_l643_643434

theorem initial_sales_quota (nike_price : ℕ) (adidas_price : ℕ) (reebok_price : ℕ)
  (nike_count : ℕ) (adidas_count : ℕ) (reebok_count : ℕ) (above_goal_amount : ℕ) : 
  nike_price = 60 →
  adidas_price = 45 →
  reebok_price = 35 →
  nike_count = 8 →
  adidas_count = 6 →
  reebok_count = 9 →
  above_goal_amount = 65 →
  let total_sales := nike_count * nike_price + adidas_count * adidas_price + reebok_count * reebok_price in
  let initial_sales_quota := total_sales - above_goal_amount in
  initial_sales_quota = 1000 :=
begin
  intros,
  sorry
end

end initial_sales_quota_l643_643434


namespace rectangle_perimeter_ratio_l643_643325

theorem rectangle_perimeter_ratio
    (initial_height : ℕ)
    (initial_width : ℕ)
    (H_initial_height : initial_height = 2)
    (H_initial_width : initial_width = 4)
    (fold1_height : ℕ)
    (fold1_width : ℕ)
    (H_fold1_height : fold1_height = initial_height / 2)
    (H_fold1_width : fold1_width = initial_width)
    (fold2_height : ℕ)
    (fold2_width : ℕ)
    (H_fold2_height : fold2_height = fold1_height)
    (H_fold2_width : fold2_width = fold1_width / 2)
    (cut_height : ℕ)
    (cut_width : ℕ)
    (H_cut_height : cut_height = fold2_height)
    (H_cut_width : cut_width = fold2_width) :
    (2 * (cut_height + cut_width)) / (2 * (fold1_height + fold1_width)) = 3 / 5 := 
    by sorry

end rectangle_perimeter_ratio_l643_643325


namespace pencils_sold_is_correct_l643_643691

-- Define the conditions
def first_two_students_pencils : Nat := 2 * 2
def next_six_students_pencils : Nat := 6 * 3
def last_two_students_pencils : Nat := 2 * 1
def total_pencils_sold : Nat := first_two_students_pencils + next_six_students_pencils + last_two_students_pencils

-- Prove that all pencils sold equals 24
theorem pencils_sold_is_correct : total_pencils_sold = 24 :=
by 
  -- Add the statement to be proved here
  sorry

end pencils_sold_is_correct_l643_643691


namespace determine_matrix_N_l643_643467

def i : Matrix 3 1 ℝ := ![![4], ![-2], ![3]]
def j : Matrix 3 1 ℝ := ![![-1], ![6], ![0]]
def k : Matrix 3 1 ℝ := 2 • i - j

theorem determine_matrix_N (N : Matrix 3 3 ℝ)
  (Ni : Matrix 3 1 ℝ) (Nj : Matrix 3 1 ℝ) (Nk : Matrix 3 1 ℝ)
  (h1 : Ni = ![![4], ![-2], ![3]])
  (h2 : Nj = ![![-1], ![6], ![0]])
  (h3 : Nk = 2 • Ni - Nj)
  (hN : N = Matrix.fromBlocks Ni Nj Nk) :
  N = ![![4, -1, 9], ![-2, 6, -10], ![3, 0, 6]] :=
by
  sorry

end determine_matrix_N_l643_643467


namespace solve_bank_account_problem_l643_643094

noncomputable def bank_account_problem : Prop :=
  ∃ (A E Z : ℝ),
    A > E ∧
    Z > A ∧
    A - E = (1/12) * (A + E) ∧
    Z - A = (1/10) * (Z + A) ∧
    1.10 * A = 1.20 * E + 20 ∧
    1.10 * A + 30 = 1.15 * Z ∧
    E = 2000 / 23

theorem solve_bank_account_problem : bank_account_problem :=
sorry

end solve_bank_account_problem_l643_643094


namespace speed_first_half_proof_l643_643419

noncomputable def speed_first_half
  (total_time: ℕ) 
  (distance: ℕ) 
  (second_half_speed: ℕ) 
  (first_half_time: ℕ) :
  ℕ :=
  distance / first_half_time

theorem speed_first_half_proof
  (total_time: ℕ)
  (distance: ℕ)
  (second_half_speed: ℕ)
  (half_distance: ℕ)
  (second_half_time: ℕ)
  (first_half_time: ℕ) :
  total_time = 12 →
  distance = 560 →
  second_half_speed = 40 →
  half_distance = distance / 2 →
  second_half_time = half_distance / second_half_speed →
  first_half_time = total_time - second_half_time →
  speed_first_half total_time half_distance second_half_speed first_half_time = 56 :=
by
  sorry

end speed_first_half_proof_l643_643419


namespace perimeter_approximation_more_accurate_l643_643717

theorem perimeter_approximation_more_accurate (n : ℕ) (h : 3 ≤ n) :
  let θ := (2 * Real.pi) / n in
  let A1A2 := 2 * Real.sin (Real.pi / n) in
  let area_approx := (n / 2) * Real.sin θ in
  let perimeter_approx := 2 * n * Real.sin (Real.pi / n) in
  let area_error := 1 - area_approx / Real.pi in
  let perimeter_error := 1 - perimeter_approx / (2 * Real.pi) in
  perimeter_error < area_error := by
  sorry

end perimeter_approximation_more_accurate_l643_643717


namespace direction_vector_of_line_l643_643670

noncomputable def direction_vector_of_line_eq : Prop :=
  ∃ u v, ∀ x y, (x / 4) + (y / 2) = 1 → (u, v) = (-2, 1)

theorem direction_vector_of_line :
  direction_vector_of_line_eq := sorry

end direction_vector_of_line_l643_643670


namespace EFGH_parallelogram_EFGH_rhombus_EFGH_square_l643_643972

variables {A B C D E F G H : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables [AddGroup E] [AddGroup F] [AddGroup G] [AddGroup H]
variables [Midpoint A B E] [Midpoint B C F] [Midpoint C D G] [Midpoint D A H]

--(1) Prove that quadrilateral EFGH is a parallelogram
theorem EFGH_parallelogram : Parallelogram E F G H :=
sorry

--(2) If AC = BD, prove that quadrilateral EFGH is a rhombus
theorem EFGH_rhombus (hAC_BD : MetricSpace.dist A C = MetricSpace.dist B D) : Rhombus E F G H :=
sorry

--(3) Under what condition regarding AC and BD, will quadrilateral EFGH be a square?
theorem EFGH_square (hAC_BD : MetricSpace.dist A C = MetricSpace.dist B D) (hAC_perp_BD : ∀ P Q : Type, P != Q → MetricSpace.angle P Q = 90) : Square E F G H :=
sorry

end EFGH_parallelogram_EFGH_rhombus_EFGH_square_l643_643972


namespace contractor_engaged_days_l643_643038

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l643_643038


namespace geometric_sequence_properties_l643_643145

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 3^(n-1)
def b (n : ℕ) : ℕ := logBase 3 (a n * a (n + 1))

-- Main theorem
theorem geometric_sequence_properties :
  (∀ n, a n = 3^(n-1)) ∧
  (∀ n, ∑ i in finset.range n, a i * b i = 1 + (n-1) * 3^n) :=
by sorry

end geometric_sequence_properties_l643_643145


namespace max_red_columns_correct_l643_643064

noncomputable def max_red_columns (m n : ℕ) (h : m ≤ n) : ℕ :=
  n - m + 1

theorem max_red_columns_correct (m n : ℕ) (h : m ≤ n) (unique_patterns : ∀ i j, i < j → (∃ k, get_row i k ≠ get_row j k)) :
  ∃ k, k = max_red_columns m n h ∧ (∀ i j, i < j → (∃ c, (c < k ∧ rows_differ_at i j c))) :=
sorry

end max_red_columns_correct_l643_643064


namespace water_tank_initial_ratio_l643_643409

variables (total_capacity: ℝ) (fill_rate: ℝ) (outflow_rate1: ℝ) (outflow_rate2: ℝ) (fill_time: ℝ) (initial_water: ℝ)
variables (net_inflow_rate: ℝ) (amount_added: ℝ) (initial_ratio: ℝ)

def effective_fill_rate := fill_rate - (outflow_rate1 + outflow_rate2)

theorem water_tank_initial_ratio :
  total_capacity = 6000 ∧
  fill_rate = 0.5 ∧
  outflow_rate1 = 0.25 ∧
  outflow_rate2 = 1/6 ∧
  fill_time = 36 ∧
  effective_fill_rate = 0.5 - (0.25 + 1/6) ∧
  amount_added = effective_fill_rate * fill_time ∧
  initial_water = total_capacity - amount_added -> 
  initial_ratio = initial_water / total_capacity :=
begin
  intros h,
  cases h with h1 h,
  cases h with h2 h,
  cases h with h3 h,
  cases h with h4 h,
  cases h with h5 h,
  cases h with h6 h,
  cases h with h7 h,
  cases h with h8 h,
  rw [h1, h2, h3, h4, h5, h6, h7, h8],
  sorry
end

end water_tank_initial_ratio_l643_643409


namespace repeating_decimal_sum_l643_643392

theorem repeating_decimal_sum (x : ℚ) (h : x = 0.272727272727) :
  let f := x.num / x.denom in f.num + f.denom = 12 := by
  sorry

end repeating_decimal_sum_l643_643392


namespace spaceship_speed_400_l643_643128

-- The function spaceship_speed takes the number of people on board and returns the speed of the spaceship.
noncomputable def spaceship_speed : ℕ → ℝ
| 0     := sorry  -- base speed without people is unknown and depends on the context
| 200   := 500
| n + 100 := (spaceship_speed n) / 2

theorem spaceship_speed_400 : spaceship_speed 400 = 125 := by sorry

end spaceship_speed_400_l643_643128


namespace binom_21_12_l643_643523

theorem binom_21_12 :
  (binomial 20 13 = 77520) →
  (binomial 20 12 = 125970) →
  (binomial 21 13 = 203490) →
  binomial 21 12 = 125970 :=
begin
  intros h1 h2 h3,
  sorry
end

end binom_21_12_l643_643523


namespace smallest_increase_between_3_and_4_l643_643126

def percent_increase (x y : ℕ) := ((y - x) * 100) / x

def question_values : List ℕ := [100, 300, 500, 800, 1300, 2100, 3400, 5500, 8900, 14400, 23300, 37700, 61000, 100000, 162000]

def smallest_percent_increase : ℕ :=
  let increases := List.map₂ (λ x y => percent_increase x y) question_values (question_values.tail!)
  increases.min!

theorem smallest_increase_between_3_and_4 :
  smallest_percent_increase = percent_increase 500 800 :=
  sorry

end smallest_increase_between_3_and_4_l643_643126


namespace angle_CHX_is_30_l643_643436

variable (A B C X Y H : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder X] [LinearOrder Y] [LinearOrder H]
variable (angle_BAC : ℝ) (angle_ABC : ℝ) (triangle_ABC : Prop) (acute_ABC : triangle_ABC → Prop) (intersect_H : H → Prop)
variables (angle_53 : angle_BAC = 53) (angle_67 : angle_ABC = 67)
variable (orthocenter_H : H → Prop) (altitudes_AX_BY_H : H)

theorem angle_CHX_is_30 {H : Type} (h_tr : triangle_ABC) (h_acute : acute_ABC h_tr) 
  (h_intersect : intersect_H H) (h_53 : angle_53) (h_67 : angle_67) :
  ∃ (CHX : ℝ), CHX = 30 :=
by sorry

end angle_CHX_is_30_l643_643436


namespace line_of_centers_parallel_l643_643879

/-- Points and conditions given -/
variables (A B O O₁ O₂ : Point) (R r : ℝ)

/-- Given points A and B on a line -/
axiom on_line : ∃ (l : Line), l.contains A ∧ l.contains B

/-- Given circle with center O and radius R -/
axiom circle_O : ∃ (c : Circle), c.center = O ∧ c.radius = R

/-- Given two circles each with radius r inscribed in the angles at points A and B with centers O₁ and O₂ respectively -/
axiom circle_O₁ : ∃ (c₁ : Circle), c₁.center = O₁ ∧ c₁.radius = r
axiom circle_O₂ : ∃ (c₂ : Circle), c₂.center = O₂ ∧ c₂.radius = r

/-- Prove that the line segment O₁ O₂ is parallel to AB -/
theorem line_of_centers_parallel : parallel (line O₁ O₂) (line A B) :=
sorry

end line_of_centers_parallel_l643_643879


namespace least_n_factorial_9450_l643_643720

theorem least_n_factorial_9450 :
  ∃ n, n > 0 ∧ 9450 ∣ nat.factorial n ∧ ∀ m, 0 < m < n → ¬ (9450 ∣ nat.factorial m) :=
by
  have h1 : 1 ≤ 2 := by norm_num
  have h2 : 1 ≤ 3^3 := by norm_num
  have h3 : 1 ≤ 5^2 := by norm_num
  have h4 : 1 ≤ 7 := by norm_num
  use 10,
  sorry

end least_n_factorial_9450_l643_643720


namespace area_of_triangle_l643_643384

noncomputable def line1 : ℝ → ℝ := λ x, 3 * x + 5
noncomputable def line2 : ℝ → ℝ := λ x, -2 * x + 12
noncomputable def y_axis : ℝ := 0

-- Define the intersection point (x, y) of line1 and line2
noncomputable def intersection : ℝ × ℝ :=
  let x := 7 / 5 in
  (x, line1 x)

-- Define the y-intercepts of line1 and line2
noncomputable def intercept1 : ℝ × ℝ := (0, line1 0)
noncomputable def intercept2 : ℝ × ℝ := (0, line2 0)

-- Define the base and height of the triangle
noncomputable def base := abs ((intercept2.snd) - (intercept1.snd))
noncomputable def height := intersection.fst

-- The area of the triangle
noncomputable def triangle_area : ℝ := 1 / 2 * base * height

theorem area_of_triangle : triangle_area = 4.9 :=
by
  -- Proof goes here
  sorry

end area_of_triangle_l643_643384


namespace minimum_S_value_l643_643013

noncomputable def S_min (n : ℕ) (a b : Fin 2n → ℝ) : ℝ :=
if h : n = 3 then 12
else if n ≥ 4 then 16
else 0

theorem minimum_S_value (n : ℕ) (a b : Fin 2n → ℝ)
  (h₁ : n ≥ 3)
  (h₂ : ∀ i : Fin 2n, 0 ≤ a i) 
  (h₃ : ∀ i : Fin 2n, 0 ≤ b i)
  (h₄ : (∑ i, a i) = ∑ i, b i ∧ (∑ i, a i) > 0)
  (h₅ : ∀ i : Fin 2n, a i * a ((i + 2) % (2 * n)) ≥ b i + b ((i + 1) % (2 * n))) :
  ∑ i, a i = S_min n a b := sorry

end minimum_S_value_l643_643013


namespace hippocrates_lunes_area_eq_triangle_area_l643_643631

theorem hippocrates_lunes_area_eq_triangle_area 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (semicircle_area : ∀ d : ℝ, (π * d^2) / 8 = (π * d^2) / 8) :
  let lune_area := (π * a^2) / 8 + (π * b^2) / 8 - (π * c^2) / 8 
  in lune_area = (a * b) / 2 := 
by
  sorry

end hippocrates_lunes_area_eq_triangle_area_l643_643631


namespace even_function_minimum_value_f_l643_643675

noncomputable def f (x : ℝ) : ℝ := real.log ((x^2 + 1) / |x|)

theorem even_function (x : ℝ) (hx : x ≠ 0) : f (-x) = f x := by
  sorry

theorem minimum_value_f (x : ℝ) (hx : x ≠ 0) : ∃ c : ℝ, c = real.log 2 ∧ ∀ y, f y ≥ c := by
  sorry

end even_function_minimum_value_f_l643_643675


namespace factor_expression_l643_643673

variable {R : Type*} [CommRing R]

theorem factor_expression (a b c : R) :
    a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) =
    (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) :=
sorry

end factor_expression_l643_643673


namespace incorrect_statement_D_l643_643579

variables {a b c l : Line} -- Assume we have some lines
variables (parallel : a ∥ b)
variables (trans_prop_parallel : ∀ {x y z : Line}, x ∥ y → y ∥ z → x ∥ z)
variables (perpendicular : ∀ {x y z : Line}, x ∥ y → x ⟂ z → y ⟂ z)

-- Definition for the angle condition:
def angle_condition (l : Line) (a b : Line) : Prop :=
  -- the corresponding interior angles cut by third line l on the same side of a and b are equal
  (∀ θ₁ θ₂ : Angle, interior_angle l a θ₁ → interior_angle l b θ₂ → θ₁ = θ₂)

-- Statement to prove that D is incorrect
theorem incorrect_statement_D
  (angle_cond : angle_condition l a b)
  : false :=
by
  sorry

end incorrect_statement_D_l643_643579


namespace teachers_gave_candy_caned_l643_643805

theorem teachers_gave_candy_caned 
  (cavities_per_cane : ℕ := 4)
  (candy_caned_from_parents : ℕ := 2)
  (candy_caned_per_teacher : ℕ := 3)
  (total_cavities : ℕ := 16)
  (allowance_ratio : ℚ := 1 / 7) :
  ∃ T : ℕ, 2 + 3 * T + allowance_ratio * (2 + 3 * T) = 4 * total_cavities :=
begin
  use 18,
  -- Simplify the assumption using the given context
  have h1 : 2 + 3 * 18 + allowance_ratio * (2 + 3 * 18) = 4 * 16,
  sorry,  -- Detailed proof steps would go here
end

end teachers_gave_candy_caned_l643_643805


namespace value_of_f_neg2_l643_643141

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - log 3 (x^2 - 3*x + 5) else -(2^(-x) - log 3 ((-x)^2 - 3*(-x) + 5))

theorem value_of_f_neg2 :
  f (-2) = -3 :=
by
  -- Conditions:
  -- f is defined on ℝ
  -- f is an odd function
  -- For x > 0, f(x) = 2^x - log_3(x^2 - 3*x + 5)

  sorry

end value_of_f_neg2_l643_643141


namespace length_of_second_train_l643_643716

-- Define the constants for the given problem
def L1 : ℝ := 100  -- Length of the first train in meters
def v1 : ℝ := 42 * (5 / 18)  -- Speed of the first train in meters per second
def v2 : ℝ := 30 * (5 / 18)  -- Speed of the second train in meters per second
def t : ℝ := 12.998960083193344  -- Time in seconds

-- Define the target length of the second train
def L2 := 159.98  -- Expected length of the second train in meters

-- State the proof problem
theorem length_of_second_train : 
  let relative_speed := v1 + v2 in
  let total_distance := relative_speed * t in
  L2 = total_distance - L1 :=
by
  -- We add the proof body here
  sorry

end length_of_second_train_l643_643716


namespace circle_curve_intersection_l643_643941

theorem circle_curve_intersection 
  {R : ℝ} (hR : R > 0) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 = R^2)
  (h_curve : ∀ x y : ℝ, abs (abs x - abs y) = 1) :
  R = 1 ∨ R = real.sqrt (2 + real.sqrt 2) := 
sorry

end circle_curve_intersection_l643_643941


namespace cos_alpha_eq_trig_expression_value_l643_643168

noncomputable def P : ℝ × ℝ := (4/5, -3/5)
noncomputable def α : ℝ := sorry -- α should be constructed such that P is on its terminal side

theorem cos_alpha_eq : ∃ (α : ℝ), cos α = 4/5 := by
  use α
  have hP := P
  sorry

theorem trig_expression_value : ∃ (α : ℝ), 
  (sin(π/2 - α) / sin(α + π) * tan(α - π) / cos(3 * π - α)) = 5 / 4 := by
  use α
  have hP := P
  sorry

end cos_alpha_eq_trig_expression_value_l643_643168


namespace f_property_P_b_f_monotonic_intervals_g_property_P2_range_m_l643_643999

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := log x + (b + 2) / (x + 1)
def h_f (x : ℝ) : ℝ := 1 / (x * (x + 1) ^ 2)

-- f has property P(b)
theorem f_property_P_b (b : ℝ) (x : ℝ) (h₀ : 1 < x) : 
  ∃ (h : ℝ → ℝ), h > 0 ∧ 
    (∀ x, f' x = h x * (x^2 - b * x + 1)) :=
sorry

-- Monotonic intervals of f
theorem f_monotonic_intervals (b : ℝ) :
  (∀ x, 1 < x → (b ≤ 2 → (f' x > 0)) ∧ 
    (b > 2 → ((1 < x ∧ x < (b + sqrt (b^2 - 4)) / 2) → 
      (f' x < 0)) ∧ ((x ≥ (b + sqrt (b^2 - 4)) / 2) → 
        (f' x > 0)))) :=
sorry

-- Range of m for g(α) - g(β) < g(x₁) - g(x₂)
theorem g_property_P2_range_m (g : ℝ → ℝ) (mx₁ x₁ x₂ : ℝ) 
  (g' : ℝ → ℝ) (h₁ : 1 < x₁) (h₂ : 1 < x₂) (h₃ : x₁ < x₂) 
  (m : ℝ) (α β : ℝ) (hα : α = mx₁ + (1 - m) * x₂) 
  (hβ : β = (1 - m) * x₁ + mx₂) (hα_gt : α > 1) 
  (hβ_gt : β > 1) (h_property_P2 : ∀ x, g' x = h x * (x - 1)^2) 
  (g'_pos : ∀ x, 1 < x → h x * (x - 1)^2 > 0) :
  (∀ x, 0 < m ∧ m < 1 → |g α - g β| < |g x₁ - g x₂|) :=
sorry

end f_property_P_b_f_monotonic_intervals_g_property_P2_range_m_l643_643999


namespace find_y_l643_643574

def G (a b c d : ℕ) : ℕ := a ^ b + c * d

theorem find_y (y : ℕ) : G 3 y 5 10 = 350 ↔ y = 5 := by
  sorry

end find_y_l643_643574


namespace eva_total_marks_l643_643845

theorem eva_total_marks :
  ∀ (maths2 arts2 science2 history2: ℕ),
  (maths2 = 80) →
  (arts2 = 90) →
  (science2 = 90) →
  (history2 = 85) →
  let maths1 := maths2 + 10,
      arts1 := arts2 - 15,
      science1 := science2 - (1 / 3 : ℚ) * science2,
      history1 := history2 + 5,
      total_marks := maths1 + arts1 + science1.natAbs + history1 + maths2 + arts2 + science2 + history2
  in total_marks = 660 :=
by
  intros
  sorry

end eva_total_marks_l643_643845


namespace cost_comparison_l643_643379

def A (x : ℝ) : ℝ :=
  if x ≤ 100 then x
  else 100 + (x - 100) * 0.9

def B (x : ℝ) : ℝ :=
  if x ≤ 50 then x
  else 50 + (x - 50) * 0.95

theorem cost_comparison (x : ℝ) :
  (x ≤ 50 → A(x) = B(x)) ∧
  (50 < x ∧ x ≤ 100 → B(x) < A(x)) ∧
  (100 < x ∧ x < 150 → B(x) < A(x)) ∧
  (x > 150 → A(x) < B(x)) ∧
  (x = 150 → A(x) = B(x)) :=
by
  sorry

end cost_comparison_l643_643379


namespace cannot_achieve_55_cents_with_six_coins_l643_643654

theorem cannot_achieve_55_cents_with_six_coins :
  ¬∃ (a b c d e : ℕ), 
    a + b + c + d + e = 6 ∧ 
    a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 55 := 
sorry

end cannot_achieve_55_cents_with_six_coins_l643_643654


namespace contractor_engagement_days_l643_643035

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l643_643035


namespace radius_incircle_hyperbola_l643_643881

noncomputable def hyperbola_eqn : {a b : ℝ} → Prop :=
λ a b, a > 0 ∧ b > 0 ∧ (∃ e : ℝ, e = 2 * Real.sqrt 3 / 3) ∧
    (∃ A : ℝ × ℝ, A = (Real.sqrt 15 / 2, 1 / 2) ∧ 
      ((Real.sqrt A.1^2 / a^2) - (A.2^2 / b^2)) = 1) ∧
      (∃ F1 F2 : ℝ × ℝ, F1 = (-2, 0) ∧ F2 = (2, 0)) ∧
    ∃ r : ℝ, r = Real.sqrt 5 - Real.sqrt 3

theorem radius_incircle_hyperbola :
    ∃ r : ℝ, r = Real.sqrt 5 - Real.sqrt 3 := sorry

end radius_incircle_hyperbola_l643_643881


namespace count_of_valid_integers_l643_643570

-- Define the range of integers between 200 and 999.
def is_in_range (n : ℕ) : Prop := 200 ≤ n ∧ n ≤ 999

-- Define the property of having a permutation that is a multiple of 22.
def has_permutation_multiple_of_22 (n : ℕ) : Prop :=
  ∃ (m : ℕ), list.permutations n.digits = m ∧ (m % 22 = 0) ∧ is_in_range m

-- Define the main theorem to prove the number of integers with the required property.
theorem count_of_valid_integers : ∑ i in finset.Icc 200 999, has_permutation_multiple_of_22 i = 100 :=
sorry


end count_of_valid_integers_l643_643570


namespace beth_comic_books_percentage_l643_643448

/-- Definition of total books Beth owns -/
def total_books : ℕ := 120

/-- Definition of percentage novels in her collection -/
def percentage_novels : ℝ := 0.65

/-- Definition of number of graphic novels in her collection -/
def graphic_novels : ℕ := 18

/-- Calculation of the percentage of comic books she owns -/
theorem beth_comic_books_percentage (total_books : ℕ) (percentage_novels : ℝ) (graphic_novels : ℕ) : 
  (100 * ((total_books * (1 - percentage_novels) - graphic_novels) / total_books) = 20) :=
by
  let non_novel_books := total_books * (1 - percentage_novels)
  let comic_books := non_novel_books - graphic_novels
  let percentage_comic_books := 100 * (comic_books / total_books)
  have h : percentage_comic_books = 20 := sorry
  assumption

end beth_comic_books_percentage_l643_643448


namespace length_sixth_episode_l643_643247

def length_first_episode : ℕ := 58
def length_second_episode : ℕ := 62
def length_third_episode : ℕ := 65
def length_fourth_episode : ℕ := 71
def length_fifth_episode : ℕ := 79
def total_viewing_time : ℕ := 450

theorem length_sixth_episode :
  length_first_episode + length_second_episode + length_third_episode + length_fourth_episode + length_fifth_episode + 115 = total_viewing_time := by
  sorry

end length_sixth_episode_l643_643247


namespace vec_eq_problem_l643_643561

-- Define the vectors and the conditions for the problem
def a := (2, 1) : ℝ × ℝ
def b := (1, -2) : ℝ × ℝ
def eq_vec := (λ (m n : ℝ), (m * a.1 + n * b.1, m * a.2 + n * b.2))

-- The goal to prove
theorem vec_eq_problem (m n : ℝ) (h : eq_vec m n = (9, -8)) : m - n = -3 := by
  sorry

end vec_eq_problem_l643_643561


namespace cat_can_pass_through_gap_l643_643444

theorem cat_can_pass_through_gap (R : ℝ) (h : ℝ) (π : ℝ) (hπ : π = Real.pi)
  (L₀ : ℝ) (L₁ : ℝ)
  (hL₀ : L₀ = 2 * π * R)
  (hL₁ : L₁ = L₀ + 1)
  (hL₁' : L₁ = 2 * π * (R + h)) :
  h = 1 / (2 * π) :=
by
  sorry

end cat_can_pass_through_gap_l643_643444


namespace distribute_balls_into_boxes_l643_643209

theorem distribute_balls_into_boxes : 
  ∀ (balls : ℕ) (boxes : ℕ), balls = 6 ∧ boxes = 3 → boxes^balls = 729 :=
by
  intros balls boxes h
  have hb : balls = 6 := h.1
  have hbox : boxes = 3 := h.2
  rw [hb, hbox]
  show 3^6 = 729
  exact Nat.pow 3 6 -- this would expand to the actual computation
  sorry

end distribute_balls_into_boxes_l643_643209


namespace rationalization_sum_l643_643303

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalization_sum : rationalize_denominator = 75 := by
  sorry

end rationalization_sum_l643_643303


namespace interval_duration_l643_643334

-- Define the conditions given in the problem
def speed_1 := 45 -- speed in the first interval in mph
def decrease := 3 -- speed decrease per interval in mph
def distance_5 := 4.4 -- distance traveled in the fifth interval in miles
def speed_5 := speed_1 - 4 * decrease -- average speed in the fifth interval in mph

-- Prove that the duration of each interval t in minutes is equal to 8
theorem interval_duration (t : ℝ) : distance_5 / speed_5 = t → t * 60 = 8 :=
by
  sorry

end interval_duration_l643_643334


namespace total_pencils_sold_l643_643689

theorem total_pencils_sold:
  let first_two := 2 * 2 in
  let next_six := 6 * 3 in
  let last_two := 2 * 1 in
  first_two + next_six + last_two = 24 :=
by
  let first_two := 2 * 2
  let next_six := 6 * 3
  let last_two := 2 * 1
  calc
    first_two + next_six + last_two = 4 + 18 + 2 := by sorry
    ... = 24 := by sorry

end total_pencils_sold_l643_643689


namespace negation_of_prop_l643_643343

theorem negation_of_prop (P : ∀ x : ℝ, exp x > 0) :
  ¬ (∀ x : ℝ, exp x > 0) ↔ ∃ x : ℝ, exp x ≤ 0 :=
by
  unfold Exp
  sorry

end negation_of_prop_l643_643343


namespace sum_of_coordinates_l643_643909

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 3 = -2) : 
  let x := 3 / 2, y := -2.5 in
  (4 * y = 2 * f (2 * x) - 6) ∧ (x + y = -1) :=
by
  sorry

end sum_of_coordinates_l643_643909


namespace parallel_BP_DQ_l643_643871

-- Definitions of the points and their properties
variable (A B C D O E F G X Y Z L M N P Q : Type)
variable [geometry A B C] [geometry D O] [geometry E F] [geometry G]
variable [geometry X Y Z] [geometry L M N] [geometry P Q]

-- Conditions from part a
variable (h1 : ¬ is_square A B C D)
variable (h2 : on_perpendicular_bisector O B D)
variable (h3 : in_interior_triangle O B C D)
variable (h4 : E = second_intersection_circle O B D A B)
variable (h5 : F = second_intersection_circle O B D A D)
variable (h6 : G = intersection BF DE)
variable (h7 : X = foot_perpendicular G A B)
variable (h8 : Y = foot_perpendicular G B D)
variable (h9 : Z = foot_perpendicular G D A)
variable (h10 : L = foot_perpendicular O C D)
variable (h11 : M = foot_perpendicular O B D)
variable (h12 : N = foot_perpendicular O B C)
variable (h13 : P = intersection XY ML)
variable (h14 : Q = intersection YZ MN)

-- Theorem statement
theorem parallel_BP_DQ : parallel (B P) (D Q) := sorry

end parallel_BP_DQ_l643_643871


namespace raft_capacity_l643_643704

theorem raft_capacity (total_without_life_jackets : ℕ) (reduction_with_life_jackets : ℕ)
  (people_needing_life_jackets : ℕ) (total_capacity_with_life_jackets : ℕ)
  (no_life_jackets_capacity : total_without_life_jackets = 21)
  (life_jackets_reduction : reduction_with_life_jackets = 7)
  (life_jackets_needed : people_needing_life_jackets = 8) :
  total_capacity_with_life_jackets = 14 :=
by
  -- Proof should be here
  sorry

end raft_capacity_l643_643704


namespace length_of_AX_l643_643295

theorem length_of_AX
  (A B C D X : Point)
  (on_circle : OnCircle A B C D (Circle 1))
  (on_diameter : OnDiameter X AD)
  (AX_eq_DX : dist A X = dist D X)
  (angle_cond : 3 * angle A B C = angle B X C)
  (angle_BXC : angle B X C = 30) :
  dist A X = 1 / 2 :=
sorry

end length_of_AX_l643_643295


namespace sum_of_terms_in_geometric_sequence_l643_643959

theorem sum_of_terms_in_geometric_sequence :
  (∀ n : ℕ, a_n > 0) →  -- All terms are positive
  a_1 = 3 →  -- First term is 3
  a_1 + a_1 * q + a_1 * q^2 = 21 →  -- Sum of the first three terms is 21
  a_3 + a_4 + a_5 = 84 :=  -- Sum of the third, fourth, and fifth terms is 84
by {
  intros hpos h1 hsum,
  set q := 2 with hq,
  have h_a3 : a_3 = a_1 * q^2,
  have h_a4 : a_4 = a_1 * q^3,
  have h_a5 : a_5 = a_1 * q^4,
  linarith,
}

end sum_of_terms_in_geometric_sequence_l643_643959


namespace max_62_transfers_l643_643958

theorem max_62_transfers (n : ℕ) (d : ℕ) (h1 : n = 1993) (h2 : d = 93)
  (h3 : ∀ (G : SimpleGraph (Fin n)), (∀ v, 93 ≤ G.degree v) → G.connected) :
  ∀ (G : SimpleGraph (Fin n)), (∀ v, d ≤ G.degree v) → G.connected → ∃ (k : ℕ), k ≤ 62 ∧ 
  (∀ u v, u ≠ v → fin.find (G.path_length u v) ≤ k) := 
by sorry

end max_62_transfers_l643_643958


namespace number_of_acceptable_teams_l643_643763

noncomputable def basketball_team : Finset (Finset ℕ) :=
  (Finset.range 12).powerset.filter (λ s, 
    s.card = 5 ∧ 
    (2 ∉ s ∨ 3 ∉ s) ∧ 
    (4 ∉ s ∨ 5 ∉ s))

theorem number_of_acceptable_teams : 
  basketball_team.card = 560 :=
sorry

end number_of_acceptable_teams_l643_643763


namespace Ashis_height_more_than_Babji_height_l643_643073

-- Definitions based on conditions
variables {A B : ℝ}
-- Condition expressing the relationship between Ashis's and Babji's height
def Babji_height (A : ℝ) : ℝ := 0.80 * A

-- The proof problem to show the percentage increase
theorem Ashis_height_more_than_Babji_height :
  B = Babji_height A → (A - B) / B * 100 = 25 :=
sorry

end Ashis_height_more_than_Babji_height_l643_643073


namespace main_problem_l643_643063

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ x y ∈ I, x < y → f y ≤ f x

def f1 (x : ℝ) : ℝ := 3 ^ |x|
def f2 (x : ℝ) : ℝ := x ^ 3
def f3 (x : ℝ) : ℝ := log (1 / |x|)
def f4 (x : ℝ) : ℝ := x ^ (4 / 3)
def f5 (x : ℝ) : ℝ := -x ^ 2 + 1

theorem main_problem :
  (is_even f3 ∧ is_monotonically_decreasing f3 {x | 0 < x}) ∧ 
  (is_even f5 ∧ is_monotonically_decreasing f5 {x | 0 < x}) ∧ 
  (¬ (is_even f1 ∧ is_monotonically_decreasing f1 {x | 0 < x})) ∧ 
  (¬ (is_even f2 ∧ is_monotonically_decreasing f2 {x | 0 < x})) ∧ 
  (¬ (is_even f4 ∧ is_monotonically_decreasing f4 {x | 0 < x})) :=
by sorry

end main_problem_l643_643063


namespace number_of_solutions_eq1_l643_643468

-- Definitions:
def eq1 (x : ℝ) : Prop :=
  (real.sqrt (9 - x) = x^2 * real.sqrt (9 - x))

-- Theorem statement:
theorem number_of_solutions_eq1 : 
  (finset.filter (λ x, eq1 x) (finset.range 10)).card = 3 :=
begin
  -- Proof goes here
  sorry
end

end number_of_solutions_eq1_l643_643468


namespace bowling_competition_award_sequences_l643_643240

theorem bowling_competition_award_sequences :
  (number_of_award_sequences : ℕ) =
  let fifth_and_fourth_outcomes := 2 in
  let winner_vs_third_outcomes := 2 in
  let winner_vs_second_outcomes := 2 in
  let winner_vs_first_outcomes := 2 in
  fifth_and_fourth_outcomes * winner_vs_third_outcomes * winner_vs_second_outcomes * winner_vs_first_outcomes = 16 := sorry

end bowling_competition_award_sequences_l643_643240


namespace cistern_empty_time_time_for_leaks_to_empty_cistern_l643_643414

theorem cistern_empty_time (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1/8 - 1/x - 1/y = 1/12) -> (1/x + 1/y = 1/24) :=
by
  intro h
  have h1 : 1/8 - 1/12 = 1/24 := by norm_num
  rw ←h1 at h
  linarith

theorem time_for_leaks_to_empty_cistern (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1/8 - 1/x - 1/y = 1/12 -> 24 = 1 / (1/x + 1/y) :=
by
  intro h
  have h_rate : 1/x + 1/y = 1/24 := cistern_empty_time x y hx hy h
  rw h_rate
  norm_num
  sorry

end cistern_empty_time_time_for_leaks_to_empty_cistern_l643_643414


namespace molecular_weight_correct_l643_643723

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_N_in_N2O3 : ℕ := 2
def num_O_in_N2O3 : ℕ := 3

def molecular_weight_N2O3 : ℝ :=
  (num_N_in_N2O3 * atomic_weight_N) + (num_O_in_N2O3 * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_N2O3 = 76.02 := by
  sorry

end molecular_weight_correct_l643_643723


namespace f_difference_l643_643525

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom local_f : ∀ x : ℝ, -2 < x ∧ x < 0 → f x = 2^x

-- State the problem
theorem f_difference :
  f 2012 - f 2011 = -1 / 2 := sorry

end f_difference_l643_643525


namespace slope_angle_of_line_l643_643351

theorem slope_angle_of_line (h : ∀ x y : ℝ, 3 * x + sqrt 3 * y + 1 = 0) : 
  ∃ θ : ℝ, θ = 120 := 
sorry

end slope_angle_of_line_l643_643351


namespace count_integers_in_range_l643_643862

theorem count_integers_in_range : 
  {n : ℤ | 20 < n^2 ∧ n^2 < 200}.to_finset.card = 20 :=
sorry

end count_integers_in_range_l643_643862


namespace sum_of_coefficients_of_y_terms_l643_643391

theorem sum_of_coefficients_of_y_terms :
  let p := (5 * x + 3 * y + 2) * (2 * x + 6 * y + 7)
  let expanded_p := 10 * x^2 + 36 * x * y + 39 * x + 18 * y^2 + 33 * y + 14
  (36 + 18 + 33) = 87 := by
  sorry

end sum_of_coefficients_of_y_terms_l643_643391


namespace equilateral_triangle_length_CI_l643_643374

theorem equilateral_triangle_length_CI {A B C M I E : Type} (hEquilateral : ∀ x y z, eq_triangle A B C 3)
  (hMidpoint : midpoint M B C) (hAngle1 : angle AM I = 90) (hAngle2 : angle A E M = 90)
  (hArea : triangle_area E M I = sqrt 3) :
  let a := 3
  let b := 2
  let c := 2
  length_CI A C I = (3 - sqrt 2) / 2 ∧ a + b + c = 7 :=
sorry

end equilateral_triangle_length_CI_l643_643374


namespace y_intercept_of_line_eq_2x_plus_1_l643_643699

theorem y_intercept_of_line_eq_2x_plus_1 :
  ∃ y : ℝ, y = 1 ∧ (∀ x : ℝ, y = 2 * x + 1 → x = 0) :=
by
  use 1
  split
  · refl
  · intros x h
    sorry

end y_intercept_of_line_eq_2x_plus_1_l643_643699


namespace least_number_to_add_l643_643341

theorem least_number_to_add (a b c d : ℕ) (x : ℕ) : 
  x = a + b + c + d → (7 * 11 * 13 * 17) - (59789 % x) = 16142 :=
by
  intros LCM_eq
  have LCM : ℕ := 7 * 11 * 13 * 17
  have H : 59789 % LCM = 875
  exact Nat.sub_eq_iff_eq_add.mpr H sorry

end least_number_to_add_l643_643341


namespace solve_y_l643_643313

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l643_643313


namespace distance_equals_expected_value_l643_643084

noncomputable def a : ℝ × ℝ := (3, -1)
noncomputable def b : ℝ × ℝ := (7, -3)
noncomputable def d : ℝ × ℝ := (2, -4)
noncomputable def e : ℝ × ℝ := (-1, 1)

def distance_between_parallel_lines (a b d e : ℝ × ℝ) : ℝ :=
  let v := (a.1 - b.1, a.2 - b.2)
  let proj_v_onto_d := let c := (v.1 * d.1 + v.2 * d.2) / (d.1 * d.1 + d.2 * d.2)
                      (c * d.1, c * d.2)
  let c := (b.1 + proj_v_onto_d.1, b.2 + proj_v_onto_d.2)
  Math.sqrt ((a.1 - c.1) ^ 2 + (a.2 - c.2) ^ 2)

theorem distance_equals_expected_value : 
  distance_between_parallel_lines a b d e = 4 * (Math.sqrt 10) / 5 :=
sorry

end distance_equals_expected_value_l643_643084


namespace complex_ordered_pairs_l643_643116

theorem complex_ordered_pairs :
  { (a, b : ℂ) // a^4 * b^7 = 1 ∧ a^8 * b^3 = 1 }.to_finset.card = 64 := 
sorry

end complex_ordered_pairs_l643_643116


namespace problem1_problem2_l643_643596

/-- 
  Problem 1: Given a binomial random variable X with n = 6 and p = 1/2, 
  prove that P(X ≤ 2) = 11/32.
-/
theorem problem1 (X : ℕ → ℝ) (n : ℕ) (p : ℝ) (hx : X ~ binomial 6 (1/2)) :  
    P(X ≤ 2) = 11/32 := 
    sorry

/-- 
  Problem 2: Given a binomial random variable X with p = 1/2, 
  prove that to have at least 98% confidence that 0.4n ≤ X ≤ 0.6n,
  the minimum n must be 1250.
-/
theorem problem2 (X : ℕ → ℝ) (p : ℝ) (hxp : X ~ binomial n (1/2)) :
    (∀ a ≥ 0, P( | X - 0.5 * n | ≤ a ) ≥ 1 - 0.25 * n / (0.1 * n)^2 → 1 - 0.25 * n / (0.1 * n)^2 ≥ 0.98 → n ≥ 1250) :=
    sorry

end problem1_problem2_l643_643596


namespace increasing_intervals_max_min_values_l643_643181

noncomputable def f (x : ℝ) : ℝ := (sqrt 3 / 2) * sin (2 * x + π / 3)

theorem increasing_intervals (k : ℤ) :
  ∀ x y, x ∈ set.Icc (k * π - 5 * π / 3) (k * π + π / 12) → f'(x) > 0 := 
begin
  sorry -- Proof to show f(x) is increasing in these intervals
end

theorem max_min_values :
  let a := -π / 12, b := 5 * π / 12 in
  ∀ x, x ∈ set.Icc a b → -sqrt 3 / 4 ≤ f(x) ∧ f(x) ≤ sqrt 3 / 2 :=
begin
  sorry -- Proof to show that the max and min values of f(x) in the interval [-π/12, 5π/12] are √3/2 and -√3/4, respectively
end

end increasing_intervals_max_min_values_l643_643181


namespace expression_value_l643_643199

def a : ℤ := 5
def b : ℤ := -3
def c : ℕ := 2

theorem expression_value : (3 * c) / (a + b) + c = 5 := by
  sorry

end expression_value_l643_643199


namespace cyclic_quadrilateral_of_equal_circumradii_tris_l643_643050

variable {A B C D O K L M N : Type} [EuclideanGeometry A B C D O K L M N]

theorem cyclic_quadrilateral_of_equal_circumradii_tris
  (h_convex : ConvexQuadrilateral A B C D)
  (h_interior : InsideQuadrilateral O A B C D)
  (h_segments : SegmentsConnecting O A B C D K L M N)
  (h_equal_circumradii : ∀ T ∈ {(O, A, K), (O, B, K), (O, B, L), (O, C, L), (O, C, M), (O, D, M), (O, D, N), (O, A, N)}, 
                         Circumradius T = c) :
  CyclicQuadrilateral A B C D :=
by
  sorry

end cyclic_quadrilateral_of_equal_circumradii_tris_l643_643050


namespace vector_calculation_l643_643880

def vector_a := (1, 2)
def vector_b := (2, -1)
def result_vector := (2 * 1 - 2, 2 * 2 + 1)

theorem vector_calculation :
    2 * vector_a - vector_b = (0, 5) := by
    calc
        2 * (1, 2) - (2, -1) = (2 * 1, 2 * 2) - (2, -1) : rfl
                          ... = (2, 4) - (2, -1)       : rfl
                          ... = (2 - 2, 4 - (-1))     : rfl
                          ... = (0, 5)                : rfl

end vector_calculation_l643_643880


namespace shaded_sectors_area_half_circle_area_l643_643500

-- Definition: Point on a plane
constant Point : Type

-- Conditions:
constant O : Point      -- Intersection point of the four lines
constant circle_center : Point -- Center of the circle
constant r : ℝ          -- Radius of the circle (assumed positive)
constant angle : ℝ      -- Angle of each sector, given as 45 degrees

def angle_is_45_degrees : Prop := angle = π / 4

-- Question: Prove the area of shaded sectors is exactly half of the circle's area
theorem shaded_sectors_area_half_circle_area
  (O : Point)
  (circle_center : Point)
  (r : ℝ)
  (angle : ℝ)
  (h_angle : angle_is_45_degrees) :
  let area_of_circle := π * r^2 in
  let shaded_sectors_area := area_of_circle / 2 in
  shaded_sectors_area = π * r^2 / 2 := 
sorry

end shaded_sectors_area_half_circle_area_l643_643500


namespace irrational_of_sqrt8_l643_643004

theorem irrational_of_sqrt8 (h1 : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (2 : ℤ) / 5 = a / b)
  (h2 : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (-3 : ℤ) = a / b)
  (h3 : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ 0 = a / b) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ real.sqrt 8 = a / b :=
begin
  sorry
end

end irrational_of_sqrt8_l643_643004


namespace tangent_line_equation_at_1_l643_643159

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Given conditions
variable {f : ℝ → ℝ}
variable (odd_f : odd_function f)
variable (h : ∀ x, x < 0 → f x = log (-x) - 3 * x)

-- Mathematical equivalent proof problem
theorem tangent_line_equation_at_1 :
  f 1 = -3 ∧ (∀ x > 0, f x = -log x - 3 * x) ∧ (∀ x > 0, deriv f x = -1 / x - 3) → 
  4 * 1 + (-3) - 1 = 0 :=
by
  sorry

end tangent_line_equation_at_1_l643_643159


namespace delivery_sequences_12_equals_778_l643_643424

def permissible_sequences (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else permissible_sequences (n-1) + permissible_sequences (n-2) + permissible_sequences (n-3)

theorem delivery_sequences_12_equals_778 : permissible_sequences 12 = 778 :=
  sorry

end delivery_sequences_12_equals_778_l643_643424


namespace solution_set_l643_643772

noncomputable def f : ℝ → ℝ := sorry

axiom f_deriv : ∀ x : ℝ, deriv f x = f' x
axiom f_condition1 : ∀ x : ℝ, f(x) + f' x > 1
axiom f_condition2 : f 0 = 2018

theorem solution_set (x : ℝ) : (e^x * f(x) - e^x > 2017) ↔ (0 < x) :=
by sorry

end solution_set_l643_643772


namespace div_condition_positive_integers_l643_643994

theorem div_condition_positive_integers 
  (a b d : ℕ) 
  (h1 : a + b ≡ 0 [MOD d]) 
  (h2 : a * b ≡ 0 [MOD d^2]) 
  (h3 : 0 < a) 
  (h4 : 0 < b) 
  (h5 : 0 < d) : 
  d ∣ a ∧ d ∣ b :=
sorry

end div_condition_positive_integers_l643_643994


namespace solve_for_y_l643_643316

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l643_643316


namespace problem_statement_l643_643577

noncomputable theory

def sums_to_nineteen (a b c d : ℕ) : Prop :=
  d + b = 10 ∧ d + c = 9 ∧ a + c = 9 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = 19

theorem problem_statement (a b c d : ℕ) (h : sums_to_nineteen a b c d) : a + b + c + d = 19 :=
by {
  sorry
}

end problem_statement_l643_643577


namespace mary_sheep_problem_l643_643285

theorem mary_sheep_problem :
  let initial_sheep := 400
  in let sheep_given_to_sister := initial_sheep / 4
  in let remaining_after_sister := initial_sheep - sheep_given_to_sister
  in let sheep_given_to_brother := remaining_after_sister / 2
  in let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  in sheep_remaining = 150 :=
by
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let remaining_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := remaining_after_sister / 2
  let sheep_remaining := remaining_after_sister - sheep_given_to_brother
  show sheep_remaining = 150 from sorry

end mary_sheep_problem_l643_643285


namespace train_length_l643_643055

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) (train_length_m : ℝ) :
  speed_kmh = 72 ->
  time_s = 16.5986721062315 ->
  bridge_length_m = 132 ->
  -- Convert speed from km/hr to m/s
  let speed_ms := speed_kmh * (1000 / 3600) in
  -- Calculate total distance covered
  let total_distance_m := speed_ms * time_s in
  -- Length of train is total distance minus bridge length
  train_length_m = total_distance_m - bridge_length_m ->
  train_length_m = 200 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, ← h4]
  sorry

end train_length_l643_643055


namespace min_words_90_percent_l643_643496

noncomputable def min_words_to_learn (total_words learned_words : ℕ) (percent_guessed_correct : ℚ) : ℕ :=
  let correct_words := learned_words + percent_guessed_correct * (total_words - learned_words)
  if correct_words / total_words ≥ (90 : ℚ) / 100 then learned_words
  else min_words_to_learn total_words (learned_words + 1) percent_guessed_correct

theorem min_words_90_percent (total_words : ℕ) (percent_guessed_correct : ℚ) (correct_threshold : ℚ)
  (h_total_words : total_words = 800)
  (h_percent_guessed_correct : percent_guessed_correct = 0.1)
  (h_correct_threshold : correct_threshold = 0.9) :
  min_words_to_learn total_words 0 percent_guessed_correct = 712 := sorry

end min_words_90_percent_l643_643496


namespace mary_sheep_remaining_l643_643283

theorem mary_sheep_remaining : 
  let initial_sheep := 400
  let sheep_given_to_sister := initial_sheep / 4
  let sheep_after_sister := initial_sheep - sheep_given_to_sister
  let sheep_given_to_brother := sheep_after_sister / 2
  let remaining_sheep := sheep_after_sister - sheep_given_to_brother
  remaining_sheep = 150 :=
by
  assume initial_sheep := 400
  have sheep_given_to_sister := initial_sheep / 4
  have sheep_after_sister := initial_sheep - sheep_given_to_sister
  have sheep_given_to_brother := sheep_after_sister / 2
  have remaining_sheep := sheep_after_sister - sheep_given_to_brother
  show remaining_sheep = 150
  sorry

end mary_sheep_remaining_l643_643283


namespace min_operations_l643_643834

-- Define the figure made up of 3n^2 rhombuses
def figure (n : ℕ) : Type := sorry

-- Define the operations allowed on the figures
def operation (f : figure n) : figure n := sorry

-- Define the problem statement
theorem min_operations (n : ℕ) : 
  (∃f₁ f₂ : figure n, f₂ = transform_to_rear_faces f₁) → 
  (∃ o : ℕ, o = n^3) := sorry

end min_operations_l643_643834


namespace claudia_weekend_earnings_l643_643824

/-- Claudia charges $10.00 for her one-hour class. 20 kids attend the Saturday’s class.
    Half that many attend the Sunday’s class. Prove Claudia makes $300.00 over the weekend. -/
theorem claudia_weekend_earnings : 
  let charge_per_kid := 10
  let saturday_attendees := 20
  let sunday_attendees := saturday_attendees / 2
  let total_attendees := saturday_attendees + sunday_attendees
  let total_earnings := charge_per_kid * total_attendees
  total_earnings = 300 := 
by 
  simp
  sorry

end claudia_weekend_earnings_l643_643824


namespace radius_of_circumscribed_sphere_l643_643892

-- Condition: SA = 2
def SA : ℝ := 2

-- Condition: SB = 4
def SB : ℝ := 4

-- Condition: SC = 4
def SC : ℝ := 4

-- Condition: The three side edges are pairwise perpendicular.
def pairwise_perpendicular : Prop := true -- This condition is described but would require geometric definition.

-- To prove: Radius of circumscribed sphere is 3
theorem radius_of_circumscribed_sphere : 
  ∀ (SA SB SC : ℝ) (pairwise_perpendicular : Prop), SA = 2 → SB = 4 → SC = 4 → pairwise_perpendicular → 
  (3 : ℝ) = 3 := by 
  intros SA SB SC pairwise_perpendicular h1 h2 h3 h4
  sorry

end radius_of_circumscribed_sphere_l643_643892


namespace count_integers_satisfying_inequality_l643_643859

theorem count_integers_satisfying_inequality : 
  (∃ n : ℤ, (20 < n^2) ∧ (n^2 < 200)) → (finset.Icc (-14 : ℤ) 14).filter (λ n, 20 < n^2 ∧ n^2 < 200).card = 20 := 
by
  sorry

end count_integers_satisfying_inequality_l643_643859


namespace complex_conjugate_product_l643_643223

noncomputable def z : ℂ := (1 : ℂ) + 2 * complex.I
noncomputable def z_conj : ℂ := complex.conj z

theorem complex_conjugate_product : z * z_conj = 5 := by
  unfold z z_conj
  rw [complex.conj_eq_re_sub_im]
  norm_num
  sorry

end complex_conjugate_product_l643_643223


namespace range_mn_squared_l643_643464

-- Let's define the conditions in Lean

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is strictly increasing
axiom h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0

-- Condition 2: f(x-1) is centrally symmetric about (1,0)
axiom h2 : ∀ x : ℝ, f (x - 1) = - f (2 - (x - 1))

-- Condition 3: Given inequality
axiom h3 : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0

-- Prove the range for m^2 + n^2 is (9, 49)
theorem range_mn_squared : ∀ m n : ℝ, f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0 →
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
sorry

end range_mn_squared_l643_643464


namespace initial_apples_l643_643298

variable picked_apples : ℕ
variable apples_on_tree : ℕ

theorem initial_apples (picked_apples : ℕ) (apples_on_tree : ℕ) : (picked_apples = 4) → (apples_on_tree = 3) → (picked_apples + apples_on_tree = 7) :=
by
  intros h1 h2
  rw [h1, h2]
  rfl

end initial_apples_l643_643298


namespace tan_sum_eq_l643_643538

theorem tan_sum_eq (α β : ℝ) 
  (z : ℂ) (u : ℂ)
  (hz : z = Complex.ofReal (cos α) + Complex.i * Complex.ofReal (sin α))
  (hu : u = Complex.ofReal (cos β) + Complex.i * Complex.ofReal (sin β))
  (hzu : z + u = Complex.ofReal (4/5) + Complex.i * Complex.ofReal (3/5)):
  Real.tan (α + β) = 6 / 7
  :=
sorry

end tan_sum_eq_l643_643538


namespace sin_2alpha_eq_fraction_l643_643158

def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 6) + a * Real.cos (2 * x)

theorem sin_2alpha_eq_fraction (a : ℝ) (α : ℝ) (h1 : a = -2) (h2 : α ∈ Set.Ioo 0 (π / 3)) (h3 : f α (-2) = 6 / 5) :
  Real.sin (2 * α) = (4 + 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_2alpha_eq_fraction_l643_643158
