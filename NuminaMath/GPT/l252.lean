import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Ring
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Convex
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Geometry.Ellipse
import Mathlib.Analysis.MetricSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Binomial
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Cast
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Cast
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Ellipse
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle
import Mathlib.Mathlib
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.Probability
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Topology.Continuity
import data.real.basic

namespace root_exists_l252_252146

noncomputable def f (x : ℝ) := log10 x + x - 3

theorem root_exists :
  ∃ (x : ℝ), (abs (f 2.6)) < 0.1 :=
begin
  sorry
end

end root_exists_l252_252146


namespace number_of_boys_in_contest_l252_252602

variables (F M : ℕ)

theorem number_of_boys_in_contest (h1 : F + M = 100) (h2 : F ≥ 9) 
  (h3 : ∀ (group : finset ℕ), group.card = 10 → ∃ x ∈ group, x < F) :
  M = 91 :=
sorry

end number_of_boys_in_contest_l252_252602


namespace minimum_cuts_to_divide_obtuse_triangle_into_acute_triangles_l252_252620

-- Definitions and conditions of the triangle and points
def obtuse_triangle (A B C : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] :
  Prop := ∃ A B C : Type, (∠ A = 120) ∨ (∠ B = 120) ∨ (∠ C = 120)

def points_on_triangle (A D G B E F C : Type) [DecidableEq A] [DecidableEq D] 
  [DecidableEq G] [DecidableEq B] [DecidableEq E] [DecidableEq F] [DecidableEq C] :
  Prop := D ∈ segment A B ∧ G ∈ segment A C ∧ E ∈ segment B C ∧ F ∈ segment B C

def perpendicular_to_incircle (O : Type) [DecidableEq O] (A B C D E F G : Type)
  [DecidableEq D] [DecidableEq G] [DecidableEq E] [DecidableEq F] :
  Prop := perpendicular (line D E) (line O B) ∧ perpendicular (line F G) (line O C)

def minimum_straight_line_cuts (A B C D E F G O : Type) [DecidableEq A] 
  [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E] [DecidableEq F]
  [DecidableEq G] [DecidableEq O] : Prop :=
  ∃ cuts : Type, cuts = 7 ∧ obtuse_triangle A B C ∧ points_on_triangle A D G B E F C ∧
  perpendicular_to_incircle O A B C D E F G

-- Statement of the math proof problem
theorem minimum_cuts_to_divide_obtuse_triangle_into_acute_triangles (A B C D E F G O : Type)
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E]
  [DecidableEq F] [DecidableEq G] [DecidableEq O] :
  obtuse_triangle A B C → points_on_triangle A D G B E F C →
  perpendicular_to_incircle O A B C D E F G →
  minimum_straight_line_cuts A B C D E F G O :=
by
  intros
  sorry

end minimum_cuts_to_divide_obtuse_triangle_into_acute_triangles_l252_252620


namespace find_dimension_l252_252095

-- Definitions of the dimensions, cost per square foot, and total cost
def width : ℕ := 4
def height : ℕ := 5
def costPerSquareFoot : ℕ := 20
def totalCost : ℕ := 1880

-- Surface area of the rectangular tank
def surfaceArea (x : ℕ) : ℕ :=
  2 * (width * height) + 2 * (width * x) + 2 * (height * x)

-- Given that the total cost equals 1880 and each square foot costs $20
def givenSurfaceArea : ℕ := totalCost / costPerSquareFoot

-- The proof statement
theorem find_dimension (x : ℕ) :
  givenSurfaceArea = surfaceArea x → x = 3 :=
by
  sorry

end find_dimension_l252_252095


namespace problem_statement_ellipse_problem_statement_distance_l252_252549

noncomputable def ellipse_foci (a b: ℝ) (h : a > b > 0) : ℝ := real.sqrt (a^2 - b^2)

theorem problem_statement_ellipse (x y a b : ℝ) (h : a > b > 0) (right_focus : (ellipse_foci a b h = sqrt 3))
(point_condition : (-1 / 4 + y ^ 2 / 1 = 1 | y = sqrt 3 / 2)):
  (a ^ 2 = 4) ∧ (b ^ 2 = 1) ∧ ((x^2 / 4) + y^2 = 1) := sorry

structure LineThroughM (l : ℝ → ℝ := λ y, t * y + m) :=
(m t: ℝ)

structure Ellipse (β: ℝ) (eq_ellipse: ∀ x y, β * x^2 + y^2 = 1)

def Circle : (x y : ℝ) := x^2 + y^2 = 4 / 7

theorem problem_statement_distance (t m : ℝ) (d : ℝ):
  (t ^ 2 = 7 / 4 * m ^ 2 - 1) →
  (m ^ 2 = 4 / 3) →
  d = abs m / sqrt (1 + t ^ 2) →
  d = sqrt (4 / 7) →
  abs m = 4 / 3 →
  t = 2 * sqrt (7) * m →
  d = (4 * sqrt 21 / 21) :=
 sorry

end problem_statement_ellipse_problem_statement_distance_l252_252549


namespace find_prime_sets_l252_252585

theorem find_prime_sets :
  ∃ (p1 p2 p3 p4 : ℕ), 
    prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ 
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ 
    p1 * p2 + p2 * p3 + p3 * p4 + p4 * p1 = 882 ∧
    ((p1, p2, p3, p4) = (2, 5, 19, 37) ∨ 
     (p1, p2, p3, p4) = (2, 11, 19, 31) ∨ 
     (p1, p2, p3, p4) = (2, 13, 19, 29)) := 
by
  sorry

end find_prime_sets_l252_252585


namespace min_value_m_n_l252_252515

noncomputable def f (x : ℝ) : ℝ := log x / log 2 - log (2 : ℝ) / log (2 : ℝ)

theorem min_value_m_n {m n : ℝ} (h1 : f m + f (2 * n) = 3) 
  (h2 : m > 2) (h3 : n > 1) : m + n ≥ 7 := 
sorry

end min_value_m_n_l252_252515


namespace average_of_side_lengths_of_squares_l252_252697

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252697


namespace find_angle_A_l252_252245

theorem find_angle_A (a b c : ℝ) (h : a^2 - c^2 = b^2 - b * c) : 
  ∃ (A : ℝ), A = π / 3 :=
by
  sorry

end find_angle_A_l252_252245


namespace round_to_nearest_whole_number_l252_252340

theorem round_to_nearest_whole_number (x : ℝ) (h1 : 7463.4997 = x) (h2 : ∀ y : ℝ, 0 ≤ y ∧ y < 0.5 → round_down y = 0) : 
  round x = 7463 :=
by
  sorry

end round_to_nearest_whole_number_l252_252340


namespace reading_minutes_per_disc_l252_252092

-- Define the total reading time
def total_reading_time := 630

-- Define the maximum capacity per disc
def max_capacity_per_disc := 80

-- Define the allowable unused space
def max_unused_space := 4

-- Define the effective capacity of each disc
def effective_capacity_per_disc := max_capacity_per_disc - max_unused_space

-- Define the number of discs needed, rounded up as a ceiling function
def number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)

-- Theorem statement: Each disc will contain 70 minutes of reading if all conditions are met
theorem reading_minutes_per_disc : ∀ (total_reading_time : ℕ) (max_capacity_per_disc : ℕ) (max_unused_space : ℕ)
  (effective_capacity_per_disc := max_capacity_per_disc - max_unused_space) 
  (number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)), 
  number_of_discs = 9 → total_reading_time / number_of_discs = 70 :=
by
  sorry

end reading_minutes_per_disc_l252_252092


namespace find_a_monotonic_intervals_and_axis_of_symmetry_l252_252963

noncomputable def f (x a : ℝ) : ℝ := sin (x + π / 6) + sin (x - π / 6) + cos x + a

theorem find_a (h : ∀ x, f x 3 ≥ 1) : 3 = 3 :=
sorry

theorem monotonic_intervals_and_axis_of_symmetry :
  (∀ k : ℤ, ∀ x : ℝ, -2 * π / 3 + 2 * k * π ≤ x ∧ x ≤ π / 3 + 2 * k * π → 2 * sin (x + π / 6) + 3 = 2 * sin (x + π / 6) + 3) ∧
  (∀ k : ℤ, ∀ x : ℝ, -5 * π / 3 + 2 * k * π ≤ x ∧ x ≤ -2 * π / 3 + 2 * k * π → 2 * sin (x + π / 6) + 3 = 2 * sin (x + π / 6) + 3) ∧
  (∀ k : ℤ, ∀ x : ℝ, x = π / 3 + k * π → 2 * sin (x + π / 6) + 3 = 2 * sin (x + π / 6) + 3) :=
sorry

end find_a_monotonic_intervals_and_axis_of_symmetry_l252_252963


namespace total_visible_area_l252_252419

variable {x : ℝ}

def area_large_rectangle (x : ℝ) : ℝ := (x + 8) * (x + 6)

def area_hole (x : ℝ) : ℝ := (2x - 4) * (x - 3)

def area_adjacent_rectangle (x : ℝ) : ℝ := (x + 2) * x

theorem total_visible_area (x : ℝ) : 
  (area_large_rectangle x) - (area_hole x) + (area_adjacent_rectangle x) = 26 * x + 36 := 
  by
  sorry

end total_visible_area_l252_252419


namespace sum_of_factors_of_24_l252_252041

theorem sum_of_factors_of_24 : 
  ∑ i in (finset.filter (λ x, 24 % x = 0) (finset.range (24 + 1))), i = 60 :=
sorry

end sum_of_factors_of_24_l252_252041


namespace max_n_value_l252_252518

theorem max_n_value (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
by
  sorry

end max_n_value_l252_252518


namespace average_side_lengths_l252_252682

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252682


namespace problem_statement_l252_252513

open Locale.Log

theorem problem_statement
  (a b : ℝ)
  (ha : a = log 25)
  (hb : b = log 36) :
  5^(a / b) + 6^(b / a) = 11 :=
sorry

end problem_statement_l252_252513


namespace can_all_ratings_occur_l252_252403

inductive Weather
  | Rain : Weather
  | NoRain : Weather

open Weather

structure Day where
  morning : Weather
  afternoon : Weather
  evening : Weather

def first_student_marks (d : Day) : String :=
  if d.morning = Rain ∨ d.afternoon = Rain ∨ d.evening = Rain then "-" else "+"

def second_student_marks (d : Day) : String :=
  if d.morning = NoRain ∨ d.afternoon = NoRain ∨ d.evening = NoRain then "+" else "-"

theorem can_all_ratings_occur (days : List Day) :
  (∀ (d : Day), (first_student_marks d = "+" ∧ second_student_marks d = "+") ∨
                (first_student_marks d = "+" ∧ second_student_marks d = "-") ∨
                (first_student_marks d = "-" ∧ second_student_marks d = "+") ∨
                (first_student_marks d = "-" ∧ second_student_marks d = "-")) :=
by
  sorry

end can_all_ratings_occur_l252_252403


namespace monthly_income_l252_252173

variable {I : ℝ} -- George's monthly income

def donated_to_charity (I : ℝ) := 0.60 * I -- 60% of the income left
def paid_in_taxes (I : ℝ) := 0.75 * donated_to_charity I -- 75% of the remaining income after donation
def saved_for_future (I : ℝ) := 0.80 * paid_in_taxes I -- 80% of the remaining income after taxes
def expenses (I : ℝ) := saved_for_future I - 125 -- Remaining income after groceries and transportation expenses
def remaining_for_entertainment := 150 -- $150 left for entertainment and miscellaneous expenses

theorem monthly_income : I = 763.89 := 
by
  -- Using the conditions of the problem
  sorry

end monthly_income_l252_252173


namespace minimum_sum_of_edge_labels_l252_252182

-- Definitions of the problem conditions
def regular_n_gon (n : ℕ) : Type := list ℤ

def edge_label (i j : ℕ) : ℕ :=
  abs (i - j)

def sum_of_edge_labels (n : ℕ) (vertices : regular_n_gon n) :=
  (list.sum ((list.finRange (n+1)).product (list.finRange (n+1))).map (λ (ij : ℕ × ℕ), edge_label ij.1 ij.2))

-- The conjecture to prove
theorem minimum_sum_of_edge_labels (n : ℕ) (vertices : regular_n_gon n) :
  let m := n / 2 in
  if even n then
    sum_of_edge_labels n vertices ≥ m * (m + 5)
  else
    let m := (n - 1) / 2 in sum_of_edge_labels n vertices ≥ 4 * m + 2 + (m + 1) ^ 2 :=
sorry

end minimum_sum_of_edge_labels_l252_252182


namespace trapezoid_problem_l252_252185

variables (ABCD : Type) [IsTrapezoid ABCD]
variables {A B C D : ABCD} {M N K : Point}
variables (hCM_MD : CM / MD = 4 / 3)
variables (hCN_NA : CN / NA = 4 / 3)
noncomputable def ratio_AD_BC : ℚ := 7 / 12

theorem trapezoid_problem :
  Ratio (AD / BC) = ratio_AD_BC :=
by
  -- Proof to be filled in here
  sorry

end trapezoid_problem_l252_252185


namespace dorothy_and_jemma_sales_l252_252473

theorem dorothy_and_jemma_sales :
  ∀ (frames_sold_by_jemma price_per_frame_jemma : ℕ)
  (price_per_frame_dorothy frames_sold_by_dorothy : ℚ)
  (total_sales_jemma total_sales_dorothy total_sales : ℚ),
  price_per_frame_jemma = 5 →
  frames_sold_by_jemma = 400 →
  price_per_frame_dorothy = price_per_frame_jemma / 2 →
  frames_sold_by_jemma = 2 * frames_sold_by_dorothy →
  total_sales_jemma = frames_sold_by_jemma * price_per_frame_jemma →
  total_sales_dorothy = frames_sold_by_dorothy * price_per_frame_dorothy →
  total_sales = total_sales_jemma + total_sales_dorothy →
  total_sales = 2500 := by
  sorry

end dorothy_and_jemma_sales_l252_252473


namespace max_matches_without_triangles_l252_252872

open Finset

-- Define the statement
theorem max_matches_without_triangles (n : ℕ) (h_n : n = 8)
  (G : SimpleGraph (Fin n))
  (h_graph : ∀ u v w : G.V, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ G.Adj u v ∧ G.Adj v w → ¬ G.Adj u w) :
  ∃ (H : SimpleGraph (Fin n)), H.edgeCount = 16 ∧ H.isTriangleFree :=
by
  sorry

end max_matches_without_triangles_l252_252872


namespace smallest_value_N_l252_252433

theorem smallest_value_N (N : ℕ) (a b c : ℕ) (h1 : N = a * b * c) (h2 : (a - 1) * (b - 1) * (c - 1) = 252) : N = 392 :=
sorry

end smallest_value_N_l252_252433


namespace find_sum_of_coefficients_l252_252178

theorem find_sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) (x : ℤ) :
    (x^5 * (x + 3)^3 = a_8 * (x + 1)^8 + a_7 * (x + 1)^7 + a_6 * (x + 1)^6 + a_5 * (x + 1)^5 + 
    a_4 * (x + 1)^4 + a_3 * (x + 1)^3 + a_2 * (x + 1)^2 + a_1 * (x + 1) + a_0) →
    7 * a_7 + 5 * a_5 + 3 * a_3 + a_1 = -8 :=
begin
  sorry
end

end find_sum_of_coefficients_l252_252178


namespace problem_part1_problem_part2_problem_part3_l252_252528

def Sn (n : ℕ) := n * (n + 1)
def an (n : ℕ) := 2 * n
def bn (n : ℕ) := 2 * (3^n + 1)
def cn (n : ℕ) := (an n * bn n) / 4
def Tn (n : ℕ) := ((2 * n - 1) * 3^(n+1) + 3) / 4 + n * (n + 1) / 2

theorem problem_part1 (n : ℕ) (h : n > 0) : 
  (∀ n > 0, Sn n = n * (n + 1)) → an n = 2 * n := 
by sorry

theorem problem_part2 (n : ℕ) (h : n > 0) : 
  (∀ n > 0, an n = (\sum i in list.range n, bn (i + 1) / (3^(i + 1) + 1) )) → bn n = 2 * (3^n + 1) := 
by sorry

theorem problem_part3 (n : ℕ) (h : n > 0) : 
  (∀ n > 0, cn n = (an n * bn n) / 4) → 
  Tn n = ((2 * n - 1) * 3^(n + 1) + 3) / 4 + n * (n + 1) / 2 := 
by sorry

end problem_part1_problem_part2_problem_part3_l252_252528


namespace selection_methods_count_l252_252914

theorem selection_methods_count :
  ∃ (students : Finset ℕ), students.card = 5 ∧
  (∃ (selection : Finset ℕ), selection.card = 4 ∧
    (∃ (friday : Finset ℕ) (saturday sunday : ℕ),
      friday.card = 2 ∧
      (∃ (temp : Finset ℕ), temp = selection \ friday ∧ temp.card = 2 ∧
        (∃ (sat : ℕ), sat ∈ temp ∧
          (∃ (sun : ℕ), sun ∈ temp \ {sat} ∧
            friday ∪ {sat} ∪ {sun} = selection ∧
            all_different [friday, {sat}, {sun}]) ∧
            (10 * 3 * 2) = 60)
))) :=
sorry

end selection_methods_count_l252_252914


namespace TV_height_l252_252436

theorem TV_height (area : ℝ) (width : ℝ) (height : ℝ) (h1 : area = 21) (h2 : width = 3) : height = 7 :=
  by
  sorry

end TV_height_l252_252436


namespace sea_ratio_southern_to_northern_l252_252754

-- Define the necessary conditions
def land_to_sea_ratio : ℝ := 29 / 71
def land_fraction_northern_hemisphere : ℝ := 3 / 4

-- Assertion to prove the required ratio of sea areas
theorem sea_ratio_southern_to_northern : 
  let total_surface_area : ℝ := 1 in
  let land_area : ℝ := total_surface_area * (29 / (29 + 71)) in
  let sea_area : ℝ := total_surface_area - land_area in
  let land_northern : ℝ := land_area * land_fraction_northern_hemisphere in
  let land_southern : ℝ := land_area - land_northern in
  let sea_northern : ℝ := (total_surface_area / 2) - land_northern in
  let sea_southern : ℝ := (total_surface_area / 2) - land_southern in
  sea_southern / sea_northern = 171 / 113 :=
by sorry

end sea_ratio_southern_to_northern_l252_252754


namespace evaluate_double_sum_l252_252479

theorem evaluate_double_sum :
  ∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m + 1) ^ 2 / (n + 1) / (m + n + 3) = 1 := by
  sorry

end evaluate_double_sum_l252_252479


namespace hardcover_books_count_l252_252871

theorem hardcover_books_count
  (h p : ℕ)
  (h_plus_p_eq_10 : h + p = 10)
  (total_cost_eq_250 : 30 * h + 20 * p = 250) :
  h = 5 :=
by
  sorry

end hardcover_books_count_l252_252871


namespace problem1_problem2_l252_252807

theorem problem1 : 
  (\dfrac { \sqrt {1+2\sin 610 ^{\circ} \cos 430 ^{\circ} }}{\sin 250 ^{\circ} +\cos 790 ^{\circ} } = -1) :=
sorry

theorem problem2 (α : ℝ) (hα : sin α + cos α = 2 / 3) : 
  (\dfrac {2\sin ^{2}α + 2\sin α\cos α}{1 + \tan α} = -5 / 9) :=
sorry

end problem1_problem2_l252_252807


namespace sum_of_odd_integers_13_to_41_l252_252778

theorem sum_of_odd_integers_13_to_41 : 
  (∑ k in Finset.filter (λ n, n % 2 = 1) (Finset.range 42), k) - ∑ k in Finset.filter (λ n, n % 2 = 1) (Finset.range 13), k = 405 :=
by
  sorry

end sum_of_odd_integers_13_to_41_l252_252778


namespace monkey_slips_back_l252_252825

theorem monkey_slips_back (s : ℝ) :
  (∀ n, 0 < s ∧ s ≤ 2) → (∃ t : ℕ, t = 20 ∧ ∃ h : ℝ, h = 22 ∧
  (19 * (3 - s) + 3 = h)) →
  s = 2 :=
by
  sorry

end monkey_slips_back_l252_252825


namespace min_value_n_minus_m_l252_252220

def f (x : ℝ) := real.exp (3 * x - 1)
def g (x : ℝ) := 1 / 3 + real.log x

theorem min_value_n_minus_m : 
  ∀ m n : ℝ, f m = g n → (n - m) ≥ (2 + real.log 3) / 3 :=
by
  sorry

end min_value_n_minus_m_l252_252220


namespace sum_of_digits_in_product_of_strings_of_nines_and_fours_53_l252_252158

theorem sum_of_digits_in_product_of_strings_of_nines_and_fours_53 :
  let nines := list.repeat 9 53
  let fours := list.repeat 4 53
  let product := (nat.of_digits 10 nines) * (nat.of_digits 10 fours)
  nat.digits 10 product.sum = 477 :=
by
  sorry

end sum_of_digits_in_product_of_strings_of_nines_and_fours_53_l252_252158


namespace even_nonzero_groups_count_l252_252906

-- Define the problem conditions and goal
theorem even_nonzero_groups_count 
  (n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 : ℕ)
  (n : ℕ) (h : n = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10) :
  let term := 2^{n1} - 2 in
  let answer := 2^{n-1} + 
    1 / 2 * ((2^{n1} - 2) * (2^{n2} - 2) * (2^{n3} - 2) * (2^{n4} - 2) * 
             (2^{n5} - 2) * (2^{n6} - 2) * (2^{n7} - 2) * (2^{n8} - 2) * 
             (2^{n9} - 2) * (2^{n10} - 2)) in
  ∃ (a : vector (fin 2) n),
  (count (λ i, (group (a, [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]) i).sum > 0) % 2 = 0) = answer :=
sorry

end even_nonzero_groups_count_l252_252906


namespace polynomial_partition_l252_252070

noncomputable def poly_increment : ℝ → ℝ → (ℝ → ℝ) → ℝ := λ a b p, p b - p a

def is_good_partition (p : ℝ → ℝ) : Prop := 
    poly_increment 0 (1/4) p + poly_increment (3/4) 1 p = poly_increment (1/4) (3/4) p

theorem polynomial_partition :
  ∀ p : ℝ → ℝ, (∀ x, p x = 0) ∨ (∃ a b c, ∀ x, p x = a + b * x + c * x^2) → is_good_partition p :=
begin
  sorry
end

end polynomial_partition_l252_252070


namespace cost_price_percentage_l252_252734

theorem cost_price_percentage (MP CP : ℝ) (h_discount : 0.75 * MP = CP * 1.171875) :
  ((CP / MP) * 100) = 64 :=
by
  sorry

end cost_price_percentage_l252_252734


namespace grammar_club_probability_l252_252673

/-- The Grammar club has 25 members: 15 boys and 10 girls. A 5-person committee
is chosen at random. The probability that the committee has at least 1 boy
and at least 1 girl is 195/208. -/
theorem grammar_club_probability (total_members boys girls : ℕ) (committee_size : ℕ) :
  total_members = 25 → boys = 15 → girls = 10 → committee_size = 5 →
  let total_ways := Nat.choose total_members committee_size,
      all_boys_ways := Nat.choose boys committee_size,
      all_girls_ways := Nat.choose girls committee_size,
      valid_committee_ways := total_ways - all_boys_ways - all_girls_ways,
      probability := valid_committee_ways / total_ways in
  probability = 195 / 208 :=
by
  intros h1 h2 h3 h4
  sorry

end grammar_club_probability_l252_252673


namespace arithmetic_geom_sequences_l252_252931

theorem arithmetic_geom_sequences
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ q, ∀ n, b (n + 1) = b n * q)
  (h1 : a 2 + a 3 = 14)
  (h2 : a 4 - a 1 = 6)
  (h3 : b 2 = a 1)
  (h4 : b 3 = a 3) :
  (∀ n, a n = 2 * n + 2) ∧ (∃ m, b 6 = a m ∧ m = 31) := sorry

end arithmetic_geom_sequences_l252_252931


namespace magnitude_of_BC_l252_252226

def vec_magnitude (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1^2 + v.2^2)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := 
  (v1.1 + v2.1, v1.2 + v2.2)

theorem magnitude_of_BC :
  let BA : ℝ × ℝ := (3, -2)
  let AC : ℝ × ℝ := (0, 6)
  vec_magnitude (vector_add BA AC) = 5 :=
by
  sorry

end magnitude_of_BC_l252_252226


namespace problem1_problem2_l252_252580

theorem problem1 (x y m n : ℝ) (hx : x ^ m = 2) (hy : y ^ n = 3) : x ^ (3 * m) + y ^ (2 * n) = 17 := 
  by
  sorry

theorem problem2 (x y : ℝ) (hx : x + 2 * y - 2 = 0) : 2 ^ x * 4 ^ y = 4 := 
  by
  sorry

end problem1_problem2_l252_252580


namespace base2_representation_of_27_l252_252873

-- The theorem states that the base-2 representation of 27 is 11011.
theorem base2_representation_of_27 : nat.binary 27 = [1, 1, 0, 1, 1] :=
by
  sorry

end base2_representation_of_27_l252_252873


namespace derivative_of_gx_eq_3x2_l252_252735

theorem derivative_of_gx_eq_3x2 (f : ℝ → ℝ) : (∀ x : ℝ, f x = (x + 1) * (x^2 - x + 1)) → (∀ x : ℝ, deriv f x = 3 * x^2) :=
by
  intro h
  sorry

end derivative_of_gx_eq_3x2_l252_252735


namespace inequality_solutions_l252_252147

theorem inequality_solutions (a : ℝ) (x : ℤ) (h_a : a ≠ 0) :
  (∃ x1 x2 x3 x4 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
  (a^2 * |a + x / a^2| + |(1 : ℤ) + x| ≤ 1 - a^3)) ↔
  (a ∈ Set.Iic (-Real.cbrt 2)) :=
sorry

end inequality_solutions_l252_252147


namespace find_domain_l252_252151

noncomputable def domain (x : ℝ) : Prop :=
  (2 * x + 1 ≥ 0) ∧ (3 - 4 * x ≥ 0)

theorem find_domain :
  {x : ℝ | domain x} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 3/4} :=
by
  sorry

end find_domain_l252_252151


namespace product_of_factors_l252_252846

theorem product_of_factors : (∏ k in Finset.range 11, (1 - 1 / (k + 2))) = 1 / 12 := by
  sorry

end product_of_factors_l252_252846


namespace finish_fourth_task_l252_252280

noncomputable def time_task_starts : ℕ := 12 -- Time in hours (12:00 PM)
noncomputable def time_task_ends : ℕ := 15 -- Time in hours (3:00 PM)
noncomputable def total_tasks : ℕ := 4 -- Total number of tasks
noncomputable def tasks_time (tasks: ℕ) := (time_task_ends - time_task_starts) * 60 / (total_tasks - 1) -- Time in minutes for each task

theorem finish_fourth_task : tasks_time 1 + ((total_tasks - 1) * tasks_time 1) = 240 := -- 4:00 PM expressed as 240 minutes from 12:00 PM
by
  sorry

end finish_fourth_task_l252_252280


namespace expression_evaluation_l252_252235

theorem expression_evaluation (k : ℚ) (h : 3 * k = 10) : (6 / 5) * k - 2 = 2 :=
by
  sorry

end expression_evaluation_l252_252235


namespace dave_total_time_l252_252862

variable (W J : ℕ)

-- Given conditions
def time_walked := W = 9
def ratio := J / W = 4 / 3

-- Statement to prove
theorem dave_total_time (time_walked : time_walked W) (ratio : ratio J W) : W + J = 21 := 
by
  sorry

end dave_total_time_l252_252862


namespace parabola_properties_l252_252369

theorem parabola_properties :
  ∃ (h k : ℝ), ∃ (d : ℕ), (h, k) = (4, -5) ∧ d = 1 ∧ (λ x : ℝ, (x - h)^2 + k) = (λ x : ℝ, (x-4)^2 - 5) :=
by
  sorry

end parabola_properties_l252_252369


namespace lending_books_l252_252792

def number_of_ways_to_lend_books 
  (books : List String) 
  (classmates : List String) 
  (conditions : ∀ book1 book2, (book1 = "Journey to the West" ∧ book2 = "Dream of the Red Chamber") → False) 
  (at_least_one_book : ∀ c ∈ classmates, ∃ b ∈ books, b ∈ c) 
  : Prop := 
  ∃ (ways : Nat), ways = 30

axiom books_list : List String := ["Romance of the Three Kingdoms", "Journey to the West", "Water Margin", "Dream of the Red Chamber"]
axiom classmates_list : List String := ["classmate1", "classmate2", "classmate3"]
axiom conditions_jw_drc (book1 book2 : String) : (book1 = "Journey to the West" ∧ book2 = "Dream of the Red Chamber") → False := sorry
axiom at_least_one_book_per_classmate (c : String) (hc : c ∈ classmates_list) : ∃ (b : String) (hb : b ∈ books_list), b ∈ c := sorry

theorem lending_books : number_of_ways_to_lend_books books_list classmates_list conditions_jw_drc at_least_one_book_per_classmate := 
by
  -- This is where the proof would be, but it is omitted as instructed.
  sorry

end lending_books_l252_252792


namespace weaving_increase_l252_252607

theorem weaving_increase (a₁ : ℕ) (S₃₀ : ℕ) (d : ℚ) (hₐ₁ : a₁ = 5) (hₛ₃₀ : S₃₀ = 390)
  (h_sum : S₃₀ = 30 * (a₁ + (a₁ + 29 * d)) / 2) : d = 16 / 29 :=
by {
  sorry
}

end weaving_increase_l252_252607


namespace number_of_intersections_is_0_or_1_l252_252364

noncomputable def num_intersection_points (f : ℝ → ℝ) : ℕ :=
if h : (∃ y : ℝ, f 2 = y) then 1 else 0

theorem number_of_intersections_is_0_or_1 (f : ℝ → ℝ) :
  num_intersection_points f = 0 ∨ num_intersection_points f = 1 :=
by sorry

end number_of_intersections_is_0_or_1_l252_252364


namespace average_side_length_of_squares_l252_252714

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252714


namespace blue_balls_taken_out_l252_252026

theorem blue_balls_taken_out (x : ℕ) :
  ∀ (total_balls : ℕ) (initial_blue_balls : ℕ)
    (remaining_probability : ℚ),
    total_balls = 25 ∧ initial_blue_balls = 9 ∧ remaining_probability = 1/5 →
    (9 - x : ℚ) / (25 - x : ℚ) = 1/5 →
    x = 5 :=
by
  intros total_balls initial_blue_balls remaining_probability
  rintro ⟨h_total_balls, h_initial_blue_balls, h_remaining_probability⟩ h_eq
  -- Proof goes here
  sorry

end blue_balls_taken_out_l252_252026


namespace find_period_for_interest_l252_252891

noncomputable def period_for_compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : ℝ :=
  (Real.log A - Real.log P) / (n * Real.log (1 + r / n))

theorem find_period_for_interest :
  period_for_compound_interest 8000 0.15 1 11109 = 2 := 
sorry

end find_period_for_interest_l252_252891


namespace max_value_of_expression_l252_252197

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 8) : 
  (1 + x) * (1 + y) ≤ 25 :=
by
  sorry

end max_value_of_expression_l252_252197


namespace function_behavior_l252_252955

noncomputable def f (x a : ℝ) : ℝ := (1 / 6) * x^2 - (1 / 2) * a * x^2 + x

theorem function_behavior (a : ℝ) (h1 : a ≤ 2) :
  convex_on ℝ (set.Ioo (-1 : ℝ) 2) (λ x, f x a) → 
  ∃ x ∈ set.Ioo (-1 : ℝ) 2, is_max_on (λ x, f x a) (set.Ioo (-1 : ℝ) 2) x ∧ 
  ¬ ∃ x ∈ set.Ioo (-1 : ℝ) 2, is_min_on (λ x, f x a) (set.Ioo (-1 : ℝ) 2) x :=
sorry

end function_behavior_l252_252955


namespace average_side_length_of_squares_l252_252689

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252689


namespace geometric_sequence_problem_l252_252181

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ a₁ q : ℝ, ∀ n, a n = a₁ * q^n

axiom a_3_eq_2 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2
axiom a_4a_6_eq_16 : ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 4 * a 6 = 16

theorem geometric_sequence_problem :
  ∃ a : ℕ → ℝ, geometric_sequence a ∧ a 3 = 2 ∧ a 4 * a 6 = 16 →
  (a 9 - a 11) / (a 5 - a 7) = 4 :=
sorry

end geometric_sequence_problem_l252_252181


namespace average_book_width_l252_252648

noncomputable def bookWidths : List ℝ := [5, 0.75, 1.5, 3, 12, 2, 7.5]

theorem average_book_width :
  (bookWidths.sum / bookWidths.length = 4.54) :=
by
  sorry

end average_book_width_l252_252648


namespace find_solutions_l252_252643

def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x + 4 else 3 * x - 6

theorem find_solutions :
  {x : ℝ | f (f x) = 5}.finite ∧ {x : ℝ | f (f x) = 5}.to_finset.card = 3 :=
by
  sorry

end find_solutions_l252_252643


namespace lcm_24_36_40_l252_252801

-- Define the natural numbers 24, 36, and 40
def n1 : ℕ := 24
def n2 : ℕ := 36
def n3 : ℕ := 40

-- Define the prime factorization of each number
def factors_n1 := [2^3, 3^1] -- 24 = 2^3 * 3^1
def factors_n2 := [2^2, 3^2] -- 36 = 2^2 * 3^2
def factors_n3 := [2^3, 5^1] -- 40 = 2^3 * 5^1

-- Prove that the LCM of n1, n2, n3 is 360
theorem lcm_24_36_40 : Nat.lcm (Nat.lcm n1 n2) n3 = 360 := sorry

end lcm_24_36_40_l252_252801


namespace quadrant_conditions_l252_252587

-- Formalizing function and conditions in Lean specifics
variable {a b : ℝ}

theorem quadrant_conditions 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 0 < a ∧ a < 1)
  (h4 : ∀ x < 0, a^x + b - 1 > 0)
  (h5 : ∀ x > 0, a^x + b - 1 > 0) :
  0 < b ∧ b < 1 := 
sorry

end quadrant_conditions_l252_252587


namespace polynomial_coefficients_l252_252190

theorem polynomial_coefficients :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (1 - 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 ∧
  a_0 = 1 ∧
  (a_1 + a_2 + a_3 + a_4 + a_5 = -2) ∧
  (a_1 + a_3 + a_5 = -122) :=
begin
  sorry
end

end polynomial_coefficients_l252_252190


namespace sum_real_solutions_eq_l252_252979

theorem sum_real_solutions_eq (b : ℝ) (hb : b > 2) :
  let y := (λ y : ℝ, sqrt (b - sqrt (b + y)) = y) in
  let sum_solutions : ℝ := (sqrt (4 * b - 3) - 1) / 2 in
  ∀ y ∈ {y : ℝ | sqrt (b - sqrt (b + y)) = y}, y = sum_solutions :=
sorry

end sum_real_solutions_eq_l252_252979


namespace domain_of_f_l252_252579

noncomputable def f (x : ℝ) := 1 / Real.sqrt (Real.logBase (1/2) (2*x + 1))

theorem domain_of_f : set.Ioo (-1/2 : ℝ) 0 = {x | f x ≥ 0} := by
  sorry

end domain_of_f_l252_252579


namespace ninth_square_more_tiles_than_eighth_l252_252456

theorem ninth_square_more_tiles_than_eighth :
  let num_tiles (n : ℕ) := n * n in
  num_tiles 9 - num_tiles 8 = 17 :=
by
  sorry

end ninth_square_more_tiles_than_eighth_l252_252456


namespace union_sets_l252_252537

theorem union_sets :
  let A := { x : ℝ | x^2 - x - 2 < 0 }
  let B := { x : ℝ | x > -2 ∧ x < 0 }
  A ∪ B = { x : ℝ | x > -2 ∧ x < 2 } :=
by
  sorry

end union_sets_l252_252537


namespace positive_difference_equation_l252_252899

noncomputable def positive_difference_solutions : ℝ :=
  let eq1 := (5 - (x : ℝ)^2 / 4)^(1 / 3) = -3
  let eq2 := x^2 = 128
  16 * Real.sqrt 2

theorem positive_difference_equation :
  (5 - x^2 / 4)^(1 / 3) = -3 → x = 8 * Real.sqrt 2 - (-8 * Real.sqrt 2) :=
by
  intro h
  sorry

end positive_difference_equation_l252_252899


namespace quadrilateral_is_parallelogram_l252_252814

def convex_quadrilateral (A B C D : Type*) : Prop :=
  ∃ P : Type*, between A B P ∧ between B C P ∧ between C D P ∧ between D A P

def bisects_area (A B C D P : Type*) (f : set Type* → ℝ) : Prop :=
  f ({A, P, B}) = f ({P, C, D}) ∧ f ({A, P, D}) = f ({P, B, C})

theorem quadrilateral_is_parallelogram {A B C D : Type*} (convex : convex_quadrilateral A B C D) 
  (bisect1: bisects_area A B C D P f) (bisect2: bisects_area A B C D P' f) : parallelogram A B C D := 
sorry

end quadrilateral_is_parallelogram_l252_252814


namespace hexagon_diagonals_sum_correct_l252_252088

noncomputable def hexagon_diagonals_sum : ℝ :=
  let AB := 40
  let S := 100
  let AC := 140
  let AD := 240
  let AE := 340
  AC + AD + AE

theorem hexagon_diagonals_sum_correct : hexagon_diagonals_sum = 720 :=
  by
  show hexagon_diagonals_sum = 720
  sorry

end hexagon_diagonals_sum_correct_l252_252088


namespace remaining_shirt_cost_l252_252905

theorem remaining_shirt_cost (total_shirts : ℕ) (cost_3_shirts : ℕ) (total_cost : ℕ) 
  (h1 : total_shirts = 5) 
  (h2 : cost_3_shirts = 3 * 15) 
  (h3 : total_cost = 85) :
  (total_cost - cost_3_shirts) / (total_shirts - 3) = 20 :=
by
  sorry

end remaining_shirt_cost_l252_252905


namespace max_value_of_expression_l252_252201

theorem max_value_of_expression (a b : ℝ)
  (h : a^2 = (1 + 2 * b) * (1 - 2 * b)) :
  ∃ m : ℝ, m = sqrt 2 ∧ (∀ x y : ℝ, x = a → y = b → (8 * x * y / (|x| + 2 * |y|)) ≤ m) :=
sorry

end max_value_of_expression_l252_252201


namespace part_1_tangent_line_a_part_2_range_a_F_derivative_l252_252556

-- Definitions of the functions f and g
def f (x : ℝ) : ℝ := (1/3) * x^3 - 6 * x
def g (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 + a

-- Part (Ⅰ): Prove that a = 18 under given conditions
theorem part_1_tangent_line_a (a : ℝ) : 
  (∀ (x : ℝ), f'(0) = -6 → (∀ x, f(x) = -6 * x → g'(x) = -6 * x + a - (1/2) * (-6) ^ 2) → a = 18) := sorry

-- Part (Ⅱ): Prove the range of real number values for a
theorem part_2_range_a (a : ℝ) :
  (3 = ℚ → ∀ (x : ℝ), F''(x) = x^2 - x - 6 →
  f(x) = g(x) has three distinct real solutions ↔ (9/2 < a ∧ a < 22/3)) := sorry

-- Helping definitions
def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem F_derivative (x : ℝ) : deriv (F x ?a) = x^2 - x - 6 := sorry

end part_1_tangent_line_a_part_2_range_a_F_derivative_l252_252556


namespace range_of_x_l252_252214

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / x) + 2 * Real.sin x

theorem range_of_x (x : ℝ) (h₀ : x > 0) (h₁ : f (1 - x) > f x) : x < (1 / 2) :=
by
  sorry

end range_of_x_l252_252214


namespace jacket_purchase_price_l252_252422

theorem jacket_purchase_price (P S SP : ℝ)
  (h1 : S = P + 0.40 * S)
  (h2 : SP = 0.80 * S)
  (h3 : SP - P = 18) :
  P = 54 :=
by
  sorry

end jacket_purchase_price_l252_252422


namespace range_of_m_l252_252558

noncomputable def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem range_of_m (m : ℝ) (h : ∀ x, x ∈ B m → x ∈ A) : m = 3 ∨ -2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_m_l252_252558


namespace antonio_meatballs_eaten_l252_252114

theorem antonio_meatballs_eaten :
  (let hamburger_per_meatball := (1 / 8 : ℝ),
       family_members := 8,
       total_hamburger := 4,
       total_meatballs := total_hamburger / hamburger_per_meatball,
       meatballs_per_family_member := total_meatballs / family_members in
   meatballs_per_family_member = 4) :=
begin
  -- Omitted proof
  sorry,
end

end antonio_meatballs_eaten_l252_252114


namespace simplify_expr_l252_252660

-- Define variables and their types if necessary
variables {x y : ℝ}

-- Define the given expression
def given_expr : ℝ := 5 * x^4 + 3 * x^2 * y - 4 - 3 * x^2 * y - 3 * x^4 - 1

-- State the theorem to prove the given expression simplifies to the correct answer
theorem simplify_expr : given_expr = 2 * x^4 - 5 :=
by sorry

end simplify_expr_l252_252660


namespace smallest_d_l252_252236

theorem smallest_d (d : ℕ) (h : 3150 * d = k ^ 2) : d = 14 :=
by
  -- assuming the condition: 3150 = 2 * 3 * 5^2 * 7
  have h_factorization : 3150 = 2 * 3 * 5^2 * 7 := by sorry
  -- based on the computation and verification, the smallest d that satisfies the condition is 14
  sorry

end smallest_d_l252_252236


namespace total_cost_of_repair_l252_252318

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end total_cost_of_repair_l252_252318


namespace positive_diff_solutions_correct_l252_252901

noncomputable def positive_diff_solutions : ℝ :=
  let eqn : ℝ → Prop := λ x, (5 - x^2 / 4) ^ (1 / 3) = -3
  16 * real.sqrt 2

theorem positive_diff_solutions_correct : 
  ∀ x1 x2, eqn x1 ∧ eqn x2 → abs (x1 - x2) = 16 * real.sqrt 2 := sorry

end positive_diff_solutions_correct_l252_252901


namespace total_pages_in_book_l252_252077

def initial_ratio_read_unread (total_pages : ℕ) (read_pages : ℕ) : Prop :=
  read_pages * 3 = (total_pages - read_pages)

def condition_after_reading (total_pages : ℕ) (read_pages : ℕ) (additional_pages : ℕ) : Prop :=
  read_pages + additional_pages = (2 * total_pages / 5)

theorem total_pages_in_book (total_pages : ℕ) (read_pages : ℕ) (additional_pages : ℕ = 48) :
  initial_ratio_read_unread total_pages read_pages ∧ condition_after_reading total_pages read_pages additional_pages → total_pages = 320 :=
by
  sorry

end total_pages_in_book_l252_252077


namespace symmetry_derivative_period_f2x_symmetry_composed_l252_252543

def f : ℝ → ℝ := sorry
variable (a c d : ℝ)
axiom h1 : ∀ x, f(a + x) = f(a - x)
axiom h2 : ∀ x, f(c + x) + f(c - x) = 2 * d
axiom h3 : a ≠ c

-- Statement A: f'(x) is symmetric about x = c
theorem symmetry_derivative : ∀ x, deriv f (c + x) = deriv f (c - x) :=
by sorry

-- Statement B: The period of f(2x) is 2|c - a|
theorem period_f2x : ∀ x, f(2 * x) = f(2 * x + 2 * |c - a|) :=
by sorry

-- Statement D: f(f(x)) is symmetric about x = a
theorem symmetry_composed : ∀ x, f(f(a + x)) = f(f(a - x)) :=
by sorry

end symmetry_derivative_period_f2x_symmetry_composed_l252_252543


namespace pow_mod_l252_252390

theorem pow_mod (n : ℕ) (h : n ≥ 2) : (5 ^ n) % 100 = 25 := by
  sorry

example : (5 ^ 2023) % 100 = 25 := by
  apply pow_mod
  -- Proving the check for n ≥ 2
  exact nat.le_of_lt (nat.lt_of_succ_lt (nat.lt_of_succ_lt (nat.succ_lt_succ (nat.succ_pos 2022))))

end pow_mod_l252_252390


namespace range_of_a_l252_252967

open Set

noncomputable def A : Set ℕ := {1, 2, 3}
noncomputable def B (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 6a - 1}

theorem range_of_a (a : ℝ) (h : A ∩ B a = {3}) : 1 ≤ a ∧ a < 2 := 
begin
  sorry
end

end range_of_a_l252_252967


namespace exam_max_incorrect_answers_l252_252826

theorem exam_max_incorrect_answers :
  ∀ (c w b : ℕ),
  (c + w + b = 30) →
  (4 * c - w ≥ 85) → 
  (c ≥ 22) →
  (w ≤ 3) :=
by
  intros c w b h1 h2 h3
  sorry

end exam_max_incorrect_answers_l252_252826


namespace average_side_length_of_squares_l252_252730

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252730


namespace distance_from_intersection_to_AB_l252_252277

theorem distance_from_intersection_to_AB
  (s : ℝ)
  (A D Y : ℝ × ℝ)
  (hA : A = (0, 0))
  (hD : D = (0, s))
  (hY_eq : (s^2 + (snd Y - s)^2 = s^2) ∧ ((fst Y)^2 + (snd Y)^2 = s^2))
  : dist Y (0, 0) = s :=
by
  sorry

end distance_from_intersection_to_AB_l252_252277


namespace tan_of_XYZ_l252_252273

noncomputable def tan_X : ℝ :=
  let XZ: ℝ := real.sqrt 17
  let YZ: ℝ := 4
  let XY: ℝ := real.sqrt (XZ^2 - YZ^2)
  XY = 1 -- This will be simplified in the Lean proof as sqrt(17 - 16) = 1
  in YZ / XY

theorem tan_of_XYZ (XZ YZ : ℝ) (hYZ : YZ = 4) (hXZ : XZ = real.sqrt 17) : 
  (∃ XY, XY = real.sqrt (XZ^2 - YZ^2) ∧ XY = 1) ∧ tan_X = 4 :=
by {
  sorry
}

end tan_of_XYZ_l252_252273


namespace winning_probability_proof_l252_252991

-- Define the context conditions
def boxes : Finset ℕ := {1, 2, 3, 4}
def probabilities (n : ℕ) := if n ∈ boxes then n / 10 else 0

def winning_probability : ℚ := 19 / 30

-- Define the main theorem statement
theorem winning_probability_proof :
  let probability_optimal_play := winning_probability in
  let m := 19 in
  let n := 30 in
  100 * m + n = 1930 := by
  sorry

end winning_probability_proof_l252_252991


namespace trig_inequality_l252_252917

open Real

theorem trig_inequality (a b c : ℝ) (h₁ : a = sin (2 * π / 7))
  (h₂ : b = cos (2 * π / 7)) (h₃ : c = tan (2 * π / 7)) :
  c > a ∧ a > b :=
by 
  sorry

end trig_inequality_l252_252917


namespace isosceles_triangle_solution_l252_252126

-- Definitions
variables {α : ℝ} -- angle α
variables (A B C F S G : Type) -- Points A, B, C, F, S (Centroid), G
variables [is_isosceles_triangle : is_isosceles A B C] -- A is the vertex, B and C are base vertices
variables (AF : ℝ) -- Length of median AF
variables (α_GT_30 α_EQ_30 α_LT_30 : Type -> Prop) -- Angles conditions

-- Prove the result conditions
theorem isosceles_triangle_solution : 
  (α > 30 → 0)
  ∧ (α = 30 → 1)
  ∧ (0 < α ∧ α < 30 → 2) :=
begin
  sorry
end

end isosceles_triangle_solution_l252_252126


namespace triangle_construction_possible_l252_252127

theorem triangle_construction_possible 
    (b c : ℝ) 
    (α : ℝ) 
    (hb : b > 0) 
    (hc : c > 0) 
    (halpha: 0 < α ∧ α < π / 2) :
    (b * Real.tan (α / 2) ≤ c ∧ c < b) ↔ ∃ A B C : ℝ × ℝ, 
    let M := (B + C) / 2 in 
    dist A C = b ∧ dist A B = c ∧ (angle A M B = α) :=
sorry

end triangle_construction_possible_l252_252127


namespace incorrect_statement_l252_252111

variable (U A : Set)  -- Assuming U and A are sets
variable (C_U C_A : Set → Set)  -- Assuming C_U and C_A are operations on sets

-- Hypotheses based on problem conditions
axiom h_CU_empty : ∀ U, C_U U = ∅
axiom h_CU_emptyset : ∀ U, C_U ∅ = U
axiom h_CU_CUA : ∀ U A, C_U (C_U A) = A
axiom h_CAA : ∀ A, C_A A = ∅

-- Statement to prove
theorem incorrect_statement (U : Set) : C_U U ≠ U := by
    sorry

end incorrect_statement_l252_252111


namespace range_of_sum_l252_252536

theorem range_of_sum (a b c : ℝ) (h1: a > b) (h2 : b > c) (h3 : a + b + c = 1) (h4 : a^2 + b^2 + c^2 = 3) :
-2/3 < b + c ∧ b + c < 0 := 
by 
  sorry

end range_of_sum_l252_252536


namespace num_palindromic_integers_even_greater_600000_l252_252409

def is_palindromic (n : ℕ) : Prop :=
  let digits := List.replicate 6 ((n % 10) / (10 ^ 6)) in
  digits = List.reverse digits

def satisfies_conditions (n : ℕ) : Prop :=
  600000 ≤ n ∧ n < 1000000 ∧ (n % 10) % 2 = 0 ∧ is_palindromic n

theorem num_palindromic_integers_even_greater_600000 :
  { n : ℕ // 600000 ≤ n ∧ n < 1000000 ∧ (n % 10) % 2 = 0 ∧ is_palindromic n }.card = 200 :=
sorry

end num_palindromic_integers_even_greater_600000_l252_252409


namespace total_volume_of_four_spheres_l252_252392

theorem total_volume_of_four_spheres (r : ℝ) (n : ℕ) (h_r : r = 3) (h_n : n = 4) :
  let V := (4/3) * Real.pi * r^3 in
  V * n = 144 * Real.pi := 
by
  -- Using the conditions provided
  rw [h_r, h_n]
  -- The proof is omitted
  sorry

end total_volume_of_four_spheres_l252_252392


namespace determinant_scaled_matrix_l252_252511

-- Definitions based on the conditions given in the problem.
def determinant2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

variable (a b c d : ℝ)
variable (h : determinant2x2 a b c d = 5)

-- The proof statement to be filled, proving the correct answer.
theorem determinant_scaled_matrix :
  determinant2x2 (2 * a) (2 * b) (2 * c) (2 * d) = 20 :=
by
  sorry

end determinant_scaled_matrix_l252_252511


namespace average_side_lengths_l252_252680

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252680


namespace average_side_length_of_squares_l252_252708

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252708


namespace compare_abc_l252_252520

noncomputable def a (x : ℝ) : ℝ := x^2 + x + Real.sqrt 2
noncomputable def b : ℝ := Real.log 3
noncomputable def c : ℝ := Real.exp (-0.5)

theorem compare_abc (x : ℝ) : b < c ∧ c < a x :=
by {
  -- Definitions based on the problem's conditions
  let a_def : ℝ := x^2 + x + Real.sqrt 2,
  let b_def : ℝ := Real.log 3,
  let c_def : ℝ := Real.exp (-0.5),

  -- Proof part is skipped with 'sorry'
  sorry
}

end compare_abc_l252_252520


namespace digit_arrangement_min_product_max_sum_l252_252654

theorem digit_arrangement_min_product_max_sum :
  ∃ (A B C D: ℕ), {A, B, C, D} = {4, 5, 7, 8} ∧
  (A * 10 + B) * (C * 10 + D) = 6290 ∧ 
  (A + C) + (B + D) = 24 :=
by
  let digits := {4, 5, 7, 8}
  let A := 7
  let B := 4
  let C := 8
  let D := 5
  have H1 : (A * 10 + B) * (C * 10 + D) = 74 * 85 := rfl
  have H2 : 74 * 85 = 6290 := rfl
  have H3 : (A + C) + (B + D) = 7 + 8 + 4 + 5 := rfl
  have H4 : 7 + 8 + 4 + 5 = 24 := rfl
  existsi [A, B, C, D]
  split
  repeat { assumption }
sorry

end digit_arrangement_min_product_max_sum_l252_252654


namespace infinite_series_sum_l252_252911

noncomputable def sum_of_series : ℝ :=
  let terms : ℕ → ℝ := λ n, if mod n 3 = 0 then 2 ^ (-n / 3) / 2 else if mod n 3 = 1 then - 2 ^ (-n / 3 - 1) else - 2 ^ (-n / 3 - 2)
  ∑' n, terms n

theorem infinite_series_sum :
  sum_of_series = 2 / 7 := by
  sorry

end infinite_series_sum_l252_252911


namespace average_side_length_of_squares_l252_252723

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252723


namespace apples_hand_out_l252_252353

theorem apples_hand_out (t p a h : ℕ) (h_t : t = 62) (h_p : p = 6) (h_a : a = 9) : h = t - (p * a) → h = 8 :=
by
  intros
  sorry

end apples_hand_out_l252_252353


namespace least_common_multiple_of_marble_sharing_l252_252833

theorem least_common_multiple_of_marble_sharing : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 7) 8) 10 = 280 :=
sorry

end least_common_multiple_of_marble_sharing_l252_252833


namespace average_side_length_of_squares_l252_252711

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252711


namespace sum_of_first_thirteen_terms_l252_252759

noncomputable def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_thirteen_terms
  (a d : ℤ)
  (h : a + (a + 8 * d) + (a + 10 * d) = 30) :
  arithmetic_sum a d 13 = 130 := 
sorry

end sum_of_first_thirteen_terms_l252_252759


namespace sin_alpha_plus_pi_over_2_l252_252943

theorem sin_alpha_plus_pi_over_2 
  (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -4 / 3) :
  Real.sin (α + Real.pi / 2) = -3 / 5 :=
by
  sorry

end sin_alpha_plus_pi_over_2_l252_252943


namespace AH_perpendicular_BP_l252_252435

open_locale classical

variables {A B C M H P : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space H] [metric_space P]

-- Definitions of points and conditions
variables (A B C : point) (M : point) (H : point) (P : point)
variables (AB BC MH : line_segment)
variables (mid_AC : midpoint A C M) (mid_MH : midpoint M H P)

-- Condition: AB = BC
variable (AB_eq_BC : AB.length = BC.length)

-- Condition: MH is perpendicular to BC
variable (MH_perpendicular_BC : perpendicular MH BC)

-- The theorem to be proved:
theorem AH_perpendicular_BP
  (A B C M H P : point)
  (M_mid_AC : midpoint A C M) 
  (H_on_BC : H ∈ BC)
  (MH_perpendicular_BC : perpendicular MH BC)
  (P_mid_MH : midpoint M H P)
  (AB_eq_BC : length (segment A B) = length (segment B C)) :
  perpendicular (line_segment A H) (line_segment B P) :=
begin
  sorry
end

end AH_perpendicular_BP_l252_252435


namespace problem_l252_252210

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

variables (f : ℝ → ℝ)
variables (h_odd : is_odd_function f)
variables (h_f1 : f 1 = 5)
variables (h_period : ∀ x, f (x + 4) = -f x)

-- Prove that f(2012) + f(2015) = -5
theorem problem :
  f 2012 + f 2015 = -5 :=
sorry

end problem_l252_252210


namespace dirk_profit_l252_252869

def total_amulets_sold (days : ℕ) (amulets_per_day : ℕ) : ℕ :=
  days * amulets_per_day

def total_revenue (total_amulets : ℕ) (price_per_amulet : ℕ) : ℕ :=
  total_amulets * price_per_amulet

def revenue_after_payment (total_revenue : ℕ) (percentage : ℝ) : ℝ :=
  total_revenue * (1.0 - percentage)

def total_cost (total_amulets : ℕ) (cost_per_amulet : ℕ) : ℕ :=
  total_amulets * cost_per_amulet

def profit (revenue : ℝ) (cost : ℕ) : ℝ :=
  revenue - cost

theorem dirk_profit :
  let days := 2
  let amulets_per_day := 25
  let price_per_amulet := 40
  let cost_per_amulet := 30
  let percentage := 0.10 in
  profit (revenue_after_payment (total_revenue (total_amulets_sold days amulets_per_day) price_per_amulet) percentage)
         (total_cost (total_amulets_sold days amulets_per_day) cost_per_amulet) = 300 := 
by 
  sorry

end dirk_profit_l252_252869


namespace record_expenditure_20_l252_252983

-- Define the concept of recording financial transactions
def record_income (amount : ℤ) : ℤ := amount

def record_expenditure (amount : ℤ) : ℤ := -amount

-- Given conditions
variable (income : ℤ) (expenditure : ℤ)

-- Condition: the income of 30 yuan is recorded as +30 yuan
axiom income_record : record_income 30 = 30

-- Prove an expenditure of 20 yuan is recorded as -20 yuan
theorem record_expenditure_20 : record_expenditure 20 = -20 := 
  by sorry

end record_expenditure_20_l252_252983


namespace solve_inequality_l252_252344

theorem solve_inequality (x : ℝ) : 
  (x - 5) / (x - 3)^2 < 0 ↔ x ∈ Set.Ioo (-(∞ : ℝ)) 3 ∪ Set.Ioo 3 5 :=
by
  sorry

end solve_inequality_l252_252344


namespace ellipse_standard_equation_l252_252953

-- Given conditions
def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)
def p : ℝ × ℝ := (5 / 2, -3 / 2)

-- Distance between the point p and foci f1, f2 should sum to 2a
def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

theorem ellipse_standard_equation :
  let a := real.sqrt 10,
      b := real.sqrt (a^2 - 2^2) in
  distance p f1 + distance p f2 = 2 * a ∧
  b^2 = a^2 - 2^2 ∧
  a > 0 ∧ b > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) := 
by
  sorry

end ellipse_standard_equation_l252_252953


namespace infinite_similar_triangles_l252_252446

open Classical

-- Given a right triangle ABC with ∠C = 90° and CD as the altitude to the hypotenuse AB
-- Points E and F are taken on BC and CA respectively such that △FED ~ △ABC

variables {α : Type*} [EuclideanGeometry α]

theorem infinite_similar_triangles (A B C D E F : α)
    (hABC : ∠BAC = 90°) (hAltitude : altitude D A B)
    (hPts : E ∈ line_segment B C ∧ F ∈ line_segment C A) :
    ∃ (S : set (triangle α)), infinite S ∧ ∀ (T ∈ S), sim T (triangle.mk F E D) (triangle.mk A B C) :=
by {
  sorry
}

end infinite_similar_triangles_l252_252446


namespace value_of_b_l252_252477

-- Defining the number sum in circles and overlap
def circle_sum := 21
def num_circles := 5
def total_sum := 69

-- Overlapping numbers
def overlap_1 := 2
def overlap_2 := 8
def overlap_3 := 9
variable (b d : ℕ)

-- Circle equation containing d
def circle_with_d := d + 5 + 9

-- Prove b = 10 given the conditions
theorem value_of_b (h₁ : num_circles * circle_sum = 105)
    (h₂ : 105 - (overlap_1 + overlap_2 + overlap_3 + b + d) = total_sum)
    (h₃ : circle_with_d d = 21) : b = 10 :=
by sorry

end value_of_b_l252_252477


namespace average_side_lengths_l252_252701

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252701


namespace find_x_l252_252879

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l252_252879


namespace vector_b_magnitude_l252_252951

variables {a b : EuclideanSpace ℝ (Fin 2)}

def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
(real_inner_product_space.angle a b).toReal

noncomputable def vector_length (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
real_inner.norm v

theorem vector_b_magnitude (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : angle_between_vectors a b = real.pi / 3)
  (h2 : vector_length a = 1)
  (h3 : vector_length (2 • a - b) = real.sqrt 7) :
  vector_length b = 3 := by
  sorry

end vector_b_magnitude_l252_252951


namespace probability_same_color_two_dice_l252_252574

theorem probability_same_color_two_dice :
  let total_sides : ℕ := 30
  let maroon_sides : ℕ := 5
  let teal_sides : ℕ := 10
  let cyan_sides : ℕ := 12
  let sparkly_sides : ℕ := 3
  (maroon_sides / total_sides)^2 + (teal_sides / total_sides)^2 + (cyan_sides / total_sides)^2 + (sparkly_sides / total_sides)^2 = 139 / 450 :=
by
  sorry

end probability_same_color_two_dice_l252_252574


namespace oranges_per_bag_l252_252476

theorem oranges_per_bag (total_oranges : ℕ) (num_bags : ℕ) (h1 : total_oranges = 1035) (h2 : num_bags = 45) :
  total_oranges / num_bags = 23 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end oranges_per_bag_l252_252476


namespace sum_of_coprime_fractions_l252_252525

theorem sum_of_coprime_fractions (n : ℕ) :
  ∑ (p q : ℕ) in {pq : ℕ × ℕ | 0 < pq.fst ∧ pq.fst < pq.snd ∧ pq.snd ≤ n ∧ Nat.gcd pq.fst pq.snd = 1 ∧ pq.fst + pq.snd > n}.toFinset (λ pq, (1 : ℚ) / (pq.fst * pq.snd)) = 1 / 2 :=
  sorry

end sum_of_coprime_fractions_l252_252525


namespace average_speed_of_journey_is_24_l252_252798

noncomputable def average_speed (D : ℝ) (speed_to_office speed_to_home : ℝ) : ℝ :=
  let time_to_office := D / speed_to_office
  let time_to_home := D / speed_to_home
  let total_distance := 2 * D
  let total_time := time_to_office + time_to_home
  total_distance / total_time

theorem average_speed_of_journey_is_24 (D : ℝ) : average_speed D 20 30 = 24 := by
  -- nonconstructive proof to fulfill theorem definition
  sorry

end average_speed_of_journey_is_24_l252_252798


namespace inequality_one_solution_set_range_of_a_if_solution_set_contains_l252_252176

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) + abs (a * x + 1)

theorem inequality_one_solution_set (x : ℝ) (h : a = 1) :
  f x a ≥ 3 ↔ x ≤ -3/2 ∨ x ≥ 3/2 := sorry

theorem range_of_a_if_solution_set_contains (h : ∀ x ∈ Icc (-1 : ℝ) 1, f x a ≤ 3 - x) :
  a ∈ Icc (-1 : ℝ) 1 := sorry

end inequality_one_solution_set_range_of_a_if_solution_set_contains_l252_252176


namespace geometric_series_sum_l252_252301

theorem geometric_series_sum :
  let a := 6 in
  let r := -1 / 3 in
  let t := a / (1 - r) in
  t = 4.5 :=
by
  let a := 6
  let r := -1 / 3
  let t := a / (1 - r)
  show t = 4.5
  sorry

end geometric_series_sum_l252_252301


namespace exists_composite_l252_252337

theorem exists_composite (n : ℤ) : ∃ x : ℤ, ¬ nat.prime (nat_abs (n * x + 1)) :=
by
  sorry

end exists_composite_l252_252337


namespace prob_teamY_wins_first_given_conditions_l252_252669

namespace TeamSeries

noncomputable def probability_teamY_wins_first_game
  (Y_wins_third : Prop)
  (X_wins_series : Prop)
  : ℝ := 5 / 12

theorem prob_teamY_wins_first_given_conditions :
  ∀ (first_to_four : ∀ (winsX winsY: ℕ), winsX = 4 ∨ winsY = 4)
    (equal_likely : ∀ (team : string), team = "X" ∨ team = "Y" → ℙ (win_single_game := 0.5))
    (no_ties : ∀ (team : string), team = "X" ∨ team = "Y" → win_single_game ≠ lose_single_game)
    (independent : ∀ (a b : Prop), indep a b)
    (Y_wins_third : Prop)
    (X_wins_series : Prop),
    probability_teamY_wins_first_game Y_wins_third X_wins_series = 5 / 12 :=
by sorry

end TeamSeries

end prob_teamY_wins_first_given_conditions_l252_252669


namespace base_8_arithmetic_l252_252774

theorem base_8_arithmetic :
    let a := 72
    let b := 45
    let c := 23
  in a - b + c = 50 :=
by
  -- Base 8 arithmetic operation
  let a_b8 := nat_of_digits 8 [7, 2]
  let b_b8 := nat_of_digits 8 [4, 5]
  let c_b8 := nat_of_digits 8 [2, 3]
  let lhs := (a_b8 - b_b8) + c_b8
  let rhs := nat_of_digits 8 [5, 0]
  lhs = rhs
  sorry

end base_8_arithmetic_l252_252774


namespace x_intercept_of_line_l252_252889

theorem x_intercept_of_line (x y : ℝ) : (4 * x + 7 * y = 28) ∧ (y = 0) → x = 7 :=
by
  sorry

end x_intercept_of_line_l252_252889


namespace lines_intersection_l252_252031

theorem lines_intersection (n c : ℝ) : 
    (∀ x y : ℝ, y = n * x + 5 → y = 4 * x + c → (x, y) = (8, 9)) → 
    n + c = -22.5 := 
by
    intro h
    sorry

end lines_intersection_l252_252031


namespace john_plays_hours_per_day_l252_252625

theorem john_plays_hours_per_day (bpm beats over_days : ℕ) (hours_per_day : ℕ) :
  bpm = 200 → beats = 72000 → over_days = 3 → 
  hours_per_day = (beats / bpm) / 60 / over_days :=
by
  intros h_bpm h_beats h_over_days
  have h1 : (beats / bpm) = 360 := by sorry
  have h2 : 360 / 60 = 6 := by sorry
  have h3 : 6 / over_days = hours_per_day := by sorry
  exact h3

end john_plays_hours_per_day_l252_252625


namespace solve_for_question_mark_l252_252806

def cube_root (x : ℝ) := x^(1/3)
def square_root (x : ℝ) := x^(1/2)

theorem solve_for_question_mark : 
  cube_root (5568 / 87) + square_root (72 * 2) = square_root 256 := by
  sorry

end solve_for_question_mark_l252_252806


namespace series_sum_property_l252_252212

def f (a : ℝ) (x : ℝ) := a^(2*x) / (a + a^(2*x))

theorem series_sum_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (finset.range 2015).sum (λ k, f a ((k + 1) / 2016)) = 2015 / 2 :=
by
  sorry

end series_sum_property_l252_252212


namespace sum_of_coefficients_l252_252001

theorem sum_of_coefficients :
  ∃ (a b c d e : ℤ), 
  (216 * (x : ℤ)^3 - 27 = (a * x - b) * (c * x^2 + d * x - e)) ∧
  a + b + c + d + e = 72 :=
by
  have h : 216 * (x : ℤ)^3 - 27 = (6 * x - 3) * (36 * x^2 + 18 * x + 9) := 
    by sorry -- placeholder for the actual factorization proof
  use [6, 3, 36, 18, 9]
  split
  · exact h
  · norm_num

end sum_of_coefficients_l252_252001


namespace weight_limit_for_48_packages_l252_252596

theorem weight_limit_for_48_packages (P : ℕ) (h1 : 0.30 * P = 72) 
                                     (h2 : 0.20 * P = 48) 
                                     (h3 : 24 = 0.10 * P) 
                                     (h4 : 48 = 72 - 24): 
                                     75 = 75 := 
begin
  -- The proof would go here
  sorry
end

end weight_limit_for_48_packages_l252_252596


namespace find_m_of_chord_length_l252_252952

theorem find_m_of_chord_length
  (m : ℝ)
  (h1 : ∀ x y, y = x + m → 4 * x^2 + y^2 = 1 → length (chord_intersection (4 * x^2 + y^2 = 1) (y = x + m)) = 2 * real.sqrt 2 / 5) :
  m = real.sqrt 5 / 2 ∨ m = - (real.sqrt 5 / 2) :=
by
  sorry

end find_m_of_chord_length_l252_252952


namespace average_side_lengths_l252_252704

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252704


namespace ordered_pairs_count_l252_252570

theorem ordered_pairs_count : ∃ (count : ℕ), count = 4 ∧
  ∀ (m n : ℕ), m > 0 → n > 0 → m ≥ n → m^2 - n^2 = 144 → (∃ (i : ℕ), i < count) := by
  sorry

end ordered_pairs_count_l252_252570


namespace tan_alpha_value_l252_252510

theorem tan_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π) (h_eq : sin α + cos α = 1 / 5) : tan α = -4 / 3 :=
sorry

end tan_alpha_value_l252_252510


namespace average_side_lengths_l252_252676

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252676


namespace decimal_to_binary_27_l252_252876

theorem decimal_to_binary_27 : nat.digits 2 27 = [1, 1, 0, 1, 1] :=
by sorry

end decimal_to_binary_27_l252_252876


namespace inequality_for_large_exponent_l252_252278

theorem inequality_for_large_exponent (u : ℕ → ℕ) (x : ℕ) (k : ℕ) (hk : k = 100) (hu : u x = 2^x) : 
  2^(2^(x : ℕ)) > 2^(k * x) :=
by 
  sorry

end inequality_for_large_exponent_l252_252278


namespace angle_C_deg_parallelogram_l252_252262

theorem angle_C_deg_parallelogram (A B C D : Type) [parallelogram A B C D] (angle_A angle_B : ℝ) (h1 : angle_A = angle_B - 40) (h2 : angle_A + angle_B = 180) :
  angle_A = angle_C := by
  sorry

end angle_C_deg_parallelogram_l252_252262


namespace sum_k_eq_26_l252_252499

theorem sum_k_eq_26 :
  let sum_of_binomials := Nat.choose 25 4 + Nat.choose 25 5
  (∑ k in Finset.filter (λ k => Nat.choose 26 k = sum_of_binomials) (Finset.range 27), k) = 26 :=
by
  sorry

end sum_k_eq_26_l252_252499


namespace problem_1_probability_problem_2_conditional_problem_3_expected_l252_252416

-- Problem (1): Probability over 3 consecutive days
def probability_event_A (days: ℕ) (prob_first_class: ℚ): ℚ :=
  sorry -- Skip the proof

-- Conditions
constant prob_first_class_product: ℚ := 0.5

-- Theorem for Problem (1)
theorem problem_1_probability: probability_event_A 3 0.5 = 27 / 64 := 
  sorry

-- Problem (2): Conditional Probability
def conditional_probability (prob_first_class: ℚ): ℚ :=
  sorry -- Skip the proof

-- Theorem for Problem (2)
theorem problem_2_conditional: conditional_probability 0.5 = 1 / 3 := 
  sorry

-- Problem (3): Expected daily profit
def expected_daily_profit (prob_first_class: ℚ) (prob_second_class: ℚ) (profit: ℤ -> ℚ): ℚ :=
  sorry -- Skip the proof

-- Conditions
constant prob_second_class_product: ℚ := 0.4
constant profits: ℤ -> ℚ :=
  λp, if p = -6000 then 0.01
     else if p = 3000 then 0.08
     else if p = 5000 then 0.1
     else if p = 12000 then 0.16
     else if p = 14000 then 0.4
     else if p = 16000 then 0.25
     else 0

-- Theorem for Problem (3)
theorem problem_3_expected: expected_daily_profit 0.5 0.4 profits = 12200 := 
  sorry

end problem_1_probability_problem_2_conditional_problem_3_expected_l252_252416


namespace group_1991_l252_252066

theorem group_1991 (n : ℕ) (h1 : 1 ≤ n) (h2 : 1991 = 2 * n ^ 2 - 1) : n = 32 := 
sorry

end group_1991_l252_252066


namespace intersection_of_A_and_B_union_of_A_and_B_l252_252291

def A : Set ℝ := {x | x * (9 - x) > 0}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} :=
sorry

theorem union_of_A_and_B : A ∪ B = {x | x < 9} :=
sorry

end intersection_of_A_and_B_union_of_A_and_B_l252_252291


namespace symmetric_point_l252_252507

theorem symmetric_point (a : ℝ) : (∃ n : ℤ, a = (2 * n * Real.pi) / 5) ↔ (sin (2 * a) = -sin (3 * a) ∧ cos (3 * a) = cos (2 * a)) :=
by
  sorry

end symmetric_point_l252_252507


namespace intersection_of_sets_l252_252313

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {x | x < -1 ∨ x > 1}

noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

noncomputable def complement_U_M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

noncomputable def intersection_N_complement_U_M : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets :
  N ∩ complement_U_M = intersection_N_complement_U_M := 
sorry

end intersection_of_sets_l252_252313


namespace product_gt_one_l252_252216

variables {a x1 x2 : ℝ}

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x * (Real.log x - a) + 1

-- Hypothesis conditions
axiom has_two_zeros (h : ∀ (a : ℝ), 1 < a -> ∃ x1 x2 : ℝ, x2 > x1 ∧ x1 > 0 ∧ f x1 a = 0 ∧ f x2 a = 0)

-- Theorem to prove
theorem product_gt_one (h₁ : 1 < a)
  (h₂ : ∃ x1 x2 : ℝ, x2 > x1 ∧ x1 > 0 ∧ f x1 a = 0 ∧ f x2 a = 0) : x1 * x2 > 1 :=
by
  -- Proof is omitted
  sorry

end product_gt_one_l252_252216


namespace identical_functions_l252_252110

theorem identical_functions : ∀ (μ : ℝ), -1 < μ ∧ μ < 1 → (√((1 + μ) / (1 - μ)) = √((1 + μ) / (1 - μ))) :=
by
  intros
  sorry

end identical_functions_l252_252110


namespace distance_between_x_intercepts_l252_252820

theorem distance_between_x_intercepts :
  let slope1 := 4
  let slope2 := -2
  let point := (8, 20)
  let line1 (x : ℝ) := slope1 * (x - point.1) + point.2
  let line2 (x : ℝ) := slope2 * (x - point.1) + point.2
  let x_intercept1 := (0 - point.2) / slope1 + point.1
  let x_intercept2 := (0 - point.2) / slope2 + point.1
  abs (x_intercept1 - x_intercept2) = 15 := sorry

end distance_between_x_intercepts_l252_252820


namespace min_square_grid_l252_252851

-- Define the conditions of the problem:
def unique_rectangle_division (n : ℕ) (blue_cells : set (ℕ × ℕ)) : Prop :=
  ∃ (rectangles : set (set (ℕ × ℕ))),
    (∀ rect ∈ rectangles, ∃ b ∈ blue_cells, rect = {b}) ∧
    (∀ x ∈ blue_cells, ∃! rect ∈ rectangles, x ∈ rect) ∧
    (∀ rect1 rect2 ∈ rectangles, rect1 ≠ rect2 → rect1 ∩ rect2 = ∅ ∧ rect1 ⊆ { (i, j) | 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n })

-- State the theorem to be proved:
theorem min_square_grid (n : ℕ) (blue_cells : set (ℕ × ℕ)) (h : blue_cells.size = 101) (h_unique : unique_rectangle_division n blue_cells) : n = 101 := 
sorry

end min_square_grid_l252_252851


namespace correct_average_of_10_numbers_l252_252732

theorem correct_average_of_10_numbers :
  ∀ (avg n incorrect correct actual_avg : ℕ),
    n = 10 →
    avg = 16 →
    incorrect = 26 →
    correct = 46 →
    actual_avg = 18 →
    (avg * n - incorrect + correct) / n = actual_avg := 
by
  intros avg n incorrect correct actual_avg h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end correct_average_of_10_numbers_l252_252732


namespace crackers_per_person_l252_252649

/-- Matthew had 45 crackers and gave equal numbers of crackers to 15 friends.
    Prove that each person ate 3 crackers. -/
theorem crackers_per_person : (total_crackers friends crackers : ℕ) (h1 : total_crackers = 45) (h2 : friends = 15) :
  (crackers_per_person : ℕ) = total_crackers / friends :=
by
  sorry

end crackers_per_person_l252_252649


namespace monthly_rent_computation_l252_252813

noncomputable def monthly_rent (length width : ℝ) (rent_per_acre : ℝ) (sq_ft_per_acre : ℝ) : ℝ :=
  let area_sq_ft := length * width
  let area_acres := area_sq_ft / sq_ft_per_acre
  area_acres * rent_per_acre

theorem monthly_rent_computation :
  monthly_rent 472 1378 42.75 43560 ≈ 638.23 :=
by
  sorry

end monthly_rent_computation_l252_252813


namespace sequence_increasing_l252_252529

noncomputable def a_n (n : ℕ) : ℝ := 2 * n / (3 * n + 1)

theorem sequence_increasing : ∀ n : ℕ, 1 ≤ n → a_n (n + 1) > a_n n :=
by
  intro n hn
  have h₁ : a_n (n + 1) = 2 * (n + 1) / (3 * (n + 1) + 1) := rfl
  have h₂ : a_n n = 2 * n / (3 * n + 1) := rfl
  rw [h₁, h₂]
  sorry

end sequence_increasing_l252_252529


namespace blocks_differ_in_two_ways_l252_252078

/-- 
A child has a set of 120 distinct blocks. Each block is one of 3 materials (plastic, wood, metal), 
3 sizes (small, medium, large), 4 colors (blue, green, red, yellow), and 5 shapes (circle, hexagon, 
square, triangle, pentagon). How many blocks in the set differ from the 'metal medium blue hexagon' 
in exactly 2 ways?
-/
def num_blocks_differ_in_two_ways : Nat := 44

theorem blocks_differ_in_two_ways (blocks : Fin 120)
    (materials : Fin 3)
    (sizes : Fin 3)
    (colors : Fin 4)
    (shapes : Fin 5)
    (fixed_block : {m // m = 2} × {s // s = 1} × {c // c = 0} × {sh // sh = 1}) :
    num_blocks_differ_in_two_ways = 44 :=
by
  -- proof steps are omitted
  sorry

end blocks_differ_in_two_ways_l252_252078


namespace regular_decagon_interior_angle_l252_252777

-- Define the number of sides in a regular decagon
def n : ℕ := 10

-- Define the formula for the sum of the interior angles of an n-sided polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the measure of one interior angle of a regular decagon
def one_interior_angle_of_regular_polygon (sum_of_angles : ℕ) (n : ℕ) : ℕ :=
  sum_of_angles / n

-- Prove that the measure of one interior angle of a regular decagon is 144 degrees
theorem regular_decagon_interior_angle : one_interior_angle_of_regular_polygon (sum_of_interior_angles 10) 10 = 144 := by
  sorry

end regular_decagon_interior_angle_l252_252777


namespace intersection_M_N_l252_252224

def set_M : Set ℝ := { x | x < 2 }
def set_N : Set ℝ := { x | x > 0 }
def set_intersection : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_M_N : set_M ∩ set_N = set_intersection := 
by
  sorry

end intersection_M_N_l252_252224


namespace distinct_values_least_count_l252_252091

theorem distinct_values_least_count (total_integers : ℕ) (mode_count : ℕ) (unique_mode : Prop) 
  (h1 : total_integers = 3200)
  (h2 : mode_count = 17)
  (h3 : unique_mode):
  ∃ (least_count : ℕ), least_count = 200 := by
  sorry

end distinct_values_least_count_l252_252091


namespace LuckyLarry_f_value_l252_252314

theorem LuckyLarry_f_value :
  ∃ (f : ℤ), 
    let a := 2,
    let b := 3,
    let c := 4,
    let d := 5 in
    a + (b - (c + (d - f))) = a + b - c + d - f ∧
    f = 5 :=
by
  sorry

end LuckyLarry_f_value_l252_252314


namespace solution_ineq_set_l252_252545

-- Given conditions
variables (a b x : ℝ)
def ineq1 := -x^2 + a*x + b ≥ 0
def interval := ∀ x, -2 ≤ x ∧ x ≤ 3 → ineq1 x

-- Proof goal in Lean 4
theorem solution_ineq_set (h : interval (λ x, ineq1)) :
  { x : ℝ | x < 2 ∨ x > 3 } :=
sorry

end solution_ineq_set_l252_252545


namespace sum_distances_from_R_l252_252508

-- Define circles with centers A, B, C, D
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Define points P and Q common to all circles
variables {P Q : MetricSpace Point}

-- Define the midpoint R of segment PQ
def R : MetricSpace Point := midpoint P Q

-- Define conditions
variables (rA rB rC rD : ℝ)
def radius_relation_A_B : Prop := rA = 3/4 * rB
def radius_relation_C_D : Prop := rC = 3/4 * rD
  
def AB_distance : ℝ := 50
def CD_distance : ℝ := 50
def PQ_distance : ℝ := 60

-- Define the distance from the midpoint R to the centers A, B, C, D
def AR : ℝ := 50
def BR : ℝ := 50
def CR : ℝ := 50
def DR : ℝ := 50

-- Prove the sum of distances from R to the centers is 200
theorem sum_distances_from_R :
  AR + BR + CR + DR = 200 :=
by {
  have h1 : AR = 50, from rfl,
  have h2 : BR = 50, from rfl,
  have h3 : CR = 50, from rfl,
  have h4 : DR = 50, from rfl,
  calc
    AR + BR + CR + DR
      = 50 + 50 + 50 + 50 : by rw [h1, h2, h3, h4]
  ... = 200 : by norm_num
}

end sum_distances_from_R_l252_252508


namespace modulo_4_equiv_2_l252_252289

open Nat

noncomputable def f (n : ℕ) [Fintype (ZMod n)] : ZMod n → ZMod n := sorry

theorem modulo_4_equiv_2 (n : ℕ) [hn : Fact (n > 0)] 
  (f : ZMod n → ZMod n)
  (h1 : ∀ x, f x ≠ x)
  (h2 : ∀ x, f (f x) = x)
  (h3 : ∀ x, f (f (f (x + 1) + 1) + 1) = x) : 
  n % 4 = 2 := 
sorry

end modulo_4_equiv_2_l252_252289


namespace value_of_a5_l252_252265
noncomputable def a : ℕ → ℚ
| 1       := 1
| (n + 2) := 1 + (-1)^(n + 2) / a (n + 1)

theorem value_of_a5 : a 5 = 2 / 3 := 
sorry

end value_of_a5_l252_252265


namespace minimum_distance_circle_to_line_l252_252925

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

def line_eq (x y : ℝ) : Prop := x - 2*y - 5 = 0

theorem minimum_distance_circle_to_line :
  (∀ (x y : ℝ), circle_eq x y → ∃ (d : ℝ), d = sqrt 5 - 1 ∧ (∀ (x' y' : ℝ), circle_eq x' y' → line_eq x' y' → distance (x', y') (0, d) ≥ d))
:= sorry

end minimum_distance_circle_to_line_l252_252925


namespace average_side_length_of_squares_l252_252685

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252685


namespace difference_cubed_divisible_by_27_l252_252658

theorem difference_cubed_divisible_by_27 (a b : ℤ) :
    ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3) % 27 = 0 := 
by
  sorry

end difference_cubed_divisible_by_27_l252_252658


namespace min_value_m_l252_252817

noncomputable def min_m (f: ℝ → ℝ) (h1: ∀ x y, 1 ≤ x ∧ x ≤ 2017 ∧ 1 ≤ y ∧ y ≤ 2017 → |f x - f y| ≤ 2 * |x - y|)
  (h2: f 1 = f 2017) : ℝ :=
  inf {m : ℝ | ∀ x y, 1 ≤ x ∧ x ≤ 2017 ∧ 1 ≤ y ∧ y ≤ 2017 → |f x - f y| ≤ m}

theorem min_value_m (f: ℝ → ℝ) (h1: ∀ x y, 1 ≤ x ∧ x ≤ 2017 ∧ 1 ≤ y ∧ y ≤ 2017 → |f x - f y| ≤ 2 * |x - y|)
  (h2: f 1 = f 2017) : min_m f h1 h2 = 2016 :=
by sorry

end min_value_m_l252_252817


namespace triangle_in_quadrilateral_probability_l252_252065

-- Definitions based on the conditions given in the problem:

def parabola_equation (p : ℝ) (y x : ℝ) : Prop := y^2 = 2 * p * x

def parabola_focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def directrix_equation (p : ℝ) (x : ℝ) : Prop := x = -p / 2

def intersects_points (F : (ℝ × ℝ)) (p : ℝ) (A B : ℝ × ℝ) (d : ℝ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  A ≠ B ∧ parabola_equation p y1 x1 ∧ parabola_equation p y2 x2 ∧
  dist A B = d

def projections (A B : ℝ × ℝ) (p : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let (_, y1) := A in
  let (_, y2) := B in
  ((-p / 2, y1), (-p / 2, y2))

-- The theorem stating the proof problem:
theorem triangle_in_quadrilateral_probability (p : ℝ) (h_pos : p > 0) :
  ∀ (A B : ℝ × ℝ),
  let F := parabola_focus p in
  let d := 3 * p in
  intersects_points F p A B d →
  let (A', B') := projections A B p in
  let area_triangle := (1 / 2 * p * dist A' B') in
  let area_quadrilateral := dist A' B' * (3 * p) in
  (area_triangle / area_quadrilateral) = (1 / 3) :=
begin
  sorry
end

end triangle_in_quadrilateral_probability_l252_252065


namespace correctness_of_option_C_l252_252937

noncomputable def vec_a : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3 / 2, -1/2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def is_orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem correctness_of_option_C :
  is_orthogonal (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) :=
by
  sorry

end correctness_of_option_C_l252_252937


namespace root_constraints_between_zero_and_twoR_l252_252661

variable (R l a : ℝ)
variable (hR : R > 0) (hl : l > 0) (ha_nonzero : a ≠ 0)

theorem root_constraints_between_zero_and_twoR :
  ∀ (x : ℝ), (2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2 = 0) →
  (0 < x ∧ x < 2 * R) ↔
  (a > 0 ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (a < 0 ∧ -2 * R < a ∧ l^2 < (2 * R - a)^2) :=
sorry

end root_constraints_between_zero_and_twoR_l252_252661


namespace y_value_l252_252346

theorem y_value (y : ℝ) : sqrt (2 + sqrt (3 * y - 4)) = sqrt 10 → y = 68 / 3 :=
by
  sorry

end y_value_l252_252346


namespace initial_apps_l252_252463

-- Define the initial condition stating the number of files Dave had initially
def files_initial : ℕ := 21

-- Define the condition after deletion
def apps_after_deletion : ℕ := 3
def files_after_deletion : ℕ := 7

-- Define the number of files deleted
def files_deleted : ℕ := 14

-- Prove that the initial number of apps Dave had was 3
theorem initial_apps (a : ℕ) (h1 : files_initial = 21) 
(h2 : files_after_deletion = 7) 
(h3 : files_deleted = 14) 
(h4 : a - 3 = 0) : a = 3 :=
by sorry

end initial_apps_l252_252463


namespace concyclic_points_l252_252612

variables {A B C E H K M : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space E] [metric_space H] [metric_space K] [metric_space M]
variables [has_dist A] [has_dist B] [has_dist C] [has_dist E] [has_dist H] [has_dist K] [has_dist M]

/-- Prove that points E, H, K, and M lie on the same circle. -/
theorem concyclic_points 
  {α β γ : ℝ}
  (obtuse_angle_C : γ > 90)
  (eq_AC_AH : dist A H = dist A C)
  (eq_BC_BE : dist B E = dist B C)
  (eq_AE_AK : dist A E = dist A K)
  (eq_BH_BM : dist B H = dist B M) :
  ∃ (O : Type*) [metric_space O] [has_dist O] (r : ℝ), 
    r > 0 ∧ dist O E = r ∧ dist O H = r ∧ dist O K = r ∧ dist O M = r :=
by 
  sorry

end concyclic_points_l252_252612


namespace find_m_l252_252123

def point (α : Type) := (α × α)

def collinear {α : Type} [LinearOrderedField α] 
  (p1 p2 p3 : point α) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

theorem find_m {m : ℚ} 
  (h : collinear (4, 10) (-3, m) (-12, 5)) : 
  m = 125 / 16 :=
by sorry

end find_m_l252_252123


namespace count_terminating_decimals_l252_252072

theorem count_terminating_decimals :
  {n : ℤ | 1 ≤ n ∧ n ≤ 510 ∧ 17 ∣ n}.finite.to_finset.card = 30 :=
by
  sorry

end count_terminating_decimals_l252_252072


namespace cube_volume_within_pyramid_l252_252427

def square_pyramid_base_side : ℝ := 2
def pyramid_lateral_faces_are_right_angle_isosceles : Prop := true

noncomputable def cube_side_length : ℝ := 1 + Real.sqrt 3

theorem cube_volume_within_pyramid : 
  (∃ cube_volume : ℝ, cube_volume = cube_side_length ^ 3) ∧ 
  cube_volume_of_base_side_2_and_right_angled_isosceles_lateral_faces := 
  exists.intro (cube_side_length ^ 3)
  (
    begin
      split,
      { refl },
      { sorry },
    end
  )

end cube_volume_within_pyramid_l252_252427


namespace find_multiple_of_brothers_l252_252104

theorem find_multiple_of_brothers : 
  ∃ x : ℕ, (x * 4) - 2 = 6 :=
by
  -- Provide the correct Lean statement for the problem
  sorry

end find_multiple_of_brothers_l252_252104


namespace shots_and_hits_l252_252834

theorem shots_and_hits (n k : ℕ) (h₀ : 10 < n) (h₁ : n < 20) (h₂ : 5 * k = 3 * (n - k)) : (n = 16) ∧ (k = 6) :=
by {
  -- We state the result that we wish to prove
  sorry
}

end shots_and_hits_l252_252834


namespace set_intersection_l252_252223

theorem set_intersection (M : Set ℤ) (N : Set ℝ):
  (M = {-2, -1, 0, 1, 2}) →
  (N = {x | (x-2) / (x+1) ≤ 0}) →
  (M ∩ N = {0, 1, 2}) :=
by
  intros hM hN
  sorry

end set_intersection_l252_252223


namespace values_of_a_and_b_l252_252592

theorem values_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2 * x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) → (a = 2 ∧ b = 1) :=
by 
  sorry

end values_of_a_and_b_l252_252592


namespace total_cost_of_repair_l252_252317

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end total_cost_of_repair_l252_252317


namespace ab_value_l252_252959

theorem ab_value 
  (a b c : ℝ)
  (h1 : a - b = 5)
  (h2 : a^2 + b^2 = 34)
  (h3 : a^3 - b^3 = 30)
  (h4 : a^2 + b^2 - c^2 = 50)
  (h5 : c = 2 * a - b) : 
  a * b = 17 := 
by 
  sorry

end ab_value_l252_252959


namespace min_distance_point_on_ellipse_l252_252885

noncomputable def ellipse := {p : ℝ × ℝ | ∃ (θ : ℝ), p = (4 * Real.cos θ, 2 * Real.sqrt 3 * Real.sin θ)}
def line := {p : ℝ × ℝ | p.1 - 2 * p.2 - 12 = 0}

noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - 4 * Real.sqrt 3 * p.2 - 12| / Real.sqrt 5

theorem min_distance_point_on_ellipse : 
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧ ∀ q ∈ ellipse, distance_to_line p ≤ distance_to_line q ∧
  distance_to_line p = 4 * Real.sqrt 5 / 5 :=
begin
  use (2, -3),
  split,
  { use 2 * Real.pi / 3,
    simp [ellipse, Real.cos, Real.sin, Real.sqrt, div_eq_mul_inv],
    norm_num },
  split,
  { intros q hq,
    dsimp [distance_to_line],
    sorry },
  { dsimp [distance_to_line],
    simp [Real.sqrt, Real.abs, div_eq_mul_inv],
    norm_num }
end

end min_distance_point_on_ellipse_l252_252885


namespace find_x_l252_252881

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l252_252881


namespace greatest_prime_factor_of_253_l252_252388

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def prime_factors (n : ℕ) : list ℕ :=
  (list.range n).filter (λ p, is_prime p ∧ n % p = 0)

noncomputable def greatest_prime_factor (n : ℕ) : ℕ :=
  (prime_factors n).maximum'

theorem greatest_prime_factor_of_253 : greatest_prime_factor 253 = 23 := by
  sorry

end greatest_prime_factor_of_253_l252_252388


namespace radius_of_circumcircle_of_triangle_l252_252766

theorem radius_of_circumcircle_of_triangle
  (R1 R2 : ℝ)
  (H1 : R1 + R2 = 12)
  (H2 : 4 * Real.sqrt 29 = Real.sqrt ((4 - 0) ^ 2 + (-4 - 0) ^ 2 + (R2 - R1) ^ 2))
  (radius_third_sphere : ℝ := 8) 
  (center_third_sphere_touches : ∀ A B C : Point, touches (Sphere.mk A 8) (Sphere.mk B 8) (Sphere.mk C 8)) :
  radius_of_circumcircle ABC = 4 * Real.sqrt 5 :=
sorry

end radius_of_circumcircle_of_triangle_l252_252766


namespace average_side_length_of_squares_l252_252731

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252731


namespace perpendicular_chords_exist_l252_252406

structure Circle (P : Type) :=
  (points : P)
  (arcs : finset ℕ)
  (lengths : points → ℕ)
  (n : ℕ)

axiom divide_circle : Circle ℕ × ℕ → Prop

def exists_perpendicular_chords (C : Circle ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 : ℕ), p1 ≠ p2 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p2 ≠ p4 ∧
  divide_circle (p1, p2) C ∧ divide_circle (p3, p4) C ∧
  p1 * p3 + p2 * p4 = 0  -- condition for perpendicular chords

theorem perpendicular_chords_exist (C : Circle ℕ) : exists_perpendicular_chords C :=
sorry

end perpendicular_chords_exist_l252_252406


namespace infinite_product_simplified_l252_252139

theorem infinite_product_simplified :
  (∏ (n : ℕ) in (Finset.range 1000), 3 ^ (n / (4 ^ n))) = (81 : ℝ) ^ (1 / 9) :=
by
  sorry

end infinite_product_simplified_l252_252139


namespace no_real_solutions_l252_252367

theorem no_real_solutions (x : ℝ) : ¬ (x > 0 ∧ x^{4 - log 10 x} = 100 * x^2) := 
sorry

end no_real_solutions_l252_252367


namespace range_of_f_l252_252496

noncomputable def f : ℝ → ℝ := λ x, (6 * x + 15) / (x - 5)

theorem range_of_f :
  set.range f = set.Ioi (6 : ℝ) ∪ set.Iio 6 := sorry

end range_of_f_l252_252496


namespace cone_height_correct_l252_252415

-- Define the base radius of the cylinder and cone
def r_cylinder := 8 -- cm
def r_cone := 8 -- cm

-- Define the height of the cylinder
def h_cylinder := 2 -- cm

-- Calculate the volume of the cylinder
def V_cylinder := (Real.pi * r_cylinder^2 * h_cylinder)

-- Define the volume of the cone, equating it to the volume of the cylinder to find the height of the cone
def h_cone := 6 -- cm

theorem cone_height_correct :
  V_cylinder = (1 / 3) * Real.pi * r_cone^2 * h_cone :=
by
  -- V_cylinder is defined as (128 * Real.pi) cm^3
  -- We need to show that it equals the volume formula for the cone with h_cone = 6 cm
  sorry

end cone_height_correct_l252_252415


namespace find_angle_of_third_cone_l252_252376

noncomputable def angle_of_third_cone := 2 * Real.arccot (2 * (Real.sqrt 3 + Real.sqrt 2))

theorem find_angle_of_third_cone
  (A : Point)
  (pi_plane : PlanePassingThrough A)
  (cone1 cone2 cone3 : Cone)
  (h1 : ConesWithCommonVertex A cone1 cone2 cone3)
  (h2 : VertexAngle cone1 = π / 3)
  (h3 : VertexAngle cone2 = π / 3)
  (h4 : TouchesPlane cone1 pi_plane)
  (h5 : TouchesPlane cone2 pi_plane)
  (h6 : TouchesPlane cone3 pi_plane) :
  VertexAngle cone3 = angle_of_third_cone :=
sorry

end find_angle_of_third_cone_l252_252376


namespace additive_inverse_2023_l252_252357

theorem additive_inverse_2023 : 
  ∃ y : ℤ, 2023 + y = 0 ∧ y = -2023 := 
by {
  use -2023,
  split,
  { 
    norm_num,
  },
  {
    refl,
  }
}

end additive_inverse_2023_l252_252357


namespace find_slope_angle_l252_252298

-- Define the differentiable function f
variable {f : ℝ → ℝ}

-- Given condition: f is differentiable
variable (h_diff : Differentiable ℝ f)

-- Given condition: the limit of the difference quotient centered at 1
def limit_condition : Prop :=
  ∀ Δx : ℝ, Δx ≠ 0 → filter.tendsto (λ Δx, (f (1 + 2 * Δx) - f (1 - Δx)) / Δx) filter.at_top (𝓝 3)

-- Goal: the slope angle of the tangent line at (1, f 1) is π/4
theorem find_slope_angle (h_limit : limit_condition f) : 
  ∃ α : ℝ, 0 ≤ α ∧ α < (2 * Real.pi) ∧ Real.tan α = 1 ∧ α = Real.pi / 4 :=
by
  sorry

end find_slope_angle_l252_252298


namespace range_of_a_l252_252962

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ -x^2 + 4*x + a = 0) ↔ (-3 ≤ a ∧ a ≤ 21) :=
by
  sorry

end range_of_a_l252_252962


namespace ratio_of_larger_to_smaller_l252_252017

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_larger_to_smaller_l252_252017


namespace coefficient_x2_in_expansion_l252_252354

theorem coefficient_x2_in_expansion :
  let expansion1 := (x + 1/x) * (x + 2)^5 in
  let c5_4 := choose 5 4 in
  let c5_2 := choose 5 2 in
  let term1 := c5_4 * 2^4 in
  let term2 := c5_2 * 2^2 in
  (term1 + term2) = 120 := by 
  sorry

end coefficient_x2_in_expansion_l252_252354


namespace sum_of_perpendicular_distances_eq_height_l252_252426

-- Definition of an equilateral triangle with side length s
def equilateral_triangle (s : ℝ) : Type :=
{ a b c : (ℝ × ℝ) // (dist a b = s) ∧ (dist b c = s) ∧ (dist c a = s) }

-- Definition of the height of an equilateral triangle
noncomputable def height {s : ℝ} (t : equilateral_triangle s) : ℝ :=
  (sqrt 3 / 2) * s

-- The theorem stating that the sum of the perpendicular distances from any internal point to the sides is equal to the height of the equilateral triangle
theorem sum_of_perpendicular_distances_eq_height {s : ℝ} (t : equilateral_triangle s) (P : ℝ × ℝ) :
  -- Assuming P is inside the triangle, which we can formalize but omit for simplicity
  let d1 := dist P (closest_point_on_line_segment P (t.1.1, t.1.2))
  let d2 := dist P (closest_point_on_line_segment P (t.1.2, t.1.3))
  let d3 := dist P (closest_point_on_line_segment P (t.1.3, t.1.1))
  in
  d1 + d2 + d3 = height t :=
sorry -- Proof omitted

end sum_of_perpendicular_distances_eq_height_l252_252426


namespace good_numbers_count_l252_252584

def is_good_number (x : ℕ) : Prop := x - (Nat.floor (Real.sqrt x)).toNat ^ 2 = 9

def count_good_numbers (n : ℕ) : ℕ :=
  (List.range n).filter is_good_number |>.length

theorem good_numbers_count : count_good_numbers 2014 = 40 := by
  sorry

end good_numbers_count_l252_252584


namespace fantasia_max_capacity_reach_l252_252253

def acre_per_person := 1
def land_acres := 40000
def base_population := 500
def population_growth_factor := 4
def years_per_growth_period := 20

def maximum_capacity := land_acres / acre_per_person

def population_at_time (years_from_2000 : ℕ) : ℕ :=
  base_population * population_growth_factor^(years_from_2000 / years_per_growth_period)

theorem fantasia_max_capacity_reach :
  ∃ t : ℕ, t = 60 ∧ population_at_time t = maximum_capacity := by sorry

end fantasia_max_capacity_reach_l252_252253


namespace no_sum_of_consecutive_integers_l252_252339

def sum_consecutive (n r : ℕ) (h : r > 1) : Prop :=
  ∃ (n_0 : ℕ),  n = r * n_0 + (r * (r - 1)) / 2

theorem no_sum_of_consecutive_integers (n : ℕ) (h : ∀ r > 1, ¬ sum_consecutive n r h) : 
  ∃ l : ℕ, n = 2^l :=
by
  sorry

end no_sum_of_consecutive_integers_l252_252339


namespace average_side_length_of_squares_l252_252686

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252686


namespace right_triangle_legs_sum_l252_252832

theorem right_triangle_legs_sum (x : ℕ) (hx1 : x * x + (x + 1) * (x + 1) = 41 * 41) : x + (x + 1) = 59 :=
by sorry

end right_triangle_legs_sum_l252_252832


namespace tan_13pi_div_3_eq_sqrt_3_l252_252165

theorem tan_13pi_div_3_eq_sqrt_3 : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 :=
  sorry

end tan_13pi_div_3_eq_sqrt_3_l252_252165


namespace M_closed_under_multiplication_l252_252013

def M : Set ℕ := {n | ∃ k : ℕ+, n = k ^ 2}

theorem M_closed_under_multiplication : ∀ (a b : ℕ+), (a ^ 2) * (b ^ 2) ∈ M :=
by
  -- Proof skipped
  sorry

end M_closed_under_multiplication_l252_252013


namespace greatest_value_inequality_l252_252493

theorem greatest_value_inequality (x : ℝ) :
  x^2 - 6 * x + 8 ≤ 0 → x ≤ 4 := 
sorry

end greatest_value_inequality_l252_252493


namespace area_of_triangle_abc_l252_252794

-- Definitions representing the conditions
def triangle_abc (A B C : Point) : Prop :=
  ∃ L : Point, 
    (angle_bisector B A C L) ∧ 
    (dist A L = 3) ∧ 
    (dist B L = 6 * Real.sqrt 5) ∧ 
    (dist C L = 4)

-- The theorem we want to prove
theorem area_of_triangle_abc {A B C : Point} (h : triangle_abc A B C) : 
  area_triangle A B C = 21 * Real.sqrt 55 / 4 :=
sorry

end area_of_triangle_abc_l252_252794


namespace cupric_cyanide_formed_l252_252569

-- Definition of the problem
def formonitrile : ℕ := 6
def copper_sulfate : ℕ := 3
def sulfuric_acid : ℕ := 3

-- Stoichiometry from the balanced equation
def stoichiometry (hcn mol_multiplier: ℕ): ℕ := 
  (hcn / mol_multiplier)

theorem cupric_cyanide_formed :
  stoichiometry formonitrile 2 = 3 := 
sorry

end cupric_cyanide_formed_l252_252569


namespace gcd_lcm_ratio_l252_252240

noncomputable def A := 4 * k
noncomputable def B := 5 * k

theorem gcd_lcm_ratio (k : ℕ) 
  (hA : A = 4 * k) 
  (hB : B = 5 * k) 
  (h_lcm : Nat.lcm A B = 180) : Nat.gcd A B = 9 
by 
  sorry

end gcd_lcm_ratio_l252_252240


namespace bucket_fill_proof_l252_252397

variables (x y : ℕ)
def tank_capacity : ℕ := 4 * x

theorem bucket_fill_proof (hx: y = x + 4) (hy: 4 * x = 3 * y): tank_capacity x = 48 :=
by {
  -- Proof steps will be here, but are elided for now
  sorry 
}

end bucket_fill_proof_l252_252397


namespace sample_size_correct_l252_252043

noncomputable def sample_size (P : ℝ) (ε : ℝ) : ℝ :=
  let t := 3.28 in  -- z-score for confidence level P = 0.99896
  let p := 0.5 in  -- worst-case scenario for p maximizing p(1-p)
  (t^2 * p * (1 - p)) / (ε^2)

theorem sample_size_correct :
  sample_size 0.99896 0.05 ≈ 1076 :=
by 
  have P := 0.99896
  have ε := 0.05
  have t := 3.28   -- corresponds to P = 0.99896
  have p := 0.5    -- worst-case scenario
  have n := (t^2 * p * (1 - p)) / (ε^2)
  sorry

end sample_size_correct_l252_252043


namespace find_x_l252_252882

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l252_252882


namespace circle_area_l252_252118

-- Condition: Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 10 * x + 4 * y + 20 = 0

-- Theorem: The area enclosed by the given circle equation is 9π
theorem circle_area : ∀ x y : ℝ, circle_eq x y → ∃ A : ℝ, A = 9 * Real.pi :=
by
  intros
  sorry

end circle_area_l252_252118


namespace tan_squared_of_cos_l252_252978

theorem tan_squared_of_cos (α : ℝ) (h : real.cos α = real.sqrt 3 / 3) : real.tan α ^ 2 = 2 :=
sorry

end tan_squared_of_cos_l252_252978


namespace find_angle_ECD_l252_252600

theorem find_angle_ECD (A B C D E : Type) 
  (angle_A : MeasureTheory.Angle) (angle_B : MeasureTheory.Angle) 
  (angle_D : MeasureTheory.Angle) (angle_BCE : MeasureTheory.Angle) :
  angle_A = 70 ∧ angle_B = 110 ∧ angle_D = 85 ∧ angle_BCE = 45 → 
  ∃ angle_ECD : MeasureTheory.Angle, angle_ECD = 155 :=
by
  assume h : (angle_A = 70) ∧ (angle_B = 110) ∧ (angle_D = 85) ∧ (angle_BCE = 45)
  sorry

end find_angle_ECD_l252_252600


namespace angle_between_vectors_l252_252956

noncomputable def vector_angle (a b : ℝ × ℝ × ℝ) := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / (Real.sqrt (a.1^2 + a.2^2 + a.3^2) * Real.sqrt (b.1^2 + b.2^2 + b.3^2)))

theorem angle_between_vectors (a b : ℝ × ℝ × ℝ) 
  (h₁ : Real.sqrt (a.1^2 + a.2^2 + a.3^2) = 2)
  (h₂ : Real.sqrt (b.1^2 + b.2^2 + b.3^2) = 3)
  (h₃ : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2) = Real.sqrt 7) :
  vector_angle a b = π / 3 :=
sorry

end angle_between_vectors_l252_252956


namespace sequence_an_is_arithmetic_sequence_general_form_of_an_sum_of_bn_l252_252012

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := 2^(n + 1) * a n / (a n + 2^n)

def sequence_an_arithmetic (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → (2^(n + 1)) / a (n + 1) = (2^n) / a n + 1

theorem sequence_an_is_arithmetic_sequence (a : ℕ → ℝ) (h1 : a 0 = 1)
    (h2 : ∀ n, a (n + 1) = (2^(n + 1) * a n) / (a n + 2^n))
    : sequence_an_arithmetic a := by sorry

theorem general_form_of_an (a : ℕ → ℝ) (h1 : a 0 = 1)
    (h2 : ∀ n, a (n + 1) = (2^(n + 1) * a n) / (a n + 2^n))
    : ∀ n : ℕ, a n = 2^n / (n + 1) := by sorry

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := (2 * n - 1) * (n + 1) * a n

def sum_b (b : ℕ → ℝ) : ℕ → ℝ
| 0     := b 0
| (n+1) := sum_b n + b (n + 1)

theorem sum_of_bn (a : ℕ → ℝ)
    (h1 : a 0 = 1)
    (h2 : ∀ n, a (n + 1) = (2^(n + 1) * a n) / (a n + 2^n))
    (h3 : ∀ n : ℕ, a n = 2^n / (n + 1))
    : ∀ n : ℕ, sum_b (b a) n = (2 * n - 3) * 2^(n + 1) + 6 := by sorry

end sequence_an_is_arithmetic_sequence_general_form_of_an_sum_of_bn_l252_252012


namespace solve_for_x_l252_252987

theorem solve_for_x (x : ℝ) (h : 3 - 1 / (1 - x) = 2 * (1 / (1 - x))) : x = 0 :=
by
  sorry

end solve_for_x_l252_252987


namespace average_side_length_of_squares_l252_252712

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252712


namespace simplify_expression_eq_zero_collinear_points_l252_252073

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  (Real.sin (-x) * Real.cos (Real.pi - x)) / (Real.sin (Real.pi + x) * Real.cos (2 * Real.pi - x))
  - (Real.sin (Real.pi - x) * Real.cos (Real.pi + x)) / (Real.cos (Real.pi / 2 - x) * Real.cos (-x))

theorem simplify_expression_eq_zero (x : ℝ) : simplify_expression x = 0 :=
  sorry

structure Point (ℝ : Type) := 
(x : ℝ) (y : ℝ)

def A (k : ℝ) : Point ℝ := ⟨k, 12⟩
def B : Point ℝ := ⟨4, 5⟩
def C (k : ℝ) : Point ℝ := ⟨10, k⟩

def vec (p1 p2 : Point ℝ) : Point ℝ :=
⟨p2.x - p1.x, p2.y - p1.y⟩

def perp (p : Point ℝ) : Point ℝ :=
  ⟨-p.y, p.x⟩

def dot (p1 p2 : Point ℝ) :=
  p1.x * p2.x + p1.y * p2.y

def collinear (p1 p2 p3 : Point ℝ) : Prop :=
(dot (vec p1 p2) (perp (vec p2 p3)) = 0)

theorem collinear_points (k : ℝ) : k = -2 ∨ k = 11 ↔ collinear (A k) B (C k) :=
  sorry

end simplify_expression_eq_zero_collinear_points_l252_252073


namespace average_side_length_of_squares_l252_252721

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252721


namespace planes_parallel_l252_252174

variables (Plane Line : Type) [InnerProductSpace ℝ Plane] [InnerProductSpace ℝ Line]

variables (alpha beta : Plane) (l m n : Line)

-- Defining conditions
variable (h1 : m ≠ n)
variable (h2 : alpha ≠ beta)
variable (h3 : m.orthogonal alpha)
variable (h4 : n.orthogonal beta)
variable (h5 : m.parallel n)

-- Statement to prove
theorem planes_parallel (h1 : m ≠ n) (h2 : alpha ≠ beta) 
  (h3 : m.orthogonal alpha) (h4 : n.orthogonal beta) 
  (h5 : m.parallel n) : alpha.parallel beta := 
sorry

end planes_parallel_l252_252174


namespace parabola_vertex_coordinate_l252_252904

theorem parabola_vertex_coordinate :
  ∀ x_P : ℝ, 
  (P : ℝ × ℝ) → 
  (P = (x_P, 1/2 * x_P^2)) → 
  (dist P (0, 1/2) = 3) →
  P.2 = 5 / 2 :=
by sorry

end parabola_vertex_coordinate_l252_252904


namespace albert_needs_more_money_l252_252838

def cost_paintbrush : Real := 1.50
def cost_paints : Real := 4.35
def cost_easel : Real := 12.65
def cost_canvas : Real := 7.95
def cost_palette : Real := 3.75
def money_albert_has : Real := 10.60
def total_cost : Real := cost_paintbrush + cost_paints + cost_easel + cost_canvas + cost_palette
def money_needed : Real := total_cost - money_albert_has

theorem albert_needs_more_money : money_needed = 19.60 := by
  sorry

end albert_needs_more_money_l252_252838


namespace find_analytic_expression_and_extreme_values_l252_252985

noncomputable def f (x : ℝ) : ℝ := -3 * x - (8 / x) + 11

theorem find_analytic_expression_and_extreme_values :
  (∀ a b c : ℝ, f a = a * a - b / a + c →
      (a = 1 ∧ f 1 = 0) ∧ (a = 2 ∧ f 2 = 1) ∧ (a = 2 ∧ (deriv f 2 = -1)) →
      (f = -3 * x - 8 / x + 11)) ∧
  (∀ x ∈ {(-2 * sqrt(6) / 3), (2 * sqrt(6) / 3)},
      ((x = -2 * sqrt(6) / 3) → (f x = 11 + 4 * sqrt(6))) ∧
      ((x = 2 * sqrt(6) / 3) → (f x = 11 - 4 * sqrt(6)))) :=
by
  sorry

end find_analytic_expression_and_extreme_values_l252_252985


namespace highest_power_of_three_divides_M_l252_252671

/--
Given the number \( M \), formed by concatenating the 2-digit integers from 24 to 87, 
prove that the highest power of 3 that divides \( M \) is 3^1.
-/
theorem highest_power_of_three_divides_M :
  let M := (24.toString ++ 25.toString ++ 26.toString ++ 27.toString ++ 28.toString ++ 29.toString ++
         30.toString ++ 31.toString ++ 32.toString ++ 33.toString ++ 34.toString ++ 35.toString ++
         36.toString ++ 37.toString ++ 38.toString ++ 39.toString ++ 40.toString ++ 41.toString ++
         42.toString ++ 43.toString ++ 44.toString ++ 45.toString ++ 46.toString ++ 47.toString ++
         48.toString ++ 49.toString ++ 50.toString ++ 51.toString ++ 52.toString ++ 53.toString ++
         54.toString ++ 55.toString ++ 56.toString ++ 57.toString ++ 58.toString ++ 59.toString ++
         60.toString ++ 61.toString ++ 62.toString ++ 63.toString ++ 64.toString ++ 65.toString ++
         66.toString ++ 67.toString ++ 68.toString ++ 69.toString ++ 70.toString ++ 71.toString ++
         72.toString ++ 73.toString ++ 74.toString ++ 75.toString ++ 76.toString ++ 77.toString ++
         78.toString ++ 79.toString ++ 80.toString ++ 81.toString ++ 82.toString ++ 83.toString ++
         84.toString ++ 85.toString ++ 86.toString ++ 87.toString).toNat in
  ∃ j, 3^j ∣ M ∧ ∀ k, 3^k ∣ M → k ≤ j := by
  sorry

end highest_power_of_three_divides_M_l252_252671


namespace distance_between_x_intercepts_l252_252823

noncomputable def slope_intercept_form (m : ℝ) (x1 y1 x : ℝ) : ℝ :=
  m * (x - x1) + y1

def x_intercept (m : ℝ) (x1 y1 : ℝ) : ℝ :=
  (y1 - m * x1) / m

theorem distance_between_x_intercepts : 
  ∀ (m1 m2 : ℝ) (x1 y1 : ℝ), 
  m1 = 4 → m2 = -2 → x1 = 8 → y1 = 20 →
  abs (x_intercept m1 x1 y1 - x_intercept m2 x1 y1) = 15 :=
by
  intros m1 m2 x1 y1 h_m1 h_m2 h_x1 h_y1
  rw [h_m1, h_m2, h_x1, h_y1]
  sorry

end distance_between_x_intercepts_l252_252823


namespace problem_solution_correct_l252_252360

theorem problem_solution_correct (k m b : ℝ) (hk : k ≠ 0) (hm : m ≠ 0) :
  (∃ x, y1 x = y2 x) →
  (∀ x, y1 (1:ℝ) = 2 ∧ y2 (1:ℝ) = 2 → k * 1 + b = m * 1 + 3) ∧
  ((∀ x_a x_b, x_a ≠ x_b → (x_a - x_b) * (y2 x_a - y2 x_b) < 0) ∧
  (b < 3 ∧ b ≠ 2 → ∀ x > 1, y1 x > y2 x)) :=
begin
  sorry
end

/-- Definitions for the linear functions y1 and y2 --/
def y1 (k b x : ℝ) : ℝ := k * x + b
def y2 (m x : ℝ) : ℝ := m * x + 3

end problem_solution_correct_l252_252360


namespace area_of_closed_figure_eq_32_over_3_l252_252488

noncomputable def area_of_closed_figure : ℝ :=
  ∫ y in -1..3, (2 * y + 3) - y^2

theorem area_of_closed_figure_eq_32_over_3 :
  area_of_closed_figure = 32 / 3 :=
by
  sorry

end area_of_closed_figure_eq_32_over_3_l252_252488


namespace angle_decomposition_and_terminal_side_l252_252531

noncomputable def α := 1200 * Real.pi / 180 -- α = 1200° in radians

theorem angle_decomposition_and_terminal_side :
  ∃ (β : ℝ) (k : ℤ), 0 ≤ β ∧ β < 2 * Real.pi ∧ α = β + 2 * k * Real.pi ∧ 
  (Real.pi / 2 < β ∧ β < Real.pi) ∧ -- β lies in the second quadrant
  set.mem β {x | x = 2 * Real.pi / 3 ∨ x = -4 * Real.pi / 3 ∈ set.Icc (-2 * Real.pi) (2 * Real.pi)} :=
begin
  sorry -- Completing the proof is not required
end

end angle_decomposition_and_terminal_side_l252_252531


namespace mark_total_cost_is_correct_l252_252323

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end mark_total_cost_is_correct_l252_252323


namespace mr_c_loses_1560_l252_252089

noncomputable theory
open_locale big_operators

def initial_value : ℝ := 8000
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.2
def tax : ℝ := 200

def first_transaction_price (initial_price : ℝ) (loss_percent : ℝ) : ℝ :=
  initial_price * (1 - loss_percent)

def second_transaction_price (first_price : ℝ) (gain_percent : ℝ) (tax : ℝ) : ℝ :=
  first_price * (1 + gain_percent) + tax

theorem mr_c_loses_1560 :
  let price_after_first_transaction := first_transaction_price initial_value loss_percentage,
      price_after_second_transaction := second_transaction_price price_after_first_transaction gain_percentage tax in
  initial_value - (price_after_second_transaction - price_after_first_transaction) = -1560 :=
by sorry

end mr_c_loses_1560_l252_252089


namespace area_of_shaded_region_l252_252605

theorem area_of_shaded_region 
  (ABCD : Type) 
  (BC : ℝ)
  (height : ℝ)
  (BE : ℝ)
  (CF : ℝ)
  (BC_length : BC = 12)
  (height_length : height = 10)
  (BE_length : BE = 5)
  (CF_length : CF = 3) :
  (BC * height - (1 / 2 * BE * height) - (1 / 2 * CF * height)) = 80 :=
by
  sorry

end area_of_shaded_region_l252_252605


namespace find_b_l252_252639

def h (x : ℝ) : ℝ :=
  if x < 0 then x else 3 * x - 17

theorem find_b (b : ℝ) (hb : b < 0) :
  h(h(h(7))) = h(h(h(b))) → b = -5 :=
by
  sorry

end find_b_l252_252639


namespace average_side_length_of_squares_l252_252684

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252684


namespace cube_face_parallel_probability_l252_252172

theorem cube_face_parallel_probability :
  ∃ (n m : ℕ), (n = 15) ∧ (m = 3) ∧ (m / n = (1 / 5 : ℝ)) := 
sorry

end cube_face_parallel_probability_l252_252172


namespace tangent_line_at_0_l252_252492

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem tangent_line_at_0 : 
  let m := (deriv f 0)
  let y₀ := f 0
  in ∀ x, (y = -x - 2) := 
by 
  sorry

end tangent_line_at_0_l252_252492


namespace quadratic_real_roots_k_range_shared_root_quadratic_m_l252_252221

theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + 3 * k = 0) → k ≤ 4 / 3 :=
begin
  intro h,
  have discriminant_non_negative : 16 - 12 * k ≥ 0,
  { -- assuming the quadratic equation has real roots,
    -- the discriminant must be non-negative
    sorry },
  linarith,
end

noncomputable def largest_integer_k := 1  -- Given largest k satisfying the condition is 1

theorem shared_root_quadratic_m (m : ℝ) :
  (∃ x : ℝ, x = 1 ∨ x = 3 ∧ ((m - 2) * x^2 + x + m - 3 = 0)) →
  m = 9 / 5 :=
begin
  intro h,
  cases h with x hx,
  cases hx with hx_l hx_r,
  { -- Case for shared root x = 1
    sorry },
  { -- Case for shared root x = 3
    sorry },
end

end quadratic_real_roots_k_range_shared_root_quadratic_m_l252_252221


namespace percentage_of_total_is_sixty_l252_252999

def num_boys := 600
def diff_boys_girls := 400
def num_girls := num_boys + diff_boys_girls
def total_people := num_boys + num_girls
def target_number := 960
def target_percentage := (target_number / total_people) * 100

theorem percentage_of_total_is_sixty :
  target_percentage = 60 := by
  sorry

end percentage_of_total_is_sixty_l252_252999


namespace repeating_decimal_seven_eq_seven_ninths_l252_252152

-- Define the conditions of the geometric series.
def a : ℚ := 7 / 10
def r : ℚ := 1 / 10
def S : ℚ := a / (1 - r)

-- Define the repeating decimal.
noncomputable def repeating_decimal_seven : ℚ := 0.7777777...

-- Statement of the problem: Prove that the repeating decimal equals the fraction.
theorem repeating_decimal_seven_eq_seven_ninths : repeating_decimal_seven = 7 / 9 :=
sorry

end repeating_decimal_seven_eq_seven_ninths_l252_252152


namespace integral_f_l252_252207

noncomputable def f (x : ℝ) : ℝ :=
  if h : -2 ≤ x ∧ x ≤ 0 then sqrt (4 - x^2)
  else if h : 0 < x ∧ x ≤ 2 then x + 2
  else 0 -- default value for x outside [-2, 2]

theorem integral_f :
  ∫ x in -2..2, f x = Real.pi + 6 :=
sorry

end integral_f_l252_252207


namespace distance_from_center_to_line_l252_252738

theorem distance_from_center_to_line :
  let center := (0 : ℝ, 0 : ℝ)
  let a := 3
  let b := 4
  let c := -25
  let d := (λ (x y : ℝ), (abs (a * x + b * y + c)) / (real.sqrt (a ^ 2 + b ^ 2)))
  d center.1 center.2 = 5 := 
sorry

end distance_from_center_to_line_l252_252738


namespace ellipse_equation_and_max_area_l252_252950

theorem ellipse_equation_and_max_area (a b c : ℝ) (k : ℝ) (x1 x2 y1 y2 : ℝ)
    (h1 : 2 / a = sqrt 6 / 3)
    (h2 : a^2 = b^2 + c^2)
    (h3 : x1 + x2 = (18 * k^2) / (3 * k^2 + 1))
    (h4 : x1 * x2 = (27 * k^2 - 6) / (3 * k^2 + 1))
    (h5 : y1 = k * (x1 - 3))
    (h6 : y2 = k * (x2 - 3))
    : (∀ x y, x^2 / 6 + y^2 / 2 = 1) ∧
      (∃ x1 x2, x1 < 3 ∧ x2 < 3 ∧ (abs $ k * (3 - x1) * (3 - x2)) ≤ sqrt 3 / 2) :=
by
  sorry

end ellipse_equation_and_max_area_l252_252950


namespace bottles_remaining_after_2_days_l252_252815

def total_bottles := 48 

def first_day_father_consumption := total_bottles / 4
def first_day_mother_consumption := total_bottles / 6
def first_day_son_consumption := total_bottles / 8

def total_first_day_consumption := first_day_father_consumption + first_day_mother_consumption + first_day_son_consumption 
def remaining_after_first_day := total_bottles - total_first_day_consumption

def second_day_father_consumption := remaining_after_first_day / 5
def remaining_after_father := remaining_after_first_day - second_day_father_consumption
def second_day_mother_consumption := remaining_after_father / 7
def remaining_after_mother := remaining_after_father - second_day_mother_consumption
def second_day_son_consumption := remaining_after_mother / 9
def remaining_after_son := remaining_after_mother - second_day_son_consumption
def second_day_daughter_consumption := remaining_after_son / 9
def remaining_after_daughter := remaining_after_son - second_day_daughter_consumption

theorem bottles_remaining_after_2_days : ∀ (total_bottles : ℕ), remaining_after_daughter = 14 := 
by
  sorry

end bottles_remaining_after_2_days_l252_252815


namespace length_PF_l252_252029

theorem length_PF (F A B P : ℝ × ℝ) (h_parabola : ∀ {x y : ℝ}, y^2 = 8 * (x + 2) → y = real.sqrt(8 * (x + 2)))
    (h_focus : F = (0, 0)) 
    (h_line : ∃ m : ℝ, m = real.tan(real.pi / 3) ∧ ∀ x : ℝ, F = (x, m * x))
    (h_inter : ∀ {x y : ℝ}, y = real.sqrt(3) * x → y^2 = 8 * (x + 2) → (x = 4 ∨ x = -8/3))
    (h_midpoint : ∀ x y : ℝ, x = (4 + -8/3) / 2 ∧ y = (4 * real.sqrt 3 + -(8 * real.sqrt 3)/3) / 2)
    (h_bisector : ∃ m_pb : ℝ, m_pb = -1 / real.sqrt 3 ∧
                   ∀ x, y = real.line -1 / real.sqrt 3 * (x - 4 / 3) = -4/3)
    (h_inter_x_axis : P = (16 / 3, 0)) : 
    real.dist F P = 16 / 3 :=
sorry

end length_PF_l252_252029


namespace chess_group_total_games_l252_252025

theorem chess_group_total_games (n : ℕ) (h_n : n = 14) : (n.choose 2) = 91 := 
by {
  rw h_n,
  exact nat.choose_eq_factorial_div_factorial (14 - 2) 2,
  sorry
}

#print chess_group_total_games

end chess_group_total_games_l252_252025


namespace farm_field_proof_l252_252816

section FarmField

variables 
  (planned_rate daily_rate : ℕ) -- planned_rate is 260 hectares/day, daily_rate is 85 hectares/day 
  (extra_days remaining_hectares : ℕ) -- extra_days is 2, remaining_hectares is 40
  (max_hours_per_day : ℕ) -- max_hours_per_day is 12

-- Definitions for soils
variables
  (A_percent B_percent C_percent : ℚ) (A_hours B_hours C_hours : ℕ)
  -- A_percent is 0.4, B_percent is 0.3, C_percent is 0.3
  -- A_hours is 4, B_hours is 6, C_hours is 3

-- Given conditions
axiom planned_rate_eq : planned_rate = 260
axiom daily_rate_eq : daily_rate = 85
axiom extra_days_eq : extra_days = 2
axiom remaining_hectares_eq : remaining_hectares = 40
axiom max_hours_per_day_eq : max_hours_per_day = 12

axiom A_percent_eq : A_percent = 0.4
axiom B_percent_eq : B_percent = 0.3
axiom C_percent_eq : C_percent = 0.3

axiom A_hours_eq : A_hours = 4
axiom B_hours_eq : B_hours = 6
axiom C_hours_eq : C_hours = 3

-- Theorem stating the problem
theorem farm_field_proof :
  ∃ (total_area initial_days : ℕ),
    total_area = 340 ∧ initial_days = 2 :=
by
  sorry

end FarmField

end farm_field_proof_l252_252816


namespace c_n_formula_is_arithmetic_sequence_of_bound_l252_252530

noncomputable def a_n (a_1 : ℝ) (n : ℕ) : ℝ := a_1 + 2 * (n - 1)
noncomputable def S_n (a_1 : ℝ) (n : ℕ) : ℝ := n * a_1 + n * (n - 1)
def b_n (a_n S_n : ℕ → ℝ) (n : ℕ) : ℝ := (a_n (n+1) - S_n n / n) / (n + 1)
def c_n (a_n S_n : ℕ → ℝ) (n : ℕ) : ℝ := (a_n (n+1) + a_n (n+2)) / 2 - S_n n / n
def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop := ∀ n, a_n (n + 1) - a_n n = 2 * a_n n

-- Problem (1)
theorem c_n_formula (a_1 : ℝ) (n : ℕ) (h_arithmetic : ∀ n, a_n a_1 (n + 1) - a_n a_1 n = 2) :
  c_n (a_n a_1) (S_n a_1) n = 1 := sorry

-- Problem (2)
theorem is_arithmetic_sequence_of_bound (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (λ : ℝ)
  (h_b_n : ∀ n, b_n a_n S_n n ≤ λ) (h_c_n : ∀ n, λ ≤ c_n a_n S_n n) :
  is_arithmetic_sequence a_n := sorry

end c_n_formula_is_arithmetic_sequence_of_bound_l252_252530


namespace taco_cheese_amount_l252_252279

theorem taco_cheese_amount :
  ∀ (x : ℕ),
  (∀ burrito_ounces taco_ounces total_ounces : ℕ, 
    burrito_ounces = 4 → 
    total_ounces = 37 →
    7 * burrito_ounces + taco_ounces = total_ounces → 
    taco_ounces = x) → 
  x = 9 :=
by
  intros x h
  apply h
  repeat { sorry }

end taco_cheese_amount_l252_252279


namespace E_score_and_correct_answers_l252_252500

-- Define the assumptions
variables {A B C D E : ℤ}
variables {Q1 Q2 Q3 Q4 Q5 Q6 Q7 : ℤ}

-- Scoring rules
def score (x : ℤ) : ℤ := if x = 1 then 1 else if x = -1 then -1 else 0

-- Scores for each question
def scores (Q1 Q2 Q3 Q4 Q5 Q6 Q7 : ℤ) : list ℤ := [score(Q1), score(Q2), score(Q3), score(Q4), score(Q5), score(Q6), score(Q7)]

-- Conditions
axiom A_score : list.sum (scores Q1 0 (-Q3) Q4 (-Q5) Q6 Q7) = 2
axiom B_score : list.sum (scores Q1 (-Q2) Q3 Q4 (-Q5) (-Q6) 0) = 2
axiom C_score : list.sum (scores 0 Q2 (-Q3) (-Q4) Q5 (-Q6) Q7) = 2
axiom D_score : list.sum (scores (-Q1) (-Q2) (-Q3) Q4 Q5 0 (-Q7)) = 2

-- Goal
theorem E_score_and_correct_answers : 
  ∃ (Q1 Q2 Q3 Q4 Q5 Q6 Q7 : ℤ), 
  let E_score := list.sum (scores Q1 Q2 Q3 Q4 Q5 Q6 Q7) in 
  E_score = 4 ∧ 
  Q1 = 1 ∧ Q2 = -1 ∧ Q3 = -1 ∧ Q4 = 1 ∧ Q5 = 1 ∧ Q6 = -1 ∧ Q7 = 1 :=
sorry

end E_score_and_correct_answers_l252_252500


namespace first_player_wins_l252_252383

-- Definitions for conditions
def board_size := 9
def central_cell := (5, 5)  -- Central cell in a 9x9 board (1-based index)

-- Statement of the problem
theorem first_player_wins : 
  ∀ (X O : ℕ → ℕ → bool),  -- functions representing board positions where crosses and noughts are placed
  (X 1 1 = true) ∧  -- The first player starts at the central cell (example for format)
  (∀ i j, X i j ↔ ¬ O i j) ∧  -- Either X or O, but not both in any cell
  (∀ i j, (i = 5 ∧ j = 5) → X i j = true) →  -- Ensures initial X at central cell
  (∀ i j, (O i j = true) → X (board_size - i + 1) (board_size - j + 1) = true) →  -- Symmetric response from first player
  ∃ (points_for_X points_for_O : ℕ), 
  (points_for_X > points_for_O) :=
sorry

end first_player_wins_l252_252383


namespace number_of_white_lights_l252_252249

theorem number_of_white_lights (red_lights low_brightness med_brightness high_brightness : ℕ) 
                                (yellow_lights blue_lights green_lights purple_lights : ℕ)
                                (extra_blue_lights extra_red_lights : ℕ) 
                                (brightness_conversion : ℕ → ℚ)
                                (h1 : low_brightness = 16)
                                (h2 : med_brightness = 1)
                                (h3 : high_brightness = 1.5)
                                (h4 : yellow_lights = 4)
                                (h5 : blue_lights = 2 * yellow_lights)
                                (h6 : green_lights = 8)
                                (h7 : purple_lights = 3)
                                (h8 : extra_blue_lights = nat.floor (0.25 * (blue_lights : ℚ)))
                                (h9 : extra_red_lights = 10)
                                (total_brightness : ℚ) :
  total_brightness = (low_brightness * 0.5 + high_brightness * 4 * 1.5 + 2 * yellow_lights * med_brightness + 
                      green_lights * 0.5 + purple_lights * 1.5 + extra_blue_lights * med_brightness +
                      extra_red_lights * 0.5) →
  38 := sorry

end number_of_white_lights_l252_252249


namespace problem1_problem2_l252_252808

-- Problem 1: Calculate: sqrt(4/9) - sqrt((-2)^4) + cbrt((19/27)-1) - (-1)^2017 == -5/3
theorem problem1 :
  (Real.sqrt (4 / 9) - Real.sqrt ((-2:ℝ) ^ 4) + (Real.cbrt ((19 / 27:ℝ) - 1)) - (-1) ^ (2017:ℝ)) = (-5 / 3:ℝ) :=
sorry

-- Problem 2: Find the value(s) of x that satisfy the condition: (x-1)^2 = 9.
theorem problem2 (x : ℝ) :
  (x - 1) ^ 2 = 9 ↔ x = 4 ∨ x = -2 :=
sorry

end problem1_problem2_l252_252808


namespace line_intersects_circle_find_slope_angle_equation_of_midpoint_trajectory_l252_252522

noncomputable def point := {x y : ℝ}

def circle (C : point → Prop) := ∀ (P : point), C P ↔ P.x^2 + (P.y - 1)^2 = 5
def line (l : point → Prop) (m : ℝ) := ∀ (P : point), l P ↔ m * P.x - P.y + 1 - m = 0

theorem line_intersects_circle (m : ℝ) :
  ∀ (l : point → Prop), (∀ (P : point), l P ↔ m * P.x - P.y + 1 - m = 0) →
  ∀ (C : point → Prop), (∀ (P : point), C P ↔ P.x^2 + (P.y - 1)^2 = 5) →
  ∃ P₁ P₂ : point, P₁ ≠ P₂ ∧ l P₁ ∧ l P₂ ∧ C P₁ ∧ C P₂ := sorry

theorem find_slope_angle (AB : ℝ) :
  AB = real.sqrt 17 →
  ∀ (m : ℝ), (m = real.sqrt 3 ∨ m = -real.sqrt 3) :=
begin
  sorry
end

theorem equation_of_midpoint_trajectory :
  ∀ (M : point), (∃ (x y : ℝ), x ^ 2 + y ^ 2 - x - 2 * y + 1 = 0) :=
begin
  sorry
end

end line_intersects_circle_find_slope_angle_equation_of_midpoint_trajectory_l252_252522


namespace am_gm_inequality_l252_252948

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : (1 + x) * (1 + y) * (1 + z) ≥ 8 :=
sorry

end am_gm_inequality_l252_252948


namespace polynomial_with_three_roots_count_l252_252124

theorem polynomial_with_three_roots_count :
  let polynomials := {P : polynomial ℝ // ∀ i, i ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → P.coeff i ∈ {0, 1}} in
  let has_three_roots (P : polynomial ℝ) := 
    ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P.eval a = 0 ∧ P.eval b = 0 ∧ P.eval c = 0 in
  let count := (finset.univ.filter (λ (P : polynomials), has_three_roots P.val)).card in
  count = 256 :=
by sorry

end polynomial_with_three_roots_count_l252_252124


namespace maximum_value_l252_252187

variable {a b : ℝ}

theorem maximum_value (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1) :
  ∃ M : ℝ, M = (2 * real.sqrt 3 + 3) / 3 ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (2 * x / (x^2 + y) + y / (x + y^2)) ≤ M) :=
by
  sorry

end maximum_value_l252_252187


namespace f_odd_f_positive_expression_range_of_m_l252_252195

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then (x + 2) * Real.exp (-x) - 2 else (x - 2) * Real.exp x + 2

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_odd : odd_function f :=
sorry

theorem f_positive_expression (x : ℝ) (hx : x > 0) : f x = (x - 2) * Real.exp x + 2 :=
sorry

theorem range_of_m :
  ∀ m, (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = m) ↔ m ∈ Set.Icc (2 - Real.exp 1) 2 :=
sorry

end f_odd_f_positive_expression_range_of_m_l252_252195


namespace simplify_expression_l252_252138

theorem simplify_expression (x : ℝ) : (7 - sqrt (3 * x ^ 2 - 12)) ^ 2 = 3 * x ^ 2 + 37 - 14 * sqrt (3 * x ^ 2 - 12) :=
by
  sorry

end simplify_expression_l252_252138


namespace general_formula_a_sum_of_bn_l252_252966

-- Definitions for conditions
def a (n : ℕ) : ℕ := sorry -- a_n = 2n^2 - n, will be proved
def b (n : ℕ) : ℝ := (-1)^n * (4 * n^2 * (n + 1)) / (a(n) * a(n + 1))
def S (n : ℕ) : ℝ := if n % 2 = 0 then (-2 * n) / (2 * n + 1) else (-2 * n - 2) / (2 * n + 1)

-- Theorem statements
theorem general_formula_a (n : ℕ ) : a n = 2 * n^2 - n :=
sorry

theorem sum_of_bn (n : ℕ ) : Σ i in finset.range n, b i = S n :=
sorry

end general_formula_a_sum_of_bn_l252_252966


namespace find_x_value_l252_252022

theorem find_x_value
  (B C D E A : Point)
  (h1 : Linear B C D)
  (h2 : angle B C D = 125)
  (h3 : angle B A C = 50)
  (h4 : Linear E A D)
  (h5 : angle D A E = 80) :
  let x := 180 - (80 + 50) - (180 - 125) in
  x = 25 :=
by
  sorry

end find_x_value_l252_252022


namespace find_true_discount_l252_252063

-- Here, we set noncomputable to avoid any computational issues with real numbers.
noncomputable def true_discount (PW BG : ℝ) : ℝ :=
  let TD := BG in TD

-- Now we assert the statement we need to prove
theorem find_true_discount {PW BG : ℝ} (hPW : PW = 576) (hBG : BG = 16) : 
  true_discount PW BG = 16 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end find_true_discount_l252_252063


namespace decimal_to_binary_27_l252_252875

theorem decimal_to_binary_27 : nat.digits 2 27 = [1, 1, 0, 1, 1] :=
by sorry

end decimal_to_binary_27_l252_252875


namespace sum_odd_integers_correct_l252_252780

def sum_odd_integers_from_13_to_41 : ℕ := 
  let a := 13
  let l := 41
  let n := 15
  n * (a + l) / 2

theorem sum_odd_integers_correct : sum_odd_integers_from_13_to_41 = 405 :=
  by sorry

end sum_odd_integers_correct_l252_252780


namespace evaluate_statements_l252_252209

noncomputable def f (x : Real) (m n : Real) : Real :=
  m * Real.sin x + n * Real.cos x

theorem evaluate_statements (m n : Real) (h : m ≠ 0 ∧ n ≠ 0) :
  (∃ φ : Real, ∃ k : Int, φ = 2*k*Real.pi + Real.pi / 4 ∧ 
  (f (Real.pi / 4) m n = f 0 m n ∧ 
   f (7 * Real.pi / 4) m n = 0 ∧ 
   f (-3 * Real.pi / 4) m n = -((m^2 + n^2).sqrt) ∧ 
   ∀ y : Real, (∃ P1 P2 P3 P4 : Real, P1 < P2 ∧ P2 < P3 ∧ P3 < P4 ∧ 
   f P1 m n = y ∧ f P2 m n = y ∧ f P3 m n = y ∧ f P4 m n = y ∧ 
   P2 - P4 = Real.pi) → False
  ) → 
  3) := sorry

end evaluate_statements_l252_252209


namespace blute_position_l252_252386

-- Define the letters in alphabetical order.
def letters : List Char := ['B', 'E', 'L', 'T', 'U']

-- Define the word for which we want to find the position.
def targetWord : List Char := ['B', 'L', 'U', 'T', 'E']

-- The theorem stating the position of the target word in the sorted permutations of the letters
theorem blute_position :
  (List.permutations letters).sorted (List.Lex.lt) (List.Lex.char_lt) |>.findIndex? (λ word => word = targetWord) == some 10 :=
by sorry

end blute_position_l252_252386


namespace question1_question2_case1_question2_case2_question2_case3_l252_252553

def f (x a : ℝ) : ℝ := x^2 + (1 - a) * x - a

theorem question1 (x : ℝ) (h : (-1 < x) ∧ (x < 3)) : f x 3 < 0 := sorry

theorem question2_case1 (x : ℝ) : f x (-1) > 0 ↔ x ≠ -1 := sorry

theorem question2_case2 (x a : ℝ) (h : a > -1) : f x a > 0 ↔ (x < -1 ∨ x > a) := sorry

theorem question2_case3 (x a : ℝ) (h : a < -1) : f x a > 0 ↔ (x < a ∨ x > -1) := sorry

end question1_question2_case1_question2_case2_question2_case3_l252_252553


namespace celtics_win_finals_in_7_games_l252_252674

noncomputable def prob_lakers_win : ℚ := 3 / 4
noncomputable def prob_celtics_win : ℚ := 1 / 4

theorem celtics_win_finals_in_7_games :
  let prob_exactly_3_wins_in_6_games := (prob_celtics_win ^ 3) * (prob_lakers_win ^ 3) * 20 in
  let prob_win_7th_game := prob_celtics_win in
  let prob_total := prob_exactly_3_wins_in_6_games * prob_win_7th_game in
  prob_total = 27 / 16384 :=
by
  sorry

end celtics_win_finals_in_7_games_l252_252674


namespace find_coordinates_l252_252819

-- Definitions and Statements
def start_point : ℝ × ℝ := (2, 5)

def end_point (x y : ℝ) : Prop :=
  x > 2 ∧ y > 5 ∧ (real.sqrt ((x - 2)^2 + (y - 5)^2)) = 10

theorem find_coordinates (x y : ℝ) (h : end_point x y) : 
  (x = 8 ∧ y = 13) := 
sorry

end find_coordinates_l252_252819


namespace sum_of_even_coeffs_eq_neg_24_l252_252941

noncomputable def polynomial : Polynomial ℝ := (Polynomial.X - 1)^4 * (Polynomial.X + 2)^5

theorem sum_of_even_coeffs_eq_neg_24 :
  let a := polynomial.coeffs;
  a[2] + a[4] + a[6] + a[8] = -24 :=
by
  sorry

end sum_of_even_coeffs_eq_neg_24_l252_252941


namespace problem_statement_l252_252949

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := a n + 2 * b n

noncomputable def b : ℕ → ℝ
| 0       := 1
| (n + 1) := a n + b n

theorem problem_statement : (a 1993) ^ 2 - 2 * (b 1993) ^ 2 = 1 := 
by sorry

end problem_statement_l252_252949


namespace statement_two_statement_three_l252_252912

section
variables {R : Type*} [Field R]
variables (a b c p q : R)
noncomputable def f (x : R) := a * x^2 + b * x + c

-- Statement ②
theorem statement_two (hpq : f a b c p = f a b c q) (hpq_neq : p ≠ q) : 
  f a b c (p + q) = c :=
sorry

-- Statement ③
theorem statement_three (hf : f a b c (p + q) = c) (hpq_neq : p ≠ q) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end

end statement_two_statement_three_l252_252912


namespace remainder_polynomial_division_l252_252156

theorem remainder_polynomial_division (x : ℝ) : 
  ∃ (q : ℝ[X]), (x ^ 5 + 3) = q * (x - 1)^2 + (5 * x - 1) := 
  sorry

end remainder_polynomial_division_l252_252156


namespace total_rooms_to_paint_l252_252827

theorem total_rooms_to_paint (h1 : ∃ t : ℕ, t = 8) 
                              (h2 : ∃ p : ℕ, p = 5) 
                              (h3 : ∃ r : ℕ, r = 32) : 
                              ∃ tr : ℕ, tr = p + (r / t) :=
by {
  use 9,
  have t_def : t = 8 := h1.some_spec,
  have p_def : p = 5 := h2.some_spec,
  have r_def : r = 32 := h3.some_spec,
  rw [p_def, t_def, r_def],
  norm_num,
  sorry
}

end total_rooms_to_paint_l252_252827


namespace problem_triangle_DEF_inradius_perimeter_l252_252144

noncomputable def triangle_incenter (A B C : Point) : Point := sorry

noncomputable def circumcircle (A B C : Point) : Circle := sorry

noncomputable def circumradius (A B C : Point) : ℝ := sorry

noncomputable def circumcircle_intersect 
  (line : Line) (circle : Circle) : Point := sorry
  
noncomputable def perimeter (A B C : Point) : ℝ := sorry

noncomputable def inradius (A B C : Point) : ℝ := sorry

theorem problem_triangle_DEF_inradius_perimeter 
  (A B C : Point) (I : Point := triangle_incenter A B C)
  (circ : Circle := circumcircle A B C) 
  (D : Point := circumcircle_intersect (line_through A I) circ)
  (E : Point := circumcircle_intersect (line_through B I) circ)
  (F : Point := circumcircle_intersect (line_through C I) circ) 
  (R : ℝ := circumradius A B C) :
  perimeter D E F ≥ perimeter A B C ∧ inradius D E F ≥ inradius A B C :=
sorry

end problem_triangle_DEF_inradius_perimeter_l252_252144


namespace lowest_temperature_in_january_2023_l252_252115

theorem lowest_temperature_in_january_2023 
  (T_Beijing T_Shanghai T_Shenzhen T_Jilin : ℝ)
  (h_Beijing : T_Beijing = -5)
  (h_Shanghai : T_Shanghai = 6)
  (h_Shenzhen : T_Shenzhen = 19)
  (h_Jilin : T_Jilin = -22) :
  T_Jilin < T_Beijing ∧ T_Jilin < T_Shanghai ∧ T_Jilin < T_Shenzhen :=
by
  sorry

end lowest_temperature_in_january_2023_l252_252115


namespace ratio_of_larger_to_smaller_l252_252016

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_larger_to_smaller_l252_252016


namespace count_real_pairs_l252_252854

theorem count_real_pairs : (∃ (n : ℕ), n = 2072 ∧ 
  ∀ (x y : ℕ), 1 ≤ x ∧ x < y ∧ y ≤ 150 → 
  ((i^x + i^y : ℂ).re ≠ 0)) :=
begin
  sorry
end

end count_real_pairs_l252_252854


namespace triangle_AEB_equilateral_l252_252642

variables {A B C D E : Type}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E]
noncomputable def angle (x y z : Type) := sorry

-- Given the lengths and angle sum condition
axiom AD_eq_BC : ∀ {A B C D : Type}, AD = BC
axiom angle_sum : ∀ {A B C D : Type}, angle D A B + angle A B C = 120

-- Equilateral triangle DEC constructed outside quadrilateral
axiom equilateral_DEC : ∀ {D E C : Type}, angle D E C = 60

-- Goal: Prove that triangle AEB is equilateral
theorem triangle_AEB_equilateral : 
  ∀ {A B D E : Type} (AD_eq_BC : AD = BC)
  (angle_sum : angle D A B + angle A B C = 120)
  (equilateral_DEC : angle D E C = 60), 
  angle A E B = 60 ∧ angle E B A = 60 ∧ angle B A E = 60 :=
by sorry

end triangle_AEB_equilateral_l252_252642


namespace AB_greater_than_BC_l252_252003

variable (x y : ℝ)
variable (h1 : x > 0)
variable (h2 : y > 0)

theorem AB_greater_than_BC (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let AB : ℝ := real.sqrt (5 * x ^ 2 + 4 * x * y + y ^ 2)
  let BC : ℝ := real.sqrt (5 * x ^ 2 + 2 * x * y + y ^ 2)
  AB > BC := 
by {
  sorry
}

end AB_greater_than_BC_l252_252003


namespace cause_of_polarization_by_electronegativity_l252_252663

-- Definition of the problem conditions as hypotheses
def strong_polarization_of_CH_bond (C_H_bond : Prop) (electronegativity : Prop) : Prop 
  := C_H_bond ∧ electronegativity

-- Given conditions: Carbon atom is in sp hybridization and C-H bond shows strong polarization
axiom carbon_sp_hybridized : Prop
axiom CH_bond_strong_polarization : Prop

-- Question: The cause of strong polarization of the C-H bond at the carbon atom in sp hybridization in alkynes
def cause_of_strong_polarization (sp_hybridization : Prop) : Prop 
  := true  -- This definition will hold as a placeholder, to indicate there is a causal connection

-- Correct answer: high electronegativity of the carbon atom in sp-hybrid state causes strong polarization
theorem cause_of_polarization_by_electronegativity 
  (high_electronegativity : Prop) 
  (sp_hybridized : Prop) 
  (polarized : Prop) 
  (H : strong_polarization_of_CH_bond polarized high_electronegativity) 
  : sp_hybridized ∧ polarized := 
  sorry

end cause_of_polarization_by_electronegativity_l252_252663


namespace find_subtracted_value_l252_252785

theorem find_subtracted_value (N V : ℤ) (hN : N = 12) (h : 4 * N - 3 = 9 * (N - V)) : V = 7 := 
by
  sorry

end find_subtracted_value_l252_252785


namespace positive_diff_solutions_correct_l252_252900

noncomputable def positive_diff_solutions : ℝ :=
  let eqn : ℝ → Prop := λ x, (5 - x^2 / 4) ^ (1 / 3) = -3
  16 * real.sqrt 2

theorem positive_diff_solutions_correct : 
  ∀ x1 x2, eqn x1 ∧ eqn x2 → abs (x1 - x2) = 16 * real.sqrt 2 := sorry

end positive_diff_solutions_correct_l252_252900


namespace find_first_round_score_l252_252229

theorem find_first_round_score (x : ℝ) :
    let second_round := 1.10 * x
    let third_round := (2/3) * second_round
    let total_points := x + second_round + third_round
    total_points = 3000 → x ≈ 1059.24 :=
by
    sorry

end find_first_round_score_l252_252229


namespace find_certain_number_l252_252583

theorem find_certain_number 
  (num : ℝ)
  (h1 : num / 14.5 = 177)
  (h2 : 29.94 / 1.45 = 17.7) : 
  num = 2566.5 := 
by 
  sorry

end find_certain_number_l252_252583


namespace maximum_and_minimum_sum_l252_252199

noncomputable def f (x : ℝ) : ℝ := sorry

noncomputable def g (x : ℝ) : ℝ :=
  f x + x / (1 + x^2) + 1

theorem maximum_and_minimum_sum :
  ∀ (f : ℝ → ℝ) (g : ℝ → ℝ),
    (∀ x y : ℝ, f (x + y) = f x + f y) →
    (∀ x : ℝ, g x = f x + x / (1 + x^2) + 1) →
    ∃ M m : ℝ, (g.max = M) ∧ (g.min = m) ∧ (M + m = 2) :=
begin
  intros f g hf hg,
  have h1 : f 0 = 0, from sorry,
  have h2 : ∀ x : ℝ, f (-x) = -f x, from sorry,
  let h := λ x, f x + x / (1 + x^2),
  have hodd : ∀ x : ℝ, h (-x) = -h x, from sorry,
  have h_min_max : h.min + h.max = 0, from sorry,
  let M := h.max + 1,
  let m := h.min + 1,
  have g_min_max : M + m = 2, from sorry,
  use M, m,
  split, -- show M and m are max and min
  sorry,
  split, -- show M + m = 2
  exact g_min_max,
end

end maximum_and_minimum_sum_l252_252199


namespace M_v_plus_2w_l252_252632

variables {α : Type*} [LinearOrderedField α]

def matrix (n m : ℕ) := array n (array m α)
def vector (n : ℕ) := array n α

def M : matrix 2 2 := sorry
def v : vector 2 := sorry
def w : vector 2 := sorry

axiom Mv : M.dot_product v = [4, 1]
axiom Mw : M.dot_product w = [-1, -3]

theorem M_v_plus_2w : M.dot_product (v + 2 * w) = [2, -5] := by
  sorry

end M_v_plus_2w_l252_252632


namespace circumcircle_radius_l252_252768

def R1 := 2 * Real.sqrt 27 + 6
def R2 := 12 - R1
def d := 4 * Real.sqrt 29
def rA := 8

theorem circumcircle_radius (ABC : Triangle)
  (B C : Point)
  (s1 s2 s3 : Sphere)
  (h_ABC_bc : (s1.center, B) ∈ Plane(ABC) ∧ (s2.center, C) ∈ Plane(ABC))
  (h_opposite_sides : s1.center.z > 0 ∧ s2.center.z < 0)
  (h_radii_sum : s1.r + s2.r = 12)
  (h_centers_dist : dist s1.center s2.center = 4 * Real.sqrt 29)
  (h_sphere_A : s3.center = A ∧ s3.r = 8 ∧ s3.externally_touches s1 ∧ s3.externally_touches s2) :
  radius_circumcircle ABC = 4 * Real.sqrt 5 :=
sorry

end circumcircle_radius_l252_252768


namespace falcons_win_probability_l252_252672

noncomputable def probability_falcons_win_at_least_five_games (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), if i ≥ k then (@Nat.choose n i : ℝ) * p^i * (1 - p)^(n - i) else 0

theorem falcons_win_probability 
  : probability_falcons_win_at_least_five_games 9 (1/2) 5 = 1/2 :=
sorry

end falcons_win_probability_l252_252672


namespace forty_percent_j_squared_l252_252913

-- Definitions from the problem statement.
def f : ℕ := 12
def b : ℕ := f + 6
def j : ℕ := 10 * (f + b)

-- The condition that needs to be proved.
theorem forty_percent_j_squared : 0.40 * (j ^ 2) = 36000 := 
by
  sorry

end forty_percent_j_squared_l252_252913


namespace cistern_fill_time_l252_252081

theorem cistern_fill_time (hF : ∀ (F : ℝ), F = 1 / 3)
                         (hE : ∀ (E : ℝ), E = 1 / 5) : 
  ∃ (t : ℝ), t = 15 / 2 :=
by
  sorry

end cistern_fill_time_l252_252081


namespace no_series_of_diode_usage_l252_252828

-- Define the problem conditions
def usamon := ℕ
def has_electron (A : usamon) : Prop := -- A placeholder for the electron state
sorry

-- Define the diode functionality
def diode (A B : usamon) :=
  if has_electron A ∧ ¬ has_electron B then
    ¬ has_electron A ∧ has_electron B
  else
    has_electron A ∨ ¬ has_electron B

noncomputable def physicist_goal_impossible : Prop :=
  ∀ (u : usamon) (v : usamon), u ≠ v → ¬ ∃ (A B : usamon), diode A B ∧ has_electron A = has_electron B

-- Statement of the problem
theorem no_series_of_diode_usage:
  usamon_count := 2015 →
  ∀ (S : fin usamon_count → usamon), 
  physicist_goal_impossible S :=
sorry

end no_series_of_diode_usage_l252_252828


namespace coefficient_x3_expansion_l252_252260

theorem coefficient_x3_expansion :
  -- General binomial coefficients for expansions
  let binom := λ n k : ℕ, nat.choose n k in
  -- Coefficient of x^3 in (1+x)^5 is binom(5, 3)
  let coeff_5 := binom 5 3 in
  -- Coefficient of x^3 in (1+x)^6 is binom(6, 3)
  let coeff_6 := binom 6 3 in
  -- Total coefficient of x^3 in the difference (1+x)^5 - (1+x)^6
  coeff_5 - coeff_6 = -10 :=
by
  -- Necessary assumptions and calculations can be filled in here
  sorry

end coefficient_x3_expansion_l252_252260


namespace sequence_general_term_l252_252557

noncomputable def a (n : ℕ) : ℝ :=
if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n ≠ 0) : 
  a n = if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1)) :=
by
  sorry

end sequence_general_term_l252_252557


namespace sum_of_x_y_l252_252237

theorem sum_of_x_y (x y : ℝ) (h : (x + y + 2) * (x + y - 1) = 0) : x + y = -2 ∨ x + y = 1 :=
by sorry

end sum_of_x_y_l252_252237


namespace constant_term_in_expansion_l252_252259

theorem constant_term_in_expansion 
  (x : ℝ) (n : ℕ) (h : ( ∑ k in finset.range n, (if 3 * k = 6 then binomial n k else 0)) = 64) : 
  (∑ k in finset.range n, (if k = 2 then binomial n k else 0)) = 15 := 
sorry

end constant_term_in_expansion_l252_252259


namespace min_n_for_root_gt_15_over_8_l252_252308

noncomputable def poly_n_min (a : ℕ → ℤ) (h : ∀ i, abs (a i) = 1) :=
  ∃ (p : polynomial ℤ), (∀ i, abs (p.coeff i) = 1) ∧ (∃ r, polynomial.is_root p r ∧ abs r > 15/8)

theorem min_n_for_root_gt_15_over_8 : poly_n_min (λ i, if i ≤ 3 then 1 else -1) (λ i, by simp) :=
  by 
    intros a h
    existsi polynomial.of_finsupp (λ i, if 0 ≤ i ∧ i ≤ 4 then 1 else 0) -- polynomial x^4 - x^3 - x^2 - x - 1
    split
    sorry
    existsi (15/8)     -- Specific r doesn't need to be explicitly constructed in Lean, showing existence is enough
    sorry

end min_n_for_root_gt_15_over_8_l252_252308


namespace common_fraction_proof_l252_252143

def expr_as_common_fraction : Prop :=
  let numerator := (3 / 6) + (4 / 5)
  let denominator := (5 / 12) + (1 / 4)
  (numerator / denominator) = (39 / 20)

theorem common_fraction_proof : expr_as_common_fraction :=
by
  sorry

end common_fraction_proof_l252_252143


namespace average_percentage_increase_l252_252281

def initial_income_A : ℝ := 60
def new_income_A : ℝ := 80
def initial_income_B : ℝ := 100
def new_income_B : ℝ := 130
def hours_worked_C : ℝ := 20
def initial_rate_C : ℝ := 8
def new_rate_C : ℝ := 10

theorem average_percentage_increase :
  let initial_weekly_income_C := hours_worked_C * initial_rate_C
  let new_weekly_income_C := hours_worked_C * new_rate_C
  let percentage_increase_A := (new_income_A - initial_income_A) / initial_income_A * 100
  let percentage_increase_B := (new_income_B - initial_income_B) / initial_income_B * 100
  let percentage_increase_C := (new_weekly_income_C - initial_weekly_income_C) / initial_weekly_income_C * 100
  let average_percentage_increase := (percentage_increase_A + percentage_increase_B + percentage_increase_C) / 3
  average_percentage_increase = 29.44 :=
by sorry

end average_percentage_increase_l252_252281


namespace tan_of_X_in_right_triangle_l252_252274

theorem tan_of_X_in_right_triangle
  (X Y Z : ℝ)
  (angle_Y : X * X + Y * Y = Z * Z)
  (YZ_val : YZ = 4)
  (XZ_val : XZ = real.sqrt 17) :
  real.tan X = 4 := sorry

end tan_of_X_in_right_triangle_l252_252274


namespace total_payment_l252_252319

def work_hours := 2
def hourly_rate := 75
def part_cost := 150

theorem total_payment : work_hours * hourly_rate + part_cost = 300 := 
by 
  calc 
  2 * 75 + 150 = 150 + 150 : by rw mul_comm 2 75
             ... = 300 : by rw add_comm 150 150

# The term "sorry" is unnecessary due to the use of "by" tactic and commutativity rules simplifying the steps directly.

end total_payment_l252_252319


namespace more_stable_performance_l252_252382

theorem more_stable_performance (S_A2 S_B2 : ℝ) (hA : S_A2 = 0.2) (hB : S_B2 = 0.09) (h : S_A2 > S_B2) : 
  "B" = "B" :=
by
  sorry

end more_stable_performance_l252_252382


namespace part_a_part_b_part_c_l252_252064

-- Define the conditions for Payneful pairs
def isPaynefulPair (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x ∈ Set.univ) ∧
  (∀ x, g x ∈ Set.univ) ∧
  (∀ x y, f (x + y) = f x * g y + g x * f y) ∧
  (∀ x y, g (x + y) = g x * g y - f x * f y) ∧
  (∃ a, f a ≠ 0)

-- Questions and corresponding proofs as Lean theorems
theorem part_a (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : f 0 = 0 ∧ g 0 = 1 := sorry

def h (f g : ℝ → ℝ) (x : ℝ) : ℝ := (f x) ^ 2 + (g x) ^ 2

theorem part_b (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : h f g 5 * h f g (-5) = 1 := sorry

theorem part_c (f g : ℝ → ℝ) (hf : isPaynefulPair f g)
  (h_bound_f : ∀ x, -10 ≤ f x ∧ f x ≤ 10) (h_bound_g : ∀ x, -10 ≤ g x ∧ g x ≤ 10):
  h f g 2021 = 1 := sorry

end part_a_part_b_part_c_l252_252064


namespace average_height_remaining_girls_l252_252352

theorem average_height_remaining_girls (avg_height_30 : ℕ) (total_avg_height : ℕ) (total_girls : ℕ) (num_girls_30 : ℕ) : 
  avg_height_30 = 160 → 
  total_avg_height = 159 → 
  total_girls = 40 → 
  num_girls_30 = 30 → 
  (let remaining_girls := total_girls - num_girls_30 in
   let total_height_30 := num_girls_30 * avg_height_30 in
   let total_height_class := total_girls * total_avg_height in
   let total_height_remaining := total_height_class - total_height_30 in
   let avg_height_remaining := total_height_remaining / remaining_girls in
   avg_height_remaining = 156) :=
by
  sorry

end average_height_remaining_girls_l252_252352


namespace diff_between_percent_and_fraction_l252_252032

theorem diff_between_percent_and_fraction :
  (0.75 * 800) - ((7 / 8) * 1200) = -450 :=
by
  sorry

end diff_between_percent_and_fraction_l252_252032


namespace mac_running_rate_l252_252445

variable (a : ℕ) (d : ℕ) (t_diff : ℕ)

def time_to_run (rate : ℕ) (distance : ℕ) : ℕ := distance / rate

theorem mac_running_rate :
  let a := 3 in
  let d := 24 in
  let t_diff := 120 in
  let t_apple := time_to_run a d * 60 in
  let t_mac := t_apple - t_diff in
  let m := d / (t_mac / 60) in
  m = 4 :=
by {
  sorry
}

end mac_running_rate_l252_252445


namespace remaining_batch_weight_l252_252103

/-- Given a list of weights of six batches, and the weights removed in two shipments,
    where the first shipment's total weight is half of the second shipment's total weight,
    prove that the remaining batch weight is 200 tons. -/
theorem remaining_batch_weight (w : list ℕ) (w1 w2 : list ℕ) 
  (hw : w = [150, 160, 180, 190, 200, 310])
  (h1 : w1.length = 2) (h2 : w2.length = 3)
  (hdisjoint : w1.to_finset ∩ w2.to_finset = ∅)
  (hw1w2 : (list.sum w1) = (list.sum w2) / 2):
  ∃ r, (r ∈ w ∧ r ∉ w1 ∧ r ∉ w2 ∧ r = 200) :=
sorry

end remaining_batch_weight_l252_252103


namespace fg_2_eq_9_l252_252578

def f (x: ℝ) := x^2
def g (x: ℝ) := -4 * x + 5

theorem fg_2_eq_9 : f (g 2) = 9 :=
by
  sorry

end fg_2_eq_9_l252_252578


namespace third_smallest_palindromic_prime_l252_252163

-- Define the properties of being a three-digit palindromic prime
def is_three_digit_palindromic_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ prime n ∧ (n / 100 = n % 10)

-- Assert that 101 and 131 are the first and second smallest three-digit palindromic primes
axiom palindromic_prime_101 : is_three_digit_palindromic_prime 101
axiom palindromic_prime_131 : is_three_digit_palindromic_prime 131

-- Define the proposition that 151 is the third smallest three-digit palindromic prime
theorem third_smallest_palindromic_prime : is_three_digit_palindromic_prime 151 :=
by
  sorry


end third_smallest_palindromic_prime_l252_252163


namespace sequence_log_property_l252_252750

theorem sequence_log_property (a b: ℝ) (h_arith: (log (a ^ 4 * b ^ 8)) + (log (a ^ 11 * b ^ 19)) = 2 * (log (a ^ 7 * b ^ 14))) : 
  (log (a ^ 2 * b ^ 136)) = (log (a ^ 4 * b ^ 8)) + 14 * ((log (a ^ 7 * b ^ 14)) - (log (a ^ 4 * b ^ 8))) :=
by 
  sorry

end sequence_log_property_l252_252750


namespace geometric_sequence_value_l252_252021

theorem geometric_sequence_value (a : ℝ) (h₁ : 280 ≠ 0) (h₂ : 35 ≠ 0) : 
  (∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8 ∧ a > 0) → a = 35 :=
by {
  sorry
}

end geometric_sequence_value_l252_252021


namespace hyperbola_params_sum_l252_252993

theorem hyperbola_params_sum :
  let h := 1
  let k := 2
  let a := 3
  let c := 7
  let b := Real.sqrt 40
  h + k + a + b = 6 + 2 * Real.sqrt 10 :=
by
  let h := 1
  let k := 2
  let a := 3
  let c := 7
  let b := Real.sqrt 40
  show h + k + a + b = 6 + 2 * Real.sqrt 10
  sorry

end hyperbola_params_sum_l252_252993


namespace max_integers_greater_than_fifteen_l252_252368

theorem max_integers_greater_than_fifteen (s : Fin 8 → ℤ) (h_sum : (∑ i, s i) = 20) :
  (∃ k, (∀ i < k, s i > 15) ∧ (k + (8 - k) = 8) ∧ (16 * k ≤ ∑ i in Finset.range k, s i) ∧ (7 = k)) :=
sorry

end max_integers_greater_than_fifteen_l252_252368


namespace average_side_length_of_squares_l252_252718

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252718


namespace largest_digit_divisible_by_6_l252_252776

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 :
  ∃ N : ℕ, N ≤ 9 ∧ N % 2 = 0 ∧ is_divisible_by_3 (14 + N) ∧
  ∀ M : ℕ, M ≤ 9 ∧ M % 2 = 0 ∧ is_divisible_by_3 (14 + M) → M ≤ N :=
begin
  let N := 4,
  use N,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intros M hM,
    cases hM with hM1 hM2,
    cases hM2 with hM3 hM4,
    by_cases hm : M = 4,
    { rw hm },
    { interval_cases M, -- This tactic considers all values for M in the range [0, 9]
      iterate,
      { contradiction <|> linarith }},
    sorry, -- This would need specific checks, but conceptually it is here
  }
end

end largest_digit_divisible_by_6_l252_252776


namespace tully_kate_age_ratio_l252_252268

theorem tully_kate_age_ratio :
  let tully_age := 60 + 1 in
  let tully_age_in_three_years := tully_age + 3 in
  let kate_age := 29 in
  let kate_age_in_three_years := kate_age + 3 in
  (tully_age_in_three_years : ℚ) / (kate_age_in_three_years : ℚ) = 2 :=
by
  sorry

end tully_kate_age_ratio_l252_252268


namespace average_side_length_of_squares_l252_252720

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252720


namespace reduce_to_at_most_eight_l252_252772

def contains_all_letters (w : List Char) : Prop :=
  'A' ∈ w ∧ 'B' ∈ w ∧ 'C' ∈ w

def equivalent (u v : List Char) : Prop :=
  sorry -- Define equivalence using operations (a) and (b)

theorem reduce_to_at_most_eight (w : List Char) :
  length w ≥ 9 → ∃ (w' : List Char), length w' ≤ 8 ∧ equivalent w w' :=
by
  sorry

end reduce_to_at_most_eight_l252_252772


namespace scientific_notation_is_correct_l252_252651

theorem scientific_notation_is_correct (h : 0.000000001 = (1 : ℝ) * 10 ^ (-9 : ℝ)) : 0.000000001 = 1 * 10 ^ (-9) :=
by 
  exact h
  sorry -- Proof is not required here, we just setup the theorem to ensure Lean can build it

end scientific_notation_is_correct_l252_252651


namespace calc_fffff_2_plus_i_l252_252505
-- Import the entirety of the necessary library

-- Define the function f
def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem to prove the final value
theorem calc_fffff_2_plus_i : f (f (f (f (2 + I)))) = 1042434 - 131072 * I :=
  sorry

end calc_fffff_2_plus_i_l252_252505


namespace smallest_k_for_perfect_square_l252_252090

def y : ℕ := 2^3 * 3^4 * 5^6 * 7^8 * 8^9 * 9^10

theorem smallest_k_for_perfect_square : ∃ k : ℕ, k > 0 ∧ (y * k) = (m^2) for some m and k = 2 :=
by sorry

end smallest_k_for_perfect_square_l252_252090


namespace polynomial_divisibility_l252_252466

open Polynomial

noncomputable def P (x : ℂ) (n m : ℕ) : ℂ := ∑ i in range (m + 1), x^(i * n)
noncomputable def Q (x : ℂ) (m : ℕ) : ℂ := ∑ i in range (m + 1), x^i

theorem polynomial_divisibility (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n):
  ((∀ x : ℂ, x^(m + 1) = 1 → x ≠ 1 → P x n m = 0) ↔ ¬ ∃ k : ℕ, n = k * (m + 1)) := by
  sorry

end polynomial_divisibility_l252_252466


namespace suzy_final_books_l252_252050

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l252_252050


namespace correct_propositions_l252_252960

section propositions

variable (L1 L2 L3 : Type) -- Representing lines
variable (P : Type) -- Representing planes

-- Condition 1: If two lines are both parallel to a third line, then these two lines are parallel
axiom parallel_to_third_line (l1 l2 l3 : L1) (h1 : l1 ∥ l3) (h2 : l2 ∥ l3) : l1 ∥ l2

-- Condition 2: If two lines are both perpendicular to a third line, then these two lines are parallel
axiom perpendicular_to_third_line (l1 l2 l3 : L2) (h1 : l1 ⟂ l3) (h2 : l2 ⟂ l3) : l1 ∥ l2

-- Condition 3: If two lines are both parallel to a plane, then these two lines are parallel
axiom parallel_to_plane (l1 l2 : P) (p : P) (h1 : l1 ∥ p) (h2 : l2 ∥ p) : l1 ∥ l2

-- Condition 4: If two lines are both perpendicular to a plane, then these two lines are parallel
axiom perpendicular_to_plane (l1 l2 : L2) (p : P) (h1 : l1 ⟂ p) (h2 : l2 ⟂ p) : l1 ∥ l2

-- Given the above conditions, we need to prove that only propositions ① and ④ are correct
theorem correct_propositions :
  (parallel_to_third_line L1 L1 L1) ∧ ¬(perpendicular_to_third_line L1 L1 L1) ∧ ¬(parallel_to_plane L1 L1) ∧ (perpendicular_to_plane L1 L1) :=
by sorry

end propositions

end correct_propositions_l252_252960


namespace compound_interest_final_amount_l252_252841

theorem compound_interest_final_amount :
  let P := 15000
  let r := 0.04
  let t := 10 in
  P * (1 + r) ^ t = 22204 :=
by
  sorry

end compound_interest_final_amount_l252_252841


namespace tangent_segments_area_l252_252180

theorem tangent_segments_area (r : ℝ) (l : ℝ) (h_r : r = 4) (h_l : l = 4) :
  let inner_area := π * r^2,
      outer_area := π * (sqrt (r^2 + (l/2)^2))^2,
      annulus_area := outer_area - inner_area
  in annulus_area = 4 * π :=
by
  sorry

end tangent_segments_area_l252_252180


namespace sum_of_r_and_s_l252_252004

noncomputable def r (x : ℝ) := (5/6) * (x + 1)
noncomputable def s (x : ℝ) := (3 / 100) * (x + 1) * (x - 2)^2

theorem sum_of_r_and_s :
  r(5) = 5 ∧ 
  s(-3) = 3 ∧ 
  (r(x) + s(x) = (x + 1) * (3 * x^2 - 12 * x + 512) / 100) :=
by
  sorry

end sum_of_r_and_s_l252_252004


namespace total_payment_l252_252321

def work_hours := 2
def hourly_rate := 75
def part_cost := 150

theorem total_payment : work_hours * hourly_rate + part_cost = 300 := 
by 
  calc 
  2 * 75 + 150 = 150 + 150 : by rw mul_comm 2 75
             ... = 300 : by rw add_comm 150 150

# The term "sorry" is unnecessary due to the use of "by" tactic and commutativity rules simplifying the steps directly.

end total_payment_l252_252321


namespace random_event_l252_252108

theorem random_event (a b : ℝ) (h1 : a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0):
  ¬ (∀ a b, a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0 → a + b < 0) :=
by
  sorry

end random_event_l252_252108


namespace geometric_sequence_term_l252_252547

-- Definitions for geometric series sum
def S_n (n : ℕ) (x : ℝ) : ℝ :=
  x * 2^(n-1) - 1/6

-- Main theorem to be proven
theorem geometric_sequence_term (n : ℕ) (x : ℝ) (a_n : ℝ) :
  (∀ n : ℕ, S_n n x = x * 2^(n-1) - 1 / 6) →
  a_n = 1/3 * 2^(n - 2) :=
sorry

end geometric_sequence_term_l252_252547


namespace integer_values_of_a_l252_252465

-- Define the polynomial P(x)
def P (a x : ℤ) : ℤ := x^3 + a * x^2 + 3 * x + 7

-- Define the main theorem
theorem integer_values_of_a (a x : ℤ) (hx : P a x = 0) (hx_is_int : x = 1 ∨ x = -1 ∨ x = 7 ∨ x = -7) :
  a = -11 ∨ a = -3 :=
by
  sorry

end integer_values_of_a_l252_252465


namespace count_valid_a1_l252_252634

theorem count_valid_a1 : 
  let seq (n : ℕ) (a : ℕ -> ℕ) := if n = 0 then a1 else 
    if a (n - 1) % 2 = 0 then a (n - 1) / 2 
    else 3 * a (n - 1) + 1
  in 
  let valid_a1 (a1 : ℕ) := 
    a1 <= 3000 ∧ 
    (∀ n < 4, (n = 0 → a1 < a (1) ∧ a1 < a (2) ∧ a1 < a (3)))
  in 
  (∃ n ≤ 3000, ∀ x, valid_a1(n)) = 750 := sorry

end count_valid_a1_l252_252634


namespace range_of_p_l252_252310

-- Conditions definitions
def A : Set ℝ :=
  { x | x^2 - x - 2 > 0 }

def B : Set ℝ :=
  { x | 3 - |x| ≥ 0 }

def C (p : ℝ) : Set ℝ :=
  { x | x^2 + 4x + 4 - p^2 < 0 ∧ p > 0 }

-- Problem statement
theorem range_of_p {p : ℝ} (hC : C p ⊆ (A ∩ B)) : 0 < p ∧ p ≤ 1 := 
  sorry

end range_of_p_l252_252310


namespace smallest_initial_number_wins_bernardo_sum_of_digits_smallest_initial_number_l252_252116

theorem smallest_initial_number_wins_bernardo (N : ℕ) (h₁ : 0 ≤ N ∧ N ≤ 1999)
  (h₂ : 16 * N + 1400 < 2000 ∧ 16 * N + 1500 ≥ 2000) : N = 32 :=
by
  sorry

theorem sum_of_digits_smallest_initial_number : ∑ d in (Nat.digits 10 32), d = 5 :=
by
  sorry

end smallest_initial_number_wins_bernardo_sum_of_digits_smallest_initial_number_l252_252116


namespace percent_increase_from_may_to_june_l252_252751

noncomputable def profit_increase_from_march_to_april (P : ℝ) : ℝ := 1.30 * P
noncomputable def profit_decrease_from_april_to_may (P : ℝ) : ℝ := 1.04 * P
noncomputable def profit_increase_from_march_to_june (P : ℝ) : ℝ := 1.56 * P

theorem percent_increase_from_may_to_june (P : ℝ) :
  (1.04 * P * (1 + 0.50)) = 1.56 * P :=
by
  sorry

end percent_increase_from_may_to_june_l252_252751


namespace book_arrangements_l252_252573

theorem book_arrangements :
  let math_books := 4
  let english_books := 4
  let groups := 2
  (groups.factorial) * (math_books.factorial) * (english_books.factorial) = 1152 :=
by
  sorry

end book_arrangements_l252_252573


namespace find_coordinates_of_A_l252_252534

theorem find_coordinates_of_A (x : ℝ) :
  let A := (x, 1, 2)
  let B := (2, 3, 4)
  (Real.sqrt ((x - 2)^2 + (1 - 3)^2 + (2 - 4)^2) = 2 * Real.sqrt 6) →
  (x = 6 ∨ x = -2) := 
by
  intros
  sorry

end find_coordinates_of_A_l252_252534


namespace sum_of_factors_of_24_l252_252042

theorem sum_of_factors_of_24 : 
  ∑ i in (finset.filter (λ x, 24 % x = 0) (finset.range (24 + 1))), i = 60 :=
sorry

end sum_of_factors_of_24_l252_252042


namespace average_side_length_of_squares_l252_252690

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252690


namespace sodium_chloride_formed_l252_252893

section 

-- Definitions based on the conditions
def hydrochloric_acid_moles : ℕ := 2
def sodium_bicarbonate_moles : ℕ := 2

-- Balanced chemical equation represented as a function (1:1 reaction ratio)
def reaction (hcl_moles naHCO3_moles : ℕ) : ℕ := min hcl_moles naHCO3_moles

-- Theorem stating the reaction outcome
theorem sodium_chloride_formed : reaction hydrochloric_acid_moles sodium_bicarbonate_moles = 2 :=
by
  -- Proof is omitted
  sorry

end

end sodium_chloride_formed_l252_252893


namespace average_side_lengths_l252_252683

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252683


namespace circumcenter_ratio_l252_252988

theorem circumcenter_ratio (A B C O : Type) 
  (triangle_ABC : triangle A B C) 
  (circumcenter_tri : circumcenter A B C = O)
  (angle_A : ∠ A = 45)
  (angle_B : ∠ B = 60) : 
  ratio_distances_circumcenter_sides O A B C = (2, sqrt 2, sqrt 3 - 1) := 
sorry

end circumcenter_ratio_l252_252988


namespace find_line_equation_l252_252589

variables {x y : ℝ}

def line1 (x y : ℝ) : Prop := x + y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

theorem find_line_equation (h : ∃ (x y : ℝ), line1 x y ∧ line2 x y) 
    (hl : ∃ (a : ℝ), ∀ x y : ℝ, (x / (2 * a)) + (y / a) = 1 → line1 x y ∧ line2 x y ∧ (l : line1 x y)) : 
    ∃ (x y : ℝ), x + 2 * y - 3 = 0 :=
by 
  sorry

end find_line_equation_l252_252589


namespace clock_lights_at_93720PM_l252_252334

/-- Given the time 9:37:20 PM, show that the number of small color lights within the acute angle
    formed between the minute hand and the hour hand is 11. -/
theorem clock_lights_at_93720PM : 
  let h_angle := 9 * 30 + (37 + 20/60) * 0.5,
      m_angle := 37 * 6 + (20/60) * 6,
      angle_between := abs (m_angle - h_angle)
  in 60 * (angle_between / 360) = 11 := 
by
  sorry

end clock_lights_at_93720PM_l252_252334


namespace parametric_to_general_eq_l252_252366

theorem parametric_to_general_eq (x y θ : ℝ) 
  (h1 : x = 2 + Real.sin θ ^ 2) 
  (h2 : y = -1 + Real.cos (2 * θ)) : 
  2 * x + y - 4 = 0 ∧ 2 ≤ x ∧ x ≤ 3 := 
sorry

end parametric_to_general_eq_l252_252366


namespace integer_sums_of_powers_l252_252286

theorem integer_sums_of_powers
  (x y : ℝ)
  (h1 : ↑(x + y) ∈ ℤ)
  (h2 : ↑(x^2 + y^2) ∈ ℤ)
  (h3 : ↑(x^3 + y^3) ∈ ℤ)
  (h4 : ↑(x^4 + y^4) ∈ ℤ) :
  ∀ n : ℕ, 0 < n → ↑(x^n + y^n) ∈ ℤ := 
  sorry

end integer_sums_of_powers_l252_252286


namespace isosceles_trapezoid_diagonal_length_l252_252361

theorem isosceles_trapezoid_diagonal_length :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (24 : ℝ, 0 : ℝ)
  let C := (18 : ℝ, 12 : ℝ)
  let D := (6 : ℝ, 12 : ℝ)
  (dist A C) = Real.sqrt 457 :=
by sorry

end isosceles_trapezoid_diagonal_length_l252_252361


namespace ellipse_area_is_12pi_l252_252119

noncomputable def ellipse_area : ℝ :=
  (1 / 2) * ∫ (t : ℝ) in 0..(2 * Real.pi), (4 * Real.cos t * (3 * Real.cos t) - 3 * Real.sin t * (-4 * Real.sin t))

theorem ellipse_area_is_12pi :
  ellipse_area = 12 * Real.pi := by
sorry

end ellipse_area_is_12pi_l252_252119


namespace range_of_x_l252_252244

theorem range_of_x (a : ℝ) (x : ℝ) (h₀ : 1 ≤ a ∧ a ≤ 3) :
  ax^2 + (a - 2)x - 2 > 0 -> x < -1 ∨ x > 2 := 
by sorry

end range_of_x_l252_252244


namespace intersection_and_sum_l252_252200

noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

theorem intersection_and_sum :
  (h(3) = 3 ∧ h(5) = 10 ∧ h(7) = 21 ∧ h(9) = 21 ∧ j(3) = 3 ∧ j(5) = 10 ∧ j(7) = 21 ∧ j(9) = 21) →
  (∃ x y, y = h(3 * x) ∧ y = 3 * j(x) ∧ x = 3 ∧ y = 21) ∧ 3 + 21 = 24 :=
by
  sorry

end intersection_and_sum_l252_252200


namespace tan_is_odd_function_l252_252056

theorem tan_is_odd_function :
  (∀ x, tan (-x) = -tan x) :=
by
  sorry

end tan_is_odd_function_l252_252056


namespace fg_eval_l252_252234

def f (x : ℤ) : ℤ := x^3
def g (x : ℤ) : ℤ := 4 * x + 5

theorem fg_eval : f (g (-2)) = -27 := by
  sorry

end fg_eval_l252_252234


namespace polar_to_rectangular_eq_ap_aq_product_l252_252668

-- First part: rectangular coordinate equation of circle C
theorem polar_to_rectangular_eq (θ : ℝ) : (let ρ := 2 * Real.cos θ in (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1) :=
sorry

-- Second part: value of |AP| * |AQ|
theorem ap_aq_product 
(x y t : ℝ)
(h_param : x = 1/2 + (Real.sqrt 3 / 2) * t ∧ y = 1/2 + 1/2 * t)
(h_A_polar : (1/2, 1/2) = (Real.sqrt 2 / 2 * Real.cos (π / 4), Real.sqrt 2 / 2 * Real.sin (π / 4)))
(h_pol_eq : (x - 1)^2 + y^2 = 1)
(h_line : ∃ P Q, P ≠ Q ∧ P ∈ ℝ ∧ Q ∈ ℝ ∧ (|t| = (1/2 + (Real.sqrt 3 / 2) * P) ∧ (|t| = (1/2 + 1/2 * Q))))
: |PQ| = 1/2 :=
sorry

end polar_to_rectangular_eq_ap_aq_product_l252_252668


namespace bakery_storage_l252_252398

theorem bakery_storage (S F B : ℕ) 
  (h1 : S * 4 = F * 5) 
  (h2 : F = 10 * B) 
  (h3 : F * 1 = (B + 60) * 8) : S = 3000 :=
sorry

end bakery_storage_l252_252398


namespace average_side_length_of_squares_l252_252717

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252717


namespace trig_problem_solution_l252_252944

noncomputable def trig_problem (α : ℝ) : Prop :=
  0 ≤ α ∧ α ≤ π ∧ sin α + cos α = 1 / 5 

theorem trig_problem_solution (α : ℝ) (h : trig_problem α) : 
  √2 * sin (2 * α - π / 4) = -17 / 25 :=
sorry

end trig_problem_solution_l252_252944


namespace books_at_end_l252_252045

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l252_252045


namespace fraction_black_part_l252_252425

theorem fraction_black_part (L : ℝ) (blue_part : ℝ) (white_part_fraction : ℝ) 
  (h1 : L = 8) (h2 : blue_part = 3.5) (h3 : white_part_fraction = 0.5) : 
  (8 - (3.5 + 0.5 * (8 - 3.5))) / 8 = 9 / 32 :=
by
  sorry

end fraction_black_part_l252_252425


namespace shape_is_cylinder_l252_252908

def is_cylinder (c : ℝ) (r θ z : ℝ) : Prop :=
  c > 0 ∧ r = c ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ True

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) (h : c > 0) :
  is_cylinder c r θ z :=
by
  -- Proof is omitted
  sorry

end shape_is_cylinder_l252_252908


namespace area_of_triangle_BDM_l252_252136

theorem area_of_triangle_BDM :
  ∀ (A B C D M : Type)
  (side_length : ℝ)
  (h1 : is_equilateral_triangle A B C)
  (h2 : dist A B = 4)
  (h3 : midpoint M A C)
  (h4 : on_line C B D)
  (h5 : ratio BC CD = 2 / 1),
  area_of_triangle B D M = 2 * sqrt 3 :=
by
  sorry

end area_of_triangle_BDM_l252_252136


namespace trig_matrix_det_zero_l252_252852

noncomputable def trig_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [Real.sin (30 * Real.pi / 180), Real.sin (60 * Real.pi / 180), Real.sin (90 * Real.pi / 180)],
    [Real.sin (150 * Real.pi / 180), Real.sin (180 * Real.pi / 180), Real.sin (210 * Real.pi / 180)],
    [Real.sin (270 * Real.pi / 180), Real.sin (300 * Real.pi / 180), Real.sin (330 * Real.pi / 180)]
  ]

theorem trig_matrix_det_zero : trig_matrix.det = 0 := by
  sorry

end trig_matrix_det_zero_l252_252852


namespace sequence_bounds_l252_252928

theorem sequence_bounds (c : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = ↑n + c / ↑n) 
  (h2 : ∀ n : ℕ+, a n ≥ a 3) : 6 ≤ c ∧ c ≤ 12 :=
by 
  -- We will prove that 6 ≤ c and c ≤ 12 given the conditions stated
  sorry

end sequence_bounds_l252_252928


namespace eggs_in_basket_l252_252230

theorem eggs_in_basket (x : ℕ) (h₁ : 600 / x + 1 = 600 / (x - 20)) : x = 120 :=
sorry

end eggs_in_basket_l252_252230


namespace trigonometric_identity_l252_252335

variables (α β : ℝ)

theorem trigonometric_identity (h₁ : sin α ≠ 0) : 
  (sin (2 * α + β) / sin α - 2 * cos (α + β) = sin β / sin α) :=
by sorry

end trigonometric_identity_l252_252335


namespace tens_digit_19_2021_l252_252868

theorem tens_digit_19_2021 : (19^2021 % 100) / 10 % 10 = 1 :=
by sorry

end tens_digit_19_2021_l252_252868


namespace cube_parallel_faces_l252_252232

-- Define a cube and the property of faces being parallel
def is_cube (s : Type*) (faces : set (set s)) := sorry
def pairs_of_parallel_faces (s : Type*) (faces : set (set s)) := sorry
def number_of_pairs (s : Type*) (faces : set (set s)) := sorry

theorem cube_parallel_faces : ∀ (s : Type*) (faces : set (set s)),
  is_cube s faces → 
  number_of_pairs s (pairs_of_parallel_faces s faces) = 3 := sorry

end cube_parallel_faces_l252_252232


namespace books_left_on_Fri_l252_252052

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l252_252052


namespace ratio_of_numbers_l252_252019

theorem ratio_of_numbers (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h₃ : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l252_252019


namespace preference_related_to_gender_P3_equals_one_third_sequence_geometric_P19_greater_than_P20_l252_252907

-- Lean code representation of the statistical proof and sequence problem.

-- Condition: The contingency table and independence test on preference for football related to gender.
def contingency_table : ℕ := 200  -- total sample size
def observed_frequencies : Mat 2 2 ℕ := ![![60, 40], ![20, 80]]
def row_totals : Vec 2 ℕ := ![100, 100]
def column_totals : Vec 2 ℕ := ![80, 120]
def alpha : ℝ := 0.001

-- Question 1: Show that preference for football is related to gender.
theorem preference_related_to_gender :
  let K_squared := (contingency_table * (observed_frequencies 0 0 * observed_frequencies 1 1 - observed_frequencies 0 1 * observed_frequencies 1 0) ^ 2) /
                   (column_totals 0 * column_totals 1 * row_totals 0 * row_totals 1)
  in K_squared > 10.828 :=
sorry

-- Conditions for the sequence problem
def P1 : ℝ := 1

-- Question 2(i): Proving P_3 = 1/3
theorem P3_equals_one_third : (P3 : ℝ) := 1 / 3

-- Question 2(ii): Prove geometric sequence and P_19 > P_20
theorem sequence_geometric (n : ℕ) (hn : n ≥ 2) : 
  ∀ n, P_n = (3 / 4) * (-1/3)^(n-1) + (1 / 4) :=
sorry

theorem P19_greater_than_P20 : 
  let P19 := (3 / 4) * (-1/3)^18 + (1 / 4)
      P20 := (3 / 4) * (-1/3)^19 + (1 / 4)
  in P19 > P20 :=
sorry

end preference_related_to_gender_P3_equals_one_third_sequence_geometric_P19_greater_than_P20_l252_252907


namespace intersection_closure_M_and_N_l252_252559

noncomputable def set_M : Set ℝ :=
  { x | 2 / x < 1 }

noncomputable def closure_M : Set ℝ :=
  Set.Icc 0 2

noncomputable def set_N : Set ℝ :=
  { y | ∃ x, y = Real.sqrt (x - 1) }

theorem intersection_closure_M_and_N :
  (closure_M ∩ set_N) = Set.Icc 0 2 :=
by
  sorry

end intersection_closure_M_and_N_l252_252559


namespace find_m_l252_252148

def f (x : ℝ) : ℝ :=
  2019 * (3.5 * x - 2.5)^(1/3) + 2018 * Real.log2 (3 * x - 1)

theorem find_m :
  ∃ m : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 2019 * (3.5 * x - 2.5)^(1/3) + 2018 * Real.log2(3 * x - 1) + m = 2020 →
    m ∈ Set.Icc (-8072) (-2017) :=
by
  sorry

end find_m_l252_252148


namespace remaining_garden_area_l252_252080

-- Definitions as per conditions
structure Garden :=
  (diameter : ℝ)
  (path_width : ℝ)

-- Garden data
def myGarden : Garden :=
  { diameter := 10, path_width := 4 }

-- The proof statement
theorem remaining_garden_area (G : Garden) : G.diameter = 10 ∧ G.path_width = 4 →
  let r := G.diameter / 2 in
  let total_area := π * r^2 in
  let path_area := G.diameter * G.path_width in
  total_area - path_area = 25 * π - 40 :=
by
  -- Here we can assume the proof steps will be filled out
  sorry

end remaining_garden_area_l252_252080


namespace max_sum_of_absolute_differences_l252_252288

theorem max_sum_of_absolute_differences (n : ℕ) (h : 2 ≤ n) (a : Fin n → Fin n) 
  (h_perm : ∀ (i : Fin n), ∃ j: Fin n, a j = i + 1) : 
  ∑ i in Finset.range (n - 1), |a (i + 1) - a i| = (n ^ 2 / 2) - 1 :=
by
  sorry

end max_sum_of_absolute_differences_l252_252288


namespace find_a_and_solutions_l252_252546

theorem find_a_and_solutions :
  ∃ (a : ℝ) (x : ℝ),
  (a + 6 = 2a - 9) ∧
  (∃ x, (a = 15) ∧ (a * x^2 - 64 = 0) ∧ (x = 0 ∨ x = 8/sqrt 15 ∨ x = -8/sqrt 15)) := 
sorry

end find_a_and_solutions_l252_252546


namespace find_a_l252_252581

theorem find_a (m : ℤ) (a : ℤ) (h₁ : (-2)^(2 * m) = a^(21 - m)) (h₂ : m = 7) : a = -2 := by
  sorry

end find_a_l252_252581


namespace special_case_m_l252_252588

theorem special_case_m (m : ℝ) :
  (∀ x : ℝ, mx^2 - 4 * x + 3 = 0 → y = mx^2 - 4 * x + 3 → (x = 0 ∧ m = 0) ∨ (x ≠ 0 ∧ m = 4/3)) :=
sorry

end special_case_m_l252_252588


namespace smartphone_price_l252_252113

theorem smartphone_price (S : ℝ) (pc_price : ℝ) (tablet_price : ℝ) 
  (total_cost : ℝ) (h1 : pc_price = S + 500) 
  (h2 : tablet_price = 2 * S + 500) 
  (h3 : S + pc_price + tablet_price = 2200) : 
  S = 300 :=
by
  sorry

end smartphone_price_l252_252113


namespace intersection_A_B_union_A_complement_B_subset_C_B_range_l252_252189

def set_A : Set ℝ := { x | 1 ≤ x ∧ x < 6 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 9 }
def set_C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 2 < x ∧ x < 6 } :=
sorry

theorem union_A_complement_B :
  set_A ∪ (compl set_B) = { x | x < 6 } ∪ { x | x ≥ 9 } :=
sorry

theorem subset_C_B_range (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end intersection_A_B_union_A_complement_B_subset_C_B_range_l252_252189


namespace a1_plus_a5_eq_55_l252_252631

theorem a1_plus_a5_eq_55
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : a 3 = 6)
  (h2 : ∀ n, S (n + 1) = 3 * S n)
  (h3 : ∀ n, S n = ∑ i in range n, a (i + 1)) :
  a 1 + a 5 = 55 :=
by
  sorry

end a1_plus_a5_eq_55_l252_252631


namespace min_h10_of_expansive_l252_252842

-- Definition of an expansive function
def expansive (f : ℕ → ℤ) : Prop :=
∀ (x y : ℕ), x > 0 → y > 0 → f(x) + f(y) > x^2 + y^2

-- Definition of the sum of h function over a specific range
def sum_h (h : ℕ → ℤ) (n : ℕ) : ℤ :=
∑ i in finset.range n, h (i + 1)

-- Lean statement for the problem
theorem min_h10_of_expansive (h : ℕ → ℤ) (H_exp : expansive h)
    (H_min : sum_h h 15 = ∑ i in finset.range 15, (i+1)^2 + ∑ i in finset.range 15, (i+1-1)^2 - 56):
  h 10 = 125 :=
sorry

end min_h10_of_expansive_l252_252842


namespace angle_C_is_180_l252_252601

noncomputable def angle_at_vertex_C (s : ℝ) : ℝ :=
  let A := (s, 0)
  let B := (2 * s, 0)
  let C := (s, sqrt(3) * s)
  -- Calculation should follow the same logic using the conditions
  let valid_eq := (s + s + 2 * s) * (s + s - 2 * s) = 12 * s ^ 2
  if valid_eq then
    180
  else
    0

open_locale real

theorem angle_C_is_180 {s : ℝ} (h : (s + s + 2 * s) * (s + s - 2 * s) = 12 * s ^ 2)
  : angle_at_vertex_C s = 180 :=
by
  sorry

end angle_C_is_180_l252_252601


namespace books_left_on_Fri_l252_252051

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l252_252051


namespace store_owner_loss_percentage_l252_252098

theorem store_owner_loss_percentage :
  ∀ (initial_value : ℝ) (profit_margin : ℝ) (loss1 : ℝ) (loss2 : ℝ) (loss3 : ℝ) (tax_rate : ℝ),
    initial_value = 100 → profit_margin = 0.10 → loss1 = 0.20 → loss2 = 0.30 → loss3 = 0.25 → tax_rate = 0.12 →
      ((initial_value - initial_value * (1 - loss1) * (1 - loss2) * (1 - loss3)) / initial_value * 100) = 58 :=
by
  intros initial_value profit_margin loss1 loss2 loss3 tax_rate h_initial_value h_profit_margin h_loss1 h_loss2 h_loss3 h_tax_rate
  -- Variable assignments as per given conditions
  have h1 : initial_value = 100 := h_initial_value
  have h2 : profit_margin = 0.10 := h_profit_margin
  have h3 : loss1 = 0.20 := h_loss1
  have h4 : loss2 = 0.30 := h_loss2
  have h5 : loss3 = 0.25 := h_loss3
  have h6 : tax_rate = 0.12 := h_tax_rate
  
  sorry

end store_owner_loss_percentage_l252_252098


namespace total_payment_l252_252320

def work_hours := 2
def hourly_rate := 75
def part_cost := 150

theorem total_payment : work_hours * hourly_rate + part_cost = 300 := 
by 
  calc 
  2 * 75 + 150 = 150 + 150 : by rw mul_comm 2 75
             ... = 300 : by rw add_comm 150 150

# The term "sorry" is unnecessary due to the use of "by" tactic and commutativity rules simplifying the steps directly.

end total_payment_l252_252320


namespace larry_wins_game_l252_252284

-- Defining probabilities for Larry and Julius
def larry_throw_prob : ℚ := 2 / 3
def julius_throw_prob : ℚ := 1 / 3

-- Calculating individual probabilities based on the description
def p1 : ℚ := larry_throw_prob
def p3 : ℚ := (julius_throw_prob ^ 2) * larry_throw_prob
def p5 : ℚ := (julius_throw_prob ^ 4) * larry_throw_prob

-- Aggregating the probability that Larry wins the game
def larry_wins_prob : ℚ := p1 + p3 + p5

-- The proof statement
theorem larry_wins_game : larry_wins_prob = 170 / 243 := by
  sorry

end larry_wins_game_l252_252284


namespace melanie_attended_games_l252_252325

theorem melanie_attended_games 
(missed_games total_games attended_games : ℕ) 
(h1 : total_games = 64) 
(h2 : missed_games = 32)
(h3 : attended_games = total_games - missed_games) 
: attended_games = 32 :=
by sorry

end melanie_attended_games_l252_252325


namespace largest_number_formed_by_1_4_5_l252_252153

def digits : List ℕ := [1, 4, 5]

theorem largest_number_formed_by_1_4_5 :
  ∀ (l : List ℕ), l ~ digits → l.perm digits → l.foldl (λ n d, n * 10 + d) 0 ≤ 541 := 
by
  sorry

end largest_number_formed_by_1_4_5_l252_252153


namespace vector_condition_l252_252972

def vec_a : ℝ × ℝ := (5, 2)
def vec_b : ℝ × ℝ := (-4, -3)
def vec_c : ℝ × ℝ := (-23, -12)

theorem vector_condition : 3 • (vec_a.1, vec_a.2) - 2 • (vec_b.1, vec_b.2) + vec_c = (0, 0) :=
by
  sorry

end vector_condition_l252_252972


namespace paired_products_not_equal_1000_paired_products_equal_10000_l252_252067

open Nat

theorem paired_products_not_equal_1000 :
  ∀ (a : Fin 1000 → ℤ), (∃ p n : Nat, p + n = 1000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) → False :=
by 
  sorry

theorem paired_products_equal_10000 :
  ∀ (a : Fin 10000 → ℤ), (∃ p n : Nat, p + n = 10000 ∧
    p * (p - 1) / 2 + n * (n - 1) / 2 = 2 * p * n) ↔ p = 5050 ∨ p = 4950 :=
by 
  sorry

end paired_products_not_equal_1000_paired_products_equal_10000_l252_252067


namespace find_number_l252_252396

theorem find_number (N : ℕ) :
  let sum := 555 + 445
  let difference := 555 - 445
  let divisor := sum
  let quotient := 2 * difference
  let remainder := 70
  N = divisor * quotient + remainder -> N = 220070 := 
by
  intro h
  sorry

end find_number_l252_252396


namespace convex_polygon_with_arith_prog_angles_l252_252747

theorem convex_polygon_with_arith_prog_angles 
  (n : ℕ) 
  (angles : Fin n → ℝ)
  (is_convex : ∀ i, angles i < 180)
  (arithmetic_progression : ∃ a d, d = 3 ∧ ∀ i, angles i = a + i * d)
  (largest_angle : ∃ i, angles i = 150)
  : n = 24 :=
sorry

end convex_polygon_with_arith_prog_angles_l252_252747


namespace tangent_sum_l252_252192

theorem tangent_sum (θ : ℝ) (h : sin θ + cos θ = sqrt 2) : tan θ + 1 / tan θ = 2 :=
sorry

end tangent_sum_l252_252192


namespace bf_parallel_ac_l252_252254

theorem bf_parallel_ac 
  (A B C D K L E F : Point)
  (h_parallelogram : parallelogram A B C D)
  (h_angle_B_obtuse : obtuse_angle (angle B))
  (h_AD_gt_AB : dist A D > dist A B)
  (h_KL_on_AC : K ∈ line_segment A C ∧ L ∈ line_segment A C)
  (h_angles_equal : angle A B K = angle A D L)
  (h_distinct_points : A ≠ K ∧ K ≠ L ∧ L ≠ C ∧ A ≠ C)
  (h_circumcircle : circline A B C Γ)
  (h_BK_intersection_E : second_intersection (line B K) Γ E)
  (h_EL_intersection_F : second_intersection (line E L) Γ F) :
  parallel (line B F) (line A C) := sorry

end bf_parallel_ac_l252_252254


namespace count_points_in_intersection_is_7_l252_252561

def isPointInSetA (x y : ℤ) : Prop :=
  (x - 3)^2 + (y - 4)^2 ≤ (5 / 2)^2

def isPointInSetB (x y : ℤ) : Prop :=
  (x - 4)^2 + (y - 5)^2 > (5 / 2)^2

def isPointInIntersection (x y : ℤ) : Prop :=
  isPointInSetA x y ∧ isPointInSetB x y

def pointsInIntersection : List (ℤ × ℤ) :=
  [(1, 5), (1, 4), (1, 3), (2, 3), (3, 2), (3, 3), (3, 4)]

theorem count_points_in_intersection_is_7 :
  (List.length pointsInIntersection = 7)
  ∧ (∀ (p : ℤ × ℤ), p ∈ pointsInIntersection → isPointInIntersection p.fst p.snd) :=
by
  sorry

end count_points_in_intersection_is_7_l252_252561


namespace probability_two_same_color_l252_252030

-- Definitions based on conditions
def total_pairs := 14
def blue_pairs := 8
def red_pairs := 4
def green_pairs := 2

def total_socks := 2 * total_pairs
def blue_socks := 2 * blue_pairs
def red_socks := 2 * red_pairs
def green_socks := 2 * green_pairs

-- Statement to prove: the probability that two randomly picked socks are of the same color
theorem probability_two_same_color : 
  ( (blue_socks * (blue_socks - 1) + red_socks * (red_socks - 1) + green_socks * (green_socks - 1)) / (total_socks * (total_socks - 1)) = (77 / 189 : ℚ) ) :=
begin
  sorry
end

end probability_two_same_color_l252_252030


namespace identify_power_function_l252_252055

theorem identify_power_function
  (A : ℝ → ℝ := λ x, 2 * x^2)
  (B : ℝ → ℝ := λ x, x^3 + x)
  (C : ℝ → ℝ := λ x, 3^x)
  (D : ℝ → ℝ := λ x, x^(1/2)) :
  ∃! f, f = D ∧ (f = A → False) ∧ (f = B → False) ∧ (f = C → False) ∧ ∀ y, (y = A ∨ y = B ∨ y = C ∨ y = D) → (y = f → y = D) :=
by
  sorry

end identify_power_function_l252_252055


namespace calculate_expression_l252_252845

theorem calculate_expression : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := 
by
  sorry

end calculate_expression_l252_252845


namespace find_two_digit_number_l252_252886

def digits (n : ℕ) := (n / 10, n % 10)

def is_sought_number (n : ℕ) :=
  let (x, y) := digits n in
  10*x + y = n ∧ (10*y + x) = n - 18

theorem find_two_digit_number :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧
    let (x, y) := digits n in
    (8 * x * y = 3 * (10 * x + y)) ∧
    is_sought_number n ∧
    n = 64 :=
by
  sorry

end find_two_digit_number_l252_252886


namespace modulus_of_complex_number_l252_252540

-- Definitions related to the conditions in the problem
noncomputable def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0

-- Given condition that needs to be satisfied
noncomputable def complexCondition (a : ℝ) : Prop :=
  isPureImaginary ((1 - complex.I) / (a + complex.I))

-- Main statement: proving the modulus of z equals sqrt(11) given the condition
theorem modulus_of_complex_number (a : ℝ) (h : complexCondition a) :
  complex.abs ((2 * a + 1) + sqrt 2 * complex.I) = sqrt 11 := 
sorry

end modulus_of_complex_number_l252_252540


namespace true_propositions_l252_252762

def sin_leq_one (x : ℝ) : Prop := sin x ≤ 1

def prop1 : Prop := ∀ x : ℝ, sin_leq_one x
def neg_prop1: Prop := ∃ x : ℝ, ¬ sin_leq_one x

def exists_sine_condition : Prop := ∃ α β : ℝ, sin (α + β) = sin α + sin β

def geom_seq (a : ℕ → ℝ) (b : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 1 * (b ^ (n - 1))

def proposition3 (a : ℕ → ℝ) (m n p q : ℕ) : Prop :=
  (m + n = p + q) ↔ (a m * a n = a p * a q)

def prop4 (p q : Prop) : Prop := (¬ p ∧ ¬ q) → ¬ (p ∨ q)

theorem true_propositions :
  (¬ prop1 ∧ exists_sine_condition ∧ ∀ (a : ℕ → ℝ) (m n p q : ℕ), ¬ proposition3 a m n p q ∧ ∀ (p q : Prop), prop4 p q) :=
by {
  sorry
}

end true_propositions_l252_252762


namespace consecutive_odd_divisibility_l252_252407

theorem consecutive_odd_divisibility {p q : ℤ} (h1 : q = p + 2) (h2 : Odd p) (h3 : Odd q) :
  (p^p + q^q) % (p + q) = 0 :=
by
  sorry

end consecutive_odd_divisibility_l252_252407


namespace distinct_distances_at_least_44_l252_252348

theorem distinct_distances_at_least_44 (n : ℕ) (distinct_points : ∀ i j (h: i ≠ j), n • ℝ) : n = 2019 → ∃ k : ℕ, k ≥ 44 :=
by
  intro h
  sorry

end distinct_distances_at_least_44_l252_252348


namespace locus_of_points_l252_252927

-- Define the basic structure
structure Rhombus (A B C D : Type) :=
(equal_sides : ∀ (a b : Type), a = b)
(equal_opposite_angles : ∀ (a b : Type), a = b)

-- Define the angles condition
def angle_sum_condition (A P D B C : Type) : Prop := 
  ∃ (angle_APD angle_BPC : ℝ), angle_APD + angle_BPC = 180

-- Define the locus condition
def is_on_diagonals (A P D C B: Type) : Prop := 
  (P = A ∧ P = C) ∨ (P = B ∧ P = D)

-- The theorem statement
theorem locus_of_points (A B C D P : Type) [RH : Rhombus A B C D]
  (h1 : angle_sum_condition A P D B C) :
  is_on_diagonals A P D C B :=
sorry

end locus_of_points_l252_252927


namespace butterfly_flutters_total_distance_l252_252410

-- Define the conditions
def start_pos : ℤ := 0
def first_move : ℤ := 4
def second_move : ℤ := -3
def third_move : ℤ := 7

-- Define a function that calculates the total distance
def total_distance (xs : List ℤ) : ℤ :=
  List.sum (List.map (fun ⟨x, y⟩ => abs (y - x)) (xs.zip xs.tail))

-- Create the butterfly's path
def path : List ℤ := [start_pos, first_move, second_move, third_move]

-- Define the proposition that we need to prove
theorem butterfly_flutters_total_distance : total_distance path = 21 := sorry

end butterfly_flutters_total_distance_l252_252410


namespace true_proposition_l252_252940

/-- Define proposition p: For any x ∈ ℝ, it always holds that 3^x <= 0 -/
def p : Prop := ∀ x : ℝ, 3^x ≤ 0

/-- Define proposition q: "x > 2" is a sufficient but not necessary condition for "x > 4" -/
def q : Prop :=  ∀ x : ℝ, x > 2 → x > 4

/-- The statement to prove -/
theorem true_proposition :
  (¬ p) ∧ (¬ q) := by
sorry

end true_proposition_l252_252940


namespace domain_of_f_l252_252739

def condition1 (x : ℝ) : Prop := 4 - |x| ≥ 0
def condition2 (x : ℝ) : Prop := (x^2 - 5 * x + 6) / (x - 3) > 0

theorem domain_of_f (x : ℝ) :
  (condition1 x) ∧ (condition2 x) ↔ ((2 < x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_l252_252739


namespace sandbox_width_l252_252628

theorem sandbox_width (P : ℕ) (W L : ℕ) (h1 : P = 30) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : W = 5 := 
sorry

end sandbox_width_l252_252628


namespace common_factor_l252_252355

-- Define the polynomials
def P1 (x : ℝ) : ℝ := x^3 + x^2
def P2 (x : ℝ) : ℝ := x^2 + 2*x + 1
def P3 (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem common_factor (x : ℝ) : ∃ (f : ℝ → ℝ), (f x = x + 1) ∧ (∃ g1 g2 g3 : ℝ → ℝ, P1 x = f x * g1 x ∧ P2 x = f x * g2 x ∧ P3 x = f x * g3 x) :=
sorry

end common_factor_l252_252355


namespace slices_per_cheesecake_l252_252447

theorem slices_per_cheesecake (price_per_slice : ℝ) (num_pies : ℕ) (total_money : ℝ) (total_slices_per_pie : ℝ) :
  price_per_slice = 7 → num_pies = 7 → total_money = 294 → total_slices_per_pie = 6 :=
by
  intro h_price h_pies h_money
  have h_total_one_pie := h_money ▸ (294 : ℝ) / (7 : ℕ) = (42 : ℝ)
  have h_slices_per_pie := h_total_one_pie ▸ 42 / 7 = (6 : ℝ)
  exact h_slices_per_pie

end slices_per_cheesecake_l252_252447


namespace symmetric_line_equation_l252_252006

theorem symmetric_line_equation :
  (∃ L : ℝ → ℝ, (∀ x, L (2 * x - 3) = x) ∧ (∀ y, y = x ↔ y = 2 * x - 3)) →
  (∃ k b : ℝ, (∀ x, L x = k * x + b) ∧ (k = 1 / 2) ∧ (b = 3 / 2)) :=
by
  sorry

end symmetric_line_equation_l252_252006


namespace compute_a4_b4_c4_l252_252347

theorem compute_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 8) (h2 : ab + ac + bc = 13) (h3 : abc = -22) : a^4 + b^4 + c^4 = 1378 :=
by
  sorry

end compute_a4_b4_c4_l252_252347


namespace divide_diagonal_into_three_equal_segments_l252_252998

section SquareDivision

variables {A B C D E F : Point} (a : ℝ)

-- Definition of points in the square
def point_A : Point := ⟨0, 0⟩
def point_B : Point := ⟨a, 0⟩
def point_C : Point := ⟨a, a⟩
def point_D : Point := ⟨0, a⟩

-- Midpoints of the sides
def point_E : Point := ⟨a, a / 2⟩
def point_F : Point := ⟨a / 2, a⟩

-- Assumption: A, B, C, D form a square
noncomputable def is_square : Prop := 
  square point_A point_B point_C point_D a

-- The theorem statement
theorem divide_diagonal_into_three_equal_segments :
  is_square A B C D a → 
  midpoint B C E → 
  midpoint C D F → 
  divides_diagonal_into_equal_segments A B C D E F 3 :=
begin
    sorry
end

end SquareDivision

end divide_diagonal_into_three_equal_segments_l252_252998


namespace evaluate_f_l252_252459

def f (x : ℝ) := x^3 + 3 * Real.sqrt x

theorem evaluate_f :
  f 3 + 3 * f 1 - 2 * f 5 = -211 + 3 * Real.sqrt 3 - 6 * Real.sqrt 5 :=
by
  sorry

end evaluate_f_l252_252459


namespace excircle_diameter_l252_252467

noncomputable def diameter_of_excircle (a b c S : ℝ) (s : ℝ) : ℝ :=
  2 * S / (s - a)

theorem excircle_diameter (a b c S h_A : ℝ) (s : ℝ) (h_v : 2 * ((a + b + c) / 2) = a + b + c) :
    diameter_of_excircle a b c S s = 2 * S / (s - a) :=
by
  sorry

end excircle_diameter_l252_252467


namespace coeff_of_x_neg_one_in_expansion_l252_252149

theorem coeff_of_x_neg_one_in_expansion :
  let binom_coeff (n k : ℕ) := nat.choose n k in
  let expand (a b : ℕ) (x : ℝ) := ∑ k in (range (a+1)), binom_coeff a k * (1 - (b/x))^(a-k) * (2/x)^k in
  let expr (x : ℝ) := (x^2 - 2) * expand 5 2 x,
  (coeff_of_term expr (-1) = 60) :=
begin
  sorry
end

end coeff_of_x_neg_one_in_expansion_l252_252149


namespace minimum_value_l252_252303

def z := ℂ
def c1 := 3 - 3 * complex.I
def c2 := 2 + complex.I
def c3 := 6 - 2 * complex.I

theorem minimum_value 
  (z : ℂ)
  (h : complex.abs (z - c1) = 3) :
  let expr := (complex.abs (z - c2))^2 + (complex.abs (z - c3))^2 in
  expr = 59 :=
sorry

end minimum_value_l252_252303


namespace solve_inequality_l252_252486

theorem solve_inequality (x : ℝ) : 
  (1 / (x^2 + 1) > 3 / x + 17 / 10) → x ∈ set.Ioo (-2 : ℝ) 0 :=
by
  have h : (x^2 + 1) > 0 := by nlinarith
  have h1 : x ≠ 0 := by
    intro hx
    rw [hx] at h
    nlinarith
  sorry
  -- Add further necessary steps or lemmas here.

end solve_inequality_l252_252486


namespace area_of_triangle_abc_l252_252795

-- Definitions representing the conditions
def triangle_abc (A B C : Point) : Prop :=
  ∃ L : Point, 
    (angle_bisector B A C L) ∧ 
    (dist A L = 3) ∧ 
    (dist B L = 6 * Real.sqrt 5) ∧ 
    (dist C L = 4)

-- The theorem we want to prove
theorem area_of_triangle_abc {A B C : Point} (h : triangle_abc A B C) : 
  area_triangle A B C = 21 * Real.sqrt 55 / 4 :=
sorry

end area_of_triangle_abc_l252_252795


namespace ellipse_equation_lambda_range_l252_252933

-- First part: equation of ellipse C
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) (e_def : e = sqrt 3 / 2) :
  a = 2 * b ∧ b = 1 := 
by
  sorry

-- Second part: range of values for λ
theorem lambda_range (x_0 y_0 : ℝ) (h1 : -1 < y_0) (h2 : y_0 < 1) 
  (h3 : x_0^2 / 4 + y_0^2 = 1) (λ : ℝ) :
  (λ^2 = 1 / (-3 * y_0^2 + 2 * y_0 + 5)) →
  λ ≥ sqrt 3 / 4 :=
by
  sorry

end ellipse_equation_lambda_range_l252_252933


namespace extremum_of_cubic_function_l252_252359

noncomputable def cubic_function (x : ℝ) : ℝ := 2 - x^2 - x^3

theorem extremum_of_cubic_function : 
  ∃ x_max x_min : ℝ, 
    cubic_function x_max = x_max_value ∧ 
    cubic_function x_min = x_min_value ∧ 
    ∀ x : ℝ, cubic_function x ≤ cubic_function x_max ∧ cubic_function x_min ≤ cubic_function x :=
sorry

end extremum_of_cubic_function_l252_252359


namespace initial_masses_of_crystals_l252_252083

section crystals

variable (a b : ℝ)

-- Conditions
def first_crystal_yearly_increase (a : ℝ) : ℝ := 0.04 * a
def second_crystal_yearly_increase (b : ℝ) : ℝ := 0.05 * b
def first_crystal_three_month_increase (a : ℝ) : ℝ := (first_crystal_yearly_increase a) / 4
def second_crystal_four_month_increase (b : ℝ) : ℝ := (second_crystal_yearly_increase b) / 3
def mass_ratio_after_20g_increase (a b : ℝ) : Prop := (a + 20) / (b + 20) = 1.5

-- Theorem
theorem initial_masses_of_crystals
    (H1 : first_crystal_three_month_increase a = second_crystal_four_month_increase b)
    (H2 : mass_ratio_after_20g_increase a b) :
    a = 100 ∧ b = 60 := by
  sorry

end crystals

end initial_masses_of_crystals_l252_252083


namespace geom_seq_log_eqn_l252_252538

theorem geom_seq_log_eqn {a : ℕ → ℝ} {b : ℕ → ℝ}
    (geom_seq : ∃ (r : ℝ) (a1 : ℝ), ∀ n : ℕ, a (n + 1) = a1 * r^n)
    (log_seq : ∀ n : ℕ, b n = Real.log (a (n + 1)) / Real.log 2)
    (b_eqn : b 1 + b 3 = 4) : a 2 = 4 :=
by
  sorry

end geom_seq_log_eqn_l252_252538


namespace number_of_passed_candidates_l252_252399

-- Definitions based on conditions:
def total_candidates : ℕ := 120
def avg_total_marks : ℝ := 35
def avg_passed_marks : ℝ := 39
def avg_failed_marks : ℝ := 15

-- The number of candidates who passed the examination:
theorem number_of_passed_candidates :
  ∃ (P F : ℕ), 
    P + F = total_candidates ∧
    39 * P + 15 * F = total_candidates * avg_total_marks ∧
    P = 100 :=
by
  sorry

end number_of_passed_candidates_l252_252399


namespace triangle_area_proof_l252_252796

noncomputable def triangle_area (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  1 / 2 * (B - A).norm * (C - A).norm * Real.sin (Real.angle (B - A) (C - A))

theorem triangle_area_proof :
  let A := EuclideanSpace.ofPoint ℝ (Fin 2) ![0, 0],
      B := EuclideanSpace.ofPoint ℝ (Fin 2) ![b, 0],
      C := EuclideanSpace.ofPoint ℝ (Fin 2) ![c_x, c_y] in
  let AL := 3,
      BL := 6 * Real.sqrt 5,
      CL := 4 in
  let area := triangle_area A B C in
  area = 21 * Real.sqrt 55 / 4 :=
begin
  sorry
end

end triangle_area_proof_l252_252796


namespace find_ff_half_l252_252177

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then x + 1 else -x + 3

theorem find_ff_half : f (f (1 / 2)) = 3 / 2 := 
by 
  sorry

end find_ff_half_l252_252177


namespace area_of_triangle_l252_252247

theorem area_of_triangle
  (a b c : ℝ) (α β γ : ℝ) 
  (h1 : b = 5) 
  (h2 : cos γ = 1 / 8)
  (h3 : cos α = 3 / 4) :
  1 / 2 * a * b * sin γ = 15 * sqrt 7 / 4 :=
by
  sorry

end area_of_triangle_l252_252247


namespace find_real_number_m_l252_252541

theorem find_real_number_m (m : ℝ) (z : ℂ) 
    (hz : z = (m^2 + 2*m - 3 : ℝ) + (m - 1 : ℝ) * complex.i)
    (hz_pure_imaginary : (m^2 + 2*m - 3 = 0) ∧ (m ≠ 1)) : m = -3 := by
  sorry

end find_real_number_m_l252_252541


namespace alternating_factorial_base_sum_correct_l252_252141

noncomputable def alternating_factorial_base_sum : ℕ :=
  ∑ n in List.range 124, (if n % 2 = 0 then 16 * (n / 2 + 1)! else -(16 * ((n - 1) / 2 + 1)!))

theorem alternating_factorial_base_sum_correct :
  f_1 - f_2 + f_3 - f_4 + ⋯ + (-1) ^ (list_length + 1) * f_j = 495 :=
begin
  sorry
end

end alternating_factorial_base_sum_correct_l252_252141


namespace students_answered_both_questions_correctly_l252_252327

theorem students_answered_both_questions_correctly :
  let students_enrolled := 30
  let answered_q1 := 25
  let answered_q2 := 22
  let no_test := 5
  let students_took_test := students_enrolled - no_test in
  let A_union_B := students_took_test in
  (answered_q1 + answered_q2 - A_union_B) = 22 :=
by
  sorry

end students_answered_both_questions_correctly_l252_252327


namespace range_of_m_l252_252744

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < |m - 2|) ↔ m < 0 ∨ m > 4 := 
sorry

end range_of_m_l252_252744


namespace coupon1_best_at_180_l252_252085

theorem coupon1_best_at_180 (P : ℝ) (hP_gt_50 : P ≥ 50) (hP_gt_100 : P ≥ 100) (hP_gt_150 : P ≥ 150):
  P = 180 →
  let discount1 := 0.15 * P
  let discount2 := 25
  let discount3 := 0.25 * (P - 150)
  discount1 > discount2 ∧ discount1 > discount3 :=
by
  intros hP_eq
  have hP1 : 0.15 * 180 = 27, from by norm_num
  have hP2 : 0.25 * (180 - 150) = 7.5, from by norm_num
  rw [hP_eq] at discount1 discount2 discount3
  rw [hP1, hP2]
  split
  { norm_num }
  { norm_num }

end coupon1_best_at_180_l252_252085


namespace find_letter_l252_252332

theorem find_letter (d : ℕ) (t : ℕ)
  (hA : ∃ (a: ℕ), a = d + 2)
  (hB : ∃ (b: ℕ), b = d + 8)
  (hC : ∃ (c: ℕ), c = d)
  (sumAB : 2 * d + 10 = d + t)
  (ht : t = d + 10) : t = d + 10 :=
begin
  sorry
end

end find_letter_l252_252332


namespace problem_is_correct_l252_252630

theorem problem_is_correct :
  let y := 58 + 104 + 142 + 184 + 304 + 368 + 3304 in
  (y % 2 = 0) ∧ (y % 4 = 0) ∧ (¬(y % 8 = 0)) ∧ (¬(y % 16 = 0)) :=
by
  let y := 58 + 104 + 142 + 184 + 304 + 368 + 3304
  have h2 : y % 2 = 0 := sorry
  have h4 : y % 4 = 0 := sorry
  have h8 : ¬(y % 8 = 0) := sorry
  have h16 : ¬(y % 16 = 0) := sorry
  exact ⟨h2, h4, h8, h16⟩

end problem_is_correct_l252_252630


namespace probability_two_digit_between_15_25_l252_252650

-- Define a type for standard six-sided dice rolls
def is_standard_six_sided_die (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- Define the set of valid two-digit numbers
def valid_two_digit_number (n : ℕ) : Prop := n ≥ 15 ∧ n ≤ 25

-- Function to form a two-digit number from two dice rolls
def form_two_digit_number (d1 d2 : ℕ) : ℕ := 10 * d1 + d2

-- The main statement of the problem
theorem probability_two_digit_between_15_25 :
  (∃ (n : ℚ), n = 5/9) ∧
  (∀ (d1 d2 : ℕ), is_standard_six_sided_die d1 → is_standard_six_sided_die d2 →
  valid_two_digit_number (form_two_digit_number d1 d2)) :=
sorry

end probability_two_digit_between_15_25_l252_252650


namespace points_A_B_D_collinear_l252_252512

variable (a b : ℝ)

theorem points_A_B_D_collinear
  (AB : ℝ × ℝ := (a, 5 * b))
  (BC : ℝ × ℝ := (-2 * a, 8 * b))
  (CD : ℝ × ℝ := (3 * a, -3 * b)) :
  AB = (BC.1 + CD.1, BC.2 + CD.2) := 
by
  sorry

end points_A_B_D_collinear_l252_252512


namespace area_BCED_l252_252611

-- Define the points and relevant line segments
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B]
[MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Define the conditions: area of triangle ABC, DE length
axiom area_ABC : real
axiom DE_length : real
axiom DE_eq_10 : DE_length = 10
axiom area_ABC_eq_36 : area_ABC = 36

-- Define the question transformed to a Lean 4 statement
theorem area_BCED : 
  (triangle_area B C E) + (triangle_area B D E) = 144 := 
by
  -- Using the given conditions
  have h := (2 * area_ABC_eq_36) / 8 -- Derive height 'h' from the area of triangle ABC
  have area_BCE := (1/2) * (22) * h
  have area_BDE := (1/2) * (DE_length) * h
  rw [DE_eq_10] at area_BDE
  have BCED_area := area_BCE + area_BDE
  rw [BCED_area, area_ABC_eq_36]
  -- The area BCED should be 144 split due to the addition of areas BCE + BDE given area is equal to 36
  sorry

end area_BCED_l252_252611


namespace find_divisor_l252_252157

theorem find_divisor (X : ℕ) (h12 : 12 ∣ (1020 - 12)) (h24 : 24 ∣ (1020 - 12)) (h48 : 48 ∣ (1020 - 12)) (h56 : 56 ∣ (1020 - 12)) :
  X = 63 :=
sorry

end find_divisor_l252_252157


namespace percentage_of_divisibles_by_6_l252_252393

-- Define the condition that checks if a number is divisible by 6.
def isDivisibleBy6 (n : ℕ) : Prop :=
  n % 6 = 0

-- Define the set of positive integers less than or equal to 120.
def lessEqual120 (n : ℕ) : Prop :=
  n ≤ 120

-- Define the set of numbers that are positive integers less than or equal to 120 and divisible by 6.
def divBy6LessEqual120Set : Finset ℕ :=
  Finset.filter isDivisibleBy6 (Finset.range 121) -- Since Finset.range n generates numbers from 0 to n-1.

-- Define the cardinality of the set of numbers that are positive integers less than or equal to 120.
def totalNumbersCardinality : ℕ :=
  120

-- Define the cardinality of the set of numbers that are positive integers less than or equal to 120 and divisible by 6.
def divBy6Cardinality : ℕ :=
  (divBy6LessEqual120Set).card

-- Define the percentage of numbers that are positive integers less than or equal to 120 and divisible by 6.
def divBy6Percentage : ℚ :=
  (divBy6Cardinality : ℚ) / totalNumbersCardinality * 100

-- State the theorem in Lean to prove the necessary percentage.
theorem percentage_of_divisibles_by_6 :
  divBy6Percentage = 16.67 := by
  sorry

end percentage_of_divisibles_by_6_l252_252393


namespace goats_in_field_l252_252994

theorem goats_in_field (total_animals cows sheep chickens : ℕ)
  (h1 : total_animals = 900)
  (h2 : cows = 250)
  (h3 : sheep = 310)
  (h4 : chickens = 180) :
  total_animals - (cows + sheep + chickens) = 160 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end goats_in_field_l252_252994


namespace question_b_l252_252789

theorem question_b (a b c : ℝ) (h : c ≠ 0) (h_eq : a / c = b / c) : a = b := 
by
  sorry

end question_b_l252_252789


namespace volume_revolution_correct_l252_252831

-- Definitions of given conditions
variables (a b : ℝ)
variables (ABCD : ℝ -> ℝ -> Prop)  -- Placeholder for the rectangle property
variables (AB AD diagonal l: ℝ)

-- Assume the rectangle properties
axiom rectangle_definition : ABCD a b

-- The lengths of the sides
axiom length_AB : AB = a
axiom length_AD : AD = b

-- Axis l passes through A
axiom axis_through_A : l = a * b  -- This is a symbolic definition to indicate axis through A

-- l is parallel to the diagonal BD
axiom axis_parallel_diagonal : l = a * b  -- This is a symbolic definition to indicate axis parallel to diagonal [BD]

-- Computing the volume of the solid of revolution
noncomputable def volume_of_solid_of_revolution (a b : ℝ) : ℝ :=
  (2 * Math.pi * a^2 * b^2) / Real.sqrt (a^2 + b^2)

-- Statement to prove
theorem volume_revolution_correct :
  ∃ V : ℝ, V = volume_of_solid_of_revolution a b :=
begin
  use (2 * Math.pi * a^2 * b^2) / Real.sqrt (a^2 + b^2),
  sorry
end

end volume_revolution_correct_l252_252831


namespace subcommittees_with_experts_l252_252365

def total_members : ℕ := 12
def experts : ℕ := 5
def non_experts : ℕ := total_members - experts
def subcommittee_size : ℕ := 5

theorem subcommittees_with_experts :
  (nat.choose total_members subcommittee_size) - (nat.choose non_experts subcommittee_size) = 771 := by
  sorry

end subcommittees_with_experts_l252_252365


namespace sphere_volume_given_surface_area_l252_252020

variable (r : ℝ)

-- Given: Surface area of a sphere is 144π cm²
def surface_area : ℝ := 4 * π * r^2

-- To prove: Volume of the sphere is 288π cm³
def volume : ℝ := (4 / 3) * π * r^3

theorem sphere_volume_given_surface_area :
  surface_area r = 144 * π → volume r = 288 * π :=
by
  -- proof goes here
  sorry

end sphere_volume_given_surface_area_l252_252020


namespace height_of_regular_triangular_pyramid_l252_252014

-- Problem Statement: Given conditions and correct answer
theorem height_of_regular_triangular_pyramid
  (a : ℝ)
  (ABC : Triangle)
  (h_reg : is_equilateral ABC)
  (P : Point)
  (M : Point := centroid ABC)
  (P_on_lateral : ∀ (X : Point), X ∈ ABC.vertices → ⦞ (angle (P -ᵥ M) (X -ᵥ M)) = 60) :
  
  height_of_pyramid ABC P = a :=
begin
  sorry
end

end height_of_regular_triangular_pyramid_l252_252014


namespace geometric_sequence_thm_l252_252599

noncomputable def geometric_sequence_term (a r : ℕ → ℕ) (n : ℕ) : ℕ :=
  a n * r^(n-1)

theorem geometric_sequence_thm (a_1 a_2 a_3 a_4 r : ℕ) (h1 : r = 2) (h2 : a_1 + a_2 = 3) : a_3 + a_4 = 12 :=
by
  sorry

end geometric_sequence_thm_l252_252599


namespace find_a_l252_252196

theorem find_a (a : ℝ) (i : ℂ := complex.I) (z : ℂ := (a - i) / (1 - i)) :
  |z| = ∫ x in 0..π, sin x - (1 / π) → a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l252_252196


namespace average_side_lengths_l252_252700

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252700


namespace reusable_bags_theorem_conditional_probability_theorem_l252_252413

def number_of_reusable_bags(data : List (Nat × Nat), total_customers : Nat, expected_customers : Nat) :=
  let people_spending_at_least_188 := data.foldl (fun acc x => acc + x.1) 0
  let frequency := (people_spending_at_least_188 : Float) / (total_customers : Float)
  let estimated_bags := Float.ofNat expected_customers * frequency
  estimated_bags

theorem reusable_bags_theorem :
  number_of_reusable_bags([(8, 2), (15, 3), (23, 5), (15, 9), (9, 11)], 100, 5000) = 3500 := by
  sorry

-- Calculation for conditional probability
def conditional_probability(data : List (Nat × Nat), total_customers : Nat) :=
  let total_less_188 := data.foldl (fun acc x => acc + x.2) 0
  let total_at_least_188 := data.foldl (fun acc x => acc + x.1) 0
  let pA := (total_less_188 : Float) / (total_customers : Float)
  let pAB := (total_less_188 : Float) / (total_customers : Float) * (total_at_least_188 : Float) / (total_customers - 1 : Float)
  let pB_given_A := pAB / pA
  pB_given_A

theorem conditional_probability_theorem :
  conditional_probability([(8, 2), (15, 3), (23, 5), (15, 9), (9, 11)], 100) = 70 / 99 := by
  sorry

end reusable_bags_theorem_conditional_probability_theorem_l252_252413


namespace length_of_stone_slab_in_cm_l252_252408

-- Defining the conditions and the proof goal
theorem length_of_stone_slab_in_cm :
  ∀ (num_slabs : ℕ) (total_area_sqm : ℝ), 
  num_slabs = 30 → total_area_sqm = 120 →
  let area_per_slab := total_area_sqm / num_slabs in
  let side_length_m := real.sqrt area_per_slab in
  let side_length_cm := side_length_m * 100 in
  side_length_cm = 200 :=
by
  intros num_slabs total_area_sqm h_num_slabs h_total_area_sqm
  rw [h_num_slabs, h_total_area_sqm]
  let area_per_slab := total_area_sqm / num_slabs
  let side_length_m := real.sqrt area_per_slab
  let side_length_cm := side_length_m * 100
  have h1 : area_per_slab = 4 := by norm_num
  have h2 : side_length_m = 2 := by norm_num [h1, real.sqrt]
  have h3 : side_length_cm = 200 := by norm_num [h2]
  exact h3

end length_of_stone_slab_in_cm_l252_252408


namespace max_red_dragons_l252_252613

theorem max_red_dragons (n : ℕ) (h_n : n = 530)
  (head1 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → ∃ (g : bool), g = true → dragons ((i - 1) % n)) 
  (head2 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → ∃ (g : bool), g = true → dragons ((i + 1) % n)) 
  (head3 : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → ∃ (g : bool), g = true → ¬ ∃ (r : bool), r = true ∧ r = dragons (i)) :
  ∃ m, m = 176 := 
sorry

end max_red_dragons_l252_252613


namespace determine_a_l252_252544

-- Define the line equation
def line (x y a: ℝ) := x + y - 2 * a = 0

-- Define the circle equation
def circle (x y a: ℝ) := (x - 1) ^ 2 + (y - a) ^ 2 = 4

-- Define the condition for the triangle to be equilateral
def equilateral_triangle (a: ℝ) : Prop :=
  (∃ x y: ℝ, line x y a ∧ circle x y a ∧
  -- Distance from the center of the circle to the line equals √3
  (abs (1 - a) / (real.sqrt 2) = real.sqrt 3))

-- Statement to be proven
theorem determine_a (a: ℝ) (h: equilateral_triangle a) : a = 1 + real.sqrt 6 ∨ a = 1 - real.sqrt 6 :=
sorry

end determine_a_l252_252544


namespace domain_of_function_is_all_real_l252_252775

def domain_function : Prop :=
  ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 6 ≠ 0

theorem domain_of_function_is_all_real :
  domain_function :=
by
  intros t
  sorry

end domain_of_function_is_all_real_l252_252775


namespace simplest_radical_form_check_l252_252439

def is_simplest_radical_form (x : ℝ) : Prop :=
  ∀ (i j : ℕ), (i * i = j) → ¬ (x = j)

def simplest_radical_form (r : ℝ) : Prop :=
  (∃ (n : ℕ), r = sqrt (n) ∧ (∀ (m : ℕ), is_simplest_radical_form (m) → n ≠ m))

theorem simplest_radical_form_check :
  simplest_radical_form (sqrt 30) ∧
  ¬ simplest_radical_form (sqrt (3 * (a ^ 2))) ∧
  ¬ simplest_radical_form (sqrt (2 / 3)) ∧
  ¬ simplest_radical_form (sqrt 24) :=
by sorry

end simplest_radical_form_check_l252_252439


namespace no_m_n_for_property_l252_252503

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def product_of_first_m_primes (m : ℕ) : ℕ :=
  @List.prod ℕ _ (List.take m (List.filter is_prime (List.range (m+2))))

theorem no_m_n_for_property (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  product_of_first_m_primes m ≠ n * (n + 1) * (n + 2) * (n + 3) :=
by {
  sorry
}

end no_m_n_for_property_l252_252503


namespace magnitude_not_equal_imaginary_part_correct_same_quadrant_not_on_graph_l252_252934

namespace ComplexProof

def z1 : ℂ := 1 + 2 * Complex.I
def z2 : ℂ := 3 - 4 * Complex.I
def z1z2 : ℂ := (1 + 2 * Complex.I) * (3 - 4 * Complex.I)

-- Statement for Option A
theorem magnitude_not_equal : |z1| ≠ |z2| := by sorry

-- Statement for Option B
theorem imaginary_part_correct : (z1z2).im = 2 := by sorry

-- Statement for Option C
theorem same_quadrant : (1 > 0 ∧ 2 > 0) ∧ (11 > 0 ∧ 2 > 0) := by sorry

-- Statement for Option D
theorem not_on_graph : ¬ (3 = 3 ∧ -4 = 2 * 3 - 2) := by sorry

end ComplexProof

end magnitude_not_equal_imaginary_part_correct_same_quadrant_not_on_graph_l252_252934


namespace power_of_point_l252_252849

-- Define the circle, points, and conditions
variables {O A B C D M P Q : Type*}
variables [incidence_geometry O A B C D M P Q]

-- The conditions of the problem
axiom diameters_perpendicular : ∀ O A B C D, perpendicular (line_through O A B) (line_through O C D)
axiom chord_intersects_diameter : ∀ A M C D P, intersects (line_through A M) (line_through C D) P
axiom point_not_center : ∀ O P, P ≠ O
axiom pb_intersects_circle_at_Q : ∀ P B Q, intersects (line_through P B) (circle_through O A)

-- The mathematical equivalent proof statement
theorem power_of_point {P Q A C D} (h : ∀ (O A B C D : Type*) (AM CD P : Point) (Q : Point),
    perpendicular (line_through O A B) (line_through O C D) ∧
    intersects (line_through A M) (line_through C D) P ∧
    P ≠ O ∧
    intersects (line_through P B) (circle_through O A)) : 
    AP * AQ = CP * PQ :=
sorry

end power_of_point_l252_252849


namespace domain_of_f_l252_252489

def f (x : ℝ) : ℝ := (x - 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | x^2 - 5 * x + 6 ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l252_252489


namespace ratio_shorter_to_longer_l252_252617

theorem ratio_shorter_to_longer (x y : ℝ) (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = y / 3) : x / y = 5 / 12 :=
sorry

end ratio_shorter_to_longer_l252_252617


namespace ratio_of_segments_l252_252464

variable (F S T : ℕ)

theorem ratio_of_segments : T = 10 → F = 2 * (S + T) → F + S + T = 90 → (T / S = 1 / 2) :=
by
  intros hT hF hSum
  sorry

end ratio_of_segments_l252_252464


namespace polynomial_coefficient_inequality_l252_252527

variable {n : ℕ} (a : Fin n → ℝ)
variable (p : ℝ → ℝ)
variable (hn : 3 ≤ n)
variable (roots_left_half_plane : ∀ (z : ℂ), z ∈ (p.root_set ℂ) → z.re < 0)

theorem polynomial_coefficient_inequality (hpoly : ∀ x, p x = ∑ i in Finset.range n, a ⟨i, sorry⟩ * x^(n - i - 1)):
  ∀ k : ℕ, 0 ≤ k → k ≤ n - 3 → a ⟨k, sorry⟩ * a ⟨k + 3, sorry⟩ < a ⟨k + 1, sorry⟩ * a ⟨k + 2, sorry⟩ :=
sorry

end polynomial_coefficient_inequality_l252_252527


namespace average_side_length_of_squares_l252_252688

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252688


namespace area_ratio_is_eight_l252_252591

-- Define conditions
variable (s₂ : ℝ) -- Let the side length of the second square be s₂.

-- Let the side length of the first square be s₁.
def s₁ := 2 * s₂ * Real.sqrt 2

-- Define the area of the first square
def A₁ := s₁^2

-- Define the area of the second square
def A₂ := s₂^2

theorem area_ratio_is_eight (s₂ : ℝ) : 
  (s₁ s₂ / s₂) ^ 2 = 8 := 
sorry

end area_ratio_is_eight_l252_252591


namespace average_of_side_lengths_of_squares_l252_252693

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252693


namespace smallest_prime_with_reversible_composite_l252_252498

def is_prime (n : ℕ) : Prop := nat.prime n
def is_composite (n : ℕ) : Prop := 2 ≤ nat.min_fac n ∧ nat.min_fac n < n 
def reverse_digits (n : ℕ) : ℕ := 
  let digits := n.digits 10 in 
  nat.of_digits 10 (list.reverse digits)

theorem smallest_prime_with_reversible_composite : (∀ n : ℕ, n ≥ 10 ∧ n < 100 ∧ is_prime n ∧ n / 10 = 4 ∧ is_composite (reverse_digits n) → n ≥ 41) 
  ∧ ∃ n : ℕ, n = 41 ∧ is_prime n ∧ n / 10 = 4 ∧ is_composite (reverse_digits n) := 
by 
  sorry

end smallest_prime_with_reversible_composite_l252_252498


namespace correct_statement_is_C_l252_252840

theorem correct_statement_is_C:
  (A : ∀ {q : Type} [has_equ (q×q)] [has_add q], (is_quadrilateral q → opposite_sides_equal q → is_parallelogram q)) →
  (B : ∀ {q : Type} [has_equ q], (is_quadrilateral q → sides_equal q four → is_rhombus q)) →
  (C : ∀ {l₁ l₂ l₃ : Type} [is_line l₁] [is_line l₂] [is_line l₃], (parallel l₁ l₃ → parallel l₂ l₃ → parallel l₁ l₂)) →
  (D : ∀ {p₁ p₂ p₃ : Type} [is_point p₁] [is_point p₂] [is_point p₃], (three_points p₁ p₂ p₃ → determine_plane p₁ p₂ p₃)) →
  (correct : Prop): correct = C :=
begin
  sorry
end

end correct_statement_is_C_l252_252840


namespace primes_with_large_gap_exists_l252_252656

noncomputable def exists_primes_with_large_gap_and_composites_between : Prop :=
  ∃ p q : ℕ, p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p > 2015 ∧ (∀ n : ℕ, p < n ∧ n < q → ¬Nat.Prime n)

theorem primes_with_large_gap_exists : exists_primes_with_large_gap_and_composites_between := sorry

end primes_with_large_gap_exists_l252_252656


namespace correct_time_is_500_l252_252743

theorem correct_time_is_500 ({T T_fast T_slow : ℕ}):
  (T_fast = T + 10) ∧ (T_slow = T - 10) ∧ (T_fast = 70) ∧ (T_slow = 50) → 
  T = 60 := 
by
  sorry

end correct_time_is_500_l252_252743


namespace prime_count_l252_252572

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def two_digit_numbers := {x : ℕ | 10 ≤ x ∧ x < 100}

def digits := {3, 5, 7}

def valid_two_digit_numbers :=
  {n | ∃ (d1 d2 : ℕ), d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2}

theorem prime_count :
  (finset.filter is_prime (finset.filter (λ n, n ∈ valid_two_digit_numbers) two_digit_numbers)).card = 3 :=
sorry

end prime_count_l252_252572


namespace find_y_l252_252981

theorem find_y (y : ℕ) (hy1 : y % 9 = 0) (hy2 : y^2 > 200) (hy3 : y < 30) : y = 18 :=
sorry

end find_y_l252_252981


namespace roots_of_transformed_quadratic_l252_252205

theorem roots_of_transformed_quadratic (a b c d x : ℝ) :
  (∀ x, (x - a) * (x - b) - x = 0 → x = c ∨ x = d) →
  (x - c) * (x - d) + x = 0 → x = a ∨ x = b :=
by
  sorry

end roots_of_transformed_quadratic_l252_252205


namespace max_fleas_on_board_l252_252652

theorem max_fleas_on_board (n : ℕ) (m : ℕ) (jump : ℕ → ℕ → ℕ × ℕ) :
  n = 10 ∧ m = 10 ∧ (∀ k : ℕ, k < 60 → ∀ (x y : ℕ), 
  (x < n) → (y < m) → 
  let (nxk, nyk) := jump x y in
  (nxk < n ∧ nyk < m) ∧ 
  abs (nxk - x) + abs (nyk - y) = 1) →
  (∃ S : set (ℕ × ℕ), S.card = 40 ∧ 
   ∀ p1 p2 : ℕ × ℕ, 
   p1 ∈ S → p2 ∈ S → 
   (∀ k : ℕ, k < 60 → 
    (let (n1k, m1k) := jump (p1.1) (p1.2) in
     let (n2k, m2k) := jump (p2.1) (p2.2) in 
     (n1k, m1k) ≠ (n2k, m2k)))) :=
sorry

end max_fleas_on_board_l252_252652


namespace distance_between_x_intercepts_l252_252821

theorem distance_between_x_intercepts :
  let slope1 := 4
  let slope2 := -2
  let point := (8, 20)
  let line1 (x : ℝ) := slope1 * (x - point.1) + point.2
  let line2 (x : ℝ) := slope2 * (x - point.1) + point.2
  let x_intercept1 := (0 - point.2) / slope1 + point.1
  let x_intercept2 := (0 - point.2) / slope2 + point.1
  abs (x_intercept1 - x_intercept2) = 15 := sorry

end distance_between_x_intercepts_l252_252821


namespace complex_conjugate_of_z_l252_252204

-- Definitions for the given problem
def z : ℂ := 3 / (1 - 2 * complex.I)

-- Theorem statement that encapsulates the proof problem
theorem complex_conjugate_of_z : complex.conj z = (3 / 5) - (6 / 5) * complex.I :=
by
  sorry

end complex_conjugate_of_z_l252_252204


namespace book_cost_l252_252375

theorem book_cost (x y : ℝ) (h₁ : 2 * y = x) (h₂ : 100 + y = x - 100) : x = 200 := by
  sorry

end book_cost_l252_252375


namespace retirement_year_l252_252394

noncomputable def year_hired := 1989
noncomputable def age_when_hired := 32
noncomputable def required_sum := 70

theorem retirement_year : 
  ∃ (year_retired : ℕ),
    (age_when_hired + 2 * (year_retired - year_hired) = required_sum) ∧ 
    (year_retired = 2008) :=
by {
  let YH := year_hired,
  let AH := age_when_hired,
  let required_sum := required_sum,
  let YR := 2008,
  have YE := YR - YH,
  have AR := AH + YE,
  have h₁: AR + YE = 70,
  have h₂: AH + 2 * (YR - YH) = 70,
  use YR,
  split,
  exact h₂,
  exact eq.refl YR,
}

end retirement_year_l252_252394


namespace extreme_value_result_l252_252311

open Real

-- Conditions
def function_has_extreme_value_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop := 
  deriv f x₀ = 0

-- The given function
noncomputable def f (x : ℝ) : ℝ := x * sin x

-- The problem statement (to prove)
theorem extreme_value_result (x₀ : ℝ) 
  (h : function_has_extreme_value_at f x₀) :
  (1 + x₀^2) * (1 + cos (2 * x₀)) = 2 :=
sorry

end extreme_value_result_l252_252311


namespace find_a_for_odd_function_l252_252637

theorem find_a_for_odd_function (f : ℝ → ℝ) (a : ℝ) (h₀ : ∀ x, f (-x) = -f x) (h₁ : ∀ x, x < 0 → f x = x^2 + a * x) (h₂ : f 3 = 6) : a = 5 :=
by
  sorry

end find_a_for_odd_function_l252_252637


namespace inequality_l252_252562

theorem inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) : 
  (a / b) + (b / c) + (c / a) + (b / a) + (a / c) + (c / b) + 6 ≥ 
  2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
sorry

end inequality_l252_252562


namespace label_points_l252_252603

-- Defining the conditions
def points {α : Type*} [metric_space α] (n : ℕ) : set α → Prop :=
λ P, P.card = n ∧ ∀ (A B C : α), A ∈ P → B ∈ P → C ∈ P → A ≠ B → B ≠ C → C ≠ A → 
∃ (angle : ℝ), angle_triang A B C angle ∧ angle > 120

-- The theorem to prove
theorem label_points {α : Type*} [metric_space α] {n : ℕ} {P : set α} (hP : points n P) :
  ∃ (l : fin n → α), (∀ i j k : fin n, 1 ≤ i.val → i.val < j.val → j.val < k.val → k.val ≤ n → 
  ∃ (angle : ℝ), angle_triang (l i) (l j) (l k) angle ∧ angle > 120) :=
sorry

end label_points_l252_252603


namespace max_chain_triangles_l252_252133

theorem max_chain_triangles (n : ℕ) (h : n > 0) : 
  ∃ k, k = n^2 - n + 1 := 
sorry

end max_chain_triangles_l252_252133


namespace difference_q_r_l252_252442

theorem difference_q_r (x : ℝ) (p q r : ℝ) 
  (h1 : 7 * x - 3 * x = 3600) 
  (h2 : q = 7 * x) 
  (h3 : r = 12 * x) :
  r - q = 4500 := 
sorry

end difference_q_r_l252_252442


namespace solve_inequality_system_l252_252345

theorem solve_inequality_system (x : ℝ) :
  (x / 3 + 2 > 0) ∧ (2 * x + 5 ≥ 3) ↔ (x ≥ -1) :=
by
  sorry

end solve_inequality_system_l252_252345


namespace probability_first_head_second_tail_l252_252783

-- Conditions
def fair_coin := true
def prob_heads := 1 / 2
def prob_tails := 1 / 2
def independent_events (A B : Prop) := true

-- Statement
theorem probability_first_head_second_tail :
  fair_coin →
  independent_events (prob_heads = 1/2) (prob_tails = 1/2) →
  (prob_heads * prob_tails) = 1/4 :=
by
  sorry

end probability_first_head_second_tail_l252_252783


namespace proof_ratio_l252_252542

variables {V : Type*} [inner_product_space ℝ V]
variables {A B C P Q : V}
variables {PA PB PC QA QB QC PQ AB : V}

noncomputable def given_conditions (P Q A B C : V) : Prop :=
  (PA + 2 • PB + 3 • PC = 0) ∧
  (2 • QA + 3 • QB + 5 • QC = 0)

noncomputable def target_statement (PQ AB : V) : Prop :=
  ∥PQ∥ / ∥AB∥ = 1/30
  
theorem proof_ratio (P Q A B C : V) (h : given_conditions P Q A B C) : target_statement (P - Q) (A - B) :=
sorry

end proof_ratio_l252_252542


namespace temperature_range_for_5_percent_deviation_l252_252378

noncomputable def approx_formula (C : ℝ) : ℝ := 2 * C + 30
noncomputable def exact_formula (C : ℝ) : ℝ := (9/5 : ℝ) * C + 32
noncomputable def deviation (C : ℝ) : ℝ := approx_formula C - exact_formula C
noncomputable def percentage_deviation (C : ℝ) : ℝ := abs (deviation C / exact_formula C)

theorem temperature_range_for_5_percent_deviation :
  ∀ (C : ℝ), 1 + 11 / 29 ≤ C ∧ C ≤ 32 + 8 / 11 ↔ percentage_deviation C ≤ 0.05 := sorry

end temperature_range_for_5_percent_deviation_l252_252378


namespace largest_collection_cardinality_l252_252290

-- Given conditions:
def n : ℕ := 2  -- n is an integer greater than 1.
def X : Type := Fin n  -- X is an n-element set.

-- A collection of subsets A_i of X is tight if the union is a proper subset of X,
-- and no element of X lies in exactly one of the A_i's.
def isTight (A : Finset (Finset X)) : Prop :=
  (⋃₀ A).card < n ∧ ∀ x ∈ X, (⋃₀ A).count x ≠ 1

-- Prove that the largest cardinality of a collection of proper non-empty subsets
-- of X, no non-empty subcollection of which is tight, is 2n - 2.
theorem largest_collection_cardinality : 
  ∃ (C : Finset (Finset X)), 
  (∀ C' ⊆ C, (C'.nonEmpty → ¬ isTight C')) ∧ 
  C.card = 2 * n - 2 :=
sorry

end largest_collection_cardinality_l252_252290


namespace range_of_f1_over_f4_l252_252923

noncomputable def positive_function_on_pos_reals := 
  {f : ℝ → ℝ // ∀ x ∈ Ioi 0, 0 < f x ∧ f x < deriv f x ∧ deriv f x < 2 * f x}

theorem range_of_f1_over_f4 (f : positive_function_on_pos_reals) :
  (1 / Real.exp 6) < f.1 1 / f.1 4 ∧ f.1 1 / f.1 4 < (1 / Real.exp 3) :=
sorry

end range_of_f1_over_f4_l252_252923


namespace not_odd_f_f_f_one_monotonic_decreasing_intervals_minimum_value_l252_252635

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x * (x + 4) else x * (x - 4)

theorem not_odd_f : ¬ (∀ x, f (-x) = -f x) :=
sorry

theorem f_f_one : f (f 1) = -3 :=
sorry

theorem monotonic_decreasing_intervals : 
  (∀ x, x ∈ Iic (-2) → ∀ y, y ∈ Iic (-2) → (x < y) → (f x > f y)) ∧
  (∀ x, x ∈ Ici 0 ∩ Iic 2 → ∀ y, y ∈ Ici 0 ∩ Iic 2 → (x < y) → (f x > f y)) :=
sorry

theorem minimum_value : ∀ x, f x ≥ -4 :=
sorry

end not_odd_f_f_f_one_monotonic_decreasing_intervals_minimum_value_l252_252635


namespace range_of_m_l252_252986

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x ≠ 0 → (y = (2 * m - 3) / x) ∈ { (x, y) | (x > 0 → y > 0) ∧ (x < 0 → y < 0) }) → m > 3 / 2 :=
sorry

end range_of_m_l252_252986


namespace field_ratio_l252_252362

theorem field_ratio (side pond_area_ratio : ℝ) (field_length : ℝ) 
  (pond_is_square: pond_area_ratio = 1/18) 
  (side_length: side = 8) 
  (field_len: field_length = 48) : 
  (field_length / (pond_area_ratio * side ^ 2 / side)) = 2 :=
by
  sorry

end field_ratio_l252_252362


namespace gcd_m_n_l252_252640

def m : ℕ := 3333333
def n : ℕ := 66666666

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end gcd_m_n_l252_252640


namespace average_side_lengths_l252_252702

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252702


namespace first_terms_b_geometric_seq_b_general_formula_a_Sn_inequality_l252_252222

def seq_a (n : ℕ) : ℚ
| 0       := 2
| (n + 1) := (4 * seq_a n - 2) / (3 * seq_a n - 1)

def seq_b (n : ℕ) : ℚ := (3 * seq_a n - 2) / (seq_a n - 1)

def S (n : ℕ) : ℚ := (finset.range n).sum (λ k, seq_a k)

theorem first_terms_b (b1 b2 b3 : ℚ) :
  seq_b 1 = 4 ∧ seq_b 2 = 8 ∧ seq_b 3 = 16 :=
sorry

theorem geometric_seq_b :
  ∀ n : ℕ, seq_b (n + 1) = 2 * seq_b n :=
sorry

theorem general_formula_a (n : ℕ) :
  seq_a n = (2 ^ (n + 1) - 2) / (2 ^ (n + 1) - 3) :=
sorry

theorem Sn_inequality (n : ℕ) :
  (↑(n + 1) * 2 ^ (n + 1) - n - 2) / (2 ^ (n + 1) - 1) < S n ∧
  S n ≤ (↑(n + 2) * 2 ^ (n - 1) - 1) / 2 ^ (n - 1) :=
sorry

end first_terms_b_geometric_seq_b_general_formula_a_Sn_inequality_l252_252222


namespace tangent_line_parallel_monotonicity_find_a_l252_252552

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := log x + a / x

-- Define the derivative f'(x)
def f_prime (x : ℝ) (a : ℝ) : ℝ := 1 / x - a / (x^2)

-- Prove Statement (I)
theorem tangent_line_parallel (a : ℝ) : (f_prime 1 a = 2) → (a = -1) :=
sorry

-- Prove Statement (II)
theorem monotonicity (a : ℝ) :
  ((∀ x > 1, (a <= 1) → (f_prime x a > 0)) ∧
  (∀ x > 1, (a > 1) → ((1 < x ∧ x < a) → (f_prime x a < 0)) ∧ ((x > a) → (f_prime x a > 0)))) :=
sorry

-- Prove Statement (III)
theorem find_a (a : ℝ) : (∃ x_0 > 1, f x_0 a <= a) → (a > 1) :=
sorry

end tangent_line_parallel_monotonicity_find_a_l252_252552


namespace part_one_part_two_part_three_l252_252554

def f (x a : ℝ) : ℝ := x^2 - ((a + 1) / a) * x + 1

theorem part_one (x : ℝ) :
  f x (1/2) ≤ 0 ↔ ((3 - Real.sqrt 5) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 5) / 2) :=
sorry

theorem part_two (a : ℝ) (ha : a > 0) :
  if h : a ≤ 1 then a < 1 / a 
  else a > 1 / a :=
sorry

theorem part_three (a x : ℝ) (ha : a > 0) :
  f x a ≤ 0 ↔ 
  if ha1 : a < 1 then a < x ∧ x < 1 / a 
  else if ha2 : a > 1 then 1 / a < x ∧ x < a 
  else x = 1 :=
sorry

end part_one_part_two_part_three_l252_252554


namespace systematic_sampling_20_l252_252330

theorem systematic_sampling_20 (n k : ℕ) (a b : ℕ) (N M : ℕ):
  n = 800 →
  k = 20 →
  a = 121 →
  b = 400 →
  N = 40 →
  M = (b - a + 1) / N →
  M = 7 :=
by
  intros h_n h_k h_a h_b h_N h_M
  rw [h_n, h_k, h_a, h_b, h_N, h_M]
  sorry

end systematic_sampling_20_l252_252330


namespace part_one_part_two_part_one_part_two_part_two_l252_252213

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 + a * x * (1 - Real.log x) - Real.log x

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x a = 3 / 2 ∧ (∀ y : ℝ, f y a ≥ 3 / 2) :=
sorry -- Proof to be filled in

theorem part_two_part_one (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
                   f.differentiable 1 (x1, a) ∧ f.differentiable 1 (x2, a) ∧ f.differentiable 1 (x3, a)) → 
  2 < a :=
sorry -- Proof to be filled in

theorem part_two_part_two (a : ℝ) (x1 x2 x3 : ℝ) 
  (h1 : x1 < x2) (h2 : x2 < x3) 
  (extreme_points : ∀ x, f'.x (a) = 0 ↔ (x = x1 ∨ x = x2 ∨ x = x3))
  (h3 : x1 * x3 = 1) : 
  x1 + x3 + 4 * x1 * x3 > 3 * a :=
sorry -- Proof to be filled in

end part_one_part_two_part_one_part_two_part_two_l252_252213


namespace parallel_lines_m_eq_neg7_l252_252969

noncomputable def line1 (m : ℝ) : ℝ → ℝ → Prop :=
λ x y, (3 + m) * x + 4 * y = 5 - 3 * m

noncomputable def line2 (m : ℝ) : ℝ → ℝ → Prop :=
λ x y, 2 * x + (5 + m) * y = 8

def lines_parallel (m : ℝ) : Prop :=
∀ (x y : ℝ), line1 m x y → line2 m x y → -((3 + m) / 4) = -(2 / (5 + m))

theorem parallel_lines_m_eq_neg7 : lines_parallel (-7) :=
sorry

end parallel_lines_m_eq_neg7_l252_252969


namespace average_spent_l252_252793

-- Define the constants used in the problem
def T : ℝ := 9.5 -- Transportation fee
def A : ℝ := 32.5 -- Admission ticket fee
def N : ℕ := 5 -- Number of people

-- Define the average calculation function
def average_spent_per_person : ℝ := (T + A) / N

-- State the theorem with the proof skipped
theorem average_spent : average_spent_per_person = 8.2 := 
by sorry

end average_spent_l252_252793


namespace smallest_positive_integer_k_l252_252497

theorem smallest_positive_integer_k {G : Type} (grid : G) [fintype G]
  (cell_color : G → ℕ) (k : ℕ) (C : ℕ) (n m : ℕ)
  (h1 : n = 100) (h2 : m = 100)
  (h3 : ∀ g, cell_color g ≤ 104)
  (h4 : ∃ k, (∀i ∈ finset.range n, ∃ j ∈ finset.range m, ∃ colors : finset ℕ, colors.card ≥ 3 ∧ (∀ c ∈ colors, ∃ g ∈ (finset.product (finset.range i) (finset.range j)), cell_color g = c)) ∨
               (∀i ∈ finset.range m, ∃ j ∈ finset.range n, ∃ colors : finset ℕ, colors.card ≥ 3 ∧ (∀ c ∈ colors, ∃ g ∈ (finset.product (finset.range i) (finset.range j)), cell_color g = c)))) :
  k = 12 :=
by
  sorry

end smallest_positive_integer_k_l252_252497


namespace mark_total_cost_is_correct_l252_252322

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end mark_total_cost_is_correct_l252_252322


namespace find_a_l252_252958

-- Defining the curve y in terms of x and a
def curve (x : ℝ) (a : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Defining the derivative of the curve
def derivative (x : ℝ) (a : ℝ) : ℝ := 4*x^3 + 2*a*x

-- The proof statement asserting the value of a
theorem find_a (a : ℝ) (h1 : derivative (-1) a = 8): a = -6 :=
by
  -- we assume here the necessary calculations and logical steps to prove the theorem
  sorry

end find_a_l252_252958


namespace part1_part2_l252_252514

noncomputable def f (x : ℝ) (a : ℝ) := (a + 2 * (Real.cos (x / 2)) ^ 2) * Real.cos (x + Real.pi / 2)

-- Prove that a = -1 given the conditions
theorem part1 (h1 : f (Real.pi / 2) a = 0) : a = -1 := sorry

-- Prove the value of cos(π/6 - 2α) given the conditions
theorem part2 (h1 : f (Real.pi / 2) a = 0) (h2 : f (α / 2) (-1) = -2 / 5) (h3 : α ∈ set.Ioo (Real.pi / 2) Real.pi) :
  Real.cos (Real.pi / 6 - 2 * α) = (-7 * Real.sqrt 3 - 24) / 50 := sorry

end part1_part2_l252_252514


namespace min_val_f_range_a_l252_252646

section proof_problem

def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Part 1: Prove the minimum value of f(x) is 3
theorem min_val_f : ∃ x0 : ℝ, ∀ x : ℝ, f(x0) ≤ f(x) ∧ f(x0) = 3 :=
sorry

-- Part 2: Prove the range of a for which g(a) ≤ f(x) ∀ x is 1 ≤ a ≤ 4
theorem range_a (a : ℝ) : (∀ x : ℝ, g(a) ≤ f(x)) ↔ (1 ≤ a ∧ a ≤ 4) :=
sorry

end proof_problem

end min_val_f_range_a_l252_252646


namespace reciprocal_of_fraction_sum_l252_252140

theorem reciprocal_of_fraction_sum : 
  (1 / (1 / 3 + 1 / 4 - 1 / 12)) = 2 := sorry

end reciprocal_of_fraction_sum_l252_252140


namespace largest_p_for_5_7_as_sum_of_consecutive_integers_l252_252154

noncomputable def largest_p (k : ℕ) : ℕ :=
  let q : ℕ := 7
  let x : ℕ := k ^ q
  let s : ℕ := 2 * x
  (s.factors.erase (2 * k ^ (q - 4)).erase 1).last

theorem largest_p_for_5_7_as_sum_of_consecutive_integers :
  largest_p 5 = 250 := by
  sorry

end largest_p_for_5_7_as_sum_of_consecutive_integers_l252_252154


namespace common_tangent_lines_count_l252_252186

def circle1 : Real → Real → Prop := λ x y, x^2 + y^2 = 4
def circle2 : Real → Real → Prop := λ x y, x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem common_tangent_lines_count :
  (∃ f : Real × Real → Real, ∀ p: Real × Real,
    circle1 p.1 p.2 ∧ circle2 p.1 p.2 → f p.1 p.2 = 0) → 
    2 :=
by
  sorry

end common_tangent_lines_count_l252_252186


namespace range_of_x_l252_252504

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : x^2 + a * x > 4 * x + a - 3 ↔ (x > 3 ∨ x < -1) := by
  sorry

end range_of_x_l252_252504


namespace angle_between_vectors_l252_252936

noncomputable def magnitude (v : Vector ℝ) : ℝ :=
  Real.sqrt $ v.dot_product v

noncomputable def is_perpendicular (u v : Vector ℝ) : Prop :=
  u.dot_product v = 0

theorem angle_between_vectors (a b : Vector ℝ)
  (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : magnitude a = Real.sqrt 2 * magnitude b)
  (h₄ : is_perpendicular (a + b) b) :
  ∃ θ, θ = 3 * Real.pi / 4 ∧ θ = Real.acos ((a.dot_product b) / (magnitude a * magnitude b)) :=
sorry

end angle_between_vectors_l252_252936


namespace incenter_is_intersection_of_angle_bisectors_l252_252791

theorem incenter_is_intersection_of_angle_bisectors {A B C : Type} [triangle : Triangle A B C] : 
  is_incenter (intersection (angle_bisector A B C)) :=
sorry

end incenter_is_intersection_of_angle_bisectors_l252_252791


namespace orthocenter_of_triangle_l252_252533

noncomputable theory

open scoped Classical

structure Circle (P : Type*) :=
(center : P) (radius : ℝ)

structure Point (P : Type*) := 
(coord : P)

variables {P : Type*} [MetricSpace P]

def diameter (O : Circle P) (A C : Point P) : Prop :=
dist A.coord C.coord = 2 * O.radius 

def arc_point (O : Circle P) (B C D : Point P) : Prop :=
/- D lies on the arc BC of circle O -/
sorry

def intersects_at_points (O1 O2 : Circle P) (A B : Point P) : Prop :=
/- O1 and O2 intersect at A and B -/
sorry

def intersects_circle (O : Circle P) (line_points : List (Point P)) (intersection_points : List (Point P)) : Prop :=
/- The line defined by line_points intersects circle O at intersection_points -/
sorry

def midpoint_on_line_segment (F : Point P) (HE BC : List (Point P)) : Prop :=
/- F is the midpoint of HE and lies on line segment BC -/
sorry

def is_orthocenter (H : Point P) (triangle_points : List (Point P)) : Prop :=
/- H is the orthocenter of the triangle defined by triangle_points -/
sorry

theorem orthocenter_of_triangle
  (O1 O2 : Circle P) (A B C D E H F : Point P)
  (h₁ : intersects_at_points O1 O2 A B)
  (h₂ : diameter O1 A C)
  (h₃ : arc_point O1 B C D)
  (h₄ : intersects_circle O2 [C, D] [E])
  (h₅ : intersects_circle O2 [A, D] [H])
  (h₆ : midpoint_on_line_segment F [H, E] [B, C]) :
  is_orthocenter H [A, C, E] :=
sorry

end orthocenter_of_triangle_l252_252533


namespace sum_of_remainders_l252_252458
noncomputable theory

-- Definitions
def is_odd_prime (p : ℕ) : Prop := p.prime ∧ p % 2 = 1

def remainder_mod (i p : ℕ) : ℕ := i^p % p^2

-- Lean 4 statement
theorem sum_of_remainders (p : ℕ) (hp : is_odd_prime p) :
  (∑ i in finset.range p \ {0}, remainder_mod i p) = p^2 * (p - 1) / 2 :=
by
  sorry

end sum_of_remainders_l252_252458


namespace find_general_term_arithmetic_sequence_l252_252957

-- Definitions needed
variable {a_n : ℕ → ℚ}
variable {S_n : ℕ → ℚ}

-- The main theorem to prove
theorem find_general_term_arithmetic_sequence 
  (h1 : a_n 4 - a_n 2 = 4)
  (h2 : S_n 3 = 9)
  (h3 : ∀ n : ℕ, S_n n = n / 2 * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) :
  (∀ n : ℕ, a_n n = 2 * n - 1) :=
by
  sorry

end find_general_term_arithmetic_sequence_l252_252957


namespace find_abcd_range_of_k_l252_252219

-- Definitions of the functions and points
def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b
def g (x : ℝ) (c d : ℝ) := Real.exp x * (c * x + d)
def P : ℝ × ℝ := (0, 2)

-- Conditions on the functions
axiom P_f : P.snd = f P.fst a b
axiom P_g : P.snd = g P.fst c d

-- Tangent line condition at point P(0,2)
axiom f_tangent : (fderiv ℝ (λ x, f x a b)) P.fst 1 = 4
axiom g_tangent : (fderiv ℝ (λ x, g x c d)) P.fst 1 = 4

-- Proof problem (1): Finding a, b, c, d
theorem find_abcd : 
  (f 0 a b = 2 ∧ fderiv ℝ (λ x, f x a b) 0 1 = 4) ∧
  (g 0 c d = 2 ∧ fderiv ℝ (λ x, g x c d) 0 1 = 4) →
  a = 4 ∧ b = 2 ∧ c = 2 ∧ d = 2 
  := sorry

-- Inequality problem (2): Range of k
theorem range_of_k (k : ℝ) : 
  (∀ x, x ≥ -2 → f x a b ≤ k * g x c d) ↔ (1 ≤ k ∧ k ≤ Real.exp 2) 
  := sorry

end find_abcd_range_of_k_l252_252219


namespace probability_same_number_l252_252117

def is_multiple (n factor : ℕ) : Prop :=
  ∃ k : ℕ, n = k * factor

def multiples_below (factor upper_limit : ℕ) : ℕ :=
  (upper_limit - 1) / factor

theorem probability_same_number :
  let upper_limit := 250
  let billy_factor := 20
  let bobbi_factor := 30
  let common_factor := 60
  let billy_multiples := multiples_below billy_factor upper_limit
  let bobbi_multiples := multiples_below bobbi_factor upper_limit
  let common_multiples := multiples_below common_factor upper_limit
  (common_multiples : ℚ) / (billy_multiples * bobbi_multiples) = 1 / 24 :=
by
  sorry

end probability_same_number_l252_252117


namespace tangent_line_through_P_l252_252918

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem tangent_line_through_P (x : ℝ) :
  let P := (-2 : ℝ, -2 : ℝ) in
  ∀ (l : ℝ → ℝ), (∃ a b : ℝ, l = λ x, a*x + b) →
  (∀ x, (l x = f x) → a = -9 ∧ b = 16) ∨ (l = λ x, -2) :=
by
  sorry

end tangent_line_through_P_l252_252918


namespace pencils_placed_l252_252145

noncomputable def total_pencils_faye_placed (rows : ℕ) (pattern_pencils_2nd : ℕ) (pattern_pencils_5th : ℕ) : ℕ :=
  let pattern_repeat := rows / 6
  in (pattern_pencils_2nd * pattern_repeat) + (pattern_pencils_5th * pattern_repeat)

theorem pencils_placed : total_pencils_faye_placed 30 76 114 = 950 := by
  sorry

end pencils_placed_l252_252145


namespace maximum_area_of_triangle_l252_252379

noncomputable def maximum_triangle_area (x : ℝ) (h1 : 30 * x > 0)
  (h2 : 12 + 30 * x > 31 * x) 
  (h3 : 30 * x + 31 * x > 12) 
  (h4 : 31 * x - 30 * x > 0) : ℝ :=
  let ABC_area_squared := 
    (61 / 16) * (61 * x^2 - 1) * (61^2 - 61 * x^2) in
  let AM_GM := (3720 : ℝ)^2 / 16 in
  max_area (sqrt AM_GM) = 930

theorem maximum_area_of_triangle : ∀ (x : ℝ),
  (30 * x > 0) →
  (12 + 30 * x > 31 * x) →
  (30 * x + 31 * x > 12) →
  (31 * x - 30 * x > 0) →
  maximum_triangle_area x (by assumption) (by assumption) (by assumption) (by assumption) = 930 :=
sorry

end maximum_area_of_triangle_l252_252379


namespace variance_of_ξ_l252_252965

noncomputable def probability_distribution (ξ : ℕ) : ℚ :=
  if ξ = 2 ∨ ξ = 4 ∨ ξ = 6 ∨ ξ = 8 ∨ ξ = 10 then 1/5 else 0

def expected_value (ξ_values : List ℕ) (prob : ℕ → ℚ) : ℚ :=
  ξ_values.map (λ ξ => ξ * prob ξ) |>.sum

def variance (ξ_values : List ℕ) (prob : ℕ → ℚ) (Eξ : ℚ) : ℚ :=
  ξ_values.map (λ ξ => prob ξ * (ξ - Eξ) ^ 2) |>.sum

theorem variance_of_ξ :
  let ξ_values := [2, 4, 6, 8, 10]
  let prob := probability_distribution
  let Eξ := expected_value ξ_values prob
  variance ξ_values prob Eξ = 8 :=
by
  -- Proof goes here
  sorry

end variance_of_ξ_l252_252965


namespace population_in_scientific_notation_l252_252105

noncomputable theory

-- Define the population in 2010
def population_yunnan_2010 : ℝ := 46000000

-- State the theorem for scientific notation
theorem population_in_scientific_notation : population_yunnan_2010 = 4.6 * 10^7 :=
by sorry

end population_in_scientific_notation_l252_252105


namespace probability_is_correct_l252_252243

noncomputable def probability_neither_prime_nor_composite : ℚ :=
  let numbers := (Finset.range 100).filter (λ n, n ≠ 0) in
  let neither_prime_nor_composite := numbers.filter (λ n, n = 1) in
  (neither_prime_nor_composite.card : ℚ) / (numbers.card : ℚ)

theorem probability_is_correct : probability_neither_prime_nor_composite = 1 / 99 := 
by
  sorry

end probability_is_correct_l252_252243


namespace hyperbola_eccentricity_range_l252_252938

theorem hyperbola_eccentricity_range
  (A B : ℝ × ℝ)
  (hA : A = (1, 2))
  (hB : B = (-1, 2))
  (P : ℝ × ℝ)
  (h_perpendicular : (P.1 - A.1) * (P.1 + B.1) + (P.2 - A.2) * (P.2 - B.2) = 0)
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h_asymptotes : ∀ (x : ℝ), (b / a) * x ≠ sqrt (1 - x^2) + 2 ∧ (b / a) * x ≠ -sqrt (1 - x^2) + 2) :
  1 < sqrt (1 + (b/a) ^ 2 - 1) / a ∧ sqrt (1 + (b/a) ^ 2 - 1) / a < 2 :=
by sorry

end hyperbola_eccentricity_range_l252_252938


namespace radius_of_circumcircle_of_triangle_l252_252767

theorem radius_of_circumcircle_of_triangle
  (R1 R2 : ℝ)
  (H1 : R1 + R2 = 12)
  (H2 : 4 * Real.sqrt 29 = Real.sqrt ((4 - 0) ^ 2 + (-4 - 0) ^ 2 + (R2 - R1) ^ 2))
  (radius_third_sphere : ℝ := 8) 
  (center_third_sphere_touches : ∀ A B C : Point, touches (Sphere.mk A 8) (Sphere.mk B 8) (Sphere.mk C 8)) :
  radius_of_circumcircle ABC = 4 * Real.sqrt 5 :=
sorry

end radius_of_circumcircle_of_triangle_l252_252767


namespace binomial_coeff_sum_l252_252909

theorem binomial_coeff_sum (n : ℕ) (h : 0 < n) :
  (∑ k in finset.range (n+1), nat.choose n k * (2 : ℕ)^k * nat.choose (n - k) ((n - k) / 2)) = nat.choose (2*n + 1) n :=
by sorry

end binomial_coeff_sum_l252_252909


namespace hyperbola_foci_asymptote_shared_l252_252358

noncomputable def hyperbola_equation (a b : ℝ) : String :=
  "y^2/" ++ a^2.repr ++ " - x^2/" ++ b^2.repr ++ " = 1"

theorem hyperbola_foci_asymptote_shared :
  let ellipse_foci := λ c : ℝ, Real.sqrt (49 - 24) = 5,
      hyperbola_asymptote := λ r : ℝ, r = 4 / 3,
      hyperbola_shared := hyperbola_equation 4 3
  in ellipse_foci 5 ∧ hyperbola_asymptote (4 / 3) ∧ hyperbola_shared = "y^2/16 - x^2/9 = 1" :=
by
  let ellipse_foci := Real.sqrt (49 - 24) = 5
  let hyperbola_asymptote := 4 / 3
  let hyperbola_shared := hyperbola_equation 4 3
  have h1 : ellipse_foci := by norm_num [ellipse_foci]
  have h2 : hyperbola_asymptote = 4 / 3 := by norm_num [hyperbola_asymptote]
  have h3 : hyperbola_shared = "y^2/16 - x^2/9 = 1" := by norm_num [hyperbola_shared]
  exact ⟨h1, h2, h3⟩

end hyperbola_foci_asymptote_shared_l252_252358


namespace initial_performers_count_l252_252870

theorem initial_performers_count (n : ℕ)
    (h1 : ∃ rows, 8 * rows = n)
    (h2 : ∃ (m : ℕ), n + 16 = m ∧ ∃ s, s * s = m)
    (h3 : ∃ (k : ℕ), n + 1 = k ∧ ∃ t, t * t = k) : 
    n = 48 := 
sorry

end initial_performers_count_l252_252870


namespace describe_production_steps_l252_252377

theorem describe_production_steps 
(
  (PFlowchart : Type)
  (description_tool : PFlowchart → Prop)
  (ProgramFlowchart : PFlowchart)
  (ProcessFlowchart : PFlowchart)
  (KnowledgeStructureDiagram : PFlowchart)
  (OrganizationalStructureDiagram : PFlowchart) 

  (H1 : description_tool ProgramFlowchart → "represents the process of completing events")
  (H2 : description_tool KnowledgeStructureDiagram → "is aimed at knowledge, not products")
  (H3 : description_tool OrganizationalStructureDiagram → "is aimed at units")
) 
: description_tool ProcessFlowchart → 
  "is used to describe the production steps of a certain product in a factory" 
:= 
sorry

end describe_production_steps_l252_252377


namespace parallel_lines_distance_l252_252970

open Real

theorem parallel_lines_distance (a c : ℝ) (hc : c > 0) :
    dist_lines (λ x y, x - y + 1) (λ x y, 3 * x + a * y - c) = sqrt 2 → (a - 3) / c = -2 := 
by
  sorry

end parallel_lines_distance_l252_252970


namespace average_side_length_of_squares_l252_252724

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252724


namespace paths_do_not_intersect_l252_252171

structure House := (id : Char)
structure School := (id : Char)

def house_a := House.mk 'A'
def house_b := House.mk 'B'
def house_c := House.mk 'C'
def house_d := House.mk 'D'

def school_a := School.mk 'A'
def school_b := School.mk 'B'
def school_c := School.mk 'C'
def school_d := School.mk 'D'

structure Path := (from : House) (to : School)

def paths := [ Path.mk house_a school_a,
               Path.mk house_b school_b,
               Path.mk house_c school_c,
               Path.mk house_d school_d ]

def valid_paths (p : List Path) : Prop :=
  -- Placeholder for actual path non-intersecting condition
  -- For now we assume the paths are valid if they map correctly from house to school
  List.All p (λ p, match p with
                   | Path.mk house school => house.id = school.id)

theorem paths_do_not_intersect : valid_paths paths :=
  by
    -- Proof to be filled in
    sorry

end paths_do_not_intersect_l252_252171


namespace real_part_of_fraction_correct_l252_252302

noncomputable def real_part_of_fraction (θ : ℝ) : ℝ :=
  let z := Complex.exp (Complex.I * θ) in
  (2 - Complex.re z) / (5 - 4 * Complex.re z)

theorem real_part_of_fraction_correct (θ : ℝ) : 
  real_part_of_fraction θ = (2 - Real.cos θ) / (5 - 4 * Real.cos θ) := 
by 
  sorry

end real_part_of_fraction_correct_l252_252302


namespace find_BD_l252_252333

-- Definitions for the conditions
def BC : ℝ := 2
def AC (c : ℝ) : ℝ := c
def AD : ℝ := 3

-- The hypothesis to be proven
theorem find_BD (c : ℝ) (hc : c ≥ √5) : 
  let AB := sqrt(c^2 + 4)
  let BD := sqrt(c^2 - 5)
  BD = sqrt(c^2 - 5) := 
by
  sorry

end find_BD_l252_252333


namespace doug_marbles_l252_252478

theorem doug_marbles (e_0 d_0 : ℕ) (h1 : e_0 = d_0 + 12) (h2 : e_0 - 20 = 17) : d_0 = 25 :=
by
  sorry

end doug_marbles_l252_252478


namespace find_PA_PB_sum_2sqrt6_l252_252257

noncomputable def polar_equation (ρ θ : ℝ) : Prop :=
  ρ - 2 * Real.cos θ - 6 * Real.sin θ + 1 / ρ = 0

noncomputable def parametric_line (t x y : ℝ) : Prop :=
  x = 3 + 1 / 2 * t ∧ y = 3 + Real.sqrt 3 / 2 * t

def point_P (x y : ℝ) : Prop :=
  x = 3 ∧ y = 3

theorem find_PA_PB_sum_2sqrt6 :
  (∃ ρ θ t₁ t₂, polar_equation ρ θ ∧ parametric_line t₁ 3 3 ∧ parametric_line t₂ 3 3 ∧
  point_P 3 3 ∧ |t₁| + |t₂| = 2 * Real.sqrt 6) := sorry

end find_PA_PB_sum_2sqrt6_l252_252257


namespace cos_C_value_sin_B_value_find_a_value_l252_252616

-- Given conditions
variables {α : Type*} [real_field α]
variables (A B C a b c : α)
axiom angle_sum : A + B + C = real.pi
axiom b_over_c : b / c = 2 * real.sqrt 3 / 3
axiom A_plus_3C : A + 3 * C = real.pi
axiom b_value : b = 3 * real.sqrt 3

/- 
Prove the following:
1. cos C = sqrt 3 / 3
2. sin B = 2 * sqrt 2 / 3
3. If b = 3 * sqrt 3, then a = 3 / 2
-/

noncomputable def cos_C := real.cos C
noncomputable def sin_B := real.sin B
noncomputable def side_a := a

-- Proof statements
theorem cos_C_value : cos_C α A B C a b c = real.sqrt 3 / 3 := sorry
theorem sin_B_value : sin_B α A B C a b c = 2 * real.sqrt 2 / 3 := sorry
theorem find_a_value : side_a α A B C a b c = 3 / 2 := sorry

end cos_C_value_sin_B_value_find_a_value_l252_252616


namespace value_of_f_2011_l252_252947

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_2011 (h_even : ∀ x : ℝ, f x = f (-x))
                       (h_sym : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f (2 + x) = f (2 - x))
                       (h_def : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f x = 2^x) : 
  f 2011 = 1 / 2 := 
sorry

end value_of_f_2011_l252_252947


namespace valid_domain_of_x_l252_252753

theorem valid_domain_of_x (x : ℝ) : 
  (x + 1 ≥ 0 ∧ x ≠ 0) ↔ (x ≥ -1 ∧ x ≠ 0) :=
by sorry

end valid_domain_of_x_l252_252753


namespace number_of_meetings_l252_252657

noncomputable def selena_radius : ℝ := 70
noncomputable def bashar_radius : ℝ := 80
noncomputable def selena_speed : ℝ := 200
noncomputable def bashar_speed : ℝ := 240
noncomputable def active_time_together : ℝ := 30

noncomputable def selena_circumference : ℝ := 2 * Real.pi * selena_radius
noncomputable def bashar_circumference : ℝ := 2 * Real.pi * bashar_radius

noncomputable def selena_angular_speed : ℝ := (selena_speed / selena_circumference) * (2 * Real.pi)
noncomputable def bashar_angular_speed : ℝ := (bashar_speed / bashar_circumference) * (2 * Real.pi)

noncomputable def relative_angular_speed : ℝ := selena_angular_speed + bashar_angular_speed
noncomputable def time_to_meet_once : ℝ := (2 * Real.pi) / relative_angular_speed

theorem number_of_meetings : Int := 
    ⌊active_time_together / time_to_meet_once⌋

example : number_of_meetings = 21 := by
  sorry

end number_of_meetings_l252_252657


namespace Multiple_of_Bs_age_10_years_ago_A_will_be_in_10_years_l252_252248

-- Define the ages of A and B
def B_age : ℕ := 48
def A_age : ℕ := B_age + 9

-- Define the ages of A and B in the past and future
def B_age_10_years_ago : ℕ := B_age - 10
def A_age_in_10_years : ℕ := A_age + 10

-- Define the multiple
def multiple : ℚ := (A_age_in_10_years : ℚ) / (B_age_10_years_ago : ℚ)

-- The theorem stating the required proof
theorem Multiple_of_Bs_age_10_years_ago_A_will_be_in_10_years :
  multiple ≈ 1.7632 := 
sorry

end Multiple_of_Bs_age_10_years_ago_A_will_be_in_10_years_l252_252248


namespace height_correctness_l252_252429

noncomputable def tray_height : ℕ :=
  let m := 7
  let n := 1
  m + n

theorem height_correctness : tray_height = 8 := by
  let w := 7 -- cut edge length
  let angle := 45 -- angle in degrees
  -- Given: height calculation process for the tray leading to height of 7
  have height_eq : w / Real.sqrt 2 = 7 := by
    calc
      w / Real.sqrt 2 = 7 : by sorry
  -- converting height in the form of √[n]{m}
  have eqn_form : Real.sqrt 7^2 = 49 := by
    calc
      Real.sqrt 7 ^ 2 = 7 ^ 2 : by sorry
      ... = 49 : by sorry
  -- leading to the final addition of m and n
  have add_eq : m + n = 7 + 1 := by sorry
  show tray_height = 8 from rfl

end height_correctness_l252_252429


namespace question_b_l252_252788

theorem question_b (a b c : ℝ) (h : c ≠ 0) (h_eq : a / c = b / c) : a = b := 
by
  sorry

end question_b_l252_252788


namespace average_price_of_5_baskets_l252_252803

/-- Saleem bought 4 baskets with an average cost of $4 each. --/
def average_cost_first_4_baskets : ℝ := 4

/-- Saleem buys the fifth basket with the price of $8. --/
def price_fifth_basket : ℝ := 8

/-- Prove that the average price of the 5 baskets is $4.80. --/
theorem average_price_of_5_baskets :
  (4 * average_cost_first_4_baskets + price_fifth_basket) / 5 = 4.80 := 
by
  sorry

end average_price_of_5_baskets_l252_252803


namespace average_side_lengths_l252_252705

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252705


namespace solution_l252_252484

theorem solution {a b : ℕ} (ha : 0 < a) (hb : 0 < b) :
  (b ^ 619 ∣ a ^ 1000 + 1) ∧ (a ^ 619 ∣ b ^ 1000 + 1) → (a = 1 ∧ b = 1) :=
begin
  sorry
end

end solution_l252_252484


namespace no_two_digit_number_divisible_l252_252471

theorem no_two_digit_number_divisible (a b : ℕ) (distinct : a ≠ b)
  (h₁ : 1 ≤ a ∧ a ≤ 9) (h₂ : 1 ≤ b ∧ b ≤ 9)
  : ¬ ∃ k : ℕ, (1 < k ∧ k ≤ 9) ∧ (10 * a + b = k * (10 * b + a)) :=
by
  sorry

end no_two_digit_number_divisible_l252_252471


namespace correct_option_b_l252_252057

theorem correct_option_b :
    (∀ x : ℤ, (x = 9) → (∀ y, abs (sqrt y) = sqrt y) → (∀ z : ℝ, z = sqrt 6 → (-z) ^ 2 = 6) 
    ∧ (∀ a b : ℝ, sqrt a * sqrt b = 4 * sqrt 6) → (∀ c d : ℝ, (c = sqrt 7) → (d = sqrt 2) → (c - d) ≠ sqrt 5)) → 
    (∀ k : ℝ, k = sqrt 6 → (-k)^2 = 6) :=
  sorry

end correct_option_b_l252_252057


namespace only_one_passes_prob_l252_252664

variable (P_A P_B P_C : ℚ)
variable (only_one_passes : ℚ)

def prob_A := 4 / 5 
def prob_B := 3 / 5
def prob_C := 7 / 10

def prob_only_A := prob_A * (1 - prob_B) * (1 - prob_C)
def prob_only_B := (1 - prob_A) * prob_B * (1 - prob_C)
def prob_only_C := (1 - prob_A) * (1 - prob_B) * prob_C

def prob_sum : ℚ := prob_only_A + prob_only_B + prob_only_C

theorem only_one_passes_prob : prob_sum = 47 / 250 := 
by sorry

end only_one_passes_prob_l252_252664


namespace AM_eq_AN_l252_252251

theorem AM_eq_AN 
  (ABC : Type) [triangle ABC]
  (acute_ABC : acute ABC)
  (BAC_ne_60 : ∠ ABC ≠ 60°)
  (tangent1 : tangent BD)
  (tangent2 : tangent CE)
  (BD_CE_eq_BC : BD = BC ∧ CE = BC)
  (DE_intersects_AB_AC : ∃ (F G : Type), DE_inter AB F ∧ DE_inter AC G)
  (CF_inter_BD : ∃ M : Type, line CF ∩ line BD = {M})
  (CE_inter_BG : ∃ N : Type, line CE ∩ line BG = {N}) :
  AM = AN :=
sorry

end AM_eq_AN_l252_252251


namespace mark_total_cost_is_correct_l252_252324

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end mark_total_cost_is_correct_l252_252324


namespace permutation_exists_l252_252336

-- Define the main problem statement
theorem permutation_exists (n : ℕ) (h : 0 < n) : ∃ σ : Fin n.succ → Fin n.succ, σ.Perm σ ∧ ∀ i j : Fin n.succ, i < j → (σ i + σ j) / 2 ∉ (σ '' { i | i < j ∧ σ i < (σ i + σ j) / 2 < σ j }) :=
sorry

end permutation_exists_l252_252336


namespace lowest_income_of_wealthiest_2000_l252_252992

theorem lowest_income_of_wealthiest_2000 (x : ℝ) : 
  (2 * 10^10 * x^(-3/2) = 2000) → x = 10^5 :=
begin
  sorry
end

end lowest_income_of_wealthiest_2000_l252_252992


namespace wheel_rpm_l252_252752

noncomputable def radius : ℝ := 35
noncomputable def speed_km_per_h : ℝ := 66
noncomputable def speed_cm_per_min := speed_km_per_h * 100000 / 60
noncomputable def circumference := 2 * Real.pi * radius
noncomputable def rpm := speed_cm_per_min / circumference

theorem wheel_rpm :
  rpm ≈ 500.23 := by
  sorry

end wheel_rpm_l252_252752


namespace sqrt_five_minus_one_half_gt_zero_point_five_l252_252137

theorem sqrt_five_minus_one_half_gt_zero_point_five : 
  ( (sqrt 5 - 1) / 2 ) > 0.5 :=
sorry

end sqrt_five_minus_one_half_gt_zero_point_five_l252_252137


namespace external_tangent_inequality_l252_252387

variable (x y z : ℝ)
variable (a b c T : ℝ)

-- Definitions based on conditions
def a_def : a = x + y := sorry
def b_def : b = y + z := sorry
def c_def : c = z + x := sorry
def T_def : T = π * x^2 + π * y^2 + π * z^2 := sorry

-- The theorem to prove
theorem external_tangent_inequality
    (a_def : a = x + y) 
    (b_def : b = y + z) 
    (c_def : c = z + x) 
    (T_def : T = π * x^2 + π * y^2 + π * z^2) : 
    π * (a + b + c) ^ 2 ≤ 12 * T := 
sorry

end external_tangent_inequality_l252_252387


namespace expand_product_l252_252142

theorem expand_product : ∀ x : ℝ, 3 * (x - 2) * (x^2 + 6) = 3 * x^3 - 6 * x^2 + 18 * x - 36 :=
by
  intro x
  calc
    3 * (x - 2) * (x^2 + 6)
        = (3 * x - 3 * 2) * (x^2 + 6) : by ring
    ... = (3 * x - 6) * (x^2 + 6) : by ring
    ... = 3 * x * (x^2 + 6) - 6 * (x^2 + 6) : by ring
    ... = 3 * x^3 + 18 * x - 6 * x^2 - 36 : by ring
    ... = 3 * x^3 - 6 * x^2 + 18 * x - 36 : by ring

end expand_product_l252_252142


namespace depth_of_water_in_smaller_container_l252_252836

theorem depth_of_water_in_smaller_container 
  (H_big : ℝ) (R_big : ℝ) (h_water : ℝ) 
  (H_small : ℝ) (R_small : ℝ) (expected_depth : ℝ) 
  (v_water_small : ℝ) 
  (v_water_big : ℝ) 
  (h_total_water : ℝ)
  (above_brim : ℝ) 
  (v_water_final : ℝ) : 

  H_big = 20 ∧ R_big = 6 ∧ h_water = 17 ∧ H_small = 18 ∧ R_small = 5 ∧ expected_depth = 2.88 ∧
  v_water_big = π * R_big^2 * H_big ∧ v_water_small = π * R_small^2 * H_small ∧ 
  h_total_water = π * R_big^2 * h_water ∧ above_brim = π * R_big^2 * (H_big - H_small) ∧ 
  v_water_final = above_brim →

  expected_depth = v_water_final / (π * R_small^2) :=
by
  intro h
  sorry

end depth_of_water_in_smaller_container_l252_252836


namespace find_trinomial_pairs_l252_252434

theorem find_trinomial_pairs : ∃ (p q r s : ℕ) (a b c : ℝ),
  q < p ∧ s < r ∧ 
  (p + r = 15 ∧
  ((a = 0 ∧ b ≠ 0 ∧ s + p = 15 ∧ r + q = 15) ∨ 
   (b = 0 ∧ a ≠ 0 ∧ t + p = 15 ∧ s + q = 15) ∨
   (a ≠ 0 ∧ b ≠ 0 ∧ c + p = 15 ∧ s + q = 15 ∧ t + s = q ∧ p = 2 * q)))
  sorry

end find_trinomial_pairs_l252_252434


namespace sum_of_largest_and_smallest_four_digit_numbers_is_11990_l252_252169

theorem sum_of_largest_and_smallest_four_digit_numbers_is_11990 (A B C D : ℕ) 
    (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D)
    (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
    (h_eq : 1001 * A + 110 * B + 110 * C + 1001 * D = 11990) :
    (min (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 1999) ∧
    (max (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 9991) :=
by
  sorry

end sum_of_largest_and_smallest_four_digit_numbers_is_11990_l252_252169


namespace distance_AC_l252_252134

theorem distance_AC (t_Eddy t_Freddy : ℕ) (d_AB : ℝ) (speed_ratio : ℝ) : 
  t_Eddy = 3 ∧ t_Freddy = 4 ∧ d_AB = 510 ∧ speed_ratio = 2.2666666666666666 → 
  ∃ d_AC : ℝ, d_AC = 300 :=
by 
  intros h
  obtain ⟨hE, hF, hD, hR⟩ := h
  -- Declare velocities
  let v_Eddy : ℝ := d_AB / t_Eddy
  let v_Freddy : ℝ := v_Eddy / speed_ratio
  let d_AC : ℝ := v_Freddy * t_Freddy
  -- Prove the distance
  use d_AC
  sorry

end distance_AC_l252_252134


namespace positive_difference_of_solutions_l252_252896

theorem positive_difference_of_solutions {x : ℝ} (h : (5 - x^2 / 4)^(1/3) = -3) : 
  let sqrt128 := real.sqrt 128 in
  |sqrt128 - (-sqrt128)| = 16 * real.sqrt 2 :=
by
  let sqrt128 := real.sqrt 128
  have h1 : 5 - x^2 / 4 = -27 := sorry
  have h2 : x^2 = 128 := sorry
  have h3 : sqrt128 = real.sqrt 128 := by sorry
  have h4 : |sqrt128 + sqrt128| = 2 * sqrt128 := by sorry
  rw [h4]
  rw [sqrt128]
  sorry

end positive_difference_of_solutions_l252_252896


namespace infinite_series_sum_l252_252461

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 5) ^ (n + 1)) = 5 / 16 :=
sorry

end infinite_series_sum_l252_252461


namespace inverse_function_point_l252_252924

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.LeftInverse f f⁻¹) (h_point : f 2 = -1) : f⁻¹ (-1) = 2 :=
by
  sorry

end inverse_function_point_l252_252924


namespace first_player_has_winning_strategy_l252_252414

-- Define the conditions and theorem statement
theorem first_player_has_winning_strategy (n : ℕ) (h_n : n ≥ 5)
  (C_polyhedron : convex_polyhedron)
  (H_three_edges : ∀ v : vertex (C_polyhedron), edges_meet_at_vertex v = 3)
  (players_alternate_turns : players_take_turns_writing_names_on_faces)
  (goal_win_condition : goal_is_common_vertex_faces_write_names) :
  ∃ winning_strategy : strategy, 
    first_player_has_winning_strategy winning_strategy :=
sorry

end first_player_has_winning_strategy_l252_252414


namespace find_value_of_expression_l252_252008

theorem find_value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : (12 * y - 5)^2 = 161 :=
sorry

end find_value_of_expression_l252_252008


namespace percent_of_rs_600_l252_252061

theorem percent_of_rs_600 : (600 * 0.25 = 150) :=
by
  sorry

end percent_of_rs_600_l252_252061


namespace average_side_length_of_squares_l252_252713

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252713


namespace find_pairs_l252_252483

theorem find_pairs (n p : ℕ) (hp : Prime p) (hnp : n ≤ 2 * p) (hdiv : (p - 1) * n + 1 % n^(p-1) = 0) :
  (n = 1 ∧ Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end find_pairs_l252_252483


namespace positive_difference_equation_l252_252897

noncomputable def positive_difference_solutions : ℝ :=
  let eq1 := (5 - (x : ℝ)^2 / 4)^(1 / 3) = -3
  let eq2 := x^2 = 128
  16 * Real.sqrt 2

theorem positive_difference_equation :
  (5 - x^2 / 4)^(1 / 3) = -3 → x = 8 * Real.sqrt 2 - (-8 * Real.sqrt 2) :=
by
  intro h
  sorry

end positive_difference_equation_l252_252897


namespace average_side_lengths_l252_252707

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252707


namespace distribute_5_cousins_in_4_rooms_l252_252329

theorem distribute_5_cousins_in_4_rooms : 
  let rooms := 4
  let cousins := 5
  ∃ ways : ℕ, ways = 67 ∧ rooms = 4 ∧ cousins = 5 := sorry

end distribute_5_cousins_in_4_rooms_l252_252329


namespace average_side_length_of_squares_l252_252715

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252715


namespace restore_triangle_ABC_l252_252269

-- let I be the incenter of triangle ABC
variable (I : Point)
-- let Ic be the C-excenter of triangle ABC
variable (I_c : Point)
-- let H be the foot of the altitude from vertex C to side AB
variable (H : Point)

-- Claim: Given I, I_c, H, we can recover the original triangle ABC
theorem restore_triangle_ABC (I I_c H : Point) : ExistsTriangleABC :=
sorry

end restore_triangle_ABC_l252_252269


namespace man_l252_252395

noncomputable def man_saves (S : ℝ) : ℝ :=
0.20 * S

noncomputable def initial_expenses (S : ℝ) : ℝ :=
0.80 * S

noncomputable def new_expenses (S : ℝ) : ℝ :=
1.10 * (0.80 * S)

noncomputable def said_savings (S : ℝ) : ℝ :=
S - new_expenses S

theorem man's_monthly_salary (S : ℝ) (h : said_savings S = 500) : S = 4166.67 :=
by
  sorry

end man_l252_252395


namespace xiao_ming_proposition_false_l252_252058

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m * m ≤ n → m = 1 ∨ m = n → m ∣ n

def check_xiao_ming_proposition : Prop :=
  ∃ n : ℕ, ∃ (k : ℕ), k < n → ∃ (p q : ℕ), p = q → n^2 - n + 11 = p * q ∧ p > 1 ∧ q > 1

theorem xiao_ming_proposition_false : ¬ (∀ n: ℕ, is_prime (n^2 - n + 11)) :=
by
  sorry

end xiao_ming_proposition_false_l252_252058


namespace average_side_length_of_squares_l252_252709

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252709


namespace percent_to_decimal_l252_252802

theorem percent_to_decimal : (2 : ℝ) / 100 = 0.02 :=
by
  -- Proof would go here
  sorry

end percent_to_decimal_l252_252802


namespace truth_values_of_p_and_q_l252_252263

theorem truth_values_of_p_and_q (p q : Prop) (h1 : p ∨ q) (h2 : ¬(p ∧ q)) (h3 : ¬p) : ¬p ∧ q :=
by
  sorry

end truth_values_of_p_and_q_l252_252263


namespace graphs_relative_position_and_intersection_l252_252460

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 5

theorem graphs_relative_position_and_intersection :
  (1 > -1.5) ∧ ( ∃ y, f 0 = y ∧ g 0 = y ) ∧ f 0 = 5 :=
by
  -- sorry to skip the proof
  sorry

end graphs_relative_position_and_intersection_l252_252460


namespace length_of_first_train_l252_252770

theorem length_of_first_train (speed_train_1 speed_train_2 : ℝ) (length_train_2 time : ℝ) : 
  speed_train_1 = 80 ∧ speed_train_2 = 65 ∧ length_train_2 = 153 ∧ time = 6.802214443534172 
  → let relative_speed := (speed_train_1 + speed_train_2) * (1000 / 3600) in
    let distance := relative_speed * time in
    let length_train_1 := distance - length_train_2 in
    length_train_1 ≈ 121 := -- "≈" denotes approximation
sorry

end length_of_first_train_l252_252770


namespace b_7_equals_41_l252_252295

-- Define the sequence and its properties
def b (n : ℕ) : ℤ :=
  match n with
  | 0     => 0
  | n + 1 => if n = 0 then 2 else
              have bm : ℤ := b n.quotient₂,
              have bn : ℤ := b n.remainder₂,
              bm + bn + n.quotient₂ * n.remainder₂ + 1

-- State that the value of b_7 equals 41 given the defined properties
theorem b_7_equals_41 : b 7 = 41 := by
  sorry

end b_7_equals_41_l252_252295


namespace average_side_length_of_squares_l252_252722

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252722


namespace income_analysis_l252_252954

noncomputable def median (xs : List ℝ) : ℝ :=
if List.length xs % 2 = 0 then
  ((xs.get (List.length xs / 2 - 1)) + (xs.get (List.length xs / 2))) / 2
else
  xs.get (List.length xs / 2)

noncomputable def mean (xs : List ℝ) : ℝ :=
(xs.sum) / (xs.length)

noncomputable def variance (xs : List ℝ) : ℝ :=
let μ := mean xs
(xs.map (λ x => (x - μ) ^ 2)).sum / xs.length

theorem income_analysis (n : ℕ) (h : n ≥ 3) (xs : List ℝ) (x_{n+1} : ℝ)
  (h_len : xs.length = n)
  (median_old : ℝ := median xs)
  (mean_old : ℝ := mean xs)
  (variance_old : ℝ := variance xs)
  (xs_new := (x_{n+1} :: xs).sort (≤)) :
  mean xs_new > mean_old ∧
  (median xs_new = median_old ∨ median xs_new ≠ median_old) ∧
  variance xs_new > variance_old :=
by
  -- proof
  sorry

end income_analysis_l252_252954


namespace fixed_point_intersection_l252_252644

-- Definitions and conditions
def ellipse_passes_through_point (a b : ℝ) (h : a > b ∧ b > 0): Prop :=
  ∃ x y : ℝ, (x = 0 ∧ y = sqrt 3 ∧ (x^2 / a^2 + y^2 / b^2 = 1))

def eccentricity_is_half (a b : ℝ) (h : a > b ∧ b > 0) : Prop := 
  ∃ c : ℝ, (c = sqrt (a^2 - b^2) ∧ c / a = 1 / 2)

def projections_intersect_fixed (a b : ℝ) (h : a > b ∧ b > 0) : Prop := 
  ∃ F A B D E : ℝ × ℝ, 
  (F = (a / 2, 0) ∧ 
   ∃ slope : ℝ, 
     A = (A.1, slope * (A.1 - F.1) + F.2) ∧ 
     B = (B.1, slope * (B.1 - F.1) + F.2) ∧ 
     D.1 = 4 ∧ E.1 = 4 ∧ 
     (Projl (A, F, B)).intersect (Projr (D, K, E)) = (5 / 2, 0))

-- Main statement 
theorem fixed_point_intersection (a b : ℝ) (h : a > b ∧ b > 0) :
  ellipse_passes_through_point a b h ∧
  eccentricity_is_half a b h →
  (a = 2) ∧
  (b = sqrt 3) ∧
  projections_intersect_fixed a b h := by
  sorry

end fixed_point_intersection_l252_252644


namespace smallest_positive_value_is_C_l252_252130

variable (sqrt7 : ℝ)
variable (sqrt17 : ℝ)
variable (sqrt34 : ℝ)

axiom sqrt7_value : sqrt7 ≈ 2.646
axiom sqrt17_value : sqrt17 ≈ 4.123
axiom sqrt34_value : sqrt34 ≈ 5.831

def optionA := 12 - 4 * sqrt7
def optionB := 4 * sqrt7 - 12
def optionC := 25 - 6 * sqrt17
def optionD := 64 - 15 * sqrt34
def optionE := 15 * sqrt34 - 64

theorem smallest_positive_value_is_C :
  optionC > 0 ∧ 
  (optionA ≤ 0 ∨ optionA > optionC) ∧ 
  (optionB ≤ 0 ∨ optionB > optionC) ∧ 
  (optionD ≤ 0 ∨ optionD > optionC) ∧ 
  (optionE ≤ 0 ∨ optionE > optionC) 
  := 
sorry

end smallest_positive_value_is_C_l252_252130


namespace principal_amount_l252_252044

theorem principal_amount (P : ℝ) (r : ℝ) (t : ℕ) (I : ℝ) :
  r = 0.12 ∧ t = 3 ∧ I = 5888 ∧ P * (1 + r / 12)^(t * 12) = P + (P - I) →
  P ≈ 10254.63 :=
by sorry

end principal_amount_l252_252044


namespace find_x_l252_252884

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l252_252884


namespace average_of_side_lengths_of_squares_l252_252692

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252692


namespace probability_at_least_7_stay_l252_252665

-- Defining the conditions
def friends : ℕ := 8
def unsure_friends : ℕ := 5
def certain_friends : ℕ := 3
def unsure_probability : ℚ := 3 / 7
def certain_probability : ℚ := 1

-- Statement of the problem: Proving the probability that at least 7 friends stay
theorem probability_at_least_7_stay : (binomial_dist 7 8 5 (3/7)) = 1539 / 16807 := by
  sorry

end probability_at_least_7_stay_l252_252665


namespace peter_twice_as_old_in_years_l252_252606

def mother_age : ℕ := 60
def harriet_current_age : ℕ := 13
def peter_current_age : ℕ := mother_age / 2
def years_later : ℕ := 4

theorem peter_twice_as_old_in_years : 
  peter_current_age + years_later = 2 * (harriet_current_age + years_later) :=
by
  -- using given conditions 
  -- Peter's current age is 30
  -- Harriet's current age is 13
  -- years_later is 4
  sorry

end peter_twice_as_old_in_years_l252_252606


namespace value_of_expression_l252_252782

theorem value_of_expression (x y : ℕ) (h₁ : x = 12) (h₂ : y = 7) : (x - y) * (x + y) = 95 := by
  -- Here we assume all necessary conditions as given:
  -- x = 12 and y = 7
  -- and we prove that (x - y)(x + y) = 95
  sorry

end value_of_expression_l252_252782


namespace average_of_side_lengths_of_squares_l252_252694

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252694


namespace ratio_Sandoval_Hawkins_l252_252604

-- Definitions based on the conditions provided in the problem
variables {S H SL : ℕ}

-- Governor Sandoval gave S addresses
variable (Sandoval_addresses : S > 0)

-- Governor Hawkins gave half as many as Governor Sandoval
variable (Hawkins_addresses : H = S / 2)

-- Governor Sloan gave ten more than Governor Sandoval
variable (Sloan_addresses : SL = S + 10)

-- The three of them gave 40 commencement addresses together
variable (total_addresses : S + H + SL = 40)

-- To prove: The ratio of the number of commencement addresses given by Governor Sandoval to the number given by Governor Hawkins is 2:1
theorem ratio_Sandoval_Hawkins : (S : ℚ) / H = 2 :=
begin
  -- This is just the statement part, so we use "sorry" to skip the proof
  sorry
end

end ratio_Sandoval_Hawkins_l252_252604


namespace solve_equation_l252_252662

noncomputable def ctg (x : ℝ) : ℝ := (cos x) / (sin x)

theorem solve_equation (x : ℝ) (k : ℤ) :
  2 * log 3 (ctg x) = log 2 (cos x) →
  x = π / 3 + 2 * π * (k : ℝ) :=
sorry

end solve_equation_l252_252662


namespace tetrahedron_volume_eq_5sqrt3_l252_252250

def Point := (ℝ × ℝ × ℝ)

noncomputable def distance (p q : Point) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

-- Assume the vertices of the tetrahedron are A, B, C, D
variables {A B C D : Point}

axiom ab_eq_5 : distance A B = 5
axiom ac_eq_5 : distance A C = 5
axiom ad_eq_5 : distance A D = 5
axiom bc_eq_3 : distance B C = 3
axiom cd_eq_4 : distance C D = 4
axiom bd_eq_5 : distance B D = 5

theorem tetrahedron_volume_eq_5sqrt3 : 
  -- Assume volume function uses vertices
  volume_of_tetrahedron A B C D = 5 * real.sqrt 3 :=
sorry

end tetrahedron_volume_eq_5sqrt3_l252_252250


namespace evaluate_expression_l252_252023

def f (x : ℕ) : ℕ :=
  match x with
  | 3 => 10
  | 4 => 17
  | 5 => 26
  | 6 => 37
  | 7 => 50
  | _ => 0  -- for any x not in the table, f(x) is undefined and defaults to 0

def f_inv (y : ℕ) : ℕ :=
  match y with
  | 10 => 3
  | 17 => 4
  | 26 => 5
  | 37 => 6
  | 50 => 7
  | _ => 0  -- for any y not in the table, f_inv(y) is undefined and defaults to 0

theorem evaluate_expression :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 :=
by
  sorry

end evaluate_expression_l252_252023


namespace greatest_integer_function_example_l252_252450

theorem greatest_integer_function_example : 
  let x := -3.7 + 1.5 in
  Int.floor x = -3 :=
by
  let x := -3.7 + 1.5
  sorry

end greatest_integer_function_example_l252_252450


namespace measure_of_angle_AGC_is_60_l252_252255

-- Definitions and conditions
variables (A B C D E F G : Type)
variables [RegularHexagon ABCDEF]
variables [IntersectsAt AC AE G]
variables [InteriorAngleRegularHexagon ABCDEF = 120]

-- Goal statement
theorem measure_of_angle_AGC_is_60 :
  (angle AGC = 60) :=
sorry

end measure_of_angle_AGC_is_60_l252_252255


namespace expenses_each_month_l252_252470
noncomputable def total_expenses (worked_hours1 worked_hours2 worked_hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (total_left : ℕ) : ℕ :=
  (worked_hours1 * rate1) + (worked_hours2 * rate2) + (worked_hours3 * rate3) - total_left

theorem expenses_each_month (hours1 : ℕ)
  (hours2 : ℕ)
  (hours3 : ℕ)
  (rate1 : ℕ)
  (rate2 : ℕ)
  (rate3 : ℕ)
  (left_over : ℕ) :
  hours1 = 20 → 
  rate1 = 10 →
  hours2 = 30 →
  rate2 = 20 →
  hours3 = 5 →
  rate3 = 40 →
  left_over = 500 → 
  total_expenses hours1 hours2 hours3 rate1 rate2 rate3 left_over = 500 := by
  intros h1 r1 h2 r2 h3 r3 l
  sorry

end expenses_each_month_l252_252470


namespace remainder_when_divided_by_x_minus_17_x_minus_53_remainder_when_divided_by_x_minus_17_x_minus_53_correct_l252_252292

def Q : ℝ → ℝ := sorry

theorem remainder_when_divided_by_x_minus_17_x_minus_53 :
  (∀ x, Q x = (x - 17) * (x - 53) * (some_R x) + (-x + 70)) :=
by
  -- Conditions
  assume Q_cond_17 : Q 17 = 53,
  assume Q_cond_53 : Q 53 = 17,
  sorry

-- You can also express it directly:
theorem remainder_when_divided_by_x_minus_17_x_minus_53_correct :
  (∀ (Q : ℝ → ℝ), (Q 17 = 53) → (Q 53 = 17) → (∀ x, Q x = (x - 17) * (x - 53) * (some_R x) + (-x + 70))) :=
by
  assume Q : ℝ → ℝ,
  assume Q_cond_17 : Q 17 = 53,
  assume Q_cond_53 : Q 53 = 17,
  sorry

end remainder_when_divided_by_x_minus_17_x_minus_53_remainder_when_divided_by_x_minus_17_x_minus_53_correct_l252_252292


namespace average_side_length_of_squares_l252_252725

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252725


namespace area_GHJ_eq_1_div_36_area_ABC_l252_252633

-- Definitions from the problem conditions
variables {A B C D F G H J : Point} 
variables (triangle_ABC : Triangle A B C)
variables (AD_median : Median A D)
variables (BF_median : Median B F)
variable (G_centroid : IsCentroid triangle_ABC G)
variable (H_mid_AB : IsMidpoint A B H)
variable (J_mid_GH : IsMidpoint G H J)

-- The equivalence to prove
theorem area_GHJ_eq_1_div_36_area_ABC (k : ℝ) :
  area (triangle G H J) = (1 / 36) * area (triangle A B C) := 
  sorry

end area_GHJ_eq_1_div_36_area_ABC_l252_252633


namespace dorothy_and_jemma_sales_l252_252472

theorem dorothy_and_jemma_sales :
  ∀ (frames_sold_by_jemma price_per_frame_jemma : ℕ)
  (price_per_frame_dorothy frames_sold_by_dorothy : ℚ)
  (total_sales_jemma total_sales_dorothy total_sales : ℚ),
  price_per_frame_jemma = 5 →
  frames_sold_by_jemma = 400 →
  price_per_frame_dorothy = price_per_frame_jemma / 2 →
  frames_sold_by_jemma = 2 * frames_sold_by_dorothy →
  total_sales_jemma = frames_sold_by_jemma * price_per_frame_jemma →
  total_sales_dorothy = frames_sold_by_dorothy * price_per_frame_dorothy →
  total_sales = total_sales_jemma + total_sales_dorothy →
  total_sales = 2500 := by
  sorry

end dorothy_and_jemma_sales_l252_252472


namespace proposition_A_necessary_not_sufficient_for_B_l252_252982

theorem proposition_A_necessary_not_sufficient_for_B (x y : ℤ) :
  (x + y ≠ 5) → (x ≠ 2 ∨ y ≠ 3) ∧ ¬((x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5)) :=
begin
  sorry
end

end proposition_A_necessary_not_sufficient_for_B_l252_252982


namespace motorcyclist_average_speed_l252_252423

theorem motorcyclist_average_speed (a : ℝ) (h₁ : 0 < a) : 
  let t₁ := a / 60,
      t₂ := a / 120,
      t_total := t₁ + t₂,
      v_avg := a / t_total
  in v_avg = 40 :=
by 
  -- assumptions from the problem
  let t₁ := a / 60,
      t₂ := a / 120,
      t_total := t₁ + t₂,
      v_avg := a / t_total
  
  -- calculation steps provided in the solution
  have h_t1 : t₁ = a / 60, from rfl,
  have h_t2 : t₂ = a / 120, from rfl,
  have h_t_total : t_total = t₁ + t₂, from rfl,
  have h_t_total_eq : t_total = a / 40, 
  { calc t_total = a / 60 + a / 120 : by rw [h_t1, h_t2]
               ... = (2 * a) / 120 + a / 120 : by rw [←div_eq_mul_one_div, mul_div_assoc, div_add, add_comm]
               ... = (3 * a) / 120 : by linarith
               ... = a / 40 : by norm_num },

  show v_avg = 40, by
  { calc v_avg = a / t_total : by trivial
             ... = a / (a / 40) : by rw [h_t_total_eq]
             ... = 40 : by field_simp [h₁] }

end motorcyclist_average_speed_l252_252423


namespace difference_of_fractions_l252_252858

theorem difference_of_fractions (x y : ℝ) (h1 : x = 497) (h2 : y = 325) :
  (2/5) * (3 * x + 7 * y) - (3/5) * (x * y) = -95408.6 := by
  rw [h1, h2]
  sorry

end difference_of_fractions_l252_252858


namespace polyhedron_zero_weighted_sum_l252_252009

variables {R : Type*} [LinearOrderedField R] [AddCommGroup R]
variables {V : Type*} [AddCommGroup V] [Module R V]

open_locale big_operators

structure polyhedron (V : Type*) [AddCommGroup V] [Module R V] :=
(vertices : list V)
(interior_point : V)
(inside : ∃ (x : V), x ∈ convex_hull R (vertices) ∧ inner x (interior_point) = 0)

theorem polyhedron_zero_weighted_sum {V : Type*} [AddCommGroup V] [Module R V] 
  (P : polyhedron V) :
  ∃ (x : V), x ∈ convex_hull R (P.vertices) :=
sorry

end polyhedron_zero_weighted_sum_l252_252009


namespace children_grouped_correctly_l252_252170

structure SiblingGroup where
  children : List String
  numBrothers : Nat

def Family1 := ["Alice", "Bětka", "Cyril"]
def Family2 := ["David", "Erika", "Filip", "Gábina"]
def Family3 := ["Hugo", "Iveta"]
def Family4 := ["Jan", "Karel", "Libor"]

def numberOfBrothers (children : List String) (child : String) : Nat :=
  list.length (children.erase child)

def Group1 := SiblingGroup.mk ["Cyril", "Hugo"] 0
def Group2 := SiblingGroup.mk ["Alice", "Bětka", "David", "Filip", "Iveta"] 1
def Group3 := SiblingGroup.mk ["Erika", "Gábina", "Jan", "Karel", "Libor"] 2

theorem children_grouped_correctly :
  let families := [Family1, Family2, Family3, Family4]
  let all_children := Family1 ++ Family2 ++ Family3 ++ Family4
  ( ∃ (groups : List SiblingGroup),
    groups = [Group1, Group2, Group3] ∧
    ∀ group ∈ groups, ∀ child ∈ group.children, 
      numberOfBrothers all_children child = group.numBrothers) :=
begin
  let families := [Family1, Family2, Family3, Family4],
  let all_children := Family1 ++ Family2 ++ Family3 ++ Family4,
  existsi [Group1, Group2, Group3],
  split,
  { refl, }, -- list of groups matches exactly
  { intros group group_in_groups child child_in_group,
    cases group_in_groups;
    cases group_in_groups;
    cases group_in_groups;
    finish },
end

end children_grouped_correctly_l252_252170


namespace tan_of_XYZ_l252_252272

noncomputable def tan_X : ℝ :=
  let XZ: ℝ := real.sqrt 17
  let YZ: ℝ := 4
  let XY: ℝ := real.sqrt (XZ^2 - YZ^2)
  XY = 1 -- This will be simplified in the Lean proof as sqrt(17 - 16) = 1
  in YZ / XY

theorem tan_of_XYZ (XZ YZ : ℝ) (hYZ : YZ = 4) (hXZ : XZ = real.sqrt 17) : 
  (∃ XY, XY = real.sqrt (XZ^2 - YZ^2) ∧ XY = 1) ∧ tan_X = 4 :=
by {
  sorry
}

end tan_of_XYZ_l252_252272


namespace train_speed_l252_252099

/-- 
Given:
- Length of train L is 390 meters (0.39 km)
- Speed of man Vm is 2 km/h
- Time to cross man T is 52 seconds

Prove:
- The speed of the train Vt is 25 km/h
--/
theorem train_speed 
  (L : ℝ) (Vm : ℝ) (T : ℝ) (Vt : ℝ)
  (h1 : L = 0.39) 
  (h2 : Vm = 2) 
  (h3 : T = 52 / 3600) 
  (h4 : Vt + Vm = L / T) :
  Vt = 25 :=
by sorry

end train_speed_l252_252099


namespace ratio_of_numbers_l252_252018

theorem ratio_of_numbers (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h₃ : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l252_252018


namespace john_scores_are_92_79_75_74_65_l252_252282

/-- Given John's scores and conditions, prove that his scores sorted from greatest to least are 92, 79, 75, 74, and 65. -/
theorem john_scores_are_92_79_75_74_65 :
  ∃ (x y : ℕ), (x, y ≠ 74 ∧ y ≠ 65 ∧ y ≠ 79 ∧ x ≠ y ∧ x < 95 ∧ y < 95 ∧ x + y = 167)
    ∧ (5 ∣ (74 + 65 + 79 + x + y)) ∧ (5 * 77 = 74 + 65 + 79 + x + y)
    ∧ list.sort (≥) [74, 65, 79, x, y] = [92, 79, 75, 74, 65] := sorry

end john_scores_are_92_79_75_74_65_l252_252282


namespace total_tourists_proof_l252_252002

noncomputable def calculate_total_tourists : ℕ :=
  let start_time := 8  
  let end_time := 17   -- 5 PM in 24-hour format
  let initial_tourists := 120
  let increment := 2
  let number_of_trips := end_time - start_time  -- total number of trips including both start and end
  let first_term := initial_tourists
  let last_term := initial_tourists + increment * (number_of_trips - 1)
  (number_of_trips * (first_term + last_term)) / 2

theorem total_tourists_proof : calculate_total_tourists = 1290 := by
  sorry

end total_tourists_proof_l252_252002


namespace no_integer_solutions_l252_252626

theorem no_integer_solutions (w l : ℕ) (hw_pos : 0 < w) (hl_pos : 0 < l) : 
  (w * l = 24 ∧ (w = l ∨ 2 * l = w)) → false :=
by 
  sorry

end no_integer_solutions_l252_252626


namespace derivative_solution_l252_252297

theorem derivative_solution {x_0 : real} (h : (λ x : real, x * real.log x)' x_0 = 2) : x_0 = real.exp 1 :=
by
  sorry

end derivative_solution_l252_252297


namespace num_of_denominators_l252_252667

def is_digit (n : ℕ) : Prop := n ≤ 9

def is_valid_abc (a b c : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ (a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

theorem num_of_denominators (a b c : ℕ) (h : is_valid_abc a b c) :
  ∃ (denoms : Finset ℕ), denoms.toFinset = {3, 9, 27, 37, 111, 333, 999} ∧ denoms.card = 7 :=
  sorry

end num_of_denominators_l252_252667


namespace third_smallest_three_digit_palindromic_prime_l252_252162

-- Definitions of palindromic prime and three-digit numbers
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_prime (n : ℕ) : Prop := nat.prime n

def is_palindromic_prime (n : ℕ) : Prop :=
  is_palindrome n ∧ is_prime n

-- Lean 4 statement.
theorem third_smallest_three_digit_palindromic_prime :
  ∃ n : ℕ, is_palindromic_prime n ∧ 101 < n ∧ 131 < n ∧
  (n = 101 ∨ n = 131 ∨ n = 151) ∧
  ∀ m : ℕ, is_palindromic_prime m → 101 < m → m < n → m ≠ 131 :=
sorry

end third_smallest_three_digit_palindromic_prime_l252_252162


namespace find_a_l252_252194

theorem find_a (a : ℕ) (h_pos : a > 0) (h_quadrant : 2 - a > 0) : a = 1 := by
  sorry

end find_a_l252_252194


namespace average_side_lengths_l252_252677

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252677


namespace length_DF_l252_252256

theorem length_DF (EF : ℝ) (cos_E : ℝ) (DF : ℝ) 
  (h1 : EF = 13)
  (h2 : cos_E = (12 * real.sqrt 169) / 169)
  (h3 : cos_E = DF / EF) : 
  DF = 12 :=
by
  sorry

end length_DF_l252_252256


namespace max_abc_l252_252188

theorem max_abc {a b c : ℝ} (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) :
  abc = 3

end max_abc_l252_252188


namespace larger_square_side_length_and_area_of_R_l252_252835

def square_side_length : ℕ := 2
def rectangle_width : ℕ := 2
def rectangle_height : ℕ := 4

theorem larger_square_side_length_and_area_of_R :
  ∃ s : ℕ, ∃ area_R : ℕ,
  (s = 4) ∧ (area_R = 4) ∧
  (s * s = square_side_length * square_side_length + rectangle_width * rectangle_height + area_R) :=
begin
  use 4,
  use 4,
  split,
  { refl, },
  split,
  { refl, },
  calc 
    (4 * 4) = 16 : by norm_num
    ... = (2 * 2) + (2 * 4) + 4 : by norm_num,
end

end larger_square_side_length_and_area_of_R_l252_252835


namespace dot_product_of_a_and_b_l252_252973

noncomputable def vector_a (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
a

noncomputable def vector_b (a b : ℝ × ℝ) (h1 : a + b = (1, -3)) (h2 : a - b = (3, 7)) : ℝ × ℝ := 
b

theorem dot_product_of_a_and_b {a b : ℝ × ℝ} 
  (h1 : a + b = (1, -3)) 
  (h2 : a - b = (3, 7)) : 
  (a.1 * b.1 + a.2 * b.2) = -12 := 
sorry

end dot_product_of_a_and_b_l252_252973


namespace find_m_l252_252576

theorem find_m (m : ℝ) :
  log 2 (m^2 - 3*m - 3) + complex.I * log 2 (m - 2) = complex.I * log 2 (m - 2) →
  m = 4 :=
by sorry

end find_m_l252_252576


namespace num_pure_Gala_trees_l252_252087

-- Define the problem statement conditions
variables (T F G H : ℝ)
variables (c1 : 0.125 * F + 0.075 * F + F = 315)
variables (c2 : F = (2 / 3) * T)
variables (c3 : H = (1 / 6) * T)
variables (c4 : T = F + G + H)

-- Prove the number of pure Gala trees G is 66
theorem num_pure_Gala_trees : G = 66 :=
by
  -- Proof will be filled out here
  sorry

end num_pure_Gala_trees_l252_252087


namespace suzy_final_books_l252_252049

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l252_252049


namespace minimum_area_triangle_OAB_l252_252524

theorem minimum_area_triangle_OAB 
  (P : Point) (A B : Point)
  (hP : P = (2,1))
  (hA : A = (a, 0))
  (hB : B = (0, b))
  (h_line_PAB : ∃ m b, (∀ x, y = m * x + b) ∧ 
                       (P.2 = m * P.1 + b) ∧ 
                       (A.2 = m * A.1 + b) ∧ 
                       (B.1 = 0 ∧ B.2 = b)) :
  area (△ O A B) = 2 := 
  sorry

end minimum_area_triangle_OAB_l252_252524


namespace rectangle_area_unchanged_l252_252675

theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) : 
  0.8 * l * 1.25 * w = 432 := 
by {
  -- The proof goes here
  sorry
}

end rectangle_area_unchanged_l252_252675


namespace equivalence_of_statements_l252_252614

-- Define the points and conditions
variables (A B C P Q S D T H K M N : Type)
variables [PointsOnLine A C P] [PointsOnLine B C Q]
variables [PointsOnLine A B S] [PointsOnLine A B D] [PointsOnLine A B T]
variables [LinesIntersect A Q B P H] [LinesIntersect P T Q S K]
variables [LinesIntersect A Q P S M] [LinesIntersect P B Q T N]

-- Define the three statements
def collinear_HKD := collinear H K D
def concurrent_AN_CD_BM := concurrent (lineThrough A N) (lineThrough C D) (lineThrough B M)
def concurrent_PS_CD_QT := concurrent (lineThrough P S) (lineThrough C D) (lineThrough Q T)

-- Prove the equivalence of the three statements
theorem equivalence_of_statements :
  (collinear_HKD ↔ concurrent_AN_CD_BM) ∧ (collinear_HKD ↔ concurrent_PS_CD_QT) :=
sorry

end equivalence_of_statements_l252_252614


namespace quadratic_equation_factored_form_correct_l252_252107

theorem quadratic_equation_factored_form_correct :
  ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intros x h
  sorry

end quadratic_equation_factored_form_correct_l252_252107


namespace math_problem_proof_l252_252299

variable (f : ℝ → ℝ)

theorem math_problem_proof :
  (∀ x y : ℝ, f(x + y) = f(x) + f(y)) →
  f (-1) = 2 →
  (∀ x > 0, f x < 0) →
  (f 0 = 0 ∧ f 2 = -4 ∧ ∀ k : ℝ, (∀ t : ℝ, f (t^2 + 3 * t) + f (t + k) ≤ 4) → k ≥ 2) :=
by
  intros h_add h_f_neg1 h_fx_pos
  have h_f_0 : f 0 = 0 := sorry
  have h_f_odd : ∀ x, f (-x) = -f x := sorry
  have h_f_2 : f 2 = -4 := sorry
  have h_k : ∀ k : ℝ, (∀ t : ℝ, f (t^2 + 3*t) + f (t + k) ≤ 4) → k ≥ 2 := sorry
  exact ⟨h_f_0, h_f_2, h_k⟩ 

end math_problem_proof_l252_252299


namespace second_day_speed_l252_252812

theorem second_day_speed
  (initial_speed : ℝ)
  (distance : ℝ)
  (time_late : ℝ)
  (time_early : ℝ)
  (second_day_speed : ℝ) :
  initial_speed = 3 → 
  distance = 1.5 → 
  time_late = 7 / 60 → 
  time_early = 8 / 60 →
  second_day_speed = distance / (distance / initial_speed - time_late - time_early) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
  (distance : ℝ) / (distance / initial_speed - time_late - time_early) = 
  (1.5 : ℝ) / (1.5 / 3 - 7 / 60 - 8 / 60) : by rw [h1, h2, h3, h4]
  ... = 6 : by norm_num

end second_day_speed_l252_252812


namespace circumcircle_radius_l252_252769

def R1 := 2 * Real.sqrt 27 + 6
def R2 := 12 - R1
def d := 4 * Real.sqrt 29
def rA := 8

theorem circumcircle_radius (ABC : Triangle)
  (B C : Point)
  (s1 s2 s3 : Sphere)
  (h_ABC_bc : (s1.center, B) ∈ Plane(ABC) ∧ (s2.center, C) ∈ Plane(ABC))
  (h_opposite_sides : s1.center.z > 0 ∧ s2.center.z < 0)
  (h_radii_sum : s1.r + s2.r = 12)
  (h_centers_dist : dist s1.center s2.center = 4 * Real.sqrt 29)
  (h_sphere_A : s3.center = A ∧ s3.r = 8 ∧ s3.externally_touches s1 ∧ s3.externally_touches s2) :
  radius_circumcircle ABC = 4 * Real.sqrt 5 :=
sorry

end circumcircle_radius_l252_252769


namespace part1_part2_l252_252215

noncomputable def f (x a : ℝ) : ℝ := Real.log x + (1 / 2) * x^2 - a * x + a

def is_monotonically_increasing (f: ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, (0 < x ∧ 0 < y ∧ x < y) → f x ≤ f y

theorem part1 (a : ℝ) :
  is_monotonically_increasing (λ x, f x a) a → a ≤ 2 :=
sorry

theorem part2 (a x1 x2 : ℝ) (h_a : a ≥ Real.exp(1/2) + Real.exp(-1/2)) (h_x1x2 : x2 > x1)
(h_fx_eq_0: (Real.log x1 + (1/2) * x1^2 - a * x1 + a = 0 ∧ Real.log x2 + (1/2) * x2^2 - a * x2 + a = 0)) :
  f x2 a - f x1 a = 1 - Real.exp(1) / 2 + 1 / (2 * Real.exp(1)) :=
sorry

end part1_part2_l252_252215


namespace commodity_prices_2008_l252_252804

theorem commodity_prices_2008:
  (∀ n: ℕ, price_X_in_year (2001 + n) = 5.20 + 0.45 * n) ∧
  (∀ n: ℕ, price_Y_in_year (2001 + n) = 7.30 + 0.20 * n) ∧
  (price_X_in_year 2001 = 5.20) ∧
  (price_Y_in_year 2001 = 7.30) →
  (∀ n: ℕ, (price_X_in_year (2008) = price_Y_in_year (2008) - 0.35)) :=
begin
  sorry
end

end commodity_prices_2008_l252_252804


namespace count_valid_triples_l252_252444

theorem count_valid_triples :
  ∃! (a c : ℕ), a ≤ 101 ∧ 101 ≤ c ∧ a * c = 101^2 :=
sorry

end count_valid_triples_l252_252444


namespace find_c_from_direction_vector_l252_252421

theorem find_c_from_direction_vector :
  ∀ (c : ℝ),
    let p1 := (⟨-3, 0⟩ : ℝ × ℝ),
    let p2 := (⟨0, 3⟩ : ℝ × ℝ),
    let direction := ⟨p2.1 - p1.1, p2.2 - p1.2⟩,
    ∃ (d : ℝ × ℝ), 
      d = ⟨3, c⟩ ∧
      d = direction →
      c = 3 := 
by
  intro c
  let p1 := (⟨-3, 0⟩ : ℝ × ℝ)
  let p2 := (⟨0, 3⟩ : ℝ × ℝ)
  let direction := ⟨p2.1 - p1.1, p2.2 - p1.2⟩
  use direction
  sorry

end find_c_from_direction_vector_l252_252421


namespace sum_series_eq_one_fourth_l252_252847

noncomputable def series_sum (s : ℕ → ℝ) : ℝ := ∑' n, s n

theorem sum_series_eq_one_fourth :
  series_sum (λ n : ℕ, 3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
by
  sorry

end sum_series_eq_one_fourth_l252_252847


namespace perfect_squares_difference_of_squares_l252_252571

theorem perfect_squares_difference_of_squares : 
  ∃ (n : ℕ), n = 70 ∧ ∀ (a : ℕ), a^2 < 20000 ∧ 
  (∃ (b : ℕ), a^2 = (b+2)^2 - b^2) ↔ a^2 = 4 * (nat.succ b) ∧ nat.succ b = d^2 → n = 70 :=
by
  sorry

end perfect_squares_difference_of_squares_l252_252571


namespace correct_statement_l252_252790

-- Let A denote the statement: "The diagonals of a square are equal and bisect each other."
def statement_A := ∀ (s : Type) [square s], diagonals_equal s ∧ bisect_each_other s

-- Let B denote the statement: "A quadrilateral with supplementary angles is a parallelogram."
def statement_B := ∀ (q : Type) [quadrilateral_with_supplementary_angles q], parallelogram q

-- Let C denote the statement: "The diagonals of a rectangle are perpendicular to each other."
def statement_C := ∀ (r : Type) [rectangle r], diagonals_perpendicular r

-- Let D denote the statement: "A quadrilateral with adjacent sides equal is a rhombus."
def statement_D := ∀ (q : Type) [quadrilateral_with_adjacent_sides_equal q], rhombus q

-- The main theorem to prove is that statement A is the correct one.
theorem correct_statement : statement_A ∧ ¬statement_B ∧ ¬statement_C ∧ ¬statement_D :=
by
  sorry

end correct_statement_l252_252790


namespace hillary_climbing_rate_l252_252566

theorem hillary_climbing_rate :
  ∀ (H : ℝ),
  (∀ (t : ℝ), t = 6 → 
   (500 * t = 3000) ∧ 
   (5000 - 1000 = 4000) → 
   ((4000 - 3000 = 1000) ∧ 
   (1000 / 1000 = 1)) → 
   (t - 1 = 5) → 
   (4000 / (t - 1) = 800)) → 
  H = 800 :=
by
  intros H t ht hEddyClimb hSummit hHillaryDescend hHillaryClimb hHillaryRate
  sorry

end hillary_climbing_rate_l252_252566


namespace solutionToSystemOfEquations_solutionToSystemOfInequalities_l252_252404

open Classical

noncomputable def solveSystemOfEquations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solutionToSystemOfEquations : ∃ (x y : ℝ), solveSystemOfEquations x y ∧ x = 4 ∧ y = 5 := by
  sorry

def solveSystemOfInequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solutionToSystemOfInequalities : ∃ x : ℝ, solveSystemOfInequalities x ∧ 1 ≤ x ∧ x < 2 := by
  sorry

end solutionToSystemOfEquations_solutionToSystemOfInequalities_l252_252404


namespace average_side_lengths_l252_252706

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252706


namespace third_smallest_palindromic_prime_l252_252164

-- Define the properties of being a three-digit palindromic prime
def is_three_digit_palindromic_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ prime n ∧ (n / 100 = n % 10)

-- Assert that 101 and 131 are the first and second smallest three-digit palindromic primes
axiom palindromic_prime_101 : is_three_digit_palindromic_prime 101
axiom palindromic_prime_131 : is_three_digit_palindromic_prime 131

-- Define the proposition that 151 is the third smallest three-digit palindromic prime
theorem third_smallest_palindromic_prime : is_three_digit_palindromic_prime 151 :=
by
  sorry


end third_smallest_palindromic_prime_l252_252164


namespace polygon_X_perimeter_l252_252457

def is_half_hexagon (P : set Point) (H : Hexagon) : Prop := 
  -- A placeholder definition to indicate that P forms one half of hexagon H by symmetrical split
  sorry

def side_length (H : Hexagon) : ℝ :=
  sorry

theorem polygon_X_perimeter (P : set Point) (H : Hexagon)
  (h_half_hexagon : is_half_hexagon P H)
  (h_side_length : side_length H = 5) :
  perimeter P = 15 :=
sorry

end polygon_X_perimeter_l252_252457


namespace proof_problem_l252_252202

noncomputable def radius_of_sphere (r : ℝ) : Prop :=
  let h := 10 in let d := 10 in let r_cylinder := d / 2 in 
    (4 * π * r^2 = 2 * π * r_cylinder * h) ∧
    r = 5

noncomputable def volume_of_sphere (v : ℝ) : Prop :=
  let r_sphere := 5 in
    v = (4 / 3) * π * r_sphere^3 ∧
    v = (500 / 3) * π

noncomputable def volume_of_cylinder (v : ℝ) : Prop :=
  let h := 10 in let r_cylinder := 5 in
    v = π * r_cylinder^2 * h ∧
    v = 250 * π

noncomputable def volume_difference (v_diff : ℝ) : Prop :=
  let v_sphere:= (500 / 3) * π in let v_cylinder := 250 * π in
    v_diff = v_cylinder - v_sphere ∧
    v_diff = (250 / 3) * π

theorem proof_problem : ∃ r v_s v_c v_diff, 
  radius_of_sphere r ∧ 
  volume_of_sphere v_s ∧ 
  volume_of_cylinder v_c ∧ 
  volume_difference v_diff := by
sorry

end proof_problem_l252_252202


namespace sum_of_three_divisible_by_three_l252_252655

open Finset 

theorem sum_of_three_divisible_by_three (S : Finset ℕ) (h : S.card = 7) :
  ∃ a b c ∈ S, (a + b + c) % 3 = 0 :=
by
  sorry

end sum_of_three_divisible_by_three_l252_252655


namespace speed_ratio_l252_252405

-- Definitions of the conditions in the problem
variables (v_A v_B : ℝ) -- speeds of A and B

-- Condition 1: positions after 3 minutes are equidistant from O
def equidistant_3min : Prop := 3 * v_A = |(-300 + 3 * v_B)|

-- Condition 2: positions after 12 minutes are equidistant from O
def equidistant_12min : Prop := 12 * v_A = |(-300 + 12 * v_B)|

-- Statement to prove
theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_12min v_A v_B) :
  v_A / v_B = 4 / 5 := sorry

end speed_ratio_l252_252405


namespace average_side_length_of_squares_l252_252710

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252710


namespace number_of_domino_arrangements_l252_252805

/-- 
Given:
1. A set of 15 dominoes where each domino represents a pair (i, j) with 0 ≤ i ≤ j ≤ 4.
2. Arrangements from left to right and right to left are considered different.

Prove:
The number of ways to arrange these 15 dominoes in a single line according to the usual rules of the game is 126760.
-/
theorem number_of_domino_arrangements : 
  let dominoes := {pair : ℕ × ℕ // pair.1 ≤ pair.2 ∧ pair.1 ≤ 4 ∧ pair.2 ≤ 4},
      arrangements := {arrangement : list dominoes // arrangement.length = 15 ∧ is_valid_arrangement arrangement}
  in finset.card arrangements = 126760 := sorry

end number_of_domino_arrangements_l252_252805


namespace concurrency_of_lines_l252_252733

-- Definitions of the given conditions
def tangent_circles (k1 k2 : Circle) (C : Point) (O1 O2 : Point) : Prop :=
  (k1.center = O1) ∧ (k2.center = O2) ∧
  externally_tangent k1 k2 C

def common_tangent (l : Line) (k1 k2 : Circle) (C : Point) : Prop :=
  is_tangent l k1 C ∧ is_tangent l k2 C

def externally_tangent_to_2 (k : Circle) (k1 k2 : Circle) : Prop :=
  externally_tangent k k1 (tangent_point k k1) ∧ externally_tangent k k2 (tangent_point k k2)

def is_diameter_perpendicular_to (A B : Point) (k : Circle) (l : Line) : Prop :=
  diameter k A B ∧ perpendicular l (line_through A B)

def same_side (O1 A : Point) (l : Line) : Prop :=
  same_side_of O1 A l

-- Now state the main theorem using these definitions.
theorem concurrency_of_lines
  (k1 k2 k : Circle)
  (O1 O2 O A B C : Point)
  (l : Line)
  (h1 : tangent_circles k1 k2 C O1 O2)
  (h2 : externally_tangent_to_2 k k1 k2)
  (h3 : common_tangent l k1 k2 C)
  (h4 : is_diameter_perpendicular_to A B k l)
  (h5 : same_side O1 A l) :
  concurrent (line_through A O2) (line_through B O1) l :=
sorry -- Proof to be filled in later.

end concurrency_of_lines_l252_252733


namespace find_value_of_a_perpendicular_lines_l252_252564

theorem find_value_of_a_perpendicular_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x - 2 → y = 2 * x + 1 → 
  (a * 2 = -1)) → a = -1/2 :=
by
  sorry

end find_value_of_a_perpendicular_lines_l252_252564


namespace relation_between_x_and_y_l252_252267

noncomputable def radius_of_circle : ℝ := a
noncomputable def side_of_equilateral_triangle : ℝ := a * sqrt 3
noncomputable def distance_from_E_to_tangent : ℝ := x
noncomputable def distance_from_E_to_AB : ℝ := y

theorem relation_between_x_and_y
  (a x : ℝ)
  (AE_DC_equal : distance_from_E_to_tangent = distance_from_E_to_AB)
  : y^2 = (x^3) / (a * sqrt 3 - x) 
:= sorry

end relation_between_x_and_y_l252_252267


namespace no_common_complex_roots_l252_252848

theorem no_common_complex_roots (a b : ℚ) :
  ¬ ∃ α : ℂ, (α^5 - α - 1 = 0) ∧ (α^2 + a * α + b = 0) :=
sorry

end no_common_complex_roots_l252_252848


namespace average_side_length_of_squares_l252_252727

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252727


namespace collinear_A2_B2_C2_l252_252929

variable (A B C : Point) (l : Line)
variable (A1 B1 C1 : Point) -- Midpoints defined with respect to line l
variable (A2 B2 C2 : Point) -- Intersection points

-- Definitions for the midpoints A1, B1, C1 
def is_midpoint (P Q R : Point) (M : Point) : Prop :=
  ∃ (X : Line), X.contains_points [P, Q] ∧ X.intersects R ∧ X.intersects M ∧ (dist M P = dist M Q)

-- Definitions for the intersections A2, B2, C2
def is_intersection (L1 L2 : Line) (P : Point) : Prop :=
  L1.contains P ∧ L2.contains P

-- The collinearity claim
theorem collinear_A2_B2_C2
  (hA1 : is_midpoint A l A1)
  (hB1 : is_midpoint B l B1)
  (hC1 : is_midpoint C l C1)
  (hA2 : is_intersection (Line.mk A A1) (Line.mk B C) A2)
  (hB2 : is_intersection (Line.mk B B1) (Line.mk A C) B2)
  (hC2 : is_intersection (Line.mk C C1) (Line.mk A B) C2) :
  collinear [A2, B2, C2] :=
sorry

end collinear_A2_B2_C2_l252_252929


namespace closest_whole_number_of_expression_l252_252453

theorem closest_whole_number_of_expression :
  let expr := (10 ^ 3000 + 10 ^ 3004) / (10 ^ 3002 + 10 ^ 3002)
  abs (expr - 50) < abs (expr - (50 - 1)) ∧ abs (expr - 50) < abs (expr - (50 + 1)) :=
by
  let expr := (10 ^ 3000 + 10 ^ 3004) / (10 ^ 3002 + 10 ^ 3002)
  have h1 : expr = 50.005, sorry
  have h2 : abs (50.005 - 50) < abs (50.005 - 49), sorry
  have h3 : abs (50.005 - 50) < abs (50.005 - 51), sorry
  exact ⟨h2, h3⟩

end closest_whole_number_of_expression_l252_252453


namespace sum_of_binom_solutions_l252_252037

theorem sum_of_binom_solutions :
  (∀ n : ℕ, (n = 14 ∨ n = 16) → (nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16)) →
  (14 + 16 = 30) :=
by
  intros h
  sorry

end sum_of_binom_solutions_l252_252037


namespace triangle_inequality_part_a_l252_252799

theorem triangle_inequality_part_a (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  a^2 + b^2 + c^2 + a * b * c < 8 :=
sorry

end triangle_inequality_part_a_l252_252799


namespace A_eq_B_l252_252877

variables (F : Type) [field F] (one_nonzero : (1 : F) + 1 ≠ 0)

def P (α : F) : F × F :=
  ( (α^2 - 1) / (α^2 + 1), 2 * α / (α^2 + 1) )

def A : set (F × F) :=
  { (β, γ) | β^2 + γ^2 = 1 }

def B : set (F × F) :=
  { (1, 0) } ∪ { P α | α ∈ F ∧ α^2 ≠ -1 }

-- Statement of the theorem
theorem A_eq_B : A F = B F one_nonzero :=
  sorry

end A_eq_B_l252_252877


namespace enclosed_area_l252_252451

theorem enclosed_area (x y : ℝ) : (|2 * x| + |2 * y| = 12) → 72 :=
by sorry

end enclosed_area_l252_252451


namespace weight_of_mixture_is_correct_l252_252371

noncomputable def weight_of_mixture (weight_a: ℝ) (weight_b: ℝ) (density_a_20: ℝ) 
                                    (density_b_20: ℝ) (density_decrement: ℝ) 
                                    (temp_increment: ℝ) (mixture_ratio: ℚ) 
                                    (total_volume: ℝ) (temp_mixture: ℝ) : ℝ :=
  let temp_change := (temp_mixture - 20) / temp_increment
  let new_density_a := density_a_20 - (density_decrement * temp_change)
  let new_density_b := density_b_20 - (density_decrement * temp_change)
  let volume_a := total_volume * (mixture_ratio / (mixture_ratio + 2))
  let volume_b := total_volume * (2 / (mixture_ratio + 2))
  let weight_a_mixture := volume_a * new_density_a
  let weight_b_mixture := volume_b * new_density_b
  (weight_a_mixture + weight_b_mixture) / 1000   -- converting to kg

theorem weight_of_mixture_is_correct:
  weight_of_mixture 900 850 0.9 0.85 0.02 5 (3 / 2) 4000 30 = 3.36 :=
by
  calc weight_of_mixture 900 850 0.9 0.85 0.02 5 (3 / 2) 4000 30
      = _ : sorry

end weight_of_mixture_is_correct_l252_252371


namespace smallest_period_f_l252_252867

noncomputable def f (x : ℝ) := Real.cos (x / 2) * (Real.sin (x / 2) - Real.sqrt 3 * Real.cos (x / 2))

theorem smallest_period_f : Real.periodic f (2 * Real.pi) :=
sorry

end smallest_period_f_l252_252867


namespace books_at_end_l252_252046

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l252_252046


namespace circle_second_x_intercept_l252_252079

/-- Given a circle with endpoints (0,0) and (12,5) as the diameter endpoints.
Prove that the x-coordinate of the second point where the circle intersects the x-axis is 12. -/
theorem circle_second_x_intercept :
  ∃ x₀ x₁ y₀ y₁ : ℝ, x₀ = 0 ∧ y₀ = 0 ∧ x₁ = 12 ∧ y₁ = 5 ∧
  let h := (x₀ + x₁) / 2,
      k := (y₀ + y₁) / 2,
      r := real.sqrt ((x₁ - h)^2 + (y₁ - k)^2)
  in (x - h)^2 + (y - k)^2 = r^2 → x = 12 ∨ x = 0 → x = 12 :=
begin
  sorry
end

end circle_second_x_intercept_l252_252079


namespace profit_difference_l252_252800

noncomputable def business_diff : ℕ :=
  let a := 8000
  let b := 10000
  let c := 12000
  let b_profit := 3500
  let total_ratio := 4 + 5 + 6
  let total_profit := b_profit * total_ratio / 5
  let a_profit := 4 * total_profit / total_ratio
  let c_profit := 6 * total_profit / total_ratio
  c_profit - a_profit

theorem profit_difference (a b c b_profit : ℕ) (ha : a = 8000) (hb : b = 10000) (hc : c = 12000) (hb_profit : b_profit = 3500)
  : business_diff = 1400 := by
  rw [ha, hb, hc, hb_profit]
  unfold business_diff
  sorry

end profit_difference_l252_252800


namespace average_of_side_lengths_of_squares_l252_252698

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252698


namespace range_of_b_l252_252242

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem range_of_b (b : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (f x1 = b) ∧ (f x2 = b) ∧ (f x3 = b))
  ↔ (-4 / 3 < b ∧ b < 28 / 3) :=
by
  sorry

end range_of_b_l252_252242


namespace sum_of_squared_distances_is_constant_l252_252638

theorem sum_of_squared_distances_is_constant
  (O : Point) (r : ℝ) (n : ℕ) [fact (0 < n)]
  (P : Point) (P_i : fin n → Point)
  (h0 : is_regular_ngon n O r P_i) (h1 : on_circle P O r) :
  (∑ i, ∥P - P_i i∥^2 = 2 * n * r^2) :=
by
  sorry

end sum_of_squared_distances_is_constant_l252_252638


namespace negation_prop_equiv_l252_252748

variable (a : ℝ)

theorem negation_prop_equiv :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2 * a * x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2 * a * x - 1 ≥ 0) :=
sorry

end negation_prop_equiv_l252_252748


namespace polya_enumeration_theorem_l252_252068

noncomputable def P (xs : Fin 6 → ℕ → ℕ): ℕ :=
  let sum_xs := (finset.univ : finset (Fin 6)).sum (λ i, xs i 1)
  let term1 := sum_xs ^ 6
  let term2 := 6 * sum_xs^2 * (finset.univ : finset (Fin 6)).sum (λ i, xs i 4)
  let term3 := 3 * sum_xs^2 * (finset.univ : finset (Fin 6)).sum (λ i, xs i 2)^2
  let term4 := 6 * (finset.univ : finset (Fin 6)).sum (λ i, xs i 2)^3
  let term5 := 8 * (finset.univ : finset (Fin 6)).sum (λ i, xs i 3)^2
  (term1 + term2 + term3 + term4 + term5) / 24

theorem polya_enumeration_theorem (x : Fin 6 → ℕ → ℕ) :
  coeff (P x) (polynomial.monomial 1 [1, 1, 1, 1, 1, 1]) = 30 :=
sorry

end polya_enumeration_theorem_l252_252068


namespace probability_square_product_l252_252764

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_favorable_outcomes : ℕ :=
  List.length [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (4, 4), (2, 8), (8, 2), (5, 5), (4, 9), (6, 6), (7, 7), (8, 8), (9, 9)]

def total_outcomes : ℕ := 12 * 8

theorem probability_square_product :
  (count_favorable_outcomes : ℚ) / (total_outcomes : ℚ) = (7 : ℚ) / (48 : ℚ) := 
by 
  sorry

end probability_square_product_l252_252764


namespace area_cross_section_l252_252128

-- Declare the variables used in the conditions
variable (a : ℝ)

-- Define midpoint function for convenience
def midpoint (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2)

-- Define vertices of the cube
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (a, 0, 0)
def C : ℝ × ℝ × ℝ := (a, a, 0)
def D : ℝ × ℝ × ℝ := (0, a, 0)
def A1 : ℝ × ℝ × ℝ := (0, 0, a)
def B1 : ℝ × ℝ × ℝ := (a, 0, a)

-- Define midpoints E and F
def E : ℝ × ℝ × ℝ := midpoint C D
def F : ℝ × ℝ × ℝ := midpoint A A1

-- Lean statement to prove the area of cross-section EFB1 is correct
theorem area_cross_section {a : ℝ} (h : a > 0) :
  let E := midpoint C D in
  let F := midpoint A A1 in
  let B1 : ℝ × ℝ × ℝ := (a, 0, a) in
  -- The area of the cross-section is as given
  (area_of_cross_section E F B1 = (11 * Real.sqrt 29) / 48 * a^2) := 
begin
  sorry,
end

end area_cross_section_l252_252128


namespace area_percent_of_smaller_rectangle_l252_252094

-- Definitions of the main geometric elements and assumptions
def larger_rectangle (w h : ℝ) : Prop := (w > 0) ∧ (h > 0)
def radius_of_circle (w h r : ℝ) : Prop := r = Real.sqrt (w^2 + h^2)
def inscribed_smaller_rectangle (w h x y : ℝ) : Prop := 
  (0 < x) ∧ (x < 1) ∧ (0 < y) ∧ (y < 1) ∧
  ((h + 2 * y * h)^2 + (x * w)^2 = w^2 + h^2)

-- Prove the area percentage relationship
theorem area_percent_of_smaller_rectangle 
  (w h x y : ℝ) 
  (hw : w > 0) (hh : h > 0)
  (hcirc : radius_of_circle w h (Real.sqrt (w^2 + h^2)))
  (hsmall_rect : inscribed_smaller_rectangle w h x y) :
  (4 * x * y) / (4.0 * 1.0) * 100 = 8.33 := sorry

end area_percent_of_smaller_rectangle_l252_252094


namespace concert_revenue_l252_252007

-- Define the prices and attendees
def adult_price := 26
def teenager_price := 18
def children_price := adult_price / 2
def num_adults := 183
def num_teenagers := 75
def num_children := 28

-- Calculate total revenue
def total_revenue := num_adults * adult_price + num_teenagers * teenager_price + num_children * children_price

-- The goal is to prove that total_revenue equals 6472
theorem concert_revenue : total_revenue = 6472 :=
by
  sorry

end concert_revenue_l252_252007


namespace stock_percentage_calculation_l252_252487

noncomputable def stock_percentage (investment_amount stock_price annual_income : ℝ) : ℝ :=
  (annual_income / (investment_amount / stock_price) / stock_price) * 100

theorem stock_percentage_calculation :
  stock_percentage 6800 136 1000 = 14.71 :=
by
  sorry

end stock_percentage_calculation_l252_252487


namespace total_time_on_road_l252_252621

def driving_time_day1 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def driving_time_day2 (jade_time krista_time break_time krista_refuel lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + break_time + krista_refuel + lunch_break

def driving_time_day3 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def total_driving_time (day1 day2 day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem total_time_on_road :
  total_driving_time 
    (driving_time_day1 8 6 1 1) 
    (driving_time_day2 7 5 0.5 (1/3) 1) 
    (driving_time_day3 6 4 1 1) 
  = 42.3333 := 
  by 
    sorry

end total_time_on_road_l252_252621


namespace largest_negative_sum_is_S19_l252_252258

noncomputable def largest_negative_sum {α : Type*} [linear_ordered_field α] (a : ℕ → α) (S : ℕ → α) : α :=
  S 19

theorem largest_negative_sum_is_S19 {α : Type*} [linear_ordered_field α] (a S : ℕ → α) 
  (h1 : a 10 < 0) 
  (h2 : a 11 > 0) 
  (h3 : a 11 > |a 10|) : largest_negative_sum a S = S 19 :=
sorry

end largest_negative_sum_is_S19_l252_252258


namespace prob_two_girls_is_one_fourth_l252_252837

-- Define the probability of giving birth to a girl
def prob_girl : ℚ := 1 / 2

-- Define the probability of having two girls
def prob_two_girls : ℚ := prob_girl * prob_girl

-- Theorem statement: The probability of having two girls is 1/4
theorem prob_two_girls_is_one_fourth : prob_two_girls = 1 / 4 :=
by sorry

end prob_two_girls_is_one_fourth_l252_252837


namespace trains_meet_in_time_l252_252401

noncomputable def time_for_trains_to_meet (length_train1 length_train2 distance_apart speed_train1 speed_train2 : ℝ) : ℝ :=
  let speed1 := speed_train1 * (1000 / 3600) in
  let speed2 := speed_train2 * (1000 / 3600) in
  let relative_speed := speed1 + speed2 in
  let total_distance := length_train1 + length_train2 + distance_apart in
  total_distance / relative_speed

theorem trains_meet_in_time :
  time_for_trains_to_meet 250 120 50 64 42 = 14.26 :=
by
  -- Sorry used for the proof to ensure code compiles successfully
  sorry

end trains_meet_in_time_l252_252401


namespace certain_number_equals_3460_l252_252412

theorem certain_number_equals_3460 : 
  ∃ x : ℕ, (12 * x = 173 * 240) ∧ (x = 3460) :=
by {
  use 3460,
  constructor,
  translate_the_conditions_directly,
  sorry
}

end certain_number_equals_3460_l252_252412


namespace members_playing_both_l252_252997

variable (N B T Neither BT : ℕ)

theorem members_playing_both (hN : N = 30) (hB : B = 17) (hT : T = 17) (hNeither : Neither = 2) 
  (hBT : BT = B + T - (N - Neither)) : BT = 6 := 
by 
  rw [hN, hB, hT, hNeither] at hBT
  exact hBT

end members_playing_both_l252_252997


namespace sequence_formula_l252_252939

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = -2) (h2 : a 2 = -1.2) :
  ∀ n, a n = 0.8 * n - 2.8 :=
by
  sorry

end sequence_formula_l252_252939


namespace PQRS_is_rectangle_l252_252926

open EuclideanGeometry

noncomputable def rectangle_ABCD_with_circumcenters (A B C D E P Q R S : Point) : Prop :=
  is_rectangle A B C D ∧
  midpoint A C = E ∧
  midpoint B D = E ∧
  circumscribed P (triangle A B E) ∧
  circumscribed Q (triangle B C E) ∧
  circumscribed R (triangle C D E) ∧
  circumscribed S (triangle D A E)

theorem PQRS_is_rectangle (A B C D E P Q R S : Point) :
  rectangle_ABCD_with_circumcenters A B C D E P Q R S →
  is_rectangle P Q R S :=
begin
  sorry
end

end PQRS_is_rectangle_l252_252926


namespace arithmetic_square_root_of_4_l252_252351

theorem arithmetic_square_root_of_4 : 
  ∃ (x : ℝ), x^2 = 4 ∧ 0 ≤ x ∧ x = 2 := 
begin
  use 2,
  split,
  { exact by norm_num, },
  split,
  { exact by norm_num, },
  { refl, }
end

end arithmetic_square_root_of_4_l252_252351


namespace average_side_lengths_l252_252679

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252679


namespace angle_BCD_measure_l252_252261

-- Defining points and lines
variables {A B C D : Type} 
(noncomputable def circle_diameter : Type := sorry) -- Definition for circle's diameter
(noncomputable def are_parallel (l1 l2 : Type) : Prop := sorry) -- Definition for parallel lines
(noncomputable def angle (p1 p2 p3 : Type) : Type := sorry) -- Definition for angle
noncomputable def angle_ratio (α β : Type) (m n : ℕ) : Prop := sorry -- Definition for angle ratio

-- Given Conditions
variables (AD BC AB DC : Type)
variables (ADB ABD : Type)

axiom diameter_AD : circle_diameter AD
axiom parallel_AD_BC : are_parallel AD BC
axiom parallel_AB_DC : are_parallel AB DC
axiom angles_ratio_3_7 : angle_ratio ADB ABD 3 7

-- The proof goal
theorem angle_BCD_measure : ∃ (BCD : Type), ∀ (AD BC AB DC ADB ABD : Type), diameter_AD AD → parallel_AD_BC AD BC → parallel_AB_DC AB DC → angles_ratio_3_7 ADB ABD → angle BCD A B C D = 63 :=
begin
  sorry
end

end angle_BCD_measure_l252_252261


namespace keith_attended_games_l252_252475

theorem keith_attended_games :
  ∀ (matches : List (String × String × String × String)) 
    (keith_conditions : String → String → String → Bool)
    (weekend_games : List (Nat))
    (Hawks_vs_Falcons_league : String × String × String × String)
    (Hawks_vs_Eagles_friendly : String × String × String × String)
    (Falcons_vs_Hawks_friendly : String × String × String × String),
    matches = [("League", "Hawks", "Eagles", "Hawk Stadium"),
              ("League", "Hawks", "Falcons", "Hawk Stadium"),
              ("League", "Eagles", "Falcons", "Eagle Stadium"),
              ("Friendly", "Hawks", "Eagles", "Eagle Stadium"),
              ("Friendly", "Hawks", "Falcons", "Falcon Stadium"),
              ("Friendly", "Eagles", "Falcons", "Falcon Stadium")] →
    keith_conditions "League" "Hawks" "Falcons" = "Day" →
    keith_conditions "Friendly" "Hawks" "Eagles" = "Night" →
    keith_conditions "Friendly" "Falcons" "Hawks" = "Weekend" →
    weekend_games = [5, 4] →
    ∃ (games_attended : List (String × String × String × String)),
      games_attended.length = 3 ∧
      (Hawks_vs_Falcons_league = ("League", "Hawks", "Falcons", "Hawk Stadium") ∧
      Hawks_vs_Eagles_friendly = ("Friendly", "Hawks", "Eagles", "Eagle Stadium") ∧
      Falcons_vs_Hawks_friendly = ("Friendly", "Falcons", "Hawks", "Falcon Stadium")) := 
begin
  sorry
end

end keith_attended_games_l252_252475


namespace both_reunions_attendance_l252_252381

theorem both_reunions_attendance 
    (total_guests : ℕ) 
    (oates_attendance : ℕ) 
    (yellow_attendance : ℕ) 
    (both_attendance : ℕ) :
    total_guests = oates_attendance + yellow_attendance - both_attendance 
    → total_guests = 100 
    → oates_attendance = 42 
    → yellow_attendance = 65 
    → both_attendance = 7 :=
by
  intros h1 h2 h3 h4
  rw [h2, h3, h4] at h1
  exact h1

end both_reunions_attendance_l252_252381


namespace jill_bought_5_packs_of_red_bouncy_balls_l252_252622

theorem jill_bought_5_packs_of_red_bouncy_balls
  (r : ℕ) -- number of packs of red bouncy balls
  (yellow_packs : ℕ := 4)
  (bouncy_balls_per_pack : ℕ := 18)
  (extra_red_bouncy_balls : ℕ := 18)
  (total_yellow_bouncy_balls : ℕ := yellow_packs * bouncy_balls_per_pack)
  (total_red_bouncy_balls : ℕ := total_yellow_bouncy_balls + extra_red_bouncy_balls)
  (h : r * bouncy_balls_per_pack = total_red_bouncy_balls) :
  r = 5 :=
by sorry

end jill_bought_5_packs_of_red_bouncy_balls_l252_252622


namespace part1_part2_l252_252517

def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0
def q (x m : ℝ) : Prop := m > 0 ∧ x^2 - 4*m*x + 3*m^2 ≤ 0

theorem part1 (x : ℝ) : 
  (∃ (m : ℝ), m = 1 ∧ (p x ∨ q x m)) → 1 ≤ x ∧ x ≤ 8 :=
by
  intros
  sorry

theorem part2 (m : ℝ) :
  (∀ x, q x m → p x) ∧ ∃ x, ¬ q x m ∧ p x → 2 ≤ m ∧ m ≤ 8/3 :=
by
  intros
  sorry

end part1_part2_l252_252517


namespace valid_distributions_count_l252_252101

theorem valid_distributions_count :
  let x := List.range 16
  let coeff := [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
  let valid_combinations := (x.map (λ i => List.replicate 16 0).replace_nth 0 0.replace_nth 11 0 
                               ++ x.map (λ i => List.replicate 16 0).replace_nth 0 1.replace_nth 11 0)
  (valid_combinations.length * 2^14 = 16384) := 
begin
  sorry
end

end valid_distributions_count_l252_252101


namespace dig_site_date_l252_252443

theorem dig_site_date (S F T Fourth : ℤ) 
  (h₁ : F = S - 352)
  (h₂ : T = F + 3700)
  (h₃ : Fourth = 2 * T)
  (h₄ : Fourth = 8400) : S = 852 := 
by 
  sorry

end dig_site_date_l252_252443


namespace rhombus_side_length_l252_252431

variable (d : ℝ) (K : ℝ) (s : ℝ)
variable (h_diagonal : ∃ d, ∀ s, s = 3 * d)
variable (h_area : K = (3 * d * d) / 2)

theorem rhombus_side_length (h_diagonal : h_diagonal) (h_area : h_area) :
  s = Real.sqrt (5 * K / 3) := sorry

end rhombus_side_length_l252_252431


namespace tangent_line_through_P_is_correct_l252_252203

-- Define the circle and the point
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 25
def pointP : ℝ × ℝ := (-1, 7)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 31 = 0

-- State the theorem
theorem tangent_line_through_P_is_correct :
  (circle_eq (-1) 7) → 
  (tangent_line (-1) 7) :=
sorry

end tangent_line_through_P_is_correct_l252_252203


namespace triangle_PR_values_l252_252990

variables (P Q R S : Type*) [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q] [InnerProductSpace ℝ R] [InnerProductSpace ℝ S]
variables (PQ PR QR : ℝ)
variables (PQ_len PR_len QR_len RS_len : ℝ)
variables (PR_val : Set ℝ)

noncomputable def possible_PR_values (a b : ℝ) : Set ℝ := {PR_val | 5 < PR_val ∧ PR_val < 20}

theorem triangle_PR_values :
  PQ_len = 12 →
  RS_len = 5 →
  PR_val = possible_PR_values 5 20 →
  (5 + 20 = 25) :=
by
  intros hPQ hRS hPR
  sorry

end triangle_PR_values_l252_252990


namespace in_proportion_d_value_l252_252946

noncomputable def d_length (a b c : ℝ) : ℝ := (b * c) / a

theorem in_proportion_d_value :
  let a := 2
  let b := 3
  let c := 6
  d_length a b c = 9 := 
by
  sorry

end in_proportion_d_value_l252_252946


namespace convex_bodies_intersect_l252_252011

noncomputable def convex_body := { s // is_convex s ∧ is_compact s }

theorem convex_bodies_intersect
  (B1 B2 : convex_body)
  (proj_xy : ∃ s, ∀ z, (B1.1 ∩ {t | t.z = z}).proj xy = (B2.1 ∩ {t | t.z = z}).proj xy)
  (proj_xz : ∃ s, ∀ y, (B1.1 ∩ {t | t.y = y}).proj xz = (B2.1 ∩ {t | t.y = y}).proj xz)
  (proj_yz : ∃ s, ∀ x, (B1.1 ∩ {t | t.x = x}).proj yz = (B2.1 ∩ {t | t.x = x}).proj yz) :
  ∃ p, p ∈ B1 ∧ p ∈ B2 := 
  sorry

end convex_bodies_intersect_l252_252011


namespace range_of_a_l252_252239

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 + (a+1)*x + a < 0) → a ∈ Set.Iio (-2 / 3) := 
sorry

end range_of_a_l252_252239


namespace third_smallest_three_digit_palindromic_prime_l252_252161

-- Definitions of palindromic prime and three-digit numbers
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_prime (n : ℕ) : Prop := nat.prime n

def is_palindromic_prime (n : ℕ) : Prop :=
  is_palindrome n ∧ is_prime n

-- Lean 4 statement.
theorem third_smallest_three_digit_palindromic_prime :
  ∃ n : ℕ, is_palindromic_prime n ∧ 101 < n ∧ 131 < n ∧
  (n = 101 ∨ n = 131 ∨ n = 151) ∧
  ∀ m : ℕ, is_palindromic_prime m → 101 < m → m < n → m ≠ 131 :=
sorry

end third_smallest_three_digit_palindromic_prime_l252_252161


namespace average_side_length_of_squares_l252_252719

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252719


namespace determinant_3y_plus_1_l252_252480

variable (y : ℝ)

def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![y + 1, y, y],
    ![y, y + 1, y],
    ![y, y, y + 1]
  ]

theorem determinant_3y_plus_1 : matrix.det = 3 * y + 1 :=
  by
    sorry

end determinant_3y_plus_1_l252_252480


namespace correct_statement_l252_252786

theorem correct_statement (a b c : ℝ) (h1 : ac = bc) (h2 : a = b) (h3 : a^2 = b^2) : 
  (∀ (c ≠ 0), (ac = bc → a = b)) ∧ 
  (∀ (c ≠ 0), (a / c = b / c → a = b)) ∧
  (a = b → a + 3 = b + 3) ∧ 
  (a^2 = b^2 → a = b) :=
by 
  sorry

end correct_statement_l252_252786


namespace radical_axis_altitude_through_A_l252_252930

-- Define the geometric setting and the hypothesis
variables {A B C D E : Type} [Geometry]
variable (triangle_ABC : triangle A B C)
variable (D_on_AB : lies_on D (segment A B))
variable (E_on_AC : lies_on E (segment A C))
variable (DE_parallel_BC : parallel (segment D E) (segment B C))

-- Define the circles with given diameters
def O1 : circle := circle_with_diameter (segment B E)
def O2 : circle := circle_with_diameter (segment C D)

-- State the theorem
theorem radical_axis_altitude_through_A :
  radical_axis O1 O2 = altitude_from A triangle_ABC :=
by
  sorry

end radical_axis_altitude_through_A_l252_252930


namespace difference_two_digit_interchanged_l252_252373

theorem difference_two_digit_interchanged
  (x y : ℕ)
  (h1 : y = 2 * x)
  (h2 : (10 * x + y) - (x + y) = 8) :
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end difference_two_digit_interchanged_l252_252373


namespace log_sum_of_geom_sequence_l252_252598

noncomputable def geometric_sequence (b : ℕ → ℝ) := ∃ r, ∀ n, b (n+1) = b n * r

theorem log_sum_of_geom_sequence :
  ∀ (b : ℕ → ℝ), (geometric_sequence b) → (b 7 * b 8 = 3) → 
  (∑ i in finset.range 14, real.log (b i) / real.log 3) = 7 :=
by
  sorry

end log_sum_of_geom_sequence_l252_252598


namespace least_number_of_times_l252_252474

noncomputable def A_eats (x : ℕ) := 1.8 * x
noncomputable def C_eats (x : ℕ) := 8 * x
def B_eats (x : ℕ) := x

theorem least_number_of_times (x : ℕ) : 
  (A_eats x).denominator = 1 ∧ A_eats x = 9 → x = 5 :=
by
  unfold A_eats
  norm_num
  intro h
  have h1 := congr_arg (coe : ℕ → ℚ) h.left
  norm_cast at h1
  norm_num1 at h1
  sorry

end least_number_of_times_l252_252474


namespace range_of_a_l252_252193

/-- Definitions for propositions p and q --/
def p (a : ℝ) : Prop := a > 0 ∧ a < 1
def q (a : ℝ) : Prop := (2 * a - 3) ^ 2 - 4 > 0

/-- Theorem stating the range of possible values for a given conditions --/
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ¬(p a) ∧ ¬(q a) = false) (h4 : p a ∨ q a) :
  (1 / 2 ≤ a ∧ a < 1) ∨ (a ≥ 5 / 2) :=
sorry

end range_of_a_l252_252193


namespace first_number_is_two_l252_252742

-- Given conditions as definitions
def seq : ℕ → ℕ
| 0 := 2
| 1 := 3
| 2 := 6
| 3 := 15
| 4 := 33
| 5 := 123
| _ := 0  -- As the sequence is not specified beyond the 6th element.

-- Statement to prove the first number in the sequence is 2
theorem first_number_is_two : seq 0 = 2 :=
by
  exact rfl

end first_number_is_two_l252_252742


namespace not_all_zero_iff_at_least_one_nonzero_l252_252758

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by 
  sorry

end not_all_zero_iff_at_least_one_nonzero_l252_252758


namespace circle_tangent_nature_l252_252763

noncomputable def circle_tangent_sum (a b c : ℝ) (h : a > b ∧ b > c) : Prop :=
  (a + b) + (b + c) - (a + c) = b

theorem circle_tangent_nature (a b c : ℝ) (h : a > b ∧ b > c) : circle_tangent_sum a b c h :=
begin
  sorry
end

end circle_tangent_nature_l252_252763


namespace cost_of_1000_pieces_of_gum_l252_252356

theorem cost_of_1000_pieces_of_gum
  (cost_per_piece : ℕ)
  (num_pieces : ℕ)
  (discount_threshold : ℕ)
  (discount_rate : ℚ)
  (conversion_rate : ℕ)
  (h_cost : cost_per_piece = 2)
  (h_pieces : num_pieces = 1000)
  (h_threshold : discount_threshold = 500)
  (h_discount : discount_rate = 0.90)
  (h_conversion : conversion_rate = 100)
  (h_more_than_threshold : num_pieces > discount_threshold) :
  (num_pieces * cost_per_piece * discount_rate) / conversion_rate = 18 := 
sorry

end cost_of_1000_pieces_of_gum_l252_252356


namespace square_in_rectangle_l252_252097

theorem square_in_rectangle (s : ℝ) (h_width : Nonempty ℝ)
  (h_len : Nonempty ℝ): 
  let width := 3 * s in 
  let length := 4.5 * s in 
  let square_area := s ^ 2 in 
  let rectangle_area := length * width in
  (square_area / rectangle_area) * 100 = 7.41 := 
by 
  sorry

end square_in_rectangle_l252_252097


namespace second_cart_travel_distance_l252_252765

-- Given definitions:
def first_cart_first_term : ℕ := 6
def first_cart_common_difference : ℕ := 8
def second_cart_first_term : ℕ := 7
def second_cart_common_difference : ℕ := 9

-- Given times:
def time_first_cart : ℕ := 35
def time_second_cart : ℕ := 33

-- Arithmetic series sum formula
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Total distance traveled by the second cart
noncomputable def distance_second_cart : ℕ :=
  arithmetic_series_sum second_cart_first_term second_cart_common_difference time_second_cart

-- Theorem to prove the distance traveled by the second cart
theorem second_cart_travel_distance : distance_second_cart = 4983 :=
  sorry

end second_cart_travel_distance_l252_252765


namespace hyperbola_intersect_lines_count_l252_252420

theorem hyperbola_intersect_lines_count :
  let hyperbola := λ x y : ℝ, x^2 - (y^2 / 2) = 1
  let F := (1 : ℝ, 0) -- Right focus of the hyperbola
  ∃ (l : ℝ → ℝ), 
    (∃ A B : ℝ × ℝ, (l 1 = 0) ∧ (hyperbola A.1 A.2) ∧ (hyperbola B.1 B.2) ∧ (|A - B| = 4)) ∧
    (count (λ f, (∃ A B : ℝ × ℝ, (l ∈ f ∧ A ∈ f ∧ B ∈ f ∧ |A - B| = 4))) = 3)
  := sorry

end hyperbola_intersect_lines_count_l252_252420


namespace value_of_x_l252_252363

noncomputable def data_set := [70, 110, x, 55, 45, 220, 85, 65, x]

theorem value_of_x :
  let x := (650 + 2 * x) / 9 in
  let sorted_set := List.sort data_set in
  sorted_set.nth 4 = x ∧
  (data_set.filter (λ y, y = x)).length = 2 →
  x = 93 := by
sorry

end value_of_x_l252_252363


namespace intersection_eq_l252_252560

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def M : Set ℝ := { x | -1/2 < x ∧ x < 1/2 }
def N : Set ℝ := { x | 0 ≤ x ∧ x * x ≤ x }

theorem intersection_eq :
  M ∩ N = { x | 0 ≤ x ∧ x < 1/2 } := by
  sorry

end intersection_eq_l252_252560


namespace evaluate_expression_l252_252481

theorem evaluate_expression : (∑ i in {3, 6, 9}, i) / (∑ j in {2, 5, 8}, j) - (∑ k in {2, 5, 8}, k) / (∑ l in {3, 6, 9}, l) = 11 / 30 := by
  sorry

end evaluate_expression_l252_252481


namespace num_unique_sums_bags_l252_252844

noncomputable def bagA : List ℕ := [1, 4, 5, 7]
noncomputable def bagB : List ℕ := [2, 5, 6, 8]

theorem num_unique_sums_bags : 
  (List.sum (List.erase_dup ((bagA.bind (λ a, bagB.map (λ b, a + b)))))).length = 9 := 
by
  -- Proof goes here
  sorry

end num_unique_sums_bags_l252_252844


namespace find_x_if_orthogonal_l252_252878

def vec1 := (3, -1, 4)
def vec2 (x : ℝ) := (x, 4, -2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_x_if_orthogonal :
  (dot_product vec1 (vec2 x) = 0) → x = 4 := by
sorry

end find_x_if_orthogonal_l252_252878


namespace num_lines_through_points_l252_252462

theorem num_lines_through_points :
  ∃ l : ℕ → ℕ → Prop, (∀ a b : ℕ, l a b → (a > 0 ∧ b > 0) ∧ 
  ∀ x y : ℕ, l x 0 → l 0 y → (y - 3) = 3 * ((x - 3)*(x - 1) + 0) / (x - 1)) ∧
  (∃ a b : ℕ, l a b) ∧ 
  (∃ l1 l2, l l1 l2) →
  (∃ l1 l2 l3 l4, l l1 l2 ∧ l l3 l4 → (l1 ≠ l3 ∨ l2 ≠ l4) ∧ 2 = 2) :=
begin
  sorry
end

end num_lines_through_points_l252_252462


namespace distinct_sums_count_l252_252516

theorem distinct_sums_count (n : ℕ) (a : Fin n.succ → ℕ) (h_distinct : Function.Injective a) :
  ∃ (S : Finset ℕ), S.card ≥ n * (n + 1) / 2 := sorry

end distinct_sums_count_l252_252516


namespace total_chips_l252_252233

theorem total_chips (rows_with_8_chips : Nat) (chips_per_row_8 : Nat) (rows_with_4_chips : Nat) (chips_per_row_4 : Nat) : 
  rows_with_8_chips = 9 → 
  chips_per_row_8 = 8 → 
  rows_with_4_chips = 1 → 
  chips_per_row_4 = 4 → 
  (rows_with_8_chips * chips_per_row_8 + rows_with_4_chips * chips_per_row_4) = 76 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end total_chips_l252_252233


namespace illuminated_area_correct_l252_252084

noncomputable def cube_illuminated_area (a ρ : ℝ) (h₁ : a = 1 / Real.sqrt 2) (h₂ : ρ = Real.sqrt (2 - Real.sqrt 3)) : ℝ :=
  (Real.sqrt 3 - 3 / 2) * (Real.pi + 3)

theorem illuminated_area_correct :
  cube_illuminated_area (1 / Real.sqrt 2) (Real.sqrt (2 - Real.sqrt 3)) (by norm_num) (by norm_num) = (Real.sqrt 3 - 3 / 2) * (Real.pi + 3) :=
sorry

end illuminated_area_correct_l252_252084


namespace triangle_angle_ratio_l252_252271

theorem triangle_angle_ratio (A B C : ℝ)
  (h1 : A < B)
  (h2 : B < C)
  (h3 : 2 * B = A + C)
  (h4 : A + B + C = 180)
  (h5 : ∃ (a b c : ℝ), c = 2 * a ∧ a / sin A = b / sin B ∧ c / sin C = b / sin B):
  A / B / C = 1 / 2 / 3 := 
by
  sorry

end triangle_angle_ratio_l252_252271


namespace base2_representation_of_27_l252_252874

-- The theorem states that the base-2 representation of 27 is 11011.
theorem base2_representation_of_27 : nat.binary 27 = [1, 1, 0, 1, 1] :=
by
  sorry

end base2_representation_of_27_l252_252874


namespace polygon_sides_diagonals_l252_252593

theorem polygon_sides_diagonals (n : ℕ) 
  (h1 : 180 * (n - 2) = 360 * 3 + 180)
  (h2 : ∑ i in range n, exterior_angle i = 360)
  (h3 : ∑ i in range n, interior_angle i = 180 * (n - 2)) :
  n = 9 ∧ (n * (n - 3)) / 2 = 27 :=
by
  sorry

end polygon_sides_diagonals_l252_252593


namespace tangent_line_b_value_l252_252469

noncomputable def b_value : ℝ := Real.log 2 - 1

theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x > 0, (fun x => Real.log x) x = (1/2) * x + b → ∃ c : ℝ, c = b) → b = Real.log 2 - 1 :=
by
  sorry

end tangent_line_b_value_l252_252469


namespace num_integers_satisfying_conditions_l252_252567

theorem num_integers_satisfying_conditions :
  let eligible := {m : ℤ | (1 / (|m| : ℝ) >= 1 / 5) ∧ (m % 3 ≠ 0)}
  cardinality eligible = 8 :=
by
  sorry

end num_integers_satisfying_conditions_l252_252567


namespace shoveling_time_l252_252623

theorem shoveling_time (joan_time mary_time : ℕ) (h1 : joan_time = 50) (h2 : mary_time = 20) : 
  let joan_rate := 1 / (joan_time : ℝ)
  let mary_rate := 1 / (mary_time : ℝ)
  let combined_rate := joan_rate + mary_rate
  let time_together := 1 / combined_rate
  nat.nearest (time_together) = 14 :=
by 
  sorry

end shoveling_time_l252_252623


namespace average_side_length_of_squares_l252_252726

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252726


namespace length_RY_l252_252615

def triangle.PQR (PQ PR QR : ℝ) : Prop :=
  PQ = 10 ∧ PQ > 0 ∧ PR > 0 ∧ QR > 0 ∧ (∃ XYZ, XYZ = 6 ∧ Y ∈ QR ∧ X ∈ PR ∧ XYZ // PQ)

def line.bisecting (bisecting_line : ℝ → Prop) (X Y Z : Point) : Prop :=
  bisecting_line ∠XZR

theorem length_RY (PQ PR QR XY X Y Z : ℝ) (h1 : triangle.PQR PQ PR QR)
  (h2 : parallel XY PQ) (h3 : XY = 6) (h4 : on_segment X PR) (h5 : on_segment Y QR)
  (h6 : line.bisecting PY X Z R) : 
  RY = 15 := 
sorry

end length_RY_l252_252615


namespace proof_problem_l252_252864

noncomputable def mass_of_CaCO3_required 
  (moles_HCl : ℕ)
  (molar_mass_Ca : ℝ)
  (molar_mass_C : ℝ)
  (molar_mass_O : ℝ) : ℝ :=
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let moles_CaCO3 := moles_HCl / 2
  moles_CaCO3 * molar_mass_CaCO3

noncomputable def change_in_enthalpy
  (moles_CaCl2 : ℕ)
  (standard_enthalpy_change : ℝ) : ℝ :=
  moles_CaCl2 * standard_enthalpy_change

theorem proof_problem :
  mass_of_CaCO3_required 4 40.08 12.01 16.00 = 200.18 ∧
  change_in_enthalpy 2 -178 = -356 :=
by {
  split,
  {
    sorry
  },
  {
    sorry
  }
}

end proof_problem_l252_252864


namespace average_side_length_of_squares_l252_252728

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252728


namespace min_value_fraction_l252_252919

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) :
  ∃ m, (∀ x y, x > 0 → y > 0 → x + y = 2 → (∃ m, m = (min ((1 / x^2) + (1 / y^2) + (1 / (x * y))))) → m = 3 :=
sorry

end min_value_fraction_l252_252919


namespace range_of_a_l252_252211

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 - 2 * a) ^ x else log a x + 1 / 3

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 3) :=
sorry

end range_of_a_l252_252211


namespace find_x_l252_252880

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l252_252880


namespace max_min_digit_proof_l252_252093

def max_number (level1: List ℕ) (level2: List ℕ) : ℕ :=
  (level1.reverse.foldl (λ acc d, acc * 10 + d) 0) * 10^4 + 
  (level2.reverse.foldl (λ acc d, acc * 10 + d) 0)

def min_number (level1: List ℕ) (level2: List ℕ) : ℕ :=
  (level1.foldl (λ acc d, acc * 10 + d) 0) * 10^4 + 
  (level2.foldl (λ acc d, acc * 10 + d) 0)

theorem max_min_digit_proof :
  (max_number [1, 4, 6, 8] [0, 0, 0, 5] = 86415000) ∧
  (min_number [1, 4, 6, 8] [0, 0, 0, 5] = 14680005) :=
by
  sorry

end max_min_digit_proof_l252_252093


namespace triangle_MNK_equilateral_l252_252380

open_locale classical

variables {A B C D E M N K : Point}

-- Define a point type.
structure Point :=
  (x y : ℝ)

-- Define the midpoint
def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

-- Define equilateral triangle structure (assuming conditions for an equilateral triangle).
structure EquilateralTriangle (A B C : Point) :=
  (AB BC CA : ℝ)
  (eq_side : AB = BC ∧ BC = CA ∧ CA = AB)

noncomputable def is_midpoint (M P Q : Point) : Prop := 
  M = midpoint P Q

-- Define the given conditions as hypotheses
variables (ABC_equilateral : EquilateralTriangle A B C)
variables (CDE_equilateral : EquilateralTriangle C D E)
variables (AE_line : Line A E)
variables (C_shared : A ≠ C ∧ B ≠ C ∧ D ≠ C ∧ E ≠ C)
variables (same_side : ∀ P : Point, P ∈ AE_line → same_side_line P C)
variables (M_midpoint : M = midpoint B D)
variables (N_midpoint : N = midpoint A C)
variables (K_midpoint : K = midpoint C E)

-- State the theorem
theorem triangle_MNK_equilateral :
  is_equilateral_triangle M N K :=
sorry

end triangle_MNK_equilateral_l252_252380


namespace amy_total_money_l252_252441

theorem amy_total_money 
  (initial_money : ℕ)
  (chores_money : ℕ)
  (birthday_money : ℕ) 
  (h1 : initial_money = 2) 
  (h2 : chores_money = 13) 
  (h3 : birthday_money = 3) : 
  initial_money + chores_money + birthday_money = 18 :=
by 
  -- Assume initial conditions
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- Show the final result
  exact rfl

end amy_total_money_l252_252441


namespace find_P0_coordinates_l252_252760

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + x - 2

theorem find_P0_coordinates :
  ∃ P0 : ℝ × ℝ, P0 = (1, 1) ∧ ∃ a : ℝ, P0 = (a, f a) ∧
  (∃ b : ℝ, ∀ x : ℝ, f'(x) = 4 * x + 1 ∧ (f' a = b) ∧ (b = 5)) :=
by
  sorry

end find_P0_coordinates_l252_252760


namespace sandwich_cost_correct_l252_252168

noncomputable def cost_per_sandwich (bread_cost meat_cost cheese_cost coupon_meat coupon_cheese : Float) (total_sandwiches : Nat) : Float :=
  let loaves := total_sandwiches / 10
  let total_bread_cost := loaves * bread_cost
  let total_meat_cost := (loaves * 2 * meat_cost) - coupon_meat
  let total_cheese_cost := (loaves * 2 * cheese_cost) - coupon_cheese
  let total_cost := total_bread_cost + total_meat_cost + total_cheese_cost
  total_cost / Float.ofNat total_sandwiches

theorem sandwich_cost_correct :
  cost_per_sandwich 4.0 5.0 4.0 1.0 1.0 50 = 2.16 :=
by sorry

end sandwich_cost_correct_l252_252168


namespace streetlights_each_square_l252_252350

-- Define the conditions
def total_streetlights : Nat := 200
def total_squares : Nat := 15
def unused_streetlights : Nat := 20

-- State the question mathematically
def streetlights_installed := total_streetlights - unused_streetlights
def streetlights_per_square := streetlights_installed / total_squares

-- The theorem we need to prove
theorem streetlights_each_square : streetlights_per_square = 12 := sorry

end streetlights_each_square_l252_252350


namespace bike_ride_energetic_time_l252_252448

theorem bike_ride_energetic_time :
  ∃ x : ℚ, (22 * x + 15 * (7.5 - x) = 142) ∧ x = (59 / 14) :=
by
  sorry

end bike_ride_energetic_time_l252_252448


namespace find_fixed_tangent_circle_l252_252563

open Real

def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def orthogonal (O P Q : Point) : Prop := (P.x * Q.x + P.y * Q.y = 0)

def tangentCircleExists (r : ℝ) (a b : Point) (M uₜ l_m k: Prop) : Prop :=
∃ (c : Circle), c.radius^2 = r*12 / 7 ∧ ∀ (P Q: Point), c tangent l_m

theorem find_fixed_tangent_circle :
  (ellipse 2 (sqrt 3) x y) ∧ (∃ r:ℝ, r > 0) →
  tangentCircleExists (sqrt (12 / 7)) A B M l_m k :=
sorry -- No proof needed

end find_fixed_tangent_circle_l252_252563


namespace graph_contains_k_disjoint_paths_l252_252306

variables {V : Type} [DecidableEq V]
variables (G : SimpleGraph V) (k : ℕ) (A B : Finset V)

theorem graph_contains_k_disjoint_paths 
  (h_sep : ∀ (S : Finset V), (Finset.card S < k) → ¬ (G.is_vertex_separator A B S))
  (h_not_sep : ∃ (S : Finset V), (Finset.card S = k) ∧ G.is_vertex_separator A B S) :
  ∃ (P : Finset (Finset V)), Finset.card P = k ∧ (∀ p ∈ P, G.path_between (p A B)) :=
sorry

end graph_contains_k_disjoint_paths_l252_252306


namespace find_some_number_l252_252034

def simplify_expr (x : ℚ) : Prop :=
  1 / 2 + ((2 / 3 * (3 / 8)) + x) - (8 / 16) = 4.25

theorem find_some_number :
  ∃ x : ℚ, simplify_expr x ∧ x = 4 :=
by
  sorry

end find_some_number_l252_252034


namespace angle_BPD_is_61_l252_252304

-- Define the properties for the triangle, midpoints, and angles
variable (A B C D E F P : Type)
variable [triangle A B C]
variable [midpoint D B C]
variable [midpoint E C A]
variable [midpoint F A B]
variable [angle_bisector P F D E]
variable [angle_bisector P F B D]
variable (angle_BAC angle_CBA angle_BPD : ℝ)

-- Given conditions
variable (h1 : angle_BAC = 37)
variable (h2 : angle_CBA = 85)

-- Prove that the angle BPD is 61 degrees
theorem angle_BPD_is_61 :
  angle_BPD = 61 :=
sorry

end angle_BPD_is_61_l252_252304


namespace probability_of_team_A_winning_is_11_over_16_l252_252670

noncomputable def prob_A_wins_series : ℚ :=
  let total_games := 5
  let wins_needed_A := 2
  let wins_needed_B := 3
  -- Assuming equal probability for each game being won by either team
  let equal_chance_of_winning := 0.5
  -- Calculation would follow similar steps omitted for brevity
  -- Assuming the problem statement proven by external logical steps
  11 / 16

theorem probability_of_team_A_winning_is_11_over_16 :
  prob_A_wins_series = 11 / 16 := 
  sorry

end probability_of_team_A_winning_is_11_over_16_l252_252670


namespace complex_conjugate_l252_252309

variable (z : ℂ)

theorem complex_conjugate (h : z - complex.I = 3 + complex.I) : conj z = 3 - 2 * complex.I := 
sorry

end complex_conjugate_l252_252309


namespace count_integers_satisfying_inequalities_l252_252863

theorem count_integers_satisfying_inequalities :
  {n : ℤ | (real.sqrt (n + 2) ≤ real.sqrt (3 * n - 4)) ∧ 
             (real.sqrt (3 * n - 4) < real.sqrt (4 * n + 1))}.to_finset.card = 3 :=
sorry

end count_integers_satisfying_inequalities_l252_252863


namespace part1_part2_l252_252227

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -4)

-- Part 1: When (a + b) is perpendicular to (-2a), find m
theorem part1 (m : ℝ) : 
  let ab := (fst a + fst (b m), snd a + snd (b m))
  let neg2a := (-2 * fst a, -2 * snd a)
  dot_prod ab neg2a = 0 -> m = -3 := by
  sorry

-- Part 2: Angle between a and b is obtuse, find the range for m
theorem part2 (m : ℝ) :
  dot_prod a (b m) < 0 ↔ m > -8 :=
  have not_collinear : ¬(2 * m = 4) := by
    intro h
    have h2 : m = 2 := by linarith
    contradiction
  by exact ⟨λ h, by linarith, λ h, by linarith [h]⟩

end part1_part2_l252_252227


namespace proportion_in_triangle_l252_252989

variable (a b c x y : ℝ)

-- Conditions from the problem
variable (h1 : ∆ABC) -- Represents triangle ABC
variable (h2 : (c*(x + y) = 2*b*c)) -- When AC = 2b and E divides AC into x and y, then c(x + y) = 2bc
variable (h3 : (c/x = a/y)) -- From the Angle Bisector Theorem

theorem proportion_in_triangle 
  (h1 : ∆ABC) 
  (h2 : c*(x + y) = 2*b*c) 
  (h3 : c/x = a/y) : x/y = c/a := 
by 
  sorry

end proportion_in_triangle_l252_252989


namespace Amy_initial_cupcakes_l252_252440

def initialCupcakes (packages : ℕ) (cupcakesPerPackage : ℕ) (eaten : ℕ) : ℕ :=
  packages * cupcakesPerPackage + eaten

theorem Amy_initial_cupcakes :
  let packages := 9
  let cupcakesPerPackage := 5
  let eaten := 5
  initialCupcakes packages cupcakesPerPackage eaten = 50 :=
by
  sorry

end Amy_initial_cupcakes_l252_252440


namespace faster_train_speed_l252_252771

noncomputable def speed_of_faster_train (length_of_train speed_slower_train passing_time : ℕ) : ℕ :=
  let rel_speed_m_per_s := (49 - speed_slower_train) * 5 / 18 in
  let total_length := length_of_train * 2 in
  if rel_speed_m_per_s * passing_time = total_length then 49 else sorry

theorem faster_train_speed :
  let length_of_train := 65 in
  let speed_slower_train := 36 in
  let passing_time := 36 in
  speed_of_faster_train length_of_train speed_slower_train passing_time = 49 :=
by
  sorry

end faster_train_speed_l252_252771


namespace sequence_an_general_formula_sequence_bn_inequality_l252_252945

theorem sequence_an_general_formula (a : ℕ → ℝ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, (sqrt (a n), a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = p.1^2 + 1}) :
  ∀ n : ℕ, a n = n :=
by
  sorry

theorem sequence_bn_inequality (a b : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, (sqrt (a n), a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = p.1^2 + 1})
  (hb0 : b 1 = 1)
  (hb1 : ∀ n : ℕ, b (n + 1) = b n + 2^a n) :
  ∀ n : ℕ, b n * b (n + 2) < (b (n + 1))^2 :=
by
  sorry

end sequence_an_general_formula_sequence_bn_inequality_l252_252945


namespace average_side_length_of_squares_l252_252729

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252729


namespace intersection_probability_intersection_distribution_expectation_l252_252920

noncomputable def C1 (a b : ℕ) : (ℕ × ℕ) → Prop := 
  fun p => let (x, y) := p; y = a * x + b

noncomputable def C2 : (ℕ × ℕ) → Prop := 
  fun p => let (x, y) := p; x^2 + y^2 = 2

theorem intersection_probability : ∃ P : ℚ, P = 19 / 25 ∧ (∀ a b, a ∈ {0, 1, 2, 3, 4} ∧ b ∈ {0, 1, 2, 3, 4} → (C1 a b = C2) ↔ P = 19 / 25) :=
  sorry

theorem intersection_distribution_expectation : 
  ∃ f : ℕ → ℚ, 
  (f 0 = 6 / 25 ∧ f 1 = 1 / 25 ∧ f 2 = 18 / 25) ∧ 
  (∀ X, X ∈ {0, 1, 2} → E[X] = 37 / 25) :=
  sorry

end intersection_probability_intersection_distribution_expectation_l252_252920


namespace solution_set_of_inequality_l252_252980

variables {R : Type*} [linear_ordered_field R]

-- Conditions
def is_decreasing (f : R → R) := ∀ x y : R, x < y → f y ≤ f x
def passes_through_A_and_B (f : R → R) := (f 0 = 3) ∧ (f 3 = -1)

-- Proof Statement
theorem solution_set_of_inequality (f : R → R) (h1 : is_decreasing f) (h2 : passes_through_A_and_B f) :
  { x : R | -1 < x ∧ x < 2 } = { x : R | |f(x+1) - 1| < 2 } :=
sorry

end solution_set_of_inequality_l252_252980


namespace f_linear_dim_R3_dim_R4_f_Pi_basis_generating_set_dim_Im_f_f_injective_f_not_bijective_l252_252125

variables (K : Type*) [Field K]

structure polynomial_ring (n : ℕ) :=
(basis : Fin n → K[X])

def R3 := polynomial_ring K 4
def R4 := polynomial_ring K 5

def f (P : K[X]) : K[X] := X * P

variable (i : Fin 4)
def P (i : Fin 5) : K[X] := X ^ i.val

-- Show that f is a linear application
theorem f_linear : ∀ (a b : K) (P Q : K[X]), f (a • P + b • Q) = a • f P + b • f Q :=
sorry

-- Dimensions of R3 and R4
theorem dim_R3 : (Basis.ofVectorSpace K R3).dim = 4 :=
sorry

theorem dim_R4 : (Basis.ofVectorSpace K R4).dim = 5 :=
sorry

-- Calculation of f(P_i(X)) for 0 <= i <= 3
theorem f_Pi (i : Fin 4) : f (P i) = P (Fin.mk (i.val + 1) (Nat.lt_of_succ_lt_succ i.2)) :=
sorry

-- Image of a basis is a generating set of Im f
theorem basis_generating_set (E F : Type*) [AddCommGroup E] [AddCommGroup F] [VectorSpace K E] [VectorSpace K F] (li : Basis (Fin 4) K E) (f : E →ₗ[K] F) :
  LinearIndependent K (LinearMap.toFun f '' set.range (li : Fin 4 → E)) :=
sorry

-- Dimension of Im f
theorem dim_Im_f : (Basis.ofVectorSpace K (LinearMap.range f)).dim = 4 :=
sorry

-- f is injective
theorem f_injective : ∀ (P : K[X]), f P = 0 → P = 0 :=
sorry

-- f is not bijective
theorem f_not_bijective : ¬(LinearMap.surjective f) :=
sorry

end f_linear_dim_R3_dim_R4_f_Pi_basis_generating_set_dim_Im_f_f_injective_f_not_bijective_l252_252125


namespace volume_tetrahedron_PQRS_eq_l252_252349

-- Given tetrahedron PQRS with edges PQ=6, PR=5, PS=5, QR=5, QS=4, RS=10/3 * sqrt(3)
variables {P Q R S : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
variable [Dist : Π (a b : Type), CoeSort (MetricSpace (a → b)) ℝ]
variable (PQ PR PS QR QS RS : ℝ)
variable (hPQ : PQ = 6)
variable (hPR : PR = 5)
variable (hPS : PS = 5)
variable (hQR : QR = 5)
variable (hQS : QS = 4)
variable (hRS : RS = 10 / 3 * Real.sqrt 3)

-- Theorem stating the volume of the tetrahedron PQRS
theorem volume_tetrahedron_PQRS_eq : 
  let V := volume_tetrahedron PQ RS PQ RS in
  V = 20 / 3 :=
by
  -- replace this with the proof
  sorry

end volume_tetrahedron_PQRS_eq_l252_252349


namespace total_cost_of_repair_l252_252316

theorem total_cost_of_repair (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) (H1 : hours = 2) (H2 : hourly_rate = 75) (H3 : part_cost = 150) :
  hours * hourly_rate + part_cost = 300 := 
by
  sorry

end total_cost_of_repair_l252_252316


namespace maximal_sequence_length_is_15_l252_252027

noncomputable def sequence_max_length : ℕ :=
  let n := 15 in
  let seq : ℕ → ℤ := λ i, if i % 10 < 5 then 1 else -1 in
  have h_seq_length : ∀ k, k < n → seq k = 1 ∨ seq k = -1, from λ k hk, by
    cases (k % 10) <;> simp [seq],
  have h_sum_10_zero : ∀ k, k + 9 < n → ∑ i in range 10, seq (k + i) = 0, from λ k hk, by
    simp [seq, sum_range_subtype, finset.sum_add_distrib, mul_add, mul_comm],
  have h_sum_12_nonzero : ∀ k, k + 11 < n → ∑ i in range 12, seq (k + i) ≠ 0, from λ k hk, by {
    have : ∑ i in range 12, seq (k + i) = 2, {
      simp [seq, sum_range_subtype, finset.sum_add_distrib, mul_add, mul_comm, mul_assoc, bit0]
    },
    exact ne_of_gt zero_lt_two,
  },
  n

#print sequence_max_length -- should return 15 with appropriate assumptions

theorem maximal_sequence_length_is_15 :
  ∃ n, (∀ seq : ℕ → ℤ,
          (∀ k < n, seq k = 1 ∨ seq k = -1) ∧
          (∀ k, k + 9 < n → ∑ i in range 10, seq (k + i) = 0) ∧
          (∀ k, k + 11 < n → ∑ i in range 12, seq (k + i) ≠ 0) →
        n ≤ 15) :=
by
  use 15
  intro seq
  intro h_seq_props
  cases h_seq_props with h_seq_val h_seq_con
  cases h_seq_con with h_seq_sum_10 h_seq_sum_12
  sorry

end maximal_sequence_length_is_15_l252_252027


namespace average_side_lengths_l252_252678

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252678


namespace modulus_z_l252_252922
noncomputable def z : ℂ := sorry

theorem modulus_z
  (hz : ∀ z : ℂ, (1 - complex.sqrt 3 * z) / (1 + complex.sqrt 3 * z) = complex.I) :
  |z| = real.sqrt 3 / 3 :=
sorry

end modulus_z_l252_252922


namespace ellipse_slope_condition_l252_252609

theorem ellipse_slope_condition (a b x y x₀ y₀ : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h_ellipse1 : x^2 / a^2 + y^2 / b^2 = 1) 
  (h_ellipse2 : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (hA : x ≠ x₀ ∨ y ≠ y₀) 
  (hB : x ≠ -x₀ ∨ y ≠ -y₀) :
  ((y - y₀) / (x - x₀)) * ((y + y₀) / (x + x₀)) = -b^2 / a^2 := 
sorry

end ellipse_slope_condition_l252_252609


namespace Dalton_saved_amount_l252_252860

theorem Dalton_saved_amount (total_cost uncle_contribution additional_needed saved_from_allowance : ℕ) 
  (h_total_cost : total_cost = 7 + 12 + 4)
  (h_uncle_contribution : uncle_contribution = 13)
  (h_additional_needed : additional_needed = 4)
  (h_current_amount : total_cost - additional_needed = 19)
  (h_saved_amount : 19 - uncle_contribution = saved_from_allowance) :
  saved_from_allowance = 6 :=
sorry

end Dalton_saved_amount_l252_252860


namespace min_value_expr_l252_252535

open Real

theorem min_value_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ c, c = 4 * sqrt 3 - 6 ∧ ∀ (z w : ℝ), z = x ∧ w = y → (3 * z) / (3 * z + 2 * w) + w / (2 * z + w) ≥ c :=
by
  sorry

end min_value_expr_l252_252535


namespace quadrilateral_extended_area_l252_252855

theorem quadrilateral_extended_area (AB BC CD DA : ℝ) (AB_BB' : AB = 8) (BC_CC' : BC = 9)
  (CD_DD' : CD = 10) (DA_AA' : DA = 11) (ABCD_area : (Area ABCD) = 15)
  (outside_triangle_ratio : ∀ (X Y: ℝ), (Area (Triangle (X) (Y) ∉ ABCD)) = 2 * (Area (Triangle (X) (Y) ∈ ABCD))):
  (Area (Quadrilateral A'B'C'D')) = 45 :=
by
  sorry

end quadrilateral_extended_area_l252_252855


namespace cos_x_geq_half_tan_A_value_l252_252075

-- Define the problem and solution for the trigonometric inequality
theorem cos_x_geq_half (x : ℝ) (k : ℤ) : 
  (cos x ≥ 1/2) ↔ (∃ k : ℤ, -π/3 + 2*k*π ≤ x ∧ x ≤ π/3 + 2*k*π) :=
sorry

-- Define the conditions for triangle problem and find tan A given sin A + cos A = sqrt(2)/2
theorem tan_A_value (A : ℝ) (h : sin A + cos A = sqrt 2 / 2) : 
  tan A = -2 - sqrt 3 :=
sorry

end cos_x_geq_half_tan_A_value_l252_252075


namespace same_increasing_interval_max_k_value_l252_252555

-- Definitions for functions f and g
def f (x a : ℝ) : ℝ := Real.exp (2 * x) - 2 * a * Real.exp x + 2 * a ^ 2
def g (x a k : ℝ) : ℝ := 2 * a * Real.log x - (Real.log x)^2 + k^2 / 8

-- Problem statement (part 1)
theorem same_increasing_interval (a : ℝ) : 
  ∀ x : ℝ, (differentiation condition and inequality) -- Specific formulation required based on derivatives pre-calculation
sorry

-- Problem statement (part 2)
theorem max_k_value (a : ℝ) (x : ℝ) (k : ℝ) (h1 : 0 < x)
  (h2 : ∀ a ∈ ℝ, f x a > g x a k) : k ≤ 4 :=
sorry

end same_increasing_interval_max_k_value_l252_252555


namespace positive_difference_of_solutions_l252_252895

theorem positive_difference_of_solutions {x : ℝ} (h : (5 - x^2 / 4)^(1/3) = -3) : 
  let sqrt128 := real.sqrt 128 in
  |sqrt128 - (-sqrt128)| = 16 * real.sqrt 2 :=
by
  let sqrt128 := real.sqrt 128
  have h1 : 5 - x^2 / 4 = -27 := sorry
  have h2 : x^2 = 128 := sorry
  have h3 : sqrt128 = real.sqrt 128 := by sorry
  have h4 : |sqrt128 + sqrt128| = 2 * sqrt128 := by sorry
  rw [h4]
  rw [sqrt128]
  sorry

end positive_difference_of_solutions_l252_252895


namespace sum_of_invalid_domain_of_g_l252_252131

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + (1 / (3 + (1 / x))))

theorem sum_of_invalid_domain_of_g : 
  (0 : ℝ) + (-1 / 3) + (-2 / 7) = -13 / 21 :=
by
  sorry

end sum_of_invalid_domain_of_g_l252_252131


namespace min_area_inscribed_triangle_fraction_l252_252252

-- Conditions
variables {A B C D E F : Type} [Nonempty A] [Nonempty B] [Nonempty C]
variables (hABC : IsoscelesRightTriangle A B C)
variables (hDEF : InscribedIsoscelesRightTriangle (triangle ABC) D E F)

-- Question and Answer
theorem min_area_inscribed_triangle_fraction
  (hABC : IsoscelesRightTriangle A B C)
  (hDEF : InscribedIsoscelesRightTriangle (triangle ABC) D E F) :
  min_area (triangle DEF) = (1 / 5) * area (triangle ABC) :=
sorry

end min_area_inscribed_triangle_fraction_l252_252252


namespace sum_odd_integers_correct_l252_252781

def sum_odd_integers_from_13_to_41 : ℕ := 
  let a := 13
  let l := 41
  let n := 15
  n * (a + l) / 2

theorem sum_odd_integers_correct : sum_odd_integers_from_13_to_41 = 405 :=
  by sorry

end sum_odd_integers_correct_l252_252781


namespace train_length_l252_252060

theorem train_length (speed_km_hr : ℝ) (time_s : ℝ) (length_of_train : ℝ) :
  speed_km_hr = 60 → time_s = 36 → length_of_train = 600.12 → 
  (speed_km_hr * (1000/3600) * time_s) = length_of_train := 
by {
  intros h_speed h_time h_length,
  subst h_speed,
  subst h_time,
  subst h_length,
  sorry
}

end train_length_l252_252060


namespace maximize_perimeter_l252_252428

theorem maximize_perimeter 
  (l : ℝ) (c_f : ℝ) (C : ℝ) (b : ℝ)
  (hl: l = 400) (hcf: c_f = 5) (hC: C = 1500) :
  ∃ (y : ℝ), y = 180 :=
by
  sorry

end maximize_perimeter_l252_252428


namespace find_circle_equation_l252_252490

noncomputable def circle_equation {x y : ℝ} (a b r : ℝ) := (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2

theorem find_circle_equation (a b r : ℝ) (h₀ : circle_equation 3 (-1) r)
  (h₁ : circle_equation 1 2 r)
  (h₂ : circle_equation (-1) (3) (Real.sqrt 5 + r)) :
  a = 20/7 ∧ b = 15/14 ∧ r^2 = 845/196 := sorry

end find_circle_equation_l252_252490


namespace tangent_normal_equations_l252_252402

open Real

noncomputable def tangent_normal_lines (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a * (sin t) ^ 3, a * (cos t) ^ 3)

theorem tangent_normal_equations (a t_0 : ℝ) (h : t_0 = π / 3) :
  let x_0 := a * (sin t_0) ^ 3,
      y_0 := a * (cos t_0) ^ 3,
      m_tangent := -cot t_0,
      m_normal := tan t_0 in
  -- equation of tangent line
  (∀ x y, y - y_0 = m_tangent * (x - x_0) → 
           y = - 1 / (√3) * x + a / 2) ∧
  -- equation of normal line
  (∀ x y, y - y_0 = -1 / m_tangent * (x - x_0) → 
           y = √3 * x - a) :=
by
  have h0 : t_0 = π / 3 := h
  sorry

end tangent_normal_equations_l252_252402


namespace sum_of_rational_roots_l252_252159

theorem sum_of_rational_roots :
  (∑ r in (multiset.filter is_rat (multiset.of_list (roots (λ x, x^3 - 8*x^2 + 13*x - 6)))), r) = 8 :=
sorry

end sum_of_rational_roots_l252_252159


namespace abs_sum_of_coef_of_poly_l252_252977

theorem abs_sum_of_coef_of_poly :
  let x := (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6) in
  ∃ a b c d : ℤ, 
    (x^4 + a*x^3 + b*x^2 + c*x + (d: ℝ) = 0) ∧
    |a + b + c + d| = 93 :=
sorry

end abs_sum_of_coef_of_poly_l252_252977


namespace distance_from_origin_to_point_l252_252996

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Math.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_origin_to_point :
  distance 0 0 20 21 = 29 := by
  sorry

end distance_from_origin_to_point_l252_252996


namespace rod_mass_l252_252028

variable (ρ : ℝ → ℝ)
variable (a b : ℝ)
variable (M : ℝ)

-- Given conditions: the density function and the length of the rod.
def density := (x : ℝ) → x^3
def length := 1

theorem rod_mass :
  (∫ x in (0 : ℝ)..(1 : ℝ), density x) = 1 / 4 :=
by
  sorry

end rod_mass_l252_252028


namespace smaller_number_l252_252737

theorem smaller_number (x y : ℝ) (h1 : y - x = (1 / 3) * y) (h2 : y = 71.99999999999999) : x = 48 :=
by
  sorry

end smaller_number_l252_252737


namespace least_number_another_factor_l252_252745

theorem least_number_another_factor {n : ℕ} (h1 : (n + 6) = 858) 
  (h2 : ∀ k, k ∈ [24, 32, 36] → (n + 6) % k = 0) : 
  (n = 852) ∧ ((852 % 71 = 0) ∧ (71 ∉ [24, 32, 36])) :=
by
  have : n = 852 := by
    rw [←h1]
    norm_num
  exact ⟨this, by sorry⟩

end least_number_another_factor_l252_252745


namespace sum_of_integer_values_l252_252036

theorem sum_of_integer_values (n : ℕ) :
  (∑ n in {n : ℕ | (nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16)}, id n) = 30 :=
by
  sorry

end sum_of_integer_values_l252_252036


namespace lateral_surface_area_prism_l252_252370

-- Define the volume and height of the prism
def volume : ℝ := 8
def height : ℝ := 2.2

-- The formula for the lateral surface area of a regular octagonal prism
def lateral_surface_area := 16 * Real.sqrt(2.2 * (Real.sqrt 2 - 1))

-- The theorem statement
theorem lateral_surface_area_prism 
  (V : ℝ := 8) 
  (H : ℝ := 2.2) 
  : lateral_surface_area = 16 * Real.sqrt(2.2 * (Real.sqrt 2 - 1)) := 
sorry

end lateral_surface_area_prism_l252_252370


namespace i_j_k_divisible_by_p_minus_1_l252_252502

theorem i_j_k_divisible_by_p_minus_1
  (p : ℕ) (hp : nat.prime p) (hp_gt_5 : p > 5)
  (a b c : ℤ) (hab : ¬(p ∣ (a - b))) (hbc : ¬(p ∣ (b - c))) (hca : ¬(p ∣ (c - a)))
  (i j k : ℕ) (hijk_sum_div : (i + j + k) % (p - 1) = 0)
  (h_expr : ∀ x : ℤ, p ∣ (x - a) * (x - b) * (x - c) * ((x - a)^i * (x - b)^j * (x - c)^k - 1)) :
  (i % (p - 1) = 0) ∧ (j % (p - 1) = 0) ∧ (k % (p - 1) = 0) :=
sorry

end i_j_k_divisible_by_p_minus_1_l252_252502


namespace average_of_side_lengths_of_squares_l252_252695

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252695


namespace sin_product_identity_l252_252853

theorem sin_product_identity :
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * (Real.sin (72 * Real.pi / 180)) = 1 / 16 := 
by 
  sorry

end sin_product_identity_l252_252853


namespace exist_empty_cell_after_move_l252_252595

-- Define the basic setup of the board and bugs
def board_size : ℕ := 5

def is_black (x y : ℕ) : Prop :=
  (x + y) % 2 = 0

def init_bug_pos (x y : ℕ) : Prop :=
  0 ≤ x ∧ x < board_size ∧ 0 ≤ y ∧ y < board_size

def move (x y : ℕ) : set (ℕ × ℕ) :=
  {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}

-- Prove that there is at least one empty cell after the bugs move
theorem exist_empty_cell_after_move :
  ∃ (x y : ℕ), init_bug_pos x y ∧ ¬∃ (x' y' : ℕ), (x', y') ∈ move x y :=
sorry

end exist_empty_cell_after_move_l252_252595


namespace isosceles_triangle_in_circle_l252_252179

-- Definitions for geometrical constructs
variables {O P : Point} {r : ℝ}

-- The condition PO >= r * sin(22.5 degrees).
-- Use the trigonometric identity sin(22.5°) = (sqrt(2) - 1) / sqrt(2)
def condition (PO r : ℝ) : Prop :=
  PO >= r * ((Real.sqrt 2 - 1) / Real.sqrt 2)

-- The main statement to prove
theorem isosceles_triangle_in_circle (h : condition (dist P O) r) : 
  ∃ (T1 T2 T3 T4 : Triangle), 
    (∀ (T : Triangle), T ∈ {T1, T2, T3, T4} → 
        is_isosceles T ∧ vertex_angle T = 45 ∧ 
        leg_through_point T P ∧ 
        inscribed_in_circle T O r) ∧ 
    (card {T1, T2, T3, T4} = 4) :=
sorry

end isosceles_triangle_in_circle_l252_252179


namespace bisectors_intersect_l252_252736

variables {A B C D X : Type}
variables [affine_space A B C D X]

-- Defining the basic points and segments for quadrilateral ABCD
variables (AB BC CD AD : A)
variables (BA BX CX)

-- Define conditions
axiom h1 : AB + CD = BC    -- Given condition

-- Define points: Additional information derived from the problem
axiom h2 : BA = BX          -- By problem definition, BA = BX on BC
axiom h3 : CX = CD          -- Implies CX = CD

-- Definitions for the bisectors
noncomputable def angle_bisector_B := (angle_bisector BA BX) -- Bisector of angle B
noncomputable def angle_bisector_C := (angle_bisector CX CD) -- Bisector of angle C
noncomputable def perp_bisector_AD := (perp_bisector AD)      -- Perpendicular bisector of AD

-- Goal to prove
theorem bisectors_intersect :
  ∃ P, P ∈ angle_bisector_B ∧ P ∈ angle_bisector_C ∧ P ∈ perp_bisector_AD :=
sorry

end bisectors_intersect_l252_252736


namespace monotonic_intervals_h_unique_tangent_point_l252_252935

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) : ℝ := f x - (x + 1) / (x - 1)

-- Problem 1: Monotonic intervals of \( h(x) \)
theorem monotonic_intervals_h :
  ∀ x : ℝ, (0 < x ∧ x < 1) ∨ (1 < x) → (h (x + 1)) > h x :=
sorry

-- Problem 2: Existence and uniqueness of \( x_0 \) such that the tangent line to \( f \) at \( x_0 \) is tangent to \( g \)
theorem unique_tangent_point :
  ∃! x0 : ℝ, (0 < x0) ∧ (line_tangent_to f x0 x0 = line_tangent_to g (Real.log x0) x0) :=
sorry

end monotonic_intervals_h_unique_tangent_point_l252_252935


namespace find_quadratic_polynomial_l252_252903

-- Given conditions
variables {P : ℝ → ℝ}
variable (x : ℝ)
hypothesis (h1 : P(3+4) = 0)
hypothesis (h2 : P(3-4) = 0)
hypothesis (h3 : is_quadratic P)
hypothesis (h4 : const_term(P) = -10)

-- Show that the polynomial P(x) = -x^2 + 6x - 10
theorem find_quadratic_polynomial (h1 : P (3 + 4 * complex.I) = 0) (h2 : P (3 - 4 * complex.I) = 0)
    (h3 : is_quadratic P) (h4 : constant_term P = -10) :
    ∃ P, P (x) = - x ^ 2 + 6 * x - 10 :=
sorry

end find_quadratic_polynomial_l252_252903


namespace sum_of_binom_solutions_l252_252038

theorem sum_of_binom_solutions :
  (∀ n : ℕ, (n = 14 ∨ n = 16) → (nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16)) →
  (14 + 16 = 30) :=
by
  intros h
  sorry

end sum_of_binom_solutions_l252_252038


namespace sum_of_integer_values_l252_252035

theorem sum_of_integer_values (n : ℕ) :
  (∑ n in {n : ℕ | (nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16)}, id n) = 30 :=
by
  sorry

end sum_of_integer_values_l252_252035


namespace max_average_growth_rate_l252_252005

theorem max_average_growth_rate 
  (P1 P2 : ℝ) (M : ℝ)
  (h1 : P1 + P2 = M) : 
  (1 + (M / 2))^2 ≥ (1 + P1) * (1 + P2) := 
by
  -- AM-GM Inequality application and other mathematical steps go here.
  sorry

end max_average_growth_rate_l252_252005


namespace sqrt_sum_inequality_l252_252641

theorem sqrt_sum_inequality (x y a b : ℝ) (hx : x^2 + y^2 = 1) (ha : 0 < a) (hb : 0 < b) :
  sqrt (a^2 * x^2 + b^2 * y^2) + sqrt (a^2 * y^2 + b^2 * x^2) ≥ a + b :=
sorry

end sqrt_sum_inequality_l252_252641


namespace overtime_rate_percentage_increase_l252_252411

noncomputable def regular_rate : ℝ := 14
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours_worked : ℝ := 57.224489795918366
noncomputable def total_earnings : ℝ := 982

theorem overtime_rate_percentage_increase :
    let regular_earnings := regular_hours * regular_rate,
        overtime_hours := total_hours_worked - regular_hours,
        overtime_earnings := total_earnings - regular_earnings,
        overtime_rate_per_hour := overtime_earnings / overtime_hours,
        percentage_increase := ((overtime_rate_per_hour - regular_rate) / regular_rate) * 100
    in percentage_increase = 75 := by
    sorry

end overtime_rate_percentage_increase_l252_252411


namespace abs_pi_expression_eq_l252_252120

theorem abs_pi_expression_eq (h : Real.pi < 3.5) : 
  |Real.pi - |Real.pi - 7|| = 7 - 2 * Real.pi :=
by 
  sorry

end abs_pi_expression_eq_l252_252120


namespace product_of_integers_with_given_pair_sums_l252_252501

theorem product_of_integers_with_given_pair_sums :
  ∃ (a b c d e : ℤ),
    set_of_pairs {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = {3, 8, 9, 16, 17, 17, 18, 22, 23, 31} ∧
    (a * b * c * d * e) = 3360 := 
sorry

end product_of_integers_with_given_pair_sums_l252_252501


namespace isosceles_triangle_base_l252_252865

theorem isosceles_triangle_base (t : ℝ) (α : ℝ) (h1 : α ≠ 0) (h2 : α ≠ π) (h3 : 0 < t):
  ∃ a : ℝ, a = 2 * sqrt (t * tan (α / 2)) :=
by
  use 2 * sqrt (t * tan (α / 2))
  sorry

end isosceles_triangle_base_l252_252865


namespace train_length_is_140_meters_l252_252100

-- Define the speed in kilometers per hour
def speed_kmph : ℝ := 98

-- Define the time taken to pass an electric pole in seconds
def time_seconds : ℝ := 5.142857142857143

-- Define the conversion factor from kmph to m/s
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Calculate the speed in meters per second
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Calculate the length of the train
def length_of_train (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem stating the length of the train is approximately 140 meters
theorem train_length_is_140_meters : 
  length_of_train speed_mps time_seconds ≈ 140 := 
sorry

end train_length_is_140_meters_l252_252100


namespace distance_between_vertices_of_hyperbola_l252_252150

def hyperbola_equation (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ), c₁ = 4 ∧ c₂ = -4 ∧
    (c₁ * x^2 + 24 * x + c₂ * y^2 + 8 * y + 44 = 0)

theorem distance_between_vertices_of_hyperbola :
  (∀ x y : ℝ, hyperbola_equation x y) → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end distance_between_vertices_of_hyperbola_l252_252150


namespace handshake_max_participants_l252_252597

theorem handshake_max_participants (N : ℕ) (hN : 5 < N) (hNotAllShaken: ∃ p1 p2 : ℕ, p1 ≠ p2 ∧ p1 < N ∧ p2 < N ∧ (∀ i : ℕ, i < N → i ≠ p1 → i ≠ p2 → ∃ j : ℕ, j < N ∧ j ≠ i ∧ j ≠ p1 ∧ j ≠ p2)) :
∃ k, k = N - 2 :=
by
  sorry

end handshake_max_participants_l252_252597


namespace rectangle_area_change_l252_252241

theorem rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A := L * B
      L' := L * 1.20
      B' := B * 0.85
      A' := L' * B'
  in A' = A * 1.02 :=
by
  -- Proof goes here
  sorry

end rectangle_area_change_l252_252241


namespace math_proof_problem_l252_252296

noncomputable def find_value (a b c : ℝ) : ℝ :=
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c))

theorem math_proof_problem (a b c : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a * b + a * c + b * c ≠ 0) :
  find_value a b c = 3 :=
by 
  -- sorry is used as we are only asked to provide the theorem statement in Lean.
  sorry

end math_proof_problem_l252_252296


namespace find_a_l252_252218

noncomputable def function_domain_is_correct (a : ℝ) : Prop :=
∀ x : ℝ, x ≤ 1 → a * 9^x + 3^x + 1 ≥ 0

theorem find_a (a : ℝ) : function_domain_is_correct a → a = -4/9 :=
begin
  intro h,
  sorry
end

end find_a_l252_252218


namespace inverse_function_exists_l252_252033

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 + 7 * x - x^2

-- Define the candidate for the inverse function g(x)
def g (x : ℝ) : ℝ := (7 + real.sqrt (37 + 4 * x)) / 2

-- Define the inverse property to be proved
theorem inverse_function_exists : ∃ g : ℝ → ℝ, (∀ x : ℝ, f (g x) = x) := 
by
  use g
  intro x
  -- Skipping the detailed steps; assert the correctness directly
  sorry

end inverse_function_exists_l252_252033


namespace cos_pi_minus_alpha_l252_252191

theorem cos_pi_minus_alpha (α : ℝ) (hα : α > π ∧ α < 3 * π / 2) (h : Real.sin α = -5/13) :
  Real.cos (π - α) = 12 / 13 := 
by
  sorry

end cos_pi_minus_alpha_l252_252191


namespace counterexample_to_conjecture_l252_252859

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)

def is_power_of_two (k : ℕ) : Prop := ∃ m : ℕ, m > 0 ∧ k = 2 ^ m

theorem counterexample_to_conjecture :
  ∃ n : ℤ, n > 5 ∧ ¬ (3 ∣ n) ∧ ¬ (∃ p k : ℕ, is_prime p ∧ is_power_of_two k ∧ n = p + k) :=
sorry

end counterexample_to_conjecture_l252_252859


namespace angle_C_modified_l252_252856

theorem angle_C_modified (A B C : ℝ) (h_eq_triangle: A = B) (h_C_modified: C = A + 40) (h_sum_angles: A + B + C = 180) : 
  C = 86.67 := 
by 
  sorry

end angle_C_modified_l252_252856


namespace count_true_propositions_l252_252961

-- Define propositions as Boolean predicates
def prop1 (x : ℝ) : Prop := (x ≠ 1) → (x^2 - 3 * x + 2 ≠ 0)
def contrapositive_prop1 (x : ℝ) : Prop := (x^2 - 3 * x + 2 = 0) → (x = 1)
def prop2 : Prop := ∀ x : ℝ, x^2 + x + 1 ≠ 0
def neg_prop2 : Prop := ∃ x : ℝ, x^2 + x + 1 = 0
def prop3 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)
def prop4 (x : ℝ) : Prop := (x > 2) → (x^2 - 3 * x + 2 > 0)

-- Provide theorem to be proved
theorem count_true_propositions :
  (∀ x, prop1 x = contrapositive_prop1 x) ∧
  (prop2 = ¬ neg_prop2) ∧
  ¬ (∀ p q, prop3 p q) ∧
  (∀ x, (x > 2) → ((x - 1) * (x - 2) > 0)) →
  3 := sorry

end count_true_propositions_l252_252961


namespace average_of_side_lengths_of_squares_l252_252696

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252696


namespace correct_proposition_l252_252550

def curve_is_ellipse (k : ℝ) : Prop :=
  9 < k ∧ k < 25

def curve_is_hyperbola_on_x_axis (k : ℝ) : Prop :=
  k < 9

theorem correct_proposition (k : ℝ) :
  (curve_is_ellipse k ∨ ¬ curve_is_ellipse k) ∧ 
  (curve_is_hyperbola_on_x_axis k ∨ ¬ curve_is_hyperbola_on_x_axis k) →
  (9 < k ∧ k < 25 → curve_is_ellipse k) ∧ 
  (curve_is_ellipse k ↔ (9 < k ∧ k < 25)) ∧ 
  (curve_is_hyperbola_on_x_axis k ↔ k < 9) → 
  (curve_is_ellipse k ∧ curve_is_hyperbola_on_x_axis k) :=
by
  sorry

end correct_proposition_l252_252550


namespace total_cars_in_group_l252_252995

theorem total_cars_in_group (C : ℕ)
  (h1 : 37 ≤ C)
  (h2 : ∃ n ≥ 51, n ≤ C)
  (h3 : ∃ n ≤ 49, n + 51 = C - 37) :
  C = 137 :=
by
  sorry

end total_cars_in_group_l252_252995


namespace find_ordered_pair_l252_252666

theorem find_ordered_pair (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroot : ∀ x : ℝ, 2 * x^2 + a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1 / 2, -3 / 4) := 
  sorry

end find_ordered_pair_l252_252666


namespace find_x_l252_252883

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 210) : x = 14 := by
  sorry

end find_x_l252_252883


namespace arctan_sum_pi_over_four_l252_252270

theorem arctan_sum_pi_over_four (a b c : ℝ) (C : ℝ) (h : Real.sin C = c / (a + b + c)) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 :=
sorry

end arctan_sum_pi_over_four_l252_252270


namespace product_difference_eq_40199_l252_252167

-- Definitions of g and h
noncomputable def g (n : ℕ) : ℕ := ∏ i in finset.filter (λ i, odd i) (finset.range (n+1)), i
noncomputable def h (n : ℕ) : ℕ := ∏ i in finset.filter (λ i, even i) (finset.range (n+1)), i

-- The theorem statement
theorem product_difference_eq_40199 : [g 201 * h 200] - [g 199 * h 198] = 40199 := 
by
  sorry

end product_difference_eq_40199_l252_252167


namespace series_result_l252_252455

noncomputable def series_sum : ℚ := (∑ n in Finset.range 100, (3 + (n + 1) * 7) / 3^(101 - (n + 1)))

theorem series_result : series_sum = 1399 / 4 := 
by
  sorry

end series_result_l252_252455


namespace correct_statement_l252_252787

theorem correct_statement (a b c : ℝ) (h1 : ac = bc) (h2 : a = b) (h3 : a^2 = b^2) : 
  (∀ (c ≠ 0), (ac = bc → a = b)) ∧ 
  (∀ (c ≠ 0), (a / c = b / c → a = b)) ∧
  (a = b → a + 3 = b + 3) ∧ 
  (a^2 = b^2 → a = b) :=
by 
  sorry

end correct_statement_l252_252787


namespace simplify_expression_l252_252659

theorem simplify_expression (x y : ℝ) : 2 - (3 - (2 + (5 - (3 * y - x)))) = 6 - 3 * y + x :=
by
  sorry

end simplify_expression_l252_252659


namespace sum_first_n_terms_l252_252183

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_n_terms
  (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h_a2a4 : a 2 + a 4 = 8)
  (h_common_diff : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∃ S_n : ℕ → ℝ, ∀ n : ℕ, S_n n = n^2 - n :=
by 
  sorry

end sum_first_n_terms_l252_252183


namespace books_at_end_l252_252047

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l252_252047


namespace sum_of_digits_divisible_by_27_not_implies_number_divisible_by_27_l252_252619

theorem sum_of_digits_divisible_by_27_not_implies_number_divisible_by_27 :
  ∃ n : ℕ, sum_of_digits n % 27 = 0 ∧ n % 27 ≠ 0 :=
begin
  -- Using the example from the solution:
  let n := 9918,
  have h1 : sum_of_digits n = 27 := by sorry, -- Sum of digits of 9918 is 27
  have h2 : 27 % 27 = 0 := by norm_num, -- 27 is divisible by 27
  have h3 : n % 27 ≠ 0 := by sorry, -- 9918 is not divisible by 27
  use n,
  exact ⟨h1, h3⟩
end

end sum_of_digits_divisible_by_27_not_implies_number_divisible_by_27_l252_252619


namespace complex_addition_l252_252916

-- Define the problem conditions in Lean
variables (a b : ℝ)
noncomputable def i : ℂ := complex.I

-- The main theorem to prove
theorem complex_addition (h : (a + 2 * i) * i = b + i) : a + b = sorry :=
by
-- Automatic unfolding of complex multiplication and equality should be here.
sorry

end complex_addition_l252_252916


namespace arithmetic_sequence_sum_eq_l252_252586

variable {α : Type} [LinearOrderedAddCommGroup α] [Module ℝ α]

theorem arithmetic_sequence_sum_eq (a : ℕ → α) (d : α)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 5 + a 6 + a 7 = 15) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
sorry

end arithmetic_sequence_sum_eq_l252_252586


namespace probability_of_720_product_l252_252784

theorem probability_of_720_product (a b c d : ℕ) (h₁ : a ∈ {1, 2, 3, 4, 5, 6})
  (h₂ : b ∈ {1, 2, 3, 4, 5, 6}) (h₃ : c ∈ {1, 2, 3, 4, 5, 6}) (h₄ : d ∈ {1, 2, 3, 4, 5, 6}) :
  (∃ (a b c d : ℕ), abcd = 720 ∧ 
  a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ d ∈ {1, 2, 3, 4, 5, 6}) →
  ((∃ (a b c d : ℕ), a*b*c*d = 720 ∧ 
      ∀ x ∈ {a, b, c, d}, x ∈ {1, 2, 3, 4, 5, 6}) → 
      ∑ (x : {a, b, c, d} | x ∈ {1, 2, 3, 4, 5, 6}), 1 / (6^4) = 1 / 324) :=
begin
  sorry
end

end probability_of_720_product_l252_252784


namespace polygon_area_is_8_l252_252890

theorem polygon_area_is_8 : 
  let vertices : List (ℝ × ℝ) := [(0, 1), (2, 0), (4, 1), (2, 4)]
  ∃ area : ℝ, area = 8 ∧ 
  area = 
    0.5 * 
    (abs (
      (vertices.nth 0).getOrElse (0, 0).fst * (vertices.nth 1).getOrElse (0, 0).snd +
      (vertices.nth 1).getOrElse (0, 0).fst * (vertices.nth 2).getOrElse (0, 0).snd +
      (vertices.nth 2).getOrElse (0, 0).fst * (vertices.nth 3).getOrElse (0, 0).snd +
      (vertices.nth 3).getOrElse (0, 0).fst * (vertices.nth 0).getOrElse (0, 0).snd -
      (vertices.nth 0).getOrElse (0, 0).snd * (vertices.nth 1).getOrElse (0, 0).fst -
      (vertices.nth 1).getOrElse (0, 0).snd * (vertices.nth 2).getOrElse (0, 0).fst -
      (vertices.nth 2).getOrElse (0, 0).snd * (vertices.nth 3).getOrElse (0, 0).fst -
      (vertices.nth 3).getOrElse (0, 0).snd * (vertices.nth 0).getOrElse (0, 0).fst)) :=
begin
  sorry
end

end polygon_area_is_8_l252_252890


namespace lines_concurrent_at_O_l252_252942

-- Variables for planes
variables {α β γ : Plane}

-- Variables for lines which are intersections of planes
variables (c : Line) (a : Line) (b : Line)

-- Variable for the intersection point
variables (O : Point)

-- Conditions provided in the problem
axiom plane_intersection_1 : α ∩ β = c
axiom plane_intersection_2 : β ∩ γ = a
axiom plane_intersection_3 : α ∩ γ = b
axiom line_intersection : a ∩ b = O

-- The theorem we need to prove
theorem lines_concurrent_at_O : (c ∩ a = O) ∧ (c ∩ b = O) ∧ (a ∩ b = O) :=
by
  sorry

end lines_concurrent_at_O_l252_252942


namespace complement_set_l252_252548

variable U : Set ℝ := {x : ℝ | true}
variable M : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}

theorem complement_set :
  (U \ M) = {x : ℝ | x < 0 ∨ x > 2} :=
by
  sorry

end complement_set_l252_252548


namespace natasha_exercises_30_minutes_daily_l252_252328

theorem natasha_exercises_30_minutes_daily (N : ℕ) :
  (∃ N, (7 * N + 10 * 9 = 5 * 60)) → N = 30 :=
by
  intro h
  cases h with m hm
  have h_eq : 7 * m + 90 = 300 := hm
  sorry

end natasha_exercises_30_minutes_daily_l252_252328


namespace select_one_from_departments_l252_252024

theorem select_one_from_departments (a b c : ℕ) (h_a : a = 2) (h_b : b = 4) (h_c : c = 3) : a + b + c = 9 :=
by
  rw [h_a, h_b, h_c]
  rfl

end select_one_from_departments_l252_252024


namespace complex_number_solution_l252_252921

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) / z = 2 + I) :
  im z = -1 ∧ abs z = Real.sqrt 2 ∧ z ^ 6 = -8 * I :=
by
  sorry

end complex_number_solution_l252_252921


namespace find_prime_and_integer_l252_252485

theorem find_prime_and_integer (p x : ℕ) (hp : Nat.Prime p) 
  (hx1 : 1 ≤ x) (hx2 : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (p, x) = (2, 1) ∨ (p, x) = (2, 2) ∨ (p, x) = (3, 1) ∨ (p, x) = (3, 3) ∨ ((p ≥ 5) ∧ (x = 1)) :=
by
  sorry

end find_prime_and_integer_l252_252485


namespace ball_center_distance_l252_252076

-- Definition of the given conditions
def diameter : ℝ := 6
def radius_of_ball : ℝ := diameter / 2
def R1 : ℝ := 150
def R2 : ℝ := 200
def R3 : ℝ := 120
def R1' : ℝ := R1 - radius_of_ball
def R2' : ℝ := R2 + radius_of_ball
def R3' : ℝ := R3 - radius_of_ball

-- Axiom stating the distance traveled by the center of the ball in each arc.
axiom arc_distance (R : ℝ) : ℝ := (1/2) * (2 * Real.pi * R)

-- Define the distance the center of the ball travels
def traveled_distance : ℝ := arc_distance R1' + arc_distance R2' + arc_distance R3'

-- The theorem to be proven
theorem ball_center_distance : traveled_distance = 467 * Real.pi := 
  by sorry

end ball_center_distance_l252_252076


namespace rectangles_arrangement_exists_l252_252618

structure Rectangle :=
(vertices : set (ℝ × ℝ))
(vertex_count : vertices.finite ∧ vertices.to_finset.card = 4)  -- Assuming rectangle has four vertices

def share_exactly_one_vertex (R1 R2 : Rectangle) : Prop := 
  (R1.vertices ∩ R2.vertices).finite ∧ (R1.vertices ∩ R2.vertices).to_finset.card = 1

def no_single_common_vertex (rects : list Rectangle) : Prop :=
  ∀ v ∈ ⋂ (R : Rectangle) in rects, R.vertices, false

theorem rectangles_arrangement_exists : 
  ∃ R1 R2 R3 R4 : Rectangle, 
    (no_single_common_vertex [R1, R2, R3, R4]) ∧
    (share_exactly_one_vertex R1 R2) ∧
    (share_exactly_one_vertex R1 R3) ∧
    (share_exactly_one_vertex R1 R4) ∧
    (share_exactly_one_vertex R2 R3) ∧
    (share_exactly_one_vertex R2 R4) ∧
    (share_exactly_one_vertex R3 R4) := 
  by
    sorry

end rectangles_arrangement_exists_l252_252618


namespace count_negative_rationals_is_two_l252_252112

theorem count_negative_rationals_is_two :
  let a := (-1 : ℚ) ^ 2007
  let b := (|(-1 : ℚ)| ^ 3)
  let c := -(1 : ℚ) ^ 18
  let d := (18 : ℚ)
  (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) = 2 := by
  sorry

end count_negative_rationals_is_two_l252_252112


namespace density_calculations_l252_252761

variable (p p1 p2 d1 : ℝ)

noncomputable def density_object : ℝ := (d1 * p) / (p - p1)
noncomputable def density_second_liquid : ℝ := (d1 * (p - p2)) / (p - p1)

theorem density_calculations :
  (density_object p p1 d1 = (d1 * p) / (p - p1)) ∧
  (density_second_liquid p p1 p2 d1 = (d1 * (p - p2)) / (p - p1)) :=
by
  unfold density_object density_second_liquid
  split
  . rfl
  . rfl

end density_calculations_l252_252761


namespace relationship_between_abc_l252_252300

-- Needed to make noncomputable functions due to the transcendental nature
noncomputable def f : ℝ → ℝ := sorry

theorem relationship_between_abc :
  (∀ x : ℝ, f x = f (-x)) -- condition 1: f is even
  ∧ (∀ x y : ℝ, x < y ∧ y < 0 → f x < f y) -- condition 2: f is increasing on (-∞, 0)
  → let a := f (Real.log (1 / 4)) 
    let b := f (Real.cos 2)
    let c := f (2 ^ 1.2)
      in c < a ∧ a < b := sorry

end relationship_between_abc_l252_252300


namespace stickers_distribution_l252_252228

theorem stickers_distribution :
  (finset.card {p : multiset (fin 5) | p.sum = 10 ∧ p.card ≤ 5}) = 29 :=
by
  -- we'll skip the proof with sorry
  sorry

end stickers_distribution_l252_252228


namespace number_of_factors_of_28350_l252_252974

-- State the given condition as a definition.
def n : ℕ := 28350
def prime_factorization (n : ℕ) : Prop :=
  n = 5^2 * 2 * 3^3 * 7

-- State the main question to prove.
theorem number_of_factors_of_28350 :
  prime_factorization n → (factors_count n = 48) :=
by
  sorry

-- Define a function to calculate the number of factors from the prime factorization.
noncomputable def factors_count (n : ℕ) : ℕ :=
  let pf := (5, 2) :: (2, 1) :: (3, 3) :: (7, 1) :: []
  pf.foldl (λ acc (p, e), acc * (e + 1)) 1

end number_of_factors_of_28350_l252_252974


namespace journey_possibility_l252_252166

noncomputable def possible_start_cities 
  (routes : List (String × String)) 
  (visited : List String) : List String :=
sorry

theorem journey_possibility :
  possible_start_cities 
    [("Saint Petersburg", "Tver"), 
     ("Yaroslavl", "Nizhny Novgorod"), 
     ("Moscow", "Kazan"), 
     ("Nizhny Novgorod", "Kazan"), 
     ("Moscow", "Tver"), 
     ("Moscow", "Nizhny Novgorod")]
    ["Saint Petersburg", "Tver", "Yaroslavl", "Nizhny Novgorod", "Moscow", "Kazan"] 
  = ["Saint Petersburg", "Yaroslavl"] :=
sorry

end journey_possibility_l252_252166


namespace area_BDE_l252_252305

-- We define a setup where the necessary conditions and point locations are specified.
def circle_centered_at_O (O B A C : Type) [normed_group O] [normed_space ℝ O] 
  (h_circle : ∃ R, ∀ P, P ∈ Metric.sphere O R ↔ ∃ x, x = B) 
  (diam_eq : Metric.dist A C = 24) := True

-- Prove that the area of triangle BDE is 45 under the given conditions
theorem area_BDE (O B A C D E : Type) [normed_group O] [normed_space ℝ O]
  (h_circle : ∃ R, ∀ P, P ∈ Metric.sphere O R ↔ ∃ x, x = B)
  (diam_eq : Metric.dist A C = 24)
  (circumcenter_OAB : ∀ (A B : O), OAB = D)
  (circumcenter_OBC : ∀ (B C : O), OBC = E)
  (sin_angle_BOC : Real.sin (Real.angle (B - O) (C - O)) = 4 / 5):
  ∃ area, area = 45 := 
by
  sorry

end area_BDE_l252_252305


namespace tetrahedrons_not_necessarily_similar_l252_252565

-- Introducing definitions for Tetrahedra and face similarity
structure Tetrahedron :=
(faces : fin 4 → set (fin 3 → ℝ))

def faces_similar (f1 f2 : set (fin 3 → ℝ)) :=
∃ (r : ℝ) (h : r > 0), ∀ (x : fin 3 → ℝ), x ∈ f1 ↔ (λ i, r * (f2 x) i) ∈ f2

def all_faces_similar (T1 T2 : Tetrahedron) :=
∀ i, ∃ j, faces_similar (T1.faces i) (T2.faces j)

def no_internal_faces_similar (T : Tetrahedron) :=
∀ i j, i ≠ j → ¬ faces_similar (T.faces i) (T.faces j)

-- The main theorem
theorem tetrahedrons_not_necessarily_similar (T1 T2 : Tetrahedron) :
  all_faces_similar T1 T2 → 
  no_internal_faces_similar T1 → 
  no_internal_faces_similar T2 → 
  ¬ (T1 = T2) :=
by
  sorry

end tetrahedrons_not_necessarily_similar_l252_252565


namespace minute_hand_angle_l252_252054

theorem minute_hand_angle (minutes : ℕ) (total_minutes_in_circle : ℕ) (circle_radians : ℝ)
  (h1 : total_minutes_in_circle = 60)
  (h2 : circle_radians = 2 * Real.pi)
  (h3 : minutes = 10) :
  let angle := (minutes * circle_radians) / total_minutes_in_circle in
  angle = Real.pi / 3 := 
by
  sorry

end minute_hand_angle_l252_252054


namespace positive_difference_equation_l252_252898

noncomputable def positive_difference_solutions : ℝ :=
  let eq1 := (5 - (x : ℝ)^2 / 4)^(1 / 3) = -3
  let eq2 := x^2 = 128
  16 * Real.sqrt 2

theorem positive_difference_equation :
  (5 - x^2 / 4)^(1 / 3) = -3 → x = 8 * Real.sqrt 2 - (-8 * Real.sqrt 2) :=
by
  intro h
  sorry

end positive_difference_equation_l252_252898


namespace solution_to_system_l252_252888

theorem solution_to_system (x y z : ℝ) (h1 : x^2 + y^2 = 6 * z) (h2 : y^2 + z^2 = 6 * x) (h3 : z^2 + x^2 = 6 * y) :
  (x = 3) ∧ (y = 3) ∧ (z = 3) :=
sorry

end solution_to_system_l252_252888


namespace inequality_condition_l252_252175

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the main theorem to be proven
theorem inequality_condition (a b : ℝ) (h_a : a > 11 / 4) (h_b : b > 3 / 2) :
  (∀ x : ℝ, |x + 1| < b → |f x + 3| < a) :=
by
  -- We state the required proof without providing the steps
  sorry

end inequality_condition_l252_252175


namespace average_side_length_of_squares_l252_252716

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l252_252716


namespace range_of_sum_of_unique_prime_factors_l252_252910

-- Define the sum of unique prime factors function s(n)
def sum_of_unique_prime_factors (n : ℕ) : ℕ :=
  (nat.factors n).erase_dup.sum

-- Prove that the range of s(n) for n >= 2 is all integers >= 2
theorem range_of_sum_of_unique_prime_factors :
  {m : ℕ | ∃ n : ℕ, n ≥ 2 ∧ sum_of_unique_prime_factors(n) = m} = {m : ℕ | m ≥ 2} :=
by
  sorry

end range_of_sum_of_unique_prime_factors_l252_252910


namespace part1_min_value_part2_two_zeros_l252_252217

-- Part 1: Define the function f for general a and specifically for a = 2.
def f (a : ℝ) (x : ℝ) : ℝ := a * exp (2 * x) + (a - 2) * exp x - x

-- Prove the minimum value of f when a = 2 is (1 / 2) + log 2.
theorem part1_min_value : 
  ∃ x_min : ℝ, f 2 x_min = (1 / 2) + Real.log 2 := 
  sorry

-- Part 2: Show that for 0 < a < 1, the function f has two zeros.
theorem part2_two_zeros (a : ℝ) (h : 0 < a ∧ a < 1) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 := 
  sorry

end part1_min_value_part2_two_zeros_l252_252217


namespace probability_x_gt_2y_is_1_over_3_l252_252829

noncomputable def probability_x_gt_2y_in_rectangle : ℝ :=
  let A_rect := 6 * 1
  let A_triangle := (1/2) * 4 * 1
  A_triangle / A_rect

theorem probability_x_gt_2y_is_1_over_3 :
  probability_x_gt_2y_in_rectangle = 1 / 3 :=
sorry

end probability_x_gt_2y_is_1_over_3_l252_252829


namespace notebook_cost_l252_252824

theorem notebook_cost (s n c : ℕ) (h1 : s > 25)
                                 (h2 : n % 2 = 1)
                                 (h3 : n > 1)
                                 (h4 : c > n)
                                 (h5 : s * n * c = 2739) :
  c = 7 :=
sorry

end notebook_cost_l252_252824


namespace field_dimension_solution_l252_252086

theorem field_dimension_solution (m : ℝ) (h₁ : (3 * m + 10) * (m - 5) = 72) : m = 7 :=
sorry

end field_dimension_solution_l252_252086


namespace solve_for_t_l252_252575

theorem solve_for_t (s t u : ℝ) 
  (h1 : 12 * s + 6 * t + 3 * u = 180)
  (h2 : s = t - 2)
  (h3 : t = u + 3) : 
  t = 10.142857 :=
begin
  sorry
end

end solve_for_t_l252_252575


namespace circumsphere_radius_of_tetrahedron_l252_252184

def tetrahedron (A B C D : Type) [metric_space A B C D] (AB AC AD BD CD BC : ℝ) : Prop :=
  (AB = 2) ∧ (AC = 2) ∧ (AD = 2) ∧ (BD = 2) ∧ (CD = 2) ∧ (BC = 3)

theorem circumsphere_radius_of_tetrahedron 
{A B C D : Type} [metric_space A B C D] 
(h : tetrahedron A B C D 2 2 2 2 2 3) : 
  radius_of_circumsphere A B C D = sqrt(21) / 3 := 
sorry

end circumsphere_radius_of_tetrahedron_l252_252184


namespace triangle_area_proof_l252_252797

noncomputable def triangle_area (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  1 / 2 * (B - A).norm * (C - A).norm * Real.sin (Real.angle (B - A) (C - A))

theorem triangle_area_proof :
  let A := EuclideanSpace.ofPoint ℝ (Fin 2) ![0, 0],
      B := EuclideanSpace.ofPoint ℝ (Fin 2) ![b, 0],
      C := EuclideanSpace.ofPoint ℝ (Fin 2) ![c_x, c_y] in
  let AL := 3,
      BL := 6 * Real.sqrt 5,
      CL := 4 in
  let area := triangle_area A B C in
  area = 21 * Real.sqrt 55 / 4 :=
begin
  sorry
end

end triangle_area_proof_l252_252797


namespace base7_addition_sum_l252_252437

theorem base7_addition_sum :
  let n1 := 256
  let n2 := 463
  let n3 := 132
  n1 + n2 + n3 = 1214 := sorry

end base7_addition_sum_l252_252437


namespace at_least_three_shaded_cells_in_2x2_square_l252_252594

theorem at_least_three_shaded_cells_in_2x2_square :
  ∀ (grid : fin 5 → fin 5 → bool),
  (∑ i : fin 5, ∑ j : fin 5, if grid i j then 1 else 0) = 16 →
  ∃ (i j : fin 4), 
  (if grid i j then 1 else 0) + (if grid i (j+1) then 1 else 0) + 
  (if grid (i+1) j then 1 else 0) + (if grid (i+1) (j+1) then 1 else 0) ≥ 3 :=
by
  sorry

end at_least_three_shaded_cells_in_2x2_square_l252_252594


namespace not_right_angled_triangle_l252_252839

theorem not_right_angled_triangle (a b c : ℕ) (h : set ℕ) :
  (h = {3, 4, 5} ∨ h = {8, 15, 17} ∨ h = {7, 24, 25}) →
  a = 6 → b = 8 → c = 11 →
  a^2 + b^2 ≠ c^2 := by
sorry

end not_right_angled_triangle_l252_252839


namespace product_of_roots_l252_252198

variable {x1 x2 : ℝ}

theorem product_of_roots (h : ∀ x, -x^2 + 3*x = 0 → (x = x1 ∨ x = x2)) :
  x1 * x2 = 0 :=
by
  sorry

end product_of_roots_l252_252198


namespace irreducible_fraction_l252_252342

theorem irreducible_fraction (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end irreducible_fraction_l252_252342


namespace total_spent_correct_l252_252106

def cost_ornamental_plants : Float := 467.00
def cost_garden_tool_set : Float := 85.00
def cost_potting_soil : Float := 38.00

def discount_plants : Float := 0.15
def discount_tools : Float := 0.10
def discount_soil : Float := 0.00

def sales_tax_rate : Float := 0.08
def surcharge : Float := 12.00

def discounted_price (original_price : Float) (discount_rate : Float) : Float :=
  original_price * (1.0 - discount_rate)

def subtotal (price_plants : Float) (price_tools : Float) (price_soil : Float) : Float :=
  price_plants + price_tools + price_soil

def sales_tax (amount : Float) (tax_rate : Float) : Float :=
  amount * tax_rate

def total (subtotal : Float) (sales_tax : Float) (surcharge : Float) : Float :=
  subtotal + sales_tax + surcharge

def final_total_spent : Float :=
  let price_plants := discounted_price cost_ornamental_plants discount_plants
  let price_tools := discounted_price cost_garden_tool_set discount_tools
  let price_soil := cost_potting_soil
  let subtotal_amount := subtotal price_plants price_tools price_soil
  let tax_amount := sales_tax subtotal_amount sales_tax_rate
  total subtotal_amount tax_amount surcharge

theorem total_spent_correct : final_total_spent = 564.37 :=
  by sorry

end total_spent_correct_l252_252106


namespace atleast_half_non_divisible_l252_252629

theorem atleast_half_non_divisible (p : ℕ) [fact p.prime] [fact (p % 2 = 1)] :
  ∃ (S : finset ℕ), S.card = (p + 1) / 2 ∧ ∀ n ∈ S, 
    (∀ k : ℕ, k < p → (∑ k in finset.range p, nat.factorial k * n ^ k) % p ≠ 0) :=
sorry

end atleast_half_non_divisible_l252_252629


namespace range_of_t_range_of_f_l252_252740

open Real

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 3

theorem range_of_t (x_min x_max : ℝ) (hx_min : x_min = -1/2) (hx_max : x_max = 1/2) :
t ∈ set.Icc (2^x_min) (2^x_max) ↔ t ∈ set.Icc (1 / sqrt 2) (sqrt 2) :=
sorry

theorem range_of_f (x : ℝ) (hx : x ∈ set.Icc (-1/2) (1/2)) :
  f x ∈ set.Icc (1/2 - sqrt 2 + 3) (2 - 2 * sqrt 2 + 3) :=
sorry

end range_of_t_range_of_f_l252_252740


namespace max_value_when_k_is_1_range_of_k_no_zeros_l252_252551

-- Define the function f(x) for a given k
def f (x : ℝ) (k : ℝ) :=
  Real.log (x - 1) - k * (x - 1) + 1

-- Question 1: Prove that the maximum value of f(x) when k=1 is 0
theorem max_value_when_k_is_1 : ∀ x > 1, f x 1 ≤ 0 := 
  sorry

-- Question 2: Prove that the range of k such that f(x) has no zeros is (1, +∞)
theorem range_of_k_no_zeros : ∀ k, (1 < k) ↔ ∀ x > 1, f x k ≠ 0 :=
  sorry

end max_value_when_k_is_1_range_of_k_no_zeros_l252_252551


namespace num_correct_propositions_l252_252645

def f (x : ℝ) : ℝ := sorry

lemma even_function : ∀ x : ℝ, f x = f (-x) := sorry
lemma periodic_function : ∀ x : ℝ, f (x + 1) = -f x := sorry
lemma increasing_on_neg_one_zero : ∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x < f y := sorry

theorem num_correct_propositions :
  (1 : ℕ) + (1 : ℕ) + (1 : ℕ) = 3 :=
by
  have h1 : f (x + 2) = f x, from
    have h2 : ∀ x, f (x + 1) ≠ f x, from periodic_function,
    sorry,
  have h3 : (∀ x : ℝ, f x = f (-x)) ∧ periodic_function ∧ increasing_on_neg_one_zero, from
    sorry,
  have h4 : f 2014 = f 0, from
    sorry,
  exact Nat.add_succ 2 1
    have : ∀ x y : ℝ, 0 < x ∧ x < y ∧ y ≤ 1 → f y < f x :=
      sorry,
    exact Nat.add_succ 1 1
      sorryineq_nat_iff,
completion,
apply Nat.lt_trans,
sorry

end num_correct_propositions_l252_252645


namespace positive_difference_of_solutions_l252_252894

theorem positive_difference_of_solutions {x : ℝ} (h : (5 - x^2 / 4)^(1/3) = -3) : 
  let sqrt128 := real.sqrt 128 in
  |sqrt128 - (-sqrt128)| = 16 * real.sqrt 2 :=
by
  let sqrt128 := real.sqrt 128
  have h1 : 5 - x^2 / 4 = -27 := sorry
  have h2 : x^2 = 128 := sorry
  have h3 : sqrt128 = real.sqrt 128 := by sorry
  have h4 : |sqrt128 + sqrt128| = 2 * sqrt128 := by sorry
  rw [h4]
  rw [sqrt128]
  sorry

end positive_difference_of_solutions_l252_252894


namespace second_meet_distance_l252_252384

/-- The given problem states that two athletes (one male, one female) run back and forth on a 110-meter 
slope starting from point A at the same time, with different speeds for uphill and downhill running. 
We need to prove that the distance from point A where they meet for the second time is 330/7 meters. -/
theorem second_meet_distance :
  let AB := 110 in
  let male_uphill_speed := 3 in
  let male_downhill_speed := 5 in
  let female_uphill_speed := 2 in
  let female_downhill_speed := 3 in
  let second_meeting_distance := 330 / 7 in
  ∃ X : ℝ, X = second_meeting_distance :=
begin
  sorry
end

end second_meet_distance_l252_252384


namespace imo1983_q6_l252_252307

theorem imo1983_q6 (a b c : ℝ) (h : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
by
  sorry

end imo1983_q6_l252_252307


namespace find_sum_r_s_t_l252_252285

noncomputable def diameter_length : ℝ := 1
noncomputable def point_M : ℝ × ℝ := (-1/2, 0)
noncomputable def point_N : ℝ × ℝ := (1/2, 0)
noncomputable def point_A : ℝ × ℝ := (0, 1/2)
noncomputable def point_B : ℝ × ℝ := (cos (3 * pi / 5), sin (3 * pi / 5))
noncomputable def point_C : ℝ × ℝ := (cos θ, -sin θ) -- C lies on the other semicircular arc
noncomputable def d : ℝ := 7 - 4 * real.sqrt 3 -- The maximum length d

theorem find_sum_r_s_t :
  let r := 7
  let s := 4
  let t := 3
  r + s + t = 14 :=
by 
  sorry

end find_sum_r_s_t_l252_252285


namespace probability_single_shot_l252_252010

-- Define the event and probability given
def event_A := "shooter hits the target at least once out of three shots"
def probability_event_A : ℝ := 0.875

-- The probability of missing in one shot is q, and missing all three is q^3, 
-- which leads to hitting at least once being 1 - q^3
theorem probability_single_shot (q : ℝ) (h : 1 - q^3 = 0.875) : 1 - q = 0.5 :=
by
  sorry

end probability_single_shot_l252_252010


namespace perp_TE_BC_l252_252287

-- Definitions for the problem
variables {A D B C E P Q T : Type} [plane : EuclideanGeometry Plane]
open EuclideanGeometry

-- Assuming points A, D, B, C are on a semicircle with diameter AD
variables [circle A D B] [circle A D C] 

-- Points P and Q are defined with certain perpendicularities
variables [perpendicular P E B D] [perpendicular P B A D]
variables [perpendicular Q E A C] [perpendicular Q C A D]

-- Intersection of diagonals and lines
variable (E : intersection (line A C) (line B D))
variable (T : intersection (line B Q) (line C P))

-- Theorem to state \( \overline{TE} \perp \overline{BC} \)
theorem perp_TE_BC (h1 : closer B A C)
  (h2 : lies_on_circle_semicircle A D B)
  (h3 : lies_on_circle_semicircle A D C)
  (h4 : intersection A C E)
  (h5 : intersection B D E)
  (h6 : perpendicular (P E) (B D))
  (h7 : perpendicular (P B) (A D))
  (h8 : perpendicular (Q E) (A C))
  (h9 : perpendicular (Q C) (A D))
  (h10 : intersection (B Q) (C P) T):
  perpendicular (T E) (B C) :=
sorry

end perp_TE_BC_l252_252287


namespace find_two_digit_numbers_l252_252971

theorem find_two_digit_numbers (x y : ℕ) (hx1 : x = 2) (hy1 : y = 1) :
  let num1 := 10 * x + y
  let num2 := 10 * y + x
  (num1 / num2 = 1.75) ∧ (num1 * x = 3.5 * num2) :=
by
  have h1 : 10 * x + y = 21 := by { rw [hx1, hy1], norm_num }
  have h2 : 10 * y + x = 12 := by { rw [hx1, hy1], norm_num }
  have hquot : (21 / 12 = 1.75) := by { norm_num }
  have hprod : (21 * 2 = 3.5 * 12) := by { norm_num }
  exact ⟨hquot, hprod⟩

end find_two_digit_numbers_l252_252971


namespace number_of_rectangles_l252_252343

theorem number_of_rectangles (horizontal_lines : Fin 6) (vertical_lines : Fin 5) 
                             (point : ℕ × ℕ) (h₁ : point = (3, 4)) : 
  ∃ ways : ℕ, ways = 24 :=
by {
  sorry
}

end number_of_rectangles_l252_252343


namespace largest_of_three_l252_252468

theorem largest_of_three : 
  let a := 3 ^ (-2)
  let b := 2 ^ (1.5)
  let c := log 2 3 
  b > a ∧ b > c := 
by
  let a := 3 ^ (-2)
  let b := 2 ^ (1.5)
  let c := log 2 3 
  sorry

end largest_of_three_l252_252468


namespace line_passes_point_a_ne_zero_l252_252746

theorem line_passes_point_a_ne_zero (a : ℝ) (h1 : ∀ (x y : ℝ), (y = 5 * x + a) → (x = a ∧ y = a^2)) (h2 : a ≠ 0) : a = 6 :=
sorry

end line_passes_point_a_ne_zero_l252_252746


namespace suzy_final_books_l252_252048

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l252_252048


namespace hexagon_interior_angles_l252_252418

theorem hexagon_interior_angles
  (A B C D E F : ℝ)
  (hA : A = 90)
  (hB : B = 120)
  (hCD : C = D)
  (hE : E = 2 * C + 20)
  (hF : F = 60)
  (hsum : A + B + C + D + E + F = 720) :
  D = 107.5 := 
by
  -- formal proof required here
  sorry

end hexagon_interior_angles_l252_252418


namespace sum_of_odd_integers_13_to_41_l252_252779

theorem sum_of_odd_integers_13_to_41 : 
  (∑ k in Finset.filter (λ n, n % 2 = 1) (Finset.range 42), k) - ∑ k in Finset.filter (λ n, n % 2 = 1) (Finset.range 13), k = 405 :=
by
  sorry

end sum_of_odd_integers_13_to_41_l252_252779


namespace point_on_ellipse_ellipse_c_eq_PA_PB_constant_l252_252932

noncomputable def ellipse_center_origin_c_general :
  { x y : ℝ // x^2 / 4 + y^2 = 1 } :=
sorry

theorem point_on_ellipse :
  (1, (sqrt 3) / 2 : ℝ) ∈ { p : ℝ × ℝ // ellipse_center_origin_c_general } :=
sorry

theorem ellipse_c_eq :
  { x y : ℝ // x^2 / 4 + y^2 = 1 } :=
begin
  have h1 : 1 / 4 + (sqrt 3 / 2)^2 = 1, 
  { simp, ring },
  assumption,
end

theorem PA_PB_constant (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) :
  ∀ P : ℝ × ℝ, ∃ line_l : ℝ → ℝ,
  (∀ x ∈ { p : ℝ × ℝ // ellipse_center_origin_c_general }, 
  abs (P.fst - x.fst) ^ 2 + abs (P.snd - x.snd) ^ 2 = 5) :=
sorry

end point_on_ellipse_ellipse_c_eq_PA_PB_constant_l252_252932


namespace minimum_distance_between_A_and_B_l252_252964

theorem minimum_distance_between_A_and_B :
  ∃ (a : ℝ), let A := (a - 2, 2 * a - 2) in
              let B := (Real.exp a, Real.exp a + a) in
              dist A B = 3 * Real.sqrt 2 := 
by
  sorry

end minimum_distance_between_A_and_B_l252_252964


namespace least_possible_c_l252_252809

theorem least_possible_c 
  (a b c : ℕ) 
  (h_avg : (a + b + c) / 3 = 20)
  (h_median : b = a + 13)
  (h_ord : a ≤ b ∧ b ≤ c)
  : c = 45 :=
sorry

end least_possible_c_l252_252809


namespace trajectory_of_point_l252_252238

theorem trajectory_of_point (P : ℝ × ℝ) 
  (h1 : dist P (0, 3) = dist P (x1, -3)) :
  ∃ p > 0, (P.fst)^2 = 2 * p * P.snd ∧ p = 6 :=
by {
  sorry
}

end trajectory_of_point_l252_252238


namespace count_valid_integers_1_to_999_l252_252568

-- Define a function to count the valid integers
def count_valid_integers : Nat :=
  let digits := [1, 2, 6, 7, 9]
  let one_digit_count := 5
  let two_digit_count := 5 * 5
  let three_digit_count := 5 * 5 * 5
  one_digit_count + two_digit_count + three_digit_count

-- The theorem we want to prove
theorem count_valid_integers_1_to_999 : count_valid_integers = 155 := by
  sorry

end count_valid_integers_1_to_999_l252_252568


namespace positive_solution_of_system_l252_252968

theorem positive_solution_of_system (x y z : ℝ) (h1 : x * y = 5 - 3 * x - 2 * y)
                                    (h2 : y * z = 8 - 5 * y - 3 * z)
                                    (h3 : x * z = 18 - 2 * x - 5 * z)
                                    (hx_pos : 0 < x) : x = 6 := 
sorry

end positive_solution_of_system_l252_252968


namespace sum_positive_diff_T_l252_252293

noncomputable def sum_positive_differences (s : Finset ℕ) : ℕ :=
  s.sum (λ x, s.sum (λ y, if x > y then x - y else 0))

def T : Finset ℕ := (Finset.range 13).image (λ n, 3^n)

theorem sum_positive_diff_T : sum_positive_differences T = 8725600 :=
by
  sorry

end sum_positive_diff_T_l252_252293


namespace last_digit_of_max_possible_value_l252_252521

theorem last_digit_of_max_possible_value :
  let S : List ℕ := List.replicate 128 1
  let operation: ℕ → ℕ → ℕ := λ a b => a * b + 1
  let final_result: ℕ := (List.foldl (λ acc x => operation acc x) 1 (List.take 127 S.tail))
  let A: ℕ := final_result
  Nat.digit A 1 = 2 := sorry

end last_digit_of_max_possible_value_l252_252521


namespace cupcake_packages_l252_252071

variables (baked_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ)

def remaining_cupcakes (baked_cupcakes eaten_cupcakes : ℕ) : ℕ := 
  baked_cupcakes - eaten_cupcakes

def number_of_packages (remaining_cupcakes cupcakes_per_package : ℕ) : ℕ := 
  remaining_cupcakes / cupcakes_per_package

theorem cupcake_packages : 
  baked_cupcakes = 20 → 
  eaten_cupcakes = 11 → 
  cupcakes_per_package = 3 → 
  number_of_packages (remaining_cupcakes baked_cupcakes eaten_cupcakes) cupcakes_per_package = 3 := 
by
  intros hb he hc
  simp [remaining_cupcakes, number_of_packages, hb, he, hc]
  sorry

end cupcake_packages_l252_252071


namespace sum_factors_of_24_l252_252040

theorem sum_factors_of_24 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_factors_of_24_l252_252040


namespace books_left_on_Fri_l252_252053

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l252_252053


namespace scientific_notation_0_056_l252_252757

theorem scientific_notation_0_056 :
  (0.056 = 5.6 * 10^(-2)) :=
by
  sorry

end scientific_notation_0_056_l252_252757


namespace batsman_boundaries_l252_252811

theorem batsman_boundaries
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_by_running : ℕ)
  (runs_by_sixes : ℕ)
  (runs_by_boundaries : ℕ)
  (half_runs : ℕ)
  (sixes_runs : ℕ)
  (boundaries_runs : ℕ)
  (total_runs_eq : total_runs = 120)
  (sixes_eq : sixes = 8)
  (half_total_eq : half_runs = total_runs / 2)
  (runs_by_running_eq : runs_by_running = half_runs)
  (sixes_runs_eq : runs_by_sixes = sixes * 6)
  (boundaries_runs_eq : runs_by_boundaries = total_runs - runs_by_running - runs_by_sixes)
  (boundaries_eq : boundaries_runs = boundaries * 4) :
  boundaries = 3 :=
by
  sorry

end batsman_boundaries_l252_252811


namespace sasha_cannot_achieve_300_l252_252341

theorem sasha_cannot_achieve_300 (numbers : Fin 30 → ℕ) :
  (∀ i j, i ≠ j → numbers i ≠ numbers j) → -- all numbers are distinct
  (∀ ops : List (Fin 29), ops.length = 29 → 
    (filter (λ x, x = 1) ops).length = 2 → -- exactly 2 LCM operations (1 for LCM, 0 for GCD)
    let result := List.foldl (λ acc t, match t with
                                        | 0 => gcd acc numbers.succ.head!
                                        | 1 => lcm acc numbers.succ.head!
                                      end) (numbers 0) ops in
    result = 300) → False := -- Final result cannot always be 300
by
  intros h1 h2
  sorry

end sasha_cannot_achieve_300_l252_252341


namespace number_of_added_groups_l252_252326

-- Define the total number of students in the class
def total_students : ℕ := 47

-- Define the number of students per table and the number of tables
def students_per_table : ℕ := 3
def number_of_tables : ℕ := 6

-- Define the number of girls in the bathroom and the multiplier for students in the canteen
def girls_in_bathroom : ℕ := 3
def canteen_multiplier : ℕ := 3

-- Define the number of foreign exchange students from each country
def foreign_exchange_germany : ℕ := 3
def foreign_exchange_france : ℕ := 3
def foreign_exchange_norway : ℕ := 3

-- Define the number of students per recently added group
def students_per_group : ℕ := 4

-- Calculate the number of students currently in the classroom
def students_in_classroom := number_of_tables * students_per_table

-- Calculate the number of students temporarily absent
def students_in_canteen := girls_in_bathroom * canteen_multiplier
def temporarily_absent := girls_in_bathroom + students_in_canteen

-- Calculate the number of foreign exchange students missing
def foreign_exchange_missing := foreign_exchange_germany + foreign_exchange_france + foreign_exchange_norway

-- Calculate the total number of students accounted for
def student_accounted_for := students_in_classroom + temporarily_absent + foreign_exchange_missing

-- The proof statement (main goal)
theorem number_of_added_groups : (total_students - student_accounted_for) / students_per_group = 2 :=
by
  sorry

end number_of_added_groups_l252_252326


namespace max_n_square_sum_eq_2500_l252_252389

theorem max_n_square_sum_eq_2500 : 
  ∃ (S : Finset ℕ) (n : ℕ), 
    n = S.card ∧
    (∀ j ∈ S, 0 < j) ∧ 
    (∀ i j ∈ S, i ≠ j → i ≠ j) ∧ 
    (S.sum (λ x, x^2) = 2500) ∧
    n = 19
:= sorry

end max_n_square_sum_eq_2500_l252_252389


namespace sandbox_width_l252_252627

theorem sandbox_width (P : ℕ) (W L : ℕ) (h1 : P = 30) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : W = 5 := 
sorry

end sandbox_width_l252_252627


namespace initial_customers_l252_252102

theorem initial_customers (x : ℕ) (h1 : x - 31 + 26 = 28) : x = 33 := 
by 
  sorry

end initial_customers_l252_252102


namespace segment_of_angle_bisector_as_geometric_locus_l252_252892

def geometric_locus (A O B : Point) (a : ℝ) : Set Point :=
  { P : Point | (inside_angle A O B P) ∧ (distance_to_line P (line_through O A) + distance_to_line P (line_through O B) = a) }

theorem segment_of_angle_bisector_as_geometric_locus :
  ∀ (A O B : Point) (a : ℝ), 
  ∃ l : Line, 
    ( ∀ P : Point, (P ∈ geometric_locus A O B a ↔ P ∈ set_of_points_on_segment_of_angle_bisector l A O B a)) :=
sorry

end segment_of_angle_bisector_as_geometric_locus_l252_252892


namespace positive_diff_solutions_correct_l252_252902

noncomputable def positive_diff_solutions : ℝ :=
  let eqn : ℝ → Prop := λ x, (5 - x^2 / 4) ^ (1 / 3) = -3
  16 * real.sqrt 2

theorem positive_diff_solutions_correct : 
  ∀ x1 x2, eqn x1 ∧ eqn x2 → abs (x1 - x2) = 16 * real.sqrt 2 := sorry

end positive_diff_solutions_correct_l252_252902


namespace expIConjugate_l252_252582

open Complex

-- Define the given condition
def expICondition (θ φ : ℝ) : Prop :=
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I

-- The theorem we want to prove
theorem expIConjugate (θ φ : ℝ) (h : expICondition θ φ) : 
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
sorry

end expIConjugate_l252_252582


namespace range_neg2a_plus_3_l252_252577

theorem range_neg2a_plus_3 (a : ℝ) (h : a < 1) : -2 * a + 3 > 1 :=
sorry

end range_neg2a_plus_3_l252_252577


namespace range_of_a_l252_252519

-- Given definitions from the problem
def p (a : ℝ) : Prop :=
  (4 - 4 * a) > 0

def q (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

-- The theorem we want to prove
theorem range_of_a (a : ℝ) : ¬ (p a ∨ q a) ↔ a ≥ 3 := 
by sorry

end range_of_a_l252_252519


namespace number_of_shaded_cubes_is_seventeen_l252_252122

-- Definitions based on the conditions in the problem.

def is_shaded (x y z : ℕ) : Prop :=
  ((x = 0 ∨ x = 2) ∧ (y = 0 ∧ z = 0 ∨ y = 2 ∧ z = 2 ∨ y = 0 ∧ z = 2 ∨ y = 2 ∧ z = 0) ∨
   (y = 0 ∨ y = 2) ∧ (x = 0 ∧ z = 0 ∨ x = 2 ∧ z = 2 ∨ x = 0 ∧ z = 2 ∨ x = 2 ∧ z = 0) ∨
   (z = 0 ∨ z = 2) ∧ (x = 0 ∧ y = 0 ∨ x = 2 ∧ y = 2 ∨ x = 0 ∧ y = 2 ∨ x = 2 ∧ y = 0))

def count_shaded_cubes : ℕ :=
  ((Finset.univ.product (Finset.univ.product Finset.univ)).filter (λ ⟨x, ⟨y, z⟩⟩, is_shaded x y z)).card

theorem number_of_shaded_cubes_is_seventeen :
  count_shaded_cubes = 17 :=
by
  sorry

end number_of_shaded_cubes_is_seventeen_l252_252122


namespace arithmetic_seq_term_ratio_l252_252225

-- Assume two arithmetic sequences a and b
def arithmetic_seq_a (n : ℕ) : ℕ := sorry
def arithmetic_seq_b (n : ℕ) : ℕ := sorry

-- Sum of first n terms of the sequences
def sum_a (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_a |>.sum
def sum_b (n : ℕ) : ℕ := (List.range (n+1)).map arithmetic_seq_b |>.sum

-- The given condition: Sn / Tn = (7n + 2) / (n + 3)
axiom sum_condition (n : ℕ) : (sum_a n) / (sum_b n) = (7 * n + 2) / (n + 3)

-- The goal: a4 / b4 = 51 / 10
theorem arithmetic_seq_term_ratio : (arithmetic_seq_a 4 : ℚ) / (arithmetic_seq_b 4 : ℚ) = 51 / 10 :=
by
  sorry

end arithmetic_seq_term_ratio_l252_252225


namespace distance_between_x_intercepts_l252_252822

noncomputable def slope_intercept_form (m : ℝ) (x1 y1 x : ℝ) : ℝ :=
  m * (x - x1) + y1

def x_intercept (m : ℝ) (x1 y1 : ℝ) : ℝ :=
  (y1 - m * x1) / m

theorem distance_between_x_intercepts : 
  ∀ (m1 m2 : ℝ) (x1 y1 : ℝ), 
  m1 = 4 → m2 = -2 → x1 = 8 → y1 = 20 →
  abs (x_intercept m1 x1 y1 - x_intercept m2 x1 y1) = 15 :=
by
  intros m1 m2 x1 y1 h_m1 h_m2 h_x1 h_y1
  rw [h_m1, h_m2, h_x1, h_y1]
  sorry

end distance_between_x_intercepts_l252_252822


namespace popsicles_consumed_l252_252331

def total_minutes (hours : ℕ) (additional_minutes : ℕ) : ℕ :=
  hours * 60 + additional_minutes

def popsicles_in_time (total_time : ℕ) (interval : ℕ) : ℕ :=
  total_time / interval

theorem popsicles_consumed : popsicles_in_time (total_minutes 4 30) 15 = 18 :=
by
  -- The proof is omitted
  sorry

end popsicles_consumed_l252_252331


namespace total_capital_end_of_first_year_l252_252843

theorem total_capital_end_of_first_year (P : ℝ) :
  let initial_investment := 500000 in
  let initial_investment_in_ten_thousand := initial_investment / 10000 in
  (initial_investment_in_ten_thousand * (1 + P)) = 50 * (1 + P) :=
by
  sorry

end total_capital_end_of_first_year_l252_252843


namespace trigonometric_order_l252_252539

noncomputable def a := Real.sin (Real.sin (2009 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2009 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2009 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2009 * Real.pi / 180))

theorem trigonometric_order :
  b < a ∧ a < d ∧ d < c :=
sorry

end trigonometric_order_l252_252539


namespace product_of_solutions_of_abs_equation_l252_252495

theorem product_of_solutions_of_abs_equation : 
  (∃ x1 x2 : ℝ, |5 * x1| + 2 = 47 ∧ |5 * x2| + 2 = 47 ∧ x1 ≠ x2 ∧ x1 * x2 = -81) :=
sorry

end product_of_solutions_of_abs_equation_l252_252495


namespace ellipse_foci_y_axis_l252_252984

theorem ellipse_foci_y_axis (k : ℝ) (h_eq : ∀ x y : ℝ, x^2 + k * y^2 = 2)
  (h_foci : ∀ x y : ℝ, x^2 ≤ 2 ∧ k * y^2 ≤ 2) :
  0 < k ∧ k < 1 :=
  sorry

end ellipse_foci_y_axis_l252_252984


namespace work_completion_l252_252818

theorem work_completion (original_men planned_days absent_men remaining_men completion_days : ℕ) :
  original_men = 180 → 
  planned_days = 55 →
  absent_men = 15 →
  remaining_men = original_men - absent_men →
  remaining_men * completion_days = original_men * planned_days →
  completion_days = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_completion_l252_252818


namespace birds_landed_l252_252372

theorem birds_landed (A B : ℕ) (h_initial : A = 12) (h_final : B = 20) : B - A = 8 :=
by {
  rw [h_initial, h_final],
  exact Nat.sub_self (12 - 20),
}

end birds_landed_l252_252372


namespace sum_values_l252_252294

def nonreal_root (z : ℂ) (ω : ℂ) : Prop :=
  ω ≠ 1 ∧ ω^3 = 1

theorem sum_values (n : ℕ) (ω : ℂ) (b : ℕ → ℝ) 
  (h1 : nonreal_root (complex.I) ω)
  (h2 : ∑ k in (finset.range n), (1 / (b k + ω)) = 3 + 4 * complex.I) :
  ∑ k in (finset.range n), ((3 * b k - 2) / (b k ^ 2 - b k + 1)) = 9 - n :=
begin
  sorry
end

end sum_values_l252_252294


namespace range_of_x_l252_252206

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then exp x + a else x^2 + 1 + a

theorem range_of_x (a : ℝ) (x : ℝ) (h : f (2 - x) a ≥ f x a) : x ≤ 1 :=
by
  sorry

end range_of_x_l252_252206


namespace count_three_digit_integers_l252_252975

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_integer_sqrt (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def ends_in_one (n : ℕ) : Prop :=
  n % 10 = 1

theorem count_three_digit_integers : 
  { n : ℕ // is_three_digit n ∧ has_integer_sqrt n ∧ ends_in_one (n * n) }.to_list.length = 5 :=
by
  sorry

end count_three_digit_integers_l252_252975


namespace intersection_is_point_A_l252_252264

-- Define the Cartesian coordinate system and points
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Conditions as given in the problem:
def line_perpendicular_to_x_axis_through (p : Point) : set Point :=
  { q | q.x = p.x }

def line_perpendicular_to_y_axis_through (p : Point) : set Point :=
  { q | q.y = p.y }

-- Specific points defined in the problem
def point_on_x_axis : Point := { x := -3, y := 0 }
def point_on_y_axis : Point := { x := 0, y := -3 }

-- Problem statement to prove the intersection point coordinates
theorem intersection_is_point_A : 
  ∃ A : Point, A ∈ line_perpendicular_to_x_axis_through point_on_x_axis ∧ 
               A ∈ line_perpendicular_to_y_axis_through point_on_y_axis ∧ 
               A = { x := -3, y := -3 } :=
sorry

end intersection_is_point_A_l252_252264


namespace charles_earns_per_hour_l252_252454

variable (D : ℝ)

theorem charles_earns_per_hour (H1 : ∀ hours_house : ℝ, earnings_house : ℝ, earnings_house = 15 * hours_house)
  (H2 : 10 = 10)
  (H3 : 3 = 3)
  (H4 : ∀ total_earnings : ℝ, total_earnings = 216)
  (H5 : ∀ earnings_house earnings_dogs, total_earnings = earnings_house + earnings_dogs) :
  D = 22 :=
by 
  let earnings_house := 15 * 10
  let earnings_dogs := 216 - earnings_house
  have D_eq_22 : D = earnings_dogs / 3 := by sorry
  exact sorry

end charles_earns_per_hour_l252_252454


namespace number_of_distinct_right_triangles_l252_252015

-- Definitions of the conditions based on the problem
def tenth_roots_of_unity : Finset ℂ := 
  finset.univ.image (λ k : Fin 10, Complex.exp (2 * Real.pi * Complex.I * k / 10))

-- Main theorem statement to prove the total count of distinct right-angled triangles
theorem number_of_distinct_right_triangles : 
  (∃ M : Finset ℂ, M = tenth_roots_of_unity ∧ 
  ∑ a b c in M, (a + b = 2 * c ∨ b + c = 2 * a ∨ c + a = 2 * b) = 40) := 
sorry

end number_of_distinct_right_triangles_l252_252015


namespace max_m_for_inequality_min_4a2_9b2_c2_l252_252135

theorem max_m_for_inequality (m : ℝ) : (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 := 
sorry

theorem min_4a2_9b2_c2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (4 * a^2 + 9 * b^2 + c^2) = 36 / 49 ∧ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49 :=
sorry

end max_m_for_inequality_min_4a2_9b2_c2_l252_252135


namespace domain_sqrt_log_l252_252866

def domain_condition1 (x : ℝ) : Prop := x + 1 ≥ 0
def domain_condition2 (x : ℝ) : Prop := 6 - 3 * x > 0

theorem domain_sqrt_log (x : ℝ) : domain_condition1 x ∧ domain_condition2 x ↔ -1 ≤ x ∧ x < 2 :=
  sorry

end domain_sqrt_log_l252_252866


namespace average_of_side_lengths_of_squares_l252_252699

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l252_252699


namespace probability_blue_face_up_l252_252773

def cube_probability_blue : ℚ := 
  let total_faces := 6
  let blue_faces := 4
  blue_faces / total_faces

theorem probability_blue_face_up :
  cube_probability_blue = 2 / 3 :=
by
  sorry

end probability_blue_face_up_l252_252773


namespace plane_through_point_parallel_to_given_l252_252491

-- Define the given point
def point := (2 : ℝ, -3 : ℝ, 5 : ℝ)

-- Define the normal vector of the plane parallel to 3x + 4y - 2z = 6
def normal_vector := (3 : ℝ, 4 : ℝ, -2 : ℝ)

-- Define the plane equation form
def plane_equation (A B C D x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

-- Prove the required plane equation given the conditions
theorem plane_through_point_parallel_to_given
  (A B C D : ℝ)
  (hA : A = 3)
  (hB : B = 4)
  (hC : C = -2)
  (hD : D = 16)
  (h_point : point.1 = 2 ∧ point.2 = -3 ∧ point.3 = 5) :
  plane_equation A B C D point.1 point.2 point.3 :=
by {
  cases h_point with hx hy,
  rw [hx, hy], -- simplify using the coordinates of the point
  unfold plane_equation,
  rw [hA, hB, hC, hD], -- simplify using the plane constants
  norm_num, -- verify the equation
  sorry
}

end plane_through_point_parallel_to_given_l252_252491


namespace friends_team_division_l252_252976

theorem friends_team_division :
  let num_friends : ℕ := 8
  let num_teams : ℕ := 4
  let ways_to_divide := num_teams ^ num_friends
  ways_to_divide = 65536 :=
by
  sorry

end friends_team_division_l252_252976


namespace average_side_length_of_squares_l252_252691

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252691


namespace tan_of_X_in_right_triangle_l252_252275

theorem tan_of_X_in_right_triangle
  (X Y Z : ℝ)
  (angle_Y : X * X + Y * Y = Z * Z)
  (YZ_val : YZ = 4)
  (XZ_val : XZ = real.sqrt 17) :
  real.tan X = 4 := sorry

end tan_of_X_in_right_triangle_l252_252275


namespace marcella_lost_10_shoes_l252_252647

theorem marcella_lost_10_shoes :
  ∀ (total_pairs matching_pairs_left : ℕ), total_pairs = 26 → matching_pairs_left = 21 → 
  2 * total_pairs - 2 * matching_pairs_left = 10 :=
by
  intros total_pairs matching_pairs_left htotal_pairs hmatching_pairs_left
  rw [htotal_pairs, hmatching_pairs_left]
  -- calculate total individual shoes
  have total_individual_shoes : ℕ := 2 * 26
  -- calculate individual shoes still in pairs
  have shoes_still_in_pairs : ℕ := 2 * 21
  -- proof goal by substitution
  show total_individual_shoes - shoes_still_in_pairs = 10, from sorry

end marcella_lost_10_shoes_l252_252647


namespace smallest_perimeter_of_triangle_with_area_sqrt3_l252_252624

open Real

-- Define an equilateral triangle with given area
def equilateral_triangle (a : ℝ) : Prop :=
  ∃ s: ℝ, s > 0 ∧ a = (sqrt 3 / 4) * s^2

-- Problem statement: Prove the smallest perimeter of such a triangle is 6.
theorem smallest_perimeter_of_triangle_with_area_sqrt3 : 
  equilateral_triangle (sqrt 3) → ∃ s: ℝ, s > 0 ∧ 3 * s = 6 :=
by 
  sorry

end smallest_perimeter_of_triangle_with_area_sqrt3_l252_252624


namespace smallest_positive_period_of_f_l252_252208

def f (x : ℝ) : ℝ := Real.cos x ^ 4 - Real.sin x ^ 4

theorem smallest_positive_period_of_f : 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∃ T > 0, ∀ x, f (x + T) = f x) → T' ≥ T) :=
by
  sorry

end smallest_positive_period_of_f_l252_252208


namespace max_children_in_class_l252_252059

theorem max_children_in_class 
  (x : ℕ) 
  (chocolates_per_box : ℚ) 
  (initial_boxes : ℕ := 6) 
  (final_boxes : ℕ := 4) 
  (chocolates_per_child_initial : ℕ := 10) 
  (remaining_chocolates_initial : ℕ := 40) 
  (chocolates_per_child_final : ℕ := 8) 
  (last_child_chocolates_min : ℕ := 4) 
  (last_child_chocolates_max : ℕ := 8) : 
  chocolates_per_box = (10 * x + 40) / initial_boxes →
  final_boxes * chocolates_per_box - chocolates_per_child_final * (x - 1) ≥ last_child_chocolates_min ∧
  final_boxes * chocolates_per_box - chocolates_per_child_final * (x - 1) < last_child_chocolates_max →
  x = 23 :=
begin
  intros h_chocolates_per_box h_ineq,
  sorry
end

end max_children_in_class_l252_252059


namespace sum_of_squares_of_geometric_sequence_l252_252266

theorem sum_of_squares_of_geometric_sequence (a : ℕ → ℕ) (n : ℕ) (h : 0 < n)
  (h_sum : ∀ n : ℕ, 0 < n → (finset.range n).sum a = 3^(n) - 1) :
  (finset.range n).sum (λ i, (a i)^2) = (9^n - 1) / 2 :=
begin
  sorry
end

end sum_of_squares_of_geometric_sequence_l252_252266


namespace Dirichlet_properties_l252_252915

def Dirichlet (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

theorem Dirichlet_properties :
  Dirichlet (ℝ.sqrt 2) = 0 ∧
  (∀ y, y = Dirichlet y → y ∈ ({0, 1} : set ℝ)) ∧
  (∀ x, x ∈ ℝ) ∧
  (∀ x : ℝ, Dirichlet (x-1) = Dirichlet x) :=
by
  sorry

end Dirichlet_properties_l252_252915


namespace pole_length_after_cuts_l252_252830

theorem pole_length_after_cuts (original_length : ℝ) (cut1_percent : ℝ) (cut2_percent : ℝ) : 
  original_length = 20 → cut1_percent = 0.70 → cut2_percent = 0.75 → 
  let length_after_first_cut := original_length * cut1_percent in
  let length_after_second_cut := length_after_first_cut * cut2_percent in
  length_after_second_cut = 10.5 :=
by
  intros h_orig h_cut1 h_cut2
  let l1 := original_length * cut1_percent
  let l2 := l1 * cut2_percent
  have l1_eq : l1 = 14, by
    rw [h_orig, h_cut1]
    norm_num
  have l2_eq : l2 = 10.5, by
    rw [l1_eq, h_cut2]
    norm_num
  exact l2_eq

end pole_length_after_cuts_l252_252830


namespace parking_lot_wheels_l252_252132

noncomputable def total_car_wheels (guest_cars : Nat) (guest_car_wheels : Nat) (parent_cars : Nat) (parent_car_wheels : Nat) : Nat :=
  guest_cars * guest_car_wheels + parent_cars * parent_car_wheels

theorem parking_lot_wheels :
  total_car_wheels 10 4 2 4 = 48 :=
by
  sorry

end parking_lot_wheels_l252_252132


namespace reservoir_solution_l252_252430

theorem reservoir_solution (x y z : ℝ) :
  8 * (1 / x - 1 / y) = 1 →
  24 * (1 / x - 1 / y - 1 / z) = 1 →
  8 * (1 / y + 1 / z) = 1 →
  x = 8 ∧ y = 24 ∧ z = 12 :=
by
  intros h1 h2 h3
  sorry

end reservoir_solution_l252_252430


namespace lucy_play_area_l252_252315

theorem lucy_play_area (shed_length : ℝ) (shed_width : ℝ) (leash_length : ℝ) 
  (h1 : shed_length = 4) (h2 : shed_width = 3) (h3 : leash_length = 4) : 
  let total_area : ℝ := (3 / 4) * Real.pi * leash_length^2 + (1 / 4) * Real.pi * 1^2 in
  total_area = 12.25 * Real.pi :=
by {
  -- sorry to skip the proof
  sorry
}

end lucy_play_area_l252_252315


namespace no_all_perfect_squares_l252_252636

theorem no_all_perfect_squares (x : ℤ) 
  (h1 : ∃ a : ℤ, 2 * x - 1 = a^2) 
  (h2 : ∃ b : ℤ, 5 * x - 1 = b^2) 
  (h3 : ∃ c : ℤ, 13 * x - 1 = c^2) : 
  False :=
sorry

end no_all_perfect_squares_l252_252636


namespace average_side_length_of_squares_l252_252687

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l252_252687


namespace series_sum_formula_l252_252452

open BigOperators

theorem series_sum_formula (n : ℕ) :
  (∑ k in Finset.range n, k * (k + 2)^2) = (n.choose 3) * (3 * n + 2) / 2 :=
sorry

end series_sum_formula_l252_252452


namespace compound_interest_time_l252_252755

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

noncomputable def compound_interest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem compound_interest_time :
  let SI := simple_interest 3225 8 5 in
  let CI_amount := 2 * SI in
  ∃ n : ℝ, compound_interest 8000 15 n = CI_amount ∧ n = 1 :=
by
  admit -- Proof goes here

end compound_interest_time_l252_252755


namespace eliza_polynomial_constant_term_l252_252861

theorem eliza_polynomial_constant_term :
  ∃ (p q : Polynomial ℤ), 
    p.degree = 5 ∧ p.monic ∧ q.degree = 5 ∧ q.monic ∧
    p.coeff 0 = q.coeff 0 ∧
    (p * q) = (Polynomial.X ^ 10 + 5 * Polynomial.X ^ 9 + 10 * Polynomial.X ^ 8 +
    10 * Polynomial.X ^ 7 + 9 * Polynomial.X ^ 6 + 8 * Polynomial.X ^ 5 +
    5 * Polynomial.X ^ 4 + 8 * Polynomial.X ^ 3 + 3 * Polynomial.X ^ 2 + 3 * Polynomial.X + 9) ∧
    p.coeff 0 = q.coeff 0 ∧
    p.coeff 0 = 3 := 
sorry

end eliza_polynomial_constant_term_l252_252861


namespace total_apples_bought_l252_252283

def apples_bought_by_Junhyeok := 7 * 16
def apples_bought_by_Jihyun := 6 * 25

theorem total_apples_bought : apples_bought_by_Junhyeok + apples_bought_by_Jihyun = 262 := by
  sorry

end total_apples_bought_l252_252283


namespace total_dots_not_visible_on_stacked_dice_l252_252509

theorem total_dots_not_visible_on_stacked_dice :
   (∑ i in {1, 2, 3, 4, 5, 6}, i) * 4 - 28 = 56 :=
by
  sorry

end total_dots_not_visible_on_stacked_dice_l252_252509


namespace correct_labels_using_systematic_sampling_l252_252129

theorem correct_labels_using_systematic_sampling :
  let interval := 10 in
  let total_missiles := 60 in
  let num_to_sample := 6 in
  ∃ (labels : list ℕ), 
    (∀ i, i < (num_to_sample - 1) → (labels.nth_le (i + 1) (sorry) - labels.nth_le i sorry) = interval) ∧
    labels = [3, 13, 23, 33, 43, 53]
:=
sorry

end correct_labels_using_systematic_sampling_l252_252129


namespace geometric_sequence_a3_eq_sqrt_5_l252_252523

theorem geometric_sequence_a3_eq_sqrt_5 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * r)
  (h_a1 : a 1 = 1) (h_a5 : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end geometric_sequence_a3_eq_sqrt_5_l252_252523


namespace pool_capacity_l252_252096

noncomputable def total_capacity : ℝ := 1000

theorem pool_capacity
    (C : ℝ)
    (H1 : 0.75 * C = 0.45 * C + 300)
    (H2 : 300 / 0.3 = 1000)
    : C = total_capacity :=
by
  -- Solution steps are omitted, proof goes here.
  sorry

end pool_capacity_l252_252096


namespace product_of_positive_x_l252_252494

theorem product_of_positive_x (q : ℕ) (hq : Prime q) (hx : ∃ x : ℕ, x^2 - 40 * x + 399 = q) : 
  ∏ x in {x : ℕ | x^2 - 40 * x + 399 = q}.to_finset = 396 :=
by 
  sorry

end product_of_positive_x_l252_252494


namespace ratio_black_haired_children_l252_252082

theorem ratio_black_haired_children 
  (n_red : ℕ) (n_total : ℕ) (ratio_red : ℕ) (ratio_blonde : ℕ) (ratio_black : ℕ)
  (h_ratio : ratio_red / ratio_red = 1 ∧ ratio_blonde / ratio_red = 2 ∧ ratio_black / ratio_red = 7 / 3)
  (h_n_red : n_red = 9)
  (h_n_total : n_total = 48) :
  (7 : ℚ) / (16 : ℚ) = (n_total * 7 / 16 : ℚ) :=
sorry

end ratio_black_haired_children_l252_252082


namespace line_perpendicular_to_plane_if_perpendicular_to_two_intersecting_lines_l252_252438

open Set

variables {α β γ : Type}

/-- Define line and plane as sets in 3D space -/
def line (l : Set (ℝ × ℝ × ℝ)) : Prop := ∀ x x' ∈ l, ∃ a : ℝ, x' = x + a • (x - l)
def plane (α : Set (ℝ × ℝ × ℝ)) : Prop := ∃ p₀ ∈ α, ∃ u₁ u₂ ∈ α, ∀ p ∈ α, ∃ a b : ℝ, p = p₀ + a • u₁ + b • u₂

variables {l : Set (ℝ × ℝ × ℝ)} {α : Set (ℝ × ℝ × ℝ)}

/-- Define when a line is perpendicular to a plane -/
def perpendicular_to_plane (l : Set (ℝ × ℝ × ℝ)) (α : Set (ℝ × ℝ × ℝ)) : Prop :=
∀ p₀ u₁ u₂, p₀ ∈ α ∧ u₁ ∈ α ∧ u₂ ∈ α → 
  let a := p₀ + u₁ - (p₀ + u₂)
  in l = a

theorem line_perpendicular_to_plane_if_perpendicular_to_two_intersecting_lines 
  (h_line : line l) (h_plane : plane α)
  (h_intertsecting_lines : ∃ p₀ u₁ u₂, u₁ ≠ u₂ ∧ u₁ ∈ α ∧ u₂ ∈ α ∧ ∀ a b, l = p₀ + a • u₁ + b • u₂) :
  perpendicular_to_plane l α :=
sorry

end line_perpendicular_to_plane_if_perpendicular_to_two_intersecting_lines_l252_252438


namespace sum_of_first_20_terms_l252_252608

theorem sum_of_first_20_terms (a d : ℚ)
  (h1 : a + (a + d) + (a + 2 * d) = 3)
  (h2 : (a + 17 * d) + (a + 18 * d) + (a + 19 * d) = 87) :
  let S_20 := 10 * ((a + (a + 19 * d)) / 2) in
  S_20 = 300 :=
by
  sorry

end sum_of_first_20_terms_l252_252608


namespace infinite_solutions_XYZ_eq_2_l252_252338

theorem infinite_solutions_XYZ_eq_2 :
  ∃ f : Int → Int × Int × Int, ∀ t : Int,
  let (X, Y, Z) := f t in X^3 + Y^3 + Z^3 = 2 :=
by
  exists (λ t => (1 + 6*t^3, 1 - 6*t^3, -6*t^2))
  intro t
  let X := 1 + 6*t^3
  let Y := 1 - 6*t^3
  let Z := -6*t^2
  sorry

end infinite_solutions_XYZ_eq_2_l252_252338


namespace smaller_of_two_digit_numbers_l252_252074

theorem smaller_of_two_digit_numbers (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4725) :
  min a b = 15 :=
sorry

end smaller_of_two_digit_numbers_l252_252074


namespace sum_factors_of_24_l252_252039

theorem sum_factors_of_24 : ∑ d in (Finset.filter (λ d, 24 % d = 0) (Finset.range 25)), d = 60 := by
  sorry

end sum_factors_of_24_l252_252039


namespace remainder_when_divided_by_100_l252_252810

/-- A basketball team has 15 available players. A fixed set of 5 players starts the game, while the other 
10 are available as substitutes. During the game, the coach may make up to 4 substitutions. No player 
removed from the game may reenter, and no two substitutions can happen simultaneously. The players 
involved and the order of substitutions matter. -/
def num_substitution_sequences : ℕ :=
  let a_0 := 1
  let a_1 := 5 * 10
  let a_2 := a_1 * 4 * 9
  let a_3 := a_2 * 3 * 8
  let a_4 := a_3 * 2 * 7
  a_0 + a_1 + a_2 + a_3 + a_4

theorem remainder_when_divided_by_100 : num_substitution_sequences % 100 = 51 :=
by
  -- proof to be written
  sorry

end remainder_when_divided_by_100_l252_252810


namespace value_of_n_l252_252312

noncomputable def random_variable_xi (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (P(k)= 1 / n)

theorem value_of_n (n : ℕ) (h1 : ∀ k, 1 ≤ k ∧ k ≤ n → (P(k) = 1 / n))
(h2 : P(1) + P(2) + P(3) = 0.3) : n = 10 :=
by 
  sorry

end value_of_n_l252_252312


namespace lucky_number_probability_l252_252385

def is_lucky (x : ℝ) : Prop :=
  ∃ (n : ℤ), (n % 2 = 1) ∧ (n = ⌊real.log x / real.log 2⌋)

def interval := set.Ioo 0 1

def lucky_set : set ℝ := {x ∈ interval | is_lucky x}

theorem lucky_number_probability : 
  measure_theory.measure.lintegral measure_theory.measure_space.volume (set.indicator lucky_set (λ x, 1)) / 
  measure_theory.measure.lintegral measure_theory.measure_space.volume (set.indicator interval (λ x, 1)) = 1 / 3 :=
sorry

end lucky_number_probability_l252_252385


namespace parabola_circle_intercept_l252_252424

theorem parabola_circle_intercept (p : ℝ) (h_pos : p > 0) :
  (∃ (x y : ℝ), y^2 = 2 * p * x ∧ x^2 + y^2 + 2 * x - 3 = 0) ∧
  (∃ (y1 y2 : ℝ), (y1 - y2)^2 + (-(p / 2) + 1)^2 = 4^2) → p = 2 :=
by sorry

end parabola_circle_intercept_l252_252424


namespace triangle_side_calculation_l252_252246

theorem triangle_side_calculation
  (a : ℝ) (A B : ℝ)
  (ha : a = 3)
  (hA : A = 30)
  (hB : B = 15) :
  let C := 180 - A - B
  let c := a * (Real.sin C) / (Real.sin A)
  c = 3 * Real.sqrt 2 := by
  sorry

end triangle_side_calculation_l252_252246


namespace bassanio_end_with_yellow_coins_l252_252449

-- Initial number of coins
def initial_red : ℕ := 3
def initial_yellow : ℕ := 4
def initial_blue : ℕ := 5

-- Invariant differences
def yellow_minus_red (y r : ℕ) := y - r
def blue_minus_yellow (b y : ℕ) := b - y
def blue_minus_red (b r : ℕ) := b - r

-- The final goal
def final_coins (r y b : ℕ) : Prop :=
  r = 0 ∧ y = 7 ∧ b = 0

-- Theorem statement
theorem bassanio_end_with_yellow_coins :
  ∃ r y b, initial_red = 3 ∧ initial_yellow = 4 ∧ initial_blue = 5 ∧
  (r = initial_red ∧ y = initial_yellow ∧ b = initial_blue ∨
   (∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 
       (r + n = k + y ∧ b - n = initial_blue - k ∧ yellow_minus_red y r = 1 ∧ blue_minus_yellow b y = 1 ∧ blue_minus_red b r = 2))) ∧
  final_coins r y b
:= sorry

end bassanio_end_with_yellow_coins_l252_252449


namespace value_of_2_Z_6_l252_252590

def Z (a b : ℝ) : ℝ := b + 10 * a - a^2

theorem value_of_2_Z_6 : Z 2 6 = 22 :=
by
  sorry

end value_of_2_Z_6_l252_252590


namespace tangent_line_equation_l252_252000

noncomputable def curve (x : ℝ) : ℝ := x / (2 * x - 1)
noncomputable def derivative_curve (x : ℝ) : ℝ := -1 / (2 * x - 1)^2

theorem tangent_line_equation : 
  let P := (1, 1 : ℝ)
  let slope := derivative_curve 1
  let tangent_line (x y : ℝ) := (y - P.2) = slope * (x - P.1)
  ∃ x y : ℝ, tangent_line x y → x + y - 2 = 0 :=
by
  sorry

end tangent_line_equation_l252_252000


namespace quadratic_has_distinct_real_roots_l252_252756

theorem quadratic_has_distinct_real_roots :
  ∃ (a b c : ℝ), a = 1 ∧ b = -3 ∧ c = -4 ∧ (b^2 - 4 * a * c > 0) :=
by
  use [1, -3, -4]
  split
  . refl
  split
  . refl
  split
  . refl
  calc (-3)^2 - 4 * 1 * (-4)
    _ = 9 + 16 := by norm_num
    _ = 25     := by norm_num
    _ > 0      := by norm_num
  done

end quadratic_has_distinct_real_roots_l252_252756


namespace area_BDE_l252_252276

-- Define the given conditions and the parameters
variables {α γ : ℝ} -- angles at base
variables (α_gt_γ : α > γ) -- α > γ
variables {AC S : ℝ} -- base AC and area S
variables (hBD : ℝ) -- height from B
variables (hBE : ℝ) -- angle bisector from B

-- The proof is formulated as a statement
theorem area_BDE (α : ℝ) (γ : ℝ) (AC : ℝ) (S : ℝ) (α_gt_γ : α > γ) (BD : ℝ) (BE : ℝ) :
  let area_BDE := S * (Real.tan ((α - γ) / 2)) * ((Real.sin α) * (Real.sin γ) / (Real.sin (α + γ)))
  in area_BDE = S * (Real.tan ((α - γ) / 2)) * (Real.sin α) * (Real.sin γ) / (Real.sin (α + γ)) :=
by
  sorry -- Proof is skipped with sorry

end area_BDE_l252_252276


namespace coloring_theorem_l252_252121

noncomputable def num_of_valid_colorings (n p : ℕ) : ℕ :=
  if p = 2 then
    if even n then 2 else 0
  else
    p * (p - 1)^(n - 1) - num_of_valid_colorings (n - 1) p

theorem coloring_theorem (n p : ℕ) :
  num_of_valid_colorings n p = 
    if p = 2 then
      if even n then 2 else 0
    else
      p * (p - 1)^(n - 1) - num_of_valid_colorings (n - 1) p :=
sorry

end coloring_theorem_l252_252121


namespace same_function_A_same_function_C_l252_252109

def fA (x : ℝ) : ℝ := x
def gA (x : ℝ) : ℝ := real.cbrt (x ^ 3)

def fB (x : ℝ) : ℝ := 
  if x = 1 ∨ x = -1 then 0 else 1
def gB (x : ℝ) : ℝ := 1

def fC (x : ℝ) : ℝ := x + 1 / x
def gC (t : ℝ) : ℝ := t + 1 / t

def fD (x : ℝ) : ℝ := real.sqrt (x + 1) * real.sqrt (x - 1)
def gD (x : ℝ) : ℝ := real.sqrt (x ^ 2 - 1)

theorem same_function_A : (∀ x, fA x = gA x) := 
by {
  intro x,
  rw [fA, gA],
  sorry
}

theorem same_function_C : (∀ x, fC x = gC x) := 
by {
  intro x,
  rw [fC, gC],
  sorry
}

end same_function_A_same_function_C_l252_252109


namespace at_least_one_digit_position_probability_l252_252374

theorem at_least_one_digit_position_probability :
  let n := 9 in
  let p := 1 - ((8 / 9) ^ 9) in
  |p - 0.653| < 0.001 :=
by
  let n := 9
  let p := 1 - ((8 / 9) ^ 9)
  have h : |p - 0.653| < 0.001 := sorry
  exact h

end at_least_one_digit_position_probability_l252_252374


namespace eccentricity_of_ellipse_l252_252857

open Real

theorem eccentricity_of_ellipse (a b c : ℝ) 
  (h1 : a > b ∧ b > 0)
  (h2 : c^2 = a^2 - b^2)
  (x : ℝ)
  (h3 : 3 * x = 2 * a)
  (h4 : sqrt 3 * x = 2 * c) :
  c / a = sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l252_252857


namespace tangent_line_to_circle_l252_252741

-- Definitions derived directly from the conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0
def passes_through_point (l : ℝ → ℝ → Prop) : Prop := l (-1) 6

-- The statement to be proven
theorem tangent_line_to_circle :
  ∃ (l : ℝ → ℝ → Prop), passes_through_point l ∧ 
    ((∀ x y, l x y ↔ 3*x - 4*y + 27 = 0) ∨ 
     (∀ x y, l x y ↔ x + 1 = 0)) :=
sorry

end tangent_line_to_circle_l252_252741


namespace find_b_c_find_a_range_l252_252417

noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c
noncomputable def g (a b c x : ℝ) : ℝ := f a b c x + 2 * x
noncomputable def f_prime (a b x : ℝ) : ℝ := x^2 - a * x + b
noncomputable def g_prime (a b x : ℝ) : ℝ := f_prime a b x + 2

theorem find_b_c (a c : ℝ) (h_f0 : f a 0 c 0 = c) (h_tangent_y_eq_1 : 1 = c) : 
  b = 0 ∧ c = 1 :=
by
  sorry

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, g_prime a 0 x ≥ 0) ↔ a ≤ 2 * Real.sqrt 2 :=
by
  sorry

end find_b_c_find_a_range_l252_252417


namespace largest_integer_k_l252_252432

def sequence_term (n : ℕ) : ℚ :=
  (1 / n - 1 / (n + 2))

def sum_sequence (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), sequence_term (i + 1)

theorem largest_integer_k (k : ℕ):
  (sum_sequence k < 1.499) ↔ k = 1998 :=
sorry

end largest_integer_k_l252_252432


namespace int_valued_fractions_l252_252887

theorem int_valued_fractions (a : ℤ) :
  ∃ k : ℤ, (a^2 - 21 * a + 17) = k * a ↔ a = 1 ∨ a = -1 ∨ a = 17 ∨ a = -17 :=
by {
  sorry
}

end int_valued_fractions_l252_252887


namespace find_f_of_3_l252_252506

variable {ℝ : Type*} [linear_ordered_field ℝ]

theorem find_f_of_3 (f : ℝ → ℝ) (cond1 : ∀ x y : ℝ, 0 < x → 0 < y → f x + f y = f x * f y + 1 - 1 / (x * y))
  (cond2 : f 2 < 1) : f 3 = 2 / 3 :=
sorry

end find_f_of_3_l252_252506


namespace clara_stickers_left_l252_252850

def initial_stickers : ℕ := 100
def stickers_given_to_boy (initial: ℕ) : ℕ := 10
def remaining_after_boy (initial given: ℕ) : ℕ := initial - given
def half_given_to_best_friends (remaining : ℕ) : ℕ := remaining / 2
def stickers_left (initial given_to_boy given_to_friends: ℕ) : ℕ :=
  initial - given_to_boy - given_to_friends

theorem clara_stickers_left :
  let initial := initial_stickers,
      given_to_boy := stickers_given_to_boy initial,
      remaining := remaining_after_boy initial given_to_boy,
      given_to_friends := half_given_to_best_friends remaining in
  stickers_left initial given_to_boy given_to_friends = 45 :=
by
  -- this is where the proof would go
  sorry

end clara_stickers_left_l252_252850


namespace average_side_lengths_l252_252681

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l252_252681


namespace min_n_real_root_poly_l252_252526

noncomputable def has_real_root (n : ℕ) :=
  ∃ x : ℝ, (100 * (∑ i in finset.range (n + 1), x^(2 * i)) + 
            101 * (∑ i in finset.range n, x^(2 * i + 1))) = 0

theorem min_n_real_root_poly : ∀ n : ℕ, (∀ m : ℕ, m < n → ¬ has_real_root m) → has_real_root n ∧ n = 100 :=
begin
  sorry
end

end min_n_real_root_poly_l252_252526


namespace equal_areas_l252_252532

variables {A B C P A1 B1 M3 : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space P] 
[metric_space A1] [metric_space B1] [metric_space M3]
variables [equiv A B] 

-- Conditions
-- M3 is the midpoint of AB, hence AM3 = M3B
def is_midpoint (M3 : Type) (A B : Type) [metric_space M3] [metric_space A] [metric_space B] : Prop :=
dist A M3 = dist M3 B

-- P is an arbitrary point on median CM3
def on_median (P : Type) (C M3 : Type) [metric_space P] [metric_space C] [metric_space M3] : Prop :=
dist C P = dist P M3

-- AP and BP intersect sides BC and AC at points A1 and B1 respectively
def intersects (A1 B1 : Type) (A B P C : Type) [metric_space A1] [metric_space B1] [metric_space A]
[metric_space B] [metric_space P] [metric_space C] (on_median : on_median P C M3) : Prop :=
-- This expresses that P on the median CM3 then AP and BP intersect at A1 and B1 respectively
dist A1 A = dist A1 B ∧ dist B1 A = dist B1 B 

-- The Lean statement: Prove that areas of triangles AA1C and BB1C are equal
theorem equal_areas (h1: is_midpoint M3 A B) (h2: on_median P C M3) (h3: intersects A1 B1 A B P C h2) :
  area AA1C = area BB1C :=
sorry


end equal_areas_l252_252532


namespace average_side_lengths_l252_252703

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l252_252703


namespace num_valid_pairs_l252_252231

def is_factor (n d : ℕ) : Prop := d ∣ n

def is_valid_pair (a c : ℕ) : Prop :=
  a < c ∧ is_factor 180 a ∧ is_factor 180 c ∧ a + c = 180

theorem num_valid_pairs : {p : (ℕ × ℕ) // is_valid_pair p.1 p.2} = 89 :=
  sorry

end num_valid_pairs_l252_252231


namespace smallest_whole_number_larger_than_perimeter_l252_252391

theorem smallest_whole_number_larger_than_perimeter (c : ℝ) (h1 : 13 < c) (h2 : c < 25) : 50 = Nat.ceil (6 + 19 + c) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l252_252391


namespace num_nat_solutions_l252_252155

theorem num_nat_solutions (x y : ℕ) (h : (x - 4) ^ 2 - 35 = (y - 3) ^ 2) :
  ∃ (n : ℕ), n = 3 :=
begin
  sorry
end

end num_nat_solutions_l252_252155


namespace infinite_series_converges_l252_252482

noncomputable def series_term (n : ℕ) : ℝ :=
  (2 * n^2 - n + 1) / (n + 3)!

theorem infinite_series_converges :
  ∑' n, series_term n = 1 / 3 :=
by
  sorry

end infinite_series_converges_l252_252482


namespace volumeFormulaCorrect_l252_252069

-- Definitions for the conditions
structure TruncatedPyramid where
  height : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Volume calculation using the given formula
def calculatedVolume (p : TruncatedPyramid) : ℝ :=
  (p.height / 6) * ((2 * p.a + p.c) * p.b + (2 * p.c + p.a) * p.d)

-- Volume calculation using the standard truncated pyramid formula
def standardVolume (p : TruncatedPyramid) : ℝ :=
  (p.height / 3) * (p.a * p.b + Real.sqrt (p.a * p.b * p.c * p.d) + p.c * p.d)

-- The proof problem
theorem volumeFormulaCorrect (p : TruncatedPyramid) :
  calculatedVolume p = standardVolume p :=
by
  -- Skipping the proof as instructed
  sorry

end volumeFormulaCorrect_l252_252069


namespace time_for_train_to_pass_pole_l252_252062

-- Definitions based on conditions
def train_length_meters : ℕ := 160
def train_speed_kmph : ℕ := 72

-- The calculated speed in m/s
def train_speed_mps : ℕ := train_speed_kmph * 1000 / 3600

-- The calculation of time taken to pass the pole
def time_to_pass_pole : ℕ := train_length_meters / train_speed_mps

-- The theorem statement
theorem time_for_train_to_pass_pole : time_to_pass_pole = 8 := sorry

end time_for_train_to_pass_pole_l252_252062


namespace decrease_percent_in_revenue_l252_252400

theorem decrease_percent_in_revenue (T C : ℝ) : 
  let original_revenue := T * C,
      new_tax_rate := T - 0.22 * T,
      new_consumption := C + 0.09 * C,
      new_revenue := new_tax_rate * new_consumption in
  original_revenue ≠ 0 →
  (original_revenue - new_revenue) / original_revenue * 100 = 15.02 := 
by
  intros
  unfold original_revenue new_tax_rate new_consumption new_revenue
  sorry

end decrease_percent_in_revenue_l252_252400


namespace rbcmul_div7_div89_l252_252749

theorem rbcmul_div7_div89 {r b c : ℕ} (h : (523000 + 100 * r + 10 * b + c) % 7 = 0 ∧ (523000 + 100 * r + 10 * b + c) % 89 = 0) :
  r * b * c = 36 :=
by
  sorry

end rbcmul_div7_div89_l252_252749


namespace imaginary_part_divide_by_i_l252_252610

theorem imaginary_part_divide_by_i (z : ℂ) (hz : z = 1 - 3 * Complex.i) : 
  Complex.im (z / Complex.i) = -1 := 
sorry

end imaginary_part_divide_by_i_l252_252610


namespace value_of_number_l252_252653

-- Define the given condition
def condition (N : ℝ) := 0.40 * N = 420

-- Define the question part
def one_fourth_one_third_two_fifth (N : ℝ) := (1 / 4) * (1 / 3) * (2 / 5) * N

-- State the theorem
theorem value_of_number (N : ℝ) (h : condition N) : one_fourth_one_third_two_fifth N = 35 :=
sorry

end value_of_number_l252_252653


namespace sum_of_roots_l252_252160

theorem sum_of_roots :
  let f : ℝ → ℝ := λ x, (tan x)^2 - 5 * tan x + 6,
      x1 := arctan 3,
      x2 := arctan 2 in
  (∀ x ∈ Icc 0 real.pi, f x = 0 → (x = x1 ∨ x = x2)) →
  (x1 + x2) = arctan 3 + arctan 2 :=
begin
  intros,
  sorry
end

end sum_of_roots_l252_252160
