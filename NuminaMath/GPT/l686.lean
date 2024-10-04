import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.EuclideanDomain.Dvd
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Powers
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.Matrix
import Mathlib.LinearAlgebra.VecStr
import Mathlib.Order.Basic
import Mathlib.SetTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace cylinder_volume_ratio_l686_686222

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l686_686222


namespace nine_point_circle_l686_686655

open EuclideanGeometry

variables {A B C P Q G : Point}

theorem nine_point_circle
  (hG : is_centroid G A B C)
  (hP : on_circumcircle P A B C)
  (hPQ : dist P Q = (1 / 2) * dist P G) :
  on_nine_point_circle Q A B C :=
sorry

end nine_point_circle_l686_686655


namespace museum_ticket_cost_l686_686817

theorem museum_ticket_cost 
  (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_cost : ℕ) (teacher_ticket_cost : ℕ)
  (h_students : num_students = 12) (h_teachers : num_teachers = 4)
  (h_student_cost : student_ticket_cost = 1) (h_teacher_cost : teacher_ticket_cost = 3) :
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end museum_ticket_cost_l686_686817


namespace simplify_tan_expression_l686_686707

theorem simplify_tan_expression :
  ∀ (tan_15 tan_30 : ℝ), 
  (tan_45 : ℝ) 
  (tan_45_eq_one : tan_45 = 1)
  (tan_sum_formula : tan_45 = (tan_15 + tan_30) / (1 - tan_15 * tan_30)),
  (1 + tan_15) * (1 + tan_30) = 2 :=
by
  intros tan_15 tan_30 tan_45 tan_45_eq_one tan_sum_formula
  sorry

end simplify_tan_expression_l686_686707


namespace simplify_tan_expression_l686_686693

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end simplify_tan_expression_l686_686693


namespace line_intersects_fixed_point_l686_686384

theorem line_intersects_fixed_point
  (a b : ℝ) (ha : a > b) (hb : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (min_distance_left_focus : ∀ P : ℝ × ℝ, P ∈ ellipse_eq → dist P (-(a - 1), 0) = 1)
  (eccentricity : b^2 + (a / 2)^2 = a^2)
  (QA QB : ℝ × ℝ → ℝ)
  (QA_perpendicular_QB : (QA + QB).norm = (QA - QB).norm) :
  ∃ fx fy : ℝ, fx = 2 / 7 ∧ fy = 0 :=
by
  sorry

end line_intersects_fixed_point_l686_686384


namespace distance_between_points_l686_686967

noncomputable def i := Complex.I
noncomputable def z1 := (-1 + 3 * i) / (1 + 2 * i)
noncomputable def z2 := 1 + (1 + i)^[10]

theorem distance_between_points :
  Real.sqrt (((z2.re - z1.re) ^ 2) + ((z2.im - z1.im) ^ 2)) = Real.sqrt 231.68 :=
by
  sorry

end distance_between_points_l686_686967


namespace distinct_units_digits_of_cubes_l686_686435

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686435


namespace percentage_of_initial_pace_l686_686846

-- Define the total distance of the marathon
def marathon_distance : ℝ := 26

-- Define the distance run in the first hour
def initial_distance : ℝ := 10

-- Define the total time of the marathon
def total_time : ℝ := 3

-- Define the time spent on the initial distance
def initial_time : ℝ := 1

-- The remaining distance to be run
def remaining_distance : ℝ := marathon_distance - initial_distance

-- The remaining time to run the remaining distance
def remaining_time : ℝ := total_time - initial_time

-- The initial pace
def initial_pace : ℝ := initial_distance / initial_time

-- The pace for the remaining distance
def remaining_pace : ℝ := remaining_distance / remaining_time

-- Prove that the percentage of the initial pace for the remaining miles is 80%
theorem percentage_of_initial_pace : (remaining_pace / initial_pace) * 100 = 80 := by
  sorry

end percentage_of_initial_pace_l686_686846


namespace tan_product_l686_686700

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end tan_product_l686_686700


namespace largest_n_A_l686_686385

-- Definitions for conditions and variables
def is_set_of_distinct_positives (A : Finset ℕ) (n : ℕ) : Prop := 
  A.card = 4 ∧ ∀ x ∈ A, x > 0

def s_A (A : Finset ℕ) := A.sum id

def n_A (A : Finset ℕ) := 
  (Finset.univ.filter (λ (i : Fin (4) × Fin (4)), i.1 < i.2 ∧ A.val.nth_le i.1 sorry + A.val.nth_le i.2 sorry ∣ s_A A)).card

-- The main problem statement
theorem largest_n_A (A : Finset ℕ) (hA : is_set_of_distinct_positives A 4):
  n_A A = 4 ↔ ∃ k : ℕ, A = {k, 5 * k, 7 * k, 11 * k} ∨ A = {k, 11 * k, 19 * k, 29 * k} := 
sorry

end largest_n_A_l686_686385


namespace problem_statement_l686_686932

theorem problem_statement 
  (x y z : ℝ)
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 :=
sorry

end problem_statement_l686_686932


namespace ratio_of_areas_l686_686742

-- Definitions based on the conditions
def rectangle_perimeter_eq_circle_circumference (l w r : ℝ) : Prop :=
  2 * l + 2 * w = 2 * π * r

def length_twice_width (l w : ℝ) : Prop :=
  l = 2 * w

-- Statement of the problem to be proved
theorem ratio_of_areas (l w r : ℝ) (h1 : rectangle_perimeter_eq_circle_circumference l w r) (h2 : length_twice_width l w) :
  ((l * w) / (π * r ^ 2)) = (2 / 9) :=
by
  sorry

end ratio_of_areas_l686_686742


namespace battery_lasts_for_8_more_hours_l686_686275

-- Definitions based on the problem conditions
def battery_life_not_in_use : ℝ := 36 -- in hours
def battery_life_in_use : ℝ := 4 -- in hours
def usage_period : ℝ := 12 -- in hours
def video_watching_time : ℝ := 120 / 60 -- converted to hours, which is 2 hours

-- Sum up all the information above and prove the given problem statement
theorem battery_lasts_for_8_more_hours 
  (battery_life_not_in_use: ℝ) 
  (battery_life_in_use: ℝ)
  (usage_period: ℝ)
  (video_watching_time: ℝ)
  (h₁: battery_life_not_in_use = 36) 
  (h₂: battery_life_in_use = 4)
  (h₃: usage_period = 12)
  (h₄: video_watching_time = 2)
  : (usage_period - video_watching_time) * (1 / battery_life_not_in_use) 
    + video_watching_time * (1 / battery_life_in_use) = 1 - (2 / 9) →
    (2 / 9) * 36 = 8 :=
sorry

end battery_lasts_for_8_more_hours_l686_686275


namespace area_of_shaded_region_l686_686765

theorem area_of_shaded_region :
  let radius := 15
  let angle := 45
  let sector_area := (angle / 360) * Math.pi * radius^2
  let triangle_area := (1 / 2) * radius^2 * Real.sin (angle * Math.pi / 180)
  let shaded_area_per_sector := sector_area - triangle_area
  let total_shaded_area := 2 * shaded_area_per_sector
  total_shaded_area = (225 * Math.pi - 450 * Real.sqrt 2) / 4 := by sorry

end area_of_shaded_region_l686_686765


namespace cylinder_volume_ratio_l686_686224

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l686_686224


namespace simplify_tan_expression_l686_686689

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end simplify_tan_expression_l686_686689


namespace ratio_of_sums_l686_686359

noncomputable def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

theorem ratio_of_sums (n : ℕ) (S1 S2 : ℕ) 
  (hn_even : n % 2 = 0)
  (hn_pos : 0 < n)
  (h_sum : sum_upto (n^2) = n^2 * (n^2 + 1) / 2)
  (h_S1S2_sum : S1 + S2 = n^2 * (n^2 + 1) / 2)
  (h_ratio : 64 * S1 = 39 * S2) :
  ∃ k : ℕ, n = 103 * k :=
sorry

end ratio_of_sums_l686_686359


namespace figure_total_area_l686_686291

theorem figure_total_area :
  let height_left_rect := 6
  let width_base_left_rect := 5
  let height_top_left_rect := 3
  let width_top_left_rect := 5
  let height_top_center_rect := 3
  let width_sum_center_rect := 10
  let height_top_right_rect := 8
  let width_top_right_rect := 2
  let area_total := (height_left_rect * width_base_left_rect) + (height_top_left_rect * width_top_left_rect) + (height_top_center_rect * width_sum_center_rect) + (height_top_right_rect * width_top_right_rect)
  area_total = 91
:= sorry

end figure_total_area_l686_686291


namespace distinct_units_digits_of_cubes_l686_686532

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686532


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686147

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686147


namespace volume_ratio_cylinders_l686_686203

open Real

noncomputable def volume_ratio_larger_to_smaller (h₁ : Real) (h₂ : Real) (circumference₁ : Real) (circumference₂ : Real) : Real :=
  let r₁ := circumference₁ / (2 * pi)
  let r₂ := circumference₂ / (2 * pi)
  let V₁ := pi * r₁^2 * h₁
  let V₂ := pi * r₂^2 * h₂
  V₂ / V₁

theorem volume_ratio_cylinders : volume_ratio_larger_to_smaller 9 6 6 9 = 9 / 4 := by
  sorry

end volume_ratio_cylinders_l686_686203


namespace unique_cyclic_permutation_reduces_to_B_l686_686020

theorem unique_cyclic_permutation_reduces_to_B (n : ℕ) (word : List Char)
  (h1 : word.length = 3 * n + 1)
  (h2 : word.count 'A' = n)
  (h3 : word.count 'B' = 2 * n + 1) 
  (reduction_rule : List Char → List Char)
  (h4 : ∀ w, reduction_rule ('A' :: 'B' :: 'B' :: 'B' :: w) = 'B' :: w) :
  ∃! w', cyclic_permutation word w' ∧ reducible_to_B reduction_rule w' :=
  sorry

-- Definitions to support the theorem
def cyclic_permutation (word word' : List Char) : Prop := sorry
def reducible_to_B (reduction_rule : List Char → List Char) (word : List Char) : Prop := sorry

end unique_cyclic_permutation_reduces_to_B_l686_686020


namespace CM_perp_EM_angle_CD_plane_MCE_l686_686624

noncomputable def EA_perp_plane_ABC : Prop := sorry -- Define relation
noncomputable def DB_perp_plane_ABC : Prop := sorry -- Define relation
noncomputable def AC_perp_BC : Prop := sorry -- Define relation

def BC (AE : ℝ) : ℝ := (3/2) * AE
def BD (AE : ℝ) : ℝ := (3/2) * AE
def a (AE : ℝ) : ℝ := (3/2) * AE
def AC (a : ℝ) : ℝ := Real.sqrt 2 * a
def AM (MB : ℝ) : Prop := 2 * MB

theorem CM_perp_EM (AE : ℝ) (a : ℝ) (MB : ℝ)
    (h1 : EA_perp_plane_ABC)
    (h2 : DB_perp_plane_ABC)
    (h3 : AC_perp_BC) 
    (h4 : BC AE = a)
    (h5 : BD AE = a)
    (h6 : AC a = Real.sqrt 2 * a)
    (h7 : AM MB) : sorry :=
sorry

theorem angle_CD_plane_MCE (AE : ℝ) (a : ℝ) (MB : ℝ)
    (h1 : EA_perp_plane_ABC)
    (h2 : DB_perp_plane_ABC)
    (h3 : AC_perp_BC) 
    (h4 : BC AE = a)
    (h5 : BD AE = a)
    (h6 : AC a = Real.sqrt 2 * a)
    (h7 : AM MB) : 
    ∃ (θ : ℝ), θ = Real.arcsin (Real.sqrt 6 / 3) :=
sorry

end CM_perp_EM_angle_CD_plane_MCE_l686_686624


namespace min_value_of_c_l686_686750

noncomputable def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

theorem min_value_of_c (c : ℕ) (n m : ℕ) (h1 : 5 * c = n^3) (h2 : 3 * c = m^2) : c = 675 := by
  sorry

end min_value_of_c_l686_686750


namespace compute_a_plus_b_l686_686909

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def diameter_of_larger_sphere : ℝ :=
  let smaller_volume := volume 6
  let larger_volume := 3 * smaller_volume
  let larger_radius := Real.cbrt (larger_volume * 3 / (4 * Real.pi))
  2 * larger_radius

theorem compute_a_plus_b : 
  ∃ (a b : ℝ), 
    diameter_of_larger_sphere = a * Real.cbrt b ∧ 
    (∃(a_int b_int : ℕ), a = a_int ∧ b = b_int ∧ Nat.coprime b_int (nat.cubed a_int) ∧ (a_int + b_int = 14)) :=
by
  sorry

end compute_a_plus_b_l686_686909


namespace cylinder_volume_ratio_l686_686223

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l686_686223


namespace distinct_units_digits_of_perfect_cube_l686_686505

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686505


namespace coefficient_of_term_in_binomial_expansion_l686_686065

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686065


namespace distinct_units_digits_of_cube_l686_686587

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686587


namespace distance_between_x_intercepts_l686_686838

theorem distance_between_x_intercepts 
  (m₁ m₂ : ℝ) (p : ℝ × ℝ) (h₁ : m₁ = 4) (h₂ : m₂ = -3) (h₃ : p = (8, 20)) : 
  let x₁ := (20 - (20 - m₁ * (8 - 8))) / m₁,
      x₂ := (20 - (20 - m₂ * (8 - 8))) / m₂ in
  |x₁ - x₂| = 35 / 3 :=
by
  sorry

end distance_between_x_intercepts_l686_686838


namespace binomial_product_l686_686897

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l686_686897


namespace distinct_units_digits_of_perfect_cubes_l686_686447

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686447


namespace possible_number_of_bores_l686_686200

-- Definitions based on the given conditions
def native : Type := ℕ
def knight (n : native) : Prop := sorry -- A knight always tells the truth
def liar (n : native) : Prop := sorry -- A liar always lies
def bore (n : native) : Prop := sorry -- A bore lies if at least one knight is next to them, otherwise, they say anything

-- There are 7 natives in a circular arrangement
def natives := fin 7

-- Each native claims, "Both of my neighbors are bores"
def claim_neighbors_are_bores (n : natives) : Prop := sorry

-- Each native is either a knight, liar, or a bore
def tribes (n : natives) : Prop := knight n ∨ liar n ∨ bore n

-- There is at least one representative from each tribe
def has_all_tribes : Prop := ∃ k : natives, knight k ∧ ∃ l : natives, liar l ∧ ∃ b : natives, bore b

-- The possible values for the number of bores are 2, 3, or 4
theorem possible_number_of_bores : 
  (∃ B : ℕ, B = 2 ∨ B = 3 ∨ B = 4) ∧
  (∀ n : natives, claim_neighbors_are_bores n) ∧
  has_all_tribes :=
sorry

end possible_number_of_bores_l686_686200


namespace cloth_sold_eq_300_l686_686861

theorem cloth_sold_eq_300 
  (total_sp : ℝ) 
  (loss_per_m : ℝ) 
  (cp_per_m : ℝ) 
  (num_cloth : ℝ) :
  total_sp = 9000 → loss_per_m = 6 → cp_per_m = 36 →
  num_cloth = (total_sp / (cp_per_m - loss_per_m)) →
  num_cloth = 300 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have h5 : 9000 / (36 - 6) = 300 := by norm_num
  rw h5 at h4
  assumption

end cloth_sold_eq_300_l686_686861


namespace number_of_correct_conclusions_l686_686424

-- Define the planes and lines
variables {Line Plane : Type} [LinearSpace Line Plane] -- Assume a linear space structure

-- Define the perpendicular and parallel relationships
def perpendicular (x : Line) (y : Line) : Prop := sorry
def perpendicular (x : Line) (y : Plane) : Prop := sorry
def perpendicular (x : Plane) (y : Plane) : Prop := sorry

def parallel (x : Line) (y : Line) : Prop := sorry
def parallel (x : Line) (y : Plane) : Prop := sorry
def parallel (x : Plane) (y : Plane) : Prop := sorry

-- Define the conditions based on the problem
variables (m n : Line) (α β : Plane)

-- Conclusions
def conclusion1 := (perpendicular α β) ∧ (parallel m α) → perpendicular m β
def conclusion2 := (perpendicular m α) ∧ (perpendicular n β) ∧ (perpendicular m n) → perpendicular α β
def conclusion3 := (perpendicular m α) ∧ (parallel m β) → perpendicular α β
def conclusion4 := (parallel m α) ∧ (parallel n β) ∧ (parallel m n) → parallel α β

-- Theorem statement
theorem number_of_correct_conclusions : 
  (conclusion1 m n α β = false) ∧
  (conclusion2 m n α β = true) ∧
  (conclusion3 m n α β = true) ∧
  (conclusion4 m n α β = false) → 
  2_correct_conclusions :=
sorry

end number_of_correct_conclusions_l686_686424


namespace greatest_five_digit_multiple_of_eleven_l686_686253

theorem greatest_five_digit_multiple_of_eleven :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧
  (10000 * B + 1000 * A + 100 * A + 10 * B + C) % 11 = 0 ∧
  (10000 * B + 1000 * A + 100 * A + 10 * B + C) = 96619 :=
begin
  -- proof goes here
  sorry
end

end greatest_five_digit_multiple_of_eleven_l686_686253


namespace simplify_tangent_expression_l686_686702

theorem simplify_tangent_expression :
  let t15 := Real.tan (15 * Real.pi / 180)
  let t30 := Real.tan (30 * Real.pi / 180)
  1 = Real.tan (45 * Real.pi / 180)
  calc
    (1 + t15) * (1 + t30) = 2 := by
  sorry

end simplify_tangent_expression_l686_686702


namespace binomial_coefficient_x3y5_in_expansion_l686_686173

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686173


namespace parallelogram_angles_l686_686611

theorem parallelogram_angles (x y : ℝ) (h_sub : y = x + 50) (h_sum : x + y = 180) : x = 65 :=
by
  sorry

end parallelogram_angles_l686_686611


namespace volume_ratio_cylinders_l686_686204

open Real

noncomputable def volume_ratio_larger_to_smaller (h₁ : Real) (h₂ : Real) (circumference₁ : Real) (circumference₂ : Real) : Real :=
  let r₁ := circumference₁ / (2 * pi)
  let r₂ := circumference₂ / (2 * pi)
  let V₁ := pi * r₁^2 * h₁
  let V₂ := pi * r₂^2 * h₂
  V₂ / V₁

theorem volume_ratio_cylinders : volume_ratio_larger_to_smaller 9 6 6 9 = 9 / 4 := by
  sorry

end volume_ratio_cylinders_l686_686204


namespace average_cost_per_individual_before_gratuity_l686_686242

theorem average_cost_per_individual_before_gratuity
  (total_bill : ℝ)
  (num_people : ℕ)
  (gratuity_percentage : ℝ)
  (bill_including_gratuity : total_bill = 840)
  (group_size : num_people = 7)
  (gratuity : gratuity_percentage = 0.20) :
  (total_bill / (1 + gratuity_percentage)) / num_people = 100 :=
by
  sorry

end average_cost_per_individual_before_gratuity_l686_686242


namespace radius_of_sphere_find_x_for_equation_l686_686753

-- Problem I2.1
theorem radius_of_sphere (r : ℝ) (V : ℝ) (h : V = 36 * π) : r = 3 :=
sorry

-- Problem I2.2
theorem find_x_for_equation (x : ℝ) (r : ℝ) (h_r : r = 3) (h : r^x + r^(1-x) = 4) (h_x_pos : x > 0) : x = 1 :=
sorry

end radius_of_sphere_find_x_for_equation_l686_686753


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686047

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686047


namespace no_integer_solution_for_sum_of_fourth_powers_l686_686713

theorem no_integer_solution_for_sum_of_fourth_powers :
  ¬ ∃ (n : Fin 14 → ℤ), (∑ i, (n i)^4) = 1599 :=
by
  sorry

end no_integer_solution_for_sum_of_fourth_powers_l686_686713


namespace parallelogram_area_l686_686882

variables (a b : Mathlib.Vec3) (u v : Mathlib.Vec3)

def cross_product_area (u v : Mathlib.Vec3) : ℝ := Mathlib.norm (Mathlib.cross u v)

theorem parallelogram_area
  (hab : Mathlib.norm (Mathlib.cross a b) = 20) 
  (hu : u = (1/2 : ℝ) • a + (5/2 : ℝ) • b)
  (hv : v = 3 • a - 2 • b) : 
  cross_product_area u v = 130 :=
sorry

end parallelogram_area_l686_686882


namespace algorithm_output_l686_686981

theorem algorithm_output (x y: Int) (h_x: x = -5) (h_y: y = 15) : 
  let x := if x < 0 then y + 3 else x;
  x - y = 3 ∧ x + y = 33 :=
by
  sorry

end algorithm_output_l686_686981


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686075

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686075


namespace distinct_cube_units_digits_l686_686463

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686463


namespace distinct_units_digits_of_integral_cubes_l686_686526

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686526


namespace calculate_expression_l686_686884

theorem calculate_expression:
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 :=
by
  sorry

end calculate_expression_l686_686884


namespace sum_sequence_2022_l686_686420

def sequence (a : ℕ → ℝ): Prop :=
a 1 = 1 ∧ ∀ n, a (n + 1) = (sqrt 3 * a n + 1) / (sqrt 3 - a n)

theorem sum_sequence_2022 :
  ∃ a : ℕ → ℝ, sequence a ∧ (∑ n in finset.range 2022, a (n + 1)) = 0 :=
sorry

end sum_sequence_2022_l686_686420


namespace coefficient_x_squared_poly_expansion_l686_686301

theorem coefficient_x_squared_poly_expansion :
  let poly := (2 * x + 1) * (x - 2) ^ 3 in
  (coeff poly 2 = 18) :=
sorry

end coefficient_x_squared_poly_expansion_l686_686301


namespace period_of_f_a_eq_1_and_max_val_and_symmetry_axis_l686_686994

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 + a

theorem period_of_f (a : ℝ) : period (λ x, f x a) = π := 
sorry

theorem a_eq_1_and_max_val_and_symmetry_axis :
  ∀ a : ℝ, (∀ x : ℝ, f x a >= 0) →
  (∃ x : ℝ, f x a = 0) →
  a = 1 ∧ 
  max (λ x, f x 1) = 4 ∧ 
  ∀ k : ℤ, ∃ x : ℝ, x = k * π / 2 + π / 6 ∧
  (f x 1) = max (λ x, f x 1) :=
sorry

end period_of_f_a_eq_1_and_max_val_and_symmetry_axis_l686_686994


namespace sequence_infinitely_divisible_by_2009_l686_686641

-- Given conditions and definitions
def sequence (a : ℕ) : ℕ → ℕ
| 0       => a
| (n + 1) => sequence n + 40^(factorial (n + 1))

theorem sequence_infinitely_divisible_by_2009 (a : ℕ) (ha : 0 < a) :
  ∃ᶠ n in at_top, 2009 ∣ sequence a n :=
sorry

end sequence_infinitely_divisible_by_2009_l686_686641


namespace distinct_units_digits_of_integral_cubes_l686_686519

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686519


namespace schedule_arrangements_l686_686613

-- Definitions based on the conditions in the problem
def morning_periods := {1, 2, 3, 4}
def afternoon_periods := {5, 6}

-- Subjects
inductive Subject
| Chinese
| Math
| English
| IT
| PE
| Geography

open Subject

-- Given conditions
def not_first_period (s : Subject) (sched : Fin 6 → Option Subject) : Prop :=
  sched 0 ≠ some s

def in_morning (s : Subject) (sched : Fin 6 → Option Subject) : Prop :=
  ∃ p, p ∈ morning_periods ∧ sched p = some s

-- Prove that the total number of different scheduling arrangements is 480
theorem schedule_arrangements : 
  ∃ sched : Fin 6 → Option Subject, 
    (∀ i, sched i ≠ none) ∧ 
    (∃ p, in_morning Math sched) ∧ 
    not_first_period PE sched ∧  
    (Finset.univ.image (λ i, sched i)).card = 6 ∧
    -- Calculation equality here
    (4 * 5 * 24 = 480) := 
sorry

end schedule_arrangements_l686_686613


namespace trapezoid_plane_l686_686783

/-- A quadrilateral is defined as a trapezoid if it has at least one pair of parallel sides -/
structure Trapezoid (V : Type*) [Inhabited V] :=
(a b c d : V)
(is_parallel : ∃ (t u v w : V), t ≠ u ∧ v ≠ w ∧
                (∀ (x : V), x ∈ affine_span ℝ ({t, u} : set V) ↔ x ∈ affine_span ℝ ({v, w} : set V)))

/-- If T is a trapezoid, then it uniquely determines a plane -/
theorem trapezoid_plane {V: Type*} [Inhabited V] (T: Trapezoid V) : 
  ∃ P : affine_subspace ℝ V, ∀ (x y : V), x ∈ P → y ∈ P → V ⊆ P :=
sorry

end trapezoid_plane_l686_686783


namespace distinct_units_digits_of_cube_l686_686480

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686480


namespace distinct_units_digits_of_cube_l686_686590

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686590


namespace distinct_units_digits_of_cubes_l686_686563

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686563


namespace price_reduction_eq_l686_686232

theorem price_reduction_eq (x : ℝ) (price_original price_final : ℝ) 
    (h1 : price_original = 400) 
    (h2 : price_final = 200) 
    (h3 : price_final = price_original * (1 - x) * (1 - x)) :
  400 * (1 - x)^2 = 200 :=
by
  sorry

end price_reduction_eq_l686_686232


namespace keith_picked_0_pears_l686_686667

structure Conditions where
  apples_total : ℕ
  apples_mike : ℕ
  apples_nancy : ℕ
  apples_keith : ℕ
  pears_keith : ℕ

theorem keith_picked_0_pears (c : Conditions) (h_total : c.apples_total = 16)
 (h_mike : c.apples_mike = 7) (h_nancy : c.apples_nancy = 3)
 (h_keith : c.apples_keith = 6) : c.pears_keith = 0 :=
by
  sorry

end keith_picked_0_pears_l686_686667


namespace sum_of_values_undefined_l686_686307

theorem sum_of_values_undefined (x : ℝ) :
  let y := (4 * x) / (3 * x^2 - 9 * x + 6) in
  (3 * x^2 - 9 * x + 6 = 0) → x = 1 ∨ x = 2 →
  ((if x = 1 then 1 else 0) + (if x = 2 then 2 else 0)) = 3 :=
by
  intro hx
  intro hx12
  cases hx12 with h1 h2
  · rw [if_pos h1, if_neg h2]
    simp
  · rw [if_neg, if_pos h2]
    simp
  intro
  intro
  sorry

end sum_of_values_undefined_l686_686307


namespace linear_inequality_in_one_variable_l686_686264

-- Define the given conditions as hypotheses
def inequality_condition_A : Prop := 2 * x - 1 > 0
def inequality_condition_B : Prop := -1 < 2
def inequality_condition_C : Prop := 3 * x - 2 * y ≤ -1
def inequality_condition_D : Prop := y^2 + 3 > 5

-- State the proof problem
theorem linear_inequality_in_one_variable (x y : ℝ) :
  (inequality_condition_A ∧ inequality_condition_B ∧ inequality_condition_C ∧ inequality_condition_D) → 
  (2 * x - 1 > 0) :=
by 
  sorry

end linear_inequality_in_one_variable_l686_686264


namespace distinct_units_digits_of_perfect_cube_l686_686514

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686514


namespace elise_loves_numbers_l686_686668

theorem elise_loves_numbers :
  ∃ (d_set : Set ℕ), (∀ n, (∃ k, n = k * 10 + 0 ∨ n = k * 10 + 1 ∨ n = k * 10 + 2 ∨ n = k * 10 + 3 ∨ n = k * 10 + 4 ∨ n = k * 10 + 5 ∨ n = k * 10 + 6 ∨ n = k * 10 + 7 ∨ n = k * 10 + 8 ∨ n = k * 10 + 9) → (n % 3 = 0 ↔ Sum.digits n % 3 = 0) → d ∈ d_set) ∧ d_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by {
  sorry
}

end elise_loves_numbers_l686_686668


namespace total_flowers_l686_686273

theorem total_flowers (F : ℕ) (h1: (2 / 5 : ℚ) * F = (2 / 5 : ℚ) * F) (h2: 10 + 14 = 24)
(h3: (3 / 5 : ℚ) * F = 24) : F = 40 :=
sorry

end total_flowers_l686_686273


namespace distinct_units_digits_of_perfect_cubes_l686_686451

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686451


namespace distinct_units_digits_of_cubes_l686_686433

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686433


namespace sum_f_1_to_2018_l686_686832

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -(x + 2)^2
  else if -1 ≤ x ∧ x < 3 then x
  else f (x - 6)

theorem sum_f_1_to_2018 : (∑ i in (finset.range 2018).image (λ n, n + 1), f (i : ℝ)) = 339 :=
sorry

end sum_f_1_to_2018_l686_686832


namespace sequence_inequality_l686_686419

open Nat

theorem sequence_inequality (k : ℝ) (a : ℕ → ℝ) (h_a : ∀ n : ℕ, a n = n^2 + k * n + 2) :
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) ↔ k > -3 :=
by
  sorry

end sequence_inequality_l686_686419


namespace correct_statement_l686_686786

-- Definitions based on the conditions
def three_points_determine_plane (p1 p2 p3 : Point) : Prop :=
  ¬ collinear p1 p2 p3

def line_and_point_determine_plane (l : Line) (p : Point) : Prop :=
  ¬ (p ∈ l)

def trapezoid_determine_plane (t : Trapezoid) : Prop :=
  ∃ p1 p2 p3 p4 : Point, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1 ∧ 
    p1 ≠ p3 ∧ p2 ≠ p4 ∧
    ¬ collinear p1 p2 p3 ∧
    ¬ collinear p1 p3 p4

def circle_center_and_points_determine_plane (c : Circle) (p1 p2 : Point) : Prop :=
  ¬ collinear c.center p1 p2

-- The proof problem
theorem correct_statement : 
  (three_points_determine_plane p1 p2 p3 ∨
   line_and_point_determine_plane l p ∨
   trapezoid_determine_plane t ∨
   circle_center_and_points_determine_plane c p1 p2) →
  trapezoid_determine_plane t :=
sorry

end correct_statement_l686_686786


namespace consecutive_even_sum_100_consecutive_even_sum_100_4_vars_l686_686898

theorem consecutive_even_sum_100 (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hx : even x) (hy : even y) (hz : even z) :
  ¬(x < y ∧ y < z ∧ x + y + z = 100 ∧ y = x + 2 ∧ z = x + 4) :=
sorry

theorem consecutive_even_sum_100_4_vars (x y z w : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < w) (hx : even x) (hy : even y) (hz : even z) (hw : even w) :
  x < y ∧ y < z ∧ z < w ∧ x + y + z + w = 100 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 :=
begin
  use 22,
  split,
  { exact even_two_mul_of_even (22 : ℕ) },
  split,
  { use 24,
    split,
    { exact even_two_mul_of_even (24 : ℕ) },
    split,
    { use 26,
      split,
      { exact even_two_mul_of_even (26 : ℕ) },
      split,
      { use 28,
        split,
        { exact even_two_mul_of_even (28 : ℕ) },
        split,
        { norm_num, },
        split,
        norm_num, },
    norm_num,}},
norm_num, sorry}

end consecutive_even_sum_100_consecutive_even_sum_100_4_vars_l686_686898


namespace isoclines_of_differential_equation_l686_686023

theorem isoclines_of_differential_equation :
  ∀ k : ℝ, ∃ y : ℝ → ℝ, ∀ (x : ℝ), y' x = 2 * x - y(x) ↔ (∀ x, y(x) = 2 * x - k) :=
by sorry

end isoclines_of_differential_equation_l686_686023


namespace units_digit_of_perfect_cube_l686_686502

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686502


namespace value_of_a_l686_686602

theorem value_of_a (a : ℚ) (h : a + a / 4 = 6 / 2) : a = 12 / 5 := by
  sorry

end value_of_a_l686_686602


namespace distinct_units_digits_of_cube_l686_686593

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686593


namespace zoo_revenue_l686_686673

def num_children_mon : ℕ := 7
def num_adults_mon : ℕ := 5
def num_children_tue : ℕ := 4
def num_adults_tue : ℕ := 2
def cost_child : ℕ := 3
def cost_adult : ℕ := 4

theorem zoo_revenue : 
  (num_children_mon * cost_child + num_adults_mon * cost_adult) + 
  (num_children_tue * cost_child + num_adults_tue * cost_adult) 
  = 61 := 
by
  sorry

end zoo_revenue_l686_686673


namespace distinct_units_digits_of_perfect_cubes_l686_686442

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686442


namespace solve_problem_l686_686263

noncomputable def f1 (x : ℝ) := |Real.cos (2 * x)|
noncomputable def f2 (x : ℝ) := |Real.sin (2 * x)|
noncomputable def f3 (x : ℝ) := Real.cos (|x|)
noncomputable def f4 (x : ℝ) := Real.sin (|x|)

theorem solve_problem (x : ℝ) : 
  ((∀ t, f1(x + t) = f1(x) ↔ t = (π / 2)) ∧ 
   MonotonicOn (λ x, f1(x)) (Set.Ioo (π/4) (π/2)))
:=
sorry

end solve_problem_l686_686263


namespace stormi_needs_more_money_to_afford_bicycle_l686_686723

-- Definitions from conditions
def money_washed_cars : ℕ := 3 * 10
def money_mowed_lawns : ℕ := 2 * 13
def bicycle_cost : ℕ := 80
def total_earnings : ℕ := money_washed_cars + money_mowed_lawns

-- The goal to prove 
theorem stormi_needs_more_money_to_afford_bicycle :
  (bicycle_cost - total_earnings) = 24 := by
  sorry

end stormi_needs_more_money_to_afford_bicycle_l686_686723


namespace music_festival_audience_l686_686272

theorem music_festival_audience (total_people : ℕ) : total_people = 431 :=
  let total_under_30_males := (total_people / 6) * (0.40 * 0.60 + 0.60 * 0.30 + 0.30 * 0.40 + 0.50 * 0.55 + 0.20 * 0.45 + 0.50 * 0.70) in
  total_under_30_males = 90 → total_people = 431 sorry

end music_festival_audience_l686_686272


namespace num_positive_integers_l686_686935

theorem num_positive_integers (N : ℕ) (h : N > 3) : (∃ (k : ℕ) (h_div : 48 % k = 0), k = N - 3) → (∃ (c : ℕ), c = 8) := sorry

end num_positive_integers_l686_686935


namespace minimum_a_l686_686854

open Real

theorem minimum_a (a : ℝ) : 
  (∀ t : ℝ, t ≥ 0 → 5 * (t + 1) ^ 2 + a / (t + 1) ^ 5 ≥ 24) ↔ 
  a ≥ 2 * sqrt ( (24 / 7) ^ 7 ) :=
begin
  sorry
end

end minimum_a_l686_686854


namespace remainder_of_2356912_div_8_l686_686770

theorem remainder_of_2356912_div_8 : 912 % 8 = 0 := 
by 
  sorry

end remainder_of_2356912_div_8_l686_686770


namespace farmer_field_area_l686_686240

theorem farmer_field_area (m : ℝ) (h : (3 * m + 5) * (m + 1) = 104) : m = 4.56 :=
sorry

end farmer_field_area_l686_686240


namespace discount_percentage_l686_686728

theorem discount_percentage
  (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : 37.5 = (SP - CP) / CP * 100) :
  (MP - SP) / MP * 100 = 12 := by
  sorry

end discount_percentage_l686_686728


namespace distinct_cube_units_digits_l686_686456

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686456


namespace num_distinct_units_digits_of_cubes_l686_686474

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686474


namespace problem_statement_l686_686647

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if 1 < x ∧ x < 2 then f (x - 2)
else if -1 < x ∧ x < 0 then -f (-x)
else if -2 < x ∧ x ≤ -1 then f (x + 2)
else 0  -- we should cover the remaining cases properly, this can be adjusted

theorem problem_statement : f (-5/2) = -1/2 := by
  sorry

end problem_statement_l686_686647


namespace no_such_circular_arrangement_l686_686632

theorem no_such_circular_arrangement (numbers : List ℕ) (h1 : numbers = List.range 1 2019) :
  ¬ ∃ f : ℕ → ℕ, (∀ n, f n ∈ numbers) ∧ (∀ n, is_odd (f n + f (n+1) + f (n+2) + f (n+3))) := sorry

end no_such_circular_arrangement_l686_686632


namespace positive_integers_dividing_product_l686_686317

theorem positive_integers_dividing_product (n : ℕ) (p : ℕ → ℕ) (k : ℕ) (h_prime_factors : ∀ i, p i ≤ k ∧ Prime (p i))
  (h_factorization : ∏ i in Finset.range k, p i = n)
  (h_divides : n ∣ ∏ i in Finset.range k, (p i + 1)) :
  ∃ r s : ℕ, (n = 2^r * 3^s) ∧ (0 ≤ s ∧ s ≤ r ∧ r ≤ 2 * s) :=
sorry

end positive_integers_dividing_product_l686_686317


namespace find_x_l686_686947

theorem find_x (x : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - i) * (Complex.ofReal x + i) = 1 + i) : x = 0 :=
by sorry

end find_x_l686_686947


namespace volleyballs_basketballs_difference_l686_686859

variable (V B : ℕ)

theorem volleyballs_basketballs_difference :
  (V + B = 14) →
  (4 * V + 5 * B = 60) →
  V - B = 6 :=
by
  intros h1 h2
  sorry

end volleyballs_basketballs_difference_l686_686859


namespace find_a_given_inequality_1_find_a_given_inequality_2_l686_686984

-- Definitions given as conditions
def f (x a : ℝ) := a * (x - 1) / (x - 2)

-- (1) Given condition and required proof of a == 1
theorem find_a_given_inequality_1 (a : ℝ) :
  (∀ (x : ℝ), 2 < x ∧ x < 3 → f x a > 2) → a = 1 :=
by
  intros h
  sorry

-- (2) Given condition and required proof of a < 2 * sqrt 2 - 3
theorem find_a_given_inequality_2 (a : ℝ) :
  (∀ (x : ℝ), 2 < x → f x a < x - 3) → a < 2 * Real.sqrt 2 - 3 :=
by
  intros h
  sorry

end find_a_given_inequality_1_find_a_given_inequality_2_l686_686984


namespace units_digit_of_perfect_cube_l686_686503

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686503


namespace inequality_proof_l686_686659

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 3) : 
  a^b + b^c + c^a ≤ a^2 + b^2 + c^2 := 
by sorry

end inequality_proof_l686_686659


namespace distinct_units_digits_of_cubes_l686_686431

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686431


namespace binomial_coefficient_x3y5_in_expansion_l686_686127

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686127


namespace even_iff_a_zero_max_value_f_l686_686989

noncomputable def f (x a : ℝ) : ℝ := -x^2 + |x - a| + a + 1

theorem even_iff_a_zero (a : ℝ) : (∀ x, f x a = f (-x) a) ↔ a = 0 :=
by {
  -- Proof is omitted
  sorry
}

theorem max_value_f (a : ℝ) : 
  ∃ max_val : ℝ, 
    ( 
      (-1/2 < a ∧ a ≤ 0 ∧ max_val = 5/4) ∨ 
      (0 < a ∧ a < 1/2 ∧ max_val = 5/4 + 2*a) ∨ 
      ((a ≤ -1/2 ∨ a ≥ 1/2) ∧ max_val = -a^2 + a + 1)
    ) :=
by {
  -- Proof is omitted
  sorry
}

end even_iff_a_zero_max_value_f_l686_686989


namespace remainder_of_50_pow_2019_plus_1_mod_7_l686_686001

theorem remainder_of_50_pow_2019_plus_1_mod_7 :
  (50 ^ 2019 + 1) % 7 = 2 :=
by
  sorry

end remainder_of_50_pow_2019_plus_1_mod_7_l686_686001


namespace distinct_units_digits_of_cubes_l686_686559

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686559


namespace distinct_units_digits_of_perfect_cube_l686_686511

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686511


namespace expected_value_is_100_cents_l686_686764

-- Definitions for the values of the coins
def value_quarter : ℕ := 25
def value_half_dollar : ℕ := 50
def value_dollar : ℕ := 100

-- Define the total value of all coins
def total_value : ℕ := 2 * value_quarter + value_half_dollar + value_dollar

-- Probability of heads for a single coin
def p_heads : ℚ := 1 / 2

-- Expected value calculation
def expected_value : ℚ := p_heads * ↑total_value

-- The theorem we need to prove
theorem expected_value_is_100_cents : expected_value = 100 :=
by
  -- This is where the proof would go, but we are omitting it
  sorry

end expected_value_is_100_cents_l686_686764


namespace sum_of_tangency_points_l686_686899

-- Define the function f(x)
def f(x : ℝ) : ℝ := max (-7 * x - 50) (max (2 * x - 2) (6 * x + 4))

-- Assume that p(x) is a quadratic polynomial tangent to f at three distinct points
-- Define x_1, x_2, x_3 to be the x-coordinates of points of tangency
-- We need to prove that x_1 + x_2 + x_3 = -4.5 under these conditions

theorem sum_of_tangency_points (x1 x2 x3 : ℝ)
  (h1 : ∃ a b c, is_quadratic (λ x, a * x^2 + b * x + c) ∧
                    is_tangent (λ x, a * x^2 + b * x + c) f x1 ∧
                    is_tangent (λ x, a * x^2 + b * x + c) f x2 ∧
                    is_tangent (λ x, a * x^2 + b * x + c) f x3 ∧
                    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  x1 + x2 + x3 = -4.5 := sorry

end sum_of_tangency_points_l686_686899


namespace coefficient_x3y5_in_expansion_l686_686163

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686163


namespace minimum_value_of_quadratic_l686_686774

theorem minimum_value_of_quadratic :
  ∃ x : ℝ, (x = 6) ∧ (∀ y : ℝ, (y^2 - 12 * y + 32) ≥ -4) :=
sorry

end minimum_value_of_quadratic_l686_686774


namespace simplify_fraction_l686_686789

theorem simplify_fraction (d : ℤ) : (5 + 4 * d) / 9 - 3 = (4 * d - 22) / 9 :=
by
  sorry

end simplify_fraction_l686_686789


namespace dinner_handshakes_l686_686756

def num_couples := 8
def num_people_per_couple := 2
def num_attendees := num_couples * num_people_per_couple

def shakes_per_person (n : Nat) := n - 2
def total_possible_shakes (n : Nat) := (n * shakes_per_person n) / 2

theorem dinner_handshakes : total_possible_shakes num_attendees = 112 :=
by
  sorry

end dinner_handshakes_l686_686756


namespace range_f_l686_686346

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_f : (Set.range f) = (Set.univ \ { -27 }) :=
by
  sorry

end range_f_l686_686346


namespace ratio_of_cylinder_volumes_l686_686211

theorem ratio_of_cylinder_volumes (h l : ℝ) (h_h : h = 6) (h_l : l = 9) :
  let r_C := h / (2 * Real.pi),
      V_C := Real.pi * r_C^2 * l,
      r_D := l / (2 * Real.pi),
      V_D := Real.pi * r_D^2 * h in
  if V_D > V_C then V_D / V_C = 3 / 4 else V_C / V_D = 3 / 4 := by
  sorry

end ratio_of_cylinder_volumes_l686_686211


namespace intervals_of_monotonicity_range_of_a_l686_686414

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * log x

theorem intervals_of_monotonicity (h : ∀ x, 0 < x → x ≠ e → f (-2) x = -2 * x + x * log x) :
  ((∀ x, 0 < x ∧ x < exp 1 → deriv (f (-2)) x < 0) ∧ (∀ x, x > exp 1 → deriv (f (-2)) x > 0)) :=
sorry

theorem range_of_a (h : ∀ x, e ≤ x → deriv (f a) x ≥ 0) : a ≥ -2 :=
sorry

end intervals_of_monotonicity_range_of_a_l686_686414


namespace exists_f_eq_n_l686_686663

open Nat Real

def P := {1, 2, 3, 4, 5}

def f (m k : ℕ) : ℕ := ∑ i in {1, 2, 3, 4, 5}, ⌊ (m * sqrt ((k + 1) / (i + 1)))⌋

theorem exists_f_eq_n (n : ℕ) (hn : 0 < n) : ∃ k ∈ P, ∃ m : ℕ, 0 < m ∧ f m k = n :=
sorry

end exists_f_eq_n_l686_686663


namespace simplify_tangent_expression_l686_686706

theorem simplify_tangent_expression :
  let t15 := Real.tan (15 * Real.pi / 180)
  let t30 := Real.tan (30 * Real.pi / 180)
  1 = Real.tan (45 * Real.pi / 180)
  calc
    (1 + t15) * (1 + t30) = 2 := by
  sorry

end simplify_tangent_expression_l686_686706


namespace chord_length_proof_l686_686290

noncomputable def chord_square_length (O_3 O_6 O_9 : ℝ) (r3 r6 r9 : ℝ) := 
  ∃ PQ : ℝ, 
  r3 = 3 ∧ 
  r6 = 6 ∧ 
  r9 = 9 ∧ 
  O_3 = O_6 + 6 ∧ 
  O_6 = O_9 - 3 ∧ 
  (PQ^2 = 224)

theorem chord_length_proof (O_3 O_6 O_9 : ℝ) (r3 r6 r9 : ℝ) (h : chord_square_length O_3 O_6 O_9 r3 r6 r9) : 
  ∃ PQ : ℝ, PQ^2 = 224 :=
begin
  sorry
end

end chord_length_proof_l686_686290


namespace graph_has_fixed_point_l686_686735

noncomputable def graphPassesThroughFixedPoint (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : Prop :=
  (1 : ℝ, 2 : ℝ) ∈ { p | ∃ x, p = (x, a^(x-1) + 1) }

theorem graph_has_fixed_point {a : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : graphPassesThroughFixedPoint a ha_pos ha_ne_one :=
  sorry

end graph_has_fixed_point_l686_686735


namespace distinct_units_digits_of_cubes_l686_686533

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686533


namespace distinct_units_digits_of_cubes_l686_686544

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686544


namespace coefficient_x3y5_in_expansion_l686_686156

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686156


namespace find_p_l686_686803

variable (m n p : ℝ)

theorem find_p (h1 : m = n / 7 - 2 / 5)
               (h2 : m + p = (n + 21) / 7 - 2 / 5) : p = 3 := by
  sorry

end find_p_l686_686803


namespace period_and_shift_of_f_l686_686373

theorem period_and_shift_of_f (x : ℝ) (m : ℝ) (f : ℝ → ℝ)
  (h_fx : ∀ y : ℝ, f(x) = real.cos(y)^2 + m * real.sin(2 * y))
  (h_symmetry : x = real.pi / 6) :
  (∀ y, f (y + real.pi) = f y) ∧
  (let g := λ y, f (y - real.pi / 3) in ∀ y, g y = 1/2 - real.cos (2 * y)) :=
begin
  sorry
end

end period_and_shift_of_f_l686_686373


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l686_686918

theorem problem_1 : 286 = 200 + 80 + 6 := sorry
theorem problem_2 : 7560 = 7000 + 500 + 60 := sorry
theorem problem_3 : 2048 = 2000 + 40 + 8 := sorry
theorem problem_4 : 8009 = 8000 + 9 := sorry
theorem problem_5 : 3070 = 3000 + 70 := sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l686_686918


namespace equilateral_triangle_inequality_l686_686658

-- Define the equilateral triangle and the point P inside it
structure EquilateralTriangle where
  A B C : Point
  (equilateral : dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B)

structure Point where
  x y : ℝ

structure LinesMeet where
  P : Point
  A₁ B₁ C₁ : Point
  A₁_on_BC : LineSegment P A₁ B₁ C₁ -- Line meeting side BC
  B₁_on_CA : LineSegment P B₁ A₁ C₁ -- Line meeting side CA
  C₁_on_AB : LineSegment P C₁ A₁ B₁ -- Line meeting side AB

-- Target statement to prove the inequality
theorem equilateral_triangle_inequality
  (ΔABC : EquilateralTriangle) (P : Point)
  (lines_meet : LinesMeet P ΔABC.A ΔABC.B ΔABC.C)
  : dist lines_meet.A₁ lines_meet.B₁ * dist lines_meet.B₁ lines_meet.C₁ * dist lines_meet.C₁ lines_meet.A₁
    ≥ dist lines_meet.A₁ ΔABC.B * dist lines_meet.B₁ ΔABC.C * dist lines_meet.C₁ ΔABC.A :=
by
  sorry

end equilateral_triangle_inequality_l686_686658


namespace optimal_game_outcome_l686_686019

theorem optimal_game_outcome :
  ∃ (choices : Fin 21 → Bool), -- Bool represents the sign, where true is positive and false is negative
  (∀ (i : Fin 21), (choices i) ∈ {true, false}) ∧ -- The sign must be either true or false
  (∀ (player_turns : ℕ → Fin 21),
    -- The player whose turn it is must alternate:
    ∀ n, (player_turns (2 * n) = i -> player_turns (2 * n + 1) = i + 1 ∧ (.choices i = .not (.choices (i + 1))))
    ∨ (player_turns (2 * n) = i + 1 -> player_turns (2 * n - 1) = i ∧ (.choices i = .not (.choices (i - 1))))) ∧
  -- The game must result in the absolute value of total sum being exactly 30:
  abs (∑ i in Finset.range 21, if choices i then i else -i) = 30
:= 
sorry

end optimal_game_outcome_l686_686019


namespace distinct_units_digits_of_cube_l686_686585

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686585


namespace derivatives_identity_l686_686682

noncomputable def derivative_of_sine_and_cosine : Prop :=
  (∀ (x : ℝ), deriv (λ x, sin x) x = cos x) ∧ (∀ (x : ℝ), deriv (λ x, cos x) x = - sin x)

theorem derivatives_identity (x : ℝ) : derivative_of_sine_and_cosine :=
  by
  sorry

end derivatives_identity_l686_686682


namespace quadratic_inequality_solution_set_l686_686978

variable {a b c : ℝ}

theorem quadratic_inequality_solution_set
  (h1 : a < 0)
  (h2 : b = -a)
  (h3 : c = -2a)
  (h4 : ∀ x, (-1 < x ∧ x < 2) ↔ ax^2 + bx + c > 0) :
  ∀ x, (x < -1 ∨ x > (1/2)) ↔ cx^2 + bx + a > 0 :=
by 
  sorry

end quadratic_inequality_solution_set_l686_686978


namespace distance_x_intercepts_correct_l686_686842

noncomputable def distance_between_x_intercepts : ℝ :=
  let slope1 : ℝ := 4
  let slope2 : ℝ := -3
  let intercept_point : Prod ℝ ℝ := (8, 20)
  let line1 (x : ℝ) : ℝ := slope1 * (x - intercept_point.1) + intercept_point.2
  let line2 (x : ℝ) : ℝ := slope2 * (x - intercept_point.1) + intercept_point.2
  let x_intercept1 : ℝ := (0 - intercept_point.2) / slope1 + intercept_point.1
  let x_intercept2 : ℝ := (0 - intercept_point.2) / slope2 + intercept_point.1
  abs (x_intercept2 - x_intercept1)

theorem distance_x_intercepts_correct :
  distance_between_x_intercepts = 35 / 3 :=
sorry

end distance_x_intercepts_correct_l686_686842


namespace total_value_of_coins_l686_686017

theorem total_value_of_coins (q d : ℕ) (total_value original_value swapped_value : ℚ)
  (h1 : q + d = 30)
  (h2 : total_value = 4.50)
  (h3 : original_value = 25 * q + 10 * d)
  (h4 : swapped_value = 10 * q + 25 * d)
  (h5 : swapped_value = original_value + 1.50) :
  total_value = original_value / 100 :=
sorry

end total_value_of_coins_l686_686017


namespace min_value_of_a_and_b_l686_686945

theorem min_value_of_a_and_b (a b : ℝ) (h : a ^ 2 + 2 * b ^ 2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x ^ 2 + 2 * y ^ 2 = 6 → x + y ≥ m) ∧ (a + b = m) :=
sorry

end min_value_of_a_and_b_l686_686945


namespace coefficient_x3y5_in_expansion_l686_686152

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686152


namespace arithmetic_sequence_8th_term_l686_686725

theorem arithmetic_sequence_8th_term 
  (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 41) : 
  a + 7 * d = 59 := 
by 
  sorry

end arithmetic_sequence_8th_term_l686_686725


namespace sequence_split_equal_parts_l686_686998

theorem sequence_split_equal_parts (n : ℕ) :
  let a := λ n : ℕ, 2012 + 3 * n
  let S_n := ∑ i in Finset.range n, a (i + 1)
  (S_n % 2 = 0) ↔ 
    (∃ k : ℕ, n = 4 * (k + 1)) ∨ 
    (∃ k : ℕ, n = 4 * (k + 24) - 1) :=
by
  sorry

end sequence_split_equal_parts_l686_686998


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686078

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686078


namespace problem_solution_l686_686367

noncomputable def area_triangle_ABC
  (R : ℝ) 
  (angle_BAC : ℝ) 
  (angle_DAC : ℝ) : ℝ :=
  let α := angle_DAC
  let β := angle_BAC
  2 * R^2 * (Real.sin α) * (Real.sin β) * (Real.sin (α + β))

theorem problem_solution :
  ∀ (R : ℝ) (angle_BAC : ℝ) (angle_DAC : ℝ),
  R = 3 →
  angle_BAC = (Real.pi / 4) →
  angle_DAC = (5 * Real.pi / 12) →
  area_triangle_ABC R angle_BAC angle_DAC = 10 :=
by intros R angle_BAC angle_DAC hR hBAC hDAC
   sorry

end problem_solution_l686_686367


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686149

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686149


namespace binomial_coefficient_x3y5_in_expansion_l686_686133

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686133


namespace problem_solution_l686_686908

theorem problem_solution :
  { x : ℝ // (x / 4 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) } = { x : ℝ // x ∈ Set.Ico (-4 : ℝ) (-(3 / 2) : ℝ) } :=
by
  sorry

end problem_solution_l686_686908


namespace num_distinct_units_digits_of_cubes_l686_686473

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686473


namespace purchases_per_customer_l686_686014

noncomputable def number_of_customers_in_cars (num_cars : ℕ) (customers_per_car : ℕ) : ℕ :=
  num_cars * customers_per_car

def total_sales (sports_store_sales : ℕ) (music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

theorem purchases_per_customer {num_cars : ℕ} {customers_per_car : ℕ} {sports_store_sales : ℕ} {music_store_sales : ℕ}
    (h1 : num_cars = 10)
    (h2 : customers_per_car = 5)
    (h3 : sports_store_sales = 20)
    (h4: music_store_sales = 30) :
    (total_sales sports_store_sales music_store_sales / number_of_customers_in_cars num_cars customers_per_car) = 1 :=
by
  sorry

end purchases_per_customer_l686_686014


namespace arithmetic_to_geometric_l686_686959

variables {α : Type*} [linear_ordered_field α]

def is_arithmetic_sequence (u : ℕ → α) (d : α) : Prop :=
∀ n, u (n + 1) = u n + d

def is_geometric_sequence (v : ℕ → α) (q : α) : Prop :=
∀ n, v (n + 1) = v n * q

theorem arithmetic_to_geometric {u : ℕ → α} {v : ℕ → α} 
(d : α) (q : α) (h_arith : is_arithmetic_sequence u d) 
(h_geom : is_geometric_sequence v q) (d_ne_zero : d ≠ 0)
(h4 : u 4 = v 4) (h7 : u 7 = v 6) (h16 : u 16 = v 8) :
q = √3 ∨ q = -√3 :=
by sorry

end arithmetic_to_geometric_l686_686959


namespace chessboard_repainting_possible_l686_686679

-- Definition of the problem conditions
def is_possible_to_repaint (n : ℕ) : Prop :=
  ∃ operations : list (ℕ × ℕ), -- List of operations, each defined by the position on the chessboard
    operations.all (λ op, is_valid_op n op) ∧
    performs_desired_repainting n operations

-- Main theorem statement
theorem chessboard_repainting_possible (n : ℕ) (h : n ≥ 3) : 
  is_possible_to_repaint n ↔ (even n ∧ n ≥ 4) :=
sorry

end chessboard_repainting_possible_l686_686679


namespace correct_option_D_l686_686779

theorem correct_option_D (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2 * x :=
by sorry

end correct_option_D_l686_686779


namespace coefficient_x3y5_in_expansion_l686_686119

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686119


namespace proposition_b_proposition_c_proposition_d_l686_686777

section PropositionB

theorem proposition_b {x y : ℝ} (h : x^2 + y^2 = 4) : 
  ∃ x1 y1, (x1 - 1)^2 + (y1 - 2 * sqrt 2)^2 = 25 :=
sorry

end PropositionB


section PropositionC

theorem proposition_c : 
  ∃ a b, (a, b) ∈ { (x, y) | x^2 + y^2 + 4 * x - 4 * y = 0 } ∩ { (x, y) | x^2 + y^2 + 2 * x - 12 = 0 } → a - 2 * b + 6 = 0 :=
sorry

end PropositionC


section PropositionD

theorem proposition_d : 
  ∃ t1 t2 t3 : Line, ∀ L, L ∈ { t | isTangent t (circle (0, 0) 2) } ∧ { t | isTangent t (circle (-1, 0) 1) } → 
  {t1, t2, t3} = {t | isTangent t (circle (0, 0) 1) } ∨ {t | isTangent t (circle (2, 4) 4) } :=
sorry

end PropositionD

end proposition_b_proposition_c_proposition_d_l686_686777


namespace find_one_third_of_product_l686_686335

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end find_one_third_of_product_l686_686335


namespace remainder_when_divided_by_6_l686_686186

theorem remainder_when_divided_by_6 (n : ℕ) (last_digit_odd : ¬ even 5) (sum_digits_div_by_3 : 3 ∣ 39) :
  n % 6 = 3 :=
  by
    have : n = 369975 := sorry
    exact sorry

end remainder_when_divided_by_6_l686_686186


namespace binomial_coefficient_x3y5_in_expansion_l686_686166

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686166


namespace red_light_adds_3_minutes_l686_686248

-- Definitions (conditions)
def first_route_time_if_all_green := 10
def second_route_time := 14
def additional_time_if_all_red := 5

-- Given that the first route is 5 minutes longer when all stoplights are red
def first_route_time_if_all_red := second_route_time + additional_time_if_all_red

-- Define red_light_time as the time each stoplight adds if it is red
def red_light_time := (first_route_time_if_all_red - first_route_time_if_all_green) / 3

-- Theorem (question == answer)
theorem red_light_adds_3_minutes :
  red_light_time = 3 :=
by
  -- proof goes here
  sorry

end red_light_adds_3_minutes_l686_686248


namespace johns_starting_elevation_l686_686637

variable (horizontal_distance : ℝ) (final_elevation : ℝ) (initial_elevation : ℝ)
variable (vertical_ascent : ℝ)

-- Given conditions
axiom h1 : (vertical_ascent / horizontal_distance) = (1 / 2)
axiom h2 : final_elevation = 1450
axiom h3 : horizontal_distance = 2700

-- Prove that John's starting elevation is 100 feet
theorem johns_starting_elevation : initial_elevation = 100 := by
  sorry

end johns_starting_elevation_l686_686637


namespace calculate_expression_l686_686288

theorem calculate_expression :
    (2^(1/2) * 4^(1/2)) + (18 / 3 * 3) - 8^(3/2) = 18 - 14 * Real.sqrt 2 := 
by 
  sorry

end calculate_expression_l686_686288


namespace volume_ratio_l686_686220

def rect_height_vert : ℝ := 9
def rect_circumference_vert : ℝ := 6
def radius_vert : ℝ := rect_circumference_vert / (2 * Real.pi)
def volume_vert : ℝ := Real.pi * radius_vert^2 * rect_height_vert
-- volume_vert is calculated as πr^2h where r = 3/π

def rect_height_horiz : ℝ := 6
def rect_circumference_horiz : ℝ := 9
def radius_horiz : ℝ := rect_circumference_horiz / (2 * Real.pi)
def volume_horiz : ℝ := Real.pi * radius_horiz^2 * rect_height_horiz
-- volume_horiz is calculated as πr^2h where r = 9/(2π)

theorem volume_ratio : (max volume_vert volume_horiz) / (min volume_vert volume_horiz) = 3 / 4 := 
by sorry

end volume_ratio_l686_686220


namespace coeff_x3y5_in_expansion_l686_686030

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686030


namespace angle_between_a_and_c_l686_686407

variables {ℝ : Type*} [inner_product_space ℝ] (a b c : ℝ)
variables (h₁ : ¬ collinear ℝ a b)
variables (h₂ : inner a b ≠ 0)

def c := a - ((inner a a) / (inner a b)) • b

theorem angle_between_a_and_c :
  real.angle a c = real.pi / 2 :=
by
  sorry

end angle_between_a_and_c_l686_686407


namespace distinct_units_digits_of_cubes_l686_686566

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686566


namespace coefficient_of_term_in_binomial_expansion_l686_686066

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686066


namespace binomial_coefficient_x3y5_in_expansion_l686_686129

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686129


namespace coefficient_of_term_in_binomial_expansion_l686_686064

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686064


namespace num_distinct_units_digits_of_cubes_l686_686471

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686471


namespace number_of_subsets_of_B_l686_686999

theorem number_of_subsets_of_B :
  let A := {1, 2, 3}
  let B := {(x, y) | x y : ℕ // x ∈ A ∧ y ∈ A ∧ x + y ∈ A}
  ∃ (n : ℕ), (∀ (b : B), b ∈ B) ∧ (2 ^ 3 = 8) :=
by
  let A := {1, 2, 3}
  let B := {(x, y) | x y : ℕ // x ∈ A ∧ y ∈ A ∧ x + y ∈ A}
  existsi 3
  split
  · intros b
    exact b.2
  · simp only [pow_succ, pow_zero, mul_one, nat.pow_succ] at *
    exact 2 ^ 3 = 8
  sorry

end number_of_subsets_of_B_l686_686999


namespace garden_black_area_percentage_black_patches_percentage_l686_686847

noncomputable def black_patch_area_percentage : ℕ → ℝ
| 1 := π
| n := (3 * (n - 1) + 1) ^ 2 * π - black_patch_area_percentage (n - 1)

theorem garden_black_area_percentage :
  black_patch_area_percentage 5 = 103 * π :=
  sorry

theorem black_patches_percentage :
  (black_patch_area_percentage 5 / (169 * π)) * 100 = 61 :=
  sorry

end garden_black_area_percentage_black_patches_percentage_l686_686847


namespace solve_equation_one_solve_equation_two_l686_686715

theorem solve_equation_one (x : ℝ) : (x - 3) ^ 2 - 4 = 0 ↔ x = 5 ∨ x = 1 := sorry

theorem solve_equation_two (x : ℝ) : (x + 2) ^ 2 - 2 * (x + 2) = 3 ↔ x = 1 ∨ x = -1 := sorry

end solve_equation_one_solve_equation_two_l686_686715


namespace distinct_units_digits_perfect_cube_l686_686575

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686575


namespace no_such_triples_l686_686878

theorem no_such_triples : ¬ ∃ a b c : ℕ, 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Prime ((a-2)*(b-2)*(c-2)+12) ∧ 
  ((a-2)*(b-2)*(c-2)+12) ∣ (a^2 + b^2 + c^2 + a*b*c - 2017) := 
by sorry

end no_such_triples_l686_686878


namespace real_part_sum_l686_686943

-- Definitions of a and b as real numbers and i as the imaginary unit
variables (a b : ℝ)
def i := Complex.I

-- Condition given in the problem
def given_condition : Prop := (a + b * i) / (2 - i) = 3 + i

-- Statement to prove
theorem real_part_sum : given_condition a b → a + b = 20 := by
  sorry

end real_part_sum_l686_686943


namespace range_f_l686_686348

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_f : (Set.range f) = (Set.univ \ { -27 }) :=
by
  sorry

end range_f_l686_686348


namespace trigonometric_product_identity_l686_686313

theorem trigonometric_product_identity :
  3.416 * (cos (Real.pi / 33)) * (cos (2 * Real.pi / 33)) * (cos (4 * Real.pi / 33)) * (cos (8 * Real.pi / 33)) * (cos (16 * Real.pi / 33)) = 1 / 32 :=
by
  sorry

end trigonometric_product_identity_l686_686313


namespace quadrilateral_is_rectangle_l686_686376

theorem quadrilateral_is_rectangle
  (A B C D M : EuclideanGeometry.Point)
  (h_collinear : ¬Collinear ℝ ![A, B, M])
  (h_eq : dist M A ^ 2 + dist M C ^ 2 = dist M B ^ 2 + dist M D ^ 2) :
  (is_rectangle A B C D) :=
sorry

end quadrilateral_is_rectangle_l686_686376


namespace binomial_term_condition_and_coefficient_l686_686599

theorem binomial_term_condition_and_coefficient (a x : ℝ) (p q n : ℕ) :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n → a^p * x^q = a^(n-k) * x^k) →
  p + q = n ∧ binomial_coefficient n p = n.factorial / (p.factorial * (n - p).factorial) :=
by {
  sorry
}

end binomial_term_condition_and_coefficient_l686_686599


namespace zoo_revenue_is_61_l686_686675

def children_monday := 7
def children_tuesday := 4
def adults_monday := 5
def adults_tuesday := 2
def child_ticket_cost := 3
def adult_ticket_cost := 4

def total_children := children_monday + children_tuesday
def total_adults := adults_monday + adults_tuesday

def total_children_cost := total_children * child_ticket_cost
def total_adult_cost := total_adults * adult_ticket_cost

def total_zoo_revenue := total_children_cost + total_adult_cost

theorem zoo_revenue_is_61 : total_zoo_revenue = 61 := by
  sorry

end zoo_revenue_is_61_l686_686675


namespace candy_division_l686_686258

theorem candy_division (total_candy num_students : ℕ) (h1 : total_candy = 344) (h2 : num_students = 43) : total_candy / num_students = 8 := by
  sorry

end candy_division_l686_686258


namespace smallest_b_for_quadratic_inequality_l686_686928

theorem smallest_b_for_quadratic_inequality : 
  ∃ b : ℝ, (b^2 - 16 * b + 63 ≤ 0) ∧ ∀ b' : ℝ, (b'^2 - 16 * b' + 63 ≤ 0) → b ≤ b' := sorry

end smallest_b_for_quadratic_inequality_l686_686928


namespace concrete_weight_l686_686198

theorem concrete_weight (side_length : ℝ) (height : ℝ) (thickness : ℝ)
  (num_units : ℕ) (outer_corners : ℕ) (inner_corners : ℕ) (density : ℝ) :
  side_length = 2 →
  height = 3 →
  thickness = 0.5 →
  num_units = 32 →
  outer_corners = 10 →
  inner_corners = 6 →
  density = 2000 →
  let unadjusted_perimeter := num_units * side_length in
  let adjusted_perimeter := unadjusted_perimeter + (outer_corners - inner_corners) * thickness in
  let volume := adjusted_perimeter * height * thickness in
  let weight := volume * density in
  weight = 198000 :=
begin
  intros,
  sorry
end

end concrete_weight_l686_686198


namespace coefficient_x3y5_in_expansion_l686_686115

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686115


namespace proof_of_tree_calculation_proof_of_truck_allocation_l686_686724

noncomputable def calculate_trees : Prop :=
  ∃ trees : ℕ × ℕ, let (chinese_scholar, white_pines) := trees in
    chinese_scholar + white_pines = 320 ∧
    chinese_scholar = white_pines + 80

noncomputable def calculate_trucks : Prop :=
  ∃ (m n : ℕ), 
    let (typeA_trucks, typeB_trucks) := (m, n) in 
    typeA_trucks + typeB_trucks = 8 ∧  -- Total 8 trucks
    40 * typeA_trucks + 20 * typeB_trucks ≥ 200 ∧  -- Capacity constraint for Chinese scholar trees
    10 * typeA_trucks + 20 * typeB_trucks ≥ 120 ∧  -- Capacity constraint for white pines
    (2960 ≤ 400 * typeA_trucks + 360 * typeB_trucks ∧  -- Checking cost condition
     400 * typeA_trucks + 360 * typeB_trucks ≤ 3040) 

theorem proof_of_tree_calculation : calculate_trees := by
  sorry

theorem proof_of_truck_allocation : calculate_trucks := by
  sorry

end proof_of_tree_calculation_proof_of_truck_allocation_l686_686724


namespace percentage_off_sale_l686_686279

theorem percentage_off_sale (original_price sale_price : ℝ) (h₁ : original_price = 350) (h₂ : sale_price = 140) :
  ((original_price - sale_price) / original_price) * 100 = 60 :=
by
  sorry

end percentage_off_sale_l686_686279


namespace binomial_coefficient_x3y5_in_expansion_l686_686178

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686178


namespace distinct_units_digits_of_cubes_l686_686556

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686556


namespace line_plane_perpendicular_l686_686968

/-- Let m and n be two different lines in space. Let α, β, γ be different planes in space.
    Given that m is perpendicular to the plane α, m is not parallel to n, and n is not parallel to the plane β,
    then the plane α is perpendicular to the plane β. -/
theorem line_plane_perpendicular {m n : Line} {α β γ : Plane} (h1 : m ≠ n) (h2 : α ≠ β) (h3 : α ≠ γ) (h4 : β ≠ γ)
  (h5 : m ⊥ α) (h6 : ¬ m ∥ n) (h7 : ¬ n ∥ β) : α ⊥ β :=
sorry

end line_plane_perpendicular_l686_686968


namespace inequality_proof_l686_686950

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / (y + z) + y^2 * z / (z + x) + z^2 * x / (x + y) ≥ 1 / 2 * (x^2 + y^2 + z^2)) :=
by sorry

end inequality_proof_l686_686950


namespace min_votes_winner_l686_686618

-- Define the conditions
def num_candidates : ℕ := 4
def total_votes : ℕ := 61

-- Theorem stating the smallest number of votes the winner can receive
theorem min_votes_winner : ∀ (votes : ℕ), votes ≤ total_votes →
  ∃ winner_votes : ℕ, winner_votes ≥ votes ∧ winner_votes = 16 :=
by
  intro votes
  intro h_votes
  use 16
  constructor
  . exact Nat.le_of_lt (by norm_num)
  . rfl

-- Using sorry to skip the proof details
sorry

end min_votes_winner_l686_686618


namespace sum_of_intervals_length_1988_l686_686791

noncomputable def f (x : ℝ) : ℝ :=
  (∑ k in (Finset.range 70).map (λ k, k + 1), k / (x - k))

theorem sum_of_intervals_length_1988 :
  (∑ k in (Finset.range 70).map (λ k, k + 1), k / (x - k) ≥ 5/4) →
  ∑ i in (Finset.range 70).map (λ k, k + 1), (x - i) = 1988 :=
begin
  sorry
end

end sum_of_intervals_length_1988_l686_686791


namespace find_circle_equation_l686_686340

noncomputable def circleEquation (a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem find_circle_equation :
  ∃ (a b r : ℝ), b = 2 * a ∧ circleEquation 1 0 r ∧ circleEquation 0 2 r :=
begin
  use (1/2), (1), (sqrt(5)/2),
  split,
  { sorry },
  split,
  { sorry },
  { sorry },
end

end find_circle_equation_l686_686340


namespace distinct_units_digits_of_cubes_l686_686552

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686552


namespace books_number_in_series_l686_686758

-- Definitions and conditions from the problem
def number_books (B : ℕ) := B
def number_movies (M : ℕ) := M
def movies_watched := 61
def books_read := 19
def diff_movies_books := 2

-- The main statement to prove
theorem books_number_in_series (B M: ℕ) 
  (h1 : M = movies_watched)
  (h2 : M - B = diff_movies_books) :
  B = 59 :=
by
  sorry

end books_number_in_series_l686_686758


namespace ratio_GM_MF_l686_686623

variables {α : Type*}
variables {A B C D E F G : α}
variables {M : α}
variables [metric_space α] 

-- Definitions based on the conditions
def rectangle (A B C D : α) := 
  -- Here should be a proper definition of the rectangle, but we skip prove it for now
  true

def midpoint (E A D : α) := 
  -- E is the midpoint of AD
  true

def passes_through (FG G M : α) :=
  -- FG is a segment that passes through midpoint M of CE
  true

def midpoint_CE (M C E : α) := 
  -- M is the midpoint of CE
  true

-- The proposition to prove
theorem ratio_GM_MF {α : Type*} [metric_space α] (A B C D E F G M : α) 
  (h1 : rectangle A B C D) 
  (h2 : midpoint E A D) 
  (h3 : passes_through F G M) 
  (h4 : midpoint_CE M C E) :
  dist G M / dist M F = 1 / 3 :=
sorry

end ratio_GM_MF_l686_686623


namespace distinct_units_digits_of_cubes_l686_686543

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686543


namespace binomial_coefficient_x3y5_in_expansion_l686_686174

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686174


namespace solution_set_equivalent_l686_686004

noncomputable def inequality_solution_set (x : ℝ) : Prop :=
  4^x + 2^(x+2) - 12 > 0

theorem solution_set_equivalent (x : ℝ) : inequality_solution_set x ↔ x > 1 := 
sorry

end solution_set_equivalent_l686_686004


namespace ratio_of_cylinder_volumes_l686_686208

theorem ratio_of_cylinder_volumes (h l : ℝ) (h_h : h = 6) (h_l : l = 9) :
  let r_C := h / (2 * Real.pi),
      V_C := Real.pi * r_C^2 * l,
      r_D := l / (2 * Real.pi),
      V_D := Real.pi * r_D^2 * h in
  if V_D > V_C then V_D / V_C = 3 / 4 else V_C / V_D = 3 / 4 := by
  sorry

end ratio_of_cylinder_volumes_l686_686208


namespace distinct_units_digits_of_integral_cubes_l686_686525

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686525


namespace coeff_x3y5_in_expansion_l686_686029

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686029


namespace circle_center_radius_l686_686408

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 2 * x + 6 * y = 0 ↔ (x - 1)^2 + (y + 3)^2 = 10 :=
begin
  sorry
end

end circle_center_radius_l686_686408


namespace distinct_units_digits_of_cube_l686_686594

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686594


namespace simplify_tangent_expression_l686_686705

theorem simplify_tangent_expression :
  let t15 := Real.tan (15 * Real.pi / 180)
  let t30 := Real.tan (30 * Real.pi / 180)
  1 = Real.tan (45 * Real.pi / 180)
  calc
    (1 + t15) * (1 + t30) = 2 := by
  sorry

end simplify_tangent_expression_l686_686705


namespace simplify_tan_expression_l686_686710

theorem simplify_tan_expression :
  ∀ (tan_15 tan_30 : ℝ), 
  (tan_45 : ℝ) 
  (tan_45_eq_one : tan_45 = 1)
  (tan_sum_formula : tan_45 = (tan_15 + tan_30) / (1 - tan_15 * tan_30)),
  (1 + tan_15) * (1 + tan_30) = 2 :=
by
  intros tan_15 tan_30 tan_45 tan_45_eq_one tan_sum_formula
  sorry

end simplify_tan_expression_l686_686710


namespace num_solutions_sin_eq_exp_3_over_4_l686_686345

theorem num_solutions_sin_eq_exp_3_over_4 (x : ℝ) :
  set.countl{ 
    x ∈ Ioc 0 (50 * real.pi) | real.sin x = (3/4) ^ x 
  } = 50 :=
sorry

end num_solutions_sin_eq_exp_3_over_4_l686_686345


namespace mass_of_alcl3_formed_l686_686412

noncomputable def molarMass (atomicMasses : List (ℕ × ℕ)) : ℕ :=
atomicMasses.foldl (λ acc elem => acc + elem.1 * elem.2) 0

theorem mass_of_alcl3_formed :
  let atomic_mass_al := 26.98
  let atomic_mass_cl := 35.45
  let molar_mass_alcl3 := 2 * atomic_mass_al + 3 * atomic_mass_cl
  let moles_al2co3 := 10
  let moles_alcl3 := 2 * moles_al2co3
  let mass_alcl3 := moles_alcl3 * molar_mass_alcl3
  mass_alcl3 = 3206.2 := sorry

end mass_of_alcl3_formed_l686_686412


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686069

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686069


namespace units_digit_of_perfect_cube_l686_686498

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686498


namespace zoo_revenue_is_61_l686_686676

def children_monday := 7
def children_tuesday := 4
def adults_monday := 5
def adults_tuesday := 2
def child_ticket_cost := 3
def adult_ticket_cost := 4

def total_children := children_monday + children_tuesday
def total_adults := adults_monday + adults_tuesday

def total_children_cost := total_children * child_ticket_cost
def total_adult_cost := total_adults * adult_ticket_cost

def total_zoo_revenue := total_children_cost + total_adult_cost

theorem zoo_revenue_is_61 : total_zoo_revenue = 61 := by
  sorry

end zoo_revenue_is_61_l686_686676


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l686_686717

theorem solve_equation1 (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
by sorry

theorem solve_equation2 (x : ℝ) : (x - 2)^2 = 5 * (x - 2) ↔ (x = 2 ∨ x = 7) :=
by sorry

theorem solve_equation3 (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) :=
by sorry

theorem solve_equation4 (x : ℝ) : x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l686_686717


namespace coeff_x3y5_in_expansion_l686_686038

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686038


namespace half_life_ratio_eq_one_l686_686829

-- Define necessary variables and conditions
variables {t T_A T_B : ℝ}
variable (NA_init : ℝ := 100)
variable (NB_init : ℝ := 200)
variable (NA_remain : ℝ := 50)
variable (NB_remain : ℝ := 100)

-- Define the decay model functions for both isotopes
def decay_A (t T_A : ℝ) : ℝ := NA_init * 2^(-t / T_A)
def decay_B (t T_B : ℝ) : ℝ := NB_init * 2^(-t / T_B)

-- State the proof problem
theorem half_life_ratio_eq_one 
  (hA : decay_A t T_A = NA_remain)
  (hB : decay_B t T_B = NB_remain) :
  T_B / T_A = 1 :=
by
  sorry

end half_life_ratio_eq_one_l686_686829


namespace total_museum_tickets_cost_l686_686818

theorem total_museum_tickets_cost (num_students num_teachers cost_student_ticket cost_teacher_ticket : ℕ) :
  num_students = 12 →
  num_teachers = 4 →
  cost_student_ticket = 1 →
  cost_teacher_ticket = 3 →
  num_students * cost_student_ticket + num_teachers * cost_teacher_ticket = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_museum_tickets_cost_l686_686818


namespace find_p_range_l686_686418

noncomputable def S (n : ℕ) : ℤ := (-1)^n * n

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1 else S n - S (n - 1)

def condition_inequality (p : ℝ) (n : ℕ) : Prop :=
  (a (n + 1) - p) * (a n - p) < 0

theorem find_p_range :
  (∀ n : ℕ, n ≥ 1 → condition_inequality p n) → -1 < p ∧ p < 3 :=
sorry

end find_p_range_l686_686418


namespace cylinder_volume_ratio_l686_686226

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l686_686226


namespace distinct_units_digits_of_perfect_cube_l686_686509

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686509


namespace jellybean_probability_l686_686261

/-- Abe holds 1 blue and 2 red jelly beans. 
    Bob holds 2 blue, 2 yellow, and 1 red jelly bean. 
    Each randomly picks a jelly bean to show the other. 
    What is the probability that the colors match? 
-/
theorem jellybean_probability :
  let abe_blue_prob := 1 / 3
  let bob_blue_prob := 2 / 5
  let abe_red_prob := 2 / 3
  let bob_red_prob := 1 / 5
  (abe_blue_prob * bob_blue_prob + abe_red_prob * bob_red_prob) = 4 / 15 :=
by
  sorry

end jellybean_probability_l686_686261


namespace problem1_problem2_l686_686948

-- Definitions for the problems' conditions
def z1 : ℂ := 5 + 10 * complex.I
def z2 : ℂ := 3 - 4 * complex.I

-- Problem 1: 
theorem problem1 : z2 - z1 = -2 - 14 * complex.I := sorry

-- Problem 2: 
theorem problem2 (z : ℂ) : (1 / z = 1 / z1 + 1 / z2) → (z = 5 - 5 / 2 * complex.I) := sorry

end problem1_problem2_l686_686948


namespace area_of_fourth_rectangle_l686_686243

theorem area_of_fourth_rectangle (a b c d : ℕ) (x y z w : ℕ)
  (h1 : a = x * y)
  (h2 : b = x * w)
  (h3 : c = z * w)
  (h4 : d = y * w)
  (h5 : (x + z) * (y + w) = a + b + c + d) : d = 15 :=
sorry

end area_of_fourth_rectangle_l686_686243


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686087

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686087


namespace min_value_of_expression_l686_686343

noncomputable def quadratic_function_min_value (a b c : ℝ) : ℝ :=
  (3 * (a * 1^2 + b * 1 + c) + 6 * (a * 0^2 + b * 0 + c) - (a * (-1)^2 + b * (-1) + c)) /
  ((a * 0^2 + b * 0 + c) - (a * (-2)^2 + b * (-2) + c))

theorem min_value_of_expression (a b c : ℝ)
  (h1 : b > 2 * a)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
  (h3 : a > 0) :
  quadratic_function_min_value a b c = 12 :=
sorry

end min_value_of_expression_l686_686343


namespace distinct_units_digits_of_integral_cubes_l686_686524

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686524


namespace systematic_sampling_l686_686754

theorem systematic_sampling (classes : ℕ) (students_per_class : ℕ) (student_number : ℕ) : 
  classes = 12 ∧ students_per_class = 50 ∧ student_number = 14 →
  "Systematic sampling" :=
by
  -- Conditions
  intros h
  sorry

end systematic_sampling_l686_686754


namespace domain_f_l686_686910

open set

noncomputable
def domain_of_function (f : ℝ → ℝ) : set ℝ :=
  {x : ℝ | ∃ y, f x = y}

theorem domain_f :
  (domain_of_function (λ x, 1 / (x^2 + 6) + 1 / (x^3 + 6 * x) + 1 / (x^4 + 6)) =
  {x : ℝ | x ≠ 0}) :=
begin
  sorry,
end

end domain_f_l686_686910


namespace distinct_units_digits_of_perfect_cubes_l686_686444

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686444


namespace find_one_third_of_product_l686_686334

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end find_one_third_of_product_l686_686334


namespace max_value_sqrt_l686_686369

noncomputable def max_expression (a b : ℝ) : ℝ :=
  sqrt (a + 1) + sqrt (b + 3)

theorem max_value_sqrt (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 5) : 
  max_expression a b ≤ 3 * sqrt 2 := 
sorry

end max_value_sqrt_l686_686369


namespace coefficient_x3_y5_in_binomial_expansion_l686_686101

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686101


namespace factorial_divisible_count_divisible_factorials_l686_686361

theorem factorial_divisible (n : ℕ) (h : 1 ≤ n ∧ n ≤ 30) : n! % (List.prod (List.range (n + 1))) = 0 :=
by {
    sorry
}

theorem count_divisible_factorials : 
  (Finset.card (Finset.filter (λ n, n! % (List.prod (List.range (n + 1))) = 0) (Finset.range 31))) = 30 :=
by {
    sorry
}

end factorial_divisible_count_divisible_factorials_l686_686361


namespace market_value_of_10_percent_yielding_8_percent_stock_l686_686822

/-- 
Given:
1. The stock yields 8%.
2. It is a 10% stock, meaning the annual dividend per share is 10% of the face value.
3. Assume the face value of the stock is $100.

Prove:
The market value of the stock is $125.
-/
theorem market_value_of_10_percent_yielding_8_percent_stock
    (annual_dividend_per_share : ℝ)
    (face_value : ℝ)
    (dividend_yield : ℝ)
    (market_value_per_share : ℝ) 
    (h1 : face_value = 100)
    (h2 : annual_dividend_per_share = 0.10 * face_value)
    (h3 : dividend_yield = 8) :
    market_value_per_share = 125 := 
by
  /-
  Here, the following conditions are already given:
  1. face_value = 100
  2. annual_dividend_per_share = 0.10 * 100 = 10
  3. dividend_yield = 8
  
  We need to prove: market_value_per_share = 125
  -/
  sorry

end market_value_of_10_percent_yielding_8_percent_stock_l686_686822


namespace main_expr_equals_target_l686_686284

-- Define the improper fractions for the mixed numbers:
def mixed_to_improper (a b : ℕ) (c : ℕ) : ℚ := (a * b + c) / b

noncomputable def mixed_1 := mixed_to_improper 5 7 2
noncomputable def mixed_2 := mixed_to_improper 3 4 3
noncomputable def mixed_3 := mixed_to_improper 4 6 1
noncomputable def mixed_4 := mixed_to_improper 2 5 1

-- Define the main expression
noncomputable def main_expr := 47 * (mixed_1 - mixed_2) / (mixed_3 + mixed_4)

-- Define the target result converted to an improper fraction
noncomputable def target_result : ℚ := (11 * 99 + 13) / 99

-- The theorem to be proved: main_expr == target_result
theorem main_expr_equals_target : main_expr = target_result :=
by sorry

end main_expr_equals_target_l686_686284


namespace min_value_b_l686_686965

noncomputable def f (x a : ℝ) := 3 * x^2 - 4 * a * x
noncomputable def g (x a b : ℝ) := 2 * a^2 * Real.log x - b
noncomputable def f' (x a : ℝ) := 6 * x - 4 * a
noncomputable def g' (x a : ℝ) := 2 * a^2 / x

theorem min_value_b (a : ℝ) (h_a : a > 0) :
  ∃ (b : ℝ), ∃ (x₀ : ℝ), 
  (f x₀ a = g x₀ a b ∧ f' x₀ a = g' x₀ a) ∧ 
  ∀ (b' : ℝ), (∀ (x' : ℝ), (f x' a = g x' a b' ∧ f' x' a = g' x' a) → b' ≥ -1 / Real.exp 2) := 
sorry

end min_value_b_l686_686965


namespace distinct_units_digits_of_cubes_l686_686547

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686547


namespace fraction_equality_l686_686604

theorem fraction_equality (p q x y : ℚ) (hpq : p / q = 4 / 5) (hx : x / y + (2 * q - p) / (2 * q + p) = 1) :
  x / y = 4 / 7 :=
by {
  sorry
}

end fraction_equality_l686_686604


namespace second_subdivision_houses_l686_686271

theorem second_subdivision_houses 
    (anna_candy_per_house : ℕ) (billy_candy_per_house : ℕ) 
    (anna_houses : ℕ) (candy_difference : ℕ) (h : ℕ) 
    (anna_candy : ℕ := anna_candy_per_house * anna_houses)
    (billy_candy : ℕ := billy_candy_per_house * h) :
    anna_candy = billy_candy + candy_difference
    → h = 75 :=
by
  intros
  unfold anna_candy billy_candy at *
  sorry

end second_subdivision_houses_l686_686271


namespace largestQuotient_l686_686302

noncomputable def maxQuotient (s : Set ℤ) :=
  { (a, b) // a ∈ s ∧ b ∈ s ∧ a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b }

theorem largestQuotient : ∀ (a b : ℤ),
  a ∈ ({-30, -5, -1, 0, 3, 9} : Set ℤ) →
  b ∈ ({-30, -5, -1, 0, 3, 9} : Set ℤ) →
  a ≠ 0 →
  b ≠ 0 →
  a ≠ b →
  a / b ≤ 30 := 
sorry

end largestQuotient_l686_686302


namespace roots_difference_l686_686901

theorem roots_difference (a b c : ℝ) (h_eq : a = 1) (h_b : b = -11) (h_c : c = 24) :
    let r1 := (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    let r2 := (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    r1 - r2 = 5 := 
by
  sorry

end roots_difference_l686_686901


namespace problem_statement_l686_686807

-- Definitions of the conditions
def circle_tangent_x_axis (C : Real → Real → Prop) (T : Real × Real) : Prop :=
  T = (2, 0) ∧ ∃ h k r, C = (λ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2) ∧ h = 2 ∧ k = r

def circle_intersects_y_axis (C : Real → Real → Prop) (A B: Real × Real) (d : Real) : Prop :=
  d = 3 ∧ ∃ r, A = (0, r + d / 2) ∧ B = (0, r - d / 2) ∧ C = (λ (x y : ℝ), (x - 2)^2 + (y - r)^2 = r^2)

def ellipse_equation (P Q : Real × Real) : Prop :=
  ∃ k : Real, ∀ x : Real, ((x^2 / 8) + ((k * x + 1)^2 / 4) = 1 → (P = (x, k * x + 1) ∨ Q = (x, k * x + 1)))

def ray_bisects_angle (A B P Q : Real × Real) : Prop :=
  -- This is a placeholder for the actual bisecting proof
  sorry 

-- Problem statement combining the conditions and proof
theorem problem_statement :
  ∃ C A B P Q,
    circle_tangent_x_axis C (2, 0) ∧
    circle_intersects_y_axis C A B 3 ∧
    ellipse_equation P Q ∧
    ray_bisects_angle A B P Q :=
sorry

end problem_statement_l686_686807


namespace proof_problem_l686_686397

theorem proof_problem (a b : ℝ) (h1 : (5 * a + 2)^(1/3) = 3) (h2 : (3 * a + b - 1)^(1/2) = 4) :
  a = 5 ∧ b = 2 ∧ (3 * a - b + 3)^(1/2) = 4 :=
by
  sorry

end proof_problem_l686_686397


namespace same_score_exists_l686_686617

open Function Nat

def contestant_score (a b : ℕ) : ℕ := 10 * a + 5 * b

theorem same_score_exists : 
  ∀ (a b c : ℕ → ℕ) (N K : ℕ), 
  (∀ i, a i + b i + c i = K) →
  2 * ∑ i in range N, b i > N * K →
  (∀ i, a i = c i) →
  ¬ bijective (λ i, contestant_score (a i) (b i) : Fin N → ℕ) :=
by
  sorry

end same_score_exists_l686_686617


namespace find_k_l686_686357

def base_k_fraction_representation (k : ℕ) (h : 0 < k) : Prop :=
  (4 * k + 1) / (k^2 - 1) = 8 / 45

theorem find_k : ∃ k : ℕ, 0 < k ∧ base_k_fraction_representation k (by decide) :=
sorry

end find_k_l686_686357


namespace coefficient_x3y5_in_expansion_l686_686122

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686122


namespace distinct_cube_units_digits_l686_686465

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686465


namespace fourth_intersection_point_is_correct_l686_686622

theorem fourth_intersection_point_is_correct :
  ∃ (a b r : ℝ), 
    let C : ℝ → ℝ → Prop := λ x y, (x - a)^2 + (y - b)^2 = r^2 ∧ x * y = 2 in
    C 3 (2 / 3) ∧ C (-4) (-1 / 2) ∧ C (1 / 2) 4 ∧ 
    C (-2 / 3) (-3) :=
sorry

end fourth_intersection_point_is_correct_l686_686622


namespace problem_conditions_l686_686977

theorem problem_conditions (a b c x : ℝ) :
  (∀ x, ax^2 + bx + c ≥ 0 ↔ (x ≤ -3 ∨ x ≥ 4)) →
  (a > 0) ∧
  (∀ x, bx + c > 0 → x > -12 = false) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ (x < -1/4 ∨ x > 1/3)) ∧
  (a + b + c ≤ 0) :=
by
  sorry

end problem_conditions_l686_686977


namespace units_digit_of_perfect_cube_l686_686500

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686500


namespace distance_between_x_intercepts_l686_686843

-- Definitions for the conditions
def line_eq (m : ℝ) (x1 y1 : ℝ) : ℝ → ℝ := λ x, m * (x - x1) + y1

def x_intercept (m : ℝ) (x1 y1 : ℝ) : ℝ := classical.some (exists_x_intercept m x1 y1)

lemma exists_x_intercept (m x1 y1 : ℝ) : ∃ x : ℝ, line_eq m x1 y1 x = 0 :=
begin
  use (y1 / m + x1),
  simp [line_eq]
end

-- Main theorem
theorem distance_between_x_intercepts : 
  let l1 := line_eq 4 8 20,
      l2 := line_eq (-3) 8 20 in
  dist (x_intercept 4 8 20) (x_intercept (-3) 8 20) = 35 / 3 :=
by {
  -- Define equations of the lines
  let line1 := (λ x, 4 * (x - 8) + 20),
  let line2 := (λ x, -3 * (x - 8) + 20),

  -- Compute x-intercepts
  let x_int1 := classical.some (exists_x_intercept 4 8 20),
  let x_int2 := classical.some (exists_x_intercept (-3) 8 20),

  -- The derivation of x-intercepts and the distance calculation should be skipped for now
  sorry
}

end distance_between_x_intercepts_l686_686843


namespace find_z_l686_686409

-- Define complex numbers and the given condition
def complex (x y : ℝ) : ℂ := x + y * Complex.I

theorem find_z (z : ℂ) (h : z + 2*conj(z) = 3 - Complex.I) : z = 1 + Complex.I :=
sorry

end find_z_l686_686409


namespace solution_to_inequality_l686_686748

theorem solution_to_inequality : 
  ∀ x : ℝ, (x + 3) * (x - 1) < 0 ↔ -3 < x ∧ x < 1 :=
by
  intro x
  sorry

end solution_to_inequality_l686_686748


namespace distinct_cube_units_digits_l686_686454

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686454


namespace right_triangle_exponentiation_l686_686737

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def h (u : ℝ) : ℝ := 3 * Real.sqrt 5 * u

theorem right_triangle_exponentiation :
  let u := log_base 8 5,
      leg1 := log_base 8 125,
      leg2 := log_base 2 25,
      hypotenuse := h u
  in 8 ^ hypotenuse = 5 ^ (3 * Real.sqrt 5) :=
by
  let u := log_base 8 5
  let leg1 := log_base 8 125
  let leg2 := log_base 2 25
  let hypotenuse := h u
  sorry

end right_triangle_exponentiation_l686_686737


namespace distinct_cube_units_digits_l686_686461

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686461


namespace plane_divides_cd_l686_686252

-- Definitions for midpoints and ratio conditions
variables (A B C D M N K L : Point)
variable (P : Plane)

-- Conditions given in the problem
variable (h1 : midpoint A B M)
variable (h2 : midpoint A C N)
variable (h3 : divides_ratio P B D K 1 3)
variable (h4 : P.contains M)
variable (h5 : P.contains N)
variable (h6 : P.contains K)
variable (h7 : on_edge K B D)

-- Conclusion to be proved
theorem plane_divides_cd (h1 : midpoint A B M) (h2 : midpoint A C N) (h3 : divides_ratio P B D K 1 3) (h4 : P.contains M) (h5 : P.contains N) (h6 : P.contains K) (h7 : on_edge K B D) : divides_ratio P C D L 1 3 :=
sorry

end plane_divides_cd_l686_686252


namespace n_mult_n_plus_1_eq_square_l686_686920

theorem n_mult_n_plus_1_eq_square (n : ℤ) : (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := 
by sorry

end n_mult_n_plus_1_eq_square_l686_686920


namespace units_digit_of_perfect_cube_l686_686499

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686499


namespace sample_stats_l686_686417

def sample : List ℝ := [8, 6, 4, 7, 11, 6, 8, 9, 10, 5]

noncomputable def sample_mean (l : List ℝ) : ℝ :=
  (l.sum) / l.length

noncomputable def sample_median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  if l.length.even then
    (sorted.get! (l.length / 2 - 1) + sorted.get! (l.length / 2)) / 2
  else sorted.get! (l.length / 2)

theorem sample_stats :
  sample_mean sample = 7.4 ∧ sample_median sample = 7.5 := 
by
  sorry

end sample_stats_l686_686417


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686074

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686074


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686045

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686045


namespace pour_85_liters_within_8_weighings_l686_686863

theorem pour_85_liters_within_8_weighings :
  ∃ (flask: ℕ → ℕ), 
  (flask 0 = 1) ∧ 
  ∀ n, flask (n+1) = if n % 2 = 0 then 2 * (flask n) + 1 else 2 * (flask n) - 1 ∧ 
  flask 7 = 85 :=
begin
  sorry
end

end pour_85_liters_within_8_weighings_l686_686863


namespace binary_to_base4_after_flip_l686_686298

theorem binary_to_base4_after_flip :
  let original := "10111001" -- binary representation in string for easy manipulation
  let modified := "10111101" -- after flipping the third bit from the right
  convert_to_base4(modified) = "2331" := sorry

/--
Additional definitions to exist within the Lean scope
(e.g. functions for flipping bits and converting between bases)
may be needed for complete alignment.
-/

end binary_to_base4_after_flip_l686_686298


namespace inner_tetrahedron_volume_ratio_l686_686865

noncomputable def volume_ratio_of_tetrahedrons (s : ℝ) : ℝ :=
  let V_original := (s^3 * Real.sqrt 2) / 12
  let a := (Real.sqrt 6 / 9) * s
  let V_inner := (a^3 * Real.sqrt 2) / 12
  V_inner / V_original

theorem inner_tetrahedron_volume_ratio {s : ℝ} (hs : s > 0) : volume_ratio_of_tetrahedrons s = 1 / 243 :=
by
  sorry

end inner_tetrahedron_volume_ratio_l686_686865


namespace fill_tank_time_l686_686940

theorem fill_tank_time (hA : 18 > 0) (hB : 20 > 0) (hC : 45 > 0) (hD : 30 > 0) : 
  (60 / 7 : ℝ) ≈ 8.57 :=
by
  -- Explanation: Define the rates according to given conditions
  let rateA := 1 / (18 : ℝ)
  let rateB := 1 / (20 : ℝ)
  let rateC := -1 / (45 : ℝ)
  let rateD := 1 / (30 : ℝ)
  
  -- Combined rate is the sum of individual rates
  let combinedRate := rateA + rateB + rateC + rateD
  
  -- Verify the combined rate using the provided calculation
  have h_combined_rate: combinedRate = 7 / 60 := sorry
  
  -- Proving the final time to fill the tank
  calc
    60 / 7 : ℝ
      ... = 1 / (7 / 60) : by sorry

end fill_tank_time_l686_686940


namespace binary_to_decimal_l686_686903

theorem binary_to_decimal : (11010 : ℕ) = 26 := by
  sorry

end binary_to_decimal_l686_686903


namespace problem_solution_l686_686651

noncomputable def x : ℝ := Real.logb 2 8
noncomputable def y : ℝ := Real.logb 8 16

theorem problem_solution : x^2 * y = 12 := by
  define x
  define y
  sorry

end problem_solution_l686_686651


namespace distinct_units_digits_of_perfect_cube_l686_686510

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686510


namespace triangle_area_l686_686609

theorem triangle_area
  (a b c : ℝ)
  (h1 : b^2 + c^2 = a^2 - b * c)
  (h2 : (vector2.mk c 0) ⬝ (vector2.mk (b * cos (π / 3)) (b * sin (π / 3))) = -4) :
  (1 / 2) * b * c * real.sin (2 * π / 3) = 2 * real.sqrt 3 :=
by
  sorry

end triangle_area_l686_686609


namespace problem_part1_problem_part2_l686_686395

-- Part 1: Original Curve C Definitions
def parametric_C (α : ℝ) : ℝ × ℝ := 
  (1 + Real.sqrt 2 * Real.cos α, 1 + Real.sqrt 2 * Real.sin α)

def polar_equation_C (ρ θ : ℝ) : Prop := 
  ρ = 2 * Real.cos θ + 2 * Real.sin θ

def point_A_rectangular : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

def inside_curve (x y : ℝ) : Prop := 
  (x - 1) ^ 2 + (y - 1) ^ 2 < 2

-- Part 2: Transformed Curve C' Definitions
def parametric_C' (α : ℝ) : ℝ × ℝ := 
  (2 + 2 * Real.sqrt 2 * Real.cos α, 
   0.5 + Real.sqrt 2 / 2 * Real.sin α)

def equation_C' (x y : ℝ) : Prop := 
  (x - 2) ^ 2 + (y - 0.5) ^ 2 = 2

-- Test problem statements
theorem problem_part1 (α : ℝ) : 
  ∃ ρ θ, (x y : ℝ), (polar_equation_C ρ θ) ∧ (inside_curve x y) := 
sorry

theorem problem_part2 : 
  ∃ x y, (equation_C' x y) := 
sorry

end problem_part1_problem_part2_l686_686395


namespace binomial_coefficient_x3y5_in_expansion_l686_686136

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686136


namespace log2_real_coeff_sum_expansion_l686_686644

-- Using the Binomial Theorem, let's define the expansion of (1 + ix)^(2011)
noncomputable def binomial_expansion (x : ℂ) := 
  (1 + x)^(2011)

-- Define T as the sum of all real coefficients of the expansion
noncomputable def sum_real_coefficients : ℂ :=
  (binomial_expansion (complex.I) + binomial_expansion (-complex.I)) / 2

-- Define the statement to prove
theorem log2_real_coeff_sum_expansion : 
  log2 (abs (sum_real_coefficients)) = 1006 :=
by
  sorry

end log2_real_coeff_sum_expansion_l686_686644


namespace inequality_not_always_true_l686_686949

theorem inequality_not_always_true (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬(∀ a > 0, ∀ b > 0, (2 / ((1 / a) + (1 / b)) ≥ Real.sqrt (a * b))) :=
sorry

end inequality_not_always_true_l686_686949


namespace distinct_units_digits_of_cubes_l686_686427

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686427


namespace roots_of_cubic_8th_power_sum_l686_686745

theorem roots_of_cubic_8th_power_sum :
  ∀ a b c : ℂ, 
  (a + b + c = 0) → 
  (a * b + b * c + c * a = -1) → 
  (a * b * c = -1) → 
  (a^8 + b^8 + c^8 = 10) := 
by
  sorry

end roots_of_cubic_8th_power_sum_l686_686745


namespace sum_of_solutions_eq_one_sixth_l686_686930

-- Define the equation and the proof goal
theorem sum_of_solutions_eq_one_sixth :
  ∑ (x : ℂ) in { x | -10 * x = 3 * x * (x - 2) + 5 * (x + 2) ∧ x ≠ 2 ∧ x ≠ -2 } = 1/6 :=
by sorry

end sum_of_solutions_eq_one_sixth_l686_686930


namespace finite_square_set_max_elements_square_set_l686_686808

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def square_set (S : Set ℕ) : Prop :=
  ∀ x y ∈ S, is_square (x * y + 1)

theorem finite_square_set (S : Set ℕ) (hS : square_set S) : S.Finite :=
  sorry

theorem max_elements_square_set (S : Set ℕ) (hS : square_set S) : S.Finite → S.toFinite.toFinset.card ≤ 3 :=
  sorry

end finite_square_set_max_elements_square_set_l686_686808


namespace range_f_l686_686347

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_f : (Set.range f) = (Set.univ \ { -27 }) :=
by
  sorry

end range_f_l686_686347


namespace prob_both_red_prob_at_least_one_red_l686_686825

open classical

-- Given conditions
def balls_cnt : ℕ := 6
def white_balls : ℕ := 4
def red_balls : ℕ := 2
def draw_cnt : ℕ := 2

-- Event definitions
def event_red : set (finset ℕ) := {s | s = {0, 1}}
def event_at_least_one_red : set (finset ℕ) :=
  {s | ∃ r ∈ {0, 1}, r ∈ s}

noncomputable def probability (event : set (finset ℕ)) : ℚ :=
  (finset.card (finset.filter (λ s, s ∈ event) (finset.powerset_len draw_cnt (finset.range balls_cnt))) : ℚ) /
  (finset.card (finset.powerset_len draw_cnt (finset.range balls_cnt)) : ℚ)

-- Proof statements
theorem prob_both_red : probability event_red = 1 / 15 := sorry

theorem prob_at_least_one_red : probability event_at_least_one_red = 3 / 5 := sorry

end prob_both_red_prob_at_least_one_red_l686_686825


namespace equal_polygon_exists_l686_686633

variables (A B : Point)
noncomputable def equal_polygon_configuration (A B : Point) : Prop :=
  ∃ (P Q R S T U V W : Point),
  -- P, Q, R, S, T, U, V, W are the intermediate points
    (polygon A P Q R A ∧ polygon R S T U R ∧ polygon U V W A U)
  ∧ (distance A B = distance A P + distance P Q + distance Q R 
    + distance R S + distance S T + distance T U + distance U V + distance V W + distance W A)
  ∧ (polygon A P Q R A = polygon R S T U R)
  ∧ (polygon R S T U R = polygon U V W A U)
  ∧ no_intersection A P Q R 
  ∧ no_intersection R S T U 
  ∧ no_intersection U V W A 
  ∧ ends_only_shared_points A P Q R S T U V W.

-- Theorem statement
theorem equal_polygon_exists (A B : Point) :
  equal_polygon_configuration A B := 
sorry

end equal_polygon_exists_l686_686633


namespace veenapaniville_private_independent_district_A_l686_686799

theorem veenapaniville_private_independent_district_A :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_B_private := 2
  let remaining_schools := total_schools - district_A_schools - district_B_schools
  let each_kind_in_C := remaining_schools / 3
  let district_C_private := each_kind_in_C
  let district_A_private := private_schools - district_B_private - district_C_private
  district_A_private = 2 := by
  sorry

end veenapaniville_private_independent_district_A_l686_686799


namespace solve_for_y_l686_686305

theorem solve_for_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by sorry

end solve_for_y_l686_686305


namespace coefficient_x3y5_in_expansion_l686_686157

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686157


namespace arcsin_neg_one_half_l686_686888

theorem arcsin_neg_one_half : arcsin (-1/2) = -π/6 :=
by
  sorry

end arcsin_neg_one_half_l686_686888


namespace rational_quotient_of_positive_reals_l686_686642

noncomputable def proof_problem : Prop :=
∀ (A : set ℝ), 
  (A.nonempty) → 
  (∀ (a b c : ℝ), a ∈ A → b ∈ A → c ∈ A → (a * b + b * c + c * a ∈ ℚ)) → 
  (∀ (a b : ℝ), a ∈ A → b ∈ A → (a / b ∈ ℚ))

theorem rational_quotient_of_positive_reals (A : set ℝ) (h1 : A.nonempty)
  (h2 : ∀ (a b c : ℝ), a ∈ A → b ∈ A → c ∈ A → (a * b + b * c + c * a ∈ ℚ)) :
  ∀ (a b : ℝ), a ∈ A → b ∈ A → (a / b ∈ ℚ) := by 
  sorry

end rational_quotient_of_positive_reals_l686_686642


namespace largest_constant_l686_686342

theorem largest_constant (x y z : ℝ) : (x^2 + y^2 + z^2 + 3 ≥ 2 * (x + y + z)) :=
by
  sorry

end largest_constant_l686_686342


namespace binomial_product_l686_686895

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l686_686895


namespace distinct_units_digits_of_cubes_l686_686434

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686434


namespace distinct_units_digits_of_cubes_l686_686562

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686562


namespace distance_between_intersections_l686_686850

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 25 = 1

def is_focus_of_ellipse (fx fy : ℝ) : Prop := (fx = 0 ∧ (fy = 4 ∨ fy = -4))

def parabola_eq (x y : ℝ) : Prop := y = x^2 / 8 + 2

theorem distance_between_intersections :
  let d := 12 * Real.sqrt 2 / 5
  ∃ x1 x2 y1 y2 : ℝ, 
    ellipse_eq x1 y1 ∧ 
    parabola_eq x1 y1 ∧
    ellipse_eq x2 y2 ∧
    parabola_eq x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

end distance_between_intersections_l686_686850


namespace distinct_units_digits_perfect_cube_l686_686581

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686581


namespace set_P_equality_l686_686421

open Set

variable {U : Set ℝ} (P : Set ℝ)
variable (h_univ : U = univ) (h_def : P = {x | abs (x - 2) ≥ 1})

theorem set_P_equality : P = {x | x ≥ 3 ∨ x ≤ 1} :=
by
  sorry

end set_P_equality_l686_686421


namespace average_writing_speed_time_to_write_10000_words_l686_686869

-- Definitions based on the problem conditions
def total_words : ℕ := 60000
def total_hours : ℝ := 90.5
def writing_speed : ℝ := 663
def words_to_write : ℕ := 10000
def writing_time : ℝ := 15.08

-- Proposition that the average writing speed is 663 words per hour
theorem average_writing_speed :
  (total_words : ℝ) / total_hours = writing_speed :=
sorry

-- Proposition that the time to write 10,000 words at the given average speed is 15.08 hours
theorem time_to_write_10000_words :
  (words_to_write : ℝ) / writing_speed = writing_time :=
sorry

end average_writing_speed_time_to_write_10000_words_l686_686869


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686077

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686077


namespace air_filter_replacement_month_l686_686314

/-- Jenna replaces air filters every 7 months. The first replacement is in January.
    Prove that the 18th replacement will be in December. -/
theorem air_filter_replacement_month (n : ℕ) (h : n = 18) : 
  let first_replacement := 1 -- January
  let months_in_year := 12
  let month := (first_replacement + (7 * (n - 1)) % months_in_year) in
  month = 12 := 
by
  sorry

end air_filter_replacement_month_l686_686314


namespace museum_ticket_cost_l686_686812

theorem museum_ticket_cost (students teachers : ℕ) (student_ticket_cost teacher_ticket_cost : ℕ) 
  (h_students : students = 12) (h_teachers : teachers = 4) 
  (h_student_ticket_cost : student_ticket_cost = 1) (h_teacher_ticket_cost : teacher_ticket_cost = 3) :
  students * student_ticket_cost + teachers * teacher_ticket_cost = 24 :=
by
  rw [h_students, h_teachers, h_student_ticket_cost, h_teacher_ticket_cost]
  exact (12 * 1 + 4 * 3) 
-- This produces 24

end museum_ticket_cost_l686_686812


namespace complex_solution_l686_686714

theorem complex_solution 
  (z : ℂ) 
  (h : abs z ^ 2 + (z + conj z) * complex.i = 1 - complex.i) :
  z = -1/2 - sqrt 3 / 2 * complex.i ∨ z = -1/2 + sqrt 3 / 2 * complex.i :=
by { sorry }

end complex_solution_l686_686714


namespace infinite_nice_implies_l686_686643

-- Definition of a "nice" number
def nice (r k : ℕ) (n : ℕ) : Prop :=
  ∀ (s : ℕ), s + k ≤ (nat.digits 10 n).length →
    r ∣ nat.ofDigits 10 (list.take k (list.drop s (nat.digits 10 n)))

theorem infinite_nice_implies (r k : ℕ) (h1 : ∀ p, p ∣ r → 50 < p) 
  (h2 : ∃ n : ℕ, ∀ m ≥ n, nice r k m) : nice r k (10^k - 1) :=
sorry -- Proof required

end infinite_nice_implies_l686_686643


namespace simplify_tan_expression_l686_686708

theorem simplify_tan_expression :
  ∀ (tan_15 tan_30 : ℝ), 
  (tan_45 : ℝ) 
  (tan_45_eq_one : tan_45 = 1)
  (tan_sum_formula : tan_45 = (tan_15 + tan_30) / (1 - tan_15 * tan_30)),
  (1 + tan_15) * (1 + tan_30) = 2 :=
by
  intros tan_15 tan_30 tan_45 tan_45_eq_one tan_sum_formula
  sorry

end simplify_tan_expression_l686_686708


namespace audrey_completion_time_l686_686880

theorem audrey_completion_time :
  ∃ (A : ℝ), 
  let Ferris_time := 3 in           -- Ferris can complete the job alone in 3 hours
  let total_time := 2 in            -- Audrey and Ferris together completed the job in 2 hours
  let Ferris_breaks := 6 * 5 / 60 in-- Ferris took 6 breaks of 5 minutes each
  let Ferris_work_time := total_time - Ferris_breaks in
  2 * (1 / A) + Ferris_work_time * (1 / Ferris_time) = 1 ∧
  A = 4 :=                          -- Audrey can complete the job alone in 4 hours
begin
  use 4,
  simp,
  sorry
end

end audrey_completion_time_l686_686880


namespace number_of_correct_conclusions_l686_686760

theorem number_of_correct_conclusions :
  let sum_of_triangle_sides_implies_sum_of_tetrahedron_faces := 
      ∀(s1 s2 s3 s4 : ℝ), s1 + s2 > s3 → s1 + s3 > s2 → s2 + s3 > s1 →
        s1 + s2 + s3 > s4 →
      ∀ (a b c d : ℝ), a + b + c > d
  let arith_seq_condition := 
      ∀ (a : ℕ → ℝ), ( ∑ i in (finset.Ico 6 11), a i / 5 = ∑ i in (finset.range 15), a i / 15)
  let geom_seq_conclusion := 
      ∀ (b : ℕ → ℝ), ( finset.prod (finset.Ico 6 11) b ^ (1/5) = finset.prod (finset.range 15) b ^ (1/15))
  let real_num_op_associative := 
      ∀ (a b c : ℝ), (a * b) * c = a * (b * c)
  let vector_op_non_associative := 
      ∀ (a b c : ℝ), (a * b) * c ≠ a * (b * c)
in sum_of_triangle_sides_implies_sum_of_tetrahedron_faces ∧ arith_seq_condition ∧ geom_seq_conclusion ∧ real_num_op_associative ∧ ¬vector_op_non_associative (
  2
).
sorry

end number_of_correct_conclusions_l686_686760


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686143

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686143


namespace number_of_new_bottle_caps_l686_686904

def threw_away := 6
def total_bottle_caps_now := 60
def found_more_bottle_caps := 44

theorem number_of_new_bottle_caps (N : ℕ) (h1 : N = threw_away + found_more_bottle_caps) : N = 50 :=
sorry

end number_of_new_bottle_caps_l686_686904


namespace remainder_of_expression_l686_686355

theorem remainder_of_expression (a b c d : ℕ) (h1 : a = 8) (h2 : b = 20) (h3 : c = 34) (h4 : d = 3) :
  (a * b ^ c + d ^ c) % 7 = 5 := 
by 
  rw [h1, h2, h3, h4]
  sorry

end remainder_of_expression_l686_686355


namespace number_of_dark_triangles_l686_686002

theorem number_of_dark_triangles (n : ℕ) : 
  ∑ i in finset.range (n + 1), (i + 1) * (i + 2) / 2 = (n * (n + 1) * (n + 2)) / 6 :=
by sorry

end number_of_dark_triangles_l686_686002


namespace domain_of_log_function_l686_686731

theorem domain_of_log_function (a : ℝ) (h : 0 < a ∧ a ≠ 1) :
  ∀ x : ℝ, f(x) = (log a (2^x - 1/2)) → (x > -1) := 
sorry

end domain_of_log_function_l686_686731


namespace number_of_pairs_l686_686344

/-- Define the elements of set S as -1, 0, 1, 2. -/
def S : Set Int := {-1, 0, 1, 2}

/-- Check if the quadratic equation ax^2 + 2x + b = 0 has real solutions. -/
def has_real_solutions (a b : Int) : Prop :=
  a = 0 ∨ 4 - 4 * a * b ≥ 0

/-- Main theorem: Finding the number of ordered pairs (a, b) such that a, b ∈ S and ax^2 + 2x + b = 0 has real solutions yields 13 pairs. -/
theorem number_of_pairs : (Set.toFinset {p : Int × Int | p.1 ∈ S ∧ p.2 ∈ S ∧ has_real_solutions p.1 p.2}).card = 13 :=
sorry

end number_of_pairs_l686_686344


namespace inscribe_rectangle_area_one_third_l686_686628

theorem inscribe_rectangle_area_one_third (A B C M N P Q : Point) (AB AC BC AD x : ℝ) 
  (hA : A ≠ B) (h1 : on_line P BC) (h2 : on_line Q BC)
  (h3 : on_line M AB) (h4 : on_line N AC) 
  (h5 : area_rectangle M N P Q = 1/3 * area_triangle A B C)
  (h6 : height_from A to BC = AD) : 
  distance A M = AB / real.sqrt 6 :=
sorry

end inscribe_rectangle_area_one_third_l686_686628


namespace distinct_units_digits_of_cubes_l686_686531

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686531


namespace total_amount_l686_686795

theorem total_amount (x y z : ℝ) 
  (hx : y = 0.45 * x) 
  (hz : z = 0.50 * x) 
  (hy_share : y = 63) : 
  x + y + z = 273 :=
by 
  sorry

end total_amount_l686_686795


namespace binomial_coefficient_x3y5_in_expansion_l686_686177

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686177


namespace area_EFGH_is_22_l686_686625

noncomputable def area_of_trapezoid (E F G H : ℝ × ℝ) : ℝ :=
  let base1 := (F.snd - E.snd).abs
  let base2 := (G.snd - H.snd).abs
  let height := (G.fst - E.fst).abs
  (base1 + base2) * height / 2

theorem area_EFGH_is_22 : 
  let E := (2, -3) : ℝ × ℝ
  let F := (2, 2) : ℝ × ℝ
  let G := (6, 8) : ℝ × ℝ
  let H := (6, 2) : ℝ × ℝ
  area_of_trapezoid E F G H = 22 :=
by
  sorry

end area_EFGH_is_22_l686_686625


namespace hyperbola_properties_l686_686338

noncomputable def asymptotes_and_eccentricity (a b : ℝ) (hyp : ∀ x y : ℝ, y^2 / a - x^2 / b = 1) : (set (ℝ × ℝ) × ℝ) :=
let c := real.sqrt (a^2 + b^2) in
let e := c / a in
({(λ x, 2 * x), (λ x, -2 * x)}, e)

theorem hyperbola_properties :
  asymptotes_and_eccentricity 4 1 (λ x y, y^2 / 4 - x^2 = 1) = 
  ({(λ x, 2 * x), (λ x, -2 * x)}, real.sqrt 5 / 2) :=
sorry

end hyperbola_properties_l686_686338


namespace coeff_x3y5_in_expansion_l686_686036

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686036


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686081

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686081


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686145

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686145


namespace sum_possible_values_l686_686914

theorem sum_possible_values (x : ℕ → ℝ) (N : ℕ) (hN : 2 * N = 2004)
  (hx : ∀ i, x i = real.sqrt 2 - 1 ∨ x i = real.sqrt 2 + 1) :
  ¬ (∑ k in finset.range N, x (2 * k) * x (2 * k + 1) = 2004) ∧
  (∀ (m : ℕ), m ∈ finset.range 502 → R ∈ {1002 + 4 * m}) :=
by
  sorry

end sum_possible_values_l686_686914


namespace adolfo_blocks_l686_686872

theorem adolfo_blocks (original_blocks added_blocks : ℕ) (h1 : original_blocks = 35) (h2 : added_blocks = 30) : original_blocks + added_blocks = 65 := by
  -- original_blocks equals 35
  rw h1
  -- added_blocks equals 30
  rw h2
  -- prove the final statement
  exact Nat.add_eq_of_eq_sub (by norm_num : 35 + 30 = 65) (by norm_num : 65 - 35 = 30)

end adolfo_blocks_l686_686872


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l686_686716

theorem solve_equation1 (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
by sorry

theorem solve_equation2 (x : ℝ) : (x - 2)^2 = 5 * (x - 2) ↔ (x = 2 ∨ x = 7) :=
by sorry

theorem solve_equation3 (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) :=
by sorry

theorem solve_equation4 (x : ℝ) : x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l686_686716


namespace quadratic_inequality_l686_686974

noncomputable def quadratic_solution_set (a b c : ℝ) (x : ℝ) : Prop :=
ax^2 + bx + c ≥ 0

theorem quadratic_inequality (a b c : ℝ) (ha : a > 0) :
  (∀ x, quadratic_solution_set a b c x) = (x ≤ -3 ∨ x ≥ 4) →
  (∀ x, -12 * x^2 + x + 1 > 0) = (x < -1/4 ∨ x > 1/3) :=
by
  intros h1 h2
  sorry

end quadratic_inequality_l686_686974


namespace coefficient_of_term_in_binomial_expansion_l686_686058

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686058


namespace evaporation_period_l686_686834

theorem evaporation_period (w e : ℝ) (percentage_evaporated : ℝ) (d : ℝ) :
  w = 10 ∧ e = 0.03 ∧ percentage_evaporated = 0.06 → d = (percentage_evaporated * w) / e := 
by intro h; cases h with hw he_px; cases he_px with he hpx;
  simp [hw, he, hpx]; sorry

end evaporation_period_l686_686834


namespace number_of_downhill_divisible_by_4_l686_686885

def is_downhill (n : Nat) : Prop :=
  let digits := Nat.digits 10 n
  (digits.length = 3) ∧ ∀ i, i < digits.length - 1 → digits.get! i > digits.get! (i + 1)

def divisible_by_4 (n : Nat) : Prop :=
  n % 4 = 0

theorem number_of_downhill_divisible_by_4 : 
  ∃ (count : Nat), (∃ l, l.length = count ∧ ∀ x ∈ l, is_downhill x ∧ divisible_by_4 x) ∧ count = 8 :=
by
  sorry

end number_of_downhill_divisible_by_4_l686_686885


namespace coffee_is_32_3_percent_decaf_l686_686241

def percent_decaf_coffee_stock (total_weight initial_weight : ℕ) (initial_A_rate initial_B_rate initial_C_rate additional_weight additional_A_rate additional_D_rate : ℚ) 
(initial_A_decaf initial_B_decaf initial_C_decaf additional_D_decaf : ℚ) : ℚ :=
  let initial_A_weight := initial_A_rate * initial_weight
  let initial_B_weight := initial_B_rate * initial_weight
  let initial_C_weight := initial_C_rate * initial_weight
  let additional_A_weight := additional_A_rate * additional_weight
  let additional_D_weight := additional_D_rate * additional_weight

  let initial_A_decaf_weight := initial_A_decaf * initial_A_weight
  let initial_B_decaf_weight := initial_B_decaf * initial_B_weight
  let initial_C_decaf_weight := initial_C_decaf * initial_C_weight
  let additional_A_decaf_weight := initial_A_decaf * additional_A_weight
  let additional_D_decaf_weight := additional_D_decaf * additional_D_weight

  let total_decaf_weight := initial_A_decaf_weight + initial_B_decaf_weight + initial_C_decaf_weight + additional_A_decaf_weight + additional_D_decaf_weight

  (total_decaf_weight / total_weight) * 100

theorem coffee_is_32_3_percent_decaf : 
  percent_decaf_coffee_stock 1000 800 (40/100) (35/100) (25/100) 200 (50/100) (50/100) (20/100) (30/100) (45/100) (65/100) = 32.3 := 
  by 
    sorry

end coffee_is_32_3_percent_decaf_l686_686241


namespace distinct_units_digits_of_cube_l686_686485

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686485


namespace largest_of_decimals_l686_686190

theorem largest_of_decimals :
  let a := 0.993
  let b := 0.9899
  let c := 0.990
  let d := 0.989
  let e := 0.9909
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  sorry

end largest_of_decimals_l686_686190


namespace distinct_units_digits_of_perfect_cube_l686_686508

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686508


namespace bird_population_1999_l686_686404

theorem bird_population_1999 (
  k : ℝ,
  h1 : ∀ n, (λ P : ℕ → ℝ, (P (n + 3) - P (n + 1) = k * P (n + 2))),
  P1997 P1998 P2001 : ℝ,
  h2 : P1997 = 100,
  h3 : P1998 = 160,
  h4 : P2001 = 292,
  x : ℝ
) : x = 247 :=
by
  have h_eq1 : x - 100 = k * 160, from sorry,
  have h_eq2 : 132 = k * x, from sorry,
  have h_k : k = 132 / x, from sorry,
  have h_eq3 : x - 100 = (132 * 160) / x, from sorry,
  have h_quad : x^2 - 100 * x - 21120 = 0, from sorry,
  have h_solution : x = (100 + real.sqrt (10000 + 84480)) / 2, from sorry,
  have h_calc : x ≈ 247, from sorry,
  exact h_calc,

end bird_population_1999_l686_686404


namespace range_of_a_l686_686986

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 + x^2 + a * x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := (x + 1)^2 + a - 1

noncomputable def g (x : ℝ) : ℝ := 1 / real.exp x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 ∈ set.Icc (1 / 2 : ℝ) 2, f' a x1 ≤ g x2) →
  a ≤ (real.sqrt real.exp 1 / real.exp 1 - 5 / 4) :=
by
  sorry

end range_of_a_l686_686986


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686048

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686048


namespace packs_bought_l686_686636

theorem packs_bought (total_uncommon : ℕ) (cards_per_pack : ℕ) (fraction_uncommon : ℚ) 
  (total_packs : ℕ) (uncommon_per_pack : ℕ)
  (h1 : cards_per_pack = 20)
  (h2 : fraction_uncommon = 1/4)
  (h3 : uncommon_per_pack = fraction_uncommon * cards_per_pack)
  (h4 : total_uncommon = 50)
  (h5 : total_packs = total_uncommon / uncommon_per_pack)
  : total_packs = 10 :=
by 
  sorry

end packs_bought_l686_686636


namespace equal_chance_of_winning_l686_686364

-- Define the players and the game conditions
inductive Player
| A | B | C | D

open Player

def game (deck : list ℕ) (P : Player) : Prop :=
  ∀ (cond : list ℕ → Player),
    (length deck = 52) →
    (∀ n < 52, n ∈ deck) →
    (∀ n, n ∈ deck → n ≠ 0) →
    let num_ace := length (filter (λ x, x = 1) deck) in
    num_ace = 4 →
    ∃ p q r s, 1 ≤ p ∧ p < q ∧ q < r ∧ r < s ∧ s ≤ 52 ∧
      let cards_drawn := λ P,
        match P with
        | A => p
        | B => q - p
        | C => r - q
        | D => s - r
        end in
      Player.A ⟹ cards_drawn A = max (cards_drawn B) (max (cards_drawn C) (cards_drawn D)) ∪
      Player.B ⟹ cards_drawn B = max (cards_drawn A) (max (cards_drawn C) (cards_drawn D)) ∪
      Player.C ⟹ cards_drawn C = max (cards_drawn A) (max (cards_drawn B) (cards_drawn D)) ∪
      Player.D ⟹ cards_drawn D = max (cards_drawn A) (max (cards_drawn C) (cards_drawn B))

-- Set the theorem with the statement
theorem equal_chance_of_winning (deck : list ℕ) :
  ∀ (P : Player), game deck P →
    ∀ p q r s, 1 ≤ p ∧ p < q ∧ q < r ∧ r < s ∧ s ≤ 52 →
    let cards_drawn := λ P,
        match P with
        | A => p
        | B => q - p
        | C => r - q
        | D => s - r
        end in
    ∀ (P1 P2 : Player), (P1 ≠ P2) → 
    (cards_drawn P1 = cards_drawn P2) ∨ 
    ((cards_drawn P1 > cards_drawn P2) ∧ (cards_drawn P1 - cards_drawn P2 ≥ 0)) :=
by sorry

end equal_chance_of_winning_l686_686364


namespace range_of_f_l686_686350

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_of_f :
  (∀ y, ∃ x, x ≠ -5 ∧ f(x) = y) ↔ (y ≠ -27) :=
begin
  sorry
end

end range_of_f_l686_686350


namespace find_vertex_X_l686_686627

theorem find_vertex_X 
  (Y Z X : ℝ × ℝ × ℝ)
  (M : ℝ × ℝ × ℝ := (2,7,-3))
  (N : ℝ × ℝ × ℝ := (1,6,-4))
  (P : ℝ × ℝ × ℝ := (3,5,5)) :
  (X = (2, 9, 4)) :=
by
  let midpoint := λ (A B : ℝ × ℝ × ℝ), ((A.1 + B.1)/2, (A.2 + B.2)/2, (A.3 + B.3)/2)
  have M_eq : midpoint Y Z = (2, 7, -3), by sorry
  have N_eq : midpoint X Z = (1, 6, -4), by sorry
  have P_eq : midpoint X Y = (3, 5, 5), by sorry
  -- Skipping the proof steps, assuming X = (2, 9, 4)
  sorry

end find_vertex_X_l686_686627


namespace binomial_product_l686_686891

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l686_686891


namespace coefficient_of_term_in_binomial_expansion_l686_686054

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686054


namespace total_students_l686_686879

theorem total_students (T : ℝ) (h1 : 0.3 * T =  0.7 * T - 616) : T = 880 :=
by sorry

end total_students_l686_686879


namespace binomial_coefficient_x3y5_in_expansion_l686_686170

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686170


namespace distinct_units_digits_of_cubes_l686_686564

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686564


namespace binomial_17_8_value_l686_686389

theorem binomial_17_8_value :
  (5005 : ℕ) * \binom(1+1) = 24310 := sorry 

end binomial_17_8_value_l686_686389


namespace problem_solution_l686_686937

def diamond (c d : ℝ) : ℝ := Real.sqrt (c^2 + d^2)

theorem problem_solution : diamond (diamond 7 24) (diamond (-24) (-7)) = 25 * Real.sqrt 2 := 
by
  sorry -- Proof is skipped with "sorry"

end problem_solution_l686_686937


namespace distinct_units_digits_of_cubes_l686_686539

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686539


namespace total_questions_l686_686864

theorem total_questions (S C I : ℕ) (h1 : S = 73) (h2 : C = 91) (h3 : S = C - 2 * I) : C + I = 100 :=
sorry

end total_questions_l686_686864


namespace polygon_intersections_l686_686683

theorem polygon_intersections :
  ∃ (n : ℕ), n = 58 ∧ 
  (∃ (P4 P5 P7 P9 : Set (ℝ × ℝ)), 
    P4 ⊆ circle ∧ P5 ⊆ circle ∧ P7 ⊆ circle ∧ P9 ⊆ circle ∧ 
    sides_intersect_twice P4 P5 ∧ sides_intersect_twice P4 P7 ∧ 
    sides_intersect_twice P4 P9 ∧ sides_intersect_twice P5 P7 ∧ 
    sides_intersect_twice P5 P9 ∧ sides_intersect_twice P7 P9 ∧ 
    no_shared_vertices P4 P5 P7 P9 ∧ no_three_sides_intersect P4 P5 P7 P9 ∧ 
    count_intersections P4 P5 + count_intersections P4 P7 + 
    count_intersections P4 P9 + count_intersections P5 P7 + 
    count_intersections P5 P9 + count_intersections P7 P9 = 58) := sorry

end polygon_intersections_l686_686683


namespace least_num_to_divisible_l686_686806

theorem least_num_to_divisible (n : ℕ) : (1056 + n) % 27 = 0 → n = 24 :=
by
  sorry

end least_num_to_divisible_l686_686806


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686044

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686044


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686040

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686040


namespace distinct_units_digits_of_cubes_l686_686555

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686555


namespace vectors_coplanar_l686_686876

def vector3 := ℝ × ℝ × ℝ

def scalar_triple_product (a b c : vector3) : ℝ :=
  match a, b, c with
  | (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) =>
    a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product (-3, 3, 3) (-4, 7, 6) (3, 0, -1) = 0 :=
by
  sorry

end vectors_coplanar_l686_686876


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686140

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686140


namespace brick_length_proof_l686_686828

-- Defining relevant parameters and conditions
def width_of_brick : ℝ := 10 -- width in cm
def height_of_brick : ℝ := 7.5 -- height in cm
def wall_length : ℝ := 26 -- length in m
def wall_width : ℝ := 2 -- width in m
def wall_height : ℝ := 0.75 -- height in m
def num_bricks : ℝ := 26000 

-- Defining known volumes for conversion
def volume_of_wall_m3 : ℝ := wall_length * wall_width * wall_height
def volume_of_wall_cm3 : ℝ := volume_of_wall_m3 * 1000000 -- converting m³ to cm³

-- Volume of one brick given the unknown length L
def volume_of_one_brick (L : ℝ) : ℝ := L * width_of_brick * height_of_brick

-- Total volume of bricks is the volume of one brick times the number of bricks
def total_volume_of_bricks (L : ℝ) : ℝ := volume_of_one_brick L * num_bricks

-- The length of the brick is found by equating the total volume of bricks to the volume of the wall
theorem brick_length_proof : ∃ L : ℝ, total_volume_of_bricks L = volume_of_wall_cm3 ∧ L = 20 :=
by
  existsi 20
  sorry

end brick_length_proof_l686_686828


namespace distinct_cube_units_digits_l686_686462

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686462


namespace solve_inequality_l686_686929

theorem solve_inequality :
  ∀ x : ℝ, (-3 * x^2 + 2 * x + 8) > 0 ↔ x ∈ set.Ioo (-4/3 : ℝ) 2 :=
by
  sorry

end solve_inequality_l686_686929


namespace stock_decrease_l686_686862

-- Define the conditions
variables (x : ℝ)

-- Stock prices after the first and second year
def price_after_first_year := 1.30 * x
def price_after_second_year := 1.10 * price_after_first_year

-- Final proof statement: To find the required percentage decrease to return to the original price
def required_percentage_decrease : Prop :=
  (∃ y : ℝ, y = price_after_second_year x * (1 - 0.3007) ∧ y = x)

theorem stock_decrease : required_percentage_decrease x :=
  sorry

end stock_decrease_l686_686862


namespace num_distinct_units_digits_of_cubes_l686_686467

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686467


namespace num_distinct_units_digits_of_cubes_l686_686476

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686476


namespace probability_of_three_draws_l686_686827

noncomputable def box_chips : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def valid_first_two_draws (a b : ℕ) : Prop :=
  a + b <= 7

def prob_three_draws_to_exceed_seven : ℚ :=
  1 / 6

theorem probability_of_three_draws :
  (∃ (draws : List ℕ), (draws.length = 3) ∧ (draws.sum > 7)
    ∧ (∀ x ∈ draws, x ∈ box_chips)
    ∧ (∀ (a b : ℕ), (a ∈ box_chips ∧ b ∈ box_chips) → valid_first_two_draws a b))
  → prob_three_draws_to_exceed_seven = 1 / 6 :=
sorry

end probability_of_three_draws_l686_686827


namespace odd_numbers_after_10_operations_l686_686010

theorem odd_numbers_after_10_operations (a b : Nat) :
  (a, b) = (0, 1) → 
  ∃ n, n = 10 →
  let odd_count := (number of odd numbers on the board after n operations)
  odd_count = 683 := 
sorry

end odd_numbers_after_10_operations_l686_686010


namespace problem1_problem2_problem2_l686_686811

-- Problem 1: Prove the given expression equals \pi - 3/8
theorem problem1 :
  0.064^(-1 / 3) - (-7 / 8)^0 + ((2 - Real.pi)^2)^(1 / 2) + 16^(-0.75) = Real.pi - 3 / 8 :=
sorry

-- Problem 2: Prove the maximum value of the quadratic function in the given interval
theorem problem2 (a : ℝ) (h : 0 < a) :
  max (f 2a) = max (if  0 < 2 * a < 2 then 4 * a^2 + 1 else 8 * a - 3) :=
sorry

/-- Define f(x) as given -/
def f (x a : ℝ) := -x^2 + 4 * a * x + 1

/-- Prove the maximum value in the interval given the conditions -/

theorem problem2  (a : ℝ) (h : 0 < a) :
  max [f 0 a, f 1 a, f 2 a] = if a < 1 then 4 * a^2 + 1 else 8 * a - 3 :=
sorry

end problem1_problem2_problem2_l686_686811


namespace train_time_to_pass_l686_686800

/-
  Given:
  - length of the train l = 60 m
  - speed of the train v = 54 kmph

  Prove:
  - time t for the train to pass a telegraph post is 4 seconds
-/
theorem train_time_to_pass (length_of_train : ℝ) (speed_in_kmph : ℝ) (time_to_pass : ℝ) 
  (conversion_factor : 1000 / 3600 = (5:ℝ) / (18 : ℝ))
  (length_of_train = 60) 
  (speed_in_kmph = 54) 
  (time_to_pass = 4): 
  time_to_pass = (length_of_train) / (speed_in_kmph * (conversion_factor)) := 
by sorry

end train_time_to_pass_l686_686800


namespace range_of_inverse_y_y_maximum_value_l686_686993

variable (x : ℝ)
def y : ℝ := (x + 2) / (x^2 + x + 1)

theorem range_of_inverse_y (h : x > -2) :
    ∃ (m : ℝ), (∀ z : ℝ, z > 2 * Real.sqrt 3 - 3 → z = 1 / y x → True) :=
sorry

theorem y_maximum_value (h : x = Real.sqrt 3 - 2) :
    y x = (2 * Real.sqrt 3 + 3) / 3 :=
sorry

end range_of_inverse_y_y_maximum_value_l686_686993


namespace one_third_of_7_times_9_l686_686318

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_7_times_9_l686_686318


namespace sum_of_interior_angles_pentagon_l686_686007

theorem sum_of_interior_angles_pentagon :
  ∑ (A B C D E : Prop), A ∧ B ∧ C ∧ D ∧ E → 
  (∑ (interiorAngleSum), interiorAngleSum = (5 - 2) * 180) :=
begin
  -- Let pentagon be A B C D E
  -- Apply the formula to find the sum
  sorry -- we skip the proof as per instruction.
end

end sum_of_interior_angles_pentagon_l686_686007


namespace factor_congruence_l686_686649

theorem factor_congruence (n : ℕ) (hn : n ≠ 0) :
  ∀ p : ℕ, p ∣ (2 * n)^(2^n) + 1 → p ≡ 1 [MOD 2^(n+1)] :=
sorry

end factor_congruence_l686_686649


namespace ratio_of_volumes_l686_686216

theorem ratio_of_volumes (h1 : ∃ r : ℝ, 2 * Real.pi * r = 6) (h2 : ∃ r : ℝ, 2 * Real.pi * r = 9) :
  (let r1 := Classical.some h1, V1 := Real.pi * r1^2 * 9,
       r2 := Classical.some h2, V2 := Real.pi * r2^2 * 6
   in V2 / V1 = 3 / 2) :=
by
  sorry

end ratio_of_volumes_l686_686216


namespace monotonically_decreasing_f_minimum_value_g_difference_l686_686991

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * Real.log x

theorem monotonically_decreasing_f (a : ℝ) :
  (∀ x ∈ set.Icc 3 5, deriv (f x a) ≤ 0) → a ≥ 50 :=
sorry

noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := x^2 + 2 * Real.log x - 2 * (b - 1) * x

theorem minimum_value_g_difference (b : ℝ) (x1 x2 : ℝ) :
  b ≥ 7/2 → x1 < x2 → (x1^2 - (b - 1) * x1 + 1 = 0) →
  (x2^2 - (b - 1) * x2 + 1 = 0) →
  g x1 b b - g x2 b b ≥ (15/4 - 4 * Real.log 2) :=
sorry

end monotonically_decreasing_f_minimum_value_g_difference_l686_686991


namespace compute_binom_product_l686_686894

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l686_686894


namespace complex_number_quadrant_l686_686598

theorem complex_number_quadrant
  (A B : ℝ) (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (z : ℂ := (Real.cos B - Real.sin A) + (Real.sin B - Real.cos A) * Complex.i) :
  z.re < 0 ∧ 0 < z.im :=
sorry

end complex_number_quadrant_l686_686598


namespace coeff_x3y5_in_expansion_l686_686034

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686034


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686071

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686071


namespace common_chord_line_correct_length_common_chord_correct_line_l_correct_l686_686423

section CircleProblems

-- Define circles
def C1 := { p : ℝ × ℝ // p.1^2 + p.2^2 + 2 * p.1 + 8 * p.2 - 8 = 0 }
def C2 := { p : ℝ × ℝ // p.1^2 + p.2^2 - 4 * p.1 - 4 * p.2 - 2 = 0 }

-- Define the line of common chords
def common_chord_line (x y : ℝ) := x + 2 * y - 1 = 0

-- The length of the common chord
def length_common_chord : ℝ := 2 * Real.sqrt 5

-- Define the point
def P : ℝ × ℝ := (3, -1)

-- Define line l equations that pass through point P and satisfy the conditions
def line_l1 (x y : ℝ) := 7 * x + 24 * y + 3 = 0
def line_l2 (x : ℝ) := x = 3

-- Theorem statements
theorem common_chord_line_correct :
  ∀ (x y : ℝ), (C1.1 = x ∧ C1.2 = y) = (x + 2 * y - 1 = 0) :=
sorry

theorem length_common_chord_correct :
  ∃ (d : ℝ), d = length_common_chord :=
sorry

theorem line_l_correct :
  ∀ (x y : ℝ), (P.1 = x ∧ P.2 = y ∧ (C1.1 = x ∧ C1.2 = y)) → 
  (line_l2 x ∨ line_l1 x y) :=
sorry

end CircleProblems

end common_chord_line_correct_length_common_chord_correct_line_l_correct_l686_686423


namespace distinct_units_digits_of_integral_cubes_l686_686527

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686527


namespace infinite_midpoints_l686_686295

theorem infinite_midpoints (S : Set (ℝ × ℝ)) (H : ∀ p ∈ S, ∃ a b ∈ S, p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) : (S = Set.univ) :=
by
  sorry

end infinite_midpoints_l686_686295


namespace train_length_calculation_l686_686866

theorem train_length_calculation
  (speed_kmph : ℝ)
  (time_seconds : ℝ)
  (train_length : ℝ)
  (h1 : speed_kmph = 80)
  (h2 : time_seconds = 8.999280057595392)
  (h3 : train_length = (80 * 1000) / 3600 * 8.999280057595392) :
  train_length = 200 := by
  sorry

end train_length_calculation_l686_686866


namespace smallest_possible_a_plus_b_l686_686656

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ gcd (a + b) 330 = 1 ∧ (b ^ b ∣ a ^ a) ∧ ¬ (b ∣ a) ∧ (a + b = 507) := 
sorry

end smallest_possible_a_plus_b_l686_686656


namespace binomial_coefficient_x3y5_in_expansion_l686_686134

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686134


namespace general_form_of_sequence_l686_686747

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 3 else 3 / (15 * n - 14)

theorem general_form_of_sequence : 
  (a_n 1 = 3) ∧ (∀ n : ℕ, n > 0 → 1 / (a_n (n + 1) + 1) - 1 / (a_n (n + 1)) = 5) :=
begin
  split,
  { 
    -- proof that a_1 = 3 is as expected, essential part is the definition itself
    sorry
  },
  { 
    -- proof for the given condition for all n > 0
    sorry
  }
end

end general_form_of_sequence_l686_686747


namespace percentage_off_at_sale_l686_686276

theorem percentage_off_at_sale
  (sale_price original_price : ℝ)
  (h1 : sale_price = 140)
  (h2 : original_price = 350) :
  (original_price - sale_price) / original_price * 100 = 60 :=
by
  sorry

end percentage_off_at_sale_l686_686276


namespace true_propositions_l686_686266

theorem true_propositions (y x : Type) (samples : list (y × x)) (b a x̄ ȳ : ℝ) (var data : list ℝ) :
  (∀ p : y × x, p ∈ samples →
     let (yval, xval) := p in
     yval = b * xval + a) →
  (∀ p : y × x, p ∈ samples ∧ b * x̄ + a = ȳ) →
  (∀ (x1 x2 x3 xn : ℝ), 
     var = [x1, x2, x3, ..., xn] →
     var_double = [2 * x1, 2 * x2, 2 * x3, ..., 2 * xn] →
     variance var = 2 →
     variance var_double = 8) →
  (∀ (corr_coeff : ℝ), 
     abs corr_coeff → 1 →
     (∀ (linear_related : Prop), 
        linear_related = true ∨ false →
        stronger_correlation linear_related → abs corr_coeff = 1)) →
  true_propositions [1, 3] :=
by sorry

end true_propositions_l686_686266


namespace units_digit_of_perfect_cube_l686_686496

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686496


namespace prove_value_of_question_l686_686201

theorem prove_value_of_question :
  let a := 9548
  let b := 7314
  let c := 3362
  let value_of_question : ℕ := by 
    sorry -- Proof steps to show the computation.

  (a + b = value_of_question) ∧ (c + 13500 = value_of_question) :=
by {
  let a := 9548
  let b := 7314
  let c := 3362
  let sum_of_a_b := a + b
  let computed_question := sum_of_a_b - c
  sorry -- Proof steps to show sum_of_a_b and the final result.
}

end prove_value_of_question_l686_686201


namespace distinct_units_digits_of_cube_l686_686486

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686486


namespace largest_base_6_five_digits_l686_686181

-- Define the base-6 number 55555 in base 10
def base_6_to_base_10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 10000) % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

theorem largest_base_6_five_digits : base_6_to_base_10 55555 = 7775 := by
  sorry

end largest_base_6_five_digits_l686_686181


namespace range_of_m_l686_686973

def A (x : ℝ) : Prop := 1/2 < x ∧ x < 1

def B (x : ℝ) (m : ℝ) : Prop := x^2 + 2 * x + 1 - m ≤ 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, A x → B x m) → 4 ≤ m := by
  sorry

end range_of_m_l686_686973


namespace storage_cost_proof_l686_686233

noncomputable def total_storage_cost : ℝ :=
  let firmA_boxes := (1080000 / (15 * 12 * 10))
  let firmA_cost := firmA_boxes * 0.5
  let firmB_boxes := (1620000 / (18 * 16 * 14)).floor
  let firmB_cost := firmB_boxes * 0.7
  let firmC_boxes := (2160000 / (20 * 15 * 12))
  let firmC_cost := firmC_boxes * 0.8
  firmA_cost + firmB_cost + firmC_cost

theorem storage_cost_proof : total_storage_cost = 1061.40 := 
by 
  sorry

end storage_cost_proof_l686_686233


namespace one_third_of_seven_times_nine_l686_686332

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 :=
by
  calc
    (1 / 3) * (7 * 9) = (1 / 3) * 63 := by rw [mul_comm 7 9, mul_assoc 1 7 9]
                      ... = 21 := by norm_num

end one_third_of_seven_times_nine_l686_686332


namespace one_third_of_seven_times_nine_l686_686328

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 :=
by
  calc
    (1 / 3) * (7 * 9) = (1 / 3) * 63 := by rw [mul_comm 7 9, mul_assoc 1 7 9]
                      ... = 21 := by norm_num

end one_third_of_seven_times_nine_l686_686328


namespace value_of_a4_l686_686380

noncomputable def a : ℕ → ℚ 
| 1     := 1
| (n+1) := a n + 1 / ((n+1) * n)

theorem value_of_a4 : a 4 = 7 / 4 := 
by {
  sorry
}

end value_of_a4_l686_686380


namespace find_a20_l686_686382

def sequence (n : ℕ) : ℝ :=
  if n = 0 then 0
  else let a_n := sequence (n - 1)
  in (a_n - real.sqrt 3) / (real.sqrt 3 * a_n + 1)

theorem find_a20 : sequence 19 = -real.sqrt 3 :=
sorry

end find_a20_l686_686382


namespace annes_score_l686_686875

theorem annes_score (a b : ℕ) (h1 : a = b + 50) (h2 : (a + b) / 2 = 150) : a = 175 := 
by
  sorry

end annes_score_l686_686875


namespace distinct_units_digits_of_perfect_cubes_l686_686448

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686448


namespace cost_price_per_meter_l686_686256

theorem cost_price_per_meter
  (S : ℝ) (L : ℝ) (C : ℝ) (total_meters : ℝ) (total_price : ℝ)
  (h1 : total_meters = 400) (h2 : total_price = 18000)
  (h3 : L = 5) (h4 : S = total_price / total_meters) 
  (h5 : C = S + L) :
  C = 50 :=
by
  sorry

end cost_price_per_meter_l686_686256


namespace binomial_product_l686_686896

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l686_686896


namespace police_officers_needed_to_man_intersections_l686_686665

-- Define the conditions: number of streets and non-intersecting pairs
def number_of_streets : ℕ := 10
def non_intersecting_pairs : ℕ := 3

-- Define the theorem stating the number of police officers needed
theorem police_officers_needed_to_man_intersections :
  ∃ (n : ℕ), n = (number_of_streets.choose 2) - non_intersecting_pairs ∧ n = 42 :=
by
  use (number_of_streets.choose 2) - non_intersecting_pairs
  split
  · simp [number_of_streets, non_intersecting_pairs]
  · sorry

end police_officers_needed_to_man_intersections_l686_686665


namespace sector_area_correct_l686_686805

noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * Real.pi * r^2

-- Given conditions
def r : ℝ := 12
def θ : ℝ := 42

-- Define expected area of the sector
def expected_area : ℝ := 52.36

-- Lean 4 statement
theorem sector_area_correct :
  sector_area r θ = expected_area :=
by
  sorry

end sector_area_correct_l686_686805


namespace distinct_units_digits_perfect_cube_l686_686574

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686574


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686046

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686046


namespace coefficient_x3y5_in_expansion_l686_686113

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686113


namespace total_museum_tickets_cost_l686_686819

theorem total_museum_tickets_cost (num_students num_teachers cost_student_ticket cost_teacher_ticket : ℕ) :
  num_students = 12 →
  num_teachers = 4 →
  cost_student_ticket = 1 →
  cost_teacher_ticket = 3 →
  num_students * cost_student_ticket + num_teachers * cost_teacher_ticket = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_museum_tickets_cost_l686_686819


namespace teacher_student_arrangement_l686_686021

theorem teacher_student_arrangement :
  let teachers := 2 in
  let students := 5 in
  let total_positions := teachers + students in
  (∃ arrangement : vector (fin (total_positions + 1)) teachers × vector (fin (total_positions + 1)) students,
  (∀ t, t ∈ arrangement.fst → t ≠ 0 ∧ t ≠ total_positions - 1) ∧ 
  (∃ i, arrangement.fst = [i, i + 1]) ∧
  fintype.card arrangement = 960) :=
sorry

end teacher_student_arrangement_l686_686021


namespace log10_10_factorial_min_n_for_2_pow_l686_686022

theorem log10_10_factorial (log10_2 log10_3 log10_7 : ℝ) (h1 : log10_2 = 0.301) (h2 : log10_3 = 0.477) (h3 : log10_7 = 0.845) : log10 (∏ i in (finset.range 10).map nat.succ, (i : ℝ)) = 6.559 := 
by sorry

theorem min_n_for_2_pow (log10_10_fac : ℝ) (h : log10_10_fac = 6.559) : ∃ (n : ℕ), 10! < 2^n ∧ n = 22 :=
by sorry

end log10_10_factorial_min_n_for_2_pow_l686_686022


namespace coefficient_x3y5_in_expansion_l686_686110

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686110


namespace distinct_units_digits_of_cube_l686_686588

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686588


namespace vector_satisfy_condition_l686_686902

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  parametrize : ℝ → Point

def l : Line :=
  { parametrize := λ t => {x := 1 + 4 * t, y := 4 + 3 * t} }

def m : Line :=
  { parametrize := λ s => {x := -5 + 4 * s, y := 6 + 3 * s} }

def A (t : ℝ) : Point := l.parametrize t
def B (s : ℝ) : Point := m.parametrize s

-- The specific point for A and B are not used directly in the further proof statement.

def v : Point := { x := -6, y := 8 }

theorem vector_satisfy_condition :
  ∃ v1 v2 : ℝ, (v1 * -6) + (v2 * 8) = 2 ∧ (v1 = -6 ∧ v2 = 8) :=
sorry

end vector_satisfy_condition_l686_686902


namespace simplify_tan_expression_l686_686690

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end simplify_tan_expression_l686_686690


namespace simplify_tan_expression_l686_686688

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end simplify_tan_expression_l686_686688


namespace option_d_is_not_right_triangle_l686_686267

def is_not_right_triangle (a b c : ℕ) (angle1 angle2 angle3 : ℕ) : Prop :=
  a + b + c = 180 ∧ (a = 90 ∨ b = 90 ∨ c = 90) → false

theorem option_d_is_not_right_triangle :
  let a := 45
  let b := 60
  let c := 75
  is_not_right_triangle 45 60 75 := {
  sorry
}

end option_d_is_not_right_triangle_l686_686267


namespace noah_doctor_visits_l686_686669

theorem noah_doctor_visits (total_holidays : ℕ) (months_in_year : ℕ) :
  total_holidays = 36 ∧ months_in_year = 12 → total_holidays / months_in_year = 3 :=
by
  intros h
  cases h with h_total h_months
  rw [h_total, h_months]
  norm_num

end noah_doctor_visits_l686_686669


namespace solutions_of_equation_l686_686187

theorem solutions_of_equation :
  ∀ x : ℝ, x * (x - 3) = x - 3 ↔ x = 1 ∨ x = 3 :=
by sorry

end solutions_of_equation_l686_686187


namespace train_length_is_600_meters_l686_686867

noncomputable def calculate_train_length (t : ℝ) (u : ℝ) (v : ℝ) (same_direction : Bool) : ℝ :=
  if same_direction then
    let relative_speed := (v - u) * 1000 / 3600
    relative_speed * t
  else
    0 -- in case directions are different which doesn't apply here

theorem train_length_is_600_meters :
  calculate_train_length 35.99712023038157 3 63 true ≈ 600 :=
by
  sorry

end train_length_is_600_meters_l686_686867


namespace distinct_units_digits_of_perfect_cubes_l686_686441

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686441


namespace coefficient_x3_y5_in_binomial_expansion_l686_686105

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686105


namespace museum_ticket_cost_l686_686813

theorem museum_ticket_cost (students teachers : ℕ) (student_ticket_cost teacher_ticket_cost : ℕ) 
  (h_students : students = 12) (h_teachers : teachers = 4) 
  (h_student_ticket_cost : student_ticket_cost = 1) (h_teacher_ticket_cost : teacher_ticket_cost = 3) :
  students * student_ticket_cost + teachers * teacher_ticket_cost = 24 :=
by
  rw [h_students, h_teachers, h_student_ticket_cost, h_teacher_ticket_cost]
  exact (12 * 1 + 4 * 3) 
-- This produces 24

end museum_ticket_cost_l686_686813


namespace greatest_number_of_groups_l686_686269

theorem greatest_number_of_groups 
  (number_of_boys : ℕ)
  (number_of_girls : ℕ)
  (total_students : ℕ)
  (ages : list ℕ)
  (skill_levels : list ℕ)
  (number_of_groups : ℕ)
  (Hboys : number_of_boys = 20)
  (Hgirls : number_of_girls = 24)
  (Htotal : total_students = number_of_boys + number_of_girls)
  (Hages : ages = [12, 13, 14])
  (Hskills : skill_levels = ["beginner", "intermediate", "advanced"])
  (Heven : ∀ n, n ∈ ages → even n)
  (Hdiv : ∀ n, n ∈ skill_levels → ∃ k, number_of_groups = k * 3)
  (Hno_leftover : ∀ k, total_students % k = 0) :
  number_of_groups = 2 := sorry

end greatest_number_of_groups_l686_686269


namespace units_digit_of_perfect_cube_l686_686495

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686495


namespace add_hex_numbers_l686_686870

theorem add_hex_numbers : 
  let a := 0x7A4 in -- 7A4 in base 16
  let b := 0x1F9 in -- 1F9 in base 16
  a + b = 0x99D :=  -- Result in base 16
  by
    sorry

end add_hex_numbers_l686_686870


namespace coefficient_of_term_in_binomial_expansion_l686_686061

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686061


namespace n_minus_one_divides_n_squared_plus_n_sub_two_l686_686657

theorem n_minus_one_divides_n_squared_plus_n_sub_two (n : ℕ) : (n - 1) ∣ (n ^ 2 + n - 2) :=
sorry

end n_minus_one_divides_n_squared_plus_n_sub_two_l686_686657


namespace infinite_points_l686_686293

theorem infinite_points (S : set (ℝ × ℝ)) (h : ∀ P ∈ S, ∃ A B ∈ S, P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : set.infinite S :=
sorry

end infinite_points_l686_686293


namespace percentage_off_sale_l686_686278

theorem percentage_off_sale (original_price sale_price : ℝ) (h₁ : original_price = 350) (h₂ : sale_price = 140) :
  ((original_price - sale_price) / original_price) * 100 = 60 :=
by
  sorry

end percentage_off_sale_l686_686278


namespace towel_bleach_decrease_l686_686259

theorem towel_bleach_decrease (L B L' B' A A' : ℝ)
    (hB : B' = 0.6 * B)
    (hA : A' = 0.42 * A)
    (hA_def : A = L * B)
    (hA'_def : A' = L' * B') :
    L' = 0.7 * L :=
by
  sorry

end towel_bleach_decrease_l686_686259


namespace simplify_tan_expression_l686_686696

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end simplify_tan_expression_l686_686696


namespace propositions_injective_functions_l686_686952

theorem propositions_injective_functions :
  (∀ (f : ℝ → ℝ), 
    (∀ (x1 x2 : ℝ), f x1 = f x2 → x1 = x2) → 
    (f = (λ x, if x < 2 then x - 1 else log 2 x) → 
      (∀ x1 x2, f x1 = f x2 → x1 = x2)) ∧
    (∀ a, a > -2 → ¬(∀ x1 x2, (x1 > 0) → (x2 > 0) → f = (λ x, (x^2 + a * x + 1) / x) → 
      (f x1 = f x2 → x1 = x2))) ∧
    (∀ (f : ℝ → ℝ), (∀ x1 x2 : ℝ, f x1 = f x2 → x1 = x2) → 
      (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2)) ∧
    (∀ (f : ℝ → ℝ), (∀ x1 x2 : ℝ, f x1 = f x2 → x1 = x2) → 
      (∃ x₀, f differentiable at x₀ ∧ (f'(x₀) = 0) → False))) :=
by sorry

end propositions_injective_functions_l686_686952


namespace distinct_units_digits_of_cube_l686_686589

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686589


namespace focal_length_is_8_l686_686733

def hyperbola_focal_length (m : ℝ) : ℝ :=
  2 * real.sqrt(((m^2 + 12) + (4 - m^2)))

theorem focal_length_is_8 (m : ℝ) (h₁ : 4 - m^2 > 0) :
  hyperbola_focal_length m = 8 :=
by
  sorry

end focal_length_is_8_l686_686733


namespace distinct_units_digits_perfect_cube_l686_686570

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686570


namespace infinite_points_l686_686292

theorem infinite_points (S : set (ℝ × ℝ)) (h : ∀ P ∈ S, ∃ A B ∈ S, P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : set.infinite S :=
sorry

end infinite_points_l686_686292


namespace mowing_field_l686_686792

theorem mowing_field (x : ℝ) 
  (h1 : 1 / 84 + 1 / x = 1 / 21) : 
  x = 28 := 
sorry

end mowing_field_l686_686792


namespace distinct_units_digits_of_cubes_l686_686565

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686565


namespace num_valid_n_for_arithmetic_sequences_l686_686422

open Nat

theorem num_valid_n_for_arithmetic_sequences
  (a_n b_n : ℕ → ℕ)
  (A_n B_n : ℕ → ℕ)
  (h : ∀ n, A_n n / B_n n = (6 * n + 54) / (n + 5))
  : (card {n : ℕ | n > 0 ∧ a_n n / b_n n ∈ ℤ}) = 4 :=
sorry

end num_valid_n_for_arithmetic_sequences_l686_686422


namespace quadratic_properties_range_of_quadratic_shifted_parabola_l686_686398

-- Part (I): Proving properties of the given quadratic function y = 2x^2 + 4x - 3
theorem quadratic_properties :
  (∀ x: ℝ, 2*x^2 + 4*x - 3) ∧
  (direction_of_opening ↑ (λ x, 2*x^2 + 4*x - 3) = "upwards") ∧
  (axis_of_symmetry ↑ (λ x, 2*x^2 + 4*x - 3) = -1) ∧ 
  (vertex_coordinates ↑ (λ x, 2*x^2 + 4*x - 3) = (-1, -5)) :=
sorry

-- Part (II): Proving the range of the function for the interval -2 ≤ x ≤ 1
theorem range_of_quadratic : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, (y = 2*x^2 + 4*x - 3)) ∧
  (-5 ≤ y ∧ y ≤ 3) :=
sorry

-- Part (III): Proving the new function after the specified shifts
theorem shifted_parabola :
  (∀ x: ℝ, (let y := 2*(x-1)^2 - 4) = 
    ((shift_right 2 : ParabolaManipulation) ∘ (shift_up 1) ↑ (λ x, 2*x^2 + 4*x - 3))) :=
sorry

end quadratic_properties_range_of_quadratic_shifted_parabola_l686_686398


namespace paint_remaining_after_second_day_l686_686848

def initial_paint_amount : ℝ := 2
def fraction_used_each_day : ℝ := 1 / 2
def paint_remaining_after_day (n : ℕ) : ℝ :=
  initial_paint_amount * (fraction_used_each_day ^ n)

theorem paint_remaining_after_second_day : 
  paint_remaining_after_day 2 = initial_paint_amount / 2 :=
by
  -- Omitting the proof as per instructions
  sorry

end paint_remaining_after_second_day_l686_686848


namespace distinct_units_digits_of_cube_l686_686586

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686586


namespace coefficient_of_term_in_binomial_expansion_l686_686055

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686055


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686051

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686051


namespace parking_lot_problem_l686_686851

theorem parking_lot_problem (n m k : ℕ) (h1 : n = 12) (h2 : m = 8) (h3 : k = 4) :
  (Nat.perm m m) * (Nat.choose (n - m + 1) 1) = Nat.perm 8 8 * Nat.choose 9 1 :=
by
  rw [h1, h2, h3]
  sorry

end parking_lot_problem_l686_686851


namespace sums_not_all_same_l686_686767

theorem sums_not_all_same (table : Fin 4 → Fin 18 → ℕ) 
  (h_table : ∀ i j, 1 ≤ table i j ∧ table i j ≤ 72) : 
  ¬ ∃ (S : ℕ), (∀ j : Fin 18, (Nat.digits 10 (∏ i : Fin 4, table i j)).sum = S) := 
sorry

end sums_not_all_same_l686_686767


namespace one_third_of_seven_times_nine_l686_686329

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 :=
by
  calc
    (1 / 3) * (7 * 9) = (1 / 3) * 63 := by rw [mul_comm 7 9, mul_assoc 1 7 9]
                      ... = 21 := by norm_num

end one_third_of_seven_times_nine_l686_686329


namespace distinct_units_digits_of_cube_l686_686491

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686491


namespace exists_quadrilateral_with_properties_l686_686759

-- Define the geometric setup with a circle, and points inside it
variables (k : Circle) (K L : Point)

-- Define the required existence of quadrilateral with the properties
theorem exists_quadrilateral_with_properties :
  ∃ (A B C D : Point), 
  isInscribedInCircle k A B C D ∧ 
  intersect (line A C) (line B D) = K ∧
  let F1 := midpoint A B in
  let F2 := midpoint B C in
  let F3 := midpoint C D in
  let F4 := midpoint D A in
  intersect (line F1 F3) (line F2 F4) = L := 
sorry

end exists_quadrilateral_with_properties_l686_686759


namespace coefficient_x3_y5_in_binomial_expansion_l686_686107

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686107


namespace find_period_l686_686661

noncomputable def f (A ω φ x : ℝ) := A * sin (ω * x + φ)

theorem find_period (A ω φ T : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x y, x ∈ Set.Icc (π / 6) (π / 2) → y ∈ Set.Icc (π / 6) (π / 2) → x ≤ y → f A ω φ x ≤ f A ω φ y) →
  f A ω φ (π / 2) = f A ω φ (2 * π / 3) →
  f A ω φ (π / 2) = -f A ω φ (π / 6) →
  T = π :=
by
  sorry

end find_period_l686_686661


namespace coeff_x3y5_in_expansion_l686_686033

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686033


namespace distinct_units_digits_of_perfect_cube_l686_686512

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686512


namespace find_same_goldfish_number_l686_686282

noncomputable def B (n : ℕ) : ℕ := 3 * 4^n
noncomputable def G (n : ℕ) : ℕ := 243 * 3^n

theorem find_same_goldfish_number : ∃ n, B n = G n :=
by sorry

end find_same_goldfish_number_l686_686282


namespace distinct_units_digits_of_cubes_l686_686437

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686437


namespace distinct_units_digits_of_cubes_l686_686554

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686554


namespace inequality_solution_interval_l686_686603

theorem inequality_solution_interval {a : ℝ} :
  (∃ x y : ℝ, x < y ∧ y - x ≤ 5 ∧ ∀ z, x < z ∧ z < y → z^2 - a*z - 6*a < 0) ↔
  (-25 ≤ a ∧ a < -24) ∨ (0 < a ∧ a ≤ 1) :=
begin
  sorry
end

end inequality_solution_interval_l686_686603


namespace area_relationship_l686_686615

variable {P : Type} [metric_space P]

/-- Represents a triangle structure with given points A, B, and C -/
structure triangle (P : Type) :=
(A B C : P)
(is_acute : ∀ (α β γ: ℝ), α + β + γ = π → α < π/2 ∧ β < π/2 ∧ γ < π/2)

noncomputable def area (t : triangle P) : ℝ := sorry

/-- Bisects an internal angle of a triangle and meets the circumcircle again at D -/
def bisector_circle_point (A B C : P) := sorry

/-- Returns points A_1, B_1, C_1 that meet the circumcircle at bisectors of respective angles -/
def intersection_points (t : triangle P) : (P × P × P) := sorry

/-- Returns points A_0, B_0, C_0 that meet the external bisectors -/
def external_bisectors_points (t : triangle P) (A1 B1 C1 : P) : (P × P × P) := sorry

/-- Calculates area of the hexagon AC1BA1CB1 -/
noncomputable def hexagon_area (A C1 B A1 C B1 : P) : ℝ := sorry

theorem area_relationship (t : triangle P) : 
    let (A1, B1, C1) := intersection_points t
    let (A0, B0, C0) := external_bisectors_points t A1 B1 C1
    in (area ⟨A0, B0, C0, t.is_acute⟩ = 2 * hexagon_area t.A C1 t.B A1 t.C B1) ∧ 
       (area ⟨A0, B0, C0, t.is_acute⟩ ≥ 4 * area t) := sorry

end area_relationship_l686_686615


namespace percent_palindromes_containing_five_l686_686849

-- Define conditions
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def contains_five (n : ℕ) : Prop := n.toString.contains '5'

-- Define the problem
theorem percent_palindromes_containing_five :
  let total_palindromes := (Finset.filter is_palindrome (Finset.range 1000)).attach.filter is_three_digit
  let palindromes_with_five := total_palindromes.filter contains_five
  ((palindromes_with_five.card * 100 / total_palindromes.card : ℝ) ≈ 20.56) := 
by
  sorry

end percent_palindromes_containing_five_l686_686849


namespace binomial_coefficient_x3y5_in_expansion_l686_686171

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686171


namespace calculate_large_exponent_l686_686881

theorem calculate_large_exponent : (1307 * 1307)^3 = 4984209203082045649 :=
by {
   sorry
}

end calculate_large_exponent_l686_686881


namespace correct_statement_l686_686785

-- Definitions based on the conditions
def three_points_determine_plane (p1 p2 p3 : Point) : Prop :=
  ¬ collinear p1 p2 p3

def line_and_point_determine_plane (l : Line) (p : Point) : Prop :=
  ¬ (p ∈ l)

def trapezoid_determine_plane (t : Trapezoid) : Prop :=
  ∃ p1 p2 p3 p4 : Point, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1 ∧ 
    p1 ≠ p3 ∧ p2 ≠ p4 ∧
    ¬ collinear p1 p2 p3 ∧
    ¬ collinear p1 p3 p4

def circle_center_and_points_determine_plane (c : Circle) (p1 p2 : Point) : Prop :=
  ¬ collinear c.center p1 p2

-- The proof problem
theorem correct_statement : 
  (three_points_determine_plane p1 p2 p3 ∨
   line_and_point_determine_plane l p ∨
   trapezoid_determine_plane t ∨
   circle_center_and_points_determine_plane c p1 p2) →
  trapezoid_determine_plane t :=
sorry

end correct_statement_l686_686785


namespace set_operation_result_l686_686664

open Set

variable {α : Type*} (U : Set α) (Z N : Set α)

theorem set_operation_result :
  U = univ → (Z ∪ compl N) = univ :=
by
  assume hU : U = univ
  sorry

end set_operation_result_l686_686664


namespace volume_ratio_l686_686221

def rect_height_vert : ℝ := 9
def rect_circumference_vert : ℝ := 6
def radius_vert : ℝ := rect_circumference_vert / (2 * Real.pi)
def volume_vert : ℝ := Real.pi * radius_vert^2 * rect_height_vert
-- volume_vert is calculated as πr^2h where r = 3/π

def rect_height_horiz : ℝ := 6
def rect_circumference_horiz : ℝ := 9
def radius_horiz : ℝ := rect_circumference_horiz / (2 * Real.pi)
def volume_horiz : ℝ := Real.pi * radius_horiz^2 * rect_height_horiz
-- volume_horiz is calculated as πr^2h where r = 9/(2π)

theorem volume_ratio : (max volume_vert volume_horiz) / (min volume_vert volume_horiz) = 3 / 4 := 
by sorry

end volume_ratio_l686_686221


namespace distinct_units_digits_of_cube_l686_686479

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686479


namespace correct_option_D_l686_686778

theorem correct_option_D (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2 * x :=
by sorry

end correct_option_D_l686_686778


namespace percentage_saving_l686_686249

theorem percentage_saving (S : ℝ) (E : ℝ) (P : ℝ) 
  (h1 : S = 5500) 
  (h2 : S - 1.20 * E = 220) 
  (h3 : E = S - (P / 100) * S) :
  P = 20 := 
by 
  intro S E P h1 h2 h3
  sorry

end percentage_saving_l686_686249


namespace general_formula_sum_inequality_l686_686956

section Problem

-- Defining the sequence a_n
def a : ℕ → ℚ
| 0       := 2/5
| (n + 1) := 2 * a n / (3 - a n)

-- (1) Prove the general formula for {1/a_n}
theorem general_formula (n : ℕ) : 1 / a n = (3/2)^n + 1 :=
sorry

-- (2) Let S_n be the sum of the first n terms of a_n. Prove the inequality for S_n.
def S_n (n : ℕ) : ℚ :=
(nat.cases_on n 0 (λ n, (finset.range (n + 1)).sum a))

theorem sum_inequality (n : ℕ) : 
  6 / 5 * (1 - (2 / 3) ^ n) ≤ S_n n ∧ S_n n < 21 / 13 :=
sorry

end Problem

end general_formula_sum_inequality_l686_686956


namespace one_third_of_seven_times_nine_l686_686324

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l686_686324


namespace midpoint_ordinate_l686_686402

-- Definitions based on given conditions
def P (a : ℝ) (h : 0 < a ∧ a < π / 2) : ℝ × ℝ := (a, sin a)
def Q (a : ℝ) (h : 0 < a ∧ a < π / 2) : ℝ × ℝ := (a, cos a)
def distance_PQ (a : ℝ) (h : 0 < a ∧ a < π / 2) : ℝ := |sin a - cos a|

-- The statement to prove
theorem midpoint_ordinate (a : ℝ) (h : 0 < a ∧ a < π / 2) (h_dist : distance_PQ a h = 1 / 4) : 
  (sin a + cos a) / 2 = sqrt 31 / 8 :=
by
  sorry

end midpoint_ordinate_l686_686402


namespace cylinder_volume_ratio_l686_686225

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l686_686225


namespace sum_remainders_l686_686365

theorem sum_remainders :
  ∀ (a b c d : ℕ),
  a % 53 = 31 →
  b % 53 = 44 →
  c % 53 = 6 →
  d % 53 = 2 →
  (a + b + c + d) % 53 = 30 :=
by
  intros a b c d ha hb hc hd
  sorry

end sum_remainders_l686_686365


namespace ellipse_equation_and_area_ratio_constant_l686_686406

theorem ellipse_equation_and_area_ratio_constant 
  (A B T : ℝ × ℝ)
  (e : ℝ)
  (C : ℝ × ℝ → Prop)
  (l BM AN : ℝ × ℝ → Prop)
  (O P Q : ℝ × ℝ)
  (hA : A = (-2, 0))
  (hB : B = (2, 0))
  (hT : T = (4, 0))
  (heccentricity : e = (sqrt 3) / 2)
  (hC : ∀ P : ℝ × ℝ, C P ↔ (P.1^2) / 4 + (P.2^2) = 1)
  (hl : ∀ P : ℝ × ℝ, l P ↔ P.1 = 4)
  (hBM : ∀ P : ℝ × ℝ, BM P ↔ P.1 = 1)
  (hAN : ∀ Q : ℝ × ℝ, AN Q ↔ Q.1 = 0) :
  (∃ PQ : ℝ × ℝ × ℝ, ∃ r : ℚ, ∀ O AQ TP : Set ℝ, 
    (S ∆ O AQ) / (S ∆ O TP) = r)
    :=
  sorry

end ellipse_equation_and_area_ratio_constant_l686_686406


namespace distinct_units_digits_of_cubes_l686_686567

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686567


namespace binomial_coefficient_x3y5_in_expansion_l686_686135

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686135


namespace volume_ratio_l686_686219

def rect_height_vert : ℝ := 9
def rect_circumference_vert : ℝ := 6
def radius_vert : ℝ := rect_circumference_vert / (2 * Real.pi)
def volume_vert : ℝ := Real.pi * radius_vert^2 * rect_height_vert
-- volume_vert is calculated as πr^2h where r = 3/π

def rect_height_horiz : ℝ := 6
def rect_circumference_horiz : ℝ := 9
def radius_horiz : ℝ := rect_circumference_horiz / (2 * Real.pi)
def volume_horiz : ℝ := Real.pi * radius_horiz^2 * rect_height_horiz
-- volume_horiz is calculated as πr^2h where r = 9/(2π)

theorem volume_ratio : (max volume_vert volume_horiz) / (min volume_vert volume_horiz) = 3 / 4 := 
by sorry

end volume_ratio_l686_686219


namespace stella_total_pages_l686_686719

-- Definitions based on conditions
def rows_per_page := 5
def stamps_per_row := 30
def pages_with_fixed_stamps := 10
def stamps_per_fixed_page := rows_per_page * stamps_per_row
def stamps_per_additional_page := 50
def total_stamps := 3500

-- The statement to prove
theorem stella_total_pages : 
  let first_10_pages_stamps := pages_with_fixed_stamps * stamps_per_fixed_page in
  let remaining_stamps := total_stamps - first_10_pages_stamps in
  let additional_pages := remaining_stamps / stamps_per_additional_page in
  let total_pages := pages_with_fixed_stamps + additional_pages in
  total_pages = 50 := by
  sorry

end stella_total_pages_l686_686719


namespace sum_max_min_f_l686_686751

theorem sum_max_min_f (x : ℝ) : 
  let f := λ x, 2*(sin x)^2 - 6*(sin x) + 2 in
  let max_val := 10 in
  let min_val := -2 in
  (∀ x, x ∈ ℝ → (sin x) ∈ [-1, 1] → f x ≤ max_val ∧ f x ≥ min_val) →
  (max_val + min_val = 8) :=
by
  let f := λ x, 2*(sin x)^2 - 6*(sin x) + 2
  let max_val := 10
  let min_val := -2
  have h1 : -1 ≤ sin x ∧ sin x ≤ 1, from sorry
  have h2 : f x ≤ max_val, from sorry
  have h3 : f x ≥ min_val, from sorry
  sorry

end sum_max_min_f_l686_686751


namespace additional_rods_needed_l686_686239

theorem additional_rods_needed (P_initial : ℝ) (sheep_initial : ℝ) (P_new : ℝ) : 
  P_initial = 50 → sheep_initial = 100 →
  P_new = 4 * real.sqrt (2 * (P_initial / 4)^2) →
  P_new - P_initial = 21 :=
by
  intros hP_initial hSheep_initial hP_new
  rw [hP_initial] at hP_new
  rw [hSheep_initial]
  have : (50 / 4 : ℝ) = 12.5 := by norm_num
  rw [this] at hP_new
  have : real.sqrt (2 * (12.5)^2) = real.sqrt (312.5) := rfl
  rw [this] at hP_new
  norm_num at hP_new
  have : 4 * real.sqrt (312.5) ≈ 70.72 := by norm_num
  rw [this] at hP_new
  norm_num
  exact hP_new

end additional_rods_needed_l686_686239


namespace coefficient_x3y5_in_expansion_l686_686120

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686120


namespace range_of_f_l686_686349

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_of_f :
  (∀ y, ∃ x, x ≠ -5 ∧ f(x) = y) ↔ (y ≠ -27) :=
begin
  sorry
end

end range_of_f_l686_686349


namespace distance_between_x_intercepts_l686_686844

-- Definitions for the conditions
def line_eq (m : ℝ) (x1 y1 : ℝ) : ℝ → ℝ := λ x, m * (x - x1) + y1

def x_intercept (m : ℝ) (x1 y1 : ℝ) : ℝ := classical.some (exists_x_intercept m x1 y1)

lemma exists_x_intercept (m x1 y1 : ℝ) : ∃ x : ℝ, line_eq m x1 y1 x = 0 :=
begin
  use (y1 / m + x1),
  simp [line_eq]
end

-- Main theorem
theorem distance_between_x_intercepts : 
  let l1 := line_eq 4 8 20,
      l2 := line_eq (-3) 8 20 in
  dist (x_intercept 4 8 20) (x_intercept (-3) 8 20) = 35 / 3 :=
by {
  -- Define equations of the lines
  let line1 := (λ x, 4 * (x - 8) + 20),
  let line2 := (λ x, -3 * (x - 8) + 20),

  -- Compute x-intercepts
  let x_int1 := classical.some (exists_x_intercept 4 8 20),
  let x_int2 := classical.some (exists_x_intercept (-3) 8 20),

  -- The derivation of x-intercepts and the distance calculation should be skipped for now
  sorry
}

end distance_between_x_intercepts_l686_686844


namespace largest_base_6_five_digits_l686_686180

-- Define the base-6 number 55555 in base 10
def base_6_to_base_10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 10000) % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

theorem largest_base_6_five_digits : base_6_to_base_10 55555 = 7775 := by
  sorry

end largest_base_6_five_digits_l686_686180


namespace average_grade_of_females_is_92_l686_686726

theorem average_grade_of_females_is_92 (F : ℝ) : 
  (∀ (overall_avg male_avg : ℝ) (num_male num_female : ℕ), 
    overall_avg = 90 ∧ male_avg = 82 ∧ num_male = 8 ∧ num_female = 32 → 
    overall_avg = (num_male * male_avg + num_female * F) / (num_male + num_female) → F = 92) :=
sorry

end average_grade_of_females_is_92_l686_686726


namespace midpoint_ordinate_l686_686401

-- Definitions based on given conditions
def P (a : ℝ) (h : 0 < a ∧ a < π / 2) : ℝ × ℝ := (a, sin a)
def Q (a : ℝ) (h : 0 < a ∧ a < π / 2) : ℝ × ℝ := (a, cos a)
def distance_PQ (a : ℝ) (h : 0 < a ∧ a < π / 2) : ℝ := |sin a - cos a|

-- The statement to prove
theorem midpoint_ordinate (a : ℝ) (h : 0 < a ∧ a < π / 2) (h_dist : distance_PQ a h = 1 / 4) : 
  (sin a + cos a) / 2 = sqrt 31 / 8 :=
by
  sorry

end midpoint_ordinate_l686_686401


namespace museum_ticket_cost_l686_686815

theorem museum_ticket_cost 
  (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_cost : ℕ) (teacher_ticket_cost : ℕ)
  (h_students : num_students = 12) (h_teachers : num_teachers = 4)
  (h_student_cost : student_ticket_cost = 1) (h_teacher_cost : teacher_ticket_cost = 3) :
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end museum_ticket_cost_l686_686815


namespace coefficient_x3y5_in_expansion_l686_686116

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686116


namespace coefficient_x3_y5_in_binomial_expansion_l686_686106

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686106


namespace polynomial_degree_and_coefficient_l686_686743

noncomputable def P : Polynomial ℤ :=
  (Polynomial.X - 1) *
  (Polynomial.X ^ 2 - 1) *
  (Polynomial.X ^ 3 - 1) *
  ... *
  (Polynomial.X ^ 2007 - 1)
  -- assuming suitable notation for the product up to x^2007-1

noncomputable def f : Polynomial ℤ :=
  Polynomial.of_terms
    (Polynomial.terms_of_degree_le P 2007)  -- hypothetical function to denote the expansion and removal

theorem polynomial_degree_and_coefficient :
  Polynomial.degree f = 1953 ∧ Polynomial.leading_coeff f = 1 :=
begin
  -- here would be the proof which we're skipping with sorry
  sorry,
end

end polynomial_degree_and_coefficient_l686_686743


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686088

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686088


namespace probability_divisible_by_45_l686_686606

theorem probability_divisible_by_45 (d : Fin 6 → ℕ)
  (h_digits : ∀ i, d i ∈ {1, 3, 4, 5, 9})
  (h_distinct : ∀ i j, i ≠ j → d i ≠ d j) :
  ∃ P : ℚ, P = 0 :=
by
  let sum_digits := ∑ i, d i 
  have h_sum : sum_digits = 25 := sorry  -- This follows from the given digits
  have not_div_9 : ¬ (sum_digits % 9 = 0) := sorry -- 25 is not divisible by 9
  have not_possible : ∀ perm, ¬ (perm % 45 = 0) := sorry -- No permutation is divisible by 45
  existsi (0 : ℚ)
  apply eq.refl

end probability_divisible_by_45_l686_686606


namespace original_cost_order_l686_686886

theorem original_cost_order (x : ℝ) 
  (coupon_discount : x - 2.5) 
  (senior_discount : (coupon_discount - 0.20 * coupon_discount)) 
  (final_cost : senior_discount = 4) : 
  x = 7.5 :=
sorry

end original_cost_order_l686_686886


namespace range_of_a_l686_686370

noncomputable def f (a x : ℝ) : ℝ := a * x - x * real.log x - a

theorem range_of_a (a : ℝ) (h : ∀ x, 2 ≤ x → f a x ≤ 0) : a ≤ 2 * real.log 2 := by
  sorry

end range_of_a_l686_686370


namespace square_of_number_l686_686922

theorem square_of_number (x : ℝ) (h : 2 * x = x / 5 + 9) : x^2 = 25 := 
sorry

end square_of_number_l686_686922


namespace one_third_of_seven_times_nine_l686_686331

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 :=
by
  calc
    (1 / 3) * (7 * 9) = (1 / 3) * 63 := by rw [mul_comm 7 9, mul_assoc 1 7 9]
                      ... = 21 := by norm_num

end one_third_of_seven_times_nine_l686_686331


namespace factorize_polynomial_l686_686917

theorem factorize_polynomial (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
by 
  sorry

end factorize_polynomial_l686_686917


namespace convert_speed_l686_686299

theorem convert_speed (v_m_s : ℚ) (conversion_factor : ℚ) :
  v_m_s = 12 / 43 → conversion_factor = 3.6 → v_m_s * conversion_factor = 1.0046511624 := by
  intros h1 h2
  have h3 : v_m_s = 12 / 43 := h1
  have h4 : conversion_factor = 3.6 := h2
  rw [h3, h4]
  norm_num
  sorry

end convert_speed_l686_686299


namespace correct_statements_l686_686782

theorem correct_statements :
  (∀ (α : ℝ) (k : ℤ), α = π / 3 + 2 * k * π → α.is_coterminal_with (π / 3)) ∧
  (π = 180 * (π / 180)) ∧
  (∀ (r θ : ℝ), r = 6 → θ = π / 3 → r * θ = 2 * π) ∧
  (∀ (x : ℝ), (π / 2 < x ∧ x < π) → (90 * (π / 180) < x ∧ x < 180 * (π / 180))) :=
by
  sorry

end correct_statements_l686_686782


namespace find_k_l686_686297

noncomputable def equation (x : ℝ) : ℝ := 
  (Real.sqrt (3 * x^2 - 8 * x + 1) + Real.sqrt (9 * x^2 - 24 * x - 8))

theorem find_k :
  ∃ k : ℝ, (∀ x : ℝ, equation x = 3) ∧ 
  (let roots := {x | equation x = 3} in 
   ∃ largest smallest : ℝ, largest ∈ roots ∧ smallest ∈ roots ∧ largest = -k * smallest) ∧ k = 9 :=
sorry

end find_k_l686_686297


namespace number_of_students_taking_art_l686_686235

noncomputable def total_students : ℕ := 500
noncomputable def students_taking_music : ℕ := 50
noncomputable def students_taking_both : ℕ := 10
noncomputable def students_taking_neither : ℕ := 440

theorem number_of_students_taking_art (A : ℕ) (h1: total_students = 500) (h2: students_taking_music = 50) 
  (h3: students_taking_both = 10) (h4: students_taking_neither = 440) : A = 20 :=
by 
  have h5 : total_students = students_taking_music - students_taking_both + A - students_taking_both + 
    students_taking_both + students_taking_neither := sorry
  have h6 : 500 = 40 + A - 10 + 10 + 440 := sorry
  have h7 : 500 = A + 480 := sorry
  have h8 : A = 20 := by linarith 
  exact h8

end number_of_students_taking_art_l686_686235


namespace find_prime_triples_l686_686921

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n

theorem find_prime_triples : 
  ∀ a b c : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ 
    a < b ∧ b < c ∧ c < 100 ∧ 
    (b + 1) * (b + 1) = (a + 1) * (c + 1) 
  ↔ (a = 2 ∧ b = 5 ∧ c = 11) ∨ 
     (a = 2 ∧ b = 11 ∧ c = 47) ∨ 
     (a = 5 ∧ b = 11 ∧ c = 23) ∨ 
     (a = 5 ∧ b = 17 ∧ c = 53) ∨ 
     (a = 7 ∧ b = 23 ∧ c = 71) ∨ 
     (a = 11 ∧ b = 23 ∧ c = 47) := 
by 
  sorry

end find_prime_triples_l686_686921


namespace num_positive_integers_n_l686_686936

theorem num_positive_integers_n (h : ∀ n : ℕ, 1638 % (n^2 - 3) = 0) : (finset.card {n | n^2 - 3 ∣ 1638}) = 4 :=
sorry

end num_positive_integers_n_l686_686936


namespace distinct_units_digits_of_cubes_l686_686428

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686428


namespace distinct_units_digits_of_cube_l686_686481

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686481


namespace binomial_product_l686_686890

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l686_686890


namespace binomial_coefficient_x3y5_in_expansion_l686_686169

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686169


namespace coefficient_x3y5_in_expansion_l686_686118

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686118


namespace raghu_investment_l686_686197

noncomputable def investment_problem (R T V : ℝ) : Prop :=
  V = 1.1 * T ∧
  T = 0.9 * R ∧
  R + T + V = 6358 ∧
  R = 2200

theorem raghu_investment
  (R T V : ℝ)
  (h1 : V = 1.1 * T)
  (h2 : T = 0.9 * R)
  (h3 : R + T + V = 6358) :
  R = 2200 :=
sorry

end raghu_investment_l686_686197


namespace card_average_2023_l686_686830

theorem card_average_2023 (n : ℕ) (h_pos : 0 < n) (h_avg : (2 * n + 1) / 3 = 2023) : n = 3034 := by
  sorry

end card_average_2023_l686_686830


namespace coefficient_of_term_in_binomial_expansion_l686_686063

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686063


namespace tangent_slope_at_C_neg_2_neg_8_l686_686926

theorem tangent_slope_at_C_neg_2_neg_8 :
    ∀ (x : ℝ), (y = x^3) → (C = (-2, -8)) → (D : ℝ) (D = 3 * (-2)^2) → D = 12 := 
by
  sorry

end tangent_slope_at_C_neg_2_neg_8_l686_686926


namespace triangle_side_lengths_l686_686744

noncomputable def radius_inscribed_circle := 4/3
def sum_of_heights := 13

theorem triangle_side_lengths :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (h_a h_b h_c : ℕ), h_a ≠ h_b ∧ h_b ≠ h_c ∧ h_a ≠ h_c ∧
  h_a + h_b + h_c = sum_of_heights ∧
  r * (a + b + c) = 8 ∧ -- (since Δ = r * s, where s = (a + b + c)/2)
  1 / 2 * a * h_a = 1 / 2 * b * h_b ∧
  1 / 2 * b * h_b = 1 / 2 * c * h_c ∧
  a = 6 ∧ b = 4 ∧ c = 3 :=
sorry

end triangle_side_lengths_l686_686744


namespace distinct_units_digits_of_integral_cubes_l686_686523

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686523


namespace units_digit_of_perfect_cube_l686_686493

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686493


namespace problem_1_problem_2_problem_3_l686_686371

-- Declare variables m and n
variables (m n : ℝ)

-- Given conditions
def condition_1 := m + n = 2
def condition_2 := m * n = -2

-- Problem 1
theorem problem_1 (h1 : condition_1) (h2 : condition_2) : 2^m * 2^n - (2^m)^n = 3 + 3/4 := 
by sorry

-- Problem 2
theorem problem_2 (h1 : condition_1) (h2 : condition_2) : (m - 4) * (n - 4) = 6 := 
by sorry

-- Problem 3
theorem problem_3 (h1 : condition_1) (h2 : condition_2) : (m - n) ^ 2 = 12 := 
by sorry

end problem_1_problem_2_problem_3_l686_686371


namespace dot_product_a_b_l686_686426

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (1, -2)

theorem dot_product_a_b :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = -3 :=
by
  sorry

end dot_product_a_b_l686_686426


namespace coefficient_of_term_in_binomial_expansion_l686_686067

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686067


namespace distinct_units_digits_of_perfect_cube_l686_686517

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686517


namespace units_digit_of_perfect_cube_l686_686494

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686494


namespace ratio_of_volumes_l686_686214

theorem ratio_of_volumes (h1 : ∃ r : ℝ, 2 * Real.pi * r = 6) (h2 : ∃ r : ℝ, 2 * Real.pi * r = 9) :
  (let r1 := Classical.some h1, V1 := Real.pi * r1^2 * 9,
       r2 := Classical.some h2, V2 := Real.pi * r2^2 * 6
   in V2 / V1 = 3 / 2) :=
by
  sorry

end ratio_of_volumes_l686_686214


namespace problem_conditions_l686_686976

theorem problem_conditions (a b c x : ℝ) :
  (∀ x, ax^2 + bx + c ≥ 0 ↔ (x ≤ -3 ∨ x ≥ 4)) →
  (a > 0) ∧
  (∀ x, bx + c > 0 → x > -12 = false) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ (x < -1/4 ∨ x > 1/3)) ∧
  (a + b + c ≤ 0) :=
by
  sorry

end problem_conditions_l686_686976


namespace solve_df1_l686_686987

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (df1 : ℝ)

-- The condition given in the problem
axiom func_def : ∀ x, f x = 2 * x * df1 + (Real.log x)

-- Express the relationship from the derivative and solve for f'(1) = -1
theorem solve_df1 : df1 = -1 :=
by
  -- Here we will insert the proof steps in Lean, but they are omitted in this statement.
  sorry

end solve_df1_l686_686987


namespace exists_minimal_AB_CD_line_l686_686386

variables {R : Type*} [Real](A B C D : affine_subspace R (space R)) (P : point R)

-- Given conditions
def parallel_planes (A B : affine_subspace R (space R)) : Prop := A ∥ B
def line_intersection (g : line R) (P : point R) : Prop := g ⊧ point P

theorem exists_minimal_AB_CD_line
    (h_parallel_AB : parallel_planes A B)
    (h_parallel_CD : parallel_planes C D)
    (P : point R):
    ∃ g : line R, line_intersection g P ∧ intersects g A B ∧ intersects g C D ∧ distance AB = distance CD ∧ minimal_distance AB :=
sorry

end exists_minimal_AB_CD_line_l686_686386


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686144

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686144


namespace distinct_units_digits_of_integral_cubes_l686_686528

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686528


namespace distinct_units_digits_of_perfect_cube_l686_686506

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686506


namespace part1_part2_l686_686985

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - 2 * cos x ^ 2

theorem part1 (β : ℝ) (h : 0 ≤ β ∧ β ≤ π / 2) : -2 ≤ f β ∧ f β ≤ 1 :=
sorry

theorem part2 (α : ℝ) (h : tan α = 2 * sqrt 3) : f α = 10 / 13 :=
sorry

end part1_part2_l686_686985


namespace curve_not_parabola_l686_686600

theorem curve_not_parabola (k : ℝ) : ¬ ∃ a b c t : ℝ, a * t^2 + b * t + c = x^2 + k * y^2 - 1 := sorry

end curve_not_parabola_l686_686600


namespace simplify_tangent_expression_l686_686704

theorem simplify_tangent_expression :
  let t15 := Real.tan (15 * Real.pi / 180)
  let t30 := Real.tan (30 * Real.pi / 180)
  1 = Real.tan (45 * Real.pi / 180)
  calc
    (1 + t15) * (1 + t30) = 2 := by
  sorry

end simplify_tangent_expression_l686_686704


namespace circle_radius_and_triangle_area_l686_686236

-- Define the structure of our geometric setup
structure Triangle :=
  (A : ℝ × ℝ)
  (M : ℝ × ℝ)
  (T : ℝ × ℝ)
  (isosceles : A ≠ M ∧ A ≠ T ∧ M ≠ T)

structure Circle :=
  (O : ℝ × ℝ)
  (R : ℝ)

-- Definitions for distances from midpoints of arcs to the triangle sides
def dist_midpoint_AT_side_AT (Ω : Circle) (A T : ℝ × ℝ) : ℝ := 3
def dist_midpoint_MT_side_MT (Ω : Circle) (M T : ℝ × ℝ) : ℝ := 1.6

-- Definition of the radius being 5 and the area of triangle
def radius_is_5 (Ω : Circle) : Prop := Ω.R = 5

def area_of_triangle (A M T : ℝ × ℝ) : ℝ :=
  let a := dist A M
  let b := dist M T
  let c := dist T A
  0.25 * real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))

def area_is_168sqrt21_div_25 (A M T : ℝ × ℝ) : Prop :=
  area_of_triangle A M T = 168 * real.sqrt 21 / 25

-- The theorem which combines the conditions and the target conclusion
theorem circle_radius_and_triangle_area (Ω : Circle) (tri : Triangle)
  (h1 : dist_midpoint_AT_side_AT Ω tri.A tri.T = 3)
  (h2 : dist_midpoint_MT_side_MT Ω tri.M tri.T = 1.6) :
  radius_is_5 Ω ∧ area_is_168sqrt21_div_25 tri.A tri.M tri.T :=
begin
  sorry
end

end circle_radius_and_triangle_area_l686_686236


namespace Francine_not_working_days_l686_686366

-- Conditions
variables (d : ℕ) -- Number of days Francine works each week
def distance_per_day : ℕ := 140 -- Distance Francine drives each day
def total_distance_4_weeks : ℕ := 2240 -- Total distance in 4 weeks
def days_per_week : ℕ := 7 -- Days in a week

-- Proving that the number of days she does not go to work every week is 3
theorem Francine_not_working_days :
  (4 * distance_per_day * d = total_distance_4_weeks) →
  ((days_per_week - d) = 3) :=
by sorry

end Francine_not_working_days_l686_686366


namespace gymnastics_mean_stability_l686_686620

theorem gymnastics_mean_stability (scores : List ℝ) (h : ∃ lowest highest, (lowest = scores.minimum ∧ highest = scores.maximum)) :
  let filtered_scores := scores.erase lowest.erase highest
  let mean (scores : List ℝ) := scores.sum / scores.length
  mean filtered_scores = mean scores - ((lowest + highest) / scores.length) := 
sorry

end gymnastics_mean_stability_l686_686620


namespace B_pow_nine_equals_identity_l686_686639

open Matrix
open Real

-- Declare the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![cos (π / 4), 0, -sin (π / 4)],
  ![0, 1, 0],
  ![sin (π / 4), 0, cos (π / 4)]
]

-- The proof statement to verify that B^9 equals the identity matrix
theorem B_pow_nine_equals_identity : B ^ 9 = 1 :=
by
  sorry

end B_pow_nine_equals_identity_l686_686639


namespace tan_product_l686_686698

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end tan_product_l686_686698


namespace positiveDifferenceOfSolutions_l686_686184

noncomputable def positiveDifferenceSolutions : ℚ :=
  let quadratic : Polynomial ℚ := Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 21 * Polynomial.X + Polynomial.C 46
  let sol1 := ((-Polynomial.C 21 : Polynomial ℚ) - (Polynomial.C 361 + Polynomial.C 4 * Polynomial.C 2 * Polynomial.C 46).sqrt) / (2 * Polynomial.C 2)
  let sol2 := ((-Polynomial.C 21 : Polynomial ℚ) + (Polynomial.C 361 + Polynomial.C 4 * Polynomial.C 2 * Polynomial.C 46).sqrt) / (2 * Polynomial.C 2)
  sol2 - sol1

theorem positiveDifferenceOfSolutions : positiveDifferenceSolutions = 27 / 2 := by
  -- the proof will go here
  sorry

end positiveDifferenceOfSolutions_l686_686184


namespace range_positive_integers_l686_686802

theorem range_positive_integers (k : List Int) (h1 : k = List.range' (-3) 12) :
  (k.filter (λ x => x > 0)).last' - (k.filter (λ x => x > 0)).head = 7 :=
by
  sorry

end range_positive_integers_l686_686802


namespace part_1_part_2_l686_686944

variable (a b c : ℝ)
variable (A B C : ℝ)

def triangle_sides_and_angles_opposite : Prop :=
  A + B + C = π ∧ a = 1 ∧ b = 1 ∧ c = 1 -- a, b, c are sides of triangle opposite angles A, B, C

def angle_relation : Prop :=
  C = 2 * A -- C = 2A

def cos_A_value : Prop :=
  Real.cos A = 3 / 4 -- cos A = 3/4

theorem part_1 (h1 : triangle_sides_and_angles_opposite) (h2 : angle_relation) (h3 : cos_A_value) :
  c / a = 3 / 2 := 
  sorry

theorem part_2 (h1 : triangle_sides_and_angles_opposite) (h2 : angle_relation) (h3 : cos_A_value) :
  2 * b = a + c := 
  sorry

end part_1_part_2_l686_686944


namespace relatively_prime_count_in_range_l686_686596

theorem relatively_prime_count_in_range :
  let count := (List.range' 11 (49 - 11 + 1)).filter (λ n => Nat.gcd n 21 = 1) in
  count.length = 28 := by
  sorry

end relatively_prime_count_in_range_l686_686596


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686092

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686092


namespace distinct_units_digits_of_perfect_cubes_l686_686452

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686452


namespace trapezoid_plane_l686_686784

/-- A quadrilateral is defined as a trapezoid if it has at least one pair of parallel sides -/
structure Trapezoid (V : Type*) [Inhabited V] :=
(a b c d : V)
(is_parallel : ∃ (t u v w : V), t ≠ u ∧ v ≠ w ∧
                (∀ (x : V), x ∈ affine_span ℝ ({t, u} : set V) ↔ x ∈ affine_span ℝ ({v, w} : set V)))

/-- If T is a trapezoid, then it uniquely determines a plane -/
theorem trapezoid_plane {V: Type*} [Inhabited V] (T: Trapezoid V) : 
  ∃ P : affine_subspace ℝ V, ∀ (x y : V), x ∈ P → y ∈ P → V ⊆ P :=
sorry

end trapezoid_plane_l686_686784


namespace range_of_f_l686_686354

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_f :
  ∀ y : ℝ, (y ≠ -27) ↔ (∃ x : ℝ, x ≠ -5 ∧ f x = y) :=
by
  intro y
  split
  · intro hy
    use (y / 3 + 4)
    split
    · intro h
      contradiction
    · simp [f, hy]
  · intro ⟨x, hx1, hx2⟩
    rw [←hx2]
    intro h
    contradiction

end range_of_f_l686_686354


namespace distinct_units_digits_of_cubes_l686_686430

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686430


namespace exactly_one_right_angle_l686_686934

noncomputable def triangle_with_one_right_angle (x : ℤ) : Prop :=
  15 + 20 > x ∧ 15 + x > 20 ∧ 20 + x > 15 ∧
  ((x^2 = 15^2 + 20^2) ∨ (15^2 + x^2 = 20^2) ∨ (20^2 + x^2 = 15^2))

theorem exactly_one_right_angle :
  ∃! (x : ℤ), triangle_with_one_right_angle x :=
begin
  sorry
end

end exactly_one_right_angle_l686_686934


namespace find_number_l686_686871

theorem find_number (x : ℤ) (h : (x + 305) / 16 = 31) : x = 191 :=
sorry

end find_number_l686_686871


namespace distinct_units_digits_of_cubes_l686_686537

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686537


namespace distance_between_x_intercepts_l686_686845

-- Definitions for the conditions
def line_eq (m : ℝ) (x1 y1 : ℝ) : ℝ → ℝ := λ x, m * (x - x1) + y1

def x_intercept (m : ℝ) (x1 y1 : ℝ) : ℝ := classical.some (exists_x_intercept m x1 y1)

lemma exists_x_intercept (m x1 y1 : ℝ) : ∃ x : ℝ, line_eq m x1 y1 x = 0 :=
begin
  use (y1 / m + x1),
  simp [line_eq]
end

-- Main theorem
theorem distance_between_x_intercepts : 
  let l1 := line_eq 4 8 20,
      l2 := line_eq (-3) 8 20 in
  dist (x_intercept 4 8 20) (x_intercept (-3) 8 20) = 35 / 3 :=
by {
  -- Define equations of the lines
  let line1 := (λ x, 4 * (x - 8) + 20),
  let line2 := (λ x, -3 * (x - 8) + 20),

  -- Compute x-intercepts
  let x_int1 := classical.some (exists_x_intercept 4 8 20),
  let x_int2 := classical.some (exists_x_intercept (-3) 8 20),

  -- The derivation of x-intercepts and the distance calculation should be skipped for now
  sorry
}

end distance_between_x_intercepts_l686_686845


namespace binomial_coefficient_x3y5_in_expansion_l686_686168

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686168


namespace sum_of_inverses_formula_l686_686670

noncomputable def sum_of_inverses (n : ℕ) : ℚ :=
  1 + ∑ i in Finset.range (n + 1), 1 / (∑ k in Finset.range (i + 1), (k + 1))

theorem sum_of_inverses_formula (n : ℕ) : sum_of_inverses n = 2 * n / (n + 1) :=
sorry

end sum_of_inverses_formula_l686_686670


namespace level_curves_correct_l686_686924

noncomputable def level_curves (f : ℝ → ℝ) (C : ℝ) : Prop :=
  ∀ (x y : ℝ), f y = sqrt (8 - abs (x - 8) - abs y) → (y = C ∨ y = -C)

-- Given condition: the form of the function
def function (x y: ℝ) : ℝ := sqrt (8 - abs (x - 8) - abs y)

-- The statement to be proven, which embodies the level curves condition
theorem level_curves_correct : 
  ∀ (x y C : ℝ), 
    function x y = function x C → 
    (0 ≤ C ∧ C ≤ sqrt 8) →
    (y = C ∨ y = -C) :=
by
  sorry

end level_curves_correct_l686_686924


namespace one_third_of_7_times_9_l686_686321

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_7_times_9_l686_686321


namespace abs_not_eq_three_implies_x_not_eq_three_l686_686304

theorem abs_not_eq_three_implies_x_not_eq_three (x : ℝ) (h : |x| ≠ 3) : x ≠ 3 :=
sorry

end abs_not_eq_three_implies_x_not_eq_three_l686_686304


namespace incorrect_operation_in_list_l686_686265

open Real

theorem incorrect_operation_in_list :
  ¬ (abs ((-2)^2) = -2) :=
by
  -- Proof will be added here
  sorry

end incorrect_operation_in_list_l686_686265


namespace coefficient_x3_y5_in_binomial_expansion_l686_686103

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686103


namespace transform_expression_l686_686262

theorem transform_expression :
  (-17) + (+3) - (-5) - (+8) = -17 + 3 + 5 - 8 :=
by sorry

end transform_expression_l686_686262


namespace speed_increase_l686_686852

theorem speed_increase 
  (initial_distance : ℝ) (initial_time : ℝ) (speed_increase_percentage : ℝ)
  (initial_speed : ℝ := initial_distance / initial_time) -- initial speed
  (new_speed : ℝ := initial_speed * (1 + speed_increase_percentage / 100)) -- new speed
  (initial_distance_eq : initial_distance = 16)
  (initial_time_eq : initial_time = 4)
  (speed_increase_percentage_eq : speed_increase_percentage = 50) :
  new_speed = 6 :=
by
  rw [initial_distance_eq, initial_time_eq, speed_increase_percentage_eq]
  have h1 : initial_speed = 16 / 4 := rfl
  have h2 : new_speed = 4 * 1.5 := calc
    new_speed = (16 / 4) * (1 + 50 / 100) : by rw [h1]
    ... = 4 * 1.5 : rfl
  show new_speed = 6
  rw h2
  exact rfl

end speed_increase_l686_686852


namespace distinct_units_digits_of_cube_l686_686595

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686595


namespace compute_binom_product_l686_686893

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l686_686893


namespace coefficient_x3y5_in_expansion_l686_686155

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686155


namespace museum_ticket_cost_l686_686814

theorem museum_ticket_cost (students teachers : ℕ) (student_ticket_cost teacher_ticket_cost : ℕ) 
  (h_students : students = 12) (h_teachers : teachers = 4) 
  (h_student_ticket_cost : student_ticket_cost = 1) (h_teacher_ticket_cost : teacher_ticket_cost = 3) :
  students * student_ticket_cost + teachers * teacher_ticket_cost = 24 :=
by
  rw [h_students, h_teachers, h_student_ticket_cost, h_teacher_ticket_cost]
  exact (12 * 1 + 4 * 3) 
-- This produces 24

end museum_ticket_cost_l686_686814


namespace simplify_tan_expression_l686_686709

theorem simplify_tan_expression :
  ∀ (tan_15 tan_30 : ℝ), 
  (tan_45 : ℝ) 
  (tan_45_eq_one : tan_45 = 1)
  (tan_sum_formula : tan_45 = (tan_15 + tan_30) / (1 - tan_15 * tan_30)),
  (1 + tan_15) * (1 + tan_30) = 2 :=
by
  intros tan_15 tan_30 tan_45 tan_45_eq_one tan_sum_formula
  sorry

end simplify_tan_expression_l686_686709


namespace total_museum_tickets_cost_l686_686820

theorem total_museum_tickets_cost (num_students num_teachers cost_student_ticket cost_teacher_ticket : ℕ) :
  num_students = 12 →
  num_teachers = 4 →
  cost_student_ticket = 1 →
  cost_teacher_ticket = 3 →
  num_students * cost_student_ticket + num_teachers * cost_teacher_ticket = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_museum_tickets_cost_l686_686820


namespace distinct_units_digits_of_perfect_cube_l686_686513

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686513


namespace coefficient_x3y5_in_expansion_l686_686159

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686159


namespace distinct_units_digits_perfect_cube_l686_686582

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686582


namespace axis_of_symmetry_for_sine_squared_l686_686732

theorem axis_of_symmetry_for_sine_squared :
  (∃ k : ℤ, 2 * x = k * π) → x = π :=
begin
  sorry
end

end axis_of_symmetry_for_sine_squared_l686_686732


namespace fixed_point_of_parabolas_l686_686650

theorem fixed_point_of_parabolas 
  (t : ℝ) 
  (fixed_x fixed_y : ℝ) 
  (hx : fixed_x = 2) 
  (hy : fixed_y = 12) 
  (H : ∀ t : ℝ, ∃ y : ℝ, y = 3 * fixed_x^2 + t * fixed_x - 2 * t) : 
  ∃ y : ℝ, y = fixed_y :=
by
  sorry

end fixed_point_of_parabolas_l686_686650


namespace ratio_of_volumes_l686_686215

theorem ratio_of_volumes (h1 : ∃ r : ℝ, 2 * Real.pi * r = 6) (h2 : ∃ r : ℝ, 2 * Real.pi * r = 9) :
  (let r1 := Classical.some h1, V1 := Real.pi * r1^2 * 9,
       r2 := Classical.some h2, V2 := Real.pi * r2^2 * 6
   in V2 / V1 = 3 / 2) :=
by
  sorry

end ratio_of_volumes_l686_686215


namespace geometry_problem_l686_686616

variable {A B C D E M N P Q : Point}
variable {f : Line}
variable {α : Angle}

-- Definitions for the conditions
def acute_angled_triangle (A B C : Point) : Prop :=
  ∀ ∠ABC < 90° ∧ ∠BCA < 90° ∧ ∠CAB < 90°

def angle_bisector (A B C : Point) (D : Point) (f : Line) : Prop :=
  is_angle_bisector A B C f ∧ f.intersect_line_segment B C = D

def altitude (A B : Point) (C : Point) (M : Point) (f : Line) : Prop :=
  is_perpendicular (C, A) (B, C) ∧ intersects f (altitude_line C A) M

def circumcircle (A B C : Point) (E : Point) : Prop :=
  E ∈ circumcircle A B C

-- The theorem to be proved, given the conditions
theorem geometry_problem
  (h1 : acute_angled_triangle A B C)
  (h2 : angle_bisector A B C D f)
  (h3 : ∃ Q, intersect_circumcircle E (altitude A B C M f Q))
  (h4 : ∃ P, intersect_circumcircle E (altitude B A C N f P)) :
  (BD / DC) = (PE * AN * ME) / (QE * CM * NP) := sorry

end geometry_problem_l686_686616


namespace equivalent_form_of_g_l686_686648

def g (x : ℝ) : ℝ := Real.sqrt (Real.sin x ^ 4 + 4 * Real.sin x ^ 2) - Real.sqrt (Real.cos x ^ 4 + 4 * Real.cos x ^ 2)

theorem equivalent_form_of_g (x : ℝ) : g x = Real.cos (2 * x) :=
by
  sorry

end equivalent_form_of_g_l686_686648


namespace Dawn_commissioned_paintings_l686_686634

theorem Dawn_commissioned_paintings (time_per_painting : ℕ) (total_earnings : ℕ) (earnings_per_hour : ℕ) 
  (h1 : time_per_painting = 2) 
  (h2 : total_earnings = 3600) 
  (h3 : earnings_per_hour = 150) : 
  (total_earnings / (time_per_painting * earnings_per_hour) = 12) :=
by 
  sorry

end Dawn_commissioned_paintings_l686_686634


namespace general_formula_of_sequence_l686_686378

open_locale classical

noncomputable theory

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = -1 ∧ ∀ n : ℕ, 0 < n → n * (a (n + 1) - a n) = 2 - a (n + 1)

theorem general_formula_of_sequence (a : ℕ → ℝ) (h : sequence a) (n : ℕ) (hn : 0 < n) : 
  a n = 2 - 3 / n := 
sorry

end general_formula_of_sequence_l686_686378


namespace tree_growth_duration_l686_686775

theorem tree_growth_duration :
  let initial_height := 4
  let growth_rate := 2
  let height_at_year (n : ℕ) := initial_height + growth_rate * n
  (∃ (n : ℕ), height_at_year 4 < height_at_year n ∧ height_at_year n = height_at_year 4 + height_at_year 4 / 3) → ∃ (n : ℕ), n = 6 :=
by
  intros h
  cases h with n hn
  use n
  have h4 := height_at_year 4
  have h_final := height_at_year n
  have h4_eq : h4 = 12 := by
    dsimp [height_at_year, initial_height, growth_rate]
    norm_num
  have h_final_eq : h_final = h4 + h4 / 3 := by
    exact hn.right
  have h4_plus_third : h4 + h4 / 3 = 16 := by
    rw [h4_eq]
    norm_num
  rw [h4_plus_third, height_at_year n] at h_final_eq
  have : n = 6 := by
    dsimp [height_at_year, initial_height, growth_rate] at *
    linarith
  exact this

end tree_growth_duration_l686_686775


namespace stormi_needs_more_money_l686_686720

theorem stormi_needs_more_money
  (cars_washed : ℕ) (price_per_car : ℕ)
  (lawns_mowed : ℕ) (price_per_lawn : ℕ)
  (bike_cost : ℕ)
  (h1 : cars_washed = 3)
  (h2 : price_per_car = 10)
  (h3 : lawns_mowed = 2)
  (h4 : price_per_lawn = 13)
  (h5 : bike_cost = 80) : 
  bike_cost - (cars_washed * price_per_car + lawns_mowed * price_per_lawn) = 24 := by
  sorry

end stormi_needs_more_money_l686_686720


namespace Sn_sum_b_first_n_general_formula_l686_686605

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def sum_a_first_n_general_formula (T3 : ℕ) (a1 a2 a5 : ℕ)
  (h1 : T3 = 9)
  (h2 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h3 : a2^2 = a1 * a5)
  : ∀ n, a_n n = 2 * n - 1 := sorry

def sum_b_first_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n i

theorem Sn_sum_b_first_n_general_formula (n : ℕ)
  : sum_b_first_n n = ↑n / (2 * n + 1) := sorry

end Sn_sum_b_first_n_general_formula_l686_686605


namespace distinct_cube_units_digits_l686_686455

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686455


namespace solve_for_x_l686_686285

noncomputable def x_log_equation : ℝ → Prop := λ x, log x 81 = 4 / 2

theorem solve_for_x : x_log_equation 9 :=
by
  sorry

end solve_for_x_l686_686285


namespace six_digit_squares_l686_686906

theorem six_digit_squares :
    ∃ n m : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 100 ≤ m ∧ m ≤ 999 ∧ n = m^2 ∧ (n = 390625 ∨ n = 141376) :=
by
  sorry

end six_digit_squares_l686_686906


namespace problem_theorem_l686_686980

noncomputable def ellipse_constants : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∃ e : ℝ, e = 1 / 2 ∧
  ∃ A F₁ F₂ : ℝ × ℝ, 
    -- Ellipse equation
    (∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧
    -- given conditions
    |A - F₁| = 2 ∧
    (|A - F₂| * |F₁ - A| = -2 * ((A - F₂) • (F₁ - A))) ∧
    -- Line l passes through F₂
    (∃ k : ℝ, ¬(k = 0)) ∧
    -- Proof parts
    ∀ (C : Prop), 
      (C ↔ ellipse_constants → ( ∃ (a b : ℝ), a > b ∧ b > 0 ∧ b^2 = 4 - 1 ∧ a = 2 ))

theorem problem_theorem : 
  (ellipse_constants → ( ∃ (a b : ℝ), a > b ∧ b > 0 ∧ b^2 = 4 - 1 ∧ a = 2 ∧ ⟨x, y⟩ : ℝ × ℝ, (x^2 / 4) + (y^2 / 3) = 1)) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 / 4) :=
by
  sorry


end problem_theorem_l686_686980


namespace scientific_notation_of_18M_l686_686000

theorem scientific_notation_of_18M : 18000000 = 1.8 * 10^7 :=
by
  sorry

end scientific_notation_of_18M_l686_686000


namespace coefficient_of_term_in_binomial_expansion_l686_686056

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686056


namespace distinct_units_digits_of_cubes_l686_686432

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686432


namespace one_third_of_7_times_9_l686_686319

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_7_times_9_l686_686319


namespace units_digit_of_perfect_cube_l686_686492

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686492


namespace coeff_x3y5_in_expansion_l686_686039

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686039


namespace range_of_f_l686_686353

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_f :
  ∀ y : ℝ, (y ≠ -27) ↔ (∃ x : ℝ, x ≠ -5 ∧ f x = y) :=
by
  intro y
  split
  · intro hy
    use (y / 3 + 4)
    split
    · intro h
      contradiction
    · simp [f, hy]
  · intro ⟨x, hx1, hx2⟩
    rw [←hx2]
    intro h
    contradiction

end range_of_f_l686_686353


namespace common_difference_of_arithmetic_sequence_l686_686005

variable (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (d : ℝ)
variable (h₁ : S_n 5 = -15) (h₂ : a_n 2 + a_n 5 = -2)

theorem common_difference_of_arithmetic_sequence :
  d = 4 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l686_686005


namespace ratio_of_cylinder_volumes_l686_686209

theorem ratio_of_cylinder_volumes (h l : ℝ) (h_h : h = 6) (h_l : l = 9) :
  let r_C := h / (2 * Real.pi),
      V_C := Real.pi * r_C^2 * l,
      r_D := l / (2 * Real.pi),
      V_D := Real.pi * r_D^2 * h in
  if V_D > V_C then V_D / V_C = 3 / 4 else V_C / V_D = 3 / 4 := by
  sorry

end ratio_of_cylinder_volumes_l686_686209


namespace work_completion_time_l686_686231

noncomputable def work_rate_A : ℚ := 1 / 12
noncomputable def work_rate_B : ℚ := 1 / 14

theorem work_completion_time : 
  (work_rate_A + work_rate_B)⁻¹ = 84 / 13 := by
  sorry

end work_completion_time_l686_686231


namespace table_tennis_match_possibilities_l686_686018

-- Define the conditions of the problem
def is_winning_condition (games: ℕ) : Prop := games = 3

-- Define the target proof problem
theorem table_tennis_match_possibilities : 
  let situations := 2 * 1 + 2 * (nat.choose 3 2) + 2 * (nat.choose 4 2)
  is_winning_condition 3 → situations = 20 :=
by
  intro h
  simp [is_winning_condition, h]
  sorry

end table_tennis_match_possibilities_l686_686018


namespace seongjun_ttakji_count_l686_686686

variable (S A : ℕ)

theorem seongjun_ttakji_count (h1 : (3/4 : ℚ) * S - 25 = 7 * (A - 50)) (h2 : A = 100) : S = 500 :=
sorry

end seongjun_ttakji_count_l686_686686


namespace smallest_fraction_l686_686196

def smallestFractionDivides (a b c d e f: ℕ) : Fraction := 
  let gcd := Nat.gcd (Nat.gcd a b) c
  let lcm := Nat.lcm (Nat.lcm d e) f
  Fraction.mk gcd lcm

theorem smallest_fraction : smallestFractionDivides 6 5 10 7 14 21 = Fraction.mk 1 42 := by
  sorry

end smallest_fraction_l686_686196


namespace sum_f_1_to_50_l686_686966

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x 
axiom f_sym : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom f_at_1 : f 1 = 2

theorem sum_f_1_to_50 : ∑ i in Finset.range 50, f (i + 1) = 2 :=
sorry

end sum_f_1_to_50_l686_686966


namespace minimum_m_value_minimum_problem_value_l686_686992

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem minimum_m_value (m : ℝ) :
  (∀ x_0 : ℝ, 3 * x_0 ∈ ℝ → f x_0 ≤ m) → m ≥ 2 :=
by
  sorry

theorem minimum_problem_value (a b : ℝ) :
  (3 * a + b = 2) →
  a > 0 → b > 0 →
  (∃ m : ℝ, m = 2 ∧ (1 / (2 * a) + 1 / (a + b)) = m) :=
by
  sorry

end minimum_m_value_minimum_problem_value_l686_686992


namespace right_triangle_hypotenuse_l686_686739

theorem right_triangle_hypotenuse :
  let a := log 8 125
  let b := log 2 25
  ∃ h : ℝ, (hypotenuse a b = h) → 8^h = 5^(3 * sqrt 5) :=
by
  let a := log 8 125
  let b := log 2 25
  use hypotenuse a b
  intros h hypotenuse_eq
  rw ← hypotenuse_eq
  sorry

end right_triangle_hypotenuse_l686_686739


namespace distinct_units_digits_of_cubes_l686_686561

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686561


namespace binomial_coefficient_x3y5_in_expansion_l686_686179

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686179


namespace test_score_after_preparation_l686_686746

-- Define the conditions in Lean 4
def score (k t : ℝ) : ℝ := k * t^2

theorem test_score_after_preparation (k t : ℝ)
    (h1 : score k 2 = 90) (h2 : k = 22.5) :
    score k 3 = 202.5 :=
by
  sorry

end test_score_after_preparation_l686_686746


namespace total_runs_of_a_b_c_l686_686797

/-- Suppose a, b, and c are the runs scored by three players in a cricket match. The ratios of the runs are given as a : b = 1 : 3 and b : c = 1 : 5. Additionally, c scored 75 runs. Prove that the total runs scored by all of them is 95. -/
theorem total_runs_of_a_b_c (a b c : ℕ) (h1 : a * 3 = b) (h2 : b * 5 = c) (h3 : c = 75) : a + b + c = 95 := 
by sorry

end total_runs_of_a_b_c_l686_686797


namespace num_distinct_units_digits_of_cubes_l686_686475

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686475


namespace problem_1_problem_2_problem_3_l686_686953

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else a (n-1) + a (n-1)

def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a (i + 1)

theorem problem_1 (n : ℕ) (h : n > 1) : ∀ i, (i < n → 1 < a i) := 
sorry

theorem problem_2 (h_geom : ∀ n, a (n + 1) = 2 * a n) : 
  S 6 / a 3 = 63 / 4 :=
sorry

theorem problem_3 (n : ℕ) : 1/2 * n * (n + 1) ≤ S n ∧ S n ≤ 2^n - 1 :=
sorry

end problem_1_problem_2_problem_3_l686_686953


namespace more_red_peaches_than_green_l686_686755

-- Given conditions
def red_peaches : Nat := 17
def green_peaches : Nat := 16

-- Statement to prove
theorem more_red_peaches_than_green : red_peaches - green_peaches = 1 :=
by
  sorry

end more_red_peaches_than_green_l686_686755


namespace calculate_expression_value_l686_686883

theorem calculate_expression_value : 
  (Real.log 4 / Real.log (Real.sqrt 2)) + Real.exp (Real.log 3) + (0.125) ^ (-2 / 3) = 9 := 
by
  sorry

end calculate_expression_value_l686_686883


namespace binomial_coefficient_x3y5_in_expansion_l686_686131

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686131


namespace min_value_fraction_expression_l686_686964

theorem min_value_fraction_expression {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := 
by
  sorry

end min_value_fraction_expression_l686_686964


namespace solution_percentage_correct_l686_686247

def Liquid (volume : ℝ) (percentage : ℝ) : Type :=
  {volume : ℝ, percentage : ℝ}

def liquid_a := Liquid.mk 100 0.25
def liquid_b := Liquid.mk 90 0.3
def liquid_c := Liquid.mk 60 0.4
def liquid_d := Liquid.mk 50 0.2
def liquid_e := Liquid.mk 30 0.1
def liquid_f := Liquid.mk 70 0.35

noncomputable def total_solution (liquids : List (ℝ × ℝ)) : ℝ :=
  liquids.foldr (λ (volume_percentage : ℝ × ℝ) acc =>
    acc + (volume_percentage.fst * volume_percentage.snd)) 0

noncomputable def total_volume (liquids : List (ℝ × ℝ)) : ℝ :=
  liquids.foldr (λ (volume_percentage : ℝ × ℝ) acc =>
    acc + volume_percentage.fst) 0

noncomputable def percentage_solution (total_solution : ℝ) (total_volume : ℝ) : ℝ :=
  (total_solution / total_volume) * 100

theorem solution_percentage_correct :
  percentage_solution (total_solution [(100, 0.25), (90, 0.3), (60, 0.4), (50, 0.2), (30, 0.1), (70, 0.35)])
                      (total_volume [(100, 0.25), (90, 0.3), (60, 0.4), (50, 0.2), (30, 0.1), (70, 0.35)])
  = 28.375 := sorry

end solution_percentage_correct_l686_686247


namespace sum_of_first_n_odd_numbers_l686_686671

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n > 0) : 
  ∑ k in finset.range n, (2 * k + 1) = (n + 1) ^ 2 := 
sorry

end sum_of_first_n_odd_numbers_l686_686671


namespace total_balls_l686_686666

theorem total_balls (colors : ℕ) (balls_per_color : ℕ) (h_colors : colors = 10) (h_balls_per_color : balls_per_color = 35) : 
    colors * balls_per_color = 350 :=
by
  -- Import necessary libraries
  sorry

end total_balls_l686_686666


namespace circle_radius_solution_l686_686960

theorem circle_radius_solution (
  (triangle_abc : Type) (O : triangle_abc)
  (a b c : triangle_abc) 
  (H_a H_b H_c R : ℝ)
  (d_Oa : dist O a = H_a)
  (d_Ob : dist O b = H_b)
  (d_Oc : dist O c = H_c)
) : R^3 - (H_a^2 + H_b^2 + H_c^2) * R - 2 * H_a * H_b * H_c = 0 := 
sorry

end circle_radius_solution_l686_686960


namespace variance_shift_l686_686608

theorem variance_shift (n : ℕ) (x : ℕ → ℝ) (h : (∑ i in Finset.range n, (x i - (∑ i in Finset.range n, x i) / n)^2) / n = 0.6) :
  ((∑ i in Finset.range n, (x i - 2013 - ((∑ i in Finset.range n, x i - 2013) / n))^2) / n = 0.6) :=
sorry

end variance_shift_l686_686608


namespace golden_seedless_cost_l686_686597

-- Definitions based on the given conditions
def cost_per_scoop_natural_seedless := 3.45
def scoops_natural_seedless := 20
def scoops_golden_seedless := 20
def desired_cost_per_scoop_mixture := 3

theorem golden_seedless_cost (x : ℝ) :
  (scoops_natural_seedless * cost_per_scoop_natural_seedless + scoops_golden_seedless * x) / (scoops_natural_seedless + scoops_golden_seedless) = desired_cost_per_scoop_mixture →
  x = 2.55 :=
by
  sorry

end golden_seedless_cost_l686_686597


namespace operation_correct_l686_686780

theorem operation_correct : ∀ (x : ℝ), (x - 1)^2 = x^2 + 1 - 2x := 
by
  intro x
  sorry

end operation_correct_l686_686780


namespace amount_lent_l686_686244

variables (A_b_lend : ℝ)
          (gain_of_B : ℝ := 315)
          (rate_A_to_B : ℝ := 0.10)
          (rate_B_to_C : ℝ := 0.13)
          (years : ℝ := 3)

-- Define the conditions
def interest_from_c (P : ℝ):= P * rate_B_to_C * years
def interest_to_a (P : ℝ) := P * rate_A_to_B * years
def gain (P : ℝ) := interest_from_c P - interest_to_a P

-- Prove the main statement
theorem amount_lent (P : ℝ) : gain P = gain_of_B → P = 1166.67 := by
  sorry

end amount_lent_l686_686244


namespace inequality_one_inequality_system_l686_686718

theorem inequality_one (x : ℝ) : 2 * x + 3 ≤ 5 * x ↔ x ≥ 1 := sorry

theorem inequality_system (x : ℝ) : 
  (5 * x - 1 ≤ 3 * (x + 1)) ∧ 
  ((2 * x - 1) / 2 - (5 * x - 1) / 4 < 1) ↔ 
  (-5 < x ∧ x ≤ 2) := sorry

end inequality_one_inequality_system_l686_686718


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686085

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686085


namespace matrix_inverse_eq_scaling_l686_686662

variable (d k : ℚ)

def B : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 3],
  ![4, 5, d],
  ![6, 7, 8]
]

theorem matrix_inverse_eq_scaling :
  (B d)⁻¹ = k • (B d) →
  d = 13/9 ∧ k = -329/52 :=
by
  sorry

end matrix_inverse_eq_scaling_l686_686662


namespace distinct_units_digits_of_cubes_l686_686560

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686560


namespace percent_gain_l686_686804

def initial_price := 100.0
def price_after_increase := initial_price * 1.35
def price_after_first_discount := price_after_increase * 0.9
def final_price := price_after_first_discount * 0.85

theorem percent_gain:
  let gain := final_price - initial_price in
  let percent_gain := (gain / initial_price) * 100 in
  percent_gain = 3.275 :=
by
  sorry

end percent_gain_l686_686804


namespace terminating_decimal_count_l686_686360

theorem terminating_decimal_count :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 1000 ∧ (∀ m : ℕ, (2100 * m).nat_abs.gcd 10 = 1 → n = 47) :=
begin
  sorry
end

end terminating_decimal_count_l686_686360


namespace largest_base6_five_digit_l686_686182

def base6_to_base10 : ℕ → ℕ
| n :=
  let digits := [5, 5, 5, 5, 5] in
  digits.zipWith (λ digit idx, digit * (6 ^ idx)) [4, 3, 2, 1, 0]
  |> List.sum

theorem largest_base6_five_digit : base6_to_base10 55555 = 7775 :=
by
  sorry

end largest_base6_five_digit_l686_686182


namespace correct_propositions_are_two_l686_686358

noncomputable def double_factorial : ℕ → ℕ 
| 0       := 1
| 1       := 1
| n       := if n % 2 = 0 then n * double_factorial (n - 2) else n * double_factorial (n - 2)

def proposition_1 : Prop := double_factorial 2011 * double_factorial 2010 = Nat.factorial 2011
def proposition_2 : Prop := double_factorial 2010 = 2 ^ 1005 * Nat.factorial 1005
def proposition_3 : Prop := double_factorial 2010 * double_factorial 2010 = Nat.factorial 2011
def proposition_4 : Prop := Nat.digits 10 (double_factorial 2011) % 10 = 5

def number_of_correct_propositions : Nat :=
  Nat.count (λ p, p) [proposition_1, proposition_2, proposition_3, proposition_4]

theorem correct_propositions_are_two : number_of_correct_propositions = 2 := by
  sorry

end correct_propositions_are_two_l686_686358


namespace simplify_tan_expression_l686_686691

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end simplify_tan_expression_l686_686691


namespace stormi_needs_more_money_l686_686721

theorem stormi_needs_more_money
  (cars_washed : ℕ) (price_per_car : ℕ)
  (lawns_mowed : ℕ) (price_per_lawn : ℕ)
  (bike_cost : ℕ)
  (h1 : cars_washed = 3)
  (h2 : price_per_car = 10)
  (h3 : lawns_mowed = 2)
  (h4 : price_per_lawn = 13)
  (h5 : bike_cost = 80) : 
  bike_cost - (cars_washed * price_per_car + lawns_mowed * price_per_lawn) = 24 := by
  sorry

end stormi_needs_more_money_l686_686721


namespace avg_fish_in_bodies_of_water_l686_686681

def BoastPoolFish : ℕ := 75
def OnumLakeFish : ℕ := BoastPoolFish + 25
def RiddlePondFish : ℕ := OnumLakeFish / 2
def RippleCreekFish : ℕ := 2 * (OnumLakeFish - BoastPoolFish)
def WhisperingSpringsFish : ℕ := (3 * RiddlePondFish) / 2

def totalFish : ℕ := BoastPoolFish + OnumLakeFish + RiddlePondFish + RippleCreekFish + WhisperingSpringsFish
def averageFish : ℕ := totalFish / 5

theorem avg_fish_in_bodies_of_water : averageFish = 68 :=
by
  sorry

end avg_fish_in_bodies_of_water_l686_686681


namespace distinct_units_digits_of_cubes_l686_686438

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686438


namespace problem_parabola_l686_686997

theorem problem_parabola (p : ℝ) (h : p > 0) :
  let C := λ x y : ℝ, y^2 = 2 * p * x,
      M := (4, -4 * real.sqrt 2),
      L := λ x y : ℝ, y = 2 * x - 8 in
  (M.2 ^ 2 = 2 * p * M.1) →
  p = 4 ∧
  let F := (2, 0),
      y_val := solve_poly_eqns 8 2 -8 in
  let A := (some x, some (y_val x)),
      B := (some x, some (y_val x)) in
  abs ((A.2 + B.2) * sqrt ((A.2 + B.2)^2 - 4 * A.2 * B.2)) / 2 = 12 := by
sorry

end problem_parabola_l686_686997


namespace distance_x_intercepts_correct_l686_686840

noncomputable def distance_between_x_intercepts : ℝ :=
  let slope1 : ℝ := 4
  let slope2 : ℝ := -3
  let intercept_point : Prod ℝ ℝ := (8, 20)
  let line1 (x : ℝ) : ℝ := slope1 * (x - intercept_point.1) + intercept_point.2
  let line2 (x : ℝ) : ℝ := slope2 * (x - intercept_point.1) + intercept_point.2
  let x_intercept1 : ℝ := (0 - intercept_point.2) / slope1 + intercept_point.1
  let x_intercept2 : ℝ := (0 - intercept_point.2) / slope2 + intercept_point.1
  abs (x_intercept2 - x_intercept1)

theorem distance_x_intercepts_correct :
  distance_between_x_intercepts = 35 / 3 :=
sorry

end distance_x_intercepts_correct_l686_686840


namespace minimum_ones_l686_686727

def is_valid_block (block : Finset (Fin 121)) : Prop :=
  ∃ cells : list (Fin 121),
  cells.length = 4 ∧
  ∃ index : Fin 2 → Fin 2 → ℕ,
  (∀ i j, index i j ∈ cells) ∧
  (∑ c in cells, c) % 2 = 1

def valid_grid (grid : Fin 121 → ℕ) : Prop :=
  (∀ block : Finset (Fin 121), block.card = 4 → is_valid_block block) ∧
  (∀ i, (grid i) = 0 ∨ (grid i) = 1)

theorem minimum_ones :
  ∃ grid : Fin 121 → ℕ,
  valid_grid grid ∧ (Finset.univ.filter (λ i, grid i = 1)).card = 25 :=
sorry

end minimum_ones_l686_686727


namespace ratio_of_cylinder_volumes_l686_686207

theorem ratio_of_cylinder_volumes (h l : ℝ) (h_h : h = 6) (h_l : l = 9) :
  let r_C := h / (2 * Real.pi),
      V_C := Real.pi * r_C^2 * l,
      r_D := l / (2 * Real.pi),
      V_D := Real.pi * r_D^2 * h in
  if V_D > V_C then V_D / V_C = 3 / 4 else V_C / V_D = 3 / 4 := by
  sorry

end ratio_of_cylinder_volumes_l686_686207


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686093

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686093


namespace one_third_of_seven_times_nine_l686_686330

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 :=
by
  calc
    (1 / 3) * (7 * 9) = (1 / 3) * 63 := by rw [mul_comm 7 9, mul_assoc 1 7 9]
                      ... = 21 := by norm_num

end one_third_of_seven_times_nine_l686_686330


namespace distinct_units_digits_of_integral_cubes_l686_686521

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686521


namespace distinct_units_digits_of_cubes_l686_686553

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686553


namespace find_one_third_of_product_l686_686336

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end find_one_third_of_product_l686_686336


namespace infinite_midpoints_l686_686294

theorem infinite_midpoints (S : Set (ℝ × ℝ)) (H : ∀ p ∈ S, ∃ a b ∈ S, p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) : (S = Set.univ) :=
by
  sorry

end infinite_midpoints_l686_686294


namespace value_of_x_l686_686396

noncomputable def point_on_terminal_side (x : ℝ) :=
  let P := (x, 3)
  ∃ θ : ℝ, (∃ y : ℝ, y = 3 ∧ P = (x, y)) ∧ (∃ cos θ = -4 / 5)

theorem value_of_x (x : ℝ) (h1 : point_on_terminal_side x) (h2 : ∃ θ : ℝ, cos θ = -4 / 5) :
  x = -4 :=
sorry

end value_of_x_l686_686396


namespace volume_ratio_l686_686217

def rect_height_vert : ℝ := 9
def rect_circumference_vert : ℝ := 6
def radius_vert : ℝ := rect_circumference_vert / (2 * Real.pi)
def volume_vert : ℝ := Real.pi * radius_vert^2 * rect_height_vert
-- volume_vert is calculated as πr^2h where r = 3/π

def rect_height_horiz : ℝ := 6
def rect_circumference_horiz : ℝ := 9
def radius_horiz : ℝ := rect_circumference_horiz / (2 * Real.pi)
def volume_horiz : ℝ := Real.pi * radius_horiz^2 * rect_height_horiz
-- volume_horiz is calculated as πr^2h where r = 9/(2π)

theorem volume_ratio : (max volume_vert volume_horiz) / (min volume_vert volume_horiz) = 3 / 4 := 
by sorry

end volume_ratio_l686_686217


namespace range_of_a_l686_686970

variables (f : ℝ → ℝ) (a : ℝ)

-- Definition of an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of a monotonically increasing function on [0, ∞)
def mono_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

-- Conditions
axiom h1 : even_function f
axiom h2 : mono_increasing_on f (Set.Ici 0)
axiom h3 : f (Real.log 2 a) + f (Real.log (1 / 2) a) ≤ 2 * f 1

-- Proof Goal
theorem range_of_a : 1 / 2 ≤ a ∧ a ≤ 2 :=
  sorry

end range_of_a_l686_686970


namespace probability_two_S_tiles_l686_686741

-- Definitions used in conditions
def tiles : List Char := ['G', 'A', 'U', 'S', 'S']

def count_ways_to_pick_two (lst : List Char) : ℕ :=
(lst.length.choose 2)

def count_favorable_outcomes (lst : List Char) : ℕ :=
(lst.count ('S' ==)) choose 2

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
(favorable : ℚ) / (total : ℚ)

-- Lean 4 statement of the proof problem
theorem probability_two_S_tiles :
  probability (count_favorable_outcomes tiles) (count_ways_to_pick_two tiles) = 1 / 10 :=
by {
  sorry
}

end probability_two_S_tiles_l686_686741


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686150

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686150


namespace find_blue_balls_l686_686229

def blueBallsCount (B : ℕ) : Prop :=
  (5 / (B + 9) : ℚ) * (4 / (B + 8) : ℚ) = 0.09523809523809523

theorem find_blue_balls : ∃ B : ℕ, blueBallsCount B ∧ B = 6 := 
by {
  use 6,
  simp [blueBallsCount],
  norm_num,
  sorry
}

end find_blue_balls_l686_686229


namespace exists_subset_with_chain_property_l686_686280

variable (A : ℕ → Set ℕ) (r : ℕ)
variable (h_disjoint : ∀ i j : ℕ, i ≠ j → set.disjoint (A i) (A j))
variable (h_union : ∀ n : ℕ, ∃ i : ℕ, i < r ∧ n ∈ A i)

theorem exists_subset_with_chain_property :
  ∃ (i : ℕ) (m : ℕ), i < r ∧ (∀ k : ℕ, ∃ a : ℕ → ℕ, (∀ j : ℕ, 1 ≤ j → j < k → 0 < a(j+1) - a(j) ∧ a(j+1) - a(j) ≤ m) ∧ (∀ j : ℕ, j < k → a j ∈ A i)) :=
by
  sorry

end exists_subset_with_chain_property_l686_686280


namespace distinct_units_digits_of_integral_cubes_l686_686530

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686530


namespace distinct_units_digits_of_cubes_l686_686542

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686542


namespace coeff_x3y5_in_expansion_l686_686037

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686037


namespace distinct_units_digits_perfect_cube_l686_686572

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686572


namespace trigonometric_signs_problem_l686_686394

open Real

theorem trigonometric_signs_problem (k : ℤ) (θ α : ℝ) 
  (hα : α = 2 * k * π - π / 5)
  (h_terminal_side : ∃ m : ℤ, θ = α + 2 * m * π) :
  (sin θ / |sin θ|) + (cos θ / |cos θ|) + (tan θ / |tan θ|) = -1 := 
sorry

end trigonometric_signs_problem_l686_686394


namespace rectangle_area_increase_l686_686798

theorem rectangle_area_increase (L W : ℝ) :
  let L' := 1.15 * L
      W' := 1.15 * W
      A  := L * W
      A' := L' * W'
  in (A' - A) / A * 100 = 32.25 :=
by
  let L' := 1.15 * L
  let W' := 1.15 * W
  let A := L * W
  let A' := L' * W'
  have h : A' = 1.15^2 * A := by sorry  -- This line proves A' = 1.15^2 * A
  show (A' - A) / A * 100 = 32.25 from sorry  -- This line proves (A' - A) / A * 100 = 32.25
  sorry

end rectangle_area_increase_l686_686798


namespace number_of_correct_propositions_l686_686982

variable (a x y : ℝ)

-- Define the propositions
def prop1 : Prop := (a^2 * x > a^2 * y) → (x > y)
def prop2 : Prop := (a^2 * x ≥ a^2 * y) → (x ≥ y)
def prop3 : Prop := (x / a^2 ≥ y / a^2) → (x ≥ y)
def prop4 : Prop := (x ≥ y) → (x / a^2 ≥ y / a^2)

-- Count the number of true propositions
def countTrueProps : ℕ :=
  if prop1 a x y then
    if prop3 a x y then 2 else 1
  else
  if prop3 a x y then 1 else 0

theorem number_of_correct_propositions (h : a ≠ 0) : countTrueProps a x y = 2 := by
  sorry

end number_of_correct_propositions_l686_686982


namespace one_third_of_seven_times_nine_l686_686327

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l686_686327


namespace points_on_common_unit_circle_l686_686015

noncomputable def unit_circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { z | (z.1 - center.1)^2 + (z.2 - center.2)^2 = radius^2 }

variables {O A B C : ℝ × ℝ}

axiom unit_circles_intersect_at_O (C1 C2 C3 : set (ℝ × ℝ)) :
  unit_circle O 1 = C1 ∧ unit_circle O 1 = C2 ∧ unit_circle O 1 = C3 ∧
  A ∈ C1 ∧ B ∈ C2 ∧ C ∈ C3

theorem points_on_common_unit_circle (A B C O : ℝ × ℝ) :
  (∃ (C1 C2 C3 : set (ℝ × ℝ)), 
   unit_circles_intersect_at_O C1 C2 C3) →
  ∃ D : ℝ × ℝ, ∃ (C : set (ℝ × ℝ)), unit_circle D 1 = C ∧ A ∈ C ∧ B ∈ C ∧ C ∈ C :=
by
  sorry

end points_on_common_unit_circle_l686_686015


namespace distinct_units_digits_of_cubes_l686_686569

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686569


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686141

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686141


namespace nth_expression_sum_of_cubes_l686_686983

theorem nth_expression_sum_of_cubes (n : ℕ) :
  (∑ k in Finset.range (2 * n), (k + 1)^3) = (n * (2 * n - 1))^2 :=
by
  sorry

end nth_expression_sum_of_cubes_l686_686983


namespace units_digit_of_sum_sequence_l686_686772

theorem units_digit_of_sum_sequence : 
  let seq := (List.range' 1 9).map (λ n, (n.factorial + n ^ 2) % 10)
  let sum_units_digits := (seq.sum % 10)
  sum_units_digits = 8 :=
by
  -- conditions for the problem
  obtain ⟨f, g⟩ : (∀ n : ℕ, n ≥ 5 → (n.factorial % 10 = 0)) ∧ (forall n, seq.sum % 10 = sum_units_digits), sorry

end units_digit_of_sum_sequence_l686_686772


namespace distinct_units_digits_perfect_cube_l686_686579

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686579


namespace distinct_units_digits_of_cubes_l686_686557

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686557


namespace planes_perpendicular_l686_686403

theorem planes_perpendicular
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ)
  (h₁ : a = (2, 4, -3))
  (h₂ : b = (-1, 2, 2))
  (h3 : a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0) :
  ∃ α β : Plane, α.normal_vector = a ∧ β.normal_vector = b ∧ α ⊥ β :=
sorry

end planes_perpendicular_l686_686403


namespace sequence_formula_l686_686954

def S (n : ℕ) : ℕ := n^2 + n + 1

def a : ℕ → ℕ
| 0       := 3
| (n+1)   := if n = 0 then 3 else 2 * (n+1)

theorem sequence_formula (n : ℕ) : 
  a n = ite (n = 1) 3 (ite (n ≥ 2) (2 * n) 0) :=
by 
  intros,
  cases n,
  { simp [a, S], 
    sorry 
  },
  { simp [a, S, nat.succ_ne_zero], 
    sorry 
  }

end sequence_formula_l686_686954


namespace water_for_1200ml_flour_l686_686016

-- Define the condition of how much water is mixed with a specific amount of flour
def water_per_flour (flour water : ℕ) : Prop :=
  water = (flour / 400) * 100

-- Given condition: Maria uses 100 mL of water for every 400 mL of flour
def condition : Prop := water_per_flour 400 100

-- Problem Statement: How many mL of water for 1200 mL of flour?
theorem water_for_1200ml_flour (h : condition) : water_per_flour 1200 300 :=
sorry

end water_for_1200ml_flour_l686_686016


namespace max_diff_S_n_S_m_l686_686381

def a_n (n : ℕ) : ℤ := -(n^2 : ℤ) + 12 * n - 32

def S_n (n : ℕ) : ℤ := ∑ i in Finset.range n, a_n (i + 1)

theorem max_diff_S_n_S_m (n m : ℕ) (hnm : n > m) :
  ∃ (N M : ℕ), N > M ∧ (S_n N - S_n M = 10) :=
sorry

end max_diff_S_n_S_m_l686_686381


namespace distinct_units_digits_perfect_cube_l686_686571

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686571


namespace distance_between_x_intercepts_l686_686839

theorem distance_between_x_intercepts 
  (m₁ m₂ : ℝ) (p : ℝ × ℝ) (h₁ : m₁ = 4) (h₂ : m₂ = -3) (h₃ : p = (8, 20)) : 
  let x₁ := (20 - (20 - m₁ * (8 - 8))) / m₁,
      x₂ := (20 - (20 - m₂ * (8 - 8))) / m₂ in
  |x₁ - x₂| = 35 / 3 :=
by
  sorry

end distance_between_x_intercepts_l686_686839


namespace units_digit_of_perfect_cube_l686_686504

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686504


namespace integral_of_sin_fraction_l686_686287

theorem integral_of_sin_fraction :
  (∫ x in 0..(Real.pi / 2), (sin x) / (5 + 3 * sin x)) = 
  (Real.pi - 5 * Real.arctan 2 + 5 * Real.arctan (3 / 4)) / 6 :=
sorry

end integral_of_sin_fraction_l686_686287


namespace probability_red_or_white_l686_686192

-- Definitions based on the conditions
def total_marbles := 20
def blue_marbles := 5
def red_marbles := 9
def white_marbles := total_marbles - (blue_marbles + red_marbles)

-- Prove that the probability of selecting a red or white marble is 3/4
theorem probability_red_or_white : (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 :=
by sorry

end probability_red_or_white_l686_686192


namespace num_distinct_units_digits_of_cubes_l686_686477

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686477


namespace permutations_not_adjacent_l686_686925

theorem permutations_not_adjacent (n : ℕ) (h : 4 ≤ n) :
  (number_of_permutations n) = (n^2 - 5n + 8) * ((n - 2)!) :=
sorry

end permutations_not_adjacent_l686_686925


namespace correct_propositions_l686_686392

variables {m n : Line} {α β : Plane}

theorem correct_propositions
  (h1 : (m ⊥ α) → (n ⊥ β) → (m ⊥ n) → (α ⊥ β))
  (h2 : (m ∥ α) → (n ∥ β) → (m ⊥ n) → (α ∥ β))
  (h3 : (m ⊥ α) → (n ∥ β) → (m ⊥ n) → (α ∥ β))
  (h4 : (m ⊥ α) → (n ∥ β) → (α ∥ β) → (m ⊥ n)) : 
  (h1 ∧ h4) ∧ ¬(h2 ∨ h3) :=
by
  sorry

end correct_propositions_l686_686392


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686042

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686042


namespace monotonic_intervals_f_range_a_f_le_x_l686_686416

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x - (a / x) - (a + 1) * log x

theorem monotonic_intervals_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≤ 1) :
  let I_increasing_1 := { x : ℝ | 0 < x ∧ x < a }
  let I_increasing_2 := { x : ℝ | 1 < x }
  let I_decreasing := { x : ℝ| a < x ∧ x < 1 }
  let I_increasing := if a = 1 then { x : ℝ | 0 < x } else I_increasing_1 ∪ I_increasing_2
  ∀ x y ∈ I_increasing, x < y → f a x < f a y ∧
  ∀ x y ∈ I_decreasing, x < y → f a x > f a y := sorry

theorem range_a_f_le_x :
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → f a x ≤ x) ↔ a ≥ 1 / (E - 1) := sorry

end monotonic_intervals_f_range_a_f_le_x_l686_686416


namespace correct_proposition_l686_686873

noncomputable def proposition_1 := 
  ∀ (line_a line_b : Type) (plane_p : Type), 
  (line_a ∥ plane_p ∧ plane_p ∋ line_b) → line_a ∥ line_b

noncomputable def proposition_2 := 
  ∀ (line_l : Type) (plane_alpha : Type), 
  (line_l ⟂ plane_alpha) ↔ (∀ (line_in_plane : plane_alpha), line_l ⟂ line_in_plane)

noncomputable def proposition_3 := 
  ∀ (line_a line_b : Type), 
  (¬(line_a ∩ line_b) → line_a ∥ line_b ∨ line_a ∧ line_b)

noncomputable def proposition_4 := 
  ∀ (plane_alpha plane_beta : Type) (p1 p2 p3 : plane_alpha), 
  (¬collinear p1 p2 p3 ∧ (∀ p, (p ∈ {p1, p2, p3} → dist p plane_beta = d))) → plane_alpha ∥ plane_beta

theorem correct_proposition :
  ¬proposition_1 ∧ ¬proposition_2 ∧ ¬proposition_3 ∧ proposition_4 := 
by {
  sorry
}

end correct_proposition_l686_686873


namespace number_divided_by_005_l686_686913

theorem number_divided_by_005 (number : ℝ) (h : number / 0.05 = 1500) : number = 75 :=
sorry

end number_divided_by_005_l686_686913


namespace find_x_values_l686_686191

noncomputable def is_prime (n : ℕ) : Prop := nat.prime n

theorem find_x_values :
  ∃ (x1 x2 : ℕ), (abs (x1 - 2008) = 0 ∨ abs (x1 - 2008) = 1) ∧
                 (abs (x2 - 2008) = 0 ∨ abs (x2 - 2008) = 1) ∧
                 (10 * x1 > 0) ∧ (10 * x2 > 0) ∧
                 is_prime ((3^x1) % 100) ∧ is_prime ((3^x2) % 100) ∧
                 x1 ≠ x2 ∧ (x1 = 2008 ∨ x1 = 2009) ∧ (x2 = 2008 ∨ x2 = 2009) :=
by  
  sorry

end find_x_values_l686_686191


namespace parabola_friendship_line_segment_length_MN_range_b_l686_686300

-- Part (1)
theorem parabola_friendship_line (a b c : ℝ) :
  (9 * a + 3 * b + c = 0) ∧ (c = 3) ∧ (a - b + c = 0) →
  (a = -1) ∧ (b = 2) ∧ (c = 3) :=
sorry

-- Part (2)
theorem segment_length_MN :
  let k := 1,
      b := 1,
      M := (1 : ℝ, 2 : ℝ),
      N := (-2 : ℝ, -1 : ℝ) in
  dist M N = 3 * Real.sqrt 2 :=
sorry

-- Part (3)
theorem range_b (a b : ℝ) :
  (∀ m : ℝ, ∃ (c : ℝ), 
    m + (1 - m) = 1 ∧ 
    a * (1 - 1 / m) ^ 2 + b * (1 - 1 / m) + (1 - m) = 0) →
  b ≥ 1 :=
sorry

end parabola_friendship_line_segment_length_MN_range_b_l686_686300


namespace magnitude_of_complex_solution_l686_686374

open Complex

theorem magnitude_of_complex_solution
  (z : ℂ)
  (h : (z + 2) / (z - 2) = I) :
  |z| = 2 :=
sorry

end magnitude_of_complex_solution_l686_686374


namespace average_speed_of_trip_l686_686794

noncomputable def average_speed : ℝ :=
  let distance1 := 30
  let distance2 := 30
  let speed1 := 48
  let speed2 := 24
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time

theorem average_speed_of_trip :
  average_speed = 32 :=
by
  -- We need to use the conditions given:
  let distance1 := 30
  let distance2 := 30
  let speed1 := 48
  let speed2 := 24
  let total_distance := distance1 + distance2

  -- Calculate the time for each segment:
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2

  -- Calculate the total time:
  let total_time := time1 + time2

  -- Calculate the average speed:
  have avg_speed := total_distance / total_time

  -- Prove the average speed is 32:
  have : avg_speed = 32 := sorry
  exact this

end average_speed_of_trip_l686_686794


namespace distinct_cube_units_digits_l686_686457

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686457


namespace percentage_off_at_sale_l686_686277

theorem percentage_off_at_sale
  (sale_price original_price : ℝ)
  (h1 : sale_price = 140)
  (h2 : original_price = 350) :
  (original_price - sale_price) / original_price * 100 = 60 :=
by
  sorry

end percentage_off_at_sale_l686_686277


namespace vectors_coplanar_l686_686877

open Matrix

-- Given vectors 
def vector_a : ℝ × ℝ × ℝ := (3, 0, 3)
def vector_b : ℝ × ℝ × ℝ := (8, 1, 6)
def vector_c : ℝ × ℝ × ℝ := (1, 1, -1)

-- Convert tuples to vector form
def to_vector3 {α} (x : α × α × α) : Fin 3 → α
  | ⟨0, _⟩ => x.1
  | ⟨1, _⟩ => x.2.1
  | ⟨2, _⟩ => x.2.2
  | _      => 0 -- This case will never occur for Fin 3

-- Create the matrix
def matrix_M : Matrix (Fin 3) (Fin 3) ℝ := 
  λ i j, match i, j with
    | ⟨0, _⟩, _ => to_vector3 vector_a j
    | ⟨1, _⟩, _ => to_vector3 vector_b j
    | ⟨2, _⟩, _ => to_vector3 vector_c j
    | _, _       => 0  -- Again, this case will never occur for Fin 3

-- Lean 4 statement of the problem.
theorem vectors_coplanar : det matrix_M = 0 :=
by sorry

end vectors_coplanar_l686_686877


namespace area_of_circle_below_line_y_eq_5_l686_686025

noncomputable def circle_area_beneath_line (x y : ℝ) : ℝ :=
  if x^2 - 4 * x + y^2 - 16 * y + 39 = 0 then
    let r := real.sqrt 29 in
    let θ := 2 * real.arccos (3 / r) in
    (29 / 2) * (θ - real.sin(θ))
  else 0

theorem area_of_circle_below_line_y_eq_5 :
  circle_area_beneath_line 2 8 = (29 / 2) * (2 * real.arccos (3 / real.sqrt 29) - real.sin (2 * real.arccos (3 / real.sqrt 29))) :=
sorry

end area_of_circle_below_line_y_eq_5_l686_686025


namespace probability_X_leq_1_l686_686251

-- Define the interval and the random variable
def interval := set.Icc (-2 : ℝ) 4
def random_variable (x : ℝ) : Prop := x ∈ interval

-- Define the probability calculation function
def geometric_probability (a b c : ℝ) : ℝ :=
  (c - a) / (b - a)

-- State the theorem for the geometric probability
theorem probability_X_leq_1 :
  geometric_probability (-2) 4 1 = 1 / 2 :=
by
  sorry

end probability_X_leq_1_l686_686251


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686080

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686080


namespace volume_ratio_cylinders_l686_686205

open Real

noncomputable def volume_ratio_larger_to_smaller (h₁ : Real) (h₂ : Real) (circumference₁ : Real) (circumference₂ : Real) : Real :=
  let r₁ := circumference₁ / (2 * pi)
  let r₂ := circumference₂ / (2 * pi)
  let V₁ := pi * r₁^2 * h₁
  let V₂ := pi * r₂^2 * h₂
  V₂ / V₁

theorem volume_ratio_cylinders : volume_ratio_larger_to_smaller 9 6 6 9 = 9 / 4 := by
  sorry

end volume_ratio_cylinders_l686_686205


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686095

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686095


namespace distinct_units_digits_perfect_cube_l686_686577

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686577


namespace tan_alpha_minus_pi_over_4_eq_neg_3_l686_686425

theorem tan_alpha_minus_pi_over_4_eq_neg_3
  (α : ℝ)
  (h1 : True) -- condition to ensure we define α in ℝ, "True" is just a dummy
  (a : ℝ × ℝ := (Real.cos α, -2))
  (b : ℝ × ℝ := (Real.sin α, 1))
  (h2 : ∃ k : ℝ, a = k • b) : 
  Real.tan (α - Real.pi / 4) = -3 :=
  sorry

end tan_alpha_minus_pi_over_4_eq_neg_3_l686_686425


namespace distinct_units_digits_of_perfect_cubes_l686_686443

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686443


namespace distinct_units_digits_of_cubes_l686_686546

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686546


namespace midpoint_ordinate_l686_686400

-- Define the functions and given conditions
def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := Real.cos x

def a : ℝ := sorry
axiom ha : 0 < a ∧ a < Real.pi / 2
axiom hpq : abs (f a - g a) = 1 / 4

-- Statement of the proof problem
theorem midpoint_ordinate :
  let P := (a, f a)
  let Q := (a, g a)
  let midpoint_y := (f a + g a) / 2
  midpoint_y = sqrt 31 / 8 :=
sorry

end midpoint_ordinate_l686_686400


namespace distinct_units_digits_of_cube_l686_686484

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686484


namespace range_of_f_l686_686351

noncomputable def f (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_of_f :
  (∀ y, ∃ x, x ≠ -5 ∧ f(x) = y) ↔ (y ≠ -27) :=
begin
  sorry
end

end range_of_f_l686_686351


namespace tan_product_l686_686697

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end tan_product_l686_686697


namespace distinct_cube_units_digits_l686_686458

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686458


namespace coefficient_of_term_in_binomial_expansion_l686_686062

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686062


namespace binomial_coefficient_x3y5_in_expansion_l686_686125

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686125


namespace find_cone_base_radius_l686_686405

-- Given conditions
variables {r l : ℝ}

-- Surface area and unfolding conditions
def cone_surface_area : ℝ := 3 * real.pi
def base_circumference (r : ℝ) : ℝ := 2 * real.pi * r
def semicircle_perimeter (l : ℝ) : ℝ := real.pi * l
def base_area (r : ℝ) : ℝ := real.pi * r^2
def semicircle_area (l : ℝ) : ℝ := (1 / 2) * real.pi * l^2

-- Theorem stating the radius of the cone's base is 1 under the given conditions
theorem find_cone_base_radius 
  (S : cone_surface_area = 3 * real.pi)
  (H1 : semicircle_perimeter l = base_circumference r)
  (H2 : 3 * real.pi = base_area r + semicircle_area l)
  : r = 1 := sorry

end find_cone_base_radius_l686_686405


namespace binomial_coefficient_x3y5_in_expansion_l686_686176

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686176


namespace probability_of_heart_and_club_is_zero_l686_686941

noncomputable def deck := ([:string]) -- abstract a card deck as a list of card representations

-- Define the conditions in Lean
def isHeart (c : string) : Prop := c.contains "Heart"
def isClub (c : string) : Prop := c.contains "Club"

def card_probability_zero_heart_and_club (deck : List string) : Prop :=
  ∀ c ∈ deck, ¬ (isHeart c ∧ isClub c)

theorem probability_of_heart_and_club_is_zero : card_probability_zero_heart_and_club ["2Heart", "3Club", ...] :=
by
  sorry

end probability_of_heart_and_club_is_zero_l686_686941


namespace binomial_coefficient_x3y5_in_expansion_l686_686137

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686137


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686142

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686142


namespace num_distinct_units_digits_of_cubes_l686_686472

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686472


namespace cos_double_angle_l686_686393

theorem cos_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : sin α - cos α = sqrt 10 / 5) : cos (2 * α) = -4 / 5 := 
by
  sorry

end cos_double_angle_l686_686393


namespace function_symmetry_about_center_l686_686734

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem function_symmetry_about_center : ∃ c : ℝ, c = Real.pi / 12 ∧ f c = 0 :=
by
  use Real.pi / 12
  split
  · exact rfl
  · show f (Real.pi / 12) = 0
  sorry

end function_symmetry_about_center_l686_686734


namespace sin_double_angle_l686_686990

-- Define the function f
def f (x : ℝ) := Math.sin x - Math.cos x

-- Given the derivative of f
axiom derivative_f : ∀ (x : ℝ), deriv f x = 2 * f x

-- Prove the value of sin(2x)
theorem sin_double_angle (x : ℝ) : Math.sin (2 * x) = 3 / 5 :=
by
  sorry

end sin_double_angle_l686_686990


namespace coefficient_x3_y5_in_binomial_expansion_l686_686097

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686097


namespace prime_numbers_count_and_sum_l686_686757

-- Definition of prime numbers less than or equal to 20
def prime_numbers_leq_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Proposition stating the number of prime numbers and their sum within 20
theorem prime_numbers_count_and_sum :
  (prime_numbers_leq_20.length = 8) ∧ (prime_numbers_leq_20.sum = 77) := by
  sorry

end prime_numbers_count_and_sum_l686_686757


namespace simplify_tan_expression_l686_686692

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end simplify_tan_expression_l686_686692


namespace midpoints_of_inscribed_equal_radii_triangles_l686_686654

theorem midpoints_of_inscribed_equal_radii_triangles
  (A B C A1 B1 C1 : Point)
  (h_eq_triangle : equilateral_triangle A B C)
  (h_A1 : is_on_segment A1 B C)
  (h_B1 : is_on_segment B1 C A)
  (h_C1 : is_on_segment C1 A B)
  (r : ℝ)
  (h_r1 : inscribed_radius C1 A B1 = r)
  (h_r2 : inscribed_radius B1 C A1 = r)
  (h_r3 : inscribed_radius A1 B C1 = r)
  (h_r4 : inscribed_radius A1 B1 C1 = r) :
  midpoint B C A1 ∧ midpoint C A B1 ∧ midpoint A B C1 :=
by
  sorry

end midpoints_of_inscribed_equal_radii_triangles_l686_686654


namespace max_mn_value_l686_686752

noncomputable def vector_max_sum (OA OB : ℝ) (m n : ℝ) : Prop :=
  (OA * OA = 4 ∧ OB * OB = 4 ∧ OA * OB = 2) →
  ((m * OA + n * OB) * (m * OA + n * OB) = 4) →
  (m + n ≤ 2 * Real.sqrt 3 / 3)

-- Here's the statement for the maximum value problem
theorem max_mn_value {m n : ℝ} (h1 : m > 0) (h2 : n > 0) :
  vector_max_sum 2 2 m n :=
sorry

end max_mn_value_l686_686752


namespace total_gallons_of_seed_l686_686887

-- Condition (1): The area of the football field is 8000 square meters.
def area_football_field : ℝ := 8000

-- Condition (2): Each square meter needs 4 times as much seed as fertilizer.
def seed_to_fertilizer_ratio : ℝ := 4

-- Condition (3): Carson uses 240 gallons of seed and fertilizer combined for every 2000 square meters.
def combined_usage_per_2000sqm : ℝ := 240
def area_unit : ℝ := 2000

-- Target: Prove that the total gallons of seed Carson uses for the entire field is 768 gallons.
theorem total_gallons_of_seed : seed_to_fertilizer_ratio * area_football_field / area_unit / (seed_to_fertilizer_ratio + 1) * combined_usage_per_2000sqm * (area_football_field / area_unit) = 768 :=
sorry

end total_gallons_of_seed_l686_686887


namespace simplify_tan_expression_l686_686711

theorem simplify_tan_expression :
  ∀ (tan_15 tan_30 : ℝ), 
  (tan_45 : ℝ) 
  (tan_45_eq_one : tan_45 = 1)
  (tan_sum_formula : tan_45 = (tan_15 + tan_30) / (1 - tan_15 * tan_30)),
  (1 + tan_15) * (1 + tan_30) = 2 :=
by
  intros tan_15 tan_30 tan_45 tan_45_eq_one tan_sum_formula
  sorry

end simplify_tan_expression_l686_686711


namespace distinct_units_digits_of_cube_l686_686488

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686488


namespace distinct_units_digits_of_cubes_l686_686551

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686551


namespace largest_base6_five_digit_l686_686183

def base6_to_base10 : ℕ → ℕ
| n :=
  let digits := [5, 5, 5, 5, 5] in
  digits.zipWith (λ digit idx, digit * (6 ^ idx)) [4, 3, 2, 1, 0]
  |> List.sum

theorem largest_base6_five_digit : base6_to_base10 55555 = 7775 :=
by
  sorry

end largest_base6_five_digit_l686_686183


namespace binomial_coefficient_x3y5_in_expansion_l686_686172

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686172


namespace time_for_pipe_A_l686_686193

-- Definitions based on the given conditions
def pipe_fill_rate_A : ℝ := 1 / t_A -- Rate of pipe A
def pipe_fill_rate_B : ℝ := 2 * pipe_fill_rate_A -- Rate of pipe B (2 times A)
def pipe_fill_rate_C : ℝ := 2 * pipe_fill_rate_B -- Rate of pipe C (2 times B)
def combined_fill_rate : ℝ := pipe_fill_rate_A + pipe_fill_rate_B + pipe_fill_rate_C -- Combined rate
def combined_fill_time : ℝ := 8 -- Time for all pipes to fill the tank together

theorem time_for_pipe_A :
  (combined_fill_rate = 1 / combined_fill_time) →
  (pipe_fill_rate_A = 1 / 56) :=
by
  intro h1
  sorry

end time_for_pipe_A_l686_686193


namespace basketball_team_starters_count_l686_686826

-- Let's define the problem conditions in Lean
def num_players : ℕ := 18
def triplets : Finset ℕ := {1, 2, 3} -- representing Bob, Bill, and Ben
def twins : Finset ℕ := {4, 5}       -- representing Tim and Tom
def required_starters : ℕ := 7
def required_triplets : ℕ := 2
def required_twins : ℕ := 1

-- The proof problem statement
theorem basketball_team_starters_count :
  ∃ (ways_to_choose_starters : ℕ),
    ways_to_choose_starters = (Nat.choose 3 2) * (Nat.choose 2 1) * (Nat.choose 13 4) ∧
    ways_to_choose_starters = 4290 :=
by
  use (Nat.choose 3 2) * (Nat.choose 2 1) * (Nat.choose 13 4)
  split
  · exact rfl
  · norm_num
    exact rfl

end basketball_team_starters_count_l686_686826


namespace exist_polynomial_l686_686308

def P (x : ℤ) : ℤ := (x^3 + 3) * (x^8 - 16)

theorem exist_polynomial :
  (¬ ∃ (r : ℚ), P r = 0) ∧ (∀ n : ℕ, ∃ m : ℤ, nat.gcd n (P m) = n) :=
by
  sorry

end exist_polynomial_l686_686308


namespace cubes_not_painted_l686_686227

def total_unit_cubes (n : ℕ) : ℕ := n * n * n

def painted_unit_cubes : ℕ := 6

def unpainted_unit_cubes (n : ℕ) : ℕ :=
  total_unit_cubes n - painted_unit_cubes

theorem cubes_not_painted (n : ℕ) (h : n = 4) : unpainted_unit_cubes n = 58 :=
by {
  rw [h, unpainted_unit_cubes, total_unit_cubes],
  norm_num,
  sorry,
}

end cubes_not_painted_l686_686227


namespace distinct_units_digits_of_cubes_l686_686538

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686538


namespace installment_payment_difference_l686_686234

-- Definitions based on problem conditions
variables (x : ℝ)
def first_8_payments := 8 * 410
def remaining_payments := 44 * (410 + x)
def total_payments := first_8_payments + remaining_payments
def num_installments := 52
def average_payment := total_payments / num_installments

-- Main theorem statement translating problem to Lean
theorem installment_payment_difference
  (h : average_payment x = 465) : x = 65 := 
sorry

end installment_payment_difference_l686_686234


namespace distinct_units_digits_of_perfect_cube_l686_686516

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686516


namespace sum_of_solutions_l686_686916

theorem sum_of_solutions :
  let satisfies_congruence := λ x : ℕ, (7 * (5 * x - 3)) % 12 = 35 % 12
  ∑ (x : ℕ) in {x : ℕ | x > 0 ∧ x ≤ 50 ∧ satisfies_congruence x}, x = 130 :=
by
  sorry

end sum_of_solutions_l686_686916


namespace simplify_tangent_expression_l686_686703

theorem simplify_tangent_expression :
  let t15 := Real.tan (15 * Real.pi / 180)
  let t30 := Real.tan (30 * Real.pi / 180)
  1 = Real.tan (45 * Real.pi / 180)
  calc
    (1 + t15) * (1 + t30) = 2 := by
  sorry

end simplify_tangent_expression_l686_686703


namespace tan_product_l686_686701

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end tan_product_l686_686701


namespace triangle_area_l686_686762

noncomputable def area_of_triangle (r R cosP cosQ cosR : ℝ) : ℝ :=
(r * R * 2 * cosP) * (r * R * 2 * cosQ) * (r * R * 2 * cosR) -- This represents the area formula in terms of cosines.

theorem triangle_area (h₁ : r = 7)
                      (h₂ : R = 25)
                      (h₃ : 2 * cosQ = cosP + cosR) :
  area_of_triangle r R cosP cosQ cosR = (7 * sqrt 3201) / 3 := 
sorry

end triangle_area_l686_686762


namespace floor_sqrt_eq_l686_686933

theorem floor_sqrt_eq (n : ℕ) (h : n > 0) : 
  (⌊ real.sqrt n + real.sqrt (n + 1) ⌋ = ⌊ real.sqrt (4 * n + 1) ⌋ 
  ∧ ⌊ real.sqrt (4 * n + 1) ⌋ = ⌊ real.sqrt (4 * n + 2) ⌋ 
  ∧ ⌊ real.sqrt (4 * n + 2) ⌋ = ⌊ real.sqrt (4 * n + 3) ⌋) :=
sorry

end floor_sqrt_eq_l686_686933


namespace max_value_of_linear_function_l686_686375

theorem max_value_of_linear_function :
  ∀ (x : ℝ), -3 ≤ x ∧ x ≤ 3 → y = 5 / 3 * x + 2 → ∃ (y_max : ℝ), y_max = 7 ∧ ∀ (x' : ℝ), -3 ≤ x' ∧ x' ≤ 3 → 5 / 3 * x' + 2 ≤ y_max :=
by
  intro x interval_x function_y
  sorry

end max_value_of_linear_function_l686_686375


namespace line_equation_l686_686923

variable (x y : ℝ)
variable (m b : ℝ) (h_m : m = 3) (h_b : b = 4)

theorem line_equation : 3 * x - y + 4 = 0 :=
by
  have h1 : y = m * x + b,
    from sorry,  -- Equation of the line in slope-intercept form
  have h2 : y = 3 * x + 4,
    from sorry,  -- Substituting slope m = 3 and intercept b = 4
  rw [h2] at h1,
  -- Rearranging to general form
  have h3 : 3 * x - y = -4,
    from sorry,
  have h4 : 3 * x - y + 4 = 0,
    from sorry,
  exact h4


end line_equation_l686_686923


namespace distinct_units_digits_of_cubes_l686_686568

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686568


namespace general_term_sum_Tn_l686_686006

open Nat Finset

-- Define initial conditions
noncomputable def a₁ : ℕ := 2
noncomputable def d : ℕ := 1
noncomputable def S₄ : ℕ := 14

-- Prove general formula for the term a_n of the sequence {a_n}
theorem general_term (n : ℕ) : ∃ a : ℕ → ℕ, (a n = n + 1) ∧ (∀ n, 0 < n → a 1 + a 2 + a 3 + a 4 = S₄) := by
  sorry

-- Define sequence {b_n} where b_n = a_{n-1} / 2^n
noncomputable def b (n : ℕ) : ℚ := (a (n - 1) : ℚ) / (2^n)

-- Prove the sum T_n
theorem sum_Tn (n : ℕ) : ∃ T : ℕ → ℚ, (T n = (∑ i in range n, b i) ) ∧ (∀ n, T n = 2 - 1 / (2^(n-1)) - n / 2^n) := by
  sorry

end general_term_sum_Tn_l686_686006


namespace coefficient_x3_y5_in_binomial_expansion_l686_686100

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686100


namespace find_b_l686_686736

noncomputable def f (a b x : ℝ) : ℝ := (1 / 12) * x ^ 2 + a * x + b

theorem find_b (a b : ℝ) (h1 : ∃ x1 x2 : ℝ, f a b x1 = 0 ∧ f a b x2 = 0 ∧ x1 + x2 = 6 ∧ x1 * x2 = 12 * b)
  (h2 : ∃ T : ℝ × ℝ, T = (3, 3) ∧ (let TA := (T.1 - x1)^2 + (T.2 - 0)^2 in
                                    let TB := (T.1 - 0)^2 + (T.2 - b)^2 in
                                    let TC := (T.1 - x2)^2 + (T.2 - 0)^2 in
                                    TA = TB ∧ TB = TC ∧ TA = TC)) : b = -6 :=
begin
  sorry
end

end find_b_l686_686736


namespace coefficient_of_term_in_binomial_expansion_l686_686060

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686060


namespace total_value_correct_l686_686250

noncomputable def total_value : ℝ :=
  let V := 3291.17647 in
  let import_tax := if V > 1000 then 0.10 * (V - 1000) else 0 in
  let transport_fee := if V > 2000 then 0.05 * (V - 2000) else 0 in
  let intermediary_fee := 0.02 * V in
  if (import_tax + transport_fee + intermediary_fee ≈ 359.50) then V else 0

theorem total_value_correct: total_value ≈ 3291.18 :=
by sorry

end total_value_correct_l686_686250


namespace value_of_b_cannot_form_arithmetic_sequence_l686_686008

theorem value_of_b 
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b > 0) :
  b = 5 * Real.sqrt 10 := 
sorry

theorem cannot_form_arithmetic_sequence 
  (d : ℝ)
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b = 5 * Real.sqrt 10) :
  ¬(∃ d, a1 + d = a2 ∧ a2 + d = a3) := 
sorry

end value_of_b_cannot_form_arithmetic_sequence_l686_686008


namespace distinct_units_digits_of_perfect_cube_l686_686515

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686515


namespace num_three_digit_numbers_with_repeats_l686_686766

theorem num_three_digit_numbers_with_repeats : 
  (finset.range 1000).filter (λ n, 99 < n ∧ ∀ d, d ∈ n.digits 10 → d < 10).card = 252 := 
sorry

end num_three_digit_numbers_with_repeats_l686_686766


namespace distinct_units_digits_of_cube_l686_686482

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686482


namespace sum_of_remainders_l686_686188

theorem sum_of_remainders (a b c : ℕ) (h₁ : a % 30 = 15) (h₂ : b % 30 = 7) (h₃ : c % 30 = 18) : 
    (a + b + c) % 30 = 10 := 
by
  sorry

end sum_of_remainders_l686_686188


namespace binomial_coefficient_x3y5_in_expansion_l686_686130

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686130


namespace revenue_from_full_price_tickets_l686_686246

theorem revenue_from_full_price_tickets (f h p : ℝ) (total_tickets : f + h = 160) (total_revenue : f * p + h * (p / 2) = 2400) :
  f * p = 960 :=
sorry

end revenue_from_full_price_tickets_l686_686246


namespace num_distinct_units_digits_of_cubes_l686_686468

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686468


namespace no_even_sum_of_four_consecutive_in_circle_l686_686629

theorem no_even_sum_of_four_consecutive_in_circle (n : ℕ) (h1 : n = 2018) :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ n) ∧ (∀ i, i < n → (f (i % n) + f ((i + 1) % n) + f ((i + 2) % n) + f ((i + 3) % n)) % 2 = 1) :=
by { sorry }

end no_even_sum_of_four_consecutive_in_circle_l686_686629


namespace coefficient_x3y5_in_expansion_l686_686111

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686111


namespace distinct_units_digits_of_integral_cubes_l686_686518

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686518


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686053

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686053


namespace distinct_units_digits_of_cube_l686_686591

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686591


namespace min_mobots_mow_lawn_l686_686626

theorem min_mobots_mow_lawn (n : ℕ) : nat.min_mobots_needed n = n :=
sorry

end min_mobots_mow_lawn_l686_686626


namespace diamond_problem_l686_686362

def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_problem : diamond (diamond 3 4 ^ 2) (diamond (-4) (-3) ^ 2) = 25 * Real.sqrt 2 :=
by
  sorry

end diamond_problem_l686_686362


namespace fruit_basket_total_cost_l686_686831

def banana_price := 1
def apple_price := 2
def strawberry_price_per_12 := 4
def avocado_price := 3
def half_bunch_grapes_price := 2
def orange_price := 1.5
def kiwi_price := 1.25
def melon_price := 3.5

def cost_of_no_discount : ℝ := 
  (4 * banana_price) + 
  (3 * apple_price) + 
  (4 * orange_price) + 
  (2 * kiwi_price) + 
  (24 / 12 * strawberry_price_per_12) + 
  (2 * avocado_price) + 
  (1 * (2 * 2)) + -- one bunch of grapes is two half-bunches
  (1 * melon_price)

def pair_discount : ℝ :=
  (2 * banana_price * 0.05) + 
  (2 * avocado_price * 0.05) + 
  (2 * kiwi_price * 0.05)

def cost_after_pair_discount := cost_of_no_discount - pair_discount

def special_offer_discount := 0.1 * cost_after_pair_discount

def final_cost := cost_after_pair_discount - special_offer_discount

theorem fruit_basket_total_cost (no_discount := cost_of_no_discount)
    (pair_disc := pair_discount)
    (pair_discounted_cost := cost_after_pair_discount)
    (special_disc := special_offer_discount)
    (total_cost := final_cost) : total_cost = 35.43 := by
  sorry

end fruit_basket_total_cost_l686_686831


namespace painted_cube_probability_l686_686311

theorem painted_cube_probability :
  let face_colors := [true, false] in -- true for green, false for yellow
  let n := 6 in -- number of faces
  let arrangements := 2^n in
  let valid_arrangements := 16 in
  let probability := valid_arrangements / arrangements.toRat in
  probability = (1 : ℚ) / 4 :=
by
  -- Proof skipped
  sorry

end painted_cube_probability_l686_686311


namespace distinct_units_digits_perfect_cube_l686_686580

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686580


namespace jerry_probability_cross_4_units_up_l686_686635

theorem jerry_probability_cross_4_units_up :
  let prob := 15 / 64
  in
  ∃ p q : ℕ, Nat.coprime p q ∧ p/q = prob ∧ 
  (∃ heads tails : ℕ, heads + tails = 10 ∧ heads - tails = 4 :=
  sorry

end jerry_probability_cross_4_units_up_l686_686635


namespace find_one_third_of_product_l686_686333

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end find_one_third_of_product_l686_686333


namespace one_third_of_seven_times_nine_l686_686325

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l686_686325


namespace f_is_odd_l686_686372

noncomputable def M_x_n (x : ℝ) (n : ℕ) : ℝ :=
  (List.range n).prod (λ k => x + k)

noncomputable def f (x : ℝ) : ℝ :=
  M_x_n (x - 3) 7 * Real.cos ((2009 : ℝ) / 2010 * x)

theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) :=
  by
    intros x
    have hM : M_x_n (x - 3) 7 = x * (x^2 - 1) * (x^2 - 4) * (x^2 - 9) :=
      sorry -- Use the given condition to prove this, omitted for brevity
    rw [f, f, hM]
    rw [Real.cos_neg]
    simp
    sorry -- Complete the proof using the parity properties

end f_is_odd_l686_686372


namespace volume_ratio_cylinders_l686_686202

open Real

noncomputable def volume_ratio_larger_to_smaller (h₁ : Real) (h₂ : Real) (circumference₁ : Real) (circumference₂ : Real) : Real :=
  let r₁ := circumference₁ / (2 * pi)
  let r₂ := circumference₂ / (2 * pi)
  let V₁ := pi * r₁^2 * h₁
  let V₂ := pi * r₂^2 * h₂
  V₂ / V₁

theorem volume_ratio_cylinders : volume_ratio_larger_to_smaller 9 6 6 9 = 9 / 4 := by
  sorry

end volume_ratio_cylinders_l686_686202


namespace probability_sum_odd_primes_leq_12_l686_686289

theorem probability_sum_odd_primes_leq_12 :
  let primes := [2, 3, 5, 7, 11] in
  let total_combinations := Nat.choose 5 2 in
  let odd_sum_combinations := 4 in
  (odd_sum_combinations / total_combinations : ℚ) = (2 / 5 : ℚ) :=
by
  sorry

end probability_sum_odd_primes_leq_12_l686_686289


namespace tetrahedron_circumradius_l686_686614

theorem tetrahedron_circumradius (A B C D : Type)
  (dist_AB : dist A B = 3) (dist_AC : dist A C = 3)
  (dist_BC : dist B C = 4) (dist_BD : dist B D = 4)
  (perp_BD : perp (line B D) (plane A B C)) :
  ∃ R : ℝ, R = sqrt 805 / 10 :=
by
  sorry

end tetrahedron_circumradius_l686_686614


namespace good_numbers_characterization_l686_686768

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → d + 1 ∣ n + 1

theorem good_numbers_characterization :
  {n : ℕ | is_good n} = {1} ∪ {p | Nat.Prime p ∧ p % 2 = 1} :=
by 
  sorry

end good_numbers_characterization_l686_686768


namespace volunteer_assignment_count_l686_686309

def ways_to_assign_volunteers (A B : Type) (posts : Finset Type) (volunteers : Finset Type) : Nat :=
  @VolunteerAssignment.permutations A B posts volunteers -- assuming permutations can count our specific problem assignment logic

theorem volunteer_assignment_count :
  let A := {A : Volunteers}
  let B := {B : Volunteers}
  let posts := {ShenYuan, QiuZhen, ScienceEducation, WeiMing}
  let volunteers := {Volunteer1, Volunteer2, Volunteer3}

  ways_to_assign_volunteers A B posts volunteers = 72 :=
  sorry

end volunteer_assignment_count_l686_686309


namespace find_value_of_2a_minus_b_l686_686969

variable (a b : ℝ^3)

-- Definitions based on the given conditions
def norm_eq_one (v : ℝ^3) := Real.sqrt (v.dot v) = 1
def angle_sixty_degrees (u v : ℝ^3) := u.dot v = 1 * 1 * (Real.cos (Float.pi / 3))

-- The proof problem statement
theorem find_value_of_2a_minus_b
  (h1 : norm_eq_one a)
  (h2 : norm_eq_one b)
  (h3 : angle_sixty_degrees a b) :
  Real.sqrt ((2 • a - b).dot (2 • a - b)) = Real.sqrt 3 := 
by
  sorry

end find_value_of_2a_minus_b_l686_686969


namespace trigonometric_simplification_l686_686391

-- Define the conditions

variables (a x y : ℝ)
variables (hx : x ∈ set.Icc (-π/4) (π/4))
variables (hy : y ∈ set.Icc (-π/4) (π/4))
variables (h1 : x^3 + sin x = 2 * a)
variables (h2 : 4 * y^3 + sin y * cos y = -a)

-- Define the theorem to prove the question yields the answer
theorem trigonometric_simplification : 3 * sin((π + x) / 2 + y) = 3 :=
by
  sorry

end trigonometric_simplification_l686_686391


namespace angle_C_in_parallelogram_l686_686621

theorem angle_C_in_parallelogram (ABCD : Type) [is_parallelogram ABCD] 
(angle_A angle_B angle_C : ℝ) (h1 : angle_A - angle_B = 20) (h2 : adjacent_angle_sum : ∀ {x y}, adjacent x y → x + y = 180 ) 
(opposite_angle_eq : ∀ {x y}, opposite x y → x = y) : 
angle_C = 100 :=
by
  sorry

end angle_C_in_parallelogram_l686_686621


namespace simplify_tan_expression_l686_686694

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end simplify_tan_expression_l686_686694


namespace distinct_units_digits_of_cube_l686_686483

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686483


namespace distinct_cube_units_digits_l686_686460

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686460


namespace min_rectangle_perimeter_l686_686856

theorem min_rectangle_perimeter (x y : ℤ) (h1 : x * y = 50) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y, x * y = 50 → 2 * (x + y) ≥ 30) ∧ 
  ∃ x y, x * y = 50 ∧ 2 * (x + y) = 30 := 
by sorry

end min_rectangle_perimeter_l686_686856


namespace parabola_ratio_l686_686245

theorem parabola_ratio (p : ℝ) (hp : p > 0) :
  let l : line := { slope := √3 }
  let F : point := (p / 2, 0)
  let parabola : curve := { equation := λ (x y : ℝ), y^2 = 2 * p * x }
  let A : point := first_quadrant_intersection(l, parabola)
  let B : point := fourth_quadrant_intersection(l, parabola)
  ∃ (AF BF : ℝ), (|AF| / |BF| = 3)
:= 
sorry

end parabola_ratio_l686_686245


namespace constant_polynomial_divides_power_of_two_sub_one_l686_686316

open Polynomial

theorem constant_polynomial_divides_power_of_two_sub_one
    (f : Polynomial ℤ)
    (h : ∀ n : ℕ, 0 < n → f.eval n ∣ 2^n - 1) :
    f = Polynomial.C 1 ∨ f = Polynomial.C (-1) :=
by sorry

end constant_polynomial_divides_power_of_two_sub_one_l686_686316


namespace expression_value_l686_686773

theorem expression_value : 5 - (3: ℚ)^(-3) = 134 / 27 := 
by
  -- Lean proof goes here.
  sorry

end expression_value_l686_686773


namespace no_root_zero_l686_686915

theorem no_root_zero : 
  (∀ x : ℝ, 5 * x^2 - 3 = 50 → x ≠ 0) ∧
  (∀ x : ℝ, (3 * x - 1)^2 = (x - 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, sqrt (x^2 - 9) = sqrt (2 * x - 2) → x ≠ 0) :=
by
  sorry

end no_root_zero_l686_686915


namespace choose_seven_sum_multiple_of_three_l686_686195

theorem choose_seven_sum_multiple_of_three :
  (finset.univ.filter (λ (s: finset ℕ), s.sum % 3 = 0 ∧ s.card = 7)).card = 12 := 
sorry

end choose_seven_sum_multiple_of_three_l686_686195


namespace circumcircle_tangent_to_line_l686_686868

-- Define the points and the circumcircle on the Lean level.
variable (A B C D E : Type)
variable [trapezoid : Trapezoid A B C D] (h_base : base A B base C D)
variable (circle_BCD : Circumcircle B C D E) (h_intersect : intersect B C D E A D E)

-- Main theorem statement
theorem circumcircle_tangent_to_line :
  tangent (Circumcircle A B E) (line B C) :=
sorry

end circumcircle_tangent_to_line_l686_686868


namespace volume_ratio_l686_686218

def rect_height_vert : ℝ := 9
def rect_circumference_vert : ℝ := 6
def radius_vert : ℝ := rect_circumference_vert / (2 * Real.pi)
def volume_vert : ℝ := Real.pi * radius_vert^2 * rect_height_vert
-- volume_vert is calculated as πr^2h where r = 3/π

def rect_height_horiz : ℝ := 6
def rect_circumference_horiz : ℝ := 9
def radius_horiz : ℝ := rect_circumference_horiz / (2 * Real.pi)
def volume_horiz : ℝ := Real.pi * radius_horiz^2 * rect_height_horiz
-- volume_horiz is calculated as πr^2h where r = 9/(2π)

theorem volume_ratio : (max volume_vert volume_horiz) / (min volume_vert volume_horiz) = 3 / 4 := 
by sorry

end volume_ratio_l686_686218


namespace length_of_circle_l686_686268

-- Define initial speeds and conditions
variables (V1 V2 : ℝ)
variables (L : ℝ) -- Length of the circle

-- Conditions
def initial_condition : Prop := V1 - V2 = 3
def extra_laps_after_speed_increase : Prop := (V1 + 10) - V2 = V1 - V2 + 10

-- Statement representing the mathematical equivalence
theorem length_of_circle
  (h1 : initial_condition V1 V2) 
  (h2 : extra_laps_after_speed_increase V1 V2) :
  L = 1250 := 
sorry

end length_of_circle_l686_686268


namespace parallelogram_area_and_unit_vector_l686_686286

open Matrix

-- Define the vectors
def u : ℝ^3 := ![2, 4, -3]
def v : ℝ^3 := ![-1, 5, 2]

-- Define the cross product function
def cross_product (a b : ℝ^3) : ℝ^3 :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

-- Define the magnitude (norm) function
def magnitude (w : ℝ^3) : ℝ :=
  Real.sqrt (w 0 ^ 2 + w 1 ^ 2 + w 2 ^ 2)

-- Define the unit vector function
def unit_vector (w : ℝ^3) : ℝ^3 :=
  let norm := magnitude w
  in ![w 0 / norm, w 1 / norm, w 2 / norm]

-- Statement of the proof problem
theorem parallelogram_area_and_unit_vector :
  let cp := cross_product u v in
  magnitude cp = Real.sqrt 726 ∧
  unit_vector cp = ![23 / Real.sqrt 726, -1 / Real.sqrt 726, 14 / Real.sqrt 726] :=
by
  sorry

end parallelogram_area_and_unit_vector_l686_686286


namespace smallest_n_for_contiguous_subsequence_sum_l686_686927

open Nat

theorem smallest_n_for_contiguous_subsequence_sum (n : ℕ) : 
  (∀ (a : Fin n → ℕ), (∀ i, a i > 0) → ∑ i, a i = 2007 → (∃ (i j : ℕ), i ≤ j ∧ ∑ k in Finset.range (j+1) \ Finset.range i, a k = 30)) ↔ n = 1018 :=
by
  sorry

end smallest_n_for_contiguous_subsequence_sum_l686_686927


namespace coeff_x3y5_in_expansion_l686_686026

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686026


namespace distinct_units_digits_of_cubes_l686_686548

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686548


namespace smallest_reciprocal_among_given_set_l686_686189

noncomputable def reciprocal (x : ℝ) : ℝ := 1 / x

theorem smallest_reciprocal_among_given_set :
  let S := { (1 / 4 : ℝ), (3 / 7 : ℝ), (-2 : ℝ), (10 : ℝ), (2023 : ℝ) } in
  ∀ x ∈ S, reciprocal (-2) < reciprocal x :=
by {
  sorry
}

end smallest_reciprocal_among_given_set_l686_686189


namespace distinct_units_digits_perfect_cube_l686_686576

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686576


namespace volume_of_soil_removal_l686_686257

theorem volume_of_soil_removal {a b m c d : ℝ} :
  (∃ (K : ℝ), K = (m / 6) * (2 * a * c + 2 * b * d + a * d + b * c)) :=
sorry

end volume_of_soil_removal_l686_686257


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686073

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686073


namespace prove_op_eq_l686_686905

-- Define the new operation ⊕
def op (x y : ℝ) := x^3 - 2*y + x

-- State that for any k, k ⊕ (k ⊕ k) = -k^3 + 3k
theorem prove_op_eq (k : ℝ) : op k (op k k) = -k^3 + 3*k :=
by 
  sorry

end prove_op_eq_l686_686905


namespace distinct_units_digits_of_perfect_cubes_l686_686446

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686446


namespace nina_pairs_with_elle_probability_l686_686612

noncomputable def total_students : ℕ := 32
noncomputable def elle_restrictions : ℕ := 2
noncomputable def nina_potential_partners : ℕ := total_students - 1 - elle_restrictions

def probability_nina_pairs_with_elle : ℚ :=
  if nina_potential_partners > 0 then 1 / nina_potential_partners else 0

theorem nina_pairs_with_elle_probability : probability_nina_pairs_with_elle = 1 / 29 :=
by
  -- sorry is used to skip the proof
  sorry

end nina_pairs_with_elle_probability_l686_686612


namespace distinct_units_digits_of_perfect_cubes_l686_686445

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686445


namespace sum_first_60_nat_l686_686931

theorem sum_first_60_nat : (∑ i in Finset.range (60 + 1), i) = 1830 := by
  sorry

end sum_first_60_nat_l686_686931


namespace coefficient_x3y5_in_expansion_l686_686154

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686154


namespace binomial_coefficient_x3y5_in_expansion_l686_686132

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686132


namespace coefficient_of_term_in_binomial_expansion_l686_686059

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686059


namespace binomial_coefficient_x3y5_in_expansion_l686_686175

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686175


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686151

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686151


namespace distinct_units_digits_of_cubes_l686_686545

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686545


namespace area_of_triangle_ABC_l686_686761

open Real

-- Definitions
variables {A B C D : Type}
variables [inhabited A] [inhabited B] [inhabited C]

-- Triangle with right angle at B
structure Triangle (A B C : Type) :=
(right_angle_at_B : B = A)
(point_D_is_foot_of_altitude : D)

-- Given conditions
variables (AD DC : ℝ)
variables (AD_pos : AD = 3)
variables (DC_pos : DC = 4)

noncomputable def triangle_area (AD DC : ℝ) (hAD : AD = 3) (hDC : DC = 4) : ℝ :=
  let BD := Real.sqrt (AD * DC) in
  let AC := AD + DC in
  1 / 2 * AC * BD

theorem area_of_triangle_ABC :
  triangle_area 3 4 AD_pos DC_pos = 7 * Real.sqrt 3 := by
  sorry

end area_of_triangle_ABC_l686_686761


namespace distinct_units_digits_of_cubes_l686_686550

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686550


namespace Oliver_has_9_dollars_left_l686_686672

def initial_amount := 9
def saved := 5
def earned := 6
def spent_frisbee := 4
def spent_puzzle := 3
def spent_stickers := 2
def spent_movie_ticket := 7
def spent_snack := 3
def gift := 8

def final_amount (initial_amount : ℕ) (saved : ℕ) (earned : ℕ) (spent_frisbee : ℕ)
                 (spent_puzzle : ℕ) (spent_stickers : ℕ) (spent_movie_ticket : ℕ)
                 (spent_snack : ℕ) (gift : ℕ) : ℕ :=
  initial_amount + saved + earned - spent_frisbee - spent_puzzle - spent_stickers - 
  spent_movie_ticket - spent_snack + gift

theorem Oliver_has_9_dollars_left :
  final_amount initial_amount saved earned spent_frisbee 
               spent_puzzle spent_stickers spent_movie_ticket 
               spent_snack gift = 9 :=
  by
  sorry

end Oliver_has_9_dollars_left_l686_686672


namespace find_PQ_length_l686_686003

-- Define the lengths of the sides of the triangles and the angle
def PQ_length : ℝ := 9
def QR_length : ℝ := 20
def PR_length : ℝ := 15
def ST_length : ℝ := 4.5
def TU_length : ℝ := 7.5
def SU_length : ℝ := 15
def angle_PQR : ℝ := 135
def angle_STU : ℝ := 135

-- Define the similarity condition
def triangles_similar (PQ QR PR ST TU SU angle_PQR angle_STU : ℝ) : Prop :=
  angle_PQR = angle_STU ∧ PQ / QR = ST / TU

-- Theorem statement
theorem find_PQ_length (PQ QR PR ST TU SU angle_PQR angle_STU: ℝ) 
  (H : triangles_similar PQ QR PR ST TU SU angle_PQR angle_STU) : PQ = 20 :=
by
  sorry

end find_PQ_length_l686_686003


namespace max_value_of_a_l686_686995

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem max_value_of_a (a b c d : ℝ) (h_deriv_bounds : ∀ x, 0 ≤ x → x ≤ 1 → abs (3 * a * x^2 + 2 * b * x + c) ≤ 1) (h_a_nonzero : a ≠ 0) :
  a ≤ 8 / 3 :=
sorry

end max_value_of_a_l686_686995


namespace reflected_ray_eqn_l686_686855

theorem reflected_ray_eqn :
  ∀ (P Q P' : ℝ × ℝ),
  P = (5, 3) →
  Q = (2, 0) →
  P' = (5, -3) →
  ∃ (a b c : ℝ), a = 1 ∧ b = 1 ∧ c = -2 ∧ (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ (∃ (k : ℝ), x = k * (5 - 2) + 2 ∧ y = k * (-3 - 0))) :=
begin
  intros P Q P',
  sorry
end

end reflected_ray_eqn_l686_686855


namespace distinct_units_digits_of_cubes_l686_686549

theorem distinct_units_digits_of_cubes : 
  ∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.card = 10 → 
    ∃ n : ℤ, (n ^ 3 % 10 = d) :=
by {
  intros d hd,
  fin_cases d;
  -- Manually enumerate cases for units digits of cubes
  any_goals {
    use d,
    norm_num,
    sorry
  }
}

end distinct_units_digits_of_cubes_l686_686549


namespace suff_but_not_necc_l686_686963

def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := (x - 2) * (x + 3) = 0

theorem suff_but_not_necc (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end suff_but_not_necc_l686_686963


namespace sequence_a_n_arithmetic_sum_inequality_l686_686955

noncomputable def sequence_a_n (n : ℕ) : ℕ := 3 * n - 1

noncomputable def sum_S_n (n : ℕ) : ℕ := (finset.range n).sum sequence_a_n

theorem sequence_a_n_arithmetic (n : ℕ) : 6 * sum_S_n n = (sequence_a_n n + 1) * (sequence_a_n n + 2) :=
  sorry

theorem sum_inequality (n : ℕ) : (finset.range n).sum (λ k, 1 / (sequence_a_n k * sequence_a_n (k+1))) < 1 / 6 :=
  sorry

end sequence_a_n_arithmetic_sum_inequality_l686_686955


namespace range_of_f_l686_686352

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_f :
  ∀ y : ℝ, (y ≠ -27) ↔ (∃ x : ℝ, x ≠ -5 ∧ f x = y) :=
by
  intro y
  split
  · intro hy
    use (y / 3 + 4)
    split
    · intro h
      contradiction
    · simp [f, hy]
  · intro ⟨x, hx1, hx2⟩
    rw [←hx2]
    intro h
    contradiction

end range_of_f_l686_686352


namespace max_triangle_area_in_grid_l686_686678

theorem max_triangle_area_in_grid
  (A B C D : ℤ × ℤ)
  (hA : A = (0, 0))
  (hB : B = (10, 0))
  (hC : C = (10, 10))
  (hD : D = (0, 10))
  (move : ℤ × ℤ → ℤ × ℤ → Prop)
  (h_move_adjacent : ∀ p q, move p q → |p.1 - q.1| + |p.2 - q.2| = 1)
  (h_never_leave_square : ∀ p q, move p q → p.1 ≥ 0 ∧ p.1 ≤ 10 ∧ p.2 ≥ 0 ∧ p.2 ≤ 10)
  (positions : list (ℤ × ℤ))
  (initial_positions : positions.head = [A, B, C])
  (final_positions : positions.last = some [B, A, D])
  (h_moves : ∀ i ∈ list.range (list.length positions - 1), move (positions.nth_le i sorry).1 (positions.nth_le (i+1) sorry).1)
  (h_min_area : ∀ p1 p2 p3 ∈ positions, ∃ A B C, (p1 = A ∨ p1 = B ∨ p1 = C) ∧ (p2 = A ∨ p2 = B ∨ p2 = C) ∧ (p3 = A ∨ p3 = B ∨ p3 = C) →
    triangle_area p1 p2 p3 ≥ t)
  (t : ℝ) :
  t ≤ 2.5 :=
by sorry

end max_triangle_area_in_grid_l686_686678


namespace ferris_wheel_rides_l686_686823

theorem ferris_wheel_rides :
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  total_people = 1260 :=
by
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  have : total_people = 1260 := by sorry
  exact this

end ferris_wheel_rides_l686_686823


namespace distinct_units_digits_of_cube_l686_686490

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686490


namespace quadratic_roots_condition_l686_686939

theorem quadratic_roots_condition (a : ℝ) :
  (∃ α : ℝ, 5 * α = -(a - 4) ∧ 4 * α^2 = a - 5) ↔ (a = 7 ∨ a = 5) :=
by
  sorry

end quadratic_roots_condition_l686_686939


namespace ratio_of_volumes_l686_686213

theorem ratio_of_volumes (h1 : ∃ r : ℝ, 2 * Real.pi * r = 6) (h2 : ∃ r : ℝ, 2 * Real.pi * r = 9) :
  (let r1 := Classical.some h1, V1 := Real.pi * r1^2 * 9,
       r2 := Classical.some h2, V2 := Real.pi * r2^2 * 6
   in V2 / V1 = 3 / 2) :=
by
  sorry

end ratio_of_volumes_l686_686213


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686041

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686041


namespace distinct_units_digits_of_perfect_cubes_l686_686440

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686440


namespace distinct_units_digits_of_cube_l686_686489

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686489


namespace profit_percent_correct_l686_686857

def purchase_price : ℝ := 225
def overhead_expenses : ℝ := 15
def selling_price : ℝ := 350

def cost_price : ℝ := purchase_price + overhead_expenses
def profit : ℝ := selling_price - cost_price
def profit_percent : ℝ := (profit / cost_price) * 100

theorem profit_percent_correct : profit_percent = 45.83 := by
  sorry

end profit_percent_correct_l686_686857


namespace max_ab_l686_686390

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 40) : 
  ab ≤ 400 :=
sorry

end max_ab_l686_686390


namespace sequence_properties_l686_686957

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Assume initial conditions about the sequences
def sequence_conditions : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, S n = (∑ i in finset.range n, a i + 1)) ∧ 
  (∀ n, 4 * S n = a n ^ 2 + 2 * a n) ∧
  (∀ n, b n = 3 * 2 ^ a n / ((2 ^ a n - 1) * (2 ^ a (n + 1) - 1))) ∧
  (∀ n, T n = ∑ i in finset.range n, b i)

theorem sequence_properties (h : sequence_conditions a S b T) :
  (∀ n, a n = 2 * n) ∧ (∀ n, T n < 1 / 3) :=
sorry

end sequence_properties_l686_686957


namespace distinct_units_digits_of_cube_l686_686584

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686584


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686052

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686052


namespace ratio_of_volumes_l686_686212

theorem ratio_of_volumes (h1 : ∃ r : ℝ, 2 * Real.pi * r = 6) (h2 : ∃ r : ℝ, 2 * Real.pi * r = 9) :
  (let r1 := Classical.some h1, V1 := Real.pi * r1^2 * 9,
       r2 := Classical.some h2, V2 := Real.pi * r2^2 * 6
   in V2 / V1 = 3 / 2) :=
by
  sorry

end ratio_of_volumes_l686_686212


namespace units_digit_of_perfect_cube_l686_686501

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686501


namespace distinct_units_digits_of_cubes_l686_686558

theorem distinct_units_digits_of_cubes :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let units_digits := {d^3 % 10 | d in digits} in
  units_digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := 
by {
  sorry
}

end distinct_units_digits_of_cubes_l686_686558


namespace café_cost_l686_686836

/-- Define a structure for the café problem with each kind of item and their corresponding cost. -/
structure CaféPrice (s c p : ℝ) where
  sandwich_price : ℝ := s
  coffee_price : ℝ := c
  pie_price : ℝ := p

/-- Given the following conditions determine the total cost:
    - 4 sandwiches, 9 cups of coffee, 1 piece of pie = $4.30
    - 7 sandwiches, 14 cups of coffee, 1 piece of pie = $7.00
    Prove that the total cost for 11 sandwiches, 23 cups of coffee, and 2 pieces of pie is $18.87 
-/
theorem café_cost
  (s c p : ℝ)
  (cond1 : 4 * s + 9 * c + p = 4.30)
  (cond2 : 7 * s + 14 * c + p = 7.00) :
  11 * s + 23 * c + 2 * p = 18.87 :=
by
  sorry

end café_cost_l686_686836


namespace f_neg_one_value_l686_686972

theorem f_neg_one_value (f : ℝ → ℝ) (b : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x → f x = 2^x + 2 * x + b) :
  f (-1) = -3 := by
sorry

end f_neg_one_value_l686_686972


namespace coeff_x3y5_in_expansion_l686_686032

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686032


namespace coefficient_x3y5_in_expansion_l686_686158

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686158


namespace simplify_evaluate_expression_l686_686712

noncomputable def a : ℝ := 2 * Real.cos (60 * Real.pi / 180) + 1

theorem simplify_evaluate_expression : (a - (a^2) / (a + 1)) / ((a^2) / ((a^2) - 1)) = 1 / 2 :=
by sorry

end simplify_evaluate_expression_l686_686712


namespace positional_relationship_l686_686946

-- Let l be a line
variable (l : Line)

-- Let α and β be two different planes
variable (α β : Plane)

-- Given that α and β are parallel planes
axiom alpha_parallel_beta : Parallel α β

-- Given that l is parallel to α
axiom l_parallel_alpha : Parallel l α

-- Prove that l is parallel to β or l is contained in β
theorem positional_relationship :
  Parallel l β ∨ Subset l β :=
sorry

end positional_relationship_l686_686946


namespace museum_ticket_cost_l686_686816

theorem museum_ticket_cost 
  (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_cost : ℕ) (teacher_ticket_cost : ℕ)
  (h_students : num_students = 12) (h_teachers : num_teachers = 4)
  (h_student_cost : student_ticket_cost = 1) (h_teacher_cost : teacher_ticket_cost = 3) :
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end museum_ticket_cost_l686_686816


namespace x_plus_y_value_l686_686194

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem x_plus_y_value :
  let x := sum_of_integers 50 70
  let y := count_even_integers 50 70
  x + y = 1271 := by
    let x := sum_of_integers 50 70
    let y := count_even_integers 50 70
    sorry

end x_plus_y_value_l686_686194


namespace general_term_l686_686377

noncomputable def F (n : ℕ) : ℝ :=
  1 / (Real.sqrt 5) * (((1 + Real.sqrt 5) / 2)^(n-2) - ((1 - Real.sqrt 5) / 2)^(n-2))

noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 5
| n+2 => a (n+1) * a n / Real.sqrt ((a (n+1))^2 + (a n)^2 + 1)

theorem general_term (n : ℕ) :
  a n = (2^(F (n+2)) * 13^(F (n+1)) * 5^(-2 * F (n+1)) - 1)^(1/2) := sorry

end general_term_l686_686377


namespace num_distinct_units_digits_of_cubes_l686_686469

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686469


namespace ac_equals_af_l686_686645

-- Define all the given entities and conditions in Lean
variables 
  (Γ₁ Γ₂ : Circle)
  (A D : Point) (h_intersect : A ≠ D)
  (B C E F : Point) (Ω : Circle)
  (h1 : Γ₁ ∩ Γ₂ = {A, D})
  (h2 : Tangent Γ₁ A B)
  (h3 : Tangent Γ₂ A C)
  (h4 : E ∈ Ray A B)
  (h5 : dist B E = dist A B)
  (h6 : F ∈ Ray A C)
  (h7 : circumcircle Ω ADE = Ω)
  (h8 : F ≠ A)
  (h9 : on_circle F Ω)
  (h10 : second_intersection_line ⟨A, C⟩ Ω = F)

-- Lean statement to show that AC = AF
theorem ac_equals_af : dist A C = dist A F := sorry

end ac_equals_af_l686_686645


namespace distinct_units_digits_of_cubes_l686_686536

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686536


namespace ratio_of_cylinder_volumes_l686_686210

theorem ratio_of_cylinder_volumes (h l : ℝ) (h_h : h = 6) (h_l : l = 9) :
  let r_C := h / (2 * Real.pi),
      V_C := Real.pi * r_C^2 * l,
      r_D := l / (2 * Real.pi),
      V_D := Real.pi * r_D^2 * h in
  if V_D > V_C then V_D / V_C = 3 / 4 else V_C / V_D = 3 / 4 := by
  sorry

end ratio_of_cylinder_volumes_l686_686210


namespace joe_lowest_test_score_dropped_l686_686801

theorem joe_lowest_test_score_dropped 
  (A B C D : ℝ) 
  (h1 : A + B + C + D = 360) 
  (h2 : A + B + C = 255) :
  D = 105 :=
sorry

end joe_lowest_test_score_dropped_l686_686801


namespace distinct_units_digits_of_cubes_l686_686534

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686534


namespace coeff_x3y5_in_expansion_l686_686035

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686035


namespace coefficient_x3y5_in_expansion_l686_686153

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686153


namespace coeff_x3y5_in_expansion_l686_686027

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686027


namespace imaginary_part_of_z_l686_686979

def z : ℂ := (1 - I) / (1 + 3 * I)

theorem imaginary_part_of_z : z.im = -2 / 5 := by
  sorry

end imaginary_part_of_z_l686_686979


namespace coefficient_x3y5_in_expansion_l686_686162

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686162


namespace range_of_a_l686_686660

def p (a : ℝ) : Prop := (a + 2) > 1
def q (a : ℝ) : Prop := (4 - 4 * a) ≥ 0
def prop_and (a : ℝ) : Prop := p a ∧ q a
def prop_or (a : ℝ) : Prop := p a ∨ q a
def valid_a (a : ℝ) : Prop := (a ∈ Set.Iic (-1)) ∨ (a ∈ Set.Ioi 1)

theorem range_of_a (a : ℝ) (h_and : ¬ prop_and a) (h_or : prop_or a) : valid_a a := 
sorry

end range_of_a_l686_686660


namespace slope_of_line_l686_686012

-- Definitions of the conditions in the problem
def line_eq (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

def y_intercept (l : ℝ → ℝ → Prop) (b : ℝ) : Prop :=
  l 0 b

-- The statement of the proof problem
theorem slope_of_line (a : ℝ) (h : y_intercept (line_eq a) (-2)) : 
  ∃ (m : ℝ), m = -2 :=
sorry

end slope_of_line_l686_686012


namespace one_third_of_seven_times_nine_l686_686326

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l686_686326


namespace distinct_units_digits_of_cubes_l686_686439

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686439


namespace distinct_cube_units_digits_l686_686453

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686453


namespace system_of_equations_solution_l686_686749

theorem system_of_equations_solution :
  ∃ x y z : ℝ, x + y = 1 ∧ y + z = 2 ∧ z + x = 3 ∧ x = 1 ∧ y = 0 ∧ z = 2 :=
by
  sorry

end system_of_equations_solution_l686_686749


namespace coins_from_brother_l686_686638

-- Defining the conditions as variables
variables (piggy_bank_coins : ℕ) (father_coins : ℕ) (given_to_Laura : ℕ) (left_coins : ℕ)

-- Setting the conditions
def conditions : Prop :=
  piggy_bank_coins = 15 ∧
  father_coins = 8 ∧
  given_to_Laura = 21 ∧
  left_coins = 15

-- The main theorem statement
theorem coins_from_brother (B : ℕ) :
  conditions piggy_bank_coins father_coins given_to_Laura left_coins →
  piggy_bank_coins + B + father_coins - given_to_Laura = left_coins →
  B = 13 :=
by
  sorry

end coins_from_brother_l686_686638


namespace stormi_needs_more_money_to_afford_bicycle_l686_686722

-- Definitions from conditions
def money_washed_cars : ℕ := 3 * 10
def money_mowed_lawns : ℕ := 2 * 13
def bicycle_cost : ℕ := 80
def total_earnings : ℕ := money_washed_cars + money_mowed_lawns

-- The goal to prove 
theorem stormi_needs_more_money_to_afford_bicycle :
  (bicycle_cost - total_earnings) = 24 := by
  sorry

end stormi_needs_more_money_to_afford_bicycle_l686_686722


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686082

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686082


namespace yanni_money_left_in_cents_l686_686790

-- Conditions
def initial_money : ℝ := 0.85
def money_from_mother : ℝ := 0.40
def money_found : ℝ := 0.50
def cost_per_toy : ℝ := 1.60
def number_of_toys : ℕ := 3
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Prove
theorem yanni_money_left_in_cents : 
  (initial_money + money_from_mother + money_found) * 100 = 175 :=
by
  sorry

end yanni_money_left_in_cents_l686_686790


namespace height_of_prism_l686_686684

def right_triangleXYZ (XYZ : Type) [inhabited XYZ] (XY XZ : XYZ → ℝ) (h : 0 < √9) : Prop :=
  XY = XZ ∧ XZ = 3 ∧ (X : XYZ) = 4.5

theorem height_of_prism (h_prism : right_triangleXYZ XYZ) (V : ℝ) (hV : V = 27) : height_of_prism = 6 :=
by
  sorry

end height_of_prism_l686_686684


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686089

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686089


namespace fish_count_on_may_1_l686_686230

noncomputable def number_of_fish_may_1 : ℕ :=
  let marked_on_may_1 := 60
  let caught_on_sep_1 := 70
  let marked_among_caught_sep_1 := 3
  let survival_rate := 0.75
  let new_fish_rate := 0.40
  let remaining_fish_rate := 0.60
  let fraction_marked := marked_among_caught_sep_1 / caught_on_sep_1
  
  -- Calculate the total number of fish present on Sep 1
  let total_sep_1 := (marked_on_may_1 * survival_rate) / fraction_marked
  
  -- Calculate the number of fish that were there since May 1
  let fish_from_may_1_on_sep_1 := total_sep_1 * remaining_fish_rate
  
  -- Calculate the total number of fish on May 1
  let fish_may_1 := fish_from_may_1_on_sep_1 / survival_rate
  
  -- Return the number of fish on May 1
  fish_may_1

theorem fish_count_on_may_1 : number_of_fish_may_1 = 840 := by sorry

end fish_count_on_may_1_l686_686230


namespace sqrt_of_nine_l686_686011

-- Definition of the principal square root function as used in the conditions.
def principal_sqrt (x : ℝ) : ℝ :=
if x >= 0 then real.sqrt x else 0  -- For real numbers, sqrt only applies for non-negatives

-- Statement of the theorem to be proved
theorem sqrt_of_nine : principal_sqrt 9 = 3 :=
by
  sorry

end sqrt_of_nine_l686_686011


namespace simplify_tan_expression_l686_686687

theorem simplify_tan_expression :
  (1 + Real.tan (15 * Real.pi / 180)) * (1 + Real.tan (30 * Real.pi / 180)) = 2 :=
by
  have h1 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h2 : Real.tan (15 * Real.pi / 180 + 30 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h3 : 15 * Real.pi / 180 + 30 * Real.pi / 180 = 45 * Real.pi / 180 := by sorry
  have h4 : Real.tan (45 * Real.pi / 180) = 
            (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) / 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  have h5 : (Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180)) = 
            (1 - Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180)) := by sorry
  sorry

end simplify_tan_expression_l686_686687


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686094

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686094


namespace num_distinct_units_digits_of_cubes_l686_686470

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686470


namespace coefficient_of_term_in_binomial_expansion_l686_686057

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l686_686057


namespace distinct_units_digits_of_integral_cubes_l686_686522

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686522


namespace right_triangle_hypotenuse_l686_686740

theorem right_triangle_hypotenuse :
  let a := log 8 125
  let b := log 2 25
  ∃ h : ℝ, (hypotenuse a b = h) → 8^h = 5^(3 * sqrt 5) :=
by
  let a := log 8 125
  let b := log 2 25
  use hypotenuse a b
  intros h hypotenuse_eq
  rw ← hypotenuse_eq
  sorry

end right_triangle_hypotenuse_l686_686740


namespace points_collinear_l686_686383

variable {A B C H P Q : Point}
variable (triangle_ABC : Triangle A B C)
variable (acute_triangle : triangle_ABC.isAcute)
variable (H_is_orthocenter : H = triangle_ABC.orthocenter)
variable (circle_diameter_BC : Circle (diameter B C))
variable (tangent_AP_P : is_tangent AP circle_diameter_BC P)
variable (tangent_AQ_Q : is_tangent AQ circle_diameter_BC Q)

theorem points_collinear (h1 : H_is_orthocenter) 
                        (h2 : tangent_AP_P) 
                        (h3 : tangent_AQ_Q) : 
                        collinear P H Q := 
sorry

end points_collinear_l686_686383


namespace parabola_focus_l686_686341

-- Define the given parabola y = 2x^2 + 14x + 1
def parabola (x : ℝ) : ℝ := 2 * x^2 + 14 * x + 1

-- Define the conditions for the focus of the parabola
theorem parabola_focus : ∃ (h k : ℝ), h = -3.5 ∧ k = -23.5 ∧ focus parabola = (h, k + 1 / (4 * 2)) :=
begin
  sorry
end

end parabola_focus_l686_686341


namespace lines_parallel_l686_686961

theorem lines_parallel (a : ℝ) : let l1 := (3 + a) * x + 4 * y = 5 - 3 * a,
                                   l2 := 2 * x + (5 + a) * y = 8
                                   in
                                   (a = -5) ↔ l1.parallel l2 :=
by
  sorry

end lines_parallel_l686_686961


namespace distinct_units_digits_of_integral_cubes_l686_686520

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686520


namespace arithmetic_sequence_max_sum_l686_686958

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a (n+1) = a 1 + n * d)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h3 : a 1 + a 5 + a 9 = 6)
  (h4 : S 11 = -11) :
  ∃ n, S n = ∑ i in finset.range n, a (i+1) ∧ n = 5 :=
sorry

end arithmetic_sequence_max_sum_l686_686958


namespace distinct_units_digits_of_integral_cubes_l686_686529

theorem distinct_units_digits_of_integral_cubes :
  (∃ S : Finset ℕ, S = {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10} ∧ S.card = 10) :=
by
  let S := {0^3 % 10, 1^3 % 10, 2^3 % 10, 3^3 % 10, 4^3 % 10, 5^3 % 10, 6^3 % 10, 7^3 % 10, 8^3 % 10, 9^3 % 10}
  have h : S.card = 10 := sorry
  exact ⟨S, rfl, h⟩

end distinct_units_digits_of_integral_cubes_l686_686529


namespace triangle_inscribed_and_arcs_l686_686810

theorem triangle_inscribed_and_arcs
  (PQ QR PR : ℝ) (X Y Z : ℝ)
  (QY XZ QX YZ PX RY : ℝ)
  (H1 : PQ = 26)
  (H2 : QR = 28) 
  (H3 : PR = 27)
  (H4 : QY = XZ)
  (H5 : QX = YZ)
  (H6 : PX = RY)
  (H7 : RY = PX + 1)
  (H8 : XZ = QX + 1)
  (H9 : QY = YZ + 2) :
  QX = 29 / 2 :=
by
  sorry

end triangle_inscribed_and_arcs_l686_686810


namespace units_digit_of_perfect_cube_l686_686497

theorem units_digit_of_perfect_cube :
  ∃ (S : Finset (Fin 10)), S.card = 10 ∧ ∀ n : ℕ, (n^3 % 10) ∈ S := by
  sorry

end units_digit_of_perfect_cube_l686_686497


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686068

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686068


namespace largest_n_value_l686_686769

theorem largest_n_value (n : ℕ) (h : (1 / 5 : ℝ) + (n / 8 : ℝ) + 1 < 2) : n ≤ 6 :=
by
  sorry

end largest_n_value_l686_686769


namespace distinct_cube_units_digits_l686_686459

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686459


namespace product_of_intersection_coordinates_l686_686185

theorem product_of_intersection_coordinates :
  let circle1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 4)^2 = 4}
  let circle2 := {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 4)^2 = 9}
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 ∧ p.1 * p.2 = 16 :=
by
  sorry

end product_of_intersection_coordinates_l686_686185


namespace mode_median_l686_686860

theorem mode_median (x : ℕ) (h_mode : mode_of_set ({1, 4, 3, 2, 5, x} : set ℕ) = 3) : 
  median_of_set ({1, 4, 3, 2, 5, x} : set ℕ) = 3 := 
by
  sorry

noncomputable def mode_of_set (s : set ℕ) : ℕ := sorry
noncomputable def median_of_set (s : set ℕ) : ℕ := sorry

end mode_median_l686_686860


namespace coefficient_x3_y5_in_binomial_expansion_l686_686096

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686096


namespace num_distinct_units_digits_of_cubes_l686_686478

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686478


namespace distinct_units_digits_of_cube_l686_686583

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686583


namespace chess_tournament_games_l686_686199

theorem chess_tournament_games (n : ℕ) (h : n = 7) : 2 * (n * (n - 1)) = 84 :=
by 
  rw [h]
  simp
  norm_num

end chess_tournament_games_l686_686199


namespace problem1_problem2_l686_686988

def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem problem1 : ∀ (x₁ x₂ : ℝ), x₁ ∈ (-1 : ℝ, ∞) → x₂ ∈ (-1 : ℝ, ∞) → x₁ < x₂ → f x₁ < f x₂ :=
by {
  intros, 
  sorry
}

theorem problem2 : ∃ (y₁ y₂ : ℝ), y₁ = 1 ∧ y₂ = 5 / 3 ∧ ∀ (y : ℝ), y ∈ (Set.range (fun x => ∃ x₀ ∈ (Set.Icc (0 : ℝ) (2 : ℝ)), f x₀ = y)) ↔ y ∈ Set.Icc y₁ y₂ :=
by {
  sorry
}

end problem1_problem2_l686_686988


namespace cos_30_plus_2α_equals_7_over_9_l686_686942

theorem cos_30_plus_2α_equals_7_over_9 (α : ℝ) 
  (h : Matrix.det ![
    [Real.cos (75 * Real.pi / 180), -Real.sin α],
    [Real.sin (75 * Real.pi / 180), Real.cos α]] = 1 / 3) :
  Real.cos ((30 * Real.pi / 180) + 2 * α) = 7 / 9 :=
sorry

end cos_30_plus_2α_equals_7_over_9_l686_686942


namespace tetrahedron_point_choice_l686_686387

-- Definitions
variables (h s1 s2 : ℝ) -- h, s1, s2 are positive real numbers
variables (A B C : ℝ)  -- A, B, C can be points in space

-- Hypothetical tetrahedron face areas and height
def height_condition (D : ℝ) : Prop := -- D is a point in space
  ∃ (D_height : ℝ), D_height = h

def area_ACD_condition (D : ℝ) : Prop := 
  ∃ (area_ACD : ℝ), area_ACD = s1

def area_BCD_condition (D : ℝ) : Prop := 
  ∃ (area_BCD : ℝ), area_BCD = s2

-- The main theorem
theorem tetrahedron_point_choice : 
  ∃ D, height_condition h D ∧ area_ACD_condition s1 D ∧ area_BCD_condition s2 D :=
sorry

end tetrahedron_point_choice_l686_686387


namespace integral_f_value_l686_686260

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if -m ≤ x ∧ x ≤ m then sin (π * x / (2 * m)) else 1

theorem integral_f_value :
  let m := 13
  let n := 20
  ∫ x in -m..n, f x m = 7 :=
by
  let m := 13
  let n := 20
  sorry

end integral_f_value_l686_686260


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686148

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686148


namespace volume_ratio_cylinders_l686_686206

open Real

noncomputable def volume_ratio_larger_to_smaller (h₁ : Real) (h₂ : Real) (circumference₁ : Real) (circumference₂ : Real) : Real :=
  let r₁ := circumference₁ / (2 * pi)
  let r₂ := circumference₂ / (2 * pi)
  let V₁ := pi * r₁^2 * h₁
  let V₂ := pi * r₂^2 * h₂
  V₂ / V₁

theorem volume_ratio_cylinders : volume_ratio_larger_to_smaller 9 6 6 9 = 9 / 4 := by
  sorry

end volume_ratio_cylinders_l686_686206


namespace inscribed_pentagon_side_len_l686_686274

theorem inscribed_pentagon_side_len (A B C D E A1 B1 C1 D1 E1 : Point) 
  (h_reg: regular_pentagon A B C D E) 
  (h_inscribed: inscribed_pentagon A1 B1 C1 D1 E1 A B C D E) 
  (h_side_len: side_length A B = 1) :
  ∃ (i : ℕ), (i < 5) ∧ (side_length (select_i i [A1, B1, C1, D1, E1])) ≥ (Real.cos (Real.pi / 5)) := 
sorry

end inscribed_pentagon_side_len_l686_686274


namespace center_of_circle_l686_686339

theorem center_of_circle (x y : ℝ) : 
  (x^2 + y^2 = 6 * x - 10 * y + 9) → 
  (∃ c : ℝ × ℝ, c = (3, -5) ∧ c.1 + c.2 = -2) :=
by
  sorry

end center_of_circle_l686_686339


namespace find_number_l686_686776

theorem find_number (x : ℕ) (h : 3 * x = 33) : x = 11 :=
sorry

end find_number_l686_686776


namespace def_triangle_angles_obtuse_l686_686296

-- Define the triangle ABC
variables (A B C : Point)
-- Define midpoints D, E, F on sides AB, BC, CA respectively
variables (D E F : Point)
-- Define circumcircle
variables (circum : Circle)
-- Conditions of the problem
axiom midpoint_D : is_midpoint D A B
axiom midpoint_E : is_midpoint E B C
axiom midpoint_F : is_midpoint F C A
axiom circum_touches_D : circum.touches_side_ab D A B
axiom circum_touches_E : circum.touches_side_bc E B C
axiom circum_touches_F : circum.touches_side_ca F C A

-- Prove that each angle of triangle DEF is obtuse
theorem def_triangle_angles_obtuse
  (h1 : is_triangle A B C)
  (h2 : is_circumscribed_circle circum A B C) :
  ∀ (angle1 angle2 angle3 : Angle), 
  (is_angle_of_triangle angle1 D E F) ∧ 
  (is_angle_of_triangle angle2 E F D) ∧ 
  (is_angle_of_triangle angle3 F D E) →
  angle_is_obtuse angle1 ∧ angle_is_obtuse angle2 ∧ angle_is_obtuse angle3 :=
sorry

end def_triangle_angles_obtuse_l686_686296


namespace inequality_proof_l686_686951

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / (y + z) + y^2 * z / (z + x) + z^2 * x / (x + y) ≥ 1 / 2 * (x^2 + y^2 + z^2)) :=
by sorry

end inequality_proof_l686_686951


namespace machine_made_8_shirts_today_l686_686874

-- Define the conditions
def shirts_per_minute : ℕ := 2
def minutes_worked_today : ℕ := 4

-- Define the expected number of shirts made today
def shirts_made_today : ℕ := shirts_per_minute * minutes_worked_today

-- The theorem stating that the shirts made today should be 8
theorem machine_made_8_shirts_today : shirts_made_today = 8 := by
  sorry

end machine_made_8_shirts_today_l686_686874


namespace scientific_notation_of_2135_billion_l686_686788

theorem scientific_notation_of_2135_billion :
  (2135 * 10^9 : ℝ) = 2.135 * 10^11 := by
  sorry

end scientific_notation_of_2135_billion_l686_686788


namespace coefficient_x3y5_in_expansion_l686_686121

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686121


namespace print_300_pages_in_15_minutes_l686_686254

theorem print_300_pages_in_15_minutes (pages_per_minute : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_minute = 20) (h2 : total_pages = 300) :
  total_pages / pages_per_minute = 15 :=
by
  simp [h1, h2]
  sorry

end print_300_pages_in_15_minutes_l686_686254


namespace circle_tangent_to_parabola_and_x_axis_eqn_l686_686237

theorem circle_tangent_to_parabola_and_x_axis_eqn :
  (∃ (h k : ℝ), k^2 = 2 * h ∧ (x - h)^2 + (y - k)^2 = 2 * h ∧ k > 0) →
    (∀ (x y : ℝ), x^2 + y^2 - x - 2 * y + 1 / 4 = 0) := by
  sorry

end circle_tangent_to_parabola_and_x_axis_eqn_l686_686237


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686079

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686079


namespace angus_has_20_dollars_less_l686_686270

theorem angus_has_20_dollars_less (
  elsa_tokens : ℕ,
  angus_tokens : ℕ,
  token_value : ℕ
)
(h1 : elsa_tokens = 60)
(h2 : angus_tokens = 55)
(h3 : token_value = 4)
:
  elsa_tokens * token_value - angus_tokens * token_value = 20 :=
by {
  rw [h1, h2, h3],
  norm_num,
  }

end angus_has_20_dollars_less_l686_686270


namespace solve_inequality_l686_686024

noncomputable def solution_set : set ℝ :=
  {x : ℝ | 3 * x^2 - 9 * x - 15 ≤ 0 }

theorem solve_inequality :
  solution_set = set.Icc ((3 - Real.sqrt 29) / 2) ((3 + Real.sqrt 29) / 2) :=
sorry

end solve_inequality_l686_686024


namespace num_distinct_units_digits_of_cubes_l686_686466

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l686_686466


namespace distinct_units_digits_of_cubes_l686_686436

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686436


namespace ellipse_equation_l686_686411

theorem ellipse_equation (a b : ℝ) (h1 : a > b ∧ b > 0) (h2 : (1 : ℝ) / a^2 + 9 / (4 * b^2) = 1) (h3 : b^2 + (a/2)^2 = a^2) (h4 : (1 : ℝ)^2 / a^2 + (3 / 2)^2 / b^2 = 1) : 
  (a = 2 ∧ b = √3) ∧ (∀ m : ℝ, ∃ t : ℝ, t^2 = 16 / 3 → m^2 = t^2 + 4) → 
  (∀ m : ℝ, m = 2 * sqrt 21 / 3 ∨ m = - 2 * sqrt 21 / 3 → 
    equation_l : (x : ℝ) = 2* √21 / 3 * y - 4 ∨ l = - 2*√21 /3 * y - 4 ) := 
sorry

end ellipse_equation_l686_686411


namespace coefficient_x3y5_in_expansion_l686_686123

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686123


namespace coeff_x3y5_in_expansion_l686_686028

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686028


namespace zoo_revenue_l686_686674

def num_children_mon : ℕ := 7
def num_adults_mon : ℕ := 5
def num_children_tue : ℕ := 4
def num_adults_tue : ℕ := 2
def cost_child : ℕ := 3
def cost_adult : ℕ := 4

theorem zoo_revenue : 
  (num_children_mon * cost_child + num_adults_mon * cost_adult) + 
  (num_children_tue * cost_child + num_adults_tue * cost_adult) 
  = 61 := 
by
  sorry

end zoo_revenue_l686_686674


namespace imaginary_roots_iff_l686_686363

theorem imaginary_roots_iff {k m : ℝ} (hk : k ≠ 0) : (exists (x : ℝ), k * x^2 + m * x + k = 0 ∧ ∃ (y : ℝ), y * 0 = 0 ∧ y ≠ 0) ↔ m ^ 2 < 4 * k ^ 2 :=
by
  sorry

end imaginary_roots_iff_l686_686363


namespace coeff_x3y5_in_expansion_l686_686031

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l686_686031


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686070

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686070


namespace bacteria_filling_dish_l686_686607

theorem bacteria_filling_dish (day_size_doubles : ∀ n : ℕ, 2 ^ (n + 1) = 2 * 2 ^ n)
  (h : ∃ d : ℕ, d = 25 ∧ (1 / 32 : ℝ) = 2 ^ (-d)) :
  ∃ t : ℕ, t = 30 ∧ (1 : ℝ) = 2 ^ (-25) * 2 ^ (t - 25) := 
  sorry

end bacteria_filling_dish_l686_686607


namespace coefficient_x3y5_in_expansion_l686_686112

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686112


namespace coefficient_x3_y5_in_binomial_expansion_l686_686109

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686109


namespace distinct_units_digits_of_cubes_l686_686541

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686541


namespace distinct_units_digits_of_perfect_cubes_l686_686449

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686449


namespace coefficient_x3y5_in_expansion_l686_686164

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686164


namespace find_one_third_of_product_l686_686337

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end find_one_third_of_product_l686_686337


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686146

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686146


namespace conic_section_is_ellipse_l686_686787

-- Defining the condition for the problem
def conic_section (x y : ℝ) : Prop :=
  sqrt(x^2 + (y + 2)^2) + sqrt((x - 6)^2 + (y - 4)^2) = 14

-- Stating the theorem
theorem conic_section_is_ellipse : ∀ x y: ℝ, conic_section x y → ∃ (E : Type) [ellipse E], E.is_conic_section (x, y) :=
by
  sorry

end conic_section_is_ellipse_l686_686787


namespace distinct_units_digits_perfect_cube_l686_686573

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686573


namespace distinct_units_digits_perfect_cube_l686_686578

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end distinct_units_digits_perfect_cube_l686_686578


namespace distinct_cube_units_digits_l686_686464

theorem distinct_cube_units_digits : 
  ∃ (s : Finset ℕ), (∀ (n : ℕ), (n % 10)^3 % 10 ∈ s) ∧ s.card = 10 := 
by 
  sorry

end distinct_cube_units_digits_l686_686464


namespace A_share_of_profit_l686_686824

section InvestmentProfit

variables (capitalA capitalB : ℕ) -- initial capitals
variables (withdrawA advanceB : ℕ) -- changes after 8 months
variables (profit : ℕ) -- total profit

def investment_months (initial : ℕ) (final : ℕ) (first_period : ℕ) (second_period : ℕ) : ℕ :=
  initial * first_period + final * second_period

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

def A_share (total_profit : ℕ) (ratioA ratioB : ℚ) : ℚ :=
  (ratioA / (ratioA + ratioB)) * total_profit

theorem A_share_of_profit :
  let capitalA := 3000
  let capitalB := 4000
  let withdrawA := 1000
  let advanceB := 1000
  let profit := 756
  let A_investment_months := investment_months capitalA (capitalA - withdrawA) 8 4
  let B_investment_months := investment_months capitalB (capitalB + advanceB) 8 4
  let ratioA := ratio A_investment_months B_investment_months
  let ratioB := ratio B_investment_months A_investment_months
  A_share profit ratioA ratioB = 288 := sorry

end InvestmentProfit

end A_share_of_profit_l686_686824


namespace inspectors_in_group_B_l686_686238

theorem inspectors_in_group_B
  (a b : ℕ)  -- a: number of original finished products, b: daily production
  (A_inspectors := 8)  -- Number of inspectors in group A
  (total_days := 5) -- Group B inspects in 5 days
  (inspects_same_speed : (2 * a + 2 * 2 * b) * total_days/A_inspectors = (2 * a + 2 * 5 * b) * (total_days/3))
  : ∃ (B_inspectors : ℕ), B_inspectors = 12 := 
by
  sorry

end inspectors_in_group_B_l686_686238


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686138

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686138


namespace box_volume_80_possible_l686_686255

theorem box_volume_80_possible :
  ∃ (x : ℕ), 10 * x^3 = 80 :=
by
  sorry

end box_volume_80_possible_l686_686255


namespace operation_correct_l686_686781

theorem operation_correct : ∀ (x : ℝ), (x - 1)^2 = x^2 + 1 - 2x := 
by
  intro x
  sorry

end operation_correct_l686_686781


namespace compute_matrix_combination_l686_686646

variables {R : Type*} [CommRing R] {M : Matrix (Fin 2) (Fin 1) R}
variables {v w u : Matrix (Fin 1) (Fin 1) R}

def matrix_v : Matrix (Fin 2) (Fin 1) R := ![![2], ![-3]]
def matrix_w : Matrix (Fin 2) (Fin 1) R := ![![-1], ![4]]
def matrix_u : Matrix (Fin 2) (Fin 1) R := ![![5], ![-2]]

theorem compute_matrix_combination
  (h1 : M.mul v = matrix_v)
  (h2 : M.mul w = matrix_w)
  (h3 : M.mul u = matrix_u) :
  M.mul (3 • v - 4 • w + 2 • u) = ![![20], ![-29]] :=
sorry

end compute_matrix_combination_l686_686646


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686090

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686090


namespace coffee_mix_price_l686_686835

theorem coffee_mix_price (
  weight1 price1 weight2 price2 total_weight : ℝ)
  (h1 : weight1 = 9)
  (h2 : price1 = 2.15)
  (h3 : weight2 = 9)
  (h4 : price2 = 2.45)
  (h5 : total_weight = 18)
  :
  (weight1 * price1 + weight2 * price2) / total_weight = 2.30 :=
by
  sorry

end coffee_mix_price_l686_686835


namespace distinct_units_digits_of_cube_l686_686592

theorem distinct_units_digits_of_cube :
  {d : ℕ | ∃ n : ℤ, (n % 10) = d ∧ ((n ^ 3) % 10) = d} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by
  sorry

end distinct_units_digits_of_cube_l686_686592


namespace prove_f_neg_2_l686_686415

noncomputable def f (a b x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Main theorem statement
theorem prove_f_neg_2 (a b : ℝ) (h : f a b 2 = 9) : f a b (-2) = 13 := 
by
  sorry

end prove_f_neg_2_l686_686415


namespace min_value_of_expression_l686_686601

open Classical

theorem min_value_of_expression (x : ℝ) (hx : x > 0) : 
  ∃ y, x + 16 / (x + 1) = y ∧ ∀ z, (z > 0 → z + 16 / (z + 1) ≥ y) := 
by
  use 7
  sorry

end min_value_of_expression_l686_686601


namespace domain_of_f_f_odd_find_alpha_l686_686413

open Real

noncomputable def f (x : ℝ) : ℝ := log (3 + x) / log 2 - log (3 - x) / log 2

theorem domain_of_f : ∀ x, -3 < x ∧ x < 3 ↔ ∀ y, f y := sorry

theorem f_odd : ∀ x, f (-x) = -f x := sorry

theorem find_alpha (α : ℝ) (h : f α = 1) : α = 1 := sorry

end domain_of_f_f_odd_find_alpha_l686_686413


namespace circle_fits_right_triangle_l686_686763

theorem circle_fits_right_triangle :
  ∃ r : ℝ, 
    ∀ (A B C : EuclideanGeometry.Point ℝ), 
      A.x - B.x = 12 ∧ 
      A.y = 5 ∧ 
      B.x = 0 ∧ 
      B.y = 0 ∧ 
      C.x = 12 ∧ 
      C.y = 0 → 
        2 * (3 * r + 1) = 25 → 
        r = 3 / 2 :=
by
  sorry

end circle_fits_right_triangle_l686_686763


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686139

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l686_686139


namespace binomial_coefficient_x3y5_in_expansion_l686_686167

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686167


namespace domain_of_h_eq_h_is_odd_h_less_than_zero_set_value_of_a_l686_686996

-- Define the functions f and g along with the conditions
def f (a : ℝ) (x : ℝ) : ℝ := log a (1 + x)
def g (a : ℝ) (x : ℝ) : ℝ := log a (1 - x)
def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

-- Questions reformulated as Lean statements
-- (I) Domain of h and its odd/even property
theorem domain_of_h_eq : ∀ (a : ℝ), a > 0 → a ≠ 1 → ∀ x : ℝ, x ∈ set.Ioo (-1 : ℝ) 1 ↔ true := 
by intros a ha hna x; sorry

theorem h_is_odd : ∀ (a : ℝ), a > 0 → a ≠ 1 → ∀ x : ℝ, h a (-x) = -h a x :=
by intros a ha hna x; sorry

-- (II) f(3) = 2 implies solution set for h(x) < 0
theorem h_less_than_zero_set : ∃ a : ℝ, f a 3 = 2 ∧ ∀ x : ℝ, h a x < 0 ↔ x ∈ set.Ioo (-1 : ℝ) 0 :=
by sorry

-- (III) a > 1 and h x ∈ [0, 1] for x ∈ [0, 1/2] implies a = 3
theorem value_of_a : ∀ (a : ℝ), a > 1 → (∀ x : ℝ, x ∈ set.Icc 0 (1/2) → h a x ∈ set.Icc 0 1) → a = 3 :=
by intros a ha hax; sorry

end domain_of_h_eq_h_is_odd_h_less_than_zero_set_value_of_a_l686_686996


namespace distinct_units_digits_of_cubes_l686_686540

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686540


namespace insects_ratio_l686_686821

theorem insects_ratio (total_insects : ℕ) (geckos : ℕ) (gecko_insects : ℕ) (lizards : ℕ)
  (H1 : geckos * gecko_insects + lizards * ((total_insects - geckos * gecko_insects) / lizards) = total_insects)
  (H2 : total_insects = 66)
  (H3 : geckos = 5)
  (H4 : gecko_insects = 6)
  (H5 : lizards = 3) :
  (total_insects - geckos * gecko_insects) / lizards / gecko_insects = 2 :=
by
  sorry

end insects_ratio_l686_686821


namespace investor_share_purchase_price_l686_686793

theorem investor_share_purchase_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (purchase_price : ℝ)
  (h1 : dividend_rate = 0.125)
  (h2 : face_value = 60)
  (h3 : roi = 0.25)
  (h4 : 0.25 = (0.125 * 60) / purchase_price) 
  : purchase_price = 30 := 
sorry

end investor_share_purchase_price_l686_686793


namespace coefficient_x3_y5_in_binomial_expansion_l686_686102

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686102


namespace true_proposition_l686_686388

-- Definitions
def p (θ : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x * sin θ + 1 ≥ 0

def q : Prop :=
  ∀ α β : ℝ, sin (α + β) ≤ sin α + sin β

-- Theorem
theorem true_proposition (θ : ℝ) : (¬ (p θ)) ∨ q :=
  by
  sorry

end true_proposition_l686_686388


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686086

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686086


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686091

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686091


namespace binomial_coefficient_x3y5_in_expansion_l686_686128

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686128


namespace sequence_general_formula_l686_686379

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
sorry

end sequence_general_formula_l686_686379


namespace no_even_sum_of_four_consecutive_in_circle_l686_686630

theorem no_even_sum_of_four_consecutive_in_circle (n : ℕ) (h1 : n = 2018) :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ n) ∧ (∀ i, i < n → (f (i % n) + f ((i + 1) % n) + f ((i + 2) % n) + f ((i + 3) % n)) % 2 = 1) :=
by { sorry }

end no_even_sum_of_four_consecutive_in_circle_l686_686630


namespace midpoint_ordinate_l686_686399

-- Define the functions and given conditions
def f (x : ℝ) : ℝ := Real.sin x
def g (x : ℝ) : ℝ := Real.cos x

def a : ℝ := sorry
axiom ha : 0 < a ∧ a < Real.pi / 2
axiom hpq : abs (f a - g a) = 1 / 4

-- Statement of the proof problem
theorem midpoint_ordinate :
  let P := (a, f a)
  let Q := (a, g a)
  let midpoint_y := (f a + g a) / 2
  midpoint_y = sqrt 31 / 8 :=
sorry

end midpoint_ordinate_l686_686399


namespace transform_C_D_to_C_l686_686962
open Point

def Point := (ℝ × ℝ)

def scale (p : Point) (factor : ℝ) : Point :=
  (factor * p.1, factor * p.2)

def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

def transform (p : Point) : Point :=
  reflect_x (scale p 2)

def original_C : Point := (-5, 2)
def original_D : Point := (0, 3)
def transformed_C' : Point := (10, -4)
def transformed_D' : Point := (0, -6)

theorem transform_C_D_to_C'_D' :
  transform original_C = transformed_C' ∧ transform original_D = transformed_D' :=
by
  sorry

end transform_C_D_to_C_l686_686962


namespace one_third_of_7_times_9_l686_686320

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_7_times_9_l686_686320


namespace num_proper_subsets_eq_seven_l686_686303

def is_natural (n : Int) : Prop := n ≥ 0

def eval_y (x : Int) : Int := -x * x + 6

def set_y : Set Int := {y | ∃ x : Int, is_natural x ∧ eval_y x = y}

def proper_subsets_count (s : Set Int) : Nat :=
  if s = ∅ then 0 else (2^(s.toFinset.card) - 1)

theorem num_proper_subsets_eq_seven :
  proper_subsets_count {y | ∃ x : Int, is_natural x ∧ eval_y x = y ∧ is_natural y} = 7 :=
by
  sorry

end num_proper_subsets_eq_seven_l686_686303


namespace simplify_tan_expression_l686_686695

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end simplify_tan_expression_l686_686695


namespace parabola_focus_directrix_l686_686971

theorem parabola_focus_directrix (a : ℝ) (h : a > 0)
    (h_dist : (1 : ℝ)/(2 * a) = 1) : a = (1 : ℝ) / 2 := 
begin
  sorry
end

end parabola_focus_directrix_l686_686971


namespace distinct_units_digits_of_cubes_l686_686535

theorem distinct_units_digits_of_cubes :
  ∃ (s : Finset ℕ), (∀ n : ℕ, n < 10 → (Finset.singleton ((n ^ 3) % 10)).Subset s) ∧ s.card = 10 :=
by
  sorry

end distinct_units_digits_of_cubes_l686_686535


namespace domain_of_f_l686_686730

noncomputable def f (x : ℝ) : ℝ := log (2) (1 - 2 * x) + 1 / (x + 1)

theorem domain_of_f : { x : ℝ | ∃ y : ℝ, f x = y } = { x : ℝ | x < 1/2 ∧ x ≠ -1 } :=
by
  sorry

end domain_of_f_l686_686730


namespace curve_passes_through_fixed_point_l686_686410

theorem curve_passes_through_fixed_point (k : ℝ) (x y : ℝ) (h : k ≠ -1) :
  (x ^ 2 + y ^ 2 + 2 * k * x + (4 * k + 10) * y + 10 * k + 20 = 0) → (x = 1 ∧ y = -3) :=
by
  sorry

end curve_passes_through_fixed_point_l686_686410


namespace distance_between_x_intercepts_l686_686837

theorem distance_between_x_intercepts 
  (m₁ m₂ : ℝ) (p : ℝ × ℝ) (h₁ : m₁ = 4) (h₂ : m₂ = -3) (h₃ : p = (8, 20)) : 
  let x₁ := (20 - (20 - m₁ * (8 - 8))) / m₁,
      x₂ := (20 - (20 - m₂ * (8 - 8))) / m₂ in
  |x₁ - x₂| = 35 / 3 :=
by
  sorry

end distance_between_x_intercepts_l686_686837


namespace garden_furniture_costs_l686_686833

theorem garden_furniture_costs (B T U : ℝ)
    (h1 : T + B + U = 765)
    (h2 : T = 2 * B)
    (h3 : U = 3 * B) :
    B = 127.5 ∧ T = 255 ∧ U = 382.5 :=
by
  sorry

end garden_furniture_costs_l686_686833


namespace coefficient_x3y5_in_expansion_l686_686165

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686165


namespace maximize_expression_l686_686652

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
(x^2 + x * y + y^2) * (x^2 + x * z + z^2) * (y^2 + y * z + z^2)

theorem maximize_expression (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) : 
    max_value_expression x y z ≤ 27 :=
sorry

end maximize_expression_l686_686652


namespace coefficient_x3y5_in_expansion_l686_686117

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686117


namespace one_third_of_7_times_9_l686_686322

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_7_times_9_l686_686322


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686083

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686083


namespace CDF_is_correct_l686_686729

noncomputable def p (x : ℝ) : ℝ := 
  if |x| > 1 then 0
  else if -1 < x ∧ x <= 0 then x + 1
  else if 0 < x ∧ x <= 1 then -x + 1
  else 0

noncomputable def F (x : ℝ) : ℝ :=
  if x <= -1 then 0
  else if -1 < x ∧ x <= 0 then (x + 1) ^ 2 / 2
  else if 0 < x ∧ x <= 1 then 1 - (1 - x) ^ 2 / 2
  else 1

theorem CDF_is_correct :
  ∀ (x : ℝ), F(x) = (if x <= -1 then 0
                      else if -1 < x ∧ x <= 0 then (x + 1) ^ 2 / 2
                      else if 0 < x ∧ x <= 1 then 1 - (1 - x) ^ 2 / 2
                      else 1) :=
by
  sorry

end CDF_is_correct_l686_686729


namespace minimum_additional_squares_to_symmetry_l686_686610

-- Define the grid as a set of points and initial shaded squares
def grid := set (ℕ × ℕ)
def initial_shaded_squares : grid := { (1,5), (2,3), (4,2) }

-- Define what it means for a grid to have two lines of symmetry: horizontal and vertical.
def has_two_lines_of_symmetry (s : grid) : Prop :=
  ∀ (x y : ℕ), (x, y) ∈ s → ((6 - x, y) ∈ s ∧ (x, 6 - y) ∈ s ∧ (6 - x, 6 - y) ∈ s)

-- Define the minimum number of additional squares needed for symmetry
def additional_squares_needed (s : grid) : ℕ :=
  let required_to_add := {p | (6 - p.1, p.2) ∉ s ∨ (p.1, 6 - p.2) ∉ s ∨ (6 - p.1, 6 - p.2) ∉ s} in
  (required_to_add \ s).size

-- Main theorem statement: prove the minimum number of additional squares needed to achieve symmetry given the initial condition is 9.
theorem minimum_additional_squares_to_symmetry :
  additional_squares_needed initial_shaded_squares = 9 :=
sorry

end minimum_additional_squares_to_symmetry_l686_686610


namespace coefficient_x3y5_in_expansion_l686_686160

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686160


namespace coefficient_x3y5_in_expansion_l686_686114

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l686_686114


namespace find_blue_balls_l686_686228

/-- 
Given the conditions that a bag contains:
- 5 red balls
- B blue balls
- 2 green balls
And the probability of picking 2 red balls at random is 0.1282051282051282,
prove that the number of blue balls (B) is 6.
--/

theorem find_blue_balls (B : ℕ) (h : 0.1282051282051282 = (10 : ℚ) / (↑((7 + B) * (6 + B)) / 2)) : B = 6 := 
by sorry

end find_blue_balls_l686_686228


namespace compute_binom_product_l686_686892

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l686_686892


namespace sara_lost_seven_marbles_l686_686685

theorem sara_lost_seven_marbles (h_initial : ℕ) (h_left : ℕ) (h_initial_eq : h_initial = 10) (h_left_eq : h_left = 3) : h_initial - h_left = 7 :=
by
  rw [h_initial_eq, h_left_eq]
  norm_num
  sorry

end sara_lost_seven_marbles_l686_686685


namespace KX_times_KQ_equilateral_triangle_l686_686653

-- Define the basic setup for the problem
def equilateral_triangle (A B C : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

def on_side (P A B : ℝ × ℝ) (dPA : ℝ) : Prop :=
  dist P A = dPA ∧ ∃ k : ℝ, P = (1 - k) • A + k • B ∧ 0 ≤ k ∧ k ≤ 1

def concurrent (A B C X Y Z : ℝ × ℝ) : Prop :=
  ∃ F : ℝ × ℝ, collinear A Z F ∧ collinear B Y F ∧ collinear C X F

-- Assume given setup and prove the required statement
theorem KX_times_KQ_equilateral_triangle :
  ∀ (A B C X Y Z P Q K : ℝ × ℝ),
    equilateral_triangle A B C 8 → 
    on_side X A B 5 → 
    on_side Y A C 3 → 
    ∃ Z, concurrent A B C X Y Z → 
    ∃ P Q K, ZX ∩ circle A X Y = P ∧ ZY ∩ circle A X Y = Q → 
    XQ ∩ YP = K →
    dist K X * dist K Q = 304 := by
  sorry

end KX_times_KQ_equilateral_triangle_l686_686653


namespace norm_neg3_v_l686_686368

variable (v : EuclideanSpace ℝ (Fin 2))

-- Hypothesis: the norm of v is 4
def norm_v_is_4 : ∥v∥ = 4 := sorry

-- Theorem: the norm of -3 * v is 12
theorem norm_neg3_v : ∥-3 • v∥ = 12 :=
by
  rw [norm_smul, real.norm_eq_abs, abs_neg, abs_of_nonneg]
  -- Here we assume norm_v_is_4 as a hypothesis
  have h : ∥v∥ = 4 := norm_v_is_4
  rw [h]
  norm_num
  linarith
sorry

end norm_neg3_v_l686_686368


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686043

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686043


namespace no_such_circular_arrangement_l686_686631

theorem no_such_circular_arrangement (numbers : List ℕ) (h1 : numbers = List.range 1 2019) :
  ¬ ∃ f : ℕ → ℕ, (∀ n, f n ∈ numbers) ∧ (∀ n, is_odd (f n + f (n+1) + f (n+2) + f (n+3))) := sorry

end no_such_circular_arrangement_l686_686631


namespace head_start_B_l686_686858

-- Defining speeds of A and B
def speed_B (v : ℝ) := v
def speed_A (v : ℝ) := 2 * v

-- Defining distances and times
def distance_total := 100
def time_A (v t : ℝ) := distance_total / speed_A v
def time_B (v t d : ℝ) := (distance_total - d) / speed_B v

-- Prove head start for B
theorem head_start_B (v t d : ℝ) (h : t = time_A v t) (h1 : t = time_B v t d) : d = 50 := 
by 
  sorry

end head_start_B_l686_686858


namespace log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l686_686911

theorem log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6 
  (h1 : 5^0.6 > 1)
  (h2 : 0 < 0.6^5 ∧ 0.6^5 < 1)
  (h3 : Real.logb 0.6 5 < 0) :
  Real.logb 0.6 5 < 0.6^5 ∧ 0.6^5 < 5^0.6 :=
sorry

end log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l686_686911


namespace binomial_coefficient_x3y5_in_expansion_l686_686124

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686124


namespace triangle_image_has_vertices_l686_686009

noncomputable def T (x y : ℝ) : ℝ × ℝ :=
(x, -y)

noncomputable def S (x y : ℝ) : ℝ × ℝ :=
(-x, y)

noncomputable def R (x y : ℝ) : ℝ × ℝ :=
(3-y, 3-x)

noncomputable def Q (x y : ℝ) : ℝ × ℝ :=
(y + 3, x + 3)

noncomputable def P (x y : ℝ) : ℝ × ℝ :=
(-y - 3, x - 3)

noncomputable def Q_pow_10 (x y : ℝ) : ℝ × ℝ :=
(x + 30, y + 30)

noncomputable def P_pow_2 (x y : ℝ) : ℝ × ℝ :=
(x - 6, y + 6)

noncomputable def P2_then_Q10 (x y : ℝ) : ℝ × ℝ :=
Q_pow_10 (x - 6, y + 6)

theorem triangle_image_has_vertices :
  let v1 := (0, 0)
  let v2 := (0, 1)
  let v3 := (2, 0)
  let image_vertices := (P2_then_Q10 0 0, P2_then_Q10 0 1, P2_then_Q10 2 0)
  image_vertices = ((24, 36), (24, 37), (26, 36)) :=
by 
  sorry

end triangle_image_has_vertices_l686_686009


namespace poles_needed_to_enclose_plot_l686_686853

-- Defining the lengths of the sides
def side1 : ℕ := 15
def side2 : ℕ := 22
def side3 : ℕ := 40
def side4 : ℕ := 30
def side5 : ℕ := 18

-- Defining the distance between poles
def dist_first_three_sides : ℕ := 4
def dist_last_two_sides : ℕ := 5

-- Defining the function to calculate required poles for a side
def calculate_poles (length : ℕ) (distance : ℕ) : ℕ :=
  (length / distance) + 1

-- Total poles needed before adjustment
def total_poles_before_adjustment : ℕ :=
  calculate_poles side1 dist_first_three_sides +
  calculate_poles side2 dist_first_three_sides +
  calculate_poles side3 dist_first_three_sides +
  calculate_poles side4 dist_last_two_sides +
  calculate_poles side5 dist_last_two_sides

-- Adjustment for shared poles at corners
def total_poles : ℕ :=
  total_poles_before_adjustment - 5

-- The theorem to prove
theorem poles_needed_to_enclose_plot : total_poles = 29 := by
  sorry

end poles_needed_to_enclose_plot_l686_686853


namespace A_squared_eq_289_l686_686315

noncomputable def f (x : ℝ) : ℝ := sqrt 17 + 74 / x

def equation_roots (x : ℝ) : Prop :=
  x = f (f x)

def sum_abs_roots : ℝ :=
  |(sqrt 17 + 17) / 2| + |(sqrt 17 - 17) / 2|

theorem A_squared_eq_289 : sum_abs_roots^2 = 289 :=
  sorry

end A_squared_eq_289_l686_686315


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686072

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686072


namespace roots_of_polynomial_l686_686912

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^3 - 3 * x^2 + 2 * x) * (x - 5) = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by 
  sorry

end roots_of_polynomial_l686_686912


namespace sum_of_first_seven_primes_with_units_digit_7_l686_686356

def is_prime (n : ℕ) : Prop := nat.prime n

def units_digit_7 (n : ℕ) : Prop := n % 10 = 7

def first_seven_primes_with_units_digit_7 := 
  {n : ℕ | is_prime n ∧ units_digit_7 n}.to_list.take 7

theorem sum_of_first_seven_primes_with_units_digit_7 :
  first_seven_primes_with_units_digit_7.sum = 379 := by
  sorry

end sum_of_first_seven_primes_with_units_digit_7_l686_686356


namespace tan_product_l686_686699

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end tan_product_l686_686699


namespace distinct_units_digits_of_cube_l686_686487

theorem distinct_units_digits_of_cube : 
  {d : ℤ | ∃ n : ℤ, d = n^3 % 10}.card = 10 :=
by
  sorry

end distinct_units_digits_of_cube_l686_686487


namespace drawing_at_least_one_red_is_certain_l686_686619

-- Defining the balls and box conditions
structure Box :=
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 

-- Let the box be defined as having 3 red balls and 2 yellow balls
def box : Box := { red_balls := 3, yellow_balls := 2 }

-- Define the event of drawing at least one red ball
def at_least_one_red (draws : ℕ) (b : Box) : Prop :=
  ∀ drawn_yellow, drawn_yellow < draws → drawn_yellow < b.yellow_balls

-- The conclusion we want to prove
theorem drawing_at_least_one_red_is_certain : at_least_one_red 3 box :=
by 
  sorry

end drawing_at_least_one_red_is_certain_l686_686619


namespace binomial_product_l686_686889

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l686_686889


namespace binomial_coefficient_x3y5_in_expansion_l686_686126

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l686_686126


namespace find_n_l686_686013

theorem find_n (n k : ℕ) (h_pos : k > 0) (h_calls : ∀ (s : Finset (Fin n)), s.card = n-2 → (∃ (f : Finset (Fin n × Fin n)), f.card = 3^k ∧ ∀ (x y : Fin n), (x, y) ∈ f → x ≠ y)) : n = 5 := 
sorry

end find_n_l686_686013


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686084

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l686_686084


namespace quadratic_inequality_l686_686975

noncomputable def quadratic_solution_set (a b c : ℝ) (x : ℝ) : Prop :=
ax^2 + bx + c ≥ 0

theorem quadratic_inequality (a b c : ℝ) (ha : a > 0) :
  (∀ x, quadratic_solution_set a b c x) = (x ≤ -3 ∨ x ≥ 4) →
  (∀ x, -12 * x^2 + x + 1 > 0) = (x < -1/4 ∨ x > 1/3) :=
by
  intros h1 h2
  sorry

end quadratic_inequality_l686_686975


namespace all_sheep_turn_blue_l686_686677

theorem all_sheep_turn_blue 
  (B : ℕ) (R : ℕ) (V : ℕ) 
  (hB : B = 22) 
  (hR : R = 18) 
  (hV : V = 15) 
  (meet : ∀ B R V : ℕ, (B ≠ 0 ∧ R ≠ 0) → (B ≠ 0 ∧ V ≠ 0) → 
    (R ≠ 0 ∧ V ≠ 0) → B + R + V = 55 → ∃ (X : ℕ), B = X ∧ R = 0 ∧ V = 0) :
  ∃ (X : ℕ), B = X ∧ R = 0 ∧ V = 0 :=
by
  have invariant_RV : (R - V) % 3 = (18 - 15) % 3 := by simp [hR, hV]
  have invariant_VB : (V - B) % 3 = (15 - 22) % 3 := by simp [hB, hV]
  have invariant_BR : (B - R) % 3 = (22 - 18) % 3 := by simp [hB, hR]
  simp [invariant_RV, invariant_VB, invariant_BR]
  exact meet B R V (λ h₁ h₂ h₃ => sorry)

end all_sheep_turn_blue_l686_686677


namespace right_triangle_exponentiation_l686_686738

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def h (u : ℝ) : ℝ := 3 * Real.sqrt 5 * u

theorem right_triangle_exponentiation :
  let u := log_base 8 5,
      leg1 := log_base 8 125,
      leg2 := log_base 2 25,
      hypotenuse := h u
  in 8 ^ hypotenuse = 5 ^ (3 * Real.sqrt 5) :=
by
  let u := log_base 8 5
  let leg1 := log_base 8 125
  let leg2 := log_base 2 25
  let hypotenuse := h u
  sorry

end right_triangle_exponentiation_l686_686738


namespace coefficient_x3_y5_in_binomial_expansion_l686_686098

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686098


namespace polynomial_pair_solution_l686_686907

-- We define the problem in terms of polynomials over real numbers
open Polynomial

theorem polynomial_pair_solution (P Q : ℝ[X]) :
  (∀ x y : ℝ, P.eval (x + Q.eval y) = Q.eval (x + P.eval y)) →
  (P = Q ∨ (∃ a b : ℝ, P = X + C a ∧ Q = X + C b)) :=
by
  intro h
  sorry

end polynomial_pair_solution_l686_686907


namespace coefficient_x3_y5_in_binomial_expansion_l686_686108

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686108


namespace unique_filling_methods_l686_686919

-- Define a 3x3 grid type and the distinct numbers
def grid := matrix (fin 3) (fin 3) (fin 3)

-- Condition: Each row and column contains no repeated numbers.
def no_repeats (g : grid) : Prop :=
  (∀ i : fin 3, (list.nodup (matrix.vec g i))) ∧
  (∀ j : fin 3, (list.nodup (matrix.vec (gᵀ) j)))

noncomputable def num_filling_methods : nat :=
  sorry -- The principle of step counting leads to 12 filling methods.

theorem unique_filling_methods :
  ∃ (n : nat), no_repeats
  ∧ num_filling_methods = 12 :=
sorry

end unique_filling_methods_l686_686919


namespace evaluate_binomial_expression_l686_686312

theorem evaluate_binomial_expression : 
  ∑ k in Finset.range 11, (-1) ^ k * (Nat.choose 10 k) = 0 :=
by
  sorry

end evaluate_binomial_expression_l686_686312


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686050

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686050


namespace geom_seq_expression_l686_686938

theorem geom_seq_expression (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 10) (h2 : a 2 + a 4 = 5) :
  ∀ n, a n = 2 ^ (4 - n) :=
by
  -- sorry is used to skip the proof
  sorry

end geom_seq_expression_l686_686938


namespace points_on_same_line_l686_686306

theorem points_on_same_line (k : ℤ) : 
  (∃ m b : ℤ, ∀ p : ℤ × ℤ, p = (1, 4) ∨ p = (3, -2) ∨ p = (6, k / 3) → p.2 = m * p.1 + b) ↔ k = -33 :=
by
  sorry

end points_on_same_line_l686_686306


namespace sum_of_solutions_l686_686771

theorem sum_of_solutions (a b c : ℚ) (h : a ≠ 0) (eq : 2 * x^2 - 7 * x - 9 = 0) : 
  (-b / a) = (7 / 2) := 
sorry

end sum_of_solutions_l686_686771


namespace distinct_units_digits_of_cubes_l686_686429

theorem distinct_units_digits_of_cubes:
  ∀ d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 
  ∃! u, u = (d^3 % 10) ∧ u ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end distinct_units_digits_of_cubes_l686_686429


namespace inverse_sum_l686_686900

def f : ℝ → ℝ :=
λ x, if x < 1 then 2 * x + 1 else x^2

noncomputable def f_inv (y : ℝ) : ℝ :=
if h : y < 1 then (y - 1) / 2 else real.sqrt y

theorem inverse_sum :
  f_inv (-3) + f_inv (-1) + f_inv (1) + f_inv (3) + f_inv (9) = 1 + real.sqrt 3 :=
by sorry

end inverse_sum_l686_686900


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686049

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l686_686049


namespace trigonometric_identity_solution_l686_686809

open Real

theorem trigonometric_identity_solution (k n l : ℤ) (x : ℝ) 
  (h : 2 * cos x ≠ sin x) : 
  (sin x ^ 3 + cos x ^ 3) / (2 * cos x - sin x) = cos (2 * x) ↔
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨
  (∃ n : ℤ, x = (π / 4) * (4 * n - 1)) ∨
  (∃ l : ℤ, x = arctan (1 / 2) + π * l) :=
sorry

end trigonometric_identity_solution_l686_686809


namespace coefficient_x3_y5_in_binomial_expansion_l686_686104

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686104


namespace coefficient_x3y5_in_expansion_l686_686161

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l686_686161


namespace distance_x_intercepts_correct_l686_686841

noncomputable def distance_between_x_intercepts : ℝ :=
  let slope1 : ℝ := 4
  let slope2 : ℝ := -3
  let intercept_point : Prod ℝ ℝ := (8, 20)
  let line1 (x : ℝ) : ℝ := slope1 * (x - intercept_point.1) + intercept_point.2
  let line2 (x : ℝ) : ℝ := slope2 * (x - intercept_point.1) + intercept_point.2
  let x_intercept1 : ℝ := (0 - intercept_point.2) / slope1 + intercept_point.1
  let x_intercept2 : ℝ := (0 - intercept_point.2) / slope2 + intercept_point.1
  abs (x_intercept2 - x_intercept1)

theorem distance_x_intercepts_correct :
  distance_between_x_intercepts = 35 / 3 :=
sorry

end distance_x_intercepts_correct_l686_686841


namespace distinct_units_digits_of_perfect_cubes_l686_686450

theorem distinct_units_digits_of_perfect_cubes : 
  (∀ n : ℤ, ∃ d : ℤ, d = n % 10 ∧ (n^3 % 10) = (d^3 % 10)) →
  10 = (∃ s : set ℤ, (∀ d ∈ s, (d^3 % 10) ∈ s) ∧ s.card = 10) :=
by
  sorry

end distinct_units_digits_of_perfect_cubes_l686_686450


namespace circle_intersection_exists_l686_686680

theorem circle_intersection_exists (a b : ℝ) :
  ∃ (m n : ℤ), (m - a)^2 + (n - b)^2 ≤ (1 / 14)^2 →
  ∀ x y, (x - a)^2 + (y - b)^2 = 100^2 :=
sorry

end circle_intersection_exists_l686_686680


namespace coefficient_of_x3_y5_in_binomial_expansion_l686_686076

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l686_686076


namespace distinct_units_digits_of_perfect_cube_l686_686507

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l686_686507


namespace baker_sold_cakes_l686_686281

theorem baker_sold_cakes :
  let initial_cakes := 62
  let more_cakes := 149
  let remaining_cakes := 67
  in initial_cakes + more_cakes - remaining_cakes = 144 :=
by
  sorry

end baker_sold_cakes_l686_686281


namespace rate_per_kg_for_mangoes_l686_686283

theorem rate_per_kg_for_mangoes (quantity_grapes : ℕ)
    (rate_grapes : ℕ)
    (quantity_mangoes : ℕ)
    (total_payment : ℕ)
    (rate_mangoes : ℕ) :
    quantity_grapes = 8 →
    rate_grapes = 70 →
    quantity_mangoes = 9 →
    total_payment = 1055 →
    8 * 70 + 9 * rate_mangoes = 1055 →
    rate_mangoes = 55 := by
  intros h1 h2 h3 h4 h5
  have h6 : 8 * 70 = 560 := by norm_num
  have h7 : 560 + 9 * rate_mangoes = 1055 := by rw [h5]
  have h8 : 1055 - 560 = 495 := by norm_num
  have h9 : 9 * rate_mangoes = 495 := by linarith
  have h10 : rate_mangoes = 55 := by linarith
  exact h10

end rate_per_kg_for_mangoes_l686_686283


namespace a_5_eq_634_l686_686310

-- Definition of the sequence a_n
def a : ℕ → ℕ
| 0       := 1
| 1       := 4
| (n + 2) := 3 * a (n + 1) + 2 * a n

-- The goal is to show that a 5 = 634
theorem a_5_eq_634 : a 5 = 634 := by {
  -- Sorry is used to skip the actual proof
  sorry,
}

end a_5_eq_634_l686_686310


namespace coefficient_x3_y5_in_binomial_expansion_l686_686099

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l686_686099


namespace one_third_of_seven_times_nine_l686_686323

theorem one_third_of_seven_times_nine : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l686_686323


namespace train_length_l686_686796

theorem train_length (L : ℝ) :
  (20 * (L + 160) = 15 * (L + 250)) -> L = 110 :=
by
  intro h
  sorry

end train_length_l686_686796


namespace PS_parallel_AE_l686_686640

/- The problem statement -/
variables {A B C D E P Q R S : Point}
variables (k : Circle)
variable (center_of_k_on_AE : Center k ∈ Line A E)
variables (touches_AB : IsTangency k P (Line A B))
variables (touches_BC : IsTangency k Q (Line B C))
variables (touches_CD : IsTangency k R (Line C D))
variables (touches_DE : IsTangency k S (Line D E))
variables (convex_pentagon_ABCDE : ConvexPentagon A B C D E)
variables (AB_CD_eq_BC_DE : Length (LineSeg A B) + Length (LineSeg C D) = Length (LineSeg B C) + Length (LineSeg D E))

theorem PS_parallel_AE :
  Parallel (Line P S) (Line A E) :=
sorry

end PS_parallel_AE_l686_686640
