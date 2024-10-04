import Mathlib

namespace average_customers_per_table_l448_448599

-- Definitions for conditions
def tables : ℝ := 9.0
def women : ℝ := 7.0
def men : ℝ := 3.0

-- Proof problem statement
theorem average_customers_per_table : (women + men) / tables = 10.0 / 9.0 :=
by
  sorry

end average_customers_per_table_l448_448599


namespace standard_deviation_of_sample_is_2_l448_448660

noncomputable def sample : List ℝ := [125, 124, 121, 123, 127]

noncomputable def sampleStandardDeviation (l: List ℝ) : ℝ :=
  let mean := (l.foldl (+) 0) / (l.length)
  let variance := (l.foldl (λ acc x -> acc + (x - mean) ^ 2) 0) / (l.length)
  Real.sqrt variance

theorem standard_deviation_of_sample_is_2 : sampleStandardDeviation sample = 2 :=
by
  sorry

end standard_deviation_of_sample_is_2_l448_448660


namespace probability_8_div_x_gt_x_l448_448949

open Set

theorem probability_8_div_x_gt_x : 
  let S := {x : ℕ | 1 ≤ x ∧ x < 10}
  let favorable := {x ∈ S | (8 : ℝ) / x > x}
  (favorable.card : ℝ) / (S.card) = 2 / 9 :=
by
  sorry

end probability_8_div_x_gt_x_l448_448949


namespace f_is_log_a_l448_448384

variable {ℝ : Type}

-- Conditions
axiom a_pos_and_ne_one (a : ℝ) : 0 < a ∧ a ≠ 1
axiom f_inverse_of_a_pow (f : ℝ → ℝ) (a : ℝ) : ∀ x, f (a ^ x) = x
axiom f_at_2 (f : ℝ → ℝ) : f 2 = 1

-- Statement to prove
theorem f_is_log_a (a : ℝ) (f : ℝ → ℝ) (H1 : 0 < a) (H2 : a ≠ 1) (H3 : f_inverse_of_a_pow f a) (H4 : f 2 = 1) : 
  ∀ x, f x = Real.log x / Real.log a :=
by
  sorry

end f_is_log_a_l448_448384


namespace max_circle_quadrilateral_intersections_l448_448150

theorem max_circle_quadrilateral_intersections :
   ∀ (circle : Type) (quadrilateral : Type), 
   (∀ (seg : Type), seg ⊆ quadrilateral → seg ∩ circle ≤ 2) ∧ quadrilateral.sides = 4 → 
   max_intersections (circle, quadrilateral) = 8 := 
by
  intros _ _ intersect_constraint sides_constraint
  sorry

end max_circle_quadrilateral_intersections_l448_448150


namespace stamps_exchange_l448_448769

theorem stamps_exchange (x : ℝ) (k: ℝ) 
  (h1 : k = x + 5) 
  (h2 : 1.16 * x + 3.8 = 1.04 * x + 2.2 + 1) : 
  x = 45 ∧ k = 50 :=
by {
  sorry,
}

end stamps_exchange_l448_448769


namespace cannot_switch_positions_l448_448899

-- Defining a lattice point as a pair of integers
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Defining the distance function between two lattice points
def distance (A B : LatticePoint) : ℤ :=
  (A.x - B.x)^2 + (A.y - B.y)^2

-- The initial positions of the pebbles A and B
def A := LatticePoint.mk 0 0
def B (p q : ℤ) := LatticePoint.mk p q

-- The invariant distance between A and B
def d (p q : ℤ) : ℤ :=
  p^2 + q^2

-- The main theorem statement, stating it's impossible to switch positions of pebbles A and B while maintaining the distance
theorem cannot_switch_positions (p q : ℤ) (H : distance A (B p q) = d p q) :
  ∀ steps : ℕ, ¬ (switch_positions A (B p q) steps) :=
  sorry

end cannot_switch_positions_l448_448899


namespace integer_root_of_polynomial_l448_448354

theorem integer_root_of_polynomial (b c : ℚ) : 
  (∀ x : ℂ, (x^4 + 7 * x^3 + b * x + c = 0) → (x = 2 + real.sqrt 5 ∨ x = 2 - real.sqrt 5 ∨ x = 0 ∨ x = -11)) :=
by
  sorry

end integer_root_of_polynomial_l448_448354


namespace midpoint_translation_l448_448828

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + t.1, p.2 + t.2)

theorem midpoint_translation :
  let s3_A : ℝ × ℝ := (3, -2)
  let s3_B : ℝ × ℝ := (-5, 4)
  let translation_vector : ℝ × ℝ := (5, 4)
  let s3_midpoint := midpoint s3_A s3_B
  let s4_midpoint := translate s3_midpoint translation_vector
  s4_midpoint = (4, 5) :=
by
  let s3_A : ℝ × ℝ := (3, -2)
  let s3_B : ℝ × ℝ := (-5, 4)
  let translation_vector : ℝ × ℝ := (5, 4)
  let s3_midpoint := midpoint s3_A s3_B
  let s4_midpoint := translate s3_midpoint translation_vector
  show s4_midpoint = (4, 5)
  -- proof here
  sorry

end midpoint_translation_l448_448828


namespace shaded_area_l448_448407

noncomputable def area_semicircle (r : ℝ) : ℝ := (1 / 2) * Math.pi * r^2
noncomputable def area_quarter_circle (r : ℝ) : ℝ := (1 / 4) * Math.pi * r^2

theorem shaded_area (radius_ADB : ℝ) (radius_BEC_CFG : ℝ) (h1 : radius_ADB = 2) (h2 : radius_BEC_CFG = 1)
    (h3 : is_midpoint D A B) (h4 : is_midpoint E B C) (h5 : touches CFG B) : 
    area_semicircle radius_ADB - 2 * area_quarter_circle radius_BEC_CFG = (3 / 2) * Math.pi :=
by
    rw [h1, h2]
    simp [area_semicircle, area_quarter_circle]
    sorry

end shaded_area_l448_448407


namespace restaurant_cost_l448_448552

section Restaurant
variable (total_people kids adults : ℕ) 
variable (meal_cost : ℕ)
variable (total_cost : ℕ)

def calculate_adults (total_people kids : ℕ) : ℕ := 
  total_people - kids

def calculate_total_cost (adults meal_cost : ℕ) : ℕ :=
  adults * meal_cost

theorem restaurant_cost (total_people kids meal_cost : ℕ) :
  total_people = 13 →
  kids = 9 →
  meal_cost = 7 →
  calculate_adults total_people kids = 4 →
  calculate_total_cost 4 meal_cost = 28 :=
by
  intros
  simp [calculate_adults, calculate_total_cost]
  sorry -- Proof would be added here
end Restaurant

end restaurant_cost_l448_448552


namespace yard_length_l448_448190

theorem yard_length (father_step : ℝ) (son_step : ℝ) (total_footprints : ℕ) 
  (h_father_step : father_step = 0.72) 
  (h_son_step : son_step = 0.54) 
  (h_total_footprints : total_footprints = 61) : 
  ∃ length : ℝ, length = 21.6 :=
by
  sorry

end yard_length_l448_448190


namespace volume_of_one_piece_l448_448961

-- Given conditions
def thickness : ℝ := 1 / 4
def diameter : ℝ := 16
def radius : ℝ := diameter / 2
def total_volume : ℝ := π * radius^2 * thickness
def pieces : ℕ := 8

-- Proof statement
theorem volume_of_one_piece : total_volume / pieces = 2 * π :=
by
  sorry

end volume_of_one_piece_l448_448961


namespace find_number_l448_448725

theorem find_number (N : ℝ) (h1 : (3 / 10) * N = 64.8) : N = 216 ∧ (1 / 3) * (1 / 4) * N = 18 := 
by 
  sorry

end find_number_l448_448725


namespace min_sum_xy_l448_448331

theorem min_sum_xy (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_hy : x ≠ y)
  (h_eq : (1 / x : ℝ) + (1 / y) = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l448_448331


namespace cubes_with_even_red_faces_count_l448_448600

def block_dimensions : ℕ × ℕ × ℕ := (6, 4, 2)
def is_painted_red : Prop := true
def total_cubes : ℕ := 48
def cubes_with_even_red_faces : ℕ := 24

theorem cubes_with_even_red_faces_count :
  ∀ (dimensions : ℕ × ℕ × ℕ) (painted_red : Prop) (cubes_count : ℕ), 
  dimensions = block_dimensions → painted_red = is_painted_red → cubes_count = total_cubes → 
  (cubes_with_even_red_faces = 24) :=
by intros dimensions painted_red cubes_count h1 h2 h3; exact sorry

end cubes_with_even_red_faces_count_l448_448600


namespace AC_length_l448_448140

-- Definition of triangle sides and their lengths
variables {A B C D E : Type} [metric_space A] [metric_space B] [metric_space C] 
[metric_space D] [metric_space E]

variables {AB BC CA AD DE EA : ℕ}
#eval (AB : ℝ)
#eval (BC : ℝ)
#eval (CA : ℝ)
#eval (AD : ℝ)
#eval (DE : ℝ)
#eval (EA : ℝ)

-- Given lengths
axiom h_AB : AB = 18
axiom h_BC : BC = 24
axiom h_CA : CA = 20
axiom h_AD : AD = 9
axiom h_DE : DE = 12
axiom h_EA : EA = 15

-- Similarity of triangles
axiom h_sim : (AB.to_real / AD.to_real) = (BC.to_real / DE.to_real) ∧ 
              (BC.to_real / DE.to_real) = (CA.to_real / EA.to_real)

-- Proof that AC equals 20 cm
theorem AC_length : CA = 20 :=
by sorry

end AC_length_l448_448140


namespace tangent_circle_condition_l448_448667

theorem tangent_circle_condition (a b : ℝ) :
  (∀ x, f x = x^3 + a * x - 2 * b) → 
  (∀ x y, circle_eq x y ↔ (x - 2)^2 + (y + 4)^2 = 5) → 
  (∃ P : ℝ × ℝ, P = (1, -2) ∧ tangent_to_circle P f) → 
  (3 * a + 2 * b = -7) :=
by
  sorry

def f (x : ℝ) : ℝ := x^3 + a * x - 2 * b

def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 4)^2 = 5

def tangent_to_circle (P : ℝ × ℝ) (f : ℝ → ℝ) : Prop := 
  let (x, y) := P in
  is_tangent_at P f ∧ (circle_eq x (f x))

-- Placeholder for tangent condition at a point, to be rigorously defined
def is_tangent_at (P : ℝ × ℝ) (f : ℝ → ℝ) : Prop := sorry

end tangent_circle_condition_l448_448667


namespace plane_through_intersection_l448_448304

def plane1 (x y z : ℝ) : Prop := x + y + 5 * z - 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 3 * y - z + 2 = 0
def pointM (x y z : ℝ) : Prop := (x, y, z) = (3, 2, 1)

theorem plane_through_intersection (x y z : ℝ) :
  plane1 x y z ∧ plane2 x y z ∧ pointM x y z → 5 * x + 14 * y - 74 * z + 31 = 0 := by
  intro h
  sorry

end plane_through_intersection_l448_448304


namespace solution_set_of_f_greater_than_one_l448_448342

theorem solution_set_of_f_greater_than_one (f : ℝ → ℝ) (h_inv : ∀ x, f (x / (x + 3)) = x) :
  {x | f x > 1} = {x | 1 / 4 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_f_greater_than_one_l448_448342


namespace exterior_angle_BAC_l448_448579

-- Define the interior angle of a regular n-gon
def interior_angle (n : ℕ) : ℝ := (180 * (n - 2)) / n

-- Conditions:
def isSquare (angle : ℝ) : Prop := angle = 90
def isNonagon (angle : ℝ) : Prop := angle = interior_angle 9

-- Statement: Prove that the measure of the exterior angle BAC is 130 degrees.
theorem exterior_angle_BAC : 
  ∀ (A B C D : Type) (common_side : Prop),
  (common_side → isSquare (90)) →
  (common_side → isNonagon (interior_angle 9)) →
  true := sorry

end exterior_angle_BAC_l448_448579


namespace domain_of_f_l448_448849

def f (x : ℝ) : ℝ := (sqrt (x - 1)) / (x - 2)

theorem domain_of_f :
  {x : ℝ | x - 1 ≥ 0 ∧ x - 2 ≠ 0} = {x : ℝ | 1 ≤ x ∧ x < 2 ∨ 2 < x} :=
by sorry

end domain_of_f_l448_448849


namespace average_speed_correct_l448_448170

noncomputable def average_speed (d v_up v_down : ℝ) : ℝ :=
  let t_up := d / v_up
  let t_down := d / v_down
  let total_distance := 2 * d
  let total_time := t_up + t_down
  total_distance / total_time

theorem average_speed_correct :
  average_speed 0.2 24 36 = 28.8 := by {
  sorry
}

end average_speed_correct_l448_448170


namespace machine_does_not_require_repair_l448_448500

noncomputable def nominal_mass := 390 -- The nominal mass M is 390 grams

def greatest_deviation_preserved := 39 -- The greatest deviation among preserved measurements is 39 grams

def deviation_unread_measurements (x : ℕ) : Prop := x < 39 -- Deviations of unread measurements are less than 39 grams

def all_deviations_no_more_than := ∀ (x : ℕ), x ≤ 39 -- All deviations are no more than 39 grams

theorem machine_does_not_require_repair 
  (M : ℕ) 
  (h_nominal_mass : M = nominal_mass)
  (h_greatest_deviation : greatest_deviation_preserved ≤ 0.1 * M)
  (h_unread_deviations : ∀ (x : ℕ), deviation_unread_measurements x) 
  (h_all_deviations : all_deviations_no_more_than):
  true := -- Prove the machine does not require repair
sorry

end machine_does_not_require_repair_l448_448500


namespace lines_parallel_to_same_are_parallel_l448_448082

theorem lines_parallel_to_same_are_parallel (L1 L2 L3 : Type) [linear_ordered_space L1] 
  (h1 : L1 ∥ L3) (h2 : L2 ∥ L3) : L1 ∥ L2 :=
sorry

end lines_parallel_to_same_are_parallel_l448_448082


namespace percentage_boys_playing_soccer_l448_448754

theorem percentage_boys_playing_soccer
  (total_students : ℕ)
  (total_boys : ℕ)
  (total_playing_soccer : ℕ)
  (girls_not_playing_soccer : ℕ)
  (total_students = 420)
  (total_boys = 320)
  (total_playing_soccer = 250)
  (girls_not_playing_soccer = 65) :
  (215 / 250) * 100 = 86 := 
by
  -- Definitions based on the conditions
  let total_girls := total_students - total_boys
  let girls_playing_soccer := total_girls - girls_not_playing_soccer
  let boys_playing_soccer := total_playing_soccer - girls_playing_soccer
  -- Simplify the percentage calculation
  have h1 : boys_playing_soccer = 215 := by sorry
  have h2 : (215 / 250) * 100 = 86 := by sorry
  exact h2

end percentage_boys_playing_soccer_l448_448754


namespace parametric_solution_of_DE_l448_448145

theorem parametric_solution_of_DE (t : ℝ) (ht : t > 0) :
  let x := Real.log t + Real.sin t,
      y := t * (1 + Real.sin t) + Real.cos t,
      y_prime := (t * Real.cos t + 1) / (1 / t + Real.cos t) in
  x = Real.log y_prime + Real.sin y_prime :=
sorry

end parametric_solution_of_DE_l448_448145


namespace machine_no_repair_needed_l448_448494

theorem machine_no_repair_needed (M : ℕ) (σ : ℕ) (greatest_deviation : ℕ) 
                                  (nominal_weight : ℕ)
                                  (h1 : greatest_deviation = 39)
                                  (h2 : greatest_deviation ≤ (0.1 * nominal_weight))
                                  (h3 : ∀ d, d < 39) : 
                                  σ ≤ greatest_deviation :=
by
  sorry

end machine_no_repair_needed_l448_448494


namespace fg_parallel_hk_and_fk_eq_gh_l448_448175

noncomputable theory
open_locale classical

variables {A B C D E F G H K : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
          [metric_space E] [metric_space F] [metric_space G] [metric_space H] [metric_space K]
variables [has_dist A] [has_dist B] [has_dist C] [has_dist D]
          [has_dist E] [has_dist F] [has_dist G] [has_dist H] [has_dist K]

-- The given conditions
variables (ABC : triangle A B C)
variables (AD BE : line A D)
variables [is_perpendicular AD.to_line ABC.side(BC)]
variables [is_perpendicular BE.to_line ABC.side(AC)]
variables (D_to_G : is_perpendicular D (AB.line))
variables (D_to_K : is_perpendicular D (AC.line))
variables (E_to_F : is_perpendicular E (AB.line))
variables (E_to_H : is_perpendicular E (BC.line))

-- The problem statement
theorem fg_parallel_hk_and_fk_eq_gh :
  parallel (line.through F G) (line.through H K)
  ∧ dist F K = dist G H :=
sorry

end fg_parallel_hk_and_fk_eq_gh_l448_448175


namespace part_i_part_ii_part_iii_l448_448325

-- Definitions corresponding to the given conditions
def Sn (n : ℕ) (a b : Fin n → ℕ) : ℕ := ∑ i, |a i - b i|

theorem part_i (a : Fin 3 → ℕ) (h : a = ⟨1, 3, 5⟩) :
  ∃ b : Fin 3 → ℕ, Sn 3 a b ∈ {3, 5, 7, 9} := sorry

theorem part_ii (n : ℕ) (h : ∀ i, a i = i + 1) :
  ∀ (b : Fin n → ℕ), Sn n a b = n * n := sorry

theorem part_iii (n : ℕ) (a b : Fin n → ℕ) :
  (∑ i : Fin n, a i +  ∑ i : Fin n, b i) % 2 = n % 2 → ∀ b' : Fin n → ℕ, Sn n a b % 2 = Sn n a b' % 2 := sorry

end part_i_part_ii_part_iii_l448_448325


namespace sin_cos_sum_l448_448374

theorem sin_cos_sum (θ : ℝ) (b : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : cos (2 * θ) = b) :
  sin θ + cos θ = sqrt ((1 - b) / 2) + sqrt ((1 + b) / 2) :=
sorry

end sin_cos_sum_l448_448374


namespace problem1_problem2_l448_448356

def setA : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | (x - m + 2) * (x - m - 2) ≤ 0}

-- Problem 1: prove that if A ∩ B = {x | 0 ≤ x ≤ 3}, then m = 2
theorem problem1 (m : ℝ) : (setA ∩ setB m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2 :=
by
  sorry

-- Problem 2: prove that if A ⊆ complement of B, then m ∈ (-∞, -3) ∪ (5, +∞)
theorem problem2 (m : ℝ) : (setA ⊆ (fun x => x ∉ setB m)) → (m < -3 ∨ m > 5) :=
by
  sorry

end problem1_problem2_l448_448356


namespace inversion_of_line_segment_l448_448523

-- Define the given elements: circle, line segment, distance
variables {O : Point} {R : ℝ} {A B : Point} {d : ℝ}

-- Assume given conditions
def given_circle (O : Point) (R : ℝ) : Circle :=
{ center := O, radius := R }

def given_line_segment (A B : Point) : Line :=
{ start := A, end := B }

def distance_from_O_to_AB (O : Point) (A B : Point) : ℝ := d

-- Main theorem statement
theorem inversion_of_line_segment (O : Point) (R d : ℝ) (A B : Point) :
  let O' := symmetric_point O A B in    -- reflection of O across AB
  let O'' := inversion O' O R in        -- inversion of O' with respect to circle(O, R)
  centered_circumference O'' = R^2 / d := 
sorry

end inversion_of_line_segment_l448_448523


namespace range_of_quadratic_l448_448863

-- Define the quadratic function -x^2 + 2x + 3
def quadratic : ℝ → ℝ := λ x, -x^2 + 2*x + 3

-- Define the interval [0, 3]
def interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- State that the range of the function for x in [0, 3] is [0, 4]
theorem range_of_quadratic :
  ∀ (y : ℝ), (∃ (x : ℝ), interval x ∧ y = quadratic x) ↔ 0 ≤ y ∧ y ≤ 4 :=
by
  sorry

end range_of_quadratic_l448_448863


namespace max_fraction_inv_eq_one_l448_448316

open Real

noncomputable def max_value_fraction_inv (x y a b : ℝ) (hx : a > 1) (hy : b > 1)
  (h1 : a^x = 3) (h2 : b^y = 3) (h3 : a + b = 2 * sqrt 3) : ℝ :=
  max (1 / x + 1 / y)

theorem max_fraction_inv_eq_one {x y a b : ℝ}
  (hx : a > 1) (hy : b > 1) (h1 : a^x = 3) (h2 : b^y = 3) (h3 : a + b = 2 * sqrt 3) :
  1 / x + 1 / y = 1 := 
sorry

end max_fraction_inv_eq_one_l448_448316


namespace find_minimum_value_l448_448301

theorem find_minimum_value (c : ℝ) : 
  (∀ c : ℝ, (c = -12) ↔ (∀ d : ℝ, (1 / 3) * d^2 + 8 * d - 7 ≥ (1 / 3) * (-12)^2 + 8 * (-12) - 7)) :=
sorry

end find_minimum_value_l448_448301


namespace opposite_number_113_is_114_l448_448042

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l448_448042


namespace number_of_sheets_is_9_l448_448973

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end number_of_sheets_is_9_l448_448973


namespace remaining_amount_l448_448924

theorem remaining_amount (deposit : ℝ) (percent : ℝ) (total_cost : ℝ) (remaining : ℝ) :
  percent = 0.10 → deposit = 105 → total_cost = deposit / percent → remaining = total_cost - deposit → remaining = 945 :=
by
  intros h_percent h_deposit h_total_cost h_remaining
  rw [h_percent, h_deposit] at h_total_cost
  rw [h_total_cost] at h_remaining
  exact h_remaining

end remaining_amount_l448_448924


namespace tangent_line_circle_range_l448_448706

theorem tangent_line_circle_range
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_tangent : ∃ (x y : ℝ), (x + y = 1) ∧ ((x - a) ^ 2 + (y - b) ^ 2 = 2)) :
  0 < ab ∧ ab ≤ 9 / 4 :=
by
  have h_ab : a + b = 3 := sorry -- derived from the tangent condition
  have h_relation : ab ≤ (3 / 2) ^ 2 := sorry -- arithmetic mean-geometric mean inequality
  have h_positive : ab > 0 := sorry -- since a > 0 and b > 0
  exact ⟨h_positive, h_relation⟩

end tangent_line_circle_range_l448_448706


namespace pipe_fills_tank_in_10_hours_l448_448195

variables (pipe_rate leak_rate : ℝ)

-- Conditions
def combined_rate := pipe_rate - leak_rate
def leak_time := 30
def combined_time := 15

-- Express leak_rate from leak_time
noncomputable def leak_rate_def : ℝ := 1 / leak_time

-- Express pipe_rate from combined_time with leak_rate considered
noncomputable def pipe_rate_def : ℝ := 1 / combined_time + leak_rate_def

-- Theorem to be proved
theorem pipe_fills_tank_in_10_hours :
  (1 / pipe_rate_def) = 10 :=
by
  sorry

end pipe_fills_tank_in_10_hours_l448_448195


namespace find_m_l448_448708

noncomputable def m_value (t : ℝ) : ℝ :=
  4 * (t ^ 2)

theorem find_m (m : ℝ) (t : ℝ) (h : t = 2) : m = 16 :=
  by 
    have ht : t = 2 := h
    have hm : m_value t = 16 := by 
      unfold m_value
      rw [ht]
      norm_num
    rw [hm]
    exact eq_refl 16

end find_m_l448_448708


namespace sequence_3078_153_l448_448488

noncomputable def sequence (n : ℕ) : ℕ → ℕ
| 0       := n
| (k + 1) := let digits := n.digits 10 in digits.map (λ d, d^3).sum

theorem sequence_3078_153 : sequence 3078 3077 = 153 :=
by sorry

end sequence_3078_153_l448_448488


namespace vacant_seats_proof_l448_448925

noncomputable def seats_vacant_proof : Prop :=
let total_seats := 600
let filled_percentage := 0.50
let vacant_percentage := 1 - filled_percentage
let vacant_seats := vacant_percentage * total_seats
in vacant_seats = 300

theorem vacant_seats_proof : seats_vacant_proof := by
  let total_seats := 600
  let filled_percentage := 0.50
  let vacant_percentage := 1 - filled_percentage
  let vacant_seats := vacant_percentage * total_seats
  have h : vacant_seats = 300 := by sorry
  exact h

end vacant_seats_proof_l448_448925


namespace distinct_real_roots_absolute_sum_l448_448160

theorem distinct_real_roots_absolute_sum {r1 r2 p : ℝ} (h_root1 : r1 ^ 2 + p * r1 + 7 = 0) 
(h_root2 : r2 ^ 2 + p * r2 + 7 = 0) (h_distinct : r1 ≠ r2) : 
|r1 + r2| > 2 * Real.sqrt 7 := 
sorry

end distinct_real_roots_absolute_sum_l448_448160


namespace max_circle_quadrilateral_intersections_l448_448149

theorem max_circle_quadrilateral_intersections :
   ∀ (circle : Type) (quadrilateral : Type), 
   (∀ (seg : Type), seg ⊆ quadrilateral → seg ∩ circle ≤ 2) ∧ quadrilateral.sides = 4 → 
   max_intersections (circle, quadrilateral) = 8 := 
by
  intros _ _ intersect_constraint sides_constraint
  sorry

end max_circle_quadrilateral_intersections_l448_448149


namespace emily_subtracts_99_from_50sq_to_get_49sq_l448_448514

-- Define the identity for squares
theorem emily_subtracts_99_from_50sq_to_get_49sq :
  ∀ (x : ℕ), (49 : ℕ) = (50 - 1) → (x = 50 → 49^2 = 50^2 - 99) := by
  intro x h1 h2
  sorry

end emily_subtracts_99_from_50sq_to_get_49sq_l448_448514


namespace iron_balls_molded_l448_448993

-- Define the dimensions of the iron bar
def length_bar : ℝ := 12
def width_bar : ℝ := 8
def height_bar : ℝ := 6

-- Define the volume calculations
def volume_iron_bar : ℝ := length_bar * width_bar * height_bar
def number_of_bars : ℝ := 10
def total_volume_bars : ℝ := volume_iron_bar * number_of_bars
def volume_iron_ball : ℝ := 8

-- Define the goal statement
theorem iron_balls_molded : total_volume_bars / volume_iron_ball = 720 :=
by
  -- Proof is to be filled in here
  sorry

end iron_balls_molded_l448_448993


namespace max_min_fractional_part_l448_448298

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

theorem max_min_fractional_part (n : ℕ) (h : 1 ≤ n ∧ n ≤ 2009) :
  (∃ n, n ∈ set.Icc 1 2009 ∧ fractional_part (n * sqrt 5) = fractional_part (1292 * sqrt 5)) ∧
  (∃ n, n ∈ set.Icc 1 2009 ∧ fractional_part (n * sqrt 5) = fractional_part (1597 * sqrt 5)) :=
sorry

end max_min_fractional_part_l448_448298


namespace cash_after_brokerage_l448_448846

theorem cash_after_brokerage (sale_amount : ℝ) (brokerage_rate : ℝ) :
  sale_amount = 109.25 → brokerage_rate = 0.0025 →
  (sale_amount - sale_amount * brokerage_rate) = 108.98 :=
by
  intros h1 h2
  sorry

end cash_after_brokerage_l448_448846


namespace number_of_correct_conclusions_l448_448213

theorem number_of_correct_conclusions :
  let statement1 := (x : ℝ) → tan x = √3 → x ≠ (π / 3 + n * π) 
  let statement2 := (x : ℝ) → (x ≠ 0 → x - sin x ≠ 0)
  let statement3 := (a b : ℝ) → |a * b| = |a| * |b| → (a / b = 1) 
  (statement1 = false) ∧ (statement2 = false) ∧ (statement3 = true) → (1 = 1) := by
  sorry

end number_of_correct_conclusions_l448_448213


namespace train_crossing_time_l448_448594

-- Definitions of given constants
def speed_kmh : ℝ := 60 -- Speed of train in km/hr
def length_m : ℝ := 200 -- Length of train in meters

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Speed in m/s
def speed_ms : ℝ := speed_kmh * conversion_factor

-- Time calculation
def time_sec : ℝ := length_m / speed_ms

-- The proof statement
theorem train_crossing_time : time_sec = 12 :=
by
  -- The proof will be provided here
  sorry

end train_crossing_time_l448_448594


namespace zach_needs_more_money_zach_more_money_needed_l448_448542

/-!
# Zach's Bike Savings Problem
Zach needs $100 to buy a brand new bike.
Weekly allowance: $5.
Earnings from mowing the lawn: $10.
Earnings from babysitting: $7 per hour.
Zach has already saved $65.
He will receive weekly allowance on Friday.
He will mow the lawn and babysit for 2 hours this Saturday.
Prove that Zach needs $6 more to buy the bike.
-/

def zach_current_savings : ℕ := 65
def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_earnings : ℕ := 10
def babysitting_rate : ℕ := 7
def babysitting_hours : ℕ := 2

theorem zach_needs_more_money : zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)) = 94 :=
by sorry

theorem zach_more_money_needed : bike_cost - (zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours))) = 6 :=
by sorry

end zach_needs_more_money_zach_more_money_needed_l448_448542


namespace hcf_of_two_numbers_l448_448842

theorem hcf_of_two_numbers (a b H : ℕ) (hcf_ab : H = gcd a b) (lcm_factors : lcm a b = H * 11 * 12) (largest_number : max (H * a) (H * b) = 744) :
  H = 24 :=
begin
  sorry
end

end hcf_of_two_numbers_l448_448842


namespace hyperbola_conjugate_axis_twice_transverse_l448_448343

theorem hyperbola_conjugate_axis_twice_transverse (m : ℝ) :
  ((λ x y, x ^ 2 - (y ^ 2) / m ^ 2 = 1) 
  ∧ (2 = 2 * 1)) → (m = 2 ∨ m = -2) :=
by
  sorry

end hyperbola_conjugate_axis_twice_transverse_l448_448343


namespace proof_problem_l448_448631

variable (x : Int) (y : Int) (m : Real)

theorem proof_problem :
  ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) ↔
  (-2 * x + 3 * y = 2 * m ∧ x - 5 * y = -11 ∧ x < 0 ∧ y > 0)
:= sorry

end proof_problem_l448_448631


namespace monotonic_interval_l448_448243

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin(x - Real.pi / 6)

def is_monotonically_increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem monotonic_interval : is_monotonically_increasing_interval f 0 (Real.pi / 2) :=
sorry

end monotonic_interval_l448_448243


namespace range_of_a_l448_448794

variable {x a : ℝ}

def p (a : ℝ) (x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

theorem range_of_a (ha : a < 0) 
  (H : (∀ x, ¬ p a x → q x) ∧ ∃ x, q x ∧ ¬ p a x ∧ ¬ q x) : a ≤ -4 := 
sorry

end range_of_a_l448_448794


namespace triangle_area_correct_l448_448413

theorem triangle_area_correct :
  ∀ (A B C : ℝ) (AB AC : ℝ) (angle_B : ℝ),
  AB = sqrt 3 →
  AC = 1 →
  angle_B = π / 6 →
  let s := (AB + AC + sqrt (AB^2 + AC^2 - 2 * AB * AC * cos angle_B)) / 2 in
  let area := sqrt (s * (s - AB) * (s - AC) * (s - sqrt (AB^2 + AC^2 - 2 * AB * AC * cos angle_B))) in
  area = sqrt 3 / 4 :=
by intros; sorry

end triangle_area_correct_l448_448413


namespace min_questions_to_identify_consecutive_pairs_l448_448070

theorem min_questions_to_identify_consecutive_pairs (n : ℕ) (cards : Finset ℕ) (h : cards = Finset.range (n + 1)) : 
  n = 100 → ∃ k : ℕ, (k < 99) ∧ (∀ (A : ℕ) (B : ℕ), A ∈ cards → B ∈ cards → ¬(A = k ∧ B = k + 1) → ∃ (num_questions : ℕ), num_questions = 98) :=
by
  intro h_n
  use 98
  split
  . exact Nat.lt_succ_self 99
  . intros A B hA hB hAB
    use 98
    sorry

end min_questions_to_identify_consecutive_pairs_l448_448070


namespace monotonic_increasing_l448_448237

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π / 6)

theorem monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 2) → f x1 < f x2 := sorry

end monotonic_increasing_l448_448237


namespace tangent_line_equation_f_monotonicity_and_extremum_l448_448795

noncomputable def f (x : ℝ) : ℝ := - (1/2) * (x - 5)^2 + 6 * Real.log x

def f' (x : ℝ) : ℝ := 5 - x + 6 / x

theorem tangent_line_equation (x : ℝ) (hx : x > 0) :
  10 * x - y - 18 = 0 := sorry

theorem f_monotonicity_and_extremum :
  (∀ x : ℝ, 0 < x ∧ x < 6 → 0 < f' x) ∧
  (∀ x : ℝ, x > 6 → f' x < 0) ∧
  f 6 = - (1/2) + 6 * Real.log 6 := sorry

end tangent_line_equation_f_monotonicity_and_extremum_l448_448795


namespace change_in_profit_rate_l448_448977

theorem change_in_profit_rate (A B C : Type) (P : ℝ) (r1 r2 : ℝ) (income_increase : ℝ) (capital : ℝ) :
  (A_receives : ℝ) = (2 / 3) → 
  (B_C_divide : ℝ) = (1 - (2 / 3)) / 2 → 
  income_increase = 300 → 
  capital = 15000 →
  ((2 / 3) * capital * (r2 / 100) - (2 / 3) * capital * (r1 / 100)) = income_increase →
  (r2 - r1) = 3 :=
by
  intros
  sorry

end change_in_profit_rate_l448_448977


namespace sequence_length_l448_448717

-- Define the arithmetic sequence
def sequence (n : ℕ) : ℝ := 4.5 + n * 4

-- Define the condition that the last term in the sequence is 56.5
def last_term_condition : Prop := ∃ n, sequence n = 56.5

-- Define the proposition that the number of terms is 14
def number_of_terms : Prop := ∃ n, sequence n = 56.5 ∧ n + 1 = 14

-- Main theorem to be proven
theorem sequence_length : number_of_terms :=
by
  sorry

end sequence_length_l448_448717


namespace largest_angle_of_consecutive_odd_int_angles_is_125_l448_448860

-- Definitions for a convex hexagon with six consecutive odd integer interior angles
def is_consecutive_odd_integers (xs : List ℕ) : Prop :=
  ∀ n, 0 ≤ n ∧ n < 5 → xs.get! n + 2 = xs.get! (n + 1)

def hexagon_angles_sum_720 (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.sum = 720

-- Main theorem statement
theorem largest_angle_of_consecutive_odd_int_angles_is_125 (xs : List ℕ) 
(h1 : is_consecutive_odd_integers xs) 
(h2 : hexagon_angles_sum_720 xs) : 
  xs.maximum = 125 := 
sorry

end largest_angle_of_consecutive_odd_int_angles_is_125_l448_448860


namespace net_loss_is_1_percent_l448_448964

noncomputable def net_loss_percent (CP SP1 SP2 SP3 SP4 : ℝ) : ℝ :=
  let TCP := 4 * CP
  let TSP := SP1 + SP2 + SP3 + SP4
  ((TCP - TSP) / TCP) * 100

theorem net_loss_is_1_percent
  (CP : ℝ)
  (HCP : CP = 1000)
  (SP1 : ℝ)
  (HSP1 : SP1 = CP * 1.1 * 0.95)
  (SP2 : ℝ)
  (HSP2 : SP2 = (CP * 0.9) * 1.02)
  (SP3 : ℝ)
  (HSP3 : SP3 = (CP * 1.2) * 1.03)
  (SP4 : ℝ)
  (HSP4 : SP4 = (CP * 0.75) * 1.01) :
  abs (net_loss_percent CP SP1 SP2 SP3 SP4 + 1.09) < 0.01 :=
by
  -- Proof omitted
  sorry

end net_loss_is_1_percent_l448_448964


namespace range_g_l448_448292

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + 
  Real.pi * (Real.arcsin (x / 3)) - 
  (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9 * x + 27)

theorem range_g : 
  set.range g = set.Icc (Real.pi^2 / 6) (25 * Real.pi^2 / 6) := 
sorry

end range_g_l448_448292


namespace boat_travel_upstream_time_l448_448569

open Real

theorem boat_travel_upstream_time :
  ∀ (d : ℝ) (t_d : ℝ) (r_b : ℝ) (t_u : ℝ),
    d = 300 → t_d = 2 → r_b = 105 →
    (∃ c : ℝ, d = (r_b + c) * t_d ∧ d = (r_b - c) * t_u) → t_u = 5 :=
by
  intros d t_d r_b t_u hd ht hr ⟨c, hc1, hc2⟩
  have h1 : 300 / 2 = 105 + c, from hc1 ▸ by rwa [hd, ht]
  have h2 : (300 = 60 * t_u), from hc2 ▸
  have h3: 300 / 60 = t_u, from h2 ▸ by rwa [hd]
  refine h3
  sorry

end boat_travel_upstream_time_l448_448569


namespace problem1_problem2_l448_448336

variables (α β : ℝ)
variables (h0 : α ∈ Ioo 0 (π / 2)) (h1 : β ∈ Ioo 0 (π / 2))
variables (h2 : sin(α + 2 * β) = (7 / 5) * sin α)
variables (h3 : tan α = 3 * tan β)

-- Problem 1: Show that tan(α + β) - 6 * tan β = 0
theorem problem1 : tan (α + β) - 6 * tan β = 0 :=
sorry

-- Problem 2: Given the above conditions and tan α = 3 * tan β, show that α = π / 4
theorem problem2 : α = π / 4 :=
sorry

end problem1_problem2_l448_448336


namespace problem_statement_l448_448701

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x - 4

theorem problem_statement (a b : ℝ) :
  (∀ x, f 2 a b = -2) ∧
  (∀ x, 3 * 2^2 + 2 * a * 2 + b = 1) ∧
  (∀ x, a = -4) ∧
  (∀ x, b = 5) ∧
  (∀ x, f 1 -4 5 = -2) ∧
  (∀ x, f (5/3) -4 5 = -58/27) := sorry

end problem_statement_l448_448701


namespace opposite_number_in_circle_l448_448020

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l448_448020


namespace int_sum_log_eq_l448_448441

theorem int_sum_log_eq (X Y Z : ℤ) (h : X.gcd Y = 1 ∧ Y.gcd Z = 1 ∧ Z.gcd X = 1)
    (eqn : X * real.log 5 / real.log 500 + Y * real.log 2 / real.log 500 = Z) : 
    X + Y + Z = 6 :=
sorry

end int_sum_log_eq_l448_448441


namespace part_1_part_2_part_3_l448_448629

def whiteHorseNumber (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem part_1 : 
  whiteHorseNumber (-2) (-4) 1 = -5/3 :=
by sorry

theorem part_2 : 
  max (whiteHorseNumber (-2) (-4) 1) (max (whiteHorseNumber (-2) 1 (-4)) 
  (max (whiteHorseNumber (-4) (-2) 1) (max (whiteHorseNumber (-4) 1 (-2)) 
  (max (whiteHorseNumber 1 (-4) (-2)) (whiteHorseNumber 1 (-2) (-4)) )))) = 2/3 :=
by sorry

theorem part_3 (x : ℚ) (h : ∃a b c : ℚ, a = -1 ∧ b = 6 ∧ c = x ∧ whiteHorseNumber a b c = 2) : 
  x = -7 ∨ x = 8 :=
by sorry

end part_1_part_2_part_3_l448_448629


namespace non_empty_proper_subsets_A_B_empty_iff_m_neg2_A_superset_B_iff_m_range_l448_448440

noncomputable def setA : Set ℤ := {x | (1 / 32 : ℝ) ≤ 2^(-x : ℝ) ∧ 2^(-x : ℝ) ≤ 4}
noncomputable def setB (m : ℝ) : Set ℝ := {x | x^2 - 3 * m * x + 2 * m^2 - m - 1 < 0}

theorem non_empty_proper_subsets_A : Set ℤ :=
  setA = {-2, -1, 0, 1, 2, 3, 4, 5} 
  → 2^8 - 2 = 254 := by sorry

theorem B_empty_iff_m_neg2 {m : ℝ} : 
  setB m = ∅ ↔ m = -2 := by sorry

theorem A_superset_B_iff_m_range {m : ℝ} :
  (∀ x, x ∈ setB m → x ∈ setA) ↔ (-1 ≤ m ∧ m ≤ 2) := by sorry

end non_empty_proper_subsets_A_B_empty_iff_m_neg2_A_superset_B_iff_m_range_l448_448440


namespace pascal_triangle_probability_l448_448604

theorem pascal_triangle_probability :
  let total_elements := 210
  let total_ones := 39
  let probability := (total_ones : ℚ) / total_elements
  probability = (13 : ℚ) / 70 := by
  -- Here we provide the necessary definitions and the statement of the theorem
  -- To keep the focus on proving the relationship, the steps and proof are abstracted out
  sorry

end pascal_triangle_probability_l448_448604


namespace distinct_products_count_l448_448367

theorem distinct_products_count : 
  let s := {2, 3, 5, 7, 11}
  in (finset.powerset s).filter (λ t, 2 ≤ t.card).image (λ t, t.prod).card = 26 :=
by
  sorry

end distinct_products_count_l448_448367


namespace number_of_correct_statements_l448_448212

theorem number_of_correct_statements :
  ¬ ({0} ∈ ({0, 1, 2} : Set ℕ)) ∧
  (∅ ⊆ {1, 2} : Set ℕ) ∧
  ({0, 1, 2} = ({2, 0, 1} : Set ℕ)) ∧
  ¬ (0 ∈ ∅ : Set ℕ) ∧
  (∀ (A : Set ℕ), A ∩ (∅ : Set ℕ) ≠ A) →
  2 = 2 :=
by
  sorry

end number_of_correct_statements_l448_448212


namespace machine_no_repair_needed_l448_448497

theorem machine_no_repair_needed (M : ℕ) (σ : ℕ) (greatest_deviation : ℕ) 
                                  (nominal_weight : ℕ)
                                  (h1 : greatest_deviation = 39)
                                  (h2 : greatest_deviation ≤ (0.1 * nominal_weight))
                                  (h3 : ∀ d, d < 39) : 
                                  σ ≤ greatest_deviation :=
by
  sorry

end machine_no_repair_needed_l448_448497


namespace population_increase_l448_448100

theorem population_increase (P : ℝ) (h₁ : 11000 * (1 + P / 100) * (1 + P / 100) = 13310) : 
  P = 10 :=
sorry

end population_increase_l448_448100


namespace find_x_l448_448715

theorem find_x (x : ℝ) : let a := (1, 1) in let b := (2, x) in
  let sum := (3, 1 + x) in let diff := (-1, 1 - x) in
  (∃ k : ℝ, sum = (-k, k * (1 - x))) → x = 1 :=
by {
  intros a b sum diff h,
  cases h with k hk,
  cases hk,
  sorry
}

end find_x_l448_448715


namespace slip_3_5_in_F_l448_448763

def slips := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]

def cup_sum (x : List ℝ) := List.sum x

def slips_dist (A B C D E F : List ℝ) : Prop :=
  cup_sum A + cup_sum B + cup_sum C + cup_sum D + cup_sum E + cup_sum F = 50 ∧ 
  cup_sum A = 6 ∧ cup_sum B = 8 ∧ cup_sum C = 10 ∧ cup_sum D = 12 ∧ cup_sum E = 14 ∧ cup_sum F = 16 ∧
  2.5 ∈ B ∧ 2.5 ∈ D ∧ 4 ∈ C

def contains_slip (c : List ℝ) (v : ℝ) : Prop := v ∈ c

theorem slip_3_5_in_F (A B C D E F : List ℝ) (h : slips_dist A B C D E F) : 
  contains_slip F 3.5 :=
sorry

end slip_3_5_in_F_l448_448763


namespace possible_values_for_a_l448_448797

def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + 4 = 0}

theorem possible_values_for_a (a : ℝ) : (B a).Nonempty ∧ B a ⊆ A ↔ a = 4 :=
sorry

end possible_values_for_a_l448_448797


namespace proof_rectangular_equation_proof_range_of_distance_PQ_l448_448709

-- Definitions for the conditions
def polar_equation_curve_C (rho θ : ℝ) : Prop :=
  rho = 2 * sqrt 2 * cos (θ - π / 4) - 2 * sin θ

def parametric_equation_line_l (x y t : ℝ) : Prop :=
  x = -2 + (sqrt 6 / 3) * t ∧ y = (sqrt 3 / 3) * t

-- Resulting rectangular equation for curve C
def rectangular_equation_curve_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Distance from the center (1, 0) of the circle to line l in rectangular form
def distance_from_center_to_line_l : ℝ := sqrt 3

-- Range of values for distance |PQ|
def range_of_distance_PQ : set ℝ :=
  set.Ici (sqrt 3 - 1)

-- Lean statements for the proofs
theorem proof_rectangular_equation :
  ∀ (x y : ℝ), (∃ θ, polar_equation_curve_C (sqrt (x^2 + y^2)) θ) → rectangular_equation_curve_C x y :=
sorry

theorem proof_range_of_distance_PQ :
  ∀ (P Q : ℝ × ℝ), (∃ t, parametric_equation_line_l P.1 P.2 t) ∧ rectangular_equation_curve_C Q.1 Q.2 →
  dist P Q ∈ range_of_distance_PQ :=
sorry

end proof_rectangular_equation_proof_range_of_distance_PQ_l448_448709


namespace race_total_people_l448_448742

theorem race_total_people (b t : ℕ) 
(h1 : b = t + 15) 
(h2 : 3 * t = 2 * b + 15) : 
b + t = 105 := 
sorry

end race_total_people_l448_448742


namespace quadratic_has_two_distinct_real_roots_l448_448678

theorem quadratic_has_two_distinct_real_roots (a : ℝ) (h : 16 + 4 * a > 0) : a > -4 ∧ a ≠ 0 :=
by
  have discriminant_pos := h
  have factor_inq : 16 + 4 * a > 0 := discriminant_pos
  sufficient_conditions: (a > -4) ∧ (a ≠ 0) sorry

end quadratic_has_two_distinct_real_roots_l448_448678


namespace maximum_a_if_f_decreasing_l448_448377

theorem maximum_a_if_f_decreasing :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → deriv (λ x, cos x - sin x) x ≤ 0) → a ≤ 3 * π / 4 :=
by
  sorry

end maximum_a_if_f_decreasing_l448_448377


namespace hyperbola_equation_and_slope_ratio_constant_l448_448352

theorem hyperbola_equation_and_slope_ratio_constant :
  (∃ b : ℝ, ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1 → (x, y) = (4, sqrt 3)) ∧
    (∀ C D : ℝ × ℝ, ∃ k1 k2 : ℝ, 
     (4, 0).1 = (ty + 4).1 ∧ (C.1) = (-2, 0).1 ∧ (D.1) = (2, 0).1 ∧ 
     (C.2) = sqrt 3 ∧ (D.2) = sqrt 3 ∧ 
      k1 = (C.2) / ((C.1) + 2) ∧ k2 = (D.2) / ((D.1) - 2) → k1 / k2 = -1 / 3)) :=
    sorry

end hyperbola_equation_and_slope_ratio_constant_l448_448352


namespace average_of_rest_of_class_l448_448172

theorem average_of_rest_of_class
  (n : ℕ)
  (h1 : n > 0)
  (avg_class : ℝ := 84)
  (avg_one_fourth : ℝ := 96)
  (total_sum : ℝ := avg_class * n)
  (sum_one_fourth : ℝ := avg_one_fourth * (n / 4))
  (sum_rest : ℝ := total_sum - sum_one_fourth)
  (num_rest : ℝ := (3 * n) / 4) :
  sum_rest / num_rest = 80 :=
sorry

end average_of_rest_of_class_l448_448172


namespace solution_set_ineq_l448_448295

theorem solution_set_ineq (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3 / 2 :=
by
  sorry

end solution_set_ineq_l448_448295


namespace find_general_term_range_of_m_l448_448681

open Real Nat

def arithmeticSequence (a d : ℝ) : ℕ → ℝ
| 0     => a
| (n+1) => arithmeticSequence a d n + d

def sumFirstNTerms (a d : ℝ) (n : ℕ) : ℝ :=
  (n + 1) * a + (n * (n + 1) / 2) * d

-- Given conditions
def S4_S2 (a : ℝ) (d : ℝ) := sumFirstNTerms a d 4 - sumFirstNTerms a d 2 = 7 * a
def S5 (a : ℝ) (d : ℝ) := sumFirstNTerms a d 5 = 30

-- Question (1)
theorem find_general_term (a d : ℝ) (hS4_S2 : S4_S2 a d) (hS5 : S5 a d) :
  ∃ n, arithmeticSequence a d n = 2 * n :=
sorry

-- Question (2)
def bn (Sn : ℕ → ℝ) (n : ℕ) : ℝ := 1 / Sn n
def Tn (bn : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in range (n + 1), bn i

theorem range_of_m (a d : ℝ) (hS4_S2 : S4_S2 a d) (hS5 : S5 a d) 
  (Sn : ℕ → ℝ) (hSn : ∀ n, Sn n = sumFirstNTerms a d n) :
   ∀ n, Tn (bn (sumFirstNTerms a d)) n < log (m^2 - m) / log 2 → 
       m ∈ (-∞, -1] ∪ [2, ∞) :=
sorry

end find_general_term_range_of_m_l448_448681


namespace oliver_final_money_l448_448068

-- Define the initial conditions as variables and constants
def initial_amount : Nat := 9
def savings : Nat := 5
def earnings : Nat := 6
def spent_frisbee : Nat := 4
def spent_puzzle : Nat := 3
def spent_stickers : Nat := 2
def movie_ticket_price : Nat := 10
def movie_ticket_discount : Nat := 20 -- 20%
def snack_price : Nat := 3
def snack_discount : Nat := 1
def birthday_gift : Nat := 8

-- Define the final amount of money Oliver has left based on the problem statement
def final_amount : Nat :=
  let total_money := initial_amount + savings + earnings
  let total_spent := spent_frisbee + spent_puzzle + spent_stickers
  let remaining_after_spending := total_money - total_spent
  let discounted_movie_ticket := movie_ticket_price * (100 - movie_ticket_discount) / 100
  let discounted_snack := snack_price - snack_discount
  let total_spent_after_discounts := discounted_movie_ticket + discounted_snack
  let remaining_after_discounts := remaining_after_spending - total_spent_after_discounts
  remaining_after_discounts + birthday_gift

-- Lean theorem statement to prove that Oliver ends up with $9
theorem oliver_final_money : final_amount = 9 := by
  sorry

end oliver_final_money_l448_448068


namespace min_area_l448_448999

def dimension_report (x : ℕ) : set ℕ := {k | k ≥ x - 1 ∧ k ≤ x + 1}

theorem min_area (a b : ℕ) (h1 : a = 4) (h2 : b = 5) :
  ∃ l w : ℕ, l ∈ dimension_report a ∧ w ∈ dimension_report b ∧ l * w = 12 :=
by
  -- Define the dimensions reported in the problem
  let l_min := 3
  let w_min := 4

  -- Assert that minimum length and width belong to their respective dimension sets
  have hl : l_min ∈ dimension_report a,
  from ⟨by decide, by decide⟩

  have hw : w_min ∈ dimension_report b,
  from ⟨by decide, by decide⟩

  -- State that the product of minimum dimensions is 12
  use l_min
  use w_min
  split,
  exact hl,
  split,
  exact hw,
  rfl

-- Note: The proof is not completed; the above statement ensures only the theorem and conditions setup as requested.

end min_area_l448_448999


namespace pet_store_satisfaction_l448_448952

-- Definition of the number of each type of pet
def num_puppies : ℕ := 12
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 9
def num_parrots : ℕ := 7

-- The number of ways to pick one of each type of pet
def pick_one_of_each : ℕ :=
  num_puppies * num_kittens * num_hamsters * num_parrots

-- The number of ways to assign four different pets to four different people
def assign_pets : ℕ := (4!).toNat

-- The total number of ways to satisfy the conditions
def total_ways : ℕ := pick_one_of_each * assign_pets

theorem pet_store_satisfaction :
  total_ways = 181440 :=
by
  sorry

end pet_store_satisfaction_l448_448952


namespace num_extreme_points_l448_448115

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3

theorem num_extreme_points : ∃ x1 x2, f' x1 = 0 ∧ f' x2 = 0 ∧ x1 ≠ x2 ∧ 
  (∀ x ∈ Ioo (-∞) x1, f' x < 0) ∧ (∀ x ∈ Ioo x1 x2, f' x > 0) ∧ (∀ x ∈ Ioo x2 (∞), f' x < 0) :=
sorry

end num_extreme_points_l448_448115


namespace sheela_deposit_percentage_l448_448831

-- Define the given conditions
def sheela_deposit : ℝ := 3400
def sheela_monthly_income : ℝ := 22666.67

-- Define the proof statement
theorem sheela_deposit_percentage :
  (sheela_deposit / sheela_monthly_income) * 100 ≈ 15 :=
by
  sorry

end sheela_deposit_percentage_l448_448831


namespace angle_QPR_is_36_l448_448607

theorem angle_QPR_is_36 (P Q R S T : Type) [neq : P ≠ Q] 
  (isosceles_triangle1 : isosceles P Q R) (isosceles_triangle2 : isosceles P S Q) 
  (isosceles_triangle3 : isosceles R S T) (isosceles_triangle4 : isosceles Q T R)
  (equal_segments1 : dist P S = dist S Q) (equal_segments2 : dist R T = dist R S) 
  (equal_segments3 : dist Q T = dist T R) :
  ∠QPR = 36 :=
sorry

end angle_QPR_is_36_l448_448607


namespace set_inter_complement_l448_448360

open Set

variable {α : Type*}
variable (U A B : Set α)

theorem set_inter_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {1, 4}) :
  ((U \ A) ∩ B) = {4} := 
by
  sorry

end set_inter_complement_l448_448360


namespace new_transport_cost_in_usd_l448_448482

noncomputable def original_weight_kg : ℝ := 80
noncomputable def percentage_change_weight : ℝ := 0.60
noncomputable def initial_bags_count : ℕ := 150
noncomputable def initial_cost_euro : ℝ := 7500
noncomputable def cost_increase_percentage : ℝ := 0.25
noncomputable def kg_to_pounds : ℝ := 2.20462
noncomputable def euro_to_usd : ℝ := 1.1

theorem new_transport_cost_in_usd :
  let new_bags_count := initial_bags_count * 5
  let new_weight_kg := original_weight_kg * percentage_change_weight
  let new_weight_pounds := new_weight_kg * kg_to_pounds
  let new_cost_euro := initial_cost_euro * (1 + cost_increase_percentage)
  let new_cost_usd := new_cost_euro * euro_to_usd 
  let total_new_cost_usd := new_cost_usd * (new_bags_count / initial_bags_count)
  in total_new_cost_usd = 51562.5 := sorry

end new_transport_cost_in_usd_l448_448482


namespace pyramid_cross_section_area_l448_448743

theorem pyramid_cross_section_area (a b : ℝ) :
  let S := point (0, 0, b)
      A := point (0, 0, 0)
      B := point (a, 0, 0)
      D := point (0, a, 0)
      M := midpoint A B
      N := midpoint A D
  in
  is_parallel_to (plane_through M N) (line_through S A) →
  area_of_cross_section (SABCD_pyramid S A B D a b) = (a * b) / 4 :=
begin
  sorry
end

end pyramid_cross_section_area_l448_448743


namespace number_of_sheets_l448_448970

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l448_448970


namespace max_points_of_intersection_circle_quadrilateral_l448_448151

theorem max_points_of_intersection_circle_quadrilateral 
  (circle : Type) (quadrilateral : Type) (side : quadrilateral → Type)
  (intersects_at : circle → side → ℕ) :
  (∀ s : side, intersects_at circle s ≤ 2) →
  cardinality quadrilateral = 4 →
  max_intersections circle quadrilateral = 8 :=
by 
  sorry

end max_points_of_intersection_circle_quadrilateral_l448_448151


namespace range_g_l448_448293

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + 
  Real.pi * (Real.arcsin (x / 3)) - 
  (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9 * x + 27)

theorem range_g : 
  set.range g = set.Icc (Real.pi^2 / 6) (25 * Real.pi^2 / 6) := 
sorry

end range_g_l448_448293


namespace tan_gt_x_plus_one_third_cubed_l448_448462

theorem tan_gt_x_plus_one_third_cubed (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
  tan x > x + (1 / 3) * x^3 :=
sorry

end tan_gt_x_plus_one_third_cubed_l448_448462


namespace find_a_l448_448721

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 8) (h3 : c = 4) : a = 0 :=
by
  sorry

end find_a_l448_448721


namespace sum_palindromic_primes_lt_70_l448_448814

def reverseDigits (n : ℕ) : ℕ :=
  -- Convert number to string, reverse it, and convert back to number
  ( (n.digits 10).reverse).foldl (λ x d, x * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 70 ∧ n ≥ 10 ∧ Nat.Prime n ∧ Nat.Prime (reverseDigits n)

theorem sum_palindromic_primes_lt_70 : 
  (Finset.filter is_palindromic_prime (Finset.range 70)).sum =
    92 := sorry

end sum_palindromic_primes_lt_70_l448_448814


namespace part1_part2_l448_448751

theorem part1 (x y : ℝ) (h1 : (1, 0) = (x, y)) (h2 : (0, 2) = (x, y)): 
    ∃ k b : ℝ, k = -2 ∧ b = 2 ∧ y = k * x + b := 
by 
  sorry

theorem part2 (m n : ℝ) (h : n = -2 * m + 2) (hm : -2 < m ∧ m ≤ 3):
    -4 ≤ n ∧ n < 6 := 
by 
  sorry

end part1_part2_l448_448751


namespace exists_g_function_l448_448423

noncomputable def f (n : ℕ) : ℝ := ∑ i in finset.range n, (1 / (i + 1 : ℝ))

theorem exists_g_function :
  ∃ g : ℕ → ℝ, ∀ n : ℕ, 2 ≤ n → (∑ i in finset.range (n - 1), f (i + 1)) = g n * f n - 1 :=
begin
  have g_def : ∀ n, 2 ≤ n → (n : ℝ) + 1 / f n = 
    ∑ i in finset.range (n - 1), f (i + 1) / f n - 1 + 1 / f n,
  { sorry },
  exact ⟨λ n, if 2 ≤ n then (n : ℝ) + 1 / f n else 0, λ n hn, (g_def n hn).symm⟩
end

end exists_g_function_l448_448423


namespace least_positive_integer_l448_448647
  
theorem least_positive_integer 
  (x : ℕ) (d n : ℕ) (p : ℕ) 
  (h_eq : x = 10^p * d + n) 
  (h_ratio : n = x / 17) 
  (h_cond1 : 1 ≤ d) 
  (h_cond2 : d ≤ 9)
  (h_nonzero : n > 0) : 
  x = 10625 :=
by
  sorry

end least_positive_integer_l448_448647


namespace ian_says_1306_l448_448448

noncomputable def number_i_say := 4 * (4 * (4 * (4 * (4 * (4 * (4 * (4 * 1 - 2) - 2) - 2) - 2) - 2) - 2) - 2) - 2

theorem ian_says_1306 (n : ℕ) : 1 ≤ n ∧ n ≤ 2000 → n = 1306 :=
by sorry

end ian_says_1306_l448_448448


namespace three_digit_integers_211_421_l448_448908

def is_one_more_than_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * k + 1

theorem three_digit_integers_211_421
  (n : ℕ) (h1 : (100 ≤ n) ∧ (n ≤ 999))
  (h2 : is_one_more_than_multiple_of n 2)
  (h3 : is_one_more_than_multiple_of n 3)
  (h4 : is_one_more_than_multiple_of n 5)
  (h5 : is_one_more_than_multiple_of n 7) :
  n = 211 ∨ n = 421 :=
sorry

end three_digit_integers_211_421_l448_448908


namespace rhombus_perimeter_and_radius_l448_448104

-- Define the rhombus with given diagonals
structure Rhombus where
  d1 : ℝ -- diagonal 1
  d2 : ℝ -- diagonal 2
  h : d1 = 20 ∧ d2 = 16

-- Define the proof problem
theorem rhombus_perimeter_and_radius (r : Rhombus) : 
  let side_length := Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)
  let perimeter := 4 * side_length
  let radius := r.d1 / 2
  perimeter = 16 * Real.sqrt 41 ∧ radius = 10 :=
by
  sorry

end rhombus_perimeter_and_radius_l448_448104


namespace regression_line_equation_proof_l448_448183

variable (x y : Fin 8 → ℝ)
variable (sx : ℝ := ∑ i, x i)
variable (sy : ℝ := ∑ i, y i)
variable (sxx : ℝ := ∑ i, (x i) ^ 2)
variable (sxy : ℝ := ∑ i, (x i) * (y i))
variable (n : ℝ := 8)
variable (mean_x : ℝ := sx / n)
variable (mean_y : ℝ := sy / n)
variable (b : ℝ := (sxy - n * mean_x * mean_y) / (sxx - n * mean_x^2))
variable (a : ℝ := mean_y - b * mean_x)
variable (regression_line : ℝ → ℝ := fun x => a + b * x)

theorem regression_line_equation_proof :
  sx = 52 →
  sy = 228 →
  sxx = 478 →
  sxy = 1849 →
  regression_line = fun x => 11.47 + 2.62 * x := by
  sorry

end regression_line_equation_proof_l448_448183


namespace right_triangle_side_ratio_l448_448661

theorem right_triangle_side_ratio 
  (O : Prop) -- the center of the inscribed circle of a right triangle
  (A B C : Point) -- vertices of the right triangle
  (h right_triangle : Triangle A B C) -- the triangle is a right triangle
  (h_hyp : ∃ c : ℝ, hypotenuse A B C = c) -- the hypotenuse is c
  (F : Point) -- midpoint of AB
  (h_O_right_angle : RightAngle (A O F)) -- AF seen at a right angle from O
  (ρ : ℝ ) -- radius of the incircle
  (D E : Point) (h_D_AB : OnTheLine D A B) (h_E_AC : OnTheLine E A C) (h_F_BC : OnTheLine F B C) -- contact points
  : ratio_of_sides A B C = (3 : 4 : 5) :=
sorry

end right_triangle_side_ratio_l448_448661


namespace wheel_speed_l448_448786

noncomputable def find_speed (circumference : ℝ) (feet_to_km : ℝ) (time_change : ℝ) (speed_increase : ℝ) : ℝ :=
  let t := circumference / (3600 * feet_to_km)
  let new_time := t - time_change / 3600
  let new_speed := speed_increase
  (new_speed * new_time) = circumference / feet_to_km →
  ((new_speed + speed_increase) * new_time) = circumference / feet_to_km

theorem wheel_speed :
  find_speed 9 (1 / 3280.84) 0.2 3 = 7.4 :=
sorry

end wheel_speed_l448_448786


namespace area_graph_abs_eq_l448_448148

theorem area_graph_abs_eq : 
  area_enclosed_by_graph (λ x y, |5 * x| + |2 * y| = 10) = 20 :=
sorry

end area_graph_abs_eq_l448_448148


namespace alice_total_pints_wednesday_l448_448806

-- Define pints of ice cream Alice bought each day
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := (1 / 3 : ℝ) * pints_monday.toReal
def pints_returned_wednesday : ℝ := (1 / 2 : ℝ) * pints_tuesday
def pints_on_wednesday : ℝ := pints_sunday.toReal + pints_monday.toReal + pints_tuesday - pints_returned_wednesday

theorem alice_total_pints_wednesday : pints_on_wednesday = 18 := by
  sorry

end alice_total_pints_wednesday_l448_448806


namespace percy_birthday_money_l448_448818

-- Definition of the conditions as constants and calculations:
def playstation_cost : ℕ := 500
def christmas_money : ℕ := 150
def game_price : ℝ := 7.5
def games_needed := 20

-- Lean problem statement to prove Percy got $200 on his birthday:
theorem percy_birthday_money :
  let total_cost := playstation_cost
  let money_from_christmas := christmas_money
  let money_from_games := games_needed * game_price.to_nat -- Convert to nat because games_needed is nat
  let money_still_needed := total_cost - money_from_christmas
  money_still_needed - money_from_games = 200 :=
by
  sorry

end percy_birthday_money_l448_448818


namespace alice_favorite_number_l448_448984

-- Definitions of conditions based on the problem statement
def is_between_30_and_150 (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 150
def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0
def is_not_multiple_of_2 (n : ℕ) : Prop := n % 2 ≠ 0
def digit_sum_is_multiple_of_5 (n : ℕ) : Prop := 
  let digits := n.toString.map (λ c => c.to_nat - '0'.to_nat)
  in digits.sum % 5 = 0

-- Proposition stating the problem and its solution
theorem alice_favorite_number : 
  ∃ n, is_between_30_and_150 n ∧ is_multiple_of_11 n ∧ is_not_multiple_of_2 n ∧ digit_sum_is_multiple_of_5 n ∧ n = 55 :=
by
  sorry

end alice_favorite_number_l448_448984


namespace inradii_equality_of_triangles_l448_448276

-- Definitions for the required elements based on the conditions
def Triangle (α : Type) := Σ a b c : α, a + b + c = 180 -- Triangle with angles summing to 180 degrees

variables {α : Type} [LinearOrderedField α]

-- Given definitions/conditions
variables (ABC : Triangle α) (A B C P D : α)
  (excircle_tangent_at_P : P = some_excircle_tangent_function ABC A B C) -- Assuming a function for excircle tangency point
  (AP_intersects_circumcircle_at_D : intersects_circumcircle AP D) -- Assuming a function for intersection with circumcircle

-- Main theorem statement
theorem inradii_equality_of_triangles :
  let r (T : Triangle α) := some_inradius_function T in -- Assuming a function for inradius
  r (mk_triangle P C D) = r (mk_triangle P B D) := -- mk_triangle creates a triangle from given vertices
sorry

end inradii_equality_of_triangles_l448_448276


namespace opposite_of_113_is_114_l448_448053

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l448_448053


namespace inequality_solution_l448_448836

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4x - 21) / (x^2 - 4) > 0 ↔ 
  x ∈ Set.Ioo (-∞) (-3) ∪ Set.Ioo (-2) 2 ∪ Set.Ioo 7 ∞ := 
by sorry

end inequality_solution_l448_448836


namespace g_f_neg4_l448_448001

def f (x : ℝ) : ℝ := 3 * x^2 - 7

variable (g : ℝ → ℝ)
axiom g_f4 : g (f 4) = 9

theorem g_f_neg4 : g (f (-4)) = 9 :=
by
  have f_neg4 : f (-4) = 3 * (-4)^2 - 7 := rfl
  have f_4 : f 4 = 3 * 4^2 - 7 := rfl
  rw [f_4] at g_f4
  rw [f_neg4, f_4]
  exact g_f4

end g_f_neg4_l448_448001


namespace min_steps_to_empty_piles_l448_448129

theorem min_steps_to_empty_piles : 
  (∃ (piles : ℕ → ℕ) (n : ℕ), (∀ i : ℕ, i < n → piles i = i + 1) ∧ n = 1990) →
  (∀ (step : ℕ), step = 1) →
  constant (number_of_steps r, ∀ r : ℕ, 1990 = ∑ (i : ℕ) in range(1990), i.right.result_of_binary(steps)):
  n
  total_steps(n,:), n + steps_left{piles[i] + k → piles[i= k - i}.singleton}
  minn_steps_remove_ left :
  (∀ (pile : ℕ), h : induction_on n) 
== 11 :=
  sorry

end min_steps_to_empty_piles_l448_448129


namespace time_taken_by_a_l448_448166

noncomputable def work_done_in_one_hour (time_taken : ℝ) : ℝ := 1 / time_taken

def total_work_done (A : ℝ) (hours_a : ℕ) (hours_b : ℕ) : ℝ :=
  hours_a * (work_done_in_one_hour A) + hours_b * (work_done_in_one_hour 12)

theorem time_taken_by_a (A : ℝ) : total_work_done A 4 2 = 1 → A = 4.8 :=
by
  intro h
  sorry

end time_taken_by_a_l448_448166


namespace compare_abc_l448_448313

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := 2.1 ^ 1.1
noncomputable def c : ℝ := Real.log 10 2 + Real.log 10 5

theorem compare_abc : b > a ∧ a > c := 
by {
  sorry
}

end compare_abc_l448_448313


namespace determine_c_l448_448247

theorem determine_c (c : ℝ) (r : ℝ) (h1 : 2 * r^2 - 8 * r - c = 0) (h2 : r ≠ 0) (h3 : 2 * (r + 5.5)^2 + 5 * (r + 5.5) = c) :
  c = 12 :=
sorry

end determine_c_l448_448247


namespace total_amount_paid_l448_448538

theorem total_amount_paid
  (initial_price : ℝ)
  (discount_rate : ℝ)
  (service_tax_rate : ℝ)
  (vat_rate : ℝ)
  (tip_rate_min : ℝ)
  (tip_rate_max : ℝ)
  (tip_chosen : ℝ)
  (discounted_price : ℝ)
  (service_tax : ℝ)
  (vat : ℝ)
  (total_before_tip : ℝ)
  (total_amount : ℝ) 
  (discount : initial_price * discount_rate)
  (dprice_formula : discounted_price = initial_price - discount)
  (service_tax_formula : service_tax = discounted_price * service_tax_rate)
  (vat_formula : vat = discounted_price * vat_rate)
  (total_before_tip_formula : total_before_tip = discounted_price + service_tax + vat)
  (tip_range_check : tip_rate_min < tip_chosen / initial_price ∧ tip_chosen / initial_price < tip_rate_max)
  (total_amount_formula : total_amount = total_before_tip + tip_chosen)
  : total_amount = 43.8425 :=
sorry

end total_amount_paid_l448_448538


namespace problem_solution_l448_448850

theorem problem_solution (x : ℝ) : (3 / (x - 3) = 4 / (x - 4)) ↔ (x = 0) := 
by
  sorry

end problem_solution_l448_448850


namespace common_point_of_symmetric_lines_l448_448793

open EuclideanGeometry

noncomputable def symmetric_point (A P : Point) := -- Definition of the symmetric point will be provided.

theorem common_point_of_symmetric_lines
  {O : Point}
  {r : ℝ}
  {circle : Circle}
  {P A B C : Point}
  (h1 : circle.center = O)
  (h2 : circle.radius = r)
  (h3 : ¬P ∈ circle)
  (l : Line)
  (h4 : l.contains P ∧ (∃ A B : Point, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ circle.contains A ∧ circle.contains B))
  (h5 : C = symmetric_point A P)
  (m : Line) 
  (h6 : m.contains B ∧ m.contains C):
  ∃ M : Point, ∀ l : Line, (l.contains P ∧ (∃ A B : Point, A ≠ B ∧ A ∈ l ∧ B ∈ l ∧ circle.contains A ∧ circle.contains B)) → 
                   (∃ C : Point, C = symmetric_point A P ∧ ∃ m : Line, m.contains B ∧ m.contains C) →
                   ∃ M : Point, (M ∈ m) :=
sorry

end common_point_of_symmetric_lines_l448_448793


namespace inverse_proposition_of_square_positive_l448_448855

theorem inverse_proposition_of_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by
  intro h
  intros x h₁
  sorry

end inverse_proposition_of_square_positive_l448_448855


namespace opposite_number_on_circle_l448_448030

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l448_448030


namespace fewest_printers_l448_448167

/-!
# Fewest Printers Purchase Problem
Given two types of computer printers costing $350 and $200 per unit, respectively,
given that the company wants to spend equal amounts on both types of printers.
Prove that the fewest number of printers the company can purchase is 11.
-/

theorem fewest_printers (p1 p2 : ℕ) (h1 : p1 = 350) (h2 : p2 = 200) :
  ∃ n1 n2 : ℕ, p1 * n1 = p2 * n2 ∧ n1 + n2 = 11 := 
sorry

end fewest_printers_l448_448167


namespace bisectors_form_rhombus_with_parallel_sides_l448_448822

-- Define the cyclic quadrilateral and its properties
variables (A B C D M N K L P Q : Type*)
variables (circle : Circle)
variables (A B C D : Point)
variables (M N : Line)
variables (circumcircle : Circle)
variables (is_cyclic : CyclicQuadrilateral A B C D)
variables (intersect_M : IntersectExtendedLine A B C D = M)
variables (intersect_N : IntersectExtendedLine A D B C = N)
variables (bisector_intersects_K : AngleBisectorIntersect M = K)
variables (bisector_intersects_L : AngleBisectorIntersect M = L)
variables (bisector_intersects_P : AngleBisectorIntersect N = P)
variables (bisector_intersects_Q : AngleBisectorIntersect N = Q)

theorem bisectors_form_rhombus_with_parallel_sides :
  -- \(KP \parallel AC\) and \(LQ \parallel BD\)
  parallel KP AC ∧ parallel LQ BD ∧ 
  -- \(KQLP\) is a rhombus
  (KQLP : Rhombus K Q L P) :=
begin
  sorry,
end

end bisectors_form_rhombus_with_parallel_sides_l448_448822


namespace monotonic_interval_l448_448241

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin(x - Real.pi / 6)

def is_monotonically_increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem monotonic_interval : is_monotonically_increasing_interval f 0 (Real.pi / 2) :=
sorry

end monotonic_interval_l448_448241


namespace single_shot_percentage_decrease_l448_448465

theorem single_shot_percentage_decrease
  (initial_salary : ℝ)
  (final_salary : ℝ := initial_salary * 0.95 * 0.90 * 0.85) :
  ((1 - final_salary / initial_salary) * 100) = 27.325 := by
  sorry

end single_shot_percentage_decrease_l448_448465


namespace opposite_113_eq_114_l448_448067

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l448_448067


namespace angle_BED_is_65_l448_448737

-- Definition of angles in degrees
def deg_to_rad (d : ℝ) : ℝ := d * real.pi / 180

-- Lean 4 Statement for the given problem
theorem angle_BED_is_65 (A B C D E : Type) 
  (angle_A angle_C : ℝ) 
  (on_AB : D)
  (on_BC : E)
  (DB BE : ℝ)
  (DB_eq_BE : DB = BE)
  (triangle_abc : angle_A = deg_to_rad 55 ∧ angle_C = deg_to_rad 75) :
  let B_angle := π - angle_A - angle_C in
  (2 * deg_to_rad 50 + 2 * B_angle = π) → (B_angle = deg_to_rad 65) :=
sorry

end angle_BED_is_65_l448_448737


namespace captain_zarnin_staffing_l448_448257

theorem captain_zarnin_staffing : 
  let total_resumes := 30
  let unsuitable_resumes := (2 / 3 : ℝ) * total_resumes
  let suitable_resumes := total_resumes - unsuitable_resumes.toNat
  suitable_resumes = 10 → 
  let ways := suitable_resumes * (suitable_resumes - 1) * 
              (suitable_resumes - 2) * (suitable_resumes - 3) * 
              (suitable_resumes - 4)
  ways = 30240 :=
by
  sorry

end captain_zarnin_staffing_l448_448257


namespace steps_A_l448_448133

theorem steps_A (t_A t_B : ℝ) (a e t : ℝ) :
  t_A = 3 * t_B →
  t_B = t / 75 →
  a + e * t = 100 →
  75 + e * t = 100 →
  a = 75 :=
by sorry

end steps_A_l448_448133


namespace find_position_1993_l448_448803

-- Define the maximum number in the nth diagonal
def max_in_diagonal (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the function that determines the diagonal of a given number
def find_diagonal (x : ℕ) : ℕ :=
  Nat.find (λ n, max_in_diagonal n > x) - 1

-- Define the function to find the position within the diagonal
def position_in_diagonal (x : ℕ) (n : ℕ) : ℕ :=
  x - max_in_diagonal (n - 1)

-- Determine the row and column given number x, the number n of diagonal, and position within diagonal
def row_col (x : ℕ) : ℕ × ℕ :=
  let n := find_diagonal x
  let pos := position_in_diagonal x n
  if n % 2 = 0 then -- Even diagonals decrease downwards
    (pos, n - pos + 1)
  else -- Odd diagonals increase upwards
    (n - pos + 1, pos)

-- Formal prove statement
theorem find_position_1993 :
  row_col 1993 = (24, 40) :=
by
  sorry

end find_position_1993_l448_448803


namespace scientific_notation_of_1500_l448_448843

theorem scientific_notation_of_1500 :
  (1500 : ℝ) = 1.5 * 10^3 :=
sorry

end scientific_notation_of_1500_l448_448843


namespace pool_filling_time_l448_448764

theorem pool_filling_time (rate_jim rate_sue rate_tony : ℝ) (h1 : rate_jim = 1 / 30) (h2 : rate_sue = 1 / 45) (h3 : rate_tony = 1 / 90) : 
     1 / (rate_jim + rate_sue + rate_tony) = 15 := by
  sorry

end pool_filling_time_l448_448764


namespace collinear_points_d_value_l448_448876

theorem collinear_points_d_value (a b c d : ℚ)
  (h1 : b = a)
  (h2 : c = -(a+1)/2)
  (collinear : (4 * d * (4 * a + 5) + a + 1 = 0)) :
  d = 9/20 :=
by {
  sorry
}

end collinear_points_d_value_l448_448876


namespace mean_temperature_is_correct_l448_448858

-- Defining the list of temperatures
def temperatures : List ℝ := [75, 74, 76, 77, 80, 81, 83, 85, 83, 85]

-- Lean statement asserting the mean temperature is 79.9
theorem mean_temperature_is_correct : temperatures.sum / (temperatures.length: ℝ) = 79.9 := 
by
  sorry

end mean_temperature_is_correct_l448_448858


namespace maximum_length_daniels_list_l448_448626

theorem maximum_length_daniels_list (n : ℕ) (D M : list ℕ) (H1 : ∀ x ∈ D, 1 ≤ x ∧ x ≤ 10)
  (H2 : ∀ i, i < n → M[i] = list.count (D[i]) D)
  (H3 : list.reverse M = D) :
  n ≤ 10 :=
sorry

end maximum_length_daniels_list_l448_448626


namespace peanut_count_l448_448387

-- Definitions
def initial_peanuts : Nat := 10
def added_peanuts : Nat := 8

-- Theorem to prove
theorem peanut_count : (initial_peanuts + added_peanuts) = 18 := 
by
  -- Proof placeholder
  sorry

end peanut_count_l448_448387


namespace max_points_of_intersection_circle_quadrilateral_l448_448152

theorem max_points_of_intersection_circle_quadrilateral 
  (circle : Type) (quadrilateral : Type) (side : quadrilateral → Type)
  (intersects_at : circle → side → ℕ) :
  (∀ s : side, intersects_at circle s ≤ 2) →
  cardinality quadrilateral = 4 →
  max_intersections circle quadrilateral = 8 :=
by 
  sorry

end max_points_of_intersection_circle_quadrilateral_l448_448152


namespace negation_of_homework_submission_l448_448857

variable {S : Type} -- S is the set of all students in this class
variable (H : S → Prop) -- H(x) means "student x has submitted the homework"

theorem negation_of_homework_submission :
  (¬ ∀ x, H x) ↔ (∃ x, ¬ H x) :=
by
  sorry

end negation_of_homework_submission_l448_448857


namespace part_a_part_b_part_c_part_d_part_e_l448_448821

variable (n : ℤ)

theorem part_a : (n^3 - n) % 3 = 0 :=
  sorry

theorem part_b : (n^5 - n) % 5 = 0 :=
  sorry

theorem part_c : (n^7 - n) % 7 = 0 :=
  sorry

theorem part_d : (n^11 - n) % 11 = 0 :=
  sorry

theorem part_e : (n^13 - n) % 13 = 0 :=
  sorry

end part_a_part_b_part_c_part_d_part_e_l448_448821


namespace cryptarithm_solution_l448_448834

theorem cryptarithm_solution :
  ∃ (M O S C W L Y P U : ℕ),
    M = 1 ∧ O = 9 ∧ S = 4 ∧ C = 5 ∧ W = 6 ∧ L = 7 ∧ Y = 0 ∧ P = 2 ∧ U = 3 ∧
    (100 * M + 10 * S + U) * 4 + (10000 * O + 1000 * L + 100 * Y + 10 * M + P) * 2 =
    100000 * M + 10000 * O + 1000 * S + 100 * C + 10 * O + W :=
begin
  -- Definitions from conditions
  let MSU := 100 * M + 10 * S + U,
  let OLYMP := 10000 * O + 1000 * L + 100 * Y + 10 * M + P,
  let MOSCOW := 100000 * M + 10000 * O + 1000 * S + 100 * C + 10 * O + W,
  -- Using given digit assignments
  use [1, 9, 4, 5, 6, 7, 0, 2, 3],
  -- Check the cryptarithm equation
  -- (4 * MSU + 2 * OLYMP = MOSCOW) is equivalent to
  -- (4 * (100 * 1 + 10 * 4 + 3) + 2 * (10000 * 9 + 1000 * 7 + 100 * 0 + 10 * 1 + 2) = 100000 * 1 + 10000 * 9 + 1000 * 4 + 100 * 5 + 10 * 9 + 6)
  sorry
end

end cryptarithm_solution_l448_448834


namespace problem_statement_l448_448695

noncomputable def complex_seq (n : ℕ) : ℂ :=
  if n = 0 then 1 else (1 : ℂ) - 1 + (n * (2 + complex.i))

theorem problem_statement : complex_seq 2015 = 4029 + 2014 * complex.i := 
by 
  sorry

end problem_statement_l448_448695


namespace find_max_dot_product_l448_448691

noncomputable def max_dot_product : ℝ :=
  let P := (λ α : ℝ, (Real.cos α, Real.sin α))
  let A := (-2 : ℝ, 0 : ℝ)
  let O := (0 : ℝ, 0 : ℝ)
  let AO := (2 : ℝ, 0 : ℝ)
  let AP := (λ α : ℝ, (Real.cos α + 2, Real.sin α))
  let dot_product := (λ α : ℝ, AO.1 * (AP α).1 + AO.2 * (AP α).2)
  (6 : ℝ)

theorem find_max_dot_product : ∃ α : ℝ, α ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ dot_product α = max_dot_product := sorry

end find_max_dot_product_l448_448691


namespace two_digit_number_determined_l448_448983

theorem two_digit_number_determined
  (x y : ℕ)
  (hx : 0 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (h : 2 * (5 * x - 3) + y = 21) :
  10 * y + x = 72 := 
sorry

end two_digit_number_determined_l448_448983


namespace matrix_reflection_square_is_identity_l448_448426

-- Assumptions
variables (R : Matrix (Fin 2) (Fin 2) ℝ)
let v : Vector ℝ 2 := ![2, 3]

-- Condition that R is the reflection matrix over vector v
axiom reflection_matrix_condition : ∀ (u : Vector ℝ 2), R.mul_vec u = u - (2 * (v.inner u / v.inner v)) • v

-- Theorem statement
theorem matrix_reflection_square_is_identity : R.mul R = (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end matrix_reflection_square_is_identity_l448_448426


namespace radius_of_circle_area_of_sector_l448_448327

theorem radius_of_circle (L : ℝ) (θ : ℝ) (hL : L = 50) (hθ : θ = 200) : 
  ∃ r : ℝ, r = 45 / Real.pi := 
by
  sorry

theorem area_of_sector (L : ℝ) (r : ℝ) (hL : L = 50) (hr : r = 45 / Real.pi) : 
  ∃ S : ℝ, S = 1125 / Real.pi := 
by
  sorry

end radius_of_circle_area_of_sector_l448_448327


namespace sequence_sum_eq_l448_448323

theorem sequence_sum_eq :
  ∀ (n : ℕ) (a : ℕ → ℝ),
    a 2 = 2 →
    a 5 = 1 / 4 →
    (∀ k : ℕ, a (k + 1) = a k * (1 / 2)) →
    (∑ i in Finset.range n, a i * a (i + 1)) = (32 / 3) * (1 - 4^(-n)) := by
  sorry

end sequence_sum_eq_l448_448323


namespace greatest_value_of_x_l448_448473

theorem greatest_value_of_x
  (x : ℕ)
  (h1 : x % 4 = 0) -- x is a multiple of 4
  (h2 : x > 0) -- x is positive
  (h3 : x^3 < 2000) -- x^3 < 2000
  : x ≤ 12 :=
by
  sorry

end greatest_value_of_x_l448_448473


namespace no_int_coeffs_l448_448824

def P (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_int_coeffs (a b c d : ℤ) : 
  ¬ (P a b c d 19 = 1 ∧ P a b c d 62 = 2) :=
by sorry

end no_int_coeffs_l448_448824


namespace initial_investment_C_l448_448587

def total_investment : ℝ := 425
def increase_A (a : ℝ) : ℝ := 0.05 * a
def increase_B (b : ℝ) : ℝ := 0.08 * b
def increase_C (c : ℝ) : ℝ := 0.10 * c

theorem initial_investment_C (a b c : ℝ) (h1 : a + b + c = total_investment)
  (h2 : increase_A a = increase_B b) (h3 : increase_B b = increase_C c) : c = 100 := by
  sorry

end initial_investment_C_l448_448587


namespace opposite_113_eq_114_l448_448066

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l448_448066


namespace question1_question2_l448_448333

section

variable {α : Type*} [LinearOrder α]

-- Definitions for sets A and B
def A (a : α) := { x : α | x < a }
def B := { x : α | -2 < x ∧ x < 4 }

-- Question 1
theorem question1 (a : α) (h : a = 3) :
  let U := A a ∪ B
  let C_U_A := { x : α | x ∈ U ∧ x ∉ A a }
  B ∪ C_U_A = { x : α | x > -2 } :=
by
  sorry

-- Question 2
theorem question2 (a : α) (h : A a ∩ B = B) :
  4 ≤ a :=
by
  sorry

end

end question1_question2_l448_448333


namespace locus_and_line_eqns_l448_448673

-- Definitions based on conditions
def circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

def point_A : ℝ × ℝ := (3, 0)

-- Main theorem to prove the locus and line equations
theorem locus_and_line_eqns (x y : ℝ) (P : ℝ × ℝ) (l : ℝ → ℝ) :
  (∃ Q, (circle Q.fst Q.snd ∧ ((x - Q.fst)^2 + (y - Q.snd)^2 = 
  2 * ((x - 3)^2 + y^2))) ∧ (x^2 + y^2 - 12 * x + 4 * y + 15 = 0)) ∧
  (∃ M N : ℝ × ℝ, l 3 = 0 ∧ (x = 3 ∨ 5 * x - 12 * y - 15 = 0) ∧ 
  ((M.fst - N.fst)^2 + (M.snd - N.snd)^2 = 64)) :=
sorry

end locus_and_line_eqns_l448_448673


namespace find_a_l448_448700

-- Given function and its condition
def f (a x : ℝ) := a * x ^ 3 + 3 * x ^ 2 + 2
def f' (a x : ℝ) := 3 * a * x ^ 2 + 6 * x

-- Condition and proof that a = -2 given the condition f'(-1) = -12
theorem find_a 
  (a : ℝ)
  (h : f' a (-1) = -12) : 
  a = -2 := 
by 
  sorry

end find_a_l448_448700


namespace monotonically_increasing_interval_l448_448226

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l448_448226


namespace scientific_notation_proof_l448_448838

-- Given number is 657,000
def number : ℕ := 657000

-- Scientific notation of the given number
def scientific_notation (n : ℕ) : Prop :=
    n = 657000 ∧ (6.57 : ℝ) * (10 : ℝ)^5 = 657000

theorem scientific_notation_proof : scientific_notation number :=
by 
  sorry

end scientific_notation_proof_l448_448838


namespace how_many_years_older_is_a_than_b_l448_448920

variable (a b c : ℕ)

theorem how_many_years_older_is_a_than_b
  (hb : b = 4)
  (hc : c = b / 2)
  (h_ages_sum : a + b + c = 12) :
  a - b = 2 := by
  sorry

end how_many_years_older_is_a_than_b_l448_448920


namespace fare_above_3_final_position_total_fare_sum_l448_448182

-- Definitions and conditions
def fare (x : ℕ) : ℝ :=
  if x ≤ 3 then 10
  else 10 + (x - 3) * 1.8

def trips : List ℤ := [5, 2, -4, -3, 10]

-- The statements to prove
theorem fare_above_3 (x : ℕ) (h : x > 3) : fare x = 1.8 * x + 4.6 :=
by sorry

theorem final_position : 
  let total_distance := trips.foldl (+) 0 in
  total_distance = 10 :=
by sorry

theorem total_fare_sum : 
  let fare_total := fare 5 + fare 2 + fare 4 + fare 3 + fare 10 in
  fare_total = 68 :=
by sorry

end fare_above_3_final_position_total_fare_sum_l448_448182


namespace average_root_and_volume_correlation_coefficient_total_volume_l448_448982

noncomputable section

open Real

def root_cross_sectional_areas : List ℝ := 
[0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]

def volumes : List ℝ := 
[0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

def sum_root_cross_sectional_areas := 0.6
def sum_volumes := 3.9

def sum_squares_root := 0.038
def sum_squares_volume := 1.6158
def sum_product := 0.2474

theorem average_root_and_volume (n : ℝ) (sumx : ℝ) (sumy : ℝ) :
  n = 10 ∧ sumx = 0.6 ∧ sumy = 3.9 →
  (sumx / n = 0.06) ∧ (sumy / n = 0.39) :=
by
  intros h
  cases h with hn hs
  cases hs with hx hy
  simp [hn, hx, hy]

theorem correlation_coefficient (s_xy : ℝ) (s_xx : ℝ) (s_yy : ℝ) :
  s_xy = 0.0134 ∧ s_xx = 0.002 ∧ s_yy = 0.0948 →
  (s_xy / (Real.sqrt (s_xx * s_yy)) = 0.97) :=
by
  intros h
  cases h with hxy hs
  cases hs with hxx hyy
  have h : (sqrt (s_xx * s_yy)) = 0.01377 := by
    simp [hxx, hyy, sqrt_mul, sqrt_eq_rfl]
  simp [hxy, h] 

theorem total_volume (total_root_area : ℝ) (avg_root : ℝ) (avg_vol : ℝ) :
  total_root_area = 186 ∧ avg_root = 0.06 ∧ avg_vol = 0.39 →
  (total_root_area * avg_vol / avg_root = 1209) :=
by
  intros h
  cases h with hx hs
  cases hs with hr hv
  simp [hx, hr, hv]

#eval average_root_and_volume 10 sum_root_cross_sectional_areas sum_volumes -- Proof goal 1
#eval correlation_coefficient 0.0134 0.002 0.0948 -- Proof goal 2
#eval total_volume 186 0.06 0.39 -- Proof goal 3

end average_root_and_volume_correlation_coefficient_total_volume_l448_448982


namespace product_sum_l448_448859

theorem product_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
(h5 : 5.6 = (a + b / 10) * (c + d / 10)) : 
(a + b / 10) + (c + d / 10) = 5.1 := 
by 
  sorry

end product_sum_l448_448859


namespace opposite_of_113_is_114_l448_448052

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l448_448052


namespace opposite_number_113_is_13_l448_448037

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l448_448037


namespace determine_monotonically_increasing_interval_l448_448221

noncomputable theory
open Real

/-- Problem Statement: Given the function f(x) = 7 * sin(x - π / 6), prove that it is monotonically increasing in the interval (0, π/2). -/
def monotonically_increasing_interval (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2

def function_f (x : ℝ) : ℝ :=
  7 * sin (x - π / 6)

theorem determine_monotonically_increasing_interval :
  ∀ x : ℝ, monotonically_increasing_interval x → monotone_on function_f (set.Ioo 0 (π / 2)) :=
sorry

end determine_monotonically_increasing_interval_l448_448221


namespace correct_statements_l448_448162

-- Define the conditions
variable {x y : Type}
variables (mean_x mean_y : ℝ) (b a : ℝ)
variables (r : ℝ) (sum_squares : ℝ)

-- Assumption that the regression line passes through the mean point
def regression_line_passes_through_mean (x_mean y_mean : ℝ) :=
∀ (x y : ℝ), y = b * x + a → y_mean = b * x_mean + a

-- Assumption that positive slope indicates positive correlation
def positive_slope_positive_correlation (b : ℝ) :=
b > 0 → ∀ (x : ℝ), (b * x + a) > (b * (x - 1) + a)

-- Assumption that larger |r| indicates stronger linear relationship
def larger_r_stronger_relationship (r : ℝ) :=
|r| ≤ 1 → ∀ (r1 r2 : ℝ), (|r1| > |r2|) → (abs r1) ≥ (abs r2)

-- Assumption that smaller sum of squares indicates better fit
def sum_of_squares_model_fit (sum_squares : ℝ) :=
(sum_squares ≥ 0) → ∀ (s1 s2 : ℝ), (s1 < s2) → (sum_squares < s2)

-- Theorem statement
theorem correct_statements
    (hx : regression_line_passes_through_mean mean_x mean_y)
    (hy : positive_slope_positive_correlation b)
    (hr : larger_r_stronger_relationship r)
    (hs : sum_of_squares_model_fit sum_squares) :
    (b > 0) ∧ (|r| ≤ 1) :=
by
    sorry

end correct_statements_l448_448162


namespace taco_cost_l448_448914

theorem taco_cost (T E : ℝ) (h1 : 2 * T + 3 * E = 7.80) (h2 : 3 * T + 5 * E = 12.70) : T = 0.90 := 
by 
  sorry

end taco_cost_l448_448914


namespace quadrilateral_area_is_35_l448_448487

def line1 (x : ℝ) : ℝ := -x + 7
def line2 (x : ℝ) : ℝ := x / 2 + 1
def line3 (x : ℝ) : ℝ := -3 / 2 * x + 21
def line4 (x : ℝ) : ℝ := 7 / 4 * x + 3 / 2

def point (x y : ℝ) := (x, y)

def A := point 2 5
def B := point 4 3
def C := point 10 6
def D := point 6 12

def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def area_quadrilateral (A B C D : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  triangle_area x1 y1 x2 y2 x3 y3 + triangle_area x3 y3 x4 y4 x1 y1

theorem quadrilateral_area_is_35 :
  area_quadrilateral A B C D = 35 := by
  sorry

end quadrilateral_area_is_35_l448_448487


namespace purely_imaginary_complex_l448_448340

theorem purely_imaginary_complex (a : ℝ) (h : (1 + 2 * a) + (a - 2) * complex.I = (0 : ℂ)) : a = -1 / 2 :=
by
  -- here we would prove that a = -1/2 given the condition h
  sorry

end purely_imaginary_complex_l448_448340


namespace win_sector_area_l448_448188

theorem win_sector_area (r : ℝ) (P : ℚ) (h_r : r = 10) (h_P : P = 3 / 7) : 
  let total_area := π * r^2 in
  let win_area := P * total_area in
  win_area = 300 * π / 7 :=
by
  sorry

end win_sector_area_l448_448188


namespace negation_of_exists_proposition_l448_448010

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) → (∀ n : ℕ, n^2 ≤ 2^n) := 
by 
  sorry

end negation_of_exists_proposition_l448_448010


namespace bananas_count_l448_448393

def students_brought_fruits (A B AB only_one : ℕ) : Prop :=
  A = 12 ∧ 
  AB = 5 ∧ 
  only_one = 10 ∧ 
  (A - AB) + (B - AB) = only_one

theorem bananas_count : ∃ B, students_brought_fruits 12 B 5 10 ∧ B = 8 :=
begin
  use 8,
  unfold students_brought_fruits,
  simp,
  sorry
end

end bananas_count_l448_448393


namespace tangent_line_equation_l448_448319

variable {r x_0 y_0 : ℝ}

def is_on_circle (x_0 y_0 r : ℝ) : Prop :=
  x_0 ^ 2 + y_0 ^ 2 = r ^ 2

def is_tangent_line (x y x_0 y_0 r : ℝ) : Prop :=
  x_0 * x + y_0 * y = r ^ 2

theorem tangent_line_equation (h : is_on_circle x_0 y_0 r) :
  ∃ x y : ℝ, is_tangent_line x y x_0 y_0 r :=
begin
  sorry
end

end tangent_line_equation_l448_448319


namespace rectangle_fourth_vertex_l448_448875

theorem rectangle_fourth_vertex (z1 z2 z3 : ℂ)
  (h1 : z1 = 3 + 2 * complex.i)
  (h2 : z2 = 1 + 1 * complex.i)
  (h3 : z3 = -1 - 2 * complex.i) :
  ∃ z4 : ℂ, z4 = -3 - 3 * complex.i :=
begin
  -- proof will go here
  sorry,
end

end rectangle_fourth_vertex_l448_448875


namespace circulation_value_l448_448641

-- Define the vector field
def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (y / 3, -3 * x, x)

-- Define the parameterization of the closed contour
def contour_x (t : ℝ) : ℝ := 2 * Real.cos t
def contour_y (t : ℝ) : ℝ := 2 * Real.sin t
def contour_z (t : ℝ) : ℝ := 1 - 2 * Real.cos t - 2 * Real.sin t

-- Define the parameterization as a function returning a tuple
def contour (t : ℝ) : ℝ × ℝ × ℝ :=
  (contour_x t, contour_y t, contour_z t)

-- Define the interval of parameterization
def parameter_interval : Set ℝ :=
  Set.Icc 0 (2 * Real.pi)

-- Define the line integral for circulation
def circulation : ℝ :=
  ∮ (parameter_interval) 
  (fun t => let (x, y, z) := contour t in
            let (a, b, c) := vector_field x y z in
            a * D (contour_x t) t + b * D (contour_y t) t + c * D (contour_z t) t)

theorem circulation_value :
  circulation = - (52 * Real.pi / 3) :=
by
  sorry

end circulation_value_l448_448641


namespace gcd_45123_32768_l448_448284

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end gcd_45123_32768_l448_448284


namespace convex_polygon_isosceles_equal_sides_l448_448943

-- Define a polygon and its properties (this is abstract and high-level)
structure Polygon :=
(vertices : finset (ℝ × ℝ))
(convex : ∀ a, a ∈ vertices → ∀ b, b ∈ vertices → ∀ c, c ∉ vertices → 
  ∃ α β, 0 < α ∧ 0 < β ∧ α + β = 1 ∧ c = (α • a + β • b : ℝ × ℝ))

-- Define non-intersecting diagonals' property
def non_intersecting_diagonals (P : Polygon) : Prop :=
sorry -- Abstract definition, need detailed construction based on geometry

-- Define isosceles triangle
structure IsoscelesTriangle :=
(vertices : finset (ℝ × ℝ))
(equal_sides : ∃ a b, a ≠ b ∧ (a ∈ vertices ∧ b ∈ vertices) ∧
  ∀ c, c ∈ vertices → c = a ∨ c = b ∨ dist a c = dist b c)

-- Prove that a convex polygon divided by non-intersecting diagonals into isosceles triangles has at least two equal sides
theorem convex_polygon_isosceles_equal_sides
  (P : Polygon) (h_division : non_intersecting_diagonals P) (h_isosceles : ∀ T ∈ triangles P h_division, IsoscelesTriangle T) :
  ∃ a b, a ≠ b ∧ a ∈ P.vertices ∧ b ∈ P.vertices ∧ dist a b = side_len :=
begin
  sorry -- The proof is omitted as indicated in the requirements
end

end convex_polygon_isosceles_equal_sides_l448_448943


namespace average_is_4_l448_448143

-- Define a 4x4 grid (matrix)
def Matrix (α : Type) (m n : Nat) := Fin m → Fin n → α

-- Define specific values and their constraints
def is_valid_grid (grid : Matrix ℕ 4 4) : Prop :=
  ∀ i j, 
    grid i j ∈ {1, 3, 5, 7} ∧ 
    (∀ k, grid i j ≠ grid i k ∨ j = k) ∧ 
    (∀ l, grid i j ≠ grid l j ∨ i = l) ∧
    (i / 2, j / 2 → ∀ i' j', (i' < 2 ∧ j' < 2) → grid (i / 2 * 2 + i') (j / 2 * 2 + j') ≠ grid i j)

def abcd_positions_valid (grid : Matrix ℕ 4 4) (A B C D : Fin 2 → Fin 2 → ℕ) : Prop :=
  grid 0 0 = A ∧ grid 0 3 = D ∧ grid 3 0 = C ∧ grid 3 3 = B

def average (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem average_is_4 :
  ∀ (grid : Matrix ℕ 4 4),
  is_valid_grid grid →
  abcd_positions_valid grid (λ _ _, 1) (λ _ _, 1) (λ _ _, 7) (λ _ _, 7) →
  average 1 1 7 7 = 4 := by
  sorry

end average_is_4_l448_448143


namespace remainder_addition_l448_448532

theorem remainder_addition (m : ℕ) (k : ℤ) (h : m = 9 * k + 4) : (m + 2025) % 9 = 4 := by
  sorry

end remainder_addition_l448_448532


namespace desired_average_score_is_correct_l448_448089

-- Conditions
def average_score_9_tests : ℕ := 82
def score_10th_test : ℕ := 92

-- Desired average score
def desired_average_score : ℕ := 83

-- Total score for 10 tests
def total_score_10_tests (avg9 : ℕ) (score10 : ℕ) : ℕ :=
  9 * avg9 + score10

-- Main theorem statement to prove
theorem desired_average_score_is_correct :
  total_score_10_tests average_score_9_tests score_10th_test / 10 = desired_average_score :=
by
  sorry

end desired_average_score_is_correct_l448_448089


namespace palindromic_primes_sum_55_l448_448817

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10 in
  let units := n % 10 in
  units * 10 + tens

-- Define the predicate for a palindromic prime
def is_palindromic_prime (n : ℕ) : Prop :=
  n < 70 ∧ 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ is_prime (reverse_digits n)

-- Define the list of all two-digit numbers
def two_digit_numbers : List ℕ :=
  List.range' 10 90

-- Construct the list of palindromic primes less than 70
def palindromic_primes : List ℕ :=
  (two_digit_numbers.filter is_palindromic_prime)

-- Define the sum of palindromic primes
def sum_palindromic_primes : ℕ :=
  palindromic_primes.sum

-- State the theorem
theorem palindromic_primes_sum_55 : sum_palindromic_primes = 55 := by
  sorry

end palindromic_primes_sum_55_l448_448817


namespace machine_no_repair_l448_448505

def nominal_mass (G_dev: ℝ) := G_dev / 0.1

theorem machine_no_repair (G_dev: ℝ) (σ: ℝ) (non_readable_dev_lt: ∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) : 
  (G_dev = 39) ∧ (σ ≤ G_dev) ∧ (∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) ∧ (G_dev ≤ 0.1 * nominal_mass(G_dev)) → 
  ¬ machine.requires_repair :=
by
  sorry

end machine_no_repair_l448_448505


namespace bugs_meet_at_point_P_after_7_minutes_l448_448138

-- Define the problem conditions
def R1 : ℝ := 7
def R2 : ℝ := 3
def v1 : ℝ := 4 * Real.pi
def v2 : ℝ := 3 * Real.pi

-- Define the circumferences
def C1 : ℝ := 2 * R1 * Real.pi
def C2 : ℝ := 2 * R2 * Real.pi

-- Define the time taken for one complete rotation
def T1 : ℝ := C1 / v1
def T2 : ℝ := C2 / v2

-- Find the meeting time
def nextMeetingTime : ℝ := Nat.lcm (7 / 2) 2

-- Statement to be proved
theorem bugs_meet_at_point_P_after_7_minutes : nextMeetingTime = 7 := by 
sorry

end bugs_meet_at_point_P_after_7_minutes_l448_448138


namespace angle_at_9_15_is_172_5_deg_l448_448903

noncomputable def minute_hand_angle (minute : ℕ) : ℝ :=
  (minute / 60.0) * 360.0 -- angle with respect to 12:00

noncomputable def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  let base_angle := (hour % 12) * 30.0 in -- each hour represents 30 degrees
  let additional_angle := (minute / 60.0) * 30.0 in -- hour hand moves 0.5 degrees per minute
  base_angle + additional_angle 

noncomputable def angle_between_hands (hour : ℕ) (minute : ℕ) : ℝ :=
  let ha := hour_hand_angle hour minute in
  let ma := minute_hand_angle minute in
  let angle_diff := (ha - ma).abs in
  if angle_diff > 180.0 then 360.0 - angle_diff else angle_diff

theorem angle_at_9_15_is_172_5_deg : angle_between_hands 9 15 = 172.5 :=
by
  -- sorry, proof skipped as requested
  sorry

end angle_at_9_15_is_172_5_deg_l448_448903


namespace faster_watch_rate_of_gain_l448_448897

theorem faster_watch_rate_of_gain
    (alarm_set_time : ℕ := 10) (night : String := "PM")
    (faster_watch_time : ℕ := 4 * 60 + 12) -- 4:12 AM in minutes
    (correct_watch_time : ℕ := 4 * 60) -- 4:00 AM in minutes
    (elapsed_hours : ℕ := 6) :
    let time_difference := faster_watch_time - correct_watch_time,
        rate_of_gain := time_difference / elapsed_hours
    in rate_of_gain = 2 :=
by
  sorry

end faster_watch_rate_of_gain_l448_448897


namespace jake_peaches_count_l448_448415

-- Define Jill's peaches
def jill_peaches : ℕ := 5

-- Define Steven's peaches based on the condition that Steven has 18 more peaches than Jill
def steven_peaches : ℕ := jill_peaches + 18

-- Define Jake's peaches based on the condition that Jake has 6 fewer peaches than Steven
def jake_peaches : ℕ := steven_peaches - 6

-- The theorem to prove that Jake has 17 peaches
theorem jake_peaches_count : jake_peaches = 17 := by
  sorry

end jake_peaches_count_l448_448415


namespace inequality_bounds_of_xyz_l448_448690

theorem inequality_bounds_of_xyz
  (x y z : ℝ)
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 6)
  (h4 : x * y + y * z + z * x = 9) :
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := 
sorry

end inequality_bounds_of_xyz_l448_448690


namespace positive_number_percentage_of_itself_is_9_l448_448197

theorem positive_number_percentage_of_itself_is_9 (x : ℝ) (hx_pos : 0 < x) (h_condition : 0.01 * x^2 = 9) : x = 30 :=
by
  sorry

end positive_number_percentage_of_itself_is_9_l448_448197


namespace sum_of_u_and_v_l448_448003

theorem sum_of_u_and_v (u v : ℤ) (h1 : 1 ≤ v) (h2 : v < u) (h3 : u^2 + v^2 = 500) : u + v = 20 := by
  sorry

end sum_of_u_and_v_l448_448003


namespace determine_monotonically_increasing_interval_l448_448222

noncomputable theory
open Real

/-- Problem Statement: Given the function f(x) = 7 * sin(x - π / 6), prove that it is monotonically increasing in the interval (0, π/2). -/
def monotonically_increasing_interval (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2

def function_f (x : ℝ) : ℝ :=
  7 * sin (x - π / 6)

theorem determine_monotonically_increasing_interval :
  ∀ x : ℝ, monotonically_increasing_interval x → monotone_on function_f (set.Ioo 0 (π / 2)) :=
sorry

end determine_monotonically_increasing_interval_l448_448222


namespace quadratic_has_two_distinct_real_roots_find_m_and_roots_l448_448710

-- The quadratic equation given
def quadratic_eqn (m : ℝ) (x : ℝ) : ℝ := x^2 + (m + 3) * x + (m + 1)

-- Part 1: Proving the equation always has two distinct real roots
theorem quadratic_has_two_distinct_real_roots (m : ℝ):
  let Δ := (m + 3)^2 - 4 * (m + 1) in
  0 < Δ := 
sorry

-- Part 2: Given |x1 - x2| = 2√2, find m and the roots of the equation
theorem find_m_and_roots (m x1 x2 : ℝ) (h_diff : |x1 - x2| = 2 * real.sqrt 2) 
  (h_root1 : quadratic_eqn m x1 = 0) 
  (h_root2 : quadratic_eqn m x2 = 0): 
  (m = -3 ∨ m = 1) ∧ 
  ((m = -3 ∧ x1 = real.sqrt 2 ∧ x2 = -real.sqrt 2) ∨ 
   (m = 1 ∧ x1 = -2 + real.sqrt 2 ∧ x2 = -2 - real.sqrt 2)) := 
sorry

end quadratic_has_two_distinct_real_roots_find_m_and_roots_l448_448710


namespace total_students_l448_448864

-- Definitions extracted from the conditions 
def ratio_boys_girls := 8 / 5
def number_of_boys := 128

-- Theorem to prove the total number of students
theorem total_students : 
  (128 + (5 / 8) * 128 = 208) ∧ ((128 : ℝ) * (13 / 8) = 208) :=
by
  sorry

end total_students_l448_448864


namespace num_teachers_under_40_sample_l448_448576

/-- A school has a total of 490 teachers. 350 teachers are under the age of 40 and 140 teachers are 
    40 years or older. The school decides to use stratified sampling to draw a sample of 70 teachers 
    for a Mandarin proficiency test, maintaining the same ratio of teachers under 40 as in the 
    total population. 
    We need to prove the number of teachers under the age of 40 that should be selected for the 
    sample is 50. -/
theorem num_teachers_under_40_sample (total_teachers : ℕ) (under_40 : ℕ) (age_40_or_older : ℕ) (sample_size : ℕ)
  (h₁ : total_teachers = 490) (h₂ : under_40 = 350) (h₃ : age_40_or_older = 140) (h₄ : sample_size = 70) :
  let sample_under_40 := under_40 * sample_size / total_teachers in
  sample_under_40 = 50 := by
begin
  sorry
end

end num_teachers_under_40_sample_l448_448576


namespace problem_proof_l448_448640

variable (k : ℝ) (a : ℝ) (m : ℝ) (p : ℝ × ℝ) (b : ℝ)

noncomputable def slope_of_line : ℝ := - (real.sqrt 3)

noncomputable def slope_angle (k : ℝ) : ℝ := real.atan k

noncomputable def new_slope_angle (a : ℝ) : ℝ := a / 4

noncomputable def slope_tangent (a : ℝ) : ℝ := real.tan a

noncomputable def equation_of_line_through_point (m : ℝ) (p : ℝ × ℝ) : string := 
  let (x1, y1) := p in
  "sqrt(3) * x - 3 * y - 6 = 0"

noncomputable def equation_of_line_with_y_intercept (m : ℝ) (b : ℝ) : string :=
  "sqrt(3) * x - 3 * y - 15 = 0"

theorem problem_proof :
  let k := slope_of_line in
  let a := slope_angle k in
  let new_slope_ang := new_slope_angle a in
  let m := slope_tangent new_slope_ang in
  (equation_of_line_through_point m (real.sqrt 3, -1) = "sqrt(3) * x - 3 * y - 6 = 0") ∧
  (equation_of_line_with_y_intercept m (-5) = "sqrt(3) * x - 3 * y - 15 = 0") :=
by
  sorry

end problem_proof_l448_448640


namespace reporters_percentage_l448_448275

theorem reporters_percentage (total_reporters : ℕ) (local_politics_percentage : ℝ) (non_politics_percentage : ℝ) :
  local_politics_percentage = 28 → non_politics_percentage = 60 → 
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  100 * (non_local_political_reporters / political_reporters) = 30 :=
by
  intros
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  sorry

end reporters_percentage_l448_448275


namespace terminating_decimal_count_l448_448267

theorem terminating_decimal_count :
  {n : ℤ | n ≥ 1 ∧ n ≤ 499 ∧ (∃ m : ℤ, n = 3 * m)}.finite ∧
  finset.card {n : ℤ | n ≥ 1 ∧ n ≤ 499 ∧ (∃ m : ℤ, n = 3 * m)}.to_finset = 166 :=
by sorry

end terminating_decimal_count_l448_448267


namespace opposite_number_in_circle_l448_448021

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l448_448021


namespace gretchen_charge_per_drawing_l448_448362

-- Given conditions
def sold_on_Saturday : ℕ := 24
def sold_on_Sunday : ℕ := 16
def total_amount : ℝ := 800
def total_drawings := sold_on_Saturday + sold_on_Sunday

-- Assertion to prove
theorem gretchen_charge_per_drawing (x : ℝ) (h : total_drawings * x = total_amount) : x = 20 :=
by
  sorry

end gretchen_charge_per_drawing_l448_448362


namespace total_number_of_tires_is_868_l448_448597

theorem total_number_of_tires_is_868 :
  ∃ (M T C : ℕ), 
  (M + T + C = 200) ∧
  (M = 1 / 5 * 200) ∧
  (T = 1 / 4 * 200) ∧
  (C = 200 - M - T) ∧
  (∀ x, x = 40 → (x * 2) = 80) ∧
  (∀ y, y = 50 → (y * 6) = 300) ∧
  (∀ z, z = 110 → (z * 4) = 440) ∧
  (∀ a, a = 110 → (1 / 3 * a).to_nat = 36) ∧
  (∀ b, b = 50 → (1 / 4 * b).to_nat = 12) ∧
  ((80 + 440 + 36) + (300 + 12) = 868) :=
sorry

end total_number_of_tires_is_868_l448_448597


namespace least_k_l448_448830

-- Definitions and conditions
def u : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 3 * u n - 3 * (u n)^2

def L : ℝ := 0.5

-- Statement of the theorem
theorem least_k (k : ℕ) : (∀ n, |u n - L| ≤ 1 / 2^20 ↔ n ≥ 0) → k = 0 :=
sorry

end least_k_l448_448830


namespace probability_first_queen_second_diamond_l448_448884

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l448_448884


namespace seating_arrangements_l448_448186

-- Define the problem's conditions
def adults := 3
def children := 3
def total_seats := 6

-- Define the statement to be proved
theorem seating_arrangements : 
  (∃ (seating : List (Bool × Nat)), 
    seating.length = total_seats ∧
    ∀ (i : Nat), i < total_seats → seating[i].fst = seating[(i + 1) % total_seats].fst → seating[i].fst = false) 
    → 
    -- Check if the number of valid seating arrangements is 12
    (number_of_valid_arrangements seating = 12) :=
sorry

end seating_arrangements_l448_448186


namespace convert_234_base5_to_binary_l448_448265

def base5_to_decimal (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 4 * 5^0

def decimal_to_binary (n : Nat) : List Nat :=
  let rec to_binary_aux (n : Nat) (accum : List Nat) : List Nat :=
    if n = 0 then accum
    else to_binary_aux (n / 2) ((n % 2) :: accum)
  to_binary_aux n []

theorem convert_234_base5_to_binary :
  (base5_to_decimal 234 = 69) ∧ (decimal_to_binary 69 = [1,0,0,0,1,0,1]) :=
by
  sorry

end convert_234_base5_to_binary_l448_448265


namespace max_value_pair_l448_448771

open Real

def f (x : ℝ) : ℝ := sqrt(x * (60 - x)) + sqrt(x * (5 - x))

theorem max_value_pair :
  ∃ x₀ M, f x₀ = M ∧ (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ M) ∧ x₀ = 60 / 13 ∧ M = 10 * sqrt 3 :=
by
  sorry

end max_value_pair_l448_448771


namespace greatest_value_of_b_l448_448285

theorem greatest_value_of_b : ∃ (b : ℝ), b = 9 ∧ (b^2 - 14*b + 45 ≤ 0) :=
by
  use 9
  split
  · rfl
  · sorry

end greatest_value_of_b_l448_448285


namespace sum_expression_l448_448789

/--
Suppose we have a sum defined as \( S \) given by:
\[
S = \sum_{n=1}^{4900} \frac{1}{\sqrt{n + \sqrt{n^2 + 1}}}
\]
and \( S \) can be expressed in the form \( p + q \sqrt{r} \),
where \( p, q, r \) are positive integers and \( r \) is not divisible by the square 
of any prime. Then we need to prove that \( p + q + r = 107 \).
-/
theorem sum_expression (S p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (∀ n, (1 ≤ n ∧ n ≤ 4900) → (1 / (sqrt (n + sqrt (n^2 + 1)))) = S) → 
  (∃ p q r: ℕ, r ≠ 0 ∧ ∀ k, prime k → k^2 ∣ r → false ∧ S = p + q * sqrt r) → 
  p + q + r = 107 := 
sorry

end sum_expression_l448_448789


namespace xiao_gao_top_25_proof_l448_448662

-- Given data
def num_students_A : ℕ := 50
def median_score_A : ℕ := 83
def xiao_gao_score : ℕ := 84

-- Predicate to check if Xiao Gao ranks in the top 25
def xiao_gao_top_25 : Prop :=
  ∀ scores : Fin num_students_A → ℕ, 
  median (scores) = median_score_A →
  (num_students_A / 2 < 50 → xiao_gao_score > median_score_A → ∃ i : Fin num_students_A, scores i = xiao_gao_score → i < 25)

-- Claim
theorem xiao_gao_top_25_proof : xiao_gao_top_25 := 
  sorry

end xiao_gao_top_25_proof_l448_448662


namespace probability_DEQ_greater_l448_448572

-- A point Q is randomly chosen inside an equilateral triangle DEF.
-- Prove that the probability that the area of triangle DEQ is greater than the areas of DFQ and EFQ is 1/3.

theorem probability_DEQ_greater (DEF : Triangle) (Q : Point) (h_equilateral : is_equilateral DEF) :
  (randomly_chosen_point_inside DEF Q) → 
  (probability (area (triangle DEQ Q DEF) > area (triangle DFQ Q DEF) ∧ 
                area (triangle DEQ Q DEF) > area (triangle EFQ Q DEF)) = 1/3) :=
by 
  sorry

end probability_DEQ_greater_l448_448572


namespace number_of_sheets_l448_448969

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l448_448969


namespace two_planes_intersect_if_one_common_point_l448_448535

-- Definitions and assumptions
variables {Plane : Type} [HasIntersect Plane Plane (set Point)]
variables {Point : Type}
variables {Line : Type} [HasIntersect Line Plane (set Point)]
variables exists_common_point : ∀ (P1 P2 : Plane), ∃ (p : Point), p ∈ P1 ∩ P2 → ∃ l, ∀ q, (q ∈ l ↔ q ∈ P1 ∧ q ∈ P2)

-- Correct statement to prove
theorem two_planes_intersect_if_one_common_point (P1 P2 : Plane) (p : Point) 
  (h : p ∈ P1 ∧ p ∈ P2) : 
  ∃ l, ∀ q, q ∈ l ↔ q ∈ P1 ∧ q ∈ P2 :=
exists_common_point P1 P2 sorry

end two_planes_intersect_if_one_common_point_l448_448535


namespace opposite_number_on_circle_l448_448031

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l448_448031


namespace equation_of_tangent_and_normal_at_t0_l448_448929

def x (t : ℝ) : ℝ := 2 * Real.log (Real.cot t) + Real.cot t
def y (t : ℝ) : ℝ := Real.tan t + Real.cot t

theorem equation_of_tangent_and_normal_at_t0 :
  let t0 := π / 4 in
  let x0 := x t0 in
  let y0 := y t0 in
  (y0 = 2 ∧ x0 = 1) := by
  sorry

end equation_of_tangent_and_normal_at_t0_l448_448929


namespace tangent_normal_lines_at_t0_l448_448645

noncomputable def parametric_curve_x (a t : ℝ) : ℝ := a * (t - Real.sin t)
noncomputable def parametric_curve_y (a t : ℝ) : ℝ := a * (1 - Real.cos t)

theorem tangent_normal_lines_at_t0 (a : ℝ) 
  (x := parametric_curve_x a)
  (y := parametric_curve_y a)
  (t0 := π / 3) :
  (∃ tangent_eq : ℝ → ℝ, 
    ∀ x_val, tangent_eq x_val = √3 * x_val - (√3 * π / 3 - 2) * a) ∧
  (∃ normal_eq : ℝ → ℝ, 
    ∀ x_val, normal_eq x_val = -x_val / √3 + a * π / (3 * √3)) :=
  by sorry

end tangent_normal_lines_at_t0_l448_448645


namespace scientific_notation_correct_l448_448840

theorem scientific_notation_correct : 657000 = 6.57 * 10^5 :=
by
  sorry

end scientific_notation_correct_l448_448840


namespace assign_roles_l448_448571

theorem assign_roles :
  let men := 6 in
  let women := 7 in
  let male_role := 1 in
  let female_roles := 2 in
  let either_gender_roles := 3 in
  let specific_man_roles := 1 in
  -- Calculating number of ways to assign each type of role
  let ways_to_assign_male_role := men in
  let ways_to_assign_female_roles := women * (women - 1) in
  let remaining_actors := (men - specific_man_roles) + women in
  let ways_to_assign_remaining_either_gender_roles := remaining_actors * (remaining_actors - 1) in
  ways_to_assign_male_role * ways_to_assign_female_roles * specific_man_roles * ways_to_assign_remaining_either_gender_roles = 27720 :=
by {
  -- Intermediate steps would go here, but are omitted as per instructions
  sorry
}

end assign_roles_l448_448571


namespace total_money_shared_l448_448985

theorem total_money_shared (A B C : ℕ) (rA rB rC : ℕ) (bens_share : ℕ) 
  (h_ratio : rA = 2 ∧ rB = 3 ∧ rC = 8)
  (h_ben : B = bens_share)
  (h_bensShareGiven : bens_share = 60) : 
  (rA * (bens_share / rB)) + bens_share + (rC * (bens_share / rB)) = 260 :=
by
  -- sorry to skip the proof
  sorry

end total_money_shared_l448_448985


namespace range_of_a_l448_448351

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then real.exp (-x) else -x^2 - 2*x + 1

theorem range_of_a : {a : ℝ | f (a - 1) ≥ f (-a^2 + 1)} = set.Icc (-2 : ℝ) 1 :=
by sorry

end range_of_a_l448_448351


namespace successive_discounts_final_price_l448_448181

noncomputable def initial_price : ℝ := 10000
noncomputable def discount1 : ℝ := 0.20
noncomputable def discount2 : ℝ := 0.10
noncomputable def discount3 : ℝ := 0.05

theorem successive_discounts_final_price :
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let final_selling_price := price_after_second_discount * (1 - discount3)
  final_selling_price = 6840 := by
  sorry

end successive_discounts_final_price_l448_448181


namespace rate_of_interest_increased_l448_448090

noncomputable def initial_simple_interest (P : ℕ) (A₁ : ℕ) (T : ℕ) : ℝ := 
  ((A₁ - P) * 100) / (P * T)

noncomputable def new_simple_interest (P : ℕ) (A₂ : ℕ) (T : ℕ) : ℝ := 
  ((A₂ - P) * 100) / (P * T)

noncomputable def percentage_increase (R₁ R₂ : ℝ) : ℝ :=
  ((R₂ - R₁) / R₁) * 100

theorem rate_of_interest_increased : 
  let P := 900
  let A₁ := 956
  let A₂ := 1064
  let T := 3
  let R₁ := initial_simple_interest P A₁ T
  let R₂ := new_simple_interest P A₂ T
  percentage_increase R₁ R₂ ≈ 192.9 := 
by
  sorry

end rate_of_interest_increased_l448_448090


namespace a_2015_value_l448_448299

noncomputable def floor_sum (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n+1), Nat.floor ((n + 2^(k-1)) / 2^k)

theorem a_2015_value : floor_sum 2015 = 2015 := 
  sorry

end a_2015_value_l448_448299


namespace rhombus_characterization_l448_448161

variable (Q : Type) [quadrilateral Q] [diagonals_perpendicular Q] [diagonals_bisect_each_other Q]

theorem rhombus_characterization {Q : Type} [quadrilateral Q] [diagonals_perpendicular Q] [diagonals_bisect_each_other Q] : rhombus Q :=
sorry

end rhombus_characterization_l448_448161


namespace parabola_constant_term_l448_448194

theorem parabola_constant_term :
  ∃ b c : ℝ, (∀ x : ℝ, (x = 2 → 3 = x^2 + b * x + c) ∧ (x = 4 → 3 = x^2 + b * x + c)) → c = 11 :=
by
  sorry

end parabola_constant_term_l448_448194


namespace asymptotes_of_hyperbola_l448_448703

-- Define the hyperbola and conditions
variables (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
def hyperbola : Prop := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
def eccentricity : ℝ := real.sqrt 5

-- Given the conditions and hyperbola, prove that the asymptotes are y = ±2x
theorem asymptotes_of_hyperbola :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → (b^2 / a^2 = 4)) →
  (y = 2 * x ∨ y = -2 * x) :=
begin
  sorry
end

end asymptotes_of_hyperbola_l448_448703


namespace trigonometric_inequality_for_tan_l448_448076

open Real

theorem trigonometric_inequality_for_tan (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) : 
  1 + tan x < 1 / (1 - sin x) :=
sorry

end trigonometric_inequality_for_tan_l448_448076


namespace train_crossing_time_l448_448592

/-- Given the speed of the train in km/hr and the length of the train in meters,
    prove that the time it takes to cross a pole is approximately 12 seconds. -/
theorem train_crossing_time (v : ℝ) (L : ℝ) (h1 : v = 60) (h2 : L = 200) :
  L / ((v * 1000) / 3600) ≈ 12 :=
by
  sorry

end train_crossing_time_l448_448592


namespace part_one_part_two_l448_448455

noncomputable def probability_of_person_B_winning (x y z : ℕ) (h : x + y + z = 6) : ℚ :=
  let p_red := (3 / 6) * (x / 6)
  let p_yellow := (2 / 6) * (y / 6)
  let p_blue := (1 / 6) * (z / 6)
  p_red + p_yellow + p_blue

noncomputable def average_score_for_person_B (x y z : ℕ) (h : x + y + z = 6) : ℚ :=
  (3 * x + 4 * y + 3 * z) / 36

theorem part_one (h : (1 : ℕ) + 2 + 3 = 6) :
  probability_of_person_B_winning 1 2 3 h = 5 / 18 := sorry

theorem part_two (h₁ : (1 : ℕ) + 4 + 1 = 6) (h₂ : ∀ x y z, x + y + z = 6 → (average_score_for_person_B x y z) ≤ 11 / 18) :
  average_score_for_person_B 1 4 1 h₁ = 11 / 18 := by
    sorry

end part_one_part_two_l448_448455


namespace lattice_points_on_graph_l448_448363

theorem lattice_points_on_graph :
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 53}.card = 4 :=
by
  sorry

end lattice_points_on_graph_l448_448363


namespace total_hats_l448_448146

theorem total_hats (B G : ℕ) (cost_blue cost_green total_cost green_quantity : ℕ)
  (h1 : cost_blue = 6)
  (h2 : cost_green = 7)
  (h3 : total_cost = 530)
  (h4 : green_quantity = 20)
  (h5 : G = green_quantity)
  (h6 : total_cost = B * cost_blue + G * cost_green) :
  B + G = 85 :=
by
  sorry

end total_hats_l448_448146


namespace smallest_positive_odd_integer_n_l448_448906

theorem smallest_positive_odd_integer_n (n : ℕ) (h_odd : n % 2 = 1) : 
  (∏ k in finset.range (n + 1), 3^(bit0 k + 1 : ℚ) / 5) > 500 ↔ n = 6 := 
begin
  sorry
end

end smallest_positive_odd_integer_n_l448_448906


namespace translation_correctness_l448_448509

theorem translation_correctness :
  ( ∀ (x : ℝ), ((x + 4)^2 - 5) = ((x + 4)^2 - 5) ) :=
by
  sorry

end translation_correctness_l448_448509


namespace trigonometric_identity_l448_448270

theorem trigonometric_identity :
  sin (real.toRadians (-10)) * cos (real.toRadians 160) - 
  sin (real.toRadians 80) * sin (real.toRadians 200) = 
  1/2 :=
by 
  sorry

end trigonometric_identity_l448_448270


namespace T_shape_exists_in_remaining_grid_l448_448659

theorem T_shape_exists_in_remaining_grid (grid_size : ℕ) (dominoes_removed : ℕ)
  (h_grid_size : grid_size = 100) (h_dominoes_removed : dominoes_removed = 1950) : 
  ∃ T_shape, T_shape.present_in_remaining_grid grid_size dominoes_removed :=
by
  sorry

end T_shape_exists_in_remaining_grid_l448_448659


namespace fraction_evaluation_l448_448550

def number_of_primes_between_10_and_30 : ℕ := 6

theorem fraction_evaluation : (number_of_primes_between_10_and_30^2 - 4) / (number_of_primes_between_10_and_30 + 2) = 4 := by
  sorry

end fraction_evaluation_l448_448550


namespace aaron_earnings_l448_448716

def time_worked_monday := 75 -- in minutes
def time_worked_tuesday := 50 -- in minutes
def time_worked_wednesday := 145 -- in minutes
def time_worked_friday := 30 -- in minutes
def hourly_rate := 3 -- dollars per hour

def total_minutes_worked := 
  time_worked_monday + time_worked_tuesday + 
  time_worked_wednesday + time_worked_friday

def total_hours_worked := total_minutes_worked / 60

def total_earnings := total_hours_worked * hourly_rate

theorem aaron_earnings :
  total_earnings = 15 := by
  sorry

end aaron_earnings_l448_448716


namespace area_of_region_W_l448_448464

structure Rhombus (P Q R T : Type) :=
  (side_length : ℝ)
  (angle_Q : ℝ)

def Region_W
  (P Q R T : Type)
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) : ℝ :=
6.25

theorem area_of_region_W
  {P Q R T : Type}
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) :
  Region_W P Q R T r h_side h_angle = 6.25 :=
sorry

end area_of_region_W_l448_448464


namespace visited_neither_count_l448_448546

theorem visited_neither_count (total group iceland norway both : ℕ)
  (group_50 : group = 50)
  (iceland_25 : iceland = 25)
  (norway_23 : norway = 23)
  (both_21 : both = 21) :
  total = 23 :=
by 
  have at_least_one := iceland + norway - both,
  have total := group - at_least_one,
  simp [group_50, iceland_25, norway_23, both_21] at *,
  sorry

end visited_neither_count_l448_448546


namespace intersection_points_on_incircle_l448_448173

-- Definitions based on given conditions
variables {A B C D M N P Q : Type} 

-- Given a parallelogram ABCD
def is_parallelogram (A B C D : Type) : Prop :=
  (∀ (x y z w: Type), (x = A ∧ y = B ∧ z = C ∧ w = D) → (x + y = z + w) ∧ (x + w = y + z))

-- Ex-circle of triangle ABD tangency points
def ex_circle_tangency (A B D M N : Type) : Prop :=
  (∀ (x y z v w : Type), (x = A ∧ y = B ∧ z = D ∧ v = M ∧ w = N) →
  (extend_side z w = v) ∧ (extend_side x y = w))

-- Intersection points of MN with BC and CD
def intersection_points (M N B C D P Q : Type) : Prop :=
  (∀ (v w x y z : Type), (v = M ∧ w = N ∧ x = B ∧ y = C ∧ z = D) →
  (intersect v w y x = P) ∧ (intersect v w z y = Q))

-- Incircle of triangle BCD touching points P and Q
def on_incircle (B C D P Q : Type) : Prop :=
  (∀ (x y z v w : Type), (x = B ∧ y = C ∧ z = D ∧ v = P ∧ w = Q) → 
  (incircle_touches_side x y v) ∧ (incircle_touches_side x z w))

-- Main theorem statement
theorem intersection_points_on_incircle
  (A B C D M N P Q : Type)
  (h_parallelogram : is_parallelogram A B C D)
  (h_ex_circle : ex_circle_tangency A B D M N)
  (h_intersection : intersection_points M N B C D P Q) :
  on_incircle B C D P Q :=
sorry

end intersection_points_on_incircle_l448_448173


namespace cone_height_of_semicircular_sheet_l448_448898

theorem cone_height_of_semicircular_sheet (R h : ℝ) (h_cond: h = R) : h = R :=
by
  exact h_cond

end cone_height_of_semicircular_sheet_l448_448898


namespace correct_calculation_l448_448912

theorem correct_calculation (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end correct_calculation_l448_448912


namespace quadratic_roots_conjugate_imaginary_l448_448685

noncomputable def is_geom_seq (p a q : ℝ) : Prop := a^2 = p * q
noncomputable def is_arith_seq (p b c q : ℝ) : Prop := ∃ d : ℝ, p = b - 2 * d ∧ c = b + d ∧ q = b + 2 * d

theorem quadratic_roots_conjugate_imaginary (a b c p q : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hpq : 0 < p ∧ 0 < q)
  (hne : p ≠ q) (hgeom : is_geom_seq p a q) (harith : is_arith_seq p b c q) :
  let Δ := 4 * (a^2) - 4 * b * c in Δ < 0 :=
by
  sorry

end quadratic_roots_conjugate_imaginary_l448_448685


namespace opposite_number_113_is_13_l448_448033

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l448_448033


namespace solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l448_448635

-- Definitions only directly appearing in the conditions problem
def consecutive_integers (x y z : ℤ) : Prop := x = y - 1 ∧ z = y + 1
def consecutive_even_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 0
def consecutive_odd_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 1

-- Problem Statements
theorem solvable_consecutive_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_integers x y z :=
sorry

theorem solvable_consecutive_even_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_even_integers x y z :=
sorry

theorem not_solvable_consecutive_odd_integers : ¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_odd_integers x y z :=
sorry

end solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l448_448635


namespace math_problem_l448_448302

theorem math_problem (x : ℝ) :
  (x^3 - 8*x^2 + 16*x > 64) ∧ (x^2 - 4*x + 5 > 0) → x > 4 :=
by
  sorry

end math_problem_l448_448302


namespace assignment_time_l448_448770

variable (t : ℝ) 

theorem assignment_time (h1 : 2 * t) (h2 : (3 / 4)) (h3 : t + 2 * t + (3 / 4) = 2) : t = 5 / 12 := 
by 
  -- Apply provided conditions and solve
  sorry

end assignment_time_l448_448770


namespace total_growing_space_is_correct_l448_448988

def garden_bed_area (length : ℕ) (width : ℕ) (count : ℕ) : ℕ :=
  length * width * count

def total_growing_space : ℕ :=
  garden_bed_area 5 4 3 +
  garden_bed_area 6 3 4 +
  garden_bed_area 7 5 2 +
  garden_bed_area 8 4 1

theorem total_growing_space_is_correct :
  total_growing_space = 234 := by
  sorry

end total_growing_space_is_correct_l448_448988


namespace monotonically_increasing_interval_l448_448230

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l448_448230


namespace sum_of_two_numbers_l448_448121

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) :
  x + y = 16 * sqrt 3 / 3 :=
by sorry

end sum_of_two_numbers_l448_448121


namespace larger_segment_proof_l448_448865

noncomputable def larger_segment_cut_off (x h : ℝ) (cond_1 : x^2 + h^2 = 40^2) (cond_2 : (100 - x)^2 + h^2 = 75^2) : ℝ :=
  100 - x

theorem larger_segment_proof :
  (∃ x h : ℝ, x^2 + h^2 = 40^2 ∧ (100 - x)^2 + h^2 = 75^2) → larger_segment_cut_off 70.125 :=
  sorry

end larger_segment_proof_l448_448865


namespace sum_of_n_for_perfect_square_l448_448420

theorem sum_of_n_for_perfect_square (n : ℕ) (Sn : ℕ) 
  (hSn : Sn = n^2 + 20 * n + 12) 
  (hn : n > 0) :
  ∃ k : ℕ, k^2 = Sn → (sum_of_possible_n = 16) :=
by
  sorry

end sum_of_n_for_perfect_square_l448_448420


namespace opposite_number_113_is_114_l448_448043

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l448_448043


namespace geom_seq_formula_sum_geom_seq_formula_l448_448322

noncomputable def a_n (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def S_n (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 1) * 2^(k + 1))

theorem geom_seq_formula : ∀ n, a_n n = 2^(n - 1) :=
by
  sorry

theorem sum_geom_seq_formula (n : ℕ) :
  S_n n = 2 + (n - 1) * 2^(n + 1) :=
by
  sorry

end geom_seq_formula_sum_geom_seq_formula_l448_448322


namespace monotonic_increasing_interval_log_l448_448856

theorem monotonic_increasing_interval_log (x : ℝ) (hx : -1 < x ∧ x < 1) :
  ∃ I : set ℝ, I = set.Ioo (-1) 0 ∧ ∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I ∧ x1 < x2 → 
  log (2:ℝ) (1 - x1^2) ≤ log (2:ℝ) (1 - x2^2) :=
sorry

end monotonic_increasing_interval_log_l448_448856


namespace exists_pos_x_i_l448_448656

theorem exists_pos_x_i (n : ℕ) (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  ∃ x : Fin n → ℝ, (∀ i, 0 < x i) ∧ (∑ i, x i = 1) ∧
    (∀ y : Fin n → ℝ, (∀ i, 0 < y i) → (∑ i, y i = 1) →
      ∑ i, (a i * x i) / (x i + y i) ≥ (1 / 2) * ∑ i, a i) :=
sorry

end exists_pos_x_i_l448_448656


namespace triangle_angle_y_l448_448904

theorem triangle_angle_y (y : ℝ) (h1 : 2 * y + (y + 10) + 4 * y = 180) : 
  y = 170 / 7 := 
by
  sorry

end triangle_angle_y_l448_448904


namespace correct_quotient_l448_448395

/-- Given that the remainder is 0, the incorrect divisor is 12,
the correct divisor is 21, and the quotient using the incorrect 
divisor is 63, the correct quotient is 36. -/
theorem correct_quotient (d D q N Q : ℕ) (h1 : d = 12) (h2 : D = 21) 
    (h3 : q = 63) (h4 : N = d * q) (h5 : N = D * Q) : Q = 36 :=
by
  rw [h1, h2, h3] at h4 h5
  sorry

end correct_quotient_l448_448395


namespace percentage_discount_l448_448454

/-- Pearl orders 5 monogrammed stockings each for her 5 grandchildren and 4 children.
Each stocking is priced at $20 before any discount. Monogramming costs an additional $5 per stocking.
The total amount paid after the discount is $1035. The task is to prove that the percentage discount applied is 8%. -/
theorem percentage_discount (n_grandchildren n_children : ℕ) (n_stockings_per_person: ℕ) (stocking_price monogram_price total_paid: ℝ) :
  n_grandchildren = 5 → n_children = 4 → n_stockings_per_person = 5 → stocking_price = 20 → monogram_price = 5 → total_paid = 1035 →
  let total_stockings := (n_grandchildren + n_children) * n_stockings_per_person in
  let original_price := total_stockings * stocking_price in
  let monogramming_cost := total_stockings * monogram_price in
  let total_cost_before_discount := original_price + monogramming_cost in
  let discount := total_cost_before_discount - total_paid in
  let percentage_discount := (discount / total_cost_before_discount) * 100 in
  percentage_discount = 8 :=
by intros;
   simp only [] at *;
   have : total_stockings = 45, by sorry;
   have : original_price = 900, by sorry;
   have : monogramming_cost = 225, by sorry;
   have : total_cost_before_discount = 1125, by sorry;
   have : discount = 90, by sorry;
   have : percentage_discount = 8, by sorry
 
end percentage_discount_l448_448454


namespace symmetry_of_graph_l448_448376

theorem symmetry_of_graph (f : ℝ → ℝ) (h : ∀ x, f(x) = f(3 - x)) : 
  ∀ x y, (y = f(x)) ↔ (y = f(1.5 - (x - 1.5))) := 
sorry

end symmetry_of_graph_l448_448376


namespace triangular_angles_l448_448339

noncomputable def measure_of_B (A : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 + Real.sin (2 * A)) / 3))

noncomputable def length_of_c (A : ℝ) : ℝ := 
  Real.sqrt (22 - 6 * Real.sqrt 13 * Real.cos (measure_of_B A))

noncomputable def area_of_triangle_ABC (A : ℝ) : ℝ := 
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * A)) / 3)

theorem triangular_angles 
  (a b c : ℝ) (b_pos : b = Real.sqrt 13) (a_pos : a = 3) (h : b * Real.cos c = (2 * a - c) * Real.cos (measure_of_B c)) :
  c = length_of_c c ∧
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * c)) / 3) = area_of_triangle_ABC c :=
by
  sorry

end triangular_angles_l448_448339


namespace total_ice_cream_sales_l448_448812

theorem total_ice_cream_sales (tuesday_sales : ℕ) (h1 : tuesday_sales = 12000)
    (wednesday_sales : ℕ) (h2 : wednesday_sales = 2 * tuesday_sales) :
    tuesday_sales + wednesday_sales = 36000 := by
  -- This is the proof statement
  sorry

end total_ice_cream_sales_l448_448812


namespace find_root_of_quadratic_equation_l448_448653

theorem find_root_of_quadratic_equation
  (a b c : ℝ)
  (h1 : 3 * a * (2 * b - 3 * c) ≠ 0)
  (h2 : 2 * b * (3 * c - 2 * a) ≠ 0)
  (h3 : 5 * c * (2 * a - 3 * b) ≠ 0)
  (r : ℝ)
  (h_roots : (r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) ∨ (r = (-2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c))) * 2)) :
  r = -2 * b * (3 * c - 2 * a) / (9 * a * (2 * b - 3 * c)) :=
by
  sorry

end find_root_of_quadratic_equation_l448_448653


namespace complement_union_l448_448359

open Set

variable (α : Type) [Fintype α] [DecidableEq α]

def U : Set α := {1, 2, 3, 4, 5, 6}
def A : Set α := {1, 2, 3, 4}
def B : Set α := {1, 3, 5}

theorem complement_union :
  (U \ (A ∪ B)) = {6} :=
by
  sorry

end complement_union_l448_448359


namespace integer_multiple_properties_l448_448075

theorem integer_multiple_properties (k : ℕ) (hk : k > 1) :
  ∃ w : ℕ, k ∣ w ∧ w < k^4 ∧ (number_of_unique_digits w ≤ 4) :=
sorry

-- Auxiliary function to count the number of unique digits in a number.
def number_of_unique_digits (n : ℕ) : ℕ := sorry

end integer_multiple_properties_l448_448075


namespace number_of_sheets_in_stack_l448_448967

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l448_448967


namespace no_two_identical_rows_l448_448916

noncomputable def table_filled_correctly (f : ℕ → ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i < n, f 1 i = 1) ∧
  (∀ i j < n, (f (i + 1) j + f (i + 1) (j + 1) + f i (j + 1) + f i j) % 2 = 0)

theorem no_two_identical_rows
  (f : ℕ → ℕ → ℕ)
  (n : ℕ)
  (h1 : ∀ i < n, f 1 i = 1)
  (h2 : ∀ i j < n, (f (i + 1) j + f (i + 1) (j + 1) + f i (j + 1) + f i j) % 2 = 0) :
  ¬ ∃ (r1 r2 : ℕ), r1 < n ∧ r2 < n ∧ r1 ≠ r2 ∧ ∀ j < n, f r1 j = f r2 j :=
  sorry

end no_two_identical_rows_l448_448916


namespace ratio_of_second_to_tallest_building_l448_448868

theorem ratio_of_second_to_tallest_building 
  (T S U V : ℝ) 
  (hT : T = 100)
  (hU : U = ½ * S) 
  (hV : V = ⅕ * U) 
  (hSum : T + S + U + V = 180) : 
  S / T = ½ := 
by
  sorry

end ratio_of_second_to_tallest_building_l448_448868


namespace projection_is_circumcenter_l448_448196

-- Define a point in three-dimensional space
structure Point (ℝ : Type) :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define a plane using a point and a normal vector
structure Plane (ℝ : Type) :=
  (point : Point ℝ)
  (normal : Point ℝ)

-- Define what it means for a point to be equidistant from three vertices of a triangle
def equidistant_from_vertices (P : Point ℝ) (A B C : Point ℝ) :=
  dist P A = dist P B ∧ dist P C = dist P B

-- Define the projection of a point onto a plane
def projection_onto_plane (P : Point ℝ) (α : Plane ℝ) : Point ℝ := sorry

-- Define the circumcenter condition for a point being equidistant from vertices of a triangle
def is_circumcenter (O A B C : Point ℝ) :=
  dist O A = dist O B ∧ dist O C = dist O B

-- Define the main theorem to be proved
theorem projection_is_circumcenter 
  {P A B C : Point ℝ} {α : Plane ℝ} 
  (h₀ : equidistant_from_vertices P A B C)
  (h₁ : P ∉ α) :
  is_circumcenter (projection_onto_plane P α) A B C := 
sorry

end projection_is_circumcenter_l448_448196


namespace find_cd_sum_l448_448373

theorem find_cd_sum : 
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ (30 + c) * (10 * d + 4) = 146 ∧ c + d = 7 :=
by {
  use [4, 3], -- example values
  split,
  -- Prove c < 10
  { linarith, },
  split,
  -- Prove d < 10
  { linarith, },
  split,
  -- Prove (30 + c) * (10 * d + 4) = 146
  { norm_num, sorry },
  -- Prove c + d = 7
  { norm_num, },
}

end find_cd_sum_l448_448373


namespace polar_circle_area_l448_448118

theorem polar_circle_area (rho theta : ℝ) (h : rho = 2 * cos theta) : 
  ∃ r : ℝ, r = 1 ∧ π * r^2 = π :=
by
  sorry

end polar_circle_area_l448_448118


namespace determine_p_x_l448_448266

theorem determine_p_x : 
  ∃ p : ℚ[X], (5 * X^4 + 3 * X^2 - 7 * X + 2 + p = 2 * X^3 + 4 * X^2 - 1) 
  ∧ p = -5 * X^4 + 2 * X^3 + X^2 + 7 * X - 3 :=
by
  use -5 * X^4 + 2 * X^3 + X^2 + 7 * X - 3
  simp
  sorry

end determine_p_x_l448_448266


namespace minimum_value_of_a_l448_448321

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp (3 * Real.log x - x)) - x^2 - (a - 4) * x - 4

theorem minimum_value_of_a (h : ∀ x > 0, f x ≤ 0) : a ≥ 4 / Real.exp 2 := by
  sorry

end minimum_value_of_a_l448_448321


namespace opposite_number_113_is_114_l448_448060

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l448_448060


namespace monotonically_increasing_interval_l448_448228

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l448_448228


namespace triangle_proof_l448_448389

noncomputable def triangle_problem: Prop :=
  let a := 3
  let cos_B := 7 / 9
  let dot_product := 7
  ∃ (c b : ℝ), 
    c = 3 ∧ 
    b = 2 ∧ 
    ∃ (sin_A : ℝ), sin_A = 2 * Real.sqrt(2) / 3 ∧
      ∃ (sin_B : ℝ), sin_B = 4 * Real.sqrt(2) / 9 ∧
      ∃ (cos_A : ℝ), cos_A = 1 / 3 ∧
        sin_A * cos_B - cos_A * sin_B = 10 * Real.sqrt (2) / 27
  
theorem triangle_proof : triangle_problem :=
begin
  sorry
end

end triangle_proof_l448_448389


namespace can_split_3x3x3_into_9_corners_l448_448174

-- Define the conditions
def number_of_cubes_in_3x3x3 : ℕ := 27
def number_of_units_in_corner : ℕ := 3
def number_of_corners : ℕ := 9

-- Prove the proposition
theorem can_split_3x3x3_into_9_corners :
  (number_of_corners * number_of_units_in_corner = number_of_cubes_in_3x3x3) :=
by
  sorry

end can_split_3x3x3_into_9_corners_l448_448174


namespace possible_values_of_t_l448_448790

theorem possible_values_of_t
  (theta : ℝ) 
  (x y t : ℝ) :
  x = Real.cos theta →
  y = Real.sin theta →
  t = (Real.sin theta) ^ 2 + (Real.cos theta) ^ 2 →
  x^2 + y^2 = 1 →
  t = 1 := by
  sorry

end possible_values_of_t_l448_448790


namespace train_crossing_time_l448_448595

-- Definitions of given constants
def speed_kmh : ℝ := 60 -- Speed of train in km/hr
def length_m : ℝ := 200 -- Length of train in meters

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Speed in m/s
def speed_ms : ℝ := speed_kmh * conversion_factor

-- Time calculation
def time_sec : ℝ := length_m / speed_ms

-- The proof statement
theorem train_crossing_time : time_sec = 12 :=
by
  -- The proof will be provided here
  sorry

end train_crossing_time_l448_448595


namespace find_m_n_l448_448637

theorem find_m_n (m n : ℕ) (h1 : m ≥ 0) (h2 : n ≥ 0) (h3 : 3^m - 7^n = 2) : m = 2 ∧ n = 1 := 
sorry

end find_m_n_l448_448637


namespace one_fourth_of_7point2_is_9_over_5_l448_448639

noncomputable def oneFourthOf (x : ℝ) : ℝ := x / 4
noncomputable def toSimplifiedFraction (x : ℝ) : ℚ :=
  let frac := ((x * 10 : ℚ), 10) in
  let gcd := frac.1.natAbs.gcd frac.2.natAbs in
  (↑(frac.1.natAbs / gcd) / ↑(frac.2.natAbs / gcd))

theorem one_fourth_of_7point2_is_9_over_5 : toSimplifiedFraction (oneFourthOf 7.2) = 9 / 5 := by
  sorry

end one_fourth_of_7point2_is_9_over_5_l448_448639


namespace regular_train_passes_by_in_4_seconds_l448_448568

theorem regular_train_passes_by_in_4_seconds
    (l_high_speed : ℕ)
    (l_regular : ℕ)
    (t_observed : ℕ)
    (v_relative : ℕ)
    (h_length_high_speed : l_high_speed = 80)
    (h_length_regular : l_regular = 100)
    (h_time_observed : t_observed = 5)
    (h_velocity : v_relative = l_regular / t_observed) :
    v_relative * 4 = l_high_speed :=
by
  sorry

end regular_train_passes_by_in_4_seconds_l448_448568


namespace max_marks_mike_could_have_got_l448_448447

theorem max_marks_mike_could_have_got (p : ℝ) (m_s : ℝ) (d : ℝ) (M : ℝ) :
  p = 0.30 → m_s = 212 → d = 13 → 0.30 * M = (212 + 13) → M = 750 :=
by
  intros hp hms hd heq
  sorry

end max_marks_mike_could_have_got_l448_448447


namespace prob_not_blue_is_9_over_14_l448_448730

-- Define the odds of drawing a blue marble
def odds_blue (success failure : ℕ) : ℕ × ℕ := (success, failure)

-- Define the total number of outcomes given the odds
def total_outcomes (odds : ℕ × ℕ) : ℕ := odds.1 + odds.2

-- Define the probability of not drawing a blue marble given the odds
def prob_not_blue (odds : ℕ × ℕ) : ℚ :=
  let total := total_outcomes odds in
  odds.2 / total

-- Given odds
def given_odds : ℕ × ℕ := odds_blue 5 9

-- The theorem to prove
theorem prob_not_blue_is_9_over_14 : prob_not_blue given_odds = 9 / 14 := by
  sorry

end prob_not_blue_is_9_over_14_l448_448730


namespace lines_parallel_to_same_line_are_parallel_l448_448083

-- Define a type for lines
constant Line : Type

-- Define the notion of parallel lines as a predicate
constant parallel : Line → Line → Prop

-- State the theorem
theorem lines_parallel_to_same_line_are_parallel 
  (A B C : Line) 
  (hAC : parallel A C) 
  (hBC : parallel B C) : 
  parallel A B :=
sorry

end lines_parallel_to_same_line_are_parallel_l448_448083


namespace area_of_triangle_MNQ_l448_448402

-- Define the points on the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the given conditions
def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 0⟩
def C : Point := ⟨2, 4⟩
def D : Point := ⟨0, 4⟩

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def M : Point := midpoint B C
def N : Point := midpoint C D
def P : Point := midpoint D A
def Q : Point := midpoint N P

-- Function to calculate the area of a triangle given three points
def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

-- Prove that the area of triangle MNQ is 1
theorem area_of_triangle_MNQ : triangle_area M N Q = 1 := by
  sorry

end area_of_triangle_MNQ_l448_448402


namespace part_a_part_b_part_c_part_d_l448_448524

-- Part (a): Explain why 3^{2^{n}} - 2^{2^{n}} can be written as the product of n distinct integers all greater than 1.
theorem part_a (n : ℕ) (h1 : n ≥ 1) : 
  ∃ (a : list ℕ), (∀ x ∈ a, x > 1) ∧ (a.nodup) ∧ list.prod a = 3^(2^n) - 2^(2^n) := sorry

-- Part (b): Verify that 3^{2^{n}} - 1 = (3^{2^{n-1}} + 1) (3^{2^{n-2}} + 1) · · · (3^2 + 1) (3 + 1) (3 - 1).
theorem part_b (n : ℕ) (h1 : n ≥ 1) : 
  3^(2^n) - 1 = list.prod (list.map (λ i, if i = 0 then 3 - 1 else 3^(2^i) + 1) (list.range (n+1))) := sorry

-- Part (c): Verify that 2^{n+1} is a divisor of 3^{2^{n}} - 1.
theorem part_c (n : ℕ) (h1 : n ≥ 1) : 
  2^(n+1) ∣ 3^(2^n) - 1 := sorry

-- Part (d): Find a positive integer n with more than 2018 positive divisors such that 3^{n-1} - 2^{n-1} is a multiple of n.
theorem part_d : 
  ∃ n : ℕ, (n > 2018) ∧ (∃ d : ℕ, d ∣ 3^(n-1) - 2^(n-1)) := sorry

end part_a_part_b_part_c_part_d_l448_448524


namespace sin_double_angle_l448_448372

theorem sin_double_angle {θ : ℝ} (h : Real.tan θ = 1 / 3) : 
  Real.sin (2 * θ) = 3 / 5 := 
  sorry

end sin_double_angle_l448_448372


namespace constant_term_expansion_l448_448481

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∃ k : ℝ, k = -21/2 ∧
  (∀ r : ℕ, (9 : ℕ).choose r * (x^(1/2))^(9-r) * ((-(1/(2*x)))^r) = k) :=
sorry

end constant_term_expansion_l448_448481


namespace banana_cost_correct_l448_448869

variable (total_cost bread_cost milk_cost apple_cost banana_cost : ℕ)

axiom totalCost : total_cost = 42
axiom breadCost : bread_cost = 9
axiom milkCost : milk_cost = 7
axiom appleCost : apple_cost = 14

theorem banana_cost_correct (h1 : total_cost = 42) (h2 : bread_cost = 9) (h3 : milk_cost = 7)
    (h4 : apple_cost = 14) : banana_cost = total_cost - (bread_cost + milk_cost + apple_cost) := 
by
  calc
    banana_cost = 42 - (9 + 7 + 14) := sorry
    ... = 12                        := sorry

end banana_cost_correct_l448_448869


namespace problem_equivalence_l448_448693

-- Definitions of the conditions
def ellipse (a b : ℝ) (h : a > b > 0) : ℝ → ℝ → Prop :=
  λ x y, x^2 / a^2 + y^2 / b^2 = 1

def parabola (h : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y^2 = 4 * h * x

def line (m x1 : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y = m * (x - x1)

def chord_length (C : ℝ → ℝ → Prop) (L : ℝ → ℝ → Prop) : ℝ := 
  -- Placeholder function, the length calculation logic is skipped
  sorry


noncomputable def right_focus_and_coincides (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b > 0) : Prop :=
  ∃ (h_2 : 2 > 0), 
  ellipse a b h 2 0 ∧ parabola 2 2 0 ∧ ellipse a b h (-(sqrt 6)) 0

noncomputable def ellipse_eq (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b > 0) : Prop :=
  ellipse a b h x y ↔ (x^2 / 6 + y^2 / 2 = 1)

noncomputable def length_of_chord (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b > 0) : Prop :=
  chord_length (ellipse a b h) (line (sqrt 3) 2) = (4 * sqrt 6) / 5

theorem problem_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b > 0) :
  right_focus_and_coincides a b ha hb h →
  ellipse_eq a b ha hb h ∧ length_of_chord a b ha hb h :=
by
  sorry

end problem_equivalence_l448_448693


namespace betty_gave_stuart_percentage_l448_448601

theorem betty_gave_stuart_percentage (P : ℝ) 
  (betty_marbles : ℝ := 60) 
  (stuart_initial_marbles : ℝ := 56) 
  (stuart_final_marbles : ℝ := 80)
  (increase_in_stuart_marbles : ℝ := stuart_final_marbles - stuart_initial_marbles)
  (betty_to_stuart : ℝ := (P / 100) * betty_marbles) :
  56 + ((P / 100) * betty_marbles) = 80 → P = 40 :=
by
  intros h
  -- Sorry is used since the proof steps are not required
  sorry

end betty_gave_stuart_percentage_l448_448601


namespace total_ice_cream_sales_l448_448811

theorem total_ice_cream_sales (tuesday_sales : ℕ) (h1 : tuesday_sales = 12000)
    (wednesday_sales : ℕ) (h2 : wednesday_sales = 2 * tuesday_sales) :
    tuesday_sales + wednesday_sales = 36000 := by
  -- This is the proof statement
  sorry

end total_ice_cream_sales_l448_448811


namespace middle_car_is_Karen_l448_448654

-- Definitions for the positions
structure Person :=
  (name : String)

-- Positions of people in the train
structure Train :=
  (car1 : Person)
  (car2 : Person)
  (car3 : Person)
  (car4 : Person)
  (car5 : Person)

-- The problem conditions represented in Lean
def valid_configuration (train : Train) : Prop :=
  let Aaron := {name := "Aaron"}
  let Darren := {name := "Darren"}
  let Karen := {name := "Karen"}
  let Maren := {name := "Maren"}
  let Sharon := {name := "Sharon"} in

  -- Conditions given in the problem
  (train.car1 = Aaron) ∧
  (train.car2 ≠ Sharon ∧ train.car3 ≠ Sharon) ∧ Sharon ∈ [train.car3, train.car4, train.car5] ∧
  (train.car2 ≠ Maren ∧ train.car3 ≠ Maren) ∧ Maren ∈ [train.car3, train.car4, train.car5] ∧
  (train.car2 = Darren → train.car3 = Karen ∨ train.car3 = Darren → train.car4 = Karen ∨ train.car4 = Darren → train.car5 = Karen) ∧
  ¬(Karen ∈ [train.car2, train.car3] ∧ Maren ∈ [train.car2, train.car3]) ∧
  ¬(Karen ∈ [train.car3, train.car4] ∧ Maren ∈ [train.car3, train.car4]) ∧
  ¬(Karen ∈ [train.car4, train.car5] ∧ Maren ∈ [train.car4, train.car5])

-- The proposition we aim to prove
theorem middle_car_is_Karen (train : Train) (h : valid_configuration train) : train.car3.name = "Karen" :=
by
  intros
  sorry

end middle_car_is_Karen_l448_448654


namespace product_of_digits_l448_448380

theorem product_of_digits (A B : ℕ) (h1 : A + B = 14) (h2 : (10 * A + B) % 4 = 0) : A * B = 48 :=
by
  sorry

end product_of_digits_l448_448380


namespace machine_no_repair_l448_448502

def nominal_mass (G_dev: ℝ) := G_dev / 0.1

theorem machine_no_repair (G_dev: ℝ) (σ: ℝ) (non_readable_dev_lt: ∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) : 
  (G_dev = 39) ∧ (σ ≤ G_dev) ∧ (∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) ∧ (G_dev ≤ 0.1 * nominal_mass(G_dev)) → 
  ¬ machine.requires_repair :=
by
  sorry

end machine_no_repair_l448_448502


namespace function_properties_l448_448109

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 1
axiom f_pos_gt_1 : ∀ x : ℝ, x > 0 → f(x) > 1
axiom f_three : f 3 = 4

theorem function_properties : (∀ x1 x2 : ℝ, x1 > x2 → f(x1) > f(x2)) ∧ (f 1 = 2) := by
  sorry

end function_properties_l448_448109


namespace ratio_of_areas_l448_448566

variable (L W : ℝ)  -- Define original length and width as variables

-- Define the conditions as assumptions
def original_area := L * W
def new_length := 1.40 * L
def new_width := 0.80 * W
def new_area := new_length * new_width

-- The theorem to prove the ratio of areas
theorem ratio_of_areas : (new_area / original_area) = 1.12 :=
by
  sorry

end ratio_of_areas_l448_448566


namespace solve_inequality_l448_448469

theorem solve_inequality (x : ℝ) (h : x ≠ -2 / 3) :
  3 - (1 / (3 * x + 2)) < 5 ↔ (x < -7 / 6 ∨ x > -2 / 3) := by
  sorry

end solve_inequality_l448_448469


namespace wise_man_reasons_black_cap_l448_448962

theorem wise_man_reasons_black_cap
  (caps : List ℕ)
  (black white : ℕ)
  (num_black : caps.count black = 3)
  (num_white : caps.count white = 2)
  (num_wise_men : ∀ (l : List ℕ), l.length = 5 → l.count black + l.count white = 5 → l) 
  (each_wise_man_has_cap : ∀ (i : ℕ), i < 3 → ∃ (c : ℕ), c ∈ caps)
  (start_silent : ∀ (i : ℕ), i < 3 → sees_two_black_caps i)
  :
  ∃ (i : ℕ), i < 3 ∧ concludes_black_cap i := 
sorry

end wise_man_reasons_black_cap_l448_448962


namespace smallest_positive_integer_satisfying_conditions_l448_448156

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (x : ℕ),
    x % 4 = 1 ∧
    x % 5 = 2 ∧
    x % 7 = 3 ∧
    ∀ y : ℕ, (y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3) → y ≥ x ∧ x = 93 :=
by
  sorry

end smallest_positive_integer_satisfying_conditions_l448_448156


namespace remainder_when_dividing_698_by_13_is_9_l448_448910

theorem remainder_when_dividing_698_by_13_is_9 :
  ∃ k m : ℤ, 242 = k * 13 + 8 ∧
             698 = m * 13 + 9 ∧
             (k + m) * 13 + 4 = 940 :=
by {
  sorry
}

end remainder_when_dividing_698_by_13_is_9_l448_448910


namespace monotonic_increasing_l448_448235

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π / 6)

theorem monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 2) → f x1 < f x2 := sorry

end monotonic_increasing_l448_448235


namespace greatest_multiple_of_four_cubed_less_than_2000_l448_448475

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ x, (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 2000) ∧ ∀ y, (y > x) ∧ (y % 4 = 0) → y^3 ≥ 2000 :=
sorry

end greatest_multiple_of_four_cubed_less_than_2000_l448_448475


namespace max_non_triangle_lines_l448_448669

theorem max_non_triangle_lines (n : ℕ) (h : n ≥ 3) (h_collinear : ∀ (A B C : Type), ¬(collinear A B C)) :
  ∃ k, k = if even n then n^2 / 4 else (n^2 - 1) / 4 := by
  sorry

end max_non_triangle_lines_l448_448669


namespace probability_queen_then_diamond_l448_448895

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l448_448895


namespace blue_eyes_blonde_hair_logic_l448_448414

theorem blue_eyes_blonde_hair_logic :
  ∀ (a b c d : ℝ), 
  (a / (a + b) > (a + c) / (a + b + c + d)) →
  (a / (a + c) > (a + b) / (a + b + c + d)) :=
by
  intro a b c d h
  sorry

end blue_eyes_blonde_hair_logic_l448_448414


namespace eventually_495_or_0_l448_448873

-- Define the three-digit number and its reverse
def digits (abc : ℕ) (a b c : ℕ) : Prop :=
  abc = 100 * a + 10 * b + c

def reverse (abc cba : ℕ) (a b c : ℕ) : Prop :=
  cba = 100 * c + 10 * b + a

-- Define the subtraction condition
def valid_subtraction (abc cba : ℕ) : Prop :=
  abc > cba ∨ cba > abc

-- Define the result of the subtraction
def result (abc cba res : ℕ) : Prop :=
  res = (if abc > cba then abc - cba else cba - abc)

-- Define the eventual number being 495 or 0
def eventual_result (res : ℕ) : Prop :=
  res = 495 ∨ res = 0

-- Main theorem stating the eventual result after repeated subtraction
theorem eventually_495_or_0 (abc cba res : ℕ) (a b c : ℕ) :
  digits abc a b c →
  reverse abc cba a b c →
  valid_subtraction abc cba →
  (∃ k : ℕ, (result abc cba (res k)) ∧ eventual_result (res k)) :=
sorry

end eventually_495_or_0_l448_448873


namespace total_birds_l448_448874

theorem total_birds (g d : Nat) (h₁ : g = 58) (h₂ : d = 37) : g + d = 95 :=
by
  sorry

end total_birds_l448_448874


namespace sum_of_consecutive_integers_eq_pow_of_two_l448_448279

theorem sum_of_consecutive_integers_eq_pow_of_two (n : ℕ) : 
  (∀ a b : ℕ, a < b → 2 * n ≠ (a + b) * (b - a + 1)) ↔ ∃ k : ℕ, n = 2 ^ k := 
sorry

end sum_of_consecutive_integers_eq_pow_of_two_l448_448279


namespace incircle_touches_ratios_l448_448110

theorem incircle_touches_ratios
  (A B C D F H K : Point)
  (incircle : Circle)
  (tangents : TangentSegment incircle D F)
  (AD_intersects : LineSegment A D H)
  (CF_intersects : LineSegment C F K) :
  \(\frac{distance F D \times distance H K}{distance F H \times distance D K} = 5\) :=
sorry

end incircle_touches_ratios_l448_448110


namespace lines_per_page_l448_448419

theorem lines_per_page
  (total_words : ℕ)
  (words_per_line : ℕ)
  (words_left : ℕ)
  (pages_filled : ℚ) :
  total_words = 400 →
  words_per_line = 10 →
  words_left = 100 →
  pages_filled = 1.5 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end lines_per_page_l448_448419


namespace ideal_skew_lines_pairs_in_cube_l448_448727

/-- Define a cube and the classifications of connecting lines -/
structure Cube :=
  (edges : ℕ)
  (face_diagonals : ℕ)
  (space_diagonals : ℕ)

def cube : Cube := {
  edges := 12,
  face_diagonals := 12,
  space_diagonals := 4
}

def is_ideal_skew_lines_pair (l1 l2 : ℝ) : Prop :=
  l1 ≠ l2 ∧ l1 ⊥ l2

theorem ideal_skew_lines_pairs_in_cube : 
  cube.edges = 12 ∧ cube.face_diagonals = 12 ∧ cube.space_diagonals = 4 →
  ∃ n : ℕ, n = 78 := 
by {
  sorry
}

end ideal_skew_lines_pairs_in_cube_l448_448727


namespace prime_count_between_90_and_100_l448_448719

open Nat

-- Define the range and condition on prime numbers
def range := {n : ℕ | 90 < n ∧ n < 100}

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Count the number of prime numbers in the given range
def count_primes_between_90_and_100 : ℕ :=
  count (λ n => n ∈ range ∧ is_prime n) (list.filter (λ n => 90 < n ∧ n < 100) (list.range 101))

-- The theorem we need to prove
theorem prime_count_between_90_and_100 : count_primes_between_90_and_100 = 1 :=
by
  sorry

end prime_count_between_90_and_100_l448_448719


namespace statement_A_statement_B_statement_C_statement_D_l448_448670

theorem statement_A (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
    a^2 + b^2 ≥ 1/2 :=
sorry

theorem statement_B (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
    2^(a - b) > 1/2 :=
sorry

theorem statement_C (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
    ¬ (log 2 a + log 2 b ≥ -2) :=
sorry

theorem statement_D (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
    sqrt a + sqrt b ≤ sqrt 2 :=
sorry

end statement_A_statement_B_statement_C_statement_D_l448_448670


namespace altitude_from_A_to_BC_l448_448326

theorem altitude_from_A_to_BC (x y : ℝ) : 
  (3 * x + 4 * y + 12 = 0) ∧ 
  (4 * x - 3 * y + 16 = 0) ∧ 
  (2 * x + y - 2 = 0) → 
  (∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1 / 2) ∧ (b = 2 - 8)) :=
by 
  sorry

end altitude_from_A_to_BC_l448_448326


namespace hyperbola_range_of_m_l448_448750

theorem hyperbola_range_of_m (m : ℝ) : (∃ f : ℝ → ℝ → ℝ, ∀ x y: ℝ, f x y = (x^2 / (4 - m) - y^2 / (2 + m))) → (4 - m) * (2 + m) > 0 → -2 < m ∧ m < 4 :=
by
  intros h_eq h_cond
  sorry

end hyperbola_range_of_m_l448_448750


namespace slope_of_line_l448_448126

theorem slope_of_line (x y : ℝ) (h : x + 2 * y + 1 = 0) : y = - (1 / 2) * x - (1 / 2) :=
by
  sorry -- The solution would be filled in here

#check slope_of_line -- additional check to ensure theorem implementation is correct

end slope_of_line_l448_448126


namespace underground_renovation_length_l448_448137

theorem underground_renovation_length (
  total_length : ℕ,
  total_length_eq : total_length = 3600,
  increased_speed_factor : ℝ,
  increased_speed_factor_eq : increased_speed_factor = 1.2,
  ahead_days : ℕ,
  ahead_days_eq : ahead_days = 10
) : (let x := real.sqrt 360000 / 10 in x * increased_speed_factor = 72) :=
by
  sorry

end underground_renovation_length_l448_448137


namespace achieve_target_ratio_l448_448936

-- Initial volume and ratio
def initial_volume : ℕ := 20
def initial_milk_ratio : ℕ := 3
def initial_water_ratio : ℕ := 2

-- Mixture removal and addition
def removal_volume : ℕ := 10
def added_milk : ℕ := 10

-- Target ratio of milk to water
def target_milk_ratio : ℕ := 9
def target_water_ratio : ℕ := 1

-- Number of operations required
def operations_needed: ℕ := 2

-- Statement of proof problem
theorem achieve_target_ratio :
  (initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) + added_milk * operations_needed) / 
  (initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) = target_milk_ratio :=
sorry

end achieve_target_ratio_l448_448936


namespace distance_from_S_to_PQR_correct_l448_448460

noncomputable def distance_from_S_to_PQR (S P Q R : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  let SP := distance S P
  let SQ := distance S Q
  let SR := distance S R
  let PQ := distance P Q
  let PR := distance P R
  let QR := distance Q R
  have h_SP_SQ := orthogonal SP SQ
  have h_SP_SR := orthogonal SP SR
  have h_SQ_SR := orthogonal SQ SR
  have h_SP := SP = 15 := by sorry
  have h_SQ := SQ = 15 := by sorry
  have h_SR := SR = 9 := by sorry
  3 * sqrt 26

theorem distance_from_S_to_PQR_correct {S P Q R : EuclideanSpace ℝ (Fin 3)}
  (h_SP_or: orthogonal S P)
  (h_SQ_or: orthogonal S Q)
  (h_SR_or: orthogonal S R)
  (h_SP_len: distance S P = 15)
  (h_SQ_len: distance S Q = 15)
  (h_SR_len: distance S R = 9) :
  distance_from_S_to_PQR S P Q R = 3 * sqrt 26 :=
by {
  sorry
}

end distance_from_S_to_PQR_correct_l448_448460


namespace find_line_equation_l448_448644

noncomputable def line_equation (x y : ℝ) := 2 * x + 3 * y - 7 = 0

theorem find_line_equation :
  (∃ x y : ℝ, 2 * x - y - 3 = 0 ∧ 4 * x - 3 * y - 5 = 0) →
  (∃ k : ℝ, ∀ x y, 2 * x + 3 * y + k = 0 →
    (2 * x - y - 3 = 0 ∧ 4 * x - 3 * y - 5 = 0 →
    2 * x + 3 * y - 7 = 0)) :=
begin
  sorry
end

end find_line_equation_l448_448644


namespace max_expression_value_l448_448406

theorem max_expression_value : 
  ∃ (a b c d : ℕ), 
    a ∈ {0, 1, 2, 3} ∧ b ∈ {0, 1, 2, 3} ∧ c ∈ {0, 1, 2, 3} ∧ d ∈ {0, 1, 2, 3} ∧ 
    (∀ (x y z w : ℕ), 
      x ∈ {0, 1, 2, 3} ∧ y ∈ {0, 1, 2, 3} ∧ z ∈ {0, 1, 2, 3} ∧ w ∈ {0, 1, 2, 3} → 
      c * a^b - d ≥ z * x^y - w) ∧
    c * a^b - d = 9 := 
by 
  sorry

end max_expression_value_l448_448406


namespace factors_diff_square_l448_448005

theorem factors_diff_square (N : ℕ) (hN : N = 2019 ^ 2 - 1) :
  (let num_factors (n : ℕ) : ℕ := n.factors.card in
   num_factors (N ^ 2) - num_factors N = 157) :=
by
  sorry

end factors_diff_square_l448_448005


namespace min_weighings_to_identify_counterfeit_coins_l448_448871

def coin := ℕ

def is_genuine (c : coin) : Prop := sorry
def is_counterfeit (c : coin) : Prop := sorry

theorem min_weighings_to_identify_counterfeit_coins (coins : list coin) 
  (h_length : coins.length = 6) 
  (h_genuine : ∃ (S : finset coin), S.card = 4 ∧ ∀ c ∈ S, is_genuine c) 
  (h_counterfeit : ∃ (S : finset coin), S.card = 2 ∧ ∀ c ∈ S, is_counterfeit c) :
  ∃ min_weighings : ℕ, min_weighings = 3 := sorry

end min_weighings_to_identify_counterfeit_coins_l448_448871


namespace speed_of_mrs_a_l448_448799

theorem speed_of_mrs_a
  (distance_between : ℝ)
  (speed_mr_a : ℝ)
  (speed_bee : ℝ)
  (distance_bee_travelled : ℝ)
  (time_bee : ℝ)
  (remaining_distance : ℝ)
  (speed_mrs_a : ℝ) :
  distance_between = 120 ∧
  speed_mr_a = 30 ∧
  speed_bee = 60 ∧
  distance_bee_travelled = 180 ∧
  time_bee = distance_bee_travelled / speed_bee ∧
  remaining_distance = distance_between - (speed_mr_a * time_bee) ∧
  speed_mrs_a = remaining_distance / time_bee →
  speed_mrs_a = 10 := by
  sorry

end speed_of_mrs_a_l448_448799


namespace sufficient_and_necessary_condition_l448_448584

theorem sufficient_and_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by 
  sorry

end sufficient_and_necessary_condition_l448_448584


namespace max_value_sin_cos_expression_max_area_perimeter_l448_448756

-- Problem 1: Maximum value of given trigonometric expression
theorem max_value_sin_cos_expression (A B C a b c : ℝ) (h1 : a / (cos C * sin B) = b / sin B + c / cos C) :
  (B = π / 4) →
  (sin (A + B) + sin A * cos A + cos (A - B) ≤ 5 / 2) :=
by sorry

-- Problem 2: Perimeter of triangle when area is maximized
theorem max_area_perimeter (A B C a b c : ℝ) (h1 : a / (cos C * sin B) = b / sin B + c / cos C) (h2 : b = sqrt 2) :
  (2 = a^2 + c^2 - sqrt 2 * a * c) →
  (a = c) →
  (a = sqrt (2 + sqrt 2)) →
  (b = sqrt 2) →
  (a + b + c = 2 * sqrt (2 + sqrt 2) + sqrt 2) :=
by sorry

end max_value_sin_cos_expression_max_area_perimeter_l448_448756


namespace train_crossing_time_l448_448590

/-- Given the speed of the train in km/hr and the length of the train in meters,
    prove that the time it takes to cross a pole is approximately 12 seconds. -/
theorem train_crossing_time (v : ℝ) (L : ℝ) (h1 : v = 60) (h2 : L = 200) :
  L / ((v * 1000) / 3600) ≈ 12 :=
by
  sorry

end train_crossing_time_l448_448590


namespace midpoint_of_AB_l448_448663

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ
  
def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2,
    y := (A.y + B.y) / 2,
    z := (A.z + B.z) / 2 }

theorem midpoint_of_AB : 
  midpoint { x := 3, y := 2, z := 1 } { x := 1, y := 0, z := 5 } = { x := 2, y := 1, z := 3 } :=
by
  sorry

end midpoint_of_AB_l448_448663


namespace point_coordinates_l448_448320

noncomputable def parametric_curve (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) := (3 * Real.cos θ, 4 * Real.sin θ)

theorem point_coordinates (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) : 
  (Real.arcsin (4 * (Real.tan θ)) = π/4) → (3 * Real.cos θ, 4 * Real.sin θ) = (12 / 5, 12 / 5) :=
by
  sorry

end point_coordinates_l448_448320


namespace distance_between_parallel_lines_l448_448386

-- Define the lines and the condition of parallelism
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (1, a, 6)
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (a - 2, 3, 2 * a)

-- Prove the distance between the lines when they are parallel
theorem distance_between_parallel_lines (a : ℝ) (ha : 3 = a * (a - 2)) : 
  a = -1 → let d := |2 * a - 6| / sqrt ((a - 2)^2 + 3^2) in d = (8 * sqrt 2) / 3 :=
by
  sorry

end distance_between_parallel_lines_l448_448386


namespace opposite_113_eq_114_l448_448062

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l448_448062


namespace CE_parallel_AR_l448_448431

noncomputable theory

open scoped Classical

variables {A B C R M D E : Type*} [MetricSpace M]

structure Triangle (A B C : M) : Prop := (is_triangle : ∃ (tra : ∀ Δ : M, Δ ∈ {A, B, C} ∧ Δ ≠ B → ∃ (circ : M),⟦Δ - circ⟧ = ⟦A - circ⟧))

def is_circumcircle (Γ : M) (A B C : M) : Prop := ∀ Δ : M, Δ ∈ {A, B, C} →⟦Δ - Γ⟧ = ⟦A - Γ⟧

def tangent_at (Γ : M) (A : M) : M := sorry -- Definition for tangent at point A on circle Γ

def midpoint (A R : M) : M := sorry -- Definition for midpoint of segment AR

def line_intersection (P Q : M) (Γ : M) : M := sorry -- Definition for intersection of line PQ with circumcircle Γ

theorem CE_parallel_AR (ABC : Triangle A B C) (Γ : M) (H1 : is_circumcircle Γ A B C) 
  (R : M) (H2 : tangent_at Γ A = R) 
  (M : M) (H3 : M = midpoint R A)
  (D : M) (H4 : D = line_intersection M B Γ)
  (E : M) (H5 : E = line_intersection R D Γ) 
  : parallel (C E) (A R) := 
sorry

end CE_parallel_AR_l448_448431


namespace convert_deg_to_rad_l448_448625

theorem convert_deg_to_rad (deg : ℝ) (h : deg = -630) : deg * (Real.pi / 180) = -7 * Real.pi / 2 :=
by
  rw [h]
  simp
  sorry

end convert_deg_to_rad_l448_448625


namespace students_left_zoo_l448_448562

theorem students_left_zoo
  (students_first_class students_second_class : ℕ)
  (chaperones teachers : ℕ)
  (initial_individuals remaining_individuals : ℕ)
  (chaperones_left remaining_individuals_after_chaperones_left : ℕ)
  (remaining_students initial_students : ℕ)
  (H1 : students_first_class = 10)
  (H2 : students_second_class = 10)
  (H3 : chaperones = 5)
  (H4 : teachers = 2)
  (H5 : initial_individuals = students_first_class + students_second_class + chaperones + teachers) 
  (H6 : initial_individuals = 27)
  (H7 : remaining_individuals = 15)
  (H8 : chaperones_left = 2)
  (H9 : remaining_individuals_after_chaperones_left = remaining_individuals - chaperones_left)
  (H10 : remaining_individuals_after_chaperones_left = 13)
  (H11 : remaining_students = remaining_individuals_after_chaperones_left - teachers)
  (H12 : remaining_students = 11)
  (H13 : initial_students = students_first_class + students_second_class)
  (H14 : initial_students = 20) :
  20 - 11 = 9 :=
by sorry

end students_left_zoo_l448_448562


namespace iPod_final_cost_l448_448801

theorem iPod_final_cost (original_price : ℝ) (discount_fraction : ℝ) (sales_tax_rate : ℝ) :
  original_price = 128 →
  discount_fraction = 7/20 →
  sales_tax_rate = 0.09 →
  let discount_amount := original_price * discount_fraction in
  let sale_price := original_price - discount_amount in
  let tax := sale_price * sales_tax_rate in
  let final_cost := sale_price + tax in
  final_cost = 90.69 :=
begin
  intros,
  sorry
end

end iPod_final_cost_l448_448801


namespace f_is_increasing_l448_448217

noncomputable theory

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

open Set

theorem f_is_increasing (f : ℝ → ℝ) (I : set ℝ) (h : ∀ x ∈ I, ∃ y ∈ I, x < y) : is_monotonically_increasing f I := sorry

example : is_monotonically_increasing f (Icc 0 (Real.pi / 2)) :=
begin
  unfold is_monotonically_increasing,
  intros x y hx hy hxy,
  have : 7 * Real.cos (x - Real.pi / 6) > 0 := sorry,
  have : 7 * Real.cos (y - Real.pi / 6) > 0 := sorry,
  calc
    f x = 7 * Real.sin (x - Real.pi / 6) : rfl
    ... < 7 * Real.sin (y - Real.pi / 6) : sorry
    ... = f y : rfl,
end

end f_is_increasing_l448_448217


namespace f_is_increasing_l448_448216

noncomputable theory

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

open Set

theorem f_is_increasing (f : ℝ → ℝ) (I : set ℝ) (h : ∀ x ∈ I, ∃ y ∈ I, x < y) : is_monotonically_increasing f I := sorry

example : is_monotonically_increasing f (Icc 0 (Real.pi / 2)) :=
begin
  unfold is_monotonically_increasing,
  intros x y hx hy hxy,
  have : 7 * Real.cos (x - Real.pi / 6) > 0 := sorry,
  have : 7 * Real.cos (y - Real.pi / 6) > 0 := sorry,
  calc
    f x = 7 * Real.sin (x - Real.pi / 6) : rfl
    ... < 7 * Real.sin (y - Real.pi / 6) : sorry
    ... = f y : rfl,
end

end f_is_increasing_l448_448216


namespace closest_to_2009_l448_448779

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Define the sequence of positive numbers
noncomputable def b_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n i

noncomputable def c_n (n : ℕ) : ℝ := ∏ i in finset.range n, b_n i

theorem closest_to_2009 : ∀ n, b_n n + c_n n = 1 → n(n + 1) = 1980 :=
by
  -- placeholders for the detailed mathematical definitions required
  sorry

end closest_to_2009_l448_448779


namespace monotonic_interval_l448_448238

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin(x - Real.pi / 6)

def is_monotonically_increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem monotonic_interval : is_monotonically_increasing_interval f 0 (Real.pi / 2) :=
sorry

end monotonic_interval_l448_448238


namespace roses_left_unsold_l448_448253

theorem roses_left_unsold (cost_per_rose : ℕ) (initial_roses : ℕ) (total_earned : ℕ) 
  (h_cost : cost_per_rose = 7) (h_initial : initial_roses = 9) (h_earned : total_earned = 35) :
  initial_roses - (total_earned / cost_per_rose) = 4 :=
by
  rw [h_cost, h_initial, h_earned]
  norm_num
  -- sorry

end roses_left_unsold_l448_448253


namespace probability_queen_then_diamond_l448_448892

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l448_448892


namespace solution_inequality_l448_448733

theorem solution_inequality (m : ℝ) :
  (∀ x : ℝ, x^2 - (m+3)*x + 3*m < 0 ↔ m ∈ Set.Icc 3 (-1) ∪ Set.Icc 6 7) →
  m = -1/2 ∨ m = 13/2 :=
sorry

end solution_inequality_l448_448733


namespace range_of_a_l448_448711

open Set

noncomputable def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

theorem range_of_a (a : ℝ) : (1 ∉ setA a) → a < 1 :=
sorry

end range_of_a_l448_448711


namespace probability_first_queen_second_diamond_l448_448883

def is_queen (card : ℕ) : Prop :=
card = 12 ∨ card = 25 ∨ card = 38 ∨ card = 51

def is_diamond (card : ℕ) : Prop :=
card % 13 = 0

def first_card_queen (first_card : ℕ) (cards : finset ℕ) : Prop :=
is_queen first_card

def second_card_diamond (second_card : ℕ) (remaining_cards : finset ℕ) : Prop :=
is_diamond second_card

theorem probability_first_queen_second_diamond (cards : finset ℕ) (h : cards.card = 52) :
  ((\sum x in cards.filter is_queen, 1 / 52) * (\sum y in cards.filter is_diamond, 1 / 51)) = (1 / 52) :=
sorry

end probability_first_queen_second_diamond_l448_448883


namespace mr_willam_land_percentage_over_taxable_land_l448_448277

def total_tax_collected : ℝ := 3840
def tax_paid_by_mr_willam : ℝ := 480
def farm_tax_percentage : ℝ := 0.45

theorem mr_willam_land_percentage_over_taxable_land :
  (tax_paid_by_mr_willam / total_tax_collected) * 100 = 5.625 :=
by
  sorry

end mr_willam_land_percentage_over_taxable_land_l448_448277


namespace measure_of_angle_F_l448_448758

-- Define the measures of the angles in terms of real numbers
variables (F E D : ℝ)

-- Define the problem conditions
def angle_D_is_75 : Prop := D = 75
def angle_E_is_4F_plus_18 : Prop := E = 4 * F + 18
def sum_of_angles_triangle_DEF : Prop := F + E + D = 180

-- State the final proof problem
theorem measure_of_angle_F (hD : angle_D_is_75) (hE : angle_E_is_4F_plus_18) (hSum : sum_of_angles_triangle_DEF) : 
  F = 17.4 :=
sorry

end measure_of_angle_F_l448_448758


namespace card_prob_queen_diamond_l448_448889

theorem card_prob_queen_diamond : 
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob in
  total_prob = 18 / 221 :=
by
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob
  sorry

end card_prob_queen_diamond_l448_448889


namespace wine_age_proof_l448_448098

-- Definitions based on conditions
def Age_Carlo_Rosi : ℕ := 40
def Age_Twin_Valley : ℕ := Age_Carlo_Rosi / 4
def Age_Franzia : ℕ := 3 * Age_Carlo_Rosi

-- We'll use a definition to represent the total age of the three brands of wine.
def Total_Age : ℕ := Age_Franzia + Age_Carlo_Rosi + Age_Twin_Valley

-- Statement to be proven
theorem wine_age_proof : Total_Age = 170 :=
by {
  sorry -- Proof goes here
}

end wine_age_proof_l448_448098


namespace total_amount_shared_l448_448210

theorem total_amount_shared (a b c : ℕ) (h_ratio : a * 5 = b * 3) (h_ben : b = 25) (h_ratio_ben : b * 12 = c * 5) :
  a + b + c = 100 := by
  sorry

end total_amount_shared_l448_448210


namespace common_ratio_geometric_sequence_general_term_arithmetic_sequence_largest_positive_integer_m_l448_448094

variables {d : ℝ} {S : ℕ → ℝ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = ∑ i in Finset.range n, a (i + 1)

def forms_geometric_sequence (S1 S2 S4 : ℝ) : Prop :=
S1 * S4 = S2 * S2

-- Problem statements
theorem common_ratio_geometric_sequence {a : ℕ → ℝ} (h1 : is_arithmetic_sequence a d)
  (h2 : sum_of_first_n_terms a S)
  (h3 : forms_geometric_sequence (S 1) (S 2) (S 4)) :
  S 2 / S 1 = 4 :=
sorry

theorem general_term_arithmetic_sequence {a : ℕ → ℝ} (h1 : is_arithmetic_sequence a d)
  (h2 : sum_of_first_n_terms a S)
  (h3 : forms_geometric_sequence (S 1) (S 2) (S 4))
  (h4 : S 2 = 4) :
  a n = 2 * n - 1 :=
sorry

def sequence_b_n (a : ℕ → ℝ) (n : ℕ) : ℝ := 3 / (a n * a (n+1))

def sum_of_first_n_b_terms (a : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
∀ n, T n = ∑ k in Finset.range n, sequence_b_n a k

theorem largest_positive_integer_m {a : ℕ → ℝ} {T : ℕ → ℝ} (h1 : is_arithmetic_sequence a d)
  (h2 : sum_of_first_n_terms a S)
  (h3 : forms_geometric_sequence (S 1) (S 2) (S 4))
  (h4 : S 2 = 4)
  (h5 : ∀ n, T n = ∑ k in Finset.range n, sequence_b_n a k) :
  ∃ m ∈ ℕ, (∀ n, T n > m / 20) ∧ m = 19 :=
sorry

end common_ratio_geometric_sequence_general_term_arithmetic_sequence_largest_positive_integer_m_l448_448094


namespace deductive_reasoning_example_l448_448132

open Int Nat

theorem deductive_reasoning_example :
  (∃ x : ℤ, x = -3) →
  (∀ x : ℤ, x ∈ Set.univ → x ∈ ℕ) →
  False :=
begin
  intros h1 h2,
  cases h1 with x hx,
  rw hx at h2,
  have h : -3 ∈ ℕ := h2 (-3) (Set.mem_univ _),
  -- This is the contradiction since -3 is not a Nat, which implies False
  exact Nat.not_mem_of_int_neg this
end

end deductive_reasoning_example_l448_448132


namespace circles_inequality_l448_448433

theorem circles_inequality (n : ℕ) (h : n ≥ 3) (C : Fin n → Set ℝ) 
  (O : Fin n → ℝ) (h1 : ∀ i, Metric.ball (O i) 1 = C i) 
  (h2 : ∀ l : Set ℝ, (∀ i j : Fin n, i ≠ j → Set.Disjoint l (C i) ∨ Set.Disjoint l (C j))) :
  ∑ i j in {i | i < j}.Finset, 1 / Real.dist (O i) (O j) ≤ (↑n - 1) * Real.pi / 4 :=
sorry

end circles_inequality_l448_448433


namespace find_circle_radius_l448_448185

-- Define the given conditions
variables (A B C : Type) [Euclidean (triangle (triangle(A, B, C)))]
variables (alpha beta S : Real)
variables (a b c R : Real)

-- Define the angles and area
variables (BAC : angle A B C = alpha)
variables (ABC : angle B A C = beta)
variables (area_ABC : triangle_area A B C = S)

-- Statement to be proved
theorem find_circle_radius :
  ∃ R : Real, R = sqrt (S * sin (alpha + beta) / (2 * sin(alpha)^3 * sin(beta))) :=
sorry

end find_circle_radius_l448_448185


namespace lunch_break_duration_l448_448445

theorem lunch_break_duration (m a : ℝ) (L : ℝ) :
  (9 - L) * (m + a) = 0.6 → 
  (7 - L) * a = 0.3 → 
  (5 - L) * m = 0.1 → 
  L = 42 / 60 :=
by sorry

end lunch_break_duration_l448_448445


namespace probability_first_queen_second_diamond_l448_448885

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l448_448885


namespace average_production_rate_optimal_price_l448_448938

-- Define conditions and average production rate
variable (initial_production_rate now_production_rate : ℝ)
variable (additional_machines : ℝ := 50)
axiom production_increase : now_production_rate = initial_production_rate + additional_machines
axiom production_time_equivalence : 600 / now_production_rate = 450 / initial_production_rate

-- Prove the average number of machines produced per day now is 200
theorem average_production_rate : now_production_rate = 200 := by
  sorry

-- Define conditions for profit maximization
variable (current_cost selling_price : ℝ)
variable (price_increase unit_decrease : ℝ := 50)
variable (price_adjustment : ℝ)
variable profit_price : ℝ := selling_price + price_adjustment

axiom initial_conditions : current_cost = 600 ∧ selling_price = 900
axiom market_research : ∀ a : ℝ, price_adjustment = a → now_production_rate - (a / unit_decrease) = 200 - (2 * a / unit_decrease)

-- Define profit function
def profit (a : ℝ) : ℝ := (selling_price + a - current_cost) * (200 - (2 * a / unit_decrease))

-- Prove the price the factory should set to maximize the daily profit is 1000 yuan
theorem optimal_price : profit_price = 1000 := by
  sorry

end average_production_rate_optimal_price_l448_448938


namespace distance_between_points_l448_448902

theorem distance_between_points : 
  let x1 := 1
  let y1 := -3
  let x2 := 4
  let y2 := 6
  let distance := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  in distance = Real.sqrt 90 := 
by
  sorry

end distance_between_points_l448_448902


namespace roots_quartic_sum_l448_448783

theorem roots_quartic_sum (c d : ℝ) (h1 : c + d = 3) (h2 : c * d = 1) (hc : Polynomial.eval c (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) (hd : Polynomial.eval d (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) :
  c * d + c + d = 4 :=
by
  sorry

end roots_quartic_sum_l448_448783


namespace standard_deviation_does_not_require_repair_l448_448491

-- Definitions based on conditions
def greatest_deviation (d : ℝ) := d = 39
def nominal_mass (M : ℝ) := 0.1 * M = 39
def unreadable_measurement_deviation (d : ℝ) := d < 39

-- Theorems to be proved
theorem standard_deviation (σ : ℝ) (d : ℝ) (M : ℝ) :
  greatest_deviation d →
  nominal_mass M →
  unreadable_measurement_deviation d →
  σ ≤ 39 :=
by
  sorry

theorem does_not_require_repair (σ : ℝ) :
  σ ≤ 39 → ¬(machine_requires_repair) :=
by
  sorry

-- Adding an assumption that if σ ≤ 39, the machine does not require repair
axiom machine_requires_repair : Prop

end standard_deviation_does_not_require_repair_l448_448491


namespace quadratic_real_roots_condition_l448_448729

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x - 1 = 0) ↔ (a ≥ -4 ∧ a ≠ 0) :=
sorry

end quadratic_real_roots_condition_l448_448729


namespace probability_one_from_alice_and_one_from_bob_l448_448208

open BigOperators

theorem probability_one_from_alice_and_one_from_bob :
  let total_letters := 12 in
  let alice_letters := 5 in
  let bob_letters := 7 in
  2 * (alice_letters / total_letters * bob_letters / (total_letters - 1)) = (35 / 66) :=
by sorry

end probability_one_from_alice_and_one_from_bob_l448_448208


namespace opposite_number_on_circle_l448_448026

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l448_448026


namespace number_of_added_terms_l448_448142

theorem number_of_added_terms (k : ℕ) (hk : 1 < k) :
  ∑ i in finset.Ico (2^k) (2^(k+1)), 1 / i = 2^k := sorry

end number_of_added_terms_l448_448142


namespace inscribed_circle_radius_l448_448619

theorem inscribed_circle_radius (a b : ℝ) : 
  let c := Real.sqrt (a^2 + b^2) in 
  let A := 2 * a * b in 
  let P := 4 * c in 
  let r := A / P in
  r = (a * b) / (2 * Real.sqrt (a^2 + b^2)) := 
by
  sorry

end inscribed_circle_radius_l448_448619


namespace machine_does_not_require_repair_l448_448498

noncomputable def nominal_mass := 390 -- The nominal mass M is 390 grams

def greatest_deviation_preserved := 39 -- The greatest deviation among preserved measurements is 39 grams

def deviation_unread_measurements (x : ℕ) : Prop := x < 39 -- Deviations of unread measurements are less than 39 grams

def all_deviations_no_more_than := ∀ (x : ℕ), x ≤ 39 -- All deviations are no more than 39 grams

theorem machine_does_not_require_repair 
  (M : ℕ) 
  (h_nominal_mass : M = nominal_mass)
  (h_greatest_deviation : greatest_deviation_preserved ≤ 0.1 * M)
  (h_unread_deviations : ∀ (x : ℕ), deviation_unread_measurements x) 
  (h_all_deviations : all_deviations_no_more_than):
  true := -- Prove the machine does not require repair
sorry

end machine_does_not_require_repair_l448_448498


namespace opposite_number_113_is_114_l448_448059

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l448_448059


namespace ferry_routes_ratio_l448_448308

theorem ferry_routes_ratio :
  ∀ (D_P D_Q : ℝ) (speed_P time_P speed_Q time_Q : ℝ),
  speed_P = 8 →
  time_P = 3 →
  speed_Q = speed_P + 4 →
  time_Q = time_P + 1 →
  D_P = speed_P * time_P →
  D_Q = speed_Q * time_Q →
  D_Q / D_P = 2 :=
by sorry

end ferry_routes_ratio_l448_448308


namespace opposite_number_113_is_13_l448_448038

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l448_448038


namespace probability_first_queen_second_diamond_l448_448887

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l448_448887


namespace coefficient_a8_l448_448731

noncomputable def polynomial := 
  ∀ x : ℝ, ∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ),
  x^2 + x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
               a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
               a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + a_10 * (x + 1)^10

theorem coefficient_a8 : ∀ (x : ℝ), ∃ (a_8 : ℝ), 
  polynomial x ∧ a_8 = 45 :=
by
  sorry

end coefficient_a8_l448_448731


namespace final_number_correct_l448_448178

noncomputable def initial_number : ℝ := 1256
noncomputable def first_increase_rate : ℝ := 3.25
noncomputable def second_increase_rate : ℝ := 1.47

theorem final_number_correct :
  initial_number * first_increase_rate * second_increase_rate = 6000.54 := 
by
  sorry

end final_number_correct_l448_448178


namespace trajectory_of_Q_l448_448674

-- Define the moving point P on the circle
def pointP (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the fixed point M
def pointM : (ℝ × ℝ) := (-1, 2)

-- Define point Q with given condition
def pointQ (a b : ℝ) : Prop :=
  let (x, y) := (-2*a - 3, -2*b + 6) in
    pointP x y

-- Prove the trajectory equation of point Q
theorem trajectory_of_Q (a b : ℝ) (hQ : pointQ a b) :
  (a + 3/2)^2 + (b - 3)^2 = 1/4 :=
by
  sorry

end trajectory_of_Q_l448_448674


namespace student_chose_121_l448_448974

theorem student_chose_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := by
  sorry

end student_chose_121_l448_448974


namespace solve_for_x_l448_448630

noncomputable def equation (x : ℝ) : Prop :=
  real.cbrt ((2 + x) ^ 2) + 3 * real.cbrt ((2 - x) ^ 2) = 4 * real.cbrt (4 - x ^ 2)

theorem solve_for_x : {x : ℝ | equation x} = {0, 13 / 7} :=
by
  sorry

end solve_for_x_l448_448630


namespace domain_of_f_f_is_odd_f_increasing_on_positive_reals_l448_448346

def f (x : ℝ) : ℝ := x - 1/x

theorem domain_of_f : {x : ℝ // x ≠ 0} = Set.univ.diff {0} := 
sorry

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := 
sorry

theorem f_increasing_on_positive_reals : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 < f x2 := 
sorry

end domain_of_f_f_is_odd_f_increasing_on_positive_reals_l448_448346


namespace opposite_number_113_is_114_l448_448044

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l448_448044


namespace program_output_for_six_l448_448826

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- The theorem we want to prove
theorem program_output_for_six : factorial 6 = 720 := by
  sorry

end program_output_for_six_l448_448826


namespace conditional_probability_l448_448099

variable (P : ℕ → ℚ)
variable (A B : ℕ)

def EventRain : Prop := P A = 4/15
def EventWind : Prop := P B = 2/15
def EventBoth : Prop := P (A * B) = 1/10

theorem conditional_probability 
  (h1 : EventRain P A) 
  (h2 : EventWind P B) 
  (h3 : EventBoth P A B) 
  : (P (A * B) / P A) = 3 / 8 := 
by
  sorry

end conditional_probability_l448_448099


namespace cosine_law_sine_law_parallelogram_sides_and_diagonals_l448_448995

variables {α : Type*} [inner_product_space ℝ α]

-- Definitions for the triangle part
variables (a b c : ℝ) (A B C : α) (angle_alpha : ℝ)

axiom triangle_sides : dist B C = a ∧ dist C A = b ∧ dist A B = c
axiom angle_def : inner (C - A) (B - A) = b * c * real.cos angle_alpha

-- Theorem of Cosines for a triangle
theorem cosine_law (hα : triangle_sides) (hangle : angle_def) :
  a^2 = b^2 + c^2 - 2 * b * c * real.cos angle_alpha := sorry

-- Theorem of Sines for a triangle
theorem sine_law (Δ : ℝ) (hα : Δ = (sqrt( -a^4 - b^4 - c^4 + 2 * b^2 * c^2 + 2 * c^2 * a^2 + 2 * a^2 * b^2)) / 4) :
  ∃ Δ, ( Δ = (sqrt( -a^4 - b^4 - c^4 + 2 * b^2 * c^2 + 2 * c^2 * a^2 + 2 * a^2 * b^2)) / 4
  ∧ (frac_sin_alpha: (sin angle_alpha / a) = (2 * Δ / (b * c))) :=
sorry

-- Definitions for the parallelogram part
variables (m n : α)
variables (a b a_sq b_sq m_sq n_sq : ℝ)

axiom parallelogram_sides (AB AD : α) : AB = a ∧ AD = b
axiom diagonals_def (AC BD : α) : AC = AB + AD ∧ BD = AB - AD

-- Relationship between sides and diagonals of a parallelogram
theorem parallelogram_sides_and_diagonals (h₁ : parallelogram_sides AB AD = (a, b)) (h₂: diagonals_def AC BD = (AB + AD, AB - AD)) :
  m_sq = ∥AB + AD∥^2 ∧ n_sq = ∥AB - AD∥^2
  ∧ m_sq + n_sq = 2 * (a^2 + b^2) := sorry

end cosine_law_sine_law_parallelogram_sides_and_diagonals_l448_448995


namespace exists_real_number_lt_neg_one_l448_448536

theorem exists_real_number_lt_neg_one : ∃ (x : ℝ), x < -1 := by
  sorry

end exists_real_number_lt_neg_one_l448_448536


namespace smallest_z_minus_x_l448_448877

theorem smallest_z_minus_x
  (x y z : ℕ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x * y * z = nat.factorial 9)
  (hxy : x < y) (hyz : y < z) :
  z - x = 99 := sorry

end smallest_z_minus_x_l448_448877


namespace total_amount_proof_l448_448585

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem total_amount_proof (x_ratio y_ratio z_ratio : ℝ) (y_share : ℝ) 
  (h1 : y_ratio = 0.45) (h2 : z_ratio = 0.50) (h3 : y_share = 54) 
  : total_amount (y_share / y_ratio) y_share (z_ratio * (y_share / y_ratio)) = 234 :=
by
  sorry

end total_amount_proof_l448_448585


namespace standard_deviation_does_not_require_repair_l448_448492

-- Definitions based on conditions
def greatest_deviation (d : ℝ) := d = 39
def nominal_mass (M : ℝ) := 0.1 * M = 39
def unreadable_measurement_deviation (d : ℝ) := d < 39

-- Theorems to be proved
theorem standard_deviation (σ : ℝ) (d : ℝ) (M : ℝ) :
  greatest_deviation d →
  nominal_mass M →
  unreadable_measurement_deviation d →
  σ ≤ 39 :=
by
  sorry

theorem does_not_require_repair (σ : ℝ) :
  σ ≤ 39 → ¬(machine_requires_repair) :=
by
  sorry

-- Adding an assumption that if σ ≤ 39, the machine does not require repair
axiom machine_requires_repair : Prop

end standard_deviation_does_not_require_repair_l448_448492


namespace problem_spec_l448_448434

noncomputable def y : ℂ := Complex.cos (Real.pi / 9) + Complex.I * Complex.sin (Real.pi / 9)

theorem problem_spec:
  let y := y
  in let expr := (3 * y + y^3) * (3 * y^3 + y^9) * 
                 (3 * y^6 + y^18) * (3 * y^7 + y^21) * 
                 (3 * y^8 + y^24) * (3 * y^9 + y^27)
  in expr = -- The correct answer should be placed here
  sorry

end problem_spec_l448_448434


namespace find_a_l448_448720

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 8) (h3 : c = 4) : a = 0 :=
by
  sorry

end find_a_l448_448720


namespace monotonically_increasing_interval_l448_448229

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l448_448229


namespace probability_first_queen_second_diamond_l448_448881

def is_queen (card : ℕ) : Prop :=
card = 12 ∨ card = 25 ∨ card = 38 ∨ card = 51

def is_diamond (card : ℕ) : Prop :=
card % 13 = 0

def first_card_queen (first_card : ℕ) (cards : finset ℕ) : Prop :=
is_queen first_card

def second_card_diamond (second_card : ℕ) (remaining_cards : finset ℕ) : Prop :=
is_diamond second_card

theorem probability_first_queen_second_diamond (cards : finset ℕ) (h : cards.card = 52) :
  ((\sum x in cards.filter is_queen, 1 / 52) * (\sum y in cards.filter is_diamond, 1 / 51)) = (1 / 52) :=
sorry

end probability_first_queen_second_diamond_l448_448881


namespace f_is_increasing_l448_448219

noncomputable theory

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

open Set

theorem f_is_increasing (f : ℝ → ℝ) (I : set ℝ) (h : ∀ x ∈ I, ∃ y ∈ I, x < y) : is_monotonically_increasing f I := sorry

example : is_monotonically_increasing f (Icc 0 (Real.pi / 2)) :=
begin
  unfold is_monotonically_increasing,
  intros x y hx hy hxy,
  have : 7 * Real.cos (x - Real.pi / 6) > 0 := sorry,
  have : 7 * Real.cos (y - Real.pi / 6) > 0 := sorry,
  calc
    f x = 7 * Real.sin (x - Real.pi / 6) : rfl
    ... < 7 * Real.sin (y - Real.pi / 6) : sorry
    ... = f y : rfl,
end

end f_is_increasing_l448_448219


namespace tangent_road_product_l448_448927

-- Define the points on the semicircle
variables {P1 P2 P3 O A B : Type} [MetricSpace O]

-- Define distances between the gates
variables (dist : O → O → ℝ)

-- Axioms for the problem conditions
axiom semicircular_town : ∀ {P1 P2 P3 : O}, 
  (dist O P1 = dist O P2) ∧ -- all points are on the same semicircle with center O
  (∀ P3, dist O P3 > 0) ∧ -- P3 is a different point on the semicircle
  (∀ P1 P2 P3, ∃ A B, -- points of intersection of tangents
     dist P1 A = dist P2 B ∧ -- tangents to the semicircle intersect
     ∠P1 A P3 ⊥ ∠P2 B P3) -- angles formed are complementary

-- The theorem to prove
theorem tangent_road_product :
  ∀ {P1 P2 P3 : O}, semicircular_town →
  dist P1 P2 ^ 2 = dist P1 P3 * dist P3 P2 :=
by sorry

end tangent_road_product_l448_448927


namespace regular_polygon_sides_l448_448959

theorem regular_polygon_sides (angle : ℝ) (h : angle = 20) : 
  ∃ n : ℕ, n = 18 ∧ 360 / angle = n :=
by {
  use 18,
  split,
  { rfl },
  { rw h, norm_num },
}

end regular_polygon_sides_l448_448959


namespace f_is_increasing_l448_448215

noncomputable theory

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

open Set

theorem f_is_increasing (f : ℝ → ℝ) (I : set ℝ) (h : ∀ x ∈ I, ∃ y ∈ I, x < y) : is_monotonically_increasing f I := sorry

example : is_monotonically_increasing f (Icc 0 (Real.pi / 2)) :=
begin
  unfold is_monotonically_increasing,
  intros x y hx hy hxy,
  have : 7 * Real.cos (x - Real.pi / 6) > 0 := sorry,
  have : 7 * Real.cos (y - Real.pi / 6) > 0 := sorry,
  calc
    f x = 7 * Real.sin (x - Real.pi / 6) : rfl
    ... < 7 * Real.sin (y - Real.pi / 6) : sorry
    ... = f y : rfl,
end

end f_is_increasing_l448_448215


namespace average_N_between_2_5_and_4_7_l448_448901

theorem average_N_between_2_5_and_4_7 :
  let N_set := {N | 2 * 70 < N ∧ N < 4 * 70 / 7} in
  (∑ N in N_set, N) / (N_set.card : ℝ) = 34 :=
by
  sorry

end average_N_between_2_5_and_4_7_l448_448901


namespace squared_recursive_2a_n_plus_1_geometric_sequence_lg_2a_n_plus_1_general_term_and_Tn_sum_and_smallest_n_l448_448694

-- Definition of the sequences and functions
def squared_recursive_sequence (A : ℕ → ℝ) :=
  ∀ n, A (n + 1) = A n ^ 2

def f (x : ℝ) : ℝ := 2 * x^2 + 2 * x

-- Given conditions
def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else (5 ^ (2^(n-2)) - 1) / 2

def b (n : ℕ) : ℝ :=
  log ((2 * a n + 1) ^ 2) / log (2 * a n + 1)

noncomputable def sequence_T (n : ℕ) : ℝ :=
  ∏ i in finset.range n, (2 * a (i + 1) + 1)

noncomputable def sequence_S (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b (i + 1)

-- Proof statements
theorem squared_recursive_2a_n_plus_1 :
  squared_recursive_sequence (λ n, 2 * a n + 1) :=
sorry

theorem geometric_sequence_lg_2a_n_plus_1 :
  ∃ a1 r, (λ n, log (2 * a n + 1)) = (λ n, a1 * r ^ n) :=
sorry

theorem general_term_and_Tn :
  ∀ n, a n = (1 / 2) * (5 ^ 2^(n-1) - 1) ∧ sequence_T n = 5 ^ (2^n - 1) :=
sorry

theorem sum_and_smallest_n :
  ∀ n, sequence_S n = 2 * n - 2 + 1 / 2^(n-1) ∧ ∃ n_min, n_min = 1008 ∧ sequence_S n_min > 2012 :=
sorry

end squared_recursive_2a_n_plus_1_geometric_sequence_lg_2a_n_plus_1_general_term_and_Tn_sum_and_smallest_n_l448_448694


namespace find_k_value_l448_448315

variable {a b : ℝ}
variable (x : ℝ)
variable (k : ℝ)
variable (f : ℝ → ℝ)

noncomputable def f (x : ℝ) : ℝ := (b * x + 1) / (2 * x + a)

theorem find_k_value (h : ∀ x, f x * f (1 / x) = k) (h2 : a * b ≠ 2) : k = 1 / 4 :=
by
  sorry

end find_k_value_l448_448315


namespace standard_deviation_does_not_require_repair_l448_448493

-- Definitions based on conditions
def greatest_deviation (d : ℝ) := d = 39
def nominal_mass (M : ℝ) := 0.1 * M = 39
def unreadable_measurement_deviation (d : ℝ) := d < 39

-- Theorems to be proved
theorem standard_deviation (σ : ℝ) (d : ℝ) (M : ℝ) :
  greatest_deviation d →
  nominal_mass M →
  unreadable_measurement_deviation d →
  σ ≤ 39 :=
by
  sorry

theorem does_not_require_repair (σ : ℝ) :
  σ ≤ 39 → ¬(machine_requires_repair) :=
by
  sorry

-- Adding an assumption that if σ ≤ 39, the machine does not require repair
axiom machine_requires_repair : Prop

end standard_deviation_does_not_require_repair_l448_448493


namespace sum_of_products_non_empty_subsets_l448_448124

variable {M : Set ℤ}
def M : Set ℤ := {-1, 0, 1, -2, 2, 10, 20, -30, 99, -100}
def non_empty_subsets (s : Set α) : Set (Set α) := {t | t ⊆ s ∧ t ≠ ∅}
def product_of_elements (s : Set ℤ) : ℤ := ∏ i in s, i
def sum_of_products_of_non_empty_subsets (s : Set ℤ) : ℤ :=
  ∑ t in non_empty_subsets s, product_of_elements t

theorem sum_of_products_non_empty_subsets :
  sum_of_products_of_non_empty_subsets M = -1 := 
by
  sorry

end sum_of_products_non_empty_subsets_l448_448124


namespace monotonic_increasing_l448_448232

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π / 6)

theorem monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 2) → f x1 < f x2 := sorry

end monotonic_increasing_l448_448232


namespace giants_need_to_win_more_games_l448_448841

/-- The Giants baseball team is trying to make their league playoff.
They have played 20 games and won 12 of them. To make the playoffs, they need to win 2/3 of 
their games over the season. If there are 10 games left, how many do they have to win to
make the playoffs? 
-/
theorem giants_need_to_win_more_games (played won needed_won total remaining required_wins additional_wins : ℕ)
    (h1 : played = 20)
    (h2 : won = 12)
    (h3 : remaining = 10)
    (h4 : total = played + remaining)
    (h5 : total = 30)
    (h6 : required_wins = 2 * total / 3)
    (h7 : additional_wins = required_wins - won) :
    additional_wins = 8 := 
    by
      -- sorry should be used if the proof steps were required.
sorry

end giants_need_to_win_more_games_l448_448841


namespace triangle_circle_condition_l448_448759

theorem triangle_circle_condition (A B C D E : Point) (θ : ℝ) 
  (h1 : 0 < θ) (h2 : θ < 60)
  (h_angleC : ∠ C = θ) (h_angleB : ∠ B = 2 * θ)
  (h_circle : Circle A (distance A B))
  (h_intersect_AC_D : intersects_at h_circle (line_through A C) D)
  (h_intersect_BC_E : intersects_at h_circle (line_through B C) E)
  : distance E C = distance A D :=
sorry

end triangle_circle_condition_l448_448759


namespace opposite_number_on_circle_l448_448027

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l448_448027


namespace train_crossing_time_l448_448591

/-- Given the speed of the train in km/hr and the length of the train in meters,
    prove that the time it takes to cross a pole is approximately 12 seconds. -/
theorem train_crossing_time (v : ℝ) (L : ℝ) (h1 : v = 60) (h2 : L = 200) :
  L / ((v * 1000) / 3600) ≈ 12 :=
by
  sorry

end train_crossing_time_l448_448591


namespace possible_gcd_values_l448_448256

theorem possible_gcd_values (a b : ℕ) (h1 : a * b = 360) (h2 : Int.gcd a b * Int.lcm a b = a * b) : 
  ∃ S, S = {d : ℕ | ∃ (p q r s t u : ℕ), 
               a = 2^p * 3^q * 5^r ∧ 
               b = 2^s * 3^t * 5^u ∧ 
               p + s = 3 ∧ 
               q + t = 2 ∧ 
               r + u = 1 ∧ 
               d = 2^(min p s) * 3^(min q t) * 5^(min r u)} ∧
        S.card = 8 :=
by 
  sorry

end possible_gcd_values_l448_448256


namespace new_person_weight_l448_448845

theorem new_person_weight:
  ∃ W : ℕ, ( ∀ n : ℕ, (n = 6) → (avg_weight_increase : ℝ, (avg_weight_increase = 3.5)) → 
  (person_weight : ℕ, (person_weight = 47)) → 
  (new_weight : ℕ, (new_weight = person_weight + n * avg_weight_increase)) → 
  (W = new_weight) ):
sorry

end new_person_weight_l448_448845


namespace number_of_divisors_l448_448775

/--
Let  \( S = \{1, 2, 3, 4, 5, 6\} \) and \( \mathcal{P} \) be the set of all nonempty subsets of \( S \).
Let \( N \) equal the number of functions \( f: \mathcal{P} \to S \) such that if \( A, B \in \mathcal{P} \)
are disjoint, then \( f(A) \neq f(B) \).
Prove that the number of positive integer divisors of \( N \) is \( 13872 \).
-/
theorem number_of_divisors (S : Set ℕ) (N : ℕ) 
    (hS : S = {1, 2, 3, 4, 5, 6})
    (hP : ∀ A B ∈ S.powerset \ {∅}, A ∩ B = ∅ → A ≠ B → ∃! f : S.powerset → S, f A ≠ f B):
    nat.divisor_count N = 13872 :=
sorry

end number_of_divisors_l448_448775


namespace distinct_real_roots_P_of_P_l448_448484

-- Define the polynomial P(x) of degree 2017
variables {R : Type*} [comm_ring R] [is_domain R]

noncomputable def P : polynomial R := sorry  -- A polynomial of degree 2017

-- Define the condition that P has k distinct real roots
variables {k : ℕ} {r : fin k → R}

-- Assume that P(r_i) = 0 for i = 1, 2, ..., k
axiom h_root : ∀ i : fin k, polynomial.eval (r i) P = 0

-- Prove that the number of distinct real roots of P(P(x)) = 0 is at least k
theorem distinct_real_roots_P_of_P (ht : degree P = 2017) : 
  ∃ m : ℕ, m ≥ k ∧ ∀ s : fin m => s ≠ t → eval (s i) (polynomial.comp P P) = 0 :=
sorry

end distinct_real_roots_P_of_P_l448_448484


namespace camping_trip_percentage_l448_448923

theorem camping_trip_percentage (t : ℕ) (h1 : 22 / 100 * t > 0) (h2 : 75 / 100 * (22 / 100 * t) ≤ t) :
  (88 / 100 * t) = t :=
by
  sorry

end camping_trip_percentage_l448_448923


namespace odd_function_at_zero_l448_448314

theorem odd_function_at_zero
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  f 0 = 0 :=
by
  sorry

end odd_function_at_zero_l448_448314


namespace opposite_113_eq_114_l448_448061

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l448_448061


namespace optimal_partition_leq_cbrt_6n_l448_448932

theorem optimal_partition_leq_cbrt_6n 
  (n : ℕ) (A : set ℕ) (hA : ∀ x, x ∈ A → x ≤ n → x > 0) :
  ∀ (k : ℕ), (∃ (a : ℕ → ℕ), 
    (∀ i, i < k → a i ∈ A) ∧
    (finset.sum finset.univ (λ i, a (i : ℕ)) = n) ∧
    (∀ (b : ℕ → ℕ), (∀ j, j < k → b j ∈ A) → 
     (finset.sum finset.univ (λ j, b (j : ℕ)) = n) → 
     k ≤ finset.card (finset.univ.image b)
    )
  ) → 
  k ≤ nat.cbrt (6 * n)
:= 
sorry

end optimal_partition_leq_cbrt_6n_l448_448932


namespace determine_amount_of_substance_l448_448990

noncomputable def amount_of_substance 
  (A : ℝ) (R : ℝ) (delta_T : ℝ) : ℝ :=
  (2 * A) / (R * delta_T)

theorem determine_amount_of_substance 
  (A : ℝ := 831) 
  (R : ℝ := 8.31) 
  (delta_T : ℝ := 100) 
  (nu : ℝ := amount_of_substance A R delta_T) :
  nu = 2 := by
  -- Conditions rewritten as definitions
  -- Definition: A = 831 J
  -- Definition: R = 8.31 J/(mol * K)
  -- Definition: delta_T = 100 K
  -- The correct answer to be proven: nu = 2 mol
  sorry

end determine_amount_of_substance_l448_448990


namespace arc_lengths_l448_448752

theorem arc_lengths (O T I R S P : Point) (r : ℝ)
  (h_angle_TIP : angle T I P = 60) 
  (h_OT : dist O T = 15) 
  (h_collinear : collinear O T I)
  (h_circle : ∀ p, dist O p = r): 
  (arc_length R P = 10 * π ∧ arc_length R S = 10 * π) :=
by
  sorry

end arc_lengths_l448_448752


namespace first_machine_defect_probability_l448_448391

/-- Probability that a randomly selected defective item was made by the first machine is 0.5 
given certain conditions. -/
theorem first_machine_defect_probability :
  let PFirstMachine := 0.4
  let PSecondMachine := 0.6
  let DefectRateFirstMachine := 0.03
  let DefectRateSecondMachine := 0.02
  let TotalDefectProbability := PFirstMachine * DefectRateFirstMachine + PSecondMachine * DefectRateSecondMachine
  let PDefectGivenFirstMachine := PFirstMachine * DefectRateFirstMachine / TotalDefectProbability
  PDefectGivenFirstMachine = 0.5 :=
by
  sorry

end first_machine_defect_probability_l448_448391


namespace value_of_expression_l448_448531

theorem value_of_expression :
  (∏ i in Finset.range 10, i + 1) =
  10!
→ ((∑ i in Finset.range 10, i + 1) *
   (∏ i in Finset.range 3, i + 1)) =
   (55 * 6)
→ (3628800 / 330) = 11000 :=
sorry

end value_of_expression_l448_448531


namespace closest_multiple_3050_l448_448159

-- Define a predicate that checks if a number is a multiple of 18
def multiple_of_18 (n : ℕ) : Prop :=
  n % 18 = 0

-- Define the function to find the closest multiple of 18 to a given number
def closest_multiple_of_18 (n : ℕ) : ℕ :=
  if multiple_of_18 n then n
  else if multiple_of_18 (n - 1) then (n - 1)
  else if multiple_of_18 (n + 1) then (n + 1)
  else if multiple_of_18 (n - 2) then (n - 2)
  else if multiple_of_18 (n + 2) then (n + 2)
  else if multiple_of_18 (n - 3) then (n - 3)
  else if multiple_of_18 (n + 3) then (n + 3)
  else if multiple_of_18 (n - 4) then (n - 4)
  else (n + 4) -- This should cover all nearby cases; extendable if necessary

-- Assert the proof problem as a Lean theorem statement
theorem closest_multiple_3050 : closest_multiple_of_18 3050 = 3042 :=
by
  sorry

end closest_multiple_3050_l448_448159


namespace card_prob_queen_diamond_l448_448890

theorem card_prob_queen_diamond : 
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob in
  total_prob = 18 / 221 :=
by
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob
  sorry

end card_prob_queen_diamond_l448_448890


namespace ram_interest_first_year_l448_448141

theorem ram_interest_first_year :
  ∀ (initial_deposit first_year_balance : ℕ),
  (initial_deposit = 500) →
  (first_year_balance = 600) →
  (first_year_balance - initial_deposit = 100) :=
by
  intros initial_deposit first_year_balance h1 h2
  rw [h1, h2]
  norm_num

end ram_interest_first_year_l448_448141


namespace digit_245_of_128_div_999_is_2_l448_448526

def repeating_decimal := "128"

def nth_repeated_digit (n : ℕ) (rep : String) : Char :=
  rep.get! ((n % rep.length) - 1)

theorem digit_245_of_128_div_999_is_2 :
  nth_repeated_digit 245 repeating_decimal = '2' :=
by
  sorry

end digit_245_of_128_div_999_is_2_l448_448526


namespace symmetry_center_of_tangent_l448_448077

noncomputable def tangentFunction (x : ℝ) : ℝ := Real.tan (2 * x - (Real.pi / 3))

theorem symmetry_center_of_tangent :
  (∃ k : ℤ, (Real.pi / 6) + (k * Real.pi / 4) = 5 * Real.pi / 12 ∧ tangentFunction ((5 * Real.pi) / 12) = 0 ) :=
sorry

end symmetry_center_of_tangent_l448_448077


namespace symmetry_axis_max_min_value_l448_448350

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos x - sin (π / 4 - x) ^ 2

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∀ x, f x = f (x + (π / 2)) →
  x = π / 4 + k * (π / 2) := sorry

theorem max_min_value (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :
  let f_shifted := λ x, f (x - π / 8)
  have (min_val : f_shifted 0 = - (√2 + 1) / 2) ∧ 
       (max_val : f_shifted (3 * π / 8) = 1 / 2) := sorry

end symmetry_axis_max_min_value_l448_448350


namespace card_prob_queen_diamond_l448_448891

theorem card_prob_queen_diamond : 
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob in
  total_prob = 18 / 221 :=
by
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob
  sorry

end card_prob_queen_diamond_l448_448891


namespace cube_sum_mod_6_l448_448294

theorem cube_sum_mod_6 : (∑ i in Finset.range 101, i^3) % 6 = 4 :=
by
  sorry

end cube_sum_mod_6_l448_448294


namespace total_tickets_sold_l448_448202

-- Definitions and conditions
def orchestra_ticket_price : ℕ := 12
def balcony_ticket_price : ℕ := 8
def total_revenue : ℕ := 3320
def ticket_difference : ℕ := 190

-- Variables
variables (x y : ℕ) -- x is the number of orchestra tickets, y is the number of balcony tickets

-- Statements of conditions
def revenue_eq : Prop := orchestra_ticket_price * x + balcony_ticket_price * y = total_revenue
def tickets_relation : Prop := y = x + ticket_difference

-- The proof problem statement
theorem total_tickets_sold (h1 : revenue_eq x y) (h2 : tickets_relation x y) : x + y = 370 :=
by
  sorry

end total_tickets_sold_l448_448202


namespace oil_bill_increase_l448_448511

theorem oil_bill_increase :
  ∀ (F x : ℝ), 
    (F / 120 = 5 / 4) → 
    ((F + x) / 120 = 3 / 2) → 
    x = 30 :=
by
  intros F x h1 h2
  -- proof
  sorry

end oil_bill_increase_l448_448511


namespace initial_number_of_men_l448_448134

-- Define initial variables and conditions
variable {M : ℕ}  -- Initial number of men
variable (Food : ℕ → ℕ)  -- Food function (amount of food for given days and men)
variable (E2 : Food 22 M)  -- Food is available for 22 days for initial men M
variable (A2 : Food 20 M)  -- After 2 days, food is still available for 20 more days for initial men M
variable (additional : ℕ := 760)  -- Additional number of men who joined
variable (E3 : Food 10 (M + additional))  -- After two days and with additional men, food lasts 10 more days

-- Proof statement
theorem initial_number_of_men : M = 760 :=
by
  sorry

end initial_number_of_men_l448_448134


namespace zach_needs_more_money_l448_448540

theorem zach_needs_more_money
  (bike_cost : ℕ) (allowance : ℕ) (mowing_payment : ℕ) (babysitting_rate : ℕ) 
  (current_savings : ℕ) (babysitting_hours : ℕ) :
  bike_cost = 100 →
  allowance = 5 →
  mowing_payment = 10 →
  babysitting_rate = 7 →
  current_savings = 65 →
  babysitting_hours = 2 →
  (bike_cost - (current_savings + (allowance + mowing_payment + babysitting_hours * babysitting_rate))) = 6 :=
by
  sorry

end zach_needs_more_money_l448_448540


namespace trigonometric_operation_l448_448707

theorem trigonometric_operation :
  let m := Real.cos (Real.pi / 6)
  let n := Real.sin (Real.pi / 6)
  let op (m n : ℝ) := m^2 - m * n - n^2
  op m n = (1 / 2 : ℝ) - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_operation_l448_448707


namespace opposite_number_113_is_114_l448_448057

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l448_448057


namespace ratio_R_to_PQ_l448_448071

noncomputable def total_amount : ℕ := 8000
noncomputable def amount_R : ℕ := 3200
noncomputable def amount_PQ : ℕ := total_amount - amount_R

theorem ratio_R_to_PQ : amount_R / amount_PQ = 2 / 3 :=
by
  have h₁ : amount_PQ = total_amount - amount_R := rfl
  have h₂ : total_amount = 8000 := rfl
  have h₃ : amount_R = 3200 := rfl
  have h₄ : amount_PQ = 8000 - 3200 := calc
    amount_PQ = total_amount - amount_R : by rw [h₁]
    ...       = 8000 - 3200           : by rw [h₂, h₃]
  have h₅ : amount_PQ = 4800 := by linarith [h₄]
  have ratio_simplified : amount_R / amount_PQ = 3200 / 4800 := calc
    amount_R / amount_PQ = 3200 / 4800 : by rw [h₃, h₅]
  have ratio_final : 3200 / 4800 = 2 / 3 := calc
    3200 / 4800 = (3200 / 1600) / (4800 / 1600) : by norm_num
    ...         = 2 / 3                         : by norm_num
  sorry

end ratio_R_to_PQ_l448_448071


namespace cos_transform_shift_l448_448515

theorem cos_transform_shift :
  ∀ (x : ℝ), (3 * cos (2 * x - π / 3)) = 3 * cos (2 * (x - π / 6)) :=
by
  intro x
  -- Note: The proof is intentionally omitted with sorry.
  sorry

end cos_transform_shift_l448_448515


namespace number_of_correct_propositions_is_one_l448_448403

theorem number_of_correct_propositions_is_one :
  let p1 := ∀ (Q : Type) [quadrilateral Q], (∀ A B C D : Q, (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) → -- A quadrilateral with two pairs of equal opposite sides.
  (∃ P1 P2 P3 P4 : Q, P1 ≠ P2 ∧ P1 ≠ P3 ∧ P1 ≠ P4 ∧ P2 ≠ P3 ∧ P2 ≠ P4 ∧ P3 ≠ P4 ∧ (dist P1 P2 = dist P3 P4) ∧ (dist P2 P3 = dist P4 P1)) =
  -- statement: is a parallelogram.
  (∃ P1' P2' P3' P4' : Q, P1' ≠ P2' ∧ P1' ≠ P3' ∧ P1' ≠ P4' ∧ P2' ≠ P3' ∧ P2' ≠ P4' ∧ P3' ≠ P4' ∧ parallel P1' P2' P3' P4' ∧ parallel P2' P3' P4' P1') ∧
  let p2 := ∀ (Q : Type) [quadrilateral Q], (∀ A B C D : Q, (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) → -- A quadrilateral with all four sides equal.
  (∃ P1 P2 P3 P4 : Q, P1 ≠ P2 ∧ P1 ≠ P3 ∧ P1 ≠ P4 ∧ P2 ≠ P3 ∧ P2 ≠ P4 ∧ P3 ≠ P4 ∧ (dist P1 P2 = dist P2 P3) ∧ (dist P2 P3 = dist P3 P4) ∧ (dist P3 P4 = dist P4 P1) ∧ (dist P4 P1 = dist P1 P2)) =
  -- statement: is a rhombus.
  (∃ P1' P2' P3' P4' : Q, P1' ≠ P2' ∧ P1' ≠ P3' ∧ P1' ≠ P4' ∧ P2' ≠ P3' ∧ P2' ≠ P4' ∧ P3' ≠ P4' ∧ rhombus P1' P2' P3' P4') ∧
  let p3 := ∀ (Q : Type) [quadrilateral Q], (∀ A B C D : Q, (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) → -- A quadrilateral with two pairs of opposite sides parallel.
  (∃ P1 P2 P3 P4 : Q, P1 ≠ P2 ∧ P1 ≠ P3 ∧ P1 ≠ P4 ∧ P2 ≠ P3 ∧ P2 ≠ P4 ∧ P3 ≠ P4 ∧ parallel P1 P2 P3 P4 ∧ parallel P2 P3 P4 P1)) =
  -- statement: is a parallelogram.
  (∃ P1' P2' P3' P4' : Q, P1' ≠ P2' ∧ P1' ≠ P3' ∧ P1' ≠ P4' ∧ P2' ≠ P3' ∧ P2' ≠ P4' ∧ P3' ≠ P4' ∧ parallelogram P1' P2' P3' P4') ∧
  let p4 := ∀ (Q : Type) [quadrilateral Q], (∀ A B C D : Q, (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) → -- The quadrilateral formed by connecting the midpoints of the sides of a space quadrilateral.
  (∃ P1 P2 P3 P4 : Q, P1 ≠ P2 ∧ P1 ≠ P3 ∧ P1 ≠ P4 ∧ P2 ≠ P3 ∧ P2 ≠ P4 ∧ P3 ≠ P4 ∧ (midpoint P1 P2 = P3) ∧ (midpoint P2 P3 = P4) ∧ (midpoint P3 P4 = P1) ∧ (midpoint P4 P1 = P2)) =
  -- statement: is a trapezoid.
  (∃ P1' P2' P3' P4' : Q, P1' ≠ P2' ∧ P1' ≠ P3' ∧ P1' ≠ P4' ∧ P2' ≠ P3' ∧ P2' ≠ P4' ∧ P3' ≠ P4' ∧ trapezoid P1' P2' P3' P4') in
  (∃ (Q : Type) [quadrilateral Q], p1 Q ∨ p2 Q ∨ p3 Q ∨ p4 Q = 1) :=
  sorry

end number_of_correct_propositions_is_one_l448_448403


namespace solve_eq_roots_l448_448468

noncomputable def solve_equation (x : ℝ) : Prop :=
  (7 * x + 2) / (3 * x^2 + 7 * x - 6) = (3 * x) / (3 * x - 2)

theorem solve_eq_roots (x : ℝ) (h₁ : x ≠ 2 / 3) :
  solve_equation x ↔ (x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3) :=
by
  sorry

end solve_eq_roots_l448_448468


namespace triangle_sides_inequality_triangle_sides_equality_condition_l448_448772

theorem triangle_sides_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem triangle_sides_equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end triangle_sides_inequality_triangle_sides_equality_condition_l448_448772


namespace g_500_eq_504_l448_448108

-- Define the function g with the given properties
axiom g : ℕ → ℕ
axiom g_g_n : ∀ n : ℕ, 0 < n → g(g(n)) = 3 * n
axiom g_5n_plus_1 : ∀ n : ℕ, 0 < n → g(5 * n + 1) = 5 * n + 3

-- Prove g(500) = 504 based on the given conditions
theorem g_500_eq_504 : g(500) = 504 :=
by
  sorry

end g_500_eq_504_l448_448108


namespace projection_length_l448_448456

-- Definitions of points, line segments, and geometric entities
variable (A B C O P: Point)
variable (AB AC BC : LineSegment)
variable (angle α β γ: Angle)
variable (AO : Line)
variable [Circumscribed O (triangle A B C)]
variable [AcuteAngled (triangle A B C)]
variable [IntersectsAt (line AO) (side BC) P]

-- Definitions of points E and F on sides AB and AC respectively.
variable (E F: Point)
variable [OnSide (point E) (side AB)]
variable [OnSide (point F) (side AC)]

-- Additional constraints
variable [Circumscribed (quadrilateral E A P F)]

-- Main statement to prove the projection independence.
theorem projection_length (segment EF: LineSegment) : 
  (∀ E F, OnSide E AB ∧ OnSide F AC ∧ Circumscribed (quadrilateral E A P F))
  → ProjectionLength EF BC = ProjectionLength EF' BC :=
sorry

end projection_length_l448_448456


namespace distinct_products_count_is_26_l448_448365

open Finset

def set_numbers : Finset ℕ := {2, 3, 5, 7, 11}

def distinct_products_count (s : Finset ℕ) : ℕ :=
  let pairs := s.powerset.filter (λ t => 2 ≤ t.card)
  pairs.card

theorem distinct_products_count_is_26 : distinct_products_count set_numbers = 26 := by
  sorry

end distinct_products_count_is_26_l448_448365


namespace distance_eq_5sqrt3_l448_448282

def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2 + (b.3 - a.3)^2)

def pointA : ℝ × ℝ × ℝ := (3, 3, 3)
def pointB : ℝ × ℝ × ℝ := (-2, -2, -2)

theorem distance_eq_5sqrt3 : distance pointA pointB = 5 * real.sqrt 3 :=
by
  sorry

end distance_eq_5sqrt3_l448_448282


namespace total_loan_amount_l448_448091

theorem total_loan_amount (years : ℕ) (months_per_year: ℕ) (monthly_payment : ℕ) (down_payment : ℕ)
  (h1 : years = 5)
  (h2 : months_per_year = 12)
  (h3 : monthly_payment = 600)
  (h4 : down_payment = 10000) :
  let total_amount_paid := monthly_payment * (years * months_per_year) + down_payment
  in total_amount_paid = 46000 :=
by
  simp [h1, h2, h3, h4]
  sorry

end total_loan_amount_l448_448091


namespace sum_integers_neg_20_to_10_l448_448529

theorem sum_integers_neg_20_to_10 :
  (Finset.sum (Finset.range (10 - (-20) + 1)) (λ k, k - 20)) = -155 :=
by
  -- skipping the proof for now
  sorry

end sum_integers_neg_20_to_10_l448_448529


namespace log_sum_eq_two_l448_448261

theorem log_sum_eq_two : log 10 4 + log 10 25 = 2 :=
by
  sorry

end log_sum_eq_two_l448_448261


namespace square_side_length_l448_448624

theorem square_side_length (r : ℝ) (side : ℝ) 
  (h1 : r = 1) -- radius of circle
  (h2 : ∃ (A B C D : ℝ×ℝ), 
    dist A (0, 0) = r ∧ -- A is on the circle
    dist B (0, 0) = r ∧ -- B is on the circle
    dist A B = side ∧ -- A and B are adjacent vertices
    dist C D = side ∧ -- C and D are the other side
    (∀ P : ℝ×ℝ, dist P (0,0) = r → dist P (midpoint A B) > dist P (C, D))) -- BC is tangent
    :
  side = 8 / 5 :=
by
  sorry

end square_side_length_l448_448624


namespace probability_six_distinct_one_repeat_l448_448527

def num_dice : ℕ := 7
def num_faces : ℕ := 6

noncomputable def total_outcomes : ℕ := num_faces ^ num_dice

noncomputable def repeated_num_choices : ℕ := num_faces
noncomputable def distinct_num_ways : ℕ := Nat.factorial num_faces.pred
noncomputable def dice_combinations : ℕ := Nat.choose num_dice 2

noncomputable def favorable_outcomes : ℕ := repeated_num_choices * distinct_num_ways * dice_combinations

noncomputable def probability : ℚ := favorable_outcomes / total_outcomes

theorem probability_six_distinct_one_repeat :
  probability = 5 / 186 := 
by
  sorry

end probability_six_distinct_one_repeat_l448_448527


namespace paratrooper_exit_distance_l448_448951

theorem paratrooper_exit_distance (S : ℝ) (hS : 0 < S) :
  ∃ d ≤ 2 * Real.sqrt (Real.pi * S), ∀ p ∈ forest S, exits_forest p d :=
begin
  sorry -- proof to be filled in
end

end paratrooper_exit_distance_l448_448951


namespace iron_balls_molded_l448_448994

-- Define the dimensions of the iron bar
def length_bar : ℝ := 12
def width_bar : ℝ := 8
def height_bar : ℝ := 6

-- Define the volume calculations
def volume_iron_bar : ℝ := length_bar * width_bar * height_bar
def number_of_bars : ℝ := 10
def total_volume_bars : ℝ := volume_iron_bar * number_of_bars
def volume_iron_ball : ℝ := 8

-- Define the goal statement
theorem iron_balls_molded : total_volume_bars / volume_iron_ball = 720 :=
by
  -- Proof is to be filled in here
  sorry

end iron_balls_molded_l448_448994


namespace ascending_order_l448_448609

theorem ascending_order (a b c d : ℝ) (h1 : a = -6) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = Real.pi) :
  a < b ∧ b < c ∧ c < d :=
by
  sorry

end ascending_order_l448_448609


namespace no_positive_integral_solutions_l448_448508

theorem no_positive_integral_solutions (x y : ℕ) (h : x > 0) (k : y > 0) :
  x^4 * y^4 - 8 * x^2 * y^2 + 12 ≠ 0 :=
by
  sorry

end no_positive_integral_solutions_l448_448508


namespace find_angle_θ_find_norm_a_plus_b_proj_a_on_a_plus_b_l448_448688

variables (a b : ℝ^3) -- Representing 3D vectors
variables (θ : ℝ)

-- Conditions
axiom norm_a : ‖a‖ = 2
axiom norm_b : ‖b‖ = 1
axiom dot_condition : (2 • a - 3 • b) ⬝ (2 • a + b) = 9

-- Prove the angle θ between a and b:
theorem find_angle_θ : (a ⬝ b) = (‖a‖ * ‖b‖ * (Real.cos θ)) ∧ θ = Real.arccos (1 / 2) := 
by 
  sorry

-- Prove norm of (a + b):
theorem find_norm_a_plus_b : ‖a + b‖ = Real.sqrt(7) :=
by 
  sorry

-- Projection of vector a in the direction of (a + b)
theorem proj_a_on_a_plus_b : 
  ((a ⬝ (a + b)) / ‖a + b‖ = 5 / Real.sqrt 7 ∧ ((a ⬝ (a + b)) / ‖a + b‖ ) = (5 * Real.sqrt 7) / 7) :=
by 
  sorry

end find_angle_θ_find_norm_a_plus_b_proj_a_on_a_plus_b_l448_448688


namespace polar_line_eq_l448_448862

theorem polar_line_eq (A P : ℝ × ℝ) (hA : A = (2, 0)) (L : ℝ × ℝ → Prop)
  (hL1 : ∀ (P : ℝ × ℝ), L P ↔ ∃ (θ : ℝ), P = (2 / cos θ, θ))
  (hL2 : ∀ (θ : ℝ), L (2 / cos θ, θ)) :
  ∀ (P : ℝ × ℝ) (ρ θ : ℝ), P = (ρ, θ) → (L P → ρ * cos θ = 2) :=
by 
  intros P ρ θ hP hLP,
  rw [hP] at hLP,
  rcases hL1 P with ⟨θ', hP'⟩,
  have hθ' : θ' = θ, from sorry,
  rw [hθ', hP'],
  exact sorry

end polar_line_eq_l448_448862


namespace f_odd_range_f_pos_l448_448438

variable (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := if a > 0 ∧ a ≠ 1 then log a (1 + 2 * x) - log a (1 - 2 * x) else 0

theorem f_odd (h1 : a > 0) (h2 : a ≠ 1) : f a x = -f a (-x) :=
by sorry

theorem range_f_pos (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → 0 < x ∧ x < 1/2) ∧ (0 < a ∧ a < 1 → -1/2 < x ∧ x < 0) :=
by sorry

end f_odd_range_f_pos_l448_448438


namespace Irene_age_is_46_l448_448273

-- Definitions based on the given conditions
def Eddie_age : ℕ := 92
def Becky_age : ℕ := Eddie_age / 4
def Irene_age : ℕ := 2 * Becky_age

-- Theorem we aim to prove that Irene's age is 46
theorem Irene_age_is_46 : Irene_age = 46 := by
  sorry

end Irene_age_is_46_l448_448273


namespace marie_packs_construction_paper_l448_448602

theorem marie_packs_construction_paper (marie_glue_sticks : ℕ) (allison_glue_sticks : ℕ) (total_allison_items : ℕ)
    (glue_sticks_difference : allison_glue_sticks = marie_glue_sticks + 8)
    (marie_glue_sticks_count : marie_glue_sticks = 15)
    (total_items_allison : total_allison_items = 28)
    (marie_construction_paper_multiplier : ℕ)
    (construction_paper_ratio : marie_construction_paper_multiplier = 6) : 
    ∃ (marie_construction_paper_packs : ℕ), marie_construction_paper_packs = 30 := 
by
  sorry

end marie_packs_construction_paper_l448_448602


namespace slices_per_large_pizza_l448_448735

theorem slices_per_large_pizza (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ) 
  (H1 : total_pizzas = 2) (H2 : slices_eaten = 7) (H3 : slices_remaining = 9) : 
  (slices_remaining + slices_eaten) / total_pizzas = 8 := 
by
  sorry

end slices_per_large_pizza_l448_448735


namespace retail_price_before_discount_l448_448575

-- Definitions of constants
def n := 15
def p_wholesale := 126
def t_bulk_discount := 0.06
def t_sales_tax := 0.08
def t_profit := 0.22
def t_customer_discount := 0.12

-- Calculation helpers
def total_wholesale_price (n : ℕ) (p : ℝ) : ℝ := n * p
def bulk_discount (total : ℝ) (discount_rate : ℝ) : ℝ := total * discount_rate
def total_cost_after_discount (total : ℝ) (discount : ℝ) : ℝ := total - discount
def profit_per_machine (p : ℝ) (profit_rate : ℝ) : ℝ := p * profit_rate
def total_profit (profit_per_machine : ℝ) (n : ℕ) : ℝ := profit_per_machine * n
def sales_tax (profit : ℝ) (tax_rate : ℝ) : ℝ := profit * tax_rate
def total_after_tax (cost : ℝ) (profit : ℝ) (tax : ℝ) : ℝ := cost + profit - tax
def price_per_machine_before_discount (total_after_tax : ℝ) (customer_discount_rate : ℝ) (n : ℕ) : ℝ :=
  total_after_tax / ((1 - customer_discount_rate) * n)

-- The theorem to prove the retail price before discount
theorem retail_price_before_discount :
  price_per_machine_before_discount 
    (total_after_tax
      (total_cost_after_discount
        (total_wholesale_price n p_wholesale)
        (bulk_discount (total_wholesale_price n p_wholesale) t_bulk_discount))
      (total_profit 
        (profit_per_machine p_wholesale t_profit) 
        n)
      (sales_tax 
        (total_profit 
          (profit_per_machine p_wholesale t_profit) 
          n)
        t_sales_tax))
    t_customer_discount
    n
  ≈ 163.59 := sorry

end retail_price_before_discount_l448_448575


namespace sarah_toads_count_l448_448135

theorem sarah_toads_count:
  let tim_toads := 30 in
  let jim_toads := tim_toads + 20 in
  let sarah_initial_toads := 2 * jim_toads in
  let sarah_toads_after_giveaway := sarah_initial_toads - (sarah_initial_toads / 4) in
  let sarah_final_toads := sarah_toads_after_giveaway + 15 in
  sarah_final_toads = 90 :=
by
  sorry

end sarah_toads_count_l448_448135


namespace correct_option_d_l448_448375

theorem correct_option_d (a b c : ℝ) (h: a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end correct_option_d_l448_448375


namespace opposite_number_on_circle_l448_448029

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l448_448029


namespace problem_1_problem_2_problem_3_l448_448699

noncomputable def f (x : ℝ) : ℝ := √3 * Real.cos x ^ 2 + Real.sin x * Real.cos x

theorem problem_1 : f (π / 6) = √3 := 
by sorry

theorem problem_2 : 
  ∀ k : ℤ, (k * π - 5 * π / 12 ≤ x) →
           (x ≤ k * π + π / 12) →
           Function.IncreasingOn f (Set.Icc (k * π - 5 * π / 12) (k * π + π / 12)) := 
by sorry

theorem problem_3 (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : f (α / 2) = 1 / 4 + √3 / 2) : 
  Real.sin (α + 7 * π / 12) = (√2 - √30) / 8 := 
by sorry

end problem_1_problem_2_problem_3_l448_448699


namespace find_min_n_l448_448006

def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 98 }

def isCoprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def hasSpecialGroups (T : Finset ℕ) : Prop :=
  ∃ A B : Finset ℕ, 
  A ∪ B = T ∧ 
  A ∩ B = ∅ ∧ 
  ∃ x ∈ A, 4 ≤ (A.filter (λ y, y ≠ x ∧ isCoprime x y)).card ∧ 
  ∃ x ∈ B, 4 ≤ (B.filter (λ y, y ≠ x ∧ ¬ isCoprime x y)).card

theorem find_min_n : ∃ n : ℕ, ∀ T : Finset ℕ, T ⊆ S → T.card = n → hasSpecialGroups T := 
begin
  use 50,
  intro T,
  intro hTS,
  intro hTcard,
  sorry
end

end find_min_n_l448_448006


namespace number_of_good_line_pairs_l448_448263

theorem number_of_good_line_pairs :
  let line1 := λ x : ℝ, 4 * x + 7
  let line2 := λ x : ℝ, 2 * x + 3
  let line3 := λ x : ℝ, -1 / 3 * x + 2
  let line4 := λ x : ℝ, 2 * x - 3
  let line5 := λ x : ℝ, -1 / 3 * x + 5 / 2
  let slope1 := 4
  let slope2 := 2
  let slope3 := -1 / 3
  let slope4 := 2
  let slope5 := -1 / 3
  in
  (slope1 = slope2 ∨ slope1 = slope3 ∨ slope1 = slope4 ∨ slope1 = slope5 
   ∨ slope2 = slope3 ∨ slope2 = slope4 ∨ slope2 = slope5 
   ∨ slope3 = slope4 ∨ slope3 = slope5 ∨ slope4 = slope5)
  ∨ (slope1 * slope2 = -1 ∨ slope1 * slope3 = -1 ∨ slope1 * slope4 = -1 ∨ slope1 * slope5 = -1 
      ∨ slope2 * slope3 = -1 ∨ slope2 * slope4 = -1 ∨ slope2 * slope5 = -1
      ∨ slope3 * slope4 = -1 ∨ slope3 * slope5 = -1 ∨ slope4 * slope5 = -1)
  = 3 :=
sorry

end number_of_good_line_pairs_l448_448263


namespace min_hypotenuse_of_right_triangle_l448_448478

theorem min_hypotenuse_of_right_triangle (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a + b + c = 6) : 
  c = 6 * (Real.sqrt 2 - 1) :=
sorry

end min_hypotenuse_of_right_triangle_l448_448478


namespace opposite_number_113_is_114_l448_448055

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l448_448055


namespace symmetric_points_y_axis_l448_448686

theorem symmetric_points_y_axis (a b : ℝ) (h1 : M a 3) (h2 : N 4 b) (h_symm : ∀ x₁ y₁ x₂ y₂, M x₁ y₁ → N x₂ y₂ → x₁ = -x₂ ∧ y₁ = y₂) : a + b = -1 :=
by {
  have h := h_symm a 3 4 b h1 h2,
  cases h with h_x h_y,
  rw h_x,
  rw h_y,
  simp,
  sorry
}

end symmetric_points_y_axis_l448_448686


namespace distance_between_parallel_lines_l448_448385

-- Define the lines and the condition of parallelism
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (1, a, 6)
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (a - 2, 3, 2 * a)

-- Prove the distance between the lines when they are parallel
theorem distance_between_parallel_lines (a : ℝ) (ha : 3 = a * (a - 2)) : 
  a = -1 → let d := |2 * a - 6| / sqrt ((a - 2)^2 + 3^2) in d = (8 * sqrt 2) / 3 :=
by
  sorry

end distance_between_parallel_lines_l448_448385


namespace trigonometric_inequality_l448_448463

theorem trigonometric_inequality (a b x : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ( (a + b) / 2 )^2 :=
by
  sorry

end trigonometric_inequality_l448_448463


namespace coefficient_x3_of_expr_l448_448642

def expr := 3 * (x - 2 * x^3) - 4 * (x^2 - 3 * x^3 + x^6) + 5 * (3 * x^2 - 2 * x^9)

theorem coefficient_x3_of_expr : 
  coefficient x^3 expr = 6 := 
sorry

end coefficient_x3_of_expr_l448_448642


namespace sum_first_four_terms_geo_seq_l448_448680

theorem sum_first_four_terms_geo_seq (q : ℝ) (a_1 : ℝ)
  (h1 : q ≠ 1) 
  (h2 : a_1 * (a_1 * q) * (a_1 * q^2) = -1/8)
  (h3 : 2 * (a_1 * q^3) = (a_1 * q) + (a_1 * q^2)) :
  (a_1 + (a_1 * q) + (a_1 * q^2) + (a_1 * q^3)) = 5 / 8 :=
  sorry

end sum_first_four_terms_geo_seq_l448_448680


namespace find_age_l448_448168

theorem find_age (x : ℕ) (h : 5 * (x + 5) - 5 * (x - 5) = x) : x = 50 :=
by
  sorry

end find_age_l448_448168


namespace min_moves_required_l448_448945

theorem min_moves_required (n : ℕ) : 
  ∃ (m : ℕ), m ≥ ⌊(n * n) / 3⌋ ∧ 
    ∀ config, initial_configuration config n → final_configuration config m → no_more_moves config := 
sorry

end min_moves_required_l448_448945


namespace parcels_division_l448_448296

theorem parcels_division (x y n : ℕ) (h : 5 + 2 * x + 3 * y = 4 * n) (hn : n = x + y) :
    n = 3 ∨ n = 4 ∨ n = 5 := 
sorry

end parcels_division_l448_448296


namespace carol_fixed_weekly_allowance_l448_448259

variable (A : ℝ) (C : ℝ) (total : ℝ) (weeks : ℝ) (avg_chores : ℝ)

theorem carol_fixed_weekly_allowance :
  (C = 1.5) →
  (weeks = 10) →
  (avg_chores = 15) →
  (total = 425) →
  (A * weeks + avg_chores * C * weeks = total) →
  (A = 20) :=
by
  intros
  sorry

end carol_fixed_weekly_allowance_l448_448259


namespace zero_of_function_l448_448128

theorem zero_of_function : (∃ x : ℝ, (λ x : ℝ, x + 1) x = 0) → (-1 : ℝ) :=
by
  sorry

end zero_of_function_l448_448128


namespace find_b10_l448_448780

def sequence (b : ℕ → ℕ) : Prop :=
  b 1 = 2 ∧ ∀ m n : ℕ, 1 ≤ m → 1 ≤ n → b (m + n) = b m + b n + 2 * m * n

theorem find_b10 (b : ℕ → ℕ) (h : sequence b) : b 10 = 110 := 
by {
  obtain ⟨h1, hmn⟩ := h,
  sorry
}

end find_b10_l448_448780


namespace zach_needs_more_money_zach_more_money_needed_l448_448541

/-!
# Zach's Bike Savings Problem
Zach needs $100 to buy a brand new bike.
Weekly allowance: $5.
Earnings from mowing the lawn: $10.
Earnings from babysitting: $7 per hour.
Zach has already saved $65.
He will receive weekly allowance on Friday.
He will mow the lawn and babysit for 2 hours this Saturday.
Prove that Zach needs $6 more to buy the bike.
-/

def zach_current_savings : ℕ := 65
def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_earnings : ℕ := 10
def babysitting_rate : ℕ := 7
def babysitting_hours : ℕ := 2

theorem zach_needs_more_money : zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)) = 94 :=
by sorry

theorem zach_more_money_needed : bike_cost - (zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours))) = 6 :=
by sorry

end zach_needs_more_money_zach_more_money_needed_l448_448541


namespace midpoints_coincide_l448_448409

-- Definitions of points, isosceles triangle with specific properties
variables {α : ℝ} -- angle α

structure Rectangle :=
(A B C D : ℝ × ℝ)
(horiz_segments : A.1 = B.1 ∧ C.1 = D.1)
(vert_segments : A.2 = D.2 ∧ B.2 = C.2)

structure IsoscelesTriangleOnBC (rect : Rectangle) :=
(K L M : ℝ × ℝ)
(angle_at_vertex : ∠ KLM = α)
(vertex_on_BC : L ∈ segment rect.B rect.C)
(base_on_AB_CD : K ∈ segment rect.A rect.B ∧ M ∈ segment rect.C rect.D)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoints_coincide (rect : Rectangle) (T1 T2 : IsoscelesTriangleOnBC rect) :
  midpoint T1.K T1.M = midpoint T2.K T2.M := sorry

end midpoints_coincide_l448_448409


namespace minimum_radius_l448_448900

-- Define the conditions of the problem including the unit square and the radius.

def unit_square := set.univ : set (ℝ × ℝ) -- Assuming we define a unit square in the Euclidean plane.

def circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  { p | dist p center ≤ radius }

def covers_unit_square (circles : list (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (x : ℝ) (y : ℝ), (0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1) →
  ∃ (center : ℝ × ℝ) (radius : ℝ), (center, radius) ∈ circles ∧ dist (x, y) center ≤ radius

-- The main statement: The minimum radius required for three congruent circles to cover the unit square.
theorem minimum_radius :
  ∃ r, r = sqrt 65 / 16 ∧
  ∀ (centers : list (ℝ × ℝ)), centers.length = 3 →
  covers_unit_square (centers.map (λ c, (c, sqrt 65 / 16))) :=
by
  sorry

end minimum_radius_l448_448900


namespace triangle_area_problem_l448_448976

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (5, -2)
def B : (ℝ × ℝ) := (10, 5)
def C : (ℝ × ℝ) := (5, 5)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  float.of_nat 1 / 2 * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)).abs

-- The theorem stating the area of the triangle with specific vertices
theorem triangle_area_problem : 
  triangle_area A B C = 17.5 := 
sorry

end triangle_area_problem_l448_448976


namespace sets_are_equal_l448_448123

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem sets_are_equal : X = Y :=
by sorry

end sets_are_equal_l448_448123


namespace distinct_products_count_is_26_l448_448366

open Finset

def set_numbers : Finset ℕ := {2, 3, 5, 7, 11}

def distinct_products_count (s : Finset ℕ) : ℕ :=
  let pairs := s.powerset.filter (λ t => 2 ≤ t.card)
  pairs.card

theorem distinct_products_count_is_26 : distinct_products_count set_numbers = 26 := by
  sorry

end distinct_products_count_is_26_l448_448366


namespace second_greatest_divisor_l448_448521

theorem second_greatest_divisor (n : ℕ) (h1 : ∃ d, nat.gcd 120 n = d ∧ nat.totient d = 4):
  ∃ k, k = 4 ∧ k = list.sort (≥) (nat.divisors (nat.gcd 120 n)).nth 1 :=
by {
  sorry
}

end second_greatest_divisor_l448_448521


namespace quadratic_has_two_distinct_real_roots_l448_448677

theorem quadratic_has_two_distinct_real_roots (a : ℝ) (h : 16 + 4 * a > 0) : a > -4 ∧ a ≠ 0 :=
by
  have discriminant_pos := h
  have factor_inq : 16 + 4 * a > 0 := discriminant_pos
  sufficient_conditions: (a > -4) ∧ (a ≠ 0) sorry

end quadratic_has_two_distinct_real_roots_l448_448677


namespace opposite_number_in_circle_l448_448025

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l448_448025


namespace andrew_bins_problem_l448_448252

theorem andrew_bins_problem :
  let a : ℕ → ℕ := λ n,
    if n = 1 then 2 else
    if n = 2 then 4 else
    if n = 3 then 7 else
    if n = 5 then 16 else
    sorry -- to define the general formula
  in 
  a 4 = 11 :=
begin
  let a : ℕ → ℕ := λ n,
    if n = 1 then 2 else
    if n = 2 then 4 else
    if n = 3 then 7 else
    if n = 5 then 16 else
    sorry, -- to define the general formula
  have h1 : a 1 = 2 := rfl,
  have h2 : a 2 = 4 := rfl,
  have h3 : a 3 = 7 := rfl,
  have h5 : a 5 = 16 := rfl,
  -- Derive the general formula for the pattern
  sorry
end

end andrew_bins_problem_l448_448252


namespace distance_between_points_l448_448633

theorem distance_between_points :
  let p1 := (3, 4, 0)
  let p2 := (6, 2, 7)
  dist p1 p2 = Real.sqrt 62 :=
by
  sorry

noncomputable def dist (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2 + (q.3 - p.3)^2)

end distance_between_points_l448_448633


namespace range_g_l448_448291

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + Real.pi * Real.arcsin (x / 3) - (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9 * x + 27)

theorem range_g :
  set.range g = set.Icc (11 * Real.pi^2 / 24) (59 * Real.pi^2 / 24) := by
  sorry

end range_g_l448_448291


namespace enlarged_sticker_height_l448_448017

-- Define the conditions as hypotheses
def original_width : ℝ := 3
def original_height : ℝ := 2
def new_width : ℝ := 12
def proportional_factor := new_width / original_width

-- Define the theorem to prove the new height
theorem enlarged_sticker_height :
  let new_height := original_height * proportional_factor in
  new_height = 8 :=
by
  sorry

end enlarged_sticker_height_l448_448017


namespace votes_polled_l448_448547

theorem votes_polled (V : ℕ) (h1 : 60% of V is winning) (h2 : wins by 1040 votes) : V = 5200 :=
sorry

end votes_polled_l448_448547


namespace probability_shadedRegion_l448_448246

noncomputable def triangleVertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((0, 0), (0, 5), (5, 0))

noncomputable def totalArea : ℝ :=
  12.5

noncomputable def shadedArea : ℝ :=
  4.5

theorem probability_shadedRegion (x y : ℝ) :
  let p := (x, y)
  let condition := x + y <= 3
  let totalArea := 12.5
  let shadedArea := 4.5
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 5 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 ∧ p.1 + p.2 ≤ 5}) →
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ p.1 + p.2 ≤ 3}) →
  (shadedArea / totalArea) = 9/25 :=
by
  sorry

end probability_shadedRegion_l448_448246


namespace jake_hours_eighth_week_to_meet_requirement_l448_448762

theorem jake_hours_eighth_week_to_meet_requirement :
  let average_hours := 12
  let total_weeks := 8
  let week_hours := [14, 9, 12, 15, 11, 13, 10]
  let sum_week_hours := week_hours.sum
  let hours_eighth_week := 12
  (sum_week_hours + hours_eighth_week) / total_weeks = average_hours
  sorry

end jake_hours_eighth_week_to_meet_requirement_l448_448762


namespace odd_phone_calls_are_even_l448_448939

theorem odd_phone_calls_are_even (n : ℕ) : Even (2 * n) :=
by
  sorry

end odd_phone_calls_are_even_l448_448939


namespace opposite_of_113_is_114_l448_448048

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l448_448048


namespace correct_statement_count_l448_448114

def is_valid_input_format (s : String) : Prop := 
  s = "a, b, c="

def is_valid_output_format (s : String) : Prop := 
  s = "\"S=\" , 7"

def is_valid_assignment (s : String) : Prop := 
  s = "r = 9"

def is_valid_arithmetic_print (s : String) : Prop := 
  s = "20.3*2"

def count_correct_statements (stmts : List String) : Nat := 
  stmts.filter (fun s => match s with
                         | "INPUT \"a, b, c=\"; a, b; c" => is_valid_input_format "a, b, c="
                         | "PRINT S=7" => is_valid_output_format "\"S=\" , 7"
                         | "9=r" => is_valid_assignment "r = 9"
                         | "PRINT 20.3*2" => is_valid_arithmetic_print "20.3*2"
                         | _ => false
                         ).length

theorem correct_statement_count : count_correct_statements ["INPUT \"a, b, c=\"; a, b; c", "PRINT S=7", "9=r", "PRINT 20.3*2"] = 1 :=
  by
    sorry

end correct_statement_count_l448_448114


namespace determine_monotonically_increasing_interval_l448_448225

noncomputable theory
open Real

/-- Problem Statement: Given the function f(x) = 7 * sin(x - π / 6), prove that it is monotonically increasing in the interval (0, π/2). -/
def monotonically_increasing_interval (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2

def function_f (x : ℝ) : ℝ :=
  7 * sin (x - π / 6)

theorem determine_monotonically_increasing_interval :
  ∀ x : ℝ, monotonically_increasing_interval x → monotone_on function_f (set.Ioo 0 (π / 2)) :=
sorry

end determine_monotonically_increasing_interval_l448_448225


namespace calc_product_l448_448616

def x : ℝ := 150.15
def y : ℝ := 12.01
def z : ℝ := 1500.15
def w : ℝ := 12

theorem calc_product :
  x * y * z * w = 32467532.8227 :=
by
  sorry

end calc_product_l448_448616


namespace monotonic_interval_l448_448240

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin(x - Real.pi / 6)

def is_monotonically_increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem monotonic_interval : is_monotonically_increasing_interval f 0 (Real.pi / 2) :=
sorry

end monotonic_interval_l448_448240


namespace midpoint_dist_eq_l448_448788

variable {A B C M : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]

noncomputable def dist := Metric.dist

theorem midpoint_dist_eq
  (hM : dist B C = 2 * dist M B)
  : dist A M ^ 2 = (2 * dist A B ^ 2 + 2 * dist A C ^ 2 - dist B C ^ 2) / 4 :=
  sorry

end midpoint_dist_eq_l448_448788


namespace find_n_from_A_k_l448_448784

theorem find_n_from_A_k (n : ℕ) (A : ℕ → ℕ) (h1 : A 1 = Int.natAbs (n + 1))
  (h2 : ∀ k : ℕ, k > 0 → A k = Int.natAbs (n + (2 * k - 1)))
  (h3 : A 100 = 2005) : n = 1806 :=
sorry

end find_n_from_A_k_l448_448784


namespace total_balloons_are_48_l448_448378

theorem total_balloons_are_48 
  (brooke_initial : ℕ) (brooke_add : ℕ) (tracy_initial : ℕ) (tracy_add : ℕ)
  (brooke_half_given : ℕ) (tracy_third_popped : ℕ) : 
  brooke_initial = 20 →
  brooke_add = 15 →
  tracy_initial = 10 →
  tracy_add = 35 →
  brooke_half_given = (brooke_initial + brooke_add) / 2 →
  tracy_third_popped = (tracy_initial + tracy_add) / 3 →
  (brooke_initial + brooke_add - brooke_half_given) + (tracy_initial + tracy_add - tracy_third_popped) = 48 := 
by
  intros
  sorry

end total_balloons_are_48_l448_448378


namespace initial_dragon_fruits_remaining_kiwis_l448_448549

variable (h d k : ℕ)    -- h: initial number of cantaloupes, d: initial number of dragon fruits, k: initial number of kiwis
variable (d_rem : ℕ)    -- d_rem: remaining number of dragon fruits after all cantaloupes are used up
variable (k_rem : ℕ)    -- k_rem: remaining number of kiwis after all cantaloupes are used up

axiom condition1 : d = 3 * h + 10
axiom condition2 : k = 2 * d
axiom condition3 : d_rem = 130
axiom condition4 : (d - d_rem) = 2 * h
axiom condition5 : k_rem = k - 10 * h

theorem initial_dragon_fruits (h : ℕ) (d : ℕ) (k : ℕ) (d_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  d_rem = 130 →
  2 * h + d_rem = d → 
  h = 120 → 
  d = 370 :=
by 
  intros
  sorry

theorem remaining_kiwis (h : ℕ) (d : ℕ) (k : ℕ) (k_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  h = 120 →
  k_rem = k - 10 * h → 
  k_rem = 140 :=
by 
  intros
  sorry

end initial_dragon_fruits_remaining_kiwis_l448_448549


namespace cistern_holds_70_liters_l448_448189

noncomputable def cistern_capacity :=
  let C := 70 in
  let leak1_rate := C / 20 in
  let leak2_rate := C / 30 in
  let tap1_rate := 4 in
  let tap2_rate := 3 in
  let total_leak_rate := leak1_rate + leak2_rate in
  let total_tap_rate := tap1_rate + tap2_rate in
  let net_rate := total_tap_rate - total_leak_rate in
  (60 * net_rate = C)

theorem cistern_holds_70_liters : cistern_capacity = true :=
by
  sorry

end cistern_holds_70_liters_l448_448189


namespace m_plus_n_eq_123_l448_448896

/- Define the smallest prime number -/
def m : ℕ := 2

/- Define the largest integer less than 150 with exactly three positive divisors -/
def n : ℕ := 121

/- Prove that the sum of m and n is 123 -/
theorem m_plus_n_eq_123 : m + n = 123 := by
  -- By definition, m is 2 and n is 121
  -- So, their sum is 123
  rfl

end m_plus_n_eq_123_l448_448896


namespace shaded_region_area_l448_448848

def area_shaded_region (r1 r2 : Real) (h : r1 * r2 = 9 / 4) : Real :=
  let larger_circle_area := π * (r1 + r2) ^ 2
  let smaller_circle_area1 := π * r1 ^ 2
  let smaller_circle_area2 := π * r2 ^ 2
  larger_circle_area - smaller_circle_area1 - smaller_circle_area2

theorem shaded_region_area (r1 r2 : Real) (h : r1 * r2 = 9 / 4) :
  area_shaded_region r1 r2 h = 9 * π / 2 := by
  sorry

end shaded_region_area_l448_448848


namespace minimum_routes_l448_448394

theorem minimum_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) :
  a + b + c ≥ 21 :=
by sorry

end minimum_routes_l448_448394


namespace monotonic_increasing_l448_448236

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π / 6)

theorem monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 2) → f x1 < f x2 := sorry

end monotonic_increasing_l448_448236


namespace cos_double_angle_l448_448371

theorem cos_double_angle (φ : ℝ) (h : ∑' n : ℕ, sin (2 * n) φ = 4) : cos (2 * φ) = -1 / 2 :=
sorry

end cos_double_angle_l448_448371


namespace solve_for_m_l448_448361

-- Define the conditions for the lines being parallel
def condition_one (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + m * y + 3 = 0

def condition_two (m : ℝ) : Prop :=
  ∃ x y : ℝ, (m - 1) * x + 2 * m * y + 2 * m = 0

def are_parallel (A B C D : ℝ) : Prop :=
  A * D = B * C

theorem solve_for_m :
  ∀ (m : ℝ),
    (condition_one m) → 
    (condition_two m) → 
    (are_parallel 1 m 3 (2 * m)) →
    (m = 0) :=
by
  intro m h1 h2 h_parallel
  sorry

end solve_for_m_l448_448361


namespace inequality_always_holds_l448_448698

theorem inequality_always_holds
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - 2^x) / (1 + 2^x))
  (h_odd : ∀ x, f (-x) = -f x)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_ineq : f (2 * a + b) + f (4 - 3 * b) > 0)
  : b - a > 2 :=
sorry

end inequality_always_holds_l448_448698


namespace find_b_vector_l448_448778

open Real

def vector := ℝ × ℝ × ℝ

noncomputable def dot (v₁ v₂ : vector) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

noncomputable def cross (v₁ v₂ : vector) : vector :=
(v₁.2 * v₂.3 - v₁.3 * v₂.2, v₁.3 * v₂.1 - v₁.1 * v₂.3, v₁.1 * v₂.2 - v₁.2 * v₂.1)

theorem find_b_vector
  (a : vector := (3, -2, 4))
  (c : vector := (4, 1, -5)) :
  ∃ b : vector, dot a (cross b c) = 0 ∧ b = (7, -1, -1) :=
by {
  sorry
}

end find_b_vector_l448_448778


namespace find_cos_alpha_l448_448664

theorem find_cos_alpha (α : ℝ) (h : (1 - Real.cos α) / Real.sin α = 3) : Real.cos α = -4/5 :=
by
  sorry

end find_cos_alpha_l448_448664


namespace magnitude_of_transformed_vector_l448_448713

theorem magnitude_of_transformed_vector :
  ∀ (x y : ℝ), 
    let a : ℝ × ℝ := (x, y), 
    let b : ℝ × ℝ := (-1, 2),
    a + b = (1, 3) →
    ‖a - 2 • b‖ = 5
:= 
begin
  intros x y a b h,
  let c := a - 2 • b,
  have : c = (4, -3),
  sorry
end

end magnitude_of_transformed_vector_l448_448713


namespace y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l448_448425

def y := 96 + 144 + 200 + 300 + 600 + 720 + 4800

theorem y_is_multiple_of_4 : y % 4 = 0 := 
by sorry

theorem y_is_not_multiple_of_8 : y % 8 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_16 : y % 16 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_32 : y % 32 ≠ 0 := 
by sorry

end y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l448_448425


namespace total_distance_ball_fifth_hit_l448_448179

theorem total_distance_ball_fifth_hit (h₀ : ℝ) (rebound_ratio : ℝ) :
  h₀ = 150 ∧ rebound_ratio = 3 / 4 →
  let h₁ := h₀ * rebound_ratio,
      h₂ := h₁ * rebound_ratio,
      h₃ := h₂ * rebound_ratio,
      h₄ := h₃ * rebound_ratio,
      h₅ := h₄ * rebound_ratio,
      total_distance := h₀ + h₁ + h₁ + h₂ + h₂ + h₃ + h₃ + h₄ + h₄ + h₅
  in total_distance = 765.703125 :=
by
  intro h₀ h₁ h₂ h₃ h₄ h₅ h₆
  sorry -- proof not required

end total_distance_ball_fifth_hit_l448_448179


namespace max_value_y_l448_448507

noncomputable def y (x : ℝ) : ℝ := x * (3 - 2 * x)

theorem max_value_y : ∃ x, 0 < x ∧ x < (3:ℝ) / 2 ∧ y x = 9 / 8 :=
by
  sorry

end max_value_y_l448_448507


namespace opposite_of_113_is_114_l448_448051

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l448_448051


namespace opposite_number_113_is_114_l448_448040

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l448_448040


namespace project_completion_days_l448_448605

theorem project_completion_days (D : ℕ) : 
  let work_rate_per_man := (2.5 / (35 * 100)) in
  let remaining_work := 15 - 2.5 in
  let total_men_after_hiring := 35 + 52.5 in
  let new_work_rate := remaining_work / (total_men_after_hiring * (D - 100)) in
  D = 300 :=
by
  sorry

end project_completion_days_l448_448605


namespace p_partitionable_divides_l448_448773

variable (p n : ℕ)

def p_partitionable (p n : ℕ) : Prop :=
  ∃ (T : Finset ℕ), T = (Finset.range (n + 1)).filter (λ x, x > 0) ∧ 
  ∃ (S : Finset (Finset ℕ)), 
    S.card = p ∧ 
    ∀ s ∈ S, (s ⊆ T ∧ ∑ x in s, x = (n * (n + 1)) / (2 * p))

theorem p_partitionable_divides (p : ℕ) [hp : Nat.Prime p] (n : ℕ) (h : p_partitionable p n) :
  p ∣ n ∨ p ∣ (n + 1) :=
sorry

end p_partitionable_divides_l448_448773


namespace alice_total_pints_wednesday_l448_448805

-- Define pints of ice cream Alice bought each day
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := (1 / 3 : ℝ) * pints_monday.toReal
def pints_returned_wednesday : ℝ := (1 / 2 : ℝ) * pints_tuesday
def pints_on_wednesday : ℝ := pints_sunday.toReal + pints_monday.toReal + pints_tuesday - pints_returned_wednesday

theorem alice_total_pints_wednesday : pints_on_wednesday = 18 := by
  sorry

end alice_total_pints_wednesday_l448_448805


namespace equal_numbers_exist_l448_448101

theorem equal_numbers_exist (a : ℕ → ℕ) (n : ℕ) (h1 : (∑ i in finset.range n, a i : ℚ) / n = 20.22) :
  ∃ i j, i < n ∧ j < n ∧ i ≠ j ∧ a i = a j :=
sorry

end equal_numbers_exist_l448_448101


namespace opposite_number_113_is_114_l448_448056

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l448_448056


namespace train_crossing_time_approx_l448_448205

noncomputable def train_crossing_time (l_train : ℝ) (l_platform : ℝ) (v_train_kmph : ℝ) : ℝ :=
  let total_distance := l_train + l_platform
  let speed_m_s := v_train_kmph * (1000 / 3600)
  total_distance / speed_m_s

theorem train_crossing_time_approx (l_train : ℝ) (l_platform : ℝ) (v_train_kmph : ℝ) :
  l_train = 140 → l_platform = 233.3632 → v_train_kmph = 84 → abs (train_crossing_time l_train l_platform v_train_kmph - 16) < 1 :=
by
  intros hltrain hlplatform hvtrain
  unfold train_crossing_time
  rw [hltrain, hlplatform, hvtrain]
  norm_num
  let t_apx : ℝ := 16
  linarith

#check train_crossing_time_approx

end train_crossing_time_approx_l448_448205


namespace max_ab_l448_448791

theorem max_ab (a b : ℝ) :
  (∃ (z : ℂ), |z| ≤ 1 ∧ z + z.conj * complex.norm z = a + b * complex.I) → ab ≤ 1 / 8 :=
by  
sorry

end max_ab_l448_448791


namespace find_x_solutions_l448_448009

theorem find_x_solutions (x : ℝ) :
  let f (x : ℝ) := x^2 - 4*x + 1
  let f2 (x : ℝ) := (f x)^2
  f (f x) = f2 x ↔ x = 2 + (Real.sqrt 13) / 2 ∨ x = 2 - (Real.sqrt 13) / 2 := by
  sorry

end find_x_solutions_l448_448009


namespace segment_length_l448_448724

-- Definitions based on the problem conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0) -- Focus of the parabola y^2 = 4x
def on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2 -- Point P lies on the parabola
def midpoint (P Q : ℝ × ℝ) : ℝ := (P.1 + Q.1) / 2 -- Midpoint formula for x-coordinates

-- Given points M and N on the parabola
variables (M N : ℝ × ℝ)
-- The straight line passes through the focus
variable (L : ℝ × ℝ → ℝ × ℝ → Prop)

-- Conditions: The line L passes through the focus and intersects the parabola at points M and N
axiom line_focus (M N : ℝ × ℝ) : L focus M ∧ L focus N
axiom points_on_parabola : on_parabola M ∧ on_parabola N
axiom midpoint_condition : midpoint M N = 3

-- Proof that the length of segment MN is 8
theorem segment_length : (M.1 + N.1) + 2 = 8 :=
by {
  have h1 : midpoint M N = 3 := midpoint_condition,
  have h2 : (M.1 + N.1) / 2 = 3 := h1,
  have h3 : M.1 + N.1 = 6 := by linarith,
  have h4 : (M.1 + N.1) + 2 = 8 := by linarith,
  exact h4,
}

end segment_length_l448_448724


namespace hyperbola_properties_l448_448078

open Real

def hyperbola_equation : Prop := ∀ x y : ℝ, y^2 / 9 - x^2 / 16 = 1

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola_equation → (x = 4 * sqrt(1 + y^2 / 9))) ∧
  (∀ x y : ℝ, hyperbola_equation → 
    (y = 3 * sqrt(1 + x^2 / 16) ∨ y = -3 * sqrt(1 + x^2 / 16))
  ) := sorry

end hyperbola_properties_l448_448078


namespace angle_of_inclination_of_focal_chord_l448_448671

theorem angle_of_inclination_of_focal_chord 
  (A B : ℝ × ℝ) 
  (length_AB : real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12)
  (passes_through_focus : ∃ c : ℝ × ℝ, c = (2.25, 0) ∧ (A = c ∨ B = c))
  (parabola : ∀ (x y : ℝ), y^2 = 9 * x ↔ (x, y) ∈ set.range (λ t : ℝ, (t^2 / 9, t))) :
  ∃ α : ℝ,
  (α = real.pi / 3 ∨ α = 2 * real.pi / 3) ∧ 
  (∃ x1 y1 x2 y2: ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ real.atan2 (y2 - y1) (x2 - x1) = α) :=
by sorry

end angle_of_inclination_of_focal_chord_l448_448671


namespace length_RS_l448_448000

theorem length_RS (A B C H R S M : Type) [euclidean_geometry] 
  (hCH : altitude CH A B C) 
  (hR : tangent_point (inscribed_circle A C H) R CH) 
  (hS : tangent_point (inscribed_circle B C H) S CH) 
  (hM : midpoint M A B) 
  (hAB : distance A B = 20) 
  (hAC : distance A C = 21) 
  (hBC : distance B C = 29)
  : distance R S = 4 := 
sorry

end length_RS_l448_448000


namespace square_ratio_l448_448079

noncomputable def sideLength (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area (sideLength : ℕ) : ℕ :=
  sideLength * sideLength

theorem square_ratio (perimeter_A perimeter_B : ℕ) 
  (hA : perimeter_A = 20) 
  (hB : perimeter_B = 40) 
  (sideLength_C : ℕ) 
  (hC : sideLength_C = 2 * sideLength perimeter_B) : 
  (area (sideLength perimeter_A) : ℚ) / area sideLength_C = 1 / 16 := 
by
  let side_A := sideLength perimeter_A
  let side_B := sideLength perimeter_B
  let side_C := 2 * side_B
  have hA_side : side_A = 5 := by sorry
  have hB_side : side_B = 10 := by sorry
  have hC_side : side_C = 20 := by sorry
  let area_A := area side_A
  let area_C := area side_C
  have hA_area : area_A = 25 := by sorry
  have hC_area : area_C = 400 := by sorry
  calc
    (area side_A : ℚ) / area side_C = 25 / 400 := by sorry
                            ... = 1/16 := by sorry

end square_ratio_l448_448079


namespace product_of_positive_real_solutions_l448_448383

theorem product_of_positive_real_solutions (x : ℂ) (h : x^8 = -256) :
  (∀ x, (realPart x > 0) → x^8 = -256 → (∏ s in {x | (realPart s) > 0}, s) = 4 := sorry

end product_of_positive_real_solutions_l448_448383


namespace opposite_number_113_is_13_l448_448036

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l448_448036


namespace find_m_l448_448606

variables (a : ℕ → ℝ) (r : ℝ) (m : ℕ)

-- Define the conditions of the problem
def exponential_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

def condition_1 (a : ℕ → ℝ) (r : ℝ) : Prop :=
  a 5 * a 6 + a 4 * a 7 = 18

def condition_2 (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a 1 * a m = 9

-- The theorem to prove based on the given conditions
theorem find_m
  (h_exp : exponential_sequence a r)
  (h_r_ne_1 : r ≠ 1)
  (h_cond1 : condition_1 a r)
  (h_cond2 : condition_2 a m) :
  m = 10 :=
sorry

end find_m_l448_448606


namespace average_score_l448_448410

theorem average_score (s1 s2 s3 : ℕ) (n : ℕ) (h1 : s1 = 115) (h2 : s2 = 118) (h3 : s3 = 115) (h4 : n = 3) :
    (s1 + s2 + s3) / n = 116 :=
by
    sorry

end average_score_l448_448410


namespace isosceles_right_triangle_exists_l448_448622

open Set

def grid_4x4 : Finset (Fin 4 × Fin 4) := 
  univ.product univ

theorem isosceles_right_triangle_exists
  (blue_points : Finset (Fin 4 × Fin 4))
  (h_bsize : Blue_points.card = 7) : 
  ∃ (a b c : Fin 4 × Fin 4), 
  a ∈ blue_points ∧ b ∈ blue_points ∧ c ∈ blue_points ∧ 
  (dist a b = dist b c ∧ dist a c = dist a b * sqrt 2) :=
begin
  sorry
end

/- Definitions used -/
# Definitions:
- grid: A \(4 \times 4\) grid (set of 16 points)
- blue_points: Any 7 blue points from this grid
# Conditions:
- ∀ blue_points: blue_points ∈ grid, card(blue_points) = 7
# Prove:
- the existence of three points a, b, c within blue_points which form an isosceles right-angled triangle
-/


end isosceles_right_triangle_exists_l448_448622


namespace coefficient_x3_expansion_l448_448281

theorem coefficient_x3_expansion :
  let f := λ n, (1 + n) ^ n
  (3 : ℕ), (20 : ℕ)
  let expression := ∑ i in (range (20 - 3 + 1)), (1 + x) ^ (i + 3)
  coefficient (x^3) expression = 5985 :=
by
  sorry

end coefficient_x3_expansion_l448_448281


namespace probability_of_valid_arrangement_l448_448254

def is_valid_arrangement (arr : List Bool) : Prop :=
  ∀ i, (∑ j in List.range (List.length arr - i), if arr.get? (i + j) = some true then 1 else 0) ≥
       (∑ j in List.range (List.length arr - i), if arr.get? (i + j) = some false then 1 else 0)

def valid_arrangements : Finset (List Bool) :=
  (List.permutations [true, true, true, false, false, false]).toFinset.filter is_valid_arrangement

def total_arrangements : Finset (List Bool) :=
  (List.permutations [true, true, true, false, false, false]).toFinset

theorem probability_of_valid_arrangement :
  valid_arrangements.card.toReal / total_arrangements.card.toReal = 1/4 := 
by
  sorry

end probability_of_valid_arrangement_l448_448254


namespace six_digit_prime_digit_unique_l448_448866

theorem six_digit_prime_digit_unique :
  (∃! (A : ℕ), A ∈ {1, 3, 5, 7, 9} ∧ Prime (103860 + A)) :=
by {
  sorry
}

end six_digit_prime_digit_unique_l448_448866


namespace lesser_fraction_of_sum_and_product_l448_448513

open Real

theorem lesser_fraction_of_sum_and_product (a b : ℚ)
  (h1 : a + b = 11 / 12)
  (h2 : a * b = 1 / 6) :
  min a b = 1 / 4 :=
sorry

end lesser_fraction_of_sum_and_product_l448_448513


namespace sum_of_place_values_of_sevens_l448_448157

noncomputable def calculate_expression : ℕ → ℕ → ℝ := λ a b, (8793.07152 * (a / b)).round 4

noncomputable def three_sevens_place_values : ℕ → ℕ → ℝ := 
  λ a b, let result := calculate_expression a b in
         let digits := result.to_digits in
         let places := digits.filter (λ d, d = 7) in
         (if places.length >= 3 then
            let [thousands, tens, tenths] := [7000.0, 70.0, 0.7] in
            thousands + tens + tenths
          else
            0.0)

theorem sum_of_place_values_of_sevens : three_sevens_place_values 4727 707 = 7070.7 := sorry

end sum_of_place_values_of_sevens_l448_448157


namespace inequality_xy_yz_zx_l448_448317

theorem inequality_xy_yz_zx {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) <= 1 / 4 * (Real.sqrt 33 + 1) :=
sorry

end inequality_xy_yz_zx_l448_448317


namespace sharks_win_at_least_five_games_l448_448844

-- Definitions based on the conditions
def number_of_games : ℕ := 9
def probability_of_win : ℚ := 1 / 2

-- Definition to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

-- Definition to calculate binomial probability
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  binomial n k * (p^k) * ((1-p)^(n-k))

-- Main theorem statement
theorem sharks_win_at_least_five_games :
  (finset.range (number_of_games + 1)).filter (λ k, k ≥ 5).sum (λ k, binomial_probability number_of_games k probability_of_win) = 1 / 2 := 
begin
  sorry
end

end sharks_win_at_least_five_games_l448_448844


namespace probability_queen_then_diamond_l448_448893

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l448_448893


namespace find_certain_number_l448_448934

def certain_number (x : ℚ) : Prop := 5 * 1.6 - (1.4 * x) / 1.3 = 4

theorem find_certain_number : certain_number (-(26/7)) :=
by 
  simp [certain_number]
  sorry

end find_certain_number_l448_448934


namespace tank_filled_to_depth_l448_448944

noncomputable def tank_volume (R H r d : ℝ) : ℝ := R^2 * H * Real.pi - (r^2 * H * Real.pi)

theorem tank_filled_to_depth (R H r d : ℝ) (h_cond : R = 5 ∧ H = 12 ∧ r = 2 ∧ d = 3) :
  tank_volume R H r d = 110 * Real.pi - 96 :=
sorry

end tank_filled_to_depth_l448_448944


namespace stamps_problem_l448_448567

theorem stamps_problem (x y : ℕ) : 
  2 * x + 6 * x + 5 * y / 2 = 60 → x = 5 ∧ y = 8 ∧ 6 * x = 30 :=
by 
  sorry

end stamps_problem_l448_448567


namespace scientific_notation_correct_l448_448839

theorem scientific_notation_correct : 657000 = 6.57 * 10^5 :=
by
  sorry

end scientific_notation_correct_l448_448839


namespace problem_solution_l448_448683

open Function

-- Definitions of the points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-3, 2⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨4, 1⟩
def D : Point := ⟨-2, 4⟩

-- Definitions of vectors
def vec (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

-- Definitions of conditions
def AB := vec A B
def AD := vec A D
def DC := vec D C

-- Definitions of dot product to check orthogonality
def dot (v w : Point) : ℝ := v.x * w.x + v.y * w.y

-- Lean statement to prove the conditions
theorem problem_solution :
  AB ≠ ⟨-4, 2⟩ ∧
  dot AB AD = 0 ∧
  AB.y * DC.x = AB.x * DC.y ∧
  ((AB.y * DC.x = AB.x * DC.y) ∧ (dot AB AD = 0) → 
  (∃ a b : ℝ, a ≠ b ∧ (a = 0 ∨ b = 0) ∧ AB = ⟨a, -a⟩  ∧ DC = ⟨3 * a, -3 * a⟩)) :=
by
  -- Proof omitted
  sorry

end problem_solution_l448_448683


namespace range_of_a_l448_448347

def is_extreme (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ (ε : ℝ), 0 < ε ∧ ∀ (y : ℝ), y ∈ set.Ioo (x - ε) (x + ε) → f y ≠ f x

def has_two_extremes (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ is_extreme f x ∧ is_extreme f y

theorem range_of_a (f : ℝ → ℝ) (n : ℕ) :
  (∃ n : ℕ, has_two_extremes (λ x, (a: ℝ) * exp x - x * log x) (n:ℝ) (n+2:ℝ)) ↔
  a ∈ set.Ioo ((log 2 * exp 1) / (exp 2)) (1 / exp 1) := sorry

end range_of_a_l448_448347


namespace Irene_age_is_46_l448_448274

-- Definitions based on the given conditions
def Eddie_age : ℕ := 92
def Becky_age : ℕ := Eddie_age / 4
def Irene_age : ℕ := 2 * Becky_age

-- Theorem we aim to prove that Irene's age is 46
theorem Irene_age_is_46 : Irene_age = 46 := by
  sorry

end Irene_age_is_46_l448_448274


namespace man_overtime_hours_correctness_l448_448544

def man_worked_overtime_hours (r h_r t : ℕ): ℕ :=
  let regular_pay := r * h_r
  let overtime_pay := t - regular_pay
  let overtime_rate := 2 * r
  overtime_pay / overtime_rate

theorem man_overtime_hours_correctness : man_worked_overtime_hours 3 40 186 = 11 := by
  sorry

end man_overtime_hours_correctness_l448_448544


namespace sequence_formula_l448_448854

-- Defining the sequence and the conditions
def bounded_seq (a : ℕ → ℝ) : Prop :=
  ∃ C > 0, ∀ n, |a n| ≤ C

-- Statement of the problem in Lean
theorem sequence_formula (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = 3 * a n - 4) →
  bounded_seq a →
  ∀ n : ℕ, a n = 2 :=
by
  intros h1 h2
  sorry

end sequence_formula_l448_448854


namespace median_length_l448_448748

namespace TriangleMedian

theorem median_length (D E F : Type*) [LinearOrderedField D] [NormedAddCommGroup E] 
  [NormedSpace D E] (a b c : E) (F_is_right : ∠b c a = 90) (angle_DEF : ∠b a c = 30)
  (DF_len : ∥c - a∥ = 6) (N : E) (N_midpoint : N = midpoint D (a, c)) :
  ∥N - b∥ = 3.4 :=
sorry

end TriangleMedian

end median_length_l448_448748


namespace sum_of_digits_of_N_is_12_l448_448777

open Nat

def divisible_by_all (x : ℕ) (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → m > 0 → m ∣ x

def second_smallest_multiple (n : ℕ) : ℕ :=
  let lcm_n := lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 7)))))
  2 * lcm_n

def sum_of_digits (n : ℕ) : ℕ :=
  (natDigits 10 n).sum

theorem sum_of_digits_of_N_is_12 : sum_of_digits (second_smallest_multiple 8) = 12 :=
  by
  sorry

end sum_of_digits_of_N_is_12_l448_448777


namespace monotonically_increasing_interval_l448_448227

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l448_448227


namespace iron_balls_molded_l448_448991

noncomputable def volume_of_iron_bar (l w h : ℝ) : ℝ :=
  l * w * h

theorem iron_balls_molded (l w h n : ℝ) (volume_of_ball : ℝ) 
  (h_l : l = 12) (h_w : w = 8) (h_h : h = 6) (h_n : n = 10) (h_ball_volume : volume_of_ball = 8) :
  (n * volume_of_iron_bar l w h) / volume_of_ball = 720 :=
by 
  rw [h_l, h_w, h_h, h_n, h_ball_volume]
  rw [volume_of_iron_bar]
  sorry

end iron_balls_molded_l448_448991


namespace tangent_points_locus_l448_448074

theorem tangent_points_locus (A B C : Point) (h_collinear : collinear A B C) (h_between : between B A C) :
  let CA := distance C A
      CB := distance C B
      r := sqrt (CA * CB)
  in
  locus_of_tangency A B C = circle_centered_at C r \setminus {A, B} :=
by
  sorry

end tangent_points_locus_l448_448074


namespace smallest_possible_value_l448_448004

open Complex

theorem smallest_possible_value (z : ℂ) (h : |z^2 - 1| = |z * (z - Complex.I)|) :
  ∃ r : ℝ, (∀ z' : ℂ, |z'^2 - 1| = |z' * (z' - Complex.I)| → |z' - Complex.I| ≥ r) ∧ r = 3 * Real.sqrt 2 / 4 :=
by sorry

end smallest_possible_value_l448_448004


namespace can_transform_to_target_l448_448610

def initial_seq : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def target_seq : list ℕ := [9, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

/-- Transformation function that reverses any four consecutive elements in a list. -/
def transform (l : list ℕ) (i : ℕ) : list ℕ :=
  if i + 3 < l.length then
    let (a, r1) := l.split_at i in
    let (b, r2) := r1.split_at 4 in
    a ++ b.reverse ++ r2
  else l

/-- The main theorem stating that we can transform the initial sequence to the target sequence. -/
theorem can_transform_to_target : ∃ (n : ℕ) (f : fin n → ℕ), 
  (∀ i, i < n → f i < 9) ∧ 
  (∀ l (h : l = initial_seq → (∀ i (hi : i < n), l = transform l (f i)) → l = target_seq)) :=
sorry

end can_transform_to_target_l448_448610


namespace boat_trip_distance_l448_448127

theorem boat_trip_distance
  (boat_speed : ℕ) (stream_speed : ℕ) (total_time : ℕ) (D : ℕ)
  (h_boat_speed : boat_speed = 9)
  (h_stream_speed : stream_speed = 6)
  (h_total_time : total_time = 84) :
  D = 210 :=
by
  have h1 : boat_speed + stream_speed = 15, from by rw [h_boat_speed, h_stream_speed]; norm_num,
  have h2 : boat_speed - stream_speed = 3, from by rw [h_boat_speed, h_stream_speed]; norm_num,
  have h3 : D / 15.0 + D / 3.0 = total_time, from sorry,
  have h4 : D * 6 / 15 = total_time, from sorry,
  have h5 : D * 6 / 15 * 15 = total_time * 15, from sorry,
  have h6 : D * 6 = total_time * 15, from sorry,
  have h7 : D = (total_time * 15) / 6, from sorry,
  calc
    D = (total_time * 15) / 6 : by exact h7
    ... = 210 : by rw h_total_time; norm_num

end boat_trip_distance_l448_448127


namespace find_group_2018_l448_448636

theorem find_group_2018 :
  ∃ n : ℕ, 2 ≤ n ∧ 2018 ≤ 2 * n * (n + 1) ∧ 2018 > 2 * (n - 1) * n :=
by
  sorry

end find_group_2018_l448_448636


namespace find_a_c_find_cos_diff_l448_448388

noncomputable def triangle_ABC : Type :=
  {a b c : ℝ} × {A B C : ℝ} 

variables (a b c : ℝ) (A B C : ℝ)
variable (abc : triangle_ABC)

def side_conditions (a c : ℝ) : Prop := 
  a > c

def dot_product_condition (abc : triangle_ABC) : Prop := 
  let ⟨a, b, c⟩ := abc.1 in 
  ⟨A, B, C⟩ := abc.2 in
  (-(a * c * (real.cos B))) = 2

def cosB_condition (B : ℝ) : Prop := 
  B = real.cos (1 / 3)

def b_value : ℝ := 3

theorem find_a_c (a c : ℝ) (abc : triangle_ABC) :
  side_conditions a c → dot_product_condition abc → cosB_condition B → b = 3 →  a = 3 ∧ c = 2 := 
sorry 

theorem find_cos_diff (abc : triangle_ABC) :
  side_conditions abc → dot_product_condition abc → cosB_condition abc → b = 3 →  (real.cos (B - C)) = (23 / 27) := 
sorry

end find_a_c_find_cos_diff_l448_448388


namespace machine_no_repair_needed_l448_448495

theorem machine_no_repair_needed (M : ℕ) (σ : ℕ) (greatest_deviation : ℕ) 
                                  (nominal_weight : ℕ)
                                  (h1 : greatest_deviation = 39)
                                  (h2 : greatest_deviation ≤ (0.1 * nominal_weight))
                                  (h3 : ∀ d, d < 39) : 
                                  σ ≤ greatest_deviation :=
by
  sorry

end machine_no_repair_needed_l448_448495


namespace riverton_soccer_physics_l448_448996

theorem riverton_soccer_physics : 
  let total_players := 15
  let math_players := 9
  let both_subjects := 3
  let only_physics := total_players - math_players
  let physics_players := only_physics + both_subjects
  physics_players = 9 :=
by
  sorry

end riverton_soccer_physics_l448_448996


namespace households_same_number_l448_448978

theorem households_same_number (t : ℕ) (h₁ : 4.9 + 0.275 * t = 2.5 + 0.7 * t) : t = 6 :=
by
  sorry

end households_same_number_l448_448978


namespace intersection_distance_l448_448749

open Real

noncomputable def line_inclination (θ : ℝ) (P : ℝ × ℝ) : (ℝ → ℝ × ℝ) :=
  λ t, (P.1 - cos θ * t, P.2 + sin θ * t)

noncomputable def polar_to_cartesian (f : ℝ → ℝ) : (ℝ → ℝ) :=
  λ θ, (λ ρ, ρ * cos θ, λ ρ, rho * sin θ) (f θ)

theorem intersection_distance (t1 t2 : ℝ) (P A B : ℝ × ℝ) (f : ℝ → ℝ) :
  t1 + t2 = 2 * sqrt 2 ∧ t1 * t2 = 1 →
  |PA| + |PB| = 2 * sqrt 2 :=
sorry

end intersection_distance_l448_448749


namespace are_all_propositions_true_l448_448623

-- Condition ①: The converse of "If xy=1, then x and y are reciprocals of each other"
def converse_xy_reciprocal (x y : ℝ) : Prop :=
  (x * y = 1) ↔ (x = 1 / y ∧ y = 1 / x)

-- Condition ②: The negation of "Triangles with equal areas are congruent"
def negation_equal_area_congruent : Prop :=
  ¬∀ {A B : Type} [MetricSpace A] [MetricSpace B] (Δ₁ Δ₂ : Triangle A) (Δ₃ Δ₄ : Triangle B),
    (area Δ₁ = area Δ₂) → (is_congruent Δ₁ Δ₂)

-- Condition ③: The contrapositive of "If m ≤ 1, then the equation x² - 2x + m = 0 has real solutions"
def contrapositive_real_solutions (m : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 - 2*x + m = 0) → (m > 1)

-- Condition ④: The contrapositive of "If A ∩ B = B, then A ⊆ B"
def contrapositive_subset (A B : Set ℝ) : Prop :=
  ¬(A ⊆ B) → (A ∩ B ≠ B)

-- Translating the proof problem to whether all propositions are true
theorem are_all_propositions_true :
  (∀ (x y : ℝ), converse_xy_reciprocal x y) ∧
  negation_equal_area_congruent ∧
  (∀ m : ℝ, contrapositive_real_solutions m) ∧
  (∀ (A B : Set ℝ), contrapositive_subset A B) :=
by
  sorry

end are_all_propositions_true_l448_448623


namespace gillian_more_than_three_times_sandi_l448_448827

-- Definitions of the conditions
def sandi_initial : ℕ := 600
def sandi_spent : ℕ := sandi_initial / 2
def gillian_spent : ℕ := 1050
def three_times_sandi_spent : ℕ := 3 * sandi_spent

-- Theorem statement with the proof to be added
theorem gillian_more_than_three_times_sandi :
  gillian_spent - three_times_sandi_spent = 150 := 
sorry

end gillian_more_than_three_times_sandi_l448_448827


namespace z_squared_in_second_quadrant_l448_448847

-- Define the complex number z
def z : ℂ := complex.ofReal (real.cos (75 * real.pi / 180)) + complex.I * complex.ofReal (real.sin (75 * real.pi / 180))

-- Define the mathematical goal: proving z^2 is in the second quadrant
theorem z_squared_in_second_quadrant : 
  let z_squared := z * z in
  (real.angle.ofReal (real.cos (2 * (75 * real.pi / 180)))) = real.angle.pi - real.angle.zero_neg := sorry

end z_squared_in_second_quadrant_l448_448847


namespace constant_for_odd_m_l448_448655

theorem constant_for_odd_m (constant : ℝ) (f : ℕ → ℝ)
  (h1 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k + 1) → f m = constant * m)
  (h2 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k) → f m = (1/2 : ℝ) * m)
  (h3 : f 5 * f 6 = 15) : constant = 1 :=
by
  sorry

end constant_for_odd_m_l448_448655


namespace smallest_k_equiv_one_l448_448528

theorem smallest_k_equiv_one (k : ℤ) (h : k > 1) :
  k % 10 = 1 ∧ k % 15 = 1 ∧ k % 9 = 1 → k = 91 :=
begin
  intro h1,
  sorry
end

end smallest_k_equiv_one_l448_448528


namespace mixed_doubles_selection_l448_448201

-- Given conditions
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- The statement to show the number of different ways to select two players is 20
theorem mixed_doubles_selection : (num_male_players * num_female_players) = 20 := by
  -- Proof to be filled in
  sorry

end mixed_doubles_selection_l448_448201


namespace number_of_sheets_is_9_l448_448971

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end number_of_sheets_is_9_l448_448971


namespace max_value_in_sample_is_10_l448_448747

noncomputable def sample_size : ℕ := 5
noncomputable def sample_mean : ℝ := 7
noncomputable def sample_variance : ℝ := 4

theorem max_value_in_sample_is_10
  (samples : Fin sample_size → ℝ)
  (h1 : (∑ i, samples i) / sample_size = sample_mean)
  (h2 : (∑ i, (samples i - sample_mean) ^ 2) / sample_size = sample_variance)
  (h3 : ∀ i j, i ≠ j → samples i ≠ samples j) : 
  ∃ i, samples i = 10 :=
sorry

end max_value_in_sample_is_10_l448_448747


namespace area_of_quadrilateral_is_correct_l448_448176

noncomputable def area_of_quadrilateral_BGFAC : ℝ :=
  let a := 3 -- side of the equilateral triangle
  let triangle_area := (a^2 * Real.sqrt 3) / 4 -- area of ABC
  let ratio_AG_GC := 2 -- ratio AG:GC = 2:1
  let area_AGC := triangle_area / 3 -- area of triangle AGC
  let area_BGC := triangle_area / 3 -- area of triangle BGC
  let area_BFC := (2 : ℝ) * triangle_area / 3 -- area of triangle BFC
  let area_BGFC := area_BGC + area_BFC -- area of quadrilateral BGFC
  area_BGFC

theorem area_of_quadrilateral_is_correct :
  area_of_quadrilateral_BGFAC = (3 * Real.sqrt 3) / 2 :=
by
  -- Proof will be provided here
  sorry

end area_of_quadrilateral_is_correct_l448_448176


namespace polynomial_quadratic_trinomial_l448_448345

theorem polynomial_quadratic_trinomial (b a : ℤ) 
    (h1 : (3 - b) = 0) 
    (h2 : a = 2) : 
    a^2 - b^2 = -5 :=
by
  have hb : b = 3 := by linarith,
  rw [hb, h2],
  calc 
    2^2 - 3^2 = 4 - 9 := by norm_num
    ... = -5 := by norm_num

end polynomial_quadratic_trinomial_l448_448345


namespace hypotenuse_of_isosceles_right_triangle_area_of_isosceles_right_triangle_l448_448519

-- Definitions for the conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = real.sqrt (a^2 + b^2)

def right_angle (angle : ℝ) : Prop :=
  angle = 90

-- Given conditions
def DE : ℝ := 8
def ED : ℝ := 8
def angle_D : ℝ := 90

-- The hypotenuse DF and area of triangle DEF
def DF : ℝ := 8 * real.sqrt 2
def area_DEF : ℝ := 32

-- Lean theorem statements
theorem hypotenuse_of_isosceles_right_triangle :
  is_isosceles_right_triangle DE ED DF :=
by
  sorry

theorem area_of_isosceles_right_triangle :
  area_DEF = (1 / 2) * DE * ED :=
by
  sorry

end hypotenuse_of_isosceles_right_triangle_area_of_isosceles_right_triangle_l448_448519


namespace opposite_number_113_is_13_l448_448035

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l448_448035


namespace gallons_in_pond_after_50_days_l448_448947

def initial_amount : ℕ := 500
def evaporation_rate : ℕ := 1
def days_passed : ℕ := 50
def total_evaporation : ℕ := days_passed * evaporation_rate
def final_amount : ℕ := initial_amount - total_evaporation

theorem gallons_in_pond_after_50_days : final_amount = 450 := by
  sorry

end gallons_in_pond_after_50_days_l448_448947


namespace sum_of_cubes_l448_448012

theorem sum_of_cubes {x y : ℝ} (h₁ : x + y = 0) (h₂ : x * y = -1) : x^3 + y^3 = 0 :=
by
  sorry

end sum_of_cubes_l448_448012


namespace inequality_holds_l448_448582

theorem inequality_holds (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end inequality_holds_l448_448582


namespace checkerboard_5_black_squares_l448_448069

def is_black (row col : ℕ) : Prop :=
  (row % 2 = col % 2)

def valid_subsquares_count (n : ℕ) : ℕ :=
  let mutable count := 0
  for size in [3:n+1] do
    for i in [0:n-size+1] do
      for j in [0:n-size+1] do
        let num_blacks := (size * size + (is_black i j).toNat) / 2
        if num_blacks ≥ 5 then
          count := count + 1
  count

theorem checkerboard_5_black_squares :
  valid_subsquares_count 10 = 172 :=
by
  sorry

end checkerboard_5_black_squares_l448_448069


namespace two_quantities_change_as_P_moves_l448_448819

noncomputable def number_of_changing_quantities (P : Point) (l : Line) (CD DE EC : ℝ) (Q : Point) 
  (h1 : P ∈ l) (h2 : l ∥ line_through C D) (h3 : midpoint Q C D)
  (h4 : on_segment C D Q) : ℕ :=
  let PQ_changes := True
  let perimeter_CDE_changes := False
  let area_CDE_changes := False
  let area_trapezoid_CDPQ_changes := True
  if PQ_changes then 
    if area_trapezoid_CDPQ_changes then 
      2 
    else 
      1
  else 
    if perimeter_CDE_changes then 
      if area_trapezoid_CDPQ_changes then 
        2 
      else 
        1
    else 
      if area_CDE_changes then 
        if area_trapezoid_CDPQ_changes then 
          2 
        else 
          0

theorem two_quantities_change_as_P_moves (P : Point) (l : Line) (C D E : Point) 
  (h1 : P ∈ l) (h2 : l ∥ line_through C D) 
  (Q : Point) (h3 : Q = midpoint C D) 
  : number_of_changing_quantities P l (dist C D) (dist D E) (dist E C) Q h1 h2 h3 (sorry) = 2 :=
sorry

end two_quantities_change_as_P_moves_l448_448819


namespace geometric_sequence_sum_l448_448396

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a n = a 1 * q ^ n

theorem geometric_sequence_sum (h : geometric_sequence a 2) (h_sum : a 1 + a 2 = 3) :
  a 4 + a 5 = 24 := by
  sorry

end geometric_sequence_sum_l448_448396


namespace tickets_needed_for_equal_distribution_l448_448997

theorem tickets_needed_for_equal_distribution :
  ∃ k : ℕ, 865 + k ≡ 0 [MOD 9] ∧ k = 8 := sorry

end tickets_needed_for_equal_distribution_l448_448997


namespace sufficient_and_necessary_condition_l448_448583

theorem sufficient_and_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by 
  sorry

end sufficient_and_necessary_condition_l448_448583


namespace geom_seq_sum_157_l448_448470

def is_geom_seq (l : List ℕ) : Prop :=
∃ q : ℕ, ∀ i j : ℕ, i < j ∧ j < l.length → l.get i * q = l.get j

def sum_is_157 (l : List ℕ) : Prop :=
 l.sum = 157

theorem geom_seq_sum_157 :
 ∀ (l : List ℕ), is_geom_seq l ∧ sum_is_157 l →
 l = [157] ∨ l = [1, 156] ∨ l = [1, 12, 144] :=
by sorry

end geom_seq_sum_157_l448_448470


namespace number_of_sheets_is_9_l448_448972

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end number_of_sheets_is_9_l448_448972


namespace inv_function_composition_l448_448870

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 5
  | 3 => 3
  | 4 => 2
  | 5 => 1
  | _ => 6

noncomputable def f_inv (y : ℕ) : ℕ :=
  if h : ∃ x, f x = y then classical.some h else 0

theorem inv_function_composition (h : ∀ y, ∃ x, f x = y) : 
  f_inv (f_inv (f_inv 2)) = 5 :=
by
  sorry

end inv_function_composition_l448_448870


namespace employee_total_correct_l448_448400

variable (total_employees : ℝ)
variable (percentage_female : ℝ)
variable (percentage_male_literate : ℝ)
variable (percentage_total_literate : ℝ)
variable (number_female_literate : ℝ)
variable (percentage_male : ℝ := 1 - percentage_female)

variables (E : ℝ) (CF : ℝ) (M : ℝ) (total_literate : ℝ)

theorem employee_total_correct :
  percentage_female = 0.60 ∧
  percentage_male_literate = 0.50 ∧
  percentage_total_literate = 0.62 ∧
  number_female_literate = 546 ∧
  (total_employees = 1300) :=
by
  -- Change these variables according to the context or find a way to prove this
  let total_employees := 1300
  have Cf := number_female_literate / (percentage_female * total_employees)
  have total_male := percentage_male * total_employees
  have male_literate := percentage_male_literate * total_male
  have total_literate := percentage_total_literate * total_employees

  -- We replace "proof statements" with sorry here
  sorry

end employee_total_correct_l448_448400


namespace analysis_method_seeks_sufficient_conditions_l448_448251

def analysis_method_conditions (conclusion : Prop) (C : Prop → Prop) : Prop :=
  ∃ (P : Prop), C P ∧ (P → conclusion)

theorem analysis_method_seeks_sufficient_conditions (conclusion : Prop) (C : Prop → Prop) :
  analysis_method_conditions conclusion C ↔ (∃ P : Prop, (P → conclusion) :=
by
  sorry

end analysis_method_seeks_sufficient_conditions_l448_448251


namespace largest_fraction_addition_l448_448564

-- Definitions for the problem conditions
def proper_fraction (a b : ℕ) : Prop :=
  a < b

def denom_less_than (d : ℕ) (bound : ℕ) : Prop :=
  d < bound

-- Main statement of the problem
theorem largest_fraction_addition :
  ∃ (a b : ℕ), (b > 0) ∧ proper_fraction (b + 7 * a) (7 * b) ∧ denom_less_than b 5 ∧ (a / b : ℚ) <= 3/4 := 
sorry

end largest_fraction_addition_l448_448564


namespace initial_toys_count_l448_448444

theorem initial_toys_count (T : ℕ) (h : 10 * T + 300 = 580) : T = 28 :=
by
  sorry

end initial_toys_count_l448_448444


namespace smallest_x_squared_l448_448776

-- Define the existence of the isosceles trapezoid ABCD with given sides
variable (A B C D O : ℝ)
variable (AB CD AD BC AO OB OD OC : ℝ)

-- Define the conditions
variable (is_isosceles_trapezoid : AB = 70 ∧ CD = 25 ∧ AD = BC)
variable (circle_conditions : AO = 10 ∧ OB = 60 ∧ OD = 12.5 ∧ OD = OC ∧ O ∈ (line_span ℝ (set.range (λ (t : ℝ), A + t • (B - A)))))

-- Define the smallest possible value of x and its square
def n_squared_value (x : ℝ) : ℝ := x = 16 ∧ n = 16 ∧ n^2 = 256

-- Mathematical goal
theorem smallest_x_squared : is_isosceles_trapezoid ∧ circle_conditions → ∃ (n : ℝ), n_squared_value n :=
by
  sorry

end smallest_x_squared_l448_448776


namespace rectangle_area_3650_l448_448381

variables (L B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := L - B = 23
def condition2 : Prop := 2 * (L + B) = 246

-- Prove that the area of the rectangle is 3650 m² given the conditions
theorem rectangle_area_3650 (h1 : condition1 L B) (h2 : condition2 L B) : L * B = 3650 := by
  sorry

end rectangle_area_3650_l448_448381


namespace number_of_true_propositions_is_zero_l448_448245

-- We first assume the propositions as given conditions (though they will be used in contradiction).

-- Proposition ①: If \( \frac{1}{a} < \frac{1}{b} \), then \( a > b \)
def proposition1 (a b : ℝ) : Prop := (1 / a < 1 / b) → (a > b)

-- Proposition ②: If \( a > b > c \), then \( a |c| > b |c| \)
def proposition2 (a b c : ℝ) : Prop := (a > b ∧ b > c) → (a * abs c > b * abs c)

-- Proposition ③: The function \( f(x) = x + \frac{1}{x} \) has a minimum value of 2
def proposition3 : Prop := (∀ x : ℝ, x > 0 → x + 1 / x ≥ 2)

-- The main theorem to prove that the number of true propositions is 0
theorem number_of_true_propositions_is_zero :
  (¬ proposition1 (-2) 1) ∧ (¬ proposition2 3 2 0) ∧ (¬ proposition3) →
  (0) = 0 :=
by
  intros h,
  exact Nat.zero_eq_zero

end number_of_true_propositions_is_zero_l448_448245


namespace train_crossing_time_l448_448204

noncomputable def length_of_train : ℝ := 120 -- meters
noncomputable def speed_of_train_kmh : ℝ := 27 -- kilometers per hour
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmh * (1000 / 3600) -- converted to meters per second
noncomputable def time_to_cross : ℝ := length_of_train / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross = 16 :=
by
  -- proof goes here
  sorry

end train_crossing_time_l448_448204


namespace remainder_theorem_solution_l448_448909

noncomputable def c_value : ℚ := -47 / 15
noncomputable def d_value : ℚ := 428 / 15

def g (c d : ℚ) (x : ℚ) : ℚ := c * x^3 - 8 * x^2 + d * x - 7

theorem remainder_theorem_solution :
  let c := c_value in
  let d := d_value in
  g c d 2 = -7 ∧ g c d (-3) = -80 :=
by
  sorry

end remainder_theorem_solution_l448_448909


namespace determine_monotonically_increasing_interval_l448_448224

noncomputable theory
open Real

/-- Problem Statement: Given the function f(x) = 7 * sin(x - π / 6), prove that it is monotonically increasing in the interval (0, π/2). -/
def monotonically_increasing_interval (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2

def function_f (x : ℝ) : ℝ :=
  7 * sin (x - π / 6)

theorem determine_monotonically_increasing_interval :
  ∀ x : ℝ, monotonically_increasing_interval x → monotone_on function_f (set.Ioo 0 (π / 2)) :=
sorry

end determine_monotonically_increasing_interval_l448_448224


namespace perimeter_third_square_l448_448861

def perimeter_of_square (side_length : ℝ) : ℝ := 4 * side_length
def area_of_square (side_length : ℝ) : ℝ := side_length * side_length

theorem perimeter_third_square 
  (P1 P2 : ℝ) (P1_eq : P1 = 60) (P2_eq : P2 = 48) : 
  let s1 := P1 / 4 
  let s2 := P2 / 4 
  let A1 := area_of_square s1
  let A2 := area_of_square s2
  let A3 := A1 - A2
  let s3 := real.sqrt A3
  let P3 := perimeter_of_square s3
  in P3 = 36 := 
by
  have h1 : s1 = 60 / 4 := by rw [P1_eq]; simp
  have h2 : s2 = 48 / 4 := by rw [P2_eq]; simp
  have hA1 : A1 = area_of_square s1 := rfl
  have hA2 : A2 = area_of_square s2 := rfl
  have hdiff : A3 = area_of_square s1 - area_of_square s2 := rfl
  have hs3 : s3 = real.sqrt (area_of_square s1 - area_of_square s2) := rfl
  have hP3 : P3 = perimeter_of_square s3 := rfl
  simp [h1, h2, hA1, hA2, hdiff, hs3, hP3],
  sorry

end perimeter_third_square_l448_448861


namespace yellow_tint_percent_l448_448948

theorem yellow_tint_percent (total_volume: ℕ) (initial_yellow_percent: ℚ) (yellow_added: ℕ) (answer: ℚ) 
  (h_initial_total: total_volume = 20) 
  (h_initial_yellow: initial_yellow_percent = 0.50) 
  (h_yellow_added: yellow_added = 6) 
  (h_answer: answer = 61.5): 
  (yellow_added + initial_yellow_percent * total_volume) / (total_volume + yellow_added) * 100 = answer := 
by 
  sorry

end yellow_tint_percent_l448_448948


namespace sin_alpha_beta_l448_448696

theorem sin_alpha_beta (a b c α β : Real) (h₁ : a * Real.cos α + b * Real.sin α + c = 0)
  (h₂ : a * Real.cos β + b * Real.sin β + c = 0) (h₃ : 0 < α) (h₄ : α < β) (h₅ : β < π) :
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) :=
by 
  sorry

end sin_alpha_beta_l448_448696


namespace last_digit_sum_of_squares_l448_448646

theorem last_digit_sum_of_squares :
  (∑ i in Finset.range 100, i ^ 2) % 10 = 5 :=
by sorry

end last_digit_sum_of_squares_l448_448646


namespace smallest_sum_of_xy_l448_448330

namespace MathProof

theorem smallest_sum_of_xy (x y : ℕ) (hx : x ≠ y) (hxy : (1 : ℝ) / x + (1 : ℝ) / y = 1 / 12) : x + y = 49 :=
sorry

end MathProof

end smallest_sum_of_xy_l448_448330


namespace positive_y_percentage_l448_448198

theorem positive_y_percentage (y : ℝ) (hy_pos : 0 < y) (h : 0.01 * y * y = 9) : y = 30 := by
  sorry

end positive_y_percentage_l448_448198


namespace machine_does_not_require_repair_l448_448499

noncomputable def nominal_mass := 390 -- The nominal mass M is 390 grams

def greatest_deviation_preserved := 39 -- The greatest deviation among preserved measurements is 39 grams

def deviation_unread_measurements (x : ℕ) : Prop := x < 39 -- Deviations of unread measurements are less than 39 grams

def all_deviations_no_more_than := ∀ (x : ℕ), x ≤ 39 -- All deviations are no more than 39 grams

theorem machine_does_not_require_repair 
  (M : ℕ) 
  (h_nominal_mass : M = nominal_mass)
  (h_greatest_deviation : greatest_deviation_preserved ≤ 0.1 * M)
  (h_unread_deviations : ∀ (x : ℕ), deviation_unread_measurements x) 
  (h_all_deviations : all_deviations_no_more_than):
  true := -- Prove the machine does not require repair
sorry

end machine_does_not_require_repair_l448_448499


namespace no_four_points_all_odd_distances_l448_448823

theorem no_four_points_all_odd_distances :
  ¬ ∃ (A B C D : ℝ × ℝ), 
    (∃ a : ℕ, (a % 2 = 1) ∧ (dist A B = a)) ∧ 
    (∃ b : ℕ, (b % 2 = 1) ∧ (dist A C = b)) ∧ 
    (∃ c : ℕ, (c % 2 = 1) ∧ (dist A D = c)) ∧ 
    (∃ d : ℕ, (d % 2 = 1) ∧ (dist B C = d)) ∧ 
    (∃ e : ℕ, (e % 2 = 1) ∧ (dist B D = e)) ∧ 
    (∃ f : ℕ, (f % 2 = 1) ∧ (dist C D = f)) :=
sorry

end no_four_points_all_odd_distances_l448_448823


namespace find_7c_plus_7d_l448_448628

noncomputable def f (c d x : ℝ) : ℝ := c * x + d
noncomputable def h (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 1

theorem find_7c_plus_7d (c d : ℝ) (h_def : ∀ x, h x = f_inv x - 5) (f_def : ∀ x, f c d x = c * x + d) (f_inv_def : ∀ x, f_inv x = 7 * x - 1) : 7 * c + 7 * d = 2 := by
  sorry

end find_7c_plus_7d_l448_448628


namespace cars_meet_after_12_hours_l448_448486

theorem cars_meet_after_12_hours :
  ∀ (d : ℕ) (v1 v2 : ℕ), d = 450 → v1 = 45 → v2 = 30 → (2 * d) / (v1 + v2) = 12 :=
by
  intros d v1 v2 hd hv1 hv2
  rw [hd, hv1, hv2]
  norm_num
  sorry

end cars_meet_after_12_hours_l448_448486


namespace pet_snake_cost_l448_448309

theorem pet_snake_cost (original_amount left_amount snake_cost : ℕ) 
  (h1 : original_amount = 73) 
  (h2 : left_amount = 18)
  (h3 : snake_cost = original_amount - left_amount) : 
  snake_cost = 55 := 
by 
  sorry

end pet_snake_cost_l448_448309


namespace determine_monotonically_increasing_interval_l448_448220

noncomputable theory
open Real

/-- Problem Statement: Given the function f(x) = 7 * sin(x - π / 6), prove that it is monotonically increasing in the interval (0, π/2). -/
def monotonically_increasing_interval (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2

def function_f (x : ℝ) : ℝ :=
  7 * sin (x - π / 6)

theorem determine_monotonically_increasing_interval :
  ∀ x : ℝ, monotonically_increasing_interval x → monotone_on function_f (set.Ioo 0 (π / 2)) :=
sorry

end determine_monotonically_increasing_interval_l448_448220


namespace comparison_of_a_b_c_l448_448427

noncomputable def a : ℝ := 0.3 ^ 0.6
noncomputable def b : ℝ := 0.5 ^ 0.6
noncomputable def c : ℝ := Real.logBase (Real.pi / 3) (Real.pi / 4)

theorem comparison_of_a_b_c : b > a ∧ a > c := by
  sorry

end comparison_of_a_b_c_l448_448427


namespace num_incorrect_statements_l448_448244

theorem num_incorrect_statements : 
  let S1 := (0 ∈ ({0} : Set ℕ))
  let S2 := ({0} ⊇ ∅)
  let S3 := (0.3 ∉ ℚ)
  let S4 := (0 ∈ ℕ)
  let S5 := ({x : ℤ | x^2 - 2 = 0} = ∅)
  ¬S1 ∧ ¬S2 ∧ ¬S3 ∧ ¬S4 ∧ ¬S5 → false :=
by sorry

end num_incorrect_statements_l448_448244


namespace friend_reading_time_faster_l448_448802

noncomputable def reading_time_friend (my_time : ℕ) (speed_ratio : ℕ) : ℕ :=
  my_time / speed_ratio

theorem friend_reading_time_faster 
  (my_time : ℕ := 180)
  (speed_ratio : ℕ := 5) :
  reading_time_friend my_time speed_ratio = 36 :=
by 
  -- Given conditions
  have h_my_time: my_time = 180 := by rfl
  have h_speed_ratio: speed_ratio = 5 := by rfl
  -- Calculate friend's reading time
  unfold reading_time_friend
  simp only [Nat.div_eq_of_eq_mul, Nat.mul_comm, Nat.succ_eq_add_one]
  sorry  -- Proof is skipped

end friend_reading_time_faster_l448_448802


namespace solution_set_l448_448429

noncomputable theory

-- Defining the function f as an odd function and the given conditions:
def isOdd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

axiom f : ℝ → ℝ
axiom f_odd : isOdd f
axiom f_mul : ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → f (x1 * x2) = f x1 + f x2
axiom f_negative : ∀ x : ℝ, (1 < x) → f x < 0
axiom f_quarter : f (1/4) = 2

-- Prove the solution set of the inequality log (f (x) + 2) < 0
theorem solution_set : { x : ℝ | Real.log (f x + 2) < 0 } = { x : ℝ | (2 < x ∧ x < 4) ∨ (-1/2 < x ∧ x < -1/4) } := sorry

end solution_set_l448_448429


namespace volume_ratio_l448_448905

noncomputable def cube_volume (a : ℝ) : ℝ := a^3

noncomputable def ratio_of_volumes (a b : ℝ) : ℝ :=
  cube_volume a / cube_volume b

theorem volume_ratio :
  ratio_of_volumes 8 (20 / 2.54) ≈ 1.046 :=
by
  have h1 : 20 / 2.54 ≈ 7.874 := sorry
  have h2 : cube_volume 8 = 512 := by
    simp [cube_volume]
  have h3 : cube_volume (20 / 2.54) ≈ 489.303 := by
    simp [cube_volume, h1]
  have h4 : ratio_of_volumes 8 (20 / 2.54) ≈ 512 / 489.303 := by
    simp [ratio_of_volumes, h2, h3]
  exact sorry

end volume_ratio_l448_448905


namespace min_value_of_f_l448_448287

def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

theorem min_value_of_f : 
  (0.4 ≤ x ∧ x ≤ 0.6) → 
  (0.3 ≤ y ∧ y ≤ 0.4) → 
  (∃ x y, f x y = 2 / 5) :=
by
  sorry

end min_value_of_f_l448_448287


namespace total_peanuts_in_box_l448_448734

def initial_peanuts := 4
def peanuts_taken_out := 3
def peanuts_added := 12

theorem total_peanuts_in_box : initial_peanuts - peanuts_taken_out + peanuts_added = 13 :=
by
sorry

end total_peanuts_in_box_l448_448734


namespace luke_payment_difference_l448_448013

noncomputable def plan1_total_payment (P : ℝ) (r : ℝ) (n : ℕ) (t1 t2 : ℕ) : ℝ :=
let A1 := P * (1 + r / n)^(n * t1) in
let half1 := A1 / 2 in
let half2 := half1 * (1 + r / n)^(n * t2) in
half1 + half2

noncomputable def plan2_total_payment (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
P * (1 + r) ^ t

noncomputable def payment_difference (P : ℝ) (r : ℝ) (n : ℕ) (t1 t2 : ℕ) : ℝ :=
(plan2_total_payment P r (t1 + t2)) - (plan1_total_payment P r n t1 t2)

theorem luke_payment_difference :
  payment_difference 20000 0.12 2 5 5 ≈ 11926 :=
sorry

end luke_payment_difference_l448_448013


namespace convex_polyhedron_divisible_l448_448461

-- Axioms defining the geometry and symmetry conditions
axiom convex_polyhedron (P : Type) : Prop
axiom centrally_symmetric_face (F : Type) : Prop
axiom has_face (P : Type) (F : Type) : Prop

-- Main theorem statement
theorem convex_polyhedron_divisible (P : Type) 
  [H1 : convex_polyhedron P] 
  [H2 : ∀ F, has_face P F → centrally_symmetric_face F] : 
  ∃ (Q : Type), Q = parallelepiped ∧ subdivides P Q :=
sorry

end convex_polyhedron_divisible_l448_448461


namespace infinitely_many_H_points_l448_448073

-- Define the curve C as (x^2 / 4) + y^2 = 1
def is_on_curve (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

-- Define point P on curve C
def is_H_point (P : ℝ × ℝ) : Prop :=
  is_on_curve P.1 P.2 ∧
  ∃ (A B : ℝ × ℝ), is_on_curve A.1 A.2 ∧ B.1 = 4 ∧
  (dist (P.1, P.2) (A.1, A.2) = dist (P.1, P.2) (B.1, B.2) ∨
   dist (P.1, P.2) (A.1, A.2) = dist (A.1, A.2) (B.1, B.2))

-- Theorem to prove the existence of infinitely many H points
theorem infinitely_many_H_points : ∃ (P : ℝ × ℝ), is_H_point P ∧ ∀ (Q : ℝ × ℝ), Q ≠ P → is_H_point Q :=
sorry


end infinitely_many_H_points_l448_448073


namespace intimate_interval_proof_l448_448428

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 3

-- Define the concept of intimate functions over an interval
def are_intimate_functions (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Prove that the interval [2, 3] is a subset of [a, b]
theorem intimate_interval_proof (a b : ℝ) (h : are_intimate_functions a b) :
  2 ≤ b ∧ a ≤ 3 :=
sorry

end intimate_interval_proof_l448_448428


namespace usual_time_to_catch_bus_l448_448525

theorem usual_time_to_catch_bus 
  (S T : ℝ) 
  (h1 : (4/5 : ℝ) * S)
  (h2 : (5/4 : ℝ) * T = T + 5) :
  T = 20 := 
by
  sorry

end usual_time_to_catch_bus_l448_448525


namespace geometry_problem_l448_448931

variables {A B C D E M : Point}
variables {circle : Circle}
variables (h1 : tangent A B circle) (h2 : tangent A C circle)
variables (h3 : secant D E circle)
variables (h4 : M = midpoint B C)

theorem geometry_problem
  (h_tangent1 : tangent A B circle)
  (h_tangent2 : tangent A C circle)
  (h_secant : secant D E circle)
  (h_midpoint : M = midpoint B C) :
  (BM^2 = DM * ME) ∧ (angle DME = 2 * angle DBE) :=
sorry

end geometry_problem_l448_448931


namespace last_two_digits_of_7_pow_2017_l448_448449

theorem last_two_digits_of_7_pow_2017 :
  let last_two_digits (n : ℕ) := n % 100,
      pattern := ∀ k : ℕ, last_two_digits (7^(4*k-2)) = 49 ∧
                         last_two_digits (7^(4*k-1)) = 43  ∧
                         last_two_digits (7^(4*k)) = 1  ∧ 
                         last_two_digits (7^(4*k + 1)) = 7
  in pattern →
     last_two_digits (7^2017) = 7 :=
by
  intros
  sorry

end last_two_digits_of_7_pow_2017_l448_448449


namespace projectiles_meet_in_84_minutes_l448_448139

def distance : ℝ := 1386 -- distance in km
def speed1 : ℝ := 445 -- speed of first projectile in km/h
def speed2 : ℝ := 545 -- speed of second projectile in km/h
def combined_speed : ℝ := speed1 + speed2

-- Define the time it takes for the projectiles to meet in hours
def time_in_hours : ℝ := distance / combined_speed

-- Define the time in minutes
def time_in_minutes : ℝ := time_in_hours * 60

theorem projectiles_meet_in_84_minutes :
  time_in_minutes = 84 := by
  sorry

end projectiles_meet_in_84_minutes_l448_448139


namespace average_speed_is_approximately_36_11_l448_448588

def average_speed (x : ℝ) : ℝ :=
  let distance1 := x
  let speed1 := 50
  let time1 := distance1 / speed1

  let distance2 := 2 * x
  let speed2 := 30
  let time2 := distance2 / speed2

  let distance3 := 0.5 * x
  let speed3 := 35
  let time3 := distance3 / speed3
  
  let distance4 := 1.5 * x
  let speed4 := 40
  let time4 := distance4 / speed4

  let total_distance := distance1 + distance2 + distance3 + distance4
  let total_time := time1 + time2 + time3 + time4
  
  total_distance / total_time

theorem average_speed_is_approximately_36_11 (x : ℝ) : average_speed x ≈ 36.11 :=
by sorry

end average_speed_is_approximately_36_11_l448_448588


namespace inscribed_square_product_l448_448580

theorem inscribed_square_product (a b : ℝ)
  (h1 : a + b = 2 * Real.sqrt 5)
  (h2 : Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2) :
  a * b = -6 := 
by
  sorry

end inscribed_square_product_l448_448580


namespace ice_cream_sales_l448_448809

theorem ice_cream_sales : 
  let tuesday_sales := 12000
  let wednesday_sales := 2 * tuesday_sales
  let total_sales := tuesday_sales + wednesday_sales
  total_sales = 36000 := 
by 
  sorry

end ice_cream_sales_l448_448809


namespace tangent_to_circle_intersecting_orthogonal_vectors_l448_448672

noncomputable theory

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the line l
def line_l (m x y : ℝ) : Prop := x + m * y = 3

-- Statement 1: If l is tangent to C, then m = 7/24
theorem tangent_to_circle (m : ℝ) :
  (∃ x y : ℝ, circle_C x y ∧ line_l m x y ∧ (∀ x' y': ℝ, (circle_C x' y' ∧ line_l m x' y') → (x' = x ∧ y' = y))) ↔
  m = 7/24 :=
sorry

-- Statement 2: There exists an m such that l intersects C at points
-- A and B, and ⟨OA, OB⟩ = 0, and the valid m are 9 ± 2 * sqrt(14).
theorem intersecting_orthogonal_vectors (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    (circle_C x1 y1 ∧ line_l m x1 y1 ∧ circle_C x2 y2 ∧ line_l m x2 y2) ∧
    (x1 * x2 + y1 * y2 = 0)) ↔
  m = 9 + 2 * real.sqrt 14 ∨ m = 9 - 2 * real.sqrt 14 :=
sorry

end tangent_to_circle_intersecting_orthogonal_vectors_l448_448672


namespace max_k_value_l448_448324

def A : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def B (i : ℕ) := {b : Finset ℕ // b ⊆ A ∧ b ≠ ∅ ∧ ∀ j ≠ i, ∃ k : Finset ℕ, k ⊆ A ∧ k ≠ ∅ ∧ (b ∩ k).card ≤ 2}

theorem max_k_value : ∃ k, k = 175 :=
  by
    sorry

end max_k_value_l448_448324


namespace alice_pints_wednesday_l448_448808

-- Initial conditions
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := pints_monday / 3
def total_pints_before_return : ℕ := pints_sunday + pints_monday + pints_tuesday
def pints_returned_wednesday : ℕ := pints_tuesday / 2
def pints_wednesday : ℕ := total_pints_before_return - pints_returned_wednesday

-- The proof statement
theorem alice_pints_wednesday : pints_wednesday = 18 :=
by
  sorry

end alice_pints_wednesday_l448_448808


namespace circumcircle_locus_l448_448940

open EuclideanGeometry

variables {R r d : ℝ} {O I : Point}

-- Conditions
axiom cond1 : 0 < r ∧ r < R
axiom cond2 : d = dist O I
axiom cond3 : ∀ (A B : Point), is_chord O A B ∧ is_tangent I A B (chord_tangent_point A B I) 

-- Prove the locus of the centers of circumcircles
theorem circumcircle_locus :
  ∃ (M : Point), ∀ (A B : Point), is_chord O A B ∧ is_tangent I A B (chord_tangent_point A B I) 
    → dist O M = (R^2 - d^2) / (2 * r) :=
by sorry

end circumcircle_locus_l448_448940


namespace quadratic_distinct_real_roots_l448_448676

theorem quadratic_distinct_real_roots (a : ℝ) :
  (ax^2 - 4x - 1 = 0) ∧ a ≠ 0 ↔ (a > -4) ∧ (a ≠ 0) := by
  sorry

end quadratic_distinct_real_roots_l448_448676


namespace palindromic_primes_sum_55_l448_448816

-- Define a function to check primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to reverse the digits of a two-digit number
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10 in
  let units := n % 10 in
  units * 10 + tens

-- Define the predicate for a palindromic prime
def is_palindromic_prime (n : ℕ) : Prop :=
  n < 70 ∧ 10 ≤ n ∧ n < 100 ∧ is_prime n ∧ is_prime (reverse_digits n)

-- Define the list of all two-digit numbers
def two_digit_numbers : List ℕ :=
  List.range' 10 90

-- Construct the list of palindromic primes less than 70
def palindromic_primes : List ℕ :=
  (two_digit_numbers.filter is_palindromic_prime)

-- Define the sum of palindromic primes
def sum_palindromic_primes : ℕ :=
  palindromic_primes.sum

-- State the theorem
theorem palindromic_primes_sum_55 : sum_palindromic_primes = 55 := by
  sorry

end palindromic_primes_sum_55_l448_448816


namespace find_initial_investment_l448_448765

variable (P : ℝ)
variable (after_one_year_growth : ℝ := 1.15 * P)
variable (add_28_to_portfolio : ℝ := after_one_year_growth + 28)
variable (after_two_years_growth : ℝ := add_28_to_portfolio * 1.10)
variable (final_value : ℝ := 132)

theorem find_initial_investment :
  final_value = after_two_years_growth → P ≈ 80 :=
by
  sorry

end find_initial_investment_l448_448765


namespace unique_painted_cube_l448_448561

-- Define the problem conditions
def is_painted (cube : Type) : Prop :=
  -- Conditions for the cube painting
  (∃ (blue_face : cube), 
    ∃ (red_faces : Finset cube) (green_faces : Finset cube), 
    red_faces.card = 3 ∧ green_faces.card = 2 ∧
    (∀ v : cube, ∃ (r : red_faces), cube.adjacent v r) ∧
    (∀ f : Finset cube, (f.card = 6) → (f = Finset.insert blue_face (Finset.union red_faces green_faces))
  )

-- Define the main theorem to be proved
theorem unique_painted_cube : ∃! (c : Type), is_painted c :=
by sorry

end unique_painted_cube_l448_448561


namespace opposite_113_eq_114_l448_448063

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l448_448063


namespace opposite_113_eq_114_l448_448065

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l448_448065


namespace count_p_shape_points_l448_448917

-- Define the problem conditions
def side_length : ℕ := 10
def point_interval : ℕ := 1
def num_sides : ℕ := 3
def correction_corners : ℕ := 2

-- Define the total expected points
def total_expected_points : ℕ := 31

-- Proof statement
theorem count_p_shape_points :
  ((side_length / point_interval + 1) * num_sides - correction_corners) = total_expected_points := by
  sorry

end count_p_shape_points_l448_448917


namespace monotonic_increasing_l448_448234

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π / 6)

theorem monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 2) → f x1 < f x2 := sorry

end monotonic_increasing_l448_448234


namespace route_y_slower_by_2_4_minutes_l448_448800
noncomputable def time_route_x : ℝ := (7 : ℝ) / (35 : ℝ)
noncomputable def time_downtown_y : ℝ := (1 : ℝ) / (10 : ℝ)
noncomputable def time_other_y : ℝ := (7 : ℝ) / (50 : ℝ)
noncomputable def time_route_y : ℝ := time_downtown_y + time_other_y

theorem route_y_slower_by_2_4_minutes :
  ((time_route_y - time_route_x) * 60) = 2.4 :=
by
  -- Provide the required proof here
  sorry

end route_y_slower_by_2_4_minutes_l448_448800


namespace quaternary_to_decimal_l448_448264

-- Define the quaternary number 30012_{(4)}
def quaternary_number : List ℕ := [3, 0, 0, 1, 2]

-- Define function to convert quaternary (base 4) to decimal
def convert_quaternary_to_decimal (num : List ℕ) : ℕ :=
  num.reverse.enum.map (fun ⟨i, d⟩ => d * 4^i).sum

-- State the theorem to prove
theorem quaternary_to_decimal : convert_quaternary_to_decimal quaternary_number = 774 :=
by
  -- Skipping the proof
  sorry

end quaternary_to_decimal_l448_448264


namespace zeros_sum_eq_3pi_div_4_l448_448439

noncomputable theory
open Real

def f (x : ℝ) := sin (4 * x + π / 4)

theorem zeros_sum_eq_3pi_div_4 
  (a : ℝ) 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x1) (h2 : x1 < x2) (h3 : x2 < x3) (h4 : x3 ≤ 9 * π / 16) 
  (h5 : f x1 + a = 0) (h6 : f x2 + a = 0) (h7 : f x3 + a = 0) 
  (h_zeros : f x = 0 → x ∈ {x1, x2, x3}) 
  : x1 + 2 * x2 + x3 = 3 * π / 4 :=
sorry

end zeros_sum_eq_3pi_div_4_l448_448439


namespace number_of_boys_l448_448744

theorem number_of_boys (girls boys : ℕ) (h1 : girls = 697) (h2 : girls - boys = 228) : boys = 469 :=
by {
  rw [h1] at h2,         -- Replace girls with 697 in the second hypothesis
  linarith,              -- Solve the resulting linear equation
}

end number_of_boys_l448_448744


namespace general_equation_of_curve_point_on_ellipse_given_conditions_l448_448404

noncomputable def curve_parametric_equation (θ : ℝ) : ℝ × ℝ := (sqrt 3 * Real.cos θ, Real.sin θ)

def line_polar_equation (θ : ℝ) : θ = Real.pi / 3

theorem general_equation_of_curve :
  ∀ (x y θ : ℝ), (x, y) = curve_parametric_equation θ → (x^2) / 3 + y^2 = 1 :=
by sorry

theorem point_on_ellipse_given_conditions :
  ∀ (x0 y0 : ℝ), (∀ t : ℝ, (x = x0 + 1/2 * t) ∧ (y = y0 + sqrt 3 / 2 * t)) ∧
                  (∃ t1 t2 : ℝ, (x0 + 1/2 * t1, y0 + sqrt 3 / 2 * t1) ∈ curve_parametric_equation ∧
                               (x0 + 1/2 * t2, y0 + sqrt 3 / 2 * t2) ∈ curve_parametric_equation ∧
                               abs (t1 * t2) = 2) →
                  x0^2 + 3 * y0^2 = 8 :=
by sorry

end general_equation_of_curve_point_on_ellipse_given_conditions_l448_448404


namespace find_smallest_pqrs_sum_l448_448435

variables (p q r s : ℕ) -- We use natural numbers to ensure p, q, r, s are positive.

theorem find_smallest_pqrs_sum :
  let A := !![4, 0; 0, 3],
      B := !![p, q; r, s],
      C := !![24, 16; -30, -19] in
  (A ⬝ B = B ⬝ C → p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 → p + q + r + s = 14) :=
sorry

end find_smallest_pqrs_sum_l448_448435


namespace even_odd_product_l448_448432

theorem even_odd_product (n : ℕ) (i : Fin n → Fin n) (h_perm : ∀ j : Fin n, ∃ k : Fin n, i k = j) :
  (∃ l, l % 2 = 0) → 
  ∀ (k : Fin n), ¬(i k = k) → 
  (n % 2 = 0 → (∃ m : ℤ, m + 1 % 2 = 1) ∨ (∃ m : ℤ, m + 1 % 2 = 0)) ∧ 
  (n % 2 = 1 → (∃ m : ℤ, m + 1 % 2 = 0)) :=
by
  sorry

end even_odd_product_l448_448432


namespace greatest_value_of_x_l448_448474

theorem greatest_value_of_x
  (x : ℕ)
  (h1 : x % 4 = 0) -- x is a multiple of 4
  (h2 : x > 0) -- x is positive
  (h3 : x^3 < 2000) -- x^3 < 2000
  : x ≤ 12 :=
by
  sorry

end greatest_value_of_x_l448_448474


namespace race_problem_l448_448390

theorem race_problem 
    (d : ℕ) (a1 : ℕ) (a2 : ℕ) 
    (h1 : d = 60)
    (h2 : a1 = 10)
    (h3 : a2 = 20) 
    (const_speed : ∀ (x y z : ℕ), x * y = z → y ≠ 0 → x = z / y) :
  (d - d * (d - a1) / (d - a2) = 12) := 
by {
  sorry
}

end race_problem_l448_448390


namespace bicycle_cost_l448_448921

theorem bicycle_cost (CP_A SP_B SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 225) : CP_A = 150 :=
by
  sorry

end bicycle_cost_l448_448921


namespace problem1_problem2_l448_448679

-- Part 1: Prove that the sum evaluates correctly under given conditions
theorem problem1 (A B t : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, a n = A * t^(n-1) + B * n + 1)
  (h2 : ∀ i, b (2 * i) = (Nat.choose 10 i)) :
  A = 0 → B = 1 → t > 1 → (∑ n in Finset.range 10, a (n+1) * b (2 * (n+1))) = 6143 := 
by
  intros hA hB ht
  sorry

-- Part 2: Prove that the value of t is 2 under given conditions
theorem problem2 (A B t : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, a n = A * t^(n-1) + B * n + 1)
  (h2 : ∀ i, b (2 * i) = (10.choose i))
  (h3 : ∑ n in Finset.range 10, (2 * a (n+1) - 2^(n+1)) * b (2 * (n+1)) = 2^11 - 2) :
  A = 1 → B = 0 → t = 2 := 
by
  intros hA hB
  sorry

end problem1_problem2_l448_448679


namespace irrational_numD_l448_448534

-- Define each real number given in the problem
def numA : ℝ := 1 / 2
def numB : ℝ := 1 / 3
def numC : ℝ := Real.sqrt 4
def numD : ℝ := Real.sqrt 5

-- State that numD is irrational
theorem irrational_numD : Irrational numD :=
sorry

end irrational_numD_l448_448534


namespace girls_additional_laps_l448_448411

def distance_per_lap : ℚ := 1 / 6
def boys_laps : ℕ := 34
def boys_distance : ℚ := boys_laps * distance_per_lap
def girls_distance : ℚ := 9
def additional_distance : ℚ := girls_distance - boys_distance
def additional_laps (distance : ℚ) (lap_distance : ℚ) : ℚ := distance / lap_distance

theorem girls_additional_laps :
  additional_laps additional_distance distance_per_lap = 20 := 
by
  sorry

end girls_additional_laps_l448_448411


namespace age_of_B_l448_448543

-- Define the ages based on the conditions
def A (x : ℕ) : ℕ := 2 * x + 2
def B (x : ℕ) : ℕ := 2 * x
def C (x : ℕ) : ℕ := x

-- The main statement to be proved
theorem age_of_B (x : ℕ) (h : A x + B x + C x = 72) : B 14 = 28 :=
by
  -- we need the proof here but we will put sorry for now
  sorry

end age_of_B_l448_448543


namespace machine_no_repair_l448_448503

def nominal_mass (G_dev: ℝ) := G_dev / 0.1

theorem machine_no_repair (G_dev: ℝ) (σ: ℝ) (non_readable_dev_lt: ∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) : 
  (G_dev = 39) ∧ (σ ≤ G_dev) ∧ (∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) ∧ (G_dev ≤ 0.1 * nominal_mass(G_dev)) → 
  ¬ machine.requires_repair :=
by
  sorry

end machine_no_repair_l448_448503


namespace cubic_roots_reciprocal_squares_sum_l448_448262

-- Define the roots a, b, and c
variables (a b c : ℝ)

-- Define the given cubic equation conditions
variables (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6)

-- Define the target statement
theorem cubic_roots_reciprocal_squares_sum :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 :=
by
  sorry

end cubic_roots_reciprocal_squares_sum_l448_448262


namespace average_speed_of_trip_l448_448919

theorem average_speed_of_trip :
  let distance_local := 60
  let speed_local := 20
  let distance_highway := 120
  let speed_highway := 60
  let total_distance := distance_local + distance_highway
  let time_local := distance_local / speed_local
  let time_highway := distance_highway / speed_highway
  let total_time := time_local + time_highway
  let average_speed := total_distance / total_time
  average_speed = 36 := 
by 
  sorry

end average_speed_of_trip_l448_448919


namespace fraction_of_managers_l448_448130

theorem fraction_of_managers (female_managers : ℕ) (total_female_employees : ℕ)
  (total_employees: ℕ) (male_employees: ℕ) (f: ℝ) :
  female_managers = 200 →
  total_female_employees = 500 →
  total_employees = total_female_employees + male_employees →
  (f * total_employees) = female_managers + (f * male_employees) →
  f = 0.4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_managers_l448_448130


namespace log_div_simplify_l448_448533

def log_identity1 (x : ℝ) (y : ℝ) (h₁: x = 16) (h₂ : y = 1/16) : x = 16 ∧ y = 16⁻¹ := sorry

theorem log_div_simplify : log 16 / log (1 / 16) = -1 :=
by
  have h₁ : 16 = 16 := by refl
  have h₂ : 1 / 16 = 16⁻¹ := by norm_num
  have log_property : log (16⁻¹) = - log 16 := sorry
  calc
    log 16 / log (1 / 16)
        = log 16 / log (16⁻¹) : by rw [←h₂]
    ... = log 16 / (- log 16) : by rw [log_property]
    ... = -1 : by norm_num

end log_div_simplify_l448_448533


namespace mean_median_mode_equal_l448_448963

noncomputable def median (l : List ℕ) : ℕ :=
  let sorted := l.sort
  if sorted.length % 2 = 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

noncomputable def mean (l : List ℕ) : ℚ :=
  l.sum / l.length

def mode (l : List ℕ) : ℕ :=
  let freq_map := l.foldl (fun m n => m.insert n (m.find_d n + 1)) Std.HashMap.empty
  freq_map.toList.maxBy (fun p => p.snd).fst

theorem mean_median_mode_equal : ∀ (l : List ℕ), l = [2, 2, 4, 5, 7, 9, 4] → mean l = 4 ∧ median l = 4 ∧ mode l = 4 :=
by
  intro l hl
  rw [hl]
  have h_mean : mean [2, 2, 4, 5, 7, 9, 4] = 4 := sorry
  have h_median : median [2, 2, 4, 5, 7, 9, 4] = 4 := sorry
  have h_mode : mode [2, 2, 4, 5, 7, 9, 4] = 4 := sorry
  exact ⟨h_mean, h_median, h_mode⟩

end mean_median_mode_equal_l448_448963


namespace train_length_proof_l448_448589

noncomputable def train_length_problem : Prop :=
  let train_speed := 56 -- Speed of train in km/hr
  let man_speed := 6 -- Speed of man in km/hr (opposite direction)
  let time_to_pass := 6.386585847325762 -- Time in seconds
  let relative_speed_km_hr := train_speed + man_speed
  let relative_speed_m_s := (relative_speed_km_hr * 1000) / 3600 -- Convert to m/s
  let length_of_train := relative_speed_m_s * time_to_pass -- Calculate length
  length_of_train = 110

theorem train_length_proof : train_length_problem :=
by
  let train_speed := 56 -- Speed of train in km/hr
  let man_speed := 6 -- Speed of man in km/hr (opposite direction)
  let time_to_pass := 6.386585847325762
  let relative_speed_km_hr := train_speed + man_speed
  let relative_speed_m_s := (relative_speed_km_hr * 1000) / 3600
  let length_of_train := relative_speed_m_s * time_to_pass
  have h1 : length_of_train = 110 := sorry
  exact h1

end train_length_proof_l448_448589


namespace range_of_g_l448_448338

theorem range_of_g (m : ℝ) (hx : 0 < x ∧ x ≤ 1) (hm : m < 0) : 
  ∃ y, y ∈ (1, ∞) ∧ (∃ x ∈ (0, 1], g(x) = y) :=
by
  sorry

end range_of_g_l448_448338


namespace opposite_number_113_is_114_l448_448046

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l448_448046


namespace desired_sum_of_angles_l448_448746

-- Define the basic structure of Triangle and Points
structure Triangle :=
(A B C : Point)

structure Isosceles (T : Triangle) :=
(eqn1 : dist T.A T.B = dist T.A T.C)
(angle_A : ℚ)

structure PointsOnSide (T : Triangle) (n : ℕ) :=
(D : Point)
(K : Fin n → Point)

noncomputable def sum_of_angles (T : Triangle) (isosceles : Isosceles T) (points : PointsOnSide T n) : ℚ :=
sorry

theorem desired_sum_of_angles (T : Triangle) (isosceles : Isosceles T) (points : PointsOnSide T n) :
  sum_of_angles T isosceles points = isosceles.angle_A / 2 :=
sorry


end desired_sum_of_angles_l448_448746


namespace police_station_distance_l448_448203

theorem police_station_distance (thief_speed police_speed: ℝ) (delay chase_time: ℝ) 
  (h_thief_speed: thief_speed = 20) 
  (h_police_speed: police_speed = 40) 
  (h_delay: delay = 1)
  (h_chase_time: chase_time = 4) : 
  ∃ D: ℝ, D = 60 :=
by
  sorry

end police_station_distance_l448_448203


namespace size_of_coffee_cup_l448_448804

-- Define the conditions and the final proof statement
variable (C : ℝ) (h1 : (1/4) * C) (h2 : (1/2) * C) (remaining_after_cold : (1/4) * C - 1 = 2)

theorem size_of_coffee_cup : C = 6 := by
  -- Here the proof would go, but we omit it with sorry
  sorry

end size_of_coffee_cup_l448_448804


namespace irene_age_is_46_l448_448271

def eddie_age : ℕ := 92

def becky_age (e_age : ℕ) : ℕ := e_age / 4

def irene_age (b_age : ℕ) : ℕ := 2 * b_age

theorem irene_age_is_46 : irene_age (becky_age eddie_age) = 46 := 
  by
    sorry

end irene_age_is_46_l448_448271


namespace find_B_investment_l448_448545

variables (A B C : ℝ) (x : ℝ)

-- A's investment
def Invest_A := 5000

-- B's investment
def Invest_B := x

-- C's investment
def Invest_C := 9000

-- C's profit share
def Profit_C := 36000

-- Total profit
def Total_Profit := 88000

-- Ratio of C's profit to total profit
def C_Profit_Ratio := Profit_C / Total_Profit

-- Total investment
def Total_Invest := Invest_A + Invest_B + Invest_C

-- Ratio of C's investment to total investment
def C_Invest_Ratio := Invest_C / Total_Invest

-- Proof that B's investment x is 8000
theorem find_B_investment (h : C_Invest_Ratio = C_Profit_Ratio) : x = 8000 := 
by 
  -- Skipping proof
  sorry

end find_B_investment_l448_448545


namespace third_beats_seventh_l448_448392

-- Definitions and conditions
variable (points : Fin 8 → ℕ)
variable (distinct_points : Function.Injective points)
variable (sum_last_four : points 1 = points 4 + points 5 + points 6 + points 7)

-- Proof statement
theorem third_beats_seventh 
  (h_distinct : ∀ i j, i ≠ j → points i ≠ points j)
  (h_sum : points 1 = points 4 + points 5 + points 6 + points 7) :
  points 2 > points 6 :=
sorry

end third_beats_seventh_l448_448392


namespace min_tiles_needed_l448_448958

-- Definitions for the problem
def tile_width : ℕ := 3
def tile_height : ℕ := 4

def region_width_ft : ℕ := 2
def region_height_ft : ℕ := 5

def inches_in_foot : ℕ := 12

-- Conversion
def region_width_in := region_width_ft * inches_in_foot
def region_height_in := region_height_ft * inches_in_foot

-- Calculations
def region_area := region_width_in * region_height_in
def tile_area := tile_width * tile_height

-- Theorem statement
theorem min_tiles_needed : region_area / tile_area = 120 := 
  sorry

end min_tiles_needed_l448_448958


namespace opposite_number_on_circle_l448_448028

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l448_448028


namespace parallel_trans_l448_448088

-- Parallel relationship
def parallel (L1 L2 : Type) : Prop := sorry

-- Definitions of lines
constant Line : Type
constant LineA : Line
constant LineB : Line
constant LineC : Line

-- Conditions
axiom ma_parallel : parallel LineA LineC
axiom mb_parallel : parallel LineB LineC

-- Theorem to prove
theorem parallel_trans :
  (parallel LineA LineC) ∧ (parallel LineB LineC) → (parallel LineA LineB) :=
by {
  intro h,
  cases h with ha hb,
  exact sorry
}

end parallel_trans_l448_448088


namespace parallel_trans_l448_448087

-- Parallel relationship
def parallel (L1 L2 : Type) : Prop := sorry

-- Definitions of lines
constant Line : Type
constant LineA : Line
constant LineB : Line
constant LineC : Line

-- Conditions
axiom ma_parallel : parallel LineA LineC
axiom mb_parallel : parallel LineB LineC

-- Theorem to prove
theorem parallel_trans :
  (parallel LineA LineC) ∧ (parallel LineB LineC) → (parallel LineA LineB) :=
by {
  intro h,
  cases h with ha hb,
  exact sorry
}

end parallel_trans_l448_448087


namespace urn_red_percentage_l448_448250

theorem urn_red_percentage (total_balls : ℕ) (red_percentage : ℝ) (desired_red_percentage : ℝ) (red_balls : ℕ) (blue_balls : ℕ) (balls_to_remove : ℕ) :
  total_balls = 120 →
  red_percentage = 0.40 →
  desired_red_percentage = 0.80 →
  red_balls = (red_percentage * total_balls).to_nat →
  blue_balls = total_balls - red_balls →
  red_percentage * total_balls = 48 →
  desired_red_percentage * (total_balls - balls_to_remove) = red_balls →
  balls_to_remove = 60 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end urn_red_percentage_l448_448250


namespace determine_monotonically_increasing_interval_l448_448223

noncomputable theory
open Real

/-- Problem Statement: Given the function f(x) = 7 * sin(x - π / 6), prove that it is monotonically increasing in the interval (0, π/2). -/
def monotonically_increasing_interval (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2

def function_f (x : ℝ) : ℝ :=
  7 * sin (x - π / 6)

theorem determine_monotonically_increasing_interval :
  ∀ x : ℝ, monotonically_increasing_interval x → monotone_on function_f (set.Ioo 0 (π / 2)) :=
sorry

end determine_monotonically_increasing_interval_l448_448223


namespace tree_volume_estimation_l448_448980

def given_data (x y : Fin 10 → ℝ) : Prop :=
  (∑ i, x i = 0.6) ∧
  (∑ i, y i = 3.9) ∧
  (∑ i, (x i) ^ 2 = 0.038) ∧
  (∑ i, (y i) ^ 2 = 1.6158) ∧
  (∑ i, x i * y i = 0.2474) ∧
  let X := 186 in
  let avg_x := 0.06 in
  let avg_y := 0.39 in
  let r := 0.97 in
  (X, avg_y / avg_x * X) = (186, 1209)

theorem tree_volume_estimation
  {x y : Fin 10 → ℝ}
  (h : given_data x y)
  : let avg_x := 0.06 in
    let avg_y := 0.39 in
    let r := 0.97 in
    let X := 186 in
    (avg_x = 0.06) ∧
    (avg_y = 0.39) ∧
    (r = 0.97) ∧
    (avg_y / avg_x * X = 1209) := 
by 
  sorry

end tree_volume_estimation_l448_448980


namespace seating_arrangements_correct_l448_448200

-- Define the round table with six seats
inductive Seat 
| A | B | C | D | E | F
deriving DecidableEq, Inhabited

-- Defining the problem condition of seating three adults and three children such that no two children sit next to each other.
def is_valid_arrangement (seating : List Seat) : Prop :=
  ∀ i j, i ≠ j → (seating i = seating j → False) ∧ 
  (seating i ∈ [Seat.A, Seat.B, Seat.C, Seat.D, Seat.E, Seat.F]) ∧ 
  (seating nth i ≠ seating nth (i+1))  -- No two children can sit next to each other

noncomputable def count_valid_seating_arrangements : ℕ :=
  (1 * (8 * 6))  -- Fix one adult in one seat, calculate ways to place the remaining two adults, and then the permutations of children.

theorem seating_arrangements_correct : count_valid_seating_arrangements = 48 := by
  sorry

end seating_arrangements_correct_l448_448200


namespace inequality_holds_l448_448581

theorem inequality_holds (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end inequality_holds_l448_448581


namespace irene_age_is_46_l448_448272

def eddie_age : ℕ := 92

def becky_age (e_age : ℕ) : ℕ := e_age / 4

def irene_age (b_age : ℕ) : ℕ := 2 * b_age

theorem irene_age_is_46 : irene_age (becky_age eddie_age) = 46 := 
  by
    sorry

end irene_age_is_46_l448_448272


namespace two_digit_number_reverse_sum_l448_448268

theorem two_digit_number_reverse_sum :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ let digit_sum (n : ℕ) := (n / 10) + (n % 10) in 
    digit_sum n = 14}.card = 5 :=
by
  sorry

end two_digit_number_reverse_sum_l448_448268


namespace digits_not_in_sequence_l448_448928

-- Define the sequence
def sequence : ℕ → ℕ
| 0 := 2
| 1 := 3
| n := let x := (sequence (n - 1)) * (sequence (n - 2))
       in if x > 9 then x / 10 else x

-- Proposition to prove 5, 7, 9 do not appear in the sequence
theorem digits_not_in_sequence (n : ℕ) :
  sequence n ≠ 5 ∧ sequence n ≠ 7 ∧ sequence n ≠ 9 :=
  sorry

end digits_not_in_sequence_l448_448928


namespace possible_remainder_degrees_l448_448911

-- Definitions
def divisor : Polynomial ℝ := 3 * Polynomial.X ^ 3 - Polynomial.X ^ 2 + 4 * Polynomial.X - 12

-- Theorem stating the degrees of the remainder when divided by the divisor
theorem possible_remainder_degrees (r : Polynomial ℝ) (p : Polynomial ℝ) (h : p = divisor * q + r) : 
  (Polynomial.degree r < Polynomial.degree divisor) :=
sorry

end possible_remainder_degrees_l448_448911


namespace length_of_AB_l448_448412

theorem length_of_AB (A B C : ℝ) (angle_A : A = 90) (BC : B = 30) (tan_C_eq : tan C = 3 * cos C) : A = 26 :=
by
  sorry

end length_of_AB_l448_448412


namespace sqrt_identity_correct_l448_448652

theorem sqrt_identity_correct:
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a < b ∧ (sqrt a + sqrt b = sqrt (1 + sqrt (45 + 18 * sqrt 5))) ∧ (a, b) = (3, 16) := by
  sorry

end sqrt_identity_correct_l448_448652


namespace expansion_arithmetic_seq_terms_l448_448728

theorem expansion_arithmetic_seq_terms (n : ℕ)
  (h1 : ∀ r : ℕ, (1 / 2) ^ r * Nat.choose n r * x ^ ((2 * n - 3 * r) / 4) = 
                  (Nat.choose n 0, (1 / 2) * Nat.choose n 1, (1 / 4) * Nat.choose n 2))
  (h_n : n = 8) :
  (let T5 := (Nat.choose 8 4 * ((sqrt x) ^ 4) * (1 / (24 * x)) ^ 4) in
      T5 = (35 / 8) * x) ∧
  (let terms := ((sqrt x) ^ 4, (35 / 8) * x, 1 / (256 * x ^ 2)) in
      terms = (x ^ 4, (35 / 8) * x, 1 / (256 * x ^ 2))) :=
sorry

end expansion_arithmetic_seq_terms_l448_448728


namespace correct_calculation_l448_448913

theorem correct_calculation (x y a b : ℝ) :
  (3*x + 3*y ≠ 6*x*y) ∧
  (x + x ≠ x^2) ∧
  (-9*y^2 + 16*y^2 ≠ 7) ∧
  (9*a^2*b - 9*a^2*b = 0) :=
by
  sorry

end correct_calculation_l448_448913


namespace knights_in_gamma_l448_448399

theorem knights_in_gamma (k L : ℕ) (a b c d : ℕ) 
  (h_total_aff : a + b + c + d = 430) 
  (h_knights_aff : 200) 
  (h_extra_aff : 230) 
  (h_liars_count : L = 115) 
  (h_liars_in_gamma : L - (119 - k) = k - 4)
  (h_liars_gt_knights : ∀ (x ∈ {a, b, c, d}), x ≠ k → (L - (x - k) > k)) :
  k = 4 := by 
  sorry

end knights_in_gamma_l448_448399


namespace average_root_and_volume_correlation_coefficient_total_volume_l448_448981

noncomputable section

open Real

def root_cross_sectional_areas : List ℝ := 
[0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]

def volumes : List ℝ := 
[0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]

def sum_root_cross_sectional_areas := 0.6
def sum_volumes := 3.9

def sum_squares_root := 0.038
def sum_squares_volume := 1.6158
def sum_product := 0.2474

theorem average_root_and_volume (n : ℝ) (sumx : ℝ) (sumy : ℝ) :
  n = 10 ∧ sumx = 0.6 ∧ sumy = 3.9 →
  (sumx / n = 0.06) ∧ (sumy / n = 0.39) :=
by
  intros h
  cases h with hn hs
  cases hs with hx hy
  simp [hn, hx, hy]

theorem correlation_coefficient (s_xy : ℝ) (s_xx : ℝ) (s_yy : ℝ) :
  s_xy = 0.0134 ∧ s_xx = 0.002 ∧ s_yy = 0.0948 →
  (s_xy / (Real.sqrt (s_xx * s_yy)) = 0.97) :=
by
  intros h
  cases h with hxy hs
  cases hs with hxx hyy
  have h : (sqrt (s_xx * s_yy)) = 0.01377 := by
    simp [hxx, hyy, sqrt_mul, sqrt_eq_rfl]
  simp [hxy, h] 

theorem total_volume (total_root_area : ℝ) (avg_root : ℝ) (avg_vol : ℝ) :
  total_root_area = 186 ∧ avg_root = 0.06 ∧ avg_vol = 0.39 →
  (total_root_area * avg_vol / avg_root = 1209) :=
by
  intros h
  cases h with hx hs
  cases hs with hr hv
  simp [hx, hr, hv]

#eval average_root_and_volume 10 sum_root_cross_sectional_areas sum_volumes -- Proof goal 1
#eval correlation_coefficient 0.0134 0.002 0.0948 -- Proof goal 2
#eval total_volume 186 0.06 0.39 -- Proof goal 3

end average_root_and_volume_correlation_coefficient_total_volume_l448_448981


namespace compare_log2_sqrt5_l448_448260

variable (a b : ℝ)

theorem compare_log2_sqrt5 (h₀ : 0 ≤ a)
                          (h₁ : 0 ≤ b)
                          (am_gm_ineq : ∀ (a b : ℝ), 0 ≤ a → 0 ≤ b → (a + b) / 2 ≥ sqrt (a * b))
                          (log_prop : ∀ (A B : ℝ), A > 0 → B > 0 → log (A * B) / log 2 = log A / log 2 + log B / log 2) :
  2 + log 6 / log 2 > 2 * sqrt 5 := by
    sorry

end compare_log2_sqrt5_l448_448260


namespace prove_d_minus_c_l448_448117

def reflect_about_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

def rotate_90_clockwise_around (center : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  let dx := p.1 - center.1
  let dy := p.2 - center.2
  (center.1 + dy, center.2 - dx)

def reverse_transform_FINAL_to_ORIGINAL (final : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let reflected := reflect_about_y_eq_x final
  rotate_90_clockwise_around center reflected

constant Q : ℝ × ℝ
constant c : ℝ
constant d : ℝ

axiom condition_Q : Q = (c, d)
axiom transform_final_point : reverse_transform_FINAL_to_ORIGINAL (4, -1) (2, 3) = Q

theorem prove_d_minus_c : d - c = -1 :=
by
  have h1 : reverse_transform_FINAL_to_ORIGINAL (4, -1) (2, 3) = (1, 0) := 
    -- calculation steps for reflection and rotation omitted here
    sorry
  have h2 : Q = (1, 0) := by
    rw [←condition_Q, transform_final_point, h1]
  have h3 : (c, d) = (1, 0) := by
    rw [←h2, condition_Q]
  simp at h3
  simp [h3]
  sorry

end prove_d_minus_c_l448_448117


namespace sum_of_solutions_l448_448158

theorem sum_of_solutions :
  (∑ x in (Finset.filter (λ x : ℕ, 0 < x ∧ x ≤ 30 ∧ 13 * (3 * x - 2) % 12 = 30 % 12) (Finset.range 31)), x) = 112 :=
by
  sorry

end sum_of_solutions_l448_448158


namespace intersection_A_B_l448_448712

noncomputable def set_A : Set ℝ := { x | ∃ k : ℤ, x = 2 * k + 1 }
def set_B : Set ℝ := { x | 0 < x ∧ x < 5 }

theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by
  sorry

end intersection_A_B_l448_448712


namespace sum_of_coefficients_l448_448310

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) (hx : (1 - 2 * x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 12 :=
sorry

end sum_of_coefficients_l448_448310


namespace lower_bound_for_bar_f_l448_448551

theorem lower_bound_for_bar_f (N : ℕ) (T : ℕ → ℕ) :
  ∃ (f : ℕ → ℕ), (∀ n, f n = max (λ k, T k) (range n)) →
  (∀ T, exist f such that (∀ n, f n = max (λ k, T k) (range n)) 
  → ∃ m, m = ⌈ log n ⌉ ) 
sorry

end lower_bound_for_bar_f_l448_448551


namespace range_of_a_l448_448704

theorem range_of_a (x a : ℝ) :
  (∀ x, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l448_448704


namespace ordered_pair_identity_l448_448289

theorem ordered_pair_identity :
  ∃ (a b : ℕ), a < b ∧ 
  a > 0 ∧ b > 0 ∧ 
  (sqrt 1 + sqrt (45 + 16 * sqrt 5) = sqrt a + sqrt b) ∧
  a = 1 ∧ b = 5 :=
by
  sorry

end ordered_pair_identity_l448_448289


namespace monotonic_interval_l448_448239

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin(x - Real.pi / 6)

def is_monotonically_increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem monotonic_interval : is_monotonically_increasing_interval f 0 (Real.pi / 2) :=
sorry

end monotonic_interval_l448_448239


namespace graph_transform_l448_448516

theorem graph_transform : ∀ (f : ℝ → ℝ),
  (∀ (x : ℝ), f(2 * x + π) = (λ x, f (x + π)) (2 * x)) :=
by
  intros f x
  sorry

end graph_transform_l448_448516


namespace six_propositions_validity_l448_448872

theorem six_propositions_validity :
  let P1 := (smallest_positive_period (λ x, sin^4 x - cos^4 x) = π)
  let P2 := (∀ α : ℝ, (∃ k : ℤ, α = 2 * k * π + π/2) ↔ (∃ k : ℤ, α = k * π + π/2))
  let P3 := (∃ x : ℝ, sin x = x)
  let P4 := (monotone (λ x, tan x))
  let P5 := (even (λ x, sin (x - π/2)))
  let P6 := (∀ (a b : ℝ^2), a • b = 0 → a = 0 ∨ b = 0) in
  P1 ∧ P5 ∧ ¬P2 ∧ ¬P3 ∧ ¬P4 ∧ ¬P6 :=
begin
  sorry
end

end six_propositions_validity_l448_448872


namespace line_equation_passes_through_and_has_normal_l448_448379

theorem line_equation_passes_through_and_has_normal (x y : ℝ) 
    (H1 : ∃ l : ℝ → ℝ, l 3 = 4)
    (H2 : ∃ n : ℝ × ℝ, n = (1, 2)) : 
    x + 2 * y - 11 = 0 :=
sorry

end line_equation_passes_through_and_has_normal_l448_448379


namespace number_of_sheets_in_stack_l448_448966

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l448_448966


namespace luke_points_per_round_l448_448014

-- Define the total number of points scored 
def totalPoints : ℕ := 8142

-- Define the number of rounds played
def rounds : ℕ := 177

-- Define the points gained per round which we need to prove
def pointsPerRound : ℕ := 46

-- Now, we can state: if Luke played 177 rounds and scored a total of 8142 points, then he gained 46 points per round
theorem luke_points_per_round :
  (totalPoints = 8142) → (rounds = 177) → (totalPoints / rounds = pointsPerRound) := by
  sorry

end luke_points_per_round_l448_448014


namespace each_person_ate_15_cakes_l448_448018

theorem each_person_ate_15_cakes :
  ∀ (cakes : ℕ) (friends : ℕ), 
  cakes = 30 → friends = 2 → 
  (cakes / friends) = 15 := 
by
  intros cakes friends hcakes hfriends
  rw [hcakes, hfriends]
  norm_num
  exact rfl

end each_person_ate_15_cakes_l448_448018


namespace equilateral_triangle_vertex_on_parabola_l448_448450

theorem equilateral_triangle_vertex_on_parabola :
  ∀ k : ℕ, 
  let x := (sqrt 3 / 2) * (2 * k - 1)
  let y := k^2 - (2 * k - 1) / 2
  in y = (1 : ℝ) / 3 * x^2 + (1 : ℝ) / 4 :=
begin
  intro k,
  let x := (sqrt 3 / 2) * (2 * k - 1),
  let y := k^2 - (2 * k - 1) / 2,
  sorry
end

end equilateral_triangle_vertex_on_parabola_l448_448450


namespace parabola_ellipse_intersection_distance_l448_448570

theorem parabola_ellipse_intersection_distance :
  ∃ (A : ℝ), 
  let ellipse : set (ℝ × ℝ) := { p | (p.1^2 / 16) + (p.2^2 / 36) = 1 },
  let parabola := { p | p.2 = (A * p.1^2 + real.sqrt 5) } in
  (∀ (x y : ℝ), x ∈ ellipse → y ∈ parabola → y ∈ ellipse → 
  ∃ d : ℝ, d = 2 * abs (real.sqrt 5 / 3)) := sorry

end parabola_ellipse_intersection_distance_l448_448570


namespace ratio_shaded_unshaded_area_eq_1_l448_448405

theorem ratio_shaded_unshaded_area_eq_1 
  (P Q R S T W : Type)
  [rhombus P Q R S] 
  [midpoint T P S]
  [midpoint W S R] : 
  ratio_unshaded_shaded_area P Q R S T W = 1 :=
sorry

end ratio_shaded_unshaded_area_eq_1_l448_448405


namespace exists_continuous_function_with_order_5_but_not_order_3_l448_448832

theorem exists_continuous_function_with_order_5_but_not_order_3 :
  ∃ (f : ℝ → ℝ), continuous f ∧ (∀ x ∈ [0,1], f x ∈ [0,1]) ∧ ¬(∃ x, f^[3] x = x ∧ ∀ m < 3, f^[m] x ≠ x) ∧ (∃ x, f^[5] x = x ∧ ∀ m < 5, f^[m] x ≠ x) :=
sorry

end exists_continuous_function_with_order_5_but_not_order_3_l448_448832


namespace isosceles_right_triangle_segment_length_l448_448111

noncomputable def hypotenuse := 40
noncomputable def radius := 9
noncomputable def segment_length := Real.sqrt 82

theorem isosceles_right_triangle_segment_length :
  ∃ (triangle : Triangle ℝ), IsoscelesRightTriangle triangle ∧ triangle.hypotenuse = hypotenuse ∧ 
  ∃ (circle : Circle ℝ), circle.radius = radius ∧ circle.touchesMidpoint (triangle.hypotenuse) ∧ 
  (circle.segment_cutoff_by_leg triangle.leg = segment_length) := 
sorry

end isosceles_right_triangle_segment_length_l448_448111


namespace max_triangle_area_axial_section_l448_448760

-- Definition of the problem conditions
noncomputable def cone (V : Point) (R : ℝ) : Prop :=
  ∃ (base : Circle), (base.center = (0, 0, 0)) ∧ (base.radius = R) ∧ (V = (0, 0, some_height))

noncomputable def is_axial_section (plane : Plane) (cone : Cone) : Prop :=
  ∃ (diameter : Line), (diameter.is_diameter_of_cone_base cone) ∧ (plane.contains_line diameter)

-- Main theorem to prove
theorem max_triangle_area_axial_section (V : Point) (R : ℝ) (plane : Plane) :
  (cone V R) → (∃ (triangle : Triangle), (triangle.is_intersection_with_plane plane cone V) → 
  (triangle.area = max_area ↔ (is_axial_section plane cone V))) :=
sorry

end max_triangle_area_axial_section_l448_448760


namespace sin_beta_two_alpha_plus_beta_l448_448312

namespace TrigProofs

-- Define the given constraints and variables
variables (α β : ℝ)
axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : 0 < β ∧ β < π / 2
axiom h3 : cos α = 4 / 5
axiom h4 : cos (α + β) = 3 / 5

-- Statement to prove for Part 1
theorem sin_beta : sin β = 7 / 25 :=
by
  sorry

-- Statement to prove for Part 2
theorem two_alpha_plus_beta : 2 * α + β = π / 2 :=
by
  sorry

end TrigProofs

end sin_beta_two_alpha_plus_beta_l448_448312


namespace average_value_is_9z_l448_448280

open Real

/-- 
  Prove that the average value of the numbers 0, 3z, 6z, 12z, and 24z is 9z.
-/
theorem average_value_is_9z (z : ℝ) : 
  let avg := (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 in
  avg = 9 * z :=
by
  let s := 0 + 3 * z + 6 * z + 12 * z + 24 * z
  let n := 5
  let avg := s / n
  have h : s = 45 * z := by ring
  have h2 : avg = (45 * z) / n := by rw [h]
  have h3 : (45 * z) / 5 = 9 * z := by norm_num
  rw [h2, h3]
  exact eq.refl (9 * z)

end average_value_is_9z_l448_448280


namespace rectangle_area_l448_448956

theorem rectangle_area :
  ∃ (x y : ℝ), (x + 3.5) * (y - 1.5) = x * y ∧
               (x - 3.5) * (y + 2.5) = x * y ∧
               2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y ∧
               x * y = 196 :=
by
  sorry

end rectangle_area_l448_448956


namespace sum_of_squared_geometric_series_l448_448106

theorem sum_of_squared_geometric_series (S : ℝ) (q : ℝ) (h_q : abs q < 1) (h_S : S = 1 / (1 - q)) :
  let new_sum := (S^2) / (2 * S - 1) in new_sum = 1 / (1 - (q^2)) :=
by 
  sorry

end sum_of_squared_geometric_series_l448_448106


namespace fish_swim_north_l448_448555

theorem fish_swim_north:
  let westward_fish := 1800 in
  let eastward_fish := 3200 in
  let caught_eastward_fish := (2/5 : ℚ) * eastward_fish in
  let caught_westward_fish := (3/4 : ℚ) * westward_fish in
  let remaining_fish := 2870 in
  (westward_fish + eastward_fish + N) - (caught_eastward_fish + caught_westward_fish) = remaining_fish → 
  N = 500 :=
by 
  sorry

end fish_swim_north_l448_448555


namespace bus_speed_kmph_l448_448941

theorem bus_speed_kmph : 
  let distance := 600.048 
  let time := 30
  (distance / time) * 3.6 = 72.006 :=
by
  sorry

end bus_speed_kmph_l448_448941


namespace find_second_wrongly_copied_number_l448_448102

theorem find_second_wrongly_copied_number
  (average_wrong_copied : ℕ → ℝ)
  (sum_wrong_copied : ℕ → ℝ)
  (n : ℕ)
  (wrong_copied_first_difference : ℝ)
  (wrong_copied_second_value : ℝ)
  (correct_average : ℕ → ℝ)
  (correct_sum : ℕ → ℝ) :
  (average_wrong_copied n = 40.2) →
  (sum_wrong_copied n = 402) →
  (wrong_copied_first_difference = 17) →
  (correct_average n = 40.1) →
  (correct_sum n = 401) →
  (wrong_copied_second_value = ? x → 31) →
  (15 = 416 - 401) :=
by
  intros h_avg h_sum h_diff h_avg_correct h_sum_correct h_value
  sorry

end find_second_wrongly_copied_number_l448_448102


namespace find_AE_l448_448249

-- Definitions of the problem conditions
variables {A B C D E : Type}{P Q : A → B → Prop}
variables (AB BC AD CE DC AE : ℝ)

-- Ensure the conditions are correct
def isosceles_triangle (AB BC : ℝ) : Prop := AB = BC

def on_ray_beyond (BA AE : ℝ) (A E : Type) : Prop := sorry -- Placeholder for ray condition definition

def on_segment (BC D : Type) : Prop := sorry -- Placeholder for point on segment definition

def angle_condition (ADC AEC : ℝ) : Prop := ADC = 60 ∧ AEC = 60

def known_lengths (AD CE DC : ℝ) : Prop := AD = 13 ∧ CE = 13 ∧ DC = 9

-- The theorem to prove
theorem find_AE
  (h_isosceles : isosceles_triangle AB BC) 
  (h_on_ray_beyond : on_ray_beyond AB AE A E)
  (h_on_segment : on_segment BC D)
  (h_angles : angle_condition 60 60)
  (h_lengths : known_lengths AD CE DC) :
  AE = 4 := sorry

end find_AE_l448_448249


namespace total_spending_correct_l448_448946

-- Define the costs and number of children for each ride and snack
def cost_ferris_wheel := 5 * 5
def cost_roller_coaster := 7 * 3
def cost_merry_go_round := 3 * 8
def cost_bumper_cars := 4 * 6

def cost_ice_cream := 8 * 2 * 5
def cost_hot_dog := 6 * 4
def cost_pizza := 4 * 3

-- Calculate the total cost
def total_cost_rides := cost_ferris_wheel + cost_roller_coaster + cost_merry_go_round + cost_bumper_cars
def total_cost_snacks := cost_ice_cream + cost_hot_dog + cost_pizza
def total_spent := total_cost_rides + total_cost_snacks

-- The statement to prove
theorem total_spending_correct : total_spent = 170 := by
  sorry

end total_spending_correct_l448_448946


namespace angle_C_value_l448_448757

-- Definitions for the angles and sides of the triangle and the trigonometric function
variables {A B C : ℝ} {a b c : ℝ}
variables [triangle_side_a : a = opposite_side_A] [triangle_side_b : b = opposite_side_B] [triangle_side_c : c = opposite_side_C]

-- Given conditions
variable (h1 : c * sin A = a * cos C)

-- Statement of the theorem
theorem angle_C_value (h1 : c * sin A = a * cos C) : C = π/4 := sorry

end angle_C_value_l448_448757


namespace greatest_multiple_of_four_cubed_less_than_2000_l448_448476

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ x, (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 2000) ∧ ∀ y, (y > x) ∧ (y % 4 = 0) → y^3 ≥ 2000 :=
sorry

end greatest_multiple_of_four_cubed_less_than_2000_l448_448476


namespace opposite_number_in_circle_l448_448022

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l448_448022


namespace f_increasing_intervals_f_range_when_x_in_interval_l448_448349

noncomputable def f (x : ℝ) := 3 * sin x * cos x - sqrt 3 * cos x ^ 2 + sqrt 3 / 2

theorem f_increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (-π/6 + k * π) (π/3 + k * π) → 
  (∃ I : Set ℝ, Set.monotonic_on f I ∧ (I = Set.Icc (-π/6 + k * π) (π/3 + k * π))) :=
sorry

theorem f_range_when_x_in_interval :
  ∀ x : ℝ, x ∈ Set.Icc (π/6) (3 * π / 4) → f x ∈ Set.Icc (-3/2) (sqrt 3) :=
sorry

end f_increasing_intervals_f_range_when_x_in_interval_l448_448349


namespace probability_reach_edge_within_five_hops_l448_448305

-- Definitions
def grid_size : ℕ := 4
def center_square : ℕ × ℕ := (3, 3)
def max_hops : ℕ := 5

-- State transition probability based on a random move
def transition_prob : ℝ := 1 / 4

-- Calculating the probability of reaching an edge within the given hops
theorem probability_reach_edge_within_five_hops : true :=
  by
    -- Placeholder for states, should include center and edge state definitions
    -- Placeholder for transition probabilities
    -- Simplify the problem as illustrated in the conditions and solution
    sorry

end probability_reach_edge_within_five_hops_l448_448305


namespace proposition_truth_count_l448_448408

namespace Geometry

def is_obtuse_angle (A : Type) : Prop := sorry
def is_obtuse_triangle (ABC : Type) : Prop := sorry

def original_proposition (A : Type) (ABC : Type) : Prop :=
is_obtuse_angle A → is_obtuse_triangle ABC

def contrapositive_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_triangle ABC) → ¬ (is_obtuse_angle A)

def converse_proposition (ABC : Type) (A : Type) : Prop :=
is_obtuse_triangle ABC → is_obtuse_angle A

def inverse_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_angle A) → ¬ (is_obtuse_triangle ABC)

theorem proposition_truth_count (A : Type) (ABC : Type) :
  (original_proposition A ABC ∧ contrapositive_proposition A ABC ∧
  ¬ (converse_proposition ABC A) ∧ ¬ (inverse_proposition A ABC)) →
  ∃ n : ℕ, n = 2 :=
sorry

end Geometry

end proposition_truth_count_l448_448408


namespace infinite_product_of_sequence_l448_448578

noncomputable def a : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 2 + (a n - 2) ^ 2

theorem infinite_product_of_sequence :
  (∏ n in (range (1000)), a n) = 2 / 7 := sorry

end infinite_product_of_sequence_l448_448578


namespace distance_from_S_to_PQR_correct_l448_448459

noncomputable def distance_from_S_to_PQR (S P Q R : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  let SP := distance S P
  let SQ := distance S Q
  let SR := distance S R
  let PQ := distance P Q
  let PR := distance P R
  let QR := distance Q R
  have h_SP_SQ := orthogonal SP SQ
  have h_SP_SR := orthogonal SP SR
  have h_SQ_SR := orthogonal SQ SR
  have h_SP := SP = 15 := by sorry
  have h_SQ := SQ = 15 := by sorry
  have h_SR := SR = 9 := by sorry
  3 * sqrt 26

theorem distance_from_S_to_PQR_correct {S P Q R : EuclideanSpace ℝ (Fin 3)}
  (h_SP_or: orthogonal S P)
  (h_SQ_or: orthogonal S Q)
  (h_SR_or: orthogonal S R)
  (h_SP_len: distance S P = 15)
  (h_SQ_len: distance S Q = 15)
  (h_SR_len: distance S R = 9) :
  distance_from_S_to_PQR S P Q R = 3 * sqrt 26 :=
by {
  sorry
}

end distance_from_S_to_PQR_correct_l448_448459


namespace excellent_pairs_eq_sum_of_divisors_l448_448421

noncomputable def is_good_pair (ν : ℝ) (m : ℕ) (a b : ℕ) : Prop :=
  a * ⌈b * ν⌉ - b * ⌊a * ν⌋ = m

noncomputable def is_excellent_pair (ν : ℝ) (m a b : ℕ) : Prop :=
  (is_good_pair ν m a b) ∧ ¬(is_good_pair ν m (a - b) b) ∧ ¬(is_good_pair ν m a (b - a))

theorem excellent_pairs_eq_sum_of_divisors (ν : ℝ) (m : ℕ) [irr : irrational ν] :
  (set.univ.filter (is_excellent_pair ν m)).count = (Finset.divisors m).sum :=
sorry

end excellent_pairs_eq_sum_of_divisors_l448_448421


namespace pedro_scores_l448_448072

noncomputable def pedro_test_scores (t1 t2 t3 t4 t5 : ℕ) : Prop :=
  t1 = 92 ∧ t2 = 85 ∧ t3 = 78 ∧
  (2 * 87 * 5 - (t1 + t2 + t3)) = (t4 + t5) ∧
  t1 < 100 ∧ t2 < 100 ∧ t3 < 100 ∧ t4 < 100 ∧ t5 < 100 ∧
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t1 ≠ t4 ∧ t1 ≠ t5 ∧
  t2 ≠ t3 ∧ t2 ≠ t4 ∧ t2 ≠ t5 ∧
  t3 ≠ t4 ∧ t3 ≠ t5 ∧
  t4 ≠ t5

theorem pedro_scores : ∃ (t1 t2 t3 t4 t5 : ℕ), 
  pedro_test_scores t1 t2 t3 t4 t5 ∧ 
  [t1, t2, t3, t4, t5].sort (≥) = [92, 91, 89, 85, 78] :=
begin
  sorry
end

end pedro_scores_l448_448072


namespace Yan_distance_ratio_l448_448537

-- Define the problem conditions
def YanWalkingTime (b : ℝ) (v : ℝ) : ℝ := b / v
def YanCombinationTime (a b : ℝ) (v : ℝ) : ℝ := a / v + (a + b) / (7 * v)

-- Prove that the ratio of Yan's distance from his home to his distance from the stadium is 3/4
theorem Yan_distance_ratio (a b : ℝ) (v : ℝ) (h : YanWalkingTime b v = YanCombinationTime a b v) :
  a / b = 3 / 4 :=
by
  -- Time condition to equation
  have h_eq : b = a + (a + b) / 7, by sorry,
  -- Solving the equation to find the ratio
  have h_solve : b = (4 / 3) * a, by sorry,
  -- Taking inverse to find the desired ratio
  exact (eq_div_iff (by norm_num)).1 h_solve


end Yan_distance_ratio_l448_448537


namespace measure_angle_R_l448_448740

theorem measure_angle_R (P Q R : ℝ) (h1 : P + Q = 60) : R = 120 :=
by
  have sum_of_angles_in_triangle : P + Q + R = 180 := sorry
  rw [h1] at sum_of_angles_in_triangle
  linarith

end measure_angle_R_l448_448740


namespace range_of_a_l448_448702

noncomputable def f (x a : ℝ) := x^2 + x + 2 * a
noncomputable def g (x : ℝ) := -1 / x

theorem range_of_a:
  (∃ (x1 x2 a : ℝ), x1 < 0 ∧ 0 < x2 ∧
                    let y1 := f x1 a,
                        y2 := g x2,
                        t := 1 / x2
                    in 
                      (2 * x1 + 1 = t^2) ∧ 
                      (-2 / x2 = -x1^2 + 2 * a) ∧ 
                      0 < t ∧ t < 1
                    )
  → a ∈ set.Ioo (-1 : ℝ) (1 / 8 : ℝ) := 
sorry

end range_of_a_l448_448702


namespace quadratic_inequality_l448_448954

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality
  (a b c : ℝ)
  (h_pos : 0 < a)
  (h_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (x : ℝ) :
  f a b c x + f a b c (x - 1) - f a b c (x + 1) > -4 * a :=
  sorry

end quadratic_inequality_l448_448954


namespace lines_parallel_to_same_are_parallel_l448_448081

theorem lines_parallel_to_same_are_parallel (L1 L2 L3 : Type) [linear_ordered_space L1] 
  (h1 : L1 ∥ L3) (h2 : L2 ∥ L3) : L1 ∥ L2 :=
sorry

end lines_parallel_to_same_are_parallel_l448_448081


namespace find_angles_and_sides_l448_448738

noncomputable def sin_value (A : ℝ) : ℝ :=
  (sin (π / 2 + A) = 11 / 14)

noncomputable def cos_value (B : ℝ) : ℝ :=
  (cos (π - B) = -1 / 2)

theorem find_angles_and_sides (A B : ℝ) (a b c : ℝ)
  (h1 : sin_value A)
  (h2 : cos_value B)
  (h3 : a = 5) :
  (sin A = 5 * sqrt 3 / 14) ∧ (B = π / 3) ∧ (b = 7) ∧ (c = 8) :=
by {
  sorry
}

end find_angles_and_sides_l448_448738


namespace product_of_positive_real_solutions_l448_448382

theorem product_of_positive_real_solutions (x : ℂ) (h : x^8 = -256) :
  (∀ x, (realPart x > 0) → x^8 = -256 → (∏ s in {x | (realPart s) > 0}, s) = 4 := sorry

end product_of_positive_real_solutions_l448_448382


namespace geometric_sequence_value_l448_448741

theorem geometric_sequence_value (a : ℕ → ℝ) (h : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n+2) = a (n+1) * (a (n+1) / a n)) :
  a 3 * a 5 = 4 → a 4 = 2 :=
by
  sorry

end geometric_sequence_value_l448_448741


namespace hyperbola_eccentricity_l448_448643

theorem hyperbola_eccentricity (a b c e : ℝ) (h_positives : 0 < a ∧ 0 < b)
  (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_asymptote_perpendicular : b / a = 2)
  (h_c : c = Real.sqrt(a^2 + b^2))
  (h_eccentricity : e = c / a) :
  e = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l448_448643


namespace min_shares_condition_a_min_shares_condition_b_l448_448556

-- Define the problem for the first condition
theorem min_shares_condition_a (shares : Fin 20 → ℕ) 
  (h_any_three : ∀ i j k, (Fin 20) → shares i + shares j + shares k > 1000) : 
  ∑ i, shares i ≥ 6679 := 
sorry

-- Define the problem for the second condition
theorem min_shares_condition_b (shares : Fin 20 → ℕ) 
  (h_any_three_consecutive : ∀ i : Fin 20, shares i + shares ((i + 1) % 20) + shares ((i + 2) % 20) > 1000) : 
  ∑ i, shares i ≥ 6674 := 
sorry

end min_shares_condition_a_min_shares_condition_b_l448_448556


namespace wall_height_l448_448557

def brick_length_m : ℝ := 0.20  -- converted from 20 cm to meters
def brick_width_m  : ℝ := 0.10  -- converted from 10 cm to meters
def brick_height_m : ℝ := 0.075 -- converted from 7.5 cm to meters
def wall_length_m : ℝ := 25     -- wall length in meters
def wall_width_m  : ℝ := 0.75   -- wall width in meters
def num_bricks : ℕ := 25000     -- number of bricks

def brick_volume : ℝ := brick_length_m * brick_width_m * brick_height_m

def total_brick_volume : ℝ := num_bricks * brick_volume

theorem wall_height
  (brick_length_m : ℝ := 0.20)
  (brick_width_m  : ℝ := 0.10)
  (brick_height_m : ℝ := 0.075)
  (wall_length_m : ℝ := 25)
  (wall_width_m  : ℝ := 0.75)
  (num_bricks : ℕ := 25000) :
  (25 * 0.75 * h) = 25000 * (0.20 * 0.10 * 0.075) → h = 2 :=
by
  intros h h_eq
  have brick_vol := 0.20 * 0.10 * 0.075
  have total_vol := 25000 * brick_vol
  rw total_vol at h_eq
  have wall_vol := 25 * 0.75 * h
  rw wall_vol at h_eq
  sorry

end wall_height_l448_448557


namespace total_handshakes_l448_448998

theorem total_handshakes (teamA teamB : Fin 6) (referees : Fin 3) : 
  let players := 12 in
  (teamA * teamB) + (players * referees) = 72 :=
by
  let handshakes_teams := 36
  let handshakes_referees := 36
  have sum_handshakes := handshakes_teams + handshakes_referees
  show sum_handshakes = 72
  rfl

end total_handshakes_l448_448998


namespace probability_of_cutting_green_wire_l448_448615

-- Define the initial condition
def initial_wire : String := "green"

-- Define the probability of switching wire at any time t
def switch_probability (t : ℕ) (h : 1 ≤ t ∧ t ≤ 15): ℝ := 1 / (2 * t^2)

-- Define the probability of not switching wire at time t
def no_switch_probability (t : ℕ) (h : 1 ≤ t ∧ t ≤ 15) : ℝ := 1 - switch_probability t h

-- Define the product of no-switching probabilities for the countdown from t = 15 to t = 2
def no_switch_product : ℝ :=
  ∏ t in (finset.range 14).map (λ n, n + 2), no_switch_probability t (by simp [lt_add_of_pos_right n zero_lt_two, add_lt_iff_neg_right])

-- Define the polynomial P(x) and the values P(1) and P(-1)
def P (x : ℝ) : ℝ :=
  ∏ t in (finset.range 14).map (λ n, n + 2), switch_probability t (by simp [lt_add_of_pos_right n zero_lt_two, add_lt_iff_neg_right]) + 
  (1 - switch_probability t (by simp [lt_add_of_pos_right n zero_lt_two, add_lt_iff_neg_right])) * x

def P_one := P 1
def P_neg_one := P (-1)

-- Final statement to prove
theorem probability_of_cutting_green_wire : 
  ((1 + P_neg_one) / 2 = 23 / 30) := 
sorry

end probability_of_cutting_green_wire_l448_448615


namespace treadmill_discount_percentage_l448_448451

theorem treadmill_discount_percentage
  (p_t : ℝ) -- original price of the treadmill
  (t_p : ℝ) -- total amount paid for treadmill and plates
  (p_plate : ℝ) -- price of each plate
  (n_plate : ℕ) -- number of plates
  (h_t : p_t = 1350)
  (h_tp : t_p = 1045)
  (h_p_plate : p_plate = 50)
  (h_n_plate : n_plate = 2) :
  ((p_t - (t_p - n_plate * p_plate)) / p_t) * 100 = 30 :=
by
  sorry

end treadmill_discount_percentage_l448_448451


namespace math_competition_l448_448930

def rank_students (rank: ℕ → ℕ) :=
  (rank 1 = 1)  ∧                               -- Ding in the 1st position
  (rank 2 = 2)  ∧                               -- Jia in the 2nd position
  (rank 3 = 3)  ∧                               -- Yi in the 3rd position
  (rank 4 = 4)  ∧                               -- Wu in the 4th position
  (rank 5 = 5)                                  -- Bing in the 5th position

axiom jia_condition (rank : ℕ → ℕ) :
  rank 2 > rank 1 ∧ rank 2 > rank 3

axiom yi_condition (rank : ℕ → ℕ) :
  (rank 2 = rank 1 + 1 ∧ rank 4 = rank 3 + 1) ∨
  (rank 1 = rank 2 + 1 ∧ rank 3 = rank 4 + 1)

axiom bing_condition (rank : ℕ → ℕ) :
  ∀ i, 2 ≤ i → rank 5 < rank i

axiom ding_condition (rank : ℕ → ℕ) :
  ∀ i, 1 ≤ i ∧ i < 5 → rank 1 > rank i

axiom wu_condition (rank : ℕ → ℕ) :
  rank 5 = 4

theorem math_competition : 
  ∃ rank: ℕ → ℕ, 
    rank_students rank ∧ 
    jia_condition rank ∧ 
    yi_condition rank ∧ 
    bing_condition rank ∧ 
    ding_condition rank ∧ 
    wu_condition rank :=
sorry

end math_competition_l448_448930


namespace intersection_complements_l448_448358

open Set

variable (U : Set (ℝ × ℝ))
variable (M : Set (ℝ × ℝ))
variable (N : Set (ℝ × ℝ))

noncomputable def complementU (A : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := U \ A

theorem intersection_complements :
  let U := {p : ℝ × ℝ | True}
  let M := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y + 2 = x - 2 ∧ x ≠ 2)}
  let N := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y ≠ x - 4)}
  ((complementU U M) ∩ (complementU U N)) = {(2, -2)} :=
by
  let U := {(x, y) : ℝ × ℝ | True}
  let M := {(x, y) : ℝ × ℝ | (y + 2) = (x - 2) ∧ x ≠ 2}
  let N := {(x, y) : ℝ × ℝ | y ≠ (x - 4)}
  have complement_M := U \ M
  have complement_N := U \ N
  sorry

end intersection_complements_l448_448358


namespace cross_sectional_area_of_cone_l448_448560

theorem cross_sectional_area_of_cone 
  (r : ℝ) 
  (h : ℝ)
  (r = 2) 
  (height_midpoint_plane : ∀ (x : ℝ), x = h / 2 ∧ x ∈ set.Ioo 0 h) 
  : real.pi = real.pi * (1 ^ 2) :=
by
  sorry

end cross_sectional_area_of_cone_l448_448560


namespace y_work_days_24_l448_448548

-- Definitions of the conditions
def x_work_days := 36
def y_work_days (d : ℕ) := d
def y_worked_days := 12
def x_remaining_work_days := 18

-- Statement of the theorem
theorem y_work_days_24 : ∃ d : ℕ, (y_worked_days / y_work_days d + x_remaining_work_days / x_work_days = 1) ∧ d = 24 :=
  sorry

end y_work_days_24_l448_448548


namespace complex_minimum_distance_l448_448792

theorem complex_minimum_distance (z : ℂ) (h : |z - 8| + |z + 3 * complex.I| = 15) : |z| = 8 / 5 :=
sorry

end complex_minimum_distance_l448_448792


namespace abc_solution_l448_448732

theorem abc_solution (a b c : ℕ) (h1 : a + b = c - 1) (h2 : a^3 + b^3 = c^2 - 1) : 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 3 ∧ b = 2 ∧ c = 6) :=
sorry

end abc_solution_l448_448732


namespace min_a2_plus_b2_l448_448192

-- Define circle and line intercept conditions
def circle_center : ℝ × ℝ := (-2, 1)
def circle_radius : ℝ := 2
def line_eq (a b x y : ℝ) : Prop := a * x + 2 * b * y - 4 = 0
def chord_length (chord_len : ℝ) : Prop := chord_len = 4

-- Define the final minimum value to prove
def min_value (a b : ℝ) : ℝ := a^2 + b^2

-- Proving the specific value considering the conditions
theorem min_a2_plus_b2 (a b : ℝ) (h1 : b = a + 2) (h2 : chord_length 4) : min_value a b = 2 := by
  sorry

end min_a2_plus_b2_l448_448192


namespace car_price_l448_448825

theorem car_price (loan_years : ℕ) (monthly_payment down_payment : ℕ) :
  loan_years = 5 → monthly_payment = 250 → down_payment = 5000 →
  let months := loan_years * 12 in
  let total_loan_payment := months * monthly_payment in
  let car_price := total_loan_payment + down_payment in
  car_price = 20000 :=
begin
  intros h1 h2 h3,
  sorry
end

end car_price_l448_448825


namespace prime_dates_in_2023_l448_448255

/-- The conditions for the problem -/
def prime_months : List ℕ := [2, 3, 5, 7, 11]

def prime_days : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def last_digit_is_prime : ℕ := 3

/-- Function to count the number of prime dates in a given month -/
def count_prime_dates_in_month (days_in_month : List ℕ) : ℕ :=
  (days_in_month.filter (λd, prime_days.contains d)).length

/-- The main theorem stating the number of prime dates in 2023 -/
theorem prime_dates_in_2023 : ∑ month in prime_months, 
  if month = 2 then count_prime_dates_in_month [2, 3, 5, 7, 11, 13, 17, 19, 23, 28] else
  if month = 11 then count_prime_dates_in_month [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 30] else
  count_prime_dates_in_month [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] = 52 := 
by
  sorry

end prime_dates_in_2023_l448_448255


namespace perpendicular_DE_EF_l448_448184

-- Define points of the regular pentagon ABCDE
variables {r : ℝ}
def A : ℝ × ℝ := (r, 0)
def B : ℝ × ℝ := (r * real.cos (2 * real.pi / 5), r * real.sin (2 * real.pi / 5))
def C : ℝ × ℝ := (r * real.cos (4 * real.pi / 5), r * real.sin (4 * real.pi / 5))
def D : ℝ × ℝ := (r * real.cos (4 * real.pi / 5), -r * real.sin (4 * real.pi / 5))
def E : ℝ × ℝ := (r * real.cos (2 * real.pi / 5), -r * real.sin (2 * real.pi / 5))

-- Define the circle c with center A passing through B and E
def circle_c (x y : ℝ) : Prop := (x - r) ^ 2 + y ^ 2 = r ^ 2

-- Define point F as the second intersection of line BC with the circle
def line_BC_param (t : ℝ) : ℝ × ℝ :=
  (r * real.cos (2 * real.pi / 5) + t * (real.cos (4 * real.pi / 5) - real.cos (2 * real.pi / 5)),
   r * real.sin (2 * real.pi / 5) + t * (real.sin (4 * real.pi / 5) - real.sin (2 * real.pi / 5)))

axiom F : ℝ × ℝ 
example (_ : circle_c F.1 F.2) (_ : F ≠ B) : Prop := sorry

theorem perpendicular_DE_EF :
  let m_DE := (E.2 - D.2) / (E.1 - D.1) in
  let m_EF := (F.2 - E.2) / (F.1 - E.1) in
  m_DE * m_EF = -1 :=
sorry -- Proof goes here

end perpendicular_DE_EF_l448_448184


namespace num_valid_n_l448_448785

theorem num_valid_n (n q r : ℤ) (h₁ : 10000 ≤ n) (h₂ : n ≤ 99999)
  (h₃ : n = 50 * q + r) (h₄ : 200 ≤ q) (h₅ : q ≤ 1999)
  (h₆ : 0 ≤ r) (h₇ : r < 50) :
  (∃ (count : ℤ), count = 14400) := by
  sorry

end num_valid_n_l448_448785


namespace length_EQ_4sqrt7_l448_448401

noncomputable def length_of_EF 
  (AB BC : ℝ) 
  (DE DF : ℝ) 
  (area_triangle_DEF : ℝ) 
  (h_congruent : DE = DF)
  (h_area_triangle : area_triangle_DEF = 28) 
  (h_area_rectangle : AB * BC = 140) 
  : ℝ := 
  let area_DEF := (DE * DF) / 2
  have h1 : area_DEF = 28 := h_area_triangle,
  have h2 : DE * DF = 56 := by 
    rw [h_congruent, mul_self_eq (area_DEF * 2)],
  have h3 : ∃ (EF : ℝ), EF^2 = 112 := by 
    use (sqrt 112), 
  4 * sqrt 7 

theorem length_EQ_4sqrt7 
  (AB BC DE DF area_triangle_DEF : ℝ)
  (h_congruent : DE = DF)
  (h_area_triangle : area_triangle_DEF = 28)
  (h_area_rectangle : AB * BC = 140) 
  : length_of_EF AB BC DE DF area_triangle_DEF h_congruent h_area_triangle h_area_rectangle = 4 * sqrt 7 := 
  sorry

end length_EQ_4sqrt7_l448_448401


namespace probability_one_nonconforming_is_0_14_l448_448131

-- Defining the pass rates for products A and B
def pass_rate_A := 0.90
def pass_rate_B := 0.95

-- Calculating the desired probability
def probability_exactly_one_nonconforming : ℝ :=
  let nonconforming_A := 1 - pass_rate_A
  let conforming_B := pass_rate_B
  let conforming_A := pass_rate_A
  let nonconforming_B := 1 - pass_rate_B
  (nonconforming_A * conforming_B) + (conforming_A * nonconforming_B)

-- The proof statement
theorem probability_one_nonconforming_is_0_14 :
  probability_exactly_one_nonconforming = 0.14 :=
sorry

end probability_one_nonconforming_is_0_14_l448_448131


namespace f_is_increasing_l448_448214

noncomputable theory

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

open Set

theorem f_is_increasing (f : ℝ → ℝ) (I : set ℝ) (h : ∀ x ∈ I, ∃ y ∈ I, x < y) : is_monotonically_increasing f I := sorry

example : is_monotonically_increasing f (Icc 0 (Real.pi / 2)) :=
begin
  unfold is_monotonically_increasing,
  intros x y hx hy hxy,
  have : 7 * Real.cos (x - Real.pi / 6) > 0 := sorry,
  have : 7 * Real.cos (y - Real.pi / 6) > 0 := sorry,
  calc
    f x = 7 * Real.sin (x - Real.pi / 6) : rfl
    ... < 7 * Real.sin (y - Real.pi / 6) : sorry
    ... = f y : rfl,
end

end f_is_increasing_l448_448214


namespace find_an_given_recurrence_l448_448668

theorem find_an_given_recurrence (a b : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = (1 / 16) * (1 + 4 * a n + real.sqrt (1 + 24 * a n))) :
  ∀ n, a n = ((2 : ℝ)^(2 - n) + 3)^2 - 1 / 24 :=
sorry

end find_an_given_recurrence_l448_448668


namespace difference_between_q_and_r_l448_448989

-- Define the variables for shares with respect to the common multiple x
def p_share (x : Nat) : Nat := 3 * x
def q_share (x : Nat) : Nat := 7 * x
def r_share (x : Nat) : Nat := 12 * x

-- Given condition: The difference between q's share and p's share is Rs. 4000
def condition_1 (x : Nat) : Prop := (q_share x - p_share x = 4000)

-- Define the theorem to prove the difference between r and q's share is Rs. 5000
theorem difference_between_q_and_r (x : Nat) (h : condition_1 x) : r_share x - q_share x = 5000 :=
by
  sorry

end difference_between_q_and_r_l448_448989


namespace number_approx_l448_448553

theorem number_approx (x : ℝ) : 
  ((258 / 100) * 1265) / x = 543.95 → x ≈ 6.005 := 
by
  sorry

end number_approx_l448_448553


namespace machine_no_repair_l448_448504

def nominal_mass (G_dev: ℝ) := G_dev / 0.1

theorem machine_no_repair (G_dev: ℝ) (σ: ℝ) (non_readable_dev_lt: ∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) : 
  (G_dev = 39) ∧ (σ ≤ G_dev) ∧ (∀ (x: ℝ), x ∉ { measurements } → |x - nominal_mass(G_dev)| < G_dev) ∧ (G_dev ≤ 0.1 * nominal_mass(G_dev)) → 
  ¬ machine.requires_repair :=
by
  sorry

end machine_no_repair_l448_448504


namespace number_of_sheets_in_stack_l448_448965

theorem number_of_sheets_in_stack (n : ℕ) (h1 : 2 * n + 2 = 74) : n / 4 = 9 := 
by
  sorry

end number_of_sheets_in_stack_l448_448965


namespace min_max_expression_l448_448297

theorem min_max_expression (x : ℝ) (h : 2 ≤ x ∧ x ≤ 7) :
  ∃ (a : ℝ) (b : ℝ), a = 11 / 3 ∧ b = 87 / 16 ∧ 
  (∀ y, 2 ≤ y ∧ y ≤ 7 → 11 / 3 ≤ (y^2 + 4*y + 10) / (2*y + 2)) ∧
  (∀ y, 2 ≤ y ∧ y ≤ 7 → (y^2 + 4*y + 10) / (2*y + 2) ≤ 87 / 16) :=
sorry

end min_max_expression_l448_448297


namespace find_angle_A_and_triangle_perimeter_l448_448739

-- Declare the main theorem using the provided conditions and the desired results
theorem find_angle_A_and_triangle_perimeter
  (a b c : ℝ) (A B : ℝ)
  (h1 : 0 < A ∧ A < Real.pi)
  (h2 : (Real.sqrt 3) * b * c * (Real.cos A) = a * (Real.sin B))
  (h3 : a = Real.sqrt 2)
  (h4 : (c / a) = (Real.sin A / Real.sin B)) :
  (A = Real.pi / 3) ∧ (a + b + c = 3 * Real.sqrt 2) :=
  sorry -- Proof is left as an exercise

end find_angle_A_and_triangle_perimeter_l448_448739


namespace areas_of_geometric_figures_with_equal_perimeter_l448_448328

theorem areas_of_geometric_figures_with_equal_perimeter (l : ℝ) (h : (l > 0)) :
  let s1 := l^2 / (4 * Real.pi)
  let s2 := l^2 / 16
  let s3 := (Real.sqrt 3) * l^2 / 36
  s1 > s2 ∧ s2 > s3 := by
  sorry

end areas_of_geometric_figures_with_equal_perimeter_l448_448328


namespace find_a_l448_448722

theorem find_a {a b c : ℕ} (h₁ : a + b = c) (h₂ : b + c = 8) (h₃ : c = 4) : a = 0 := by
  sorry

end find_a_l448_448722


namespace machine_no_repair_needed_l448_448496

theorem machine_no_repair_needed (M : ℕ) (σ : ℕ) (greatest_deviation : ℕ) 
                                  (nominal_weight : ℕ)
                                  (h1 : greatest_deviation = 39)
                                  (h2 : greatest_deviation ≤ (0.1 * nominal_weight))
                                  (h3 : ∀ d, d < 39) : 
                                  σ ≤ greatest_deviation :=
by
  sorry

end machine_no_repair_needed_l448_448496


namespace opposite_of_113_is_114_l448_448047

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l448_448047


namespace arccos_solution_l448_448833

theorem arccos_solution (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) (hx : arccos (3 * x) - arccos x = π / 6) :
  x = 1 / Real.sqrt (40 - 12 * Real.sqrt 3) ∨ x = -1 / Real.sqrt (40 - 12 * Real.sqrt 3) := sorry

end arccos_solution_l448_448833


namespace train_length_correct_l448_448206

-- Conditions
def speed_of_train : ℝ := 63 -- in km/hr
def speed_of_man : ℝ := 3 -- in km/hr
def time_to_cross : ℝ := 14.998800095992321 -- in seconds

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s : ℝ := 5 / 18

-- Relative speed in m/s
def relative_speed : ℝ := (speed_of_train - speed_of_man) * km_per_hr_to_m_per_s

-- Length of the train (we expect it to be 250 meters)
def length_of_train (relative_speed : ℝ) (time_to_cross : ℝ) : ℝ :=
  relative_speed * time_to_cross

-- Proof statement
theorem train_length_correct :
  length_of_train relative_speed time_to_cross = 250 :=
sorry

end train_length_correct_l448_448206


namespace find_z_find_m_l448_448437

noncomputable def complex_number_z (a b : ℝ) : ℂ := a + b * complex.I

constant z : ℂ
constant m : ℝ

axiom pos_real_part : z.re > 0
axiom modulus_z : complex.abs z = real.sqrt 10
axiom angle_bisector_condition : (complex_number_z (1 - 2 * complex.I.re) (2 * complex.I.re + complex.I.im).im) ∈ 
       complex.arg(z * (1 + 2 * complex.I)) = π / 2
axiom purely_imaginary_condition : (conj(z) + ((m - complex.I) / (1 + complex.I))).re = 0
     
theorem find_z : z = 3 - complex.I := sorry

theorem find_m : m = -5 := sorry

end find_z_find_m_l448_448437


namespace cube_coloring_l448_448620

-- Definitions based on the given problem conditions
def colors : Nat := 4
def faces : Nat := 6

-- The main theorem statement needing a proof
theorem cube_coloring (C : Nat) (F : Nat) (H_colors : C = 4) (H_faces : F = 6) (H_adj_not_same : ∀ (c₁ c₂ : ℕ), c₁ ≠ c₂ → True): 
    ∃ N : Nat, N = 96 :=
by 
  use 96
  sorry -- Proof omitted as per instructions

end cube_coloring_l448_448620


namespace parabola_equation_l448_448950

theorem parabola_equation :
  ∃ a b c d e f : ℤ,
    (a > 0) ∧
    (Int.gcd (Nat.abs a) (Nat.gcd (Nat.abs b)
      (Nat.gcd (Nat.abs c) (Nat.gcd (Nat.abs d)
      (Nat.gcd (Nat.abs e) (Nat.abs f))))) = 1) ∧
    (a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0) ∧
    -- Parabola with focus and directrix
    (let focus := (4 : ℤ, -2 : ℤ) in
     let directrix := (4*x + 5*y - 20 : ℤ) in
     True) := -- This is a placeholder for the actual geometrical constraint
  (∃ x y : ℤ, (25*x^2 + 16*y^2 - 168*x + 364*y + 260 = 0))
by {
  sorry
}

end parabola_equation_l448_448950


namespace max_value_of_a_exists_max_value_of_a_l448_448355

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  a ≤ (Real.sqrt 6 / 3) :=
sorry

theorem exists_max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ∃ a_max: ℝ, a_max = (Real.sqrt 6 / 3) ∧ (∀ a', (a' ≤ a_max)) :=
sorry

end max_value_of_a_exists_max_value_of_a_l448_448355


namespace greatest_gcd_f_l448_448657

def f (n : ℕ) : ℕ := 70 + n^2

def g (n : ℕ) : ℕ := Nat.gcd (f n) (f (n + 1))

theorem greatest_gcd_f (n : ℕ) (h : 0 < n) : g n = 281 :=
  sorry

end greatest_gcd_f_l448_448657


namespace arithmetic_sequence_fraction_l448_448682

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_pos : ∀ n, a n > 0)
variable (h_arith_seq : a 3 = a 1 + 2 * a 2)
variable (h_geom_seq : ∀ n, a (n + 1) = a n * q)

theorem arithmetic_sequence_fraction :
  (a 8 + a 9) / (a 7 + a 8) = sqrt 2 + 1 := by
  sorry

end arithmetic_sequence_fraction_l448_448682


namespace binary_division_remainder_correct_l448_448155

-- Define the last two digits of the binary number
def b_1 : ℕ := 1
def b_0 : ℕ := 1

-- Define the function to calculate the remainder when dividing by 4
def binary_remainder (b1 b0 : ℕ) : ℕ := 2 * b1 + b0

-- Expected remainder in binary form
def remainder_in_binary : ℕ := 0b11  -- '11' in binary is 3 in decimal

-- The theorem to prove
theorem binary_division_remainder_correct :
  binary_remainder b_1 b_0 = remainder_in_binary :=
by
  -- Proof goes here
  sorry

end binary_division_remainder_correct_l448_448155


namespace distinct_products_count_l448_448368

theorem distinct_products_count : 
  let s := {2, 3, 5, 7, 11}
  in (finset.powerset s).filter (λ t, 2 ≤ t.card).image (λ t, t.prod).card = 26 :=
by
  sorry

end distinct_products_count_l448_448368


namespace circle_equation_standard_form_l448_448105

/-- The equation x^2 + y^2 + 2x - 4y - 6 = 0 represents a circle with center at (-1,2) and radius sqrt(11). -/
theorem circle_equation_standard_form :
  ∃ (h k : ℝ) (r : ℝ), (∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 11 ↔ x^2 + y^2 + 2*x - 4*y - 6 = 0)
 ∧ h = -1
 ∧ k = 2
 ∧ r = real.sqrt 11 :=
sorry

end circle_equation_standard_form_l448_448105


namespace find_stone_counted_as_99_l448_448658

theorem find_stone_counted_as_99 :
  ∃ n : ℕ, n ≡ 99 [MOD 27] ∧ considerable_stone(n) = 10 :=
by
  sorry

end find_stone_counted_as_99_l448_448658


namespace trigonometric_quadrant_l448_448554

-- Define the conditions as assumptions
variables {θ : ℝ}

-- The proof problem statement in Lean 4
theorem trigonometric_quadrant (h1 : sin (π - θ) < 0) (h2 : tan (π + θ) > 0) : 
  ϑ ≥ π ∧ ϑ < 3 * π / 2 :=
sorry

end trigonometric_quadrant_l448_448554


namespace number_of_pieces_l448_448718

def length_piece : ℝ := 0.40
def total_length : ℝ := 47.5

theorem number_of_pieces : ⌊total_length / length_piece⌋ = 118 := by
  sorry

end number_of_pieces_l448_448718


namespace train_crossing_time_l448_448593

-- Definitions of given constants
def speed_kmh : ℝ := 60 -- Speed of train in km/hr
def length_m : ℝ := 200 -- Length of train in meters

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Speed in m/s
def speed_ms : ℝ := speed_kmh * conversion_factor

-- Time calculation
def time_sec : ℝ := length_m / speed_ms

-- The proof statement
theorem train_crossing_time : time_sec = 12 :=
by
  -- The proof will be provided here
  sorry

end train_crossing_time_l448_448593


namespace problem1_problem2_l448_448344

theorem problem1 (l : ℝ) (l1 : 2 * l.1 + l.2 - 5 = 0) (l2 : l.1 - 2 * l.2 = 0) (p : Prop) :
  (∃ c, l.1 - l.2 + c = 0 ∧ l.1 = 2 ∧ l.2 = 1) → l.1 - l.2 - 1 = 0 := by
  sorry

theorem problem2 (l : ℝ) (l1 : 2 * l.1 + l.2 - 5 = 0) (l2 : l.1 - 2 * l.2 = 0) :
  (∃ a b, l.1 / a + l.2 / b = 1 ∧ 1 / 2 * a * b = 4 ∧ l.1 = 2 ∧ l.2 = 1) → l.1 + 2 * l.2 - 4 = 0 := by
  sorry

end problem1_problem2_l448_448344


namespace abs_inequality_solution_l448_448634

theorem abs_inequality_solution (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end abs_inequality_solution_l448_448634


namespace total_time_for_week_l448_448207

namespace SchoolTime

-- Definitions for each day's time based on conditions
def Monday_time : ℝ :=
  let lessons_1_hour := 4 * 1 -- 4 lessons of 1 hour each
  let lessons_30_minutes := 2 * 0.5 -- 2 lessons of 30 minutes each
  lessons_1_hour + lessons_30_minutes 

def Tuesday_time : ℝ :=
  let lessons_1_hour := 3 * 1 -- 3 lessons of 1 hour each
  let lesson_1_5_hour := 1.5 -- 1 lesson of 1.5 hours
  let lesson_45_minutes := 0.75 -- 1 lesson of 45 minutes (0.75 hours)
  lessons_1_hour + lesson_1_5_hour + lesson_45_minutes

def Wednesday_time : ℝ := 2 * Tuesday_time

def Thursday_time : ℝ :=
  let mathematics := 3.5
  let science := 2
  let history := 1
  let another_subject := 0.5 -- 30 minutes
  mathematics + science + history + another_subject

def Friday_time : ℝ := Wednesday_time / 2

-- Prove the total time for the week
theorem total_time_for_week : 
  Monday_time + Tuesday_time + Wednesday_time + Thursday_time + Friday_time = 33 := sorry

end SchoolTime

end total_time_for_week_l448_448207


namespace monotonic_interval_l448_448242

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin(x - Real.pi / 6)

def is_monotonically_increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem monotonic_interval : is_monotonically_increasing_interval f 0 (Real.pi / 2) :=
sorry

end monotonic_interval_l448_448242


namespace ratio_of_segments_l448_448926

-- Definition of points and conditions
variables {A B C D X Y O : Type*}
variables (O_center : inscribed_circle_center O A B C D)
variables (L : line {passes_through O intersecting X on AB and Y on CD})
variables (angle_equality : ∠ AXY = ∠ DYX)

-- Theorem statement
theorem ratio_of_segments (AX BX CY DY : Type*) :
  AX / BX = CY / DY :=
  sorry

end ratio_of_segments_l448_448926


namespace game_necessarily_ends_alice_wins_l448_448933

-- Define the game state and moves
def card_state := ℕ → bool -- A function mapping card positions to boolean values indicating red (true) or white (false) side up
def is_valid_move (state : card_state) (start : ℕ) : Prop :=
  start + 49 < 2023 ∧ state start -- The start position must be such that all 50 cards are within the 2023 cards and the start card is red

def flip_block_50 (state : card_state) (start : ℕ) : card_state :=
  λ i, if start ≤ i ∧ i < start + 50 then bnot (state i) else state i -- Flip the sides of the 50 consecutive cards

-- Condition a: Prove the game necessarily ends
theorem game_necessarily_ends (initial_state : card_state) (h : ∀ i, initial_state i = tt) : ∃ n, ¬∃ start, is_valid_move (iterate (flip_block_50) n initial_state) start := 
sorry

-- Condition b: Prove Alice has a winning strategy
theorem alice_wins (initial_state : card_state) (h : ∀ i, initial_state i = tt) : ∃ f : Π (state : card_state) (is_alice_turn : bool), card_state, 
  (∀ state, (∃ start, is_valid_move state start) → ∃ start, is_valid_move (f state tt) start) ∧ -- Alice can always make a move
  (∀ state, ¬(∃ start, is_valid_move state start) → ¬(∃ start, is_valid_move (f state tt) start)) -- If Bob can't make a move, Alice wins
  :=
sorry

end game_necessarily_ends_alice_wins_l448_448933


namespace boxes_difference_l448_448915

theorem boxes_difference (white_balls red_balls balls_per_box : ℕ)
  (h_white : white_balls = 30)
  (h_red : red_balls = 18)
  (h_box : balls_per_box = 6) :
  (white_balls / balls_per_box) - (red_balls / balls_per_box) = 2 :=
by 
  sorry

end boxes_difference_l448_448915


namespace candies_per_block_l448_448466

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) (h1 : candies_per_house = 7) (h2 : houses_per_block = 5) :
  candies_per_house * houses_per_block = 35 :=
by 
  -- Placeholder for the formal proof
  sorry

end candies_per_block_l448_448466


namespace half_of_expression_correct_l448_448147

theorem half_of_expression_correct :
  (2^12 + 3 * 2^10) / 2 = 2^9 * 7 :=
by
  sorry

end half_of_expression_correct_l448_448147


namespace lines_parallel_to_same_line_are_parallel_l448_448085

-- Define a type for lines
constant Line : Type

-- Define the notion of parallel lines as a predicate
constant parallel : Line → Line → Prop

-- State the theorem
theorem lines_parallel_to_same_line_are_parallel 
  (A B C : Line) 
  (hAC : parallel A C) 
  (hBC : parallel B C) : 
  parallel A B :=
sorry

end lines_parallel_to_same_line_are_parallel_l448_448085


namespace sum_q_t_12_l448_448007

section

open Classical BigOperators Polynomial

noncomputable def T : Finset (Fin 12 →ₘ Fin 2) :=
  Finset.univ

def q_t (t : Fin 12 →ₘ Fin 2) : ℕ → ℚ :=
  λ x : ℕ, if h : x < 12 then t ⟨x, h⟩ else 0

noncomputable def q : ℕ → ℚ :=
  λ x : ℕ, T.sum (λ t, q_t t x)

theorem sum_q_t_12 : q 12 = 2048 :=
by sorry

end

end sum_q_t_12_l448_448007


namespace find_solutions_l448_448638

theorem find_solutions (x : ℝ) : 
  (∃ x : ℝ, (real.rpow (3 - x) (1/4) + real.sqrt (x + 2) = 2)
           ∧ ((x = 2) ∨ (abs (x - 0.990) < 0.001))) :=
  sorry

end find_solutions_l448_448638


namespace smallest_positive_x_palindrome_multiple_of_5_l448_448907

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def smallest_positive_x : ℕ :=
  1119

theorem smallest_positive_x_palindrome_multiple_of_5 (x p : ℕ) 
  (h1 : p = x + 7890)
  (h2 : is_palindrome p)
  (h3 : is_multiple_of_5 p)
  (h4 : p > 7890) : 
  smallest_positive_x = x := sorry

end smallest_positive_x_palindrome_multiple_of_5_l448_448907


namespace card_prob_queen_diamond_l448_448888

theorem card_prob_queen_diamond : 
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob in
  total_prob = 18 / 221 :=
by
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob
  sorry

end card_prob_queen_diamond_l448_448888


namespace intersection_point_l448_448269

noncomputable def line1 := λ (x y : ℝ), 2 * x + y + 2
noncomputable def line2 := λ (x y : ℝ), (-2) * x + 4 * y - 2

theorem intersection_point (x y a : ℝ) (h1 : line1 x y = 0) (h2 : line2 x y = 0) :
  a = -2 → x = -1 ∧ y = 0 := by sorry

end intersection_point_l448_448269


namespace opposite_number_113_is_13_l448_448034

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l448_448034


namespace max_percentage_change_l448_448612

theorem max_percentage_change
  (initial_yes : ℝ := 0.4) (initial_no : ℝ := 0.6)
  (final_yes : ℝ := 0.8) (final_no : ℝ := 0.2)
  (fixed_no_percentage : ℝ := 0.2) :
  (∃ max_change : ℝ, max_change = 0.4) :=
by
  let fixed_no := initial_no * fixed_no_percentage
  let changeable_no := initial_no - fixed_no
  let required_yes_increase := final_yes - initial_yes
  let max_possible_change := changeable_no + required_yes_increase
  have h : max_possible_change = 0.4,
  {
    sorry
  }
  use max_possible_change
  exact h

end max_percentage_change_l448_448612


namespace integral_of_exp_sin_l448_448286

-- Given constants α and β
variables (α β : ℝ)

-- Function to integrate
def integrand (x : ℝ) : ℝ := exp(α * x) * sin(β * x)

-- Integral of the function
noncomputable def integral := ∫ x, integrand α β x

-- Expected result of the integral
noncomputable def expected_result (x : ℝ) : ℝ := exp(α * x) * (α * sin(β * x) - β * cos(β * x)) / (α^2 + β^2) + arbitrary ℝ

-- Proof statement
theorem integral_of_exp_sin :
  integral = expected_result :=
sorry

end integral_of_exp_sin_l448_448286


namespace product_of_fractions_l448_448621

open BigOperators

theorem product_of_fractions :
  (∏ n in Finset.range 9, (n + 2)^3 - 1) / (∏ n in Finset.range 9, (n + 2)^3 + 1) = 74 / 55 :=
by
  sorry

end product_of_fractions_l448_448621


namespace least_positive_integer_l448_448648
  
theorem least_positive_integer 
  (x : ℕ) (d n : ℕ) (p : ℕ) 
  (h_eq : x = 10^p * d + n) 
  (h_ratio : n = x / 17) 
  (h_cond1 : 1 ≤ d) 
  (h_cond2 : d ≤ 9)
  (h_nonzero : n > 0) : 
  x = 10625 :=
by
  sorry

end least_positive_integer_l448_448648


namespace frog_reaches_riverbank_l448_448397

-- Definitions for stone positions
def stones : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}

-- Probability of reaching the riverbank from each stone position
noncomputable def P : ℕ → ℝ
| 0 => 0
| 6 => 1
| N => if h : 0 < N ∧ N < 6 then (N : ℝ)/6 * P (N - 1) + (1 - (N : ℝ)/6) * P (N + 1) else 0

-- Main theorem to prove the probability of starting from stone 2
theorem frog_reaches_riverbank : P 2 = 4 / 9 := by sorry

end frog_reaches_riverbank_l448_448397


namespace shift_graph_right_3_units_l448_448136

-- Definitions of the given functions
def f (x : ℝ) : ℝ := (1 / 2) ^ x
def g (x : ℝ) : ℝ := 8 * 2 ^ (-x)

-- The theorem statement
theorem shift_graph_right_3_units :
  ∃ h : ℝ → ℝ, (∀ x, h (x - 3) = f x) ∧ (h = g) :=
sorry

end shift_graph_right_3_units_l448_448136


namespace price_of_pants_l448_448016

theorem price_of_pants (S P H : ℝ) (h1 : 0.8 * S + P + H = 340) (h2 : S = (3 / 4) * P) (h3 : H = P + 10) : P = 91.67 :=
by sorry

end price_of_pants_l448_448016


namespace percentage_increase_l448_448417

theorem percentage_increase (old_earnings new_earnings: ℝ) (h_old: old_earnings = 60) (h_new: new_earnings = 84) :
  ((new_earnings - old_earnings) / old_earnings) * 100 = 40 := 
by {
  rw [h_old, h_new],
  norm_num,
  simp,
  rw div_mul_cancel _ (by norm_num),
  norm_num,
sorry
}

end percentage_increase_l448_448417


namespace notebook_distribution_l448_448171

theorem notebook_distribution (C N : ℕ) 
  (h1 : N / C = C / 8)
  (h2 : N / (C / 2) = 16) : 
  N = 512 :=
begin
  sorry
end

end notebook_distribution_l448_448171


namespace candle_height_at_half_T_l448_448191
-- Import required Lean libraries

-- Definitions based on conditions in a)
def initial_height : ℕ := 150
def T : ℕ := 169875 /- Total time calculated in the solution -/
def burn_time (k : ℕ) : ℕ := 15 * k

-- Statement to prove: given the initial height and burn rate conditions,
-- the height of the candle after T/2 seconds is 44 centimeters.
theorem candle_height_at_half_T :
  let half_T := T / 2 in
  let m := 106 in
  let k_burn_time := (7.5 * m * (m + 1)) in
  let height_after_half_T := initial_height - m in
  initial_height = 150 ∧
  (∀ k, k_burn_time = (7.5 * (k * (k + 1)))) ∧
  half_T = 84937.5 ∧ 
  k_burn_time ≤ half_T < (7.5 * ((m + 1) * (m + 2))) ∧
  height_after_half_T = 44 := 
by
  sorry

end candle_height_at_half_T_l448_448191


namespace ice_cream_sales_l448_448810

theorem ice_cream_sales : 
  let tuesday_sales := 12000
  let wednesday_sales := 2 * tuesday_sales
  let total_sales := tuesday_sales + wednesday_sales
  total_sales = 36000 := 
by 
  sorry

end ice_cream_sales_l448_448810


namespace cubes_in_fig_6_surface_area_fig_10_l448_448611

-- Define the function to calculate the number of unit cubes in Fig. n
def cubes_in_fig (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Define the function to calculate the surface area of the solid figure for Fig. n
def surface_area_fig (n : ℕ) : ℕ := 6 * n * n

-- Theorem statements
theorem cubes_in_fig_6 : cubes_in_fig 6 = 91 :=
by sorry

theorem surface_area_fig_10 : surface_area_fig 10 = 600 :=
by sorry

end cubes_in_fig_6_surface_area_fig_10_l448_448611


namespace complements_intersection_l448_448798

open Set

noncomputable def U : Set ℕ := { x | x ≤ 5 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complements_intersection :
  (U \ A) ∩ (U \ B) = {0, 5} :=
by
  sorry

end complements_intersection_l448_448798


namespace natural_number_40_times_smaller_l448_448650

-- Define the sum of the first (n-1) natural numbers
def sum_natural_numbers (n : ℕ) := (n * (n - 1)) / 2

-- Define the proof statement
theorem natural_number_40_times_smaller (n : ℕ) (h : sum_natural_numbers n = 40 * n) : n = 81 :=
by {
  -- The proof is left as an exercise
  sorry
}

end natural_number_40_times_smaller_l448_448650


namespace manager_final_price_l448_448574

noncomputable def wholesale_cost : ℝ := 200
noncomputable def retail_price : ℝ := wholesale_cost + 0.2 * wholesale_cost
noncomputable def manager_discount : ℝ := 0.1 * retail_price
noncomputable def price_after_manager_discount : ℝ := retail_price - manager_discount
noncomputable def weekend_sale_discount : ℝ := 0.1 * price_after_manager_discount
noncomputable def price_after_weekend_sale : ℝ := price_after_manager_discount - weekend_sale_discount
noncomputable def sales_tax : ℝ := 0.08 * price_after_weekend_sale
noncomputable def total_price : ℝ := price_after_weekend_sale + sales_tax

theorem manager_final_price : total_price = 209.95 := by
  sorry

end manager_final_price_l448_448574


namespace correct_distance_from_S_to_face_PQR_l448_448457

noncomputable def distance_from_S_to_face_PQR
  (P Q R S : ℝ × ℝ × ℝ)
  (h1 : P ≠ S) (h2 : Q ≠ S) (h3 : R ≠ S)
  (h4 : S.1 = 0 ∧ S.2 = 0 ∧ S.3 = 0)
  (h5 : P.1 = 15 ∧ P.2 = 0 ∧ P.3 = 0)
  (h6 : Q.1 = 0 ∧ Q.2 = 15 ∧ Q.3 = 0)
  (h7 : R.1 = 0 ∧ R.2 = 0 ∧ R.3 = 9)
  : ℝ :=
9

theorem correct_distance_from_S_to_face_PQR
  (P Q R S : ℝ × ℝ × ℝ)
  (h1 : P ≠ S) (h2 : Q ≠ S) (h3 : R ≠ S)
  (h4 : S.1 = 0 ∧ S.2 = 0 ∧ S.3 = 0)
  (h5 : P.1 = 15 ∧ P.2 = 0 ∧ P.3 = 0)
  (h6 : Q.1 = 0 ∧ Q.2 = 15 ∧ Q.3 = 0)
  (h7 : R.1 = 0 ∧ R.2 = 0 ∧ R.3 = 9)
  : distance_from_S_to_face_PQR P Q R S h1 h2 h3 h4 h5 h6 h7 = 9 :=
by
  sorry

end correct_distance_from_S_to_face_PQR_l448_448457


namespace inequality_range_l448_448705

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 :=
  sorry

end inequality_range_l448_448705


namespace chord_length_l448_448103

noncomputable def chordLengthOfCircle (c : ℝ × ℝ) (r : ℝ) (k : ℝ) : ℝ :=
  let x := c.1
  let y := c.2
  let d := abs (y - k * x) / sqrt (1 + k^2)
  2 * sqrt (r^2 - d^2)

theorem chord_length {x y r k : ℝ} (h_circle : x = 0 ∧ y = 2 ∧ r = 2)
  (h_line : k = 1) : chordLengthOfCircle (0, 2) 2 1 = 2 * sqrt 2 := by
  sorry

end chord_length_l448_448103


namespace twenty_four_fifty_seventh_digit_l448_448787

def concatIntegers (n : ℕ) : String :=
  (List.range' 1 n).foldr (λ i acc => acc ++ i.repr) ""

theorem twenty_four_fifty_seventh_digit:
  (concatIntegers 1000).toList.nth 2456 = some '5' :=
  sorry

end twenty_four_fifty_seventh_digit_l448_448787


namespace remaining_question_points_l448_448164

theorem remaining_question_points (total_points : ℕ) (total_questions : ℕ) 
  (two_point_questions : ℕ) (two_point_value : ℕ) :
  total_points = 100 →
  total_questions = 40 →
  two_point_questions = 30 →
  two_point_value = 2 →
  let remaining_points := total_points - (two_point_questions * two_point_value) in
  let remaining_questions := total_questions - two_point_questions in
  remaining_questions > 0 →
  remaining_points / remaining_questions = 4 :=
by
  intros
  sorry

end remaining_question_points_l448_448164


namespace number_of_valid_n_l448_448364

theorem number_of_valid_n :
  let non_neg_integers_n := {n | n < 120 ∧ (∃ α : ℕ, n = 2 * α + 1 ∧ (∃ m : ℕ, m = α * (α + 1) ∧ m % 4 = 0))},
      count := non_neg_integers_n.toList.length
  in count = 30 :=
by
  sorry

end number_of_valid_n_l448_448364


namespace derivative_at_two_l448_448666

noncomputable def f (x : ℝ) (f_prime_2 : ℝ) : ℝ := x^2 + 3 * f_prime_2 * x

theorem derivative_at_two (f_prime_2 : ℝ) : 
  (derivative (fun x => f x f_prime_2)) 2 = -2 :=
by
  have h : ∀ x, (derivative (fun x => f x f_prime_2)) x = 2 * x + 3 * f_prime_2
  { intro x
    rw [f, derivative_polynomial]
    sorry }
  rw [h]
  suffices : 2 * 2 + 3 * f_prime_2 = -2
  { assumption }
  sorry

end derivative_at_two_l448_448666


namespace min_sum_xy_l448_448332

theorem min_sum_xy (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_hy : x ≠ y)
  (h_eq : (1 / x : ℝ) + (1 / y) = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l448_448332


namespace sum_of_perimeters_l448_448510

open Nat

theorem sum_of_perimeters (X Y Z : ℕ) (hX : Prime X) (hXY : Prime (X + Y)) (hZ : Prime Z) : 
  X + (X + Y) + Z ≥ 20 := 
sorry

end sum_of_perimeters_l448_448510


namespace rod_quadrilateral_count_l448_448418

/-- Given rods of lengths ranging from 1 to 50, with excluded rods of lengths 4 cm, 9 cm, and 22 cm,
    determine how many rods can be chosen as the fourth rod to form a quadrilateral with positive
    area with the three excluded rods. --/
theorem rod_quadrilateral_count : 
  let excluded_rods := [4, 9, 22] in
  let valid_rods := [ d | d in (List.range' 1 50), 9 < d ∧ d < 35 ∧ d ∉ excluded_rods ] in
  valid_rods.length = 23 := 
sorry

end rod_quadrilateral_count_l448_448418


namespace find_abcdef_l448_448097

def repeating_decimal_to_fraction_abcd (a b c d : ℕ) : ℚ :=
  (1000 * a + 100 * b + 10 * c + d) / 9999

def repeating_decimal_to_fraction_abcdef (a b c d e f : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) / 999999

theorem find_abcdef :
  ∀ a b c d e f : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  (repeating_decimal_to_fraction_abcd a b c d + repeating_decimal_to_fraction_abcdef a b c d e f = 49 / 999) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 490) :=
by
  repeat {sorry}

end find_abcdef_l448_448097


namespace distance_from_D_to_AB_l448_448753

variables {A B C D E : Type} 
          (area_ADE area_BCE area_CDE length_AB : ℝ)

def quadrilateral_ABCD (A B C D E : Type) : Prop := 
  -- Dummy definition, replace with actual geometrical construct if necessary
  true

noncomputable def distance_from_point_to_line (D : Type) (line_AB : Type) : ℝ := 
  -- Dummy definition, replace with actual calculation if necessary
  sorry

theorem distance_from_D_to_AB
  (A B C D E : Type)
  (h1 : quadrilateral_ABCD A B C D E)
  (h2 : area_ADE = 12)
  (h3 : area_BCE = 45)
  (h4 : area_CDE = 18)
  (h5 : length_AB = 7) : 
  distance_from_point_to_line D (A, B) = 12 := 
sorry

end distance_from_D_to_AB_l448_448753


namespace moving_circle_trajectory_is_ellipse_l448_448193

noncomputable def trajectory_of_center (x y : ℝ) : Prop :=
  let ellipse_eq := x^2 / 4 + y^2 / 3 = 1 
  ellipse_eq ∧ x ≠ -2

theorem moving_circle_trajectory_is_ellipse
  (M_1 M_2 center : ℝ × ℝ)
  (r1 r2 R : ℝ)
  (h1 : M_1 = (-1, 0))
  (h2 : M_2 = (1, 0))
  (h3 : r1 = 1)
  (h4 : r2 = 3)
  (h5 : (center.1 + 1)^2 + center.2^2 = (1 + R)^2)
  (h6 : (center.1 - 1)^2 + center.2^2 = (3 - R)^2) :
  trajectory_of_center center.1 center.2 :=
by sorry

end moving_circle_trajectory_is_ellipse_l448_448193


namespace ascending_order_proof_l448_448608

-- Definitions of the conversion factors
def cm_to_m (cm: ℕ) : ℕ := cm / 100
def dm_to_m (dm: ℕ) : ℕ := dm / 10
def mm_to_m (mm: ℕ) : ℚ := mm / 1000
def km_to_m (km: ℕ) : ℕ := km * 1000

theorem ascending_order_proof : 
  let a := mm_to_m 900,
      b := 2,
      c := cm_to_m 300,
      d := dm_to_m 80,
      e := km_to_m 1 in
  a < b ∧ b < c ∧ c < d ∧ d < e :=
by 
  sorry

end ascending_order_proof_l448_448608


namespace f_2006_eq_1_l448_448489

noncomputable def f : ℤ → ℤ := sorry
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3 : ∀ x : ℤ, f (3 * (x + 1)) = f (3 * x + 1)
axiom f_at_1 : f 1 = -1

theorem f_2006_eq_1 : f 2006 = 1 := by
  sorry

end f_2006_eq_1_l448_448489


namespace sum_palindromic_primes_lt_70_l448_448815

def reverseDigits (n : ℕ) : ℕ :=
  -- Convert number to string, reverse it, and convert back to number
  ( (n.digits 10).reverse).foldl (λ x d, x * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 70 ∧ n ≥ 10 ∧ Nat.Prime n ∧ Nat.Prime (reverseDigits n)

theorem sum_palindromic_primes_lt_70 : 
  (Finset.filter is_palindromic_prime (Finset.range 70)).sum =
    92 := sorry

end sum_palindromic_primes_lt_70_l448_448815


namespace simplify_complex_expression_l448_448480

open Complex

def complex_expression : ℂ :=
  (1 + I) * (3 + 4 * I) / I

theorem simplify_complex_expression :
  complex_expression = 7 + I := by
  sorry

end simplify_complex_expression_l448_448480


namespace common_points_count_l448_448288

noncomputable def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
noncomputable def eq2 (x y : ℝ) : Prop := (x + 2 * y - 5) * (3 * x - 4 * y + 6) = 0

theorem common_points_count : 
  (∃ x1 y1 : ℝ, eq1 x1 y1 ∧ eq2 x1 y1) ∧
  (∃ x2 y2 : ℝ, eq1 x2 y2 ∧ eq2 x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)) ∧
  (∃ x3 y3 : ℝ, eq1 x3 y3 ∧ eq2 x3 y3 ∧ (x3 ≠ x1 ∧ x3 ≠ x2 ∧ y3 ≠ y1 ∧ y3 ≠ y2)) ∧ 
  (∃ x4 y4 : ℝ, eq1 x4 y4 ∧ eq2 x4 y4 ∧ (x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ y4 ≠ y1 ∧ y4 ≠ y2 ∧ y4 ≠ y3)) ∧ 
  ∀ x y : ℝ, (eq1 x y ∧ eq2 x y) → (((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))) :=
by
  sorry

end common_points_count_l448_448288


namespace solve_logarithmic_equation_l448_448093

theorem solve_logarithmic_equation :
  ∃ (x₁ x₂ : ℝ), (log 2 (x₁^2 - 8 * x₁ - 20) = 5 ∧ log 2 (x₂^2 - 8 * x₂ - 20) = 5) ∧
  irrational (4 + 2 * Real.sqrt 17) ∧ irrational (4 - 2 * Real.sqrt 17) ∧
  (x₁ = 4 + 2 * Real.sqrt 17) ∧ (x₂ = 4 - 2 * Real.sqrt 17) :=
by
  sorry

end solve_logarithmic_equation_l448_448093


namespace opposite_number_113_is_114_l448_448058

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l448_448058


namespace least_comic_books_l448_448767

theorem least_comic_books (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 4 = 1) : n = 17 :=
sorry

end least_comic_books_l448_448767


namespace jason_optimal_reroll_probability_l448_448416

-- Define the probability function based on the three dice roll problem
def probability_of_rerolling_two_dice : ℚ := 
  -- As per the problem, the computed and fixed probability is 7/64.
  7 / 64

-- Prove that Jason's optimal strategy leads to rerolling exactly two dice with a probability of 7/64.
theorem jason_optimal_reroll_probability : probability_of_rerolling_two_dice = 7 / 64 := 
  sorry

end jason_optimal_reroll_probability_l448_448416


namespace chocolate_probability_l448_448258

theorem chocolate_probability :
  ∃ (total chocolates caramels nougats truffles peanut_clusters : ℕ),
    total = 50 ∧
    caramels = 3 ∧
    nougats = 2 * caramels ∧
    truffles = caramels + 6 ∧
    peanut_clusters = total - (caramels + nougats + truffles) ∧
    (peanut_clusters.toFloat / total.toFloat) * 100 = 64 := 
by
  -- Definitions as per conditions
  let total := 50
  let caramels := 3
  let nougats := 2 * caramels
  let truffles := caramels + 6
  let peanut_clusters := total - (caramels + nougats + truffles)
  -- Calculate the expected percentage
  have h1: (peanut_clusters.toFloat / total.toFloat) * 100 = 64 := by
    norm_num
  -- conclusion
  exact ⟨total, caramels, nougats, truffles, peanut_clusters, rfl, rfl, rfl, rfl, rfl, h1⟩

end chocolate_probability_l448_448258


namespace max_value_in_interval_l448_448649

theorem max_value_in_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → x^4 - 2 * x^2 + 5 ≤ 13 :=
by
  sorry

end max_value_in_interval_l448_448649


namespace possible_root_magnitudes_l448_448472

noncomputable def distinct_complex_roots (P : Polynomial ℂ) : Prop :=
  ∃ r1 r2 r3 r4 r5 : ℂ,
    r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r1 ≠ r5 ∧
    r2 ≠ r3 ∧ r2 ≠ r4 ∧ r2 ≠ r5 ∧
    r3 ≠ r4 ∧ r3 ≠ r5 ∧
    r4 ≠ r5 ∧
    P.roots = [r1, r2, r3, r4, r5]

theorem possible_root_magnitudes (P Q : Polynomial ℂ)
  (h₁ : P.degree = 5)
  (h₂ : distinct_complex_roots P)
  (h₃ : P * Q = P.comp (Polynomial.map (Polynomial.C (1:ℂ)) ^ 2)) :
  ∃ (S : set ℂ), S = {4, 5} ∧ 
  (∀ r₁ r₂ r₃ r₄ r₅ : ℂ, (P.roots = [r₁, r₂, r₃, r₄, r₅]) →
   |r₁| + |r₂| + |r₃| + |r₄| + |r₅| ∈ S) :=
sorry

end possible_root_magnitudes_l448_448472


namespace probability_red_white_blue_no_replacement_l448_448937

theorem probability_red_white_blue_no_replacement :
  let total_balls := 5 + 6 + 4 + 3
  let prob_red := (5 : ℚ) / total_balls
  let prob_white := (4 : ℚ) / (total_balls - 1)
  let prob_blue := (3 : ℚ) / (total_balls - 2)
  (prob_red * prob_white * prob_blue = (5 : ℚ) / 408) :=
by
  let total_balls := 18
  let prob_red := (5 : ℚ) / total_balls
  let prob_white := (4 : ℚ) / (total_balls - 1)
  let prob_blue := (3 : ℚ) / (total_balls - 2)
  have h : prob_red * prob_white * prob_blue = (5 : ℚ * 4 * 3) / (total_balls * (total_balls - 1) * (total_balls - 2)) :=
    by norm_num
  rw [h]
  norm_num
  sorry

end probability_red_white_blue_no_replacement_l448_448937


namespace projection_problem_l448_448122

open Real

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

theorem projection_problem :
  let w := (12 / 13, 5 / 13)
  let v1 := (2, 4)
  let v2 := (4, 1)
  projection v1 w = (12 / 13, 5 / 13) →
  projection v2 w = (636 / 169, 265 / 169) :=
by
  sorry

end projection_problem_l448_448122


namespace cube_of_prism_volume_l448_448957

theorem cube_of_prism_volume (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by
  sorry

end cube_of_prism_volume_l448_448957


namespace stuffed_animal_cost_l448_448442

theorem stuffed_animal_cost (cost_coloring_book : ℝ) (num_coloring_books : ℕ)
                            (cost_pack_peanuts : ℝ) (num_packs_peanuts : ℕ)
                            (total_given : ℝ) (expected_cost_stuffed_animal : ℝ) :
  cost_coloring_book = 4 → 
  num_coloring_books = 2 → 
  cost_pack_peanuts = 1.5 → 
  num_packs_peanuts = 4 →
  total_given = 25 →
  expected_cost_stuffed_animal = 11 →
  let total_cost_coloring_books := cost_coloring_book * num_coloring_books in
  let total_cost_peanuts := cost_pack_peanuts * num_packs_peanuts in
  let combined_cost := total_cost_coloring_books + total_cost_peanuts in
  total_given - combined_cost = expected_cost_stuffed_animal :=
by
  intros h1 h2 h3 h4 h5 h6
  let total_cost_coloring_books := cost_coloring_book * num_coloring_books
  let total_cost_peanuts := cost_pack_peanuts * num_packs_peanuts
  let combined_cost := total_cost_coloring_books + total_cost_peanuts
  have h7 : total_given - combined_cost = expected_cost_stuffed_animal, from sorry
  exact h7

end stuffed_animal_cost_l448_448442


namespace pears_to_peaches_l448_448766

-- Define the weights of pears and peaches
variables (pear peach : ℝ) 

-- Given conditions: 9 pears weigh the same as 6 peaches
axiom weight_ratio : 9 * pear = 6 * peach

-- Theorem to prove: 36 pears weigh the same as 24 peaches
theorem pears_to_peaches (h : 9 * pear = 6 * peach) : 36 * pear = 24 * peach :=
by
  sorry

end pears_to_peaches_l448_448766


namespace opposite_of_113_is_114_l448_448049

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l448_448049


namespace water_level_rise_l448_448922

/-- This definition represents a steel vessel with given dimensions and a cubical box. -/
structure SteelVessel where
  length : ℝ
  breadth : ℝ
  box_edge : ℝ

def base_area (v : SteelVessel) : ℝ :=
  v.length * v.breadth

def box_volume (v : SteelVessel) : ℝ :=
  v.box_edge ^ 3

def water_rise (v : SteelVessel) : ℝ :=
  box_volume v / base_area v

theorem water_level_rise (v : SteelVessel) (h : v.length = 60) (h1 : v.breadth = 30) (h2 : v.box_edge = 30) :
  water_rise v = 15 := by
  sorry

end water_level_rise_l448_448922


namespace bounded_region_area_correct_l448_448573

def octagon_area (s : ℝ) : ℝ :=
  2 * (1 + real.sqrt 2)

def arc_area (s : ℝ) : ℝ :=
  (π * (2 + real.sqrt 2)) / 2

def bounded_region_area (s : ℝ) : ℝ :=
  octagon_area s - arc_area s + 2 * real.sqrt 3

theorem bounded_region_area_correct (s : ℝ) (h : s = 1) :
  bounded_region_area s = 2 * (1 + real.sqrt 2) - (π * (2 + real.sqrt 2)) / 2 + 2 * real.sqrt 3 :=
by
  sorry

end bounded_region_area_correct_l448_448573


namespace unique_common_root_m_value_l448_448736

theorem unique_common_root_m_value (m : ℝ) (h : m > 5) :
  (∃ x : ℝ, x^2 - 5 * x + 6 = 0 ∧ x^2 + 2 * x - 2 * m + 1 = 0) →
  m = 8 :=
by
  sorry

end unique_common_root_m_value_l448_448736


namespace stratified_sampling_red_balls_l448_448745

theorem stratified_sampling_red_balls (total_balls red_balls sample_size : ℕ) (h_total : total_balls = 100) (h_red : red_balls = 20) (h_sample : sample_size = 10) :
  (sample_size * (red_balls / total_balls)) = 2 := by
  sorry

end stratified_sampling_red_balls_l448_448745


namespace f_is_increasing_l448_448218

noncomputable theory

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

open Set

theorem f_is_increasing (f : ℝ → ℝ) (I : set ℝ) (h : ∀ x ∈ I, ∃ y ∈ I, x < y) : is_monotonically_increasing f I := sorry

example : is_monotonically_increasing f (Icc 0 (Real.pi / 2)) :=
begin
  unfold is_monotonically_increasing,
  intros x y hx hy hxy,
  have : 7 * Real.cos (x - Real.pi / 6) > 0 := sorry,
  have : 7 * Real.cos (y - Real.pi / 6) > 0 := sorry,
  calc
    f x = 7 * Real.sin (x - Real.pi / 6) : rfl
    ... < 7 * Real.sin (y - Real.pi / 6) : sorry
    ... = f y : rfl,
end

end f_is_increasing_l448_448218


namespace probability_first_queen_second_diamond_l448_448882

def is_queen (card : ℕ) : Prop :=
card = 12 ∨ card = 25 ∨ card = 38 ∨ card = 51

def is_diamond (card : ℕ) : Prop :=
card % 13 = 0

def first_card_queen (first_card : ℕ) (cards : finset ℕ) : Prop :=
is_queen first_card

def second_card_diamond (second_card : ℕ) (remaining_cards : finset ℕ) : Prop :=
is_diamond second_card

theorem probability_first_queen_second_diamond (cards : finset ℕ) (h : cards.card = 52) :
  ((\sum x in cards.filter is_queen, 1 / 52) * (\sum y in cards.filter is_diamond, 1 / 51)) = (1 / 52) :=
sorry

end probability_first_queen_second_diamond_l448_448882


namespace camera_filter_kit_savings_l448_448558

variable (kit_price : ℝ) (single_prices : List ℝ)
variable (correct_saving_amount : ℝ)

theorem camera_filter_kit_savings
    (h1 : kit_price = 145.75)
    (h2 : single_prices = [3 * 9.50, 2 * 15.30, 1 * 20.75, 2 * 25.80])
    (h3 : correct_saving_amount = -14.30) :
    (single_prices.sum - kit_price = correct_saving_amount) :=
by
  sorry

end camera_filter_kit_savings_l448_448558


namespace appropriate_presentation_length_l448_448879

-- Definitions and conditions
def ideal_speaking_rate : ℕ := 160
def min_minutes : ℕ := 20
def max_minutes : ℕ := 40
def appropriate_words_range (words : ℕ) : Prop :=
  words ≥ (min_minutes * ideal_speaking_rate) ∧ words ≤ (max_minutes * ideal_speaking_rate)

-- Statement to prove
theorem appropriate_presentation_length : appropriate_words_range 5000 :=
by sorry

end appropriate_presentation_length_l448_448879


namespace episodes_per_week_correct_l448_448878

-- Definitions for given conditions
def episode_length := 20 -- in minutes
def filming_multiplier := 1.5 -- 50% longer means 1 + 0.5 = 1.5
def total_film_time := 600 -- 10 hours converted to minutes (10 * 60)
def weeks := 4

-- Helper calculations
def filming_time_per_episode := episode_length * filming_multiplier
def total_episodes := total_film_time / filming_time_per_episode
def episodes_per_week := total_episodes / weeks

-- Lean theorem stating the equivalent proof problem
theorem episodes_per_week_correct : 
    episodes_per_week = 5 := 
by
    -- Placeholder for proof
    sorry

end episodes_per_week_correct_l448_448878


namespace minimum_value_of_x_plus_3y_l448_448430

theorem minimum_value_of_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/(x + 1) + 1/(y + 1) = 1/4) : x + 3 * y = 5 + 4 * real.sqrt 3 :=
sorry

end minimum_value_of_x_plus_3y_l448_448430


namespace part_time_employees_l448_448398

theorem part_time_employees (T U : ℕ) (prob_not_part_time_nor_uninsured : ℚ) (UP_ratio : ℚ) :
  T = 350 →
  U = 104 →
  UP_ratio = 0.125 →
  prob_not_part_time_nor_uninsured = 0.5857142857142857 →
  ∃ P : ℕ, P = 54 :=
by
  intros hT hU hUP_ratio hProb
  let UP := UP_ratio * U
  let N := prob_not_part_time_nor_uninsured * T
  have h1 : UP = 13 := by
    calc UP = 0.125 * 104 : by rw [hUP_ratio, hU]
      ... = 13 : by norm_num
  have h2 : N = 205 := by
    calc N = 0.5857142857142857 * 350 : by rw [hProb, hT]
      ... = 205 : by norm_num

  let P := 246 - 192
  use P
  have h3 : P = 54 := by norm_num
  exact h3

end part_time_employees_l448_448398


namespace max_time_required_approx_l448_448180

-- Define the number of digits in the combination
def num_digits : Nat := 5

-- Define the possible number of choices for each digit
def choices_per_digit : Nat := 9

-- Define the time taken for each trial in seconds
def time_per_trial_seconds : ℝ := 7

-- Calculate the total number of combinations
def total_combinations : ℝ := choices_per_digit ^ num_digits

-- Calculate the maximum time required in seconds
def max_time_seconds : ℝ := total_combinations * time_per_trial_seconds

-- Convert the maximum time required to hours
def max_time_hours : ℝ := max_time_seconds / 3600

-- Theorem: The maximum time required to open the bag is approximately 114.82 hours
theorem max_time_required_approx :
  abs (max_time_hours - 114.82) < 0.01 := by
  sorry

end max_time_required_approx_l448_448180


namespace probability_penny_dime_halfdollar_tails_is_1_over_8_l448_448479

def probability_penny_dime_halfdollar_tails : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_penny_dime_halfdollar_tails_is_1_over_8 :
  probability_penny_dime_halfdollar_tails = 1 / 8 :=
by
  sorry

end probability_penny_dime_halfdollar_tails_is_1_over_8_l448_448479


namespace monotonically_increasing_and_even_l448_448987

theorem monotonically_increasing_and_even :
  (∀ x, x ∈ set.Ioi (0 : ℝ) → 0 ≤ x^2 ∧ ∀ y, y ∈ set.Ioi (0 : ℝ) ∧ y > x → x^2 < y^2) ∧
  (∀ x, x^2 = (-x)^2) ∧
  ¬((∀ x, x ∈ set.Ioi (0 : ℝ) → 0 ≥ -x^3 ∧ ∀ y, y ∈ set.Ioi (0 : ℝ) ∧ y > x → -x^3 < -y^3) ∧
     (∀ x, -x^3 = (-(-x))^3)) ∧
  ¬((∀ x, x ∈ set.Ioi (0 : ℝ) → ∀ y, y ∈ set.Ioi (0 : ℝ) ∧ y > x → -log |x| < -log |y| ) ∧
     (∀ x, -log |x| = -log |-x|)) ∧
  ¬((∀ x, x ∈ set.Ioi (0 : ℝ) → 2^x > 0 ∧ ∀ y, y ∈ set.Ioi (0 : ℝ) ∧ y > x → 2^x < 2^y) ∧
     (∀ x, 2^x = 2^(-x))) :=
begin
  -- Proof omitted
  sorry
end

end monotonically_increasing_and_even_l448_448987


namespace contradiction_proof_l448_448522

theorem contradiction_proof (a b c : ℝ) :
  (a - 1) * (b - 1) * (c - 1) > 0 → (a > 1) ∨ (b > 1) ∨ (c > 1) :=
by
  intros h
  by_contra h1
  push_neg at h1
  have h2 : ((a - 1) ≤ 0) ∧ ((b - 1) ≤ 0) ∧ ((c - 1) ≤ 0) := by
    split
    all_goals
    { auto }
  have h3 : (a - 1) * (b - 1) * (c - 1) ≤ 0 := by
    nlinarith
  exact h2
  sorry

end contradiction_proof_l448_448522


namespace find_y_l448_448337

theorem find_y (y : ℝ) (h : (8 + 15 + 22 + 5 + y) / 5 = 12) : y = 10 :=
by
  -- the proof is skipped
  sorry

end find_y_l448_448337


namespace quasi_regular_polyhedron_properties_l448_448796

theorem quasi_regular_polyhedron_properties
    (n : ℕ) (G P B : ℕ) :
    let T_faces := G in
    let T_edges := P in
    let T_vertices := B in
    let T'_faces := T_vertices in
    let quasi_faces := T_faces + T'_faces in
    let quasi_vertices := T_edges in
    let quasi_edges := 2 * T_edges in
    (quasi_faces = G + B) ∧
    (quasi_edges = 2 * P) ∧
    (quasi_vertices = P)
:= sorry

end quasi_regular_polyhedron_properties_l448_448796


namespace combination_with_repetition_l448_448307

namespace CombinationProblem

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem combination_with_repetition {n r : ℕ} (h : r ≥ n) :
    ∃ (N : ℕ), N = binom (r - 1) (n - 1) :=
by
  use binom (r - 1) (n - 1)
  sorry

end CombinationProblem

end combination_with_repetition_l448_448307


namespace quadrilateral_inequality_l448_448485

-- Define the convex quadrilateral and the properties of its sides
structure Quadrilateral (A B C D : Type) :=
(area_split : A → B → C → D → Prop)    -- predicate representing the area is split by AC
(sides_comparison : A → B → D → Prop)  -- predicate representing AB > AD
(target_comparison : B → C → D → Prop) -- predicate representing BC < DC

-- The main theorem
theorem quadrilateral_inequality (A B C D : Type)
  (q : Quadrilateral A B C D)
  (h_area_split : q.area_split A B C D)
  (h_sides_comparison : q.sides_comparison A B D) :
  q.target_comparison B C D :=
sorry

end quadrilateral_inequality_l448_448485


namespace cos_add_pi_over_4_l448_448665

theorem cos_add_pi_over_4 (α : ℝ) (h : Real.sin (α - π/4) = 1/3) : Real.cos (π/4 + α) = -1/3 := 
  sorry

end cos_add_pi_over_4_l448_448665


namespace opposite_number_in_circle_l448_448019

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l448_448019


namespace intersection_of_sets_l448_448357

open Set

theorem intersection_of_sets : 
  ∀ (M N : Set ℕ), M = {1, 2, 3} → N = {2, 3, 4} → M ∩ N = {2, 3} :=
by
  intros M N hM hN
  rw [hM, hN]
  simp
  sorry

end intersection_of_sets_l448_448357


namespace stuffed_animal_cost_l448_448443

theorem stuffed_animal_cost (cost_coloring_book : ℝ) (num_coloring_books : ℕ)
                            (cost_pack_peanuts : ℝ) (num_packs_peanuts : ℕ)
                            (total_given : ℝ) (expected_cost_stuffed_animal : ℝ) :
  cost_coloring_book = 4 → 
  num_coloring_books = 2 → 
  cost_pack_peanuts = 1.5 → 
  num_packs_peanuts = 4 →
  total_given = 25 →
  expected_cost_stuffed_animal = 11 →
  let total_cost_coloring_books := cost_coloring_book * num_coloring_books in
  let total_cost_peanuts := cost_pack_peanuts * num_packs_peanuts in
  let combined_cost := total_cost_coloring_books + total_cost_peanuts in
  total_given - combined_cost = expected_cost_stuffed_animal :=
by
  intros h1 h2 h3 h4 h5 h6
  let total_cost_coloring_books := cost_coloring_book * num_coloring_books
  let total_cost_peanuts := cost_pack_peanuts * num_packs_peanuts
  let combined_cost := total_cost_coloring_books + total_cost_peanuts
  have h7 : total_given - combined_cost = expected_cost_stuffed_animal, from sorry
  exact h7

end stuffed_animal_cost_l448_448443


namespace find_b1_l448_448761

noncomputable def f (x : ℝ) := 8 + 32 * x - 12 * x ^ 2 - 4 * x ^ 3 + x ^ 4
noncomputable def g (x x1 x2 x3 x4 : ℝ) := (x - x1^2) * (x - x2^2) * (x - x3^2) * (x - x4^2)

theorem find_b1 :
  (∃ x1 x2 x3 x4 : ℝ, (∀ x : ℝ, f x = (x - x1) * (x - x2) * (x - x3) * (x - x4)) ∧
   (coeff (polynomialC (g x x1 x2 x3 x4)) 1 = -1216)) : sorry

end find_b1_l448_448761


namespace semicircle_circumference_is_correct_l448_448116

theorem semicircle_circumference_is_correct :
  let length := 14
  let breadth := 10
  let rectangle_perimeter := 2 * (length + breadth)
  let square_side := rectangle_perimeter / 4
  let semicircle_diameter := square_side
  let pi_approx := 3.14
  let semicircle_circumference := (pi_approx * semicircle_diameter) / 2
  semicircle_circumference = 18.84 :=
by
  let length := 14
  let breadth := 10
  let rectangle_perimeter := 2 * (length + breadth)
  let square_side := rectangle_perimeter / 4
  let semicircle_diameter := square_side
  let pi_approx := 3.14
  let semicircle_circumference := (pi_approx * semicircle_diameter) / 2
  exact sorry

end semicircle_circumference_is_correct_l448_448116


namespace intersection_A_B_l448_448311

def A (y : ℝ) : Prop := ∃ x : ℝ, y = -x^2 + 2*x - 1
def B (y : ℝ) : Prop := ∃ x : ℝ, y = 2*x + 1

theorem intersection_A_B :
  {y : ℝ | A y} ∩ {y : ℝ | B y} = {y : ℝ | y ≤ 0} :=
sorry

end intersection_A_B_l448_448311


namespace problem_l448_448782

variable (a b c : ℝ)

def a_def : a = Real.log (1 / 2) := sorry
def b_def : b = Real.exp (1 / Real.exp 1) := sorry
def c_def : c = Real.exp (-2) := sorry

theorem problem (ha : a = Real.log (1 / 2)) 
               (hb : b = Real.exp (1 / Real.exp 1)) 
               (hc : c = Real.exp (-2)) : 
               a < c ∧ c < b := 
by
  rw [ha, hb, hc]
  sorry

end problem_l448_448782


namespace sophia_book_length_l448_448471

theorem sophia_book_length (P : ℕ) 
  (h1 : Sophia finished 2/3 of a book)
  (h2 : The number of pages Sophia has read is 90 more than the number of pages she has left to read) : P = 270 :=
sorry

end sophia_book_length_l448_448471


namespace days_collected_money_l448_448446

-- Defining constants and parameters based on the conditions
def households_per_day : ℕ := 20
def money_per_pair : ℕ := 40
def total_money_collected : ℕ := 2000
def money_from_households : ℕ := (households_per_day / 2) * money_per_pair

-- The theorem that needs to be proven
theorem days_collected_money :
  (total_money_collected / money_from_households) = 5 :=
sorry -- Proof not provided

end days_collected_money_l448_448446


namespace find_a_and_b_l448_448689

theorem find_a_and_b (a b : ℝ) (hlt₁ : a ≠ 1) (hlt₂ : b ≠ 1) (hgt₁ : a > 0) (hgt₂ : b > 0)
  (hlog : Real.logBase a 2 = Real.logBase (a / 2) 4 ∧ Real.logBase a 2 = Real.logBase b 3) :
  a = 1/2 ∧ b = 1/3 :=
by
  sorry

end find_a_and_b_l448_448689


namespace quadratic_distinct_real_roots_l448_448675

theorem quadratic_distinct_real_roots (a : ℝ) :
  (ax^2 - 4x - 1 = 0) ∧ a ≠ 0 ↔ (a > -4) ∧ (a ≠ 0) := by
  sorry

end quadratic_distinct_real_roots_l448_448675


namespace wellington_population_l448_448119

theorem wellington_population 
  (W P L : ℕ)
  (h1 : P = 7 * W)
  (h2 : P = L + 800)
  (h3 : P + L = 11800) : 
  W = 900 :=
by
  sorry

end wellington_population_l448_448119


namespace decompose_vectors_l448_448627

variables (e1 e2 : Vect) 

noncomputable def AB : Vect := 3 • e2
noncomputable def AC : Vect := 1.5 • e1
noncomputable def CA : Vect := -1.5 • e1
noncomputable def BC : Vect := 1.5 • e1 - 3 • e2

theorem decompose_vectors (e1 e2 : Vect):
  (AB = 0 • e1 + 3 • e2) ∧
  (AC = 1.5 • e1 + 0 • e2) ∧
  (CA = -1.5 • e1 + 0 • e2) ∧
  (BC = 1.5 • e1 - 3 • e2) :=
sorry

end decompose_vectors_l448_448627


namespace quadratic_inequality_solution_l448_448835

-- Definitions for the quadratic polynomial and its roots.
def a : ℚ := -5
def b : ℚ := 7
def c : ℚ := 2

def discrim : ℚ := b^2 - 4 * a * c
def x1 : ℚ := (-b - real.sqrt discrim) / (2 * a)
def x2 : ℚ := (-b + real.sqrt discrim) / (2 * a)

-- Problem statement: proving the solution of inequality.
theorem quadratic_inequality_solution :
  {x : ℝ | -5 * x^2 + 7 * x + 2 > 0} = set.Ioo x1 x2 :=
by
  sorry

end quadratic_inequality_solution_l448_448835


namespace scientific_notation_proof_l448_448837

-- Given number is 657,000
def number : ℕ := 657000

-- Scientific notation of the given number
def scientific_notation (n : ℕ) : Prop :=
    n = 657000 ∧ (6.57 : ℝ) * (10 : ℝ)^5 = 657000

theorem scientific_notation_proof : scientific_notation number :=
by 
  sorry

end scientific_notation_proof_l448_448837


namespace opposite_number_113_is_114_l448_448041

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l448_448041


namespace minimum_cards_to_turn_over_l448_448107

/--
Given five cards with the faces: 2, M, 3, A, E.
Prove that the minimum number of cards needed to turn over
to verify the statement "If a card has a vowel on one face,
then it has an even number on the other" is 3.
-/
theorem minimum_cards_to_turn_over :
  let cards := ['2', 'M', '3', 'A', 'E'] in
  (∀ (card : Char), card ∈ cards → 
     (¬is_vowel card ∨ has_even_number_on_other_side card)) →
  ∃ (count : ℕ), count = 3 :=
sorry

/-- A helper function to check vowels -/
def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

/-- A helper function placeholder to check if a card with a letter has an even number on the other side -/
def has_even_number_on_other_side (card : Char) : Prop :=
  sorry

end minimum_cards_to_turn_over_l448_448107


namespace monotonically_increasing_interval_l448_448231

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l448_448231


namespace find_number_l448_448726

theorem find_number 
  (x y n : ℝ)
  (h1 : n * x = 0.04 * y)
  (h2 : (y - x) / (y + x) = 0.948051948051948) :
  n = 37.5 :=
sorry  -- proof omitted

end find_number_l448_448726


namespace maximum_pyramid_volume_l448_448154

-- Variables and constants given in the problem
variables (A B C S : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ S]
variables (AB AC : ℝ) (angleABC : ℝ) 
constant (sin_angleBAC : ℝ) (max_angle : ℝ )

-- Given conditions
def AB_len : Prop := AB = 5
def AC_len : Prop := AC = 8
def sin_val : Prop := sin_angleBAC = 3/5
def max_angle_val : Prop := max_angle = 60

-- Statement to prove
theorem maximum_pyramid_volume : 
  AB_len AB → AC_len AC → sin_val sin_angleBAC → max_angle_val max_angle →
  ∃ V : ℝ, V = 10 * Real.sqrt 51 :=
by
  sorry

end maximum_pyramid_volume_l448_448154


namespace AB_passes_fixed_point_locus_of_N_l448_448353

-- Definition of the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Definition of the point M which is the right-angle vertex
def M : ℝ × ℝ := (1, 2)

-- Statement for Part 1: Prove line AB passes through a fixed point
theorem AB_passes_fixed_point 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) :
    ∃ P : ℝ × ℝ, P = (5, -2) := sorry

-- Statement for Part 2: Find the locus of point N
theorem locus_of_N 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) 
    (N : ℝ × ℝ)
    (hN : ∃ t : ℝ, N = (t, -(t - 3))) :
    (N.1 - 3)^2 + N.2^2 = 8 ∧ N.1 ≠ 1 := sorry

end AB_passes_fixed_point_locus_of_N_l448_448353


namespace maximum_pyramid_volume_l448_448153

-- Variables and constants given in the problem
variables (A B C S : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ S]
variables (AB AC : ℝ) (angleABC : ℝ) 
constant (sin_angleBAC : ℝ) (max_angle : ℝ )

-- Given conditions
def AB_len : Prop := AB = 5
def AC_len : Prop := AC = 8
def sin_val : Prop := sin_angleBAC = 3/5
def max_angle_val : Prop := max_angle = 60

-- Statement to prove
theorem maximum_pyramid_volume : 
  AB_len AB → AC_len AC → sin_val sin_angleBAC → max_angle_val max_angle →
  ∃ V : ℝ, V = 10 * Real.sqrt 51 :=
by
  sorry

end maximum_pyramid_volume_l448_448153


namespace alice_pints_wednesday_l448_448807

-- Initial conditions
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := pints_monday / 3
def total_pints_before_return : ℕ := pints_sunday + pints_monday + pints_tuesday
def pints_returned_wednesday : ℕ := pints_tuesday / 2
def pints_wednesday : ℕ := total_pints_before_return - pints_returned_wednesday

-- The proof statement
theorem alice_pints_wednesday : pints_wednesday = 18 :=
by
  sorry

end alice_pints_wednesday_l448_448807


namespace number_of_sheets_l448_448968

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end number_of_sheets_l448_448968


namespace even_function_value_at_5_l448_448852

variable {α β : Type}
variable (f : α → β)

def is_even_function (f : α → β) := ∀ x : α, f x = f (-x)

theorem even_function_value_at_5 (h_even : is_even_function f) (h_value : f (-5) = 9) : f 5 = 9 :=
by
  sorry

end even_function_value_at_5_l448_448852


namespace yanna_gave_100_l448_448163

/--
Yanna buys 10 shirts at $5 each and 3 pairs of sandals at $3 each, 
and she receives $41 in change. Prove that she gave $100.
-/
theorem yanna_gave_100 :
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  total_cost + change = 100 :=
by
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  show total_cost + change = 100
  sorry

end yanna_gave_100_l448_448163


namespace point_on_decreasing_line_l448_448853

theorem point_on_decreasing_line (k : ℝ) : (k < 0 ∧ ∃ k, -1 = k * (3 - 2) + 4) ↔ (3, -1) :=
by
  sorry

end point_on_decreasing_line_l448_448853


namespace probability_first_queen_second_diamond_l448_448886

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l448_448886


namespace no_roots_of_form_one_over_n_l448_448209

theorem no_roots_of_form_one_over_n (a b c : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_c : c % 2 = 1) :
  ∀ n : ℕ, ¬(a * (1 / (n:ℚ))^2 + b * (1 / (n:ℚ)) + c = 0) := by
  sorry

end no_roots_of_form_one_over_n_l448_448209


namespace find_a_l448_448723

theorem find_a {a b c : ℕ} (h₁ : a + b = c) (h₂ : b + c = 8) (h₃ : c = 4) : a = 0 := by
  sorry

end find_a_l448_448723


namespace trigonometric_identity_l448_448918

theorem trigonometric_identity (α : ℝ) (h : sin α ≠ 0) :
  sin (7 * α) / sin α - 2 * (cos (2 * α) + cos (4 * α) + cos (6 * α)) - 1 = 0 :=
by
  sorry

end trigonometric_identity_l448_448918


namespace correct_distance_from_S_to_face_PQR_l448_448458

noncomputable def distance_from_S_to_face_PQR
  (P Q R S : ℝ × ℝ × ℝ)
  (h1 : P ≠ S) (h2 : Q ≠ S) (h3 : R ≠ S)
  (h4 : S.1 = 0 ∧ S.2 = 0 ∧ S.3 = 0)
  (h5 : P.1 = 15 ∧ P.2 = 0 ∧ P.3 = 0)
  (h6 : Q.1 = 0 ∧ Q.2 = 15 ∧ Q.3 = 0)
  (h7 : R.1 = 0 ∧ R.2 = 0 ∧ R.3 = 9)
  : ℝ :=
9

theorem correct_distance_from_S_to_face_PQR
  (P Q R S : ℝ × ℝ × ℝ)
  (h1 : P ≠ S) (h2 : Q ≠ S) (h3 : R ≠ S)
  (h4 : S.1 = 0 ∧ S.2 = 0 ∧ S.3 = 0)
  (h5 : P.1 = 15 ∧ P.2 = 0 ∧ P.3 = 0)
  (h6 : Q.1 = 0 ∧ Q.2 = 15 ∧ Q.3 = 0)
  (h7 : R.1 = 0 ∧ R.2 = 0 ∧ R.3 = 9)
  : distance_from_S_to_face_PQR P Q R S h1 h2 h3 h4 h5 h6 h7 = 9 :=
by
  sorry

end correct_distance_from_S_to_face_PQR_l448_448458


namespace range_g_l448_448290

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + Real.pi * Real.arcsin (x / 3) - (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9 * x + 27)

theorem range_g :
  set.range g = set.Icc (11 * Real.pi^2 / 24) (59 * Real.pi^2 / 24) := by
  sorry

end range_g_l448_448290


namespace tree_volume_estimation_l448_448979

def given_data (x y : Fin 10 → ℝ) : Prop :=
  (∑ i, x i = 0.6) ∧
  (∑ i, y i = 3.9) ∧
  (∑ i, (x i) ^ 2 = 0.038) ∧
  (∑ i, (y i) ^ 2 = 1.6158) ∧
  (∑ i, x i * y i = 0.2474) ∧
  let X := 186 in
  let avg_x := 0.06 in
  let avg_y := 0.39 in
  let r := 0.97 in
  (X, avg_y / avg_x * X) = (186, 1209)

theorem tree_volume_estimation
  {x y : Fin 10 → ℝ}
  (h : given_data x y)
  : let avg_x := 0.06 in
    let avg_y := 0.39 in
    let r := 0.97 in
    let X := 186 in
    (avg_x = 0.06) ∧
    (avg_y = 0.39) ∧
    (r = 0.97) ∧
    (avg_y / avg_x * X = 1209) := 
by 
  sorry

end tree_volume_estimation_l448_448979


namespace count_possible_P_l448_448755

-- Define the distinct digits with initial conditions
def digits : Type := {n // n ≥ 0 ∧ n ≤ 9}

-- Define the parameters P, Q, R, S as distinct digits
variables (P Q R S : digits)

-- Define the condition that P, Q, R, S are distinct.
def distinct (P Q R S : digits) : Prop := 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S

-- Assertion conditions based on a valid subtraction layout
def valid_subtraction (P Q R S : digits) : Prop :=
  Q.val - P.val = S.val ∧ (P.val - R.val = P.val) ∧ (P.val - Q.val = S.val)

-- Prove that there are exactly 9 possible values for P.
theorem count_possible_P : ∃ n : ℕ, n = 9 ∧ ∀ P Q R S : digits, distinct P Q R S → valid_subtraction P Q R S → n = 9 :=
by sorry

end count_possible_P_l448_448755


namespace inequality_solution_l448_448867

theorem inequality_solution (x : ℝ) : (x^2 - x - 2 < 0) ↔ (-1 < x ∧ x < 2) :=
by
  sorry

end inequality_solution_l448_448867


namespace f_property_l448_448424

def f (x : ℝ) : ℝ := 1 / (4 ^ x + 2)

theorem f_property (x : ℝ) : f(x) + f(1 - x) = 1 / 2 :=
by
  sorry

end f_property_l448_448424


namespace opposite_number_113_is_114_l448_448045

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l448_448045


namespace shaded_area_satisfies_inequality_l448_448986

theorem shaded_area_satisfies_inequality (x y : ℝ) (h₁ : 0 < x) (h₂ : x ≠ 1) :
  (log x (log x (y^2)) > 0) ↔ (if 1 < x then y^2 > x else 0 < x ∧ x < y^2 ∧ y^2 < 1 / x) :=
by {
    sorry
}

end shaded_area_satisfies_inequality_l448_448986


namespace solution_set_of_inequality_l448_448334

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : ∀ x, f x < 1 / 2) (hf1 : f 1 = 1) (hf' : ∀ x, has_deriv_at f (f x) x) :
  {x | f x < (x / 2) + 1 / 2} = {x | -1 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l448_448334


namespace price_decreased_by_approx_28_59_percent_l448_448935

-- Original price definition
def original_price : Float := 79.95

-- First discount rate definition
def first_discount_rate : Float := 0.20

-- Additional discount rate definition
def additional_discount_rate : Float := 0.15

-- Sales tax rate definition
def sales_tax_rate : Float := 0.05

-- Final sale price calculation and percent decrease calculation
theorem price_decreased_by_approx_28_59_percent :
  let first_discounted_price := original_price * (1 - first_discount_rate)
  let second_discounted_price := first_discounted_price * (1 - additional_discount_rate)
  let final_price := second_discounted_price * (1 + sales_tax_rate)
  let percent_decrease := ((original_price - final_price) / original_price) * 100
  percent_decrease ≈ 28.59 :=
by
  sorry

end price_decreased_by_approx_28_59_percent_l448_448935


namespace birthday_is_sunday_l448_448483

theorem birthday_is_sunday
  (today_is_thursday : today = Thursday)
  (day_before_day_before_yesterday_is_two_days_after_day_before_birthday : 
    day_before day_before yesterday = two_days_after (day_before B)) :
  B = Sunday :=
sorry

end birthday_is_sunday_l448_448483


namespace no_square_has_units_digit_seven_l448_448370

theorem no_square_has_units_digit_seven :
  ¬ ∃ n : ℕ, n ≤ 9 ∧ (n^2 % 10) = 7 := by
  sorry

end no_square_has_units_digit_seven_l448_448370


namespace bridge_length_correct_l448_448596

def train_length : ℕ := 150
def speed_km_hr : ℕ := 45
def time_sec : ℕ := 30

def speed_m_s : ℝ := (speed_km_hr * 1000) / 3600
def distance_traveled : ℝ := speed_m_s * time_sec
def bridge_length : ℝ := distance_traveled - train_length

theorem bridge_length_correct :
  bridge_length = 225 := by
  -- Speed conversion
  have h1: speed_m_s = 12.5 := by
    unfold speed_m_s
    calc
      (45 * 1000) / 3600 = 45000 / 3600 := by ring
      ... = 12.5 := by norm_num
  -- Distance calculation
  have h2: distance_traveled = 375 := by
    unfold distance_traveled
    rw [h1]
    calc
      12.5 * 30 = 375 := by norm_num
  -- Bridge length calculation
  unfold bridge_length
  rw [h2]
  calc
    375 - 150 = 225 := by norm_num

end bridge_length_correct_l448_448596


namespace simplify_complex_expr_l448_448092

theorem simplify_complex_expr : ∀ (i : ℂ), (4 - 2 * i) - (7 - 2 * i) + (6 - 3 * i) = 3 - 3 * i := by
  intro i
  sorry

end simplify_complex_expr_l448_448092


namespace zach_needs_more_money_l448_448539

theorem zach_needs_more_money
  (bike_cost : ℕ) (allowance : ℕ) (mowing_payment : ℕ) (babysitting_rate : ℕ) 
  (current_savings : ℕ) (babysitting_hours : ℕ) :
  bike_cost = 100 →
  allowance = 5 →
  mowing_payment = 10 →
  babysitting_rate = 7 →
  current_savings = 65 →
  babysitting_hours = 2 →
  (bike_cost - (current_savings + (allowance + mowing_payment + babysitting_hours * babysitting_rate))) = 6 :=
by
  sorry

end zach_needs_more_money_l448_448539


namespace sum_of_powers_of_i_l448_448467

theorem sum_of_powers_of_i : 
  (∑ k in Finset.range 2012, (complex.I ^ k)) = 0 :=
by
  -- Definitions for the cyclical pattern
  have h0 : complex.I ^ 0 = 1 := by simp [complex.I]  
  have h1 : complex.I ^ 1 = complex.I := by simp [complex.I]  
  have h2 : complex.I ^ 2 = -1 := by simp [complex.I]  
  have h3 : complex.I ^ 3 = -complex.I := by simp [complex.I]  
  -- Prove the sum
  sorry

end sum_of_powers_of_i_l448_448467


namespace cars_meet_and_crush_fly_l448_448112

noncomputable def time_to_meet (L v_A v_B : ℝ) : ℝ := L / (v_A + v_B)

theorem cars_meet_and_crush_fly :
  ∀ (L v_A v_B v_fly : ℝ), L = 300 → v_A = 50 → v_B = 100 → v_fly = 150 → time_to_meet L v_A v_B = 2 :=
by
  intros L v_A v_B v_fly L_eq v_A_eq v_B_eq v_fly_eq
  rw [L_eq, v_A_eq, v_B_eq]
  simp [time_to_meet]
  norm_num

end cars_meet_and_crush_fly_l448_448112


namespace arrange_courses_non_consecutive_l448_448975

theorem arrange_courses_non_consecutive :
  let periods := 7
  let courses := 4
  ∃! ways, -- There exists a unique number of ways
  (ways = !4! ∧ periods ≥ 2 ⋅ courses + 1) := 24 :=
by sorry

end arrange_courses_non_consecutive_l448_448975


namespace polynomial_sum_proof_l448_448120

noncomputable def given_polynomial_sum (m n : ℚ) : Prop :=
  let p : ℚ[X] := 5 * X^2 - 4 * X + m
  let q : ℚ[X] := 2 * X^2 + n * X - 5
  let product : ℚ[X] := 10 * X^4 - 28 * X^3 + 23 * X^2 - 18 * X + 15
  p * q = product ∧ m + n = 35 / 3

theorem polynomial_sum_proof : ∃ (m n : ℚ), given_polynomial_sum m n :=
sorry

end polynomial_sum_proof_l448_448120


namespace derivative_of_sin_squared_is_sin_2x_l448_448617

open Real

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2

theorem derivative_of_sin_squared_is_sin_2x : 
  ∀ x : ℝ, deriv f x = sin (2 * x) :=
by
  sorry

end derivative_of_sin_squared_is_sin_2x_l448_448617


namespace river_depth_l448_448199

theorem river_depth (width depth : ℝ) (flow_rate_kmph : ℝ) (volume_m3_per_min : ℝ) 
  (h1 : width = 75) 
  (h2 : flow_rate_kmph = 4) 
  (h3 : volume_m3_per_min = 35000) : 
  depth = 7 := 
by
  sorry

end river_depth_l448_448199


namespace machine_does_not_require_repair_l448_448501

noncomputable def nominal_mass := 390 -- The nominal mass M is 390 grams

def greatest_deviation_preserved := 39 -- The greatest deviation among preserved measurements is 39 grams

def deviation_unread_measurements (x : ℕ) : Prop := x < 39 -- Deviations of unread measurements are less than 39 grams

def all_deviations_no_more_than := ∀ (x : ℕ), x ≤ 39 -- All deviations are no more than 39 grams

theorem machine_does_not_require_repair 
  (M : ℕ) 
  (h_nominal_mass : M = nominal_mass)
  (h_greatest_deviation : greatest_deviation_preserved ≤ 0.1 * M)
  (h_unread_deviations : ∀ (x : ℕ), deviation_unread_measurements x) 
  (h_all_deviations : all_deviations_no_more_than):
  true := -- Prove the machine does not require repair
sorry

end machine_does_not_require_repair_l448_448501


namespace incorrect_digits_count_l448_448452

theorem incorrect_digits_count (a b : Nat) (h_a : a = 102325) (h_b : b = 109395) : 
  (Nat.digits 10 a).zip (Nat.digits 10 b) |>.filter (fun (x, y) => x ≠ y) |>.length = 3 := 
by
  sorry

end incorrect_digits_count_l448_448452


namespace monotonic_increasing_l448_448233

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π / 6)

theorem monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 2) → f x1 < f x2 := sorry

end monotonic_increasing_l448_448233


namespace opposite_number_on_circle_l448_448032

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l448_448032


namespace fraction_of_left_handed_non_throwers_l448_448813

theorem fraction_of_left_handed_non_throwers
    (total_players : ℕ) (num_throwers : ℕ) (num_right_handed : ℕ)
    (H1 : total_players = 67) (H2 : num_throwers = 37) (H3 : num_right_handed = 57) (H4 : num_throwers ≤ num_right_handed) :
    ((total_players - num_throwers) - (num_right_handed - num_throwers)) / (total_players - num_throwers) = 1 / 3 :=
by
  simp [H1, H2, H3, H4]
  sorry

end fraction_of_left_handed_non_throwers_l448_448813


namespace max_weight_difference_l448_448453

theorem max_weight_difference (w1 w2 w3 : ℝ) 
(brand1 : 25 - 0.1 ≤ w1 ∧ w1 ≤ 25 + 0.1) 
(brand2 : 25 - 0.2 ≤ w2 ∧ w2 ≤ 25 + 0.2)
(brand3 : 25 - 0.3 ≤ w3 ∧ w3 ≤ 25 + 0.3) : 
∀ w1 w2 w3, max (max (abs (w1 - w2)) (abs (w1 - w3))) (abs (w2 - w3)) = 0.6 :=
by
  sorry

end max_weight_difference_l448_448453


namespace quadratic_roots_c_l448_448422

theorem quadratic_roots_c (a b c : ℝ) (h_roots : ∀ x, x ^ 2 - 7 * x + c = 0 → x = a ∨ x = b) 
  (h_sum_sq : a^2 + b^2 = 17) : c = 16 := 
by
  have h_sum := Vieta's_formula_sum a b c,
  have h_prod := Vieta's_formula_product a b c,
  show c = 16, from sorry

-- Definitions for Vieta's formulas
def Vieta's_formula_sum (a b c : ℝ) (h_roots : ∀ x, x ^ 2 - 7 * x + c = 0 → x = a ∨ x = b) : a + b = 7 := 
by sorry

def Vieta's_formula_product (a b c : ℝ) (h_roots : ∀ x, x ^ 2 - 7 * x + c = 0 → x = a ∨ x = b) : a * b = c := 
by sorry

end quadratic_roots_c_l448_448422


namespace infinitely_many_solutions_l448_448008

theorem infinitely_many_solutions
  (a b c : ℕ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (coprime_ab : Nat.gcd a b = 1)
  (coprime_ac_or_bc : Nat.gcd c a = 1 ∨ Nat.gcd c b = 1) :
  ∃ (m n k : ℕ) (infinite_t : ℕ → ℕ), (m ≠ n ∧ n ≠ k ∧ k ≠ m) ∧ (m > 0 ∧ n > 0 ∧ k > 0) ∧ 
  (∀ t : ℕ, m = infinite_t t * b ∧ n = infinite_t t * a ∧ k = (infinite_t t * a * b + 1) / c ∧ 
   ma + nb = kc) :=
begin
  sorry
end

end infinitely_many_solutions_l448_448008


namespace contestant_wins_quiz_prob_l448_448955

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
nat.choose n k

def probability_of_winning := 
  (binomial_coeff 4 2 * (1/4)^2 * (3/4)^2) + 
  (binomial_coeff 4 3 * (1/4)^3 * (3/4)) + 
  (binomial_coeff 4 4 * (1/4)^4)

theorem contestant_wins_quiz_prob : probability_of_winning = 121 / 256 := by
  -- Proof steps would go here
  sorry

end contestant_wins_quiz_prob_l448_448955


namespace opposite_number_in_circle_l448_448023

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l448_448023


namespace lines_parallel_to_same_line_are_parallel_l448_448084

-- Define a type for lines
constant Line : Type

-- Define the notion of parallel lines as a predicate
constant parallel : Line → Line → Prop

-- State the theorem
theorem lines_parallel_to_same_line_are_parallel 
  (A B C : Line) 
  (hAC : parallel A C) 
  (hBC : parallel B C) : 
  parallel A B :=
sorry

end lines_parallel_to_same_line_are_parallel_l448_448084


namespace jason_test_scores_l448_448512

theorem jason_test_scores :
  let first_three := [86, 72, 68],
      mean_score := 80,
      total_score := 6 * mean_score,
      scores := [86, 72, 68, 89, 87, 78] in
  list.sum first_three + 254 = total_score ∧
  (∀ x ∈ scores, x < 90 ∧ (multiset.card (multiset.of_list scores) = list.length scores)) →
  (list.sort (≥) scores = [89, 87, 86, 78, 72, 68]) := 
by
  sorry

end jason_test_scores_l448_448512


namespace handshakes_count_l448_448613

theorem handshakes_count :
  let team_size := 6
  let referee_count := 3
  let handshakes_between_teams := team_size * team_size
  let total_players := 2 * team_size
  let handshakes_with_referees := total_players * referee_count
  handshakes_between_teams + handshakes_with_referees = 72 :=
by 
  let team_size := 6
  let referee_count := 3
  let handshakes_between_teams := team_size * team_size
  let total_players := 2 * team_size
  let handshakes_with_referees := total_players * referee_count
  have hbt : handshakes_between_teams = 36, from rfl
  have hwr: handshakes_with_referees = 36, from rfl
  show handshakes_between_teams + handshakes_with_referees = 72, by
    simp [hbt, hwr]
    sorry

end handshakes_count_l448_448613


namespace hyperbola_smaller_focus_l448_448632

noncomputable def smaller_focus_coordinates : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 3
  let b := 7
  let c := Real.sqrt (a^2 + b^2)
  (h - c, k)

theorem hyperbola_smaller_focus :
  (smaller_focus_coordinates = (Real.sqrt 58 - 2.62, 20)) :=
by
  sorry

end hyperbola_smaller_focus_l448_448632


namespace find_CE_l448_448011

theorem find_CE (AB BC AC BD : ℝ) (OD OE : ℝ) :
  AB = 5 → BC = 6 → AC = 7 → BD = 5 → OD = OE → 
  ∃ CE : ℝ, CE = Real.sqrt 59 - 3 :=
by
  intros hAB hBC hAC hBD hOD_OE
  use Real.sqrt 59 - 3
  sorry

end find_CE_l448_448011


namespace function_domain_range_fx_leq_1_range_x_l448_448697

noncomputable def f (x : ℝ) := real.sqrt (4 - 8^x)

-- The Lean statement proving the equivalent domain and range
theorem function_domain_range :
  (∀ (x : ℝ), x ≤ 2/3 ↔ 4 - 8^x ≥ 0) ∧
  (∀ (x : ℝ), x ≤ 2/3 → 0 ≤ f x ∧ f x < 2) :=
by sorry

-- The Lean statement proving the additional condition for f(x) <= 1
theorem fx_leq_1_range_x (x : ℝ) :
  (f x ≤ 1) ↔ (real.log 3 / real.log 8 ≤ x ∧ x ≤ 2/3) :=
by sorry

end function_domain_range_fx_leq_1_range_x_l448_448697


namespace no_minus_three_in_range_l448_448303

theorem no_minus_three_in_range (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 3 ≠ -3) ↔ b^2 < 24 :=
by
  sorry

end no_minus_three_in_range_l448_448303


namespace min_value_l448_448692

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y + 1 / x + 2 / y = 13 / 2) :
  x - 1 / y ≥ - 1 / 2 :=
sorry

end min_value_l448_448692


namespace max_min_difference_l448_448348

theorem max_min_difference (f : ℝ → ℝ) (a b : ℝ) (h1 : a = -3) (h2 : b = 3) (h3 : ∀ x, f x = x^3 - 12 * x + 8) : 
  let M := Real.sup (Set.image f (Set.closedInterval a b))
  let m := Real.inf (Set.image f (Set.closedInterval a b))
  M - m = 32 :=
by
  sorry

end max_min_difference_l448_448348


namespace greek_anthology_problem_l448_448306

def pool_fill_time
  (rate1 rate2 rate3 rate4 : ℝ) 
  (t1 t2 t3 t4 : ℝ) 
  (h1 : rate1 = 1 / t1) 
  (h2 : rate2 = 1 / t2) 
  (h3 : rate3 = 1 / t3) 
  (h4 : rate4 = 1 / t4) : Prop :=
  1 / (rate1 + rate2 + rate3 + rate4) = 12 / 25

theorem greek_anthology_problem 
  (t1 t2 t3 t4 : ℝ) 
  (ht1 : t1 = 1) 
  (ht2 : t2 = 2) 
  (ht3 : t3 = 3) 
  (ht4 : t4 = 4) 
  : pool_fill_time (1 / t1) (1 / t2) (1 / t3) (1 / t4) t1 t2 t3 t4
      (by rw ht1) 
      (by rw ht2) 
      (by rw ht3) 
      (by rw ht4) := 
sorry

end greek_anthology_problem_l448_448306


namespace profit_percentage_is_10_percent_l448_448960

theorem profit_percentage_is_10_percent
  (market_price_per_pen : ℕ)
  (retailer_buys_40_pens_for_36_price : 40 * market_price_per_pen = 36 * market_price_per_pen)
  (discount_percentage : ℕ)
  (selling_price_with_discount : ℕ) :
  discount_percentage = 1 →
  selling_price_with_discount = market_price_per_pen - (market_price_per_pen / 100) →
  (selling_price_with_discount * 40 - 36 * market_price_per_pen) / (36 * market_price_per_pen) * 100 = 10 :=
by
  sorry

end profit_percentage_is_10_percent_l448_448960


namespace completely_red_larger_cube_possible_l448_448603

-- Definitions based on conditions
def small_cube_faces := 6
def num_of_small_cubes := 8
def total_faces : ℕ := num_of_small_cubes * small_cube_faces
def blue_faces := total_faces / 3
def red_faces := total_faces - blue_faces
def visible_faces_per_cube := small_cube_faces / 2
def total_visible_faces := num_of_small_cubes * visible_faces_per_cube
def visible_red_faces := total_visible_faces / 3
def visible_blue_faces := total_visible_faces - visible_red_faces

theorem completely_red_larger_cube_possible :
  ∃ (arrangement : fin num_of_small_cubes → fin small_cube_faces → bool),
    (∀ i j, arrangement i j = (some_condition_that_ensures_red_faces)) :=
sorry

end completely_red_larger_cube_possible_l448_448603


namespace potato_slice_length_l448_448586

theorem potato_slice_length (x : ℕ) (h1 : 600 = x + (x + 50)) : x + 50 = 325 :=
by
  sorry

end potato_slice_length_l448_448586


namespace opposite_number_113_is_114_l448_448054

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l448_448054


namespace vector_length_sum_l448_448714

open Real

variables (a b : ℝ × ℝ)

-- Definitions of the conditions
def norm (v : ℝ × ℝ) := sqrt (v.1 * v.1 + v.2 * v.2)
def dot (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2

axiom norm_a : norm a = 1
axiom norm_b : norm b = 1
axiom norm_2a_sub_b : norm (2 * a.1 - b.1, 2 * a.2 - b.2) = 2

-- Proof goal
theorem vector_length_sum : norm (a.1 + b.1, a.2 + b.2) = sqrt 5 / sqrt 2 :=
by sorry

end vector_length_sum_l448_448714


namespace tangent_line_at_origin_l448_448781

-- Definitions for conditions
variables {a : ℝ}
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 2)

-- Assumptions
axiom is_even_function (a : ℝ) : ∀ x : ℝ, f' a x = f' a (-x)

-- Proof statement in Lean
theorem tangent_line_at_origin (a : ℝ) (ha : is_even_function a) : 
  f' a 0 = -2 → ∀ (x : ℝ), f a x = -2 * x :=
sorry

end tangent_line_at_origin_l448_448781


namespace opposite_of_113_is_114_l448_448050

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l448_448050


namespace iron_balls_molded_l448_448992

noncomputable def volume_of_iron_bar (l w h : ℝ) : ℝ :=
  l * w * h

theorem iron_balls_molded (l w h n : ℝ) (volume_of_ball : ℝ) 
  (h_l : l = 12) (h_w : w = 8) (h_h : h = 6) (h_n : n = 10) (h_ball_volume : volume_of_ball = 8) :
  (n * volume_of_iron_bar l w h) / volume_of_ball = 720 :=
by 
  rw [h_l, h_w, h_h, h_n, h_ball_volume]
  rw [volume_of_iron_bar]
  sorry

end iron_balls_molded_l448_448992


namespace probability_first_queen_second_diamond_l448_448880

def is_queen (card : ℕ) : Prop :=
card = 12 ∨ card = 25 ∨ card = 38 ∨ card = 51

def is_diamond (card : ℕ) : Prop :=
card % 13 = 0

def first_card_queen (first_card : ℕ) (cards : finset ℕ) : Prop :=
is_queen first_card

def second_card_diamond (second_card : ℕ) (remaining_cards : finset ℕ) : Prop :=
is_diamond second_card

theorem probability_first_queen_second_diamond (cards : finset ℕ) (h : cards.card = 52) :
  ((\sum x in cards.filter is_queen, 1 / 52) * (\sum y in cards.filter is_diamond, 1 / 51)) = (1 / 52) :=
sorry

end probability_first_queen_second_diamond_l448_448880


namespace opposite_113_eq_114_l448_448064

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l448_448064


namespace crate_stack_probability_l448_448520

open BigOperators

theorem crate_stack_probability :
  ∃ m n : ℕ,
  nat.coprime m n ∧
  (m = 957 ∧ n = 531441) ∧
  12.crate_stack_probability(2, 5, 7, 50) = (m / n : ℚ) :=
by
  sorry

def crate_stack_probability (num_crates : ℕ) (dim1 dim2 dim3 target_height : ℕ) : ℚ :=
  let valid_combos := [((8, 2, 2), 495), ((6, 4, 1), 462)]
  let total_ways := ∑ k in valid_combos, k.2
  let total_possibilities := 3 ^ num_crates
  (total_ways : ℚ) / total_possibilities

end crate_stack_probability_l448_448520


namespace cistern_width_proof_l448_448559

noncomputable def cistern_width (length : ℝ) (height : ℝ) (wet_surface_area : ℝ) : ℝ :=
  let w := ((wet_surface_area - (length * height)) / (length * 2 + height))
  in w

theorem cistern_width_proof :
  cistern_width 7 1.4 68.6 ≈ 3.82 :=
by
  sorry

end cistern_width_proof_l448_448559


namespace incorrect_rounding_option_B_l448_448144

def round_to_nearest (x : ℝ) (precision : ℝ) : ℝ :=
  (Real.floor (x / precision + 0.5)) * precision

theorem incorrect_rounding_option_B :
  let x := 5.13586
  let option_A := round_to_nearest x 0.01 = 5.14
  let option_B := round_to_nearest x 0.01 ≠ 5.136
  let option_C := round_to_nearest x 0.01 = 5.14
  let option_D := round_to_nearest x 0.0001 = 5.1359
  option_B :=
by
  sorry

end incorrect_rounding_option_B_l448_448144


namespace palindrome_count_l448_448563

theorem palindrome_count :
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  (A_choices * B_choices * C_choices) = 900 :=
by
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  show (A_choices * B_choices * C_choices) = 900
  sorry

end palindrome_count_l448_448563


namespace alcohol_percentage_in_first_vessel_l448_448598

theorem alcohol_percentage_in_first_vessel:
  ∃(x : ℝ), (3 * x / 100 + 2 = 2.75) → (x = 25) :=
by
  use 25
  intro h
  have h1 : 3 * 25 / 100 + 2 = 2.75,
  calc
    3 * 25 / 100 + 2 = 0.75 + 2 : by norm_num
    ... = 2.75 : by norm_num
  exact h1.symm.trans h

end alcohol_percentage_in_first_vessel_l448_448598


namespace club_officer_election_l448_448942

theorem club_officer_election (members : Finset ℕ) (h_size : members.card = 12) :
  ∃ ways : ℕ, ways = 12 * 11 * 10 * 9 * 8 ∧ ways = 95040 :=
by
  have h_president := members.card;
  rw h_size at h_president;
  have h_president_choices : h_president = 12 := h_size;
  have h_vp_choices : 11 = 12 - 1 := by simp;
  have h_secretary_choices : 10 = 12 - 2 := by simp;
  have h_treasurer_choices : 9 = 12 - 3 := by simp;
  have h_morale_officer_choices : 8 = 12 - 4 := by simp;
  existsi (12 * 11 * 10 * 9 * 8);
  split;
  { simp [h_president_choices, h_vp_choices, h_secretary_choices, h_treasurer_choices, h_morale_officer_choices] };
  { exact rfl }

end club_officer_election_l448_448942


namespace common_tangent_and_perpendicular_secondary_tangents_l448_448851

noncomputable theory

-- Definitions for parabolas with a common focus at origin and perpendicular axes
def first_parabola {p : ℝ} (x y : ℝ) : Prop :=
  y = (x^2 - p^2) / (2 * p)

def second_parabola {q : ℝ} (x y : ℝ) : Prop :=
  x = (y^2 - q^2) / (2 * q)

-- Statement of the problem in Lean 4
theorem common_tangent_and_perpendicular_secondary_tangents
  {p q : ℝ} :
  (∃ x y : ℝ, first_parabola x y ∧ q * x + p * y + (p^2 + q^2) / 2 = 0) ∧
  (∀ x₁ y₁ : ℝ, (q * x₁ + p * y₁ + (p^2 + q^2) / 2 = 0) →
    (∃ x₂ y₂ : ℝ, first_parabola x₂ y₂ ∧ second_parabola x₂ y₂ ∧
      ((x₁ - x₂) * (y₁ - y₂) = - ((x₂ * y₁) / (p*q)))) := sorry

end common_tangent_and_perpendicular_secondary_tangents_l448_448851


namespace count_valid_numbers_l448_448211

-- Define the problem in Lean
def digits := {1, 2, 3, 4, 5}
def no_repetition (l : List ℕ) : Prop := l.nodup
def valid_number (n : ℕ) : Prop :=
  (23145 < n) ∧ (n < 43521) ∧ no_repetition (n.digits 10) ∧ (n.digits 10).toFinset ⊆ digits

theorem count_valid_numbers : 
  (finset.range 100000).filter valid_number).card = 58 := 
begin
  sorry
end

end count_valid_numbers_l448_448211


namespace EqualAnglesInIsoscelesTriangle_l448_448774

theorem EqualAnglesInIsoscelesTriangle
  {O A B C P Q K L : Type*}
  [T : TriangleIsoCenter O A B C]
  (QQ_symmetric_P : Symmetric P Q (midpoint A B))
  (OQ_intersects_AB_at_K : Intersects O Q A B K)
  (circle_AKO_intersects_AC_at_L : CircleIntersectsThroughPoints A K O A C L) :
  ∠ALP = ∠CLO :=
sorry

end EqualAnglesInIsoscelesTriangle_l448_448774


namespace sum_binom_fraction_eq_factorial_prod_l448_448820

-- Define the problem
theorem sum_binom_fraction_eq_factorial_prod (n : ℕ) (hn : n > 0) (x : ℝ) 
(hx : ∀ k : ℕ, k ∈ finset.range (n+1) → x ≠ -k) : 
  (∑ k in finset.range (n+1), (-1:ℝ)^k * (nat.choose n k) * (x/(x + k))) =
  (nat.factorial n) / (∏ k in finset.range (n+1), (x + k)) := 
sorry

end sum_binom_fraction_eq_factorial_prod_l448_448820


namespace greatest_possible_average_speed_l448_448614

-- Definitions for initial, final readings, and speed calculations
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString;
  s = s.reverse

theorem greatest_possible_average_speed :
  ∀ (initial final : ℕ)
  (speed_limit : ℕ)
  (time : ℕ),
  initial = 31513 →
  (final >= initial ∧ final - initial ≤ speed_limit * time) →
  is_palindrome final →
  speed_limit = 65 →
  time = 3 →
  final = 31713 →
  let distance := final - initial in
  distance / time = 200 / 3 :=
by 
  sorry

end greatest_possible_average_speed_l448_448614


namespace tan_195_l448_448687

theorem tan_195 (a : ℝ) (h : Real.cos 165 = a) : Real.tan 195 = - (Real.sqrt (1 - a^2)) / a := 
sorry

end tan_195_l448_448687


namespace marc_watches_fraction_l448_448015

/--
Suppose Marc bought 50 episodes of a show online 
and he needs 10 days to finish watching all the episodes.
Prove that the fraction of episodes Marc watches each day is 1/10.
-/
theorem marc_watches_fraction :
  ∀ (episodes : ℕ) (days : ℕ) (fraction : ℚ), episodes = 50 → days = 10 → fraction = 1/10 →
  (episodes / days : ℚ) / episodes = fraction :=
by
  intros episodes days fraction h_episodes h_days h_fraction
  rw [←h_episodes, ←h_days, ←h_fraction]
  norm_num
  exact h_fraction

end marc_watches_fraction_l448_448015


namespace probability_queen_then_diamond_l448_448894

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l448_448894


namespace prob1_prob2_prob3_l448_448300

-- Problem 1: Different Solutions
theorem prob1 (x : ℝ) (h1 : 2 * x + 8 > 5 * x + 2) 
  (h2 : 2 * x + 8 + 4 / (x - 1) > 5 * x + 2 + 4 / (x - 1)) : 
  h1 ≠ h2 := 
sorry

-- Problem 2: Same Solutions
theorem prob2 (x : ℝ) (h3 : 2 * x + 8 < 5 * x + 2)
  (h4 : 2 * x + 8 + 4 / (x - 1) < 5 * x + 2 + 4 / (x - 1)) : 
  h3 = h4 := 
sorry

-- Problem 3: Same Solutions 
theorem prob3 (x : ℝ) (h5 : 3 / (x - 1) > (x + 2) / (x - 2)) 
  (h6 : 3 / (x - 1) + (3 * x - 4) / (x - 1) > (x + 2) / (x - 2) + (3 * x - 4) / (x - 1)) : 
  h5 = h6 := 
sorry

end prob1_prob2_prob3_l448_448300


namespace opposite_number_113_is_13_l448_448039

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l448_448039


namespace kamala_overestimation_l448_448095

theorem kamala_overestimation : 
  let p := 150
  let q := 50
  let k := 2
  let d := 3
  let p_approx := 160
  let q_approx := 45
  let k_approx := 1
  let d_approx := 4
  let true_value := (p / q) - k + d
  let approx_value := (p_approx / q_approx) - k_approx + d_approx
  approx_value > true_value := 
  by 
  -- Skipping the detailed proof steps.
  sorry

end kamala_overestimation_l448_448095


namespace standard_deviation_does_not_require_repair_l448_448490

-- Definitions based on conditions
def greatest_deviation (d : ℝ) := d = 39
def nominal_mass (M : ℝ) := 0.1 * M = 39
def unreadable_measurement_deviation (d : ℝ) := d < 39

-- Theorems to be proved
theorem standard_deviation (σ : ℝ) (d : ℝ) (M : ℝ) :
  greatest_deviation d →
  nominal_mass M →
  unreadable_measurement_deviation d →
  σ ≤ 39 :=
by
  sorry

theorem does_not_require_repair (σ : ℝ) :
  σ ≤ 39 → ¬(machine_requires_repair) :=
by
  sorry

-- Adding an assumption that if σ ≤ 39, the machine does not require repair
axiom machine_requires_repair : Prop

end standard_deviation_does_not_require_repair_l448_448490


namespace smaller_cone_height_l448_448565

noncomputable def frustum_height : ℝ := 40
noncomputable def upper_base_area : ℝ := 36 * π
noncomputable def lower_base_area : ℝ := 144 * π

def radius_from_area (area : ℝ) : ℝ := real.sqrt (area / π)
noncomputable def upper_radius : ℝ := radius_from_area upper_base_area
noncomputable def lower_radius : ℝ := radius_from_area lower_base_area

-- Mathematically equivalent proof problem
theorem smaller_cone_height : 
  (upper_radius = 6) → 
  (lower_radius = 12) → 
  (∃ x : ℝ, x + frustum_height = 2 * x) :=
by
  intros h_upper_radius h_lower_radius
  use 20
  simp [h_upper_radius, h_lower_radius, frustum_height]
  sorry

end smaller_cone_height_l448_448565


namespace problem_I_equation_of_curve_C_problem_II_max_area_triangle_O_B_D_l448_448318

/-- The given problem specifies a circle \(C_1\) tangentially intersecting a line and
  describes points and lines that satisfy certain geometric conditions.
  We need to find the equation of an ellipse and the maximum area of a specific triangle. -/

theorem problem_I_equation_of_curve_C :
  ∀ x y : ℝ, (x^2 + y^2 = 9) → (2 * x + sqrt 3 * y^2 = 3) → 
                                  ∃ x y : ℝ, (x^2 / 9 + y^2 / 3 = 1) :=
by sorry

theorem problem_II_max_area_triangle_O_B_D
  (x y m : ℝ) 
  (h₀ : ∀ x y : ℝ, (x^2 / 9 + y^2 / 3 = 1)) 
  (intersection : ∃ (B D : ℝ → ℝ), 2 * x + y + m = 0)
  (h₁ : ∃ (O : ℝ), |m| / sqrt 5 = d ∧ d * B D ≥ 0)
  (h₂ : ∀ (l l₁ : ℝ → Prop), l ⊥ l₁)
  : ∃ A : ℝ, max_area_triangle O B D = 3 * sqrt 3 / 2 :=
by sorry

end problem_I_equation_of_curve_C_problem_II_max_area_triangle_O_B_D_l448_448318


namespace smallest_prime_factor_in_C_l448_448829

def smallest_prime_factor (n : Nat) : Nat :=
  Nat.find (λ p => p.Prime ∧ p ∣ n)

noncomputable def C := {33, 35, 37, 39, 41}

theorem smallest_prime_factor_in_C :
  (smallest_prime_factor 33 = 3 ∧ smallest_prime_factor 39 = 3) ∧
  ∀ n ∈ C, smallest_prime_factor n ≥ 3 :=
by
  have h1 : smallest_prime_factor 33 = 3 := sorry
  have h2 : smallest_prime_factor 39 = 3 := sorry
  have h3 : smallest_prime_factor 35 = 5 := sorry
  have h4 : smallest_prime_factor 37 = 37 := sorry
  have h5 : smallest_prime_factor 41 = 41 := sorry
  exact ⟨⟨h1, h2⟩, by intros n hn; cases hn; rw [hn];
    exact (by norm_num : 3 ≤ 3) <|>
    exact (by norm_num : 3 ≤ 5) <|>
    exact (by norm_num : 3 ≤ 37) <|>
    exact (by norm_num : 3 ≤ 41)⟩

end smallest_prime_factor_in_C_l448_448829


namespace combined_percentage_increase_l448_448248

def initial_interval_days : ℝ := 50
def additive_A_effect : ℝ := 0.20
def additive_B_effect : ℝ := 0.30
def additive_C_effect : ℝ := 0.40

theorem combined_percentage_increase :
  ((1 + additive_A_effect) * (1 + additive_B_effect) * (1 + additive_C_effect) - 1) * 100 = 118.4 :=
by
  norm_num
  sorry

end combined_percentage_increase_l448_448248


namespace fastest_pipe_is_4_l448_448953

/-- There are five pipes with flow rates Q_1, Q_2, Q_3, Q_4, and Q_5.
    The ordering of their flow rates is given by:
    (1) Q_1 > Q_3
    (2) Q_2 < Q_4
    (3) Q_3 < Q_5
    (4) Q_4 > Q_1
    (5) Q_5 < Q_2
    We need to prove that single pipe Q_4 will fill the pool the fastest.
 -/
theorem fastest_pipe_is_4 
  (Q1 Q2 Q3 Q4 Q5 : ℝ)
  (h1 : Q1 > Q3)
  (h2 : Q2 < Q4)
  (h3 : Q3 < Q5)
  (h4 : Q4 > Q1)
  (h5 : Q5 < Q2) :
  Q4 > Q1 ∧ Q4 > Q2 ∧ Q4 > Q3 ∧ Q4 > Q5 :=
by
  sorry

end fastest_pipe_is_4_l448_448953


namespace big_plate_diameter_l448_448187

def diameter_of_big_plate (d : ℝ) (fraction_uncovered : ℝ) : ℝ :=
  let A_small := π * (d / 2) ^ 2
  let D_sq := d ^ 2 / (1 - fraction_uncovered)
  real.sqrt D_sq

theorem big_plate_diameter (d : ℝ) (fraction_uncovered : ℝ) (h_d : d = 10) (h_fraction : fraction_uncovered = 0.3055555555555555) :
  diameter_of_big_plate d fraction_uncovered = 12 :=
by
  rw [h_d, h_fraction]
  simp [diameter_of_big_plate]
  norm_num
  sorry

#print axioms big_plate_diameter

end big_plate_diameter_l448_448187


namespace total_nails_needed_l448_448768

-- Definitions based on problem conditions
def nails_per_plank : ℕ := 2
def planks_needed : ℕ := 2

-- Theorem statement: Prove that the total number of nails John needs is 4.
theorem total_nails_needed : nails_per_plank * planks_needed = 4 := by
  sorry

end total_nails_needed_l448_448768


namespace smallest_sum_of_xy_l448_448329

namespace MathProof

theorem smallest_sum_of_xy (x y : ℕ) (hx : x ≠ y) (hxy : (1 : ℝ) / x + (1 : ℝ) / y = 1 / 12) : x + y = 49 :=
sorry

end MathProof

end smallest_sum_of_xy_l448_448329


namespace positional_relationship_l448_448177

variables {Point Line Plane : Type}
variables (a b : Line) (α : Plane)

-- Condition 1: Line a is parallel to Plane α
def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry

-- Condition 2: Line b is contained within Plane α
def line_contained_within_plane (b : Line) (α : Plane) : Prop := sorry

-- The positional relationship between line a and line b is either parallel or skew
def lines_parallel_or_skew (a b : Line) : Prop := sorry

theorem positional_relationship (ha : line_parallel_to_plane a α) (hb : line_contained_within_plane b α) :
  lines_parallel_or_skew a b :=
sorry

end positional_relationship_l448_448177


namespace translated_point_on_sin2x_l448_448518

theorem translated_point_on_sin2x 
  (s : ℝ) (t : ℝ) 
  (h1 : t = sin (2 * (π / 4) - π / 3)) 
  (h2 : ∃ x', x' + s = π / 4 ∧ sin (2 * x') = sin 2x) 
  (h3 : 0 < s) 
  : t = 1 / 2 ∧ s = π / 6 :=
sorry

end translated_point_on_sin2x_l448_448518


namespace complex_number_problem_l448_448436

theorem complex_number_problem
  (z : ℂ)
  (h1 : 8 * complex.norm_sq z = 3 * complex.norm_sq (z + 3) + complex.norm_sq (z^2 + 2) + 42)
  (h2 : complex.norm_sq z = 4)
  : z - 6 / z = -1.965 :=
  sorry

end complex_number_problem_l448_448436


namespace system_of_equations_correct_l448_448125

theorem system_of_equations_correct (x y : ℤ) :
  (8 * x - 3 = y) ∧ (7 * x + 4 = y) :=
sorry

end system_of_equations_correct_l448_448125


namespace number_of_ways_to_fill_grid_l448_448278

theorem number_of_ways_to_fill_grid :
  ∃ (grid : matrix (fin 3) (fin 3) ℕ),
    (∀ i j, grid i j ∈ finset.range 10) ∧ 
    grid 0 0 = 1 ∧ grid 1 1 = 4 ∧ grid 2 2 = 9 ∧
    (∀ i j, i < 2 → grid i j < grid (i + 1) j) ∧
    (∀ i j, j < 2 → grid i j < grid i (j + 1)) ∧
    ((finset.univ.bind (λ i, finset.univ.image (λ j, grid i j))).card = 9) ∧
    grid.card (λ v, v ∈ {2, 3, 5, 6, 7, 8}) = 6 ∧ 
    some proof that shows there are exactly 12 valid configurations.

end number_of_ways_to_fill_grid_l448_448278


namespace difference_between_two_greatest_values_l448_448169

-- Definition of the variables and conditions
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

variables (a b c x : ℕ)

def conditions (a b c : ℕ) := is_digit a ∧ is_digit b ∧ is_digit c ∧ 2 * a = b ∧ b = 4 * c ∧ a > 0

-- Definition of x as a 3-digit number given a, b, and c
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def smallest_x : ℕ := three_digit_number 2 4 1
def largest_x : ℕ := three_digit_number 4 8 2

def difference_two_greatest_values (a b c : ℕ) : ℕ := largest_x - smallest_x

-- The proof statement
theorem difference_between_two_greatest_values (a b c : ℕ) (h : conditions a b c) : 
  ∀ x1 x2 : ℕ, 
    three_digit_number 2 4 1 = x1 →
    three_digit_number 4 8 2 = x2 →
    difference_two_greatest_values a b c = 241 :=
by
  sorry

end difference_between_two_greatest_values_l448_448169


namespace sum_first_20_odd_integers_gt_10_l448_448530

theorem sum_first_20_odd_integers_gt_10 :
  let a1 := 11 in
  let d := 2 in
  let n := 20 in
  let an := a1 + (n - 1) * d in
  let Sn := (n / 2) * (a1 + an) in
  Sn = 600 :=
by
  let a1 := 11
  let d := 2
  let n := 20
  let an := a1 + (n - 1) * d
  let Sn := (n / 2) * (a1 + an)
  sorry

end sum_first_20_odd_integers_gt_10_l448_448530


namespace trajectory_of_P_l448_448684

-- Define point M, point N, and the condition for point P
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the condition for point P
def condition_P (P : ℝ × ℝ) : Prop := 
  let d_PM := real.sqrt ((P.1 + 2)^2 + P.2^2)
  let d_PN := real.sqrt ((P.1 - 2)^2 + P.2^2)
  d_PM - d_PN = 2 * real.sqrt 2

-- The proof statement
theorem trajectory_of_P (P : ℝ × ℝ) (h : condition_P P) : 
  (P.1^2 / 2 - P.2^2 / 2 = 1) ∧ (P.1 > 0) :=
sorry

end trajectory_of_P_l448_448684


namespace num_divisors_2004_pow_n_in_factorial_2004_l448_448369

theorem num_divisors_2004_pow_n_in_factorial_2004 :
  ∃ n, ∀ (k : ℕ), 2004^k ∣ nat.factorial 2004 ↔ k ≤ n :=
by
  sorry

end num_divisors_2004_pow_n_in_factorial_2004_l448_448369


namespace triangle_altitude_condition_l448_448113

theorem triangle_altitude_condition (h_a h_b h_c : ℝ) (h_a_val : h_a = 8) (h_b_val : h_b = 12) :
  ∀ h_c_val ∈ {4, 7, 8, 12, 23}, (h_a = 8 ∧ h_b = 12 ∧ h_c = h_c_val) → h_c > 4.8 := 
by
  intro h_c_val
  intro h_c_val_in_set
  intro conds
  sorry

end triangle_altitude_condition_l448_448113


namespace opposite_number_in_circle_l448_448024

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l448_448024


namespace simplified_expr_l448_448165

variable (a b : ℝ)

-- Mathematical conditions from part a)
def condition1 : Prop := (a = 4.91)
def condition2 : Prop := (b = 0.09)

-- The mathematical expression from part c
def expr :=
  ((a ^ 2 - b ^ 2) * (a ^ 2 + b ^ (2 / 3) + a * b ^ (1 / 3))) /
  (a * b ^ (1 / 3) + a * a ^ (1 / 2) - b * b ^ (1 / 3) - (a * b ^ 2) ^ (1 / 2)) /
  ((a ^ 3 - b) / (a * b ^ (1 / 3) - (a ^ 3 * b ^ 2) ^ (1 / 6) - b ^ (2 / 3) + a * a ^ (1 / 2)))

-- The theorem to prove
theorem simplified_expr : condition1 a b → condition2 a b → expr a b = 5 :=
by
  sorry

end simplified_expr_l448_448165


namespace find_f_l448_448335

def f : ℝ → ℝ := sorry

theorem find_f (x : ℝ) : f (x + 2) = 2 * x + 3 → f x = 2 * x - 1 :=
by
  intro h
  -- Proof goes here 
  sorry

end find_f_l448_448335


namespace find_probability_l448_448341

variable {ξ : Type*} [Fintype ξ] [DecidableEq ξ]
variable (P : ξ → ℚ)
variable (m : ℚ)

-- Condition 1: The given distribution list of ξ.
def distribution_list (k : ℚ) : Prop :=
  P k = (m / (2^k))

-- Condition 2: The sum of probabilities equating to 1.
def sum_probabilities_eq_one : Prop :=
  m * (1/2 + 1/4 + 1/8 + 1/16) = 1

theorem find_probability
  (h1 : ∀ k ∈ ({1, 2, 3, 4} : set ℚ), distribution_list P m k)
  (h2 : sum_probabilities_eq_one m) :
  P 3 + P 4 = 1/5 :=
sorry

end find_probability_l448_448341


namespace sales_tax_difference_l448_448618

theorem sales_tax_difference (price : ℝ) (tax_rate1 tax_rate2 : ℝ) :
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 →
  (price * tax_rate1 - price * tax_rate2 = 0.5) :=
by
  intros
  sorry

end sales_tax_difference_l448_448618


namespace circle_equation_l448_448283

theorem circle_equation (a b : ℝ) (r : ℝ) (x y : ℝ) :
  (b = -2 * a) ∧ (∃ (a b : ℝ), ∀ (p : point), p = (2,-1) → 
  (a = 1) ∧ (b = -2) ∧ (r = sqrt 2) ∧ ((x-1)^2 + (y+2)^2 = r^2)) → 
  (x-1)^2 + (y+2)^2 = 2 :=
by sorry

end circle_equation_l448_448283


namespace discount_application_l448_448506

open Real

theorem discount_application :
  let list_price := 68.0
  let first_discount := 0.10
  let second_discount := 0.08235294117647069
  let price_after_first_discount := list_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount ≈ 56.16 :=
by
  let list_price := 68.0
  let first_discount := 0.10
  let second_discount := 0.08235294117647069
  let price_after_first_discount := list_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  apply sorry

end discount_application_l448_448506


namespace moles_of_water_combined_l448_448651

-- Definitions
def moles_Be2C_combined := 1 -- 1 mole of Beryllium carbide combined
def moles_BeOH2_formed := 2 -- 2 moles of Beryllium hydroxide formed

-- The statement to prove
theorem moles_of_water_combined :
  let balanced_reaction := 1 * 4 = 2 * (2 / 2) in
  (moles_Be2C_combined = 1 ∧ moles_BeOH2_formed = 2) →
  ∃ moles_H2O_combined, moles_H2O_combined = 4 :=
by
  sorry

end moles_of_water_combined_l448_448651


namespace max_value_of_Tn_at_4_l448_448096

open Real Nat

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 0 * ((1 - (sqrt 2) ^ n) / (1 - sqrt 2))

theorem max_value_of_Tn_at_4 {a : ℕ → ℝ} (h_geom : geometric_sequence a (sqrt 2)) 
  (h_pos : 0 < a 0) (h_nat : ∀ n, 0 < n → 0 < a n) 
  (S : ℕ → ℝ := sum_first_n_terms a) :
  (let T : ℕ → ℝ := λ n, (17 * S n - S (2 * n)) / (a (n + 1))
  in ∀ n, T n ≤ T 4) :=
by
  sorry

end max_value_of_Tn_at_4_l448_448096


namespace find_largest_m_l448_448577

-- Define the sequences a_n and b_n as per conditions
def a_seq (n : ℕ) : ℝ :=
if n = 0 then 1/2 else
let a_n := a_seq (n - 1) in
  a_n / (2 * a_n + 3)

def b_seq (n : ℕ) : ℝ := 1 + 1 / a_seq n

-- Define the summation involving log base 3 of b_k
def summation_log (n : ℕ) : ℝ :=
(finset.range n).sum (λ k, 1 / real.log 3 (b_seq (k + 1)))

-- Define the problem statement
theorem find_largest_m :
  ∃ m : ℤ, ∀ (n : ℤ), n ≥ 2 → summation_log n > (m / 24) ∧ m = 13 :=
sorry -- proof goes here

end find_largest_m_l448_448577


namespace parallel_trans_l448_448086

-- Parallel relationship
def parallel (L1 L2 : Type) : Prop := sorry

-- Definitions of lines
constant Line : Type
constant LineA : Line
constant LineB : Line
constant LineC : Line

-- Conditions
axiom ma_parallel : parallel LineA LineC
axiom mb_parallel : parallel LineB LineC

-- Theorem to prove
theorem parallel_trans :
  (parallel LineA LineC) ∧ (parallel LineB LineC) → (parallel LineA LineB) :=
by {
  intro h,
  cases h with ha hb,
  exact sorry
}

end parallel_trans_l448_448086


namespace portrait_is_father_l448_448477

variable (Person : Type) 
variable (is_only_child : Prop) 
variable (portrait_son_of_father : Prop)
variable (person_in_portrait_is_father : Prop)

axiom (grew_up_alone : is_only_child)
axiom (portrait_statement : portrait_son_of_father)

-- Proof problem: Given the conditions, prove that the person in the portrait is the person's father
theorem portrait_is_father (h1 : is_only_child) (h2 : portrait_son_of_father) : person_in_portrait_is_father := 
by
  sorry

end portrait_is_father_l448_448477


namespace complex_root_seventh_power_l448_448002

theorem complex_root_seventh_power (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end complex_root_seventh_power_l448_448002


namespace lines_parallel_to_same_are_parallel_l448_448080

theorem lines_parallel_to_same_are_parallel (L1 L2 L3 : Type) [linear_ordered_space L1] 
  (h1 : L1 ∥ L3) (h2 : L2 ∥ L3) : L1 ∥ L2 :=
sorry

end lines_parallel_to_same_are_parallel_l448_448080


namespace tom_spent_10_minutes_on_bus_l448_448517

theorem tom_spent_10_minutes_on_bus :
  let start_time := 7 * 60 + 15  -- minutes from midnight for 7:15 a.m.
  let end_time := 17 * 60        -- minutes from midnight for 5:00 p.m.
  let total_time := end_time - start_time
  let class_time := 7 * 55
  let lunch_time := 40
  let activity_time := 2.5 * 60
  let school_time := class_time + lunch_time + activity_time
in 
total_time - school_time = 10 := 
sorry

end tom_spent_10_minutes_on_bus_l448_448517
