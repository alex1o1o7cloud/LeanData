import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_fresh_grapes_l1238_123820

/-- The percentage of water in fresh grapes, given the weight of dried grapes obtained from a fixed amount of fresh grapes -/
theorem water_percentage_in_fresh_grapes 
  (fresh_weight : ℝ) 
  (dried_weight : ℝ) 
  (dried_water_percentage : ℝ) : 
  fresh_weight > 0 → 
  dried_weight > 0 → 
  dried_water_percentage = 10 → 
  dried_weight = 33.33333333333333 → 
  fresh_weight = 100 → 
  (fresh_weight - (fresh_weight * (70 / 100))) = (dried_weight * (1 - dried_water_percentage / 100)) := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_percentage_in_fresh_grapes_l1238_123820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1238_123807

theorem triangle_inequalities (A B C : Real) (h : A + B + C = Real.pi) :
  (0 < Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ∧ 
   Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ≤ 1/8) ∧
  (-1 < Real.cos A * Real.cos B * Real.cos C ∧ 
   Real.cos A * Real.cos B * Real.cos C ≤ 1/8) ∧
  (1 < Real.sin (A/2) + Real.sin (B/2) + Real.sin (C/2) ∧ 
   Real.sin (A/2) + Real.sin (B/2) + Real.sin (C/2) ≤ 3/2) ∧
  (2 < Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) ∧ 
   Real.cos (A/2) + Real.cos (B/2) + Real.cos (C/2) ≤ (3 * Real.sqrt 3)/2) ∧
  (0 < Real.sin A + Real.sin B + Real.sin C ∧ 
   Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3)/2) ∧
  (1 < Real.cos A + Real.cos B + Real.cos C ∧ 
   Real.cos A + Real.cos B + Real.cos C ≤ 3/2) ∧
  (1/Real.sin A + 1/Real.sin B + 1/Real.sin C ≥ 2 * Real.sqrt 3) ∧
  (1/Real.cos A^2 + 1/Real.cos B^2 + 1/Real.cos C^2 ≥ 3) := by
  sorry

#check triangle_inequalities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1238_123807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_y_l1238_123847

noncomputable def m : ℝ := Real.tan (22.5 * Real.pi / 180) / (1 - Real.tan (22.5 * Real.pi / 180)^2)

noncomputable def y (x : ℝ) : ℝ := 2 * m * x + 3 / (x - 1) + 1

theorem min_value_y :
  ∀ x > 1, y x ≥ 2 * Real.sqrt 3 + 2 ∧ ∃ x₀ > 1, y x₀ = 2 * Real.sqrt 3 + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_y_l1238_123847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_sum_l1238_123898

/-- A rhombus with side length 2 -/
structure Rhombus where
  side_length : ℝ
  is_two : side_length = 2

/-- The sum of the lengths of the diagonals of a rhombus with side length 2 -/
noncomputable def diagonal_sum (r : Rhombus) : ℝ :=
  let e := r.side_length * Real.sqrt 2
  let f := r.side_length * Real.sqrt 2
  e + f

theorem rhombus_diagonal_sum : ∀ r : Rhombus, diagonal_sum r = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_diagonal_sum_l1238_123898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_age_when_double_l1238_123894

/-- Hannah's age a certain number of years ago -/
def hannah_age_past : ℕ → ℕ := sorry

/-- July's age a certain number of years ago -/
def july_age_past : ℕ → ℕ := sorry

/-- Current age of July -/
def july_age_current : ℕ := sorry

/-- Current age of July's husband -/
def july_husband_age : ℕ := sorry

/-- The number of years that have passed -/
def years_passed : ℕ := 20

theorem hannah_age_when_double :
  (∀ (x : ℕ), hannah_age_past x = 2 * july_age_past x) →
  july_husband_age = july_age_current + 2 →
  july_husband_age = 25 →
  hannah_age_past years_passed = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_age_when_double_l1238_123894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_l1238_123868

/-- The number of isosceles triangles with integer side lengths and perimeter 100 -/
def num_isosceles_triangles : ℕ :=
  (Finset.filter (λ a : ℕ ↦
    let base := 100 - 2 * a
    a > base ∧ a + a + base = 100
  ) (Finset.range 101)).card

theorem count_isosceles_triangles : num_isosceles_triangles = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_isosceles_triangles_l1238_123868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l1238_123865

/-- The total amount earned by three workers given their individual work rates and one worker's share --/
noncomputable def total_earnings (x_days y_days z_days : ℝ) (z_share : ℝ) : ℝ :=
  let x_rate := 1 / x_days
  let y_rate := 1 / y_days
  let z_rate := 1 / z_days
  let total_rate := x_rate + y_rate + z_rate
  let total_days := 1 / total_rate
  let z_portion := (z_rate * total_days) / 1
  (z_share / z_portion) * 1

theorem total_earnings_proof (x_days y_days z_days z_share : ℝ) 
  (hx : x_days = 2) 
  (hy : y_days = 4) 
  (hz : z_days = 6) 
  (hz_share : z_share = 1090.909090909091) :
  total_earnings x_days y_days z_days z_share = 5995 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_earnings 2 4 6 1090.909090909091

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_earnings_proof_l1238_123865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1238_123869

noncomputable section

def f (x : ℝ) : ℝ := x / (Real.exp x)

theorem f_derivative : 
  deriv f = λ x => (1 - x) / (Real.exp x) := by
  -- The proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l1238_123869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l1238_123834

/-- The equation of a hyperbola in the form (ax + b)^2 / c^2 - (dy + e)^2 / f^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a hyperbola --/
noncomputable def center (h : Hyperbola) : ℝ × ℝ :=
  (- h.b / h.a, h.e / h.d)

/-- The given hyperbola --/
def given_hyperbola : Hyperbola :=
  { a := 4
    b := 8
    c := 9
    d := 2
    e := -6
    f := 7 }

theorem hyperbola_center :
  center given_hyperbola = (-2, 3) := by
  unfold center given_hyperbola
  simp
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l1238_123834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_l1238_123848

/-- A point (a, b) is inside the unit circle if a^2 + b^2 < 1 -/
def inside_unit_circle (a b : ℝ) : Prop := a^2 + b^2 < 1

/-- The distance from the origin to the line ax + by = 1 -/
noncomputable def distance_to_line (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

/-- Two geometric objects are separate if the distance between them is greater than 0 -/
def are_separate (d : ℝ) : Prop := d > 1

/-- Theorem stating that if a point (a, b) is inside the unit circle,
    then the line ax + by = 1 is separate from the unit circle -/
theorem line_circle_separate (a b : ℝ) :
  inside_unit_circle a b → are_separate (distance_to_line a b) := by
  intro h
  unfold inside_unit_circle at h
  unfold are_separate
  unfold distance_to_line
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_l1238_123848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_problem_solution_correct_l1238_123828

/-- Two cyclists ride towards each other from points A and B -/
structure CyclistProblem where
  /-- Time (in hours) for first cyclist to reach B after meeting -/
  t1 : ℚ
  /-- Time (in hours) for second cyclist to reach A after meeting -/
  t2 : ℚ
  /-- Speed of first cyclist -/
  v1 : ℚ
  /-- Speed of second cyclist -/
  v2 : ℚ
  /-- Cyclists ride at constant speeds -/
  constant_speed : v1 > 0 ∧ v2 > 0

/-- The solution to the cyclist problem -/
def solve_cyclist_problem (p : CyclistProblem) : ℚ × ℚ :=
  (1, 3/2)

/-- Theorem stating the correctness of the solution -/
theorem cyclist_problem_solution_correct (p : CyclistProblem)
    (h1 : p.t1 = 2/3)
    (h2 : p.t2 = 3/2) :
    solve_cyclist_problem p = (1, 3/2) := by
  unfold solve_cyclist_problem
  rfl

#check cyclist_problem_solution_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_problem_solution_correct_l1238_123828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l1238_123899

/-- Number of terms in an arithmetic sequence -/
def num_terms (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℕ :=
  ((aₙ - a₁) / d + 1).toNat

/-- Theorem: The number of terms in the arithmetic sequence from -5 to 50 with common difference 5 is 12 -/
theorem arithmetic_sequence_terms : num_terms (-5) 50 5 = 12 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_terms_l1238_123899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_area_difference_l1238_123853

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem paper_area_difference : ℝ := by
  let sheet1_length : ℝ := 11
  let sheet1_width : ℝ := 17
  let sheet2_length : ℝ := 8.5
  let sheet2_width : ℝ := 11
  let sheet1_area := 2 * sheet1_length * sheet1_width
  let sheet2_area := 2 * sheet2_length * sheet2_width
  have h : sheet1_area - sheet2_area = 187 := by sorry
  exact 187


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_area_difference_l1238_123853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_equilateral_triangle_forms_two_cones_l1238_123887

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  -- Add necessary fields
  side_length : ℝ
  center : ℝ × ℝ

/-- Represents a cone -/
structure Cone where
  -- Add necessary fields
  base_radius : ℝ
  height : ℝ

/-- Represents the result of rotating a shape around an axis -/
def RotationResult := Cone × Cone

/-- Predicate to check if a line contains the base of a triangle -/
def base_line_contains_triangle_base (triangle : EquilateralTriangle) (base_line : Set (ℝ × ℝ)) : Prop :=
  sorry -- Placeholder for the actual condition

/-- Function to rotate a shape around an axis -/
def rotate_around_axis (triangle : EquilateralTriangle) (base_line : Set (ℝ × ℝ)) : RotationResult :=
  sorry -- Placeholder for the actual rotation logic

/-- 
  Given an equilateral triangle and the line containing its base,
  rotating the triangle around this line forms two cones.
-/
theorem rotate_equilateral_triangle_forms_two_cones 
  (triangle : EquilateralTriangle) 
  (base_line : Set (ℝ × ℝ)) 
  (h_base : base_line_contains_triangle_base triangle base_line) :
  ∃ (result : RotationResult), 
    rotate_around_axis triangle base_line = result := by
  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_equilateral_triangle_forms_two_cones_l1238_123887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_decrease_percentage_l1238_123805

noncomputable def membership_decrease (initial_increase : ℝ) (total_change : ℝ) : ℝ :=
  let fall_membership := 100 + initial_increase
  let spring_membership := 100 + total_change
  let spring_decrease := fall_membership - spring_membership
  spring_decrease / fall_membership * 100

-- The actual theorem
theorem spring_decrease_percentage :
  abs (membership_decrease 7 (-13.33) - 19) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_decrease_percentage_l1238_123805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_interval_l1238_123850

open Set
open Function
open Real

noncomputable def f (x : ℝ) := Real.cos (1/2 * x - π/3)

theorem f_strictly_increasing_interval :
  let S := {x : ℝ | x ∈ Set.Icc (-2*π) (2*π) ∧ StrictMono (fun t ↦ f (t + x))}
  S = Ioo (-4*π/3) (2*π/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_interval_l1238_123850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_comparison_l1238_123811

theorem power_comparison (n : ℕ) :
  (n < 2 → (n : ℝ) ^ (n + 1 : ℕ) < (n + 1 : ℝ) ^ n) ∧
  (n > 3 → (n : ℝ) ^ (n + 1 : ℕ) > (n + 1 : ℝ) ^ n) := by
  sorry

-- Example application for n = 2004
example : (2004 : ℝ) ^ 2005 > (2005 : ℝ) ^ 2004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_comparison_l1238_123811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f2_g2_value_l1238_123813

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the symmetry properties
axiom f_symmetry : ∀ x : ℝ, f x = f (2 - x)
axiom g_symmetry : ∀ x : ℝ, g x + g (2 - x) = -4

-- Define the given equation
axiom given_equation : ∀ x : ℝ, f x + g x = 9^x + x^3 + 1

-- Theorem to prove
theorem f2_g2_value : f 2 * g 2 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f2_g2_value_l1238_123813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1238_123824

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point A
def A : ℝ × ℝ := (-2, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define vector AO
def AO : ℝ × ℝ := (O.fst - A.fst, O.snd - A.snd)

-- Define vector AP for a point P on the circle
def AP (x y : ℝ) : ℝ × ℝ := (x - A.fst, y - A.snd)

-- Define dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.fst * v2.fst + v1.snd * v2.snd

-- Theorem statement
theorem max_dot_product :
  ∀ x y : ℝ, on_circle x y →
  dot_product AO (AP x y) ≤ 6 ∧
  ∃ x₀ y₀ : ℝ, on_circle x₀ y₀ ∧ dot_product AO (AP x₀ y₀) = 6 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_l1238_123824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l1238_123819

/-- Represents the time it takes Joe to get from home to school -/
noncomputable def total_time (walking_speed : ℝ) (walking_time : ℝ) (stop_time : ℝ) (running_speed_multiplier : ℝ) : ℝ :=
  walking_time + stop_time + walking_time / running_speed_multiplier

/-- Theorem stating that Joe's total time to get to school is 15.5 minutes -/
theorem joe_school_time : 
  ∀ (walking_speed : ℝ),
  walking_speed > 0 →
  total_time walking_speed 10 3 4 = 15.5 :=
by
  intros walking_speed h
  unfold total_time
  -- The actual proof would go here, but we'll use sorry for now
  sorry

#check joe_school_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_school_time_l1238_123819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_and_angle_condition_l1238_123889

def a : Fin 3 → ℝ := ![8, -5, -3]
def c : Fin 3 → ℝ := ![-3, -2, 3]
def b : Fin 3 → ℝ := ![-2, 1, 1]

theorem collinear_and_angle_condition :
  (∃ t : ℝ, b = fun i => a i + t * (c i - a i)) ∧
  (Finset.sum (Finset.range 3) (fun i => (a i) * (b i)))^2 = 
  4 * (Finset.sum (Finset.range 3) (fun i => (b i) * (c i)))^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_and_angle_condition_l1238_123889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_distance_sin_2α_value_l1238_123829

noncomputable def ω : ℝ := 2

noncomputable def f (x : ℝ) : ℝ := Real.cos (ω * x)

theorem symmetry_distance (x : ℝ) : f (x + π / 2) = f x := by sorry

theorem sin_2α_value (α : ℝ) 
  (h1 : α > 5 * π / 12) 
  (h2 : α < π / 2) 
  (h3 : f (α + π / 3) = 1 / 3) : 
  Real.sin (2 * α) = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_distance_sin_2α_value_l1238_123829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1238_123804

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the point on the tangent line
def point : ℝ × ℝ := (3, 5)

-- Define the possible tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 5*x - 12*y + 45 = 0
def tangent_line_2 (x : ℝ) : Prop := x = 3

theorem tangent_line_to_circle :
  ∃ (x y : ℝ), circle_eq x y ∧
  ((tangent_line_1 x y ∧ (x, y) ≠ point) ∨ tangent_line_2 x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1238_123804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_increasing_divisor_sum_ratio_l1238_123801

/-- Sum of positive divisors of n -/
def s (n : ℕ) : ℕ := sorry

/-- Theorem: There are infinitely many n such that s(n)/n > s(m)/m for all m < n -/
theorem infinitely_many_increasing_divisor_sum_ratio :
  ∃ (f : ℕ → ℕ), StrictMono f ∧
    ∀ (i : ℕ) (m : ℕ), m < f i → (s (f i) : ℚ) / (f i : ℚ) > (s m : ℚ) / (m : ℚ) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_increasing_divisor_sum_ratio_l1238_123801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_is_two_l1238_123879

/-- Represents a cistern with given dimensions and wet surface area. -/
structure Cistern where
  length : ℝ
  depth : ℝ
  total_wet_area : ℝ

/-- Calculates the width of a cistern based on its dimensions and wet surface area. -/
noncomputable def calculate_cistern_width (c : Cistern) : ℝ :=
  (c.total_wet_area - 2 * c.length * c.depth) / (2 * c.depth + c.length)

/-- Theorem stating that a cistern with the given dimensions has a width of 2 meters. -/
theorem cistern_width_is_two (c : Cistern) 
    (h1 : c.length = 4)
    (h2 : c.depth = 1.25)
    (h3 : c.total_wet_area = 23) :
    calculate_cistern_width c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_width_is_two_l1238_123879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1238_123892

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (3 + i) / (1 - i)

theorem z_in_first_quadrant : 
  z.re > 0 ∧ z.im > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1238_123892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_broccoli_purchase_l1238_123812

noncomputable def broccoli_price : ℝ := 4
noncomputable def orange_price : ℝ := 0.75
def orange_quantity : ℕ := 3
noncomputable def cabbage_price : ℝ := 3.75
noncomputable def bacon_price : ℝ := 3
noncomputable def chicken_price : ℝ := 3
noncomputable def chicken_quantity : ℝ := 2
noncomputable def meat_budget_percentage : ℝ := 0.33

noncomputable def total_non_broccoli_cost : ℝ := 
  orange_price * orange_quantity + cabbage_price + bacon_price + chicken_price * chicken_quantity

noncomputable def meat_cost : ℝ := bacon_price + chicken_price * chicken_quantity

noncomputable def total_budget : ℝ := meat_cost / meat_budget_percentage

noncomputable def broccoli_cost : ℝ := total_budget - total_non_broccoli_cost

theorem janet_broccoli_purchase : 
  ⌊broccoli_cost / broccoli_price⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_broccoli_purchase_l1238_123812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1238_123802

def a : ℕ → ℚ
  | 0 => 1  -- Adding a case for 0
  | 1 => 1
  | 2 => 5/3
  | (n+3) => (5/3) * a (n+2) - (2/3) * a (n+1)

def b (n : ℕ) : ℚ := a (n+1) - a n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → b n = (2/3)^n) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 3 - 3 * (2/3)^n) := by
  sorry

#eval a 0
#eval a 1
#eval a 2
#eval a 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1238_123802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_division_l1238_123808

/-- Represents a shape type used in cutting the square -/
structure ShapeType where
  cells : ℕ

/-- The square's dimension -/
def square_dim : ℕ := 120

/-- The total number of cells in the square -/
def total_cells : ℕ := square_dim * square_dim

/-- Predicate to check if a list of ShapeTypes satisfies the condition that 
    the difference in cell count between any two types is divisible by 3 -/
def valid_shape_types (types : List ShapeType) : Prop :=
  ∀ t1 t2, t1 ∈ types → t2 ∈ types → (t1.cells - t2.cells) % 3 = 0

/-- Theorem stating the impossibility of dividing the square into N+5 shapes 
    under the given conditions -/
theorem impossible_division (N : ℕ) (types : List ShapeType) 
  (h_valid : valid_shape_types types) :
  ¬∃ (division : List ShapeType), 
    (division.length = N + 5) ∧ 
    (∀ s, s ∈ division → s ∈ types) ∧
    (division.map ShapeType.cells).sum = total_cells := by
  sorry

#check impossible_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_division_l1238_123808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l1238_123855

/-- The set of points (x, y) on the unit circle generated by r = sin(θ) for 0 ≤ θ ≤ t -/
def circlePoints (t : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ p = (Real.sin θ * Real.cos θ, Real.sin θ * Real.sin θ)}

/-- The unit circle -/
def unitCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- π is the smallest positive real number t such that circlePoints t equals the unit circle -/
theorem smallest_t_for_complete_circle :
  ∀ t > 0, circlePoints t = unitCircle → t ≥ Real.pi ∧
  circlePoints Real.pi = unitCircle := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_complete_circle_l1238_123855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calvins_scorpions_l1238_123880

/-- 
Given Calvin's bug collection:
- He has 12 giant roaches
- He has half as many crickets as roaches
- He has twice as many caterpillars as scorpions
- He has 27 insects in total

Prove that Calvin has 3 scorpions.
-/
theorem calvins_scorpions (scorpions : ℕ) :
  let roaches : ℕ := 12
  let crickets : ℕ := roaches / 2
  let caterpillars : ℕ := 2 * scorpions
  roaches + crickets + scorpions + caterpillars = 27 →
  scorpions = 3 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calvins_scorpions_l1238_123880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_l1238_123825

/-- Ellipse type with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with slope k passing through point S -/
structure Line where
  k : ℝ
  s : Point

/-- Definition of the ellipse C -/
def ellipse_C (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Dot product of two vectors -/
def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

theorem ellipse_equation_and_fixed_point 
  (e : Ellipse) 
  (F1 F2 P : Point) 
  (h_ecc : eccentricity e = Real.sqrt 2 / 2)
  (h_P_dist : distance (Point.mk 0 0) P = Real.sqrt 7 / 2)
  (h_P_dot : dot_product (Point.mk (P.x - F1.x) (P.y - F1.y)) (Point.mk (P.x - F2.x) (P.y - F2.y)) = 3/4)
  : 
  (∀ (p : Point), ellipse_C e p ↔ p.x^2 / 2 + p.y^2 = 1) ∧ 
  (∀ (l : Line), l.s = Point.mk 0 (1/3) → 
    ∃ (A B : Point), ellipse_C e A ∧ ellipse_C e B ∧ 
      dot_product (Point.mk (A.x - 0) (A.y - 1)) (Point.mk (B.x - 0) (B.y - 1)) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_fixed_point_l1238_123825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l1238_123883

theorem divisibility_property (n : ℕ) :
  (n^(n^n + n^3 + 3^n) + 4*n - n^3 + n^2 + 6) % (n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l1238_123883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_l1238_123840

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |x - y| < ε

notation:50 a " ≈ " b => approx_equal a b

/-- The total number of marbles in a collection, given the number of red marbles and relationships between colors. -/
theorem total_marbles (r : ℝ) : ∃ (total : ℝ), approx_equal total (4.47 * r) := by
  let b := r / 1.3  -- blue marbles
  let g := 1.5 * r  -- green marbles
  let y := 0.8 * g  -- yellow marbles
  let total := b + r + g + y
  
  -- Existence of total
  use total
  
  -- Proof that total ≈ 4.47 * r
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marbles_l1238_123840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_minimum_l1238_123876

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := 
  |p.x|

/-- The parabola y^2 = 2x -/
def onParabola (p : Point) : Prop :=
  p.y^2 = 2 * p.x

/-- Point D -/
noncomputable def D : Point :=
  { x := 2, y := 3/2 * Real.sqrt 3 }

theorem parabola_distance_sum_minimum :
  ∀ (p : Point), onParabola p →
    distance p D + distanceToYAxis p ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_sum_minimum_l1238_123876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1238_123851

def problem (x y : ℝ) : Prop :=
  let a : ℝ × ℝ × ℝ := (2, 4, x)
  let b : ℝ × ℝ × ℝ := (2, y, 2)
  (2^2 + 4^2 + x^2 = 36) ∧ 
  (2*2 + 4*y + x*2 = 0) →
  (x + y = -3 ∨ x + y = 1)

theorem problem_solution : ∀ x y : ℝ, problem x y := by
  intro x y
  unfold problem
  intro h
  rcases h with ⟨h1, h2⟩
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1238_123851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_difference_implies_divisibility_l1238_123814

-- State the theorem
theorem lcm_difference_implies_divisibility (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) 
  (h : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) : 
  x ∣ y ∨ y ∣ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_difference_implies_divisibility_l1238_123814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_area_l1238_123874

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  (2 * t.c - t.b) * (cos t.A) = t.a * (cos t.B)

-- Theorem for the first part
theorem angle_A_measure (t : Triangle) 
  (h : satisfiesCondition t) : t.A = π / 3 := by
  sorry

-- Function to calculate the area of the triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * (sin t.A)

-- Theorem for the second part
theorem max_area (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.a = 2) : 
  triangleArea t ≤ Real.sqrt 3 ∧ 
  ∃ (t' : Triangle), satisfiesCondition t' ∧ t'.a = 2 ∧ triangleArea t' = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_area_l1238_123874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1238_123878

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - (1 / 2) * Real.cos (2 * x)

theorem f_properties :
  (∀ x : ℝ, f (π/3 + x) = f (π/3 - x)) ∧ 
  (∀ y : ℝ, y ∈ Set.Icc (-1 : ℝ) 1 ↔ ∃ x ∈ Set.Icc (-π/6 : ℝ) (π/2 : ℝ), f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1238_123878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inflection_point_on_line_l1238_123823

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -4*x + 3*Real.sin x - Real.cos x

-- Define the first derivative of f
noncomputable def f' (x : ℝ) : ℝ := -4 + 3*Real.cos x + Real.sin x

-- Define the second derivative of f
noncomputable def f'' (x : ℝ) : ℝ := -3*Real.sin x + Real.cos x

-- Define the inflection point condition
def is_inflection_point (x₀ : ℝ) : Prop := f'' x₀ = 0

-- Theorem statement
theorem inflection_point_on_line (x₀ : ℝ) (h : is_inflection_point x₀) :
  ∃ (y : ℝ), y = f x₀ ∧ y = -4 * x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inflection_point_on_line_l1238_123823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleConfiguration_l1238_123827

/-- Represents a configuration of folded paper strip -/
def Configuration := List Nat

/-- Represents all possible configurations -/
def AllConfigurations : List Configuration :=
  [[3, 5, 4, 2, 1], [3, 4, 5, 1, 2], [3, 2, 1, 4, 5], [3, 1, 2, 4, 5], [3, 4, 2, 1, 5]]

/-- Checks if a configuration is valid after folding -/
def isValidFold (c : Configuration) : Bool :=
  sorry

theorem impossibleConfiguration :
  ∃! c, c ∈ AllConfigurations ∧ ¬(isValidFold c) ∧ c = [3, 4, 2, 1, 5] := by
  sorry

#check impossibleConfiguration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleConfiguration_l1238_123827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1238_123886

-- Define set A
def A : Set ℝ := {x | x > 1/3}

-- Define set B
def B : Set ℝ := {y | -3 ≤ y ∧ y ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc (1/3) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1238_123886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_l1238_123821

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def asymptote (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0 ∨ x - Real.sqrt 3 * y = 0

noncomputable def eccentricity_ellipse (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

noncomputable def eccentricity_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem eccentricity_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ x y, hyperbola a b x y → asymptote x y) :
  eccentricity_ellipse a b * eccentricity_hyperbola a b = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_product_l1238_123821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_half_l1238_123870

/-- Represents a tiling of a plane with hexagons and triangles -/
structure HexTriTiling where
  s : ℝ  -- Side length of hexagons and triangles

/-- Area of a hexagon -/
noncomputable def hex_area (t : HexTriTiling) : ℝ :=
  (3 * Real.sqrt 3) / 2 * t.s^2

/-- Area of a triangle -/
noncomputable def tri_area (t : HexTriTiling) : ℝ :=
  Real.sqrt 3 / 4 * t.s^2

/-- The fraction of area covered by hexagons in the tiling -/
noncomputable def hexagon_area_fraction (t : HexTriTiling) : ℝ :=
  hex_area t / (hex_area t + 6 * tri_area t)

/-- Theorem stating that hexagons cover 50% of the area in the tiling -/
theorem hexagon_area_half (t : HexTriTiling) :
  hexagon_area_fraction t = 1 / 2 := by
  sorry

#eval "Proof completed."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_half_l1238_123870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_probability_l1238_123863

/-- The number of socks -/
def total_socks : ℕ := 10

/-- The number of colors -/
def num_colors : ℕ := 5

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The number of socks drawn -/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly two pairs of socks with the same color -/
theorem two_pairs_probability : 
  (Nat.choose total_socks socks_drawn : ℚ)⁻¹ * 
  (Nat.choose num_colors 3 * Nat.choose 3 2 * socks_per_color : ℚ) = 5 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_probability_l1238_123863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_proposition_correct_l1238_123846

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β : Plane)

-- Define the propositions
def proposition1 (Line Plane : Type) 
  (perpendicular_planes : Plane → Plane → Prop) 
  (perpendicular_plane : Line → Plane → Prop) 
  (parallel_plane : Line → Plane → Prop) : Prop := 
  ∀ (α β : Plane) (l : Line), 
    perpendicular_planes α β → perpendicular_plane l α → parallel_plane l β

def proposition2 (Line Plane : Type) 
  (perpendicular_planes : Plane → Plane → Prop) 
  (contained_in : Line → Plane → Prop) 
  (perpendicular_plane : Line → Plane → Prop) : Prop := 
  ∀ (α β : Plane) (l : Line), 
    perpendicular_planes α β → contained_in l α → perpendicular_plane l β

def proposition3 (Line : Type) 
  (perpendicular : Line → Line → Prop) 
  (parallel : Line → Line → Prop) : Prop := 
  ∀ (l m n : Line), 
    perpendicular l m → perpendicular m n → parallel l n

def proposition4 (Line Plane : Type) 
  (perpendicular_plane : Line → Plane → Prop) 
  (parallel_plane : Line → Plane → Prop) 
  (parallel_planes : Plane → Plane → Prop) 
  (perpendicular : Line → Line → Prop) : Prop := 
  ∀ (m n : Line) (α β : Plane), 
    perpendicular_plane m α → parallel_plane n β → parallel_planes α β → perpendicular m n

-- Theorem statement
theorem only_one_proposition_correct : 
  ∃ (Line Plane : Type) 
    (perpendicular : Line → Line → Prop)
    (parallel : Line → Line → Prop)
    (perpendicular_plane : Line → Plane → Prop)
    (parallel_plane : Line → Plane → Prop)
    (perpendicular_planes : Plane → Plane → Prop)
    (parallel_planes : Plane → Plane → Prop)
    (contained_in : Line → Plane → Prop),
  ¬(proposition1 Line Plane perpendicular_planes perpendicular_plane parallel_plane) ∧ 
  ¬(proposition2 Line Plane perpendicular_planes contained_in perpendicular_plane) ∧ 
  ¬(proposition3 Line perpendicular parallel) ∧ 
  (proposition4 Line Plane perpendicular_plane parallel_plane parallel_planes perpendicular) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_proposition_correct_l1238_123846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_proof_l1238_123809

theorem city_distance_proof (S : ℕ) : 
  (∀ x : ℕ, x ≤ S → Nat.gcd x (S - x) ∈ ({1, 3, 13} : Set ℕ)) →
  (∀ T : ℕ, T < S → ∃ y : ℕ, y ≤ T ∧ Nat.gcd y (T - y) ∉ ({1, 3, 13} : Set ℕ)) →
  S = 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_proof_l1238_123809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_theorem_l1238_123835

/-- Represents the properties of a car's fuel efficiency -/
structure CarFuelEfficiency where
  city_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ
  highway_city_mpg_difference : ℚ

/-- Calculates the miles per tankful on the highway given a car's fuel efficiency -/
def highway_miles_per_tankful (car : CarFuelEfficiency) : ℚ :=
  let tank_size := car.city_miles_per_tankful / car.city_miles_per_gallon
  let highway_mpg := car.city_miles_per_gallon + car.highway_city_mpg_difference
  highway_mpg * tank_size

/-- Theorem stating that given specific car properties, the highway miles per tankful is 462 -/
theorem highway_miles_theorem (car : CarFuelEfficiency) 
  (h1 : car.city_miles_per_tankful = 336)
  (h2 : car.city_miles_per_gallon = 24)
  (h3 : car.highway_city_mpg_difference = 9) :
  highway_miles_per_tankful car = 462 := by
  -- Unfold the definition of highway_miles_per_tankful
  unfold highway_miles_per_tankful
  -- Simplify the expression using the given hypotheses
  simp [h1, h2, h3]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_miles_theorem_l1238_123835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1238_123884

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def y (x φ : ℝ) : ℝ := f (x + Real.pi/3) φ

theorem function_properties (φ : ℝ) 
  (h1 : -Real.pi/2 < φ ∧ φ < Real.pi/2) 
  (h2 : ∀ x, f x φ = f (8*Real.pi/3 - x) φ) : 
  (∀ x, y x φ = -y (-x) φ) ∧ 
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi/4 → y x₂ φ < y x₁ φ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1238_123884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_one_l1238_123888

theorem triplet_sum_not_one : 
  let t1 : (ℚ × ℚ × ℚ) := (1/4, 2/4, 1/4)
  let t2 : (ℤ × ℤ × ℤ) := (-3, 5, -1)
  let t3 : (ℚ × ℚ × ℚ) := (2/10, 4/10, 4/10)
  let t4 : (ℚ × ℚ × ℚ) := (9/10, -1/2, 3/5)
  (t1.fst + t1.snd.fst + t1.snd.snd = 1) ∧ 
  (t2.fst + t2.snd.fst + t2.snd.snd = 1) ∧ 
  (t3.fst + t3.snd.fst + t3.snd.snd = 1) ∧ 
  (t4.fst + t4.snd.fst + t4.snd.snd ≠ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_one_l1238_123888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1238_123842

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  hsum : A + B + C = π

theorem triangle_properties (t : Triangle) 
  (h : (Real.sin t.A) / (Real.cos t.B * Real.cos t.C) = 2 * t.a^2 / (t.a^2 + t.c^2 - t.b^2)) :
  t.C = π/4 ∧ 
  (∀ (D : ℝ), D ∈ Set.Icc 0 t.a → 
    (t.a - D = 2 * D ∧ Real.cos t.B = 3/5) → 
      Real.tan (Real.arcsin ((Real.sin t.B * t.c) / (t.a - D)) - t.C) = 8/15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1238_123842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_250_l1238_123826

/-- Represents a trapezoid ABCD with midpoints E and F -/
structure Trapezoid :=
  (AB : ℝ) -- Length of side AB
  (CD : ℝ) -- Length of side CD
  (h_EF : ℝ) -- Height from E to CD

/-- The area of trapezoid EFCD given trapezoid ABCD -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  1/2 * (t.CD + (t.AB + t.CD) / 2) * t.h_EF

/-- Theorem stating the area of trapezoid EFCD is 250 square units -/
theorem area_EFCD_is_250 (t : Trapezoid) 
    (h_AB : t.AB = 10) 
    (h_CD : t.CD = 30) 
    (h_EF : t.h_EF = 10) : 
  area_EFCD t = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_250_l1238_123826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_cylinder_volume_l1238_123831

/-- The height of the cone -/
noncomputable def cone_height : ℝ := 27

/-- The base radius of the cone -/
noncomputable def cone_radius : ℝ := 9

/-- The volume of a cylinder inscribed in the cone as a function of its height -/
noncomputable def cylinder_volume (h : ℝ) : ℝ :=
  (Real.pi / 9) * (cone_height - h)^2 * h

/-- The maximum volume of a cylinder inscribed in the cone -/
noncomputable def max_cylinder_volume : ℝ := 324 * Real.pi

theorem max_inscribed_cylinder_volume :
  ∃ (h : ℝ), h > 0 ∧ h < cone_height ∧
  cylinder_volume h = max_cylinder_volume ∧
  ∀ (h' : ℝ), h' > 0 → h' < cone_height → cylinder_volume h' ≤ max_cylinder_volume := by
  sorry

#check max_inscribed_cylinder_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_cylinder_volume_l1238_123831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l1238_123882

/-- The ellipse defined by the equation 4x^2 + y^2 = 2 -/
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 2

/-- The line defined by the equation 2x - y - 8 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y - 8 = 0

/-- The distance from a point (x, y) to the line 2x - y - 8 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (2 * x - y - 8) / Real.sqrt 5

/-- The minimum distance from any point on the ellipse to the line is 6√5/5 -/
theorem min_distance_ellipse_to_line :
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 / 5 ∧
  ∀ (x y : ℝ), ellipse x y → d ≤ distance_to_line x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l1238_123882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_complex_fraction_l1238_123856

noncomputable def complex_distance (z : ℂ) : ℝ :=
  Real.sqrt ((z.re ^ 2) + (z.im ^ 2))

theorem distance_of_complex_fraction : 
  complex_distance (Complex.I / (1 + Complex.I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_of_complex_fraction_l1238_123856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_third_power_solutions_l1238_123873

open Real

theorem sin_eq_third_power_solutions (f : ℝ → ℝ) (h : ∀ x, f x = sin x - (1/3)^x) :
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 < x ∧ x < 150 * π ∧ f x = 0) ∧
    S.card ≥ 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_third_power_solutions_l1238_123873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_four_digit_solution_l1238_123852

theorem least_positive_four_digit_solution :
  let x : ℤ := 1704
  (5 * x) % 20 = 30 % 20 ∧
  (3 * x + 11) % 14 = 20 % 14 ∧
  ((-3 : ℤ) * x + 2) % 35 = x % 35 ∧
  x ≥ 1000 ∧ x < 10000 ∧
  ∀ y : ℤ, y ≥ 1000 ∧ y < 10000 ∧
    (5 * y) % 20 = 30 % 20 ∧
    (3 * y + 11) % 14 = 20 % 14 ∧
    ((-3 : ℤ) * y + 2) % 35 = y % 35 →
    y ≥ x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_four_digit_solution_l1238_123852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_in_cube_l1238_123854

theorem hexagon_in_cube (hexagon_side : ℝ) (cube_side : ℝ) : 
  hexagon_side = 2/3 ∧ cube_side = 1 → 
  ∃ (max_hexagon_side : ℝ), max_hexagon_side = (Real.sqrt 2) / 2 ∧ hexagon_side ≤ max_hexagon_side :=
by
  intro h
  use (Real.sqrt 2) / 2
  constructor
  · rfl
  · have h1 : hexagon_side = 2/3 := h.left
    have h2 : (Real.sqrt 2) / 2 > 2/3 := by sorry
    rw [h1]
    exact le_of_lt h2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_in_cube_l1238_123854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_distance_is_20_l1238_123817

/-- Represents the course Pat took -/
structure Course where
  bicycle_speed : ℝ  -- in miles per hour
  bicycle_time : ℝ   -- in minutes
  run_speed : ℝ      -- in miles per hour
  total_time : ℝ     -- in minutes

/-- Calculate the total distance of the course -/
noncomputable def total_distance (c : Course) : ℝ :=
  let bicycle_distance := c.bicycle_speed * (c.bicycle_time / 60)
  let run_time := c.total_time - c.bicycle_time
  let run_distance := c.run_speed * (run_time / 60)
  bicycle_distance + run_distance

/-- The theorem to prove -/
theorem course_distance_is_20 :
  ∃ (c : Course),
    c.bicycle_speed = 30 ∧
    c.bicycle_time = 12 ∧
    c.run_speed = 8 ∧
    c.total_time = 117 ∧
    total_distance c = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_distance_is_20_l1238_123817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l1238_123860

theorem tan_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 1/3) (h2 : π/2 < α ∧ α < π) : 
  Real.tan (π - α) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_l1238_123860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_count_orange_count_is_23_l1238_123875

/-- Given a bowl with apples and oranges, prove the initial number of oranges. -/
theorem orange_count (initial_apples : ℕ) (oranges_removed : ℕ) : ℕ :=
  let initial_oranges : ℕ := initial_apples + oranges_removed
  have h : initial_apples = initial_oranges - oranges_removed := by sorry
  have : initial_apples = 10 := by sorry
  have : oranges_removed = 13 := by sorry
  initial_oranges

/-- Proof that the initial number of oranges is 23. -/
theorem orange_count_is_23 : orange_count 10 13 = 23 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_count_orange_count_is_23_l1238_123875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_theorem_l1238_123895

/-- Calculates the cost of white washing a room's walls given its dimensions and openings. -/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ) 
                     (doorLength doorWidth : ℝ) 
                     (windowLength windowWidth : ℝ) 
                     (numWindows : ℕ) 
                     (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := doorLength * doorWidth
  let windowArea := windowLength * windowWidth * (numWindows : ℝ)
  let areaToWhitewash := wallArea - doorArea - windowArea
  areaToWhitewash * costPerSquareFoot

/-- Theorem stating the cost of white washing for the given room specifications. -/
theorem whitewashing_cost_theorem : 
  whitewashingCost 25 15 12 6 3 4 3 3 6 = 5436 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_whitewashing_cost_theorem_l1238_123895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_multiplication_l1238_123843

theorem average_after_multiplication (numbers : Fin 7 → ℝ) :
  (Finset.sum Finset.univ numbers / 7 = 26) →
  let new_numbers := λ i => 5 * numbers i
  Finset.sum Finset.univ new_numbers / 7 = 130 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_multiplication_l1238_123843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_judys_shopping_cost_l1238_123897

/-- Represents the shopping items with their quantities and prices -/
structure ShoppingItem where
  quantity : ℕ
  price : ℚ
  discount : ℚ
  deriving Repr

/-- Calculates the total cost of a shopping trip -/
def calculateTotalCost (items : List ShoppingItem) (couponValue : ℚ) (couponThreshold : ℚ) : ℚ :=
  let subtotal := items.foldl (fun acc item => acc + (item.quantity : ℚ) * item.price * (1 - item.discount)) 0
  if subtotal ≥ couponThreshold then subtotal - couponValue else subtotal

/-- Judy's shopping list with discounts applied -/
def judysShoppingList : List ShoppingItem := [
  ⟨6, 1, 1/2⟩,  -- Carrots: buy-one-get-one-free
  ⟨3, 3, 0⟩,    -- Milk: no discount
  ⟨2, 4, 1/2⟩,  -- Pineapples: half price
  ⟨3, 5, 0⟩,    -- Flour: no discount
  ⟨1, 8, 0⟩     -- Ice cream: no discount
]

/-- Theorem stating that Judy spends $33 on her shopping trip -/
theorem judys_shopping_cost :
  calculateTotalCost judysShoppingList 6 30 = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_judys_shopping_cost_l1238_123897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_probability_l1238_123830

noncomputable section

/-- The unit square -/
def unitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- The point (5/8, 3/8) -/
def fixedPoint : ℝ × ℝ := (5/8, 3/8)

/-- The set of points in the unit square that form a line with slope ≥ 1/2 with the fixed point -/
def slopeSet : Set (ℝ × ℝ) :=
  {p ∈ unitSquare | p.1 ≠ fixedPoint.1 ∧ (p.2 - fixedPoint.2) / (p.1 - fixedPoint.1) ≥ 1/2}

/-- The probability measure on the unit square -/
def μ : MeasureTheory.Measure (ℝ × ℝ) := sorry

/-- The theorem stating the probability of selecting a point in the slope set -/
theorem slope_probability : μ slopeSet / μ unitSquare = 43/128 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_probability_l1238_123830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1238_123818

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | (x + 1) * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1238_123818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tara_time_for_target_distance_l1238_123858

-- Define the given conditions
noncomputable def alex_distance : ℝ := 4.5
noncomputable def alex_time : ℝ := 27
noncomputable def tara_initial_distance : ℝ := 3
noncomputable def tara_target_distance : ℝ := 6

-- Define the relationship between Tara's and Alex's initial times
noncomputable def tara_initial_time : ℝ := alex_time / 2

-- Theorem to prove
theorem tara_time_for_target_distance : 
  (tara_target_distance / tara_initial_distance) * tara_initial_time = alex_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tara_time_for_target_distance_l1238_123858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l1238_123859

theorem complex_modulus_example : Complex.abs (-1 + 2*Complex.I) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_example_l1238_123859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1238_123862

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

-- State the theorem
theorem f_properties :
  -- Part I: The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (let T := Real.pi; T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Part II: The range of f(x) on [0, π/2] is [-√3, (2-√3)/2]
  (∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = y) ↔
    y ∈ Set.Icc (-Real.sqrt 3) ((2 - Real.sqrt 3) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1238_123862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_equipment_total_l1238_123871

/-- Represents the equipment distribution and transfers between two sites --/
structure EquipmentDistribution where
  x : ℕ  -- Initial higher-class equipment at first site
  y : ℕ  -- Initial first-class equipment at second site
  higher_class_less : x < y

/-- Calculates the final distribution after transfers --/
def final_distribution (d : EquipmentDistribution) : ℕ × ℕ :=
  let first_after_initial := (7 * d.x) / 10
  let second_after_initial := d.y + (d.x - first_after_initial)
  let second_to_first := second_after_initial / 10
  let first_final := first_after_initial + second_to_first
  let second_final := second_after_initial - second_to_first
  (first_final, second_final)

/-- Theorem stating the total amount of first-class equipment --/
theorem first_class_equipment_total (d : EquipmentDistribution) :
  d.x = 10 ∧ d.y = 17 →
  let (first_final, second_final) := final_distribution d
  first_final = second_final + 6 ∧
  second_final > (d.y * 102) / 100 ∧
  d.y = 17 := by
  sorry

#eval final_distribution { x := 10, y := 17, higher_class_less := by norm_num }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_equipment_total_l1238_123871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_spread_equation_l1238_123800

/-- Represents the number of people infected by each person in each round -/
def x : ℕ := sorry

/-- The total number of infected people after two rounds -/
def total_infected : ℕ := (1 + x)^2

/-- The given total number of infected people after two rounds -/
def given_total : ℕ := 81

/-- Theorem stating that the equation correctly represents the infection spread -/
theorem infection_spread_equation : total_infected = given_total :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infection_spread_equation_l1238_123800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_and_read_aloud_l1238_123803

-- Define a function to represent a fraction
def Fraction (numerator denominator : ℕ) : ℚ := numerator / denominator

-- Define a function to represent how a number is read aloud
def ReadAloud : ℝ → String := sorry

-- Define the specific number we're working with
def number : ℝ := 90.58

-- Theorem statement
theorem fraction_and_read_aloud :
  (Fraction 8 9 = 8 / 9) ∧
  (ReadAloud number = "ninety point five eight") := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_and_read_aloud_l1238_123803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_height_l1238_123861

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- The setup of the kite problem -/
structure KiteProblem where
  O : Point3D -- Center point on the ground
  A : Point3D -- North point
  B : Point3D -- West point
  C : Point3D -- South point
  D : Point3D -- East point
  K : Point3D -- Kite position

  -- Conditions
  north : A.x = O.x ∧ A.y > O.y ∧ A.z = O.z
  west : B.y = O.y ∧ B.x < O.x ∧ B.z = O.z
  south : C.x = O.x ∧ C.y < O.y ∧ C.z = O.z
  east : D.y = O.y ∧ D.x > O.x ∧ D.z = O.z
  kite_above : K.x = O.x ∧ K.y = O.y ∧ K.z > O.z
  ab_distance : distance A B = 100
  ka_length : distance K A = 120
  kb_length : distance K B = 110

theorem kite_height (problem : KiteProblem) : 
  distance problem.O problem.K = 50 * Real.sqrt 3.3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_height_l1238_123861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cubes_from_large_cube_l1238_123810

theorem small_cubes_from_large_cube : 
  ∀ (L : ℝ), L > 0 → 
  (L^3) / ((L/4)^3) = 64 := by
  intro L hL
  have h1 : (L/4)^3 ≠ 0 := by
    apply pow_ne_zero
    exact div_ne_zero (ne_of_gt hL) (by norm_num)
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_cubes_from_large_cube_l1238_123810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_equation_part1_friendly_equation_part2_friendly_equation_part3_l1238_123872

-- Definition of "friendly equation"
def is_friendly_equation (f g : ℝ → Prop) : Prop :=
  ∃ x₀ y₀ : ℝ, f x₀ ∧ g y₀ ∧ x₀ + y₀ = 100

-- Part 1
theorem friendly_equation_part1 :
  is_friendly_equation (λ x ↦ 3*x - 2*x - 102 = 0) (λ y ↦ |y| = 2) := by
  sorry

-- Part 2
theorem friendly_equation_part2 (a : ℝ) :
  is_friendly_equation (λ x ↦ x - (2*x - 2*a)/3 = a + 1) (λ y ↦ |2*y - 2| + 3 = 5) →
  (a = 97 ∨ a = 95) := by
  sorry

-- Part 3
theorem friendly_equation_part3 (m n : ℝ) :
  is_friendly_equation (λ x ↦ m*x + 45*n = 54*m) (λ y ↦ 2*m*|y - 49| + m*(y - 1)/45 = m + n) →
  (m + n)/n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_equation_part1_friendly_equation_part2_friendly_equation_part3_l1238_123872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_small_triangle_l1238_123885

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- A predicate stating that a polygon can be placed inside a unit square -/
def FitsInUnitSquare (n : ℕ) (p : ConvexPolygon n) : Prop :=
  ∀ v : Fin n, let (x, y) := p.vertices v; 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

/-- The area of a triangle given by three points -/
noncomputable def TriangleArea (a b c : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁))

/-- The main theorem -/
theorem convex_polygon_small_triangle (n : ℕ) (p : ConvexPolygon n) 
  (h : FitsInUnitSquare n p) :
  ∃ (i j k : Fin n), TriangleArea (p.vertices i) (p.vertices j) (p.vertices k) ≤ 8 / n^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_small_triangle_l1238_123885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_projection_sides_l1238_123867

/-- A regular dodecahedron -/
structure RegularDodecahedron where

/-- An orthogonal projection of a shape onto a plane -/
def OrthogonalProjection (Shape : Type) := Shape → ℕ

/-- The number of sides in the polygon formed by the orthogonal projection of a dodecahedron -/
def ProjectionSides : OrthogonalProjection RegularDodecahedron → ℕ := sorry

/-- The smallest number of sides in the polygon formed by the orthogonal projection of a dodecahedron is 6 -/
theorem smallest_projection_sides :
  ∃ (proj : OrthogonalProjection RegularDodecahedron),
    ProjectionSides proj = 6 ∧
    ∀ (other_proj : OrthogonalProjection RegularDodecahedron),
      ProjectionSides other_proj ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_projection_sides_l1238_123867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2023_in_third_quadrant_l1238_123839

noncomputable def angle_to_quadrant (angle : ℝ) : ℕ :=
  let reduced_angle := angle % 360
  if 0 ≤ reduced_angle ∧ reduced_angle < 90 then 1
  else if 90 ≤ reduced_angle ∧ reduced_angle < 180 then 2
  else if 180 ≤ reduced_angle ∧ reduced_angle < 270 then 3
  else 4

theorem angle_2023_in_third_quadrant : 
  angle_to_quadrant 2023 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2023_in_third_quadrant_l1238_123839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equal_perimeter_l1238_123896

/-- The area of a rectangle with perimeter equal to a triangle with sides 7.1, 8.9, and 10.0,
    and length twice its width, is approximately 37.54. -/
theorem rectangle_area_equal_perimeter (w : ℝ) (h : ℝ) :
  h = 2 * w →
  2 * (h + w) = 7.1 + 8.9 + 10.0 →
  abs (w * h - 37.54) < 0.01 := by
  intro h_eq_2w perimeter_eq
  -- Here we would normally provide the proof steps
  sorry

#check rectangle_area_equal_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equal_perimeter_l1238_123896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1238_123845

-- Define the linear function
noncomputable def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Theorem statement
theorem inverse_proportion_quadrants
  (k : ℝ)
  (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function k x₁ > linear_function k x₂) :
  ∀ x y : ℝ, x ≠ 0 → y = inverse_proportion k x →
  (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1238_123845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_28_57_percent_l1238_123881

/-- Represents a rectangular field with length and width -/
structure RectangularField where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter path length (sum of length and width) -/
def perimeterPath (field : RectangularField) : ℝ :=
  field.length + field.width

/-- Calculates the diagonal path length using the Pythagorean theorem -/
noncomputable def diagonalPath (field : RectangularField) : ℝ :=
  Real.sqrt (field.length ^ 2 + field.width ^ 2)

/-- Calculates the percentage difference between perimeter and diagonal paths -/
noncomputable def pathDifferencePercentage (field : RectangularField) : ℝ :=
  (perimeterPath field - diagonalPath field) / perimeterPath field * 100

theorem path_difference_approx_28_57_percent (field : RectangularField) 
    (h1 : field.length = 3) 
    (h2 : field.width = 4) : 
    ∃ ε > 0, abs (pathDifferencePercentage field - 28.57) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_difference_approx_28_57_percent_l1238_123881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1238_123891

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - 4)

theorem f_properties :
  (∀ x y : ℝ, 2 < x ∧ x < y → f x > f y) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x ∈ Set.Icc (-6 : ℝ) (-3), f x ≤ -3/16) ∧
  (∀ x ∈ Set.Icc (-6 : ℝ) (-3), f x ≥ -3/5) ∧
  (∃ x ∈ Set.Icc (-6 : ℝ) (-3), f x = -3/16) ∧
  (∃ x ∈ Set.Icc (-6 : ℝ) (-3), f x = -3/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1238_123891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_zero_one_power_l1238_123844

-- Define a as a positive real number
noncomputable def a : ℝ := Real.exp (10 * 0.02)

-- State the theorem
theorem zero_point_zero_one_power : (0.01 : ℝ) ^ (0.01 : ℝ) = 1 / (10 * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_zero_one_power_l1238_123844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_complex_equation_l1238_123893

open Complex

theorem locus_of_complex_equation (z₁ z₂ : ℂ) (l : ℝ) (h : l > 0) :
  let S := {z : ℂ | ‖z - z₁‖ = l * ‖z - z₂‖}
  (l = 1 → ∃ (a b c : ℝ), S = {z : ℂ | (a * z.re + b * z.im = c)}) ∧
  (l ≠ 1 → ∃ (center : ℂ) (radius : ℝ), S = {z : ℂ | ‖z - center‖ = radius}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_complex_equation_l1238_123893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_53_l1238_123833

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n → 1007 ≤ n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_53_l1238_123833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l1238_123841

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line represented by two points -/
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- Check if a point is outside a circle -/
def isOutside (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

/-- Check if a point is between two other points on a line -/
def isBetween (p q r : ℝ × ℝ) : Prop :=
  let (px, py) := p
  let (qx, qy) := q
  let (rx, ry) := r
  (qx - px) * (rx - qx) ≥ 0 ∧ (qy - py) * (ry - qy) ≥ 0

/-- The intersection point of two lines -/
noncomputable def lineIntersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- The symmetric line of a given line with respect to another line -/
noncomputable def symmetricLine (l : Line) (axis : Line) : Line := sorry

/-- Check if a point is on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a point is on a line -/
def onLine (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (x1, y1) := l.p1
  let (x2, y2) := l.p2
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

theorem fixed_intersection_point
  (K : Circle) (A : ℝ × ℝ) (ε : Line)
  (hA : isOutside K A)
  (hε : ε.p1 ≠ A ∨ ε.p2 ≠ A)
  (B C : ℝ × ℝ)
  (hBC : onCircle K B ∧ onCircle K C ∧ isBetween A B C)
  (E D : ℝ × ℝ)
  (hED : onCircle K E ∧ onCircle K D ∧ isBetween A E D)
  (hSymmetric : symmetricLine ε (Line.mk A K.center) = Line.mk E D) :
  ∃ P : ℝ × ℝ, onLine (Line.mk A K.center) P ∧
    P = lineIntersection (Line.mk B D) (Line.mk C E) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_intersection_point_l1238_123841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1238_123864

-- Define the function f(x) = x^(-1/2)
noncomputable def f (x : ℝ) : ℝ := x^(-(1/2 : ℝ))

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x > 0 ∧ f x = y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1238_123864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1238_123877

/-- Calculates the speed in km/h given a distance in meters and time in minutes -/
noncomputable def calculate_speed (distance_m : ℝ) (time_min : ℝ) : ℝ :=
  (distance_m / 1000) / (time_min / 60)

theorem speed_calculation (distance_m time_min : ℝ) 
  (h1 : distance_m = 600)
  (h2 : time_min = 5) :
  calculate_speed distance_m time_min = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l1238_123877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_case_I_case_II_l1238_123836

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (-3, -1)

-- Define vector BC
def vecBC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define a general point D on line BC
def D : ℝ → ℝ × ℝ := λ t => (B.1 + t * vecBC.1, B.2 + t * vecBC.2)

-- Case I: BC = 2BD
theorem case_I : 
  ∃ t : ℝ, vecBC = (2 * (D t).1 - 2 * B.1, 2 * (D t).2 - 2 * B.2) → 
  D (2/3) = (0, 1/2) := by sorry

-- Case II: AD perpendicular to BC
theorem case_II :
  ∃ t : ℝ, ((D t).1 - A.1) * vecBC.1 + ((D t).2 - A.2) * vecBC.2 = 0 → 
  D (5/6) = (9/5, 7/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_case_I_case_II_l1238_123836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1238_123849

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - (1/5)^x

theorem function_properties (x₁ x₂ : ℝ) 
  (h1 : x₁ ≥ 1) (h2 : x₂ ≥ 1) (h3 : x₁ < x₂) : 
  f x₁ > f x₂ ∧ f (Real.sqrt (x₁ * x₂)) > Real.sqrt (f x₁ * f x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1238_123849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_not_q_l1238_123816

theorem proposition_p_and_not_q : (∀ x : ℝ, (2 : ℝ)^x > 0) ∧ ¬(∃ x : ℝ, Real.sin x + Real.cos x > Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_not_q_l1238_123816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_properties_l1238_123837

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_half_properties :
  (∀ x, f (-x) = -f x) ∧  -- odd function
  (∀ x, f (x + 2 * Real.pi) = f x) ∧  -- period is 2π
  (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi) -- minimum positive period
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_properties_l1238_123837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1238_123815

-- Define the ellipse parameters
noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := Real.sqrt 2
noncomputable def e : ℝ := Real.sqrt 6 / 3

-- Define the lower vertex and point P
def D : ℝ × ℝ := (0, -1)
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem ellipse_properties :
  -- The ellipse has foci on x-axis, lower vertex at D, and eccentricity e
  (c^2 = a^2 - b^2) ∧ (e = c/a) ∧ (D.2 = -b) →
  -- 1. Standard equation of the ellipse
  (∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 3 + y^2 = 1)) ∧
  -- 2. Tangent line equation
  (∀ k, (∃ x y, x^2 / 3 + y^2 = 1 ∧ y = k*x + P.2 ∧ 
    ∀ x' y', x'^2 / 3 + y'^2 = 1 → y' ≤ k*x' + P.2) ↔ 
    (k = 1 ∨ k = -1)) ∧
  -- 3. Maximum area of triangle DMN
  (∃ S : ℝ, S = 3 * Real.sqrt 3 / 4 ∧
    ∀ k x₁ y₁ x₂ y₂,
      x₁^2 / 3 + y₁^2 = 1 ∧ y₁ = k*x₁ + P.2 ∧
      x₂^2 / 3 + y₂^2 = 1 ∧ y₂ = k*x₂ + P.2 ∧
      x₁ ≠ x₂ →
      abs ((x₁ - x₂) * (P.2 - D.2) / 2) ≤ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1238_123815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_equality_l1238_123890

theorem algebraic_expression_equality : 
  (3 * Real.sqrt (Real.sqrt (7 + Real.sqrt 48))) / (2 * (Real.sqrt 2 + Real.sqrt 6)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_expression_equality_l1238_123890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lin_journey_time_l1238_123857

/-- Represents Lin's journey with given conditions -/
structure Journey where
  highway_distance : ℝ
  trail_distance : ℝ
  highway_speed_multiplier : ℝ
  trail_time : ℝ

/-- Calculates the total time of the journey -/
noncomputable def total_time (j : Journey) : ℝ :=
  j.trail_time + j.highway_distance / (j.highway_speed_multiplier * (j.trail_distance / j.trail_time))

/-- Theorem stating that Lin's journey takes 90 minutes -/
theorem lin_journey_time :
  ∀ (j : Journey),
    j.highway_distance = 100 ∧
    j.trail_distance = 20 ∧
    j.highway_speed_multiplier = 4 ∧
    j.trail_time = 40 →
    total_time j = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lin_journey_time_l1238_123857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_method_1_correct_distribution_method_2_correct_distribution_method_3_correct_l1238_123838

open BigOperators Finset

-- Define the number of books and people
def num_books : ℕ := 6
def num_people : ℕ := 3

-- Define the distribution methods
def distribution_method_1 : ℕ := (num_books.choose 2) * ((num_books - 2).choose 2)
def distribution_method_2 : ℕ := (num_books.choose 1) * ((num_books - 1).choose 2) * ((num_books - 3).choose 3) * 6
def distribution_method_3 : ℕ := ((num_books.choose 2) * ((num_books - 2).choose 2) * ((num_books - 4).choose 2)) / 6

-- Theorem statements
theorem distribution_method_1_correct : distribution_method_1 = 90 := by sorry

theorem distribution_method_2_correct : distribution_method_2 = 360 := by sorry

theorem distribution_method_3_correct : 
  distribution_method_3 = ((num_books.choose 2) * ((num_books - 2).choose 2) * ((num_books - 4).choose 2)) / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_method_1_correct_distribution_method_2_correct_distribution_method_3_correct_l1238_123838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_truncated_pyramid_volume_l1238_123866

/-- A regular hexagonal truncated pyramid inscribed in a sphere -/
structure HexagonalTruncatedPyramid where
  R : ℝ  -- radius of the sphere
  -- Assume other necessary properties (e.g., inscribed in sphere, base passes through center, etc.)

/-- The volume of a hexagonal truncated pyramid -/
noncomputable def volume (p : HexagonalTruncatedPyramid) : ℝ := (21 * p.R ^ 3) / 16

/-- Theorem: The volume of the specified hexagonal truncated pyramid is (21 * R^3) / 16 -/
theorem hexagonal_truncated_pyramid_volume (p : HexagonalTruncatedPyramid) :
  volume p = (21 * p.R ^ 3) / 16 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_truncated_pyramid_volume_l1238_123866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coal_consumption_theorem_l1238_123822

/-- The amount of coal consumed by engines in metric tonnes -/
def coal_consumption (engines : ℕ) (hours : ℕ) : ℝ := sorry

/-- The number of engines of the first type -/
def engines_type1 : ℕ := 9

/-- The number of hours engines of the first type work per day -/
def hours_type1 : ℕ := 8

/-- The coal consumption of engines of the first type -/
def consumption_type1 : ℝ := 24

/-- The number of engines of the second type -/
def engines_type2 : ℕ := 8

/-- The number of hours engines of the second type work per day -/
def hours_type2 : ℕ := 13

/-- The coal consumption of engines of the second type -/
def consumption_type2 : ℝ := 26

theorem coal_consumption_theorem :
  coal_consumption engines_type2 hours_type2 = consumption_type2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coal_consumption_theorem_l1238_123822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_mod_four_count_l1238_123806

theorem three_digit_mod_four_count : 
  (Finset.filter (fun n : ℕ => 
    100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2) (Finset.range 1000)).card = 225 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_mod_four_count_l1238_123806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l1238_123832

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The property that (n+i)^6 is an integer -/
def is_integer_power (n : ℤ) : Prop :=
  ∃ m : ℤ, (n : ℂ) + i ^ 6 = m

/-- The theorem stating that there's exactly one integer n such that (n+i)^6 is an integer -/
theorem unique_integer_power :
  ∃! n : ℤ, is_integer_power n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l1238_123832
