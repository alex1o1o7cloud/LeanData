import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_Y_l457_45758

theorem calculate_Y (P Q Y : ℝ) : 
  (P = 4012 / 4 ∧ Q = P / 2 ∧ Y = P - Q) → Y = 501.5 := by
  intro h
  have hP : P = 1003 := by
    rw [h.left]
    norm_num
  have hQ : Q = 501.5 := by
    rw [h.right.left, hP]
    norm_num
  rw [h.right.right, hP, hQ]
  norm_num
  
#check calculate_Y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_Y_l457_45758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l457_45708

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- State the theorem
theorem isosceles_triangle (abc : Triangle) 
  (h : Real.sin abc.A = 2 * Real.cos abc.B * Real.sin abc.C) : 
  ∃ (x y : Real), abc.A = x ∧ abc.B = y ∧ abc.C = y ∨ 
                  abc.A = y ∧ abc.B = x ∧ abc.C = y ∨ 
                  abc.A = y ∧ abc.B = y ∧ abc.C = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l457_45708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l457_45750

theorem trig_inequality : 
  Real.cos (2 * π / 7) < Real.sin (5 * π / 7) ∧ Real.sin (5 * π / 7) < Real.tan (2 * π / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l457_45750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l457_45736

-- Define the circle
def is_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def is_tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 4

-- Define point A
def point_A : ℝ × ℝ := (-2, 0)

-- Define point B
def point_B : ℝ × ℝ := (2, 0)

-- Define the geometric sequence condition
def is_geometric_sequence (x y : ℝ) : Prop :=
  (x^2 + y^2) ^ 2 = ((x + 2)^2 + y^2) * ((x - 2)^2 + y^2)

-- Main theorem
theorem trajectory_equation :
  ∀ x y : ℝ,
  is_circle x y →
  (∃ a b : ℝ, is_tangent_line a b) →
  is_geometric_sequence x y →
  x^2 - y^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l457_45736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_symmetric_f_l457_45757

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (x / 2) + a * Real.cos (x / 2)

theorem max_value_of_symmetric_f (a : ℝ) :
  (∀ x : ℝ, f a x = f a (3 * Real.pi - x)) →
  (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) →
  (∃ x : ℝ, f a x = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_symmetric_f_l457_45757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_after_loss_marbles_830_divisible_marbles_solution_l457_45775

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 10) % 3 = 0 ∧
  (n + 10) % 4 = 0 ∧
  (n + 10) % 5 = 0 ∧
  (n + 10) % 6 = 0 ∧
  (n + 10) % 7 = 0 ∧
  (n + 10) % 8 = 0

theorem least_marbles_after_loss : ∀ m : ℕ, 
  is_divisible_by_all m → m ≥ 830 :=
by sorry

theorem marbles_830_divisible : is_divisible_by_all 830 :=
by sorry

theorem marbles_solution : 
  ∃ n, is_divisible_by_all n ∧ ∀ m, is_divisible_by_all m → m ≥ n :=
by
  use 830
  constructor
  · exact marbles_830_divisible
  · exact least_marbles_after_loss

#check marbles_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_after_loss_marbles_830_divisible_marbles_solution_l457_45775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_hourly_rate_l457_45706

/-- Diana's work schedule and earnings for a specific week -/
structure DianaWorkWeek where
  mon_hours : ℕ := 10
  tue_hours : ℕ := 15
  wed_hours : ℕ := 10
  thu_hours : ℕ := 15
  fri_hours : ℕ := 10
  total_earnings : ℚ := 1800
  saturday_earnings : ℚ := 200
  bonus : ℚ := 150
  wed_multiplier : ℕ := 2

/-- Calculates Diana's regular hourly rate based on her work week -/
def regular_hourly_rate (week : DianaWorkWeek) : ℚ :=
  let regular_hours := week.mon_hours + week.tue_hours + week.thu_hours + week.fri_hours
  let double_hours := week.wed_hours * week.wed_multiplier
  let regular_earnings := week.total_earnings - week.saturday_earnings - week.bonus
  regular_earnings / (regular_hours + double_hours : ℚ)

/-- Theorem stating that Diana's regular hourly rate is $1450/70 -/
theorem diana_hourly_rate (week : DianaWorkWeek) :
  regular_hourly_rate week = 1450 / 70 := by
  sorry

def default_week : DianaWorkWeek := {}

#eval regular_hourly_rate default_week

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_hourly_rate_l457_45706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_specific_lines_l457_45737

/-- The cosine of the acute angle between two lines -/
noncomputable def cosine_angle_between_lines (v1 v2 : ℝ × ℝ) : ℝ :=
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let magnitude1 := Real.sqrt (v1.1^2 + v1.2^2)
  let magnitude2 := Real.sqrt (v2.1^2 + v2.2^2)
  dot_product / (magnitude1 * magnitude2)

/-- The first line's direction vector -/
def v1 : ℝ × ℝ := (4, 5)

/-- The second line's direction vector -/
def v2 : ℝ × ℝ := (2, 7)

theorem cosine_specific_lines :
  cosine_angle_between_lines v1 v2 = 43 / (Real.sqrt 41 * Real.sqrt 53) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_specific_lines_l457_45737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l457_45753

theorem simplify_expression : 
  Real.sqrt 3 * (3 : ℝ)^(1/2 : ℝ) + 12 / 3 * 2 - (4 : ℝ)^(3/2 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l457_45753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_line_and_xy_plane_l457_45790

/-- The point where the line passing through (2,3,2) and (4,0,7) intersects the xy-plane -/
noncomputable def intersection_point : ℝ × ℝ × ℝ := (6/5, 21/5, 0)

/-- The first point on the line -/
def point1 : ℝ × ℝ × ℝ := (2, 3, 2)

/-- The second point on the line -/
def point2 : ℝ × ℝ × ℝ := (4, 0, 7)

/-- Theorem stating that the intersection_point is on the line passing through point1 and point2,
    and lies on the xy-plane -/
theorem intersection_point_on_line_and_xy_plane :
  ∃ t : ℝ, intersection_point = point1 + t • (point2 - point1) ∧
  intersection_point.2.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_line_and_xy_plane_l457_45790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_D_approx_l457_45719

noncomputable section

-- Define the perimeter of the smaller square
def small_square_perimeter : ℝ := 8

-- Define the area of the larger square
def large_square_area : ℝ := 81

-- Define the side length of the smaller square
noncomputable def small_square_side : ℝ := small_square_perimeter / 4

-- Define the side length of the larger square
noncomputable def large_square_side : ℝ := Real.sqrt large_square_area

-- Define the horizontal distance between C and D
noncomputable def horizontal_distance : ℝ := large_square_side + small_square_side

-- Define the vertical distance between C and D
noncomputable def vertical_distance : ℝ := large_square_side - small_square_side

-- Theorem to prove
theorem distance_C_to_D_approx : 
  ∃ ε > 0, |Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) - 13.0| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_D_approx_l457_45719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_double_property_l457_45792

def last_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log n 10 + 1

def move_last_to_first (n : ℕ) : ℕ :=
  (last_digit n) * (10 ^ (num_digits (remove_last_digit n))) + (remove_last_digit n)

theorem unique_number_with_double_property :
  ∃! (N : ℕ), 
    N > 0 ∧ 
    last_digit N = 2 ∧ 
    move_last_to_first N = 2 * N ∧
    N = 105263157894736842 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_double_property_l457_45792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l457_45767

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 - (Real.sin x) / (x^4 + x^2 + 1)

-- State the theorem
theorem sum_of_max_and_min_f :
  (⨆ (x : ℝ), f x) + (⨅ (x : ℝ), f x) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l457_45767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_13_l457_45773

-- Define the function f recursively
def f : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (v + 2) => 4^(v+2) * (f (v+1)) - 16^(v+1) * (f v) + v * 2^(v^2)

-- State the theorem
theorem divisibility_by_13 : 
  (13 ∣ f 1989) ∧ (13 ∣ f 1990) ∧ (13 ∣ f 1991) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_13_l457_45773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_decoration_l457_45751

/-- Represents the number of branches allocated to each brother -/
def branches : ℕ := sorry

/-- Represents the number of ornaments allocated to each brother -/
def ornaments : ℕ := sorry

/-- Chuck needs one more branch to hang one ornament per branch -/
axiom chuck_condition : ornaments = branches + 1

/-- Huck has one empty branch when hanging two ornaments per branch -/
axiom huck_condition : 2 * branches = ornaments - 1

/-- The solution is unique and correct -/
theorem christmas_tree_decoration :
  branches = 3 ∧ ornaments = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_decoration_l457_45751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_M_for_probability_condition_l457_45723

/-- The probability function P(M) -/
noncomputable def P (M : ℕ) : ℚ :=
  1 - ((1/3 * (M : ℚ) + 2) / ((M : ℚ) + 1)) ^ 2

/-- M is a multiple of 3 -/
def isMultipleOf3 (M : ℕ) : Prop :=
  ∃ k : ℕ, M = 3 * k

/-- The main theorem -/
theorem least_M_for_probability_condition :
  ∃ M : ℕ, isMultipleOf3 M ∧
    P M < 50/81 ∧
    ∀ N : ℕ, isMultipleOf3 N → P N < 50/81 → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_M_for_probability_condition_l457_45723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l457_45765

/-- Represents a right square pyramid -/
structure RightSquarePyramid where
  upper_base_edge : ℝ
  lower_base_edge : ℝ
  height : ℝ

/-- Calculate the side surface area of a right square pyramid -/
noncomputable def side_surface_area (p : RightSquarePyramid) : ℝ :=
  let slant_height := Real.sqrt ((((p.lower_base_edge - p.upper_base_edge) / 2) ^ 2) + (p.height ^ 2))
  2 * (p.upper_base_edge + p.lower_base_edge) * slant_height

/-- Calculate the volume of a right square pyramid -/
noncomputable def volume (p : RightSquarePyramid) : ℝ :=
  (1 / 3) * (p.upper_base_edge ^ 2 + p.lower_base_edge ^ 2 + 
    Real.sqrt (p.upper_base_edge ^ 2 * p.lower_base_edge ^ 2)) * p.height

/-- Theorem stating the side surface area and volume of the given pyramid -/
theorem pyramid_properties : 
  let p := RightSquarePyramid.mk 4 10 4
  side_surface_area p = 140 ∧ volume p = 208 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l457_45765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_capacity_bounds_l457_45759

/-- The number of trips needed to transport 40 tons -/
def n : ℕ := 5

/-- The truck's load capacity in tons -/
noncomputable def x : ℝ := sorry

/-- The amount of cargo in the last trip when transporting 40 tons -/
noncomputable def last_load_40 : ℝ := sorry

/-- The amount of cargo in the last trip when transporting 80 tons -/
noncomputable def last_load_80 : ℝ := sorry

theorem truck_capacity_bounds :
  (∀ k : ℕ, k < n → k * x ≤ 40) ∧
  (n * x > 40) ∧
  ((n - 1) * x + last_load_40 = 40) ∧
  (0 < last_load_40) ∧
  (last_load_40 ≤ x) ∧
  (∀ k : ℕ, k < n + 5 → k * x ≤ 80) ∧
  ((n + 5) * x > 80) ∧
  ((n + 4) * x + last_load_80 = 80) ∧
  (0 < last_load_80) ∧
  (last_load_80 ≤ x) →
  (7 + 3/11 : ℝ) ≤ x ∧ x < (8 + 8/9 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_capacity_bounds_l457_45759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l457_45734

/-- The function f(u) = 4u^2 / (1 + 4u^2) -/
noncomputable def f (u : ℝ) : ℝ := 4 * u^2 / (1 + 4 * u^2)

/-- The system of equations -/
def system_equations (x y z : ℝ) : Prop :=
  f x = y ∧ f y = z ∧ f z = x

/-- The theorem stating the solutions to the system of equations -/
theorem system_solutions :
  ∀ x y z : ℝ, system_equations x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solutions_l457_45734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_roses_to_remove_l457_45784

theorem yellow_roses_to_remove (red_roses yellow_roses : ℕ) 
  (h1 : red_roses = 30)
  (h2 : yellow_roses = 19) : 
  (yellow_roses : ℤ) - (2 : ℚ) / 7 * (red_roses + yellow_roses : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_roses_to_remove_l457_45784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l457_45776

theorem power_equation_solution (x : ℝ) : (27 : ℝ)^x * (27 : ℝ)^x * (27 : ℝ)^x = (81 : ℝ)^4 → x = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l457_45776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l457_45788

-- Define the concept of an angle being in the second quadrant
def in_second_quadrant (α : Real) : Prop :=
  0 < α ∧ α < Real.pi ∧ Real.sin α > 0 ∧ Real.cos α < 0

-- Define the theorem
theorem angle_in_second_quadrant (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi)  -- α is an interior angle of a triangle
  (h2 : Real.sin α * Real.tan α < 0)    -- given condition
  : in_second_quadrant α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l457_45788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_20gon_symmetry_sum_l457_45716

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : 0 < n

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (n : ℕ) : ℕ := n

/-- The smallest positive angle (in degrees) for which a regular polygon has rotational symmetry -/
noncomputable def smallestRotationAngle (n : ℕ) : ℝ := 360 / n

/-- Theorem: For a regular 20-gon, the sum of its number of lines of symmetry
    and the smallest positive angle (in degrees) for which it has rotational symmetry is 38 -/
theorem regular_20gon_symmetry_sum :
  (linesOfSymmetry 20 : ℝ) + smallestRotationAngle 20 = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_20gon_symmetry_sum_l457_45716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_intersection_l457_45701

/-- The function f(x) = (1/3)x³ + x² + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

/-- Theorem: If f(x) has two extreme points and the line through these points
    intersects the x-axis at a point on f(x), then a = 2/3 -/
theorem extreme_points_intersection (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    f_derivative a x₁ = 0 ∧ 
    f_derivative a x₂ = 0 ∧ 
    (∃ x₀ : ℝ, 
      f a x₀ = 0 ∧ 
      (f a x₀ - f a x₁) / (x₀ - x₁) = (f a x₂ - f a x₁) / (x₂ - x₁))) →
  a = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_intersection_l457_45701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l457_45722

open Set
open Function
open Interval

-- Define the function f
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (Real.log x + (x - t)^2) / x

-- Define the derivative of f with respect to x
noncomputable def f_deriv (t : ℝ) (x : ℝ) : ℝ := (x^2 - 2*t*x + t^2 - Real.log x) / x^2

theorem t_range (t : ℝ) :
  (∀ x ∈ Icc 1 2, f t x > -x * f_deriv t x) → t < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l457_45722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_shifted_data_l457_45785

-- Define the set of positive numbers
variable (x₁ x₂ x₃ : ℝ)
variable (h₁ : x₁ > 0)
variable (h₂ : x₂ > 0)
variable (h₃ : x₃ > 0)

-- Define the variance
noncomputable def variance (x₁ x₂ x₃ : ℝ) : ℝ := (1/3) * (x₁^2 + x₂^2 + x₃^2 - 12)

-- Define the mean of the new data
noncomputable def new_mean (x₁ x₂ x₃ : ℝ) : ℝ := ((x₁ + 1) + (x₂ + 1) + (x₃ + 1)) / 3

-- Theorem statement
theorem mean_of_shifted_data (h : variance x₁ x₂ x₃ = 0) : new_mean x₁ x₂ x₃ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_shifted_data_l457_45785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_l457_45732

/-- The speed of a particle with position (3t + 5, 5t - 7) at time t is √34. -/
theorem particle_speed : ∀ t : ℝ, Real.sqrt 34 = 
  let position := (3 * t + 5, 5 * t - 7)
  let velocity := (3, 5)
  Real.sqrt (velocity.1 ^ 2 + velocity.2 ^ 2) := by
    intro t
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_l457_45732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l457_45714

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The point A -/
def A : ℝ × ℝ := (-2, 0)

/-- The point B -/
def B : ℝ × ℝ := (2, 4)

/-- The point on the x-axis we're proving is equidistant -/
def P : ℝ × ℝ := (2, 0)

theorem equidistant_point :
  distance A.1 A.2 P.1 P.2 = distance B.1 B.2 P.1 P.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l457_45714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_A_B_l457_45724

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 7*x + 10 < 0}

-- State the theorem
theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = Set.Iic 3 ∪ Set.Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_A_B_l457_45724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_group_size_l457_45756

/-- Represents the work capacity of a group of workers --/
noncomputable def WorkCapacity (n : ℕ) (d : ℝ) : ℝ := n / d

theorem original_group_size (d₁ d₂ : ℝ) (n : ℕ) (h₁ : d₁ = 15) (h₂ : d₂ = 18) :
  WorkCapacity n d₁ = WorkCapacity (n - 8) d₂ → n = 40 := by
  sorry

#check original_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_group_size_l457_45756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l457_45704

noncomputable def ω : ℂ := 7 + 3 * Complex.I

theorem complex_modulus_equality : Complex.abs (ω^2 + 4*ω + 40) = 54 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l457_45704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_31_solutions_l457_45726

/-- The number of real solutions to the equation x/50 = cos(x) -/
def num_solutions : ℕ := 31

/-- The equation we're solving -/
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

/-- Theorem stating that the equation has exactly 31 real solutions -/
theorem equation_has_31_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, equation x) ∧ S.card = num_solutions :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_31_solutions_l457_45726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dist_to_focus_is_four_l457_45747

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * para.p * x

/-- Distance from a point to the focus of a parabola -/
noncomputable def distToFocus (para : Parabola) (point : PointOnParabola para) : ℝ :=
  point.x + para.p / 2

/-- Theorem: Distance to focus is 4 under given conditions -/
theorem dist_to_focus_is_four (para : Parabola) 
    (M : PointOnParabola para) 
    (hM : M.y = 4) 
    (hChord : ∃ (y : ℝ), (distToFocus para M)^2 = 7 + (M.x + 1)^2) :
  distToFocus para M = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dist_to_focus_is_four_l457_45747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tails_after_flip_l457_45742

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the circle of coins -/
def CoinCircle (n : ℕ) := Fin (2 * n + 1) → CoinState

/-- The initial state of the coin circle with all coins heads up -/
def initialState (n : ℕ) : CoinCircle n :=
  fun _ => CoinState.Heads

/-- Performs a single flip on a coin -/
def flipCoin (s : CoinState) : CoinState :=
  match s with
  | CoinState.Heads => CoinState.Tails
  | CoinState.Tails => CoinState.Heads

/-- Performs the flipping process on the coin circle -/
def flipProcess (n : ℕ) (circle : CoinCircle n) : CoinCircle n :=
  sorry

/-- Counts the number of tails in the coin circle -/
def countTails (n : ℕ) (circle : CoinCircle n) : ℕ :=
  sorry

/-- Theorem stating that after the flipping process, exactly one coin is tails up -/
theorem one_tails_after_flip (n : ℕ) :
  countTails n (flipProcess n (initialState n)) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tails_after_flip_l457_45742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_probability_l457_45730

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationDuration (observationInterval : ℕ) : ℕ :=
  3 * observationInterval

/-- Theorem: The probability of observing a color change is 3/20 -/
theorem color_change_probability (cycle : TrafficLightCycle) 
    (h1 : cycle.green = 45)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 50)
    (observationInterval : ℕ)
    (h4 : observationInterval = 5) :
  (changeObservationDuration observationInterval : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

#eval cycleDuration ⟨45, 5, 50⟩
#eval changeObservationDuration 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_probability_l457_45730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l457_45799

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

/-- The hyperbola equation x^2/25 - y^2/16 = 1 -/
def is_hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 16 = 1

theorem hyperbola_eccentricity :
  eccentricity 5 4 = Real.sqrt 41 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l457_45799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l457_45709

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + 1/x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 0 ∧ x ≠ -1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l457_45709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l457_45789

theorem sin_double_angle (x : ℝ) :
  Real.sin (π / 4 - x) = 3 / 5 → Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l457_45789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l457_45782

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -2

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y - focus.2 = m * (x - focus.1)

-- Define the intersection points of the line and the parabola
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line_through_focus m p.1 p.2}

-- Define the distance from a point to the directrix
def distance_to_directrix (p : ℝ × ℝ) : ℝ :=
  p.1 - (-2)

-- Main theorem
theorem parabola_intersection_length
  (A B : ℝ × ℝ)
  (m : ℝ)
  (h1 : A ∈ intersection_points m)
  (h2 : B ∈ intersection_points m)
  (h3 : A ≠ B)
  (h4 : distance_to_directrix A = 6) :
  ‖A - B‖ = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l457_45782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_not_neg_187_div_25_l457_45743

/-- Two lines in ℝ³ defined by their parametric equations -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determines if two lines in ℝ³ are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ t u : ℝ, 
    (l1.point.1 + t * l1.direction.1 ≠ l2.point.1 + u * l2.direction.1) ∨
    (l1.point.2.1 + t * l1.direction.2.1 ≠ l2.point.2.1 + u * l2.direction.2.1) ∨
    (l1.point.2.2 + t * l1.direction.2.2 ≠ l2.point.2.2 + u * l2.direction.2.2)

theorem lines_skew_iff_b_not_neg_187_div_25 (b : ℝ) :
  let l1 : Line3D := ⟨(2, 3, b), (3, 4, 5)⟩
  let l2 : Line3D := ⟨(5, 0, -1), (7, 1, 2)⟩
  are_skew l1 l2 ↔ b ≠ -187/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_skew_iff_b_not_neg_187_div_25_l457_45743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_l457_45774

theorem polynomial_constant (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  let P (x : ℝ) := 
    c * (x - a) * (x - b) / ((c - a) * (c - b)) +
    a * (x - b) * (x - c) / ((a - b) * (a - c)) +
    b * (x - c) * (x - a) / ((b - c) * (b - a)) + 1
  ∀ x, P x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_l457_45774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_lease_cost_per_mile_l457_45777

/-- Represents Tom's car leasing scenario -/
structure CarLease where
  miles_short_days : ℕ  -- Miles driven on Mon, Wed, Fri
  miles_long_days : ℕ   -- Miles driven on other days
  weekly_fee : ℚ        -- Weekly fee in dollars
  total_yearly_cost : ℚ -- Total cost for the year in dollars

/-- Calculate the cost per mile for Tom's car lease -/
def cost_per_mile (lease : CarLease) : ℚ :=
  let miles_per_week := 3 * lease.miles_short_days + 4 * lease.miles_long_days
  let miles_per_year := 52 * miles_per_week
  let yearly_fees := 52 * lease.weekly_fee
  let cost_for_miles := lease.total_yearly_cost - yearly_fees
  cost_for_miles / miles_per_year

/-- Theorem stating that the cost per mile for Tom's specific scenario is approximately $0.0909 -/
theorem tom_lease_cost_per_mile :
  let tom_lease : CarLease := {
    miles_short_days := 50,
    miles_long_days := 100,
    weekly_fee := 100,
    total_yearly_cost := 7800
  }
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.0001 ∧ |cost_per_mile tom_lease - 0.0909| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_lease_cost_per_mile_l457_45777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l457_45764

/-- The function for which we want to find the horizontal asymptote -/
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 4) / (4 * x^2 + 6 * x - 3)

/-- Theorem stating that the horizontal asymptote of f(x) is 1.5 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, x > M → |f x - 1.5| < ε := by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l457_45764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_cube_root_l457_45711

theorem segment_length_cube_root : 
  ∃ a b : ℝ, |a - (27 : ℝ)^(1/3)| = 4 ∧ |b - (27 : ℝ)^(1/3)| = 4 ∧ 
  (∀ y : ℝ, |y - (27 : ℝ)^(1/3)| = 4 → (y = a ∨ y = b)) ∧
  |a - b| = 8 :=
by
  -- We'll use 3 as the cube root of 27
  let cubeRoot : ℝ := 3
  
  -- Define the two endpoints
  let a : ℝ := cubeRoot + 4
  let b : ℝ := cubeRoot - 4
  
  -- Prove the existence of a and b
  use a, b
  
  apply And.intro
  · -- Prove |a - (27 : ℝ)^(1/3)| = 4
    sorry
  
  apply And.intro
  · -- Prove |b - (27 : ℝ)^(1/3)| = 4
    sorry
  
  apply And.intro
  · -- Prove that a and b are the only solutions
    sorry
  
  -- Prove |a - b| = 8
  calc
    |a - b| = |(cubeRoot + 4) - (cubeRoot - 4)| := by rfl
    _       = |8| := by ring
    _       = 8 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_cube_root_l457_45711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AEH_l457_45744

/-- Regular octagon with side length 4 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 4)

/-- The area of triangle AEH in a regular octagon -/
noncomputable def triangle_AEH_area (octagon : RegularOctagon) : ℝ :=
  8 * Real.sqrt 2 + 8

/-- Theorem: The area of triangle AEH in a regular octagon with side length 4 is 8√2 + 8 -/
theorem area_triangle_AEH (octagon : RegularOctagon) :
  triangle_AEH_area octagon = 8 * Real.sqrt 2 + 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_AEH_l457_45744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cut_area_l457_45795

/-- The area of a new flat surface created by cutting a cylinder --/
theorem cylinder_cut_area (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  let arc_angle : ℝ := π / 2
  let base_area : ℝ := (1 / 4) * π * r^2
  let stretch_factor : ℝ := h / (r * Real.sqrt 2)
  base_area * stretch_factor = (25 * Real.sqrt 2) / 2 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cut_area_l457_45795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l457_45721

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt ((a + c)^2 + (b + d)^2) ≤ Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ∧
  Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ≤ 
    Real.sqrt ((a + c)^2 + (b + d)^2) + (2 * abs (a * d - b * c)) / Real.sqrt ((a + c)^2 + (b + d)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l457_45721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l457_45780

/-- The minimum area of a triangle with two fixed integer coordinate vertices 
    and the third vertex having integer coordinates. -/
theorem min_triangle_area : ℝ := by
  -- Define the coordinates of points A and B
  let A : ℤ × ℤ := (0, 0)
  let B : ℤ × ℤ := (42, 18)
  
  -- Define the area function for a triangle with integer coordinates
  let area (x y z : ℤ × ℤ) : ℝ :=
    (1/2 : ℝ) * |x.1 * y.2 + y.1 * z.2 + z.1 * x.2 - (x.2 * y.1 + y.2 * z.1 + z.2 * x.1)|

  -- State that the minimum area is 3
  have min_area : (∀ C : ℤ × ℤ, area A B C ≥ 3) ∧ (∃ C : ℤ × ℤ, area A B C = 3) := by
    sorry

  -- Return the minimum area
  exact 3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l457_45780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_function_add_one_gauss_function_sum_inequality_gauss_function_quadratic_inequality_l457_45779

-- Define the Gauss function as noncomputable
noncomputable def gauss_function (x : ℝ) : ℤ := Int.floor x

-- Theorem for statement B
theorem gauss_function_add_one (x : ℝ) :
  gauss_function (x + 1) = gauss_function x + 1 := by sorry

-- Theorem for statement C
theorem gauss_function_sum_inequality (x y : ℝ) :
  gauss_function (x + y) ≥ gauss_function x + gauss_function y := by sorry

-- Theorem for statement D
theorem gauss_function_quadratic_inequality :
  {x : ℝ | 2 * (gauss_function x)^2 - gauss_function x - 3 ≥ 0} =
  Set.Ici 2 ∪ Set.Iio 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_function_add_one_gauss_function_sum_inequality_gauss_function_quadratic_inequality_l457_45779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_approx_l457_45798

/-- Atomic mass of Barium in g/mol -/
noncomputable def atomic_mass_Ba : ℝ := 137.327

/-- Atomic mass of Strontium in g/mol -/
noncomputable def atomic_mass_Sr : ℝ := 87.62

/-- Atomic mass of Bromine in g/mol -/
noncomputable def atomic_mass_Br : ℝ := 79.904

/-- Mass of BaBr2 in the mixture in grams -/
noncomputable def mass_BaBr2 : ℝ := 8

/-- Mass of SrBr2 in the mixture in grams -/
noncomputable def mass_SrBr2 : ℝ := 4

/-- Calculate the mass percentage of Br in a mixture of BaBr2 and SrBr2 -/
noncomputable def mass_percentage_Br : ℝ :=
  let molar_mass_BaBr2 := atomic_mass_Ba + 2 * atomic_mass_Br
  let molar_mass_SrBr2 := atomic_mass_Sr + 2 * atomic_mass_Br
  let mass_Br_in_BaBr2 := (mass_BaBr2 / molar_mass_BaBr2) * (2 * atomic_mass_Br)
  let mass_Br_in_SrBr2 := (mass_SrBr2 / molar_mass_SrBr2) * (2 * atomic_mass_Br)
  let total_mass_Br := mass_Br_in_BaBr2 + mass_Br_in_SrBr2
  let total_mass_mixture := mass_BaBr2 + mass_SrBr2
  (total_mass_Br / total_mass_mixture) * 100

theorem mass_percentage_Br_approx :
  abs (mass_percentage_Br - 57.42) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Br_approx_l457_45798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_fourth_power_l457_45745

variable (n p : ℝ)
variable (r₁ r₂ : ℂ)

def u (n p : ℝ) : ℝ := -(n^4 - 4*p*n^2 + 2*p^2)
def v (p : ℝ) : ℝ := p^4

theorem quadratic_roots_fourth_power (n p : ℝ) (r₁ r₂ : ℂ) :
  (r₁^2 - n*r₁ + p = 0) →
  (r₂^2 - n*r₂ + p = 0) →
  (r₁^4)^2 + u n p * (r₁^4) + v p = 0 ∧ (r₂^4)^2 + u n p * (r₂^4) + v p = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_fourth_power_l457_45745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_formula_l457_45707

/-- The area of a regular hexadecagon inscribed in a circle with radius r -/
noncomputable def hexadecagonArea (r : ℝ) : ℝ := 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)

/-- Theorem stating that the area of a regular hexadecagon inscribed in a circle
    with radius r is equal to 4r^2(√(2 - √2)) square inches -/
theorem hexadecagon_area_formula (r : ℝ) (r_pos : r > 0) :
  hexadecagonArea r = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2) := by
  -- Unfold the definition of hexadecagonArea
  unfold hexadecagonArea
  -- The equation is now trivially true
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_formula_l457_45707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l457_45786

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1)

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := Real.exp x + 1 / (x + 1)

theorem tangent_perpendicular_line (n : ℝ) : 
  (f_deriv 0 * (1 / n) = -1) → n = -2 := by
  intro h
  -- The proof steps would go here
  sorry

#check tangent_perpendicular_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l457_45786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_equals_zero_l457_45768

theorem ab_equals_zero (a b : ℝ) 
  (h : ∀ x : ℝ, b^2 * x^2 + |a| = -(b^2 * (-x)^2 + |a|)) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_equals_zero_l457_45768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_number_divisible_by_737_with_binary_digits_l457_45781

theorem existence_of_number_divisible_by_737_with_binary_digits : 
  ∃ n : ℕ, n > 0 ∧ (737 ∣ n) ∧ (∀ d : ℕ, d ∈ (n.digits 10) → d = 0 ∨ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_number_divisible_by_737_with_binary_digits_l457_45781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l457_45755

theorem sin_alpha_value (α : Real) :
  (∃ P : Real × Real, P.1 = Real.sin (11 * Real.pi / 6) ∧ P.2 = Real.cos (11 * Real.pi / 6) ∧ 
   P.1^2 + P.2^2 = 1 ∧ 
   Real.sin α = P.2) →
  Real.sin α = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l457_45755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roads_for_elza_win_l457_45771

/-- A game where Elza and Susy erase cities from a map -/
structure CityErasingGame where
  total_cities : ℕ
  roads : ℕ

/-- Elza's winning condition -/
def ElzaWins (game : CityErasingGame) : Prop :=
  ∃ (strategy : ℕ → ℕ), 
    (∀ (move : ℕ), move < game.total_cities - 2 → strategy move < game.total_cities - move) ∧
    (game.roads ≥ (game.total_cities - 1) / 2)

/-- The theorem stating the minimum number of roads for Elza's winning strategy -/
theorem min_roads_for_elza_win :
  ∀ (game : CityErasingGame),
    game.total_cities = 2013 →
    (∀ (n : ℕ), n < 1006 → ¬ElzaWins {total_cities := 2013, roads := n}) ∧
    ElzaWins {total_cities := 2013, roads := 1006} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_roads_for_elza_win_l457_45771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_completion_time_l457_45717

/-- Given workers A, B, and C working on a job, prove that C can complete the job in 30 days. -/
theorem worker_c_completion_time 
  (work_rate_a : ℚ) (work_rate_b : ℚ) (work_rate_c : ℚ)
  (ha : work_rate_a = 1 / 30)
  (hb : work_rate_b = 1 / 30)
  (hc : work_rate_c * 10 = 1 - (work_rate_a * 10 + work_rate_b * 10)) :
  work_rate_c = 1 / 30 := by
  sorry

#check worker_c_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_completion_time_l457_45717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l457_45766

/-- The ellipse C with foci F₁(-1,0) and F₂(1,0) -/
structure Ellipse where
  a : ℝ
  h : a > 0

/-- Points A and B on the ellipse -/
structure PointsAB where
  m : ℝ
  n : ℝ

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ := 
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

/-- Helper function to calculate the distance from a point to a line -/
noncomputable def distance_point_to_line (px py x1 y1 x2 y2 : ℝ) : ℝ :=
  abs ((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / 
    Real.sqrt ((y2 - y1)^2 + (x2 - x1)^2)

/-- Theorem about the ellipse C and points A and B -/
theorem ellipse_theorem (C : Ellipse) (P : PointsAB) : 
  -- The equation of ellipse C
  (∀ x y : ℝ, x^2 / C.a^2 + y^2 / 3 = 1 ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  -- The area of triangle ABF₁ when it's isosceles
  (P.m * P.n = 3 ∧ P.m^2 - P.n^2 = 8 → 
    area_triangle (-2) P.m (-1) 0 2 P.n = 5) ∧
  -- The minimum sum of distances from F₁ and F₂ to line AB
  (∃ d : ℝ, d = Real.sqrt 3 * 2 ∧
    ∀ m n : ℝ, m * n = 3 → 
      distance_point_to_line (-1) 0 (-2) m 2 n + 
      distance_point_to_line 1 0 (-2) m 2 n ≥ d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l457_45766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_approximation_l457_45797

/-- The number of trials -/
def n : ℕ := 900

/-- The probability of event A occurring in a single trial -/
def p : ℝ := 0.8

/-- The complementary probability -/
def q : ℝ := 1 - p

/-- The mean of the binomial distribution -/
def μ : ℝ := n * p

/-- The standard deviation of the binomial distribution -/
noncomputable def σ : ℝ := Real.sqrt (n * p * q)

/-- The probability mass function for the binomial distribution -/
def binomialPMF (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p ^ k * q ^ (n - k)

/-- The normal approximation to the binomial distribution -/
noncomputable def normalApprox (k : ℕ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) *
  Real.exp (-(1/2) * ((k : ℝ) - μ)^2 / σ^2)

/-- A custom "near equality" relation for real numbers -/
def isNearEq (x y : ℝ) : Prop := abs (x - y) < 0.0001

/-- The theorem to be proved -/
theorem binomial_approximation :
  isNearEq (normalApprox 750) 0.00146 ∧
  isNearEq (normalApprox 710) 0.0236 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_approximation_l457_45797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_calculation_l457_45760

def initial_bread : ℕ := 200

def day1_fraction : ℚ := 1/4
def day2_fraction : ℚ := 2/5
def day3_fraction : ℚ := 1/2

def remaining_bread : ℕ := 45

theorem bread_calculation :
  let day1_remaining := initial_bread - (initial_bread : ℚ) * day1_fraction
  let day2_remaining := day1_remaining - day1_remaining * day2_fraction
  let day3_remaining := day2_remaining - day2_remaining * day3_fraction
  ⌊day3_remaining⌋ = remaining_bread := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_calculation_l457_45760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l457_45741

theorem sin_minus_cos_value (α : ℝ) 
  (h1 : Real.sin (2 * α) = 1/2) 
  (h2 : 0 < α ∧ α < Real.pi/4) : 
  Real.sin α - Real.cos α = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l457_45741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_f_ge_g_l457_45720

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1) - 2 / x + 1
noncomputable def g (x : ℝ) : ℝ := Real.log x / x + 2

-- State the theorems to be proved
theorem g_max_value : ∃ (x : ℝ), x > 0 ∧ g x = Real.exp (-1) + 2 ∧ ∀ (y : ℝ), y > 0 → g y ≤ g x := by
  sorry

theorem f_ge_g : ∀ (x : ℝ), x > 0 → f x ≥ g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_f_ge_g_l457_45720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_square_inequality_l457_45740

theorem contrapositive_square_inequality :
  (∀ x y : ℝ, x^2 > y^2 → x > y) ↔ (∀ x y : ℝ, x ≤ y → x^2 ≤ y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_square_inequality_l457_45740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_out_of_pocket_result_l457_45727

/-- Calculates the amount Jim is out of pocket after buying two wedding rings and selling one. -/
noncomputable def jim_out_of_pocket (initial_ring_cost : ℝ) (exchange_rate : ℝ) (exchange_fee : ℝ) : ℝ :=
  let desired_ring_cost : ℝ := 2 * initial_ring_cost
  let sold_ring_value : ℝ := initial_ring_cost / 2
  let euros_received : ℝ := sold_ring_value * exchange_rate
  let dollars_before_fee : ℝ := euros_received / exchange_rate
  let fee_amount : ℝ := dollars_before_fee * exchange_fee
  let final_sale_price : ℝ := dollars_before_fee - fee_amount
  initial_ring_cost + desired_ring_cost - final_sale_price

/-- Theorem stating that Jim is out of pocket $25,100 given the problem conditions. -/
theorem jim_out_of_pocket_result :
  jim_out_of_pocket 10000 0.8 0.02 = 25100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_out_of_pocket_result_l457_45727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l457_45791

/-- Represents a point on the railroad track -/
structure Point where
  position : ℝ

/-- Represents the train's journey -/
structure Journey where
  start : Point
  middle : Point
  finish : Point
  speed_first_leg : ℝ
  speed_second_leg : ℝ

/-- Calculates the average speed of the entire journey -/
noncomputable def average_speed (j : Journey) : ℝ :=
  let d₁ := j.middle.position - j.start.position
  let d₂ := j.finish.position - j.middle.position
  let t₁ := d₁ / j.speed_first_leg
  let t₂ := d₂ / j.speed_second_leg
  (d₁ + d₂) / (t₁ + t₂)

theorem train_average_speed (j : Journey) 
  (h₁ : j.middle.position - j.start.position = 2 * (j.finish.position - j.middle.position))
  (h₂ : j.speed_first_leg = 100)
  (h₃ : j.speed_second_leg = 75) :
  average_speed j = 90 := by
  sorry

#eval "This is a placeholder for evaluation."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l457_45791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_D_n_formula_l457_45703

/-- The greatest integer function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The region D_n -/
def D_n (n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 / (n + 1/2) ≤ p.2 ∧ p.2 ≤ (floor (p.1 + 1) : ℝ) - p.1 ∧ p.1 ≥ 0}

/-- The area of D_n -/
noncomputable def area_D_n (n : ℝ) : ℝ := sorry

/-- Theorem: The area of D_n is 1/2 * ((n+3/2)/(n+1/2)) -/
theorem area_D_n_formula (n : ℝ) (hn : n > 0) : 
  area_D_n n = 1/2 * ((n + 3/2) / (n + 1/2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_D_n_formula_l457_45703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_min_a_value_l457_45762

noncomputable section

open Real MeasureTheory

def f (a b x : ℝ) : ℝ := b * x / log x - a * x

theorem f_monotone_intervals (b : ℝ) :
  (∀ x > Real.exp 1, StrictMonoOn (f 0 b) (Set.Ioi (Real.exp 1))) ∧
  (∀ x ∈ Set.Ioo 0 1 ∪ Set.Ioo 1 (Real.exp 1), StrictMonoOn (f 0 b) (Set.Ioo 0 1 ∪ Set.Ioo 1 (Real.exp 1))) :=
sorry

theorem min_a_value :
  let f (x : ℝ) := x / log x - a * x
  let f' (x : ℝ) := (log x - 1) / (log x)^2 - a
  ∃ a : ℝ, a = 1/2 - 1/(4 * (Real.exp 1)^2) ∧
    ∀ a' ≥ a, ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp 1) ((Real.exp 1)^2) ∧ 
              x₂ ∈ Set.Icc (Real.exp 1) ((Real.exp 1)^2) ∧
              f x₁ ≤ f' x₂ + a' :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_min_a_value_l457_45762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l457_45713

theorem smallest_n_for_sqrt_difference : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∀ m : ℕ, m > 0 → m < n → Real.sqrt (m : ℝ) - Real.sqrt ((m : ℝ) - 1) ≥ 0.01) ∧
  (Real.sqrt (n : ℝ) - Real.sqrt ((n : ℝ) - 1) < 0.01) ∧
  n = 2501 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l457_45713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l457_45748

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 4 else 2^(-x) - 4

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_def : ∀ x, x ≥ 0 → f x = 2^x - 4) :
  {a : ℝ | f (a - 2) > 0} = Set.Ioi 4 ∪ Set.Iic 0 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l457_45748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l457_45793

/-- Represents the time it takes to empty a cistern with a leak -/
noncomputable def time_to_empty (normal_fill_time hours_with_leak : ℝ) : ℝ :=
  let fill_rate := 1 / normal_fill_time
  let combined_rate := 1 / hours_with_leak
  let leak_rate := fill_rate - combined_rate
  1 / leak_rate

/-- Theorem stating that for a cistern that fills in 4 hours normally
    and 6 hours with a leak, it takes 12 hours for the leak to empty it -/
theorem cistern_leak_emptying_time :
  time_to_empty 4 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_emptying_time_l457_45793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_101_l457_45712

/-- Given a curve y = x^2 + 1 and points (n, a_n) on the curve for positive integers n,
    prove that a₁₀ = 101. -/
theorem a_10_equals_101 :
  ∃ (a : ℕ+ → ℝ), 
  (∀ (n : ℕ+), (n : ℝ)^2 + 1 = a n) →
  a 10 = 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_101_l457_45712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_three_digit_n_with_arithmetic_progression_l457_45718

/-- 
Given a natural number n, this function returns true if there exist three consecutive
terms in the expansion of (a + 2b)^n whose binomial coefficients form an arithmetic progression,
and false otherwise.
-/
def has_arithmetic_progression_coefficients (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ k < n ∧ 
    2 * (Nat.choose n k) = (Nat.choose n (k-1)) + (Nat.choose n (k+1))

/-- 
Theorem stating that 959 is the largest three-digit positive integer n such that
in the expansion of (a + 2b)^n, there exist three consecutive terms whose
binomial coefficients form an arithmetic progression.
-/
theorem largest_three_digit_n_with_arithmetic_progression : 
  (∀ n : ℕ, n ≤ 999 → has_arithmetic_progression_coefficients n → n ≤ 959) ∧
  has_arithmetic_progression_coefficients 959 := by
  sorry

#check largest_three_digit_n_with_arithmetic_progression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_largest_three_digit_n_with_arithmetic_progression_l457_45718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l457_45752

noncomputable def g (x : ℝ) : ℝ := 3 / (2 * x^8 - 5)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The rest of the proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l457_45752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_monotone_increasing_l457_45735

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem sqrt_monotone_increasing :
  Monotone f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_monotone_increasing_l457_45735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_rice_cost_is_15_l457_45787

/-- Calculates the cost of the second variety of rice given the costs of the first variety and the mixture, and the mixing ratio. -/
noncomputable def second_rice_cost (first_cost mixture_cost : ℝ) (ratio : ℚ) : ℝ :=
  (mixture_cost - first_cost * (ratio / (1 + ratio))) * (1 + 1 / ratio)

/-- Theorem stating that given the specific costs and ratio, the second rice variety costs 15 rs/kg -/
theorem second_rice_cost_is_15 :
  second_rice_cost 6 7.5 (5/6 : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_rice_cost_is_15_l457_45787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l457_45749

noncomputable def curve (x : ℝ) : ℝ := Real.log 7 - Real.log x

noncomputable def interval_start : ℝ := Real.sqrt 3
noncomputable def interval_end : ℝ := Real.sqrt 8

theorem arc_length_theorem :
  ∫ x in interval_start..interval_end, Real.sqrt (1 + (deriv curve x) ^ 2) = 1 + (1/2) * Real.log (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l457_45749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_3_minus_e_l457_45700

noncomputable def f (n : ℕ) : ℝ := ∑' k : ℕ, (1 : ℝ) / ((k : ℝ)^n * k.factorial)

theorem sum_of_f_equals_3_minus_e : ∑' n : ℕ, f (n + 2) = 3 - Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_3_minus_e_l457_45700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_h_minimum_l457_45725

-- Define the functions and their properties
noncomputable def f (x : ℝ) := Real.sqrt (1 + x) + Real.sqrt (1 - x)
noncomputable def g (x : ℝ) := Real.sqrt (1 - x^2)
noncomputable def F (a : ℝ) (x : ℝ) := f x + 2 * a * g x

-- Define the domain of f
def f_domain (x : ℝ) := -1 ≤ x ∧ x ≤ 1

-- Define h(a) as the maximum value of F(x) over the domain of f
noncomputable def h (a : ℝ) := ⨆ (x : ℝ) (hx : f_domain x), F a x

-- State the theorem
theorem f_properties_and_h_minimum :
  (∀ x, f_domain x → f (-x) = f x) ∧ 
  (∀ y, (∃ x, f_domain x ∧ f x = y) ↔ Real.sqrt 2 ≤ y ∧ y ≤ 2) ∧
  (∀ a, a < 0 → h a ≥ Real.sqrt 2) ∧
  (∃ a, a < 0 ∧ h a = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_h_minimum_l457_45725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_equals_height_l457_45733

/-- An equilateral triangle with side length a -/
structure EquilateralTriangle where
  a : ℝ
  a_pos : 0 < a

/-- A point inside an equilateral triangle -/
structure InteriorPoint (T : EquilateralTriangle) where
  x : ℝ
  y : ℝ
  in_triangle : True  -- Placeholder condition, replace with actual condition later

/-- The sum of perpendicular distances from a point to the sides of an equilateral triangle -/
noncomputable def sum_of_perpendiculars (T : EquilateralTriangle) (P : InteriorPoint T) : ℝ :=
  -- Placeholder definition, replace with actual calculation later
  0

/-- The height of an equilateral triangle -/
noncomputable def triangle_height (T : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 2) * T.a

/-- Theorem: The sum of perpendiculars from any interior point to the sides of an equilateral triangle
    is equal to the height of the triangle -/
theorem sum_of_perpendiculars_equals_height (T : EquilateralTriangle) (P : InteriorPoint T) :
  sum_of_perpendiculars T P = triangle_height T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perpendiculars_equals_height_l457_45733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l457_45702

/-- Given a finite universal set U and two subsets A and B, prove that the union of their complements is equal to a specific set. -/
theorem complement_union_problem (U A B : Set Char) : 
  U = {'a', 'b', 'c', 'd'} →
  A = {'a', 'b'} →
  B = {'b', 'c', 'd'} →
  (U \ A) ∪ (U \ B) = {'a', 'c', 'd'} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l457_45702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l457_45796

noncomputable def f (α : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then
    (1/2) * (|x + Real.sin α| + |x + 2 * Real.sin α|) + (3/2) * Real.sin α
  else
    -(1/2) * (|-x + Real.sin α| + |-x + 2 * Real.sin α|) - (3/2) * Real.sin α

theorem f_property (α : ℝ) :
  (∀ x : ℝ, f α x + f α (-x) = 0) ∧
  (∀ x : ℝ, f α (x - 3 * Real.sqrt 3) ≤ f α x) →
  ∃ k : ℤ, 2 * ↑k * Real.pi - Real.pi / 3 ≤ α ∧ α ≤ 2 * ↑k * Real.pi + 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l457_45796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ambiguous_positions_interval_l457_45794

-- Define the clock face with 60 units
def clock_units : ℕ := 60

-- Define the position of the minute hand
def α : ℕ → ℕ := sorry

-- Define the position of the hour hand
def β : ℕ → ℕ := sorry

-- Define the number of full hours passed
def k (t : ℕ) : ℕ := β t / 5

-- Define the relationship between α and β
axiom hand_relationship (t : ℕ) : α t = 12 * β t - k t * clock_units

-- Define the ambiguous position condition
def is_ambiguous (t : ℕ) : Prop :=
  (11 * α t) % clock_units = β t % clock_units ∧
  (11 * β t) % clock_units = α t % clock_units

-- Define the interval between ambiguous positions (in seconds)
def ambiguous_interval : ℕ := 321  -- 5 minutes and 21 seconds in seconds

-- Theorem: Ambiguous positions occur at intervals of 5 minutes and 21 seconds
theorem ambiguous_positions_interval (t₁ t₂ : ℕ) :
  is_ambiguous t₁ ∧ is_ambiguous t₂ ∧ t₁ < t₂ ∧
  (∀ t, t₁ < t ∧ t < t₂ → ¬is_ambiguous t) →
  t₂ - t₁ = ambiguous_interval := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ambiguous_positions_interval_l457_45794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_bounds_l457_45763

noncomputable def inverse_prop_func (k : ℝ) (x : ℝ) : ℝ := (k - 4) / x

def passes_first_third_quadrants (k : ℝ) : Prop :=
  ∀ x, x ≠ 0 → (x > 0 → inverse_prop_func k x > 0) ∧ (x < 0 → inverse_prop_func k x < 0)

def passes_two_points (k a : ℝ) : Prop :=
  ∃ y₁ y₂, y₂ < y₁ ∧ 
    inverse_prop_func k (a + 5) = y₁ ∧
    inverse_prop_func k (2 * a + 1) = y₂

theorem inverse_prop_bounds (k a : ℝ) :
  passes_first_third_quadrants k →
  a > 0 →
  passes_two_points k a →
  k > 4 ∧ a > 4 := by
  sorry

#check inverse_prop_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_bounds_l457_45763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_abc_l457_45739

noncomputable def a : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def b : ℝ := (1/3) ^ (1/5 : ℝ)
noncomputable def c : ℝ := 2 ^ (1/3 : ℝ)

theorem order_abc : a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_abc_l457_45739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_squares_on_circle_is_special_l457_45778

/-- A triangle with squares constructed on its sides --/
structure TriangleWithSquares where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  squareAB : Set (ℝ × ℝ)
  squareBC : Set (ℝ × ℝ)
  squareCA : Set (ℝ × ℝ)

/-- The condition that the outer vertices of squares lie on the same circle --/
def outerVerticesOnCircle (t : TriangleWithSquares) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ p, p ∈ t.squareAB ∪ t.squareBC ∪ t.squareCA →
      (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

/-- Definition of an equilateral triangle --/
def isEquilateral (t : TriangleWithSquares) : Prop :=
  let d := fun p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = d t.B t.C ∧ d t.B t.C = d t.C t.A

/-- Definition of an isosceles right triangle --/
def isIsoscelesRight (t : TriangleWithSquares) : Prop :=
  let d := fun p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d t.A t.B = d t.A t.C ∧ (d t.A t.B)^2 + (d t.A t.C)^2 = (d t.B t.C)^2) ∨
  (d t.B t.A = d t.B t.C ∧ (d t.B t.A)^2 + (d t.B t.C)^2 = (d t.A t.C)^2) ∨
  (d t.C t.A = d t.C t.B ∧ (d t.C t.A)^2 + (d t.C t.B)^2 = (d t.A t.B)^2)

/-- The main theorem --/
theorem triangle_with_squares_on_circle_is_special (t : TriangleWithSquares) :
  outerVerticesOnCircle t → isEquilateral t ∨ isIsoscelesRight t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_squares_on_circle_is_special_l457_45778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_equality_condition_l457_45761

theorem abs_sum_equality_condition :
  (∀ y : ℝ, y ≥ 1 → |y + 1| + |y - 1| = 2*|y|) ∧
  (∃ z : ℝ, z < 1 ∧ |z + 1| + |z - 1| = 2*|z|) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_sum_equality_condition_l457_45761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l457_45746

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) / (x - 2)

theorem range_of_f :
  {x : ℝ | f x ∈ Set.range f} = {x : ℝ | x ≥ -1 ∧ x ≠ 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l457_45746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_conditions_l457_45754

theorem natural_number_conditions (x : ℕ) : 
  (((2 * x > 70) ∧ (3 * x > 25) ∧ (x ≥ 10)) ∧ 
   ¬((x > 100) ∨ (x ≤ 5))) ↔ x = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_number_conditions_l457_45754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_neg_pi_over_24_l457_45728

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) * Real.cos x

-- State the theorem
theorem f_at_neg_pi_over_24 :
  f (-π/24) = (2 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_neg_pi_over_24_l457_45728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_completely_symmetric_set_structure_l457_45705

-- Define a completely symmetric set
def CompletelySymmetricSet (S : Set (EuclideanSpace ℝ (Fin 3))) : Prop :=
  (S.ncard ≥ 3) ∧ 
  ∀ A B : EuclideanSpace ℝ (Fin 3), A ∈ S → B ∈ S → A ≠ B → 
    ∃ (symm : EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3)),
      Isometry symm ∧ symm '' S = S ∧ ∀ x, symm (symm x) = x ∧
      (∀ p, dist p A = dist p B → symm p = p)

-- Define the possible outcomes
inductive RegularShape
  | RegularPolygon (n : ℕ)
  | RegularTetrahedron
  | RegularOctahedron

-- Define a function to get vertices of a regular shape
noncomputable def verticesOf : RegularShape → Set (EuclideanSpace ℝ (Fin 3))
  | RegularShape.RegularPolygon n => sorry
  | RegularShape.RegularTetrahedron => sorry
  | RegularShape.RegularOctahedron => sorry

-- State the theorem
theorem completely_symmetric_set_structure 
  (S : Set (EuclideanSpace ℝ (Fin 3))) 
  (h_sym : CompletelySymmetricSet S) 
  (h_finite : S.Finite) :
  ∃ shape : RegularShape, S = verticesOf shape := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_completely_symmetric_set_structure_l457_45705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l457_45783

theorem sequence_formula (n : ℕ) : 
  ∀ a : ℕ → ℕ, (∀ k, a k = 2^k - 1) → 
    (a 1 = 1) ∧ (a 2 = 3) ∧ (a 3 = 7) ∧ (a 4 = 15) ∧ (a 5 = 31) :=
by
  intro a h
  apply And.intro
  · exact h 1
  apply And.intro
  · exact h 2
  apply And.intro
  · exact h 3
  apply And.intro
  · exact h 4
  · exact h 5

#check sequence_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l457_45783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l457_45729

/-- The function f as defined in the problem -/
noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (Real.pi/2 - α) * Real.cos (3*Real.pi/2 - α) * Real.tan (5*Real.pi + α)) / 
  (Real.tan (-α - Real.pi) * Real.sin (α - 3*Real.pi))

/-- Theorem stating the result of the problem -/
theorem problem_solution (α : ℝ) 
  (h1 : Real.cos (α - 3*Real.pi/2) = 1/5) 
  (h2 : 3*Real.pi/2 < α ∧ α < 2*Real.pi) : 
  f α = -2*Real.sqrt 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l457_45729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l457_45769

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * a * x - 1 / x

theorem tangent_line_and_monotonicity 
  (a : ℝ) 
  (h_a_pos : a > 0) :
  (f_derivative a 1 = -2 → a = 1 ∧ 
    ∃ (y : ℝ → ℝ), (∀ x, y x = -2 * (x - 1)) ∧ (∀ x, 2 * x + y x - 2 = 0)) ∧
  (∀ x > 0, Monotone (f a) ↔ a ≥ 1/8) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l457_45769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_has_more_fanta_l457_45715

-- Define the initial amount of Fanta in Vasya's bottle
variable (a : ℝ)

-- Define the initial amounts of Fanta for both Vasya and Petya
def vasya_initial (a : ℝ) : ℝ := a
def petya_initial (a : ℝ) : ℝ := 1.1 * a

-- Define the remaining amounts after drinking
def vasya_remaining (a : ℝ) : ℝ := vasya_initial a - 0.02 * vasya_initial a
def petya_remaining (a : ℝ) : ℝ := petya_initial a - 0.11 * petya_initial a

-- Theorem stating that Vasya has more Fanta left
theorem vasya_has_more_fanta (a : ℝ) (h : a > 0) : vasya_remaining a > petya_remaining a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_has_more_fanta_l457_45715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statistical_order_correct_order_for_reading_time_study_l457_45770

/-- Represents the steps in a statistical study -/
inductive StatisticalStep
  | SampleSelection
  | DataOrganization
  | DataAnalysis
  | ConclusionDrawing

/-- Defines the correct order of statistical steps -/
def correct_order : List StatisticalStep :=
  [StatisticalStep.SampleSelection, 
   StatisticalStep.DataOrganization, 
   StatisticalStep.DataAnalysis, 
   StatisticalStep.ConclusionDrawing]

/-- The population size for the study -/
def population_size : Nat := 40000

/-- The sample size for the study -/
def sample_size : Nat := 400

/-- Theorem stating that the defined order is correct for determining average daily reading time -/
theorem correct_statistical_order 
  (population : Nat) (sample : Nat)
  (h1 : population > sample)
  (h2 : sample > 0) :
  correct_order = [StatisticalStep.SampleSelection, 
                   StatisticalStep.DataOrganization, 
                   StatisticalStep.DataAnalysis, 
                   StatisticalStep.ConclusionDrawing] :=
by
  rfl

/-- The main theorem proving the correct order for the specific problem -/
theorem correct_order_for_reading_time_study :
  correct_order = [StatisticalStep.SampleSelection, 
                   StatisticalStep.DataOrganization, 
                   StatisticalStep.DataAnalysis, 
                   StatisticalStep.ConclusionDrawing] :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statistical_order_correct_order_for_reading_time_study_l457_45770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l457_45710

theorem min_distance_squared (a b c d : ℝ) 
  (h : (b + a^2 - 3*Real.log a)^2 + (c - d + 2)^2 = 0) : 
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y z w : ℝ), 
    (w + x^2 - 3*Real.log x)^2 + (y - z + 2)^2 = 0 → 
    (x - y)^2 + (w - z)^2 ≥ m := by
  sorry

#check min_distance_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l457_45710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_for_7_4_km_l457_45731

/-- Calculates the taxi fare based on the given policy -/
noncomputable def calculateFare (distance : ℝ) : ℝ :=
  let flatRate := 8
  let flatRateDistance := 3
  let additionalRate := 1.5
  if distance ≤ flatRateDistance then
    flatRate
  else
    flatRate + (distance - flatRateDistance) * additionalRate

/-- Rounds a number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  Int.floor (x + 0.5)

/-- Theorem stating that for a 7.4 km trip, the rounded fare is 15 yuan -/
theorem taxi_fare_for_7_4_km :
  roundToNearest (calculateFare 7.4) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_for_7_4_km_l457_45731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l457_45738

theorem right_triangle_area (a c : ℝ) (h1 : a = 6) (h2 : c = 10) : 
  (1/2) * a * Real.sqrt (c^2 - a^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l457_45738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_15x15_board_l457_45772

/-- Represents a chess board -/
structure ChessBoard :=
  (size : Nat)
  (valid_move : Nat → Bool)
  (no_revisit : Bool)

/-- Defines the maximum number of squares a piece can cover on a chess board -/
def max_squares_covered (board : ChessBoard) : Nat :=
  sorry

/-- Theorem stating the maximum number of squares covered on a 15x15 board -/
theorem max_squares_15x15_board :
  ∀ (board : ChessBoard),
    board.size = 15 ∧
    (∀ n, board.valid_move n ↔ (n = 8 ∨ n = 9)) ∧
    board.no_revisit →
    max_squares_covered board = 196 :=
by
  sorry

#check max_squares_15x15_board

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_15x15_board_l457_45772
