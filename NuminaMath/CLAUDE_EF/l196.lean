import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_ab_product_l196_19693

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + b * x

theorem local_max_implies_ab_product (a b : ℝ) :
  (∀ x, f a b x ≤ f a b (-1)) ∧  -- local maximum at x = -1
  (f a b (-1) = 5/3) →           -- maximum value is 5/3
  a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_implies_ab_product_l196_19693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_color_with_infinite_divisible_points_l196_19642

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a coloring function that assigns a color to each integer
def Coloring := ℤ → Color

-- Define a property for a color to have infinitely many points divisible by k
def InfinitelyManyDivisibleBy (c : Color) (f : Coloring) (k : ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℤ, (m : ℚ) > n ∧ k ∣ m.natAbs ∧ f m = c

-- The main theorem
theorem exists_color_with_infinite_divisible_points (f : Coloring) :
  ∃ c : Color, ∀ k : ℕ, InfinitelyManyDivisibleBy c f k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_color_with_infinite_divisible_points_l196_19642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l196_19619

open Real

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 3 * sin (2 * x - π / 3 + φ)

theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, f φ (abs x) = f φ x) : φ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l196_19619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l196_19605

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: kx + (1-k)y - 3 = 0 -/
noncomputable def slope_l1 (k : ℝ) : ℝ := -k / (1 - k)

/-- The slope of line l2: (k-1)x + (2k+3)y - 2 = 0 -/
noncomputable def slope_l2 (k : ℝ) : ℝ := -(k - 1) / (2*k + 3)

theorem perpendicular_lines (k : ℝ) :
  perpendicular (slope_l1 k) (slope_l2 k) → k = -3 ∨ k = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l196_19605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l196_19687

theorem cos_alpha_value (α : ℝ) (h : Real.cos α = -2/3) : 
  1 / (1 + (Real.tan α)^2) = 4/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l196_19687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_rational_function_l196_19654

theorem integral_of_rational_function (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
  deriv (fun x => Real.log (|x - 2|) - 1 / (2 * (x + 2)^2)) x =
  (x^3 + 6*x^2 + 13*x + 6) / ((x - 2) * (x + 2)^3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_rational_function_l196_19654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_nine_min_value_achieved_l196_19631

/-- A quadratic function f(x) = ax^2 + bx where a > 0 and b > 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The derivative of f(x) at x = 1 is 2 -/
def tangent_slope_condition (f : QuadraticFunction) : Prop :=
  2 * f.a + f.b = 2

/-- The expression to be minimized -/
noncomputable def expression_to_minimize (f : QuadraticFunction) : ℝ :=
  (8 * f.a + f.b) / (f.a * f.b)

/-- The main theorem: if the tangent slope condition is met, 
    the minimum value of the expression is 9 -/
theorem min_value_is_nine (f : QuadraticFunction) 
  (h : tangent_slope_condition f) : 
  (∀ g : QuadraticFunction, expression_to_minimize f ≤ expression_to_minimize g) ∧
  (∃ g : QuadraticFunction, expression_to_minimize g = 9) := by
  sorry

/-- Proof that the minimum value is achieved when a = 1/3 and b = 4/3 -/
theorem min_value_achieved (f : QuadraticFunction) 
  (h : tangent_slope_condition f) :
  ∃ g : QuadraticFunction, g.a = 1/3 ∧ g.b = 4/3 ∧ expression_to_minimize g = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_nine_min_value_achieved_l196_19631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_combined_sectors_l196_19630

/-- The area of two adjacent circular sectors with radius 15, 
    one with a central angle of 30° and the other with a central angle of 45° -/
noncomputable def area_two_sectors (r : ℝ) (angle1 angle2 : ℝ) : ℝ :=
  (angle1 / 360) * Real.pi * r^2 + (angle2 / 360) * Real.pi * r^2

/-- Theorem stating that the area of the combined sectors is 46.875π -/
theorem area_combined_sectors :
  area_two_sectors 15 30 45 = 46.875 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_combined_sectors_l196_19630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l196_19636

-- Define the line
def line (x y : ℝ) : Prop := y - x + 2 = 0

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B
def A : ℝ × ℝ := (5, 3)
def B : ℝ × ℝ := (0, -2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_difference :
  line A.1 A.2 ∧ line B.1 B.2 ∧
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  A ≠ B ∧
  |distance A P - distance B P| = 5 - Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l196_19636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_two_l196_19643

def mySequence : List ℕ := [2, 3, 6, 15, 33, 123]

theorem first_number_is_two : mySequence.head? = some 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_two_l196_19643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_ellipse_circle_intersection_l196_19604

/-- Theorem: Eccentricity range for ellipse-circle intersection
Given an ellipse and a circle with specific properties, if they have four distinct
intersection points, then the eccentricity of the ellipse falls within a specific range. -/
theorem eccentricity_range_ellipse_circle_intersection
  (a b c : ℝ) 
  (h_positive : 0 < b ∧ b < a) 
  (h_ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → ∃ (t : ℝ), x = a * Real.cos t ∧ y = b * Real.sin t)
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 = (b/2 + c)^2 → ∃ (θ : ℝ), x = (b/2 + c) * Real.cos θ ∧ y = (b/2 + c) * Real.sin θ)
  (h_focal : c^2 = a^2 - b^2)
  (h_intersect : ∃ (p1 p2 p3 p4 : ℝ × ℝ), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    (∀ (p : ℝ × ℝ), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 →
      p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ p.1^2 + p.2^2 = (b/2 + c)^2)) :
  let e := c / a
  Real.sqrt 5 / 5 < e ∧ e < 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_ellipse_circle_intersection_l196_19604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_water_amount_l196_19698

/-- Represents the composition of a buffer solution -/
structure BufferComposition where
  concentrate : ℚ
  water : ℚ
  total : ℚ

/-- Calculates the amount of distilled water needed for a given volume of buffer solution -/
def water_needed (original : BufferComposition) (target_volume : ℚ) : ℚ :=
  (original.water / original.total) * target_volume

/-- Theorem stating the correct amount of distilled water needed -/
theorem correct_water_amount :
  let original := BufferComposition.mk (5/100) (3/100) (8/100)
  let target_volume := 64/100
  water_needed original target_volume = 24/100 := by
  sorry

#eval water_needed (BufferComposition.mk (5/100) (3/100) (8/100)) (64/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_water_amount_l196_19698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l196_19633

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > b → (2 : ℝ)^(a+1) > (2 : ℝ)^b) ∧ ¬((2 : ℝ)^(a+1) > (2 : ℝ)^b → a > b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_condition_l196_19633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2018_equals_2_l196_19695

def x : ℕ → ℤ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | k + 2 => x (k + 1) + 1 - 4 * (Int.floor ((k + 2 : ℚ) / 4) - Int.floor ((k + 1 : ℚ) / 4))

theorem x_2018_equals_2 : x 2018 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2018_equals_2_l196_19695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l196_19675

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := l.c / l.b

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := l.c / l.a

/-- The line 3x + 4y = 12 -/
def line : Line := { a := 3, b := 4, c := 12 }

theorem line_intercepts :
  y_intercept line = 3 ∧ x_intercept line = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l196_19675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l196_19640

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi

def hasArithmeticAngles (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

def hasArea (t : Triangle) (area : ℝ) : Prop :=
  1/2 * t.a * t.c * Real.sin t.B = area

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : hasArithmeticAngles t)
  (h3 : hasArea t (Real.sqrt 3 / 2)) :
  t.a * t.c = 2 ∧ 
  (t.b = Real.sqrt 3 ∧ t.a > t.c → t.a = 2 ∧ t.c = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l196_19640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_fraction_after_mixing_l196_19610

/-- Given two bottles with different capacities and oil contents, 
    calculate the fraction of oil in a third bottle after mixing. -/
theorem oil_fraction_after_mixing (C : ℝ) (h1 : C > 0) : 
  (C + C/2) / (C + C/2 + C/2 + 3*C/4) = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_fraction_after_mixing_l196_19610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l196_19678

/-- Given a quadratic function f(x) = ax^2 + 2x + c with its lowest point at (-1, -2),
    this theorem proves properties about the function and a related inequality. -/
theorem quadratic_function_properties (a c : ℝ) 
    (f : ℝ → ℝ) 
    (hf : ∀ x, f x = a * x^2 + 2 * x + c) 
    (hmin : f (-1) = -2 ∧ ∀ x, f x ≥ -2) : 
    (∃ S : Set ℝ, S = {x | x < -4 ∨ x > 2} ∧ ∀ x, f x > 7 ↔ x ∈ S) ∧ 
    (∃ T : Set ℝ, T = Set.Icc 3 (3 + Real.sqrt 2) ∧ 
      ∀ t, t ∈ T ↔ ∀ x ∈ Set.Icc 2 4, f (x - t) ≤ x - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l196_19678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_triangle_l196_19644

theorem min_side_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  c * Real.cos B = a + b / 2 →
  (1 / 2) * a * b * Real.sin C = (Real.sqrt 3 / 12) * c →
  c ≥ 1 ∧ ∃ (a' b' c' : ℝ), c' = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_triangle_l196_19644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_grade_classes_count_l196_19608

def third_grade_class_size : ℕ := 30
def fourth_grade_classes : ℕ := 4
def fourth_grade_class_size : ℕ := 28
def fifth_grade_classes : ℕ := 4
def fifth_grade_class_size : ℕ := 27
def hamburger_cost : ℚ := 2.1
def carrots_cost : ℚ := 0.5
def cookie_cost : ℚ := 0.2
def total_lunch_cost : ℚ := 1036

theorem third_grade_classes_count :
  ∃ (x : ℕ),
    x * third_grade_class_size +
    fourth_grade_classes * fourth_grade_class_size +
    fifth_grade_classes * fifth_grade_class_size =
    (total_lunch_cost / (hamburger_cost + carrots_cost + cookie_cost)).floor ∧
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_grade_classes_count_l196_19608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l196_19634

/-- Represents the time taken to complete a job -/
def CompletionTime := ℝ

/-- Represents the rate at which work is done -/
def WorkRate := ℝ

theorem work_completion_time 
  (time_A : CompletionTime) 
  (time_AB : CompletionTime) 
  (h1 : time_A = (10 : ℝ)) 
  (h2 : time_AB = (5 : ℝ)) : 
  ∃ (time_B : CompletionTime), time_B = (10 : ℝ) := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l196_19634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_l196_19606

/-- Represents a point on a line --/
structure Point where
  x : ℝ

/-- Represents the cyclist --/
structure Cyclist where
  start : Point
  destination : Point

/-- Represents the pedestrian --/
structure Pedestrian where
  start : Point
  destination : Point

/-- The problem setup --/
def problem_setup : Prop :=
  ∃ (A B C : Point) (cyclist : Cyclist) (pedestrian : Pedestrian),
    A.x < B.x ∧ B.x < C.x ∧
    B.x - A.x = 3 ∧
    C.x - B.x = 4 ∧
    cyclist.start = A ∧
    cyclist.destination = C ∧
    pedestrian.start = B ∧
    pedestrian.destination = A

/-- The meeting point theorem --/
theorem meeting_point (setup : problem_setup) :
  ∃ (D : Point) (A B : Point),
    D.x - A.x = 2.1 ∧
    (D.x - A.x) / (B.x - D.x) = 7 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_l196_19606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_is_quintuple_soda_price_l196_19655

/-- Represents the price of a cinema ticket in Brazilian Reais -/
def ticket_price : ℝ := sorry

/-- Represents the price of a soda in Brazilian Reais -/
def soda_price : ℝ := sorry

/-- The number of times Joãozinho went to the cinema -/
def cinema_visits : ℕ := 6

/-- The number of sodas Joãozinho drinks each time he goes to the cinema -/
def sodas_per_visit : ℕ := 2

/-- The total number of sodas Joãozinho drank -/
def total_sodas : ℕ := 20

/-- Joãozinho's total allowance in Brazilian Reais -/
def total_allowance : ℝ := 50

theorem ticket_price_is_quintuple_soda_price :
  (cinema_visits * ticket_price + total_sodas * soda_price = total_allowance) →
  (cinema_visits * sodas_per_visit * soda_price + 
   (total_sodas - cinema_visits * sodas_per_visit) * soda_price = total_allowance) →
  ((cinema_visits + 1) * ticket_price + (cinema_visits + 1) * soda_price = 
   cinema_visits * ticket_price + cinema_visits * sodas_per_visit * soda_price) →
  ticket_price = 5 * soda_price := by
  sorry

#check ticket_price_is_quintuple_soda_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_is_quintuple_soda_price_l196_19655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_value_l196_19659

-- Define the sequence a_n
def a : ℕ → ℤ
  | 0 => -2  -- Add this case to handle n = 0
  | 1 => -2
  | n + 2 => 3 * a (n + 1) - 2

-- Define the partial sum S_n
def S (n : ℕ) : ℚ := (3 / 2) * (a n) + n

-- Define the properties of function f
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (2 - x) = f x)  -- f(2-x) = f(x)

theorem a_2021_value (f : ℝ → ℝ) (h : f_properties f) : f (a 2021) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_value_l196_19659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l196_19645

/-- Proves that for a right cone with given dimensions, the ratio of new height to original height is 9/40 when volume changes -/
theorem cone_height_ratio (circumference : ℝ) (original_height : ℝ) (new_volume : ℝ) :
  circumference = 24 * Real.pi →
  original_height = 40 →
  new_volume = 432 * Real.pi →
  (3 * new_volume / (Real.pi * (circumference / (2 * Real.pi))^2)) / original_height = 9 / 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l196_19645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l196_19692

theorem sequence_convergence (x y : ℕ → ℝ) 
  (hx : ∀ n, x n > 0) 
  (hy : ∀ n, y n > 0) 
  (hx_next : ∀ n, x (n + 1) ≥ (x n + y n) / 2) 
  (hy_next : ∀ n, y (n + 1) ≥ Real.sqrt ((x n ^ 2 + y n ^ 2) / 2)) :
  (∃ l₁, Filter.Tendsto (λ n => x n + y n) Filter.atTop (nhds l₁)) ∧
  (∃ l₂, Filter.Tendsto (λ n => x n * y n) Filter.atTop (nhds l₂)) ∧
  (∃ l, Filter.Tendsto x Filter.atTop (nhds l) ∧ Filter.Tendsto y Filter.atTop (nhds l)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l196_19692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_sin_sin_solution_count_l196_19689

open Real

theorem sin_eq_sin_sin_solution_count :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin 0.9 ∧ sin x = sin (sin x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_sin_sin_solution_count_l196_19689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l196_19648

/-- A power function passing through (4, 1/2) and satisfying f(a+1) < f(10-2a) -/
noncomputable def f (x : ℝ) : ℝ := x^(-(1/4 : ℝ))

theorem power_function_range (a : ℝ) :
  f 4 = 1/2 ∧ 
  (∀ x > 0, f x = x^(-(1/4 : ℝ))) ∧
  f (a + 1) < f (10 - 2*a) →
  3 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l196_19648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l196_19653

noncomputable def vector1 : ℝ × ℝ := (4, 1)
noncomputable def vector2 : ℝ × ℝ := (-1, 3)

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let v_norm_squared := v.1 * v.1 + v.2 * v.2
  ((dot_product / v_norm_squared) * v.1, (dot_product / v_norm_squared) * v.2)

theorem projection_equality (v : ℝ × ℝ) :
  projection vector1 v = projection vector2 v → 
  projection vector1 v = (26/29, 65/29) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equality_l196_19653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_lcm_is_nine_l196_19650

def x (n : ℕ) : ℕ := 2^(2^n) + 1

def sequence_list : List ℕ := List.range 1970 |>.map (fun i => x (i + 2))

theorem last_digit_of_lcm_is_nine :
  (sequence_list.foldl Nat.lcm 1) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_lcm_is_nine_l196_19650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l196_19616

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 5 + a 8 = 15) :
  S a 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l196_19616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_set_intersections_l196_19649

-- Define the universe of triangles
variable (Triangle : Type)

-- Define the properties of triangles
variable (isIsosceles : Triangle → Prop)
variable (isRight : Triangle → Prop)
variable (isAcute : Triangle → Prop)

-- Define the sets
def A (Triangle : Type) (isIsosceles : Triangle → Prop) : Set Triangle := {x | isIsosceles x}
def B (Triangle : Type) (isRight : Triangle → Prop) : Set Triangle := {x | isRight x}
def C (Triangle : Type) (isAcute : Triangle → Prop) : Set Triangle := {x | isAcute x}

-- Define an isosceles right triangle
def isIsoscelesRight (Triangle : Type) (isIsosceles isRight : Triangle → Prop) (t : Triangle) : Prop := 
  isIsosceles t ∧ isRight t

-- State the theorem
theorem triangle_set_intersections (Triangle : Type) 
  (isIsosceles isRight isAcute : Triangle → Prop) :
  (A Triangle isIsosceles ∩ B Triangle isRight = {x : Triangle | isIsoscelesRight Triangle isIsosceles isRight x}) ∧ 
  (B Triangle isRight ∩ C Triangle isAcute = ∅) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_set_intersections_l196_19649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_divided_by_ten_l196_19697

/-- Represents the maximum number of cars passing per hour -/
noncomputable def max_cars_per_hour : ℝ := 3000

/-- Represents the length of each car in meters -/
def car_length : ℝ := 5

/-- Represents the safety distance in car lengths per 10 km/h -/
def safety_distance_ratio : ℝ := 1

/-- Calculates the number of cars passing per hour given the speed in km/h -/
noncomputable def cars_per_hour (speed : ℝ) : ℝ :=
  (3600 * speed) / (car_length * (safety_distance_ratio * speed / 10 + 1))

theorem max_cars_divided_by_ten :
  (⌊max_cars_per_hour⌋ : ℝ) / 10 = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cars_divided_by_ten_l196_19697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l196_19672

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
def b : ℝ × ℝ := (-2, 0)

theorem vector_operations :
  let norm := λ v : ℝ × ℝ => Real.sqrt (v.1^2 + v.2^2)
  let dot := λ v w : ℝ × ℝ => v.1 * w.1 + v.2 * w.2
  let angle := λ v w : ℝ × ℝ => Real.arccos ((dot v w) / (norm v * norm w))
  (norm (a.1 - b.1, a.2 - b.2) = 2 * Real.sqrt 3) ∧
  (angle (a.1 - b.1, a.2 - b.2) a = π / 6) ∧
  ((∀ t : ℝ, norm (a.1 - t * b.1, a.2 - t * b.2) ≥ Real.sqrt 3) ∧
   (∃ t₀ : ℝ, norm (a.1 - t₀ * b.1, a.2 - t₀ * b.2) = Real.sqrt 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l196_19672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_lines_l196_19603

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define point P as the intersection of l₁ and l₂
def P : ℝ × ℝ := (-2, 2)

-- Define the distance function
noncomputable def distance (x y : ℝ) (a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

-- Define the parallel and perpendicular line equations
def parallel_line (x y : ℝ) : Prop := 3 * x - y + 8 = 0
def perpendicular_line (x y : ℝ) : Prop := x + 3 * y - 4 = 0

theorem intersection_point_and_lines :
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧
  (distance P.1 P.2 4 (-3) (-6) = 4) ∧
  (∀ x y, parallel_line x y ↔ y - P.2 = 3 * (x - P.1)) ∧
  (∀ x y, perpendicular_line x y ↔ y - P.2 = (-1/3) * (x - P.1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_and_lines_l196_19603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_vertex_with_low_outdegree_l196_19674

/-- A directed graph with n vertices. -/
structure DirectedGraph (n : ℕ) where
  edges : Fin n → Fin n → Bool

/-- A query function that returns the direction of an edge. -/
def query (G : DirectedGraph n) (u v : Fin n) : Bool := G.edges u v

/-- A function that determines if there exists a vertex with out-degree at most 1. -/
def hasVertexWithOutDegreeAtMostOne (G : DirectedGraph n) : Prop :=
  ∃ v : Fin n, (Finset.sum (Finset.univ : Finset (Fin n)) (fun u => if query G v u then 1 else 0)) ≤ 1

/-- The main theorem stating that we can determine if there exists a vertex
    with out-degree at most 1 using at most 4n queries. -/
theorem determine_vertex_with_low_outdegree (n : ℕ) (h : n ≥ 2) :
  ∃ (f : DirectedGraph n → Bool),
    (∀ G : DirectedGraph n, f G = hasVertexWithOutDegreeAtMostOne G) ∧
    (∀ G : DirectedGraph n, (Finset.sum (Finset.univ : Finset (Fin n × Fin n))
      (fun ⟨u, v⟩ => if f G = query G u v then 1 else 0)) ≤ 4 * n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_vertex_with_low_outdegree_l196_19674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l196_19601

theorem trig_identity (θ c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : Real.tan θ ^ 4 / c + (1 / Real.tan θ) ^ 4 / d = 1 / (c + d)) :
  Real.tan θ ^ 8 / c ^ 3 + (1 / Real.tan θ) ^ 8 / d ^ 3 = (c ^ 5 + d ^ 5) / (c * d) ^ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l196_19601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l196_19679

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The equation of asymptotes for a hyperbola -/
def asymptote_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  λ x y ↦ (h.b / h.a) * x = y ∨ (h.b / h.a) * x = -y

theorem hyperbola_asymptotes (h : Hyperbola) (h_ecc : eccentricity h = 2) :
  asymptote_equation h = λ x y ↦ Real.sqrt 3 * x = y ∨ Real.sqrt 3 * x = -y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l196_19679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_equality_l196_19611

-- Define functions A and B
noncomputable def A (x : ℝ) : ℝ := 3 * Real.sqrt x
def B (x : ℝ) : ℝ := x^3

-- State the theorem
theorem nested_function_equality : A (B (A (B (A (B 2))))) = 792 * (6 ^ (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_equality_l196_19611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_purchase_optimal_l196_19682

/-- Parameters for the rice purchasing problem -/
structure RicePurchaseParams where
  price : ℝ  -- Price per ton of rice
  fee : ℝ    -- Transportation and labor fee per purchase
  daily_consumption : ℝ  -- Daily rice consumption in tons
  storage_cost : ℝ  -- Storage cost per ton per day
  discount_threshold : ℝ  -- Minimum purchase amount for discount in tons
  discount_rate : ℝ  -- Discount rate as a decimal

/-- Calculate the average daily cost without discount -/
noncomputable def avg_daily_cost (params : RicePurchaseParams) (n : ℝ) : ℝ :=
  n + params.fee / n + params.price * params.daily_consumption

/-- Calculate the average daily cost with discount -/
noncomputable def avg_daily_cost_with_discount (params : RicePurchaseParams) (m : ℝ) : ℝ :=
  m + params.fee / m + params.price * params.daily_consumption * (1 - params.discount_rate)

/-- Main theorem for the rice purchasing problem -/
theorem rice_purchase_optimal (params : RicePurchaseParams) :
  (params.price = 1500) →
  (params.fee = 100) →
  (params.daily_consumption = 1) →
  (params.storage_cost = 2) →
  (params.discount_threshold = 20) →
  (params.discount_rate = 0.05) →
  (∃ (n : ℝ), n = 10 ∧ 
    ∀ (x : ℝ), x > 0 → avg_daily_cost params n ≤ avg_daily_cost params x) ∧
  (∃ (m : ℝ), m ≥ params.discount_threshold ∧ 
    avg_daily_cost_with_discount params m < avg_daily_cost params 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_purchase_optimal_l196_19682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_cable_payment_l196_19673

noncomputable def cable_cost (first_100_cost : ℝ) (total_channels : ℕ) : ℝ :=
  if total_channels ≤ 100 then first_100_cost
  else first_100_cost + (first_100_cost / 2)

noncomputable def james_share (total_cost : ℝ) : ℝ := total_cost / 2

theorem james_cable_payment :
  let first_100_cost : ℝ := 100
  let total_channels : ℕ := 200
  let total_cost := cable_cost first_100_cost total_channels
  james_share total_cost = 75 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_cable_payment_l196_19673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_an_over_n_l196_19638

def a : ℕ → ℚ
  | 0 => 33  -- Add this case to cover Nat.zero
  | 1 => 33
  | n+2 => (1/2) * (n+2)^2 - (1/2) * (n+2) + 33

theorem min_an_over_n : 
  ∀ n : ℕ, n ≥ 1 → a n / n ≥ a 8 / 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_an_over_n_l196_19638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_ratio_is_four_to_one_l196_19668

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  total_wings : Nat
  painting_wings : Nat
  artifacts_per_wing : Nat
  large_paintings : Nat
  small_paintings_per_wing : Nat

/-- Calculates the ratio of artifacts to paintings in the museum -/
def artifact_to_painting_ratio (m : Museum) : ℚ :=
  let artifact_wings := m.total_wings - m.painting_wings
  let total_artifacts := artifact_wings * m.artifacts_per_wing
  let total_paintings := m.large_paintings + (m.painting_wings - m.large_paintings) * m.small_paintings_per_wing
  (total_artifacts : ℚ) / total_paintings

/-- Theorem stating that for the given museum configuration, the ratio of artifacts to paintings is 4:1 -/
theorem museum_ratio_is_four_to_one :
  let m : Museum := {
    total_wings := 8,
    painting_wings := 3,
    artifacts_per_wing := 20,
    large_paintings := 1,
    small_paintings_per_wing := 12
  }
  artifact_to_painting_ratio m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_ratio_is_four_to_one_l196_19668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l196_19677

theorem k_value : ∃ k : ℕ, 3 * 6 * 4 * k = Nat.factorial 8 ∧ k = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l196_19677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_difference_maximum_l196_19691

def N : ℕ := 10^6

theorem binomial_difference_maximum (a : ℕ) (h : a ≤ N - 1) :
  Nat.choose N (a + 1) - Nat.choose N a ≤ Nat.choose N 499500 - Nat.choose N 499499 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_difference_maximum_l196_19691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_l196_19658

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C)

theorem angle_B_is_30_degrees 
  (a b : ℝ) (A B : ℝ) 
  (h1 : a = 2 * Real.sqrt 2) 
  (h2 : b = 2) 
  (h3 : A = Real.pi / 4) 
  (h4 : triangle_ABC a b (a * Real.sin A / Real.sin B) A B (Real.pi - A - B)) : 
  B = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_l196_19658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l196_19641

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x + 3*Real.sin x + 4

-- State the theorem
theorem f_symmetry (a : ℝ) (h : f 1 = a) : f (-1) = 8 - a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l196_19641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l196_19681

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_of_sine_function (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π/2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_odd : ∀ x, f ω φ (x + π/6) = -f ω φ (-x + π/6)) :
  ∀ x, f ω φ (x + π/12) = -f ω φ (-x - π/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l196_19681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19635

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
    T = Real.pi ∧
  ∃ (M : ℝ),
    (∀ (x : ℝ), f x ≤ M) ∧
    M = Real.sqrt 2 - 1 ∧
  ∀ (x : ℝ),
    f x = M ↔ ∃ (k : ℤ), x = Real.pi / 8 + k * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19663

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

-- Theorem statement
theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The minimum value on [0, π/2] is 0
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 0 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≥ 0) ∧
  -- The maximum value on [0, π/2] is 1 + √2
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1 + Real.sqrt 2 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ 1 + Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l196_19690

/-- The trajectory of point P given the conditions of the problem -/
theorem trajectory_of_P (A B P : ℝ × ℝ) (h1 : ‖A - B‖ = 3) 
  (h2 : A.2 = 0) (h3 : B.1 = 0) (h4 : P = (1/3 : ℝ) • A + (2/3 : ℝ) • B) :
  P.1^2 + P.2^2/4 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l196_19690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_l196_19669

/-- Represents a point on the circular track -/
structure Point where
  label : Fin 2022

/-- The circular track with 2022 equally spaced points -/
structure Track where
  points : Fin 2022 → Point

/-- The path Bunbun takes, represented as a sequence of points -/
def BunbunPath (track : Track) : List Point :=
  (List.range 2022).map track.points ++ [track.points 0]

/-- The length of the shorter arc between two points on the track -/
noncomputable def shorterArcLength (p q : Point) : ℝ := sorry

/-- The total length of Bunbun's path -/
noncomputable def pathLength (path : List Point) : ℝ :=
  List.sum (List.zipWith shorterArcLength path path.tail)

/-- The main theorem stating the maximal possible sum of arc lengths -/
theorem max_path_length (track : Track) : 
  ∃ (labeling : Fin 2022 → Fin 2022), 
    pathLength (BunbunPath (⟨fun i => track.points (labeling i)⟩)) ≤ 2042222 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_l196_19669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l196_19688

-- Define the centers of the circles
variable (A B C : ℝ × ℝ)

-- Define the radii of the circles
def r_A : ℝ := 2
def r_B : ℝ := 4
def r_C : ℝ := 5

-- Define the horizontal line m
def m : ℝ → ℝ := λ x ↦ 0

-- Axioms based on the problem conditions
axiom tangent_to_m : 
  A.2 = r_A ∧ B.2 = r_B ∧ C.2 = r_C

axiom internal_tangent_AB : 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (r_A - r_B)^2

axiom external_tangent_BC : 
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (r_B + r_C)^2

-- Theorem to prove
theorem area_of_triangle_ABC : 
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  area = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l196_19688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_2_l196_19627

/-- The first circle: x^2 + y^2 - 4 = 0 -/
def circle1 (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 4

/-- The second circle: x^2 + y^2 - 4x + 4y - 12 = 0 -/
def circle2 (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 - 4*p.1 + 4*p.2 = 12

/-- The length of the common chord of two circles -/
noncomputable def common_chord_length (c1 c2 : (ℝ × ℝ) → Prop) : ℝ := sorry

theorem common_chord_length_is_2_sqrt_2 :
  common_chord_length circle1 circle2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_2_l196_19627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19620

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi/3)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧  -- Smallest positive period is 2π
  (∃ y, f y > f (Real.pi/2)) ∧  -- f(π/2) is not the maximum value
  (∀ x, f x = Real.sin (x + Real.pi/3)) := by  -- f(x) = sin(x + π/3) for all x
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_hourly_rate_l196_19666

/-- Madeline's financial situation --/
structure MadelineFinances where
  rent : ℚ
  groceries : ℚ
  medical : ℚ
  utilities : ℚ
  emergency : ℚ
  hours : ℚ

/-- Calculate Madeline's hourly rate --/
def hourlyRate (f : MadelineFinances) : ℚ :=
  (f.rent + f.groceries + f.medical + f.utilities + f.emergency) / f.hours

/-- Theorem: Madeline's hourly rate is approximately $14.93 --/
theorem madeline_hourly_rate :
  let f : MadelineFinances := {
    rent := 1200,
    groceries := 400,
    medical := 200,
    utilities := 60,
    emergency := 200,
    hours := 138
  }
  ∃ (ε : ℚ), ε > 0 ∧ |hourlyRate f - 14.93| < ε
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_hourly_rate_l196_19666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_correct_l196_19639

-- Define a vector space V with a norm
variable {V : Type*} [NormedAddCommGroup V]

-- Define the original proposition
def original_proposition (a b : V) : Prop :=
  a = -b → ‖a‖ = ‖b‖

-- Define the inverse proposition
def inverse_proposition (a b : V) : Prop :=
  ‖a‖ = ‖b‖ → a = -b

-- State the theorem
theorem inverse_proposition_correct :
  (∀ a b : V, original_proposition a b) ↔ (∀ a b : V, inverse_proposition a b) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proposition_correct_l196_19639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_neg_x_over_3_l196_19607

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x/3)

-- State the theorem
theorem derivative_of_exp_neg_x_over_3 :
  deriv f = λ x ↦ -1/3 * Real.exp (-x/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_neg_x_over_3_l196_19607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l196_19680

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - x ≥ 0}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l196_19680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_croissant_orange_price_ratio_l196_19602

/-- The ratio of the cost of a croissant to an orange -/
noncomputable def croissant_orange_ratio (c o : ℝ) : ℝ := c / o

theorem croissant_orange_price_ratio :
  ∀ c o : ℝ,
  c > 0 → o > 0 →
  5 * c + 4 * o > 0 →
  3 * (5 * c + 4 * o) = 4 * c + 10 * o →
  croissant_orange_ratio c o = 2 / 11 :=
by
  intros c o hc ho h1 h2
  unfold croissant_orange_ratio
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check croissant_orange_price_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_croissant_orange_price_ratio_l196_19602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_similar_stars_count_n_is_product_of_first_five_primes_l196_19670

/-- The product of the first five primes -/
def n : ℕ := 2310

/-- The set of integers from 1 to n -/
def S : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

/-- The function to count non-similar regular n-pointed stars -/
def count_non_similar_stars (n : ℕ) : ℕ :=
  (n.totient - 2) / 2

/-- Theorem stating the number of non-similar regular 2310-pointed stars -/
theorem non_similar_stars_count : count_non_similar_stars n = 224 := by
  sorry

/-- The first five primes -/
def first_five_primes : List ℕ := [2, 3, 5, 7, 11]

/-- Theorem stating that n is the product of the first five primes -/
theorem n_is_product_of_first_five_primes : n = first_five_primes.prod := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_similar_stars_count_n_is_product_of_first_five_primes_l196_19670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_with_geometric_subsequence_implies_rational_l196_19609

/-- An arithmetic sequence with first term A and common difference d -/
structure ArithmeticSequence (α : Type*) [Add α] [Zero α] where
  A : α
  d : α
  d_neq_zero : d ≠ 0

/-- A geometric sequence -/
structure GeometricSequence (α : Type*) [Mul α] where
  first : α
  ratio : α

/-- A subsequence of an arithmetic sequence -/
def IsSubsequenceOf (geom : GeometricSequence ℝ) (arith : ArithmeticSequence ℝ) : Prop :=
  ∃ (f : ℕ → ℕ), Monotone f ∧
    ∀ n, ∃ k, geom.first * geom.ratio ^ n = arith.A + (f k : ℝ) * arith.d

theorem arithmetic_with_geometric_subsequence_implies_rational
  (arith : ArithmeticSequence ℝ)
  (geom : GeometricSequence ℝ)
  (h : IsSubsequenceOf geom arith) :
  ∃ (q : ℚ), arith.A / arith.d = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_with_geometric_subsequence_implies_rational_l196_19609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matching_color_probability_l196_19696

/-- Represents the different colors of jelly beans -/
inductive Color where
  | Green
  | Red
  | Yellow
  | Blue
deriving Repr

/-- Represents a person's jelly bean collection -/
structure JellyBeanCollection where
  green : Nat
  red : Nat
  yellow : Nat
  blue : Nat

/-- Abe's jelly bean collection -/
def abe : JellyBeanCollection :=
  { green := 2, red := 1, yellow := 0, blue := 0 }

/-- Bob's jelly bean collection -/
def bob : JellyBeanCollection :=
  { green := 2, red := 2, yellow := 1, blue := 1 }

/-- Calculates the total number of jelly beans in a collection -/
def totalJellyBeans (collection : JellyBeanCollection) : Nat :=
  collection.green + collection.red + collection.yellow + collection.blue

/-- Calculates the probability of picking a specific color from a collection -/
def probabilityOfColor (collection : JellyBeanCollection) (color : Color) : Rat :=
  match color with
  | Color.Green => collection.green / (totalJellyBeans collection)
  | Color.Red => collection.red / (totalJellyBeans collection)
  | Color.Yellow => collection.yellow / (totalJellyBeans collection)
  | Color.Blue => collection.blue / (totalJellyBeans collection)

/-- Theorem: The probability of Abe and Bob showing jelly beans of the same color is 1/3 -/
theorem matching_color_probability :
  (probabilityOfColor abe Color.Green * probabilityOfColor bob Color.Green) +
  (probabilityOfColor abe Color.Red * probabilityOfColor bob Color.Red) = 1/3 := by
  sorry

#eval probabilityOfColor abe Color.Green -- For testing
#eval probabilityOfColor bob Color.Red -- For testing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matching_color_probability_l196_19696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pigment_is_40_percent_l196_19651

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue_percent : ℝ
  red_percent : ℝ
  yellow_percent : ℝ

/-- Calculates the percentage of blue pigment in the brown paint -/
def blue_pigment_percentage (
  maroon : PaintMixture)
  (green : PaintMixture)
  (brown_weight : ℝ)
  (red_pigment_weight : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the percentage of blue pigment in the brown paint is 40% -/
theorem blue_pigment_is_40_percent (
  maroon : PaintMixture)
  (green : PaintMixture)
  (brown_weight : ℝ)
  (red_pigment_weight : ℝ)
  (h1 : maroon.blue_percent = 0.5)
  (h2 : maroon.red_percent = 0.5)
  (h3 : green.blue_percent = 0.3)
  (h4 : green.yellow_percent = 0.7)
  (h5 : brown_weight = 10)
  (h6 : red_pigment_weight = 2.5) :
  blue_pigment_percentage maroon green brown_weight red_pigment_weight = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_pigment_is_40_percent_l196_19651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l196_19684

/-- Definition of the function f --/
noncomputable def f (k a x : ℝ) : ℝ := k * (a^x) - a^(-x)

/-- Definition of the function g --/
noncomputable def g (a m x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m * f 1 a x

theorem problem_solution (a k m : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x, f k a x = -f k a (-x)) →
  (f 1 a 1 > 0) →
  (f 1 a 1 = 3/2) →
  (∀ x ≥ 1, g a m x ≥ -2) →
  (∃ x ≥ 1, g a m x = -2) →
  (f k a 0 = 0 ∧ k = 1 ∧ m = 2) :=
by sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l196_19684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sum_of_distances_l196_19660

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  true -- Placeholder, can be expanded with actual triangle conditions

-- Define the concept of an altitude
def IsAltitude (D : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  true -- Placeholder, can be expanded with actual altitude conditions

-- Define a point on a line segment
def PointOnSegment (P B C : ℝ × ℝ) : Prop :=
  true -- Placeholder, can be expanded with actual conditions for P on BC

-- Define the distance between two points
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the angle measurement
noncomputable def Angle (A B C : ℝ × ℝ) : ℝ :=
  0 -- Placeholder, can be expanded with actual angle calculation

theorem minimum_sum_of_distances 
  (A B C D E : ℝ × ℝ) 
  (h₁ : Triangle A B C)
  (h₂ : IsAltitude D A B C)
  (h₃ : IsAltitude E A B C)
  (h₄ : Angle A B C = 60)
  (h₅ : Distance A E = 8)
  (h₆ : Distance B E = 4) :
  ∃ (P : ℝ × ℝ), PointOnSegment P B C ∧ 
    ∀ (Q : ℝ × ℝ), PointOnSegment Q B C → 
      Distance P D + Distance P E ≤ Distance Q D + Distance Q E ∧
      Distance P D + Distance P E = (20 * Real.sqrt 7) / 7 :=
by
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sum_of_distances_l196_19660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l196_19626

theorem triangle_angle_measure (a b c : ℝ) (h1 : a = 7) (h2 : b = 5) (h3 : c = 3) :
  Real.arccos ((c^2 + b^2 - a^2) / (2 * b * c)) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l196_19626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_year_growth_l196_19623

noncomputable def initial_value : ℝ := 70400

noncomputable def first_year_increase (x : ℝ) : ℝ := x * (1 + 1/8)

noncomputable def second_year_increase (x : ℝ) : ℝ := x * (1 + 1/6)

theorem two_year_growth :
  second_year_increase (first_year_increase initial_value) = 92400 := by
  -- Expand the definitions
  unfold second_year_increase first_year_increase initial_value
  -- Simplify the expression
  simp [mul_add, mul_one]
  -- Perform the numerical calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_year_growth_l196_19623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19612

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- State the theorem
theorem f_properties (a : ℝ) :
  -- Part 1: Monotonicity
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a → f a x₁ > f a x₂) ∧
  (∀ x₁ x₂ : ℝ, 0 < 1/a ∧ 1/a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
  -- Part 2: Upper bound on b
  (a > 0 → ∀ b : ℝ, (∀ x : ℝ, x > 0 → f a x ≥ a^2/2 + b) → b ≤ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l196_19612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_fraction_parts_l196_19600

def repeating_decimal : ℚ := 37 / 999

theorem product_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.Coprime n d ∧ 
  n * d = 27 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_fraction_parts_l196_19600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_dragon_heads_l196_19699

def DragonType := Nat → Option (Fin 3)
def HeadCount := Nat → Nat

structure DragonArrangement where
  headCount : HeadCount
  dragonType : DragonType

def isValidArrangement (arr : DragonArrangement) : Prop :=
  -- There are 15 dragons
  (∀ i, i < 15 → arr.headCount i > 0) ∧
  -- Neighboring dragons' heads differ by 1
  (∀ i, i < 14 → (arr.headCount (i + 1) = arr.headCount i + 1 ∨ 
                   arr.headCount (i + 1) = arr.headCount i - 1)) ∧
  -- Classification of dragons
  (∀ i, i < 15 → 
    match arr.dragonType i with
    | some 0 => i > 0 ∧ i < 14 ∧ 
                arr.headCount i > arr.headCount (i - 1) ∧ 
                arr.headCount i > arr.headCount (i + 1)  -- Cunning
    | some 1 => i > 0 ∧ i < 14 ∧ 
                arr.headCount i < arr.headCount (i - 1) ∧ 
                arr.headCount i < arr.headCount (i + 1)  -- Strong
    | some 2 => true  -- Ordinary
    | none => false
  ) ∧
  -- Exactly 4 cunning dragons with specified head counts
  (∃ a b c d, a < b ∧ b < c ∧ c < d ∧ d < 15 ∧
    arr.dragonType a = some 0 ∧ arr.headCount a = 4 ∧
    arr.dragonType b = some 0 ∧ arr.headCount b = 6 ∧
    arr.dragonType c = some 0 ∧ arr.headCount c = 7 ∧
    arr.dragonType d = some 0 ∧ arr.headCount d = 7 ∧
    (∀ i, i < 15 → arr.dragonType i = some 0 → i = a ∨ i = b ∨ i = c ∨ i = d)) ∧
  -- Exactly 3 strong dragons with specified head counts
  (∃ x y z, x < y ∧ y < z ∧ z < 15 ∧
    arr.dragonType x = some 1 ∧ arr.headCount x = 3 ∧
    arr.dragonType y = some 1 ∧ arr.headCount y = 3 ∧
    arr.dragonType z = some 1 ∧ arr.headCount z = 6 ∧
    (∀ i, i < 15 → arr.dragonType i = some 1 → i = x ∨ i = y ∨ i = z)) ∧
  -- First and last dragons have the same number of heads
  (arr.headCount 0 = arr.headCount 14)

theorem first_dragon_heads (arr : DragonArrangement) 
  (h : isValidArrangement arr) : arr.headCount 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_dragon_heads_l196_19699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l196_19652

/-- A pair of integers (x, y) that satisfies x^2 + y^2 = 65 -/
def LatticePoint : Type := { p : ℤ × ℤ // p.1^2 + p.2^2 = 65 }

/-- The set of all lattice points (x, y) satisfying x^2 + y^2 = 65 -/
def AllLatticePoints : Finset LatticePoint := sorry

/-- A pair of real numbers (a, b) that satisfies the given conditions -/
def ValidPair : Type := { p : ℝ × ℝ // ∃ (x y : ℤ), p.1 * x + p.2 * y = 1 ∧ x^2 + y^2 = 65 }

/-- The set of all valid pairs (a, b) -/
def AllValidPairs : Finset ValidPair := sorry

/-- Proof that AllValidPairs is finite -/
instance : Fintype ValidPair := sorry

/-- The main theorem stating that the number of valid pairs is 128 -/
theorem count_valid_pairs : Fintype.card ValidPair = 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l196_19652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seokjin_rank_theorem_l196_19657

def jimin_rank : ℕ := 4

def seokjin_rank (people_between : ℕ) : ℕ := jimin_rank + people_between + 1

theorem seokjin_rank_theorem (people_between : ℕ) 
  (h1 : people_between = 19) 
  (h2 : jimin_rank < seokjin_rank people_between) : 
  seokjin_rank people_between = jimin_rank + people_between + 1 :=
by
  rfl

#check seokjin_rank
#check seokjin_rank_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seokjin_rank_theorem_l196_19657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_empty_time_l196_19629

/-- Represents the time it takes for a cistern to empty due to a leak, given the normal fill time and the fill time with a leak. -/
noncomputable def leakEmptyTime (normalFillTime leakyFillTime : ℝ) : ℝ :=
  let normalRate := 1 / normalFillTime
  let leakyRate := 1 / leakyFillTime
  let leakRate := normalRate - leakyRate
  1 / leakRate

/-- Theorem stating that for a cistern that normally fills in 12 hours and takes 14 hours to fill with a leak, it will take 84 hours for the leak to empty the full cistern. -/
theorem cistern_leak_empty_time :
  leakEmptyTime 12 14 = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_leak_empty_time_l196_19629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l196_19683

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h : seq.a 10 = S seq 4) :
  S seq 8 / seq.a 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l196_19683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l196_19617

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (30 - x))

-- State the theorem
theorem g_max_value :
  ∃ (N : ℝ), N = 10 * Real.sqrt 21 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 30 → g x ≤ N ∧
  ∃ x₁ : ℝ, x₁ = 30 ∧ g x₁ = N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l196_19617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_one_over_two_to_fifteen_l196_19664

-- Define the fraction we're working with
def our_fraction : ℚ := 1 / (2^15)

-- Define a function to get the last digit of a rational number's decimal expansion
def last_digit_of_decimal_expansion (q : ℚ) : ℕ :=
  (((q * 10^15).floor : ℤ).toNat) % 10

-- State the theorem
theorem last_digit_of_one_over_two_to_fifteen :
  last_digit_of_decimal_expansion our_fraction = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_one_over_two_to_fifteen_l196_19664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_criterion_l196_19656

/-- Represents the digits of a natural number in base 10 -/
def digits (n : ℕ) : List ℕ := sorry

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := (digits n).sum

/-- Divisibility criterion for 3 and 9 -/
theorem divisibility_criterion (n : ℕ) :
  (∃ k : ℕ, n = 3 * k) = (∃ m : ℕ, digit_sum n = 3 * m) ∧
  (∃ k : ℕ, n = 9 * k) = (∃ m : ℕ, digit_sum n = 9 * m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_criterion_l196_19656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l196_19613

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -4 ∧ x ≤ -1 then -1 - x/2
  else if x > -1 ∧ x ≤ 3 then Real.sqrt (9 - (x - 3)^2) - 3
  else if x > 3 ∧ x ≤ 5 then 1.5 * (x - 3)
  else 0  -- This else case is added to make the function total

-- State the theorem about |g(x)|
theorem abs_g_piecewise (x : ℝ) :
  (x ≥ -4 ∧ x ≤ -1 → |g x| = 1 + x/2) ∧
  (x > -1 ∧ x ≤ 3 → |g x| = 3 - Real.sqrt (9 - (x - 3)^2)) ∧
  (x > 3 ∧ x ≤ 5 → |g x| = 1.5 * (x - 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l196_19613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_l196_19665

noncomputable def angle_of_inclination (slope : ℝ) : ℝ := Real.arctan slope

def line_equation (x y : ℝ) : Prop := y - 3 = Real.sqrt 3 * (x - 4)

theorem angle_of_line : 
  angle_of_inclination (Real.sqrt 3) = π / 3 := by
  sorry

#check angle_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_line_l196_19665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l196_19646

/-- Calculates the average speed of a car given two speed-time pairs -/
noncomputable def average_speed (speed1 speed2 time1 time2 : ℝ) : ℝ :=
  let distance1 := speed1 * (time1 / 60)
  let distance2 := speed2 * (time2 / 60)
  let total_distance := distance1 + distance2
  let total_time := (time1 + time2) / 60
  total_distance / total_time

/-- Theorem: The average speed of a car driving at 60 km/h for 20 minutes
    and then at 90 km/h for 40 minutes is 80 km/h -/
theorem car_average_speed :
  average_speed 60 90 20 40 = 80 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l196_19646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_real_implies_a_equals_two_l196_19667

theorem product_real_implies_a_equals_two (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := a - I
  (z₁ * z₂).im = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_real_implies_a_equals_two_l196_19667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l196_19662

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp 1

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l196_19662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l196_19618

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type representing y = m(x-x₀) + y₀ -/
structure Line where
  m : ℝ  -- slope
  x₀ : ℝ -- x-coordinate of a point on the line
  y₀ : ℝ -- y-coordinate of a point on the line

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def focus : Point := ⟨1, 0⟩

def parabola_C : Parabola → Prop :=
  fun p => p.y^2 = 4*p.x

noncomputable def line_l : Line :=
  ⟨Real.sqrt 3, 1, 0⟩

def intersects (l : Line) (p : Parabola) (A B : Point) : Prop :=
  parabola_C p ∧ A.y = l.m * (A.x - l.x₀) + l.y₀ ∧ B.y = l.m * (B.x - l.x₀) + l.y₀

noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

theorem length_of_AB (A B : Point) :
  parabola_C (Parabola.mk A.x A.y (by sorry)) →
  parabola_C (Parabola.mk B.x B.y (by sorry)) →
  intersects line_l (Parabola.mk A.x A.y (by sorry)) A B →
  distance A B = 16/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l196_19618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_function_condition_l196_19685

-- Define the storm function property
def is_storm_function (f : ℝ → ℝ) (U : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ U → x₂ ∈ U → |f x₁ - f x₂| < 1

-- Define our specific function
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - b*x + 1

-- Define the interval [-1, 1]
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem storm_function_condition (b : ℝ) :
  is_storm_function (f b) interval ↔ 1 - Real.sqrt 2 < b ∧ b < Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_function_condition_l196_19685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bees_after_two_days_l196_19628

/-- Calculates the number of bees in a hive after a given number of days, 
    given an initial count and daily increase rate. -/
def bees_after_days (initial_count : ℕ) (increase_rate : ℚ) (days : ℕ) : ℕ :=
  (((initial_count : ℚ) * (1 + increase_rate) ^ days).ceil).toNat

/-- Theorem stating that the total number of bees in two hives after two days
    is 456, given the initial counts and daily increase rate. -/
theorem total_bees_after_two_days 
  (hive_a_initial : ℕ) (hive_b_initial : ℕ) (increase_rate : ℚ)
  (h1 : hive_a_initial = 144)
  (h2 : hive_b_initial = 172)
  (h3 : increase_rate = 1/5) : 
  bees_after_days hive_a_initial increase_rate 2 + 
  bees_after_days hive_b_initial increase_rate 2 = 456 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bees_after_two_days_l196_19628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l196_19686

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 12 = 0
def circle2 (x y : ℝ) : Prop := (x - 7)^2 + (y - 1)^2 = 36

-- Define the center and radius of circle1
def center1 : ℝ × ℝ := (3, -2)
def radius1 : ℝ := 1

-- Define the center and radius of circle2
def center2 : ℝ × ℝ := (7, 1)
def radius2 : ℝ := 6

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem stating that the circles are internally tangent
theorem circles_internally_tangent : 
  distance_between_centers = radius2 - radius1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l196_19686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_angle_contradiction_l196_19637

-- Define Triangle structure
structure Triangle where
  obtuse_angles : Nat

theorem triangle_obtuse_angle_contradiction :
  (¬ (∀ t : Triangle, t.obtuse_angles ≤ 1)) ↔ (∃ t : Triangle, t.obtuse_angles ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_angle_contradiction_l196_19637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l196_19671

/-- The eccentricity of a hyperbola with specific geometric properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let C : ℝ → ℝ → Prop := λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1
  let F : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)
  let asymptote1 := λ x ↦ b / a * x
  let asymptote2 := λ x ↦ -b / a * x
  ∃ (M N : ℝ × ℝ),
    (M.2 = asymptote1 M.1) ∧ 
    (N.2 = asymptote2 N.1) ∧
    ((F.1 - M.1) * (asymptote1 1 - asymptote1 0) = -(F.2 - M.2)) ∧
    (3 * (F.1 - M.1), 3 * (F.2 - M.2)) = (N.1 - F.1, N.2 - F.2) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 6 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l196_19671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_length_l196_19614

/-- An octagon inscribed in a circle with alternating side lengths -/
structure InscribedOctagon where
  radius : ℝ
  side_length1 : ℝ
  side_length2 : ℝ
  side_length1_is_4 : side_length1 = 4
  side_length2_is_6 : side_length2 = 6

/-- The chord that divides the octagon into two quadrilaterals -/
noncomputable def dividing_chord (octagon : InscribedOctagon) : ℝ :=
  2 * octagon.radius * (Real.sqrt 6 + Real.sqrt 2) / 4

/-- Theorem stating the length of the dividing chord -/
theorem dividing_chord_length (octagon : InscribedOctagon) :
  dividing_chord octagon = 2 * octagon.radius * (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  rfl

#check dividing_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_chord_length_l196_19614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_periodic_sqrt_sequences_l196_19622

-- Define the sequences
noncomputable def α (n : ℕ) : ℕ := Int.toNat ⌊(Real.sqrt 10)^n⌋
noncomputable def β (n : ℕ) : ℕ := Int.toNat ⌊(Real.sqrt 2)^n⌋

-- Define what it means for a sequence to be eventually periodic
def eventually_periodic (f : ℕ → ℕ) : Prop :=
  ∃ (n₀ p : ℕ), p > 0 ∧ ∀ n ≥ n₀, f (n + p) = f n

-- Theorem statement
theorem not_periodic_sqrt_sequences :
  ¬(eventually_periodic α) ∧ ¬(eventually_periodic β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_periodic_sqrt_sequences_l196_19622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l196_19661

/-- Two circles in a 2D plane --/
structure TwoCircles where
  C₁ : Set (ℝ × ℝ) -- First circle
  C₂ : Set (ℝ × ℝ) -- Second circle

/-- Definition of the first circle C₁ --/
def circle_C₁ (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = 4

/-- Definition of the second circle C₂ --/
def circle_C₂ (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 - 4*p.1 + 4*p.2 - 12 = 0

/-- The two circles in our problem --/
def our_circles : TwoCircles where
  C₁ := {p : ℝ × ℝ | circle_C₁ p}
  C₂ := {p : ℝ × ℝ | circle_C₂ p}

/-- The common chord of two circles --/
def common_chord (tc : TwoCircles) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ tc.C₁ ∧ p ∈ tc.C₂}

/-- The theorem to be proved --/
theorem common_chord_properties (tc : TwoCircles) 
  (h : tc = our_circles) : 
  (∀ p, p ∈ common_chord tc → p.1 - p.2 + 2 = 0) ∧
  (∃ p q, p ∈ common_chord tc ∧ q ∈ common_chord tc ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_properties_l196_19661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sequence_a_l196_19625

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Adding the case for 0 to avoid the "missing cases" error
  | 1 => 3
  | (n + 2) => (2 * (n + 2) * sequence_a (n + 1) - n - 2) / (n + 1)

theorem divisibility_of_sequence_a (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ m : ℕ, m > 0 ∧ (∃ k : ℤ, sequence_a m = k * p) ∧ (∃ l : ℤ, sequence_a (m + 1) = l * p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_sequence_a_l196_19625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shirt_price_is_nine_l196_19624

/-- The greatest possible whole-dollar price of a shirt that allows Alec to buy 20 shirts
    given the specified conditions. -/
def max_shirt_price (total_budget : ℕ) (entrance_fee : ℕ) (num_shirts : ℕ) (tax_rate : ℚ) : ℕ :=
  let remaining_budget : ℕ := total_budget - entrance_fee
  let effective_budget : ℚ := (remaining_budget : ℚ) / (1 + tax_rate)
  (effective_budget / num_shirts).floor.toNat

/-- Theorem stating that the maximum shirt price is $9 given the problem conditions. -/
theorem max_shirt_price_is_nine :
  max_shirt_price 200 5 20 (6/100) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shirt_price_is_nine_l196_19624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equivalence_l196_19676

theorem remainder_equivalence (x : ℕ) (hx : x > 0) (h : 100 % x = 3) : 197 % x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_equivalence_l196_19676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l196_19694

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 8*y + 12 = 0

-- Define the line l1 on which the center lies
def center_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the line l2 that intercepts the chord
def chord_line (x y : ℝ) : Prop := 3*x + 4*y - 8 = 0

-- Theorem statement
theorem chord_length :
  (∀ x y, circle_eq x y → (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0)) →
  (∃ cx cy, center_line cx cy ∧ ∀ x y, circle_eq x y ↔ (x - cx)^2 + (y - cy)^2 = (cx - 0)^2 + (cy - 2)^2) →
  (∃ x1 y1 x2 y2, circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ chord_line x1 y1 ∧ chord_line x2 y2 ∧ 
    ((x2 - x1)^2 + (y2 - y1)^2)^(1/2 : ℝ) = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l196_19694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_item_value_calculation_l196_19615

/-- Calculates the total import tax and fees for an item --/
def total_tax_and_fees (value : ℝ) : ℝ :=
  let import_tax := 0.07 * (value - 1000)
  let weight_charge := 0.15 * (55 - 50)
  let origin_surcharge := 0.05 * value

/-- Theorem stating that the total value of the item is approximately $1524.58 --/
theorem item_value_calculation :
  ∃ (value : ℝ), total_tax_and_fees value = 112.70 ∧ abs (value - 1524.58) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_item_value_calculation_l196_19615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_bags_on_monday_l196_19647

/-- The number of bags Tiffany had on Monday -/
def bags_on_monday : ℕ := sorry

/-- The number of bags Tiffany found the next day -/
def bags_found_next_day : ℕ := 2

/-- The total number of bags Tiffany had after finding more -/
def total_bags : ℕ := 6

theorem tiffany_bags_on_monday : 
  bags_on_monday = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_bags_on_monday_l196_19647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_cost_theorem_l196_19621

/-- Represents the cost of fabric per yard -/
noncomputable def fabric_cost : ℚ := 2

/-- Represents the amount of fabric delivered on Monday -/
def monday_delivery : ℚ := 20

/-- Represents the amount of fabric delivered on Tuesday -/
def tuesday_delivery : ℚ := 2 * monday_delivery

/-- Represents the amount of fabric delivered on Wednesday -/
def wednesday_delivery : ℚ := (1 / 4) * tuesday_delivery

/-- Represents the total earnings from Monday to Wednesday -/
def total_earnings : ℚ := 140

theorem fabric_cost_theorem :
  fabric_cost * (monday_delivery + tuesday_delivery + wednesday_delivery) = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fabric_cost_theorem_l196_19621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l196_19632

/-- Calculates the total milk production for a given number of cows and days,
    considering an efficiency increase after a certain period. -/
noncomputable def total_milk_production (x y z w p q : ℝ) : ℝ :=
  (p * y * (1.5 * q - 0.5 * w)) / (x * z)

/-- Theorem stating that the total_milk_production function correctly calculates
    the milk production under the given conditions. -/
theorem milk_production_theorem
  (x y z w p q : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hw : w ≥ 0)
  (hp : p > 0)
  (hq : q ≥ 0) :
  total_milk_production x y z w p q =
    (p * y * (1.5 * q - 0.5 * w)) / (x * z) :=
by
  -- Unfold the definition of total_milk_production
  unfold total_milk_production
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_theorem_l196_19632
