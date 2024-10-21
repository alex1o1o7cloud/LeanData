import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_plan_comparison_l834_83493

/-- Calculates the annual cost difference between T-Mobile and M-Mobile plans for a given number of lines -/
def annual_cost_difference (num_lines : ℕ) : ℚ :=
  let t_mobile_base : ℚ := 50
  let t_mobile_additional : ℚ := 16
  let t_mobile_autopay_discount : ℚ := 1/10
  let t_mobile_unlimited_data : ℚ := 3
  let m_mobile_base : ℚ := 45
  let m_mobile_additional : ℚ := 14
  let m_mobile_activation : ℚ := 20

  let t_mobile_monthly := t_mobile_base + t_mobile_additional * (num_lines - 2 : ℚ) + t_mobile_unlimited_data * (num_lines : ℚ)
  let t_mobile_discounted := t_mobile_monthly * (1 - t_mobile_autopay_discount)
  let t_mobile_annual := t_mobile_discounted * 12

  let m_mobile_monthly := m_mobile_base + m_mobile_additional * (num_lines - 2 : ℚ)
  let m_mobile_annual := m_mobile_monthly * 12 + m_mobile_activation * (num_lines : ℚ)

  t_mobile_annual - m_mobile_annual

theorem mobile_plan_comparison :
  annual_cost_difference 5 = 764/10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mobile_plan_comparison_l834_83493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_symmetric_points_imply_k_value_l834_83409

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if two points are symmetric about y = x line -/
def symmetricAboutYEqualX (p q : Point) : Prop :=
  p.x = q.y ∧ p.y = q.x

/-- The exponential function -/
noncomputable def exp_func (x : ℝ) : ℝ := Real.exp x

/-- The linear function with slope k -/
def linear_func (k : ℝ) (x : ℝ) : ℝ := k * x

theorem unique_symmetric_points_imply_k_value (k : ℝ) :
  k > 0 →
  (∃! (p q : Point),
    p.y = exp_func p.x ∧
    q.y = linear_func k q.x ∧
    symmetricAboutYEqualX p q) →
  k = 1 / Real.exp 1 := by
  sorry

#check unique_symmetric_points_imply_k_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_symmetric_points_imply_k_value_l834_83409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l834_83401

/-- Given real numbers a, b, c, and points P, Q, R in ℝ³ such that
    the midpoint of QR is (a,0,0),
    the midpoint of PR is (0,b,0),
    and the midpoint of PQ is (0,0,c),
    prove that (PQ^2 + PR^2 + QR^2) / (a^2 + b^2 + c^2) = 8 -/
theorem midpoint_triangle_ratio (a b c : ℝ) (P Q R : Fin 3 → ℝ) 
    (h1 : ((Q 0 + R 0) / 2 = a) ∧ ((Q 1 + R 1) / 2 = 0) ∧ ((Q 2 + R 2) / 2 = 0))
    (h2 : ((P 0 + R 0) / 2 = 0) ∧ ((P 1 + R 1) / 2 = b) ∧ ((P 2 + R 2) / 2 = 0))
    (h3 : ((P 0 + Q 0) / 2 = 0) ∧ ((P 1 + Q 1) / 2 = 0) ∧ ((P 2 + Q 2) / 2 = c)) :
    (dist P Q)^2 + (dist P R)^2 + (dist Q R)^2 = 8 * (a^2 + b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l834_83401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_lucky_multiple_of_8_l834_83407

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def isLucky (n : ℕ) : Prop := n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 :
  ∀ k : ℕ, k > 0 ∧ k < 16 ∧ k % 8 = 0 → isLucky k ∧
  ¬isLucky 16 ∧ 16 % 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_lucky_multiple_of_8_l834_83407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_intersection_three_circles_l834_83488

/-- The area of intersection of three equal circles, where each circle passes through
    the centers of the other two. This function is assumed to exist and correctly
    calculate the area based on the given conditions. -/
noncomputable def area_of_intersection_three_circles (r : ℝ) : ℝ :=
  (1/2 : ℝ) * r^2 * (Real.pi - Real.sqrt 3)

/-- The area of intersection of three equal circles, where each circle passes through
    the centers of the other two. -/
theorem area_intersection_three_circles (r : ℝ) (hr : r > 0) :
  area_of_intersection_three_circles r = (1/2 : ℝ) * r^2 * (Real.pi - Real.sqrt 3) :=
by
  -- Unfold the definition of area_of_intersection_three_circles
  unfold area_of_intersection_three_circles
  -- The equality now holds by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_intersection_three_circles_l834_83488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_s_for_area_20_l834_83499

-- Define the triangle PQR
noncomputable def P : ℝ × ℝ := (0, 10)
noncomputable def Q : ℝ × ℝ := (3, 0)
noncomputable def R : ℝ × ℝ := (10, 0)

-- Define the function for the area of triangle PVW given s
noncomputable def area_PVW (s : ℝ) : ℝ :=
  (1 / 2) * (70 - 17 * s + (7 * s^2) / 10)

-- Theorem statement
theorem exists_unique_s_for_area_20 :
  ∃! s : ℝ, 0 < s ∧ s < 10 ∧ area_PVW s = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_s_for_area_20_l834_83499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_common_language_l834_83411

/-- Represents a mathematician -/
structure Mathematician where
  id : ℕ
  languages : Finset String

/-- The proposition that needs to be proved -/
theorem mathematicians_common_language 
  (mathematicians : Finset Mathematician)
  (h_count : mathematicians.card = 9)
  (h_max_languages : ∀ m ∈ mathematicians, m.languages.card ≤ 3)
  (h_common_language : ∀ m₁ m₂ m₃, m₁ ∈ mathematicians → m₂ ∈ mathematicians → m₃ ∈ mathematicians → 
    m₁ ≠ m₂ ∧ m₂ ≠ m₃ ∧ m₁ ≠ m₃ → 
    ∃ l, l ∈ m₁.languages ∩ m₂.languages ∪ m₂.languages ∩ m₃.languages ∪ m₁.languages ∩ m₃.languages) :
  ∃ l, (mathematicians.filter (λ m : Mathematician => l ∈ m.languages)).card ≥ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_common_language_l834_83411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_equality_l834_83451

-- Define the custom operation ⊙
noncomputable def odot (a b : ℝ) : ℝ := a^3 / b^2

-- State the theorem
theorem custom_operation_equality :
  (odot (odot 2 4) 6) - (odot 2 (odot 4 6)) = -2591/288 :=
by
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operation_equality_l834_83451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l834_83487

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 18*x + 8*y = -72

-- Define the area of the circle
noncomputable def circle_area : ℝ := 25 * Real.pi

-- Theorem statement
theorem circle_area_proof :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    circle_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_proof_l834_83487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l834_83439

noncomputable section

open Real

theorem triangle_ABC_properties (A B C : ℝ) (a b c : ℝ) (S : ℝ) 
  (h1 : Real.cos (2 * C) - 3 * Real.cos C = 1)
  (h2 : c = sqrt 7)
  (h3 : S = 3 * sqrt 3 / 2)
  (h4 : S = 1/2 * a * b * Real.sin C)
  (h5 : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (h6 : 0 < C ∧ C < π)
  (h7 : Real.sin A / a = Real.sin B / b)
  (h8 : Real.sin A / a = Real.sin C / c) :
  (C = 2 * π / 3) ∧
  (sqrt 7 * (Real.sin A + Real.sin B) = sqrt 39 / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l834_83439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l834_83454

/-- A type representing a kitten (point) on a plane -/
structure Kitten where
  x : ℝ
  y : ℝ

/-- A type representing an arrangement of kittens -/
structure Arrangement where
  kittens : Finset Kitten
  size : kittens.card = 6

/-- Predicate to check if three kittens are collinear -/
def collinear (a b c : Kitten) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)

/-- Predicate to check if an arrangement satisfies the row conditions -/
def satisfiesRowConditions (arr : Arrangement) : Prop :=
  (∃ r1 r2 r3 : Finset Kitten,
    r1.card = 3 ∧ r2.card = 3 ∧ r3.card = 3 ∧
    (∀ k1 k2 k3, k1 ∈ r1 → k2 ∈ r1 → k3 ∈ r1 → collinear k1 k2 k3) ∧
    (∀ k1 k2 k3, k1 ∈ r2 → k2 ∈ r2 → k3 ∈ r2 → collinear k1 k2 k3) ∧
    (∀ k1 k2 k3, k1 ∈ r3 → k2 ∈ r3 → k3 ∈ r3 → collinear k1 k2 k3)) ∧
  (∃ r1 r2 r3 r4 r5 r6 : Finset Kitten,
    r1.card = 2 ∧ r2.card = 2 ∧ r3.card = 2 ∧
    r4.card = 2 ∧ r5.card = 2 ∧ r6.card = 2 ∧
    (∀ k1 k2, k1 ∈ r1 → k2 ∈ r1 → collinear k1 k2 (Kitten.mk 0 0)) ∧
    (∀ k1 k2, k1 ∈ r2 → k2 ∈ r2 → collinear k1 k2 (Kitten.mk 0 0)) ∧
    (∀ k1 k2, k1 ∈ r3 → k2 ∈ r3 → collinear k1 k2 (Kitten.mk 0 0)) ∧
    (∀ k1 k2, k1 ∈ r4 → k2 ∈ r4 → collinear k1 k2 (Kitten.mk 0 0)) ∧
    (∀ k1 k2, k1 ∈ r5 → k2 ∈ r5 → collinear k1 k2 (Kitten.mk 0 0)) ∧
    (∀ k1 k2, k1 ∈ r6 → k2 ∈ r6 → collinear k1 k2 (Kitten.mk 0 0)))

/-- Theorem stating that there exists an arrangement satisfying the conditions -/
theorem exists_valid_arrangement :
  ∃ arr : Arrangement, satisfiesRowConditions arr := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l834_83454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l834_83462

/-- A rectangle type with basic properties -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ := length * width
  perimeter : ℝ := 2 * (length + width)
  isDividedIntoFourIdenticalSquares : Prop

/-- Given a rectangle ABCD divided into four identical squares with a perimeter of 160 cm,
    prove that its area is 1024 square centimeters. -/
theorem rectangle_area (ABCD : Rectangle) 
  (p : ABCD.perimeter = 160) 
  (h : ABCD.isDividedIntoFourIdenticalSquares) : 
  ABCD.area = 1024 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l834_83462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l834_83433

/-- Definition of the sequence a_n -/
def a : ℕ+ → ℝ := sorry

/-- Definition of S_n as the sum of the first n terms of a_n -/
def S : ℕ+ → ℝ := sorry

/-- Definition of b_n in terms of a_n -/
noncomputable def b (n : ℕ+) : ℝ := a n + Real.log (a n) / Real.log 3

/-- Definition of T_n as the sum of the first n terms of b_n -/
def T : ℕ+ → ℝ := sorry

/-- Main theorem statement -/
theorem sequence_proof (n : ℕ+) :
  a 1 = 3 ∧ 
  (∀ k : ℕ+, 2 * S k = 3 * a k - 3) →
  (a n = 3^(n : ℝ)) ∧ 
  (T n = (3^((n : ℝ) + 1) + (n : ℝ)^2 + (n : ℝ) - 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l834_83433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_per_square_foot_l834_83436

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular tank -/
noncomputable def surfaceArea (d : TankDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the cost per square foot of insulation -/
noncomputable def costPerSquareFoot (dimensions : TankDimensions) (totalCost : ℝ) : ℝ :=
  totalCost / surfaceArea dimensions

/-- Theorem: The cost per square foot of insulation for the given tank is $20 -/
theorem insulation_cost_per_square_foot :
  let tank_dimensions : TankDimensions := { length := 7, width := 3, height := 2 }
  let total_cost : ℝ := 1640
  costPerSquareFoot tank_dimensions total_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_per_square_foot_l834_83436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_l834_83431

/-- 
Given a right prism with an isosceles triangular base:
- a: base length of the triangle
- α: base angle of the triangle
- The lateral surface area is equal to the sum of the areas of the bases
-/
theorem prism_volume (a : ℝ) (α : ℝ) (h_a_pos : 0 < a) (h_α_pos : 0 < α) (h_α_lt_pi_half : α < π / 2) :
  let lateral_area := 2 * a * (a / (2 * Real.cos α)) * Real.sin α
  let base_area := (1 / 2) * a * (a / 2 * Real.tan α)
  let volume := base_area * (a / 2 * Real.tan (α / 2))
  (lateral_area = 2 * base_area) →
  volume = (1 / 8) * a^3 * Real.tan α * Real.tan (α / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_l834_83431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l834_83477

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1 + 2^x + 4^x * a) / 3)

/-- The theorem stating the conditions for f(x) to be defined -/
theorem f_defined_iff (a : ℝ) : 
  (∀ x < 1, ∃ y, f a x = y) ↔ a ≥ -3/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l834_83477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_are_close_l834_83404

-- Define the functions f₁ and f₂
noncomputable def f₁ (a : ℝ) (x : ℝ) : ℝ := Real.log (x - 3 * a) / Real.log a
noncomputable def f₂ (a : ℝ) (x : ℝ) : ℝ := Real.log (1 / (x - a)) / Real.log a

-- Define the property of being "close" on an interval
def areClose (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem functions_are_close (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  areClose (f₁ a) (f₂ a) (a + 2) (a + 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_are_close_l834_83404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_ratio_l834_83450

/-- Represents the hours charged by each person --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Defines the conditions of the problem --/
def satisfiesConditions (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.mark = h.kate + 75 ∧
  h.kate + h.pat + h.mark = 135

/-- Theorem statement --/
theorem project_hours_ratio (h : ProjectHours) 
  (hc : satisfiesConditions h) : 
  h.pat * 3 = h.mark := by
  -- Proof steps would go here
  sorry

#check project_hours_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_ratio_l834_83450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_combined_l834_83444

/-- The average age of a group of people given their total age and count -/
def average_age (total_age : ℕ) (count : ℕ) : ℚ :=
  (total_age : ℚ) / (count : ℚ)

theorem average_age_combined : 
  let fifth_graders_count : ℕ := 40
  let fifth_graders_avg_age : ℕ := 10
  let parents_count : ℕ := 60
  let parents_avg_age : ℕ := 35
  let teachers_count : ℕ := 10
  let teachers_avg_age : ℕ := 45
  let total_count : ℕ := fifth_graders_count + parents_count + teachers_count
  let total_age : ℕ := 
    fifth_graders_count * fifth_graders_avg_age +
    parents_count * parents_avg_age +
    teachers_count * teachers_avg_age
  average_age total_age total_count = 2950 / 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_combined_l834_83444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_at_x_6_l834_83479

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Given a Line3D, compute the y-coordinate for a given x-coordinate -/
noncomputable def yCoordinate (l : Line3D) (x : ℝ) : ℝ :=
  let (x1, y1, _) := l.point1
  let (x2, y2, _) := l.point2
  let slope := (y2 - y1) / (x2 - x1)
  y1 + slope * (x - x1)

/-- The main theorem to prove -/
theorem y_coordinate_at_x_6 (l : Line3D) :
  l.point1 = (3, 3, 2) → l.point2 = (7, 0, -4) → yCoordinate l 6 = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_at_x_6_l834_83479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_f_f_inequality_solution_set_l834_83468

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x - 1 else 1 - 2^x

-- State the properties of f
axiom f_odd : ∀ x, f (-x) = -f x

-- Theorem 1: f(1) = -1
theorem f_at_one : f 1 = -1 := by
  sorry

-- Theorem 2: The solution set for f(f(x)) ≤ 7 is (-∞, 2]
theorem f_f_inequality_solution_set : 
  ∀ x, f (f x) ≤ 7 ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_one_f_f_inequality_solution_set_l834_83468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l834_83482

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A →
  b + c = Real.sqrt 10 →
  a = 2 →
  A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l834_83482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_existence_l834_83475

-- Define the parabola E
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the circle F
structure Circle where
  equation : ℝ → ℝ → Prop

-- Define a line
structure Line where
  equation : ℝ → ℝ → Prop

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

def origin : Point := ⟨0, 0⟩

-- Helper function to calculate distance between two points
noncomputable def dist (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the theorem
theorem parabola_and_line_existence 
  (E : Parabola)
  (F : Circle)
  (vertex_at_origin : E.equation origin.x origin.y)
  (circle_equation : ∀ x y, F.equation x y ↔ x^2 + y^2 - 4*x + 3 = 0)
  (focus_at_circle_center : ∃ f : Point, F.equation f.x f.y ∧ f.x = 2 ∧ f.y = 0) :
  (∃ p : ℝ, p > 0 ∧ ∀ x y, E.equation x y ↔ y^2 = 8*x) ∧
  (∃ l : Line, (l.equation 2 2 ∨ l.equation 2 (-2)) ∧
    ∃ A B C D : Point,
      E.equation A.x A.y ∧ E.equation D.x D.y ∧
      F.equation B.x B.y ∧ F.equation C.x C.y ∧
      l.equation A.x A.y ∧ l.equation B.x B.y ∧ l.equation C.x C.y ∧ l.equation D.x D.y ∧
      A.x ≥ 0 ∧ A.y ≥ 0 ∧ B.x ≥ 0 ∧ B.y ≥ 0 ∧ C.x ≥ 0 ∧ C.y ≤ 0 ∧ D.x ≥ 0 ∧ D.y ≤ 0 ∧
      2 * dist B C = (dist A B + dist C D) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_existence_l834_83475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_midpoint_to_line_l834_83473

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 4

-- Define the point A
def point_A : ℝ × ℝ := (2, 0)

-- Define a point on the circle
noncomputable def point_on_circle (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)

-- Define the midpoint of a chord
noncomputable def midpoint_chord (α : ℝ) : ℝ × ℝ := (Real.cos α + 1, Real.sin α)

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := abs (x + y - 4) / Real.sqrt 2

-- State the theorem
theorem min_distance_midpoint_to_line :
  ∃ (min_dist : ℝ), min_dist = (3 * Real.sqrt 2 - 2) / 2 ∧
  ∀ (α : ℝ), distance_to_line (midpoint_chord α).1 (midpoint_chord α).2 ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_midpoint_to_line_l834_83473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonals_count_valid_x_l834_83428

theorem rectangular_prism_diagonals (x : ℕ) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 = 1867^2 ∧ 
    a^2 + c^2 = 2019^2 ∧ 
    b^2 + c^2 = x^2) ↔ 
  (769 ≤ x ∧ x ≤ 2749) :=
sorry

theorem count_valid_x : 
  Finset.card (Finset.range 2750 \ Finset.range 769) = 1981 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonals_count_valid_x_l834_83428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_circle_radius_proof_l834_83420

/-- The radius of a smaller circle in an arrangement where six congruent circles
    are placed inside a larger circle of radius 10, with three circles lined up
    east-west and three north-south, touching each other and the larger circle's boundary. -/
noncomputable def small_circle_radius : ℚ := 10 / 3

/-- Theorem stating that the radius of the smaller circles is 10/3 meters. -/
theorem small_circle_radius_proof (large_radius : ℚ) 
  (h1 : large_radius = 10)
  (h2 : ∃ (small_radius : ℚ), 6 * small_radius = 2 * large_radius) :
  small_circle_radius = 10 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_circle_radius_proof_l834_83420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_birthday_was_thursday_l834_83426

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def two_days_after (d : DayOfWeek) : DayOfWeek :=
  next_day (next_day d)

def prev_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

theorem vasyas_birthday_was_thursday :
  ∀ (statement_day : DayOfWeek),
    (two_days_after statement_day = DayOfWeek.Sunday) →
    (next_day statement_day ≠ DayOfWeek.Sunday) →
    (next_day (next_day statement_day) = DayOfWeek.Sunday) →
    (statement_day = DayOfWeek.Friday) →
    (prev_day statement_day = DayOfWeek.Thursday) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_birthday_was_thursday_l834_83426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_2_monotonicity_non_positive_a_monotonicity_positive_a_l834_83417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x - a / x

theorem extreme_value_at_2 (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ∈ Set.Ioo (2 - ε) (2 + ε) → f a x ≥ f a 2) →
  a = 4 := by sorry

theorem monotonicity_non_positive_a (a : ℝ) (h : a ≤ 0) :
  StrictMono (f a) := by sorry

theorem monotonicity_positive_a (a : ℝ) (h : a > 0) :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.sqrt a → f a x > f a y) ∧
  (∀ x y, Real.sqrt a < x ∧ x < y → f a x < f a y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_2_monotonicity_non_positive_a_monotonicity_positive_a_l834_83417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l834_83419

noncomputable def my_sequence (n : ℝ) : ℝ :=
  ((2 * n - 3)^3 - (n + 5)^3) / ((3 * n - 1)^3 + (2 * n + 3)^3)

theorem my_sequence_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |my_sequence n - 1/5| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l834_83419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_mean_weight_l834_83429

noncomputable def player_weights : List ℝ := [64, 68, 71, 73, 76, 76, 77, 78, 80, 82, 85, 87, 89, 89]

noncomputable def mean_weight (weights : List ℝ) : ℝ :=
  (weights.sum) / (weights.length : ℝ)

theorem soccer_team_mean_weight :
  ∃ ε > 0, |mean_weight player_weights - 75.357| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_mean_weight_l834_83429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_achieve_ratio_l834_83402

/-- The daily progress rate as a decimal -/
def progress_rate : ℝ := 0.03

/-- The daily regress rate as a decimal -/
def regress_rate : ℝ := 0.03

/-- The ratio we want to achieve -/
def target_ratio : ℝ := 1000

/-- The number of days needed to achieve the target ratio -/
def days_needed : ℝ := 115

/-- Theorem stating that the number of days needed to achieve the target ratio
    is approximately 115 days -/
theorem days_to_achieve_ratio :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  ((1 + progress_rate) / (1 - regress_rate))^days_needed ∈ 
    Set.Icc (target_ratio - ε) (target_ratio + ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_achieve_ratio_l834_83402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_is_sine_l834_83497

-- Define the angular frequency ω
noncomputable def ω : ℝ := 2

-- Define the original cosine function
noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (ω * x - Real.pi / 2)

-- Define the shifted function f
noncomputable def f (x : ℝ) : ℝ := g (x - Real.pi / 8)

-- State the theorem
theorem shifted_function_is_sine : 
  ∀ x : ℝ, f x = 3 * Real.sin (2 * x - Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_is_sine_l834_83497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l834_83495

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 * Real.tan x + Real.cos x ^ 4 * (1 / Real.tan x)

theorem f_range : 
  Set.range f = {y : ℝ | y ≤ -1/2 ∨ y ≥ 1/2} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l834_83495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l834_83432

/-- An isosceles triangle with a given altitude --/
structure IsoscelesTriangle where
  -- The length of the equal sides
  side : ℝ
  -- The length of the altitude
  altitude : ℝ
  -- Assumption that side and altitude are positive
  side_pos : side > 0
  altitude_pos : altitude > 0

/-- The area of an isosceles triangle --/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  t.altitude ^ 2 / Real.sqrt 3

/-- Theorem: The area of an isosceles triangle with given side and altitude --/
theorem isosceles_triangle_area (t : IsoscelesTriangle) :
  area t = t.altitude ^ 2 / Real.sqrt 3 := by
  -- Unfold the definition of area
  unfold area
  -- The equality is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l834_83432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_abc_l834_83416

open BigOperators

def sum_series (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, ((-1:ℤ)^k : ℚ) * (k^3 + k^2 + k + 1 : ℚ) / k.factorial

theorem smallest_sum_abc : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    sum_series 50 = (a : ℚ) / b.factorial - c ∧
    (∀ (a' b' c' : ℕ), a' > 0 → b' > 0 → c' > 0 → 
      sum_series 50 = (a' : ℚ) / b'.factorial - c' →
      a + b + c ≤ a' + b' + c') ∧
    a + b + c = 2653 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_abc_l834_83416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l834_83466

theorem triangle_inequality (h l R r : ℝ) : 
  h > 0 → l > 0 → R > 0 → r > 0 → h / l ≥ Real.sqrt (2 * r / R) := by
  intros h_pos l_pos R_pos r_pos
  -- The proof steps would go here
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l834_83466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_to_center_ratio_l834_83415

/-- A square with a point inside dividing it into four triangles -/
structure SquareWithPoint where
  s : ℝ  -- Side length of the square
  p : ℝ × ℝ  -- Coordinates of point P
  h_inside : p.1 ≥ 0 ∧ p.1 ≤ s ∧ p.2 ≥ 0 ∧ p.2 ≤ s  -- P is inside the square
  h_areas : ∃ (a b c d : ℝ), 
    a + b + c + d = s^2 ∧  -- Total area is s^2
    (a : ℝ) / 1 = (b : ℝ) / 2 ∧ (b : ℝ) / 2 = (c : ℝ) / 3 ∧ (c : ℝ) / 3 = (d : ℝ) / 4  -- Area ratios

/-- The center of the square -/
noncomputable def center (sq : SquareWithPoint) : ℝ × ℝ := (sq.s / 2, sq.s / 2)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The ratio of side length to distance from center to P is 2 -/
theorem side_to_center_ratio (sq : SquareWithPoint) :
  sq.s / distance (center sq) sq.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_to_center_ratio_l834_83415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_cosine_sine_equation_proof_l834_83490

noncomputable def count_solutions_cosine_sine_equation : ℕ :=
  let f : ℝ → ℝ := λ x => Real.cos x ^ 2 + 3 * Real.sin x ^ 2
  let S := { x : ℝ | -20 < x ∧ x < 100 ∧ f x = 1 }
  38

theorem count_solutions_cosine_sine_equation_proof :
  count_solutions_cosine_sine_equation = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_cosine_sine_equation_proof_l834_83490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_2010_mod_1000_l834_83459

-- Define the polynomial q(x)
def q (x : ℤ) : ℤ := (Finset.range 2011).sum (λ i ↦ x^i)

-- Define the divisor polynomial
def divisor (x : ℤ) : ℤ := x^4 - x^3 + 2*x^2 - x + 1

-- Define s(x) as the remainder when q(x) is divided by the divisor
def s (x : ℤ) : ℤ := q x % divisor x

-- Theorem statement
theorem remainder_of_s_2010_mod_1000 : |s 2010| % 1000 = 111 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_s_2010_mod_1000_l834_83459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_C_distance_when_B_finishes_l834_83491

-- Define the track length and race distance
noncomputable def track_length : ℝ := 400
noncomputable def race_distance : ℝ := 800

-- Define the speeds of runners relative to A's speed
noncomputable def speed_A : ℝ := 1
noncomputable def speed_B : ℝ := 8/7
noncomputable def speed_C : ℝ := 6/7

-- Theorem statement
theorem runner_C_distance_when_B_finishes :
  let time_B_finishes := race_distance / (speed_B * track_length)
  let distance_C_covers := speed_C * track_length * time_B_finishes
  race_distance - distance_C_covers = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_C_distance_when_B_finishes_l834_83491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_l834_83423

/-- A circle passing through point (3,2), with its center on the line y=2x, and tangent to the line y=2x+5 -/
def special_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  let (a, b) := center
  b = 2 * a ∧  -- center on y=2x
  (3 - a)^2 + (2 - b)^2 = radius^2 ∧  -- passes through (3,2)
  |2 * a - 2 * a + 5| / Real.sqrt 5 = radius  -- tangent to y=2x+5

theorem special_circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    special_circle center radius ∧
    ((center = (2, 4) ∧ radius^2 = 5) ∨
     (center = (4/5, 8/5) ∧ radius^2 = 5)) := by
  sorry

#check special_circle_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_l834_83423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l834_83469

/-- Line with equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Returns true if the point is in the first quadrant -/
def in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Returns true if the point is below or on the line -/
def below_or_on_line (p : Point) (l : Line) : Prop :=
  p.y ≤ l.m * p.x + l.b

/-- Returns true if the point is between two lines -/
def between_lines (p : Point) (l1 l2 : Line) : Prop :=
  (l1.m * p.x + l1.b ≥ p.y) ∧ (p.y > l2.m * p.x + l2.b)

/-- The area of the triangle formed by a line in the first quadrant -/
noncomputable def triangle_area (l : Line) : ℝ :=
  let x_intercept := l.b / (-l.m)
  (1/2) * x_intercept * l.b

theorem probability_between_lines (l m : Line) 
  (hl : l.m = -2 ∧ l.b = 8) 
  (hm : m.m = -3 ∧ m.b = 8) : 
  (triangle_area l - triangle_area m) / triangle_area l = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l834_83469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_four_thirds_l834_83470

-- Define the function f
noncomputable def f (y : ℝ) : ℝ := 
  let x := (y - 1)^2
  (x + 3) / (x - 1)

-- State the theorem
theorem f_value_at_four_thirds :
  f (4/3) = -7/2 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_four_thirds_l834_83470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l834_83492

-- Define the constants
noncomputable def a : ℝ := Real.rpow 0.2 0.5
noncomputable def b : ℝ := -Real.log 10 / Real.log 0.2
noncomputable def c : ℝ := Real.rpow 0.2 0.2

-- State the theorem
theorem relationship_abc : a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l834_83492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_weight_proof_l834_83427

/-- The original weight of water in grams -/
noncomputable def m : ℝ := 1500 / 19

/-- The amount of water remaining after the first day -/
noncomputable def first_day_remaining : ℝ := 0.9 * m

/-- The amount of water remaining after the second day -/
noncomputable def second_day_remaining : ℝ := 0.9 * first_day_remaining

/-- The total amount of water evaporated -/
noncomputable def total_evaporated : ℝ := m - second_day_remaining

theorem water_weight_proof : m = 1500 / 19 :=
  by
  -- Define m explicitly
  have h_m : m = 1500 / 19 := rfl
  
  -- Calculate the total evaporated water
  have h_total_evaporated : total_evaporated = 0.1 * m + 0.1 * (0.9 * m) := by
    unfold total_evaporated
    unfold second_day_remaining
    unfold first_day_remaining
    ring
  
  -- Show that the total evaporated water is 15 grams
  have h_evaporated_15 : total_evaporated = 15 := by
    rw [h_total_evaporated, h_m]
    norm_num
  
  -- The proof is complete since we defined m as 1500 / 19
  exact h_m


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_weight_proof_l834_83427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l834_83447

noncomputable def original_fraction : ℝ := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13)

noncomputable def rationalized_fraction : ℝ := (-12 * Real.sqrt 7 + 9 * Real.sqrt 13) / 5

def A : ℤ := -12
def B : ℕ := 7
def C : ℕ := 9
def D : ℕ := 13
def E : ℕ := 5

theorem rationalize_and_sum :
  (original_fraction = rationalized_fraction) ∧
  (A + B + C + D + E = 22) ∧
  (B < D) ∧
  (Int.gcd A E = 1) ∧ (Int.gcd C E = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l834_83447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l834_83430

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x a : ℝ) : ℝ := 2^x + a

-- Define the intervals
def I₁ : Set ℝ := Set.Icc (1/2) 1
def I₂ : Set ℝ := Set.Icc 2 3

-- State the theorem
theorem range_of_a (h : ∀ x₁ ∈ I₁, ∃ x₂ ∈ I₂, f x₁ ≥ g x₂ a) : 
  Set.Iic 1 = {a | ∀ x₁ ∈ I₁, ∃ x₂ ∈ I₂, f x₁ ≥ g x₂ a} := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l834_83430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l834_83464

/-- A trapezoid with specific properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  height : ℝ
  isIsosceles : EF = GH

/-- The perimeter of a trapezoid with given properties -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  2 * t.EF + 2 * Real.sqrt (t.height^2 + ((t.GH - t.EF) / 2)^2)

/-- Theorem stating the perimeter of a specific trapezoid -/
theorem trapezoid_perimeter :
  ∀ t : Trapezoid,
    t.EF = 10 ∧ t.GH = 12 ∧ t.height = 5 →
    perimeter t = 22 + 2 * Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l834_83464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_stub_ratio_l834_83496

/-- Represents the length of a candle stub as a function of time -/
noncomputable def candleStubLength (initialLength : ℝ) (burnTime : ℝ) (elapsedTime : ℝ) : ℝ :=
  initialLength * (burnTime - elapsedTime) / burnTime

theorem candle_stub_ratio (initialLength : ℝ) (burnTime1 burnTime2 elapsedTime : ℝ) 
  (h1 : burnTime1 = 5)
  (h2 : burnTime2 = 3)
  (h3 : elapsedTime = 2.5)
  (h4 : initialLength > 0) :
  candleStubLength initialLength burnTime1 elapsedTime = 
  3 * candleStubLength initialLength burnTime2 elapsedTime := by
  sorry

#check candle_stub_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_stub_ratio_l834_83496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_chips_l834_83463

/-- The number of chips on one side of the equilateral triangle -/
def triangle_side : ℕ := sorry

/-- The number of chips on one side of the square -/
def square_side : ℕ := sorry

/-- The total number of chips -/
def total_chips : ℕ := sorry

/-- The relationship between the side lengths of the triangle and square -/
axiom side_relationship : square_side = triangle_side - 2

/-- The formula for calculating the total number of chips in the triangle -/
axiom triangle_chips : total_chips = 3 * triangle_side - 3

/-- The formula for calculating the total number of chips in the square -/
axiom square_chips : total_chips = 4 * square_side - 4

/-- The theorem stating that Petya has 24 chips -/
theorem petya_chips : total_chips = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_chips_l834_83463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l834_83471

noncomputable section

/-- The cost price of an article given its selling price and profit margin -/
def cost_price (selling_price : ℝ) (profit_margin : ℝ) : ℝ :=
  selling_price / (1 + profit_margin)

/-- The marked price of an article given its selling price and discount rate -/
def marked_price (selling_price : ℝ) (discount_rate : ℝ) : ℝ :=
  selling_price / (1 - discount_rate)

theorem article_cost_price :
  let selling_price : ℝ := 65.97
  let profit_margin : ℝ := 0.25
  let discount_rate : ℝ := 0.10
  let calculated_cost_price := cost_price selling_price profit_margin
  ∃ ε > 0, |calculated_cost_price - 52.776| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l834_83471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l834_83437

-- Define the curves C1 and C2
noncomputable def C1 (t : ℝ) : ℝ × ℝ := (6 + (Real.sqrt 3 / 2) * t, t / 2)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (10 * Real.cos θ * Real.cos θ, 10 * Real.cos θ * Real.sin θ)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, C1 t = p ∧ C2 θ = p}

-- State the theorem
theorem distance_between_intersection_points :
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧ ‖A - B‖ = 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l834_83437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l834_83481

/-- The cost price of a book, given the profit/loss condition -/
noncomputable def cost_price (price : ℝ) : ℝ :=
  let profit_price := price * 1.09
  let loss_price := price * 0.91
  if profit_price - loss_price = 9 then price else 0

/-- Theorem stating that the cost price of the book is 50 -/
theorem book_cost_price : cost_price 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l834_83481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportional_k_l834_83498

noncomputable def inverse_proportional (k : ℝ) (h : k ≠ 0) : ℝ → ℝ := fun x ↦ k / x

theorem inverse_proportional_k (k : ℝ) (h : k ≠ 0) :
  inverse_proportional k h 1 = 3 → k = 3 := by
  intro h1
  -- Proof steps would go here
  sorry

#check inverse_proportional
#check inverse_proportional_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportional_k_l834_83498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_prediction_l834_83489

/-- Data point representing advertising cost and sales -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

def data : List DataPoint := [
  ⟨2, 29⟩, ⟨3, 41⟩, ⟨4, 50⟩, ⟨5, 59⟩, ⟨6, 71⟩
]

def model_slope : ℝ := 10.2

theorem sales_prediction (model : LinearRegression) 
  (h1 : model.slope = model_slope)
  (h2 : ∀ p ∈ data, p.y = model.slope * p.x + model.intercept) :
  model.slope * 8 + model.intercept = 90.8 := by
  sorry

#check sales_prediction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_prediction_l834_83489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_minus_e_l834_83494

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the symmetry condition
axiom symmetry : ∀ x, f (x - 1) = f (1 - x)

-- Define the periodicity condition for x ≥ 0
axiom periodicity : ∀ x, x ≥ 0 → f (x + 2) = f x

-- Define the function behavior on [0,1]
axiom on_unit_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = Real.exp x - 1

-- State the theorem to be proved
theorem f_sum_equals_one_minus_e : f (-2017) + f 2018 = 1 - Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_minus_e_l834_83494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l834_83438

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + a*x + b

-- Define the function g
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a b x - 2*Real.sqrt 3

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := -x^2 + a

-- State the theorem
theorem cubic_function_properties :
  ∀ (a b : ℝ),
    (∀ x, f a b x ≥ f a b (-Real.sqrt 3)) →
    (f' a (-Real.sqrt 3) = 0) →
    (f a b (2*Real.sqrt 3) = 0) →
    (a = 3 ∧ b = 2*Real.sqrt 3) ∧
    (∀ m : ℝ,
      ((-Real.sqrt 3 < m ∧ m ≤ 0) → (∀ x, -Real.sqrt 3 ≤ x ∧ x ≤ m → g 3 (2*Real.sqrt 3) x ≠ 0)) ∧
      ((0 < m ∧ m ≤ 3) → (∃! x, -Real.sqrt 3 ≤ x ∧ x ≤ m ∧ g 3 (2*Real.sqrt 3) x = 0)) ∧
      ((3 < m) → (∃ x y, -Real.sqrt 3 ≤ x ∧ x ≤ m ∧ -Real.sqrt 3 ≤ y ∧ y ≤ m ∧ x ≠ y ∧ g 3 (2*Real.sqrt 3) x = 0 ∧ g 3 (2*Real.sqrt 3) y = 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l834_83438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l834_83443

/-- Definition of the sum of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Theorem: If {aₙ} is a geometric sequence with a₁ = 4 and common ratio q,
    and if {Sₙ + 2} is also a geometric sequence where Sₙ is the sum of the first n terms of {aₙ},
    then q = 3 -/
theorem geometric_sequence_property (q : ℝ) : 
  (∀ n : ℕ, geometric_sum 4 q n + 2 = (geometric_sum 4 q 1 + 2) * q^(n-1)) → 
  q = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l834_83443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l834_83412

-- Problem 1
theorem problem_1 : (-1)^2 + (Real.pi - 3.14)^0 + 2 * Real.sin (60 * π / 180) + |1 - Real.sqrt 3| - Real.sqrt 12 = 1 := by
  sorry

-- Problem 2
theorem problem_2 : {x : ℝ | x * (x - 2) = x - 2} = {1, 2} := by
  sorry

-- Problem 3
theorem problem_3 (x : ℝ) (h1 : x > 0) (h2 : x ≤ 2) (h3 : x ≠ 2) :
  (x + 2 + 4 / (x - 2)) / (x^3 / (x^2 - 4*x + 4)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l834_83412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_increasing_f_is_odd_and_increasing_l834_83476

-- Define the function f(x) = lg((1+x)/(1-x))
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | x < 1 ∧ x > -1}

-- Theorem stating that f is an odd function
theorem f_is_odd : ∀ x, x ∈ dom_f → f (-x) = -f x := by sorry

-- Theorem stating that f is an increasing function
theorem f_is_increasing : ∀ x y, x ∈ dom_f → y ∈ dom_f → x < y → f x < f y := by sorry

-- Main theorem combining both properties
theorem f_is_odd_and_increasing :
  (∀ x, x ∈ dom_f → f (-x) = -f x) ∧
  (∀ x y, x ∈ dom_f → y ∈ dom_f → x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_increasing_f_is_odd_and_increasing_l834_83476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_numbers_l834_83445

theorem four_digit_divisible_numbers : 
  ∃! n : ℕ, n = (Finset.filter (λ x : ℕ ↦ 
    1000 ≤ x ∧ x ≤ 9999 ∧ 
    x % 2 = 0 ∧ x % 3 = 0 ∧ x % 5 = 0 ∧ x % 7 = 0 ∧ x % 11 = 0
  ) (Finset.range 10000)).card ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_divisible_numbers_l834_83445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_l834_83480

/-- A line passing through (1,2) with vertical intercept twice the horizontal intercept -/
structure SpecialLine where
  line_equation : ℝ → ℝ
  horizontal_intercept : ℝ
  vertical_intercept : ℝ
  -- The line passes through (1,2)
  passes_through : line_equation 1 = 2
  -- The vertical intercept is twice the horizontal intercept
  intercept_relation : vertical_intercept = 2 * horizontal_intercept

/-- The equation of a SpecialLine is either 2x - y = 0 or 2x + y - 4 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, l.line_equation x = y ↔ (2 * x - y = 0 ∨ 2 * x + y - 4 = 0)) :=
by
  sorry

#check special_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_line_equation_l834_83480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l834_83455

noncomputable def curve (x y : ℝ) : Prop := x * y = 2

noncomputable def point1 : ℝ × ℝ := (3, 2/3)
noncomputable def point2 : ℝ × ℝ := (-4, -1/2)
noncomputable def point3 : ℝ × ℝ := (1/2, 4)

noncomputable def point4 : ℝ × ℝ := (2/3, 3)

theorem fourth_intersection_point :
  curve point4.1 point4.2 ∧
  curve point1.1 point1.2 ∧
  curve point2.1 point2.2 ∧
  curve point3.1 point3.2 ∧
  ∃ (h k R : ℝ),
    (point1.1 - h)^2 + (point1.2 - k)^2 = R^2 ∧
    (point2.1 - h)^2 + (point2.2 - k)^2 = R^2 ∧
    (point3.1 - h)^2 + (point3.2 - k)^2 = R^2 ∧
    (point4.1 - h)^2 + (point4.2 - k)^2 = R^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_intersection_point_l834_83455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_perimeter_l834_83449

/-- A triangle with side lengths in arithmetic sequence and largest angle sine of √3/2 has perimeter 15 -/
theorem triangle_special_perimeter :
  ∀ (a b c : ℝ) (A B C : ℝ),
  a > b ∧ b > c ∧ c > 0 →  -- Side lengths in descending order
  a - b = 2 ∧ b - c = 2 →  -- Arithmetic sequence with common difference 2
  A > B ∧ A > C →          -- A is the largest angle
  Real.sin A = Real.sqrt 3 / 2 →         -- Sine of largest angle is √3/2
  a + b + c = 15 :=        -- Perimeter is 15
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_perimeter_l834_83449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_festival_special_savings_l834_83422

-- Define the regular price of a hat
noncomputable def regular_price : ℝ := 40

-- Define the discount percentages
noncomputable def second_hat_discount : ℝ := 0.2
noncomputable def third_hat_discount : ℝ := 0.25
noncomputable def fourth_hat_discount : ℝ := 0.6

-- Define the function to calculate the discounted price
noncomputable def discounted_price (discount : ℝ) : ℝ := regular_price * (1 - discount)

-- Define the total cost of four hats with the "festival special"
noncomputable def total_cost_with_discount : ℝ := 
  regular_price + 
  discounted_price second_hat_discount + 
  discounted_price third_hat_discount + 
  discounted_price fourth_hat_discount

-- Define the total cost of four hats without discount
noncomputable def total_cost_without_discount : ℝ := 4 * regular_price

-- Define the savings
noncomputable def savings : ℝ := total_cost_without_discount - total_cost_with_discount

-- Define the percentage saved
noncomputable def percentage_saved : ℝ := (savings / total_cost_without_discount) * 100

-- Theorem to prove
theorem festival_special_savings : percentage_saved = 26.25 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_festival_special_savings_l834_83422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_g_eq_3_has_three_solutions_l834_83467

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then -x + 4 else 3*x - 6

-- State the theorem
theorem g_g_eq_3_has_three_solutions :
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x : ℝ, g (g x) = 3 ↔ x ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_g_eq_3_has_three_solutions_l834_83467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l834_83456

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (a^2 - b^2) / a^2 = 3/4  -- eccentricity² = (√3/2)²
  h4 : a^2 + b^2 = 5  -- |CD| = √5

/-- The main theorem -/
theorem ellipse_properties (Γ : Ellipse) :
  (Γ.a = 2 ∧ Γ.b = 1) ∧
  ∀ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / Γ.a^2 + y₁^2 / Γ.b^2 = 1) →  -- A on Γ
    (x₂^2 / Γ.a^2 + y₂^2 / Γ.b^2 = 1) →  -- B on Γ
    (y₁ - 0) / (x₁ - Real.sqrt 3) = k →  -- Line AB has slope k
    (y₂ - 0) / (x₂ - Real.sqrt 3) = k →  -- and passes through right focus (√3, 0)
    x₁ * x₂ / Γ.a^2 + y₁ * y₂ / Γ.b^2 = 0 →
    (1/2) * abs (x₁ - x₂) * (abs (k * Real.sqrt 3) / Real.sqrt (1 + k^2)) = 1  -- Area of triangle AOB
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l834_83456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l834_83435

/-- Represents a circle with a center and radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if they intersect at exactly one point. -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- The distance between the centers of two circles. -/
def center_distance (c1 c2 : Circle) : ℝ := sorry

theorem tangent_circles_distance (c1 c2 : Circle) :
  are_tangent c1 c2 → c1.radius = 3 → c2.radius = 2 →
  center_distance c1 c2 = 1 ∨ center_distance c1 c2 = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_distance_l834_83435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_data_l834_83440

-- Define the universe
variable (U : Finset ℕ)

-- Define the sets as subsets of U
variable (A B C : Finset ℕ)

-- Define the cardinalities
axiom card_A : (A : Finset ℕ).card = 25
axiom card_B : (B : Finset ℕ).card = 30
axiom card_C : (C : Finset ℕ).card = 28
axiom card_A_inter_C : (A ∩ C).card = 18
axiom card_B_inter_C : (B ∩ C).card = 17
axiom card_A_inter_B : (A ∩ B).card = 16
axiom card_A_inter_B_inter_C : (A ∩ B ∩ C).card = 15
axiom card_U : U.card = 45

-- Define the theorem
theorem inconsistent_data : (A ∪ B ∪ C).card < U.card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inconsistent_data_l834_83440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l834_83434

noncomputable section

-- Define the functions f and g
def f (p q x : ℝ) : ℝ := x^2 + p*x + q
def g (x : ℝ) : ℝ := 2*x + 1/x^2

-- Define the interval
def interval : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem max_value_of_f (p q : ℝ) :
  (∃ x ∈ interval, ∀ y ∈ interval, f p q x ≤ f p q y ∧ g x ≤ g y) →
  (∃ x ∈ interval, f p q x = g x) →
  (∃ x ∈ interval, ∀ y ∈ interval, f p q y ≤ 4) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l834_83434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_ABCG_l834_83400

-- Define the cube ABCDEFGH
def cube_ABCDEFGH : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 2}

-- Define vertices A, B, C, and G
def A : Fin 3 → ℝ := λ _ => 0
def B : Fin 3 → ℝ := λ i => if i = 0 then 2 else 0
def C : Fin 3 → ℝ := λ i => if i < 2 then 2 else 0
def G : Fin 3 → ℝ := λ _ => 2

-- Define the volume of a pyramid
noncomputable def pyramid_volume (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Theorem statement
theorem volume_pyramid_ABCG :
  let base_area := (1 / 2) * 2 * 2
  let height := 2
  pyramid_volume base_area height = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_pyramid_ABCG_l834_83400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_haydens_hourly_wage_l834_83483

/-- Represents Hayden's work day as a limousine driver -/
structure LimoDriverDay where
  hourlyWage : ℝ
  hoursWorked : ℝ
  ridesGiven : ℕ
  goodReviews : ℕ
  gasGallons : ℝ
  gasPricePerGallon : ℝ
  rideBonus : ℝ
  reviewBonus : ℝ
  totalOwed : ℝ

/-- Calculates the total amount owed to Hayden based on his work day -/
def calculateTotalOwed (day : LimoDriverDay) : ℝ :=
  day.hourlyWage * day.hoursWorked +
  (day.ridesGiven : ℝ) * day.rideBonus +
  (day.goodReviews : ℝ) * day.reviewBonus +
  day.gasGallons * day.gasPricePerGallon

/-- Theorem stating that Hayden's hourly wage is $15 given the conditions of his work day -/
theorem haydens_hourly_wage (day : LimoDriverDay) 
  (h1 : day.hoursWorked = 8)
  (h2 : day.ridesGiven = 3)
  (h3 : day.goodReviews = 2)
  (h4 : day.gasGallons = 17)
  (h5 : day.gasPricePerGallon = 3)
  (h6 : day.rideBonus = 5)
  (h7 : day.reviewBonus = 20)
  (h8 : day.totalOwed = 226)
  (h9 : calculateTotalOwed day = day.totalOwed) :
  day.hourlyWage = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_haydens_hourly_wage_l834_83483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l834_83403

/-- The shortest distance from any point on y = ln x to y = x + 1 is √2 -/
theorem shortest_distance_ln_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (P : ℝ × ℝ), (P.2 = Real.log P.1) → 
  ∀ (Q : ℝ × ℝ), (Q.2 = Q.1 + 1) →
  d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  -- We'll use d = √2 as our witness
  use Real.sqrt 2
  constructor
  · -- Prove d = √2
    rfl
  · -- Prove the inequality
    intros P hP Q hQ
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l834_83403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_axis_intersection_l834_83474

/-- A circle with diameter endpoints (2,2) and (10,8) intersects the x-axis at a second point with x-coordinate 6 -/
theorem circle_x_axis_intersection :
  ∀ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) →
    ((2, 2) ∈ C ∧ (10, 8) ∈ C) →
    (∃ (x : ℝ), x ≠ 2 ∧ (x, 0) ∈ C) →
    (∃ (x : ℝ), x ≠ 2 ∧ (x, 0) ∈ C ∧ x = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_x_axis_intersection_l834_83474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_binomials_first_15_rows_l834_83486

/-- Counts the number of even binomial coefficients in a given row of Pascal's Triangle. -/
def countEvenBinomials (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (fun k => Nat.choose n k % 2 = 0) |>.length

/-- The sum of even binomial coefficients in the first 15 rows of Pascal's Triangle. -/
def sumEvenBinomials : ℕ :=
  (List.range 15).map countEvenBinomials |>.sum

theorem even_binomials_first_15_rows :
  sumEvenBinomials = 60 := by
  sorry

#eval sumEvenBinomials  -- Should output 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_binomials_first_15_rows_l834_83486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_at_2_0_l834_83484

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^x - 16 / (2^x)

-- Theorem statement
theorem f_symmetry_at_2_0 :
  ∀ x : ℝ, f (4 - x) = -f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_at_2_0_l834_83484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_and_tangent_l834_83453

noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (a b x : ℝ) : ℝ := a * x + b / x

theorem function_intersection_and_tangent 
  (a b : ℝ) 
  (h1 : g a b 1 = 0) 
  (h2 : deriv f 1 = deriv (g a b) 1) :
  (a = 1/2 ∧ b = -1/2) ∧
  (∀ x > 0, 
    (0 < x ∧ x ≤ 1 → f x ≥ g a b x) ∧ 
    (x > 1 → f x < g a b x)) := by
  sorry

#check function_intersection_and_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_intersection_and_tangent_l834_83453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cloth_donated_l834_83413

-- Define the initial cloth size
noncomputable def initial_cloth_size : ℝ := 100

-- Define the number of times the cloth is cut
def num_cuts : ℕ := 2

-- Function to calculate the amount of cloth donated after each cut
noncomputable def cloth_donated_per_cut (remaining_cloth : ℝ) : ℝ := remaining_cloth / 2

-- Function to calculate the remaining cloth after each cut
noncomputable def remaining_cloth_after_cut (remaining_cloth : ℝ) : ℝ := remaining_cloth / 2

-- Theorem stating the total amount of cloth donated
theorem total_cloth_donated :
  (cloth_donated_per_cut initial_cloth_size +
   cloth_donated_per_cut (remaining_cloth_after_cut initial_cloth_size)) = 75 := by
  sorry

#eval num_cuts -- This will compile and run

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cloth_donated_l834_83413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_x_times_one_plus_x_to_six_l834_83406

theorem coefficient_x_cube_in_x_times_one_plus_x_to_six :
  (Polynomial.coeff (Polynomial.X * (1 + Polynomial.X)^6 : Polynomial ℚ) 3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cube_in_x_times_one_plus_x_to_six_l834_83406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_zero_sum_six_even_zero_sum_l834_83442

def Grid (n : ℕ) := Fin n → Fin n → Int

def rowSum (g : Grid n) (i : Fin n) : Int :=
  (Finset.univ.sum fun j => g i j)

def colSum (g : Grid n) (j : Fin n) : Int :=
  (Finset.univ.sum fun i => g i j)

def Sn (g : Grid n) : Int :=
  (Finset.univ.sum fun i => rowSum g i) + (Finset.univ.sum fun j => colSum g j)

def isValidLabeling (g : Grid n) : Prop :=
  ∀ i j, g i j = 1 ∨ g i j = -1

theorem no_odd_zero_sum (n : ℕ) (h : n ≥ 2) (hodd : Odd n) :
  ¬∃ g : Grid n, isValidLabeling g ∧ Sn g = 0 := by sorry

theorem six_even_zero_sum (n : ℕ) (h : n ≥ 2) (heven : Even n) :
  ∃ g₁ g₂ g₃ g₄ g₅ g₆ : Grid n,
    (∀ i, i ∈ [g₁, g₂, g₃, g₄, g₅, g₆] → isValidLabeling i ∧ Sn i = 0) ∧
    (∀ i j, i ∈ [g₁, g₂, g₃, g₄, g₅, g₆] → j ∈ [g₁, g₂, g₃, g₄, g₅, g₆] → i ≠ j → i ≠ j) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_odd_zero_sum_six_even_zero_sum_l834_83442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_invertible_interval_for_g_l834_83458

def g (x : ℝ) := 3 * x^2 - 6 * x - 2

theorem largest_invertible_interval_for_g :
  ∀ a b : ℝ, a < 1 ∧ 1 ≤ b →
  (∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → g x = g y → x = y) →
  b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_invertible_interval_for_g_l834_83458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l834_83421

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}
def Q : Set ℝ := {x : ℝ | |x| < 3}

-- Define the union of P and Q
def PUnionQ : Set ℝ := P ∪ Q

-- State the theorem
theorem union_of_P_and_Q : PUnionQ = Set.Ioc (-3) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_P_and_Q_l834_83421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l834_83452

theorem trigonometric_problem (α : ℝ)
  (h1 : Real.sin α = -(2 * Real.sqrt 5) / 5)
  (h2 : Real.tan α < 0) :
  (Real.tan α = -2) ∧
  ((2 * Real.sin (α + π) + Real.cos (2 * π - α)) / (Real.cos (α - π / 2) - Real.sin (3 * π / 2 + α)) = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l834_83452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_uniqueness_l834_83408

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  medianA : ℝ
  medianB : ℝ
  medianC : ℝ

-- Define the properties
noncomputable def anglesBetweenSides (t : Triangle) : ℝ × ℝ × ℝ := (t.angleA, t.angleB, t.angleC)
noncomputable def ratioOfSideLengths (t : Triangle) : ℝ × ℝ × ℝ := (t.a / t.b, t.b / t.c, t.c / t.a)
noncomputable def ratioSideToMedian (t : Triangle) : ℝ := t.a / t.medianA
noncomputable def sumOfAngles (t : Triangle) : ℝ := t.angleA + t.angleB + t.angleC
noncomputable def oneAngleAndOppositeSide (t : Triangle) : ℝ × ℝ := (t.angleA, t.a)

-- Theorem statement
theorem triangle_uniqueness :
  (∃ t1 t2 : Triangle, t1 ≠ t2 ∧ ratioSideToMedian t1 = ratioSideToMedian t2) ∧
  (∀ t1 t2 : Triangle, anglesBetweenSides t1 = anglesBetweenSides t2 → t1 = t2) ∧
  (∀ t1 t2 : Triangle, ratioOfSideLengths t1 = ratioOfSideLengths t2 → t1 = t2) ∧
  (∀ t1 t2 : Triangle, sumOfAngles t1 = sumOfAngles t2) ∧
  (∀ t1 t2 : Triangle, oneAngleAndOppositeSide t1 = oneAngleAndOppositeSide t2 → t1 ≠ t2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_uniqueness_l834_83408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_recolorable_column_l834_83418

/-- Represents a coloring of an n × n chessboard -/
def Coloring (n m : ℕ+) := Fin n → Fin n → Fin m

/-- Predicate to check if two rows are colored exactly the same -/
def RowsIdentical (c : Coloring n m) (i j : Fin n) : Prop :=
  ∀ k : Fin n, c i k = c j k

/-- Predicate to check if no two rows are colored exactly the same -/
def NoIdenticalRows (c : Coloring n m) : Prop :=
  ∀ i j : Fin n, i ≠ j → ¬(RowsIdentical c i j)

/-- Function to recolor a column to white -/
def RecolorColumn (c : Coloring n m) (col : Fin n) : Coloring n m :=
  fun i j => if j = col then (0 : Fin m) else c i j

/-- Theorem stating that there exists a column that can be recolored white
    such that no two rows are still colored exactly the same after recoloring -/
theorem exists_recolorable_column (n m : ℕ+) (c : Coloring n m) 
  (h : NoIdenticalRows c) : 
  ∃ col : Fin n, NoIdenticalRows (RecolorColumn c col) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_recolorable_column_l834_83418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cream_fraction_after_transfer_l834_83446

/-- Represents a cup with a certain amount of coffee and cream -/
structure Cup where
  coffee : ℚ
  cream : ℚ

/-- The process of transferring liquid between cups -/
def transfer (cup1 cup2 : Cup) (amount : ℚ) : Cup × Cup :=
  let total := cup1.coffee + cup1.cream
  let coffee_ratio := cup1.coffee / total
  let cream_ratio := cup1.cream / total
  let new_cup1 : Cup := {
    coffee := cup1.coffee - amount * coffee_ratio,
    cream := cup1.cream - amount * cream_ratio
  }
  let new_cup2 : Cup := {
    coffee := cup2.coffee + amount * coffee_ratio,
    cream := cup2.cream + amount * cream_ratio
  }
  (new_cup1, new_cup2)

theorem cream_fraction_after_transfer :
  let cup1_initial : Cup := { coffee := 4, cream := 0 }
  let cup2_initial : Cup := { coffee := 0, cream := 4 }
  let (cup1_mid, cup2_mid) := transfer cup1_initial cup2_initial 2
  let (cup1_final, _) := transfer cup2_mid cup1_mid 3
  cup1_final.cream / (cup1_final.coffee + cup1_final.cream) = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cream_fraction_after_transfer_l834_83446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l834_83425

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem function_symmetry 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + 4 * Real.pi) = f ω x) 
  (h_smallest_period : ∀ T, T > 0 → (∀ x, f ω (x + T) = f ω x) → T ≥ 4 * Real.pi) :
  ∀ x, f ω (x - Real.pi / 3) = -f ω (-x - Real.pi / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l834_83425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberries_count_l834_83414

/-- Represents the number of blueberries in each blue box -/
def blueberries_per_box : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries_per_box : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def berry_increase : ℕ := 15

/-- The increase in difference between strawberries and blueberries when replacing a blue box with a red box -/
def difference_increase : ℕ := 87

theorem blueberries_count : 
  (strawberries_per_box - blueberries_per_box = berry_increase) →
  (strawberries_per_box + blueberries_per_box = difference_increase) →
  blueberries_per_box = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blueberries_count_l834_83414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l834_83460

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the parallel lines
def line1 (p : ℝ × ℝ) : Prop := 4 * p.1 + 3 * p.2 + 1 = 0
def line2 (p : ℝ × ℝ) : Prop := 4 * p.1 + 3 * p.2 + 6 = 0

-- Define the length of the intercepted segment
noncomputable def segment_length : ℝ := Real.sqrt 2

-- Define the possible equations of line l
def line_eq1 (p : ℝ × ℝ) : Prop := p.1 + 7 * p.2 = 15
def line_eq2 (p : ℝ × ℝ) : Prop := 7 * p.1 - p.2 = 5

-- Theorem statement
theorem line_equation :
  ∃ (l : (ℝ × ℝ) → Prop),
    (l M) ∧
    (∃ (A B : ℝ × ℝ),
      line1 A ∧ line2 B ∧
      l A ∧ l B ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = segment_length) →
    ((∀ p, l p ↔ line_eq1 p) ∨ (∀ p, l p ↔ line_eq2 p)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l834_83460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l834_83410

noncomputable section

open Real

theorem triangle_ratio_proof (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- C = π/4
  C = π / 4 →
  -- c = 2√2
  c = 2 * sqrt 2 →
  -- a, b, c are sides opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- The ratio equals 4
  (a + 2 * c) / (sin A + 2 * sin C) = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l834_83410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_achievable_forty_percent_solution_l834_83472

/-- Represents the state of the two containers -/
structure ContainerState where
  living_water : ℝ
  ordinary_water : ℝ

/-- Checks if a given state represents a 40% solution of living water -/
def is_forty_percent_solution (state : ContainerState) : Prop :=
  state.living_water / (state.living_water + state.ordinary_water) = 0.4

/-- Represents a transfer operation between containers -/
def transfer (state1 state2 : ContainerState) (amount : ℝ) : ContainerState × ContainerState :=
  sorry

/-- Checks if it's possible to achieve a 40% solution through transfers -/
def can_achieve_forty_percent (a : ℝ) : Prop :=
  ∃ (n : ℕ) (transfers : Fin n → ℝ),
    let initial_state : ContainerState × ContainerState := ⟨⟨a, 0⟩, ⟨0, 1⟩⟩
    let final_state := (List.foldl (fun (s : ContainerState × ContainerState) (t : ℝ) => transfer s.1 s.2 t) initial_state (List.ofFn transfers))
    (is_forty_percent_solution final_state.1 ∨ is_forty_percent_solution final_state.2)

/-- The main theorem stating when it's possible to achieve a 40% solution -/
theorem achievable_forty_percent_solution (a : ℝ) :
    0 < a ∧ a < 1 ∧ a ≠ 2/3 ↔ can_achieve_forty_percent a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_achievable_forty_percent_solution_l834_83472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_z_27_l834_83457

-- Define the variables and constants
variable (x y z m n : ℝ)

-- Define the conditions
def directly_proportional (x y z m : ℝ) : Prop :=
  x = m * y^3

def inversely_proportional (y z n : ℝ) : Prop :=
  y = n / z^(1/3)

def initial_condition (x z : ℝ) : Prop :=
  (z = 8) → (x = 3)

-- State the theorem
theorem x_value_when_z_27 
  (h1 : directly_proportional x y z m)
  (h2 : inversely_proportional y z n)
  (h3 : initial_condition x z) :
  (z = 27) → (x = 8/9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_z_27_l834_83457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_is_40_l834_83441

/-- The area of a rectangular garden in square meters -/
noncomputable def garden_area (width_cm : ℝ) (length_cm : ℝ) : ℝ :=
  (width_cm * length_cm) / 10000

/-- Theorem: The area of a rectangular garden with width 500 cm and length 800 cm is 40 square meters -/
theorem garden_area_is_40 :
  garden_area 500 800 = 40 := by
  -- Unfold the definition of garden_area
  unfold garden_area
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_is_40_l834_83441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l834_83478

/-- The number of ways to arrange 4 distinct characters in a row -/
def arrange_four_distinct : ℕ := 24  -- 4! = 24

/-- The number of ways to choose 3 positions from 5 available spaces -/
def choose_three_from_five : ℕ := Nat.choose 5 3

/-- The number of ways to choose 3 positions from 4 available spaces -/
def choose_three_from_four : ℕ := Nat.choose 4 3

/-- The number of arrangements where A and A are not adjacent -/
def arrangements_A_not_adjacent : ℕ := 2 * arrange_four_distinct

/-- The total number of arrangements satisfying all conditions -/
def total_arrangements : ℕ := arrangements_A_not_adjacent * choose_three_from_five - 
                               arrange_four_distinct * choose_three_from_four

theorem arrangement_count : total_arrangements = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_l834_83478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_12th_row_l834_83461

/-- Represents a lattice with a given number of rows -/
structure MyLattice where
  rows : ℕ
  elements_per_row : ℕ
  diff_between_consecutive : ℕ
  last_number : ℕ → ℕ

/-- The fourth number in the nth row of a lattice -/
def fourth_number (l : MyLattice) (n : ℕ) : ℕ :=
  l.last_number n - (l.elements_per_row - 4) * l.diff_between_consecutive

theorem fourth_number_12th_row :
  ∀ l : MyLattice,
    l.rows = 12 →
    l.elements_per_row = 5 →
    l.diff_between_consecutive = 1 →
    (∀ n, l.last_number n = 5 * n) →
    fourth_number l 12 = 58 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_12th_row_l834_83461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equality_l834_83465

theorem sqrt_expressions_equality :
  (∀ (x : ℝ), x > 0 → ∃ y : ℝ, y > 0 ∧ y * y = x) →
  (Real.sqrt 27 + Real.sqrt (1/3) - Real.sqrt 12 = (4 * Real.sqrt 3) / 3) ∧
  (Real.sqrt 6 * Real.sqrt 2 + Real.sqrt 24 / Real.sqrt 3 - Real.sqrt 48 = 2 * Real.sqrt 2 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equality_l834_83465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l834_83485

/-- The function f(x) = -e^x - x -/
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

/-- The function g(x) = ax + 2cos(x) -/
noncomputable def g (a x : ℝ) : ℝ := a * x + 2 * Real.cos x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := -Real.exp x - 1

/-- The derivative of g(x) -/
noncomputable def g_deriv (a x : ℝ) : ℝ := a - 2 * Real.sin x

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f_deriv x * g_deriv a y = -1) ↔ a ∈ Set.Icc (-1 : ℝ) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l834_83485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_of_triangle_l834_83448

theorem smallest_perimeter_of_triangle (d e f : ℝ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2) / (2 * d * e) = 15/17 →
  (e^2 + f^2 - d^2) / (2 * e * f) = 3/5 →
  (f^2 + d^2 - e^2) / (2 * f * d) = -1/8 →
  d + e + f ≥ 504 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_of_triangle_l834_83448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_area_l834_83424

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by four points -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A star shape defined by eight points -/
structure StarShape where
  A : Point
  E : Point
  B : Point
  F : Point
  C : Point
  G : Point
  D : Point
  H : Point

/-- The area of a shape -/
noncomputable def area (s : Square) : ℝ := sorry

/-- Check if a point is on the edge of a square -/
def onEdge (p : Point) (s : Square) : Prop := sorry

/-- The main theorem -/
theorem star_area (s : Square) (star : StarShape) :
  area s = 72 →
  onEdge star.E s →
  onEdge star.F s →
  onEdge star.G s →
  onEdge star.H s →
  star.A = s.A →
  star.B = s.B →
  star.C = s.C →
  star.D = s.D →
  ∃ (starArea : ℝ), starArea = 24 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_area_l834_83424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_configurations_count_l834_83405

/-- Represents a grid of lightbulbs -/
def LightGrid := Fin 20 → Fin 16 → Bool

/-- Represents the set of switches -/
def Switches := Fin 36 → Bool

/-- Applies a switch configuration to a grid -/
def applySwitch (grid : LightGrid) (switches : Switches) : LightGrid :=
  sorry

/-- Checks if two grids are distinct -/
def isDistinct (grid1 grid2 : LightGrid) : Prop :=
  ∃ (i : Fin 20) (j : Fin 16), grid1 i j ≠ grid2 i j

/-- The set of all possible grid configurations -/
noncomputable def allConfigurations : Set LightGrid :=
  {grid | ∃ switches, grid = applySwitch (fun _ _ => false) switches}

/-- Proof that the set of configurations is finite -/
instance : Fintype allConfigurations := sorry

theorem distinct_configurations_count :
  Fintype.card allConfigurations = 2^35 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_configurations_count_l834_83405
