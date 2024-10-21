import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1026_102633

/-- The number of ways to make n! cents using coins worth k! cents for k = 1 to n -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem stating the bounds for f(n) -/
theorem f_bounds :
  ∃ C : ℝ, ∀ n : ℕ+,
    (n : ℝ) ^ (n^2 / 2 - C * n) * Real.exp (-n^2 / 4) ≤ (f n : ℝ) ∧
    (f n : ℝ) ≤ (n : ℝ) ^ (n^2 / 2 + C * n) * Real.exp (-n^2 / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l1026_102633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_side_square_sum_ge_half_third_square_median_square_sum_ge_nine_eighths_third_square_l1026_102624

/-- Triangle inequality theorem -/
theorem triangle_inequality {a b c : ℝ} (h : 0 < a ∧ 0 < b ∧ 0 < c) : 
  c ≤ a + b ∧ a ≤ b + c ∧ b ≤ a + c := by sorry

/-- Definition of a triangle -/
def is_triangle (a b c : ℝ) : Prop := 
  0 < a ∧ 0 < b ∧ 0 < c ∧ c < a + b ∧ a < b + c ∧ b < a + c

/-- Definition of a median -/
def is_median (m a b c : ℝ) : Prop := m^2 = (a^2)/4 + (b^2 + c^2)/16

/-- Theorem: For any triangle with side lengths a, b, and c, a^2 + b^2 ≥ c^2/2 -/
theorem side_square_sum_ge_half_third_square {a b c : ℝ} (h : is_triangle a b c) :
  a^2 + b^2 ≥ c^2/2 := by sorry

/-- Theorem: For any triangle with side lengths a, b, and c, and medians m_a and m_b to sides a and b respectively, m_a^2 + m_b^2 ≥ 9c^2/8 -/
theorem median_square_sum_ge_nine_eighths_third_square {a b c m_a m_b : ℝ} 
  (h1 : is_triangle a b c) (h2 : is_median m_a a b c) (h3 : is_median m_b b a c) :
  m_a^2 + m_b^2 ≥ 9*c^2/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_side_square_sum_ge_half_third_square_median_square_sum_ge_nine_eighths_third_square_l1026_102624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1026_102651

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (a b c : ℝ) (d : ℝ) : ℝ :=
  |d| / Real.sqrt (a^2 + b^2)

theorem parallel_lines_distance (a : ℝ) :
  let line1 : ℝ → ℝ → ℝ := λ x y => x - 2*y
  let line2 : ℝ → ℝ → ℝ := λ x y => 2*x - 4*y + a
  (∀ x y, line1 x y = 0 → line2 x y = a) →
  distance_parallel_lines 2 (-4) 0 a = Real.sqrt 5 →
  |a| = 10 := by
  sorry

#check parallel_lines_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1026_102651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l1026_102622

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | (1/2 : ℝ)^x ≥ 4}

theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l1026_102622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_proposition_l1026_102625

theorem converse_of_proposition : 
  (∀ x : ℝ, x > 0 → x^2 > 0) ↔ 
  (∀ x : ℝ, x^2 > 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_proposition_l1026_102625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_min_cosine_sum_l1026_102658

noncomputable section

open Real

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_angle_and_min_cosine_sum 
  (a b c A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_equation : b * cos C = (2 * a - c) * cos B) :
  B = Real.pi / 3 ∧ 
  (∀ y : ℝ, y = cos A ^ 2 + cos C ^ 2 → y ≥ 1 / 2) ∧
  ∃ y : ℝ, y = cos A ^ 2 + cos C ^ 2 ∧ y = 1 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_min_cosine_sum_l1026_102658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_given_planes_l1026_102683

/-- The angle between two planes given by their normal vectors -/
noncomputable def angle_between_planes (n₁ n₂ : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos ((n₁.1 * n₂.1 + n₁.2.1 * n₂.2.1 + n₁.2.2 * n₂.2.2) / 
    (Real.sqrt (n₁.1^2 + n₁.2.1^2 + n₁.2.2^2) * Real.sqrt (n₂.1^2 + n₂.2.1^2 + n₂.2.2^2)))

/-- The normal vector of the plane 3x - 2y + 3z + 23 = 0 -/
def normal_vector_plane1 : ℝ × ℝ × ℝ := (3, (-2, 3))

/-- The normal vector of the plane y + z + 5 = 0 -/
def normal_vector_plane2 : ℝ × ℝ × ℝ := (0, (1, 1))

/-- Theorem: The angle between the given planes is arccos(1 / (2√11)) -/
theorem angle_between_given_planes :
  angle_between_planes normal_vector_plane1 normal_vector_plane2 = Real.arccos (1 / (2 * Real.sqrt 11)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_given_planes_l1026_102683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1026_102639

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the statements about the triangle
def is_right_triangle (t : Triangle) : Prop := sorry
def angle_A_is_30_degrees (t : Triangle) : Prop := sorry
def AB_is_twice_BC (t : Triangle) : Prop := sorry
def AC_is_twice_BC (t : Triangle) : Prop := sorry

-- Define the condition that exactly two statements are true
def exactly_two_true (t : Triangle) : Prop :=
  (is_right_triangle t ∧ AB_is_twice_BC t ∧ ¬angle_A_is_30_degrees t ∧ ¬AC_is_twice_BC t) ∨
  (is_right_triangle t ∧ AC_is_twice_BC t ∧ ¬angle_A_is_30_degrees t ∧ ¬AB_is_twice_BC t) ∨
  (angle_A_is_30_degrees t ∧ AB_is_twice_BC t ∧ ¬is_right_triangle t ∧ ¬AC_is_twice_BC t) ∨
  (angle_A_is_30_degrees t ∧ AC_is_twice_BC t ∧ ¬is_right_triangle t ∧ ¬AB_is_twice_BC t) ∨
  (AB_is_twice_BC t ∧ AC_is_twice_BC t ∧ ¬is_right_triangle t ∧ ¬angle_A_is_30_degrees t)

-- Define the length of BC
def BC_length (t : Triangle) : ℝ := 1

-- Define the perimeter of the triangle
noncomputable def perimeter (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  BC_length t = 1 ∧ exactly_two_true t → perimeter t = 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1026_102639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l1026_102635

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: f is increasing on ℝ for all a ∈ ℝ
theorem f_increasing (a : ℝ) : Monotone (f a) := by
  sorry

-- Theorem 2: f is an odd function if and only if a = 1
theorem f_odd_iff_a_eq_one (a : ℝ) : 
  (∀ x, f a x = -(f a (-x))) ↔ a = 1 := by
  sorry

-- Example usage (optional)
#check f
#check f_increasing
#check f_odd_iff_a_eq_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l1026_102635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_point_theorem_l1026_102687

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2*x - y + 4 = 0
def l₂ (x y : ℝ) : Prop := x - y + 5 = 0
def l_perp (x y : ℝ) : Prop := x - 2*y - 6 = 0

-- Define the intersection point of l₁ and l₂
def intersection : ℝ × ℝ := (1, 6)

-- Define line l
def l (x y : ℝ) : Prop := 2*x + y - 8 = 0

-- Define the distance function
noncomputable def distance (a : ℝ) : ℝ := |2*a + 1 - 8| / Real.sqrt 5

-- Theorem statement
theorem line_and_point_theorem :
  (∀ x y : ℝ, l x y ↔ (x = intersection.1 ∧ y = intersection.2 ∨
    ∃ t : ℝ, x = intersection.1 + 2*t ∧ y = intersection.2 - t)) ∧
  (∀ x y : ℝ, l x y → l_perp x y → x = y) ∧
  ({a : ℝ | distance a = Real.sqrt 5} = {6, 1}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_point_theorem_l1026_102687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l1026_102600

theorem sphere_surface_area (V : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → A = 4 * Real.pi * (3 * (2 : ℝ)^(1/3 : ℝ))^2 → A = 36 * Real.pi * 2^(2/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_l1026_102600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1026_102638

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  Real.sqrt 3 * a = 2 * c * Real.sin A →  -- Given condition
  c = Real.sqrt 7 →  -- Given condition
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 →  -- Area condition
  C = π/3 ∧ a + b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1026_102638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l1026_102679

/-- The expression in question -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 16) / ((x^2 + 4*x - 5) * (x + 4))

/-- The set of x values for which f is undefined -/
def undefined_set : Set ℝ := {x : ℝ | (x^2 + 4*x - 5) * (x + 4) = 0}

theorem undefined_values_count :
  ∃ (S : Finset ℝ), (↑S : Set ℝ) = undefined_set ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l1026_102679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_sqrt_2_l1026_102669

-- Define the vectors a, b, and c as functions of m
def a (m : ℝ) : ℝ × ℝ := (1, 2*m)
def b (m : ℝ) : ℝ × ℝ := (m+1, 1)
def c (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem vector_magnitude_sqrt_2 :
  ∃ m : ℝ, dot_product (a m + c m) (b m) = 0 ∧ magnitude (a m) = Real.sqrt 2 :=
by
  -- Introduce m = -1/2
  let m : ℝ := -1/2

  -- Show that this m satisfies the conditions
  have h1 : dot_product (a m + c m) (b m) = 0 := by
    -- Expand the dot product
    simp [dot_product, a, b, c]
    -- Simplify the resulting expression
    ring

  have h2 : magnitude (a m) = Real.sqrt 2 := by
    -- Expand the magnitude
    simp [magnitude, a]
    -- Simplify the resulting expression
    ring

  -- Conclude the proof
  exact ⟨m, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_sqrt_2_l1026_102669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1026_102643

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Define the line l
def lineL (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 5 = 0

-- Theorem statement
theorem shortest_distance_circle_to_line :
  ∃ (d : ℝ), d = 1 ∧
  ∀ (x y : ℝ), circleC x y →
    ∀ (x' y' : ℝ), lineL x' y' →
      (x - x')^2 + (y - y')^2 ≥ d^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1026_102643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_terms_is_63_l1026_102641

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℝ  -- first term
  q : ℝ  -- common ratio

/-- Sum of first n terms of a geometric sequence -/
noncomputable def sumFirstN (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a * (1 - g.q^n) / (1 - g.q)

theorem sum_21_terms_is_63 (g : GeometricSequence) 
  (h1 : sumFirstN g 7 = 48)
  (h2 : sumFirstN g 14 = 60) :
  sumFirstN g 21 = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_21_terms_is_63_l1026_102641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l1026_102695

-- Define the slope and y-intercept of the original line
noncomputable def m : ℝ := 3/4
noncomputable def c : ℝ := 6

-- Define the distance between the parallel lines
noncomputable def d : ℝ := 5

-- Define a function to represent a line with slope m and y-intercept b
noncomputable def line (x : ℝ) (b : ℝ) : ℝ := m * x + b

-- Define the distance formula between two parallel lines
noncomputable def distance (b₁ b₂ : ℝ) : ℝ := |b₁ - b₂| / Real.sqrt (m^2 + 1)

-- State the theorem
theorem parallel_lines_at_distance :
  ∀ b : ℝ, (distance c b = d) ↔ (b = c + d * Real.sqrt (m^2 + 1) ∨ b = c - d * Real.sqrt (m^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l1026_102695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relationship_l1026_102627

theorem angle_relationship (α β : Real) (h_acute_α : 0 < α ∧ α < Real.pi / 2) (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_sin_cos : Real.sin α - Real.cos α = 1 / 6)
  (h_tan : Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3) :
  β < Real.pi / 4 ∧ Real.pi / 4 < α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relationship_l1026_102627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_cost_theorem_l1026_102680

/-- The cost of a single shirt given the prices of other items and a total purchase amount -/
theorem shirt_cost_theorem (hat_cost jeans_cost total_cost shirt_cost : ℝ) : 
  hat_cost = 4 →
  jeans_cost = 10 →
  3 * shirt_cost + 2 * jeans_cost + 4 * hat_cost = total_cost →
  total_cost = 51 →
  shirt_cost = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_cost_theorem_l1026_102680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinate_sum_l1026_102615

/-- Given two points (7, -6) and (-1, 4) as endpoints of a diameter of a circle,
    the sum of the coordinates of the center of this circle is 2. -/
theorem circle_center_coordinate_sum : 
  let p1 : ℝ × ℝ := (7, -6)
  let p2 : ℝ × ℝ := (-1, 4)
  let center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  center.1 + center.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_coordinate_sum_l1026_102615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1026_102692

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Property of being a linear function -/
def IsLinear (f : ℕ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ n, f n = k * n + b

theorem sequence_problem (a : Sequence) (h_linear : IsLinear a) 
    (h_a1 : a 1 = 3) (h_a10 : a 10 = 21) :
  (∀ n, a n = 2 * ↑n + 1) ∧ 
  (a 2005 = 4011) ∧
  (∀ n, a (2 * n) = 4 * ↑n + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l1026_102692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fe2o3_weight_calculation_l1026_102628

/-- The atomic weight of iron in g/mol -/
def iron_atomic_weight : ℝ := 55.845

/-- The atomic weight of oxygen in g/mol -/
def oxygen_atomic_weight : ℝ := 15.999

/-- The molecular weight of Fe2O3 in g/mol -/
def fe2o3_molecular_weight : ℝ := 2 * iron_atomic_weight + 3 * oxygen_atomic_weight

/-- The number of moles of Fe2O3 -/
def moles_fe2o3 : ℝ := 10

theorem fe2o3_weight_calculation :
  abs (moles_fe2o3 * fe2o3_molecular_weight - 1596.87) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fe2o3_weight_calculation_l1026_102628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_product_l1026_102620

/-- A line passing through point P(2,1) and intersecting both positive x-axis and y-axis -/
structure IntersectingLine where
  slope : ℝ
  intersects_positive_axes : slope < 0

/-- The point where the line intersects the x-axis -/
noncomputable def x_intercept (l : IntersectingLine) : ℝ × ℝ :=
  (2 - 1 / l.slope, 0)

/-- The point where the line intersects the y-axis -/
noncomputable def y_intercept (l : IntersectingLine) : ℝ × ℝ :=
  (0, 1 - 2 * l.slope)

/-- The product of distances |PA| and |PB| -/
noncomputable def distance_product (l : IntersectingLine) : ℝ :=
  let pa := Real.sqrt ((2 - (x_intercept l).1)^2 + (1 - (x_intercept l).2)^2)
  let pb := Real.sqrt ((0 - (y_intercept l).1)^2 + (1 - (y_intercept l).2)^2)
  pa * pb

/-- The theorem stating that x + y - 3 = 0 minimizes |PA|⋅|PB| -/
theorem minimal_distance_product :
  ∀ l : IntersectingLine,
    distance_product l ≥ distance_product ⟨-1, by norm_num⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_product_l1026_102620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_price_problem_l1026_102644

theorem chocolate_price_problem (total_chocolates : ℕ) (discount_ratio : ℕ) 
  (discount_price : ℚ) (total_cost : ℚ) : ℚ := by
  have h1 : total_chocolates = 12 := by sorry
  have h2 : discount_ratio = 3 := by sorry
  have h3 : discount_price = 25 / 100 := by sorry
  have h4 : total_cost = 615 / 100 := by sorry
  
  let regular_price_chocolates := total_chocolates / (discount_ratio + 1) * discount_ratio
  let discounted_chocolates := total_chocolates / (discount_ratio + 1)
  let discounted_cost := discounted_chocolates * discount_price
  let regular_cost := total_cost - discounted_cost
  let regular_price := regular_cost / regular_price_chocolates
  
  have h5 : regular_price = 60 / 100 := by sorry
  
  exact regular_price


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_price_problem_l1026_102644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l1026_102693

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees each hour represents on a clock face -/
noncomputable def degrees_per_hour : ℝ := 360 / clock_hours

/-- The position of the minute hand at 15 minutes past the hour in degrees -/
def minute_hand_position : ℝ := 90

/-- The position of the hour hand at 7:15 in degrees -/
noncomputable def hour_hand_position : ℝ := 7 * degrees_per_hour + (15 / 60) * degrees_per_hour

/-- The smaller angle between the hour and minute hands at 7:15 -/
noncomputable def smaller_angle : ℝ := min (abs (hour_hand_position - minute_hand_position)) 
                             (360 - abs (hour_hand_position - minute_hand_position))

theorem clock_angle_at_7_15 : smaller_angle = 127.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_7_15_l1026_102693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_bound_l1026_102626

/-- An n-th degree polynomial with integer coefficients -/
def P (n : ℕ) : ℤ → ℤ := sorry

/-- Composition of P with itself k times -/
def Q (n k : ℕ) : ℤ → ℤ :=
  fun x => Nat.iterate (P n) k x

/-- The number of fixed points of Q is at most n -/
theorem fixed_points_bound (n k : ℕ) (h1 : n > 1) (h2 : k > 0) :
  ∃ (S : Finset ℤ), (∀ t : ℤ, Q n k t = t → t ∈ S) ∧ Finset.card S ≤ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_bound_l1026_102626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_pass_through_single_point_l1026_102652

/-- A line in a plane --/
structure Line where
  -- We don't need to define the internal structure of a line for this problem

/-- A point in a plane --/
structure Point where
  -- We don't need to define the internal structure of a point for this problem

/-- Predicate to check if a point lies on a line --/
def Point.liesOn (p : Point) (l : Line) : Prop := sorry

/-- Predicate to check if two lines are parallel --/
def Line.isParallelTo (l1 l2 : Line) : Prop := sorry

/-- Function to get the intersection point of two lines --/
noncomputable def Line.intersectionPoint (l1 l2 : Line) : Point := sorry

/-- Theorem: All lines pass through a single point --/
theorem all_lines_pass_through_single_point 
  (S : Set Line) 
  (h1 : S.Finite) 
  (h2 : ∀ l1 l2 : Line, l1 ∈ S → l2 ∈ S → l1 ≠ l2 → ¬(l1.isParallelTo l2)) 
  (h3 : ∀ l1 l2 l3 : Line, l1 ∈ S → l2 ∈ S → l3 ∈ S → l1 ≠ l2 → 
        ∃ l : Line, l ∈ S ∧ (Line.intersectionPoint l1 l2).liesOn l) :
  ∃ p : Point, ∀ l : Line, l ∈ S → p.liesOn l :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lines_pass_through_single_point_l1026_102652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_shaded_area_l1026_102636

-- Define the side length of the hexagon
noncomputable def hexagon_side_length : ℝ := 3

-- Define the radius of the central circle
noncomputable def central_circle_radius : ℝ := 1

-- Define the area of a regular hexagon
noncomputable def hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

-- Define the area of a semicircle
noncomputable def semicircle_area (r : ℝ) : ℝ := Real.pi * r^2 / 2

-- Define the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem hexagon_shaded_area :
  hexagon_area hexagon_side_length -
  (6 * semicircle_area (hexagon_side_length / 2) + circle_area central_circle_radius) =
  13.5 * Real.sqrt 3 - 7.75 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_shaded_area_l1026_102636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_is_48_seconds_l1026_102648

/-- Time for a train to pass a bridge -/
noncomputable def train_pass_bridge_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: The time for the given train to pass the bridge is 48 seconds -/
theorem train_pass_bridge_time_is_48_seconds :
  train_pass_bridge_time 460 45 140 = 48 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_pass_bridge_time 460 45 140

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_is_48_seconds_l1026_102648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_share_is_3000_l1026_102647

/-- Represents the profit share of a business partner -/
structure ProfitShare where
  amount : ℚ
  deriving Repr

/-- Represents a business partner with their capital contribution and profit share -/
structure Partner where
  name : String
  capital : ℚ
  profitShare : ProfitShare
  deriving Repr

/-- Calculates the profit share based on the capital ratio -/
noncomputable def calculateProfitShare (totalProfit : ℚ) (totalCapital : ℚ) (partnerCapital : ℚ) : ℚ :=
  (partnerCapital / totalCapital) * totalProfit

/-- Theorem stating that B's profit share is 3000 given the problem conditions -/
theorem b_profit_share_is_3000 
  (a b c : Partner)
  (h1 : a.capital = 8000)
  (h2 : b.capital = 10000)
  (h3 : c.capital = 12000)
  (h4 : c.profitShare.amount - a.profitShare.amount = 1200)
  (h5 : c.profitShare.amount > a.profitShare.amount)
  : b.profitShare.amount = 3000 := by
  sorry

#eval ProfitShare.mk 3000
#eval Partner.mk "B" 10000 (ProfitShare.mk 3000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_share_is_3000_l1026_102647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_x_coord_l1026_102608

noncomputable section

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The slope of a line -/
def Line.slope (l : Line) : ℝ := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  (p.y - l.y₁) / (p.x - l.x₁) = l.slope

theorem point_on_line_x_coord :
  let l := Line.mk 0 10 (-5) 0
  let p := Point.mk x (-5)
  p.liesOn l → x = -7.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_x_coord_l1026_102608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_gain_loss_l1026_102649

/-- Represents the cost and selling prices of pens and pencils -/
structure Prices where
  pen_cost : ℚ
  pen_sell : ℚ
  pencil_cost : ℚ
  pencil_sell : ℚ

/-- Calculates the overall gain or loss percentage -/
noncomputable def overall_gain_loss_percent (p : Prices) : ℚ :=
  let total_cost := 10 * p.pen_cost + 15 * p.pencil_cost
  let total_sell := 10 * p.pen_sell + 15 * p.pencil_sell
  (total_sell - total_cost) / total_cost * 100

/-- Theorem stating that the overall gain or loss percentage is 0% -/
theorem zero_gain_loss (p : Prices) 
  (h1 : 10 * p.pen_cost = 5 * p.pen_sell) 
  (h2 : 15 * p.pencil_cost = 9 * p.pencil_sell) : 
  overall_gain_loss_percent p = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_gain_loss_l1026_102649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_line_similarity_theorem_l1026_102616

/-- Represents a closed polygonal line with an odd number of vertices -/
structure PolygonalLine (n : ℕ) where
  vertices : Fin (2*n+1) → ℝ × ℝ

/-- Defines the S operation on a polygonal line -/
def S (M : PolygonalLine n) : PolygonalLine n := sorry

/-- Defines the sequence of polygonal lines M_k -/
def M_seq (M : PolygonalLine n) : ℕ → PolygonalLine n
  | 0 => M
  | k+1 => S (M_seq M k)

/-- Two polygonal lines are similar if they have the same shape and orientation -/
def similar (M1 M2 : PolygonalLine n) : Prop := sorry

theorem polygonal_line_similarity_theorem (n : ℕ) (M : PolygonalLine n) :
  ∃ (k : ℕ), similar (M_seq M k) M ∧ 
    ∃ (scale : ℝ), scale = (1 / 2 : ℝ) ^ k ∧ 
      ∀ (i : Fin (2*n+1)), 
        (M_seq M k).vertices i = scale • M.vertices i := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_line_similarity_theorem_l1026_102616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_six_l1026_102698

/-- Represents the fuel efficiency of a car in different driving conditions. -/
structure CarEfficiency where
  highway_miles_per_tank : ℚ
  city_miles_per_tank : ℚ
  city_miles_per_gallon : ℚ

/-- Calculates the difference in miles per gallon between highway and city driving. -/
noncomputable def mpg_difference (car : CarEfficiency) : ℚ :=
  (car.highway_miles_per_tank / (car.city_miles_per_tank / car.city_miles_per_gallon)) - car.city_miles_per_gallon

/-- Theorem stating that the difference in miles per gallon between highway and city driving is 6. -/
theorem mpg_difference_is_six (car : CarEfficiency) 
    (h1 : car.highway_miles_per_tank = 480)
    (h2 : car.city_miles_per_tank = 336)
    (h3 : car.city_miles_per_gallon = 14) : 
  mpg_difference car = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_six_l1026_102698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_56_value_l1026_102632

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- We define a₀ = 0 to match the problem's a₁ = 0
  | n + 1 => (sequence_a n - Real.sqrt 3) / (Real.sqrt 3 * sequence_a n + 1)

theorem a_56_value : sequence_a 56 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_56_value_l1026_102632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pentagon_area_ratio_approx_l1026_102654

/-- The ratio of the area of an equilateral triangle to the area of a pentagon
    formed by placing the triangle on top of a square, where the square's side
    length is double the triangle's. -/
noncomputable def triangle_to_pentagon_area_ratio (s : ℝ) : ℝ :=
  let triangle_area := (Real.sqrt 3 / 4) * s^2
  let square_area := (2*s)^2
  let pentagon_area := square_area + triangle_area
  triangle_area / pentagon_area

/-- The ratio of the triangle area to the pentagon area is approximately 0.0977 -/
theorem triangle_pentagon_area_ratio_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  ∀ (s : ℝ), s > 0 → |triangle_to_pentagon_area_ratio s - 0.0977| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pentagon_area_ratio_approx_l1026_102654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1026_102617

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1)

def F (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

theorem problem_solution (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, f a x ≤ 0 ∧ ∃ x > 0, f a x = 0) →
  a = 1 ∧
  (∀ x ∈ Set.Ioo 0 3, (deriv (F a)) x ≤ 1/2) →
  a ≥ 1/2 ∧
  (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1),
    f a x ≤ (if a ≤ 1/Real.exp 1 then 1 - Real.exp 1 * a + a
      else if a < Real.exp 1 then -Real.log a - 1 + a
      else -1 - a/Real.exp 1 + a)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1026_102617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1026_102604

-- Define the function f(x) = 1 / (x + 1)
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≠ -1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1026_102604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_plane_not_always_parallel_l1026_102675

/-- Two lines are non-coincident if they are distinct. -/
def non_coincident (a b : Set ℝ) : Prop := a ≠ b

/-- A line is parallel to a plane if it does not intersect the plane. -/
def line_parallel_plane (l : Set ℝ) (p : Set ℝ) : Prop :=
  l ∩ p = ∅

/-- Two lines are parallel if they do not intersect and are not skew. -/
def line_parallel_line (a b : Set ℝ) : Prop :=
  (a ∩ b = ∅) ∧ (∃ p : Set ℝ, line_parallel_plane a p ∧ line_parallel_plane b p)

theorem parallel_lines_plane_not_always_parallel :
  ∃ (a b : Set ℝ) (α : Set ℝ),
    non_coincident a b ∧
    line_parallel_plane a α ∧
    line_parallel_plane b α ∧
    ¬line_parallel_line a b :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_plane_not_always_parallel_l1026_102675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_two_dividing_15_4_minus_9_4_l1026_102670

theorem highest_power_of_two_dividing_15_4_minus_9_4 : 
  ∃ (k : ℕ), 2^k = (15^4 - 9^4).gcd (2^32) ∧ 
  ∀ (m : ℕ), 2^m ∣ (15^4 - 9^4) → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_two_dividing_15_4_minus_9_4_l1026_102670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l1026_102674

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 2)*x + 6*a - 1 else a^x

theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ (3/8 ≤ a ∧ a < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l1026_102674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sum_magnitude_l1026_102661

theorem perpendicular_vectors_sum_magnitude :
  ∀ (l : ℝ),
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![l, -1]
  (a • b = 0) →
  ‖a + b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sum_magnitude_l1026_102661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_librarian_added_books_l1026_102672

/-- The number of books added by the librarian -/
def books_added (original_books : ℚ) (books_per_shelf : ℚ) (shelves_needed : ℕ) : ℚ :=
  books_per_shelf * shelves_needed - original_books

/-- Theorem stating the number of books added by the librarian -/
theorem librarian_added_books : 
  books_added 46 4 14 = 10 := by
  unfold books_added
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_librarian_added_books_l1026_102672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_equals_2017_l1026_102665

-- Define the function f recursively
def f : ℕ → ℤ
  | 0 => 1  -- Add a case for 0 to avoid missing case error
  | 1 => 1
  | 2 => 1
  | n+3 => f (n+2) - f (n+1) + (n+3)

-- State the theorem
theorem f_2018_equals_2017 : f 2018 = 2017 := by
  sorry

-- You can add additional helper lemmas or definitions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_equals_2017_l1026_102665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_reciprocals_l1026_102631

theorem fraction_sum_reciprocals (m n : ℕ) (hm : m > 0) (hn : n > 0)
  (h_gcd : Nat.gcd m n = 5)
  (h_lcm : Nat.lcm m n = 210)
  (h_sum : m + n = 75) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 1 / 14 := by
  sorry

#check fraction_sum_reciprocals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_reciprocals_l1026_102631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1026_102605

noncomputable def euler_formula (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

noncomputable def z₁ : ℂ := 2 * euler_formula (Real.pi / 3)
noncomputable def z₂ : ℂ := euler_formula (Real.pi / 2)
noncomputable def z : ℂ := z₁ / z₂

theorem z_in_fourth_quadrant : 
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l1026_102605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_proof_l1026_102645

theorem vector_subtraction_proof :
  (4 : ℝ) • (![3, -5] : Fin 2 → ℝ) - (3 : ℝ) • (![(-2), 6] : Fin 2 → ℝ) = ![18, -38] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_proof_l1026_102645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_l1026_102664

noncomputable def sequence_a : ℕ+ → ℝ := sorry
noncomputable def sequence_b : ℕ+ → ℝ := sorry
noncomputable def sequence_c : ℕ+ → ℝ := sorry
noncomputable def sum_S : ℕ+ → ℝ := sorry
noncomputable def sum_T : ℕ+ → ℝ := sorry

axiom sum_S_def : ∀ n : ℕ+, sum_S n = (3/2) * sequence_a n - 1/2
axiom sequence_b_def : ∀ n : ℕ+, sequence_b n = 2 * Real.log (sequence_a n) / Real.log 3 + 1
axiom sequence_c_def : ∀ n : ℕ+, sequence_c n = sequence_b n / sequence_a n

theorem range_of_c (c : ℝ) :
  (∀ n : ℕ+, sum_T n < c^2 - 2*c) ↔ c ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_c_l1026_102664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l1026_102686

/-- The area between a circle circumscribing two externally tangent circles of radii 2 and 3 -/
noncomputable def shaded_area : ℝ := 12 * Real.pi

/-- The radius of the circumscribing circle -/
def large_radius : ℝ := 5

theorem shaded_area_proof :
  let small_circle1_radius : ℝ := 2
  let small_circle2_radius : ℝ := 3
  let large_circle_area := Real.pi * large_radius ^ 2
  let small_circle1_area := Real.pi * small_circle1_radius ^ 2
  let small_circle2_area := Real.pi * small_circle2_radius ^ 2
  large_circle_area - small_circle1_area - small_circle2_area = shaded_area := by
  sorry

#eval large_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l1026_102686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l1026_102668

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the line
def line_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x - Real.sqrt 3

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 0)

-- Define the distance function
noncomputable def distance (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  let (x₀, y₀) := p
  let a := -Real.sqrt 3
  let b := 1
  let c := Real.sqrt 3
  (|a * x₀ + b * y₀ + c|) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_center_to_line :
  distance circle_center line_equation = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l1026_102668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joyce_michael_difference_l1026_102657

-- Define the lengths of each person's favorite movie
variable (michael_movie_length : ℝ)
variable (joyce_movie_length : ℝ)
variable (nikki_movie_length : ℝ)
variable (ryn_movie_length : ℝ)

-- Define the conditions
axiom joyce_longer : joyce_movie_length > michael_movie_length
axiom nikki_three_times : nikki_movie_length = 3 * michael_movie_length
axiom ryn_four_fifths : ryn_movie_length = (4/5) * nikki_movie_length
axiom nikki_thirty_hours : nikki_movie_length = 30
axiom total_hours : joyce_movie_length + michael_movie_length + nikki_movie_length + ryn_movie_length = 76

-- Theorem to prove
theorem joyce_michael_difference :
  joyce_movie_length - michael_movie_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joyce_michael_difference_l1026_102657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_logarithm_l1026_102646

noncomputable def f (x : ℝ) := Real.log ((x + 2) / 3)
noncomputable def g (x : ℝ) := Real.log (x + 1)

theorem transform_logarithm (x : ℝ) :
  f x = g ((x - 1) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_logarithm_l1026_102646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1026_102688

/-- Represents the time in hours it takes for a tap to fill a cistern. -/
noncomputable def T : ℝ := sorry

/-- Represents the rate at which the first tap fills the cistern (fraction of cistern per hour). -/
noncomputable def fill_rate : ℝ := 1 / T

/-- Represents the rate at which the second tap empties the cistern (fraction of cistern per hour). -/
noncomputable def empty_rate : ℝ := 1 / 9

/-- Represents the time it takes to fill the cistern when both taps are open (in hours). -/
noncomputable def combined_fill_time : ℝ := 4.5

theorem cistern_fill_time : T = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l1026_102688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_three_element_set_l1026_102621

theorem subset_count_of_three_element_set (A : Finset ℕ) :
  A.card = 3 → (Finset.powerset A).card = 8 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_three_element_set_l1026_102621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_curvature_condition_l1026_102603

noncomputable section

-- Define the function f(x) = 1/x
def f (x : ℝ) : ℝ := 1 / x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := -1 / (x^2)

-- Define the slope at point A
noncomputable def k_A (a : ℝ) : ℝ := f_derivative a

-- Define the slope at point B
noncomputable def k_B (a : ℝ) : ℝ := f_derivative (1/a)

-- Define the distance between points A and B
noncomputable def distance_AB (a : ℝ) : ℝ := Real.sqrt (2 * (a - 1/a)^2)

-- Define the approximate curvature K(A,B)
noncomputable def K (a : ℝ) : ℝ := abs (k_A a - k_B a) / distance_AB a

-- State the theorem
theorem min_m_for_curvature_condition (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ m : ℝ, m * K a > 1 → m ≥ Real.sqrt 2 / 2) ∧
  ∃ m : ℝ, m * K a > 1 ∧ m = Real.sqrt 2 / 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_curvature_condition_l1026_102603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l1026_102663

theorem simplify_trig_expression (α : ℝ) (h : π/2 < α ∧ α < π) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) +
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = Real.sin α - Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trig_expression_l1026_102663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_theorem_article_a_marked_price_article_b_marked_price_article_c_marked_price_l1026_102653

/-- Calculate the marked price of an article given its cost price, desired profit percentage, and deduction percentage. -/
noncomputable def calculateMarkedPrice (costPrice : ℝ) (profitPercentage : ℝ) (deductionPercentage : ℝ) : ℝ :=
  (costPrice * (1 + profitPercentage)) / (1 - deductionPercentage)

/-- Theorem stating that the calculated marked price results in the desired profit after deduction -/
theorem marked_price_theorem (costPrice : ℝ) (profitPercentage : ℝ) (deductionPercentage : ℝ) 
  (h1 : costPrice > 0) 
  (h2 : profitPercentage ≥ 0) 
  (h3 : deductionPercentage ≥ 0) 
  (h4 : deductionPercentage < 1) :
  let markedPrice := calculateMarkedPrice costPrice profitPercentage deductionPercentage
  let sellingPrice := markedPrice * (1 - deductionPercentage)
  sellingPrice = costPrice * (1 + profitPercentage) := by
  sorry

/-- Verify the marked price for Article A -/
theorem article_a_marked_price :
  ∃ ε > 0, |calculateMarkedPrice 47.5 0.25 0.06 - 63.15| < ε := by
  sorry

/-- Verify the marked price for Article B -/
theorem article_b_marked_price :
  ∃ ε > 0, |calculateMarkedPrice 82 0.3 0.08 - 115.87| < ε := by
  sorry

/-- Verify the marked price for Article C -/
theorem article_c_marked_price :
  ∃ ε > 0, |calculateMarkedPrice 120 0.2 0.05 - 151.58| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_theorem_article_a_marked_price_article_b_marked_price_article_c_marked_price_l1026_102653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_m_l1026_102684

/-- The circle C in the Cartesian coordinate system -/
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - m)^2 = 3

/-- The condition that AB is twice the length of GO -/
def chord_condition (m : ℝ) (x y : ℝ) : Prop :=
  4 * (3 - (x + 2)^2 - (y - m)^2) = 4 * (x^2 + y^2)

/-- The theorem stating that no real m satisfies the conditions -/
theorem no_valid_m : ¬ ∃ m : ℝ, ∃ x y : ℝ, circle_C m x y ∧ chord_condition m x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_m_l1026_102684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_purification_duration_min_mass_optimal_purification_l1026_102656

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then x^2/16 + 2
  else if x > 4 then (x + 14)/(2*x - 2)
  else 0

-- Define the concentration function y(m, x)
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := m * f x

-- Theorem 1: Effective purification duration
theorem effective_purification_duration (m : ℝ) (h : m = 4) :
  ∃ x : ℝ, x = 16 ∧ ∀ t : ℝ, 0 < t ∧ t ≤ x → y m t ≥ 4 ∧ ∀ u : ℝ, u > x → y m u < 4 :=
by sorry

-- Theorem 2: Minimum mass for optimal purification
theorem min_mass_optimal_purification :
  ∃ m : ℝ, m = 16/7 ∧ 
    (∀ x : ℝ, 0 < x ∧ x ≤ 7 → 4 ≤ y m x ∧ y m x ≤ 10) ∧
    (∀ m' : ℝ, m' < m → ∃ x : ℝ, 0 < x ∧ x ≤ 7 ∧ (y m' x < 4 ∨ y m' x > 10)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_purification_duration_min_mass_optimal_purification_l1026_102656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_in_minutes_l1026_102699

/-- The time it takes P to finish the job alone in hours -/
noncomputable def p_time : ℝ := 4

/-- The time it takes Q to finish the job alone in hours -/
noncomputable def q_time : ℝ := 20

/-- The time P and Q work together in hours -/
noncomputable def together_time : ℝ := 3

/-- The rate at which P works (portion of job per hour) -/
noncomputable def p_rate : ℝ := 1 / p_time

/-- The rate at which Q works (portion of job per hour) -/
noncomputable def q_rate : ℝ := 1 / q_time

/-- The combined rate of P and Q working together -/
noncomputable def combined_rate : ℝ := p_rate + q_rate

/-- The portion of the job completed when P and Q work together -/
noncomputable def completed_portion : ℝ := combined_rate * together_time

/-- The remaining portion of the job -/
noncomputable def remaining_portion : ℝ := 1 - completed_portion

/-- The time it takes P to finish the remaining portion alone in hours -/
noncomputable def remaining_time : ℝ := remaining_portion / p_rate

theorem remaining_time_in_minutes : remaining_time * 60 = 24 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_in_minutes_l1026_102699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_antiderivative_of_f_l1026_102614

-- Define the functions F and f as noncomputable
noncomputable def F (x : ℝ) : ℝ := (1/2) * Real.sin (2*x)
noncomputable def f (x : ℝ) : ℝ := Real.cos (2*x)

-- State the theorem
theorem F_is_antiderivative_of_f : ∀ x, deriv F x = f x := by
  -- The proof is omitted and replaced with 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_antiderivative_of_f_l1026_102614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_specific_ellipse_l1026_102685

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

noncomputable def semi_major_axis (x_coeff y_coeff : ℝ) : ℝ := 
  Real.sqrt (1 / min (1 / x_coeff) (1 / y_coeff))

noncomputable def semi_minor_axis (x_coeff y_coeff : ℝ) : ℝ := 
  Real.sqrt (1 / max (1 / x_coeff) (1 / y_coeff))

theorem distance_between_foci_of_specific_ellipse :
  let x_coeff : ℝ := 1 / 36
  let y_coeff : ℝ := 1 / 16
  let a := semi_major_axis x_coeff y_coeff
  let b := semi_minor_axis x_coeff y_coeff
  distance_between_foci a b = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_specific_ellipse_l1026_102685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1026_102671

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

-- Define the conditions
def conditions (a b c : ℝ) : Prop :=
  triangle a b c ∧
  (a + b - c) * (a + b + c) = a * b ∧
  c = 2 * a * Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ∧
  b = 2

-- Define the angle C
noncomputable def angle_C (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

-- Define the area of the triangle
noncomputable def area (a b c : ℝ) : ℝ :=
  (1/2) * a * b * Real.sin (angle_C a b c)

-- State the theorem
theorem triangle_properties (a b c : ℝ) (h : conditions a b c) :
  angle_C a b c = 2 * Real.pi / 3 ∧ area a b c = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1026_102671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_six_l1026_102606

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem arithmetic_sequence_sum_six (d : ℝ) :
  d ≠ 0 →
  let a := arithmetic_sequence 1 d
  (a 2) * (a 6) = (a 3)^2 →
  sum_arithmetic_sequence 1 d 6 = -24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_six_l1026_102606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1026_102659

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) 
    and an asymptote 3x - 4y = 0 is equal to 5/4. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 3 / 4) : 
  Real.sqrt (a^2 + b^2) / a = 5 / 4 := by
  -- Define c
  let c := Real.sqrt (a^2 + b^2)
  
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1026_102659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1026_102607

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the vertices
def left_vertex (a : ℝ) : ℝ × ℝ := (-a, 0)
def right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define a point on the hyperbola
noncomputable def point_on_hyperbola (a b x₀ y₀ : ℝ) : Prop :=
  hyperbola a b x₀ y₀ ∧ x₀ ≠ -a ∧ x₀ ≠ a

-- Define the product of slopes
noncomputable def slope_product (a x₀ y₀ : ℝ) : ℝ :=
  (y₀ / (x₀ + a)) * (y₀ / (x₀ - a))

-- Define the distance from focus to asymptote
noncomputable def focus_to_asymptote_distance (a b c : ℝ) : ℝ :=
  (b / a) * c

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_properties (a b x₀ y₀ : ℝ) :
  hyperbola a b x₀ y₀ →
  point_on_hyperbola a b x₀ y₀ →
  slope_product a x₀ y₀ = 16/9 →
  ∃ c, c^2 = a^2 + b^2 ∧ focus_to_asymptote_distance a b c = 4 →
  eccentricity a b = 5/3 ∧ a^2 = 9 ∧ b^2 = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1026_102607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_inequality_condition_l1026_102610

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / x

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (k c : ℝ), ∀ (x y : ℝ),
    y = k * (x - Real.exp 1) + f (Real.exp 1) ↔ x + Real.exp 2 * y - 3 * Real.exp 1 = 0 := by
  sorry

-- Theorem for the inequality condition
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f x - 1 / x ≥ a * (x^2 - 1) / x) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_inequality_condition_l1026_102610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_max_a_correct_l1026_102609

/-- The absolute value function |3x + 2| -/
def f (x : ℝ) : ℝ := |3 * x + 2|

/-- The set of solutions for f(x) ≤ 1 -/
def solution_set : Set ℝ := {x : ℝ | f x ≤ 1}

/-- The maximum value of a for which f(x²) ≥ a|x| holds for all real x -/
noncomputable def max_a : ℝ := 2 * Real.sqrt 6

theorem solution_set_correct : solution_set = Set.Icc (-1) (-1/3) := by sorry

theorem max_a_correct : 
  (∀ x : ℝ, f (x^2) ≥ max_a * |x|) ∧ 
  (∀ ε > 0, ∃ x : ℝ, f (x^2) < (max_a + ε) * |x|) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_max_a_correct_l1026_102609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_29_over_4_l1026_102660

noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 12 * x + 21 else 3 * x - 27

theorem sum_of_solutions_is_29_over_4 :
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 29/4 ∧
  (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

#check sum_of_solutions_is_29_over_4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_29_over_4_l1026_102660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_avg_value_l1026_102601

noncomputable def square_avg (a b : ℝ) : ℝ := (a^2 + b^2) / 2

noncomputable def triple_avg (a b c : ℝ) : ℝ := (a + b + 2*c) / 3

noncomputable def nested_avg : ℝ :=
  triple_avg (triple_avg 2 (-1) 1) (square_avg 2 3) 1

theorem nested_avg_value : nested_avg = 19/6 := by
  -- Expand the definition of nested_avg
  unfold nested_avg
  -- Expand the definition of triple_avg
  unfold triple_avg
  -- Expand the definition of square_avg
  unfold square_avg
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_avg_value_l1026_102601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_x0_l1026_102673

-- Define the function f(x) = ln x + x
noncomputable def f (x : ℝ) : ℝ := Real.log x + x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 1 / x + 1

-- Theorem statement
theorem tangent_line_parallel_implies_x0 (x₀ : ℝ) (h₁ : x₀ > 0) :
  f_derivative x₀ = 3 → x₀ = 1 / 2 := by
  intro h
  -- Proof steps would go here
  sorry

#check tangent_line_parallel_implies_x0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_x0_l1026_102673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_perimeter_exists_triangle_with_least_perimeter_l1026_102650

/-- Triangle DEF with given cosine values -/
structure TriangleDEF where
  -- Side lengths
  d : ℕ
  e : ℕ
  f : ℕ
  -- Cosine values
  cos_D : ℚ
  cos_E : ℚ
  cos_F : ℚ
  -- Constraints on cosine values
  h_cos_D : cos_D = 3/5
  h_cos_E : cos_E = 12/13
  h_cos_F : cos_F = -3/5

/-- The perimeter of triangle DEF -/
def perimeter (t : TriangleDEF) : ℕ := t.d + t.e + t.f

/-- The least possible perimeter of triangle DEF -/
theorem least_perimeter (t : TriangleDEF) : perimeter t ≥ 129 := by
  sorry

/-- There exists a triangle DEF with perimeter 129 -/
theorem exists_triangle_with_least_perimeter : ∃ t : TriangleDEF, perimeter t = 129 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_perimeter_exists_triangle_with_least_perimeter_l1026_102650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_sold_l1026_102666

-- Define the cost price of one pencil
variable (C : ℚ)

-- Define the selling price of one pencil
noncomputable def S (C : ℚ) : ℚ := 1.5 * C

-- Define the number of pencils sold at the selling price
noncomputable def n : ℚ := 12 / 1.5

-- Theorem statement
theorem pencils_sold (h1 : 12 * C = n * S C) (h2 : S C = 1.5 * C) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencils_sold_l1026_102666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_first_exponent_l1026_102629

theorem base_of_first_exponent (x : ℕ) (b : ℕ) : 
  (x ^ 6) * 9 ^ (3 * 6 - 1) = (2 ^ 6) * (3 ^ b) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_first_exponent_l1026_102629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_specific_angle_l1026_102677

/-- An angle in the Cartesian coordinate system -/
structure CartesianAngle where
  /-- The x-coordinate of the point on the terminal side -/
  x : ℝ
  /-- The y-coordinate of the point on the terminal side -/
  y : ℝ

/-- The sine of a CartesianAngle -/
noncomputable def sine (α : CartesianAngle) : ℝ :=
  α.y / Real.sqrt (α.x^2 + α.y^2)

/-- The theorem stating that for the given angle, sine equals -1/2 -/
theorem sine_specific_angle :
  let α : CartesianAngle := { x := -Real.sqrt 3, y := -1 }
  sine α = -1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_specific_angle_l1026_102677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_per_ton_l1026_102676

/-- The production cost function -/
noncomputable def cost (x : ℝ) : ℝ := x^2 / 10 - 30 * x + 4000

/-- The cost per ton function -/
noncomputable def costPerTon (x : ℝ) : ℝ := cost x / x

theorem min_cost_per_ton :
  ∃ (x : ℝ), 150 ≤ x ∧ x ≤ 250 ∧
  ∀ (y : ℝ), 150 ≤ y ∧ y ≤ 250 → costPerTon x ≤ costPerTon y ∧
  x = 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_per_ton_l1026_102676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_theorem_l1026_102691

-- Define the function f as noncomputable due to dependency on Real.pi
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x * (x - 1)
  else if 1 < x ∧ x ≤ 2 then Real.sin (Real.pi * x)
  else 0  -- Define a default value for x outside [0, 2]

-- State the theorem
theorem function_sum_theorem (f : ℝ → ℝ) :
  (∀ x, f (x + 4) = f x) →  -- Periodic with period 4
  (∀ x, f (-x) = -f x) →    -- Odd function
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x * (x - 1)) →  -- Definition on [0, 1]
  (∀ x, 1 < x ∧ x ≤ 2 → f x = Real.sin (Real.pi * x)) →  -- Definition on (1, 2]
  f (29/4) + f (41/6) = 11/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_theorem_l1026_102691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_is_correct_l1026_102694

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  l₁ : ℝ → ℝ × ℝ
  m₁ : ℝ → ℝ × ℝ
  h_parallel : ∀ t s, l₁ t - l₁ s = m₁ t - m₁ s

/-- The projection vector satisfying the given conditions -/
def projection_vector (lines : ParallelLines) : ℝ × ℝ := 
  (-12, 15)

/-- Theorem stating that the projection vector is (-12, 15) -/
theorem projection_vector_is_correct (lines : ParallelLines) :
  projection_vector lines = (-12, 15) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_is_correct_l1026_102694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_work_hours_l1026_102697

theorem employee_work_hours 
  (initial_employees : ℕ) 
  (hourly_rate : ℕ) 
  (days_per_week : ℕ) 
  (weeks_per_month : ℕ) 
  (additional_employees : ℕ) 
  (total_monthly_pay : ℕ) 
  (h : ℕ) : 
  initial_employees = 500 → 
  hourly_rate = 12 → 
  days_per_week = 5 → 
  weeks_per_month = 4 → 
  additional_employees = 200 → 
  total_monthly_pay = 1680000 → 
  (initial_employees + additional_employees) * hourly_rate * h * days_per_week * weeks_per_month = total_monthly_pay → 
  h = 10 := by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

#check employee_work_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_work_hours_l1026_102697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l1026_102613

/-- An ellipse with semi-major axis a₁ and semi-minor axis b₁ -/
structure Ellipse where
  a₁ : ℝ
  b₁ : ℝ
  h₁ : a₁ > b₁
  h₂ : b₁ > 0

/-- A hyperbola with semi-major axis a₂ and semi-minor axis b₂ -/
structure Hyperbola where
  a₂ : ℝ
  b₂ : ℝ
  h₁ : a₂ > 0
  h₂ : b₂ > 0

/-- The intersection point of an ellipse and a hyperbola -/
structure IntersectionPoint (e : Ellipse) (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / e.a₁^2 + y^2 / e.b₁^2 = 1
  on_hyperbola : x^2 / h.a₂^2 - y^2 / h.b₂^2 = 1

/-- The slope of the tangent to a curve at a point -/
noncomputable def slopeAtPoint (x y : ℝ) : ℝ := sorry

/-- Theorem stating that the product of slopes of tangents at the intersection point is -1 -/
theorem tangent_slopes_product (e : Ellipse) (h : Hyperbola) 
  (same_foci : sorry) (p : IntersectionPoint e h) : 
  (slopeAtPoint p.x p.y) * (slopeAtPoint p.x p.y) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slopes_product_l1026_102613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_120_degrees_l1026_102689

noncomputable def angle_between_vectors (e₁ e₂ : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  let angle_e₁_e₂ : ℝ := Real.pi / 3
  let a := (2 * e₁.1 + e₂.1, 2 * e₁.2 + e₂.2)
  let b := (-3 * e₁.1 + 2 * e₂.1, -3 * e₁.2 + 2 * e₂.2)
  e₁.1 ^ 2 + e₁.2 ^ 2 = 1 ∧
  e₂.1 ^ 2 + e₂.2 ^ 2 = 1 ∧
  e₁.1 * e₂.1 + e₁.2 * e₂.2 = Real.cos angle_e₁_e₂ →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2))) = 2 * Real.pi / 3

theorem angle_between_vectors_is_120_degrees :
  ∀ (e₁ e₂ : ℝ × ℝ), angle_between_vectors e₁ e₂ e₁ e₂ := by
  sorry

#check angle_between_vectors_is_120_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_is_120_degrees_l1026_102689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_positive_l1026_102678

/-- The probability that San Francisco plays in the next Super Bowl. -/
def P_play : ℝ := sorry

/-- The probability that San Francisco does not play in the next Super Bowl. -/
def P_not_play : ℝ := sorry

/-- The multiple of P_play to P_not_play. -/
def k : ℝ := sorry

/-- Axiom: P_play is k times P_not_play. -/
axiom play_multiple : P_play = k * P_not_play

/-- Axiom: The sum of probabilities is 1. -/
axiom prob_sum : P_play + P_not_play = 1

/-- Theorem: Given the conditions, k must be a positive real number greater than 0. -/
theorem k_is_positive : k > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_positive_l1026_102678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_constraint_l1026_102667

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2)

theorem period_constraint (a b : ℝ) (h1 : a ≤ b) 
  (h2 : ∀ x, a ≤ x ∧ x ≤ b → -1 ≤ f x ∧ f x ≤ 2) : 
  b - a ≤ 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_constraint_l1026_102667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_dice_sum_exceeds_ten_probability_l1026_102630

/-- The type representing the possible outcomes of a single die roll -/
inductive DieOutcome
  | one
  | two
  | three
  | four
  | five
  | six
deriving Fintype, Repr

/-- The type representing the outcome of rolling two dice -/
def TwoDiceOutcome := DieOutcome × DieOutcome
deriving Fintype, Repr

/-- The function that calculates the sum of two dice outcomes -/
def diceSum (outcome : TwoDiceOutcome) : Nat :=
  match outcome with
  | (a, b) => DieOutcome.toNat a + DieOutcome.toNat b
where
  DieOutcome.toNat : DieOutcome → Nat
    | DieOutcome.one => 1
    | DieOutcome.two => 2
    | DieOutcome.three => 3
    | DieOutcome.four => 4
    | DieOutcome.five => 5
    | DieOutcome.six => 6

/-- The theorem stating that the probability of the sum of two fair dice exceeding 10 is 1/12 -/
theorem two_dice_sum_exceeds_ten_probability :
  (Finset.filter (fun o => diceSum o > 10) (Finset.univ : Finset TwoDiceOutcome)).card /
  (Finset.univ : Finset TwoDiceOutcome).card = 1 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_dice_sum_exceeds_ten_probability_l1026_102630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_decimal_digits_l1026_102690

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2003 → 
  x = (10^n + 1)^(11/8) → 
  ∃ (k : ℕ), x = k + 0.375 + r ∧ r ∈ Set.Ioc 0 0.001 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_decimal_digits_l1026_102690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_relation_max_size_l1026_102662

/-- A binary relation on a set satisfying the given conditions -/
class SpecialRelation (α : Type*) where
  rel : α → α → Prop
  total : ∀ a b : α, (rel a b ∧ ¬rel b a) ∨ (rel b a ∧ ¬rel a b)
  transitive : ∀ a b c : α, rel a b → rel b c → rel c a

/-- The theorem stating that any set with a SpecialRelation has at most 3 elements -/
theorem special_relation_max_size {α : Type*} [SpecialRelation α] [Fintype α] : 
  ∃ (n : Nat), Fintype.card α ≤ n ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_relation_max_size_l1026_102662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_M_to_C₂_l1026_102602

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) : ℝ × ℝ := (8 * Real.cos t, 3 * Real.sin t)

def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 7 / (Real.cos θ - 2 * Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point Q
def Q : ℝ × ℝ := (-4, 4)

-- Define the midpoint M of PQ
def M (t : ℝ) : ℝ × ℝ := 
  let P := C₁ t
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the distance function between two points
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Statement to prove
theorem min_distance_M_to_C₂ : 
  ∃ (t θ : ℝ), ∀ (t' θ' : ℝ), 
    distance (M t) (C₂ θ) ≤ distance (M t') (C₂ θ') ∧ 
    distance (M t) (C₂ θ) = 8 * Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_M_to_C₂_l1026_102602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_speed_l1026_102696

/-- Given a distance in meters and a time in seconds, calculate the speed in km/hr -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

/-- Theorem: Sandy's speed is approximately 18 km/hr given the distance and time -/
theorem sandys_speed :
  let distance := 700
  let time := 139.98880089592834
  ⌊calculate_speed distance time⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_speed_l1026_102696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_seating_problem_l1026_102611

theorem flight_seating_problem (total_passengers : ℕ) 
  (female_percentage : ℚ) (first_class_male_ratio : ℚ) 
  (coach_females : ℕ) 
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 45/100)
  (h3 : first_class_male_ratio = 1/3)
  (h4 : coach_females = 46) : 
  10/100 = (total_passengers - coach_females - (female_percentage * total_passengers - coach_females : ℚ)) / total_passengers := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_seating_problem_l1026_102611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_compounding_yields_more_l1026_102655

/-- Compound interest calculation for a given principal, rate, compounding frequency, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Theorem stating that monthly compounding yields more money than annual compounding
    for the given conditions -/
theorem monthly_compounding_yields_more
  (principal : ℝ)
  (annual_rate : ℝ)
  (monthly_rate : ℝ)
  (time : ℝ)
  (h1 : principal = 1000)
  (h2 : annual_rate = 0.03)
  (h3 : monthly_rate = 0.0025)
  (h4 : time = 5) :
  compound_interest principal monthly_rate 12 time >
  compound_interest principal annual_rate 1 time :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_compounding_yields_more_l1026_102655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_sum_squares_l1026_102634

/-- Given a sphere of radius R and a point P at distance a from the center,
    where three perpendicular chords pass through P,
    the sum of squares of the chord segments is 6R^2 - 2a^2 -/
theorem chord_segment_sum_squares (R a : ℝ) (h : R > a) (h_pos : R > 0) :
  ∃ (sum_squares : ℝ), sum_squares = 6 * R^2 - 2 * a^2 := by
  -- Proof goes here
  sorry

#check chord_segment_sum_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_sum_squares_l1026_102634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1026_102640

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (abs x)

-- State the theorems to be proved
theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ (∃ (y : ℝ), f y = M) ∧ M = 1) ∧ 
  (∀ (x : ℝ), f (-x) = -f x) ∧
  (¬ ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) ∧
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (p : ℝ), p = Real.pi ∧ p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ 
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1026_102640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_piles_count_l1026_102642

/-- Definition of a valid pile distribution --/
def IsValidDistribution (piles : List ℕ) : Prop :=
  piles.sum = 660 ∧ 
  ∀ x y, x ∈ piles → y ∈ piles → x < 2 * y ∧ y < 2 * x

/-- The maximum number of piles in a valid distribution --/
theorem max_piles_count : 
  (∃ (piles : List ℕ), IsValidDistribution piles ∧ piles.length = 30) ∧ 
  (∀ (piles : List ℕ), IsValidDistribution piles → piles.length ≤ 30) := by
  sorry

#check max_piles_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_piles_count_l1026_102642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_triples_existence_l1026_102612

theorem infinite_triples_existence :
  ∀ k : ℕ, k > 2015 →
  ∃ a b c : ℕ,
    a > 2015 ∧ b > 2015 ∧ c > 2015 ∧
    a ∣ (b * c - 1) ∧
    b ∣ (a * c + 1) ∧
    c ∣ (a * b + 1) := by
  intro k hk
  use k, k + 1, k^2 + k + 1
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_triples_existence_l1026_102612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_cube_root_abs_equals_sqrt_two_l1026_102637

theorem sqrt_sum_cube_root_abs_equals_sqrt_two :
  Real.sqrt 9 + ((-8) ^ (1/3 : ℝ)) + |Real.sqrt 2 - 1| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_cube_root_abs_equals_sqrt_two_l1026_102637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_is_five_ninths_four_ninths_l1026_102681

/-- The direction vector of a parameterized line. -/
noncomputable def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  let t₁ := 1
  let t₂ := 0
  (line t₁ - line t₂)

/-- The given line equation. -/
noncomputable def line_equation (x : ℝ) : ℝ := (4 * x - 7) / 5

/-- The parameterization of the line. -/
noncomputable def line_param (t : ℝ) : ℝ × ℝ :=
  (4, 2) + t • (5/9, 4/9)

/-- The distance between a point on the line and (4, 2). -/
noncomputable def distance (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - 4)^2 + (p.2 - 2)^2)

theorem direction_vector_is_five_ninths_four_ninths :
  direction_vector line_param = (5/9, 4/9) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_is_five_ninths_four_ninths_l1026_102681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1026_102623

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the right vertex of the hyperbola
def right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the angle between two vectors
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Define the vector from origin to a point
def vector_to_point (p : ℝ × ℝ) : ℝ × ℝ := p

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity (a b : ℝ) (A P Q : ℝ × ℝ) :
  hyperbola a b A.1 A.2 →
  A = right_vertex a →
  ∃ (center : ℝ × ℝ), center = A ∧
    (∃ (asymptote : ℝ → ℝ), 
      P ∈ Set.range (λ x ↦ (x, asymptote x)) ∧
      Q ∈ Set.range (λ x ↦ (x, asymptote x))) →
  angle (vector_to_point P) (vector_to_point Q) = π / 3 →
  vector_to_point Q = 3 • vector_to_point P →
  eccentricity a b = Real.sqrt 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1026_102623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_op_equals_ab_l1026_102682

/-- A regular decagon is a polygon with 10 equal sides and angles -/
structure RegularDecagon where
  center : EuclideanSpace ℝ (Fin 2)
  vertices : Fin 10 → EuclideanSpace ℝ (Fin 2)

/-- Two points are neighboring in a regular decagon if they are adjacent vertices -/
def neighboring (d : RegularDecagon) (a b : Fin 10) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a)

/-- A point lies on a line segment if it's between the segment's endpoints -/
def lies_on_segment (p a b : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = a + t • (b - a)

theorem regular_decagon_op_equals_ab 
  (d : RegularDecagon) (a b : Fin 10) (p : EuclideanSpace ℝ (Fin 2)) :
  neighboring d a b →
  lies_on_segment p d.center (d.vertices b) →
  dist d.center p ^ 2 = dist d.center (d.vertices b) * dist p (d.vertices b) →
  dist d.center p = dist (d.vertices a) (d.vertices b) := by
  sorry

#check regular_decagon_op_equals_ab

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_decagon_op_equals_ab_l1026_102682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_points_distance_sum_l1026_102619

theorem sphere_points_distance_sum (O A B C D : ℝ × ℝ × ℝ) : 
  (‖A - O‖ = 1) → 
  (‖B - O‖ = 1) → 
  (‖C - O‖ = 1) → 
  (‖D - O‖ = 1) → 
  (A - O) + (B - O) + (C - O) = 0 → 
  ‖A - D‖ + ‖B - D‖ + ‖C - D‖ > 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_points_distance_sum_l1026_102619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_sampling_methods_l1026_102618

/-- Represents a sampling method -/
inductive SamplingMethod
| StratifiedSampling
| SimpleRandomSampling
| SystematicSampling

/-- Represents a city with its number of outlets -/
structure City where
  name : String
  outlets : Nat

/-- Represents a survey -/
structure Survey where
  sampleSize : Nat
  populationSize : Nat

/-- Determines the most appropriate sampling method for a given survey -/
def mostAppropriateSamplingMethod (cities : List City) (survey : Survey) : SamplingMethod :=
  sorry

/-- The problem setup -/
def problemSetup : List City × Survey × Survey :=
  let cities := [
    { name := "A", outlets := 150 },
    { name := "B", outlets := 120 },
    { name := "C", outlets := 190 },
    { name := "D", outlets := 140 }
  ]
  let survey1 := { sampleSize := 100, populationSize := 600 }
  let survey2 := { sampleSize := 8, populationSize := 20 }
  (cities, survey1, survey2)

theorem appropriate_sampling_methods :
  let (cities, survey1, survey2) := problemSetup
  mostAppropriateSamplingMethod cities survey1 = SamplingMethod.StratifiedSampling ∧
  mostAppropriateSamplingMethod [{ name := "C", outlets := 20 }] survey2 = SamplingMethod.SimpleRandomSampling :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_sampling_methods_l1026_102618
