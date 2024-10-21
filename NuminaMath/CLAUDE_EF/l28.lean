import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_half_ln_five_l28_2855

/-- The integrand function -/
noncomputable def f (x : ℝ) : ℝ :=
  (4 * Real.sqrt (2 - x) - Real.sqrt (x + 2)) /
  ((Real.sqrt (x + 2) + 4 * Real.sqrt (2 - x)) * (x + 2)^2)

/-- The theorem statement -/
theorem integral_equals_half_ln_five :
  ∫ x in (0)..(2), f x = (1/2) * Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_half_ln_five_l28_2855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_distance_theorem_l28_2807

/-- Represents a bicycle with two wheels of different sizes -/
structure Bicycle where
  back_wheel_perimeter : ℝ
  front_wheel_perimeter : ℝ

/-- The distance traveled by the bicycle -/
noncomputable def distance_traveled (b : Bicycle) (extra_revolutions : ℝ) : ℝ :=
  (extra_revolutions * b.back_wheel_perimeter * b.front_wheel_perimeter) /
  (b.front_wheel_perimeter - b.back_wheel_perimeter)

theorem bicycle_distance_theorem (b : Bicycle) 
  (h1 : b.back_wheel_perimeter = 9)
  (h2 : b.front_wheel_perimeter = 7)
  (h3 : extra_revolutions = 10) :
  distance_traveled b extra_revolutions = 315 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_distance_theorem_l28_2807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_divisibility_by_13_l28_2865

theorem five_digit_divisibility_by_13 : 
  ∃! (count : ℕ), count = 6400 ∧ 
  count = (Finset.filter 
    (fun n : ℕ ↦ 10000 ≤ n ∧ n ≤ 99999 ∧ 
      (n / 100 + n % 100) % 13 = 0)
    (Finset.range 100000)).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_divisibility_by_13_l28_2865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completes_in_20_days_l28_2874

/-- The number of days it takes A to complete the work -/
noncomputable def a_days : ℝ := 15

/-- The number of days A and B work together -/
noncomputable def work_together_days : ℝ := 4

/-- The fraction of work remaining after A and B work together -/
noncomputable def work_remaining : ℝ := 8 / 15

/-- The number of days it takes B to complete the work alone -/
noncomputable def b_days : ℝ := 20

/-- Theorem stating that B takes 20 days to complete the work alone -/
theorem b_completes_in_20_days :
  (work_together_days * (1 / a_days + 1 / b_days) = 1 - work_remaining) →
  b_days = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completes_in_20_days_l28_2874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l28_2852

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a * q^(n - 1)

theorem geometric_sequence_common_ratio 
  (a : ℝ) (q : ℝ) (h_pos : a > 0) 
  (h_increasing : ∀ n : ℕ, geometric_sequence a q n < geometric_sequence a q (n + 1))
  (h_equation : 2 * (geometric_sequence a q 4 + geometric_sequence a q 6) = 5 * geometric_sequence a q 5) :
  q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l28_2852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_is_open_one_to_one_point_five_l28_2872

open Real

/-- The function g as defined in the problem -/
noncomputable def g (a b c : ℝ) : ℝ := a / (a + b + 1) + b / (b + c + 1) + c / (c + a + 1)

/-- The theorem stating the range of g for positive real inputs -/
theorem g_range_is_open_one_to_one_point_five :
  ∀ (S : Set ℝ), (∀ (x : ℝ), x ∈ S ↔ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ g a b c = x) →
  S = Set.Ioo 1 (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_is_open_one_to_one_point_five_l28_2872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_ring_volume_increase_l28_2818

/-- The volume of a circular ring (torus) with major radius R and minor radius r -/
noncomputable def torusVolume (R r : ℝ) : ℝ := 2 * Real.pi^2 * R * r^2

/-- The percentage increase in a quantity -/
noncomputable def percentageIncrease (initial final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem circular_ring_volume_increase
  (r t : ℝ) -- Initial radius and thickness
  (hr : r > 0)
  (ht : t > 0) :
  percentageIncrease (torusVolume r (t/2)) (torusVolume (1.01 * r) (t/2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_ring_volume_increase_l28_2818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l28_2870

/-- A hyperbola with a vertex at (3, 0) and focal length 10 has the standard equation x²/9 - y²/16 = 1 -/
theorem hyperbola_equation : 
  let vertex : ℝ × ℝ := (3, 0)
  let focal_length : ℝ := 10
  ∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l28_2870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_neg_one_l28_2876

theorem arcsin_neg_one :
  Real.arcsin (-1) = -π/2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_neg_one_l28_2876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_coordinates_l28_2811

/-- The quadratic function f(x) = 2x^2 - 4x + 5 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

/-- The vertex of a quadratic function ax^2 + bx + c is at x = -b/(2a) -/
noncomputable def vertex_x (a b c : ℝ) : ℝ := -b / (2 * a)

/-- The y-coordinate of the vertex is obtained by evaluating f at vertex_x -/
noncomputable def vertex_y (a b c : ℝ) : ℝ := f (vertex_x a b c)

/-- The vertex coordinates of f(x) = 2x^2 - 4x + 5 are (1, 3) -/
theorem vertex_coordinates :
  (vertex_x 2 (-4) 5, vertex_y 2 (-4) 5) = (1, 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_coordinates_l28_2811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_nonprime_less_than_50_l28_2887

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem largest_consecutive_nonprime_less_than_50 :
  ∃ (a : ℕ),
    a + 6 = 32 ∧
    (∀ i : ℕ, i < 7 → 
      let n := a + i
      10 ≤ n ∧ n < 50 ∧ ¬(is_prime n)) ∧
    (∀ b : ℕ,
      b > a →
      ¬(∀ i : ℕ, i < 7 →
        let n := b + i
        10 ≤ n ∧ n < 50 ∧ ¬(is_prime n))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_nonprime_less_than_50_l28_2887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_terminates_l28_2893

/-- A procedure that transforms a multiset of integers by replacing two equal elements
    with their decreased and increased versions. -/
def transform_multiset (p q : ℕ+) (s : Multiset ℤ) : Multiset ℤ :=
  sorry

/-- The main theorem stating that the transformation procedure terminates
    with a set of distinct integers after a finite number of steps. -/
theorem transformation_terminates (n p q : ℕ+) (initial : Multiset ℤ) :
  ∃ (k : ℕ) (final : Multiset ℤ),
    (Multiset.card initial = n) →
    (∀ i, Multiset.count i final ≤ 1) ∧
    (∃ (steps : ℕ → Multiset ℤ),
      steps 0 = initial ∧
      steps k = final ∧
      ∀ i : ℕ, i < k → steps (i + 1) = transform_multiset p q (steps i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_terminates_l28_2893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_negative_derivative_l28_2866

variable (h : ℝ → ℝ)
variable (a : ℝ)

-- The tangent line to the curve y = h(x) at the point (a, h(a)) is given by 2x + y + 1 = 0
def tangent_line (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- The derivative of h at a is equal to the slope of the tangent line
def derivative_equals_slope (h : ℝ → ℝ) (a : ℝ) : Prop := deriv h a = -2

theorem tangent_line_implies_negative_derivative (h : ℝ → ℝ) (a : ℝ) :
  tangent_line a (h a) → deriv h a < 0 := by
  intro h_tangent
  have h_deriv : deriv h a = -2 := by
    -- Here we would prove that the derivative equals -2 based on the tangent line equation
    sorry
  rw [h_deriv]
  exact neg_neg_of_pos (by norm_num)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_negative_derivative_l28_2866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l28_2820

/-- The length of a goods train given its speed, platform length, and crossing time -/
theorem goods_train_length
  (train_speed : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_speed = 72)  -- km/hr
  (h2 : platform_length = 300)  -- meters
  (h3 : crossing_time = 26)  -- seconds
  : ℝ :=
by
  -- Convert train speed to m/s
  let train_speed_ms := train_speed * 1000 / 3600
  
  -- Calculate total distance covered
  let total_distance := train_speed_ms * crossing_time
  
  -- Calculate train length
  let train_length := total_distance - platform_length
  
  -- Return the train length
  exact train_length

-- Example usage (commented out to avoid evaluation errors)
-- #eval goods_train_length 72 300 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l28_2820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_arrangement_probability_l28_2884

def total_trees : ℕ := 15
def maple_trees : ℕ := 5
def oak_trees : ℕ := 3
def birch_trees : ℕ := 7

theorem tree_arrangement_probability :
  (maple_trees * (total_trees - birch_trees).choose birch_trees : ℚ) /
  ((total_trees - 1).choose birch_trees * (total_trees - birch_trees - 1).choose (maple_trees - 1)) =
  1 / 3003 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_arrangement_probability_l28_2884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outbound_speed_l28_2894

/-- Given a round trip with return speed of 88 mph and average speed of 99 mph,
    prove that the outbound speed is 113 mph. -/
theorem outbound_speed (return_speed average_speed outbound_speed : ℝ) : 
  return_speed = 88 → 
  average_speed = 99 → 
  average_speed = (2 * outbound_speed * return_speed) / (outbound_speed + return_speed) →
  outbound_speed = 113 := by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

#check outbound_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outbound_speed_l28_2894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_line_equation_l28_2821

/-- The equation of a line passing through two points -/
def line_equation (x1 y1 x2 y2 : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- The equation of a circle with center (a, b) and radius r -/
def circle_equation (a b r : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (x - a)^2 + (y - b)^2 = r^2

/-- Two points are symmetric about a line if the line is the perpendicular bisector of the segment connecting the points -/
def symmetric_about_line (x1 y1 x2 y2 : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (xm ym : ℝ), 
    xm = (x1 + x2) / 2 ∧ 
    ym = (y1 + y2) / 2 ∧ 
    l xm ym ∧
    ∀ x y, (y2 - y1) * (x - xm) = -(x2 - x1) * (y - ym)

theorem circle_symmetry_line_equation :
  ∀ (x y : ℝ),
    (circle_equation 7 (-4) 3 x y) ∧ 
    (circle_equation (-5) 6 3 x y) ∧
    (symmetric_about_line 7 (-4) (-5) 6 (λ x y ↦ 6*x - 5*y - 1 = 0)) →
    (λ x y ↦ 6*x - 5*y - 1 = 0) x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_line_equation_l28_2821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_roots_of_unity_real_tenth_power_l28_2880

theorem thirtieth_roots_of_unity_real_tenth_power :
  let S := {z : ℂ | z^30 = 1}
  ∃ T : Finset ℂ, (↑T : Set ℂ) ⊆ S ∧ Finset.card T = 20 ∧ 
    (∀ z ∈ T, (z^10 : ℂ).re = z^10 ∧ (z^10 : ℂ).im = 0) ∧
    ∀ z ∈ S \ (↑T : Set ℂ), (z^10 : ℂ).im ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_roots_of_unity_real_tenth_power_l28_2880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_and_tangent_lines_l28_2891

noncomputable section

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-1, 0)
def P : ℝ × ℝ := (Real.sqrt 2, 2)

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y + 1 = 0

-- Define B as the symmetric point of A
def B : ℝ × ℝ := (1, 0)

-- Define the circumcircle equation
def is_on_circumcircle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- Define the tangent line equations
def is_tangent_line (x y : ℝ) : Prop := 
  x = Real.sqrt 2 ∨ Real.sqrt 2 * x + 4 * y - 10 = 0

theorem circumcircle_and_tangent_lines :
  (∀ x y, is_on_circumcircle x y ↔ (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) ∧
  (∀ x y, is_on_circumcircle x y ∧ is_tangent_line x y → (x - P.1)^2 + (y - P.2)^2 = 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_and_tangent_lines_l28_2891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l28_2883

theorem sum_remainder_theorem (a b c : ℕ) 
  (ha : a % 30 = 13)
  (hb : b % 30 = 19)
  (hc : c % 30 = 22) : 
  (a + b + c) % 30 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l28_2883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l28_2864

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := (1/3) * x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := x^2 + 1

-- Define the point of interest
def point : ℝ × ℝ := (1, 4/3)

-- Define the slope of the tangent line at the point
def tangent_slope : ℝ := f' point.1

-- Define the y-intercept of the tangent line
def y_intercept : ℝ := point.2 - tangent_slope * point.1

-- Define the x-intercept of the tangent line
def x_intercept : ℝ := -y_intercept / tangent_slope

-- Theorem: The area of the triangle is 1/9
theorem tangent_triangle_area :
  (1/2) * x_intercept * y_intercept = 1/9 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l28_2864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_APR_l28_2844

-- Define the circle and points
variable (circle : Set (EuclideanSpace ℝ (Fin 2)))
variable (A B C P Q R : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_tangent (line : Set (EuclideanSpace ℝ (Fin 2))) (circle : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

def is_midpoint (M P1 P2 : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

noncomputable def distance (P1 P2 : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def line_through (P1 P2 : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- State the theorem
theorem perimeter_of_triangle_APR :
  is_tangent (line_through A B) circle →
  is_tangent (line_through A C) circle →
  is_tangent (line_through P R) circle →
  B ∈ circle →
  C ∈ circle →
  Q ∈ circle →
  P ∈ line_through A B →
  R ∈ line_through A C →
  distance A B = 30 →
  is_midpoint Q B P →
  distance A P + distance P R + distance R A = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_APR_l28_2844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2002_equals_3_l28_2888

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => Nat.gcd (a n) (n + 1) + 1

theorem a_2002_equals_3 : a 2002 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2002_equals_3_l28_2888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_positive_in_first_third_quadrant_l28_2845

theorem sin_cos_positive_in_first_third_quadrant (a : ℝ) (α : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : (2 * a) = Real.cos α * (Real.sqrt ((2 * a)^2 + (3 * a)^2)))
  (h3 : (3 * a) = Real.sin α * (Real.sqrt ((2 * a)^2 + (3 * a)^2))) :
  Real.sin α * Real.cos α > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_positive_in_first_third_quadrant_l28_2845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l28_2837

open Set
open Function

noncomputable def s (x : ℝ) : ℝ := 1 / ((2 - x)^2 + 1)

theorem s_range : range s = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l28_2837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisibility_l28_2827

theorem smallest_n_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(2^2010 ∣ 3^(2*m) - 1)) ∧ (2^2010 ∣ 3^(2*n) - 1) ↔ n = 2^2007 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisibility_l28_2827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_locus_l28_2814

/-- The locus of point P after rotating vector MQ 90° clockwise around M on an ellipse -/
theorem ellipse_locus (a b P : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hP : 0 < P) (hPa : P < a) :
  let ellipse := fun x y ↦ x^2 / a^2 + y^2 / b^2 = 1
  ∀ x y, (∃ θ : ℝ, ellipse (a * Real.cos θ) (b * Real.sin θ) ∧
    x - P = -b * Real.sin θ ∧ y = P - a * Real.cos θ) →
  (x - P)^2 / b^2 + (y - P)^2 / a^2 = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_locus_l28_2814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_l28_2833

-- Define a right triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0  -- Right angle at C
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6  -- AB = 6
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3  -- AC = 3

-- Define the theorem
theorem sin_B_value (t : RightTriangle) : 
  Real.sin (Real.arctan ((t.C.2 - t.A.2) / (t.C.1 - t.A.1))) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_value_l28_2833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strength_order_l28_2854

-- Define the type for individuals
inductive Person : Type
  | A | B | C | D

-- Define a relation for "stronger than"
def stronger_than : Person → Person → Prop := sorry

-- Define a relation for "evenly matched"
def evenly_matched : List Person → List Person → Prop := sorry

-- Define a function to represent combined strength
def combined_strength : List Person → Person := sorry

-- Define the conditions
axiom condition1 : evenly_matched [Person.A, Person.B] [Person.C, Person.D]
axiom condition2 : stronger_than Person.A Person.C ∧ stronger_than Person.D Person.B
axiom condition3 : stronger_than Person.B (combined_strength [Person.A, Person.C])

-- Define the theorem to be proved
theorem strength_order :
  stronger_than Person.D Person.B ∧
  stronger_than Person.B Person.A ∧
  stronger_than Person.A Person.C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strength_order_l28_2854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salinity_mixture_theorem_l28_2816

/-- Represents a saline solution with a given volume and salinity -/
structure SalineSolution where
  volume : ℝ
  salinity : ℝ

/-- Calculates the salinity of a mixture of two saline solutions -/
noncomputable def mixtureSalinity (a b : SalineSolution) : ℝ :=
  (a.volume * a.salinity + b.volume * b.salinity) / (a.volume + b.volume)

theorem salinity_mixture_theorem (a b : SalineSolution) 
  (h1 : a.salinity = 0.08)
  (h2 : b.salinity = 0.05)
  (h3 : ∃ (va vb : ℝ), va > 0 ∧ vb > 0 ∧ mixtureSalinity ⟨va, a.salinity⟩ ⟨vb, b.salinity⟩ = 0.062) :
  mixtureSalinity ⟨0.25, a.salinity⟩ ⟨1/6, b.salinity⟩ = 0.045 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salinity_mixture_theorem_l28_2816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relationship_l28_2838

/-- A function f with a real parameter k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2 / x^3 - 3

/-- Theorem stating the relationship between f(ln 6) and f(ln(1/6)) -/
theorem f_relationship (k : ℝ) :
  f k (Real.log 6) = 1 → f k (Real.log (1/6)) = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_relationship_l28_2838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l28_2842

theorem inequality_holds_iff_a_leq_neg_two (a : ℝ) :
  a < 0 →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 > 1 + Real.cos x) ↔
  a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_neg_two_l28_2842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l28_2886

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Point type representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the minimum value of |MP| + |MF| -/
theorem min_sum_distances (para : Parabola) (P : Point) :
  para.equation = (fun x y => y^2 = 4*x) →
  para.focus = (1, 0) →
  P = ⟨4, 1⟩ →
  (∀ M : Point, para.equation M.x M.y →
    distance M P + distance M ⟨para.focus.1, para.focus.2⟩ ≥ 6) ∧
  (∃ M : Point, para.equation M.x M.y ∧
    distance M P + distance M ⟨para.focus.1, para.focus.2⟩ = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l28_2886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l28_2803

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.mk (a^2 - 1) (a - 1)).re = 0 ∧ (Complex.mk (a^2 - 1) (a - 1)).im ≠ 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l28_2803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_correct_closest_point_l28_2867

/-- Represents the position of the bug -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculates the final position of the bug after infinite moves -/
noncomputable def finalPosition : Position :=
  let firstMove : ℝ := 1
  let secondMove : ℝ := 1/2
  let thirdMove : ℝ := 1/4
  let ratio : ℝ := -1/4
  { x := firstMove + (thirdMove / Real.sqrt 2) * (1 / (1 + (-ratio)))
  , y := secondMove + (thirdMove / Real.sqrt 2) * (1 / (1 + (-ratio))) }

/-- Theorem stating that the calculated final position is correct -/
theorem final_position_correct :
  finalPosition.x = 1 + (1 / (4 * Real.sqrt 2)) * (1 / (1 + 1/4)) ∧
  finalPosition.y = 1/2 + (1 / (4 * Real.sqrt 2)) * (1 / (1 + 1/4)) := by
  sorry

/-- Function to calculate distance between two positions -/
noncomputable def distance (p1 p2 : Position) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating which point the bug comes closest to -/
theorem closest_point (p : Position) :
  p.x = 1.1 ∧ p.y = 0.7 →
  ∀ q : Position, q ≠ p → distance finalPosition p ≤ distance finalPosition q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_correct_closest_point_l28_2867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_sum_ratio_l28_2899

theorem chessboard_sum_ratio (n : ℕ) (h1 : Even n) (h2 : n > 0) :
  let total_sum := (n^2 * (n^2 + 1) : ℚ) / 2
  let S1 := (39 : ℚ) / 103 * total_sum
  let S2 := total_sum - S1
  (∃ k : ℕ, n = 103 * k) ↔ (S1 / S2 = 39 / 64 ∧ S1.den = 1 ∧ S2.den = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_sum_ratio_l28_2899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BAD_is_obtuse_l28_2895

noncomputable section

-- Define the triangles
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 18 ∧ dist B C = 12 ∧ dist C A = 8

def triangle_ACD (A C D : ℝ × ℝ) : Prop :=
  dist C D = 7 ∧ dist D A = 6 ∧ dist C A = 8

-- Define the angle BAD
noncomputable def angle_BAD (A B D : ℝ × ℝ) : ℝ :=
  Real.arccos ((dist A B)^2 + (dist A D)^2 - (dist B D)^2) / (2 * dist A B * dist A D)

-- Theorem statement
theorem angle_BAD_is_obtuse 
  (A B C D : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : triangle_ACD A C D) 
  (h3 : (B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1) > 0) : -- Ensures B and D are on opposite sides of AC
  angle_BAD A B D > Real.pi / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BAD_is_obtuse_l28_2895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_enrolled_specific_l28_2815

/-- The number of students not enrolled in biology, mathematics, or literature classes --/
def students_not_enrolled (total : ℕ) (bio_percent : ℚ) (math_percent : ℚ) (lit_percent : ℚ) : ℕ :=
  total - (Int.floor (↑total * bio_percent) + Int.floor (↑total * math_percent) + Int.floor (↑total * lit_percent)).toNat

/-- Theorem stating the number of students not enrolled in specific classes --/
theorem students_not_enrolled_specific : 
  students_not_enrolled 1050 (275/1000) (329/1000) (15/100) = 260 := by
  sorry

#eval students_not_enrolled 1050 (275/1000) (329/1000) (15/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_not_enrolled_specific_l28_2815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_discount_l28_2824

theorem dress_discount (d : ℝ) (h : d > 0) : 
  let discounted_price := d * (1 - 0.15)
  let staff_price := discounted_price * (1 - 0.10)
  staff_price = d * 0.765 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_discount_l28_2824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_proof_l28_2896

/-- The point in the yz-plane that is equidistant from three given points -/
noncomputable def equidistant_point : ℝ × ℝ × ℝ := (0, -1, 2/3)

/-- The first given point -/
def point1 : ℝ × ℝ × ℝ := (0, 2, -1)

/-- The second given point -/
def point2 : ℝ × ℝ × ℝ := (1, 3, 1)

/-- The third given point -/
def point3 : ℝ × ℝ × ℝ := (-1, 1, 3)

/-- Calculate the squared distance between two points in 3D space -/
def squared_distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2

/-- Theorem stating that the equidistant_point is equidistant from the three given points -/
theorem equidistant_point_proof :
  squared_distance equidistant_point point1 = squared_distance equidistant_point point2 ∧
  squared_distance equidistant_point point1 = squared_distance equidistant_point point3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_proof_l28_2896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_sum_l28_2834

/-- A configuration of three circles and an equilateral triangle -/
structure CircleTriangleConfig where
  /-- The radius of each circle -/
  radius : ℝ
  /-- The distance between the centers of any two circles -/
  center_distance : ℝ
  /-- The side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- The area of the equilateral triangle -/
  triangle_area : ℝ

/-- The specific configuration for our problem -/
noncomputable def problem_config : CircleTriangleConfig where
  radius := 32
  center_distance := 64
  triangle_side := 128
  triangle_area := 4096 * Real.sqrt 3

/-- The theorem to be proved -/
theorem circle_triangle_area_sum (config : CircleTriangleConfig) 
  (h1 : config.center_distance = 2 * config.radius)
  (h2 : config.triangle_side = config.center_distance + 2 * config.radius)
  (h3 : config.triangle_area = (Real.sqrt 3 / 4) * config.triangle_side ^ 2)
  (h4 : ∃ (a b : ℕ), config.triangle_area = Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ)) :
  ∃ (a b : ℕ), a + b = 2260992 ∧ 
    config.triangle_area = Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) := by
  sorry

#check circle_triangle_area_sum problem_config

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_sum_l28_2834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sum_of_squares_l28_2850

noncomputable section

open Real EuclideanGeometry

theorem right_triangle_sum_of_squares (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_right_angle : angle A C B = π / 2)
  (h_hypotenuse : dist A B = 3) : 
  (dist A B)^2 + (dist B C)^2 + (dist A C)^2 = 18 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sum_of_squares_l28_2850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_share_interest_rate_is_11_percent_l28_2812

-- Define the given constants
noncomputable def total_investment : ℝ := 100000
noncomputable def first_share_yield : ℝ := 0.09
noncomputable def total_interest_rate : ℝ := 0.095
noncomputable def second_share_investment : ℝ := 24999.999999999996

-- Define the function to calculate the interest rate of the second share
noncomputable def second_share_interest_rate : ℝ :=
  let first_share_investment := total_investment - second_share_investment
  let total_interest := total_interest_rate * total_investment
  let first_share_interest := first_share_yield * first_share_investment
  let second_share_interest := total_interest - first_share_interest
  (second_share_interest / second_share_investment) * 100

-- Theorem statement
theorem second_share_interest_rate_is_11_percent :
  second_share_interest_rate = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_share_interest_rate_is_11_percent_l28_2812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_for_right_triangle_l28_2869

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Triangle type -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Check if three points are collinear -/
def collinear (a b c : Point) : Prop :=
  (b.y - a.y) * (c.x - a.x) = (c.y - a.y) * (b.x - a.x)

/-- Check if a triangle has a right angle at the first point -/
def rightAngleAt (t : Triangle) : Prop :=
  (t.b.x - t.a.x) * (t.c.x - t.a.x) + (t.b.y - t.a.y) * (t.c.y - t.a.y) = 0

/-- Main theorem -/
theorem unique_point_for_right_triangle (C : Parabola) (A : Point) :
  C.p > 0 →
  C.eq = fun x y ↦ y^2 = 2 * C.p * x →
  A.x = C.p / 2 →
  A.y = C.p →
  ∃! T : Point,
    T.x = 5 * C.p / 2 ∧
    T.y = -C.p ∧
    ∀ (l : Line) (B D : Point),
      (l.slope * T.x + l.intercept = T.y) →
      (C.eq B.x B.y) →
      (C.eq D.x D.y) →
      (l.slope * B.x + l.intercept = B.y) →
      (l.slope * D.x + l.intercept = D.y) →
      ¬collinear A B D →
      rightAngleAt (Triangle.mk A B D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_for_right_triangle_l28_2869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_natural_logarithm_l28_2835

theorem base_of_natural_logarithm : ℕ := by
  -- The base of the natural logarithm
  let base : ℕ := 176

  -- The divisor
  let divisor : ℕ := 19

  -- The quotient when base is divided by divisor
  let quotient : ℕ := 9

  -- The remainder when base is divided by divisor
  let remainder : ℕ := 5

  -- The base satisfies the division property
  have division_property : base = divisor * quotient + remainder := by
    -- Proof of the division property
    rfl

  -- Proof that the base is 176
  have base_is_176 : base = 176 := by
    rfl

  exact base


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_of_natural_logarithm_l28_2835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_zero_l28_2822

/-- Given a function f(x) = x|m - x| where x ∈ ℝ and f(4) = 0, prove that m = 4 -/
theorem function_value_zero (m : ℝ) : 
  (∀ x : ℝ, (fun x => x * |m - x|) x = (fun x => x * |m - x|) x) → 
  (fun x => x * |m - x|) 4 = 0 → 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_zero_l28_2822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_speed_theorem_l28_2897

/-- The speed of the moon revolving around the earth -/
noncomputable def moon_speed_km_per_hour : ℝ := 3780

/-- The number of seconds in an hour -/
noncomputable def seconds_per_hour : ℝ := 3600

/-- The speed of the moon revolving around the earth in kilometers per second -/
noncomputable def moon_speed_km_per_second : ℝ := moon_speed_km_per_hour / seconds_per_hour

theorem moon_speed_theorem : 
  moon_speed_km_per_second = moon_speed_km_per_hour / seconds_per_hour :=
by
  -- Unfold the definition of moon_speed_km_per_second
  unfold moon_speed_km_per_second
  -- The equality now holds by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_speed_theorem_l28_2897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_selections_l28_2832

def balls : Finset ℕ := Finset.range 12

def is_odd_sum (selection : Finset ℕ) : Bool :=
  (selection.sum id) % 2 = 1

theorem odd_sum_selections : 
  (balls.powerset.filter (fun s => s.card = 5 ∧ is_odd_sum s)).card = 236 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_selections_l28_2832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sue_fraction_of_kendra_buttons_l28_2890

theorem sue_fraction_of_kendra_buttons (mari_buttons : ℕ) (sue_buttons : ℕ) :
  mari_buttons = 8 →
  sue_buttons = 22 →
  let kendra_buttons := 5 * mari_buttons + 4
  (sue_buttons : ℚ) / kendra_buttons = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sue_fraction_of_kendra_buttons_l28_2890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_to_patches_l28_2879

/-- The radius of a circular patch formed by pouring a cylinder of liquid onto water -/
noncomputable def patch_radius (cylinder_radius : ℝ) (cylinder_height : ℝ) (patch_thickness : ℝ) : ℝ :=
  (cylinder_radius * Real.sqrt (cylinder_height / patch_thickness)) / Real.sqrt 2

theorem cylinder_to_patches 
  (cylinder_radius : ℝ) 
  (cylinder_height : ℝ) 
  (patch_thickness : ℝ) 
  (h_radius : cylinder_radius = 3) 
  (h_height : cylinder_height = 6) 
  (h_thickness : patch_thickness = 0.2) :
  patch_radius cylinder_radius cylinder_height patch_thickness = 3 * Real.sqrt 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_to_patches_l28_2879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_concurrent_lines_l28_2826

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point1 : Point3D
  point2 : Point3D

/-- Represents a triangular prism -/
structure TriangularPrism where
  -- Vertices of one triangular face
  A : Point3D
  B : Point3D
  C : Point3D
  -- Vertices of the other triangular face
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  -- Intersection points of diagonals in quadrilateral faces
  A0 : Point3D
  B0 : Point3D
  C0 : Point3D

/-- Check if a point lies on a line -/
def pointOnLine (P : Point3D) (L : Line3D) : Prop :=
  ∃ t : ℝ, P = Point3D.mk
    (L.point1.x + t * (L.point2.x - L.point1.x))
    (L.point1.y + t * (L.point2.y - L.point1.y))
    (L.point1.z + t * (L.point2.z - L.point1.z))

/-- 
Theorem: In a triangular prism, the three lines connecting each vertex of one triangular face 
to the intersection of diagonals in the opposite quadrilateral face are concurrent.
-/
theorem triangular_prism_concurrent_lines (prism : TriangularPrism) : 
  ∃ (P : Point3D), 
    pointOnLine P (Line3D.mk prism.A1 prism.A0) ∧ 
    pointOnLine P (Line3D.mk prism.B1 prism.B0) ∧ 
    pointOnLine P (Line3D.mk prism.C1 prism.C0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_concurrent_lines_l28_2826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_and_value_l28_2862

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x + Real.pi / 4)

theorem tan_period_and_value (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x : ℝ, f ω (x + 2 * Real.pi) = f ω x) :
  ω = 1 / 2 ∧ f ω (Real.pi / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_period_and_value_l28_2862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_eq_one_l28_2819

/-- The function y = |sin²x - 4sin x - a| has a maximum value of 4 -/
def has_max_4 (a : ℝ) : Prop :=
  ∃ (M : ℝ), M = 4 ∧ ∀ x, |Real.sin x ^ 2 - 4 * Real.sin x - a| ≤ M

/-- Theorem stating that if the function y = |sin²x - 4sin x - a| has a maximum value of 4,
    then a = 1 -/
theorem max_value_implies_a_eq_one :
  ∀ a : ℝ, has_max_4 a → a = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_eq_one_l28_2819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_terminating_decimal_l28_2858

theorem smallest_terminating_decimal (n : ℕ+) : 
  (∀ m : ℕ+, m < n → ¬(∃ k : ℕ, ∃ a b : ℕ, m.val * k = (m.val + 70) * (2^a * 5^b))) → 
  (∃ k : ℕ, ∃ a b : ℕ, n.val * k = (n.val + 70) * (2^a * 5^b)) → 
  n = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_terminating_decimal_l28_2858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_equals_two_l28_2878

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.sqrt x - x else -(Real.sqrt (-x) + x)

-- State the theorem
theorem f_neg_four_equals_two :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x ≥ 0, f x = Real.sqrt x - x) →  -- definition for x ≥ 0
  f (-4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_four_equals_two_l28_2878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l28_2825

-- Part 1
def f₁ (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

def M : Set ℝ := {x | f₁ x > 2}

theorem part_1 : M = Set.Iic (-1) ∪ Set.Ioi 5 := by sorry

-- Part 2
def f₂ (a x : ℝ) : ℝ := |x - 1| - |x + 2*a^2|

theorem part_2 : (∀ x, f₂ a x < -3*a) → a ∈ Set.Ioo (-1) (-1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_1_part_2_l28_2825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangular_sail_height_l28_2868

/-- The total area of canvas needed -/
def total_area : ℚ := 58

/-- The length of the rectangular sail -/
def rect_length : ℚ := 5

/-- The width of the rectangular sail -/
def rect_width : ℚ := 8

/-- The base of the first triangular sail -/
def tri1_base : ℚ := 3

/-- The height of the first triangular sail -/
def tri1_height : ℚ := 4

/-- The base of the second triangular sail -/
def tri2_base : ℚ := 4

/-- Calculate the area of a rectangle -/
def rect_area (length width : ℚ) : ℚ := length * width

/-- Calculate the area of a right triangle -/
def tri_area (base height : ℚ) : ℚ := (base * height) / 2

/-- The height of the second triangular sail -/
def tri2_height : ℚ := 6

theorem second_triangular_sail_height :
  tri2_height = (total_area - rect_area rect_length rect_width - tri_area tri1_base tri1_height) / tri2_base * 2 := by
  sorry

#eval tri2_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_triangular_sail_height_l28_2868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_sum_l28_2831

theorem imaginary_unit_sum : ∀ i : ℂ, i^2 = -1 → i^8 + i^20 + (i^32)⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_sum_l28_2831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relations_l28_2806

-- Define the necessary structures
structure Line
structure Plane

-- Define the relationships as axioms
axiom perpendicular : Line → Plane → Prop
axiom lies_in : Line → Plane → Prop
axiom parallel : Plane → Plane → Prop
axiom perpendicular_lines : Line → Line → Prop
axiom parallel_lines : Line → Line → Prop
axiom perpendicular_planes : Plane → Plane → Prop

-- State the theorem
theorem geometric_relations 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : lies_in m β) :
  (parallel α β → perpendicular_lines l m) ∧ 
  (parallel_lines l m → perpendicular_planes α β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_relations_l28_2806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_diagonal_specific_rhombus_l28_2817

/-- A rhombus with given area and diagonal ratio -/
structure Rhombus where
  area : ℝ
  diagonal_ratio : ℝ × ℝ
  area_positive : 0 < area
  ratio_positive : 0 < diagonal_ratio.1 ∧ 0 < diagonal_ratio.2

/-- The length of the longest diagonal of a rhombus -/
noncomputable def longest_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (2 * r.area * r.diagonal_ratio.1 / r.diagonal_ratio.2)

/-- Theorem: The length of the longest diagonal of a rhombus with area 144 and diagonal ratio 4:3 is 8√6 -/
theorem longest_diagonal_specific_rhombus :
    let r : Rhombus := ⟨144, (4, 3), by norm_num, by norm_num⟩
    longest_diagonal r = 8 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_diagonal_specific_rhombus_l28_2817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l28_2875

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + b)

noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem inverse_function_condition (b : ℝ) :
  (∀ x, f b x ≠ 0 → f_inv (f b x) = x) → b = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l28_2875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l28_2843

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a-1) * x + 1

theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 4, (∀ y ∈ Set.Ioo 1 4, x < y → f a y < f a x)) ∧
  (∀ x ∈ Set.Ioi 6, (∀ y ∈ Set.Ioi 6, x < y → f a x < f a y)) →
  a ∈ Set.Ioo 5 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l28_2843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l28_2823

theorem equation_solution (m : ℝ) : (1 / 5 : ℝ) ^ m * (1 / 4 : ℝ) ^ 2 = 1 / ((10 : ℝ) ^ 4) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l28_2823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l28_2877

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem min_phi_value (φ : ℝ) :
  φ > 0 →
  (∀ x, ∃ y, 2 * Real.sin (4 * x - 2 * φ + Real.pi / 4) = y) →
  (∀ x, 2 * Real.sin (4 * x - 2 * φ + Real.pi / 4) = 
        2 * Real.sin (4 * (Real.pi / 2 - x) - 2 * φ + Real.pi / 4)) →
  (∀ k : ℤ, φ ≥ 3 * Real.pi / 8) →
  φ = 3 * Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_value_l28_2877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_duck_flock_size_l28_2863

/-- Represents the number of birds in a flock -/
def FlockSize := Nat

/-- Represents the number of flocks -/
def FlockCount := Nat

/-- The size of seagull flocks -/
def seagull_flock_size : Nat := 10

/-- The minimum total number of ducks observed -/
def min_total_ducks : Nat := 90

/-- Checks if the total number of ducks equals the total number of seagulls -/
def equal_totals (duck_flock_size : Nat) (duck_flock_count : Nat) 
  (seagull_flock_count : Nat) : Prop :=
  duck_flock_size * duck_flock_count = seagull_flock_size * seagull_flock_count

/-- Checks if a number is a factor of another number -/
def is_factor (a b : Nat) : Prop := b % a = 0

/-- The main theorem: the smallest duck flock size satisfying all conditions is 10 -/
theorem smallest_duck_flock_size : 
  ∃ (duck_flock_size : Nat) (duck_flock_count seagull_flock_count : Nat),
    duck_flock_size > 1 ∧
    is_factor duck_flock_size min_total_ducks ∧
    equal_totals duck_flock_size duck_flock_count seagull_flock_count ∧
    ∀ (other_size : Nat),
      other_size > 1 →
      is_factor other_size min_total_ducks →
      (∃ (other_duck_count other_seagull_count : Nat),
        equal_totals other_size other_duck_count other_seagull_count) →
      duck_flock_size ≤ other_size :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_duck_flock_size_l28_2863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_quarter_circle_surface_area_proof_l28_2889

/-- The surface area generated by rotating a quarter-circle arc of radius R 
    around its tangent line at one endpoint -/
noncomputable def rotated_quarter_circle_surface_area (R : ℝ) : ℝ :=
  Real.pi * R^2 * (Real.pi - 2)

/-- Theorem stating that the surface area generated by rotating a quarter-circle arc 
    of radius R around its tangent line at one endpoint is equal to πR²(π - 2) -/
theorem rotated_quarter_circle_surface_area_proof (R : ℝ) (h : R > 0) :
  rotated_quarter_circle_surface_area R = Real.pi * R^2 * (Real.pi - 2) :=
by
  -- Unfold the definition of rotated_quarter_circle_surface_area
  unfold rotated_quarter_circle_surface_area
  -- The goal now matches the right-hand side exactly
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_quarter_circle_surface_area_proof_l28_2889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l28_2800

def is_arithmetic_sequence (a b : ℝ) : Prop := 2 * b = 2 * a + b

def is_geometric_sequence (a b : ℝ) : Prop := b ^ 2 = a ^ 2 * b

theorem range_of_m (a b m : ℝ) (h1 : is_arithmetic_sequence a b) 
  (h2 : is_geometric_sequence a b) (h3 : 0 < Real.log (a * b) / Real.log m) 
  (h4 : Real.log (a * b) / Real.log m < 1) : m > 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l28_2800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_property_l28_2805

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := a * Real.log x + 0.5 * x^2 + (1 - b) * x

-- Define the theorem
theorem extreme_points_sum_property 
  (a b x₁ x₂ : ℝ) 
  (h1 : 8 * 1 - 2 * g a b 1 - 3 = 0)  -- Tangent line condition
  (h2 : b = a + 1)                    -- Relation between a and b
  (h3 : x₁ > 0 ∧ x₂ > 0)              -- Ensure x₁ and x₂ are in the domain of g
  (h4 : deriv (g a b) x₁ = 0)    -- x₁ is an extreme point
  (h5 : deriv (g a b) x₂ = 0)    -- x₂ is an extreme point
  : g a b x₁ + g a b x₂ + 4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_property_l28_2805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l28_2885

theorem tan_pi_4_minus_alpha (α : ℝ) 
  (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.cos α = -4/5) : 
  Real.tan (π/4 - α) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l28_2885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l28_2881

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus and directrix
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define point M
def M : ℝ × ℝ := (-1, 0)

-- Define the theorem
theorem parabola_properties :
  ∃ p : ℝ,
    -- The parabola exists
    (∃ x y : ℝ, parabola p x y) ∧
    -- M is on the directrix
    directrix p (M.1) ∧
    -- p = 2
    p = 2 ∧
    -- When D, A, and F are collinear, |BF| = 3|AF|
    (∀ A B D : ℝ × ℝ,
      parabola p A.1 A.2 →
      parabola p B.1 B.2 →
      D.1 = M.1 →
      (D.2 - A.2) / (D.1 - A.1) = (A.2 - (focus p).2) / (A.1 - (focus p).1) →
      ‖B - focus p‖ = 3 * ‖A - focus p‖) ∧
    -- |AF| + |BF| > 2|MF|
    (∀ A B : ℝ × ℝ,
      parabola p A.1 A.2 →
      parabola p B.1 B.2 →
      ‖A - focus p‖ + ‖B - focus p‖ > 2 * ‖M - focus p‖) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l28_2881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l28_2861

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- Represents a line -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The focal distance of the ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  focal_distance e / e.a

/-- Check if a point (x, y) is on the ellipse -/
def on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Check if a point (x, y) is on the line -/
def on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Main theorem -/
theorem ellipse_line_intersection 
  (e : Ellipse) 
  (l : Line) 
  (h_focal : focal_distance e = 2)
  (h_ecc : eccentricity e = 1/2)
  (h_line_point : on_line l 0 1)
  (h_intersect : ∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ 
    on_ellipse e x1 y1 ∧ 
    on_ellipse e x2 y2 ∧ 
    on_line l x1 y1 ∧ 
    on_line l x2 y2 ∧ 
    distance x1 y1 x2 y2 = 3 * Real.sqrt 5 / 2) :
  l.slope = 1/2 ∨ l.slope = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l28_2861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l28_2847

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path for the cowboy's journey -/
theorem cowboy_shortest_path (stream_y : ℝ) (cowboy_init : Point) (cabin : Point) : ℝ := by
  have h1 : stream_y = 0 := by sorry
  have h2 : cowboy_init = { x := 0, y := -4 } := by sorry
  have h3 : cabin = { x := 8, y := -11 } := by sorry
  have h4 : distance cowboy_init { x := cowboy_init.x, y := stream_y } +
            distance { x := cowboy_init.x, y := -cowboy_init.y } cabin = 17 := by sorry
  exact 17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l28_2847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l28_2840

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: If the ratio of the 6th partial sum to the 3rd partial sum of a geometric sequence is 9,
    then the common ratio of the sequence is 2 -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (h : q ≠ 1) :
  (geometricSum a q 6) / (geometricSum a q 3) = 9 → q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l28_2840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_unique_integer_intersection_l28_2882

theorem range_of_a_for_unique_integer_intersection (a : ℝ) : 
  let A := {x : ℝ | x^2 + 2*x - 8 > 0}
  let B := {x : ℝ | x^2 - 2*a*x + 4 ≤ 0}
  a > 0 ∧ 
  (∃! (n : ℤ), (n : ℝ) ∈ A ∩ B) →
  13/6 ≤ a ∧ a < 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_unique_integer_intersection_l28_2882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l28_2860

noncomputable def f (x : ℝ) : ℝ := |x - 1|

noncomputable def g (x : ℝ) : ℝ :=
  if x ≠ -1 then |x^2 - 1| / |x + 1|
  else 2

theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  intro x
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l28_2860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_even_function_l28_2828

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (8 - a*x - 2*x^2)

theorem domain_of_even_function (a : ℝ) :
  (∀ x, f a x = f a (-x)) →
  {x : ℝ | f a x ≥ 0} = Set.Icc (-2) 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_even_function_l28_2828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l28_2892

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x - 2

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f (x + a) + x

-- Theorem statement
theorem problem_solution :
  (∀ x, f (x - 1) = x^2 - 3*x) →
  (∀ x, f x = x^2 - x - 2) ∧
  (∀ a, 
    (a ≥ 1 → IsMinOn (g a) (Set.Icc (-1) 3) (g a (-1))) ∧
    (-3 < a ∧ a < 1 → IsMinOn (g a) (Set.Icc (-1) 3) (g a (-a))) ∧
    (a ≤ -3 → IsMinOn (g a) (Set.Icc (-1) 3) (g a 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l28_2892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l28_2802

-- Define the function f on the domain {x₁, x₂, x₃}
variable (f : ℝ → ℝ)
variable (x₁ x₂ x₃ : ℝ)

-- Define the domain of f
def domain (x : ℝ) : Prop := x = x₁ ∨ x = x₂ ∨ x = x₃

-- Define the area of a triangle formed by three points
def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State that the graph of y = f(x) forms a triangle with area 27
axiom original_triangle_area : 
  triangle_area (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 27

-- Define the transformed function g(x) = 3f(3x)
def g (x : ℝ) : ℝ := 3 * f (3 * x)

-- State the theorem to be proved
theorem transformed_triangle_area : 
  triangle_area (x₁/3, 3 * f x₁) (x₂/3, 3 * f x₂) (x₃/3, 3 * f x₃) = 27 := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_triangle_area_l28_2802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_meets_train_probability_l28_2871

-- Define the time range in minutes (2:00 PM to 4:00 PM)
def timeRange : ℝ := 120

-- Define the waiting time of the train in minutes
def waitingTime : ℝ := 30

-- Define the probability space
def Ω : Set (ℝ × ℝ) := Set.prod (Set.Icc 0 timeRange) (Set.Icc 0 timeRange)

-- Define the event where Susan meets the train
def meetEvent : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | p.1 ≥ p.2 ∧ p.1 ≤ p.2 + waitingTime ∧ p ∈ Ω}

-- State the theorem
theorem susan_meets_train_probability : 
  (MeasureTheory.volume meetEvent) / (MeasureTheory.volume Ω) = 7 / 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_meets_train_probability_l28_2871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l28_2808

-- Define the complex number z
noncomputable def z : ℂ := 10 / (3 + Complex.I) - 2 * Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l28_2808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l28_2856

noncomputable def f (x : ℝ) := Real.log (1 - Real.tan x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∃ k : ℤ, -π/2 + k*π < x ∧ x < π/4 + k*π} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l28_2856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l28_2813

/-- The set of integers from 1 to 50, inclusive -/
def S : Finset ℕ := Finset.range 50

/-- The set of multiples of 4 in S -/
def M : Finset ℕ := S.filter (fun n => n % 4 = 0)

/-- The probability of choosing a number from M in a single draw -/
noncomputable def p : ℚ := (M.card : ℚ) / (S.card : ℚ)

/-- The probability of choosing at least one multiple of 4 when randomly selecting 
    two integers from 1 to 50 (inclusive) with replacement -/
noncomputable def probability_at_least_one_multiple_of_four : ℚ := 1 - (1 - p)^2

/-- The main theorem: the probability of choosing at least one multiple of 4 
    in two independent draws is 528/1250 -/
theorem main_theorem : probability_at_least_one_multiple_of_four = 528 / 1250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l28_2813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_slope_l28_2849

/-- The general form equation of a line passing through a point with a given slope -/
def general_form_equation (x₀ y₀ m : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y ↦ m * (x - x₀) + (y - y₀) = 0

/-- Theorem: The general form equation of the line passing through (1, 2) with slope -3 is 3x + y - 5 = 0 -/
theorem line_through_point_with_slope :
  general_form_equation 1 2 (-3) = λ x y ↦ 3 * x + y - 5 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_slope_l28_2849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l28_2810

-- Define the function f(x) = 2/x
noncomputable def f (x : ℝ) : ℝ := 2 / x

-- Define the point on the curve
def point : ℝ × ℝ := (1, 2)

-- Theorem: The equation of the tangent line to f(x) at (1, 2) is 2x + y - 4 = 0
theorem tangent_line_equation :
  let x₀ := point.1
  let y₀ := point.2
  let m := -2 / x₀^2  -- Derivative of f(x) at x₀
  ∀ x y, y - y₀ = m * (x - x₀) ↔ 2 * x + y - 4 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l28_2810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l28_2809

/-- The area of the region bounded by y = sin x (0 ≤ x ≤ π) and y = 1/2 -/
noncomputable def enclosed_area : ℝ :=
  ∫ x in (Real.pi/6)..(5*Real.pi/6), Real.sin x - 1/2

/-- Theorem stating that the enclosed area is equal to √3 - π/3 -/
theorem enclosed_area_value : enclosed_area = Real.sqrt 3 - Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_value_l28_2809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_implication_l28_2848

/-- A function is convex in the sense of squares on an interval -/
def SquareConvex (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f (Real.sqrt ((x₁^2 + x₂^2) / 2)) ≤ Real.sqrt ((f x₁^2 + f x₂^2) / 2)

/-- A function is geometrically convex on an interval -/
def GeomConvex (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f (Real.sqrt (x₁ * x₂)) ≤ Real.sqrt (f x₁ * f x₂)

/-- A function is arithmetically convex (midpoint convex) on an interval -/
def ArithConvex (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

theorem convexity_implication (f : ℝ → ℝ) (I : Set ℝ) :
  SquareConvex f I ∧ GeomConvex f I → ArithConvex f I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convexity_implication_l28_2848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l28_2830

/-- Circle C with center (4, -1) and radius 5 -/
def C : Set (ℝ × ℝ) := {p | (p.1 - 4)^2 + (p.2 + 1)^2 = 25}

/-- Point M on the circle C -/
def M : ℝ × ℝ := (0, 2)

/-- Line l₁ with equation 4x - ay + 2 = 0 -/
def l₁ (a : ℝ) : Set (ℝ × ℝ) := {p | 4 * p.1 - a * p.2 + 2 = 0}

/-- Tangent line l to circle C at point M -/
def l : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 + 6 = 0}

theorem distance_between_parallel_lines :
  ∃ a : ℝ, M ∈ C ∧ (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l₁ a → (p.1 - q.1) * 3 = (p.2 - q.2) * 4) ∧ 
  (∃ d : ℝ, d = 4/5 ∧ ∀ p ∈ l, ∃ q ∈ l₁ a, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l28_2830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l28_2829

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the fixed point P
def P : ℝ × ℝ := (3, 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (M : ℝ × ℝ), parabola M.1 M.2 →
    distance M P + distance M focus ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l28_2829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_monotonic_decreasing_interval_l28_2859

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 4)

-- Define the monotonic decreasing interval
def monotonic_decreasing_interval (k : ℤ) : Set ℝ := 
  Set.Icc (k * π + π / 8) (k * π + 5 * π / 8)

-- State the theorem
theorem cos_monotonic_decreasing_interval (k : ℤ) : 
  StrictMonoOn f (monotonic_decreasing_interval k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_monotonic_decreasing_interval_l28_2859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_sum_vectors_l28_2841

theorem max_magnitude_sum_vectors (a b : ℝ × ℝ × ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  ∃ (c : ℝ × ℝ × ℝ), ‖c‖ = 3 ∧ ∀ (x : ℝ × ℝ × ℝ), ‖a + 2 • b‖ ≤ ‖c‖ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_sum_vectors_l28_2841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_product_probability_l28_2801

/-- The probability of rolling two dice with outcomes satisfying x + y + xy = 12 -/
theorem dice_sum_product_probability : 
  let outcomes := Finset.range 6
  let valid_pairs := Finset.filter (fun p => p.1 + p.2 + p.1 * p.2 = 12) (outcomes.product outcomes)
  (Finset.card valid_pairs : ℚ) / (Finset.card (outcomes.product outcomes) : ℚ) = 1 / 36 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_product_probability_l28_2801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l28_2873

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define the line l with slope 1
def line_l (x y a b : ℝ) : Prop := y - b = x - a

-- Define the property that a point is on the circle
def on_circle (x y : ℝ) : Prop := circle_C x y

-- Define the property that a point is on the line
def on_line (x y a b : ℝ) : Prop := line_l x y a b

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs ((x₂ - x₁)*(y₃ - y₁) - (x₃ - x₁)*(y₂ - y₁))

theorem circle_and_line_properties :
  ∃ (center_x center_y radius : ℝ) (a₁ b₁ a₂ b₂ max_area : ℝ),
    -- 1) Center and radius of the circle
    center_x = 1 ∧ center_y = -2 ∧ radius = 3 ∧
    ∀ (x y : ℝ), on_circle x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧
    -- 2) Existence of lines l satisfying the conditions
    (∀ (x y : ℝ), on_line x y a₁ b₁ ↔ x - y - 4 = 0) ∧
    (∀ (x y : ℝ), on_line x y a₂ b₂ ↔ x - y + 1 = 0) ∧
    -- 3) Maximum area of triangle CAB
    max_area = 9/2 ∧
    ∀ (a b : ℝ), 
      (∃ (x₁ y₁ x₂ y₂ : ℝ), 
        on_circle x₁ y₁ ∧ on_circle x₂ y₂ ∧ 
        on_line x₁ y₁ a b ∧ on_line x₂ y₂ a b ∧ 
        x₁ ≠ x₂ ∧ y₁ ≠ y₂) →
      area_triangle center_x center_y x₁ y₁ x₂ y₂ ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l28_2873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l28_2846

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.ABC : Triangle := { A := (0, 0), B := (-1, -3), C := (1, 1) }

def altitude_CE : Line := { a := 4, b := 3, c := -7 }
def median_AD : Line := { a := 1, b := -3, c := -3 }

-- Helper functions (defined as Props)
def is_altitude (l : Line) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop := sorry
def is_median (l : Line) (B : ℝ × ℝ) (C : ℝ × ℝ) : Prop := sorry
def is_side (l : Line) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem triangle_properties :
  ∃ (ABC : Triangle),
    ABC.B = (-1, -3) ∧
    (∃ (CE : Line), CE = altitude_CE ∧ is_altitude CE ABC.A ABC.B) ∧
    (∃ (AD : Line), AD = median_AD ∧ is_median AD ABC.B ABC.C) →
    (∃ (AB : Line), AB = { a := 3, b := -4, c := -9 } ∧ is_side AB ABC.A ABC.B) ∧
    ABC.C = (1, 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l28_2846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l28_2853

-- Define a real-valued function type
def RealFunction := ℝ → ℝ

-- Define symmetry about the origin
def symmetric_about_origin (f : RealFunction) : Prop :=
  ∀ x, f (-x) = -f x

-- Define even function
def is_even (f : RealFunction) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem necessary_not_sufficient :
  (∀ f : RealFunction, symmetric_about_origin f → is_even (fun x ↦ |f x|)) ∧
  (∃ f : RealFunction, is_even (fun x ↦ |f x|) ∧ ¬symmetric_about_origin f) := by
  sorry

#check necessary_not_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l28_2853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l28_2851

def partition_ways (n : ℕ) : ℕ := 
  (Nat.factorial (3 * n))^2 / (Nat.factorial n)^3 / 2^(2 * n)

theorem partition_theorem (n : ℕ) :
  partition_ways n = (Nat.factorial (3 * n))^2 / (Nat.factorial n)^3 / 2^(2 * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_theorem_l28_2851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_statements_l28_2839

-- Define the basic geometric concepts
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Angle where
  measure : ℝ

-- Define the geometric relationships
def shortest_distance (p q : Point) : ℝ := sorry

def vertical_angles (α β : Angle) : Prop := sorry

def complementary_angles (α β : Angle) : Prop := sorry

def parallel_lines (l m : Line) : Prop := sorry

def point_not_on_line (p : Point) (l : Line) : Prop := sorry

-- Define the statements
def statement_A : Prop :=
  ∀ p q : Point, shortest_distance p q = shortest_distance q p

def statement_B : Prop :=
  ∀ α β : Angle, vertical_angles α β → α = β

def statement_C : Prop :=
  ∀ α β : Angle, complementary_angles α β → α.measure = β.measure → α.measure = 45

def statement_D : Prop :=
  ∀ p : Point, ∀ l : Line, point_not_on_line p l →
    ∃! m : Line, parallel_lines l m ∧ point_not_on_line p m

-- Theorem stating that A, B, C are correct, and D is also correct but identified as "incorrect"
theorem geometric_statements :
  statement_A ∧ statement_B ∧ statement_C ∧ statement_D ∧
  (statement_D → ¬(statement_A ∧ statement_B ∧ statement_C ∧ statement_D)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_statements_l28_2839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l28_2857

open Real

noncomputable def f (x : ℝ) : ℝ := (x * (1 - x)) / ((x + 1) * (x + 2) * (2 * x + 1))

theorem f_max_value :
  ∃ (M : ℝ), M = (1/3) * (8 * Real.sqrt 2 - 5 * Real.sqrt 5) ∧
  (∀ x : ℝ, x > 0 ∧ x ≤ 1 → f x ≤ M) ∧
  (∃ x : ℝ, x > 0 ∧ x ≤ 1 ∧ f x = M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l28_2857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l28_2804

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → 
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l28_2804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_sunday_l28_2898

-- Define the days of the week
inductive Day : Type
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day
  | Sunday : Day

-- Define the sports
inductive Sport : Type
  | Basketball : Sport
  | Tennis : Sport
  | Running : Sport
  | Swimming : Sport
  | Cycling : Sport

-- Define Mahdi's schedule
def schedule : Day → Sport := sorry

-- Define successor function for Day
def Day.succ : Day → Day
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Conditions
axiom monday_basketball : schedule Day.Monday = Sport.Basketball
axiom wednesday_tennis : schedule Day.Wednesday = Sport.Tennis

axiom running_days : ∃ (d1 d2 d3 : Day), 
  schedule d1 = Sport.Running ∧ 
  schedule d2 = Sport.Running ∧ 
  schedule d3 = Sport.Running ∧ 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

axiom consecutive_running : ∃ (d1 d2 : Day), 
  schedule d1 = Sport.Running ∧ 
  schedule d2 = Sport.Running ∧ 
  (Day.succ d1 = d2 ∨ Day.succ d2 = d1)

axiom swimming_day : ∃ (d : Day), schedule d = Sport.Swimming
axiom cycling_day : ∃ (d : Day), schedule d = Sport.Cycling

axiom no_cycle_before_swim : ∀ (d : Day), 
  schedule d = Sport.Cycling → schedule (Day.succ d) ≠ Sport.Swimming

axiom no_cycle_after_run : ∀ (d : Day), 
  schedule d = Sport.Running → schedule (Day.succ d) ≠ Sport.Cycling

-- Theorem to prove
theorem mahdi_swims_on_sunday : schedule Day.Sunday = Sport.Swimming := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_sunday_l28_2898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_coprime_u_l28_2836

noncomputable def u (n : ℕ) : ℤ := ⌊((3 + Real.sqrt 10)^(2^n)) / 4⌋ + 1

theorem pairwise_coprime_u : ∀ m n : ℕ, m ≠ n → Int.gcd (u m) (u n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairwise_coprime_u_l28_2836
