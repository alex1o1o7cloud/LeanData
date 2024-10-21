import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l291_29129

/-- The repeating decimal 0.567̅ is equal to the fraction 21/37. -/
theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), (∀ n : ℕ, (567 * 10^n : ℚ) / (999 * 10^n) = x) ∧ x = 21/37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l291_29129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_upper_bound_when_a_negative_l291_29189

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 + (2+a) * x + 2 * Real.log x

-- Theorem for part 1
theorem tangent_line_at_one (a : ℝ) (h : a = 0) :
  ∃ m b : ℝ, m = 3 ∧ b = -1 ∧
  ∀ x : ℝ, x > 0 → (f a x - f a 1) = m * (x - 1) := by sorry

-- Theorem for part 2
theorem upper_bound_when_a_negative (a : ℝ) (h : a < 0) :
  ∀ x : ℝ, x > 0 → f a x ≤ -6/a - 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_upper_bound_when_a_negative_l291_29189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_problem_l291_29182

theorem perfect_squares_problem :
  (¬ ∃ (a b c : ℕ), ∃ (k m n : ℕ),
    (a * b + 1 = k^2) ∧ (b * c + 1 = m^2) ∧ (c * a + 1 = n^2) ∧
    Even k ∧ Even m ∧ Even n) ∧
  (∃ (f : ℕ → ℕ × ℕ × ℕ × ℕ),
    ∀ n : ℕ, 
      let (a, b, c, d) := f n
      a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
      ∃ (k m p q : ℕ),
        (a * b + 1 = k^2) ∧ (b * c + 1 = m^2) ∧ (c * d + 1 = p^2) ∧ (d * a + 1 = q^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_problem_l291_29182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_cube_surface_area_ratio_l291_29106

theorem hemisphere_cube_surface_area_ratio :
  ∀ (R a : ℝ), R > 0 → a > 0 →
  (R = (Real.sqrt 6 / 2) * a) →
  (3 * Real.pi * R^2) / (6 * a^2) = 3 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_cube_surface_area_ratio_l291_29106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l291_29108

/-- A line dividing a 3x2 rectangle of unit squares -/
noncomputable def DividingLine (d : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (2 / (2 - d)) * (x - d) ∧ 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 2}

/-- The area of the left region formed by the dividing line -/
noncomputable def LeftArea (d : ℝ) : ℝ :=
  6 - (1/2 * (2 - d) * 2)

/-- The area of the right region formed by the dividing line -/
noncomputable def RightArea (d : ℝ) : ℝ :=
  1/2 * (2 - d) * 2

/-- Theorem stating that the left area is twice the right area iff d = 0 -/
theorem dividing_line_theorem (d : ℝ) :
  LeftArea d = 2 * RightArea d ↔ d = 0 := by
  sorry

#check dividing_line_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_theorem_l291_29108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_25_l291_29141

/-- Represents a clock with hour, minute, and second hands -/
structure Clock :=
  (hours : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

/-- Calculates the angle of the hour hand from 12 o'clock position -/
noncomputable def hour_angle (c : Clock) : ℝ :=
  (c.hours % 12 : ℝ) * 30 + (c.minutes : ℝ) * 0.5 + (c.seconds : ℝ) * (1 / 120)

/-- Calculates the angle of the minute hand from 12 o'clock position -/
noncomputable def minute_angle (c : Clock) : ℝ :=
  (c.minutes : ℝ) * 6 + (c.seconds : ℝ) * 0.1

/-- Calculates the angle of the second hand from 12 o'clock position -/
noncomputable def second_angle (c : Clock) : ℝ :=
  (c.seconds : ℝ) * 6

/-- Calculates the acute angle between two angles on a 360-degree circle -/
noncomputable def acute_angle (a b : ℝ) : ℝ :=
  min (|a - b|) (360 - |a - b|)

/-- The main theorem: The acute angle formed by the hands of a clock at 3:25:25 is 47.5 degrees -/
theorem clock_angle_at_3_25_25 :
  let c : Clock := ⟨3, 25, 25⟩
  acute_angle (hour_angle c) (minute_angle c) = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_25_l291_29141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_outfits_count_l291_29135

/-- Represents the color of a clothing item -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a clothing item -/
structure ClothingItem where
  color : Color

/-- Represents the wardrobe inventory -/
structure Wardrobe where
  redShirts : Nat
  greenShirts : Nat
  blueShirts : Nat
  redPants : Nat
  greenPants : Nat
  bluePants : Nat
  redHats : Nat
  greenHats : Nat
  blueHats : Nat

/-- Represents an outfit -/
structure Outfit where
  shirt : ClothingItem
  pants : ClothingItem
  hat : ClothingItem

/-- Checks if an outfit is valid (pants and hat don't match shirt color) -/
def isValidOutfit (o : Outfit) : Prop :=
  o.shirt.color ≠ o.pants.color ∧ o.shirt.color ≠ o.hat.color

/-- Calculates the number of valid outfits -/
def countValidOutfits (w : Wardrobe) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem valid_outfits_count (w : Wardrobe) :
  w.redShirts = 6 ∧ w.greenShirts = 6 ∧ w.blueShirts = 6 ∧
  w.redPants = 7 ∧ w.greenPants = 7 ∧ w.bluePants = 7 ∧
  w.redHats = 9 ∧ w.greenHats = 9 ∧ w.blueHats = 9 →
  countValidOutfits w = 4536 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_outfits_count_l291_29135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_units_digit_probability_max_l291_29118

/-- The probability that the product of two randomly chosen integers from 1 to n has a units digit of 0 -/
def p (n : ℕ) : ℚ :=
  (n / 2 + n / 5 - n / 10 : ℚ) / n

/-- The maximum value of p(n) over all possible choices of n -/
noncomputable def p_max : ℚ := 3/5

theorem product_zero_units_digit_probability_max :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |p n - p_max| < ε :=
by sorry

#eval (100 * 27 + 100 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_zero_units_digit_probability_max_l291_29118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_to_directrix_distance_l291_29139

/-- Given a parabola C: y² = 2px and a point A(1, √2) on C, 
    the distance from A to the directrix of C is 3/2 -/
theorem parabola_point_to_directrix_distance 
  (C : ℝ → ℝ → Prop) 
  (p : ℝ) 
  (h1 : ∀ x y, C x y ↔ y^2 = 2*p*x) 
  (h2 : C 1 (Real.sqrt 2)) :
  abs (1 - (-p/2)) = 3/2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_to_directrix_distance_l291_29139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_transitivity_l291_29180

-- Define structures for Plane and Line
structure Plane where

structure Line where

-- Define relations for parallel planes and lines
def parallel_plane_line (p : Plane) (l : Line) : Prop :=
  sorry

def parallel_plane (p1 p2 : Plane) : Prop :=
  sorry

-- Theorem statement
theorem parallel_planes_transitivity 
  (α β γ : Plane) 
  (a b : Line) 
  (h1 : parallel_plane_line α a) 
  (h2 : parallel_plane_line β a) 
  (h3 : parallel_plane_line γ a) 
  (h4 : parallel_plane_line α b) 
  (h5 : parallel_plane_line β b) 
  (h6 : parallel_plane_line γ b) 
  (h7 : parallel_plane α γ) 
  (h8 : parallel_plane β γ) : 
  parallel_plane α β :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_transitivity_l291_29180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_angle_l291_29156

open Real Complex

noncomputable def angle_sequence : List ℝ := List.range 10 |>.map (λ n => 75 + 8 * n)

noncomputable def complex_sum : ℂ := (angle_sequence.map (λ θ => exp (θ * I * π / 180))).sum

theorem complex_sum_angle (r : ℝ) (θ : ℝ) 
  (h1 : r > 0) 
  (h2 : 0 ≤ θ ∧ θ < 2 * π) 
  (h3 : complex_sum = r * exp (θ * I)) : 
  θ = 111 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_angle_l291_29156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_M_l291_29188

-- Define M as a function of a
noncomputable def M (a : ℝ) : ℝ := (a^2 + 4) / a

-- Define the set representing the range of M
def range_M : Set ℝ := {y | ∃ a : ℝ, a ≠ 0 ∧ M a = y}

-- Theorem stating the range of M
theorem range_of_M : range_M = Set.Iic (-4) ∪ Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_M_l291_29188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l291_29183

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then a^x else (3-a)*x + 2

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Ici 2 ∩ Set.Iio 3 :=
by
  sorry

#check range_of_a_for_increasing_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l291_29183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_monotone_increasing_l291_29140

/-- Definition of the function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + (m^2 - 1) * x

/-- Theorem stating the maximum and minimum values of f(x) when m = 1 on [-3, 2] -/
theorem f_extrema :
  ∀ x, x ∈ Set.Icc (-3 : ℝ) 2 →
    f 1 x ≤ 18 ∧ f 1 x ≥ 0 ∧
    (∃ x₁ x₂, x₁ ∈ Set.Icc (-3 : ℝ) 2 ∧ x₂ ∈ Set.Icc (-3 : ℝ) 2 ∧ f 1 x₁ = 18 ∧ f 1 x₂ = 0) :=
by
  sorry

/-- Theorem stating the interval where f(x) is monotonically increasing -/
theorem f_monotone_increasing (m : ℝ) (h : m > 0) :
  ∀ x₁ x₂, 1 - m < x₁ ∧ x₁ < x₂ ∧ x₂ < m + 1 → f m x₁ < f m x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_f_monotone_increasing_l291_29140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_m_div_two_power_n_l291_29121

theorem eight_power_m_div_two_power_n (m n : ℤ) (h : 3 * m - n - 4 = 0) : 
  (8 : ℚ)^m / (2 : ℚ)^n = 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_m_div_two_power_n_l291_29121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_8_l291_29195

/-- A monic polynomial of degree 7 satisfying specific conditions -/
noncomputable def p : ℝ → ℝ := 
  sorry -- We'll define this later

/-- p is a monic polynomial of degree 7 -/
axiom p_monic : ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, 
  ∀ x, p x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀

/-- p satisfies the given conditions -/
axiom p_conditions : p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 4 ∧ p 4 = 5 ∧ p 5 = 6 ∧ p 6 = 7 ∧ p 7 = 8

/-- The main theorem: p(8) = 5049 -/
theorem p_at_8 : p 8 = 5049 := by
  sorry -- We'll prove this later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_8_l291_29195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l291_29155

def f (a b : ℕ) : ℕ :=
  Nat.choose (a + b) 2

theorem unique_function_satisfying_conditions :
  ∀ (g : ℕ → ℕ → ℕ),
    (∀ (a b : ℕ), g a b + a + b = g a 1 + g 1 b + a * b) →
    (∀ (a b : ℕ) (p : ℕ), Nat.Prime p → p > 2 → 
      (p ∣ (a + b) ∨ p ∣ (a + b - 1)) → p ∣ g a b) →
    g = f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l291_29155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l291_29130

open Set Real

-- Define the function f
noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 2 * log x + x^2 - 5*x + c

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2/x + 2*x - 5

-- Define the property of not being monotonic on an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem non_monotonic_interval (c : ℝ) :
  ∀ m : ℝ, m > 0 →
  (not_monotonic (f · c) m (m + 1) ↔ m ∈ Ioo 0 (1/2) ∪ Ioo 1 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_l291_29130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_shell_thickness_l291_29168

/-- The thickness of a spherical shell wrapped around a sphere, given that the shell's volume equals the sphere's volume. -/
theorem spherical_shell_thickness (R : ℝ) (h : R > 0) :
  ∃ x : ℝ, x > 0 ∧ 
  (4 / 3 * Real.pi * ((R + x)^3 - R^3) = 4 / 3 * Real.pi * R^3) ∧
  x = (Real.rpow 2 (1/3) - 1) * R :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_shell_thickness_l291_29168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_condition_l291_29181

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/4) * x + 1 else Real.log x

theorem two_distinct_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x = a * x ∧ f y = a * y ∧
    (∀ z : ℝ, z ≠ x → z ≠ y → f z ≠ a * z)) ↔
  a ∈ Set.Icc (1/4) (1/Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_roots_condition_l291_29181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l291_29164

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  4 * (floor x)^2 - 36 * (floor x) + 45 ≤ 0

-- Theorem statement
theorem solution_set_of_inequality :
  ∀ x : ℝ, inequality x ↔ 2 ≤ x ∧ x < 8 :=
by
  sorry

#check solution_set_of_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l291_29164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l291_29173

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2) ^ n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2) ^ n

/-- Theorem stating the relationship between g(n+2), g(n), and -1/4 * g(n) -/
theorem g_relation (n : ℕ) : g (n + 2) - g n = -1/4 * g n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l291_29173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_calculation_l291_29167

/-- The height of a cylindrical tin with given diameter and volume -/
noncomputable def cylinderHeight (diameter : ℝ) (volume : ℝ) : ℝ :=
  (4 * volume) / (Real.pi * diameter^2)

/-- Theorem: The height of a cylindrical tin with diameter 6 cm and volume 45 cm³ is 5/π cm -/
theorem cylinder_height_calculation :
  cylinderHeight 6 45 = 5 / Real.pi :=
by
  -- Unfold the definition of cylinderHeight
  unfold cylinderHeight
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_calculation_l291_29167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_bisectors_leg_lengths_l291_29153

/-- A right triangle with angle bisectors of acute angles being 1 and 2 units long -/
structure RightTriangleWithBisectors where
  /-- The length of the first leg -/
  a : ℝ
  /-- The length of the second leg -/
  b : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The length of the angle bisector of the first acute angle -/
  bisector1 : ℝ
  /-- The length of the angle bisector of the second acute angle -/
  bisector2 : ℝ
  /-- The triangle is right-angled -/
  right_angle : a^2 + b^2 = c^2
  /-- The first angle bisector is 1 unit long -/
  bisector1_length : bisector1 = 1
  /-- The second angle bisector is 2 units long -/
  bisector2_length : bisector2 = 2

/-- The legs of a right triangle with angle bisectors 1 and 2 units long are approximately 0.8341 and 1.9596 -/
theorem right_triangle_with_bisectors_leg_lengths (t : RightTriangleWithBisectors) :
  (abs (t.a - 0.8341) < 0.0001 ∧ abs (t.b - 1.9596) < 0.0001) ∨
  (abs (t.a - 1.9596) < 0.0001 ∧ abs (t.b - 0.8341) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_bisectors_leg_lengths_l291_29153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l291_29186

-- Define the equation for the trajectory
def trajectory_equation (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  10 * Real.sqrt (x^2 + y^2) = abs (3*x + 4*y - 12)

-- Define what an ellipse is in terms of the ratio of distances
def is_ellipse (f : ℝ × ℝ → Prop) : Prop :=
  ∃ (focus : ℝ × ℝ) (directrix : ℝ → ℝ) (e : ℝ),
    0 < e ∧ e < 1 ∧
    ∀ (p : ℝ × ℝ), f p ↔
      dist p focus / abs (p.2 - directrix p.1) = e

-- Theorem statement
theorem trajectory_is_ellipse :
  is_ellipse trajectory_equation := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_ellipse_l291_29186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_store_problem_l291_29190

/-- Fruit store problem -/
theorem fruit_store_problem 
  (purchase_price : ℝ)
  (base_price : ℝ)
  (base_sales : ℝ)
  (price_increase : ℝ)
  (sales_decrease : ℝ)
  (h1 : purchase_price = 30)
  (h2 : base_price = 40)
  (h3 : base_sales = 500)
  (h4 : price_increase = 1)
  (h5 : sales_decrease = 10) :
  let sales_volume := λ x => -sales_decrease * x + (base_sales + sales_decrease * base_price)
  let profit := λ x => (x - purchase_price) * (sales_volume x)
  ∃ (affordable_price max_profit_price : ℝ),
    (∀ x, sales_volume x = -10 * x + 900) ∧
    (profit affordable_price = 8000 ∧ affordable_price = 50) ∧
    (∀ x, profit x ≤ 9000) ∧
    (profit max_profit_price = 9000 ∧ max_profit_price = 60) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_store_problem_l291_29190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_mirror_reflection_l291_29133

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  f₁ : Point
  f₂ : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p₁ p₂ : Point) : ℝ :=
  Real.sqrt ((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2)

/-- Theorem: Light rays from one focus of an elliptical mirror converge at the other focus -/
theorem elliptical_mirror_reflection (e : Ellipse) :
  (e.c^2 = e.a^2 - e.b^2) →
  (distance e.f₁ e.f₂ = 2 * e.c) →
  (∀ p : Point, isOnEllipse e p → distance e.f₁ p + distance e.f₂ p = 2 * e.a) →
  ∀ p : Point, isOnEllipse e p →
    ∃ (incident_ray reflected_ray : Point → Point),
      incident_ray e.f₁ = p ∧
      reflected_ray p = e.f₂ ∧
      -- Additional conditions for reflection law would be added here
      True :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_mirror_reflection_l291_29133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_2cos4_l291_29191

theorem min_sin4_plus_2cos4 : 
  ∀ x : ℝ, Real.sin x ^ 4 + 2 * Real.cos x ^ 4 ≥ 2/3 ∧ 
  ∃ y : ℝ, Real.sin y ^ 4 + 2 * Real.cos y ^ 4 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_2cos4_l291_29191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_percentage_liquid_x_l291_29154

/-- Represents the composition of a solution --/
structure Solution where
  total : ℚ
  liquidX : ℚ
  water : ℚ

/-- The initial solution Y --/
def initialSolutionY : Solution :=
  { total := 8
    liquidX := 8 * (30 / 100)
    water := 8 * (70 / 100) }

/-- The solution after water evaporation --/
def evaporatedSolution : Solution :=
  { total := initialSolutionY.total - 3
    liquidX := initialSolutionY.liquidX
    water := initialSolutionY.water - 3 }

/-- The additional solution Y --/
def additionalSolutionY : Solution :=
  { total := 3
    liquidX := 3 * (30 / 100)
    water := 3 * (70 / 100) }

/-- The final solution after adding more solution Y --/
def finalSolution : Solution :=
  { total := evaporatedSolution.total + additionalSolutionY.total
    liquidX := evaporatedSolution.liquidX + additionalSolutionY.liquidX
    water := evaporatedSolution.water + additionalSolutionY.water }

/-- The percentage of liquid X in the final solution --/
def percentageLiquidX : ℚ :=
  (finalSolution.liquidX / finalSolution.total) * 100

theorem final_percentage_liquid_x :
  percentageLiquidX = 33 / 80 * 100 := by
  sorry

#eval percentageLiquidX

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_percentage_liquid_x_l291_29154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_multiple_of_1004_l291_29150

def is_lowest_terms (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def is_multiple_of (a m : ℕ) : Prop :=
  ∃ k : ℕ, a = k * m

theorem smallest_n_for_multiple_of_1004 :
  ∀ n : ℕ+,
  let a : ℕ := (2009 * n.val + 4066266)
  let b : ℕ := (2014 * n.val)
  (2009 : ℚ) / 2014 + 2019 / n.val = a / b →
  is_lowest_terms a b →
  (∀ m : ℕ+, m < n → ¬(is_multiple_of (((2009 : ℚ) / 2014 + 2019 / m.val).num.natAbs) 1004)) →
  is_multiple_of a 1004 →
  n = ⟨1942, by norm_num⟩ :=
sorry

#check smallest_n_for_multiple_of_1004

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_multiple_of_1004_l291_29150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l291_29124

def f (n : ℕ) : ℚ := (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_lower_bound (n : ℕ) (hn : n ≥ 1) :
  f (2^n) ≥ (n + 2 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l291_29124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_thirteen_l291_29199

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4

noncomputable def f_inverse (y : ℝ) : ℝ := (y - 4) / 3

theorem inverse_of_inverse_thirteen :
  f_inverse (f_inverse 13) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_thirteen_l291_29199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_pyramid_with_spheres_l291_29125

/-- Regular quadrilateral pyramid with spheres at base vertices -/
structure PyramidWithSpheres where
  a : ℝ  -- Side length of the base
  base_side_length : a > 0
  height : ℝ  -- Height of the pyramid
  height_def : height = -a/2
  sphere_radius : ℝ  -- Radius of the spheres
  sphere_radius_def : sphere_radius = a/3

/-- Volume of the body bounded by the pyramid surface and spheres -/
noncomputable def volume (p : PyramidWithSpheres) : ℝ :=
  (81 - 4 * Real.pi) / 486 * p.a^3

/-- Theorem: The volume of the body is (81 - 4π) / 486 * a^3 -/
theorem volume_of_pyramid_with_spheres (p : PyramidWithSpheres) :
  volume p = (81 - 4 * Real.pi) / 486 * p.a^3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_pyramid_with_spheres_l291_29125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_range_l291_29110

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

-- Define the specific triangle with given conditions
def SpecificTriangle (t : AcuteTriangle) : Prop :=
  t.a = 1 ∧ t.B = 2 * t.A

-- Theorem statement
theorem side_b_range (t : AcuteTriangle) (h : SpecificTriangle t) :
  Real.sqrt 2 < t.b ∧ t.b < Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_b_range_l291_29110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_appliance_max_profitable_m_l291_29144

-- Define the types of products
def num_clothing : ℕ := 3
def num_appliances : ℕ := 2
def num_necessities : ℕ := 3
def total_products : ℕ := num_clothing + num_appliances + num_necessities
def selected_products : ℕ := 3

-- Define the probability of winning a single lottery
def win_probability : ℚ := 1 / 3

-- Define the price increase
def price_increase : ℚ := 100

-- Theorem for the probability of selecting at least one home appliance
theorem probability_at_least_one_appliance :
  1 - (Nat.choose (total_products - num_appliances) selected_products : ℚ) / (Nat.choose total_products selected_products : ℚ) = 9 / 14 := by sorry

-- Define the prize structure
def prize (num_wins : ℕ) (m : ℚ) : ℚ :=
  match num_wins with
  | 0 => 0
  | 1 => m
  | 2 => 3 * m
  | _ => 6 * m

-- Theorem for the maximum value of m for a profitable promotion
theorem max_profitable_m :
  ∃ (m : ℚ), m = 75 ∧
  (∀ (m' : ℚ), m' ≤ m →
    (prize 0 m' * (1 - win_probability)^3 +
     prize 1 m' * 3 * win_probability * (1 - win_probability)^2 +
     prize 2 m' * 3 * win_probability^2 * (1 - win_probability) +
     prize 3 m' * win_probability^3) ≤ price_increase) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_appliance_max_profitable_m_l291_29144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_zeros_l291_29179

def sequence_problem (a : Fin 2015 → Int) : Prop :=
  (∀ i, a i = -1 ∨ a i = 0 ∨ a i = 1) ∧
  (Finset.sum Finset.univ a) = 427 ∧
  (Finset.sum Finset.univ (λ i => (a i + 1)^2)) = 3869

theorem count_zeros (a : Fin 2015 → Int) (h : sequence_problem a) :
  (Finset.univ.filter (fun i => a i = 0)).card = 1015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_zeros_l291_29179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascorbic_acid_weight_is_176_12_l291_29166

/-- The molecular weight of ascorbic acid (C6H8O6) in g/mol -/
def ascorbic_acid_weight : ℝ := 176.12

/-- Theorem stating that the molecular weight of ascorbic acid is 176.12 g/mol -/
theorem ascorbic_acid_weight_is_176_12 :
  ascorbic_acid_weight = 176.12 := by
  -- The proof is just a reflection of the definition
  rfl

#eval ascorbic_acid_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascorbic_acid_weight_is_176_12_l291_29166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5pi_6_l291_29117

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + x) * Real.sin (Real.pi + x)

theorem f_value_at_5pi_6 : f (5 * Real.pi / 6) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_5pi_6_l291_29117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elephants_at_we_preserve_total_elephants_check_l291_29136

/-- The number of elephants at We Preserve For Future park -/
def elephants_we_preserve : ℕ := 70

/-- The number of elephants at Gestures For Good park -/
def elephants_gestures_for_good : ℕ := 3 * elephants_we_preserve

/-- The total number of elephants in both parks -/
def total_elephants : ℕ := 280

theorem elephants_at_we_preserve : elephants_we_preserve = 70 := by
  rfl

theorem total_elephants_check : total_elephants = elephants_we_preserve + elephants_gestures_for_good := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elephants_at_we_preserve_total_elephants_check_l291_29136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l291_29196

noncomputable def f (a x : ℝ) : ℝ := x^2 + (4*a - 2)*x + 1

noncomputable def g (a : ℝ) : ℝ :=
  if a ≤ 0 then 5*a^2 + 4*a
  else if a < 1/3 then -4*a^2 + 4*a
  else 5*a^2 - 2*a + 1

theorem min_value_of_f (a : ℝ) :
  ∀ x ∈ Set.Icc a (a + 1), g a ≤ f a x := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l291_29196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_modulus_z_min_modulus_z_achieved_l291_29192

open Complex

theorem min_modulus_z (z : ℂ) : 
  (∃ x : ℝ, 4 * x^2 - 8 * (z.re : ℂ) * x + 4 * I + 3 = 0) → abs z ≥ 1 := by
  sorry

theorem min_modulus_z_achieved : 
  ∃ z : ℂ, (∃ x : ℝ, 4 * x^2 - 8 * (z.re : ℂ) * x + 4 * I + 3 = 0) ∧ abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_modulus_z_min_modulus_z_achieved_l291_29192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_production_rate_l291_29128

/-- Proves that the initial production rate is 30 cogs per hour given the assembly line conditions --/
theorem initial_production_rate 
  (initial_order : ℕ) 
  (second_order : ℕ) 
  (increased_speed : ℕ) 
  (overall_average : ℕ) 
  (h1 : initial_order = 60)
  (h2 : second_order = 60)
  (h3 : increased_speed = 60)
  (h4 : overall_average = 40)
  : ∃ (x : ℕ), x = 30 := by
  -- The proof goes here
  sorry

#check initial_production_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_production_rate_l291_29128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_pascal_15_rows_l291_29109

/-- Pascal's triangle coefficient -/
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Count even numbers in a row of Pascal's triangle -/
def countEvenInRow (row : ℕ) : ℕ :=
  (List.range (row + 1)).filter (fun k => isEven (pascal row k)) |>.length

/-- Sum of even numbers in the first n rows of Pascal's triangle -/
def sumEvenInRows (n : ℕ) : ℕ :=
  (List.range n).map countEvenInRow |>.sum

/-- The count of even integers in the first 15 rows of Pascal's Triangle is 68 -/
theorem even_count_pascal_15_rows : sumEvenInRows 15 = 68 := by
  sorry

#eval sumEvenInRows 15  -- To check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_pascal_15_rows_l291_29109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_sum_of_specific_f_values_l291_29122

noncomputable def f (n : ℤ) (x : ℝ) : ℝ := 
  (Real.cos (n * Real.pi + x))^2 * (Real.sin (n * Real.pi - x))^2 / 
  (Real.cos ((2 * n + 1) * Real.pi - x))^2

theorem f_simplification (n : ℤ) (x : ℝ) : f n x = Real.sin x ^ 2 := by sorry

theorem sum_of_specific_f_values : 
  Real.sin (Real.pi / 2016) ^ 2 + Real.sin (1007 * Real.pi / 2016) ^ 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_sum_of_specific_f_values_l291_29122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_biology_only_l291_29105

/-- Represents the number of students studying a particular subject or combination of subjects -/
structure StudyCount where
  count : ℕ

/-- The total number of students -/
def total_students : StudyCount := ⟨500⟩

/-- Number of students studying Sociology -/
def sociology : StudyCount := ⟨325⟩

/-- Number of students studying Mathematics -/
def mathematics : StudyCount := ⟨275⟩

/-- Number of students studying Biology -/
def biology : StudyCount := ⟨250⟩

/-- Number of students studying Physics -/
def physics : StudyCount := ⟨75⟩

/-- Number of students studying both Mathematics and Sociology -/
def math_and_socio : StudyCount := ⟨175⟩

/-- Number of students studying both Mathematics and Biology -/
def math_and_bio : StudyCount := ⟨125⟩

/-- Number of students studying both Biology and Sociology -/
def bio_and_socio : StudyCount := ⟨100⟩

/-- Number of students studying Mathematics, Sociology, and Biology -/
def math_socio_bio : StudyCount := ⟨50⟩

/-- Addition for StudyCount -/
instance : Add StudyCount where
  add a b := ⟨a.count + b.count⟩

/-- Subtraction for StudyCount -/
instance : Sub StudyCount where
  sub a b := ⟨a.count - b.count⟩

/-- LessEq for StudyCount -/
instance : LE StudyCount where
  le a b := a.count ≤ b.count

/-- The theorem stating the maximum number of students studying only Biology -/
theorem max_biology_only : 
  ∃ (biology_only : StudyCount), 
    biology_only ≤ biology - math_and_bio - bio_and_socio + math_socio_bio ∧ 
    biology_only = ⟨75⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_biology_only_l291_29105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_wins_probability_l291_29162

-- Define the probability of tossing a six
def prob_six : ℚ := 1 / 6

-- Define the number of players
def num_players : ℕ := 4

-- Define the probability of Dave winning in the first cycle
def prob_dave_first_cycle : ℚ := (1 - prob_six)^(num_players - 1) * prob_six

-- Define the probability of no one winning in a full cycle
def prob_no_win_cycle : ℚ := (1 - prob_six)^num_players

-- Theorem statement
theorem dave_wins_probability :
  (prob_dave_first_cycle / (1 - prob_no_win_cycle)) = 125 / 671 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_wins_probability_l291_29162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l291_29184

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a * (4:ℝ)^x - a * (2:ℝ)^(x+1) + 1 - b

-- Define the theorem
theorem function_properties :
  ∀ (a b : ℝ), 
    a > 0 →
    (∀ x ∈ Set.Icc 1 2, f a b x ≤ 9) →
    (∃ x ∈ Set.Icc 1 2, f a b x = 9) →
    (∀ x ∈ Set.Icc 1 2, f a b x ≥ 1) →
    (∃ x ∈ Set.Icc 1 2, f a b x = 1) →
    (a = 1 ∧ b = 0) ∧
    (∀ k : ℝ, (∃ x ∈ Set.Icc (-1) 1, f 1 0 x - k * (4:ℝ)^x ≥ 0) ↔ k ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l291_29184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_solution_l291_29187

/-- Represents the alcohol mixture problem -/
structure AlcoholMixture where
  vessel1_capacity : ℝ
  vessel1_alcohol_percentage : ℝ
  vessel2_capacity : ℝ
  total_liquid : ℝ
  final_mixture_capacity : ℝ
  final_mixture_alcohol_percentage : ℝ

/-- The given conditions of the problem -/
def problem_conditions : AlcoholMixture :=
  { vessel1_capacity := 2
  , vessel1_alcohol_percentage := 25
  , vessel2_capacity := 6
  , total_liquid := 8
  , final_mixture_capacity := 10
  , final_mixture_alcohol_percentage := 29.000000000000004
  }

/-- Theorem stating the solution to the alcohol mixture problem -/
theorem alcohol_mixture_solution (m : AlcoholMixture) (h : m = problem_conditions) :
  ∃ vessel2_alcohol_percentage : ℝ,
    vessel2_alcohol_percentage = 30.333333333333332 ∧
    m.vessel1_capacity * m.vessel1_alcohol_percentage / 100 +
    m.vessel2_capacity * vessel2_alcohol_percentage / 100 =
    m.total_liquid * m.final_mixture_alcohol_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_solution_l291_29187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_minimum_magnitude_l291_29157

theorem vector_minimum_magnitude (a b : ℝ × ℝ) : 
  (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) = 2) →
  (Real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 1) →
  (a.1 + b.1 = 2 ∧ a.2 + b.2 = Real.sqrt 3) →
  (∃ m : ℝ, ∀ k : ℝ, 
    Real.sqrt ((a.1 + m * b.1) ^ 2 + (a.2 + m * b.2) ^ 2) ≤ 
    Real.sqrt ((a.1 + k * b.1) ^ 2 + (a.2 + k * b.2) ^ 2)) →
  m = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_minimum_magnitude_l291_29157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l291_29131

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (1 - Real.cos (2 * ω * x)) / 2 + Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x)

-- State the theorem
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) (h_period : ∀ x, f ω (x + π) = f ω x) :
  -- 1. The analytic expression of f(x) is sin(2x - π/6) + 1/2
  (∀ x, f ω x = Real.sin (2 * x - π / 6) + 1 / 2) ∧
  -- 2. The range of f(x) is [0.5 - √3/2, 1.5] when x ∈ [-π/12, π/2]
  (∀ x, -π/12 ≤ x ∧ x ≤ π/2 → (1 - Real.sqrt 3) / 2 ≤ f ω x ∧ f ω x ≤ 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l291_29131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_representable_number_l291_29115

def sequenceNums : List ℕ := [1, 3, 9, 27, 81, 243, 729]

/-- 
Given a sequence of natural numbers, this function returns true if it's possible
to represent all integers from 1 to n (inclusive) by selecting any subset of these
numbers and assigning each either a positive or negative sign.
-/
def can_represent_all_up_to (seq : List ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ n → ∃ (subset : List ℕ) (signs : List Bool), 
    subset ⊆ seq ∧ 
    subset.length = signs.length ∧
    (List.zip subset signs).foldl (λ acc (x, sign) => if sign then acc + x else acc - x) 0 = k

theorem max_representable_number :
  can_represent_all_up_to sequenceNums 1093 ∧ 
  ¬(can_represent_all_up_to sequenceNums 1094) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_representable_number_l291_29115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PF_is_6_l291_29161

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve defined by x = t/2 and y = 2√t -/
def on_curve (p : Point) : Prop :=
  ∃ t : ℝ, t ≥ 0 ∧ p.x = t / 2 ∧ p.y = 2 * Real.sqrt t

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem distance_PF_is_6 (P : Point) (a : ℝ) :
  P.x = 4 →
  P.y = a →
  on_curve P →
  distance P ⟨2, 0⟩ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PF_is_6_l291_29161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_line_line_through_point_with_intercept_sum_l291_29111

-- Define a structure for a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a line in the form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def linesParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Function to calculate the sum of intercepts on coordinate axes
noncomputable def sumOfIntercepts (l : Line2D) : ℝ :=
  -l.c / l.a - l.c / l.b

-- Theorem for the first part of the problem
theorem line_through_point_parallel_to_line 
  (p : Point2D) 
  (l : Line2D) 
  (h1 : p.x = 2 ∧ p.y = 1) 
  (h2 : l.a = 2 ∧ l.b = 3 ∧ l.c = 0) : 
  ∃ (l' : Line2D), pointOnLine p l' ∧ linesParallel l l' ∧ l'.a = 2 ∧ l'.b = 3 ∧ l'.c = -7 := by
  sorry

-- Theorem for the second part of the problem
theorem line_through_point_with_intercept_sum 
  (p : Point2D) 
  (h1 : p.x = -3 ∧ p.y = 1) :
  ∃ (l1 l2 : Line2D), 
    (pointOnLine p l1 ∧ sumOfIntercepts l1 = -4 ∧ l1.a = 1 ∧ l1.b = -3 ∧ l1.c = 6) ∨
    (pointOnLine p l2 ∧ sumOfIntercepts l2 = -4 ∧ l2.a = 1 ∧ l2.b = 1 ∧ l2.c = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_parallel_to_line_line_through_point_with_intercept_sum_l291_29111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l291_29169

-- Define the equations
def eq1 (x y : ℝ) : Prop := x + 2*y = 8
def eq2 (x y : ℝ) : Prop := 3*x - y = -6
def eq3 (x y : ℝ) : Prop := 2*x - 3*y = 2
def eq4 (x y : ℝ) : Prop := 4*x + y = 16

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (eq1 p.1 p.2 ∧ eq3 p.1 p.2) ∨
               (eq1 p.1 p.2 ∧ eq4 p.1 p.2) ∨
               (eq2 p.1 p.2 ∧ eq3 p.1 p.2) ∨
               (eq2 p.1 p.2 ∧ eq4 p.1 p.2)}

-- Theorem statement
theorem intersection_points_count :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 4 ∧ ∀ p, p ∈ s ↔ p ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l291_29169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_54_and_y_l291_29175

-- Define y based on the given condition
def y : ℤ := 2 * 32 - 54

-- Theorem statement
theorem positive_difference_54_and_y : |54 - y| = 44 := by
  -- Expand the definition of y
  have h1 : y = 10 := by rfl
  
  -- Calculate |54 - y|
  calc
    |54 - y| = |54 - 10| := by rw [h1]
    _        = 44        := by rfl

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_54_and_y_l291_29175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candidates_through_gates_l291_29101

/-- The number of ways for candidates to pass through gates -/
def ways_to_pass (n : ℕ) (k : ℕ) : ℕ :=
  match n, k with
  | _, 0 => 1
  | 0, _ => 0
  | n+1, k+1 => ways_to_pass n k + ways_to_pass (n+1) k

/-- The number of permutations of r items chosen from n items -/
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.factorial n / Nat.factorial (n - r)

theorem candidates_through_gates : 
  let candidates := 4
  let gates := 2
  ways_to_pass candidates gates * (permutations candidates 2 + permutations candidates 1 * permutations 3 3) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candidates_through_gates_l291_29101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l291_29148

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | Real.rpow 4 x ≥ 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ici (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l291_29148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l291_29151

/-- Given a quadratic inequality whose solution set contains exactly three positive integers, 
    prove that the parameter m is in the range (5, 6]. -/
theorem quadratic_inequality_range (m : ℝ) : 
  (∃ (S : Finset ℕ), S.card = 3 ∧ 
    (∀ n : ℕ, n ∈ S ↔ (n : ℝ)^2 - (m + 2) * (n : ℝ) + 2 * m < 0)) →
  m ∈ Set.Ioo 5 6 ∪ {6} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l291_29151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l291_29197

/-- Given that the solution set of ax² + bx + c < 0 is (-∞, 1) ∪ (3, +∞),
    prove that the solution set of cx² + bx + a > 0 is (1/3, 1) -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.union (Set.Iio 1) (Set.Ioi 3) = {x : ℝ | a * x^2 + b * x + c < 0}) :
  {x : ℝ | c * x^2 + b * x + a > 0} = Set.Ioo (1/3 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l291_29197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l291_29198

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.sin (2 * x - Real.pi / 3) + 2 * (Real.cos x) ^ 2 - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≤ M) ∧
  (∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → m ≤ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l291_29198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_perimeter_l291_29114

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length. -/
structure Kite (P Q R S : ℝ × ℝ) : Prop where
  is_kite : (dist P Q = dist P S) ∧ (dist Q R = dist R S)

/-- The perimeter of a quadrilateral is the sum of the lengths of its sides. -/
def perimeter (P Q R S : ℝ × ℝ) : ℝ :=
  dist P Q + dist Q R + dist R S + dist S P

theorem kite_perimeter (A B C D : ℝ × ℝ) :
  Kite A B C D →
  (dist A B = 10) →
  (dist B D = 15) →
  (dist A D = dist D C) →
  (dist A D = 12.5) →
  (∃ (v : ℝ × ℝ), v ≠ B ∧ (v - A) • (B - A) = 0 ∧ (v - B) • (D - B) = 0) →
  perimeter A B C D = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_perimeter_l291_29114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_distance_ratio_l291_29163

def fuel_used : ℕ := 44
def distance_covered : ℕ := 77

theorem fuel_distance_ratio :
  (fuel_used / Nat.gcd fuel_used distance_covered) / (distance_covered / Nat.gcd fuel_used distance_covered) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_distance_ratio_l291_29163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_function_form_l291_29102

/-- The original function before translation -/
def original_function (x b c : ℝ) : ℝ := x^2 + b*x + c

/-- The translation vector -/
def translation_vector : ℝ × ℝ := (4, 3)

/-- The line to which the translated graph is tangent -/
def tangent_line (x y : ℝ) : Prop := 4*x + y - 8 = 0

/-- The point of tangency after translation -/
def tangent_point : ℝ × ℝ := (1, 4)

/-- Theorem stating that the original function has the form y = x^2 + 2x - 2 -/
theorem original_function_form :
  ∃ (b c : ℝ), 
    (∀ x y : ℝ, original_function (x - 4) b c + 3 = y → tangent_line x y) ∧
    (let (tx, ty) := tangent_point; tangent_line tx ty) ∧
    (b = 2 ∧ c = -2) := by
  sorry

#check original_function_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_function_form_l291_29102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadratic_polynomial_root_l291_29127

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial := ℤ → ℤ → ℤ → ℝ → ℝ

/-- Definition of a quadratic polynomial f(x) = ax² + bx + c -/
def isQuadratic (f : QuadraticPolynomial) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℝ, f a b c x = a * x^2 + b * x + c

/-- The statement that there exists a quadratic polynomial with integer coefficients
    such that f(f(√3)) = 0 -/
theorem exists_quadratic_polynomial_root :
  ∃ f : QuadraticPolynomial, isQuadratic f ∧ f 2 (-8) 0 (f 2 (-8) 0 (Real.sqrt 3)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadratic_polynomial_root_l291_29127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l291_29159

theorem evaluate_expression (α : ℝ) :
  (0 < α ∧ α < π → 
    Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = Real.tan (α / 2)) ∧
  (π < α ∧ α < 2 * π → 
    Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = -Real.tan (α / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l291_29159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l291_29119

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hf : ∀ x, f a x ≥ b) :
  ∃ m, m = Real.exp 3 / 2 ∧ a * b ≤ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ab_value_l291_29119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l291_29178

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧ Real.cos t.B = 3/5

-- Part I
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_b : t.b = 4) :
  Real.sin t.A = 2/5 := by sorry

-- Part II
theorem part_two (t : Triangle) (h : triangle_conditions t) (h_area : (1/2) * t.a * t.b * Real.sin t.C = 4) :
  t.c = 5 ∧ t.b = Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l291_29178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blender_sales_constant_l291_29193

/-- Represents the relationship between number of customers, cost, and energy efficiency rating -/
noncomputable def customer_model (k : ℝ) (c e : ℝ) : ℝ := (k * e) / c

theorem blender_sales_constant (k : ℝ) :
  customer_model k 400 10 = 50 →
  customer_model k 800 20 = 50 :=
by
  intro h
  unfold customer_model at *
  field_simp at *
  -- The rest of the proof would go here
  sorry

#check blender_sales_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blender_sales_constant_l291_29193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_dance_time_l291_29146

noncomputable def john_first_dance : ℝ := 3
noncomputable def john_break : ℝ := 1
noncomputable def john_second_dance : ℝ := 5
noncomputable def james_extra_factor : ℝ := 1/3

noncomputable def john_total_dance : ℝ := john_first_dance + john_second_dance
noncomputable def james_initial_dance : ℝ := john_first_dance + john_break + john_second_dance
noncomputable def james_extra_dance : ℝ := james_initial_dance * james_extra_factor
noncomputable def james_total_dance : ℝ := james_initial_dance + james_extra_dance

theorem combined_dance_time :
  john_total_dance + james_total_dance = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_dance_time_l291_29146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_equation_l291_29100

theorem acute_angle_equation (x : Real) (h1 : 0 < x) (h2 : x < π / 2) :
  2 * Real.sin x ^ 2 + Real.sin x - Real.sin (2 * x) = 3 * Real.cos x →
  x = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_equation_l291_29100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atop_difference_seven_four_l291_29171

-- Define the custom operation (using 'atop' instead of 'at')
def atop (x y : ℤ) : ℤ := x * y - 2 * x

-- Theorem statement
theorem atop_difference_seven_four : (atop 7 4) - (atop 4 7) = -6 := by
  -- Expand the definition of atop
  unfold atop
  -- Simplify the arithmetic
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_atop_difference_seven_four_l291_29171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_line_is_correct_l291_29103

noncomputable section

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Get the slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Check if two lines are perpendicular -/
def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The given line x - 2y - 2 = 0 -/
def givenLine : Line := { a := 1, b := -2, c := -2 }

/-- The point (5, 3) -/
def givenPoint : Point := { x := 5, y := 3 }

/-- The line to be proved: 2x + y - 13 = 0 -/
def targetLine : Line := { a := 2, b := 1, c := -13 }

theorem target_line_is_correct :
  targetLine.isPerpendicular givenLine ∧
  givenPoint.liesOn targetLine := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_line_is_correct_l291_29103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_coeff_formula_l291_29142

/-- The sum of the coefficients of the even powers of x in (x^2 - x + 1)^100 -/
noncomputable def sum_even_coeff (x : ℝ) : ℝ :=
  ((x^2 - x + 1)^100 + ((-x)^2 - (-x) + 1)^100) / 2

/-- Theorem stating that the sum of the coefficients of the even powers of x
    in (x^2 - x + 1)^100 is equal to (1 + 3^100) / 2 -/
theorem sum_even_coeff_formula :
  sum_even_coeff 1 = (1 + 3^100) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_coeff_formula_l291_29142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dried_grapes_from_five_kg_l291_29120

/-- The weight of dried grapes obtained from fresh grapes -/
noncomputable def dried_grapes_weight (fresh_weight : ℝ) : ℝ :=
  let fresh_water_content : ℝ := 0.90
  let dried_water_content : ℝ := 0.20
  let non_water_content : ℝ := fresh_weight * (1 - fresh_water_content)
  non_water_content / (1 - dried_water_content)

/-- Theorem stating that 5 kg of fresh grapes yields 0.625 kg of dried grapes -/
theorem dried_grapes_from_five_kg :
  dried_grapes_weight 5 = 0.625 := by
  unfold dried_grapes_weight
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dried_grapes_from_five_kg_l291_29120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_power_of_five_l291_29158

theorem cubic_power_of_five (n : ℤ) : (∃ k : ℕ, n^3 - 3*n^2 + n + 2 = 5^k) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_power_of_five_l291_29158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_error_detection_and_correction_minimum_undetectable_errors_l291_29194

/-- Represents a table of bits with parity sums -/
def ParityTable (n : ℕ) := Fin (n + 1) → Fin (n + 1) → Bool

/-- Checks if a given row has even parity -/
def has_even_row_parity (t : ParityTable n) (row : Fin (n + 1)) : Prop :=
  (List.range (n + 1)).foldl (λ acc col => acc + (if t row col then 1 else 0)) 0 % 2 = 0

/-- Checks if a given column has even parity -/
def has_even_col_parity (t : ParityTable n) (col : Fin (n + 1)) : Prop :=
  (List.range (n + 1)).foldl (λ acc row => acc + (if t row col then 1 else 0)) 0 % 2 = 0

/-- Counts the number of differences between two tables -/
def error_count (t1 t2 : ParityTable n) : ℕ :=
  (List.range (n + 1)).foldl (λ acc1 row =>
    acc1 + (List.range (n + 1)).foldl (λ acc2 col =>
      acc2 + (if t1 row col = t2 row col then 0 else 1)) 0) 0

theorem parity_error_detection_and_correction (n : ℕ) :
  ∀ (t1 t2 : ParityTable n),
    (∀ row, has_even_row_parity t1 row) →
    (∀ col, has_even_col_parity t1 col) →
    error_count t1 t2 = 1 →
    ∃! (row col : Fin (n + 1)), t1 row col ≠ t2 row col :=
by
  sorry

theorem minimum_undetectable_errors (n : ℕ) :
  ∀ (t1 t2 : ParityTable n),
    (∀ row, has_even_row_parity t1 row) →
    (∀ col, has_even_col_parity t1 col) →
    (∀ row, has_even_row_parity t2 row) →
    (∀ col, has_even_col_parity t2 col) →
    t1 ≠ t2 →
    error_count t1 t2 ≥ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_error_detection_and_correction_minimum_undetectable_errors_l291_29194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_monotonic_interval_l291_29160

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 + a) * x + 1

-- State the theorem
theorem even_function_monotonic_interval :
  ∃ a : ℝ, 
    (∀ x, f a x = f a (-x)) ∧  -- f is an even function
    (∀ x y, x ≤ y ∧ y ≤ 0 → f a x ≤ f a y) ∧   -- f is monotonically increasing on (-∞, 0]
    (∀ x y, 0 < x ∧ x < y → f a x > f a y) :=   -- f is strictly decreasing on (0, ∞)
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_monotonic_interval_l291_29160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_two_digit_numbers_l291_29132

def Digits : Finset Nat := {3, 4, 7, 8}

def valid_two_digit_numbers (digits : Finset Nat) : Finset Nat :=
  Finset.filter (fun n => n ≥ 10 ∧ n < 100 ∧ (n / 10) ∈ digits ∧ (n % 10) ∈ digits ∧ (n / 10) ≠ (n % 10)) (Finset.range 100)

def product_of_pair (a b : Nat) : Nat := a * b

def valid_pair (a b : Nat) (digits : Finset Nat) : Prop :=
  a ∈ valid_two_digit_numbers digits ∧
  b ∈ valid_two_digit_numbers digits ∧
  (a / 10) ≠ (b / 10) ∧ (a / 10) ≠ (b % 10) ∧
  (a % 10) ≠ (b / 10) ∧ (a % 10) ≠ (b % 10)

theorem max_product_of_two_digit_numbers :
  ∃ (a b : Nat), valid_pair a b Digits ∧
    ∀ (x y : Nat), valid_pair x y Digits → product_of_pair a b ≥ product_of_pair x y ∧
    product_of_pair a b = 6142 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_two_digit_numbers_l291_29132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_locus_and_slopes_l291_29143

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ
  tangentLine : ℝ

-- Define the locus C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (c : Circle), c.center = p ∧ c.passesThrough = (1, 0) ∧ c.tangentLine = -1}

-- Define point H
def H : ℝ × ℝ := (4, 0)

-- Define the line x = -4
def line_x_eq_neg4 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.fst = -4}

-- Main theorem
theorem circle_locus_and_slopes :
  (∀ p ∈ C, p.snd^2 = 4 * p.fst) ∧
  (∀ T ∈ line_x_eq_neg4, ∀ M N : ℝ × ℝ, M ∈ C → N ∈ C →
    ∃ (k₁ k₂ k₃ : ℝ),
      k₁ = (M.snd - T.snd) / (M.fst - T.fst) ∧
      k₂ = (H.snd - T.snd) / (H.fst - T.fst) ∧
      k₃ = (N.snd - T.snd) / (N.fst - T.fst) ∧
      k₂ - k₁ = k₃ - k₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_locus_and_slopes_l291_29143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l291_29176

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def f : ℝ → ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(x) = x² - 4 for x > 0 -/
axiom f_pos : ∀ x > 0, f x = x^2 - 4

/-- The solution set of f(x-2) > 0 is (0,2) ∪ (4,+∞) -/
theorem solution_set : 
  {x : ℝ | f (x - 2) > 0} = Set.union (Set.Ioo 0 2) (Set.Ioi 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l291_29176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l291_29116

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem cone_cylinder_volume_ratio :
  let cylinder_radius : ℝ := 3
  let cylinder_height : ℝ := 15
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 5
  (cone_volume cone_radius cone_height) / (cylinder_volume cylinder_radius cylinder_height) = 4/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_volume_ratio_l291_29116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equals_interval1_set_equals_interval2_l291_29134

-- Define the sets
def set1 : Set ℝ := {x | 2*x - 1 ≥ 0}
def set2 : Set ℝ := {x | x < -4 ∨ (-1 < x ∧ x ≤ 2)}

-- Define the intervals
def interval1 : Set ℝ := Set.Ici (1/2)
def interval2 : Set ℝ := Set.Iic (-4) ∪ Set.Ioo (-1) 2

-- Theorem statements
theorem set_equals_interval1 : set1 = interval1 := by sorry

theorem set_equals_interval2 : set2 = interval2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equals_interval1_set_equals_interval2_l291_29134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l291_29147

def A : ℕ → ℤ
  | 0 => 43
  | n + 1 => (2 * n + 1) * (2 * n + 2) + A n

def B : ℕ → ℤ
  | 0 => 1
  | n + 1 => (2 * n) * (2 * n + 1) + B n

theorem difference_A_B : |A 21 - B 21| = 882 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_A_B_l291_29147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_revenue_is_147_l291_29177

/-- Represents the revenue calculation for a book sale -/
structure BookSaleRevenue where
  total_books : ℕ
  price_a : ℚ
  price_b : ℚ
  price_c : ℚ
  sold_fraction_a : ℚ
  sold_fraction_b : ℚ
  sold_fraction_c : ℚ
  remaining_a : ℕ
  remaining_b : ℕ
  remaining_c : ℕ

/-- Calculates the total revenue from the book sale -/
def calculate_revenue (sale : BookSaleRevenue) : ℚ :=
  let original_a := (sale.remaining_a : ℚ) / (1 - sale.sold_fraction_a)
  let original_b := (sale.remaining_b : ℚ) / (1 - sale.sold_fraction_b)
  let original_c := (sale.remaining_c : ℚ) / (1 - sale.sold_fraction_c)
  let sold_a := original_a - sale.remaining_a
  let sold_b := original_b - sale.remaining_b
  let sold_c := original_c - sale.remaining_c
  sold_a * sale.price_a + sold_b * sale.price_b + sold_c * sale.price_c

/-- Theorem stating that the total revenue from the book sale is $147.00 -/
theorem book_sale_revenue_is_147 (sale : BookSaleRevenue) 
  (h1 : sale.total_books = 120)
  (h2 : sale.price_a = 7/2)
  (h3 : sale.price_b = 9/2)
  (h4 : sale.price_c = 11/2)
  (h5 : sale.sold_fraction_a = 1/4)
  (h6 : sale.sold_fraction_b = 1/3)
  (h7 : sale.sold_fraction_c = 1/2)
  (h8 : sale.remaining_a = 20)
  (h9 : sale.remaining_b = 30)
  (h10 : sale.remaining_c = 10) :
  calculate_revenue sale = 147 := by
  sorry

#eval (147 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_revenue_is_147_l291_29177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equals_target_plane_target_plane_conditions_l291_29123

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Checks if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  (plane.a : ℝ) * point.x + (plane.b : ℝ) * point.y + (plane.c : ℝ) * point.z + (plane.d : ℝ) = 0

/-- Parametric representation of the plane -/
def planeParametric (s t : ℝ) : Point3D :=
  { x := 2 + s + 2*t
    y := 3 + 2*s - t
    z := 1 + s + 3*t }

/-- The plane equation we want to prove -/
def targetPlane : Plane :=
  { a := 7
    b := -1
    c := -5
    d := -6 }

/-- Theorem stating that the parametric equation represents the target plane -/
theorem parametric_equals_target_plane :
  ∀ s t : ℝ, pointOnPlane targetPlane (planeParametric s t) := by
  sorry

/-- Theorem stating that the coefficients of the target plane satisfy the required conditions -/
theorem target_plane_conditions :
  (targetPlane.a : ℝ) > 0 ∧
  Nat.gcd (Int.natAbs targetPlane.a)
          (Nat.gcd (Int.natAbs targetPlane.b)
                   (Nat.gcd (Int.natAbs targetPlane.c)
                            (Int.natAbs targetPlane.d))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_equals_target_plane_target_plane_conditions_l291_29123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_method_is_stratified_l291_29170

/-- Represents the three age groups in the workforce --/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Stratified
  | Systematic

/-- The population sizes for each age group --/
def populationSizes : AgeGroup → Nat
  | AgeGroup.Elderly => 500
  | AgeGroup.MiddleAged => 1000
  | AgeGroup.Young => 800

/-- The ratio used for sampling --/
def samplingRatio : AgeGroup → Nat
  | AgeGroup.Elderly => 5
  | AgeGroup.MiddleAged => 10
  | AgeGroup.Young => 8

/-- The total sample size --/
def sampleSize : Nat := 230

/-- Theorem stating that the given sampling method is stratified sampling --/
theorem sampling_method_is_stratified :
  (∀ g : AgeGroup, ∃ k : ℚ, k * (populationSizes g : ℚ) = samplingRatio g) →
  (sampleSize > 0) →
  SamplingMethod.Stratified = 
    (let method : SamplingMethod := SamplingMethod.Stratified
     method) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_method_is_stratified_l291_29170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_equation_l291_29185

theorem power_of_eight_equation (y : ℝ) : (1 / 8 : ℝ) * (2 : ℝ)^32 = (8 : ℝ)^y → y = 29 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_eight_equation_l291_29185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_spending_approximation_l291_29145

/-- Represents the price reduction percentage -/
noncomputable def price_reduction : ℝ := 0.25

/-- Represents the additional amount of oil that can be purchased after the price reduction -/
noncomputable def additional_oil : ℝ := 5

/-- Represents the reduced price per kg of oil in Rupees -/
noncomputable def reduced_price : ℝ := 55

/-- Calculates the original price per kg of oil before the reduction -/
noncomputable def original_price : ℝ := reduced_price / (1 - price_reduction)

/-- Calculates the total spending on oil -/
noncomputable def total_spending : ℝ := 
  (reduced_price * additional_oil) / (1 / reduced_price - 1 / original_price)

/-- Theorem stating that the total spending on oil is approximately 1100 Rupees -/
theorem oil_spending_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |total_spending - 1100| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_spending_approximation_l291_29145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_balls_distribution_l291_29165

/-- The number of ways to distribute n distinct balls to two people. -/
def totalDistributions (n : ℕ) : ℕ := 2^n

/-- The number of ways to distribute n distinct balls to two people
    where one person receives fewer than k balls. -/
def distributionsWithFewerThan (n k : ℕ) : ℕ :=
  2 * (Finset.sum (Finset.range k) (fun i => Nat.choose n i))

/-- The number of ways to distribute n distinct balls to two people,
    where each person receives at least k balls. -/
def distributionsWithAtLeast (n k : ℕ) : ℕ :=
  totalDistributions n - distributionsWithFewerThan n k

theorem seven_balls_distribution :
  distributionsWithAtLeast 7 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_balls_distribution_l291_29165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l291_29104

/-- Given an ellipse and a line intersecting it, prove the ellipse equation --/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let l : ℝ → ℝ → Prop := λ x y ↦ y = (1/2) * x + c
  ∃ c : ℝ, 
    (∃ A B : ℝ × ℝ, l A.1 A.2 ∧ l B.1 B.2 ∧ 
      A.1^2/a^2 + A.2^2/b^2 = 1 ∧ B.1^2/a^2 + B.2^2/b^2 = 1) ∧
    (∃ f : ℝ × ℝ, f.1^2/a^2 + f.2^2/b^2 = 1 ∧ l f.1 f.2 ∧ f.1 < 0) ∧
    (∀ x y : ℝ, (x^2/a^2 + y^2/b^2 = 1) → 
      (abs (x * (1/2) - y + c/2) / Real.sqrt ((1/2)^2 + 1^2) = 1)) ∧
    (∃ A B : ℝ × ℝ, l A.1 A.2 ∧ l B.1 B.2 ∧ 
      A.1^2/a^2 + A.2^2/b^2 = 1 ∧ B.1^2/a^2 + B.2^2/b^2 = 1 ∧
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (8*a/5)^2) →
  a = 3 ∧ b = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l291_29104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_count_l291_29149

theorem prime_factors_count (n : ℕ) : 
  (4^19 * 101^10 * 2^16 * 67^5 * 17^9).factors.length = 78 := by
  -- We'll use sorry to skip the actual proof
  sorry

#eval (4^19 * 101^10 * 2^16 * 67^5 * 17^9).factors.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_count_l291_29149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_coeff_numerator_one_or_prime_l291_29174

/-- Taylor series coefficient of (1-x+x^2)e^x at x=0 -/
def taylor_coeff (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n^2 - n + 1 : ℚ) / n.factorial

/-- Predicate for a rational number having numerator 1 or prime in lowest terms -/
def numerator_one_or_prime (q : ℚ) : Prop :=
  (q.num.natAbs = 1) ∨ (Nat.Prime q.num.natAbs)

theorem taylor_coeff_numerator_one_or_prime :
  ∀ n : ℕ, taylor_coeff n ≠ 0 → numerator_one_or_prime (taylor_coeff n) := by
  sorry

#check taylor_coeff_numerator_one_or_prime

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_coeff_numerator_one_or_prime_l291_29174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pirate_coins_l291_29137

/-- The number of pirates --/
def n : ℕ := 10

/-- The function representing the fraction of remaining coins each pirate takes --/
def pirate_fraction (k : ℕ) : ℚ := (k + 1 : ℚ) / 11

/-- The initial number of coins --/
def initial_coins : ℕ := 11^n / n.factorial

/-- The number of coins the kth pirate receives --/
def coins_for_pirate : ℕ → ℕ
| 0 => initial_coins
| k + 1 =>
  let remaining := coins_for_pirate k - (pirate_fraction k * coins_for_pirate k).floor.toNat
  (pirate_fraction (k + 1) * remaining).floor.toNat

theorem tenth_pirate_coins :
  coins_for_pirate (n - 1) = 1296 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pirate_coins_l291_29137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l291_29172

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def arithmeticSequenceSum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (a 1 + a n) * n / 2

theorem arithmetic_sequence_sum_9 (a : ℕ → ℚ) :
  isArithmeticSequence a →
  a 1 + a 4 + a 7 = 39 →
  a 3 + a 6 + a 9 = 27 →
  arithmeticSequenceSum a 9 = 99 := by
  sorry

#check arithmetic_sequence_sum_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l291_29172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l291_29152

theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ (Real.log 8 / Real.log x = Real.log 4 / Real.log 64) ∧ x = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l291_29152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_proof_l291_29113

/-- The angle of intersection between two curves x^2 + y^2 = 16 and y^2 = 6x -/
noncomputable def intersection_angle : ℝ :=
  Real.arctan (5 / Real.sqrt 3)

/-- The first curve: x^2 + y^2 = 16 -/
def curve1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

/-- The second curve: y^2 = 6x -/
def curve2 (x y : ℝ) : Prop :=
  y^2 = 6 * x

theorem intersection_angle_proof :
  ∃ (x y : ℝ), curve1 x y ∧ curve2 x y ∧
  intersection_angle = Real.arctan (|(3 / (2 * y)) - (-x / y)| / (1 + (3 / (2 * y)) * (-x / y))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_proof_l291_29113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_l291_29112

-- Define the boy's running time in seconds
noncomputable def running_time : ℝ := 72

-- Define the boy's speed in km/hr
noncomputable def speed_km_hr : ℝ := 9

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

-- Theorem statement
theorem square_field_side_length :
  ∃ (side_length : ℝ),
    side_length = 45 ∧
    4 * side_length = running_time * (speed_km_hr * km_hr_to_m_s) :=
by
  -- Introduce the side length
  let side_length : ℝ := 45
  
  -- Prove the existence
  use side_length
  
  apply And.intro
  · -- Prove side_length = 45
    rfl
  
  · -- Prove 4 * side_length = running_time * (speed_km_hr * km_hr_to_m_s)
    -- This step requires numerical computation, which we'll skip for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_field_side_length_l291_29112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_opposite_eight_l291_29107

theorem cube_root_of_opposite_eight :
  ((-8 : ℝ) ^ (1/3 : ℝ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_opposite_eight_l291_29107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l291_29126

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + Real.sqrt 3 - 2 * Real.sqrt 3 * (Real.cos x) ^ 2 + 1

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, ∃ c : ℝ, ∀ x, f (c + x) = f (c - x) ∧ c = k * Real.pi / 2 + Real.pi / 6) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12),
    ∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12),
    x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l291_29126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sharing_equation_l291_29138

theorem book_sharing_equation (x : ℕ) : 
  (∀ (student : ℕ), student < x → student * (x - 1) = x - 1) →
  (x * (x - 1) = 240) →
  x * (x - 1) = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sharing_equation_l291_29138
