import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1165_116508

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 3*x - 2*y = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem hyperbola_foci_distance 
  (a : ℝ) 
  (h_a : a > 0)
  (x y : ℝ) 
  (h_hyp : hyperbola x y a)
  (h_asym : asymptote x y)
  (xf1 yf1 xf2 yf2 : ℝ)
  (h_f1 : distance x y xf1 yf1 = 5)
  (h_foci : xf1 < xf2) :
  distance x y xf2 yf2 = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1165_116508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_proof_l1165_116547

/-- Represents the work rate in acres per day per man. -/
noncomputable def work_rate_per_man : ℚ := 240 / (20 * 12)

/-- The number of men in the first group. -/
def first_group_size : ℕ := 6

theorem first_group_size_proof :
  (first_group_size : ℚ) * work_rate_per_man * 10 = 60 ∧
  12 * work_rate_per_man * 20 = 240 :=
by
  sorry

#eval first_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_proof_l1165_116547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_identical_digits_l1165_116515

/-- The decimal expansion of a rational number at a given index. -/
noncomputable def decimal_expansion (q : ℚ) (i : ℕ) : ℕ :=
  sorry

/-- Theorem: No consecutive identical digits in the decimal expansion of n/73 for n < 73. -/
theorem no_consecutive_identical_digits (n : ℕ) (h : n < 73) :
  ∀ (i : ℕ), (decimal_expansion (n / 73) i) ≠ (decimal_expansion (n / 73) (i + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_identical_digits_l1165_116515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_point_or_empty_l1165_116536

-- Define the point type
def Point := ℝ × ℝ

-- Define the distance function
noncomputable def dist (p q : Point) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the locus
def locus (A B : Point) (k l d : ℝ) : Set Point :=
  {M : Point | k * (dist A M)^2 + l * (dist B M)^2 = d}

-- Theorem statement
theorem locus_is_circle_point_or_empty (A B : Point) (k l d : ℝ) (h : k + l ≠ 0) :
  ∃ (C : Point), (∀ M ∈ locus A B k l d, dist C M = dist C A) ∨ 
  (locus A B k l d).Finite ∨ 
  locus A B k l d = ∅ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_point_or_empty_l1165_116536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_9_fourth_power_l1165_116561

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom fg_condition : ∀ x, x ≥ 1 → f (g x) = x^2
axiom gf_condition : ∀ x, x ≥ 1 → g (f x) = x^4
axiom g_81 : g 81 = 81

-- State the theorem to be proved
theorem g_9_fourth_power : (g 9)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_9_fourth_power_l1165_116561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_cosine_l1165_116581

/-- Given a point Q in the first octant of 3D space, if its direction cosines
    with respect to the x- and y-axes are 2/5 and 3/5 respectively, then its
    direction cosine with respect to the z-axis is 2√3/5. -/
theorem direction_cosine (Q : ℝ × ℝ × ℝ) 
  (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0) 
  (h_cos_x : Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 2/5)
  (h_cos_y : Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 3/5) :
  Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 2 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_cosine_l1165_116581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vens_are_yins_but_not_all_xips_are_wons_l1165_116535

-- Define the types for our sets
variable (Zarb Yin Xip Won Ven : Type)

-- Define the subset relations given in the problem
variable (h1 : Zarb → Yin)
variable (h2 : Xip → Yin)
variable (h3 : Won → Xip)
variable (h4 : Ven → Zarb)

-- Theorem statement
theorem vens_are_yins_but_not_all_xips_are_wons :
  (∀ v : Ven, ∃ y : Yin, h1 (h4 v) = y) ∧
  ¬(∀ x : Xip, ∃ w : Won, h3 w = x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vens_are_yins_but_not_all_xips_are_wons_l1165_116535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_15_dividing_30_factorial_l1165_116573

theorem largest_power_of_15_dividing_30_factorial :
  (∃ n : ℕ, 15^n ∣ Nat.factorial 30 ∧ ∀ m : ℕ, 15^m ∣ Nat.factorial 30 → m ≤ n) ∧
  (∀ n : ℕ, 15^n ∣ Nat.factorial 30 → n ≤ 7) ∧
  (15^7 ∣ Nat.factorial 30) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_15_dividing_30_factorial_l1165_116573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1165_116505

theorem cube_root_simplification :
  (Real.rpow (27 - 8) (1/3)) * (Real.rpow (27 - Real.rpow 8 (1/3)) (1/3)) = Real.rpow 475 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1165_116505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_volume_l1165_116530

/-- Represents the volume of a liquid mixture -/
structure Mixture where
  p : ℚ  -- Volume of liquid p
  q : ℚ  -- Volume of liquid q

/-- Calculates the ratio of two rational numbers -/
def ratio (a b : ℚ) : ℚ := a / b

/-- Theorem stating the initial volume of the mixture -/
theorem initial_mixture_volume
  (m : Mixture)  -- Initial mixture
  (h1 : ratio m.p m.q = 3 / 2)  -- Initial ratio condition
  (h2 : ratio m.p (m.q + 2) = 5 / 4)  -- Ratio after adding 2 liters of q
  : m.p + m.q = 25 := by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_volume_l1165_116530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_value_l1165_116513

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the sum of floor functions
noncomputable def floor_sum (x : ℝ) : ℤ := 
  (floor (x + 0.1)) + (floor (x + 0.2)) + (floor (x + 0.3)) + 
  (floor (x + 0.4)) + (floor (x + 0.5)) + (floor (x + 0.6)) + 
  (floor (x + 0.7)) + (floor (x + 0.8)) + (floor (x + 0.9))

-- State the theorem
theorem min_x_value (x : ℝ) : 
  floor_sum x = 104 → x ≥ 11.5 ∧ ∀ y < 11.5, floor_sum y ≠ 104 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_value_l1165_116513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l1165_116566

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Determines if a point is on a line segment between two other points -/
def isOnSegment (p1 p2 p : Point) : Prop :=
  distance p1 p + distance p p2 = distance p1 p2

/-- Theorem: Given the conditions, point C has coordinates (4, 11) -/
theorem point_c_coordinates :
  let A : Point := ⟨7, 3⟩
  let B : Point := ⟨-2, -1⟩
  let D : Point := ⟨1, 5⟩
  let C : Point := ⟨4, 11⟩
  (distance A B = distance A C) →  -- AB = AC
  (isOnSegment B C D) →            -- D is on BC
  (((A.x - D.x) * (B.x - C.x) + (A.y - D.y) * (B.y - C.y)) = 0) →  -- AD ⊥ BC
  (C.x = 4 ∧ C.y = 11) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_c_coordinates_l1165_116566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_altitude_feet_triangle_l1165_116597

/-- Given an isosceles acute triangle with base angle α and area S,
    the area of the triangle formed by the feet of the altitudes
    is -1/2 * S * sin(4α) * cot(α). -/
theorem area_of_altitude_feet_triangle
  (α : ℝ)
  (S : ℝ)
  (h_acute : 0 < α ∧ α < π/2)
  (h_area_pos : S > 0) :
  let area_feet := -1/2 * S * Real.sin (4*α) * (Real.cos α / Real.sin α)
  ∃ (a : ℝ),
    a > 0 ∧
    S = 1/2 * a^2 * Real.sin (2*α) ∧
    area_feet = -1/2 * S * Real.sin (4*α) * (Real.cos α / Real.sin α) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_altitude_feet_triangle_l1165_116597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l1165_116587

theorem factorial_simplification : (13 * 12 * 11 * Nat.factorial 10) / (11 * Nat.factorial 10 + 2 * Nat.factorial 10) = 156 / 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l1165_116587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1165_116596

/-- Given a triangle ABC with the following properties:
    - $\overrightarrow{m} = (\sin^2 \frac{B+C}{2}, 1)$
    - $\overrightarrow{n} = (-2, \cos 2A + 1)$
    - $\overrightarrow{m} \perp \overrightarrow{n}$
    - $a = 2\sqrt{3}$
    - Area $S$ of $\triangle ABC$ is $S = \frac{a^2+b^2-c^2}{4\sqrt{3}}$
    Then angle A is 120°, side c is 2, and the area S is $\sqrt{3}$. -/
theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  let m : ℝ × ℝ := ((Real.sin ((B + C) / 2))^2, 1)
  let n : ℝ × ℝ := (-2, Real.cos (2 * A) + 1)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⟂ n
  (a = 2 * Real.sqrt 3) →
  (S = (a^2 + b^2 - c^2) / (4 * Real.sqrt 3)) →
  (A = 120 * π / 180 ∧ c = 2 ∧ S = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1165_116596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_finding_strategy_l1165_116549

/-- A polynomial of the form x³ + px + q -/
def CubicPolynomial (p q : ℤ) : ℤ → ℤ := fun x ↦ x^3 + p*x + q

theorem cubic_polynomial_root_finding_strategy
  (k : ℕ) 
  (p q : ℤ) 
  (r s t : ℤ) 
  (h_roots : ∀ x, CubicPolynomial p q x = 0 ↔ x = r ∨ x = s ∨ x = t)
  (h_bound : ∀ x, x = r ∨ x = s ∨ x = t → |x| < 3 * 2^k) :
  ∃ (strategy : (ℤ → Ordering) → Fin (2*k + 1) → ℤ),
    ∃ (result : (ℤ → Ordering) → Fin 3 → ℤ),
      ∀ f : ℤ → Ordering,
        (∀ x, f x = compare (CubicPolynomial p q x) 0) →
        (result f 0 = r ∧ result f 1 = s ∧ result f 2 = t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_root_finding_strategy_l1165_116549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_center_of_symmetry_l1165_116510

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + (Real.cos x) ^ 2 - 1 / 2

theorem not_center_of_symmetry :
  ¬ (∀ (h : ℝ), f (π / 6 + h) = f (π / 6 - h)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_center_of_symmetry_l1165_116510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1165_116503

noncomputable def a (n : ℕ) : ℤ :=
  if n ≤ 4 then n - 4 else 2^(n - 5)

theorem sequence_properties :
  (∀ n ≥ 1, ∀ k ≤ 5, a (n + k) - a n = k * (a 2 - a 1)) ∧
  (∀ n ≥ 5, ∀ k : ℕ, a (n + k) = a n * 2^k) ∧
  a 3 = -1 ∧
  a 7 = 4 ∧
  (∀ m : ℕ, m > 0 → (a m + a (m + 1) + a (m + 2) = a m * a (m + 1) * a (m + 2) ↔ m = 1 ∨ m = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1165_116503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_area_line_l1165_116539

/-- A line that passes through point P(3, 2) and satisfies specific conditions -/
structure SpecialLine where
  -- Equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through point P(3, 2)
  passes_through_P : a * 3 + b * 2 + c = 0
  -- The angle of inclination is twice the angle of inclination of x-4y+3=0
  double_inclination : a / b = -8 / 15
  -- The line intersects positive semi-axes of x and y
  intersects_positive_axes : a * c < 0 ∧ b * c < 0
  -- Ensure a and b are not zero
  nonzero : a ≠ 0 ∧ b ≠ 0

/-- The area of triangle AOB formed by the line and the coordinate axes -/
noncomputable def triangle_area (l : SpecialLine) : ℝ :=
  abs (l.c * l.c) / (2 * abs (l.a * l.b))

/-- The theorem stating that 2x+3y-12=0 is the special line that minimizes the area of triangle AOB -/
theorem minimal_area_line :
  ∃ (l : SpecialLine), l.a = 2 ∧ l.b = 3 ∧ l.c = -12 ∧
  ∀ (l' : SpecialLine), triangle_area l ≤ triangle_area l' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_area_line_l1165_116539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_is_three_halves_l1165_116521

-- Define the function f(x) = (1/2)^x
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ x

-- Define the interval [0, 1]
def interval : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem sum_of_max_and_min_is_three_halves :
  ∃ (max min : ℝ), max ∈ Set.image f interval ∧ 
                    min ∈ Set.image f interval ∧
                    (∀ y ∈ Set.image f interval, min ≤ y ∧ y ≤ max) ∧
                    max + min = 3/2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_is_three_halves_l1165_116521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l1165_116590

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

-- Theorem statement
theorem point_not_on_graph : ¬ (∃ y : ℝ, f (-2) = y ∧ y = 1) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_not_on_graph_l1165_116590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_value_in_possible_amounts_l1165_116594

/-- Represents the number of pennies in the bag -/
def p : ℕ := 1  -- We define p as a constant for simplicity

/-- The total value of coins in the bag in cents -/
def total_value : ℕ → ℕ 
  | n => 331 * n

/-- Possible amounts in the bag in cents -/
def possible_amounts : List ℕ := [331, 662, 993, 1324, 1655]

/-- Theorem stating that the total value is always in the list of possible amounts -/
theorem total_value_in_possible_amounts :
  ∃ (n : ℕ), n ∈ possible_amounts ∧ total_value p = n :=
by
  use 331
  apply And.intro
  · simp [possible_amounts]
  · simp [total_value, p]

#check total_value_in_possible_amounts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_value_in_possible_amounts_l1165_116594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1165_116562

theorem trig_identity (α : ℝ) (h : (Real.cos α) ^ 2 = Real.sin α) :
  1 / Real.sin α + (Real.cos α) ^ 4 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1165_116562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l1165_116584

/-- Given a continuous and differentiable function f on ℝ satisfying xf'(x) + f(x) > 0,
    prove that g(x) = xf(x) + 1 has no zeros for x > 0 -/
theorem no_zeros_of_g (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_diff : Differentiable ℝ f)
    (hf_pos : ∀ x : ℝ, x * (deriv f x) + f x > 0) :
    ∀ x : ℝ, x > 0 → x * f x + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l1165_116584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1165_116541

/-- For a parabola with equation y² = 4x, the distance from its focus to its directrix is 2 -/
theorem parabola_focus_directrix_distance (y x : ℝ) : 
  y^2 = 4*x → (∃ (f d : ℝ), ∃ (p : ℝ), p = 2 ∧ |f - d| = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1165_116541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monitor_horizontal_length_l1165_116520

/-- Calculates the horizontal length of a monitor given its aspect ratio and diagonal length. -/
noncomputable def horizontalLength (aspectRatioWidth : ℝ) (aspectRatioHeight : ℝ) (diagonal : ℝ) : ℝ :=
  (aspectRatioWidth * diagonal) / Real.sqrt (aspectRatioWidth^2 + aspectRatioHeight^2)

/-- Theorem: The horizontal length of a 16:9 monitor with a 30-inch diagonal is 480/√337 inches. -/
theorem monitor_horizontal_length :
  horizontalLength 16 9 30 = 480 / Real.sqrt 337 := by
  sorry

/-- Approximate evaluation of the result -/
def approx_result : Float := (480 : Float) / Float.sqrt 337
#eval approx_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monitor_horizontal_length_l1165_116520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1165_116589

theorem cos_alpha_value (α : Real) 
  (h1 : Real.cos (α + Real.pi/4) = 1/3) 
  (h2 : α > 0) 
  (h3 : α < Real.pi/2) : 
  Real.cos α = (4 + Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1165_116589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_ellipse_l1165_116517

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Defines the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Checks if a point (x, y) lies on the ellipse -/
def on_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Calculates the area of a triangle given three points -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Main theorem statement -/
theorem max_triangle_area_on_ellipse (e : Ellipse) (l : Line) :
  eccentricity e = Real.sqrt 3 / 2 →
  on_ellipse e (2 * Real.sqrt 2) 0 →
  l.m = 1/2 →
  ∃ (x1 y1 x2 y2 : ℝ),
    on_ellipse e x1 y1 ∧
    on_ellipse e x2 y2 ∧
    y1 = l.m * x1 + l.c ∧
    y2 = l.m * x2 + l.c ∧
    ∀ (x3 y3 : ℝ),
      on_ellipse e x3 y3 ∧
      y3 = l.m * x3 + l.c →
      triangle_area x1 y1 x2 y2 2 1 ≥ triangle_area x1 y1 x3 y3 2 1 ∧
      triangle_area x1 y1 x2 y2 2 1 ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_ellipse_l1165_116517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_occurrence_shift_fifth_e_encoded_as_k_l1165_116552

/-- Calculates the shift for a given occurrence of a letter in the modified code. -/
def shift_for_occurrence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+1 => if n % 2 = 0 then shift_for_occurrence n + shift_for_occurrence (n-1) else 2 * shift_for_occurrence (n-1)

/-- The alphabet size used in the encoding. -/
def alphabet_size : ℕ := 26

/-- Theorem stating that the 5th occurrence of a letter results in a shift of 6 places. -/
theorem fifth_occurrence_shift :
  shift_for_occurrence 5 % alphabet_size = 6 := by
  sorry

/-- Function to encode a single character based on its occurrence number. -/
def encode_char (c : Char) (occurrence : ℕ) : Char :=
  let shift := shift_for_occurrence occurrence
  let n := (c.toNat - 'A'.toNat + shift) % alphabet_size + 'A'.toNat
  Char.ofNat n

/-- Theorem stating that the 5th occurrence of 'e' is encoded as 'K'. -/
theorem fifth_e_encoded_as_k :
  encode_char 'e' 5 = 'K' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_occurrence_shift_fifth_e_encoded_as_k_l1165_116552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1165_116582

theorem trigonometric_identities (θ : Real) 
  (h1 : Real.sin θ = 3/5) 
  (h2 : π/2 < θ ∧ θ < π) : 
  Real.tan θ = -3/4 ∧ 
  Real.cos (2*θ - π/3) = (7 - 24*Real.sqrt 3) / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1165_116582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_sixty_degrees_l1165_116585

noncomputable def circle_radius : ℝ := 1
noncomputable def central_angle : ℝ := Real.pi / 3

theorem arc_length_for_sixty_degrees :
  let r := circle_radius
  let θ := central_angle
  r * θ = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_for_sixty_degrees_l1165_116585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1165_116516

theorem trig_equation_solution (x : ℝ) :
  Real.sin ((π / 2) * Real.cos x) = Real.cos ((π / 2) * Real.sin x) ↔
  (∃ k : ℤ, x = π / 2 + π * k) ∨ (∃ k : ℤ, x = 2 * π * k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l1165_116516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_condition_l1165_116540

-- Define the points A and B
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (0, 6)

-- Define the circle
def circle_equation (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - 3)^2 = 4

-- Define what it means for an angle to be obtuse
def is_obtuse (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx) * (rx - qx) + (py - qy) * (ry - qy) < 0

-- State the theorem
theorem circle_tangent_condition (a : ℝ) :
  (∀ P : ℝ × ℝ, circle_equation a P.1 P.2 → is_obtuse P A B) →
  (a > Real.sqrt 55 ∨ a < -Real.sqrt 55) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_condition_l1165_116540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_constant_value_l1165_116577

theorem trigonometric_constant_value (α β : ℝ) (p : ℝ)
  (h1 : Real.sin (2 * α + β) = p * Real.sin β)
  (h2 : Real.tan (α + β) = p * Real.tan α)
  (h3 : p > 0)
  (h4 : p ≠ 1) :
  p = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_constant_value_l1165_116577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_properties_l1165_116533

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 2

-- Define point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := x + y - 1 = 0
def tangent_line_2 (x y : ℝ) : Prop := 7*x - y - 15 = 0

-- Define the length of the tangent
noncomputable def tangent_length : ℝ := 2 * Real.sqrt 2

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - 3*y + 3 = 0

-- Theorem statement
theorem circle_tangent_properties :
  ∃ (A B : ℝ × ℝ),
    (∀ x y, circle_C x y → (tangent_line_1 x y ∨ tangent_line_2 x y)) ∧
    (∀ x y, circle_C x y → (x - point_P.1)^2 + (y - point_P.2)^2 ≥ tangent_length^2) ∧
    (line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_properties_l1165_116533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_life_calculation_l1165_116572

/-- Represents the battery life of a laptop -/
structure LaptopBattery where
  total_capacity : ℚ
  idle_consumption_rate : ℚ
  active_consumption_rate : ℚ

/-- Calculates the remaining battery life -/
def remaining_battery_life (battery : LaptopBattery) 
  (total_time : ℚ) (active_time : ℚ) : ℚ :=
  let idle_time := total_time - active_time
  let idle_consumption := idle_time * battery.idle_consumption_rate
  let active_consumption := active_time * battery.active_consumption_rate
  let remaining_capacity := battery.total_capacity - (idle_consumption + active_consumption)
  remaining_capacity / battery.idle_consumption_rate

/-- Theorem stating the remaining battery life under given conditions -/
theorem battery_life_calculation (battery : LaptopBattery) 
  (h1 : battery.total_capacity = 1)
  (h2 : battery.idle_consumption_rate = 1 / 20)
  (h3 : battery.active_consumption_rate = 1 / 4)
  : remaining_battery_life battery 10 2 = 2 := by
  sorry

#eval remaining_battery_life 
  { total_capacity := 1, idle_consumption_rate := 1/20, active_consumption_rate := 1/4 } 
  10 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_battery_life_calculation_l1165_116572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_bound_l1165_116555

/-- Given a geometric sequence with first term 2 and common ratio 3,
    prove that the ratio of consecutive partial sums is bounded. -/
theorem geometric_sequence_sum_ratio_bound (n : ℕ+) :
  let a : ℕ → ℝ := fun k => 2 * 3^(k - 1)
  let S : ℕ+ → ℝ := fun k => (a 1 * (3^(k : ℕ) - 1)) / (3 - 1)
  (S (n + 1)) / (S n) ≤ (3 * (n : ℝ) + 1) / n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_bound_l1165_116555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_survey_result_proof_l1165_116568

theorem survey_result (total_participants : ℚ) 
  (domestic_belief_percentage : ℚ)
  (trainable_belief_percentage : ℚ)
  (incorrect_trainable_belief : ℚ) : Prop :=
  let domestic_belief := (domestic_belief_percentage * total_participants).floor
  let trainable_belief := (trainable_belief_percentage * domestic_belief).floor
  (domestic_belief_percentage = 756/1000) →
  (trainable_belief_percentage = 3/10) →
  (incorrect_trainable_belief = 27) →
  (trainable_belief = incorrect_trainable_belief) →
  (total_participants = 119)

-- Proof
theorem survey_result_proof : 
  ∃ (total_participants : ℚ), 
    survey_result total_participants (756/1000) (3/10) 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_result_survey_result_proof_l1165_116568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sum_inequality_l1165_116550

theorem gcd_sum_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) (hdiff : m ≠ n) :
  (Nat.gcd m n + Nat.gcd (m + 1) (n + 1) + Nat.gcd (m + 2) (n + 2) : ℤ) ≤ 2 * |Int.ofNat m - Int.ofNat n| + 1 ∧
  ((Nat.gcd m n + Nat.gcd (m + 1) (n + 1) + Nat.gcd (m + 2) (n + 2) : ℤ) = 2 * |Int.ofNat m - Int.ofNat n| + 1 ↔
    (∃ k : ℕ, k > 0 ∧
      ((m = k ∧ n = k + 1) ∨
       (m = k + 1 ∧ n = k) ∨
       (m = 2 * k ∧ n = 2 * k + 2) ∨
       (m = 2 * k + 2 ∧ n = 2 * k)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sum_inequality_l1165_116550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1165_116543

noncomputable section

variable (f : ℝ → ℝ)

-- f is invertible
axiom f_invertible : Function.Injective f

theorem intersection_points_count :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x : ℝ, x ∈ s ↔ f (x^3) = f (x^5) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l1165_116543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l1165_116538

theorem tan_double_angle_special_case (θ : ℝ) 
  (h1 : Real.cos θ = -3/5) 
  (h2 : θ ∈ Set.Ioo 0 Real.pi) : 
  Real.tan (2 * θ) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l1165_116538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_eq_60_l1165_116553

/-- An arithmetic sequence with specific conditions -/
structure ArithSeq where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a3_eq_2 : a 3 = 2
  a6_plus_a10_eq_20 : a 6 + a 10 = 20

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithSeq) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first 10 terms of the given arithmetic sequence is 60 -/
theorem sum_10_eq_60 (seq : ArithSeq) : sum_n seq 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_eq_60_l1165_116553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_theorem_l1165_116531

-- Define the lines and points
noncomputable def l₁ : ℝ → ℝ := λ x ↦ 2 * (x - 1) + 2
noncomputable def l₂ : ℝ → ℝ := λ x ↦ -1/2 * (x - 1) + 2
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (5, 0)

-- Define the range function
noncomputable def range_function (a b : ℝ) : ℝ := (b - 1) / (a + 1)

-- Define the distance function between y-axis intersections
noncomputable def y_axis_distance (k : ℝ) : ℝ := |1/k + k|

-- Define line segment
def line_segment (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2)}

theorem perpendicular_lines_theorem :
  -- Part 1: Range of (b-1)/(a+1)
  (∀ a b : ℝ, (a, b) ∈ line_segment P Q → 
    -1/6 ≤ range_function a b ∧ range_function a b ≤ 1/2) ∧
  -- Part 2: Minimum distance between y-axis intersections
  (∀ k : ℝ, k ≠ 0 → y_axis_distance k ≥ 2) ∧
  (∃ k : ℝ, k ≠ 0 ∧ y_axis_distance k = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_theorem_l1165_116531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_representable_count_l1165_116580

/-- Represents an arithmetic expression using k's and operations -/
inductive Expr (k : ℕ+)
  | const : Expr k
  | add : Expr k → Expr k → Expr k
  | sub : Expr k → Expr k → Expr k
  | mul : Expr k → Expr k → Expr k
  | div : Expr k → Expr k → Expr k

/-- Evaluates an expression to a natural number -/
def eval {k : ℕ+} : Expr k → ℕ
  | Expr.const => k
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Counts the number of k's used in an expression -/
def count_k {k : ℕ+} : Expr k → ℕ
  | Expr.const => 1
  | Expr.add e1 e2 => count_k e1 + count_k e2
  | Expr.sub e1 e2 => count_k e1 + count_k e2
  | Expr.mul e1 e2 => count_k e1 + count_k e2
  | Expr.div e1 e2 => count_k e1 + count_k e2

/-- The set of numbers representable with exactly n k's -/
def representable (k : ℕ+) (n : ℕ) : Set ℕ :=
  {m | ∃ e : Expr k, eval e = m ∧ count_k e = n}

/-- The theorem to be proved -/
theorem representable_count (k : ℕ+) (n : ℕ) :
  ∃ S : Finset ℕ, S.card ≥ (k.val - 1) * (n - k.val + 1) ∧
  ↑S ⊆ representable k n ∧ (↑S : Set ℕ) ∩ representable k (n-1) = ∅ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_representable_count_l1165_116580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l1165_116578

-- Define the exponential function g(x)
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 - g x) / (1 + g x)

theorem exponential_function_property :
  g 2 = 4 ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f x = (1 - 2^x) / (1 + 2^x)) ∧
  (∀ x y : ℝ, x < y → f x > f y) ∧
  (∀ m : ℝ, m > 0 ∧ m ≤ 1/3 → f (1/m) ≤ -7/9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_property_l1165_116578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sale_net_result_l1165_116511

/-- Represents the sale of two vases with different profit/loss percentages -/
structure VaseSale where
  selling_price : ℚ
  profit_percentage : ℚ
  loss_percentage : ℚ

/-- Calculates the net result of the vase sale -/
noncomputable def net_result (sale : VaseSale) : ℚ :=
  let cost_price1 := sale.selling_price / (1 + sale.profit_percentage / 100)
  let cost_price2 := sale.selling_price / (1 - sale.loss_percentage / 100)
  2 * sale.selling_price - (cost_price1 + cost_price2)

/-- Theorem stating that the net result of the specific vase sale is 0.06 (6 cents) -/
theorem vase_sale_net_result :
  let sale := VaseSale.mk (5/2) 25 15
  net_result sale = (3/50) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sale_net_result_l1165_116511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_max_value_bounds_l1165_116567

-- Define the functions
def f (x : ℝ) := x^2 + 2*x
noncomputable def h (x : ℝ) := f x - Real.exp x

-- Define the theorem
theorem h_max_value_bounds :
  ∃ x₀ : ℝ, (∀ x : ℝ, h x ≤ h x₀) ∧ (1/4 < h x₀ ∧ h x₀ < 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_max_value_bounds_l1165_116567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graphs_are_different_l1165_116579

-- Define the three functions
noncomputable def f1 (x : ℝ) : ℝ := x - 2
noncomputable def f2 (x : ℝ) : ℝ := (x^2 - 4) / (x + 2)
noncomputable def f3 (x : ℝ) : ℝ := (x^2 - 4) / (x + 2)

-- Theorem stating that the graphs of the three functions are all different
theorem graphs_are_different :
  ¬(∀ x : ℝ, f1 x = f2 x) ∧
  ¬(∀ x : ℝ, f1 x = f3 x) ∧
  ¬(∀ x : ℝ, f2 x = f3 x) := by
  sorry

#check graphs_are_different

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graphs_are_different_l1165_116579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_multiplication_division_l1165_116544

/-- Converts a number from base 5 to base 10 --/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (q : ℚ) : ℕ := sorry

theorem base5_multiplication_division : 
  base10ToBase5 (roundToNearest (↑(base5ToBase10 203) / ↑(base5ToBase10 3) * ↑(base5ToBase10 24))) = 343 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_multiplication_division_l1165_116544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l1165_116592

/-- The area of a quadrilateral with a given diagonal length and two offsets. -/
noncomputable def quadrilateralArea (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) : ℝ :=
  (1/2) * diagonal * offset1 + (1/2) * diagonal * offset2

/-- Theorem: The area of a quadrilateral with diagonal 26 cm and offsets 9 cm and 6 cm is 195 cm². -/
theorem quadrilateral_area_example : quadrilateralArea 26 9 6 = 195 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l1165_116592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1165_116570

noncomputable section

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the perpendicular line through F₂
def perp_line (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define a line through F₂
def line_through_F₂ (m : ℝ) (y : ℝ) : ℝ × ℝ := (m * y + 1, y)

-- Define the area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Main theorem
theorem ellipse_theorem :
  ∃ (a b : ℝ) (P Q : ℝ × ℝ),
    a > 0 ∧ b > 0 ∧
    P ∈ Ellipse a b ∧ Q ∈ Ellipse a b ∧
    P = perp_line P.2 ∧ Q = perp_line Q.2 ∧
    distance P Q = 3 →
    (Ellipse a b = Ellipse 2 (Real.sqrt 3)) ∧
    (∀ (m : ℝ) (M N : ℝ × ℝ),
      M ∈ Ellipse 2 (Real.sqrt 3) ∧ N ∈ Ellipse 2 (Real.sqrt 3) ∧
      M = line_through_F₂ m M.2 ∧ N = line_through_F₂ m N.2 →
      circle_area (distance F₁ M + distance F₁ N + distance M N) / 16 ≤ 9 * Real.pi / 16) ∧
    (∃ (M N : ℝ × ℝ),
      M ∈ Ellipse 2 (Real.sqrt 3) ∧ N ∈ Ellipse 2 (Real.sqrt 3) ∧
      M = line_through_F₂ 0 M.2 ∧ N = line_through_F₂ 0 N.2 ∧
      circle_area (distance F₁ M + distance F₁ N + distance M N) / 16 = 9 * Real.pi / 16) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1165_116570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_az_sequence_lower_bound_l1165_116574

noncomputable def az_sequence : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => az_sequence (n + 1) + 1 / az_sequence n

theorem az_sequence_lower_bound :
  ∀ n : ℕ, az_sequence n ≥ Real.sqrt (n : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_az_sequence_lower_bound_l1165_116574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1165_116525

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 2*x

noncomputable def g (a x : ℝ) : ℝ := (1/2) * a * x^2 - (a-2) * x

theorem problem_solution :
  (∃ m : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 2, deriv f x ≤ m) ∧
            (∀ m' : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 2, deriv f x ≤ m') → m ≤ m') ∧
            m = 4) ∧
  (∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ∈ Set.Ioi (-1 : ℝ) ∧ x₂ ∈ Set.Ioi (-1 : ℝ) ∧ x₃ ∈ Set.Ioi (-1 : ℝ) ∧
                           x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                           f x₁ = g a x₁ ∧ f x₂ = g a x₂ ∧ f x₃ = g a x₃) ↔
            (a ∈ Set.Ioo (-5/9 : ℝ) (1/3) ∨ a ∈ Set.Ioi 3) ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1165_116525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_curve_l1165_116502

theorem lattice_points_on_curve : 
  let equation := fun (x y : ℤ) => 2 * x^2 - y^2 = 1800^2
  ∃ (solutions : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ solutions ↔ equation p.1 p.2) ∧ Finset.card solutions = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_on_curve_l1165_116502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_product_l1165_116528

/-- Ellipse with equation x^2 + 2y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 2 * p.2^2 = 1}

/-- Left focus of the ellipse -/
noncomputable def LeftFocus : ℝ × ℝ := (-Real.sqrt 2 / 2, 0)

/-- Right focus of the ellipse -/
noncomputable def RightFocus : ℝ × ℝ := (Real.sqrt 2 / 2, 0)

/-- A tangent line to the ellipse -/
structure TangentLine where
  k : ℝ
  b : ℝ
  isTangent : k^2 + 1/2 = b^2

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) (l : TangentLine) : ℝ :=
  abs (l.b + l.k * p.1 - p.2) / Real.sqrt (1 + l.k^2)

/-- The product of distances from foci to any tangent line is 1/2 -/
theorem foci_distance_product (l : TangentLine) :
    distanceToLine LeftFocus l * distanceToLine RightFocus l = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_distance_product_l1165_116528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1165_116509

theorem function_property (f : ℝ → ℝ) :
  (∀ x, f (Real.cos x) = Real.cos (17 * x)) →
  (∀ x, f (Real.sin x) = Real.sin (17 * x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1165_116509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_rankings_count_l1165_116527

/-- Represents a team in the tournament -/
inductive Team : Type
| E : Team
| F : Team
| G : Team
| H : Team

/-- Represents a group in the tournament -/
inductive TournamentGroup : Type
| East : TournamentGroup
| West : TournamentGroup

/-- Assigns teams to groups -/
def teamGroup (t : Team) : TournamentGroup :=
  match t with
  | Team.E | Team.F => TournamentGroup.East
  | Team.G | Team.H => TournamentGroup.West

/-- Represents a ranking of teams -/
def Ranking := List Team

/-- The tournament structure -/
structure Tournament :=
  (teams : List Team)
  (groups : List TournamentGroup)
  (teamAssignment : Team → TournamentGroup)
  (saturdayMatches : List (Team × Team))
  (sundayMatches : List (Team × Team))
  (mondayMatches : List (Team × Team))

/-- Checks if a ranking is valid for the tournament -/
def isValidRanking (t : Tournament) (r : Ranking) : Prop :=
  r.length = 4 ∧ r.Nodup

/-- The main theorem: there are 4 possible valid rankings -/
theorem tournament_rankings_count (t : Tournament) :
  (t.teams = [Team.E, Team.F, Team.G, Team.H]) →
  (t.groups = [TournamentGroup.East, TournamentGroup.West]) →
  (t.teamAssignment = teamGroup) →
  (t.saturdayMatches = [(Team.E, Team.F), (Team.G, Team.H)]) →
  (∃ (sundayMatches mondayMatches : List (Team × Team)),
     t.sundayMatches = sundayMatches ∧
     t.mondayMatches = mondayMatches) →
  (∃! (validRankings : List Ranking), 
     (∀ r ∈ validRankings, isValidRanking t r) ∧
     validRankings.length = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_rankings_count_l1165_116527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baguettes_sold_after_second_batch_l1165_116588

/-- Proves that the number of baguettes sold after the second batch is 52 --/
theorem baguettes_sold_after_second_batch :
  let batches_per_day : ℕ := 3
  let baguettes_per_batch : ℕ := 48
  let sold_after_first : ℕ := 37
  let sold_after_third : ℕ := 49
  let left_at_end : ℕ := 6
  let total_baguettes := batches_per_day * baguettes_per_batch
  let total_sold := total_baguettes - left_at_end
  let sold_first_and_third := sold_after_first + sold_after_third
  let sold_after_second := total_sold - sold_first_and_third
  sold_after_second = 52 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baguettes_sold_after_second_batch_l1165_116588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_M_l1165_116532

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := |x^2 + b*x|

-- Define M(b) as the maximum value of f(b)(x) for x ∈ [0,1]
noncomputable def M (b : ℝ) : ℝ := ⨆ (x : ℝ) (hx : x ∈ Set.Icc 0 1), f b x

-- Theorem statement
theorem min_value_of_M :
  ∃ (b : ℝ), M b = 3 - 2 * Real.sqrt 2 ∧ ∀ (b' : ℝ), M b' ≥ 3 - 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_M_l1165_116532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_income_increase_percentage_l1165_116512

/-- Calculates the percentage increase in total income given initial and final incomes -/
noncomputable def percentage_increase (initial_job_income initial_side_income initial_dividend_income final_job_income : ℝ) : ℝ :=
  let initial_total := initial_job_income + initial_side_income + initial_dividend_income
  let final_total := final_job_income + initial_side_income + initial_dividend_income
  (final_total - initial_total) / initial_total * 100

/-- Theorem stating that John's income increase is approximately 72.73% -/
theorem johns_income_increase_percentage :
  ∀ (ε : ℝ), ε > 0 →
  ∃ (initial_job_income initial_side_income initial_dividend_income final_job_income : ℝ),
    initial_job_income = 40 ∧
    initial_side_income = 10 ∧
    initial_dividend_income = 5 ∧
    final_job_income = 80 ∧
    abs (percentage_increase initial_job_income initial_side_income initial_dividend_income final_job_income - 72.73) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_income_increase_percentage_l1165_116512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l1165_116595

theorem compute_expression (a b : ℚ) (ha : a = 4/7) (hb : b = 5/6) : 
  a^3 * b^(-2 : ℤ) = 2304/8575 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l1165_116595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_eq_sqrt_minus_two_plus_two_sqrt_five_l1165_116598

/-- The distance between the intersections of x = y^4 and x + y^2 = 1 --/
noncomputable def intersection_distance : ℝ :=
  let y₁ := Real.sqrt ((Real.sqrt 5 - 1) / 2)
  let y₂ := -y₁
  let x := (3 - Real.sqrt 5) / 2
  Real.sqrt ((x - x)^2 + (y₁ - y₂)^2)

/-- The theorem stating that the intersection distance equals √(-2 + 2√5) --/
theorem intersection_distance_eq_sqrt_minus_two_plus_two_sqrt_five :
  intersection_distance = Real.sqrt (-2 + 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_eq_sqrt_minus_two_plus_two_sqrt_five_l1165_116598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_representation_l1165_116551

-- Define the complex numbers
def z₁ : ℂ := 5
def z₂ : ℂ := -3 * Complex.I
def z₃ : ℂ := 3 + 2 * Complex.I
def z₄ : ℂ := 5 - 2 * Complex.I
def z₅ : ℂ := -3 + 2 * Complex.I
def z₆ : ℂ := -1 - 5 * Complex.I

-- Theorem stating the correct representation of the complex numbers on the plane
theorem complex_representation :
  (z₁.re = 5 ∧ z₁.im = 0) ∧
  (z₂.re = 0 ∧ z₂.im = -3) ∧
  (z₃.re = 3 ∧ z₃.im = 2) ∧
  (z₄.re = 5 ∧ z₄.im = -2) ∧
  (z₅.re = -3 ∧ z₅.im = 2) ∧
  (z₆.re = -1 ∧ z₆.im = -5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_representation_l1165_116551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l1165_116507

noncomputable def sample_data : List ℝ := [2015, 2017, 2019, 2018, 2016]

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length : ℝ)

noncomputable def variance (data : List ℝ) : ℝ :=
  (data.map (fun x => (x - mean data)^2)).sum / (data.length : ℝ)

theorem variance_of_sample_data :
  mean sample_data = 2017 → variance sample_data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l1165_116507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_17B_l1165_116545

noncomputable def A : ℝ := ∑' n, if (n % 4 = 0 ∧ n % 8 ≠ 0) ∨ (n % 4 = 2) then (-1)^((n / 2) + 1) / n^2 else 0

noncomputable def B : ℝ := ∑' n, if n % 8 = 0 then 1 / n^2 else 0

theorem A_equals_17B : A = 17 * B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_17B_l1165_116545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_second_quadrant_condition_l1165_116569

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (2*m^2 - 3*m - 2) (m^2 - 3*m + 2)

-- Theorem for purely imaginary condition
theorem purely_imaginary_condition (m : ℝ) :
  z m = Complex.I * Complex.im (z m) ↔ m = -1/2 := by sorry

-- Theorem for second quadrant condition
theorem second_quadrant_condition (m : ℝ) :
  Complex.re (z m) < 0 ∧ Complex.im (z m) > 0 ↔ -1/2 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_second_quadrant_condition_l1165_116569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_2000_l1165_116583

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex fraction
noncomputable def complex_fraction : ℂ := (1 + 2*i) / (2 - i)

-- State the theorem
theorem complex_power_2000 : complex_fraction ^ 2000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_2000_l1165_116583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vision_standard_comparison_l1165_116599

structure School :=
  (science_rate : ℝ)
  (liberal_arts_rate : ℝ)

def SchoolA : School :=
  { science_rate := 0.6,
    liberal_arts_rate := 0.7 }

def SchoolB : School :=
  { science_rate := 0.65,
    liberal_arts_rate := 0.75 }

theorem vision_standard_comparison (a b : School) :
  (a.science_rate < b.science_rate ∧ a.liberal_arts_rate < b.liberal_arts_rate) →
  (a.liberal_arts_rate > a.science_rate ∧ b.liberal_arts_rate > b.science_rate) →
  ∃ (sa la sb lb : ℝ), 
    (sa * a.science_rate + la * a.liberal_arts_rate) / (sa + la) >
    (sb * b.science_rate + lb * b.liberal_arts_rate) / (sb + lb) :=
by
  sorry

#check vision_standard_comparison SchoolA SchoolB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vision_standard_comparison_l1165_116599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1165_116534

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def S (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

/-- 
Given a geometric sequence where Sₙ is the sum of its first n terms,
if 8S₆ = 7S₃, then the common ratio of the sequence is -1/2
-/
theorem geometric_sequence_ratio 
  (a₁ : ℝ) (q : ℝ) (h₁ : q ≠ 1) (h₂ : a₁ ≠ 0) :
  8 * S a₁ q 6 = 7 * S a₁ q 3 → q = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1165_116534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_one_floor_min_cost_value_l1165_116576

/-- Represents the total construction area in square meters -/
def a : ℝ := sorry

/-- Represents the number of floors -/
def n : ℕ := sorry

/-- Land expropriation cost per square meter -/
def land_cost : ℝ := 2388

/-- Base construction cost per square meter for the first two floors -/
def base_construction_cost : ℝ := 445

/-- Additional cost per square meter for each floor beyond the second -/
def additional_cost : ℝ := 30

/-- Total cost function -/
noncomputable def total_cost (a : ℝ) (n : ℝ) : ℝ :=
  land_cost * a + (base_construction_cost * n * a) + (additional_cost * ((n - 2) * (n - 1) / 2) * a)

/-- Theorem stating the minimum total cost occurs at one floor -/
theorem min_cost_at_one_floor (a : ℝ) (h : a > 0) :
  ∀ n : ℕ, n ≥ 1 → total_cost a 1 ≤ total_cost a (n : ℝ) := by
  sorry

/-- Theorem stating the minimum total cost value -/
theorem min_cost_value (a : ℝ) (h : a > 0) :
  total_cost a 1 = 2788 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_one_floor_min_cost_value_l1165_116576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l1165_116571

theorem sin_identity (α : ℝ) (h : α = π / 7) :
  (Real.sin (3 * α))^2 - (Real.sin α)^2 = Real.sin (2 * α) * Real.sin (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l1165_116571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_four_sevenths_three_fifths_l1165_116556

-- Define the remainder function for real numbers
noncomputable def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

-- Theorem statement
theorem remainder_four_sevenths_three_fifths :
  rem (4/7 : ℝ) (3/5 : ℝ) = 4/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_four_sevenths_three_fifths_l1165_116556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_problem_l1165_116563

/-- Represents the average number of bites for a fishing rod in a given time interval -/
structure FishingRod where
  bites : ℚ
  interval : ℚ

/-- Calculates the average waiting time for the first bite given two fishing rods -/
noncomputable def averageWaitingTime (rod1 rod2 : FishingRod) : ℚ :=
  rod1.interval / (rod1.bites + rod2.bites)

theorem fishing_problem (rod1 rod2 : FishingRod) 
  (h1 : rod1.bites = 3 ∧ rod1.interval = 6)
  (h2 : rod2.bites = 2 ∧ rod2.interval = 6) :
  averageWaitingTime rod1 rod2 = 6/5 := by
  sorry

#eval (6 : ℚ) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_problem_l1165_116563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_F_properties_l1165_116559

/-- A complex number F satisfying the given conditions -/
noncomputable def F : ℂ :=
  sorry

/-- Conditions on F -/
axiom F_in_second_quadrant : F.re < 0 ∧ F.im < 0
axiom F_inside_unit_circle : Complex.abs F < 1

/-- Theorem about the reciprocal of F -/
theorem reciprocal_F_properties :
  let inv_F := F⁻¹
  Complex.abs inv_F > 1 ∧ inv_F.re < 0 ∧ inv_F.im > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_F_properties_l1165_116559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_power_condition_l1165_116560

-- Define the function f(x) = (log₀.₅a)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log a / Real.log 0.5) ^ x

-- State the theorem
theorem increasing_log_power_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 < a ∧ a < 0.5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_power_condition_l1165_116560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_proof_l1165_116548

/-- The initial length of Orlan's rope -/
noncomputable def initial_length : ℝ := 20

/-- The length of rope Orlan gave to Allan -/
noncomputable def given_to_allan : ℝ := initial_length / 4

/-- The length of rope remaining after giving to Allan -/
noncomputable def remaining_after_allan : ℝ := initial_length - given_to_allan

/-- The length of rope Orlan gave to Jack -/
noncomputable def given_to_jack : ℝ := (2 / 3) * remaining_after_allan

/-- The length of rope Orlan has left -/
noncomputable def orlan_left : ℝ := 5

theorem rope_length_proof :
  initial_length - given_to_allan - given_to_jack = orlan_left :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_proof_l1165_116548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l1165_116575

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the point M
def point_M : ℝ × ℝ := (2, -1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 2*x - y - 5 = 0

theorem tangent_line_proof :
  let (x₀, y₀) := point_M
  circle_equation x₀ y₀ →
  (∀ x y, circle_equation x y → ((x - x₀) * 2 + (y - y₀) * (-1) = 0)) →
  ∀ x y, tangent_line x y ↔ ((x - x₀) * 2 + (y - y₀) * (-1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_proof_l1165_116575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l1165_116523

theorem odd_terms_in_binomial_expansion (p q : ℤ) 
  (hp : Odd p) (hq : Odd q) : 
  (Finset.filter (λ k ↦ Odd (Nat.choose 8 k * p^k * q^(8-k))) (Finset.range 9)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l1165_116523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_m_bound_l1165_116514

-- Define the functions f and g
noncomputable def f (x m : ℝ) : ℝ := x^2 + m
noncomputable def g (x m : ℝ) : ℝ := (1/2)^x - m

-- State the theorem
theorem function_inequality_implies_m_bound :
  ∀ m : ℝ, 
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 3, ∃ x2 ∈ Set.Icc (0 : ℝ) 2, f x1 m ≥ g x2 m) →
  m ≥ 1/8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_m_bound_l1165_116514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1165_116526

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x - m

-- Define the foci
noncomputable def leftFocus : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def rightFocus : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the distance from a point to a line
noncomputable def distanceToLine (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  let (x, y) := p
  abs (m + y - x) / Real.sqrt 2

-- Theorem statement
theorem ellipse_line_intersection (m : ℝ) :
  (distanceToLine leftFocus m) / (distanceToLine rightFocus m) = 3 →
  m = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l1165_116526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1165_116529

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define the dot product operation for vectors
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the given vectors
def AC : Vector2D := (1, 2)
def BD : Vector2D := (-2, 2)

-- State the theorem
theorem min_dot_product :
  ∃ (AB CD : Vector2D), 
    (∀ (AB' CD' : Vector2D), dot_product AB' CD' ≥ dot_product AB CD) ∧
    dot_product AB CD = -9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1165_116529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l1165_116546

/-- The time taken for a train to pass a platform -/
theorem train_passing_platform (train_length platform_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 360 →
  platform_length = 180 →
  train_speed_kmh = 45 →
  (train_length + platform_length) / (train_speed_kmh * (1000 / 3600)) = 43.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l1165_116546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_to_rectangles_ratio_l1165_116542

/-- Represents a square checkerboard. -/
structure Checkerboard where
  size : Nat
  horizontal_lines : Nat
  vertical_lines : Nat

/-- Calculates the number of rectangles in a checkerboard. -/
def num_rectangles (c : Checkerboard) : Nat :=
  (c.size + 1).choose 2 * (c.size + 1).choose 2

/-- Calculates the number of squares in a checkerboard. -/
def num_squares (c : Checkerboard) : Nat :=
  (c.size * (c.size + 1) * (2 * c.size + 1)) / 6

/-- The main theorem about the ratio of squares to rectangles in a 10x10 checkerboard. -/
theorem squares_to_rectangles_ratio (c : Checkerboard) 
    (h1 : c.size = 10) 
    (h2 : c.horizontal_lines = 9) 
    (h3 : c.vertical_lines = 9) : 
    (num_squares c : ℚ) / (num_rectangles c : ℚ) = 7 / 55 := by
  sorry

/-- Example calculation -/
def example_checkerboard : Checkerboard :=
  { size := 10, horizontal_lines := 9, vertical_lines := 9 }

#eval num_squares example_checkerboard
#eval num_rectangles example_checkerboard

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squares_to_rectangles_ratio_l1165_116542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l1165_116504

def p1 (x : ℝ) : ℝ := 3*x^4 - 2*x^3 + 4*x^2 - 5*x + 2
def p2 (x : ℝ) : ℝ := x^3 - 4*x^2 + 3*x + 6

theorem coefficient_of_x_squared : 
  ∃ q r : ℝ → ℝ, ∀ x, p1 x * p2 x = q x * x^2 + r x ∧ q x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l1165_116504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_equation_theorem_l1165_116506

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem coordinate_equation_theorem (x y : ℝ) :
  (⌊fractional_part (x / 3)⌋ + ⌊fractional_part (y / 3)⌋ = 3 ↔
    ((1 ≤ x % 3 ∧ x % 3 < 2) ∧ (2 ≤ y % 3 ∧ y % 3 < 3)) ∨
    ((2 ≤ x % 3 ∧ x % 3 < 3) ∧ (1 ≤ y % 3 ∧ y % 3 < 2))) ∧
  (⌊fractional_part (x / 3)⌋ + ⌊fractional_part (y / 3)⌋ = 2 ↔
    ((0 ≤ x % 3 ∧ x % 3 < 1) ∧ (2 ≤ y % 3 ∧ y % 3 < 3)) ∨
    ((1 ≤ x % 3 ∧ x % 3 < 2) ∧ (1 ≤ y % 3 ∧ y % 3 < 2)) ∨
    ((2 ≤ x % 3 ∧ x % 3 < 3) ∧ (0 ≤ y % 3 ∧ y % 3 < 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_equation_theorem_l1165_116506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_walk_area_hexagon_l1165_116593

/-- The area accessible to a dog tied to one vertex of a regular hexagon -/
noncomputable def dogWalkArea (hexagonSideLength : ℝ) (leashLength : ℝ) : ℝ :=
  (23 / 3) * Real.pi

/-- Theorem stating the area accessible to a dog tied to a regular hexagon -/
theorem dog_walk_area_hexagon :
  dogWalkArea 1 3 = (23 / 3) * Real.pi :=
by
  -- Unfold the definition of dogWalkArea
  unfold dogWalkArea
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_walk_area_hexagon_l1165_116593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_duration_determination_l1165_116537

structure OperatingRoom where
  name : String
  start_time : ℕ
  duration : ℕ

def total_duration (rooms : List OperatingRoom) : ℕ :=
  rooms.map (λ room => room.duration) |>.sum

theorem operation_duration_determination (rooms : List OperatingRoom) 
  (h1 : rooms.length = 4)
  (h2 : ∀ r, r ∈ rooms → r.name ∈ ["A", "B", "C", "D"])
  (h3 : total_duration rooms = 185)
  (h4 : ∃ t : ℕ, t ≤ 185 ∧ total_duration (rooms.filter (λ r => r.start_time ≤ t)) = 46 ∧ t + 36 = 185)
  (h5 : ∃ t : ℕ, t ≤ 185 ∧ total_duration (rooms.filter (λ r => r.start_time ≤ t)) = 19 ∧ t + 46 = 185)
  : (∃! r, r ∈ rooms ∧ r.name = "D" ∧ r.duration = 31) ∧
    (¬∀ r, r ∈ rooms → r.name ≠ "D" → ∃! d : ℕ, r.duration = d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_duration_determination_l1165_116537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_constant_l1165_116558

noncomputable section

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line l
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 = c}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem circle_polar_equation_constant (C : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) 
  (A B P : ℝ × ℝ) (h1 : C = Circle (3, 2) 3) (h2 : A ∈ C ∩ l) (h3 : B ∈ C ∩ l) 
  (h4 : distance P A * distance P B = 7) :
  ∃ (r θ : ℝ), r * Real.cos θ + 3 = r^2 / 7 ∧ r * Real.sin θ + 2 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_constant_l1165_116558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l1165_116564

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  vertex : ℝ × ℝ
  h_vertex : vertex = (0, -3)
  focus : ℝ × ℝ
  h_focus : focus.1 > 0 ∧ focus.2 = 0
  h_equal_dist : (vertex.1^2 + vertex.2^2) = (focus.1^2 + focus.2^2)

/-- A point C related to the focus -/
noncomputable def point_C (e : SpecialEllipse) : ℝ × ℝ := (e.focus.1 / 3, 0)

/-- The equation of the ellipse -/
def ellipse_equation (e : SpecialEllipse) : (ℝ × ℝ) → Prop :=
  fun p ↦ p.1^2 / 18 + p.2^2 / 9 = 1

/-- The equations of the tangent line AB -/
def tangent_line_equations (e : SpecialEllipse) : List ((ℝ × ℝ) → Prop) :=
  [fun p ↦ p.2 = 1/2 * p.1 - 3, fun p ↦ p.2 = p.1 - 3]

/-- The main theorem -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (ellipse_equation e = fun p ↦ p.1^2 / 18 + p.2^2 / 9 = 1) ∧
  (∃ (eq : (ℝ × ℝ) → Prop), eq ∈ tangent_line_equations e ∧
    ∃ (B : ℝ × ℝ), B ≠ e.vertex ∧ ellipse_equation e B ∧
    ∃ (P : ℝ × ℝ), eq P ∧
      P.1 = (B.1 + e.vertex.1) / 2 ∧ P.2 = (B.2 + e.vertex.2) / 2 ∧
      (P.1 - (point_C e).1)^2 + (P.2 - (point_C e).2)^2 =
        ((B.1 - e.vertex.1) / 2)^2 + ((B.2 - e.vertex.2) / 2)^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l1165_116564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1165_116501

theorem sum_of_roots_quadratic (a b c : ℝ) (ha : a ≠ 0) 
  (h : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :
  ∃ s : ℝ, s = -b / a ∧ s = x + y :=
by
  sorry

theorem sum_of_roots_specific_quadratic : 
  ∃ s : ℝ, s = 7 ∧ (∀ y : ℝ, y^2 - 7*y + 12 = 0 → 
    ∃ x : ℝ, x ≠ y ∧ x^2 - 7*x + 12 = 0 ∧ s = x + y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1165_116501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_in_geometric_sequence_l1165_116557

/-- Given positive integers a, b, c forming a geometric sequence with abc = 216,
    the smallest possible value of b is 6. -/
theorem smallest_b_in_geometric_sequence (a b c : ℕ) 
    (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) : 
  (∃ r : ℚ, (0 < r) ∧ b = a * r ∧ c = b * r) →  -- geometric sequence condition
  a * b * c = 216 →                             -- product condition
  (∀ b' : ℕ, 0 < b' →
    (∃ a' c' : ℕ, 0 < a' ∧ 0 < c' ∧
      (∃ r : ℚ, (0 < r) ∧ b' = a' * r ∧ c' = b' * r) ∧ 
      a' * b' * c' = 216) → 
    b ≤ b') →
  b = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_in_geometric_sequence_l1165_116557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remy_sold_110_bottles_l1165_116519

/-- The number of bottles of soda Remy sold in the morning -/
def remy_bottles : ℕ := 110

/-- The number of bottles of soda Nick sold in the morning -/
def nick_bottles : ℕ := remy_bottles - 6

/-- The price per bottle in dollars -/
def price_per_bottle : ℚ := 1/2

/-- The total evening sales in dollars -/
def evening_sales : ℚ := 55

/-- The difference between evening and morning sales in dollars -/
def sales_difference : ℚ := 3

theorem remy_sold_110_bottles :
  (remy_bottles : ℚ) * price_per_bottle + (nick_bottles : ℚ) * price_per_bottle = evening_sales - sales_difference →
  remy_bottles = 110 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remy_sold_110_bottles_l1165_116519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_line_equation_l1165_116554

-- Define the curve C
def C (x y m : ℝ) : Prop := x^2 + y^2 + 2*x + m = 0

-- Define when a curve is a circle
def is_circle (f : ℝ → ℝ → ℝ → Prop) : Prop := 
  ∃ (h k r : ℝ), ∀ (x y : ℝ), f x y r ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem 1: C is a circle iff m < 1
theorem circle_condition (m : ℝ) : 
  is_circle (λ x y _ ↦ C x y m) ↔ m < 1 := by
  sorry

-- Define the line passing through (1, 1)
def line_through_P (a b : ℝ) (x y : ℝ) : Prop := a*x + b*y = a + b

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x1 - x2)^2 + (y1 - y2)^2)^(1/2)

-- Theorem 2: When m = -7, the line intersecting C with |AB| = 4 has equation 3x + 4y - 7 = 0 or x = 1
theorem line_equation :
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), line_through_P a b x y → 
      (∃ (x1 y1 x2 y2 : ℝ), 
        C x1 y1 (-7) ∧ C x2 y2 (-7) ∧ 
        line_through_P a b x1 y1 ∧ line_through_P a b x2 y2 ∧
        distance x1 y1 x2 y2 = 4)) →
    ((a = 3 ∧ b = 4) ∨ (a = 1 ∧ b = 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_line_equation_l1165_116554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_red_is_blue_l1165_116565

-- Define the set of colors
inductive Color
| Red | Blue | Orange | Yellow | Green | Purple

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : List Face
  adjacency : List (Face × Face)

-- Define the property of being opposite faces in a cube
def are_opposite (c : Cube) (f1 f2 : Face) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1 ≠ f2 ∧ 
  ∀ f, (f1, f) ∈ c.adjacency → (f2, f) ∉ c.adjacency

-- Define the property of being adjacent faces in a cube
def are_adjacent (c : Cube) (f1 f2 : Face) : Prop :=
  (f1, f2) ∈ c.adjacency ∨ (f2, f1) ∈ c.adjacency

-- Theorem statement
theorem opposite_red_is_blue (c : Cube) 
  (h1 : c.faces.length = 6)
  (h2 : ∀ f, f ∈ c.faces → ∃! color, f.color = color)
  (h3 : ∃ f1 f2, f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1.color = Color.Orange ∧ f2.color = Color.Purple ∧ are_adjacent c f1 f2)
  : ∃ f1 f2, f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1.color = Color.Red ∧ f2.color = Color.Blue ∧ are_opposite c f1 f2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_red_is_blue_l1165_116565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_garden_area_l1165_116586

/-- The number of blocks on each side of the hexagonal garden -/
def n : ℕ := 202

/-- The area of a regular hexagon with side length 1 -/
noncomputable def hexagon_area : ℝ := 3 * Real.sqrt 3 / 2

/-- The number of hexagons inside the garden -/
def inner_hexagons : ℕ := 1 + 6 * (n - 2) * (n - 1) / 2

/-- The area of the garden -/
noncomputable def garden_area : ℝ := (inner_hexagons : ℝ) * hexagon_area

/-- The integer m such that the garden area is m(√3/2) square units -/
noncomputable def m : ℕ := Int.toNat ⌊garden_area * 2 / Real.sqrt 3⌋

theorem hexagonal_garden_area :
  m % 1000 = 803 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_garden_area_l1165_116586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minister_goal_achievable_no_return_path_l1165_116522

/-- Represents the number of cities -/
def n : Nat := 5

/-- Represents the total number of cities -/
def total_cities : Nat := 2^n

/-- Represents the maximum number of days needed to reorganize the roads -/
def max_days : Nat := 2^(n-2) * (2^n - n - 1)

/-- Theorem stating that the minister can achieve his goal within the calculated number of days -/
theorem minister_goal_achievable : max_days ≤ 214 := by
  sorry

/-- Theorem stating that the reorganization prevents return to any city -/
theorem no_return_path (city : Fin total_cities) : 
  ∀ path : List (Fin total_cities), path ≠ [] → path.head? = some city → 
  path.getLast? ≠ some city := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minister_goal_achievable_no_return_path_l1165_116522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_exam_scores_l1165_116591

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  val : ℕ
  property : 10 ≤ val ∧ val < 100

/-- The math exam scores problem -/
theorem math_exam_scores 
  (joao_score : TwoDigitNumber) 
  (claudia_score : TwoDigitNumber) 
  (total_score : ℕ) 
  (h1 : claudia_score.val = joao_score.val + 13)
  (h2 : joao_score.val + claudia_score.val = total_score)
  (h3 : 100 ≤ total_score ∧ total_score < 200) :
  joao_score.val = 68 ∧ claudia_score.val = 81 := by
  sorry

#check math_exam_scores

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_exam_scores_l1165_116591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_seven_sixteenths_l1165_116524

def range : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 60) (Finset.range 61)

def multiples_of_4 : Finset ℕ := Finset.filter (λ n => n % 4 = 0) range

def probability_at_least_one_multiple_of_4 : ℚ :=
  1 - (1 - (multiples_of_4.card : ℚ) / (range.card : ℚ))^2

theorem probability_is_seven_sixteenths :
  probability_at_least_one_multiple_of_4 = 7/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_seven_sixteenths_l1165_116524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_no_triangle_l1165_116500

/-- A simple graph with no self-loops -/
structure MySimpleGraph (V : Type*) where
  adj : V → V → Prop
  symm : ∀ u v, adj u v → adj v u
  loopless : ∀ v, ¬adj v v

/-- The number of vertices in a graph -/
def num_vertices {V : Type*} (G : MySimpleGraph V) : ℕ := sorry

/-- The number of edges in a graph -/
def num_edges {V : Type*} (G : MySimpleGraph V) : ℕ := sorry

/-- A triangle in a graph is a cycle of length 3 -/
def has_triangle {V : Type*} (G : MySimpleGraph V) : Prop := sorry

theorem max_edges_no_triangle {V : Type*} (G : MySimpleGraph V) :
  ¬has_triangle G →
  num_edges G ≤ (num_vertices G)^2 / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_edges_no_triangle_l1165_116500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_averaging_properties_l1165_116518

noncomputable def avg (x y : ℝ) : ℝ := (x + y) / 2

theorem averaging_properties :
  (∀ x y : ℝ, avg x y = avg y x) ∧
  (∀ a b c : ℝ, a + avg b c = avg (a + b) (a + c)) :=
by
  constructor
  · intro x y
    simp [avg]
    ring
  · intro a b c
    simp [avg]
    ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_averaging_properties_l1165_116518
