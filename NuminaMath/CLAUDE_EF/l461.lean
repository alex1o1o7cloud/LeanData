import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_integer_l461_46125

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 2 * a n + Real.sqrt (3 * (a n)^2 + 1)

theorem a_is_integer : ∀ n : ℕ, ∃ k : ℤ, a n = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_integer_l461_46125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l461_46160

theorem equation_solution : ∃ (d e f : ℕ+), 
  (5 * Real.sqrt (Real.rpow 6 (1/4) - Real.rpow 3 (1/3)) = Real.rpow d.val (1/3) + Real.rpow e.val (1/3) - Real.rpow f.val (1/4)) ∧ 
  (d.val + e.val + f.val = 91) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l461_46160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_sum_theorem_l461_46186

def CircularArrangement (n : Nat) := Fin n → Int

def ValidArrangement (arr : CircularArrangement 2012) : Prop :=
  (∀ i, arr i = 1 ∨ arr i = -1) ∧
  (∀ i, (Finset.range 10).sum (fun j ↦ arr ((i + j) % 2012)) ≠ 0)

def ArrangementSum (arr : CircularArrangement 2012) : Int :=
  (Finset.range 2012).sum (fun i ↦ arr i)

theorem circular_arrangement_sum_theorem :
  ∀ arr : CircularArrangement 2012,
    ValidArrangement arr →
    ∃ S : Int,
      ArrangementSum arr = S ∧
      S % 2 = 0 ∧
      ((S ≥ 404 ∧ S ≤ 2012) ∨ (S ≥ -2012 ∧ S ≤ -404)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_sum_theorem_l461_46186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l461_46189

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (C.1 - A.1, C.2 - A.2)
  let w := (C.1 - B.1, C.2 - B.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (-2,3), (6,-1), and (10,4) is 28 -/
theorem triangle_area_example : 
  triangle_area (-2, 3) (6, -1) (10, 4) = 28 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l461_46189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_XY_length_l461_46132

/-- Represents a 60-30-90 triangle -/
structure Triangle60_30_90 where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  is_60_30_90 : True  -- This is a placeholder for the 60-30-90 property

/-- The length of a side in a triangle -/
noncomputable def side_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem side_XY_length (t : Triangle60_30_90) (h : side_length t.X t.Z = 12) :
  side_length t.X t.Y = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_XY_length_l461_46132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_positive_reals_f_min_max_on_interval_l461_46114

noncomputable def f (x : ℝ) : ℝ := x - 1 / x

theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by sorry

theorem f_increasing_on_positive_reals : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

theorem f_min_max_on_interval : 
  (∀ x : ℝ, 1 ≤ x → x ≤ 4 → 0 ≤ f x) ∧ 
  (∀ x : ℝ, 1 ≤ x → x ≤ 4 → f x ≤ 15/4) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 0) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 15/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_increasing_on_positive_reals_f_min_max_on_interval_l461_46114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_around_cylinders_l461_46196

/-- Given a rope that can make 70 rounds around a cylinder with a base radius of 14 cm,
    prove that it can make 49 rounds around a cylinder with a base radius of 20 cm. -/
theorem rope_around_cylinders (rope_length : ℝ) : 
  rope_length = 70 * 2 * Real.pi * 14 → 
  ⌊rope_length / (2 * Real.pi * 20)⌋ = 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_around_cylinders_l461_46196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_approx_32_percent_l461_46146

/-- Represents a circle in the pattern -/
structure Circle where
  radius : ℝ
  isBlack : Bool

/-- Generates the sequence of circles in the pattern -/
def generatePattern : List Circle :=
  let rec aux (n : ℕ) (r : ℝ) (isBlack : Bool) : List Circle :=
    if n = 0 then []
    else { radius := r, isBlack := isBlack } :: aux (n - 1) (r + if isBlack then 3 else 1) (not isBlack)
  aux 7 3 true

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius ^ 2

/-- Calculates the net black area in the pattern -/
noncomputable def blackArea (pattern : List Circle) : ℝ :=
  let areas := pattern.map circleArea
  let blackAreas := List.zipWith (fun c a => if c.isBlack then a else 0) pattern areas
  List.foldl (fun acc x => acc + x) 0 blackAreas

/-- Calculates the total area of the pattern -/
noncomputable def totalArea (pattern : List Circle) : ℝ :=
  match pattern.getLast? with
  | some lastCircle => circleArea lastCircle
  | none => 0

/-- The main theorem to be proved -/
theorem black_percentage_approx_32_percent :
  let pattern := generatePattern
  let blackPercentage := (blackArea pattern / totalArea pattern) * 100
  ∃ ε > 0, abs (blackPercentage - 32) < ε := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_approx_32_percent_l461_46146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l461_46191

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- State the theorem
theorem problem_statement 
  (a b : ℝ) 
  (h_odd : ∀ x, x ∈ Set.Ioo (-1) 1 → f a b x = -f a b (-x))
  (h_domain : Set.range (f a b) = Set.Ioo (-1) 1)
  (h_value : f a b 1 = 1/2) :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f a b x = x / (1 + x^2)) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Ioo (-1) 1 → x₂ ∈ Set.Ioo (-1) 1 → x₁ < x₂ → f a b x₁ < f a b x₂) ∧
  (∀ t : ℝ, f a b (2*t-1) + f a b (t-1) < 0 ↔ 0 < t ∧ t < 2/3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l461_46191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_relationship_l461_46136

theorem negation_relationship :
  (∀ x : ℝ, (|x + 1| > 2 → x ≥ 2)) →
  (∀ x : ℝ, (¬(|x + 1| > 2) → ¬(x ≥ 2))) ∧
  ¬(∀ x : ℝ, (¬(x ≥ 2) → ¬(|x + 1| > 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_relationship_l461_46136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lagrange_mean_value_exponential_l461_46127

theorem lagrange_mean_value_exponential :
  ∃ ξ : ℝ, 0 < ξ ∧ ξ < 1 ∧ Real.exp 1 - Real.exp 0 = Real.exp ξ * (1 - 0) ∧ ξ = Real.log (Real.exp 1 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lagrange_mean_value_exponential_l461_46127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_607080_has_one_zero_l461_46111

/-- Represents the verbal pronunciation of a number -/
def NumberPronunciation : Type := String

/-- Counts the occurrences of "zero" in a string -/
def countZeros (s : String) : Nat :=
  s.split (· = ' ') |>.filter (· = "zero") |>.length

/-- Returns the verbal pronunciation of a number -/
noncomputable def pronounceNumber (n : Nat) : NumberPronunciation :=
  sorry  -- Implementation details omitted

/-- Theorem stating that 607080 is the only number among the given options
    that is pronounced with exactly one "zero" -/
theorem only_607080_has_one_zero :
  let numbers := [2900707, 29004000, 607080]
  ∀ n ∈ numbers, countZeros (pronounceNumber n) = 1 ↔ n = 607080 := by
  sorry

-- Remove the #eval line as it's not necessary for building and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_607080_has_one_zero_l461_46111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squared_distances_l461_46153

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_sum_of_squared_distances (a b c : V) 
  (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hc : ‖c‖ = 4) : 
  ‖a - b‖^2 + ‖a - c‖^2 + ‖b - c‖^2 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squared_distances_l461_46153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_l461_46143

-- Define the complex number z as a function of x
noncomputable def z (x : ℝ) : ℂ := ⟨Real.log (x^2 - 2*x - 2), x^2 + 3*x + 2⟩

-- Theorem for when z is a real number
theorem z_is_real (x : ℝ) : (z x).im = 0 ↔ x = -1 ∨ x = -2 := by
  sorry

-- Theorem for when z is a purely imaginary number
theorem z_is_purely_imaginary (x : ℝ) : (z x).re = 1 ∧ (z x).im ≠ 0 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_purely_imaginary_l461_46143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l461_46115

-- Define the slopes of the two lines
noncomputable def m₁ : ℝ := -1
noncomputable def m₂ : ℝ := 4

-- Define the angle between the lines
noncomputable def θ : ℝ := Real.arctan ((m₂ - m₁) / (1 + m₁ * m₂))

-- Define the slope of the angle bisector
noncomputable def k : ℝ := (m₁ + m₂ - Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

-- Theorem statement
theorem angle_bisector_slope :
  θ > π / 2 ∧ k = -1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_slope_l461_46115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l461_46175

-- Define the triangle ABC
structure Triangle where
  A : Real  -- Angle A
  B : Real  -- Angle B
  C : Real  -- Angle C
  a : Real  -- Side opposite to A
  b : Real  -- Side opposite to B
  c : Real  -- Side opposite to C

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the specific conditions for our triangle
def specialTriangle (t : Triangle) : Prop :=
  isValidTriangle t ∧
  t.A = 2 * t.B ∧
  (1/2) * t.a * t.c * Real.sin t.B = (1/4) * t.a^2

-- State the theorem
theorem special_triangle_properties (t : Triangle) 
  (h : specialTriangle t) : 
  t.a^2 = t.b * (t.b + t.c) ∧ 
  (t.B = Real.pi/4 ∨ t.B = Real.pi/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l461_46175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l461_46109

theorem trigonometric_identities (α : ℝ) 
  (h1 : α > π / 2) (h2 : α < π) (h3 : Real.sin α = 3 / 5) : 
  (Real.sin (π / 4 + α) = -Real.sqrt 2 / 10) ∧ 
  (Real.cos (π / 6 - 2 * α) = (7 * Real.sqrt 3 - 24) / 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l461_46109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_nonempty_intersection_l461_46129

theorem max_subsets_with_nonempty_intersection (n : ℕ) (hn : n > 0) :
  (∃ (k : ℕ), k = 2^(n - 1) ∧
    ∃ (S : Finset (Finset (Fin n))),
      S.card = k ∧
      ∀ (A B : Finset (Fin n)), A ∈ S → B ∈ S → A ≠ B → (A ∩ B).Nonempty) ∧
  (∀ (m : ℕ), m > 2^(n - 1) →
    ¬∃ (S : Finset (Finset (Fin n))),
      S.card = m ∧
      ∀ (A B : Finset (Fin n)), A ∈ S → B ∈ S → A ≠ B → (A ∩ B).Nonempty) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_nonempty_intersection_l461_46129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_positive_f_inequality_range_l461_46161

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.exp x - 2 * Real.log x
noncomputable def g (x : ℝ) := Real.cos x - 1 + x^2 / 2

-- Theorem 1: g(x) > 0 for x > 0
theorem g_positive (x : ℝ) (h : x > 0) : g x > 0 := by sorry

-- Theorem 2: Range of a for which the inequality holds
theorem f_inequality_range (a : ℝ) :
  (∀ x > 0, f x ≥ 2 * Real.log ((x + 1) / x) + (x + 2) / (x + 1) - Real.cos (abs (a * x))) ↔
  a ∈ Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_positive_f_inequality_range_l461_46161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_negative_one_l461_46133

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

-- State the theorem
theorem second_derivative_at_negative_one :
  (deriv (deriv f)) (-1) = -Real.exp 1 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_at_negative_one_l461_46133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l461_46100

-- Define the triangle ABC
def triangle_ABC : Set (ℝ × ℝ) := sorry

-- Define the lengths of sides AB and AC
def AB : ℝ := 4
def AC : ℝ := 3

-- Define angle A in radians (30 degrees = π/6 radians)
noncomputable def angle_A : ℝ := Real.pi / 6

-- Define the area of triangle ABC
noncomputable def area_ABC : ℝ := (1/2) * AB * AC * Real.sin angle_A

-- Theorem statement
theorem area_of_triangle_ABC : area_ABC = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l461_46100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_n_l461_46167

theorem subset_sum_divisible_by_n (n : ℕ) (hn : 0 < n) (a : Fin n → ℤ) :
  ∃ (s : Finset (Fin n)) (hs : s.Nonempty),
    (s.sum (λ i => a i)) % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_n_l461_46167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_perimeter_relationship_l461_46165

theorem square_triangle_area_perimeter_relationship : ∃ (s t : ℝ), 
  s > 0 ∧ t > 0 ∧ 
  s^2 = 3 * t ∧ 
  (Real.sqrt 3 / 4) * t^2 = 4 * s ∧ 
  s^2 = 12 * (4^(1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_area_perimeter_relationship_l461_46165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l461_46190

/-- The volume of a cylinder with hemispheres at both ends -/
noncomputable def cylinderWithHemispheresVolume (radius : ℝ) (length : ℝ) : ℝ :=
  Real.pi * radius^2 * length + (4/3) * Real.pi * radius^3

/-- The problem statement -/
theorem segment_length (radius : ℝ) (volume : ℝ) (length : ℝ) :
  radius = 5 →
  volume = 900 * Real.pi →
  cylinderWithHemispheresVolume radius length = volume →
  length = 88/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l461_46190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_monotone_increasing_l461_46121

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x * (2 * Real.cos x) + 3

-- Theorem for the maximum value of f(x)
theorem f_max_value : ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), f x ≤ M := by sorry

-- Theorem for the monotonically increasing interval of f(x)
theorem f_monotone_increasing (k : ℤ) : 
  StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi/4) (k * Real.pi + Real.pi/4)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_monotone_increasing_l461_46121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l461_46182

/-- The eccentricity of an ellipse defined by x^2 + 4y^2 = 1 is √3/2 -/
theorem ellipse_eccentricity : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + 4 * y^2 = 1}
  let a := (1 : ℝ)
  let b := (1/2 : ℝ)
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l461_46182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l461_46195

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/4) + f (x + 3*Real.pi/4)

-- Theorem statement
theorem function_properties :
  (f (Real.pi/2) = 1) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ 2*Real.pi) ∧
  (∀ x, g x ≥ -2) ∧ (∃ x, g x = -2) := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l461_46195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l461_46188

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- Slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- Half of the focal distance -/
  c : ℝ

/-- The standard equation of a hyperbola -/
inductive StandardEquation
  | XAxisFoci (a b : ℝ) : StandardEquation
  | YAxisFoci (a b : ℝ) : StandardEquation

/-- Proposition: A hyperbola has a specific standard equation -/
def HasStandardEquation (h : Hyperbola) (eq : StandardEquation) : Prop :=
  match eq with
  | StandardEquation.XAxisFoci a b => h.asymptote_slope = b / a ∧ h.c ^ 2 = a ^ 2 + b ^ 2
  | StandardEquation.YAxisFoci a b => h.asymptote_slope = a / b ∧ h.c ^ 2 = a ^ 2 + b ^ 2

/-- Theorem: Given a hyperbola with asymptote slope 4/3 and focal distance 20,
    its standard equation is either x²/36 - y²/64 = 1 or y²/64 - x²/36 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) 
    (h_slope : h.asymptote_slope = 4/3)
    (h_focal : h.c = 10) :
    HasStandardEquation h (StandardEquation.XAxisFoci 6 8) ∨
    HasStandardEquation h (StandardEquation.YAxisFoci 8 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l461_46188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_tax_calculation_l461_46107

theorem robot_tax_calculation (initial_money robot_cost : ℚ) (num_robots : ℕ) (change : ℚ) : 
  initial_money = 80 ∧ 
  robot_cost = 8.75 ∧ 
  num_robots = 7 ∧ 
  change = 11.53 → 
  initial_money - change - (robot_cost * num_robots) = 7.22 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_tax_calculation_l461_46107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l461_46110

/-- Represents a circular sector -/
structure Sector where
  radius : ℝ
  arcLength : ℝ

/-- The circumference of a sector -/
noncomputable def Sector.circumference (s : Sector) : ℝ :=
  2 * s.radius + s.arcLength

/-- The area of a sector -/
noncomputable def Sector.area (s : Sector) : ℝ :=
  1/2 * s.radius * s.arcLength

/-- The central angle of a sector in radians -/
noncomputable def Sector.centralAngle (s : Sector) : ℝ :=
  s.arcLength / s.radius

theorem sector_central_angle (s : Sector) 
  (h_circ : s.circumference = 8)
  (h_area : s.area = 4) : 
  s.centralAngle = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_l461_46110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l461_46102

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a parabola -/
def IsOnParabola (p : Point) (vertex focus : Point) : Prop :=
  (p.x - vertex.x)^2 = 4 * (focus.y - vertex.y) * (p.y - vertex.y)

/-- Distance between two points -/
noncomputable def Distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Finding the point on the parabola -/
theorem parabola_point_theorem (p : Point) :
  let vertex : Point := ⟨0, 0⟩
  let focus : Point := ⟨0, 1⟩
  IsOnParabola p vertex focus ∧ 
  p.x ≥ 0 ∧ p.y ≥ 0 ∧
  Distance p focus = 51 →
  p = ⟨10 * Real.sqrt 2, 50⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l461_46102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l461_46176

/-- Represents a hyperbola with equation y²/2 - x² = 1 -/
structure Hyperbola where
  -- The equation y²/2 - x² = 1 is implicitly defined by the structure

/-- Focal length of the hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * Real.sqrt 3

/-- Asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | Real.sqrt 2 * x = y ∨ Real.sqrt 2 * x = -y}

/-- Theorem stating the focal length and asymptotes of the hyperbola -/
theorem hyperbola_properties (h : Hyperbola) :
  (focal_length h = 2 * Real.sqrt 3) ∧
  (asymptotes h = {(x, y) | Real.sqrt 2 * x = y ∨ Real.sqrt 2 * x = -y}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l461_46176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l461_46155

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)

theorem f_properties :
  (∀ x, f (x + Real.pi) = f x) ∧ 
  (∀ x, f (Real.pi/3 + x) = f (Real.pi/3 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l461_46155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_in_S_l461_46151

-- Define the set S
def S : Set ℕ := {n : ℕ | n ≥ 1 ∧ n ≤ 100}

-- Define the conditions for the set
def satisfiesConditions (S : Set ℕ) : Prop :=
  (∀ a ∈ S, ∀ b ∈ S, a ≠ b → ∃ c ∈ S, Nat.gcd a c = 1 ∧ Nat.gcd b c = 1) ∧
  (∀ a ∈ S, ∀ b ∈ S, a ≠ b → ∃ d ∈ S, Nat.gcd a d > 1 ∧ Nat.gcd b d > 1)

-- Theorem statement
theorem max_elements_in_S :
  ∃ (T : Finset ℕ), (↑T : Set ℕ) ⊆ S ∧ satisfiesConditions ↑T ∧ T.card = 72 ∧
  ∀ (U : Finset ℕ), (↑U : Set ℕ) ⊆ S → satisfiesConditions ↑U → U.card ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_in_S_l461_46151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l461_46193

noncomputable def expression : ℝ := Real.sqrt ((Real.log 8 / Real.log 4) + (Real.log 4 / Real.log 8))

theorem expression_value : expression = Real.sqrt (13 / 6) := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l461_46193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_n_b_odd_sum_l461_46150

def a : ℕ → ℕ := sorry

def S : ℕ → ℕ := sorry

axiom a_1 : a 1 = 1

axiom S_2n (n : ℕ) : S (2 * n) = 2 * (a n)^2 + a n

def b (n : ℕ) : ℕ := 2^(a n)

theorem a_eq_n (n : ℕ) : a n = n := by sorry

theorem b_odd_sum (n : ℕ) : 
  (Finset.range (n + 1)).sum (fun i => b (2 * i + 1)) = (2 / 3) * (4^(n + 1) - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_n_b_odd_sum_l461_46150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_from_cyclic_inequalities_l461_46130

def cyclicIndex (i n : Nat) : Nat :=
  ((i - 1) % n) + 1

theorem equality_from_cyclic_inequalities (n : Nat) (x : Fin n → ℝ) 
  (h : ∀ i : Fin n, 2 * x i - 5 * x ⟨cyclicIndex (i.val + 1) n, by sorry⟩ + 3 * x ⟨cyclicIndex (i.val + 2) n, by sorry⟩ ≥ 0) :
  ∀ i j : Fin n, x i = x j :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_from_cyclic_inequalities_l461_46130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_existence_l461_46120

theorem difference_existence (S : Finset ℕ) (h1 : S.card = 70) (h2 : ∀ n ∈ S, n ≤ 200) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b = 4 ∨ a - b = 5 ∨ a - b = 9 ∨ b - a = 4 ∨ b - a = 5 ∨ b - a = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_existence_l461_46120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_binomial_coefficients_l461_46184

theorem infinite_coprime_binomial_coefficients (k l : ℕ+) :
  ∃ (S : Set ℕ), (∀ m ∈ S, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_binomial_coefficients_l461_46184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l461_46171

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (f (1 + x) + f (1 - x^2) ≥ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l461_46171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l461_46152

/-- The angle of the minute hand at a given number of minutes past the hour -/
def minuteHandAngle (minutes : ℕ) : ℝ := 6 * minutes

/-- The angle of the hour hand at a given number of hours and minutes past 12:00 -/
def hourHandAngle (hours minutes : ℕ) : ℝ := 30 * hours + 0.5 * minutes

/-- The smaller angle between two angles on a clock face -/
noncomputable def smallerAngle (angle1 angle2 : ℝ) : ℝ :=
  min (abs (angle1 - angle2)) (360 - abs (angle1 - angle2))

theorem clock_angle_at_3_40 :
  smallerAngle (minuteHandAngle 40) (hourHandAngle 3 40) = 130 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l461_46152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l461_46101

open Real

/-- Given a triangle ABC with centroid G and any point P on the plane, 
    the sum of squared distances from P to the vertices of the triangle 
    is equal to the sum of squared distances from G to the vertices 
    plus three times the squared distance from G to P. -/
theorem centroid_distance_relation 
  (A B C G P : ℝ × ℝ) 
  (h_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) : 
  let dist_squared := λ X Y : ℝ × ℝ ↦ (X.1 - Y.1)^2 + (X.2 - Y.2)^2
  dist_squared P A + dist_squared P B + dist_squared P C = 
  dist_squared G A + dist_squared G B + dist_squared G C + 3 * dist_squared G P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_relation_l461_46101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l461_46180

theorem possible_values_of_a (a b c d e : ℝ) 
  (eq1 : a * b + a * c + a * d + a * e = -1)
  (eq2 : b * c + b * d + b * e + b * a = -1)
  (eq3 : c * d + c * e + c * a + c * b = -1)
  (eq4 : d * e + d * a + d * b + d * c = -1)
  (eq5 : e * a + e * b + e * c + e * d = -1) :
  a = Real.sqrt 2 / 2 ∨ a = -(Real.sqrt 2 / 2) ∨ a = Real.sqrt 2 ∨ a = -(Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_values_of_a_l461_46180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equivalence_l461_46131

theorem expression_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) :
  (1 / (x^2 * y^2)) / (1 / x^4 - 1 / y^4) = (x^2 * y^2) / (y^4 - x^4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equivalence_l461_46131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_geq_three_halves_l461_46179

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1/2) * Real.cos (2*x) - 2*a*(Real.sin x + Real.cos x) + (4*a - 3)*x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 
  -Real.sin (2*x) - 2*a*(Real.cos x - Real.sin x) + 4*a - 3

theorem monotone_increasing_f_implies_a_geq_three_halves :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), MonotoneOn (f a) (Set.Icc 0 (Real.pi / 2))) →
  a ≥ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_geq_three_halves_l461_46179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_floating_properties_l461_46157

/-- Represents a solid, homogeneous sphere --/
structure Sphere where
  radius : ℝ
  density : ℝ
  density_pos : 0 < density
  density_lt_one : density < 1

/-- The apex angle of a spherical sector cut from the sphere that floats with its conical surface contacting water --/
noncomputable def apex_angle (s : Sphere) : ℝ :=
  Real.arccos ((Real.sqrt (1 + 8 * s.density) - 1) / 2)

/-- The distance from the water surface to the apex of a spherical sector cut from the sphere that floats with its spherical cap in water --/
noncomputable def apex_distance (s : Sphere) : ℝ :=
  |s.radius - (s.radius / 2) * (3 + Real.sqrt (9 - 8 * s.density))|

theorem sphere_floating_properties (s : Sphere) :
  (apex_angle s = Real.arccos ((Real.sqrt (1 + 8 * s.density) - 1) / 2)) ∧
  (apex_distance s = |s.radius - (s.radius / 2) * (3 + Real.sqrt (9 - 8 * s.density))|) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_floating_properties_l461_46157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l461_46134

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 - t, t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the general equation of line l
def line_l_general (x y : ℝ) : Prop := x + y = 1

-- Theorem statement
theorem intersection_distance :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  curve_C_cartesian (line_l t₁).1 (line_l t₁).2 ∧
  curve_C_cartesian (line_l t₂).1 (line_l t₂).2 ∧
  (line_l t₁).1 + (line_l t₁).2 = 1 ∧
  (line_l t₂).1 + (line_l t₂).2 = 1 ∧
  ((line_l t₁).1 - (line_l t₂).1)^2 + ((line_l t₁).2 - (line_l t₂).2)^2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l461_46134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_width_l461_46163

/-- Represents the width of a rectangular garden. -/
noncomputable def width : ℝ := sorry

/-- Represents the length of a rectangular garden. -/
noncomputable def length : ℝ := width + 20

/-- The area of the garden is at least 150 sq. ft. -/
axiom area_constraint : width * length ≥ 150

/-- The width is non-negative. -/
axiom width_non_negative : width ≥ 0

/-- The minimum width that satisfies the constraints is 5 ft. -/
theorem min_width : width ≥ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_width_l461_46163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cotton_iron_mass_equality_l461_46181

/-- Represents the mass of a substance in kilograms. -/
def Mass : Type := ℝ

/-- The mass of 1 kilogram of cotton. -/
def cotton_mass : Mass := (1 : ℝ)

/-- The mass of 1 kilogram of iron. -/
def iron_mass : Mass := (1 : ℝ)

/-- Theorem stating that the mass of 1 kilogram of cotton is equal to the mass of 1 kilogram of iron. -/
theorem cotton_iron_mass_equality : cotton_mass = iron_mass := by
  rfl

#check cotton_iron_mass_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cotton_iron_mass_equality_l461_46181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l461_46126

theorem min_value_trig_expression (α : Real) (h : α ∈ Set.Ioo 0 (π / 2)) :
  (Real.sin α)^3 / Real.cos α + (Real.cos α)^3 / Real.sin α ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l461_46126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l461_46108

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 5 = 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 2) + 4

-- Define the vertical line case
def vertical_line (x : ℝ) : Prop := x = 2

-- Define the chord length
def chord_length : ℝ := 6

-- Theorem statement
theorem line_equation :
  ∀ k : ℝ,
  (∃ x y : ℝ, line_l k x y ∧ circle_C x y) →
  (∀ x y : ℝ, line_l k x y → circle_C x y → 
    ∃ x' y' : ℝ, line_l k x' y' ∧ circle_C x' y' ∧ (x - x')^2 + (y - y')^2 = chord_length^2) →
  (k = 3/4 ∨ vertical_line 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l461_46108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l461_46154

-- Define the angle function
def angle (c : ℝ) : ℝ := c^2 - 3*c + 17

-- State the theorem
theorem find_c : 
  ∃ (c b : ℝ),
    Real.sin (angle c * π / 180) = 4 / (b - 2) ∧
    0 < angle c ∧ 
    angle c < 90 ∧
    c > 0 ∧
    c = 7 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l461_46154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_10_equals_4_l461_46140

def T (n : ℕ) : ℕ := 
  if n ≥ 2 then (n * (n + 1)) / 2 - 1 else 0

noncomputable def P (n : ℕ) : ℚ :=
  if n ≥ 3 then 
    (Finset.range (n - 2)).prod (fun k => (T (k + 3) : ℚ) / ((T (k + 3) : ℚ) - 1))
  else 0

theorem P_10_equals_4 : P 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_10_equals_4_l461_46140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_extrema_l461_46177

noncomputable def f (x : ℝ) := 220 * Real.sin (100 * Real.pi * x) - 220 * Real.sin (100 * Real.pi * x + 2 * Real.pi / 3)

theorem min_difference_of_extrema (x₁ x₂ : ℝ) :
  (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  ∃ y₁ y₂, |y₂ - y₁| = (1 : ℝ) / 100 ∧ ∀ z₁ z₂, (∀ x, f z₁ ≤ f x ∧ f x ≤ f z₂) → |y₂ - y₁| ≤ |z₂ - z₁| :=
by
  sorry

#check min_difference_of_extrema

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_of_extrema_l461_46177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficients_equal_min_n_for_constant_term_l461_46148

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression (2x^3 + 1/x^2)^n
noncomputable def expression (x : ℝ) (n : ℕ) : ℝ := (2 * x^3 + 1/x^2)^n

-- Theorem 1: Binomial coefficients of 5th and 6th terms are equal when n = 9
theorem binomial_coefficients_equal :
  ∀ n : ℕ, n > 0 → (binomial n 4 = binomial n 5) → n = 9 := by
  sorry

-- Theorem 2: Minimum n for constant term is 5
theorem min_n_for_constant_term :
  ∀ n : ℕ, n > 0 → 
  (∃ r : ℕ, 3*n = 5*r) → 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬∃ r : ℕ, 3*m = 5*r) →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficients_equal_min_n_for_constant_term_l461_46148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_ellipse_transform_l461_46124

/-- A matrix that transforms the unit circle to an ellipse -/
def TransformMatrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, 0; 0, b]

/-- The condition that a and b are positive -/
def PositiveParams (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

/-- The condition that the matrix transforms the unit circle to the ellipse -/
def TransformsCircleToEllipse (A : Matrix (Fin 2) (Fin 2) ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 = 1 → 
    (let x' := A 0 0 * x + A 0 1 * y
     let y' := A 1 0 * x + A 1 1 * y
     x'^2 / a^2 + y'^2 / b^2 = 1)

/-- The main theorem -/
theorem circle_to_ellipse_transform 
  (a b : ℝ) 
  (h1 : PositiveParams a b)
  (h2 : TransformsCircleToEllipse (TransformMatrix a b) a b) :
  a = 2 ∧ b = 1/2 ∧ 
  (TransformMatrix a b)⁻¹ = !![1/2, 0; 0, 2] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_ellipse_transform_l461_46124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l461_46173

theorem cosine_sum_product_form :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, Real.cos (2*x) + Real.cos (6*x) + Real.cos (10*x) + Real.cos (14*x) = 
     (a : ℝ) * Real.cos (b*x) * Real.cos (c*x) * Real.cos (d*x)) ∧
    a + b + c + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l461_46173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_squared_sum_eq_two_l461_46139

theorem sine_squared_sum_eq_two (z : ℝ) :
  (Real.sin (2 * z))^2 + (Real.sin (3 * z))^2 + (Real.sin (4 * z))^2 + (Real.sin (5 * z))^2 = 2 →
  (∃ n : ℤ, z = (π / 14 : ℝ) * (2 * ↑n + 1)) ∨ 
  (∃ m : ℤ, z = (π / 4 : ℝ) * (2 * ↑m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_squared_sum_eq_two_l461_46139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l461_46144

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- A circle with center (x, y) and radius r -/
structure Circle where
  x : ℝ
  y : ℝ
  r : ℝ

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem hyperbola_circle_intersection (h : Hyperbola) (c : Circle) :
  eccentricity h = Real.sqrt 5 →
  c.x = 2 ∧ c.y = 3 ∧ c.r = 1 →
  ∃ (A B : ℝ × ℝ), 
    (A.1 - 2)^2 + (A.2 - 3)^2 = 1 ∧
    (B.1 - 2)^2 + (B.2 - 3)^2 = 1 ∧
    distance A.1 A.2 B.1 B.2 = 4 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l461_46144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sines_equal_implies_parallelogram_or_trapezoid_l461_46113

/-- A convex quadrilateral with vertices A, B, C, D -/
structure ConvexQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The angle at vertex A in a quadrilateral -/
noncomputable def angle_A (q : ConvexQuadrilateral) : ℝ := sorry

/-- The angle at vertex B in a quadrilateral -/
noncomputable def angle_B (q : ConvexQuadrilateral) : ℝ := sorry

/-- The angle at vertex C in a quadrilateral -/
noncomputable def angle_C (q : ConvexQuadrilateral) : ℝ := sorry

/-- The angle at vertex D in a quadrilateral -/
noncomputable def angle_D (q : ConvexQuadrilateral) : ℝ := sorry

/-- Predicate to check if a quadrilateral is a parallelogram -/
def is_parallelogram (q : ConvexQuadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is a trapezoid -/
def is_trapezoid (q : ConvexQuadrilateral) : Prop := sorry

/-- Theorem: If the sum of sines of opposite angles in a convex quadrilateral are equal,
    then the quadrilateral is either a parallelogram or a trapezoid -/
theorem sum_of_sines_equal_implies_parallelogram_or_trapezoid (q : ConvexQuadrilateral) :
  Real.sin (angle_A q) + Real.sin (angle_C q) = Real.sin (angle_B q) + Real.sin (angle_D q) →
  is_parallelogram q ∨ is_trapezoid q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sines_equal_implies_parallelogram_or_trapezoid_l461_46113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_is_pi_over_two_l461_46122

/-- A regular octagon in the complex plane with center at the origin -/
structure RegularOctagon where
  /-- The distance between opposite sides of the octagon -/
  opposite_sides_distance : ℝ
  /-- One pair of sides is parallel to the real axis -/
  parallel_to_real_axis : Prop

/-- The region outside the octagon -/
def R (octagon : RegularOctagon) : Set ℂ :=
  {z : ℂ | ¬(z ∈ Set.univ)} -- Placeholder definition

/-- The transformation z ↦ 1/z -/
noncomputable def inverse_transform (z : ℂ) : ℂ := 1 / z

/-- The set S, which is the image of R under the inverse transform -/
noncomputable def S (octagon : RegularOctagon) : Set ℂ :=
  {w : ℂ | ∃ z ∈ R octagon, w = inverse_transform z}

/-- The area of a set in the complex plane -/
noncomputable def area (s : Set ℂ) : ℝ := sorry

theorem area_of_S_is_pi_over_two (octagon : RegularOctagon) 
  (h : octagon.opposite_sides_distance = 2) :
  area (S octagon) = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_is_pi_over_two_l461_46122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_markup_is_40_percent_l461_46142

/-- Represents the markup percentage applied by a merchant -/
def markup_percentage : ℝ → Prop := sorry

/-- Represents the discount percentage applied to the marked price -/
def discount_percentage : ℝ → Prop := sorry

/-- Represents the profit percentage after discount -/
def profit_percentage : ℝ → Prop := sorry

/-- Calculates the marked price given the cost price and markup percentage -/
noncomputable def marked_price (cost_price : ℝ) (markup : ℝ) : ℝ :=
  cost_price * (1 + markup / 100)

/-- Calculates the selling price after discount -/
noncomputable def selling_price (marked_price : ℝ) (discount : ℝ) : ℝ :=
  marked_price * (1 - discount / 100)

/-- Theorem: The initial markup percentage is 40% -/
theorem initial_markup_is_40_percent (cost_price : ℝ) :
  cost_price > 0 →
  markup_percentage 40 →
  discount_percentage 10 →
  profit_percentage 26 →
  let mp := marked_price cost_price 40
  let sp := selling_price mp 10
  sp = cost_price * (1 + 26 / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_markup_is_40_percent_l461_46142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_return_speed_l461_46137

/-- Calculates the average speed of the return trip given the conditions of Cole's journey -/
theorem cole_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : 
  speed_to_work = 75 →
  total_time = 1 →
  time_to_work = 35 / 60 →
  let distance := speed_to_work * time_to_work
  let time_return := total_time - time_to_work
  let speed_return := distance / time_return
  speed_return = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_return_speed_l461_46137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l461_46128

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π/4 - α) = Real.sqrt 2/4) : 
  Real.sin (2*α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l461_46128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l461_46168

noncomputable def f (x : ℝ) : ℝ := (x^2 + Real.sin (2*x)) / Real.exp x

theorem derivative_of_f :
  deriv f = λ x => (2*(1 + Real.cos (2*x)) - 2*x - Real.sin (2*x)) / Real.exp x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l461_46168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l461_46106

theorem cos_alpha_plus_pi_third (α : Real) 
  (h1 : Real.sin α = 1/3) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.cos (α + π/3) = -(2*Real.sqrt 2 + Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l461_46106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_in_range_l461_46194

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*m*x + 3) / Real.log (1/2)

theorem f_increasing_iff_m_in_range (m : ℝ) :
  (∀ x y, x < y ∧ y < 1 → f m x < f m y) ↔ m ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_in_range_l461_46194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_not_sufficient_for_congruence_l461_46135

-- Define two triangles
variable (A B C A' B' C' : EuclideanSpace ℝ (Fin 2))

-- Define the angles of the triangles
noncomputable def angle_A : ℝ := sorry
noncomputable def angle_B : ℝ := sorry
noncomputable def angle_C : ℝ := sorry
noncomputable def angle_A' : ℝ := sorry
noncomputable def angle_B' : ℝ := sorry
noncomputable def angle_C' : ℝ := sorry

-- Define congruence of triangles
def triangles_congruent (A B C A' B' C' : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Theorem statement
theorem angles_not_sufficient_for_congruence :
  angle_A = angle_A' ∧ angle_B = angle_B' ∧ angle_C = angle_C' →
  ¬(∀ A B C A' B' C', triangles_congruent A B C A' B' C') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_not_sufficient_for_congruence_l461_46135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l461_46164

/-- The distance covered by a wheel with given diameter and number of revolutions -/
noncomputable def distance_covered (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  Real.pi * diameter * revolutions

/-- Theorem: A wheel with diameter 14 cm making 33.03002729754322 revolutions covers 1452.996 cm -/
theorem wheel_distance_theorem :
  let diameter := (14 : ℝ)
  let revolutions := (33.03002729754322 : ℝ)
  abs (distance_covered diameter revolutions - 1452.996) < 0.001 := by
  -- Unfold the definitions
  unfold distance_covered
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l461_46164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_price_calculation_l461_46178

/-- Calculates the selling price of a scooter given its purchase price, repair costs, and gain percent. -/
noncomputable def scooter_selling_price (purchase_price repair_costs : ℚ) (gain_percent : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_costs
  let gain := (gain_percent / 100) * total_cost
  total_cost + gain

/-- Theorem stating that for a scooter with a purchase price of $800, repair costs of $200,
    and a gain percent of 20%, the selling price is $1200. -/
theorem scooter_price_calculation :
  scooter_selling_price 800 200 20 = 1200 := by
  -- Unfold the definition of scooter_selling_price
  unfold scooter_selling_price
  -- Simplify the arithmetic
  simp [add_assoc, mul_add, mul_comm, mul_div_cancel']
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_price_calculation_l461_46178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_equals_cristina_pace_l461_46119

/-- A race between two runners with a head start -/
structure Race where
  length : ℚ
  head_start : ℚ
  cristina_pace : ℚ
  catch_up_time : ℚ

/-- Calculate Nicky's pace given the race parameters -/
def nicky_pace (r : Race) : ℚ :=
  r.length / (r.catch_up_time + r.head_start)

/-- Theorem stating that Nicky's pace equals Cristina's pace under given conditions -/
theorem nicky_pace_equals_cristina_pace (r : Race) 
  (h1 : r.length = 300)
  (h2 : r.head_start = 12)
  (h3 : r.cristina_pace = 5)
  (h4 : r.catch_up_time = 30) :
  nicky_pace r = r.cristina_pace := by
  sorry

/-- Evaluate Nicky's pace for the given race parameters -/
def main : IO Unit := do
  let race : Race := { 
    length := 300, 
    head_start := 12, 
    cristina_pace := 5, 
    catch_up_time := 30 
  }
  IO.println s!"Nicky's pace: {nicky_pace race}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_pace_equals_cristina_pace_l461_46119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l461_46118

-- Define the circles
def circle_M₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 2
def circle_M₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y + 1 = 0

-- Define the centers of the circles
def M₁ : ℝ × ℝ := (2, 1)
def M₂ : ℝ × ℝ := (1, -1)

-- Define a point on the y-axis
def P (t : ℝ) : ℝ × ℝ := (0, t)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (t : ℝ), ∀ (s : ℝ), 
    distance (P t) M₁ + distance (P t) M₂ ≤ distance (P s) M₁ + distance (P s) M₂ ∧
    distance (P t) M₁ + distance (P t) M₂ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l461_46118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l461_46162

noncomputable section

-- Function 1
def f1 (x : ℝ) : ℝ := x^(-1/4 : ℝ)

-- Function 2
def f2 (x : ℝ) : ℝ := x / (x + 1)

-- Function 3
def f3 (x : ℝ) : ℝ := x + 9/x

theorem problem_solution :
  (∀ x y, 0 < x ∧ x < y → f1 y < f1 x) ∧
  (∀ x y, -1 < x ∧ x < y → f2 x < f2 y) ∧
  (∀ a b, 0 < a ∧ a < b → 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f3 y ≤ f3 x) →
    (∀ x y, -b ≤ x ∧ x < y ∧ y ≤ -a → f3 y ≤ f3 x)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l461_46162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_ratio_is_one_to_two_l461_46192

/- Define the problem parameters -/
def thread_per_keychain : ℕ := 12
def class_friends : ℕ := 6
def total_thread : ℕ := 108

/- Define the function to calculate the ratio -/
def friend_ratio : ℚ := by
  /- Calculate total keychains -/
  let total_keychains := total_thread / thread_per_keychain
  /- Calculate after-school club friends -/
  let club_friends := total_keychains - class_friends
  /- Return the ratio as a rational number -/
  exact (club_friends : ℚ) / (class_friends : ℚ)

/- State the theorem -/
theorem friend_ratio_is_one_to_two : friend_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_ratio_is_one_to_two_l461_46192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_condition_l461_46141

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  ab_nonzero : a * b ≠ 0

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.b

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.a

/-- A line has equal intercepts if its x-intercept equals its y-intercept -/
def has_equal_intercepts (l : Line) : Prop :=
  x_intercept l = y_intercept l

/-- 
Theorem: A line has equal intercepts on both coordinate axes 
if and only if c = 0 or (c ≠ 0 and a = b)
-/
theorem equal_intercepts_condition (l : Line) : 
  has_equal_intercepts l ↔ (l.c = 0 ∨ (l.c ≠ 0 ∧ l.a = l.b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_condition_l461_46141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_traffic_flow_range_l461_46105

/-- The traffic flow function --/
noncomputable def y (v : ℝ) : ℝ := 92 * v / (v^2 + 3*v + 1600)

/-- The maximum traffic flow occurs at v = 40 --/
theorem max_traffic_flow :
  ∀ v : ℝ, v > 0 → y v ≤ y 40 := by sorry

/-- The range of v that ensures traffic flow is at least 1 --/
theorem traffic_flow_range :
  ∀ v : ℝ, v > 0 → (y v ≥ 1 ↔ 25 ≤ v ∧ v ≤ 64) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_traffic_flow_range_l461_46105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l461_46149

theorem angle_relation (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan α = (1 + Real.sin β) / Real.cos β) : 
  2 * α - β = π/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l461_46149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_hyperbola_K_unique_l461_46183

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (K : ℝ) : ℝ := Real.sqrt (|K| + |K/2|)

/-- Theorem: For a hyperbola x^2 - 2y^2 = K with focal distance 6, K = ±6 -/
theorem hyperbola_focal_distance (K : ℝ) : 
  focal_distance K = 6 → K = 6 ∨ K = -6 := by
  sorry

/-- Corollary: The value of K is unique up to sign -/
theorem hyperbola_K_unique (K₁ K₂ : ℝ) :
  focal_distance K₁ = 6 → focal_distance K₂ = 6 → K₁ = K₂ ∨ K₁ = -K₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_hyperbola_K_unique_l461_46183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_population_l461_46197

def total_rabbits (n : ℕ) : ℚ :=
  (5 * 2^(n+2) - 5 * (-1)^n - 3) / 6

theorem rabbit_population (n : ℕ) :
  ∃ (a b : ℕ → ℕ),
    a 0 = 1 ∧ a 1 = 4 ∧ b 0 = 1 ∧ b 1 = 3 ∧
    (∀ k ≥ 2, a k = a (k-1) + 3 * b (k-2)) ∧
    (∀ k ≥ 2, b k = b (k-1) + 2 * b (k-2)) ∧
    (a n + b n = total_rabbits n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_population_l461_46197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_ratio_l461_46103

theorem partnership_investment_ratio :
  ∀ (a b c : ℚ) (total_profit b_share : ℚ),
    a = 3 * b →                   -- A invests 3 times as much as B
    total_profit = 5500 →         -- Total profit is 5500
    b_share = 1000 →              -- B's share is 1000
    b_share / total_profit = b / (a + b + c) →  -- B's share proportion
    b / c = 2 / 3 :=              -- Ratio of B's to C's investment
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_investment_ratio_l461_46103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_cone_height_l461_46172

/-- The volume of a sphere -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cone -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

/-- Theorem: The height of a cone formed by recasting a sphere -/
theorem sphere_to_cone_height 
  (sphere_diameter : ℝ) 
  (cone_base_diameter : ℝ) 
  (h : ℝ) :
  sphere_diameter = 6 →
  cone_base_diameter = 12 →
  sphere_volume (sphere_diameter / 2) = cone_volume (cone_base_diameter / 2) h →
  h = 3 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_cone_height_l461_46172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_l461_46156

-- Define the set of available digits
def AvailableDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the structure of the grid
structure Grid where
  vertical : Fin 4 → Nat
  horizontal : Fin 5 → Nat
  intersection1 : vertical 1 = horizontal 1
  intersection2 : vertical 2 = horizontal 2

-- Define the conditions
def ValidGrid (g : Grid) : Prop :=
  (∀ i j, i ≠ j → g.vertical i ≠ g.vertical j) ∧
  (∀ i j, i ≠ j → g.horizontal i ≠ g.horizontal j) ∧
  (∀ i, g.vertical i ∈ AvailableDigits) ∧
  (∀ i, g.horizontal i ∈ AvailableDigits) ∧
  (g.vertical 0 + g.vertical 1 + g.vertical 2 + g.vertical 3 = 30) ∧
  (g.horizontal 0 + g.horizontal 1 + g.horizontal 2 + g.horizontal 3 + g.horizontal 4 = 25)

-- Theorem statement
theorem sum_of_digits (g : Grid) (h : ValidGrid g) :
  ∃ (digits : Finset Nat),
    digits ⊆ AvailableDigits ∧
    digits.card = 7 ∧
    (∀ i, g.vertical i ∈ digits) ∧
    (∀ i, g.horizontal i ∈ digits) ∧
    Finset.sum digits id = 33 := by
  sorry

#check sum_of_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_l461_46156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_distance_at_blast_sound_l461_46174

-- Constants from the problem
def blast_time : ℝ := 45
def powderman_speed_yards : ℝ := 10
def sound_speed_feet : ℝ := 1200
def reaction_time : ℝ := 2

-- Convert powderman's speed to feet per second
def powderman_speed_feet : ℝ := powderman_speed_yards * 3

-- Function for powderman's distance over time
def powderman_distance (t : ℝ) : ℝ := powderman_speed_feet * t

-- Function for sound's distance over time (after blast)
def sound_distance (t : ℝ) : ℝ := sound_speed_feet * (t - blast_time)

-- Theorem to prove
theorem powderman_distance_at_blast_sound : 
  ∃ t : ℝ, t > blast_time + reaction_time ∧ 
  powderman_distance t = sound_distance t ∧ 
  Int.floor (powderman_distance t / 3) = 461 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_powderman_distance_at_blast_sound_l461_46174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangents_intersect_on_omega_l461_46104

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the structure for a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the structure for a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define the structure for a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of being convex
def is_convex (q : Quadrilateral) : Prop := sorry

-- Define the property of a circle being tangent to a line segment
def is_tangent_to_segment (c : Circle) (A B : Point) : Prop := sorry

-- Define the property of a circle being an incircle of a triangle
def is_incircle (c : Circle) (A B C : Point) : Prop := sorry

-- Define the external common tangents of two circles
def external_common_tangents (c1 c2 : Circle) : Set Line := sorry

-- Define the intersection point of two lines
noncomputable def intersection_point (l1 l2 : Line) : Point := sorry

-- Define the boundary of a circle
def Circle.boundary (c : Circle) : Set Point := sorry

-- Main theorem
theorem external_tangents_intersect_on_omega 
  (ABCD : Quadrilateral) 
  (ω ω₁ ω₂ : Circle) 
  (h_convex : is_convex ABCD)
  (h_ω_tangent_AB : is_tangent_to_segment ω ABCD.A ABCD.B)
  (h_ω_tangent_BC : is_tangent_to_segment ω ABCD.B ABCD.C)
  (h_ω_tangent_AD : is_tangent_to_segment ω ABCD.A ABCD.D)
  (h_ω_tangent_DC : is_tangent_to_segment ω ABCD.D ABCD.C)
  (h_ω₁_incircle : is_incircle ω₁ ABCD.A ABCD.B ABCD.C)
  (h_ω₂_incircle : is_incircle ω₂ ABCD.A ABCD.D ABCD.C) :
  ∃ (P : Point), P ∈ ω.boundary ∧ 
    ∃ (l1 l2 : Line), l1 ∈ external_common_tangents ω₁ ω₂ ∧ 
                      l2 ∈ external_common_tangents ω₁ ω₂ ∧
                      P = intersection_point l1 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangents_intersect_on_omega_l461_46104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_15_degrees_radius_6_l461_46147

-- Define the radius
def radius : ℝ := 6

-- Define the central angle in degrees
def central_angle_degrees : ℝ := 15

-- Define the central angle in radians
noncomputable def central_angle_radians : ℝ := (Real.pi / 180) * central_angle_degrees

-- Define the sector area function
noncomputable def sector_area (r : ℝ) (α : ℝ) : ℝ := (1 / 2) * α * r^2

-- Theorem statement
theorem sector_area_15_degrees_radius_6 :
  sector_area radius central_angle_radians = (3 * Real.pi) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_15_degrees_radius_6_l461_46147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_matrix_adds_three_to_top_right_zero_matrix_is_answer_l461_46185

theorem no_matrix_adds_three_to_top_right : 
  ¬ ∃ (M : Matrix (Fin 2) (Fin 2) ℝ), 
    ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), 
      (M • A) = ![![A 0 0, A 0 1 + 3], ![A 1 0, A 1 1]] := by sorry

theorem zero_matrix_is_answer : 
  (∀ (A : Matrix (Fin 2) (Fin 2) ℝ), 
    ((0 : Matrix (Fin 2) (Fin 2) ℝ) • A) ≠ ![![A 0 0, A 0 1 + 3], ![A 1 0, A 1 1]]) → 
  (0 : Matrix (Fin 2) (Fin 2) ℝ) = 
    ![![(0 : ℝ), 0], ![0, 0]] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_matrix_adds_three_to_top_right_zero_matrix_is_answer_l461_46185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_981_l461_46199

/-- A function that converts a natural number to its ternary representation -/
def toTernary (n : ℕ) : List ℕ := sorry

/-- A function that checks if a ternary representation contains only 0 and 1 -/
def isValidTernary (l : List ℕ) : Bool :=
  l.all (λ x => x = 0 ∨ x = 1)

/-- A function that converts a ternary representation to a natural number -/
def fromTernary (l : List ℕ) : ℕ := sorry

/-- The sequence of numbers whose ternary representation contains only 0 and 1 -/
def validTernarySequence : List ℕ :=
  (List.range (2^100)).filter (λ n => isValidTernary (toTernary n))

theorem hundredth_term_is_981 :
  validTernarySequence[99]? = some 981 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_981_l461_46199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_circumcenter_distance_l461_46117

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define properties of the triangle
def isIsoscelesRight (t : Triangle) : Prop :=
  t.A.1 = 0 ∧ t.A.2 = 0 ∧
  t.B.1 = 6 ∧ t.B.2 = 0 ∧
  t.C.1 = 0 ∧ t.C.2 = 6

-- Define the circumcenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ :=
  ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem isosceles_right_triangle_circumcenter_distance (t : Triangle) :
  isIsoscelesRight t →
  distance (circumcenter t) t.B = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_circumcenter_distance_l461_46117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l461_46145

-- Define the interval (1/10, 1)
def I : Set ℝ := {x | 1/10 < x ∧ x < 1}

-- State the theorem
theorem log_inequality (m : ℝ) (h : m ∈ I) :
  Real.log m ^ 3 > Real.log m ∧ Real.log m > Real.log (m^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l461_46145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_empty_iff_exists_complement_subset_l461_46166

universe u

theorem set_intersection_empty_iff_exists_complement_subset {U : Type u} (A B : Set U) :
  A ∩ B = ∅ ↔ ∃ C : Set U, A ⊆ C ∧ B ⊆ Cᶜ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_empty_iff_exists_complement_subset_l461_46166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_iff_specific_a_l461_46170

-- Define the equation
noncomputable def f (x a : ℝ) : ℝ :=
  (8:ℝ)^(|x - a|) * (Real.log (x^2 + 2*x + 5) / Real.log 5) +
  (2:ℝ)^(x^2 + 2*x) * (Real.log (3 * |x - a| + 4) / Real.log (Real.sqrt 5))

-- Theorem statement
theorem three_solutions_iff_specific_a :
  ∀ a : ℝ, (∃! x₁ x₂ x₃ : ℝ, f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ↔
  (a = -7/4 ∨ a = -1 ∨ a = -1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_iff_specific_a_l461_46170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_simplified_l461_46198

-- Define the expression as a function of y
noncomputable def expression (y : ℝ) : ℝ := Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y)

-- Define the simplified form
noncomputable def simplified (y : ℝ) : ℝ := 30 * y * Real.sqrt (12 * y)

-- State the theorem
theorem expression_equals_simplified (y : ℝ) (h : y > 0) : expression y = simplified y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_simplified_l461_46198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_reason_for_slow_growth_l461_46123

-- Define the context
def pre_industrial_revolution : Prop := True

-- Define the conditions
def low_social_productivity : Prop := True
def high_birth_rate : Prop := True
def high_mortality_rate : Prop := True

-- Define the natural population growth rate
def natural_population_growth_rate : ℝ → Prop := λ _ => True
def low_natural_population_growth_rate : Prop := ∃ r : ℝ, natural_population_growth_rate r ∧ r < 0.01

-- Define population growth rate
def population_growth_rate : ℝ → Prop := λ _ => True
def slow_population_growth : Prop := ∃ r : ℝ, population_growth_rate r ∧ r < 0.005

-- Theorem statement
theorem fundamental_reason_for_slow_growth 
  (h1 : pre_industrial_revolution)
  (h2 : low_social_productivity)
  (h3 : high_birth_rate)
  (h4 : high_mortality_rate) :
  low_natural_population_growth_rate → slow_population_growth :=
by
  intro h5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundamental_reason_for_slow_growth_l461_46123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_fourth_power_derivative_l461_46158

open Set
open MeasureTheory
open Interval
open Real

theorem integral_f_fourth_power_derivative (f : ℝ → ℝ) (hf : Continuous f) :
  (∫ x in Icc 0 1, f x * (deriv f x)) = 0 →
  (∫ x in Icc 0 1, (f x)^2 * (deriv f x)) = 18 →
  (∫ x in Icc 0 1, (f x)^4 * (deriv f x)) = 486/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_fourth_power_derivative_l461_46158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l461_46187

theorem sin_double_angle_special_case (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)  -- α is in the first quadrant
  (h2 : Real.sin α = 3 / 5) :    -- sin α = 3/5
  Real.sin (2 * α) = 24 / 25 :=  -- sin 2α = 24/25
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l461_46187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_in_53_l461_46138

theorem remainder_six_in_53 : ∃! n : ℕ, 53 % n = 6 ∧ n > 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_in_53_l461_46138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l461_46159

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 4)

theorem f_properties :
  (∀ x, f x = 3 * Real.cos (2 * x - Real.pi / 4)) ∧
  (∀ x, f (-Real.pi / 8 + x) = f (-Real.pi / 8 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l461_46159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cube_bisection_pieces_l461_46169

/-- Represents a three-dimensional unit cube. -/
structure UnitCube where
  -- No specific fields needed for a unit cube

/-- Represents a plane that bisects the cube. -/
structure BisectingPlane where
  -- No specific fields needed for this abstract representation

/-- Predicate to check if a plane is a perpendicular bisector of two points. -/
def is_perpendicular_bisector (p : BisectingPlane) (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Counts the number of pieces resulting from cutting a unit cube with bisecting planes. -/
def count_pieces (cube : UnitCube) (planes : List BisectingPlane) : ℕ :=
  sorry

/-- Theorem stating that cutting a unit cube with all possible bisecting planes results in 96 pieces. -/
theorem unit_cube_bisection_pieces :
  ∀ (cube : UnitCube) (all_planes : List BisectingPlane),
  (∀ (v1 v2 : ℝ × ℝ × ℝ), v1 ≠ v2 → ∃ (p : BisectingPlane), p ∈ all_planes ∧ is_perpendicular_bisector p v1 v2) →
  count_pieces cube all_planes = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cube_bisection_pieces_l461_46169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_proof_l461_46112

theorem expression_proof (a : ℝ) : 
  (∃ E : ℝ → ℝ, -6 * a^2 = 3 * (E a + 2)) ↔ 
  (fun x => -2 * x^2 - 2) a = -2 * a^2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_proof_l461_46112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l461_46116

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + x * Real.cos x - 2

theorem tangent_line_at_zero :
  ∃ (m b : ℝ), ∀ (x : ℝ),
    (deriv f 0 * x + f 0) = m * x + b ∧ m = 2 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l461_46116
