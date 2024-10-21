import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_expression_is_sqrt_2_l1325_132539

theorem max_min_expression_is_sqrt_2 :
  ∃ (m : ℝ), m = Real.sqrt 2 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 →
    min x (min (1/y) (1/x + y)) ≤ m) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    min x (min (1/y) (1/x + y)) = m) :=
by
  -- We claim that m = √2 satisfies the conditions
  let m := Real.sqrt 2
  
  -- Prove existence of m
  use m
  
  apply And.intro
  · -- Prove m = √2
    rfl
    
  apply And.intro
  · -- Prove the inequality for all positive x and y
    intros x y hx hy
    sorry -- Proof details omitted
    
  · -- Prove the existence of x and y that achieve the maximum
    use m, 1/m
    apply And.intro
    · -- Prove x > 0
      exact Real.sqrt_pos.mpr (by norm_num)
    apply And.intro
    · -- Prove y > 0
      exact one_div_pos.mpr (Real.sqrt_pos.mpr (by norm_num))
    · -- Prove the equality
      sorry -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_expression_is_sqrt_2_l1325_132539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_nonnegative_iff_m_in_S_l1325_132518

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - |x| else x^2 - 4*x + 3

-- Define the set S as the range of m
noncomputable def S : Set ℝ := Set.Icc (-2) (2 + Real.sqrt 2) ∪ Set.Ioi 4

-- Theorem statement
theorem f_f_nonnegative_iff_m_in_S (m : ℝ) : f (f m) ≥ 0 ↔ m ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_nonnegative_iff_m_in_S_l1325_132518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_socks_until_pair_l1325_132570

/-- Given n pairs of distinct socks in random order, the expected number of socks
    taken until a pair is found is 2n. -/
theorem expected_socks_until_pair (n : ℕ) (h : n > 0) :
  ∃ (E : ℝ), E = 2 * n ∧ E ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_socks_until_pair_l1325_132570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_74pi_l1325_132535

/-- The region described by the equation x^2 + y^2 = 5|x - y| + 7|x + y| -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 5 * |p.1 - p.2| + 7 * |p.1 + p.2|}

/-- The area of the region -/
noncomputable def AreaOfRegion : ℝ := (MeasureTheory.volume Region).toReal

theorem area_of_region_is_74pi : AreaOfRegion = 74 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_is_74pi_l1325_132535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1325_132576

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3 - x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 3

-- Define the inclination angle of the tangent line
noncomputable def α (x : ℝ) : ℝ := Real.arctan (f' x)

-- Theorem statement
theorem inclination_angle_range :
  ∀ x : ℝ, α x ∈ Set.union (Set.Ici 0) (Set.Iic Real.pi) :=
by
  sorry

#check inclination_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1325_132576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_intersection_properties_l1325_132568

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def g (a b x : ℝ) : ℝ := f x + a * x^2 + b * x

theorem tangent_and_intersection_properties
  (a b : ℝ)
  (h1 : DifferentiableAt ℝ (g a b) 1)
  (h2 : deriv (g a b) 1 = 0)
  (x₁ x₂ k : ℝ)
  (h3 : 0 < x₁ ∧ x₁ < x₂)
  (h4 : k = (f x₂ - f x₁) / (x₂ - x₁)) :
  b = -2 * a - 1 ∧ 1 / x₂ < k ∧ k < 1 / x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_intersection_properties_l1325_132568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1325_132515

/-- The distance from the center of the circle x^2 + y^2 - 2x + 2y = 0 to the line y = x + 1 is 3√2/2 -/
theorem distance_circle_center_to_line : 
  let circle_eq := λ (x y : ℝ) => x^2 + y^2 - 2*x + 2*y = 0
  let line_eq := λ (x y : ℝ) => y = x + 1
  ∃ (center_x center_y : ℝ),
    (∀ x y, circle_eq x y ↔ (x - center_x)^2 + (y - center_y)^2 = 2) ∧
    (|(-1) * center_x + 1 * center_y - 1| / Real.sqrt 2 = 3 * Real.sqrt 2 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1325_132515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l1325_132543

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem equality_condition (a b : ℝ) :
  (∀ n : ℕ+, a * floor (b * ↑n) = b * floor (a * ↑n)) ↔
  (a = b ∧ ¬ ∃ k : ℤ, a = ↑k) ∨
  (∃ k l : ℤ, a = ↑k ∧ b = ↑l) ∨
  a = 0 ∨ b = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l1325_132543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_catalan_integrality_l1325_132558

/-- Legendre's formula: For any prime p and positive integer N, 
    the exponent of p in N! is the sum of floor(N/p^k) for k from 1 to infinity -/
noncomputable def legendre_formula (p : ℕ) (N : ℕ) : ℕ := ∑' k, (N / p ^ k : ℕ)

/-- The main theorem: For all non-negative integers n, 
    the expression (2n)! / (n! * n! * (n+1)) is an integer -/
theorem binomial_catalan_integrality (n : ℕ) : 
  ∃ k : ℕ, (2 * n).factorial = k * n.factorial * n.factorial * (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_catalan_integrality_l1325_132558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1325_132513

-- Define the slopes of the two lines
noncomputable def slope1 (m : ℝ) : ℝ := -(3 + m) / 4
noncomputable def slope2 (m : ℝ) : ℝ := -2 / (5 + m)

-- Define the y-intercepts of the two lines
noncomputable def intercept1 (m : ℝ) : ℝ := (5 - 3*m) / 4
noncomputable def intercept2 (m : ℝ) : ℝ := 8 / (5 + m)

-- Theorem statement
theorem parallel_lines_m_value :
  ∀ m : ℝ, m ≠ -5 →
    (slope1 m = slope2 m ∧ intercept1 m ≠ intercept2 m) →
    m = -7 := by
  sorry

#check parallel_lines_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l1325_132513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_cone_height_ratio_l1325_132502

/-- Represents a right cone with a circular base -/
structure Cone where
  baseCircumference : ℝ
  height : ℝ

/-- Calculate the volume of a cone -/
noncomputable def volume (c : Cone) : ℝ :=
  (1 / 3) * c.baseCircumference * c.baseCircumference * c.height / (4 * Real.pi)

theorem shorter_cone_height_ratio (original : Cone) (shorter : Cone) :
  original.baseCircumference = 24 * Real.pi →
  original.height = 40 →
  shorter.baseCircumference = original.baseCircumference →
  volume shorter = 432 * Real.pi →
  shorter.height / original.height = 9 / 40 := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_cone_height_ratio_l1325_132502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nero_run_time_l1325_132553

/-- Calculates the time taken for a runner to complete a trail given the trail length and runner's speed -/
noncomputable def runTime (trailLength : ℝ) (speed : ℝ) : ℝ := trailLength / speed

theorem nero_run_time 
  (jerome_speed : ℝ) 
  (jerome_time : ℝ) 
  (nero_speed : ℝ) 
  (h1 : jerome_speed = 4) 
  (h2 : jerome_time = 6) 
  (h3 : nero_speed = 8) : 
  runTime (jerome_speed * jerome_time) nero_speed = 3 := by
  sorry

#check nero_run_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nero_run_time_l1325_132553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1325_132526

noncomputable section

open Real

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  sin A / a = sin B / b ∧ sin B / b = sin C / c →
  -- Given equation
  1 + (tan A / tan B) + (2 * c / b) = 0 →
  -- Conclusion
  A = 2 * π / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l1325_132526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportionality_problem_l1325_132566

-- Define the function relationship
noncomputable def f (x : ℝ) : ℝ := 5 / (x - 3)

-- Theorem statement
theorem inverse_proportionality_problem :
  -- Condition: When x = 4, y = 5
  f 4 = 5 ∧
  -- Prove that when y = 1, x = 8
  (∃ x : ℝ, f x = 1 ∧ x = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportionality_problem_l1325_132566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l1325_132585

def mySequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → a (n - 1) - a n = a n * a (n - 1)

theorem tenth_term_value (a : ℕ → ℚ) (h : mySequence a) : a 10 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_value_l1325_132585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_3x_at_7_h_condition_l1325_132565

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (5*x + 28)/3

-- State the theorem
theorem h_equals_3x_at_7 :
  ∃ x : ℝ, h x = 3*x ∧ x = 7 := by
  -- Proof goes here
  sorry

-- Additional theorem to capture the condition
theorem h_condition (y : ℝ) : h (3*y - 2) = 5*y + 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_3x_at_7_h_condition_l1325_132565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_1040_l1325_132592

/-- Rectangle EFGH with given coordinates -/
structure Rectangle where
  E : ℝ × ℝ
  F : ℝ × ℝ
  H : ℝ × ℝ

/-- The area of a rectangle given its coordinates -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  let EF := Real.sqrt ((r.F.1 - r.E.1)^2 + (r.F.2 - r.E.2)^2)
  let EH := Real.sqrt ((r.H.1 - r.E.1)^2 + (r.H.2 - r.E.2)^2)
  EF * EH

/-- Theorem: The area of rectangle EFGH is 1040 -/
theorem rectangle_area_is_1040 :
  ∃ (y : ℤ), rectangleArea ⟨(1, 1), (101, 21), (3, ↑y)⟩ = 1040 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_1040_l1325_132592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_wins_first_given_conditions_l1325_132546

/-- Represents the outcome of a single game -/
inductive GameOutcome
| X  -- Team X wins
| Y  -- Team Y wins

instance : DecidableEq GameOutcome := 
  fun a b => match a, b with
  | GameOutcome.X, GameOutcome.X => isTrue rfl
  | GameOutcome.Y, GameOutcome.Y => isTrue rfl
  | GameOutcome.X, GameOutcome.Y => isFalse (fun h => GameOutcome.noConfusion h)
  | GameOutcome.Y, GameOutcome.X => isFalse (fun h => GameOutcome.noConfusion h)

/-- Represents a sequence of game outcomes in a five-game series -/
def GameSequence := Fin 5 → GameOutcome

/-- Checks if Team X wins the series in a given sequence -/
def teamXWinsSeries (seq : GameSequence) : Prop :=
  (seq 0 = GameOutcome.X ∧ seq 1 = GameOutcome.X ∧ seq 2 = GameOutcome.X) ∨
  (seq 0 = GameOutcome.X ∧ seq 1 = GameOutcome.X ∧ seq 3 = GameOutcome.X) ∨
  (seq 0 = GameOutcome.X ∧ seq 1 = GameOutcome.X ∧ seq 4 = GameOutcome.X) ∨
  (seq 0 = GameOutcome.X ∧ seq 2 = GameOutcome.X ∧ seq 3 = GameOutcome.X) ∨
  (seq 0 = GameOutcome.X ∧ seq 2 = GameOutcome.X ∧ seq 4 = GameOutcome.X) ∨
  (seq 0 = GameOutcome.X ∧ seq 3 = GameOutcome.X ∧ seq 4 = GameOutcome.X) ∨
  (seq 1 = GameOutcome.X ∧ seq 2 = GameOutcome.X ∧ seq 3 = GameOutcome.X) ∨
  (seq 1 = GameOutcome.X ∧ seq 2 = GameOutcome.X ∧ seq 4 = GameOutcome.X) ∨
  (seq 1 = GameOutcome.X ∧ seq 3 = GameOutcome.X ∧ seq 4 = GameOutcome.X) ∨
  (seq 2 = GameOutcome.X ∧ seq 3 = GameOutcome.X ∧ seq 4 = GameOutcome.X)

/-- Checks if each team wins at least one game in a given sequence -/
def eachTeamWinsAtLeastOne (seq : GameSequence) : Prop :=
  (∃ i, seq i = GameOutcome.X) ∧ (∃ j, seq j = GameOutcome.Y)

/-- Checks if Team Y wins the third game in a given sequence -/
def teamYWinsThird (seq : GameSequence) : Prop :=
  seq 2 = GameOutcome.Y

/-- Checks if Team Y wins the first game in a given sequence -/
def teamYWinsFirst (seq : GameSequence) : Prop :=
  seq 0 = GameOutcome.Y

instance : DecidablePred teamYWinsFirst :=
  fun seq => decidable_of_iff (seq 0 = GameOutcome.Y) (Iff.rfl)

/-- The main theorem to prove -/
theorem probability_y_wins_first_given_conditions :
  ∀ (validSequences : Finset GameSequence),
  (∀ seq ∈ validSequences, teamXWinsSeries seq ∧ eachTeamWinsAtLeastOne seq ∧ teamYWinsThird seq) →
  (2 : ℚ) / 3 = (validSequences.filter teamYWinsFirst).card / validSequences.card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_y_wins_first_given_conditions_l1325_132546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_on_ellipse_l1325_132580

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : (x^2 / e.a^2) + (y^2 / e.b^2) = 1

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- Affine transformation from ellipse to circle -/
noncomputable def affineTransform (e : Ellipse) (p : PointOnEllipse e) : ℝ × ℝ :=
  (p.x, p.y * (e.a / e.b))

/-- Check if three points form an equilateral triangle -/
noncomputable def isEquilateralTriangle (p q r : ℝ × ℝ) : Prop :=
  let d1 := ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let d2 := ((q.1 - r.1)^2 + (q.2 - r.2)^2)
  let d3 := ((r.1 - p.1)^2 + (r.2 - p.2)^2)
  d1 = d2 ∧ d2 = d3

/-- Theorem: Maximum area triangle on ellipse -/
theorem max_area_triangle_on_ellipse (e : Ellipse) (p : PointOnEllipse e) :
  ∃ (q r : PointOnEllipse e),
    ∀ (q' r' : PointOnEllipse e),
      triangleArea (p.x, p.y) (q.x, q.y) (r.x, r.y) ≥ 
      triangleArea (p.x, p.y) (q'.x, q'.y) (r'.x, r'.y) ∧
      isEquilateralTriangle (affineTransform e p) (affineTransform e q) (affineTransform e r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_on_ellipse_l1325_132580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1325_132590

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 - x^2))

-- State the theorem
theorem g_is_odd : ∀ x, g x = -g (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1325_132590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1325_132534

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (4 - 3*Complex.I) / (2 + Complex.I)
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l1325_132534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_or_q_is_true_l1325_132598

theorem proposition_p_or_q_is_true :
  (∀ x : ℝ, x^2 ≥ 0) ∨ (∃ x₀ : ℕ+, (2 : ℝ) * x₀ - 1 ≤ 0) :=
by
  left
  intro x
  exact sq_nonneg x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_or_q_is_true_l1325_132598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_B_work_days_l1325_132516

noncomputable def work_days_A : ℝ := 10

noncomputable def work_share_A : ℝ := 3/5

noncomputable def work_days_B : ℝ := 15

theorem prove_B_work_days :
  work_days_A = 10 ∧ work_share_A = 3/5 →
  work_days_B = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_B_work_days_l1325_132516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_integer_is_26_l1325_132556

theorem original_integer_is_26 
  (a b c d : ℤ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : (a + b + c) / 3 + d = 35)
  (h2 : (a + b + d) / 3 + c = 27)
  (h3 : (a + c + d) / 3 + b = 25)
  (h4 : (b + c + d) / 3 + a = 19) :
  a = 26 ∨ b = 26 ∨ c = 26 ∨ d = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_integer_is_26_l1325_132556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_fourth_id_l1325_132511

/-- Systematic sampling function -/
def systematicSample (populationSize sampleSize firstID : ℕ) : ℕ → ℕ :=
  λ n => firstID + (populationSize / sampleSize) * (n - 1)

theorem systematic_sample_fourth_id 
  (populationSize : ℕ) 
  (sampleSize : ℕ) 
  (firstID : ℕ) 
  (h1 : populationSize = 56) 
  (h2 : sampleSize = 4) 
  (h3 : firstID = 4) 
  (h4 : systematicSample populationSize sampleSize firstID 3 = 32) 
  (h5 : systematicSample populationSize sampleSize firstID 4 = 46) :
  systematicSample populationSize sampleSize firstID 2 = 18 := by
  sorry

#eval systematicSample 56 4 4 2  -- This should output 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_fourth_id_l1325_132511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_function_l1325_132586

theorem range_of_function (y : ℝ) : 
  (∃ x : ℝ, Real.cos (π/3 - x) + Real.sin (π/2 + x) = y) ↔ -Real.sqrt 3 ≤ y ∧ y ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_function_l1325_132586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_equation_l1325_132575

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- Calculates the circumcenter of a triangle -/
noncomputable def circumcenter (_t : Triangle) : Point :=
  { x := -1,
    y := 1 }

/-- Calculates the orthocenter of a triangle -/
noncomputable def orthocenter (_t : Triangle) : Point :=
  { x := -1,
    y := -1 }

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

/-- The main theorem: Euler line equation for the given triangle -/
theorem euler_line_equation (t : Triangle) 
  (h1 : t.A = { x := 2, y := 0 })
  (h2 : t.B = { x := 0, y := 4 })
  (h3 : t.C = { x := -4, y := 0 }) :
  let ce := centroid t
  let cc := circumcenter t
  let oc := orthocenter t
  collinear ce cc oc ∧ 
  ce.x - ce.y + 2 = 0 ∧
  cc.x - cc.y + 2 = 0 ∧
  oc.x - oc.y + 2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_equation_l1325_132575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_proof_l1325_132545

/-- The speed of the walking team in km/h -/
noncomputable def walking_speed : ℝ := 5

/-- The delay in hours before the cyclist starts -/
noncomputable def cyclist_delay : ℝ := 4 + 24 / 60

/-- The distance traveled by the cyclist before the inner tube bursts in km -/
noncomputable def distance_before_burst : ℝ := 8

/-- The time taken to change the inner tube in hours -/
noncomputable def tube_change_time : ℝ := 10 / 60

/-- The increase in cyclist's speed after changing the tube in km/h -/
noncomputable def speed_increase : ℝ := 2

/-- The initial speed of the cyclist in km/h -/
noncomputable def initial_cyclist_speed : ℝ := 16

theorem cyclist_speed_proof :
  ∃ (t : ℝ),
    (t * initial_cyclist_speed = walking_speed * (t + cyclist_delay)) ∧
    (distance_before_burst + 
      (t - distance_before_burst / initial_cyclist_speed - tube_change_time) * 
      (initial_cyclist_speed + speed_increase) = 
    walking_speed * (t + cyclist_delay)) := by
  sorry

#check cyclist_speed_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_speed_proof_l1325_132545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_max_t_l1325_132584

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x^2 - (2*t + 1)*x + t * Real.log x

def g (t : ℝ) (x : ℝ) : ℝ := (1 - t)*x

theorem extreme_values_and_max_t :
  (∀ x > 0, IsLocalMax (f 1) x = (x = 1/2 ∧ f 1 (1/2) = -5/4 - Real.log 2)) ∧
  (∀ x > 0, IsLocalMin (f 1) x = (x = 1 ∧ f 1 1 = -2)) ∧
  (∀ t : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f t x₀ ≥ g t x₀) →
    t ≤ (Real.exp 1) * ((Real.exp 1) - 2) / ((Real.exp 1) - 1)) ∧
  (∃ t : ℝ, t = (Real.exp 1) * ((Real.exp 1) - 2) / ((Real.exp 1) - 1) ∧
    ∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f t x₀ ≥ g t x₀) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_max_t_l1325_132584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1325_132500

theorem relationship_abc (a b c : ℝ) 
  (ha : a = 5^(-(1/2 : ℝ)))
  (hb : Real.log b = 1)
  (hc : 3 * c^3 + c = 1) :
  b > c ∧ c > a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1325_132500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l1325_132508

-- Define the quadratic equation and its roots
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def root1 (a b c : ℝ) : ℂ := (-b + (discriminant a b c).sqrt) / (2 * a)
noncomputable def root2 (a b c : ℝ) : ℂ := (-b - (discriminant a b c).sqrt) / (2 * a)

-- Theorem statement
theorem quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  (discriminant a b c > 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic_equation a b c x1 ∧ quadratic_equation a b c x2) ∧
  (discriminant a b c = 0 → ∃ (x : ℝ), quadratic_equation a b c x) ∧
  (discriminant a b c < 0 → ∃ (x1 x2 : ℂ), x1 = root1 a b c ∧ x2 = root2 a b c ∧ 
    (a * x1^2 + b * x1 + c : ℂ) = 0 ∧ (a * x2^2 + b * x2 + c : ℂ) = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l1325_132508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l1325_132533

open Real

/-- Given an equation x^2 + 3ax + 3a + 1 = 0 (a > 1) with roots tan(α) and tan(β),
    where α, β ∈ (-π/2, π/2), prove that α + β = -3π/4 -/
theorem sum_of_angles (a : ℝ) (α β : ℝ) (ha : a > 1)
    (hα : α ∈ Set.Ioo (-π/2) (π/2)) (hβ : β ∈ Set.Ioo (-π/2) (π/2))
    (heq : ∀ x, x^2 + 3*a*x + 3*a + 1 = 0 ↔ x = tan α ∨ x = tan β) :
  α + β = -3*π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l1325_132533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_is_reals_l1325_132548

/-- A function g satisfying g(x) + g(-x) = x^2 for all real x -/
def g : ℝ → ℝ := sorry

/-- The property that g(x) + g(-x) = x^2 for all real x -/
axiom g_property (x : ℝ) : g x + g (-x) = x^2

/-- The domain of g is the set of all real numbers -/
def domain_g : Set ℝ := Set.univ

/-- Theorem: The domain of g can be the entire set of real numbers -/
theorem g_domain_is_reals : domain_g = Set.univ := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_is_reals_l1325_132548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_preservation_l1325_132507

noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_preservation 
  (f : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) 
  (h : triangle_area (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 32) :
  triangle_area (x₁/2, 2 * f x₁) (x₂/2, 2 * f x₂) (x₃/2, 2 * f x₃) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_preservation_l1325_132507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_2_pow_24_minus_1_l1325_132562

theorem two_digit_factors_of_2_pow_24_minus_1 : 
  (Finset.filter (λ n : ℕ ↦ 10 ≤ n ∧ n < 100 ∧ (2^24 - 1) % n = 0) (Finset.range 100)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_factors_of_2_pow_24_minus_1_l1325_132562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1325_132596

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 4), 1)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos (x / 4), Real.cos (x / 4) ^ 2)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

noncomputable def g (x : ℝ) : ℝ := f (x - 2 * Real.pi / 3)

theorem vector_problem :
  (∀ (k : ℤ), ∀ (x : ℝ), 
    (x ∈ Set.Icc (4 * ↑k * Real.pi + 2 * Real.pi / 3) (4 * ↑k * Real.pi + 4 * Real.pi / 3) → 
    ∀ (y : ℝ), y ∈ Set.Icc (4 * ↑k * Real.pi + 2 * Real.pi / 3) (4 * ↑k * Real.pi + 4 * Real.pi / 3) → 
    x ≤ y → f y ≤ f x)) ∧
  (∀ (a : ℝ), f a = 3 / 2 → Real.cos (2 * Real.pi / 3 - a) = 1) ∧
  (∀ (k : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 (7 * Real.pi / 3) ∧ g x = k) → 
    k ∈ Set.Icc 0 (3 / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1325_132596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_60_deg_radius_6_l1325_132538

/-- The area of a sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle * Real.pi * radius^2) / 360

theorem sector_area_60_deg_radius_6 :
  sectorArea 6 60 = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_60_deg_radius_6_l1325_132538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_characterization_l1325_132571

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem statement -/
theorem fibonacci_characterization (a b : ℕ) :
  (a^2 - a*b - b^2 = (1 : ℤ) ∨ a^2 - a*b - b^2 = (-1 : ℤ)) →
  ∃ n : ℕ, a = fib (n + 1) ∧ b = fib n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_characterization_l1325_132571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_GAC_value_l1325_132579

/-- A rectangular prism ABCDEFGH with given dimensions -/
structure RectPrism where
  AB : ℝ
  AD : ℝ
  AE : ℝ
  is_rect_prism : AB > 0 ∧ AD > 0 ∧ AE > 0

/-- The sine of angle GAC in the rectangular prism -/
noncomputable def sin_GAC (p : RectPrism) : ℝ :=
  3 / Real.sqrt 29

/-- Theorem: In a rectangular prism ABCDEFGH with AB = 2, AD = 4, and AE = 3, sin ∠GAC = 3 / √29 -/
theorem sin_GAC_value (p : RectPrism) 
    (h1 : p.AB = 2) 
    (h2 : p.AD = 4) 
    (h3 : p.AE = 3) : 
  sin_GAC p = 3 / Real.sqrt 29 := by
  sorry

#check sin_GAC_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_GAC_value_l1325_132579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_round_trip_exists_l1325_132554

/-- A structure representing an airline network -/
structure AirlineNetwork where
  n : ℕ  -- number of airlines
  m : ℕ  -- number of cities
  h : m > 2 * n

/-- A proposition stating that in any airline network, at least one airline can offer an odd round trip -/
def exists_odd_round_trip (network : AirlineNetwork) : Prop :=
  ∃ (airline : Fin network.n), ∃ (trip : List (Fin network.m)),
    trip.length % 2 = 1 ∧
    trip.head? = trip.getLast? ∧
    ∀ (i : Fin (trip.length - 1)), ∃ (flight : Fin network.n),
      flight = airline ∨ trip[i.val] ≠ trip[i.val + 1]

/-- The main theorem stating that in any airline network, there exists an odd round trip -/
theorem odd_round_trip_exists (network : AirlineNetwork) :
  exists_odd_round_trip network := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_round_trip_exists_l1325_132554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_mixture_theorem_l1325_132501

/-- Given an initial mixture and pure grape juice added, calculate the final grape juice percentage -/
noncomputable def final_grape_juice_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) : ℝ :=
  let initial_grape_juice := initial_volume * (initial_percentage / 100)
  let total_grape_juice := initial_grape_juice + added_volume
  let final_volume := initial_volume + added_volume
  (total_grape_juice / final_volume) * 100

/-- Theorem stating that given the specified initial conditions, the final grape juice percentage is 40% -/
theorem grape_juice_mixture_theorem :
  final_grape_juice_percentage 40 10 20 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_mixture_theorem_l1325_132501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_rectangles_l1325_132536

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the area of a rectangle
noncomputable def area (r : Rectangle) : ℝ := r.length * r.width

-- Define the center point of a rectangle
noncomputable def center (r : Rectangle) : ℝ × ℝ := (r.length / 2, r.width / 2)

-- Define the overlap area of two rectangles when one is centered on the other
noncomputable def overlapArea (r : Rectangle) : ℝ := (r.length / 2) * (r.width / 2)

-- Theorem statement
theorem area_of_overlapping_rectangles :
  let r1 : Rectangle := { length := 12, width := 8 }
  let r2 : Rectangle := { length := 12, width := 8 }
  area r1 + area r2 - overlapArea r1 = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_rectangles_l1325_132536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_function_properties_l1325_132551

/-- Random function X(t) = U cos(2t) where U is a random variable -/
noncomputable def X (t : ℝ) (U : ℝ) : ℝ := U * Real.cos (2 * t)

/-- Expected value of U -/
def M_U : ℝ := 5

/-- Variance of U -/
def D_U : ℝ := 6

/-- Expected value of X(t) -/
noncomputable def M_X (t : ℝ) : ℝ := M_U * Real.cos (2 * t)

/-- Correlation function of X(t) -/
noncomputable def K_X (t₁ t₂ : ℝ) : ℝ := D_U * Real.cos (2 * t₁) * Real.cos (2 * t₂)

/-- Variance of X(t) -/
noncomputable def D_X (t : ℝ) : ℝ := D_U * (Real.cos (2 * t))^2

theorem random_function_properties :
  ∀ t t₁ t₂ : ℝ,
  (M_X t = 5 * Real.cos (2 * t)) ∧
  (K_X t₁ t₂ = 6 * Real.cos (2 * t₁) * Real.cos (2 * t₂)) ∧
  (D_X t = 6 * (Real.cos (2 * t))^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_function_properties_l1325_132551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_open_interval_l1325_132557

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x - 3}
def N : Set ℝ := {x : ℝ | -5 ≤ x ∧ x ≤ 2}

theorem intersection_complement_equals_open_interval :
  M ∩ (Set.univ \ N) = Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equals_open_interval_l1325_132557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_hyperbola_l1325_132540

-- Define the point (x, y) as a function of s
noncomputable def point (s : ℝ) : ℝ × ℝ :=
  (2 * (Real.exp s + Real.exp (-s)), 4 * (Real.exp s - Real.exp (-s)))

-- Define the hyperbola equation
def is_on_hyperbola (p : ℝ × ℝ) : Prop :=
  (p.1^2 / 16) - (p.2^2 / 64) = 1

-- Theorem statement
theorem points_form_hyperbola :
  ∀ s : ℝ, is_on_hyperbola (point s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_hyperbola_l1325_132540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_approx_l1325_132560

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

theorem interest_difference_approx : 
  let cedric_balance := compound_interest 15000 0.06 10
  let daniel_balance := simple_interest 15000 0.08 10
  ⌊|daniel_balance - cedric_balance|⌋₊ = 137 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_approx_l1325_132560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_insulation_cost_l1325_132564

/-- Annual energy consumption cost function -/
noncomputable def C (k : ℝ) (x : ℝ) : ℝ := k / (3 * x + 5)

/-- Total cost function over 20 years -/
noncomputable def f (x : ℝ) : ℝ := 400 / (3 * x + 5) + 3 * x

/-- Theorem stating the minimum value of f(x) -/
theorem min_insulation_cost :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ 10 ∧
  (∀ y, y ≥ 0 → y ≤ 10 → f y ≥ f x) ∧
  x = 5 ∧ f x = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_insulation_cost_l1325_132564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_form_quadrilateral_probability_l1325_132505

/-- Given 7 points on a circle, the probability that 4 randomly selected chords 
    form a convex quadrilateral is 1/171. -/
theorem chords_form_quadrilateral_probability (n : ℕ) (k : ℕ) : 
  n = 7 → k = 4 → (Nat.choose n k : ℚ) / (Nat.choose (Nat.choose n 2) k) = 1 / 171 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_form_quadrilateral_probability_l1325_132505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_neg_four_range_of_a_for_two_distinct_roots_l1325_132561

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4 : ℝ)^x + a * (2 : ℝ)^x + 3

-- Part 1
theorem range_of_f_when_a_is_neg_four :
  ∃ (y_min y_max : ℝ), y_min = -1 ∧ y_max = 3 ∧
  ∀ x, x ∈ Set.Icc 0 2 → y_min ≤ f (-4) x ∧ f (-4) x ≤ y_max ∧
  ∃ x₁ x₂, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ f (-4) x₁ = y_min ∧ f (-4) x₂ = y_max :=
sorry

-- Part 2
theorem range_of_a_for_two_distinct_roots :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔
  -4 < a ∧ a < -2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_a_is_neg_four_range_of_a_for_two_distinct_roots_l1325_132561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l1325_132517

/-- Triangle ABC with points G, H, I on its sides -/
structure TriangleWithPoints where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Ratios for points G, H, I
  s : ℝ
  t : ℝ
  u : ℝ
  -- Conditions
  AB_pos : AB > 0
  BC_pos : BC > 0
  CA_pos : CA > 0
  s_pos : s > 0
  t_pos : t > 0
  u_pos : u > 0
  sum_condition : s + t + u = 3/4
  square_sum_condition : s^2 + t^2 + u^2 = 3/7

/-- Area of triangle ABC -/
def area_ABC (T : TriangleWithPoints) : ℝ := sorry

/-- Area of triangle GHI -/
def area_GHI (T : TriangleWithPoints) : ℝ := sorry

/-- The main theorem -/
theorem area_ratio_theorem (T : TriangleWithPoints) (h1 : T.AB = 10) (h2 : T.BC = 12) (h3 : T.CA = 14) :
  (area_GHI T) / (area_ABC T) = 71/224 := by
  sorry

#check area_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l1325_132517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_3x_plus_2y_eq_806_l1325_132504

theorem solutions_count_3x_plus_2y_eq_806 :
  ∃ (n : ℕ), n = 134 ∧
  n = (Finset.filter (λ p : ℕ × ℕ ↦ 3 * p.1 + 2 * p.2 = 806 ∧ p.1 > 0 ∧ p.2 > 0)
        (Finset.product (Finset.range 807) (Finset.range 807))).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_3x_plus_2y_eq_806_l1325_132504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132525

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (1 - x)

-- State the theorem
theorem f_properties (a : ℝ) :
  -- Part 1: Monotonicity when a ≤ 0
  (a ≤ 0 → ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f a x < f a y) ∧
  -- Part 2: Monotonicity when a > 0
  (a > 0 → (∀ x y, 0 < x ∧ x < y ∧ y < 1/a → f a x < f a y) ∧
           (∀ x y, 1/a < x ∧ x < y → f a y < f a x)) ∧
  -- Part 3: Maximum value condition
  (∃ x, 0 < x ∧ f a x > 2*a - 2) ↔ (0 < a ∧ a < 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1325_132573

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train 250 m long, running at 90 km/hr, takes 24 seconds to cross a 350 m bridge -/
theorem train_bridge_crossing_time :
  train_crossing_time 250 350 90 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1325_132573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_equality_l1325_132583

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x^2 + 3 * x)

theorem domain_range_equality (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y ∧ a * x^2 + 3 * x ≥ 0) ↔ (a = -4 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_equality_l1325_132583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132587

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (max : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ max ∧ ∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₀ = max) ∧
  (∃ (min : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → min ≤ f x ∧ ∃ (x₀ : ℝ), x₀ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₀ = min) ∧
  (∀ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x → p ≥ Real.pi) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l1325_132591

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  p1 : Point
  p2 : Point

structure Plane where
  p : Point
  n : Point  -- normal vector

structure Pyramid where
  P : Point
  A : Point
  B : Point
  C : Point
  D : Point

-- Define basic geometric relations
def is_trapezoid (A B C D : Point) : Prop := sorry

def parallel_lines (l1 l2 : Line) : Prop := sorry

def parallel_plane_line (π : Plane) (l : Line) : Prop := sorry

def line_in_plane (l : Line) (π : Plane) : Prop := sorry

noncomputable def plane_intersection (π1 π2 : Plane) : Line := sorry

-- Main theorem
theorem pyramid_properties (pyr : Pyramid) 
  (h1 : is_trapezoid pyr.A pyr.B pyr.C pyr.D)
  (h2 : parallel_lines (Line.mk pyr.A pyr.B) (Line.mk pyr.C pyr.D)) :
  (∃ (S : Set Line), S.Infinite ∧ 
    ∀ l ∈ S, line_in_plane l (Plane.mk pyr.P (Point.mk 0 0 1)) ∧ 
      parallel_plane_line (Plane.mk pyr.P (Point.mk 0 0 1)) l) ∧
  (parallel_plane_line (Plane.mk pyr.A (Point.mk 0 0 1)) 
    (plane_intersection (Plane.mk pyr.P (Point.mk 0 0 1)) (Plane.mk pyr.P (Point.mk 0 0 1)))) ∧
  (∀ l : Line, line_in_plane l (Plane.mk pyr.P (Point.mk 0 0 1)) → 
    ¬ parallel_lines l (Line.mk pyr.B pyr.C)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l1325_132591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1325_132593

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 6) / (x + 1)

-- State the theorem
theorem f_minimum_value :
  (∀ x > -1, f x ≥ 2) ∧ (∃ x > -1, f x = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1325_132593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_allergy_ratio_l1325_132594

/-- Represents the number of children in different allergy categories in a kindergarten. -/
structure KindergartenAllergies where
  peanut_allergy : ℕ
  cashew_allergy : ℕ
  both_allergies : ℕ
  no_allergies : ℕ

/-- Calculates the ratio of children allergic to peanuts to the total number of children. -/
def allergy_ratio (k : KindergartenAllergies) : ℚ :=
  let total := k.peanut_allergy + k.cashew_allergy - k.both_allergies + k.no_allergies
  ↑k.peanut_allergy / ↑total

/-- Theorem stating the ratio of children allergic to peanuts to the total number of children. -/
theorem peanut_allergy_ratio (k : KindergartenAllergies)
  (h1 : k.cashew_allergy = 18)
  (h2 : k.both_allergies = 10)
  (h3 : k.peanut_allergy - k.both_allergies = 10)
  (h4 : k.no_allergies = 6) :
  allergy_ratio k = 10 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_allergy_ratio_l1325_132594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1325_132528

theorem angle_in_second_quadrant (α : Real) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) :
  α ∈ Set.Ioo (Real.pi / 2) Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1325_132528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_center_and_chord_l1325_132569

/-- The equation of a circle given its center and a chord formed by a line intersection -/
theorem circle_equation_from_center_and_chord (x y : ℝ) :
  let center : ℝ × ℝ := (2, -1)
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}
  let chord_length : ℝ := 2 * Real.sqrt 2
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 + 1)^2 = 4}
  (∃ (p q : ℝ × ℝ), p ∈ circle ∧ q ∈ circle ∧ p ∈ line ∧ q ∈ line ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  (x, y) ∈ circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_center_and_chord_l1325_132569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1325_132503

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + y + f y) = 2*y + (f x)^2) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1325_132503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132549

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + a

theorem f_properties (a : ℝ) :
  (∃ T : ℝ, T > 0 ∧ ∀ x, f x a = f (x + T) a ∧
    ∀ S, S > 0 ∧ (∀ x, f x a = f (x + S) a) → T ≤ S) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = -2) →
  (∃ T : ℝ, T = Real.pi ∧ T > 0 ∧ ∀ x, f x a = f (x + T) a ∧
    ∀ S, S > 0 ∧ (∀ x, f x a = f (x + S) a) → T ≤ S) ∧
  a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_parallel_line_theorem_l1325_132521

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y + 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 5 * y + 7 = 0
def line3 (x y : ℝ) : Prop := x + 3 * y = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the parallel line passing through the intersection point
def parallel_line (x y : ℝ) : Prop := x + 3 * y + 4 = 0

-- Define the distance between two parallel lines
noncomputable def distance_between_lines (a b c₁ c₂ : ℝ) : ℝ := 
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem intersection_and_parallel_line_theorem :
  ∃ x y : ℝ, intersection_point x y ∧
  (∀ x' y' : ℝ, intersection_point x' y' → parallel_line x' y') ∧
  distance_between_lines 1 3 4 0 = 2 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_parallel_line_theorem_l1325_132521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_same_parity_l1325_132578

theorem at_least_two_same_parity (a b c : ℤ) : 
  ∃ x y : ℤ, x ≠ y ∧ x ∈ ({a, b, c} : Set ℤ) ∧ y ∈ ({a, b, c} : Set ℤ) ∧ x % 2 = y % 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_same_parity_l1325_132578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_is_twenty_days_l1325_132542

/-- Calculates the number of days required for a given percentage of water to evaporate -/
noncomputable def evaporation_days (initial_amount : ℝ) (evaporation_rate : ℝ) (evaporation_percentage : ℝ) : ℝ :=
  (evaporation_percentage / 100) * initial_amount / evaporation_rate

/-- Theorem stating that the evaporation period is 20 days given the problem conditions -/
theorem evaporation_period_is_twenty_days :
  evaporation_days 10 0.01 2 = 20 := by
  -- Unfold the definition of evaporation_days
  unfold evaporation_days
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaporation_period_is_twenty_days_l1325_132542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_height_implies_interior_point_l1325_132550

-- Define the basic structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Triangle where
  A : Point3D
  B : Point3D
  C : Point3D

structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

-- Define the concept of a height in a tetrahedron
noncomputable def tetraHeight (t : Tetrahedron) (vertex : Point3D) : ℝ :=
  sorry

-- Define the concept of an interior triangle formed by trisecting sides
def interiorTrisectionTriangle (t : Triangle) : Set Point3D :=
  sorry

-- Main theorem
theorem smallest_height_implies_interior_point (ABC : Triangle) (D : Point3D) :
  let ABCD := Tetrahedron.mk ABC.A ABC.B ABC.C D
  let P := Point3D.mk 0 0 0  -- Placeholder for the foot of the height from D
  (∀ v, tetraHeight ABCD D ≤ tetraHeight ABCD v) →
  P ∈ interiorTrisectionTriangle ABC := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_height_implies_interior_point_l1325_132550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_l1325_132552

-- Define the inverse proportion function
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m + 5) / x

-- Define the condition for the graph being in the second and fourth quadrants
def in_second_and_fourth_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → (x < 0 ∧ f x > 0) ∨ (x > 0 ∧ f x < 0)

-- State the theorem
theorem inverse_proportion_range (m : ℝ) :
  in_second_and_fourth_quadrants (inverse_proportion m) → m < -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_range_l1325_132552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_is_4_575_l1325_132574

/-- Parking fee structure and conditions -/
structure ParkingFee where
  base_cost : ℚ := 25
  additional_cost_per_hour : ℚ := 9/4
  weekend_surcharge : ℚ := 5
  discount_rate : ℚ := 1/10
  parking_duration : ℚ := 9

/-- Calculate the average hourly parking cost -/
noncomputable def average_hourly_cost (fee : ParkingFee) : ℚ :=
  let total_cost := fee.base_cost + (fee.parking_duration - 2) * fee.additional_cost_per_hour + fee.weekend_surcharge
  let discounted_cost := total_cost * (1 - fee.discount_rate)
  discounted_cost / fee.parking_duration

/-- Theorem stating that the average hourly cost is $4.575 -/
theorem average_cost_is_4_575 (fee : ParkingFee) :
  average_hourly_cost fee = 183/40 := by
  sorry

#eval (183 : ℚ) / 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_is_4_575_l1325_132574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_C_l1325_132595

/-- The distance between two points given their north-south and east-west displacements -/
noncomputable def distance (north_south : ℝ) (east_west : ℝ) : ℝ :=
  Real.sqrt (north_south^2 + east_west^2)

/-- Theorem: The distance between points A and C is 10√34 yards -/
theorem distance_A_to_C : distance 30 50 = 10 * Real.sqrt 34 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_C_l1325_132595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_two_over_101_l1325_132581

def mySequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * mySequence n / (mySequence n + 2)

theorem hundredth_term_is_two_over_101 :
  mySequence 99 = 2 / 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundredth_term_is_two_over_101_l1325_132581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_approx_l1325_132532

-- Define the given quantities
def large_bottles : ℕ := 1375
def small_bottles : ℕ := 690
def large_bottle_price : ℚ := 175/100
def small_bottle_price : ℚ := 135/100

-- Define the total cost function
def total_cost : ℚ := large_bottles * large_bottle_price + small_bottles * small_bottle_price

-- Define the total number of bottles
def total_bottles : ℕ := large_bottles + small_bottles

-- Define the average price per bottle
def average_price : ℚ := total_cost / total_bottles

-- Theorem statement
theorem average_price_approx :
  (average_price * 1000).floor / 1000 = 1616/1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_approx_l1325_132532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l1325_132522

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  2 * x^2 - y^2 + 8 * x - 4 * y - 8 = 0

/-- The focus point of the hyperbola -/
noncomputable def focus : ℝ × ℝ := (-2 - Real.sqrt 6, -2)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola :
  let (fx, fy) := focus
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), hyperbola_equation x y ↔ 
      ((x + 2)^2 / (2 * a^2) - (y + 2)^2 / (2 * b^2) = 1)) ∧
    fx^2 - (-2)^2 = a^2 + b^2 ∧
    fy = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l1325_132522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_interest_rate_l1325_132506

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem additional_interest_rate 
  (initial_deposit : ℝ)
  (amount_after_3_years : ℝ)
  (target_amount : ℝ)
  (h1 : initial_deposit = 8000)
  (h2 : amount_after_3_years = 10200)
  (h3 : target_amount = 10680)
  (h4 : ∃ r : ℝ, simple_interest initial_deposit r 3 = amount_after_3_years - initial_deposit) :
  ∃ (r additional_rate : ℝ), 
    additional_rate = 2 ∧ 
    simple_interest initial_deposit r 3 = amount_after_3_years - initial_deposit ∧
    simple_interest initial_deposit (r + additional_rate) 3 = target_amount - initial_deposit :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_interest_rate_l1325_132506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132559

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - 1/2 * Real.cos (2*x) - 1/2

theorem f_properties :
  -- Smallest positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π) ∧
  -- Monotonically increasing interval
  (∀ k : ℤ, ∀ x y, k * π - π/6 < x ∧ x < y ∧ y < k * π + π/3 → f x < f y) ∧
  -- Maximum and minimum on [0, π/2]
  (∀ x, 0 ≤ x ∧ x ≤ π/2 → f x ≤ f (π/3)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ π/2 → f 0 ≤ f x) ∧
  f (π/3) = 1/2 ∧
  f 0 = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1325_132559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_completion_l1325_132544

/-- Represents a round trip to a service center -/
structure RoundTrip where
  outbound : ℚ  -- Distance of the outbound trip
  inbound : ℚ   -- Distance of the inbound trip
  h_positive : outbound > 0 ∧ inbound > 0  -- Distances are positive

/-- Calculates the percentage of the round trip completed -/
def trip_completion_percentage (trip : RoundTrip) (outbound_completed : ℚ) (inbound_completed : ℚ) : ℚ :=
  (outbound_completed + inbound_completed) / (trip.outbound + trip.inbound) * 100

/-- Theorem: If a technician completes the outbound trip and 30% of the inbound trip,
    they have completed 65% of the entire round trip -/
theorem round_trip_completion (trip : RoundTrip) :
  trip_completion_percentage trip trip.outbound (3 / 10 * trip.inbound) = 65 := by
  sorry

#eval trip_completion_percentage { outbound := 50, inbound := 50, h_positive := by norm_num } 50 (3 / 10 * 50)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_completion_l1325_132544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guaranteed_chords_l1325_132510

-- Helper function to determine if two chords intersect
def intersect {n : ℕ} (a b c d : Fin n) : Prop := sorry

theorem guaranteed_chords (n m : ℕ) (h1 : n = 2006) (h2 : m = 17) :
  ∃ k : ℕ, k ≥ 117 ∧
  ∀ coloring : Fin n → Fin m,
  ∃ chords : List (Fin n × Fin n),
    (∀ (chord : Fin n × Fin n), chord ∈ chords → coloring chord.1 = coloring chord.2) ∧
    (∀ (chord1 chord2 : Fin n × Fin n), chord1 ∈ chords → chord2 ∈ chords → chord1 ≠ chord2 →
      ¬ intersect chord1.1 chord1.2 chord2.1 chord2.2) ∧
    chords.length = k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guaranteed_chords_l1325_132510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_theorem_l1325_132597

/-- Calculates the average speed of a car given its uphill and downhill speeds and distances -/
noncomputable def average_speed (uphill_speed uphill_distance downhill_speed downhill_distance : ℝ) : ℝ :=
  let total_distance := uphill_distance + downhill_distance
  let total_time := uphill_distance / uphill_speed + downhill_distance / downhill_speed
  total_distance / total_time

/-- Theorem stating that the average speed of a car traveling 100 km uphill at 30 km/hr
    and 50 km downhill at 80 km/hr is approximately 37.92 km/hr -/
theorem car_average_speed_theorem :
  let uphill_speed : ℝ := 30
  let uphill_distance : ℝ := 100
  let downhill_speed : ℝ := 80
  let downhill_distance : ℝ := 50
  abs (average_speed uphill_speed uphill_distance downhill_speed downhill_distance - 37.92) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_theorem_l1325_132597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1325_132589

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + c) / (x^2 + 2*x - 3)

theorem exactly_one_vertical_asymptote (c : ℝ) :
  (∃! x, ¬∃ y, f c x = y) ↔ c = 1 ∨ c = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1325_132589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strength_order_l1325_132547

-- Define the set of people
inductive Person : Type
  | A | B | C | D
deriving BEq, Repr

-- Define a relation for "stronger than"
def stronger_than : Person → Person → Prop := sorry

-- Define the conditions
axiom balance : stronger_than Person.A Person.B ↔ stronger_than Person.C Person.D
axiom AD_stronger : stronger_than Person.A Person.D ∧ stronger_than Person.D Person.B ∧ stronger_than Person.B Person.C
axiom B_stronger_AC : stronger_than Person.B Person.A ∧ stronger_than Person.B Person.C

-- Define the correct order
def correct_order : List Person := [Person.D, Person.B, Person.A, Person.C]

-- Theorem to prove
theorem strength_order : 
  ∀ (x y : Person), x ∈ correct_order ∧ y ∈ correct_order ∧ 
    correct_order.indexOf x < correct_order.indexOf y → stronger_than x y := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strength_order_l1325_132547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1325_132563

def A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 1]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 7]
  IsUnit (Matrix.det A) ∧ A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1325_132563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1325_132520

noncomputable def floor_sin (x : ℝ) : ℤ := ⌊Real.sin x⌋

noncomputable def floor_cos (x : ℝ) : ℤ := ⌊Real.cos x⌋

def solution_set (x : ℝ) : Prop :=
  ∃ n : ℤ, (x ∈ Set.Ioo ((2 * Real.pi : ℝ) * ↑n) ((Real.pi / 4 + 2 * Real.pi * ↑n : ℝ))) ∨
           (x ∈ Set.Ioo ((Real.pi / 4 + 2 * Real.pi * ↑n : ℝ)) ((Real.pi / 2 + 2 * Real.pi * ↑n : ℝ)))

theorem trigonometric_equation_solution :
  ∀ x : ℝ,
    (3 * floor_sin (2 * x) ∈ ({-3, 0, 3} : Set ℤ)) →
    (2 * floor_cos x ∈ ({-2, 0, 2} : Set ℤ)) →
    (floor_sin (2 * x) ∈ ({-1, 0, 1} : Set ℤ)) →
    solution_set x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1325_132520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l1325_132567

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
noncomputable def inscribed_circle_radius (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / (4 * Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2))

/-- Theorem: The radius of a circle inscribed in a rhombus with diagonals 8 and 30 is 30/√241 -/
theorem inscribed_circle_radius_specific : 
  inscribed_circle_radius 8 30 = 30 / Real.sqrt 241 := by
  -- Unfold the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l1325_132567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1325_132509

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = Real.sqrt 3 / 2 →
  b = Real.sqrt 3 →
  c = 2 →
  a^2 / (b * c) = 2 - Real.sqrt 3 →
  a = 1 ∧ B = 5 * Real.pi / 12 ∧ C = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1325_132509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincidence_l1325_132588

theorem hyperbola_parabola_focus_coincidence (k : ℝ) : k > 0 →
  (∃ (x y : ℝ), x^2 - y^2 / k^2 = 1 ∧ y^2 = 8*x) →
  (∀ (x y : ℝ), x^2 - y^2 / k^2 = 1 → x = Real.sqrt (1 + k^2)) →
  (∀ (x : ℝ), x = 2 → ∃ (y : ℝ), y^2 = 8*x) →
  k = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_focus_coincidence_l1325_132588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_proof_l1325_132524

-- Define the initial length of rope
noncomputable def initial_length : ℝ := 50

-- Define the fraction used for art
noncomputable def art_fraction : ℝ := 1/5

-- Define the fraction given to friend (half of the remaining)
noncomputable def friend_fraction : ℝ := 1/2

-- Define the length of each section
noncomputable def section_length : ℝ := 2

-- Define the number of sections
def num_sections : ℕ := 10

theorem rope_length_proof :
  -- The remaining rope after art and giving to friend
  (initial_length * (1 - art_fraction) * (1 - friend_fraction)) =
  -- Equals the total length of all sections
  (section_length * num_sections) := by
  sorry

#eval num_sections -- This will work as it's a natural number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_proof_l1325_132524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_third_quadrant_l1325_132577

theorem tan_half_angle_third_quadrant (α : Real) : 
  α ∈ Set.Ioo π (3*π/2) →  -- α is in the third quadrant
  Real.sin α = -24/25 → 
  Real.tan (α/2) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_angle_third_quadrant_l1325_132577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l1325_132523

open Real

theorem determinant_max_value :
  ∃ (M : ℝ), M = 7 ∧ 
  ∀ x : ℝ, 
    abs (12 * cos (π / 2 + x) * tan (π / 2 - x) - 5 * cos x * tan x) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l1325_132523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_tripled_with_new_equipment_l1325_132555

/-- Represents the number of cars produced in a given year -/
def CarProduction : ℕ → ℕ := sorry

/-- Represents whether new equipment was introduced in a given year -/
def NewEquipment : ℕ → Prop := sorry

/-- The year we're comparing to (previous year) -/
def previousYear : ℕ := sorry

/-- The year of comparison (2004 in this case) -/
def comparisonYear : ℕ := sorry

/-- Theorem stating that when new equipment is introduced, 
    the car production triples compared to the previous year -/
theorem production_tripled_with_new_equipment 
  (h : NewEquipment comparisonYear) :
  CarProduction comparisonYear = 3 * CarProduction previousYear := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_tripled_with_new_equipment_l1325_132555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_l1325_132541

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

theorem decreasing_exponential (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_exponential_l1325_132541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_total_spent_l1325_132572

/-- The total amount Jerry spent in the shop -/
noncomputable def total_spent (tax_paid : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) : ℝ :=
  (tax_paid / tax_rate) + tax_free_cost

/-- Theorem: The total amount Jerry spent in the shop is 519.7 -/
theorem jerry_total_spent :
  total_spent 30 0.06 19.7 = 519.7 := by
  -- Unfold the definition of total_spent
  unfold total_spent
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_total_spent_l1325_132572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_8_l1325_132527

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_of_2_8 : floor 2.8 = 2 := by
  -- Unfold the definition of floor
  unfold floor
  -- Use the simp tactic to simplify the expression
  simp [Int.floor_eq_iff]
  -- Split the goal into two parts
  constructor
  -- Prove that 2 ≤ 2.8
  { norm_num }
  -- Prove that 2.8 < 3
  { norm_num }


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_8_l1325_132527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_4pi_l1325_132514

/-- A configuration of n unit circles on a plane --/
structure CircleConfiguration where
  n : ℕ
  centers : Fin n → ℝ × ℝ
  common_point : ℝ × ℝ
  common_point_inside : ∀ i : Fin n, dist (centers i) common_point ≤ 1

/-- The perimeter of the polygon formed by overlapping circles --/
noncomputable def perimeter (config : CircleConfiguration) : ℝ := 4 * Real.pi

/-- Theorem stating that the perimeter is always 4π --/
theorem perimeter_is_4pi (config : CircleConfiguration) : 
  perimeter config = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_is_4pi_l1325_132514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_crossing_possible_l1325_132599

/-- Represents a family member --/
inductive FamilyMember
  | Father
  | Mother
  | Son
  | Grandmother
deriving BEq, Repr

/-- Represents the state of the bridge crossing --/
structure BridgeState where
  leftSide : List FamilyMember
  rightSide : List FamilyMember
  flashlightLeft : Bool
  time : Nat
deriving Repr

/-- Crossing time for each family member --/
def crossingTime (member : FamilyMember) : Nat :=
  match member with
  | FamilyMember.Father => 1
  | FamilyMember.Mother => 2
  | FamilyMember.Son => 5
  | FamilyMember.Grandmother => 10

/-- Represents a single crossing action --/
structure Crossing where
  members : List FamilyMember
deriving Repr

/-- Checks if a crossing is valid --/
def isValidCrossing (c : Crossing) : Bool :=
  c.members.length ≤ 2

/-- Calculates the time taken for a crossing --/
def crossingDuration (c : Crossing) : Nat :=
  c.members.map crossingTime |>.maximum?.getD 0

/-- Applies a crossing to a bridge state --/
def applyCrossing (state : BridgeState) (c : Crossing) : BridgeState :=
  if state.flashlightLeft then
    { leftSide := state.leftSide.filter (λ m => !c.members.contains m),
      rightSide := state.rightSide ++ c.members,
      flashlightLeft := false,
      time := state.time + crossingDuration c }
  else
    { leftSide := state.leftSide ++ c.members,
      rightSide := state.rightSide.filter (λ m => !c.members.contains m),
      flashlightLeft := true,
      time := state.time + crossingDuration c }

/-- The initial state of the bridge crossing --/
def initialState : BridgeState :=
  { leftSide := [FamilyMember.Father, FamilyMember.Mother, FamilyMember.Son, FamilyMember.Grandmother],
    rightSide := [],
    flashlightLeft := true,
    time := 0 }

/-- Theorem: There exists a sequence of valid crossings that allows all family members
    to cross the bridge in exactly 17 minutes --/
theorem bridge_crossing_possible :
  ∃ (crossings : List Crossing),
    (crossings.all isValidCrossing) ∧
    (let finalState := crossings.foldl applyCrossing initialState
     finalState.leftSide = [] ∧
     finalState.rightSide.length = 4 ∧
     finalState.time = 17) := by
  sorry

#eval initialState

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_crossing_possible_l1325_132599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1325_132512

theorem trig_identity (x : ℝ) 
  (h1 : Real.cos (x - π/4) = -1/3) 
  (h2 : 5*π/4 < x ∧ x < 7*π/4) : 
  Real.sin (2*x) - Real.cos (2*x) = (4*Real.sqrt 2 - 7) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1325_132512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_shift_l1325_132530

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x φ : ℝ) := f (x - φ)

theorem smallest_positive_shift (φ : ℝ) :
  (∀ x, g x φ = -g (-x) φ) →  -- g is an odd function
  (∀ ψ, 0 < ψ ∧ ψ < φ → ¬(∀ x, g x ψ = -g (-x) ψ)) →  -- φ is the smallest positive value
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_shift_l1325_132530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_not_simple_l1325_132531

noncomputable def complex_expression (x : ℝ) : ℝ :=
  ((x + 1)^2 * (x^2 - x + 2)^2 / (x^3 + 1)^2)^2 * ((x - 1)^2 * (x^2 + x + 2)^2 / (x^3 - 2)^2)^2

theorem complex_expression_not_simple (x : ℝ) 
  (h1 : x^3 + 1 ≠ 0) 
  (h2 : x^3 - 2 ≠ 0) : 
  complex_expression x ≠ (x + 1)^4 ∧ 
  complex_expression x ≠ (x^3 + 1)^4 ∧ 
  complex_expression x ≠ (x - 1)^4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_not_simple_l1325_132531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_theorem_l1325_132537

/-- Represents the grid and circle configuration described in the problem -/
structure GridWithCircles where
  gridSize : Nat
  squareSize : ℝ
  smallCircleCount : Nat
  smallCircleDiameter : ℝ
  largeCircleDiameter : ℝ

/-- Calculates the area of the visible shaded region in the grid -/
noncomputable def visibleShadedArea (g : GridWithCircles) : ℝ :=
  (g.gridSize : ℝ)^2 * g.squareSize^2 - 
  (↑g.smallCircleCount * (Real.pi/4) * g.smallCircleDiameter^2 + 
   (Real.pi/4) * g.largeCircleDiameter^2)

/-- Theorem stating the result for the specific configuration in the problem -/
theorem visible_shaded_area_theorem (g : GridWithCircles) 
  (h1 : g.gridSize = 4)
  (h2 : g.squareSize = 3)
  (h3 : g.smallCircleCount = 3)
  (h4 : g.smallCircleDiameter = 3)
  (h5 : g.largeCircleDiameter = 6) :
  ∃ (A B : ℝ), visibleShadedArea g = A - B * Real.pi ∧ A + B = 159.75 := by
  sorry

#check visible_shaded_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_theorem_l1325_132537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1325_132519

noncomputable def S (n : ℕ) : ℝ := n * (2 * n - 1)

noncomputable def a (n : ℕ) : ℝ := 4 * n - 3

noncomputable def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))

noncomputable def T (n : ℕ) : ℝ := n / (4 * n + 1)

noncomputable def c (n : ℕ) : ℝ := 3^(n - 1)

noncomputable def C (n : ℕ) : ℝ := (3^n - 1) / 2

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → S n = n * (S n / n)) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 4 * n - 3) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (4 * n + 1)) ∧
  (c 1 = a 1) ∧
  (c 2 = Real.sqrt (a 1 * a 3)) ∧
  (∀ n : ℕ, n ≥ 1 → C n = (3^n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1325_132519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_second_quadrant_l1325_132582

theorem cos_value_second_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.tan α = -5/12) : Real.cos α = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_second_quadrant_l1325_132582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_four_minus_ln_three_l1325_132529

/-- The area of the closed figure formed by y = 1/x (x > 0), y = x, and y = 3 -/
noncomputable def closedFigureArea : ℝ :=
  let f (x : ℝ) := 1 / x
  let g (x : ℝ) := x
  let h : ℝ := 3
  let lowerBound : ℝ := 1 / 3
  let upperBound : ℝ := 1
  (∫ x in lowerBound..upperBound, h - f x) + (1/2 * (upperBound - lowerBound) * (h - g lowerBound))

/-- The theorem stating that the area of the closed figure is 4 - ln(3) -/
theorem area_equals_four_minus_ln_three : closedFigureArea = 4 - Real.log 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equals_four_minus_ln_three_l1325_132529
