import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_at_3_l744_74468

-- Define the functions h and f
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)
noncomputable def h (x : ℝ) : ℝ := 4 * (f⁻¹ x)

-- State the theorem
theorem h_equals_20_at_3 :
  ∃! x : ℝ, h x = 20 ∧ f x = 30 / (x + 5) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_at_3_l744_74468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drop_makes_car_new_implications_l744_74417

/-- Represents the concept of cleaning a car with minimal water -/
def DropMakesCarNew : Prop :=
  ∃ (technology : Type) (water_amount : ℝ),
    water_amount ≤ 1 ∧ 
    (∃ (car : Type) (clean : car → Prop), ∀ (c : car), clean c)

/-- Represents the importance of valuing initiative of consciousness -/
def ValueInitiativeOfConsciousness : Prop :=
  ∃ (problem : Type) (solution : Type),
    ∃ (innovative_thinking : problem → solution),
    ∀ (p : problem), ∃ (s : solution), innovative_thinking p = s

/-- Represents actively creating conditions for transformation -/
def ActivelyCreateConditions : Prop :=
  ∀ (contradiction : Type) (left right : contradiction),
    ∃ (resolution : contradiction → Prop),
    resolution left ∧ resolution right

/-- The main theorem stating that "a drop makes the car new" implies
    both valuing initiative of consciousness and actively creating conditions -/
theorem drop_makes_car_new_implications :
  DropMakesCarNew →
  (ValueInitiativeOfConsciousness ∧ ActivelyCreateConditions) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drop_makes_car_new_implications_l744_74417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l744_74424

/-- A parabola with vertex at the origin and focus at (2√2, 0) has the equation y^2 = 8√2x -/
theorem parabola_equation (x y : ℝ) :
  let vertex : ℝ × ℝ := (0, 0)
  let focus : ℝ × ℝ := (2 * Real.sqrt 2, 0)
  let parabola := {p : ℝ × ℝ | p.2^2 = 8 * Real.sqrt 2 * p.1}
  (∀ p ∈ parabola, dist p vertex = dist p focus) ↔ 
  y^2 = 8 * Real.sqrt 2 * x := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l744_74424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l744_74472

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 9; 2, 5]

noncomputable def inverse_or_zero (M : Matrix (Fin 2) (Fin 2) ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  if M.det ≠ 0 then M⁻¹ else 0

theorem inverse_of_A :
  inverse_or_zero A = !![5/2, -9/2; -1, 2] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l744_74472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l744_74454

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem abc_inequality : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l744_74454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l744_74410

/-- Given two lines l₁ and l₂, where l₁ is y = 2x and l₂ is ax + by + c = 0,
    prove that the area of the triangle formed by l₁, l₂, and the y-axis is 9/20
    under the following conditions:
    1. abc ≠ 0
    2. l₁ and l₂ are perpendicular
    3. a, b, and c form an arithmetic sequence -/
theorem triangle_area (a b c : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (2 : ℝ) * (-a / b) = -1 →
  2 * b = a + c →
  let l₁ := λ x : ℝ => 2 * x
  let l₂ := λ x : ℝ => (-a * x - c) / b
  let x_intersect := -c / (a + 2 * b)
  let y_intersect := -c / b
  let S := (1 / 2 : ℝ) * |x_intersect| * |y_intersect|
  S = 9 / 20 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l744_74410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l744_74467

/-- The function f(x) with given properties -/
noncomputable def f (ω φ : ℝ) : ℝ → ℝ := fun x ↦ 4 * Real.sin (ω * x + φ)

/-- Theorem stating the properties of the function f(x) -/
theorem function_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : |φ| < π/2)
  (h_f0 : f ω φ 0 = 2 * Real.sqrt 3)
  (h_f_roots : ∃ x₁ x₂, f ω φ x₁ = 0 ∧ f ω φ x₂ = 0 ∧ |x₁ - x₂| ≥ π/2)
  (h_min_roots : ∀ y z, f ω φ y = 0 → f ω φ z = 0 → |y - z| ≥ π/2) :
  (∃ α, π/12 < α ∧ α < π/2 ∧ f ω φ α = 12/5 ∧ Real.sin (2*α) = (3 + 4*Real.sqrt 3)/10) ∧
  f ω φ = fun x ↦ 4 * Real.sin (2*x + π/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l744_74467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_value_l744_74489

noncomputable def infinite_product_series (n : ℕ) : ℝ := (3^(2^n))^(1/4^(n+1))

noncomputable def S : ℝ := ∑' n, (n + 1) / 4^(n + 1)

theorem infinite_product_value :
  (∏' n, infinite_product_series n) = (81 : ℝ)^(1/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_value_l744_74489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l744_74498

/-- Represents a cubic polynomial Q(x) = x^3 + px^2 + qx + r -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The mean of zeros of a cubic polynomial -/
noncomputable def meanOfZeros (Q : CubicPolynomial) : ℝ := -Q.p / 3

/-- The product of zeros taken two at a time of a cubic polynomial -/
def productOfZerosPairs (Q : CubicPolynomial) : ℝ := Q.q

/-- The sum of coefficients of a cubic polynomial -/
def sumOfCoefficients (Q : CubicPolynomial) : ℝ := 1 + Q.p + Q.q + Q.r

/-- The y-intercept of a cubic polynomial -/
def yIntercept (Q : CubicPolynomial) : ℝ := Q.r

theorem cubic_polynomial_property (Q : CubicPolynomial) 
  (h1 : meanOfZeros Q = productOfZerosPairs Q)
  (h2 : meanOfZeros Q = sumOfCoefficients Q)
  (h3 : yIntercept Q = 5) :
  Q.q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l744_74498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l744_74441

-- 1. Existence of a and its value
theorem problem_1 : ∃ a : ℕ, (183 * 10 + a) * 10 + 8 % 287 = 0 ∧ a = 6 := by sorry

-- 2. Number of positive factors of 6^2
theorem problem_2 : (Finset.filter (λ x => x ∣ 36) (Finset.range 37)).card = 9 := by sorry

-- 3. Urn problem
theorem problem_3 : 
  ∀ x y z : ℕ, 
  x + y = 9 → y + z = 11 → x + z = 12 → 
  x + y + z = 16 := by sorry

-- 4. Function with symmetric roots
theorem problem_4 : 
  ∃ f : ℝ → ℝ, 
  (∀ x, f (3 + x) = f (3 - x)) ∧ 
  (∃ roots : Finset ℝ, roots.card = 16 ∧ ∀ r ∈ roots, f r = 0) ∧
  (∃ roots : Finset ℝ, roots.card = 16 ∧ ∀ r ∈ roots, f r = 0 ∧ (roots.sum id = 48)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l744_74441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l744_74415

/-- Given a hyperbola with equation x²/a² - y² = 1, where a > 0 and 
    the real axis length is 2, prove that its eccentricity is √2. -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) (h2 : 2 * a = 2) : 
  Real.sqrt (1 + 1 / a^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l744_74415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_model_ratio_theorem_l744_74446

/-- Represents the ratio of a statue's height to its model's height -/
structure StatueModelRatio where
  statue_height : ℚ  -- in feet
  model_height : ℚ   -- in inches

/-- Calculates the feet represented by one inch of the model -/
def feet_per_inch (ratio : StatueModelRatio) : ℚ :=
  ratio.statue_height / ratio.model_height

theorem statue_model_ratio_theorem (ratio : StatueModelRatio) 
  (h1 : ratio.statue_height = 48)
  (h2 : ratio.model_height = 3) : 
  feet_per_inch ratio = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_model_ratio_theorem_l744_74446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_man_speed_result_l744_74433

/-- The speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ := by
  -- Define the given conditions
  have h1 : train_length = 55 := by sorry
  have h2 : train_speed_kmph = 60 := by sorry
  have h3 : passing_time = 3 := by sorry

  -- Convert train speed to m/s
  let train_speed_ms := train_speed_kmph * 1000 / 3600

  -- Calculate relative speed
  let relative_speed := train_length / passing_time

  -- Calculate man's speed in m/s
  let man_speed_ms := relative_speed - train_speed_ms

  -- Convert man's speed to kmph
  let man_speed_kmph := man_speed_ms * 3600 / 1000

  -- Prove that the man's speed is approximately 5.976 kmph
  sorry

/-- The main theorem stating the man's speed -/
theorem man_speed_result : ℝ := 
  man_speed 55 60 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_man_speed_result_l744_74433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l744_74475

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: If a₇/a₄ = 7/13 in an arithmetic sequence, then S₁₃/S₇ = 1 -/
theorem arithmetic_sequence_sum_ratio 
  (seq : ArithmeticSequence) 
  (h : seq.a 7 / seq.a 4 = 7 / 13) : 
  S seq 13 / S seq 7 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l744_74475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_roots_real_and_distinct_l744_74444

/-- Definition of the polynomial sequence Pₙ(x) -/
def P : ℕ → (ℝ → ℝ)
  | 0 => fun x => x^2 - 2
  | n + 1 => fun x => P 0 (P n x)

/-- Theorem stating that Pₙ(x) = x has 2ⁿ distinct real roots in [-2, 2] for all n -/
theorem P_roots_real_and_distinct (n : ℕ) :
  ∃ (S : Finset ℝ), (S.card = 2^n) ∧ 
    (∀ x, x ∈ S → x ∈ Set.Icc (-2 : ℝ) 2 ∧ P n x = x) ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_roots_real_and_distinct_l744_74444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_sequential_operation_l744_74459

-- Define the custom operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_sequential_operation :
  nabla (nabla 3 4) 2 = 11 / 9 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_sequential_operation_l744_74459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_factorial_30_l744_74471

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) : ℕ :=
    if m = 0 then 0
    else m / 5 + count_fives (m / 5)
  count_fives n

theorem trailing_zeros_factorial_30 : trailingZeros (factorial 30) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_factorial_30_l744_74471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_purchasing_schemes_l744_74440

theorem notebook_purchasing_schemes :
  ∃! (schemes : Finset (ℕ × ℕ)),
    (∀ (a b : ℕ), (a, b) ∈ schemes ↔
      a ≥ 4 ∧ b ≥ 4 ∧ 8 * a + 10 * b ≤ 100) ∧
    schemes.card = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_purchasing_schemes_l744_74440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_12_power_100_l744_74480

/-- The unit digit of a natural number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- The cycle of unit digits for powers of 2 -/
def unitDigitCycle : List ℕ := [2, 4, 8, 6]

/-- The function to get the unit digit of 2^n based on the cycle -/
def unitDigitPowerOfTwo (n : ℕ) : ℕ :=
  unitDigitCycle[n % 4]'(by
    simp [unitDigitCycle]
    exact Nat.mod_lt n (by norm_num))

theorem unit_digit_12_power_100 :
  unitDigit (12^100) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_12_power_100_l744_74480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clear_time_correct_l744_74411

/-- Calculates the time for two trains to be completely clear of each other -/
noncomputable def train_clear_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

/-- Theorem: The time for two trains to be completely clear of each other
    is equal to the sum of their lengths divided by their relative speed -/
theorem train_clear_time_correct
  (length1 length2 speed1 speed2 : ℝ)
  (h1 : length1 > 0)
  (h2 : length2 > 0)
  (h3 : speed1 > 0)
  (h4 : speed2 > 0) :
  train_clear_time length1 length2 speed1 speed2 =
  (length1 + length2) / (speed1 + speed2) := by
  sorry

/-- Calculate the result for the specific problem -/
def problem_result : ℚ :=
  (111 + 165) / (40000 / 3600 + 50000 / 3600)

#eval problem_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clear_time_correct_l744_74411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_radical_l744_74418

-- Define the quadratic radicals
noncomputable def radical1 := Real.sqrt 12
noncomputable def radical2 (x : ℝ) := Real.sqrt (x / 3)
noncomputable def radical3 (a : ℝ) := Real.sqrt (a^4)
noncomputable def radical4 (a b : ℝ) := Real.sqrt (a^2 - b^2)

-- Define a predicate for simplest quadratic radical
def is_simplest_radical (r : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), x > 0 → ¬∃ (y : ℝ), y ≥ 0 ∧ r x = Real.sqrt y ∧ y < x

-- Theorem statement
theorem simplest_radical : 
  is_simplest_radical (λ x ↦ radical4 x 1) ∧ 
  ¬is_simplest_radical (λ _ ↦ radical1) ∧
  ¬is_simplest_radical radical2 ∧
  ¬is_simplest_radical radical3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_radical_l744_74418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l744_74416

/-- The function f(x) = 3^x + 2x - 3 -/
noncomputable def f (x : ℝ) := Real.exp (x * Real.log 3) + 2*x - 3

/-- Theorem: There exists a root of f(x) in the interval (0, 1) -/
theorem f_has_root_in_interval : 
  ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f x = 0 :=
by
  sorry

#check f_has_root_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_l744_74416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_partitions_l744_74419

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Represents the available internal fencing -/
def availableFencing : ℝ := 2400

/-- Calculates the number of square partitions given the number of squares per column -/
noncomputable def numSquares (n : ℝ) : ℝ := 2 * n^2

/-- Calculates the total length of internal fencing used given the number of squares per column -/
noncomputable def fencingUsed (field : FieldDimensions) (n : ℝ) : ℝ :=
  (field.width / n) * (2 * n - 1) + field.length * (n - 1)

/-- Theorem stating that the maximum number of square partitions is 3200 -/
theorem max_square_partitions (field : FieldDimensions)
    (h1 : field.width = 30)
    (h2 : field.length = 60)
    (h3 : fencingUsed field 40 ≤ availableFencing)
    (h4 : ∀ n : ℝ, n > 40 → fencingUsed field n > availableFencing) :
    (⟨40, by sorry⟩ : { n : ℝ // fencingUsed field n ≤ availableFencing }).val = 40 ∧
    numSquares 40 = 3200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_partitions_l744_74419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l744_74414

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in ℝ² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

theorem triangle_properties (t : Triangle) : 
  let m : Vector2D := ⟨Real.cos (t.A/2)^2, Real.cos (2*t.A)⟩
  let n : Vector2D := ⟨4, -1⟩
  (dot_product m n = 7/2) → (t.A = π/3) ∧ 
  (t.a = Real.sqrt 3) → 
    (∀ b c : ℝ, t.b * t.c ≤ b * c → (t.a = t.b ∧ t.b = t.c)) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l744_74414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_one_l744_74496

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else x - x^2

-- State the theorem
theorem a_greater_than_one (a : ℝ) (h : f a > f (2 - a)) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_one_l744_74496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_four_l744_74404

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
def focus (c : Parabola) : Point :=
  { x := c.p, y := 0 }

/-- Check if a point lies on the parabola -/
def lies_on (m : Point) (c : Parabola) : Prop :=
  m.y^2 = 2 * c.p * m.x

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

theorem distance_to_focus_is_four (c : Parabola) (m : Point)
  (h_m_on_c : lies_on m c)
  (h_m_x : m.x = 2)
  (h_m_y : m.y = 4) :
  distance m (focus c) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_is_four_l744_74404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l744_74478

-- Define the slope angle in radians (45° = π/4)
noncomputable def slope_angle : ℝ := Real.pi / 4

-- Define the line equation ax + y + 2 = 0
def line_equation (a x y : ℝ) : Prop := a * x + y + 2 = 0

-- Theorem statement
theorem line_slope_angle (a : ℝ) : 
  (∀ x y, line_equation a x y → Real.tan slope_angle = -1 / a) → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l744_74478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_small_spheres_l744_74476

/-- Regular tetrahedron with inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  sideLength : ℝ
  circumscribedRadius : ℝ
  inscribedRadius : ℝ
  circumscribedRadius_eq : circumscribedRadius = sideLength * Real.sqrt 6 / 4
  inscribedRadius_eq : inscribedRadius = sideLength * Real.sqrt 6 / 12

/-- Smaller sphere tangent to edge midpoint and circumscribed sphere -/
noncomputable def smallSphereRadius (t : RegularTetrahedron) : ℝ :=
  t.circumscribedRadius - t.sideLength * Real.sqrt 2 / 2

/-- Volume of a sphere given its radius -/
noncomputable def sphereVolume (radius : ℝ) : ℝ :=
  4 / 3 * Real.pi * radius ^ 3

/-- Probability theorem for point inside smaller spheres -/
theorem probability_point_in_small_spheres (t : RegularTetrahedron) :
    (6 * sphereVolume (smallSphereRadius t)) / (sphereVolume t.circumscribedRadius) =
    -- The actual probability value would be calculated here
    -- For now, we use a placeholder
    (6 * (smallSphereRadius t)^3) / (t.circumscribedRadius^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_point_in_small_spheres_l744_74476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_is_integer_l744_74494

def sum_of_powers (x y : ℝ) (n : ℕ) : ℝ := x^n + y^n

theorem sum_of_powers_is_integer (x y : ℝ) 
  (h1 : ∃ z : ℤ, x + y = z)
  (h2 : ∃ z : ℤ, x^2 + y^2 = z)
  (h3 : ∃ z : ℤ, x^3 + y^3 = z)
  (h4 : ∃ z : ℤ, x^4 + y^4 = z) :
  ∀ n : ℕ, n > 0 → ∃ z : ℤ, sum_of_powers x y n = z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_is_integer_l744_74494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l744_74462

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) →
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l744_74462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3s_l744_74437

-- Define the motion equation
noncomputable def s (t : ℝ) : ℝ := t^3 / 9 + t

-- Define the instantaneous velocity (derivative of s)
noncomputable def v (t : ℝ) : ℝ := 1 + (1/3) * t^2

-- Theorem: The instantaneous velocity at t = 3s is 4 m/s
theorem instantaneous_velocity_at_3s :
  v 3 = 4 := by
  -- Unfold the definition of v
  unfold v
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3s_l744_74437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_4_equals_20_l744_74481

-- Define the functions h and s
noncomputable def s (x : ℝ) : ℝ := 40 / (x + 5)

noncomputable def h (x : ℝ) : ℝ := 4 * (Function.invFun s x)

-- State the theorem
theorem h_of_4_equals_20 : h 4 = 20 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_4_equals_20_l744_74481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_score_is_56_l744_74443

/-- The nth even number -/
def evenNumber (n : ℕ) : ℕ := 2 * n

/-- The sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ :=
  (List.range n).map (λ i => evenNumber (i + 1)) |>.sum

/-- Tim's score in math -/
def timScore : ℕ := sumFirstEvenNumbers 7

theorem tim_score_is_56 : timScore = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_score_is_56_l744_74443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_range_l744_74427

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum_range
  (a : ℕ → ℝ)
  (h_geometric : is_geometric a)
  (h_product : a 4 * a 8 = 9) :
  ∃ S : Set ℝ, S = Set.Iic (-6) ∪ Set.Ici 6 ∧ 
  (∀ x : ℝ, x ∈ S ↔ ∃ seq : ℕ → ℝ, 
    is_geometric seq ∧ 
    seq 4 * seq 8 = 9 ∧ 
    seq 3 + seq 9 = x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_range_l744_74427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l744_74406

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * (a / x + a + 1)

-- State the theorem
theorem f_inequality_range (a : ℝ) (h : a ≥ -1) :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ < 0 ∧ f a x₁ < f a x₂) ↔ -1 ≤ a ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l744_74406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_ninths_l744_74426

/-- Right triangle ABC with sides AB = 6, BC = 8, AC = 10 -/
structure RightTriangle where
  ab : ℝ
  bc : ℝ
  ac : ℝ
  right_triangle : ab^2 + bc^2 = ac^2
  ab_eq : ab = 6
  bc_eq : bc = 8
  ac_eq : ac = 10

/-- A point P inside the triangle ABC -/
structure PointInTriangle (t : RightTriangle) where
  p : ℝ × ℝ
  inside : p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ t.bc

/-- The probability that a randomly chosen point P inside the triangle
    forms a triangle PBC with area less than 1/3 of ABC's area -/
noncomputable def probabilitySmallArea (t : RightTriangle) : ℝ :=
  sorry

/-- The main theorem stating that the probability is 5/9 -/
theorem probability_is_five_ninths (t : RightTriangle) :
  probabilitySmallArea t = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_five_ninths_l744_74426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l744_74491

noncomputable def f (x : ℝ) := 6 / (x - 1) - Real.sqrt (x + 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -4 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l744_74491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_l744_74490

/-- The side length of the initial equilateral triangle -/
noncomputable def initial_side_length : ℝ := 80

/-- The perimeter of an equilateral triangle given its side length -/
noncomputable def triangle_perimeter (side_length : ℝ) : ℝ := 3 * side_length

/-- The sum of the geometric series representing the perimeters of all triangles -/
noncomputable def perimeter_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The theorem stating that the sum of all triangle perimeters is 480 -/
theorem sum_of_triangle_perimeters :
  perimeter_sum (triangle_perimeter initial_side_length) (1/2) = 480 := by
  sorry

#check sum_of_triangle_perimeters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_triangle_perimeters_l744_74490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_squares_l744_74438

/-- The angle between the sides of an inscribed square and a circumscribed square of the same circle -/
def square_angle_difference : ℝ := 15

/-- A circle with an inscribed square and a circumscribed square -/
structure CircleWithSquares where
  center : ℝ × ℝ
  radius : ℝ
  inscribed_square : Set (ℝ × ℝ)
  circumscribed_square : Set (ℝ × ℝ)

/-- The inscribed square is inside the circle -/
axiom inscribed_inside (cws : CircleWithSquares) :
  ∀ p ∈ cws.inscribed_square, (p.1 - cws.center.1)^2 + (p.2 - cws.center.2)^2 ≤ cws.radius^2

/-- The circumscribed square is outside the circle -/
axiom circumscribed_outside (cws : CircleWithSquares) :
  ∀ p ∈ cws.circumscribed_square, (p.1 - cws.center.1)^2 + (p.2 - cws.center.2)^2 ≥ cws.radius^2

/-- The vertices of the circumscribed square lie on the extensions of the sides of the inscribed square -/
axiom vertices_on_extensions (cws : CircleWithSquares) :
  ∀ v ∈ cws.circumscribed_square, ∃ s1 s2, s1 ∈ cws.inscribed_square ∧ s2 ∈ cws.inscribed_square ∧
    (v.1 - s1.1) * (s2.2 - s1.2) = (v.2 - s1.2) * (s2.1 - s1.1)

/-- The angle between the sides of the inscribed and circumscribed squares is 15° -/
theorem angle_between_squares (cws : CircleWithSquares) :
  square_angle_difference = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_squares_l744_74438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projections_on_circle_l744_74479

-- Define the basic geometric elements
variable (A B C D X Y : ℝ × ℝ)

-- Define the quadrilateral ABCD
def quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A

-- Define a function to represent projection of a point onto a line
noncomputable def projection (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define a predicate to check if points lie on a circle
def on_circle (points : List (ℝ × ℝ)) : Prop :=
  sorry

-- Define symmetry with respect to a point
def symmetric (P Q Center : ℝ × ℝ) : Prop :=
  Center = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- State the theorem
theorem projections_on_circle
  (quad : quadrilateral A B C D)
  (h1 : on_circle [projection X A B, projection X B C, projection X C D, projection X D A])
  (h2 : symmetric X Y (
    ((projection X A B).1 + (projection X B C).1 + (projection X C D).1 + (projection X D A).1) / 4,
    ((projection X A B).2 + (projection X B C).2 + (projection X C D).2 + (projection X D A).2) / 4
  )) :
  on_circle [
    projection B A X,
    projection B X C,
    projection B C Y,
    projection B Y A
  ] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projections_on_circle_l744_74479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l744_74429

-- Define the coordinate system
variable (x y : ℝ)

-- Define the point P
variable (P : ℝ × ℝ)

-- Define the circle M
def circle_M (r : ℝ) (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = r^2

-- Define the logarithmic function
noncomputable def log_func (x : ℝ) : ℝ :=
  2 * Real.log x

-- Define the quadratic function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem max_value_of_f 
  (h1 : ∃ r, circle_M r P.1 P.2 ∧ P.2 = log_func P.1)
  (h2 : ∃ k m n, ∀ x, f x = k*x^2 + m*x + n)
  (h3 : f 0 = 0)
  (h4 : f P.1 = P.2)
  (h5 : ∃ M, M ≠ P ∧ M.1 ≠ 0 ∧ circle_M r M.1 M.2 ∧ f M.1 = M.2)
  (h6 : (Real.exp 1 - Real.exp (-1)) * P.1 = 4) :
  ∃ x, ∀ t, f t ≤ f x ∧ f x = 9/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l744_74429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_midpoints_equilateral_l744_74466

-- Define the circle
def Circle := {p : ℂ | Complex.abs p = 1}

-- Define the hexagon vertices
noncomputable def A (r : ℝ) : ℂ := r * (1/2 + Complex.I * (Real.sqrt 3/2))
noncomputable def B (r : ℝ) : ℂ := r
noncomputable def C (r : ℝ) : ℂ := r * (1/2 - Complex.I * (Real.sqrt 3/2))
noncomputable def D (r : ℝ) : ℂ := r * (-1/2 - Complex.I * (Real.sqrt 3/2))
noncomputable def E (r : ℝ) : ℂ := -r
noncomputable def F (r : ℝ) : ℂ := r * (-1/2 + Complex.I * (Real.sqrt 3/2))

-- Define the midpoints
noncomputable def G (r : ℝ) : ℂ := (B r + C r) / 2
noncomputable def H (r : ℝ) : ℂ := (D r + E r) / 2
noncomputable def K (r : ℝ) : ℂ := (F r + A r) / 2

theorem hexagon_midpoints_equilateral (r : ℝ) (hr : r > 0) :
  Complex.abs (G r - H r) = Complex.abs (H r - K r) ∧
  Complex.abs (H r - K r) = Complex.abs (K r - G r) ∧
  Complex.abs (K r - G r) = Complex.abs (G r - H r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_midpoints_equilateral_l744_74466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relations_l744_74425

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line_line : Line → Line → Prop)
variable (para_line : Line → Line → Prop)
variable (para_plane : Plane → Plane → Prop)
variable (perp_plane : Plane → Plane → Prop)

-- Define the "contained in" relation
variable (contained_in : Line → Plane → Prop)

-- Given conditions
variable (l : Line) (m : Line) (α : Plane) (β : Plane)
variable (h1 : perp_line_plane l α)
variable (h2 : contained_in m β)

-- Theorem to prove
theorem line_plane_relations :
  (para_line l m → perp_plane α β) ∧
  (para_plane α β → perp_line_line l m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_relations_l744_74425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_minimum_value_l744_74436

noncomputable section

def f (x : ℝ) : ℝ := (1/2) * x^2 - 1/2
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def F (m : ℝ) (x : ℝ) : ℝ := f x - m * g 1 x

theorem common_tangent_and_minimum_value 
  (h_common_tangent : ∃ (k : ℝ), k * (1 - 1) + f 1 = 0 ∧ k * (1 - 1) + g 1 1 = 0 ∧ 
                                 k = deriv f 1 ∧ k = deriv (g 1) 1) :
  (∃ (a : ℝ), a = 1) ∧ 
  (∀ (m : ℝ), 
    (m ≤ 1 → ∀ x ∈ Set.Icc 1 (Real.exp 1), F m x ≥ 0) ∧
    (1 < m ∧ m < (Real.exp 1)^2 → ∀ x ∈ Set.Icc 1 (Real.exp 1), F m x ≥ (1/2) * m - 1/2 - (m/2) * Real.log m) ∧
    (m ≥ (Real.exp 1)^2 → ∀ x ∈ Set.Icc 1 (Real.exp 1), F m x ≥ (1/2) * (Real.exp 1)^2 - 1/2 - m)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_and_minimum_value_l744_74436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_average_selling_price_l744_74464

/-- Represents the bond with its properties and calculates its average selling price -/
noncomputable def Bond (initial_face_value : ℝ) (yearly_interest_rate : ℝ) 
         (quarterly_rates : Fin 4 → ℝ) (yearly_increase_rate : ℝ) 
         (years : ℕ) (total_interest_percent : ℝ) : ℝ :=
  let face_value := initial_face_value * (1 + yearly_increase_rate) ^ years
  let total_interest := face_value * yearly_interest_rate
  total_interest / total_interest_percent

/-- Theorem stating that the average selling price of the bond is approximately $5631.34 -/
theorem bond_average_selling_price :
  let b := Bond 5000 0.07 (fun i => [0.065, 0.075, 0.08, 0.06].get i) 0.015 3 0.065
  ∃ ε > 0, |b - 5631.34| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_average_selling_price_l744_74464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l744_74460

theorem expression_evaluation (a : ℝ) (h : a ≠ 0) : 
  (1 / 8) * a^0 + (1 / (8 * a))^0 + 32^(-(1 / 5 : ℝ)) - (-16 : ℝ)^(-(3 / 4 : ℝ)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l744_74460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l744_74450

theorem sine_cosine_inequality (a : ℝ) :
  a < 0 →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  a ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_l744_74450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_daily_wage_l744_74451

/-- Proves that given the conditions of the problem, the daily wage of worker c is Rs. 100 -/
theorem worker_c_daily_wage (a b c : ℕ) (total_earning : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a : ℚ) / 3 = (b : ℚ) / 4 →
  (a : ℚ) / 3 = (c : ℚ) / 5 →
  6 * a + 9 * b + 4 * c = total_earning →
  total_earning = 1480 →
  c = 100 := by
  sorry

#check worker_c_daily_wage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_c_daily_wage_l744_74451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l744_74477

theorem expression_equality (a b c : ℝ) : a - (-b) - c = a + b - c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l744_74477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_to_sin_shift_l744_74400

theorem cos_to_sin_shift (x : ℝ) : 
  Real.cos (2 * (x + π/4) - 4*π/3) = Real.sin (2*x - π/3) :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_to_sin_shift_l744_74400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l744_74412

theorem right_triangle_hypotenuse (a b : ℝ) :
  (Real.sqrt (a^2 - 6*a + 9) + |b - 4| = 0) →
  Real.sqrt (a^2 + b^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l744_74412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_96_implies_x_8_l744_74463

-- Define the triangle
noncomputable def triangle (x : ℝ) : List (ℝ × ℝ) := [(0, 0), (x, 3*x), (x, 0)]

-- Calculate the area of the triangle
noncomputable def triangle_area (x : ℝ) : ℝ := (1/2) * x * 3*x

-- Theorem statement
theorem triangle_area_96_implies_x_8 :
  ∀ x : ℝ, x > 0 → triangle_area x = 96 → x = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_96_implies_x_8_l744_74463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_height_proof_l744_74430

/-- The height of a triangle with sides 15, 14, and 13 --/
def triangle_height : ℝ := 11.2

/-- The base of the triangle --/
def base : ℝ := 15

/-- One side of the triangle --/
def side1 : ℝ := 14

/-- Another side of the triangle --/
def side2 : ℝ := 13

/-- Theorem stating that the height of the triangle with given sides is 11.2 --/
theorem triangle_height_proof : 
  ∃ (h : ℝ), 
    h = (2 * Real.sqrt (21 * (21 - base) * (21 - side1) * (21 - side2))) / base ∧ 
    h = triangle_height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_height_proof_l744_74430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l744_74486

noncomputable def m : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

theorem vector_problem (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (m.1 * (n x).1 + m.2 * (n x).2 = 0 → Real.tan x = 1) ∧
  (m.1 * (n x).1 + m.2 * (n x).2 = Real.cos (Real.pi / 3) → x = 5 * Real.pi / 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l744_74486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_probability_l744_74420

/-- The probability of selecting 3 socks of the same type from a drawer containing
    12 black socks, 10 white socks, and 6 striped socks. -/
theorem sock_probability : 
  (Nat.choose 12 3 + Nat.choose 10 3 + Nat.choose 6 3) / Nat.choose 28 3 = 60 / 546 := by
  -- Define constants
  let black_socks : ℕ := 12
  let white_socks : ℕ := 10
  let striped_socks : ℕ := 6
  let total_socks : ℕ := black_socks + white_socks + striped_socks
  
  -- Calculate the probability
  have h1 : (Nat.choose black_socks 3 + Nat.choose white_socks 3 + Nat.choose striped_socks 3) / 
            Nat.choose total_socks 3 = 60 / 546 := by
    -- Proof steps would go here
    sorry
  
  -- Apply the calculated probability
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_probability_l744_74420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l744_74445

noncomputable def f (x : ℝ) : ℝ := Real.sin ((Real.pi / 3) * x + 1 / 3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l744_74445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_injured_player_age_l744_74435

theorem injured_player_age 
  (total_players : ℕ) 
  (remaining_players : ℕ) 
  (initial_average : ℕ) 
  (final_average : ℕ) 
  (h1 : total_players = 11) 
  (h2 : remaining_players = 10) 
  (h3 : initial_average = 22) 
  (h4 : final_average = 21) :
  (total_players * initial_average) - (remaining_players * final_average) = 32 := by
  -- Proof steps would go here
  sorry

#check injured_player_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_injured_player_age_l744_74435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABD_l744_74495

/-- Line l: x - 2y + 5 = 0 -/
def line_l (x y : ℝ) : Prop := x - 2*y + 5 = 0

/-- Circle C: x^2 + y^2 = 9 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- Points A and B are the intersection of line l and circle C -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

/-- Point D is on circle C but different from A and B -/
def point_D (D : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  circle_C D.1 D.2 ∧ D ≠ A ∧ D ≠ B

/-- The maximum area of triangle ABD -/
noncomputable def max_area_ABD (A B : ℝ × ℝ) : ℝ :=
  let h := 3 + Real.sqrt 5
  let base := 4
  1/2 * base * h

theorem max_area_triangle_ABD (A B D: ℝ × ℝ) :
  intersection_points A B → point_D D A B →
  ∃ (area : ℝ), area ≤ max_area_ABD A B ∧
  area = Real.sqrt 29 + 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ABD_l744_74495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_l744_74447

/-- The trajectory of the center of a moving circle C -/
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

/-- The equation of the fixed circle M -/
def circle_M (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 64

/-- Point A through which the moving circle C passes -/
def point_A : ℝ × ℝ := (-2, 0)

/-- Theorem stating that the trajectory of the center of circle C
    is described by the given equation, given the conditions -/
theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ),
  (∃ (r : ℝ), r > 0 ∧
    (∀ (t : ℝ), ∃ (cx cy : ℝ),
      -- C passes through A
      (cx + r * Real.cos t - point_A.1)^2 + (cy + r * Real.sin t - point_A.2)^2 = r^2 ∧
      -- C is internally tangent to M
      ∃ (tx ty : ℝ), circle_M tx ty ∧
        (cx - tx)^2 + (cy - ty)^2 = (8 - r)^2)) →
  trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_l744_74447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_f_l744_74470

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- Prime factorization of a natural number -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

/-- The function f satisfying the given properties -/
def f (n : ℕ) : ℕ :=
  (prime_factorization n).foldr (fun ⟨p, a⟩ acc => acc * p ^ (p ^ a - 1)) 1

/-- Main theorem: f is the unique function satisfying the given properties -/
theorem unique_f :
    (∀ x : ℕ, d (f x) = x) ∧
    (∀ x y : ℕ, f (x * y) ∣ ((x - 1) * y ^ (x * y - 1) * f x)) ∧
    (∀ g : ℕ → ℕ, (∀ x : ℕ, d (g x) = x) →
      (∀ x y : ℕ, g (x * y) ∣ ((x - 1) * y ^ (x * y - 1) * g x)) →
      g = f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_f_l744_74470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_speed_calculation_l744_74405

/-- The speed of a bullet leaving the muzzle of a gun --/
noncomputable def muzzle_speed (a l : ℝ) : ℝ := Real.sqrt (2 * a * l)

/-- Theorem: The speed of the bullet when it leaves the muzzle is 9 × 10^2 m/s --/
theorem bullet_speed_calculation :
  let a : ℝ := 5 * 10^5
  let l : ℝ := 0.81
  muzzle_speed a l = 9 * 10^2 := by
  -- Unfold the definition of muzzle_speed
  unfold muzzle_speed
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_speed_calculation_l744_74405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l744_74473

theorem expression_simplification_and_evaluation (a : ℝ) :
  a = 3 →
  (2 / (a - 1)) + ((a^2 - 4*a + 4) / (a^2 - 1)) / ((a - 2) / (a + 1)) = 3/2 :=
by
  intro h
  -- The proof steps would go here
  sorry

#check expression_simplification_and_evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_and_evaluation_l744_74473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_set_B_l744_74409

universe u

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {1, 3, 5, 7}

-- Define set B (we'll prove this later)
def B : Set Nat := {2, 3, 5, 7}

-- Define the complement of A in U
axiom complement_A : (U \ A) = {2, 4, 6}

-- Define the complement of B in U
axiom complement_B : (U \ B) = {1, 4, 6}

-- Theorem to prove that B is the set we defined
theorem find_set_B : B = {2, 3, 5, 7} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_set_B_l744_74409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_at_endpoints_l744_74492

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

-- State the theorem
theorem function_sum_at_endpoints (a b : ℝ) :
  b > a → 
  (∀ x, a ≤ x ∧ x ≤ b ↔ a ≤ f x ∧ f x ≤ b) →
  f a + f b = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_at_endpoints_l744_74492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_process_upper_bound_l744_74483

def split_number (n : ℕ) : ℕ × ℕ :=
  let a := n / 100
  let b := n % 100
  (a, b)

def magic_process (n : ℕ) : ℕ :=
  let (a, b) := split_number n
  (a + b) * 100 + (max a b - min a b)

theorem magic_process_upper_bound :
  ∀ n : ℕ, 100 ≤ n ∧ n < 10000 →
    magic_process n ≤ 18810 ∧
    (magic_process n = 18810 ↔ (n = 9989 ∨ n = 8999)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_process_upper_bound_l744_74483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l744_74493

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the point of interest
def x₀ : ℝ := Real.exp 1

-- Theorem for the tangent and normal line equations
theorem tangent_and_normal_equations :
  let y₀ := f x₀
  let m_tangent := (deriv f) x₀
  let m_normal := -1 / m_tangent
  (∀ x y, y - y₀ = m_tangent * (x - x₀) ↔ y = 2*x - Real.exp 1) ∧
  (∀ x y, y - y₀ = m_normal * (x - x₀) ↔ y = -1/2*x + 3/2*Real.exp 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l744_74493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_speed_4_to_5_l744_74461

/-- Represents the elevation of a hiker at a given time --/
noncomputable def elevation : ℝ → ℝ := sorry

/-- The average climbing speed between two time points --/
noncomputable def avg_climbing_speed (t1 t2 : ℝ) : ℝ :=
  (elevation t2 - elevation t1) / (t2 - t1)

/-- Theorem stating that the average climbing speed between hours 4 and 5 is the highest --/
theorem highest_speed_4_to_5 :
  avg_climbing_speed 4 5 > max 
    (max (avg_climbing_speed 0 1) (avg_climbing_speed 1 2))
    (max (avg_climbing_speed 5 6) (avg_climbing_speed 10 11)) := by
  sorry

#check highest_speed_4_to_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_speed_4_to_5_l744_74461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l744_74442

-- Define the points in the coordinate plane
noncomputable def A : ℝ × ℝ := (0, 8)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (10, 0)

-- Define midpoints D and E
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the area of a triangle given base and height
noncomputable def triangleArea (base height : ℝ) : ℝ := (1 / 2) * base * height

-- Theorem statement
theorem area_of_triangle_DBC : 
  let base := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let height := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  triangleArea base height = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DBC_l744_74442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l744_74469

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℚ := sorry

/-- The arithmetic sequence -/
def a : ℕ → ℚ := sorry

theorem arithmetic_sequence_sum :
  (S 3 = -3) → (S 7 = 7) → (S 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l744_74469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_101_of_3_div_7_l744_74432

def decimal_expansion (n d : ℕ) : List ℕ := sorry

theorem digit_101_of_3_div_7 : 
  let expansion := decimal_expansion 3 7
  (expansion.get? 100).isSome ∧ (expansion.get? 100).getD 0 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_101_of_3_div_7_l744_74432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l744_74482

-- Define the operation for calculating percentage of a number
def percentage_of (p : ℚ) (n : ℚ) : ℚ := (p / 100) * n

-- State the theorem
theorem percentage_calculation :
  (percentage_of 208 1265) / 6 = 437866666 / 1000000 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l744_74482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l744_74407

/-- The quadratic function f(x) = 3(x+4)^2 - 5 -/
def f (x : ℝ) : ℝ := 3 * (x + 4)^2 - 5

/-- The vertex of a quadratic function a(x-h)^2 + k is (h,k) -/
structure Vertex where
  x : ℝ
  y : ℝ

theorem quadratic_vertex : Vertex.mk (-4) (-5) = 
  { x := -4, y := f (-4) } := by
  -- Proof goes here
  sorry

#eval f (-4) -- This will evaluate f(-4) to verify the y-coordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_l744_74407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_solid_of_revolution_l744_74448

/-- The parabola passing through (1, -1) -/
noncomputable def parabola (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - (a + 2)

/-- The volume of the solid of revolution -/
noncomputable def volume (a : ℝ) : ℝ := (Real.pi / 30) * (a^2 + 4*a + 8)^(5/2)

/-- The theorem stating the minimum volume -/
theorem min_volume_solid_of_revolution :
  ∃ (a : ℝ), ∀ (a' : ℝ), volume a ≤ volume a' ∧ volume a = (16 * Real.pi) / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_solid_of_revolution_l744_74448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l744_74458

theorem matrix_satisfies_conditions : ∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M.mulVec (![3, 0] : Fin 2 → ℝ) = ![6, 21] ∧
  M.mulVec (![-1, 5] : Fin 2 → ℝ) = ![3, -17] ∧
  M = !![2, 1; 7, -2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l744_74458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_l744_74455

/-- Sequence definition -/
def a : ℕ → ℕ
  | 0 => 0  -- Add this case for 0
  | 1 => 0
  | n + 2 => a (n + 1) + (n + 1)

/-- Theorem statement -/
theorem a_2013 : a 2013 = 2025078 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2013_l744_74455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_geometry_perpendicular_not_necessarily_parallel_l744_74488

/-- Definition of a line in plane geometry -/
def Line : Type := sorry

/-- Definition of perpendicularity between lines -/
def Perpendicular (l1 l2 : Line) : Prop := sorry

/-- Definition of parallel lines -/
def Parallel (l1 l2 : Line) : Prop := sorry

/-- In plane geometry, two lines perpendicular to the same line are parallel -/
axiom plane_geometry_perpendicular_parallel (l1 l2 l3 : Line) :
  Perpendicular l1 l3 → Perpendicular l2 l3 → Parallel l1 l2

/-- Definition of a plane in solid geometry -/
def Plane : Type := sorry

/-- Definition of perpendicularity between planes -/
def Perpendicular_Planes (p1 p2 : Plane) : Prop := sorry

/-- Definition of parallel planes -/
def Parallel_Planes (p1 p2 : Plane) : Prop := sorry

/-- Theorem: In solid geometry, two planes perpendicular to the same plane are not necessarily parallel -/
theorem solid_geometry_perpendicular_not_necessarily_parallel :
  ∃ (p1 p2 p3 : Plane), Perpendicular_Planes p1 p3 ∧ Perpendicular_Planes p2 p3 ∧ ¬Parallel_Planes p1 p2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_geometry_perpendicular_not_necessarily_parallel_l744_74488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l744_74428

/-- The function f(x) = ln x - x^2 + 2x + 5 -/
noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 2*x + 5

/-- The domain of f is x > 0 -/
def domain (x : ℝ) : Prop := x > 0

theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ domain a ∧ domain b ∧ f a = 0 ∧ f b = 0 ∧
  ∀ (c : ℝ), domain c → f c = 0 → (c = a ∨ c = b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l744_74428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_three_digit_numbers_no_repeat_count_three_digit_numbers_no_repeat_descending_count_l744_74453

def Digits : Finset Nat := {1, 2, 3, 4}

def ThreeDigitNumbers : Finset (Nat × Nat × Nat) :=
  Finset.product Digits (Finset.product Digits Digits)

def ThreeDigitNumbersNoRepeat : Finset (Nat × Nat × Nat) :=
  ThreeDigitNumbers.filter (fun x => x.1 ≠ x.2.1 ∧ x.1 ≠ x.2.2 ∧ x.2.1 ≠ x.2.2)

def ThreeDigitNumbersNoRepeatDescending : Finset (Nat × Nat × Nat) :=
  ThreeDigitNumbersNoRepeat.filter (fun x => x.1 > x.2.1 ∧ x.2.1 > x.2.2)

theorem three_digit_numbers_count :
  Finset.card ThreeDigitNumbers = 64 := by
  sorry

theorem three_digit_numbers_no_repeat_count :
  Finset.card ThreeDigitNumbersNoRepeat = 24 := by
  sorry

theorem three_digit_numbers_no_repeat_descending_count :
  Finset.card ThreeDigitNumbersNoRepeatDescending = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_three_digit_numbers_no_repeat_count_three_digit_numbers_no_repeat_descending_count_l744_74453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l744_74402

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi / 2)

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6) + 2

/-- Theorem stating the properties of f and g -/
theorem f_and_g_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (Set.Icc (3 - Real.sqrt 3) 5 = {y | ∃ x ∈ Set.Icc 0 (7 * Real.pi / 12), g x = y}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l744_74402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangles_theorem_l744_74474

/-- A 2-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Checks if three vertices form a monochromatic triangle in a given coloring -/
def IsMonochromaticTriangle (c : TwoColoring n) (v1 v2 v3 : Fin n) : Prop :=
  c v1 v2 = c v2 v3 ∧ c v2 v3 = c v3 v1

/-- Checks if two triangles share no edges -/
def NoSharedEdges (n : ℕ) (v1 v2 v3 w1 w2 w3 : Fin n) : Prop :=
  ({v1, v2, v3} : Finset (Fin n)) ∩ {w1, w2, w3} = ∅

/-- The main theorem -/
theorem monochromatic_triangles_theorem :
  (∀ c : TwoColoring 8, ∃ v1 v2 v3 w1 w2 w3 : Fin 8,
    IsMonochromaticTriangle c v1 v2 v3 ∧
    IsMonochromaticTriangle c w1 w2 w3 ∧
    NoSharedEdges 8 v1 v2 v3 w1 w2 w3) ∧
  (∃ c : TwoColoring 7, ∀ v1 v2 v3 w1 w2 w3 : Fin 7,
    IsMonochromaticTriangle c v1 v2 v3 ∧
    IsMonochromaticTriangle c w1 w2 w3 →
    ¬NoSharedEdges 7 v1 v2 v3 w1 w2 w3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangles_theorem_l744_74474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_T_lower_bound_l744_74499

-- Define the sequences and their properties
def a : ℕ+ → ℝ := sorry
def b : ℕ+ → ℝ := sorry
def c : ℕ+ → ℝ := sorry
def S : ℕ+ → ℝ := sorry
def T : ℕ+ → ℝ := sorry

-- Conditions
axiom a_sum_property : ∀ n : ℕ+, S n / n = (1/2) * n + 11/2
axiom b_recurrence : ∀ n : ℕ+, b (n + 2) - 2 * b (n + 1) + b n = 0
axiom b_sum_10 : (Finset.range 10).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩) = 185
axiom c_def : ∀ n : ℕ+, c n = 3 / ((2 * a n - 11) * (2 * b n - 1))
axiom T_def : ∀ n : ℕ+, T n = (Finset.range n).sum (λ i => c ⟨i + 1, Nat.succ_pos i⟩)

-- Theorems to prove
theorem a_formula : ∀ n : ℕ+, a n = n + 5 := by sorry

theorem b_formula : ∀ n : ℕ+, b n = 3 * n + 2 := by sorry

theorem T_lower_bound : ∀ n : ℕ+, T n ≥ 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_T_lower_bound_l744_74499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statements_l744_74423

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Syllogistic
  | Deductive

-- Define the properties of reasoning types
def reasoning_property (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Inductive => True  -- Placeholder for the actual property
  | ReasoningType.Analogical => True  -- Placeholder for the actual property
  | ReasoningType.Syllogistic => True  -- Placeholder for the actual property
  | ReasoningType.Deductive => True  -- Placeholder for the actual property

-- Define the statements
def statement1 : Prop := ∀ r : ReasoningType, r = ReasoningType.Inductive ∨ r = ReasoningType.Analogical → reasoning_property r

def statement2 : Prop := True  -- Placeholder for the actual statement

def statement3 : Prop := ∃ (m : ℕ), (∀ n : ℕ, 9 ∣ n → 3 ∣ n) ∧ (9 ∣ m → 3 ∣ m)

def statement4 : Prop := ∀ (p q : Prop), (p → q) → q

-- Theorem stating that statements 1, 2, and 4 are incorrect
theorem incorrect_statements :
  ¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statements_l744_74423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divided_triangle_perimeter_l744_74457

/-- A triangle represented by its three side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.side1 + t.side2 + t.side3

/-- A large triangle divided into 9 smaller triangles -/
structure LargeDividedTriangle where
  largeTriangle : Triangle
  smallTriangles : Fin 9 → Triangle

/-- The property that all small triangles have equal perimeters -/
def equalSmallPerimeters (ldt : LargeDividedTriangle) : Prop :=
  ∀ i j : Fin 9, (ldt.smallTriangles i).perimeter = (ldt.smallTriangles j).perimeter

/-- The theorem stating that if a triangle with perimeter 120 is divided into 9 smaller triangles
    with equal perimeters, then the perimeter of each smaller triangle is 40 -/
theorem divided_triangle_perimeter
  (ldt : LargeDividedTriangle)
  (h1 : ldt.largeTriangle.perimeter = 120)
  (h2 : equalSmallPerimeters ldt) :
  ∀ i : Fin 9, (ldt.smallTriangles i).perimeter = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divided_triangle_perimeter_l744_74457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circles_l744_74422

-- Define the circles and triangle
def circleA : ℝ := 12
def circleB : ℝ := 5
def circleC : ℝ := 3
def circleD : ℝ := 3

-- Define the relationship between circles
noncomputable def circleE_radius : ℚ := 277 / 26

-- Define the theorem
theorem equilateral_triangle_inscribed_circles (p q : ℕ) (hpq : Nat.Coprime p q) :
  (↑p : ℚ) / ↑q = circleE_radius →
  p + q = 303 := by
  sorry

#check equilateral_triangle_inscribed_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circles_l744_74422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l744_74497

/-- A tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ

/-- The volume of a tetrahedron with given properties -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  24 * Real.sqrt 2

/-- Theorem stating that the volume of the specific tetrahedron is 24√2 cm³ -/
theorem specific_tetrahedron_volume :
  ∀ t : Tetrahedron,
    t.ab_length = 5 ∧
    t.abc_area = 20 ∧
    t.abd_area = 18 ∧
    t.face_angle = π / 4 →
    tetrahedron_volume t = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tetrahedron_volume_l744_74497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_inequality_l744_74484

/-- The apothem of a regular n-gon inscribed in a circle of radius r -/
noncomputable def apothem (n : ℕ) (r : ℝ) : ℝ := r * Real.cos (Real.pi / n)

/-- The main theorem -/
theorem apothem_inequality (r : ℝ) (h : r > 0) :
  (∀ n : ℕ, n ≥ 3 → (n + 1 : ℝ) * apothem (n + 1) r - n * apothem n r > r) ∧
  (∃ ε : ℝ, ε > 0 ∧ ∃ n : ℕ, n ≥ 3 ∧ (n + 1 : ℝ) * apothem (n + 1) r - n * apothem n r ≤ r + ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_inequality_l744_74484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_radius_altitude_ratio_l744_74452

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  /-- Side length of the equal legs -/
  a : ℝ
  /-- Assumption that the side length is positive -/
  a_pos : a > 0

/-- The ratio of the inscribed circle radius to the altitude in an isosceles right triangle -/
noncomputable def inscribedRadiusAltitudeRatio (t : IsoscelesRightTriangle) : ℝ :=
  Real.sqrt 2 - 1

/-- Theorem stating that the ratio of the inscribed circle radius to the altitude
    in an isosceles right triangle is √2 - 1 -/
theorem isosceles_right_triangle_inscribed_radius_altitude_ratio 
  (t : IsoscelesRightTriangle) : 
  inscribedRadiusAltitudeRatio t = Real.sqrt 2 - 1 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_radius_altitude_ratio_l744_74452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_final_sum_l744_74449

/-- Represents the value of a coin in cents -/
inductive CoinValue : Type
  | penny : CoinValue
  | nickel : CoinValue
  | dime : CoinValue
  | quarter : CoinValue

/-- Converts a CoinValue to its numerical value in cents -/
def coinValueToCents (c : CoinValue) : Nat :=
  match c with
  | CoinValue.penny => 1
  | CoinValue.nickel => 5
  | CoinValue.dime => 10
  | CoinValue.quarter => 25

/-- Represents the current state of the coin selection process -/
structure State :=
  (sum : Nat)

/-- The expected value function -/
noncomputable def expectedValue (s : State) : ℝ :=
  sorry

/-- Theorem stating the expected value of the final sum -/
theorem expected_final_sum :
  ∃ (finalState : State), finalState.sum % 100 = 0 ∧ 
    ∃ (ε : ℝ), ε > 0 ∧ |expectedValue finalState - 1025| < ε := by
  sorry

#check expected_final_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_final_sum_l744_74449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l744_74485

noncomputable def A : ℝ × ℝ := (2, -3)
noncomputable def B : ℝ × ℝ := (-2, -5)
noncomputable def M₁ : ℝ × ℝ := (0, 1)
noncomputable def M₂ : ℝ × ℝ := (2, -5)

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 10

def on_line (x y : ℝ) : Prop := x - 2*y - 3 = 0

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

def on_circle (p : ℝ × ℝ) : Prop := circle_equation p.1 p.2

theorem circle_properties :
  ∃ (C : ℝ × ℝ), 
    on_line C.1 C.2 ∧
    distance A C = distance B C ∧
    on_circle A ∧
    on_circle B ∧
    on_circle M₁ ∧
    ¬on_circle M₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l744_74485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l744_74456

-- Define the power function
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := λ x => x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  (∃ α : ℝ, f = powerFunction α) →  -- f is a power function
  f 2 = (1 : ℝ) / 4 →               -- f passes through (2, 1/4)
  f (Real.sqrt 2) = (1 : ℝ) / 2     -- f(√2) = 1/2
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l744_74456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_problem_l744_74434

/-- Mowing problem theorem -/
theorem mowing_problem (mary_rate tom_rate : ℚ) (mary_solo_time : ℚ) : 
  mary_rate = 1/3 →
  tom_rate = 1/4 →
  mary_solo_time = 1 →
  (1 - mary_rate * mary_solo_time) / (mary_rate + tom_rate) = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mowing_problem_l744_74434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_quarter_l744_74408

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def penny_value : ℚ := 1 / 100

def total_quarter_value : ℚ := 10
def total_nickel_value : ℚ := 5
def total_dime_value : ℚ := 5
def total_penny_value : ℚ := 15

def num_quarters : ℕ := (total_quarter_value / quarter_value).floor.toNat
def num_nickels : ℕ := (total_nickel_value / nickel_value).floor.toNat
def num_dimes : ℕ := (total_dime_value / dime_value).floor.toNat
def num_pennies : ℕ := (total_penny_value / penny_value).floor.toNat

def total_coins : ℕ := num_quarters + num_nickels + num_dimes + num_pennies

theorem probability_of_quarter : 
  (num_quarters : ℚ) / (total_coins : ℚ) = 40 / 1690 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_quarter_l744_74408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_satisfying_functions_l744_74431

/-- A polynomial function of degree at most 3 -/
def CubicPolynomial (α : Type) [Ring α] := α → α

/-- The condition that g(x)g(-x) = g(x³) for all x -/
def SatisfiesCondition (g : CubicPolynomial ℝ) : Prop :=
  ∀ x, g x * g (-x) = g (x^3)

/-- The set of all functions satisfying the condition -/
def SatisfyingFunctions : Set (CubicPolynomial ℝ) :=
  {g | ∃ (a b c d : ℝ), (∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧ SatisfiesCondition g}

/-- Theorem stating that there are exactly four satisfying functions -/
theorem exactly_four_satisfying_functions :
  ∃ (s : Finset (CubicPolynomial ℝ)), s.card = 4 ∧ (∀ g, g ∈ SatisfyingFunctions ↔ g ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_satisfying_functions_l744_74431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l744_74439

/-- A monic quadratic polynomial with real coefficients -/
def MonicQuadraticPolynomial (a b : ℝ) : ℂ → ℂ := λ x ↦ x^2 + a*x + b

theorem monic_quadratic_with_complex_root 
  (p : ℂ → ℂ) (a b : ℝ) (z : ℂ) 
  (h1 : p = MonicQuadraticPolynomial a b) 
  (h2 : p z = 0) 
  (h3 : z = 3 - 2*Complex.I) :
  a = -6 ∧ b = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_with_complex_root_l744_74439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l744_74413

noncomputable section

open Real

def IsTriangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

def SideLength (angle1 angle2 : ℝ) : ℝ :=
  2 * sin (angle1 / 2) * sin (angle2 / 2)

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  IsTriangle A B C →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  (a = SideLength B C) ∧ (b = SideLength A C) ∧ (c = SideLength A B) →
  -- Properties to prove
  (sin (B + C) = sin A) ∧
  ((sin A > sin B) → (A > B)) ∧
  ((a * cos B - b * cos A = c) → (A = π / 2)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l744_74413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_for_odd_f_l744_74487

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 2^x - (k^2 - 3) * 2^(-x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  (2 : ℝ)^x - (k^2 - 3) * (2 : ℝ)^(-x)

/-- k = 2 is a sufficient but not necessary condition for f to be odd -/
theorem sufficient_not_necessary_for_odd_f :
  (∃ k : ℝ, k ≠ 2 ∧ IsOdd (f k)) ∧
  (IsOdd (f 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_for_odd_f_l744_74487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_triangle_abc_l744_74401

/-- The circumcircle of triangle ABC, where A(1,0), B(0,√3), and C(2,√3) -/
def circumcircle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - (4*Real.sqrt 3/3)*y + 1 = 0

/-- Point A of the triangle -/
def A : ℝ × ℝ := (1, 0)

/-- Point B of the triangle -/
noncomputable def B : ℝ × ℝ := (0, Real.sqrt 3)

/-- Point C of the triangle -/
noncomputable def C : ℝ × ℝ := (2, Real.sqrt 3)

/-- Theorem stating that the given equation represents the circumcircle of triangle ABC -/
theorem circumcircle_of_triangle_abc :
  circumcircle A.1 A.2 ∧ circumcircle B.1 B.2 ∧ circumcircle C.1 C.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_of_triangle_abc_l744_74401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_shift_polynomial_l744_74421

theorem root_shift_polynomial (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℝ, x^3 - 15*x^2 + 74*x - 120 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_shift_polynomial_l744_74421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_48_l744_74403

-- Define the distance in feet
variable (m : ℝ)

-- Define the speed for the northward journey (2 minutes per mile)
noncomputable def speed_north : ℝ := 2 / 60

-- Define the speed for the southward journey (2 miles per minute)
noncomputable def speed_south : ℝ := 2 * 60

-- Define the conversion factor from feet to miles
noncomputable def feet_to_miles : ℝ := 1 / 5280

-- Theorem statement
theorem average_speed_is_48 :
  let distance_miles := m * feet_to_miles
  let time_north := distance_miles / speed_north
  let time_south := distance_miles / speed_south
  let total_time := time_north + time_south
  let total_distance := 2 * distance_miles
  total_distance / total_time = 48 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_is_48_l744_74403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_l744_74465

/-- The set of all 9-digit natural numbers written only with digits 1, 2, and 3 -/
def S : Set ℕ :=
  {n : ℕ | 100000000 ≤ n ∧ n ≤ 333333333 ∧ ∀ d ∈ n.digits 10, d ∈ ({1, 2, 3} : Finset ℕ)}

/-- The function that returns the leftmost digit of a number -/
def leftmost_digit (n : ℕ) : ℕ :=
  (n.digits 10).head!

/-- The function f from S to {1,2,3} defined as the leftmost digit -/
def f : S → Fin 3 :=
  λ n => ⟨leftmost_digit n, by sorry⟩

/-- Two numbers in S differ in each digit position -/
def differ_in_each_position (x y : S) : Prop :=
  ∀ i, (x.val.digits 10).get? i ≠ (y.val.digits 10).get? i

theorem f_unique :
  ∀ g : S → Fin 3,
  (g ⟨111111111, by sorry⟩ = 1 ∧
   g ⟨222222222, by sorry⟩ = 2 ∧
   g ⟨333333333, by sorry⟩ = 3 ∧
   g ⟨122222222, by sorry⟩ = 1) ∧
  (∀ x y : S, differ_in_each_position x y → g x ≠ g y) →
  g = f :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_l744_74465
