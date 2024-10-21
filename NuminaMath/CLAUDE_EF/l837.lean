import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l837_83708

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * ((1 + r / 100) ^ n - 1)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * (t : ℝ) / 100

theorem interest_calculation (P : ℝ) (h : compound_interest P 10 2 = 693) :
  simple_interest P 10 2 = 660 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l837_83708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_proportion_calculate_fare_prop_taxi_fare_calculation_l837_83750

/-- Represents the fare for a taxi ride -/
structure TaxiFare where
  distance : ℝ
  cost : ℝ

/-- Proves that if taxi fare is directly proportional to distance, 
    and a 50-mile ride costs $120, then a 70-mile ride costs $168 -/
theorem taxi_fare_proportion 
  (fare : ℝ → ℝ) 
  (h_prop : ∀ (d₁ d₂ : ℝ), d₁ > 0 → d₂ > 0 → fare d₁ / d₁ = fare d₂ / d₂) 
  (h_initial : fare 50 = 120) : 
  fare 70 = 168 := by
  sorry

/-- Calculates the fare for a given distance based on a known fare -/
noncomputable def calculate_fare (known : TaxiFare) (new_distance : ℝ) : ℝ :=
  (known.cost / known.distance) * new_distance

/-- Proves that the calculate_fare function satisfies the proportion property -/
theorem calculate_fare_prop (known : TaxiFare) (d : ℝ) 
  (h_pos : known.distance > 0) : 
  calculate_fare known d / d = known.cost / known.distance := by
  sorry

/-- Proves that using calculate_fare with the given information yields the correct result -/
theorem taxi_fare_calculation (known : TaxiFare) 
  (h_known : known.distance = 50 ∧ known.cost = 120) :
  calculate_fare known 70 = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_proportion_calculate_fare_prop_taxi_fare_calculation_l837_83750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rank_of_skew_symmetric_matrix_l837_83740

/-- Define the skew-symmetric matrix A as described in the problem -/
def A (n : ℕ) : Matrix (Fin (2*n+1)) (Fin (2*n+1)) ℤ :=
  Matrix.of (λ i j =>
    if i = j then 0
    else if (i.val - j.val + (2*n+1)) % (2*n+1) ≤ n then 1
    else if (j.val - i.val + (2*n+1)) % (2*n+1) ≤ n then -1
    else if (i.val - j.val + (2*n+1)) % (2*n+1) > n ∧ (i.val - j.val + (2*n+1)) % (2*n+1) ≤ 2*n then -1
    else 1)

/-- The theorem stating that the rank of matrix A is 2n -/
theorem rank_of_skew_symmetric_matrix (n : ℕ) :
  Matrix.rank (A n) = 2 * n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rank_of_skew_symmetric_matrix_l837_83740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_unique_digits_l837_83702

/-- Represents a natural number with no zero digits -/
def NonZeroDigitNumber : Type := { n : ℕ | ∀ d, d ∈ Nat.digits 10 n → d ≠ 0 }

/-- Represents the operations allowed on the number -/
inductive Operation
| DeleteAdjacentIdentical : Operation
| InsertIdentical : Operation

/-- Represents the result of applying operations to a number -/
noncomputable def ApplyOperations (n : NonZeroDigitNumber) (ops : List Operation) : ℕ := sorry

/-- Predicate to check if all digits in a number are unique -/
def AllUnique (n : ℕ) : Prop := 
  let digits := Nat.digits 10 n
  List.Nodup digits

/-- Main theorem statement -/
theorem transform_to_unique_digits (N : NonZeroDigitNumber) :
  ∃ (ops : List Operation), 
    let result := ApplyOperations N ops
    AllUnique result ∧ result < 10^9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_to_unique_digits_l837_83702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l837_83768

/-- Given that -1, a, b, c, -9 form a geometric sequence, prove that b = -3 -/
theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h : ∃ r : ℝ, r ≠ 0 ∧ a = -1 * r ∧ b = a * r ∧ c = b * r ∧ -9 = c * r) : b = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l837_83768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l837_83752

/-- The equation of a conic section -/
noncomputable def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 2)^2) + Real.sqrt ((x + 3)^2 + (y - 5)^2) = 12

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  (∀ x y, conic_equation x y ↔ 
    distance x y 2 (-2) + distance x y (-3) 5 = 12) ∧
  distance 2 (-2) (-3) 5 < 12 →
  ∃ a b c d e f : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ x y, conic_equation x y ↔ (x - c)^2 / a^2 + (y - d)^2 / b^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l837_83752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_cross_section_area_l837_83778

/-- Represents a rectangular parallelepiped -/
structure RectangularParallelepiped where
  ab : ℝ
  ad : ℝ
  aa1 : ℝ

/-- Calculates the area of the cross-section with minimal perimeter -/
noncomputable def minPerimeterCrossSectionArea (p : RectangularParallelepiped) : ℝ :=
  p.ab * p.ad * Real.sqrt (1 + 2 * ((p.aa1 / (p.ab + p.ad)) ^ 2))

theorem minimal_perimeter_cross_section_area 
    (p : RectangularParallelepiped) 
    (h1 : p.ab = 4) 
    (h2 : p.ad = 2) 
    (h3 : p.aa1 = 3 * Real.sqrt 2) : 
  minPerimeterCrossSectionArea p = 8 * Real.sqrt 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_cross_section_area_l837_83778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_difference_l837_83763

theorem marble_probability_difference 
  (red_marbles : ℝ) (black_marbles : ℝ) :
  red_marbles = 1002 →
  black_marbles = 999 →
  let total_marbles := red_marbles + black_marbles
  let same_color_prob := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1)) / (total_marbles * (total_marbles - 1))
  let diff_color_prob := (red_marbles * black_marbles * 2) / (total_marbles * (total_marbles - 1))
  |same_color_prob - diff_color_prob| = 83 / 166750 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_difference_l837_83763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_m_eq_neg_one_l837_83732

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^(m^2 - 4*m + 1)

-- Define the property of being monotonically increasing on (0, +∞)
def is_monotone_increasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → g x < g y

-- State the theorem
theorem f_monotone_iff_m_eq_neg_one :
  ∀ m : ℝ, is_monotone_increasing (f m) ↔ m = -1 := by
  sorry

#check f_monotone_iff_m_eq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_m_eq_neg_one_l837_83732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_periodic_l837_83746

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n → n < 90 → Real.tan (n * π / 180) = Real.tan (312 * π / 180) → n = -48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_equality_periodic_l837_83746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_zero_value_l837_83704

/-- A polynomial of degree 7 satisfying specific conditions -/
noncomputable def p : Polynomial ℝ :=
  sorry

/-- The condition that p(2^n) = 1/(2^(n+1)) for n = 0, 1, 2, ..., 7 -/
axiom p_condition : ∀ n : ℕ, n ≤ 7 → p.eval (2^n) = 1 / (2^(n+1))

/-- The degree of p is 7 -/
axiom p_degree : p.natDegree = 7

/-- Theorem: The value of p(0) is 255/(2^28) -/
theorem p_zero_value : p.eval 0 = 255 / (2^28) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_zero_value_l837_83704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_sqrt_calculation_l837_83730

-- Define the "★" operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a ≤ b then b else Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem star_sqrt_calculation :
  star (Real.sqrt 7) (star (Real.sqrt 2) (Real.sqrt 3)) = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_sqrt_calculation_l837_83730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l837_83701

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a / Real.sqrt b + b / Real.sqrt a) > (Real.sqrt a + Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l837_83701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_theta_l837_83792

theorem cos_pi_third_plus_theta (θ : ℝ) :
  Real.cos (π / 6 - θ) = 2 * Real.sqrt 2 / 3 →
  Real.cos (π / 3 + θ) = 1 / 3 ∨ Real.cos (π / 3 + θ) = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_third_plus_theta_l837_83792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_is_6_5_l837_83735

-- Define the height function
noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

-- Define the derivative of h
noncomputable def h' : ℝ → ℝ := deriv h

-- Theorem statement
theorem initial_velocity_is_6_5 : 
  h' 0 = 6.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_velocity_is_6_5_l837_83735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l837_83764

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Point P on the ellipse -/
def point_P : ℝ × ℝ := (2 * Real.sqrt 6 / 3, -1)

/-- Sum of distances from P to foci -/
def foci_distance_sum : ℝ := 4

/-- Point R -/
def point_R : ℝ × ℝ := (4, 0)

/-- Theorem statement -/
theorem ellipse_properties
  (a b : ℝ)
  (h_ellipse : ellipse_C (point_P.1) (point_P.2) a b)
  (h_foci : foci_distance_sum = 4) :
  (∃ (l : ℝ),
    ∀ (M N G : ℝ × ℝ),
    ellipse_C M.1 M.2 a b →
    ellipse_C N.1 N.2 a b →
    ellipse_C G.1 G.2 a b →
    (∃ (k : ℝ), M.2 - point_R.2 = k * (M.1 - point_R.1) ∧
                N.2 - point_R.2 = k * (N.1 - point_R.1)) →
    G.1 = M.1 ∧ G.2 = -M.2 →
    ∃ (F2 : ℝ × ℝ), F2 = (1, 0) ∧
      (G.1 - F2.1, G.2 - F2.2) = l • (N.1 - F2.1, N.2 - F2.2)) ∧
  a^2 = 4 ∧ b^2 = 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l837_83764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_m_cubed_split_l837_83794

theorem smallest_number_in_m_cubed_split (m : ℕ) (h1 : m > 1) : 
  (∃ (k : ℕ), k = m^3 ∧ k = (m^2 - m + 1) + m * (m - 1)) →
  (m^2 - m + 1 = 211) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_in_m_cubed_split_l837_83794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_linear_function_coefficient_sum_l837_83726

/-- Given a linear function g and its inverse, prove that the sum of coefficients is -2 -/
theorem inverse_linear_function_coefficient_sum (c d : ℝ) 
  (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
  (h1 : ∀ x, g x = c * x + d)
  (h2 : ∀ x, g_inv x = d * x + c)
  (h3 : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x) : c + d = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_linear_function_coefficient_sum_l837_83726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l837_83754

-- Define the function f(x) = x - sin(x)
noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, (f (x + 1) + f (1 - 4 * x) > 0) ↔ (x < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l837_83754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l837_83711

/-- A power function that passes through (-2, -1/8) and equals 27 at x = 1/3 -/
noncomputable def PowerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_theorem :
  ∃ α : ℝ, PowerFunction α (-2) = -1/8 ∧ PowerFunction α (1/3) = 27 := by
  -- We know α = -3 from the problem solution
  use -3
  constructor
  · -- Prove PowerFunction (-3) (-2) = -1/8
    simp [PowerFunction]
    norm_num
  · -- Prove PowerFunction (-3) (1/3) = 27
    simp [PowerFunction]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l837_83711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_b_value_l837_83770

/-- Definition of a line with equation x + y = b -/
def line_x_plus_y_eq_b (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = b}

/-- Definition of midpoint -/
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

/-- Definition of perpendicular bisector -/
def is_perpendicular_bisector (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  ∃ M : ℝ × ℝ, is_midpoint M A B ∧ M ∈ l

/-- The perpendicular bisector of a line segment passes through its midpoint -/
axiom perpendicular_bisector_passes_through_midpoint 
  {A B M : ℝ × ℝ} {l : Set (ℝ × ℝ)} :
  is_perpendicular_bisector l A B → is_midpoint M A B → M ∈ l

/-- Main theorem -/
theorem perpendicular_bisector_b_value :
  ∀ b : ℝ, 
  is_perpendicular_bisector (line_x_plus_y_eq_b b) (-2, 1) (4, 5) →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_b_value_l837_83770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l837_83717

/-- The speed of the goods train given the following conditions:
  1. A girl is sitting in a train traveling at 100 km/h.
  2. The goods train is traveling in the opposite direction.
  3. The goods train takes 6 seconds to pass the girl.
  4. The goods train is 560 m long.
-/
theorem goods_train_speed
  (girl_train_speed : ℝ)
  (passing_time : ℝ)
  (goods_train_length : ℝ)
  (goods_train_speed : ℝ) :
  girl_train_speed = 100 →
  passing_time = 6 / 3600 →
  goods_train_length = 0.56 →
  goods_train_speed = (goods_train_length / passing_time) - girl_train_speed →
  goods_train_speed = 236 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l837_83717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83709

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 3*x + 2*a
  else x - a * Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ (1 ≤ a ∧ a ≤ Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_change_l837_83751

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
def QuadraticPolynomial (a b c : ℝ) := fun (x : ℝ) ↦ a * x^2 + b * x + c

/-- The minimum value of a quadratic polynomial -/
noncomputable def MinValue (a b c : ℝ) : ℝ := -b^2 / (4 * a) + c

theorem quadratic_minimum_change (a b c : ℝ) (ha : a > 0) :
  MinValue (a + 3) b c - MinValue a b c = 9 →
  MinValue (a - 1) b c - MinValue a b c = -9 →
  MinValue (a + 1) b c - MinValue a b c = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_change_l837_83751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_receptivity_compare_receptivity_duration_high_receptivity_l837_83739

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if x > 10 ∧ x ≤ 16 then 59
  else if x > 16 ∧ x ≤ 30 then -3 * x + 107
  else 0  -- Default value for x outside the specified ranges

-- Theorem 1: Maximum value and duration
theorem max_receptivity :
  (∀ x : ℝ, 0 < x → x ≤ 30 → f x ≤ 59) ∧
  (∀ x : ℝ, 10 < x → x ≤ 16 → f x = 59) := by
  sorry

-- Theorem 2: Comparison at 5 and 20 minutes
theorem compare_receptivity :
  f 5 > f 20 := by
  sorry

-- Theorem 3: Duration of high receptivity
theorem duration_high_receptivity :
  ∃ t1 t2 : ℝ, 0 < t1 ∧ t1 < t2 ∧ t2 ≤ 30 ∧
  (∀ x : ℝ, t1 ≤ x → x ≤ t2 → f x ≥ 55) ∧
  t2 - t1 < 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_receptivity_compare_receptivity_duration_high_receptivity_l837_83739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83779

/-- The function f as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - 3

/-- The theorem statement -/
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → x₂ * f a x₁ - x₁ * f a x₂ < a * (x₁ - x₂)) →
  a ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l837_83775

/-- The length of the chord intercepted by the circle x^2 + y^2 = 1 on the line x + y - 1 = 0 is √2 -/
theorem chord_length_circle_line : 
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - 1 = 0}
  let chord := circle ∩ line
  ∃ (a b c d : ℝ), ((a, b) ∈ chord ∧ (c, d) ∈ chord ∧ (a, b) ≠ (c, d) ∧
    Real.sqrt ((a - c)^2 + (b - d)^2) = Real.sqrt 2)
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_circle_line_l837_83775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l837_83718

-- Define the line equation
noncomputable def line_equation (x y m : ℝ) : Prop := 2 * x - m * y + 1 = 0

-- Define the y-intercept
noncomputable def y_intercept (m : ℝ) : ℝ := 1 / m

-- Theorem statement
theorem line_slope (m : ℝ) :
  y_intercept m = 1/4 → (∃ k : ℝ, ∀ x y : ℝ, line_equation x y m → y = (1/2) * x + k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l837_83718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deductive_reasoning_proof_l837_83777

-- Define the domain D
variable (D : Set ℝ)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative of f and g
variable (f' g' : ℝ → ℝ)

-- Define the property of monotonically increasing function
def MonotonicallyIncreasing (h : ℝ → ℝ) (S : Set ℝ) :=
  ∀ x y, x ∈ S → y ∈ S → x < y → h x < h y

-- State the theorem
theorem deductive_reasoning_proof :
  (∀ x, x ∈ D → f' x > 0 → MonotonicallyIncreasing f D) →
  (∀ x, g' x = 2 * x) →
  (∀ x, x > 0 → g' x > 0) →
  MonotonicallyIncreasing g (Set.Ici 0) →
  ∃ premise conclusion, premise → conclusion :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deductive_reasoning_proof_l837_83777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_depth_is_90_l837_83780

/-- Calculates the desired depth given the initial and new work conditions -/
noncomputable def desired_depth (initial_men : ℕ) (initial_hours : ℕ) (initial_depth : ℝ)
                  (extra_men : ℕ) (new_hours : ℕ) : ℝ :=
  let total_men := initial_men + extra_men
  let additional_depth := initial_depth * (initial_men * initial_hours : ℝ) / ((total_men * new_hours) : ℝ)
  initial_depth + additional_depth

/-- The desired depth is 90 meters given the specified work conditions -/
theorem desired_depth_is_90 :
  desired_depth 45 8 40 30 6 = 90 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_depth_is_90_l837_83780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_negative_one_l837_83789

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

-- State the theorem
theorem tangent_slope_at_negative_one :
  (deriv f) (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_negative_one_l837_83789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_gt_2k_plus_2020_l837_83721

/-- g(k) is the maximum possible number of points in the plane such that 
    pairwise distances between these points have only k different values. -/
def g (k : ℕ+) : ℕ := sorry

/-- There exists a positive integer k such that g(k) > 2k + 2020 -/
theorem exists_k_gt_2k_plus_2020 : ∃ k : ℕ+, g k > 2 * k + 2020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_gt_2k_plus_2020_l837_83721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_invariance_day_1999_product_l837_83772

/-- The product of two numbers remains invariant when replaced by their arithmetic and harmonic means -/
theorem product_invariance (n : ℕ) : 
  let f : ℝ × ℝ → ℝ × ℝ := fun (p : ℝ × ℝ) => ((p.1 + p.2) / 2, (2 * p.1 * p.2) / (p.1 + p.2))
  let iterate : ℝ × ℝ → ℕ → ℝ × ℝ := fun p m => Nat.recOn m p (fun _ q => f q)
  let final := iterate (1, 2) n
  final.1 * final.2 = 2 := by
  sorry

/-- The product of the numbers on the board after 1999 days is 2 -/
theorem day_1999_product : 
  let f : ℝ × ℝ → ℝ × ℝ := fun (p : ℝ × ℝ) => ((p.1 + p.2) / 2, (2 * p.1 * p.2) / (p.1 + p.2))
  let iterate : ℝ × ℝ → ℕ → ℝ × ℝ := fun p m => Nat.recOn m p (fun _ q => f q)
  let final := iterate (1, 2) 1999
  final.1 * final.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_invariance_day_1999_product_l837_83772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l837_83724

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

-- Theorem for the properties of f
theorem f_properties :
  -- 1. Smallest positive period
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- 2. Monotonically increasing interval
  (∀ (k : ℤ), ∀ (x y : ℝ), -π/3 + k * π ≤ x ∧ x < y ∧ y ≤ π/6 + k * π → f x < f y) ∧
  -- 3. Axis of symmetry
  (∀ (k : ℤ), ∀ (x : ℝ), f (π/6 + k * π/2 + x) = f (π/6 + k * π/2 - x)) ∧
  -- 4. Center of symmetry
  (∀ (k : ℤ), ∀ (x : ℝ), f (-π/12 + k * π/2 + x) = -f (-π/12 + k * π/2 - x) + f (-π/12 + k * π/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l837_83724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_percent_profit_is_35_l837_83762

-- Define the cost price, labeled price, and selling price
noncomputable def cost_price : ℝ := 100
noncomputable def labeled_price : ℝ := cost_price * 1.5
noncomputable def selling_price : ℝ := labeled_price * 0.9

-- Define the actual profit
noncomputable def actual_profit : ℝ := selling_price - cost_price

-- Define the percent profit
noncomputable def percent_profit : ℝ := (actual_profit / cost_price) * 100

-- Theorem statement
theorem actual_percent_profit_is_35 : 
  percent_profit = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_percent_profit_is_35_l837_83762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_arithmetic_sequence_l837_83757

/-- Given an arithmetic sequence with first term 1/2, second term 5/6, and third term 7/6,
    prove that its tenth term is 7/2. -/
theorem tenth_term_of_arithmetic_sequence :
  ∀ a : ℕ → ℚ,
  a 1 = 1/2 →
  a 2 = 5/6 →
  a 3 = 7/6 →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  a 10 = 7/2 :=
by
  intro a h1 h2 h3 h_arith
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_arithmetic_sequence_l837_83757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l837_83765

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (1/2 * x + Real.pi/6) + 2

theorem f_properties :
  -- Smallest positive period is 4π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Monotonically decreasing on [2π/3 + 4kπ, 8π/3 + 4kπ] for all k ∈ ℤ
  (∀ (k : ℤ) (x y : ℝ),
    2*Real.pi/3 + 4*k*Real.pi ≤ x ∧ x < y ∧ y ≤ 8*Real.pi/3 + 4*k*Real.pi → f y < f x) ∧
  -- Maximum value is 4 and occurs when x = 4kπ + 2π/3 for all k ∈ ℤ
  (∀ (x : ℝ), f x ≤ 4) ∧
  (∀ (k : ℤ), f (4*k*Real.pi + 2*Real.pi/3) = 4) ∧
  -- Minimum value is 0 and occurs when x = 4kπ + 8π/3 for all k ∈ ℤ
  (∀ (x : ℝ), f x ≥ 0) ∧
  (∀ (k : ℤ), f (4*k*Real.pi + 8*Real.pi/3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l837_83765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt_34_l837_83705

/-- The speed of a particle moving in a 2D plane -/
noncomputable def particleSpeed (pos : ℝ → ℝ × ℝ) : ℝ :=
  let vel := fun t => (pos (t + 1) - pos t)
  Real.sqrt ((vel 0).1^ 2 + (vel 0).2^ 2)

/-- The position of the particle at time t -/
def particlePosition (t : ℝ) : ℝ × ℝ :=
  (3 * t + 4, 5 * t - 8)

theorem particle_speed_is_sqrt_34 :
  particleSpeed particlePosition = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_is_sqrt_34_l837_83705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_separation_l837_83760

/-- The time when Adam and Simon are 80 miles apart -/
noncomputable def separation_time : ℝ := 40 * Real.sqrt 41 / 41

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 10

/-- Simon's speed in miles per hour -/
def simon_speed : ℝ := 8

/-- The distance between Adam and Simon after separation_time hours -/
def separation_distance : ℝ := 80

theorem bicycle_separation :
  adam_speed * separation_time ^ 2 + simon_speed * separation_time ^ 2 = separation_distance ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_separation_l837_83760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_angle_l837_83793

theorem max_value_angle (A : Real) : 
  (∀ θ : Real, Real.sin (A / 2) + Real.sqrt 3 * Real.cos (A / 2) ≥ Real.sin (θ / 2) + Real.sqrt 3 * Real.cos (θ / 2)) →
  A = 120 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_angle_l837_83793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_with_magnitude_increase_l837_83733

-- Define the relationship between energy and Richter magnitude
def energy_magnitude_relation (E : ℝ) (M : ℝ) : Prop :=
  Real.log E / Real.log 10 = 4.8 + 1.5 * M

-- Theorem statement
theorem energy_increase_with_magnitude_increase
  (E₁ E₂ M : ℝ) (h₁ : energy_magnitude_relation E₁ M) (h₂ : energy_magnitude_relation E₂ (M + 1)) :
  E₂ / E₁ = (10 : ℝ) ^ (3 / 2) := by
  sorry

#check energy_increase_with_magnitude_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_with_magnitude_increase_l837_83733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_m_l837_83720

/-- A function is quadratic if it can be written as ax^2 + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function defined by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * (x ^ (m^2 - 3*m + 2))

/-- Theorem stating that m = 0 is the only value that makes f a quadratic function -/
theorem unique_quadratic_m : 
  (∃ m : ℝ, IsQuadratic (f m)) ↔ (∃ m : ℝ, m = 0 ∧ IsQuadratic (f m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_m_l837_83720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AEP_measure_l837_83787

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line segment between two points -/
structure Segment where
  start : Point
  endpoint : Point

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The configuration of points and segments in the problem -/
structure Configuration where
  A : Point
  B : Point
  E : Point
  F : Point
  P : Point
  AB : Segment
  BE : Segment
  BF : Segment
  FE : Segment
  EP : Segment
  semicircle_AB : Semicircle
  semicircle_BE : Semicircle

/-- The conditions given in the problem -/
class ProblemConditions (config : Configuration) where
  E_midpoint_AB : config.E = Point.mk ((config.A.x + config.B.x) / 2) ((config.A.y + config.B.y) / 2)
  BF_FE_ratio : (config.BF.endpoint.x - config.BF.start.x) / (config.FE.endpoint.x - config.FE.start.x) = 1 / 3
  semicircles_diameters : config.semicircle_AB.radius * 2 = config.AB.endpoint.x - config.AB.start.x ∧
                          config.semicircle_BE.radius * 2 = config.BE.endpoint.x - config.BE.start.x
  EP_splits_equally : sorry  -- This condition requires more complex geometry definitions

/-- The theorem to be proved -/
theorem angle_AEP_measure (config : Configuration) [ProblemConditions config] :
  let angle_AEP := sorry  -- Definition of angle measure
  angle_AEP = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_AEP_measure_l837_83787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_third_quadrant_l837_83731

theorem tan_double_angle_third_quadrant (α : ℝ) :
  (π / 2 < α ∧ α < π) →  -- α is in the third quadrant
  Real.sin (π - α) = -3/5 →   -- given condition
  Real.tan (2 * α) = 24/7 :=  -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_third_quadrant_l837_83731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_triangle_series_sum_l837_83795

theorem leibniz_triangle_series_sum : 
  ∑' (n : ℕ), (1 : ℝ) / (n * (n + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leibniz_triangle_series_sum_l837_83795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l837_83727

open Real

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * log x

-- Define the derivative of f(x)
noncomputable def f_deriv (x : ℝ) : ℝ := log x + 1

-- Theorem statement
theorem f_monotone_increasing :
  StrictMonoOn f {x | x > exp (-1)} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l837_83727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersecting_l837_83722

/-- The circles x^2 + y^2 = 2 and x^2 + y^2 + 4y + 3 = 0 are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 = 2 ∧ x^2 + y^2 + 4*y + 3 = 0) :=
by
  sorry

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*y + 3 = 0

/-- The center of the first circle -/
def center1 : ℝ × ℝ := (0, 0)

/-- The radius of the first circle -/
noncomputable def radius1 : ℝ := Real.sqrt 2

/-- The center of the second circle -/
def center2 : ℝ × ℝ := (0, -2)

/-- The radius of the second circle -/
def radius2 : ℝ := 1

/-- The distance between the centers of the two circles -/
def distance_between_centers : ℝ := 2

/-- Theorem stating that the circles are intersecting -/
theorem circles_intersecting :
  radius1 - radius2 < distance_between_centers ∧
  distance_between_centers < radius1 + radius2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersecting_l837_83722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l837_83742

theorem inequality_equivalence (x : ℝ) (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  (2^(3*x - 1) < (2 : ℝ) ↔ x < 2/3) ∧
  (a > 1 → (a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) ↔ x < 4/3)) ∧
  (0 < a ∧ a < 1 → (a^(3*x^2 + 3*x - 1) < a^(3*x^2 + 3) ↔ x > 4/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l837_83742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l837_83738

-- Define the points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (6, 5)

-- Define the parabola
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 8 * P.1

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (min_dist : ℝ), min_dist = 8 ∧
  ∀ (P : ℝ × ℝ), on_parabola P → distance A P + distance B P ≥ min_dist := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l837_83738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_log_inequality_l837_83753

theorem exponential_inequality_implies_log_inequality (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_log_inequality_l837_83753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_three_not_in_range_l837_83719

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, y ≠ 3 → ∃ x : ℝ, f x = y :=
by
  sorry

-- State that 3 is not in the range of f
theorem three_not_in_range :
  ¬∃ x : ℝ, f x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_three_not_in_range_l837_83719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_airline_connects_over_200_cities_l837_83767

/-- Represents a city --/
def City : Type := ℕ

/-- Represents an airline --/
def Airline : Type := ℕ

/-- The total number of cities --/
def totalCities : ℕ := 600

/-- The total number of airlines --/
def totalAirlines : ℕ := 6

/-- A function that assigns an airline to each pair of cities --/
def flightAssignment : City → City → Airline := sorry

/-- A predicate that checks if two cities are connected by an airline (directly or indirectly) --/
def areConnected (airline : Airline) (city1 city2 : City) : Prop := sorry

/-- The main theorem --/
theorem no_airline_connects_over_200_cities :
  ∃ (assignment : City → City → Airline),
    ∀ (airline : Airline),
      ¬∃ (cities : Finset City),
        (cities.card > 200) ∧
        (∀ (c1 c2 : City), c1 ∈ cities → c2 ∈ cities → areConnected airline c1 c2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_airline_connects_over_200_cities_l837_83767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_broken_line_even_segments_broken_line_existence_l837_83707

/-- A closed self-intersecting broken line in the plane -/
structure BrokenLine where
  segments : ℕ
  closed : Bool
  self_intersecting : Bool

/-- Predicate for a valid broken line according to the problem conditions -/
def is_valid_broken_line (bl : BrokenLine) : Prop :=
  bl.closed ∧ 
  bl.self_intersecting ∧ 
  ∀ s₁ s₂ : Fin bl.segments, s₁ ≠ s₂ → 
    (∃! s₃, s₃ ≠ s₁ ∧ s₃ ≠ s₂ ∧ s₁.val ≠ s₃.val) ∧
  ∀ v : Fin (bl.segments + 1), ¬ (∃ s : Fin bl.segments, v.val ≠ s.val)

/-- The main theorem: a valid broken line must have an even number of segments -/
theorem valid_broken_line_even_segments (bl : BrokenLine) : 
  is_valid_broken_line bl → Even bl.segments := by
  sorry

/-- Corollary for the specific cases in the problem -/
theorem broken_line_existence (n : ℕ) : 
  (n = 999 ∨ n = 100) → 
  (∃ bl : BrokenLine, bl.segments = n ∧ is_valid_broken_line bl) ↔ Even n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_broken_line_even_segments_broken_line_existence_l837_83707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_difference_sum_property_l837_83749

theorem subset_difference_sum_property (n : ℕ) (hn : n ≤ 70) :
  ∀ A : Finset ℕ, A ⊆ Finset.range 51 → A.card = 35 →
    ∃ a b, a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b = n ∨ a + b = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_difference_sum_property_l837_83749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l837_83744

def number_of_ways_to_distribute (n m : ℕ) : ℕ := m^n

theorem distribution_count (n m : ℕ) : 
  n > 0 → m > 0 → number_of_ways_to_distribute n m = m^n :=
by
  intros hn hm
  rfl

#eval number_of_ways_to_distribute 7 12  -- This will evaluate to 35831808

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l837_83744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_value_l837_83706

-- Define the line equation
def line (x y a : ℝ) : Prop := x - y + a = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Define the perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Main theorem
theorem intersection_perpendicular_value (a : ℝ) :
  (∃ A B : ℝ × ℝ,
    line A.1 A.2 a ∧ circle_eq A.1 A.2 ∧
    line B.1 B.2 a ∧ circle_eq B.1 B.2 ∧
    perpendicular A B circle_center) →
  a = 0 ∨ a = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_value_l837_83706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_count_l837_83790

/-- A set with 2019 elements -/
def S : Type := Fin 2019

/-- The collection of subsets of S satisfying the given conditions -/
def SubsetCollection (n : ℕ) : Type := Fin n → Set S

/-- The condition that the union of any three subsets gives S -/
def UnionOfThreeIsS {n : ℕ} (collection : SubsetCollection n) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → (collection i ∪ collection j ∪ collection k) = Set.univ

/-- The condition that the union of any two subsets does not give S -/
def UnionOfTwoIsNotS {n : ℕ} (collection : SubsetCollection n) : Prop :=
  ∀ i j, i ≠ j → (collection i ∪ collection j) ≠ Set.univ

/-- The theorem stating the maximum value of n -/
theorem max_subsets_count :
  ∃ (n : ℕ), n = 64 ∧
  (∃ (collection : SubsetCollection n),
    UnionOfThreeIsS collection ∧ UnionOfTwoIsNotS collection) ∧
  (∀ (m : ℕ), m > 64 →
    ¬∃ (collection : SubsetCollection m),
      UnionOfThreeIsS collection ∧ UnionOfTwoIsNotS collection) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_count_l837_83790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_xcosx_l837_83737

open Set
open MeasureTheory
open Interval
open Real

-- Define the function to be integrated
noncomputable def f (x : ℝ) : ℝ := sqrt (1 - x^2) + x * cos x

-- State the theorem
theorem integral_sqrt_plus_xcosx :
  ∫ x in Icc (-1 : ℝ) 1, f x = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_plus_xcosx_l837_83737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_ratio_l837_83714

theorem pyramid_volume_ratio (V U : ℝ) (m n m₁ n₁ : ℝ) 
  (hV : V > 0) (hU : U > 0) (hm : m > 0) (hn : n > 0) (hm₁ : m₁ > 0) (hn₁ : n₁ > 0) :
  U / V = (m₁ + n₁)^2 / (m + n)^2 := by
  sorry

#check pyramid_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_ratio_l837_83714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l837_83774

theorem problem_solution : 
  ((Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 = Real.sqrt 2) ∧ 
  ((Real.sqrt 3 - 1) * (Real.sqrt 3 + 1) + 2 * (Real.sqrt 3 - 2) = 2 * Real.sqrt 3 - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l837_83774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tangent_l837_83766

theorem arithmetic_sequence_tangent (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  (a 1 + a 7 + a 13 = Real.pi) →                    -- given sum condition
  Real.tan (a 2 + a 12) = -Real.sqrt 3 :=           -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tangent_l837_83766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_methods_necessary_l837_83761

/-- Represents the types of sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
deriving BEq, Repr

/-- Represents a population with different household types --/
structure Population where
  total : ℕ
  farmers : ℕ
  workers : ℕ
  intellectuals : ℕ
  total_eq : total = farmers + workers + intellectuals

/-- Represents a sampling plan --/
structure SamplingPlan where
  population : Population
  sample_size : ℕ
  methods : List SamplingMethod

/-- Checks if a sampling plan is valid for the given population --/
def is_valid_sampling_plan (plan : SamplingPlan) : Prop :=
  plan.sample_size > 0 ∧
  plan.sample_size ≤ plan.population.total ∧
  plan.methods.length > 0

/-- Theorem stating that all three sampling methods are necessary --/
theorem all_methods_necessary (pop : Population)
  (h_pop : pop.total = 2000 ∧ pop.farmers = 1800 ∧ pop.workers = 100 ∧ pop.intellectuals = 100)
  (sample_size : ℕ) (h_sample : sample_size = 40) :
  ∃ (plan : SamplingPlan),
    is_valid_sampling_plan plan ∧
    plan.population = pop ∧
    plan.sample_size = sample_size ∧
    plan.methods.length = 3 ∧
    plan.methods.contains SamplingMethod.SimpleRandom ∧
    plan.methods.contains SamplingMethod.Systematic ∧
    plan.methods.contains SamplingMethod.Stratified := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_methods_necessary_l837_83761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l837_83771

theorem equation_solution (k : ℝ) : (1/2 : ℝ)^(23 : ℝ) * (1/81 : ℝ)^k = (1/18 : ℝ)^(23 : ℝ) → k = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l837_83771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_construction_l837_83700

/-- Given a length and two angles, a unique triangle can be constructed -/
theorem unique_triangle_construction (a : ℝ) (α β : ℝ) 
  (h1 : a > 0) 
  (h2 : α > 0) 
  (h3 : β > 0) 
  (h4 : α + β < π) : 
  ∃! (b c : ℝ), 
    b > 0 ∧ c > 0 ∧ 
    a / Real.sin α = b / Real.sin β ∧
    a / Real.sin α = c / Real.sin (π - α - β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_construction_l837_83700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l837_83747

/-- The probability of drawing marbles from a bag containing 5 blue, 7 white, and 4 red marbles
    until 3 remain, with those 3 being one of each color. -/
theorem marble_probability (blue : ℕ) (white : ℕ) (red : ℕ) 
  (h_blue : blue = 5) (h_white : white = 7) (h_red : red = 4) :
  let total := blue + white + red
  let drawn := total - 3
  let favorable := (blue.choose (blue - 1)) * (white.choose (white - 1)) * (red.choose (red - 1))
  let total_ways := total.choose drawn
  (favorable : ℚ) / total_ways = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l837_83747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l837_83796

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  R : Real -- circumradius
  r : Real -- inradius

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.R = Real.sqrt 3 ∧
  Real.sin t.B ^ 2 + Real.sin t.C ^ 2 - Real.sin t.B * Real.sin t.C = Real.sin t.A ^ 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.a = 3 ∧ 0 < t.r ∧ t.r ≤ Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l837_83796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l837_83759

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 - a) * x - 3 else Real.log x / Real.log a

-- State the theorem
theorem increasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 2 ≤ a ∧ a < 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l837_83759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_and_constant_l837_83712

noncomputable def f (p : ℝ) (x : ℝ) : ℝ := x^p
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem tangent_point_and_constant (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f p x = g x ∧ (deriv (f p)) x = (deriv g) x) →
  p = 1 / Real.exp 1 ∧
  ∃ x y : ℝ, x = Real.exp (Real.exp 1) ∧ y = Real.exp 1 ∧ f p x = y ∧ g x = y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_and_constant_l837_83712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83758

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then x^2 - x + 3 else x + 2/x

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ |x/2 + a|) ↔ -47/16 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_area_line_minimizes_area_l837_83785

/-- A line passing through a point and intersecting positive x and y axes -/
structure IntersectingLine where
  -- The slope and y-intercept of the line
  m : ℝ
  b : ℝ
  -- The point the line passes through
  p : ℝ × ℝ
  -- The line passes through the given point
  point_on_line : p.2 = m * p.1 + b
  -- The line intersects positive x and y axes
  intersects_axes : 0 < -b/m ∧ 0 < b

/-- The area of the triangle formed by the line and the axes -/
noncomputable def triangle_area (l : IntersectingLine) : ℝ :=
  (-l.b / l.m) * l.b / 2

/-- The equation of the line in the form ax + by + c = 0 -/
def line_equation (l : IntersectingLine) : ℝ × ℝ × ℝ :=
  (l.m, -1, l.b)

theorem line_through_point_with_area
  (l : IntersectingLine)
  (h : l.p = (3, 2))
  (area : triangle_area l = 12) :
  line_equation l = (2, 3, -12) := by
  sorry

theorem line_minimizes_area
  (l : IntersectingLine)
  (h : l.p = (3, 2)) :
  line_equation l = (2, 3, -12) ↔ 
  ∀ (l' : IntersectingLine), l'.p = (3, 2) → triangle_area l ≤ triangle_area l' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_area_line_minimizes_area_l837_83785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_last_number_l837_83736

theorem sequence_last_number (a : ℕ → ℤ) (h1 : a 1 = 5) 
  (h2 : ∀ n ∈ Finset.range 32, (Finset.range 6).sum (λ i ↦ a (n + i)) = 29) :
  a 37 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_last_number_l837_83736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_MN_max_area_BPQ_l837_83776

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define point A
def A : ℝ × ℝ := (1, 1/2)

-- Define point B
def B : ℝ × ℝ := (1, 2)

-- Part I: Slope of line MN
theorem slope_of_MN :
  ∀ (M N : ℝ × ℝ),
  ellipse M.1 M.2 → ellipse N.1 N.2 →
  (M.1 + N.1) / 2 = A.1 ∧ (M.2 + N.2) / 2 = A.2 →
  (N.2 - M.2) / (N.1 - M.1) = -1 :=
by
  sorry

-- Part II: Maximum area of triangle BPQ
theorem max_area_BPQ :
  ∃ (t : ℝ),
  t ≠ 0 →
  ∀ (P Q : ℝ × ℝ),
  ellipse P.1 P.2 → ellipse Q.1 Q.2 →
  P.2 = 2 * P.1 + t → Q.2 = 2 * Q.1 + t →
  ∀ (area : ℝ),
  area = abs ((B.1 - P.1) * (Q.2 - P.2) - (B.2 - P.2) * (Q.1 - P.1)) / 2 →
  area ≤ Real.sqrt 2 / 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_MN_max_area_BPQ_l837_83776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_length_l837_83799

-- Define the triangular pyramid
structure TriangularPyramid where
  S : EuclideanSpace ℝ (Fin 3)
  A : EuclideanSpace ℝ (Fin 3)
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)

-- Define the properties of the pyramid
def isEqualLateralEdges (p : TriangularPyramid) : Prop :=
  dist p.S p.A = dist p.S p.B ∧ dist p.S p.B = dist p.S p.C

-- We'll assume the existence of a dihedral angle function
noncomputable def dihedralAngle (p q r : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

def sumDihedralAngles180 (p : TriangularPyramid) : Prop :=
  dihedralAngle p.S p.A p.B + dihedralAngle p.S p.C p.B = 180

-- Define the theorem
theorem lateral_edge_length 
  (p : TriangularPyramid) 
  (h1 : isEqualLateralEdges p) 
  (h2 : sumDihedralAngles180 p) 
  (a c : ℝ) 
  (h3 : dist p.A p.B = a) 
  (h4 : dist p.A p.C = c) : 
  dist p.S p.A = (Real.sqrt (a^2 + c^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_edge_length_l837_83799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_ii_l837_83713

-- Define the circles
def CircleI : ℝ → Prop := λ r ↦ r = 3

def CircleII : ℝ → Prop := λ r ↦ ∃ (r_i : ℝ), CircleI r_i ∧ r = 2 * r_i

-- Define the relationship between the circles
def CircleRelationship (r_i r_ii : ℝ) : Prop :=
  CircleI r_i ∧ CircleII r_ii

-- Define the area of a circle
noncomputable def CircleArea (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem area_of_circle_ii :
  ∀ (r_i r_ii : ℝ),
  CircleRelationship r_i r_ii →
  CircleArea r_ii = 36 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_circle_ii_l837_83713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_laps_calculation_l837_83781

/-- The length of one lap on a standard running track in meters. -/
def standard_lap_length : ℚ := 400

/-- The total distance of the race in meters. -/
def race_distance : ℚ := 5000

/-- The number of laps in the race. -/
def number_of_laps : ℚ := race_distance / standard_lap_length

theorem race_laps_calculation :
  number_of_laps = 25/2 :=
by
  unfold number_of_laps race_distance standard_lap_length
  norm_num

#eval number_of_laps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_laps_calculation_l837_83781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l837_83710

noncomputable def f (x : ℝ) := 1 / Real.sqrt (x + 2) - Real.sqrt (3 - x)

theorem f_domain : Set.Ioo (-2 : ℝ) 3 ⊆ {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l837_83710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_composition_l837_83773

-- Define the functions P and Q
noncomputable def P (x : ℝ) : ℝ := 3 * Real.sqrt x
def Q (x : ℝ) : ℝ := x^3

-- State the theorem
theorem nested_function_composition :
  P (Q (P (Q (P (Q 2))))) = 1944 * Real.rpow 6 (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_composition_l837_83773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cdh_ch_d_is_parallelogram_l837_83716

-- Define the points
variable (A B C D H_c H_d : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def inscribed_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the orthocenter of a triangle
def orthocenter (H X Y Z : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define a parallelogram
def Parallelogram (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem cdh_ch_d_is_parallelogram 
  (h_inscribed : inscribed_quadrilateral A B C D)
  (h_orthocenter_c : orthocenter H_c A B D)
  (h_orthocenter_d : orthocenter H_d A B C) :
  Parallelogram C D H_c H_d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cdh_ch_d_is_parallelogram_l837_83716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l837_83798

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Define the interval
def I : Set ℝ := Set.Icc (-3) 3

-- State the theorem
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧ 
               (∀ x ∈ I, m ≤ f x) ∧
               (M - m = 32) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_difference_l837_83798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_7pi_18_g_solution_set_l837_83769

noncomputable section

open Real

/-- The function f(x) defined as sin(3x + π/6) + 1 -/
def f (x : ℝ) : ℝ := sin (3 * x + π / 6) + 1

/-- The function g(x) derived from f(x) -/
def g (x : ℝ) : ℝ := sqrt 2 * sin (3 * x - π / 3) + sqrt 2

/-- The set of integers -/
def ℤSet : Set ℤ := Set.univ

/-- The solution set for g(x) = √2/2 -/
def solution_set : Set ℝ :=
  {x | ∃ k : ℤ, x = 2 * π / 3 * ↑k + π / 2 ∨ x = 2 * π / 3 * ↑k + 13 * π / 18}

theorem f_value_at_7pi_18 :
  f (7 * π / 18) = 1 - sqrt 3 / 2 := by sorry

theorem g_solution_set :
  ∀ x : ℝ, g x = sqrt 2 / 2 ↔ x ∈ solution_set := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_7pi_18_g_solution_set_l837_83769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l837_83786

def a : ℕ → ℚ
| 0 => 2
| n + 1 => 2 / (a n + 1)

def b (n : ℕ) : ℚ := |((a n + 2) / (a n - 1))|

theorem b_formula (n : ℕ) : b n = 2^(n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l837_83786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_of_squares_polynomial_with_negative_value_has_real_zero_l837_83703

/-- A real coefficient polynomial with positive leading coefficient and no real zeros -/
structure PositiveRealPolynomial where
  p : Polynomial ℝ
  leading_positive : 0 < p.leadingCoeff
  no_real_zeros : ∀ x : ℝ, p.eval x ≠ 0

/-- A real coefficient polynomial with positive leading coefficient -/
structure PositiveLeadingPolynomial where
  q : Polynomial ℝ
  leading_positive : 0 < q.leadingCoeff

theorem polynomial_sum_of_squares (p : PositiveRealPolynomial) :
  ∃ f g : Polynomial ℝ, p.p = f^2 + g^2 := by
  sorry

theorem polynomial_with_negative_value_has_real_zero
  (q : PositiveLeadingPolynomial) (a : ℝ) (ha : q.q.eval a < 0) :
  ∃ x : ℝ, q.q.eval x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_of_squares_polynomial_with_negative_value_has_real_zero_l837_83703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_congruences_l837_83745

theorem min_sum_with_congruences :
  ∃ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a ^ b) % 10 = 4 ∧
    (b ^ c) % 10 = 2 ∧
    (c ^ a) % 10 = 9 ∧
    (∀ (x y z : ℕ),
      x > 0 → y > 0 → z > 0 →
      (x ^ y) % 10 = 4 →
      (y ^ z) % 10 = 2 →
      (z ^ x) % 10 = 9 →
      a + b + c ≤ x + y + z) ∧
    a + b + c = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_with_congruences_l837_83745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l837_83784

/-- The x-intercept of a line passing through two given points -/
noncomputable def x_intercept (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ - y₁ * (x₁ - x₂) / (y₁ - y₂)

/-- Theorem: The x-intercept of a line passing through (10, 3) and (-2, -3) is 4 -/
theorem x_intercept_specific_line : x_intercept 10 3 (-2) (-3) = 4 := by
  -- Unfold the definition of x_intercept
  unfold x_intercept
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l837_83784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83725

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x^2 + 3*x else Real.log (x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, |f x| ≥ a * x) ↔ a ∈ Set.Icc (-3) 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l837_83725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequality_l837_83788

theorem logarithm_inequality : (0.8 : Real)^(3.1 : Real) < Real.log 7 / Real.log 3 ∧ Real.log 7 / Real.log 3 < (2 : Real)^(1.1 : Real) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequality_l837_83788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_g_derivative_positive_l837_83783

noncomputable section

open Real Set

def f (a x : ℝ) : ℝ := a / x + log x - 1

def g (x : ℝ) : ℝ := (log x - 1) * exp x + x

theorem f_minimum (a : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (exp 1), ∃ y ∈ Set.Ioo (0 : ℝ) (exp 1), f a y < f a x) ∨
  (∃ x ∈ Set.Ioo (0 : ℝ) (exp 1), ∀ y ∈ Set.Ioo (0 : ℝ) (exp 1), f a x ≤ f a y ∧ f a x = log a) ∨
  (∀ y ∈ Set.Ioo (0 : ℝ) (exp 1), f a (exp 1) ≤ f a y ∧ f a (exp 1) = a / exp 1) :=
sorry

theorem g_derivative_positive :
  ∀ x ∈ Set.Ioo (0 : ℝ) (exp 1), HasDerivAt g ((1 / x + log x - 1) * exp x + 1) x :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_g_derivative_positive_l837_83783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l837_83741

-- Define the point P
def P : ℝ × ℝ := (-1, 3)

-- Define the distance from origin to P
noncomputable def r : ℝ := Real.sqrt ((P.1)^2 + (P.2)^2)

-- Theorem statement
theorem cos_alpha_value (α : ℝ) (h : ∃ t : ℝ, t > 0 ∧ t • P = (Real.cos α, Real.sin α)) : 
  Real.cos α = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l837_83741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_difference_l837_83797

/-- The sum of the geometric series for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

/-- Proof that T(b) - T(-b) = 324b for b satisfying the given conditions -/
theorem geometric_series_difference (b : ℝ) 
    (h1 : -1 < b) (h2 : b < 1) 
    (h3 : T b * T (-b) = 3240) : 
  T b - T (-b) = 324 * b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_difference_l837_83797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_monthly_price_is_500_l837_83723

def normal_monthly_price (hourly_wage : ℚ) (hours_per_week : ℚ) (weeks_per_month : ℚ) (months_per_year : ℚ) (annual_insurance_cost : ℚ) : ℚ :=
  let monthly_income := hourly_wage * hours_per_week * weeks_per_month
  let annual_income := monthly_income * months_per_year
  let government_contribution_rate := 
    if annual_income < 10000 then 0.9
    else if annual_income ≤ 40000 then 0.5
    else 0.2
  let total_annual_cost := annual_insurance_cost / (1 - government_contribution_rate)
  total_annual_cost / months_per_year

theorem normal_monthly_price_is_500 :
  normal_monthly_price 25 30 4 12 3000 = 500 := by
  -- The proof goes here
  sorry

#eval normal_monthly_price 25 30 4 12 3000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_monthly_price_is_500_l837_83723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_current_speed_l837_83755

/-- Represents the speed of a boat in km/h -/
noncomputable def boat_speed (distance : ℝ) (time_minutes : ℝ) : ℝ :=
  (distance / time_minutes) * 60

/-- Calculates the speed of the current given upstream and downstream speeds -/
noncomputable def current_speed (upstream_speed downstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem boat_current_speed : 
  let upstream_speed := boat_speed 1 25
  let downstream_speed := boat_speed 1 12
  current_speed upstream_speed downstream_speed = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_current_speed_l837_83755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_subset_of_intersection_eq_l837_83715

variable {U : Type} [Nonempty U]

theorem complement_subset_of_intersection_eq (M N : Set U) 
  (h1 : M ∩ N = N) (h2 : M ⊆ univ) (h3 : N ⊆ univ) : 
  (Mᶜ) ⊆ (Nᶜ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_subset_of_intersection_eq_l837_83715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l837_83729

-- Define a and b as variables
variable (a b : ℝ)

-- Define the given conditions
axiom a_def : Real.exp (Real.log 2 * a) = 3
axiom b_def : Real.exp (Real.log 3 * b) = 2

-- Define the function f
noncomputable def f (x : ℝ) := a^x + x - b

-- State the theorem
theorem f_has_unique_zero : ∃! x, f a b x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l837_83729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_distinct_values_of_z_l837_83728

/-- Given two four-digit integers x and y where y is formed by reversing
    the middle two digits of x, prove that |x - y| can take 10 distinct values. -/
theorem ten_distinct_values_of_z (x y : ℕ) (h1 : 1000 ≤ x ∧ x ≤ 9999)
    (h2 : 1000 ≤ y ∧ y ≤ 9999)
    (h3 : ∃ (a b c d : ℕ), x = 1000 * a + 100 * b + 10 * c + d ∧
                            y = 1000 * a + 100 * c + 10 * b + d ∧
                            1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) :
    ∃ (S : Finset ℕ), S.card = 10 ∧ ∀ z, z ∈ S ↔ ∃ (x' y' : ℕ), 
        (1000 ≤ x' ∧ x' ≤ 9999) ∧ 
        (1000 ≤ y' ∧ y' ≤ 9999) ∧
        (∃ (a b c d : ℕ), x' = 1000 * a + 100 * b + 10 * c + d ∧
                          y' = 1000 * a + 100 * c + 10 * b + d ∧
                          1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) ∧
        z = Int.natAbs (x' - y') :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_distinct_values_of_z_l837_83728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_characterization_l837_83791

theorem subset_characterization (A : Set ℕ) (h1 : A.Nonempty) (h2 : A.ncard ≥ 2) :
  (∀ x y, x ∈ A → y ∈ A → x ≠ y → (x + y) / Nat.gcd x y ∈ A) →
  ∃ d : ℕ, d ≥ 3 ∧ A = {d, d * (d - 1)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_characterization_l837_83791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l837_83756

-- Define the series A
noncomputable def A : ℝ := ∑' n, if n % 4 ≠ 0 then ((-1) ^ ((n - 1) / 4)) / n^2 else 0

-- Define the series B
noncomputable def B : ℝ := ∑' n, if n % 4 = 0 then ((-1) ^ (n / 4 - 1)) / n^2 else 0

-- Theorem statement
theorem A_div_B_eq_17 : A / B = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l837_83756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_negative_sqrt3_sqrt3_l837_83743

theorem polar_coordinates_of_negative_sqrt3_sqrt3 :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r * (Real.cos θ) = -Real.sqrt 3 ∧
  r * (Real.sin θ) = Real.sqrt 3 ∧
  r = Real.sqrt 6 ∧
  θ = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinates_of_negative_sqrt3_sqrt3_l837_83743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l837_83782

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The focal length of a hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 + h.b^2)

/-- The area of triangle ODE formed by the origin and the intersection points of x=a with the asymptotes -/
def triangle_area (h : Hyperbola) : ℝ := h.a * h.b

theorem min_focal_length (h : Hyperbola) (h_area : triangle_area h = 8) :
  8 ≤ focal_length h ∧ ∃ (h' : Hyperbola), triangle_area h' = 8 ∧ focal_length h' = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l837_83782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l837_83734

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) :
  total = 850 →
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  ↑(total - (total * (muslim_percent + hindu_percent + sikh_percent)).num / (muslim_percent + hindu_percent + sikh_percent).num) = 153 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_communities_count_l837_83734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chef_wage_25_percent_greater_l837_83748

/-- Represents the hourly wages at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℚ
  chef : ℚ
  dishwasher : ℚ

/-- Conditions for wages at Joe's Steakhouse -/
def wage_conditions (w : SteakhouseWages) : Prop :=
  w.manager = 17/2 ∧
  w.chef = w.manager - 51/16 ∧
  w.dishwasher = w.manager / 2

/-- The percentage difference between chef and dishwasher wages -/
def chef_dishwasher_percentage (w : SteakhouseWages) : ℚ :=
  (w.chef - w.dishwasher) / w.dishwasher * 100

/-- Theorem stating that the chef's wage is 25% greater than the dishwasher's wage -/
theorem chef_wage_25_percent_greater (w : SteakhouseWages) 
  (h : wage_conditions w) : chef_dishwasher_percentage w = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chef_wage_25_percent_greater_l837_83748
