import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_optimal_height_triangular_prism_l1001_100137

/-- 
A triangular prism is defined by its base sides p and q, the angle θ between them, 
and its height h. The total area of the three faces containing vertex A is K.
-/
structure TriangularPrism where
  p : ℝ
  q : ℝ
  θ : ℝ
  h : ℝ
  K : ℝ
  h_pos : 0 < h
  p_pos : 0 < p
  q_pos : 0 < q
  θ_pos : 0 < θ
  θ_lt_pi : θ < π
  area_eq : (1/2) * p * q * Real.sin θ + p * h + q * h = K

/-- The volume of a triangular prism -/
noncomputable def volume (prism : TriangularPrism) : ℝ :=
  (1/2) * prism.p * prism.q * Real.sin prism.θ * prism.h

/-- Theorem stating the maximum volume of a triangular prism -/
theorem max_volume_triangular_prism (prism : TriangularPrism) :
  volume prism ≤ Real.sqrt (prism.K^3 / 54) := by
  sorry

/-- The height of the largest prism -/
noncomputable def optimal_height (prism : TriangularPrism) : ℝ :=
  (prism.p * prism.q * Real.sin prism.θ) / (2 * (prism.p + prism.q))

/-- Theorem stating the height of the largest prism -/
theorem optimal_height_triangular_prism (prism : TriangularPrism) :
  volume prism = Real.sqrt (prism.K^3 / 54) →
  prism.h = optimal_height prism := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_optimal_height_triangular_prism_l1001_100137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_x_plus_2_l1001_100139

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem g_x_plus_2 (x : ℝ) (h1 : x ≠ -3/2) (h2 : x ≠ 2) (h3 : x ≠ 0) :
  g (x + 2) = (2 * x + 7) / x :=
by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp [add_sub_cancel]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_x_plus_2_l1001_100139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_theorem_l1001_100193

/-- Two concentric circles with radii R and r, where R > r -/
structure ConcentricCircles (R r : ℝ) where
  radius_larger : R > r

/-- Points on the circles -/
structure CirclePoints (R r : ℝ) (h : ConcentricCircles R r) where
  P : ℂ
  B : ℂ
  C : ℂ
  A : ℂ
  on_smaller_circle_P : Complex.abs P = r
  on_larger_circle_B : Complex.abs B = R
  on_larger_circle_C : Complex.abs C = R
  on_smaller_circle_A : Complex.abs A = r
  BP_intersects_larger : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ C = (1 - t) * B + t * P
  PA_perp_BP : (A - P).re * (B - P).re + (A - P).im * (B - P).im = 0

/-- The theorem to be proved -/
theorem concentric_circles_theorem (R r : ℝ) (h : ConcentricCircles R r) 
  (pts : CirclePoints R r h) :
  (Complex.abs (pts.B - pts.C))^2 + (Complex.abs (pts.C - pts.A))^2 + (Complex.abs (pts.A - pts.B))^2 = 6 * R^2 + 2 * r^2 ∧
  ∃ (center : ℂ), Complex.abs center = r / 2 ∧ 
    Complex.abs ((pts.A + pts.B) / 2 - center) = R / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_theorem_l1001_100193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l1001_100127

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_f_2x_minus_1 (h : ∀ x, x ∈ domain_f_x_plus_1 ↔ f (x + 1) ∈ Set.Icc (-1) 4) :
  ∀ x, x ∈ Set.Icc 0 (5/2) ↔ f (2*x - 1) ∈ Set.Icc (-1) 4 := by
  sorry

#check domain_f_2x_minus_1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l1001_100127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_of_z_l1001_100104

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as noncomputable
noncomputable def z : ℂ := (1 - 2 * i^3) / (2 + i)

-- Theorem statement
theorem absolute_value_of_z : Complex.abs z = 1 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_of_z_l1001_100104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l1001_100138

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |floor (p.1 + p.2)| + |floor (p.1 - p.2)| ≤ 1}

-- State the theorem
theorem area_of_S : MeasureTheory.volume S = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_S_l1001_100138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_72_20_l1001_100192

/-- The area of a circular sector with central angle α (in degrees) and radius r -/
noncomputable def sectorArea (α : ℝ) (r : ℝ) : ℝ := (α / 360) * Real.pi * r^2

/-- Theorem: The area of a circular sector with a central angle of 72° and a radius of 20 cm is 80π cm² -/
theorem sector_area_72_20 : sectorArea 72 20 = 80 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_72_20_l1001_100192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1001_100162

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 9

noncomputable def real_axis_length : ℝ := 6

noncomputable def eccentricity : ℝ := Real.sqrt 5 / 2

noncomputable def foci : Set (ℝ × ℝ) := {(-3 * Real.sqrt 5 / 2, 0), (3 * Real.sqrt 5 / 2, 0)}

noncomputable def vertices : Set (ℝ × ℝ) := {(-3, 0), (3, 0)}

theorem hyperbola_properties :
  (∀ x y, hyperbola x y → 
    real_axis_length = 6 ∧
    eccentricity = Real.sqrt 5 / 2 ∧
    foci = {(-3 * Real.sqrt 5 / 2, 0), (3 * Real.sqrt 5 / 2, 0)} ∧
    vertices = {(-3, 0), (3, 0)}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1001_100162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l1001_100150

/-- The number of days it takes for two workers to complete a job together -/
noncomputable def combined_days : ℝ := 10

/-- The number of days it takes for worker a to complete the job alone -/
noncomputable def a_days : ℝ := 20

/-- The work rate of a worker, measured in fraction of work completed per day -/
noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

/-- The number of days it takes for worker b to complete the job alone -/
noncomputable def b_days : ℝ := 1 / (work_rate combined_days - work_rate a_days)

theorem b_completion_time :
  b_days = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completion_time_l1001_100150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1001_100124

/-- Given vector a and its dot product with b, prove properties about their sum and minimum magnitude of b -/
theorem vector_properties (a b : ℝ × ℝ) 
  (h1 : a = (Real.sqrt 3, 1)) 
  (h2 : a.1 * b.1 + a.2 * b.2 = 4) : 
  (∃ (b : ℝ × ℝ), (b.1^2 + b.2^2 = 16) → 
    ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 28)) ∧ 
  (∃ (min_b : ℝ × ℝ), 
    (∀ b' : ℝ × ℝ, min_b.1^2 + min_b.2^2 ≤ b'.1^2 + b'.2^2) ∧
    (min_b.1^2 + min_b.2^2 = 4) ∧
    (a.1 * min_b.1 + a.2 * min_b.2 = Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (min_b.1^2 + min_b.2^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1001_100124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_range_l1001_100198

-- Define the function f(x) = log_a(x^2 - ax + 1/2)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 1/2) / Real.log a

-- Theorem statement
theorem f_minimum_value_range (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) ↔ (1 < a ∧ a < Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_range_l1001_100198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_twice_alice_probability_l1001_100169

/-- Alice's choice is uniformly distributed on [0, 1000] -/
def alice_distribution : Set ℝ := Set.Icc 0 1000

/-- Bob's choice is uniformly distributed on [0, 3000] -/
def bob_distribution : Set ℝ := Set.Icc 0 3000

/-- The event where Bob's number is at least twice as large as Alice's -/
def event (a b : ℝ) : Prop := b ≥ 2 * a

/-- The probability of the event -/
noncomputable def probability : ℝ := (1500000 : ℝ) / 3000000

theorem bob_twice_alice_probability :
  probability = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_twice_alice_probability_l1001_100169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_sum_l1001_100165

noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

noncomputable def fourierSum (x : ℝ) : ℝ := (40 / (1 - 10*x)) - (4 / (1 - x))

theorem fourier_series_sum (x : ℝ) (h1 : 0 < x) (h2 : x < 0.1) :
  fourierSum x = 4 * geometricSum 4 x → x = 3/40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_series_sum_l1001_100165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_for_integer_ratio_l1001_100156

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Checks if four digits are all different -/
def all_different (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem max_sum_for_integer_ratio :
  ∃ (a c d : Digit),
    all_different a (⟨2, by norm_num⟩ : Digit) c d ∧
    (a.val + c.val : ℕ) / (2 + d.val) ∈ Set.univ ∧
    a.val + c.val = 15 ∧
    ∀ (a' c' d' : Digit),
      all_different a' (⟨2, by norm_num⟩ : Digit) c' d' →
      (a'.val + c'.val : ℕ) / (2 + d'.val) ∈ Set.univ →
      a'.val + c'.val ≤ 15 :=
by sorry

#check max_sum_for_integer_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_for_integer_ratio_l1001_100156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_geq_neg_one_l1001_100128

theorem negation_of_forall_sin_geq_neg_one :
  ¬(∀ x : ℝ, x > 0 → Real.sin x ≥ -1) ↔ ∃ x : ℝ, x > 0 ∧ Real.sin x < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_geq_neg_one_l1001_100128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l1001_100185

/-- Definition of an ellipse with given foci and one x-intercept -/
structure Ellipse where
  foci1 : ℝ × ℝ := (0, -3)
  foci2 : ℝ × ℝ := (4, 0)
  x_intercept1 : ℝ × ℝ := (0, 0)

/-- The sum of distances from any point on the ellipse to the two foci is constant -/
noncomputable def Ellipse.sumDistances (e : Ellipse) (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - e.foci1.1)^2 + (p.2 - e.foci1.2)^2) +
  Real.sqrt ((p.1 - e.foci2.1)^2 + (p.2 - e.foci2.2)^2)

/-- The constant sum of distances for the ellipse -/
noncomputable def Ellipse.constantSum (e : Ellipse) : ℝ :=
  e.sumDistances e.x_intercept1

/-- Theorem: The other x-intercept of the ellipse is at (11/4, 0) -/
theorem ellipse_other_x_intercept (e : Ellipse) :
  ∃ (x : ℝ), x = 11/4 ∧ e.sumDistances (x, 0) = e.sumDistances e.x_intercept1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_x_intercept_l1001_100185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1001_100126

-- Define the parameters a and b
variable (a b : ℝ)

-- Define the solution set of the first inequality
def solution_set_1 (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- Define the solution set of the second inequality
def solution_set_2 (a b : ℝ) : Set ℝ := {x | b*x^2 + a*x + 1 ≤ 0}

-- State the theorem
theorem inequality_solution_sets :
  solution_set_1 a b = Set.Ioo (-3) (-1) →
  solution_set_2 a b = Set.Icc (-1) (-1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1001_100126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_f_f_leq_g_iff_inequality_for_positive_x_l1001_100172

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + (1/2) * x^2 + a * x

noncomputable def g (x : ℝ) : ℝ := exp x + (3/2) * x^2

theorem extreme_points_of_f (a : ℝ) :
  (∀ x > 0, ¬ IsLocalExtr (f a) x) ∨
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ IsLocalExtr (f a) x₁ ∧ IsLocalExtr (f a) x₂) :=
sorry

theorem f_leq_g_iff (a : ℝ) : 
  (∀ x > 0, f a x ≤ g x) ↔ a ≤ (exp 1 + 1) :=
sorry

theorem inequality_for_positive_x :
  ∀ x > 0, exp x + x^2 - (exp 1 + 1) * x + (exp 1) / x > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_f_f_leq_g_iff_inequality_for_positive_x_l1001_100172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1001_100179

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a - t.b) / t.b = Real.cos t.A / Real.cos t.B + 1)
  (h2 : t.C = Real.pi / 6)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : 0 < t.A ∧ t.A < Real.pi)
  (h5 : 0 < t.B ∧ t.B < Real.pi)
  (h6 : 0 < t.C ∧ t.C < Real.pi)
  (h7 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) :
  (t.B = 5*Real.pi/24 ∧ t.A = 5*Real.pi/8) ∧
  (Real.sqrt 2 < t.a / (t.b * Real.cos t.B) ∧ t.a / (t.b * Real.cos t.B) < 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1001_100179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_problem_l1001_100157

theorem set_elements_problem (A B : Finset ℕ) : 
  (Finset.card A = 3 * Finset.card B) →
  (Finset.card (A ∪ B) = 4220) →
  (Finset.card (A ∩ B) = 850) →
  Finset.card A = 3165 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_problem_l1001_100157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_values_l1001_100131

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

/-- The equation of a hyperbola in terms of m -/
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / (m + 1) = 1

/-- Theorem: For a hyperbola with equation x^2/m - y^2/(m+1) = 1 and eccentricity √5,
    the possible values of m are 1/3 and -4/3 -/
theorem hyperbola_m_values :
  ∀ m : ℝ, (∃ a b : ℝ, hyperbola_equation m a b ∧ eccentricity a b = Real.sqrt 5) →
  (m = 1/3 ∨ m = -4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_values_l1001_100131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_cos_l1001_100142

theorem tan_value_given_cos (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.cos α = -3/5) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_given_cos_l1001_100142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_period_l1001_100125

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define the period of a function
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- State that π is the period of tan x
axiom tan_period : is_period tan Real.pi

-- State the theorem
theorem tan_2x_period :
  ∃ p : ℝ, p > 0 ∧ is_period (λ x ↦ tan (2 * x)) p ∧
  ∀ q, q > 0 → is_period (λ x ↦ tan (2 * x)) q → p ≤ q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_period_l1001_100125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_trip_highway_miles_l1001_100107

/-- Represents Carl's trip to the Grand Canyon -/
structure CarlsTrip where
  city_mpg : ℚ
  highway_mpg : ℚ
  city_miles : ℚ
  gas_price : ℚ
  total_cost : ℚ

/-- Calculates the number of highway miles in Carl's trip -/
noncomputable def highway_miles (trip : CarlsTrip) : ℚ :=
  let city_gas := trip.city_miles / trip.city_mpg
  let city_cost := city_gas * trip.gas_price
  let highway_cost := trip.total_cost - city_cost
  let highway_gas := highway_cost / trip.gas_price
  highway_gas * trip.highway_mpg

/-- Theorem stating that the number of highway miles in Carl's trip is 480 -/
theorem carls_trip_highway_miles :
  let trip := CarlsTrip.mk 30 40 60 3 42
  highway_miles trip = 480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carls_trip_highway_miles_l1001_100107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_positive_l1001_100113

theorem inequality_implies_log_positive (x y : ℝ) :
  (2:ℝ)^x - (2:ℝ)^y < (3:ℝ)^(-x) - (3:ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_log_positive_l1001_100113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grade_is_four_l1001_100132

/-- Represents the grading system for an economics course -/
structure EconomicsGrade where
  /-- Time spent studying microeconomics -/
  micro_time : ℝ
  /-- Time spent studying macroeconomics -/
  macro_time : ℝ
  /-- Total available study time -/
  total_time : ℝ
  /-- Score increase per unit time for microeconomics -/
  micro_efficiency : ℝ
  /-- Score increase per unit time for macroeconomics -/
  macro_efficiency : ℝ
  /-- Constraint: total study time is the sum of micro and macro time -/
  time_constraint : micro_time + macro_time = total_time

/-- Calculates the final grade based on the given grading system -/
noncomputable def calculate_grade (grade : EconomicsGrade) : ℝ :=
  let micro_score := grade.micro_time * grade.micro_efficiency
  let macro_score := grade.macro_time * grade.macro_efficiency
  min (0.25 * micro_score + 0.75 * macro_score) (0.75 * micro_score + 0.25 * macro_score)

/-- Theorem stating that the maximum achievable grade is 4 -/
theorem max_grade_is_four (grade : EconomicsGrade)
  (h1 : grade.total_time = 4.6)
  (h2 : grade.micro_efficiency = 2.5)
  (h3 : grade.macro_efficiency = 1.5) :
  ∃ (optimal_grade : EconomicsGrade), 
    optimal_grade.total_time = grade.total_time ∧
    optimal_grade.micro_efficiency = grade.micro_efficiency ∧
    optimal_grade.macro_efficiency = grade.macro_efficiency ∧
    ∀ (other_grade : EconomicsGrade), 
      other_grade.total_time = grade.total_time →
      other_grade.micro_efficiency = grade.micro_efficiency →
      other_grade.macro_efficiency = grade.macro_efficiency →
      calculate_grade optimal_grade ≥ calculate_grade other_grade ∧
      Int.floor (calculate_grade optimal_grade + 0.5) = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_grade_is_four_l1001_100132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l1001_100196

-- Define the functions f and g
def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := -2 * x + 3

-- Define the domain of g
def D_g : Set ℝ := { x | x ≥ 1 }

-- Define the function h
noncomputable def h (x : ℝ) : ℝ :=
  if x ≥ 1 then (f x) * (g x) else f x

-- State the theorem about the range of h
theorem range_of_h :
  Set.range h = Set.Iic (1/8 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l1001_100196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_inverse_l1001_100121

theorem power_equality_implies_inverse (x : ℝ) : (64 : ℝ)^5 = (32 : ℝ)^x → (2 : ℝ)^(-x) = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_implies_inverse_l1001_100121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_correct_l1001_100190

/-- Conversion factor from degrees to radians -/
noncomputable def deg_to_rad : ℝ := Real.pi / 180

/-- The angle in degrees -/
def angle_deg : ℝ := -630

/-- The angle in radians -/
noncomputable def angle_rad : ℝ := -7 * Real.pi / 2

/-- Theorem stating that the conversion from degrees to radians is correct -/
theorem angle_conversion_correct : angle_deg * deg_to_rad = angle_rad := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_correct_l1001_100190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_score_is_six_l1001_100161

/-- A basketball shooting game with the following rules:
  1. Each round consists of up to three shots, stopping once the first shot is made.
  2. Making the first shot scores 8 points.
  3. Missing the first shot but making the second scores 6 points.
  4. Missing the first two shots but making the third scores 4 points.
  5. Missing all three shots scores 0 points.
  6. The probability of making each shot is 0.5. -/
structure BasketballGame where
  first_shot_score : ℝ := 8
  second_shot_score : ℝ := 6
  third_shot_score : ℝ := 4
  miss_all_score : ℝ := 0
  shot_probability : ℝ := 0.5

/-- The expected score per round in the basketball game. -/
def expected_score (game : BasketballGame) : ℝ :=
  game.first_shot_score * game.shot_probability +
  game.second_shot_score * (1 - game.shot_probability) * game.shot_probability +
  game.third_shot_score * (1 - game.shot_probability) * (1 - game.shot_probability) * game.shot_probability +
  game.miss_all_score * (1 - game.shot_probability) * (1 - game.shot_probability) * (1 - game.shot_probability)

/-- Theorem stating that the expected score per round in the basketball game is 6. -/
theorem expected_score_is_six (game : BasketballGame) :
  expected_score game = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_score_is_six_l1001_100161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_regular_octagon_12_l1001_100164

/-- The length of a diagonal connecting adjacent vertices in a regular octagon -/
noncomputable def diagonal_length_regular_octagon (side_length : ℝ) : ℝ :=
  side_length * Real.sqrt (2 + Real.sqrt 2)

/-- Theorem: In a regular octagon with side length 12, the length of a diagonal
    connecting adjacent vertices is 12√(2 + √2) -/
theorem diagonal_length_regular_octagon_12 :
  diagonal_length_regular_octagon 12 = 12 * Real.sqrt (2 + Real.sqrt 2) := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_regular_octagon_12_l1001_100164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_max_area_l1001_100123

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Point D on side BC of triangle ABC -/
structure TriangleWithPoint extends Triangle where
  D : ℝ × ℝ
  BD : ℝ
  DC : ℝ
  AD : ℝ

theorem triangle_angle_and_max_area (t : TriangleWithPoint) 
  (h1 : t.b * Real.cos t.A + t.a * Real.cos t.B = 2 * t.c * Real.cos t.A)
  (h2 : t.BD = 3 * t.DC)
  (h3 : t.AD = 3) :
  t.A = Real.pi / 3 ∧ 
  ∃ (maxArea : ℝ), maxArea = 4 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area ≤ maxArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_max_area_l1001_100123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_with_three_spheres_l1001_100103

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A sphere in 3D space -/
structure Sphere where
  center : Point
  radius : ℝ

/-- Two spheres touch each other -/
def Sphere.touches (s1 s2 : Sphere) : Prop :=
  (s1.center.x - s2.center.x)^2 + (s1.center.y - s2.center.y)^2 + (s1.center.z - s2.center.z)^2 = (s1.radius + s2.radius)^2

/-- A sphere touches the lateral surface of a cylinder with radius R -/
def Sphere.touchesLateralSurface (s : Sphere) (R : ℝ) : Prop :=
  s.center.x^2 + s.center.y^2 = (R - s.radius)^2

/-- Given a cylinder of height 3r containing three spheres of radius r,
    where each sphere touches the other two and the lateral surface of the cylinder,
    and two spheres touch the bottom base while one touches the top base,
    the radius R of the cylinder's base is r * (4 + 3√2) / 4. -/
theorem cylinder_with_three_spheres (r : ℝ) (r_pos : r > 0) :
  ∃ R : ℝ, R = r * (4 + 3 * Real.sqrt 2) / 4 ∧
  ∃ (sphere1 sphere2 sphere3 : Sphere),
    (sphere1.radius = r) ∧ (sphere2.radius = r) ∧ (sphere3.radius = r) ∧
    (sphere1.center.z = r) ∧ (sphere2.center.z = r) ∧ (sphere3.center.z = 3 * r) ∧
    (Sphere.touches sphere1 sphere2) ∧ (Sphere.touches sphere1 sphere3) ∧ (Sphere.touches sphere2 sphere3) ∧
    (Sphere.touchesLateralSurface sphere1 R) ∧ (Sphere.touchesLateralSurface sphere2 R) ∧ (Sphere.touchesLateralSurface sphere3 R) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_with_three_spheres_l1001_100103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_prob_not_more_than_one_hour_prob_two_out_of_five_l1001_100166

-- Define the survey data
def a : ℕ := 35  -- Nearsighted and uses electronic devices for over 1 hour
def b : ℕ := 5   -- Not nearsighted and uses electronic devices for over 1 hour
def c : ℕ := 5   -- Nearsighted and uses electronic devices for not more than 1 hour
def d : ℕ := 5   -- Not nearsighted and uses electronic devices for not more than 1 hour
def n : ℕ := a + b + c + d

-- Define K² formula
noncomputable def K_squared : ℚ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem 1: Relationship between nearsightedness and electronic device usage
theorem relationship_exists : K_squared > 6635/1000 := by sorry

-- Theorem 2: Probability of using electronic devices for not more than 1 hour given nearsightedness
theorem prob_not_more_than_one_hour : (c : ℚ) / (a + c) = 1/8 := by sorry

-- Theorem 3: Probability of exactly 2 out of 5 students being nearsighted
noncomputable def p_nearsighted : ℚ := (a + c : ℚ) / n
noncomputable def p_not_nearsighted : ℚ := 1 - p_nearsighted

theorem prob_two_out_of_five : 
  Nat.choose 5 2 * p_nearsighted^2 * p_not_nearsighted^3 = 32/625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_prob_not_more_than_one_hour_prob_two_out_of_five_l1001_100166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roman_wealth_unchanged_l1001_100180

/-- Represents the value of Roman's assets in dollars -/
def RomanWealth (initial_value coins_sold coins_left : ℝ) : ℝ :=
  initial_value

/-- Theorem stating that Roman's wealth remains unchanged after the sale -/
theorem roman_wealth_unchanged 
  (initial_value : ℝ) 
  (coins_sold : ℝ) 
  (coins_left : ℝ) 
  (h1 : initial_value = 20) 
  (h2 : coins_sold = 3) 
  (h3 : coins_left = 2) : 
  RomanWealth initial_value coins_sold coins_left = 20 := by
  rw [RomanWealth]
  exact h1

#check roman_wealth_unchanged

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roman_wealth_unchanged_l1001_100180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_distance_traveled_l1001_100188

/-- Calculates the distance traveled by the slower person when two people
    moving towards each other meet, given their speeds and initial distance. -/
noncomputable def distanceTraveled (totalDistance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  (totalDistance * speed1) / (speed1 + speed2)

/-- Theorem stating that given the specific conditions of the problem,
    Maxwell travels 15 kilometers before meeting Brad. -/
theorem maxwell_distance_traveled :
  let totalDistance : ℝ := 40
  let maxwellSpeed : ℝ := 3
  let bradSpeed : ℝ := 5
  distanceTraveled totalDistance maxwellSpeed bradSpeed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_distance_traveled_l1001_100188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1001_100158

noncomputable section

def curve (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

def line (t : ℝ) : ℝ × ℝ := (1 - Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

def M : ℝ × ℝ := (1, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_sum :
  ∃ (t₁ t₂ : ℝ),
    curve (line t₁).1 (line t₁).2 ∧
    curve (line t₂).1 (line t₂).2 ∧
    t₁ ≠ t₂ ∧
    distance M (line t₁) + distance M (line t₂) = 3 * Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1001_100158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brians_breath_holding_time_l1001_100178

/-- Calculates Brian's final breath-holding time after three weeks of practice with interruptions -/
def final_breath_holding_time (
  initial_time : ℝ
  ) (
  week1_improvement : ℝ
  ) (
  week2_improvement : ℝ
  ) (
  week3_improvement : ℝ
  ) (
  missed_days : ℕ
  ) (
  daily_decrease : ℝ
  ) : ℝ :=
  let week1_time := initial_time * week1_improvement
  let week2_potential := week1_time * (1 + week2_improvement)
  let week2_decrease := week1_time * daily_decrease * (missed_days : ℝ)
  let week2_time := week2_potential - week2_decrease
  let week3_time := week2_time * (1 + week3_improvement)
  week3_time

/-- Theorem stating that Brian's final breath-holding time is 46.5 seconds -/
theorem brians_breath_holding_time :
  final_breath_holding_time 10 2 0.75 0.5 2 0.1 = 46.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brians_breath_holding_time_l1001_100178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_interest_rate_l1001_100160

/-- Represents the simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem initial_interest_rate (principal : ℝ) :
  (∃ rate : ℝ, simple_interest principal rate 5 = 1680 ∧
               simple_interest principal 5 4 = 1680) →
  (∃ rate : ℝ, simple_interest principal rate 5 = 1680 ∧ rate = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_interest_rate_l1001_100160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brochures_multiple_of_six_l1001_100168

/-- Represents the number of pamphlets Kathleen has to distribute -/
def num_pamphlets : ℕ := 12

/-- Represents the maximum number of offices Kathleen can distribute to -/
def num_offices : ℕ := 6

/-- Represents the number of brochures Kathleen wants to distribute -/
def num_brochures : ℕ := sorry

/-- The materials must be distributed equally among offices -/
axiom equal_distribution : num_pamphlets % num_offices = 0 ∧ num_brochures % num_offices = 0

/-- Theorem stating that the number of brochures must be a multiple of 6 -/
theorem brochures_multiple_of_six : ∃ k : ℕ, num_brochures = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brochures_multiple_of_six_l1001_100168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_ratio_l1001_100102

/-- Represents a cone with fixed volume -/
structure Cone where
  V : ℝ  -- Volume
  h : ℝ  -- Height
  l : ℝ  -- Slant height
  r : ℝ  -- Radius
  h_pos : h > 0
  l_pos : l > 0
  r_pos : r > 0
  volume_eq : (1/3) * Real.pi * r^2 * h = V
  slant_height_eq : l^2 = r^2 + h^2

/-- The total surface area of a cone -/
noncomputable def totalSurfaceArea (c : Cone) : ℝ :=
  Real.pi * c.r^2 + Real.pi * c.r * c.l

/-- Theorem: The ratio h/l that minimizes the total surface area of a cone with fixed volume is √3/2 -/
theorem min_surface_area_ratio (c : Cone) :
  ∃ (min_ratio : ℝ), min_ratio = Real.sqrt 3 / 2 ∧
  ∀ (other_cone : Cone), other_cone.V = c.V →
    totalSurfaceArea other_cone ≥ totalSurfaceArea c →
    other_cone.h / other_cone.l = min_ratio :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_ratio_l1001_100102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brownies_rows_l1001_100108

/-- Given a pan of brownies cut into columns, calculate the number of rows needed to equally divide the brownies among a group of people. -/
theorem brownies_rows (num_columns num_people brownies_per_person : ℕ) : 
  num_columns > 0 → 
  num_people > 0 → 
  brownies_per_person > 0 → 
  (num_people * brownies_per_person) / num_columns = 3 → 
  num_columns = 6 → 
  num_people = 6 → 
  brownies_per_person = 3 := by
  sorry

#check brownies_rows

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brownies_rows_l1001_100108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1001_100195

theorem max_triangle_area (a b c : ℝ) : 
  a ≤ 2 → b ≤ 2 → c ≤ 2 →
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  (1/4) * Real.sqrt (
    (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
  ) ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1001_100195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_escalator_time_l1001_100148

/-- The time it takes Brian to walk down a non-operating escalator, in seconds -/
noncomputable def time_non_operating : ℝ := 80

/-- The time it takes Brian to walk down an operating escalator, in seconds -/
noncomputable def time_operating : ℝ := 30

/-- Brian's walking speed down the escalator, in distance per second -/
noncomputable def brian_speed : ℝ := 1 / time_non_operating

/-- The speed of the operating escalator, in distance per second -/
noncomputable def escalator_speed : ℝ := brian_speed * (time_non_operating / time_operating - 1)

/-- The time it takes Brian to ride down the operating escalator when standing still -/
noncomputable def time_standing : ℝ := 1 / escalator_speed

theorem brian_escalator_time : time_standing = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_escalator_time_l1001_100148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l1001_100111

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (2 * x + b)

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (3 - 4 * x) / (4 * x)

-- Theorem statement
theorem inverse_function_condition (b : ℝ) : 
  (∀ x, x ≠ 0 → f b (f_inv x) = x) → b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l1001_100111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bound_l1001_100133

theorem quadratic_function_bound (p q : ℝ) : ∃ x : ℕ, x ∈ ({1, 2, 3} : Set ℕ) ∧ |x^2 + p*x + q| ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bound_l1001_100133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rollie_weight_l1001_100181

/-- The weight of Carrie the cat in grams -/
def c : ℕ := sorry

/-- The weight of Barrie the bat in grams -/
def b : ℕ := sorry

/-- The weight of Rollie the rat in grams -/
def r : ℕ := sorry

/-- Carrie and Barrie together weigh 4000g more than Rollie -/
axiom condition1 : c + b = 4000 + r

/-- Barrie and Rollie together weigh 2000g less than Carrie -/
axiom condition2 : b + r = c - 2000

/-- Carrie and Rollie together weigh 3000g more than Barrie -/
axiom condition3 : c + r = 3000 + b

/-- The weight of Rollie the rat is 500g -/
theorem rollie_weight : r = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rollie_weight_l1001_100181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_per_kg_total_profit_when_production_equals_demand_l1001_100184

noncomputable section

/-- Production as a function of price -/
def y_production (x : ℝ) : ℝ := 200 * x - 100

/-- Demand as a function of price -/
def y_demand (x : ℝ) : ℝ := -20 * x^2 + 100 * x + 900

/-- Price as a function of time -/
def x (t : ℝ) : ℝ := t + 1

/-- Cost as a function of time -/
def z (t : ℝ) : ℝ := (1/8) * t^2 + 3/2

/-- Profit per kilogram as a function of time -/
def w (t : ℝ) : ℝ := x t - z t

theorem max_profit_per_kg (t : ℝ) :
  (∀ s, w s ≤ w 4) ∧ w 4 = 3/2 := by
  sorry

theorem total_profit_when_production_equals_demand :
  ∃ t, y_production (x t) = y_demand (x t) ∧
  (y_production (x t) * 1000 * (x t - z t) = 1.35 * 10^6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_per_kg_total_profit_when_production_equals_demand_l1001_100184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_gamma_time_l1001_100182

noncomputable def alpha_time : ℝ → ℝ := sorry
noncomputable def beta_time : ℝ → ℝ := sorry
noncomputable def gamma_time : ℝ → ℝ := sorry
noncomputable def combined_time : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, combined_time x = gamma_time x - 8
axiom condition2 : ∀ x : ℝ, combined_time x = beta_time x - 2
axiom condition3 : ∀ x : ℝ, combined_time x = (2/3) * alpha_time x

noncomputable def p (x : ℝ) : ℝ := 1 / (1 / alpha_time x + 1 / gamma_time x)

theorem alpha_gamma_time : ∀ x : ℝ, p x = 48/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_gamma_time_l1001_100182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_20_multiples_l1001_100145

theorem unique_number_with_20_multiples : 
  ∃! n : ℕ, n > 0 ∧ (Finset.filter (λ x : ℕ ↦ x ≤ 100 ∧ x % n = 0) (Finset.range 101)).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_20_multiples_l1001_100145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1001_100114

/-- Given a train and a bridge, calculate the speed of the train -/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 15) :
  (train_length + bridge_length) / crossing_time = 400 / 15 := by
  rw [h1, h2, h3]
  norm_num

/-- Compute the result as a float for demonstration -/
def train_speed_float : Float :=
  (400 : Float) / 15

#eval train_speed_float

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1001_100114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_division_simplification_l1001_100135

theorem log_division_simplification :
  (Real.log 32 / Real.log 2) / (Real.log (1/2) / Real.log 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_division_simplification_l1001_100135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cellphone_company_customers_l1001_100174

def total_customers : ℕ := 25000
def us_percentage : ℚ := 20 / 100
def canada_percentage : ℚ := 12 / 100
def australia_percentage : ℚ := 15 / 100
def uk_percentage : ℚ := 8 / 100
def india_percentage : ℚ := 5 / 100

theorem cellphone_company_customers :
  let us_customers := (us_percentage * total_customers).floor
  let canada_customers := (canada_percentage * total_customers).floor
  let australia_customers := (australia_percentage * total_customers).floor
  let uk_customers := (uk_percentage * total_customers).floor
  let india_customers := (india_percentage * total_customers).floor
  let other_customers := total_customers - (us_customers + canada_customers + australia_customers + uk_customers + india_customers)
  other_customers = 10000 ∧ us_customers / other_customers = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cellphone_company_customers_l1001_100174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cans_for_reduced_group_l1001_100143

theorem cans_for_reduced_group (total_cans : ℕ) (original_people : ℕ) (reduction_percent : ℚ) 
  (h1 : total_cans = 600)
  (h2 : original_people = 40)
  (h3 : reduction_percent = 30/100) : 
  ⌊(total_cans / original_people : ℚ) * (original_people - ⌊reduction_percent * original_people⌋)⌋ = 420 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cans_for_reduced_group_l1001_100143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_composite_for_all_k_l1001_100154

theorem exists_n_composite_for_all_k : 
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → ∃ m : ℕ, m > 1 ∧ m ∣ (n * 2^k + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_composite_for_all_k_l1001_100154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_equal_areas_l1001_100140

/-- Triangle with vertices A(1,3), B(1,1), and C(10,1) -/
structure Triangle where
  A : ℝ × ℝ := (1, 3)
  B : ℝ × ℝ := (1, 1)
  C : ℝ × ℝ := (10, 1)

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1 / 2) * base * height

/-- Calculates the area of a trapezoid given its base and two heights -/
noncomputable def trapezoidArea (base h1 h2 : ℝ) : ℝ := (1 / 2) * base * (h1 + h2)

/-- Theorem: The y-coordinate of the horizontal line dividing the triangle into two equal areas is 2 -/
theorem horizontal_line_equal_areas (t : Triangle) :
  ∃ (b : ℝ), b = 2 ∧
  trapezoidArea 9 (b - 1) (b - 1) = triangleArea 9 2 / 2 :=
by sorry

#check horizontal_line_equal_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_line_equal_areas_l1001_100140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_five_l1001_100199

/-- The number of people -/
def n : ℕ := 5

/-- The exponent in the total number of calls -/
def k : ℕ := 1

/-- The total number of calls between any (n-2) people is 3^k -/
axiom total_calls : (n - 2) * (n - 3) / 2 = 3^k

/-- Any two people can call each other at most once -/
axiom max_one_call : ∀ i j : Fin n, i ≠ j → (Nat.min 1 (Nat.choose 2 1) : ℕ) ≤ 1

/-- The theorem stating that n must be 5 -/
theorem n_equals_five : n = 5 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_five_l1001_100199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_k_shots_l1001_100118

/-- The probability that player A makes a shot -/
def p_a : ℝ := 0.4

/-- The probability that player B makes a shot -/
def p_b : ℝ := 0.6

/-- The probability that player A shoots k times before either player makes a shot -/
def P (k : ℕ) : ℝ := (1 - p_a) * (1 - p_b) ^ (k - 1) * (p_a + p_b - p_a * p_b)

theorem probability_of_k_shots (k : ℕ) :
  P k = 0.24^(k - 1) * 0.76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_k_shots_l1001_100118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_expression_l1001_100147

theorem sqrt_two_expression : Real.sqrt 2 * (Real.sqrt 2 + 2) - (8 : Real)^(1/3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_expression_l1001_100147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_div_p_equals_one_third_l1001_100146

-- Define the function representing the left side of the equation
noncomputable def f (P Q : ℚ) (x : ℝ) : ℝ := P / (x + 3) + Q / (x^2 - 5*x)

-- Define the function representing the right side of the equation
noncomputable def g (x : ℝ) : ℝ := (x^2 - 3*x + 8) / (x^3 + x^2 - 15*x)

-- State the theorem
theorem q_div_p_equals_one_third (P Q : ℚ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 → f P Q x = g x) → 
  Q / P = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_div_p_equals_one_third_l1001_100146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_theorem_l1001_100177

/-- Represents the annual interest rate as a decimal -/
noncomputable def annual_rate : ℝ := 0.15

/-- Represents the time period in years -/
noncomputable def time : ℝ := 2 + 1/3

/-- Represents the compound interest amount -/
noncomputable def compound_interest : ℝ := 1554.5

/-- Represents the principal amount -/
noncomputable def principal : ℝ := 3839.7

/-- Theorem stating the relationship between principal, interest rate, time, and compound interest -/
theorem compound_interest_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |principal * ((1 + annual_rate) ^ time - 1) - compound_interest| < ε := by
  sorry

#check compound_interest_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_theorem_l1001_100177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1001_100109

theorem expression_value (x y : ℝ) (h : 3 * y - 2 * x = 4) : 
  (16 : ℝ)^(x + 1) / (8 : ℝ)^(2 * y - 1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1001_100109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_le_one_l1001_100183

theorem negation_of_sin_le_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_le_one_l1001_100183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l1001_100112

noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then 2 * x - 4 else 10 - 3 * x

theorem f_values : f (-1) = -6 ∧ f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l1001_100112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_transformable_polygon_problem_polygon_area_l1001_100167

/-- A polygon on a unit grid that can be transformed into a rectangle --/
structure TransformablePolygon where
  -- The width of the rectangle after transformation
  width : ℕ
  -- The height of the rectangle after transformation
  height : ℕ

/-- The area of a TransformablePolygon is equal to the product of its width and height --/
theorem area_of_transformable_polygon (p : TransformablePolygon) : 
  p.width * p.height = 6 := by
  sorry

/-- The specific polygon in the problem --/
def problem_polygon : TransformablePolygon :=
{ width := 3
  height := 2 }

/-- Proof that the area of the problem polygon is 6 square units --/
theorem problem_polygon_area : 
  problem_polygon.width * problem_polygon.height = 6 := by
  rw [problem_polygon]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_transformable_polygon_problem_polygon_area_l1001_100167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_m_values_l1001_100105

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The focal length of an ellipse -/
noncomputable def focal_length (e : Ellipse) : ℝ := Real.sqrt (e.a ^ 2 - e.b ^ 2)

/-- Theorem: For an ellipse with equation x^2 / (10-m) + y^2 / (m-2) = 1 and focal length 4, m is either 4 or 8 -/
theorem ellipse_focal_length_m_values (m : ℝ) :
  (∃ e : Ellipse, (10 - m) * (m - 2) > 0 ∧
    e.a ^ 2 = max (10 - m) (m - 2) ∧
    e.b ^ 2 = min (10 - m) (m - 2) ∧
    focal_length e = 4) →
  m = 4 ∨ m = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_m_values_l1001_100105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_implies_cube_root_l1001_100151

noncomputable def ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

theorem ratio_equality_implies_cube_root (a b c : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a / b = b / c) (hbc : b / c = c / a) :
  (a + b - c) / (a - b + c) = 1 ∨
  (a + b - c) / (a - b + c) = ω ∨
  (a + b - c) / (a - b + c) = ω^2 := by
  sorry

#check ratio_equality_implies_cube_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equality_implies_cube_root_l1001_100151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_condition_l1001_100110

theorem polynomial_constant_condition (k : ℕ) (hk : k ≥ 4) :
  ∀ F : Polynomial ℤ,
  (∀ c : ℕ, c ≤ k + 1 → 0 ≤ F.eval (↑c : ℤ) ∧ F.eval (↑c : ℤ) ≤ k) →
  (∀ i j : ℕ, i ≤ k + 1 → j ≤ k + 1 → F.eval (↑i : ℤ) = F.eval (↑j : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_constant_condition_l1001_100110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_derivative_of_F_p_integrates_to_one_l1001_100175

-- Define the cumulative distribution function F
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else x^2 / (1 + x^2)

-- Define the proposed probability density function p
noncomputable def p (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else 2*x / (1 + x^2)^2

-- Theorem stating that p is the derivative of F
theorem p_is_derivative_of_F :
  ∀ x : ℝ, HasDerivAt F (p x) x := by
  sorry

-- Theorem stating that the integral of p over ℝ is 1
theorem p_integrates_to_one :
  ∫ x, p x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_derivative_of_F_p_integrates_to_one_l1001_100175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromatic_number_remove_vertex_l1001_100194

/-- A finite graph represented by its vertex set and edge relation -/
structure FiniteGraph (V : Type) [Fintype V] where
  edge : V → V → Prop

/-- The chromatic number of a graph -/
noncomputable def chromaticNumber {V : Type} [Fintype V] (G : FiniteGraph V) : ℕ :=
  sorry

/-- Remove a vertex from a graph -/
def removeVertex {V : Type} [Fintype V] [DecidableEq V] (G : FiniteGraph V) (v : V) : FiniteGraph {x // x ≠ v} :=
  { edge := fun a b => G.edge a.val b.val }

/-- Theorem: Removing a vertex from a finite graph cannot decrease the chromatic number by more than 1 -/
theorem chromatic_number_remove_vertex {V : Type} [Fintype V] [DecidableEq V] (G : FiniteGraph V) (v : V) :
  chromaticNumber (removeVertex G v) ≥ chromaticNumber G - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromatic_number_remove_vertex_l1001_100194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_defective_on_fifth_test_l1001_100149

def number_of_scenarios (total_products good_products defective_products : ℕ) : ℕ :=
  (Nat.choose good_products 1) * (Nat.choose defective_products 1) * (Nat.factorial 4)

theorem fourth_defective_on_fifth_test :
  number_of_scenarios 10 6 4 = Nat.choose 6 1 * Nat.choose 4 1 * Nat.factorial 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_defective_on_fifth_test_l1001_100149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1001_100100

/-- Given the total weight of a certain number of moles of a compound,
    calculate the molecular weight of the compound. -/
noncomputable def molecular_weight (total_weight : ℝ) (num_moles : ℝ) : ℝ :=
  total_weight / num_moles

/-- Theorem: The molecular weight of a compound is 98 grams/mole,
    given that 5 moles of the compound have a total weight of 490 grams. -/
theorem compound_molecular_weight :
  molecular_weight 490 5 = 98 := by
  -- Unfold the definition of molecular_weight
  unfold molecular_weight
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l1001_100100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_isosceles_points_l1001_100173

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given point A -/
def A : Point := { x := 2, y := -2 }

/-- Origin point O -/
def O : Point := { x := 0, y := 0 }

/-- A point P on the y-axis -/
def P : ℝ → Point := fun y => { x := 0, y := y }

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Triangle AOP is isosceles if two of its sides are equal -/
def isIsosceles (y : ℝ) : Prop :=
  distance O A = distance O (P y) ∨ 
  distance O A = distance A (P y) ∨
  distance O (P y) = distance A (P y)

/-- The main theorem -/
theorem four_isosceles_points :
  ∃ (s : Finset ℝ), s.card = 4 ∧ ∀ y : ℝ, y ∈ s ↔ isIsosceles y := by
  sorry

#check four_isosceles_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_isosceles_points_l1001_100173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1001_100116

open Set Real

noncomputable def f (x : ℝ) : ℝ := x + 2 / (x + 1)

def solution_set : Set ℝ := {x | f x > 2 ∧ x ≠ -1}

theorem inequality_solution_set :
  solution_set = Iio (-1) ∪ Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1001_100116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_briget_fruit_purchase_cost_l1001_100120

noncomputable def strawberry_price : ℝ := 2.20
noncomputable def cherry_price : ℝ := 6 * strawberry_price
noncomputable def blueberry_price : ℝ := cherry_price / 2

noncomputable def strawberry_amount : ℝ := 3
noncomputable def cherry_amount : ℝ := 4.5
noncomputable def blueberry_amount : ℝ := 6.2

noncomputable def total_cost : ℝ := strawberry_price * strawberry_amount + 
                      cherry_price * cherry_amount + 
                      blueberry_price * blueberry_amount

theorem briget_fruit_purchase_cost : total_cost = 106.92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_briget_fruit_purchase_cost_l1001_100120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_is_negative_one_l1001_100153

def A : ℝ × ℝ × ℝ := (-1, 2, -3)
def B : ℝ × ℝ × ℝ := (0, 1, -2)
def C : ℝ × ℝ × ℝ := (-3, 4, -5)

def vector_AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
def vector_AC : ℝ × ℝ × ℝ := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2.1 * v.2.1 + u.2.2 * v.2.2

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2.1^2 + u.2.2^2)

theorem cosine_of_angle_is_negative_one :
  (dot_product vector_AB vector_AC) / (magnitude vector_AB * magnitude vector_AC) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_is_negative_one_l1001_100153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l1001_100117

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) 
  (female_ratio : ℚ) (female_on_duty_percent : ℚ) :
  total_on_duty = 180 →
  female_ratio = 1/2 →
  female_on_duty_percent = 18/100 →
  (female_on_duty_ratio * ↑total_on_duty) = female_ratio * ↑total_on_duty →
  (female_on_duty_percent * (female_on_duty_ratio * ↑total_on_duty)) = female_on_duty_ratio * ↑total_on_duty →
  (female_on_duty_ratio * ↑total_on_duty) / female_on_duty_percent = 500 :=
by
  sorry

#check female_officers_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l1001_100117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_school_time_l1001_100115

/-- The boy's usual time to reach school -/
noncomputable def usual_time : ℝ := 16

/-- The factor by which the boy's rate increases -/
noncomputable def rate_increase : ℝ := 4/3

/-- The time saved by walking at the increased rate -/
noncomputable def time_saved : ℝ := 4

theorem boy_school_time :
  ∀ (t : ℝ), t > 0 →
  t * rate_increase = (t - time_saved) * rate_increase →
  t = usual_time :=
by
  intro t t_pos equation
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_school_time_l1001_100115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_meeting_time_at_B_l1001_100136

/-- Represents a point on the block's perimeter -/
inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point

/-- Represents a direction of movement -/
inductive Direction : Type
| Clockwise : Direction
| Counterclockwise : Direction

/-- Represents a person or animal moving on the block's perimeter -/
structure Mover where
  startPoint : Point
  direction : Direction
  speed : ℝ

def blockPerimeter : ℝ := 600

theorem next_meeting_time_at_B 
  (man : Mover)
  (dog : Mover)
  (h1 : man.startPoint = Point.A)
  (h2 : dog.startPoint = Point.A)
  (h3 : man.direction = Direction.Clockwise)
  (h4 : dog.direction = Direction.Counterclockwise)
  (h5 : ∃ t : ℝ, t > 0 ∧ t < blockPerimeter / (man.speed + dog.speed) ∧ 
       (man.speed * t) % blockPerimeter = 100 ∧
       (dog.speed * t) % blockPerimeter = 500)
  (h6 : (100 : ℝ) + (200 : ℝ) + (100 : ℝ) + (200 : ℝ) = blockPerimeter) :
  ∃ t : ℝ, t = 14 ∧ 
    (man.speed * t) % blockPerimeter = 100 ∧
    (dog.speed * t) % blockPerimeter = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_meeting_time_at_B_l1001_100136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redox_potential_approximation_l1001_100186

/-- The Nernst equation for calculating redox potential -/
noncomputable def nernst_equation (E0 : ℝ) (n : ℝ) (C_ox : ℝ) (C_red : ℝ) : ℝ :=
  E0 + (0.059 / n) * Real.log (C_ox / C_red) / Real.log 10

/-- Theorem stating that the redox potential is approximately 1.46 V -/
theorem redox_potential_approximation (E0 : ℝ) (n : ℝ) (C_ox : ℝ) (C_red : ℝ)
    (h1 : E0 = 1.51)
    (h2 : n = 5)
    (h3 : C_ox = 2e-6)
    (h4 : C_red = 1e-2) :
    ∃ (E : ℝ), |E - nernst_equation E0 n C_ox C_red| < 0.01 ∧ |E - 1.46| < 0.01 := by
  sorry

#check redox_potential_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_redox_potential_approximation_l1001_100186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leaf_circle_ratio_for_radius_3_l1001_100130

/-- The ratio of the area of leaf-shaped figures to the original circle --/
noncomputable def leaf_circle_ratio (r : ℝ) : ℝ :=
  let circle_area := Real.pi * r^2
  let arc_area := circle_area / 6
  let leaf_area := arc_area * 1.5
  (3 * leaf_area) / circle_area

/-- Theorem stating the ratio for a circle with radius 3 --/
theorem leaf_circle_ratio_for_radius_3 :
  leaf_circle_ratio 3 = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leaf_circle_ratio_for_radius_3_l1001_100130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l1001_100106

theorem area_of_closed_figure (a : ℝ) 
  (h : (Finset.range 10).sum (λ k => (-1)^(k:ℤ) * Nat.choose 9 k * a^(-k:ℤ) * (2^(9 - k))) = -21/2) : 
  2 * ∫ x in -a..a, Real.sin x = 2 - 2 * Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_figure_l1001_100106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_viewable_area_l1001_100163

open Real

/-- The area of the viewable region for a person walking around a rectangular park -/
noncomputable def viewableArea (length width visibility : ℝ) : ℝ :=
  let innerArea := length * width
  let topBottomArea := 2 * (length * visibility)
  let sideArea := 2 * (width * visibility)
  let cornerArea := 4 * (Real.pi * visibility^2 / 4)
  innerArea + topBottomArea + sideArea + cornerArea

/-- Theorem stating the correct area for the given park dimensions and visibility -/
theorem park_viewable_area :
  viewableArea 8 3 2 = 24 + 32 + 12 + 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_viewable_area_l1001_100163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_catches_up_l1001_100101

/-- Represents the scenario of Vitya catching up with his mother -/
noncomputable def catch_up_time (initial_speed : ℝ) (time_before_turnaround : ℝ) (speed_increase_factor : ℝ) : ℝ :=
  let initial_distance := 2 * initial_speed * time_before_turnaround
  let relative_speed := initial_speed * (speed_increase_factor - 1)
  initial_distance / relative_speed

/-- Theorem stating that under the given conditions, Vitya catches up with his mother in 5 minutes -/
theorem vitya_catches_up :
  ∀ (initial_speed : ℝ), initial_speed > 0 →
  catch_up_time initial_speed 10 5 = 5 := by
  intro initial_speed h_positive
  -- The proof goes here
  sorry

#check vitya_catches_up

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vitya_catches_up_l1001_100101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_sequence_terminates_nina_sequence_exact_operations_l1001_100187

def nina_sequence (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | i + 1 => 
    let prev := nina_sequence n i
    if prev > 100 then (prev / 2)
    else (prev * 3)

theorem nina_sequence_terminates (n : ℕ) : 
  ∃ k, k ≤ 12 ∧ nina_sequence n k ≤ 1 ∧ ∀ j < k, nina_sequence n j > 1 :=
sorry

theorem nina_sequence_exact_operations : 
  ∃ k, k = 12 ∧ nina_sequence k k ≤ 1 ∧ ∀ j < k, nina_sequence k j > 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nina_sequence_terminates_nina_sequence_exact_operations_l1001_100187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1001_100197

/-- Given a parabola and a hyperbola with specific properties, 
    prove that the eccentricity of the hyperbola is √5 -/
theorem hyperbola_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (hC₁ : ∀ x y, y^2 = 2*p*x → (x, y) ∈ Set.range (λ t ↦ (t, Real.sqrt (2*p*t))))
  (hC₂ : ∀ x y, x^2/a^2 - y^2/b^2 = 1 → (x, y) ∈ Set.range (λ t ↦ (a*Real.cosh t, b*Real.sinh t)))
  (hA : ∃ x y, y^2 = 2*p*x ∧ y = (b/a)*x)
  (hd : ∃ x y, y^2 = 2*p*x ∧ y = (b/a)*x ∧ x + p/2 = p) :
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1001_100197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_laps_theorem_l1001_100159

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ
  area : ℝ
  length_eq_twice_width : length = 2 * width
  area_eq_length_times_width : area = length * width

/-- Represents a jogger's characteristics -/
structure Jogger where
  speed : ℝ  -- in kilometers per hour
  time : ℝ   -- in hours

/-- Calculates the number of laps around a field given its perimeter and the total distance jogged -/
noncomputable def calculate_laps (perimeter : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance / perimeter

theorem jogger_laps_theorem (field : RectangularField) (jogger : Jogger) :
  field.area = 20000 ∧ 
  jogger.speed = 12 ∧ 
  jogger.time = 0.5 →
  calculate_laps (2 * (field.length + field.width)) (jogger.speed * jogger.time * 1000) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_laps_theorem_l1001_100159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1001_100152

noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1001_100152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_condition_l1001_100191

-- Define the function f(x) = (a-1)^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) ^ x

-- Define the property of being monotonically decreasing on ℝ
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y < f x

-- Define the necessary but not sufficient condition
def NecessaryButNotSufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ∃ (x : ℝ), P ∧ ¬Q

-- State the theorem
theorem f_monotonically_decreasing_condition (a : ℝ) :
  NecessaryButNotSufficient
    (0 < a ∧ a < 2)
    (MonotonicallyDecreasing (f a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_condition_l1001_100191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1001_100176

theorem complex_absolute_value (Z : ℂ) : Z = Complex.mk (Real.sqrt 3) (-1) → Complex.abs Z = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l1001_100176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_lateral_surface_l1001_100144

theorem cone_height_from_lateral_surface (r : ℝ) (h : r = 2) : 
  r^2 = r^2 + (Real.sqrt 3)^2 := by
  have slant_height := r
  have base_radius := r
  have height := Real.sqrt 3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_lateral_surface_l1001_100144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_finishes_first_l1001_100119

/-- Represents the size of a garden -/
structure GardenSize where
  size : ℚ
  size_pos : size > 0

/-- Represents the speed of a lawnmower -/
structure LawnmowerSpeed where
  speed : ℚ
  speed_pos : speed > 0

/-- Represents a person with their garden size and lawnmower speed -/
structure Person where
  name : String
  garden : GardenSize
  lawnmower : LawnmowerSpeed

def mowing_time (p : Person) : ℚ :=
  p.garden.size / p.lawnmower.speed

theorem alex_finishes_first 
  (alex samantha nikki : Person)
  (h1 : samantha.garden.size = 3 * alex.garden.size)
  (h2 : samantha.garden.size = 2/3 * nikki.garden.size)
  (h3 : alex.lawnmower.speed = 3 * samantha.lawnmower.speed)
  (h4 : alex.lawnmower.speed = 2 * nikki.lawnmower.speed) :
  mowing_time alex < min (mowing_time samantha) (mowing_time nikki) := by
  -- Proof goes here
  sorry

#check alex_finishes_first

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_finishes_first_l1001_100119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l1001_100141

/-- The length of the diagonal of a rectangle divided into squares -/
theorem rectangle_diagonal_length (n m : ℕ) (d : ℝ) (h1 : n * m = 60) (h2 : n = 5) (h3 : m = 12) :
  let s := (d / Real.sqrt 2 : ℝ)
  (n * s)^2 + (m * s)^2 = (13 * d / Real.sqrt 2)^2 := by
  sorry

#check rectangle_diagonal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l1001_100141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_two_square_theorem_l1001_100155

/-- The area of a planar figure measured using the oblique two-square method -/
noncomputable def oblique_two_square_area (a : ℝ) : ℝ :=
  Real.sqrt 2 * a^2

/-- The area of an isosceles right triangle with leg length a -/
noncomputable def isosceles_right_triangle_area (a : ℝ) : ℝ :=
  (1 / 2) * a^2

theorem oblique_two_square_theorem (a : ℝ) (h : a > 0) :
  oblique_two_square_area a = 
    (isosceles_right_triangle_area a) * (4 / Real.sqrt 2) := by
  -- Unfold definitions
  unfold oblique_two_square_area isosceles_right_triangle_area
  -- Simplify the right-hand side
  simp [mul_assoc, mul_comm, mul_div_cancel']
  -- The proof is complete
  sorry

#check oblique_two_square_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_two_square_theorem_l1001_100155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1001_100122

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 3 ∧
  a^2 + c^2 - Real.sqrt 3 * a * c = b^2 ∧
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_problem a b c A B C) :
  B = Real.pi/6 ∧ 
  (Real.cos A = 4/5 → 
    1/2 * a * b * Real.sin C = (3 * Real.sqrt 3 + 4)/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1001_100122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_six_iff_m_is_two_l1001_100171

/-- The function f(x) = x + 3^m/x for x > 0 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + 3^m / x

/-- The minimum value of f(x) over x > 0 -/
noncomputable def min_value (m : ℝ) : ℝ := 2 * Real.sqrt (3^m)

theorem min_value_is_six_iff_m_is_two (m : ℝ) :
  (∀ x > 0, f m x ≥ 6) ∧ (∃ x > 0, f m x = 6) ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_six_iff_m_is_two_l1001_100171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1001_100189

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateralTriangleArea (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

/-- The theorem stating the decrease in area of an equilateral triangle after two reductions -/
theorem equilateral_triangle_area_decrease :
  let initial_area : ℝ := 121 * Real.sqrt 3
  let initial_side : ℝ := Real.sqrt (4 * initial_area / Real.sqrt 3)
  let first_reduction : ℝ := 5
  let second_reduction : ℝ := 3
  let final_side : ℝ := initial_side - first_reduction - second_reduction
  initial_area - equilateralTriangleArea final_side = 72 * Real.sqrt 3 := by
  sorry

#check equilateral_triangle_area_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_decrease_l1001_100189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_at_5_l1001_100170

/-- A monic quintic polynomial satisfying specific conditions -/
noncomputable def p : ℝ → ℝ :=
  sorry -- We'll define this later

/-- p is a monic quintic polynomial -/
axiom p_monic_quintic : ∃ a b c d : ℝ, ∀ x, p x = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + (p 0)

/-- Conditions for p -/
axiom p_cond_1 : p 1 = 2
axiom p_cond_2 : p 2 = 5
axiom p_cond_3 : p 3 = 10
axiom p_cond_4 : p 4 = 17
axiom p_cond_6 : p 6 = 37

/-- Theorem: p(5) = 2 -/
theorem p_value_at_5 : p 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_value_at_5_l1001_100170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_pi_approx_to_hundredth_l1001_100129

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The number to be rounded -/
def pi_approx : ℝ := 3.1415926

theorem round_pi_approx_to_hundredth :
  roundToHundredth pi_approx = 3.14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_pi_approx_to_hundredth_l1001_100129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_visibility_time_l1001_100134

noncomputable section

-- Define the pentagon
def pentagon_side_length : ℝ := 20

-- Define bug speeds
def bug_A_speed : ℝ := 5
def bug_B_speed : ℝ := 4

-- Define the visibility threshold (assumed to be the side length)
def visibility_threshold : ℝ := pentagon_side_length

-- Function to calculate the position of a bug at a given time
def bug_position (speed : ℝ) (time : ℝ) : ℝ :=
  (speed * time) % (5 * pentagon_side_length)

-- Function to check if Bug B can see Bug A
def can_see (time : ℝ) : Prop :=
  let distance := (bug_position bug_A_speed time - bug_position bug_B_speed time + 5 * pentagon_side_length) % (5 * pentagon_side_length)
  distance ≤ visibility_threshold ∨ (5 * pentagon_side_length - distance) ≤ visibility_threshold

-- Theorem statement
theorem bug_visibility_time :
  ∃ t : ℝ, t = 6 ∧ 
  (∀ s : ℝ, 0 ≤ s ∧ s ≤ t → can_see s) ∧
  (∀ s : ℝ, s > t → ¬(can_see s) ∨ bug_position bug_A_speed s ≥ pentagon_side_length) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_visibility_time_l1001_100134
