import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l970_97065

/-- The curve defined by the polar equation r = 2 sin θ sec θ is a parabola -/
theorem polar_to_cartesian_parabola :
  ∀ (r θ x y : ℝ),
  (r = 2 * Real.sin θ * (1 / Real.cos θ)) →
  (r = Real.sqrt (x^2 + y^2)) →
  (Real.sin θ = y / Real.sqrt (x^2 + y^2)) →
  (Real.cos θ = x / Real.sqrt (x^2 + y^2)) →
  (x^2 = 2 * y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_cartesian_parabola_l970_97065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_domain_correct_f_odd_or_even_existence_f_odd_or_even_value_l970_97092

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((x + 2 * a + 1) / (x - 3 * a + 1))

def domain (a : ℝ) : Set ℝ :=
  if a > 0 then
    {x | x < -2 * a - 1 ∨ x > 3 * a - 1}
  else
    {x | x < 3 * a - 1 ∨ x > -2 * a - 1}

theorem f_satisfies_condition (a : ℝ) (x : ℝ) (h : a ≠ 0) :
  f a (a * x - 1) = Real.log ((x + 2) / (x - 3)) := by sorry

theorem domain_correct (a : ℝ) (x : ℝ) :
  x ∈ domain a ↔ f a x ≠ 0 := by sorry

theorem f_odd_or_even_existence :
  ∃ a : ℝ, (∀ x, f a x = f a (-x)) ∨ (∀ x, f a x = -f a (-x)) := by sorry

theorem f_odd_or_even_value :
  ∃! a : ℝ, (∀ x, f a x = f a (-x)) ∨ (∀ x, f a x = -f a (-x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_domain_correct_f_odd_or_even_existence_f_odd_or_even_value_l970_97092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_multiple_of_four_l970_97093

theorem probability_at_least_one_multiple_of_four : ℝ := by
  -- Define the range of integers
  let range : Finset ℕ := Finset.range 60

  -- Define the set of multiples of 4 within the range
  let multiples_of_four : Finset ℕ := range.filter (fun n => n % 4 = 0 ∧ n > 0)

  -- Define the probability of choosing a multiple of 4
  let prob_multiple_of_four : ℚ := (multiples_of_four.card : ℚ) / (range.card : ℚ)

  -- Define the probability of not choosing a multiple of 4
  let prob_not_multiple_of_four : ℚ := 1 - prob_multiple_of_four

  -- The probability of choosing at least one multiple of 4 in two attempts
  let prob_at_least_one_multiple_of_four : ℚ := 1 - prob_not_multiple_of_four ^ 2

  -- Convert the result to ℝ
  have h : prob_at_least_one_multiple_of_four = 7/16 := by sorry

  -- Final step: convert ℚ to ℝ
  exact (7/16 : ℝ)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_multiple_of_four_l970_97093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smart_integers_divisible_by_25_fraction_l970_97005

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_smart_integer (n : ℕ) : Prop :=
  Even n ∧ 
  n > 20 ∧ 
  n < 120 ∧ 
  (sum_of_digits n = 10)

def count_smart_integers : ℕ := sorry

def count_smart_integers_divisible_by_25 : ℕ := sorry

theorem smart_integers_divisible_by_25_fraction :
  (count_smart_integers_divisible_by_25 : ℚ) / count_smart_integers = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smart_integers_divisible_by_25_fraction_l970_97005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picasso_postcards_consecutive_probability_l970_97024

def total_postcards : ℕ := 12
def picasso_postcards : ℕ := 4

theorem picasso_postcards_consecutive_probability :
  (Nat.factorial (total_postcards - picasso_postcards + 1) * Nat.factorial picasso_postcards) / 
  (Nat.factorial total_postcards : ℚ) = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picasso_postcards_consecutive_probability_l970_97024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_outside_circles_area_l970_97074

/-- The area of the region inside a rectangle but outside three quarter circles -/
theorem rectangle_outside_circles_area :
  let rectangle_area : ℝ := 6 * 4
  let circle_a_area : ℝ := Real.pi * 2^2
  let circle_b_area : ℝ := Real.pi * 3^2
  let circle_c_area : ℝ := Real.pi * 4^2
  let quarter_circles_area : ℝ := (circle_a_area + circle_b_area + circle_c_area) / 4
  let outside_area : ℝ := rectangle_area - quarter_circles_area
  ∃ ε > 0, |outside_area - 1.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_outside_circles_area_l970_97074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l970_97000

theorem expression_simplification (y : ℝ) (h : y ≠ 0) :
  5 / (2 * y^(-4 : ℤ)) * (4 * y^3 / 3) + y / y^(-2 : ℤ) = (10 * y^7 + 3 * y^3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l970_97000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f4_decreasing_l970_97007

-- Define the four functions
noncomputable def f1 (x : ℝ) : ℝ := 2 / x
noncomputable def f2 (x : ℝ) : ℝ := -2 / x
def f3 (x : ℝ) : ℝ := 2 * x
def f4 (x : ℝ) : ℝ := -2 * x

-- Define a property that checks if a function decreases as x increases
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x2 < f x1

-- Theorem stating that only f4 (y=-2x) is decreasing
theorem only_f4_decreasing :
  is_decreasing f4 ∧ 
  ¬(is_decreasing f1) ∧ 
  ¬(is_decreasing f2) ∧ 
  ¬(is_decreasing f3) := by
  sorry

#check only_f4_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f4_decreasing_l970_97007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l970_97050

-- Define the line equation
def line_equation (x y θ : ℝ) : Prop :=
  x * Real.cos θ + y - 1 = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (m : ℝ) : ℝ :=
  Real.arctan m

-- Theorem statement
theorem inclination_angle_range :
  ∀ θ : ℝ, ∃ α : ℝ,
    (∃ x y : ℝ, line_equation x y θ) →
    inclination_angle (-Real.cos θ) = α ∧
    ((0 ≤ α ∧ α ≤ Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l970_97050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l970_97039

theorem cos_minus_sin_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : π/4 < α)
  (h3 : α < π/2) :
  Real.cos α - Real.sin α = -Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l970_97039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l970_97041

theorem triangle_area_proof (n : ℕ) (h_n : n > 0) : 
  let a : ℕ → ℚ := λ k => 1 / (k * (k + 1))
  let S : ℕ → ℚ := λ m => (Finset.range m).sum (λ k => a (k + 1))
  S n = 9/10 →
  let line := λ x y => x / (n + 1) + y / n = 1
  let area := (n + 1) * n / 2
  area = 45 := by
    intro h
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_proof_l970_97041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_dms_3_76_multiply_degrees_add_dms_l970_97091

-- Problem 1
noncomputable def decimal_to_dms (d : ℝ) : ℕ × ℕ × ℕ :=
  let degrees := Int.floor d
  let minutes := Int.floor ((d - degrees) * 60)
  let seconds := Int.floor (((d - degrees) * 60 - minutes) * 60)
  (degrees.toNat, minutes.toNat, seconds.toNat)

theorem decimal_to_dms_3_76 :
  decimal_to_dms 3.76 = (3, 45, 36) := by sorry

-- Problem 2
theorem multiply_degrees :
  0.5 * 5 = 2.5 := by sorry

-- Problem 3
def dms_to_seconds (d : ℕ) (m : ℕ) (s : ℕ) : ℕ :=
  d * 3600 + m * 60 + s

def seconds_to_dms (s : ℕ) : ℕ × ℕ × ℕ :=
  let d := s / 3600
  let m := (s % 3600) / 60
  let sec := s % 60
  (d, m, sec)

theorem add_dms :
  seconds_to_dms (dms_to_seconds 15 48 36 + dms_to_seconds 37 27 59) = (53, 16, 35) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_dms_3_76_multiply_degrees_add_dms_l970_97091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_p_l970_97011

-- Define the variables as natural numbers (digits are positive integers)
variable (a b k m p r : ℕ+)

-- Define the conditions
def condition1 (a b k : ℕ+) : Prop := a + b = k
def condition2 (k m p : ℕ+) : Prop := k + m = p
def condition3 (p a r : ℕ+) : Prop := p + a = r
def condition4 (b m r : ℕ+) : Prop := b + m + r = 18

-- Theorem statement
theorem value_of_p 
  (h1 : condition1 a b k) 
  (h2 : condition2 k m p) 
  (h3 : condition3 p a r) 
  (h4 : condition4 b m r) : p = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_p_l970_97011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_radius_l970_97004

/-- The equation of the circle representing the cake's boundary -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 1 = 2*x + 5*y

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := 5/2

/-- Theorem stating that the radius of the circle described by the equation is 5/2 -/
theorem cake_radius :
  ∃ (h k r : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ r = circle_radius :=
by
  -- We'll use h = 1, k = 5/2 as the center, and r = 5/2 as the radius
  use 1, 5/2, 5/2
  constructor
  · intro x y
    -- The proof of the equivalence goes here
    sorry
  · -- Proof that r = circle_radius
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_radius_l970_97004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_is_six_l970_97012

-- Define the curves and ray
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0
def C₂ (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 4 * Real.cos θ
def l (x y : ℝ) : Prop := y = (3/4) * x ∧ x ≥ 0

-- Define the intersection points (noncomputable due to the use of Real.sqrt)
noncomputable def intersectionPoint (C : (ℝ → ℝ → Prop)) : ℝ × ℝ :=
  ⟨0, 0⟩ -- Placeholder value, actual computation would be more complex

-- Theorem statement
theorem intersection_product_is_six :
  let A := intersectionPoint C₁
  let B := intersectionPoint C₂
  (Real.sqrt ((A.1)^2 + (A.2)^2)) * (Real.sqrt ((B.1)^2 + (B.2)^2)) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_is_six_l970_97012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l970_97085

/-- The area of a rectangle with given vertices in a rectangular coordinate system -/
theorem rectangle_area (x1 y1 x2 y2 : ℝ) : 
  let vertices := [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
  (∀ (p : ℝ × ℝ), p ∈ vertices → (p.1 = x1 ∨ p.1 = x2) ∧ (p.2 = y1 ∨ p.2 = y2)) →
  (x1 ≠ x2 ∧ y1 ≠ y2) →
  (x2 - x1) * (y1 - y2) = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l970_97085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l970_97089

/-- An arithmetic sequence with m terms and common difference d -/
structure ArithmeticSequence where
  m : ℕ
  d : ℝ
  a₁ : ℝ
  h_m : m ≥ 3
  h_d : d > 0

/-- The n-th term of the arithmetic sequence -/
def ArithmeticSequence.a (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1 : ℝ) * seq.d

/-- The exponential function f(x) = e^(-x/d) -/
noncomputable def f (seq : ArithmeticSequence) (x : ℝ) : ℝ :=
  Real.exp (-x / seq.d)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (¬ ∃ r : ℝ, ∀ n : ℕ, n < seq.m → seq.a (n + 1) = r * seq.a n) ∧
  (∀ i : ℕ, i < seq.m - 1 →
    ∃ y : ℝ → ℝ, y = f seq ∧
    (deriv y (seq.a i) * (seq.a (i + 1) - seq.a i) + y (seq.a i) = 0)) ∧
  (∀ m : ℕ, m ≥ 3 →
    (∃ σ : Equiv ℕ ℕ, ∀ n : ℕ, n < m →
      ∃ r : ℝ, seq.a (σ (n + 1)) = r * seq.a (σ n)) ↔ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l970_97089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_arrival_time_theorem_l970_97068

/-- Represents a point in the travel path -/
structure Point where
  name : String

/-- Represents a person traveling -/
structure Person where
  name : String

/-- Represents the travel scenario -/
structure TravelScenario where
  start : Point
  finish : Point
  distance : ℝ
  personA : Person
  personB : Person
  walkingSpeed : ℝ
  alternativeSpeed : ℝ

/-- Helper function to calculate arrival time -/
noncomputable def arrival_time (p : Person) (start : Point) (finish : Point) (scenario : TravelScenario) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem simultaneous_arrival_time_theorem (scenario : TravelScenario)
    (h1 : scenario.distance = 15)
    (h2 : scenario.walkingSpeed = 1)
    (h3 : scenario.alternativeSpeed = 4) :
    ∃ (t : ℝ), t = 3/11 ∧ 
    (∀ (p : Person), p = scenario.personA ∨ p = scenario.personB →
      arrival_time p scenario.start scenario.finish scenario = t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_arrival_time_theorem_l970_97068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_two_zeros_range_f_squared_minus_af_one_integer_solution_l970_97057

/- Define the function f(x) = x / e^x for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

/- Theorem for the maximum value of f(x) -/
theorem f_max_value : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = 1 / Real.exp 1 := by
  sorry

/- Theorem for the range of m where g(x) = f(x) - m has two zeros -/
theorem g_two_zeros_range (m : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f x₁ = m ∧ f x₂ = m) ↔ 
  (0 < m ∧ m < 1 / Real.exp 1) := by
  sorry

/- Theorem for the range of a where f^2(x) - af(x) > 0 has only one integer solution -/
theorem f_squared_minus_af_one_integer_solution (a : ℝ) :
  (∃! (n : ℤ), (f (n : ℝ))^2 - a * f (n : ℝ) > 0) ↔ 
  (2 / Real.exp 2 ≤ a ∧ a < 1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_two_zeros_range_f_squared_minus_af_one_integer_solution_l970_97057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₂_to_C_l970_97025

/-- Curve C₂ -/
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, 2 * Real.sin θ)

/-- Line representing curve C -/
def C (x y : ℝ) : Prop := 2 * y + x = 10

/-- Distance from a point to the line C -/
noncomputable def distance_to_C (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  |2 * y + x - 10| / Real.sqrt 5

theorem min_distance_C₂_to_C :
  ∀ θ : ℝ, distance_to_C (C₂ θ) ≥ Real.sqrt 5 := by
  sorry

#check min_distance_C₂_to_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C₂_to_C_l970_97025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_sequence_l970_97071

noncomputable def a : ℕ → ℚ
| 0 => 1/3
| n+1 => 1 + (a n - 1)^2

theorem infinite_product_sequence :
  ∏' n, a n = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_sequence_l970_97071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_lower_bound_l970_97083

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- The x-coordinate of the focus of a hyperbola -/
noncomputable def focus_x (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_eccentricity_lower_bound (h : Hyperbola) 
  (hB : ∃ (x : ℝ), |x - focus_x h| > 2 * (h.a + focus_x h)) :
  eccentricity h > Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_lower_bound_l970_97083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l970_97075

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) (h_arith : arithmetic_sequence a d)
  (h_a3 : a 3 = 3) (h_s7 : sum_arithmetic_sequence a 7 = 14) :
  d = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l970_97075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_equals_closed_interval_l970_97088

open Set Real

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 + x - 2 > 0}

-- Define set N
def N : Set ℝ := {x : ℝ | (1/2 : ℝ)^(x-1) ≥ 2}

-- Theorem statement
theorem complement_M_intersect_N_equals_closed_interval :
  (U \ M) ∩ N = Icc (-2 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_equals_closed_interval_l970_97088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l970_97079

-- Define the function f
noncomputable def f (a : ℝ) : ℝ := ∫ x in (0:ℝ)..(1:ℝ), 2 * a * x^2 - a^2 * x

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), (∀ a, f a ≤ max) ∧ (∃ a₀, f a₀ = max) ∧ max = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l970_97079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_mark_is_60_l970_97084

/-- Represents a class with number of students and average mark -/
structure MyClass where
  students : Nat
  avgMark : Nat

/-- Calculates the total marks for a class -/
def totalMarks (c : MyClass) : Nat := c.students * c.avgMark

/-- The four classes in the school -/
def classA : MyClass := ⟨30, 40⟩
def classB : MyClass := ⟨50, 70⟩
def classC : MyClass := ⟨25, 55⟩
def classD : MyClass := ⟨45, 65⟩

/-- The list of all classes -/
def classes : List MyClass := [classA, classB, classC, classD]

/-- Theorem: The overall average mark for all students is 60 -/
theorem overall_average_mark_is_60 :
  (classes.map totalMarks).sum / (classes.map (·.students)).sum = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_mark_is_60_l970_97084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l970_97095

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1/a}
  else if a = 1 then ∅
  else if a > 1 then {x | 1/a < x ∧ x < 1}
  else Set.univ

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | f a x < 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l970_97095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_equation_l970_97029

theorem matrix_determinant_equation (x : ℝ) : 
  (Matrix.det !![3*x, 4; 2*x, x] : ℝ) = -2 ↔ 3*x^2 - 8*x + 2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_determinant_equation_l970_97029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sallys_credit_cards_theorem_l970_97073

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards and their transactions -/
def SallysCreditCards (G : ℝ) : Prop :=
  ∃ (gold platinum diamond : CreditCard),
    -- Initial card limits
    gold.limit = G ∧
    platinum.limit = 2 * G ∧
    diamond.limit = 3 * G ∧
    -- Initial card balances
    gold.balance = G / 3 ∧
    platinum.balance = G / 3 ∧
    diamond.balance = G / 3 ∧
    -- After first transfer (gold to platinum)
    let platinum_after_first := CreditCard.mk platinum.limit (platinum.balance + gold.balance);
    -- After second transfer (half of platinum to diamond)
    let platinum_final := CreditCard.mk platinum.limit (platinum_after_first.balance / 2);
    -- Portion of platinum limit unspent
    (platinum_final.limit - platinum_final.balance) / platinum_final.limit = 5 / 6

theorem sallys_credit_cards_theorem (G : ℝ) (h : G > 0) :
  SallysCreditCards G := by
  sorry

#check sallys_credit_cards_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sallys_credit_cards_theorem_l970_97073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_50_l970_97044

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * f y - 2 * y * f x = f (x / y)

/-- The main theorem stating that f(50) = 0 for any function satisfying the functional equation -/
theorem function_value_at_50 (f : ℝ → ℝ) (h : FunctionalEquation f) : f 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_50_l970_97044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l970_97033

/-- Represents the cost of a pack of pencils in cents -/
def pack_cost : ℕ := 20

/-- The number of seventh graders who bought pencils -/
def seventh_graders : ℕ := 25

/-- The number of sixth graders who bought pencils -/
def sixth_graders : ℕ := 18

/-- The total amount paid by seventh graders in cents -/
def seventh_total : ℕ := 275

/-- The total amount paid by sixth graders in cents -/
def sixth_total : ℕ := 216

/-- The number of pencils in each pack -/
def pencils_per_pack : ℕ := 2

theorem pencil_buyers_difference :
  (sixth_graders : Int) - (seventh_graders : Int) = -7 := by
  -- Convert natural numbers to integers for subtraction
  have h1 : (sixth_graders : Int) = 18 := rfl
  have h2 : (seventh_graders : Int) = 25 := rfl
  -- Perform the subtraction
  calc
    (sixth_graders : Int) - (seventh_graders : Int) = 18 - 25 := by rw [h1, h2]
    _ = -7 := rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_buyers_difference_l970_97033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compressed_rhombus_diagonals_l970_97061

/-- Represents a rhombus with its properties and compression behavior. -/
structure CompressibleRhombus where
  side_length : ℝ
  long_diagonal : ℝ
  compression_ratio : ℝ

/-- Calculates the new diagonal lengths after compression. -/
noncomputable def new_diagonals (r : CompressibleRhombus) : ℝ × ℝ :=
  let short_diagonal := 2 * Real.sqrt (r.side_length^2 - (r.long_diagonal/2)^2)
  let compression := (r.long_diagonal - short_diagonal) / (1 + r.compression_ratio)
  (r.long_diagonal - compression, short_diagonal + r.compression_ratio * compression)

/-- Theorem stating the new diagonal lengths after compression for a specific rhombus. -/
theorem compressed_rhombus_diagonals :
  let r : CompressibleRhombus := { side_length := 20, long_diagonal := 32, compression_ratio := 1.2 }
  let (new_long, new_short) := new_diagonals r
  (abs (new_long - 29.38) < 0.01) ∧ (abs (new_short - 27.14) < 0.01) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compressed_rhombus_diagonals_l970_97061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_grids_l970_97086

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Fin 16

-- Define the properties of a valid grid
def valid_grid (g : Grid) : Prop :=
  -- Each cell contains a unique number from 1 to 16
  (∀ i j, g i j ≠ 0) ∧
  (∀ i₁ j₁ i₂ j₂, (i₁ ≠ i₂ ∨ j₁ ≠ j₂) → g i₁ j₁ ≠ g i₂ j₂) ∧
  -- Numbers increase from left to right
  (∀ i j₁ j₂, j₁ < j₂ → g i j₁ < g i j₂) ∧
  -- Numbers increase from top to bottom
  (∀ i₁ i₂ j, i₁ < i₂ → g i₁ j < g i₂ j) ∧
  -- 1 is in the top left corner
  g 0 0 = 1 ∧
  -- 16 is in the bottom right corner
  g 3 3 = 16

-- Theorem statement
theorem count_valid_grids :
  ∃ (s : Finset Grid), (∀ g ∈ s, valid_grid g) ∧ s.card = 14400 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_grids_l970_97086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_primes_product_remainder_l970_97022

def is_odd_prime (n : ℕ) : Bool :=
  Nat.Prime n ∧ n % 2 = 1

def product_of_odd_primes_less_than (n : ℕ) : ℕ :=
  (List.range n).filter is_odd_prime |>.prod

theorem odd_primes_product_remainder :
  product_of_odd_primes_less_than 32 % 32 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_primes_product_remainder_l970_97022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_4_min_of_max_value_l970_97078

-- Define the quadratic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a/2

-- Theorem 1: Maximum value when a = 4
theorem max_value_when_a_4 :
  ∃ (M : ℝ), M = 2 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f 4 x ≤ M :=
by sorry

-- Theorem 2: Minimum value of the maximum value
theorem min_of_max_value :
  ∃ (t : ℝ), t = 1/2 ∧ 
  (∀ a : ℝ, ∃ (M : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ M) ∧ M ≥ t) ∧
  (∀ ε > 0, ∃ a : ℝ, ∃ (M : ℝ), 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≤ M) ∧ M < t + ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_4_min_of_max_value_l970_97078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forwarding_equation_l970_97013

/-- Represents the number of friends invited in each round -/
def x : ℕ := sorry

/-- The total number of participants after two rounds of forwarding -/
def total_participants : ℕ := 241

/-- Theorem stating that the equation x^2 + x + 1 = 241 correctly represents
    the total number of participants in a two-round forwarding process -/
theorem forwarding_equation :
  x^2 + x + 1 = total_participants :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forwarding_equation_l970_97013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_concyclic_and_center_locus_l970_97099

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y = 1 / x

-- Define the two lines passing through point T(0, a)
def line1 (x y a : ℝ) : Prop := y - a = 2 * x
def line2 (x y a : ℝ) : Prop := y - a = (1 / 2) * x

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ hyperbola x y ∧ (line1 x y a ∨ line2 x y a)}

-- Define the property of being concyclic
def concyclic (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ S, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2

-- Define the center of the circle
def circle_center (a : ℝ) : ℝ × ℝ := (-5/4 * a, a)

-- Theorem statement
theorem intersection_points_concyclic_and_center_locus :
  (∀ a : ℝ, concyclic (intersection_points a)) ∧
  (∀ x y : ℝ, (∃ a : ℝ, (x, y) = circle_center a) ↔ y = -4/5 * x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_concyclic_and_center_locus_l970_97099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l970_97069

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  intro x
  intro h
  contradiction


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l970_97069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_min_value_is_nine_halves_min_value_achieved_l970_97096

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1/3 : ℝ)^(x-2) = 3^y) : 
  ∀ a b : ℝ, a > 0 → b > 0 → (1/3 : ℝ)^(a-2) = 3^b → 4/x + 1/y ≤ 4/a + 1/b :=
by
  sorry

theorem min_value_is_nine_halves (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1/3 : ℝ)^(x-2) = 3^y) : 
  4/x + 1/y ≥ 9/2 :=
by
  sorry

theorem min_value_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1/3 : ℝ)^(x-2) = 3^y) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (1/3 : ℝ)^(a-2) = 3^b ∧ 4/a + 1/b = 9/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_min_value_is_nine_halves_min_value_achieved_l970_97096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_problem_l970_97009

/-- The height of a pole given its shadow length and the shadow length of a tree of known height -/
noncomputable def pole_height (pole_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) : ℝ :=
  (pole_shadow * tree_height) / tree_shadow

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem pole_height_problem (pole_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ) 
    (h_pole_shadow : pole_shadow = 84)
    (h_tree_height : tree_height = 28)
    (h_tree_shadow : tree_shadow = 32) :
    round_to_nearest (pole_height pole_shadow tree_height tree_shadow) = 74 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_height_problem_l970_97009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_half_circle_area_l970_97059

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 5*x

-- Define a as the derivative of f at x = 2
noncomputable def a : ℝ := (deriv f) 2

-- State the theorem
theorem integral_equals_half_circle_area :
  ∫ x in -a..a, Real.sqrt (a^2 - x^2) = (49 * Real.pi) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_half_circle_area_l970_97059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l970_97066

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.a 3 = S seq 3 ∧ seq.a 3 = 3 → seq.a 4 + seq.a 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l970_97066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_rental_cost_l970_97023

def rental_cost (daily_rate : ℝ) (per_mile_rate : ℝ) (days : ℕ) (miles : ℕ) (discount_rate : ℝ) : ℝ :=
  let base_cost := daily_rate * (days : ℝ) + per_mile_rate * (miles : ℝ)
  let discount := if days > 4 then discount_rate * base_cost else 0
  base_cost - discount

theorem sarah_rental_cost :
  rental_cost 30 0.25 5 500 0.1 = 247.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarah_rental_cost_l970_97023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_6_years_l970_97058

/-- The price of a computer after a certain number of price decreases -/
noncomputable def price_after_decreases (initial_price : ℝ) (num_decreases : ℕ) : ℝ :=
  initial_price * (2/3)^num_decreases

/-- Theorem: The price of a computer that initially costs 8100 yuan will be 2400 yuan after 6 years,
    given that the price decreases by one-third every two years -/
theorem price_decrease_6_years :
  price_after_decreases 8100 3 = 2400 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_6_years_l970_97058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_r_values_l970_97046

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle (renamed to avoid conflict)
def myCircle (x y r : ℝ) : Prop := (x - 4)^2 + y^2 = r^2

-- Define a line
structure Line where
  slope : Option ℝ
  intercept : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions of the problem
def problem_conditions (l : Line) (A B M : Point) (r : ℝ) : Prop :=
  -- Line l intersects parabola at A and B
  parabola A.x A.y ∧ parabola B.x B.y
  -- M is the midpoint of AB
  ∧ M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2
  -- Line l is tangent to the circle at M
  ∧ myCircle M.x M.y r
  -- r is positive
  ∧ r > 0

-- Theorem statement
theorem possible_r_values (l₁ l₂ : Line) (A₁ B₁ M₁ A₂ B₂ M₂ : Point) (r : ℝ) :
  problem_conditions l₁ A₁ B₁ M₁ r
  ∧ problem_conditions l₂ A₂ B₂ M₂ r
  ∧ l₁ ≠ l₂
  ∧ (∀ l A B M, problem_conditions l A B M r → (l = l₁ ∨ l = l₂))
  → 0 < r ∧ r ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_r_values_l970_97046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_statement_l970_97027

theorem one_correct_statement : 
  (∃ (a b : ℝ) (n : ℕ), (a + b)^n ≠ a^n + b^n) ∧ 
  (∃ (α β : ℝ), Real.sin (α + β) ≠ Real.sin α * Real.sin β) ∧ 
  (∀ (a b : ℝ × ℝ), (a.1 + b.1)^2 + (a.2 + b.2)^2 = 
    (a.1^2 + a.2^2) + 2 * (a.1 * b.1 + a.2 * b.2) + (b.1^2 + b.2^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_statement_l970_97027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_A_range_l970_97030

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

theorem f_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < Real.pi / 2)
  (h_periodic : ∀ x, Real.sin (ω * (x + Real.pi / 2) + φ) = -Real.sin (ω * x + φ))
  (h_odd : ∀ x, Real.sin (ω * (x + Real.pi / 6) + φ) = -Real.sin (ω * (-x + Real.pi / 6) + φ)) :
  f = λ x ↦ Real.sin (ω * x + φ) := by sorry

theorem f_A_range (A B C a b c : ℝ)
  (h_acute : 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2)
  (h_triangle : A + B + C = Real.pi)
  (h_sides : (2 * c - a) * Real.cos B = b * Real.cos A) :
  0 < f A ∧ f A ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_A_range_l970_97030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_mod_1000_l970_97002

/-- The number of permutations of "DDDDDDEEEEEFFFFFFFFFF" with constraints -/
def M : ℕ :=
  Finset.sum (Finset.range 5) (λ d =>
    Nat.choose 5 (5 - d) * Nat.choose 5 d * Nat.choose 8 (d + 3))

/-- The main theorem stating the result of the permutation count modulo 1000 -/
theorem permutation_count_mod_1000 : M % 1000 = 406 := by
  sorry

#eval M % 1000  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_count_mod_1000_l970_97002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l970_97047

noncomputable def f (x : ℝ) := Real.cos x ^ 2 + Real.sin (x - Real.pi / 2) ^ 2

noncomputable def g (x : ℝ) := f (x - Real.pi / 12) - 1

theorem problem_solution :
  (∀ x, f x = 1 + Real.cos (2 * x)) ∧
  (∀ x, f x = f (-x)) ∧
  (∀ x, g x = Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ k : ℤ, ∃ x, g x = 0 ∧ x = ↑k * Real.pi / 2 + Real.pi / 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l970_97047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_log2_l970_97015

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * Real.log 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_log2_l970_97015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_repair_problem_l970_97049

/-- The number of boards Petrov nailed with 2 nails each -/
def x : ℕ := sorry

/-- The number of boards Petrov nailed with 3 nails each -/
def y : ℕ := sorry

/-- The number of boards Vasechkin nailed with 3 nails each -/
def u : ℕ := sorry

/-- The number of boards Vasechkin nailed with 5 nails each -/
def v : ℕ := sorry

/-- The total number of boards each person nailed -/
def n : ℕ := sorry

theorem fence_repair_problem :
  (2 * x + 3 * y = 87) ∧ 
  (3 * u + 5 * v = 94) ∧ 
  (x + y = n) ∧ 
  (u + v = n) →
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_repair_problem_l970_97049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_with_tangent_circles_l970_97017

/-- Given three circles of radius 2 with centers P, Q, and R forming a right triangle
    with PQ as the hypotenuse, and these circles being tangent to one another and to
    two sides of triangle ABC, the perimeter of triangle ABC is 8 + 4√2. -/
theorem triangle_perimeter_with_tangent_circles (P Q R A B C : ℝ × ℝ) :
  let circle_radius : ℝ := 2
  let is_right_triangle (X Y Z : ℝ × ℝ) := 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (X.1 - Z.1)^2 + (X.2 - Z.2)^2 + (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2
  let distance (X Y : ℝ × ℝ) := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let is_tangent (C1 C2 : ℝ × ℝ) := distance C1 C2 = 2 * circle_radius
  let perimeter := distance A B + distance B C + distance C A
  is_right_triangle P Q R ∧
  is_tangent P Q ∧ is_tangent Q R ∧ is_tangent R P ∧
  (∃ (X Y : ℝ × ℝ), X ∈ ({A, B, C} : Set (ℝ × ℝ)) ∧ Y ∈ ({A, B, C} : Set (ℝ × ℝ)) ∧ X ≠ Y ∧
    is_tangent P X ∧ is_tangent Q Y) →
  perimeter = 8 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_with_tangent_circles_l970_97017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l970_97048

noncomputable def expression : ℝ := (56 * 0.57 * 0.85) / (2.8 * 19 * 1.7)

theorem expression_approximation : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.05 ∧ |expression - 0.3| < ε := by
  -- We'll use 0.01 as our ε
  use 0.01
  apply And.intro
  · -- Prove ε > 0
    norm_num
  · apply And.intro
    · -- Prove ε < 0.05
      norm_num
    · -- Prove |expression - 0.3| < ε
      -- This step would require actual computation, which is complex in Lean
      -- For now, we'll use sorry to skip the proof
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l970_97048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_splitting_sum_l970_97020

/-- Represents the state of stone piles -/
structure StonePiles where
  piles : List Nat

/-- The initial state with 25 stones in one pile -/
def initial_state : StonePiles :=
  { piles := [25] }

/-- Function to split a pile -/
def split_pile (n : Nat) : List Nat :=
  if n ≤ 1 then [n] else [n / 2, n - n / 2]

/-- Function to perform one step of splitting -/
def split_step (s : StonePiles) : StonePiles × Nat :=
  match s.piles with
  | [] => (s, 0)
  | p::ps =>
      let new_piles := split_pile p
      let product := if new_piles.length = 2 then new_piles[0]! * new_piles[1]! else 0
      ({ piles := new_piles ++ ps }, product)

/-- Predicate to check if all piles have one stone -/
def all_ones (s : StonePiles) : Prop :=
  s.piles.all (· = 1)

/-- Helper function to iterate split_step -/
def iterate_split_step (s : StonePiles) (n : Nat) : StonePiles × List Nat :=
  match n with
  | 0 => (s, [])
  | n + 1 =>
      let (s', product) := split_step s
      let (final, products) := iterate_split_step s' n
      (final, product :: products)

/-- The main theorem to prove -/
theorem stone_splitting_sum (s : StonePiles) : 
  s = initial_state → 
  ∃ (final : StonePiles) (products : List Nat), 
    (∃ (n : Nat), iterate_split_step s n = (final, products)) ∧ 
    all_ones final ∧ 
    products.sum = 300 := by
  sorry

#check stone_splitting_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_splitting_sum_l970_97020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l970_97037

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let F : ℝ × ℝ := (-c, 0)
  let A : ℝ × ℝ := (-a^2 / c, -b * (-a^2 / c) / a)
  let B : ℝ × ℝ := (a^2 * c / (b^2 - a^2), b * (a^2 * c / (b^2 - a^2)) / a)
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    (3 * (A.1 - F.1) = B.1 - F.1 ∧ 3 * (A.2 - F.2) = B.2 - F.2)) →
  c^2 / a^2 = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l970_97037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_l970_97034

noncomputable def sample : List ℝ := [9, 10, 11]

noncomputable def average (l : List ℝ) (x y : ℝ) : ℝ := 
  (l.sum + x + y) / (l.length + 2 : ℝ)

noncomputable def variance (l : List ℝ) (x y : ℝ) (avg : ℝ) : ℝ :=
  (l.map (λ z => (z - avg)^2)).sum / (l.length + 2 : ℝ) +
  ((x - avg)^2 + (y - avg)^2) / (l.length + 2 : ℝ)

theorem product_xy (x y : ℝ) 
  (h_avg : average sample x y = 10)
  (h_var : variance sample x y 10 = 4) : 
  x * y = 91 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_l970_97034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l970_97026

-- Define the original cosine function
noncomputable def original_function (x : ℝ) : ℝ := Real.cos x

-- Define the translation operation
def translate (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

-- Define the x-coordinate shrinking operation
def shrink_x (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (k * x)

-- Theorem statement
theorem cosine_transformation :
  let f := original_function
  let g := translate f (π / 3)
  let h := shrink_x g 2
  ∀ x, h x = Real.cos (2 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l970_97026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_unique_solution_when_b_not_one_solution_set_when_b_is_one_l970_97040

/-- The solution set for the equation √((x+2)² + 4a² - 4) + √((x-2)² + 4a² - 4) = 4b -/
def SolutionSet (b : ℝ) : Set ℝ :=
  if b ∈ Set.Icc 0 1 ∪ Set.Ioi 1 then {0}
  else if b = 1 then Set.Icc (-2) 2
  else ∅

/-- The main theorem stating the solution set for different values of b -/
theorem solution_set_characterization (a b : ℝ) (hb : b ≥ 0) :
  ∀ x, x ∈ SolutionSet b ↔ 
    Real.sqrt ((x + 2)^2 + 4 * a^2 - 4) + Real.sqrt ((x - 2)^2 + 4 * a^2 - 4) = 4 * b :=
by sorry

/-- Theorem stating that when b ≠ 1, the solution is unique and equal to 0 -/
theorem unique_solution_when_b_not_one (a b : ℝ) (hb : b ≥ 0) (hb_neq : b ≠ 1) :
  SolutionSet b = {0} :=
by sorry

/-- Theorem stating that when b = 1, the solution set is [-2, 2] -/
theorem solution_set_when_b_is_one (a : ℝ) :
  SolutionSet 1 = Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_unique_solution_when_b_not_one_solution_set_when_b_is_one_l970_97040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_proof_l970_97077

noncomputable section

def initial_amount : ℝ := 8000
def time_period : ℝ := 2
def borrow_rate : ℝ := 0.04
def borrow_compounds : ℝ := 4
def lend_rate : ℝ := 0.06
def lend_compounds : ℝ := 2

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounds) ^ (compounds * time)

noncomputable def borrowed_amount : ℝ := compound_interest initial_amount borrow_rate borrow_compounds time_period
noncomputable def lent_amount : ℝ := compound_interest initial_amount lend_rate lend_compounds time_period

noncomputable def total_gain : ℝ := lent_amount - borrowed_amount
noncomputable def gain_per_year : ℝ := total_gain / time_period

theorem transaction_gain_proof : 
  ∃ ε > 0, |gain_per_year - 170.61| < ε :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_proof_l970_97077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XYZ_area_l970_97054

noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_XYZ_area :
  let xy : ℝ := 31
  let yz : ℝ := 31
  let xz : ℝ := 46
  triangleArea xy yz xz = 476.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_XYZ_area_l970_97054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_number_theorem_l970_97003

/-- Checks if a number is coprime with 36 -/
def isCoprime36 (n : ℕ) : Prop := Nat.Coprime n 36

/-- Constructs A from B by moving the last digit to the first position -/
def constructA (b : ℕ) : ℕ :=
  let lastDigit := b % 10
  let remainingDigits := b / 10
  lastDigit * 10^7 + remainingDigits

theorem eight_digit_number_theorem :
  ∃ (bMin bMax : ℕ),
    bMin > 77777777 ∧
    bMax > 77777777 ∧
    isCoprime36 bMin ∧
    isCoprime36 bMax ∧
    constructA bMin = 17777779 ∧
    constructA bMax = 99999998 ∧
    (∀ b : ℕ, b > 77777777 → isCoprime36 b →
      17777779 ≤ constructA b ∧ constructA b ≤ 99999998) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_number_theorem_l970_97003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l970_97056

open Real

/-- The function f(θ) to be maximized -/
noncomputable def f (θ : ℝ) : ℝ := 
  sqrt (1 - cos θ + sin θ) + sqrt (cos θ + 2) + sqrt (3 - sin θ)

/-- Theorem stating that the maximum value of f(θ) is 3√2 for θ in [0, π] -/
theorem max_value_of_f :
  ∃ (θ : ℝ), 0 ≤ θ ∧ θ ≤ π ∧ f θ = 3 * sqrt 2 ∧
  ∀ (φ : ℝ), 0 ≤ φ ∧ φ ≤ π → f φ ≤ 3 * sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l970_97056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_men_all_properties_l970_97045

/-- Represents the properties a man can have -/
structure ManProperties where
  married : Bool
  hasTV : Bool
  hasRadio : Bool
  hasAC : Bool

/-- The set of all men -/
def Men : Finset ManProperties := sorry

/-- The total number of men is 100 -/
axiom total_men : Men.card = 100

/-- The number of married men is 85 -/
axiom married_men : (Men.filter (λ m => m.married)).card = 85

/-- The number of men with TV is 75 -/
axiom tv_men : (Men.filter (λ m => m.hasTV)).card = 75

/-- The number of men with radio is 85 -/
axiom radio_men : (Men.filter (λ m => m.hasRadio)).card = 85

/-- The number of men with AC is 70 -/
axiom ac_men : (Men.filter (λ m => m.hasAC)).card = 70

/-- The theorem stating the maximum number of men with all properties -/
theorem max_men_all_properties :
  (Men.filter (λ m => m.married ∧ m.hasTV ∧ m.hasRadio ∧ m.hasAC)).card ≤ 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_men_all_properties_l970_97045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_line_l970_97060

theorem curve_tangent_line (a : ℝ) : 
  (∃ x : ℝ, x^3 - 2*x + a = x + 1 ∧ 3*x^2 - 2 = 1) → (a = -1 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_line_l970_97060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_tetrahedron_properties_l970_97043

/-- The radius of the circumscribed sphere of an equilateral tetrahedron -/
noncomputable def circumradius (a b c : ℝ) : ℝ := Real.sqrt ((a^2 + b^2 + c^2) / 8)

/-- The volume of an equilateral tetrahedron -/
noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  Real.sqrt ((a^2 + c^2 - b^2) * (a^2 + b^2 - c^2) * (b^2 + c^2 - a^2)) / (6 * Real.sqrt 2)

/-- Theorem: The formulas for circumradius and volume of an equilateral tetrahedron -/
theorem equilateral_tetrahedron_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let R := circumradius a b c
  let V := tetrahedron_volume a b c
  (R = Real.sqrt ((a^2 + b^2 + c^2) / 8)) ∧
  (V = Real.sqrt ((a^2 + c^2 - b^2) * (a^2 + b^2 - c^2) * (b^2 + c^2 - a^2)) / (6 * Real.sqrt 2)) :=
by
  -- Unfold the definitions of R and V
  unfold circumradius tetrahedron_volume
  -- The equality follows directly from the definitions
  simp

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_tetrahedron_properties_l970_97043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l970_97016

/-- A right circular cone with a base diameter and a cross-section triangle
    with a specified vertex angle. -/
structure Cone :=
  (base_diameter : ℝ)
  (vertex_angle : ℝ)

/-- A sphere inscribed in the cone, tangent to its sides and sitting on the base. -/
structure InscribedSphere (c : Cone) :=
  (radius : ℝ)

/-- The volume of a sphere. -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

theorem inscribed_sphere_volume (c : Cone) (s : InscribedSphere c) :
  c.base_diameter = 18 ∧ c.vertex_angle = Real.pi/2 →
  sphere_volume s.radius = 121.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l970_97016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_perimeter_of_divided_triangle_l970_97076

/-- Represents an isosceles triangle with given base and height -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the perimeter of a piece of the divided triangle -/
noncomputable def piece_perimeter (t : IsoscelesTriangle) (k : ℕ) : ℝ :=
  1 + Real.sqrt (t.height^2 + (k * (t.base / 10))^2) + 
      Real.sqrt (t.height^2 + ((k + 1) * (t.base / 10))^2)

/-- Theorem stating the greatest perimeter among 10 equal area pieces -/
theorem greatest_perimeter_of_divided_triangle (t : IsoscelesTriangle) 
  (h_base : t.base = 10) (h_height : t.height = 12) :
  ∃ (max_perimeter : ℝ), 
    (∀ k, k < 10 → piece_perimeter t k ≤ max_perimeter) ∧
    (abs (max_perimeter - 31.62) < 0.005) := by
  sorry

#check greatest_perimeter_of_divided_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_perimeter_of_divided_triangle_l970_97076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_needed_is_twelve_l970_97021

/-- Calculates the total gas needed for a trip with given distances and fuel efficiencies -/
noncomputable def total_gas_needed (city_mpg : ℝ) (highway_mpg : ℝ) (city_distance : ℝ) (highway_distance_grandma : ℝ) (highway_distance_jane : ℝ) : ℝ :=
  city_distance / city_mpg + (highway_distance_grandma + highway_distance_jane) / highway_mpg

/-- Theorem stating that the total gas needed for the given trip is 12 gallons -/
theorem gas_needed_is_twelve :
  let city_mpg : ℝ := 20
  let highway_mpg : ℝ := 25
  let city_distance : ℝ := 60
  let highway_distance_grandma : ℝ := 150
  let highway_distance_jane : ℝ := 75
  total_gas_needed city_mpg highway_mpg city_distance highway_distance_grandma highway_distance_jane = 12 :=
by
  -- Unfold the definition of total_gas_needed
  unfold total_gas_needed
  -- Simplify the expression
  simp
  -- The proof is completed with 'sorry' as requested
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_needed_is_twelve_l970_97021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_trade_increases_consumption_our_world_consumption_increase_l970_97019

/-- Represents a country with its production capabilities -/
structure Country where
  eggplantYield : ℕ
  cornYield : ℕ

/-- The world consisting of two countries -/
structure World where
  countryA : Country
  countryB : Country

/-- Represents the total consumption of vegetables -/
def totalConsumption (w : World) (isFreeTradeOption : Bool) : ℕ :=
  if isFreeTradeOption then
    let specializedProduction := w.countryA.cornYield + w.countryB.eggplantYield
    2 * specializedProduction
  else
    let combinedEggplantYield := w.countryA.eggplantYield + w.countryB.eggplantYield
    let combinedCornYield := w.countryA.cornYield + w.countryB.cornYield
    2 * min combinedEggplantYield combinedCornYield

/-- Theorem: Free trade leads to higher total consumption -/
theorem free_trade_increases_consumption (w : World) :
    totalConsumption w true > totalConsumption w false := by
  sorry

/-- The world with given production capabilities -/
def ourWorld : World :=
  { countryA := { eggplantYield := 10, cornYield := 8 },
    countryB := { eggplantYield := 18, cornYield := 12 } }

/-- Corollary: In our specific world, free trade increases consumption from 24 to 36 -/
theorem our_world_consumption_increase :
    totalConsumption ourWorld true = 36 ∧
    totalConsumption ourWorld false = 24 := by
  sorry

#eval totalConsumption ourWorld true
#eval totalConsumption ourWorld false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_trade_increases_consumption_our_world_consumption_increase_l970_97019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_eight_squared_seven_cubed_nine_squared_l970_97006

theorem factors_of_eight_squared_seven_cubed_nine_squared :
  (Finset.card (Nat.divisors (8^2 * 7^3 * 9^2))) = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_eight_squared_seven_cubed_nine_squared_l970_97006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_specific_l970_97036

/-- The height of a tray formed from a square paper --/
noncomputable def tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : ℝ :=
  let diagonal_length := side_length * Real.sqrt 2
  let cut_length := Real.sqrt (2 * cut_distance^2)
  (cut_length + cut_distance) / Real.sqrt 2

/-- Theorem stating the height of the tray for the given conditions --/
theorem tray_height_specific : 
  tray_height 120 (Real.sqrt 21) (π/4) = Real.sqrt (Real.sqrt 3444) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tray_height_specific_l970_97036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l970_97031

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x - 3)

-- Define the domain E(f)
def E_f : Set ℝ := {y | y ≠ 1}

-- Theorem statement
theorem domain_of_f : Set.range f = E_f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l970_97031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l970_97014

theorem cube_surface_area_increase : 
  ∀ (s : ℝ), s > 0 → 
  (6 * (1.5 * s)^2 - 6 * s^2) / (6 * s^2) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l970_97014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_and_bisector_l970_97082

-- Define the line l
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 2 = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_O A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_O B.1 B.2 ∧
  A ≠ B

-- Define the length of a chord
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 0

theorem intersection_chord_and_bisector 
  (A B : ℝ × ℝ) (h : intersection_points A B) : 
  (chord_length A B = 2) ∧ 
  (∀ x y, perp_bisector x y ↔ (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_and_bisector_l970_97082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_sector_auditorium_sampling_shooting_events_complementary_trig_inequality_l970_97028

-- Statement ①
theorem arc_length_sector (angle : Real) (radius : Real) :
  angle = 2 * π / 3 ∧ radius = 2 →
  2 * π * radius * (angle / (2 * π)) = 4 * π / 3 := by sorry

-- Statement ②
def systematic_sampling (total_rows : Nat) (seats_per_row : Nat) (selected_seat : Nat) :=
  ∀ row, row ≤ total_rows → ∃ student : Nat × Nat, student.1 = row ∧ student.2 = selected_seat

theorem auditorium_sampling :
  systematic_sampling 25 20 15 := by sorry

-- Statement ③
def complementary_events (A B : Set (Nat → Bool)) :=
  (∀ x, A x ∨ B x) ∧ (∀ x, ¬(A x ∧ B x))

theorem shooting_events_complementary :
  let hit_at_least_once := λ e : Nat → Bool => e 0 ∨ e 1
  let miss_both_times := λ e : Nat → Bool => ¬(e 0) ∧ ¬(e 1)
  complementary_events hit_at_least_once miss_both_times := by sorry

-- Statement ④
theorem trig_inequality (x : Real) :
  0 < x ∧ x < π / 2 →
  Real.sin x < x ∧ x < Real.tan x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_sector_auditorium_sampling_shooting_events_complementary_trig_inequality_l970_97028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_area_correct_l970_97097

/-- Represents a closed shape inside a unit circle -/
structure ClosedShape where
  area : ℝ
  inside_unit_circle : area ≤ Real.pi

/-- Represents the result of throwing beans randomly into a unit circle -/
structure BeanThrow where
  total : ℕ
  inside_shape : ℕ
  shape : ClosedShape
  valid_throw : inside_shape ≤ total

/-- The estimated area of a closed shape based on a random bean throw experiment -/
noncomputable def estimate_area (throw : BeanThrow) : ℝ :=
  (throw.inside_shape : ℝ) * Real.pi / (throw.total : ℝ)

/-- Theorem stating that the estimated area equals the actual area of the closed shape -/
theorem estimate_area_correct (throw : BeanThrow) : 
  estimate_area throw = throw.shape.area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_area_correct_l970_97097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_winning_strategy_l970_97035

/-- Represents a circle in the game --/
inductive Circle : Type
| A | B | One | Two | Three | Four | Five | Six | Seven

/-- Represents a line connecting three circles --/
def Line := (Circle × Circle × Circle)

/-- The game board configuration --/
structure GameBoard :=
  (circles : List Circle)
  (lines : List Line)
  (player_a_colored : Circle)
  (player_b_colored : Circle)

/-- Represents a player's move --/
def Move := Circle

/-- Checks if a move is valid --/
def is_valid_move (board : GameBoard) (move : Move) : Prop :=
  move ∈ board.circles ∧ move ≠ board.player_a_colored ∧ move ≠ board.player_b_colored

/-- Checks if a player has won --/
def is_winning_move (board : GameBoard) (move : Move) : Prop :=
  ∃ (line : Line), line ∈ board.lines ∧ 
    ((move = line.1 ∧ board.player_a_colored = line.2.1 ∧ board.player_b_colored = line.2.2) ∨
     (move = line.2.1 ∧ board.player_a_colored = line.1 ∧ board.player_b_colored = line.2.2) ∨
     (move = line.2.2 ∧ board.player_a_colored = line.1 ∧ board.player_b_colored = line.2.1))

/-- The initial game board --/
def initial_board : GameBoard :=
  { circles := [Circle.A, Circle.B, Circle.One, Circle.Two, Circle.Three, Circle.Four, Circle.Five, Circle.Six, Circle.Seven],
    lines := [
      (Circle.A, Circle.One, Circle.Two),
      (Circle.A, Circle.Three, Circle.Six),
      (Circle.A, Circle.Four, Circle.Seven),
      (Circle.B, Circle.One, Circle.Five),
      (Circle.B, Circle.Three, Circle.Four),
      (Circle.B, Circle.Six, Circle.Seven),
      (Circle.Two, Circle.Four, Circle.Six),
      (Circle.Two, Circle.Five, Circle.Seven),
      (Circle.One, Circle.Three, Circle.Seven)
    ],
    player_a_colored := Circle.A,
    player_b_colored := Circle.B
  }

/-- Theorem: Player A has a winning strategy if and only if their next move is to color circle 2, 3, or 4 --/
theorem player_a_winning_strategy :
  ∀ (move : Move),
    is_valid_move initial_board move →
    (is_winning_move initial_board move ∨
     (∃ (counter_move : Move),
       is_valid_move initial_board counter_move ∧
       ¬is_winning_move initial_board counter_move ∧
       ¬(∃ (winning_move : Move),
         is_valid_move initial_board winning_move ∧
         is_winning_move initial_board winning_move))) ↔
    (move = Circle.Two ∨ move = Circle.Three ∨ move = Circle.Four) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_winning_strategy_l970_97035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l970_97018

/-- The distance covered by a wheel with given diameter and number of revolutions -/
noncomputable def distance_covered (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  Real.pi * diameter * revolutions

/-- Theorem stating that a wheel with 10 cm diameter taking 16.81528662420382 revolutions 
    covers approximately 528.54 cm -/
theorem wheel_distance_theorem :
  let d := (10 : ℝ)
  let r := 16.81528662420382
  abs (distance_covered d r - 528.54) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_distance_theorem_l970_97018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_propositions_l970_97010

-- Define the types for lines and planes
structure Line : Type := (id : ℕ)
structure Plane : Type := (id : ℕ)

-- Define the basic operations and relations
def subset (l : Line) (p : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) : Line := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane_plane (p1 p2 : Plane) : Prop := sorry
def parallel_plane_plane (p1 p2 : Plane) : Prop := sorry

-- Define the given lines and planes
variable (a b : Line)
variable (α β γ : Plane)

-- Define the propositions
def proposition1 (a b : Line) (α β : Plane) : Prop := 
  (intersect α β = a) → (subset b α) → (perpendicular a b) → (perpendicular_plane_plane α β)

def proposition2 (a : Line) (α β : Plane) : Prop := 
  (subset a α) → (∀ l : Line, subset l β → perpendicular a l) → (perpendicular_plane_plane α β)

def proposition3 (a b : Line) (α β γ : Plane) : Prop := 
  (perpendicular_plane_plane α β) → (intersect α β = a) → (intersect α γ = b) → (perpendicular a b)

def proposition4 (a : Line) (α : Plane) : Prop := 
  ¬(perpendicular_line_plane a α) → ¬(∃ (l : Line), subset l α ∧ perpendicular a l)

def proposition5 (a b : Line) (α β : Plane) : Prop := 
  (perpendicular_line_plane a α) → (perpendicular_line_plane b β) → (parallel_plane_plane α β)

-- Theorem to prove
theorem evaluate_propositions (a b : Line) (α β γ : Plane) :
  proposition2 a α β ∧ proposition5 a b α β ∧ ¬proposition1 a b α β ∧ ¬proposition3 a b α β γ ∧ ¬proposition4 a α :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_propositions_l970_97010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_M_div_100_l970_97032

noncomputable def M : ℚ := (2 * 17).factorial * (1 / ((3 * 16).factorial) + 1 / ((4 * 15).factorial) + 
  1 / ((5 * 14).factorial) + 1 / ((6 * 13).factorial) + 1 / ((7 * 12).factorial) + 
  1 / ((8 * 11).factorial) + 1 / ((9 * 10).factorial))

theorem greatest_integer_less_than_M_div_100 : 
  ⌊M / 100⌋ = 275 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_M_div_100_l970_97032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_pdf_l970_97067

/-- A function representing a potential probability density function -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else α * Real.exp (-α * x)

/-- Theorem stating that f is a probability density function when α > 0 -/
theorem f_is_pdf (α : ℝ) (h : α > 0) :
  (∀ x, f α x ≥ 0) ∧ (∫ x, f α x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_pdf_l970_97067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l970_97090

/-- A polynomial with real coefficients. -/
def MyPolynomial := ℝ → ℝ

/-- The property that P satisfies the given equation for all x, y, z with x + y + z = 0. -/
def SatisfiesEquation (P : MyPolynomial) : Prop :=
  ∀ x y z : ℝ, x + y + z = 0 →
    P (x + y)^3 + P (y + z)^3 + P (z + x)^3 = 3 * P ((x + y) * (y + z) * (z + x))

/-- The theorem stating that if P satisfies the equation, it must be one of 0, x, or -x. -/
theorem polynomial_characterization (P : MyPolynomial) (h : SatisfiesEquation P) :
    (∀ x, P x = 0) ∨ (∀ x, P x = x) ∨ (∀ x, P x = -x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_characterization_l970_97090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l970_97070

noncomputable def surfaceArea (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem cylinder_surface_area (R : ℝ) (h : R > 0) :
  let S := {(x, y, z) : ℝ × ℝ × ℝ | x^2 + y^2 = R^2 ∧ 0 ≤ z ∧ z ≤ y}
  surfaceArea S = 2 * R^2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l970_97070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_negative_one_l970_97062

-- Define the circle's center and the point on the circle
def center : ℝ × ℝ := (1, 3)
def point : ℝ × ℝ := (4, 6)

-- Define the slope of the radius
noncomputable def radius_slope : ℝ := (point.2 - center.2) / (point.1 - center.1)

-- Define the slope of the tangent line
noncomputable def tangent_slope : ℝ := -1 / radius_slope

-- Theorem statement
theorem tangent_slope_is_negative_one :
  tangent_slope = -1 := by
  -- Expand the definitions
  unfold tangent_slope radius_slope
  -- Simplify the expression
  simp [center, point]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_is_negative_one_l970_97062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_bike_prices_l970_97081

/-- Represents the sales and pricing of mountain bikes over three months -/
structure MountainBikeSales where
  july_sales : ℝ
  august_price_increase : ℝ
  august_sales : ℝ
  september_price_decrease_percent : ℝ
  september_profit_percent : ℝ

/-- Calculates the selling price per bike in August -/
noncomputable def august_price (s : MountainBikeSales) : ℝ :=
  s.august_sales * s.july_sales / (s.july_sales + s.august_price_increase * s.august_sales)

/-- Calculates the cost price of each mountain bike -/
noncomputable def cost_price (s : MountainBikeSales) : ℝ :=
  let september_price := august_price s * (1 - s.september_price_decrease_percent)
  september_price / (1 + s.september_profit_percent)

/-- Theorem stating the correct selling price in August and cost price -/
theorem mountain_bike_prices (s : MountainBikeSales) 
  (h1 : s.july_sales = 22500)
  (h2 : s.august_price_increase = 100)
  (h3 : s.august_sales = 25000)
  (h4 : s.september_price_decrease_percent = 0.15)
  (h5 : s.september_profit_percent = 0.25) :
  august_price s = 1000 ∧ cost_price s = 680 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_bike_prices_l970_97081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alec_notebook_price_l970_97094

/-- Calculates the highest possible whole-dollar price per notebook given the conditions --/
def highest_notebook_price (budget : ℕ) (num_notebooks : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  let remaining_budget := budget - entrance_fee
  let max_price_with_tax := (remaining_budget : ℚ) / (1 + tax_rate)
  let max_price_per_notebook := max_price_with_tax / num_notebooks
  (Int.floor max_price_per_notebook).toNat

/-- The theorem stating the highest possible notebook price for Alec --/
theorem alec_notebook_price :
  highest_notebook_price 160 20 5 (7/100) = 7 := by
  sorry

#eval highest_notebook_price 160 20 5 (7/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alec_notebook_price_l970_97094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_concentration_l970_97063

/-- Proves that mixing two alcohol solutions results in a specific concentration -/
theorem alcohol_mixture_concentration
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percent : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percent : ℝ)
  (total_volume : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percent = 20)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percent = 55)
  (h5 : total_volume = 8) :
  (vessel1_capacity * (vessel1_alcohol_percent / 100) +
   vessel2_capacity * (vessel2_alcohol_percent / 100)) / total_volume * 100 = 46.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_concentration_l970_97063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l970_97098

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

theorem symmetry_axis (x : ℝ) : f (Real.pi / 6 + x) = f (Real.pi / 6 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_l970_97098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_parallel_condition_l970_97064

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the orthocenter, centroid, and H₃ point
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry
noncomputable def centroid (t : Triangle) : ℝ × ℝ := sorry
noncomputable def H₃ (t : Triangle) : ℝ × ℝ := sorry

-- Define the tangent function
noncomputable def tg (θ : ℝ) : ℝ := sorry

-- Define a line
structure Line where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

-- Define parallelism between a line and a side of the triangle
def is_parallel_to_side (l : Line) (t : Triangle) : Prop := sorry

-- Define angle of a point in a triangle
noncomputable def angle_at (t : Triangle) (p : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem euler_line_parallel_condition (t : Triangle) :
  is_parallel_to_side (Line.mk (orthocenter t) (centroid t)) t ↔ 
  tg (angle_at t t.A) * tg (angle_at t t.B) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_parallel_condition_l970_97064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_surjective_l970_97053

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def g (x : ℝ) : ℝ := 2 * (floor x : ℝ) - x

theorem g_surjective : ∀ y : ℝ, ∃ x : ℝ, g x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_surjective_l970_97053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_theorem_l970_97055

/-- A line passing through (3,0) intersecting the circle (x-1)^2 + y^2 = 1 -/
structure IntersectingLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The line passes through (3,0) -/
  passes_through : k * 3 = 0
  /-- The line intersects the circle -/
  intersects_circle : ∃ (x y : ℝ), k * (x - 3) = y ∧ (x - 1)^2 + y^2 = 1

/-- The range of inclination angles for lines intersecting the circle -/
def inclination_angle_range (l : IntersectingLine) : Set ℝ :=
  {θ | 0 ≤ θ ∧ θ ≤ Real.pi/6} ∪ {θ | 5*Real.pi/6 ≤ θ ∧ θ ≤ Real.pi}

/-- Theorem stating the range of inclination angles -/
theorem inclination_angle_range_theorem (l : IntersectingLine) :
  ∃ θ, θ ∈ inclination_angle_range l ∧ Real.tan θ = l.k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_theorem_l970_97055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_satisfies_projections_l970_97038

noncomputable def u : ℝ × ℝ := (-2/5, 81/10)

noncomputable def proj (a : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := a.1 * v.1 + a.2 * v.2
  let norm_squared := a.1^2 + a.2^2
  (dot / norm_squared * a.1, dot / norm_squared * a.2)

theorem u_satisfies_projections :
  proj (3, 2) u = (45/13, 30/13) ∧
  proj (1, 4) u = (32/17, 128/17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_satisfies_projections_l970_97038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l970_97072

theorem problem_statement : (256 : ℝ) ^ (16/100 : ℝ) * (256 : ℝ) ^ (9/100 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l970_97072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l970_97087

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y + 1 = 0

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 6

theorem circle_radius_proof :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l970_97087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2007_is_zero_l970_97051

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | n + 2 => (Real.sqrt 2 * sequence_a (n + 1) + Real.sqrt 6) / (sequence_a (n + 1) - Real.sqrt 2)

theorem a_2007_is_zero : sequence_a 2007 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2007_is_zero_l970_97051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l970_97001

theorem equation_roots_properties (a b : ℝ) 
  (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
       (|x^2 + a*x + b| = 2) ∧ 
       (|y^2 + a*y + b| = 2) ∧ 
       (|z^2 + a*z + b| = 2)) : 
  (a^2 - 4*b - 8 = 0) ∧ 
  (∀ x y z : ℝ, x ≠ y → y ≠ z → x ≠ z → 
    (|x^2 + a*x + b| = 2) → (|y^2 + a*y + b| = 2) → (|z^2 + a*z + b| = 2) →
    (x + y + z = 180) → (x = 60 ∨ y = 60 ∨ z = 60)) ∧
  (∀ x y z : ℝ, x ≠ y → y ≠ z → x ≠ z → 
    (|x^2 + a*x + b| = 2) → (|y^2 + a*y + b| = 2) → (|z^2 + a*z + b| = 2) →
    (x^2 + y^2 = z^2 ∨ y^2 + z^2 = x^2 ∨ z^2 + x^2 = y^2) → 
    (a = -16 ∧ b = 62)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l970_97001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_percentage_l970_97080

theorem alcohol_solution_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (initial_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_alcohol = 6.5)
  (h3 : added_water = 3.5)
  (h4 : final_percentage = 17)
  (h5 : final_percentage / 100 * (initial_volume + added_alcohol + added_water) =
        initial_percentage / 100 * initial_volume + added_alcohol) :
  initial_percentage = 5 := by
  sorry

#check alcohol_solution_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_percentage_l970_97080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unsold_books_l970_97008

theorem unsold_books (total_books : ℕ) (sold_price : ℚ) (total_revenue : ℚ) :
  (2 : ℚ) / 3 * total_books * sold_price = total_revenue →
  sold_price = 7 / 2 →
  total_revenue = 280 →
  (total_books : ℚ) / 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unsold_books_l970_97008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l970_97042

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop :=
  x = t ∧ y = 1 + 2 * t

-- Theorem statement
theorem line_intersects_circle :
  ∃ (x y : ℝ), (∃ (t : ℝ), line_l t x y) ∧
               (∃ (ρ θ : ℝ), circle_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l970_97042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_sum_l970_97052

/-- Given a triangle PQR with vertices P(-3,4), Q(-12,-10), and R(4,-2),
    if the equation of the angle bisector of ∠P is of the form 3x + y + c = 0,
    then a + c equals a specific value. -/
theorem angle_bisector_sum (c : ℝ) : 
  let P : ℝ × ℝ := (-3, 4)
  let Q : ℝ × ℝ := (-12, -10)
  let R : ℝ × ℝ := (4, -2)
  let angle_bisector_equation (x y : ℝ) := 3 * x + y + c = 0
  ∃ (result : ℝ), (3 + c = result) ∧ 
    (∀ x y : ℝ, ((x, y) ∈ Set.Icc P (midpoint ℝ Q R) ↔ angle_bisector_equation x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_sum_l970_97052
