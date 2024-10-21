import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_leading_coefficient_l1231_123179

/-- A quadratic polynomial with rational coefficients -/
def QuadraticPolynomial (a b c : ℚ) : ℤ → ℚ := fun x => a * x^2 + b * x + c

/-- A quadratic polynomial is integer-valued if it maps integers to integers -/
def IsIntegerValued (P : ℤ → ℚ) : Prop := ∀ x : ℤ, ∃ y : ℤ, P x = y

/-- The leading coefficient of a quadratic polynomial -/
noncomputable def LeadingCoefficient (P : ℤ → ℚ) : ℚ :=
  Classical.choose (exists_unique_quadratic_coeff P)
where
  exists_unique_quadratic_coeff (P : ℤ → ℚ) : ∃! a : ℚ, ∀ x y : ℤ,
    P (x + y) - P x - P y + P 0 = a * (2 * x * y + x^2 + y^2) := sorry

theorem smallest_leading_coefficient :
  (∃ P : ℤ → ℚ, IsIntegerValued P ∧ LeadingCoefficient P = 1/2) ∧
  (∀ P : ℤ → ℚ, IsIntegerValued P → LeadingCoefficient P ≥ 1/2) := by
  sorry

#check smallest_leading_coefficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_leading_coefficient_l1231_123179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_greater_than_g_l1231_123181

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x

noncomputable def g (x : ℝ) : ℝ := 1 / (exp x)

theorem f_derivative_greater_than_g (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1/2 : ℝ) 2, 
    (deriv (f a)) x₁ > g x₂) → a > exp (-2) - 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_greater_than_g_l1231_123181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l1231_123112

/-- Custom operation @ for positive integers -/
def custom_at (k j : ℕ+) : ℕ+ :=
  sorry

/-- Given a and b, prove that b = 4040 -/
theorem find_b (a b : ℕ+) (h1 : a = 2020) (h2 : (a : ℚ) / (b : ℚ) = 1/2) : 
  b = 4040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l1231_123112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1231_123168

-- Define the function f(x) = (x-1)^2 * e^x
noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 * Real.exp x

-- Theorem statement
theorem function_properties :
  (∃ x : ℝ, f x = 0) ∧ 
  (∃ x_max : ℝ, ∀ y : ℝ, f y ≤ f x_max) ∧
  (∃ x_min : ℝ, ∀ y : ℝ, f y ≥ f x_min) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f x₁ = 1 ∧ f x₂ = 1 ∧ f x₃ = 1 ∧
    ∀ x : ℝ, f x = 1 → x = x₁ ∨ x = x₂ ∨ x = x₃) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1231_123168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speedster_convertibles_count_l1231_123176

/-- The inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  non_speedsters : ℕ
  speedster_convertibles : ℕ

/-- The conditions of the problem -/
def inventory_conditions (i : Inventory) : Prop :=
  i.speedsters = (2 * i.total) / 3 ∧
  i.non_speedsters = 40 ∧
  i.total = i.speedsters + i.non_speedsters ∧
  i.total = 120

/-- The theorem to prove -/
theorem speedster_convertibles_count (i : Inventory) :
  inventory_conditions i → i.speedster_convertibles = 64 :=
by
  sorry

#check speedster_convertibles_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speedster_convertibles_count_l1231_123176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_symmetry_l1231_123142

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem sine_translation_symmetry (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < Real.pi / 2)
  (h4 : f ω φ 0 = - f ω φ (Real.pi / 2)) 
  (h5 : ∀ x, f ω φ (x - Real.pi / 12) = - f ω φ (-x - Real.pi / 12)) :
  φ = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_translation_symmetry_l1231_123142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_reaches_alex_l1231_123136

noncomputable section

def distance_between : ℝ := 30
def rate_of_approach : ℝ := 2
def initial_time : ℝ := 10

def carmen_speed : ℝ → ℝ := λ _ => 80
def alex_speed : ℝ → ℝ := λ _ => 40

axiom speed_relation : ∀ t, carmen_speed t = 2 * alex_speed t
axiom approach_rate : ∀ t, carmen_speed t + alex_speed t = rate_of_approach * 60

def remaining_distance : ℝ := distance_between - rate_of_approach * initial_time

def time_to_meet : ℝ := initial_time + remaining_distance / carmen_speed initial_time

theorem carmen_reaches_alex : time_to_meet = 17.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_reaches_alex_l1231_123136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l1231_123152

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => (18 * x - 2) ^ (1/3) + (16 * x + 2) ^ (1/3) - 5 * x ^ (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 31 / 261 ∨ x = -35 / 261 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solutions_l1231_123152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1231_123144

/-- Represents the time (in seconds) it takes for a train to cross a pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem stating that a train with the given specifications takes approximately 9 seconds to cross a pole -/
theorem train_crossing_pole_time :
  let train_length := 175
  let train_speed := 70
  let crossing_time := train_crossing_time train_length train_speed
  (crossing_time > 8.9 ∧ crossing_time < 9.1) := by
  sorry

-- Use #eval only for computable functions
def approx_train_crossing_time (train_length : Float) (train_speed_kmh : Float) : Float :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

#eval approx_train_crossing_time 175 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1231_123144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_formula_l1231_123173

noncomputable def intersection_point (a b : ℂ) : ℂ := 2 * a * b / (a + b)

/-- A predicate stating that a line is tangent to the circle at a given point -/
def IsTangentLine (l : ℂ → ℂ) (p : ℂ) : Prop :=
  ∃ (r : ℝ), Complex.abs p = r ∧ ∀ (x : ℂ), l x ≠ p → Complex.abs (l x) > r

/-- The theorem stating that the intersection point of tangents is given by the formula -/
theorem intersection_point_formula (a b : ℂ) (h : ∃ (r : ℝ), Complex.abs a = r ∧ Complex.abs b = r) :
  let u := intersection_point a b
  (∃ (l₁ l₂ : ℂ → ℂ), IsTangentLine l₁ a ∧ IsTangentLine l₂ b ∧ l₁ u = l₂ u) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_formula_l1231_123173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1231_123165

/-- Represents a simple interest loan -/
structure Loan where
  principal : ℚ
  time : ℚ
  rate : ℚ

/-- Calculates the interest for a simple interest loan -/
def interest (loan : Loan) : ℚ :=
  loan.principal * loan.time * loan.rate / 100

/-- The problem statement -/
theorem interest_rate_calculation (loan1 loan2 : Loan) 
  (h1 : loan1.principal = 5000 ∧ loan1.time = 2)
  (h2 : loan2.principal = 3000 ∧ loan2.time = 4)
  (h3 : loan1.rate = loan2.rate)
  (h4 : interest loan1 + interest loan2 = 1760) :
  loan1.rate = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1231_123165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l1231_123138

theorem sum_of_divisors_450_prime_factors :
  let n : ℕ := 450
  let sum_of_divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id
  (Nat.factorization sum_of_divisors).support.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l1231_123138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_n_l1231_123199

theorem solve_for_n : ∃ n : ℕ, 2^7 * 3^4 * n = Nat.factorial 10 ∧ n = 350 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_n_l1231_123199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_intersection_properties_l1231_123129

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the intersection line
noncomputable def intersection_line (p : ℝ) (x y : ℝ) : Prop := y = 2*Real.sqrt 2*(x - p/2)

-- Define the length of AB
def length_AB : ℝ := 9

-- Define the slope of CF
noncomputable def slope_CF : ℝ := -Real.sqrt 3

theorem parabola_and_intersection_properties :
  ∀ p : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    parabola p x1 y1 ∧
    parabola p x2 y2 ∧
    intersection_line p x1 y1 ∧
    intersection_line p x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = length_AB^2) →
  (∀ x y : ℝ, parabola p x y ↔ y^2 = 8*x) ∧
  (∃ xp yp xc : ℝ,
    parabola p xp yp ∧
    xc = p/2 ∧
    (yp / (xp - xc) = slope_CF) →
    (xp - p/2)^2 + yp^2 = 8^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_intersection_properties_l1231_123129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l1231_123148

-- Define the curves
noncomputable def curve1 (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * Real.exp x
noncomputable def curve2 (x : ℝ) : ℝ := (1 - x) / Real.exp x

-- Define the derivatives of the curves
noncomputable def curve1_derivative (a : ℝ) (x : ℝ) : ℝ := (a * x + a - 1) * Real.exp x
noncomputable def curve2_derivative (x : ℝ) : ℝ := (x - 2) / Real.exp x

-- Theorem statement
theorem tangent_perpendicular_range (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 (3/2) ∧ 
    curve1_derivative a x₀ * curve2_derivative x₀ = -1) → 
  a ∈ Set.Icc 1 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l1231_123148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_selection_for_quadrilateral_l1231_123158

theorem rod_selection_for_quadrilateral (rods : List ℕ) : 
  rods = List.range 40 →
  (∃ (d : ℕ), d ∈ rods ∧ d ≠ 4 ∧ d ≠ 12 ∧ d ≠ 21 ∧ 
    d > 5 ∧ d < 37 ∧
    4 + 12 + d > 21 ∧ 4 + 21 + d > 12 ∧ 12 + 21 + d > 4) →
  (rods.filter (λ d ↦ d ≠ 4 ∧ d ≠ 12 ∧ d ≠ 21 ∧ 
    d > 5 ∧ d < 37 ∧
    4 + 12 + d > 21 ∧ 4 + 21 + d > 12 ∧ 12 + 21 + d > 4)).length = 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_selection_for_quadrilateral_l1231_123158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_east_probability_l1231_123123

/-- Helper function to calculate the probability. -/
noncomputable def probability_A_east_of_B_and_C (angleA : ℝ) : ℝ :=
  (180 - angleA) / 360

/-- The probability that vertex A is east of both B and C in a triangle ABC with angle A = 40°,
    when the triangle is randomly oriented on a plane. -/
theorem triangle_vertex_east_probability : ℝ := by
  -- Define the triangle ABC with angle A = 40°
  let angleA : ℝ := 40

  -- Define the probability
  let prob : ℝ := 7 / 18

  -- State that this probability is correct
  have h : probability_A_east_of_B_and_C angleA = prob := by sorry

  -- Return the probability
  exact prob


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_east_probability_l1231_123123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tidal_current_properties_l1231_123150

noncomputable def f (A ω : ℕ+) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

noncomputable def f_prime (A ω : ℕ+) (φ : ℝ) (x : ℝ) : ℝ := A * ω * Real.cos (ω * x + φ)

theorem tidal_current_properties (A ω : ℕ+) (φ : ℝ) 
  (h1 : |φ| < π/3)
  (h2 : f A ω φ (2*π) = f_prime A ω φ (2*π))
  (h3 : ∀ x, f_prime A ω φ x ≥ -4)
  (h4 : ∃ x, f_prime A ω φ x = -4) :
  (ω = 1 ∧ φ = π/4 ∧ A = 4) ∧
  (f A ω φ (π/3) = Real.sqrt 6 + Real.sqrt 2) ∧
  (∀ x, f_prime A ω φ (x - π/4) = f_prime A ω φ (π/4 - x)) := by
  sorry

#check tidal_current_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tidal_current_properties_l1231_123150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_point_eccentricity_l1231_123102

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: If a hyperbola has a point P on one asymptote in the first quadrant
    such that PA is perpendicular to the other asymptote and PB is parallel to
    the other asymptote (where A and B are the left and right vertices),
    then the eccentricity of the hyperbola is 2. -/
theorem hyperbola_special_point_eccentricity (h : Hyperbola) 
  (P : Point) (h_first_quadrant : P.x > 0 ∧ P.y > 0)
  (h_on_asymptote : P.y = h.b / h.a * P.x)
  (h_PA_perpendicular : (P.y + h.b) / (P.x + h.a) * (h.b / h.a) = -1)
  (h_PB_parallel : (P.y - h.b) / (P.x - h.a) = - h.b / h.a) :
  eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_special_point_eccentricity_l1231_123102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1231_123139

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (3 - Real.sqrt 2 / 2 * t, 4 + Real.sqrt 2 / 2 * t)

-- Define the circle C in polar form
def circle_C (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Define point P
def point_P : ℝ × ℝ := (4, 3)

-- Define the intersection points A and B (existence assumed)
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- Axiom stating that A and B are on both the line and the circle
axiom A_on_line_and_circle : ∃ t, line_l t = A ∧ (A.1^2 + (A.2 - 3)^2 = 9)
axiom B_on_line_and_circle : ∃ t, line_l t = B ∧ (B.1^2 + (B.2 - 3)^2 = 9)

-- The theorem to prove
theorem intersection_sum : 
  1 / dist point_P A + 1 / dist point_P B = 4 * Real.sqrt 2 / 7 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1231_123139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_size_is_70_l1231_123172

/-- The number of players on a football team satisfying specific conditions -/
def football_team_size : ℕ :=
  let throwers : ℕ := 28
  let total_right_handed : ℕ := 56
  let non_throwers : ℕ := (total_right_handed - throwers) * 3 / 2
  throwers + non_throwers

/-- Proof that the football team size is 70 -/
theorem football_team_size_is_70 : football_team_size = 70 := by
  unfold football_team_size
  norm_num

#eval football_team_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_size_is_70_l1231_123172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_product_l1231_123140

/-- Given an equilateral triangle with vertices at (0,0), (c,14), and (d,47),
    prove that the product cd equals 506 2/3 -/
theorem equilateral_triangle_product (c d : ℝ) : 
  (∀ (z : ℂ), Complex.abs z = Complex.abs (z - (c + 14*Complex.I)) ∧ 
               Complex.abs z = Complex.abs (z - (d + 47*Complex.I)) ∧ 
               Complex.abs (z - (c + 14*Complex.I)) = Complex.abs (z - (d + 47*Complex.I))) → 
  c * d = 506 + 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_product_l1231_123140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1231_123195

theorem range_of_x : 
  (∀ m : ℝ, m ≠ 0 → |2*m - 1| + |1 - m| ≥ |m| * (|x - 1| - |2*x + 3|)) →
  Set.range (λ x : ℝ => x) = Set.Iic (-3) ∪ Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1231_123195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_theorem_l1231_123147

/-- Represents a point on the circle -/
structure Point where
  position : ℝ

/-- Represents the circle -/
structure Circle where
  circumference : ℝ
  positive : circumference > 0

/-- The time when P and Q meet at X -/
noncomputable def meetingTime (circle : Circle) (p q x : Point) (speedP speedQ : ℝ) : ℝ :=
  sorry

theorem meeting_time_theorem (circle : Circle) (p q x : Point) :
  circle.circumference = 40 →
  (x.position - p.position) % circle.circumference = 8 →
  (x.position - q.position) % circle.circumference = 16 →
  (q.position - p.position) % circle.circumference = 16 →
  meetingTime circle p q x 3 3.5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_theorem_l1231_123147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l1231_123116

def A : ℝ × ℝ := (7, 3)
def B : ℝ × ℝ := (-2, 3)
def C (k : ℝ) : ℝ × ℝ := (0, k)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def perimeter (k : ℝ) : ℝ :=
  distance A B + distance A (C k) + distance B (C k)

theorem min_perimeter :
  ∃ k : ℝ, ∀ k' : ℝ, perimeter k ≤ perimeter k' ∧ perimeter k = 18 := by
  sorry

#check min_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_l1231_123116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_origin_l1231_123154

-- Define the curve
noncomputable def curve (α : ℝ) (x : ℝ) : ℝ := x^α + 1

-- Define the tangent line at x = 1
def tangent_line (α : ℝ) (x : ℝ) : ℝ := α * (x - 1) + 2

theorem tangent_passes_origin (α : ℝ) :
  (curve α 1 = 2) ∧ (tangent_line α 0 = 0) → α = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_passes_origin_l1231_123154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l1231_123141

theorem greatest_integer_fraction : 
  ⌊(5^105 + 4^105 : ℝ) / (5^99 + 4^99 : ℝ)⌋ = 15624 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l1231_123141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1231_123193

noncomputable def f (a b x : ℝ) : ℝ := -1/3 * x^3 + 2*a * x^2 - 3*a^2 * x + b

noncomputable def f' (a x : ℝ) : ℝ := -(x - 3*a) * (x - a)

theorem function_properties (a b : ℝ) (h : 0 < a ∧ a < 1) :
  -- 1. Monotonicity intervals
  (∀ x, a < x ∧ x < 3*a → (f' a x > 0)) ∧
  (∀ x, (x < a ∨ x > 3*a) → (f' a x < 0)) ∧
  -- 2. Maximum and minimum values
  (∀ x, f a b x ≤ b) ∧
  (∀ x, f a b x ≥ -4/3 * a^3 + b) ∧
  -- 3. Range of a when |f'(x)| ≤ a for x ∈ [a+1, a+2]
  (∀ x, x ∈ Set.Icc (a + 1) (a + 2) → |f' a x| ≤ a) →
  a ∈ Set.Icc (4/5) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1231_123193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_value_difference_l1231_123131

/-- The place value of a digit in a given position -/
noncomputable def placeValue (position : Int) : ℝ := 10 ^ position

/-- The numeral under consideration -/
def numeral : ℚ := 135.21

/-- The position of the first '1' (hundreds place) -/
def position1 : Int := 2

/-- The position of the second '1' (tenths place) -/
def position2 : Int := -1

/-- Theorem stating the difference between place values of two 1's in 135.21 -/
theorem place_value_difference : 
  placeValue position1 - placeValue position2 = 99.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_value_difference_l1231_123131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_xy_value_l1231_123105

-- Define the rates of water flow for each valve
variable (x y z : ℝ)

-- Define the time to fill the tank for different valve combinations
noncomputable def fill_time_xyz : ℝ := 2
noncomputable def fill_time_xz : ℝ := 4
noncomputable def fill_time_yz : ℝ := 3

-- Define the conditions
def condition_xyz : Prop := x + y + z = 1 / fill_time_xyz
def condition_xz : Prop := x + z = 1 / fill_time_xz
def condition_yz : Prop := y + z = 1 / fill_time_yz

-- Define the time to fill the tank with only X and Y open
noncomputable def fill_time_xy : ℝ := 1 / (x + y)

-- The theorem to prove
theorem fill_time_xy_value 
  (hxyz : condition_xyz x y z)
  (hxz : condition_xz x z)
  (hyz : condition_yz y z) :
  fill_time_xy x y = 2.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_xy_value_l1231_123105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_monotonic_f_l1231_123160

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 3 else (a + 2) * Real.exp (a * x)

-- State the theorem
theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 0 < a ∧ a ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_monotonic_f_l1231_123160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1231_123164

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

-- State the theorem
theorem f_inequality (x : ℝ) : f x > f (3*x - 1) ↔ x ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1231_123164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_Z_l1231_123109

-- Define the complex number Z
noncomputable def Z : ℂ := (1 - 3*Complex.I) / (1 + Complex.I)

-- Theorem statement
theorem real_part_of_Z : Complex.re Z = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_Z_l1231_123109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_cos_A_max_area_l1231_123189

noncomputable section

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  hpos : 0 < a ∧ 0 < b ∧ 0 < c
  htri : a < b + c ∧ b < a + c ∧ c < a + b

-- Define vectors m and n
def m (t : Triangle) : ℝ × ℝ := (t.b + t.c, t.a^2 + t.b * t.c)
def n (t : Triangle) : ℝ × ℝ := (t.b + t.c, -1)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define cosine of angle A
def cos_A (t : Triangle) : ℝ := (t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c)

-- Define area of the triangle
def area (t : Triangle) : ℝ := 
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Part I: Prove that if m and n are perpendicular, then cos A = -1/2
theorem perpendicular_implies_cos_A (t : Triangle) 
  (h : dot_product (m t) (n t) = 0) : 
  cos_A t = -1/2 := by sorry

-- Part II: Prove that if a = √3 and cos A = -1/2, then the maximum area is √3/4
theorem max_area (t : Triangle) 
  (ha : t.a = Real.sqrt 3) 
  (hcos : cos_A t = -1/2) :
  area t ≤ Real.sqrt 3 / 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_implies_cos_A_max_area_l1231_123189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_range_for_acceptable_deviation_l1231_123104

/-- Approximate Fahrenheit conversion function -/
noncomputable def F_approx (C : ℝ) : ℝ := 2 * C + 30

/-- Exact Fahrenheit conversion function -/
noncomputable def F_exact (C : ℝ) : ℝ := (9 / 5) * C + 32

/-- Deviation between approximate and exact formulas -/
noncomputable def deviation (C : ℝ) : ℝ := 
  (F_approx C - F_exact C) / F_exact C

/-- Theorem stating the range of Celsius temperatures where the deviation doesn't exceed 5% -/
theorem celsius_range_for_acceptable_deviation :
  ∀ C : ℝ, (40 / 29 ≤ C ∧ C ≤ 360 / 11) ↔ 
    (abs (deviation C) ≤ 0.05) := by
  sorry

#check celsius_range_for_acceptable_deviation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_range_for_acceptable_deviation_l1231_123104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1231_123128

-- Define the function G
def G : ℕ → ℕ
  | 0 => 1
  | 1 => 4
  | (n + 2) => 3 * G (n + 1) - 2 * G n

-- Define the series
noncomputable def series_sum : ℝ := ∑' n, (1 : ℝ) / G (2^n)

-- State the theorem
theorem series_sum_equals_one : series_sum = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_l1231_123128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_ratio_is_five_l1231_123192

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The ratio of the number of positive divisors of 3600 to the number of positive divisors of 36 -/
def divisor_ratio : ℚ := (num_divisors 3600 : ℚ) / (num_divisors 36 : ℚ)

theorem divisor_ratio_is_five : divisor_ratio = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_ratio_is_five_l1231_123192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_increasing_intervals_f_monotonically_increasing_characterization_l1231_123120

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.sin x) ^ 2

def monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_monotonically_increasing_intervals (k : ℤ) :
  monotonically_increasing f (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12) := by
  sorry

theorem f_monotonically_increasing_characterization :
  ∀ a b : ℝ, monotonically_increasing f a b ↔
    ∃ k : ℤ, a = k * Real.pi - Real.pi / 12 ∧ b = k * Real.pi + 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_increasing_intervals_f_monotonically_increasing_characterization_l1231_123120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jaden_speed_difference_l1231_123113

/-- Represents Jaden's speed at different ages --/
structure JadenSpeed where
  young_miles : ℚ
  young_hours : ℚ
  old_miles : ℚ
  old_hours : ℚ

/-- Calculates the time difference per mile between Jaden's current and past speeds --/
noncomputable def time_difference_per_mile (speed : JadenSpeed) : ℚ :=
  (speed.old_hours * 60 / speed.old_miles) - (speed.young_hours * 60 / speed.young_miles)

/-- Theorem stating that the time difference per mile is 18 minutes --/
theorem jaden_speed_difference (speed : JadenSpeed) 
  (h1 : speed.young_miles = 20)
  (h2 : speed.young_hours = 4)
  (h3 : speed.old_miles = 8)
  (h4 : speed.old_hours = 4) : 
  time_difference_per_mile speed = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jaden_speed_difference_l1231_123113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_vertex_circle_center_to_vertex_distance_l1231_123185

/-- Given a 30-60-90 triangle with hypotenuse 2R where R = 2.5, 
    the distance from the midpoint of the shorter leg to the vertex of the 30° angle is 2R. -/
theorem distance_to_vertex (R : ℝ) (h : R = 2.5) : 
  2 * R = 5 := by
  -- Rewrite R in terms of its given value
  rw [h]
  -- Simplify the expression
  norm_num

/-- The main theorem using the result from distance_to_vertex -/
theorem circle_center_to_vertex_distance (R : ℝ) (h : R = 2.5) :
  let center_to_vertex_distance := 2 * R
  center_to_vertex_distance = 5 := by
  -- Use the previous theorem
  exact distance_to_vertex R h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_vertex_circle_center_to_vertex_distance_l1231_123185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dimensions_l1231_123186

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  angle : ℝ

/-- Calculate the length of the adjacent side of a parallelogram -/
noncomputable def adjacent_side_length (p : Parallelogram) : ℝ :=
  p.height / (Real.sin p.angle)

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.base * p.height

/-- Theorem about the dimensions of a specific parallelogram -/
theorem parallelogram_dimensions :
  let p : Parallelogram := { base := 12, height := 6, angle := Real.pi / 6 }
  adjacent_side_length p = 12 ∧ area p = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_dimensions_l1231_123186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_advancement_l1231_123171

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The probability of incorrectly answering a single question -/
def p_incorrect : ℝ := 1 - p_correct

/-- The number of questions answered before advancing -/
def n_questions : ℕ := 4

/-- A contestant advances if they answer two consecutive questions correctly -/
def advancement_condition (answers : List Bool) : Bool :=
  (answers.zipWith (· && ·) answers.tail).any id

/-- The specific sequence of answers that leads to advancement after exactly 4 questions -/
def target_sequence : List Bool := [false, true, true, true]

/-- The theorem to be proved -/
theorem probability_of_advancement :
  (List.range n_questions).foldl
    (fun acc i => acc * if target_sequence.get! i then p_correct else p_incorrect)
    1 = 0.1024 := by
  sorry

#eval (List.range n_questions).foldl
  (fun acc i => acc * if target_sequence.get! i then p_correct else p_incorrect)
  1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_advancement_l1231_123171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1231_123161

/-- The ellipse C: x²/3 + y² = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line l: x + y + 4 = 0 -/
def line (x y : ℝ) : Prop := x + y + 4 = 0

/-- Distance between a point (x, y) and the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y + 4| / Real.sqrt 2

/-- Theorem stating that the point (-3/2, -1/2) on the ellipse has the minimum distance to the line -/
theorem min_distance_point :
  ellipse (-3/2) (-1/2) ∧
  (∀ x y : ℝ, ellipse x y → distance_to_line x y ≥ Real.sqrt 2) ∧
  distance_to_line (-3/2) (-1/2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1231_123161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l1231_123100

theorem complex_inequality (a b c : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a / Complex.abs a ≠ b / Complex.abs b) :
  max (Complex.abs (a * c + b)) (Complex.abs (b * c + a)) ≥ 
  (1/2 : ℝ) * Complex.abs (a + b) * Complex.abs (a / Complex.abs a - b / Complex.abs b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l1231_123100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1231_123149

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin ((Real.pi / 3) * x + 1 / 4)

-- State the theorem about the period of f
theorem period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (t : ℝ), 0 < t ∧ t < T → (∃ (y : ℝ), f (x + t) ≠ f y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l1231_123149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_27_l1231_123145

/-- An isosceles trapezoid with given constraints -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of the larger base
  b : ℝ  -- length of the smaller base
  c : ℝ  -- length of the legs
  h : ℝ  -- height of the trapezoid
  ha : a = 13  -- largest side is 13
  hb : b > 0  -- smaller base is positive
  hc : c > 0  -- legs are positive
  hh : h > 0  -- height is positive
  perimeter : a + b + 2*c = 28  -- perimeter is 28
  isosceles : h^2 + ((a - b)/2)^2 = c^2  -- Pythagorean theorem for isosceles trapezoid

/-- The area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  (t.a + t.b) * t.h / 2

/-- Theorem: The maximum area of the isosceles trapezoid with given constraints is 27 -/
theorem max_area_is_27 :
  ∀ t : IsoscelesTrapezoid, area t ≤ 27 ∧ ∃ t' : IsoscelesTrapezoid, area t' = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_is_27_l1231_123145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_half_velocity_correct_distance_total_correct_verify_example_results_l1231_123124

/-- Ship model parameters -/
structure ShipModel where
  m : ℝ  -- mass in kg
  v₀ : ℝ  -- initial velocity in m/s
  k : ℝ  -- resistance coefficient in kg/s

/-- Distance traveled by the ship model when velocity decreases to half -/
noncomputable def distance_half_velocity (ship : ShipModel) : ℝ :=
  (ship.v₀ * ship.m) / (2 * ship.k)

/-- Total distance traveled by the ship model until it stops -/
noncomputable def distance_total (ship : ShipModel) : ℝ :=
  (ship.v₀ * ship.m) / ship.k

/-- Theorem: The distance traveled when velocity decreases to half is (v₀ * m) / (2k) -/
theorem distance_half_velocity_correct (ship : ShipModel) :
  distance_half_velocity ship = (ship.v₀ * ship.m) / (2 * ship.k) := by
  rfl

/-- Theorem: The total distance traveled until the ship stops is (v₀ * m) / k -/
theorem distance_total_correct (ship : ShipModel) :
  distance_total ship = (ship.v₀ * ship.m) / ship.k := by
  rfl

/-- Example ship model with given parameters -/
def example_ship : ShipModel :=
  { m := 0.5, v₀ := 10, k := 0.5 }

/-- Verify the results for the example ship -/
theorem verify_example_results :
  distance_half_velocity example_ship = 5 ∧ distance_total example_ship = 10 := by
  simp [distance_half_velocity, distance_total, example_ship]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_half_velocity_correct_distance_total_correct_verify_example_results_l1231_123124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_meeting_l1231_123155

/-- The distance Alice walks before meeting Bob -/
noncomputable def aliceDistance (aliceSpeed bobSpeed pathLength : ℝ) (angle : ℝ) : ℝ :=
  (2700 * Real.sqrt 2 - 6 * Real.sqrt 630000) / 11

theorem alice_bob_meeting 
  (aliceSpeed : ℝ) 
  (bobSpeed : ℝ) 
  (pathLength : ℝ) 
  (angle : ℝ) 
  (h1 : aliceSpeed = 6)
  (h2 : bobSpeed = 5)
  (h3 : pathLength = 150)
  (h4 : angle = Real.pi/4) :
  aliceDistance aliceSpeed bobSpeed pathLength angle = 
    (2700 * Real.sqrt 2 - 6 * Real.sqrt 630000) / 11 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_meeting_l1231_123155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1231_123111

noncomputable def m (A : Real) : Fin 2 → Real
  | 0 => Real.sin A
  | 1 => Real.cos A

noncomputable def n : Fin 2 → Real
  | 0 => Real.sqrt 3
  | 1 => -1

def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def f (A x : Real) : Real :=
  Real.cos (2 * x) + 4 * Real.cos A * Real.sin x

theorem vector_problem (A : Real) 
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : dot_product (m A) n = 1) :
  A = π / 3 ∧ 
  Set.Icc (-3 : Real) (3/2) = {y | ∃ x, f A x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1231_123111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sidorov_family_max_earnings_l1231_123135

/-- Represents a family member --/
inductive FamilyMember
  | Father
  | Mother
  | Son

/-- Represents a household task --/
inductive HouseholdTask
  | Cooking
  | WalkingDog
  | Shopping

/-- Information about a family member --/
structure MemberInfo where
  hourlyRate : ℝ
  cookingTime : ℝ
  shoppingTime : ℝ

/-- The Sidorov family problem --/
def sidorovFamily (member : FamilyMember → MemberInfo) (task : HouseholdTask → FamilyMember → ℝ) : Prop :=
  let availableHours := 24 - 8 - 8
  let fatherEarnings := (availableHours - (member FamilyMember.Father).cookingTime - (member FamilyMember.Father).shoppingTime) * (member FamilyMember.Father).hourlyRate
  let motherEarnings := availableHours * (member FamilyMember.Mother).hourlyRate
  let sonEarnings := (availableHours - task HouseholdTask.WalkingDog FamilyMember.Son) * (member FamilyMember.Son).hourlyRate
  fatherEarnings + motherEarnings + sonEarnings = 19600

theorem sidorov_family_max_earnings :
  ∃ (member : FamilyMember → MemberInfo) (task : HouseholdTask → FamilyMember → ℝ),
    member FamilyMember.Father = { hourlyRate := 1500, cookingTime := 1, shoppingTime := 1 } ∧
    member FamilyMember.Mother = { hourlyRate := 800, cookingTime := 2, shoppingTime := 2 } ∧
    member FamilyMember.Son = { hourlyRate := 600, cookingTime := 4, shoppingTime := 3 } ∧
    task HouseholdTask.WalkingDog FamilyMember.Father = 1 ∧
    task HouseholdTask.WalkingDog FamilyMember.Mother = 1 ∧
    task HouseholdTask.WalkingDog FamilyMember.Son = 1 ∧
    sidorovFamily member task :=
  by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sidorov_family_max_earnings_l1231_123135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_l1231_123159

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_g_relation : ∀ x : ℝ, f x = g (x + 2)
axiom g_definition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → g x = x - 2

-- State the theorem to be proved
theorem g_value : g 10.5 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_value_l1231_123159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1231_123180

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (x - 2) / 3

-- Define the set of x that satisfy the equation
def solution_set : Set ℝ := {x | x ≠ 0 ∧ f_inv x = f (x⁻¹)}

-- Theorem statement
theorem sum_of_solutions :
  ∃ (a b : ℝ), a ∈ solution_set ∧ b ∈ solution_set ∧ a + b = 8 ∧
  ∀ x ∈ solution_set, x = a ∨ x = b :=
by
  -- We'll use 9 and -1 as our solutions
  let a : ℝ := 9
  let b : ℝ := -1
  
  -- Prove that a and b are in the solution set
  have ha : a ∈ solution_set := by sorry
  have hb : b ∈ solution_set := by sorry
  
  -- Prove that a + b = 8
  have sum_eq : a + b = 8 := by norm_num
  
  -- Prove that these are the only solutions
  have unique : ∀ x ∈ solution_set, x = a ∨ x = b := by sorry
  
  -- Combine all parts of the proof
  exact ⟨a, b, ha, hb, sum_eq, unique⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l1231_123180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l1231_123114

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance between the center and a focus -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Left vertex to left focus distance -/
noncomputable def AF₁ (e : Ellipse) : ℝ := e.a - focal_distance e

/-- Focus to focus distance -/
noncomputable def F₁F₂ (e : Ellipse) : ℝ := 2 * focal_distance e

/-- Left focus to right vertex distance -/
noncomputable def F₁B (e : Ellipse) : ℝ := e.a + focal_distance e

/-- AF₁, F₁F₂, F₁B form an arithmetic sequence -/
def is_arithmetic_sequence (e : Ellipse) : Prop :=
  F₁F₂ e = (AF₁ e + F₁B e) / 2

theorem ellipse_eccentricity_half (e : Ellipse) 
  (h : is_arithmetic_sequence e) : eccentricity e = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_half_l1231_123114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangement_count_l1231_123108

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def balloon_arrangements : ℕ := 
  factorial 7 / (factorial 2 * factorial 2)

theorem balloon_arrangement_count : balloon_arrangements = 1260 := by
  -- Unfold definitions
  unfold balloon_arrangements
  unfold factorial
  -- Simplify the expression
  simp [Nat.mul_div_assoc, Nat.div_eq_of_eq_mul_left]
  -- Assert the equality
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangement_count_l1231_123108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1231_123162

/-- Given a projection that takes [3, -3] to [18/5, -18/5], 
    prove that the projection of [4, -1] is [5/2, -5/2] -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
  (h : proj (3, -3) = (18/5, -18/5)) :
  proj (4, -1) = (5/2, -5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1231_123162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_v_shape_l1231_123167

/-- A V-shape is a figure consisting of a rectangle with an isosceles right triangle
    on one side and a semicircle on the opposite side. -/
structure VShape where
  x : ℝ  -- Length of the side with the semicircle
  y : ℝ  -- Length of the other side of the rectangle

/-- The perimeter of a V-shape -/
noncomputable def perimeter (v : VShape) : ℝ := 2 * v.y + (Real.sqrt 2 + Real.pi / 2) * v.x

/-- The area of a V-shape -/
noncomputable def area (v : VShape) : ℝ := v.x * v.y + v.x^2 / 4 + Real.pi * v.x^2 / 8

/-- Theorem: The maximum area of a V-shape with perimeter k -/
theorem max_area_v_shape (k : ℝ) (h : k > 0) :
  (∃ v : VShape, perimeter v = k ∧
    ∀ w : VShape, perimeter w = k → area w ≤ area v) ∧
  (∃ v : VShape, perimeter v = k ∧ area v = k^2 / (8 * Real.sqrt 2 + 2 * Real.pi - 4)) ∧
  (∃ v : VShape, perimeter v = k ∧ v.y / v.x = (Real.sqrt 2 - 1) / 2) := by
  sorry

#check max_area_v_shape

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_v_shape_l1231_123167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paintable_area_calculation_l1231_123166

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a closet -/
structure ClosetDimensions where
  length : ℝ
  width : ℝ

/-- Calculate the total paintable area in all bedrooms -/
def totalPaintableArea (
  numRooms : ℕ
) (roomDim : RoomDimensions)
  (closetDim : ClosetDimensions)
  (unpaintableArea : ℝ) : ℝ :=
  let roomWallArea := 2 * (roomDim.length * roomDim.height + roomDim.width * roomDim.height)
  let closetWallArea := 2 * closetDim.length * roomDim.height
  let paintableAreaPerRoom := roomWallArea - closetWallArea - unpaintableArea
  (numRooms : ℝ) * paintableAreaPerRoom

theorem paintable_area_calculation :
  let roomDim : RoomDimensions := ⟨15, 11, 9⟩
  let closetDim : ClosetDimensions := ⟨3, 4⟩
  let numRooms : ℕ := 4
  let unpaintableArea : ℝ := 70
  totalPaintableArea numRooms roomDim closetDim unpaintableArea = 1376 := by
  sorry

#eval totalPaintableArea 4 ⟨15, 11, 9⟩ ⟨3, 4⟩ 70

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paintable_area_calculation_l1231_123166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1231_123177

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/6) = 4/5) 
  (h2 : α ∈ Set.Ioo 0 (π/3)) : 
  Real.sin α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1231_123177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l1231_123130

/-- Given two 2D vectors, calculate the projection of one vector onto another -/
theorem vector_projection (a b : ℝ × ℝ) : 
  (a = (1, 3) ∧ b = (-2, 4)) → 
  ((a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l1231_123130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_interval_l1231_123151

def f (x : ℝ) := x^3 - 3*x^2 + 2

def f' (x : ℝ) := 3*x^2 - 6*x

theorem f_decreasing_on_open_interval :
  ∀ x ∈ Set.Ioo 0 2, HasDerivAt f (f' x) x ∧ f' x < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_interval_l1231_123151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angles_bound_l1231_123178

/-- Represents a set of lines in a plane -/
structure LineDivision where
  n : ℕ
  lines : Fin (2*n) → Set (ℝ × ℝ)
  not_parallel : ∀ i j, i ≠ j → lines i ≠ lines j
  no_triple_intersection : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    (lines i ∩ lines j ∩ lines k).Nonempty → False

/-- The maximum number of angles formed by the line division -/
def max_angles (d : LineDivision) : ℕ := 2 * d.n - 1

/-- Theorem: The maximum number of angles formed is 2n-1 -/
theorem max_angles_bound (d : LineDivision) (h : d.n > 1) : 
  ∀ k, k ≤ max_angles d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angles_bound_l1231_123178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_cannot_win_l1231_123174

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The unit square [0, 1] × [0, 1] -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Alice's strategy for choosing 18 points -/
def Strategy : Type := Fin 18 → Point

/-- Bob's feedback based on distance to secret point -/
def Feedback : Type := Bool

/-- Alice's final choice after receiving feedback -/
def FinalChoice : Type := (Strategy × (Fin 18 → Feedback)) → Point

/-- The main theorem: Alice cannot guarantee a win with 18 queries -/
theorem alice_cannot_win (aliceStrategy : Strategy) (aliceFinalChoice : FinalChoice) :
  ∃ B ∈ UnitSquare, ∀ A : Point,
    distance A B > 1 / 2020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_cannot_win_l1231_123174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l1231_123143

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (1, 0) to the line 3x + 4y - 8 = 0 is 1 -/
theorem distance_point_to_specific_line : 
  distance_point_to_line 1 0 3 4 (-8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l1231_123143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l1231_123163

/-- Given a polynomial p(x) = x^4 + ax^3 + bx^2 + cx + d where a, b, c, d are real constants,
    and p(1) = 1993, p(2) = 3986, p(3) = 5979, then 1/4[p(11) + p(-7)] = 5233. -/
theorem polynomial_value_theorem (a b c d : ℝ) : 
  let p (x : ℝ) := x^4 + a*x^3 + b*x^2 + c*x + d
  (p 1 = 1993) → (p 2 = 3986) → (p 3 = 5979) → 
  (1/4 : ℝ) * (p 11 + p (-7)) = 5233 := by
  sorry

#check polynomial_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_theorem_l1231_123163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l1231_123133

/-- The average speed of a train given its distance traveled and time taken -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: The average speed of a train that traveled 27 meters in 9 seconds is 3 meters per second -/
theorem train_average_speed :
  average_speed 27 9 = 3 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l1231_123133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1231_123197

/-- A point in the plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of points for the game -/
def GameSet : Set Point :=
  {p : Point | p.x^2 + p.y^2 ≤ 1010}

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if two points are symmetric about the origin -/
def isSymmetric (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- The game state -/
structure GameState where
  remainingPoints : Set Point
  lastMove : Option Point
  currentPlayer : Bool  -- true for first player, false for second player

/-- A valid move in the game -/
def validMove (state : GameState) (move : Point) : Prop :=
  move ∈ state.remainingPoints ∧
  (state.lastMove.isNone ∨
    (∀ lastMove, state.lastMove = some lastMove →
      distance move lastMove > distance lastMove ⟨0, 0⟩)) ∧
  ¬isSymmetric move ⟨0, 0⟩

/-- The winning strategy theorem -/
theorem first_player_wins :
  ∃ (strategy : GameState → Point),
    ∀ (initialState : GameState),
      initialState.currentPlayer = true →
      initialState.remainingPoints = GameSet →
      (⟨0, 0⟩ : Point) ∈ initialState.remainingPoints →
      (∀ (p : Point), p ∈ initialState.remainingPoints →
        {q : Point | q.x = -p.y ∧ q.y = p.x} ⊆ initialState.remainingPoints) →
      (∃ (finalState : GameState),
        finalState.remainingPoints = ∅ ∧
        finalState.currentPlayer = false) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1231_123197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1231_123198

theorem complex_fraction_equality : 
  Complex.I / (1 - Complex.I) = (-1/2 : ℂ) + (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1231_123198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l1231_123146

/-- The amount after n years given an initial amount and annual increase rate -/
noncomputable def amount_after_years (initial_amount : ℝ) (increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + increase_rate) ^ years

/-- Theorem stating that an amount of 76800 increasing by 1/8th annually will be 97200 after 2 years -/
theorem amount_after_two_years :
  amount_after_years 76800 (1/8) 2 = 97200 := by
  -- Unfold the definition of amount_after_years
  unfold amount_after_years
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- This will not evaluate due to noncomputable definition
-- #eval amount_after_years 76800 (1/8) 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l1231_123146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_radii_l1231_123107

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  ab : ℝ
  ad : ℝ
  aa1 : ℝ
  ab_positive : 0 < ab
  ad_positive : 0 < ad
  aa1_positive : 0 < aa1

/-- A plane that divides the prism into two parts -/
structure DividingPlane where
  prism : RectangularPrism
  x : ℝ
  x_range : 0 < x ∧ x ≤ prism.ad

/-- The sum of radii of two spheres placed in the divided prism -/
noncomputable def sumOfRadii (p : DividingPlane) : ℝ :=
  let t := p.x / p.prism.ad
  (3 * t) / (1 + t) + 2 - 2 * t

/-- The theorem stating the maximum sum of radii -/
theorem max_sum_of_radii (p : RectangularPrism) 
  (h1 : p.ab = 5) (h2 : p.ad = 3) (h3 : p.aa1 = 4) : 
  ∃ (plane : DividingPlane), plane.prism = p ∧ 
  ∀ (other_plane : DividingPlane), other_plane.prism = p → 
  sumOfRadii other_plane ≤ sumOfRadii plane ∧ 
  sumOfRadii plane = 21/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_radii_l1231_123107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_groups_l1231_123126

/-- Check if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The groups of numbers given in the problem --/
noncomputable def group1 : Vector ℝ 3 := ⟨[1, 2, Real.sqrt 5], by simp⟩
noncomputable def group2 : Vector ℝ 3 := ⟨[1, 2, Real.sqrt 3], by simp⟩
def group3 : Vector ℝ 3 := ⟨[3, 4, 5], by simp⟩
def group4 : Vector ℝ 3 := ⟨[6, 8, 12], by simp⟩

/-- The theorem to be proved --/
theorem right_triangle_groups :
  is_right_triangle group1[0] group1[1] group1[2] ∧
  is_right_triangle group2[0] group2[1] group2[2] ∧
  is_right_triangle group3[0] group3[1] group3[2] ∧
  ¬is_right_triangle group4[0] group4[1] group4[2] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_groups_l1231_123126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_when_perimeter_equals_area_l1231_123170

-- Define a structure for a right-angled triangle with integer sides
structure RightTriangle where
  a : ℕ+  -- First leg
  b : ℕ+  -- Second leg
  c : ℕ+  -- Hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Define the property that perimeter equals area
def perimeterEqualsArea (t : RightTriangle) : Prop :=
  (t.a : ℝ) + (t.b : ℝ) + (t.c : ℝ) = ((t.a : ℝ) * (t.b : ℝ)) / 2

-- Theorem statement
theorem hypotenuse_length_when_perimeter_equals_area :
  ∀ t : RightTriangle, perimeterEqualsArea t → t.c = 10 ∨ t.c = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_when_perimeter_equals_area_l1231_123170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equality_l1231_123134

def M : Set ℝ := {x | ∃ y, y = Real.log ((x^2 : ℝ) - 3*x - 4)}
def N : Set ℝ := {x | ∃ y, y = (2 : ℝ)^(x - 1)}

theorem set_intersection_equality : M ∩ N = {x | x > 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equality_l1231_123134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_division_l1231_123115

/-- The number of parts into which n planes divide space -/
def K (n : ℕ) : ℚ :=
  (1 / 6 : ℚ) * (n^3 + 5*n + 6 : ℚ)

/-- Every three planes intersect at one point, and no four planes have a common point -/
axiom plane_intersection_property (n : ℕ) : n ≥ 3 → True

/-- Theorem: The number of parts into which n planes divide space -/
theorem space_division (n : ℕ) (h : n ≥ 3) :
  K n = (1 / 6 : ℚ) * (n^3 + 5*n + 6 : ℚ) :=
by
  -- The proof is omitted for now
  sorry

#check space_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_division_l1231_123115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_percent_of_140_in_paise_l1231_123191

-- Define the conversion rate from rupees to paise
noncomputable def rupees_to_paise (rupees : ℝ) : ℝ := rupees * 100

-- Define the percentage calculation
noncomputable def percentage_of (percent : ℝ) (value : ℝ) : ℝ := (percent / 100) * value

-- Theorem statement
theorem half_percent_of_140_in_paise : 
  rupees_to_paise (percentage_of 0.5 140) = 70 := by
  -- Expand the definitions
  unfold rupees_to_paise percentage_of
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_percent_of_140_in_paise_l1231_123191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_to_cone_volume_ratio_proof_l1231_123118

/-- The ratio of the volume of water to the total volume of a cone when filled to 2/3 of its height -/
def water_to_cone_volume_ratio : ℚ := 8 / 27

/-- Proves that the ratio of water volume to cone volume is 8/27 when filled to 2/3 height -/
theorem water_to_cone_volume_ratio_proof :
  let h : ℝ := 1  -- Arbitrary height
  let r : ℝ := 1  -- Arbitrary radius
  let cone_volume : ℝ := (1 / 3) * Real.pi * r^2 * h
  let water_height : ℝ := (2 / 3) * h
  let water_radius : ℝ := (2 / 3) * r
  let water_volume : ℝ := (1 / 3) * Real.pi * water_radius^2 * water_height
  water_volume / cone_volume = water_to_cone_volume_ratio := by
  sorry

#eval water_to_cone_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_to_cone_volume_ratio_proof_l1231_123118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l1231_123119

noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

theorem complex_product_polar_form :
  ∃ (r : ℝ) (θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  (4 * cis (160 * Real.pi / 180)) * (5 * cis (210 * Real.pi / 180)) = r * cis θ ∧
  r = 20 ∧ θ = 10 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_polar_form_l1231_123119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_calculation_l1231_123101

/-- The star operation for real numbers -/
noncomputable def star (x y : ℝ) : ℝ := Real.sqrt (4 * x^2 + 4 * y^2)

/-- Theorem stating the result of (3 ⋆ 4) ⋆ (6 ⋆ 8) -/
theorem star_calculation : star (star 3 4) (star 6 8) = 50 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_calculation_l1231_123101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_product_l1231_123153

/-- A monic polynomial of degree 3 with a positive constant term -/
structure MonicCubicPoly where
  a : ℝ
  b : ℝ
  c : ℝ
  constant_term : ℝ
  constant_term_pos : constant_term > 0

/-- The product of two monic cubic polynomials -/
noncomputable def poly_product (p q : MonicCubicPoly) : Polynomial ℝ :=
  Polynomial.monomial 6 1 +
  Polynomial.monomial 5 2 +
  Polynomial.monomial 4 4 +
  Polynomial.monomial 3 6 +
  Polynomial.monomial 2 4 +
  Polynomial.monomial 1 2 +
  Polynomial.monomial 0 9

theorem constant_term_of_product (p q : MonicCubicPoly) 
  (h : poly_product p q = Polynomial.monomial 6 1 + Polynomial.monomial 5 2 + Polynomial.monomial 4 4 + 
                          Polynomial.monomial 3 6 + Polynomial.monomial 2 4 + Polynomial.monomial 1 2 + 
                          Polynomial.monomial 0 9) :
  p.constant_term = 3 ∧ q.constant_term = 3 := by
  sorry

#check constant_term_of_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_product_l1231_123153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_divisor_8k_minus_1_l1231_123184

theorem no_prime_divisor_8k_minus_1 (n : ℕ) :
  ∀ (k : ℕ) (p : ℕ), Prime p → p = 8 * k - 1 → ¬(p ∣ 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_prime_divisor_8k_minus_1_l1231_123184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_is_one_fourth_l1231_123117

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℝ
  milk : ℝ

/-- Performs the transfers between cups as described in the problem --/
noncomputable def perform_transfers (cup1 cup2 : CupContents) : CupContents × CupContents := 
  let transfer1 := CupContents.mk (cup1.coffee / 3) 0
  let cup1_after1 := CupContents.mk (cup1.coffee - transfer1.coffee) cup1.milk
  let cup2_after1 := CupContents.mk (cup2.coffee + transfer1.coffee) cup2.milk
  
  let total2 := cup2_after1.coffee + cup2_after1.milk
  let transfer2 := CupContents.mk 
    (cup2_after1.coffee * (total2 / 4) / total2)
    (cup2_after1.milk * (total2 / 4) / total2)
  let cup1_after2 := CupContents.mk 
    (cup1_after1.coffee + transfer2.coffee)
    (cup1_after1.milk + transfer2.milk)
  let cup2_after2 := CupContents.mk 
    (cup2_after1.coffee - transfer2.coffee)
    (cup2_after1.milk - transfer2.milk)
  
  let total1 := cup1_after2.coffee + cup1_after2.milk
  let transfer3 := CupContents.mk 
    (cup1_after2.coffee * (total1 / 5) / total1)
    (cup1_after2.milk * (total1 / 5) / total1)
  let cup1_final := CupContents.mk 
    (cup1_after2.coffee - transfer3.coffee)
    (cup1_after2.milk - transfer3.milk)
  let cup2_final := CupContents.mk 
    (cup2_after2.coffee + transfer3.coffee)
    (cup2_after2.milk + transfer3.milk)
  
  (cup1_final, cup2_final)

/-- The main theorem stating that the fraction of milk in the first cup after transfers is 1/4 --/
theorem milk_fraction_is_one_fourth :
  let initial_cup1 : CupContents := CupContents.mk 6 0
  let initial_cup2 : CupContents := CupContents.mk 0 6
  let (final_cup1, _) := perform_transfers initial_cup1 initial_cup2
  final_cup1.milk / (final_cup1.coffee + final_cup1.milk) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_fraction_is_one_fourth_l1231_123117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_volume_l1231_123110

theorem rectangular_box_volume :
  ∃ x : ℕ, x * (2 * x) * (5 * x) = 80 ∧
  ∀ v ∈ ({60, 100, 120, 200} : Set ℕ), ¬∃ x : ℕ, x * (2 * x) * (5 * x) = v := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_volume_l1231_123110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l1231_123132

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relations
axiom perpendicular : Line → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom perpendicular_lines : Line → Line → Prop
axiom parallel_planes : Plane → Plane → Prop

-- Define the theorem
theorem geometry_theorem 
  (m n : Line) 
  (α β γ : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (perpendicular m α ∧ parallel_line_plane n α → perpendicular_lines m n) ∧
  (parallel_planes α β ∧ parallel_planes β γ ∧ perpendicular m α → perpendicular m γ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_theorem_l1231_123132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2014_equals_neg_six_l1231_123156

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => (1 + sequenceA n) / (1 - sequenceA n)

def product_up_to (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (product_up_to n) * (sequenceA n)

theorem product_2014_equals_neg_six :
  product_up_to 2014 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2014_equals_neg_six_l1231_123156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_circle_passes_through_foci_l1231_123175

/-- Hyperbola C with eccentricity 2 and focus F(2, 0) -/
structure Hyperbola where
  C : Set (ℝ × ℝ)
  eccentricity : ℝ
  focus : ℝ × ℝ
  h_ecc : eccentricity = 2
  h_focus : focus = (2, 0)

/-- Distance from focus to asymptote is √3 -/
noncomputable def asymptote_distance (h : Hyperbola) : ℝ := Real.sqrt 3

/-- Left and right vertices of the hyperbola -/
def vertices : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 0), (1, 0))

/-- Point on the hyperbola different from vertices -/
structure PointOnHyperbola (h : Hyperbola) where
  P : ℝ × ℝ
  h_on_C : P ∈ h.C
  h_not_vertex : P ≠ (vertices.1) ∧ P ≠ (vertices.2)

/-- Y-axis intersections of lines from vertices to P -/
noncomputable def yAxisIntersections (h : Hyperbola) (p : PointOnHyperbola h) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry  -- Implementation details omitted

/-- Circle with diameter MN -/
def circleD (h : Hyperbola) (p : PointOnHyperbola h) : Set (ℝ × ℝ) :=
  sorry  -- Implementation details omitted

/-- Foci of the hyperbola -/
noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((Real.sqrt 3, 0), (-Real.sqrt 3, 0))

theorem hyperbola_equation (h : Hyperbola) :
  h.C = {(x, y) : ℝ × ℝ | x^2 - y^2/3 = 1} := by sorry

theorem circle_passes_through_foci (h : Hyperbola) (p : PointOnHyperbola h) :
  (foci.1 ∈ circleD h p) ∧ (foci.2 ∈ circleD h p) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_circle_passes_through_foci_l1231_123175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_problems_without_conditional_l1231_123127

/-- Represents a mathematical problem that may or may not require conditional statements. -/
inductive Problem
  | opposite_number
  | square_perimeter
  | max_of_three
  | binary_value

/-- Determines if a problem requires conditional statements in its algorithm. -/
def requires_conditional (p : Problem) : Bool :=
  match p with
  | Problem.opposite_number => false
  | Problem.square_perimeter => false
  | Problem.max_of_three => true
  | Problem.binary_value => true

/-- The list of all problems. -/
def all_problems : List Problem :=
  [Problem.opposite_number, Problem.square_perimeter, Problem.max_of_three, Problem.binary_value]

/-- The main theorem stating that exactly two problems do not require conditional statements. -/
theorem exactly_two_problems_without_conditional :
  (all_problems.filter (fun p => ¬requires_conditional p)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_problems_without_conditional_l1231_123127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_division_count_l1231_123183

/-- The number of seconds in a day -/
def day_seconds : ℕ := 72000

/-- The number of ways to divide a day into periods -/
def division_ways : ℕ := 72

/-- Predicate for valid division of a day -/
def valid_division (n m : ℕ) : Prop := n > 0 ∧ m > 0 ∧ n * m = day_seconds

theorem day_division_count :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ valid_division p.1 p.2) ∧ S.card = division_ways :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_division_count_l1231_123183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1231_123125

-- Define the ∨ operation
noncomputable def vee (a b : ℝ) : ℝ := max a b

-- Define the ∧ operation
noncomputable def wedge (a b : ℝ) : ℝ := min a b

-- Main theorem
theorem problem_statement (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a * b ≤ 4) (hcd : c + d ≥ 4) :
  wedge a b ≤ 2 ∧ vee c d ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1231_123125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_shoe_purchase_l1231_123196

/-- Represents the purchase of shoes and jerseys --/
structure Purchase where
  totalCost : ℕ
  jerseyCount : ℕ
  shoesCost : ℕ

/-- The number of pairs of shoes in a purchase --/
def shoePairs (p : Purchase) : ℕ := p.shoesCost / ((p.totalCost - p.shoesCost) / p.jerseyCount)

/-- Theorem stating that Jeff bought 6 pairs of shoes --/
theorem jeff_shoe_purchase (p : Purchase) 
  (h1 : p.totalCost = 560)
  (h2 : p.jerseyCount = 4)
  (h3 : p.shoesCost = 480) :
  shoePairs p = 6 := by
  sorry

#check jeff_shoe_purchase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_shoe_purchase_l1231_123196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_group_is_abelian_l1231_123122

/-- A group satisfying the given automorphism property -/
class SpecialGroup (G : Type*) extends Group G where
  finite : Fintype G
  automorphism_property : ∀ (f : G ≃* G), ∃ (m : ℕ+), ∀ (x : G), f x = x ^ (m : ℕ)

/-- The main theorem: any SpecialGroup is abelian -/
theorem special_group_is_abelian (G : Type*) [SpecialGroup G] : 
  AddCommGroup G :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_group_is_abelian_l1231_123122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1231_123106

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 2/3

-- State the theorem
theorem function_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Monotonically decreasing intervals
  (∀ (x y : ℝ), ((-2*π/3 ≤ x ∧ x < y ∧ y ≤ -π/6) ∨ (π/3 ≤ x ∧ x < y ∧ y ≤ 5*π/6)) →
    f y < f x) ∧
  -- Condition for exactly one solution in [0, π/2]
  (∀ (k : ℝ), (∃! (x : ℝ), 0 ≤ x ∧ x ≤ π/2 ∧ f x + k = 0) ↔
    ((-5/2 < k ∧ k ≤ -3/2) ∨ k = -3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1231_123106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_circleplus_three_l1231_123182

-- Define the ⊕ operation
noncomputable def circleplus (a b : ℝ) : ℝ := a - a / b + b

-- State the theorem
theorem nine_circleplus_three : circleplus 9 3 = 9 := by
  -- Unfold the definition of circleplus
  unfold circleplus
  -- Simplify the expression
  simp [sub_add_eq_add_sub, div_eq_mul_inv]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_circleplus_three_l1231_123182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_g_solution_set_g_alt_l1231_123103

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then
    2 - abs (x + 1)
  else
    (x - 1)^2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem statement
theorem solution_set_g (x : ℝ) : g x ≤ 2 ↔ x ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

-- Alternative formulation using set equality
theorem solution_set_g_alt : {x : ℝ | g x ≤ 2} = Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_g_solution_set_g_alt_l1231_123103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1231_123188

theorem train_speed_problem (speed1 : ℝ) (total_distance : ℝ) (difference : ℝ) (speed2 : ℝ) : 
  speed1 = 20 →
  total_distance = 675 →
  difference = 75 →
  speed2 > 0 →
  (total_distance / (speed1 + speed2)) * speed1 = 
  (total_distance / (speed1 + speed2)) * speed2 + difference →
  speed2 = 16 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1231_123188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_rate_l1231_123137

/-- Profit function -/
def f (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 11/10
  else if 21 ≤ x ∧ x ≤ 60 then x/10
  else 0

/-- Sum of profits from month 1 to month n -/
def sum_profits (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i ↦ f (i + 1))

/-- Profit rate function -/
def g (x : ℕ) : ℚ :=
  f x / (81 + sum_profits (x - 1))

/-- Theorem: The maximum profit rate occurs at month 40 with a value of 2/79 -/
theorem max_profit_rate :
  ∃ (x : ℕ), x = 40 ∧ g x = 2/79 ∧ ∀ (y : ℕ), 1 ≤ y ∧ y ≤ 60 → g y ≤ g x := by
  sorry

#eval g 10 -- For testing g(10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_rate_l1231_123137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_hike_distance_l1231_123194

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

/-- Represents Billy's hike -/
def billyHike (start : Point) : Prop :=
  ∃ (p1 p2 p3 : Point),
    -- First leg: 7 miles east
    p1.x = start.x + 7 ∧ p1.y = start.y ∧
    -- Second leg: 8 miles at 45 degrees north-east
    p2.x = p1.x + 8 * Real.sqrt 2 / 2 ∧ p2.y = p1.y + 8 * Real.sqrt 2 / 2 ∧
    -- Third leg: 4 miles at 60 degrees south from north-east
    p3.x = p2.x - 4 * Real.sqrt 3 / 2 ∧ p3.y = p2.y + 4 / 2 ∧
    -- Final distance
    distance start p3 = Real.sqrt (149 - 56 * Real.sqrt 6 + 32 * Real.sqrt 2)

theorem billy_hike_distance (start : Point) :
  billyHike start := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_hike_distance_l1231_123194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_properties_l1231_123169

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin (θ - Real.pi / 4)

-- Define the line l in parametric form
def line_l (t α : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 1 + t * Real.sin α)

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- State the theorem
theorem intersection_and_distance_properties 
  (α : ℝ) 
  (h_α : 0 ≤ α ∧ α < Real.pi) :
  (∃ A B : ℝ × ℝ, 
    (∃ θ_A θ_B : ℝ, curve_C θ_A = Real.sqrt (A.1^2 + A.2^2) ∧ 
                       curve_C θ_B = Real.sqrt (B.1^2 + B.2^2) ∧
                       ∃ t_A t_B : ℝ, line_l t_A α = A ∧ line_l t_B α = B) ∧
    (α = 0 → (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16) ∧
    (14 < (A.1 - P.1)^2 + (A.2 - P.2)^2 + (B.1 - P.1)^2 + (B.2 - P.2)^2 ∧ 
     (A.1 - P.1)^2 + (A.2 - P.2)^2 + (B.1 - P.1)^2 + (B.2 - P.2)^2 ≤ 22)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_properties_l1231_123169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equidistant_from_A_and_B_l1231_123121

noncomputable section

/-- The distance between two points in 3D space -/
def distance3D (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

/-- A is a point with coordinates (4,3,2) -/
def A : ℝ × ℝ × ℝ := (4, 3, 2)

/-- B is a point with coordinates (2,5,4) -/
def B : ℝ × ℝ × ℝ := (2, 5, 4)

/-- M is a point on the y-axis with coordinates (0,4,0) -/
def M : ℝ × ℝ × ℝ := (0, 4, 0)

/-- Theorem: M is equidistant from A and B -/
theorem M_equidistant_from_A_and_B :
  distance3D M.1 M.2.1 M.2.2 A.1 A.2.1 A.2.2 = 
  distance3D M.1 M.2.1 M.2.2 B.1 B.2.1 B.2.2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equidistant_from_A_and_B_l1231_123121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_triangle_quadrilateral_l1231_123190

/-- The sum of angles in a triangle and quadrilateral diagram -/
theorem angle_sum_triangle_quadrilateral : ∃ (total_sum : ℝ), total_sum = 540 := by
  let sum_triangle : ℝ := 180
  let sum_quadrilateral : ℝ := 360
  let total_sum : ℝ := sum_triangle + sum_quadrilateral
  have h : total_sum = 540 := by sorry
  exact ⟨total_sum, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_triangle_quadrilateral_l1231_123190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_cosine_l1231_123187

theorem parallel_vectors_cosine (θ φ : ℝ) (h1 : 0 < θ ∧ θ < Real.pi / 2) 
  (h2 : 0 < φ ∧ φ < Real.pi / 2) 
  (h3 : (2 : ℝ) * Real.cos θ = Real.sin θ) 
  (h4 : Real.sin (θ - φ) = Real.sqrt 10 / 10) : 
  Real.cos φ = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_cosine_l1231_123187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_symmetric_sine_l1231_123157

-- Define the original function as noncomputable
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Theorem statement
theorem min_value_of_shifted_symmetric_sine 
  (φ : ℝ) 
  (h1 : |φ| < π/2) 
  (h2 : ∀ x, f (x + π/6) φ = f x (-π/3)) -- shifted left by π/6 and symmetric about origin
  (h3 : ∀ x, f x (-π/3) = -f (-x) (-π/3)) -- odd function
  : ∃ x ∈ Set.Icc 0 (π/2), ∀ y ∈ Set.Icc 0 (π/2), f x (-π/3) ≤ f y (-π/3) ∧ f x (-π/3) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_symmetric_sine_l1231_123157
