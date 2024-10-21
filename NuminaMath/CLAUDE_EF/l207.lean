import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_a_squared_equals_n_l207_20780

def a_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n / n + n / a n

theorem floor_a_squared_equals_n (a : ℕ → ℝ) (h : a_sequence a) :
  ∀ n : ℕ, n ≥ 4 → ⌊(a n)^2⌋ = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_a_squared_equals_n_l207_20780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l207_20777

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 130)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_speed_kmh * 1000 / 3600 * crossing_time - train_length = 245 :=
by
  -- Convert speed from km/h to m/s
  have train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  
  -- Calculate total distance
  have total_distance : ℝ := train_speed_ms * crossing_time
  
  -- Calculate bridge length
  have bridge_length : ℝ := total_distance - train_length
  
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l207_20777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l207_20727

open Complex Real

-- Define the complex number -1/2 - (√3 * i)/2
noncomputable def target_complex : ℂ := -1/2 - (Real.sqrt 3 * I)/2

-- Define the equation z^6 = target_complex
def equation (z : ℂ) : Prop := z^6 = target_complex

-- Define the set of solutions
def solutions : Set ℝ := {θ : ℝ | 0 ≤ θ ∧ θ < 2*π ∧ equation (exp (θ*I))}

-- Theorem statement
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.card = 6 ∧ (∀ θ ∈ s, θ ∈ solutions) ∧ (s.sum id) * (180/π) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l207_20727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_curvature_exists_parabola_curvature_bound_l207_20781

-- Define the curvature function
noncomputable def curvature (f : ℝ → ℝ) (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := f x₁
  let y₂ := f x₂
  let k₁ := (deriv f) x₁
  let k₂ := (deriv f) x₂
  |k₁ - k₂| / Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Statement 1: There exists a function with constant curvature
theorem constant_curvature_exists : ∃ (f : ℝ → ℝ) (c : ℝ), ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → curvature f x₁ x₂ = c := by
  sorry

-- Statement 2: Curvature bound for parabola
theorem parabola_curvature_bound :
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → curvature (λ x => x^2 + 1) x₁ x₂ ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_curvature_exists_parabola_curvature_bound_l207_20781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l207_20786

theorem sum_of_coefficients (x y z : ℝ) : 
  (x - 2*y + 3*z)^12 = 4096 ↔ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_l207_20786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paperClipsInTwoCases_eq_l207_20735

/-- The number of paper clips in two cases, given:
  c: number of cartons per case
  b: number of boxes per carton
  Each box contains 500 paper clips -/
def paperClipsInTwoCases (c b : ℕ) : ℕ := 2 * c * b * 500

/-- Theorem stating that the number of paper clips in two cases
    is equal to 2 * (c * b) * 500 -/
theorem paperClipsInTwoCases_eq (c b : ℕ) :
  paperClipsInTwoCases c b = 2 * (c * b) * 500 := by
  unfold paperClipsInTwoCases
  ring

#eval paperClipsInTwoCases 3 4  -- Example: 3 cartons per case, 4 boxes per carton

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paperClipsInTwoCases_eq_l207_20735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l207_20775

/-- The distance between a point and a line in 2D space -/
def pointToLineDistance (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The distance from the focus to the asymptote of the hyperbola x²/2 - y²/4 = 1 is 2 -/
theorem hyperbola_focus_to_asymptote_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / 2 - y^2 / 4 = 1}
  let focus : ℝ × ℝ := (Real.sqrt 6, 0)
  let asymptote := {(x, y) : ℝ × ℝ | y = Real.sqrt 2 * x}
  (pointToLineDistance focus asymptote) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_to_asymptote_distance_l207_20775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l207_20738

def is_constant_poly (P : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x, P x = c

theorem polynomial_functional_equation
  (k : ℕ) (P : Polynomial ℝ) :
  (∀ x, P.eval (P.eval x) = (P.eval x) ^ k) →
  (P = Polynomial.monomial k 1) ∨
  (is_constant_poly P.eval ∧
    (k = 1 ∨
    (Even k ∧ ∀ x, P.eval x = 1) ∨
    (Odd k ∧ k > 1 ∧ ∀ x, P.eval x = 1 ∨ P.eval x = -1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l207_20738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_satisfies_equation_l207_20704

/-- The speed of a car satisfying the given conditions -/
noncomputable def car_speed : ℝ :=
  (3600 : ℝ) / 47

/-- Theorem stating that the car_speed satisfies the given equation -/
theorem car_speed_satisfies_equation :
  1 / car_speed = 1 / 80 + 2 / 3600 := by
  sorry

-- Use #eval with a rational approximation instead of the exact real number
#eval (3600 : ℚ) / 47

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_satisfies_equation_l207_20704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_equation_l207_20711

theorem smallest_angle_equation (φ : Real) : 
  (φ > 0) → 
  (φ < 360) → 
  (Real.cos φ = Real.sin 45 + Real.cos 37 - Real.sin 23 - Real.cos 11) → 
  (∀ θ : Real, 0 < θ ∧ θ < φ → Real.cos θ ≠ Real.sin 45 + Real.cos 37 - Real.sin 23 - Real.cos 11) → 
  φ = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_equation_l207_20711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l207_20763

noncomputable def f (p : ℝ) (x : ℝ) : ℝ := 2^x + (p - 1) * 2^(-x)

noncomputable def g (p k : ℝ) (x : ℝ) : ℝ := f p (2*x) - 2*k * (2^x - 2^(-x))

theorem problem_solution (p k m : ℝ) :
  (∀ x, f p x = f p (-x)) →
  (∀ x ≥ 1, g p k x ≥ -4) ∧ (∃ x ≥ 1, g p k x = -4) →
  (∀ x, f p (2*x) > m * f p x - 4) →
  p = 2 ∧ k = Real.sqrt 6 ∧ m < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l207_20763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_probability_binomial_max_probability_half_l207_20755

/-- The binomial probability mass function -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The theorem stating that P(X=k) is maximized when k = 10 for X ~ B(20, p) -/
theorem binomial_max_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  ∀ k : ℕ, k ≤ 20 → binomial_pmf 20 p 10 ≥ binomial_pmf 20 p k :=
by
  sorry

/-- Corollary: When p is close to 1/2, k = 10 maximizes P(X=k) -/
theorem binomial_max_probability_half (p : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : |p - 1/2| < 1/40) :
  ∀ k : ℕ, k ≤ 20 → binomial_pmf 20 p 10 ≥ binomial_pmf 20 p k :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_max_probability_binomial_max_probability_half_l207_20755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l207_20730

-- Define the function f(x) = xe^(-x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 1 / Real.exp 1 := by
  -- Proof goes here
  sorry

-- You can add more helper lemmas or theorems if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l207_20730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_angles_l207_20779

/-- Represents a parabola defined by y^2 = 6x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : ∀ x y, equation x y ↔ y^2 = 6 * x

/-- Represents a chord of the parabola --/
structure Chord where
  length : ℝ
  passes_through_focus : Bool

/-- The focus of the parabola y^2 = 6x --/
noncomputable def parabola_focus : ℝ × ℝ := (3/2, 0)

/-- Theorem stating the slope angles of a chord passing through the focus --/
theorem chord_slope_angles (p : Parabola) (c : Chord) :
  c.length = 12 ∧ c.passes_through_focus →
  (∃ θ₁ θ₂, θ₁ = π/6 ∧ θ₂ = 5*π/6 ∧ 
   (∀ x y, p.equation x y → 
    ∃ m, (y - (parabola_focus.2)) = m * (x - (parabola_focus.1)) ∧
    (m = Real.tan θ₁ ∨ m = Real.tan θ₂))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_slope_angles_l207_20779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_is_D_l207_20791

-- Define the equations
def equation_A (x y : ℝ) : Prop := y = 2*x - 3*y
def equation_B (x y : ℝ) : Prop := x = 2 - 3*y
def equation_C (x y : ℝ) : Prop := -y = 2*x - 1
def equation_D (x y : ℝ) : Prop := y = x

-- Define what it means for y to be expressed algebraically in terms of x
def is_algebraic_in_x (f : ℝ → ℝ) : Prop := ∃ (g : ℝ → ℝ), ∀ x, f x = g x

-- Helper function for equation A (to avoid infinite recursion)
noncomputable def equation_A_func (x : ℝ) : ℝ := 
  (2*x) / 4

-- Theorem statement
theorem correct_equation_is_D :
  (is_algebraic_in_x (λ x => x)) ∧
  (¬ is_algebraic_in_x equation_A_func) ∧
  (¬ is_algebraic_in_x (λ x => (2 - 3*x) / 3)) ∧
  (¬ is_algebraic_in_x (λ x => -(2*x - 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_is_D_l207_20791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_calculation_l207_20723

/-- Represents the interest rate structure over time -/
structure InterestRates :=
  (rate1 : ℝ) (period1 : ℝ)
  (rate2 : ℝ) (period2 : ℝ)
  (rate3 : ℝ) (period3 : ℝ)
  (rate4 : ℝ) (period4 : ℝ)

/-- Calculates the total interest factor given an interest rate structure -/
def totalInterestFactor (rates : InterestRates) : ℝ :=
  rates.rate1 * rates.period1 +
  rates.rate2 * rates.period2 +
  rates.rate3 * rates.period3 +
  rates.rate4 * rates.period4

/-- The main theorem stating the relationship between total interest, 
    interest rates, and principal amount -/
theorem borrowed_amount_calculation 
  (rates : InterestRates)
  (total_interest : ℝ)
  (h1 : rates.rate1 = 0.065 ∧ rates.period1 = 2.5)
  (h2 : rates.rate2 = 0.095 ∧ rates.period2 = 3.75)
  (h3 : rates.rate3 = 0.11 ∧ rates.period3 = 1.5)
  (h4 : rates.rate4 = 0.145 ∧ rates.period4 = 4.25)
  (h5 : total_interest = 14500) :
  ∃ (principal : ℝ), 
    principal * totalInterestFactor rates = total_interest ∧ 
    (abs (principal - 11153.85) < 0.01) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_calculation_l207_20723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_36_l207_20732

theorem divisors_of_36 : 
  (Finset.filter (fun x : ℕ => x > 1 ∧ (36 % x = 0)) (Finset.range 37)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_36_l207_20732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_45_l207_20799

/-- The sum of all positive integer divisors of 45 is 78 -/
theorem sum_of_divisors_45 : (Finset.filter (λ x ↦ 45 % x = 0) (Finset.range 46)).sum id = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_45_l207_20799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dreamy_vacation_probability_l207_20784

def probability_of_success : ℝ := 0.4

def number_of_trials : ℕ := 5

def number_of_successes : ℕ := 3

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem dreamy_vacation_probability :
  ∃ ε > 0, |binomial_probability number_of_trials number_of_successes probability_of_success - 0.2304| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dreamy_vacation_probability_l207_20784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_is_two_fifty_l207_20709

/-- Represents the pizza order and payment scenario --/
structure PizzaOrder where
  total_slices : ℕ
  plain_cost : ℚ
  pepperoni_cost : ℚ
  pepperoni_fraction : ℚ
  jake_plain_slices : ℕ

/-- Calculates the payment difference between Jake and Mia --/
def payment_difference (order : PizzaOrder) : ℚ :=
  let total_cost := order.plain_cost + order.pepperoni_cost
  let cost_per_slice := total_cost / order.total_slices
  let pepperoni_slices := (order.pepperoni_fraction * order.total_slices).floor
  let jake_payment := cost_per_slice * (pepperoni_slices + order.jake_plain_slices)
  let mia_payment := cost_per_slice * (order.total_slices - pepperoni_slices - order.jake_plain_slices)
  jake_payment - mia_payment

/-- Theorem stating that the payment difference is $2.50 --/
theorem payment_difference_is_two_fifty (order : PizzaOrder) 
  (h1 : order.total_slices = 12)
  (h2 : order.plain_cost = 12)
  (h3 : order.pepperoni_cost = 3)
  (h4 : order.pepperoni_fraction = 1/3)
  (h5 : order.jake_plain_slices = 3) :
  payment_difference order = 5/2 := by
  sorry

#eval payment_difference {
  total_slices := 12,
  plain_cost := 12,
  pepperoni_cost := 3,
  pepperoni_fraction := 1/3,
  jake_plain_slices := 3
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_is_two_fifty_l207_20709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_decimal_irreducible_poly_l207_20726

/-- Represents a natural number in decimal notation as a list of its digits. -/
def DecimalNotation := List Nat

/-- Checks if a natural number is prime. -/
def isPrime (n : Nat) : Prop := sorry

/-- Converts a DecimalNotation to a natural number. -/
def decimalToNat (d : DecimalNotation) : Nat := sorry

/-- Constructs a polynomial from a DecimalNotation. -/
def toPoly (d : DecimalNotation) : Polynomial ℤ := sorry

/-- Theorem: If p is prime in decimal notation, then its corresponding polynomial is irreducible. -/
theorem prime_decimal_irreducible_poly (p : DecimalNotation) :
  p.head! > 0 →
  isPrime (decimalToNat p) →
  Irreducible (toPoly p) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_decimal_irreducible_poly_l207_20726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l207_20744

/-- Quadratic function type -/
def QuadraticFunction (a m : ℝ) (x : ℝ) : ℝ := a * (x - m)^2 + a * (x - m)

/-- Area of a triangle given three points -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

/-- Theorem for the quadratic function properties -/
theorem quadratic_function_properties
  (a m : ℝ)
  (h_a : a > 0)
  (h_m : m > 0)
  (A B : ℝ × ℝ)
  (h_AB : A.1 < B.1 ∧ QuadraticFunction a m A.1 = 0 ∧ QuadraticFunction a m B.1 = 0)
  (C : ℝ × ℝ)
  (h_C : C.2 = QuadraticFunction a m C.1 ∧ ∀ x, QuadraticFunction a m x ≥ C.2)
  (D : ℝ × ℝ)
  (h_D : D.1 = 0 ∧ D.2 > 0 ∧ QuadraticFunction a m D.1 = D.2)
  (h_area : area_triangle A B C = (1/3) * area_triangle A B D) :
  (B.1 - A.1 = 1) ∧ (m = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l207_20744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l207_20713

theorem solution_set_of_inequality :
  let S := {x : ℝ | (2 : ℝ)^(x^2 + 2*x - 4) ≤ 1/2}
  S = Set.Icc (-3) 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l207_20713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_correct_angular_frequency_eq_beta_l207_20716

/-- Represents the properties of a thin, uniform rod undergoing small oscillations. -/
structure OscillatingRod where
  m : ℝ  -- mass of the rod
  L : ℝ  -- length of the rod
  g : ℝ  -- acceleration due to gravity
  d : ℝ  -- parameter related to rotational inertia
  k : ℝ  -- suspension point relative to center

/-- The angular frequency of the rod's oscillations. -/
noncomputable def angular_frequency (rod : OscillatingRod) : ℝ :=
  (rod.k * (rod.g / rod.d)).sqrt

/-- The beta parameter for the rod's oscillations. -/
noncomputable def beta (rod : OscillatingRod) : ℝ :=
  (rod.k / (1 + rod.k^2)).sqrt

/-- Theorem stating that the beta parameter is correctly defined for the oscillating rod. -/
theorem beta_correct (rod : OscillatingRod) :
  beta rod = (rod.k / (1 + rod.k^2)).sqrt :=
by
  -- The proof is omitted
  sorry

/-- Theorem relating the angular frequency to the beta parameter. -/
theorem angular_frequency_eq_beta (rod : OscillatingRod) :
  angular_frequency rod = beta rod * (rod.g / rod.d).sqrt :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_correct_angular_frequency_eq_beta_l207_20716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l207_20792

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The theorem to be proved -/
theorem ellipse_equation (f1 f2 a b : Point) (ab_length : ℝ) :
  f1 = Point.mk (-1) 0 →
  f2 = Point.mk 1 0 →
  a.x = 1 →
  b.x = 1 →
  distance a b = ab_length →
  ab_length = 3 →
  let e := Ellipse.mk 2 (Real.sqrt 3)
  isOnEllipse a e ∧ isOnEllipse b e ∧
  ∀ (p : Point), isOnEllipse p e ↔ p.x^2 / 4 + p.y^2 / 3 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l207_20792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l207_20721

/-- Helper function to calculate Euclidean distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The total distance traveled by a laser beam reflecting off axes -/
theorem laser_beam_distance : ∃ (y_reflection x_reflection : ℝ × ℝ),
  let start : ℝ × ℝ := (2, 3)
  let end_point : ℝ × ℝ := (8, 3)
  let segment1 := distance start y_reflection
  let segment2 := distance y_reflection x_reflection
  let segment3 := distance x_reflection end_point
  let total_distance := segment1 + segment2 + segment3
  total_distance = 2 * Real.sqrt 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l207_20721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_opens_downwards_implies_m_eq_neg_two_l207_20768

/-- A function f is quadratic if it can be expressed as f(x) = ax^2 + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- A quadratic function opens downwards if its leading coefficient is negative -/
def OpensDownwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Our specific function -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 - 2) + 2 * x - 1

theorem quadratic_opens_downwards_implies_m_eq_neg_two :
  ∀ m : ℝ, IsQuadratic (f m) → OpensDownwards (f m) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_opens_downwards_implies_m_eq_neg_two_l207_20768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_equality_infinite_solutions_l207_20789

-- Define the function for which we're taking the limit
noncomputable def f (a b x : ℝ) : ℝ := (Real.sin x)^2 / (Real.exp (a * x) - 2 * b * x - 1)

theorem limit_equality (a b : ℝ) :
  Filter.Tendsto (f a b) (nhds 0) (nhds (1/2)) ↔ a = 2 ∨ a = -2 := by
  sorry

-- The number of ordered pairs (a, b) satisfying the limit equality is infinite
theorem infinite_solutions :
  Set.Infinite {p : ℝ × ℝ | Filter.Tendsto (f p.1 p.2) (nhds 0) (nhds (1/2))} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_equality_infinite_solutions_l207_20789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_inequality_l207_20770

/-- A triangle in a 2D Euclidean space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  (((t.A.1 + t.B.1 + t.C.1) / 3), ((t.A.2 + t.B.2 + t.C.2) / 3))

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to vertices -/
noncomputable def s₁ (t : Triangle) : ℝ :=
  let G := centroid t
  distance G t.A + distance G t.B + distance G t.C

/-- Sum of sides of the triangle -/
noncomputable def s₂ (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: The sum of the sides of a triangle is greater than 3 times
    the sum of the distances from the centroid to each vertex -/
theorem centroid_distance_inequality (t : Triangle) : s₂ t > 3 * s₁ t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_inequality_l207_20770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pm_eq_mq_l207_20742

/-- Given a square ABCD with side length 15, point E on DC 8 units from D,
    and M the midpoint of AE, prove that PM = MQ where P and Q are the
    intersections of the perpendicular bisector of AE with AD and BC respectively. -/
theorem pm_eq_mq (A B C D E M P Q : ℝ × ℝ) : 
  (A = (0, 15) ∧ B = (15, 15) ∧ C = (15, 0) ∧ D = (0, 0)) →  -- Square ABCD
  (E = (8, 0)) →  -- E on DC, 8 units from D
  (M = (4, 7.5)) →  -- M is midpoint of AE
  (∃ k : ℝ, P = (18, 15) ∧ Q = (-10, 0) ∧  -- P and Q on perpendicular bisector
    (P.2 - M.2) = k * (P.1 - M.1) ∧
    (Q.2 - M.2) = k * (Q.1 - M.1) ∧
    k = 8/15) →  -- Slope of perpendicular bisector
  dist M P = dist M Q :=
by sorry

/-- Helper function to calculate Euclidean distance -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pm_eq_mq_l207_20742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_traveled_l207_20715

/-- Proves that the actual distance traveled is 15 km given the conditions -/
theorem actual_distance_traveled
  (actual_speed : ℝ)
  (hypothetical_speed : ℝ)
  (additional_distance : ℝ)
  (h1 : actual_speed = 3)
  (h2 : hypothetical_speed = 6)
  (h3 : additional_distance = 15)
  (h4 : hypothetical_speed * (additional_distance / (hypothetical_speed - actual_speed)) =
        actual_speed * (additional_distance / (hypothetical_speed - actual_speed)) + additional_distance) :
  actual_speed * (additional_distance / (hypothetical_speed - actual_speed)) = 15 := by
  sorry

#check actual_distance_traveled

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_traveled_l207_20715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_even_6437_5054_l207_20796

/-- Round a real number to the nearest even integer -/
noncomputable def roundToNearestEven (x : ℝ) : ℤ :=
  let roundedInt := Int.floor (x + 0.5)
  if roundedInt % 2 = 0 then
    roundedInt
  else if x > (roundedInt : ℝ) then
    roundedInt + 1
  else
    roundedInt - 1

/-- The problem statement -/
theorem round_to_nearest_even_6437_5054 :
  roundToNearestEven 6437.5054 = 6438 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_even_6437_5054_l207_20796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_line_arrangements_l207_20773

/-- The number of ways to arrange 5 students in a line --/
def total_arrangements : ℕ := 120

/-- The number of ways to arrange 3 students --/
def three_student_arrangements : ℕ := 6

/-- The number of ways to arrange 2 students --/
def two_student_arrangements : ℕ := 2

/-- The number of ways to arrange 4 entities --/
def four_entity_arrangements : ℕ := 24

/-- The number of pairs of students from a group of 3 --/
def number_of_pairs : ℕ := 3

theorem student_line_arrangements :
  total_arrangements - 
  (three_student_arrangements * three_student_arrangements) - 
  (number_of_pairs * two_student_arrangements * four_entity_arrangements - 
   2 * three_student_arrangements * three_student_arrangements) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_line_arrangements_l207_20773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_profit_percent_l207_20746

/-- Calculate the profit percent for a retailer selling a radio -/
theorem radio_profit_percent (purchase_price overhead_expenses selling_price : ℚ) : 
  purchase_price = 232 →
  overhead_expenses = 15 →
  selling_price = 300 →
  let total_cost_price := purchase_price + overhead_expenses
  let profit := selling_price - total_cost_price
  let profit_percent := (profit / total_cost_price) * 100
  abs (profit_percent - 21.46) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_profit_percent_l207_20746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_eight_to_twelve_l207_20703

theorem fourth_root_eight_to_twelve : (8 : ℝ) ^ ((1/4 : ℝ) * 12) = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_eight_to_twelve_l207_20703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_value_l207_20759

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 17/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_value_l207_20759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_prism_volume_ratio_l207_20710

/-- The ratio of the volume of a right circular cylinder inscribed in a right square prism -/
theorem cylinder_prism_volume_ratio :
  ∀ (r h : ℝ),
    r > 0 → h > 0 →
    (π * r^2 * h) / (4 * r^2 * h) = π / 4 := by
  intros r h hr hh
  -- Simplify the fraction
  have h1 : (π * r^2 * h) / (4 * r^2 * h) = (π * r^2 * h) / (4 * r^2 * h) := by rfl
  -- Cancel common factors
  have h2 : (π * r^2 * h) / (4 * r^2 * h) = π / 4 := by
    field_simp [hr, hh]
    ring
  -- Conclude the proof
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_prism_volume_ratio_l207_20710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_axis_l207_20794

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin ((2/3) * x + (3 * Real.pi) / 4)

-- Define the transformation
noncomputable def transform (x : ℝ) : ℝ := (1/3) * x + Real.pi / 8

-- Define the resulting function g
noncomputable def g (x : ℝ) : ℝ := f (transform x)

-- State the theorem
theorem g_symmetry_axis :
  ∀ x : ℝ, g (Real.pi / 2 - x) = g (Real.pi / 2 + x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_axis_l207_20794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l207_20760

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 10 + x - 3

-- State the theorem
theorem zero_point_interval (k : ℤ) (x₀ : ℝ) : 
  (f x₀ = 0) → (x₀ > k) → (x₀ < k + 1) → (k = 2) := by
  sorry

#check zero_point_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l207_20760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_longer_segment_l207_20765

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a line segment MN of length 2, and P is the golden section point of MN,
    prove that the length of the longer segment MP is (√5 - 1). -/
theorem golden_section_longer_segment :
  ∀ (M N P : ℝ), 
    (N - M = 2) →  -- Length of MN is 2
    (P - M) / (N - P) = φ →  -- P is the golden section point
    (P - M = Real.sqrt 5 - 1) :=  -- Length of MP is (√5 - 1)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_longer_segment_l207_20765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_in_second_vessel_l207_20725

/-- Proves that the percentage of alcohol in the second vessel is approximately 38.33% --/
theorem alcohol_percentage_in_second_vessel 
  (vessel1_capacity : ℝ) 
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (total_liquid : ℝ)
  (final_mixture_capacity : ℝ)
  (final_mixture_alcohol_percentage : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 6)
  (h4 : total_liquid = 8)
  (h5 : final_mixture_capacity = 10)
  (h6 : final_mixture_alcohol_percentage = 35)
  : ∃ (vessel2_alcohol_percentage : ℝ), 
    (abs (vessel2_alcohol_percentage - 38.33) < 0.01 ∧ 
     vessel1_capacity * (vessel1_alcohol_percentage / 100) + 
     vessel2_capacity * (vessel2_alcohol_percentage / 100) = 
     total_liquid * (final_mixture_alcohol_percentage / 100)) :=
by
  sorry

#check alcohol_percentage_in_second_vessel

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_percentage_in_second_vessel_l207_20725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l207_20719

theorem function_characterization (f : ℕ → ℕ) :
  (∀ n : ℕ, 2 * n + 2001 ≤ f (f n) + f n ∧ f (f n) + f n ≤ 2 * n + 2002) →
  (∀ n : ℕ, f n = n + 667) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l207_20719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sneakers_price_l207_20756

/-- Calculate the discounted price of an item given its original price and discount percentage. -/
noncomputable def discounted_price (original_price : ℝ) (discount_percent : ℝ) : ℝ :=
  original_price * (1 - discount_percent / 100)

/-- Theorem stating that a 260 yuan item with a 30% discount costs 182 yuan. -/
theorem sneakers_price : discounted_price 260 30 = 182 := by
  -- Unfold the definition of discounted_price
  unfold discounted_price
  -- Simplify the arithmetic expression
  simp [mul_sub, mul_div_cancel']
  -- Check that the result is equal to 182
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sneakers_price_l207_20756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_case_angle_60_degrees_l207_20745

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² + c² - b² = ac, then angle B measures 60°. -/
theorem triangle_special_case_angle_60_degrees 
  (a b c : ℝ) (A B C : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 + c^2 - b^2 = a*c → B = π/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_case_angle_60_degrees_l207_20745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_preimage_of_neg_one_third_l207_20778

/-- The function f(x) = (2-x)/(3x+4) -/
noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

/-- The domain of f excludes x = -4/3 -/
def f_domain (x : ℝ) : Prop := x ≠ -4/3

/-- Theorem: There is no x in the domain of f such that f(x) = -1/3 -/
theorem no_preimage_of_neg_one_third :
  ∀ x : ℝ, f_domain x → f x ≠ -1/3 := by
  intro x hx
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_preimage_of_neg_one_third_l207_20778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_satisfies_conditions_l207_20728

noncomputable section

-- Define the four given functions
def f_A (x : ℝ) := Real.sin (x / 2)
def f_B (x : ℝ) := Real.cos (2 * x)
def f_C (x : ℝ) := Real.tan (x - Real.pi / 4)
def f_D (x : ℝ) := Real.sin (2 * x + Real.pi / 4)

-- Define the properties we're looking for
def has_period_pi (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + Real.pi) = f x

def is_decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem cos_2x_satisfies_conditions :
  has_period_pi f_B ∧
  is_decreasing_on_interval f_B 0 (Real.pi / 2) ∧
  (¬ has_period_pi f_A ∨ ¬ is_decreasing_on_interval f_A 0 (Real.pi / 2)) ∧
  (¬ has_period_pi f_C ∨ ¬ is_decreasing_on_interval f_C 0 (Real.pi / 2)) ∧
  (¬ has_period_pi f_D ∨ ¬ is_decreasing_on_interval f_D 0 (Real.pi / 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_satisfies_conditions_l207_20728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_correct_may_bill_correct_july_bill_correct_l207_20783

/-- Represents the tiered electricity pricing system -/
noncomputable def electricity_bill (x : ℝ) : ℝ :=
  if x ≤ 170 then 0.52 * x
  else if x ≤ 260 then 0.57 * x - 8.5
  else 0.82 * x - 73.5

/-- Theorem stating that the electricity bill function is correct -/
theorem electricity_bill_correct (x : ℝ) (h : x ≥ 0) :
  electricity_bill x =
    if x ≤ 170 then 0.52 * x
    else if x ≤ 260 then 0.57 * x - 8.5
    else 0.82 * x - 73.5 :=
by sorry

/-- Theorem for part (1) of the problem -/
theorem may_bill_correct :
  electricity_bill 160 = 83.2 :=
by sorry

/-- Theorem for part (3) of the problem -/
theorem july_bill_correct :
  electricity_bill 240 = 128.3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_correct_may_bill_correct_july_bill_correct_l207_20783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangles_count_l207_20706

theorem right_angled_triangles_count : 
  Finset.card (Finset.filter (fun p : ℕ × ℕ => 
    let a := p.1
    let b := p.2
    a^2 + b^2 = (b + 1)^2 ∧ 
    0 < a ∧ 
    0 < b ∧ 
    b < 2011
  ) (Finset.product (Finset.range 2011) (Finset.range 2011))) = 31 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_triangles_count_l207_20706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_equals_five_fourths_l207_20712

-- Define the function f
noncomputable def f : ℝ → ℝ := fun t => ((t - 1) / 2)^2 + 1

-- State the theorem
theorem f_zero_equals_five_fourths :
  (∀ x : ℝ, f (2 * x + 1) = x^2 + 1) → f 0 = 5/4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_equals_five_fourths_l207_20712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_contains_triangle_l207_20749

/-- A convex polygon in a square division --/
structure PolygonInSquare where
  sides : ℕ
  is_convex : Bool

/-- A division of a square into convex polygons --/
structure SquareDivision where
  polygons : List PolygonInSquare
  distinct_sides : ∀ p q, p ∈ polygons → q ∈ polygons → p ≠ q → p.sides ≠ q.sides
  more_than_one : polygons.length > 1

/-- The main theorem: Any valid square division contains a triangle --/
theorem square_division_contains_triangle (d : SquareDivision) :
  ∃ p, p ∈ d.polygons ∧ p.sides = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_contains_triangle_l207_20749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_envelope_probability_l207_20739

/-- IndependentEvents represents the condition that the events are independent --/
axiom IndependentEvents : Prop

/-- ProbabilityAtLeastOne calculates the probability that at least one of three events occurs --/
noncomputable def ProbabilityAtLeastOne (p_A p_B p_C : ℝ) : ℝ :=
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C)

/-- Theorem stating that the probability of at least one student receiving more than $1 is 7/8 --/
theorem red_envelope_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 2/3) 
  (h_B : p_B = 1/2) 
  (h_C : p_C = 1/4) 
  (h_independent : IndependentEvents) : 
  ProbabilityAtLeastOne p_A p_B p_C = 7/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_envelope_probability_l207_20739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_probability_l207_20705

/-- Represents a point on the lattice grid -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- A path is a list of directions -/
def AntPath := List Direction

/-- The probability of a single step in any direction -/
def stepProbability : ℚ := 1 / 4

/-- The number of steps in the path -/
def numSteps : Nat := 6

/-- The starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- The ending point -/
def endPoint : Point := ⟨1, 1⟩

/-- Function to check if a path leads from startPoint to endPoint -/
def isValidPath (p : AntPath) : Bool :=
  sorry

/-- Function to count the number of valid paths -/
noncomputable def countValidPaths : Nat :=
  sorry

/-- The main theorem to prove -/
theorem ant_path_probability :
  (countValidPaths : ℚ) * stepProbability ^ numSteps = 5 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_path_probability_l207_20705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_range_of_a_l207_20708

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry
def f' : ℝ → ℝ := sorry

-- Define the parameter a
variable (a : ℝ)

-- State the theorem
theorem local_max_range_of_a :
  (∀ x, f' x = a * (x + 1) * (x - a)) →  -- Condition on f'
  (∃ δ > 0, ∀ x, |x - a| < δ → f x ≤ f a) →  -- Local maximum at x = a
  -1 < a ∧ a < 0 :=  -- Conclusion: a is in (-1, 0)
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_range_of_a_l207_20708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_line_equation_l207_20736

/-- The focus of a parabola with equation x² = 4y -/
def parabola_focus : ℝ × ℝ := (0, 1)

/-- The right focus of a hyperbola with equation x²/4 - y²/5 = 1 -/
def hyperbola_right_focus : ℝ × ℝ := (3, 0)

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (p₁ p₂ : ℝ × ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y - p₁.2) * (p₂.1 - p₁.1) = (x - p₁.1) * (p₂.2 - p₁.2)

/-- The theorem stating that the line passing through the focus of the parabola
    and the right focus of the hyperbola has the equation x + 3y - 3 = 0 -/
theorem focus_line_equation :
  ∀ x y, line_equation parabola_focus hyperbola_right_focus x y ↔ x + 3*y - 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_line_equation_l207_20736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_container_price_l207_20762

noncomputable def cylinderVolume (diameter : ℝ) (height : ℝ) : ℝ :=
  Real.pi * (diameter / 2) ^ 2 * height

noncomputable def pricePerUnit (price : ℝ) (volume : ℝ) : ℝ :=
  price / volume

noncomputable def applyDiscount (price : ℝ) (discountRate : ℝ) : ℝ :=
  price * (1 - discountRate)

theorem larger_container_price (smallDiameter smallHeight smallPrice largeDiameter largeHeight discountRate : ℝ) :
  smallDiameter = 5 →
  smallHeight = 8 →
  smallPrice = 1.5 →
  largeDiameter = 10 →
  largeHeight = 10 →
  discountRate = 0.1 →
  let smallVolume := cylinderVolume smallDiameter smallHeight
  let largeVolume := cylinderVolume largeDiameter largeHeight
  let smallPricePerUnit := pricePerUnit smallPrice smallVolume
  let discountedPricePerUnit := applyDiscount smallPricePerUnit discountRate
  let largePrice := discountedPricePerUnit * largeVolume
  largePrice = 6.75 := by
    sorry

#check larger_container_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_container_price_l207_20762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_half_less_than_reciprocal_l207_20764

theorem only_half_less_than_reciprocal :
  ∀ x : ℚ, x ∈ ({-1/2, -1, 0, 1/2, 2} : Set ℚ) →
    (x < (1 / x) ↔ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_half_less_than_reciprocal_l207_20764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_braking_distance_approx_78m_l207_20752

/-- Calculates the minimum braking distance for a car given its maximum incline angle and initial speed -/
noncomputable def min_braking_distance (max_incline_angle : Real) (initial_speed : Real) : Real :=
  let g : Real := 10
  let friction_coeff : Real := Real.sin max_incline_angle / Real.cos max_incline_angle
  let deceleration : Real := friction_coeff * g
  initial_speed^2 / (2 * deceleration)

/-- Theorem stating that the minimum braking distance for a car with given parameters is approximately 78 meters -/
theorem braking_distance_approx_78m :
  let max_incline_angle : Real := 30 * Real.pi / 180
  let initial_speed : Real := 30
  let braking_distance := min_braking_distance max_incline_angle initial_speed
  ‖braking_distance - 78‖ < 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_braking_distance_approx_78m_l207_20752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_MB_MC_l207_20774

-- Define the circle E in polar coordinates
def circle_E (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -x

-- Define point A as the intersection of line l and circle E
def point_A : ℝ × ℝ := (-2, 2)

-- Define midpoint M
def point_M : ℝ × ℝ := (-1, 1)

-- Define circle E in Cartesian coordinates
def circle_E_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the maximum value of ||MB| - |MC||
noncomputable def max_diff : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem max_difference_MB_MC :
  ∀ B C : ℝ × ℝ,
  circle_E_cartesian B.1 B.2 →
  circle_E_cartesian C.1 C.2 →
  ∃ (t : ℝ), B.1 = point_M.1 + t * Real.cos (π/4) ∧ B.2 = point_M.2 + t * Real.sin (π/4) →
  ∃ (s : ℝ), C.1 = point_M.1 + s * Real.cos (π/4) ∧ C.2 = point_M.2 + s * Real.sin (π/4) →
  |Real.sqrt ((B.1 - point_M.1)^2 + (B.2 - point_M.2)^2) - 
   Real.sqrt ((C.1 - point_M.1)^2 + (C.2 - point_M.2)^2)| ≤ max_diff :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_MB_MC_l207_20774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l207_20761

/-- The function f(x) = (2 sin x + 1) / (3 sin x - 5) -/
noncomputable def f (x : ℝ) : ℝ := (2 * Real.sin x + 1) / (3 * Real.sin x - 5)

/-- The smallest positive period of f(x) is 2π -/
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l207_20761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_odd_is_nine_tenths_l207_20737

def S : Finset Nat := {1, 2, 3, 4, 5}

def is_odd (n : Nat) : Bool := n % 2 = 1

def prob_at_least_one_odd (s : Finset Nat) : Rat :=
  let total_combinations := Nat.choose s.card 2
  let both_even_combinations := Nat.choose (s.filter (fun x => !is_odd x)).card 2
  1 - (both_even_combinations : Rat) / total_combinations

theorem prob_at_least_one_odd_is_nine_tenths :
  prob_at_least_one_odd S = 9/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_odd_is_nine_tenths_l207_20737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l207_20776

noncomputable def train_length : ℝ := 100  -- meters
noncomputable def train_speed_kmh : ℝ := 162  -- km/hr

noncomputable def km_per_hour_to_meter_per_second (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

noncomputable def time_to_cross_pole (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_crossing_time :
  let train_speed_ms := km_per_hour_to_meter_per_second train_speed_kmh
  abs (time_to_cross_pole train_length train_speed_ms - 2.22) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l207_20776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ants_sugar_harvesting_l207_20767

/-- Calculates the additional hours needed to harvest all sugar -/
noncomputable def additional_hours_needed (initial_sugar : ℝ) (removal_rate : ℝ) (hours_passed : ℝ) : ℝ :=
  (initial_sugar - removal_rate * hours_passed) / removal_rate

/-- Theorem: Given the initial conditions, the ants need 3 more hours to harvest all sugar -/
theorem ants_sugar_harvesting (initial_sugar : ℝ) (removal_rate : ℝ) (hours_passed : ℝ)
    (h1 : initial_sugar = 24)
    (h2 : removal_rate = 4)
    (h3 : hours_passed = 3) :
    additional_hours_needed initial_sugar removal_rate hours_passed = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ants_sugar_harvesting_l207_20767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_schedule_achieves_goal_l207_20707

/-- Represents the summer work scenario -/
structure SummerWork where
  hourly_rate : ℚ
  original_weeks : ℕ
  original_hours_per_week : ℕ
  sick_weeks : ℕ
  new_hours_per_week : ℕ

/-- Calculates the total earnings for a given work scenario -/
def total_earnings (work : SummerWork) : ℚ :=
  work.hourly_rate * (work.original_weeks - work.sick_weeks : ℚ) * (work.new_hours_per_week : ℚ)

/-- Theorem stating that the new work schedule achieves the same earnings as the original plan -/
theorem new_schedule_achieves_goal (work : SummerWork) 
  (h1 : work.original_weeks = 15)
  (h2 : work.original_hours_per_week = 15)
  (h3 : work.sick_weeks = 3)
  (h4 : work.new_hours_per_week = 19) :
  total_earnings work = work.hourly_rate * (work.original_weeks : ℚ) * (work.original_hours_per_week : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_schedule_achieves_goal_l207_20707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l207_20747

theorem shaded_areas_equality (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  let r : Real := 1  -- Assume unit circle for simplicity
  let sector_area : Real := (1 / 2) * r^2 * θ
  let triangle_area : Real := (1 / 2) * r^2 * Real.tan θ * Real.sin θ
  (triangle_area - sector_area = sector_area) ↔ Real.tan θ = 2 * θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l207_20747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_island_words_l207_20748

def alphabet_size : ℕ := 24
def max_word_length : ℕ := 4

def words_with_specific_letter (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words_with_specific_letter : ℕ :=
  Finset.sum (Finset.range max_word_length) (λ i ↦ words_with_specific_letter (i + 1))

theorem puzzle_island_words :
  total_words_with_specific_letter = 53640 :=
by
  -- Expand the definition
  unfold total_words_with_specific_letter
  unfold words_with_specific_letter
  -- Evaluate the sum
  simp [alphabet_size, max_word_length]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_island_words_l207_20748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weight_is_60_l207_20718

/-- Represents the initial weight of the vest in pounds -/
def initial_weight : ℝ := 60

/-- Represents the weight increase percentage -/
def weight_increase_percentage : ℝ := 0.60

/-- Represents the weight of each ingot in pounds -/
def ingot_weight : ℝ := 2

/-- Represents the cost of each ingot in dollars -/
def ingot_cost : ℝ := 5

/-- Represents the discount percentage for buying more than 10 ingots -/
def discount_percentage : ℝ := 0.20

/-- Represents the total cost to increase the weight -/
def total_cost : ℝ := 72

/-- Theorem stating that the initial weight of the vest was 60 pounds -/
theorem initial_weight_is_60 :
  initial_weight = 60 := by
  -- The proof goes here
  sorry

#eval initial_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weight_is_60_l207_20718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_l207_20771

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2
  | (n + 1) => (66 ^ (1/4 : ℝ)) * sequence_a n

theorem a_50_value : sequence_a 49 = 66 ^ (49/4 : ℝ) * 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_50_value_l207_20771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_shelter_ratio_l207_20785

/-- Represents the number of cats in an animal shelter -/
structure CatShelter where
  initial : ℕ
  taken_in : ℕ
  adopted : ℕ
  kittens_born : ℕ
  returned : ℕ
  final : ℕ

/-- The ratio of two natural numbers -/
def ratio (a b : ℕ) : ℚ := a / b

theorem cat_shelter_ratio (shelter : CatShelter) : 
  shelter.taken_in = 12 →
  shelter.adopted = 3 →
  shelter.kittens_born = 5 →
  shelter.returned = 1 →
  shelter.final = 19 →
  ratio shelter.initial shelter.taken_in = 1 / 2 := by
  intros h1 h2 h3 h4 h5
  -- The proof steps would go here
  sorry

#check cat_shelter_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_shelter_ratio_l207_20785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l207_20793

-- Define the complex number z
noncomputable def z : ℂ := (-2 + 3 * Complex.I) / Complex.I

-- Theorem statement
theorem z_in_first_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = 1 :=
by
  -- Simplify z
  have h : z = 3 + 2 * Complex.I := by
    -- Proof of simplification
    sorry
  
  -- Show that the real part is positive
  have h_re : z.re = 3 := by
    rw [h]
    simp
  have h_re_pos : Real.sign z.re = 1 := by
    rw [h_re]
    exact Real.sign_of_pos (by norm_num)

  -- Show that the imaginary part is positive
  have h_im : z.im = 2 := by
    rw [h]
    simp
  have h_im_pos : Real.sign z.im = 1 := by
    rw [h_im]
    exact Real.sign_of_pos (by norm_num)

  -- Combine the results
  exact ⟨h_re_pos, h_im_pos⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l207_20793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l207_20724

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_value (m : ℝ) :
  A ∩ B m = Set.Icc 0 3 → m = 2 := by sorry

-- Theorem 2
theorem subset_complement_implies_m_range (m : ℝ) :
  A ⊆ (B m)ᶜ → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_implies_m_range_l207_20724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAF_l207_20750

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (3, 0)

-- Define point A
def point_A : ℝ × ℝ := (0, 8)

-- Define a point P on the hyperbola
def point_P : ℝ × ℝ := (3, 8)

-- Theorem statement
theorem area_of_triangle_PAF :
  hyperbola point_P.fst point_P.snd →
  (point_P.fst - right_focus.fst = 0) →
  ∃ (area : ℝ), area = 12 ∧
    area = (1/2) * |point_P.fst - point_A.fst| * |point_P.snd - right_focus.snd| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAF_l207_20750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_in_factorial_factors_l207_20751

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_difference_in_factorial_factors :
  ∀ p q r s : ℕ+,
  p.val * q.val * r.val * s.val = factorial 9 →
  p < q ∧ q < r ∧ r < s →
  (∀ p' q' r' s' : ℕ+,
    p'.val * q'.val * r'.val * s'.val = factorial 9 →
    p' < q' ∧ q' < r' ∧ r' < s' →
    s.val - p.val ≤ s'.val - p'.val) →
  s.val - p.val = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_in_factorial_factors_l207_20751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_optimal_investment_l207_20720

/-- The profit function for commodity A -/
noncomputable def profit_A (t : ℝ) : ℝ := (1/5) * t

/-- The profit function for commodity B -/
noncomputable def profit_B (t : ℝ) : ℝ := (3/5) * Real.sqrt t

/-- The total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (3 - x)

/-- The theorem stating the maximum profit and optimal investment -/
theorem max_profit_optimal_investment :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ 3 ∧
  total_profit x = 21/20 ∧
  (∀ (y : ℝ), y ≥ 0 → y ≤ 3 → total_profit y ≤ total_profit x) ∧
  x = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_optimal_investment_l207_20720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_sum_of_fractions_l207_20701

open Complex

theorem imaginary_part_sum_of_fractions : 
  (1 / (I - 2) + 1 / (1 - 2*I)).im = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_sum_of_fractions_l207_20701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_combined_resistance_l207_20733

/-- Combined resistance of parallel resistors -/
noncomputable def combined_resistance (x y z : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z)

/-- Theorem: The combined resistance of three parallel resistors with 
    resistances 3 ohms, 5 ohms, and 7 ohms is 105/71 ohms -/
theorem parallel_resistors_combined_resistance :
  combined_resistance 3 5 7 = 105/71 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_combined_resistance_l207_20733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_three_two_relation_l207_20702

theorem log_base_three_two_relation (x : ℝ) :
  x * Real.log 2 / Real.log 3 = 1 → (2 : ℝ)^x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_base_three_two_relation_l207_20702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_average_l207_20754

/-- A monic polynomial of degree n with real coefficients -/
def MonicPolynomial (n : ℕ) := {p : Polynomial ℝ // p.Monic ∧ p.degree = n}

/-- A polynomial with n real roots -/
def PolynomialWithNRealRoots (n : ℕ) := {p : Polynomial ℝ // ∃ (roots : Finset ℝ), roots.card = n ∧ ∀ x ∈ roots, p.eval x = 0}

theorem monic_polynomial_average (n : ℕ) (f : MonicPolynomial n) :
  ∃ (g h : PolynomialWithNRealRoots n),
    g.1.Monic ∧ h.1.Monic ∧ f.1 = (g.1 + h.1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_average_l207_20754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_is_one_over_256_l207_20766

/-- Represents a 4x4 grid of squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The probability of a square being initially black -/
noncomputable def initial_black_prob : ℝ := 1 / 2

/-- Rotates a position 90 degrees clockwise in a 4x4 grid -/
def rotate (i j : Fin 4) : Fin 4 × Fin 4 := sorry

/-- The probability that a 4x4 grid is entirely black after the described operation -/
noncomputable def prob_all_black_after_operation : ℝ := sorry

/-- Main theorem: The probability of the grid being entirely black after the operation is 1/256 -/
theorem prob_all_black_is_one_over_256 : prob_all_black_after_operation = 1 / 256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_is_one_over_256_l207_20766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_28000_l207_20714

/-- Represents a position in the company with its title, number of employees, and salary --/
structure Position where
  title : String
  count : Nat
  salary : Nat
deriving Inhabited

/-- The list of positions in the company --/
def positions : List Position := [
  ⟨"CEO", 1, 150000⟩,
  ⟨"Senior Vice-President", 4, 100000⟩,
  ⟨"Manager", 12, 80000⟩,
  ⟨"Assistant Manager", 8, 60000⟩,
  ⟨"Office Clerk", 58, 28000⟩
]

/-- The total number of employees in the company --/
def totalEmployees : Nat := (positions.map Position.count).sum

/-- Calculates the median salary of the employees --/
def medianSalary : Nat :=
  let medianPosition := (totalEmployees + 1) / 2
  let cumulativeCounts := positions.scanl (fun acc p => acc + p.count) 0
  match cumulativeCounts.findIdx? (· ≥ medianPosition) with
  | some idx => (positions.get! idx).salary
  | none => 0  -- This case should never occur given our data

/-- Theorem stating that the median salary is $28,000 --/
theorem median_salary_is_28000 : medianSalary = 28000 := by
  sorry

#eval medianSalary -- This will output the calculated median salary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_28000_l207_20714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabel_earnings_is_36_l207_20722

/-- The amount of money Isabel earned from selling necklaces at a garage sale -/
def isabel_earnings : ℕ :=
  let bead_necklaces : ℕ := 3
  let gem_stone_necklaces : ℕ := 3
  let price_per_necklace : ℕ := 6
  let total_necklaces : ℕ := bead_necklaces + gem_stone_necklaces
  total_necklaces * price_per_necklace

#eval isabel_earnings -- Should output 36

/-- Proof that Isabel earned 36 dollars -/
theorem isabel_earnings_is_36 : isabel_earnings = 36 := by
  unfold isabel_earnings
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabel_earnings_is_36_l207_20722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_for_tan_negative_one_third_l207_20700

theorem cos_double_angle_for_tan_negative_one_third (θ : ℝ) :
  Real.tan θ = -1/3 → Real.cos (2 * θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_for_tan_negative_one_third_l207_20700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_implies_m_l207_20757

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop := x^2 / 4 + y^2 / m = 1

-- Define the eccentricity
noncomputable def eccentricity (m : ℝ) : ℝ := 
  if m > 4 then Real.sqrt (m - 4) / Real.sqrt m
  else Real.sqrt (4 - m) / 2

-- Theorem statement
theorem ellipse_eccentricity_implies_m (m : ℝ) :
  (∃ x y : ℝ, ellipse_equation x y m) →
  eccentricity m = Real.sqrt 3 / 2 →
  m = 1 ∨ m = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_implies_m_l207_20757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samia_walking_distance_l207_20769

/-- Represents the problem of calculating Samia's walking distance --/
theorem samia_walking_distance :
  ∀ (total_distance : ℝ) (jogging_distance : ℝ) (walking_distance : ℝ)
    (jogging_speed : ℝ) (walking_speed : ℝ) (total_time : ℝ),
  jogging_speed = 8 →
  walking_speed = 4 →
  total_distance = jogging_distance + walking_distance →
  jogging_distance = (1/3) * total_distance →
  walking_distance = (2/3) * total_distance →
  total_time = 105/60 →
  (jogging_distance / jogging_speed + walking_distance / walking_speed) = total_time →
  walking_distance = 5.6 := by
  intros total_distance jogging_distance walking_distance jogging_speed walking_speed total_time
  intros h_jogging_speed h_walking_speed h_total_distance h_jogging_distance h_walking_distance h_total_time h_time_equation
  sorry

#check samia_walking_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samia_walking_distance_l207_20769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_bus_fare_and_passenger_change_l207_20758

/-- Demand function --/
def demand (p : ℝ) : ℝ := 4200 - 100 * p

/-- Bus company cost function --/
def busCompanyCost (y : ℝ) : ℝ := 10 * y + 225

/-- Train fare --/
def trainFare : ℝ := 4

/-- Train capacity --/
def trainCapacity : ℝ := 800

/-- Profit function for bus company with train service --/
def profitWithTrain (p : ℝ) : ℝ :=
  p * (demand p - trainCapacity) - busCompanyCost (demand p - trainCapacity)

/-- Profit function for bus company without train service --/
def profitWithoutTrain (p : ℝ) : ℝ :=
  p * demand p - busCompanyCost (demand p)

/-- Optimal bus fare with train service --/
noncomputable def optimalBusFareWithTrain : ℝ := 22

/-- Optimal bus fare without train service --/
noncomputable def optimalBusFareWithoutTrain : ℝ := 26

/-- Theorem stating the optimal bus fare and change in passengers --/
theorem optimal_bus_fare_and_passenger_change :
  optimalBusFareWithTrain = 22 ∧
  demand optimalBusFareWithTrain - trainCapacity + trainCapacity -
    demand optimalBusFareWithoutTrain = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_bus_fare_and_passenger_change_l207_20758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_M_subset_A_l207_20729

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 ≤ 5*x - 4}

-- Define the set M (parameterized by a)
def M (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+2)*x + 2*a ≤ 0}

-- Theorem statement
theorem range_of_a_for_M_subset_A :
  {a : ℝ | ∀ x : ℝ, x ∈ M a → x ∈ A} = Set.Icc 1 4 := by
  sorry

-- Where:
-- Set.Icc 1 4 represents the closed interval [1, 4]
-- ∀ x : ℝ, x ∈ M a → x ∈ A expresses M ⊆ A

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_M_subset_A_l207_20729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_120_degrees_from_bisector_relation_l207_20731

/-- Given a triangle ABC with sides a, b, c and angle bisector l_a from vertex A,
    if 1/b + 1/c = 1/l_a, then angle A is 120 degrees. -/
theorem angle_120_degrees_from_bisector_relation (a b c l_a : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < l_a) :
  (1 / b + 1 / c = 1 / l_a) → ∃ A : ℝ, A = 120 * Real.pi / 180 ∧ Real.cos (A / 2) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_120_degrees_from_bisector_relation_l207_20731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_condition_l207_20734

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) ^ (a * x^2 - 4*x + 3)

-- Part 1: Monotonicity intervals when a = -1
theorem monotonicity_intervals :
  (∀ x y : ℝ, x < -2 → y < -2 → x < y → f (-1) y < f (-1) x) ∧
  (∀ x y : ℝ, -2 < x → -2 < y → x < y → f (-1) x < f (-1) y) :=
sorry

-- Part 2: Value of a when f(x) has a maximum value of 3
theorem max_value_condition :
  ∀ a : ℝ, (∃ x : ℝ, f a x = 3) ∧ (∀ y : ℝ, f a y ≤ 3) → a = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_condition_l207_20734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_sixth_l207_20795

/-- Rectangle ABCD with dimensions AB = 2 and BC = 1 -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  h_AB : AB = 2
  h_BC : BC = 1

/-- Square EFGH with side length 1 -/
structure Square where
  side : ℝ
  h_side : side = 1

/-- Configuration of the shapes -/
structure Configuration where
  rect : Rectangle
  sq : Square
  /-- C is the midpoint of FH -/
  h_C_midpoint : True
  /-- D is the midpoint of HE -/
  h_D_midpoint : True
  /-- E is at point A -/
  h_E_at_A : True
  /-- H is at point B -/
  h_H_at_B : True

/-- Area of quadrilateral AJEC -/
def area_quadrilateral_AJEC (config : Configuration) : ℝ := 0.5

/-- Area of rectangle -/
def area_rectangle (rect : Rectangle) : ℝ := rect.AB * rect.BC

/-- Area of square -/
def area_square (sq : Square) : ℝ := sq.side * sq.side

/-- The main theorem -/
theorem area_ratio_is_one_sixth (config : Configuration) :
  (area_quadrilateral_AJEC config) / (area_rectangle config.rect + area_square config.sq) = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_sixth_l207_20795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_and_symmetric_points_l207_20782

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the relation between A and P
def AP_relation (A P : ℝ × ℝ) : Prop :=
  P.1 - A.1 = -2 * (A.1 - F.1) ∧ P.2 - A.2 = -2 * (A.2 - F.2)

-- Define the symmetric point with respect to y = 2x
noncomputable def symmetric_point (Q : ℝ × ℝ) : ℝ × ℝ :=
  let t := Q.1
  (-3/5 * t, 4/5 * t)

theorem trajectory_of_P_and_symmetric_points :
  (∀ A P : ℝ × ℝ, C A.1 A.2 → AP_relation A P → C (2 - P.1) (-P.2)) ∧
  (∀ Q : ℝ × ℝ, Q.2 = 0 → 
    (C (symmetric_point Q).1 (symmetric_point Q).2 ↔ Q = (0, 0) ∨ Q = (-15/4, 0))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_and_symmetric_points_l207_20782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_property_a_general_term_a_initial_l207_20753

noncomputable def a (n : ℕ) : ℝ := 1 / (n : ℝ)^2

theorem a_property (n : ℕ) (hn : n > 0) :
  Real.sqrt (a n) - Real.sqrt (a (n + 1)) = Real.sqrt (a n * a (n + 1)) :=
by sorry

theorem a_general_term (n : ℕ) (hn : n > 0) :
  a n = 1 / (n : ℝ)^2 :=
by sorry

-- Initial condition
theorem a_initial : a 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_property_a_general_term_a_initial_l207_20753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_through_points_l207_20741

/-- Given a circle with center on the x-axis passing through points (0,5) and (2,4),
    prove that its radius is 5√17/4 -/
theorem circle_radius_through_points : 
  ∃ (x : ℝ), 
    (∀ (y : ℝ), y = 0 → 
      (x - 0)^2 + (y - 5)^2 = (x - 2)^2 + (y - 4)^2) →
    Real.sqrt ((x - 0)^2 + 5^2) = (5 * Real.sqrt 17) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_through_points_l207_20741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_difference_l207_20772

theorem quadratic_solution_difference : 
  ∃ r₁ r₂ : ℝ, (r₁^2 - 7*r₁ + 10 = 0) ∧ (r₂^2 - 7*r₂ + 10 = 0) ∧ |r₁ - r₂| = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_difference_l207_20772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_15_l207_20798

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the condition that f⁻¹(g(x)) = x^4 - 4
axiom f_g_relation : ∀ x : ℝ, Function.invFun f (g x) = x^4 - 4

-- Define that g has an inverse
axiom g_has_inverse : Function.Bijective g

-- State the theorem to be proved
theorem g_inverse_f_15 : Function.invFun g (f 15) = (19 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_f_15_l207_20798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetry_point_l207_20740

noncomputable def f (x : ℝ) := Real.tan (x / 2 - Real.pi / 6)

theorem not_symmetry_point :
  f (2 * Real.pi / 3) ≠ 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_symmetry_point_l207_20740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_equivalent_to_dips_l207_20788

/-- Represents the number of units of a particular currency -/
structure Currency where
  amount : ℚ
  name : String

/-- Defines the exchange rate between two currencies -/
def exchange_rate (from_currency to_currency : Currency) : ℚ := 
  from_currency.amount / to_currency.amount

/-- Given conditions of the problem -/
axiom daps_to_dops : 
  exchange_rate (Currency.mk 5 "daps") (Currency.mk 4 "dops") = 1

axiom dops_to_dips : 
  exchange_rate (Currency.mk 3 "dops") (Currency.mk 10 "dips") = 1

/-- The theorem to be proved -/
theorem daps_equivalent_to_dips : 
  exchange_rate (Currency.mk 10.5 "daps") (Currency.mk 28 "dips") = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daps_equivalent_to_dips_l207_20788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l207_20717

/-- Converts polar coordinates (r, θ) to rectangular coordinates (x, y) -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given point in polar coordinates -/
noncomputable def polar_point : ℝ × ℝ := (2 * Real.sqrt 3, 2 * Real.pi / 3)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ := (-Real.sqrt 3, 3)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular polar_point.1 polar_point.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l207_20717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_area_ratio_l207_20743

/-- Definition of a triangle -/
structure Triangle where
  -- We'll leave this as a placeholder for now
  mk :: 

/-- Similarity relation between two triangles -/
def Similar (T1 T2 : Triangle) : Prop := 
  sorry

/-- Ratio of medians of two triangles -/
def MedianRatio (T1 T2 : Triangle) : ℝ := 
  sorry

/-- Ratio of areas of two triangles -/
def AreaRatio (T1 T2 : Triangle) : ℝ := 
  sorry

/-- Given two similar triangles with a median ratio of 1:√2, prove their area ratio is 1:2 -/
theorem similar_triangles_area_ratio 
  (T1 T2 : Triangle) 
  (h_similar : Similar T1 T2) 
  (h_median_ratio : MedianRatio T1 T2 = Real.sqrt 2) : 
  AreaRatio T1 T2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_area_ratio_l207_20743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_linear_expr_condition_l207_20797

noncomputable def F (x m : ℝ) : ℝ := (8 * x^2 + 20 * x + 5 * m) / 8

noncomputable def linear_expr (a b x : ℝ) : ℝ := a * x + b

theorem square_of_linear_expr_condition (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, F x m = (linear_expr a b x)^2) ↔ m = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_linear_expr_condition_l207_20797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_avoid_losing_l207_20790

/-- A strategy for picking vectors -/
def Strategy := List (Fin 1992) → Fin 1992

/-- The game state after all vectors have been picked -/
structure GameState (v : Fin 1992 → ℝ × ℝ) where
  firstPlayerSum : ℝ × ℝ
  secondPlayerSum : ℝ × ℝ

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- Determine the game result based on the final state -/
noncomputable def gameResult (state : GameState v) : GameResult :=
  let firstMagnitude := Real.sqrt (state.firstPlayerSum.1^2 + state.firstPlayerSum.2^2)
  let secondMagnitude := Real.sqrt (state.secondPlayerSum.1^2 + state.secondPlayerSum.2^2)
  if firstMagnitude > secondMagnitude then
    GameResult.FirstPlayerWins
  else if secondMagnitude > firstMagnitude then
    GameResult.SecondPlayerWins
  else
    GameResult.Draw

/-- Play the game with a given strategy for the first player -/
def playGame (v : Fin 1992 → ℝ × ℝ) (strategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating that there exists a strategy for the first player to always avoid losing -/
theorem first_player_can_avoid_losing :
  ∃ (strategy : Strategy), ∀ (v : Fin 1992 → ℝ × ℝ),
    playGame v strategy ≠ GameResult.SecondPlayerWins := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_avoid_losing_l207_20790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_product_l207_20787

theorem last_three_digits_product (a b c : ℕ) : 
  ((a + b) % 10 = c % 10) ∧ 
  ((b + c) % 10 = a % 10) ∧ 
  ((c + a) % 10 = b % 10) →
  ∃ (x : ℕ), x ∈ ({0, 250, 500, 750} : Set ℕ) ∧ (a * b * c) % 1000 = x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_product_l207_20787
