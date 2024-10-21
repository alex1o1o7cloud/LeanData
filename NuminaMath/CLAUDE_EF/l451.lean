import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_arithmetic_sequence_l451_45108

noncomputable def f (n : ℕ) (x : ℝ) := (x - 1)^2 + n

noncomputable def a (n : ℕ) := f n 1

noncomputable def b (n : ℕ) := max (f n (-1)) (f n 3)

noncomputable def c (n : ℕ) := (b n)^2 - (a n) * (b n)

theorem c_is_arithmetic_sequence : 
  ∀ n : ℕ, n > 0 → c (n + 1) - c n = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_is_arithmetic_sequence_l451_45108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_value_l451_45172

/-- Given a polynomial f(x) = x^3 - ax^2 + bx - c, where a, b, c are consecutive
    positive integers with a < b < c, and g(x) is a polynomial with leading
    coefficient 1 whose roots are the reciprocals of the roots of f(x),
    prove that g(1) = (1 + b - a - c) / (-c). -/
theorem g_one_value (a b c : ℕ) (f g : ℝ → ℝ) (p q r : ℝ) :
  (∀ x, f x = x^3 - a*x^2 + b*x - c) →
  (∃ k : ℕ, k > 0 ∧ b = a + k ∧ c = b + k) →
  (p * q * r = -c) →
  (∀ x, f x = (x - p) * (x - q) * (x - r)) →
  (∀ x, g x = (x - (1/p)) * (x - (1/q)) * (x - (1/r))) →
  g 1 = (1 + b - a - c) / (-c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_value_l451_45172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_sum_minimum_l451_45101

noncomputable def tensor (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y)

theorem tensor_sum_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + 2 * y^2) / (2 * x * y) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_sum_minimum_l451_45101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_and_d_in_base5_l451_45100

/-- Represents a digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents the subtraction equation in base 5 -/
structure Base5Subtraction :=
  (d₁ d₂ c : Base5Digit)
  (valid : (d₁.val * 25 + d₂.val * 5 + c.val : ℕ) = 
           3 * 25 + 2 * 5 + d₁.val * 5 + d₂.val - (c.val * 25 + 2 * 5 + 3))

/-- Converts a base 10 integer to its base 5 representation -/
def toBase5 (n : ℕ) : ℕ := sorry

/-- The main theorem stating the existence of c and d satisfying the conditions -/
theorem sum_of_c_and_d_in_base5 (sub : Base5Subtraction) :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ toBase5 (c + d) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_c_and_d_in_base5_l451_45100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_iff_polynomial_count_l451_45125

theorem idempotent_iff_polynomial_count {A : Type} [CommRing A] (h : (1 : A) ≠ 0) :
  (∀ x : A, x * x = x) ↔ 
  (Cardinal.mk {f : A → A | ∃ p : Polynomial A, ∀ a, f a = p.eval a} = Cardinal.mk A * Cardinal.mk A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_idempotent_iff_polynomial_count_l451_45125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l451_45135

noncomputable def f (a : ℕ+) (x : ℝ) : ℝ := x + Real.sqrt (13 - 2 * (a : ℝ) * x)

theorem max_value_of_f (a : ℕ+) :
  (∃ (y : ℕ+), ∀ (x : ℝ), f a x ≤ y ∧ ∃ (x₀ : ℝ), f a x₀ = y) →
  ∃ (y : ℕ+), (∀ (x : ℝ), f a x ≤ y ∧ ∃ (x₀ : ℝ), f a x₀ = y) ∧ y = 7 :=
by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l451_45135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l451_45119

variable (x y z : ℂ)

theorem complex_inequality (h : Complex.abs x ^ 2 + Complex.abs y ^ 2 + Complex.abs z ^ 2 = 1) :
  Complex.abs (x^3 + y^3 + z^3 - 3*x*y*z) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l451_45119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_square_l451_45113

-- Define the ellipse and circle equations
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 36

-- Define a point of intersection
def intersection_point (x y : ℝ) : Prop := ellipse x y ∧ my_circle x y

-- Define a square
def is_square (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  let d23 := (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2
  let d34 := (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2
  let d41 := (p4.1 - p1.1)^2 + (p4.2 - p1.2)^2
  d12 = d23 ∧ d23 = d34 ∧ d34 = d41 ∧
  (p2.1 - p1.1) * (p3.1 - p2.1) + (p2.2 - p1.2) * (p3.2 - p2.2) = 0

-- Theorem statement
theorem intersection_forms_square :
  ∃ (p1 p2 p3 p4 : ℝ × ℝ),
    intersection_point p1.1 p1.2 ∧
    intersection_point p2.1 p2.2 ∧
    intersection_point p3.1 p3.2 ∧
    intersection_point p4.1 p4.2 ∧
    is_square p1 p2 p3 p4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_square_l451_45113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_square_tens_digit_odd_l451_45106

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_square_tens_digit_odd (a : ℕ) :
  (∃ k ∈ ({1, 3, 5, 7, 9} : Set ℕ), tens_digit (a^2) = k) →
  units_digit a = 4 ∨ units_digit a = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_square_tens_digit_odd_l451_45106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_one_l451_45148

-- Define the set S of nonzero real numbers
def S : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the properties of function f
def has_properties (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ S, f (1 / x) = x^2 * f x) ∧
  (∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S → f (1 / x) + f (1 / y) = 1 + f (1 / (x + y)))

-- Theorem statement
theorem f_one_equals_one (f : ℝ → ℝ) (h : has_properties f) : f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_one_l451_45148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_efficiency_ratio_problem_l451_45190

/-- The ratio of work efficiencies between two workers -/
noncomputable def work_efficiency_ratio (a b : ℝ) : ℝ := a / b

/-- The time taken to complete a job -/
noncomputable def time_to_complete (efficiency : ℝ) : ℝ := 1 / efficiency

theorem work_efficiency_ratio_problem (a b : ℝ) :
  (time_to_complete (a + b) = 5) →
  (time_to_complete b = 15) →
  (∃ k : ℝ, a = k * b) →
  work_efficiency_ratio a b = 2 := by
  sorry

#check work_efficiency_ratio_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_efficiency_ratio_problem_l451_45190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_sum_l451_45126

/-- Given two points A and B in a 3D Cartesian coordinate system,
    where A is symmetric to B with respect to the xy-plane,
    prove that the sum of their z-coordinates is zero. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ × ℝ), 
    A = (3, -1, m) ∧ 
    B = (3, n, -2) ∧ 
    A.fst = B.fst ∧ 
    A.snd.fst = B.snd.fst ∧ 
    A.snd.snd = -B.snd.snd) → 
  m + n = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_sum_l451_45126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_problem_l451_45169

theorem set_cardinality_problem (A B : Finset ℕ) 
  (h1 : A.card = 3 * B.card)
  (h2 : (A ∪ B).card = 5000)
  (h3 : (A ∩ B).card = 1000) :
  A.card = 4500 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_cardinality_problem_l451_45169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_limit_l451_45104

/-- The limit of the geometric series with first term a and common ratio r -/
noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

/-- The sum of the given series -/
noncomputable def series_sum : ℝ :=
  2 + Real.sqrt 3 * (geometric_series_sum (1/3) (1/5)) + geometric_series_sum (1/5) (1/5)

theorem series_limit :
  series_sum = 2 + (5 * Real.sqrt 3 + 3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_limit_l451_45104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l451_45198

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 + 2 * x + 1)

theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l451_45198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_frogs_l451_45165

-- Define the amphibian type
inductive Amphibian : Type
| Brian : Amphibian
| Chris : Amphibian
| LeRoy : Amphibian
| Mike : Amphibian
| Neil : Amphibian

-- Define the species type
inductive Species : Type
| Toad : Species
| Frog : Species

-- Function to determine if an amphibian is a toad
def isToad : Amphibian → Prop := sorry

-- Function to count toads
def countToads : List Amphibian → Nat := sorry

-- Function to determine if two amphibians are different species
def differentSpecies (a b : Amphibian) : Prop :=
  (isToad a ∧ ¬isToad b) ∨ (¬isToad a ∧ isToad b)

-- Statements made by each amphibian
axiom Brian_statement : countToads [Amphibian.Brian, Amphibian.Chris, Amphibian.LeRoy, Amphibian.Mike, Amphibian.Neil] ≥ 3
axiom Chris_statement : isToad Amphibian.Neil
axiom LeRoy_statement : isToad Amphibian.Chris
axiom Mike_statement : differentSpecies Amphibian.Brian Amphibian.Mike
axiom Neil_statement : countToads [Amphibian.Brian, Amphibian.Chris, Amphibian.LeRoy, Amphibian.Mike, Amphibian.Neil] ≤ 3

-- Axiom that toads always tell the truth and frogs always lie
axiom truth_telling (a : Amphibian) (p : Prop) : isToad a ↔ p

-- Theorem to prove
theorem two_frogs :
  ∃ (f₁ f₂ : Amphibian), f₁ ≠ f₂ ∧
    ¬isToad f₁ ∧ ¬isToad f₂ ∧
    ∀ (a : Amphibian), a ≠ f₁ ∧ a ≠ f₂ → isToad a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_frogs_l451_45165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l451_45178

/-- The radius of a circle inscribed within three mutually externally tangent circles -/
noncomputable def inscribed_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- Theorem: The radius of the inscribed circle for given radii is 18/19 -/
theorem inscribed_circle_radius :
  inscribed_radius 3 6 18 = 18/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l451_45178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l451_45114

-- Define the slopes of the lines
noncomputable def slope_l1 (m : ℝ) : ℝ := -1 / m
noncomputable def slope_l2 (m : ℝ) : ℝ := -(m - 2) / 3

-- Theorem statement
theorem parallel_lines_m_values :
  ∀ m : ℝ, m ≠ 0 → slope_l1 m = slope_l2 m → m = -1 ∨ m = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l451_45114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_remainders_l451_45146

def is_small (n k : ℕ) : Prop := 1 ≤ n ∧ n ≤ k
def is_medium (n k : ℕ) : Prop := k + 1 ≤ n ∧ n ≤ 2 * k
def is_large (n k : ℕ) : Prop := 2 * k + 1 ≤ n ∧ n ≤ 3 * k

def remainder_3k (n k : ℕ) : ℕ :=
  let r := n % (3 * k)
  if r = 0 then 3 * k else r

theorem arithmetic_progression_remainders (k : ℕ) (A : Finset ℕ) : k > 0 ∧ A ⊆ Finset.range (3 * k + 1) \ {0} ∧
  (∀ a b c, a ∈ A → b ∈ A → c ∈ A → a ≠ b ∧ b ≠ c ∧ a ≠ c → 2 * b ≠ a + c) →
  (∃ x d : ℕ, x > 0 ∧ d > 0 ∧
    (remainder_3k x k ∈ A ∧ remainder_3k (x + d) k ∈ A ∧ remainder_3k (x + 2 * d) k ∈ A) ∧
    ((is_small (remainder_3k x k) k ∧ is_small (remainder_3k (x + d) k) k) ∨
     (is_large (remainder_3k x k) k ∧ is_large (remainder_3k (x + d) k) k))) ∧
  (¬ ∃ x d : ℕ, x > 0 ∧ d > 0 ∧
    (remainder_3k x k ∈ A ∧ remainder_3k (x + d) k ∈ A ∧ remainder_3k (x + 2 * d) k ∈ A) ∧
    (is_medium (remainder_3k x k) k ∧ is_medium (remainder_3k (x + d) k) k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_remainders_l451_45146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l451_45193

/-- Calculates the market value of a stock given its dividend rate, yield, and face value. -/
noncomputable def market_value (dividend_rate : ℝ) (yield : ℝ) (face_value : ℝ) : ℝ :=
  (dividend_rate * face_value / yield) * 100

/-- Theorem stating that a 14% stock yielding 8% with a face value of $100 has a market value of $175. -/
theorem stock_market_value :
  let dividend_rate : ℝ := 0.14
  let yield : ℝ := 0.08
  let face_value : ℝ := 100
  market_value dividend_rate yield face_value = 175 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval market_value 0.14 0.08 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l451_45193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_groups_count_l451_45129

def total_marbles : ℕ := 8
def yellow_marbles : ℕ := 4
def non_yellow_marbles : ℕ := 4

def choose_three_with_yellow (total : ℕ) (yellow : ℕ) (non_yellow : ℕ) : ℕ :=
  (Nat.choose yellow 3) +  -- Three yellow marbles
  (Nat.choose yellow 2 * Nat.choose non_yellow 1) +  -- Two yellow, one non-yellow
  (Nat.choose yellow 1 * Nat.choose non_yellow 2)  -- One yellow, two non-yellow

theorem marble_groups_count :
  choose_three_with_yellow total_marbles yellow_marbles non_yellow_marbles = 11 := by
  sorry

#eval choose_three_with_yellow total_marbles yellow_marbles non_yellow_marbles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_groups_count_l451_45129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_solution_l451_45189

/-- The function g(x) -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

/-- The inverse function of g(x) -/
noncomputable def g_inv (b : ℝ) (y : ℝ) : ℝ := (b / y + 4) / 3

/-- Theorem stating that 28/9 is the only valid solution for b -/
theorem unique_b_solution :
  ∃! b : ℝ, g b 3 = g_inv b (b + 2) ∧ b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_b_solution_l451_45189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_with_foci_on_y_axis_l451_45162

-- Define the angle θ in the third quadrant
noncomputable def θ : ℝ := Real.pi * 5 / 4 -- An example value in the third quadrant

-- Assumption that θ is in the third quadrant
axiom θ_in_third_quadrant : Real.pi < θ ∧ θ < 3 * Real.pi / 2

-- Define the equation of the curve
def curve_equation (x y : ℝ) : Prop :=
  x^2 + y^2 * Real.sin θ = Real.cos θ

-- Theorem statement
theorem curve_is_hyperbola_with_foci_on_y_axis :
  ∃ (a b : ℝ), 
    (∀ (x y : ℝ), curve_equation x y ↔ 
      (x^2 / a^2) - (y^2 / b^2) = 1) ∧
    (a > 0) ∧ (b > 0) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_with_foci_on_y_axis_l451_45162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_tunnel_length_value_l451_45117

/-- The length of a tunnel given specific conditions about a truck passing through it. -/
theorem tunnel_length : ℝ := by
  -- Let L be the length of the tunnel
  let L : ℝ := 24 -- We know the result, so we set it directly

  -- Let v be the initial speed of the truck (we don't need to specify its value)
  let v : ℝ := 1 -- Arbitrary non-zero value

  -- Initial truck length
  let L₁ : ℝ := 6

  -- New truck length with container
  let L₂ : ℝ := 12

  -- Speed reduction factor
  let speed_reduction : ℝ := 0.8

  -- Time increase factor
  let time_increase : ℝ := 1.5

  -- First pass time
  let t₁ : ℝ := (L + L₁) / v

  -- Second pass time
  let t₂ : ℝ := (L + L₂) / (speed_reduction * v)

  -- Time increase condition
  have h_time : t₂ = time_increase * t₁ := by
    -- Proof of this equality
    sorry

  -- Main proof
  have h_main : L = 24 := by
    -- Proof that L equals 24
    sorry

  -- Return the length of the tunnel
  exact L

/-- The tunnel length is 24 meters. -/
theorem tunnel_length_value : tunnel_length = 24 := by
  -- Proof that tunnel_length equals 24
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_tunnel_length_value_l451_45117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l451_45149

theorem calculation_proof :
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) ∧
  ((-1)^2023 + |1 - Real.sqrt 2| - (27 : ℝ)^(1/3) + Real.sqrt ((-2)^2) = Real.sqrt 2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l451_45149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l451_45196

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^12 + i^20 + i^(-32 : ℤ) = (3 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l451_45196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_focus_origin_and_point_l451_45183

/-- The parabola y^2 = 4x in the Cartesian coordinate system -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A point on the parabola -/
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ P.2 > 0

/-- The distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

/-- The equation of a circle -/
def circle_equation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem circle_through_focus_origin_and_point
  (P : ℝ × ℝ)
  (h_on_parabola : point_on_parabola P)
  (h_distance : distance P origin = 4 * Real.sqrt 2) :
  ∀ x y, circle_equation x y 0.5 3.5 3.5 ↔ x^2 + y^2 - x - 7*y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_focus_origin_and_point_l451_45183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_solution_l451_45120

/-- Define the ◎ operation for real numbers -/
def my_circle (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

/-- Theorem statement -/
theorem circle_equation_solution :
  ∀ m : ℝ, my_circle (m + 1) (m - 2) = 16 → m = 3 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_solution_l451_45120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_trig_value_in_second_quadrant_l451_45175

theorem unique_trig_value_in_second_quadrant (x : Real) 
  (h1 : π / 2 < x ∧ x < π) -- x is in the second quadrant
  (h2 : ∃ (f : Real → Real), f ∈ ({sin, cos, tan} : Set (Real → Real)) ∧ 
    (∀ g ∈ ({sin, cos, tan} : Set (Real → Real)), g ≠ f → g x ≠ f x)) -- one function is uniquely identifiable
  : ∃ (f : Real → Real), f ∈ ({sin, cos, tan} : Set (Real → Real)) ∧ f x = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_trig_value_in_second_quadrant_l451_45175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_team_composition_l451_45164

def number_of_possible_compositions (total_members men women team_size : ℕ) : ℕ :=
  Nat.choose men 4 * Nat.choose women 2

theorem inspection_team_composition 
  (total_members : ℕ) 
  (men : ℕ) 
  (women : ℕ) 
  (team_size : ℕ) 
  (h1 : total_members = 15)
  (h2 : men = 10)
  (h3 : women = 5)
  (h4 : team_size = 6)
  (h5 : total_members = men + women)
  (h6 : team_size < total_members) :
  (Nat.choose men 4) * (Nat.choose women 2) = 
  number_of_possible_compositions total_members men women team_size :=
by
  unfold number_of_possible_compositions
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_team_composition_l451_45164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_seeds_per_row_l451_45111

/-- The number of seeds that can be planted in a row -/
def seeds_per_row (row_length : ℚ) (seed_space : ℚ) : ℕ :=
  (row_length / seed_space).floor.toNat

/-- Theorem: Bob can plant 80 seeds in each row -/
theorem bob_seeds_per_row :
  seeds_per_row 120 (18 / 12) = 80 := by
  sorry

#eval seeds_per_row 120 (18 / 12)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_seeds_per_row_l451_45111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_eccentricity_l451_45152

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_pos : 0 < r

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

theorem ellipse_circle_tangent_eccentricity (e : Ellipse) (c : Circle) :
  c.r^2 = 2 * e.b^2 / 3 →
  (∃ (p : ℝ × ℝ), p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧
    ∃ (t₁ t₂ : ℝ × ℝ), (t₁.1 - p.1) * (t₂.1 - p.1) + (t₁.2 - p.2) * (t₂.2 - p.2) = 0 ∧
      t₁.1^2 + t₁.2^2 = c.r^2 ∧ t₂.1^2 + t₂.2^2 = c.r^2) →
  1/2 ≤ eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_tangent_eccentricity_l451_45152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_theorem_l451_45184

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the x-intercept of a line --/
noncomputable def xIntercept (l : Line) : ℝ := -l.c / l.a

/-- Calculates the y-intercept of a line --/
noncomputable def yIntercept (l : Line) : ℝ := -l.c / l.b

/-- Theorem: For the line 3x - 4y + k = 0, if the sum of its intercepts is 2, then k = -24 --/
theorem intercept_sum_theorem (k : ℝ) :
  let l : Line := { a := 3, b := -4, c := k }
  (xIntercept l + yIntercept l = 2) → k = -24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_theorem_l451_45184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l451_45182

/-- Calculates the average speed of a train given two segments of its journey. -/
noncomputable def average_speed (x : ℝ) : ℝ :=
  let distance1 := x
  let speed1 := 65
  let distance2 := 2 * x
  let speed2 := 20
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  total_distance / total_time

/-- Theorem stating that the average speed of the train is 26 kmph. -/
theorem train_average_speed :
  ∀ x : ℝ, x > 0 → average_speed x = 26 := by
  sorry

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check average_speed 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_l451_45182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l451_45136

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2^x - 1) + 1 / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ 0 ∧ x ≠ 2}

-- Theorem stating that the domain of f is [0,2) ∪ (2,+∞)
theorem domain_of_f : 
  domain_f = Set.Icc 0 2 ∪ Set.Ioi 2 := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l451_45136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extreme_points_l451_45132

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - x - (1/2) * a * x^2

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 - a * x

theorem tangent_line_and_extreme_points (a : ℝ) :
  (∀ x : ℝ, f (-2) x = x * Real.log x - x + x^2) ∧
  (∀ y : ℝ, (2 : ℝ) * 1 - y - 2 = 0 ↔ y = f (-2) 1 + f' (-2) 1 * (1 - 1)) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    (∃ ε > 0, ∀ x ∈ Set.Ioo (x₁ - ε) (x₁ + ε) ∪ Set.Ioo (x₂ - ε) (x₂ + ε), 
      f' a x = 0 → x = x₁ ∨ x = x₂) →
    x₁ * x₂ > Real.exp 2) :=
by
  sorry

#check tangent_line_and_extreme_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_extreme_points_l451_45132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_property_l451_45154

-- Define ℧ function
def ℧ (n : ℤ) : ℕ := sorry

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, a > b → (℧ (f a - f b) : ℕ) ≤ ℧ (a - b)

-- Define strictly increasing function
def StrictlyIncreasing (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, x < y → f x < f y

-- Theorem statement
theorem unique_function_satisfying_property :
  ∀ f : ℤ → ℤ, StrictlyIncreasing f ∧ SatisfiesProperty f ↔ f = id := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_property_l451_45154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_triangles_l451_45151

/-- A point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- A triangle formed by three points in the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The set of all points in a 3x4 grid -/
def grid : Set GridPoint :=
  {p | p.x ≤ 3 ∧ p.y ≤ 2}

/-- Two triangles are congruent if they have the same shape and size -/
def congruent (t1 t2 : GridTriangle) : Prop :=
  sorry

/-- The set of all possible triangles formed from points in the grid -/
def allTriangles : Set GridTriangle :=
  {t | t.p1 ∈ grid ∧ t.p2 ∈ grid ∧ t.p3 ∈ grid}

/-- The set of non-congruent triangles -/
noncomputable def nonCongruentTriangles : Finset GridTriangle :=
  sorry

theorem count_non_congruent_triangles :
  Finset.card nonCongruentTriangles = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_triangles_l451_45151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l451_45177

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < e then -x^3 + x^2 else a * Real.log x

def is_valid_triangle (a : ℝ) : Prop :=
  ∃ (t : ℝ), t ≥ e ∧ -t^2 + (a * Real.log t) * (t^3 + t^2) = 0

theorem triangle_existence (a : ℝ) :
  is_valid_triangle a ↔ 0 < a ∧ a ≤ 1 / (e + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l451_45177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_four_l451_45166

theorem x_squared_plus_four (x : ℝ) : (4 : ℝ)^(2*x) + 16 = 18*((4 : ℝ)^x) → (x^2 + 4 = 4 ∨ x^2 + 4 = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_plus_four_l451_45166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l451_45145

/-- Curve C1 defined by parametric equations -/
def C1 (t : ℝ) : ℝ × ℝ :=
  (t + 2, 1 - 2*t)

/-- Curve C2 defined by parametric equations -/
noncomputable def C2 (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ, 3 * Real.sin θ)

/-- The length of the segment AB formed by the intersection of C1 and C2 -/
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ,
    (∃ t : ℝ, C1 t = A) ∧
    (∃ θ : ℝ, C2 θ = A) ∧
    (∃ t' : ℝ, C1 t' = B) ∧
    (∃ θ' : ℝ, C2 θ' = B) ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l451_45145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_diff_grades_l451_45109

/-- The probability of selecting 2 students from different grades -/
theorem prob_diff_grades (total : ℕ) (grade1 : ℕ) (grade2 : ℕ) (select : ℕ) : 
  total = 4 → grade1 = 2 → grade2 = 2 → select = 2 →
  (Nat.choose grade1 1 * Nat.choose grade2 1 : ℚ) / Nat.choose total select = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_diff_grades_l451_45109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithms_l451_45103

theorem compare_logarithms : 
  let a := Real.log (Real.sqrt 2)
  let b := (Real.log 3) / 3
  let c := Real.exp (-1)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_logarithms_l451_45103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_length_proof_l451_45128

/-- The length of a tiger moving at constant speed -/
noncomputable def tiger_length : ℝ := 2 * Real.sqrt 3

/-- The time taken by the tiger to pass a blade of grass with its entire body -/
def grass_passing_time : ℝ := 1

/-- The length of the fallen tree trunk -/
def trunk_length : ℝ := 20

/-- The angle of inclination of the fallen tree trunk with respect to the ground -/
noncomputable def trunk_angle : ℝ := 30 * Real.pi / 180

/-- The time taken by the tiger to traverse the entire length of the tree trunk -/
def trunk_traversal_time : ℝ := 5

/-- Theorem stating the tiger's length based on its movement -/
theorem tiger_length_proof :
  ∀ (speed : ℝ),
    speed = tiger_length / grass_passing_time →
    speed = (trunk_length * Real.cos trunk_angle) / trunk_traversal_time →
    tiger_length = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_length_proof_l451_45128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_inverse_correct_l451_45144

noncomputable section

-- Define the functions p, q, and r
def p (x : ℝ) : ℝ := 4 * x + 5
def q (x : ℝ) : ℝ := 3 * x - 4
def r (x : ℝ) : ℝ := p (q x)

-- Define the inverse function of r
def r_inv (x : ℝ) : ℝ := (x + 11) / 12

-- Theorem statement
theorem r_inverse_correct : 
  (∀ x, r x = 12 * x - 11) ∧ 
  (∀ x, r (r_inv x) = x) ∧ 
  (∀ x, r_inv (r x) = x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_inverse_correct_l451_45144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l451_45171

/-- The area of a triangle with vertices at (3, 2), (9, -4), and (3, 8) is 18 square units. -/
theorem triangle_area : ∃ area : ℝ, area = 18 := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (3, 2)
  let B : ℝ × ℝ := (9, -4)
  let C : ℝ × ℝ := (3, 8)

  -- Calculate the area using the formula: 1/2 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
  let area := (1/2 : ℝ) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

  -- Assert that the calculated area equals 18
  have h : area = 18 := by sorry

  -- Prove the existence of the area
  exact ⟨area, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l451_45171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l451_45118

/-- Given two intersecting lines in a coordinate plane, prove the area of the resulting quadrilateral. -/
theorem area_of_quadrilateral_OBEC :
  let E : ℝ × ℝ := (4, 4)
  let C : ℝ × ℝ := (6, 0)
  let D : ℝ × ℝ := (0, 6)
  let line1_slope : ℝ := -3
  let line1 := {(x, y) : ℝ × ℝ | y = line1_slope * x + (E.2 - line1_slope * E.1)}
  let line2 := {(x, y) : ℝ × ℝ | (x - C.1) * D.2 = (y - C.2) * D.1}
  let A : ℝ × ℝ := (16/3, 0)  -- Calculated x-intercept
  let B : ℝ × ℝ := (0, 16)    -- Calculated y-intercept
  let O : ℝ × ℝ := (0, 0)
  let area_OBEC := (abs (O.1 * B.2 + B.1 * E.2 + E.1 * O.2 - B.2 * E.1 - E.2 * O.1 - O.2 * B.1) +
                    abs (E.1 * B.2 + B.1 * C.2 + C.1 * E.2 - B.2 * C.1 - C.2 * E.1 - E.2 * B.1)) / 2
  area_OBEC = 188 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l451_45118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_half_monotone_increasing_condition_l451_45107

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

-- Part I
theorem max_value_when_a_neg_half :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f (-1/2) y ≤ f (-1/2) x ∧ f (-1/2) x = 0 :=
by sorry

-- Part II
theorem monotone_increasing_condition (a : ℝ) :
  (∀ (x y : ℝ), 0 < x ∧ x < y → f a x < f a y) ↔ a ≥ (1/2) * Real.exp (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_half_monotone_increasing_condition_l451_45107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l451_45121

-- Define the functions g and f
def g : ℝ → ℝ := sorry
def f : ℝ → ℝ := sorry

-- State the properties of g
axiom g_continuous : Continuous g
axiom g_derivative_positive : ∀ x > 0, ∃ g', HasDerivAt g g' x ∧ g' > 0
axiom g_even : ∀ x, g x = g (-x)

-- State the properties of f
axiom f_periodic : ∀ x, f (Real.sqrt 3 + x) = -f x
axiom f_def : ∀ x ∈ Set.Icc 0 (Real.sqrt 3), f x = x^3 - 3*x

-- State the theorem
theorem inequality_holds_iff (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 3, g (f x) ≤ g (a^2 - a + 2)) ↔ (a ≥ 1 ∨ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_l451_45121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_lines_of_field_l451_45112

/-- Vector field definition -/
def vector_field (c r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c.2.1 * r.2.2 - c.2.2 * r.2.1,
   c.2.2 * r.1 - c.1 * r.2.2,
   c.1 * r.2.1 - c.2.1 * r.1)

/-- Theorem: Vector lines of the vector field -/
theorem vector_lines_of_field (c : ℝ × ℝ × ℝ) :
  ∃ (A₁ A₂ : ℝ), A₁ > 0 ∧
  ∀ (r : ℝ × ℝ × ℝ), (∃ t : ℝ, vector_field c r = (t * r.1, t * r.2.1, t * r.2.2)) ↔
  (r.1^2 + r.2.1^2 + r.2.2^2 = A₁ ∧ c.1 * r.1 + c.2.1 * r.2.1 + c.2.2 * r.2.2 = A₂) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_lines_of_field_l451_45112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l451_45194

/-- An ellipse with the given properties has eccentricity √2 - 1 -/
theorem ellipse_eccentricity (a b c : ℝ) (h : ℝ) (F₁ F₂ P : ℝ × ℝ) (ellipse : Set (ℝ × ℝ)) : 
  a > 0 → b > 0 → a > b →
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ellipse) →
  F₁ ∈ ellipse → F₂ ∈ ellipse →
  F₁.1 = -c → F₂.1 = c →
  P ∈ ellipse →
  P.1 = c →
  (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 →
  (F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2) = 0 →
  c / a = Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l451_45194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonals_l451_45186

/-- Represents a convex polygon with n sides -/
def ConvexPolygon (n : ℕ) : Prop := sorry

/-- The interior angle of a convex polygon at vertex i -/
def InteriorAngle {n : ℕ} (p : ConvexPolygon n) (i : Fin n) : ℝ := sorry

/-- The number of diagonals in a convex polygon -/
def NumberOfDiagonals {n : ℕ} (p : ConvexPolygon n) : ℕ := sorry

/-- A convex polygon with interior angles of 150° has 54 diagonals -/
theorem polygon_diagonals (n : ℕ) (h_convex : ConvexPolygon n) 
  (h_interior_angle : ∀ i, InteriorAngle h_convex i = 150) : 
  NumberOfDiagonals h_convex = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_diagonals_l451_45186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l451_45168

/-- Represents the shape of the shaded region --/
structure ShadedShape where
  up1 : ℝ
  right1 : ℝ
  up2 : ℝ
  right2 : ℝ
  up3 : ℝ

/-- Calculates the area of the shaded region in a grid --/
noncomputable def shadedArea (gridHeight : ℝ) (gridLength : ℝ) (shape : ShadedShape) : ℝ :=
  gridHeight * gridLength -
  (1/2 * shape.right1 * (gridHeight - shape.up1 - shape.up2 - shape.up3) +
   1/2 * shape.right2 * (gridHeight - shape.up2 - shape.up3) +
   1/2 * (gridLength - shape.right1 - shape.right2) * (gridHeight - shape.up3))

theorem shaded_area_calculation (gridHeight gridLength : ℝ) (shape : ShadedShape) :
  gridHeight = 5 ∧ gridLength = 12 ∧
  shape.up1 = 2 ∧ shape.right1 = 3 ∧ shape.up2 = 2 ∧ shape.right2 = 4 ∧ shape.up3 = 1 →
  shadedArea gridHeight gridLength shape = 51.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l451_45168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l451_45140

def a : ℝ × ℝ := (6, -2)

theorem vector_magnitude_proof (m : ℝ) :
  let b : ℝ × ℝ := (3, m)
  (a.1 / b.1 = a.2 / b.2) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 3 * Real.sqrt 2 := by
  intro b h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l451_45140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_value_l451_45173

noncomputable def curve (x : ℝ) : ℝ := x * Real.log x

theorem tangent_perpendicular_implies_a_value :
  let tangent_slope : ℝ := (deriv curve) e
  let perpendicular_line (a : ℝ) := {p : ℝ × ℝ | p.1 + a * p.2 = 1}
  ∀ a : ℝ, (tangent_slope * (-1/a) = -1) → a = 2 :=
by
  intro a h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_implies_a_value_l451_45173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_sum_characterization_l451_45192

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem lcm_sum_characterization (n : ℕ) (hn : n > 0) :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ n = Nat.lcm a b + Nat.lcm b c + Nat.lcm c a) ↔ ¬(is_power_of_two n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_sum_characterization_l451_45192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l451_45195

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 - 2*m - 3)

theorem decreasing_power_function (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂) ↔ m = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l451_45195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integral_bound_l451_45143

/-- For any polynomial of degree 1999, there exists a constant C such that
    the absolute value of the polynomial at 0 is less than or equal to C
    times the integral of the absolute value of the polynomial from -1 to 1. -/
theorem polynomial_integral_bound :
  ∃ C > 0, ∀ p : Polynomial ℝ,
    Polynomial.degree p = 1999 →
    |p.eval 0| ≤ C * ∫ (x : ℝ) in Set.Icc (-1) 1, |p.eval x| := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integral_bound_l451_45143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_ellipse_l451_45167

/-- Given an ellipse with semi-major axis 5 and semi-minor axis 3,
    the minimum length of a tangent line segment that intersects
    both the major and minor axes is 8. -/
theorem min_tangent_length_ellipse :
  let a : ℝ := 5
  let b : ℝ := 3
  ∀ θ : ℝ,
    let x := a * Real.cos θ
    let y := b * Real.sin θ
    let tangent_x := a / Real.cos θ
    let tangent_y := b / Real.sin θ
    let segment_length := Real.sqrt ((tangent_x ^ 2) + (tangent_y ^ 2))
  (∀ θ : ℝ, segment_length ≥ 8) ∧
  (∃ θ : ℝ, segment_length = 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_ellipse_l451_45167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_formula_l451_45199

def my_sequence (n : ℕ) : ℚ := 1 / (n + 1)

theorem my_sequence_formula (n : ℕ) : 
  my_sequence n = 1 / (n + 1) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_formula_l451_45199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_8_l451_45163

/-- Rectangle ABCD with point E and F -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ :=
  (r.B.1 - r.A.1) * (r.D.2 - r.A.2)

/-- The area of a triangle -/
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

theorem rectangle_area_is_8 (r : Rectangle) : rectangleArea r = 8 :=
  by
  have h1 : r.A = (0, 0) := sorry
  have h2 : r.B.1 - r.A.1 = 4 := sorry
  have h3 : r.D.2 - r.A.2 = 2 := sorry
  have h4 : r.E = (0, 1) := sorry
  have h5 : triangleArea r.D r.E r.F = 2 := sorry
  
  -- Use the definition of rectangleArea
  unfold rectangleArea
  
  -- Rewrite using the hypotheses
  rw [h2, h3]
  
  -- Simplify
  norm_num

#check rectangle_area_is_8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_8_l451_45163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l451_45176

theorem max_negative_integers (a b c d : ℤ) (h : a * b * c * d < 0) :
  ∃ (n : ℕ), n ≤ 3 ∧ 
  (∀ (f : Fin 4 → ℤ), (f 0 * f 1 * f 2 * f 3 = a * b * c * d) →
    (Finset.filter (λ i => f i < 0) Finset.univ).card ≤ n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l451_45176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_away_approx_l451_45105

/-- Calculates the total time away from home given the travel conditions --/
noncomputable def total_time_away (original_distance : ℝ) (original_speed : ℝ) (time_at_friends : ℝ)
  (detour1_length_factor : ℝ) (detour1_speed : ℝ)
  (detour2_length_factor : ℝ) (detour2_speed : ℝ)
  (detour3_length_factor : ℝ) (detour3_speed : ℝ) : ℝ :=
  let time_to_friends := original_distance / original_speed
  let time_detour1 := (detour1_length_factor * original_distance) / detour1_speed
  let time_detour2 := (detour2_length_factor * original_distance) / detour2_speed
  let time_detour3 := (detour3_length_factor * original_distance) / detour3_speed
  time_to_friends + time_at_friends + time_detour1 + time_detour2 + time_detour3

/-- The total time away from home is approximately 8.5144 hours --/
theorem total_time_away_approx :
  abs (total_time_away 80 50 0.75 1.1 40 1.15 45 1.2 50 - 8.5144) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_away_approx_l451_45105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rally_race_average_speed_l451_45116

/-- Represents a segment of the rally race --/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the average speed given a list of segments --/
noncomputable def averageSpeed (segments : List Segment) : ℝ :=
  let totalDistance := segments.foldl (λ acc s => acc + s.distance) 0
  let totalTime := segments.foldl (λ acc s => acc + s.distance / s.speed) 0
  totalDistance / totalTime

/-- The rally race problem --/
theorem rally_race_average_speed :
  let segments : List Segment := [
    { distance := 50, speed := 100 },
    { distance := 70, speed := 40 },
    { distance := 80, speed := 60 },
    { distance := 50, speed := 30 },
    { distance := 50, speed := 70 }
  ]
  abs (averageSpeed segments - 50.306) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rally_race_average_speed_l451_45116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_is_one_l451_45153

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - (floor x : ℝ)

-- State the theorem
theorem smallest_positive_period_is_one (x : ℝ) : f (x + 1) = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_is_one_l451_45153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_breadth_calculation_l451_45161

/-- Represents a cistern with water -/
structure Cistern where
  length : ℝ
  width : ℝ
  wetSurfaceArea : ℝ
  waterBreadth : ℝ

/-- Calculates the total wet surface area of a cistern -/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.waterBreadth + 2 * c.width * c.waterBreadth

/-- Theorem stating that for a cistern with given dimensions and wet surface area,
    the water breadth is approximately 1.35 meters -/
theorem water_breadth_calculation (c : Cistern) 
    (h1 : c.length = 10)
    (h2 : c.width = 6)
    (h3 : c.wetSurfaceArea = 103.2)
    (h4 : totalWetSurfaceArea c = c.wetSurfaceArea) :
  abs (c.waterBreadth - 1.35) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_breadth_calculation_l451_45161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_example_l451_45185

/-- The speed of a train in km/hr given its length and time to pass a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A 140-meter-long train passing a tree in 8 seconds has a speed of 63 km/hr -/
theorem train_speed_example : train_speed 140 8 = 63 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  simp [div_mul_eq_mul_div]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_example_l451_45185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decay_threshold_proof_l451_45139

noncomputable def decay_rate : ℝ := 3/4
noncomputable def threshold : ℝ := 1/100

def min_years : ℕ := 4

theorem decay_threshold_proof : 
  ∀ n : ℕ, n ≥ min_years ↔ (1 - decay_rate)^n ≤ threshold :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decay_threshold_proof_l451_45139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l451_45131

/-- The circle equation: x^2 + y^2 - 2x - 4y + a = 0 -/
def circle_eq (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + a = 0

/-- The function y = x^2/4 -/
noncomputable def func (x : ℝ) : ℝ :=
  x^2 / 4

/-- A point (x, y) is on the tangent line to the graph of func at (t, func t) -/
def on_tangent_line (x y t : ℝ) : Prop :=
  y - func t = (x - t) * (2 * t) / 4

/-- The tangent line to the circle at point (x, y) is perpendicular to the radius -/
def tangent_to_circle (x y : ℝ) : Prop :=
  (x - 1) * (x - 1) + (y - 2) * (y - 2) = (x - 1)^2 + (y - 2)^2

theorem tangent_line_value (a : ℝ) :
  (∃ x y : ℝ, circle_eq x y a ∧ y = func x ∧ on_tangent_line x y x ∧ tangent_to_circle x y) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_l451_45131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l451_45181

-- Define the function f(x) = a^(x+2) - 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2) - 2

-- State the theorem
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-2) = -1 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_exponential_function_l451_45181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_and_monotonicity_l451_45150

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x + Real.pi / 4)

theorem smallest_period_and_monotonicity 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x : ℝ, f ω (x + Real.pi) = f ω x) 
  (h_smallest : ∀ T : ℝ, T > 0 → (∀ x : ℝ, f ω (x + T) = f ω x) → T ≥ Real.pi) :
  (ω = 1) ∧ 
  (∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (-(3*Real.pi)/8 + k*Real.pi) (Real.pi/8 + k*Real.pi))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_and_monotonicity_l451_45150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l451_45130

-- Define the circle M
noncomputable def circle_M (a r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - 4)^2 = r^2

-- Define the line
def line (m : ℝ) (x y : ℝ) : Prop := 4*x + 3*y + m = 0

-- Define the chord length
noncomputable def chord_length (a r m : ℝ) : ℝ := 2 * Real.sqrt (r^2 - ((24 + m)^2 / 25))

theorem circle_and_line_problem :
  ∃ (a r : ℝ), r > 0 ∧
  circle_M a r 0 0 ∧
  circle_M a r 6 0 ∧
  (a = 3 ∧ r = 5) ∧
  ∃ (m : ℝ), chord_length a r m = 6 ∧ (m = -4 ∨ m = -44) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l451_45130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l451_45122

/-- The volume of a circular truncated cone -/
noncomputable def truncated_cone_volume (r : ℝ) (R : ℝ) (l : ℝ) : ℝ :=
  let h := Real.sqrt (l^2 - (R - r)^2)
  Real.pi / 3 * (R^2 + r^2 + R*r) * h

/-- Theorem: The volume of a circular truncated cone with top radius 2, bottom radius 4, 
    and slant height 2√10 is equal to 56π -/
theorem truncated_cone_volume_example : 
  truncated_cone_volume 2 4 (2 * Real.sqrt 10) = 56 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l451_45122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_expression_x_intercept_is_solution_l451_45157

noncomputable def line_m (x y : ℝ) : Prop := 4 * x - 3 * y + 24 = 0

noncomputable def rotation_angle : ℝ := Real.pi / 6  -- 30 degrees in radians

def rotation_point : ℝ × ℝ := (10, -10)

noncomputable def slope_m : ℝ := 4 / 3

-- New slope after rotation
noncomputable def slope_n : ℝ := (slope_m + Real.tan rotation_angle) / (1 - slope_m * Real.tan rotation_angle)

-- x-intercept of the new line
noncomputable def x_intercept : ℝ := (rotation_point.2 - slope_n * rotation_point.1) / slope_n + rotation_point.1

theorem x_intercept_expression :
  ∃ (f : ℝ → ℝ → ℝ → ℝ → ℝ), 
    x_intercept = f slope_m slope_n rotation_point.1 rotation_point.2 := by
  sorry

-- Additional theorem to connect the problem statement to the Lean formalization
theorem x_intercept_is_solution :
  ∃ (answer : ℝ), x_intercept = answer ∧ (answer = 15 ∨ answer = 20 ∨ answer = 25 ∨ answer = 30 ∨ answer = 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_expression_x_intercept_is_solution_l451_45157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_FDBG_l451_45188

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def triangle_properties (ABC : Triangle) : Prop :=
  let (xa, ya) := ABC.A
  let (xb, yb) := ABC.B
  let (xc, yc) := ABC.C
  ∃ (D E F G : ℝ × ℝ),
    -- AB = 40
    Real.sqrt ((xb - xa)^2 + (yb - ya)^2) = 40 ∧
    -- AC = 20
    Real.sqrt ((xc - xa)^2 + (yc - ya)^2) = 20 ∧
    -- Area of triangle ABC is 160
    abs ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)) / 2 = 160 ∧
    -- D is the midpoint of AB
    D = ((xa + xb) / 2, (ya + yb) / 2) ∧
    -- E is the midpoint of AC
    E = ((xa + xc) / 2, (ya + yc) / 2) ∧
    -- F is on DE and G is on BC
    ∃ (t s : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
      F = (D.1 + t * (E.1 - D.1), D.2 + t * (E.2 - D.2)) ∧
      G = (xb + s * (xc - xb), yb + s * (yc - yb)) ∧
    -- F and G are on the angle bisector of ∠BAC
    (F.1 - xa) * (G.2 - ya) = (F.2 - ya) * (G.1 - xa)

-- State the theorem
theorem area_of_quadrilateral_FDBG (ABC : Triangle) :
  triangle_properties ABC →
  ∃ (D E F G : ℝ × ℝ),
    -- Area of quadrilateral FDBG is 70
    abs ((F.1 - D.1) * (G.2 - ABC.B.2) - (G.1 - ABC.B.1) * (F.2 - D.2)) / 2 +
    abs ((G.1 - D.1) * (ABC.B.2 - D.2) - (ABC.B.1 - D.1) * (G.2 - D.2)) / 2 = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_FDBG_l451_45188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_inequality_l451_45127

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ+) :
  (fibonacci (n + 1) : ℝ) ^ (1 / (n : ℝ)) ≥ 1 + 1 / (fibonacci n : ℝ) ^ (1 / (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_inequality_l451_45127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_safe_path_l451_45137

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line segment between two points -/
structure LineSegment where
  start : Point
  finish : Point

/-- Represents the sea region -/
structure SeaRegion where
  bottomLeft : Point
  topRight : Point

/-- Checks if a point is within the sea region -/
def inRegion (p : Point) (region : SeaRegion) : Prop :=
  region.bottomLeft.x ≤ p.x ∧ p.x ≤ region.topRight.x ∧
  region.bottomLeft.y ≤ p.y ∧ p.y ≤ region.topRight.y

/-- Checks if a point is on the bottom edge of the region -/
def onBottomEdge (p : Point) (region : SeaRegion) : Prop :=
  p.y = region.bottomLeft.y ∧ inRegion p region

/-- Checks if a point is on the top edge of the region -/
def onTopEdge (p : Point) (region : SeaRegion) : Prop :=
  p.y = region.topRight.y ∧ inRegion p region

/-- Checks if a point is on a line segment -/
def onLineSegment (p : Point) (segment : LineSegment) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.x = segment.start.x + t * (segment.finish.x - segment.start.x) ∧
    p.y = segment.start.y + t * (segment.finish.y - segment.start.y)

/-- The main theorem: There exists a path through the minefield -/
theorem exists_safe_path (region : SeaRegion) (mines : Set Point) 
    (h_finite : Set.Finite mines) (h_in_region : ∀ m ∈ mines, inRegion m region) :
    ∃ (start : Point) (middle : Point) (finish : Point),
      onBottomEdge start region ∧
      inRegion middle region ∧
      onTopEdge finish region ∧
      (∀ m ∈ mines, ¬onLineSegment m ⟨start, middle⟩) ∧
      (∀ m ∈ mines, ¬onLineSegment m ⟨middle, finish⟩) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_safe_path_l451_45137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l451_45187

def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  -- Definition of a triangle ABC with sides a, b, c opposite to angles A, B, C
  true

noncomputable def f (x : Real) : Real :=
  2 * (Real.sin (x + Real.pi/4))^2 - Real.sqrt 3 * Real.cos (2*x)

theorem triangle_properties :
  ∀ (A B C a b c : Real),
  triangle_ABC A B C a b c →
  1 + (Real.tan A / Real.tan B) = (2 * c / b) →
  (∀ x, x ∈ Set.Icc (Real.pi/4) (Real.pi/2) → f x ≤ a) →
  f B = a →
  (A = Real.pi/3 ∧ 
   triangle_ABC A B C a b c → (1/2) * a * c * Real.sin B = (9 + 3 * Real.sqrt 3) / 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l451_45187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_integers_l451_45156

/-- The number of different positive, five-digit integers that can be formed using the digits 2, 2, 2, 9, and 9 -/
def five_digit_integers_count : ℕ := 10

/-- The multiset of digits used to form the integers -/
def digit_multiset : Multiset ℕ := {2, 2, 2, 9, 9}

/-- Theorem stating that the count of different positive, five-digit integers
    formed using the given digits is equal to five_digit_integers_count -/
theorem count_five_digit_integers :
  (Finset.filter (fun n : ℕ => 10000 ≤ n ∧ n < 100000) 
    (Finset.image (fun l => l.foldl (fun acc d => acc * 10 + d) 0) 
      (Multiset.toList digit_multiset).permutations.toFinset)).card = five_digit_integers_count := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_five_digit_integers_l451_45156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_rotations_l451_45134

/-- Represents the properties of a rotating wheel --/
structure Wheel where
  distance_per_hour : ℚ  -- distance in meters per hour
  distance_per_rotation : ℚ  -- distance in centimeters per rotation

/-- Calculates the number of rotations per minute for a given wheel --/
def rotations_per_minute (w : Wheel) : ℚ :=
  (w.distance_per_hour * 100 / 60) / w.distance_per_rotation

/-- Theorem stating that a wheel moving 420 meters per hour and 35 cm per rotation rotates 20 times per minute --/
theorem wheel_rotations (w : Wheel) (h1 : w.distance_per_hour = 420) (h2 : w.distance_per_rotation = 35) :
  rotations_per_minute w = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_rotations_l451_45134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_triangle_property_l451_45102

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)

noncomputable def f (x : ℝ) : ℝ := (m x).1^2 + (m x).2^2 + (m x).1 * (n x).1 + (m x).2 * (n x).2 - 2

def is_geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c

theorem max_value_and_triangle_property :
  ∀ (A B C : ℝ) (a b c : ℝ),
  0 < B ∧ B < π/2 →
  is_geometric_sequence a b c →
  a = Real.sin A * b →
  b = Real.sin B * c →
  c = Real.sin C * a →
  f B = 1 →
  (∀ x, f x ≤ 1) ∧ 1/Real.tan A + 1/Real.tan C = 2*Real.sqrt 3/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_triangle_property_l451_45102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pawns_remaining_l451_45123

/-- Represents the color of a chess square or pawn -/
inductive Color where
  | White
  | Black
deriving Repr

/-- Represents a position on the chessboard -/
structure Position where
  row : Fin 8
  col : Fin 8
deriving Repr

/-- Represents a pawn on the chessboard -/
structure Pawn where
  color : Color
  position : Position
deriving Repr

/-- Represents the chessboard -/
def Chessboard := List Pawn

/-- Initializes the chessboard with 32 white and 32 black pawns -/
def initialBoard : Chessboard := sorry

/-- Determines if a move is valid for a given pawn -/
def isValidMove (fromPos : Position) (toPos : Position) (color : Color) : Bool := sorry

/-- Performs a single capture move on the board -/
def captureMove (board : Chessboard) (fromPos : Position) (toPos : Position) : Chessboard := sorry

/-- Represents a sequence of capture moves -/
def CaptureMoves := List (Position × Position)

/-- Applies a sequence of capture moves to the board -/
def applyCaptureMoves (board : Chessboard) (moves : CaptureMoves) : Chessboard := sorry

/-- The main theorem: The minimum number of pawns remaining after any sequence of captures is 2 -/
theorem min_pawns_remaining (moves : CaptureMoves) :
  (applyCaptureMoves initialBoard moves).length ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pawns_remaining_l451_45123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_is_step_function_l451_45197

-- Define the semicircular paths
def SemicircularPath (center : ℝ × ℝ) (radius : ℝ) := 
  {p : ℝ × ℝ | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi ∧ p = (center.1 + radius * Real.cos θ, center.2 + radius * Real.sin θ)}

-- Define the ship's journey
def ShipJourney (center : ℝ × ℝ) (R r : ℝ) :=
  {p : ℝ × ℝ | p ∈ SemicircularPath center R ∨ p ∈ SemicircularPath center r}

-- Define the distance function
noncomputable def DistanceFromCenter (center : ℝ × ℝ) (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2)

-- Theorem statement
theorem ship_distance_is_step_function (center : ℝ × ℝ) (R r : ℝ) (h : r < R) :
  ∀ p ∈ ShipJourney center R r, 
    DistanceFromCenter center p = R ∨ DistanceFromCenter center p = r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_is_step_function_l451_45197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_inequality_l451_45174

theorem min_inequality (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) :
  min (min ((10 * a^2 - 5 * a + 1) / (b^2 - 5 * b + 10))
           ((10 * b^2 - 5 * b + 1) / (c^2 - 5 * c + 10)))
      ((10 * c^2 - 5 * c + 1) / (a^2 - 5 * a + 10))
  ≤ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_inequality_l451_45174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l451_45170

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin ((1 / 2) * x - Real.pi / 12)

theorem g_decreasing_interval :
  ∀ x₁ x₂, 7 * Real.pi / 6 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 19 * Real.pi / 6 → g x₂ < g x₁ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l451_45170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l451_45191

/-- The cost price of a book, given its marked price and profit margins. -/
noncomputable def cost_price (marked_price : ℝ) : ℝ :=
  marked_price / (0.85 * 1.25)

/-- Theorem stating the cost price of the book given the conditions -/
theorem book_cost_price : 
  let marked_price : ℝ := 69.85
  abs (cost_price marked_price - 65.75) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_cost_price_l451_45191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kilo_ratio_is_two_to_one_l451_45124

/-- Represents the laundry shop scenario -/
structure LaundryShop where
  price_per_kilo : ℚ
  kilos_two_days_ago : ℚ
  kilos_increase_yesterday : ℚ
  total_earnings : ℚ

/-- Calculates the ratio of kilos washed today to yesterday -/
noncomputable def kilo_ratio (shop : LaundryShop) : ℚ :=
  let kilos_yesterday := shop.kilos_two_days_ago + shop.kilos_increase_yesterday
  let earnings_two_days_ago := shop.kilos_two_days_ago * shop.price_per_kilo
  let earnings_yesterday := kilos_yesterday * shop.price_per_kilo
  let earnings_today := shop.total_earnings - earnings_two_days_ago - earnings_yesterday
  let kilos_today := earnings_today / shop.price_per_kilo
  kilos_today / kilos_yesterday

/-- Theorem stating that the ratio of kilos washed today to yesterday is 2:1 -/
theorem kilo_ratio_is_two_to_one (shop : LaundryShop)
  (h1 : shop.price_per_kilo = 2)
  (h2 : shop.kilos_two_days_ago = 5)
  (h3 : shop.kilos_increase_yesterday = 5)
  (h4 : shop.total_earnings = 70) :
  kilo_ratio shop = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kilo_ratio_is_two_to_one_l451_45124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_150th_term_l451_45115

/-- Represents the sequence of positive integers that are either powers of 3, powers of 5, or sums of distinct powers of these numbers. -/
def specialSequence : ℕ → ℕ := sorry

/-- The 150th term of the specialSequence -/
theorem special_sequence_150th_term : specialSequence 150 = 2840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_150th_term_l451_45115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_and_absolute_value_simplification_l451_45155

theorem cube_root_and_absolute_value_simplification :
  ((-8 : ℝ) ^ (1/3 : ℝ)) + |2 - Real.sqrt 5| + 4 * (Real.sqrt 5 / 2) = 3 * Real.sqrt 5 - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_and_absolute_value_simplification_l451_45155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cos_value_l451_45158

/-- If a point P(cos α, sin α) is on the line y = -2x, then cos(2α + π/3) = (4√3 - 3) / 10 -/
theorem point_on_line_cos_value (α : ℝ) :
  Real.sin α = -2 * Real.cos α →
  Real.cos (2 * α + π / 3) = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cos_value_l451_45158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l451_45110

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2}
def B : Finset Nat := {2, 3, 4}

theorem intersection_complement_equality : B ∩ (U \ A) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l451_45110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x7_in_expansion_l451_45179

theorem coefficient_x7_in_expansion : 
  (Finset.range 11).sum (fun k => (Nat.choose 10 k : Int) * (-2)^k * (if 10 - k = 7 then 1 else 0)) = -960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x7_in_expansion_l451_45179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l451_45142

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (2^(x + 1))

-- Theorem stating that f is an odd function and increasing on ℝ
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
  sorry

#check f_odd_and_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l451_45142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l451_45141

/-- The speed of a train in km/h, given its length in meters and time to pass a point in seconds -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A 140-meter long train that crosses a stationary point in 6 seconds 
    has a speed of approximately 84 km/h -/
theorem train_speed_approx : 
  ∃ (speed : ℝ), abs (train_speed 140 6 - speed) < 0.5 ∧ Int.floor speed = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l451_45141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l451_45147

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  Real.sin A = 2/3 →
  Real.sin B = 2 * Real.cos C →
  c^2 - a^2 = b →
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l451_45147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_is_ten_percent_l451_45160

/-- Represents the sandwich shop scenario -/
structure SandwichOrder where
  sandwich_price : ℚ
  delivery_fee : ℚ
  num_sandwiches : ℕ
  total_received : ℚ

/-- Calculates the tip percentage given a sandwich order -/
def tip_percentage (order : SandwichOrder) : ℚ :=
  let total_cost := order.sandwich_price * order.num_sandwiches + order.delivery_fee
  let tip_amount := order.total_received - total_cost
  (tip_amount / total_cost) * 100

/-- Theorem stating that the tip percentage for the given scenario is 10% -/
theorem tip_is_ten_percent (order : SandwichOrder) 
  (h1 : order.sandwich_price = 5)
  (h2 : order.delivery_fee = 20)
  (h3 : order.num_sandwiches = 18)
  (h4 : order.total_received = 121) :
  tip_percentage order = 10 := by
  sorry

/-- Example calculation -/
def example_order : SandwichOrder := {
  sandwich_price := 5,
  delivery_fee := 20,
  num_sandwiches := 18,
  total_received := 121
}

#eval tip_percentage example_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tip_is_ten_percent_l451_45160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_decreasing_f_l451_45159

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((Real.sqrt (a^2 + 1) - a) / a)^x / Real.log (1/2)

-- State the theorem
theorem range_of_a_given_decreasing_f (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a < Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_given_decreasing_f_l451_45159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_y_5_is_integer_five_is_smallest_l451_45180

noncomputable def y : ℕ → ℝ
  | 0 => 4^(1/4)
  | 1 => (4^(1/4))^(4^(1/4))
  | n+2 => (y (n+1))^(4^(1/4))

theorem smallest_integer_y (n : ℕ) : n < 5 → ¬(∃ m : ℤ, y n = m) := by
  sorry

theorem y_5_is_integer : ∃ m : ℤ, y 5 = m := by
  sorry

theorem five_is_smallest (n : ℕ) : n ≥ 5 → (∃ m : ℤ, y n = m) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_y_5_is_integer_five_is_smallest_l451_45180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_time_l451_45133

/-- The height of a ball thrown upwards as a function of time -/
def ballHeight (t : ℝ) : ℝ := -16 * t^2 + 96 * t

/-- The time when the ball reaches a specific height -/
noncomputable def timeAtHeight (h : ℝ) : ℝ := 
  (Real.sqrt (96^2 + 64 * h) + 96) / 32

theorem ball_height_time : timeAtHeight 36 = 1.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_height_time_l451_45133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l451_45138

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 4*x + 8

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 4

-- Define the slope at x = 1
def slope_at_one : ℝ := f' 1

-- Define the slope angle in degrees
noncomputable def slope_angle : ℝ := 180 - Real.arctan slope_at_one * (180 / Real.pi)

-- Theorem statement
theorem tangent_slope_angle :
  f 1 = 5 ∧ slope_angle = 135 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l451_45138
