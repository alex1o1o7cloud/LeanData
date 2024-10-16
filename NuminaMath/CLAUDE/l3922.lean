import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_inequality_l3922_392211

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) : 
  Real.sqrt a - Real.sqrt (a - 1) < Real.sqrt (a - 2) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3922_392211


namespace NUMINAMATH_CALUDE_polynomial_sum_l3922_392249

theorem polynomial_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 5*x^2 + 8*x - 12) → 
  a + b + c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3922_392249


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3922_392299

/-- A quadratic function f(x) = ax^2 + bx + c where a > 0 and f(1) = 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem sufficient_not_necessary_condition
  (a b c : ℝ)
  (h_a_pos : a > 0)
  (h_f_1_eq_0 : QuadraticFunction a b c 1 = 0) :
  (∀ a b c, b > 2 * a → QuadraticFunction a b c (-2) < 0) ∧
  (∃ a b c, QuadraticFunction a b c (-2) < 0 ∧ b ≤ 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3922_392299


namespace NUMINAMATH_CALUDE_grocer_coffee_solution_l3922_392265

/-- Represents the grocer's coffee inventory --/
structure CoffeeInventory where
  initial : ℝ
  decafRatio : ℝ
  newPurchase : ℝ
  newDecafRatio : ℝ
  finalDecafRatio : ℝ

/-- The grocer's coffee inventory problem --/
def grocerProblem : CoffeeInventory where
  initial := 400  -- This is what we want to prove
  decafRatio := 0.2
  newPurchase := 100
  newDecafRatio := 0.5
  finalDecafRatio := 0.26

/-- Theorem stating the solution to the grocer's coffee inventory problem --/
theorem grocer_coffee_solution (inv : CoffeeInventory) : 
  inv.initial = 400 ∧ 
  inv.decafRatio = 0.2 ∧ 
  inv.newPurchase = 100 ∧ 
  inv.newDecafRatio = 0.5 ∧ 
  inv.finalDecafRatio = 0.26 →
  inv.finalDecafRatio * (inv.initial + inv.newPurchase) = 
    inv.decafRatio * inv.initial + inv.newDecafRatio * inv.newPurchase := by
  sorry

#check grocer_coffee_solution grocerProblem

end NUMINAMATH_CALUDE_grocer_coffee_solution_l3922_392265


namespace NUMINAMATH_CALUDE_second_quadrant_complex_l3922_392279

theorem second_quadrant_complex (a b : ℝ) : 
  (Complex.ofReal a + Complex.I * Complex.ofReal b).im > 0 ∧ 
  (Complex.ofReal a + Complex.I * Complex.ofReal b).re < 0 → 
  a < 0 ∧ b > 0 := by sorry

end NUMINAMATH_CALUDE_second_quadrant_complex_l3922_392279


namespace NUMINAMATH_CALUDE_distribute_students_count_l3922_392214

/-- The number of ways to distribute four students into three classes -/
def distribute_students : ℕ :=
  let total_distributions := (4 : ℕ).choose 2 * (3 : ℕ).factorial
  let invalid_distributions := (3 : ℕ).factorial
  total_distributions - invalid_distributions

/-- Theorem stating that the number of valid distributions is 30 -/
theorem distribute_students_count : distribute_students = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribute_students_count_l3922_392214


namespace NUMINAMATH_CALUDE_two_equal_intercept_lines_l3922_392258

/-- A line with equal intercepts passing through (1, 2) -/
structure EqualInterceptLine where
  /-- The equation of the line in the form x + y = a -/
  a : ℝ
  /-- The line passes through (1, 2) -/
  passes_through_point : a = 3
  /-- The line has equal intercepts on x and y axes -/
  equal_intercepts : True

/-- The number of lines with equal intercepts passing through (1, 2) -/
def count_equal_intercept_lines : Nat :=
  2

/-- Theorem stating that there are exactly two lines with equal intercepts passing through (1, 2) -/
theorem two_equal_intercept_lines :
  (∃ (l₁ l₂ : EqualInterceptLine), l₁ ≠ l₂) ∧
  (∀ (l₁ l₂ l₃ : EqualInterceptLine), l₁ = l₂ ∨ l₁ = l₃ ∨ l₂ = l₃) :=
sorry

end NUMINAMATH_CALUDE_two_equal_intercept_lines_l3922_392258


namespace NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_ge_5_l3922_392207

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_number_divisible_by_four_primes_ge_5 :
  ∃ (n : Nat) (p₁ p₂ p₃ p₄ : Nat),
    n > 0 ∧
    is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
    p₁ ≥ 5 ∧ p₂ ≥ 5 ∧ p₃ ≥ 5 ∧ p₄ ≥ 5 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧
    (∀ m : Nat, m > 0 ∧ m < n →
      ¬(∃ (q₁ q₂ q₃ q₄ : Nat),
        is_prime q₁ ∧ is_prime q₂ ∧ is_prime q₃ ∧ is_prime q₄ ∧
        q₁ ≥ 5 ∧ q₂ ≥ 5 ∧ q₃ ≥ 5 ∧ q₄ ≥ 5 ∧
        q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
        m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
    n = 5005 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_four_primes_ge_5_l3922_392207


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l3922_392277

/-- The solution set of x^2 - ax - x < 0 -/
def M (a : ℝ) : Set ℝ :=
  {x | x^2 - a*x - x < 0}

/-- The solution set of x^2 - 2x - 3 ≤ 0 -/
def N : Set ℝ :=
  {x | x^2 - 2*x - 3 ≤ 0}

/-- The theorem stating the range of a for which M(a) ⊆ N -/
theorem range_of_a_for_subset : 
  {a : ℝ | M a ⊆ N} = {a : ℝ | -2 ≤ a ∧ a ≤ 2} := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l3922_392277


namespace NUMINAMATH_CALUDE_system_solution_l3922_392246

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 6 * y = -2) ∧ 
    (5 * x + 3 * y = 13/2) ∧ 
    (x = 7/22) ∧ 
    (y = 6/11) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3922_392246


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3922_392230

theorem election_winner_percentage (total_votes : ℕ) (vote_majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 400 →
  vote_majority = 160 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = vote_majority :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3922_392230


namespace NUMINAMATH_CALUDE_handshake_count_l3922_392274

/-- Represents the number of students in the class -/
def num_students : ℕ := 40

/-- Represents the length of the counting sequence -/
def sequence_length : ℕ := 4

/-- Calculates the number of initial pairs facing each other -/
def initial_pairs : ℕ := num_students / sequence_length

/-- Calculates the sum of handshakes in subsequent rounds -/
def subsequent_handshakes : ℕ := (initial_pairs * (initial_pairs + 1)) / 2

/-- Calculates the total number of handshakes -/
def total_handshakes : ℕ := initial_pairs + 3 * subsequent_handshakes

/-- Theorem stating that the total number of handshakes is 175 -/
theorem handshake_count : total_handshakes = 175 := by sorry

end NUMINAMATH_CALUDE_handshake_count_l3922_392274


namespace NUMINAMATH_CALUDE_new_continental_math_institute_enrollment_l3922_392216

theorem new_continental_math_institute_enrollment :
  ∃! n : ℕ, n < 550 ∧ n % 19 = 15 ∧ n % 17 = 10 ∧ n = 509 := by
  sorry

end NUMINAMATH_CALUDE_new_continental_math_institute_enrollment_l3922_392216


namespace NUMINAMATH_CALUDE_min_chords_for_complete_circuit_l3922_392271

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of chords needed
    to return to the starting point is 3. -/
theorem min_chords_for_complete_circuit (angle_between_chords : ℝ) : 
  angle_between_chords = 60 → (∃ n : ℕ, n * (180 - angle_between_chords) = 360 ∧ ∀ m : ℕ, m * (180 - angle_between_chords) = 360 → n ≤ m) → 
  (∃ n : ℕ, n * (180 - angle_between_chords) = 360 ∧ ∀ m : ℕ, m * (180 - angle_between_chords) = 360 → n ≤ m) ∧ 
  (∀ n : ℕ, n * (180 - angle_between_chords) = 360 → n ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_chords_for_complete_circuit_l3922_392271


namespace NUMINAMATH_CALUDE_number_puzzle_l3922_392205

/-- If a number x, when divided by 9 and then subtracted by 100, results in 10, then x must be equal to 990. -/
theorem number_puzzle (x : ℚ) : (x / 9) - 100 = 10 → x = 990 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3922_392205


namespace NUMINAMATH_CALUDE_decimal_computation_l3922_392286

theorem decimal_computation : (0.25 / 0.005) * 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_computation_l3922_392286


namespace NUMINAMATH_CALUDE_product_with_miscopied_digit_l3922_392220

theorem product_with_miscopied_digit (x y : ℕ) 
  (h1 : x * y = 4500)
  (h2 : x * (y - 2) = 4380) :
  x = 60 ∧ y = 75 := by
sorry

end NUMINAMATH_CALUDE_product_with_miscopied_digit_l3922_392220


namespace NUMINAMATH_CALUDE_line_plane_intersection_l3922_392226

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (intersect : Plane → Plane → Line)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersects : Line → Line → Prop)

-- State the theorem
theorem line_plane_intersection 
  (m n : Line) (α β : Plane) :
  intersect α β = m → subset n α → 
  (parallel m n) ∨ (intersects m n) := by
  sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l3922_392226


namespace NUMINAMATH_CALUDE_speech_competition_probability_l3922_392285

theorem speech_competition_probability 
  (m n : ℕ) 
  (prob_at_least_one_female : ℝ) 
  (h1 : prob_at_least_one_female = 4/5) :
  1 - prob_at_least_one_female = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_probability_l3922_392285


namespace NUMINAMATH_CALUDE_hemisphere_with_cylinder_surface_area_l3922_392233

/-- The total surface area of a hemisphere with a cylindrical protrusion -/
theorem hemisphere_with_cylinder_surface_area (r : ℝ) (h : r > 0) :
  let base_area := π * r^2
  let hemisphere_surface := 2 * π * r^2
  let cylinder_surface := 2 * π * r^2
  base_area + hemisphere_surface + cylinder_surface = 5 * π * r^2 := by
sorry

end NUMINAMATH_CALUDE_hemisphere_with_cylinder_surface_area_l3922_392233


namespace NUMINAMATH_CALUDE_parabola_a_range_l3922_392287

/-- Parabola defined by y = ax^2 - 2a^2x + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a : a ≠ 0
  h_c : c > 0

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = p.a * x^2 - 2 * p.a^2 * x + p.c

theorem parabola_a_range (p : Parabola) 
  (point1 : PointOnParabola p) 
  (point2 : PointOnParabola p)
  (h_x1 : point1.x = 2 * p.a + 1)
  (h_x2 : 2 ≤ point2.x ∧ point2.x ≤ 4)
  (h_y : point1.y > p.c ∧ p.c > point2.y) :
  p.a > 2 ∨ p.a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_a_range_l3922_392287


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3922_392234

/-- Given a function f: ℝ → ℝ with a tangent line y = 1/2 * x + 2 at x = 1,
    prove that f(1) + f'(1) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 1 + (deriv f 1) * (x - 1) = 1/2 * x + 2) : 
    f 1 + deriv f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3922_392234


namespace NUMINAMATH_CALUDE_max_stamps_is_125_l3922_392278

/-- The maximum number of stamps that can be purchased with a given budget. -/
def max_stamps (budget : ℕ) (price_low : ℕ) (price_high : ℕ) (threshold : ℕ) : ℕ :=
  max (min (budget / price_high) threshold) (budget / price_low)

/-- Proof that 125 stamps is the maximum number that can be purchased with 5000 cents,
    given the pricing conditions. -/
theorem max_stamps_is_125 :
  max_stamps 5000 40 45 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_is_125_l3922_392278


namespace NUMINAMATH_CALUDE_chris_age_l3922_392264

theorem chris_age (a b c : ℕ) : 
  (a + b + c) / 3 = 12 →
  b - 5 = 2 * (c + 2) →
  b + 3 = a + 3 →
  c = 4 :=
by sorry

end NUMINAMATH_CALUDE_chris_age_l3922_392264


namespace NUMINAMATH_CALUDE_sum_of_three_hexagons_l3922_392208

theorem sum_of_three_hexagons :
  ∀ (square hexagon : ℚ),
  (3 * square + 2 * hexagon = 18) →
  (2 * square + 3 * hexagon = 20) →
  (3 * hexagon = 72 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_hexagons_l3922_392208


namespace NUMINAMATH_CALUDE_odd_and_even_implies_zero_range_even_function_abs_property_l3922_392229

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem 1: If a function is both odd and even, its range is {0}
theorem odd_and_even_implies_zero_range (f : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven f) : 
  ∀ x, f x = 0 := by sorry

-- Theorem 2: If a function is even, then f(|x|) = f(x)
theorem even_function_abs_property (f : ℝ → ℝ) 
  (h_even : IsEven f) : 
  ∀ x, f (|x|) = f x := by sorry

end NUMINAMATH_CALUDE_odd_and_even_implies_zero_range_even_function_abs_property_l3922_392229


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3922_392221

theorem complex_fraction_simplification :
  (5 : ℂ) / (Complex.I - 2) = -2 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3922_392221


namespace NUMINAMATH_CALUDE_intersection_A_B_l3922_392224

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3922_392224


namespace NUMINAMATH_CALUDE_equation_solution_l3922_392243

theorem equation_solution :
  ∃ x : ℚ, x + 3*x = 500 - (4*x + 5*x) → x = 500/13 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3922_392243


namespace NUMINAMATH_CALUDE_distribute_5_4_l3922_392223

/-- The number of ways to distribute n distinct items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 4 identical bags, allowing empty bags, is 36. -/
theorem distribute_5_4 : distribute 5 4 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_5_4_l3922_392223


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_attained_l3922_392204

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  (9 / x + 16 / y + 25 / z) ≥ 24 := by
  sorry

theorem min_value_attained (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 6) : 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 6 ∧ 
  (9 / x₀ + 16 / y₀ + 25 / z₀) = 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_attained_l3922_392204


namespace NUMINAMATH_CALUDE_three_zeros_condition_l3922_392253

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Theorem: For f(x) to have exactly 3 zeros, 'a' must be in the range (-∞, -3) -/
theorem three_zeros_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l3922_392253


namespace NUMINAMATH_CALUDE_students_per_group_l3922_392227

theorem students_per_group 
  (total_students : ℕ) 
  (students_not_picked : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 36) 
  (h2 : students_not_picked = 9) 
  (h3 : num_groups = 3) : 
  (total_students - students_not_picked) / num_groups = 9 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l3922_392227


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l3922_392296

/-- Given sets A and B, prove that their intersection is empty if and only if a is in the specified range -/
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | x < -1 ∨ x > 5}
  (A ∩ B = ∅) ↔ (a > 3 ∨ (-1/2 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l3922_392296


namespace NUMINAMATH_CALUDE_profit_margin_increase_l3922_392283

theorem profit_margin_increase (P S : ℝ) (r : ℝ) : 
  P > 0 → S > P →
  (S - P) / P * 100 = r →
  (S - 0.92 * P) / (0.92 * P) * 100 = r + 10 →
  r = 15 := by
sorry

end NUMINAMATH_CALUDE_profit_margin_increase_l3922_392283


namespace NUMINAMATH_CALUDE_correspondence_proof_l3922_392284

/-- Given sets A and B, and a mapping f from A to B defined as
    f(x, y) = (x + 2y, 2x - y), prove that (1, 1) in A
    corresponds to (3, 1) in B under this mapping. -/
theorem correspondence_proof (A B : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ × ℝ)
    (hf : ∀ (x y : ℝ), f (x, y) = (x + 2*y, 2*x - y))
    (hA : (1, 1) ∈ A) (hB : (3, 1) ∈ B) :
    f (1, 1) = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_correspondence_proof_l3922_392284


namespace NUMINAMATH_CALUDE_ellipse_C_and_point_T_l3922_392203

/-- The ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The circle M -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 4 = 0

/-- The line l passing through (1,0) and intersecting C at A and B -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - 1)

/-- The angle OTA equals OTB -/
def angle_OTA_eq_OTB (t : ℝ) (xA yA xB yB : ℝ) : Prop :=
  (yA / (xA - t)) + (yB / (xB - t)) = 0

theorem ellipse_C_and_point_T :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ b = 1 ∧
  (∃ (c : ℝ), c^2 = a^2 - b^2 ∧
    (∀ x y : ℝ, x + c*y - c = 0 → circle_M x y)) →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2/4 + y^2 = 1) ∧
  (∃ t : ℝ, t = 4 ∧
    ∀ m xA yA xB yB : ℝ,
      line_l m xA yA ∧ line_l m xB yB ∧
      ellipse_C xA yA a b ∧ ellipse_C xB yB a b →
      angle_OTA_eq_OTB t xA yA xB yB) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_C_and_point_T_l3922_392203


namespace NUMINAMATH_CALUDE_max_value_expression_l3922_392292

theorem max_value_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) * (x - y + z) ≤ 2187/216 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3922_392292


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_15_16_l3922_392297

theorem smallest_divisible_by_12_15_16 : 
  ∃ n : ℕ+, (∀ m : ℕ+, 12 ∣ m ∧ 15 ∣ m ∧ 16 ∣ m → n ≤ m) ∧ 12 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n :=
by
  use 240
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_15_16_l3922_392297


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3922_392245

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3922_392245


namespace NUMINAMATH_CALUDE_paco_cookies_l3922_392250

theorem paco_cookies (initial_cookies : ℕ) (eaten_cookies : ℕ) (given_cookies : ℕ) 
  (h1 : initial_cookies = 17)
  (h2 : eaten_cookies = 14)
  (h3 : eaten_cookies + given_cookies ≤ initial_cookies) :
  given_cookies = 3 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l3922_392250


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l3922_392282

theorem complex_magnitude_equality (t : ℝ) (h : t > 0) :
  Complex.abs (-4 + 2 * t * Complex.I) = 3 * Real.sqrt 5 ↔ t = Real.sqrt 29 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l3922_392282


namespace NUMINAMATH_CALUDE_xyz_equals_six_l3922_392268

theorem xyz_equals_six (a b c x y z : ℂ) 
  (nonzero_a : a ≠ 0) (nonzero_b : b ≠ 0) (nonzero_c : c ≠ 0) 
  (nonzero_x : x ≠ 0) (nonzero_y : y ≠ 0) (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 3))
  (eq_b : b = (a + c) / (y - 3))
  (eq_c : c = (a + b) / (z - 3))
  (sum_prod : x * y + x * z + y * z = 10)
  (sum : x + y + z = 6) : 
  x * y * z = 6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_equals_six_l3922_392268


namespace NUMINAMATH_CALUDE_sketch_finalization_orders_l3922_392210

/-- Represents the order of sketches in the stack -/
def sketchOrder : List Nat := [2, 4, 1, 3, 5, 7, 6, 10, 9, 8]

/-- Represents the sketches completed before lunch -/
def completedSketches : List Nat := [8, 4]

/-- Calculates the number of possible orders for finalizing remaining sketches -/
def possibleOrders (order : List Nat) (completed : List Nat) : Nat :=
  sorry

theorem sketch_finalization_orders :
  possibleOrders sketchOrder completedSketches = 64 := by
  sorry

end NUMINAMATH_CALUDE_sketch_finalization_orders_l3922_392210


namespace NUMINAMATH_CALUDE_company_survey_l3922_392261

/-- The number of employees who do not use social networks -/
def non_users : ℕ := 40

/-- The fraction of social network users who use VKontakte -/
def vk_users : ℚ := 3/4

/-- The fraction of social network users who use both VKontakte and Odnoklassniki -/
def both_users : ℚ := 13/20

/-- The fraction of total employees who use Odnoklassniki -/
def ok_users : ℚ := 5/6

/-- The total number of employees in the company -/
def total_employees : ℕ := 540

theorem company_survey :
  ∃ (N : ℕ),
    N = total_employees ∧
    (N - non_users : ℚ) * (vk_users + (1 - vk_users)) = N * ok_users :=
by sorry

end NUMINAMATH_CALUDE_company_survey_l3922_392261


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3922_392218

-- Define the triangle
structure RightTriangle where
  a : ℝ  -- Length of one leg
  b : ℝ  -- Length of the other leg
  c : ℝ  -- Length of the hypotenuse
  right_angled : a^2 + b^2 = c^2  -- Pythagorean theorem
  sum_of_squares : a^2 + b^2 + c^2 = 2450
  hypotenuse_relation : c = b + 10

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) : t.c = 35 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3922_392218


namespace NUMINAMATH_CALUDE_our_parabola_properties_l3922_392248

/-- A parabola with specific properties -/
structure Parabola where
  -- The equation of the parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  c_pos : c > 0
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1

/-- The specific parabola we are interested in -/
def our_parabola : Parabola where
  a := 0
  b := 0
  c := 1
  d := -8
  e := -8
  f := 16
  c_pos := by sorry
  gcd_one := by sorry

/-- Theorem stating that our_parabola satisfies all the required properties -/
theorem our_parabola_properties :
  -- Passes through (2,8)
  (2 : ℝ)^2 * our_parabola.a + 2 * 8 * our_parabola.b + 8^2 * our_parabola.c + 2 * our_parabola.d + 8 * our_parabola.e + our_parabola.f = 0 ∧
  -- Vertex lies on y-axis (x-coordinate of vertex is 0)
  our_parabola.b^2 - 4 * our_parabola.a * our_parabola.c = 0 ∧
  -- y-coordinate of focus is 4
  (our_parabola.e^2 - 4 * our_parabola.c * our_parabola.f) / (4 * our_parabola.c^2) = 4 ∧
  -- Axis of symmetry is parallel to x-axis
  our_parabola.b = 0 ∧ our_parabola.a = 0 := by
  sorry

end NUMINAMATH_CALUDE_our_parabola_properties_l3922_392248


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l3922_392225

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- State the theorem
theorem intersection_nonempty_implies_a_geq_neg_one (a : ℝ) :
  (A ∩ B a).Nonempty → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_one_l3922_392225


namespace NUMINAMATH_CALUDE_teachers_liking_beverages_l3922_392259

theorem teachers_liking_beverages 
  (total : ℕ) 
  (tea : ℕ) 
  (coffee : ℕ) 
  (h1 : total = 90)
  (h2 : tea = 66)
  (h3 : coffee = 42)
  (h4 : ∃ (both neither : ℕ), both = 3 * neither ∧ tea + coffee - both + neither = total) :
  ∃ (at_least_one : ℕ), at_least_one = 81 ∧ at_least_one = tea + coffee - (tea + coffee - total + (total - tea - coffee) / 2) :=
by sorry

end NUMINAMATH_CALUDE_teachers_liking_beverages_l3922_392259


namespace NUMINAMATH_CALUDE_divisor_problem_l3922_392202

theorem divisor_problem (number : ℕ) (divisor : ℕ) : 
  number = 36 →
  ((number + 10) * 2 / divisor) - 2 = 88 / 2 →
  divisor = 2 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l3922_392202


namespace NUMINAMATH_CALUDE_quadratic_minimum_unique_minimum_l3922_392232

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem quadratic_minimum (x : ℝ) : f x ≥ f 7 := by
  sorry

theorem unique_minimum : ∀ x : ℝ, x ≠ 7 → f x > f 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_unique_minimum_l3922_392232


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_equals_three_l3922_392217

theorem quadratic_root_implies_u_equals_three (u : ℝ) : 
  (6 * ((-19 + Real.sqrt 289) / 12)^2 + 19 * ((-19 + Real.sqrt 289) / 12) + u = 0) → u = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_equals_three_l3922_392217


namespace NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l3922_392273

def admission_fee (adults children : ℕ) : ℕ := 30 * adults + 15 * children

def is_valid_combination (adults children : ℕ) : Prop :=
  adults ≥ 2 ∧ children ≥ 2 ∧ admission_fee adults children = 2250

def ratio_difference (adults children : ℕ) : ℚ :=
  |((adults : ℚ) / (children : ℚ)) - 1|

theorem closest_ratio_is_one_to_one :
  ∃ (a c : ℕ), is_valid_combination a c ∧
    ∀ (x y : ℕ), is_valid_combination x y →
      ratio_difference a c ≤ ratio_difference x y :=
sorry

end NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l3922_392273


namespace NUMINAMATH_CALUDE_single_burger_cost_is_one_l3922_392281

/-- Calculates the cost of a single burger given the total spent, total number of hamburgers,
    number of double burgers, and cost of a double burger. -/
def single_burger_cost (total_spent : ℚ) (total_burgers : ℕ) (double_burgers : ℕ) (double_cost : ℚ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let double_total := double_burgers * double_cost
  let single_total := total_spent - double_total
  single_total / single_burgers

/-- Proves that the cost of a single burger is $1.00 given the specified conditions. -/
theorem single_burger_cost_is_one :
  single_burger_cost 70.50 50 41 1.50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_single_burger_cost_is_one_l3922_392281


namespace NUMINAMATH_CALUDE_complex_number_properties_l3922_392288

open Complex

theorem complex_number_properties (z : ℂ) (h : I * (z + 1) = -2 + 2*I) : 
  z.im = 2 ∧ abs (z / (1 - 2*I)) ^ 2015 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3922_392288


namespace NUMINAMATH_CALUDE_units_digit_of_large_power_l3922_392254

/-- The units' digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The result of 3824^428 -/
def large_power : ℕ := 3824^428

theorem units_digit_of_large_power :
  units_digit large_power = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_large_power_l3922_392254


namespace NUMINAMATH_CALUDE_hugo_mountain_elevation_l3922_392228

/-- The elevation of Hugo's mountain in feet -/
def hugo_elevation : ℝ := 10000

/-- The elevation of Boris' mountain in feet -/
def boris_elevation : ℝ := hugo_elevation - 2500

theorem hugo_mountain_elevation :
  (3 * hugo_elevation = 4 * boris_elevation) ∧
  (boris_elevation = hugo_elevation - 2500) →
  hugo_elevation = 10000 := by
  sorry

end NUMINAMATH_CALUDE_hugo_mountain_elevation_l3922_392228


namespace NUMINAMATH_CALUDE_a_51_equals_101_l3922_392275

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem a_51_equals_101 (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 51 = 101 := by
  sorry

end NUMINAMATH_CALUDE_a_51_equals_101_l3922_392275


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3922_392267

theorem coefficient_x4_in_expansion : 
  (Finset.range 8).sum (fun k => Nat.choose 7 k * (1^(7-k) * x^(2*k))) = 
  21 * x^4 + (Finset.range 8).sum (fun k => if k ≠ 2 then Nat.choose 7 k * (1^(7-k) * x^(2*k)) else 0) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l3922_392267


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l3922_392280

theorem sally_pokemon_cards (initial cards_from_dan cards_bought cards_traded cards_lost : ℕ) 
  (h1 : initial = 27)
  (h2 : cards_from_dan = 41)
  (h3 : cards_bought = 20)
  (h4 : cards_traded = 15)
  (h5 : cards_lost = 7) :
  initial + cards_from_dan + cards_bought - cards_traded - cards_lost = 66 := by
  sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l3922_392280


namespace NUMINAMATH_CALUDE_third_dog_food_consumption_l3922_392219

/-- Given information about three dogs' food consumption, prove the amount eaten by the third dog -/
theorem third_dog_food_consumption
  (total_dogs : ℕ)
  (average_consumption : ℝ)
  (first_dog_consumption : ℝ)
  (h_total_dogs : total_dogs = 3)
  (h_average : average_consumption = 15)
  (h_first_dog : first_dog_consumption = 13)
  (h_second_dog : ∃ (second_dog_consumption : ℝ), second_dog_consumption = 2 * first_dog_consumption) :
  ∃ (third_dog_consumption : ℝ),
    third_dog_consumption = total_dogs * average_consumption - (first_dog_consumption + 2 * first_dog_consumption) :=
by sorry

end NUMINAMATH_CALUDE_third_dog_food_consumption_l3922_392219


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3922_392200

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3922_392200


namespace NUMINAMATH_CALUDE_max_value_squared_sum_max_value_squared_sum_achieved_l3922_392222

theorem max_value_squared_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
by sorry

theorem max_value_squared_sum_achieved (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ a^2 + b^2 + c^4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_sum_max_value_squared_sum_achieved_l3922_392222


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l3922_392215

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l3922_392215


namespace NUMINAMATH_CALUDE_permutations_without_patterns_l3922_392266

/-- The total number of permutations of 4 x's, 3 y's, and 2 z's -/
def total_permutations : ℕ := 1260

/-- The set of permutations where the pattern xxxx appears -/
def A₁ : Finset (List Char) := sorry

/-- The set of permutations where the pattern yyy appears -/
def A₂ : Finset (List Char) := sorry

/-- The set of permutations where the pattern zz appears -/
def A₃ : Finset (List Char) := sorry

/-- The theorem to be proved -/
theorem permutations_without_patterns (h₁ : Finset.card A₁ = 60) 
  (h₂ : Finset.card A₂ = 105) (h₃ : Finset.card A₃ = 280)
  (h₄ : Finset.card (A₁ ∩ A₂) = 12) (h₅ : Finset.card (A₁ ∩ A₃) = 20)
  (h₆ : Finset.card (A₂ ∩ A₃) = 30) (h₇ : Finset.card (A₁ ∩ A₂ ∩ A₃) = 6) :
  total_permutations - Finset.card (A₁ ∪ A₂ ∪ A₃) = 871 := by
  sorry

end NUMINAMATH_CALUDE_permutations_without_patterns_l3922_392266


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3922_392298

theorem sum_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ ∀ M : ℝ, ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 1/y > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3922_392298


namespace NUMINAMATH_CALUDE_parallelogram_vertices_parabola_parallel_intersection_l3922_392294

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def isParallelogram (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x = p4.x - p3.x ∧ p2.y - p1.y = p4.y - p3.y) ∨
  (p3.x - p1.x = p4.x - p2.x ∧ p3.y - p1.y = p4.y - p2.y)

/-- The parabola equation x = y^2 -/
def onParabola (p : Point) : Prop :=
  p.x = p.y^2

/-- Theorem about parallelogram vertices -/
theorem parallelogram_vertices :
  ∀ p : Point,
  isParallelogram ⟨0, 0⟩ ⟨1, 1⟩ ⟨1, 0⟩ p →
  (p = ⟨0, 1⟩ ∨ p = ⟨0, -1⟩ ∨ p = ⟨2, 1⟩) :=
sorry

/-- Theorem about parallel lines intersecting parabola -/
theorem parabola_parallel_intersection (a : ℝ) :
  a ≠ 0 → a ≠ 1 → a ≠ -1 →
  ∀ v : Point,
  onParabola ⟨0, 0⟩ ∧ onParabola ⟨1, 1⟩ ∧ onParabola ⟨a^2, a⟩ ∧ onParabola v →
  (∃ l1 l2 : ℝ → ℝ, l1 0 = 0 ∧ l1 1 = 1 ∧ l1 (a^2) = a ∧ l1 v.x = v.y ∧
               l2 0 = 0 ∧ l2 1 = 1 ∧ l2 (a^2) = a ∧ l2 v.x = v.y ∧
               ∀ x, l1 x - l2 x = (l1 1 - l2 1)) →
  (v = ⟨4, a⟩ ∨ v = ⟨4, -a⟩) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_vertices_parabola_parallel_intersection_l3922_392294


namespace NUMINAMATH_CALUDE_shortest_side_is_10_area_is_integer_l3922_392212

/-- Represents a triangle with integer side lengths and area --/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  area : ℕ
  sum_eq_perimeter : a + b + c = 48
  one_side_eq_21 : a = 21 ∨ b = 21 ∨ c = 21

/-- The shortest side of the triangle is 10 --/
theorem shortest_side_is_10 (t : IntegerTriangle) : 
  min t.a (min t.b t.c) = 10 := by
  sorry

/-- The area of the triangle is an integer --/
theorem area_is_integer (t : IntegerTriangle) : 
  ∃ (s : ℕ), 4 * t.area = s * s * (t.a + t.b - t.c) * (t.a + t.c - t.b) * (t.b + t.c - t.a) := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_is_10_area_is_integer_l3922_392212


namespace NUMINAMATH_CALUDE_problem_statement_l3922_392241

theorem problem_statement (x y z : ℝ) 
  (h1 : 1 / x = 2 / (y + z))
  (h2 : 1 / x = 3 / (z + x))
  (h3 : 1 / x = (x^2 - y - z) / (x + y + z))
  (h4 : x ≠ 0)
  (h5 : y + z ≠ 0)
  (h6 : z + x ≠ 0)
  (h7 : x + y + z ≠ 0) :
  (z - y) / x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3922_392241


namespace NUMINAMATH_CALUDE_taxi_theorem_l3922_392235

def taxi_distances : List ℤ := [5, 2, -4, -3, 6]
def fuel_rate : ℚ := 0.3
def base_fare : ℚ := 8
def base_distance : ℚ := 3
def extra_fare_rate : ℚ := 1.6

def final_position (distances : List ℤ) : ℤ :=
  distances.sum

def total_distance (distances : List ℤ) : ℕ :=
  distances.map Int.natAbs |>.sum

def fuel_consumed (distances : List ℤ) (rate : ℚ) : ℚ :=
  rate * (total_distance distances : ℚ)

def fare_for_distance (d : ℚ) : ℚ :=
  if d ≤ base_distance then base_fare
  else base_fare + extra_fare_rate * (d - base_distance)

def total_fare (distances : List ℤ) : ℚ :=
  distances.map (fun d => fare_for_distance (Int.natAbs d : ℚ)) |>.sum

theorem taxi_theorem :
  final_position taxi_distances = 6 ∧
  fuel_consumed taxi_distances fuel_rate = 6 ∧
  total_fare taxi_distances = 49.6 := by
  sorry

end NUMINAMATH_CALUDE_taxi_theorem_l3922_392235


namespace NUMINAMATH_CALUDE_smallest_factor_perfect_square_l3922_392289

theorem smallest_factor_perfect_square : 
  (∀ k : ℕ, k < 14 → ¬ ∃ m : ℕ, 3150 * k = m * m) ∧ 
  ∃ n : ℕ, 3150 * 14 = n * n := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_perfect_square_l3922_392289


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l3922_392242

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ initial_mean = 180 ∧ incorrect_value = 135 ∧ correct_value = 155 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum + (correct_value - incorrect_value)
  corrected_sum / n = 180.67 := by
sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l3922_392242


namespace NUMINAMATH_CALUDE_earth_hour_seating_l3922_392237

theorem earth_hour_seating (x : ℕ) : 30 * x + 8 = 31 * x - 26 := by
  sorry

end NUMINAMATH_CALUDE_earth_hour_seating_l3922_392237


namespace NUMINAMATH_CALUDE_cube_root_two_identity_l3922_392260

theorem cube_root_two_identity (s : ℝ) : s = 1 / (1 - Real.rpow 2 (1/3)) → s = -(1 + Real.rpow 2 (1/3) + Real.rpow 2 (2/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_two_identity_l3922_392260


namespace NUMINAMATH_CALUDE_playoff_average_points_l3922_392269

/-- Represents a hockey team's record --/
structure TeamRecord where
  wins : ℕ
  ties : ℕ

/-- Calculates the points for a team given their record --/
def calculatePoints (record : TeamRecord) : ℕ :=
  2 * record.wins + record.ties

/-- Theorem: The average points of the playoff teams is 27 --/
theorem playoff_average_points :
  let team1 : TeamRecord := ⟨12, 4⟩
  let team2 : TeamRecord := ⟨13, 1⟩
  let team3 : TeamRecord := ⟨8, 10⟩
  let totalPoints := calculatePoints team1 + calculatePoints team2 + calculatePoints team3
  totalPoints / 3 = 27 := by
  sorry


end NUMINAMATH_CALUDE_playoff_average_points_l3922_392269


namespace NUMINAMATH_CALUDE_maria_savings_percentage_l3922_392262

/-- Represents the "sundown deal" discount structure -/
structure SundownDeal where
  regular_price : ℝ
  second_pair_discount : ℝ
  additional_pair_discount : ℝ

/-- Calculates the total cost and savings for a given number of pairs -/
def calculate_deal (deal : SundownDeal) (num_pairs : ℕ) : ℝ × ℝ :=
  let regular_total := deal.regular_price * num_pairs
  let discounted_total := 
    if num_pairs ≥ 1 then deal.regular_price else 0 +
    if num_pairs ≥ 2 then deal.regular_price * (1 - deal.second_pair_discount) else 0 +
    if num_pairs > 2 then deal.regular_price * (1 - deal.additional_pair_discount) * (num_pairs - 2) else 0
  let savings := regular_total - discounted_total
  (discounted_total, savings)

/-- Theorem stating that Maria's savings percentage is 42% -/
theorem maria_savings_percentage (deal : SundownDeal) 
  (h1 : deal.regular_price = 60)
  (h2 : deal.second_pair_discount = 0.3)
  (h3 : deal.additional_pair_discount = 0.6) :
  let (_, savings) := calculate_deal deal 5
  let regular_total := deal.regular_price * 5
  (savings / regular_total) * 100 = 42 := by
  sorry


end NUMINAMATH_CALUDE_maria_savings_percentage_l3922_392262


namespace NUMINAMATH_CALUDE_smallest_x_value_l3922_392257

theorem smallest_x_value (x : ℝ) :
  (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 6) →
  x ≥ (-13 - Real.sqrt 17) / 2 ∧
  ∃ y : ℝ, y < (-13 - Real.sqrt 17) / 2 ∧ (y^2 - 5*y - 84) / (y - 9) ≠ 4 / (y + 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3922_392257


namespace NUMINAMATH_CALUDE_student_path_probability_l3922_392240

/-- Represents a point on the city map -/
structure Point where
  east : Nat
  south : Nat

/-- Calculates the number of paths between two points -/
def num_paths (start finish : Point) : Nat :=
  Nat.choose (finish.east - start.east + finish.south - start.south) (finish.east - start.east)

/-- The probability of choosing a specific path -/
def path_probability (start finish : Point) : ℚ :=
  1 / 2 ^ (finish.east - start.east + finish.south - start.south)

theorem student_path_probability :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨4, 3⟩
  let C : Point := ⟨2, 1⟩
  let D : Point := ⟨3, 2⟩
  let total_paths := num_paths A B
  let paths_through_C_and_D := num_paths A C * num_paths C D * num_paths D B
  paths_through_C_and_D / total_paths = 12 / 35 := by
  sorry

#eval num_paths ⟨0, 0⟩ ⟨4, 3⟩  -- Should output 35
#eval num_paths ⟨0, 0⟩ ⟨2, 1⟩ * num_paths ⟨2, 1⟩ ⟨3, 2⟩ * num_paths ⟨3, 2⟩ ⟨4, 3⟩  -- Should output 12

end NUMINAMATH_CALUDE_student_path_probability_l3922_392240


namespace NUMINAMATH_CALUDE_original_number_before_increase_l3922_392255

theorem original_number_before_increase (x : ℝ) : x * 1.5 = 165 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l3922_392255


namespace NUMINAMATH_CALUDE_min_fence_length_for_given_garden_l3922_392251

/-- Calculates the minimum fence length for a rectangular garden with one side against a wall -/
def min_fence_length (length width : ℝ) : ℝ :=
  2 * width + length

theorem min_fence_length_for_given_garden :
  min_fence_length 32 14 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_fence_length_for_given_garden_l3922_392251


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3922_392270

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3922_392270


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3922_392293

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation : 
  let l1 : Line := { a := 4, b := -5, c := 9 }
  let p : Point := { x := 4, y := 1 }
  let l2 : Line := { a := 5, b := 4, c := -24 }
  (l2.contains p ∧ Line.perpendicular l1 l2) → 
  ∀ (x y : ℝ), 5 * x + 4 * y - 24 = 0 ↔ l2.contains { x := x, y := y } :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3922_392293


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3922_392247

theorem sqrt_sum_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  2 * Real.sqrt (a + b + c + d) ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3922_392247


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3922_392252

theorem arithmetic_calculations :
  (14 - 25 + 12 - 17 = -16) ∧
  ((1/2 + 5/6 - 7/12) / (-1/36) = -27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3922_392252


namespace NUMINAMATH_CALUDE_triangle_angle_at_least_60_degrees_l3922_392201

theorem triangle_angle_at_least_60_degrees (A B C : ℝ) :
  A + B + C = 180 → A > 0 → B > 0 → C > 0 → (A ≥ 60 ∨ B ≥ 60 ∨ C ≥ 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_at_least_60_degrees_l3922_392201


namespace NUMINAMATH_CALUDE_percent_problem_l3922_392209

theorem percent_problem (x : ℝ) : (24 / x = 30 / 100) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l3922_392209


namespace NUMINAMATH_CALUDE_height_on_longest_side_l3922_392238

/-- Given a triangle with side lengths 6, 8, and 10, prove that the height on the longest side is 4.8 -/
theorem height_on_longest_side (a b c h : ℝ) : 
  a = 6 → b = 8 → c = 10 → 
  a^2 + b^2 = c^2 → 
  (1/2) * c * h = (1/2) * a * b → 
  h = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_height_on_longest_side_l3922_392238


namespace NUMINAMATH_CALUDE_tire_cost_l3922_392276

theorem tire_cost (n : ℕ+) (total_cost battery_cost : ℚ) 
  (h1 : total_cost = 224)
  (h2 : battery_cost = 56) :
  (total_cost - battery_cost) / n = (224 - 56) / n :=
by sorry

end NUMINAMATH_CALUDE_tire_cost_l3922_392276


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l3922_392295

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 60 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l3922_392295


namespace NUMINAMATH_CALUDE_count_with_six_seven_l3922_392236

/-- The number of integers from 1 to 512 in base 8 that don't use digits 6 or 7 -/
def count_without_six_seven : ℕ := 215

/-- The total number of integers we're considering -/
def total_count : ℕ := 512

theorem count_with_six_seven :
  total_count - count_without_six_seven = 297 := by
  sorry

end NUMINAMATH_CALUDE_count_with_six_seven_l3922_392236


namespace NUMINAMATH_CALUDE_least_clock_equivalent_after_10_l3922_392239

def clock_equivalent (h : ℕ) : Prop :=
  (h ^ 2 - h) % 12 = 0

theorem least_clock_equivalent_after_10 :
  ∀ h : ℕ, h > 10 → clock_equivalent h → h ≥ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_after_10_l3922_392239


namespace NUMINAMATH_CALUDE_baking_time_with_oven_failure_l3922_392272

/-- The time taken to make caramel-apple coffee cakes on a day when the oven failed -/
theorem baking_time_with_oven_failure 
  (assembly_time : ℝ) 
  (normal_baking_time : ℝ) 
  (decoration_time : ℝ) 
  (h1 : assembly_time = 1) 
  (h2 : normal_baking_time = 1.5) 
  (h3 : decoration_time = 1) :
  assembly_time + 2 * normal_baking_time + decoration_time = 5 := by
sorry

end NUMINAMATH_CALUDE_baking_time_with_oven_failure_l3922_392272


namespace NUMINAMATH_CALUDE_find_x_l3922_392291

theorem find_x : ∃ x : ℤ, 9873 + x = 13800 ∧ x = 3927 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3922_392291


namespace NUMINAMATH_CALUDE_set_relationship_l3922_392263

-- Define the sets M, P, and S
def M : Set ℤ := {x | ∃ k : ℤ, x = 3*k - 2}
def P : Set ℤ := {y | ∃ n : ℤ, y = 3*n + 1}
def S : Set ℤ := {z | ∃ m : ℤ, z = 6*m + 1}

-- State the theorem
theorem set_relationship : S ⊆ P ∧ P = M := by sorry

end NUMINAMATH_CALUDE_set_relationship_l3922_392263


namespace NUMINAMATH_CALUDE_journey_distance_l3922_392206

/-- Proves that a journey with given conditions has a total distance of 224 km -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧
    distance = 224 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3922_392206


namespace NUMINAMATH_CALUDE_six_digit_integers_count_l3922_392256

/-- The number of different six-digit integers that can be formed using the digits 2, 2, 2, 5, 5, and 9 -/
def six_digit_integers : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different six-digit integers
    formed using the digits 2, 2, 2, 5, 5, and 9 is equal to 60 -/
theorem six_digit_integers_count : six_digit_integers = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_integers_count_l3922_392256


namespace NUMINAMATH_CALUDE_problem_statement_l3922_392213

theorem problem_statement (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a * b^2 = c/a - b) : 
  ((a^2 * b^2 / c^2) - (2/c) + (1/(a^2 * b^2)) + (2*a*b / c^2) - (2/(a*b*c))) / 
  ((2/(a*b)) - (2*a*b/c)) / (101/c) = -1/202 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3922_392213


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3922_392231

theorem complex_product_magnitude : 
  Complex.abs ((-6 * Real.sqrt 3 + 6 * Complex.I) * (2 * Real.sqrt 2 - 2 * Complex.I)) = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3922_392231


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l3922_392290

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 8 →
  r = q + 8 →
  s = r + 8 →
  ones_digit p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l3922_392290


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3922_392244

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 40) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = 135) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3922_392244
