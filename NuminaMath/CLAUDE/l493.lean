import Mathlib

namespace NUMINAMATH_CALUDE_english_math_only_count_l493_49345

/-- The number of students taking at least one subject -/
def total_students : ℕ := 28

/-- The number of students taking Mathematics and History, but not English -/
def math_history_only : ℕ := 6

theorem english_math_only_count :
  ∀ (math_only english_math_only math_english_history english_history_only : ℕ),
  -- The number taking Mathematics and English only equals the number taking Mathematics only
  math_only = english_math_only →
  -- No student takes English only or History only
  -- Six students take Mathematics and History, but not English (already defined as math_history_only)
  -- The number taking English and History only is five times the number taking all three subjects
  english_history_only = 5 * math_english_history →
  -- The number taking all three subjects is even and non-zero
  math_english_history % 2 = 0 ∧ math_english_history > 0 →
  -- The total number of students is correct
  total_students = math_only + english_math_only + math_history_only + english_history_only + math_english_history →
  -- Prove that the number of students taking English and Mathematics only is 5
  english_math_only = 5 := by
sorry

end NUMINAMATH_CALUDE_english_math_only_count_l493_49345


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l493_49363

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 7/5 of a right angle
  a + b = 7 / 5 * 90 →
  -- One angle is 40° larger than the other
  b = a + 40 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 83°
  max a (max b c) = 83 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l493_49363


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l493_49307

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l493_49307


namespace NUMINAMATH_CALUDE_work_completion_time_l493_49375

/-- Given that A can do a work in 8 days and A and B together can do the work in 16/3 days,
    prove that B can do the work alone in 16 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 8) (hab : 1 / a + 1 / b = 3 / 16) :
  b = 16 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l493_49375


namespace NUMINAMATH_CALUDE_nuts_left_l493_49371

theorem nuts_left (total : ℕ) (eaten_fraction : ℚ) (left : ℕ) : 
  total = 30 → eaten_fraction = 5/6 → left = total - (eaten_fraction * total) → left = 5 := by
  sorry

end NUMINAMATH_CALUDE_nuts_left_l493_49371


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l493_49350

/-- Given a circle with equation x^2 + y^2 - 4x = 0, its symmetric circle
    with respect to the line x = 0 has the equation x^2 + y^2 + 4x = 0 -/
theorem symmetric_circle_equation : 
  ∀ (x y : ℝ), (x^2 + y^2 - 4*x = 0) → 
  ∃ (x' y' : ℝ), (x'^2 + y'^2 + 4*x' = 0) ∧ (x' = -x) ∧ (y' = y) := by
sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l493_49350


namespace NUMINAMATH_CALUDE_inequality_proof_l493_49320

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l493_49320


namespace NUMINAMATH_CALUDE_abs_geq_one_necessary_not_sufficient_for_x_gt_two_l493_49340

theorem abs_geq_one_necessary_not_sufficient_for_x_gt_two :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧
  (∃ x : ℝ, |x| ≥ 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_geq_one_necessary_not_sufficient_for_x_gt_two_l493_49340


namespace NUMINAMATH_CALUDE_expected_value_is_negative_half_l493_49388

/-- A three-sided coin with probabilities and payoffs -/
structure ThreeSidedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  prob_edge : ℚ
  payoff_heads : ℚ
  payoff_tails : ℚ
  payoff_edge : ℚ

/-- Expected value of winnings for a three-sided coin -/
def expected_value (coin : ThreeSidedCoin) : ℚ :=
  coin.prob_heads * coin.payoff_heads +
  coin.prob_tails * coin.payoff_tails +
  coin.prob_edge * coin.payoff_edge

/-- Theorem: Expected value of winnings for the given coin is -1/2 -/
theorem expected_value_is_negative_half :
  let coin : ThreeSidedCoin := {
    prob_heads := 1/4,
    prob_tails := 2/4,
    prob_edge := 1/4,
    payoff_heads := 4,
    payoff_tails := -3,
    payoff_edge := 0
  }
  expected_value coin = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_negative_half_l493_49388


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_four_l493_49323

theorem sqrt_sum_equals_four :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_four_l493_49323


namespace NUMINAMATH_CALUDE_power_of_power_l493_49334

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l493_49334


namespace NUMINAMATH_CALUDE_only_setD_cannot_form_triangle_l493_49365

/-- A set of three line segments that might form a triangle -/
structure TriangleSegments where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three line segments can form a triangle -/
def canFormTriangle (t : TriangleSegments) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The four sets of line segments given in the problem -/
def setA : TriangleSegments := ⟨3, 4, 5⟩
def setB : TriangleSegments := ⟨5, 10, 8⟩
def setC : TriangleSegments := ⟨5, 4.5, 8⟩
def setD : TriangleSegments := ⟨7, 7, 15⟩

/-- Theorem: Among the given sets, only set D cannot form a triangle -/
theorem only_setD_cannot_form_triangle :
  canFormTriangle setA ∧ 
  canFormTriangle setB ∧ 
  canFormTriangle setC ∧ 
  ¬canFormTriangle setD := by
  sorry

end NUMINAMATH_CALUDE_only_setD_cannot_form_triangle_l493_49365


namespace NUMINAMATH_CALUDE_chromium_content_bounds_l493_49377

/-- Represents the chromium content in an alloy mixture -/
structure ChromiumAlloy where
  x : ℝ  -- Relative mass of 1st alloy
  y : ℝ  -- Relative mass of 2nd alloy
  z : ℝ  -- Relative mass of 3rd alloy
  k : ℝ  -- Chromium content

/-- Conditions for a valid ChromiumAlloy -/
def is_valid_alloy (a : ChromiumAlloy) : Prop :=
  a.x ≥ 0 ∧ a.y ≥ 0 ∧ a.z ≥ 0 ∧
  a.x + a.y + a.z = 1 ∧
  0.9 * a.x + 0.3 * a.z = 0.45 ∧
  0.4 * a.x + 0.1 * a.y + 0.5 * a.z = a.k

theorem chromium_content_bounds (a : ChromiumAlloy) 
  (h : is_valid_alloy a) : 
  a.k ≥ 0.25 ∧ a.k ≤ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_chromium_content_bounds_l493_49377


namespace NUMINAMATH_CALUDE_both_hit_probability_l493_49330

/-- The probability of person A hitting the target -/
def prob_A : ℚ := 8 / 10

/-- The probability of person B hitting the target -/
def prob_B : ℚ := 7 / 10

/-- The theorem stating that the probability of both A and B hitting the target
    is equal to the product of their individual probabilities -/
theorem both_hit_probability :
  (prob_A * prob_B : ℚ) = 14 / 25 := by sorry

end NUMINAMATH_CALUDE_both_hit_probability_l493_49330


namespace NUMINAMATH_CALUDE_smallest_square_area_for_radius_6_l493_49326

/-- The area of the smallest square that can contain a circle with a given radius -/
def smallest_square_area (radius : ℝ) : ℝ :=
  (2 * radius) ^ 2

/-- Theorem: The area of the smallest square that can contain a circle with a radius of 6 is 144 -/
theorem smallest_square_area_for_radius_6 :
  smallest_square_area 6 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_for_radius_6_l493_49326


namespace NUMINAMATH_CALUDE_conditions_implications_l493_49351

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between A, B, C, and D
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_C : C → B
axiom C_nec_not_suff_D : (D → C) ∧ ¬(C → D)

-- State the theorem to be proved
theorem conditions_implications :
  -- B is a necessary but not sufficient condition for A
  ((B → A) ∧ ¬(A → B)) ∧
  -- A is a sufficient but not necessary condition for C
  ((A → C) ∧ ¬(C → A)) ∧
  -- D is neither a sufficient nor necessary condition for A
  (¬(D → A) ∧ ¬(A → D)) := by
  sorry

end NUMINAMATH_CALUDE_conditions_implications_l493_49351


namespace NUMINAMATH_CALUDE_nancy_coffee_days_l493_49303

/-- Represents Nancy's coffee buying habits and expenses -/
structure CoffeeExpense where
  double_espresso_price : ℚ
  iced_coffee_price : ℚ
  total_spent : ℚ

/-- Calculates the number of days Nancy has been buying coffee -/
def days_buying_coffee (expense : CoffeeExpense) : ℚ :=
  expense.total_spent / (expense.double_espresso_price + expense.iced_coffee_price)

/-- Theorem stating that Nancy has been buying coffee for 20 days -/
theorem nancy_coffee_days :
  let expense : CoffeeExpense := {
    double_espresso_price := 3,
    iced_coffee_price := 5/2,
    total_spent := 110
  }
  days_buying_coffee expense = 20 := by
  sorry

end NUMINAMATH_CALUDE_nancy_coffee_days_l493_49303


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l493_49364

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 15 + (3*x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l493_49364


namespace NUMINAMATH_CALUDE_fourth_root_of_polynomial_l493_49390

/-- Given a polynomial x^4 - px^3 + qx^2 - rx + s = 0 where three of its roots are
    the tangents of the angles of a triangle, the fourth root is (r - p) / (q - s - 1). -/
theorem fourth_root_of_polynomial (p q r s : ℝ) (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (h_roots : ∃ (ρ : ℝ), (Real.tan A) * (Real.tan B) * (Real.tan C) * ρ = s ∧
                        (Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + 
                        (Real.tan C) * (Real.tan A) + 
                        ((Real.tan A) + (Real.tan B) + (Real.tan C)) * ρ = q ∧
                        (Real.tan A) * (Real.tan B) * (Real.tan C) + 
                        ((Real.tan A) * (Real.tan B) + (Real.tan B) * (Real.tan C) + 
                        (Real.tan C) * (Real.tan A)) * ρ = r ∧
                        (Real.tan A) + (Real.tan B) + (Real.tan C) + ρ = p) :
  ∃ (ρ : ℝ), ρ = (r - p) / (q - s - 1) ∧
              ρ^4 - p*ρ^3 + q*ρ^2 - r*ρ + s = 0 :=
sorry

end NUMINAMATH_CALUDE_fourth_root_of_polynomial_l493_49390


namespace NUMINAMATH_CALUDE_geese_flock_count_l493_49347

theorem geese_flock_count : ∃ x : ℕ, 
  (x + x + x / 2 + x / 4 + 1 = 100) ∧ (x = 36) := by
  sorry

end NUMINAMATH_CALUDE_geese_flock_count_l493_49347


namespace NUMINAMATH_CALUDE_average_difference_l493_49357

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 70 + x) / 3) + 8 → x = 16 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l493_49357


namespace NUMINAMATH_CALUDE_prime_sum_squares_l493_49337

theorem prime_sum_squares (p q m : ℕ) : 
  p.Prime → q.Prime → p ≠ q →
  p^2 - 2001*p + m = 0 →
  q^2 - 2001*q + m = 0 →
  p^2 + q^2 = 3996005 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l493_49337


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l493_49322

theorem binomial_coefficient_equality (n : ℕ) (h : n ≥ 6) :
  (3^5 : ℚ) * (Nat.choose n 5) = (3^6 : ℚ) * (Nat.choose n 6) ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l493_49322


namespace NUMINAMATH_CALUDE_line_equation_problem_1_line_equation_problem_2_l493_49302

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statements
theorem line_equation_problem_1 (l : Line) (A : Point) :
  A.x = 0 ∧ A.y = 2 ∧ 
  (l.a^2 / (l.a^2 + l.b^2) = 1/4) →
  ∃ (k : ℝ), k > 0 ∧ l.a = k * Real.sqrt 3 ∧ l.b = -3 * k ∧ l.c = 6 * k :=
sorry

theorem line_equation_problem_2 (l l₁ : Line) (A : Point) :
  A.x = 2 ∧ A.y = 1 ∧
  l₁.a = 3 ∧ l₁.b = 4 ∧ l₁.c = 5 ∧
  (l.a / l.b = (l₁.a / l₁.b) / 2) →
  ∃ (k : ℝ), k > 0 ∧ l.a = 3 * k ∧ l.b = -k ∧ l.c = -5 * k :=
sorry

end NUMINAMATH_CALUDE_line_equation_problem_1_line_equation_problem_2_l493_49302


namespace NUMINAMATH_CALUDE_power_of_2_probability_l493_49327

/-- A number is a four-digit number in base 4 if it's between 1000₄ and 3333₄ inclusive -/
def IsFourDigitBase4 (n : ℕ) : Prop :=
  64 ≤ n ∧ n ≤ 255

/-- The count of four-digit numbers in base 4 -/
def CountFourDigitBase4 : ℕ := 255 - 64 + 1

/-- A number is a power of 2 if its log base 2 is an integer -/
def IsPowerOf2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- The count of powers of 2 that are four-digit numbers in base 4 -/
def CountPowerOf2FourDigitBase4 : ℕ := 2

/-- The probability of a randomly chosen four-digit number in base 4 being a power of 2 -/
def ProbabilityPowerOf2FourDigitBase4 : ℚ :=
  CountPowerOf2FourDigitBase4 / CountFourDigitBase4

theorem power_of_2_probability :
  ProbabilityPowerOf2FourDigitBase4 = 1 / 96 := by
  sorry

end NUMINAMATH_CALUDE_power_of_2_probability_l493_49327


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l493_49315

theorem number_of_divisors_of_36 : Finset.card (Finset.filter (· ∣ 36) (Finset.range 37)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l493_49315


namespace NUMINAMATH_CALUDE_ali_baba_maximum_value_l493_49316

/-- Represents the problem of maximizing the value of gold and diamonds in one trip --/
theorem ali_baba_maximum_value :
  let gold_weight : ℝ := 200
  let diamond_weight : ℝ := 40
  let max_carry_weight : ℝ := 100
  let gold_value_per_kg : ℝ := 20
  let diamond_value_per_kg : ℝ := 60
  
  ∀ x y : ℝ,
  x ≥ 0 → y ≥ 0 →
  x + y = max_carry_weight →
  x * gold_value_per_kg + y * diamond_value_per_kg ≤ 3000 :=
by sorry

end NUMINAMATH_CALUDE_ali_baba_maximum_value_l493_49316


namespace NUMINAMATH_CALUDE_expression_bounds_l493_49393

theorem expression_bounds : 1 < (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 ∧ 
                            (3 * Real.sqrt 2 - Real.sqrt 12) * Real.sqrt 3 < 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l493_49393


namespace NUMINAMATH_CALUDE_triangle_vertices_l493_49341

structure Triangle where
  a : ℝ
  m_a : ℝ
  s_a : ℝ

def is_valid_vertex (t : Triangle) (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = t.s_a^2 ∧ 
  |y| = t.m_a

theorem triangle_vertices (t : Triangle) 
  (h1 : t.a = 10) 
  (h2 : t.m_a = 4) 
  (h3 : t.s_a = 5) : 
  (is_valid_vertex t 8 4 ∧ 
   is_valid_vertex t 8 (-4) ∧ 
   is_valid_vertex t 2 4 ∧ 
   is_valid_vertex t 2 (-4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_vertices_l493_49341


namespace NUMINAMATH_CALUDE_right_triangle_side_relation_l493_49313

theorem right_triangle_side_relation (a d : ℝ) :
  (a > 0) →
  (d > 0) →
  (a ≤ a + 2*d) →
  (a + 2*d ≤ a + 4*d) →
  (a + 4*d)^2 = a^2 + (a + 2*d)^2 →
  a = d*(1 + Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_relation_l493_49313


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l493_49352

theorem reciprocal_sum_of_roots (a b c : ℚ) (α β : ℚ) :
  a ≠ 0 →
  (∃ x y : ℚ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) →
  (∀ x : ℚ, a * x^2 + b * x + c = 0 → (α = 1/x ∨ β = 1/x)) →
  a = 6 ∧ b = 5 ∧ c = 7 →
  α + β = -5/7 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l493_49352


namespace NUMINAMATH_CALUDE_square_area_relation_l493_49329

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := 2*a + 3*b
  let area_I := (diagonal_I^2) / 2
  let area_II := 3 * area_I
  area_II = (3 * (2*a + 3*b)^2) / 2 := by sorry

end NUMINAMATH_CALUDE_square_area_relation_l493_49329


namespace NUMINAMATH_CALUDE_bus_ride_time_l493_49336

def total_trip_time : ℕ := 8 * 60  -- 8 hours in minutes
def walk_time : ℕ := 15
def train_ride_time : ℕ := 6 * 60  -- 6 hours in minutes

def wait_time : ℕ := 2 * walk_time

def time_without_bus : ℕ := train_ride_time + walk_time + wait_time

theorem bus_ride_time : total_trip_time - time_without_bus = 75 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_time_l493_49336


namespace NUMINAMATH_CALUDE_ellipse_properties_l493_49380

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (m x y : ℝ) : Prop := y = m * (x - 1)

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line m p.1 p.2}

-- Theorem statement
theorem ellipse_properties :
  -- Part 1: Standard equation of the ellipse
  (∀ x y : ℝ, ellipse x y ↔ x^2 / 3 + y^2 / 2 = 1) ∧
  -- Part 2: Line intersects ellipse at two distinct points
  (∀ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B) ∧
  -- Part 3: No real m exists such that the circle with diameter AB passes through origin
  ¬(∃ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧
    A.1 * B.1 + A.2 * B.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l493_49380


namespace NUMINAMATH_CALUDE_percentage_increase_in_workers_l493_49305

theorem percentage_increase_in_workers (original : ℕ) (new : ℕ) : 
  original = 852 → new = 1065 → (new - original) / original * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_in_workers_l493_49305


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l493_49367

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, Q x = a * x^2 + b * x + c

theorem sum_of_roots_zero (Q : ℝ → ℝ) 
  (h_quad : is_quadratic Q)
  (h_ineq : ∀ x : ℝ, Q (x^3 - x) ≥ Q (x^2 - 1)) :
  ∃ r₁ r₂ : ℝ, (∀ x, Q x = 0 ↔ x = r₁ ∨ x = r₂) ∧ r₁ + r₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l493_49367


namespace NUMINAMATH_CALUDE_max_value_of_operation_achievable_max_value_l493_49381

theorem max_value_of_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 2 * (200 - n) ≤ 380 :=
by
  sorry

theorem achievable_max_value : 
  ∃ (n : ℕ), (10 ≤ n ∧ n ≤ 99) ∧ 2 * (200 - n) = 380 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_operation_achievable_max_value_l493_49381


namespace NUMINAMATH_CALUDE_largest_m_base_10_l493_49311

theorem largest_m_base_10 (m : ℕ) (A B C : ℕ) : 
  m > 0 ∧ 
  m = 25 * A + 5 * B + C ∧ 
  m = 81 * C + 9 * B + A ∧ 
  A < 5 ∧ B < 5 ∧ C < 5 ∧
  A < 9 ∧ B < 9 ∧ C < 9 →
  m ≤ 61 := by
sorry

end NUMINAMATH_CALUDE_largest_m_base_10_l493_49311


namespace NUMINAMATH_CALUDE_fish_pond_problem_l493_49399

/-- Represents the number of fish in a pond. -/
def N : ℕ := sorry

/-- The number of fish initially tagged and released. -/
def tagged_fish : ℕ := 40

/-- The number of fish caught in the second catch. -/
def second_catch : ℕ := 40

/-- The number of tagged fish found in the second catch. -/
def tagged_in_second_catch : ℕ := 2

/-- The fraction of tagged fish in the second catch. -/
def fraction_tagged_in_catch : ℚ := tagged_in_second_catch / second_catch

/-- The fraction of tagged fish in the pond. -/
def fraction_tagged_in_pond : ℚ := tagged_fish / N

theorem fish_pond_problem :
  fraction_tagged_in_catch = fraction_tagged_in_pond →
  N = 800 :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_problem_l493_49399


namespace NUMINAMATH_CALUDE_probability_first_odd_given_two_odd_one_even_l493_49384

/-- Represents the outcome of drawing a ball -/
inductive BallOutcome
  | Odd
  | Even

/-- Represents the result of drawing three balls -/
structure ThreeBallDraw where
  first : BallOutcome
  second : BallOutcome
  third : BallOutcome

def is_valid_draw (draw : ThreeBallDraw) : Prop :=
  (draw.first = BallOutcome.Odd ∧ draw.second = BallOutcome.Odd ∧ draw.third = BallOutcome.Even) ∨
  (draw.first = BallOutcome.Odd ∧ draw.second = BallOutcome.Even ∧ draw.third = BallOutcome.Odd)

def probability_first_odd (total_balls : ℕ) (odd_balls : ℕ) : ℚ :=
  (odd_balls : ℚ) / (total_balls : ℚ)

theorem probability_first_odd_given_two_odd_one_even 
  (total_balls : ℕ) (odd_balls : ℕ) (h1 : total_balls = 100) (h2 : odd_balls = 50) :
  probability_first_odd total_balls odd_balls = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_first_odd_given_two_odd_one_even_l493_49384


namespace NUMINAMATH_CALUDE_zach_allowance_is_five_l493_49379

/-- Calculates Zach's weekly allowance given the conditions of his savings and earnings -/
def zachsAllowance (bikeCost lawnMowingPay babysittingRatePerHour babysittingHours currentSavings additionalNeeded : ℕ) : ℕ :=
  let totalNeeded := bikeCost - additionalNeeded
  let remainingToEarn := totalNeeded - currentSavings
  let otherEarnings := lawnMowingPay + babysittingRatePerHour * babysittingHours
  remainingToEarn - otherEarnings

/-- Proves that Zach's weekly allowance is $5 given the specified conditions -/
theorem zach_allowance_is_five :
  zachsAllowance 100 10 7 2 65 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_zach_allowance_is_five_l493_49379


namespace NUMINAMATH_CALUDE_class_average_score_l493_49394

theorem class_average_score (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_average : ℚ) (group2_average : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 10 →
  group2_students = 10 →
  group1_average = 80 →
  group2_average = 60 →
  (group1_students * group1_average + group2_students * group2_average) / total_students = 70 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l493_49394


namespace NUMINAMATH_CALUDE_simplest_quadratic_root_l493_49361

theorem simplest_quadratic_root (x : ℝ) : 
  (∃ (k : ℚ), Real.sqrt (x + 1) = k * Real.sqrt (5 / 2)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplest_quadratic_root_l493_49361


namespace NUMINAMATH_CALUDE_school_pencil_order_l493_49325

/-- The number of pencils each student receives -/
def pencils_per_student : ℕ := 3

/-- The number of students in the school -/
def number_of_students : ℕ := 65

/-- The total number of pencils ordered by the school -/
def total_pencils : ℕ := pencils_per_student * number_of_students

/-- Theorem stating that the total number of pencils ordered is 195 -/
theorem school_pencil_order : total_pencils = 195 := by
  sorry

end NUMINAMATH_CALUDE_school_pencil_order_l493_49325


namespace NUMINAMATH_CALUDE_identical_numbers_l493_49318

theorem identical_numbers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y^2 = y + 1 / x^2) (h2 : y^2 + 1 / x = x^2 + 1 / y) :
  x = y :=
by sorry

end NUMINAMATH_CALUDE_identical_numbers_l493_49318


namespace NUMINAMATH_CALUDE_balloon_arrangements_l493_49333

theorem balloon_arrangements (n : ℕ) (n1 n2 n3 n4 n5 : ℕ) : 
  n = 7 → 
  n1 = 2 → 
  n2 = 2 → 
  n3 = 1 → 
  n4 = 1 → 
  n5 = 1 → 
  (n.factorial) / (n1.factorial * n2.factorial * n3.factorial * n4.factorial * n5.factorial) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l493_49333


namespace NUMINAMATH_CALUDE_patrons_in_cars_patrons_in_cars_is_twelve_l493_49300

/-- The number of patrons who came in cars to a golf tournament -/
theorem patrons_in_cars (num_carts : ℕ) (cart_capacity : ℕ) (bus_patrons : ℕ) : ℕ :=
  num_carts * cart_capacity - bus_patrons

/-- Proof that the number of patrons who came in cars is 12 -/
theorem patrons_in_cars_is_twelve : patrons_in_cars 13 3 27 = 12 := by
  sorry

end NUMINAMATH_CALUDE_patrons_in_cars_patrons_in_cars_is_twelve_l493_49300


namespace NUMINAMATH_CALUDE_average_math_score_l493_49397

def june_score : ℝ := 94.5
def patty_score : ℝ := 87.5
def josh_score : ℝ := 99.75
def henry_score : ℝ := 95.5
def lucy_score : ℝ := 91
def mark_score : ℝ := 97.25

def num_children : ℕ := 6

theorem average_math_score :
  (june_score + patty_score + josh_score + henry_score + lucy_score + mark_score) / num_children = 94.25 := by
  sorry

end NUMINAMATH_CALUDE_average_math_score_l493_49397


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l493_49358

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l493_49358


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l493_49383

/-- Given the conditions of a class height measurement error, prove the number of boys in the class. -/
theorem number_of_boys_in_class 
  (n : ℕ) -- number of boys
  (initial_average : ℝ) -- initial average height
  (wrong_height : ℝ) -- wrongly recorded height
  (correct_height : ℝ) -- correct height of the boy
  (actual_average : ℝ) -- actual average height
  (h1 : initial_average = 182)
  (h2 : wrong_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_average = 180)
  (h5 : n * initial_average - wrong_height + correct_height = n * actual_average) :
  n = 30 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l493_49383


namespace NUMINAMATH_CALUDE_share_face_value_l493_49359

/-- Given a share with the following properties:
  * dividend_rate: The dividend rate of the share (9%)
  * desired_return: The desired return on investment (12%)
  * market_value: The market value of the share in Rs. (15)
  
  This theorem proves that the face value of the share is Rs. 20. -/
theorem share_face_value
  (dividend_rate : ℝ)
  (desired_return : ℝ)
  (market_value : ℝ)
  (h1 : dividend_rate = 0.09)
  (h2 : desired_return = 0.12)
  (h3 : market_value = 15) :
  (desired_return * market_value) / dividend_rate = 20 := by
  sorry

#eval (0.12 * 15) / 0.09  -- Expected output: 20

end NUMINAMATH_CALUDE_share_face_value_l493_49359


namespace NUMINAMATH_CALUDE_prime_factorization_of_large_number_l493_49310

theorem prime_factorization_of_large_number :
  1007021035035021007001 = 7^7 * 11^7 * 13^7 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_of_large_number_l493_49310


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l493_49382

/-- The circle x^2 + y^2 = m^2 is tangent to the line x - y = m if and only if m = 0 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m^2 ∧ x - y = m ∧ 
    (∀ (x' y' : ℝ), x'^2 + y'^2 = m^2 → x' - y' = m → (x', y') = (x, y))) ↔ 
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l493_49382


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_l493_49392

/-- Represents the souvenir selling scenario with given conditions -/
structure SouvenirSales where
  cost_price : ℝ := 6
  base_price : ℝ := 8
  base_sales : ℝ := 200
  price_sales_ratio : ℝ := 10
  max_price : ℝ := 12

/-- Calculates daily sales based on selling price -/
def daily_sales (s : SouvenirSales) (x : ℝ) : ℝ :=
  s.base_sales - s.price_sales_ratio * (x - s.base_price)

/-- Calculates daily profit based on selling price -/
def daily_profit (s : SouvenirSales) (x : ℝ) : ℝ :=
  (x - s.cost_price) * (daily_sales s x)

/-- Theorem stating the maximum profit occurs at the maximum allowed price -/
theorem max_profit_at_max_price (s : SouvenirSales) :
  ∀ x, s.cost_price ≤ x ∧ x ≤ s.max_price →
    daily_profit s x ≤ daily_profit s s.max_price :=
sorry

/-- Theorem stating the value of the maximum profit -/
theorem max_profit_value (s : SouvenirSales) :
  daily_profit s s.max_price = 960 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_l493_49392


namespace NUMINAMATH_CALUDE_lindsey_squat_weight_l493_49301

/-- Calculates the total weight Lindsey will be squatting -/
def total_squat_weight (band_a : ℕ) (band_b : ℕ) (band_c : ℕ) 
                       (leg_weight : ℕ) (dumbbell : ℕ) : ℕ :=
  2 * (band_a + band_b + band_c) + 2 * leg_weight + dumbbell

/-- Proves that Lindsey's total squat weight is 65 pounds -/
theorem lindsey_squat_weight :
  total_squat_weight 7 5 3 10 15 = 65 :=
by sorry

end NUMINAMATH_CALUDE_lindsey_squat_weight_l493_49301


namespace NUMINAMATH_CALUDE_jonas_sequence_l493_49332

/-- Sequence of positive multiples of 13 in ascending order -/
def multiples_of_13 : ℕ → ℕ := λ n => 13 * (n + 1)

/-- The nth digit in the sequence of multiples of 13 -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Whether a number appears in the sequence of multiples of 13 -/
def appears_in_sequence (m : ℕ) : Prop := ∃ k : ℕ, multiples_of_13 k = m

theorem jonas_sequence :
  (nth_digit 2019 = 8) ∧ appears_in_sequence 2019 := by sorry

end NUMINAMATH_CALUDE_jonas_sequence_l493_49332


namespace NUMINAMATH_CALUDE_mary_flour_amount_l493_49343

/-- Given a recipe that requires a total amount of flour and the amount still needed to be added,
    calculate the amount of flour already put in. -/
def flour_already_added (total : ℕ) (to_add : ℕ) : ℕ :=
  total - to_add

/-- Theorem: Mary has already put in 2 cups of flour -/
theorem mary_flour_amount : flour_already_added 8 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_amount_l493_49343


namespace NUMINAMATH_CALUDE_solution_set_of_f_l493_49373

/-- A function f that satisfies the given conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x - 2) * (a * x + b)

/-- The theorem stating the solution set of f(2-x) > 0 -/
theorem solution_set_of_f (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- f is even
  (∀ x y, 0 < x → x < y → f a b x < f a b y) →  -- f is monotonically increasing in (0, +∞)
  (∀ x, f a b (2 - x) > 0 ↔ x < 0 ∨ x > 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_l493_49373


namespace NUMINAMATH_CALUDE_ratio_equals_one_l493_49309

theorem ratio_equals_one (a b c : ℝ) 
  (eq1 : 2*a + 13*b + 3*c = 90)
  (eq2 : 3*a + 9*b + c = 72) :
  (3*b + c) / (a + 2*b) = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_equals_one_l493_49309


namespace NUMINAMATH_CALUDE_point_on_line_extension_l493_49372

theorem point_on_line_extension (A B C D : EuclideanSpace ℝ (Fin 2)) :
  (D - A) = 2 • (B - A) - (C - A) →
  ∃ t : ℝ, t > 1 ∧ D = C + t • (B - C) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_extension_l493_49372


namespace NUMINAMATH_CALUDE_cubic_decreasing_l493_49370

-- Define the cubic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 1

-- State the theorem
theorem cubic_decreasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_decreasing_l493_49370


namespace NUMINAMATH_CALUDE_expression_equality_l493_49312

theorem expression_equality : (2 * Real.sqrt 2 - 1)^2 + (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = 7 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l493_49312


namespace NUMINAMATH_CALUDE_max_m_for_real_rational_roots_l493_49324

theorem max_m_for_real_rational_roots :
  let a : ℚ := 2
  let b : ℚ := -5
  let f (x m : ℚ) := a * x^2 + b * x + m
  ∀ m : ℚ, (∃ x y : ℚ, x ≠ y ∧ f x m = 0 ∧ f y m = 0) →
    m ≤ 25/8 ∧
    ¬∃ m' : ℚ, m < m' ∧ (∃ x y : ℚ, x ≠ y ∧ f x m' = 0 ∧ f y m' = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_m_for_real_rational_roots_l493_49324


namespace NUMINAMATH_CALUDE_max_profit_140000_l493_49366

structure ProductionPlan where
  productA : ℕ
  productB : ℕ

def componentAUsage (plan : ProductionPlan) : ℕ := 4 * plan.productA
def componentBUsage (plan : ProductionPlan) : ℕ := 4 * plan.productB
def totalHours (plan : ProductionPlan) : ℕ := plan.productA + 2 * plan.productB
def profit (plan : ProductionPlan) : ℕ := 20000 * plan.productA + 30000 * plan.productB

def isValidPlan (plan : ProductionPlan) : Prop :=
  componentAUsage plan ≤ 16 ∧
  componentBUsage plan ≤ 12 ∧
  totalHours plan ≤ 8

theorem max_profit_140000 :
  ∃ (optimalPlan : ProductionPlan),
    isValidPlan optimalPlan ∧
    profit optimalPlan = 140000 ∧
    ∀ (plan : ProductionPlan), isValidPlan plan → profit plan ≤ profit optimalPlan :=
sorry

end NUMINAMATH_CALUDE_max_profit_140000_l493_49366


namespace NUMINAMATH_CALUDE_average_income_P_and_Q_l493_49308

theorem average_income_P_and_Q (P Q R : ℕ) : 
  (Q + R) / 2 = 6250 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (P + Q) / 2 = 5050 := by
  sorry

end NUMINAMATH_CALUDE_average_income_P_and_Q_l493_49308


namespace NUMINAMATH_CALUDE_domain_intersection_complement_l493_49398

-- Define the universal set as real numbers
def U : Type := ℝ

-- Define the function f(x) = ln(1-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x)

-- Define the domain M of f
def M : Set ℝ := {x | x < 1}

-- Define the set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem domain_intersection_complement :
  M ∩ (Set.univ \ N) = Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_complement_l493_49398


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l493_49331

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l493_49331


namespace NUMINAMATH_CALUDE_linear_function_properties_l493_49386

/-- Linear function defined as f(x) = -2x + 4 -/
def f (x : ℝ) : ℝ := -2 * x + 4

theorem linear_function_properties :
  /- Property 1: For any two points on the graph, if x₁ < x₂, then f(x₁) > f(x₂) -/
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
  /- Property 2: The graph does not pass through the third quadrant -/
  (∀ x y : ℝ, f x = y → (x ≤ 0 → y ≥ 0) ∧ (y ≤ 0 → x ≥ 0)) ∧
  /- Property 3: Shifting the graph down by 4 units results in y = -2x -/
  (∀ x : ℝ, f x - 4 = -2 * x) ∧
  /- Property 4: The x-intercept is at (2, 0) -/
  (f 2 = 0 ∧ ∀ x : ℝ, f x = 0 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l493_49386


namespace NUMINAMATH_CALUDE_cos_2x_over_cos_pi_4_plus_x_l493_49339

theorem cos_2x_over_cos_pi_4_plus_x (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin (π/4 - x) = 5/13) : 
  Real.cos (2*x) / Real.cos (π/4 + x) = 24/13 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_over_cos_pi_4_plus_x_l493_49339


namespace NUMINAMATH_CALUDE_rectangle_parallel_to_diagonals_l493_49395

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a point on the side of a square
structure PointOnSquareSide where
  square : Square
  x : ℝ
  y : ℝ
  on_side : (x = 0 ∧ 0 ≤ y ∧ y ≤ square.side) ∨
            (y = 0 ∧ 0 ≤ x ∧ x ≤ square.side) ∨
            (x = square.side ∧ 0 ≤ y ∧ y ≤ square.side) ∨
            (y = square.side ∧ 0 ≤ x ∧ x ≤ square.side)

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ
  width_positive : width > 0
  height_positive : height > 0

-- Theorem statement
theorem rectangle_parallel_to_diagonals
  (s : Square) (p : PointOnSquareSide) (h : p.square = s) :
  ∃ (r : Rectangle), 
    -- One vertex of the rectangle is at point p
    (r.width = p.x ∧ r.height = p.y) ∨
    (r.width = s.side - p.x ∧ r.height = p.y) ∨
    (r.width = p.x ∧ r.height = s.side - p.y) ∨
    (r.width = s.side - p.x ∧ r.height = s.side - p.y) ∧
    -- Sides of the rectangle are parallel to the diagonals of the square
    (r.width / r.height = 1 ∨ r.width / r.height = -1) :=
sorry

end NUMINAMATH_CALUDE_rectangle_parallel_to_diagonals_l493_49395


namespace NUMINAMATH_CALUDE_total_carrots_grown_l493_49376

theorem total_carrots_grown (sandy_carrots sam_carrots : ℕ) 
  (h1 : sandy_carrots = 6) 
  (h2 : sam_carrots = 3) : 
  sandy_carrots + sam_carrots = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_grown_l493_49376


namespace NUMINAMATH_CALUDE_hidden_primes_average_l493_49314

-- Define the type for our cards
structure Card where
  visible : ℕ
  hidden : ℕ

-- Define the property of being consecutive primes
def ConsecutivePrimes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p > q) ∧ ∀ k, q < k → k < p → ¬Nat.Prime k

-- State the theorem
theorem hidden_primes_average (card1 card2 : Card) :
  card1.visible = 18 →
  card2.visible = 27 →
  card1.visible + card1.hidden = card2.visible + card2.hidden →
  ConsecutivePrimes card1.hidden card2.hidden →
  card1.hidden - card2.hidden = 9 →
  (card1.hidden + card2.hidden) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l493_49314


namespace NUMINAMATH_CALUDE_business_value_calculation_l493_49378

theorem business_value_calculation (owned_share : ℚ) (sold_portion : ℚ) (sale_price : ℕ) :
  owned_share = 2/3 →
  sold_portion = 3/4 →
  sale_price = 75000 →
  (sale_price : ℚ) / (owned_share * sold_portion) = 150000 := by
  sorry

end NUMINAMATH_CALUDE_business_value_calculation_l493_49378


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l493_49306

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (6*x - 5 < 3*x + 4) → x ≤ 2 ∧ (6*2 - 5 < 3*2 + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l493_49306


namespace NUMINAMATH_CALUDE_calculate_expression_l493_49368

theorem calculate_expression (a : ℝ) : (-2 * a^2)^3 / a^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l493_49368


namespace NUMINAMATH_CALUDE_train_length_l493_49389

/-- 
Given a train with a speed of 180 km/h that crosses an electric pole in 50 seconds, 
the length of the train is 2500 meters.
-/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 180 → time = 50 → length = speed * (1000 / 3600) * time → length = 2500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l493_49389


namespace NUMINAMATH_CALUDE_square_difference_square_difference_40_l493_49346

theorem square_difference (n : ℕ) : (n + 1)^2 - (n - 1)^2 = 4 * n := by
  -- The proof goes here
  sorry

-- Define the specific case for n = 40
def n : ℕ := 40

-- State the theorem for the specific case
theorem square_difference_40 : (n + 1)^2 - (n - 1)^2 = 160 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_difference_square_difference_40_l493_49346


namespace NUMINAMATH_CALUDE_book_words_per_page_l493_49356

theorem book_words_per_page (total_pages : Nat) (max_words_per_page : Nat) (remainder : Nat) :
  total_pages = 150 →
  max_words_per_page = 100 →
  remainder = 198 →
  ∃ p : Nat,
    p ≤ max_words_per_page ∧
    (total_pages * p) % 221 = remainder ∧
    p = 93 :=
by sorry

end NUMINAMATH_CALUDE_book_words_per_page_l493_49356


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l493_49387

theorem condition_p_sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l493_49387


namespace NUMINAMATH_CALUDE_pen_cost_proof_l493_49348

theorem pen_cost_proof (total_students : ℕ) (total_cost : ℚ) : ∃ (buyers pens_per_student cost_per_pen : ℕ),
  total_students = 40 ∧
  total_cost = 2091 / 100 ∧
  buyers > total_students / 2 ∧
  buyers ≤ total_students ∧
  pens_per_student % 2 = 1 ∧
  pens_per_student > 1 ∧
  Nat.Prime cost_per_pen ∧
  buyers * pens_per_student * cost_per_pen = 2091 ∧
  cost_per_pen = 47 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_proof_l493_49348


namespace NUMINAMATH_CALUDE_angle_bisector_slope_l493_49374

/-- The slope of the angle bisector of the acute angle formed at the origin
    by the lines y = x and y = 4x is -5/3 + √2. -/
theorem angle_bisector_slope : ℝ := by
  -- Define the slopes of the two lines
  let m₁ : ℝ := 1
  let m₂ : ℝ := 4

  -- Define the slope of the angle bisector
  let k : ℝ := (m₁ + m₂ + Real.sqrt (1 + m₁^2 + m₂^2)) / (1 - m₁ * m₂)

  -- Prove that k equals -5/3 + √2
  sorry

end NUMINAMATH_CALUDE_angle_bisector_slope_l493_49374


namespace NUMINAMATH_CALUDE_tom_typing_speed_l493_49396

theorem tom_typing_speed (words_per_page : ℕ) (pages_typed : ℕ) (minutes_taken : ℕ) :
  words_per_page = 450 →
  pages_typed = 10 →
  minutes_taken = 50 →
  (words_per_page * pages_typed) / minutes_taken = 90 := by
  sorry

end NUMINAMATH_CALUDE_tom_typing_speed_l493_49396


namespace NUMINAMATH_CALUDE_solution_set_theorem_l493_49362

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, f x + x * (deriv f x) > 0)

-- Define the theorem
theorem solution_set_theorem :
  {x : ℝ | (deriv f (Real.sqrt (x + 1))) > Real.sqrt (x - 1) * f (Real.sqrt (x^2 - 1))} =
  {x : ℝ | 1 ≤ x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l493_49362


namespace NUMINAMATH_CALUDE_not_right_triangle_l493_49328

theorem not_right_triangle (a b c : ℝ) (ha : a = 1/3) (hb : b = 1/4) (hc : c = 1/5) :
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l493_49328


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l493_49319

theorem sin_negative_1740_degrees : 
  Real.sin ((-1740 : ℝ) * π / 180) = (Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l493_49319


namespace NUMINAMATH_CALUDE_band_practice_schedule_l493_49391

theorem band_practice_schedule (anthony ben carlos dean : ℕ) 
  (h1 : anthony = 5)
  (h2 : ben = 6)
  (h3 : carlos = 8)
  (h4 : dean = 9) :
  Nat.lcm anthony (Nat.lcm ben (Nat.lcm carlos dean)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_band_practice_schedule_l493_49391


namespace NUMINAMATH_CALUDE_sum_of_solutions_square_equation_l493_49360

theorem sum_of_solutions_square_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 49 ∧ (x₂ - 8)^2 = 49 ∧ x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_square_equation_l493_49360


namespace NUMINAMATH_CALUDE_wholesale_price_proof_l493_49369

def retail_price : ℝ := 144

theorem wholesale_price_proof :
  ∃ (wholesale_price : ℝ),
    wholesale_price = 108 ∧
    retail_price = 144 ∧
    retail_price * 0.9 = wholesale_price + wholesale_price * 0.2 :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_proof_l493_49369


namespace NUMINAMATH_CALUDE_fence_painting_ways_l493_49304

/-- Represents the number of colors available for painting --/
def num_colors : ℕ := 3

/-- Represents the number of boards in the fence --/
def num_boards : ℕ := 10

/-- Calculates the total number of ways to paint the fence with any two adjacent boards having different colors --/
def total_ways : ℕ := num_colors * (2^(num_boards - 1))

/-- Calculates the number of ways to paint the fence using only two colors --/
def two_color_ways : ℕ := num_colors * (num_colors - 1)

/-- Theorem: The number of ways to paint a fence of 10 boards with 3 colors, 
    such that any two adjacent boards are of different colors and all three colors are used, 
    is equal to 1530 --/
theorem fence_painting_ways : 
  total_ways - two_color_ways = 1530 := by sorry

end NUMINAMATH_CALUDE_fence_painting_ways_l493_49304


namespace NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l493_49335

theorem wire_cut_square_octagon_ratio (a b : ℝ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) :
  (a^2 / 16 = b^2 * (1 + Real.sqrt 2) / 32) → a / b = Real.sqrt ((2 + Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_square_octagon_ratio_l493_49335


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_exists_141_max_141_solution_l493_49354

theorem greatest_integer_with_gcf_three (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 3 → n ≤ 141 :=
by
  sorry

theorem exists_141 : 141 < 150 ∧ Nat.gcd 141 24 = 3 :=
by
  sorry

theorem max_141 : ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ 141 :=
by
  sorry

theorem solution : (∃ n, n < 150 ∧ Nat.gcd n 24 = 3 ∧ ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n) ∧
                   (∀ n, n < 150 ∧ Nat.gcd n 24 = 3 ∧ ∀ m, m < 150 ∧ Nat.gcd m 24 = 3 → m ≤ n → n = 141) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_three_exists_141_max_141_solution_l493_49354


namespace NUMINAMATH_CALUDE_samantha_routes_l493_49353

/-- Represents a location on a grid --/
structure Location :=
  (x : ℤ) (y : ℤ)

/-- Calculates the number of shortest paths between two locations --/
def num_shortest_paths (start finish : Location) : ℕ :=
  sorry

/-- Samantha's home location relative to the southwest corner of City Park --/
def home : Location :=
  { x := -1, y := -3 }

/-- Southwest corner of City Park --/
def park_sw : Location :=
  { x := 0, y := 0 }

/-- Northeast corner of City Park --/
def park_ne : Location :=
  { x := 0, y := 0 }

/-- Samantha's school location relative to the northeast corner of City Park --/
def school : Location :=
  { x := 3, y := 1 }

/-- Library location relative to the school --/
def library : Location :=
  { x := 2, y := 1 }

/-- Total number of routes Samantha can take --/
def total_routes : ℕ :=
  (num_shortest_paths home park_sw) *
  (num_shortest_paths park_ne school) *
  (num_shortest_paths school library)

theorem samantha_routes :
  total_routes = 48 :=
sorry

end NUMINAMATH_CALUDE_samantha_routes_l493_49353


namespace NUMINAMATH_CALUDE_kamals_biology_marks_l493_49342

def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def total_subjects : ℕ := 5
def average_marks : ℕ := 75

theorem kamals_biology_marks :
  ∃ (biology_marks : ℕ),
    biology_marks = total_subjects * average_marks - (english_marks + mathematics_marks + physics_marks + chemistry_marks) :=
by sorry

end NUMINAMATH_CALUDE_kamals_biology_marks_l493_49342


namespace NUMINAMATH_CALUDE_negative_power_six_interpretation_l493_49349

theorem negative_power_six_interpretation :
  -2^6 = -(2 * 2 * 2 * 2 * 2 * 2) := by
  sorry

end NUMINAMATH_CALUDE_negative_power_six_interpretation_l493_49349


namespace NUMINAMATH_CALUDE_janes_change_l493_49321

/-- The change Jane receives when buying an apple -/
theorem janes_change (apple_price : ℚ) (paid_amount : ℚ) (change : ℚ) : 
  apple_price = 0.75 → paid_amount = 5 → change = paid_amount - apple_price → change = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_janes_change_l493_49321


namespace NUMINAMATH_CALUDE_junior_count_l493_49338

theorem junior_count (total : ℕ) (junior_percent : ℚ) (senior_percent : ℚ) :
  total = 28 →
  junior_percent = 1/4 →
  senior_percent = 1/10 →
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total ∧
    junior_percent * juniors = senior_percent * seniors ∧
    juniors = 8 := by
  sorry

end NUMINAMATH_CALUDE_junior_count_l493_49338


namespace NUMINAMATH_CALUDE_expression_evaluation_l493_49344

theorem expression_evaluation :
  let a : ℚ := -1/3
  (3*a - 1)^2 + 3*a*(3*a + 2) = 3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l493_49344


namespace NUMINAMATH_CALUDE_negation_of_proposition_l493_49317

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l493_49317


namespace NUMINAMATH_CALUDE_f_value_at_5pi_3_l493_49385

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

theorem f_value_at_5pi_3 (f : ℝ → ℝ) (h1 : is_even f)
  (h2 : smallest_positive_period f π)
  (h3 : ∀ x ∈ Set.Icc 0 (π/2), f x = Real.cos x) :
  f (5*π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_5pi_3_l493_49385


namespace NUMINAMATH_CALUDE_vector_problem_l493_49355

/-- Given vectors a and b, if vector c satisfies the conditions, then c equals the expected result. -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (2, -3) → 
  (∃ (k : ℝ), c + a = k • b) → -- (c+a) ∥ b
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) → -- c ⟂ (a+b)
  c = (-7/9, -7/3) := by
sorry


end NUMINAMATH_CALUDE_vector_problem_l493_49355
