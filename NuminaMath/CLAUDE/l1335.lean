import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l1335_133550

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 20) 
  (h2 : b + d = 4) : 
  a + c = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l1335_133550


namespace NUMINAMATH_CALUDE_F_two_F_perfect_square_l1335_133552

def best_decomposition (n : ℕ+) : ℕ+ × ℕ+ :=
  sorry

def F (n : ℕ+) : ℚ :=
  let (p, q) := best_decomposition n
  (p : ℚ) / q

theorem F_two : F 2 = 1/2 := by sorry

theorem F_perfect_square (n : ℕ+) (h : ∃ m : ℕ+, n = m * m) : F n = 1 := by sorry

end NUMINAMATH_CALUDE_F_two_F_perfect_square_l1335_133552


namespace NUMINAMATH_CALUDE_expand_product_l1335_133532

theorem expand_product (x : ℝ) : (3 * x + 4) * (x - 2) = 3 * x^2 - 2 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1335_133532


namespace NUMINAMATH_CALUDE_complex_magnitude_l1335_133543

/-- Given two complex numbers z₁ and z₂, where z₁/z₂ is purely imaginary,
    prove that the magnitude of z₁ is 10/3. -/
theorem complex_magnitude (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → Complex.abs z₁ = 10/3 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1335_133543


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1335_133508

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1335_133508


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1335_133588

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := y^2 / 9 - x^2 / 16 = 1

/-- The asymptote equation -/
def asymptote_eq (m x y : ℝ) : Prop := y = m * x ∨ y = -m * x

/-- Theorem: The value of m for the given hyperbola's asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℝ), (∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq m x y) ∧ m = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1335_133588


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1335_133507

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1335_133507


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1335_133528

theorem inscribed_circle_radius_right_triangle 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_leg : a = 15) 
  (h_proj : c - b = 16) : 
  (a + b - c) / 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1335_133528


namespace NUMINAMATH_CALUDE_unanswered_questions_l1335_133540

/-- Represents the scoring system and results for a math contest. -/
structure ContestScoring where
  total_questions : ℕ
  new_correct_points : ℕ
  new_unanswered_points : ℕ
  old_start_points : ℕ
  old_correct_points : ℕ
  old_wrong_points : ℕ
  new_score : ℕ
  old_score : ℕ

/-- Theorem stating that given the contest scoring system and Alice's scores,
    the number of unanswered questions is 8. -/
theorem unanswered_questions (cs : ContestScoring)
  (h1 : cs.total_questions = 30)
  (h2 : cs.new_correct_points = 6)
  (h3 : cs.new_unanswered_points = 3)
  (h4 : cs.old_start_points = 40)
  (h5 : cs.old_correct_points = 5)
  (h6 : cs.old_wrong_points = 2)
  (h7 : cs.new_score = 108)
  (h8 : cs.old_score = 94) :
  ∃ (c w u : ℕ), c + w + u = cs.total_questions ∧
                 cs.new_correct_points * c + cs.new_unanswered_points * u = cs.new_score ∧
                 cs.old_start_points + cs.old_correct_points * c - cs.old_wrong_points * w = cs.old_score ∧
                 u = 8 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_questions_l1335_133540


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1335_133522

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) :
  |x - y| = 8 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1335_133522


namespace NUMINAMATH_CALUDE_intersection_equals_sqrt_set_l1335_133580

-- Define the square S
def S : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the set Ct for a given t
def C (t : ℝ) : Set (ℝ × ℝ) := {p ∈ S | p.1 / t + p.2 / (1 - t) ≥ 1}

-- Define the intersection of all Ct
def intersectionC : Set (ℝ × ℝ) := ⋂ t ∈ {t | 0 < t ∧ t < 1}, C t

-- Define the set of points (x, y) in S such that √x + √y ≥ 1
def sqrtSet : Set (ℝ × ℝ) := {p ∈ S | Real.sqrt p.1 + Real.sqrt p.2 ≥ 1}

-- State the theorem
theorem intersection_equals_sqrt_set : intersectionC = sqrtSet := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_sqrt_set_l1335_133580


namespace NUMINAMATH_CALUDE_cone_height_l1335_133553

/-- The height of a cone given its slant height and lateral area -/
theorem cone_height (l : ℝ) (area : ℝ) (h : l = 13 ∧ area = 65 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ area = Real.pi * r * l ∧ Real.sqrt (l^2 - r^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l1335_133553


namespace NUMINAMATH_CALUDE_limestone_amount_l1335_133583

/-- Represents the composition of a cement compound -/
structure CementCompound where
  limestone : ℝ
  shale : ℝ
  total_weight : ℝ
  limestone_cost : ℝ
  shale_cost : ℝ
  compound_cost : ℝ

/-- Theorem stating the correct amount of limestone in the compound -/
theorem limestone_amount (c : CementCompound) 
  (h1 : c.total_weight = 100)
  (h2 : c.limestone_cost = 3)
  (h3 : c.shale_cost = 5)
  (h4 : c.compound_cost = 4.25)
  (h5 : c.limestone + c.shale = c.total_weight)
  (h6 : c.limestone * c.limestone_cost + c.shale * c.shale_cost = c.total_weight * c.compound_cost) :
  c.limestone = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_limestone_amount_l1335_133583


namespace NUMINAMATH_CALUDE_min_area_triangle_l1335_133513

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := (1 / a) * Real.log x

theorem min_area_triangle (a : ℝ) (h : a ≠ 0) :
  let P := (0, a)
  let Q := (Real.exp (a^2), a)
  let R := (0, a - 1/a)
  let area := (Real.exp (a^2)) / (2 * |a|)
  ∃ (min_area : ℝ), min_area = Real.exp (1/2) / Real.sqrt 2 ∧
    ∀ a' : ℝ, a' ≠ 0 → area ≥ min_area := by sorry

end NUMINAMATH_CALUDE_min_area_triangle_l1335_133513


namespace NUMINAMATH_CALUDE_coefficient_x4_proof_l1335_133547

/-- The coefficient of x^4 in the expansion of (x - 1/(2x))^10 -/
def coefficient_x4 : ℤ := -15

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x4_proof :
  coefficient_x4 = binomial 10 3 * (-1/2)^3 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_proof_l1335_133547


namespace NUMINAMATH_CALUDE_excircle_geometric_mean_implies_side_relation_l1335_133534

/-- 
Given a triangle with sides a, b, and c, and excircle radii ra, rb, and rc opposite to sides a, b, and c respectively,
if rc is the geometric mean of ra and rb, then c = (a^2 + b^2) / (a + b).
-/
theorem excircle_geometric_mean_implies_side_relation 
  {a b c ra rb rc : ℝ} 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ra > 0 ∧ rb > 0 ∧ rc > 0)
  (h_triangle : c < a + b)
  (h_geometric_mean : rc^2 = ra * rb) :
  c = (a^2 + b^2) / (a + b) := by
sorry

end NUMINAMATH_CALUDE_excircle_geometric_mean_implies_side_relation_l1335_133534


namespace NUMINAMATH_CALUDE_escape_theorem_l1335_133521

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circular pond -/
structure Pond where
  center : Point
  radius : ℝ

/-- Represents a person with swimming and running speeds -/
structure Person where
  position : Point
  swimSpeed : ℝ
  runSpeed : ℝ

/-- Checks if a person can escape from another in a circular pond -/
def canEscape (pond : Pond) (escaper : Person) (chaser : Person) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  ∃ (escapePoint : Point),
    (escapePoint.x - pond.center.x)^2 + (escapePoint.y - pond.center.y)^2 > pond.radius^2 ∧
    (escapePoint.x - escaper.position.x)^2 + (escapePoint.y - escaper.position.y)^2 ≤ (escaper.swimSpeed * t)^2 ∧
    (escapePoint.x - chaser.position.x)^2 + (escapePoint.y - chaser.position.y)^2 > (chaser.runSpeed * t)^2

theorem escape_theorem (pond : Pond) (x y : Person) :
  x.position = pond.center →
  (y.position.x - pond.center.x)^2 + (y.position.y - pond.center.y)^2 = pond.radius^2 →
  y.runSpeed = 4 * x.swimSpeed →
  x.runSpeed > 4 * x.swimSpeed →
  canEscape pond x y :=
sorry

end NUMINAMATH_CALUDE_escape_theorem_l1335_133521


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l1335_133578

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals available. -/
def num_crystals : ℕ := 6

/-- The number of herbs incompatible with one specific crystal. -/
def incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l1335_133578


namespace NUMINAMATH_CALUDE_average_parking_cost_senior_student_l1335_133517

/-- Calculates the average hourly parking cost for a senior citizen or student
    parking for 9 hours on a weekend, given the specified fee structure. -/
theorem average_parking_cost_senior_student (base_cost : ℝ) (additional_hourly_rate : ℝ)
  (weekend_surcharge : ℝ) (discount_rate : ℝ) (parking_duration : ℕ) :
  base_cost = 20 →
  additional_hourly_rate = 1.75 →
  weekend_surcharge = 5 →
  discount_rate = 0.1 →
  parking_duration = 9 →
  let total_cost := base_cost + (parking_duration - 2 : ℕ) * additional_hourly_rate + weekend_surcharge
  let discounted_cost := total_cost * (1 - discount_rate)
  let average_hourly_cost := discounted_cost / parking_duration
  average_hourly_cost = 3.725 := by
sorry

end NUMINAMATH_CALUDE_average_parking_cost_senior_student_l1335_133517


namespace NUMINAMATH_CALUDE_square_binomial_constant_l1335_133571

theorem square_binomial_constant (b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l1335_133571


namespace NUMINAMATH_CALUDE_wall_building_time_l1335_133501

/-- Represents the time taken to build a wall given the number of workers -/
def build_time (workers : ℕ) : ℝ :=
  sorry

/-- The number of workers in the initial scenario -/
def initial_workers : ℕ := 20

/-- The number of days taken in the initial scenario -/
def initial_days : ℝ := 6

/-- The number of workers in the new scenario -/
def new_workers : ℕ := 30

theorem wall_building_time :
  (build_time initial_workers = initial_days) →
  (∀ w₁ w₂ : ℕ, w₁ * build_time w₁ = w₂ * build_time w₂) →
  (build_time new_workers = 4.0) :=
sorry

end NUMINAMATH_CALUDE_wall_building_time_l1335_133501


namespace NUMINAMATH_CALUDE_smallest_radius_is_one_l1335_133573

/-- Triangle ABC with a circle inscribed on side AB -/
structure TriangleWithInscribedCircle where
  /-- Length of side AC -/
  ac : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The circle's center is on side AB -/
  center_on_ab : Bool
  /-- The circle is tangent to sides AC and BC -/
  tangent_to_ac_bc : Bool

/-- The smallest positive integer radius for the given triangle configuration -/
def smallest_integer_radius (t : TriangleWithInscribedCircle) : ℕ :=
  sorry

/-- Theorem stating that the smallest positive integer radius is 1 -/
theorem smallest_radius_is_one :
  ∀ t : TriangleWithInscribedCircle,
    t.ac = 5 ∧ t.bc = 3 ∧ t.center_on_ab ∧ t.tangent_to_ac_bc →
    smallest_integer_radius t = 1 :=
  sorry

end NUMINAMATH_CALUDE_smallest_radius_is_one_l1335_133573


namespace NUMINAMATH_CALUDE_composite_function_solution_l1335_133582

theorem composite_function_solution (h k : ℝ → ℝ) (b : ℝ) :
  (∀ x, h x = x / 3 + 2) →
  (∀ x, k x = 5 - 2 * x) →
  h (k b) = 4 →
  b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_composite_function_solution_l1335_133582


namespace NUMINAMATH_CALUDE_divisibility_of_p_l1335_133594

theorem divisibility_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 75)
  (h4 : 80 < Nat.gcd s.val p.val)
  (h5 : Nat.gcd s.val p.val < 120) :
  5 ∣ p.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_l1335_133594


namespace NUMINAMATH_CALUDE_perpendicular_lines_minimum_product_l1335_133563

theorem perpendicular_lines_minimum_product (b a : ℝ) : 
  b > 0 → 
  ((b^2 + 1) * (-b^2) = -1) →
  ab ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_minimum_product_l1335_133563


namespace NUMINAMATH_CALUDE_total_prime_factors_is_27_l1335_133559

/-- The total number of prime factors in the expression (4)^11 * (7)^3 * (11)^2 -/
def totalPrimeFactors : ℕ :=
  let four_factorization := 2 * 2
  let four_exponent := 11
  let seven_exponent := 3
  let eleven_exponent := 2
  (four_factorization * four_exponent) + seven_exponent + eleven_exponent

/-- Theorem stating that the total number of prime factors in the given expression is 27 -/
theorem total_prime_factors_is_27 : totalPrimeFactors = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_prime_factors_is_27_l1335_133559


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1335_133504

/-- The intersection point of two lines -/
def intersection_point (m1 a1 m2 a2 : ℚ) : ℚ × ℚ :=
  let x := (a2 - a1) / (m1 - m2)
  let y := m1 * x + a1
  (x, y)

/-- First line: y = 3x -/
def line1 (x : ℚ) : ℚ := 3 * x

/-- Second line: y + 6 = -9x, or y = -9x - 6 -/
def line2 (x : ℚ) : ℚ := -9 * x - 6

theorem intersection_of_lines :
  intersection_point 3 0 (-9) (-6) = (-1/2, -3/2) ∧
  line1 (-1/2) = -3/2 ∧
  line2 (-1/2) = -3/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1335_133504


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_three_l1335_133556

theorem subset_implies_m_equals_three (A B : Set ℕ) (m : ℕ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_three_l1335_133556


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l1335_133502

-- Define the quadratic equation
def quadratic (x k : ℝ) : Prop := x^2 - 3*x + k = 0

-- Define the condition on the roots
def root_condition (x₁ x₂ : ℝ) : Prop := x₁*x₂ + 2*x₁ + 2*x₂ = 1

-- Theorem statement
theorem quadratic_root_sum_product 
  (k : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : quadratic x₁ k) 
  (h2 : quadratic x₂ k) 
  (h3 : x₁ ≠ x₂) 
  (h4 : root_condition x₁ x₂) : 
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l1335_133502


namespace NUMINAMATH_CALUDE_inserted_digit_divisible_by_seven_l1335_133591

theorem inserted_digit_divisible_by_seven :
  ∀ x : ℕ, x < 10 →
    (20000 + x * 100 + 6) % 7 = 0 ↔ x = 0 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_inserted_digit_divisible_by_seven_l1335_133591


namespace NUMINAMATH_CALUDE_sin_transform_l1335_133529

/-- Given a function f(x) = sin(x - π/3), prove that after stretching the x-coordinates
    to twice their original length and shifting the resulting graph to the left by π/3 units,
    the resulting function is g(x) = sin(1/2x - π/6) -/
theorem sin_transform (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.sin (x - π/3)
  let g : ℝ → ℝ := fun x => Real.sin (x/2 - π/6)
  let h : ℝ → ℝ := fun x => f (x/2 + π/3)
  h = g := by sorry

end NUMINAMATH_CALUDE_sin_transform_l1335_133529


namespace NUMINAMATH_CALUDE_lcm_18_10_l1335_133587

theorem lcm_18_10 : Nat.lcm 18 10 = 36 :=
by
  have h1 : Nat.gcd 18 10 = 5 := by sorry
  sorry

end NUMINAMATH_CALUDE_lcm_18_10_l1335_133587


namespace NUMINAMATH_CALUDE_rectangle_square_division_l1335_133590

theorem rectangle_square_division (n : ℕ) : 
  (∃ (a b c d : ℕ), 
    a * b = n ∧ 
    c * d = n + 76 ∧ 
    a * d = b * c) → 
  n = 324 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_division_l1335_133590


namespace NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l1335_133575

theorem gcd_8_factorial_10_factorial :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l1335_133575


namespace NUMINAMATH_CALUDE_product_sum_equality_l1335_133546

theorem product_sum_equality : 25 * 13 * 2 + 15 * 13 * 7 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l1335_133546


namespace NUMINAMATH_CALUDE_pairing_fraction_l1335_133585

theorem pairing_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) :
  n = 4 * s / 3 →
  (s / 3 + n / 4) / (s + n) = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_pairing_fraction_l1335_133585


namespace NUMINAMATH_CALUDE_problem_statement_l1335_133520

open Set

def p (m : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (m * x^2 - m * x + 1)

def q (m : ℝ) : Prop := ∃ x₀ ∈ Icc 0 3, x₀^2 - 2*x₀ - m ≥ 0

theorem problem_statement (m : ℝ) :
  (q m ↔ m ∈ Iic 3) ∧
  ((p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Iio 0 ∪ Ioo 3 4) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1335_133520


namespace NUMINAMATH_CALUDE_increase_and_subtract_l1335_133500

theorem increase_and_subtract (initial : ℝ) (increase_percent : ℝ) (subtract : ℝ) : 
  initial = 75 → increase_percent = 150 → subtract = 40 →
  initial * (1 + increase_percent / 100) - subtract = 147.5 := by
  sorry

end NUMINAMATH_CALUDE_increase_and_subtract_l1335_133500


namespace NUMINAMATH_CALUDE_option_a_false_option_b_true_option_c_false_option_d_true_l1335_133545

-- Option A
theorem option_a_false (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ¬(1 / a > 1 / b) := by
sorry

-- Option B
theorem option_b_true (a b : ℝ) (h : a < b) (h2 : b < 0) : 
  a^2 > a * b := by
sorry

-- Option C
theorem option_c_false : 
  ∃ a b : ℝ, a > b ∧ ¬(|a| > |b|) := by
sorry

-- Option D
theorem option_d_true (a : ℝ) (h : a > 2) : 
  a + 4 / (a - 2) ≥ 6 := by
sorry

-- Final answer
def correct_options : List Char := ['B', 'D']

end NUMINAMATH_CALUDE_option_a_false_option_b_true_option_c_false_option_d_true_l1335_133545


namespace NUMINAMATH_CALUDE_tooth_fairy_calculation_l1335_133570

theorem tooth_fairy_calculation (total_amount : ℕ) (total_teeth : ℕ) (lost_teeth : ℕ) (first_tooth_amount : ℕ) :
  total_teeth = 20 →
  total_amount = 54 →
  lost_teeth = 2 →
  first_tooth_amount = 20 →
  (total_amount - first_tooth_amount) / (total_teeth - lost_teeth - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_tooth_fairy_calculation_l1335_133570


namespace NUMINAMATH_CALUDE_smallest_odd_number_l1335_133535

theorem smallest_odd_number (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 2) + (x + 4) = (x + 4) + 28) →  -- sum condition
  (x = 13) :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_smallest_odd_number_l1335_133535


namespace NUMINAMATH_CALUDE_probability_of_three_positive_answers_l1335_133538

/-- The probability of getting exactly k successes in n trials,
    where the probability of success on each trial is p. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of questions asked -/
def total_questions : ℕ := 7

/-- The number of positive answers we're interested in -/
def positive_answers : ℕ := 3

/-- The probability of a positive answer for each question -/
def positive_probability : ℚ := 3/7

theorem probability_of_three_positive_answers :
  binomial_probability total_questions positive_answers positive_probability = 242112/823543 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_positive_answers_l1335_133538


namespace NUMINAMATH_CALUDE_mike_picked_limes_l1335_133544

/-- The number of limes Alyssa ate -/
def limes_eaten : ℝ := 25.0

/-- The number of limes left -/
def limes_left : ℕ := 7

/-- The number of limes Mike picked -/
def mikes_limes : ℝ := limes_eaten + limes_left

theorem mike_picked_limes : mikes_limes = 32 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_limes_l1335_133544


namespace NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_element_l1335_133503

theorem pascals_triangle_56th_row_second_element :
  let n : ℕ := 56  -- The row number (0-indexed) with 57 elements
  let k : ℕ := 1   -- The position of the second element (0-indexed)
  Nat.choose n k = 56 := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_56th_row_second_element_l1335_133503


namespace NUMINAMATH_CALUDE_odd_function_extension_l1335_133592

-- Define the function f on the positive real numbers
def f_pos (x : ℝ) : ℝ := x * (x - 1)

-- State the theorem
theorem odd_function_extension {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x > 0, f x = f_pos x) : 
  ∀ x < 0, f x = x * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1335_133592


namespace NUMINAMATH_CALUDE_tan_product_seventh_roots_l1335_133512

theorem tan_product_seventh_roots : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_roots_l1335_133512


namespace NUMINAMATH_CALUDE_compare_large_exponents_l1335_133599

theorem compare_large_exponents :
  20^(19^20) > 19^(20^19) := by
  sorry

end NUMINAMATH_CALUDE_compare_large_exponents_l1335_133599


namespace NUMINAMATH_CALUDE_polygon_sides_l1335_133514

theorem polygon_sides (sum_known_angles : ℕ) (angle_a angle_b angle_c : ℕ) :
  sum_known_angles = 3780 →
  angle_a = 3 * angle_c →
  angle_b = 3 * angle_c →
  ∃ (n : ℕ), n = 23 ∧ sum_known_angles = 180 * (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1335_133514


namespace NUMINAMATH_CALUDE_sum_xyz_equals_five_l1335_133584

theorem sum_xyz_equals_five (x y z : ℝ) 
  (eq1 : x + 2*y + 3*z = 10) 
  (eq2 : 4*x + 3*y + 2*z = 15) : 
  x + y + z = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_five_l1335_133584


namespace NUMINAMATH_CALUDE_train_length_calculation_l1335_133564

/-- Given a train that crosses a platform in 54 seconds and a signal pole in 18 seconds,
    where the platform length is 600.0000000000001 meters, prove that the length of the train
    is 300.00000000000005 meters. -/
theorem train_length_calculation (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 54)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 600.0000000000001) :
    ∃ (train_length : ℝ), train_length = 300.00000000000005 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1335_133564


namespace NUMINAMATH_CALUDE_range_of_a_l1335_133506

open Set

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x < a}
def B : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (A a ∪ (Bᶜ) = univ) → a ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1335_133506


namespace NUMINAMATH_CALUDE_three_number_sum_l1335_133542

theorem three_number_sum (A B C : ℝ) (h1 : A/B = 2/3) (h2 : B/C = 5/8) (h3 : B = 30) :
  A + B + C = 98 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l1335_133542


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1335_133548

/-- An isosceles triangle with side lengths 2 and 4 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_isosceles : base = 2 ∧ leg = 4

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2 * t.leg

/-- Theorem: The perimeter of an isosceles triangle with side lengths 2 and 4 is 10 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 10 := by
  sorry

#check isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1335_133548


namespace NUMINAMATH_CALUDE_semicircle_triangle_area_ratio_l1335_133593

/-- Given a triangle ABC with sides in ratio 2:3:4 and an inscribed semicircle
    with diameter on the longest side, the ratio of the area of the semicircle
    to the area of the triangle is π√15 / 12 -/
theorem semicircle_triangle_area_ratio (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ratio : a / b = 2 / 3 ∧ b / c = 3 / 4) (h_triangle : (a + b > c) ∧ (b + c > a) ∧ (c + a > b)) :
  let s := (a + b + c) / 2
  let triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let semicircle_area := π * c^2 / 8
  semicircle_area / triangle_area = π * Real.sqrt 15 / 12 := by
sorry

end NUMINAMATH_CALUDE_semicircle_triangle_area_ratio_l1335_133593


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_522_l1335_133574

theorem sin_n_equals_cos_522 :
  ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (522 * π / 180) :=
by
  use -72
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_522_l1335_133574


namespace NUMINAMATH_CALUDE_tourist_walking_speed_l1335_133505

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hMinutesValid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Represents the problem scenario -/
structure TouristProblem where
  scheduledArrival : Time
  actualArrival : Time
  busSpeed : ℝ
  earlyArrival : ℕ

/-- Calculates the tourists' walking speed -/
noncomputable def touristSpeed (problem : TouristProblem) : ℝ :=
  let walkingTime := timeDifference problem.actualArrival problem.scheduledArrival - problem.earlyArrival
  let distance := problem.busSpeed * (problem.earlyArrival / 2) / 60
  distance / (walkingTime / 60)

/-- The main theorem to prove -/
theorem tourist_walking_speed (problem : TouristProblem) 
  (hScheduledArrival : problem.scheduledArrival = ⟨5, 0, by norm_num⟩)
  (hActualArrival : problem.actualArrival = ⟨3, 10, by norm_num⟩)
  (hBusSpeed : problem.busSpeed = 60)
  (hEarlyArrival : problem.earlyArrival = 20) :
  touristSpeed problem = 6 := by
  sorry

end NUMINAMATH_CALUDE_tourist_walking_speed_l1335_133505


namespace NUMINAMATH_CALUDE_existence_of_x0_l1335_133518

theorem existence_of_x0 (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc 1 9, |a * x₀ + b + 9 / x₀| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x0_l1335_133518


namespace NUMINAMATH_CALUDE_number_wall_top_l1335_133572

/-- Represents a number wall with 5 base numbers -/
structure NumberWall (a b c d e : ℕ) where
  level1 : Fin 5 → ℕ
  level2 : Fin 4 → ℕ
  level3 : Fin 3 → ℕ
  level4 : Fin 2 → ℕ
  top : ℕ
  base_correct : level1 = ![a, b, c, d, e]
  level2_correct : ∀ i : Fin 4, level2 i = level1 i + level1 (i.succ)
  level3_correct : ∀ i : Fin 3, level3 i = level2 i + level2 (i.succ)
  level4_correct : ∀ i : Fin 2, level4 i = level3 i + level3 (i.succ)
  top_correct : top = level4 0 + level4 1

/-- The theorem stating that the top of the number wall is x + 103 -/
theorem number_wall_top (x : ℕ) : 
  ∀ (w : NumberWall x 4 8 7 11), w.top = x + 103 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_top_l1335_133572


namespace NUMINAMATH_CALUDE_hexagon_planes_count_l1335_133549

/-- A regular dodecahedron in three-dimensional space. -/
structure RegularDodecahedron

/-- A plane in three-dimensional space. -/
structure Plane

/-- The number of large diagonals in a regular dodecahedron. -/
def num_large_diagonals : ℕ := 10

/-- The number of planes perpendicular to each large diagonal that produce a regular hexagon slice. -/
def planes_per_diagonal : ℕ := 3

/-- A function that counts the number of planes intersecting a regular dodecahedron to produce a regular hexagon. -/
def count_hexagon_planes (d : RegularDodecahedron) : ℕ :=
  num_large_diagonals * planes_per_diagonal

/-- Theorem stating that the number of planes intersecting a regular dodecahedron to produce a regular hexagon is 30. -/
theorem hexagon_planes_count (d : RegularDodecahedron) :
  count_hexagon_planes d = 30 := by sorry

end NUMINAMATH_CALUDE_hexagon_planes_count_l1335_133549


namespace NUMINAMATH_CALUDE_pet_insurance_savings_l1335_133539

def surgery_cost : ℝ := 7500

def plan_a_months : ℕ := 24
def plan_a_premium : ℝ := 15
def plan_a_coverage : ℝ := 0.60

def plan_b_months : ℕ := 12
def plan_b_premium : ℝ := 25
def plan_b_coverage : ℝ := 0.90

def yearly_deductible : ℝ := 100

def total_premiums : ℝ := plan_a_months * plan_a_premium + plan_b_months * plan_b_premium

def total_deductibles : ℝ := 2 * yearly_deductible + yearly_deductible

def insurance_coverage : ℝ := plan_b_coverage * (surgery_cost - yearly_deductible)

def cost_with_insurance : ℝ := total_premiums + total_deductibles + (surgery_cost - insurance_coverage)

theorem pet_insurance_savings : surgery_cost - cost_with_insurance = 5700 := by
  sorry

end NUMINAMATH_CALUDE_pet_insurance_savings_l1335_133539


namespace NUMINAMATH_CALUDE_multiplication_error_correction_l1335_133569

theorem multiplication_error_correction (N : ℝ) (x : ℝ) : 
  (((N * x - N / 5) / (N * x)) * 100 = 93.33333333333333) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_multiplication_error_correction_l1335_133569


namespace NUMINAMATH_CALUDE_no_positive_triples_sum_l1335_133595

theorem no_positive_triples_sum : 
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a = b + c ∧ b = c + a ∧ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_triples_sum_l1335_133595


namespace NUMINAMATH_CALUDE_bedroom_doors_count_l1335_133577

theorem bedroom_doors_count : 
  ∀ (outside_doors bedroom_doors : ℕ) 
    (outside_door_cost bedroom_door_cost total_cost : ℚ),
  outside_doors = 2 →
  outside_door_cost = 20 →
  bedroom_door_cost = outside_door_cost / 2 →
  total_cost = 70 →
  outside_doors * outside_door_cost + bedroom_doors * bedroom_door_cost = total_cost →
  bedroom_doors = 3 := by
sorry

end NUMINAMATH_CALUDE_bedroom_doors_count_l1335_133577


namespace NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1335_133510

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 36) 
  (product_eq : x * y = 320) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_numbers_l1335_133510


namespace NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l1335_133566

/-- A graph is a set of vertices and a set of edges between them. -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The degree of a vertex in a graph is the number of edges connected to it. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each consecutive pair is connected by an edge. -/
def is_path (G : Graph V) (path : List V) : Prop := sorry

/-- A cycle is a path that starts and ends at the same vertex. -/
def is_cycle (G : Graph V) (cycle : List V) : Prop := sorry

/-- The main theorem: In any graph where each vertex has degree at least 3,
    there exists a cycle whose length is not divisible by 3. -/
theorem exists_cycle_not_div_by_three {V : Type} (G : Graph V) :
  (∀ v : V, degree G v ≥ 3) →
  ∃ cycle : List V, is_cycle G cycle ∧ (cycle.length % 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_cycle_not_div_by_three_l1335_133566


namespace NUMINAMATH_CALUDE_natural_exp_inequality_l1335_133589

theorem natural_exp_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_natural_exp_inequality_l1335_133589


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_neg_three_l1335_133519

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem decreasing_function_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 4 → f a x₁ > f a x₂) → a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_leq_neg_three_l1335_133519


namespace NUMINAMATH_CALUDE_crayon_distribution_l1335_133526

theorem crayon_distribution (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) : 
  total_crayons = 24 → num_people = 3 → crayons_per_person = total_crayons / num_people → crayons_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_l1335_133526


namespace NUMINAMATH_CALUDE_train_length_l1335_133515

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (bridge_length : ℝ)
  (h1 : train_speed = 45)  -- km/hr
  (h2 : crossing_time = 30 / 3600)  -- convert seconds to hours
  (h3 : bridge_length = 220 / 1000)  -- convert meters to kilometers
  : (train_speed * crossing_time - bridge_length) * 1000 = 155 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1335_133515


namespace NUMINAMATH_CALUDE_eighteen_power_equality_l1335_133565

theorem eighteen_power_equality (m n : ℤ) (P Q : ℝ) 
  (hP : P = 2^m) (hQ : Q = 3^n) : 
  18^(m+n) = P^(m+n) * Q^(2*(m+n)) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_power_equality_l1335_133565


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1335_133523

theorem quadratic_equation_solution (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) → 
  m^2 - 3 * m + 2 = 0 → 
  m - 1 ≠ 0 → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1335_133523


namespace NUMINAMATH_CALUDE_coin_difference_l1335_133531

def coin_values : List Nat := [5, 15, 20]

def target_amount : Nat := 50

def min_coins (values : List Nat) (target : Nat) : Nat :=
  sorry

def max_coins (values : List Nat) (target : Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 6 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_l1335_133531


namespace NUMINAMATH_CALUDE_pencil_cost_l1335_133598

theorem pencil_cost (total_students : ℕ) (total_cost : ℚ) : ∃ (buyers pencils_per_student pencil_cost : ℕ),
  total_students = 30 ∧
  total_cost = 1771 / 100 ∧
  buyers > total_students / 2 ∧
  buyers ≤ total_students ∧
  pencils_per_student > 1 ∧
  pencil_cost > pencils_per_student ∧
  buyers * pencils_per_student * pencil_cost = 1771 ∧
  pencil_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l1335_133598


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1335_133524

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length 2√3. -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (2, -1))
  (h_focus : focus = (2, -3))
  (h_semi_major_endpoint : semi_major_endpoint = (2, 3)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l1335_133524


namespace NUMINAMATH_CALUDE_alice_notebook_savings_l1335_133581

/-- The amount Alice saves when buying notebooks during a sale -/
theorem alice_notebook_savings (number_of_notebooks : ℕ) (original_price : ℚ) (discount_rate : ℚ) :
  number_of_notebooks = 8 →
  original_price = 375/100 →
  discount_rate = 25/100 →
  (number_of_notebooks * original_price) - (number_of_notebooks * (original_price * (1 - discount_rate))) = 75/10 := by
  sorry

end NUMINAMATH_CALUDE_alice_notebook_savings_l1335_133581


namespace NUMINAMATH_CALUDE_probability_at_2_3_after_5_moves_l1335_133558

/-- Represents the probability of a particle reaching a specific point after a number of moves -/
def particle_probability (x y n : ℕ) : ℚ :=
  if x + y = n then
    (n.choose y : ℚ) * (1/2)^n
  else
    0

/-- Theorem stating the probability of reaching (2,3) after 5 moves -/
theorem probability_at_2_3_after_5_moves :
  particle_probability 2 3 5 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_2_3_after_5_moves_l1335_133558


namespace NUMINAMATH_CALUDE_simplify_expressions_l1335_133567

variable (a b : ℝ)

theorem simplify_expressions :
  (4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b) ∧
  (2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1335_133567


namespace NUMINAMATH_CALUDE_min_expression_upper_bound_l1335_133555

theorem min_expression_upper_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min (min (min (1 / a) (2 / b)) (4 / c)) (Real.rpow (a * b * c) (1 / 3)) ≤ Real.sqrt 2 ∧
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    min (min (min (1 / a) (2 / b)) (4 / c)) (Real.rpow (a * b * c) (1 / 3)) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_expression_upper_bound_l1335_133555


namespace NUMINAMATH_CALUDE_special_bin_op_property_l1335_133562

/-- A binary operation on a set S satisfying (a * b) * a = b for all a, b ∈ S -/
class SpecialBinOp (S : Type) where
  op : S → S → S
  identity : ∀ a b : S, op (op a b) a = b

/-- 
If S has a binary operation satisfying (a * b) * a = b for all a, b ∈ S,
then a * (b * a) = b for all a, b ∈ S
-/
theorem special_bin_op_property {S : Type} [SpecialBinOp S] :
  ∀ a b : S, SpecialBinOp.op a (SpecialBinOp.op b a) = b :=
by sorry

end NUMINAMATH_CALUDE_special_bin_op_property_l1335_133562


namespace NUMINAMATH_CALUDE_dollar_op_five_neg_two_l1335_133511

def dollar_op (x y : ℤ) : ℤ := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_five_neg_two : dollar_op 5 (-2) = -45 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_five_neg_two_l1335_133511


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1335_133541

/-- Hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0) -/
structure Hyperbola (a b : ℝ) : Prop where
  a_pos : a > 0
  b_pos : b > 0

/-- Line l with equation y = 2x - 2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 * p.1 - 2}

/-- The asymptote of the hyperbola C -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (b / a) * p.1 ∨ p.2 = -(b / a) * p.1}

/-- Predicate to check if a line is parallel to an asymptote -/
def is_parallel_to_asymptote (h : Hyperbola a b) : Prop :=
  b / a = 2

/-- Predicate to check if a line passes through a vertex of the hyperbola -/
def passes_through_vertex (h : Hyperbola a b) : Prop :=
  (1, 0) ∈ Line

/-- The distance from the focus of the hyperbola to its asymptote -/
def focus_asymptote_distance (h : Hyperbola a b) : ℝ := b

/-- Main theorem -/
theorem hyperbola_focus_asymptote_distance 
  (h : Hyperbola a b) 
  (parallel : is_parallel_to_asymptote h) 
  (through_vertex : passes_through_vertex h) : 
  focus_asymptote_distance h = 2 := by
    sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l1335_133541


namespace NUMINAMATH_CALUDE_expression_value_l1335_133576

theorem expression_value :
  let x : ℚ := 2
  let y : ℚ := 3
  let z : ℚ := 4
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1335_133576


namespace NUMINAMATH_CALUDE_catrionas_fish_count_catrionas_aquarium_l1335_133586

theorem catrionas_fish_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun goldfish angelfish guppies total =>
    (goldfish = 8) →
    (angelfish = goldfish + 4) →
    (guppies = 2 * angelfish) →
    (total = goldfish + angelfish + guppies) →
    (total = 44)

-- Proof
theorem catrionas_aquarium : catrionas_fish_count 8 12 24 44 := by
  sorry

end NUMINAMATH_CALUDE_catrionas_fish_count_catrionas_aquarium_l1335_133586


namespace NUMINAMATH_CALUDE_initial_men_count_l1335_133509

/-- Given a group of men where:
  * The average age increases by 2 years when two women replace two men
  * The replaced men are 10 and 12 years old
  * The average age of the women is 21 years
  Prove that the initial number of men in the group is 10 -/
theorem initial_men_count (M : ℕ) (A : ℚ) : 
  (M * A - 22 + 42 = M * (A + 2)) → M = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l1335_133509


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1335_133536

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  prop1 : a 2 * a 6 = 16
  prop2 : a 4 + a 8 = 8

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : seq.a 20 / seq.a 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1335_133536


namespace NUMINAMATH_CALUDE_fixed_point_of_arithmetic_sequence_l1335_133530

/-- If k, -1, and b form an arithmetic sequence, then the line y = kx + b passes through the point (1, -2). -/
theorem fixed_point_of_arithmetic_sequence (k b : ℝ) : 
  (k - (-1) = (-1) - b) → 
  ∃ (y : ℝ), y = k * 1 + b ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_arithmetic_sequence_l1335_133530


namespace NUMINAMATH_CALUDE_max_value_of_a_l1335_133525

theorem max_value_of_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → Real.sqrt x - Real.sqrt (4 - x) ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → Real.sqrt x - Real.sqrt (4 - x) ≥ b) → b ≤ -2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1335_133525


namespace NUMINAMATH_CALUDE_hoseok_payment_l1335_133579

/-- The price of item (a) bought by Hoseok at the mart -/
def item_price : ℕ := 7450

/-- The number of 1000 won bills used -/
def bills_1000 : ℕ := 7

/-- The number of 100 won coins used -/
def coins_100 : ℕ := 4

/-- The number of 10 won coins used -/
def coins_10 : ℕ := 5

/-- The denomination of the bills used -/
def bill_value : ℕ := 1000

/-- The denomination of the first type of coins used -/
def coin_value_100 : ℕ := 100

/-- The denomination of the second type of coins used -/
def coin_value_10 : ℕ := 10

theorem hoseok_payment :
  item_price = bills_1000 * bill_value + coins_100 * coin_value_100 + coins_10 * coin_value_10 :=
by sorry

end NUMINAMATH_CALUDE_hoseok_payment_l1335_133579


namespace NUMINAMATH_CALUDE_factorial_difference_l1335_133527

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1335_133527


namespace NUMINAMATH_CALUDE_tree_height_difference_l1335_133554

theorem tree_height_difference : 
  let maple_height : ℚ := 13 + 1/4
  let pine_height : ℚ := 19 + 3/8
  pine_height - maple_height = 6 + 1/8 := by
sorry

end NUMINAMATH_CALUDE_tree_height_difference_l1335_133554


namespace NUMINAMATH_CALUDE_fractional_decomposition_sum_l1335_133537

theorem fractional_decomposition_sum (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_fractional_decomposition_sum_l1335_133537


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1335_133596

/-- Proves that the equation of a line passing through point (2, -1) with an inclination angle of π/4 is x - y - 3 = 0 -/
theorem line_equation_through_point_with_inclination (x y : ℝ) :
  let point : ℝ × ℝ := (2, -1)
  let inclination : ℝ := π / 4
  let slope : ℝ := Real.tan inclination
  (y - point.2 = slope * (x - point.1)) → (x - y - 3 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l1335_133596


namespace NUMINAMATH_CALUDE_second_year_undeclared_fraction_l1335_133551

theorem second_year_undeclared_fraction :
  let total : ℚ := 1
  let first_year : ℚ := 1/4
  let second_year : ℚ := 1/2
  let third_year : ℚ := 1/6
  let fourth_year : ℚ := 1/12
  let first_year_undeclared : ℚ := 4/5
  let second_year_undeclared : ℚ := 3/4
  let third_year_undeclared : ℚ := 1/3
  let fourth_year_undeclared : ℚ := 1/6
  
  first_year + second_year + third_year + fourth_year = total →
  second_year * second_year_undeclared = 1/3
  := by sorry

end NUMINAMATH_CALUDE_second_year_undeclared_fraction_l1335_133551


namespace NUMINAMATH_CALUDE_yunas_grandfather_age_l1335_133533

/-- Proves the age of Yuna's grandfather given the ages and age differences of family members. -/
theorem yunas_grandfather_age 
  (yuna_age : ℕ) 
  (father_age_diff : ℕ) 
  (grandfather_age_diff : ℕ) 
  (h1 : yuna_age = 9)
  (h2 : father_age_diff = 27)
  (h3 : grandfather_age_diff = 23) : 
  yuna_age + father_age_diff + grandfather_age_diff = 59 :=
by sorry

end NUMINAMATH_CALUDE_yunas_grandfather_age_l1335_133533


namespace NUMINAMATH_CALUDE_product_of_nonreal_roots_l1335_133561

theorem product_of_nonreal_roots : ∃ (r₁ r₂ : ℂ),
  (r₁ ∈ {z : ℂ | z^4 - 4*z^3 + 6*z^2 - 4*z = 2005 ∧ z.im ≠ 0}) ∧
  (r₂ ∈ {z : ℂ | z^4 - 4*z^3 + 6*z^2 - 4*z = 2005 ∧ z.im ≠ 0}) ∧
  r₁ ≠ r₂ ∧
  r₁ * r₂ = 1 + Real.sqrt 2006 :=
by sorry

end NUMINAMATH_CALUDE_product_of_nonreal_roots_l1335_133561


namespace NUMINAMATH_CALUDE_cadence_earnings_increase_l1335_133597

/-- Proves that the percentage increase in Cadence's monthly earnings at her new company
    compared to her old company is 20%, given the specified conditions. -/
theorem cadence_earnings_increase (
  old_company_duration : ℕ := 3 * 12
  ) (old_company_monthly_salary : ℕ := 5000)
  (new_company_duration_increase : ℕ := 5)
  (total_earnings : ℕ := 426000) : Real :=
  by
  -- Define the duration of employment at the new company
  let new_company_duration : ℕ := old_company_duration + new_company_duration_increase

  -- Calculate total earnings from the old company
  let old_company_total : ℕ := old_company_duration * old_company_monthly_salary

  -- Calculate total earnings from the new company
  let new_company_total : ℕ := total_earnings - old_company_total

  -- Calculate monthly salary at the new company
  let new_company_monthly_salary : ℕ := new_company_total / new_company_duration

  -- Calculate the percentage increase
  let percentage_increase : Real := 
    (new_company_monthly_salary - old_company_monthly_salary : Real) / old_company_monthly_salary * 100

  -- Prove that the percentage increase is 20%
  sorry


end NUMINAMATH_CALUDE_cadence_earnings_increase_l1335_133597


namespace NUMINAMATH_CALUDE_nathaniel_best_friends_l1335_133557

theorem nathaniel_best_friends (total_tickets : ℕ) (tickets_per_friend : ℕ) (tickets_left : ℕ) :
  total_tickets = 11 →
  tickets_per_friend = 2 →
  tickets_left = 3 →
  (total_tickets - tickets_left) / tickets_per_friend = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_nathaniel_best_friends_l1335_133557


namespace NUMINAMATH_CALUDE_triangle_area_l1335_133560

theorem triangle_area (b c : ℝ) (angle_C : ℝ) (h1 : b = 1) (h2 : c = Real.sqrt 3) (h3 : angle_C = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin (Real.pi / 6) = Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1335_133560


namespace NUMINAMATH_CALUDE_grid_toothpicks_l1335_133568

/-- Calculates the total number of toothpicks in a grid with internal lines -/
def total_toothpicks (length width spacing : ℕ) : ℕ :=
  let vertical_lines := length / spacing + 1 + length % spacing
  let horizontal_lines := width / spacing + 1 + width % spacing
  vertical_lines * width + horizontal_lines * length

/-- Proves that a grid of 50x40 toothpicks with internal lines every 10 toothpicks uses 4490 toothpicks -/
theorem grid_toothpicks : total_toothpicks 50 40 10 = 4490 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l1335_133568


namespace NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l1335_133516

theorem four_digit_perfect_cubes_divisible_by_16 :
  (∃! (count : ℕ), ∃ (S : Finset ℕ),
    S.card = count ∧
    (∀ n ∈ S, 1000 ≤ n ∧ n ≤ 9999) ∧
    (∀ n ∈ S, ∃ m : ℕ, n = m^3) ∧
    (∀ n ∈ S, n % 16 = 0) ∧
    (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m^3) ∧ n % 16 = 0 → n ∈ S)) ∧
  count = 3 :=
sorry

end NUMINAMATH_CALUDE_four_digit_perfect_cubes_divisible_by_16_l1335_133516
