import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_proof_l1741_174188

/-- Given a point M on the line 2x + y - 1 = 0 and points (3,0) and (0,1) on a circle centered at M,
    prove that the equation of this circle is (x-1)² + (y+1)² = 5 -/
theorem circle_equation_proof (M : ℝ × ℝ) :
  (2 * M.1 + M.2 - 1 = 0) →
  ((3 : ℝ) - M.1)^2 + (0 - M.2)^2 = ((0 : ℝ) - M.1)^2 + (1 - M.2)^2 →
  ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ↔ (x - M.1)^2 + (y - M.2)^2 = ((3 : ℝ) - M.1)^2 + (0 - M.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1741_174188


namespace NUMINAMATH_CALUDE_sqrt_24_minus_3sqrt_2_3_l1741_174170

theorem sqrt_24_minus_3sqrt_2_3 : Real.sqrt 24 - 3 * Real.sqrt (2/3) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_24_minus_3sqrt_2_3_l1741_174170


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1741_174112

theorem absolute_value_inequality (x : ℝ) : |2*x + 1| - 2*|x - 1| > 0 ↔ x > (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1741_174112


namespace NUMINAMATH_CALUDE_min_shots_theorem_l1741_174194

/-- The hit rate for each shot -/
def hit_rate : ℝ := 0.25

/-- The desired probability of hitting the target at least once -/
def desired_probability : ℝ := 0.75

/-- The probability of hitting the target at least once given n shots -/
def prob_hit_at_least_once (n : ℕ) : ℝ := 1 - (1 - hit_rate) ^ n

/-- The minimum number of shots required to achieve the desired probability -/
def min_shots : ℕ := 5

theorem min_shots_theorem :
  (∀ k < min_shots, prob_hit_at_least_once k < desired_probability) ∧
  prob_hit_at_least_once min_shots ≥ desired_probability :=
by sorry

end NUMINAMATH_CALUDE_min_shots_theorem_l1741_174194


namespace NUMINAMATH_CALUDE_fraction_problem_l1741_174118

theorem fraction_problem (f : ℚ) : f * 16 + 5 = 13 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1741_174118


namespace NUMINAMATH_CALUDE_james_initial_balance_l1741_174103

def ticket_cost_1 : ℚ := 150
def ticket_cost_2 : ℚ := 150
def ticket_cost_3 : ℚ := ticket_cost_1 / 3
def total_cost : ℚ := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def james_share : ℚ := total_cost / 2
def remaining_balance : ℚ := 325

theorem james_initial_balance :
  ∀ x : ℚ, x - james_share = remaining_balance → x = 500 :=
by sorry

end NUMINAMATH_CALUDE_james_initial_balance_l1741_174103


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1741_174175

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1741_174175


namespace NUMINAMATH_CALUDE_marcus_bird_count_l1741_174172

theorem marcus_bird_count (humphrey_count darrel_count average_count : ℕ) 
  (h1 : humphrey_count = 11)
  (h2 : darrel_count = 9)
  (h3 : average_count = 9)
  (h4 : (humphrey_count + darrel_count + marcus_count) / 3 = average_count) :
  marcus_count = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_marcus_bird_count_l1741_174172


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1741_174193

/-- Given a cubic polynomial g(x) = cx³ + 5x² + dx + 7, prove that if g(2) = 19 and g(-1) = -9, 
    then c = -25/3 and d = 88/3 -/
theorem polynomial_remainder (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 + 5 * x^2 + d * x + 7
  (g 2 = 19 ∧ g (-1) = -9) → c = -25/3 ∧ d = 88/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1741_174193


namespace NUMINAMATH_CALUDE_gabrielle_cardinals_count_l1741_174139

/-- Represents the number of birds of each type seen by a person -/
structure BirdCount where
  robins : ℕ
  cardinals : ℕ
  bluejays : ℕ

/-- Calculates the total number of birds seen -/
def total_birds (count : BirdCount) : ℕ :=
  count.robins + count.cardinals + count.bluejays

/-- The bird counts for Chase and Gabrielle -/
def chase : BirdCount := { robins := 2, cardinals := 5, bluejays := 3 }
def gabrielle : BirdCount := { robins := 5, cardinals := 0, bluejays := 3 }

theorem gabrielle_cardinals_count : gabrielle.cardinals = 4 := by
  have h1 : total_birds chase = 10 := by sorry
  have h2 : total_birds gabrielle = (120 * total_birds chase) / 100 := by sorry
  have h3 : gabrielle.robins + gabrielle.bluejays = 8 := by sorry
  sorry

end NUMINAMATH_CALUDE_gabrielle_cardinals_count_l1741_174139


namespace NUMINAMATH_CALUDE_point_q_coordinates_l1741_174195

/-- Given two points P and Q in a 2D Cartesian coordinate system, prove that Q has coordinates (1, -3) -/
theorem point_q_coordinates
  (P Q : ℝ × ℝ) -- P and Q are points in 2D space
  (h_P : P = (1, 2)) -- P has coordinates (1, 2)
  (h_Q_below : Q.2 < 0) -- Q is below the x-axis
  (h_parallel : P.1 = Q.1) -- PQ is parallel to the y-axis
  (h_distance : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5) -- PQ = 5
  : Q = (1, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_q_coordinates_l1741_174195


namespace NUMINAMATH_CALUDE_remainder_of_large_sum_l1741_174176

theorem remainder_of_large_sum (n : ℕ) : (7 * 10^20 + 2^20) % 11 = 9 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_large_sum_l1741_174176


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1741_174180

theorem rectangular_to_polar_conversion :
  ∀ (x y : ℝ), x = 6 ∧ y = 2 * Real.sqrt 3 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 4 * Real.sqrt 3 ∧ θ = Real.pi / 6 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1741_174180


namespace NUMINAMATH_CALUDE_polynomial_integer_coefficients_l1741_174196

theorem polynomial_integer_coefficients (a b c : ℚ) : 
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) → 
  (∃ (a' b' c' : ℤ), a = a' ∧ b = b' ∧ c = c') := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_coefficients_l1741_174196


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l1741_174164

theorem triangle_abc_theorem (a b c : ℝ) (A B C : ℝ) :
  a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A →
  Real.cos A = 1 / 3 →
  B = π / 6 ∧ Real.sin C = (2 * Real.sqrt 6 + 1) / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l1741_174164


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_existence_l1741_174155

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ c > 0, p c) ↔ (∀ c > 0, ¬p c) :=
by sorry

def has_solution (c : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + c = 0

theorem negation_of_quadratic_existence :
  (¬∃ c > 0, has_solution c) ↔ (∀ c > 0, ¬has_solution c) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_existence_l1741_174155


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_numerator_denominator_l1741_174179

def repeating_decimal : ℚ := 345 / 999

theorem repeating_decimal_fraction :
  repeating_decimal = 115 / 111 :=
sorry

theorem sum_numerator_denominator :
  115 + 111 = 226 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_numerator_denominator_l1741_174179


namespace NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_four_l1741_174143

theorem smallest_addition_for_multiple_of_four : 
  (∃ n : ℕ+, 4 ∣ (587 + n) ∧ ∀ m : ℕ+, 4 ∣ (587 + m) → n ≤ m) ∧ 
  (∀ n : ℕ+, (4 ∣ (587 + n) ∧ ∀ m : ℕ+, 4 ∣ (587 + m) → n ≤ m) → n = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_four_l1741_174143


namespace NUMINAMATH_CALUDE_hide_and_seek_l1741_174148

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem to prove
theorem hide_and_seek :
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_l1741_174148


namespace NUMINAMATH_CALUDE_margaret_mean_score_l1741_174102

def scores : List ℕ := [85, 88, 90, 92, 94, 96, 100]

def cyprian_score_count : ℕ := 4
def margaret_score_count : ℕ := 3
def cyprian_mean : ℚ := 92

theorem margaret_mean_score (h1 : scores.length = cyprian_score_count + margaret_score_count)
  (h2 : cyprian_mean = (scores.sum - (scores.sum - cyprian_mean * cyprian_score_count)) / cyprian_score_count) :
  (scores.sum - cyprian_mean * cyprian_score_count) / margaret_score_count = 92.33 := by
  sorry

end NUMINAMATH_CALUDE_margaret_mean_score_l1741_174102


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1741_174122

theorem quadratic_equations_solutions :
  (∀ x, 2 * x^2 - 4 * x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x, x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1741_174122


namespace NUMINAMATH_CALUDE_quadratic_minimum_less_than_neg_six_l1741_174130

/-- A quadratic function satisfying specific point conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  f (-2) = 6 ∧ f 0 = -4 ∧ f 1 = -6 ∧ f 3 = -4

/-- The theorem stating that the minimum value of the quadratic function is less than -6 -/
theorem quadratic_minimum_less_than_neg_six (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x < -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_less_than_neg_six_l1741_174130


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1741_174163

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1741_174163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1741_174157

theorem arithmetic_sequence_sum (a : ℝ) :
  (a + 6 * 2 = 20) →  -- seventh term is 20
  (a + 2 + a = 18)    -- sum of first two terms is 18
:= by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1741_174157


namespace NUMINAMATH_CALUDE_parallel_lines_iff_coplanar_l1741_174159

-- Define the types for points and planes
variable (Point Plane : Type*)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the "on_plane" relation for points and planes
variable (on_plane : Point → Plane → Prop)

-- Define the parallel relation for lines (represented by two points each)
variable (parallel_lines : (Point × Point) → (Point × Point) → Prop)

-- Define the coplanar relation for four points
variable (coplanar : Point → Point → Point → Point → Prop)

-- State the theorem
theorem parallel_lines_iff_coplanar
  (α β : Plane) (A B C D : Point)
  (h_planes_parallel : parallel_planes α β)
  (h_A_on_α : on_plane A α)
  (h_C_on_α : on_plane C α)
  (h_B_on_β : on_plane B β)
  (h_D_on_β : on_plane D β) :
  parallel_lines (A, C) (B, D) ↔ coplanar A B C D :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_iff_coplanar_l1741_174159


namespace NUMINAMATH_CALUDE_max_slope_product_l1741_174181

theorem max_slope_product (m₁ m₂ : ℝ) : 
  m₁ ≠ 0 → m₂ ≠ 0 →                           -- non-horizontal, non-vertical lines
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 →         -- 45° angle intersection
  m₂ = 6 * m₁ →                               -- one slope is 6 times the other
  ∃ (p : ℝ), p = m₁ * m₂ ∧ p ≤ (3/2 : ℝ) ∧ 
  ∀ (q : ℝ), (∃ (n₁ n₂ : ℝ), n₁ ≠ 0 ∧ n₂ ≠ 0 ∧ 
              |((n₂ - n₁) / (1 + n₁ * n₂))| = 1 ∧ 
              n₂ = 6 * n₁ ∧ q = n₁ * n₂) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_max_slope_product_l1741_174181


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1741_174151

theorem polynomial_evaluation : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1741_174151


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1741_174123

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1741_174123


namespace NUMINAMATH_CALUDE_min_value_expression_l1741_174182

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 * a / (b * c^2 + b)) + (1 / (a * b * c^2 + a * b)) + 3 * c^2 ≥ 6 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1741_174182


namespace NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l1741_174110

def A : ℝ × ℝ := (5, 1)
def B : ℝ × ℝ := (7, -3)
def C : ℝ × ℝ := (2, -8)

def circumcircle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 = 25

theorem circumcircle_of_triangle_ABC :
  circumcircle_equation A.1 A.2 ∧
  circumcircle_equation B.1 B.2 ∧
  circumcircle_equation C.1 C.2 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_of_triangle_ABC_l1741_174110


namespace NUMINAMATH_CALUDE_base_10_to_6_conversion_l1741_174138

/-- Converts a base-10 number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- Converts a list of digits in base-6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 6 * acc + d) 0

theorem base_10_to_6_conversion :
  fromBase6 (toBase6 110) = 110 :=
sorry

end NUMINAMATH_CALUDE_base_10_to_6_conversion_l1741_174138


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1741_174185

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - 1/13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1741_174185


namespace NUMINAMATH_CALUDE_product_of_powers_l1741_174116

theorem product_of_powers (x y : ℝ) : -x^2 * y^3 * (2 * x * y^2) = -2 * x^3 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l1741_174116


namespace NUMINAMATH_CALUDE_average_score_is_106_l1741_174149

/-- The average bowling score of three people -/
def average_bowling_score (score1 score2 score3 : ℕ) : ℚ :=
  (score1 + score2 + score3 : ℚ) / 3

/-- Theorem: The average bowling score of Gretchen, Mitzi, and Beth is 106 -/
theorem average_score_is_106 :
  average_bowling_score 120 113 85 = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_106_l1741_174149


namespace NUMINAMATH_CALUDE_volunteer_distribution_l1741_174108

/-- The number of ways to distribute volunteers to pavilions -/
def distribute_volunteers (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- Two specific volunteers cannot be in the same pavilion -/
def separate_volunteers (n : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution :
  distribute_volunteers 5 3 2 - separate_volunteers 3 = 114 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l1741_174108


namespace NUMINAMATH_CALUDE_circ_equation_solutions_l1741_174114

/-- Custom operation ∘ -/
def circ (a b : ℤ) : ℤ := a + b - a * b

/-- Theorem statement -/
theorem circ_equation_solutions :
  ∀ x y z : ℤ, circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0 ↔
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = 0 ∧ y = 2 ∧ z = 2) ∨
     (x = 2 ∧ y = 0 ∧ z = 2) ∨
     (x = 2 ∧ y = 2 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circ_equation_solutions_l1741_174114


namespace NUMINAMATH_CALUDE_prime_divisors_50_factorial_l1741_174105

/-- The number of prime divisors of 50! -/
def num_prime_divisors_50_factorial : ℕ := sorry

/-- Theorem stating that the number of prime divisors of 50! is 15 -/
theorem prime_divisors_50_factorial :
  num_prime_divisors_50_factorial = 15 := by sorry

end NUMINAMATH_CALUDE_prime_divisors_50_factorial_l1741_174105


namespace NUMINAMATH_CALUDE_total_puppies_l1741_174121

theorem total_puppies (female_puppies male_puppies : ℕ) 
  (h1 : female_puppies = 2)
  (h2 : male_puppies = 10)
  (h3 : (female_puppies : ℚ) / male_puppies = 0.2) :
  female_puppies + male_puppies = 12 := by
sorry

end NUMINAMATH_CALUDE_total_puppies_l1741_174121


namespace NUMINAMATH_CALUDE_probability_age_21_to_30_l1741_174183

theorem probability_age_21_to_30 (total_people : ℕ) (people_21_to_30 : ℕ) 
  (h1 : total_people = 160) (h2 : people_21_to_30 = 70) : 
  (people_21_to_30 : ℚ) / total_people = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_age_21_to_30_l1741_174183


namespace NUMINAMATH_CALUDE_fraction_simplification_l1741_174166

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1741_174166


namespace NUMINAMATH_CALUDE_lance_penny_savings_l1741_174126

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Lance's penny savings problem -/
theorem lance_penny_savings :
  arithmetic_sum 5 2 20 = 480 := by
  sorry

end NUMINAMATH_CALUDE_lance_penny_savings_l1741_174126


namespace NUMINAMATH_CALUDE_new_function_not_transformation_of_original_l1741_174152

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the new quadratic function
def new_function (x : ℝ) : ℝ := x^2

-- Define a general quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem statement
theorem new_function_not_transformation_of_original :
  ¬∃ (h k : ℝ), ∀ x, new_function x = original_function (x - h) + k ∧
  ¬∃ (h k : ℝ), ∀ x, new_function x = original_function (-(x - h)) + k :=
by sorry

end NUMINAMATH_CALUDE_new_function_not_transformation_of_original_l1741_174152


namespace NUMINAMATH_CALUDE_representative_count_l1741_174186

/-- The number of ways to choose a math class representative from a class with a given number of boys and girls. -/
def choose_representative (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  num_boys + num_girls

/-- Theorem: The number of ways to choose a math class representative from a class with 26 boys and 24 girls is 50. -/
theorem representative_count : choose_representative 26 24 = 50 := by
  sorry

end NUMINAMATH_CALUDE_representative_count_l1741_174186


namespace NUMINAMATH_CALUDE_odd_function_implies_a_equals_two_l1741_174109

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a

theorem odd_function_implies_a_equals_two (a : ℝ) :
  (∀ x, f a (x + 1) = -f a (-x + 1)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_equals_two_l1741_174109


namespace NUMINAMATH_CALUDE_john_leftover_percentage_l1741_174147

/-- The percentage of earnings John spent on rent -/
def rent_percentage : ℝ := 40

/-- The percentage less than rent that John spent on the dishwasher -/
def dishwasher_percentage_less : ℝ := 30

/-- Theorem stating that the percentage of John's earnings left over is 48% -/
theorem john_leftover_percentage : 
  100 - (rent_percentage + (100 - dishwasher_percentage_less) / 100 * rent_percentage) = 48 := by
  sorry

end NUMINAMATH_CALUDE_john_leftover_percentage_l1741_174147


namespace NUMINAMATH_CALUDE_senior_score_is_140_8_l1741_174111

/-- Represents the AHSME exam results at Century High School -/
structure AHSMEResults where
  total_students : ℕ
  average_score : ℝ
  senior_ratio : ℝ
  senior_score_ratio : ℝ

/-- Calculates the mean score of seniors given AHSME results -/
def senior_mean_score (results : AHSMEResults) : ℝ :=
  sorry

/-- Theorem stating that the mean score of seniors is 140.8 -/
theorem senior_score_is_140_8 (results : AHSMEResults)
  (h1 : results.total_students = 120)
  (h2 : results.average_score = 110)
  (h3 : results.senior_ratio = 1 / 1.4)
  (h4 : results.senior_score_ratio = 1.6) :
  senior_mean_score results = 140.8 := by
  sorry

end NUMINAMATH_CALUDE_senior_score_is_140_8_l1741_174111


namespace NUMINAMATH_CALUDE_vanishing_function_l1741_174162

theorem vanishing_function (g : ℝ → ℝ) (h₁ : Continuous (deriv g)) 
  (h₂ : g 0 = 0) (h₃ : ∀ x, |deriv g x| ≤ |g x|) : 
  ∀ x, g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_vanishing_function_l1741_174162


namespace NUMINAMATH_CALUDE_triangle_inequality_l1741_174144

theorem triangle_inequality (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : 5 * a * b * c > a^3 + b^3 + c^3) :
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1741_174144


namespace NUMINAMATH_CALUDE_bluegrass_percentage_in_mixtureX_l1741_174174

-- Define the seed mixtures and their compositions
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

-- Define the given conditions
def mixtureX : SeedMixture := { ryegrass := 40, bluegrass := 0, fescue := 0 }
def mixtureY : SeedMixture := { ryegrass := 25, bluegrass := 0, fescue := 75 }

-- Define the combined mixture
def combinedMixture : SeedMixture := { ryegrass := 38, bluegrass := 0, fescue := 0 }

-- Weight percentage of mixture X in the combined mixture
def weightPercentageX : ℝ := 86.67

-- Theorem to prove
theorem bluegrass_percentage_in_mixtureX : mixtureX.bluegrass = 60 := by
  sorry

end NUMINAMATH_CALUDE_bluegrass_percentage_in_mixtureX_l1741_174174


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1741_174146

/-- Represents the roots of the quadratic equation x^2 - 8x + 15 = 0 --/
def roots : Set ℝ := {x : ℝ | x^2 - 8*x + 15 = 0}

/-- Represents an isosceles triangle with side lengths from the roots --/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  h_roots : side1 ∈ roots ∧ side2 ∈ roots
  h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base

/-- The perimeter of an isosceles triangle --/
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.base

/-- Theorem stating that the perimeter of the isosceles triangle is either 11 or 13 --/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  perimeter t = 11 ∨ perimeter t = 13 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1741_174146


namespace NUMINAMATH_CALUDE_lunch_pizzas_calculation_l1741_174107

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem lunch_pizzas_calculation :
  total_pizzas - dinner_pizzas = 9 := by
  sorry

end NUMINAMATH_CALUDE_lunch_pizzas_calculation_l1741_174107


namespace NUMINAMATH_CALUDE_probability_of_two_boys_l1741_174169

theorem probability_of_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = 15)
  (h2 : boys = 9)
  (h3 : girls = 6)
  (h4 : total = boys + girls) :
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ) = 12 / 35 := by
sorry

end NUMINAMATH_CALUDE_probability_of_two_boys_l1741_174169


namespace NUMINAMATH_CALUDE_emma_toast_pieces_l1741_174154

/-- Given a loaf of bread with an initial number of slices, 
    calculate the number of toast pieces that can be made 
    after some slices are eaten and leaving one slice remaining. --/
def toastPieces (initialSlices : ℕ) (eatenSlices : ℕ) (slicesPerToast : ℕ) : ℕ :=
  ((initialSlices - eatenSlices - 1) / slicesPerToast : ℕ)

/-- Theorem stating that given the specific conditions of the problem,
    the number of toast pieces made is 10. --/
theorem emma_toast_pieces : 
  toastPieces 27 6 2 = 10 := by sorry

end NUMINAMATH_CALUDE_emma_toast_pieces_l1741_174154


namespace NUMINAMATH_CALUDE_trapezium_area_l1741_174125

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) :
  (1/2 : ℝ) * (a + b) * h = 27 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_l1741_174125


namespace NUMINAMATH_CALUDE_circle_line_distance_l1741_174101

theorem circle_line_distance (M : ℝ × ℝ) :
  (M.1 - 5)^2 + (M.2 - 3)^2 = 9 →
  (∃ d : ℝ, d = |3 * M.1 + 4 * M.2 - 2| / (3^2 + 4^2).sqrt ∧ d = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l1741_174101


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l1741_174187

theorem product_of_five_consecutive_integers_not_square (n : ℕ+) :
  ¬∃ k : ℕ, (n : ℕ) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2 :=
sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_not_square_l1741_174187


namespace NUMINAMATH_CALUDE_dessert_preference_l1741_174167

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) : 
  total = 50 → apple = 22 → chocolate = 20 → neither = 17 →
  apple + chocolate - (total - neither) = 9 := by
sorry

end NUMINAMATH_CALUDE_dessert_preference_l1741_174167


namespace NUMINAMATH_CALUDE_turtle_marathon_time_l1741_174168

/-- The time taken by a turtle to complete a marathon -/
theorem turtle_marathon_time (turtle_speed : ℝ) (marathon_distance : ℝ) :
  turtle_speed = 15 →
  marathon_distance = 42195 →
  ∃ (days hours minutes : ℕ),
    (days = 1 ∧ hours = 22 ∧ minutes = 53) ∧
    (days * 24 * 60 + hours * 60 + minutes : ℝ) * turtle_speed = marathon_distance :=
by sorry

end NUMINAMATH_CALUDE_turtle_marathon_time_l1741_174168


namespace NUMINAMATH_CALUDE_washing_machine_cycle_time_l1741_174128

theorem washing_machine_cycle_time 
  (total_items : ℕ) 
  (machine_capacity : ℕ) 
  (total_wash_time_minutes : ℕ) 
  (h1 : total_items = 60)
  (h2 : machine_capacity = 15)
  (h3 : total_wash_time_minutes = 180) :
  total_wash_time_minutes / (total_items / machine_capacity) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_washing_machine_cycle_time_l1741_174128


namespace NUMINAMATH_CALUDE_possible_values_of_a_plus_b_l1741_174158

theorem possible_values_of_a_plus_b (a b : ℝ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 1) 
  (h3 : a - b < 0) : 
  a + b = -6 ∨ a + b = -4 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_plus_b_l1741_174158


namespace NUMINAMATH_CALUDE_area_36_implies_a_plus_b_6_l1741_174165

/-- A quadrilateral with vertices defined by a positive integer a -/
structure Quadrilateral (a : ℕ+) where
  P : ℝ × ℝ := (a, a)
  Q : ℝ × ℝ := (a, -a)
  R : ℝ × ℝ := (-a, -a)
  S : ℝ × ℝ := (-a, a)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral a) : ℝ := sorry

/-- Theorem: If the area of quadrilateral PQRS is 36, then a+b = 6 -/
theorem area_36_implies_a_plus_b_6 (a b : ℕ+) (q : Quadrilateral a) :
  area q = 36 → a + b = 6 := by sorry

end NUMINAMATH_CALUDE_area_36_implies_a_plus_b_6_l1741_174165


namespace NUMINAMATH_CALUDE_percentage_men_not_speaking_french_or_spanish_l1741_174199

theorem percentage_men_not_speaking_french_or_spanish :
  let total_men_percentage : ℚ := 100
  let french_speaking_men_percentage : ℚ := 55
  let spanish_speaking_men_percentage : ℚ := 35
  let other_languages_men_percentage : ℚ := 10
  (total_men_percentage = french_speaking_men_percentage + spanish_speaking_men_percentage + other_languages_men_percentage) →
  (other_languages_men_percentage = 10) :=
by sorry

end NUMINAMATH_CALUDE_percentage_men_not_speaking_french_or_spanish_l1741_174199


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1741_174192

/-- Given a quadratic function f(x) = ax² + bx + c with specific properties, 
    prove that a > 0 and 4a + b = 0 -/
theorem quadratic_function_property (a b c : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  (f 0 = f 4) ∧ (f 0 > f 1) → a > 0 ∧ 4 * a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1741_174192


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l1741_174104

theorem right_triangle_leg_square (a b c : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l1741_174104


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l1741_174150

theorem no_linear_term_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ∃ b c d : ℝ, (x^2 + a*x - 2) * (x - 1) = x^3 + b*x^2 + d) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l1741_174150


namespace NUMINAMATH_CALUDE_area_at_stage_8_l1741_174171

/-- The area of a rectangle formed by adding squares -/
def rectangleArea (numSquares : ℕ) (squareSide : ℕ) : ℕ :=
  numSquares * (squareSide * squareSide)

/-- Theorem: The area of a rectangle formed by adding 8 squares, each 4 inches by 4 inches, is 128 square inches -/
theorem area_at_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l1741_174171


namespace NUMINAMATH_CALUDE_points_on_line_l1741_174178

/-- Given a line with equation x = 2y + 5, prove that for any real number n,
    the points (m, n) and (m + 1, n + 0.5) lie on this line, where m = 2n + 5. -/
theorem points_on_line (n : ℝ) : 
  let m : ℝ := 2 * n + 5
  let point1 : ℝ × ℝ := (m, n)
  let point2 : ℝ × ℝ := (m + 1, n + 0.5)
  (point1.1 = 2 * point1.2 + 5) ∧ (point2.1 = 2 * point2.2 + 5) :=
by
  sorry


end NUMINAMATH_CALUDE_points_on_line_l1741_174178


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1741_174173

theorem max_value_of_expression (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : 2*a + 3*b = 5) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2*x + 3*y = 5 → (2*x + 2)*(3*y + 1) ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1741_174173


namespace NUMINAMATH_CALUDE_max_dimes_count_l1741_174153

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.1

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The total amount of money Sasha has in dollars -/
def total_money : ℚ := 3.5

/-- Theorem: Given $3.50 in coins and an equal number of dimes and pennies, 
    the maximum number of dimes possible is 31 -/
theorem max_dimes_count : 
  ∃ (d : ℕ), d ≤ 31 ∧ 
  ∀ (n : ℕ), n * (dime_value + penny_value) ≤ total_money → n ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_dimes_count_l1741_174153


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_denominator_l1741_174177

theorem simplify_and_rationalize_denominator :
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_denominator_l1741_174177


namespace NUMINAMATH_CALUDE_not_necessarily_no_mass_infection_l1741_174120

/-- Represents the daily increase in suspected cases over 10 days -/
def DailyIncrease := Fin 10 → ℕ

/-- The sign of no mass infection -/
def NoMassInfection (d : DailyIncrease) : Prop :=
  ∀ i, d i ≤ 7

/-- The median of a DailyIncrease is 2 -/
def MedianIsTwo (d : DailyIncrease) : Prop :=
  ∃ (sorted : Fin 10 → ℕ), (∀ i j, i ≤ j → sorted i ≤ sorted j) ∧
    (∀ i, ∃ j, d j = sorted i) ∧
    sorted 4 = 2 ∧ sorted 5 = 2

/-- The mode of a DailyIncrease is 3 -/
def ModeIsThree (d : DailyIncrease) : Prop :=
  ∃ (count : ℕ → ℕ), (∀ n, count n = (Finset.univ.filter (λ i => d i = n)).card) ∧
    ∀ n, count 3 ≥ count n

theorem not_necessarily_no_mass_infection :
  ∃ d : DailyIncrease, MedianIsTwo d ∧ ModeIsThree d ∧ ¬NoMassInfection d :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_no_mass_infection_l1741_174120


namespace NUMINAMATH_CALUDE_prime_sum_special_equation_l1741_174161

theorem prime_sum_special_equation (p q : ℕ) : 
  Prime p → Prime q → q^5 - 2*p^2 = 1 → p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_special_equation_l1741_174161


namespace NUMINAMATH_CALUDE_f_iter_formula_l1741_174135

def f (x : ℝ) := 3 * x + 2

def f_iter : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ f_iter n

theorem f_iter_formula (n : ℕ) (x : ℝ) : 
  f_iter n x = 3^n * x + 3^n - 1 := by sorry

end NUMINAMATH_CALUDE_f_iter_formula_l1741_174135


namespace NUMINAMATH_CALUDE_isabellas_hourly_rate_l1741_174137

/-- Calculates Isabella's hourly rate given her work schedule and total earnings --/
theorem isabellas_hourly_rate 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (total_weeks : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hours_per_day = 5)
  (h2 : days_per_week = 6)
  (h3 : total_weeks = 7)
  (h4 : total_earnings = 1050) :
  total_earnings / (hours_per_day * days_per_week * total_weeks) = 5 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hourly_rate_l1741_174137


namespace NUMINAMATH_CALUDE_function_value_at_two_l1741_174136

theorem function_value_at_two
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x + 2 * f (1 / x) = 2 * x + 1) :
  f 2 = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1741_174136


namespace NUMINAMATH_CALUDE_megans_earnings_l1741_174100

/-- Calculates the total earnings for a worker given their work schedule and hourly rate. -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (num_months : ℕ) : ℚ :=
  (hours_per_day : ℚ) * hourly_rate * (days_per_month : ℚ) * (num_months : ℚ)

/-- Proves that Megan's total earnings for two months of work is $2400. -/
theorem megans_earnings :
  let hours_per_day : ℕ := 8
  let hourly_rate : ℚ := 15/2  -- $7.50 expressed as a rational number
  let days_per_month : ℕ := 20
  let num_months : ℕ := 2
  total_earnings hours_per_day hourly_rate days_per_month num_months = 2400 := by
  sorry


end NUMINAMATH_CALUDE_megans_earnings_l1741_174100


namespace NUMINAMATH_CALUDE_function_equation_solution_l1741_174133

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) = f x * y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1741_174133


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1741_174184

-- Define the propositions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q : 
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1741_174184


namespace NUMINAMATH_CALUDE_floor_divisibility_l1741_174115

theorem floor_divisibility (n : ℕ) : 
  ∃ k : ℤ, (⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ = 2^(n+1) * k) ∧ 
           ¬∃ m : ℤ, (⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ = 2^(n+2) * m) :=
by sorry

end NUMINAMATH_CALUDE_floor_divisibility_l1741_174115


namespace NUMINAMATH_CALUDE_trapezoid_bc_length_l1741_174189

-- Define the trapezoid and its properties
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab_length : ℝ
  cd_length : ℝ
  ad_cd_angle : ℝ

-- Define the theorem
theorem trapezoid_bc_length (t : Trapezoid) 
  (h1 : t.area = 200)
  (h2 : t.altitude = 10)
  (h3 : t.ab_length = 15)
  (h4 : t.cd_length = 25)
  (h5 : t.ad_cd_angle = π/4)
  : ∃ (bc_length : ℝ), bc_length = (200 - (25 * Real.sqrt 5 + 25 * Real.sqrt 21)) / 10 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_bc_length_l1741_174189


namespace NUMINAMATH_CALUDE_volunteer_selection_probability_l1741_174119

theorem volunteer_selection_probability 
  (total_students : ℕ) 
  (eliminated : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = 2018) 
  (h2 : eliminated = 18) 
  (h3 : selected = 50) :
  (selected : ℚ) / total_students = 25 / 1009 := by
sorry

end NUMINAMATH_CALUDE_volunteer_selection_probability_l1741_174119


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l1741_174131

theorem complex_modulus_equation : ∃ (t : ℝ), t > 0 ∧ Complex.abs (3 - 3 + t * Complex.I) = 5 ∧ t = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l1741_174131


namespace NUMINAMATH_CALUDE_next_square_number_proof_l1741_174145

/-- The next square number after 4356 composed of four consecutive digits -/
def next_square_number : ℕ := 5476

/-- The square root of the next square number -/
def square_root : ℕ := 74

/-- Predicate to check if a number is composed of four consecutive digits -/
def is_composed_of_four_consecutive_digits (n : ℕ) : Prop :=
  ∃ (a : ℕ), a > 0 ∧ a < 7 ∧
  (n = a * 1000 + (a + 1) * 100 + (a + 2) * 10 + (a + 3) ∨
   n = a * 1000 + (a + 1) * 100 + (a + 3) * 10 + (a + 2) ∨
   n = a * 1000 + (a + 2) * 100 + (a + 1) * 10 + (a + 3) ∨
   n = a * 1000 + (a + 2) * 100 + (a + 3) * 10 + (a + 1) ∨
   n = a * 1000 + (a + 3) * 100 + (a + 1) * 10 + (a + 2) ∨
   n = a * 1000 + (a + 3) * 100 + (a + 2) * 10 + (a + 1))

theorem next_square_number_proof :
  next_square_number = square_root ^ 2 ∧
  is_composed_of_four_consecutive_digits next_square_number ∧
  ∀ (n : ℕ), 4356 < n ∧ n < next_square_number →
    ¬(∃ (m : ℕ), n = m ^ 2 ∧ is_composed_of_four_consecutive_digits n) :=
by sorry

end NUMINAMATH_CALUDE_next_square_number_proof_l1741_174145


namespace NUMINAMATH_CALUDE_exam_score_problem_l1741_174156

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 110) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 34 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1741_174156


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1741_174127

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1741_174127


namespace NUMINAMATH_CALUDE_line_through_point_l1741_174191

/-- Given a line with equation 3kx - 2 = 4y passing through the point (-1/2, -5),
    prove that k = 12. -/
theorem line_through_point (k : ℝ) : 
  (3 * k * (-1/2) - 2 = 4 * (-5)) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1741_174191


namespace NUMINAMATH_CALUDE_turnip_solution_l1741_174134

/-- The number of turnips grown by Melanie, Benny, and Caroline, and the difference between
    the combined turnips of Melanie and Benny versus Caroline's turnips. -/
def turnip_problem (melanie_turnips benny_turnips caroline_turnips : ℕ) : Prop :=
  let combined_turnips := melanie_turnips + benny_turnips
  combined_turnips - caroline_turnips = 80

/-- The theorem stating the solution to the turnip problem -/
theorem turnip_solution : turnip_problem 139 113 172 := by
  sorry

end NUMINAMATH_CALUDE_turnip_solution_l1741_174134


namespace NUMINAMATH_CALUDE_root_product_reciprocal_sum_l1741_174117

theorem root_product_reciprocal_sum (p q : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + p*x + q = (x - x1) * (x - x2))
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ x^2 + q*x + p = (x - x3) * (x - x4))
  (h3 : ∀ x1 x2 x3 x4 : ℝ, 
    (x^2 + p*x + q = (x - x1) * (x - x2)) → 
    (x^2 + q*x + p = (x - x3) * (x - x4)) → 
    x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4) :
  ∃ x1 x2 x3 x4 : ℝ, 
    (x^2 + p*x + q = (x - x1) * (x - x2)) ∧
    (x^2 + q*x + p = (x - x3) * (x - x4)) ∧
    1 / (x1 * x3) + 1 / (x1 * x4) + 1 / (x2 * x3) + 1 / (x2 * x4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_product_reciprocal_sum_l1741_174117


namespace NUMINAMATH_CALUDE_textbook_order_solution_l1741_174106

/-- Represents the textbook order problem -/
structure TextbookOrder where
  red_cost : ℝ
  trad_cost : ℝ
  red_price_ratio : ℝ
  quantity_diff : ℕ
  total_quantity : ℕ
  max_trad_quantity : ℕ
  max_total_cost : ℝ

/-- Theorem stating the solution to the textbook order problem -/
theorem textbook_order_solution (order : TextbookOrder)
  (h1 : order.red_cost = 14000)
  (h2 : order.trad_cost = 7000)
  (h3 : order.red_price_ratio = 1.4)
  (h4 : order.quantity_diff = 300)
  (h5 : order.total_quantity = 1000)
  (h6 : order.max_trad_quantity = 400)
  (h7 : order.max_total_cost = 12880) :
  ∃ (red_price trad_price min_cost : ℝ),
    red_price = 14 ∧
    trad_price = 10 ∧
    min_cost = 12400 ∧
    red_price = order.red_price_ratio * trad_price ∧
    order.red_cost / red_price - order.trad_cost / trad_price = order.quantity_diff ∧
    min_cost ≤ order.max_total_cost ∧
    (∀ (trad_quantity : ℕ),
      trad_quantity ≤ order.max_trad_quantity →
      trad_quantity * trad_price + (order.total_quantity - trad_quantity) * red_price ≥ min_cost) :=
by sorry

end NUMINAMATH_CALUDE_textbook_order_solution_l1741_174106


namespace NUMINAMATH_CALUDE_additional_push_ups_l1741_174132

def push_ups (x : ℕ) : ℕ → ℕ
  | 1 => 10
  | 2 => 10 + x
  | 3 => 10 + 2*x
  | _ => 0

theorem additional_push_ups :
  ∃ x : ℕ, (push_ups x 1 + push_ups x 2 + push_ups x 3 = 45) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_additional_push_ups_l1741_174132


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l1741_174124

theorem solution_set_reciprocal_gt_one :
  {x : ℝ | (1 : ℝ) / x > 1} = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l1741_174124


namespace NUMINAMATH_CALUDE_rectangular_field_width_l1741_174142

theorem rectangular_field_width (width length : ℝ) (h1 : length = (7/5) * width) (h2 : 2 * (length + width) = 432) : width = 90 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l1741_174142


namespace NUMINAMATH_CALUDE_prob_red_given_red_half_l1741_174140

/-- A bag with red and yellow balls -/
structure Bag where
  total : ℕ
  red : ℕ
  yellow : ℕ
  h_total : total = red + yellow

/-- The probability of drawing a red ball in the second draw given a red ball in the first draw -/
def prob_red_given_red (b : Bag) : ℚ :=
  (b.red - 1) / (b.total - 1)

/-- The theorem stating the probability is 1/2 for the given bag -/
theorem prob_red_given_red_half (b : Bag) 
  (h_total : b.total = 5)
  (h_red : b.red = 3)
  (h_yellow : b.yellow = 2) : 
  prob_red_given_red b = 1/2 := by
sorry

end NUMINAMATH_CALUDE_prob_red_given_red_half_l1741_174140


namespace NUMINAMATH_CALUDE_trig_identity_l1741_174129

theorem trig_identity : Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1741_174129


namespace NUMINAMATH_CALUDE_x_squared_plus_one_is_quadratic_l1741_174113

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that x² + 1 = 0 is a quadratic equation in one variable x -/
theorem x_squared_plus_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_one_is_quadratic_l1741_174113


namespace NUMINAMATH_CALUDE_kamal_english_marks_l1741_174141

/-- Represents the marks of a student in various subjects -/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of marks -/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology) / 5

theorem kamal_english_marks :
  ∃ (m : Marks),
    m.mathematics = 60 ∧
    m.physics = 82 ∧
    m.chemistry = 67 ∧
    m.biology = 85 ∧
    average m = 74 ∧
    m.english = 76 := by
  sorry

end NUMINAMATH_CALUDE_kamal_english_marks_l1741_174141


namespace NUMINAMATH_CALUDE_min_value_inequality_l1741_174160

theorem min_value_inequality (a b m n : ℝ) : 
  a > 0 → b > 0 → m > 0 → n > 0 → 
  a + b = 1 → m * n = 2 → 
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1741_174160


namespace NUMINAMATH_CALUDE_mary_overtime_rate_increase_l1741_174198

/-- Represents Mary's work schedule and pay structure -/
structure MaryWorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  regularRate : ℚ
  maxEarnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtimeRateIncrease (schedule : MaryWorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularHours * schedule.regularRate
  let overtimeEarnings := schedule.maxEarnings - regularEarnings
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - schedule.regularRate) / schedule.regularRate) * 100

/-- Theorem stating that Mary's overtime rate increase is 25% -/
theorem mary_overtime_rate_increase :
  let schedule := MaryWorkSchedule.mk 70 20 8 660
  overtimeRateIncrease schedule = 25 := by
  sorry

end NUMINAMATH_CALUDE_mary_overtime_rate_increase_l1741_174198


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l1741_174190

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 56

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 2184 :=
by sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l1741_174190


namespace NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_neg_one_l1741_174197

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Theorem statement
theorem intersection_empty_implies_k_leq_neg_one (k : ℝ) : 
  M ∩ N k = ∅ → k ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_neg_one_l1741_174197
