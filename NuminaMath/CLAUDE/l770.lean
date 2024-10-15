import Mathlib

namespace NUMINAMATH_CALUDE_log_sum_equals_two_l770_77046

theorem log_sum_equals_two : 2 * Real.log 63 + Real.log 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l770_77046


namespace NUMINAMATH_CALUDE_profit_margin_ratio_l770_77028

/-- Prove that for an article with selling price S, cost C, and profit margin M = (1/n)S, 
    the ratio of M to C is equal to 1/(n-1) -/
theorem profit_margin_ratio (n : ℝ) (S : ℝ) (C : ℝ) (M : ℝ) 
    (h1 : n ≠ 0) 
    (h2 : n ≠ 1)
    (h3 : M = (1/n) * S) 
    (h4 : C = S - M) : 
  M / C = 1 / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_profit_margin_ratio_l770_77028


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l770_77082

/-- The set of valid n values for which the polynomial decomposition is possible -/
def valid_n : Set ℕ :=
  {0, 1, 3, 7, 15, 12, 18, 25, 37, 51, 75, 151, 246, 493, 987, 1975}

/-- Predicate to check if a list of coefficients is valid for a given n -/
def valid_coefficients (n : ℕ) (coeffs : List ℕ) : Prop :=
  coeffs.length = n ∧
  coeffs.Nodup ∧
  ∀ a ∈ coeffs, 0 < a ∧ a ≤ n

/-- The main theorem stating the condition for valid polynomial decomposition -/
theorem polynomial_decomposition (n : ℕ) :
  (∃ coeffs : List ℕ, valid_coefficients n coeffs) ↔ n ∈ valid_n := by
  sorry

#check polynomial_decomposition

end NUMINAMATH_CALUDE_polynomial_decomposition_l770_77082


namespace NUMINAMATH_CALUDE_range_when_p_and_q_range_when_p_or_q_and_not_p_and_q_l770_77085

-- Define propositions p and q
def p (m : ℝ) : Prop := 2^m > Real.sqrt 2

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m^2 = 0 ∧ x₂^2 - 2*x₂ + m^2 = 0

-- Theorem for the first part
theorem range_when_p_and_q (m : ℝ) :
  p m ∧ q m → m > 1/2 ∧ m < 1 :=
by sorry

-- Theorem for the second part
theorem range_when_p_or_q_and_not_p_and_q (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > -1 ∧ m ≤ 1/2) ∨ m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_when_p_and_q_range_when_p_or_q_and_not_p_and_q_l770_77085


namespace NUMINAMATH_CALUDE_hat_markup_price_l770_77008

theorem hat_markup_price (P : ℝ) 
  (h1 : 2 * P - (P + 0.7 * P) = 6) : 
  P + 0.7 * P = 34 := by
  sorry

end NUMINAMATH_CALUDE_hat_markup_price_l770_77008


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l770_77003

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^4 + 2*x^2 - 3 = (x^2 + 3*x + 2) * q + (-21*x - 21) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l770_77003


namespace NUMINAMATH_CALUDE_expected_informed_after_pairing_l770_77097

/-- Represents the scenario of scientists sharing news during a conference break -/
def ScientistNewsSharing (total : ℕ) (initial_informed : ℕ) : Prop :=
  total = 18 ∧ initial_informed = 10

/-- Calculates the expected number of scientists who know the news after pairing -/
noncomputable def expected_informed (total : ℕ) (initial_informed : ℕ) : ℝ :=
  initial_informed + (total - initial_informed) * (initial_informed / (total - 1))

/-- Theorem stating the expected number of informed scientists after pairing -/
theorem expected_informed_after_pairing {total initial_informed : ℕ} 
  (h : ScientistNewsSharing total initial_informed) :
  expected_informed total initial_informed = 14.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_informed_after_pairing_l770_77097


namespace NUMINAMATH_CALUDE_multiplication_equality_l770_77014

theorem multiplication_equality (x : ℝ) : x * 240 = 173 * 240 ↔ x = 173 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l770_77014


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l770_77052

theorem arithmetic_mean_difference (p q r : ℝ) : 
  (p + q) / 2 = 10 → (q + r) / 2 = 24 → r - p = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l770_77052


namespace NUMINAMATH_CALUDE_cells_covered_by_two_squares_l770_77000

/-- Represents a square on a graph paper --/
structure Square where
  size : ℕ
  position : ℕ × ℕ

/-- Represents the configuration of squares on the graph paper --/
def SquareConfiguration := List Square

/-- Counts the number of cells covered by exactly two squares in a given configuration --/
def countCellsCoveredByTwoSquares (config : SquareConfiguration) : ℕ :=
  sorry

/-- The specific configuration of squares from the problem --/
def problemConfiguration : SquareConfiguration :=
  [{ size := 5, position := (0, 0) },
   { size := 5, position := (3, 0) },
   { size := 5, position := (3, 3) }]

theorem cells_covered_by_two_squares :
  countCellsCoveredByTwoSquares problemConfiguration = 13 := by
  sorry

end NUMINAMATH_CALUDE_cells_covered_by_two_squares_l770_77000


namespace NUMINAMATH_CALUDE_a_not_in_A_l770_77002

def A : Set ℝ := {x | x ≤ 4}

theorem a_not_in_A : 3 * Real.sqrt 3 ∉ A := by sorry

end NUMINAMATH_CALUDE_a_not_in_A_l770_77002


namespace NUMINAMATH_CALUDE_min_tangent_length_is_4_l770_77056

/-- The circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- The line of symmetry for circle C -/
def symmetry_line (a b x y : ℝ) : Prop :=
  2*a*x + b*y + 6 = 0

/-- The point (a, b) -/
structure Point where
  a : ℝ
  b : ℝ

/-- The minimum tangent length from a point to a circle -/
def min_tangent_length (p : Point) (C : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem min_tangent_length_is_4 (a b : ℝ) :
  symmetry_line a b a b →
  min_tangent_length (Point.mk a b) circle_C = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_tangent_length_is_4_l770_77056


namespace NUMINAMATH_CALUDE_product_bounds_l770_77058

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2+Real.sqrt 3)/8 := by
  sorry

end NUMINAMATH_CALUDE_product_bounds_l770_77058


namespace NUMINAMATH_CALUDE_school_club_members_l770_77042

theorem school_club_members :
  ∃! n : ℕ,
    200 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 3 ∧
    n % 8 = 5 ∧
    n % 9 = 7 ∧
    n = 269 := by sorry

end NUMINAMATH_CALUDE_school_club_members_l770_77042


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l770_77051

theorem power_fraction_simplification :
  (1 : ℝ) / ((-5^4)^2) * (-5)^9 = -5 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l770_77051


namespace NUMINAMATH_CALUDE_one_zero_in_interval_l770_77068

def f (x : ℝ) := -x^2 + 8*x - 14

theorem one_zero_in_interval :
  ∃! x, x ∈ Set.Icc 2 5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_one_zero_in_interval_l770_77068


namespace NUMINAMATH_CALUDE_angle_problem_l770_77090

theorem angle_problem (x : ℝ) : 
  x + (3 * x - 10) = 180 → x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l770_77090


namespace NUMINAMATH_CALUDE_expression_evaluation_l770_77091

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x + x * y = 801 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l770_77091


namespace NUMINAMATH_CALUDE_triangle_tan_C_l770_77035

/-- Given a triangle ABC with sides a, b, and c satisfying 3a² + 3b² - 3c² + 2ab = 0,
    prove that tan C = -2√2 -/
theorem triangle_tan_C (a b c : ℝ) (h : 3 * a^2 + 3 * b^2 - 3 * c^2 + 2 * a * b = 0) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  Real.tan C = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_tan_C_l770_77035


namespace NUMINAMATH_CALUDE_negate_200_times_minus_one_l770_77016

/-- Represents the result of negating a number n times -/
def negate_n_times (n : ℕ) : ℤ → ℤ :=
  match n with
  | 0 => id
  | n + 1 => λ x => -(negate_n_times n x)

/-- The theorem states that negating -1 200 times results in -1 -/
theorem negate_200_times_minus_one :
  negate_n_times 200 (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_negate_200_times_minus_one_l770_77016


namespace NUMINAMATH_CALUDE_hat_markup_price_l770_77088

theorem hat_markup_price (P : ℝ) (h : 2 * P - 1.6 * P = 6) : 1.6 * P = 24 := by
  sorry

end NUMINAMATH_CALUDE_hat_markup_price_l770_77088


namespace NUMINAMATH_CALUDE_circle_diameter_l770_77073

theorem circle_diameter (A : ℝ) (r : ℝ) (D : ℝ) : 
  A = 100 * Real.pi → A = Real.pi * r^2 → D = 2 * r → D = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l770_77073


namespace NUMINAMATH_CALUDE_probability_D_given_E_l770_77076

-- Define the regions D and E
def region_D (x y : ℝ) : Prop := y ≤ 1 ∧ y ≥ x^2
def region_E (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Define the areas of regions D and E
noncomputable def area_D : ℝ := 4/3
noncomputable def area_E : ℝ := 2

-- State the theorem
theorem probability_D_given_E : 
  (area_D / area_E) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_probability_D_given_E_l770_77076


namespace NUMINAMATH_CALUDE_angle_range_l770_77025

theorem angle_range (θ α : Real) : 
  (∃ (x y : Real), x = Real.sin (α - π/3) ∧ y = Real.sqrt 3 ∧ 
    x = Real.sin θ ∧ y = Real.cos θ) →
  Real.sin (2*θ) ≤ 0 →
  -2*π/3 ≤ α ∧ α ≤ π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_range_l770_77025


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l770_77049

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem fourth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) : 
  a 4 = 9 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l770_77049


namespace NUMINAMATH_CALUDE_complex_square_problem_l770_77093

theorem complex_square_problem (z : ℂ) (h : z⁻¹ = 1 + Complex.I) : z^2 = -Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_problem_l770_77093


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l770_77020

theorem sqrt_equation_solution (x : ℝ) : 
  x ≥ 2 → 
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) ↔
  (11 ≤ x ∧ x ≤ 27) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l770_77020


namespace NUMINAMATH_CALUDE_man_downstream_speed_l770_77030

/-- Given a man's upstream speed and the stream speed, calculates his downstream speed -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Proves that given the specified conditions, the man's downstream speed is 14 kmph -/
theorem man_downstream_speed :
  let upstream_speed : ℝ := 8
  let stream_speed : ℝ := 3
  downstream_speed upstream_speed stream_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l770_77030


namespace NUMINAMATH_CALUDE_animals_per_aquarium_l770_77044

/-- Given that Tyler has 8 aquariums and 512 saltwater animals in total,
    prove that there are 64 animals in each aquarium. -/
theorem animals_per_aquarium (num_aquariums : ℕ) (total_animals : ℕ) 
  (h1 : num_aquariums = 8) (h2 : total_animals = 512) :
  total_animals / num_aquariums = 64 := by
  sorry

end NUMINAMATH_CALUDE_animals_per_aquarium_l770_77044


namespace NUMINAMATH_CALUDE_income_ratio_proof_l770_77010

/-- Given two persons P1 and P2 with the following conditions:
    1. The ratio of their expenditures is 3:2
    2. Each saves Rs. 1800 at the end of the year
    3. The income of P1 is Rs. 4500
    Prove that the ratio of their incomes is 5:4 -/
theorem income_ratio_proof (expenditure_ratio : ℚ) (savings : ℕ) (income_p1 : ℕ) :
  expenditure_ratio = 3/2 →
  savings = 1800 →
  income_p1 = 4500 →
  ∃ (income_p2 : ℕ), (income_p1 : ℚ) / income_p2 = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_income_ratio_proof_l770_77010


namespace NUMINAMATH_CALUDE_digits_used_128_l770_77084

/-- The number of digits used to number pages from 1 to n -/
def digits_used (n : ℕ) : ℕ :=
  (min n 9) +
  (if n ≥ 10 then 2 * (min (n - 9) 90) else 0) +
  (if n ≥ 100 then 3 * (n - 99) else 0)

/-- The theorem stating that the number of digits used to number pages from 1 to 128 is 276 -/
theorem digits_used_128 : digits_used 128 = 276 := by
  sorry

end NUMINAMATH_CALUDE_digits_used_128_l770_77084


namespace NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l770_77041

-- Define the complex number
def z : ℂ := 4 - 15 * Complex.I

-- State the theorem
theorem magnitude_of_4_minus_15i : Complex.abs z = Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l770_77041


namespace NUMINAMATH_CALUDE_inequality_solution_l770_77096

theorem inequality_solution (x : ℝ) :
  x ≠ 2 → x ≠ 0 →
  ((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ 
   (0 < x ∧ x ≤ 1/2) ∨ (2 < x ∧ x ≤ 11/2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l770_77096


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l770_77067

def a : Fin 2 → ℝ := ![(-1), 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

theorem perpendicular_vectors (m : ℝ) : 
  (∀ i : Fin 2, (a + b m) i * a i = 0) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l770_77067


namespace NUMINAMATH_CALUDE_sarah_speeding_tickets_l770_77011

theorem sarah_speeding_tickets (total_tickets : ℕ) (mark_parking : ℕ) :
  total_tickets = 24 →
  mark_parking = 8 →
  ∃ (sarah_speeding : ℕ),
    sarah_speeding = 6 ∧
    sarah_speeding + sarah_speeding + mark_parking + mark_parking / 2 = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_sarah_speeding_tickets_l770_77011


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_two_l770_77071

def A : Set ℕ := {x : ℕ | 5 + 4 * x - x^2 > 0}

def B : Set ℕ := {x : ℕ | x < 3}

theorem A_intersect_B_eq_zero_one_two : A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_two_l770_77071


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_plus_c_l770_77040

theorem min_value_a_plus_b_plus_c (a b c : ℝ) 
  (h1 : a^2 + b^2 ≤ c) (h2 : c ≤ 1) : 
  ∀ x y z : ℝ, x^2 + y^2 ≤ z ∧ z ≤ 1 → a + b + c ≤ x + y + z ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀^2 + b₀^2 ≤ c₀ ∧ c₀ ≤ 1 ∧ a₀ + b₀ + c₀ = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_plus_c_l770_77040


namespace NUMINAMATH_CALUDE_angies_age_problem_l770_77094

theorem angies_age_problem (angie_age : ℕ) (certain_number : ℕ) : 
  angie_age = 8 → 2 * angie_age + certain_number = 20 → certain_number = 4 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_problem_l770_77094


namespace NUMINAMATH_CALUDE_mirror_area_l770_77029

/-- The area of a rectangular mirror inside a frame -/
theorem mirror_area (frame_width : ℕ) (frame_height : ℕ) (frame_thickness : ℕ) : 
  frame_width = 90 ∧ frame_height = 70 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 2400 := by
sorry

end NUMINAMATH_CALUDE_mirror_area_l770_77029


namespace NUMINAMATH_CALUDE_triangle_inequality_l770_77036

theorem triangle_inequality (A B C : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : A + B + C = π) : 
  (Real.sin (2*A) + Real.sin (2*B))^2 / (Real.sin A * Real.sin B) + 
  (Real.sin (2*B) + Real.sin (2*C))^2 / (Real.sin B * Real.sin C) + 
  (Real.sin (2*C) + Real.sin (2*A))^2 / (Real.sin C * Real.sin A) ≤ 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l770_77036


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_l770_77026

/-- Represents a two-digit number in the dozenal (base 12) system -/
structure DozenalNumber :=
  (tens : Nat)
  (ones : Nat)
  (tens_valid : 1 ≤ tens ∧ tens ≤ 11)
  (ones_valid : ones ≤ 11)

/-- Converts a DozenalNumber to its decimal representation -/
def toDecimal (n : DozenalNumber) : Nat :=
  12 * n.tens + n.ones

/-- Calculates the sum of digits of a DozenalNumber -/
def digitSum (n : DozenalNumber) : Nat :=
  n.tens + n.ones

/-- Checks if a DozenalNumber satisfies the given condition -/
def satisfiesCondition (n : DozenalNumber) : Prop :=
  (toDecimal n - digitSum n) % 12 = 5

theorem count_satisfying_numbers :
  ∃ (numbers : Finset DozenalNumber),
    numbers.card = 12 ∧
    (∀ n : DozenalNumber, n ∈ numbers ↔ satisfiesCondition n) :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_numbers_l770_77026


namespace NUMINAMATH_CALUDE_unripe_orange_harvest_l770_77069

/-- The number of sacks of unripe oranges harvested per day -/
def daily_unripe_harvest : ℕ := 65

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges harvested over the harvest period -/
def total_unripe_harvest : ℕ := daily_unripe_harvest * harvest_days

theorem unripe_orange_harvest : total_unripe_harvest = 390 := by
  sorry

end NUMINAMATH_CALUDE_unripe_orange_harvest_l770_77069


namespace NUMINAMATH_CALUDE_dividend_from_quotient_and_remainder_l770_77065

theorem dividend_from_quotient_and_remainder :
  ∀ (dividend quotient remainder : ℕ),
    dividend = 23 * quotient + remainder →
    quotient = 38 →
    remainder = 7 →
    dividend = 881 := by
  sorry

end NUMINAMATH_CALUDE_dividend_from_quotient_and_remainder_l770_77065


namespace NUMINAMATH_CALUDE_vitya_catches_up_in_5_minutes_l770_77081

-- Define the initial walking speed
def initial_speed : ℝ := 1

-- Define the time they walk before Vitya turns back
def initial_time : ℝ := 10

-- Define Vitya's speed multiplier when he starts chasing
def speed_multiplier : ℝ := 5

-- Define the theorem
theorem vitya_catches_up_in_5_minutes :
  let distance := 2 * initial_speed * initial_time
  let relative_speed := speed_multiplier * initial_speed - initial_speed
  distance / relative_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_vitya_catches_up_in_5_minutes_l770_77081


namespace NUMINAMATH_CALUDE_special_product_equality_l770_77089

theorem special_product_equality (x y : ℝ) : 
  (2 * x^3 - 5 * y^2) * (4 * x^6 + 10 * x^3 * y^2 + 25 * y^4) = 8 * x^9 - 125 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_special_product_equality_l770_77089


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l770_77095

/-- Given two supplementary angles in a ratio of 5:3, the measure of the smaller angle is 67.5° -/
theorem supplementary_angles_ratio (angle1 angle2 : ℝ) : 
  angle1 + angle2 = 180 →  -- supplementary angles
  angle1 / angle2 = 5 / 3 →  -- ratio of 5:3
  min angle1 angle2 = 67.5 :=  -- smaller angle is 67.5°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l770_77095


namespace NUMINAMATH_CALUDE_esha_lag_behind_anusha_l770_77053

/-- Represents the runners in the race -/
inductive Runner
| Anusha
| Banu
| Esha

/-- The race parameters and conditions -/
structure RaceConditions where
  race_length : ℝ
  speeds : Runner → ℝ
  anusha_fastest : speeds Runner.Anusha > speeds Runner.Banu ∧ speeds Runner.Banu > speeds Runner.Esha
  banu_lag : speeds Runner.Banu / speeds Runner.Anusha = 9 / 10
  esha_lag : speeds Runner.Esha / speeds Runner.Banu = 9 / 10

/-- The theorem to be proved -/
theorem esha_lag_behind_anusha (rc : RaceConditions) (h : rc.race_length = 100) :
  rc.race_length - (rc.speeds Runner.Esha / rc.speeds Runner.Anusha) * rc.race_length = 19 := by
  sorry

end NUMINAMATH_CALUDE_esha_lag_behind_anusha_l770_77053


namespace NUMINAMATH_CALUDE_decreasing_geometric_sequence_characterization_l770_77070

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem decreasing_geometric_sequence_characterization
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) :
  (a 1 > a 2 ∧ a 2 > a 3) ↔ is_decreasing_sequence a :=
by sorry

end NUMINAMATH_CALUDE_decreasing_geometric_sequence_characterization_l770_77070


namespace NUMINAMATH_CALUDE_price_change_l770_77018

theorem price_change (r s : ℝ) (h : r ≠ -100) (h2 : s ≠ 100) : 
  let initial_price := (10000 : ℝ) / (10000 + 100 * (r - s) - r * s)
  let price_after_increase := initial_price * (1 + r / 100)
  let final_price := price_after_increase * (1 - s / 100)
  final_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_price_change_l770_77018


namespace NUMINAMATH_CALUDE_final_savings_calculation_correct_l770_77087

/-- Calculates the final savings given initial savings, monthly income, monthly expenses, and number of months. -/
def calculate_final_savings (initial_savings monthly_income monthly_expenses : ℕ) (num_months : ℕ) : ℕ :=
  initial_savings + num_months * monthly_income - num_months * monthly_expenses

/-- Theorem stating that the final savings calculation is correct for the given problem. -/
theorem final_savings_calculation_correct :
  let initial_savings : ℕ := 849400
  let monthly_income : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthly_expenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let num_months : ℕ := 5
  calculate_final_savings initial_savings monthly_income monthly_expenses num_months = 1106900 := by
  sorry

#eval calculate_final_savings 849400 110000 58500 5

end NUMINAMATH_CALUDE_final_savings_calculation_correct_l770_77087


namespace NUMINAMATH_CALUDE_polygon_triangle_division_l770_77047

/-- 
A polygon where the sum of interior angles is twice the sum of exterior angles
can be divided into at most 4 triangles by connecting one vertex to all others.
-/
theorem polygon_triangle_division :
  ∀ (n : ℕ), 
  (n - 2) * 180 = 2 * 360 →
  n - 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polygon_triangle_division_l770_77047


namespace NUMINAMATH_CALUDE_min_balls_for_three_same_color_60_6_l770_77005

/-- Given a bag with colored balls, returns the minimum number of balls
    that must be picked to ensure at least three balls of the same color are picked. -/
def min_balls_for_three_same_color (total_balls : ℕ) (balls_per_color : ℕ) : ℕ :=
  2 * (total_balls / balls_per_color) + 1

/-- Proves that for a bag with 60 balls and 6 balls of each color,
    the minimum number of balls to pick to ensure at least three of the same color is 21. -/
theorem min_balls_for_three_same_color_60_6 :
  min_balls_for_three_same_color 60 6 = 21 := by
  sorry

#eval min_balls_for_three_same_color 60 6

end NUMINAMATH_CALUDE_min_balls_for_three_same_color_60_6_l770_77005


namespace NUMINAMATH_CALUDE_fish_tanks_theorem_l770_77012

/-- The total number of fish in three tanks, where one tank has a given number of fish
    and the other two have twice as many fish each as the first. -/
def total_fish (first_tank_fish : ℕ) : ℕ :=
  first_tank_fish + 2 * (2 * first_tank_fish)

/-- Theorem stating that with 3 fish tanks, where one tank has 20 fish and the other two
    have twice as many fish each as the first, the total number of fish is 100. -/
theorem fish_tanks_theorem : total_fish 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_fish_tanks_theorem_l770_77012


namespace NUMINAMATH_CALUDE_triangle_angle_120_l770_77045

/-- In a triangle ABC with side lengths a, b, and c, if a^2 = b^2 + bc + c^2, then angle A is 120° -/
theorem triangle_angle_120 (a b c : ℝ) (h : a^2 = b^2 + b*c + c^2) :
  let A := Real.arccos ((c^2 + b^2 - a^2) / (2*b*c))
  A = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_120_l770_77045


namespace NUMINAMATH_CALUDE_fourth_shot_probability_l770_77062

/-- The probability of making a shot given the previous shot was made -/
def p_make_given_make : ℚ := 2/3

/-- The probability of making a shot given the previous shot was missed -/
def p_make_given_miss : ℚ := 1/3

/-- The probability of making the first shot -/
def p_first_shot : ℚ := 2/3

/-- The probability of making the n-th shot -/
def p_nth_shot (n : ℕ) : ℚ :=
  1/2 * (1 + 1 / 3^n)

theorem fourth_shot_probability :
  p_nth_shot 4 = 41/81 :=
sorry

end NUMINAMATH_CALUDE_fourth_shot_probability_l770_77062


namespace NUMINAMATH_CALUDE_triangle_inequalities_l770_77009

theorem triangle_inequalities (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ c ∧ c ≤ 1) 
  (h4 : a + b + c = 2) : 
  (a * b * c + 28 / 27 ≥ a * b + b * c + c * a) ∧ 
  (a * b + b * c + c * a ≥ a * b * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l770_77009


namespace NUMINAMATH_CALUDE_problem_solution_l770_77086

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 5) 
  (h3 : x = 9) : 
  y = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l770_77086


namespace NUMINAMATH_CALUDE_integral_x_plus_sqrt_4_minus_x_squared_l770_77007

open Set
open MeasureTheory
open Interval

/-- The definite integral of x + √(4 - x^2) from -2 to 2 equals 2π -/
theorem integral_x_plus_sqrt_4_minus_x_squared : 
  ∫ x in (-2)..2, (x + Real.sqrt (4 - x^2)) = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_sqrt_4_minus_x_squared_l770_77007


namespace NUMINAMATH_CALUDE_equal_pairs_l770_77021

theorem equal_pairs (x y z : ℝ) (h : xy + z = yz + x ∧ yz + x = zx + y) :
  x = y ∨ y = z ∨ z = x := by
  sorry

end NUMINAMATH_CALUDE_equal_pairs_l770_77021


namespace NUMINAMATH_CALUDE_a_minus_b_value_l770_77015

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) :
  a - b = 7 ∨ a - b = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l770_77015


namespace NUMINAMATH_CALUDE_cut_cylinder_unpainted_face_area_l770_77055

/-- The area of an unpainted face of a cut cylinder -/
theorem cut_cylinder_unpainted_face_area (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  let sector_area := π * r^2 / 4
  let triangle_area := r^2 / 2
  let unpainted_face_area := h * (sector_area + triangle_area)
  unpainted_face_area = 62.5 * π + 125 := by
  sorry

end NUMINAMATH_CALUDE_cut_cylinder_unpainted_face_area_l770_77055


namespace NUMINAMATH_CALUDE_pages_read_on_thursday_l770_77031

theorem pages_read_on_thursday (wednesday_pages friday_pages total_pages : ℕ) 
  (h1 : wednesday_pages = 18)
  (h2 : friday_pages = 23)
  (h3 : total_pages = 60) :
  total_pages - (wednesday_pages + friday_pages) = 19 := by
sorry

end NUMINAMATH_CALUDE_pages_read_on_thursday_l770_77031


namespace NUMINAMATH_CALUDE_equal_numbers_product_l770_77078

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 24 →
  a = 20 →
  b = 25 →
  c = 33 →
  d = e →
  d * e = 441 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l770_77078


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l770_77064

theorem consecutive_integers_sum_of_squares : 
  ∀ x : ℕ, 
    x > 0 → 
    x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) → 
    x^2 + (x + 1)^2 + (x + 2)^2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l770_77064


namespace NUMINAMATH_CALUDE_james_annual_training_hours_l770_77061

/-- Calculates the total training hours per year for an athlete with a specific schedule. -/
def training_hours_per_year (sessions_per_day : ℕ) (hours_per_session : ℕ) (training_days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  sessions_per_day * hours_per_session * training_days_per_week * weeks_per_year

/-- Proves that James' training schedule results in 2080 hours per year. -/
theorem james_annual_training_hours :
  training_hours_per_year 2 4 5 52 = 2080 := by
  sorry

#eval training_hours_per_year 2 4 5 52

end NUMINAMATH_CALUDE_james_annual_training_hours_l770_77061


namespace NUMINAMATH_CALUDE_expression_evaluation_l770_77017

theorem expression_evaluation (a : ℝ) (h : a = -6) :
  (1 - a / (a - 3)) / ((a^2 + 3*a) / (a^2 - 9)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l770_77017


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l770_77060

theorem lunch_cost_proof (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 22 → difference = 5 → 
  (∃ (your_cost : ℝ), your_cost + (your_cost + difference) = total) →
  friend_cost = 13.5 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l770_77060


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_7350_l770_77083

theorem gcd_lcm_sum_75_7350 : Nat.gcd 75 7350 + Nat.lcm 75 7350 = 3225 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_7350_l770_77083


namespace NUMINAMATH_CALUDE_kenya_has_more_peanuts_l770_77080

/-- The number of peanuts Jose has -/
def jose_peanuts : ℕ := 85

/-- The number of peanuts Kenya has -/
def kenya_peanuts : ℕ := 133

/-- The difference in peanuts between Kenya and Jose -/
def peanut_difference : ℕ := kenya_peanuts - jose_peanuts

theorem kenya_has_more_peanuts : peanut_difference = 48 := by
  sorry

end NUMINAMATH_CALUDE_kenya_has_more_peanuts_l770_77080


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_sqrt2_minus_half_achievable_min_value_l770_77048

theorem min_value_of_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∀ x y z w, x ≥ 0 ∧ y ≥ 0 ∧ z > 0 ∧ w > 0 ∧ z + w ≥ x + y →
  (b / (c + d)) + (c / (a + b)) ≤ (z / (w + y)) + (w / (x + z)) :=
by sorry

theorem min_value_sqrt2_minus_half (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 :=
by sorry

theorem achievable_min_value (a d b c : ℝ) :
  ∃ a d b c, a ≥ 0 ∧ d ≥ 0 ∧ b > 0 ∧ c > 0 ∧ b + c ≥ a + d ∧
  (b / (c + d)) + (c / (a + b)) = Real.sqrt 2 - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_sqrt2_minus_half_achievable_min_value_l770_77048


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_three_l770_77001

def A : Set ℝ := {2, 3}
def B (a : ℝ) : Set ℝ := {1, 2, a}

theorem subset_implies_a_equals_three (h : A ⊆ B a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_three_l770_77001


namespace NUMINAMATH_CALUDE_decreasing_linear_function_l770_77077

def linear_function (k b x : ℝ) : ℝ := k * x + b

theorem decreasing_linear_function (k b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function k b x₁ > linear_function k b x₂) ↔ k < 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_l770_77077


namespace NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l770_77022

theorem system_solution_negative_implies_m_range (m : ℝ) : 
  (∃ x y : ℝ, x - y = 2*m + 7 ∧ x + y = 4*m - 3 ∧ x < 0 ∧ y < 0) → m < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l770_77022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_a_7_value_l770_77034

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 1 + a 7 = a 3 + a 5 := by sorry

theorem a_7_value (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 + a 5 = 10) : 
  a 7 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_a_7_value_l770_77034


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l770_77019

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem derivative_f_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l770_77019


namespace NUMINAMATH_CALUDE_squared_one_necessary_not_sufficient_l770_77066

theorem squared_one_necessary_not_sufficient (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) := by sorry

end NUMINAMATH_CALUDE_squared_one_necessary_not_sufficient_l770_77066


namespace NUMINAMATH_CALUDE_nancy_football_tickets_l770_77032

/-- The total amount Nancy spends on football tickets for three months -/
def total_spent (this_month_games : ℕ) (this_month_price : ℕ) 
                (last_month_games : ℕ) (last_month_price : ℕ) 
                (next_month_games : ℕ) (next_month_price : ℕ) : ℕ :=
  this_month_games * this_month_price + 
  last_month_games * last_month_price + 
  next_month_games * next_month_price

theorem nancy_football_tickets : 
  total_spent 9 5 8 4 7 6 = 119 := by
  sorry

end NUMINAMATH_CALUDE_nancy_football_tickets_l770_77032


namespace NUMINAMATH_CALUDE_inequality_range_l770_77063

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 6| - |x - 4| ≤ a^2 - 3*a) ↔ 
  (a ≤ -2 ∨ a ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l770_77063


namespace NUMINAMATH_CALUDE_circle_equation_l770_77054

/-- Theorem: Equation of a Circle
    For any point (x, y) on a circle with radius R and center (a, b),
    the equation (x-a)^2 + (y-b)^2 = R^2 holds. -/
theorem circle_equation (R a b x y : ℝ) (h : (x - a)^2 + (y - b)^2 = R^2) :
  (x - a)^2 + (y - b)^2 = R^2 := by
  sorry

#check circle_equation

end NUMINAMATH_CALUDE_circle_equation_l770_77054


namespace NUMINAMATH_CALUDE_probability_even_greater_than_10_l770_77075

def ball_set : Finset ℕ := {1, 2, 3, 4, 5}

def is_valid_product (a b : ℕ) : Bool :=
  Even (a * b) ∧ a * b > 10

def valid_outcomes : Finset (ℕ × ℕ) :=
  ball_set.product ball_set

def favorable_outcomes : Finset (ℕ × ℕ) :=
  valid_outcomes.filter (fun p => is_valid_product p.1 p.2)

theorem probability_even_greater_than_10 :
  (favorable_outcomes.card : ℚ) / valid_outcomes.card = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_even_greater_than_10_l770_77075


namespace NUMINAMATH_CALUDE_more_polygons_with_specific_point_l770_77059

theorem more_polygons_with_specific_point (n : ℕ) (h : n = 16) :
  let total_polygons := 2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)
  let polygons_with_point := 2^(n-1) - (Nat.choose (n-1) 0 + Nat.choose (n-1) 1)
  let polygons_without_point := total_polygons - polygons_with_point
  polygons_with_point > polygons_without_point := by
sorry

end NUMINAMATH_CALUDE_more_polygons_with_specific_point_l770_77059


namespace NUMINAMATH_CALUDE_least_value_theorem_l770_77024

theorem least_value_theorem (x y z : ℕ+) 
  (h1 : 5 * y.val = 6 * z.val)
  (h2 : x.val + y.val + z.val = 26) :
  5 * y.val = 30 := by
sorry

end NUMINAMATH_CALUDE_least_value_theorem_l770_77024


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l770_77057

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals : diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l770_77057


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l770_77050

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the volleyball club -/
def total_players : ℕ := 18

/-- The number of quadruplets -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 6

/-- The number of non-quadruplet players -/
def other_players : ℕ := total_players - num_quadruplets

theorem volleyball_team_selection :
  (binomial total_players num_starters) -
  (binomial other_players (num_starters - num_quadruplets)) -
  (binomial other_players num_starters) = 15470 := by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l770_77050


namespace NUMINAMATH_CALUDE_candy_cost_theorem_l770_77074

/-- Calculates the cost of purchasing chocolate candies with a bulk discount -/
def calculate_candy_cost (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let boxes_needed := total_candies / candies_per_box
  let total_cost := boxes_needed * cost_per_box
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

/-- The cost of purchasing 450 chocolate candies is $67.5 -/
theorem candy_cost_theorem :
  calculate_candy_cost 30 5 450 300 (1/10) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_theorem_l770_77074


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l770_77079

theorem lucky_larry_coincidence (a b c d e : ℤ) : 
  a = 1 → b = 2 → c = 3 → d = 4 → 
  (a - b - c - d + e = a - (b - (c - (d + e)))) → e = 3 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l770_77079


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l770_77013

theorem complex_modulus_problem (z : ℂ) : (z - 3) * (1 - 3*I) = 10 → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l770_77013


namespace NUMINAMATH_CALUDE_polynomial_value_l770_77033

/-- Given a polynomial function f(x) = ax^5 + bx^3 - cx + 2 where f(-3) = 9, 
    prove that f(3) = -5 -/
theorem polynomial_value (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 - c * x + 2
  f (-3) = 9 → f 3 = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l770_77033


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l770_77092

theorem angle_in_second_quadrant (α : Real) (x : Real) :
  -- α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- P(x,6) is on the terminal side of α
  x < 0 →
  -- sin α = 3/5
  Real.sin α = 3 / 5 →
  -- x = -8
  x = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l770_77092


namespace NUMINAMATH_CALUDE_solution_set_inequality_l770_77099

theorem solution_set_inequality (x : ℝ) :
  (Set.Icc 1 2 : Set ℝ) = {x | -x^2 + 3*x - 2 ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l770_77099


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l770_77023

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) :
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_quarter_l770_77023


namespace NUMINAMATH_CALUDE_velocity_at_5_seconds_l770_77038

-- Define the position function
def s (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 4

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 6 * t - 2

-- Theorem statement
theorem velocity_at_5_seconds :
  v 5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_velocity_at_5_seconds_l770_77038


namespace NUMINAMATH_CALUDE_det_B_is_one_l770_77006

def B (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]

theorem det_B_is_one (a d : ℝ) (h : B a d + (B a d)⁻¹ = 0) : 
  Matrix.det (B a d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_B_is_one_l770_77006


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l770_77039

/-- Given a fractional equation and conditions on its solution, 
    this theorem proves the range of values for the parameter a. -/
theorem fractional_equation_solution_range (a x : ℝ) : 
  (a / (x + 2) = 1 - 3 / (x + 2)) →
  (x < 0) →
  (a < -1 ∧ a ≠ -3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l770_77039


namespace NUMINAMATH_CALUDE_f_of_g_10_l770_77027

def g (x : ℝ) : ℝ := 4 * x + 6

def f (x : ℝ) : ℝ := 6 * x - 10

theorem f_of_g_10 : f (g 10) = 266 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_10_l770_77027


namespace NUMINAMATH_CALUDE_product_abcd_zero_l770_77037

theorem product_abcd_zero 
  (a b c d : ℝ) 
  (eq1 : 3*a + 2*b + 4*c + 6*d = 60)
  (eq2 : 4*(d+c) = b^2)
  (eq3 : 4*b + 2*c = a)
  (eq4 : c - 2 = d) :
  a * b * c * d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_zero_l770_77037


namespace NUMINAMATH_CALUDE_income_calculation_l770_77004

/-- Proves that given a person's income and expenditure are in the ratio 4:3, 
    and their savings are Rs. 5,000, their income is Rs. 20,000. -/
theorem income_calculation (income expenditure savings : ℕ) : 
  income * 3 = expenditure * 4 →  -- Income and expenditure ratio is 4:3
  income - expenditure = savings → -- Savings definition
  savings = 5000 →                -- Given savings amount
  income = 20000 :=               -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l770_77004


namespace NUMINAMATH_CALUDE_tiffany_lives_lost_l770_77072

theorem tiffany_lives_lost (initial_lives gained_lives final_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : gained_lives = 27)
  (h3 : final_lives = 56) :
  initial_lives - (final_lives - gained_lives) = 14 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_lives_lost_l770_77072


namespace NUMINAMATH_CALUDE_new_average_production_l770_77098

theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) :
  n = 1 ∧ past_average = 50 ∧ today_production = 60 →
  (n * past_average + today_production) / (n + 1) = 55 := by
sorry

end NUMINAMATH_CALUDE_new_average_production_l770_77098


namespace NUMINAMATH_CALUDE_annas_ebook_readers_l770_77043

theorem annas_ebook_readers (anna_readers john_initial_readers john_final_readers total_readers : ℕ) 
  (h1 : john_initial_readers = anna_readers - 15)
  (h2 : john_final_readers = john_initial_readers - 3)
  (h3 : anna_readers + john_final_readers = total_readers)
  (h4 : total_readers = 82) : anna_readers = 50 := by
  sorry

end NUMINAMATH_CALUDE_annas_ebook_readers_l770_77043
