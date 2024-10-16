import Mathlib

namespace NUMINAMATH_CALUDE_colten_chickens_l1140_114001

theorem colten_chickens (total : ℕ) (q s c : ℕ) : 
  total = 383 →
  q = 2 * s + 25 →
  s = 3 * c - 4 →
  q + s + c = total →
  c = 37 := by
sorry

end NUMINAMATH_CALUDE_colten_chickens_l1140_114001


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l1140_114078

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (gain_heads : ℚ) (loss_tails : ℚ) : ℚ :=
  p_heads * gain_heads + p_tails * (-loss_tails)

theorem unfair_coin_expected_value :
  let p_heads : ℚ := 3/5
  let p_tails : ℚ := 2/5
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 6
  coin_flip_expected_value p_heads p_tails gain_heads loss_tails = 3/5 :=
by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l1140_114078


namespace NUMINAMATH_CALUDE_f_range_theorem_l1140_114098

-- Define the function f
def f (x y z : ℝ) := (z - x) * (z - y)

-- State the theorem
theorem f_range_theorem (x y z : ℝ) 
  (h1 : x + y + z = 1) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) 
  (h4 : z ≥ 0) :
  ∃ (w : ℝ), f x y z = w ∧ w ∈ Set.Icc (-1/8 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_theorem_l1140_114098


namespace NUMINAMATH_CALUDE_fleas_perished_count_l1140_114037

/-- Represents the count of fleas in an ear -/
structure FleaCount where
  adultA : ℕ
  adultB : ℕ
  nymphA : ℕ
  nymphB : ℕ

/-- Represents the survival rates for different flea types -/
structure SurvivalRates where
  adultA : ℚ
  adultB : ℚ
  nymphA : ℚ
  nymphB : ℚ

def rightEar : FleaCount := {
  adultA := 42,
  adultB := 80,
  nymphA := 37,
  nymphB := 67
}

def leftEar : FleaCount := {
  adultA := 29,
  adultB := 64,
  nymphA := 71,
  nymphB := 45
}

def survivalRates : SurvivalRates := {
  adultA := 3/4,
  adultB := 3/5,
  nymphA := 2/5,
  nymphB := 11/20
}

/-- Calculates the number of fleas that perished in an ear -/
def fleaPerished (ear : FleaCount) (rates : SurvivalRates) : ℚ :=
  ear.adultA * (1 - rates.adultA) +
  ear.adultB * (1 - rates.adultB) +
  ear.nymphA * (1 - rates.nymphA) +
  ear.nymphB * (1 - rates.nymphB)

theorem fleas_perished_count :
  ⌊fleaPerished rightEar survivalRates + fleaPerished leftEar survivalRates⌋ = 190 := by
  sorry

end NUMINAMATH_CALUDE_fleas_perished_count_l1140_114037


namespace NUMINAMATH_CALUDE_units_digit_sum_l1140_114063

/-- The base of the number system -/
def base : ℕ := 8

/-- The first number in base 8 -/
def num1 : ℕ := 63

/-- The second number in base 8 -/
def num2 : ℕ := 74

/-- The units digit of the first number -/
def units_digit1 : ℕ := 3

/-- The units digit of the second number -/
def units_digit2 : ℕ := 4

/-- Theorem: The units digit of the sum of num1 and num2 in base 8 is 7 -/
theorem units_digit_sum : (num1 + num2) % base = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_l1140_114063


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1140_114023

theorem min_four_dollar_frisbees 
  (total_frisbees : ℕ) 
  (total_receipts : ℕ) 
  (h_total : total_frisbees = 60) 
  (h_receipts : total_receipts = 204) : 
  ∃ (three_dollar : ℕ) (four_dollar : ℕ), 
    three_dollar + four_dollar = total_frisbees ∧ 
    3 * three_dollar + 4 * four_dollar = total_receipts ∧ 
    four_dollar ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l1140_114023


namespace NUMINAMATH_CALUDE_equation_solution_l1140_114073

theorem equation_solution : ∃! x : ℝ, (64 : ℝ)^(x - 1) / (4 : ℝ)^(x - 1) = (256 : ℝ)^(2*x) ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1140_114073


namespace NUMINAMATH_CALUDE_point_quadrant_theorem_l1140_114042

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Definition of the third quadrant -/
def in_third_quadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(x, y-2) is in the fourth quadrant, then B(y-2, -x) is in the third quadrant -/
theorem point_quadrant_theorem (x y : ℝ) :
  let A : Point2D := ⟨x, y - 2⟩
  let B : Point2D := ⟨y - 2, -x⟩
  in_fourth_quadrant A → in_third_quadrant B := by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_theorem_l1140_114042


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l1140_114082

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement is valid. -/
def is_valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧
  table.seated_people ≤ table.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < table.total_chairs →
    ∃ (occupied_seat : ℕ), occupied_seat < table.total_chairs ∧
      (new_seat = (occupied_seat + 1) % table.total_chairs ∨
       new_seat = (occupied_seat - 1 + table.total_chairs) % table.total_chairs)

/-- The main theorem stating the smallest valid number of seated people. -/
theorem smallest_valid_seating :
  ∀ (table : CircularTable),
    table.total_chairs = 80 →
    (is_valid_seating table ↔ table.seated_people ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l1140_114082


namespace NUMINAMATH_CALUDE_opposite_numbers_product_l1140_114027

theorem opposite_numbers_product (x y : ℝ) : 
  (|x - 3| + |y + 1| = 0) → xy = -3 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_product_l1140_114027


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_values_l1140_114092

-- Define the two lines
def line1 (t k : ℝ) : ℝ × ℝ × ℝ := (1 + t, 2 + 2*t, 3 - k*t)
def line2 (u k : ℝ) : ℝ × ℝ × ℝ := (2 + u, 5 + k*u, 6 + u)

-- Define the condition for the lines to be coplanar
def are_coplanar (k : ℝ) : Prop :=
  ∃ t u, line1 t k = line2 u k

-- State the theorem
theorem lines_coplanar_iff_k_values :
  ∀ k : ℝ, are_coplanar k ↔ (k = -2 + Real.sqrt 6 ∨ k = -2 - Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_values_l1140_114092


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1140_114031

theorem polynomial_simplification (x : ℝ) :
  (3 * x^6 + 2 * x^5 - x^4 + 3 * x^2 + 15) - (x^6 + 4 * x^5 + 3 * x^3 - 2 * x^2 + 20) =
  2 * x^6 - 2 * x^5 - x^4 + 5 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1140_114031


namespace NUMINAMATH_CALUDE_exists_non_squareable_number_l1140_114080

/-- A complication is adding a single digit to a number. -/
def Complication := Nat → Nat

/-- Apply a sequence of complications to a number. -/
def applyComplications (n : Nat) (complications : List Complication) : Nat :=
  complications.foldl (fun acc c => c acc) n

/-- Check if a number is a perfect square. -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem exists_non_squareable_number : 
  ∃ n : Nat, ∀ complications : List Complication, 
    complications.length ≤ 100 → 
    ¬(isPerfectSquare (applyComplications n complications)) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_squareable_number_l1140_114080


namespace NUMINAMATH_CALUDE_decompose_4_705_l1140_114074

theorem decompose_4_705 : 
  ∃ (units hundredths thousandths : ℕ),
    4.705 = (units : ℝ) + (7 : ℝ) / 10 + (thousandths : ℝ) / 1000 ∧
    units = 4 ∧
    thousandths = 5 := by
  sorry

end NUMINAMATH_CALUDE_decompose_4_705_l1140_114074


namespace NUMINAMATH_CALUDE_point_distance_constraint_l1140_114059

/-- Given points A(1, 0) and B(4, 0), and a point P on the line x + my - 1 = 0
    such that |PA| = 2|PB|, prove that the range of values for m is m ≥ √3 or m ≤ -√3. -/
theorem point_distance_constraint (m : ℝ) : 
  (∃ (x y : ℝ), x + m * y - 1 = 0 ∧ 
   (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)) ↔ 
  (m ≥ Real.sqrt 3 ∨ m ≤ -Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_point_distance_constraint_l1140_114059


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l1140_114060

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : Prime d) :
  let r := n % d
  r = (n - (n - r)) % d ∧ 
  ∀ m : Nat, m < r → (n - m) % d ≠ 0 :=
by
  sorry

#eval 2376819 % 139  -- This should evaluate to 135

theorem specific_case : 
  let n := 2376819
  let d := 139
  Prime d ∧ 
  (n - 135) % d = 0 ∧
  ∀ m : Nat, m < 135 → (n - m) % d ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l1140_114060


namespace NUMINAMATH_CALUDE_prob_third_given_a_wins_l1140_114057

/-- The probability of Player A winning a single game -/
def p : ℚ := 2/3

/-- The probability of Player A winning the championship -/
def prob_a_wins : ℚ := p^2 + 2*p^2*(1-p)

/-- The probability of the match going to the third game and Player A winning -/
def prob_third_and_a_wins : ℚ := 2*p^2*(1-p)

/-- The probability of the match going to the third game given that Player A wins the championship -/
theorem prob_third_given_a_wins : 
  prob_third_and_a_wins / prob_a_wins = 2/5 := by sorry

end NUMINAMATH_CALUDE_prob_third_given_a_wins_l1140_114057


namespace NUMINAMATH_CALUDE_sqrt_fraction_equals_two_power_fifteen_l1140_114010

theorem sqrt_fraction_equals_two_power_fifteen :
  let thirty_two := (2 : ℝ) ^ 5
  let sixteen := (2 : ℝ) ^ 4
  (((thirty_two ^ 15 + sixteen ^ 15) / (thirty_two ^ 6 + sixteen ^ 18)) ^ (1/2 : ℝ)) = (2 : ℝ) ^ 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equals_two_power_fifteen_l1140_114010


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l1140_114075

/-- Represents the sale of cloth by a shopkeeper -/
structure ClothSale where
  totalSellingPrice : ℕ
  lossPerMetre : ℕ
  costPricePerMetre : ℕ

/-- Calculates the number of metres of cloth sold given the sale details -/
def metresSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMetre - sale.lossPerMetre)

/-- Theorem stating that for the given conditions, the shopkeeper sold 200 metres of cloth -/
theorem shopkeeper_cloth_sale :
  let sale : ClothSale := {
    totalSellingPrice := 12000,
    lossPerMetre := 6,
    costPricePerMetre := 66
  }
  metresSold sale = 200 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l1140_114075


namespace NUMINAMATH_CALUDE_twice_total_credits_l1140_114052

/-- Given the high school credits of three students (Aria, Emily, and Spencer),
    where Emily has 20 credits, Aria has twice as many credits as Emily,
    and Emily has twice as many credits as Spencer,
    prove that twice the total number of credits for all three is 140. -/
theorem twice_total_credits (emily_credits : ℕ) 
  (h1 : emily_credits = 20)
  (h2 : ∃ aria_credits : ℕ, aria_credits = 2 * emily_credits)
  (h3 : ∃ spencer_credits : ℕ, emily_credits = 2 * spencer_credits) :
  2 * (emily_credits + 2 * emily_credits + emily_credits / 2) = 140 :=
by sorry

end NUMINAMATH_CALUDE_twice_total_credits_l1140_114052


namespace NUMINAMATH_CALUDE_phi_range_for_monotonic_interval_l1140_114091

/-- Given a function f(x) = -2 sin(2x + φ) where |φ| < π, 
    if (π/5, 5π/8) is a monotonically increasing interval of f(x),
    then π/10 ≤ φ ≤ π/4 -/
theorem phi_range_for_monotonic_interval (φ : Real) :
  (|φ| < π) →
  (∀ x₁ x₂, π/5 < x₁ ∧ x₁ < x₂ ∧ x₂ < 5*π/8 → 
    (-2 * Real.sin (2*x₁ + φ)) < (-2 * Real.sin (2*x₂ + φ))) →
  π/10 ≤ φ ∧ φ ≤ π/4 := by
  sorry

end NUMINAMATH_CALUDE_phi_range_for_monotonic_interval_l1140_114091


namespace NUMINAMATH_CALUDE_greatest_common_divisor_360_90_under_60_l1140_114046

theorem greatest_common_divisor_360_90_under_60 : 
  ∃ (n : ℕ), n = 30 ∧ 
  n ∣ 360 ∧ 
  n < 60 ∧ 
  n ∣ 90 ∧
  ∀ (m : ℕ), m ∣ 360 → m < 60 → m ∣ 90 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_360_90_under_60_l1140_114046


namespace NUMINAMATH_CALUDE_dance_team_size_l1140_114033

theorem dance_team_size (initial_size : ℕ) (quit : ℕ) (new_members : ℕ) : 
  initial_size = 25 → quit = 8 → new_members = 13 → 
  initial_size - quit + new_members = 30 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_size_l1140_114033


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l1140_114094

theorem fractional_equation_simplification (x : ℝ) :
  (x / (2 * x - 1) - 3 = 2 / (1 - 2 * x)) ↔ (x - 3 * (2 * x - 1) = -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l1140_114094


namespace NUMINAMATH_CALUDE_inequality_proof_l1140_114053

theorem inequality_proof (x : ℝ) : 
  -7 < x ∧ x < -0.775 → (x + Real.sqrt 3) / (x + 10) > (3*x + 2*Real.sqrt 3) / (2*x + 14) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1140_114053


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1140_114065

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (4, x + 1)
  parallel a b → x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1140_114065


namespace NUMINAMATH_CALUDE_sum_of_seventh_terms_l1140_114056

/-- Given two arithmetic sequences a and b, prove that a₇ + b₇ = 8 -/
theorem sum_of_seventh_terms (a b : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ d : ℝ, a (n + 1) - a n = d) →  -- a is an arithmetic sequence
  (∀ n : ℕ, ∃ d : ℝ, b (n + 1) - b n = d) →  -- b is an arithmetic sequence
  a 2 + b 2 = 3 →                            -- given condition
  a 4 + b 4 = 5 →                            -- given condition
  a 7 + b 7 = 8 :=                           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_sum_of_seventh_terms_l1140_114056


namespace NUMINAMATH_CALUDE_initial_column_size_l1140_114049

/-- The number of people in each column initially -/
def people_per_column : ℕ := 30

/-- The total number of people -/
def total_people : ℕ := people_per_column * 16

/-- The number of columns formed when 48 people stand in each column -/
def columns_with_48 : ℕ := total_people / 48

theorem initial_column_size :
  (total_people = people_per_column * 16) ∧
  (total_people = 48 * 10) ∧
  (columns_with_48 = 10) →
  people_per_column = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_column_size_l1140_114049


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l1140_114072

theorem largest_multiple_of_9_less_than_100 : 
  ∀ n : ℕ, n * 9 < 100 → n * 9 ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l1140_114072


namespace NUMINAMATH_CALUDE_opposite_to_silver_is_pink_l1140_114058

-- Define the colors
inductive Color
  | Pink
  | Teal
  | Maroon
  | Lilac
  | Silver
  | Crimson

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : List Face
  hinged : List (Face × Face)

-- Define the property of opposite faces
def areOpposite (c : Cube) (f1 f2 : Face) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1 ≠ f2

-- State the theorem
theorem opposite_to_silver_is_pink (c : Cube) :
  (∃ f1 f2 : Face, f1.color = Color.Silver ∧ f2.color = Color.Pink ∧ areOpposite c f1 f2) :=
by sorry

end NUMINAMATH_CALUDE_opposite_to_silver_is_pink_l1140_114058


namespace NUMINAMATH_CALUDE_table_satisfies_conditions_l1140_114084

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_consecutive_prime_product (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ q = p + 2 ∧ n = p * q

def table : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![2, 1, 8, 7],
    ![7, 3, 8, 7],
    ![7, 7, 4, 4],
    ![7, 8, 4, 4]]

theorem table_satisfies_conditions :
  (∀ i j, table i j < 10) ∧
  (∀ i, table i 0 ≠ 0) ∧
  (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ 
    1000 * table 0 0 + 100 * table 0 1 + 10 * table 0 2 + table 0 3 = p^q) ∧
  (is_consecutive_prime_product 
    (1000 * table 1 0 + 100 * table 1 1 + 10 * table 1 2 + table 1 3)) ∧
  (is_perfect_square 
    (1000 * table 2 0 + 100 * table 2 1 + 10 * table 2 2 + table 2 3)) ∧
  ((1000 * table 3 0 + 100 * table 3 1 + 10 * table 3 2 + table 3 3) % 37 = 0) :=
by sorry

end NUMINAMATH_CALUDE_table_satisfies_conditions_l1140_114084


namespace NUMINAMATH_CALUDE_average_page_count_l1140_114062

theorem average_page_count (total_students : ℕ) 
  (group1_count group2_count group3_count group4_count : ℕ)
  (group1_pages group2_pages group3_pages group4_pages : ℕ) : 
  total_students = 30 →
  group1_count = 8 →
  group2_count = 10 →
  group3_count = 7 →
  group4_count = 5 →
  group1_pages = 3 →
  group2_pages = 5 →
  group3_pages = 2 →
  group4_pages = 4 →
  (group1_count * group1_pages + 
   group2_count * group2_pages + 
   group3_count * group3_pages + 
   group4_count * group4_pages : ℚ) / total_students = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_average_page_count_l1140_114062


namespace NUMINAMATH_CALUDE_problem_statement_l1140_114081

theorem problem_statement : (2025^2 - 2025) / 2025 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1140_114081


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1140_114025

theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (-1, 3) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  θ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1140_114025


namespace NUMINAMATH_CALUDE_single_point_conic_section_l1140_114043

theorem single_point_conic_section (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 6 * p.2 + d = 0) → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_single_point_conic_section_l1140_114043


namespace NUMINAMATH_CALUDE_not_right_triangle_when_A_eq_B_eq_3C_l1140_114068

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Theorem statement
theorem not_right_triangle_when_A_eq_B_eq_3C (t : Triangle) 
  (h : t.A = t.B ∧ t.A = 3 * t.C) : 
  ¬ is_right_triangle t := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_when_A_eq_B_eq_3C_l1140_114068


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l1140_114047

theorem smallest_prime_factor_of_1729 :
  (Nat.minFac 1729 = 7) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l1140_114047


namespace NUMINAMATH_CALUDE_prime_sum_112_l1140_114007

theorem prime_sum_112 :
  ∃ (S : Finset Nat), 
    (∀ p ∈ S, Nat.Prime p ∧ p > 10) ∧ 
    (S.sum id = 112) ∧ 
    (S.card = 6) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_112_l1140_114007


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l1140_114055

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem distance_between_complex_points :
  let z1 : ℂ := 7 - 4*I
  let z2 : ℂ := 2 + 8*I
  let A : ℝ × ℝ := complex_to_point z1
  let B : ℝ × ℝ := complex_to_point z2
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l1140_114055


namespace NUMINAMATH_CALUDE_no_valid_operation_no_standard_op_satisfies_equation_l1140_114013

-- Define the type for standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the arithmetic operation
def applyOp (op : ArithOp) (a b : Int) : Int :=
  match op with
  | ArithOp.Add => a + b
  | ArithOp.Sub => a - b
  | ArithOp.Mul => a * b
  | ArithOp.Div => a / b

-- Theorem statement
theorem no_valid_operation :
  ∀ (op : ArithOp), (applyOp op 8 4) + 5 - (3 - 2) ≠ 4 := by
  sorry

-- Main theorem that proves no standard operation satisfies the equation
theorem no_standard_op_satisfies_equation :
  ¬ (∃ (op : ArithOp), (applyOp op 8 4) + 5 - (3 - 2) = 4) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_operation_no_standard_op_satisfies_equation_l1140_114013


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1140_114029

theorem solution_set_inequality (x : ℝ) :
  (1 / (x - 1) ≥ -1) ↔ (x ≤ 0 ∨ x > 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1140_114029


namespace NUMINAMATH_CALUDE_a_ge_one_l1140_114006

open Real

/-- The function f(x) = a * ln(x) + (1/2) * x^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + (1/2) * x^2

/-- Theorem stating that if f satisfies the given condition, then a ≥ 1 -/
theorem a_ge_one (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 2) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_a_ge_one_l1140_114006


namespace NUMINAMATH_CALUDE_special_triangle_properties_l1140_114041

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 4 * t.b * Real.cos t.B ∧
  t.b = 2 * Real.sqrt 19 ∧
  (1 / 2) * t.a * t.b * Real.sin t.C = 6 * Real.sqrt 15

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.cos t.B = 1 / 4 ∧ t.a + t.b + t.c = 14 + 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l1140_114041


namespace NUMINAMATH_CALUDE_final_number_is_fifty_l1140_114039

/-- Represents the state of the board at any given time -/
structure BoardState where
  ones : Nat
  fours : Nat
  others : List Nat

/-- The operation of replacing two numbers with their Pythagorean sum -/
def replaceTwo (x y : Nat) : Nat :=
  Nat.sqrt (x^2 + y^2)

/-- The process of reducing the board until only one number remains -/
def reduceBoard : BoardState → Nat
| s => if s.ones + s.fours + s.others.length = 1
       then if s.ones = 1 then 1
            else if s.fours = 1 then 4
            else s.others.head!
       else sorry -- recursively apply replaceTwo

theorem final_number_is_fifty :
  ∀ (finalNum : Nat),
  (∃ (s : BoardState), s.ones = 900 ∧ s.fours = 100 ∧ s.others = [] ∧
   reduceBoard s = finalNum) →
  finalNum = 50 := by
  sorry

#check final_number_is_fifty

end NUMINAMATH_CALUDE_final_number_is_fifty_l1140_114039


namespace NUMINAMATH_CALUDE_only_set_A_is_right_triangle_l1140_114038

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of numbers
def set_A : List ℕ := [5, 12, 13]
def set_B : List ℕ := [3, 4, 6]
def set_C : List ℕ := [4, 5, 6]
def set_D : List ℕ := [5, 7, 9]

-- Theorem to prove
theorem only_set_A_is_right_triangle :
  (is_right_triangle 5 12 13) ∧
  ¬(is_right_triangle 3 4 6) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 7 9) :=
sorry

end NUMINAMATH_CALUDE_only_set_A_is_right_triangle_l1140_114038


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l1140_114005

-- Define the complex number z
def z (t : ℝ) : ℂ := 9 + t * Complex.I

-- State the theorem
theorem complex_modulus_equation (t : ℝ) :
  (t > 0 ∧ Complex.abs (z t) = 15) → t = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l1140_114005


namespace NUMINAMATH_CALUDE_shot_cost_calculation_l1140_114095

def total_shot_cost (num_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) : ℕ :=
  num_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot

theorem shot_cost_calculation :
  total_shot_cost 3 4 2 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_shot_cost_calculation_l1140_114095


namespace NUMINAMATH_CALUDE_sector_central_angle_l1140_114028

/-- Given a sector with circumference 10 and area 4, prove that its central angle is π/2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 10) (h2 : (1/2) * l * r = 4) :
  l / r = 1/2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1140_114028


namespace NUMINAMATH_CALUDE_bob_improvement_percentage_l1140_114034

/-- The percentage improvement needed to match a target time -/
def percentage_improvement (current_time target_time : ℕ) : ℚ :=
  (current_time - target_time : ℚ) / current_time * 100

/-- Bob's current mile time in seconds -/
def bob_time : ℕ := 640

/-- Bob's sister's mile time in seconds -/
def sister_time : ℕ := 320

/-- Theorem: Bob needs to improve his time by 50% to match his sister's time -/
theorem bob_improvement_percentage :
  percentage_improvement bob_time sister_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_bob_improvement_percentage_l1140_114034


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1140_114071

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) / n = 144 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1140_114071


namespace NUMINAMATH_CALUDE_weight_of_single_pencil_l1140_114018

/-- The weight of a single pencil given the weight of a dozen pencils -/
theorem weight_of_single_pencil (dozen_weight : ℝ) (h : dozen_weight = 182.88) :
  dozen_weight / 12 = 15.24 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_single_pencil_l1140_114018


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1140_114026

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and downstream travel information. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 168)
  (h3 : downstream_time = 6)
  : ∃ (boat_speed : ℝ), boat_speed = 24 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1140_114026


namespace NUMINAMATH_CALUDE_regular_polyhedron_spheres_l1140_114012

/-- A regular polyhedron -/
structure RegularPolyhedron where
  -- We don't need to define the internal structure,
  -- as the problem doesn't rely on specific properties

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points -/
def distance (p q : Point3D) : ℝ :=
  sorry

/-- Distance from a point to a face of the polyhedron -/
def distanceToFace (p : Point3D) (poly : RegularPolyhedron) (face : Nat) : ℝ :=
  sorry

/-- Get a vertex of the polyhedron -/
def getVertex (poly : RegularPolyhedron) (v : Nat) : Point3D :=
  sorry

/-- Number of vertices in the polyhedron -/
def numVertices (poly : RegularPolyhedron) : Nat :=
  sorry

/-- Number of faces in the polyhedron -/
def numFaces (poly : RegularPolyhedron) : Nat :=
  sorry

/-- Theorem: For any regular polyhedron, there exists a point O such that
    1) The distance from O to all vertices is constant
    2) The distance from O to all faces is constant -/
theorem regular_polyhedron_spheres (poly : RegularPolyhedron) :
  ∃ (O : Point3D),
    (∀ (i j : Nat), i < numVertices poly → j < numVertices poly →
      distance O (getVertex poly i) = distance O (getVertex poly j)) ∧
    (∀ (i j : Nat), i < numFaces poly → j < numFaces poly →
      distanceToFace O poly i = distanceToFace O poly j) :=
by sorry

end NUMINAMATH_CALUDE_regular_polyhedron_spheres_l1140_114012


namespace NUMINAMATH_CALUDE_find_a_min_value_g_l1140_114017

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the function g
def g (x : ℝ) : ℝ := f 2 x - |x + 1|

-- Theorem for part (I)
theorem find_a : 
  (∀ x, f 2 x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) → 
  (∃! a : ℝ, ∀ x, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧ 
  (∀ x, f 2 x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) :=
sorry

-- Theorem for part (II)
theorem min_value_g : 
  IsLeast {y | ∃ x, g x = y} (-1/2) :=
sorry

end NUMINAMATH_CALUDE_find_a_min_value_g_l1140_114017


namespace NUMINAMATH_CALUDE_derivative_of_f_tangent_line_at_one_l1140_114022

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x * Real.log x

-- State the theorem for the derivative of f
theorem derivative_of_f :
  deriv f = fun x => 2 * x + Real.log x + 1 :=
sorry

-- Define the tangent line function
def tangent_line (x y : ℝ) : ℝ := 3 * x - y - 2

-- State the theorem for the tangent line at x=1
theorem tangent_line_at_one :
  ∀ x y, f 1 = y → deriv f 1 * (x - 1) + y = tangent_line x y :=
sorry

end NUMINAMATH_CALUDE_derivative_of_f_tangent_line_at_one_l1140_114022


namespace NUMINAMATH_CALUDE_sugar_percentage_after_addition_l1140_114044

def initial_volume : ℝ := 340
def water_percentage : ℝ := 0.75
def kola_percentage : ℝ := 0.05
def added_sugar : ℝ := 3.2
def added_water : ℝ := 12
def added_kola : ℝ := 6.8

theorem sugar_percentage_after_addition :
  let initial_sugar_percentage : ℝ := 1 - water_percentage - kola_percentage
  let initial_sugar_volume : ℝ := initial_sugar_percentage * initial_volume
  let final_sugar_volume : ℝ := initial_sugar_volume + added_sugar
  let final_volume : ℝ := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_percentage : ℝ := final_sugar_volume / final_volume
  ∃ ε > 0, |final_sugar_percentage - 0.1967| < ε :=
by sorry

end NUMINAMATH_CALUDE_sugar_percentage_after_addition_l1140_114044


namespace NUMINAMATH_CALUDE_original_price_correct_l1140_114089

/-- The original price of a single article before discounts and taxes -/
def original_price : ℝ := 669.99

/-- The discount rate for purchases of 2 or more articles -/
def discount_rate : ℝ := 0.24

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.08

/-- The number of articles purchased -/
def num_articles : ℕ := 3

/-- The total cost after discount and tax -/
def total_cost : ℝ := 1649.43

/-- Theorem stating that the original price is correct given the conditions -/
theorem original_price_correct : 
  num_articles * (original_price * (1 - discount_rate)) * (1 + sales_tax_rate) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_original_price_correct_l1140_114089


namespace NUMINAMATH_CALUDE_sum_interior_angles_increases_l1140_114070

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles increases as the number of sides increases from 3 to n -/
theorem sum_interior_angles_increases (n : ℕ) (h : n > 3) :
  sum_interior_angles n > sum_interior_angles 3 := by
  sorry


end NUMINAMATH_CALUDE_sum_interior_angles_increases_l1140_114070


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1140_114085

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 + 24 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1140_114085


namespace NUMINAMATH_CALUDE_female_height_calculation_l1140_114077

theorem female_height_calculation (total_avg : ℝ) (male_avg : ℝ) (ratio : ℝ) 
  (h1 : total_avg = 180)
  (h2 : male_avg = 185)
  (h3 : ratio = 2) :
  ∃ female_avg : ℝ, female_avg = 170 ∧ 
  (ratio * female_avg + male_avg) / (ratio + 1) = total_avg :=
by sorry

end NUMINAMATH_CALUDE_female_height_calculation_l1140_114077


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1140_114087

def U : Set Nat := {0, 1, 2, 3, 5, 6, 8}
def A : Set Nat := {1, 5, 8}
def B : Set Nat := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1140_114087


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l1140_114004

/-- Given a square and circle intersecting such that each side of the square contains a chord
    of the circle, and each chord is twice the length of the radius of the circle,
    the ratio of the area of the square to the area of the circle is (7 - 4√3) / π. -/
theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := r * (2 - Real.sqrt 3)
  (s^2) / (π * r^2) = (7 - 4 * Real.sqrt 3) / π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l1140_114004


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l1140_114014

-- Define the function f
def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + x + c

-- Define the property of being an odd function on an interval
def is_odd_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ a + b = 0

-- Theorem statement
theorem odd_function_sum_zero (a b c : ℝ) :
  is_odd_on (f a c) a b → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l1140_114014


namespace NUMINAMATH_CALUDE_plot_length_proof_l1140_114015

/-- Given a rectangular plot with width 50 meters, prove that if 56 poles
    are needed when placed 5 meters apart along the perimeter,
    then the length of the plot is 80 meters. -/
theorem plot_length_proof (width : ℝ) (num_poles : ℕ) (pole_distance : ℝ) (length : ℝ) :
  width = 50 →
  num_poles = 56 →
  pole_distance = 5 →
  2 * ((length / pole_distance) + 1) + 2 * ((width / pole_distance) + 1) = num_poles →
  length = 80 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_proof_l1140_114015


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1140_114067

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 45 → b = 60 → c^2 = a^2 + b^2 → c = 75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1140_114067


namespace NUMINAMATH_CALUDE_prob_at_least_8_stay_correct_l1140_114086

def total_people : ℕ := 10
def certain_people : ℕ := 5
def uncertain_people : ℕ := 5
def uncertain_stay_prob : ℚ := 3/7

def prob_at_least_8_stay : ℚ := 4563/16807

theorem prob_at_least_8_stay_correct :
  let prob_8_stay := (uncertain_people.choose 3) * (uncertain_stay_prob^3 * (1 - uncertain_stay_prob)^2)
  let prob_10_stay := uncertain_stay_prob^uncertain_people
  prob_at_least_8_stay = prob_8_stay + prob_10_stay :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_8_stay_correct_l1140_114086


namespace NUMINAMATH_CALUDE_third_team_pieces_l1140_114093

theorem third_team_pieces (total : ℕ) (first_team : ℕ) (second_team : ℕ) 
  (h1 : total = 500) 
  (h2 : first_team = 189) 
  (h3 : second_team = 131) : 
  total - (first_team + second_team) = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_third_team_pieces_l1140_114093


namespace NUMINAMATH_CALUDE_partition_fifth_power_l1140_114040

/-- Number of partitions of a 1 × n rectangle into 1 × 1 squares and broken dominoes -/
def F (n : ℕ) : ℕ :=
  sorry

/-- A broken domino consists of two 1 × 1 squares separated by four squares -/
def is_broken_domino (tile : List (ℕ × ℕ)) : Prop :=
  tile.length = 2 ∧ ∃ i : ℕ, tile = [(i, 1), (i + 5, 1)]

/-- A valid tiling of a 1 × n rectangle -/
def valid_tiling (n : ℕ) (tiling : List (List (ℕ × ℕ))) : Prop :=
  (tiling.join.map Prod.fst).toFinset = Finset.range n ∧
  ∀ tile ∈ tiling, tile.length = 1 ∨ is_broken_domino tile

theorem partition_fifth_power (n : ℕ) :
  (F (5 * n) : ℕ) = (F n) ^ 5 :=
sorry

end NUMINAMATH_CALUDE_partition_fifth_power_l1140_114040


namespace NUMINAMATH_CALUDE_johannes_earnings_today_l1140_114079

/-- Represents the earnings and sales of a vegetable shop owner over three days -/
structure VegetableShopEarnings where
  cabbage_price : ℝ
  wednesday_earnings : ℝ
  friday_earnings : ℝ
  total_cabbage_sold : ℝ

/-- Calculates the earnings for today given the total earnings and previous days' earnings -/
def earnings_today (shop : VegetableShopEarnings) : ℝ :=
  shop.cabbage_price * shop.total_cabbage_sold - (shop.wednesday_earnings + shop.friday_earnings)

/-- Theorem stating that given the specific conditions, Johannes earned $42 today -/
theorem johannes_earnings_today :
  let shop : VegetableShopEarnings := {
    cabbage_price := 2,
    wednesday_earnings := 30,
    friday_earnings := 24,
    total_cabbage_sold := 48
  }
  earnings_today shop = 42 := by sorry

end NUMINAMATH_CALUDE_johannes_earnings_today_l1140_114079


namespace NUMINAMATH_CALUDE_sequence_a_property_l1140_114054

def sequence_a (n : ℕ) : ℚ :=
  3 / (15 * n - 14)

theorem sequence_a_property :
  (sequence_a 1 = 3) ∧
  (∀ n : ℕ, n > 0 → 1 / (sequence_a (n + 1) + 1) - 1 / (sequence_a n) = 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_property_l1140_114054


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1140_114090

/-- Simple interest calculation -/
theorem simple_interest_rate
  (principal : ℝ)
  (time : ℝ)
  (interest : ℝ)
  (h1 : principal = 10000)
  (h2 : time = 1)
  (h3 : interest = 900) :
  (interest / (principal * time)) * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1140_114090


namespace NUMINAMATH_CALUDE_correct_number_of_pretzels_l1140_114045

/-- The number of pretzels in Mille's snack packs. -/
def pretzels : ℕ := 64

/-- The number of kids in the class. -/
def kids : ℕ := 16

/-- The number of items in each baggie. -/
def items_per_baggie : ℕ := 22

/-- The number of suckers. -/
def suckers : ℕ := 32

/-- Theorem stating that the number of pretzels is correct given the conditions. -/
theorem correct_number_of_pretzels :
  pretzels * 5 + suckers = kids * items_per_baggie :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_pretzels_l1140_114045


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_abs_coeff_l1140_114035

theorem binomial_expansion_sum_abs_coeff :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 32 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_abs_coeff_l1140_114035


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l1140_114097

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l1140_114097


namespace NUMINAMATH_CALUDE_complex_equality_implies_power_l1140_114066

/-- Given complex numbers z₁ and z₂, where z₁ = -1 + 3i and z₂ = a + bi³,
    if z₁ = z₂, then b^a = -1/3 -/
theorem complex_equality_implies_power (a b : ℝ) :
  let z₁ : ℂ := -1 + 3 * Complex.I
  let z₂ : ℂ := a + b * Complex.I^3
  z₁ = z₂ → b^a = -1/3 := by sorry

end NUMINAMATH_CALUDE_complex_equality_implies_power_l1140_114066


namespace NUMINAMATH_CALUDE_circumscribed_circle_equation_l1140_114096

/-- The equation of a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- A triangle defined by three lines -/
structure Triangle where
  side1 : Line
  side2 : Line
  side3 : Line

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The circumscribed circle of a triangle -/
def circumscribedCircle (t : Triangle) : Circle := sorry

/-- Theorem: The circumscribed circle of the given triangle has the equation (x - 2)^2 + (y - 2)^2 = 25 -/
theorem circumscribed_circle_equation (t : Triangle) 
  (h1 : t.side1 = ⟨1, 1⟩) 
  (h2 : t.side2 = ⟨-1/2, -2⟩) 
  (h3 : t.side3 = ⟨3, -9⟩) : 
  circumscribedCircle t = ⟨2, 2, 5⟩ := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_equation_l1140_114096


namespace NUMINAMATH_CALUDE_line_through_points_l1140_114032

/-- A line passing through (0, -2) and (1, 0) also passes through (7, b). Prove b = 12. -/
theorem line_through_points (b : ℝ) : 
  (∃ m c : ℝ, (0 = m * 0 + c ∧ -2 = m * 0 + c) ∧ 
              (0 = m * 1 + c) ∧ 
              (b = m * 7 + c)) → 
  b = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1140_114032


namespace NUMINAMATH_CALUDE_certain_value_proof_l1140_114048

theorem certain_value_proof (x : ℤ) : 
  (∀ n : ℤ, 101 * n^2 ≤ x → n ≤ 10) ∧ 
  (∃ n : ℤ, n = 10 ∧ 101 * n^2 ≤ x) →
  x = 10100 :=
by sorry

end NUMINAMATH_CALUDE_certain_value_proof_l1140_114048


namespace NUMINAMATH_CALUDE_count_m_gons_correct_l1140_114011

/-- Given integers m and n where 4 < m < n, and a regular polygon with 2n+1 sides,
    this function computes the number of convex m-gons with vertices from the polygon's vertices
    and exactly two acute interior angles. -/
def count_m_gons (m n : ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

/-- Theorem stating that count_m_gons correctly computes the number of m-gons
    satisfying the given conditions. -/
theorem count_m_gons_correct (m n : ℕ) (h1 : 4 < m) (h2 : m < n) :
  count_m_gons m n = (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
by sorry

end NUMINAMATH_CALUDE_count_m_gons_correct_l1140_114011


namespace NUMINAMATH_CALUDE_special_set_properties_l1140_114019

/-- A set M with specific closure properties -/
structure SpecialSet (M : Set ℝ) : Prop where
  zero_in : (0 : ℝ) ∈ M
  one_in : (1 : ℝ) ∈ M
  closed_sub : ∀ x y, x ∈ M → y ∈ M → (x - y) ∈ M
  closed_inv : ∀ x, x ∈ M → x ≠ 0 → (1 / x) ∈ M

/-- Properties of the special set M -/
theorem special_set_properties (M : Set ℝ) (h : SpecialSet M) :
  (1 / 3 ∈ M) ∧
  (-1 ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x, x ∈ M → x^2 ∈ M) := by
  sorry

end NUMINAMATH_CALUDE_special_set_properties_l1140_114019


namespace NUMINAMATH_CALUDE_rocket_coaster_capacity_l1140_114069

/-- Represents a roller coaster with two types of cars -/
structure RollerCoaster where
  total_cars : ℕ
  four_passenger_cars : ℕ
  six_passenger_cars : ℕ

/-- Calculates the total capacity of a roller coaster -/
def total_capacity (rc : RollerCoaster) : ℕ :=
  rc.four_passenger_cars * 4 + rc.six_passenger_cars * 6

/-- The Rocket Coaster specification -/
def rocket_coaster : RollerCoaster := {
  total_cars := 15,
  four_passenger_cars := 9,
  six_passenger_cars := 15 - 9
}

theorem rocket_coaster_capacity :
  total_capacity rocket_coaster = 72 := by
  sorry

end NUMINAMATH_CALUDE_rocket_coaster_capacity_l1140_114069


namespace NUMINAMATH_CALUDE_smallest_set_size_for_divisibility_by_20_l1140_114003

theorem smallest_set_size_for_divisibility_by_20 :
  ∃ (n : ℕ), n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T →
    a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d ∨
    ¬(20 ∣ (a + b - c - d))) :=
by
  sorry

#check smallest_set_size_for_divisibility_by_20

end NUMINAMATH_CALUDE_smallest_set_size_for_divisibility_by_20_l1140_114003


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l1140_114008

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17) 
  (h2 : c + a = 20) 
  (h3 : a + b = 23) : 
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l1140_114008


namespace NUMINAMATH_CALUDE_derivative_of_y_l1140_114099

noncomputable def y (x : ℝ) : ℝ := x + Real.cos x

theorem derivative_of_y (x : ℝ) : 
  deriv y x = 1 - Real.sin x := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l1140_114099


namespace NUMINAMATH_CALUDE_car_rental_rate_proof_l1140_114051

/-- The daily rate of the first car rental company -/
def first_company_rate : ℝ := 17.99

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.18

/-- The daily rate of City Rentals -/
def city_rentals_rate : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.16

/-- The number of miles at which the cost is the same for both companies -/
def equal_cost_miles : ℝ := 48

theorem car_rental_rate_proof :
  first_company_rate + first_company_per_mile * equal_cost_miles =
  city_rentals_rate + city_rentals_per_mile * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rate_proof_l1140_114051


namespace NUMINAMATH_CALUDE_assignment_statement_valid_l1140_114002

-- Define what constitutes a valid variable name
def IsValidVariableName (name : String) : Prop := name.length > 0 ∧ name.all Char.isAlpha

-- Define what constitutes a valid arithmetic expression
inductive ArithmeticExpression
  | Var : String → ArithmeticExpression
  | Num : Int → ArithmeticExpression
  | Add : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression
  | Mul : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression
  | Sub : ArithmeticExpression → ArithmeticExpression → ArithmeticExpression

-- Define what constitutes a valid assignment statement
structure AssignmentStatement where
  lhs : String
  rhs : ArithmeticExpression
  valid : IsValidVariableName lhs

-- The statement we want to prove
theorem assignment_statement_valid :
  ∃ (stmt : AssignmentStatement),
    stmt.lhs = "A" ∧
    stmt.rhs = ArithmeticExpression.Sub
      (ArithmeticExpression.Add
        (ArithmeticExpression.Mul
          (ArithmeticExpression.Var "A")
          (ArithmeticExpression.Var "A"))
        (ArithmeticExpression.Var "A"))
      (ArithmeticExpression.Num 3) :=
by sorry


end NUMINAMATH_CALUDE_assignment_statement_valid_l1140_114002


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l1140_114009

theorem geometric_arithmetic_progression_problem :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ * c₁ = b₁^2 ∧ a₁ + c₁ = 2*(b₁ + 8) ∧ a₁ * (c₁ + 64) = (b₁ + 8)^2) ∧
    (a₂ * c₂ = b₂^2 ∧ a₂ + c₂ = 2*(b₂ + 8) ∧ a₂ * (c₂ + 64) = (b₂ + 8)^2) ∧
    (a₁ = 4/9 ∧ b₁ = -20/9 ∧ c₁ = 100/9) ∧
    (a₂ = 4 ∧ b₂ = 12 ∧ c₂ = 36) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_problem_l1140_114009


namespace NUMINAMATH_CALUDE_m_range_l1140_114016

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem m_range (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1140_114016


namespace NUMINAMATH_CALUDE_karen_start_time_l1140_114036

/-- Proves that Karen starts 4 minutes late in the car race with Tom -/
theorem karen_start_time (karen_speed tom_speed tom_distance karen_win_margin : ℝ) 
  (h1 : karen_speed = 60) 
  (h2 : tom_speed = 45)
  (h3 : tom_distance = 24)
  (h4 : karen_win_margin = 4) : 
  (tom_distance / tom_speed - (tom_distance + karen_win_margin) / karen_speed) * 60 = 4 := by
  sorry


end NUMINAMATH_CALUDE_karen_start_time_l1140_114036


namespace NUMINAMATH_CALUDE_vector_relation_l1140_114030

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define points A, B, and C
variable (A B C : V)

-- Define the theorem
theorem vector_relation (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) 
                        (h2 : C - A = (3/5) • (B - A)) : 
  C - A = -(3/2) • (C - B) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_l1140_114030


namespace NUMINAMATH_CALUDE_k_h_negative_three_equals_fifteen_l1140_114020

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x^2 - 12

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_negative_three_equals_fifteen
  (h_def : ∀ x, h x = 5 * x^2 - 12)
  (k_h_three : k (h 3) = 15) :
  k (h (-3)) = 15 := by
sorry

end NUMINAMATH_CALUDE_k_h_negative_three_equals_fifteen_l1140_114020


namespace NUMINAMATH_CALUDE_entertainment_expense_calculation_l1140_114021

def entertainment_expense (initial_amount : ℝ) (food_percentage : ℝ) (phone_percentage : ℝ) (final_amount : ℝ) : ℝ :=
  let food_expense := initial_amount * food_percentage
  let after_food := initial_amount - food_expense
  let phone_expense := after_food * phone_percentage
  let after_phone := after_food - phone_expense
  after_phone - final_amount

theorem entertainment_expense_calculation :
  entertainment_expense 200 0.60 0.25 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_entertainment_expense_calculation_l1140_114021


namespace NUMINAMATH_CALUDE_symmetric_axis_of_quadratic_function_l1140_114088

/-- The symmetric axis of a quadratic function -/
def symmetric_axis (f : ℝ → ℝ) : ℝ := sorry

/-- A quadratic function in factored form -/
def quadratic_function (a b : ℝ) : ℝ → ℝ := λ x ↦ (x - a) * (x + b)

theorem symmetric_axis_of_quadratic_function :
  ∀ (f : ℝ → ℝ), f = quadratic_function 3 5 →
  symmetric_axis f = -1 := by sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_quadratic_function_l1140_114088


namespace NUMINAMATH_CALUDE_sqrt_product_equals_two_l1140_114000

theorem sqrt_product_equals_two : Real.sqrt (2/3) * Real.sqrt 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_two_l1140_114000


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1140_114050

theorem modular_congruence_solution :
  ∀ m n : ℕ,
  0 ≤ m ∧ m ≤ 17 →
  0 ≤ n ∧ n ≤ 13 →
  m ≡ 98765 [MOD 18] →
  n ≡ 98765 [MOD 14] →
  m = 17 ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1140_114050


namespace NUMINAMATH_CALUDE_program_output_correct_verify_output_l1140_114061

/-- Represents the result of the program execution -/
structure ProgramResult where
  x : Int
  y : Int

/-- Executes the program logic based on initial values -/
def executeProgram (initialX initialY : Int) : ProgramResult :=
  if initialX < 0 then
    { x := initialY - 4, y := initialY }
  else
    { x := initialX, y := initialY + 4 }

/-- Theorem stating the program output for given initial values -/
theorem program_output_correct :
  let result := executeProgram 2 (-30)
  result.x - result.y = 28 ∧ result.y - result.x = -28 := by
  sorry

/-- Verifies that the program output matches the expected result -/
theorem verify_output :
  let result := executeProgram 2 (-30)
  (result.x - result.y, result.y - result.x) = (28, -28) := by
  sorry

end NUMINAMATH_CALUDE_program_output_correct_verify_output_l1140_114061


namespace NUMINAMATH_CALUDE_jeremy_earnings_l1140_114024

theorem jeremy_earnings (steven_rate mark_rate steven_rooms mark_rooms : ℚ) 
  (h1 : steven_rate = 12 / 3)
  (h2 : mark_rate = 10 / 4)
  (h3 : steven_rooms = 8 / 3)
  (h4 : mark_rooms = 9 / 4) :
  steven_rate * steven_rooms + mark_rate * mark_rooms = 391 / 24 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_earnings_l1140_114024


namespace NUMINAMATH_CALUDE_jake_weight_loss_l1140_114064

/-- Jake needs to lose weight to weigh twice as much as his sister. -/
theorem jake_weight_loss (total_weight sister_weight jake_weight : ℕ) 
  (h1 : total_weight = 153)
  (h2 : jake_weight = 113)
  (h3 : total_weight = sister_weight + jake_weight) :
  jake_weight - 2 * sister_weight = 33 := by
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l1140_114064


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1140_114083

/-- Given a hyperbola with equation x²/4 - y² = 1, prove its transverse axis length and asymptote equations -/
theorem hyperbola_properties :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/4 - y^2 = 1}
  ∃ (transverse_axis_length : ℝ) (asymptote_slope : ℝ),
    transverse_axis_length = 4 ∧
    asymptote_slope = 1/2 ∧
    (∀ (x y : ℝ), (x, y) ∈ hyperbola → 
      (y = asymptote_slope * x ∨ y = -asymptote_slope * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1140_114083


namespace NUMINAMATH_CALUDE_nancy_purchase_cost_l1140_114076

/-- The cost of a set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of a set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of crystal bead sets purchased -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets purchased -/
def metal_sets : ℕ := 2

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := crystal_cost * crystal_sets + metal_cost * metal_sets

theorem nancy_purchase_cost : total_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_nancy_purchase_cost_l1140_114076
