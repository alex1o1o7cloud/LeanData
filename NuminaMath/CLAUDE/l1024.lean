import Mathlib

namespace NUMINAMATH_CALUDE_correct_calculation_l1024_102491

theorem correct_calculation : 
  (-2 - 3 = -5) ∧ 
  (-3^2 ≠ -6) ∧ 
  (1/2 / 2 ≠ 2 * 2) ∧ 
  ((-2/3)^2 ≠ 4/3) := by
sorry

end NUMINAMATH_CALUDE_correct_calculation_l1024_102491


namespace NUMINAMATH_CALUDE_mary_warm_hours_l1024_102494

/-- The number of sticks of wood produced by chopping up a chair. -/
def sticksPerChair : ℕ := 6

/-- The number of sticks of wood produced by chopping up a table. -/
def sticksPerTable : ℕ := 9

/-- The number of sticks of wood produced by chopping up a stool. -/
def sticksPerStool : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour to stay warm. -/
def sticksPerHour : ℕ := 5

/-- The number of chairs Mary chops up. -/
def numChairs : ℕ := 18

/-- The number of tables Mary chops up. -/
def numTables : ℕ := 6

/-- The number of stools Mary chops up. -/
def numStools : ℕ := 4

/-- Theorem stating that Mary can keep warm for 34 hours with the firewood from the chopped furniture. -/
theorem mary_warm_hours : 
  (numChairs * sticksPerChair + numTables * sticksPerTable + numStools * sticksPerStool) / sticksPerHour = 34 := by
  sorry


end NUMINAMATH_CALUDE_mary_warm_hours_l1024_102494


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1024_102477

-- Define the original number
def original_number : ℝ := 0.0000084

-- Define the scientific notation components
def significand : ℝ := 8.4
def exponent : ℤ := -6

-- Theorem statement
theorem scientific_notation_proof :
  original_number = significand * (10 : ℝ) ^ exponent :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1024_102477


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1024_102452

theorem cubic_equation_solution (A : ℕ) (a b s : ℤ) 
  (h_A : A = 1 ∨ A = 2 ∨ A = 3)
  (h_coprime : Int.gcd a b = 1)
  (h_eq : a^2 + A * b^2 = s^3) :
  ∃ u v : ℤ, 
    s = u^2 + A * v^2 ∧
    a = u^3 - 3 * A * u * v^2 ∧
    b = 3 * u^2 * v - A * v^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1024_102452


namespace NUMINAMATH_CALUDE_square_inequality_negative_l1024_102473

theorem square_inequality_negative (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_negative_l1024_102473


namespace NUMINAMATH_CALUDE_union_equality_iff_m_range_l1024_102495

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + 2*m < 0}

-- State the theorem
theorem union_equality_iff_m_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (-1/2 ≤ m ∧ m ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_union_equality_iff_m_range_l1024_102495


namespace NUMINAMATH_CALUDE_fourth_root_unity_sum_l1024_102484

/-- Given a nonreal complex number ω that is a fourth root of unity,
    prove that (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 -/
theorem fourth_root_unity_sum (ω : ℂ) 
  (h1 : ω^4 = 1) 
  (h2 : ω ≠ 1 ∧ ω ≠ -1 ∧ ω ≠ Complex.I ∧ ω ≠ -Complex.I) : 
  (1 - ω + ω^3)^4 + (1 + ω - ω^3)^4 = -14 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_unity_sum_l1024_102484


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1024_102454

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_8th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 23)
  (h_6th : a 6 = 47) :
  a 8 = 71 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1024_102454


namespace NUMINAMATH_CALUDE_remainder_172_pow_172_mod_13_l1024_102496

theorem remainder_172_pow_172_mod_13 : 172^172 % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_172_pow_172_mod_13_l1024_102496


namespace NUMINAMATH_CALUDE_optimal_seedlings_optimal_seedlings_count_l1024_102485

/-- Represents the profit per pot as a function of the number of seedlings -/
def profit_per_pot (n : ℕ) : ℝ :=
  n * (5 - 0.5 * (n - 4 : ℝ))

/-- The target profit per pot -/
def target_profit : ℝ := 24

/-- Theorem stating that 6 seedlings per pot achieves the target profit while minimizing costs -/
theorem optimal_seedlings :
  (profit_per_pot 6 = target_profit) ∧
  (∀ m : ℕ, m < 6 → profit_per_pot m < target_profit) ∧
  (∀ m : ℕ, m > 6 → profit_per_pot m ≤ target_profit) :=
sorry

/-- Corollary: 6 is the optimal number of seedlings per pot -/
theorem optimal_seedlings_count : ℕ :=
6

end NUMINAMATH_CALUDE_optimal_seedlings_optimal_seedlings_count_l1024_102485


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1024_102413

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_area_inequality (t : Triangle) :
  area t / (t.a * t.b + t.b * t.c + t.c * t.a) ≤ 1 / (4 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1024_102413


namespace NUMINAMATH_CALUDE_unique_solution_l1024_102446

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

/-- The main theorem stating that -3/4 is the unique solution -/
theorem unique_solution (a : ℝ) (h : a ≠ 0) :
  f a (1 - a) = f a (1 + a) ↔ a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1024_102446


namespace NUMINAMATH_CALUDE_unit_conversions_l1024_102461

-- Define the conversion rates
def kg_per_ton : ℝ := 1000
def sq_dm_per_sq_m : ℝ := 100

-- Define the theorem
theorem unit_conversions :
  (8 : ℝ) + 800 / kg_per_ton = 8.8 ∧
  6.32 * sq_dm_per_sq_m = 632 :=
by sorry

end NUMINAMATH_CALUDE_unit_conversions_l1024_102461


namespace NUMINAMATH_CALUDE_amy_work_schedule_l1024_102453

/-- Amy's work schedule and earnings problem -/
theorem amy_work_schedule (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ)
  (school_weeks : ℕ) (school_earnings : ℕ) :
  summer_weeks = 12 →
  summer_hours_per_week = 40 →
  summer_earnings = 4800 →
  school_weeks = 36 →
  school_earnings = 7200 →
  (school_earnings / (summer_earnings / (summer_weeks * summer_hours_per_week))) / school_weeks = 20 := by
  sorry

#check amy_work_schedule

end NUMINAMATH_CALUDE_amy_work_schedule_l1024_102453


namespace NUMINAMATH_CALUDE_salon_average_customers_l1024_102411

def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

def days_per_week : ℕ := 7

def average_daily_customers : ℚ :=
  (customers_per_day.sum : ℚ) / days_per_week

theorem salon_average_customers :
  average_daily_customers = 13.57 := by
  sorry

end NUMINAMATH_CALUDE_salon_average_customers_l1024_102411


namespace NUMINAMATH_CALUDE_polynomial_equation_sum_l1024_102421

theorem polynomial_equation_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x + c) = x^3 + 5*x^2 - 6*x - 4) → 
  a + b + c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equation_sum_l1024_102421


namespace NUMINAMATH_CALUDE_prime_divisor_ge_11_l1024_102423

def is_valid_digit (d : Nat) : Prop := d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

def all_digits_valid (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem prime_divisor_ge_11 (B : Nat) (h1 : B > 10) (h2 : all_digits_valid B) :
  ∃ p : Nat, p.Prime ∧ p ≥ 11 ∧ p ∣ B :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_ge_11_l1024_102423


namespace NUMINAMATH_CALUDE_blue_red_face_ratio_l1024_102497

theorem blue_red_face_ratio (n : ℕ) (h : n = 13) : 
  let red_area := 6 * n^2
  let total_area := 6 * n^3
  let blue_area := total_area - red_area
  blue_area / red_area = 12 := by sorry

end NUMINAMATH_CALUDE_blue_red_face_ratio_l1024_102497


namespace NUMINAMATH_CALUDE_min_dot_product_of_vectors_l1024_102438

/-- Given plane vectors AC and BD, prove the minimum value of AB · CD -/
theorem min_dot_product_of_vectors (A B C D : ℝ × ℝ) : 
  (C.1 - A.1 = 1 ∧ C.2 - A.2 = 2) →  -- AC = (1, 2)
  (D.1 - B.1 = -2 ∧ D.2 - B.2 = 2) →  -- BD = (-2, 2)
  ∃ (min : ℝ), min = -9/4 ∧ 
    ∀ (AB CD : ℝ × ℝ), 
      AB.1 = B.1 - A.1 ∧ AB.2 = B.2 - A.2 →
      CD.1 = D.1 - C.1 ∧ CD.2 = D.2 - C.2 →
      AB.1 * CD.1 + AB.2 * CD.2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_of_vectors_l1024_102438


namespace NUMINAMATH_CALUDE_front_top_area_ratio_l1024_102407

/-- A rectangular box with given properties -/
structure Box where
  volume : ℝ
  side_area : ℝ
  top_area : ℝ
  front_area : ℝ
  top_side_ratio : ℝ

/-- The theorem stating the ratio of front face area to top face area -/
theorem front_top_area_ratio (b : Box) 
  (h_volume : b.volume = 5184)
  (h_side_area : b.side_area = 288)
  (h_top_side_ratio : b.top_area = 1.5 * b.side_area) :
  b.front_area / b.top_area = 1 / 2 := by
  sorry

#check front_top_area_ratio

end NUMINAMATH_CALUDE_front_top_area_ratio_l1024_102407


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l1024_102475

theorem min_product_of_three_numbers (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  x ≤ 2*y ∧ x ≤ 2*z ∧ y ≤ 2*x ∧ y ≤ 2*z ∧ z ≤ 2*x ∧ z ≤ 2*y →
  x * y * z ≥ 1/32 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l1024_102475


namespace NUMINAMATH_CALUDE_line_perp_plane_iff_perp_all_lines_l1024_102457

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Predicate for a line being perpendicular to a plane -/
def perpendicular_to_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a line being perpendicular to another line -/
def perpendicular_to_line (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line being inside a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Theorem stating the equivalence of a line being perpendicular to a plane
    and being perpendicular to all lines in that plane -/
theorem line_perp_plane_iff_perp_all_lines (l : Line3D) (α : Plane3D) :
  perpendicular_to_plane l α ↔ ∀ m : Line3D, line_in_plane m α → perpendicular_to_line l m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_iff_perp_all_lines_l1024_102457


namespace NUMINAMATH_CALUDE_tournament_probability_l1024_102432

/-- The probability of two specific participants playing each other in a tournament --/
theorem tournament_probability (n : ℕ) (h : n = 26) :
  let total_matches := n - 1
  let total_pairs := n * (n - 1) / 2
  (total_matches : ℚ) / total_pairs = 1 / 13 :=
by sorry

end NUMINAMATH_CALUDE_tournament_probability_l1024_102432


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1024_102439

theorem interest_rate_calculation (total_investment : ℝ) (first_part : ℝ) (second_part_rate : ℝ) (total_interest : ℝ) : 
  total_investment = 3600 →
  first_part = 1800 →
  second_part_rate = 5 →
  total_interest = 144 →
  (first_part * (3 / 100)) + ((total_investment - first_part) * (second_part_rate / 100)) = total_interest :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1024_102439


namespace NUMINAMATH_CALUDE_divisor_with_remainder_one_l1024_102402

theorem divisor_with_remainder_one (n : ℕ) : 
  ∃ k : ℕ, 2^200 - 3 = k * (2^100 - 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_with_remainder_one_l1024_102402


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_distinct_roots_l1024_102464

theorem at_least_one_equation_has_distinct_roots (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_distinct_roots_l1024_102464


namespace NUMINAMATH_CALUDE_volleyball_team_lineup_l1024_102422

/-- The number of players in the volleyball team -/
def total_players : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of twins -/
def num_twins : ℕ := 2

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of valid starting lineups -/
def valid_lineups : ℕ := 9778

theorem volleyball_team_lineup :
  (Nat.choose total_players num_starters) -
  (Nat.choose (total_players - num_triplets) (num_starters - num_triplets)) -
  (Nat.choose (total_players - num_twins) (num_starters - num_twins)) +
  (Nat.choose (total_players - num_triplets - num_twins) (num_starters - num_triplets - num_twins)) =
  valid_lineups :=
sorry

end NUMINAMATH_CALUDE_volleyball_team_lineup_l1024_102422


namespace NUMINAMATH_CALUDE_product_of_decimals_product_of_fractions_l1024_102444

/-- Proves that (-0.4) * (-0.8) * (-1.25) * 2.5 = -1 -/
theorem product_of_decimals : (-0.4) * (-0.8) * (-1.25) * 2.5 = -1 := by
  sorry

/-- Proves that (-5/8) * (3/14) * (-16/5) * (-7/6) = -1/2 -/
theorem product_of_fractions : (-5/8 : ℚ) * (3/14 : ℚ) * (-16/5 : ℚ) * (-7/6 : ℚ) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_product_of_fractions_l1024_102444


namespace NUMINAMATH_CALUDE_square_area_is_17_l1024_102417

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by four vertices -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Calculate the area of a square given its four vertices -/
def squareArea (s : Square) : ℝ :=
  squaredDistance s.P s.Q

/-- The specific square from the problem -/
def problemSquare : Square :=
  { P := { x := 1, y := 2 },
    Q := { x := -3, y := 3 },
    R := { x := -2, y := 8 },
    S := { x := 2, y := 7 } }

theorem square_area_is_17 :
  squareArea problemSquare = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_17_l1024_102417


namespace NUMINAMATH_CALUDE_jack_apple_distribution_l1024_102426

theorem jack_apple_distribution (total_apples : ℕ) (given_to_father : ℕ) (num_friends : ℕ) :
  total_apples = 55 →
  given_to_father = 10 →
  num_friends = 4 →
  (total_apples - given_to_father) % (num_friends + 1) = 0 →
  (total_apples - given_to_father) / (num_friends + 1) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_apple_distribution_l1024_102426


namespace NUMINAMATH_CALUDE_average_score_is_two_l1024_102478

/-- Represents the distribution of scores in a class test --/
structure ScoreDistribution where
  score3 : Real
  score2 : Real
  score1 : Real
  score0 : Real
  sum_to_one : score3 + score2 + score1 + score0 = 1

/-- Calculates the average score given a score distribution --/
def averageScore (d : ScoreDistribution) : Real :=
  3 * d.score3 + 2 * d.score2 + 1 * d.score1 + 0 * d.score0

/-- Theorem: The average score for the given distribution is 2.0 --/
theorem average_score_is_two :
  let d : ScoreDistribution := {
    score3 := 0.3,
    score2 := 0.5,
    score1 := 0.1,
    score0 := 0.1,
    sum_to_one := by norm_num
  }
  averageScore d = 2.0 := by sorry

end NUMINAMATH_CALUDE_average_score_is_two_l1024_102478


namespace NUMINAMATH_CALUDE_inscribed_circle_arithmetic_progression_l1024_102437

theorem inscribed_circle_arithmetic_progression (a b c r : ℝ) :
  (0 < r) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (a + b > c) →
  (b + c > a) →
  (c + a > b) →
  (∃ d : ℝ, d > 0 ∧ a = 2*r + d ∧ b = 2*r + 2*d ∧ c = 2*r + 3*d) →
  (∃ k : ℝ, k > 0 ∧ a = 3*k ∧ b = 4*k ∧ c = 5*k) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_arithmetic_progression_l1024_102437


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1024_102466

theorem absolute_value_equation_solution :
  ∃ x : ℚ, |6 * x - 8| = 0 ∧ x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1024_102466


namespace NUMINAMATH_CALUDE_second_largest_power_of_ten_in_170_factorial_l1024_102406

/-- The number of factors of 5 in the prime factorization of n! -/
def count_factors_of_five (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem second_largest_power_of_ten_in_170_factorial : 
  40 = (count_factors_of_five 170) - 1 :=
sorry

end NUMINAMATH_CALUDE_second_largest_power_of_ten_in_170_factorial_l1024_102406


namespace NUMINAMATH_CALUDE_binary_remainder_is_two_l1024_102441

/-- Given a binary number represented as a list of bits (least significant bit first),
    calculate the remainder when divided by 4. -/
def binary_remainder_mod_4 (bits : List Bool) : Nat :=
  match bits with
  | [] => 0
  | [b₀] => if b₀ then 1 else 0
  | b₀ :: b₁ :: _ => (if b₁ then 2 else 0) + (if b₀ then 1 else 0)

/-- The binary representation of 100101110010₂ (least significant bit first) -/
def binary_number : List Bool :=
  [false, true, false, false, true, true, true, false, true, false, false, true]

/-- Theorem stating that the remainder when 100101110010₂ is divided by 4 is 2 -/
theorem binary_remainder_is_two :
  binary_remainder_mod_4 binary_number = 2 := by
  sorry


end NUMINAMATH_CALUDE_binary_remainder_is_two_l1024_102441


namespace NUMINAMATH_CALUDE_function_bound_l1024_102429

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, |x₁ - x₂| ≤ 1 → |f x₂ - f x₁| ≤ 1) ∧ f 0 = 1

/-- The main theorem -/
theorem function_bound (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x : ℝ, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l1024_102429


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1024_102420

/-- Given a triangle ABC with side lengths a and c, and angle A, proves that angle B has two possible values. -/
theorem triangle_angle_B (a c : ℝ) (A : ℝ) (h1 : a = 5 * Real.sqrt 2) (h2 : c = 10) (h3 : A = π / 6) :
  ∃ (B : ℝ), (B = π * 7 / 12 ∨ B = π / 12) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_B_l1024_102420


namespace NUMINAMATH_CALUDE_root_shift_cubic_l1024_102419

/-- Given a cubic polynomial with roots p, q, and r, 
    find the monic polynomial with roots p + 3, q + 3, and r + 3 -/
theorem root_shift_cubic (p q r : ℂ) : 
  (p^3 - 4*p^2 + 9*p - 7 = 0) ∧ 
  (q^3 - 4*q^2 + 9*q - 7 = 0) ∧ 
  (r^3 - 4*r^2 + 9*r - 7 = 0) → 
  ∃ (a b c : ℂ), 
    (∀ x : ℂ, x^3 - 13*x^2 + 60*x - 90 = (x - (p + 3)) * (x - (q + 3)) * (x - (r + 3))) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_cubic_l1024_102419


namespace NUMINAMATH_CALUDE_current_speed_l1024_102434

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 16)
  (h2 : speed_against_current = 9.6) :
  ∃ (current_speed : ℝ), current_speed = 3.2 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_current_speed_l1024_102434


namespace NUMINAMATH_CALUDE_pomelos_last_week_l1024_102450

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of boxes shipped last week -/
def boxes_last_week : ℕ := 10

/-- Represents the number of boxes shipped this week -/
def boxes_this_week : ℕ := 20

/-- Represents the total number of dozens of pomelos shipped -/
def total_dozens : ℕ := 60

/-- Theorem stating that the number of pomelos shipped last week is 240 -/
theorem pomelos_last_week :
  (total_dozens * dozen) / (boxes_last_week + boxes_this_week) * boxes_last_week = 240 := by
  sorry


end NUMINAMATH_CALUDE_pomelos_last_week_l1024_102450


namespace NUMINAMATH_CALUDE_cards_at_home_l1024_102479

def cards_in_hospital : ℕ := 403
def total_cards : ℕ := 690

theorem cards_at_home : total_cards - cards_in_hospital = 287 := by
  sorry

end NUMINAMATH_CALUDE_cards_at_home_l1024_102479


namespace NUMINAMATH_CALUDE_revenue_comparison_l1024_102462

theorem revenue_comparison (last_year_revenue : ℝ) : 
  let projected_revenue := last_year_revenue * 1.20
  let actual_revenue := last_year_revenue * 0.90
  actual_revenue / projected_revenue = 0.75 := by
sorry

end NUMINAMATH_CALUDE_revenue_comparison_l1024_102462


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1024_102445

-- Define the logarithm functions
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_sum_equals_two : lg 0.01 + log2 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1024_102445


namespace NUMINAMATH_CALUDE_cyclic_fraction_theorem_l1024_102456

theorem cyclic_fraction_theorem (x y z k : ℝ) :
  (x / (y + z) = k ∧ y / (z + x) = k ∧ z / (x + y) = k) →
  (k = 1/2 ∨ k = -1) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_fraction_theorem_l1024_102456


namespace NUMINAMATH_CALUDE_det_special_matrix_l1024_102492

/-- The determinant of the matrix [[1, x, x^2], [1, x+1, (x+1)^2], [1, x, (x+1)^2]] is equal to x + 1 -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![1, x, x^2; 1, x+1, (x+1)^2; 1, x, (x+1)^2] = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1024_102492


namespace NUMINAMATH_CALUDE_division_problem_l1024_102488

theorem division_problem : 
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1024_102488


namespace NUMINAMATH_CALUDE_simplify_fraction_l1024_102404

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1024_102404


namespace NUMINAMATH_CALUDE_bug_return_probability_l1024_102498

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - Q n)

/-- The probability of returning to the starting vertex on the eighth move -/
theorem bug_return_probability : Q 8 = 43/128 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l1024_102498


namespace NUMINAMATH_CALUDE_no_real_solutions_l1024_102424

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3*x + 8)^2 + 4 = -2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1024_102424


namespace NUMINAMATH_CALUDE_innings_played_l1024_102433

/-- Represents the number of innings played by a cricket player. -/
def innings : ℕ := sorry

/-- Represents the current average runs of the player. -/
def currentAverage : ℕ := 24

/-- Represents the runs needed in the next innings. -/
def nextInningsRuns : ℕ := 96

/-- Represents the increase in average after the next innings. -/
def averageIncrease : ℕ := 8

/-- Theorem stating that the number of innings played is 8. -/
theorem innings_played : innings = 8 := by sorry

end NUMINAMATH_CALUDE_innings_played_l1024_102433


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1024_102470

theorem min_value_of_expression (x : ℝ) :
  (15 - x) * (9 - x) * (15 + x) * (9 + x) ≥ -5184 :=
by sorry

theorem min_value_attained :
  ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_attained_l1024_102470


namespace NUMINAMATH_CALUDE_bus_capacity_proof_l1024_102463

theorem bus_capacity_proof (C : ℚ) 
  (h1 : (3 / 4) * C + (4 / 5) * C = 310) : C = 200 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_proof_l1024_102463


namespace NUMINAMATH_CALUDE_parallel_transitive_infinite_perpendicular_to_skew_l1024_102405

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internals of the line structure
  -- as we're only interested in the relationships between lines

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- Perpendicular relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Skew relation between two lines -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- The set of all lines perpendicular to two given lines -/
def perpendicularLines (l1 l2 : Line3D) : Set Line3D := sorry

theorem parallel_transitive (a b c : Line3D) :
  parallel a b → parallel b c → parallel a c := by sorry

theorem infinite_perpendicular_to_skew (a b : Line3D) :
  skew a b → Set.Infinite (perpendicularLines a b) := by sorry

end NUMINAMATH_CALUDE_parallel_transitive_infinite_perpendicular_to_skew_l1024_102405


namespace NUMINAMATH_CALUDE_total_score_is_40_l1024_102471

def game1_score : ℕ := 10
def game2_score : ℕ := 14
def game3_score : ℕ := 6

def first_three_games_total : ℕ := game1_score + game2_score + game3_score
def first_three_games_average : ℕ := first_three_games_total / 3
def game4_score : ℕ := first_three_games_average

def total_score : ℕ := first_three_games_total + game4_score

theorem total_score_is_40 : total_score = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_40_l1024_102471


namespace NUMINAMATH_CALUDE_inverse_of_A_l1024_102436

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l1024_102436


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1024_102409

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence : 
  let a₁ := (1 : ℚ) / 2
  let a₂ := (3 : ℚ) / 4
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 10 = (11 : ℚ) / 4 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1024_102409


namespace NUMINAMATH_CALUDE_f_bounds_l1024_102401

/-- The maximum number of elements from Example 1 -/
def f (n : ℕ) : ℕ := sorry

/-- Proof that f(n) satisfies the given inequality -/
theorem f_bounds (n : ℕ) (hn : n > 0) : 
  (1 / 6 : ℚ) * (n^2 - 4*n : ℚ) ≤ (f n : ℚ) ∧ (f n : ℚ) ≤ (1 / 6 : ℚ) * (n^2 - n : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_f_bounds_l1024_102401


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1024_102400

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  (∀ x, 2*x^2 + b*x + a < 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1024_102400


namespace NUMINAMATH_CALUDE_jean_buys_two_cards_per_grandchild_l1024_102416

/-- Represents the scenario of Jean's gift-giving to her grandchildren --/
structure GiftGiving where
  num_grandchildren : ℕ
  amount_per_card : ℕ
  total_amount : ℕ

/-- Calculates the number of cards bought for each grandchild --/
def cards_per_grandchild (g : GiftGiving) : ℕ :=
  (g.total_amount / g.amount_per_card) / g.num_grandchildren

/-- Theorem stating that Jean buys 2 cards for each grandchild --/
theorem jean_buys_two_cards_per_grandchild :
  ∀ (g : GiftGiving),
    g.num_grandchildren = 3 →
    g.amount_per_card = 80 →
    g.total_amount = 480 →
    cards_per_grandchild g = 2 := by
  sorry

end NUMINAMATH_CALUDE_jean_buys_two_cards_per_grandchild_l1024_102416


namespace NUMINAMATH_CALUDE_price_per_working_game_l1024_102474

def total_games : ℕ := 10
def non_working_games : ℕ := 8
def total_earnings : ℕ := 12

theorem price_per_working_game :
  (total_earnings : ℚ) / (total_games - non_working_games) = 6 := by
  sorry

end NUMINAMATH_CALUDE_price_per_working_game_l1024_102474


namespace NUMINAMATH_CALUDE_composition_equality_l1024_102458

-- Define the functions f and g
def f (b : ℝ) (x : ℝ) : ℝ := 5 * x + b
def g (b : ℝ) (x : ℝ) : ℝ := b * x + 4

-- State the theorem
theorem composition_equality (b e : ℝ) : 
  (∀ x, f b (g b x) = 15 * x + e) → e = 23 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l1024_102458


namespace NUMINAMATH_CALUDE_x_eleven_percent_greater_than_80_l1024_102476

/-- If x is 11 percent greater than 80, then x equals 88.8 -/
theorem x_eleven_percent_greater_than_80 (x : ℝ) :
  x = 80 * (1 + 11 / 100) → x = 88.8 := by
  sorry

end NUMINAMATH_CALUDE_x_eleven_percent_greater_than_80_l1024_102476


namespace NUMINAMATH_CALUDE_haley_origami_papers_l1024_102425

/-- The number of origami papers Haley has to give away -/
def total_papers : ℕ := 48

/-- The number of Haley's cousins -/
def num_cousins : ℕ := 6

/-- The number of papers each cousin would receive if Haley distributes all her papers equally -/
def papers_per_cousin : ℕ := 8

/-- Theorem stating that the total number of origami papers Haley has to give away is 48 -/
theorem haley_origami_papers :
  total_papers = num_cousins * papers_per_cousin :=
by sorry

end NUMINAMATH_CALUDE_haley_origami_papers_l1024_102425


namespace NUMINAMATH_CALUDE_class_size_l1024_102490

theorem class_size (chinese : ℕ) (math : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chinese = 15)
  (h2 : math = 18)
  (h3 : both = 8)
  (h4 : neither = 20) :
  chinese + math - both + neither = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1024_102490


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1024_102442

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k + 1) * x + (k^2 - 3) = 0) ↔ 
  ((1 - 2 * Real.sqrt 10) / 3 ≤ k ∧ k ≤ (1 + 2 * Real.sqrt 10) / 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1024_102442


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1024_102468

/-- The number of intersection points between a line and a hyperbola -/
theorem line_hyperbola_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃! p : ℝ × ℝ, 
    (p.2 = (b / a) * p.1 + 3) ∧ 
    ((p.1^2 / a^2) - (p.2^2 / b^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1024_102468


namespace NUMINAMATH_CALUDE_exponent_equivalence_l1024_102440

theorem exponent_equivalence (y : ℕ) (some_exponent : ℕ) 
  (h1 : 9^y = 3^some_exponent) (h2 : y = 8) : some_exponent = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equivalence_l1024_102440


namespace NUMINAMATH_CALUDE_tower_of_two_divisibility_l1024_102443

def f : ℕ → ℕ
| 0 => 2
| (n + 1) => 2^(f n)

theorem tower_of_two_divisibility (n : ℕ) (h : n ≥ 2) :
  n ∣ (f n - f (n - 1)) :=
sorry

end NUMINAMATH_CALUDE_tower_of_two_divisibility_l1024_102443


namespace NUMINAMATH_CALUDE_layla_point_difference_l1024_102427

theorem layla_point_difference (total_points layla_points : ℕ) 
  (h1 : total_points = 345) 
  (h2 : layla_points = 180) : 
  layla_points - (total_points - layla_points) = 15 := by
  sorry

end NUMINAMATH_CALUDE_layla_point_difference_l1024_102427


namespace NUMINAMATH_CALUDE_deck_size_l1024_102435

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  r + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_l1024_102435


namespace NUMINAMATH_CALUDE_greg_is_sixteen_l1024_102430

-- Define the ages of the siblings
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem to prove Greg's age
theorem greg_is_sixteen : greg_age = 16 := by
  sorry


end NUMINAMATH_CALUDE_greg_is_sixteen_l1024_102430


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1024_102412

theorem algebraic_expression_equality (a b : ℝ) (h : a^2 + 2*b^2 - 1 = 0) :
  (a - b)^2 + b*(2*a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1024_102412


namespace NUMINAMATH_CALUDE_b2_properties_b2_b4_equality_a_and_x_relation_l1024_102465

theorem b2_properties (B₂ : ℝ) (A : ℝ) (x : ℝ) : 
  B₂ = B₂^2 - 2 →
  (B₂ = -1 ∨ B₂ = 2) ∧
  (B₂ = -1 → (A = 1 ∨ A = -1) ∧ ¬(∃ x, x + 1/x = 1)) ∧
  (B₂ = 2 → (A = 2 ∨ A = -2) ∧ (x = 1 ∨ x = -1)) :=
by sorry

theorem b2_b4_equality (B₂ B₄ : ℝ) :
  B₂ = B₄ → B₂ = B₂^2 - 2 :=
by sorry

theorem a_and_x_relation (A x : ℝ) :
  A = x + 1/x →
  (A = 2 → x = 1) ∧
  (A = -2 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_b2_properties_b2_b4_equality_a_and_x_relation_l1024_102465


namespace NUMINAMATH_CALUDE_sin_translation_l1024_102467

open Real

theorem sin_translation (t S : ℝ) (k : ℤ) : 
  (1 = sin (2 * t)) → 
  (S > 0) → 
  (1 = sin (2 * (t + S) - π / 3)) → 
  (t = π / 4 + k * π ∧ S ≥ π / 6) :=
sorry

end NUMINAMATH_CALUDE_sin_translation_l1024_102467


namespace NUMINAMATH_CALUDE_article_font_pages_l1024_102481

theorem article_font_pages (total_words : ℕ) (large_font_words : ℕ) (small_font_words : ℕ) (total_pages : ℕ) :
  total_words = 48000 →
  large_font_words = 1800 →
  small_font_words = 2400 →
  total_pages = 21 →
  ∃ (large_pages : ℕ) (small_pages : ℕ),
    large_pages + small_pages = total_pages ∧
    large_pages * large_font_words + small_pages * small_font_words = total_words ∧
    large_pages = 4 :=
by sorry

end NUMINAMATH_CALUDE_article_font_pages_l1024_102481


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1024_102403

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- 
  Theorem: The radius of a cylinder inscribed in a cone
  Given:
  - A right circular cone with diameter 8 and altitude 10
  - A right circular cylinder inscribed in the cone
  - The axes of the cylinder and cone coincide
  - The height of the cylinder is three times its radius
  Prove: The radius of the cylinder is 20/11
-/
theorem inscribed_cylinder_radius (cone : Cone) (cyl : Cylinder) :
  cone.diameter = 8 →
  cone.altitude = 10 →
  cyl.height = 3 * cyl.radius →
  cyl.radius = 20 / 11 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1024_102403


namespace NUMINAMATH_CALUDE_apple_count_l1024_102447

/-- Given a box of fruit with apples and oranges, prove that the number of apples is 14 -/
theorem apple_count (total_oranges : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) : 
  total_oranges = 26 →
  removed_oranges = 20 →
  apple_percentage = 70 / 100 →
  (∃ (apples : ℕ), 
    (apples : ℚ) / ((apples : ℚ) + (total_oranges - removed_oranges : ℚ)) = apple_percentage ∧
    apples = 14) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_l1024_102447


namespace NUMINAMATH_CALUDE_train_crossing_time_l1024_102451

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 80

/-- Represents the length of the train in meters -/
def train_length : ℝ := 200

/-- Represents the time it takes for the train to cross the pole in seconds -/
def crossing_time : ℝ := 9

/-- Theorem stating that a train with the given speed and length takes 9 seconds to cross a pole -/
theorem train_crossing_time :
  (train_length / (train_speed * 1000 / 3600)) = crossing_time := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1024_102451


namespace NUMINAMATH_CALUDE_factor_expression_l1024_102482

theorem factor_expression (a b : ℝ) : 56 * b^2 * a^2 + 168 * b * a = 56 * b * a * (b * a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1024_102482


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1024_102499

def f (x : ℝ) : ℝ := -4 * x^3 + 3 * x + 2

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 0 1 → f y ≤ f x) ∧
  f x = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l1024_102499


namespace NUMINAMATH_CALUDE_chess_team_girls_l1024_102493

theorem chess_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 18 →
  total = boys + girls →
  attended = boys + girls / 3 →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_chess_team_girls_l1024_102493


namespace NUMINAMATH_CALUDE_distance_between_externally_tangent_circles_l1024_102431

/-- The distance between centers of two externally tangent circles is the sum of their radii -/
theorem distance_between_externally_tangent_circles 
  (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 8) 
  (h₃ : d = r₁ + r₂) : 
  d = 11 := by sorry

end NUMINAMATH_CALUDE_distance_between_externally_tangent_circles_l1024_102431


namespace NUMINAMATH_CALUDE_cubic_function_c_value_l1024_102408

/-- A function f: ℝ → ℝ has exactly two roots if there exist exactly two distinct real numbers x₁ and x₂ such that f(x₁) = f(x₂) = 0 -/
def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂

/-- The main theorem stating that if y = x³ - 3x + c has exactly two roots, then c = -2 or c = 2 -/
theorem cubic_function_c_value (c : ℝ) :
  has_exactly_two_roots (λ x : ℝ => x^3 - 3*x + c) → c = -2 ∨ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_c_value_l1024_102408


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l1024_102414

theorem greatest_prime_factor_of_154 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 154 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 154 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_154_l1024_102414


namespace NUMINAMATH_CALUDE_count_D_eq_2_is_30_l1024_102418

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 127 for which D(n) = 2 -/
def count_D_eq_2 : ℕ := sorry

theorem count_D_eq_2_is_30 : count_D_eq_2 = 30 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_2_is_30_l1024_102418


namespace NUMINAMATH_CALUDE_triangle_division_result_l1024_102472

-- Define the process of dividing triangles
def divide_triangles (n : ℕ) : ℕ := 3^n

-- Define the side length after n iterations
def side_length (n : ℕ) : ℚ := 1 / 2^n

-- Theorem statement
theorem triangle_division_result :
  let iterations : ℕ := 12
  let final_count : ℕ := divide_triangles iterations
  let final_side_length : ℚ := side_length iterations
  final_count = 531441 ∧ final_side_length = 1 / 2^12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_division_result_l1024_102472


namespace NUMINAMATH_CALUDE_prob_random_twin_prob_twins_in_three_expected_twin_pairs_l1024_102449

/-- Represents the probability model for twins in Schwambrania -/
structure TwinProbability where
  /-- The probability of twins being born -/
  p : ℝ
  /-- Assumption that p is between 0 and 1 -/
  h_p_bounds : 0 ≤ p ∧ p ≤ 1
  /-- Assumption that triplets do not exist -/
  h_no_triplets : True

/-- Theorem for the probability of a random person being a twin -/
theorem prob_random_twin (model : TwinProbability) :
  (2 * model.p) / (model.p + 1) = Real.exp (Real.log (2 * model.p) - Real.log (model.p + 1)) :=
sorry

/-- Theorem for the probability of having at least one pair of twins in a family with three children -/
theorem prob_twins_in_three (model : TwinProbability) :
  (2 * model.p) / (2 * model.p + (1 - model.p)^2) =
  Real.exp (Real.log (2 * model.p) - Real.log (2 * model.p + (1 - model.p)^2)) :=
sorry

/-- Theorem for the expected number of twin pairs among N first-graders -/
theorem expected_twin_pairs (model : TwinProbability) (N : ℕ) :
  (N : ℝ) * model.p / (model.p + 1) =
  Real.exp (Real.log N + Real.log model.p - Real.log (model.p + 1)) :=
sorry

end NUMINAMATH_CALUDE_prob_random_twin_prob_twins_in_three_expected_twin_pairs_l1024_102449


namespace NUMINAMATH_CALUDE_molecular_weight_CaO_is_56_l1024_102448

/-- The molecular weight of CaO in grams per mole -/
def molecular_weight_CaO : ℝ := 56

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles of CaO in grams -/
def given_weight : ℝ := 392

/-- Theorem stating that the molecular weight of CaO is 56 grams/mole -/
theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = given_weight / given_moles :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_CaO_is_56_l1024_102448


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l1024_102460

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def digit_at (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10 ^ place)) % 10

theorem four_digit_number_theorem (n : ℕ) :
  is_valid_number n ∧ 
  (digit_at n 0 + digit_at n 1 - 4 * digit_at n 3 = 1) ∧
  (digit_at n 0 + 10 * digit_at n 1 - 2 * digit_at n 2 = 14) →
  n = 1014 ∨ n = 2218 ∨ n = 1932 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l1024_102460


namespace NUMINAMATH_CALUDE_class_size_l1024_102459

/-- Given a class with a hair color ratio of 3:6:7 (red:blonde:black) and 9 red-haired kids,
    the total number of kids in the class is 48. -/
theorem class_size (red blonde black : ℕ) (total : ℕ) : 
  red = 3 → blonde = 6 → black = 7 → -- ratio condition
  red + blonde + black = total → -- total parts in ratio
  9 * total = 48 * red → -- condition for 9 red-haired kids
  total = 48 := by sorry

end NUMINAMATH_CALUDE_class_size_l1024_102459


namespace NUMINAMATH_CALUDE_coupe_price_proof_l1024_102455

/-- The amount for which Melissa sold the coupe -/
def coupe_price : ℝ := 30000

/-- The amount for which Melissa sold the SUV -/
def suv_price : ℝ := 2 * coupe_price

/-- The commission rate -/
def commission_rate : ℝ := 0.02

/-- The total commission from both sales -/
def total_commission : ℝ := 1800

theorem coupe_price_proof :
  commission_rate * (coupe_price + suv_price) = total_commission :=
sorry

end NUMINAMATH_CALUDE_coupe_price_proof_l1024_102455


namespace NUMINAMATH_CALUDE_davis_remaining_sticks_l1024_102487

/-- The number of popsicle sticks Miss Davis had initially -/
def initial_sticks : ℕ := 170

/-- The number of popsicle sticks given to each group -/
def sticks_per_group : ℕ := 15

/-- The number of groups in Miss Davis's class -/
def number_of_groups : ℕ := 10

/-- The number of popsicle sticks Miss Davis has left -/
def remaining_sticks : ℕ := initial_sticks - (sticks_per_group * number_of_groups)

theorem davis_remaining_sticks : remaining_sticks = 20 := by
  sorry

end NUMINAMATH_CALUDE_davis_remaining_sticks_l1024_102487


namespace NUMINAMATH_CALUDE_canned_food_bins_l1024_102410

theorem canned_food_bins (soup_bins vegetables_bins pasta_bins : Real) 
  (h1 : soup_bins = 0.12)
  (h2 : vegetables_bins = 0.12)
  (h3 : pasta_bins = 0.5) :
  soup_bins + vegetables_bins + pasta_bins = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_canned_food_bins_l1024_102410


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l1024_102486

theorem percentage_of_male_employees (total_employees : ℕ) 
  (males_below_50 : ℕ) (h1 : total_employees = 6400) 
  (h2 : males_below_50 = 3120) : 
  (males_below_50 : ℚ) / (0.75 * total_employees) = 65 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_employees_l1024_102486


namespace NUMINAMATH_CALUDE_julia_played_with_34_kids_l1024_102428

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 17

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := 2

/-- The total number of kids Julia played with -/
def total_kids : ℕ := monday_kids + tuesday_kids + wednesday_kids

theorem julia_played_with_34_kids : total_kids = 34 := by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_34_kids_l1024_102428


namespace NUMINAMATH_CALUDE_expression_factorization_l1024_102415

theorem expression_factorization (x y : ℝ) :
  (3 * x^3 + 28 * x^2 * y + 4 * x) - (-4 * x^3 + 5 * x^2 * y - 4 * x) = x * (x + 8) * (7 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1024_102415


namespace NUMINAMATH_CALUDE_inequality_proof_l1024_102469

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1024_102469


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1024_102480

theorem arithmetic_computation : -12 * 5 - (-4 * -2) + (-15 * -3) / 3 = -53 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1024_102480


namespace NUMINAMATH_CALUDE_tan_double_angle_problem_l1024_102489

theorem tan_double_angle_problem (θ : Real) 
  (h1 : Real.tan (2 * θ) = -2) 
  (h2 : π < 2 * θ) 
  (h3 : 2 * θ < 2 * π) : 
  Real.sin θ ^ 4 - Real.cos θ ^ 4 = -1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_problem_l1024_102489


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l1024_102483

theorem batsman_running_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (h1 : total_runs = 125)
  (h2 : boundaries = 5)
  (h3 : sixes = 5) :
  (total_runs - (boundaries * 4 + sixes * 6)) / total_runs * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l1024_102483
