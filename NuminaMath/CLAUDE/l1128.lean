import Mathlib

namespace NUMINAMATH_CALUDE_even_function_tangent_slope_l1128_112893

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x > 0 then a * x^2 / (x + 1) else a * x^2 / (-x + 1)

theorem even_function_tangent_slope (a : ℝ) :
  (∀ x, f a x = f a (-x)) →
  (∀ x > 0, f a x = a * x^2 / (x + 1)) →
  (deriv (f a)) (-1) = -1 →
  a = 4/3 := by sorry

end NUMINAMATH_CALUDE_even_function_tangent_slope_l1128_112893


namespace NUMINAMATH_CALUDE_hoseok_fruit_difference_l1128_112842

/-- The number of lemons eaten minus the number of pears eaten by Hoseok -/
def lemon_pear_difference (apples pears tangerines lemons watermelons : ℕ) : ℤ :=
  lemons - pears

theorem hoseok_fruit_difference :
  lemon_pear_difference 8 5 12 17 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_fruit_difference_l1128_112842


namespace NUMINAMATH_CALUDE_inequality_proof_l1128_112801

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1128_112801


namespace NUMINAMATH_CALUDE_simplify_fraction_l1128_112892

theorem simplify_fraction : 3 * (11 / 4) * (16 / -55) = -12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1128_112892


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1128_112848

theorem tangent_line_equation (x y : ℝ) : 
  y = 2 * x * Real.tan x →
  (2 + Real.pi / 2) * (Real.pi / 4) - (Real.pi / 2) - Real.pi^2 / 4 = 0 →
  (2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1128_112848


namespace NUMINAMATH_CALUDE_correct_prediction_probability_l1128_112879

def num_monday_classes : ℕ := 5
def num_tuesday_classes : ℕ := 6
def total_classes : ℕ := num_monday_classes + num_tuesday_classes
def correct_predictions : ℕ := 7
def monday_correct_predictions : ℕ := 3

theorem correct_prediction_probability :
  (Nat.choose num_monday_classes monday_correct_predictions * (1/2)^num_monday_classes) *
  (Nat.choose num_tuesday_classes (correct_predictions - monday_correct_predictions) * (1/2)^num_tuesday_classes) /
  (Nat.choose total_classes correct_predictions * (1/2)^total_classes) = 5/11 := by
sorry

end NUMINAMATH_CALUDE_correct_prediction_probability_l1128_112879


namespace NUMINAMATH_CALUDE_cleaning_frequency_in_year_l1128_112806

/-- The number of times a person cleans themselves in 52 weeks, given they take
    a bath twice a week and a shower once a week. -/
def cleaningFrequency (bathsPerWeek showerPerWeek weeksInYear : ℕ) : ℕ :=
  (bathsPerWeek + showerPerWeek) * weeksInYear

/-- Theorem stating that a person who takes a bath twice a week and a shower once a week
    cleans themselves 156 times in 52 weeks. -/
theorem cleaning_frequency_in_year :
  cleaningFrequency 2 1 52 = 156 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_frequency_in_year_l1128_112806


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1128_112838

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nth_term
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : ArithmeticSequence a d)
  (h1 : a 1 = 11)
  (h2 : d = 2)
  (h3 : ∃ n : ℕ, a n = 2009) :
  ∃ n : ℕ, n = 1000 ∧ a n = 2009 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1128_112838


namespace NUMINAMATH_CALUDE_minimum_soldiers_to_add_l1128_112858

theorem minimum_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  (84 - N % 84) = 82 :=
sorry

end NUMINAMATH_CALUDE_minimum_soldiers_to_add_l1128_112858


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l1128_112813

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_sum_theorem (a b c : ℕ) 
  (ha : isPrime a) (hb : isPrime b) (hc : isPrime c)
  (h1 : b + c = 13) (h2 : c^2 - a^2 = 72) : a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l1128_112813


namespace NUMINAMATH_CALUDE_unique_polynomial_l1128_112895

/-- A polynomial satisfying the given conditions -/
def p : ℝ → ℝ := λ x => x^2 + 1

/-- The theorem stating that p is the unique polynomial satisfying the conditions -/
theorem unique_polynomial :
  (p 3 = 10) ∧
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) ∧
  (∀ q : ℝ → ℝ, (q 3 = 10 ∧ ∀ x y : ℝ, q x * q y = q x + q y + q (x * y) - 3) → q = p) :=
by sorry

end NUMINAMATH_CALUDE_unique_polynomial_l1128_112895


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1128_112877

theorem absolute_value_inequality (x : ℝ) :
  x ≠ 1 →
  (|(2 * x - 1) / (x - 1)| > 2) ↔ (x > 3/4 ∧ x < 1) ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1128_112877


namespace NUMINAMATH_CALUDE_sqrt_35_between_5_and_6_l1128_112876

theorem sqrt_35_between_5_and_6 : 5 < Real.sqrt 35 ∧ Real.sqrt 35 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_35_between_5_and_6_l1128_112876


namespace NUMINAMATH_CALUDE_monster_perimeter_l1128_112884

theorem monster_perimeter (r : ℝ) (θ : ℝ) : 
  r = 2 → θ = 270 * π / 180 → 
  r * θ + 2 * r = 3 * π + 4 := by sorry

end NUMINAMATH_CALUDE_monster_perimeter_l1128_112884


namespace NUMINAMATH_CALUDE_sock_pair_probability_l1128_112885

def number_of_socks : ℕ := 10
def number_of_colors : ℕ := 5
def socks_per_color : ℕ := 2
def socks_drawn : ℕ := 6

theorem sock_pair_probability :
  let total_combinations := Nat.choose number_of_socks socks_drawn
  let pair_combinations := Nat.choose number_of_colors 2 * Nat.choose (number_of_colors - 2) 2 * 4
  (pair_combinations : ℚ) / total_combinations = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_probability_l1128_112885


namespace NUMINAMATH_CALUDE_expression_value_l1128_112859

theorem expression_value (a b c : ℤ) :
  a = 18 ∧ b = 20 ∧ c = 22 →
  (a - (b - c)) - ((a - b) - c) = 44 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1128_112859


namespace NUMINAMATH_CALUDE_root_implies_k_value_l1128_112852

theorem root_implies_k_value (k : ℝ) : 
  (6 * ((-25 - Real.sqrt 409) / 12)^2 + 25 * ((-25 - Real.sqrt 409) / 12) + k = 0) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l1128_112852


namespace NUMINAMATH_CALUDE_chocolate_bar_pieces_l1128_112833

theorem chocolate_bar_pieces :
  ∀ (total : ℕ),
  (total / 2 : ℕ) + (total / 4 : ℕ) + 15 = total →
  total = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_pieces_l1128_112833


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1128_112829

theorem quadratic_root_problem (k : ℝ) : 
  (2 : ℝ)^2 + 2 - k = 0 → (-3 : ℝ)^2 + (-3) - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1128_112829


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1128_112819

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 3 →
    1 / (x^3 - x^2 - 21*x + 45) = A / (x + 5) + B / (x - 3) + C / ((x - 3)^2)) →
  A = 1/64 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1128_112819


namespace NUMINAMATH_CALUDE_same_row_twice_l1128_112862

theorem same_row_twice (num_rows : Nat) (num_people : Nat) :
  num_rows = 7 →
  num_people = 50 →
  ∃ (p1 p2 : Nat) (r : Nat),
    p1 ≠ p2 ∧
    p1 < num_people ∧
    p2 < num_people ∧
    r < num_rows ∧
    (∃ (morning_seating afternoon_seating : Nat → Nat),
      morning_seating p1 = r ∧
      morning_seating p2 = r ∧
      afternoon_seating p1 = r ∧
      afternoon_seating p2 = r) :=
by sorry

end NUMINAMATH_CALUDE_same_row_twice_l1128_112862


namespace NUMINAMATH_CALUDE_gcd_g_10_g_13_l1128_112863

def g (x : ℤ) : ℤ := x^3 - 3*x^2 + x + 2050

theorem gcd_g_10_g_13 : Int.gcd (g 10) (g 13) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_g_10_g_13_l1128_112863


namespace NUMINAMATH_CALUDE_min_students_same_choice_l1128_112887

theorem min_students_same_choice (n : ℕ) (m : ℕ) (h1 : n = 45) (h2 : m = 6) :
  ∃ k : ℕ, k ≥ 16 ∧ k * m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_min_students_same_choice_l1128_112887


namespace NUMINAMATH_CALUDE_certain_number_solution_l1128_112800

theorem certain_number_solution : ∃ x : ℚ, (40 * 30 + (12 + 8) * x) / 5 = 1212 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l1128_112800


namespace NUMINAMATH_CALUDE_lasagna_pieces_sum_to_six_l1128_112870

/-- Represents the number of lasagna pieces each person eats -/
structure LasagnaPieces where
  manny : ℚ
  aaron : ℚ
  kai : ℚ
  raphael : ℚ
  lisa : ℚ

/-- Calculates the total number of lasagna pieces eaten -/
def total_pieces (pieces : LasagnaPieces) : ℚ :=
  pieces.manny + pieces.aaron + pieces.kai + pieces.raphael + pieces.lisa

/-- Theorem stating the total number of lasagna pieces equals 6 -/
theorem lasagna_pieces_sum_to_six : ∃ (pieces : LasagnaPieces), 
  pieces.manny = 1 ∧ 
  pieces.aaron = 0 ∧ 
  pieces.kai = 2 * pieces.manny ∧ 
  pieces.raphael = pieces.manny / 2 ∧ 
  pieces.lisa = 2 + pieces.raphael ∧ 
  total_pieces pieces = 6 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_pieces_sum_to_six_l1128_112870


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l1128_112894

/-- Given a line segment AB with points C, D, and E such that AB = 4AD = 4BE
    and AD = DC = CE = EB, the probability of a random point on AB being
    between C and D is 1/4. -/
theorem probability_between_C_and_D (A B C D E : ℝ) : 
  A < C ∧ C < D ∧ D < E ∧ E < B →
  B - A = 4 * (D - A) →
  B - A = 4 * (B - E) →
  D - A = C - D →
  C - D = E - C →
  E - C = B - E →
  (D - C) / (B - A) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l1128_112894


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1128_112836

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 3*I) / (1 - I) → z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1128_112836


namespace NUMINAMATH_CALUDE_gcd_difference_is_perfect_square_l1128_112873

theorem gcd_difference_is_perfect_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), Nat.gcd x (Nat.gcd y z) * (y - x) = k * k := by
  sorry

end NUMINAMATH_CALUDE_gcd_difference_is_perfect_square_l1128_112873


namespace NUMINAMATH_CALUDE_nested_custom_op_equals_two_l1128_112883

/-- Custom operation defined as (a + b) / c -/
def customOp (a b c : ℚ) : ℚ := (a + b) / c

/-- Nested application of customOp -/
def nestedCustomOp (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℚ) : ℚ :=
  customOp (customOp a₁ b₁ c₁) (customOp a₂ b₂ c₂) (customOp a₃ b₃ c₃)

/-- Theorem stating that the nested custom operation equals 2 -/
theorem nested_custom_op_equals_two :
  nestedCustomOp 120 60 180 4 2 6 20 10 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_custom_op_equals_two_l1128_112883


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1128_112865

theorem simplify_square_roots : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 45 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1128_112865


namespace NUMINAMATH_CALUDE_two_digit_primes_with_prime_digits_l1128_112889

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def digits_are_prime (n : ℕ) : Prop :=
  is_prime (n / 10) ∧ is_prime (n % 10)

theorem two_digit_primes_with_prime_digits :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit n ∧ is_prime n ∧ digits_are_prime n) ∧
    (∀ n, is_two_digit n → is_prime n → digits_are_prime n → n ∈ s) ∧
    s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_prime_digits_l1128_112889


namespace NUMINAMATH_CALUDE_repeating_decimal_proof_l1128_112872

theorem repeating_decimal_proof : ∃ (n : ℕ), n ≥ 10 ∧ n < 100 ∧ 
  (48 * (n / 99 : ℚ) - 48 * (n / 100 : ℚ) = 1 / 5) ∧
  (100 * (n / 99 : ℚ) - (n / 99 : ℚ) = n) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_proof_l1128_112872


namespace NUMINAMATH_CALUDE_a_2016_value_l1128_112821

def sequence_sum (n : ℕ) : ℕ := n ^ 2

theorem a_2016_value :
  let a : ℕ → ℕ := fun n => sequence_sum n - sequence_sum (n - 1)
  a 2016 = 4031 := by
  sorry

end NUMINAMATH_CALUDE_a_2016_value_l1128_112821


namespace NUMINAMATH_CALUDE_sara_sent_nine_letters_in_february_l1128_112835

/-- The number of letters Sara sent in February -/
def letters_in_february : ℕ := 33 - (6 + 3 * 6)

/-- Proof that Sara sent 9 letters in February -/
theorem sara_sent_nine_letters_in_february :
  letters_in_february = 9 := by
  sorry

#eval letters_in_february

end NUMINAMATH_CALUDE_sara_sent_nine_letters_in_february_l1128_112835


namespace NUMINAMATH_CALUDE_parabola_triangle_property_l1128_112832

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line y = x + 3
def line (x y : ℝ) : Prop := y = x + 3

-- Define a point on the parabola
structure PointOnParabola (p : ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : parabola p x y

-- Define the theorem
theorem parabola_triangle_property (p : ℝ) :
  parabola p 1 2 →  -- The parabola passes through (1, 2)
  ∀ (A : PointOnParabola p),
    A.x ≠ 1 ∨ A.y ≠ 2 →  -- A is different from (1, 2)
    ∃ (P B : ℝ × ℝ),
      -- P is on the line AC and y = x + 3
      (∃ t : ℝ, P.1 = (1 - t) * 1 + t * A.x ∧ P.2 = (1 - t) * 2 + t * A.y) ∧
      line P.1 P.2 ∧
      -- B is on the parabola and has the same y-coordinate as P
      parabola p B.1 B.2 ∧ B.2 = P.2 →
      -- 1. AB passes through (3, 2)
      (∃ s : ℝ, 3 = (1 - s) * A.x + s * B.1 ∧ 2 = (1 - s) * A.y + s * B.2) ∧
      -- 2. The minimum area of triangle ABC is 4√2
      (∀ (area : ℝ), area ≥ 0 ∧ area * area = 32 → 
        ∃ (A' : PointOnParabola p) (P' B' : ℝ × ℝ),
          A'.x ≠ 1 ∨ A'.y ≠ 2 ∧
          (∃ t : ℝ, P'.1 = (1 - t) * 1 + t * A'.x ∧ P'.2 = (1 - t) * 2 + t * A'.y) ∧
          line P'.1 P'.2 ∧
          parabola p B'.1 B'.2 ∧ B'.2 = P'.2 ∧
          area = (1/2) * Real.sqrt ((A'.x - 1)^2 + (A'.y - 2)^2) * Real.sqrt ((B'.1 - 1)^2 + (B'.2 - 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_property_l1128_112832


namespace NUMINAMATH_CALUDE_sum_of_segments_is_81_l1128_112830

/-- Represents the structure of triangles within the larger triangle -/
structure TriangleStructure where
  large_perimeter : ℝ
  small_side_length : ℝ
  small_triangle_count : ℕ

/-- The specific triangle structure from the problem -/
def problem_structure : TriangleStructure where
  large_perimeter := 24
  small_side_length := 1
  small_triangle_count := 27

/-- Calculates the sum of all segment lengths in the structure -/
def sum_of_segments (ts : TriangleStructure) : ℝ :=
  ts.small_triangle_count * (3 * ts.small_side_length)

/-- Theorem stating the sum of all segments in the given structure is 81 -/
theorem sum_of_segments_is_81 :
  sum_of_segments problem_structure = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_segments_is_81_l1128_112830


namespace NUMINAMATH_CALUDE_timothy_cows_l1128_112845

def total_cost : ℕ := 147700
def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def cow_cost : ℕ := 1000
def chicken_cost : ℕ := 100 * 5
def solar_installation_cost : ℕ := 6 * 100
def solar_equipment_cost : ℕ := 6000

def other_costs : ℕ := land_cost + house_cost + chicken_cost + solar_installation_cost + solar_equipment_cost

theorem timothy_cows :
  (total_cost - other_costs) / cow_cost = 20 := by sorry

end NUMINAMATH_CALUDE_timothy_cows_l1128_112845


namespace NUMINAMATH_CALUDE_ellipse_line_theorem_l1128_112891

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: For an ellipse with given properties, if a line intersects a chord at its midpoint A(1, 1/2), then the line has equation x + 2y - 2 = 0 -/
theorem ellipse_line_theorem (e : Ellipse) (l : Line) :
  e.center = (0, 0) ∧
  e.left_focus = (-Real.sqrt 3, 0) ∧
  e.right_vertex = (2, 0) ∧
  (∃ (B C : ℝ × ℝ), B ≠ C ∧ (1, 1/2) = ((B.1 + C.1)/2, (B.2 + C.2)/2) ∧
    (∀ (x y : ℝ), x^2/4 + y^2 = 1 → l.a * x + l.b * y + l.c = 0 → (x, y) = B ∨ (x, y) = C)) →
  l.a = 1 ∧ l.b = 2 ∧ l.c = -2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_theorem_l1128_112891


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l1128_112851

theorem rectangular_prism_width 
  (l h w d : ℝ) 
  (h_def : h = 2 * l)
  (l_val : l = 5)
  (diagonal : d = 17)
  (diag_eq : d^2 = l^2 + w^2 + h^2) :
  w = 2 * Real.sqrt 41 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l1128_112851


namespace NUMINAMATH_CALUDE_open_set_classification_l1128_112815

-- Define the concept of an open set in R²
def is_open_set (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ A → ∃ (r : ℝ), r > 0 ∧ 
    {q : ℝ × ℝ | Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2) < r} ⊆ A

-- Define the sets
def set1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def set2 : Set (ℝ × ℝ) := {p | p.1 + p.2 + 2 > 0}
def set3 : Set (ℝ × ℝ) := {p | |p.1 + p.2| ≤ 6}
def set4 : Set (ℝ × ℝ) := {p | 0 < p.1^2 + (p.2 - Real.sqrt 2)^2 ∧ p.1^2 + (p.2 - Real.sqrt 2)^2 < 1}

-- State the theorem
theorem open_set_classification :
  ¬(is_open_set set1) ∧
  (is_open_set set2) ∧
  ¬(is_open_set set3) ∧
  (is_open_set set4) :=
sorry

end NUMINAMATH_CALUDE_open_set_classification_l1128_112815


namespace NUMINAMATH_CALUDE_remainder_theorem_l1128_112854

theorem remainder_theorem (n : ℕ) (h1 : n > 0) (h2 : (n + 1) % 6 = 4) : n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1128_112854


namespace NUMINAMATH_CALUDE_distinct_values_of_d_l1128_112825

theorem distinct_values_of_d (d : ℂ) (u v w x : ℂ) 
  (h_distinct : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x)
  (h_eq : ∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
                   (z - d*u) * (z - d*v) * (z - d*w) * (z - d*x)) :
  ∃! (S : Finset ℂ), S.card = 4 ∧ ∀ d' : ℂ, d' ∈ S ↔ 
    (∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
              (z - d'*u) * (z - d'*v) * (z - d'*w) * (z - d'*x)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_values_of_d_l1128_112825


namespace NUMINAMATH_CALUDE_cubes_in_box_percentage_l1128_112878

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.height

/-- Calculates the number of cubes that can fit along a dimension -/
def cubesFit (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (b : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  (cubesFit b.length cubeSize) * (cubesFit b.width cubeSize) * (cubesFit b.height cubeSize)

/-- Calculates the volume occupied by cubes -/
def cubesVolume (numCubes : ℕ) (cubeSize : ℕ) : ℕ :=
  numCubes * (cubeSize * cubeSize * cubeSize)

/-- Calculates the percentage of box volume occupied by cubes -/
def percentageOccupied (boxVol : ℕ) (cubesVol : ℕ) : ℚ :=
  (cubesVol : ℚ) / (boxVol : ℚ) * 100

/-- Theorem: The percentage of volume occupied by 3-inch cubes in an 8x6x12 inch box is 75% -/
theorem cubes_in_box_percentage :
  let box := BoxDimensions.mk 8 6 12
  let cubeSize := 3
  let boxVol := boxVolume box
  let numCubes := totalCubes box cubeSize
  let cubesVol := cubesVolume numCubes cubeSize
  percentageOccupied boxVol cubesVol = 75 := by
  sorry

end NUMINAMATH_CALUDE_cubes_in_box_percentage_l1128_112878


namespace NUMINAMATH_CALUDE_unit_square_tiling_l1128_112855

/-- A rectangle is considered "good" if it can be tiled by rectangles similar to 1 × (3 + ∛3) -/
def is_good (a b : ℝ) : Prop := sorry

/-- The scaling property of good rectangles -/
axiom good_scale (a b c : ℝ) (h : c > 0) :
  is_good a b → is_good (a * c) (b * c)

/-- The integer multiple property of good rectangles -/
axiom good_int_multiple (m n : ℝ) (j : ℕ) (h : j > 0) :
  is_good m n → is_good m (n * j)

/-- The main theorem: the unit square can be tiled with rectangles similar to 1 × (3 + ∛3) -/
theorem unit_square_tiling :
  ∃ (tiling : Set (ℝ × ℝ)), 
    (∀ (rect : ℝ × ℝ), rect ∈ tiling → is_good rect.1 rect.2) ∧
    (∃ (f : ℝ × ℝ → ℝ × ℝ), 
      (∀ x y, f (x, y) = (x, y)) ∧
      (∀ (rect : ℝ × ℝ), rect ∈ tiling → 
        ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ f (rect.1, rect.2) = (a, b) ∧ 
        b / a = 3 + Real.rpow 3 (1/3 : ℝ))) :=
sorry

end NUMINAMATH_CALUDE_unit_square_tiling_l1128_112855


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1128_112857

theorem polynomial_factorization : 
  ∀ x : ℤ, x^15 + x^10 + 1 = (x^3 + x^2 + 1) * (x^12 - x^11 + x^9 - x^8 + x^6 - x^4 + x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1128_112857


namespace NUMINAMATH_CALUDE_calories_consumed_l1128_112898

/-- Given a package of candy with 3 servings of 120 calories each,
    prove that eating half the package results in consuming 180 calories. -/
theorem calories_consumed (servings : ℕ) (calories_per_serving : ℕ) (portion_eaten : ℚ) : 
  servings = 3 → 
  calories_per_serving = 120 → 
  portion_eaten = 1/2 →
  (↑servings * ↑calories_per_serving : ℚ) * portion_eaten = 180 := by
sorry

end NUMINAMATH_CALUDE_calories_consumed_l1128_112898


namespace NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l1128_112831

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for the first part
theorem complement_union_A_B : 
  (Aᶜ ∪ Bᶜ) = {x | x ≤ 2 ∨ x ≥ 10} := by sorry

-- Theorem for the second part
theorem complement_A_inter_B : 
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_complement_A_inter_B_l1128_112831


namespace NUMINAMATH_CALUDE_tangent_slope_three_cubic_l1128_112808

theorem tangent_slope_three_cubic (x : ℝ) : 
  (∃ y : ℝ, y = x^3 ∧ (3 * x^2 = 3)) ↔ (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_three_cubic_l1128_112808


namespace NUMINAMATH_CALUDE_max_k_for_arithmetic_sequences_l1128_112810

/-- An arithmetic sequence -/
def ArithmeticSequence (a d : ℝ) : ℕ → ℝ := fun n ↦ a + (n - 1) * d

theorem max_k_for_arithmetic_sequences (a₁ a₂ d₁ d₂ : ℝ) (k : ℕ) :
  k > 1 →
  (ArithmeticSequence a₁ d₁ (k - 1)) * (ArithmeticSequence a₂ d₂ (k - 1)) = 42 →
  (ArithmeticSequence a₁ d₁ k) * (ArithmeticSequence a₂ d₂ k) = 30 →
  (ArithmeticSequence a₁ d₁ (k + 1)) * (ArithmeticSequence a₂ d₂ (k + 1)) = 16 →
  a₁ = a₂ →
  k ≤ 14 ∧ ∃ (a d₁ d₂ : ℝ), k = 14 ∧
    (ArithmeticSequence a d₁ 13) * (ArithmeticSequence a d₂ 13) = 42 ∧
    (ArithmeticSequence a d₁ 14) * (ArithmeticSequence a d₂ 14) = 30 ∧
    (ArithmeticSequence a d₁ 15) * (ArithmeticSequence a d₂ 15) = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_k_for_arithmetic_sequences_l1128_112810


namespace NUMINAMATH_CALUDE_sum_of_roots_l1128_112882

theorem sum_of_roots (k m : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : 2 * y₁^2 - k * y₁ - m = 0)
  (h₂ : 2 * y₂^2 - k * y₂ - m = 0)
  (h₃ : y₁ ≠ y₂) : 
  y₁ + y₂ = k / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1128_112882


namespace NUMINAMATH_CALUDE_share_distribution_l1128_112867

theorem share_distribution (total : ℝ) (share_a : ℝ) (share_b : ℝ) (share_c : ℝ) :
  total = 246 →
  share_b = 0.65 →
  share_c = 48 →
  share_a + share_b + share_c = 1 →
  share_c / total = 48 / 246 :=
by sorry

end NUMINAMATH_CALUDE_share_distribution_l1128_112867


namespace NUMINAMATH_CALUDE_complement_A_union_B_A_inter_complement_B_l1128_112803

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 1 < x ∧ x < 9}

-- Theorem for the first part
theorem complement_A_union_B : 
  (Set.univ \ A) ∪ B = {x | x < 0 ∨ x > 1} := by sorry

-- Theorem for the second part
theorem A_inter_complement_B : 
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_A_inter_complement_B_l1128_112803


namespace NUMINAMATH_CALUDE_gems_per_dollar_l1128_112805

/-- Proves that the number of gems per dollar is 100 given the problem conditions -/
theorem gems_per_dollar (total_spent : ℝ) (bonus_rate : ℝ) (final_gems : ℝ) :
  total_spent = 250 →
  bonus_rate = 0.2 →
  final_gems = 30000 →
  (final_gems / (total_spent * (1 + bonus_rate))) = 100 := by
sorry

end NUMINAMATH_CALUDE_gems_per_dollar_l1128_112805


namespace NUMINAMATH_CALUDE_rational_sum_and_sum_of_squares_coprime_to_six_l1128_112802

theorem rational_sum_and_sum_of_squares_coprime_to_six (a b : ℚ) :
  let S := a + b
  (S = a + b) → (S = a^2 + b^2) → ∃ (m k : ℤ), S = m / k ∧ k ≠ 0 ∧ Nat.Coprime k.natAbs 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_and_sum_of_squares_coprime_to_six_l1128_112802


namespace NUMINAMATH_CALUDE_tricycle_count_l1128_112868

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_wheels = 26) : ∃ (bicycles tricycles : ℕ),
  bicycles + tricycles = total_children ∧ 
  2 * bicycles + 3 * tricycles = total_wheels ∧
  tricycles = 6 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l1128_112868


namespace NUMINAMATH_CALUDE_third_smallest_prime_cubed_to_fourth_l1128_112874

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem third_smallest_prime_cubed_to_fourth : (nthPrime 3) ^ 3 ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_third_smallest_prime_cubed_to_fourth_l1128_112874


namespace NUMINAMATH_CALUDE_pear_apple_equivalence_l1128_112826

/-- The cost of fruits at Joe's Fruit Stand -/
structure FruitCost where
  pear : ℕ
  grape : ℕ
  apple : ℕ

/-- The relation between pears and grapes -/
def pear_grape_relation (c : FruitCost) : Prop :=
  4 * c.pear = 3 * c.grape

/-- The relation between grapes and apples -/
def grape_apple_relation (c : FruitCost) : Prop :=
  9 * c.grape = 6 * c.apple

/-- Theorem stating the cost equivalence of 24 pears and 12 apples -/
theorem pear_apple_equivalence (c : FruitCost) 
  (h1 : pear_grape_relation c) 
  (h2 : grape_apple_relation c) : 
  24 * c.pear = 12 * c.apple :=
by
  sorry

end NUMINAMATH_CALUDE_pear_apple_equivalence_l1128_112826


namespace NUMINAMATH_CALUDE_sequence_properties_l1128_112880

def a : ℕ → ℕ
  | 0 => 2
  | n + 1 => (a n)^2 - a n + 1

theorem sequence_properties :
  (∀ m n : ℕ, m ≠ n → Nat.gcd (a m) (a n) = 1) ∧
  (∑' k : ℕ, (1 : ℝ) / (a k)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1128_112880


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l1128_112823

theorem parallelogram_altitude_base_ratio 
  (area : ℝ) (base : ℝ) (altitude : ℝ) 
  (h_area : area = 288) 
  (h_base : base = 12) 
  (h_area_formula : area = base * altitude) : 
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l1128_112823


namespace NUMINAMATH_CALUDE_tv_price_difference_l1128_112899

def budget : ℕ := 1000
def initial_discount : ℕ := 100
def additional_discount_percent : ℕ := 20

theorem tv_price_difference : 
  let price_after_initial_discount := budget - initial_discount
  let additional_discount := price_after_initial_discount * additional_discount_percent / 100
  let final_price := price_after_initial_discount - additional_discount
  budget - final_price = 280 := by sorry

end NUMINAMATH_CALUDE_tv_price_difference_l1128_112899


namespace NUMINAMATH_CALUDE_chocolate_difference_l1128_112853

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- Theorem stating the difference in chocolate consumption between Robert and Nickel -/
theorem chocolate_difference : robert_chocolates - nickel_chocolates = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l1128_112853


namespace NUMINAMATH_CALUDE_maple_trees_planted_l1128_112837

theorem maple_trees_planted (initial_maple : ℕ) (final_maple : ℕ) :
  initial_maple = 2 →
  final_maple = 11 →
  final_maple - initial_maple = 9 :=
by sorry

end NUMINAMATH_CALUDE_maple_trees_planted_l1128_112837


namespace NUMINAMATH_CALUDE_equation_solution_l1128_112816

theorem equation_solution (x y z k : ℝ) :
  (9 / (x - y) = k / (x + z)) ∧ (k / (x + z) = 16 / (z + y)) → k = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1128_112816


namespace NUMINAMATH_CALUDE_person_speed_in_mph_l1128_112890

/-- Prove that a person crossing a 2500-meter street in 8 minutes has a speed of approximately 11.65 miles per hour. -/
theorem person_speed_in_mph : ∃ (speed : ℝ), abs (speed - 11.65) < 0.01 :=
  let street_length : ℝ := 2500 -- meters
  let crossing_time : ℝ := 8 -- minutes
  let meters_per_mile : ℝ := 1609.34
  let minutes_per_hour : ℝ := 60
  let distance_miles : ℝ := street_length / meters_per_mile
  let time_hours : ℝ := crossing_time / minutes_per_hour
  let speed : ℝ := distance_miles / time_hours
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_person_speed_in_mph_l1128_112890


namespace NUMINAMATH_CALUDE_systematic_sampling_l1128_112839

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium -/
structure Auditorium where
  totalSeats : Nat
  seatsPerRow : Nat

/-- Represents a selection of students -/
structure Selection where
  seatNumber : Nat
  count : Nat

/-- Determines the sampling method based on the auditorium and selection -/
def determineSamplingMethod (a : Auditorium) (s : Selection) : SamplingMethod :=
  sorry

/-- Theorem: Selecting students with seat number 15 from the given auditorium is a systematic sampling method -/
theorem systematic_sampling (a : Auditorium) (s : Selection) :
  a.totalSeats = 25 →
  a.seatsPerRow = 20 →
  s.seatNumber = 15 →
  s.count = 25 →
  determineSamplingMethod a s = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1128_112839


namespace NUMINAMATH_CALUDE_shopping_ratio_l1128_112843

theorem shopping_ratio : 
  let emma_spent : ℕ := 58
  let elsa_spent : ℕ := 2 * emma_spent
  let total_spent : ℕ := 638
  let elizabeth_spent : ℕ := total_spent - (emma_spent + elsa_spent)
  (elizabeth_spent : ℚ) / (elsa_spent : ℚ) = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_shopping_ratio_l1128_112843


namespace NUMINAMATH_CALUDE_problem_solution_l1128_112822

/-- Represents the contents of a box of colored balls -/
structure Box where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Calculates the probability of Person B winning given the contents of both boxes -/
def probability_b_wins (box_a box_b : Box) : Rat :=
  let total_a := box_a.red + box_a.yellow + box_a.blue
  let total_b := box_b.red + box_b.yellow + box_b.blue
  ((box_a.red * box_b.red) + (box_a.yellow * box_b.yellow) + (box_a.blue * box_b.blue)) / (total_a * total_b)

/-- Calculates the average score for Person B given the contents of both boxes -/
def average_score_b (box_a box_b : Box) : Rat :=
  let total_a := box_a.red + box_a.yellow + box_a.blue
  let total_b := box_b.red + box_b.yellow + box_b.blue
  ((box_a.red * box_b.red * 1) + (box_a.yellow * box_b.yellow * 2) + (box_a.blue * box_b.blue * 3)) / (total_a * total_b)

theorem problem_solution :
  let box_a : Box := ⟨3, 2, 1⟩
  let box_b1 : Box := ⟨1, 2, 3⟩
  let box_b2 : Box := ⟨1, 4, 1⟩
  (probability_b_wins box_a box_b1 = 5/18) ∧
  (average_score_b box_a box_b2 = 11/18) ∧
  (∀ (x y z : Nat), x + y + z = 6 → average_score_b box_a ⟨x, y, z⟩ ≤ 11/18) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1128_112822


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1128_112897

theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h2 : a 2 = 9) (h3 : a 5 = 243) : a 4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1128_112897


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1128_112875

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  (1/a + 9/b + 25/c + 49/d + 81/e + 121/f) ≥ 129.6 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1128_112875


namespace NUMINAMATH_CALUDE_probability_both_from_c_l1128_112861

structure Workshop where
  name : String
  quantity : Nat

def total_quantity (workshops : List Workshop) : Nat :=
  workshops.foldl (fun acc w => acc + w.quantity) 0

def sample_size : Nat := 6

def stratified_sample (w : Workshop) (total : Nat) : Nat :=
  w.quantity * sample_size / total

theorem probability_both_from_c (workshops : List Workshop) :
  let total := total_quantity workshops
  let c_workshop := workshops.find? (fun w => w.name = "C")
  match c_workshop with
  | some c =>
    let c_samples := stratified_sample c total
    (c_samples.choose 2) / (sample_size.choose 2) = 1 / 5
  | none => False
  := by sorry

end NUMINAMATH_CALUDE_probability_both_from_c_l1128_112861


namespace NUMINAMATH_CALUDE_percentage_calculation_l1128_112827

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * 200 - 30 = 50 → P = 40 :=
by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1128_112827


namespace NUMINAMATH_CALUDE_average_score_theorem_l1128_112881

def perfect_score : ℕ := 30
def deduction_per_mistake : ℕ := 2

def madeline_mistakes : ℕ := 2
def leo_mistakes : ℕ := 2 * madeline_mistakes
def brent_mistakes : ℕ := leo_mistakes + 1
def nicholas_mistakes : ℕ := 3 * madeline_mistakes

def brent_score : ℕ := 25
def nicholas_score : ℕ := brent_score - 5

def student_score (mistakes : ℕ) : ℕ := perfect_score - mistakes * deduction_per_mistake

theorem average_score_theorem : 
  (student_score madeline_mistakes + student_score leo_mistakes + brent_score + nicholas_score) / 4 = 83 / 4 := by
  sorry

end NUMINAMATH_CALUDE_average_score_theorem_l1128_112881


namespace NUMINAMATH_CALUDE_solution_when_m_is_one_solution_for_general_m_l1128_112896

-- Define the inequality
def inequality (m x : ℝ) : Prop := (2*m - m*x)/2 > x/2 - 1

-- Theorem for part 1
theorem solution_when_m_is_one :
  ∀ x : ℝ, inequality 1 x ↔ x < 2 := by sorry

-- Theorem for part 2
theorem solution_for_general_m :
  ∀ m x : ℝ, m ≠ -1 →
    (inequality m x ↔ (m > -1 ∧ x < 2) ∨ (m < -1 ∧ x > 2)) := by sorry

end NUMINAMATH_CALUDE_solution_when_m_is_one_solution_for_general_m_l1128_112896


namespace NUMINAMATH_CALUDE_paving_stone_width_l1128_112846

/-- Theorem: Width of paving stones in a rectangular courtyard -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (stone_count : ℕ)
  (h1 : courtyard_length = 20)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : stone_count = 66)
  : ∃ (stone_width : ℝ),
    courtyard_length * courtyard_width = stone_count * (stone_length * stone_width) ∧
    stone_width = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1128_112846


namespace NUMINAMATH_CALUDE_simplify_expression_l1128_112886

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) :
  (1 - a / (a + 1)) / ((a^2 - a) / (a^2 - 1)) = 1 / a :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1128_112886


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l1128_112817

/-- Given a cylinder with volume 150π cm³, prove that:
    1. A cone with the same base radius and height as the cylinder has a volume of 50π cm³
    2. A sphere with the same radius as the cylinder has a volume of 200π cm³ -/
theorem cylinder_cone_sphere_volumes (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  π * r^2 * h = 150 * π →
  (1/3 : ℝ) * π * r^2 * h = 50 * π ∧ 
  (4/3 : ℝ) * π * r^3 = 200 * π := by
  sorry

#check cylinder_cone_sphere_volumes

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l1128_112817


namespace NUMINAMATH_CALUDE_min_value_xy_l1128_112834

theorem min_value_xy (x y : ℕ+) (h1 : x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ) > 0) 
  (h2 : ∃ (z : ℕ), z^2 ≠ x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ)) : 
  x^2 + y^2 - 2017*(x:ℕ)*(y:ℕ) ≥ 2019 := by
sorry

end NUMINAMATH_CALUDE_min_value_xy_l1128_112834


namespace NUMINAMATH_CALUDE_function_equivalence_l1128_112814

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x, f x = 1/2 * x^2 - x + 3/2 := by
sorry

end NUMINAMATH_CALUDE_function_equivalence_l1128_112814


namespace NUMINAMATH_CALUDE_good_set_closed_under_addition_l1128_112856

-- Define a "good set"
def is_good_set (A : Set ℚ) : Prop :=
  (0 ∈ A) ∧ (1 ∈ A) ∧
  (∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A) ∧
  (∀ x, x ∈ A → x ≠ 0 → (1 / x) ∈ A)

-- Theorem statement
theorem good_set_closed_under_addition (A : Set ℚ) (h : is_good_set A) :
  ∀ x y, x ∈ A → y ∈ A → (x + y) ∈ A :=
by sorry

end NUMINAMATH_CALUDE_good_set_closed_under_addition_l1128_112856


namespace NUMINAMATH_CALUDE_sandy_jacket_return_l1128_112849

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := 12.14

/-- The net amount Sandy spent on clothes -/
def net_spent : ℝ := 18.70

/-- The amount Sandy received for returning the jacket -/
def jacket_return : ℝ := shorts_cost + shirt_cost - net_spent

theorem sandy_jacket_return : jacket_return = 7.43 := by
  sorry

end NUMINAMATH_CALUDE_sandy_jacket_return_l1128_112849


namespace NUMINAMATH_CALUDE_silver_division_problem_l1128_112812

theorem silver_division_problem (x y : ℤ) : 
  y = 7 * x + 4 ∧ y = 9 * x - 8 → y = 46 := by
sorry

end NUMINAMATH_CALUDE_silver_division_problem_l1128_112812


namespace NUMINAMATH_CALUDE_equation_solution_l1128_112828

theorem equation_solution :
  ∃ (x : ℝ), 
    (3*x - 1 ≥ 0) ∧ 
    (x + 4 > 0) ∧ 
    (Real.sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3*x - 1)) = 0) ∧
    (x = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1128_112828


namespace NUMINAMATH_CALUDE_import_export_scientific_notation_l1128_112807

def billion : ℝ := 1000000000

theorem import_export_scientific_notation (volume : ℝ) (h : volume = 214.7 * billion) :
  ∃ (a : ℝ) (n : ℤ), volume = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_import_export_scientific_notation_l1128_112807


namespace NUMINAMATH_CALUDE_tournament_games_count_l1128_112847

/-- Represents a basketball tournament with a preliminary round and main tournament. -/
structure BasketballTournament where
  preliminaryTeams : Nat
  preliminarySpots : Nat
  mainTournamentTeams : Nat

/-- Calculates the number of games in the preliminary round. -/
def preliminaryGames (t : BasketballTournament) : Nat :=
  t.preliminarySpots

/-- Calculates the number of games in the main tournament. -/
def mainTournamentGames (t : BasketballTournament) : Nat :=
  t.mainTournamentTeams - 1

/-- Calculates the total number of games in the entire tournament. -/
def totalGames (t : BasketballTournament) : Nat :=
  preliminaryGames t + mainTournamentGames t

/-- Theorem stating that the total number of games in the specific tournament setup is 17. -/
theorem tournament_games_count :
  ∃ (t : BasketballTournament),
    t.preliminaryTeams = 4 ∧
    t.preliminarySpots = 2 ∧
    t.mainTournamentTeams = 16 ∧
    totalGames t = 17 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_count_l1128_112847


namespace NUMINAMATH_CALUDE_expression_evaluation_l1128_112844

theorem expression_evaluation (b c : ℕ) (h1 : b = 2) (h2 : c = 3) : 
  (b^3 * b^4) + c^2 = 137 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1128_112844


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l1128_112860

theorem solve_equation_for_x :
  ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1200.0000000000002 ∧ X = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l1128_112860


namespace NUMINAMATH_CALUDE_limit_fraction_is_two_l1128_112824

theorem limit_fraction_is_two : ∀ ε > 0, ∃ N : ℕ, ∀ n > N,
  |((2 * n - 3 : ℝ) / (n + 2 : ℝ)) - 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_fraction_is_two_l1128_112824


namespace NUMINAMATH_CALUDE_not_in_range_iff_c_in_interval_l1128_112866

/-- The function g(x) defined in terms of c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 3

/-- Theorem stating that -3 is not in the range of g(x) iff c ∈ (-2√6, 2√6) -/
theorem not_in_range_iff_c_in_interval (c : ℝ) : 
  (∀ x : ℝ, g c x ≠ -3) ↔ c ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_not_in_range_iff_c_in_interval_l1128_112866


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1128_112871

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧ ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1128_112871


namespace NUMINAMATH_CALUDE_average_of_series_l1128_112841

/-- The average of a series z, 3z, 5z, 9z, and 17z is 7z -/
theorem average_of_series (z : ℝ) : (z + 3*z + 5*z + 9*z + 17*z) / 5 = 7*z := by
  sorry

end NUMINAMATH_CALUDE_average_of_series_l1128_112841


namespace NUMINAMATH_CALUDE_sought_hyperbola_satisfies_conditions_l1128_112804

/-- Given hyperbola equation -/
def given_hyperbola (x y : ℝ) : Prop :=
  x^2 / 5 - y^2 / 4 = 1

/-- Asymptotes of the given hyperbola -/
def given_asymptotes (x y : ℝ) : Prop :=
  y = (2 / Real.sqrt 5) * x ∨ y = -(2 / Real.sqrt 5) * x

/-- The equation of the sought hyperbola -/
def sought_hyperbola (x y : ℝ) : Prop :=
  5 * y^2 / 4 - x^2 = 1

/-- Theorem stating that the sought hyperbola satisfies the required conditions -/
theorem sought_hyperbola_satisfies_conditions :
  (∀ x y : ℝ, given_asymptotes x y ↔ (y = (2 / Real.sqrt 5) * x ∨ y = -(2 / Real.sqrt 5) * x)) ∧
  sought_hyperbola 2 2 :=
sorry


end NUMINAMATH_CALUDE_sought_hyperbola_satisfies_conditions_l1128_112804


namespace NUMINAMATH_CALUDE_a_most_stable_l1128_112818

/-- Represents a participant in the shooting test -/
inductive Participant
| A
| B
| C
| D

/-- Returns the variance of a participant's scores -/
def variance (p : Participant) : ℝ :=
  match p with
  | Participant.A => 0.54
  | Participant.B => 0.61
  | Participant.C => 0.7
  | Participant.D => 0.63

/-- Determines if a participant has the most stable performance -/
def has_most_stable_performance (p : Participant) : Prop :=
  ∀ q : Participant, variance p ≤ variance q

/-- Theorem: A has the most stable shooting performance -/
theorem a_most_stable : has_most_stable_performance Participant.A := by
  sorry

end NUMINAMATH_CALUDE_a_most_stable_l1128_112818


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1128_112840

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 7 + a^(x - 1)
  f 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1128_112840


namespace NUMINAMATH_CALUDE_solve_equation_l1128_112888

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1128_112888


namespace NUMINAMATH_CALUDE_total_children_count_l1128_112811

/-- The number of toy cars given to boys -/
def toy_cars : ℕ := 134

/-- The number of dolls given to girls -/
def dolls : ℕ := 269

/-- The number of board games given to both boys and girls -/
def board_games : ℕ := 87

/-- Every child received only one toy -/
axiom one_toy_per_child : True

/-- The total number of children attending the event -/
def total_children : ℕ := toy_cars + dolls

theorem total_children_count : total_children = 403 := by sorry

end NUMINAMATH_CALUDE_total_children_count_l1128_112811


namespace NUMINAMATH_CALUDE_sixteenth_root_of_sixteen_l1128_112809

theorem sixteenth_root_of_sixteen (n : ℝ) : (16 : ℝ) ^ (1/4 : ℝ) = 2^n → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixteenth_root_of_sixteen_l1128_112809


namespace NUMINAMATH_CALUDE_statement_B_statement_D_l1128_112869

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Statement B
theorem statement_B :
  perpendicular m n →
  perpendicular_plane m α →
  perpendicular_plane n β →
  perpendicular_planes α β :=
sorry

-- Statement D
theorem statement_D :
  parallel_planes α β →
  perpendicular_plane m α →
  parallel n β →
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_statement_B_statement_D_l1128_112869


namespace NUMINAMATH_CALUDE_max_value_implies_m_l1128_112864

/-- Given that the maximum value of f(x) = sin(x + π/2) + cos(x - π/2) + m is 2√2, prove that m = √2 -/
theorem max_value_implies_m (f : ℝ → ℝ) (m : ℝ) 
  (h : ∀ x, f x = Real.sin (x + π/2) + Real.cos (x - π/2) + m) 
  (h_max : ∃ x₀, ∀ x, f x ≤ f x₀ ∧ f x₀ = 2 * Real.sqrt 2) : 
  m = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l1128_112864


namespace NUMINAMATH_CALUDE_john_photos_count_l1128_112820

/-- The number of photos each person brings and the total slots in the album --/
def photo_problem (cristina_photos sarah_photos clarissa_photos total_slots : ℕ) : Prop :=
  ∃ john_photos : ℕ,
    john_photos = total_slots - (cristina_photos + sarah_photos + clarissa_photos)

/-- Theorem stating that John brings 10 photos given the problem conditions --/
theorem john_photos_count :
  photo_problem 7 9 14 40 → ∃ john_photos : ℕ, john_photos = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_john_photos_count_l1128_112820


namespace NUMINAMATH_CALUDE_claudia_earnings_l1128_112850

def class_price : ℕ := 10
def saturday_attendance : ℕ := 20
def sunday_attendance : ℕ := saturday_attendance / 2

def total_earnings : ℕ := class_price * (saturday_attendance + sunday_attendance)

theorem claudia_earnings : total_earnings = 300 := by
  sorry

end NUMINAMATH_CALUDE_claudia_earnings_l1128_112850
