import Mathlib

namespace NUMINAMATH_CALUDE_g_of_7_eq_92_l750_75016

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_of_7_eq_92 : g 7 = 92 := by sorry

end NUMINAMATH_CALUDE_g_of_7_eq_92_l750_75016


namespace NUMINAMATH_CALUDE_sin_18_cos_36_equals_quarter_l750_75059

theorem sin_18_cos_36_equals_quarter : Real.sin (18 * π / 180) * Real.cos (36 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_36_equals_quarter_l750_75059


namespace NUMINAMATH_CALUDE_specific_pyramid_base_edge_length_l750_75001

/-- A square pyramid with a sphere inside --/
structure PyramidWithSphere where
  pyramid_height : ℝ
  sphere_radius : ℝ
  sphere_tangent_to_faces : Bool
  sphere_contacts_base : Bool

/-- Calculates the edge length of the pyramid's base --/
def base_edge_length (p : PyramidWithSphere) : ℝ :=
  sorry

/-- Theorem stating the base edge length of the specific pyramid --/
theorem specific_pyramid_base_edge_length :
  let p : PyramidWithSphere := {
    pyramid_height := 9,
    sphere_radius := 3,
    sphere_tangent_to_faces := true,
    sphere_contacts_base := true
  }
  base_edge_length p = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_base_edge_length_l750_75001


namespace NUMINAMATH_CALUDE_professor_k_lectures_l750_75042

def num_jokes : ℕ := 8

theorem professor_k_lectures (num_jokes : ℕ) (h : num_jokes = 8) :
  (Finset.sum (Finset.range 2) (λ i => Nat.choose num_jokes (i + 2))) = 84 := by
  sorry

end NUMINAMATH_CALUDE_professor_k_lectures_l750_75042


namespace NUMINAMATH_CALUDE_angle_inequality_l750_75096

open Real

theorem angle_inequality (θ : Real) (h1 : 3 * π / 4 < θ) (h2 : θ < π) :
  ∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * sin θ - x * (1 - x) + (1 - x)^2 * cos θ + 2 * x * (1 - x) * sqrt (cos θ * sin θ) > 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l750_75096


namespace NUMINAMATH_CALUDE_base6_addition_proof_l750_75037

/-- Convert a base 6 number to base 10 -/
def base6to10 (x y z : Nat) : Nat :=
  x * 36 + y * 6 + z

/-- Addition in base 6 -/
def addBase6 (x₁ y₁ z₁ x₂ y₂ z₂ : Nat) : Nat × Nat × Nat :=
  let sum := base6to10 x₁ y₁ z₁ + base6to10 x₂ y₂ z₂
  (sum / 36, (sum % 36) / 6, sum % 6)

theorem base6_addition_proof (C D : Nat) :
  C < 6 ∧ D < 6 ∧
  addBase6 5 C D 0 5 2 = (1, 2, C) →
  C + D = 5 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_proof_l750_75037


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l750_75088

theorem algebraic_expression_value (x : ℝ) (h : 3 * x^2 - x - 1 = 0) :
  (2 * x + 3) * (2 * x - 3) - 2 * x * (1 - x) = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l750_75088


namespace NUMINAMATH_CALUDE_water_bucket_ratio_l750_75049

/-- Given two partially filled buckets of water, a and b, prove that the ratio of water in bucket b 
    to bucket a after transferring 6 liters from b to a is 1:2, given the initial conditions. -/
theorem water_bucket_ratio : 
  ∀ (a b : ℝ),
  a = 13.2 →
  a - 6 = (1/3) * (b + 6) →
  (b - 6) / (a + 6) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_water_bucket_ratio_l750_75049


namespace NUMINAMATH_CALUDE_fishing_catches_proof_l750_75055

theorem fishing_catches_proof (a b c d : ℕ) : 
  a + b = 7 ∧ 
  a + c = 9 ∧ 
  a + d = 14 ∧ 
  b + c = 14 ∧ 
  b + d = 19 ∧ 
  c + d = 21 →
  (a = 1 ∧ b = 6 ∧ c = 8 ∧ d = 13) ∨
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 13) ∨
  (a = 6 ∧ b = 1 ∧ c = 8 ∧ d = 13) ∨
  (a = 6 ∧ b = 8 ∧ c = 1 ∧ d = 13) ∨
  (a = 8 ∧ b = 1 ∧ c = 6 ∧ d = 13) ∨
  (a = 8 ∧ b = 6 ∧ c = 1 ∧ d = 13) :=
by sorry

end NUMINAMATH_CALUDE_fishing_catches_proof_l750_75055


namespace NUMINAMATH_CALUDE_nine_b_equals_eighteen_l750_75073

theorem nine_b_equals_eighteen (a b : ℤ) 
  (h1 : 6 * a + 3 * b = 0) 
  (h2 : b - 3 = a) : 
  9 * b = 18 := by
sorry

end NUMINAMATH_CALUDE_nine_b_equals_eighteen_l750_75073


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l750_75009

/-- Represents a four-digit number in the form 28a3 where a is a digit -/
def fourDigitNumber (a : ℕ) : ℕ := 2803 + 100 * a

/-- The denominator of the original fraction -/
def denominator : ℕ := 7276

/-- Theorem stating that 641 is the solution to the fraction equation -/
theorem fraction_equation_solution :
  ∃ (a : ℕ), a < 10 ∧ 
  (fourDigitNumber a - 641) * 7 = 2 * (denominator + 641) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l750_75009


namespace NUMINAMATH_CALUDE_difference_of_squares_l750_75010

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 24) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 192 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l750_75010


namespace NUMINAMATH_CALUDE_binomial_1493_1492_l750_75062

theorem binomial_1493_1492 : Nat.choose 1493 1492 = 1493 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1493_1492_l750_75062


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l750_75057

theorem point_in_third_quadrant (a b : ℝ) : a + b < 0 → a * b > 0 → a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l750_75057


namespace NUMINAMATH_CALUDE_race_track_width_l750_75070

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 880 →
  outer_radius = 140.0563499208679 →
  ∃ width : ℝ, abs (width - ((2 * Real.pi * outer_radius - inner_circumference) / 2)) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_race_track_width_l750_75070


namespace NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l750_75092

theorem probability_second_genuine_given_first_genuine 
  (total_products : ℕ) 
  (genuine_products : ℕ) 
  (defective_products : ℕ) 
  (h1 : total_products = genuine_products + defective_products)
  (h2 : genuine_products = 6)
  (h3 : defective_products = 4) :
  (genuine_products - 1) / (total_products - 1) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_genuine_given_first_genuine_l750_75092


namespace NUMINAMATH_CALUDE_average_reading_days_is_64_l750_75078

/-- Represents the reading speed ratio between Emery and Serena for books -/
def book_speed_ratio : ℚ := 5

/-- Represents the reading speed ratio between Emery and Serena for articles -/
def article_speed_ratio : ℚ := 3

/-- Represents the number of days it takes Emery to read the book -/
def emery_book_days : ℕ := 20

/-- Represents the number of days it takes Emery to read the article -/
def emery_article_days : ℕ := 2

/-- Calculates the average number of days for Emery and Serena to read both the book and the article -/
def average_reading_days : ℚ := 
  let serena_book_days := emery_book_days * book_speed_ratio
  let serena_article_days := emery_article_days * article_speed_ratio
  let emery_total_days := emery_book_days + emery_article_days
  let serena_total_days := serena_book_days + serena_article_days
  (emery_total_days + serena_total_days) / 2

theorem average_reading_days_is_64 : average_reading_days = 64 := by
  sorry

end NUMINAMATH_CALUDE_average_reading_days_is_64_l750_75078


namespace NUMINAMATH_CALUDE_justin_and_tim_same_game_l750_75060

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem justin_and_tim_same_game :
  let total_combinations := Nat.choose total_players players_per_game
  let games_with_justin_and_tim := Nat.choose (total_players - 2) (players_per_game - 2)
  games_with_justin_and_tim = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_and_tim_same_game_l750_75060


namespace NUMINAMATH_CALUDE_largest_room_length_l750_75044

theorem largest_room_length 
  (largest_width : ℝ) 
  (smallest_width smallest_length : ℝ) 
  (area_difference : ℝ) 
  (h1 : largest_width = 45)
  (h2 : smallest_width = 15)
  (h3 : smallest_length = 8)
  (h4 : area_difference = 1230)
  (h5 : largest_width * largest_length - smallest_width * smallest_length = area_difference) :
  largest_length = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_room_length_l750_75044


namespace NUMINAMATH_CALUDE_all_dice_same_probability_l750_75024

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being tossed -/
def numberOfDice : ℕ := 5

/-- The probability of all dice showing the same number -/
def probabilityAllSame : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

theorem all_dice_same_probability :
  probabilityAllSame = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_all_dice_same_probability_l750_75024


namespace NUMINAMATH_CALUDE_dividend_calculation_l750_75084

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 6) : 
  divisor * quotient + remainder = 159 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l750_75084


namespace NUMINAMATH_CALUDE_intersection_M_N_l750_75023

def M : Set ℝ := {0, 1, 2, 3}
def N : Set ℝ := {x | x^2 + x - 6 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l750_75023


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l750_75029

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that the polynomial satisfies the given equation -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z ≠ 0 → 2*x*y*z = x + y + z →
    P x / (y*z) + P y / (z*x) + P z / (x*y) = P (x - y) + P (y - z) + P (z - x)

/-- The theorem stating that any polynomial satisfying the equation must be of the form c(x^2 + 3) -/
theorem polynomial_equation_solution (P : RealPolynomial) 
    (h : SatisfiesEquation P) : 
    ∃ (c : ℝ), ∀ x, P x = c * (x^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l750_75029


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l750_75075

/-- A rectangular prism with dimensions 3, 4, and 5 units. -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The number of face diagonals in a rectangular prism. -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism. -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism. -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

/-- Theorem: The total number of diagonals in a rectangular prism with dimensions 3, 4, and 5 is 16. -/
theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l750_75075


namespace NUMINAMATH_CALUDE_bathing_suits_for_women_l750_75094

theorem bathing_suits_for_women (total : ℕ) (men : ℕ) (women : ℕ) : 
  total = 19766 → men = 14797 → women = total - men → women = 4969 := by
sorry

end NUMINAMATH_CALUDE_bathing_suits_for_women_l750_75094


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l750_75020

theorem sarahs_bowling_score (greg_score sarah_score : ℝ) : 
  sarah_score = greg_score + 50 →
  (greg_score + sarah_score) / 2 = 122.4 →
  sarah_score = 147.4 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l750_75020


namespace NUMINAMATH_CALUDE_mps_to_kmph_conversion_l750_75082

-- Define the conversion factor from mps to kmph
def mps_to_kmph_factor : ℝ := 3.6

-- Define the speed in mps
def speed_mps : ℝ := 15

-- Define the speed in kmph
def speed_kmph : ℝ := 54

-- Theorem to prove the conversion
theorem mps_to_kmph_conversion :
  speed_mps * mps_to_kmph_factor = speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_mps_to_kmph_conversion_l750_75082


namespace NUMINAMATH_CALUDE_f_max_value_l750_75013

/-- The function f(x) = 10x - 5x^2 -/
def f (x : ℝ) : ℝ := 10 * x - 5 * x^2

/-- The maximum value of f(x) for any real x is 5 -/
theorem f_max_value : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l750_75013


namespace NUMINAMATH_CALUDE_hundredth_term_is_397_l750_75083

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- The 100th term of the specific arithmetic sequence -/
def hundredthTerm : ℝ := arithmeticSequenceTerm 1 4 100

theorem hundredth_term_is_397 : hundredthTerm = 397 := by sorry

end NUMINAMATH_CALUDE_hundredth_term_is_397_l750_75083


namespace NUMINAMATH_CALUDE_set_intersection_example_l750_75089

theorem set_intersection_example : 
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {0, 2, 4}
  A ∩ B = {0, 2} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l750_75089


namespace NUMINAMATH_CALUDE_correct_guess_probability_l750_75035

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts -/
def guess_probability : ℚ := 1/5

/-- The number of possible digits for each position in the password -/
def digit_options : ℕ := 10

/-- The number of attempts allowed to guess the last digit -/
def max_attempts : ℕ := 2

theorem correct_guess_probability :
  guess_probability = 1 / digit_options + (1 - 1 / digit_options) * (1 / (digit_options - 1)) :=
sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l750_75035


namespace NUMINAMATH_CALUDE_range_of_fraction_l750_75099

-- Define the quadratic equation
def quadratic (a b x : ℝ) : Prop := x^2 + a*x + 2*b - 2 = 0

-- Define the theorem
theorem range_of_fraction (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic a b x₁ ∧ quadratic a b x₂ ∧
    0 < x₁ ∧ x₁ < 1 ∧ 
    1 < x₂ ∧ x₂ < 2) →
  1/2 < (b - 4) / (a - 1) ∧ (b - 4) / (a - 1) < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l750_75099


namespace NUMINAMATH_CALUDE_triangle_property_l750_75033

theorem triangle_property (A B C : ℝ) (hABC : A + B + C = π) 
  (hDot : (Real.cos A * Real.cos C + Real.sin A * Real.sin C) * 
          (Real.cos A * Real.cos B + Real.sin A * Real.sin B) = 
          3 * (Real.cos B * Real.cos A + Real.sin B * Real.sin A) * 
             (Real.cos B * Real.cos C + Real.sin B * Real.sin C)) :
  (Real.tan B = 3 * Real.tan A) ∧ 
  (Real.cos C = Real.sqrt 5 / 5 → A = π / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l750_75033


namespace NUMINAMATH_CALUDE_donut_distribution_unique_l750_75021

/-- The distribution of donuts among five people -/
def DonutDistribution : Type := ℕ × ℕ × ℕ × ℕ × ℕ

/-- The total number of donuts -/
def total_donuts : ℕ := 60

/-- Check if a distribution satisfies the given conditions -/
def is_valid_distribution (d : DonutDistribution) : Prop :=
  let (alpha, beta, gamma, delta, epsilon) := d
  delta = 8 ∧
  beta = 3 * gamma ∧
  alpha = 2 * delta ∧
  epsilon = gamma - 4 ∧
  alpha + beta + gamma + delta + epsilon = total_donuts

/-- The correct distribution of donuts -/
def correct_distribution : DonutDistribution := (16, 24, 8, 8, 4)

/-- Theorem stating that the correct distribution is the only valid distribution -/
theorem donut_distribution_unique :
  ∀ d : DonutDistribution, is_valid_distribution d → d = correct_distribution := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_unique_l750_75021


namespace NUMINAMATH_CALUDE_hyperbola_single_intersection_lines_l750_75031

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (positive_a : a > 0)
  (positive_b : b > 0)

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ
  c : ℝ

/-- Function to check if a line intersects a hyperbola at only one point -/
def intersects_at_one_point (h : Hyperbola) (l : Line) : Prop :=
  ∃! p : Point, p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1 ∧ p.y = l.m * p.x + l.c

/-- Theorem statement -/
theorem hyperbola_single_intersection_lines 
  (h : Hyperbola) 
  (p : Point) 
  (h_eq : h.a = 1 ∧ h.b = 2) 
  (p_eq : p.x = 1 ∧ p.y = 1) :
  ∃! (lines : Finset Line), 
    lines.card = 4 ∧ 
    ∀ l ∈ lines, intersects_at_one_point h l ∧ p.y = l.m * p.x + l.c :=
sorry

end NUMINAMATH_CALUDE_hyperbola_single_intersection_lines_l750_75031


namespace NUMINAMATH_CALUDE_simplify_expression_l750_75085

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a/b + b/a - 2/(a*b) = 1 - 1/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l750_75085


namespace NUMINAMATH_CALUDE_sum_greatest_odd_divisors_l750_75030

/-- The sum of the greatest odd divisors of natural numbers from 1 to 2^n -/
def S (n : ℕ) : ℕ :=
  (Finset.range (2^n + 1)).sum (λ m => Nat.gcd m ((2^n).div m))

/-- For any natural number n, 3 times the sum of the greatest odd divisors
    of natural numbers from 1 to 2^n equals 4^n + 2 -/
theorem sum_greatest_odd_divisors (n : ℕ) : 3 * S n = 4^n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_greatest_odd_divisors_l750_75030


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l750_75046

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (a / (1 - r) = 10) →
  ((a + 4) / (1 - r) = 15) →
  r = 1/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l750_75046


namespace NUMINAMATH_CALUDE_desired_depth_is_18_l750_75081

/-- Represents the digging scenario with given parameters -/
structure DiggingScenario where
  initial_men : ℕ
  initial_hours : ℕ
  initial_depth : ℕ
  new_hours : ℕ
  extra_men : ℕ

/-- Calculates the desired depth for a given digging scenario -/
def desired_depth (scenario : DiggingScenario) : ℚ :=
  (scenario.initial_men * scenario.initial_hours * scenario.initial_depth : ℚ) /
  ((scenario.initial_men + scenario.extra_men) * scenario.new_hours)

/-- Theorem stating that the desired depth for the given scenario is 18 meters -/
theorem desired_depth_is_18 (scenario : DiggingScenario) 
    (h1 : scenario.initial_men = 9)
    (h2 : scenario.initial_hours = 8)
    (h3 : scenario.initial_depth = 30)
    (h4 : scenario.new_hours = 6)
    (h5 : scenario.extra_men = 11) :
  desired_depth scenario = 18 := by
  sorry

#eval desired_depth { initial_men := 9, initial_hours := 8, initial_depth := 30, new_hours := 6, extra_men := 11 }

end NUMINAMATH_CALUDE_desired_depth_is_18_l750_75081


namespace NUMINAMATH_CALUDE_segment_length_on_ellipse_l750_75066

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem segment_length_on_ellipse :
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  (∃ t : ℝ, A = F₁ + t • (B - F₁)) →  -- A, B, and F₁ are collinear
  distance F₂ A + distance F₂ B = 12 →
  distance A B = 8 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_on_ellipse_l750_75066


namespace NUMINAMATH_CALUDE_blue_sequin_rows_l750_75053

/-- The number of sequins in each row of blue sequins -/
def blue_sequins_per_row : ℕ := 8

/-- The number of rows of purple sequins -/
def purple_rows : ℕ := 5

/-- The number of sequins in each row of purple sequins -/
def purple_sequins_per_row : ℕ := 12

/-- The number of rows of green sequins -/
def green_rows : ℕ := 9

/-- The number of sequins in each row of green sequins -/
def green_sequins_per_row : ℕ := 6

/-- The total number of sequins -/
def total_sequins : ℕ := 162

/-- Theorem: The number of rows of blue sequins is 6 -/
theorem blue_sequin_rows : 
  (total_sequins - (purple_rows * purple_sequins_per_row + green_rows * green_sequins_per_row)) / blue_sequins_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_sequin_rows_l750_75053


namespace NUMINAMATH_CALUDE_pizza_consumption_proof_l750_75047

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem pizza_consumption_proof :
  let initial_fraction : ℚ := 1/3
  let remaining_fraction : ℚ := 2/3
  let num_trips : ℕ := 6
  geometric_sum initial_fraction remaining_fraction num_trips = 364/729 := by
  sorry

end NUMINAMATH_CALUDE_pizza_consumption_proof_l750_75047


namespace NUMINAMATH_CALUDE_completed_square_q_value_l750_75036

theorem completed_square_q_value (a b c : ℝ) (h : a = 1 ∧ b = -6 ∧ c = 5) :
  ∃ (p q : ℝ), ∀ x, (x^2 + b*x + c = 0 ↔ (x + p)^2 = q) ∧ q = 4 := by
  sorry

end NUMINAMATH_CALUDE_completed_square_q_value_l750_75036


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l750_75067

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 10 = 3 →
  a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l750_75067


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l750_75051

def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_proof (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_ratio : ∀ n, 2 * a n = 3 * a (n + 1))
  (h_product : a 2 * a 5 = 8 / 27) :
  is_geometric a ∧ a 6 = 16 / 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l750_75051


namespace NUMINAMATH_CALUDE_root_difference_l750_75095

theorem root_difference (r s : ℝ) : 
  (∃ x, (1984 * x)^2 - 1983 * 1985 * x - 1 = 0 ∧ r = x ∧ 
    ∀ y, ((1984 * y)^2 - 1983 * 1985 * y - 1 = 0 → y ≤ r)) →
  (∃ x, 1983 * x^2 - 1984 * x + 1 = 0 ∧ s = x ∧ 
    ∀ y, (1983 * y^2 - 1984 * y + 1 = 0 → s ≤ y)) →
  r - s = 1982 / 1983 := by
sorry

end NUMINAMATH_CALUDE_root_difference_l750_75095


namespace NUMINAMATH_CALUDE_inner_square_probability_l750_75045

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Checks if a square is on the perimeter or center lines -/
def is_on_perimeter_or_center (b : Board) (row col : ℕ) : Prop :=
  row = 1 ∨ row = b.size ∨ col = 1 ∨ col = b.size ∨
  row = b.size / 2 ∨ row = b.size / 2 + 1 ∨
  col = b.size / 2 ∨ col = b.size / 2 + 1

/-- Counts squares not on perimeter or center lines -/
def count_inner_squares (b : Board) : ℕ :=
  (b.size - 4) * (b.size - 4)

/-- The main theorem -/
theorem inner_square_probability (b : Board) (h : b.size = 10) :
  (count_inner_squares b : ℚ) / (b.size * b.size : ℚ) = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_inner_square_probability_l750_75045


namespace NUMINAMATH_CALUDE_cone_base_radius_l750_75077

/-- Represents a cone with given properties -/
structure Cone where
  surface_area : ℝ
  lateral_surface_semicircle : Prop

/-- Theorem: Given a cone with surface area 12π and lateral surface unfolding into a semicircle, 
    the radius of its base circle is 2 -/
theorem cone_base_radius 
  (cone : Cone) 
  (h1 : cone.surface_area = 12 * Real.pi) 
  (h2 : cone.lateral_surface_semicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r > 0 ∧ 
  cone.surface_area = Real.pi * r^2 + Real.pi * r * (2 * r) := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l750_75077


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l750_75091

def expand_polynomial (x : ℝ) : ℝ := (2*x+5)*(3*x^2+x+6) - 4*(x^3-3*x^2+5*x-1)

theorem nonzero_terms_count :
  ∃ (a b c d : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  ∀ x, expand_polynomial x = a*x^3 + b*x^2 + c*x + d :=
sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l750_75091


namespace NUMINAMATH_CALUDE_sports_meeting_participation_l750_75090

theorem sports_meeting_participation (field_events track_events both : ℕ) 
  (h1 : field_events = 15)
  (h2 : track_events = 13)
  (h3 : both = 5) :
  field_events + track_events - both = 23 :=
by sorry

end NUMINAMATH_CALUDE_sports_meeting_participation_l750_75090


namespace NUMINAMATH_CALUDE_circle_properties_l750_75076

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define the center coordinates
def center : ℝ × ℝ := (-1, 2)

-- Define the radius
def radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l750_75076


namespace NUMINAMATH_CALUDE_function_inequality_l750_75050

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x < 1) (h_f3 : f 3 = 4) :
  ∀ x, f (x + 1) < x + 2 ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l750_75050


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l750_75087

/-- The eccentricity of a hyperbola with equation x^2/m - y^2/5 = 1 is 3/2,
    given that m > 0 and its right focus coincides with the focus of y^2 = 12x -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) : ∃ (a b c : ℝ),
  m = a^2 ∧
  b^2 = 5 ∧
  c^2 = a^2 + b^2 ∧
  c = 3 ∧
  c / a = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l750_75087


namespace NUMINAMATH_CALUDE_pizza_cost_l750_75054

theorem pizza_cost (total_cost : ℝ) (num_pizzas : ℕ) (h1 : total_cost = 24) (h2 : num_pizzas = 3) :
  total_cost / num_pizzas = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_l750_75054


namespace NUMINAMATH_CALUDE_intersection_M_N_l750_75027

def M : Set ℤ := {-1, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = x^2}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l750_75027


namespace NUMINAMATH_CALUDE_product_of_numbers_l750_75056

theorem product_of_numbers (x y : ℝ) : 
  x - y = 7 → x^2 + y^2 = 85 → x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l750_75056


namespace NUMINAMATH_CALUDE_wheat_flour_price_l750_75019

theorem wheat_flour_price (initial_amount : ℕ) (rice_price : ℕ) (rice_packets : ℕ)
  (soda_price : ℕ) (wheat_packets : ℕ) (remaining_balance : ℕ) :
  initial_amount = 500 →
  rice_price = 20 →
  rice_packets = 2 →
  soda_price = 150 →
  wheat_packets = 3 →
  remaining_balance = 235 →
  ∃ (wheat_price : ℕ),
    wheat_price * wheat_packets = initial_amount - remaining_balance - (rice_price * rice_packets + soda_price) ∧
    wheat_price = 25 := by
  sorry

#check wheat_flour_price

end NUMINAMATH_CALUDE_wheat_flour_price_l750_75019


namespace NUMINAMATH_CALUDE_cheryl_eggs_count_l750_75011

/-- The number of eggs found by Kevin -/
def kevin_eggs : ℕ := 5

/-- The number of eggs found by Bonnie -/
def bonnie_eggs : ℕ := 13

/-- The number of eggs found by George -/
def george_eggs : ℕ := 9

/-- The number of additional eggs Cheryl found compared to the others -/
def cheryl_additional_eggs : ℕ := 29

/-- Theorem stating that Cheryl found 56 eggs -/
theorem cheryl_eggs_count : 
  kevin_eggs + bonnie_eggs + george_eggs + cheryl_additional_eggs = 56 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_eggs_count_l750_75011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l750_75041

/-- Given an arithmetic sequence with first term 3 and common difference 4,
    the 20th term of the sequence is 79. -/
theorem arithmetic_sequence_20th_term :
  let a₁ : ℕ := 3  -- first term
  let d : ℕ := 4   -- common difference
  let n : ℕ := 20  -- term number we're looking for
  let aₙ : ℕ := a₁ + (n - 1) * d  -- formula for nth term of arithmetic sequence
  aₙ = 79 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l750_75041


namespace NUMINAMATH_CALUDE_parabola_directrix_coefficient_l750_75048

/-- For a parabola y = ax^2 with directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix_coefficient : 
  ∀ (a : ℝ), (∀ x y : ℝ, y = a * x^2) → 
  (∃ k : ℝ, k = 2 ∧ ∀ x : ℝ, k = -(1 / (4 * a))) → 
  a = -1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_coefficient_l750_75048


namespace NUMINAMATH_CALUDE_exists_monochromatic_triplet_l750_75097

/-- A coloring of natural numbers using two colors. -/
def Coloring := ℕ → Bool

/-- Predicate to check if three natural numbers form a valid triplet. -/
def ValidTriplet (x y z : ℕ) : Prop :=
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x * y = z^2

/-- Theorem stating that for any two-color painting of natural numbers,
    there always exist three distinct natural numbers x, y, and z
    of the same color such that xy = z^2. -/
theorem exists_monochromatic_triplet (c : Coloring) :
  ∃ x y z : ℕ, ValidTriplet x y z ∧ c x = c y ∧ c y = c z :=
sorry

end NUMINAMATH_CALUDE_exists_monochromatic_triplet_l750_75097


namespace NUMINAMATH_CALUDE_sequence_negative_start_l750_75093

def sequence_term (n : ℤ) : ℤ := 21 + 4*n - n^2

theorem sequence_negative_start :
  ∀ n : ℕ, n ≥ 8 → sequence_term n < 0 ∧ 
  ∀ k : ℕ, k < 8 → sequence_term k ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_negative_start_l750_75093


namespace NUMINAMATH_CALUDE_average_sale_calculation_l750_75003

def sales : List ℕ := [6535, 6927, 6855, 7230, 6562]
def required_sale : ℕ := 4891
def num_months : ℕ := 6

theorem average_sale_calculation :
  (sales.sum + required_sale) / num_months = 6500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_calculation_l750_75003


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l750_75005

theorem geometric_series_first_term
  (r : ℝ)
  (hr : |r| < 1)
  (h_sum : (∑' n, r^n) * a = 15)
  (h_sum_squares : (∑' n, (r^n)^2) * a^2 = 45) :
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l750_75005


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_23_l750_75017

/-- A scalene triangle with prime side lengths greater than 3 and prime perimeter. -/
structure ScaleneTriangle where
  /-- First side length -/
  a : ℕ
  /-- Second side length -/
  b : ℕ
  /-- Third side length -/
  c : ℕ
  /-- Proof that a is prime -/
  a_prime : Nat.Prime a
  /-- Proof that b is prime -/
  b_prime : Nat.Prime b
  /-- Proof that c is prime -/
  c_prime : Nat.Prime c
  /-- Proof that a is greater than 3 -/
  a_gt_three : a > 3
  /-- Proof that b is greater than 3 -/
  b_gt_three : b > 3
  /-- Proof that c is greater than 3 -/
  c_gt_three : c > 3
  /-- Proof that a, b, and c are distinct -/
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
  /-- Proof that a, b, and c form a valid triangle -/
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a
  /-- Proof that the perimeter is prime -/
  perimeter_prime : Nat.Prime (a + b + c)

/-- The smallest possible perimeter of a scalene triangle with the given conditions is 23. -/
theorem smallest_perimeter_is_23 : ∀ t : ScaleneTriangle, t.a + t.b + t.c ≥ 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_23_l750_75017


namespace NUMINAMATH_CALUDE_jason_retirement_age_l750_75014

def military_career (join_age : ℕ) (years_to_chief : ℕ) (years_after_master_chief : ℕ) : ℕ → Prop :=
  fun retirement_age =>
    ∃ (years_to_master_chief : ℕ),
      years_to_master_chief = years_to_chief + (years_to_chief * 25 / 100) ∧
      retirement_age = join_age + years_to_chief + years_to_master_chief + years_after_master_chief

theorem jason_retirement_age :
  military_career 18 8 10 46 := by
  sorry

end NUMINAMATH_CALUDE_jason_retirement_age_l750_75014


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l750_75007

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1 ∨ x ≤ -3}
def B : Set ℝ := {x | -4 < x ∧ x < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x ≤ -3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x < 0 ∨ x ≥ 1} := by sorry

-- Theorem for the complement of A with respect to ℝ
theorem complement_A : Aᶜ = {x | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_complement_A_l750_75007


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l750_75022

theorem smallest_n_for_exact_tax : ∃ (x : ℕ), 
  (x : ℚ) * (106 : ℚ) / (100 : ℚ) = 53 ∧ 
  ∀ (n : ℕ), n < 53 → ¬∃ (y : ℕ), (y : ℚ) * (106 : ℚ) / (100 : ℚ) = n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l750_75022


namespace NUMINAMATH_CALUDE_triangle_side_length_l750_75015

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c - t.b = 2 ∧
  Real.cos (t.A / 2) = Real.sqrt 3 / 3 ∧
  1/2 * t.b * t.c * Real.sin t.A = 5 * Real.sqrt 2

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  triangle_conditions t → t.a = 2 * Real.sqrt 11 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l750_75015


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_l750_75002

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 11 = 36) (h3 : a 8 = 24) : a 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_l750_75002


namespace NUMINAMATH_CALUDE_cement_calculation_l750_75012

/-- The renovation project requires materials in truck-loads -/
structure RenovationMaterials where
  total : ℚ
  sand : ℚ
  dirt : ℚ

/-- Calculate the truck-loads of cement required for the renovation project -/
def cement_required (materials : RenovationMaterials) : ℚ :=
  materials.total - (materials.sand + materials.dirt)

theorem cement_calculation (materials : RenovationMaterials) 
  (h1 : materials.total = 0.6666666666666666)
  (h2 : materials.sand = 0.16666666666666666)
  (h3 : materials.dirt = 0.3333333333333333) :
  cement_required materials = 0.1666666666666666 := by
  sorry

#eval cement_required ⟨0.6666666666666666, 0.16666666666666666, 0.3333333333333333⟩

end NUMINAMATH_CALUDE_cement_calculation_l750_75012


namespace NUMINAMATH_CALUDE_parabola_shift_l750_75079

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the horizontal shift
def shift : ℝ := 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - shift)^2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - shift) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l750_75079


namespace NUMINAMATH_CALUDE_sugar_delivery_problem_l750_75072

/-- Represents the sugar delivery problem -/
def SugarDelivery (total_bags : ℕ) (total_weight : ℝ) (granulated_ratio : ℝ) (sugar_mass_ratio : ℝ) : Prop :=
  ∃ (sugar_bags : ℕ) (granulated_bags : ℕ) (sugar_weight : ℝ) (granulated_weight : ℝ),
    -- Total number of bags
    sugar_bags + granulated_bags = total_bags ∧
    -- Granulated sugar bags ratio
    granulated_bags = (1 + granulated_ratio) * sugar_bags ∧
    -- Total weight
    sugar_weight + granulated_weight = total_weight ∧
    -- Mass ratio between sugar and granulated sugar bags
    sugar_weight * granulated_bags = sugar_mass_ratio * granulated_weight * sugar_bags ∧
    -- Correct weights
    sugar_weight = 3 ∧ granulated_weight = 1.8

theorem sugar_delivery_problem :
  SugarDelivery 63 4.8 0.25 0.75 :=
sorry

end NUMINAMATH_CALUDE_sugar_delivery_problem_l750_75072


namespace NUMINAMATH_CALUDE_dave_shows_per_week_l750_75052

theorem dave_shows_per_week :
  let strings_per_show : ℕ := 2
  let total_weeks : ℕ := 12
  let total_strings : ℕ := 144
  let shows_per_week : ℕ := total_strings / (strings_per_show * total_weeks)
  shows_per_week = 6 :=
by sorry

end NUMINAMATH_CALUDE_dave_shows_per_week_l750_75052


namespace NUMINAMATH_CALUDE_carson_clawed_39_times_l750_75080

/-- The number of times Carson gets clawed in the zoo enclosure. -/
def total_claws (num_wombats : ℕ) (num_rheas : ℕ) (claws_per_wombat : ℕ) (claws_per_rhea : ℕ) : ℕ :=
  num_wombats * claws_per_wombat + num_rheas * claws_per_rhea

/-- Theorem stating that Carson gets clawed 39 times. -/
theorem carson_clawed_39_times :
  total_claws 9 3 4 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_carson_clawed_39_times_l750_75080


namespace NUMINAMATH_CALUDE_safari_count_difference_l750_75086

/-- The number of animals Josie counted on safari --/
structure SafariCount where
  antelopes : ℕ
  rabbits : ℕ
  hyenas : ℕ
  wild_dogs : ℕ
  leopards : ℕ

/-- The conditions of Josie's safari count --/
def safari_conditions (count : SafariCount) : Prop :=
  count.antelopes = 80 ∧
  count.rabbits = count.antelopes + 34 ∧
  count.hyenas < count.antelopes + count.rabbits ∧
  count.wild_dogs = count.hyenas + 50 ∧
  count.leopards * 2 = count.rabbits ∧
  count.antelopes + count.rabbits + count.hyenas + count.wild_dogs + count.leopards = 605

/-- The theorem stating the difference between hyenas and the sum of antelopes and rabbits --/
theorem safari_count_difference (count : SafariCount) 
  (h : safari_conditions count) : 
  count.antelopes + count.rabbits - count.hyenas = 42 := by
  sorry

end NUMINAMATH_CALUDE_safari_count_difference_l750_75086


namespace NUMINAMATH_CALUDE_spade_calculation_l750_75006

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 2 (spade 3 (spade 1 4)) = -46652 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l750_75006


namespace NUMINAMATH_CALUDE_sin_cos_sum_zero_l750_75008

theorem sin_cos_sum_zero : 
  Real.sin (35 * π / 6) + Real.cos (-11 * π / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_zero_l750_75008


namespace NUMINAMATH_CALUDE_playset_cost_indeterminate_l750_75038

theorem playset_cost_indeterminate 
  (lumber_inflation : ℝ) 
  (nails_inflation : ℝ) 
  (fabric_inflation : ℝ) 
  (total_increase : ℝ) 
  (h1 : lumber_inflation = 0.20)
  (h2 : nails_inflation = 0.10)
  (h3 : fabric_inflation = 0.05)
  (h4 : total_increase = 97) :
  ∃ (L N F : ℝ), 
    L * lumber_inflation + N * nails_inflation + F * fabric_inflation = total_increase ∧
    ∃ (L' N' F' : ℝ), 
      L' ≠ L ∧
      L' * lumber_inflation + N' * nails_inflation + F' * fabric_inflation = total_increase :=
by sorry

end NUMINAMATH_CALUDE_playset_cost_indeterminate_l750_75038


namespace NUMINAMATH_CALUDE_percentage_relationships_l750_75064

/-- Given the relationships between a, b, c, d, and e, prove the relative percentages. -/
theorem percentage_relationships (a b c d e : ℝ) 
  (hc_a : c = 0.25 * a)
  (hc_b : c = 0.5 * b)
  (hd_a : d = 0.4 * a)
  (hd_b : d = 0.2 * b)
  (he_d : e = 0.35 * d)
  (he_c : e = 0.15 * c) :
  b = 0.5 * a ∧ c = 0.625 * d ∧ d = (1 / 0.35) * e := by
  sorry


end NUMINAMATH_CALUDE_percentage_relationships_l750_75064


namespace NUMINAMATH_CALUDE_die_product_divisibility_l750_75071

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem die_product_divisibility :
  let die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  ∀ S : Finset ℕ, S ⊆ die_numbers → S.card = 7 →
    let product := S.prod id
    (is_divisible product 192) ∧
    (∀ n > 192, ∃ T : Finset ℕ, T ⊆ die_numbers ∧ T.card = 7 ∧ ¬(is_divisible (T.prod id) n)) :=
by sorry

end NUMINAMATH_CALUDE_die_product_divisibility_l750_75071


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l750_75040

theorem pasta_preference_ratio :
  ∀ (total students : ℕ) (pasta_types : ℕ) (spaghetti_pref manicotti_pref : ℕ),
    total = 800 →
    pasta_types = 5 →
    spaghetti_pref = 300 →
    manicotti_pref = 120 →
    (spaghetti_pref : ℚ) / manicotti_pref = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l750_75040


namespace NUMINAMATH_CALUDE_blue_to_purple_ratio_l750_75074

/-- Represents the number of beads of each color in a necklace. -/
structure BeadCounts where
  purple : ℕ
  blue : ℕ
  green : ℕ

/-- The properties of the necklace as described in the problem. -/
def necklace_properties (b : BeadCounts) : Prop :=
  b.purple = 7 ∧
  b.green = b.blue + 11 ∧
  b.purple + b.blue + b.green = 46

/-- The theorem stating the ratio of blue to purple beads is 2:1. -/
theorem blue_to_purple_ratio (b : BeadCounts) :
  necklace_properties b → b.blue = 2 * b.purple := by
  sorry

end NUMINAMATH_CALUDE_blue_to_purple_ratio_l750_75074


namespace NUMINAMATH_CALUDE_seventeenth_replacement_in_may_l750_75004

/-- Represents months of the year -/
inductive Month
| january | february | march | april | may | june | july 
| august | september | october | november | december

/-- Calculates the number of months after January for a given replacement number -/
def monthsAfterStart (replacementNumber : Nat) : Nat :=
  7 * (replacementNumber - 1)

/-- Converts a number of months after January to the corresponding Month -/
def monthsToMonth (months : Nat) : Month :=
  match months % 12 with
  | 0 => Month.january
  | 1 => Month.february
  | 2 => Month.march
  | 3 => Month.april
  | 4 => Month.may
  | 5 => Month.june
  | 6 => Month.july
  | 7 => Month.august
  | 8 => Month.september
  | 9 => Month.october
  | 10 => Month.november
  | _ => Month.december

theorem seventeenth_replacement_in_may : 
  monthsToMonth (monthsAfterStart 17) = Month.may := by
  sorry

end NUMINAMATH_CALUDE_seventeenth_replacement_in_may_l750_75004


namespace NUMINAMATH_CALUDE_dara_wait_time_l750_75032

/-- Calculates the number of years Dara has to wait to reach the adjusted minimum age for employment. -/
def years_to_wait (current_min_age : ℕ) (jane_age : ℕ) (tom_age_diff : ℕ) (old_min_age : ℕ) : ℕ :=
  let dara_current_age := (jane_age + 6) / 2 - 6
  let years_passed := tom_age_diff + jane_age - old_min_age
  let periods_passed := years_passed / 5
  let new_min_age := current_min_age + periods_passed
  new_min_age - dara_current_age

/-- The number of years Dara has to wait is 16. -/
theorem dara_wait_time : years_to_wait 25 28 10 24 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dara_wait_time_l750_75032


namespace NUMINAMATH_CALUDE_jack_grassy_time_is_six_l750_75026

/-- Represents the race up the hill -/
structure HillRace where
  jackSandyTime : ℝ
  jackSpeedIncrease : ℝ
  jillTotalTime : ℝ
  jillFinishDifference : ℝ

/-- Calculates Jack's time on the grassy second half of the hill -/
def jackGrassyTime (race : HillRace) : ℝ :=
  race.jillTotalTime - race.jillFinishDifference - race.jackSandyTime

/-- Theorem stating that Jack's time on the grassy second half is 6 seconds -/
theorem jack_grassy_time_is_six (race : HillRace) 
  (h1 : race.jackSandyTime = 19)
  (h2 : race.jackSpeedIncrease = 0.25)
  (h3 : race.jillTotalTime = 32)
  (h4 : race.jillFinishDifference = 7) :
  jackGrassyTime race = 6 := by
  sorry

#check jack_grassy_time_is_six

end NUMINAMATH_CALUDE_jack_grassy_time_is_six_l750_75026


namespace NUMINAMATH_CALUDE_inequality_proof_l750_75098

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b^2 / a + a^2 / b ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l750_75098


namespace NUMINAMATH_CALUDE_orchids_count_l750_75068

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  initial_roses : Nat
  initial_orchids : Nat
  current_roses : Nat
  orchid_rose_difference : Nat

/-- Calculates the current number of orchids in the vase -/
def current_orchids (vase : FlowerVase) : Nat :=
  vase.current_roses + vase.orchid_rose_difference

theorem orchids_count (vase : FlowerVase) 
  (h1 : vase.initial_roses = 7)
  (h2 : vase.initial_orchids = 12)
  (h3 : vase.current_roses = 11)
  (h4 : vase.orchid_rose_difference = 9) :
  current_orchids vase = 20 := by
  sorry

end NUMINAMATH_CALUDE_orchids_count_l750_75068


namespace NUMINAMATH_CALUDE_total_cost_calculation_l750_75034

/-- The total cost of buying mineral water and yogurt -/
def total_cost (m n : ℕ) : ℚ :=
  2.5 * m + 4 * n

/-- Theorem stating the total cost calculation -/
theorem total_cost_calculation (m n : ℕ) :
  total_cost m n = 2.5 * m + 4 * n := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l750_75034


namespace NUMINAMATH_CALUDE_n_squared_minus_one_divisible_by_24_l750_75063

theorem n_squared_minus_one_divisible_by_24 (n : ℤ) 
  (h1 : ¬ 2 ∣ n) (h2 : ¬ 3 ∣ n) : 24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_n_squared_minus_one_divisible_by_24_l750_75063


namespace NUMINAMATH_CALUDE_problem_solution_l750_75061

-- Define the function f
def f (a b x : ℝ) := |x - a| - |x + b|

-- Define the function g
def g (a b x : ℝ) := -x^2 - a*x - b

theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hmax_exists : ∃ x, f a b x = 3) 
  (hg_less_f : ∀ x ≥ a, g a b x < f a b x) : 
  (a + b = 3) ∧ (1/2 < a ∧ a < 3) := by
sorry


end NUMINAMATH_CALUDE_problem_solution_l750_75061


namespace NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l750_75058

/-- Proves that the total number of tickets sold is 336 given the specified conditions -/
theorem snooker_tournament_ticket_sales
  (vip_cost : ℕ)
  (general_cost : ℕ)
  (total_revenue : ℕ)
  (ticket_difference : ℕ)
  (h1 : vip_cost = 45)
  (h2 : general_cost = 20)
  (h3 : total_revenue = 7500)
  (h4 : ticket_difference = 276)
  (h5 : ∃ (vip general : ℕ),
    vip_cost * vip + general_cost * general = total_revenue ∧
    vip + ticket_difference = general) :
  ∃ (vip general : ℕ), vip + general = 336 := by
  sorry

end NUMINAMATH_CALUDE_snooker_tournament_ticket_sales_l750_75058


namespace NUMINAMATH_CALUDE_multiplication_fraction_equality_l750_75039

theorem multiplication_fraction_equality : 12 * (1 / 8) * 32 = 48 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_equality_l750_75039


namespace NUMINAMATH_CALUDE_optimal_planting_strategy_l750_75043

/-- Represents the cost and planting details for a flower planting project --/
structure FlowerPlanting where
  costA : ℝ  -- Cost per pot of type A flowers
  costB : ℝ  -- Cost per pot of type B flowers
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℝ  -- Survival rate of type A flowers
  survivalRateB : ℝ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Calculates the total cost of planting flowers --/
def totalCost (fp : FlowerPlanting) (potsA : ℕ) : ℝ :=
  fp.costA * potsA + fp.costB * (fp.totalPots - potsA)

/-- Calculates the number of pots to be replaced next year --/
def replacementPots (fp : FlowerPlanting) (potsA : ℕ) : ℝ :=
  (1 - fp.survivalRateA) * potsA + (1 - fp.survivalRateB) * (fp.totalPots - potsA)

/-- Theorem stating the optimal planting strategy and minimum cost --/
theorem optimal_planting_strategy (fp : FlowerPlanting) 
    (h1 : 3 * fp.costA + 4 * fp.costB = 360)
    (h2 : 4 * fp.costA + 3 * fp.costB = 340)
    (h3 : fp.totalPots = 600)
    (h4 : fp.survivalRateA = 0.7)
    (h5 : fp.survivalRateB = 0.9)
    (h6 : fp.maxReplacement = 100) :
    ∃ (optimalA : ℕ), 
      optimalA = 200 ∧ 
      replacementPots fp optimalA ≤ fp.maxReplacement ∧
      ∀ (potsA : ℕ), replacementPots fp potsA ≤ fp.maxReplacement → 
        totalCost fp optimalA ≤ totalCost fp potsA ∧
      totalCost fp optimalA = 32000 := by
  sorry

end NUMINAMATH_CALUDE_optimal_planting_strategy_l750_75043


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l750_75025

/-- Given an isosceles triangle with equal sides of length x and base of length y,
    if a median to one of the equal sides divides the perimeter into parts of 15 cm and 6 cm,
    then the length of the base (y) is 1 cm. -/
theorem isosceles_triangle_base_length
  (x y : ℝ)
  (isosceles : x > 0)
  (perimeter_division : x + x/2 = 15 ∧ y + x/2 = 6 ∨ x + x/2 = 6 ∧ y + x/2 = 15)
  (triangle_inequality : x + x > y ∧ x + y > x ∧ x + y > x) :
  y = 1 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l750_75025


namespace NUMINAMATH_CALUDE_polygon_sides_l750_75028

theorem polygon_sides (n : ℕ) : n = 8 ↔ 
  (n - 2) * 180 = 3 * 360 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_l750_75028


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_falling_object_time_l750_75069

-- Part 1: Solving (x-1)^2 = 49
theorem solve_quadratic_equation :
  ∀ x : ℝ, (x - 1)^2 = 49 ↔ x = 8 ∨ x = -6 :=
by sorry

-- Part 2: Finding the time for an object to reach the ground
theorem falling_object_time (h t : ℝ) :
  h = 4.9 * t^2 →
  h = 10 →
  t = 10 / 7 :=
by sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_falling_object_time_l750_75069


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_ratio_l750_75000

theorem binomial_expansion_coefficient_ratio (n : ℕ) : 
  4 * (n.choose 2) = 7 * (2 * n) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_ratio_l750_75000


namespace NUMINAMATH_CALUDE_mean_of_fractions_l750_75065

theorem mean_of_fractions (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 1/8) :
  (a + b + c) / 3 = 7/24 := by
sorry

end NUMINAMATH_CALUDE_mean_of_fractions_l750_75065


namespace NUMINAMATH_CALUDE_last_locker_opened_l750_75018

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the direction the student is moving -/
inductive Direction
| Forward
| Backward

/-- Represents the student's action on a locker -/
def StudentAction := Nat → LockerState → Direction → (LockerState × Direction)

/-- The number of lockers in the corridor -/
def numLockers : Nat := 500

/-- The locker opening process -/
def openLockers (action : StudentAction) (n : Nat) : Nat :=
  sorry -- Implementation of the locker opening process

theorem last_locker_opened (action : StudentAction) :
  openLockers action numLockers = 242 := by
  sorry

#check last_locker_opened

end NUMINAMATH_CALUDE_last_locker_opened_l750_75018
