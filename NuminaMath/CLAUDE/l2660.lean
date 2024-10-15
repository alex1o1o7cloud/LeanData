import Mathlib

namespace NUMINAMATH_CALUDE_ratio_expression_value_l2660_266094

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l2660_266094


namespace NUMINAMATH_CALUDE_table_color_change_l2660_266083

/-- Represents the color of a cell in the table -/
inductive CellColor
| White
| Black
| Orange

/-- Represents a 3n × 3n table with the given coloring pattern -/
def Table (n : ℕ) := Fin (3*n) → Fin (3*n) → CellColor

/-- Predicate to check if a given 2×2 square can be chosen for color change -/
def CanChangeSquare (t : Table n) (i j : Fin (3*n-1)) : Prop := True

/-- Predicate to check if the table has all white cells turned to black and all black cells turned to white -/
def IsTargetState (t : Table n) : Prop := True

/-- Predicate to check if it's possible to reach the target state in a finite number of steps -/
def CanReachTargetState (n : ℕ) : Prop := 
  ∃ (t : Table n), IsTargetState t

theorem table_color_change (n : ℕ) : 
  CanReachTargetState n ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_table_color_change_l2660_266083


namespace NUMINAMATH_CALUDE_business_value_l2660_266037

theorem business_value (man_share : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  man_share = 1/3 →
  sold_fraction = 3/5 →
  sale_price = 15000 →
  (sale_price : ℚ) / sold_fraction / man_share = 75000 :=
by sorry

end NUMINAMATH_CALUDE_business_value_l2660_266037


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2660_266018

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_inequality
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d ≠ 0)
  (h3 : ∀ n : ℕ, a n > 0) :
  a 1 * a 8 < a 4 * a 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2660_266018


namespace NUMINAMATH_CALUDE_number_theory_problem_no_solution_2014_l2660_266008

theorem number_theory_problem (a x y : ℕ+) (h : x ≠ y) :
  a * x + Nat.gcd a x + Nat.lcm a x ≠ a * y + Nat.gcd a y + Nat.lcm a y :=
by sorry

theorem no_solution_2014 :
  ¬∃ (a b : ℕ+), a * b + Nat.gcd a b + Nat.lcm a b = 2014 :=
by sorry

end NUMINAMATH_CALUDE_number_theory_problem_no_solution_2014_l2660_266008


namespace NUMINAMATH_CALUDE_meaningful_expression_l2660_266084

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 2)) ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2660_266084


namespace NUMINAMATH_CALUDE_encyclopedia_pages_l2660_266090

/-- The number of chapters in the encyclopedia -/
def num_chapters : ℕ := 7

/-- The number of pages in each chapter of the encyclopedia -/
def pages_per_chapter : ℕ := 566

/-- The total number of pages in the encyclopedia -/
def total_pages : ℕ := num_chapters * pages_per_chapter

/-- Theorem stating that the total number of pages in the encyclopedia is 3962 -/
theorem encyclopedia_pages : total_pages = 3962 := by
  sorry

end NUMINAMATH_CALUDE_encyclopedia_pages_l2660_266090


namespace NUMINAMATH_CALUDE_water_experiment_result_l2660_266086

/-- Calculates the remaining water after an experiment and addition. -/
def remaining_water (initial : ℚ) (used : ℚ) (added : ℚ) : ℚ :=
  initial - used + added

/-- Proves that given the specific amounts in the problem, the remaining water is 13/6 gallons. -/
theorem water_experiment_result :
  remaining_water 3 (4/3) (1/2) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_water_experiment_result_l2660_266086


namespace NUMINAMATH_CALUDE_max_n_is_largest_l2660_266002

/-- Represents the sum of digits of a natural number -/
def S (a : ℕ) : ℕ := sorry

/-- Checks if all digits of a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The maximum natural number satisfying the given conditions -/
def max_n : ℕ := 3210

theorem max_n_is_largest :
  ∀ n : ℕ, 
  has_distinct_digits n → 
  S (3 * n) = 3 * S n → 
  n ≤ max_n :=
sorry

end NUMINAMATH_CALUDE_max_n_is_largest_l2660_266002


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_p_l2660_266098

/-- Given a quadratic equation x^2 - px + 2q = 0 where p and q are its roots and both non-zero,
    the sum of the roots is equal to p. -/
theorem sum_of_roots_equals_p (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
    (h : ∀ x, x^2 - p*x + 2*q = 0 ↔ x = p ∨ x = q) : 
  p + q = p := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_p_l2660_266098


namespace NUMINAMATH_CALUDE_direct_proportional_function_points_l2660_266042

/-- A direct proportional function passing through (2, -3) also passes through (4, -6) -/
theorem direct_proportional_function_points : ∃ (k : ℝ), k * 2 = -3 ∧ k * 4 = -6 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportional_function_points_l2660_266042


namespace NUMINAMATH_CALUDE_oranges_returned_l2660_266068

def oranges_problem (initial_oranges : ℕ) (eaten_oranges : ℕ) (final_oranges : ℕ) : ℕ :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen_oranges := remaining_after_eating / 2
  let remaining_after_theft := remaining_after_eating - stolen_oranges
  final_oranges - remaining_after_theft

theorem oranges_returned (initial_oranges eaten_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 60)
  (h2 : eaten_oranges = 10)
  (h3 : final_oranges = 30) : 
  oranges_problem initial_oranges eaten_oranges final_oranges = 5 := by
  sorry

#eval oranges_problem 60 10 30

end NUMINAMATH_CALUDE_oranges_returned_l2660_266068


namespace NUMINAMATH_CALUDE_interest_rate_20_percent_l2660_266003

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

-- State the theorem
theorem interest_rate_20_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : compound_interest P r 3 = 3000) 
  (h2 : compound_interest P r 4 = 3600) :
  r = 0.2 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_20_percent_l2660_266003


namespace NUMINAMATH_CALUDE_train_crossing_time_l2660_266073

/-- The time taken for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 210 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 28 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2660_266073


namespace NUMINAMATH_CALUDE_sticker_count_l2660_266000

/-- Given a number of stickers per page and a number of pages, 
    calculate the total number of stickers -/
def total_stickers (stickers_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  stickers_per_page * num_pages

/-- Theorem: The total number of stickers is 220 when there are 10 stickers per page and 22 pages -/
theorem sticker_count : total_stickers 10 22 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l2660_266000


namespace NUMINAMATH_CALUDE_tower_arrangements_l2660_266058

def num_red_cubes : ℕ := 2
def num_blue_cubes : ℕ := 4
def num_green_cubes : ℕ := 3
def tower_height : ℕ := 8

def remaining_cubes : ℕ := tower_height - 1
def remaining_blue_cubes : ℕ := num_blue_cubes - 1
def remaining_red_cubes : ℕ := num_red_cubes
def remaining_green_cubes : ℕ := num_green_cubes - 1

theorem tower_arrangements :
  (remaining_cubes.factorial) / (remaining_blue_cubes.factorial * remaining_red_cubes.factorial * remaining_green_cubes.factorial) = 210 := by
  sorry

end NUMINAMATH_CALUDE_tower_arrangements_l2660_266058


namespace NUMINAMATH_CALUDE_sum_of_digits_7_pow_1050_l2660_266030

theorem sum_of_digits_7_pow_1050 : ∃ (a b : ℕ), 
  7^1050 % 100 = 10 * a + b ∧ a + b = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_pow_1050_l2660_266030


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2660_266019

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define when two lines are parallel
def parallel (a : ℝ) : Prop := ∀ x y z w : ℝ, l₁ a x y ∧ l₂ a z w → (a = 2 * (a + 1))

-- Statement to prove
theorem sufficient_not_necessary (a : ℝ) :
  (a = 1 → parallel a) ∧ ¬(parallel a → a = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2660_266019


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l2660_266052

theorem simplify_radical_sum : Real.sqrt 50 + Real.sqrt 18 = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l2660_266052


namespace NUMINAMATH_CALUDE_overlap_range_l2660_266067

theorem overlap_range (total : ℕ) (math : ℕ) (chem : ℕ) (x : ℕ) 
  (h_total : total = 45)
  (h_math : math = 28)
  (h_chem : chem = 21)
  (h_overlap : x ≤ math ∧ x ≤ chem)
  (h_inclusion : math + chem - x ≤ total) :
  4 ≤ x ∧ x ≤ 21 := by
sorry

end NUMINAMATH_CALUDE_overlap_range_l2660_266067


namespace NUMINAMATH_CALUDE_range_of_a_l2660_266056

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x ≤ a^2 - a - 3

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*a)^x > (5 - 2*a)^y

theorem range_of_a : 
  (∀ a : ℝ, (p a ∨ q a)) ∧ (¬∃ a : ℝ, p a ∧ q a) → 
  {a : ℝ | a = 2 ∨ a ≥ 5/2} = {a : ℝ | ∃ x : ℝ, p x ∨ q x} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2660_266056


namespace NUMINAMATH_CALUDE_problem_statement_l2660_266031

theorem problem_statement (A B C : ℚ) 
  (h1 : 1 / A = -3)
  (h2 : 2 / B = 4)
  (h3 : 3 / C = 1 / 2) :
  6 * A - 8 * B + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2660_266031


namespace NUMINAMATH_CALUDE_regular_pyramid_volume_l2660_266074

-- Define the properties of the pyramid
structure RegularPyramid where
  l : ℝ  -- lateral edge length
  interior_angle_sum : ℝ  -- sum of interior angles of the base polygon
  lateral_angle : ℝ  -- angle between lateral edge and height

-- Define the theorem
theorem regular_pyramid_volume 
  (p : RegularPyramid) 
  (h1 : p.interior_angle_sum = 720)
  (h2 : p.lateral_angle = 30) : 
  ∃ (v : ℝ), v = (3 * p.l ^ 3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_regular_pyramid_volume_l2660_266074


namespace NUMINAMATH_CALUDE_intersection_condition_l2660_266023

-- Define the line and parabola
def line (k x : ℝ) : ℝ := k * x - 2 * k + 2
def parabola (a x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3 * a

-- Define the condition for intersection
def hasCommonPoint (a : ℝ) : Prop :=
  ∀ k, ∃ x, line k x = parabola a x

-- State the theorem
theorem intersection_condition :
  ∀ a : ℝ, hasCommonPoint a ↔ (a ≤ -2/3 ∨ a > 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l2660_266023


namespace NUMINAMATH_CALUDE_function_inequality_l2660_266048

theorem function_inequality (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂) →
  (∀ x, f x = x + 4/x) →
  (∀ x, g x = 2^x + a) →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l2660_266048


namespace NUMINAMATH_CALUDE_speaker_is_tweedledee_l2660_266085

-- Define the brothers
inductive Brother
| Tweedledum
| Tweedledee

-- Define the card suits
inductive Suit
| Black
| Red

-- Define the speaker
structure Speaker where
  identity : Brother
  card : Suit

-- Define the statement made by the speaker
def statement (s : Speaker) : Prop :=
  s.identity = Brother.Tweedledum → s.card ≠ Suit.Black

-- Theorem: The speaker must be Tweedledee
theorem speaker_is_tweedledee (s : Speaker) (h : statement s) : 
  s.identity = Brother.Tweedledee :=
sorry

end NUMINAMATH_CALUDE_speaker_is_tweedledee_l2660_266085


namespace NUMINAMATH_CALUDE_melanie_brownies_given_out_l2660_266079

def total_brownies : ℕ := 12 * 25

def bake_sale_brownies : ℕ := (7 * total_brownies) / 10

def remaining_after_bake_sale : ℕ := total_brownies - bake_sale_brownies

def container_brownies : ℕ := (2 * remaining_after_bake_sale) / 3

def remaining_after_container : ℕ := remaining_after_bake_sale - container_brownies

def charity_brownies : ℕ := (2 * remaining_after_container) / 5

def brownies_given_out : ℕ := remaining_after_container - charity_brownies

theorem melanie_brownies_given_out : brownies_given_out = 18 := by
  sorry

end NUMINAMATH_CALUDE_melanie_brownies_given_out_l2660_266079


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2660_266061

theorem simplify_trig_expression (x : ℝ) (h : 5 * π / 4 < x ∧ x < 3 * π / 2) :
  Real.sqrt (1 - 2 * Real.sin x * Real.cos x) = Real.cos x - Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2660_266061


namespace NUMINAMATH_CALUDE_find_y_l2660_266077

theorem find_y : ∃ y : ℝ, (Real.sqrt (1 + Real.sqrt (4 * y - 5)) = Real.sqrt 8) ∧ y = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2660_266077


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l2660_266049

/-- Represents the pattern of alternating semicircles -/
structure SemicirclePattern where
  diameter : ℝ
  patternLength : ℝ

/-- Calculates the total shaded area of the semicircle pattern -/
def totalShadedArea (pattern : SemicirclePattern) : ℝ :=
  sorry

/-- Theorem stating that the total shaded area for the given pattern is 6.75π -/
theorem semicircle_pattern_area 
  (pattern : SemicirclePattern) 
  (h1 : pattern.diameter = 3)
  (h2 : pattern.patternLength = 10) : 
  totalShadedArea pattern = 6.75 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l2660_266049


namespace NUMINAMATH_CALUDE_solution_range_l2660_266009

-- Define the operation @
def op (p q : ℝ) : ℝ := p + q - p * q

-- Define the main theorem
theorem solution_range (m : ℝ) :
  (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
    op 2 (x₁ : ℝ) > 0 ∧ op (x₁ : ℝ) 3 ≤ m ∧
    op 2 (x₂ : ℝ) > 0 ∧ op (x₂ : ℝ) 3 ≤ m ∧
    (∀ (x : ℤ), x ≠ x₁ ∧ x ≠ x₂ → op 2 (x : ℝ) ≤ 0 ∨ op (x : ℝ) 3 > m)) →
  3 ≤ m ∧ m < 5 :=
sorry

end NUMINAMATH_CALUDE_solution_range_l2660_266009


namespace NUMINAMATH_CALUDE_choir_members_count_l2660_266075

theorem choir_members_count : ∃! n : ℕ, 
  200 < n ∧ n < 300 ∧ 
  (∃ k : ℕ, n + 4 = 10 * k) ∧
  (∃ m : ℕ, n + 5 = 11 * m) := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l2660_266075


namespace NUMINAMATH_CALUDE_product_of_exponents_l2660_266022

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^3 = 272 → 
  3^r + 54 = 135 → 
  7^2 + 6^s = 527 → 
  p * r * s = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l2660_266022


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l2660_266026

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l2660_266026


namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_equation_l2660_266069

/-- Represents the average monthly growth rate of profit from January to March -/
def monthly_growth_rate : ℝ → Prop :=
  fun x => 3 * (1 + x)^2 = 3.63

/-- The profit in January -/
def january_profit : ℝ := 30000

/-- The profit in March -/
def march_profit : ℝ := 36300

/-- Theorem stating the equation for the average monthly growth rate -/
theorem average_monthly_growth_rate_equation :
  ∃ x : ℝ, monthly_growth_rate x ∧
    march_profit = january_profit * (1 + x)^2 :=
  sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_equation_l2660_266069


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l2660_266011

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 80 = 1 ∧ 
    (13 * b) % 80 = 1 ∧ 
    ((3 * a + 9 * b) % 80) = 2 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l2660_266011


namespace NUMINAMATH_CALUDE_equality_of_ratios_implies_equality_of_squares_l2660_266076

theorem equality_of_ratios_implies_equality_of_squares
  (x y z : ℝ) (h : x / y = 3 / z) :
  9 * y^2 = x^2 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_equality_of_ratios_implies_equality_of_squares_l2660_266076


namespace NUMINAMATH_CALUDE_binomial_product_theorem_l2660_266028

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the smallest prime number greater than 10
def smallest_prime_gt_10 : ℕ := 11

-- Theorem statement
theorem binomial_product_theorem :
  binomial 18 6 * smallest_prime_gt_10 = 80080 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_theorem_l2660_266028


namespace NUMINAMATH_CALUDE_tan_two_theta_value_l2660_266021

theorem tan_two_theta_value (θ : Real) 
  (h : 2 * Real.sin (π / 2 + θ) + Real.sin (π + θ) = 0) : 
  Real.tan (2 * θ) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_theta_value_l2660_266021


namespace NUMINAMATH_CALUDE_unique_number_satisfying_equation_l2660_266051

theorem unique_number_satisfying_equation : ∃! x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_equation_l2660_266051


namespace NUMINAMATH_CALUDE_eccentricity_classification_l2660_266082

theorem eccentricity_classification (x₁ x₂ : ℝ) : 
  2 * x₁^2 - 5 * x₁ + 2 = 0 →
  2 * x₂^2 - 5 * x₂ + 2 = 0 →
  x₁ ≠ x₂ →
  ((0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂) ∨ (0 < x₂ ∧ x₂ < 1 ∧ 1 < x₁)) :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_classification_l2660_266082


namespace NUMINAMATH_CALUDE_point_on_line_l2660_266081

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A point lies on a line if and only if it can be expressed as a linear combination of two points on that line. -/
theorem point_on_line (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔
  ∃ s : ℝ, X - A = s • (B - A) :=
sorry

end NUMINAMATH_CALUDE_point_on_line_l2660_266081


namespace NUMINAMATH_CALUDE_max_n_for_coprime_with_prime_l2660_266013

/-- A function that checks if a list of integers is pairwise coprime -/
def IsPairwiseCoprime (list : List Int) : Prop :=
  ∀ i j, i ≠ j → i ∈ list → j ∈ list → Int.gcd i j = 1

/-- A function that checks if a number is prime -/
def IsPrime (n : Int) : Prop :=
  n > 1 ∧ ∀ m, 1 < m → m < n → ¬(n % m = 0)

/-- The main theorem -/
theorem max_n_for_coprime_with_prime : 
  (∀ (list : List Int), list.length = 5 → 
    (∀ x ∈ list, x ≥ 1 ∧ x ≤ 48) → 
    IsPairwiseCoprime list → 
    (∃ x ∈ list, IsPrime x)) ∧ 
  (∃ (list : List Int), list.length = 5 ∧ 
    (∀ x ∈ list, x ≥ 1 ∧ x ≤ 49) ∧ 
    IsPairwiseCoprime list ∧ 
    (∀ x ∈ list, ¬IsPrime x)) := by
  sorry

end NUMINAMATH_CALUDE_max_n_for_coprime_with_prime_l2660_266013


namespace NUMINAMATH_CALUDE_probability_of_not_losing_l2660_266016

theorem probability_of_not_losing (prob_draw prob_win : ℚ) 
  (h1 : prob_draw = 1/2) 
  (h2 : prob_win = 1/3) : 
  prob_draw + prob_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_not_losing_l2660_266016


namespace NUMINAMATH_CALUDE_jacksons_email_deletion_l2660_266006

theorem jacksons_email_deletion (initial_deletion : ℕ) (first_received : ℕ) 
  (second_received : ℕ) (final_received : ℕ) (final_inbox : ℕ) :
  initial_deletion = 50 →
  first_received = 15 →
  second_received = 5 →
  final_received = 10 →
  final_inbox = 30 →
  ∃ (second_deletion : ℕ), 
    second_deletion = 50 ∧
    final_inbox = first_received + second_received + final_received - initial_deletion - second_deletion :=
by sorry

end NUMINAMATH_CALUDE_jacksons_email_deletion_l2660_266006


namespace NUMINAMATH_CALUDE_carolyns_silverware_percentage_l2660_266014

/-- Represents the count of each type of silverware --/
structure SilverwareCount where
  knives : Int
  forks : Int
  spoons : Int
  teaspoons : Int

/-- Calculates the total count of silverware --/
def total_count (s : SilverwareCount) : Int :=
  s.knives + s.forks + s.spoons + s.teaspoons

/-- Represents a trade of silverware --/
structure Trade where
  give_knives : Int
  give_forks : Int
  give_spoons : Int
  give_teaspoons : Int
  receive_knives : Int
  receive_forks : Int
  receive_spoons : Int
  receive_teaspoons : Int

/-- Applies a trade to a silverware count --/
def apply_trade (s : SilverwareCount) (t : Trade) : SilverwareCount :=
  { knives := s.knives - t.give_knives + t.receive_knives,
    forks := s.forks - t.give_forks + t.receive_forks,
    spoons := s.spoons - t.give_spoons + t.receive_spoons,
    teaspoons := s.teaspoons - t.give_teaspoons + t.receive_teaspoons }

/-- Theorem representing Carolyn's silverware problem --/
theorem carolyns_silverware_percentage :
  let initial_count : SilverwareCount := { knives := 6, forks := 12, spoons := 18, teaspoons := 24 }
  let trade1 : Trade := { give_knives := 10, give_forks := 0, give_spoons := 0, give_teaspoons := 0,
                          receive_knives := 0, receive_forks := 0, receive_spoons := 0, receive_teaspoons := 6 }
  let trade2 : Trade := { give_knives := 0, give_forks := 8, give_spoons := 0, give_teaspoons := 0,
                          receive_knives := 0, receive_forks := 0, receive_spoons := 3, receive_teaspoons := 0 }
  let after_trades := apply_trade (apply_trade initial_count trade1) trade2
  let final_count := { after_trades with knives := after_trades.knives + 7 }
  (final_count.knives : Real) / (total_count final_count : Real) * 100 = 3 / 58 * 100 :=
by sorry

end NUMINAMATH_CALUDE_carolyns_silverware_percentage_l2660_266014


namespace NUMINAMATH_CALUDE_speed_conversion_l2660_266096

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def speed_mps : ℝ := 12.7788

/-- Theorem stating the conversion of the given speed from m/s to km/h -/
theorem speed_conversion :
  speed_mps * mps_to_kmph = 45.96368 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l2660_266096


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l2660_266097

theorem midpoint_distance_theorem (t : ℝ) : 
  let P : ℝ × ℝ := (2 * t - 3, 2)
  let Q : ℝ × ℝ := (-2, 2 * t + 1)
  let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2 = t ^ 2 + 1 →
  t = 1 + Real.sqrt (3 / 2) ∨ t = 1 - Real.sqrt (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l2660_266097


namespace NUMINAMATH_CALUDE_room_height_proof_l2660_266024

theorem room_height_proof (l b h : ℝ) : 
  l = 12 → b = 8 → (l^2 + b^2 + h^2 = 17^2) → h = 9 := by sorry

end NUMINAMATH_CALUDE_room_height_proof_l2660_266024


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2660_266007

open Real

theorem sin_30_degrees :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2660_266007


namespace NUMINAMATH_CALUDE_sort_table_in_99_moves_l2660_266050

/-- Represents a 10x10 table of distinct integers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j i' j', t i j = t i' j' → i = i' ∧ j = j'

/-- Predicate to check if the table is sorted in ascending order -/
def is_sorted (t : Table) : Prop :=
  (∀ i j j', j < j' → t i j < t i j') ∧
  (∀ i i' j, i < i' → t i j < t i' j)

/-- Represents a rectangular subset of the table -/
structure Rectangle where
  top_left : Fin 10 × Fin 10
  bottom_right : Fin 10 × Fin 10

/-- Represents a move (180° rotation of a rectangular subset) -/
def Move := Rectangle

/-- Applies a move to the table -/
def apply_move (t : Table) (m : Move) : Table :=
  sorry

/-- Theorem: It's always possible to sort the table in 99 or fewer moves -/
theorem sort_table_in_99_moves (t : Table) (h : all_distinct t) :
  ∃ (moves : List Move), moves.length ≤ 99 ∧ is_sorted (moves.foldl apply_move t) :=
  sorry

end NUMINAMATH_CALUDE_sort_table_in_99_moves_l2660_266050


namespace NUMINAMATH_CALUDE_reflection_theorem_l2660_266010

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point with respect to another point -/
def reflect (p : Point3D) (center : Point3D) : Point3D :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y,
    z := 2 * center.z - p.z }

/-- Perform a sequence of reflections -/
def reflectSequence (p : Point3D) (centers : List Point3D) : Point3D :=
  centers.foldl reflect p

theorem reflection_theorem (A O₁ O₂ O₃ : Point3D) :
  reflectSequence (reflectSequence A [O₁, O₂, O₃]) [O₁, O₂, O₃] = A := by
  sorry


end NUMINAMATH_CALUDE_reflection_theorem_l2660_266010


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l2660_266060

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  (2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) →
  (∃ m : ℝ, m = 54 ∧ ∀ x : ℝ, 2 * a 8 + a 7 ≥ m) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l2660_266060


namespace NUMINAMATH_CALUDE_proposition_false_iff_m_in_range_l2660_266099

/-- The proposition is false for all real x when m is in [2,6) -/
theorem proposition_false_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x^2 + (m - 2) * x + 1 > 0) ↔ (2 ≤ m ∧ m < 6) :=
by sorry

end NUMINAMATH_CALUDE_proposition_false_iff_m_in_range_l2660_266099


namespace NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2660_266059

theorem probability_of_one_out_of_four (S : Finset α) (h : S.card = 4) :
  ∀ a ∈ S, (1 : ℝ) / S.card = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2660_266059


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2660_266034

/-- The eccentricity of an ellipse with a focus shared with the parabola y^2 = x -/
theorem ellipse_eccentricity (a : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = x}
  let ellipse := {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / 3) = 1}
  let parabola_focus : ℝ × ℝ := (1/4, 0)
  (parabola_focus ∈ ellipse) →
  (∃ c b : ℝ, c^2 + b^2 = a^2 ∧ c = 1/4 ∧ b^2 = 3) →
  (c / a = 1/7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2660_266034


namespace NUMINAMATH_CALUDE_m_divided_by_8_l2660_266089

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1011) : m / 8 = 2^4041 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l2660_266089


namespace NUMINAMATH_CALUDE_smallest_n_for_cube_assembly_l2660_266095

theorem smallest_n_for_cube_assembly (n : ℕ) : 
  (∀ m : ℕ, m < n → m^3 < (2*m)^3 - (2*m - 2)^3) ∧ 
  n^3 ≥ (2*n)^3 - (2*n - 2)^3 → 
  n = 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_cube_assembly_l2660_266095


namespace NUMINAMATH_CALUDE_sum_to_k_perfect_cube_l2660_266054

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem sum_to_k_perfect_cube :
  ∀ k : ℕ, k > 0 → k < 200 →
    (is_perfect_cube (sum_to_k k) ↔ k = 1 ∨ k = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_perfect_cube_l2660_266054


namespace NUMINAMATH_CALUDE_fraction_arithmetic_l2660_266039

theorem fraction_arithmetic : (2 : ℚ) / 9 * 4 / 5 - 1 / 45 = 7 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_arithmetic_l2660_266039


namespace NUMINAMATH_CALUDE_point_relationship_on_line_l2660_266087

/-- Proves that for two points on a line with positive slope and non-negative y-intercept,
    if the x-coordinate of the first point is greater than the x-coordinate of the second point,
    then the y-coordinate of the first point is greater than the y-coordinate of the second point. -/
theorem point_relationship_on_line (k b x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : y₁ = k * x₁ + b)
  (h2 : y₂ = k * x₂ + b)
  (h3 : k > 0)
  (h4 : b ≥ 0)
  (h5 : x₁ > x₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_point_relationship_on_line_l2660_266087


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l2660_266064

theorem least_number_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 34 = 4 ∧
  n % 48 = 6 ∧
  n % 5 = 2 ∧
  ∀ m : ℕ, m > 0 ∧ m % 34 = 4 ∧ m % 48 = 6 ∧ m % 5 = 2 → n ≤ m :=
by
  use 4082
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l2660_266064


namespace NUMINAMATH_CALUDE_student_sister_weight_l2660_266062

/-- The combined weight of a student and his sister -/
theorem student_sister_weight (student_weight sister_weight : ℝ) : 
  student_weight = 71 →
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 104 := by
sorry

end NUMINAMATH_CALUDE_student_sister_weight_l2660_266062


namespace NUMINAMATH_CALUDE_smallest_inverse_undefined_twenty_two_satisfies_smallest_inverse_undefined_is_22_l2660_266001

theorem smallest_inverse_undefined (a : ℕ) : a > 0 ∧ 
  (∀ x : ℕ, x * a % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * a % 77 ≠ 1) → 
  a ≥ 22 := by
sorry

theorem twenty_two_satisfies : 
  (∀ x : ℕ, x * 22 % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * 22 % 77 ≠ 1) := by
sorry

theorem smallest_inverse_undefined_is_22 : 
  ∃! a : ℕ, a > 0 ∧ 
  (∀ x : ℕ, x * a % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * a % 77 ≠ 1) ∧ 
  ∀ b : ℕ, b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 77 ≠ 1) → 
  a ≤ b := by
sorry

end NUMINAMATH_CALUDE_smallest_inverse_undefined_twenty_two_satisfies_smallest_inverse_undefined_is_22_l2660_266001


namespace NUMINAMATH_CALUDE_dave_trays_first_table_l2660_266017

/-- The number of trays Dave can carry per trip -/
def trays_per_trip : ℕ := 9

/-- The number of trips Dave made -/
def total_trips : ℕ := 8

/-- The number of trays Dave picked up from the second table -/
def trays_from_second_table : ℕ := 55

/-- The number of trays Dave picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * total_trips - trays_from_second_table

theorem dave_trays_first_table : trays_from_first_table = 17 := by
  sorry

end NUMINAMATH_CALUDE_dave_trays_first_table_l2660_266017


namespace NUMINAMATH_CALUDE_solution_check_l2660_266092

def is_solution (x : ℝ) : Prop :=
  4 * x + 5 = 8 * x - 3

theorem solution_check :
  is_solution 2 ∧ ¬is_solution 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_check_l2660_266092


namespace NUMINAMATH_CALUDE_pizza_problem_l2660_266015

/-- Calculates the total number of pizza pieces carried by children -/
def total_pizza_pieces (num_children : ℕ) (pizzas_per_child : ℕ) (pieces_per_pizza : ℕ) : ℕ :=
  num_children * pizzas_per_child * pieces_per_pizza

/-- Proves that 10 children buying 20 pizzas each, with 6 pieces per pizza, carry 1200 pieces total -/
theorem pizza_problem : total_pizza_pieces 10 20 6 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l2660_266015


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l2660_266071

theorem max_triangle_side_length (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different side lengths
  a + b + c = 30 →         -- Perimeter is 30
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a + b > c ∧ b + c > a ∧ a + c > b →  -- Triangle inequality
  a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 :=
by sorry

#check max_triangle_side_length

end NUMINAMATH_CALUDE_max_triangle_side_length_l2660_266071


namespace NUMINAMATH_CALUDE_square_division_correct_l2660_266088

/-- Represents the state of the square division process after n iterations -/
structure SquareDivision (n : ℕ) where
  /-- The number of remaining squares -/
  remaining_squares : ℕ
  /-- The side length of each remaining square -/
  side_length : ℚ
  /-- The total area of removed squares -/
  removed_area : ℚ

/-- The result of the square division process after n iterations -/
def square_division_result (n : ℕ) : SquareDivision n :=
  { remaining_squares := 8^n,
    side_length := 1 / 3^n,
    removed_area := 1 - (8/9)^n }

/-- Theorem stating the correctness of the square division result -/
theorem square_division_correct (n : ℕ) :
  (square_division_result n).remaining_squares = 8^n ∧
  (square_division_result n).side_length = 1 / 3^n ∧
  (square_division_result n).removed_area = 1 - (8/9)^n :=
by sorry

end NUMINAMATH_CALUDE_square_division_correct_l2660_266088


namespace NUMINAMATH_CALUDE_A_initial_investment_l2660_266091

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := 27000

/-- Represents the investment of B in rupees -/
def B_investment : ℝ := 36000

/-- Represents the number of months in a year -/
def months_in_year : ℝ := 12

/-- Represents the number of months after which B joined -/
def B_join_time : ℝ := 7.5

/-- Represents the ratio of profit sharing between A and B -/
def profit_ratio : ℝ := 2

theorem A_initial_investment :
  A_investment * months_in_year = 
  profit_ratio * B_investment * (months_in_year - B_join_time) :=
by sorry

end NUMINAMATH_CALUDE_A_initial_investment_l2660_266091


namespace NUMINAMATH_CALUDE_prob_at_least_one_women_pair_l2660_266066

/-- The number of young men in the group -/
def num_men : ℕ := 6

/-- The number of young women in the group -/
def num_women : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair up the group -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair up without any all-women pairs -/
def pairings_without_women_pairs : ℕ := num_women.factorial

/-- The probability of at least one pair consisting of two women -/
theorem prob_at_least_one_women_pair :
  (total_pairings - pairings_without_women_pairs) / total_pairings = 9675 / 10395 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_women_pair_l2660_266066


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l2660_266012

/-- The count of three-digit numbers without exactly two identical adjacent digits -/
def validThreeDigitNumbers : ℕ :=
  let totalThreeDigitNumbers := 900
  let excludedNumbers := 162
  totalThreeDigitNumbers - excludedNumbers

/-- Theorem stating that the count of valid three-digit numbers is 738 -/
theorem valid_three_digit_numbers_count :
  validThreeDigitNumbers = 738 := by
  sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l2660_266012


namespace NUMINAMATH_CALUDE_expression_simplification_l2660_266047

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 - y^2) / (x*y) - (x^2*y - y^3) / (x^2*y - x*y^2) = (x^2 - x*y - 2*y^2) / (x*y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2660_266047


namespace NUMINAMATH_CALUDE_door_rod_equation_l2660_266072

theorem door_rod_equation (x : ℝ) : (x - 4)^2 + (x - 2)^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_door_rod_equation_l2660_266072


namespace NUMINAMATH_CALUDE_maries_speed_l2660_266004

/-- Given that Marie can bike 372 miles in 31 hours, prove that her speed is 12 miles per hour. -/
theorem maries_speed (distance : ℝ) (time : ℝ) (h1 : distance = 372) (h2 : time = 31) :
  distance / time = 12 := by
  sorry

end NUMINAMATH_CALUDE_maries_speed_l2660_266004


namespace NUMINAMATH_CALUDE_negation_equivalence_l2660_266078

theorem negation_equivalence :
  (¬ ∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2660_266078


namespace NUMINAMATH_CALUDE_alyssa_grapes_cost_l2660_266029

/-- The amount Alyssa paid for grapes -/
def grapesCost (totalSpent refund : ℚ) : ℚ := totalSpent + refund

/-- Proof that Alyssa paid $12.08 for grapes -/
theorem alyssa_grapes_cost : 
  let totalSpent : ℚ := 223/100
  let cherryRefund : ℚ := 985/100
  grapesCost totalSpent cherryRefund = 1208/100 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_grapes_cost_l2660_266029


namespace NUMINAMATH_CALUDE_train_passes_jogger_l2660_266044

/-- Proves that a train passes a jogger in 40 seconds given specific conditions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 280 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 40 := by
  sorry

#check train_passes_jogger

end NUMINAMATH_CALUDE_train_passes_jogger_l2660_266044


namespace NUMINAMATH_CALUDE_binomial_8_4_l2660_266040

theorem binomial_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_4_l2660_266040


namespace NUMINAMATH_CALUDE_rectangle_area_l2660_266038

/-- The area of the rectangle formed by the intersections of x^4 + y^4 = 100 and xy = 4 -/
theorem rectangle_area : ∃ (a b : ℝ), 
  (a^4 + b^4 = 100) ∧ 
  (a * b = 4) ∧ 
  (2 * (a^2 - b^2) = 4 * Real.sqrt 17) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2660_266038


namespace NUMINAMATH_CALUDE_sqrt_equation_equals_difference_l2660_266093

theorem sqrt_equation_equals_difference (a b : ℤ) : 
  Real.sqrt (16 - 12 * Real.cos (40 * π / 180)) = a + b * (1 / Real.cos (40 * π / 180)) →
  a = 4 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_equals_difference_l2660_266093


namespace NUMINAMATH_CALUDE_guthrie_market_souvenirs_cost_l2660_266055

/-- The total cost of souvenirs distributed at Guthrie Market's Grand Opening -/
theorem guthrie_market_souvenirs_cost :
  let type1_cost : ℚ := 20 / 100  -- 20 cents in dollars
  let type2_cost : ℚ := 25 / 100  -- 25 cents in dollars
  let total_souvenirs : ℕ := 1000
  let type2_quantity : ℕ := 400
  let type1_quantity : ℕ := total_souvenirs - type2_quantity
  let total_cost : ℚ := type1_quantity * type1_cost + type2_quantity * type2_cost
  total_cost = 220 / 100  -- $220 in decimal form
:= by sorry

end NUMINAMATH_CALUDE_guthrie_market_souvenirs_cost_l2660_266055


namespace NUMINAMATH_CALUDE_divisor_exists_l2660_266057

def N : ℕ := 123456789101112131415161718192021222324252627282930313233343536373839404142434481

theorem divisor_exists : ∃ D : ℕ, D > 0 ∧ N % D = 36 := by
  sorry

end NUMINAMATH_CALUDE_divisor_exists_l2660_266057


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2660_266043

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) →
  ∃ (k : ℤ), b = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2660_266043


namespace NUMINAMATH_CALUDE_fence_sheets_count_l2660_266025

/-- Represents the number of fence panels in the fence. -/
def num_panels : ℕ := 10

/-- Represents the number of metal beams in each fence panel. -/
def beams_per_panel : ℕ := 2

/-- Represents the number of metal rods in each sheet. -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal rods in each beam. -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the fence. -/
def total_rods : ℕ := 380

/-- Calculates the number of metal sheets in each fence panel. -/
def sheets_per_panel : ℕ :=
  let total_rods_per_panel := total_rods / num_panels
  let rods_for_beams := beams_per_panel * rods_per_beam
  (total_rods_per_panel - rods_for_beams) / rods_per_sheet

theorem fence_sheets_count : sheets_per_panel = 3 := by
  sorry

end NUMINAMATH_CALUDE_fence_sheets_count_l2660_266025


namespace NUMINAMATH_CALUDE_largest_valid_number_l2660_266053

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n.digits 10).sum % 6 = 0

theorem largest_valid_number : 
  is_valid_number 936 ∧ ∀ m, is_valid_number m → m ≤ 936 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2660_266053


namespace NUMINAMATH_CALUDE_fraction_equality_implies_sum_l2660_266032

theorem fraction_equality_implies_sum (A B : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 23) / (x^2 - 9*x + 20) = A / (x - 4) + 5 / (x - 5)) →
  A + B = 11/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_sum_l2660_266032


namespace NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_five_l2660_266063

def is_abundant (n : ℕ) : Prop :=
  n > 0 ∧ (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id > n)

def is_multiple_of_five (n : ℕ) : Prop :=
  n % 5 = 0

theorem smallest_abundant_not_multiple_of_five : 
  (∀ k : ℕ, k < 12 → ¬(is_abundant k ∧ ¬is_multiple_of_five k)) ∧ 
  (is_abundant 12 ∧ ¬is_multiple_of_five 12) := by
  sorry

end NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_five_l2660_266063


namespace NUMINAMATH_CALUDE_pizza_topping_cost_l2660_266041

/-- The cost per pizza in dollars -/
def cost_per_pizza : ℚ := 10

/-- The number of pizzas ordered -/
def num_pizzas : ℕ := 3

/-- The total number of toppings across all pizzas -/
def total_toppings : ℕ := 4

/-- The tip amount in dollars -/
def tip : ℚ := 5

/-- The total cost of the order including tip in dollars -/
def total_cost : ℚ := 39

/-- The cost per topping in dollars -/
def cost_per_topping : ℚ := 1

theorem pizza_topping_cost :
  cost_per_pizza * num_pizzas + cost_per_topping * total_toppings + tip = total_cost :=
sorry

end NUMINAMATH_CALUDE_pizza_topping_cost_l2660_266041


namespace NUMINAMATH_CALUDE_simplify_expression_l2660_266020

theorem simplify_expression (x : ℝ) : (x + 15) + (100 * x + 15) = 101 * x + 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2660_266020


namespace NUMINAMATH_CALUDE_fraction_calculation_l2660_266065

theorem fraction_calculation : (8 / 15 - 7 / 9) + 3 / 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2660_266065


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l2660_266036

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}

-- State the theorem
theorem intersection_implies_sum (m n : ℝ) : 
  M ∩ N m = {x | 3 < x ∧ x < n} → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l2660_266036


namespace NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l2660_266045

/-- Two lines in three-dimensional space -/
structure Line3D where
  -- We don't need to define the internal structure of a line
  -- for this problem, so we leave it abstract

/-- Predicate for two lines intersecting -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for two lines being parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for two lines being skew -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

theorem non_intersecting_lines_parallel_or_skew 
  (l1 l2 : Line3D) (h : ¬ intersect l1 l2) : 
  parallel l1 l2 ∨ skew l1 l2 :=
by
  sorry

end NUMINAMATH_CALUDE_non_intersecting_lines_parallel_or_skew_l2660_266045


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2660_266046

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2660_266046


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l2660_266027

theorem least_n_for_inequality : 
  (∀ n : ℕ, n > 0 → (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15 → n ≥ 4) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15) := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l2660_266027


namespace NUMINAMATH_CALUDE_complex_number_system_l2660_266080

theorem complex_number_system (a b c : ℂ) (h_real : a.im = 0) 
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 4)
  (h_prod : a * b * c = 4) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_system_l2660_266080


namespace NUMINAMATH_CALUDE_lisas_large_spoons_lisas_large_spoons_is_ten_l2660_266033

/-- Calculates the number of large spoons in Lisa's new cutlery set -/
theorem lisas_large_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) 
  (decorative_spoons : ℕ) (new_teaspoons : ℕ) (total_spoons : ℕ) : ℕ :=
  let kept_spoons := num_children * baby_spoons_per_child + decorative_spoons
  let known_spoons := kept_spoons + new_teaspoons
  total_spoons - known_spoons

/-- Proves that the number of large spoons in Lisa's new cutlery set is 10 -/
theorem lisas_large_spoons_is_ten :
  lisas_large_spoons 4 3 2 15 39 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lisas_large_spoons_lisas_large_spoons_is_ten_l2660_266033


namespace NUMINAMATH_CALUDE_m_upper_bound_l2660_266070

theorem m_upper_bound (f : ℝ → ℝ) (m : ℝ) :
  (∀ x > 0, f x = Real.exp x + Real.exp (-x)) →
  (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) →
  m ≤ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_m_upper_bound_l2660_266070


namespace NUMINAMATH_CALUDE_sequence_properties_l2660_266005

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1 / (2 * n - 1)

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → 1 / (2 * sequence_a (n + 1)) = 1 / (2 * sequence_a n) + 1) →
  (∀ n : ℕ, n > 0 → 1 / sequence_a (n + 1) - 1 / sequence_a n = 2) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 1 / (2 * n - 1)) ∧
  (∀ n : ℕ, n > 0 → 
    (Finset.range n).sum (λ i => sequence_a i * sequence_a (i + 1)) = n / (2 * n + 1)) ∧
  (∀ n : ℕ, n > 0 → 
    ((Finset.range n).sum (λ i => sequence_a i * sequence_a (i + 1)) > 16 / 33) ↔ n > 16) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2660_266005


namespace NUMINAMATH_CALUDE_average_height_problem_l2660_266035

/-- Given a class of girls with specific average heights, prove the average height of a subgroup -/
theorem average_height_problem (total_girls : ℕ) (subgroup_girls : ℕ) (remaining_girls : ℕ)
  (subgroup_avg_height : ℝ) (remaining_avg_height : ℝ) (total_avg_height : ℝ)
  (h1 : total_girls = subgroup_girls + remaining_girls)
  (h2 : total_girls = 40)
  (h3 : subgroup_girls = 30)
  (h4 : remaining_avg_height = 156)
  (h5 : total_avg_height = 159) :
  subgroup_avg_height = 160 := by
sorry


end NUMINAMATH_CALUDE_average_height_problem_l2660_266035
