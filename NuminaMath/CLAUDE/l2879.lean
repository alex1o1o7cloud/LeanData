import Mathlib

namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l2879_287944

/-- Represents a parabola in 2D space -/
structure Parabola where
  equation : ℝ → ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { equation := fun x => p.equation (x - h) + v }

/-- The original parabola y = 3x² -/
def original_parabola : Parabola :=
  { equation := fun x => 3 * x^2 }

/-- The shifted parabola -/
def shifted_parabola : Parabola :=
  shift_parabola original_parabola 1 2

theorem shifted_parabola_equation :
  shifted_parabola.equation = fun x => 3 * (x - 1)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l2879_287944


namespace NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_lower_bound_l2879_287974

/-- Represents a tiling of a rectangle -/
structure Tiling (m n : ℕ) :=
  (pieces : ℕ)
  (is_valid : Bool)

/-- The number of ways to tile a 5 × 2k rectangle with 2k pieces -/
def tiling_count (k : ℕ) : ℕ := sorry

theorem rectangle_tiling (n : ℕ) (t : Tiling 5 n) :
  t.pieces = n ∧ t.is_valid → Even n := by sorry

theorem tiling_count_lower_bound (k : ℕ) :
  k ≥ 3 → tiling_count k > 2 * 3^(k-1) := by sorry

end NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_lower_bound_l2879_287974


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_minus_one_problem_4_l2879_287957

theorem gcd_powers_of_two_minus_one (a b : Nat) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := by sorry

theorem problem_4 : Nat.gcd (2^6 - 1) (2^9 - 1) = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_minus_one_problem_4_l2879_287957


namespace NUMINAMATH_CALUDE_x_positive_iff_sum_ge_two_l2879_287917

theorem x_positive_iff_sum_ge_two (x : ℝ) : x > 0 ↔ x + 1/x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_x_positive_iff_sum_ge_two_l2879_287917


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2879_287972

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Complex.I) = -2 * Complex.I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2879_287972


namespace NUMINAMATH_CALUDE_parabola_inequality_l2879_287975

def f (x : ℝ) : ℝ := -(x - 2)^2

theorem parabola_inequality : f (-1) < f 4 ∧ f 4 < f 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_inequality_l2879_287975


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l2879_287955

def income : ℕ := 10000
def savings : ℕ := 2000

def expenditure : ℕ := income - savings

def ratio_simplify (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem income_expenditure_ratio :
  ratio_simplify income expenditure = (5, 4) := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l2879_287955


namespace NUMINAMATH_CALUDE_staircase_climbing_ways_l2879_287912

/-- The number of ways to climb n steps, where one can go up by 1, 2, or 3 steps at a time. -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | k + 3 => climbStairs k + climbStairs (k + 1) + climbStairs (k + 2)

/-- The number of steps in the staircase -/
def numSteps : ℕ := 10

/-- Theorem stating that there are 274 ways to climb a 10-step staircase -/
theorem staircase_climbing_ways : climbStairs numSteps = 274 := by
  sorry

end NUMINAMATH_CALUDE_staircase_climbing_ways_l2879_287912


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2879_287959

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := m^2 - 1 + (m + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) →
  (m = 1 ∧ (1 : ℂ) / (1 + z) = (1 : ℂ) / 5 - (2 : ℂ) / 5 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2879_287959


namespace NUMINAMATH_CALUDE_expected_value_red_balls_l2879_287980

/-- The expected value of drawing red balls in a specific scenario -/
theorem expected_value_red_balls :
  let total_balls : ℕ := 6
  let red_balls : ℕ := 4
  let white_balls : ℕ := 2
  let num_draws : ℕ := 6
  let p : ℚ := red_balls / total_balls
  let E_ξ : ℚ := num_draws * p
  E_ξ = 4 := by sorry

end NUMINAMATH_CALUDE_expected_value_red_balls_l2879_287980


namespace NUMINAMATH_CALUDE_exam_mean_score_l2879_287983

theorem exam_mean_score (score_below mean score_above : ℝ) 
  (h1 : score_below = mean - 7 * (score_above - mean) / 3)
  (h2 : score_above = mean + 3 * (score_above - mean) / 3)
  (h3 : score_below = 86)
  (h4 : score_above = 90) :
  mean = 88.8 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2879_287983


namespace NUMINAMATH_CALUDE_square_area_error_l2879_287971

/-- Given a square with a side measurement error of 38% in excess,
    the percentage of error in the calculated area is 90.44%. -/
theorem square_area_error (S : ℝ) (S_pos : S > 0) :
  let measured_side := S * (1 + 0.38)
  let actual_area := S^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.9044 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l2879_287971


namespace NUMINAMATH_CALUDE_roots_power_sum_divisible_l2879_287993

/-- Given two roots of a quadratic equation with a prime coefficient,
    their p-th powers sum to a multiple of p². -/
theorem roots_power_sum_divisible (p : ℕ) (x₁ x₂ : ℝ) 
  (h_prime : Nat.Prime p) 
  (h_p_gt_two : p > 2) 
  (h_roots : x₁^2 - p*x₁ + 1 = 0 ∧ x₂^2 - p*x₂ + 1 = 0) : 
  ∃ (k : ℤ), x₁^p + x₂^p = k * p^2 := by
  sorry

#check roots_power_sum_divisible

end NUMINAMATH_CALUDE_roots_power_sum_divisible_l2879_287993


namespace NUMINAMATH_CALUDE_chore_division_proof_l2879_287967

/-- Time to sweep one room in minutes -/
def sweep_time_per_room : ℕ := 3

/-- Time to wash one dish in minutes -/
def wash_dish_time : ℕ := 2

/-- Number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- Number of laundry loads Billy does -/
def billy_laundry_loads : ℕ := 2

/-- Number of dishes Billy washes -/
def billy_dishes : ℕ := 6

/-- Time to do one load of laundry in minutes -/
def laundry_time : ℕ := 9

theorem chore_division_proof :
  anna_rooms * sweep_time_per_room = 
  billy_laundry_loads * laundry_time + billy_dishes * wash_dish_time :=
by sorry

end NUMINAMATH_CALUDE_chore_division_proof_l2879_287967


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2879_287945

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2879_287945


namespace NUMINAMATH_CALUDE_isabel_homework_problem_l2879_287901

/-- Given a total number of problems, number of finished problems, and number of remaining pages,
    calculate the number of problems per page, assuming each page has an equal number of problems. -/
def problems_per_page (total : ℕ) (finished : ℕ) (pages : ℕ) : ℕ :=
  (total - finished) / pages

/-- Theorem stating that for the given problem, there are 8 problems per page. -/
theorem isabel_homework_problem :
  problems_per_page 72 32 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problem_l2879_287901


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l2879_287923

theorem polynomial_division_degree (f d q r : Polynomial ℝ) : 
  Polynomial.degree f = 15 →
  Polynomial.degree q = 8 →
  Polynomial.degree r = 2 →
  f = d * q + r →
  Polynomial.degree r < Polynomial.degree d →
  Polynomial.degree d = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l2879_287923


namespace NUMINAMATH_CALUDE_number_problem_l2879_287965

theorem number_problem : ∃ x : ℝ, x + 3 * x = 20 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_number_problem_l2879_287965


namespace NUMINAMATH_CALUDE_B_max_at_181_l2879_287925

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence B_k as defined in the problem -/
def B (k : ℕ) : ℝ := (binomial 2000 k) * (0.1 ^ k)

/-- The theorem stating that B_k is maximum when k = 181 -/
theorem B_max_at_181 : ∀ k ∈ Finset.range 2001, B 181 ≥ B k := by sorry

end NUMINAMATH_CALUDE_B_max_at_181_l2879_287925


namespace NUMINAMATH_CALUDE_max_acute_angles_2000_sided_polygon_l2879_287936

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  convex : Bool
  sides_eq : sides = n

/-- The maximum number of acute angles in a convex polygon -/
def max_acute_angles (p : ConvexPolygon n) : ℕ :=
  sorry

/-- Theorem: The maximum number of acute angles in a convex 2000-sided polygon is 3 -/
theorem max_acute_angles_2000_sided_polygon :
  ∀ (p : ConvexPolygon 2000), max_acute_angles p = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_acute_angles_2000_sided_polygon_l2879_287936


namespace NUMINAMATH_CALUDE_select_products_l2879_287918

theorem select_products (total : ℕ) (qualified : ℕ) (unqualified : ℕ) (select : ℕ) 
    (h1 : total = qualified + unqualified) 
    (h2 : total = 50) 
    (h3 : qualified = 47) 
    (h4 : unqualified = 3) 
    (h5 : select = 4) : 
    (Nat.choose unqualified 1 * Nat.choose qualified 3 + 
     Nat.choose unqualified 2 * Nat.choose qualified 2 + 
     Nat.choose unqualified 3 * Nat.choose qualified 1) = 
    (Nat.choose total 4 - Nat.choose qualified 4) := by
  sorry

end NUMINAMATH_CALUDE_select_products_l2879_287918


namespace NUMINAMATH_CALUDE_remainder_2457633_div_25_l2879_287998

theorem remainder_2457633_div_25 : 2457633 % 25 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2457633_div_25_l2879_287998


namespace NUMINAMATH_CALUDE_vasyas_birthday_vasyas_birthday_was_thursday_l2879_287946

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day after tomorrow
def dayAfterTomorrow (d : DayOfWeek) : DayOfWeek :=
  nextDay (nextDay d)

theorem vasyas_birthday (today : DayOfWeek) 
  (h1 : dayAfterTomorrow today = DayOfWeek.Sunday) 
  (h2 : nextDay today ≠ DayOfWeek.Sunday) : 
  nextDay (nextDay (nextDay today)) = DayOfWeek.Sunday := by
  sorry

-- The main theorem
theorem vasyas_birthday_was_thursday : 
  ∃ (today : DayOfWeek), 
    dayAfterTomorrow today = DayOfWeek.Sunday ∧ 
    nextDay today ≠ DayOfWeek.Sunday ∧
    nextDay (nextDay (nextDay today)) = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_vasyas_birthday_vasyas_birthday_was_thursday_l2879_287946


namespace NUMINAMATH_CALUDE_pascal_triangle_entries_l2879_287913

/-- The number of entries in the n-th row of Pascal's Triangle -/
def entriesInRow (n : ℕ) : ℕ := n + 1

/-- The sum of entries in the first n rows of Pascal's Triangle -/
def sumOfEntries (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem pascal_triangle_entries : sumOfEntries 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_entries_l2879_287913


namespace NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l2879_287970

def n : ℕ := 2310

-- Function to count positive divisors
def count_divisors (m : ℕ) : ℕ := sorry

-- Function to sum positive divisors
def sum_divisors (m : ℕ) : ℕ := sorry

-- Theorem stating the number of positive divisors of 2310
theorem number_of_divisors : count_divisors n = 32 := by sorry

-- Theorem stating the sum of positive divisors of 2310
theorem sum_of_divisors : sum_divisors n = 6912 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l2879_287970


namespace NUMINAMATH_CALUDE_remainder_of_sum_times_three_div_six_l2879_287921

/-- The sum of an arithmetic sequence with first term a, common difference d, and n terms -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The number of terms in the arithmetic sequence with first term 2, common difference 6, and last term 266 -/
def n : ℕ := 45

/-- The first term of the arithmetic sequence -/
def a : ℕ := 2

/-- The common difference of the arithmetic sequence -/
def d : ℕ := 6

/-- The last term of the arithmetic sequence -/
def last_term : ℕ := 266

theorem remainder_of_sum_times_three_div_six :
  (3 * arithmetic_sum a d n) % 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_remainder_of_sum_times_three_div_six_l2879_287921


namespace NUMINAMATH_CALUDE_seven_trapezoid_solutions_l2879_287961

/-- The number of solutions for the trapezoidal park problem -/
def trapezoid_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    p.1 % 5 = 0 ∧
    p.2 % 5 = 0 ∧
    75 * (p.1 + p.2) / 2 = 2250 ∧
    p.1 ≤ p.2)
    (Finset.product (Finset.range 61) (Finset.range 61))).card

/-- The theorem stating that there are exactly seven solutions -/
theorem seven_trapezoid_solutions : trapezoid_solutions = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_trapezoid_solutions_l2879_287961


namespace NUMINAMATH_CALUDE_inequality_addition_l2879_287927

theorem inequality_addition {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l2879_287927


namespace NUMINAMATH_CALUDE_min_sum_unpainted_cells_l2879_287931

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Checks if a cell is a corner cell -/
def is_corner (i j : Fin 10) : Prop :=
  (i = 0 ∨ i = 9) ∧ (j = 0 ∨ j = 9)

/-- Checks if two cells are neighbors -/
def are_neighbors (i1 j1 i2 j2 : Fin 10) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2))

/-- Checks if a cell should be painted based on its neighbors -/
def should_be_painted (t : Table) (i j : Fin 10) : Prop :=
  ∃ (i1 j1 i2 j2 : Fin 10), 
    are_neighbors i j i1 j1 ∧ 
    are_neighbors i j i2 j2 ∧ 
    t i j < t i1 j1 ∧ 
    t i j > t i2 j2

/-- The main theorem -/
theorem min_sum_unpainted_cells (t : Table) :
  (∃! (i1 j1 i2 j2 : Fin 10), 
    ¬is_corner i1 j1 ∧ 
    ¬is_corner i2 j2 ∧ 
    ¬should_be_painted t i1 j1 ∧ 
    ¬should_be_painted t i2 j2 ∧ 
    (∀ (i j : Fin 10), (i ≠ i1 ∨ j ≠ j1) ∧ (i ≠ i2 ∨ j ≠ j2) → should_be_painted t i j)) →
  (∃ (i1 j1 i2 j2 : Fin 10), 
    ¬is_corner i1 j1 ∧ 
    ¬is_corner i2 j2 ∧ 
    ¬should_be_painted t i1 j1 ∧ 
    ¬should_be_painted t i2 j2 ∧ 
    t i1 j1 + t i2 j2 = 3 ∧
    (∀ (k1 l1 k2 l2 : Fin 10), 
      ¬is_corner k1 l1 ∧ 
      ¬is_corner k2 l2 ∧ 
      ¬should_be_painted t k1 l1 ∧ 
      ¬should_be_painted t k2 l2 → 
      t k1 l1 + t k2 l2 ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_unpainted_cells_l2879_287931


namespace NUMINAMATH_CALUDE_second_number_is_30_l2879_287992

theorem second_number_is_30 (a b c : ℚ) 
  (sum_eq : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8) :
  b = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_30_l2879_287992


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_seven_power_minus_one_l2879_287963

theorem largest_power_of_two_dividing_seven_power_minus_one :
  (∃ (n : ℕ), 2^n ∣ (7^2048 - 1)) ∧
  (∀ (m : ℕ), m > 14 → ¬(2^m ∣ (7^2048 - 1))) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_seven_power_minus_one_l2879_287963


namespace NUMINAMATH_CALUDE_red_balls_count_l2879_287988

def bag_sizes : List Nat := [7, 15, 16, 10, 23]

def total_balls : Nat := bag_sizes.sum

structure BallConfiguration where
  red : Nat
  yellow : Nat
  blue : Nat

def is_valid_configuration (config : BallConfiguration) : Prop :=
  config.red ∈ bag_sizes ∧
  config.yellow + config.blue = total_balls - config.red ∧
  config.yellow = 2 * config.blue

theorem red_balls_count : ∃ (config : BallConfiguration), 
  is_valid_configuration config ∧ config.red = 23 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2879_287988


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l2879_287903

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) := 2 * a^2 + 3 * b - a * b

/-- Theorem stating that 4 * 3 = 29 under the custom multiplication -/
theorem custom_mult_four_three : custom_mult 4 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l2879_287903


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2879_287948

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 176) (h2 : divisor = 19) (h3 : quotient = 9) :
  dividend - divisor * quotient = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2879_287948


namespace NUMINAMATH_CALUDE_club_equation_solution_l2879_287902

/-- Define the ♣ operation -/
def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 7

/-- Theorem stating that 17 is the unique solution to A ♣ 6 = 70 -/
theorem club_equation_solution :
  ∃! A : ℝ, club A 6 = 70 ∧ A = 17 := by
  sorry

end NUMINAMATH_CALUDE_club_equation_solution_l2879_287902


namespace NUMINAMATH_CALUDE_max_ab_value_l2879_287906

def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : ∃ (ε : ℝ), ε ≠ 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → 
    f a b x ≤ f a b 1 ∨ f a b x ≥ f a b 1) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∃ (ε : ℝ), ε ≠ 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → 
      f a' b' x ≤ f a' b' 1 ∨ f a' b' x ≥ f a' b' 1) → 
    a' * b' ≤ a * b) →
  a * b = 9 := by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2879_287906


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2879_287929

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (h_given_line : given_line = ⟨2, 1, -5⟩) 
  (h_point : point = ⟨1, 0⟩) : 
  ∃ (parallel_line : Line), 
    parallelLines given_line parallel_line ∧ 
    pointOnLine point parallel_line ∧ 
    parallel_line = ⟨2, 1, -2⟩ :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2879_287929


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2879_287922

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = x -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b = a) : 
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2879_287922


namespace NUMINAMATH_CALUDE_boxes_ordered_correct_l2879_287904

/-- Represents the number of apples in each box -/
def apples_per_box : ℕ := 300

/-- Represents the fraction of stock sold -/
def fraction_sold : ℚ := 3/4

/-- Represents the number of unsold apples -/
def unsold_apples : ℕ := 750

/-- Calculates the number of boxes ordered each week -/
def boxes_ordered : ℕ := 10

/-- Proves that the number of boxes ordered is correct given the conditions -/
theorem boxes_ordered_correct :
  (1 - fraction_sold) * (apples_per_box * boxes_ordered) = unsold_apples := by sorry

end NUMINAMATH_CALUDE_boxes_ordered_correct_l2879_287904


namespace NUMINAMATH_CALUDE_max_value_expression_l2879_287997

theorem max_value_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) ≤ 3/2 * (c^2 + d^2)) ∧
  (∃ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) = 3/2 * (c^2 + d^2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2879_287997


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l2879_287900

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ ∃ k, l2 a (x + k) (y + k * (a / 2))

-- Define perpendicular lines
def perpendicular (a : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, 
  l1 a x₁ y₁ ∧ l2 a x₂ y₂ → (x₂ - x₁) * (a * (x₂ - x₁) + 2 * (y₂ - y₁)) + (y₂ - y₁) * ((a - 1) * (x₂ - x₁) + (y₂ - y₁)) = 0

-- Theorem for parallel lines
theorem parallel_lines : ∀ a : ℝ, parallel a → a = -1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines : ∀ a : ℝ, perpendicular a → a = 2/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l2879_287900


namespace NUMINAMATH_CALUDE_laura_debt_l2879_287932

/-- Calculates the total amount owed after applying simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the total amount owed after one year is $37.10 -/
theorem laura_debt : 
  let principal : ℝ := 35
  let rate : ℝ := 0.06
  let time : ℝ := 1
  total_amount_owed principal rate time = 37.10 := by
sorry

end NUMINAMATH_CALUDE_laura_debt_l2879_287932


namespace NUMINAMATH_CALUDE_probability_mathematics_letter_l2879_287907

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_mathematics_letter : 
  (unique_letters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_mathematics_letter_l2879_287907


namespace NUMINAMATH_CALUDE_lavender_candles_count_l2879_287977

theorem lavender_candles_count (almond coconut lavender : ℕ) : 
  almond = 10 →
  coconut = (3 * almond) / 2 →
  lavender = 2 * coconut →
  lavender = 30 := by
sorry

end NUMINAMATH_CALUDE_lavender_candles_count_l2879_287977


namespace NUMINAMATH_CALUDE_solve_josie_problem_l2879_287979

def josie_problem (initial_amount : ℕ) (cassette_cost : ℕ) (num_cassettes : ℕ) (remaining_amount : ℕ) : Prop :=
  let total_cassette_cost := cassette_cost * num_cassettes
  let amount_after_cassettes := initial_amount - total_cassette_cost
  let headphone_cost := amount_after_cassettes - remaining_amount
  headphone_cost = 25

theorem solve_josie_problem :
  josie_problem 50 9 2 7 := by sorry

end NUMINAMATH_CALUDE_solve_josie_problem_l2879_287979


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l2879_287996

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLucky (n : ℕ) : Prop := n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf7 (n : ℕ) : Prop := n % 7 = 0

theorem least_non_lucky_multiple_of_7 : 
  (∀ k : ℕ, k > 0 ∧ k < 14 ∧ isMultipleOf7 k → isLucky k) ∧ 
  isMultipleOf7 14 ∧ 
  ¬isLucky 14 := by sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l2879_287996


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2879_287915

/-- Given a line L1 with equation 3x + 6y = 12 and a point P (2, -1),
    prove that the line L2 with equation y = -1/2x is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (∃ (m b : ℝ), 3*x + 6*y = 12 ↔ y = m*x + b) → -- L1 exists
  (y = -1/2 * x) →                              -- L2 equation
  (∃ (m : ℝ), 3*x + 6*y = 12 ↔ y = m*x + 2) →   -- L1 in slope-intercept form
  (-1 = -1/2 * 2 + 0) →                         -- L2 passes through (2, -1)
  (∃ (k : ℝ), y = -1/2 * x + k ∧ -1 = -1/2 * 2 + k) -- L2 in point-slope form
  :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2879_287915


namespace NUMINAMATH_CALUDE_trajectory_of_point_M_l2879_287964

/-- The trajectory of point M given the specified conditions -/
theorem trajectory_of_point_M :
  ∀ (x y : ℝ),
    x ≠ -1 →
    x ≠ 1 →
    y ≠ 0 →
    (y / (x + 1)) / (y / (x - 1)) = 2 →
    x = -3 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_M_l2879_287964


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l2879_287951

/-- The total shaded area of a carpet with a circle and squares -/
theorem carpet_shaded_area (carpet_side : ℝ) (circle_diameter : ℝ) (square_side : ℝ) : 
  carpet_side = 12 →
  carpet_side / circle_diameter = 4 →
  circle_diameter / square_side = 4 →
  (π * (circle_diameter / 2)^2) + (8 * square_side^2) = (9 * π / 4) + (9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l2879_287951


namespace NUMINAMATH_CALUDE_johann_manipulation_l2879_287926

theorem johann_manipulation (x y k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hk : k > 1) : 
  x * k - y / k > x - y := by
  sorry

end NUMINAMATH_CALUDE_johann_manipulation_l2879_287926


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l2879_287919

theorem unique_solution_floor_equation :
  ∀ n : ℤ, (⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ↔ n = 7 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l2879_287919


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_curve_l2879_287943

/-- The value of 'a' for which the asymptotes of the hyperbola x²/9 - y²/4 = 1 
    are precisely the two tangent lines of the curve y = ax² + 1/3 -/
theorem hyperbola_asymptotes_tangent_curve (a : ℝ) : 
  (∀ x y : ℝ, x^2/9 - y^2/4 = 1 → 
    ∃ k : ℝ, (y = k*x ∨ y = -k*x) ∧ 
    (∀ x₀ : ℝ, (k*x₀ = a*x₀^2 + 1/3 → 
      ∀ x : ℝ, k*x ≤ a*x^2 + 1/3) ∧
    (-k*x₀ = a*x₀^2 + 1/3 → 
      ∀ x : ℝ, -k*x ≤ a*x^2 + 1/3))) →
  a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_tangent_curve_l2879_287943


namespace NUMINAMATH_CALUDE_weaving_factory_profit_maximization_l2879_287954

/-- Represents the profit maximization problem in a weaving factory --/
theorem weaving_factory_profit_maximization 
  (total_workers : ℕ) 
  (fabric_per_worker : ℕ) 
  (clothing_per_worker : ℕ) 
  (fabric_per_clothing : ℚ) 
  (fabric_profit : ℚ) 
  (clothing_profit : ℕ) 
  (h_total : total_workers = 150)
  (h_fabric : fabric_per_worker = 30)
  (h_clothing : clothing_per_worker = 4)
  (h_fabric_clothing : fabric_per_clothing = 3/2)
  (h_fabric_profit : fabric_profit = 2)
  (h_clothing_profit : clothing_profit = 25) :
  ∃ (x : ℕ), 
    x ≤ total_workers ∧ 
    (clothing_profit * clothing_per_worker * x : ℚ) + 
    (fabric_profit * (fabric_per_worker * (total_workers - x) - 
    fabric_per_clothing * clothing_per_worker * x)) = 11800 ∧
    x = 100 := by
  sorry

end NUMINAMATH_CALUDE_weaving_factory_profit_maximization_l2879_287954


namespace NUMINAMATH_CALUDE_flower_count_l2879_287938

theorem flower_count (red_green : ℕ) (red_yellow : ℕ) (green_yellow : ℕ)
  (h1 : red_green = 62)
  (h2 : red_yellow = 49)
  (h3 : green_yellow = 77) :
  ∃ (red green yellow : ℕ),
    red + green = red_green ∧
    red + yellow = red_yellow ∧
    green + yellow = green_yellow ∧
    red = 17 ∧ green = 45 ∧ yellow = 32 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l2879_287938


namespace NUMINAMATH_CALUDE_annas_lemonade_sales_l2879_287966

/-- Anna's lemonade sales problem -/
theorem annas_lemonade_sales 
  (plain_glasses : ℕ) 
  (plain_price : ℚ) 
  (plain_strawberry_difference : ℚ) 
  (h1 : plain_glasses = 36)
  (h2 : plain_price = 3/4)
  (h3 : plain_strawberry_difference = 11) :
  (plain_glasses : ℚ) * plain_price - plain_strawberry_difference = 16 := by
  sorry


end NUMINAMATH_CALUDE_annas_lemonade_sales_l2879_287966


namespace NUMINAMATH_CALUDE_water_truck_capacity_l2879_287940

/-- The maximum capacity of the water truck in tons -/
def truck_capacity : ℝ := 12

/-- The amount of water (in tons) injected by pipe A when used with pipe C -/
def water_A_with_C : ℝ := 4

/-- The amount of water (in tons) injected by pipe B when used with pipe C -/
def water_B_with_C : ℝ := 6

/-- The ratio of pipe B's injection rate to pipe A's injection rate -/
def rate_ratio_B_to_A : ℝ := 2

theorem water_truck_capacity :
  truck_capacity = water_A_with_C * rate_ratio_B_to_A ∧
  truck_capacity = water_B_with_C + water_A_with_C :=
by sorry

end NUMINAMATH_CALUDE_water_truck_capacity_l2879_287940


namespace NUMINAMATH_CALUDE_a_on_diameter_bck_l2879_287935

/-- Given a triangle ABC with vertices A(a,b), B(0,0), C(c,0), and K as the intersection
    of the bisector of the exterior angle at C and the interior angle at B,
    prove that A lies on the line passing through K and the circumcenter of triangle BCK. -/
theorem a_on_diameter_bck (a b c : ℝ) : 
  let A : ℝ × ℝ := (a, b)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (c, 0)
  let u : ℝ := Real.sqrt (a^2 + b^2)
  let w : ℝ := Real.sqrt ((a - c)^2 + b^2)
  let K : ℝ × ℝ := (c * (a + u) / (c + u - w), b * c / (c + u - w))
  let O : ℝ × ℝ := (c / 2, c * (a - c + w) * (c + u + w) / (2 * b * (c + u - w)))
  (∃ t : ℝ, A = (1 - t) • K + t • O) := by
    sorry

end NUMINAMATH_CALUDE_a_on_diameter_bck_l2879_287935


namespace NUMINAMATH_CALUDE_handshake_theorem_l2879_287937

def num_people : ℕ := 8

def handshake_arrangements (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (n - 1) * handshake_arrangements (n - 2)

theorem handshake_theorem :
  handshake_arrangements num_people = 105 :=
by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l2879_287937


namespace NUMINAMATH_CALUDE_expressions_evaluation_l2879_287987

theorem expressions_evaluation :
  let expr1 := (1) * (Real.sqrt 48 - 4 * Real.sqrt (1/8)) - (2 * Real.sqrt (1/3) - 2 * Real.sqrt 0.5)
  let expr2 := Real.sqrt ((-2)^2) - |1 - Real.sqrt 3| + (3 - Real.sqrt 3) * (1 + 1 / Real.sqrt 3)
  (expr1 = (10/3) * Real.sqrt 3) ∧ (expr2 = 5 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_expressions_evaluation_l2879_287987


namespace NUMINAMATH_CALUDE_probability_same_parity_l2879_287991

-- Define the type for function parity
inductive Parity
| Even
| Odd
| Neither

-- Define a function to represent the parity of each given function
def function_parity : Fin 4 → Parity
| 0 => Parity.Neither  -- y = x^3 + 3x^2
| 1 => Parity.Even     -- y = (e^x + e^-x) / 2
| 2 => Parity.Odd      -- y = log_2 ((3-x)/(3+x))
| 3 => Parity.Even     -- y = x sin x

-- Define a function to check if two functions have the same parity
def same_parity (f1 f2 : Fin 4) : Bool :=
  match function_parity f1, function_parity f2 with
  | Parity.Even, Parity.Even => true
  | Parity.Odd, Parity.Odd => true
  | _, _ => false

-- Theorem statement
theorem probability_same_parity :
  (Finset.filter (fun p => same_parity p.1 p.2) (Finset.univ : Finset (Fin 4 × Fin 4))).card /
  (Finset.univ : Finset (Fin 4 × Fin 4)).card = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_same_parity_l2879_287991


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2879_287982

theorem rectangle_max_area (perimeter : ℝ) (width : ℝ) (length : ℝ) (area : ℝ) : 
  perimeter = 40 →
  length = 2 * width →
  perimeter = 2 * (length + width) →
  area = length * width →
  area = 800 / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2879_287982


namespace NUMINAMATH_CALUDE_prob_select_dime_l2879_287952

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The total value of quarters in the container in dollars -/
def total_quarters_value : ℚ := 12.50

/-- The total value of nickels in the container in dollars -/
def total_nickels_value : ℚ := 15.00

/-- The total value of dimes in the container in dollars -/
def total_dimes_value : ℚ := 5.00

/-- The probability of randomly selecting a dime from the container -/
theorem prob_select_dime : 
  (total_dimes_value / dime_value) / 
  ((total_quarters_value / quarter_value) + 
   (total_nickels_value / nickel_value) + 
   (total_dimes_value / dime_value)) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_prob_select_dime_l2879_287952


namespace NUMINAMATH_CALUDE_total_worksheets_l2879_287924

/-- Given a teacher grading worksheets, this theorem proves the total number of worksheets. -/
theorem total_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) : 
  problems_per_worksheet = 4 →
  graded_worksheets = 5 →
  remaining_problems = 16 →
  graded_worksheets + (remaining_problems / problems_per_worksheet) = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_worksheets_l2879_287924


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2879_287999

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.07 →
  price * tax_rate1 - price * tax_rate2 = 0.25 := by sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l2879_287999


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_solution_is_correct_l2879_287920

theorem largest_multiple_of_seven (n : ℤ) : 
  (n % 7 = 0 ∧ -n > -150) → n ≤ 147 :=
by sorry

theorem solution_is_correct : 
  147 % 7 = 0 ∧ -147 > -150 ∧ 
  ∀ m : ℤ, (m % 7 = 0 ∧ -m > -150) → m ≤ 147 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_solution_is_correct_l2879_287920


namespace NUMINAMATH_CALUDE_odd_composite_quotient_l2879_287962

theorem odd_composite_quotient : 
  let first_four := [9, 15, 21, 25]
  let next_four := [27, 33, 35, 39]
  (first_four.prod : ℚ) / (next_four.prod : ℚ) = 25 / 429 := by
  sorry

end NUMINAMATH_CALUDE_odd_composite_quotient_l2879_287962


namespace NUMINAMATH_CALUDE_gcd_7163_209_l2879_287973

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_gcd_7163_209_l2879_287973


namespace NUMINAMATH_CALUDE_fibonacci_seventh_term_l2879_287914

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_seventh_term : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_seventh_term_l2879_287914


namespace NUMINAMATH_CALUDE_intersection_A_B_l2879_287994

def A : Set ℝ := { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2879_287994


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2879_287930

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 3) : a^3 + b^3 = 280 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2879_287930


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l2879_287908

theorem bakery_flour_usage : 
  let wheat_flour : ℝ := 0.2
  let white_flour : ℝ := 0.1
  let rye_flour : ℝ := 0.15
  let almond_flour : ℝ := 0.05
  let rice_flour : ℝ := 0.1
  wheat_flour + white_flour + rye_flour + almond_flour + rice_flour = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l2879_287908


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l2879_287995

theorem cyclist_speed_ratio : 
  ∀ (v_A v_B v_C : ℝ),
    v_A > 0 → v_B > 0 → v_C > 0 →
    (v_A - v_B) * 4 = 20 →
    (v_A + v_C) * 2 = 30 →
    v_A / v_B = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l2879_287995


namespace NUMINAMATH_CALUDE_half_work_completed_l2879_287958

/-- Represents the highway construction project -/
structure HighwayProject where
  initialMen : ℕ
  totalLength : ℝ
  initialDays : ℕ
  initialHoursPerDay : ℕ
  actualDays : ℕ
  additionalMen : ℕ
  newHoursPerDay : ℕ

/-- Calculates the fraction of work completed -/
def fractionCompleted (project : HighwayProject) : ℚ :=
  let initialManHours := project.initialMen * project.initialDays * project.initialHoursPerDay
  let actualManHours := project.initialMen * project.actualDays * project.initialHoursPerDay
  actualManHours / initialManHours

/-- Theorem stating that the fraction of work completed is 1/2 -/
theorem half_work_completed (project : HighwayProject) 
  (h1 : project.initialMen = 100)
  (h2 : project.totalLength = 2)
  (h3 : project.initialDays = 50)
  (h4 : project.initialHoursPerDay = 8)
  (h5 : project.actualDays = 25)
  (h6 : project.additionalMen = 60)
  (h7 : project.newHoursPerDay = 10)
  (h8 : (project.initialMen + project.additionalMen) * (project.initialDays - project.actualDays) * project.newHoursPerDay = project.initialMen * project.initialDays * project.initialHoursPerDay / 2) :
  fractionCompleted project = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_half_work_completed_l2879_287958


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_equal_l2879_287978

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a transversal
def transversal (t l1 l2 : Line) : Prop := sorry

-- Define alternate interior angles
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 t : Line) : Prop := sorry

-- Define corresponding angles
def correspondingAngles (a1 a2 : Angle) (l1 l2 t : Line) : Prop := sorry

-- Theorem statement
theorem parallel_lines_corresponding_angles_not_always_equal :
  ∃ (l1 l2 t : Line) (a1 a2 : Angle),
    parallel l1 l2 ∧
    transversal t l1 l2 ∧
    correspondingAngles a1 a2 l1 l2 t ∧
    a1 ≠ a2 :=
  sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_equal_l2879_287978


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l2879_287969

theorem quadratic_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    (∀ x, a * x^2 - b * x + c = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (∀ y, c * y^2 - a * y + b = 0 ↔ y = y₁ ∨ y = y₂) ∧
    (b / a ≥ 0) ∧
    (c / a = 9 * (a / c))) →
  (b / a) / (b / c) = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l2879_287969


namespace NUMINAMATH_CALUDE_no_married_triple_possible_all_married_triples_possible_l2879_287934

/-- Represents a person in the alien race -/
structure Person where
  gender : Fin 3
  likes : Fin 3 → Finset (Fin n)

/-- Represents a married triple -/
structure MarriedTriple where
  male : Fin n
  female : Fin n
  emale : Fin n

/-- The set of all persons in the colony -/
def Colony (n : ℕ) : Finset Person := sorry

/-- Predicate to check if a person likes another person -/
def likes (p1 p2 : Person) : Prop := sorry

/-- Predicate to check if a triple is a valid married triple -/
def isMarriedTriple (t : MarriedTriple) (c : Colony n) : Prop := sorry

theorem no_married_triple_possible 
  (n : ℕ) 
  (k : ℕ) 
  (h1 : Even n) 
  (h2 : k ≥ n / 2) : 
  ∃ (c : Colony n), ∀ (t : MarriedTriple), ¬isMarriedTriple t c := by sorry

theorem all_married_triples_possible 
  (n : ℕ) 
  (k : ℕ) 
  (h : k ≥ 3 * n / 4) : 
  ∃ (c : Colony n), ∃ (triples : Finset MarriedTriple), 
    (∀ t ∈ triples, isMarriedTriple t c) ∧ 
    (triples.card = n) := by sorry

end NUMINAMATH_CALUDE_no_married_triple_possible_all_married_triples_possible_l2879_287934


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l2879_287947

theorem rectangle_width_proof (w : ℝ) (h1 : w > 0) : 
  (∃ l : ℝ, l > 0 ∧ l = 3 * w ∧ l + w = 3 * (l * w)) → w = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l2879_287947


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_closed_one_l2879_287976

def M : Set ℝ := {x | 0 < Real.log (x + 1) ∧ Real.log (x + 1) < 3}

def N : Set ℝ := {y | ∃ x ∈ M, y = Real.sin x}

theorem M_intersect_N_eq_open_zero_closed_one : M ∩ N = Set.Ioo 0 1 ∪ {1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_closed_one_l2879_287976


namespace NUMINAMATH_CALUDE_f_increasing_for_x_gt_1_l2879_287939

-- Define the function f(x) = (x-1)^2 + 1
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

-- State the theorem
theorem f_increasing_for_x_gt_1 : ∀ x > 1, deriv f x > 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_for_x_gt_1_l2879_287939


namespace NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l2879_287950

/-- Represents a counter in the Ring Mafia game -/
inductive Counter
| Mafia
| Town

/-- Represents the state of the Ring Mafia game -/
structure GameState where
  counters : List Counter
  total_counters : Nat
  mafia_counters : Nat
  town_counters : Nat

/-- Represents a strategy for Tony -/
def TonyStrategy := GameState → List Nat

/-- Represents a strategy for Madeline -/
def MadelineStrategy := GameState → Nat

/-- Checks if a game state is valid according to the rules -/
def is_valid_game_state (state : GameState) : Prop :=
  state.total_counters = 2019 ∧
  state.mafia_counters = 673 ∧
  state.town_counters = 1346 ∧
  state.counters.length = state.total_counters

/-- Checks if the game has ended -/
def game_ended (state : GameState) : Prop :=
  state.mafia_counters = 0 ∨ state.town_counters = 0

/-- Theorem: There is no winning strategy for Tony in Ring Mafia -/
theorem no_winning_strategy_for_tony :
  ∀ (initial_state : GameState),
    is_valid_game_state initial_state →
    ∀ (tony_strategy : TonyStrategy),
    ∃ (madeline_strategy : MadelineStrategy),
    ∃ (final_state : GameState),
      game_ended final_state ∧
      final_state.town_counters = 0 :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_for_tony_l2879_287950


namespace NUMINAMATH_CALUDE_snail_wins_l2879_287956

-- Define the race parameters
def race_distance : ℝ := 200

-- Define the animals' movements
structure Snail where
  speed : ℝ
  
structure Rabbit where
  initial_distance : ℝ
  speed : ℝ
  run_time1 : ℝ
  nap_time1 : ℝ
  run_time2 : ℝ
  nap_time2 : ℝ

-- Define the race conditions
def race_conditions (s : Snail) (r : Rabbit) : Prop :=
  s.speed > 0 ∧
  r.speed > 0 ∧
  r.initial_distance = 120 ∧
  r.run_time1 > 0 ∧
  r.nap_time1 > 0 ∧
  r.run_time2 > 0 ∧
  r.nap_time2 > 0 ∧
  r.initial_distance + r.speed * (r.run_time1 + r.run_time2) = race_distance

-- Theorem statement
theorem snail_wins (s : Snail) (r : Rabbit) 
  (h : race_conditions s r) : 
  s.speed * (r.run_time1 + r.nap_time1 + r.run_time2 + r.nap_time2) = race_distance :=
sorry

end NUMINAMATH_CALUDE_snail_wins_l2879_287956


namespace NUMINAMATH_CALUDE_milburg_children_l2879_287909

/-- The number of children in Milburg -/
def children : ℕ := 8243 - 5256

/-- The total population of Milburg -/
def total_population : ℕ := 8243

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

theorem milburg_children : children = 2987 := by
  sorry

end NUMINAMATH_CALUDE_milburg_children_l2879_287909


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2879_287985

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (∀ x : ℝ, x^2 + 16*x = 100 ↔ x = Real.sqrt a - b ∨ x = -Real.sqrt a - b) ∧ 
  (Real.sqrt a - b > 0) ∧
  (a + b = 172) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2879_287985


namespace NUMINAMATH_CALUDE_contrapositive_rhombus_diagonals_l2879_287911

-- Define a quadrilateral type
structure Quadrilateral where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define the property of being a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define the property of having perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  sorry

-- State the theorem
theorem contrapositive_rhombus_diagonals :
  (∀ q : Quadrilateral, ¬(has_perpendicular_diagonals q) → ¬(is_rhombus q)) ↔
  (∀ q : Quadrilateral, is_rhombus q → has_perpendicular_diagonals q) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_rhombus_diagonals_l2879_287911


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2879_287990

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let nonzero_digits := digits.filter (· ≠ 0)
  (nonzero_digits.reverse.take 2).foldl (fun acc d => acc * 10 + d) 0

theorem last_two_nonzero_digits_80_factorial :
  last_two_nonzero_digits (factorial 80) = 12 := by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2879_287990


namespace NUMINAMATH_CALUDE_smallest_positive_angle_for_neg1990_l2879_287984

-- Define the concept of angle equivalence
def angle_equivalent (a b : Int) : Prop :=
  ∃ k : Int, b - a = k * 360

-- Define the smallest positive equivalent angle
def smallest_positive_equivalent (a : Int) : Int :=
  let b := a % 360
  if b < 0 then b + 360 else b

-- Theorem statement
theorem smallest_positive_angle_for_neg1990 :
  smallest_positive_equivalent (-1990) = 170 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_for_neg1990_l2879_287984


namespace NUMINAMATH_CALUDE_aquarium_width_l2879_287910

theorem aquarium_width (length height : ℝ) (volume_final : ℝ) : 
  length = 4 → height = 3 → volume_final = 54 → 
  ∃ (width : ℝ), 3 * ((length * width * height) / 4) = volume_final ∧ width = 6 := by
sorry

end NUMINAMATH_CALUDE_aquarium_width_l2879_287910


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_zero_not_in_odd_units_digits_smallest_non_odd_units_digit_is_zero_l2879_287989

def OddUnitsDigits : Set ℕ := {1, 3, 5, 7, 9}
def StandardDigits : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit : 
  ∀ d ∈ StandardDigits, d ∉ OddUnitsDigits → d ≥ 0 :=
by sorry

theorem zero_not_in_odd_units_digits : 0 ∉ OddUnitsDigits :=
by sorry

theorem smallest_non_odd_units_digit_is_zero : 
  ∀ d ∈ StandardDigits, d ∉ OddUnitsDigits → d ≥ 0 ∧ 0 ∉ OddUnitsDigits ∧ 0 ∈ StandardDigits :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_zero_not_in_odd_units_digits_smallest_non_odd_units_digit_is_zero_l2879_287989


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2879_287933

theorem no_real_roots_for_nonzero_k (k : ℝ) (hk : k ≠ 0) :
  ∀ x : ℝ, x^2 + k*x + 2*k^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2879_287933


namespace NUMINAMATH_CALUDE_combination_permutation_equation_l2879_287928

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Falling factorial -/
def fallingFactorial (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

theorem combination_permutation_equation : 
  ∃ x : ℕ, binomial (x + 5) x = binomial (x + 3) (x - 1) + binomial (x + 3) (x - 2) + 
    (3 * fallingFactorial (x + 3) 3) / 4 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_equation_l2879_287928


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l2879_287960

/-- Given 60 feet of fencing, the maximum area of a rectangular pen is 225 square feet. -/
theorem max_rectangular_pen_area :
  ∀ (width height : ℝ),
  width > 0 →
  height > 0 →
  2 * (width + height) = 60 →
  width * height ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l2879_287960


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l2879_287942

theorem linear_function_not_in_first_quadrant :
  ∀ x y : ℝ, y = -2 * x - 1 → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l2879_287942


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2879_287986

theorem cube_volume_problem (s : ℝ) : 
  s > 0 → 
  (s + 2) * (s - 3) * s - s^3 = 26 → 
  s^3 = 343 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2879_287986


namespace NUMINAMATH_CALUDE_condition1_arrangements_condition2_arrangements_condition3_arrangements_l2879_287916

def num_boys : ℕ := 5
def num_girls : ℕ := 3
def num_subjects : ℕ := 5

def arrangements_condition1 : ℕ := 5520
def arrangements_condition2 : ℕ := 3360
def arrangements_condition3 : ℕ := 360

/-- The number of ways to select representatives under condition 1 -/
theorem condition1_arrangements :
  (Nat.choose num_boys num_boys +
   Nat.choose num_boys (num_boys - 1) * Nat.choose num_girls 1 +
   Nat.choose num_boys (num_boys - 2) * Nat.choose num_girls 2) *
  Nat.factorial num_subjects = arrangements_condition1 := by sorry

/-- The number of ways to select representatives under condition 2 -/
theorem condition2_arrangements :
  Nat.choose (num_boys + num_girls - 1) (num_subjects - 1) *
  (num_subjects - 1) * Nat.factorial (num_subjects - 1) = arrangements_condition2 := by sorry

/-- The number of ways to select representatives under condition 3 -/
theorem condition3_arrangements :
  Nat.choose (num_boys + num_girls - 2) (num_subjects - 2) *
  (num_subjects - 2) * Nat.factorial (num_subjects - 2) = arrangements_condition3 := by sorry

end NUMINAMATH_CALUDE_condition1_arrangements_condition2_arrangements_condition3_arrangements_l2879_287916


namespace NUMINAMATH_CALUDE_peanut_difference_l2879_287953

theorem peanut_difference (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_peanuts = 133)
  (h3 : kenya_peanuts > jose_peanuts) :
  kenya_peanuts - jose_peanuts = 48 := by
  sorry

end NUMINAMATH_CALUDE_peanut_difference_l2879_287953


namespace NUMINAMATH_CALUDE_chord_arrangement_count_l2879_287905

/-- The number of ways to choose k items from n items without replacement and without regard to order. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to connect 4 points with 3 chords such that each chord intersects the other two. -/
def fourPointConnections : ℕ := 8

/-- The number of ways to connect 5 points with 3 chords such that exactly two chords share a common endpoint and the remaining chord intersects these two. -/
def fivePointConnections : ℕ := 5

/-- The total number of ways to arrange three intersecting chords among 20 points on a circle. -/
def totalChordArrangements : ℕ := 
  choose 20 3 + choose 20 4 * fourPointConnections + 
  choose 20 5 * fivePointConnections + choose 20 6

theorem chord_arrangement_count : totalChordArrangements = 156180 := by
  sorry

end NUMINAMATH_CALUDE_chord_arrangement_count_l2879_287905


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2879_287981

theorem expression_simplification_and_evaluation :
  ∀ a : ℕ,
    a ≠ 0 →
    a ≠ 1 →
    2 * a - 3 ≤ 1 →
    a = 2 →
    (a - (2 * a - 1) / a) / ((a^2 - 1) / a) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2879_287981


namespace NUMINAMATH_CALUDE_fifth_stair_area_fifth_stair_perimeter_twelfth_stair_area_twentyfifth_stair_perimeter_l2879_287968

-- Define the stair structure
structure Stair :=
  (n : ℕ)

-- Define the area function
def area (s : Stair) : ℕ :=
  s.n * (s.n + 1) / 2

-- Define the perimeter function
def perimeter (s : Stair) : ℕ :=
  4 * s.n

-- Theorem statements
theorem fifth_stair_area :
  area { n := 5 } = 15 := by sorry

theorem fifth_stair_perimeter :
  perimeter { n := 5 } = 20 := by sorry

theorem twelfth_stair_area :
  area { n := 12 } = 78 := by sorry

theorem twentyfifth_stair_perimeter :
  perimeter { n := 25 } = 100 := by sorry

end NUMINAMATH_CALUDE_fifth_stair_area_fifth_stair_perimeter_twelfth_stair_area_twentyfifth_stair_perimeter_l2879_287968


namespace NUMINAMATH_CALUDE_students_passed_l2879_287949

theorem students_passed (total : ℕ) (failure_rate : ℚ) : 
  total = 1000 → failure_rate = 0.4 → (total : ℚ) * (1 - failure_rate) = 600 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_l2879_287949


namespace NUMINAMATH_CALUDE_isosceles_max_perimeter_l2879_287941

-- Define a triangle
structure Triangle where
  base : ℝ
  angle : ℝ
  side1 : ℝ
  side2 : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.base + t.side1 + t.side2

-- Define an isosceles triangle
def isIsosceles (t : Triangle) : Prop := t.side1 = t.side2

-- Theorem statement
theorem isosceles_max_perimeter (b a : ℝ) :
  ∀ (t : Triangle), t.base = b → t.angle = a →
    ∃ (t_iso : Triangle), t_iso.base = b ∧ t_iso.angle = a ∧ isIsosceles t_iso ∧
      perimeter t_iso ≥ perimeter t :=
sorry

end NUMINAMATH_CALUDE_isosceles_max_perimeter_l2879_287941
