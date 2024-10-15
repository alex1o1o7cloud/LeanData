import Mathlib

namespace NUMINAMATH_CALUDE_fraction_sum_squared_l3883_388372

theorem fraction_sum_squared (x y z m n p : ℝ) 
  (h1 : x/m + y/n + z/p = 1)
  (h2 : m/x + n/y + p/z = 0) :
  x^2/m^2 + y^2/n^2 + z^2/p^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l3883_388372


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3883_388313

theorem fraction_sum_equality (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 =
  1 / (b - c) + 1 / (c - a) + 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3883_388313


namespace NUMINAMATH_CALUDE_team_a_remaining_days_l3883_388302

/-- The number of days Team A needs to complete the project alone initially -/
def team_a_initial_days : ℚ := 24

/-- The number of days Team B needs to complete the project alone initially -/
def team_b_initial_days : ℚ := 18

/-- The number of days Team A needs to complete the project after receiving 6 people from Team B -/
def team_a_after_transfer_days : ℚ := 18

/-- The number of days Team B needs to complete the project after transferring 6 people to Team A -/
def team_b_after_transfer_days : ℚ := 24

/-- The number of days Team B works alone -/
def team_b_alone_days : ℚ := 6

/-- The number of days both teams work together -/
def teams_together_days : ℚ := 4

/-- The efficiency of one person per day -/
def efficiency_per_person : ℚ := 1 / 432

/-- The number of people in Team A -/
def team_a_people : ℚ := team_a_initial_days / efficiency_per_person

/-- The number of people in Team B -/
def team_b_people : ℚ := team_b_initial_days / efficiency_per_person

/-- The theorem stating that Team A needs 26/3 more days to complete the project -/
theorem team_a_remaining_days : 
  ∃ (m : ℚ), 
    (team_a_people * efficiency_per_person * (team_b_alone_days + teams_together_days) + 
     team_b_people * teams_together_days * efficiency_per_person + 
     team_a_people * m * efficiency_per_person = 1) ∧ 
    m = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_team_a_remaining_days_l3883_388302


namespace NUMINAMATH_CALUDE_point_c_in_second_quadrant_l3883_388317

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Given points -/
def pointA : Point := ⟨5, 3⟩
def pointB : Point := ⟨5, -3⟩
def pointC : Point := ⟨-5, 3⟩
def pointD : Point := ⟨-5, -3⟩

/-- Theorem: Point C is the only point in the second quadrant -/
theorem point_c_in_second_quadrant :
  isInSecondQuadrant pointC ∧
  ¬isInSecondQuadrant pointA ∧
  ¬isInSecondQuadrant pointB ∧
  ¬isInSecondQuadrant pointD :=
by sorry

end NUMINAMATH_CALUDE_point_c_in_second_quadrant_l3883_388317


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3883_388394

theorem negation_of_existence_proposition :
  (¬∃ x : ℝ, x > 0 ∧ Real.sin x > 2^x - 1) ↔ (∀ x : ℝ, x > 0 → Real.sin x ≤ 2^x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3883_388394


namespace NUMINAMATH_CALUDE_square_min_rotation_angle_l3883_388387

/-- The minimum rotation angle for a square to coincide with its original position -/
def min_rotation_angle_square : ℝ := 90

/-- A square has rotational symmetry of order 4 -/
def rotational_symmetry_order_square : ℕ := 4

theorem square_min_rotation_angle :
  min_rotation_angle_square = 360 / rotational_symmetry_order_square :=
by sorry

end NUMINAMATH_CALUDE_square_min_rotation_angle_l3883_388387


namespace NUMINAMATH_CALUDE_ten_lines_intersection_points_l3883_388365

/-- The number of intersection points of n lines in a plane, where no lines are parallel
    and exactly two lines pass through each intersection point. -/
def intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 2

/-- Given 10 lines in a plane where no lines are parallel and exactly two lines pass through
    each intersection point, the number of intersection points is 45. -/
theorem ten_lines_intersection_points :
  intersection_points 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_lines_intersection_points_l3883_388365


namespace NUMINAMATH_CALUDE_pasture_consumption_l3883_388301

/-- Represents the pasture scenario with cows and grass -/
structure Pasture where
  /-- Amount of grass each cow eats per day -/
  daily_consumption : ℝ
  /-- Daily growth rate of the grass -/
  daily_growth : ℝ
  /-- Original amount of grass in the pasture -/
  initial_grass : ℝ

/-- Given the conditions, proves that 94 cows will consume all grass in 28 days -/
theorem pasture_consumption (p : Pasture) : 
  (p.initial_grass + 25 * p.daily_growth = 100 * 25 * p.daily_consumption) →
  (p.initial_grass + 35 * p.daily_growth = 84 * 35 * p.daily_consumption) →
  (p.initial_grass + 28 * p.daily_growth = 94 * 28 * p.daily_consumption) :=
by sorry

end NUMINAMATH_CALUDE_pasture_consumption_l3883_388301


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3883_388391

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3883_388391


namespace NUMINAMATH_CALUDE_euclidean_continued_fraction_connection_l3883_388332

/-- Euclidean algorithm steps -/
def euclidean_steps (m n : ℕ) : List (ℕ × ℕ) :=
  sorry

/-- Continued fraction representation -/
def continued_fraction (as : List ℕ) : ℚ :=
  sorry

/-- Theorem connecting Euclidean algorithm and continued fractions -/
theorem euclidean_continued_fraction_connection (m n : ℕ) (h : m < n) :
  let steps := euclidean_steps m n
  let as := steps.map Prod.fst
  ∀ k, k ≤ steps.length →
    continued_fraction (as.drop k) =
      (steps.get! k).snd / (steps.get! (k - 1)).snd :=
by sorry

end NUMINAMATH_CALUDE_euclidean_continued_fraction_connection_l3883_388332


namespace NUMINAMATH_CALUDE_smallest_positive_e_l3883_388342

def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

theorem smallest_positive_e (a b c d e : ℤ) : 
  let p := fun (x : ℝ) => (a : ℝ) * x^4 + (b : ℝ) * x^3 + (c : ℝ) * x^2 + (d : ℝ) * x + (e : ℝ)
  (is_root p (-3) ∧ is_root p 6 ∧ is_root p 10 ∧ is_root p (-1/2)) →
  (e > 0) →
  (∀ e' : ℤ, e' > 0 → 
    let p' := fun (x : ℝ) => (a : ℝ) * x^4 + (b : ℝ) * x^3 + (c : ℝ) * x^2 + (d : ℝ) * x + (e' : ℝ)
    (is_root p' (-3) ∧ is_root p' 6 ∧ is_root p' 10 ∧ is_root p' (-1/2)) → e' ≥ e) →
  e = 180 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_e_l3883_388342


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3883_388388

theorem complex_fraction_sum : 
  let U := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - 3) - 1 / (3 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - Real.sqrt 11)
  U = 10 + Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3883_388388


namespace NUMINAMATH_CALUDE_tangent_line_a_range_l3883_388395

/-- The line equation ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 = 0

/-- The first circle equation (x-1)² + y² = 1 -/
def circle1_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The second circle equation x² + (y-1)² = 1/4 -/
def circle2_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1/4

/-- The line is tangent to both circles -/
def is_tangent_to_both_circles (a : ℝ) : Prop :=
  ∃ x y, line_equation a x y ∧ 
         ((circle1_equation x y ∧ ¬∃ x' y', x' ≠ x ∧ y' ≠ y ∧ line_equation a x' y' ∧ circle1_equation x' y') ∨
          (circle2_equation x y ∧ ¬∃ x' y', x' ≠ x ∧ y' ≠ y ∧ line_equation a x' y' ∧ circle2_equation x' y'))

theorem tangent_line_a_range :
  ∀ a : ℝ, is_tangent_to_both_circles a ↔ -Real.sqrt 3 < a ∧ a < 3/4 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_a_range_l3883_388395


namespace NUMINAMATH_CALUDE_find_divisor_l3883_388326

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 12401)
  (h2 : quotient = 76)
  (h3 : remainder = 13)
  (h4 : dividend = quotient * 163 + remainder) :
  163 = dividend / quotient :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l3883_388326


namespace NUMINAMATH_CALUDE_point_P_properties_l3883_388368

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-3*a - 4, 2 + a)

-- Define the point Q
def Q : ℝ × ℝ := (5, 8)

theorem point_P_properties (a : ℝ) :
  -- Case 1: P lies on x-axis
  (P a).1 = 2 ∧ (P a).2 = 0 → a = -2
  ∧
  -- Case 2: PQ is parallel to y-axis
  (P a).1 = Q.1 → a = -3
  ∧
  -- Case 3: P is in second quadrant and equidistant from axes
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| → 
    a = -1 ∧ (-1 : ℝ)^2023 + 2023 = 2022 :=
by sorry

end NUMINAMATH_CALUDE_point_P_properties_l3883_388368


namespace NUMINAMATH_CALUDE_max_students_distribution_l3883_388346

theorem max_students_distribution (pens toys books : ℕ) 
  (h_pens : pens = 451) 
  (h_toys : toys = 410) 
  (h_books : books = 325) : 
  (∃ (students : ℕ), students > 0 ∧ 
    pens % students = 0 ∧ 
    toys % students = 0 ∧ 
    books % students = 0) →
  (∀ (n : ℕ), n > 1 → 
    (pens % n ≠ 0 ∨ toys % n ≠ 0 ∨ books % n ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3883_388346


namespace NUMINAMATH_CALUDE_sand_weight_difference_l3883_388397

def box_weight : ℕ := 250
def box_filled_weight : ℕ := 1780
def bucket_weight : ℕ := 460
def bucket_filled_weight : ℕ := 2250

theorem sand_weight_difference :
  (bucket_filled_weight - bucket_weight) - (box_filled_weight - box_weight) = 260 :=
by sorry

end NUMINAMATH_CALUDE_sand_weight_difference_l3883_388397


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_of_2_52_l3883_388355

def decimal_to_fraction (d : ℚ) : ℤ × ℤ :=
  let n := d.num
  let d := d.den
  let g := n.gcd d
  (n / g, d / g)

theorem sum_of_fraction_parts_of_2_52 :
  let (n, d) := decimal_to_fraction (252 / 100)
  n + d = 88 := by sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_of_2_52_l3883_388355


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3883_388314

/-- Calculates the simple interest rate given the principal, time, and interest amount. -/
def calculate_interest_rate (principal : ℕ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / ((principal : ℚ) * (time : ℚ))

theorem interest_rate_problem (principal time interest_difference : ℕ) 
  (h1 : principal = 3000)
  (h2 : time = 5)
  (h3 : interest_difference = 2400)
  (h4 : principal - interest_difference > 0) :
  calculate_interest_rate principal time (principal - interest_difference) = 4 := by
  sorry

#eval calculate_interest_rate 3000 5 600

end NUMINAMATH_CALUDE_interest_rate_problem_l3883_388314


namespace NUMINAMATH_CALUDE_arithmetic_matrix_properties_l3883_388340

/-- Represents a matrix with the given properties -/
def ArithmeticMatrix (n : ℕ) (d : ℕ → ℝ) : Prop :=
  n ≥ 3 ∧
  ∀ m k, m ≤ n → k ≤ n → 
    (∃ a : ℕ → ℕ → ℝ, 
      a m k = 1 + (k - 1) * d m ∧
      (∀ i, i ≤ n → a i 1 = 1) ∧
      (∀ i j, i ≤ n → j < n → a i (j + 1) - a i j = d i) ∧
      (∀ i j, i < n → j ≤ n → a (i + 1) j - a i j = a (i + 1) 1 - a i 1))

/-- The main theorem -/
theorem arithmetic_matrix_properties {n : ℕ} {d : ℕ → ℝ} 
  (h : ArithmeticMatrix n d) :
  (∃ c : ℝ, d 2 - d 1 = d 3 - d 2) ∧
  (∀ m, 3 ≤ m → m ≤ n → d m = (2 - m) * d 1 + (m - 1) * d 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_matrix_properties_l3883_388340


namespace NUMINAMATH_CALUDE_triathlon_completion_time_l3883_388389

/-- A triathlon participant's speeds and completion time -/
theorem triathlon_completion_time 
  (swim_dist : ℝ) 
  (cycle_dist : ℝ) 
  (run_dist : ℝ) 
  (swim_speed : ℝ) 
  (h1 : swim_dist = 1.5) 
  (h2 : cycle_dist = 40) 
  (h3 : run_dist = 10) 
  (h4 : swim_speed > 0) 
  (h5 : swim_speed * 5 * 2.5 * (swim_dist / swim_speed + run_dist / (5 * swim_speed)) = 
        cycle_dist + swim_speed * 5 * 2.5 * 6) : 
  swim_dist / swim_speed + cycle_dist / (swim_speed * 5 * 2.5) + run_dist / (swim_speed * 5) = 134 :=
by sorry

end NUMINAMATH_CALUDE_triathlon_completion_time_l3883_388389


namespace NUMINAMATH_CALUDE_period_of_cos_3x_l3883_388339

/-- The period of cos(3x) is 2π/3 -/
theorem period_of_cos_3x :
  let f : ℝ → ℝ := λ x ↦ Real.cos (3 * x)
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ x, f (x + S) ≠ f x :=
by sorry

end NUMINAMATH_CALUDE_period_of_cos_3x_l3883_388339


namespace NUMINAMATH_CALUDE_cos_alpha_plus_7pi_12_l3883_388367

theorem cos_alpha_plus_7pi_12 (α : ℝ) (h : Real.sin (α + π/12) = 1/3) :
  Real.cos (α + 7*π/12) = -(1 + Real.sqrt 24) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_7pi_12_l3883_388367


namespace NUMINAMATH_CALUDE_factor_expression_l3883_388366

theorem factor_expression (y : ℝ) : 3*y*(y-4) + 8*(y-4) - 2*(y-4) = 3*(y+2)*(y-4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3883_388366


namespace NUMINAMATH_CALUDE_largest_six_digit_with_product_60_l3883_388305

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def digit_product (n : ℕ) : ℕ := (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem largest_six_digit_with_product_60 :
  ∃ M : ℕ, is_six_digit M ∧ 
           digit_product M = 60 ∧ 
           (∀ n : ℕ, is_six_digit n → digit_product n = 60 → n ≤ M) ∧
           digit_sum M = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_with_product_60_l3883_388305


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l3883_388392

/-- A rectangular prism is a three-dimensional shape with six faces. -/
structure RectangularPrism where
  faces : Fin 6 → Rectangle

/-- The number of edges in a rectangular prism -/
def edges (p : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def corners (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def faces (p : RectangularPrism) : ℕ := 6

/-- The sum of edges, corners, and faces in a rectangular prism is 26 -/
theorem rectangular_prism_sum (p : RectangularPrism) : 
  edges p + corners p + faces p = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l3883_388392


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3883_388348

theorem nested_fraction_evaluation : 
  2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3883_388348


namespace NUMINAMATH_CALUDE_complex_modulus_l3883_388373

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = -3 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3883_388373


namespace NUMINAMATH_CALUDE_f_properties_l3883_388304

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem f_properties : 
  (∀ x ≠ 0, f (-x) = -f x) ∧ 
  (∀ x, ∃ y, f x = y) ∧
  (∀ y, f ⁻¹' {y} ≠ ∅ → -1 < y ∧ y < 1) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3883_388304


namespace NUMINAMATH_CALUDE_deposit_time_problem_l3883_388398

/-- Proves that given the conditions of the problem, the deposit time is 3 years -/
theorem deposit_time_problem (initial_deposit : ℝ) (final_amount : ℝ) (final_amount_higher_rate : ℝ) 
  (h1 : initial_deposit = 8000)
  (h2 : final_amount = 10200)
  (h3 : final_amount_higher_rate = 10680) :
  ∃ (r : ℝ), 
    final_amount = initial_deposit + initial_deposit * (r / 100) * 3 ∧
    final_amount_higher_rate = initial_deposit + initial_deposit * ((r + 2) / 100) * 3 :=
sorry

end NUMINAMATH_CALUDE_deposit_time_problem_l3883_388398


namespace NUMINAMATH_CALUDE_breath_holding_improvement_l3883_388393

/-- Calculates the final breath-holding time after three weeks of practice --/
def final_breath_holding_time (initial_time : ℝ) : ℝ :=
  let after_first_week := initial_time * 2
  let after_second_week := after_first_week * 2
  after_second_week * 1.5

/-- Theorem stating that given an initial breath-holding time of 10 seconds,
    the final time after three weeks of practice is 60 seconds --/
theorem breath_holding_improvement :
  final_breath_holding_time 10 = 60 := by
  sorry

#eval final_breath_holding_time 10

end NUMINAMATH_CALUDE_breath_holding_improvement_l3883_388393


namespace NUMINAMATH_CALUDE_unique_prime_square_product_l3883_388396

theorem unique_prime_square_product (a b c : ℕ) : 
  (Nat.Prime (a^2 + 1)) ∧ 
  (Nat.Prime (b^2 + 1)) ∧ 
  ((a^2 + 1) * (b^2 + 1) = c^2 + 1) →
  a = 2 ∧ b = 1 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_square_product_l3883_388396


namespace NUMINAMATH_CALUDE_false_or_false_is_false_l3883_388375

theorem false_or_false_is_false (p q : Prop) (hp : ¬p) (hq : ¬q) : ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_false_or_false_is_false_l3883_388375


namespace NUMINAMATH_CALUDE_bills_equal_at_100_minutes_l3883_388358

/-- United Telephone's base rate in dollars -/
def united_base : ℚ := 7

/-- United Telephone's per-minute charge in dollars -/
def united_per_minute : ℚ := 1/4

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℚ := 12

/-- Atlantic Call's per-minute charge in dollars -/
def atlantic_per_minute : ℚ := 1/5

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℚ := 100

theorem bills_equal_at_100_minutes :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
sorry

end NUMINAMATH_CALUDE_bills_equal_at_100_minutes_l3883_388358


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3883_388307

theorem rationalize_denominator : 
  ∃ (A B C D E F G H I : ℤ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11) = 
      (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + G * Real.sqrt H) / I ∧
    I > 0 ∧
    A = -6 ∧ B = 5 ∧ C = -8 ∧ D = 3 ∧ E = 3 ∧ F = 11 ∧ G = 1 ∧ H = 165 ∧ I = 51 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3883_388307


namespace NUMINAMATH_CALUDE_max_pairs_proof_max_pairs_achievable_l3883_388356

/-- The maximum number of pairs that can be chosen from the set {1, 2, ..., 2017}
    such that a_i < b_i, no two pairs share a common element, and all sums a_i + b_i
    are distinct and less than or equal to 2017. -/
def max_pairs : ℕ := 806

theorem max_pairs_proof :
  ∀ (k : ℕ) (a b : Fin k → ℕ),
  (∀ i : Fin k, a i < b i) →
  (∀ i : Fin k, b i ≤ 2017) →
  (∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) →
  (∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) →
  (∀ i : Fin k, a i + b i ≤ 2017) →
  k ≤ max_pairs :=
by sorry

theorem max_pairs_achievable :
  ∃ (k : ℕ) (a b : Fin k → ℕ),
  k = max_pairs ∧
  (∀ i : Fin k, a i < b i) ∧
  (∀ i : Fin k, b i ≤ 2017) ∧
  (∀ i j : Fin k, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) ∧
  (∀ i j : Fin k, i ≠ j → a i + b i ≠ a j + b j) ∧
  (∀ i : Fin k, a i + b i ≤ 2017) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_proof_max_pairs_achievable_l3883_388356


namespace NUMINAMATH_CALUDE_smallest_n_for_equal_cost_l3883_388351

theorem smallest_n_for_equal_cost : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬∃ (r g b : ℕ+), 18 * r = 21 * g ∧ 21 * g = 25 * b ∧ 25 * b = 24 * m) ∧
  (∃ (r g b : ℕ+), 18 * r = 21 * g ∧ 21 * g = 25 * b ∧ 25 * b = 24 * n) ∧
  n = 132 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_equal_cost_l3883_388351


namespace NUMINAMATH_CALUDE_fraction_chain_l3883_388334

theorem fraction_chain (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4)
  : e / a = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_chain_l3883_388334


namespace NUMINAMATH_CALUDE_max_volume_cube_l3883_388362

/-- A rectangular solid with length l, width w, and height h -/
structure RectangularSolid where
  l : ℝ
  w : ℝ
  h : ℝ
  l_pos : 0 < l
  w_pos : 0 < w
  h_pos : 0 < h

/-- The surface area of a rectangular solid -/
def surfaceArea (r : RectangularSolid) : ℝ :=
  2 * (r.l * r.w + r.l * r.h + r.w * r.h)

/-- The volume of a rectangular solid -/
def volume (r : RectangularSolid) : ℝ :=
  r.l * r.w * r.h

/-- Theorem: Among all rectangular solids with a fixed surface area S,
    the cube has the maximum volume, and this maximum volume is (S/6)^(3/2) -/
theorem max_volume_cube (S : ℝ) (h_pos : 0 < S) :
  ∃ (max_vol : ℝ),
    (∀ (r : RectangularSolid), surfaceArea r = S → volume r ≤ max_vol) ∧
    (∃ (cube : RectangularSolid), surfaceArea cube = S ∧ volume cube = max_vol) ∧
    max_vol = (S / 6) ^ (3/2) :=
  sorry

end NUMINAMATH_CALUDE_max_volume_cube_l3883_388362


namespace NUMINAMATH_CALUDE_dress_pocket_ratio_l3883_388354

/-- Proves that the ratio of dresses with 2 pockets to the total number of dresses with pockets is 1:3 --/
theorem dress_pocket_ratio :
  ∀ (x y : ℕ),
  -- Total number of dresses
  24 = x + y + (24 / 2) →
  -- Total number of pockets
  2 * x + 3 * y = 32 →
  -- Ratio of dresses with 2 pockets to total dresses with pockets
  x / (x + y) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_dress_pocket_ratio_l3883_388354


namespace NUMINAMATH_CALUDE_sunday_to_weekday_ratio_is_correct_l3883_388324

/-- The weight ratio of Sunday papers to Monday-Saturday papers --/
def sunday_to_weekday_ratio : ℚ :=
  let weekday_paper_weight : ℚ := 8  -- ounces
  let papers_per_day : ℕ := 250
  let weeks : ℕ := 10
  let weekdays_per_week : ℕ := 6
  let recycling_rate : ℚ := 100 / 2000  -- $/pound

  let total_weekday_papers : ℕ := papers_per_day * weekdays_per_week * weeks
  let total_weekday_weight : ℚ := weekday_paper_weight * total_weekday_papers
  
  let total_sunday_papers : ℕ := papers_per_day * weeks
  let total_sunday_weight : ℚ := 2000 * 16  -- 1 ton in ounces
  
  let sunday_paper_weight : ℚ := total_sunday_weight / total_sunday_papers
  
  sunday_paper_weight / weekday_paper_weight

theorem sunday_to_weekday_ratio_is_correct : sunday_to_weekday_ratio = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_sunday_to_weekday_ratio_is_correct_l3883_388324


namespace NUMINAMATH_CALUDE_tv_show_cost_per_episode_l3883_388328

/-- Given a TV show season with the following properties:
  * The season has 22 episodes
  * The total cost of the season is $35,200
  * The second half of the season costs 120% more per episode than the first half
  Prove that the cost per episode for the first half of the season is $1,000. -/
theorem tv_show_cost_per_episode 
  (total_episodes : ℕ) 
  (total_cost : ℚ) 
  (second_half_increase : ℚ) :
  total_episodes = 22 →
  total_cost = 35200 →
  second_half_increase = 1.2 →
  let first_half_cost := total_cost / (total_episodes / 2 * (1 + 1 + second_half_increase))
  first_half_cost = 1000 := by
sorry

end NUMINAMATH_CALUDE_tv_show_cost_per_episode_l3883_388328


namespace NUMINAMATH_CALUDE_sum_of_digits_l3883_388335

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_single_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def divisible_by_11 (n : ℕ) : Prop := ∃ k : ℕ, n = 11 * k

theorem sum_of_digits (a b : ℕ) : 
  is_single_digit a → 
  is_single_digit b → 
  is_three_digit (700 + 10 * a + 1) →
  is_three_digit (100 * b + 60 + 5) →
  (700 + 10 * a + 1) + 184 = (100 * b + 60 + 5) →
  divisible_by_11 (100 * b + 60 + 5) →
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3883_388335


namespace NUMINAMATH_CALUDE_probability_three_ones_two_twos_l3883_388310

def roll_probability : ℕ → ℚ
  | 1 => 1/6
  | 2 => 1/6
  | _ => 2/3

def num_rolls : ℕ := 5

theorem probability_three_ones_two_twos :
  (Nat.choose num_rolls 3) * (roll_probability 1)^3 * (roll_probability 2)^2 = 5/3888 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_ones_two_twos_l3883_388310


namespace NUMINAMATH_CALUDE_game_result_l3883_388336

def score (n : Nat) : Nat :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def allieRolls : List Nat := [3, 5, 6, 2, 4]
def bettyRolls : List Nat := [3, 2, 1, 6, 4]

def totalScore (rolls : List Nat) : Nat :=
  (rolls.map score).sum

theorem game_result : totalScore allieRolls * totalScore bettyRolls = 256 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l3883_388336


namespace NUMINAMATH_CALUDE_max_a_value_l3883_388378

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 3

theorem max_a_value (a : ℝ) :
  (∃ m n : ℝ, m ∈ Set.Icc 1 5 ∧ n ∈ Set.Icc 1 5 ∧ n - m ≥ 2 ∧ f a m = f a n) →
  a ≤ Real.log 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3883_388378


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3883_388359

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 10*m^2 + 11*m + 2 = 81*n^3 + 27*n^2 + 3*n - 8 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3883_388359


namespace NUMINAMATH_CALUDE_monotonic_range_a_l3883_388361

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem monotonic_range_a :
  (∀ a : ℝ, ∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_range_a_l3883_388361


namespace NUMINAMATH_CALUDE_derivative_of_square_root_l3883_388337

theorem derivative_of_square_root (x : ℝ) (h : x > 0) :
  deriv (fun x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by
sorry

end NUMINAMATH_CALUDE_derivative_of_square_root_l3883_388337


namespace NUMINAMATH_CALUDE_wholesale_price_correct_l3883_388331

/-- The retail price of the machine -/
def retail_price : ℝ := 167.99999999999997

/-- The discount rate applied to the retail price -/
def discount_rate : ℝ := 0.10

/-- The profit rate as a percentage of the wholesale price -/
def profit_rate : ℝ := 0.20

/-- The wholesale price of the machine -/
def wholesale_price : ℝ := 126.00

/-- Theorem stating that the given wholesale price is correct -/
theorem wholesale_price_correct : 
  wholesale_price = (retail_price * (1 - discount_rate)) / (1 + profit_rate) :=
sorry

end NUMINAMATH_CALUDE_wholesale_price_correct_l3883_388331


namespace NUMINAMATH_CALUDE_number_equation_l3883_388370

theorem number_equation (x : ℚ) (N : ℚ) : x = 9 → (N - 5 / x = 4 + 4 / x ↔ N = 5) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3883_388370


namespace NUMINAMATH_CALUDE_area_triangle_PF1F2_is_sqrt15_l3883_388308

/-- The ellipse with equation x²/9 + y²/5 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) + (p.2^2 / 5) = 1}

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point on the ellipse satisfying the given condition -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of triangle PF₁F₂ is √15 -/
theorem area_triangle_PF1F2_is_sqrt15 
  (h_P_on_ellipse : P ∈ Ellipse)
  (h_PF1_eq_2PF2 : distance P F1 = 2 * distance P F2) :
  (1/2) * distance F1 F2 * Real.sqrt (distance P F1 ^ 2 - (distance F1 F2 / 2) ^ 2) = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_PF1F2_is_sqrt15_l3883_388308


namespace NUMINAMATH_CALUDE_max_stores_visited_l3883_388353

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) (double_visitors : ℕ) 
  (h1 : total_stores = 8)
  (h2 : total_visits = 23)
  (h3 : total_shoppers = 12)
  (h4 : double_visitors = 8)
  (h5 : double_visitors ≤ total_shoppers)
  (h6 : double_visitors * 2 ≤ total_visits)
  (h7 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits ≤ total_stores ∧ 
    max_visits * 1 + (total_shoppers - 1) * 1 + double_visitors * 1 = total_visits ∧
    ∀ n : ℕ, n ≤ total_shoppers → n * total_stores ≥ total_visits → n ≥ total_shoppers - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l3883_388353


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l3883_388376

/-- The volume of a sphere circumscribing a cube with edge length 2 is 4√3π -/
theorem sphere_volume_circumscribing_cube (cube_edge : ℝ) (sphere_volume : ℝ) : 
  cube_edge = 2 →
  sphere_volume = (4 / 3) * Real.pi * (Real.sqrt 3) ^ 3 →
  sphere_volume = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_cube_l3883_388376


namespace NUMINAMATH_CALUDE_outfits_from_five_shirts_three_pants_l3883_388377

/-- The number of outfits that can be made from a given number of shirts and pants -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) : ℕ := shirts * pants

/-- Theorem: Given 5 shirts and 3 pairs of pants, the number of outfits is 15 -/
theorem outfits_from_five_shirts_three_pants : 
  number_of_outfits 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfits_from_five_shirts_three_pants_l3883_388377


namespace NUMINAMATH_CALUDE_oak_trees_after_five_days_l3883_388344

/-- Calculates the final number of oak trees in the park after 5 days -/
def final_oak_trees (initial : ℕ) (plant_rate_1 plant_rate_2 remove_rate_1 remove_rate_2 : ℕ) : ℕ :=
  let net_change_1 := (plant_rate_1 - remove_rate_1) * 2
  let net_change_2 := (plant_rate_2 - remove_rate_1)
  let net_change_3 := (plant_rate_2 - remove_rate_2) * 2
  initial + net_change_1 + net_change_2 + net_change_3

/-- Theorem stating that given the initial number of oak trees and planting/removal rates, 
    the final number of oak trees after 5 days will be 15 -/
theorem oak_trees_after_five_days :
  final_oak_trees 5 3 4 2 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_five_days_l3883_388344


namespace NUMINAMATH_CALUDE_cost_difference_l3883_388381

def ice_cream_quantity : ℕ := 100
def yoghurt_quantity : ℕ := 35
def cheese_quantity : ℕ := 50
def milk_quantity : ℕ := 20

def ice_cream_price : ℚ := 12
def yoghurt_price : ℚ := 3
def cheese_price : ℚ := 8
def milk_price : ℚ := 4

def ice_cream_discount : ℚ := 0.05
def yoghurt_tax : ℚ := 0.08
def cheese_discount : ℚ := 0.10

def returned_ice_cream : ℕ := 10
def returned_cheese : ℕ := 5

def adjusted_ice_cream_cost : ℚ :=
  (ice_cream_quantity * ice_cream_price) * (1 - ice_cream_discount) -
  (returned_ice_cream * ice_cream_price)

def adjusted_yoghurt_cost : ℚ :=
  (yoghurt_quantity * yoghurt_price) * (1 + yoghurt_tax)

def adjusted_cheese_cost : ℚ :=
  (cheese_quantity * cheese_price) * (1 - cheese_discount) -
  (returned_cheese * cheese_price)

def adjusted_milk_cost : ℚ :=
  milk_quantity * milk_price

theorem cost_difference :
  adjusted_ice_cream_cost + adjusted_cheese_cost -
  (adjusted_yoghurt_cost + adjusted_milk_cost) = 1146.60 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l3883_388381


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3883_388386

/-- The maximum area of an equilateral triangle inscribed in a 12x5 rectangle --/
theorem max_area_equilateral_triangle_in_rectangle :
  ∃ (A : ℝ),
    A = (25 : ℝ) * Real.sqrt 3 / 3 ∧
    ∀ (s : ℝ),
      s > 0 →
      s ≤ 12 →
      s * Real.sqrt 3 / 2 ≤ 5 →
      (Real.sqrt 3 / 4) * s^2 ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l3883_388386


namespace NUMINAMATH_CALUDE_car_distance_traveled_l3883_388306

theorem car_distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 23 → time = 3 → distance = speed * time → distance = 69 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l3883_388306


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3883_388311

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 < 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3883_388311


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3883_388399

theorem simplify_sqrt_expression :
  (2 * Real.sqrt 10) / (Real.sqrt 4 + Real.sqrt 3 + Real.sqrt 5) =
  (4 * Real.sqrt 10 - 15 * Real.sqrt 2) / 11 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3883_388399


namespace NUMINAMATH_CALUDE_emma_harry_weight_l3883_388300

/-- Given the weights of pairs of students, prove the combined weight of Emma and Harry -/
theorem emma_harry_weight
  (emma_fiona : ℝ)
  (fiona_george : ℝ)
  (george_harry : ℝ)
  (h_emma_fiona : emma_fiona = 280)
  (h_fiona_george : fiona_george = 260)
  (h_george_harry : george_harry = 290) :
  ∃ (emma harry : ℝ),
    emma + harry = 310 ∧
    ∃ (fiona george : ℝ),
      emma + fiona = emma_fiona ∧
      fiona + george = fiona_george ∧
      george + harry = george_harry :=
sorry

end NUMINAMATH_CALUDE_emma_harry_weight_l3883_388300


namespace NUMINAMATH_CALUDE_brittany_age_is_32_l3883_388315

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

/-- Theorem: Brittany's age when she returns from vacation is 32 -/
theorem brittany_age_is_32 : brittany_age_after_vacation 25 3 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_brittany_age_is_32_l3883_388315


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3883_388385

theorem inequality_system_solution (x : ℝ) : 
  (1 - (2*x - 1)/2 > (3*x - 1)/4 ∧ 2 - 3*x ≤ 4 - x) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3883_388385


namespace NUMINAMATH_CALUDE_notebooks_per_student_in_second_half_l3883_388322

/-- Given a classroom with students and notebooks, prove that each student
    in the second half has 3 notebooks. -/
theorem notebooks_per_student_in_second_half
  (total_students : ℕ)
  (total_notebooks : ℕ)
  (notebooks_per_first_half_student : ℕ)
  (h1 : total_students = 28)
  (h2 : total_notebooks = 112)
  (h3 : notebooks_per_first_half_student = 5)
  (h4 : 2 ∣ total_students) :
  (total_notebooks - (total_students / 2 * notebooks_per_first_half_student)) / (total_students / 2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_notebooks_per_student_in_second_half_l3883_388322


namespace NUMINAMATH_CALUDE_equation_proof_l3883_388323

theorem equation_proof : Real.sqrt ((5568 / 87) ^ (1/3) + Real.sqrt (72 * 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3883_388323


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3883_388352

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3883_388352


namespace NUMINAMATH_CALUDE_product_of_factors_for_six_factor_number_l3883_388371

def has_six_factors (x : ℕ) : Prop :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 6

def product_of_factors (x : ℕ) : ℕ :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).prod id

theorem product_of_factors_for_six_factor_number (x : ℕ) 
  (h1 : x > 1) (h2 : has_six_factors x) : 
  product_of_factors x = x^3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_factors_for_six_factor_number_l3883_388371


namespace NUMINAMATH_CALUDE_monotone_iff_bound_l3883_388312

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*x + m

/-- f is monotonically increasing -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x : ℝ, f' m x ≥ 0

theorem monotone_iff_bound (m : ℝ) :
  is_monotone_increasing m ↔ m ≥ 4/3 :=
sorry

end NUMINAMATH_CALUDE_monotone_iff_bound_l3883_388312


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3883_388360

/-- Given two rectangles with equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a length of 9 inches, prove that the width of the second rectangle is 20 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 12)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 9)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3883_388360


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3883_388318

/-- Given a geometric sequence with first term a₁ = 2, 
    the minimum value of 3a₂ + 6a₃ is -3/2. -/
theorem min_value_geometric_sequence (r : ℝ) : 
  let a₁ : ℝ := 2
  let a₂ : ℝ := a₁ * r
  let a₃ : ℝ := a₂ * r
  3 * a₂ + 6 * a₃ ≥ -3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3883_388318


namespace NUMINAMATH_CALUDE_expand_polynomial_l3883_388343

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x - 6) = 4 * x^3 + 7 * x^2 - 21 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3883_388343


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equation_solutions_l3883_388303

theorem sum_of_reciprocal_equation_solutions : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ j k : ℕ, j > 0 ∧ k > 0 ∧ 1 / j + 1 / k = (1 : ℚ) / 4 ∧ n = j + k) ∧ 
  (∀ j k : ℕ, j > 0 → k > 0 → 1 / j + 1 / k = (1 : ℚ) / 4 → (j + k) ∈ S) ∧
  S.sum id = 59 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equation_solutions_l3883_388303


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3883_388330

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (1, 1)

theorem vector_perpendicular : (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3883_388330


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l3883_388357

/-- Given an equilateral cone with volume V and a tetrahedron circumscribed around it 
    with an equilateral triangle base, this theorem proves the volume and surface area 
    of the tetrahedron. -/
theorem tetrahedron_volume_and_surface_area 
  (V : ℝ) -- Volume of the equilateral cone
  (h : V > 0) -- Assumption that volume is positive
  : 
  ∃ (K F : ℝ), 
    K = (3 * V * Real.sqrt 3) / Real.pi ∧ 
    F = 9 * Real.sqrt 3 * (((3 * V ^ 2) / Real.pi ^ 2) ^ (1/3 : ℝ)) ∧
    K > 0 ∧ 
    F > 0
  := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l3883_388357


namespace NUMINAMATH_CALUDE_square_divisibility_l3883_388325

theorem square_divisibility (n : ℕ+) (h : ∀ m : ℕ+, m ∣ n → m ≤ 12) : 144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l3883_388325


namespace NUMINAMATH_CALUDE_quadratic_range_on_unit_interval_l3883_388363

/-- The range of a quadratic function on a closed interval --/
theorem quadratic_range_on_unit_interval
  (a b c : ℝ) (ha : a < 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∃ (min max : ℝ), min = c ∧ max = -b^2 / (4 * a) + c ∧
    Set.Icc min max = Set.Icc 0 1 ∩ Set.range f :=
by sorry

end NUMINAMATH_CALUDE_quadratic_range_on_unit_interval_l3883_388363


namespace NUMINAMATH_CALUDE_valid_purchase_options_l3883_388374

/-- Represents the price of an item in kopecks -/
def ItemPrice : ℕ → Prop := λ p => ∃ (a : ℕ), p = 100 * a + 99

/-- The total cost of the purchase in kopecks -/
def TotalCost : ℕ := 20083

/-- Proposition that n is a valid number of items purchased -/
def ValidPurchase (n : ℕ) : Prop :=
  ∃ (p : ℕ), ItemPrice p ∧ n * p = TotalCost

theorem valid_purchase_options :
  ∀ n : ℕ, ValidPurchase n ↔ (n = 17 ∨ n = 117) :=
sorry

end NUMINAMATH_CALUDE_valid_purchase_options_l3883_388374


namespace NUMINAMATH_CALUDE_smallest_cube_box_volume_l3883_388349

def cone_height : ℝ := 20
def cone_base_diameter : ℝ := 18

theorem smallest_cube_box_volume (h : cone_height ≥ cone_base_diameter) :
  let box_side := max cone_height cone_base_diameter
  box_side ^ 3 = 8000 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_box_volume_l3883_388349


namespace NUMINAMATH_CALUDE_sum_of_digits_3125_base6_l3883_388350

/-- Converts a natural number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of digits of 3125 in base 6 equals 15 -/
theorem sum_of_digits_3125_base6 : sumDigits (toBase6 3125) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_3125_base6_l3883_388350


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3883_388390

theorem consecutive_even_numbers_sum (n : ℕ) (sum : ℕ) (start : ℕ) : 
  (sum = (n / 2) * (2 * start + (n - 1) * 2)) →
  (start = 32) →
  (sum = 140) →
  (n = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3883_388390


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3883_388347

theorem polynomial_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^11 + 3*y^10 + 5*y^9 + 3*y^8 + 5*y^7) = 
  15*y^12 - y^11 + 9*y^10 - y^9 + 9*y^8 - 10*y^7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3883_388347


namespace NUMINAMATH_CALUDE_jenny_calculation_l3883_388345

theorem jenny_calculation (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 := by
  sorry

end NUMINAMATH_CALUDE_jenny_calculation_l3883_388345


namespace NUMINAMATH_CALUDE_circle_properties_l3883_388379

-- Define the circle family
def circle_family (t : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x - 2*t*y + t^2 + 4*t + 8 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the tangent line m
def line_m (x y : ℝ) : Prop := y = -1 ∧ x = 2

theorem circle_properties :
  -- Part 1: Centers lie on y = x - 3
  (∀ t : ℝ, t ≠ -1 → ∃ x y : ℝ, circle_family t x y ∧ y = x - 3) ∧
  -- Part 2: Maximum chord length is 2√2
  (∃ max_length : ℝ, max_length = 2 * Real.sqrt 2 ∧
    ∀ t : ℝ, t ≠ -1 →
      ∀ x₁ y₁ x₂ y₂ : ℝ,
        circle_family t x₁ y₁ ∧ line_l x₁ y₁ ∧
        circle_family t x₂ y₂ ∧ line_l x₂ y₂ →
        Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ max_length) ∧
  -- Part 3: Line m is tangent to all circles
  (∀ t : ℝ, t ≠ -1 →
    ∃ x y : ℝ, circle_family t x y ∧ line_m x y ∧
    ∀ x' y' : ℝ, circle_family t x' y' →
      (x' - x)^2 + (y' - y)^2 ≥ 0 ∧
      ((x' - x)^2 + (y' - y)^2 = 0 → x' = x ∧ y' = y)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3883_388379


namespace NUMINAMATH_CALUDE_complex_number_location_l3883_388319

theorem complex_number_location :
  let z : ℂ := (2 - I) / I
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3883_388319


namespace NUMINAMATH_CALUDE_band_member_earnings_l3883_388341

theorem band_member_earnings (attendees : ℕ) (ticket_price : ℝ) (band_share : ℝ) (band_members : ℕ) : 
  attendees = 500 → 
  ticket_price = 30 → 
  band_share = 0.7 → 
  band_members = 4 → 
  (attendees * ticket_price * band_share) / band_members = 2625 := by
sorry

end NUMINAMATH_CALUDE_band_member_earnings_l3883_388341


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l3883_388321

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ∨ (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (-2) 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l3883_388321


namespace NUMINAMATH_CALUDE_geometric_sum_seven_terms_l3883_388364

theorem geometric_sum_seven_terms : 
  let a : ℚ := 1/4  -- first term
  let r : ℚ := 1/4  -- common ratio
  let n : ℕ := 7    -- number of terms
  let S := a * (1 - r^n) / (1 - r)  -- formula for sum of geometric series
  S = 16383/49152 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_seven_terms_l3883_388364


namespace NUMINAMATH_CALUDE_p_minus_q_equals_two_l3883_388316

-- Define an invertible function g
variable (g : ℝ → ℝ)
variable (hg : Function.Injective g)

-- Define p and q based on the given conditions
variable (p q : ℝ)
variable (hp : g p = 3)
variable (hq : g q = 5)

-- State the theorem
theorem p_minus_q_equals_two : p - q = 2 := by
  sorry

end NUMINAMATH_CALUDE_p_minus_q_equals_two_l3883_388316


namespace NUMINAMATH_CALUDE_celebrity_photo_matching_probability_l3883_388384

theorem celebrity_photo_matching_probability :
  ∀ (n : ℕ) (k : ℕ),
    n = 5 →
    k = 2 →
    (Nat.choose n k * k.factorial : ℚ)⁻¹ = 1 / 20 :=
by sorry

end NUMINAMATH_CALUDE_celebrity_photo_matching_probability_l3883_388384


namespace NUMINAMATH_CALUDE_f_sum_zero_a_geq_2_sufficient_not_necessary_l3883_388380

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1) / Real.log 10

-- Define the domain A of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the function g (a is a parameter)
def g (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (1 - a^2 - 2*a*x - x^2)

-- Define the domain B of function g
def B (a : ℝ) : Set ℝ := {x | 1 - a^2 - 2*a*x - x^2 ≥ 0}

-- Statement 1: f(1/2013) + f(-1/2013) = 0
theorem f_sum_zero : f (1/2013) + f (-1/2013) = 0 := by sorry

-- Statement 2: a ≥ 2 is sufficient but not necessary for A ∩ B = ∅
theorem a_geq_2_sufficient_not_necessary :
  (∀ a : ℝ, a ≥ 2 → A ∩ B a = ∅) ∧
  ¬(∀ a : ℝ, A ∩ B a = ∅ → a ≥ 2) := by sorry

end

end NUMINAMATH_CALUDE_f_sum_zero_a_geq_2_sufficient_not_necessary_l3883_388380


namespace NUMINAMATH_CALUDE_beaus_age_proof_l3883_388333

/-- Represents Beau's age today -/
def beaus_age_today : ℕ := 42

/-- Represents the age of Beau's sons today -/
def sons_age_today : ℕ := 16

/-- The number of Beau's sons (triplets) -/
def number_of_sons : ℕ := 3

/-- The number of years ago when the sum of sons' ages equaled Beau's age -/
def years_ago : ℕ := 3

theorem beaus_age_proof :
  (sons_age_today - years_ago) * number_of_sons + years_ago = beaus_age_today :=
by sorry

end NUMINAMATH_CALUDE_beaus_age_proof_l3883_388333


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3883_388329

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (2, x) (1, 2) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3883_388329


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3883_388320

theorem fifteenth_student_age (total_students : Nat) (avg_age : Nat) (group1_size : Nat) (group1_avg : Nat) (group2_size : Nat) (group2_avg : Nat) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 3 →
  group1_avg = 14 →
  group2_size = 11 →
  group2_avg = 16 →
  (total_students * avg_age) - (group1_size * group1_avg + group2_size * group2_avg) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l3883_388320


namespace NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_r_two_satisfies_condition_l3883_388382

theorem largest_r_for_sequence_convergence (r : ℝ) :
  r > 2 →
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, 0 < a n) ∧
    (∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))) ∧
    (¬ ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n) :=
by sorry

theorem r_two_satisfies_condition :
  ∀ (a : ℕ → ℕ), (∀ n : ℕ, 0 < a n) →
    (∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + 2 * a (n + 1))) →
    ∃ M : ℕ, ∀ n ≥ M, a (n + 2) = a n :=
by sorry

end NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_r_two_satisfies_condition_l3883_388382


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3883_388327

/-- A regular nonagon is a 9-sided polygon with all sides equal and all angles equal. -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- Two diagonals intersect if they have a point in common inside the nonagon. -/
def Intersect (n : RegularNonagon) (d1 d2 : Diagonal n) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def Probability (event : Prop) : ℚ := sorry

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  Probability (∃ (d1 d2 : Diagonal n), Intersect n d1 d2) = 14 / 39 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3883_388327


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l3883_388338

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem number_puzzle_solution :
  ∀ (A B C : ℕ),
  (sum_of_digits A = B) →
  (sum_of_digits B = C) →
  (A + B + C = 60) →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l3883_388338


namespace NUMINAMATH_CALUDE_square_difference_equality_l3883_388309

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3883_388309


namespace NUMINAMATH_CALUDE_simplify_fraction_l3883_388369

theorem simplify_fraction : (140 : ℚ) / 210 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3883_388369


namespace NUMINAMATH_CALUDE_expression_simplification_l3883_388383

theorem expression_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a + b ≠ 0) :
  (3 * a^2 + 3 * a * b + 3 * b^2) / (4 * a + 4 * b) *
  (2 * a^2 - 2 * b^2) / (9 * a^3 - 9 * b^3) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3883_388383
