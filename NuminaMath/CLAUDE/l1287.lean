import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_vertex_l1287_128702

/-- The quadratic function f(x) = 2(x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := 1

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-3)^2 + 1 is at (3,1) -/
theorem quadratic_vertex : 
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1287_128702


namespace NUMINAMATH_CALUDE_temperature_at_speed_0_4_l1287_128790

/-- The temperature in degrees Celsius given the speed of sound in meters per second -/
def temperature (v : ℝ) : ℝ := 15 * v^2

/-- Theorem: When the speed of sound is 0.4 m/s, the temperature is 2.4°C -/
theorem temperature_at_speed_0_4 : temperature 0.4 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_temperature_at_speed_0_4_l1287_128790


namespace NUMINAMATH_CALUDE_first_day_is_wednesday_l1287_128782

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that the 22nd day of a month is a Wednesday, 
    prove that the 1st day of that month is also a Wednesday -/
theorem first_day_is_wednesday 
  (h : ∃ (m : List DayInMonth), 
    m.length = 31 ∧ 
    (∃ (d : DayInMonth), d ∈ m ∧ d.day = 22 ∧ d.dayOfWeek = DayOfWeek.Wednesday)) :
  ∃ (m : List DayInMonth),
    m.length = 31 ∧
    (∃ (d : DayInMonth), d ∈ m ∧ d.day = 1 ∧ d.dayOfWeek = DayOfWeek.Wednesday) :=
by sorry

end NUMINAMATH_CALUDE_first_day_is_wednesday_l1287_128782


namespace NUMINAMATH_CALUDE_apple_pie_consumption_l1287_128717

theorem apple_pie_consumption (apples_per_serving : ℝ) (num_guests : ℕ) (num_pies : ℕ) (servings_per_pie : ℕ) :
  apples_per_serving = 1.5 →
  num_guests = 12 →
  num_pies = 3 →
  servings_per_pie = 8 →
  (num_pies * servings_per_pie * apples_per_serving) / num_guests = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_consumption_l1287_128717


namespace NUMINAMATH_CALUDE_average_weight_e_f_l1287_128772

theorem average_weight_e_f (d e f : ℝ) 
  (h1 : (d + e + f) / 3 = 42)
  (h2 : (d + e) / 2 = 35)
  (h3 : e = 26) :
  (e + f) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_average_weight_e_f_l1287_128772


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l1287_128770

theorem power_tower_mod_1000 : 7^(7^(7^7)) ≡ 343 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l1287_128770


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l1287_128780

theorem solve_cubic_equation (n : ℝ) :
  (n - 5) ^ 3 = (1 / 9)⁻¹ ↔ n = 5 + 3 ^ (2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l1287_128780


namespace NUMINAMATH_CALUDE_inequality_proof_l1287_128746

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 < 4*b) (h2 : c^2 < 4*d) : 
  ((a + c)/2)^2 < 4*((b + d)/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1287_128746


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_on_interval_a_range_for_nonnegative_l1287_128753

-- Define the function f(x) = x³ - ax²
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_line_at_one (a : ℝ) (h : f_derivative a 1 = 3) :
  ∃ m b : ℝ, m = 3 ∧ b = -2 ∧ ∀ x : ℝ, (f a x - (f a 1)) = m * (x - 1) := by sorry

theorem max_value_on_interval :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 2 → f 0 x ≤ M := by sorry

theorem a_range_for_nonnegative (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x + x ≥ 0) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_on_interval_a_range_for_nonnegative_l1287_128753


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l1287_128795

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 150 →
    (n - 2) * 180 = n * interior_angle →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l1287_128795


namespace NUMINAMATH_CALUDE_supermarket_spending_l1287_128703

theorem supermarket_spending (total : ℚ) : 
  (1/5 : ℚ) * total + (1/3 : ℚ) * total + (1/10 : ℚ) * total + 11 = total → 
  total = 30 :=
by sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1287_128703


namespace NUMINAMATH_CALUDE_sue_candy_count_l1287_128740

/-- Represents the number of candies each person has -/
structure CandyCount where
  bob : Nat
  mary : Nat
  john : Nat
  sam : Nat
  sue : Nat

/-- The total number of candies for all friends -/
def totalCandies (cc : CandyCount) : Nat :=
  cc.bob + cc.mary + cc.john + cc.sam + cc.sue

theorem sue_candy_count (cc : CandyCount) 
  (h1 : cc.bob = 10)
  (h2 : cc.mary = 5)
  (h3 : cc.john = 5)
  (h4 : cc.sam = 10)
  (h5 : totalCandies cc = 50) :
  cc.sue = 20 := by
  sorry


end NUMINAMATH_CALUDE_sue_candy_count_l1287_128740


namespace NUMINAMATH_CALUDE_max_d_value_l1287_128707

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_form (d e : ℕ) : ℕ := 707330 + d * 1000 + e

theorem max_d_value :
  ∃ (d e : ℕ),
    is_digit d ∧
    is_digit e ∧
    number_form d e % 33 = 0 ∧
    (∀ (d' e' : ℕ), is_digit d' ∧ is_digit e' ∧ number_form d' e' % 33 = 0 → d' ≤ d) ∧
    d = 6 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1287_128707


namespace NUMINAMATH_CALUDE_least_partition_size_formula_l1287_128737

/-- A move that can be applied to a permutation -/
inductive Move
  | MedianFirst : Move  -- If a is the median, replace a,b,c with b,c,a
  | MedianLast : Move   -- If c is the median, replace a,b,c with c,a,b

/-- A permutation of 1, 2, 3, ..., n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The number of inversions in a permutation -/
def inversions (n : ℕ) (σ : Permutation n) : ℕ :=
  sorry

/-- Whether two permutations are obtainable from each other by a sequence of moves -/
def obtainable (n : ℕ) (σ τ : Permutation n) : Prop :=
  sorry

/-- The least number of sets in a partition of all n! permutations -/
def least_partition_size (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: the least number of sets in the partition is n^2 - 3n + 4 -/
theorem least_partition_size_formula (n : ℕ) :
  least_partition_size n = n^2 - 3*n + 4 :=
sorry

end NUMINAMATH_CALUDE_least_partition_size_formula_l1287_128737


namespace NUMINAMATH_CALUDE_work_completion_time_l1287_128766

/-- The time taken to complete a work given the rates of two workers and their working pattern -/
theorem work_completion_time
  (rate_A rate_B : ℝ)  -- Rates at which A and B can complete the work alone
  (days_together : ℝ)  -- Number of days A and B work together
  (h_rate_A : rate_A = 1 / 15)  -- A can complete the work in 15 days
  (h_rate_B : rate_B = 1 / 10)  -- B can complete the work in 10 days
  (h_days_together : days_together = 2)  -- A and B work together for 2 days
  : ∃ (total_days : ℝ), total_days = 12 ∧ 
    rate_A * (total_days - days_together) + (rate_A + rate_B) * days_together = 1 :=
by sorry


end NUMINAMATH_CALUDE_work_completion_time_l1287_128766


namespace NUMINAMATH_CALUDE_domino_tiling_theorem_l1287_128798

/-- Represents a rectangular grid -/
structure Rectangle where
  m : ℕ
  n : ℕ

/-- Represents a domino tile -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Represents a tiling of a rectangle with dominoes -/
def Tiling (r : Rectangle) (d : Domino) := Unit

/-- Predicate to check if a tiling has no straight cuts -/
def has_no_straight_cuts (t : Tiling r d) : Prop := sorry

theorem domino_tiling_theorem :
  /- Part a -/
  (∀ (r : Rectangle) (d : Domino), r.m = 6 ∧ r.n = 6 ∧ d.length = 1 ∧ d.width = 2 →
    ¬ ∃ (t : Tiling r d), has_no_straight_cuts t) ∧
  /- Part b -/
  (∀ (r : Rectangle) (d : Domino), r.m > 6 ∧ r.n > 6 ∧ (r.m * r.n) % 2 = 0 ∧ d.length = 1 ∧ d.width = 2 →
    ∃ (t : Tiling r d), has_no_straight_cuts t) ∧
  /- Part c -/
  (∃ (r : Rectangle) (d : Domino) (t : Tiling r d), r.m = 6 ∧ r.n = 8 ∧ d.length = 1 ∧ d.width = 2 ∧
    has_no_straight_cuts t) :=
by sorry

end NUMINAMATH_CALUDE_domino_tiling_theorem_l1287_128798


namespace NUMINAMATH_CALUDE_factoring_expression_l1287_128791

theorem factoring_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1287_128791


namespace NUMINAMATH_CALUDE_function_value_at_seven_l1287_128725

/-- Given a function f(x) = ax^7 + bx^3 + cx - 5 where a, b, c are constants,
    if f(-7) = 7, then f(7) = -17 -/
theorem function_value_at_seven 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^7 + b * x^3 + c * x - 5) 
  (h2 : f (-7) = 7) : 
  f 7 = -17 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_seven_l1287_128725


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1287_128788

-- Expression 1
theorem simplify_expression_1 (a b : ℝ) :
  -2 * a^2 * b - 3 * a * b^2 + 3 * a^2 * b - 4 * a * b^2 = a^2 * b - 7 * a * b^2 := by
  sorry

-- Expression 2
theorem simplify_expression_2 (x y z : ℝ) :
  2 * (x * y * z - 3 * x) + 5 * (2 * x - 3 * x * y * z) = 4 * x - 13 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1287_128788


namespace NUMINAMATH_CALUDE_unique_square_friendly_l1287_128760

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

/-- Definition of a square-friendly integer -/
def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18*m + c)

/-- Theorem: 81 is the only square-friendly integer -/
theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 :=
sorry

end NUMINAMATH_CALUDE_unique_square_friendly_l1287_128760


namespace NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l1287_128715

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) :
  |p - q| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l1287_128715


namespace NUMINAMATH_CALUDE_max_abs_z_l1287_128705

/-- Given a complex number z satisfying |z - 8| + |z + 6i| = 10, 
    the maximum value of |z| is 8. -/
theorem max_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 6*I) = 10) : 
  ∃ (w : ℂ), Complex.abs (w - 8) + Complex.abs (w + 6*I) = 10 ∧ 
             ∀ (u : ℂ), Complex.abs (u - 8) + Complex.abs (u + 6*I) = 10 → 
             Complex.abs u ≤ Complex.abs w ∧
             Complex.abs w = 8 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_l1287_128705


namespace NUMINAMATH_CALUDE_log_equation_equivalence_l1287_128708

theorem log_equation_equivalence (x : ℝ) :
  (∀ y : ℝ, y > 0 → ∃ z : ℝ, Real.exp z = y) →  -- This ensures logarithms are defined for positive reals
  (x > (3/2) ↔ (Real.log (x+5) + Real.log (2*x-3) = Real.log (2*x^2 + x - 15))) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_equivalence_l1287_128708


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1287_128784

theorem perpendicular_slope (x y : ℝ) :
  let original_line := {(x, y) | 5 * x - 2 * y = 10}
  let original_slope : ℝ := 5 / 2
  let perpendicular_slope : ℝ := -1 / original_slope
  perpendicular_slope = -2 / 5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1287_128784


namespace NUMINAMATH_CALUDE_munchausen_claim_correct_l1287_128723

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem munchausen_claim_correct : 
  ∃ (a b : ℕ), 
    (10^9 ≤ a ∧ a < 10^10) ∧ 
    (10^9 ≤ b ∧ b < 10^10) ∧ 
    a ≠ b ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    a + sumOfDigits (a^2) = b + sumOfDigits (b^2) := by
  sorry

end NUMINAMATH_CALUDE_munchausen_claim_correct_l1287_128723


namespace NUMINAMATH_CALUDE_multiple_of_six_squared_gt_200_lt_30_l1287_128713

theorem multiple_of_six_squared_gt_200_lt_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 200)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_six_squared_gt_200_lt_30_l1287_128713


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l1287_128704

theorem alcohol_mixture_problem (A W : ℝ) :
  A / W = 2 / 5 →
  A / (W + 10) = 2 / 7 →
  A = 10 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l1287_128704


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l1287_128789

theorem sin_thirteen_pi_sixths : Real.sin (13 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l1287_128789


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l1287_128729

/-- Given a circle with center O and radius R, and a segment of length a,
    the locus of midpoints of all chords of length a is a circle concentric
    to the original circle with radius √(R² - a²/4). -/
theorem locus_of_midpoints (O : ℝ × ℝ) (R a : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < 2*R) :
  ∃ (C : Set (ℝ × ℝ)),
    C = {P | ∃ (A B : ℝ × ℝ),
      (A.1 - O.1)^2 + (A.2 - O.2)^2 = R^2 ∧
      (B.1 - O.1)^2 + (B.2 - O.2)^2 = R^2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
      P = ((A.1 + B.1)/2, (A.2 + B.2)/2)} ∧
    C = {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2 - a^2/4} :=
by
  sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l1287_128729


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1287_128792

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 12 = 5 / (x - 12) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1287_128792


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1287_128777

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1287_128777


namespace NUMINAMATH_CALUDE_mitch_earnings_l1287_128743

/-- Represents Mitch's work schedule and earnings --/
structure MitchSchedule where
  weekday_hours : ℕ
  weekend_hours : ℕ
  weekday_rate : ℕ
  weekend_rate : ℕ

/-- Calculates Mitch's weekly earnings --/
def weekly_earnings (schedule : MitchSchedule) : ℕ :=
  (schedule.weekday_hours * 5 * schedule.weekday_rate) +
  (schedule.weekend_hours * 2 * schedule.weekend_rate)

/-- Theorem stating Mitch's weekly earnings --/
theorem mitch_earnings : 
  let schedule := MitchSchedule.mk 5 3 3 6
  weekly_earnings schedule = 111 := by
  sorry


end NUMINAMATH_CALUDE_mitch_earnings_l1287_128743


namespace NUMINAMATH_CALUDE_george_remaining_eggs_l1287_128794

/-- Calculates the remaining number of eggs given the initial inventory and sold amount. -/
def remaining_eggs (cases : ℕ) (boxes_per_case : ℕ) (eggs_per_box : ℕ) (boxes_sold : ℕ) : ℕ :=
  cases * boxes_per_case * eggs_per_box - boxes_sold * eggs_per_box

/-- Proves that George has 648 eggs remaining after selling 3 boxes. -/
theorem george_remaining_eggs :
  remaining_eggs 7 12 8 3 = 648 := by
  sorry

end NUMINAMATH_CALUDE_george_remaining_eggs_l1287_128794


namespace NUMINAMATH_CALUDE_box_length_proof_l1287_128720

/-- Proves that the length of a box with given dimensions and cube requirements is 10 cm -/
theorem box_length_proof (width : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h_width : width = 13)
  (h_height : height = 5)
  (h_cube_volume : cube_volume = 5)
  (h_min_cubes : min_cubes = 130) :
  (min_cubes : ℝ) * cube_volume / (width * height) = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_length_proof_l1287_128720


namespace NUMINAMATH_CALUDE_rainfall_ratio_l1287_128709

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    prove the ratio of rainfall in the second week to the first week. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) 
    (h1 : total = 35)
    (h2 : second_week = 21) :
    second_week / (total - second_week) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_l1287_128709


namespace NUMINAMATH_CALUDE_largest_integer_below_sqrt_two_l1287_128734

theorem largest_integer_below_sqrt_two :
  ∀ n : ℕ, n > 0 ∧ n < Real.sqrt 2 → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_below_sqrt_two_l1287_128734


namespace NUMINAMATH_CALUDE_polynomial_coefficient_properties_l1287_128745

theorem polynomial_coefficient_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_properties_l1287_128745


namespace NUMINAMATH_CALUDE_teacher_age_l1287_128779

theorem teacher_age (num_students : ℕ) (avg_age_students : ℕ) (avg_increase : ℕ) : 
  num_students = 22 →
  avg_age_students = 21 →
  avg_increase = 1 →
  (num_students * avg_age_students + 44) / (num_students + 1) = avg_age_students + avg_increase :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l1287_128779


namespace NUMINAMATH_CALUDE_amys_garden_space_l1287_128739

/-- Calculates the total square feet of growing space for Amy's garden beds -/
theorem amys_garden_space : 
  let small_bed_length : ℝ := 3
  let small_bed_width : ℝ := 3
  let large_bed_length : ℝ := 4
  let large_bed_width : ℝ := 3
  let num_small_beds : ℕ := 2
  let num_large_beds : ℕ := 2
  
  let small_bed_area := small_bed_length * small_bed_width
  let large_bed_area := large_bed_length * large_bed_width
  let total_area := (num_small_beds : ℝ) * small_bed_area + (num_large_beds : ℝ) * large_bed_area
  
  total_area = 42 := by sorry

end NUMINAMATH_CALUDE_amys_garden_space_l1287_128739


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1287_128738

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 14

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := 6

/-- The total number of people who can ride the Ferris wheel at the same time -/
def total_people : ℕ := num_seats * people_per_seat

theorem ferris_wheel_capacity : total_people = 84 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1287_128738


namespace NUMINAMATH_CALUDE_theodore_sturgeon_collection_hardcovers_l1287_128796

/-- Given a collection of books with two price options and a total cost,
    calculate the number of books purchased at the higher price. -/
def hardcover_count (total_volumes : ℕ) (paperback_price hardcover_price : ℕ) (total_cost : ℕ) : ℕ :=
  let h := (2 * total_cost - paperback_price * total_volumes) / (2 * (hardcover_price - paperback_price))
  h

/-- Theorem stating that given the specific conditions of the problem,
    the number of hardcover books purchased is 6. -/
theorem theodore_sturgeon_collection_hardcovers :
  hardcover_count 12 15 30 270 = 6 := by
  sorry

end NUMINAMATH_CALUDE_theodore_sturgeon_collection_hardcovers_l1287_128796


namespace NUMINAMATH_CALUDE_roof_area_theorem_l1287_128742

/-- Represents the dimensions and area of a rectangular roof. -/
structure RectangularRoof where
  width : ℚ
  length : ℚ
  area : ℚ

/-- Calculates the area of a rectangular roof given its width and length. -/
def calculateArea (w : ℚ) (l : ℚ) : ℚ := w * l

/-- Theorem: The area of a rectangular roof with specific proportions is 455 1/9 square feet. -/
theorem roof_area_theorem (roof : RectangularRoof) : 
  roof.length = 3 * roof.width → 
  roof.length - roof.width = 32 → 
  roof.area = calculateArea roof.width roof.length → 
  roof.area = 455 + 1/9 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_theorem_l1287_128742


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1287_128776

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def players_to_select : ℕ := 7
def quadruplets_to_select : ℕ := 3

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_to_select) *
  (Nat.choose (total_players - quadruplets) (players_to_select - quadruplets_to_select)) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1287_128776


namespace NUMINAMATH_CALUDE_blue_eyed_students_l1287_128701

theorem blue_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 40 →
  both = 8 →
  neither = 5 →
  ∃ (blue : ℕ), 
    blue + (3 * blue - both) + both + neither = total ∧
    blue = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_eyed_students_l1287_128701


namespace NUMINAMATH_CALUDE_least_N_for_probability_condition_l1287_128754

def P (N : ℕ) : ℚ :=
  (⌊(2 * N : ℚ) / 5⌋ + (N - ⌈(3 * N : ℚ) / 5⌉)) / (N + 1 : ℚ)

theorem least_N_for_probability_condition :
  (∀ k : ℕ, k % 5 = 0 ∧ 0 < k ∧ k < 480 → P k ≥ 321/400) ∧
  P 480 < 321/400 := by
  sorry

end NUMINAMATH_CALUDE_least_N_for_probability_condition_l1287_128754


namespace NUMINAMATH_CALUDE_arithmetic_geq_geometric_l1287_128750

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geq_geometric
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h1 : a 1 = b 1)
  (h1_pos : a 1 > 0)
  (hn : a n = b n)
  (hn_pos : a n > 0)
  (n : ℕ)
  (hn_gt_1 : n > 1) :
  ∀ m : ℕ, 1 < m → m < n → a m ≥ b m :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geq_geometric_l1287_128750


namespace NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l1287_128747

theorem x_equation_implies_polynomial_value (x : ℝ) (h : x + 1/x = Real.sqrt 7) :
  x^12 - 8*x^8 + x^4 = 1365 := by
sorry

end NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l1287_128747


namespace NUMINAMATH_CALUDE_coin_trick_theorem_l1287_128762

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a row of 27 coins -/
def CoinRow := Vector CoinState 27

/-- Represents a selection of 5 coins from the row -/
def CoinSelection := Vector Nat 5

/-- Function to check if all coins in a selection are facing the same way -/
def allSameFacing (row : CoinRow) (selection : CoinSelection) : Prop :=
  ∀ i j, i < 5 → j < 5 → row.get (selection.get i) = row.get (selection.get j)

/-- The main theorem stating that it's always possible to select 10 coins facing the same way,
    such that 5 of them can determine the state of the other 5 -/
theorem coin_trick_theorem (row : CoinRow) :
  ∃ (selection1 selection2 : CoinSelection),
    allSameFacing row selection1 ∧
    allSameFacing row selection2 ∧
    (∀ i, i < 5 → selection1.get i ≠ selection2.get i) ∧
    (∃ f : CoinSelection → CoinSelection, f selection1 = selection2) :=
sorry

end NUMINAMATH_CALUDE_coin_trick_theorem_l1287_128762


namespace NUMINAMATH_CALUDE_anthonys_pets_l1287_128751

theorem anthonys_pets (initial_pets : ℕ) : 
  (initial_pets - 6 : ℚ) * (4/5) = 8 → initial_pets = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_anthonys_pets_l1287_128751


namespace NUMINAMATH_CALUDE_trader_profit_l1287_128714

theorem trader_profit (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let discount_rate : ℝ := 0.4
  let increase_rate : ℝ := 0.8
  let purchase_price : ℝ := original_price * (1 - discount_rate)
  let selling_price : ℝ := purchase_price * (1 + increase_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_trader_profit_l1287_128714


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1287_128773

theorem geometric_sequence_minimum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ r > 0, ∀ n, a (n + 1) = a n * r) (h_prod : a 5 * a 6 = 16) :
  (∀ x, a 2 + a 9 ≥ x) → x ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1287_128773


namespace NUMINAMATH_CALUDE_rectangle_area_l1287_128774

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * (1 + ratio) = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1287_128774


namespace NUMINAMATH_CALUDE_find_number_l1287_128771

theorem find_number : ∃ x : ℤ, x - 263 + 419 = 725 ∧ x = 569 := by sorry

end NUMINAMATH_CALUDE_find_number_l1287_128771


namespace NUMINAMATH_CALUDE_unique_solution_at_two_no_unique_solution_above_two_no_unique_solution_below_two_max_a_for_unique_solution_l1287_128726

/-- The system of equations has a unique solution when a = 2 -/
theorem unique_solution_at_two (x y a : ℝ) : 
  (y = 1 - Real.sqrt x ∧ 
   a - 2 * (a - y)^2 = Real.sqrt x ∧ 
   ∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) 
  → a = 2 := by
  sorry

/-- For any a > 2, the system of equations does not have a unique solution -/
theorem no_unique_solution_above_two (a : ℝ) :
  a > 2 → ¬(∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) := by
  sorry

/-- For any 0 ≤ a < 2, the system of equations does not have a unique solution -/
theorem no_unique_solution_below_two (a : ℝ) :
  0 ≤ a ∧ a < 2 → ¬(∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) := by
  sorry

/-- The maximum value of a for which the system has a unique solution is 2 -/
theorem max_a_for_unique_solution :
  ∃ (a : ℝ), (∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ a - 2 * (a - y)^2 = Real.sqrt x) ∧
  ∀ (b : ℝ), (∃! (x y : ℝ), y = 1 - Real.sqrt x ∧ b - 2 * (b - y)^2 = Real.sqrt x) → b ≤ a := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_at_two_no_unique_solution_above_two_no_unique_solution_below_two_max_a_for_unique_solution_l1287_128726


namespace NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l1287_128781

theorem abs_one_point_five_minus_sqrt_two : |1.5 - Real.sqrt 2| = 1.5 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l1287_128781


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l1287_128755

theorem quadratic_root_proof : ∃ x : ℝ, x^2 - 4*x*Real.sqrt 2 + 8 = 0 ∧ x = 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l1287_128755


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1287_128783

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.82 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/82 - 1) * 100 := by
  sorry

#eval (100/82 - 1) * 100 -- This will output approximately 21.95

end NUMINAMATH_CALUDE_profit_percent_calculation_l1287_128783


namespace NUMINAMATH_CALUDE_expand_product_l1287_128716

theorem expand_product (x : ℝ) : 3 * (x - 3) * (x + 5) = 3 * x^2 + 6 * x - 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1287_128716


namespace NUMINAMATH_CALUDE_complex_number_not_in_fourth_quadrant_l1287_128728

theorem complex_number_not_in_fourth_quadrant :
  ∀ (m : ℝ) (z : ℂ), (1 - Complex.I) * z = m + Complex.I →
    ¬(Complex.re z > 0 ∧ Complex.im z < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_not_in_fourth_quadrant_l1287_128728


namespace NUMINAMATH_CALUDE_tens_digit_of_9_power_2023_l1287_128765

theorem tens_digit_of_9_power_2023 (h1 : 9^10 % 50 = 1) (h2 : 9^3 % 50 = 29) :
  (9^2023 / 10) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_power_2023_l1287_128765


namespace NUMINAMATH_CALUDE_surface_integral_I_value_surface_integral_J_value_surface_integral_K_value_l1287_128731

-- Define the surface integrals
def surface_integral_I (a : ℝ) : ℝ := sorry

def surface_integral_J : ℝ := sorry

def surface_integral_K : ℝ := sorry

-- State the theorems to be proved
theorem surface_integral_I_value (a : ℝ) (h : a > 0) : 
  surface_integral_I a = (4 * Real.pi / 5) * a^(5/2) := by sorry

theorem surface_integral_J_value : 
  surface_integral_J = (8 * Real.pi / 3) - (4 / 15) := by sorry

theorem surface_integral_K_value : 
  surface_integral_K = -1 / 6 := by sorry

end NUMINAMATH_CALUDE_surface_integral_I_value_surface_integral_J_value_surface_integral_K_value_l1287_128731


namespace NUMINAMATH_CALUDE_transformation_converts_curve_l1287_128741

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = 2 * Real.sin (3 * x)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop := y' = Real.sin x'

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop := x' = 3 * x ∧ y' = (1/2) * y

-- Theorem statement
theorem transformation_converts_curve :
  ∀ x y x' y' : ℝ,
  original_curve x y →
  transformation x y x' y' →
  transformed_curve x' y' :=
sorry

end NUMINAMATH_CALUDE_transformation_converts_curve_l1287_128741


namespace NUMINAMATH_CALUDE_orange_boxes_pigeonhole_l1287_128759

theorem orange_boxes_pigeonhole (total_boxes : ℕ) (min_oranges max_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ n : ℕ, n ≥ 5 ∧ ∃ k : ℕ, k ≥ min_oranges ∧ k ≤ max_oranges ∧ 
    (∃ boxes : Finset (Fin total_boxes), boxes.card = n ∧ 
      ∀ i ∈ boxes, ∃ f : Fin total_boxes → ℕ, f i = k) :=
by sorry

end NUMINAMATH_CALUDE_orange_boxes_pigeonhole_l1287_128759


namespace NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l1287_128768

theorem at_least_one_less_than_or_equal_to_one
  (x y z : ℝ)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (pos_z : 0 < z)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l1287_128768


namespace NUMINAMATH_CALUDE_bottle_production_rate_l1287_128752

/-- Given that 5 identical machines can produce 900 bottles in 4 minutes at a constant rate,
    prove that 6 such machines can produce 270 bottles per minute. -/
theorem bottle_production_rate (rate : ℕ → ℕ → ℕ) : 
  (rate 5 4 = 900) → (rate 6 1 = 270) :=
by
  sorry


end NUMINAMATH_CALUDE_bottle_production_rate_l1287_128752


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1287_128710

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (-1, x)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  parallel vector_a (vector_b x) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1287_128710


namespace NUMINAMATH_CALUDE_common_ratio_satisfies_cubic_cubic_solution_approx_l1287_128758

/-- A geometric progression with positive terms where each term is the sum of the next three terms -/
structure GeometricProgressionWithSumProperty where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a geometric progression with the sum property satisfies a cubic equation -/
theorem common_ratio_satisfies_cubic (gp : GeometricProgressionWithSumProperty) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 :=
sorry

/-- The positive real solution to the cubic equation x³ + x² + x - 1 = 0 is approximately 0.5437 -/
theorem cubic_solution_approx :
  ∃ x : ℝ, x > 0 ∧ x^3 + x^2 + x - 1 = 0 ∧ abs (x - 0.5437) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_satisfies_cubic_cubic_solution_approx_l1287_128758


namespace NUMINAMATH_CALUDE_custom_op_theorem_l1287_128700

/-- Definition of the custom operation ⊕ -/
def custom_op (m n : ℝ) : ℝ := m * n * (m - n)

/-- Theorem stating that (a + b) ⊕ a = a^2 * b + a * b^2 -/
theorem custom_op_theorem (a b : ℝ) : custom_op (a + b) a = a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l1287_128700


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l1287_128706

theorem midpoint_x_coordinate_sum (a b c : ℝ) :
  let S := a + b + c
  let midpoint1 := (a + b) / 2
  let midpoint2 := (a + c) / 2
  let midpoint3 := (b + c) / 2
  midpoint1 + midpoint2 + midpoint3 = S := by sorry

end NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l1287_128706


namespace NUMINAMATH_CALUDE_star_op_value_l1287_128763

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

-- Theorem statement
theorem star_op_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + b = 15 → a * b = 56 → star_op a b = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_star_op_value_l1287_128763


namespace NUMINAMATH_CALUDE_mike_changed_tires_on_12_motorcycles_l1287_128719

/-- The number of motorcycles Mike changed tires on -/
def num_motorcycles (total_tires num_cars tires_per_car tires_per_motorcycle : ℕ) : ℕ :=
  (total_tires - num_cars * tires_per_car) / tires_per_motorcycle

theorem mike_changed_tires_on_12_motorcycles :
  num_motorcycles 64 10 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mike_changed_tires_on_12_motorcycles_l1287_128719


namespace NUMINAMATH_CALUDE_geometry_test_passing_l1287_128721

theorem geometry_test_passing (total_problems : Nat) (passing_percentage : Rat) 
  (hp : total_problems = 50)
  (hq : passing_percentage = 85 / 100) : 
  (max_missed_problems : Nat) → 
  (max_missed_problems = total_problems - Int.ceil (passing_percentage * total_problems)) ∧
  max_missed_problems = 7 := by
  sorry

end NUMINAMATH_CALUDE_geometry_test_passing_l1287_128721


namespace NUMINAMATH_CALUDE_mother_age_twice_alex_age_l1287_128778

/-- The year when Alex's mother's age will be twice his age -/
def target_year : ℕ := 2025

/-- Alex's birth year -/
def alex_birth_year : ℕ := 1997

/-- The year of Alex's 7th birthday -/
def seventh_birthday_year : ℕ := 2004

/-- Alex's age on his 7th birthday -/
def alex_age_seventh_birthday : ℕ := 7

/-- Alex's mother's age on Alex's 7th birthday -/
def mother_age_seventh_birthday : ℕ := 35

theorem mother_age_twice_alex_age :
  (target_year - seventh_birthday_year) + alex_age_seventh_birthday = 
  (target_year - seventh_birthday_year + mother_age_seventh_birthday) / 2 ∧
  mother_age_seventh_birthday = 5 * alex_age_seventh_birthday ∧
  seventh_birthday_year - alex_birth_year = alex_age_seventh_birthday :=
by sorry

end NUMINAMATH_CALUDE_mother_age_twice_alex_age_l1287_128778


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1287_128764

/-- Given a group of 10 persons, if replacing one person with a new person
    weighing 100 kg increases the average weight by 3.5 kg,
    then the weight of the replaced person is 65 kg. -/
theorem replaced_person_weight
  (n : ℕ) (initial_average : ℝ) (new_person_weight : ℝ) (average_increase : ℝ) :
  n = 10 →
  new_person_weight = 100 →
  average_increase = 3.5 →
  initial_average + average_increase = (n * initial_average - replaced_weight + new_person_weight) / n →
  replaced_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l1287_128764


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1287_128767

theorem opposite_of_2023 : 
  ∃ x : ℤ, (x + 2023 = 0) ∧ (x = -2023) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1287_128767


namespace NUMINAMATH_CALUDE_C_div_D_eq_17_l1287_128757

noncomputable def C : ℝ := ∑' n, if n % 4 ≠ 0 ∧ n % 2 = 0 then (-1)^((n/2) % 2 + 1) / n^2 else 0

noncomputable def D : ℝ := ∑' n, if n % 4 = 0 then (-1)^(n/4 + 1) / n^2 else 0

theorem C_div_D_eq_17 : C / D = 17 := by sorry

end NUMINAMATH_CALUDE_C_div_D_eq_17_l1287_128757


namespace NUMINAMATH_CALUDE_arc_length_of_inscribed_pentagon_l1287_128735

-- Define the circle radius
def circle_radius : ℝ := 5

-- Define the number of sides in a regular pentagon
def pentagon_sides : ℕ := 5

-- Theorem statement
theorem arc_length_of_inscribed_pentagon (π : ℝ) :
  let circumference := 2 * π * circle_radius
  let arc_length := circumference / pentagon_sides
  arc_length = 2 * π := by sorry

end NUMINAMATH_CALUDE_arc_length_of_inscribed_pentagon_l1287_128735


namespace NUMINAMATH_CALUDE_factorization_equality_l1287_128730

theorem factorization_equality (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1287_128730


namespace NUMINAMATH_CALUDE_max_b_rectangular_prism_l1287_128785

theorem max_b_rectangular_prism (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_rectangular_prism_l1287_128785


namespace NUMINAMATH_CALUDE_birds_theorem_l1287_128736

def birds_problem (grey_birds : ℕ) (white_birds : ℕ) : Prop :=
  white_birds = grey_birds + 6 ∧
  grey_birds = 40 ∧
  (grey_birds / 2 + white_birds = 66)

theorem birds_theorem :
  ∃ (grey_birds white_birds : ℕ), birds_problem grey_birds white_birds :=
sorry

end NUMINAMATH_CALUDE_birds_theorem_l1287_128736


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l1287_128769

theorem largest_stamps_per_page : Nat.gcd (Nat.gcd 1020 1275) 1350 = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l1287_128769


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l1287_128733

/-- An arithmetic sequence is represented by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- Get the nth term of an arithmetic sequence. -/
def ArithmeticSequence.nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

theorem fifth_term_is_five
  (seq : ArithmeticSequence)
  (h : seq.nth_term 2 + seq.nth_term 4 = 10) :
  seq.nth_term 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l1287_128733


namespace NUMINAMATH_CALUDE_four_people_permutations_l1287_128756

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 4 distinct individuals -/
def num_people : ℕ := 4

/-- Theorem: The number of ways 4 people can stand in a line is 24 -/
theorem four_people_permutations : permutations num_people = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_people_permutations_l1287_128756


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_ratio_l1287_128712

/-- Given a function f(x) = 2m*sin(x) - n*cos(x), if x = π/3 is an axis of symmetry
    for the graph of f(x), then n/m = -2√3/3 -/
theorem symmetry_axis_implies_ratio (m n : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, 2 * m * Real.sin x - n * Real.cos x =
    2 * m * Real.sin (2 * π / 3 - x) - n * Real.cos (2 * π / 3 - x)) →
  n / m = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_ratio_l1287_128712


namespace NUMINAMATH_CALUDE_orchestra_overlap_l1287_128793

theorem orchestra_overlap (total : ℕ) (violin : ℕ) (keyboard : ℕ) (neither : ℕ) : 
  total = 42 → violin = 25 → keyboard = 22 → neither = 3 →
  violin + keyboard - (total - neither) = 8 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_overlap_l1287_128793


namespace NUMINAMATH_CALUDE_total_supervisors_is_21_l1287_128748

/-- The number of buses used for the field trip. -/
def num_buses : ℕ := 7

/-- The number of adult supervisors per bus. -/
def supervisors_per_bus : ℕ := 3

/-- The total number of supervisors for the field trip. -/
def total_supervisors : ℕ := num_buses * supervisors_per_bus

/-- Theorem stating that the total number of supervisors is 21. -/
theorem total_supervisors_is_21 : total_supervisors = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_supervisors_is_21_l1287_128748


namespace NUMINAMATH_CALUDE_line_intersection_with_y_axis_l1287_128727

/-- Given a line passing through points (3, 10) and (-7, -6), 
    prove that its intersection with the y-axis is the point (0, 5.2) -/
theorem line_intersection_with_y_axis :
  let p₁ : ℝ × ℝ := (3, 10)
  let p₂ : ℝ × ℝ := (-7, -6)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  let y_intercept : ℝ := line 0
  (0, y_intercept) = (0, 5.2) := by sorry

end NUMINAMATH_CALUDE_line_intersection_with_y_axis_l1287_128727


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l1287_128732

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2*x^2 - 2 = 2*(x+1)*(x-1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l1287_128732


namespace NUMINAMATH_CALUDE_set_operations_l1287_128786

def A : Set ℝ := {x | 2 < x ∧ x < 6}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m}

theorem set_operations (m : ℝ) :
  (m = 2 → A ∪ B m = A) ∧
  (B m ⊆ A ↔ m ≤ 3) ∧
  (B m ≠ ∅ ∧ A ∩ B m = ∅ ↔ m ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l1287_128786


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l1287_128718

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1) : 
  Complex.abs (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l1287_128718


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1287_128724

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity
  (m n : Line) (α β : Plane)
  (diff_lines : m ≠ n)
  (diff_planes : α ≠ β)
  (m_parallel_n : parallel m n)
  (n_perp_β : perpendicular n β)
  (m_subset_α : subset m α) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1287_128724


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l1287_128797

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 156 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l1287_128797


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l1287_128744

theorem unique_solution_power_equation :
  ∃! (n m : ℕ), n > 0 ∧ m > 0 ∧ n^5 + n^4 = 7^m - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l1287_128744


namespace NUMINAMATH_CALUDE_group_purchase_equations_l1287_128787

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  x : ℕ  -- number of people
  y : ℕ  -- price of the item

/-- Defines the conditions of the group purchase -/
def validGroupPurchase (gp : GroupPurchase) : Prop :=
  (9 * gp.x - gp.y = 4) ∧ (gp.y - 6 * gp.x = 5)

/-- Theorem stating that the given system of equations correctly represents the group purchase scenario -/
theorem group_purchase_equations (gp : GroupPurchase) : 
  validGroupPurchase gp ↔ (9 * gp.x - gp.y = 4 ∧ gp.y - 6 * gp.x = 5) :=
sorry

end NUMINAMATH_CALUDE_group_purchase_equations_l1287_128787


namespace NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l1287_128761

theorem sum_of_square_roots_inequality (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_inequality_l1287_128761


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l1287_128749

theorem triangle_angle_difference (A B C : ℝ) : 
  A = 24 →
  B = 5 * A →
  A + B + C = 180 →
  C - A = 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l1287_128749


namespace NUMINAMATH_CALUDE_distance_proof_l1287_128711

def point : ℝ × ℝ × ℝ := (2, 1, -5)

def line_point : ℝ × ℝ × ℝ := (4, -3, 2)
def line_direction : ℝ × ℝ × ℝ := (-1, 4, 3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_proof : 
  distance_to_line point line_point line_direction = Real.sqrt (34489 / 676) := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l1287_128711


namespace NUMINAMATH_CALUDE_event_B_more_likely_l1287_128775

/-- Represents the number of sides on a fair die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 3

/-- Probability of event A: some number appears at least twice in three throws -/
def probA : ℚ := 4 / 9

/-- Probability of event B: three different numbers appear in three throws -/
def probB : ℚ := 5 / 9

/-- Theorem stating that event B is more likely than event A -/
theorem event_B_more_likely : probB > probA := by
  sorry

end NUMINAMATH_CALUDE_event_B_more_likely_l1287_128775


namespace NUMINAMATH_CALUDE_pencil_cost_calculation_l1287_128799

/-- The original cost of a pencil before discount -/
def original_cost : ℝ := 4.00

/-- The discount applied to the pencil -/
def discount : ℝ := 0.63

/-- The final price of the pencil after discount -/
def final_price : ℝ := 3.37

/-- Theorem stating that the original cost minus the discount equals the final price -/
theorem pencil_cost_calculation : original_cost - discount = final_price := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_calculation_l1287_128799


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1287_128722

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1287_128722
