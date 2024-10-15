import Mathlib

namespace NUMINAMATH_CALUDE_a_gt_b_iff_f_a_gt_f_b_l435_43546

theorem a_gt_b_iff_f_a_gt_f_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a + Real.log a > b + Real.log b :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_iff_f_a_gt_f_b_l435_43546


namespace NUMINAMATH_CALUDE_expression_evaluation_l435_43523

theorem expression_evaluation (b : ℚ) (h : b = 4/3) :
  (3 * b^2 - 14 * b + 5) * (3 * b - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l435_43523


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l435_43560

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n : ℕ := (l - a) / d + 1
  n * (a + l) / 2

/-- Theorem: The sum of the arithmetic sequence with first term 2, last term 102, and common difference 5 is 1092 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum 2 102 5 = 1092 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l435_43560


namespace NUMINAMATH_CALUDE_line_parameterization_l435_43582

/-- The line y = 5x - 7 is parameterized by (x, y) = (r, 2) + t(3, k). 
    This theorem proves that r = 9/5 and k = 15. -/
theorem line_parameterization (x y r k t : ℝ) : 
  y = 5 * x - 7 ∧ 
  x = r + 3 * t ∧ 
  y = 2 + k * t → 
  r = 9 / 5 ∧ k = 15 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l435_43582


namespace NUMINAMATH_CALUDE_journey_time_theorem_l435_43514

/-- Represents the time and distance relationship for a journey to the supermarket -/
structure JourneyTime where
  bike_speed : ℝ
  walk_speed : ℝ
  total_distance : ℝ

/-- The journey time satisfies the given conditions -/
def satisfies_conditions (j : JourneyTime) : Prop :=
  j.bike_speed * 12 + j.walk_speed * 20 = j.total_distance ∧
  j.bike_speed * 8 + j.walk_speed * 36 = j.total_distance

/-- The theorem to be proved -/
theorem journey_time_theorem (j : JourneyTime) (h : satisfies_conditions j) :
  (j.total_distance - j.bike_speed * 2) / j.walk_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_theorem_l435_43514


namespace NUMINAMATH_CALUDE_expression_evaluation_l435_43572

theorem expression_evaluation : (50 - (2210 - 251)) + (2210 - (251 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l435_43572


namespace NUMINAMATH_CALUDE_equation_solution_l435_43528

theorem equation_solution : ∃! x : ℝ, (2 / (x - 3) = 3 / x) ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l435_43528


namespace NUMINAMATH_CALUDE_smaller_number_of_sum_and_product_l435_43536

theorem smaller_number_of_sum_and_product (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) :
  min x y = 3 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_of_sum_and_product_l435_43536


namespace NUMINAMATH_CALUDE_min_value_expression_l435_43521

theorem min_value_expression (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  1 / (2*x + y)^2 + 4 / (x - 2*y)^2 ≥ 3/5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l435_43521


namespace NUMINAMATH_CALUDE_square_diff_divided_by_three_l435_43574

theorem square_diff_divided_by_three : (123^2 - 120^2) / 3 = 243 := by sorry

end NUMINAMATH_CALUDE_square_diff_divided_by_three_l435_43574


namespace NUMINAMATH_CALUDE_calculation_proof_l435_43567

theorem calculation_proof :
  (4.8 * (3.5 - 2.1) / 7 = 0.96) ∧
  (18.75 - 0.23 * 2 - 4.54 = 13.75) ∧
  (0.9 + 99 * 0.9 = 90) ∧
  (4 / 0.8 - 0.8 / 4 = 4.8) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l435_43567


namespace NUMINAMATH_CALUDE_shortest_side_theorem_l435_43549

theorem shortest_side_theorem (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c → b + c > a → a + c > b → 
  a^2 + b^2 > 5*c^2 → 
  c < a ∧ c < b :=
sorry

end NUMINAMATH_CALUDE_shortest_side_theorem_l435_43549


namespace NUMINAMATH_CALUDE_larger_number_proof_l435_43501

theorem larger_number_proof (A B : ℕ+) : 
  (Nat.gcd A B = 20) → 
  (∃ (x : ℕ+), Nat.lcm A B = 20 * 11 * 15 * x) → 
  (A ≤ B) →
  B = 300 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l435_43501


namespace NUMINAMATH_CALUDE_school_trip_photos_l435_43587

theorem school_trip_photos (claire lisa robert : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 28) :
  claire = 14 := by
sorry

end NUMINAMATH_CALUDE_school_trip_photos_l435_43587


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l435_43524

theorem no_simultaneous_squares : ¬∃ (m n : ℕ), ∃ (k l : ℕ), m^2 + n = k^2 ∧ n^2 + m = l^2 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l435_43524


namespace NUMINAMATH_CALUDE_faucet_drip_properties_l435_43579

/-- Represents the volume of water dripped from a faucet -/
def water_volume (time_minutes : ℝ) : ℝ :=
  6 * time_minutes

theorem faucet_drip_properties :
  (∀ x : ℝ, water_volume x = 6 * x) ∧
  (water_volume 50 = 300) := by
  sorry

end NUMINAMATH_CALUDE_faucet_drip_properties_l435_43579


namespace NUMINAMATH_CALUDE_actual_time_when_clock_shows_7pm_l435_43539

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Converts Time to minutes since midnight -/
def timeToMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

/-- Represents a clock that may gain or lose time -/
structure Clock where
  rate : ℚ  -- Rate of time gain/loss (1 means accurate, >1 means gaining time)

theorem actual_time_when_clock_shows_7pm 
  (c : Clock) 
  (h1 : c.rate = 7 / 6)  -- Clock gains 5 minutes in 30 minutes
  (h2 : timeToMinutes { hours := 7, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } = 
        c.rate * timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num }) :
  timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } = 
  timeToMinutes { hours := 18, minutes := 0, hLt24 := by norm_num, mLt60 := by norm_num } := by
  sorry

end NUMINAMATH_CALUDE_actual_time_when_clock_shows_7pm_l435_43539


namespace NUMINAMATH_CALUDE_inequality_proof_l435_43580

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * (a + b) + a * c * (a + c) + b * c * (b + c)) / (a * b * c) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l435_43580


namespace NUMINAMATH_CALUDE_fraction_equivalence_l435_43581

theorem fraction_equivalence : 
  ∀ x : ℝ, x ≠ 0 → (x / (740/999)) * (5/9) = x / 1.4814814814814814 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l435_43581


namespace NUMINAMATH_CALUDE_induction_contrapositive_l435_43507

theorem induction_contrapositive (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  (¬ P 4) →
  (¬ P 3) :=
by sorry

end NUMINAMATH_CALUDE_induction_contrapositive_l435_43507


namespace NUMINAMATH_CALUDE_manager_wage_l435_43541

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def wage_conditions (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.2 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.manager - 3.4

/-- The theorem stating that under the given conditions, the manager's hourly wage is $8.50 -/
theorem manager_wage (w : Wages) (h : wage_conditions w) : w.manager = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_manager_wage_l435_43541


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l435_43565

theorem correct_quadratic_equation :
  ∃ (r₁ r₂ : ℝ), 
    (r₁ + r₂ = 9) ∧ 
    (r₁ * r₂ = 18) ∧ 
    (∃ (s₁ s₂ : ℝ), s₁ + s₂ = 5 - 1 ∧ s₁ + s₂ = 9) ∧
    (r₁ * r₂ = r₁ * r₂ - 9 * (r₁ + r₂) + 18) := by
  sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l435_43565


namespace NUMINAMATH_CALUDE_largest_divisible_n_l435_43555

theorem largest_divisible_n : ∃ (n : ℕ), n = 180 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 20) ∣ (m^3 + 1000))) ∧ 
  ((n + 20) ∣ (n^3 + 1000)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l435_43555


namespace NUMINAMATH_CALUDE_age_difference_is_24_l435_43573

/-- Proves that the age difference between Ana and Claudia is 24 years --/
theorem age_difference_is_24 (A C : ℕ) (n : ℕ) : 
  A = C + n →                 -- Ana is n years older than Claudia
  A - 3 = 6 * (C - 3) →       -- Three years ago, Ana was 6 times as old as Claudia
  A = C^3 →                   -- This year Ana's age is the cube of Claudia's age
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_age_difference_is_24_l435_43573


namespace NUMINAMATH_CALUDE_multiple_p_solutions_l435_43562

/-- The probability of getting exactly k heads in n tosses of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 heads in 5 tosses -/
def w (p : ℝ) : ℝ := binomial_probability 5 3 p

/-- There exist at least two distinct values of p in (0, 1) that satisfy w(p) = 144/625 -/
theorem multiple_p_solutions : ∃ p₁ p₂ : ℝ, 0 < p₁ ∧ p₁ < 1 ∧ 0 < p₂ ∧ p₂ < 1 ∧ p₁ ≠ p₂ ∧ w p₁ = 144/625 ∧ w p₂ = 144/625 := by
  sorry

end NUMINAMATH_CALUDE_multiple_p_solutions_l435_43562


namespace NUMINAMATH_CALUDE_average_marks_l435_43543

theorem average_marks (total_subjects : ℕ) (avg_five_subjects : ℝ) (sixth_subject_marks : ℝ) :
  total_subjects = 6 →
  avg_five_subjects = 74 →
  sixth_subject_marks = 104 →
  (avg_five_subjects * 5 + sixth_subject_marks) / total_subjects = 79 :=
by
  sorry

end NUMINAMATH_CALUDE_average_marks_l435_43543


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l435_43535

/-- Converts a binary number represented as a list of bits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The binary representation of 1010 101₂ -/
def binary_num : List Bool := [true, false, true, false, true, false, true]

/-- The octal representation of 125₈ -/
def octal_num : List ℕ := [1, 2, 5]

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry

#eval binary_to_decimal binary_num
#eval decimal_to_octal (binary_to_decimal binary_num)

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l435_43535


namespace NUMINAMATH_CALUDE_gym_visitors_l435_43511

theorem gym_visitors (initial_count : ℕ) (left_count : ℕ) (final_count : ℕ) :
  final_count ≥ initial_count - left_count →
  (final_count - (initial_count - left_count)) = 
  (final_count + left_count - initial_count) :=
by sorry

end NUMINAMATH_CALUDE_gym_visitors_l435_43511


namespace NUMINAMATH_CALUDE_inequality_solution_l435_43519

theorem inequality_solution (x : ℝ) : 
  (5 - 1 / (3 * x + 4) < 7) ↔ (x < -11/6 ∨ x > -4/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l435_43519


namespace NUMINAMATH_CALUDE_stool_sticks_calculation_l435_43504

/-- The number of sticks of wood a chair makes -/
def chair_sticks : ℕ := 6

/-- The number of sticks of wood a table makes -/
def table_sticks : ℕ := 9

/-- The number of sticks Mary needs to burn per hour to stay warm -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped up -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chopped up -/
def stools_chopped : ℕ := 4

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

/-- The number of sticks of wood a stool makes -/
def stool_sticks : ℕ := 2

theorem stool_sticks_calculation :
  stool_sticks * stools_chopped = 
    hours_warm * sticks_per_hour - 
    (chair_sticks * chairs_chopped + table_sticks * tables_chopped) :=
by sorry

end NUMINAMATH_CALUDE_stool_sticks_calculation_l435_43504


namespace NUMINAMATH_CALUDE_triangle_cosine_l435_43592

theorem triangle_cosine (X Y Z : ℝ) (h1 : X + Y + Z = Real.pi) 
  (h2 : X = Real.pi / 2) (h3 : Y = Real.pi / 4) (h4 : Real.tan Z = 1 / 2) : 
  Real.cos Z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_l435_43592


namespace NUMINAMATH_CALUDE_solve_for_k_l435_43597

theorem solve_for_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (heq : k * x - y = 3) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l435_43597


namespace NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l435_43532

theorem smallest_distance_between_points_on_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 5*I)) = 2)
  (hw : Complex.abs (w - (-3 + 4*I)) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 106 - 6 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 - 5*I)) = 2 → 
      Complex.abs (w' - (-3 + 4*I)) = 4 → 
      Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l435_43532


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l435_43577

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a : ℚ), x = Real.sqrt a ∧ ∀ (b : ℚ), b ≠ a → Real.sqrt b ≠ x

theorem simplest_quadratic_radical :
  let options : List ℝ := [1 / Real.sqrt 3, Real.sqrt (5 / 6), Real.sqrt 24, Real.sqrt 21]
  ∀ y ∈ options, is_simplest_quadratic_radical (Real.sqrt 21) ∧ 
    (is_simplest_quadratic_radical y → y = Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l435_43577


namespace NUMINAMATH_CALUDE_unit_square_fits_in_parallelogram_l435_43568

/-- A parallelogram with heights greater than 1 -/
structure Parallelogram where
  heights : ℝ → ℝ
  height_gt_one : ∀ h, heights h > 1

/-- A unit square -/
structure UnitSquare where
  side_length : ℝ
  is_unit : side_length = 1

/-- A placement of a shape inside a parallelogram -/
structure Placement (P : Parallelogram) (S : Type) where
  is_inside : S → Bool

/-- Theorem: For any parallelogram with heights greater than 1, 
    there exists a placement of a unit square inside it -/
theorem unit_square_fits_in_parallelogram (P : Parallelogram) :
  ∃ (U : UnitSquare) (place : Placement P UnitSquare), place.is_inside U = true := by
  sorry

end NUMINAMATH_CALUDE_unit_square_fits_in_parallelogram_l435_43568


namespace NUMINAMATH_CALUDE_solve_for_B_l435_43505

theorem solve_for_B : ∃ B : ℝ, (4 * B + 4 - 3 = 33) ∧ (B = 8) := by sorry

end NUMINAMATH_CALUDE_solve_for_B_l435_43505


namespace NUMINAMATH_CALUDE_cube_of_eight_l435_43529

theorem cube_of_eight : 8^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_eight_l435_43529


namespace NUMINAMATH_CALUDE_unique_solution_l435_43584

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: 2001 is the only natural number n that satisfies n + S(n) = 2004 -/
theorem unique_solution : ∀ n : ℕ, n + S n = 2004 ↔ n = 2001 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l435_43584


namespace NUMINAMATH_CALUDE_square_of_1307_squared_l435_43586

theorem square_of_1307_squared : (1307 * 1307)^2 = 2918129502401 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1307_squared_l435_43586


namespace NUMINAMATH_CALUDE_f_monotonicity_l435_43520

def f (m n : ℕ) (x : ℝ) : ℝ := x^(m/n)

theorem f_monotonicity (m n : ℕ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m n x₁ < f m n x₂) ∧
  (n % 2 = 1 ∧ m % 2 = 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f m n x₁ > f m n x₂) ∧
  (n % 2 = 1 ∧ m % 2 = 1 → ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f m n x₁ < f m n x₂) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_l435_43520


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l435_43548

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

-- State the theorem
theorem complement_of_P_in_U : 
  U \ P = Set.Ioo (-1) 6 := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l435_43548


namespace NUMINAMATH_CALUDE_smallest_cookie_packages_l435_43566

theorem smallest_cookie_packages (cookie_per_package : Nat) (milk_per_package : Nat) 
  (h1 : cookie_per_package = 5) (h2 : milk_per_package = 7) :
  ∃ n : Nat, n > 0 ∧ (cookie_per_package * n) % milk_per_package = 0 ∧
  ∀ m : Nat, m > 0 ∧ (cookie_per_package * m) % milk_per_package = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_packages_l435_43566


namespace NUMINAMATH_CALUDE_prime_power_plus_three_l435_43591

theorem prime_power_plus_three (P : ℕ) : 
  Prime P → Prime (P^6 + 3) → P^10 + 3 = 1027 := by sorry

end NUMINAMATH_CALUDE_prime_power_plus_three_l435_43591


namespace NUMINAMATH_CALUDE_coefficient_of_negative_2pi_ab_squared_l435_43585

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℝ) (x : String) : ℝ := sorry

/-- A monomial is an algebraic expression consisting of a single term. -/
def is_monomial (x : String) : Prop := sorry

theorem coefficient_of_negative_2pi_ab_squared :
  is_monomial "-2πab²" → coefficient (-2 * Real.pi) "ab²" = -2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_coefficient_of_negative_2pi_ab_squared_l435_43585


namespace NUMINAMATH_CALUDE_point_on_y_axis_l435_43503

theorem point_on_y_axis (m : ℝ) :
  (m + 1 = 0) → ((m + 1, m + 4) : ℝ × ℝ) = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l435_43503


namespace NUMINAMATH_CALUDE_selection_methods_equality_l435_43551

def num_male_students : ℕ := 20
def num_female_students : ℕ := 30
def total_students : ℕ := num_male_students + num_female_students
def num_selected : ℕ := 4

theorem selection_methods_equality :
  (Nat.choose total_students num_selected - Nat.choose num_male_students num_selected - Nat.choose num_female_students num_selected) =
  (Nat.choose num_male_students 1 * Nat.choose num_female_students 3 +
   Nat.choose num_male_students 2 * Nat.choose num_female_students 2 +
   Nat.choose num_male_students 3 * Nat.choose num_female_students 1) :=
by sorry

end NUMINAMATH_CALUDE_selection_methods_equality_l435_43551


namespace NUMINAMATH_CALUDE_distance_to_x_axis_on_ellipse_l435_43518

/-- The distance from a point on an ellipse to the x-axis, given specific conditions -/
theorem distance_to_x_axis_on_ellipse (x y : ℝ) : 
  (x^2 / 2 + y^2 / 6 = 1) →  -- Point (x, y) is on the ellipse
  (x * x + (y + 2) * (y - 2) = 0) →  -- Dot product condition
  |y| = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_on_ellipse_l435_43518


namespace NUMINAMATH_CALUDE_employee_not_on_first_day_l435_43578

def num_employees : ℕ := 6
def num_days : ℕ := 3
def employees_per_day : ℕ := 2

def probability_not_on_first_day : ℚ :=
  2 / 3

theorem employee_not_on_first_day :
  let total_arrangements := (num_employees.choose employees_per_day) * 
                            ((num_employees - employees_per_day).choose employees_per_day) * 
                            ((num_employees - 2 * employees_per_day).choose employees_per_day)
  let arrangements_without_A := (num_employees - 1).choose 1 * 
                                ((num_employees - employees_per_day).choose employees_per_day) * 
                                ((num_employees - 2 * employees_per_day).choose employees_per_day)
  (arrangements_without_A : ℚ) / total_arrangements = probability_not_on_first_day :=
sorry

end NUMINAMATH_CALUDE_employee_not_on_first_day_l435_43578


namespace NUMINAMATH_CALUDE_quadratic_intersection_point_l435_43557

theorem quadratic_intersection_point 
  (a b c d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : b ≠ 0) : 
  let f1 := fun x : ℝ => a * x^2 + b * x + c
  let f2 := fun x : ℝ => a * x^2 - b * x + c + d
  ∃! p : ℝ × ℝ, 
    f1 p.1 = f2 p.1 ∧ 
    p = (d / (2 * b), a * (d^2 / (4 * b^2)) + d / 2 + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_point_l435_43557


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l435_43553

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -9 ∧ x₂ = -1 ∧ 
  (x₁^2 + 10*x₁ + 9 = 0) ∧ 
  (x₂^2 + 10*x₂ + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l435_43553


namespace NUMINAMATH_CALUDE_max_age_on_aubrey_birthday_l435_43527

/-- The age difference between Luka and Aubrey -/
def age_difference : ℕ := 2

/-- Luka's age when Max was born -/
def luka_age_at_max_birth : ℕ := 4

/-- Aubrey's age for which we want to find Max's age -/
def aubrey_target_age : ℕ := 8

/-- Max's age when Aubrey reaches the target age -/
def max_age : ℕ := aubrey_target_age - age_difference

theorem max_age_on_aubrey_birthday :
  max_age = 6 := by sorry

end NUMINAMATH_CALUDE_max_age_on_aubrey_birthday_l435_43527


namespace NUMINAMATH_CALUDE_circle_area_difference_l435_43545

theorem circle_area_difference : 
  let r1 : ℝ := 20
  let d2 : ℝ := 20
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 300 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l435_43545


namespace NUMINAMATH_CALUDE_parallelogram_base_l435_43500

/-- The base of a parallelogram with area 240 square cm and height 10 cm is 24 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 240 ∧ height = 10 ∧ area = base * height → base = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l435_43500


namespace NUMINAMATH_CALUDE_quadratic_root_range_l435_43554

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x > 1 ∧ y < -1 ∧ 
   x^2 + (a^2 + 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 + 1)*y + a - 2 = 0) →
  -1 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l435_43554


namespace NUMINAMATH_CALUDE_two_people_two_rooms_probability_prove_two_people_two_rooms_probability_l435_43502

/-- The probability of two individuals randomly choosing different rooms out of two available rooms -/
theorem two_people_two_rooms_probability : ℝ :=
  1 / 2

/-- Prove that the probability of two individuals randomly choosing different rooms out of two available rooms is 1/2 -/
theorem prove_two_people_two_rooms_probability :
  two_people_two_rooms_probability = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_people_two_rooms_probability_prove_two_people_two_rooms_probability_l435_43502


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l435_43550

/-- A function that returns true if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns true if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- A function that returns true if a number is a multiple of 9 -/
def isMultipleOf9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

/-- A function that returns the tens digit of a two-digit number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- A function that returns the ones digit of a two-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- A function that returns true if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem unique_two_digit_number :
  ∃! n : ℕ, isTwoDigit n ∧ isOdd n ∧ isMultipleOf9 n ∧
    isPerfectSquare (tensDigit n * onesDigit n) ∧ n = 99 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l435_43550


namespace NUMINAMATH_CALUDE_constant_term_zero_implies_m_negative_one_l435_43552

/-- The quadratic equation in x with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * x - m^2 + 1

/-- The constant term of the quadratic equation -/
def constant_term (m : ℝ) : ℝ := quadratic_equation m 0

theorem constant_term_zero_implies_m_negative_one :
  constant_term (-1) = 0 ∧ (∀ m : ℝ, constant_term m = 0 → m = -1) :=
sorry

end NUMINAMATH_CALUDE_constant_term_zero_implies_m_negative_one_l435_43552


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l435_43571

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (δ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (γ + δ) = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l435_43571


namespace NUMINAMATH_CALUDE_gcd_364_154_l435_43593

theorem gcd_364_154 : Nat.gcd 364 154 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_364_154_l435_43593


namespace NUMINAMATH_CALUDE_no_n_exists_for_combination_equality_l435_43506

theorem no_n_exists_for_combination_equality :
  ¬ ∃ (n : ℕ), n > 0 ∧ (Nat.choose n 3 = Nat.choose (n-1) 3 + Nat.choose (n-1) 4) := by
  sorry

end NUMINAMATH_CALUDE_no_n_exists_for_combination_equality_l435_43506


namespace NUMINAMATH_CALUDE_value_of_A_l435_43508

/-- Given the value assignments for letters and words, prove the value of A -/
theorem value_of_A (H M A T E : ℤ)
  (h1 : H = 10)
  (h2 : M + A + T + H = 35)
  (h3 : T + E + A + M = 42)
  (h4 : M + E + E + T = 38) :
  A = 21 := by
  sorry

end NUMINAMATH_CALUDE_value_of_A_l435_43508


namespace NUMINAMATH_CALUDE_least_integer_greater_than_negative_eighteen_fifths_l435_43576

theorem least_integer_greater_than_negative_eighteen_fifths :
  ∃ n : ℤ, n > -18/5 ∧ ∀ m : ℤ, m > -18/5 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_negative_eighteen_fifths_l435_43576


namespace NUMINAMATH_CALUDE_division_of_fractions_l435_43534

theorem division_of_fractions : (5 : ℚ) / 6 / (7 / 4) = 10 / 21 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l435_43534


namespace NUMINAMATH_CALUDE_smallest_m_chess_tournament_l435_43594

theorem smallest_m_chess_tournament : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (
    (∃ (x : ℕ), x > 0 ∧
      (4 * k * (4 * k - 1)) / 2 = 11 * x ∧
      8 * x + 3 * x = (4 * k * (4 * k - 1)) / 2
    ) → k ≥ m
  )) ∧ 
  (∃ (x : ℕ), x > 0 ∧
    (4 * m * (4 * m - 1)) / 2 = 11 * x ∧
    8 * x + 3 * x = (4 * m * (4 * m - 1)) / 2
  ) ∧
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_chess_tournament_l435_43594


namespace NUMINAMATH_CALUDE_marnie_chips_consumption_l435_43544

/-- Given a bag of chips and Marnie's eating pattern, calculate the number of days to finish the bag -/
def days_to_finish_chips (total_chips : ℕ) (first_day_consumption : ℕ) (daily_consumption : ℕ) : ℕ :=
  1 + ((total_chips - first_day_consumption) + daily_consumption - 1) / daily_consumption

/-- Theorem: It takes Marnie 10 days to eat the whole bag of chips -/
theorem marnie_chips_consumption :
  days_to_finish_chips 100 10 10 = 10 := by
  sorry

#eval days_to_finish_chips 100 10 10

end NUMINAMATH_CALUDE_marnie_chips_consumption_l435_43544


namespace NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l435_43556

/-- Given a quadratic polynomial P(x) = ax² + bx + c where a ≠ 0,
    if P(x) = x - 2 has exactly one root and
    P(x) = 1 - x/2 has exactly one root,
    then the discriminant of P(x) is -1/2 -/
theorem quadratic_polynomial_discriminant
  (a b c : ℝ) (ha : a ≠ 0)
  (h1 : ∃! x, a * x^2 + b * x + c = x - 2)
  (h2 : ∃! x, a * x^2 + b * x + c = 1 - x / 2) :
  b^2 - 4*a*c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_discriminant_l435_43556


namespace NUMINAMATH_CALUDE_max_value_on_circle_l435_43599

theorem max_value_on_circle :
  let circle := {p : ℝ × ℝ | (p.1^2 + p.2^2 + 4*p.1 - 6*p.2 + 4) = 0}
  ∃ (max : ℝ), max = -13 ∧ 
    (∀ p ∈ circle, 3*p.1 - 4*p.2 ≤ max) ∧
    (∃ p ∈ circle, 3*p.1 - 4*p.2 = max) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l435_43599


namespace NUMINAMATH_CALUDE_complement_union_theorem_complement_intersect_theorem_l435_43531

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem complement_union_theorem : 
  (Set.univ \ (A ∪ B)) = {x : ℝ | x ≤ 2 ∨ x ≥ 10} := by sorry

theorem complement_intersect_theorem :
  ((Set.univ \ A) ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_complement_intersect_theorem_l435_43531


namespace NUMINAMATH_CALUDE_polynomial_factorization_l435_43522

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 2*x) * (x^2 + 2*x + 2) + 1 = (x + 1)^4 ∧
  (x^2 - 4*x) * (x^2 - 4*x + 8) + 16 = (x - 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l435_43522


namespace NUMINAMATH_CALUDE_same_commission_list_price_is_65_l435_43516

-- Define the list price
def list_price : ℝ := 65

-- Define Alice's selling price
def alice_selling_price : ℝ := list_price - 15

-- Define Bob's selling price
def bob_selling_price : ℝ := list_price - 25

-- Define Alice's commission rate
def alice_commission_rate : ℝ := 0.12

-- Define Bob's commission rate
def bob_commission_rate : ℝ := 0.15

-- Theorem stating that Alice and Bob get the same commission
theorem same_commission :
  alice_commission_rate * alice_selling_price = bob_commission_rate * bob_selling_price :=
by sorry

-- Main theorem proving that the list price is 65
theorem list_price_is_65 : list_price = 65 :=
by sorry

end NUMINAMATH_CALUDE_same_commission_list_price_is_65_l435_43516


namespace NUMINAMATH_CALUDE_average_income_Q_R_l435_43542

theorem average_income_Q_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_Q_R_l435_43542


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_l435_43596

/-- The number of unit squares in the nth ring around a 2x3 rectangle -/
def ring_squares (n : ℕ) : ℕ := 4 * n + 8

/-- Theorem: The 100th ring contains 408 unit squares -/
theorem hundredth_ring_squares :
  ring_squares 100 = 408 := by sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_l435_43596


namespace NUMINAMATH_CALUDE_cycle_selling_price_l435_43595

/-- Given a cycle bought for Rs. 930 and sold with a gain of 30.107526881720432%,
    prove that the selling price is Rs. 1210. -/
theorem cycle_selling_price (cost_price : ℝ) (gain_percentage : ℝ) (selling_price : ℝ) :
  cost_price = 930 →
  gain_percentage = 30.107526881720432 →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  selling_price = 1210 :=
by sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l435_43595


namespace NUMINAMATH_CALUDE_nancy_antacids_per_month_l435_43525

/-- Calculates the number of antacids Nancy takes per month -/
def antacids_per_month (indian_antacids : ℕ) (mexican_antacids : ℕ) (other_antacids : ℕ)
  (indian_freq : ℕ) (mexican_freq : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let days_per_week := 7
  let other_days := days_per_week - indian_freq - mexican_freq
  let weekly_antacids := indian_antacids * indian_freq + mexican_antacids * mexican_freq + other_antacids * other_days
  weekly_antacids * weeks_per_month

/-- Proves that Nancy takes 60 antacids per month given the specified conditions -/
theorem nancy_antacids_per_month :
  antacids_per_month 3 2 1 3 2 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_nancy_antacids_per_month_l435_43525


namespace NUMINAMATH_CALUDE_estimate_2_sqrt_5_l435_43530

theorem estimate_2_sqrt_5 : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 := by
  sorry

end NUMINAMATH_CALUDE_estimate_2_sqrt_5_l435_43530


namespace NUMINAMATH_CALUDE_sum_of_parts_l435_43589

theorem sum_of_parts (x y : ℝ) : x + y = 24 → y = 13 → y > x → 7 * x + 5 * y = 142 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_l435_43589


namespace NUMINAMATH_CALUDE_complex_division_by_i_l435_43513

theorem complex_division_by_i (z : ℂ) : z.re = -2 ∧ z.im = -1 → z / Complex.I = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_by_i_l435_43513


namespace NUMINAMATH_CALUDE_part_one_part_two_l435_43575

-- Define the logarithmic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem for part 1
theorem part_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 8 = 3) :
  a = 2 := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 0 < x ∧ x ≤ 1/2}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 1/2 ≤ x ∧ x < 2/3}) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l435_43575


namespace NUMINAMATH_CALUDE_existence_of_integers_l435_43558

theorem existence_of_integers : ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l435_43558


namespace NUMINAMATH_CALUDE_range_of_m_l435_43547

theorem range_of_m (x m : ℝ) : 
  (∀ x, -1 < x ∧ x < 4 → x > 2*m^2 - 3) ∧ 
  (∃ x, x > 2*m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) → 
  -1 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l435_43547


namespace NUMINAMATH_CALUDE_votes_against_percentage_l435_43559

theorem votes_against_percentage (total_votes : ℕ) (difference : ℕ) :
  total_votes = 330 →
  difference = 66 →
  let votes_against := (total_votes - difference) / 2
  let percentage_against := (votes_against : ℚ) / total_votes * 100
  percentage_against = 40 := by
  sorry

end NUMINAMATH_CALUDE_votes_against_percentage_l435_43559


namespace NUMINAMATH_CALUDE_unique_pair_existence_l435_43570

theorem unique_pair_existence (n : ℕ+) :
  ∃! (k l : ℕ), 0 ≤ l ∧ l < k ∧ n = (k * (k - 1)) / 2 + l := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_existence_l435_43570


namespace NUMINAMATH_CALUDE_intersection_abscissas_l435_43540

-- Define the parabola and line equations
def parabola (x : ℝ) : ℝ := x^2 - 4*x
def line : ℝ := 5

-- Define the intersection points
def intersection_points : Set ℝ := {x | parabola x = line}

-- Theorem statement
theorem intersection_abscissas :
  intersection_points = {-1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_abscissas_l435_43540


namespace NUMINAMATH_CALUDE_midpoint_fraction_l435_43517

theorem midpoint_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 6
  (a + b) / 2 = (19 : ℚ) / 24 := by
sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l435_43517


namespace NUMINAMATH_CALUDE_ratio_fourth_term_l435_43510

theorem ratio_fourth_term (x y : ℝ) (hx : x = 0.8571428571428571) :
  (0.75 : ℝ) / x = 7 / y → y = 8 := by
sorry

end NUMINAMATH_CALUDE_ratio_fourth_term_l435_43510


namespace NUMINAMATH_CALUDE_cubic_root_sum_l435_43563

theorem cubic_root_sum (a b c : ℝ) : 
  (40 * a^3 - 60 * a^2 + 25 * a - 1 = 0) →
  (40 * b^3 - 60 * b^2 + 25 * b - 1 = 0) →
  (40 * c^3 - 60 * c^2 + 25 * c - 1 = 0) →
  (0 < a) ∧ (a < 1) →
  (0 < b) ∧ (b < 1) →
  (0 < c) ∧ (c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l435_43563


namespace NUMINAMATH_CALUDE_jimmy_action_figures_sale_discount_l435_43561

theorem jimmy_action_figures_sale_discount (total_figures : ℕ) 
  (regular_figure_value : ℚ) (special_figure_value : ℚ) (total_earned : ℚ) :
  total_figures = 5 →
  regular_figure_value = 15 →
  special_figure_value = 20 →
  total_earned = 55 →
  (4 * regular_figure_value + special_figure_value - total_earned) / total_figures = 5 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_action_figures_sale_discount_l435_43561


namespace NUMINAMATH_CALUDE_valid_a_values_l435_43512

def A (a : ℝ) : Set ℝ := {2, 1 - a, a^2 - a + 2}

theorem valid_a_values : ∀ a : ℝ, 4 ∈ A a ↔ a = -3 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_valid_a_values_l435_43512


namespace NUMINAMATH_CALUDE_triangle_shape_determination_l435_43583

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of two sides and the included angle of a triangle -/
def ratio_two_sides_and_angle (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratios of the three angle bisectors of a triangle -/
def ratio_angle_bisectors (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratios of the three medians of a triangle -/
def ratio_medians (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratio of the circumradius to the inradius of a triangle -/
def ratio_circumradius_to_inradius (t : Triangle) : ℝ := sorry

/-- Two angles of a triangle -/
def two_angles (t : Triangle) : ℝ × ℝ := sorry

/-- Two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop := sorry

/-- The shape of a triangle is uniquely determined by a given property
    if any two triangles with the same property are similar -/
def uniquely_determines_shape (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → are_similar t1 t2

theorem triangle_shape_determination :
  uniquely_determines_shape ratio_two_sides_and_angle ∧
  uniquely_determines_shape ratio_angle_bisectors ∧
  uniquely_determines_shape ratio_medians ∧
  ¬ uniquely_determines_shape ratio_circumradius_to_inradius ∧
  uniquely_determines_shape two_angles := by sorry

end NUMINAMATH_CALUDE_triangle_shape_determination_l435_43583


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l435_43538

theorem trigonometric_inequality (α : ℝ) : 4 * Real.sin (3 * α) + 5 ≥ 4 * Real.cos (2 * α) + 5 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l435_43538


namespace NUMINAMATH_CALUDE_tangent_line_equation_monotonic_increase_condition_l435_43588

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Theorem for the tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f (-1) 1 = 0 ∧ 
  (deriv (f (-1))) 1 = -Real.log 2 →
  (Real.log 2 * x + y - Real.log 2 = 0) ↔ 
  y = (deriv (f (-1))) 1 * (x - 1) + f (-1) 1 :=
sorry

-- Theorem for monotonic increase condition
theorem monotonic_increase_condition (a : ℝ) :
  Monotone (f a) ↔ a ≥ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_monotonic_increase_condition_l435_43588


namespace NUMINAMATH_CALUDE_product_347_6_base9_l435_43537

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Theorem: The product of 347₉ and 6₉ in base 9 is 2316₉ --/
theorem product_347_6_base9 :
  base10ToBase9 (base9ToBase10 [7, 4, 3] * base9ToBase10 [6]) = [6, 1, 3, 2] := by
  sorry

end NUMINAMATH_CALUDE_product_347_6_base9_l435_43537


namespace NUMINAMATH_CALUDE_number_calculation_l435_43590

theorem number_calculation (N : ℝ) : (0.15 * 0.30 * 0.50 * N = 108) → N = 4800 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l435_43590


namespace NUMINAMATH_CALUDE_max_min_values_l435_43526

noncomputable def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  ∃ (m n : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x a ≤ m) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x a = m) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → n ≤ f x a) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x a = n) ∧
    m = 1 + a ∧
    n = -3 + a :=
by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l435_43526


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l435_43515

theorem repeating_decimal_sum (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 12 / 13 →
  a = 4 ∧ b = 6 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l435_43515


namespace NUMINAMATH_CALUDE_rory_tank_water_l435_43569

/-- Calculates the final amount of water in Rory's tank after a rainstorm --/
def final_water_amount (initial_water : ℝ) (inflow_rate_1 inflow_rate_2 : ℝ) 
  (leak_rate : ℝ) (evap_rate_1 evap_rate_2 : ℝ) (evap_reduction : ℝ) 
  (duration_1 duration_2 : ℝ) : ℝ :=
  let total_inflow := inflow_rate_1 * duration_1 + inflow_rate_2 * duration_2
  let total_leak := leak_rate * (duration_1 + duration_2)
  let total_evap := (evap_rate_1 * duration_1 + evap_rate_2 * duration_2) * (1 - evap_reduction)
  initial_water + total_inflow - total_leak - total_evap

/-- Theorem stating the final amount of water in Rory's tank --/
theorem rory_tank_water : 
  final_water_amount 100 2 3 0.5 0.2 0.1 0.75 45 45 = 276.625 := by
  sorry

end NUMINAMATH_CALUDE_rory_tank_water_l435_43569


namespace NUMINAMATH_CALUDE_milk_for_six_cookies_l435_43509

/-- Represents the number of cups of milk required for a given number of cookies -/
def milkRequired (cookies : ℕ) : ℚ :=
  sorry

theorem milk_for_six_cookies :
  let cookies_per_quart : ℕ := 24 / 4
  let pints_per_quart : ℕ := 2
  let cups_per_pint : ℕ := 2
  milkRequired 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_six_cookies_l435_43509


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l435_43533

theorem range_of_y_over_x (x y : ℝ) (h1 : 3 * x - 2 * y - 5 = 0) (h2 : 1 ≤ x) (h3 : x ≤ 2) :
  ∃ (z : ℝ), z = y / x ∧ -1 ≤ z ∧ z ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l435_43533


namespace NUMINAMATH_CALUDE_min_value_of_expression_l435_43598

theorem min_value_of_expression (a b : ℕ) (ha : 0 < a ∧ a ≤ 5) (hb : 0 < b ∧ b ≤ 5) :
  ∀ x y : ℕ, (0 < x ∧ x ≤ 5) → (0 < y ∧ y ≤ 5) → 
  a^2 - a*b + 2*b ≤ x^2 - x*y + 2*y ∧ 
  a^2 - a*b + 2*b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l435_43598


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l435_43564

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (4 * a 1 = a m) →
  (a m)^2 = a 1 * a n →
  (m + n = 6) →
  (1 / m + 4 / n ≥ 3 / 2) ∧
  (∃ m₀ n₀ : ℕ, m₀ + n₀ = 6 ∧ 1 / m₀ + 4 / n₀ = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l435_43564
