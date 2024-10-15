import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_proof_l718_71890

/-- Represents the total population -/
def total_population : ℕ := 27 + 54 + 81

/-- Represents the number of elderly people in the population -/
def elderly_population : ℕ := 27

/-- Represents the number of elderly people in the sample -/
def elderly_sample : ℕ := 3

/-- Represents the total sample size -/
def sample_size : ℕ := 18

/-- Proves that the given sample size is correct for the stratified sampling -/
theorem stratified_sampling_proof :
  (elderly_sample : ℚ) / elderly_population = sample_size / total_population :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proof_l718_71890


namespace NUMINAMATH_CALUDE_exam_full_marks_l718_71879

theorem exam_full_marks (A B C D F : ℝ) 
  (hA : A = 0.9 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.8 * D)
  (hAmarks : A = 360)
  (hDpercent : D = 0.8 * F) : 
  F = 500 := by
sorry

end NUMINAMATH_CALUDE_exam_full_marks_l718_71879


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l718_71804

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerMod10 (base : ℕ) (exp : ℕ) : ℕ :=
  (base ^ exp) % 10

theorem units_digit_of_sum : unitsDigit ((33 : ℕ)^43 + (43 : ℕ)^32) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l718_71804


namespace NUMINAMATH_CALUDE_system_solution_l718_71805

theorem system_solution (a b c x y z : ℝ) : 
  (a * y + b * x = c ∧ c * x + a * z = b ∧ b * z + c * y = a) ↔
  ((a * b * c ≠ 0 ∧ 
    x = (b^2 + c^2 - a^2) / (2*b*c) ∧ 
    y = (a^2 + c^2 - b^2) / (2*a*c) ∧ 
    z = (a^2 + b^2 - c^2) / (2*a*b)) ∨
   (a = 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    ((x = 1 ∧ y = z) ∨ (x = 1 ∧ y = -z))) ∨
   (b = 0 ∧ a ≠ 0 ∧ c ≠ 0 ∧ 
    ((y = 1 ∧ x = z) ∨ (y = 1 ∧ x = -z))) ∨
   (c = 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ 
    ((z = 1 ∧ x = y) ∨ (z = 1 ∧ x = -y))) ∨
   (a = 0 ∧ b = 0 ∧ c = 0)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l718_71805


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l718_71845

theorem problem_1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := by sorry

theorem problem_2 : (4 * Real.sqrt 6 - 6 * Real.sqrt 3) / (2 * Real.sqrt 3) = 2 * Real.sqrt 2 - 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l718_71845


namespace NUMINAMATH_CALUDE_f_properties_l718_71874

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x + a * Real.log x

theorem f_properties (a : ℝ) (h_a : a > 0) :
  (∃ (x_min : ℝ), x_min > 0 ∧ x_min = 1/2 ∧ 
    (∀ (x : ℝ), x > 0 → f a x ≥ f a x_min)) ∧
  (¬∃ (x_max : ℝ), x_max > 0 ∧ 
    (∀ (x : ℝ), x > 0 → f a x ≤ f a x_max)) ∧
  ((∃ (x : ℝ), x > 0 ∧ f a x < 2) ↔ (a > 0 ∧ a ≠ 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l718_71874


namespace NUMINAMATH_CALUDE_min_value_expression_l718_71850

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((2*a + 2*a*b - b*(b + 1))^2 + (b - 4*a^2 + 2*a*(b + 1))^2) / (4*a^2 + b^2) ≥ 1 ∧
  ((2*1 + 2*1*1 - 1*(1 + 1))^2 + (1 - 4*1^2 + 2*1*(1 + 1))^2) / (4*1^2 + 1^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l718_71850


namespace NUMINAMATH_CALUDE_smaller_number_is_42_l718_71899

theorem smaller_number_is_42 (x y : ℕ) (h1 : x + y = 96) (h2 : y = x + 12) : x = 42 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_is_42_l718_71899


namespace NUMINAMATH_CALUDE_worker_time_relationship_l718_71812

/-- Given a batch of parts and a production rate, this theorem establishes
    the relationship between the number of workers and the time needed to complete the task. -/
theorem worker_time_relationship 
  (total_parts : ℕ) 
  (production_rate : ℕ) 
  (h1 : total_parts = 200)
  (h2 : production_rate = 10) :
  ∀ x y : ℝ, x > 0 → (y = (total_parts : ℝ) / (production_rate * x)) ↔ y = 20 / x :=
by sorry

end NUMINAMATH_CALUDE_worker_time_relationship_l718_71812


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l718_71801

/-- Given David's marks in various subjects and his average, prove his Chemistry marks -/
theorem davids_chemistry_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 95)
  (h3 : physics_marks = 82)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 93)
  (h6 : (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks : ℚ) / 5 = average_marks) :
  chemistry_marks = 97 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l718_71801


namespace NUMINAMATH_CALUDE_digits_of_2_pow_15_times_5_pow_10_l718_71814

/-- The number of digits in 2^15 * 5^10 is 12 -/
theorem digits_of_2_pow_15_times_5_pow_10 : 
  (Nat.digits 10 (2^15 * 5^10)).length = 12 := by sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_15_times_5_pow_10_l718_71814


namespace NUMINAMATH_CALUDE_orange_harvest_per_day_l718_71838

theorem orange_harvest_per_day (total_sacks : ℕ) (total_days : ℕ) (sacks_per_day : ℕ) :
  total_sacks = 24 →
  total_days = 3 →
  sacks_per_day = total_sacks / total_days →
  sacks_per_day = 8 := by
sorry

end NUMINAMATH_CALUDE_orange_harvest_per_day_l718_71838


namespace NUMINAMATH_CALUDE_complex_root_quadratic_equation_l718_71809

theorem complex_root_quadratic_equation (a b : ℝ) :
  (∃ (x : ℂ), x = 1 + Complex.I * Real.sqrt 3 ∧ a * x^2 + b * x + 1 = 0) →
  a = (1 : ℝ) / 4 ∧ b = -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_equation_l718_71809


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_20_l718_71875

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_20 :
  units_digit (factorial_sum 20) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_20_l718_71875


namespace NUMINAMATH_CALUDE_gwen_book_count_l718_71878

/-- The number of books on each shelf. -/
def books_per_shelf : ℕ := 4

/-- The number of shelves containing mystery books. -/
def mystery_shelves : ℕ := 5

/-- The number of shelves containing picture books. -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has. -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_book_count : total_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_gwen_book_count_l718_71878


namespace NUMINAMATH_CALUDE_culture_medium_composition_l718_71849

/-- Represents the composition of a culture medium --/
structure CultureMedium where
  salineWater : ℝ
  nutrientBroth : ℝ
  pureWater : ℝ

/-- The initial mixture ratio --/
def initialMixture : CultureMedium := {
  salineWater := 0.1
  nutrientBroth := 0.05
  pureWater := 0
}

/-- The required total volume of the culture medium in liters --/
def totalVolume : ℝ := 1

/-- The required percentage of pure water in the final mixture --/
def pureWaterPercentage : ℝ := 0.3

theorem culture_medium_composition :
  ∃ (final : CultureMedium),
    final.salineWater + final.nutrientBroth + final.pureWater = totalVolume ∧
    final.nutrientBroth / (final.salineWater + final.nutrientBroth) = initialMixture.nutrientBroth / (initialMixture.salineWater + initialMixture.nutrientBroth) ∧
    final.pureWater = totalVolume * pureWaterPercentage ∧
    final.nutrientBroth = 1/3 ∧
    final.pureWater = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_culture_medium_composition_l718_71849


namespace NUMINAMATH_CALUDE_group_size_is_21_l718_71826

/-- Represents the Pinterest group --/
structure PinterestGroup where
  /-- Number of people in the group --/
  people : ℕ
  /-- Average number of pins contributed per person per day --/
  pinsPerDay : ℕ
  /-- Number of pins deleted per person per week --/
  pinsDeletedPerWeek : ℕ
  /-- Initial number of pins --/
  initialPins : ℕ
  /-- Number of pins after 4 weeks --/
  pinsAfterMonth : ℕ

/-- Calculates the number of people in the Pinterest group --/
def calculateGroupSize (group : PinterestGroup) : ℕ :=
  let netPinsPerWeek := group.pinsPerDay * 7 - group.pinsDeletedPerWeek
  let pinsAddedPerPerson := netPinsPerWeek * 4
  let totalPinsAdded := group.pinsAfterMonth - group.initialPins
  totalPinsAdded / pinsAddedPerPerson

/-- Theorem stating that the number of people in the group is 21 --/
theorem group_size_is_21 (group : PinterestGroup) 
  (h1 : group.pinsPerDay = 10)
  (h2 : group.pinsDeletedPerWeek = 5)
  (h3 : group.initialPins = 1000)
  (h4 : group.pinsAfterMonth = 6600) :
  calculateGroupSize group = 21 := by
  sorry

#eval calculateGroupSize { 
  people := 0,  -- This value doesn't matter for the calculation
  pinsPerDay := 10, 
  pinsDeletedPerWeek := 5, 
  initialPins := 1000, 
  pinsAfterMonth := 6600 
}

end NUMINAMATH_CALUDE_group_size_is_21_l718_71826


namespace NUMINAMATH_CALUDE_shadow_problem_l718_71896

theorem shadow_problem (cube_edge : ℝ) (shadow_area : ℝ) (y : ℝ) : 
  cube_edge = 2 →
  shadow_area = 200 →
  y > 0 →
  y = (Real.sqrt (shadow_area + cube_edge ^ 2)) →
  ⌊1000 * y⌋ = 14280 := by
sorry

end NUMINAMATH_CALUDE_shadow_problem_l718_71896


namespace NUMINAMATH_CALUDE_quadratic_function_constraint_l718_71819

/-- Given a quadratic function f(x) = ax^2 + bx + c where a ≠ 0,
    if f(-1) = 0 and x ≤ f(x) ≤ (1/2)(x^2 + 1) for all x ∈ ℝ,
    then a = 1/4 -/
theorem quadratic_function_constraint (a b c : ℝ) (ha : a ≠ 0) :
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f (-1) = 0) →
  (∀ x : ℝ, x ≤ f x ∧ f x ≤ (1/2) * (x^2 + 1)) →
  a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_constraint_l718_71819


namespace NUMINAMATH_CALUDE_unique_solution_l718_71864

theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ 5^29 * x^15 = 2 * 10^29 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l718_71864


namespace NUMINAMATH_CALUDE_largest_intersection_is_one_l718_71865

/-- The polynomial function f(x) = x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - b

/-- The linear function g(x) = cx - d -/
def g (c d : ℝ) (x : ℝ) : ℝ := c*x - d

/-- The difference between f and g -/
def h (b c d : ℝ) (x : ℝ) : ℝ := f b x - g c d x

theorem largest_intersection_is_one (b c d : ℝ) :
  (∃ p q r : ℝ, p < q ∧ q < r ∧ 
    (∀ x : ℝ, h b c d x = 0 ↔ x = p ∨ x = q ∨ x = r)) →
  r = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_is_one_l718_71865


namespace NUMINAMATH_CALUDE_function_equivalence_l718_71834

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + 2*x + 2

-- State the theorem
theorem function_equivalence :
  (∀ x ≥ 0, f (Real.sqrt x - 1) = x + 1) →
  (∀ x ≥ -1, f x = x^2 + 2*x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l718_71834


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l718_71868

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallelLine : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem line_parallel_to_intersection_of_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : parallelLinePlane a α)
  (h2 : parallelLinePlane a β)
  (h3 : intersection α β = b) :
  parallelLine a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l718_71868


namespace NUMINAMATH_CALUDE_betty_cookie_consumption_l718_71831

/-- The number of cookies Betty eats per day -/
def cookies_per_day : ℕ := 7

/-- The number of brownies Betty eats per day -/
def brownies_per_day : ℕ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The difference between cookies and brownies after a week -/
def cookie_brownie_difference : ℕ := 36

theorem betty_cookie_consumption :
  cookies_per_day * days_in_week - brownies_per_day * days_in_week = cookie_brownie_difference :=
by sorry

end NUMINAMATH_CALUDE_betty_cookie_consumption_l718_71831


namespace NUMINAMATH_CALUDE_magic_square_solution_l718_71873

/-- Represents a 3x3 magic square with some known entries -/
structure MagicSquare where
  x : ℤ
  sum : ℤ

/-- The magic square property: all rows, columns, and diagonals sum to the same value -/
def magic_square_property (m : MagicSquare) : Prop :=
  ∃ (d e f g h : ℤ),
    m.x + 21 + 50 = m.sum ∧
    m.x + 3 + f = m.sum ∧
    50 + e + h = m.sum ∧
    m.x + d + h = m.sum ∧
    3 + d + e = m.sum ∧
    f + g + h = m.sum

/-- The theorem stating that x must be 106 in the given magic square -/
theorem magic_square_solution (m : MagicSquare) 
  (h : magic_square_property m) : m.x = 106 := by
  sorry

#check magic_square_solution

end NUMINAMATH_CALUDE_magic_square_solution_l718_71873


namespace NUMINAMATH_CALUDE_no_identical_lines_l718_71871

theorem no_identical_lines : ¬∃ (d k : ℝ), ∀ (x y : ℝ),
  (4 * x + d * y + k = 0 ↔ k * x - 3 * y + 18 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_identical_lines_l718_71871


namespace NUMINAMATH_CALUDE_feuerbach_centers_parallelogram_or_collinear_l718_71893

/-- A point in the plane -/
structure Point := (x y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point := sorry

/-- The center of the Feuerbach circle of a triangle -/
def feuerbachCenter (A B C : Point) : Point := sorry

/-- Predicate to check if four points form a parallelogram -/
def isParallelogram (P Q R S : Point) : Prop := sorry

/-- Predicate to check if four points are collinear -/
def areCollinear (P Q R S : Point) : Prop := sorry

/-- Main theorem -/
theorem feuerbach_centers_parallelogram_or_collinear (q : Quadrilateral) :
  let E := diagonalIntersection q
  let F1 := feuerbachCenter q.A q.B E
  let F2 := feuerbachCenter q.B q.C E
  let F3 := feuerbachCenter q.C q.D E
  let F4 := feuerbachCenter q.D q.A E
  isParallelogram F1 F2 F3 F4 ∨ areCollinear F1 F2 F3 F4 := by
  sorry

end NUMINAMATH_CALUDE_feuerbach_centers_parallelogram_or_collinear_l718_71893


namespace NUMINAMATH_CALUDE_commission_problem_l718_71840

/-- Calculates the total sales amount given the commission rate and commission amount -/
def calculateTotalSales (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem: Given a commission rate of 5% and a commission amount of 12.50, the total sales amount is 250 -/
theorem commission_problem :
  let commissionRate : ℚ := 5
  let commissionAmount : ℚ := 12.50
  calculateTotalSales commissionRate commissionAmount = 250 := by
  sorry

end NUMINAMATH_CALUDE_commission_problem_l718_71840


namespace NUMINAMATH_CALUDE_max_k_for_even_quadratic_min_one_l718_71862

/-- A quadratic function f(x) = x^2 + mx + n -/
def f (m n x : ℝ) : ℝ := x^2 + m*x + n

/-- The absolute value function h(x) = |f(x)| -/
def h (m n x : ℝ) : ℝ := |f m n x|

/-- Theorem: Maximum value of k for even quadratic function with minimum 1 -/
theorem max_k_for_even_quadratic_min_one :
  ∃ (k : ℝ), k = 1/2 ∧
  ∀ (m n : ℝ),
    (∀ x, f m n (-x) = f m n x) →  -- f is even
    (∀ x, f m n x ≥ 1) →           -- minimum of f is 1
    (∃ M, M ≥ k ∧
      ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → h m n x ≤ M) →  -- max of h in [-1,1] is M ≥ k
    k ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_even_quadratic_min_one_l718_71862


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l718_71889

theorem semicircle_area_ratio :
  let AB : ℝ := 10
  let AC : ℝ := 6
  let CB : ℝ := 4
  let large_semicircle_area : ℝ := (1/2) * Real.pi * (AB/2)^2
  let small_semicircle1_area : ℝ := (1/2) * Real.pi * (AC/2)^2
  let small_semicircle2_area : ℝ := (1/2) * Real.pi * (CB/2)^2
  let shaded_area : ℝ := large_semicircle_area - small_semicircle1_area - small_semicircle2_area
  let circle_area : ℝ := Real.pi * (CB/2)^2
  (shaded_area / circle_area) = (3/2) := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l718_71889


namespace NUMINAMATH_CALUDE_function_from_derivative_and_point_l718_71839

/-- Given a function f: ℝ → ℝ with f'(x) = 4x³ for all x and f(1) = -1, 
    prove that f(x) = x⁴ - 2 for all x ∈ ℝ -/
theorem function_from_derivative_and_point (f : ℝ → ℝ) 
    (h1 : ∀ x, deriv f x = 4 * x^3)
    (h2 : f 1 = -1) :
    ∀ x, f x = x^4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_from_derivative_and_point_l718_71839


namespace NUMINAMATH_CALUDE_problem_statement_l718_71884

theorem problem_statement (x : ℝ) (y : ℝ) (h_y_pos : y > 0) : 
  let A : Set ℝ := {x^2 + x + 1, -x, -x - 1}
  let B : Set ℝ := {-y, -y/2, y + 1}
  A = B → x^2 + y^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l718_71884


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l718_71816

theorem solve_exponential_equation : 
  ∃ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^(2*x) = (64 : ℝ)^6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l718_71816


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l718_71818

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of crystals incompatible with one specific herb. -/
def num_incompatible : ℕ := 3

/-- The number of viable combinations for the wizard's elixir. -/
def viable_combinations : ℕ := num_herbs * num_crystals - num_incompatible

theorem wizard_elixir_combinations :
  viable_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l718_71818


namespace NUMINAMATH_CALUDE_euler_minus_i_pi_is_real_l718_71883

theorem euler_minus_i_pi_is_real : Complex.im (Complex.exp (-Complex.I * Real.pi)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_euler_minus_i_pi_is_real_l718_71883


namespace NUMINAMATH_CALUDE_ferris_wheel_small_seat_capacity_l718_71823

/-- The number of small seats on the Ferris wheel -/
def small_seats : ℕ := 2

/-- The number of people each small seat can hold -/
def small_seat_capacity : ℕ := 14

/-- The total number of people who can ride on small seats -/
def total_small_seat_riders : ℕ := small_seats * small_seat_capacity

theorem ferris_wheel_small_seat_capacity : total_small_seat_riders = 28 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_small_seat_capacity_l718_71823


namespace NUMINAMATH_CALUDE_john_apartment_number_l718_71825

/-- Represents a skyscraper with 10 apartments on each floor. -/
structure Skyscraper where
  /-- John's apartment number -/
  john_apartment : ℕ
  /-- Mary's apartment number -/
  mary_apartment : ℕ
  /-- John's floor number -/
  john_floor : ℕ

/-- 
Given a skyscraper with 10 apartments on each floor, 
if John's floor number is equal to Mary's apartment number 
and the sum of their apartment numbers is 239, 
then John lives in apartment 217.
-/
theorem john_apartment_number (s : Skyscraper) : 
  s.john_floor = s.mary_apartment → 
  s.john_apartment + s.mary_apartment = 239 → 
  s.john_apartment = 217 := by
sorry

end NUMINAMATH_CALUDE_john_apartment_number_l718_71825


namespace NUMINAMATH_CALUDE_davids_math_marks_l718_71870

/-- Given David's marks in various subjects and his average, prove his Mathematics marks --/
theorem davids_math_marks
  (english : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (num_subjects : ℕ)
  (h1 : english = 96)
  (h2 : physics = 82)
  (h3 : chemistry = 87)
  (h4 : biology = 92)
  (h5 : average = 90.4)
  (h6 : num_subjects = 5)
  : ∃ (math : ℕ), math = 95 ∧ 
    (english + math + physics + chemistry + biology) / num_subjects = average :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l718_71870


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l718_71800

-- Define the polynomial
def f (x : ℂ) : ℂ := x^3 - x^2 - 1

-- State the theorem
theorem cubic_equation_roots :
  ∃ (a b c : ℂ), 
    (a + b + c = 1) ∧ 
    (a * b + a * c + b * c = 0) ∧ 
    (a * b * c = -1) ∧ 
    (f a = 0) ∧ (f b = 0) ∧ (f c = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l718_71800


namespace NUMINAMATH_CALUDE_population_growth_determinants_l718_71837

-- Define the factors that can potentially influence population growth
structure PopulationFactors where
  birthRate : ℝ
  deathRate : ℝ
  totalPopulation : ℝ
  socialProductionRate : ℝ
  naturalGrowthRate : ℝ

-- Define population growth pattern as a function of factors
def populationGrowthPattern (factors : PopulationFactors) : ℝ := sorry

-- Theorem stating that population growth pattern is determined by birth rate, death rate, and natural growth rate
theorem population_growth_determinants (factors : PopulationFactors) :
  populationGrowthPattern factors =
    populationGrowthPattern ⟨factors.birthRate, factors.deathRate, 0, 0, factors.naturalGrowthRate⟩ :=
by sorry

end NUMINAMATH_CALUDE_population_growth_determinants_l718_71837


namespace NUMINAMATH_CALUDE_incorrect_correct_sum_l718_71822

theorem incorrect_correct_sum : ∃ x : ℤ, 
  (x - 5 + 14 = 39) ∧ (39 + (5 * x + 14) = 203) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_correct_sum_l718_71822


namespace NUMINAMATH_CALUDE_cubic_function_c_range_l718_71803

theorem cubic_function_c_range (a b c : ℝ) :
  let f := fun x => x^3 + a*x^2 + b*x + c
  (0 < f (-1) ∧ f (-1) = f (-2) ∧ f (-2) = f (-3) ∧ f (-3) ≤ 3) →
  (6 < c ∧ c ≤ 9) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_c_range_l718_71803


namespace NUMINAMATH_CALUDE_weight_problem_l718_71846

/-- Proves that the initial number of students is 19 given the conditions of the weight problem. -/
theorem weight_problem (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ)
  (h1 : initial_avg = 15)
  (h2 : new_avg = 14.6)
  (h3 : new_student_weight = 7) :
  ∃ n : ℕ, n = 19 ∧ 
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by sorry

end NUMINAMATH_CALUDE_weight_problem_l718_71846


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l718_71854

/-- The y-intercept of the line x - 2y = 5 is -5/2 -/
theorem y_intercept_of_line (x y : ℝ) : x - 2*y = 5 → y = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l718_71854


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l718_71861

theorem maintenance_check_increase (original_time : ℝ) (increase_percent : ℝ) (new_time : ℝ) :
  original_time = 20 →
  increase_percent = 25 →
  new_time = original_time * (1 + increase_percent / 100) →
  new_time = 25 :=
by sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l718_71861


namespace NUMINAMATH_CALUDE_rejected_products_percentage_l718_71872

theorem rejected_products_percentage
  (john_reject_rate : ℝ)
  (jane_reject_rate : ℝ)
  (jane_inspect_fraction : ℝ)
  (h1 : john_reject_rate = 0.007)
  (h2 : jane_reject_rate = 0.008)
  (h3 : jane_inspect_fraction = 0.5)
  : (john_reject_rate * (1 - jane_inspect_fraction) + jane_reject_rate * jane_inspect_fraction) * 100 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_rejected_products_percentage_l718_71872


namespace NUMINAMATH_CALUDE_remaining_distance_l718_71806

/-- Calculates the remaining distance in a bike course -/
theorem remaining_distance (total_course : ℝ) (before_break : ℝ) (after_break : ℝ) :
  total_course = 10.5 ∧ before_break = 1.5 ∧ after_break = 3.73 →
  (total_course - (before_break + after_break)) * 1000 = 5270 := by
  sorry

#check remaining_distance

end NUMINAMATH_CALUDE_remaining_distance_l718_71806


namespace NUMINAMATH_CALUDE_units_digit_of_1505_odd_squares_sum_l718_71892

/-- The units digit of the sum of the squares of the first n odd, positive integers -/
def unitsDigitOfOddSquaresSum (n : ℕ) : ℕ :=
  (n / 5 * 5) % 10

theorem units_digit_of_1505_odd_squares_sum :
  unitsDigitOfOddSquaresSum 1505 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_1505_odd_squares_sum_l718_71892


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l718_71829

noncomputable def f (x : ℝ) : ℝ := (1 - 2 * x^3)^10

theorem f_derivative_at_one : 
  deriv f 1 = 60 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l718_71829


namespace NUMINAMATH_CALUDE_factorial_ratio_l718_71888

theorem factorial_ratio : (11 : ℕ).factorial / ((7 : ℕ).factorial * (4 : ℕ).factorial) = 330 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l718_71888


namespace NUMINAMATH_CALUDE_max_value_expression_l718_71821

theorem max_value_expression (a b c x : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a - x) * (x + Real.sqrt (x^2 + b^2 + c)) ≤ a^2 + b^2 + c :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l718_71821


namespace NUMINAMATH_CALUDE_estimate_fish_population_verify_fish_estimate_l718_71807

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (recapture : ℕ) (marked_recaught : ℕ) : ℕ :=
  let estimated_population := initial_catch * recapture / marked_recaught
  -- Proof that estimated_population = 750 given the conditions
  sorry

/-- Verifies the estimated fish population for the given problem. -/
theorem verify_fish_estimate : estimate_fish_population 30 50 2 = 750 := by
  -- Proof that the estimate is correct for the given values
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_verify_fish_estimate_l718_71807


namespace NUMINAMATH_CALUDE_motorist_journey_l718_71895

theorem motorist_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 6 → speed1 = 60 → speed2 = 48 → 
  (total_time / 2 * speed1) + (total_time / 2 * speed2) = 324 := by
sorry

end NUMINAMATH_CALUDE_motorist_journey_l718_71895


namespace NUMINAMATH_CALUDE_david_score_l718_71851

/-- Calculates the score of a player in a Scrabble game given the opponent's initial lead,
    the opponent's play, and the opponent's final lead. -/
def calculate_score (initial_lead : ℕ) (opponent_play : ℕ) (final_lead : ℕ) : ℕ :=
  initial_lead + opponent_play - final_lead

/-- Theorem stating that David's score in the Scrabble game is 32 points. -/
theorem david_score :
  calculate_score 22 15 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_david_score_l718_71851


namespace NUMINAMATH_CALUDE_rectangle_width_and_ratio_l718_71848

-- Define the rectangle
structure Rectangle where
  initial_length : ℝ
  new_length : ℝ
  new_perimeter : ℝ

-- Define the theorem
theorem rectangle_width_and_ratio 
  (rect : Rectangle) 
  (h1 : rect.initial_length = 8) 
  (h2 : rect.new_length = 12) 
  (h3 : rect.new_perimeter = 36) : 
  ∃ (new_width : ℝ), 
    new_width = 6 ∧ 
    new_width / rect.new_length = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_width_and_ratio_l718_71848


namespace NUMINAMATH_CALUDE_other_radius_length_l718_71853

/-- A circle is defined by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- All radii of a circle have the same length -/
axiom circle_radii_equal (c : Circle) (p q : ℝ × ℝ) :
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 →
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 →
  ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2).sqrt =
  ((q.1 - c.center.1)^2 + (q.2 - c.center.2)^2).sqrt

theorem other_radius_length (c : Circle) (p q : ℝ × ℝ) 
    (hp : (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2)
    (hq : (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2)
    (h_radius : ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2).sqrt = 2) :
    ((q.1 - c.center.1)^2 + (q.2 - c.center.2)^2).sqrt = 2 := by
  sorry

end NUMINAMATH_CALUDE_other_radius_length_l718_71853


namespace NUMINAMATH_CALUDE_august_eighth_is_saturday_l718_71876

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  mondays : Nat
  tuesdays : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (m : Month) (date : Nat) : DayOfWeek := sorry

theorem august_eighth_is_saturday 
  (m : Month) 
  (h1 : m.days = 31) 
  (h2 : m.mondays = 5) 
  (h3 : m.tuesdays = 4) : 
  dayOfWeek m 8 = DayOfWeek.Saturday := by sorry

end NUMINAMATH_CALUDE_august_eighth_is_saturday_l718_71876


namespace NUMINAMATH_CALUDE_xy_value_l718_71880

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l718_71880


namespace NUMINAMATH_CALUDE_window_width_window_width_is_six_l718_71860

/-- The width of a window in a bedroom, given the room dimensions and areas of doors and windows. -/
theorem window_width : ℝ :=
  let room_width : ℝ := 20
  let room_length : ℝ := 20
  let room_height : ℝ := 8
  let door1_width : ℝ := 3
  let door1_height : ℝ := 7
  let door2_width : ℝ := 5
  let door2_height : ℝ := 7
  let window_height : ℝ := 4
  let total_paint_area : ℝ := 560
  let total_wall_area : ℝ := 4 * room_width * room_height
  let door1_area : ℝ := door1_width * door1_height
  let door2_area : ℝ := door2_width * door2_height
  let window_width : ℝ := (total_wall_area - door1_area - door2_area - total_paint_area) / window_height
  window_width

/-- Proof that the window width is 6 feet. -/
theorem window_width_is_six : window_width = 6 := by
  sorry

end NUMINAMATH_CALUDE_window_width_window_width_is_six_l718_71860


namespace NUMINAMATH_CALUDE_expenditure_estimate_l718_71824

/-- The regression line equation for a company's expenditure (y) based on revenue (x) -/
def regression_line (x : ℝ) (a : ℝ) : ℝ := 0.8 * x + a

/-- Theorem: Given the regression line equation, when revenue is 7 billion yuan, 
    the estimated expenditure is 4.4 billion yuan -/
theorem expenditure_estimate (a : ℝ) : 
  ∃ (y : ℝ), regression_line 7 a = y ∧ y = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_estimate_l718_71824


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l718_71836

theorem negation_of_proposition (P : ℝ → Prop) :
  (∀ x : ℝ, P x) ↔ ¬(∃ x : ℝ, ¬(P x)) :=
by sorry

theorem negation_of_inequality :
  ¬(∀ x : ℝ, x^2 + 2 > 2*x) ↔ (∃ x : ℝ, x^2 + 2 ≤ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l718_71836


namespace NUMINAMATH_CALUDE_beetles_eaten_in_forest_l718_71897

/-- The number of beetles eaten in a forest each day -/
def beetles_eaten_per_day (jaguars : ℕ) (snakes_per_jaguar : ℕ) (birds_per_snake : ℕ) (beetles_per_bird : ℕ) : ℕ :=
  jaguars * snakes_per_jaguar * birds_per_snake * beetles_per_bird

/-- Theorem stating the number of beetles eaten in a specific forest scenario -/
theorem beetles_eaten_in_forest :
  beetles_eaten_per_day 6 5 3 12 = 1080 := by
  sorry

#eval beetles_eaten_per_day 6 5 3 12

end NUMINAMATH_CALUDE_beetles_eaten_in_forest_l718_71897


namespace NUMINAMATH_CALUDE_equation_one_solutions_l718_71811

theorem equation_one_solutions : 
  ∀ x : ℝ, (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l718_71811


namespace NUMINAMATH_CALUDE_fifth_polygon_exterior_angles_sum_l718_71833

/-- Represents a polygon in the sequence -/
structure Polygon where
  sides : ℕ

/-- Generates the next polygon in the sequence -/
def nextPolygon (p : Polygon) : Polygon :=
  { sides := p.sides + 2 }

/-- The sequence of polygons -/
def polygonSequence : ℕ → Polygon
  | 0 => { sides := 4 }  -- Square
  | n + 1 => nextPolygon (polygonSequence n)

/-- Sum of exterior angles of a polygon -/
def sumExteriorAngles (p : Polygon) : ℝ := 360

theorem fifth_polygon_exterior_angles_sum :
  sumExteriorAngles (polygonSequence 4) = 360 := by
  sorry

end NUMINAMATH_CALUDE_fifth_polygon_exterior_angles_sum_l718_71833


namespace NUMINAMATH_CALUDE_sin_cos_225_degrees_l718_71830

theorem sin_cos_225_degrees :
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 ∧
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_225_degrees_l718_71830


namespace NUMINAMATH_CALUDE_cricket_time_calculation_l718_71859

/-- The total time Sean and Indira played cricket together -/
def total_cricket_time (sean_daily_time : ℕ) (sean_days : ℕ) (indira_time : ℕ) : ℕ :=
  sean_daily_time * sean_days + indira_time

/-- Theorem stating the total time Sean and Indira played cricket -/
theorem cricket_time_calculation :
  total_cricket_time 50 14 812 = 1512 := by
  sorry

end NUMINAMATH_CALUDE_cricket_time_calculation_l718_71859


namespace NUMINAMATH_CALUDE_five_items_three_bags_l718_71808

/-- The number of ways to distribute n distinct items into k identical bags --/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags --/
theorem five_items_three_bags : distributionWays 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_five_items_three_bags_l718_71808


namespace NUMINAMATH_CALUDE_two_solutions_for_second_trace_l718_71828

/-- Represents a trace of a plane -/
structure Trace where
  -- Add necessary fields

/-- Represents an inclination angle -/
structure InclinationAngle where
  -- Add necessary fields

/-- Represents a plane -/
structure Plane where
  firstTrace : Trace
  firstInclinationAngle : InclinationAngle
  axisPointOutside : Bool

/-- Represents a solution for the second trace -/
structure SecondTraceSolution where
  -- Add necessary fields

/-- 
Given a plane's first trace, first inclination angle, and the condition that the axis point 
is outside the drawing frame, there exist exactly two possible solutions for the second trace.
-/
theorem two_solutions_for_second_trace (p : Plane) : 
  p.axisPointOutside → ∃! (s : Finset SecondTraceSolution), s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_for_second_trace_l718_71828


namespace NUMINAMATH_CALUDE_music_purchase_total_spent_l718_71827

/-- Represents the purchase of music albums -/
structure MusicPurchase where
  country_albums : ℕ
  pop_albums : ℕ
  country_price : ℕ
  pop_price : ℕ
  songs_per_album : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ

/-- Calculates the total cost before discounts -/
def total_cost_before_discounts (purchase : MusicPurchase) : ℕ :=
  purchase.country_albums * purchase.country_price + purchase.pop_albums * purchase.pop_price

/-- Calculates the number of discounts -/
def number_of_discounts (purchase : MusicPurchase) : ℕ :=
  (purchase.country_albums + purchase.pop_albums) / purchase.discount_threshold

/-- Calculates the total amount spent after discounts -/
def total_amount_spent (purchase : MusicPurchase) : ℕ :=
  total_cost_before_discounts purchase - number_of_discounts purchase * purchase.discount_amount

/-- Theorem: The total amount spent on music albums after applying discounts is $108 -/
theorem music_purchase_total_spent (purchase : MusicPurchase) 
  (h1 : purchase.country_albums = 4)
  (h2 : purchase.pop_albums = 5)
  (h3 : purchase.country_price = 12)
  (h4 : purchase.pop_price = 15)
  (h5 : purchase.songs_per_album = 8)
  (h6 : purchase.discount_threshold = 3)
  (h7 : purchase.discount_amount = 5) :
  total_amount_spent purchase = 108 := by
  sorry

#eval total_amount_spent {
  country_albums := 4,
  pop_albums := 5,
  country_price := 12,
  pop_price := 15,
  songs_per_album := 8,
  discount_threshold := 3,
  discount_amount := 5
}

end NUMINAMATH_CALUDE_music_purchase_total_spent_l718_71827


namespace NUMINAMATH_CALUDE_distance_between_centers_l718_71894

/-- Right triangle ABC with given side lengths -/
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  right_angle : AB^2 = BC^2 + AC^2

/-- Circle tangent to a side of the triangle and passing through the opposite vertex -/
structure TangentCircle (t : RightTriangle) where
  center : ℝ × ℝ
  tangent_point : ℝ × ℝ
  passing_point : ℝ × ℝ

/-- Configuration of the problem -/
structure TriangleCirclesConfig where
  triangle : RightTriangle
  circle_Q : TangentCircle triangle
  circle_R : TangentCircle triangle
  h_Q_tangent_BC : circle_Q.tangent_point.1 = 0 ∧ circle_Q.tangent_point.2 = 0
  h_Q_passes_A : circle_Q.passing_point.1 = 0 ∧ circle_Q.passing_point.2 = triangle.AC
  h_R_tangent_AC : circle_R.tangent_point.1 = 0 ∧ circle_R.tangent_point.2 = triangle.AC
  h_R_passes_B : circle_R.passing_point.1 = triangle.BC ∧ circle_R.passing_point.2 = 0

/-- The main theorem -/
theorem distance_between_centers (config : TriangleCirclesConfig)
  (h_triangle : config.triangle.AB = 13 ∧ config.triangle.BC = 5 ∧ config.triangle.AC = 12) :
  Real.sqrt ((config.circle_Q.center.1 - config.circle_R.center.1)^2 +
             (config.circle_Q.center.2 - config.circle_R.center.2)^2) = 33.8 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l718_71894


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l718_71858

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem z_in_fourth_quadrant (z : ℂ) 
  (h : determinant z 1 Complex.I Complex.I = 2 + Complex.I) : 
  0 < z.re ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l718_71858


namespace NUMINAMATH_CALUDE_cone_min_lateral_area_l718_71866

/-- For a cone with volume π/6, when its lateral area is minimum, 
    the tangent of the angle between the slant height and the base is √2 -/
theorem cone_min_lateral_area (r h : ℝ) : 
  r > 0 → h > 0 → 
  (1/3) * π * r^2 * h = π/6 →
  (∀ r' h', r' > 0 → h' > 0 → (1/3) * π * r'^2 * h' = π/6 → 
    π * r * (r^2 + h^2).sqrt ≤ π * r' * (r'^2 + h'^2).sqrt) →
  h / r = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cone_min_lateral_area_l718_71866


namespace NUMINAMATH_CALUDE_simplify_expression_l718_71863

theorem simplify_expression (x : ℝ) : 2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l718_71863


namespace NUMINAMATH_CALUDE_NH4I_molecular_weight_l718_71885

/-- The molecular weight of NH4I in grams per mole -/
def molecular_weight_NH4I : ℝ := 145

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 9

/-- The total weight in grams for the given number of moles -/
def given_total_weight : ℝ := 1305

theorem NH4I_molecular_weight :
  molecular_weight_NH4I = given_total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_NH4I_molecular_weight_l718_71885


namespace NUMINAMATH_CALUDE_students_walking_home_l718_71891

theorem students_walking_home (total : ℚ) (bus : ℚ) (auto : ℚ) (bike : ℚ) (walk : ℚ) : 
  bus = 1/3 * total → auto = 1/5 * total → bike = 1/15 * total → 
  walk = total - (bus + auto + bike) →
  walk = 2/5 * total :=
sorry

end NUMINAMATH_CALUDE_students_walking_home_l718_71891


namespace NUMINAMATH_CALUDE_gcf_60_90_l718_71844

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_90_l718_71844


namespace NUMINAMATH_CALUDE_divisibility_implies_difference_one_l718_71877

theorem divisibility_implies_difference_one
  (a b c d : ℕ)
  (h1 : (a * b - c * d) ∣ a)
  (h2 : (a * b - c * d) ∣ b)
  (h3 : (a * b - c * d) ∣ c)
  (h4 : (a * b - c * d) ∣ d) :
  a * b - c * d = 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_difference_one_l718_71877


namespace NUMINAMATH_CALUDE_whole_milk_fat_percentage_l718_71898

/-- The percentage of fat in low-fat milk -/
def low_fat_milk_percentage : ℝ := 3

/-- The percentage difference between low-fat and semi-skimmed milk -/
def low_fat_semi_skimmed_difference : ℝ := 25

/-- The percentage difference between semi-skimmed and whole milk -/
def semi_skimmed_whole_difference : ℝ := 20

/-- The percentage of fat in whole milk -/
def whole_milk_percentage : ℝ := 5

theorem whole_milk_fat_percentage :
  (low_fat_milk_percentage / (1 - low_fat_semi_skimmed_difference / 100)) / (1 - semi_skimmed_whole_difference / 100) = whole_milk_percentage := by
  sorry

end NUMINAMATH_CALUDE_whole_milk_fat_percentage_l718_71898


namespace NUMINAMATH_CALUDE_train_speed_calculation_l718_71869

/-- Calculates the speed of a train given the parameters of a passing goods train --/
theorem train_speed_calculation (goods_train_speed : ℝ) (goods_train_length : ℝ) (passing_time : ℝ) :
  goods_train_speed = 50.4 →
  goods_train_length = 240 →
  passing_time = 10 →
  ∃ (man_train_speed : ℝ), man_train_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l718_71869


namespace NUMINAMATH_CALUDE_candy_total_l718_71813

def candy_problem (tabitha stan : ℕ) : Prop :=
  ∃ (julie carlos veronica benjamin : ℕ),
    tabitha = 22 ∧
    stan = 16 ∧
    julie = tabitha / 2 ∧
    carlos = 2 * stan ∧
    veronica = julie + stan ∧
    benjamin = (tabitha + carlos) / 2 + 9 ∧
    tabitha + stan + julie + carlos + veronica + benjamin = 144

theorem candy_total : candy_problem 22 16 := by
  sorry

end NUMINAMATH_CALUDE_candy_total_l718_71813


namespace NUMINAMATH_CALUDE_price_reduction_equation_l718_71847

/-- Proves the correct equation for a price reduction scenario -/
theorem price_reduction_equation (x : ℝ) : 
  (∃ (original_price final_price : ℝ),
    original_price = 200 ∧ 
    final_price = 162 ∧ 
    final_price = original_price * (1 - x)^2) ↔ 
  200 * (1 - x)^2 = 162 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l718_71847


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_five_in_range_l718_71815

theorem unique_square_divisible_by_five_in_range : 
  ∃! x : ℕ, x^2 = x ∧ x % 5 = 0 ∧ 50 < x^2 ∧ x^2 < 120 :=
by sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_five_in_range_l718_71815


namespace NUMINAMATH_CALUDE_train_speed_problem_l718_71842

/-- Proves that given two trains of specified lengths running in opposite directions,
    where one train has a known speed and the time to cross each other is known,
    the speed of the other train can be determined. -/
theorem train_speed_problem (length1 length2 known_speed crossing_time : ℝ) :
  length1 = 140 ∧
  length2 = 190 ∧
  known_speed = 40 ∧
  crossing_time = 11.879049676025918 →
  ∃ other_speed : ℝ,
    other_speed = 60 ∧
    (length1 + length2) / crossing_time * 3.6 = known_speed + other_speed :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l718_71842


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l718_71810

noncomputable def P : (ℝ → ℝ → ℝ) → Prop :=
  λ p => ∀ x y : ℝ, p (x^2) (y^2) = p ((x + y)^2 / 2) ((x - y)^2 / 2)

theorem polynomial_functional_equation :
  ∀ p : ℝ → ℝ → ℝ, P p ↔ ∃ q : ℝ → ℝ → ℝ, ∀ x y : ℝ, p x y = q (x + y) (x * y * (x - y)^2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l718_71810


namespace NUMINAMATH_CALUDE_laborer_average_salary_l718_71820

/-- Calculates the average monthly salary of laborers in a factory --/
theorem laborer_average_salary
  (total_workers : ℕ)
  (total_average_salary : ℚ)
  (num_supervisors : ℕ)
  (supervisor_average_salary : ℚ)
  (num_laborers : ℕ)
  (h_total_workers : total_workers = num_supervisors + num_laborers)
  (h_num_supervisors : num_supervisors = 6)
  (h_num_laborers : num_laborers = 42)
  (h_total_average_salary : total_average_salary = 1250)
  (h_supervisor_average_salary : supervisor_average_salary = 2450) :
  let laborer_total_salary := total_workers * total_average_salary - num_supervisors * supervisor_average_salary
  (laborer_total_salary / num_laborers) = 1078.57 := by
sorry

#eval (48 * 1250 - 6 * 2450) / 42

end NUMINAMATH_CALUDE_laborer_average_salary_l718_71820


namespace NUMINAMATH_CALUDE_banana_arrangements_l718_71882

def word := "BANANA"

def letter_count : Nat := word.length

def b_count : Nat := (word.toList.filter (· == 'B')).length
def n_count : Nat := (word.toList.filter (· == 'N')).length
def a_count : Nat := (word.toList.filter (· == 'A')).length

def distinct_arrangements : Nat := letter_count.factorial / (b_count.factorial * n_count.factorial * a_count.factorial)

theorem banana_arrangements :
  letter_count = 6 ∧ b_count = 1 ∧ n_count = 2 ∧ a_count = 3 →
  distinct_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l718_71882


namespace NUMINAMATH_CALUDE_books_bound_calculation_remaining_paper_condition_l718_71881

/-- Represents the number of books bound in a bookbinding workshop. -/
def books_bound (initial_white : ℕ) (initial_colored : ℕ) : ℕ :=
  initial_white - (initial_colored - initial_white)

/-- Theorem stating the number of books bound given the initial quantities and conditions. -/
theorem books_bound_calculation :
  let initial_white := 92
  let initial_colored := 135
  books_bound initial_white initial_colored = 178 :=
by
  sorry

/-- Theorem verifying the remaining paper condition after binding. -/
theorem remaining_paper_condition (initial_white initial_colored : ℕ) :
  let bound := books_bound initial_white initial_colored
  initial_white - bound = (initial_colored - bound) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_books_bound_calculation_remaining_paper_condition_l718_71881


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l718_71802

-- Define the quadrilateral properties
def diagonal : ℝ := 30
def offset1 : ℝ := 6
def area : ℝ := 225

-- Theorem to prove
theorem quadrilateral_offset (offset2 : ℝ) :
  area = (diagonal * (offset1 + offset2)) / 2 → offset2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l718_71802


namespace NUMINAMATH_CALUDE_ratio_of_numbers_with_sum_gcd_equal_lcm_l718_71843

theorem ratio_of_numbers_with_sum_gcd_equal_lcm (A B : ℕ) (h1 : A ≥ B) :
  A + B + Nat.gcd A B = Nat.lcm A B → (A : ℚ) / B = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_with_sum_gcd_equal_lcm_l718_71843


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equality_l718_71852

theorem arithmetic_sequence_equality (n : ℕ) (a b : Fin n → ℕ) :
  n ≥ 2018 →
  (∀ i : Fin n, a i ≤ 5 * n ∧ b i ≤ 5 * n) →
  (∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ b i ≠ b j) →
  (∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = d * (i - j)) →
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equality_l718_71852


namespace NUMINAMATH_CALUDE_distance_is_90km_l718_71887

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance traveled downstream is 90 km -/
theorem distance_is_90km (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ)
  (h1 : boat_speed = 25)
  (h2 : stream_speed = 5)
  (h3 : time = 3) :
  distance_downstream boat_speed stream_speed time = 90 := by
  sorry

#eval distance_downstream 25 5 3

end NUMINAMATH_CALUDE_distance_is_90km_l718_71887


namespace NUMINAMATH_CALUDE_quadratic_roots_l718_71855

theorem quadratic_roots (a b : ℝ) (h1 : a * b ≠ 0) 
  (h2 : a^2 + 2*b*a + a = 0) (h3 : b^2 + 2*b*b + a = 0) : 
  a = -3 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l718_71855


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l718_71857

/-- Represents the number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes is 5 -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 5 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l718_71857


namespace NUMINAMATH_CALUDE_karen_group_size_l718_71835

/-- Proves that if Zack tutors students in groups of 14, and both Zack and Karen tutor
    the same total number of 70 students, then Karen must also tutor students in groups of 14. -/
theorem karen_group_size (zack_group_size : ℕ) (total_students : ℕ) (karen_group_size : ℕ) :
  zack_group_size = 14 →
  total_students = 70 →
  total_students % zack_group_size = 0 →
  total_students % karen_group_size = 0 →
  total_students / zack_group_size = total_students / karen_group_size →
  karen_group_size = 14 := by
sorry

end NUMINAMATH_CALUDE_karen_group_size_l718_71835


namespace NUMINAMATH_CALUDE_intersection_of_lines_l718_71841

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) :
  A = (5, -6, 8) →
  B = (15, -16, 13) →
  C = (1, 4, -5) →
  D = (3, -4, 11) →
  ∃ t s : ℝ,
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (1 + 2*s, 4 - 8*s, -5 + 16*s) ∧
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (-10/3, 14/3, -1/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l718_71841


namespace NUMINAMATH_CALUDE_returning_players_l718_71856

theorem returning_players (new_players : ℕ) (group_size : ℕ) (total_groups : ℕ) : 
  new_players = 48 → group_size = 6 → total_groups = 9 → 
  (total_groups * group_size) - new_players = 6 := by
  sorry

end NUMINAMATH_CALUDE_returning_players_l718_71856


namespace NUMINAMATH_CALUDE_work_day_ends_at_target_time_l718_71817

-- Define the start time, lunch time, and total work hours
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes
def lunch_time : Nat := 13 * 60  -- 1:00 PM in minutes
def total_work_minutes : Nat := 9 * 60  -- 9 hours in minutes
def lunch_break_minutes : Nat := 30

-- Define the end time we want to prove
def target_end_time : Nat := 17 * 60 + 30  -- 5:30 PM in minutes

-- Theorem to prove
theorem work_day_ends_at_target_time :
  start_time + total_work_minutes + lunch_break_minutes = target_end_time := by
  sorry


end NUMINAMATH_CALUDE_work_day_ends_at_target_time_l718_71817


namespace NUMINAMATH_CALUDE_total_weight_of_CaO_l718_71867

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of moles of CaO -/
def moles_CaO : ℝ := 7

/-- The molecular weight of CaO in g/mol -/
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O

/-- The total weight of CaO in grams -/
def total_weight_CaO : ℝ := molecular_weight_CaO * moles_CaO

/-- Theorem stating the total weight of 7 moles of CaO -/
theorem total_weight_of_CaO : total_weight_CaO = 392.56 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_CaO_l718_71867


namespace NUMINAMATH_CALUDE_number_decrease_theorem_l718_71832

theorem number_decrease_theorem :
  (∃ (N k a x : ℕ), k ≥ 1 ∧ 1 ≤ a ∧ a ≤ 9 ∧ x < 10^k ∧ N = 10^k * a + x ∧ N = 57 * x) ∧
  (¬ ∃ (N k a x : ℕ), k ≥ 1 ∧ 1 ≤ a ∧ a ≤ 9 ∧ x < 10^k ∧ N = 10^k * a + x ∧ N = 58 * x) :=
by sorry

end NUMINAMATH_CALUDE_number_decrease_theorem_l718_71832


namespace NUMINAMATH_CALUDE_waste_bread_price_is_correct_l718_71886

/-- Calculates the price per pound of wasted bread products given the following conditions:
  * Minimum wage is $8/hour
  * 20 pounds of meat wasted at $5/pound
  * 15 pounds of fruits and vegetables wasted at $4/pound
  * 60 pounds of bread products wasted (price unknown)
  * 10 hours of time-and-a-half pay for janitorial staff (normal pay $10/hour)
  * Total work hours to pay for everything is 50 hours
-/
def wasteBreadPrice (
  minWage : ℚ)
  (meatWeight : ℚ)
  (meatPrice : ℚ)
  (fruitVegWeight : ℚ)
  (fruitVegPrice : ℚ)
  (breadWeight : ℚ)
  (janitorHours : ℚ)
  (janitorWage : ℚ)
  (totalWorkHours : ℚ) : ℚ :=
  let meatCost := meatWeight * meatPrice
  let fruitVegCost := fruitVegWeight * fruitVegPrice
  let janitorCost := janitorHours * (janitorWage * 1.5)
  let totalEarnings := totalWorkHours * minWage
  let breadCost := totalEarnings - (meatCost + fruitVegCost + janitorCost)
  breadCost / breadWeight

theorem waste_bread_price_is_correct :
  wasteBreadPrice 8 20 5 15 4 60 10 10 50 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_waste_bread_price_is_correct_l718_71886
