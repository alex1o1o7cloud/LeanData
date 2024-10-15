import Mathlib

namespace NUMINAMATH_CALUDE_prob_even_sum_first_15_primes_l3080_308003

/-- The number of prime numbers we're considering -/
def n : ℕ := 15

/-- The number of prime numbers we're selecting -/
def k : ℕ := 5

/-- The number of odd primes among the first n primes -/
def odd_primes : ℕ := n - 1

/-- The probability of selecting k primes from n primes such that their sum is even -/
def prob_even_sum (n k odd_primes : ℕ) : ℚ :=
  (Nat.choose odd_primes k + Nat.choose odd_primes (k - 3)) / Nat.choose n k

theorem prob_even_sum_first_15_primes : 
  prob_even_sum n k odd_primes = 2093 / 3003 :=
sorry

end NUMINAMATH_CALUDE_prob_even_sum_first_15_primes_l3080_308003


namespace NUMINAMATH_CALUDE_altitude_df_length_l3080_308091

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  /-- Length of side DC -/
  dc : ℝ
  /-- Length of segment EB -/
  eb : ℝ
  /-- Length of altitude DE -/
  de : ℝ
  /-- Ensures dc is positive -/
  dc_pos : dc > 0
  /-- Ensures eb is positive -/
  eb_pos : eb > 0
  /-- Ensures de is positive -/
  de_pos : de > 0
  /-- Ensures eb is less than dc (as EB is part of AB which is equal to DC) -/
  eb_lt_dc : eb < dc

/-- Theorem stating that under the given conditions, DF = 5 -/
theorem altitude_df_length (p : Parallelogram) (h1 : p.dc = 15) (h2 : p.eb = 3) (h3 : p.de = 5) :
  ∃ df : ℝ, df = 5 ∧ df > 0 := by
  sorry

end NUMINAMATH_CALUDE_altitude_df_length_l3080_308091


namespace NUMINAMATH_CALUDE_product_digits_l3080_308089

def a : ℕ := 7123456789
def b : ℕ := 23567891234

theorem product_digits : (String.length (toString (a * b))) = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_digits_l3080_308089


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3080_308085

/-- The length of the diagonal of a rectangle with length 20√5 and width 10√3 is 10√23 -/
theorem rectangle_diagonal (length width diagonal : ℝ) 
  (h_length : length = 20 * Real.sqrt 5)
  (h_width : width = 10 * Real.sqrt 3)
  (h_diagonal : diagonal^2 = length^2 + width^2) : 
  diagonal = 10 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3080_308085


namespace NUMINAMATH_CALUDE_complex_product_real_l3080_308072

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 + a * Complex.I
  let z₂ : ℂ := a - 3 * Complex.I
  (z₁ * z₂).im = 0 ↔ a = 3 ∨ a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l3080_308072


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l3080_308059

theorem circle_radius_from_area (A : Real) (r : Real) :
  A = Real.pi * r^2 → A = 64 * Real.pi → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l3080_308059


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3080_308077

theorem negative_fraction_comparison : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3080_308077


namespace NUMINAMATH_CALUDE_remaining_amount_is_14_90_l3080_308041

-- Define the initial amount and item costs
def initial_amount : ℚ := 78
def kite_cost : ℚ := 8
def frisbee_cost : ℚ := 9
def roller_skates_cost : ℚ := 15
def roller_skates_discount : ℚ := 0.1
def lego_cost : ℚ := 25
def lego_coupon : ℚ := 5
def puzzle_cost : ℚ := 12
def puzzle_tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining amount
def remaining_amount : ℚ :=
  initial_amount -
  (kite_cost +
   frisbee_cost +
   (roller_skates_cost * (1 - roller_skates_discount)) +
   (lego_cost - lego_coupon) +
   (puzzle_cost * (1 + puzzle_tax_rate)))

-- Theorem stating that the remaining amount is $14.90
theorem remaining_amount_is_14_90 :
  remaining_amount = 14.90 := by sorry

end NUMINAMATH_CALUDE_remaining_amount_is_14_90_l3080_308041


namespace NUMINAMATH_CALUDE_percentage_of_number_l3080_308040

theorem percentage_of_number (x : ℝ) (y : ℝ) (z : ℝ) : 
  x = (y / 100) * z → x = 120 ∧ y = 150 ∧ z = 80 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_number_l3080_308040


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3080_308032

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ
  f : ℝ → ℝ
  f_def : ∀ x, f x = x^2 + b*x + c
  f_sin_nonneg : ∀ α, f (Real.sin α) ≥ 0
  f_cos_nonpos : ∀ β, f (2 + Real.cos β) ≤ 0

theorem quadratic_function_properties (qf : QuadraticFunction) :
  qf.f 1 = 0 ∧ 
  qf.c ≥ 3 ∧
  (∀ α, qf.f (Real.sin α) ≤ 8 → qf.f = fun x ↦ x^2 - 4*x + 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3080_308032


namespace NUMINAMATH_CALUDE_angela_insects_l3080_308045

theorem angela_insects (dean_insects : ℕ) (jacob_insects : ℕ) (angela_insects : ℕ)
  (h1 : dean_insects = 30)
  (h2 : jacob_insects = 5 * dean_insects)
  (h3 : angela_insects = jacob_insects / 2) :
  angela_insects = 75 := by
  sorry

end NUMINAMATH_CALUDE_angela_insects_l3080_308045


namespace NUMINAMATH_CALUDE_no_double_application_plus_one_l3080_308039

theorem no_double_application_plus_one : 
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1 := by
sorry

end NUMINAMATH_CALUDE_no_double_application_plus_one_l3080_308039


namespace NUMINAMATH_CALUDE_craig_remaining_apples_l3080_308046

/-- Theorem: Craig's remaining apples after sharing -/
theorem craig_remaining_apples (initial_apples shared_apples : ℕ) 
  (h1 : initial_apples = 20)
  (h2 : shared_apples = 7) :
  initial_apples - shared_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_craig_remaining_apples_l3080_308046


namespace NUMINAMATH_CALUDE_word_problems_count_l3080_308094

theorem word_problems_count (total_questions : ℕ) 
                             (addition_subtraction_problems : ℕ) 
                             (steve_answered : ℕ) : 
  total_questions = 45 →
  addition_subtraction_problems = 28 →
  steve_answered = 38 →
  total_questions - steve_answered = 7 →
  total_questions - addition_subtraction_problems = 17 := by
  sorry

end NUMINAMATH_CALUDE_word_problems_count_l3080_308094


namespace NUMINAMATH_CALUDE_square_difference_ratio_l3080_308011

theorem square_difference_ratio : 
  (1630^2 - 1623^2) / (1640^2 - 1613^2) = 7/27 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l3080_308011


namespace NUMINAMATH_CALUDE_leo_assignment_time_theorem_l3080_308064

theorem leo_assignment_time_theorem :
  ∀ (first_part second_part third_part first_break second_break total_time : ℕ),
    first_part = 25 →
    second_part = 2 * first_part →
    first_break = 10 →
    second_break = 15 →
    total_time = 150 →
    total_time = first_part + second_part + third_part + first_break + second_break →
    third_part = 50 := by
  sorry

end NUMINAMATH_CALUDE_leo_assignment_time_theorem_l3080_308064


namespace NUMINAMATH_CALUDE_largest_inscribable_rectangle_area_l3080_308084

/-- The area of the largest inscribable rectangle between two congruent equilateral triangles
    within a rectangle of width 8 and length 12 -/
theorem largest_inscribable_rectangle_area
  (width : ℝ) (length : ℝ)
  (h_width : width = 8)
  (h_length : length = 12)
  (triangle_side : ℝ)
  (h_triangle_side : triangle_side = 8 * Real.sqrt 3 / 3)
  (inscribed_height : ℝ)
  (h_inscribed_height : inscribed_height = width - triangle_side)
  : inscribed_height * length = 96 - 32 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribable_rectangle_area_l3080_308084


namespace NUMINAMATH_CALUDE_min_value_not_neg_half_l3080_308056

open Real

theorem min_value_not_neg_half (g : ℝ → ℝ) :
  (∀ x, g x = -Real.sqrt 3 * Real.sin (2 * x) + 1) →
  ¬(∃ x ∈ Set.Icc (π / 6) (π / 2), ∀ y ∈ Set.Icc (π / 6) (π / 2), g x ≤ g y ∧ g x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_not_neg_half_l3080_308056


namespace NUMINAMATH_CALUDE_paper_folding_l3080_308023

theorem paper_folding (s : ℝ) (x : ℝ) :
  s^2 = 18 →
  (1/2) * x^2 = 18 - x^2 →
  ∃ d : ℝ, d = 2 * Real.sqrt 6 ∧ d^2 = 2 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_paper_folding_l3080_308023


namespace NUMINAMATH_CALUDE_equation_solutions_l3080_308018

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2) ∧
  (∀ x : ℝ, (x - 1)^3 = -8 ↔ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3080_308018


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l3080_308078

/-- The nature of roots of a quadratic equation based on parameters a and b -/
theorem quadratic_roots_nature (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x => (a^2 + b^2) * x^2 + 4 * a * b * x + 2 * a * b
  (a = b → (∃! x, f x = 0)) ∧
  (a ≠ b → a * b > 0 → ∀ x, f x ≠ 0) ∧
  (a ≠ b → a * b < 0 → ∃ x y, x ≠ y ∧ f x = 0 ∧ f y = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_nature_l3080_308078


namespace NUMINAMATH_CALUDE_calculate_speed_l3080_308015

/-- Given two people moving in opposite directions, calculate the speed of one person given the speed of the other and their total distance after a certain time. -/
theorem calculate_speed (riya_speed priya_speed : ℝ) (time : ℝ) (total_distance : ℝ) : 
  riya_speed = 24 →
  time = 0.75 →
  total_distance = 44.25 →
  priya_speed * time + riya_speed * time = total_distance →
  priya_speed = 35 := by
sorry

end NUMINAMATH_CALUDE_calculate_speed_l3080_308015


namespace NUMINAMATH_CALUDE_max_books_on_shelf_l3080_308028

theorem max_books_on_shelf (n : ℕ) (s₁ s₂ S : ℕ) : 
  (S + s₁ ≥ (n - 2) / 2) →
  (S + s₂ < (n - 2) / 3) →
  (n ≤ 12) :=
sorry

end NUMINAMATH_CALUDE_max_books_on_shelf_l3080_308028


namespace NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l3080_308031

/-- Two lines are parallel if their slopes are equal and they are not identical. -/
def are_parallel (m1 a1 b1 c1 m2 a2 b2 c2 : ℝ) : Prop :=
  m1 = m2 ∧ (a1, b1, c1) ≠ (a2, b2, c2)

/-- The condition for the given lines to be parallel. -/
def parallel_condition (a : ℝ) : Prop :=
  are_parallel (-a/2) a 2 1 (-3/(a-1)) 3 (a-1) (-2)

theorem a_equals_3_sufficient_not_necessary :
  (∃ a, a ≠ 3 ∧ parallel_condition a) ∧
  (parallel_condition 3) :=
sorry

end NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l3080_308031


namespace NUMINAMATH_CALUDE_x_to_y_value_l3080_308074

theorem x_to_y_value (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 1) = 0) : x^y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_value_l3080_308074


namespace NUMINAMATH_CALUDE_max_points_without_equilateral_triangle_l3080_308038

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the set of 10 points: vertices, centroid, and trisection points -/
def TrianglePoints (t : EquilateralTriangle) : Finset Point :=
  sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (p1 p2 p3 : Point) : Prop :=
  sorry

/-- The main theorem -/
theorem max_points_without_equilateral_triangle (t : EquilateralTriangle) :
  ∃ (s : Finset Point), s ⊆ TrianglePoints t ∧ s.card = 6 ∧
  ∀ (p1 p2 p3 : Point), p1 ∈ s → p2 ∈ s → p3 ∈ s → ¬(isEquilateral p1 p2 p3) ∧
  ∀ (s' : Finset Point), s' ⊆ TrianglePoints t →
    (∀ (p1 p2 p3 : Point), p1 ∈ s' → p2 ∈ s' → p3 ∈ s' → ¬(isEquilateral p1 p2 p3)) →
    s'.card ≤ 6 :=
  sorry

end NUMINAMATH_CALUDE_max_points_without_equilateral_triangle_l3080_308038


namespace NUMINAMATH_CALUDE_solve_complex_equation_l3080_308043

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z * (1 + i) = 2 + i

-- Theorem statement
theorem solve_complex_equation :
  ∀ z : ℂ, equation z → z = (3/2 : ℝ) - (1/2 : ℝ) * i :=
by
  sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l3080_308043


namespace NUMINAMATH_CALUDE_remainder_problem_l3080_308014

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 35 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3080_308014


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3080_308049

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, a * x^2 + b * x + c < 0) : 
  b / a < c / a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3080_308049


namespace NUMINAMATH_CALUDE_math_preference_gender_related_l3080_308069

/-- Represents the survey data and critical value for the chi-square test -/
structure SurveyData where
  total_students : Nat
  male_percentage : Rat
  total_math_liking : Nat
  female_math_liking : Nat
  critical_value : Rat

/-- Calculates the chi-square statistic for the given survey data -/
def calculate_chi_square (data : SurveyData) : Rat :=
  sorry

/-- Theorem stating that the calculated chi-square value exceeds the critical value -/
theorem math_preference_gender_related (data : SurveyData) :
  data.total_students = 100 ∧
  data.male_percentage = 55/100 ∧
  data.total_math_liking = 40 ∧
  data.female_math_liking = 20 ∧
  data.critical_value = 7879/1000 →
  calculate_chi_square data > data.critical_value :=
sorry

end NUMINAMATH_CALUDE_math_preference_gender_related_l3080_308069


namespace NUMINAMATH_CALUDE_zeta_sum_seventh_power_l3080_308076

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 58 := by
  sorry

end NUMINAMATH_CALUDE_zeta_sum_seventh_power_l3080_308076


namespace NUMINAMATH_CALUDE_initial_workers_correct_l3080_308070

/-- Represents the initial number of workers -/
def initial_workers : ℕ := 120

/-- Represents the total number of days to complete the wall -/
def total_days : ℕ := 50

/-- Represents the number of days after which progress is measured -/
def progress_days : ℕ := 25

/-- Represents the fraction of work completed after progress_days -/
def work_completed : ℚ := 2/5

/-- Represents the additional workers needed to complete on time -/
def additional_workers : ℕ := 30

/-- Proves that the initial number of workers is correct given the conditions -/
theorem initial_workers_correct :
  initial_workers * total_days = (initial_workers + additional_workers) * 
    (total_days * work_completed + progress_days * (1 - work_completed)) :=
by sorry

end NUMINAMATH_CALUDE_initial_workers_correct_l3080_308070


namespace NUMINAMATH_CALUDE_replaced_person_age_l3080_308051

/-- Represents a group of people with their ages -/
structure AgeGroup where
  size : ℕ
  average_age : ℝ

/-- Theorem stating the age of the replaced person -/
theorem replaced_person_age (group : AgeGroup) (h1 : group.size = 10) 
  (h2 : ∃ (new_average : ℝ), new_average = group.average_age - 3) 
  (h3 : ∃ (new_person_age : ℝ), new_person_age = 18) : 
  ∃ (replaced_age : ℝ), replaced_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_age_l3080_308051


namespace NUMINAMATH_CALUDE_necessary_condition_for_124_l3080_308099

/-- A line in the form y = (m/n)x - 1/n -/
structure Line where
  m : ℝ
  n : ℝ
  n_nonzero : n ≠ 0

/-- Predicate for a line passing through the first, second, and fourth quadrants -/
def passes_through_124 (l : Line) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₄ y₄ : ℝ),
    x₁ > 0 ∧ y₁ > 0 ∧  -- First quadrant
    x₂ < 0 ∧ y₂ > 0 ∧  -- Second quadrant
    x₄ > 0 ∧ y₄ < 0 ∧  -- Fourth quadrant
    y₁ = (l.m / l.n) * x₁ - 1 / l.n ∧
    y₂ = (l.m / l.n) * x₂ - 1 / l.n ∧
    y₄ = (l.m / l.n) * x₄ - 1 / l.n

/-- Theorem stating the necessary condition -/
theorem necessary_condition_for_124 (l : Line) :
  passes_through_124 l → l.m > 0 ∧ l.n < 0 :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_124_l3080_308099


namespace NUMINAMATH_CALUDE_abs_S_eq_1024_l3080_308022

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the expression S
def S : ℂ := (1 + i)^18 - (1 - i)^18

-- Theorem statement
theorem abs_S_eq_1024 : Complex.abs S = 1024 := by
  sorry

end NUMINAMATH_CALUDE_abs_S_eq_1024_l3080_308022


namespace NUMINAMATH_CALUDE_system_solution_l3080_308036

theorem system_solution :
  ∀ (x y z : ℝ),
    (x + 1) * y * z = 12 ∧
    (y + 1) * z * x = 4 ∧
    (z + 1) * x * y = 4 →
    ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3080_308036


namespace NUMINAMATH_CALUDE_meatballs_on_plate_l3080_308008

theorem meatballs_on_plate (num_sons : ℕ) (fraction_eaten : ℚ) (meatballs_left : ℕ) : 
  num_sons = 3 → 
  fraction_eaten = 2/3 → 
  meatballs_left = 3 → 
  ∃ (initial_meatballs : ℕ), 
    initial_meatballs = 3 ∧ 
    (num_sons : ℚ) * ((1 : ℚ) - fraction_eaten) * initial_meatballs = meatballs_left :=
by sorry

end NUMINAMATH_CALUDE_meatballs_on_plate_l3080_308008


namespace NUMINAMATH_CALUDE_f_2017_value_l3080_308066

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2017_value (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_value : f (-1) = 6) :
  f 2017 = -6 := by
sorry

end NUMINAMATH_CALUDE_f_2017_value_l3080_308066


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_sum_less_than_1000_max_consecutive_integers_1000_l3080_308093

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 44 ↔ n * (n + 1) ≤ 2000 := by
  sorry

theorem largest_n_sum_less_than_1000 : ∀ k > 44, k * (k + 1) > 2000 := by
  sorry

theorem max_consecutive_integers_1000 : 
  (∀ m ≤ 44, m * (m + 1) ≤ 2000) ∧
  (∀ k > 44, k * (k + 1) > 2000) := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_sum_less_than_1000_max_consecutive_integers_1000_l3080_308093


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3080_308055

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 3, 6}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem to prove
theorem intersection_complement_equals : A ∩ (U \ B) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3080_308055


namespace NUMINAMATH_CALUDE_pizza_toppings_l3080_308019

theorem pizza_toppings (total_slices ham_slices olive_slices : ℕ) 
  (h_total : total_slices = 16)
  (h_ham : ham_slices = 8)
  (h_olive : olive_slices = 12)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ ham_slices ∨ slice ≤ olive_slices)) :
  ∃ both : ℕ, both = ham_slices + olive_slices - total_slices :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3080_308019


namespace NUMINAMATH_CALUDE_binomial_probability_two_l3080_308053

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ  -- The random variable

/-- The probability mass function for a binomial distribution -/
def binomialProbability (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- The theorem stating that P(X=2) = 80/243 for X ~ B(6, 1/3) -/
theorem binomial_probability_two (X : BinomialDistribution 6 (1/3)) :
  binomialProbability 6 (1/3) 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_l3080_308053


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3080_308063

theorem complex_fraction_equality (a : ℝ) : (a + Complex.I) / (2 - Complex.I) = 1 + Complex.I → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3080_308063


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l3080_308082

theorem sqrt_three_times_sqrt_twelve : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l3080_308082


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_equality_l3080_308061

theorem quadratic_inequality_implies_equality (x : ℝ) :
  -2 * x^2 + 5 * x - 2 > 0 →
  Real.sqrt (4 * x^2 - 4 * x + 1) + 2 * abs (x - 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_equality_l3080_308061


namespace NUMINAMATH_CALUDE_original_salary_proof_l3080_308044

/-- Given a 6% raise resulting in a new salary of $530, prove that the original salary was $500. -/
theorem original_salary_proof (original_salary : ℝ) : 
  original_salary * 1.06 = 530 → original_salary = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_proof_l3080_308044


namespace NUMINAMATH_CALUDE_divisibility_problem_l3080_308026

theorem divisibility_problem (a b c d : ℤ) 
  (h : (a^4 + b^4 + c^4 + d^4) % 5 = 0) : 
  625 ∣ (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3080_308026


namespace NUMINAMATH_CALUDE_parallel_planes_lines_l3080_308048

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the contained relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (line_parallel : Line → Line → Prop)

-- Define the skew relation for lines
variable (line_skew : Line → Line → Prop)

-- Define the intersection relation for lines
variable (line_intersect : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_lines
  (α β : Plane) (a b : Line)
  (h_parallel : plane_parallel α β)
  (h_a_in_α : line_in_plane a α)
  (h_b_in_β : line_in_plane b β) :
  (¬ line_intersect a b) ∧
  (line_parallel a b ∨ line_skew a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_lines_l3080_308048


namespace NUMINAMATH_CALUDE_katies_cupcakes_l3080_308017

theorem katies_cupcakes (cupcakes cookies left_over sold : ℕ) :
  cookies = 5 →
  left_over = 8 →
  sold = 4 →
  cupcakes + cookies = left_over + sold →
  cupcakes = 7 := by
sorry

end NUMINAMATH_CALUDE_katies_cupcakes_l3080_308017


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l3080_308092

theorem money_left_after_purchase (initial_amount spent_amount : ℕ) : 
  initial_amount = 90 → spent_amount = 78 → initial_amount - spent_amount = 12 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l3080_308092


namespace NUMINAMATH_CALUDE_initial_courses_is_three_l3080_308035

/-- Represents the wall construction problem -/
def WallProblem (initial_courses : ℕ) : Prop :=
  let bricks_per_course : ℕ := 400
  let added_courses : ℕ := 2
  let removed_bricks : ℕ := 200
  let total_bricks : ℕ := 1800
  (initial_courses * bricks_per_course) + (added_courses * bricks_per_course) - removed_bricks = total_bricks

/-- Theorem stating that the initial number of courses is 3 -/
theorem initial_courses_is_three : WallProblem 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_courses_is_three_l3080_308035


namespace NUMINAMATH_CALUDE_expression_value_l3080_308005

theorem expression_value (m : ℝ) (h : 1 / (m - 2) = 1) : 2 / (m - 2) - m + 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3080_308005


namespace NUMINAMATH_CALUDE_coefficient_equals_20th_term_l3080_308058

theorem coefficient_equals_20th_term : 
  let binomial (n k : ℕ) := Nat.choose n k
  let coefficient := binomial 5 4 + binomial 6 4 + binomial 7 4
  let a (n : ℕ) := 3 * n - 5
  coefficient = a 20 := by sorry

end NUMINAMATH_CALUDE_coefficient_equals_20th_term_l3080_308058


namespace NUMINAMATH_CALUDE_multiple_problem_l3080_308021

theorem multiple_problem (n : ℝ) (m : ℝ) (h1 : n = 25.0) (h2 : m * n = 3 * n - 25) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l3080_308021


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l3080_308083

def arithmetic_sum (a₁ n : ℕ) : ℕ := n * (a₁ + (a₁ + n - 1)) / 2

theorem piggy_bank_problem (initial_amount final_amount : ℕ) : 
  final_amount = 1478 →
  arithmetic_sum 1 52 = 1378 →
  initial_amount = final_amount - arithmetic_sum 1 52 →
  initial_amount = 100 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l3080_308083


namespace NUMINAMATH_CALUDE_solution_exists_l3080_308024

theorem solution_exists (x : ℝ) : 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l3080_308024


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l3080_308080

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (r : ℝ) (r_pos : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r)^3) = 1 / 13.5 := by
  sorry

#check sphere_to_hemisphere_volume_ratio

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l3080_308080


namespace NUMINAMATH_CALUDE_cube_sum_problem_l3080_308097

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l3080_308097


namespace NUMINAMATH_CALUDE_f_2011_equals_2011_l3080_308054

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the main property of f
variable (h : ∀ a b : ℝ, f (a * f b) = a * b)

-- Theorem statement
theorem f_2011_equals_2011 : f 2011 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_f_2011_equals_2011_l3080_308054


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l3080_308037

def LinearFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

theorem linear_function_decreasing (a b : ℝ) :
  (∀ x y, x < y → LinearFunction a b x > LinearFunction a b y) ↔ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l3080_308037


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3080_308052

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 2 + a 3 = 1 →
  a 2 + a 3 + a 4 = 2 →
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3080_308052


namespace NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l3080_308016

theorem circle_center_in_second_quadrant (a : ℝ) (h : a > 12) :
  let center := (-(a/2), a)
  (center.1 < 0 ∧ center.2 > 0) ∧
  (∀ x y : ℝ, x^2 + y^2 + a*x - 2*a*y + a^2 + 3*a = 0 ↔ 
    (x - center.1)^2 + (y - center.2)^2 = (a^2/4 - 3*a)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l3080_308016


namespace NUMINAMATH_CALUDE_total_votes_l3080_308012

theorem total_votes (jerry_votes : ℕ) (vote_difference : ℕ) : 
  jerry_votes = 108375 →
  vote_difference = 20196 →
  jerry_votes + (jerry_votes - vote_difference) = 196554 :=
by
  sorry

end NUMINAMATH_CALUDE_total_votes_l3080_308012


namespace NUMINAMATH_CALUDE_condition_relationship_l3080_308009

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 1 → 1/x < 1) ∧
  (∃ x, 1/x < 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3080_308009


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3080_308050

theorem smallest_two_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 17 ∧ 
  (∀ m : ℕ, m % 17 = 0 ∧ 10 ≤ m ∧ m < 100 → n ≤ m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3080_308050


namespace NUMINAMATH_CALUDE_system_solution_value_l3080_308075

theorem system_solution_value (x y m : ℝ) : 
  (2 * x + 6 * y = 25) →
  (6 * x + 2 * y = -11) →
  (x - y = m - 1) →
  m = -8 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_value_l3080_308075


namespace NUMINAMATH_CALUDE_cosine_angle_special_vectors_l3080_308067

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem cosine_angle_special_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0)
  (norm_a : ‖a‖ = 1) (norm_b : ‖b‖ = 1) (norm_sum : ‖a + 2 • b‖ = 1) :
  inner a b / (‖a‖ * ‖b‖) = -1 :=
sorry

end NUMINAMATH_CALUDE_cosine_angle_special_vectors_l3080_308067


namespace NUMINAMATH_CALUDE_least_sum_exponents_500_l3080_308087

/-- Represents a sum of distinct powers of 2 -/
def DistinctPowersOfTwo := List Nat

/-- Checks if a list of natural numbers represents distinct powers of 2 -/
def isDistinctPowersOfTwo (l : List Nat) : Prop :=
  l.Nodup ∧ ∀ n ∈ l, ∃ k, n = 2^k

/-- Computes the sum of a list of natural numbers -/
def sumList (l : List Nat) : Nat :=
  l.foldl (·+·) 0

/-- Computes the sum of exponents for a list of powers of 2 -/
def sumExponents (l : DistinctPowersOfTwo) : Nat :=
  sumList (l.map (fun n => (Nat.log n 2)))

/-- The main theorem to be proved -/
theorem least_sum_exponents_500 :
  (∃ (l : DistinctPowersOfTwo),
    isDistinctPowersOfTwo l ∧
    l.length ≥ 2 ∧
    sumList l = 500 ∧
    (∀ (m : DistinctPowersOfTwo),
      isDistinctPowersOfTwo m → m.length ≥ 2 → sumList m = 500 →
      sumExponents l ≤ sumExponents m)) ∧
  (∃ (l : DistinctPowersOfTwo),
    isDistinctPowersOfTwo l ∧
    l.length ≥ 2 ∧
    sumList l = 500 ∧
    sumExponents l = 30) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_exponents_500_l3080_308087


namespace NUMINAMATH_CALUDE_x_over_y_value_l3080_308027

theorem x_over_y_value (x y : ℝ) 
  (h1 : 3 < (2*x - y)/(x + 2*y)) 
  (h2 : (2*x - y)/(x + 2*y) < 7) 
  (h3 : ∃ (n : ℤ), x/y = n) : 
  x/y = -4 := by sorry

end NUMINAMATH_CALUDE_x_over_y_value_l3080_308027


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3080_308071

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 8 = 24 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3080_308071


namespace NUMINAMATH_CALUDE_train_crossing_time_l3080_308013

/-- Proves that a train 600 meters long, traveling at 144 km/hr, takes 15 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 600 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 15 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l3080_308013


namespace NUMINAMATH_CALUDE_complex_equation_solution_pure_imaginary_condition_l3080_308006

-- Problem 1
theorem complex_equation_solution (a b : ℝ) (h : (a + Complex.I) * (1 + Complex.I) = b * Complex.I) :
  a = 1 ∧ b = 2 := by sorry

-- Problem 2
theorem pure_imaginary_condition (m : ℝ) 
  (h : ∃ (k : ℝ), Complex.mk (m^2 + m - 2) (m^2 - 1) = Complex.I * k) :
  m = -2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_pure_imaginary_condition_l3080_308006


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3080_308073

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 10 + a 16 = 30) :
  a 18 - 2 * a 14 = -10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3080_308073


namespace NUMINAMATH_CALUDE_real_estate_calendar_problem_l3080_308042

/-- Proves that given the conditions of the real estate problem, the number of calendars ordered is 200 -/
theorem real_estate_calendar_problem :
  ∀ (calendar_cost date_book_cost : ℚ) (total_items : ℕ) (total_spent : ℚ) (calendars date_books : ℕ),
    calendar_cost = 3/4 →
    date_book_cost = 1/2 →
    total_items = 500 →
    total_spent = 300 →
    calendars + date_books = total_items →
    calendar_cost * calendars + date_book_cost * date_books = total_spent →
    calendars = 200 := by
  sorry

end NUMINAMATH_CALUDE_real_estate_calendar_problem_l3080_308042


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3080_308010

/-- A rectangle with whole number dimensions and perimeter 40 has a maximum area of 100 -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l + w = 20 →
  ∀ l' w' : ℕ,
  l' + w' = 20 →
  l * w ≤ 100 ∧
  (∃ l'' w'' : ℕ, l'' + w'' = 20 ∧ l'' * w'' = 100) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3080_308010


namespace NUMINAMATH_CALUDE_not_closed_under_addition_l3080_308079

-- Define a "good set" S
def GoodSet (S : Set ℤ) : Prop :=
  ∀ a b : ℤ, (a^2 - b^2) ∈ S

-- Theorem statement
theorem not_closed_under_addition
  (S : Set ℤ) (hS : S.Nonempty) (hGood : GoodSet S) :
  ¬ (∀ x y : ℤ, x ∈ S → y ∈ S → (x + y) ∈ S) :=
sorry

end NUMINAMATH_CALUDE_not_closed_under_addition_l3080_308079


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l3080_308062

/-- The perimeter of a semicircle with radius 7 is approximately 35.99 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 7
  let perimeter : ℝ := 2 * r + π * r
  ∃ ε > 0, abs (perimeter - 35.99) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l3080_308062


namespace NUMINAMATH_CALUDE_greatest_a_divisible_by_three_l3080_308098

theorem greatest_a_divisible_by_three : 
  ∀ a : ℕ, 
    a < 10 → 
    (168 * 10000 + a * 100 + 26) % 3 = 0 → 
    a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_a_divisible_by_three_l3080_308098


namespace NUMINAMATH_CALUDE_monomials_not_like_terms_l3080_308047

/-- Definition of a monomial -/
structure Monomial (α : Type*) [CommRing α] :=
  (coeff : α)
  (vars : List (Nat × Nat))  -- List of (variable index, exponent) pairs

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def areLikeTerms {α : Type*} [CommRing α] (m1 m2 : Monomial α) : Prop :=
  m1.vars = m2.vars

/-- Representation of the monomial -12a^2b -/
def m1 : Monomial ℚ :=
  ⟨-12, [(1, 2), (2, 1)]⟩  -- Assuming variable indices: 1 for a, 2 for b

/-- Representation of the monomial 2ab^2/3 -/
def m2 : Monomial ℚ :=
  ⟨2/3, [(1, 1), (2, 2)]⟩

theorem monomials_not_like_terms : ¬(areLikeTerms m1 m2) := by
  sorry


end NUMINAMATH_CALUDE_monomials_not_like_terms_l3080_308047


namespace NUMINAMATH_CALUDE_parabola_equation_l3080_308065

/-- A parabola with vertex at the origin and axis at x = 3/2 has the equation y² = -6x -/
theorem parabola_equation (y x : ℝ) : 
  (∀ (p : ℝ), p > 0 → y^2 = -2*p*x) → -- General equation of parabola with vertex at origin
  (3/2 : ℝ) = p/2 →                   -- Axis of parabola is at x = 3/2
  y^2 = -6*x :=                       -- Equation to be proved
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3080_308065


namespace NUMINAMATH_CALUDE_crate_weight_l3080_308002

/-- Given an empty truck weighing 9600 kg and a total weight of 38000 kg when loaded with 40 identical crates, 
    prove that each crate weighs 710 kg. -/
theorem crate_weight (empty_truck_weight : ℕ) (loaded_truck_weight : ℕ) (num_crates : ℕ) :
  empty_truck_weight = 9600 →
  loaded_truck_weight = 38000 →
  num_crates = 40 →
  (loaded_truck_weight - empty_truck_weight) / num_crates = 710 :=
by sorry

end NUMINAMATH_CALUDE_crate_weight_l3080_308002


namespace NUMINAMATH_CALUDE_kyuhyung_cards_l3080_308095

/-- The number of cards in Kyuhyung's possession -/
def total_cards : ℕ := 103

/-- The side length of the square arrangement -/
def side_length : ℕ := 10

/-- The number of cards left over after forming the square -/
def leftover_cards : ℕ := 3

/-- The number of additional cards needed to fill the outer perimeter -/
def perimeter_cards : ℕ := 44

theorem kyuhyung_cards :
  total_cards = side_length^2 + leftover_cards ∧
  (side_length + 2)^2 - side_length^2 = perimeter_cards :=
by sorry

end NUMINAMATH_CALUDE_kyuhyung_cards_l3080_308095


namespace NUMINAMATH_CALUDE_eugene_model_house_l3080_308033

/-- The number of toothpicks required for each card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in a deck -/
def cards_in_deck : ℕ := 52

/-- The number of cards Eugene didn't use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in each box -/
def toothpicks_per_box : ℕ := 450

/-- Calculate the number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ :=
  let cards_used := cards_in_deck - unused_cards
  let total_toothpicks := cards_used * toothpicks_per_card
  (total_toothpicks + toothpicks_per_box - 1) / toothpicks_per_box

theorem eugene_model_house :
  boxes_used = 6 := by
  sorry

end NUMINAMATH_CALUDE_eugene_model_house_l3080_308033


namespace NUMINAMATH_CALUDE_lucas_income_l3080_308007

/-- Represents the tax structure and Lucas's income --/
structure TaxSystem where
  p : ℝ  -- Base tax rate as a decimal
  income : ℝ  -- Lucas's annual income
  taxPaid : ℝ  -- Total tax paid by Lucas

/-- The tax system satisfies the given conditions --/
def validTaxSystem (ts : TaxSystem) : Prop :=
  ts.taxPaid = (0.01 * ts.p * 35000 + 0.01 * (ts.p + 4) * (ts.income - 35000))
  ∧ ts.taxPaid = 0.01 * (ts.p + 0.5) * ts.income
  ∧ ts.income ≥ 35000

/-- Theorem stating that Lucas's income is $40000 --/
theorem lucas_income (ts : TaxSystem) (h : validTaxSystem ts) : ts.income = 40000 := by
  sorry

end NUMINAMATH_CALUDE_lucas_income_l3080_308007


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3080_308060

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 2 →                    -- a_1 = 2
  (a 1 + a 2 + a 3 = 26) →     -- S_3 = 26
  q = 3 ∨ q = -4 :=            -- conclusion: q is 3 or -4
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3080_308060


namespace NUMINAMATH_CALUDE_job_completion_time_l3080_308004

/-- Calculates the remaining days to complete a job given initial and additional workers -/
def remaining_days (initial_workers : ℕ) (initial_days : ℕ) (days_worked : ℕ) (additional_workers : ℕ) : ℚ :=
  let total_work := initial_workers * initial_days
  let work_done := initial_workers * days_worked
  let remaining_work := total_work - work_done
  let total_workers := initial_workers + additional_workers
  remaining_work / total_workers

theorem job_completion_time : remaining_days 6 8 3 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3080_308004


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l3080_308096

/-- Given a college with students playing cricket or basketball, 
    this theorem proves the number of students playing both sports. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (cricket : ℕ) 
  (basketball : ℕ) 
  (h1 : total = 880) 
  (h2 : cricket = 500) 
  (h3 : basketball = 600) : 
  cricket + basketball - total = 220 := by
  sorry


end NUMINAMATH_CALUDE_students_playing_both_sports_l3080_308096


namespace NUMINAMATH_CALUDE_triangle_base_length_l3080_308086

/-- The length of the base of a triangle with altitude 12 cm and area equal to a square with side 6 cm is 6 cm. -/
theorem triangle_base_length (base altitude : ℝ) (h1 : altitude = 12) 
  (h2 : (base * altitude) / 2 = 6 * 6) : base = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3080_308086


namespace NUMINAMATH_CALUDE_years_until_double_age_l3080_308000

/-- Represents the age difference problem between a father and son -/
structure AgeDifference where
  son_age : ℕ
  father_age : ℕ
  years_until_double : ℕ

/-- The age difference scenario satisfies the given conditions -/
def valid_age_difference (ad : AgeDifference) : Prop :=
  ad.son_age = 10 ∧
  ad.father_age = 40 ∧
  ad.father_age = 4 * ad.son_age ∧
  ad.father_age + ad.years_until_double = 2 * (ad.son_age + ad.years_until_double)

/-- Theorem stating that the number of years until the father is twice as old as the son is 20 -/
theorem years_until_double_age : ∀ ad : AgeDifference, valid_age_difference ad → ad.years_until_double = 20 := by
  sorry

end NUMINAMATH_CALUDE_years_until_double_age_l3080_308000


namespace NUMINAMATH_CALUDE_cost_per_person_l3080_308001

def total_cost : ℚ := 12100
def num_people : ℕ := 11

theorem cost_per_person :
  total_cost / num_people = 1100 :=
sorry

end NUMINAMATH_CALUDE_cost_per_person_l3080_308001


namespace NUMINAMATH_CALUDE_deriv_sin_plus_cos_at_pi_fourth_l3080_308034

/-- The derivative of sin(x) + cos(x) at π/4 is 0 -/
theorem deriv_sin_plus_cos_at_pi_fourth (f : ℝ → ℝ) (h : f = λ x => Real.sin x + Real.cos x) :
  deriv f (π / 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_deriv_sin_plus_cos_at_pi_fourth_l3080_308034


namespace NUMINAMATH_CALUDE_volume_of_removed_tetrahedra_l3080_308081

-- Define the cube
def cube_side_length : ℝ := 2

-- Define the number of segments per edge
def segments_per_edge : ℕ := 3

-- Define the number of corners (tetrahedra)
def num_corners : ℕ := 8

-- Theorem statement
theorem volume_of_removed_tetrahedra :
  let segment_length : ℝ := cube_side_length / segments_per_edge
  let base_area : ℝ := (1 / 2) * segment_length^2
  let tetrahedron_height : ℝ := segment_length
  let tetrahedron_volume : ℝ := (1 / 3) * base_area * tetrahedron_height
  let total_volume : ℝ := num_corners * tetrahedron_volume
  total_volume = 32 / 81 := by sorry

end NUMINAMATH_CALUDE_volume_of_removed_tetrahedra_l3080_308081


namespace NUMINAMATH_CALUDE_office_paper_sheets_per_pack_l3080_308090

/-- The number of sheets in each pack of printer paper -/
def sheets_per_pack (total_packs : ℕ) (documents_per_day : ℕ) (days_lasted : ℕ) : ℕ :=
  (documents_per_day * days_lasted) / total_packs

/-- Theorem stating the number of sheets in each pack of printer paper -/
theorem office_paper_sheets_per_pack :
  sheets_per_pack 2 80 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_office_paper_sheets_per_pack_l3080_308090


namespace NUMINAMATH_CALUDE_temperature_decrease_l3080_308030

theorem temperature_decrease (current_temp : ℝ) (decrease_factor : ℝ) :
  current_temp = 84 →
  decrease_factor = 3/4 →
  current_temp - (decrease_factor * current_temp) = 21 := by
sorry

end NUMINAMATH_CALUDE_temperature_decrease_l3080_308030


namespace NUMINAMATH_CALUDE_zero_subset_X_l3080_308088

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_X_l3080_308088


namespace NUMINAMATH_CALUDE_constant_point_on_line_l3080_308057

/-- The line equation passing through a constant point regardless of m -/
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x - (2 * m - 1) * y = 3 * m - 4

/-- The theorem stating that (-1, -2) satisfies the line equation for all m -/
theorem constant_point_on_line :
  ∀ m : ℝ, line_equation m (-1) (-2) :=
by sorry

end NUMINAMATH_CALUDE_constant_point_on_line_l3080_308057


namespace NUMINAMATH_CALUDE_smallest_b_value_l3080_308020

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ c : ℕ+, c.val < b.val → 
    ¬(∃ d : ℕ+, d.val - c.val = 8 ∧ 
      Nat.gcd ((d.val^3 + c.val^3) / (d.val + c.val)) (d.val * c.val) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3080_308020


namespace NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l3080_308025

/-- Represents the yield of blueberry containers per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 72

/-- 
Calculates the minimum number of bushes needed to obtain at least the target number of zucchinis.
-/
def min_bushes_needed : ℕ :=
  ((target_zucchinis * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush : ℕ)

theorem min_bushes_for_zucchinis :
  min_bushes_needed = 22 ∧
  min_bushes_needed * containers_per_bush ≥ target_zucchinis * containers_per_zucchini ∧
  (min_bushes_needed - 1) * containers_per_bush < target_zucchinis * containers_per_zucchini :=
by sorry

end NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l3080_308025


namespace NUMINAMATH_CALUDE_unit_vector_AB_l3080_308068

/-- Given points A(1,3) and B(4,-1), the unit vector in the same direction as vector AB is (3/5, -4/5) -/
theorem unit_vector_AB (A B : ℝ × ℝ) (h : A = (1, 3) ∧ B = (4, -1)) :
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let magnitude : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  (AB.1 / magnitude, AB.2 / magnitude) = (3/5, -4/5) := by
  sorry


end NUMINAMATH_CALUDE_unit_vector_AB_l3080_308068


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l3080_308029

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    is_prime d ∧ d < 20 ∧
    is_prime e ∧ e < 20 ∧
    is_prime (e^2 + 10*d) ∧
    n = d * e * (e^2 + 10*d) ∧
    (∀ (n' d' e' : ℕ),
      is_prime d' ∧ d' < 20 ∧
      is_prime e' ∧ e' < 20 ∧
      is_prime (e'^2 + 10*d') ∧
      n' = d' * e' * (e'^2 + 10*d') →
      n' ≤ n) ∧
    sum_of_digits n = 16 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l3080_308029
