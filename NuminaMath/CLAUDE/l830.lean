import Mathlib

namespace NUMINAMATH_CALUDE_shooter_hit_rate_l830_83055

theorem shooter_hit_rate (shots : ℕ) (prob_hit_at_least_once : ℚ) (hit_rate : ℚ) :
  shots = 4 →
  prob_hit_at_least_once = 80 / 81 →
  (1 - (1 - hit_rate) ^ shots) = prob_hit_at_least_once →
  hit_rate = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shooter_hit_rate_l830_83055


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l830_83024

theorem geometric_sequence_seventh_term (x : ℝ) (b : ℕ → ℝ) 
  (h1 : b 1 = Real.sin x ^ 2)
  (h2 : b 2 = Real.sin x * Real.cos x)
  (h3 : b 3 = (Real.cos x ^ 2) / (Real.sin x))
  (h_geom : ∀ n : ℕ, n ≥ 1 → b (n + 1) = (b 2 / b 1) * b n) :
  b 7 = Real.cos x + Real.sin x :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l830_83024


namespace NUMINAMATH_CALUDE_lunch_combinations_l830_83058

/-- The number of different types of meat dishes -/
def num_meat_dishes : ℕ := 4

/-- The number of different types of vegetable dishes -/
def num_veg_dishes : ℕ := 7

/-- The number of meat dishes chosen in the first combination method -/
def meat_choice_1 : ℕ := 2

/-- The number of vegetable dishes chosen in both combination methods -/
def veg_choice : ℕ := 2

/-- The number of meat dishes chosen in the second combination method -/
def meat_choice_2 : ℕ := 1

/-- The total number of lunch combinations -/
def total_combinations : ℕ := Nat.choose num_meat_dishes meat_choice_1 * Nat.choose num_veg_dishes veg_choice +
                               Nat.choose num_meat_dishes meat_choice_2 * Nat.choose num_veg_dishes veg_choice

theorem lunch_combinations : total_combinations = 210 := by
  sorry

end NUMINAMATH_CALUDE_lunch_combinations_l830_83058


namespace NUMINAMATH_CALUDE_complement_of_P_union_Q_is_M_l830_83090

def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

theorem complement_of_P_union_Q_is_M : (P ∪ Q)ᶜ = M := by sorry

end NUMINAMATH_CALUDE_complement_of_P_union_Q_is_M_l830_83090


namespace NUMINAMATH_CALUDE_product_and_sum_of_factors_l830_83043

theorem product_and_sum_of_factors : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8775 ∧ 
  a + b = 110 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_factors_l830_83043


namespace NUMINAMATH_CALUDE_beaver_change_l830_83086

theorem beaver_change (initial_beavers initial_chipmunks chipmunk_decrease total_animals : ℕ) :
  initial_beavers = 20 →
  initial_chipmunks = 40 →
  chipmunk_decrease = 10 →
  total_animals = 130 →
  (total_animals - (initial_beavers + initial_chipmunks)) - (initial_chipmunks - chipmunk_decrease) - initial_beavers = 20 := by
  sorry

end NUMINAMATH_CALUDE_beaver_change_l830_83086


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l830_83048

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : 
  a = 12 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l830_83048


namespace NUMINAMATH_CALUDE_concert_attendance_l830_83080

/-- The number of buses used for the concert. -/
def num_buses : ℕ := 8

/-- The number of students each bus can carry. -/
def students_per_bus : ℕ := 45

/-- The total number of students who went to the concert. -/
def total_students : ℕ := num_buses * students_per_bus

theorem concert_attendance : total_students = 360 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l830_83080


namespace NUMINAMATH_CALUDE_manager_salary_is_4200_l830_83033

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager's salary is added. -/
def managerSalary (numEmployees : ℕ) (avgSalary : ℚ) (avgIncrease : ℚ) : ℚ :=
  (avgSalary + avgIncrease) * (numEmployees + 1) - avgSalary * numEmployees

/-- Proves that the manager's salary is 4200 given the problem conditions. -/
theorem manager_salary_is_4200 :
  managerSalary 15 1800 150 = 4200 := by
  sorry

#eval managerSalary 15 1800 150

end NUMINAMATH_CALUDE_manager_salary_is_4200_l830_83033


namespace NUMINAMATH_CALUDE_equal_expressions_l830_83015

theorem equal_expressions : 10006 - 8008 = 10000 - 8002 := by sorry

end NUMINAMATH_CALUDE_equal_expressions_l830_83015


namespace NUMINAMATH_CALUDE_sqrt_25_times_sqrt_25_l830_83085

theorem sqrt_25_times_sqrt_25 : Real.sqrt (25 * Real.sqrt 25) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_25_times_sqrt_25_l830_83085


namespace NUMINAMATH_CALUDE_prime_factor_sum_squares_l830_83072

theorem prime_factor_sum_squares (n : ℕ+) : 
  (∃ p q : ℕ, 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ n ∧ 
    q ∣ n ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ n → p ≤ r) ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ n → r ≤ q) ∧
    p^2 + q^2 = n + 9) ↔ 
  n = 9 ∨ n = 20 := by
sorry


end NUMINAMATH_CALUDE_prime_factor_sum_squares_l830_83072


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l830_83083

theorem divisibility_equivalence (a b : ℤ) : 
  (13 ∣ (2*a + 3*b)) ↔ (13 ∣ (2*b - 3*a)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l830_83083


namespace NUMINAMATH_CALUDE_pigeonhole_divisibility_l830_83071

theorem pigeonhole_divisibility (x : Fin 2020 → ℤ) :
  ∃ i j : Fin 2020, i ≠ j ∧ (x j - x i) % 2019 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_divisibility_l830_83071


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l830_83031

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

theorem batsman_average_after_12th_innings 
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 65 = b.average + 3)
  : newAverage b 65 = 32 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l830_83031


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l830_83065

theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ), ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
    (-x^2 + 4*x - 5) / (x^3 + x) = A / x + (B*x + C) / (x^2 + 1) ∧
    A = -5 ∧ B = 4 ∧ C = 4 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l830_83065


namespace NUMINAMATH_CALUDE_circle_intersection_properties_l830_83051

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the intersection points
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ A ≠ B

-- Theorem statement
theorem circle_intersection_properties
  (A B : ℝ × ℝ) (h : intersectionPoints A B) :
  -- 1. Equation of the line containing chord AB
  (∀ x y : ℝ, (x - y - 3 = 0) ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2))) ∧
  -- 2. Length of the common chord AB
  Real.sqrt 2 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  -- 3. Equation of the perpendicular bisector of AB
  (∀ x y : ℝ, (x + y = 0) ↔ ((x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_properties_l830_83051


namespace NUMINAMATH_CALUDE_square_construction_impossibility_l830_83037

theorem square_construction_impossibility (k : ℕ) (h : k ≥ 2) :
  ¬ (∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ S → x.1 = 1 ∧ x.2 ≤ k) ∧ 
    (S.sum (λ x => x.1 * x.2) = k * k) ∧
    (S.card ≤ k)) := by
  sorry

end NUMINAMATH_CALUDE_square_construction_impossibility_l830_83037


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l830_83068

theorem slower_speed_calculation (distance : ℝ) (time_saved : ℝ) (faster_speed : ℝ) :
  distance = 1200 →
  time_saved = 4 →
  faster_speed = 60 →
  ∃ slower_speed : ℝ,
    (distance / slower_speed) - (distance / faster_speed) = time_saved ∧
    slower_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l830_83068


namespace NUMINAMATH_CALUDE_mean_calculation_l830_83027

theorem mean_calculation (x y : ℝ) : 
  (28 + x + 50 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + y + x) / 5 = (258 + y) / 5 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l830_83027


namespace NUMINAMATH_CALUDE_sum_of_odd_prime_divisors_of_90_l830_83049

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is a divisor of 90
def isDivisorOf90 (n : ℕ) : Prop :=
  90 % n = 0

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Prop :=
  n % 2 ≠ 0

-- Theorem statement
theorem sum_of_odd_prime_divisors_of_90 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, isPrime n ∧ isOdd n ∧ isDivisorOf90 n) ∧ 
    (∀ n : ℕ, isPrime n → isOdd n → isDivisorOf90 n → n ∈ S) ∧
    (S.sum id = 8) :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_prime_divisors_of_90_l830_83049


namespace NUMINAMATH_CALUDE_sum_of_first_few_primes_equals_41_l830_83066

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a unique positive integer n such that the sum of the first n prime numbers equals 41, and that n = 6 -/
theorem sum_of_first_few_primes_equals_41 :
  ∃! n : ℕ, n > 0 ∧ sumFirstNPrimes n = 41 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_few_primes_equals_41_l830_83066


namespace NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l830_83097

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem problem_solution : (5 : ℚ) / ((7 : ℚ) / 13) = 65 / 7 :=
by sorry

end NUMINAMATH_CALUDE_division_by_fraction_problem_solution_l830_83097


namespace NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_deriv_l830_83001

/-- The function f(x) = -x^3 + ax^2 - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f_deriv (a : ℝ) :
  (∃ (x : ℝ), f_deriv a x = 0 ∧ x = 2) →
  (∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 → n' ∈ Set.Icc (-1 : ℝ) 1 →
      f a m + f_deriv a n ≤ f a m' + f_deriv a n') →
  ∃ (m n : ℝ), m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_deriv a n = -13 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_deriv_l830_83001


namespace NUMINAMATH_CALUDE_smallest_b_for_five_not_in_range_l830_83050

theorem smallest_b_for_five_not_in_range :
  ∃ (b : ℤ), (∀ x : ℝ, x^2 + b*x + 10 ≠ 5) ∧
             (∀ c : ℤ, c < b → ∃ x : ℝ, x^2 + c*x + 10 = 5) ∧
             b = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_five_not_in_range_l830_83050


namespace NUMINAMATH_CALUDE_triangle_side_length_l830_83087

/-- Given a triangle ABC where the internal angles form an arithmetic sequence,
    and sides a = 4 and c = 3, prove that the length of side b is √13 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  B = (A + C) / 2 →      -- Angles form an arithmetic sequence
  a = 4 →                -- Length of side a
  c = 3 →                -- Length of side c
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →  -- Cosine rule
  b = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l830_83087


namespace NUMINAMATH_CALUDE_smallest_seating_arrangement_l830_83032

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if a seating arrangement satisfies the condition that any new person must sit next to someone -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ seating.total_chairs → 
    ∃ i j, i ≠ j ∧ 
           (i % seating.total_chairs + 1 = k ∨ (i + 1) % seating.total_chairs + 1 = k) ∧
           (j % seating.total_chairs + 1 = k ∨ (j + 1) % seating.total_chairs + 1 = k)

/-- The main theorem to prove -/
theorem smallest_seating_arrangement :
  ∀ n < 25, ¬(satisfies_condition ⟨100, n⟩) ∧ 
  satisfies_condition ⟨100, 25⟩ := by
  sorry

#check smallest_seating_arrangement

end NUMINAMATH_CALUDE_smallest_seating_arrangement_l830_83032


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l830_83093

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
if the distance from a vertex to one of its asymptotes is b/2,
then its eccentricity is 2.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_distance : (b * a) / Real.sqrt (a^2 + b^2) = b / 2) : 
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l830_83093


namespace NUMINAMATH_CALUDE_wall_painting_theorem_l830_83019

theorem wall_painting_theorem (heidi_time tim_time total_time : ℚ) 
  (h1 : heidi_time = 45)
  (h2 : tim_time = 30)
  (h3 : total_time = 9) :
  let heidi_rate : ℚ := 1 / heidi_time
  let tim_rate : ℚ := 1 / tim_time
  let combined_rate : ℚ := heidi_rate + tim_rate
  (combined_rate * total_time) = 1/2 := by sorry

end NUMINAMATH_CALUDE_wall_painting_theorem_l830_83019


namespace NUMINAMATH_CALUDE_complex_equation_solution_l830_83042

theorem complex_equation_solution (z : ℂ) : (z - 2*Complex.I) * (2 - Complex.I) = 5 → z = 2 + 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l830_83042


namespace NUMINAMATH_CALUDE_changhee_semester_average_l830_83020

/-- Calculates the average score for a semester given midterm and final exam scores and subject counts. -/
def semesterAverage (midtermAvg : ℚ) (midtermSubjects : ℕ) (finalAvg : ℚ) (finalSubjects : ℕ) : ℚ :=
  (midtermAvg * midtermSubjects + finalAvg * finalSubjects) / (midtermSubjects + finalSubjects)

/-- Proves that Changhee's semester average is 83.5 given the exam scores and subject counts. -/
theorem changhee_semester_average :
  semesterAverage 83.1 10 84 8 = 83.5 := by
  sorry

#eval semesterAverage 83.1 10 84 8

end NUMINAMATH_CALUDE_changhee_semester_average_l830_83020


namespace NUMINAMATH_CALUDE_logic_statement_l830_83008

theorem logic_statement :
  let p : Prop := 2 + 3 = 5
  let q : Prop := 5 < 4
  (p ∨ q) ∧ ¬(p ∧ q) ∧ p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_logic_statement_l830_83008


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l830_83052

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 2}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l830_83052


namespace NUMINAMATH_CALUDE_absolute_value_equation_one_negative_root_l830_83079

theorem absolute_value_equation_one_negative_root (a : ℝ) : 
  (∃! x : ℝ, x < 0 ∧ |x| = a * x + 1) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_one_negative_root_l830_83079


namespace NUMINAMATH_CALUDE_blue_cross_coverage_l830_83075

/-- Represents a circular flag with overlapping crosses -/
structure CircularFlag where
  /-- The total area of the flag -/
  total_area : ℝ
  /-- The area covered by the blue cross -/
  blue_cross_area : ℝ
  /-- The area covered by the red cross -/
  red_cross_area : ℝ
  /-- The area covered by both crosses combined -/
  combined_crosses_area : ℝ
  /-- The red cross is half the width of the blue cross -/
  red_half_blue : blue_cross_area = 2 * red_cross_area
  /-- The combined area of both crosses is 50% of the flag's area -/
  combined_half_total : combined_crosses_area = 0.5 * total_area
  /-- The red cross covers 20% of the flag's area -/
  red_fifth_total : red_cross_area = 0.2 * total_area

/-- Theorem stating that the blue cross alone covers 30% of the flag's area -/
theorem blue_cross_coverage (flag : CircularFlag) : 
  flag.blue_cross_area = 0.3 * flag.total_area := by
  sorry

end NUMINAMATH_CALUDE_blue_cross_coverage_l830_83075


namespace NUMINAMATH_CALUDE_k_range_theorem_l830_83081

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

/-- A function f has a maximum value of a and a minimum value of b on the interval [0, k] -/
def HasMaxMinOn (f : ℝ → ℝ) (a b k : ℝ) : Prop :=
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ k → f x ≤ a) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ f x = a) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ k → b ≤ f x) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ k ∧ f x = b)

theorem k_range_theorem (k : ℝ) :
  let p := IsIncreasing (λ x : ℝ => k * x + 1)
  let q := HasMaxMinOn (λ x : ℝ => x^2 - 2*x + 3) 3 2 k
  (¬(p ∧ q)) ∧ (p ∨ q) → k ∈ Set.Ioo 0 1 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_k_range_theorem_l830_83081


namespace NUMINAMATH_CALUDE_real_part_z_2017_l830_83025

def z : ℂ := 1 + Complex.I

theorem real_part_z_2017 : (z^2017).re = 2^1008 := by sorry

end NUMINAMATH_CALUDE_real_part_z_2017_l830_83025


namespace NUMINAMATH_CALUDE_beaver_count_correct_l830_83040

/-- The number of beavers in the first scenario -/
def num_beavers : ℕ := 20

/-- The time taken by the first group of beavers to build the dam -/
def time_first : ℕ := 18

/-- The number of beavers in the second scenario -/
def num_beavers_second : ℕ := 12

/-- The time taken by the second group of beavers to build the dam -/
def time_second : ℕ := 30

/-- The theorem stating that the calculated number of beavers is correct -/
theorem beaver_count_correct :
  num_beavers * time_first = num_beavers_second * time_second :=
by sorry

end NUMINAMATH_CALUDE_beaver_count_correct_l830_83040


namespace NUMINAMATH_CALUDE_library_books_distribution_l830_83070

theorem library_books_distribution (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  (a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28) := by
  sorry

end NUMINAMATH_CALUDE_library_books_distribution_l830_83070


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l830_83000

theorem gcd_power_minus_one (k : ℤ) : Int.gcd (k^1024 - 1) (k^1035 - 1) = k - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l830_83000


namespace NUMINAMATH_CALUDE_tangent_line_and_m_range_l830_83059

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x - 1) + Real.log x + 1

theorem tangent_line_and_m_range :
  (∀ x : ℝ, x > 0 → (x - f x - 1 = 0 → x = 1)) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1/x)) + 1 ≥ 0) ↔ m ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_m_range_l830_83059


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l830_83054

/-- A right triangle with an inscribed rectangle -/
structure RightTriangleWithRectangle where
  -- The lengths of the legs of the right triangle
  ab : ℝ
  ac : ℝ
  -- The sides of the inscribed rectangle
  ad : ℝ
  am : ℝ
  -- Conditions
  ab_positive : 0 < ab
  ac_positive : 0 < ac
  ad_positive : 0 < ad
  am_positive : 0 < am
  ad_le_ab : ad ≤ ab
  am_le_ac : am ≤ ac

/-- The theorem statement -/
theorem inscribed_rectangle_sides
  (triangle : RightTriangleWithRectangle)
  (h_ab : triangle.ab = 5)
  (h_ac : triangle.ac = 12)
  (h_area : triangle.ad * triangle.am = 40 / 3)
  (h_diagonal : triangle.ad ^ 2 + triangle.am ^ 2 < 8 ^ 2) :
  triangle.ad = 4 ∧ triangle.am = 10 / 3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l830_83054


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l830_83099

/-- Given a rectangular solid with side lengths x, y, and z, 
    if the surface area is 11 and the sum of the lengths of all edges is 24, 
    then the length of one of its diagonals is 5. -/
theorem rectangular_solid_diagonal 
  (x y z : ℝ) 
  (h1 : 2*x*y + 2*y*z + 2*x*z = 11) 
  (h2 : 4*(x + y + z) = 24) : 
  Real.sqrt (x^2 + y^2 + z^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l830_83099


namespace NUMINAMATH_CALUDE_function_upper_bound_l830_83047

theorem function_upper_bound (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l830_83047


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_square_l830_83073

theorem power_of_two_greater_than_square (n : ℕ) (h : n ≥ 1) :
  2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_square_l830_83073


namespace NUMINAMATH_CALUDE_sin_fifth_power_coefficients_sum_of_squares_l830_83077

theorem sin_fifth_power_coefficients_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ : ℝ),
    (∀ θ : ℝ, (Real.sin θ)^5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 63 / 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_fifth_power_coefficients_sum_of_squares_l830_83077


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l830_83044

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : ℕ
  num_districts : ℕ
  precincts_per_district : ℕ
  voters_per_precinct : ℕ

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The theorem stating the minimum number of voters for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingStructure) 
  (h1 : vs.total_voters = 135)
  (h2 : vs.num_districts = 5)
  (h3 : vs.precincts_per_district = 9)
  (h4 : vs.voters_per_precinct = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.precincts_per_district * vs.voters_per_precinct) :
  min_voters_to_win vs = 30 := by
  sorry

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l830_83044


namespace NUMINAMATH_CALUDE_cheesecake_calories_l830_83056

/-- Represents a cheesecake with its properties -/
structure Cheesecake where
  calories_per_slice : ℕ
  quarter_slices : ℕ

/-- Calculates the total calories in a cheesecake -/
def total_calories (c : Cheesecake) : ℕ :=
  c.calories_per_slice * (4 * c.quarter_slices)

/-- Proves that the total calories in the given cheesecake is 2800 -/
theorem cheesecake_calories (c : Cheesecake)
    (h1 : c.calories_per_slice = 350)
    (h2 : c.quarter_slices = 2) :
    total_calories c = 2800 := by
  sorry

#eval total_calories { calories_per_slice := 350, quarter_slices := 2 }

end NUMINAMATH_CALUDE_cheesecake_calories_l830_83056


namespace NUMINAMATH_CALUDE_trigonometric_identity_l830_83057

theorem trigonometric_identity (α : ℝ) : 
  (Real.cos (2 * α))^4 - 6 * (Real.cos (2 * α))^2 * (Real.sin (2 * α))^2 + (Real.sin (2 * α))^4 = Real.cos (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l830_83057


namespace NUMINAMATH_CALUDE_intersection_points_on_ellipse_l830_83005

/-- The points of intersection of two parametric lines lie on an ellipse -/
theorem intersection_points_on_ellipse (s : ℝ) : 
  ∃ (a b : ℝ) (h : a > 0 ∧ b > 0), 
    ∀ (x y : ℝ), 
      (s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) → 
      (x^2 / a^2 + y^2 / b^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_intersection_points_on_ellipse_l830_83005


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_b_and_eccentricity_l830_83012

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of the hyperbola -/
def asymptote (h : Hyperbola) (x : ℝ) : ℝ := h.b * x

theorem hyperbola_asymptote_through_point_implies_b_and_eccentricity
  (h : Hyperbola)
  (h_asymptote : asymptote h 1 = 2) :
  h.b = 2 ∧ Real.sqrt ((1 : ℝ)^2 + h.b^2) / 1 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_b_and_eccentricity_l830_83012


namespace NUMINAMATH_CALUDE_cube_size_is_eight_l830_83064

/-- Represents a cube of size n --/
structure Cube (n : ℕ) where
  size : n > 0

/-- Number of small cubes with no faces painted in a cube of size n --/
def unpainted (c : Cube n) : ℕ := (n - 2)^3

/-- Number of small cubes with exactly two faces painted in a cube of size n --/
def two_faces_painted (c : Cube n) : ℕ := 12 * (n - 2)

/-- Theorem stating that for a cube where the number of unpainted small cubes
    is three times the number of small cubes with two faces painted,
    the size of the cube must be 8 --/
theorem cube_size_is_eight (c : Cube n)
  (h : unpainted c = 3 * two_faces_painted c) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_size_is_eight_l830_83064


namespace NUMINAMATH_CALUDE_cos_75_degrees_l830_83029

theorem cos_75_degrees : 
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l830_83029


namespace NUMINAMATH_CALUDE_min_surface_area_3x3x3_minus_5_l830_83067

/-- Represents a 3D cube composed of unit cubes -/
structure Cube3D where
  size : Nat
  total_units : Nat

/-- Represents the remaining solid after removing some unit cubes -/
structure RemainingCube where
  original : Cube3D
  removed : Nat

/-- Calculates the minimum surface area of the remaining solid -/
def min_surface_area (rc : RemainingCube) : Nat :=
  sorry

/-- Theorem stating the minimum surface area after removing 5 unit cubes from a 3x3x3 cube -/
theorem min_surface_area_3x3x3_minus_5 :
  let original_cube : Cube3D := { size := 3, total_units := 27 }
  let remaining_cube : RemainingCube := { original := original_cube, removed := 5 }
  min_surface_area remaining_cube = 50 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_3x3x3_minus_5_l830_83067


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l830_83088

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, a 1 = 1 ∧ ∀ n ≥ 3, a n = 100 ∧ ∀ k : ℕ, a (k + 1) = a k + d

/-- The set of possible n values -/
def PossibleN : Set ℕ := {4, 10, 12, 34, 100}

/-- The main theorem -/
theorem arithmetic_sequence_n_values (a : ℕ → ℕ) :
  ArithmeticSequence a →
  (∀ n : ℕ, n ∈ PossibleN ↔ (n ≥ 3 ∧ a n = 100)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l830_83088


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l830_83095

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x - 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l830_83095


namespace NUMINAMATH_CALUDE_field_trip_adults_l830_83009

theorem field_trip_adults (van_capacity : ℕ) (num_vans : ℕ) (num_students : ℕ) :
  van_capacity = 4 →
  num_vans = 2 →
  num_students = 2 →
  ∃ (num_adults : ℕ), num_adults + num_students = num_vans * van_capacity ∧ num_adults = 6 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_adults_l830_83009


namespace NUMINAMATH_CALUDE_eggs_equal_to_rice_l830_83002

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℚ := 33/100

/-- The cost of a liter of kerosene in dollars -/
def kerosene_cost : ℚ := 22/100

/-- The number of eggs that cost as much as a half-liter of kerosene -/
def eggs_per_half_liter : ℕ := 4

/-- Theorem stating that 12 eggs cost as much as a pound of rice -/
theorem eggs_equal_to_rice : ℕ := by
  sorry

end NUMINAMATH_CALUDE_eggs_equal_to_rice_l830_83002


namespace NUMINAMATH_CALUDE_simplify_expression_l830_83092

theorem simplify_expression : (625 : ℝ) ^ (1/4 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l830_83092


namespace NUMINAMATH_CALUDE_subset_family_bound_l830_83091

theorem subset_family_bound (n k m : ℕ) (B : Fin m → Finset (Fin n)) :
  (∀ i, (B i).card = k) →
  (k ≥ 2) →
  (∀ i j, i < j → (B i ∩ B j).card ≤ 1) →
  m ≤ ⌊(n : ℝ) / k * ⌊(n - 1 : ℝ) / (k - 1)⌋⌋ :=
by sorry

end NUMINAMATH_CALUDE_subset_family_bound_l830_83091


namespace NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l830_83010

theorem repeated_root_implies_m_equals_two (x m : ℝ) : 
  (2 / (x - 1) + 3 = m / (x - 1)) →  -- Condition 1
  (x - 1 = 0) →                      -- Condition 2 (repeated root implies x - 1 = 0)
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l830_83010


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l830_83061

/-- The man's rowing speed in still water given his speeds with and against the stream -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 6)
  (h2 : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l830_83061


namespace NUMINAMATH_CALUDE_population_reaches_capacity_years_to_max_capacity_l830_83034

/-- The maximum capacity of the realm in people -/
def max_capacity : ℕ := 35000 / 2

/-- The initial population in 2023 -/
def initial_population : ℕ := 500

/-- The population growth factor every 20 years -/
def growth_factor : ℕ := 2

/-- The population after n 20-year periods -/
def population (n : ℕ) : ℕ := initial_population * growth_factor ^ n

/-- The number of 20-year periods after which the population reaches or exceeds the maximum capacity -/
def periods_to_max_capacity : ℕ := 5

theorem population_reaches_capacity :
  population periods_to_max_capacity ≥ max_capacity ∧
  population (periods_to_max_capacity - 1) < max_capacity :=
sorry

theorem years_to_max_capacity : periods_to_max_capacity * 20 = 100 :=
sorry

end NUMINAMATH_CALUDE_population_reaches_capacity_years_to_max_capacity_l830_83034


namespace NUMINAMATH_CALUDE_steven_has_16_apples_l830_83006

/-- Represents the number of fruits a person has -/
structure FruitCount where
  peaches : ℕ
  apples : ℕ

/-- Given information about Steven and Jake's fruit counts -/
def steven_jake_fruits : Prop :=
  ∃ (steven jake : FruitCount),
    steven.peaches = 17 ∧
    steven.peaches = steven.apples + 1 ∧
    jake.peaches + 6 = steven.peaches ∧
    jake.apples = steven.apples + 8

/-- Theorem stating that Steven has 16 apples -/
theorem steven_has_16_apples :
  steven_jake_fruits → ∃ (steven : FruitCount), steven.apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_steven_has_16_apples_l830_83006


namespace NUMINAMATH_CALUDE_juans_number_puzzle_l830_83094

theorem juans_number_puzzle (n : ℝ) : ((2 * (n + 2) - 2) / 2 = 7) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_puzzle_l830_83094


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l830_83041

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The octagon formed by connecting midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem stating that the area of the midpoint octagon is 1/4 of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 / 4) * area o :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l830_83041


namespace NUMINAMATH_CALUDE_runners_meet_count_l830_83014

-- Constants
def total_time : ℝ := 45
def odell_speed : ℝ := 260
def odell_radius : ℝ := 70
def kershaw_speed : ℝ := 320
def kershaw_radius : ℝ := 80
def kershaw_delay : ℝ := 5

-- Theorem statement
theorem runners_meet_count :
  let odell_angular_speed := odell_speed / odell_radius
  let kershaw_angular_speed := kershaw_speed / kershaw_radius
  let relative_angular_speed := odell_angular_speed + kershaw_angular_speed
  let effective_time := total_time - kershaw_delay
  let meet_count := ⌊(effective_time * relative_angular_speed) / (2 * Real.pi)⌋
  meet_count = 49 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_count_l830_83014


namespace NUMINAMATH_CALUDE_exam_max_score_l830_83035

/-- The maximum score awarded in an exam given the following conditions:
    1. Gibi scored 59 percent
    2. Jigi scored 55 percent
    3. Mike scored 99 percent
    4. Lizzy scored 67 percent
    5. The average mark scored by all 4 students is 490 -/
theorem exam_max_score :
  let gibi_percent : ℚ := 59 / 100
  let jigi_percent : ℚ := 55 / 100
  let mike_percent : ℚ := 99 / 100
  let lizzy_percent : ℚ := 67 / 100
  let num_students : ℕ := 4
  let average_score : ℚ := 490
  let total_score : ℚ := average_score * num_students
  let sum_percentages : ℚ := gibi_percent + jigi_percent + mike_percent + lizzy_percent
  max_score * sum_percentages = total_score →
  max_score = 700 := by
sorry


end NUMINAMATH_CALUDE_exam_max_score_l830_83035


namespace NUMINAMATH_CALUDE_ambiguous_date_and_longest_periods_l830_83011

/-- Represents a date in DD/MM format -/
structure Date :=
  (day : Nat)
  (month : Nat)

/-- Checks if a date is valid in both DD/MM and MM/DD formats -/
def Date.isAmbiguous (d : Date) : Prop :=
  d.day ≤ 12 ∧ d.month ≤ 12 ∧ d.day ≠ d.month

/-- Checks if a date is within the range of January 2nd to January 12th or December 2nd to December 12th -/
def Date.isInLongestAmbiguousPeriod (d : Date) : Prop :=
  (d.month = 1 ∧ d.day ≥ 2 ∧ d.day ≤ 12) ∨ (d.month = 12 ∧ d.day ≥ 2 ∧ d.day ≤ 12)

theorem ambiguous_date_and_longest_periods :
  (∃ d : Date, d.day = 3 ∧ d.month = 12 ∧ d.isAmbiguous) ∧
  (∀ d : Date, d.isAmbiguous → d.isInLongestAmbiguousPeriod ∨ ¬d.isInLongestAmbiguousPeriod) ∧
  (∀ d : Date, d.isInLongestAmbiguousPeriod → d.isAmbiguous) :=
sorry

end NUMINAMATH_CALUDE_ambiguous_date_and_longest_periods_l830_83011


namespace NUMINAMATH_CALUDE_expression_value_l830_83023

theorem expression_value : 105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l830_83023


namespace NUMINAMATH_CALUDE_coffee_beans_remaining_l830_83082

theorem coffee_beans_remaining (jar_weight empty_weight full_weight remaining_weight : ℝ)
  (h1 : empty_weight = 0.2 * full_weight)
  (h2 : remaining_weight = 0.6 * full_weight)
  (h3 : empty_weight > 0)
  (h4 : full_weight > empty_weight) : 
  let beans_weight := full_weight - empty_weight
  let defective_weight := 0.1 * beans_weight
  let remaining_beans := remaining_weight - empty_weight
  (remaining_beans - defective_weight) / (beans_weight - defective_weight) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_coffee_beans_remaining_l830_83082


namespace NUMINAMATH_CALUDE_diana_weekly_earnings_l830_83030

/-- Represents Diana's work schedule and earnings --/
structure DianaWork where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  hourly_rate : ℕ

/-- Calculates Diana's weekly earnings based on her work schedule --/
def weekly_earnings (d : DianaWork) : ℕ :=
  (d.monday_hours + d.tuesday_hours + d.wednesday_hours + d.thursday_hours + d.friday_hours) * d.hourly_rate

/-- Diana's actual work schedule --/
def diana : DianaWork :=
  { monday_hours := 10
    tuesday_hours := 15
    wednesday_hours := 10
    thursday_hours := 15
    friday_hours := 10
    hourly_rate := 30 }

/-- Theorem stating that Diana's weekly earnings are $1800 --/
theorem diana_weekly_earnings :
  weekly_earnings diana = 1800 := by
  sorry


end NUMINAMATH_CALUDE_diana_weekly_earnings_l830_83030


namespace NUMINAMATH_CALUDE_first_tier_tax_percentage_l830_83004

theorem first_tier_tax_percentage
  (first_tier_limit : ℝ)
  (second_tier_rate : ℝ)
  (car_price : ℝ)
  (total_tax : ℝ)
  (h1 : first_tier_limit = 11000)
  (h2 : second_tier_rate = 0.09)
  (h3 : car_price = 18000)
  (h4 : total_tax = 1950) :
  ∃ first_tier_rate : ℝ,
    first_tier_rate = 0.12 ∧
    total_tax = first_tier_rate * first_tier_limit +
                second_tier_rate * (car_price - first_tier_limit) := by
  sorry

end NUMINAMATH_CALUDE_first_tier_tax_percentage_l830_83004


namespace NUMINAMATH_CALUDE_january_savings_l830_83016

def savings_sequence (initial : ℝ) (n : ℕ) : ℝ :=
  initial + 4 * (n - 1)

def total_savings (initial : ℝ) (months : ℕ) : ℝ :=
  (List.range months).map (savings_sequence initial) |>.sum

theorem january_savings (x : ℝ) : total_savings x 6 = 126 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_january_savings_l830_83016


namespace NUMINAMATH_CALUDE_expression_equals_1997_with_ten_threes_l830_83038

theorem expression_equals_1997_with_ten_threes : 
  ∃ (a b c d e f g h i j : ℕ), 
    a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 3 ∧ e = 3 ∧ f = 3 ∧ g = 3 ∧ h = 3 ∧ i = 3 ∧ j = 3 ∧
    a * (b * 111 + c) + d * (e * 111 + f) - g / h = 1997 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_1997_with_ten_threes_l830_83038


namespace NUMINAMATH_CALUDE_sector_area_l830_83045

/-- The area of a circular sector with radius R and circumference 4R is R^2 -/
theorem sector_area (R : ℝ) (R_pos : R > 0) : 
  let circumference := 4 * R
  let arc_length := circumference - 2 * R
  let sector_area := (1 / 2) * arc_length * R
  sector_area = R^2 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l830_83045


namespace NUMINAMATH_CALUDE_certain_number_proof_l830_83021

theorem certain_number_proof (x : ℝ) : 0.28 * x + 0.45 * 250 = 224.5 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l830_83021


namespace NUMINAMATH_CALUDE_three_W_five_l830_83098

-- Define the operation W
def W (a b : ℤ) : ℤ := b + 7 * a - a ^ 2

-- Theorem to prove
theorem three_W_five : W 3 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_three_W_five_l830_83098


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_tangent_point_property_l830_83078

/-- A quadrilateral inscribed in a circle with an inscribed circle inside it. -/
structure InscribedQuadrilateral where
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ
  /-- The quadrilateral is inscribed in a circle -/
  inscribed_in_circle : Bool
  /-- There is a circle inscribed in the quadrilateral -/
  has_inscribed_circle : Bool

/-- The point of tangency divides a side into two segments -/
def tangent_point_division (q : InscribedQuadrilateral) : ℝ × ℝ := sorry

/-- The theorem stating the property of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_tangent_point_property 
  (q : InscribedQuadrilateral)
  (h1 : q.sides 0 = 65)
  (h2 : q.sides 1 = 95)
  (h3 : q.sides 2 = 125)
  (h4 : q.sides 3 = 105)
  (h5 : q.inscribed_in_circle = true)
  (h6 : q.has_inscribed_circle = true) :
  let (x, y) := tangent_point_division q
  |x - y| = 14 := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_tangent_point_property_l830_83078


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l830_83022

theorem units_digit_of_7_pow_2050 : 7^2050 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l830_83022


namespace NUMINAMATH_CALUDE_decimal_expansion_2023rd_digit_l830_83063

/-- The decimal expansion of 7/26 -/
def decimal_expansion : ℚ := 7 / 26

/-- The length of the repeating block in the decimal expansion of 7/26 -/
def repeating_block_length : ℕ := 9

/-- The position of the 2023rd digit within the repeating block -/
def position_in_block : ℕ := 2023 % repeating_block_length

/-- The 2023rd digit past the decimal point in the decimal expansion of 7/26 -/
def digit_2023 : ℕ := 3

theorem decimal_expansion_2023rd_digit :
  digit_2023 = 3 :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_2023rd_digit_l830_83063


namespace NUMINAMATH_CALUDE_max_triangle_area_l830_83036

/-- Parabola with focus at (0,1) and equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (0, 1)

/-- Vector from F to a point -/
def vec (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - F.1, p.2 - F.2)

/-- Condition that A, B, C are on the parabola and FA + FB + FC = 0 -/
def PointsCondition (A B C : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ C ∈ Parabola ∧
  vec A + vec B + vec C = (0, 0)

/-- Area of a triangle given three points -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The theorem to be proved -/
theorem max_triangle_area :
  ∀ A B C : ℝ × ℝ,
  PointsCondition A B C →
  TriangleArea A B C ≤ (3 * Real.sqrt 6) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l830_83036


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l830_83013

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (rows : ℕ) : Type := Unit

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n - 1

/-- The probability of randomly selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ := countOnes n / totalElements n

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 39 / 210 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l830_83013


namespace NUMINAMATH_CALUDE_third_book_words_l830_83053

-- Define the given constants
def days : ℕ := 10
def books : ℕ := 3
def reading_speed : ℕ := 100 -- words per hour
def first_book_words : ℕ := 200
def second_book_words : ℕ := 400
def average_reading_time : ℕ := 54 -- minutes per day

-- Define the theorem
theorem third_book_words :
  let total_reading_time : ℕ := days * average_reading_time
  let total_reading_hours : ℕ := total_reading_time / 60
  let total_words : ℕ := total_reading_hours * reading_speed
  let first_two_books_words : ℕ := first_book_words + second_book_words
  total_words - first_two_books_words = 300 := by
  sorry

end NUMINAMATH_CALUDE_third_book_words_l830_83053


namespace NUMINAMATH_CALUDE_correct_mean_after_error_correction_l830_83062

theorem correct_mean_after_error_correction (n : ℕ) (initial_mean : ℚ) (wrong_value correct_value : ℚ) :
  n = 30 →
  initial_mean = 250 →
  wrong_value = 135 →
  correct_value = 165 →
  (n : ℚ) * initial_mean + (correct_value - wrong_value) = n * 251 :=
by sorry

end NUMINAMATH_CALUDE_correct_mean_after_error_correction_l830_83062


namespace NUMINAMATH_CALUDE_union_of_sets_l830_83039

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {2, 4, 5, 6}
  A ∪ B = {1, 2, 3, 4, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l830_83039


namespace NUMINAMATH_CALUDE_find_g_of_x_l830_83018

/-- Given that 4x^4 + 2x^2 - 7x + 3 + g(x) = 5x^3 - 8x^2 + 4x - 1,
    prove that g(x) = -4x^4 + 5x^3 - 10x^2 + 11x - 4 -/
theorem find_g_of_x (g : ℝ → ℝ) :
  (∀ x : ℝ, 4 * x^4 + 2 * x^2 - 7 * x + 3 + g x = 5 * x^3 - 8 * x^2 + 4 * x - 1) →
  (∀ x : ℝ, g x = -4 * x^4 + 5 * x^3 - 10 * x^2 + 11 * x - 4) :=
by sorry

end NUMINAMATH_CALUDE_find_g_of_x_l830_83018


namespace NUMINAMATH_CALUDE_lindas_savings_l830_83074

theorem lindas_savings (savings : ℝ) : 
  (2 / 3 : ℝ) * savings + 250 = savings → savings = 750 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l830_83074


namespace NUMINAMATH_CALUDE_min_value_of_expression_l830_83007

/-- Given a line mx + ny + 2 = 0 intersecting a circle (x+3)^2 + (y+1)^2 = 1 at a chord of length 2,
    the minimum value of 1/m + 3/n is 6, where m > 0 and n > 0 -/
theorem min_value_of_expression (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), m*x + n*y + 2 = 0 ∧ (x+3)^2 + (y+1)^2 = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), m*x₁ + n*y₁ + 2 = 0 ∧ m*x₂ + n*y₂ + 2 = 0 ∧
                         (x₁+3)^2 + (y₁+1)^2 = 1 ∧ (x₂+3)^2 + (y₂+1)^2 = 1 ∧
                         (x₁-x₂)^2 + (y₁-y₂)^2 = 4) →
  (1/m + 3/n ≥ 6) ∧ (∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 1/m₀ + 3/n₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l830_83007


namespace NUMINAMATH_CALUDE_max_candy_leftover_l830_83017

theorem max_candy_leftover (x : ℕ+) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ 0 < r ∧ r ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l830_83017


namespace NUMINAMATH_CALUDE_steve_initial_berries_l830_83069

theorem steve_initial_berries (stacy_initial : ℕ) (steve_takes : ℕ) (difference : ℕ) : 
  stacy_initial = 32 →
  steve_takes = 4 →
  stacy_initial - (steve_takes + difference) = stacy_initial - 7 →
  stacy_initial - difference - steve_takes = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_steve_initial_berries_l830_83069


namespace NUMINAMATH_CALUDE_y_intercept_from_slope_and_x_intercept_l830_83060

/-- A line with given slope and x-intercept has a specific y-intercept -/
theorem y_intercept_from_slope_and_x_intercept 
  (slope : ℝ) (x_intercept : ℝ) (y_intercept : ℝ) :
  slope = -3 →
  x_intercept = 4 →
  y_intercept = slope * 0 + (- slope * x_intercept) →
  (0, y_intercept) = (0, 12) := by
sorry

end NUMINAMATH_CALUDE_y_intercept_from_slope_and_x_intercept_l830_83060


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l830_83026

/-- For a quadratic equation (k+2)x^2 + 4x + 1 = 0 to have two distinct real roots, 
    k must satisfy: k < 2 and k ≠ -2 -/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k + 2) * x^2 + 4 * x + 1 = 0 ∧ 
   (k + 2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l830_83026


namespace NUMINAMATH_CALUDE_intersection_and_lines_l830_83084

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the intersection point A(m, n)
def m : ℝ := -2
def n : ℝ := 3

-- Define line l
def l (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- Theorem statement
theorem intersection_and_lines :
  (l₁ m n ∧ l₂ m n) ∧  -- A is the intersection of l₁ and l₂
  (∀ x y : ℝ, x + 2 * y - 4 = 0 ↔ (x - m) * 2 + (y - n) * 1 = 0) ∧  -- l₃ equation
  (∀ x y : ℝ, 2 * x - 3 * y + 13 = 0 ↔ (y - n) = (2 / 3) * (x - m)) :=  -- l₄ equation
by sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l830_83084


namespace NUMINAMATH_CALUDE_rocket_height_problem_l830_83089

theorem rocket_height_problem (h : ℝ) : 
  h + 2 * h = 1500 → h = 500 := by
  sorry

end NUMINAMATH_CALUDE_rocket_height_problem_l830_83089


namespace NUMINAMATH_CALUDE_max_valid_config_l830_83046

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  white : Nat
  black : Nat

/-- Checks if a configuration is valid for an 8x8 chessboard -/
def is_valid_config (config : ChessboardConfig) : Prop :=
  config.white + config.black ≤ 64 ∧
  config.white = 2 * config.black ∧
  (config.white + config.black) % 8 = 0

/-- The maximum valid configuration -/
def max_config : ChessboardConfig :=
  ⟨32, 16⟩

/-- Theorem: The maximum valid configuration is (32, 16) -/
theorem max_valid_config :
  is_valid_config max_config ∧
  ∀ (c : ChessboardConfig), is_valid_config c → c.white ≤ max_config.white ∧ c.black ≤ max_config.black :=
by sorry


end NUMINAMATH_CALUDE_max_valid_config_l830_83046


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l830_83096

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^2 - 22 * X + 63 = (X - 3) * q + 24 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l830_83096


namespace NUMINAMATH_CALUDE_servant_worked_nine_months_l830_83003

/-- Represents the salary and work duration of a servant --/
structure ServantSalary where
  yearly_cash : ℕ  -- Yearly cash salary in Rupees
  turban_value : ℕ  -- Value of the turban in Rupees
  received_cash : ℕ  -- Cash received when leaving in Rupees
  months_worked : ℕ  -- Number of months worked

/-- Calculates the number of months a servant worked based on their salary structure --/
def calculate_months_worked (s : ServantSalary) : ℕ :=
  ((s.received_cash + s.turban_value) * 12) / (s.yearly_cash + s.turban_value)

/-- Theorem stating that under the given conditions, the servant worked for 9 months --/
theorem servant_worked_nine_months (s : ServantSalary) 
  (h1 : s.yearly_cash = 90)
  (h2 : s.turban_value = 90)
  (h3 : s.received_cash = 45) :
  calculate_months_worked s = 9 := by
  sorry

end NUMINAMATH_CALUDE_servant_worked_nine_months_l830_83003


namespace NUMINAMATH_CALUDE_one_pencil_two_pens_cost_l830_83028

/-- The cost of a single pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a single pen -/
def pen_cost : ℝ := sorry

/-- The first given condition: three pencils and four pens cost $3.20 -/
axiom condition1 : 3 * pencil_cost + 4 * pen_cost = 3.20

/-- The second given condition: two pencils and three pens cost $2.50 -/
axiom condition2 : 2 * pencil_cost + 3 * pen_cost = 2.50

/-- Theorem stating that one pencil and two pens cost $1.80 -/
theorem one_pencil_two_pens_cost : pencil_cost + 2 * pen_cost = 1.80 := by sorry

end NUMINAMATH_CALUDE_one_pencil_two_pens_cost_l830_83028


namespace NUMINAMATH_CALUDE_inequality_solution_set_l830_83076

theorem inequality_solution_set (x : ℝ) : (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l830_83076
