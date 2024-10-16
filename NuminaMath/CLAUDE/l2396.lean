import Mathlib

namespace NUMINAMATH_CALUDE_max_min_difference_l2396_239684

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Define the interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem max_min_difference :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧
               (∀ x ∈ I, m ≤ f x) ∧
               (M - m = 32) :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_l2396_239684


namespace NUMINAMATH_CALUDE_add_negative_two_l2396_239631

theorem add_negative_two : 1 + (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_add_negative_two_l2396_239631


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l2396_239675

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l2396_239675


namespace NUMINAMATH_CALUDE_sqrt_53_plus_20_sqrt_7_representation_l2396_239698

theorem sqrt_53_plus_20_sqrt_7_representation : 
  ∃ (a b c : ℤ), 
    (∀ (n : ℕ), n > 1 → ¬ (∃ (k : ℕ), c = n^2 * k)) → 
    Real.sqrt (53 + 20 * Real.sqrt 7) = a + b * Real.sqrt c ∧ 
    a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_53_plus_20_sqrt_7_representation_l2396_239698


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2396_239624

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 20500 →
  b % 5 = 0 →
  b = 10 * a + 5 →
  b - a = 16777 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2396_239624


namespace NUMINAMATH_CALUDE_exactly_six_solutions_l2396_239679

/-- The number of ordered pairs of positive integers satisfying 3/m + 6/n = 1 -/
def solution_count : ℕ := 6

/-- Predicate for ordered pairs (m,n) satisfying the equation -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (3 : ℚ) / m.val + (6 : ℚ) / n.val = 1

/-- The theorem stating that there are exactly 6 solutions -/
theorem exactly_six_solutions :
  ∃! (s : Finset (ℕ+ × ℕ+)), 
    s.card = solution_count ∧ 
    (∀ p ∈ s, satisfies_equation p.1 p.2) ∧
    (∀ m n : ℕ+, satisfies_equation m n → (m, n) ∈ s) :=
  sorry

end NUMINAMATH_CALUDE_exactly_six_solutions_l2396_239679


namespace NUMINAMATH_CALUDE_kickball_total_players_kickball_problem_l2396_239670

theorem kickball_total_players (wed_morning : ℕ) (wed_afternoon_increase : ℕ) 
  (thu_morning_decrease : ℕ) (thu_lunchtime_decrease : ℕ) : ℕ :=
  let wed_afternoon := wed_morning + wed_afternoon_increase
  let thu_morning := wed_morning - thu_morning_decrease
  let thu_afternoon := thu_morning - thu_lunchtime_decrease
  let wed_total := wed_morning + wed_afternoon
  let thu_total := thu_morning + thu_afternoon
  wed_total + thu_total

theorem kickball_problem :
  kickball_total_players 37 15 9 7 = 138 := by
  sorry

end NUMINAMATH_CALUDE_kickball_total_players_kickball_problem_l2396_239670


namespace NUMINAMATH_CALUDE_distance_between_points_l2396_239643

theorem distance_between_points : Real.sqrt ((24 - 0)^2 + (0 - 10)^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2396_239643


namespace NUMINAMATH_CALUDE_exercise_book_distribution_l2396_239678

theorem exercise_book_distribution (total_books : ℕ) (num_classes : ℕ) 
  (h1 : total_books = 338) (h2 : num_classes = 3) :
  ∃ (books_per_class : ℕ) (books_left : ℕ),
    books_per_class = 112 ∧ 
    books_left = 2 ∧
    total_books = books_per_class * num_classes + books_left :=
by sorry

end NUMINAMATH_CALUDE_exercise_book_distribution_l2396_239678


namespace NUMINAMATH_CALUDE_complex_division_equality_l2396_239659

/-- Given that i is the imaginary unit, prove that (2 + 4i) / (1 + i) = 3 + i -/
theorem complex_division_equality : (2 + 4 * I) / (1 + I) = 3 + I := by sorry

end NUMINAMATH_CALUDE_complex_division_equality_l2396_239659


namespace NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_maximum_value_l2396_239628

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x + 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 12

-- Theorem for the tangent line equation
theorem tangent_line_equation : 
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ y - f 1 = f' 1 * (x - 1) := by sorry

-- Theorem for the minimum value
theorem minimum_value : 
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -14 ∧ ∀ y ∈ Set.Icc (-3 : ℝ) 3, f y ≥ f x := by sorry

-- Theorem for the maximum value
theorem maximum_value : 
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 18 ∧ ∀ y ∈ Set.Icc (-3 : ℝ) 3, f y ≤ f x := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_maximum_value_l2396_239628


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2396_239618

theorem complex_equation_solution :
  ∀ z : ℂ, z = Complex.I * (2 - z) → z = 1 + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2396_239618


namespace NUMINAMATH_CALUDE_summer_camp_two_talents_l2396_239621

/-- Represents the number of students with various talents in a summer camp. -/
structure SummerCamp where
  total : ℕ
  cannotSing : ℕ
  cannotDance : ℕ
  cannotAct : ℕ

/-- Calculates the number of students with two talents. -/
def studentsWithTwoTalents (camp : SummerCamp) : ℕ :=
  let canSing := camp.total - camp.cannotSing
  let canDance := camp.total - camp.cannotDance
  let canAct := camp.total - camp.cannotAct
  canSing + canDance + canAct - camp.total

/-- The main theorem stating that in the given summer camp, 64 students have two talents. -/
theorem summer_camp_two_talents :
  let camp := SummerCamp.mk 100 42 65 29
  studentsWithTwoTalents camp = 64 := by
  sorry


end NUMINAMATH_CALUDE_summer_camp_two_talents_l2396_239621


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l2396_239651

theorem fixed_point_power_function (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 4 = 2 →
  f 16 = 4 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l2396_239651


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2396_239688

/-- Given two points M and N in a 2D plane, where N is symmetric to M about the y-axis,
    this theorem proves that the coordinates of M with respect to N are (2, 1) when M has coordinates (-2, 1). -/
theorem symmetric_point_coordinates (M N : ℝ × ℝ) :
  M = (-2, 1) →
  N.1 = -M.1 ∧ N.2 = M.2 →
  (M.1 - N.1, M.2 - N.2) = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2396_239688


namespace NUMINAMATH_CALUDE_linear_dependency_condition_l2396_239683

-- Define the vectors
def v1 : Fin 2 → ℝ := ![2, 4]
def v2 (k : ℝ) : Fin 2 → ℝ := ![1, k]

-- Define linear dependency
def is_linearly_dependent (v1 v2 : Fin 2 → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a • v1 + b • v2 = 0)

-- Theorem statement
theorem linear_dependency_condition (k : ℝ) :
  is_linearly_dependent v1 (v2 k) ↔ k = 2 :=
sorry

end NUMINAMATH_CALUDE_linear_dependency_condition_l2396_239683


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_l2396_239634

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_l2396_239634


namespace NUMINAMATH_CALUDE_problem_solution_l2396_239686

theorem problem_solution : (3/4)^2017 * (-1-1/3)^2018 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2396_239686


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2396_239613

def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℤ) (q : ℤ) 
  (h_seq : geometric_sequence a q)
  (h_a4 : a 4 = 27)
  (h_q : q = -3) :
  a 7 = -729 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2396_239613


namespace NUMINAMATH_CALUDE_probability_three_of_a_kind_after_reroll_l2396_239669

/-- The probability of getting at least three dice showing the same value after re-rolling the unmatched die in a specific dice configuration. -/
theorem probability_three_of_a_kind_after_reroll (n : ℕ) (p : ℚ) : 
  n = 5 → -- number of dice
  p = 1 / 3 → -- probability we want to prove
  ∃ (X Y : ℕ), -- the two pair values
    X ≠ Y ∧ 
    X ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧ 
    Y ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
  p = (1 : ℚ) / 6 + (1 : ℚ) / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_of_a_kind_after_reroll_l2396_239669


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2396_239664

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 2) * x₁^2 - 2 * x₁ + (1/2) = 0 ∧ 
    (k - 2) * x₂^2 - 2 * x₂ + (1/2) = 0) ↔ 
  (k < 4 ∧ k ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2396_239664


namespace NUMINAMATH_CALUDE_allocation_methods_count_l2396_239658

def number_of_doctors : ℕ := 3
def number_of_nurses : ℕ := 6
def number_of_schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

theorem allocation_methods_count :
  (Nat.choose number_of_doctors doctors_per_school) *
  (Nat.choose number_of_nurses nurses_per_school) *
  (Nat.choose (number_of_doctors - doctors_per_school) doctors_per_school) *
  (Nat.choose (number_of_nurses - nurses_per_school) nurses_per_school) = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_methods_count_l2396_239658


namespace NUMINAMATH_CALUDE_boys_at_least_35_percent_l2396_239660

/-- Represents a child camp with 3-rooms and 4-rooms -/
structure ChildCamp where
  girls_3room : ℕ
  girls_4room : ℕ
  boys_3room : ℕ
  boys_4room : ℕ

/-- The proportion of boys in the camp -/
def boy_proportion (camp : ChildCamp) : ℚ :=
  (3 * camp.boys_3room + 4 * camp.boys_4room) / 
  (3 * camp.girls_3room + 4 * camp.girls_4room + 3 * camp.boys_3room + 4 * camp.boys_4room)

/-- Theorem stating that the proportion of boys is at least 35% -/
theorem boys_at_least_35_percent (camp : ChildCamp) 
  (h1 : 2 * (camp.girls_4room + camp.boys_4room) ≥ 
        camp.girls_3room + camp.girls_4room + camp.boys_3room + camp.boys_4room)
  (h2 : 3 * camp.girls_3room ≥ 8 * camp.girls_4room) :
  boy_proportion camp ≥ 7/20 := by
  sorry

end NUMINAMATH_CALUDE_boys_at_least_35_percent_l2396_239660


namespace NUMINAMATH_CALUDE_four_numbers_between_l2396_239642

theorem four_numbers_between :
  ∃ (a b c d : ℝ), 5.45 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 5.47 := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_between_l2396_239642


namespace NUMINAMATH_CALUDE_equation_solution_l2396_239666

theorem equation_solution :
  ∃! x : ℝ, x ≠ -3 ∧ (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2396_239666


namespace NUMINAMATH_CALUDE_altitude_inscribed_radius_relation_l2396_239694

-- Define a triangle type
structure Triangle where
  -- Three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure the triangle inequality holds
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the altitudes of the triangle
def altitude (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define the inscribed circle radius
def inscribed_radius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem altitude_inscribed_radius_relation (t : Triangle) :
  let (h₁, h₂, h₃) := altitude t
  let r := inscribed_radius t
  1 / h₁ + 1 / h₂ + 1 / h₃ = 1 / r := by sorry

end NUMINAMATH_CALUDE_altitude_inscribed_radius_relation_l2396_239694


namespace NUMINAMATH_CALUDE_maintenance_check_time_l2396_239608

/-- 
Proves that if an additive doubles the time between maintenance checks 
and the new time is 60 days, then the original time was 30 days.
-/
theorem maintenance_check_time (original_time : ℕ) : 
  (2 * original_time = 60) → original_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l2396_239608


namespace NUMINAMATH_CALUDE_smallest_steps_l2396_239674

theorem smallest_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 3 →
  n ≥ 59 :=
by sorry

end NUMINAMATH_CALUDE_smallest_steps_l2396_239674


namespace NUMINAMATH_CALUDE_train_length_equals_distance_traveled_l2396_239682

/-- Calculates the length of a train based on its speed and the time it takes to pass through a tunnel. -/
def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem stating that the length of a train is equal to the distance it travels while passing through a tunnel. -/
theorem train_length_equals_distance_traveled (speed : ℝ) (time : ℝ) :
  train_length speed time = speed * time :=
by
  sorry

#check train_length_equals_distance_traveled

end NUMINAMATH_CALUDE_train_length_equals_distance_traveled_l2396_239682


namespace NUMINAMATH_CALUDE_consecutive_non_prime_non_prime_power_l2396_239633

/-- For any positive integer n, there exists a positive integer k such that 
    for all i in {1, ..., n}, k + i is neither prime nor a prime power. -/
theorem consecutive_non_prime_non_prime_power (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (k + i)) ∧ ¬(∃ p m : ℕ, Nat.Prime p ∧ 1 < m ∧ k + i = p^m) :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_non_prime_power_l2396_239633


namespace NUMINAMATH_CALUDE_cat_grooming_time_is_640_l2396_239695

/-- Represents the time taken to groom a cat -/
def catGroomingTime (
  clipTime : ℕ)  -- Time to clip one nail in seconds
  (cleanEarTime : ℕ)  -- Time to clean one ear in seconds
  (shampooTime : ℕ)  -- Time to shampoo in minutes
  (clawsPerFoot : ℕ)  -- Number of claws per foot
  (feetCount : ℕ)  -- Number of feet
  (earCount : ℕ)  -- Number of ears
  (secondsPerMinute : ℕ)  -- Number of seconds in a minute
  : ℕ :=
  (clipTime * clawsPerFoot * feetCount) +  -- Time for clipping nails
  (cleanEarTime * earCount) +  -- Time for cleaning ears
  (shampooTime * secondsPerMinute)  -- Time for shampooing

theorem cat_grooming_time_is_640 :
  catGroomingTime 10 90 5 4 4 2 60 = 640 := by
  sorry

#eval catGroomingTime 10 90 5 4 4 2 60

end NUMINAMATH_CALUDE_cat_grooming_time_is_640_l2396_239695


namespace NUMINAMATH_CALUDE_max_silver_medals_for_27_points_l2396_239603

/-- Represents the types of medals in the competition -/
inductive Medal
| Gold
| Silver
| Bronze

/-- Returns the point value of a given medal -/
def medal_points (m : Medal) : Nat :=
  match m with
  | Medal.Gold => 5
  | Medal.Silver => 3
  | Medal.Bronze => 1

/-- Represents a competitor's medal collection -/
structure MedalCollection where
  gold : Nat
  silver : Nat
  bronze : Nat

/-- Calculates the total points for a given medal collection -/
def total_points (mc : MedalCollection) : Nat :=
  mc.gold * medal_points Medal.Gold +
  mc.silver * medal_points Medal.Silver +
  mc.bronze * medal_points Medal.Bronze

/-- The main theorem to prove -/
theorem max_silver_medals_for_27_points :
  ∃ (mc : MedalCollection),
    total_points mc = 27 ∧
    mc.gold + mc.silver + mc.bronze ≤ 8 ∧
    mc.silver = 4 ∧
    ∀ (mc' : MedalCollection),
      total_points mc' = 27 →
      mc'.gold + mc'.silver + mc'.bronze ≤ 8 →
      mc'.silver ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_max_silver_medals_for_27_points_l2396_239603


namespace NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l2396_239627

theorem average_weight_of_class (group1_count : ℕ) (group1_avg : ℚ) 
                                 (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count : ℕ := group1_count + group2_count
  let total_weight : ℚ := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count

theorem class_average_weight :
  average_weight_of_class 26 (50.25 : ℚ) 8 (45.15 : ℚ) = (49.05 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l2396_239627


namespace NUMINAMATH_CALUDE_intersection_B_complement_A_l2396_239696

def I : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 3}

theorem intersection_B_complement_A : B ∩ (I \ A) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_B_complement_A_l2396_239696


namespace NUMINAMATH_CALUDE_equality_sum_l2396_239653

theorem equality_sum (M N : ℚ) : 
  (3 / 5 : ℚ) = M / 75 ∧ (3 / 5 : ℚ) = 90 / N → M + N = 195 := by
  sorry

end NUMINAMATH_CALUDE_equality_sum_l2396_239653


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2396_239685

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 160) :
  |x - y| = 2 * Real.sqrt 65 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2396_239685


namespace NUMINAMATH_CALUDE_translation_theorem_l2396_239606

/-- Represents a quadratic function of the form y = a(x-h)^2 + k --/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a quadratic function horizontally --/
def translate (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h - d, k := f.k }

/-- The initial quadratic function y = 3(x-2)^2 + 1 --/
def initial_function : QuadraticFunction :=
  { a := 3, h := 2, k := 1 }

/-- Theorem stating that translating the initial function 2 units right then 2 units left
    results in y = 3x^2 + 3 --/
theorem translation_theorem :
  let f1 := translate initial_function (-2)
  let f2 := translate f1 2
  f2.a * (X - f2.h)^2 + f2.k = 3 * X^2 + 3 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l2396_239606


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2396_239649

theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 3 ∧ (2 / (x - 3) + (x + m) / (3 - x) = 2)) →
  m ≤ 8 ∧ m ≠ -1 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2396_239649


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2396_239677

/-- The speed of a boat in still water, given its speed with and against a stream. -/
theorem boat_speed_in_still_water (along_stream speed_against_stream : ℝ) 
  (h1 : along_stream = 15)
  (h2 : speed_against_stream = 5) :
  (along_stream + speed_against_stream) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2396_239677


namespace NUMINAMATH_CALUDE_barbara_candies_l2396_239661

/-- The number of candies Barbara used -/
def candies_used (initial : ℝ) (remaining : ℕ) : ℝ :=
  initial - remaining

theorem barbara_candies : 
  let initial : ℝ := 18.0
  let remaining : ℕ := 9
  candies_used initial remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_l2396_239661


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_sixth_powers_l2396_239691

theorem quadratic_roots_sum_of_sixth_powers :
  ∀ p q : ℝ,
  (p^2 - 2*p*Real.sqrt 3 + 2 = 0) →
  (q^2 - 2*q*Real.sqrt 3 + 2 = 0) →
  p^6 + q^6 = 3120 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_sixth_powers_l2396_239691


namespace NUMINAMATH_CALUDE_smallest_square_division_l2396_239655

/-- A structure representing a division of a square into smaller squares. -/
structure SquareDivision (n : ℕ) :=
  (num_40 : ℕ)  -- number of 40x40 squares
  (num_49 : ℕ)  -- number of 49x49 squares
  (valid : 40 * num_40 + 49 * num_49 = n)
  (non_empty : num_40 > 0 ∧ num_49 > 0)

/-- The theorem stating that 2000 is the smallest n that satisfies the conditions. -/
theorem smallest_square_division :
  (∃ (d : SquareDivision 2000), True) ∧
  (∀ m : ℕ, m < 2000 → ¬∃ (d : SquareDivision m), True) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_division_l2396_239655


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l2396_239616

theorem sum_of_squares_problem (a b c d k p : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 390)
  (h2 : a*b + b*c + c*a + a*d + b*d + c*d = 5)
  (h3 : a*d + b*d + c*d = k)
  (h4 : a^2*b^2*c^2*d^2 = p) :
  a + b + c + d = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l2396_239616


namespace NUMINAMATH_CALUDE_lions_escaped_l2396_239697

/-- The number of rhinos that escaped -/
def rhinos : ℕ := 2

/-- The time (in hours) to recover each animal -/
def recovery_time : ℕ := 2

/-- The total time (in hours) spent recovering all animals -/
def total_time : ℕ := 10

/-- The number of lions that escaped -/
def lions : ℕ := (total_time - rhinos * recovery_time) / recovery_time

theorem lions_escaped :
  lions = 3 :=
by sorry

end NUMINAMATH_CALUDE_lions_escaped_l2396_239697


namespace NUMINAMATH_CALUDE_function_satisfying_conditions_l2396_239629

theorem function_satisfying_conditions (f : ℚ → ℚ) 
  (h1 : f 0 = 0) 
  (h2 : ∀ x y : ℚ, f (f x + f y) = x + y) : 
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_conditions_l2396_239629


namespace NUMINAMATH_CALUDE_school_choir_robe_cost_l2396_239681

/-- Calculates the total cost of buying additional robes for a school choir, including discount and sales tax. -/
theorem school_choir_robe_cost
  (total_robes_needed : ℕ)
  (robes_owned : ℕ)
  (cost_per_robe : ℚ)
  (discount_rate : ℚ)
  (discount_threshold : ℕ)
  (sales_tax_rate : ℚ)
  (h1 : total_robes_needed = 30)
  (h2 : robes_owned = 12)
  (h3 : cost_per_robe = 2)
  (h4 : discount_rate = 15 / 100)
  (h5 : discount_threshold = 10)
  (h6 : sales_tax_rate = 8 / 100)
  : ∃ (final_cost : ℚ), final_cost = 3305 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_school_choir_robe_cost_l2396_239681


namespace NUMINAMATH_CALUDE_max_identical_trapezoids_l2396_239620

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  upper_base : ℕ
  lower_base : ℕ
  acute_angle : ℝ

/-- The original trapezoid -/
def original : IsoscelesTrapezoid :=
  { upper_base := 2015
  , lower_base := 2016
  , acute_angle := sorry }  -- We don't need the actual angle value for this problem

/-- Predicate for a valid cut-out trapezoid -/
def is_valid_cutout (t : IsoscelesTrapezoid) : Prop :=
  t.lower_base - t.upper_base = 1 ∧
  t.acute_angle = original.acute_angle

/-- The maximum number of identical trapezoids that can be cut out -/
def max_cutouts : ℕ := 4029

/-- The main theorem -/
theorem max_identical_trapezoids :
  ∀ (t : IsoscelesTrapezoid),
    is_valid_cutout t →
    (∀ (n : ℕ), n > max_cutouts →
      ¬ ∃ (ts : Fin n → IsoscelesTrapezoid),
        (∀ i, is_valid_cutout (ts i)) ∧
        (∀ i j, ts i = ts j)) :=
by sorry

end NUMINAMATH_CALUDE_max_identical_trapezoids_l2396_239620


namespace NUMINAMATH_CALUDE_beth_coin_sale_l2396_239665

/-- Given Beth's initial gold coins and Carl's gift, prove the number of coins Beth sold when she sold half her total. -/
theorem beth_coin_sale (initial_coins : ℕ) (gift_coins : ℕ) : 
  initial_coins = 125 → gift_coins = 35 → (initial_coins + gift_coins) / 2 = 80 := by
sorry

end NUMINAMATH_CALUDE_beth_coin_sale_l2396_239665


namespace NUMINAMATH_CALUDE_total_cost_is_660_l2396_239602

/-- Represents the cost of t-shirts for employees -/
structure TShirtCost where
  white_men : ℕ
  black_men : ℕ
  women_discount : ℕ
  total_employees : ℕ

/-- Calculates the total cost of t-shirts given the conditions -/
def total_cost (c : TShirtCost) : ℕ :=
  let employees_per_type := c.total_employees / 4
  let white_men_cost := c.white_men * employees_per_type
  let white_women_cost := (c.white_men - c.women_discount) * employees_per_type
  let black_men_cost := c.black_men * employees_per_type
  let black_women_cost := (c.black_men - c.women_discount) * employees_per_type
  white_men_cost + white_women_cost + black_men_cost + black_women_cost

/-- Theorem stating that the total cost of t-shirts is $660 -/
theorem total_cost_is_660 (c : TShirtCost)
  (h1 : c.white_men = 20)
  (h2 : c.black_men = 18)
  (h3 : c.women_discount = 5)
  (h4 : c.total_employees = 40) :
  total_cost c = 660 := by
  sorry

#eval total_cost { white_men := 20, black_men := 18, women_discount := 5, total_employees := 40 }

end NUMINAMATH_CALUDE_total_cost_is_660_l2396_239602


namespace NUMINAMATH_CALUDE_derek_dogs_at_six_l2396_239601

theorem derek_dogs_at_six (dogs_at_six cars_at_six : ℕ) 
  (h1 : dogs_at_six = 3 * cars_at_six)
  (h2 : cars_at_six + 210 = 2 * 120)
  : dogs_at_six = 90 := by
  sorry

end NUMINAMATH_CALUDE_derek_dogs_at_six_l2396_239601


namespace NUMINAMATH_CALUDE_marble_sum_theorem_l2396_239604

theorem marble_sum_theorem (atticus jensen cruz : ℕ) : 
  atticus = 4 → 
  cruz = 8 → 
  atticus = jensen / 2 → 
  3 * (atticus + jensen + cruz) = 60 := by
  sorry

end NUMINAMATH_CALUDE_marble_sum_theorem_l2396_239604


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2396_239609

theorem cubic_equation_solution (x : ℚ) : (5*x - 2)^3 + 125 = 0 ↔ x = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2396_239609


namespace NUMINAMATH_CALUDE_equation_proof_l2396_239692

theorem equation_proof : 300 * 2 + (12 + 4) * (1 / 8) = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2396_239692


namespace NUMINAMATH_CALUDE_total_jump_distance_l2396_239687

/-- The total distance jumped by a grasshopper and a frog -/
def total_jump (grasshopper_jump frog_jump : ℕ) : ℕ :=
  grasshopper_jump + frog_jump

/-- Theorem: The total jump distance is 66 inches -/
theorem total_jump_distance :
  total_jump 31 35 = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_jump_distance_l2396_239687


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2396_239614

theorem consecutive_integers_sum_of_squares : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a * (a + 1) * (a + 2) = 12 * (3 * a + 3)) → 
  (a^2 + (a + 1)^2 + (a + 2)^2 = 149) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l2396_239614


namespace NUMINAMATH_CALUDE_base4_division_theorem_l2396_239656

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_division_theorem :
  let dividend := [3, 1, 2, 3]  -- 3213₄ in reverse order
  let divisor := [3, 1]         -- 13₄ in reverse order
  let quotient := [1, 0, 2]     -- 201₄ in reverse order
  (base4_to_base10 dividend) / (base4_to_base10 divisor) = base4_to_base10 quotient :=
by sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l2396_239656


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2396_239630

theorem solution_set_inequality (x : ℝ) :
  {x : ℝ | 3 * x - x^2 ≥ 0} = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2396_239630


namespace NUMINAMATH_CALUDE_choose_one_from_each_set_l2396_239640

theorem choose_one_from_each_set : 
  ∀ (novels textbooks : ℕ), 
  novels = 5 → 
  textbooks = 6 → 
  novels * textbooks = 30 := by
sorry

end NUMINAMATH_CALUDE_choose_one_from_each_set_l2396_239640


namespace NUMINAMATH_CALUDE_distance_between_red_lights_l2396_239693

/-- The distance between lights in inches -/
def light_spacing : ℕ := 8

/-- The number of lights in a complete color pattern cycle -/
def pattern_length : ℕ := 2 + 3 + 1

/-- The position of the nth red light in the sequence -/
def red_light_position (n : ℕ) : ℕ :=
  (n - 1) / 2 * pattern_length + (n - 1) % 2 + 1

/-- Convert inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

theorem distance_between_red_lights :
  inches_to_feet (light_spacing * (red_light_position 15 - red_light_position 4)) = 19.3 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_red_lights_l2396_239693


namespace NUMINAMATH_CALUDE_darcys_shorts_l2396_239623

theorem darcys_shorts (total_shirts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) (remaining_to_fold : ℕ) : 
  total_shirts = 20 →
  folded_shirts = 12 →
  folded_shorts = 5 →
  remaining_to_fold = 11 →
  total_shirts + (folded_shorts + (remaining_to_fold - (total_shirts - folded_shirts))) = 28 :=
by sorry

end NUMINAMATH_CALUDE_darcys_shorts_l2396_239623


namespace NUMINAMATH_CALUDE_square_with_semicircular_arcs_perimeter_l2396_239641

/-- The perimeter of a region bounded by semicircular arcs constructed on the sides of a square -/
theorem square_with_semicircular_arcs_perimeter (side_length : Real) : 
  side_length = 4 / Real.pi → 
  (4 : Real) * Real.pi * (side_length / 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_with_semicircular_arcs_perimeter_l2396_239641


namespace NUMINAMATH_CALUDE_peach_basket_problem_l2396_239676

theorem peach_basket_problem (n : ℕ) : 
  n % 4 = 2 →
  n % 6 = 4 →
  (n + 2) % 8 = 0 →
  120 ≤ n →
  n ≤ 150 →
  n = 142 :=
by sorry

end NUMINAMATH_CALUDE_peach_basket_problem_l2396_239676


namespace NUMINAMATH_CALUDE_m_range_l2396_239636

-- Define the quadratic equations
def eq1 (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 = 0
def eq2 (m : ℝ) (x : ℝ) : Prop := 4*x^2 + 4*(m+2)*x + 1 = 0

-- Define the conditions
def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ eq1 m x ∧ eq1 m y

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x, ¬(eq2 m x)

-- State the theorem
theorem m_range (m : ℝ) :
  has_two_distinct_roots m ∧ has_no_real_roots m ↔ -3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2396_239636


namespace NUMINAMATH_CALUDE_train_journey_encryption_train_journey_l2396_239619

/-- Represents a city name as a list of alphabet positions --/
def CityCode := List Nat

/-- Defines the alphabet positions for letters --/
def alphabetPosition (c : Char) : Nat :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'U' => 21
  | 'K' => 11
  | _ => 0

/-- Encodes a city name to a list of alphabet positions --/
def encodeCity (name : String) : CityCode :=
  name.toList.map alphabetPosition

/-- Theorem: The encrypted city names represent Ufa and Baku --/
theorem train_journey_encryption (departure : CityCode) (arrival : CityCode) : 
  (departure = [21, 2, 1, 21] ∧ arrival = [2, 1, 11, 21]) →
  (encodeCity "UFA" = departure ∧ encodeCity "BAKU" = arrival) :=
by
  sorry

/-- Main theorem: The train traveled from Ufa to Baku --/
theorem train_journey : 
  ∃ (departure arrival : CityCode),
    departure = [21, 2, 1, 21] ∧
    arrival = [2, 1, 11, 21] ∧
    encodeCity "UFA" = departure ∧
    encodeCity "BAKU" = arrival :=
by
  sorry

end NUMINAMATH_CALUDE_train_journey_encryption_train_journey_l2396_239619


namespace NUMINAMATH_CALUDE_bus_rental_problem_l2396_239638

/-- Represents the capacity of buses --/
structure BusCapacity where
  typeA : ℕ
  typeB : ℕ

/-- Represents the rental plan --/
structure RentalPlan where
  bus65 : ℕ
  bus45 : ℕ
  bus30 : ℕ

/-- The main theorem to prove --/
theorem bus_rental_problem 
  (capacity : BusCapacity) 
  (plan : RentalPlan) : 
  (3 * capacity.typeA + 2 * capacity.typeB = 195) →
  (2 * capacity.typeA + 4 * capacity.typeB = 210) →
  (capacity.typeA = 45) →
  (capacity.typeB = 30) →
  (plan.bus65 = 2) →
  (plan.bus45 = 2) →
  (plan.bus30 = 3) →
  (65 * plan.bus65 + 45 * plan.bus45 + 30 * plan.bus30 = 303 + 7) →
  (plan.bus65 + plan.bus45 + plan.bus30 = 7) →
  True := by
  sorry

end NUMINAMATH_CALUDE_bus_rental_problem_l2396_239638


namespace NUMINAMATH_CALUDE_gcd_18_30_45_l2396_239667

theorem gcd_18_30_45 : Nat.gcd 18 (Nat.gcd 30 45) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_45_l2396_239667


namespace NUMINAMATH_CALUDE_inequality_and_equality_l2396_239607

theorem inequality_and_equality (x : ℝ) (h : x > 0) :
  Real.sqrt (1 / (3 * x + 1)) + Real.sqrt (x / (x + 3)) ≥ 1 ∧
  (Real.sqrt (1 / (3 * x + 1)) + Real.sqrt (x / (x + 3)) = 1 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_l2396_239607


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2396_239626

/-- The equation of a line passing through (3,4) and tangent to x^2 + y^2 = 25 is 3x + 4y - 25 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (x^2 + y^2 = 25) →  -- Circle equation
  ((3:ℝ)^2 + 4^2 = 25) →  -- Point (3,4) lies on the circle
  (∃ k : ℝ, y - 4 = k * (x - 3)) →  -- Line passes through (3,4)
  (∀ p : ℝ × ℝ, p.1^2 + p.2^2 = 25 → (3 * p.1 + 4 * p.2 - 25 = 0 → p = (3, 4))) →  -- Line touches circle at only one point
  (3 * x + 4 * y - 25 = 0) -- Equation of the tangent line
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2396_239626


namespace NUMINAMATH_CALUDE_no_unique_p_for_expected_value_l2396_239668

theorem no_unique_p_for_expected_value :
  ¬ ∃! p₀ : ℝ, 0 < p₀ ∧ p₀ < 1 ∧ 6 * p₀^2 - 5 * p₀^3 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_no_unique_p_for_expected_value_l2396_239668


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l2396_239663

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection_plane_plane : Plane → Plane → Line)

theorem line_parallel_to_intersection_of_parallel_planes 
  (a : Line) (α β : Plane) (b : Line) :
  parallel_line_plane a α →
  parallel_line_plane a β →
  intersection_plane_plane α β = b →
  parallel_line_line a b := by
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_of_parallel_planes_l2396_239663


namespace NUMINAMATH_CALUDE_frisbee_tournament_committees_l2396_239654

theorem frisbee_tournament_committees :
  let total_teams : ℕ := 4
  let members_per_team : ℕ := 8
  let host_committee_members : ℕ := 4
  let non_host_committee_members : ℕ := 2
  let total_committee_members : ℕ := 10

  (total_teams * (Nat.choose members_per_team host_committee_members) *
   (Nat.choose members_per_team non_host_committee_members) ^ (total_teams - 1)) = 6593280 :=
by sorry

end NUMINAMATH_CALUDE_frisbee_tournament_committees_l2396_239654


namespace NUMINAMATH_CALUDE_number_of_boats_l2396_239648

/-- Given a lake with boats, where each boat has 3 people and there are 15 people on boats,
    prove that the number of boats is 5. -/
theorem number_of_boats (people_per_boat : ℕ) (total_people : ℕ) (num_boats : ℕ) :
  people_per_boat = 3 →
  total_people = 15 →
  num_boats * people_per_boat = total_people →
  num_boats = 5 := by
sorry

end NUMINAMATH_CALUDE_number_of_boats_l2396_239648


namespace NUMINAMATH_CALUDE_percentage_apartments_with_two_residents_l2396_239615

theorem percentage_apartments_with_two_residents
  (total_apartments : ℕ)
  (percentage_with_at_least_one : ℚ)
  (apartments_with_one : ℕ)
  (h1 : total_apartments = 120)
  (h2 : percentage_with_at_least_one = 85 / 100)
  (h3 : apartments_with_one = 30) :
  (((percentage_with_at_least_one * total_apartments) - apartments_with_one) / total_apartments) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_apartments_with_two_residents_l2396_239615


namespace NUMINAMATH_CALUDE_flour_calculation_l2396_239657

/-- The amount of flour originally called for in the recipe -/
def original_flour : ℝ := 7

/-- The extra amount of flour Mary added -/
def extra_flour : ℝ := 2

/-- The total amount of flour Mary used -/
def total_flour : ℝ := 9

/-- Theorem stating that the original amount of flour plus the extra amount equals the total amount -/
theorem flour_calculation : original_flour + extra_flour = total_flour := by
  sorry

end NUMINAMATH_CALUDE_flour_calculation_l2396_239657


namespace NUMINAMATH_CALUDE_job_selection_probability_l2396_239632

theorem job_selection_probability (jamie_prob tom_prob : ℚ) 
  (h1 : jamie_prob = 2 / 3)
  (h2 : tom_prob = 5 / 7) :
  jamie_prob * tom_prob = 10 / 21 := by
sorry

end NUMINAMATH_CALUDE_job_selection_probability_l2396_239632


namespace NUMINAMATH_CALUDE_tangent_line_slope_positive_l2396_239644

/-- Given a function f: ℝ → ℝ, if the tangent line to the curve y = f(x) at the point (2, f(2)) 
    passes through the point (-1, 2), then f'(2) > 0. -/
theorem tangent_line_slope_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : (deriv f 2) * (2 - (-1)) = f 2 - 2) : 
  deriv f 2 > 0 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_slope_positive_l2396_239644


namespace NUMINAMATH_CALUDE_square_side_length_equals_rectangle_root_area_l2396_239637

theorem square_side_length_equals_rectangle_root_area 
  (rectangle_length : ℝ) 
  (rectangle_breadth : ℝ) 
  (square_side : ℝ) 
  (h1 : rectangle_length = 250) 
  (h2 : rectangle_breadth = 160) 
  (h3 : square_side * square_side = rectangle_length * rectangle_breadth) : 
  square_side = 200 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_equals_rectangle_root_area_l2396_239637


namespace NUMINAMATH_CALUDE_complex_equality_l2396_239645

theorem complex_equality (z : ℂ) : z = -1.5 - (1/6)*I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
  Complex.abs (z - 2) = Complex.abs (z - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l2396_239645


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2396_239639

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (λ i => (Nat.choose n i) * (k ^ (n - i)))

theorem balls_in_boxes : distribute_balls 6 2 = 665 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2396_239639


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l2396_239612

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l2396_239612


namespace NUMINAMATH_CALUDE_bomb_defusal_probability_l2396_239647

theorem bomb_defusal_probability :
  let n : ℕ := 4  -- Total number of wires
  let k : ℕ := 2  -- Number of wires that need to be cut
  let total_combinations : ℕ := n.choose k  -- Total number of possible combinations
  let successful_combinations : ℕ := 1  -- Number of successful combinations
  (successful_combinations : ℚ) / total_combinations = 1 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_bomb_defusal_probability_l2396_239647


namespace NUMINAMATH_CALUDE_sum_equality_existence_l2396_239699

theorem sum_equality_existence (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_n : n > 3)
  (h_pos : ∀ i, a i > 0)
  (h_strict : ∀ i j, i < j → a i < a j)
  (h_upper : a (Fin.last n) ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n),
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
    k ≠ l ∧ k ≠ m ∧
    l ≠ m ∧
    a i.succ + a j.succ = a k.succ + a l.succ ∧
    a i.succ + a j.succ = a m.succ :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_existence_l2396_239699


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l2396_239611

theorem exterior_angle_regular_octagon :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    n = 8 →
    interior_angle = (180 * (n - 2 : ℝ)) / n →
    exterior_angle = 180 - interior_angle →
    exterior_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l2396_239611


namespace NUMINAMATH_CALUDE_sum_congruence_mod_9_l2396_239625

theorem sum_congruence_mod_9 : 
  (1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_9_l2396_239625


namespace NUMINAMATH_CALUDE_temperature_conversion_l2396_239689

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 105 → k = 221 := by sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2396_239689


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2396_239690

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 179)
  (sum_of_numbers : a + b + c = 21) :
  a*b + b*c + a*c = 131 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2396_239690


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l2396_239635

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits. -/
def binary (bits : List Bool) : ℕ := binaryToNat bits

theorem binary_addition_subtraction :
  let a := binary [true, true, true, false, true]  -- 11101₂
  let b := binary [true, true, false, true]        -- 1101₂
  let c := binary [true, false, true, true, false] -- 10110₂
  let d := binary [true, false, true, true]        -- 1011₂
  let result := binary [true, true, false, true, true] -- 11011₂
  a + b - c + d = result := by sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l2396_239635


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_31_l2396_239650

theorem smallest_m_divisible_by_31 :
  ∃ (m : ℕ), m = 30 ∧
  (∀ (n : ℕ), n > 0 → 31 ∣ (m + 2^(5*n))) ∧
  (∀ (k : ℕ), k < m → ∃ (n : ℕ), n > 0 ∧ ¬(31 ∣ (k + 2^(5*n)))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_31_l2396_239650


namespace NUMINAMATH_CALUDE_range_of_m_l2396_239610

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m}

-- State the theorem
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 := by
  sorry

-- Note: The actual proof is omitted as per your instructions

end NUMINAMATH_CALUDE_range_of_m_l2396_239610


namespace NUMINAMATH_CALUDE_vector_calculation_l2396_239622

def vector_subtraction (v w : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => v i - w i

def scalar_mult (a : ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ := fun i => a * v i

theorem vector_calculation :
  let v : Fin 2 → ℝ := ![5, -3]
  let w : Fin 2 → ℝ := ![3, -4]
  vector_subtraction v (scalar_mult (-2) w) = ![11, -11] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l2396_239622


namespace NUMINAMATH_CALUDE_point_P_quadrants_l2396_239652

def is_root (x : ℝ) : Prop := (2 * x - 1) * (x + 1) = 0

def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_P_quadrants :
  ∃ (x y : ℝ), (is_root x ∧ is_root y) →
    (in_second_quadrant x y ∨ in_fourth_quadrant x y) ∧
    ¬(in_second_quadrant x y ∧ in_fourth_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_point_P_quadrants_l2396_239652


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_discriminant_of_specific_quadratic_l2396_239680

theorem discriminant_of_quadratic (a b c : ℝ) : 
  (a ≠ 0) → (b^2 - 4*a*c = (b^2 - 4*a*c)) := by sorry

theorem discriminant_of_specific_quadratic : 
  let a : ℝ := 4
  let b : ℝ := -9
  let c : ℝ := -15
  b^2 - 4*a*c = 321 := by sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_discriminant_of_specific_quadratic_l2396_239680


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2396_239600

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 24 x = Nat.choose 24 (3*x - 8)) → (x = 4 ∨ x = 8) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2396_239600


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_90_and_m_l2396_239605

theorem greatest_common_divisor_of_90_and_m (m : ℕ) 
  (h1 : ∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
    (∀ (d : ℕ), d ∣ 90 ∧ d ∣ m ↔ d = d1 ∨ d = d2 ∨ d = d3)) :
  ∃ (d : ℕ), d ∣ 90 ∧ d ∣ m ∧ d = 9 ∧ 
    ∀ (x : ℕ), x ∣ 90 ∧ x ∣ m → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_90_and_m_l2396_239605


namespace NUMINAMATH_CALUDE_quadratic_order_l2396_239617

theorem quadratic_order (m y₁ y₂ y₃ : ℝ) (hm : m < -2) 
  (h₁ : y₁ = (m - 1)^2 + 2*(m - 1))
  (h₂ : y₂ = m^2 + 2*m)
  (h₃ : y₃ = (m + 1)^2 + 2*(m + 1)) :
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_quadratic_order_l2396_239617


namespace NUMINAMATH_CALUDE_recreation_area_tents_l2396_239672

/-- Represents the number of tents in different parts of the campsite -/
structure CampsiteTents where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : CampsiteTents) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents :
  ∃ (c : CampsiteTents),
    c.north = 100 ∧
    c.east = 2 * c.north ∧
    c.center = 4 * c.north ∧
    c.south = 200 ∧
    total_tents c = 900 := by
  sorry

end NUMINAMATH_CALUDE_recreation_area_tents_l2396_239672


namespace NUMINAMATH_CALUDE_inverse_inequality_l2396_239671

theorem inverse_inequality (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l2396_239671


namespace NUMINAMATH_CALUDE_max_perfect_squares_l2396_239662

theorem max_perfect_squares (a b : ℕ) (h : a ≠ b) : 
  let products := [a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)]
  (∃ (n : ℕ), n * n ∈ products) ∧ 
  ¬(∃ (m n : ℕ) (hm : m * m ∈ products) (hn : n * n ∈ products), m ≠ n) :=
by sorry

end NUMINAMATH_CALUDE_max_perfect_squares_l2396_239662


namespace NUMINAMATH_CALUDE_square_times_abs_fraction_equals_three_l2396_239673

theorem square_times_abs_fraction_equals_three :
  (-3)^2 * |-(1/3)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_times_abs_fraction_equals_three_l2396_239673


namespace NUMINAMATH_CALUDE_plot_length_is_65_meters_l2396_239646

/-- Represents a rectangular plot with its dimensions and fencing cost -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- The length is 30 meters more than the breadth -/
def lengthCondition (plot : RectangularPlot) : Prop :=
  plot.length = plot.breadth + 30

/-- The cost of fencing at 26.50 per meter is Rs. 5300 -/
def fencingCostCondition (plot : RectangularPlot) : Prop :=
  plot.fencingCostPerMeter = 26.50 ∧ plot.totalFencingCost = 5300

/-- The perimeter of the plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Theorem stating that under given conditions, the length of the plot is 65 meters -/
theorem plot_length_is_65_meters (plot : RectangularPlot) 
  (h1 : lengthCondition plot) 
  (h2 : fencingCostCondition plot) : 
  plot.length = 65 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_65_meters_l2396_239646
