import Mathlib

namespace NUMINAMATH_CALUDE_age_difference_l265_26589

/-- Given information about Jacob and Michael's ages, prove their current age difference -/
theorem age_difference (jacob_current : ℕ) (michael_current : ℕ) : 
  (jacob_current + 4 = 13) → 
  (michael_current + 3 = 2 * (jacob_current + 3)) →
  (michael_current - jacob_current = 12) := by
sorry

end NUMINAMATH_CALUDE_age_difference_l265_26589


namespace NUMINAMATH_CALUDE_fraction_inequality_l265_26552

theorem fraction_inequality (a b : ℝ) : a < b → b < 0 → (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l265_26552


namespace NUMINAMATH_CALUDE_range_of_a_l265_26531

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 5 else a^x + 2*a + 2

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, y ≥ 3 → ∃ x : ℝ, f a x = y) ∧ 
  (∀ x : ℝ, f a x ≥ 3) →
  a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l265_26531


namespace NUMINAMATH_CALUDE_chemical_mixture_theorem_l265_26526

/-- Represents a chemical solution with percentages of two components -/
structure Solution :=
  (percent_a : ℝ)
  (percent_b : ℝ)
  (sum_to_one : percent_a + percent_b = 1)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution_x : Solution)
  (solution_y : Solution)
  (percent_x : ℝ)
  (percent_y : ℝ)
  (sum_to_one : percent_x + percent_y = 1)

/-- Calculates the percentage of chemical a in a mixture -/
def percent_a_in_mixture (m : Mixture) : ℝ :=
  m.percent_x * m.solution_x.percent_a + m.percent_y * m.solution_y.percent_a

theorem chemical_mixture_theorem (x y : Solution) 
  (hx : x.percent_a = 0.4) 
  (hy : y.percent_a = 0.5) : 
  let m : Mixture := {
    solution_x := x,
    solution_y := y,
    percent_x := 0.3,
    percent_y := 0.7,
    sum_to_one := by norm_num
  }
  percent_a_in_mixture m = 0.47 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_theorem_l265_26526


namespace NUMINAMATH_CALUDE_cyclists_distance_l265_26588

/-- Calculates the distance between two cyclists traveling in opposite directions -/
def distance_between_cyclists (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem stating the distance between two cyclists after 2 hours -/
theorem cyclists_distance :
  let speed1 : ℝ := 10  -- Speed of first cyclist in km/h
  let speed2 : ℝ := 15  -- Speed of second cyclist in km/h
  let time : ℝ := 2     -- Time in hours
  distance_between_cyclists speed1 speed2 time = 50 := by
  sorry

#check cyclists_distance

end NUMINAMATH_CALUDE_cyclists_distance_l265_26588


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l265_26595

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l265_26595


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l265_26596

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ (15 * n - 1) % 11 = 0 ∧ ∀ (m : ℕ), m > 0 → (15 * m - 1) % 11 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l265_26596


namespace NUMINAMATH_CALUDE_first_day_is_thursday_l265_26514

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  saturdays : Nat
  sundays : Nat

/-- Function to determine the first day of the month -/
def firstDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Theorem stating that in a month with 31 days, 5 Saturdays, and 4 Sundays, 
    the first day is Thursday -/
theorem first_day_is_thursday :
  ∀ (m : Month), m.days = 31 → m.saturdays = 5 → m.sundays = 4 →
  firstDayOfMonth m = DayOfWeek.Thursday :=
  sorry

end NUMINAMATH_CALUDE_first_day_is_thursday_l265_26514


namespace NUMINAMATH_CALUDE_rooks_arrangement_count_l265_26553

/-- The number of squares on a chessboard -/
def chessboardSquares : ℕ := 64

/-- The number of squares threatened by a rook (excluding its own square) -/
def squaresThreatened : ℕ := 14

/-- The number of ways to arrange two rooks on a chessboard such that they cannot capture each other -/
def rooksArrangements : ℕ := chessboardSquares * (chessboardSquares - squaresThreatened - 1)

theorem rooks_arrangement_count :
  rooksArrangements = 3136 := by sorry

end NUMINAMATH_CALUDE_rooks_arrangement_count_l265_26553


namespace NUMINAMATH_CALUDE_odd_prime_gcd_sum_and_fraction_l265_26528

theorem odd_prime_gcd_sum_and_fraction (p a b : ℕ) : 
  Nat.Prime p → p % 2 = 1 → Nat.Coprime a b → 
  Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := by
sorry

end NUMINAMATH_CALUDE_odd_prime_gcd_sum_and_fraction_l265_26528


namespace NUMINAMATH_CALUDE_function_inequality_l265_26593

/-- For any differentiable function f on ℝ, if (x + 1)f'(x) ≥ 0 for all x in ℝ, 
    then f(0) + f(-2) ≥ 2f(-1) -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, (x + 1) * deriv f x ≥ 0) : 
  f 0 + f (-2) ≥ 2 * f (-1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l265_26593


namespace NUMINAMATH_CALUDE_greatest_multiple_3_4_under_500_l265_26584

theorem greatest_multiple_3_4_under_500 : ∃ n : ℕ, n = 492 ∧ 
  (∀ m : ℕ, m < 500 ∧ 3 ∣ m ∧ 4 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_3_4_under_500_l265_26584


namespace NUMINAMATH_CALUDE_solve_system_l265_26576

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 1) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l265_26576


namespace NUMINAMATH_CALUDE_first_group_size_correct_l265_26542

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 78

/-- The number of days the first group takes to repair the road -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group takes to repair the road -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l265_26542


namespace NUMINAMATH_CALUDE_chick_hits_l265_26525

theorem chick_hits (chick monkey dog : ℕ) : 
  chick * 9 + monkey * 5 + dog * 2 = 61 →
  chick + monkey + dog = 10 →
  chick ≥ 1 →
  monkey ≥ 1 →
  dog ≥ 1 →
  chick = 5 :=
by sorry

end NUMINAMATH_CALUDE_chick_hits_l265_26525


namespace NUMINAMATH_CALUDE_fibonacci_sum_equals_two_l265_26507

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sum of the series
noncomputable def fibonacciSum : ℝ := ∑' n, (fib n : ℝ) / 2^n

-- Theorem statement
theorem fibonacci_sum_equals_two : fibonacciSum = 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_equals_two_l265_26507


namespace NUMINAMATH_CALUDE_probability_at_least_two_liking_chi_square_association_l265_26558

-- Define the total number of students and their preferences
def total_students : ℕ := 200
def students_liking : ℕ := 140
def students_disliking : ℕ := 60

-- Define the gender-based preferences
def male_liking : ℕ := 60
def male_disliking : ℕ := 40
def female_liking : ℕ := 80
def female_disliking : ℕ := 20

-- Define the significance level
def alpha : ℝ := 0.005

-- Define the critical value for α = 0.005
def critical_value : ℝ := 7.879

-- Theorem 1: Probability of selecting at least 2 students who like employment
theorem probability_at_least_two_liking :
  (Nat.choose 3 2 * (students_liking / total_students)^2 * (students_disliking / total_students) +
   (students_liking / total_students)^3) = 98 / 125 := by sorry

-- Theorem 2: Chi-square test for association between intention and gender
theorem chi_square_association :
  let n := total_students
  let a := male_liking
  let b := male_disliking
  let c := female_liking
  let d := female_disliking
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) > critical_value := by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_liking_chi_square_association_l265_26558


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l265_26567

theorem least_n_with_gcd_conditions : 
  ∃ (n : ℕ), n > 1000 ∧ 
  Nat.gcd 30 (n + 80) = 15 ∧ 
  Nat.gcd (n + 30) 100 = 50 ∧
  (∀ m : ℕ, m > 1000 → 
    (Nat.gcd 30 (m + 80) = 15 ∧ Nat.gcd (m + 30) 100 = 50) → 
    m ≥ n) ∧
  n = 1270 :=
sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l265_26567


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l265_26536

theorem unique_solution_to_equation : ∃! x : ℝ, 
  (x ≠ 5 ∧ x ≠ 3) ∧ 
  (x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3) = 
  (x - 5) * (x - 3) * (x - 5) ∧ 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l265_26536


namespace NUMINAMATH_CALUDE_dilation_problem_l265_26551

/-- Dilation of a complex number -/
def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

/-- The problem statement -/
theorem dilation_problem : dilation (-1 + 2*I) 4 (3 + 4*I) = 15 + 10*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_problem_l265_26551


namespace NUMINAMATH_CALUDE_functional_equation_solution_l265_26580

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (f x + y + 1) = x + f y + 1) →
  ((∀ n : ℤ, f n = n) ∨ (∀ n : ℤ, f n = -n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l265_26580


namespace NUMINAMATH_CALUDE_pen_count_difference_l265_26549

theorem pen_count_difference (red : ℕ) (black : ℕ) (blue : ℕ) : 
  red = 8 →
  black = red + 10 →
  red + black + blue = 41 →
  blue > red →
  blue - red = 7 := by
sorry

end NUMINAMATH_CALUDE_pen_count_difference_l265_26549


namespace NUMINAMATH_CALUDE_inequality_solution_l265_26554

theorem inequality_solution (p x y : ℝ) : 
  p = x + y → x^2 + 4*y^2 + 8*y + 4 ≤ 4*x → -3 - Real.sqrt 5 ≤ p ∧ p ≤ -3 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l265_26554


namespace NUMINAMATH_CALUDE_combinatorial_identity_l265_26532

theorem combinatorial_identity 
  (n k m : ℕ) 
  (h1 : 1 ≤ k) 
  (h2 : k < m) 
  (h3 : m ≤ n) : 
  (Finset.sum (Finset.range (k + 1)) (λ i => Nat.choose k i * Nat.choose n (m - i))) = 
  Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l265_26532


namespace NUMINAMATH_CALUDE_total_course_hours_l265_26572

/-- Represents the total hours spent on a course over the duration of 24 weeks --/
structure CourseHours where
  weekly : ℕ
  additional : ℕ

/-- Calculates the total hours for a course over 24 weeks --/
def totalHours (c : CourseHours) : ℕ := c.weekly * 24 + c.additional

/-- Data analytics course structure --/
def dataAnalyticsCourse : CourseHours :=
  { weekly := 14,  -- 10 hours class + 4 hours homework
    additional := 90 }  -- 48 hours lab sessions + 42 hours projects

/-- Programming course structure --/
def programmingCourse : CourseHours :=
  { weekly := 18,  -- 4 hours class + 8 hours lab + 6 hours assignments
    additional := 0 }

/-- Statistics course structure --/
def statisticsCourse : CourseHours :=
  { weekly := 11,  -- 6 hours class + 2 hours lab + 3 hours group projects
    additional := 45 }  -- 5 hours/week for 9 weeks for exam study

/-- The main theorem stating the total hours spent on all courses --/
theorem total_course_hours :
  totalHours dataAnalyticsCourse +
  totalHours programmingCourse +
  totalHours statisticsCourse = 1167 := by
  sorry

end NUMINAMATH_CALUDE_total_course_hours_l265_26572


namespace NUMINAMATH_CALUDE_integer_sequence_count_l265_26556

def sequence_term (n : ℕ) : ℚ :=
  16200 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem integer_sequence_count : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) ∧
  (∃! (n : ℕ), n > 0 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) ∧
  (∃ (n : ℕ), n = 3 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) :=
by sorry

end NUMINAMATH_CALUDE_integer_sequence_count_l265_26556


namespace NUMINAMATH_CALUDE_max_value_trig_product_l265_26500

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) *
  (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_product_l265_26500


namespace NUMINAMATH_CALUDE_largest_sum_l265_26530

theorem largest_sum : 
  let expr1 := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
  let expr2 := (1/4 : ℚ) - (1/6 : ℚ)
  let expr3 := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
  let expr4 := (1/4 : ℚ) - (1/8 : ℚ)
  let expr5 := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)
  expr3 = (5/12 : ℚ) ∧ 
  expr3 > expr1 ∧ 
  expr3 > expr2 ∧ 
  expr3 > expr4 ∧ 
  expr3 > expr5 :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_l265_26530


namespace NUMINAMATH_CALUDE_propositions_true_l265_26538

theorem propositions_true :
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → (a - c) / c > (b - c) / b) ∧
  (∀ a b : ℝ, a > |b| → a^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_propositions_true_l265_26538


namespace NUMINAMATH_CALUDE_anika_age_l265_26522

/-- Given the ages of Ben, Clara, and Anika, prove that Anika is 15 years old. -/
theorem anika_age (ben_age clara_age anika_age : ℕ) 
  (h1 : clara_age = ben_age + 5)
  (h2 : anika_age = clara_age - 10)
  (h3 : ben_age = 20) : 
  anika_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_anika_age_l265_26522


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l265_26516

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 2

-- Define point A
def point_A : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem intersection_distance_sum :
  ∃ (P Q : ℝ × ℝ),
    line_l P.1 P.2 ∧ circle_C P.1 P.2 ∧
    line_l Q.1 Q.2 ∧ circle_C Q.1 Q.2 ∧
    P ≠ Q ∧
    Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) +
    Real.sqrt ((Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2) =
    Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l265_26516


namespace NUMINAMATH_CALUDE_solution_sum_l265_26591

theorem solution_sum (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l265_26591


namespace NUMINAMATH_CALUDE_lowest_power_x4_l265_26503

theorem lowest_power_x4 (x : ℝ) : 
  let A : ℝ := 1/3
  let B : ℝ := -1/9
  let C : ℝ := 5/81
  let f : ℝ → ℝ := λ x => (1 + A*x + B*x^2 + C*x^3)^3 - (1 + x)
  ∃ (D E F G H I : ℝ), f x = D*x^4 + E*x^5 + F*x^6 + G*x^7 + H*x^8 + I*x^9 ∧ D ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_lowest_power_x4_l265_26503


namespace NUMINAMATH_CALUDE_opposite_of_2023_l265_26512

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l265_26512


namespace NUMINAMATH_CALUDE_oscar_age_is_6_l265_26548

/-- Christina's age in 5 years -/
def christina_age_in_5_years : ℕ := 80 / 2

/-- Christina's current age -/
def christina_current_age : ℕ := christina_age_in_5_years - 5

/-- Oscar's age in 15 years -/
def oscar_age_in_15_years : ℕ := (3 * christina_current_age) / 5

/-- Oscar's current age -/
def oscar_current_age : ℕ := oscar_age_in_15_years - 15

theorem oscar_age_is_6 : oscar_current_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_oscar_age_is_6_l265_26548


namespace NUMINAMATH_CALUDE_percentage_problem_l265_26562

theorem percentage_problem :
  ∃ x : ℝ, 0.0425 * x = 2.125 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l265_26562


namespace NUMINAMATH_CALUDE_intersection_of_lines_l265_26544

/-- Given two lines m and n that intersect at (2, 7), 
    where m has equation y = 2x + 3 and n has equation y = kx + 1,
    prove that k = 3. -/
theorem intersection_of_lines (k : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 3 → y = k*x + 1 → x = 2 ∧ y = 7) → 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l265_26544


namespace NUMINAMATH_CALUDE_sum_product_quadratic_l265_26537

theorem sum_product_quadratic (S P x y : ℝ) :
  x + y = S ∧ x * y = P →
  ∃ t : ℝ, t ^ 2 - S * t + P = 0 ∧ (t = x ∨ t = y) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_quadratic_l265_26537


namespace NUMINAMATH_CALUDE_task_completion_ways_l265_26566

theorem task_completion_ways (m₁ m₂ : ℕ) : ∃ N : ℕ, N = m₁ + m₂ := by
  sorry

end NUMINAMATH_CALUDE_task_completion_ways_l265_26566


namespace NUMINAMATH_CALUDE_exist_six_games_twelve_players_l265_26564

structure Tournament where
  players : Finset ℕ
  games : Finset (ℕ × ℕ)
  player_in_game : ∀ p ∈ players, ∃ g ∈ games, p ∈ g.1 :: g.2 :: []

theorem exist_six_games_twelve_players (t : Tournament) 
  (h1 : t.players.card = 20)
  (h2 : t.games.card = 14) :
  ∃ (subset_games : Finset (ℕ × ℕ)) (subset_players : Finset ℕ),
    subset_games ⊆ t.games ∧
    subset_games.card = 6 ∧
    subset_players ⊆ t.players ∧
    subset_players.card = 12 ∧
    ∀ g ∈ subset_games, g.1 ∈ subset_players ∧ g.2 ∈ subset_players :=
sorry

end NUMINAMATH_CALUDE_exist_six_games_twelve_players_l265_26564


namespace NUMINAMATH_CALUDE_books_in_boxes_l265_26502

theorem books_in_boxes (total_books : ℕ) (books_per_box : ℕ) (num_boxes : ℕ) : 
  total_books = 24 → books_per_box = 3 → num_boxes * books_per_box = total_books → num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_books_in_boxes_l265_26502


namespace NUMINAMATH_CALUDE_unique_solution_at_85_l265_26547

/-- Represents the American High School Mathematics Examination (AHSME) -/
structure AHSME where
  total_questions : Nat
  score_formula : (correct : Nat) → (wrong : Nat) → Int

/-- Defines the specific AHSME instance -/
def ahsme : AHSME :=
  { total_questions := 30
  , score_formula := λ c w => 30 + 4 * c - w }

/-- Theorem stating the uniqueness of the solution for a score of 85 -/
theorem unique_solution_at_85 (exam : AHSME := ahsme) :
  ∃! (c w : Nat), c + w ≤ exam.total_questions ∧
                  exam.score_formula c w = 85 ∧
                  (∀ s, 85 > s → s > 85 - 4 →
                    ¬∃! (c' w' : Nat), c' + w' ≤ exam.total_questions ∧
                                      exam.score_formula c' w' = s) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_at_85_l265_26547


namespace NUMINAMATH_CALUDE_ratio_problem_l265_26571

theorem ratio_problem (a b c d : ℚ) 
  (h1 : b / a = 3)
  (h2 : c / b = 4)
  (h3 : d = 2 * b - a) :
  (a + b + d) / (b + c + d) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l265_26571


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l265_26539

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 4 ∧ b = 8 ∧ c = 8 → -- Two sides are 8, one side is 4
  (a + b > c ∧ b + c > a ∧ c + a > b) → -- Triangle inequality
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l265_26539


namespace NUMINAMATH_CALUDE_widest_strip_width_l265_26594

theorem widest_strip_width (bolt_width_1 bolt_width_2 : ℕ) 
  (h1 : bolt_width_1 = 45) 
  (h2 : bolt_width_2 = 60) : 
  Nat.gcd bolt_width_1 bolt_width_2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_widest_strip_width_l265_26594


namespace NUMINAMATH_CALUDE_helen_cookies_baked_this_morning_l265_26585

theorem helen_cookies_baked_this_morning (total_cookies : ℕ) (yesterday_cookies : ℕ) 
  (h1 : total_cookies = 574)
  (h2 : yesterday_cookies = 435) :
  total_cookies - yesterday_cookies = 139 := by
sorry

end NUMINAMATH_CALUDE_helen_cookies_baked_this_morning_l265_26585


namespace NUMINAMATH_CALUDE_teacher_age_l265_26518

theorem teacher_age (n : ℕ) (initial_avg : ℚ) (student_age : ℕ) (final_avg : ℚ) 
  (h1 : n = 30)
  (h2 : initial_avg = 10)
  (h3 : student_age = 11)
  (h4 : final_avg = 11) :
  (n : ℚ) * initial_avg - student_age + (n - 1 : ℚ) * final_avg - ((n - 1 : ℚ) * initial_avg - student_age) = 30 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l265_26518


namespace NUMINAMATH_CALUDE_correct_average_l265_26543

-- Define the number of elements in the set
def n : ℕ := 20

-- Define the initial incorrect average
def incorrect_avg : ℚ := 25.6

-- Define the three pairs of incorrect and correct numbers
def num1 : (ℚ × ℚ) := (57.5, 78.5)
def num2 : (ℚ × ℚ) := (25.25, 35.25)
def num3 : (ℚ × ℚ) := (24.25, 47.5)

-- Define the correct average
def correct_avg : ℚ := 28.3125

-- Theorem statement
theorem correct_average : 
  let incorrect_sum := n * incorrect_avg
  let diff1 := num1.2 - num1.1
  let diff2 := num2.2 - num2.1
  let diff3 := num3.2 - num3.1
  let correct_sum := incorrect_sum + diff1 + diff2 + diff3
  correct_sum / n = correct_avg := by sorry

end NUMINAMATH_CALUDE_correct_average_l265_26543


namespace NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l265_26501

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(k^2 + k)

-- Define a function to check if three sides form an isosceles triangle
def is_isosceles (a b c : ℝ) : Prop := (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Theorem statement
theorem quadratic_roots_and_isosceles_triangle (k : ℝ) :
  (∀ k, discriminant k > 0) ∧
  (∃ x y, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 ∧ is_isosceles x y 4) ↔
  (k = 3 ∨ k = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_isosceles_triangle_l265_26501


namespace NUMINAMATH_CALUDE_james_total_toys_l265_26579

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

theorem james_total_toys : total_toys = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_total_toys_l265_26579


namespace NUMINAMATH_CALUDE_survey_solution_l265_26515

def survey_problem (mac_preference : ℕ) (no_preference : ℕ) : Prop :=
  let both_preference : ℕ := mac_preference / 3
  let total_students : ℕ := mac_preference + both_preference + no_preference
  (mac_preference = 60) ∧ (no_preference = 90) → (total_students = 170)

theorem survey_solution : survey_problem 60 90 := by
  sorry

end NUMINAMATH_CALUDE_survey_solution_l265_26515


namespace NUMINAMATH_CALUDE_bus_speed_calculation_prove_bus_speed_l265_26508

/-- The speed of buses traveling along a country road -/
def bus_speed : ℝ := 46

/-- The speed of the cyclist -/
def cyclist_speed : ℝ := 16

/-- The number of buses counted approaching from the front -/
def buses_front : ℕ := 31

/-- The number of buses counted from behind -/
def buses_behind : ℕ := 15

theorem bus_speed_calculation :
  bus_speed * (buses_front : ℝ) / (bus_speed + cyclist_speed) = 
  bus_speed * (buses_behind : ℝ) / (bus_speed - cyclist_speed) :=
by sorry

/-- The main theorem proving the speed of the buses -/
theorem prove_bus_speed : 
  ∃ (speed : ℝ), speed > 0 ∧ 
  speed * (buses_front : ℝ) / (speed + cyclist_speed) = 
  speed * (buses_behind : ℝ) / (speed - cyclist_speed) ∧
  speed = bus_speed :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_calculation_prove_bus_speed_l265_26508


namespace NUMINAMATH_CALUDE_birthday_money_calculation_l265_26569

def money_spent : ℕ := 34
def money_left : ℕ := 33

theorem birthday_money_calculation :
  money_spent + money_left = 67 := by sorry

end NUMINAMATH_CALUDE_birthday_money_calculation_l265_26569


namespace NUMINAMATH_CALUDE_triangle_area_implies_angle_l265_26513

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S_ABC = (a^2 + b^2 - c^2) / 4, then the measure of angle C is π/4. -/
theorem triangle_area_implies_angle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a^2 + b^2 - c^2) / 4 = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :
  Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_angle_l265_26513


namespace NUMINAMATH_CALUDE_find_z_l265_26598

theorem find_z (m : ℕ) (z : ℝ) 
  (h1 : ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (z ^ 16)) = 1 / (2 * (10 ^ 31)))
  (h2 : m = 31) : z = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_z_l265_26598


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l265_26550

theorem binomial_coefficient_identity (n k : ℕ+) (h : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) ∧
  k * Nat.choose n k = (n - k + 1) * Nat.choose n (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l265_26550


namespace NUMINAMATH_CALUDE_square_area_percentage_l265_26586

/-- Given a rectangle enclosing a square, this theorem proves the percentage
    of the rectangle's area occupied by the square. -/
theorem square_area_percentage (s : ℝ) (h1 : s > 0) : 
  let w := 3 * s  -- width of rectangle
  let l := 3 * w / 2  -- length of rectangle
  let square_area := s^2
  let rectangle_area := l * w
  (square_area / rectangle_area) * 100 = 200 / 27 := by sorry

end NUMINAMATH_CALUDE_square_area_percentage_l265_26586


namespace NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l265_26560

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) :
  m^4 + n^4 = 97 := by
sorry

end NUMINAMATH_CALUDE_m_fourth_plus_n_fourth_l265_26560


namespace NUMINAMATH_CALUDE_fraction_equality_l265_26592

theorem fraction_equality (x : ℝ) : (2 + x) / (4 + x) = (3 + x) / (7 + x) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l265_26592


namespace NUMINAMATH_CALUDE_units_digit_150_factorial_is_zero_l265_26599

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_150_factorial_is_zero :
  unitsDigit (factorial 150) = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_150_factorial_is_zero_l265_26599


namespace NUMINAMATH_CALUDE_optimal_tax_and_revenue_correct_l265_26511

/-- Market model with linear supply and demand functions -/
structure MarketModel where
  -- Supply function coefficients
  supply_slope : ℝ
  supply_intercept : ℝ
  -- Demand function coefficient (slope)
  demand_slope : ℝ
  -- Elasticity ratio at equilibrium
  elasticity_ratio : ℝ
  -- Tax rate
  tax_rate : ℝ
  -- Consumer price after tax
  consumer_price : ℝ

/-- Calculate the optimal tax rate and maximum tax revenue -/
def optimal_tax_and_revenue (model : MarketModel) : ℝ × ℝ :=
  -- Placeholder for the actual calculation
  (60, 8640)

/-- Theorem stating the optimal tax rate and maximum tax revenue -/
theorem optimal_tax_and_revenue_correct (model : MarketModel) :
  model.supply_slope = 6 ∧
  model.supply_intercept = -312 ∧
  model.demand_slope = -4 ∧
  model.elasticity_ratio = 1.5 ∧
  model.tax_rate = 30 ∧
  model.consumer_price = 118 →
  optimal_tax_and_revenue model = (60, 8640) := by
  sorry

end NUMINAMATH_CALUDE_optimal_tax_and_revenue_correct_l265_26511


namespace NUMINAMATH_CALUDE_a_10_value_l265_26535

def sequence_property (a : ℕ+ → ℤ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p + a q

theorem a_10_value (a : ℕ+ → ℤ) (h1 : sequence_property a) (h2 : a 2 = -6) :
  a 10 = -30 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l265_26535


namespace NUMINAMATH_CALUDE_angle_supplement_in_parallel_lines_l265_26506

-- Define the structure for our parallel lines and transversal system
structure ParallelLinesSystem where
  -- The smallest angle created by the transversal with line m
  smallest_angle : ℝ
  -- The angle between the transversal and line n on the same side
  other_angle : ℝ

-- Define our theorem
theorem angle_supplement_in_parallel_lines 
  (system : ParallelLinesSystem) 
  (h1 : system.smallest_angle = 40)
  (h2 : system.other_angle = 70) :
  180 - system.other_angle = 110 :=
by
  sorry

#check angle_supplement_in_parallel_lines

end NUMINAMATH_CALUDE_angle_supplement_in_parallel_lines_l265_26506


namespace NUMINAMATH_CALUDE_count_valid_words_l265_26570

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of total possible words without restrictions -/
def total_words : ℕ := 
  (alphabet_size ^ 1) + (alphabet_size ^ 2) + (alphabet_size ^ 3) + 
  (alphabet_size ^ 4) + (alphabet_size ^ 5)

/-- The number of words with fewer than two 'A's -/
def words_with_less_than_two_a : ℕ := 
  ((alphabet_size - 1) ^ 2) + (2 * (alphabet_size - 1)) + 
  ((alphabet_size - 1) ^ 3) + (3 * (alphabet_size - 1) ^ 2) + 
  ((alphabet_size - 1) ^ 4) + (4 * (alphabet_size - 1) ^ 3) + 
  ((alphabet_size - 1) ^ 5) + (5 * (alphabet_size - 1) ^ 4)

/-- The number of valid words in the language -/
def valid_words : ℕ := total_words - words_with_less_than_two_a

theorem count_valid_words : 
  valid_words = (25^1 + 25^2 + 25^3 + 25^4 + 25^5) - 
                (24^2 + 2 * 24 + 24^3 + 3 * 24^2 + 24^4 + 4 * 24^3 + 24^5 + 5 * 24^4) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_words_l265_26570


namespace NUMINAMATH_CALUDE_two_green_marbles_probability_l265_26504

/-- The probability of drawing two green marbles without replacement from a jar -/
theorem two_green_marbles_probability
  (red : ℕ) (green : ℕ) (white : ℕ)
  (h_red : red = 4)
  (h_green : green = 5)
  (h_white : white = 12)
  : (green / (red + green + white)) * ((green - 1) / (red + green + white - 1)) = 1 / 21 :=
by sorry

end NUMINAMATH_CALUDE_two_green_marbles_probability_l265_26504


namespace NUMINAMATH_CALUDE_prob_two_red_two_blue_correct_l265_26517

/-- The probability of selecting 2 red and 2 blue marbles from a bag -/
def probability_two_red_two_blue : ℚ :=
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let selected_marbles : ℕ := 4
  616 / 1615

/-- Theorem stating that the probability of selecting 2 red and 2 blue marbles
    from a bag with 12 red and 8 blue marbles, when 4 marbles are selected
    at random without replacement, is equal to 616/1615 -/
theorem prob_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_two_blue_correct_l265_26517


namespace NUMINAMATH_CALUDE_ice_cream_cost_l265_26534

/-- Given the following conditions:
    - 16 chapatis, each costing Rs. 6
    - 5 plates of rice, each costing Rs. 45
    - 7 plates of mixed vegetable, each costing Rs. 70
    - 6 ice-cream cups
    - Total amount paid: Rs. 931
    Prove that the cost of each ice-cream cup is Rs. 20. -/
theorem ice_cream_cost (chapati_count : ℕ) (chapati_cost : ℕ)
                       (rice_count : ℕ) (rice_cost : ℕ)
                       (veg_count : ℕ) (veg_cost : ℕ)
                       (ice_cream_count : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  chapati_cost = 6 →
  rice_count = 5 →
  rice_cost = 45 →
  veg_count = 7 →
  veg_cost = 70 →
  ice_cream_count = 6 →
  total_paid = 931 →
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + veg_count * veg_cost)) / ice_cream_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l265_26534


namespace NUMINAMATH_CALUDE_reflection_of_A_across_y_axis_l265_26577

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-2, 5)

theorem reflection_of_A_across_y_axis :
  reflect_y A = (2, 5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_A_across_y_axis_l265_26577


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l265_26541

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-1, 3)

theorem a_perpendicular_to_a_minus_b : 
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) := by sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l265_26541


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l265_26559

def systematicSample (n : ℕ) (k : ℕ) (start : ℕ) : List ℕ :=
  List.range k |>.map (fun i => start + i * (n / k))

theorem correct_systematic_sample :
  systematicSample 20 4 5 = [5, 10, 15, 20] := by
  sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l265_26559


namespace NUMINAMATH_CALUDE_system_solutions_l265_26563

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x / (x - a) + y / (y - b) = 2 ∧ a * x + b * y = 2 * a * b

-- Theorem statement
theorem system_solutions (a b : ℝ) :
  (∀ x y : ℝ, system x y a b → x = 2 * a * b / (a + b) ∧ y = 2 * a * b / (a + b)) ∨
  (a = b ∧ ∀ x y : ℝ, system x y a b → x + y = 2 * a) ∨
  (a = -b ∧ ¬∃ x y : ℝ, system x y a b) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l265_26563


namespace NUMINAMATH_CALUDE_line_intersection_range_l265_26521

/-- Given a line y = 2x + (3-a) intersecting the x-axis between points (3,0) and (4,0) inclusive, 
    the range of values for a is 9 ≤ a ≤ 11. -/
theorem line_intersection_range (a : ℝ) : 
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 4 ∧ 0 = 2*x + (3-a)) → 
  (9 ≤ a ∧ a ≤ 11) := by
sorry

end NUMINAMATH_CALUDE_line_intersection_range_l265_26521


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l265_26578

theorem polynomial_evaluation :
  let y : ℤ := -2
  y^3 - y^2 + y - 1 = -7 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l265_26578


namespace NUMINAMATH_CALUDE_count_unique_polygonal_chains_l265_26523

/-- The number of unique closed 2n-segment polygonal chains on an n x n grid -/
def uniquePolygonalChains (n : ℕ) : ℕ :=
  (n.factorial * (n - 1).factorial) / 2

/-- Theorem stating the number of unique closed 2n-segment polygonal chains
    that can be drawn on an n x n grid, passing through all horizontal and
    vertical lines exactly once -/
theorem count_unique_polygonal_chains (n : ℕ) (h : n > 0) :
  uniquePolygonalChains n = (n.factorial * (n - 1).factorial) / 2 := by
  sorry

end NUMINAMATH_CALUDE_count_unique_polygonal_chains_l265_26523


namespace NUMINAMATH_CALUDE_building_height_l265_26597

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 50)
  : (flagpole_height / flagpole_shadow) * building_shadow = 20 := by
  sorry

end NUMINAMATH_CALUDE_building_height_l265_26597


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l265_26520

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  a - b - c - d + e = a - (b - (c - (d + e))) →
  e / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l265_26520


namespace NUMINAMATH_CALUDE_flower_garden_area_proof_l265_26505

/-- The area of a circular flower garden -/
def flower_garden_area (radius : ℝ) (pi : ℝ) : ℝ :=
  pi * radius ^ 2

/-- Proof that the area of a circular flower garden with radius 0.6 meters is 1.08 square meters, given that π is assumed to be 3 -/
theorem flower_garden_area_proof :
  let radius : ℝ := 0.6
  let pi : ℝ := 3
  flower_garden_area radius pi = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_flower_garden_area_proof_l265_26505


namespace NUMINAMATH_CALUDE_dinner_bill_split_l265_26527

theorem dinner_bill_split (total_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (tax_percent : ℝ) :
  total_bill = 425 →
  num_people = 15 →
  tip_percent = 0.18 →
  tax_percent = 0.08 →
  (total_bill * (1 + tip_percent + tax_percent)) / num_people = 35.70 := by
  sorry

end NUMINAMATH_CALUDE_dinner_bill_split_l265_26527


namespace NUMINAMATH_CALUDE_base7_25_to_binary_l265_26573

def base7ToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 7 + (n % 10)

def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBinary (m / 2) ((m % 2) :: acc)
  toBinary n []

theorem base7_25_to_binary :
  decimalToBinary (base7ToDecimal 25) = [1, 0, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base7_25_to_binary_l265_26573


namespace NUMINAMATH_CALUDE_reciprocal_sum_geometric_progression_l265_26568

theorem reciprocal_sum_geometric_progression 
  (q : ℝ) (n : ℕ) (S : ℝ) (h1 : q ≠ 1) :
  let a := 3
  let r := q^2
  let original_sum := a * (1 - r^(2*n)) / (1 - r)
  let reciprocal_sum := (1/a) * (1 - (1/r)^(2*n)) / (1 - 1/r)
  S = original_sum →
  reciprocal_sum = 1/S :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_geometric_progression_l265_26568


namespace NUMINAMATH_CALUDE_sum_of_symmetric_roots_l265_26565

/-- A function f: ℝ → ℝ that satisfies f(1-x) = f(1+x) for all real x -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (1 + x)

/-- The theorem stating that if f is symmetric about 1 and has exactly 2009 real roots,
    then the sum of these roots is 2009 -/
theorem sum_of_symmetric_roots
  (f : ℝ → ℝ)
  (h_sym : SymmetricAboutOne f)
  (h_roots : ∃! (s : Finset ℝ), s.card = 2009 ∧ ∀ x ∈ s, f x = 0) :
  ∃ (s : Finset ℝ), s.card = 2009 ∧ (∀ x ∈ s, f x = 0) ∧ (s.sum id = 2009) :=
sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_roots_l265_26565


namespace NUMINAMATH_CALUDE_j_value_at_one_l265_26561

theorem j_value_at_one (p q r : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x^3 + p*x^2 + 2*x + 20 = 0) ∧
    (y^3 + p*y^2 + 2*y + 20 = 0) ∧
    (z^3 + p*z^2 + 2*z + 20 = 0)) →
  (∀ x : ℝ, x^3 + p*x^2 + 2*x + 20 = 0 → x^4 + 2*x^3 + q*x^2 + 150*x + r = 0) →
  1^4 + 2*1^3 + q*1^2 + 150*1 + r = -13755 :=
by sorry

end NUMINAMATH_CALUDE_j_value_at_one_l265_26561


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l265_26509

theorem basketball_team_cutoff (girls : ℕ) (boys : ℕ) (called_back : ℕ) 
  (h1 : girls = 9)
  (h2 : boys = 14)
  (h3 : called_back = 2) :
  girls + boys - called_back = 21 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l265_26509


namespace NUMINAMATH_CALUDE_always_two_real_roots_integer_roots_condition_l265_26510

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 - a*x + (a - 1)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (a : ℝ) :
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation a x = 0 ∧ quadratic_equation a y = 0 :=
sorry

-- Theorem 2: When roots are integers and one is twice the other, a = 3
theorem integer_roots_condition (a : ℝ) :
  (∃ x y : ℤ, x ≠ y ∧ quadratic_equation a (x : ℝ) = 0 ∧ quadratic_equation a (y : ℝ) = 0 ∧ y = 2*x) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_integer_roots_condition_l265_26510


namespace NUMINAMATH_CALUDE_min_value_expression_l265_26590

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l265_26590


namespace NUMINAMATH_CALUDE_pyramid_on_cylinder_radius_l265_26583

/-- A regular square pyramid with all edges equal to 1 -/
structure RegularSquarePyramid where
  edge_length : ℝ
  edge_equal : edge_length = 1

/-- An infinite right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ

/-- Predicate to check if all vertices of the pyramid lie on the lateral surface of the cylinder -/
def vertices_on_cylinder (p : RegularSquarePyramid) (c : RightCircularCylinder) : Prop :=
  sorry

/-- The main theorem stating the possible values of the cylinder's radius -/
theorem pyramid_on_cylinder_radius (p : RegularSquarePyramid) (c : RightCircularCylinder) :
  vertices_on_cylinder p c → (c.radius = 3 / (4 * Real.sqrt 2) ∨ c.radius = 1 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_pyramid_on_cylinder_radius_l265_26583


namespace NUMINAMATH_CALUDE_pizza_dough_liquids_l265_26575

/-- Pizza dough recipe calculation -/
theorem pizza_dough_liquids (milk_ratio : ℚ) (flour_ratio : ℚ) (flour_amount : ℚ) :
  milk_ratio = 75 →
  flour_ratio = 375 →
  flour_amount = 1125 →
  let portions := flour_amount / flour_ratio
  let milk_amount := portions * milk_ratio
  let water_amount := milk_amount / 2
  milk_amount + water_amount = 337.5 := by
  sorry

#check pizza_dough_liquids

end NUMINAMATH_CALUDE_pizza_dough_liquids_l265_26575


namespace NUMINAMATH_CALUDE_system_equation_result_l265_26533

theorem system_equation_result (a b A B C : ℝ) (x : ℝ) 
  (h1 : a * Real.sin x + b * Real.cos x = 0)
  (h2 : A * Real.sin (2 * x) + B * Real.cos (2 * x) = C)
  (h3 : a ≠ 0) :
  2 * a * b * A + (b^2 - a^2) * B + (a^2 + b^2) * C = 0 :=
by sorry

end NUMINAMATH_CALUDE_system_equation_result_l265_26533


namespace NUMINAMATH_CALUDE_complex_collinear_solution_l265_26529

def collinear (a b c : ℂ) : Prop :=
  ∃ t : ℝ, b - a = t • (c - a) ∨ c - a = t • (b - a)

theorem complex_collinear_solution (z : ℂ) :
  collinear 1 Complex.I z ∧ Complex.abs z = 5 →
  z = 4 - 3 * Complex.I ∨ z = -3 + 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_collinear_solution_l265_26529


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l265_26557

/-- Given an equilateral triangle with side length 30 cm and a square with the same perimeter,
    the area of the square is 506.25 cm^2. -/
theorem square_area_equal_perimeter (triangle_side : ℝ) (square_side : ℝ) : 
  triangle_side = 30 →
  3 * triangle_side = 4 * square_side →
  square_side^2 = 506.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l265_26557


namespace NUMINAMATH_CALUDE_nth_equation_holds_l265_26546

theorem nth_equation_holds (n : ℕ) :
  (n : ℚ) / (n + 2) * (1 - 1 / (n + 1)) = n^2 / ((n + 1) * (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l265_26546


namespace NUMINAMATH_CALUDE_equation_solutions_and_first_m_first_m_above_1959_l265_26587

theorem equation_solutions_and_first_m (m n : ℕ+) :
  (8 * m - 7 = n^2) ↔ 
  (∃ s : ℕ, m = 1 + s * (s + 1) / 2 ∧ n = 2 * s + 1) :=
sorry

theorem first_m_above_1959 :
  (∃ m₀ : ℕ+, m₀ > 1959 ∧ 
   (∀ m : ℕ+, m > 1959 ∧ (∃ n : ℕ+, 8 * m - 7 = n^2) → m ≥ m₀) ∧
   m₀ = 2017) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_and_first_m_first_m_above_1959_l265_26587


namespace NUMINAMATH_CALUDE_cone_height_l265_26574

/-- Given a cone with slant height 2√2 cm and lateral surface area 4 cm², its height is 2 cm. -/
theorem cone_height (s : ℝ) (A : ℝ) (h : ℝ) :
  s = 2 * Real.sqrt 2 →
  A = 4 →
  A = π * s * (Real.sqrt (s^2 - h^2)) →
  h = 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l265_26574


namespace NUMINAMATH_CALUDE_fibonacci_property_l265_26555

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_property :
  let a := fibonacci
  (a 0 * a 2 + a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7) -
  (a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 + a 6^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_property_l265_26555


namespace NUMINAMATH_CALUDE_water_remaining_l265_26582

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l265_26582


namespace NUMINAMATH_CALUDE_picklminster_to_quickville_distance_l265_26519

/-- The distance between Picklminster and Quickville satisfies the given conditions -/
theorem picklminster_to_quickville_distance :
  ∃ (d : ℝ) (vA vB vC vD : ℝ),
    d > 0 ∧ vA > 0 ∧ vB > 0 ∧ vC > 0 ∧ vD > 0 ∧
    120 * vC = vA * (d - 120) ∧
    140 * vD = vA * (d - 140) ∧
    126 * vB = vC * (d - 126) ∧
    vB = vD ∧
    d = 210 :=
by
  sorry

#check picklminster_to_quickville_distance

end NUMINAMATH_CALUDE_picklminster_to_quickville_distance_l265_26519


namespace NUMINAMATH_CALUDE_path_count_equals_combination_l265_26581

/-- The width of the grid -/
def grid_width : ℕ := 6

/-- The height of the grid -/
def grid_height : ℕ := 5

/-- The total number of steps required to reach from A to B -/
def total_steps : ℕ := grid_width + grid_height - 2

/-- The number of vertical steps required -/
def vertical_steps : ℕ := grid_height - 1

theorem path_count_equals_combination : 
  (Nat.choose total_steps vertical_steps) = 126 := by sorry

end NUMINAMATH_CALUDE_path_count_equals_combination_l265_26581


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l265_26524

theorem simplify_sqrt_expression :
  (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 - 4 * Real.sqrt (1/2) = 4 * Real.sqrt 3 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l265_26524


namespace NUMINAMATH_CALUDE_smallest_divisible_term_l265_26545

/-- An integer sequence satisfying the given recurrence relation -/
def IntegerSequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)

/-- The property that 2008 divides a_2007 -/
def DivisibilityCondition (a : ℕ → ℤ) : Prop :=
  2008 ∣ a 2007

/-- The main theorem statement -/
theorem smallest_divisible_term
  (a : ℕ → ℤ)
  (h_seq : IntegerSequence a)
  (h_div : DivisibilityCondition a) :
  (∀ n : ℕ, 2 ≤ n ∧ n < 501 → ¬(2008 ∣ a n)) ∧
  (2008 ∣ a 501) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_term_l265_26545


namespace NUMINAMATH_CALUDE_f_properties_l265_26540

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a
  else Real.log x / Real.log a

-- Define monotonicity for a function on ℝ
def Monotonic (g : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → g x ≤ g y ∨ ∀ x y, x ≤ y → g y ≤ g x

-- Theorem statement
theorem f_properties :
  (f 2 (f 2 2) = 0) ∧
  (∀ a : ℝ, Monotonic (f a) ↔ 1/7 ≤ a ∧ a < 1/3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l265_26540
