import Mathlib

namespace NUMINAMATH_CALUDE_regular_square_pyramid_side_edge_l1003_100396

theorem regular_square_pyramid_side_edge 
  (base_edge : ℝ) 
  (volume : ℝ) 
  (h : base_edge = 4 * Real.sqrt 2) 
  (h' : volume = 32) : 
  ∃ (side_edge : ℝ), side_edge = 5 := by
sorry

end NUMINAMATH_CALUDE_regular_square_pyramid_side_edge_l1003_100396


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l1003_100328

def total_knights : ℕ := 30
def chosen_knights : ℕ := 3

def probability_adjacent_knights : ℚ :=
  1 - (27 * 25 * 23) / (3 * total_knights.choose chosen_knights)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 34 / 35 := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l1003_100328


namespace NUMINAMATH_CALUDE_couch_price_after_changes_l1003_100344

theorem couch_price_after_changes (initial_price : ℝ) 
  (h_initial : initial_price = 62500) : 
  let increase_factor := 1.2
  let decrease_factor := 0.8
  let final_factor := (increase_factor ^ 3) * (decrease_factor ^ 3)
  initial_price * final_factor = 55296 := by sorry

end NUMINAMATH_CALUDE_couch_price_after_changes_l1003_100344


namespace NUMINAMATH_CALUDE_flower_color_difference_l1003_100386

/-- Given the following flower counts:
  - Total flowers: 60
  - Yellow and white flowers: 13
  - Red and yellow flowers: 17
  - Red and white flowers: 14
  - Blue and yellow flowers: 16

  Prove that there are 4 more flowers containing red than white. -/
theorem flower_color_difference
  (total : ℕ)
  (yellow_white : ℕ)
  (red_yellow : ℕ)
  (red_white : ℕ)
  (blue_yellow : ℕ)
  (h_total : total = 60)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14)
  (h_blue_yellow : blue_yellow = 16) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end NUMINAMATH_CALUDE_flower_color_difference_l1003_100386


namespace NUMINAMATH_CALUDE_f_monotone_range_l1003_100315

/-- The function f(x) defined as x^2 + a|x-1| -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 1)

/-- The theorem stating the range of 'a' for which f is monotonically increasing on [0, +∞) -/
theorem f_monotone_range (a : ℝ) :
  (∀ x y, 0 ≤ x ∧ x ≤ y → f a x ≤ f a y) ↔ -2 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_range_l1003_100315


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1003_100381

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -9)
  parallel a b → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1003_100381


namespace NUMINAMATH_CALUDE_graph_tangency_l1003_100320

noncomputable def tangentPoints (a : ℝ) : Prop :=
  (a > 0 ∧ a ≠ 1) →
  ∃ x > 0, (a^x = x ∧ (a^x * Real.log a = 1 ∨ a^x * Real.log a = -1))

theorem graph_tangency :
  ∀ a : ℝ, tangentPoints a ↔ (a = Real.exp (1 / Real.exp 1) ∨ a = Real.exp (-1 / Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_graph_tangency_l1003_100320


namespace NUMINAMATH_CALUDE_class_average_l1003_100346

theorem class_average (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_average : ℚ) (group2_average : ℚ) :
  total_students = 40 →
  group1_students = 28 →
  group2_students = 12 →
  group1_average = 68 / 100 →
  group2_average = 77 / 100 →
  let total_score := group1_students * group1_average + group2_students * group2_average
  let class_average := total_score / total_students
  class_average = 707 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l1003_100346


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l1003_100347

theorem units_digit_of_quotient : ∃ n : ℕ, (7^1993 + 5^1993) / 6 = 10 * n + 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l1003_100347


namespace NUMINAMATH_CALUDE_fruit_basket_count_l1003_100355

/-- The number of fruit baskets -/
def num_baskets : ℕ := 4

/-- The number of apples in each of the first three baskets -/
def apples_per_basket : ℕ := 9

/-- The number of oranges in each of the first three baskets -/
def oranges_per_basket : ℕ := 15

/-- The number of bananas in each of the first three baskets -/
def bananas_per_basket : ℕ := 14

/-- The number of fruits reduced in the fourth basket -/
def reduction : ℕ := 2

/-- The total number of fruits in all baskets -/
def total_fruits : ℕ := 146

theorem fruit_basket_count :
  (3 * (apples_per_basket + oranges_per_basket + bananas_per_basket)) +
  ((apples_per_basket - reduction) + (oranges_per_basket - reduction) + (bananas_per_basket - reduction)) = total_fruits :=
by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l1003_100355


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1003_100364

def polynomial (x : ℤ) : ℤ := x^3 + 3*x^2 - 4*x - 13

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = {-13, -1, 1, 13} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1003_100364


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_negative_four_l1003_100331

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2 - 6*a

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

-- Theorem statement
theorem extremum_implies_a_equals_negative_four (a b : ℝ) :
  f' a b 2 = 0 ∧ f a b 2 = 8 → a = -4 :=
by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_negative_four_l1003_100331


namespace NUMINAMATH_CALUDE_binomial_sum_problem_l1003_100356

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem binomial_sum_problem : 
  (binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006) ∧ 
  (binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32) :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_problem_l1003_100356


namespace NUMINAMATH_CALUDE_smallest_number_l1003_100368

-- Define the numbers in their respective bases
def binary_num : ℕ := 63  -- 111111₍₂₎
def base_6_num : ℕ := 66  -- 150₍₆₎
def base_4_num : ℕ := 64  -- 1000₍₄₎
def octal_num : ℕ := 65   -- 101₍₈₎

-- Theorem statement
theorem smallest_number :
  binary_num < base_6_num ∧ 
  binary_num < base_4_num ∧ 
  binary_num < octal_num :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1003_100368


namespace NUMINAMATH_CALUDE_zero_derivative_not_implies_extreme_point_l1003_100332

/-- A function f : ℝ → ℝ is differentiable and its derivative at 0 is 0 -/
def has_zero_derivative_at_zero (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ deriv f 0 = 0

/-- A point x₀ is an extreme value point of f if f(x₀) is either a maximum or minimum value of f -/
def is_extreme_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∀ x, f x ≤ f x₀) ∨ (∀ x, f x₀ ≤ f x)

/-- The statement that if f'(x₀) = 0, then x₀ is an extreme point of f is false -/
theorem zero_derivative_not_implies_extreme_point :
  ¬ (∀ f : ℝ → ℝ, has_zero_derivative_at_zero f → is_extreme_point f 0) :=
sorry

end NUMINAMATH_CALUDE_zero_derivative_not_implies_extreme_point_l1003_100332


namespace NUMINAMATH_CALUDE_max_segments_theorem_l1003_100359

/-- Represents an equilateral triangle divided into smaller equilateral triangles --/
structure DividedTriangle where
  n : ℕ  -- number of parts each side is divided into

/-- The maximum number of segments that can be marked without forming a complete smaller triangle --/
def max_marked_segments (t : DividedTriangle) : ℕ := t.n * (t.n + 1)

/-- Theorem stating the maximum number of segments that can be marked --/
theorem max_segments_theorem (t : DividedTriangle) :
  max_marked_segments t = t.n * (t.n + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_segments_theorem_l1003_100359


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l1003_100380

theorem quadratic_equation_k_value (x₁ x₂ k : ℝ) : 
  x₁^2 - 3*x₁ + k = 0 →
  x₂^2 - 3*x₂ + k = 0 →
  x₁*x₂ + 2*x₁ + 2*x₂ = 1 →
  k = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l1003_100380


namespace NUMINAMATH_CALUDE_read_book_series_l1003_100310

/-- The number of weeks required to read a book series -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let remaining_books := total_books - first_week - second_week
  let additional_weeks := (remaining_books + subsequent_weeks - 1) / subsequent_weeks
  2 + additional_weeks

/-- Theorem: It takes 7 weeks to read the book series under given conditions -/
theorem read_book_series : weeks_to_read 54 6 3 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_read_book_series_l1003_100310


namespace NUMINAMATH_CALUDE_paul_sold_94_books_l1003_100372

/-- Calculates the number of books Paul sold given his initial, purchased, and final book counts. -/
def books_sold (initial : ℕ) (purchased : ℕ) (final : ℕ) : ℕ :=
  initial + purchased - final

theorem paul_sold_94_books : books_sold 2 150 58 = 94 := by
  sorry

end NUMINAMATH_CALUDE_paul_sold_94_books_l1003_100372


namespace NUMINAMATH_CALUDE_total_students_is_59_l1003_100378

/-- Represents a group of students with subgroups taking history and statistics -/
structure StudentGroup where
  total : ℕ
  history : ℕ
  statistics : ℕ
  both : ℕ
  history_only : ℕ
  history_or_statistics : ℕ

/-- The properties of the student group as described in the problem -/
def problem_group : StudentGroup where
  history := 36
  statistics := 32
  history_or_statistics := 59
  history_only := 27
  both := 36 - 27  -- Derived from history - history_only
  total := 59  -- This is what we want to prove

/-- Theorem stating that the total number of students in the group is 59 -/
theorem total_students_is_59 (g : StudentGroup) 
  (h1 : g.history = problem_group.history)
  (h2 : g.statistics = problem_group.statistics)
  (h3 : g.history_or_statistics = problem_group.history_or_statistics)
  (h4 : g.history_only = problem_group.history_only)
  (h5 : g.both = g.history - g.history_only)
  (h6 : g.history_or_statistics = g.history + g.statistics - g.both) :
  g.total = problem_group.total := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_59_l1003_100378


namespace NUMINAMATH_CALUDE_cos_2theta_plus_pi_l1003_100384

theorem cos_2theta_plus_pi (θ : Real) (h : Real.tan θ = 2) : 
  Real.cos (2 * θ + Real.pi) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_plus_pi_l1003_100384


namespace NUMINAMATH_CALUDE_angle_trigonometric_identity_l1003_100335

theorem angle_trigonometric_identity (α : Real) (m n : Real) : 
  -- Conditions
  α ∈ Set.Icc 0 π ∧ 
  m^2 + n^2 = 1 ∧ 
  n / m = -2 →
  -- Conclusion
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_angle_trigonometric_identity_l1003_100335


namespace NUMINAMATH_CALUDE_probability_even_and_prime_on_two_dice_l1003_100399

/-- A die is a finite set of natural numbers from 1 to 6 -/
def Die : Finset ℕ := Finset.range 6

/-- Even numbers on a die -/
def EvenNumbers : Finset ℕ := {2, 4, 6}

/-- Prime numbers on a die -/
def PrimeNumbers : Finset ℕ := {2, 3, 5}

/-- The probability of an event occurring in a finite sample space -/
def probability (event : Finset ℕ) (sampleSpace : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem probability_even_and_prime_on_two_dice : 
  (probability EvenNumbers Die) * (probability PrimeNumbers Die) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_and_prime_on_two_dice_l1003_100399


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1003_100348

/-- Represents the probability of hitting a specific ring -/
structure RingProbability where
  ring : Nat
  probability : Real

/-- Calculates the probability of hitting either the 10-ring or 9-ring -/
def prob_10_or_9 (probs : List RingProbability) : Real :=
  (probs.filter (fun p => p.ring == 10 || p.ring == 9)).map (fun p => p.probability) |>.sum

/-- Calculates the probability of hitting below the 7-ring -/
def prob_below_7 (probs : List RingProbability) : Real :=
  1 - (probs.map (fun p => p.probability) |>.sum)

/-- Theorem stating the probabilities for the given shooting scenario -/
theorem shooting_probabilities (probs : List RingProbability) 
  (h10 : RingProbability.mk 10 0.21 ∈ probs)
  (h9 : RingProbability.mk 9 0.23 ∈ probs)
  (h8 : RingProbability.mk 8 0.25 ∈ probs)
  (h7 : RingProbability.mk 7 0.28 ∈ probs)
  (h_no_other : ∀ p ∈ probs, p.ring ∈ [7, 8, 9, 10]) :
  prob_10_or_9 probs = 0.44 ∧ prob_below_7 probs = 0.03 := by
  sorry


end NUMINAMATH_CALUDE_shooting_probabilities_l1003_100348


namespace NUMINAMATH_CALUDE_line_properties_l1003_100333

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := y + 1 = k * (x - 2)

-- Theorem statement
theorem line_properties :
  -- 1. Countless lines through (2, -1)
  (∃ (S : Set ℝ), Infinite S ∧ ∀ k ∈ S, line_equation k 2 (-1)) ∧
  -- 2. Always passes through a fixed point
  (∃ (x₀ y₀ : ℝ), ∀ k, line_equation k x₀ y₀) ∧
  -- 3. Cannot be perpendicular to x-axis
  (∀ k, line_equation k 0 0 → k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1003_100333


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l1003_100367

def initial_amount : ℚ := 180

def sandwich_fraction : ℚ := 1 / 5
def museum_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

def remaining_amount : ℚ := initial_amount * (1 - sandwich_fraction - museum_fraction - book_fraction)

theorem jennifer_remaining_money : remaining_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l1003_100367


namespace NUMINAMATH_CALUDE_zero_function_inequality_l1003_100360

theorem zero_function_inequality (f : ℝ → ℝ) :
  (∀ (x y : ℝ), x ≠ 0 → f (x^2 + y) ≥ (1/x + 1) * f y) →
  ∀ x, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_function_inequality_l1003_100360


namespace NUMINAMATH_CALUDE_central_symmetry_is_rotation_by_180_l1003_100318

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define rotation by 180 degrees around a point
def rotateBy180 (center : Point2D) (point : Point2D) : Point2D :=
  { x := 2 * center.x - point.x,
    y := 2 * center.y - point.y }

-- Define central symmetry
def centralSymmetry (center : Point2D) (point : Point2D) : Point2D :=
  rotateBy180 center point

-- Theorem statement
theorem central_symmetry_is_rotation_by_180 (O : Point2D) (A : Point2D) :
  centralSymmetry O A = rotateBy180 O A := by sorry

end NUMINAMATH_CALUDE_central_symmetry_is_rotation_by_180_l1003_100318


namespace NUMINAMATH_CALUDE_total_players_is_77_l1003_100398

/-- The number of cricket players -/
def cricket_players : ℕ := 22

/-- The number of hockey players -/
def hockey_players : ℕ := 15

/-- The number of football players -/
def football_players : ℕ := 21

/-- The number of softball players -/
def softball_players : ℕ := 19

/-- Theorem stating that the total number of players is 77 -/
theorem total_players_is_77 : 
  cricket_players + hockey_players + football_players + softball_players = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_players_is_77_l1003_100398


namespace NUMINAMATH_CALUDE_students_in_other_communities_l1003_100388

theorem students_in_other_communities 
  (total_students : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) :
  total_students = 1520 →
  muslim_percent = 41/100 →
  hindu_percent = 32/100 →
  sikh_percent = 12/100 →
  (total_students : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 228 := by
  sorry

end NUMINAMATH_CALUDE_students_in_other_communities_l1003_100388


namespace NUMINAMATH_CALUDE_meetings_count_is_four_l1003_100336

/-- Represents the meeting problem between Michael and the garbage truck --/
structure MeetingProblem where
  michael_speed : ℝ
  pail_distance : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def number_of_meetings (p : MeetingProblem) : ℕ :=
  sorry

/-- The main theorem stating that the number of meetings is 4 --/
theorem meetings_count_is_four :
  ∀ (p : MeetingProblem),
    p.michael_speed = 6 ∧
    p.pail_distance = 150 ∧
    p.truck_speed = 12 ∧
    p.truck_stop_time = 20 →
    number_of_meetings p = 4 :=
  sorry

end NUMINAMATH_CALUDE_meetings_count_is_four_l1003_100336


namespace NUMINAMATH_CALUDE_sheila_tue_thu_hours_l1003_100391

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hoursPerDayMWF : ℕ  -- Hours worked on Monday, Wednesday, Friday
  daysWorkedMWF : ℕ   -- Number of days worked (Monday, Wednesday, Friday)
  weeklyEarnings : ℕ  -- Total earnings per week
  hourlyRate : ℕ      -- Hourly rate of pay

/-- Calculates the total hours worked on Tuesday and Thursday -/
def hoursTueThu (schedule : WorkSchedule) : ℕ :=
  let mwfHours := schedule.hoursPerDayMWF * schedule.daysWorkedMWF
  let mwfEarnings := mwfHours * schedule.hourlyRate
  let tueThuEarnings := schedule.weeklyEarnings - mwfEarnings
  tueThuEarnings / schedule.hourlyRate

/-- Theorem: Given Sheila's work schedule, she works 12 hours on Tuesday and Thursday combined -/
theorem sheila_tue_thu_hours (schedule : WorkSchedule) 
  (h1 : schedule.hoursPerDayMWF = 8)
  (h2 : schedule.daysWorkedMWF = 3)
  (h3 : schedule.weeklyEarnings = 360)
  (h4 : schedule.hourlyRate = 10) :
  hoursTueThu schedule = 12 := by
  sorry

#eval hoursTueThu { hoursPerDayMWF := 8, daysWorkedMWF := 3, weeklyEarnings := 360, hourlyRate := 10 }

end NUMINAMATH_CALUDE_sheila_tue_thu_hours_l1003_100391


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1003_100389

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₂ = -1 and a₃ = 4,
    prove that a₄ + a₅ = 17. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 1 + a 2 = -1)
    (h_third : a 3 = 4) : 
  a 4 + a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1003_100389


namespace NUMINAMATH_CALUDE_expression_values_l1003_100361

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a^2 - b*c = b^2 - a*c) (h2 : b^2 - a*c = c^2 - a*b) :
  (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b) = 7/2) ∨
  (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b) = -7) := by
  sorry

#check expression_values

end NUMINAMATH_CALUDE_expression_values_l1003_100361


namespace NUMINAMATH_CALUDE_equal_focal_distances_l1003_100314

/-- The first curve equation -/
def curve1 (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (16 - k) - p.2^2 / k = 1}

/-- The second curve equation -/
def curve2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 9 * p.1^2 + 25 * p.2^2 = 225}

/-- The focal distance of a curve -/
def focalDistance (curve : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating the necessary and sufficient condition for equal focal distances -/
theorem equal_focal_distances :
  ∀ k : ℝ, (focalDistance (curve1 k) = focalDistance curve2) ↔ (0 < k ∧ k < 16) :=
sorry

end NUMINAMATH_CALUDE_equal_focal_distances_l1003_100314


namespace NUMINAMATH_CALUDE_book_selection_theorem_l1003_100337

theorem book_selection_theorem (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 2) :
  (Nat.choose (n - k) (m - k)) = (Nat.choose 6 3) :=
sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l1003_100337


namespace NUMINAMATH_CALUDE_function_properties_l1003_100387

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + 4) / x

-- State the theorem
theorem function_properties (a : ℝ) :
  f a 1 = 5 →
  (a = 1 ∧
   (∀ x : ℝ, x ≠ 0 → f a (-x) = -(f a x)) ∧
   (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l1003_100387


namespace NUMINAMATH_CALUDE_triangle_abc_acute_angled_l1003_100362

theorem triangle_abc_acute_angled (A B C : ℝ) 
  (h1 : A + B + C = 180) 
  (h2 : A = B) 
  (h3 : A = 2 * C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_acute_angled_l1003_100362


namespace NUMINAMATH_CALUDE_log_equation_solution_l1003_100351

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x / Real.log 3) + (Real.log x / Real.log 9) = 5 → x = (3^10)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1003_100351


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1003_100366

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem composition_of_even_is_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1003_100366


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l1003_100306

/-- The number of candy pieces left after combining and eating some. -/
def candy_left (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  katie_candy + sister_candy - eaten_candy

/-- Theorem stating the number of candy pieces left in the given scenario. -/
theorem halloween_candy_theorem :
  candy_left 8 23 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l1003_100306


namespace NUMINAMATH_CALUDE_ellipse_equation_l1003_100382

/-- Given an ellipse with semi-major axis a, eccentricity e, and a triangle AF₁B 
    formed by a line through the right focus F₂ intersecting the ellipse at A and B, 
    prove that if a = √3, e = √3/3, and the perimeter of AF₁B is 4√3, 
    then the equation of the ellipse is x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : c / a = Real.sqrt 3 / 3) 
  (h3 : 4 * a = 4 * Real.sqrt 3) (h4 : b^2 = a^2 - c^2) (h5 : a > b) (h6 : b > 0) :
  ∀ (x y : ℝ), x^2 / 3 + y^2 / 2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1003_100382


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l1003_100377

/-- A regular hexagon -/
structure RegularHexagon where
  -- Add any necessary properties here

/-- A diagonal of a regular hexagon -/
structure Diagonal (h : RegularHexagon) where
  -- Add any necessary properties here

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (h : RegularHexagon) (d1 d2 : Diagonal h) : Prop :=
  sorry

/-- The set of all diagonals in a regular hexagon -/
def all_diagonals (h : RegularHexagon) : Set (Diagonal h) :=
  sorry

/-- The probability that two randomly chosen diagonals intersect inside the hexagon -/
def intersection_probability (h : RegularHexagon) : ℚ :=
  sorry

theorem hexagon_diagonal_intersection_probability (h : RegularHexagon) :
  intersection_probability h = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l1003_100377


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1003_100330

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540 degrees -/
theorem sum_interior_angles_pentagon : 
  sum_interior_angles pentagon_sides = 540 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1003_100330


namespace NUMINAMATH_CALUDE_square_difference_1002_1000_l1003_100327

theorem square_difference_1002_1000 : 1002^2 - 1000^2 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_1002_1000_l1003_100327


namespace NUMINAMATH_CALUDE_product_equality_l1003_100358

theorem product_equality (h : 213 * 16 = 3408) : 1.6 * 21.3 = 34.08 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1003_100358


namespace NUMINAMATH_CALUDE_apples_per_box_l1003_100395

/-- Given the following conditions:
    - There are 180 apples in each crate
    - 12 crates of apples were delivered
    - 160 apples were rotten and thrown away
    - The remaining apples were packed into 100 boxes
    Prove that there are 20 apples in each box -/
theorem apples_per_box :
  ∀ (apples_per_crate crates_delivered rotten_apples total_boxes : ℕ),
    apples_per_crate = 180 →
    crates_delivered = 12 →
    rotten_apples = 160 →
    total_boxes = 100 →
    (apples_per_crate * crates_delivered - rotten_apples) / total_boxes = 20 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_box_l1003_100395


namespace NUMINAMATH_CALUDE_cost_per_set_is_20_verify_profit_equation_l1003_100371

/-- Represents the manufacturing and sales scenario of horseshoe sets -/
structure HorseshoeManufacturing where
  initialOutlay : ℝ
  sellingPrice : ℝ
  setsSold : ℕ
  profit : ℝ
  costPerSet : ℝ

/-- The cost per set is $20 given the specified conditions -/
theorem cost_per_set_is_20 (h : HorseshoeManufacturing) 
    (h_initialOutlay : h.initialOutlay = 10000)
    (h_sellingPrice : h.sellingPrice = 50)
    (h_setsSold : h.setsSold = 500)
    (h_profit : h.profit = 5000) :
    h.costPerSet = 20 := by
  sorry

/-- Verifies that the calculated cost per set satisfies the profit equation -/
theorem verify_profit_equation (h : HorseshoeManufacturing) 
    (h_initialOutlay : h.initialOutlay = 10000)
    (h_sellingPrice : h.sellingPrice = 50)
    (h_setsSold : h.setsSold = 500)
    (h_profit : h.profit = 5000)
    (h_costPerSet : h.costPerSet = 20) :
    h.profit = h.sellingPrice * h.setsSold - (h.initialOutlay + h.costPerSet * h.setsSold) := by
  sorry

end NUMINAMATH_CALUDE_cost_per_set_is_20_verify_profit_equation_l1003_100371


namespace NUMINAMATH_CALUDE_inequality_proof_l1003_100341

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hxy : x * y < 0) :
  (b / a + a / b ≥ 2) ∧ (x / y + y / x ≤ -2) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1003_100341


namespace NUMINAMATH_CALUDE_cosine_sum_and_square_l1003_100342

theorem cosine_sum_and_square (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_and_square_l1003_100342


namespace NUMINAMATH_CALUDE_last_bead_color_l1003_100350

def bead_colors := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def necklace_length : Nat := 85

theorem last_bead_color (h : necklace_length = 85) :
  bead_colors[(necklace_length - 1) % bead_colors.length] = "yellow" := by
  sorry

end NUMINAMATH_CALUDE_last_bead_color_l1003_100350


namespace NUMINAMATH_CALUDE_abcNegative_neither_sufficient_nor_necessary_l1003_100329

-- Define a struct to represent the curve ax^2 + by^2 = c
structure QuadraticCurve where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a QuadraticCurve to be a hyperbola
def isHyperbola (curve : QuadraticCurve) : Prop :=
  sorry

-- Define the condition abc < 0
def abcNegative (curve : QuadraticCurve) : Prop :=
  curve.a * curve.b * curve.c < 0

-- Theorem stating that abcNegative is neither sufficient nor necessary for isHyperbola
theorem abcNegative_neither_sufficient_nor_necessary :
  (∃ curve : QuadraticCurve, abcNegative curve ∧ ¬isHyperbola curve) ∧
  (∃ curve : QuadraticCurve, isHyperbola curve ∧ ¬abcNegative curve) :=
sorry

end NUMINAMATH_CALUDE_abcNegative_neither_sufficient_nor_necessary_l1003_100329


namespace NUMINAMATH_CALUDE_trampoline_jumps_l1003_100324

/-- The number of times the first person jumped on the trampoline -/
def ronald_jumps : ℕ := 157

/-- The number of times the second person jumped on the trampoline -/
def rupert_jumps : ℕ := ronald_jumps + 86

/-- The total number of jumps on the trampoline -/
def total_jumps : ℕ := 400

theorem trampoline_jumps : ronald_jumps + rupert_jumps = total_jumps := by
  sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l1003_100324


namespace NUMINAMATH_CALUDE_power_of_three_in_product_l1003_100308

theorem power_of_three_in_product (w : ℕ+) : 
  (∃ k : ℕ, 936 * w = 2^5 * 11^2 * k) → 
  (132 ≤ w) →
  (∃ m : ℕ, 936 * w = 3^3 * m ∧ ∀ n > 3, ¬(∃ l : ℕ, 936 * w = 3^n * l)) := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_in_product_l1003_100308


namespace NUMINAMATH_CALUDE_direction_cannot_determine_position_l1003_100307

-- Define a type for positions
structure Position where
  x : ℝ
  y : ℝ

-- Define a type for directions
structure Direction where
  angle : ℝ

-- Define a function to check if a piece of data can determine a position
def canDeterminePosition (data : Type) : Prop :=
  ∃ (f : data → Position), Function.Injective f

-- Theorem statement
theorem direction_cannot_determine_position :
  ¬ (canDeterminePosition Direction) :=
sorry

end NUMINAMATH_CALUDE_direction_cannot_determine_position_l1003_100307


namespace NUMINAMATH_CALUDE_principal_value_range_of_argument_l1003_100354

theorem principal_value_range_of_argument (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (k : ℕ) (θ : ℝ), k ≤ 1 ∧ 
  Complex.arg z = θ ∧
  k * Real.pi - Real.arccos (-1/2) ≤ θ ∧ 
  θ ≤ k * Real.pi + Real.arccos (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_principal_value_range_of_argument_l1003_100354


namespace NUMINAMATH_CALUDE_max_square_garden_area_l1003_100385

theorem max_square_garden_area (perimeter : ℕ) (side_length : ℕ) : 
  perimeter = 160 →
  4 * side_length = perimeter →
  ∀ s : ℕ, 4 * s ≤ perimeter → s ^ 2 ≤ side_length ^ 2 :=
by
  sorry

#check max_square_garden_area

end NUMINAMATH_CALUDE_max_square_garden_area_l1003_100385


namespace NUMINAMATH_CALUDE_equivalent_discount_equivalent_discount_proof_l1003_100393

theorem equivalent_discount : ℝ → Prop :=
  fun x => 
    let first_discount := 0.15
    let second_discount := 0.10
    let third_discount := 0.05
    let price_after_discounts := (1 - first_discount) * (1 - second_discount) * (1 - third_discount) * x
    let equivalent_single_discount := 0.273
    price_after_discounts = (1 - equivalent_single_discount) * x

-- The proof is omitted
theorem equivalent_discount_proof : ∀ x : ℝ, equivalent_discount x :=
  sorry

end NUMINAMATH_CALUDE_equivalent_discount_equivalent_discount_proof_l1003_100393


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l1003_100322

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion (x y z : ℝ) :
  x = 3 ∧ y = -3 * Real.sqrt 3 ∧ z = 5 →
  ∃ (r θ : ℝ),
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 6 ∧
    θ = 4 * Real.pi / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = 5 :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l1003_100322


namespace NUMINAMATH_CALUDE_three_digit_number_transformation_l1003_100311

theorem three_digit_number_transformation (n : ℕ) (x y z : ℕ) : 
  x * 100 + y * 10 + z = 178 → 
  n = 2 → 
  (x + n) * 100 + (y - n) * 10 + (z - n) = n * (x * 100 + y * 10 + z) := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_transformation_l1003_100311


namespace NUMINAMATH_CALUDE_ammonium_nitrate_formation_l1003_100303

-- Define the chemical species
def Ammonia : Type := Unit
def NitricAcid : Type := Unit
def AmmoniumNitrate : Type := Unit

-- Define the reaction
def reaction (nh3 : ℕ) (hno3 : ℕ) : ℕ :=
  min nh3 hno3

-- State the theorem
theorem ammonium_nitrate_formation 
  (nh3 : ℕ) -- Some moles of Ammonia
  (hno3 : ℕ) -- Moles of Nitric acid
  (h1 : hno3 = 3) -- 3 moles of Nitric acid are used
  (h2 : reaction nh3 hno3 = 3) -- Total moles of Ammonium nitrate formed are 3
  : reaction nh3 hno3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ammonium_nitrate_formation_l1003_100303


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1003_100383

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 2 ∧ 
  (13 ∣ (964807 - x)) ∧ 
  ∀ (y : ℕ), y < x → ¬(13 ∣ (964807 - y)) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1003_100383


namespace NUMINAMATH_CALUDE_range_of_c_l1003_100375

/-- A condition is sufficient but not necessary -/
def SufficientButNotNecessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

theorem range_of_c (a c : ℝ) :
  SufficientButNotNecessary (a ≥ 1/8) (∀ x > 0, 2*x + a/x ≥ c) →
  c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l1003_100375


namespace NUMINAMATH_CALUDE_green_light_most_probable_l1003_100323

-- Define the durations of each light
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

-- Define the total cycle duration
def total_duration : ℕ := red_duration + yellow_duration + green_duration

-- Define the probabilities of encountering each light
def prob_red : ℚ := red_duration / total_duration
def prob_yellow : ℚ := yellow_duration / total_duration
def prob_green : ℚ := green_duration / total_duration

-- Theorem: The probability of encountering a green light is higher than the other lights
theorem green_light_most_probable : 
  prob_green > prob_red ∧ prob_green > prob_yellow :=
sorry

end NUMINAMATH_CALUDE_green_light_most_probable_l1003_100323


namespace NUMINAMATH_CALUDE_max_twin_prime_sum_200_l1003_100301

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_twin_prime (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q - p = 2

def max_twin_prime_sum : ℕ := 396

theorem max_twin_prime_sum_200 :
  ∀ p q : ℕ,
  p ≤ 200 → q ≤ 200 →
  is_twin_prime p q →
  p + q ≤ max_twin_prime_sum :=
sorry

end NUMINAMATH_CALUDE_max_twin_prime_sum_200_l1003_100301


namespace NUMINAMATH_CALUDE_bottle_production_l1003_100379

/-- Given that 6 identical machines produce 300 bottles per minute at a constant rate,
    10 such machines will produce 2000 bottles in 4 minutes. -/
theorem bottle_production (machines : ℕ) (bottles_per_minute : ℕ) (time : ℕ) : 
  machines = 6 → bottles_per_minute = 300 → time = 4 →
  (10 : ℕ) * bottles_per_minute * time / machines = 2000 :=
by sorry

end NUMINAMATH_CALUDE_bottle_production_l1003_100379


namespace NUMINAMATH_CALUDE_sum_first_20_triangular_numbers_l1003_100374

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular_numbers (n : ℕ) : ℕ :=
  (List.range n).map triangular_number |>.sum

/-- Theorem: The sum of the first 20 triangular numbers is 1540 -/
theorem sum_first_20_triangular_numbers :
  sum_triangular_numbers 20 = 1540 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_20_triangular_numbers_l1003_100374


namespace NUMINAMATH_CALUDE_inequality_proof_l1003_100357

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x^4 + y^2*z^2)/(x^(5/2)*(y+z)) + 
  (y^4 + z^2*x^2)/(y^(5/2)*(z+x)) + 
  (z^4 + y^2*x^2)/(z^(5/2)*(y+x)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1003_100357


namespace NUMINAMATH_CALUDE_flood_damage_conversion_l1003_100334

/-- Conversion of flood damage from Canadian to American dollars -/
theorem flood_damage_conversion (damage_cad : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_cad = 50000000)
  (h2 : exchange_rate = 1.25)
  : damage_cad / exchange_rate = 40000000 := by
  sorry

end NUMINAMATH_CALUDE_flood_damage_conversion_l1003_100334


namespace NUMINAMATH_CALUDE_min_value_expression_l1003_100319

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1003_100319


namespace NUMINAMATH_CALUDE_jane_max_tickets_l1003_100363

/-- Represents the maximum number of tickets Jane can buy given the conditions. -/
def max_tickets (regular_price : ℕ) (discount_price : ℕ) (budget : ℕ) (discount_threshold : ℕ) : ℕ :=
  let regular_tickets := min discount_threshold (budget / regular_price)
  let remaining_budget := budget - regular_tickets * regular_price
  let extra_tickets := remaining_budget / discount_price
  regular_tickets + extra_tickets

/-- Theorem stating that the maximum number of tickets Jane can buy is 19. -/
theorem jane_max_tickets :
  max_tickets 15 12 135 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l1003_100363


namespace NUMINAMATH_CALUDE_linear_function_properties_l1003_100349

/-- A linear function of the form y = mx + 4m - 2 -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := m * x + 4 * m - 2

theorem linear_function_properties :
  ∃ m : ℝ, 
    (∃ y : ℝ, y ≠ -2 ∧ linear_function m 0 = y) ∧ 
    (let f := linear_function (1/3);
     ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ f x₁ > 0 ∧ x₂ < 0 ∧ f x₂ < 0 ∧ x₃ > 0 ∧ f x₃ < 0) ∧
    (linear_function (1/2) 0 = 0) ∧
    (∀ m : ℝ, linear_function m (-4) = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1003_100349


namespace NUMINAMATH_CALUDE_max_value_operation_l1003_100302

theorem max_value_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 4 * (300 - n) ≤ 1160 :=
by sorry

end NUMINAMATH_CALUDE_max_value_operation_l1003_100302


namespace NUMINAMATH_CALUDE_min_value_3a_4b_l1003_100352

theorem min_value_3a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a + b) * (a + 2 * b) + a + b = 9) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ (x + y) * (x + 2 * y) + x + y = 9 → 
  3 * x + 4 * y ≥ 6 * Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_3a_4b_l1003_100352


namespace NUMINAMATH_CALUDE_solve_trailer_problem_l1003_100370

/-- Represents the trailer home problem --/
def trailer_problem (initial_count : ℕ) (initial_avg_age : ℕ) (current_avg_age : ℕ) (time_elapsed : ℕ) : Prop :=
  ∃ (new_count : ℕ),
    (initial_count * (initial_avg_age + time_elapsed) + new_count * time_elapsed) / (initial_count + new_count) = current_avg_age

/-- The theorem statement for the trailer home problem --/
theorem solve_trailer_problem :
  trailer_problem 30 15 10 3 → ∃ (new_count : ℕ), new_count = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_trailer_problem_l1003_100370


namespace NUMINAMATH_CALUDE_matrix_product_equality_l1003_100321

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 4]

theorem matrix_product_equality :
  A * B = !![23, -5; 24, -20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l1003_100321


namespace NUMINAMATH_CALUDE_intersection_distance_l1003_100373

/-- The distance between the intersection points of the line y = x and the circle (x-2)^2 + (y-1)^2 = 1 is √2. -/
theorem intersection_distance :
  ∃ (P Q : ℝ × ℝ),
    (P.1 = P.2 ∧ (P.1 - 2)^2 + (P.2 - 1)^2 = 1) ∧
    (Q.1 = Q.2 ∧ (Q.1 - 2)^2 + (Q.2 - 1)^2 = 1) ∧
    P ≠ Q ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l1003_100373


namespace NUMINAMATH_CALUDE_sum_of_integers_l1003_100338

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 8) (h2 : a * b = 65) : a + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1003_100338


namespace NUMINAMATH_CALUDE_peter_bird_count_l1003_100390

/-- The fraction of birds that are ducks -/
def duck_fraction : ℚ := 1/3

/-- The cost of chicken feed per bird in dollars -/
def chicken_feed_cost : ℚ := 2

/-- The total cost to feed all chickens in dollars -/
def total_chicken_feed_cost : ℚ := 20

/-- The total number of birds Peter has -/
def total_birds : ℕ := 15

theorem peter_bird_count :
  (1 - duck_fraction) * total_birds = total_chicken_feed_cost / chicken_feed_cost :=
by sorry

end NUMINAMATH_CALUDE_peter_bird_count_l1003_100390


namespace NUMINAMATH_CALUDE_lucia_dance_cost_l1003_100369

/-- Represents the cost of dance classes for a week -/
structure DanceClassesCost where
  hip_hop_classes : Nat
  ballet_classes : Nat
  jazz_classes : Nat
  hip_hop_cost : Nat
  ballet_cost : Nat
  jazz_cost : Nat

/-- Calculates the total cost of dance classes for a week -/
def total_cost (c : DanceClassesCost) : Nat :=
  c.hip_hop_classes * c.hip_hop_cost +
  c.ballet_classes * c.ballet_cost +
  c.jazz_classes * c.jazz_cost

/-- Theorem stating that Lucia's total dance class cost for a week is $52 -/
theorem lucia_dance_cost :
  let c : DanceClassesCost := {
    hip_hop_classes := 2,
    ballet_classes := 2,
    jazz_classes := 1,
    hip_hop_cost := 10,
    ballet_cost := 12,
    jazz_cost := 8
  }
  total_cost c = 52 := by
  sorry


end NUMINAMATH_CALUDE_lucia_dance_cost_l1003_100369


namespace NUMINAMATH_CALUDE_problem_statement_l1003_100353

theorem problem_statement (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + c + 1 = 0) 
  (h3 : a = 1) : 
  b^2 - 4*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1003_100353


namespace NUMINAMATH_CALUDE_power_of_power_l1003_100394

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1003_100394


namespace NUMINAMATH_CALUDE_eating_out_budget_fraction_l1003_100316

/-- Given a family's budget allocation, calculate the fraction spent on eating out -/
theorem eating_out_budget_fraction 
  (grocery_fraction : ℝ) 
  (total_food_fraction : ℝ) 
  (h1 : grocery_fraction = 0.6) 
  (h2 : total_food_fraction = 0.8) : 
  total_food_fraction - grocery_fraction = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_eating_out_budget_fraction_l1003_100316


namespace NUMINAMATH_CALUDE_glove_selection_theorem_l1003_100317

/-- The number of pairs of gloves -/
def num_pairs : ℕ := 4

/-- The number of gloves to be selected -/
def num_selected : ℕ := 4

/-- The total number of gloves -/
def total_gloves : ℕ := 2 * num_pairs

/-- The number of ways to select gloves such that at least two form a pair -/
def ways_with_pair : ℕ := 54

theorem glove_selection_theorem :
  (Nat.choose total_gloves num_selected) - (2^num_pairs) = ways_with_pair :=
sorry

end NUMINAMATH_CALUDE_glove_selection_theorem_l1003_100317


namespace NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l1003_100392

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.c = 2 ∧ 2 * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.C

theorem angle_C_measure (t : Triangle) (h : TriangleConditions t) : 
  t.C = π / 3 := by sorry

theorem triangle_area (t : Triangle) (h : TriangleConditions t) 
  (h2 : 2 * Real.sin (2 * t.A) + Real.sin (2 * t.B + t.C) = Real.sin t.C) : 
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l1003_100392


namespace NUMINAMATH_CALUDE_total_wrappers_collected_l1003_100300

/-- The total number of wrappers collected by four friends is the sum of their individual collections. -/
theorem total_wrappers_collected
  (andy_wrappers : ℕ)
  (max_wrappers : ℕ)
  (zoe_wrappers : ℕ)
  (mia_wrappers : ℕ)
  (h1 : andy_wrappers = 34)
  (h2 : max_wrappers = 15)
  (h3 : zoe_wrappers = 25)
  (h4 : mia_wrappers = 19) :
  andy_wrappers + max_wrappers + zoe_wrappers + mia_wrappers = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_wrappers_collected_l1003_100300


namespace NUMINAMATH_CALUDE_base_8_subtraction_example_l1003_100304

/-- Subtraction in base 8 -/
def base_8_subtraction (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 8 -/
def to_base_8 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 8 to base 10 -/
def from_base_8 (n : ℕ) : ℕ :=
  sorry

theorem base_8_subtraction_example :
  base_8_subtraction (from_base_8 7463) (from_base_8 3154) = from_base_8 4317 :=
sorry

end NUMINAMATH_CALUDE_base_8_subtraction_example_l1003_100304


namespace NUMINAMATH_CALUDE_equation_solutions_l1003_100339

theorem equation_solutions :
  (∃ x : ℚ, 4 - 3 * x = 6 - 5 * x ∧ x = 1) ∧
  (∃ x : ℚ, 7 - 3 * (x - 1) = -x ∧ x = 5) ∧
  (∃ x : ℚ, (3 * x - 1) / 2 = 1 - (x - 1) / 6 ∧ x = 1) ∧
  (∃ x : ℚ, (2 * x - 1) / 3 - x = (2 * x + 1) / 4 - 1 ∧ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1003_100339


namespace NUMINAMATH_CALUDE_white_ring_weight_l1003_100326

/-- Given the weights of three plastic rings (orange, purple, and white) and their total weight,
    this theorem proves that the weight of the white ring is 0.42 ounces. -/
theorem white_ring_weight
  (orange_weight : ℝ)
  (purple_weight : ℝ)
  (total_weight : ℝ)
  (h1 : orange_weight = 0.08)
  (h2 : purple_weight = 0.33)
  (h3 : total_weight = 0.83)
  : total_weight - (orange_weight + purple_weight) = 0.42 := by
  sorry

#eval (0.83 : ℝ) - ((0.08 : ℝ) + (0.33 : ℝ))

end NUMINAMATH_CALUDE_white_ring_weight_l1003_100326


namespace NUMINAMATH_CALUDE_jack_final_plate_count_l1003_100309

/-- Represents the number of plates Jack has of each type and the total number of plates --/
structure PlateCount where
  flower : ℕ
  checked : ℕ
  polkaDot : ℕ
  total : ℕ

/-- Calculates the final number of plates Jack has --/
def finalPlateCount (initial : PlateCount) : PlateCount :=
  let newPolkaDot := 2 * initial.checked
  let newFlower := initial.flower - 1
  { flower := newFlower
  , checked := initial.checked
  , polkaDot := newPolkaDot
  , total := newFlower + initial.checked + newPolkaDot
  }

/-- Theorem stating that Jack ends up with 27 plates --/
theorem jack_final_plate_count :
  let initial := { flower := 4, checked := 8, polkaDot := 0, total := 12 : PlateCount }
  (finalPlateCount initial).total = 27 := by
  sorry

end NUMINAMATH_CALUDE_jack_final_plate_count_l1003_100309


namespace NUMINAMATH_CALUDE_unique_prime_digit_l1003_100325

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The six-digit number formed by appending B to 30420 -/
def number (B : ℕ) : ℕ := 304200 + B

/-- The theorem stating that there's a unique B that makes the number prime, and it's 1 -/
theorem unique_prime_digit :
  ∃! B : ℕ, B < 10 ∧ isPrime (number B) ∧ B = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_digit_l1003_100325


namespace NUMINAMATH_CALUDE_fraction_value_l1003_100397

theorem fraction_value (x y : ℝ) (h : 2 * x = -y) :
  x * y / (x^2 - y^2) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_value_l1003_100397


namespace NUMINAMATH_CALUDE_seedling_ratio_l1003_100305

theorem seedling_ratio (first_day : ℕ) (total : ℕ) : 
  first_day = 200 → total = 1200 → 
  (total - first_day) / first_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_seedling_ratio_l1003_100305


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1003_100345

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 + 1 ≥ C * (a + b + c)) ∧
  (C = 2 / Real.sqrt 3) ∧
  (∀ (D : ℝ), (∀ (a b c : ℝ), a^2 + b^2 + c^2 + 1 ≥ D * (a + b + c)) → D ≤ C) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1003_100345


namespace NUMINAMATH_CALUDE_regression_line_equation_l1003_100340

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a given equation represents the regression line -/
def is_regression_line_equation (line : RegressionLine) (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = line.slope * (x - line.point.1) + line.point.2) ∧
  (f line.point.1 = line.point.2)

/-- The theorem stating the equation of the specific regression line -/
theorem regression_line_equation (line : RegressionLine) 
  (h1 : line.slope = 6.5)
  (h2 : line.point = (2, 3)) :
  is_regression_line_equation line (λ x => -10 + 6.5 * x) := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l1003_100340


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1003_100376

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  ∀ k : ℝ,
  let a : ℝ × ℝ := (2 * k + 2, 4)
  let b : ℝ × ℝ := (k + 1, 8)
  are_parallel a b → k = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1003_100376


namespace NUMINAMATH_CALUDE_least_number_of_marbles_eight_forty_satisfies_least_number_is_eight_forty_l1003_100313

theorem least_number_of_marbles (n : ℕ) : n > 0 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 840 := by
  sorry

theorem eight_forty_satisfies (n : ℕ) : 
  3 ∣ 840 ∧ 4 ∣ 840 ∧ 5 ∣ 840 ∧ 7 ∣ 840 ∧ 8 ∣ 840 := by
  sorry

theorem least_number_is_eight_forty : 
  ∃ (n : ℕ), n > 0 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧
  ∀ (m : ℕ), (m > 0 ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 8 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_marbles_eight_forty_satisfies_least_number_is_eight_forty_l1003_100313


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l1003_100365

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/5
def prob_trap : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  33614/1250000 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l1003_100365


namespace NUMINAMATH_CALUDE_sin_45_degrees_l1003_100312

theorem sin_45_degrees :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l1003_100312


namespace NUMINAMATH_CALUDE_composition_equality_l1003_100343

-- Define the functions f and h
def f (m n x : ℝ) : ℝ := m * x + n
def h (p q r x : ℝ) : ℝ := p * x^2 + q * x + r

-- State the theorem
theorem composition_equality (m n p q r : ℝ) :
  (∀ x, f m n (h p q r x) = h p q r (f m n x)) ↔ (m = p ∧ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l1003_100343
