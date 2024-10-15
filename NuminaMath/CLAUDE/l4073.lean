import Mathlib

namespace NUMINAMATH_CALUDE_sum_even_number_of_even_is_even_sum_even_number_of_odd_is_even_sum_odd_number_of_even_is_even_sum_odd_number_of_odd_is_odd_l4073_407377

-- Define what it means for a number to be even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what it means for a number to be odd
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Define a function that sums a list of integers
def SumList (list : List ℤ) : ℤ := list.foldl (· + ·) 0

-- Theorem 1: Sum of an even number of even integers is even
theorem sum_even_number_of_even_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, IsEven x) : 
  IsEven (SumList list) := by sorry

-- Theorem 2: Sum of an even number of odd integers is even
theorem sum_even_number_of_odd_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, IsOdd x) : 
  IsEven (SumList list) := by sorry

-- Theorem 3: Sum of an odd number of even integers is even
theorem sum_odd_number_of_even_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, IsEven x) : 
  IsEven (SumList list) := by sorry

-- Theorem 4: Sum of an odd number of odd integers is odd
theorem sum_odd_number_of_odd_is_odd (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, IsOdd x) : 
  IsOdd (SumList list) := by sorry

end NUMINAMATH_CALUDE_sum_even_number_of_even_is_even_sum_even_number_of_odd_is_even_sum_odd_number_of_even_is_even_sum_odd_number_of_odd_is_odd_l4073_407377


namespace NUMINAMATH_CALUDE_marble_jar_problem_l4073_407302

theorem marble_jar_problem (jar1_blue_ratio : ℚ) (jar1_green_ratio : ℚ)
  (jar2_blue_ratio : ℚ) (jar2_green_ratio : ℚ) (total_green : ℕ) :
  jar1_blue_ratio = 7 / 10 →
  jar1_green_ratio = 3 / 10 →
  jar2_blue_ratio = 6 / 10 →
  jar2_green_ratio = 4 / 10 →
  total_green = 80 →
  ∃ (total_jar1 total_jar2 : ℕ),
    total_jar1 = total_jar2 ∧
    (jar1_green_ratio * total_jar1 + jar2_green_ratio * total_jar2 : ℚ) = total_green ∧
    ⌊jar1_blue_ratio * total_jar1 - jar2_blue_ratio * total_jar2⌋ = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l4073_407302


namespace NUMINAMATH_CALUDE_sqrt_negative_square_defined_unique_l4073_407339

theorem sqrt_negative_square_defined_unique : 
  ∃! a : ℝ, ∃ x : ℝ, x^2 = -(1-a)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_square_defined_unique_l4073_407339


namespace NUMINAMATH_CALUDE_equation_solutions_count_l4073_407376

theorem equation_solutions_count : 
  ∃! (pairs : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ pairs ↔ (1 : ℚ) / y - (1 : ℚ) / (y + 2) = (1 : ℚ) / (3 * 2^x)) ∧
    pairs.length = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l4073_407376


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l4073_407330

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l4073_407330


namespace NUMINAMATH_CALUDE_max_books_borrowed_l4073_407347

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 60) (h2 : zero_books = 4) (h3 : one_book = 18) 
  (h4 : two_books = 20) (h5 : avg_books = 5/2) : 
  ∃ (max_books : ℕ), max_books = 41 ∧ 
  ∀ (student_books : ℕ), student_books ≤ max_books :=
by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l4073_407347


namespace NUMINAMATH_CALUDE_min_operations_to_256_l4073_407333

/-- Represents the allowed operations -/
inductive Operation
  | AddOne
  | MultiplyTwo

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : OperationSequence) : ℕ :=
  ops.foldl (fun n op => match op with
    | Operation.AddOne => n + 1
    | Operation.MultiplyTwo => n * 2) start

/-- Checks if a sequence of operations transforms start into target -/
def isValidSequence (start target : ℕ) (ops : OperationSequence) : Prop :=
  applyOperations start ops = target

/-- The main theorem to be proved -/
theorem min_operations_to_256 :
  ∃ (ops : OperationSequence), isValidSequence 1 256 ops ∧ 
    ops.length = 8 ∧
    (∀ (other_ops : OperationSequence), isValidSequence 1 256 other_ops → 
      ops.length ≤ other_ops.length) :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_256_l4073_407333


namespace NUMINAMATH_CALUDE_negative_twenty_one_div_three_l4073_407360

theorem negative_twenty_one_div_three : -21 / 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_negative_twenty_one_div_three_l4073_407360


namespace NUMINAMATH_CALUDE_car_sales_profit_percentage_l4073_407364

/-- Represents the sale of a car with its selling price and profit/loss percentage -/
structure CarSale where
  selling_price : ℝ
  profit_percentage : ℝ

/-- Calculates the overall profit percentage for a list of car sales -/
def overall_profit_percentage (sales : List CarSale) : ℝ :=
  sorry

/-- The main theorem stating the overall profit percentage for the given car sales -/
theorem car_sales_profit_percentage : 
  let sales := [
    CarSale.mk 404415 15,
    CarSale.mk 404415 (-15),
    CarSale.mk 550000 10
  ]
  abs (overall_profit_percentage sales - 2.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_car_sales_profit_percentage_l4073_407364


namespace NUMINAMATH_CALUDE_eric_return_time_l4073_407345

def running_time : ℕ := 20
def jogging_time : ℕ := 10
def time_to_park : ℕ := running_time + jogging_time
def return_time_factor : ℕ := 3

theorem eric_return_time : time_to_park * return_time_factor = 90 := by
  sorry

end NUMINAMATH_CALUDE_eric_return_time_l4073_407345


namespace NUMINAMATH_CALUDE_measure_of_angle_ABC_l4073_407384

-- Define the angles
def angle_ABC : ℝ := sorry
def angle_ABD : ℝ := 30
def angle_CBD : ℝ := 90

-- State the theorem
theorem measure_of_angle_ABC :
  angle_ABC = 60 ∧ 
  angle_CBD = 90 ∧ 
  angle_ABD = 30 ∧ 
  angle_ABC + angle_ABD + angle_CBD = 180 :=
by sorry

end NUMINAMATH_CALUDE_measure_of_angle_ABC_l4073_407384


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l4073_407310

theorem quadratic_unique_solution (b c : ℝ) : 
  (∃! x, 3 * x^2 + b * x + c = 0) →
  b + c = 15 →
  3 * c = b^2 →
  b = (-3 + 3 * Real.sqrt 21) / 2 ∧ 
  c = (33 - 3 * Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l4073_407310


namespace NUMINAMATH_CALUDE_smallest_bound_inequality_l4073_407303

theorem smallest_bound_inequality (a b c : ℝ) : 
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ ε > 0, ∃ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| > (M - ε)*(a^2 + b^2 + c^2)^2 ∧
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_bound_inequality_l4073_407303


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l4073_407324

theorem inequality_proof (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 4) : 
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) := by
  sorry

theorem equality_condition (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 = 1 + Real.sqrt (a * b * c * d) ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l4073_407324


namespace NUMINAMATH_CALUDE_decimal_computation_l4073_407337

theorem decimal_computation : (0.25 / 0.005) * 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_computation_l4073_407337


namespace NUMINAMATH_CALUDE_mark_gigs_total_duration_l4073_407365

/-- Represents the duration of Mark's gigs over two weeks -/
def MarkGigsDuration : ℕ :=
  let days_in_two_weeks : ℕ := 2 * 7
  let gigs_count : ℕ := days_in_two_weeks / 2
  let short_song_duration : ℕ := 5
  let long_song_duration : ℕ := 2 * short_song_duration
  let gig_duration : ℕ := 2 * short_song_duration + long_song_duration
  gigs_count * gig_duration

theorem mark_gigs_total_duration :
  MarkGigsDuration = 140 := by
  sorry

end NUMINAMATH_CALUDE_mark_gigs_total_duration_l4073_407365


namespace NUMINAMATH_CALUDE_trig_roots_equation_l4073_407350

theorem trig_roots_equation (θ : ℝ) (a : ℝ) :
  (∀ x, x^2 - a*x + a = 0 ↔ x = Real.sin θ ∨ x = Real.cos θ) →
  Real.cos (θ - 3*Real.pi/2) + Real.sin (3*Real.pi/2 + θ) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_roots_equation_l4073_407350


namespace NUMINAMATH_CALUDE_age_relationships_l4073_407317

-- Define variables for ages
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := a + b = b + c + d + 18
def condition2 : Prop := a / c = 3 / 2

-- Define the theorem
theorem age_relationships 
  (h1 : condition1 a b c d) 
  (h2 : condition2 a c) : 
  c = (2/3) * a ∧ d = (1/3) * a - 18 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_age_relationships_l4073_407317


namespace NUMINAMATH_CALUDE_billy_video_watching_l4073_407331

def total_time : ℕ := 90
def video_watch_time : ℕ := 4
def search_time : ℕ := 3
def break_time : ℕ := 5
def trial_count : ℕ := 5
def suggestions_per_trial : ℕ := 15
def additional_categories : ℕ := 2
def suggestions_per_category : ℕ := 10

def max_videos_watched : ℕ := 13

theorem billy_video_watching :
  let total_search_time := search_time * trial_count
  let total_break_time := break_time * (trial_count - 1)
  let available_watch_time := total_time - (total_search_time + total_break_time)
  max_videos_watched = available_watch_time / video_watch_time ∧
  max_videos_watched ≤ suggestions_per_trial * trial_count +
                       suggestions_per_category * additional_categories :=
by sorry

end NUMINAMATH_CALUDE_billy_video_watching_l4073_407331


namespace NUMINAMATH_CALUDE_permutations_of_red_l4073_407340

def word : String := "red"

theorem permutations_of_red (w : String) (h : w = word) : 
  Nat.factorial w.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_red_l4073_407340


namespace NUMINAMATH_CALUDE_max_value_theorem_l4073_407355

theorem max_value_theorem (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 9) 
  (h2 : 3 * a + 5 * b ≤ 12) : 
  2 * a + b ≤ 39 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4073_407355


namespace NUMINAMATH_CALUDE_fermat_number_prime_divisors_l4073_407342

theorem fermat_number_prime_divisors (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_divides : p ∣ 2^(2^n) + 1) : 
  ∃ k : ℕ, p = k * 2^(n + 1) + 1 := by
sorry

end NUMINAMATH_CALUDE_fermat_number_prime_divisors_l4073_407342


namespace NUMINAMATH_CALUDE_largest_multiple_under_1000_l4073_407363

theorem largest_multiple_under_1000 : ∃ n : ℕ, 
  n = 990 ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  n < 1000 ∧
  ∀ m : ℕ, (m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_under_1000_l4073_407363


namespace NUMINAMATH_CALUDE_quadratic_equation_transform_l4073_407358

/-- Given a quadratic equation 4x^2 + 16x - 400 = 0, prove that when transformed
    into the form (x + k)^2 = t, the value of t is 104. -/
theorem quadratic_equation_transform (x k t : ℝ) : 
  (4 * x^2 + 16 * x - 400 = 0) → 
  (∃ k, ∀ x, 4 * x^2 + 16 * x - 400 = 0 ↔ (x + k)^2 = t) →
  t = 104 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_transform_l4073_407358


namespace NUMINAMATH_CALUDE_games_played_together_count_l4073_407387

/-- The number of players in the league -/
def totalPlayers : ℕ := 12

/-- The number of players in each game -/
def playersPerGame : ℕ := 6

/-- Function to calculate the number of games two specific players play together -/
def gamesPlayedTogether : ℕ := sorry

theorem games_played_together_count :
  gamesPlayedTogether = 210 := by sorry

end NUMINAMATH_CALUDE_games_played_together_count_l4073_407387


namespace NUMINAMATH_CALUDE_original_number_proof_l4073_407397

theorem original_number_proof :
  ∃ (n : ℕ), n = 3830 ∧ (∃ (k : ℕ), n - 5 = 15 * k) ∧
  (∀ (m : ℕ), m < 5 → ¬(∃ (j : ℕ), n - m = 15 * j)) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l4073_407397


namespace NUMINAMATH_CALUDE_complex_norm_problem_l4073_407373

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 17)
  (h2 : Complex.abs (z + 3 * w) = 4)
  (h3 : Complex.abs (z + w) = 6) :
  Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l4073_407373


namespace NUMINAMATH_CALUDE_jessica_attended_one_game_l4073_407313

/-- The number of soccer games Jessica actually attended -/
def jessica_attended (total games : ℕ) (planned : ℕ) (skipped : ℕ) (rescheduled : ℕ) (additional_missed : ℕ) : ℕ :=
  planned - skipped - additional_missed

/-- Theorem stating that Jessica attended 1 game given the problem conditions -/
theorem jessica_attended_one_game :
  jessica_attended 12 8 3 2 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jessica_attended_one_game_l4073_407313


namespace NUMINAMATH_CALUDE_baking_time_with_oven_failure_l4073_407322

/-- The time taken to make caramel-apple coffee cakes on a day when the oven failed -/
theorem baking_time_with_oven_failure 
  (assembly_time : ℝ) 
  (normal_baking_time : ℝ) 
  (decoration_time : ℝ) 
  (h1 : assembly_time = 1) 
  (h2 : normal_baking_time = 1.5) 
  (h3 : decoration_time = 1) :
  assembly_time + 2 * normal_baking_time + decoration_time = 5 := by
sorry

end NUMINAMATH_CALUDE_baking_time_with_oven_failure_l4073_407322


namespace NUMINAMATH_CALUDE_total_students_in_clubs_l4073_407395

def math_club_size : ℕ := 15
def science_club_size : ℕ := 10
def art_club_size : ℕ := 12
def math_science_overlap : ℕ := 5

theorem total_students_in_clubs : 
  math_club_size + science_club_size + art_club_size - math_science_overlap = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_students_in_clubs_l4073_407395


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l4073_407381

theorem right_triangle_area : ℕ → ℕ → ℕ → Prop :=
  fun a b c =>
    (a * a + b * b = c * c) →  -- Pythagorean theorem
    (2 * b * b - 23 * b + 11 = 0) →  -- One leg satisfies the equation
    (a > 0 ∧ b > 0 ∧ c > 0) →  -- All sides are positive
    ((a * b) / 2 = 330)  -- Area of the triangle

-- The proof of this theorem
theorem right_triangle_area_proof :
  ∃ (a b c : ℕ), right_triangle_area a b c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l4073_407381


namespace NUMINAMATH_CALUDE_sum_and_one_known_l4073_407382

theorem sum_and_one_known (x y : ℤ) : x + y = -26 ∧ x = 11 → y = -37 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_one_known_l4073_407382


namespace NUMINAMATH_CALUDE_triangle_side_length_l4073_407388

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) (pq pr pn : ℝ) : Prop :=
  let (px, py) := t.P
  let (qx, qy) := t.Q
  let (rx, ry) := t.R
  let (nx, ny) := ((qx + rx) / 2, (qy + ry) / 2)  -- N is midpoint of QR
  (px - qx)^2 + (py - qy)^2 = pq^2 ∧  -- PQ = 6
  (px - rx)^2 + (py - ry)^2 = pr^2 ∧  -- PR = 10
  (px - nx)^2 + (py - ny)^2 = pn^2    -- PN = 5

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  is_valid_triangle t 6 10 5 →
  let (qx, qy) := t.Q
  let (rx, ry) := t.R
  (qx - rx)^2 + (qy - ry)^2 = 4 * 43 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4073_407388


namespace NUMINAMATH_CALUDE_fifth_root_equality_l4073_407301

theorem fifth_root_equality : ∃ (x y : ℤ), (119287 - 48682 * Real.sqrt 6) ^ (1/5 : ℝ) = x + y * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_fifth_root_equality_l4073_407301


namespace NUMINAMATH_CALUDE_reflection_segment_length_C_l4073_407327

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_segment_length (x y : ℝ) : ℝ :=
  2 * |y|

/-- Theorem: The length of the segment from C(4, 3) to its reflection C' over the x-axis is 6 --/
theorem reflection_segment_length_C : reflection_segment_length 4 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_segment_length_C_l4073_407327


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l4073_407380

theorem max_sum_of_squares (x y : ℤ) : 3 * x^2 + 5 * y^2 = 345 → (x + y ≤ 13) ∧ ∃ (a b : ℤ), 3 * a^2 + 5 * b^2 = 345 ∧ a + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l4073_407380


namespace NUMINAMATH_CALUDE_problem_statement_l4073_407391

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Define the set T
def T : Set ℝ := {a | -Real.sqrt 3 < a ∧ a < Real.sqrt 3}

-- Theorem statement
theorem problem_statement :
  (∀ x : ℝ, f x > a^2) →
  (∀ m n : ℝ, m ∈ T → n ∈ T → Real.sqrt 3 * |m + n| < |m * n + 3|) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4073_407391


namespace NUMINAMATH_CALUDE_fraction_problem_l4073_407338

theorem fraction_problem (a b : ℝ) (f : ℝ) : 
  a - b = 8 → 
  a + b = 24 → 
  f * (a + b) = 6 → 
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l4073_407338


namespace NUMINAMATH_CALUDE_gym_students_count_l4073_407370

theorem gym_students_count :
  ∀ (students_on_floor : ℕ) (total_students : ℕ),
    -- 4 students are on the bleachers
    total_students = students_on_floor + 4 →
    -- The ratio of students on the floor to total students is 11:13
    (students_on_floor : ℚ) / total_students = 11 / 13 →
    -- The total number of students is 26
    total_students = 26 := by
  sorry

end NUMINAMATH_CALUDE_gym_students_count_l4073_407370


namespace NUMINAMATH_CALUDE_composition_ratio_l4073_407359

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 2 * x - 1

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 79 / 37 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l4073_407359


namespace NUMINAMATH_CALUDE_percentage_of_a_l4073_407393

theorem percentage_of_a (a b c : ℝ) (P : ℝ) : 
  (P / 100) * a = 8 →
  (8 / 100) * b = 4 →
  c = b / a →
  P = 16 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_a_l4073_407393


namespace NUMINAMATH_CALUDE_hyperbola_tangent_coincidence_l4073_407369

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Define the curve
def curve (a x y : ℝ) : Prop := y = a * x^2 + 1/3

-- Define the asymptotes of the hyperbola
def asymptotes (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

-- Define the condition that asymptotes coincide with tangents
def coincide_with_tangents (a : ℝ) : Prop :=
  ∀ x y : ℝ, asymptotes x y → ∃ t : ℝ, curve a t y ∧ 
  (∀ s : ℝ, s ≠ t → curve a s (a * s^2 + 1/3) → (a * s^2 + 1/3 - y) * (s - t) > 0)

-- Theorem statement
theorem hyperbola_tangent_coincidence :
  ∀ a : ℝ, coincide_with_tangents a → a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_coincidence_l4073_407369


namespace NUMINAMATH_CALUDE_undefined_fraction_roots_product_l4073_407385

theorem undefined_fraction_roots_product : ∃ (r₁ r₂ : ℝ), 
  (r₁^2 - 4*r₁ - 12 = 0) ∧ 
  (r₂^2 - 4*r₂ - 12 = 0) ∧ 
  (r₁ ≠ r₂) ∧
  (r₁ * r₂ = -12) := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_roots_product_l4073_407385


namespace NUMINAMATH_CALUDE_tan_roots_problem_l4073_407306

open Real

theorem tan_roots_problem (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : (tan α)^2 - 5*(tan α) + 6 = 0) (h4 : (tan β)^2 - 5*(tan β) + 6 = 0) :
  (α + β = 3*π/4) ∧ ¬∃(x : Real), tan (2*(α + β)) = x := by
  sorry

end NUMINAMATH_CALUDE_tan_roots_problem_l4073_407306


namespace NUMINAMATH_CALUDE_circumradius_range_l4073_407321

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- Points P and Q on side AB, and R on side CD of a unit square -/
structure TrianglePoints (square : UnitSquare) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t, 0)
  Q_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t, 0)
  R_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (t, 1)

/-- The circumradius of a triangle -/
def circumradius (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the range of possible circumradius values -/
theorem circumradius_range (square : UnitSquare) (points : TrianglePoints square) :
  1/2 < circumradius points.P points.Q points.R ∧ 
  circumradius points.P points.Q points.R ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_range_l4073_407321


namespace NUMINAMATH_CALUDE_jenny_sold_192_packs_l4073_407368

-- Define the number of boxes sold
def boxes_sold : Float := 24.0

-- Define the number of packs per box
def packs_per_box : Float := 8.0

-- Define the total number of packs sold
def total_packs : Float := boxes_sold * packs_per_box

-- Theorem statement
theorem jenny_sold_192_packs : total_packs = 192.0 := by
  sorry

end NUMINAMATH_CALUDE_jenny_sold_192_packs_l4073_407368


namespace NUMINAMATH_CALUDE_book_pages_theorem_l4073_407311

/-- Calculates the total number of pages in a book given the number of chapters and pages per chapter -/
def totalPages (chapters : ℕ) (pagesPerChapter : ℕ) : ℕ :=
  chapters * pagesPerChapter

/-- Theorem stating that a book with 31 chapters, each 61 pages long, has 1891 pages in total -/
theorem book_pages_theorem :
  totalPages 31 61 = 1891 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l4073_407311


namespace NUMINAMATH_CALUDE_solution_set_equals_expected_solutions_l4073_407329

/-- The set of solutions to the system of equations -/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | 3 * (x^2 + y^2 + z^2) = 1 ∧ x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3}

/-- The set of expected solutions -/
def ExpectedSolutions : Set (ℝ × ℝ × ℝ) :=
  {(1/3, 1/3, 1/3), (-1/3, -1/3, -1/3), (1/Real.sqrt 3, 0, 0), (0, 1/Real.sqrt 3, 0), (0, 0, 1/Real.sqrt 3)}

/-- Theorem stating that the solution set is equal to the expected solutions -/
theorem solution_set_equals_expected_solutions : SolutionSet = ExpectedSolutions := by
  sorry


end NUMINAMATH_CALUDE_solution_set_equals_expected_solutions_l4073_407329


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4073_407354

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I →
  z = -1 + (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4073_407354


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4073_407351

theorem complex_equation_solution (z : ℂ) 
  (h : 18 * Complex.normSq z = 2 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 2) + 48) : 
  z + 12 / z = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4073_407351


namespace NUMINAMATH_CALUDE_parallelogram_properties_l4073_407307

/-- Represents a parallelogram with specific properties -/
structure Parallelogram where
  /-- Length of the shorter side -/
  side_short : ℝ
  /-- Length of the longer side -/
  side_long : ℝ
  /-- Length of the first diagonal -/
  diag1 : ℝ
  /-- Length of the second diagonal -/
  diag2 : ℝ
  /-- The difference between the lengths of the sides is 7 -/
  side_diff : side_long - side_short = 7
  /-- A perpendicular from a vertex divides a diagonal into segments of 6 and 15 -/
  diag_segments : diag1 = 6 + 15

/-- Theorem stating the properties of the specific parallelogram -/
theorem parallelogram_properties : 
  ∃ (p : Parallelogram), 
    p.side_short = 10 ∧ 
    p.side_long = 17 ∧ 
    p.diag1 = 21 ∧ 
    p.diag2 = Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_properties_l4073_407307


namespace NUMINAMATH_CALUDE_tangent_line_problem_l4073_407372

-- Define the curve
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b

-- Define the tangent line
def tangent_line (x k : ℝ) : ℝ := k*x + 1

-- Define the derivative of the curve
def curve_derivative (x a : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_problem (a b k : ℝ) : 
  curve 1 a b = 2 →
  tangent_line 1 k = 2 →
  curve_derivative 1 a = k →
  b - a = 5 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_problem_l4073_407372


namespace NUMINAMATH_CALUDE_min_value_sum_l4073_407392

/-- Given two circles C₁ and C₂, where C₁ always bisects the circumference of C₂,
    prove that the minimum value of 1/m + 2/n is 3 -/
theorem min_value_sum (m n : ℝ) : m > 0 → n > 0 → 
  (∀ x y : ℝ, (x - m)^2 + (y - 2*n)^2 = m^2 + 4*n^2 + 10 → 
              (x + 1)^2 + (y + 1)^2 = 2 → 
              ∃ k : ℝ, (m + 1)*x + (2*n + 1)*y + 5 = k * ((x + 1)^2 + (y + 1)^2 - 2)) →
  (1 / m + 2 / n) ≥ 3 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1 / m₀ + 2 / n₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l4073_407392


namespace NUMINAMATH_CALUDE_find_number_l4073_407308

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 6) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_find_number_l4073_407308


namespace NUMINAMATH_CALUDE_cylinder_symmetry_properties_l4073_407346

/-- Represents the type of a rotational cylinder -/
inductive CylinderType
  | DoubleSidedBounded
  | SingleSidedBounded
  | DoubleSidedUnbounded

/-- Represents the symmetry properties of a cylinder -/
structure CylinderSymmetry where
  hasAxisSymmetry : Bool
  hasPerpendicularPlaneSymmetry : Bool
  hasBundlePlanesSymmetry : Bool
  hasCenterSymmetry : Bool
  hasInfiniteCentersSymmetry : Bool
  hasTwoSystemsPlanesSymmetry : Bool

/-- Returns the symmetry properties for a given cylinder type -/
def getSymmetryProperties (cType : CylinderType) : CylinderSymmetry :=
  match cType with
  | CylinderType.DoubleSidedBounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := true,
      hasBundlePlanesSymmetry := true,
      hasCenterSymmetry := true,
      hasInfiniteCentersSymmetry := false,
      hasTwoSystemsPlanesSymmetry := false
    }
  | CylinderType.SingleSidedBounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := false,
      hasBundlePlanesSymmetry := true,
      hasCenterSymmetry := false,
      hasInfiniteCentersSymmetry := false,
      hasTwoSystemsPlanesSymmetry := false
    }
  | CylinderType.DoubleSidedUnbounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := false,
      hasBundlePlanesSymmetry := false,
      hasCenterSymmetry := false,
      hasInfiniteCentersSymmetry := true,
      hasTwoSystemsPlanesSymmetry := true
    }

theorem cylinder_symmetry_properties (cType : CylinderType) :
  (getSymmetryProperties cType).hasAxisSymmetry = true ∧
  ((cType = CylinderType.DoubleSidedBounded) → (getSymmetryProperties cType).hasPerpendicularPlaneSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedBounded ∨ cType = CylinderType.SingleSidedBounded) → (getSymmetryProperties cType).hasBundlePlanesSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedBounded) → (getSymmetryProperties cType).hasCenterSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedUnbounded) → (getSymmetryProperties cType).hasInfiniteCentersSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedUnbounded) → (getSymmetryProperties cType).hasTwoSystemsPlanesSymmetry = true) :=
by
  sorry


end NUMINAMATH_CALUDE_cylinder_symmetry_properties_l4073_407346


namespace NUMINAMATH_CALUDE_grape_popsicles_count_l4073_407314

/-- The number of cherry popsicles -/
def cherry_popsicles : ℕ := 13

/-- The number of banana popsicles -/
def banana_popsicles : ℕ := 2

/-- The total number of popsicles -/
def total_popsicles : ℕ := 17

/-- The number of grape popsicles -/
def grape_popsicles : ℕ := total_popsicles - cherry_popsicles - banana_popsicles

theorem grape_popsicles_count : grape_popsicles = 2 := by
  sorry

end NUMINAMATH_CALUDE_grape_popsicles_count_l4073_407314


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l4073_407362

theorem piggy_bank_problem (initial_amount : ℝ) : 
  initial_amount = 204 → 
  (initial_amount * (1 - 0.6) * (1 - 0.5) * (1 - 0.35)) = 26.52 :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l4073_407362


namespace NUMINAMATH_CALUDE_multiplication_division_remainder_problem_l4073_407315

theorem multiplication_division_remainder_problem :
  ∃ (x : ℕ), (55 * x) % 8 = 7 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_remainder_problem_l4073_407315


namespace NUMINAMATH_CALUDE_sufficient_condition_l4073_407309

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The statement that a_4 and a_12 are roots of x^2 + 3x = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 + 3 * a 4 = 0 ∧ a 12 ^ 2 + 3 * a 12 = 0

/-- The theorem stating that the conditions are sufficient for a_8 = ±1 -/
theorem sufficient_condition (a : ℕ → ℝ) :
  geometric_sequence a → roots_condition a → (a 8 = 1 ∨ a 8 = -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_l4073_407309


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_l4073_407343

theorem no_real_solutions_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) ↔ k < -9/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_l4073_407343


namespace NUMINAMATH_CALUDE_negative_two_and_negative_half_are_reciprocals_l4073_407318

-- Define the concept of reciprocals
def are_reciprocals (a b : ℚ) : Prop := a * b = 1

-- Theorem statement
theorem negative_two_and_negative_half_are_reciprocals :
  are_reciprocals (-2) (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_and_negative_half_are_reciprocals_l4073_407318


namespace NUMINAMATH_CALUDE_arrangements_theorem_l4073_407378

/-- The number of people in the row -/
def n : ℕ := 6

/-- The number of different arrangements where both person A and person B
    are on the same side of person C -/
def arrangements_same_side (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem arrangements_theorem :
  arrangements_same_side n = 480 :=
sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l4073_407378


namespace NUMINAMATH_CALUDE_circular_garden_radius_l4073_407334

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 3) * π * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l4073_407334


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l4073_407352

theorem fractional_equation_solution :
  ∃ x : ℝ, (((1 - x) / (2 - x)) - 1 = ((2 * x - 5) / (x - 2))) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l4073_407352


namespace NUMINAMATH_CALUDE_consecutive_squares_theorem_l4073_407361

theorem consecutive_squares_theorem :
  (∀ x : ℤ, ¬∃ y : ℤ, 3 * x^2 + 2 = y^2) ∧
  (∀ x : ℤ, ¬∃ y : ℤ, 6 * x^2 + 6 * x + 19 = y^2) ∧
  (∃ x : ℤ, ∃ y : ℤ, 11 * x^2 + 110 = y^2) ∧
  (∃ y : ℤ, 11 * 23^2 + 110 = y^2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_theorem_l4073_407361


namespace NUMINAMATH_CALUDE_median_condition_implies_right_triangle_l4073_407300

/-- Given a triangle with medians m₁, m₂, and m₃, if m₁² + m₂² = 5m₃², then the triangle is right. -/
theorem median_condition_implies_right_triangle 
  (m₁ m₂ m₃ : ℝ) 
  (h_medians : ∃ (a b c : ℝ), 
    m₁^2 = (2*(b^2 + c^2) - a^2) / 4 ∧ 
    m₂^2 = (2*(a^2 + c^2) - b^2) / 4 ∧ 
    m₃^2 = (2*(a^2 + b^2) - c^2) / 4)
  (h_condition : m₁^2 + m₂^2 = 5 * m₃^2) :
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_median_condition_implies_right_triangle_l4073_407300


namespace NUMINAMATH_CALUDE_two_numbers_difference_l4073_407394

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (triple_minus_four : 3 * y - 4 * x = 14) :
  |y - x| = 9.714 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l4073_407394


namespace NUMINAMATH_CALUDE_rem_one_third_neg_three_fourths_l4073_407336

-- Definition of the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- Theorem statement
theorem rem_one_third_neg_three_fourths :
  rem (1/3) (-3/4) = -5/12 := by sorry

end NUMINAMATH_CALUDE_rem_one_third_neg_three_fourths_l4073_407336


namespace NUMINAMATH_CALUDE_intersection_point_l4073_407357

/-- The line equation is y = -3x + 3 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 3

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line y = -3x + 3 with the x-axis is (1, 0) -/
theorem intersection_point :
  ∃ (x y : ℝ), line_equation x y ∧ on_x_axis x y ∧ x = 1 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l4073_407357


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l4073_407344

theorem greatest_integer_radius (r : ℝ) (h : r > 0) (area_constraint : π * r^2 < 100 * π) :
  ⌊r⌋ ≤ 9 ∧ ∃ (r' : ℝ), π * r'^2 < 100 * π ∧ ⌊r'⌋ = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l4073_407344


namespace NUMINAMATH_CALUDE_punger_pages_needed_l4073_407356

/-- The number of pages needed to hold all baseball cards -/
def pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack + cards_per_page - 1) / cards_per_page

/-- Proof that 42 pages are needed for Punger's baseball cards -/
theorem punger_pages_needed :
  pages_needed 60 7 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_punger_pages_needed_l4073_407356


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l4073_407396

theorem probability_yellow_ball (total_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : total_balls = 8) (h2 : yellow_balls = 5) :
  (yellow_balls : ℚ) / total_balls = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l4073_407396


namespace NUMINAMATH_CALUDE_gcd_1729_1337_l4073_407304

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1337_l4073_407304


namespace NUMINAMATH_CALUDE_expression_evaluation_l4073_407399

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 5 + x * (5 + x) - 5^2
  let denominator := x - 5 + x^2
  numerator / denominator = -26 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4073_407399


namespace NUMINAMATH_CALUDE_composite_number_division_l4073_407325

def first_seven_composite_product : ℕ := 4 * 6 * 8 * 9 * 10 * 12 * 14
def next_eight_composite_product : ℕ := 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25

theorem composite_number_division :
  (first_seven_composite_product : ℚ) / next_eight_composite_product = 1 / 2475 := by
  sorry

end NUMINAMATH_CALUDE_composite_number_division_l4073_407325


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l4073_407371

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, n = 986 ∧ 
  n % 17 = 0 ∧ 
  n ≥ 100 ∧ n < 1000 ∧
  ∀ m : ℕ, m % 17 = 0 → m ≥ 100 → m < 1000 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l4073_407371


namespace NUMINAMATH_CALUDE_james_payment_l4073_407374

/-- The cost of cable program for James and his roommate -/
def cable_cost (first_100_cost : ℕ) (total_channels : ℕ) : ℕ :=
  if total_channels ≤ 100 then
    first_100_cost
  else
    first_100_cost + (first_100_cost / 2)

/-- James' share of the cable cost -/
def james_share (total_cost : ℕ) : ℕ := total_cost / 2

theorem james_payment (first_100_cost : ℕ) (total_channels : ℕ) :
  first_100_cost = 100 →
  total_channels = 200 →
  james_share (cable_cost first_100_cost total_channels) = 75 := by
  sorry

#eval james_share (cable_cost 100 200)

end NUMINAMATH_CALUDE_james_payment_l4073_407374


namespace NUMINAMATH_CALUDE_florist_roses_count_l4073_407341

theorem florist_roses_count (initial : ℕ) (sold : ℕ) (picked : ℕ) : 
  initial = 50 → sold = 15 → picked = 21 → initial - sold + picked = 56 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_count_l4073_407341


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_y_l4073_407323

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 8*b - a*b = 0 ∧ a + b = 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_y_l4073_407323


namespace NUMINAMATH_CALUDE_a_profit_is_25_percent_l4073_407353

/-- Represents the profit percentage as a rational number between 0 and 1 -/
def ProfitPercentage := { x : ℚ // 0 ≤ x ∧ x ≤ 1 }

/-- The bicycle sale scenario -/
structure BicycleSale where
  cost_price_A : ℚ
  selling_price_BC : ℚ
  profit_percentage_B : ProfitPercentage

/-- Calculate the profit percentage of A -/
def profit_percentage_A (sale : BicycleSale) : ProfitPercentage :=
  sorry

/-- Theorem stating that A's profit percentage is 25% given the conditions -/
theorem a_profit_is_25_percent (sale : BicycleSale) 
  (h1 : sale.cost_price_A = 144)
  (h2 : sale.selling_price_BC = 225)
  (h3 : sale.profit_percentage_B = ⟨1/4, by norm_num⟩) :
  profit_percentage_A sale = ⟨1/4, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_a_profit_is_25_percent_l4073_407353


namespace NUMINAMATH_CALUDE_states_fraction_1790_1799_l4073_407386

theorem states_fraction_1790_1799 (total_states : ℕ) (states_1790_1799 : ℕ) :
  total_states = 30 →
  states_1790_1799 = 9 →
  (states_1790_1799 : ℚ) / total_states = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_1790_1799_l4073_407386


namespace NUMINAMATH_CALUDE_school_travel_time_l4073_407328

/-- If a boy reaches school t minutes early when walking at 1.2 times his usual speed,
    his usual time to reach school is 6t minutes. -/
theorem school_travel_time (t : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) 
    (h2 : usual_time > 0) 
    (h3 : usual_speed * usual_time = 1.2 * usual_speed * (usual_time - t)) : 
  usual_time = 6 * t := by
sorry

end NUMINAMATH_CALUDE_school_travel_time_l4073_407328


namespace NUMINAMATH_CALUDE_unique_valid_pair_l4073_407366

def has_one_solution (a b c : ℝ) : Prop :=
  (b^2 - 4*a*c = 0) ∧ (a ≠ 0)

def valid_pair (b c : ℕ+) : Prop :=
  has_one_solution 1 (2*b) (2*c) ∧ has_one_solution 1 (3*c) (3*b)

theorem unique_valid_pair : ∃! p : ℕ+ × ℕ+, valid_pair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_pair_l4073_407366


namespace NUMINAMATH_CALUDE_triangle_properties_l4073_407348

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  -- Given condition
  cos (2*C) - cos (2*A) = 2 * sin (π/3 + C) * sin (π/3 - C) →
  -- Part 1: Prove A = π/3
  A = π/3 ∧
  -- Part 2: Prove range of 2b-c
  (a = sqrt 3 ∧ b ≥ a → 2*b - c ≥ sqrt 3 ∧ 2*b - c < 2 * sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4073_407348


namespace NUMINAMATH_CALUDE_sin_cos_range_l4073_407367

theorem sin_cos_range (x : ℝ) : 29/27 ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧ Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_range_l4073_407367


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4073_407389

theorem geometric_sequence_sum (a r : ℝ) (h1 : a * (1 - r^1000) / (1 - r) = 1024) 
  (h2 : a * (1 - r^2000) / (1 - r) = 2040) : 
  a * (1 - r^3000) / (1 - r) = 3048 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4073_407389


namespace NUMINAMATH_CALUDE_equation_solution_l4073_407335

theorem equation_solution :
  ∃! x : ℚ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + x :=
by
  use -22/3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4073_407335


namespace NUMINAMATH_CALUDE_f_recursion_l4073_407398

/-- A function that computes the sum of binomial coefficients (n choose i) where k divides (n-2i) -/
def f (k : ℕ) (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => if k ∣ (n - 2*i) then Nat.choose n i else 0)

/-- Theorem stating the recursion relation for f_n -/
theorem f_recursion (k : ℕ) (n : ℕ) (h : k > 1) (h_odd : Odd k) :
  (f k n)^2 = (Finset.range (n + 1)).sum (λ i => Nat.choose n i * f k i * f k (n - i)) := by
  sorry

end NUMINAMATH_CALUDE_f_recursion_l4073_407398


namespace NUMINAMATH_CALUDE_product_plus_245_divisible_by_5_l4073_407379

theorem product_plus_245_divisible_by_5 : ∃ k : ℤ, (1250 * 1625 * 1830 * 2075 + 245 : ℤ) = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_product_plus_245_divisible_by_5_l4073_407379


namespace NUMINAMATH_CALUDE_odd_function_property_l4073_407349

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Property of the function f as given in the problem -/
def HasProperty (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem odd_function_property (f : ℝ → ℝ) (h_odd : IsOdd f) (h_prop : HasProperty f) :
  f (-4) > f (-6) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l4073_407349


namespace NUMINAMATH_CALUDE_f_of_3_equals_41_l4073_407326

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

theorem f_of_3_equals_41 : f 3 = 41 := by sorry

end NUMINAMATH_CALUDE_f_of_3_equals_41_l4073_407326


namespace NUMINAMATH_CALUDE_commute_time_difference_l4073_407305

/-- Given a set of 5 numbers {x, y, 10, 11, 9} with a mean of 10 and variance of 2, prove that |x-y| = 4 -/
theorem commute_time_difference (x y : ℝ) 
  (mean_eq : (x + y + 10 + 11 + 9) / 5 = 10) 
  (variance_eq : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l4073_407305


namespace NUMINAMATH_CALUDE_pen_price_before_discount_l4073_407312

-- Define the problem parameters
def num_pens : ℕ := 30
def num_pencils : ℕ := 75
def total_cost : ℚ := 570
def pen_discount : ℚ := 0.1
def pencil_tax : ℚ := 0.05
def avg_pencil_price : ℚ := 2

-- Define the theorem
theorem pen_price_before_discount :
  let pencil_cost := num_pencils * avg_pencil_price
  let pencil_cost_with_tax := pencil_cost * (1 + pencil_tax)
  let pen_cost_with_discount := total_cost - pencil_cost_with_tax
  let pen_cost_before_discount := pen_cost_with_discount / (1 - pen_discount)
  let avg_pen_price := pen_cost_before_discount / num_pens
  ∃ (x : ℚ), abs (x - avg_pen_price) < 0.005 ∧ x = 15.28 :=
by sorry


end NUMINAMATH_CALUDE_pen_price_before_discount_l4073_407312


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l4073_407390

open Set Real

def P : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def Q : Set ℝ := {x | log x < 1}

theorem intersection_P_complement_Q : P ∩ (univ \ Q) = {-3} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l4073_407390


namespace NUMINAMATH_CALUDE_roots_product_plus_one_l4073_407316

theorem roots_product_plus_one (a b : ℝ) : 
  a^2 + 2*a - 2023 = 0 → 
  b^2 + 2*b - 2023 = 0 → 
  (a + 1) * (b + 1) = -2024 := by
sorry

end NUMINAMATH_CALUDE_roots_product_plus_one_l4073_407316


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonic_increase_intervals_l4073_407375

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (x = 1 → a*x + b*y + c = 0)) ∧
  a = 2 ∧ b = -1 ∧ c = 1 := by sorry

-- Theorem for the intervals of monotonic increase
theorem monotonic_increase_intervals :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x : ℝ, (x < -a ∨ x > a) → (∀ y : ℝ, x < y → f x < f y)) ∧
  a = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonic_increase_intervals_l4073_407375


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l4073_407383

/-- Represents the fare structure of a taxi service -/
structure TaxiFare where
  baseFare : ℝ
  mileageRate : ℝ
  minuteRate : ℝ

/-- Calculates the total fare for a taxi trip -/
def calculateFare (fare : TaxiFare) (miles : ℝ) (minutes : ℝ) : ℝ :=
  fare.baseFare + fare.mileageRate * miles + fare.minuteRate * minutes

/-- Theorem: Given the fare structure and initial trip data, 
    a 60-mile trip lasting 90 minutes will cost $200 -/
theorem taxi_fare_calculation 
  (fare : TaxiFare)
  (h1 : fare.baseFare = 20)
  (h2 : fare.minuteRate = 0.5)
  (h3 : calculateFare fare 40 60 = 140) :
  calculateFare fare 60 90 = 200 := by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_calculation_l4073_407383


namespace NUMINAMATH_CALUDE_ratio_comparison_l4073_407332

theorem ratio_comparison (y : ℕ) (h : y > 4) : (3 : ℚ) / 4 < 3 / y :=
sorry

end NUMINAMATH_CALUDE_ratio_comparison_l4073_407332


namespace NUMINAMATH_CALUDE_tree_height_differences_l4073_407319

def pine_height : ℚ := 15 + 1/4
def birch_height : ℚ := 20 + 1/2
def maple_height : ℚ := 18 + 3/4

theorem tree_height_differences :
  (birch_height - pine_height = 5 + 1/4) ∧
  (birch_height - maple_height = 1 + 3/4) := by
  sorry

end NUMINAMATH_CALUDE_tree_height_differences_l4073_407319


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l4073_407320

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 8 →
  r = q + 8 →
  s = r + 8 →
  ones_digit p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l4073_407320
