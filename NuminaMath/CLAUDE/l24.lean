import Mathlib

namespace NUMINAMATH_CALUDE_line_passes_through_I_III_IV_l24_2459

-- Define the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Define the quadrants
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem line_passes_through_I_III_IV :
  (∃ x y : ℝ, y = line x ∧ in_quadrant_I x y) ∧
  (∃ x y : ℝ, y = line x ∧ in_quadrant_III x y) ∧
  (∃ x y : ℝ, y = line x ∧ in_quadrant_IV x y) ∧
  ¬(∃ x y : ℝ, y = line x ∧ in_quadrant_II x y) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_I_III_IV_l24_2459


namespace NUMINAMATH_CALUDE_sum_nonzero_digits_base8_999_l24_2420

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the non-zero elements of a list -/
def sumNonZero (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of non-zero digits in the base 8 representation of 999 is 19 -/
theorem sum_nonzero_digits_base8_999 : sumNonZero (toBase8 999) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_nonzero_digits_base8_999_l24_2420


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l24_2435

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  S : ℕ → ℝ
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem: For an arithmetic sequence where S_17 = 17/2, a_3 + a_15 = 1 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
    (h : seq.S 17 = 17 / 2) : seq.a 3 + seq.a 15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l24_2435


namespace NUMINAMATH_CALUDE_equation_solution_l24_2472

theorem equation_solution : 
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 8*x) + Real.sqrt (x + 8) = 40 - 3*x := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l24_2472


namespace NUMINAMATH_CALUDE_indeterminate_157th_digit_l24_2416

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ :=
  sorry

/-- The nth digit after the decimal point in the decimal representation of q -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem indeterminate_157th_digit :
  ∀ (d : ℕ),
  (∃ (q : ℚ), q = 525 / 2027 ∧ 
    (∀ (i : ℕ), i < 6 → nth_digit_after_decimal q (i + 1) = nth_digit_after_decimal (0.258973 : ℚ) (i + 1))) →
  (∃ (r : ℚ), r ≠ q ∧ 
    (∀ (i : ℕ), i < 6 → nth_digit_after_decimal r (i + 1) = nth_digit_after_decimal (0.258973 : ℚ) (i + 1)) ∧
    nth_digit_after_decimal r 157 ≠ d) :=
by
  sorry


end NUMINAMATH_CALUDE_indeterminate_157th_digit_l24_2416


namespace NUMINAMATH_CALUDE_circle_area_comparison_l24_2497

theorem circle_area_comparison (R : ℝ) (h : R > 0) :
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let total_small_circles_area := 4 * small_circle_area
  large_circle_area = total_small_circles_area := by
sorry

end NUMINAMATH_CALUDE_circle_area_comparison_l24_2497


namespace NUMINAMATH_CALUDE_diagonal_to_larger_base_ratio_l24_2431

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smaller_base : ℝ
  /-- The length of the larger base -/
  larger_base : ℝ
  /-- The length of the diagonal -/
  diagonal : ℝ
  /-- The height of the trapezoid -/
  altitude : ℝ
  /-- The smaller base is positive -/
  smaller_base_pos : 0 < smaller_base
  /-- The larger base is greater than the smaller base -/
  base_order : smaller_base < larger_base
  /-- The smaller base equals half the diagonal -/
  smaller_base_eq_half_diagonal : smaller_base = diagonal / 2
  /-- The altitude equals two-thirds of the smaller base -/
  altitude_eq_two_thirds_smaller_base : altitude = 2 / 3 * smaller_base

/-- The ratio of the diagonal to the larger base in the specific isosceles trapezoid -/
theorem diagonal_to_larger_base_ratio (t : IsoscelesTrapezoid) : 
  t.diagonal / t.larger_base = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_to_larger_base_ratio_l24_2431


namespace NUMINAMATH_CALUDE_min_people_to_complete_task_l24_2449

/-- Proves the minimum number of people needed to complete a task on time -/
theorem min_people_to_complete_task
  (total_days : ℕ)
  (days_worked : ℕ)
  (initial_people : ℕ)
  (work_completed : ℚ)
  (h1 : total_days = 40)
  (h2 : days_worked = 10)
  (h3 : initial_people = 12)
  (h4 : work_completed = 2 / 5)
  (h5 : days_worked < total_days) :
  let remaining_days := total_days - days_worked
  let remaining_work := 1 - work_completed
  let work_rate_per_day := work_completed / days_worked / initial_people
  ⌈(remaining_work / (work_rate_per_day * remaining_days))⌉ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_people_to_complete_task_l24_2449


namespace NUMINAMATH_CALUDE_xiaojun_pen_refills_l24_2458

/-- The number of pen refills Xiaojun bought -/
def num_pen_refills : ℕ := 2

/-- The cost of each pen refill in yuan -/
def pen_refill_cost : ℕ := 2

/-- The cost of each eraser in yuan (positive integer) -/
def eraser_cost : ℕ := 2

/-- The total amount spent in yuan -/
def total_spent : ℕ := 6

/-- The number of erasers Xiaojun bought -/
def num_erasers : ℕ := 1

theorem xiaojun_pen_refills :
  num_pen_refills = 2 ∧
  pen_refill_cost = 2 ∧
  eraser_cost > 0 ∧
  total_spent = 6 ∧
  num_pen_refills = 2 * num_erasers ∧
  total_spent = num_pen_refills * pen_refill_cost + num_erasers * eraser_cost :=
by sorry

#check xiaojun_pen_refills

end NUMINAMATH_CALUDE_xiaojun_pen_refills_l24_2458


namespace NUMINAMATH_CALUDE_charlie_score_l24_2494

theorem charlie_score (team_total : ℕ) (num_players : ℕ) (others_average : ℕ) (h1 : team_total = 60) (h2 : num_players = 8) (h3 : others_average = 5) :
  team_total - (num_players - 1) * others_average = 25 := by
  sorry

end NUMINAMATH_CALUDE_charlie_score_l24_2494


namespace NUMINAMATH_CALUDE_factorization_equality_l24_2498

theorem factorization_equality (x y : ℝ) : y - 2*x*y + x^2*y = y*(1-x)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l24_2498


namespace NUMINAMATH_CALUDE_johns_raise_l24_2492

/-- Proves that if an amount x is increased by 9.090909090909092% to reach $60, then x is equal to $55 -/
theorem johns_raise (x : ℝ) : 
  x * (1 + 0.09090909090909092) = 60 → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_johns_raise_l24_2492


namespace NUMINAMATH_CALUDE_select_five_from_eight_l24_2440

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l24_2440


namespace NUMINAMATH_CALUDE_solution_set_part1_min_value_condition_l24_2414

-- Define the function f
def f (x a b : ℝ) : ℝ := 2 * abs (x + a) + abs (3 * x - b)

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 0 ≥ 3 * abs x + 1} = {x : ℝ | x ≥ -1/2 ∨ x ≤ -3/2} := by sorry

-- Part 2
theorem min_value_condition :
  ∀ a b : ℝ, a > 0 → b > 0 → (∀ x : ℝ, f x a b ≥ 2) → (∃ x : ℝ, f x a b = 2) →
  3 * a + b = 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_min_value_condition_l24_2414


namespace NUMINAMATH_CALUDE_qr_length_l24_2454

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ

/-- Circle with center and two points it passes through -/
structure Circle where
  center : ℝ × ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The problem setup -/
def ProblemSetup (t : Triangle) (c1 c2 : Circle) : Prop :=
  t.DE = 7 ∧ t.EF = 24 ∧ t.DF = 25 ∧
  c1.center.1 = c1.point1.1 ∧ -- Q is on the same vertical line as D
  c1.point2 = (t.DF, 0) ∧ -- F is at (25, 0)
  c2.center.2 = c2.point1.2 ∧ -- R is on the same horizontal line as E
  c2.point2 = (0, 0) -- D is at (0, 0)

theorem qr_length (t : Triangle) (c1 c2 : Circle) 
  (h : ProblemSetup t c1 c2) : 
  ‖c1.center - c2.center‖ = 8075 / 84 := by
  sorry

end NUMINAMATH_CALUDE_qr_length_l24_2454


namespace NUMINAMATH_CALUDE_pages_copied_for_fifty_dollars_l24_2491

/-- Given that 4 pages can be copied for 10 cents, prove that $50 allows for copying 2000 pages. -/
theorem pages_copied_for_fifty_dollars (cost_per_four_pages : ℚ) (pages_per_fifty_dollars : ℕ) :
  cost_per_four_pages = 10 / 100 →
  pages_per_fifty_dollars = 2000 :=
by sorry

end NUMINAMATH_CALUDE_pages_copied_for_fifty_dollars_l24_2491


namespace NUMINAMATH_CALUDE_air_quality_probability_l24_2405

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) 
  (h1 : p_good = 0.8) 
  (h2 : p_consecutive = 0.68) : 
  p_consecutive / p_good = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l24_2405


namespace NUMINAMATH_CALUDE_jake_reading_theorem_l24_2439

def read_pattern (first_day : ℕ) : ℕ → ℕ
  | 1 => first_day
  | 2 => first_day - 20
  | 3 => 2 * (first_day - 20)
  | 4 => first_day / 2
  | _ => 0

def total_pages_read (first_day : ℕ) : ℕ :=
  (read_pattern first_day 1) + (read_pattern first_day 2) + 
  (read_pattern first_day 3) + (read_pattern first_day 4)

theorem jake_reading_theorem (book_chapters book_pages : ℕ) 
  (h1 : book_chapters = 8) (h2 : book_pages = 130) (h3 : read_pattern 37 1 = 37) :
  total_pages_read 37 = 106 := by sorry

end NUMINAMATH_CALUDE_jake_reading_theorem_l24_2439


namespace NUMINAMATH_CALUDE_abs_y_plus_sqrt_y_plus_two_squared_l24_2419

theorem abs_y_plus_sqrt_y_plus_two_squared (y : ℝ) (h : y > 1) :
  |y + Real.sqrt ((y + 2)^2)| = 2*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_y_plus_sqrt_y_plus_two_squared_l24_2419


namespace NUMINAMATH_CALUDE_count_numbers_with_remainder_l24_2474

theorem count_numbers_with_remainder (n : ℕ) : 
  (Finset.filter (fun N => N > 17 ∧ 2017 % N = 17) (Finset.range (2017 + 1))).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_remainder_l24_2474


namespace NUMINAMATH_CALUDE_total_selections_is_57_l24_2469

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement where
  total_circles : Nat
  horizontal_rows : List Nat
  diagonal_length : Nat

/-- Calculates the number of ways to select three consecutive circles in a row -/
def consecutive_selections (row_length : Nat) : Nat :=
  max (row_length - 2) 0

/-- Calculates the total number of ways to select three consecutive circles in the figure -/
def total_selections (arrangement : CircleArrangement) : Nat :=
  let horizontal_selections := arrangement.horizontal_rows.map consecutive_selections |>.sum
  let diagonal_selections := List.range arrangement.diagonal_length |>.map consecutive_selections |>.sum
  horizontal_selections + 2 * diagonal_selections

/-- The main theorem stating that the total number of selections is 57 -/
theorem total_selections_is_57 (arrangement : CircleArrangement) :
  arrangement.total_circles = 33 →
  arrangement.horizontal_rows = [6, 5, 4, 3, 2, 1] →
  arrangement.diagonal_length = 6 →
  total_selections arrangement = 57 := by
  sorry

#eval total_selections { total_circles := 33, horizontal_rows := [6, 5, 4, 3, 2, 1], diagonal_length := 6 }

end NUMINAMATH_CALUDE_total_selections_is_57_l24_2469


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_100_l24_2470

theorem greatest_common_divisor_under_100 : ∃ (n : ℕ), n = 90 ∧ 
  n ∣ 540 ∧ n < 100 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 ∧ m < 100 ∧ m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_100_l24_2470


namespace NUMINAMATH_CALUDE_smallest_number_proof_l24_2421

def smallest_number : ℕ := 3153

theorem smallest_number_proof :
  (∀ n : ℕ, n < smallest_number →
    ¬(((n + 3) % 18 = 0) ∧ ((n + 3) % 25 = 0) ∧ ((n + 3) % 21 = 0))) ∧
  ((smallest_number + 3) % 18 = 0) ∧
  ((smallest_number + 3) % 25 = 0) ∧
  ((smallest_number + 3) % 21 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l24_2421


namespace NUMINAMATH_CALUDE_player_b_always_wins_l24_2438

/-- Represents a player's move in the game -/
structure Move where
  value : ℕ

/-- Represents the state of the game after each round -/
structure GameState where
  round : ℕ
  player_a_move : Move
  player_b_move : Move
  player_a_score : ℕ
  player_b_score : ℕ

/-- The game setup with n rounds and increment d -/
structure GameSetup where
  n : ℕ
  d : ℕ
  h1 : n > 1
  h2 : d ≥ 1

/-- A strategy for player B -/
def PlayerBStrategy (setup : GameSetup) : GameState → Move := sorry

/-- Checks if a move is valid according to the game rules -/
def isValidMove (setup : GameSetup) (prev : GameState) (curr : Move) : Prop := sorry

/-- Calculates the score for a round -/
def calculateScore (a_move : Move) (b_move : Move) : ℕ × ℕ := sorry

/-- Simulates the game for n rounds -/
def playGame (setup : GameSetup) (strategy : GameState → Move) : GameState := sorry

/-- Theorem: Player B always has a winning strategy -/
theorem player_b_always_wins (setup : GameSetup) :
  ∃ (strategy : GameState → Move),
    (playGame setup strategy).player_b_score ≥ (playGame setup strategy).player_a_score := by
  sorry

end NUMINAMATH_CALUDE_player_b_always_wins_l24_2438


namespace NUMINAMATH_CALUDE_sound_speed_in_new_rod_l24_2476

/-- The speed of sound in a new rod given experimental data -/
theorem sound_speed_in_new_rod (a b l : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : l > 0) (h4 : b > a) : ∃ v : ℝ,
  v = 3 * l / (2 * (b - a)) ∧
  (∃ (t1 t2 t3 t4 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t4 > 0 ∧
    t1 + t2 + t3 = a ∧
    t1 = 2 * (t2 + t3) ∧
    t1 + t4 + t3 = b ∧
    t1 + t4 = 2 * t3 ∧
    v = l / t4) :=
by sorry

end NUMINAMATH_CALUDE_sound_speed_in_new_rod_l24_2476


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l24_2447

theorem express_y_in_terms_of_x (x y : ℝ) : 2 * x - y = 4 → y = 2 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l24_2447


namespace NUMINAMATH_CALUDE_return_trip_duration_l24_2418

def time_to_park : ℕ := 20 + 10

def return_trip_factor : ℕ := 3

theorem return_trip_duration : 
  return_trip_factor * time_to_park = 90 :=
by sorry

end NUMINAMATH_CALUDE_return_trip_duration_l24_2418


namespace NUMINAMATH_CALUDE_discontinuous_function_l24_2461

def M (f : ℝ → ℝ) (x : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of (λ i j => if i = j then 1 + f (x i) else f (x j))

theorem discontinuous_function
  (f : ℝ → ℝ)
  (f_nonzero : ∀ x, f x ≠ 0)
  (f_condition : f 2014 = 1 - f 2013)
  (det_zero : ∀ (n : ℕ) (x : Fin n → ℝ), Function.Injective x → Matrix.det (M f x) = 0) :
  ¬Continuous f :=
sorry

end NUMINAMATH_CALUDE_discontinuous_function_l24_2461


namespace NUMINAMATH_CALUDE_jordan_running_time_l24_2451

/-- Given that Jordan ran 3 miles in 1/3 of the time it took Steve to run 4 miles,
    and Steve ran 4 miles in 32 minutes, prove that Jordan would take 224/9 minutes
    to run 7 miles. -/
theorem jordan_running_time (steve_time : ℝ) (jordan_distance : ℝ) :
  steve_time = 32 →
  jordan_distance = 7 →
  (3 / (1/3 * steve_time)) * jordan_distance = 224/9 := by
  sorry

end NUMINAMATH_CALUDE_jordan_running_time_l24_2451


namespace NUMINAMATH_CALUDE_image_fixed_point_l24_2417

variable {S : Type*} [Finite S]

-- Define the set of all functions from S to S
def AllFunctions (S : Type*) := S → S

-- Define the image of a set under a function
def Image (f : S → S) (A : Set S) : Set S := {y | ∃ x ∈ A, f x = y}

-- Main theorem
theorem image_fixed_point
  (f : S → S)
  (h : ∀ g : S → S, g ≠ f → (f ∘ g ∘ f) ≠ (g ∘ f ∘ g)) :
  Image f (Image f (Set.univ : Set S)) = Image f (Set.univ : Set S) :=
sorry

end NUMINAMATH_CALUDE_image_fixed_point_l24_2417


namespace NUMINAMATH_CALUDE_stream_speed_l24_2473

theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 39 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 13 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l24_2473


namespace NUMINAMATH_CALUDE_derivative_problems_l24_2489

open Real

theorem derivative_problems :
  (∀ x : ℝ, x ≠ 0 → deriv (λ x => x * (1 + 2/x + 2/x^2)) x = 1 - 2/x^2) ∧
  (∀ x : ℝ, deriv (λ x => x^4 - 3*x^2 - 5*x + 6) x = 4*x^3 - 6*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_derivative_problems_l24_2489


namespace NUMINAMATH_CALUDE_circle_radius_squared_l24_2481

-- Define the circle and points
variable (r : ℝ) -- radius of the circle
variable (A B C D P : ℝ × ℝ) -- points in 2D plane

-- Define the conditions
def AB : ℝ := 12 -- length of chord AB
def CD : ℝ := 8 -- length of chord CD
def BP : ℝ := 9 -- distance from B to P

-- Define the angle condition
def angle_APD_is_right : Prop := sorry

-- Define that P is outside the circle
def P_outside_circle : Prop := sorry

-- Define that AB and CD extended intersect at P
def chords_intersect_at_P : Prop := sorry

-- Theorem statement
theorem circle_radius_squared 
  (h1 : AB = 12)
  (h2 : CD = 8)
  (h3 : BP = 9)
  (h4 : angle_APD_is_right)
  (h5 : P_outside_circle)
  (h6 : chords_intersect_at_P) :
  r^2 = 97.361 := by sorry

end NUMINAMATH_CALUDE_circle_radius_squared_l24_2481


namespace NUMINAMATH_CALUDE_cattle_milk_production_l24_2484

theorem cattle_milk_production 
  (total_cattle : ℕ) 
  (male_percentage : ℚ) 
  (female_percentage : ℚ) 
  (num_male_cows : ℕ) 
  (avg_milk_per_cow : ℚ) : 
  male_percentage = 2/5 →
  female_percentage = 3/5 →
  num_male_cows = 50 →
  avg_milk_per_cow = 2 →
  (↑num_male_cows : ℚ) / male_percentage = ↑total_cattle →
  (↑total_cattle * female_percentage * avg_milk_per_cow : ℚ) = 150 := by
  sorry

end NUMINAMATH_CALUDE_cattle_milk_production_l24_2484


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l24_2437

theorem largest_four_digit_divisible_by_35 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 35 = 0 → n ≤ 9975 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_35_l24_2437


namespace NUMINAMATH_CALUDE_basketball_handshakes_l24_2457

/-- Number of players in each team -/
def team_size : ℕ := 6

/-- Number of teams -/
def num_teams : ℕ := 2

/-- Number of referees -/
def num_referees : ℕ := 3

/-- Total number of players -/
def total_players : ℕ := team_size * num_teams

/-- Function to calculate the number of handshakes between two teams -/
def inter_team_handshakes : ℕ := team_size * team_size

/-- Function to calculate the number of handshakes within a team -/
def intra_team_handshakes : ℕ := team_size.choose 2

/-- Function to calculate the number of handshakes between players and referees -/
def player_referee_handshakes : ℕ := total_players * num_referees

/-- The total number of handshakes in the basketball game -/
def total_handshakes : ℕ := 
  inter_team_handshakes + 
  (intra_team_handshakes * num_teams) + 
  player_referee_handshakes

theorem basketball_handshakes : total_handshakes = 102 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l24_2457


namespace NUMINAMATH_CALUDE_interest_rate_proof_l24_2495

/-- Proves that the annual interest rate is 5% given the specified conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℕ) (amount : ℝ) :
  principal = 973.913043478261 →
  time = 3 →
  amount = 1120 →
  (amount - principal) / (principal * time) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l24_2495


namespace NUMINAMATH_CALUDE_kirsty_model_purchase_l24_2432

/-- The number of models Kirsty can buy at the new price -/
def new_quantity : ℕ := 27

/-- The initial price of each model in dollars -/
def initial_price : ℚ := 45/100

/-- The new price of each model in dollars -/
def new_price : ℚ := 1/2

/-- The initial number of models Kirsty planned to buy -/
def initial_quantity : ℕ := 30

theorem kirsty_model_purchase :
  initial_quantity * initial_price = new_quantity * new_price :=
sorry


end NUMINAMATH_CALUDE_kirsty_model_purchase_l24_2432


namespace NUMINAMATH_CALUDE_circle_area_ratio_l24_2493

/-- Given three circles S, R, and T, where R's diameter is 20% of S's diameter,
    and T's diameter is 40% of R's diameter, prove that the combined area of
    R and T is 4.64% of the area of S. -/
theorem circle_area_ratio (S R T : ℝ) (hR : R = 0.2 * S) (hT : T = 0.4 * R) :
  (π * (R / 2)^2 + π * (T / 2)^2) / (π * (S / 2)^2) = 0.0464 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l24_2493


namespace NUMINAMATH_CALUDE_power_2017_mod_11_l24_2496

theorem power_2017_mod_11 : 2^2017 % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_2017_mod_11_l24_2496


namespace NUMINAMATH_CALUDE_positive_sum_of_squares_implies_inequality_l24_2445

theorem positive_sum_of_squares_implies_inequality 
  (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x^2 + y^2 + z^2 = 8) : 
  x^8 + y^8 + z^8 > 16 * Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_of_squares_implies_inequality_l24_2445


namespace NUMINAMATH_CALUDE_janet_action_figures_l24_2411

def action_figure_count (initial : ℕ) (sold : ℕ) (bought : ℕ) (brother_factor : ℕ) : ℕ :=
  let remaining := initial - sold
  let after_buying := remaining + bought
  let brother_collection := after_buying * brother_factor
  after_buying + brother_collection

theorem janet_action_figures :
  action_figure_count 10 6 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l24_2411


namespace NUMINAMATH_CALUDE_eight_mile_taxi_ride_cost_l24_2446

/-- Calculates the cost of a taxi ride given the base fare, cost per mile, and total miles traveled. -/
def taxiRideCost (baseFare : ℝ) (costPerMile : ℝ) (miles : ℝ) : ℝ :=
  baseFare + costPerMile * miles

/-- Theorem stating that an 8-mile taxi ride with a $2.00 base fare and $0.30 per mile costs $4.40. -/
theorem eight_mile_taxi_ride_cost :
  taxiRideCost 2.00 0.30 8 = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_eight_mile_taxi_ride_cost_l24_2446


namespace NUMINAMATH_CALUDE_remaining_trip_time_l24_2436

/-- Proves that the time to complete the second half of a 510 km journey at 85 km/h is 3 hours -/
theorem remaining_trip_time (total_distance : ℝ) (speed : ℝ) (h1 : total_distance = 510) (h2 : speed = 85) :
  (total_distance / 2) / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_trip_time_l24_2436


namespace NUMINAMATH_CALUDE_range_of_fraction_l24_2424

theorem range_of_fraction (x y : ℝ) (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) :
  1/8 < x/y ∧ x/y < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l24_2424


namespace NUMINAMATH_CALUDE_miss_grayson_class_size_l24_2415

/-- The number of students in Miss Grayson's class -/
def num_students : ℕ := sorry

/-- The amount raised by the students -/
def amount_raised : ℕ := sorry

/-- The cost of the trip -/
def trip_cost : ℕ := sorry

/-- The remaining fund after the trip -/
def remaining_fund : ℕ := sorry

theorem miss_grayson_class_size :
  (amount_raised = num_students * 5) →
  (trip_cost = num_students * 7) →
  (amount_raised - trip_cost = remaining_fund) →
  (remaining_fund = 10) →
  (num_students = 5) := by sorry

end NUMINAMATH_CALUDE_miss_grayson_class_size_l24_2415


namespace NUMINAMATH_CALUDE_min_sticks_arrangement_l24_2468

theorem min_sticks_arrangement (n : ℕ) : n = 1012 ↔ 
  (n > 1000) ∧
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ m : ℕ, n = 5 * m + 2) ∧
  (∃ p : ℕ, n = 2 * p * (p + 1)) ∧
  (∀ x : ℕ, x > 1000 → 
    ((∃ k : ℕ, x = 3 * k + 1) ∧
     (∃ m : ℕ, x = 5 * m + 2) ∧
     (∃ p : ℕ, x = 2 * p * (p + 1))) → x ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_min_sticks_arrangement_l24_2468


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l24_2428

theorem regular_polygon_sides (n : ℕ) : 
  n > 3 → 
  (n : ℚ) / (n * (n - 3) / 2 : ℚ) = 1/4 → 
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l24_2428


namespace NUMINAMATH_CALUDE_subtraction_result_l24_2463

theorem subtraction_result (x : ℝ) (h : x / 10 = 6) : x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l24_2463


namespace NUMINAMATH_CALUDE_ellipse1_passes_through_points_ellipse2_passes_through_point_ellipse3_passes_through_point_ellipse2_axis_ratio_ellipse3_axis_ratio_l24_2475

-- Define the ellipse equations
def ellipse1 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 3 = 1
def ellipse2 (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1
def ellipse3 (x y : ℝ) : Prop := y^2 / 81 + x^2 / 9 = 1

-- Theorem for the first ellipse
theorem ellipse1_passes_through_points :
  ellipse1 (Real.sqrt 6) 1 ∧ ellipse1 (-Real.sqrt 3) (-Real.sqrt 2) := by sorry

-- Theorems for the second and third ellipses
theorem ellipse2_passes_through_point : ellipse2 3 0 := by sorry
theorem ellipse3_passes_through_point : ellipse3 3 0 := by sorry

theorem ellipse2_axis_ratio :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = 3 * b ∧
  ∀ (x y : ℝ), ellipse2 x y ↔ x^2 / a^2 + y^2 / b^2 = 1 := by sorry

theorem ellipse3_axis_ratio :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = 3 * b ∧
  ∀ (x y : ℝ), ellipse3 x y ↔ y^2 / a^2 + x^2 / b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse1_passes_through_points_ellipse2_passes_through_point_ellipse3_passes_through_point_ellipse2_axis_ratio_ellipse3_axis_ratio_l24_2475


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_sum_10_div_9_l24_2444

theorem no_four_digit_numbers_sum_10_div_9 : 
  ¬∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    (∃ (a b c d : ℕ), n = 1000*a + 100*b + 10*c + d ∧ a + b + c + d = 10) ∧
    n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_sum_10_div_9_l24_2444


namespace NUMINAMATH_CALUDE_negative_subtraction_l24_2430

theorem negative_subtraction (a b : ℤ) : -5 - (-2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_subtraction_l24_2430


namespace NUMINAMATH_CALUDE_yellow_mugs_count_l24_2455

/-- Represents the number of mugs of each color in Hannah's collection --/
structure MugCollection where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  other : ℕ

/-- The conditions of Hannah's mug collection --/
def hannahsMugs : MugCollection → Prop
  | m => m.red + m.blue + m.yellow + m.other = 40 ∧
         m.blue = 3 * m.red ∧
         m.red = m.yellow / 2 ∧
         m.other = 4

theorem yellow_mugs_count (m : MugCollection) (h : hannahsMugs m) : m.yellow = 12 := by
  sorry

#check yellow_mugs_count

end NUMINAMATH_CALUDE_yellow_mugs_count_l24_2455


namespace NUMINAMATH_CALUDE_larger_cuboid_width_l24_2434

theorem larger_cuboid_width
  (small_length small_width small_height : ℝ)
  (large_length large_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : small_length = 5)
  (h2 : small_width = 4)
  (h3 : small_height = 3)
  (h4 : large_length = 16)
  (h5 : large_height = 12)
  (h6 : num_small_cuboids = 32)
  (h7 : small_length * small_width * small_height * num_small_cuboids = large_length * large_height * (large_length * large_height / (small_length * small_width * small_height * num_small_cuboids))) :
  large_length * large_height / (small_length * small_width * small_height * num_small_cuboids) = 10 := by
sorry

end NUMINAMATH_CALUDE_larger_cuboid_width_l24_2434


namespace NUMINAMATH_CALUDE_corrected_mean_l24_2485

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 40 →
  incorrect_value = 15 →
  correct_value = 45 →
  let total_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := total_sum + difference
  corrected_sum / n = 40.6 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l24_2485


namespace NUMINAMATH_CALUDE_davids_remaining_money_l24_2448

theorem davids_remaining_money (initial : ℝ) (remaining : ℝ) (spent : ℝ) : 
  initial = 1500 →
  remaining + spent = initial →
  remaining < spent →
  remaining < 750 := by
sorry

end NUMINAMATH_CALUDE_davids_remaining_money_l24_2448


namespace NUMINAMATH_CALUDE_business_value_calculation_l24_2407

/-- Calculates the total value of a business given partial ownership and sale information. -/
theorem business_value_calculation (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  total_ownership = 2 / 3 →
  sold_fraction = 3 / 4 →
  sale_price = 30000 →
  (total_ownership * sold_fraction * sale_price) / (total_ownership * sold_fraction) = 60000 := by
  sorry

end NUMINAMATH_CALUDE_business_value_calculation_l24_2407


namespace NUMINAMATH_CALUDE_range_of_m_l24_2403

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : x + 2*y = x*y) 
  (h_ineq : ∀ m : ℝ, m^2 + 2*m < x + 2*y) : 
  m ∈ Set.Ioo (-2 : ℝ) 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l24_2403


namespace NUMINAMATH_CALUDE_triangle_properties_l24_2409

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  (B = π / 4 → Real.sqrt 3 * b = Real.sqrt 2 * a) ∧
  (a = Real.sqrt 3 ∧ b + c = 3 → b * c = 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l24_2409


namespace NUMINAMATH_CALUDE_magic_triangle_max_sum_l24_2402

theorem magic_triangle_max_sum (a b c d e f : ℕ) : 
  a ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  b ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  c ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  d ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  e ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  f ∈ ({13, 14, 15, 16, 17, 18} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f →
  a + b + c = c + d + e ∧ c + d + e = e + f + a →
  a + b + c ≤ 48 := by
sorry

end NUMINAMATH_CALUDE_magic_triangle_max_sum_l24_2402


namespace NUMINAMATH_CALUDE_multiple_properties_l24_2487

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a + b = 4 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l24_2487


namespace NUMINAMATH_CALUDE_square_perimeter_l24_2429

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 588 → perimeter = 56 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l24_2429


namespace NUMINAMATH_CALUDE_probability_two_tails_two_heads_probability_two_tails_two_heads_proof_l24_2400

/-- The probability of getting exactly two tails and two heads when tossing four coins -/
theorem probability_two_tails_two_heads : ℚ :=
  let p : ℚ := 1 / 2  -- probability of getting heads (or tails) on a single coin toss
  let n : ℕ := 4  -- number of coins
  let k : ℕ := 2  -- number of successes (heads or tails) we're looking for
  3 / 8

theorem probability_two_tails_two_heads_proof :
  probability_two_tails_two_heads = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_tails_two_heads_probability_two_tails_two_heads_proof_l24_2400


namespace NUMINAMATH_CALUDE_g_at_5_l24_2464

/-- A function g satisfying the given equation for all real x -/
def g : ℝ → ℝ := sorry

/-- The main property of function g -/
axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1

/-- The theorem to be proved -/
theorem g_at_5 : g 5 = 3/4 := by sorry

end NUMINAMATH_CALUDE_g_at_5_l24_2464


namespace NUMINAMATH_CALUDE_birthday_celebration_attendance_l24_2404

theorem birthday_celebration_attendance 
  (total_guests : ℕ) 
  (women_percentage men_percentage : ℚ) 
  (men_left_fraction women_left_fraction : ℚ) 
  (children_left : ℕ) 
  (h1 : total_guests = 750)
  (h2 : women_percentage = 432 / 1000)
  (h3 : men_percentage = 314 / 1000)
  (h4 : men_left_fraction = 5 / 12)
  (h5 : women_left_fraction = 7 / 15)
  (h6 : children_left = 19) :
  ∃ (women_count men_count children_count : ℕ),
    women_count + men_count + children_count = total_guests ∧
    women_count = ⌊women_percentage * total_guests⌋ ∧
    men_count = ⌈men_percentage * total_guests⌉ ∧
    children_count = total_guests - women_count - men_count ∧
    total_guests - 
      (⌊men_left_fraction * men_count⌋ + 
       ⌊women_left_fraction * women_count⌋ + 
       children_left) = 482 := by
  sorry


end NUMINAMATH_CALUDE_birthday_celebration_attendance_l24_2404


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_unit_interval_l24_2425

-- Define set A
def A : Set ℝ := {x | x + 1 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (1 - 2^x)}

-- Theorem stating that the intersection of A and B is [0, 1)
theorem A_intersect_B_eq_unit_interval :
  A ∩ B = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_unit_interval_l24_2425


namespace NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l24_2471

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 12)
  (h4 : both = 7) :
  total - (coffee + tea - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_neither_coffee_nor_tea_l24_2471


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l24_2453

theorem sum_of_roots_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2004*x + 2021
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 2004) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l24_2453


namespace NUMINAMATH_CALUDE_ratio_problem_l24_2427

theorem ratio_problem (a b x m : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  a / b = 4 / 5 ∧
  x = a * (1 + 0.25) ∧
  m = b * (1 - 0.80) →
  m / x = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l24_2427


namespace NUMINAMATH_CALUDE_det_A_positive_iff_x_gt_one_l24_2443

/-- Definition of a 2x2 matrix A with elements dependent on x -/
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 3 - x; 1, x]

/-- Definition of determinant for 2x2 matrix -/
def det2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

/-- Theorem stating that det(A) > 0 iff x > 1 -/
theorem det_A_positive_iff_x_gt_one :
  ∀ x : ℝ, det2x2 (A x) > 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_det_A_positive_iff_x_gt_one_l24_2443


namespace NUMINAMATH_CALUDE_cube_inequality_l24_2477

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l24_2477


namespace NUMINAMATH_CALUDE_periodicity_2pi_l24_2460

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y

/-- The periodicity theorem -/
theorem periodicity_2pi (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

end NUMINAMATH_CALUDE_periodicity_2pi_l24_2460


namespace NUMINAMATH_CALUDE_evaluate_expression_l24_2499

theorem evaluate_expression (b c : ℕ) (hb : b = 2) (hc : c = 5) :
  b^3 * b^4 * c^2 = 3200 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l24_2499


namespace NUMINAMATH_CALUDE_intersecting_lines_y_intercept_sum_l24_2452

/-- Given two lines that intersect at a specific point, prove their y-intercepts sum to zero -/
theorem intersecting_lines_y_intercept_sum (a b : ℝ) : 
  (3 = (1/3) * (-3) + a) ∧ (-3 = (1/3) * 3 + b) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_y_intercept_sum_l24_2452


namespace NUMINAMATH_CALUDE_f_properties_l24_2412

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (a / (a^2 - 1)) * (a^x - a^(-x))

theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- 1. f is an increasing odd function on ℝ
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  -- 2. For x ∈ (-1, 1), f(1-m) + f(1-m^2) < 0 implies m ∈ (1, √2)
  (∀ m : ℝ, (∀ x : ℝ, -1 < x ∧ x < 1 → f a (1-m) + f a (1-m^2) < 0) → 
    1 < m ∧ m < Real.sqrt 2) ∧
  -- 3. For x ∈ (-∞, 2), f(x) - 4 < 0 implies a ∈ (2 - √3, 2 + √3) \ {1}
  ((∀ x : ℝ, x < 2 → f a x - 4 < 0) → 
    2 - Real.sqrt 3 < a ∧ a < 2 + Real.sqrt 3 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l24_2412


namespace NUMINAMATH_CALUDE_abs_neg_five_equals_five_l24_2480

theorem abs_neg_five_equals_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_equals_five_l24_2480


namespace NUMINAMATH_CALUDE_total_squares_on_grid_l24_2426

/-- Represents a point on the 5x5 grid -/
structure GridPoint where
  x : Fin 5
  y : Fin 5

/-- Represents the set of 20 nails on the grid -/
def NailSet : Set GridPoint :=
  sorry

/-- Determines if four points form a square -/
def isSquare (p1 p2 p3 p4 : GridPoint) : Prop :=
  sorry

/-- Counts the number of squares that can be formed using the nails -/
def countSquares (nails : Set GridPoint) : Nat :=
  sorry

theorem total_squares_on_grid :
  countSquares NailSet = 21 :=
sorry

end NUMINAMATH_CALUDE_total_squares_on_grid_l24_2426


namespace NUMINAMATH_CALUDE_orc_sword_weight_l24_2410

/-- Given a total weight of swords, number of squads, and orcs per squad,
    calculates the weight each orc must carry. -/
def weight_per_orc (total_weight : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) : ℚ :=
  total_weight / (num_squads * orcs_per_squad)

/-- Proves that given 1200 pounds of swords, 10 squads, and 8 orcs per squad,
    each orc must carry 15 pounds of swords. -/
theorem orc_sword_weight :
  weight_per_orc 1200 10 8 = 15 := by
  sorry

#eval weight_per_orc 1200 10 8

end NUMINAMATH_CALUDE_orc_sword_weight_l24_2410


namespace NUMINAMATH_CALUDE_mike_buys_36_games_l24_2483

/-- Represents the number of days Mike worked --/
def total_days : ℕ := 20

/-- Represents the earnings per lawn in dollars --/
def earnings_per_lawn : ℕ := 5

/-- Represents the number of lawns mowed on a weekday --/
def lawns_per_weekday : ℕ := 2

/-- Represents the number of lawns mowed on a weekend day --/
def lawns_per_weekend : ℕ := 3

/-- Represents the cost of new mower blades in dollars --/
def cost_of_blades : ℕ := 24

/-- Represents the cost of gasoline in dollars --/
def cost_of_gas : ℕ := 15

/-- Represents the cost of each game in dollars --/
def cost_per_game : ℕ := 5

/-- Calculates the number of games Mike can buy --/
def games_mike_can_buy : ℕ :=
  let weekdays := 16
  let weekend_days := 4
  let total_lawns := weekdays * lawns_per_weekday + weekend_days * lawns_per_weekend
  let total_earnings := total_lawns * earnings_per_lawn
  let total_expenses := cost_of_blades + cost_of_gas
  let money_left := total_earnings - total_expenses
  money_left / cost_per_game

/-- Theorem stating that Mike can buy 36 games --/
theorem mike_buys_36_games : games_mike_can_buy = 36 := by
  sorry

end NUMINAMATH_CALUDE_mike_buys_36_games_l24_2483


namespace NUMINAMATH_CALUDE_exponent_multiplication_l24_2466

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l24_2466


namespace NUMINAMATH_CALUDE_floor_sqrt_17_squared_l24_2488

theorem floor_sqrt_17_squared : ⌊Real.sqrt 17⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_17_squared_l24_2488


namespace NUMINAMATH_CALUDE_point_B_coordinates_l24_2465

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem point_B_coordinates :
  let A : Point2D := ⟨2, -3⟩
  let AB_length : ℝ := 4
  let B_parallel_to_x_axis : ℝ → Prop := λ y => y = A.y
  ∃ (B : Point2D), (B.x = -2 ∨ B.x = 6) ∧ 
                   B_parallel_to_x_axis B.y ∧ 
                   ((B.x - A.x)^2 + (B.y - A.y)^2 = AB_length^2) :=
by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l24_2465


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_B_to_B_l24_2479

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis (x y : ℝ) :
  Real.sqrt ((x - x)^2 + ((-y) - y)^2) = 2 * abs y := by sorry

/-- The specific case for point B(1, 4) --/
theorem distance_B_to_B'_is_8 :
  Real.sqrt ((1 - 1)^2 + ((-4) - 4)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_B_to_B_l24_2479


namespace NUMINAMATH_CALUDE_lines_intersection_l24_2423

theorem lines_intersection (k : ℝ) : 
  ∃ (x y : ℝ), ∀ (k : ℝ), k * x + y + 3 * k + 1 = 0 ∧ x = -3 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_intersection_l24_2423


namespace NUMINAMATH_CALUDE_sum_of_consecutive_even_integers_l24_2486

/-- Three consecutive even integers where the sum of the first and third is 128 -/
structure ConsecutiveEvenIntegers where
  a : ℤ
  b : ℤ
  c : ℤ
  consecutive : b = a + 2 ∧ c = b + 2
  even : Even a
  sum_first_third : a + c = 128

/-- The sum of three consecutive even integers is 192 when the sum of the first and third is 128 -/
theorem sum_of_consecutive_even_integers (x : ConsecutiveEvenIntegers) : x.a + x.b + x.c = 192 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_even_integers_l24_2486


namespace NUMINAMATH_CALUDE_positive_real_inequality_l24_2450

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / (x + 2*y + 3*z)) + (y / (y + 2*z + 3*x)) + (z / (z + 2*x + 3*y)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l24_2450


namespace NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l24_2456

theorem limit_sequence_equals_one_over_e :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((10 * n - 3) / (10 * n - 1)) ^ (5 * n) - (1 / Real.exp 1)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l24_2456


namespace NUMINAMATH_CALUDE_scooter_depreciation_l24_2490

theorem scooter_depreciation (initial_value : ℝ) : 
  (initial_value * (3/4)^5 = 9492.1875) → initial_value = 40000 := by
  sorry

end NUMINAMATH_CALUDE_scooter_depreciation_l24_2490


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l24_2401

/-- The focal length of the hyperbola y²/9 - x²/7 = 1 is 8 -/
theorem hyperbola_focal_length : ∃ (a b c : ℝ), 
  (a^2 = 9 ∧ b^2 = 7) → 
  (∀ (x y : ℝ), y^2 / 9 - x^2 / 7 = 1 → (x / a)^2 - (y / b)^2 = 1) →
  c^2 = a^2 + b^2 →
  2 * c = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l24_2401


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_l24_2442

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (n_boys n_girls : ℕ) : 
  let n_total := n_boys + n_girls
  let n_pairs := n_total - 1
  let p_boy := n_boys / n_total
  let p_girl := n_girls / n_total
  let p_adjacent := p_boy * p_girl + p_girl * p_boy
  n_boys = 10 → n_girls = 15 → n_pairs * p_adjacent = 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_l24_2442


namespace NUMINAMATH_CALUDE_inequality_proof_l24_2406

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l24_2406


namespace NUMINAMATH_CALUDE_max_value_implies_a_l24_2441

def f (x : ℝ) := x^2 - 2*x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 2), f x ≤ 4) ∧
  (∃ x ∈ Set.Icc a (a + 2), f x = 4) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l24_2441


namespace NUMINAMATH_CALUDE_island_population_l24_2478

theorem island_population (centipedes humans sheep : ℕ) : 
  centipedes = 100 →
  centipedes = 2 * humans →
  sheep = humans / 2 →
  sheep + humans = 75 := by
sorry

end NUMINAMATH_CALUDE_island_population_l24_2478


namespace NUMINAMATH_CALUDE_cookies_left_for_neil_l24_2467

def total_cookies : ℕ := 20
def fraction_given : ℚ := 2/5

theorem cookies_left_for_neil :
  total_cookies - (total_cookies * fraction_given).floor = 12 :=
by sorry

end NUMINAMATH_CALUDE_cookies_left_for_neil_l24_2467


namespace NUMINAMATH_CALUDE_units_digit_of_power_difference_l24_2413

theorem units_digit_of_power_difference : ∃ n : ℕ, (5^2019 - 3^2019) % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_power_difference_l24_2413


namespace NUMINAMATH_CALUDE_snow_volume_calculation_l24_2422

/-- Calculates the total volume of snow on a sidewalk with two layers -/
theorem snow_volume_calculation 
  (length : ℝ) 
  (width : ℝ) 
  (depth1 : ℝ) 
  (depth2 : ℝ) 
  (h1 : length = 25) 
  (h2 : width = 3) 
  (h3 : depth1 = 1/3) 
  (h4 : depth2 = 1/4) : 
  length * width * depth1 + length * width * depth2 = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_snow_volume_calculation_l24_2422


namespace NUMINAMATH_CALUDE_square_product_theorem_l24_2433

class FiniteSquareRing (R : Type) extends Ring R where
  finite : Finite R
  square_sum_is_square : ∀ a b : R, ∃ c : R, a ^ 2 + b ^ 2 = c ^ 2

theorem square_product_theorem {R : Type} [FiniteSquareRing R] :
  ∀ a b c : R, ∃ d : R, 2 * a * b * c = d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_product_theorem_l24_2433


namespace NUMINAMATH_CALUDE_solution_x_value_l24_2462

theorem solution_x_value (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 80) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_value_l24_2462


namespace NUMINAMATH_CALUDE_fraction_inequality_l24_2408

theorem fraction_inequality (a b t : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : t > 0) :
  a / b > (a + t) / (b + t) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l24_2408


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l24_2482

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A = 2 * B) (h3 : A = 3 * C) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l24_2482
