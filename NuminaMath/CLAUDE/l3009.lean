import Mathlib

namespace NUMINAMATH_CALUDE_P_less_than_Q_l3009_300911

theorem P_less_than_Q (a : ℝ) (ha : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 7) < Real.sqrt (a + 3) + Real.sqrt (a + 4) :=
by sorry

end NUMINAMATH_CALUDE_P_less_than_Q_l3009_300911


namespace NUMINAMATH_CALUDE_cookie_distribution_l3009_300915

theorem cookie_distribution (total : ℚ) (blue green red : ℚ) : 
  blue + green + red = total →
  blue + green = 2/3 * total →
  blue = 1/4 * total →
  green / (blue + green) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3009_300915


namespace NUMINAMATH_CALUDE_wedge_volume_l3009_300922

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h r : ℝ) (h1 : d = 20) (h2 : h = d) (h3 : r = d / 2) : 
  ∃ (m : ℕ), (1 / 3) * π * r^2 * h = m * π ∧ m = 667 := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l3009_300922


namespace NUMINAMATH_CALUDE_johns_gym_time_l3009_300950

/-- Represents the number of times John goes to the gym per week -/
def gym_visits_per_week : ℕ := 3

/-- Represents the number of hours John spends weightlifting each gym visit -/
def weightlifting_hours : ℚ := 1

/-- Represents the fraction of weightlifting time spent on warming up and cardio -/
def warmup_cardio_fraction : ℚ := 1 / 3

/-- Calculates the total hours John spends at the gym per week -/
def total_gym_hours : ℚ :=
  gym_visits_per_week * (weightlifting_hours + warmup_cardio_fraction * weightlifting_hours)

theorem johns_gym_time : total_gym_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_gym_time_l3009_300950


namespace NUMINAMATH_CALUDE_billy_tickets_l3009_300941

theorem billy_tickets (tickets_won : ℕ) (tickets_left : ℕ) (difference : ℕ) : 
  tickets_left = 32 →
  difference = 16 →
  tickets_won - tickets_left = difference →
  tickets_won = 48 := by
sorry

end NUMINAMATH_CALUDE_billy_tickets_l3009_300941


namespace NUMINAMATH_CALUDE_car_gasoline_usage_l3009_300986

/-- Calculates the amount of gasoline used by a car given its efficiency, speed, and travel time. -/
def gasoline_used (efficiency : Real) (speed : Real) (time : Real) : Real :=
  efficiency * speed * time

theorem car_gasoline_usage :
  let efficiency : Real := 0.14  -- liters per kilometer
  let speed : Real := 93.6       -- kilometers per hour
  let time : Real := 2.5         -- hours
  gasoline_used efficiency speed time = 32.76 := by
  sorry

end NUMINAMATH_CALUDE_car_gasoline_usage_l3009_300986


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l3009_300971

theorem choose_three_from_nine : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l3009_300971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3009_300992

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 1 + a 3 = 2) : 
  a 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3009_300992


namespace NUMINAMATH_CALUDE_max_xy_value_l3009_300970

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l3009_300970


namespace NUMINAMATH_CALUDE_median_and_mode_are_23_l3009_300995

/-- Represents a shoe size distribution --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of a shoe size distribution --/
def median (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- The given shoe size distribution --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40 }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 := by
  sorry

end NUMINAMATH_CALUDE_median_and_mode_are_23_l3009_300995


namespace NUMINAMATH_CALUDE_simplify_expression_l3009_300960

theorem simplify_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 252 / Real.sqrt 108) + (Real.sqrt 88 / Real.sqrt 22) = (21 + 2 * Real.sqrt 21) / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3009_300960


namespace NUMINAMATH_CALUDE_inequality_solution_f_above_g_l3009_300923

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Theorem for the solution of the inequality
theorem inequality_solution (a : ℝ) :
  {x : ℝ | f x + a - 1 > 0} = {x : ℝ | x < a + 1 ∨ x > 3 - a} :=
sorry

-- Theorem for the condition of f being above g
theorem f_above_g :
  ∀ m, (∀ x, f x > g x m) ↔ m < 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_f_above_g_l3009_300923


namespace NUMINAMATH_CALUDE_chord_length_theorem_l3009_300933

/-- The chord length theorem -/
theorem chord_length_theorem (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), x - y + m = 0 ∧ (x - 1)^2 + (y - 1)^2 = 3) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ - y₁ + m = 0 ∧ (x₁ - 1)^2 + (y₁ - 1)^2 = 3 ∧
    x₂ - y₂ + m = 0 ∧ (x₂ - 1)^2 + (y₂ - 1)^2 = 3 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = m^2) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l3009_300933


namespace NUMINAMATH_CALUDE_prob_white_given_red_is_two_ninths_l3009_300905

/-- The number of red balls in the box -/
def num_red : ℕ := 3

/-- The number of white balls in the box -/
def num_white : ℕ := 2

/-- The number of black balls in the box -/
def num_black : ℕ := 5

/-- The total number of balls in the box -/
def total_balls : ℕ := num_red + num_white + num_black

/-- The probability of picking a white ball on the second draw given that the first ball picked is red -/
def prob_white_given_red : ℚ := num_white / (total_balls - 1)

theorem prob_white_given_red_is_two_ninths :
  prob_white_given_red = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_given_red_is_two_ninths_l3009_300905


namespace NUMINAMATH_CALUDE_train_crossing_time_l3009_300909

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 375 →
  train_speed_kmh = 90 →
  crossing_time = 15 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3009_300909


namespace NUMINAMATH_CALUDE_probability_even_sum_four_primes_l3009_300969

-- Define the set of first twelve prime numbers
def first_twelve_primes : Finset Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

-- Define a function to check if a number is even
def is_even (n : Nat) : Bool := n % 2 = 0

-- Define a function to calculate the sum of a list of numbers
def sum_list (l : List Nat) : Nat := l.foldl (·+·) 0

-- Theorem statement
theorem probability_even_sum_four_primes :
  let all_selections := Finset.powerset first_twelve_primes
  let valid_selections := all_selections.filter (fun s => s.card = 4)
  let even_sum_selections := valid_selections.filter (fun s => is_even (sum_list s.toList))
  (even_sum_selections.card : Rat) / valid_selections.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_four_primes_l3009_300969


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3009_300964

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3009_300964


namespace NUMINAMATH_CALUDE_last_remaining_number_l3009_300928

/-- Represents the process of skipping and marking numbers -/
def josephus_process (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for n = 50, the last remaining number is 49 -/
theorem last_remaining_number : josephus_process 50 = 49 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_number_l3009_300928


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3009_300931

theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → (k = 3 ∨ k = -3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3009_300931


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l3009_300973

theorem existence_of_special_numbers : ∃ (a : Fin 15 → ℕ),
  (∀ i, ∃ k, a i = 35 * k) ∧
  (∀ i j, i ≠ j → ¬(a i ∣ a j)) ∧
  (∀ i j, (a i)^6 ∣ (a j)^5) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l3009_300973


namespace NUMINAMATH_CALUDE_number_of_observations_l3009_300974

theorem number_of_observations (initial_mean : ℝ) (wrong_obs : ℝ) (correct_obs : ℝ) (new_mean : ℝ)
  (h1 : initial_mean = 36)
  (h2 : wrong_obs = 23)
  (h3 : correct_obs = 45)
  (h4 : new_mean = 36.5) :
  ∃ (n : ℕ), n * initial_mean + (correct_obs - wrong_obs) = n * new_mean ∧ n = 44 := by
sorry

end NUMINAMATH_CALUDE_number_of_observations_l3009_300974


namespace NUMINAMATH_CALUDE_wedding_attendance_percentage_l3009_300980

def expected_attendees : ℕ := 220
def actual_attendees : ℕ := 209

theorem wedding_attendance_percentage :
  (expected_attendees - actual_attendees : ℚ) / expected_attendees * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_wedding_attendance_percentage_l3009_300980


namespace NUMINAMATH_CALUDE_dinner_total_cost_l3009_300979

def food_cost : ℝ := 30
def sales_tax_rate : ℝ := 0.095
def tip_rate : ℝ := 0.10

theorem dinner_total_cost : 
  food_cost + (food_cost * sales_tax_rate) + (food_cost * tip_rate) = 35.85 := by
  sorry

end NUMINAMATH_CALUDE_dinner_total_cost_l3009_300979


namespace NUMINAMATH_CALUDE_existence_of_sequence_l3009_300967

theorem existence_of_sequence (p q : ℝ) (y : Fin 2017 → ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q = 1) :
  ∃ x : Fin 2017 → ℝ, ∀ i : Fin 2017,
    p * max (x i) (x (i.succ)) + q * min (x i) (x (i.succ)) = y i :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l3009_300967


namespace NUMINAMATH_CALUDE_sequence_not_ap_or_gp_l3009_300935

-- Define the sequence
def a : ℕ → ℕ
  | n => if n % 2 = 0 then ((n / 2) + 1)^2 else (n / 2 + 1) * (n / 2 + 2)

-- State the theorem
theorem sequence_not_ap_or_gp :
  -- The sequence is increasing
  (∀ n : ℕ, a n < a (n + 1)) ∧
  -- Each even-indexed term is the arithmetic mean of its neighbors
  (∀ n : ℕ, a (2 * n) = (a (2 * n - 1) + a (2 * n + 1)) / 2) ∧
  -- Each odd-indexed term is the geometric mean of its neighbors
  (∀ n : ℕ, n > 0 → a (2 * n - 1) = Int.sqrt (a (2 * n - 2) * a (2 * n))) ∧
  -- The sequence never becomes an arithmetic progression
  (∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ a (n + 2) - a (n + 1) ≠ a (n + 1) - a n) ∧
  -- The sequence never becomes a geometric progression
  (∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ a (n + 2) * a n ≠ (a (n + 1))^2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_ap_or_gp_l3009_300935


namespace NUMINAMATH_CALUDE_total_turnips_is_105_l3009_300961

/-- The number of turnips Keith grows per day -/
def keith_turnips_per_day : ℕ := 6

/-- The number of days Keith grows turnips -/
def keith_days : ℕ := 7

/-- The number of turnips Alyssa grows every two days -/
def alyssa_turnips_per_two_days : ℕ := 9

/-- The number of days Alyssa grows turnips -/
def alyssa_days : ℕ := 14

/-- The total number of turnips grown by Keith and Alyssa -/
def total_turnips : ℕ :=
  keith_turnips_per_day * keith_days +
  (alyssa_turnips_per_two_days * (alyssa_days / 2))

theorem total_turnips_is_105 : total_turnips = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_is_105_l3009_300961


namespace NUMINAMATH_CALUDE_g_zero_at_neg_three_iff_s_eq_neg_192_l3009_300914

/-- The function g(x) defined in the problem -/
def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

/-- Theorem stating that g(-3) = 0 if and only if s = -192 -/
theorem g_zero_at_neg_three_iff_s_eq_neg_192 :
  ∀ s : ℝ, g (-3) s = 0 ↔ s = -192 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_neg_three_iff_s_eq_neg_192_l3009_300914


namespace NUMINAMATH_CALUDE_special_factors_count_l3009_300956

/-- A function that returns the number of positive factors of 60 that are multiples of 5 but not multiples of 3 -/
def count_special_factors : ℕ :=
  (Finset.filter (fun n => 60 % n = 0 ∧ n % 5 = 0 ∧ n % 3 ≠ 0) (Finset.range 61)).card

/-- Theorem stating that the number of positive factors of 60 that are multiples of 5 but not multiples of 3 is 2 -/
theorem special_factors_count : count_special_factors = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_factors_count_l3009_300956


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3009_300957

def M : Set ℝ := {x | x^2 - 6*x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3009_300957


namespace NUMINAMATH_CALUDE_distinct_values_count_l3009_300903

-- Define a type for expressions
inductive Expr
  | Num : ℕ → Expr
  | Pow : Expr → Expr → Expr

-- Define a function to evaluate expressions
def eval : Expr → ℕ
  | Expr.Num n => n
  | Expr.Pow a b => (eval a) ^ (eval b)

-- Define the base expression
def baseExpr : Expr :=
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3)))

-- Define all possible parenthesizations
def parenthesizations : List Expr := [
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3))),
  Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Num 3)),
  Expr.Pow (Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Num 3)) (Expr.Num 3),
  Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Pow (Expr.Num 3) (Expr.Num 3))) (Expr.Num 3),
  Expr.Pow (Expr.Pow (Expr.Num 3) (Expr.Num 3)) (Expr.Pow (Expr.Num 3) (Expr.Num 3))
]

-- Theorem: The number of distinct values is 3
theorem distinct_values_count :
  (parenthesizations.map eval).toFinset.card = 3 := by sorry

end NUMINAMATH_CALUDE_distinct_values_count_l3009_300903


namespace NUMINAMATH_CALUDE_second_candidate_votes_l3009_300981

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) : 
  total_votes = 600 → 
  first_candidate_percentage = 60 / 100 → 
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 240 := by
  sorry

end NUMINAMATH_CALUDE_second_candidate_votes_l3009_300981


namespace NUMINAMATH_CALUDE_jennie_drive_time_l3009_300972

def drive_time_proof (distance : ℝ) (time_with_traffic : ℝ) (speed_difference : ℝ) : Prop :=
  let speed_with_traffic := distance / time_with_traffic
  let speed_no_traffic := speed_with_traffic + speed_difference
  let time_no_traffic := distance / speed_no_traffic
  distance = 200 ∧ time_with_traffic = 5 ∧ speed_difference = 10 →
  time_no_traffic = 4

theorem jennie_drive_time : drive_time_proof 200 5 10 := by
  sorry

end NUMINAMATH_CALUDE_jennie_drive_time_l3009_300972


namespace NUMINAMATH_CALUDE_not_all_negative_l3009_300953

theorem not_all_negative (a b c d : ℝ) 
  (sum_three_geq_fourth : 
    (a + b + c ≥ d) ∧ 
    (a + b + d ≥ c) ∧ 
    (a + c + d ≥ b) ∧ 
    (b + c + d ≥ a)) : 
  ¬(a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) := by
sorry

end NUMINAMATH_CALUDE_not_all_negative_l3009_300953


namespace NUMINAMATH_CALUDE_vector_sum_inequality_l3009_300948

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (a b c d : V)

theorem vector_sum_inequality (h : a + b + c + d = 0) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ ≥ ‖a + d‖ + ‖b + d‖ + ‖c + d‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_inequality_l3009_300948


namespace NUMINAMATH_CALUDE_coin_division_problem_l3009_300919

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (∀ m : ℕ, m > 0 → m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 8 = 6) →
  (n % 7 = 5) →
  (n % 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l3009_300919


namespace NUMINAMATH_CALUDE_average_mark_is_76_l3009_300988

def marks : List ℝ := [80, 70, 60, 90, 80]

theorem average_mark_is_76 : (marks.sum / marks.length : ℝ) = 76 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_is_76_l3009_300988


namespace NUMINAMATH_CALUDE_max_candies_equals_complete_graph_edges_l3009_300993

/-- The number of ones initially on the board -/
def initial_ones : Nat := 30

/-- The number of minutes the process continues -/
def total_minutes : Nat := 30

/-- Represents the board state at any given time -/
structure Board where
  numbers : List Nat

/-- Represents a single operation of erasing two numbers and writing their sum -/
def erase_and_sum (b : Board) (i j : Nat) : Board := sorry

/-- The number of candies eaten in a single operation -/
def candies_eaten (b : Board) (i j : Nat) : Nat := sorry

/-- The maximum number of candies that can be eaten -/
def max_candies : Nat := (initial_ones * (initial_ones - 1)) / 2

/-- Theorem stating that the maximum number of candies eaten is equal to
    the number of edges in a complete graph with 'initial_ones' vertices -/
theorem max_candies_equals_complete_graph_edges :
  max_candies = (initial_ones * (initial_ones - 1)) / 2 := by sorry

end NUMINAMATH_CALUDE_max_candies_equals_complete_graph_edges_l3009_300993


namespace NUMINAMATH_CALUDE_line_slope_m_l3009_300917

theorem line_slope_m (m : ℝ) : 
  m > 0 → 
  ((m - 4) / (2 - m) = 2 * m) →
  m = (3 + Real.sqrt 41) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_line_slope_m_l3009_300917


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3009_300925

/-- Given a triangle ABC with side lengths a, b, c, and a point P in the plane of the triangle
    with distances PA = p, PB = q, PC = r, prove that pq/ab + qr/bc + rp/ac ≥ 1 -/
theorem triangle_inequality_sum (a b c p q r : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0) :
  p * q / (a * b) + q * r / (b * c) + r * p / (a * c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3009_300925


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3009_300994

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_fourth : a 4 = 16)
  (h_tenth : a 10 = 2) :
  a 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3009_300994


namespace NUMINAMATH_CALUDE_max_crates_first_trip_solution_l3009_300983

/-- The maximum number of crates that can be carried in the first part of the trip -/
def max_crates_first_trip (total_crates : ℕ) (min_crate_weight : ℕ) (max_trip_weight : ℕ) : ℕ :=
  min (total_crates) (max_trip_weight / min_crate_weight)

theorem max_crates_first_trip_solution :
  max_crates_first_trip 12 120 600 = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_crates_first_trip_solution_l3009_300983


namespace NUMINAMATH_CALUDE_max_interval_length_l3009_300906

theorem max_interval_length (a b : ℝ) (h1 : a < 0) 
  (h2 : ∀ x ∈ Set.Ioo a b, (3 * x^2 + a) * (2 * x + b) ≥ 0) : 
  (b - a) ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_interval_length_l3009_300906


namespace NUMINAMATH_CALUDE_binomial_square_condition_l3009_300929

/-- If 9x^2 + 30x + a is the square of a binomial, then a = 25 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, 9*x^2 + 30*x + a = (b*x + c)^2) → a = 25 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l3009_300929


namespace NUMINAMATH_CALUDE_candidate_votes_l3009_300954

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) : 
  total_votes = 560000 →
  invalid_percent = 15/100 →
  candidate_percent = 75/100 →
  ↑⌊(total_votes : ℚ) * (1 - invalid_percent) * candidate_percent⌋ = 357000 := by
sorry

end NUMINAMATH_CALUDE_candidate_votes_l3009_300954


namespace NUMINAMATH_CALUDE_exam_marks_l3009_300985

theorem exam_marks (full_marks : ℕ) (A B C D : ℕ) : 
  full_marks = 500 →
  A = B - B / 10 →
  B = C + C / 4 →
  C = D - D / 5 →
  D = full_marks * 4 / 5 →
  A = 360 := by sorry

end NUMINAMATH_CALUDE_exam_marks_l3009_300985


namespace NUMINAMATH_CALUDE_problem_solution_l3009_300926

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3009_300926


namespace NUMINAMATH_CALUDE_percentage_good_fruits_l3009_300949

/-- Calculates the percentage of fruits in good condition given the number of oranges and bananas and their respective rotten percentages. -/
theorem percentage_good_fruits (oranges bananas : ℕ) (rotten_oranges_percent rotten_bananas_percent : ℚ) :
  oranges = 600 →
  bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  rotten_bananas_percent = 3 / 100 →
  (((oranges + bananas : ℚ) - (oranges * rotten_oranges_percent + bananas * rotten_bananas_percent)) / (oranges + bananas) * 100 : ℚ) = 89.8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_good_fruits_l3009_300949


namespace NUMINAMATH_CALUDE_square_property_contradiction_l3009_300924

theorem square_property_contradiction (property : ℝ → ℝ) 
  (h_prop : ∀ x : ℝ, property x = (x^2) * property 1) : 
  ¬ (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b = 5 * a ∧ property b = 5 * property a) :=
sorry

end NUMINAMATH_CALUDE_square_property_contradiction_l3009_300924


namespace NUMINAMATH_CALUDE_negation_of_implication_l3009_300910

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a + 1 > b) ↔ (a ≤ b → a + 1 ≤ b) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3009_300910


namespace NUMINAMATH_CALUDE_always_odd_l3009_300984

theorem always_odd (n : ℤ) : ∃ k : ℤ, (n + 1)^3 - n^3 = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l3009_300984


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3009_300959

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧
  (∃ a, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3009_300959


namespace NUMINAMATH_CALUDE_metallic_sheet_length_is_48_l3009_300932

/-- Represents the dimensions and properties of a metallic sheet and the box made from it. -/
structure MetallicSheet where
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Calculates the length of the original metallic sheet given its properties. -/
def calculateLength (sheet : MetallicSheet) : ℝ :=
  sorry

/-- Theorem stating that for a sheet with width 36m, cut size 8m, and resulting box volume 5120m³,
    the original length is 48m. -/
theorem metallic_sheet_length_is_48 :
  let sheet : MetallicSheet := ⟨36, 8, 5120⟩
  calculateLength sheet = 48 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_is_48_l3009_300932


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3009_300990

theorem fraction_to_decimal : (13 : ℚ) / (2 * 5^8) = 0.00001664 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3009_300990


namespace NUMINAMATH_CALUDE_myrtle_absence_duration_l3009_300913

/-- Proves that Myrtle was gone for 21 days given the conditions of the problem -/
theorem myrtle_absence_duration (daily_production neighbor_took dropped remaining : ℕ) 
  (h1 : daily_production = 3)
  (h2 : neighbor_took = 12)
  (h3 : dropped = 5)
  (h4 : remaining = 46) :
  ∃ d : ℕ, d * daily_production - neighbor_took - dropped = remaining ∧ d = 21 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_absence_duration_l3009_300913


namespace NUMINAMATH_CALUDE_swim_meet_transportation_l3009_300977

/-- Represents the swim meet transportation problem -/
theorem swim_meet_transportation (num_cars : ℕ) (people_per_car : ℕ) (people_per_van : ℕ)
  (max_car_capacity : ℕ) (max_van_capacity : ℕ) (additional_capacity : ℕ) :
  num_cars = 2 →
  people_per_car = 5 →
  people_per_van = 3 →
  max_car_capacity = 6 →
  max_van_capacity = 8 →
  additional_capacity = 17 →
  ∃ (num_vans : ℕ), 
    num_vans = 3 ∧
    (num_cars * max_car_capacity + num_vans * max_van_capacity) - 
    (num_cars * people_per_car + num_vans * people_per_van) = additional_capacity :=
by sorry

end NUMINAMATH_CALUDE_swim_meet_transportation_l3009_300977


namespace NUMINAMATH_CALUDE_fence_length_l3009_300996

/-- For a rectangular yard with one side of 40 feet and an area of 480 square feet,
    the sum of the lengths of the other three sides is 64 feet. -/
theorem fence_length (length width : ℝ) : 
  width = 40 → 
  length * width = 480 → 
  2 * length + width = 64 := by
sorry

end NUMINAMATH_CALUDE_fence_length_l3009_300996


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3009_300908

theorem sum_of_coefficients_equals_one (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3009_300908


namespace NUMINAMATH_CALUDE_bodhi_yacht_balance_l3009_300999

/-- The number of sheep needed to balance a yacht -/
def sheep_needed (cows foxes : ℕ) : ℕ :=
  let zebras := 3 * foxes
  let total_needed := 100
  total_needed - (cows + foxes + zebras)

/-- Theorem stating the number of sheep needed for Mr. Bodhi's yacht -/
theorem bodhi_yacht_balance :
  sheep_needed 20 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bodhi_yacht_balance_l3009_300999


namespace NUMINAMATH_CALUDE_sum_of_tangent_slopes_l3009_300921

/-- The parabola P with equation y = x^2 + 10x -/
def P (x y : ℝ) : Prop := y = x^2 + 10 * x

/-- The point Q (10, 5) -/
def Q : ℝ × ℝ := (10, 5)

/-- A line through Q with slope m -/
def line_through_Q (m : ℝ) (x y : ℝ) : Prop :=
  y - Q.2 = m * (x - Q.1)

/-- The sum of slopes of tangent lines to P passing through Q is 60 -/
theorem sum_of_tangent_slopes :
  ∃ r s : ℝ,
    (∀ m : ℝ, r < m ∧ m < s ↔
      ¬∃ x y : ℝ, P x y ∧ line_through_Q m x y) ∧
    r + s = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tangent_slopes_l3009_300921


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3009_300936

theorem two_digit_number_property (c d h : ℕ) (m : ℕ) (y : ℤ) :
  c < 10 →
  d < 10 →
  m = 10 * c + d →
  m = h * (c + d) →
  (10 * d + c : ℤ) = y * (c + d) →
  y = 12 - h :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3009_300936


namespace NUMINAMATH_CALUDE_not_prime_n_pow_n_minus_6n_plus_5_l3009_300937

theorem not_prime_n_pow_n_minus_6n_plus_5 (n : ℕ) : ¬ Prime (n^n - 6*n + 5) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n_pow_n_minus_6n_plus_5_l3009_300937


namespace NUMINAMATH_CALUDE_performance_arrangements_l3009_300907

def original_programs : ℕ := 6
def added_programs : ℕ := 3
def available_spaces : ℕ := original_programs - 1

theorem performance_arrangements : 
  (Nat.descFactorial available_spaces added_programs) + 
  (Nat.descFactorial 3 2 * Nat.descFactorial available_spaces 2) + 
  (5 * Nat.descFactorial 3 3) = 210 := by sorry

end NUMINAMATH_CALUDE_performance_arrangements_l3009_300907


namespace NUMINAMATH_CALUDE_jordan_fourth_period_shots_l3009_300902

/-- Given Jordan's shot-blocking performance in a hockey game, prove the number of shots blocked in the fourth period. -/
theorem jordan_fourth_period_shots 
  (first_period : ℕ) 
  (second_period : ℕ)
  (third_period : ℕ)
  (total_shots : ℕ)
  (h1 : first_period = 4)
  (h2 : second_period = 2 * first_period)
  (h3 : third_period = second_period - 3)
  (h4 : total_shots = 21)
  : total_shots - (first_period + second_period + third_period) = 4 := by
  sorry

end NUMINAMATH_CALUDE_jordan_fourth_period_shots_l3009_300902


namespace NUMINAMATH_CALUDE_tan_sum_squared_l3009_300920

theorem tan_sum_squared (a b : Real) :
  3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b + 1) = 0 →
  (Real.tan (a / 2) + Real.tan (b / 2))^2 = 6 ∨ (Real.tan (a / 2) + Real.tan (b / 2))^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_squared_l3009_300920


namespace NUMINAMATH_CALUDE_two_valid_m_values_l3009_300963

/-- A right triangle in the coordinate plane with legs parallel to the axes -/
structure RightTriangle where
  a : ℝ  -- x-coordinate of the point on the x-axis
  b : ℝ  -- y-coordinate of the point on the y-axis

/-- Check if the given m value satisfies the conditions for the right triangle -/
def satisfiesConditions (t : RightTriangle) (m : ℝ) : Prop :=
  3 * (t.a / 2) + 1 = 0 ∧  -- Condition for the line y = 3x + 1
  t.b / 2 = 2 ∧           -- Condition for the line y = mx + 2
  (t.b / 2) / (t.a / 2) = 4  -- Condition for the ratio of slopes

/-- The theorem stating that there are exactly two values of m that satisfy the conditions -/
theorem two_valid_m_values :
  ∃ m₁ m₂ : ℝ,
    m₁ ≠ m₂ ∧
    (∃ t : RightTriangle, satisfiesConditions t m₁) ∧
    (∃ t : RightTriangle, satisfiesConditions t m₂) ∧
    (∀ m : ℝ, (∃ t : RightTriangle, satisfiesConditions t m) → m = m₁ ∨ m = m₂) :=
  sorry

end NUMINAMATH_CALUDE_two_valid_m_values_l3009_300963


namespace NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l3009_300930

/-- The amount of caffeine Lisa consumed over her goal -/
def caffeine_over_goal (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_consumed : ℕ) : ℕ :=
  max ((caffeine_per_cup * cups_consumed) - daily_goal) 0

/-- Theorem stating that Lisa consumed 40 mg of caffeine over her goal -/
theorem lisa_caffeine_over_goal :
  caffeine_over_goal 80 200 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l3009_300930


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3009_300962

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_prime_divisor_of_factorial_sum : 
  (Nat.factors (factorial 13 + factorial 14)).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3009_300962


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_1_5_l3009_300952

/-- A function f is symmetric about the line x = 1.5 if f(x) = f(3 - x) for all x. -/
def is_symmetric_about_1_5 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (3 - x)

/-- The line x = 1.5 is an axis of symmetry for a function f if 
    for any point (x, f(x)) on the graph, the point (3 - x, f(x)) is also on the graph. -/
def is_axis_of_symmetry_1_5 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y → f (3 - x) = y

theorem symmetry_implies_axis_1_5 (f : ℝ → ℝ) :
  is_symmetric_about_1_5 f → is_axis_of_symmetry_1_5 f :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_axis_1_5_l3009_300952


namespace NUMINAMATH_CALUDE_distance_between_points_l3009_300965

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.1)^2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3009_300965


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l3009_300900

theorem largest_inscribed_triangle_area (r : ℝ) (hr : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (1/2) * diameter * r
  max_triangle_area = 64 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l3009_300900


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l3009_300912

theorem quadratic_expression_values (m n : ℤ) 
  (hm : |m| = 3) 
  (hn : |n| = 2) 
  (hmn : m < n) : 
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l3009_300912


namespace NUMINAMATH_CALUDE_scientific_notation_of_35_8_billion_l3009_300947

theorem scientific_notation_of_35_8_billion : 
  (35800000000 : ℝ) = 3.58 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_35_8_billion_l3009_300947


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3009_300989

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle --/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ b + c > a ∧ a + c > b

/-- The main theorem --/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 3 ∧ b > 3 ∧ c > 3 ∧
    isScaleneTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 23) ∧
    (∀ (x y z : ℕ),
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 3 ∧ y > 3 ∧ z > 3 ∧
      isScaleneTriangle x y z ∧
      isPrime (x + y + z) →
      x + y + z ≥ 23) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3009_300989


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3009_300944

theorem complex_fraction_simplification :
  (3 + 4 * Complex.I) / (5 - 2 * Complex.I) = 7/29 + 26/29 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3009_300944


namespace NUMINAMATH_CALUDE_painted_cube_probability_l3009_300942

/-- Represents a 5x5x5 cube with three faces sharing a vertex painted -/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ
  size_eq : size = 5
  faces_eq : painted_faces = 3

/-- The number of unit cubes with exactly three painted faces -/
def num_three_painted (cube : PaintedCube) : ℕ := 8

/-- The number of unit cubes with exactly one painted face -/
def num_one_painted (cube : PaintedCube) : ℕ := 3 * (cube.size - 2)^2

/-- The total number of unit cubes in the large cube -/
def total_cubes (cube : PaintedCube) : ℕ := cube.size^3

/-- The number of ways to choose two cubes from the total -/
def total_combinations (cube : PaintedCube) : ℕ := (total_cubes cube).choose 2

/-- The number of ways to choose one cube with three painted faces and one with one painted face -/
def favorable_outcomes (cube : PaintedCube) : ℕ := (num_three_painted cube) * (num_one_painted cube)

/-- The probability of selecting one cube with three painted faces and one with one painted face -/
def probability (cube : PaintedCube) : ℚ :=
  (favorable_outcomes cube : ℚ) / (total_combinations cube : ℚ)

theorem painted_cube_probability (cube : PaintedCube) :
  probability cube = 24 / 875 := by sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l3009_300942


namespace NUMINAMATH_CALUDE_correct_factorizations_l3009_300976

theorem correct_factorizations (x y : ℝ) : 
  (x^2 + x*y + y^2 ≠ (x + y)^2) ∧ 
  (-x^2 + 2*x*y - y^2 = -(x - y)^2) ∧ 
  (x^2 + 6*x*y - 9*y^2 ≠ (x - 3*y)^2) ∧ 
  (-x^2 + 1/4 = (1/2 + x)*(1/2 - x)) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorizations_l3009_300976


namespace NUMINAMATH_CALUDE_ratio_problem_l3009_300998

theorem ratio_problem (a b : ℝ) : 
  (a / b = 3 / 8) → 
  ((a - 24) / (b - 24) = 4 / 9) → 
  max a b = 192 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3009_300998


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3009_300901

theorem quadratic_root_difference : 
  let a : ℝ := 6 + 3 * Real.sqrt 5
  let b : ℝ := -(3 + Real.sqrt 5)
  let c : ℝ := 1
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / a
  root_difference = (Real.sqrt 6 - Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3009_300901


namespace NUMINAMATH_CALUDE_blackjack_payout_40_dollars_l3009_300966

/-- Calculates the total amount received for a blackjack bet -/
def blackjack_payout (bet : ℚ) (payout_ratio : ℚ × ℚ) : ℚ :=
  bet + bet * (payout_ratio.1 / payout_ratio.2)

/-- Theorem: The total amount received for a $40 blackjack bet with 3:2 payout is $100 -/
theorem blackjack_payout_40_dollars :
  blackjack_payout 40 (3, 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_blackjack_payout_40_dollars_l3009_300966


namespace NUMINAMATH_CALUDE_x_cubed_minus_2x_plus_1_l3009_300946

theorem x_cubed_minus_2x_plus_1 (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_minus_2x_plus_1_l3009_300946


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3009_300987

theorem minimum_value_theorem (x y : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : x * y = 1/2) :
  (2 / (1 - x) + 1 / (1 - y)) ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3009_300987


namespace NUMINAMATH_CALUDE_polygon_with_360_degree_sum_has_4_sides_l3009_300943

theorem polygon_with_360_degree_sum_has_4_sides :
  ∀ (n : ℕ), n ≥ 3 →
  (n - 2) * 180 = 360 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_360_degree_sum_has_4_sides_l3009_300943


namespace NUMINAMATH_CALUDE_closest_to_zero_l3009_300916

def integers : List Int := [-1101, 1011, -1010, -1001, 1110]

theorem closest_to_zero (n : Int) (h : n ∈ integers) : 
  ∀ m ∈ integers, |n| ≤ |m| ↔ n = -1001 :=
by
  sorry

#check closest_to_zero

end NUMINAMATH_CALUDE_closest_to_zero_l3009_300916


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l3009_300939

theorem no_real_solution_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 5*x + 6)) ∧
              (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 5*x + 6 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l3009_300939


namespace NUMINAMATH_CALUDE_parabola_equation_theorem_l3009_300991

/-- A parabola with vertex at the origin and symmetric about the y-axis. -/
structure Parabola where
  /-- The parameter of the parabola, which is half the length of the chord passing through the focus and perpendicular to the axis of symmetry. -/
  p : ℝ
  /-- The chord passing through the focus and perpendicular to the y-axis has length 16. -/
  chord_length : p * 2 = 16

/-- The equation of a parabola with vertex at the origin and symmetric about the y-axis. -/
def parabola_equation (par : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = 4 * par.p * y) ∨ (x^2 = -4 * par.p * y)

/-- Theorem stating that the equation of the parabola is either x^2 = 32y or x^2 = -32y. -/
theorem parabola_equation_theorem (par : Parabola) : parabola_equation par :=
  sorry

end NUMINAMATH_CALUDE_parabola_equation_theorem_l3009_300991


namespace NUMINAMATH_CALUDE_problem_statement_l3009_300938

def A : Set ℝ := {x | x^2 - x - 2 < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - (2*a+6)*x + a^2 + 6*a ≤ 0}

theorem problem_statement :
  (∀ a : ℝ, (A ⊂ B a ∧ A ≠ B a) → -4 ≤ a ∧ a ≤ -1) ∧
  (∀ a : ℝ, (A ∩ B a = ∅) → a ≤ -7 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3009_300938


namespace NUMINAMATH_CALUDE_exists_common_language_l3009_300997

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  languages : Finset String
  lang_count : languages.card ≤ 4

/-- The set of all scientists at the conference -/
def Scientists : Finset Scientist :=
  sorry

/-- The number of scientists at the conference -/
axiom scientist_count : Scientists.card = 200

/-- For any three scientists, at least two share a common language -/
axiom common_language (s1 s2 s3 : Scientist) :
  s1 ∈ Scientists → s2 ∈ Scientists → s3 ∈ Scientists →
  ∃ (l : String), (l ∈ s1.languages ∧ l ∈ s2.languages) ∨
                  (l ∈ s1.languages ∧ l ∈ s3.languages) ∨
                  (l ∈ s2.languages ∧ l ∈ s3.languages)

/-- Main theorem: There exists a language spoken by at least 26 scientists -/
theorem exists_common_language :
  ∃ (l : String), (Scientists.filter (fun s => l ∈ s.languages)).card ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_exists_common_language_l3009_300997


namespace NUMINAMATH_CALUDE_first_set_cost_l3009_300945

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_cost : ℝ := 50

/-- The cost of 2 footballs and 3 soccer balls in dollars -/
def two_footballs_three_soccer_cost : ℝ := 220

theorem first_set_cost : 3 * football_cost + soccer_cost = 155 :=
  by sorry

end NUMINAMATH_CALUDE_first_set_cost_l3009_300945


namespace NUMINAMATH_CALUDE_distinct_intersection_points_l3009_300904

/-- A line in a plane -/
structure Line :=
  (id : ℕ)

/-- A point where at least two lines intersect -/
structure IntersectionPoint :=
  (lines : Finset Line)

/-- The set of all lines in the plane -/
def all_lines : Finset Line := sorry

/-- The set of all intersection points -/
def intersection_points : Finset IntersectionPoint := sorry

theorem distinct_intersection_points :
  (∀ l ∈ all_lines, ∀ l' ∈ all_lines, l ≠ l' → l.id ≠ l'.id) →  -- lines are distinct
  (Finset.card all_lines = 5) →  -- there are five lines
  (∀ p ∈ intersection_points, Finset.card p.lines ≥ 2) →  -- each intersection point has at least two lines
  (∀ p ∈ intersection_points, Finset.card p.lines ≤ 3) →  -- no more than three lines intersect at a point
  Finset.card intersection_points = 10 :=  -- there are 10 distinct intersection points
by sorry

end NUMINAMATH_CALUDE_distinct_intersection_points_l3009_300904


namespace NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l3009_300982

theorem unique_function_satisfying_inequality (a c d : ℝ) :
  ∃! f : ℝ → ℝ, ∀ x : ℝ, f (a * x + c) + d ≤ x ∧ x ≤ f (x + d) + c :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l3009_300982


namespace NUMINAMATH_CALUDE_average_pencils_is_111_75_l3009_300934

def anna_pencils : ℕ := 50

def harry_pencils : ℕ := 2 * anna_pencils - 19

def lucy_pencils : ℕ := 3 * anna_pencils - 13

def david_pencils : ℕ := 4 * anna_pencils - 21

def total_pencils : ℕ := anna_pencils + harry_pencils + lucy_pencils + david_pencils

def average_pencils : ℚ := total_pencils / 4

theorem average_pencils_is_111_75 : average_pencils = 111.75 := by
  sorry

end NUMINAMATH_CALUDE_average_pencils_is_111_75_l3009_300934


namespace NUMINAMATH_CALUDE_mod_congruence_l3009_300927

theorem mod_congruence (m : ℕ) : 
  (65 * 90 * 111 ≡ m [ZMOD 20]) → 
  (0 ≤ m ∧ m < 20) → 
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_l3009_300927


namespace NUMINAMATH_CALUDE_building_floors_l3009_300955

/-- The number of floors in a building given Earl's movements and position -/
theorem building_floors
  (P Q R S T X : ℕ)
  (h_x_lower : 1 < X)
  (h_x_upper : X < 50)
  : ∃ (F : ℕ), F = 1 + P - Q + R - S + T + X :=
by sorry

end NUMINAMATH_CALUDE_building_floors_l3009_300955


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_coefficient_x4_eq_neg35_l3009_300975

theorem binomial_expansion_coefficient (a : ℝ) : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * a^k * a^(7-k)) = (1 + a)^7 :=
sorry

theorem coefficient_x4_eq_neg35 (a : ℝ) : 
  (Nat.choose 7 3) * a^3 = -35 → a = -1 :=
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_coefficient_x4_eq_neg35_l3009_300975


namespace NUMINAMATH_CALUDE_g_g_is_odd_l3009_300918

def f (x : ℝ) := x^3

def g (x : ℝ) := f (f x)

theorem g_g_is_odd (h1 : ∀ x, f (-x) = -f x) : 
  ∀ x, g (g (-x)) = -(g (g x)) := by sorry

end NUMINAMATH_CALUDE_g_g_is_odd_l3009_300918


namespace NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l3009_300951

theorem triangle_inequality_from_sum_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
  c < a + b ∧ a < b + c ∧ b < c + a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l3009_300951


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3009_300978

theorem complex_number_in_first_quadrant 
  (m n : ℝ) 
  (h : (m : ℂ) / (1 + Complex.I) = 1 - n * Complex.I) : 
  m > 0 ∧ n > 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3009_300978


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l3009_300958

theorem pure_imaginary_square (a : ℝ) : 
  (Complex.I * ((1 : ℂ) + a * Complex.I)^2).re = 0 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l3009_300958


namespace NUMINAMATH_CALUDE_wallpaper_removal_time_is_32_5_l3009_300940

/-- The time it takes to remove wallpaper from the remaining rooms -/
def total_wallpaper_removal_time (dining_remaining_walls : Nat) 
                                 (dining_time_per_wall : Real)
                                 (living_fast_walls : Nat) 
                                 (living_fast_time : Real)
                                 (living_slow_walls : Nat) 
                                 (living_slow_time : Real)
                                 (bedroom_walls : Nat) 
                                 (bedroom_time_per_wall : Real)
                                 (hallway_slow_wall : Nat) 
                                 (hallway_slow_time : Real)
                                 (hallway_fast_walls : Nat) 
                                 (hallway_fast_time : Real) : Real :=
  dining_remaining_walls * dining_time_per_wall +
  living_fast_walls * living_fast_time +
  living_slow_walls * living_slow_time +
  bedroom_walls * bedroom_time_per_wall +
  hallway_slow_wall * hallway_slow_time +
  hallway_fast_walls * hallway_fast_time

/-- Theorem stating that the total wallpaper removal time is 32.5 hours -/
theorem wallpaper_removal_time_is_32_5 : 
  total_wallpaper_removal_time 3 1.5 2 1 2 2.5 3 3 1 4 4 2 = 32.5 := by
  sorry


end NUMINAMATH_CALUDE_wallpaper_removal_time_is_32_5_l3009_300940


namespace NUMINAMATH_CALUDE_line_inclination_45_deg_l3009_300968

/-- Given two points P(-2, m) and Q(m, 4) on a line with inclination angle 45°, prove m = 1 -/
theorem line_inclination_45_deg (m : ℝ) : 
  let P : ℝ × ℝ := (-2, m)
  let Q : ℝ × ℝ := (m, 4)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_45_deg_l3009_300968
