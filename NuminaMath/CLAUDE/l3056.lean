import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_factorization_l3056_305691

theorem quadratic_factorization :
  ∀ x : ℝ, x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3056_305691


namespace NUMINAMATH_CALUDE_complex_number_line_l3056_305649

theorem complex_number_line (z : ℂ) (h : z * (1 + Complex.I)^2 = 1 - Complex.I) :
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_complex_number_line_l3056_305649


namespace NUMINAMATH_CALUDE_choose_3_from_10_l3056_305685

-- Define the number of items to choose from
def n : ℕ := 10

-- Define the number of items to be chosen
def k : ℕ := 3

-- Theorem stating that choosing 3 out of 10 items results in 120 possibilities
theorem choose_3_from_10 : Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_10_l3056_305685


namespace NUMINAMATH_CALUDE_alphabet_size_l3056_305644

theorem alphabet_size (dot_and_line : ℕ) (line_no_dot : ℕ) (dot_no_line : ℕ)
  (h1 : dot_and_line = 20)
  (h2 : line_no_dot = 36)
  (h3 : dot_no_line = 4)
  : dot_and_line + line_no_dot + dot_no_line = 60 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_size_l3056_305644


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l3056_305613

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of new ice cream flavors -/
def new_flavors : ℕ := distribute 5 5

theorem ice_cream_flavors : new_flavors = 126 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l3056_305613


namespace NUMINAMATH_CALUDE_theater_attendance_l3056_305678

/-- Proves that the total number of attendees is 24 given the ticket prices, revenue, and number of children --/
theorem theater_attendance
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_price = 16)
  (h2 : child_price = 9)
  (h3 : total_revenue = 258)
  (h4 : num_children = 18)
  (h5 : ∃ num_adults : ℕ, adult_price * num_adults + child_price * num_children = total_revenue) :
  num_children + (total_revenue - child_price * num_children) / adult_price = 24 :=
by sorry

end NUMINAMATH_CALUDE_theater_attendance_l3056_305678


namespace NUMINAMATH_CALUDE_g_of_f_3_l3056_305658

def f (x : ℝ) : ℝ := x^3 + 3

def g (x : ℝ) : ℝ := 2*x^2 + 2*x + x^3 + 1

theorem g_of_f_3 : g (f 3) = 28861 := by
  sorry

end NUMINAMATH_CALUDE_g_of_f_3_l3056_305658


namespace NUMINAMATH_CALUDE_proposition_truth_values_l3056_305614

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_solution (x : ℤ) : Prop := x^2 + x - 2 = 0

theorem proposition_truth_values :
  ((is_prime 3 ∨ is_even 3) = true) ∧
  ((is_prime 3 ∧ is_even 3) = false) ∧
  ((¬is_prime 3) = false) ∧
  ((is_solution (-2) ∨ is_solution 1) = true) ∧
  ((is_solution (-2) ∧ is_solution 1) = true) ∧
  ((¬is_solution (-2)) = false) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l3056_305614


namespace NUMINAMATH_CALUDE_power_equation_l3056_305629

theorem power_equation : 32^4 * 4^5 = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l3056_305629


namespace NUMINAMATH_CALUDE_equation_equivalence_l3056_305625

theorem equation_equivalence (x : ℝ) : 
  (x + 3) / 3 - (x - 1) / 6 = (5 - x) / 2 ↔ 2*x + 6 - x + 1 = 15 - 3*x := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3056_305625


namespace NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l3056_305682

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  5 / Real.sqrt 2 ≤ Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + Real.sqrt (c^2 + (1-d)^2) + 
    Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) ≤ 5 :=
by sorry

theorem bounds_are_tight : 
  ∃ (a b c d e : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 1) ∧ 
    (0 ≤ d ∧ d ≤ 1) ∧ (0 ≤ e ∧ e ≤ 1) ∧
    Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + Real.sqrt (c^2 + (1-d)^2) + 
    Real.sqrt (d^2 + (1-e)^2) + Real.sqrt (e^2 + (1-a)^2) = 5 / Real.sqrt 2 ∧
  ∃ (a' b' c' d' e' : ℝ), (0 ≤ a' ∧ a' ≤ 1) ∧ (0 ≤ b' ∧ b' ≤ 1) ∧ (0 ≤ c' ∧ c' ≤ 1) ∧ 
    (0 ≤ d' ∧ d' ≤ 1) ∧ (0 ≤ e' ∧ e' ≤ 1) ∧
    Real.sqrt (a'^2 + (1-b')^2) + Real.sqrt (b'^2 + (1-c')^2) + Real.sqrt (c'^2 + (1-d')^2) + 
    Real.sqrt (d'^2 + (1-e')^2) + Real.sqrt (e'^2 + (1-a')^2) = 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l3056_305682


namespace NUMINAMATH_CALUDE_sqrt_5_irrational_l3056_305670

-- Define the set of numbers
def number_set : Set ℝ := {0.618, 22/7, Real.sqrt 5, -3}

-- Define irrationality
def is_irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Theorem statement
theorem sqrt_5_irrational : ∃ (x : ℝ), x ∈ number_set ∧ is_irrational x :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_irrational_l3056_305670


namespace NUMINAMATH_CALUDE_cats_awake_l3056_305666

theorem cats_awake (total : ℕ) (asleep : ℕ) (h1 : total = 98) (h2 : asleep = 92) :
  total - asleep = 6 := by
  sorry

end NUMINAMATH_CALUDE_cats_awake_l3056_305666


namespace NUMINAMATH_CALUDE_binomial_10_5_l3056_305602

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_5_l3056_305602


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3056_305647

/-- Given a line ax + y - 2 = 0 intersecting a circle (x-1)² + (y-a)² = 4 at points A and B,
    where AB is the diameter of the circle, prove that a = 1. -/
theorem intersection_line_circle (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 + A.2 - 2 = 0) ∧ 
    ((A.1 - 1)^2 + (A.2 - a)^2 = 4) ∧
    (a * B.1 + B.2 - 2 = 0) ∧ 
    ((B.1 - 1)^2 + (B.2 - a)^2 = 4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l3056_305647


namespace NUMINAMATH_CALUDE_soccer_match_players_l3056_305621

theorem soccer_match_players (total_socks : ℕ) (socks_per_player : ℕ) : 
  total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_players_l3056_305621


namespace NUMINAMATH_CALUDE_cracker_cost_is_350_l3056_305628

/-- The cost of a box of crackers in dollars -/
def cracker_cost : ℝ := sorry

/-- The total cost before discount in dollars -/
def total_cost_before_discount : ℝ := 5 + 4 * 2 + 3.5 + cracker_cost

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.1

/-- The total cost after discount in dollars -/
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount_rate)

theorem cracker_cost_is_350 :
  cracker_cost = 3.5 ∧ total_cost_after_discount = 18 := by sorry

end NUMINAMATH_CALUDE_cracker_cost_is_350_l3056_305628


namespace NUMINAMATH_CALUDE_tower_heights_count_l3056_305604

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : Nat
  width : Nat
  height : Nat

/-- Represents the possible orientations of a brick -/
inductive BrickOrientation
  | Length
  | Width
  | Height

/-- Calculates the number of different tower heights achievable -/
def calculateTowerHeights (brickDimensions : BrickDimensions) (totalBricks : Nat) : Nat :=
  sorry

/-- Theorem stating the number of different tower heights achievable -/
theorem tower_heights_count (brickDimensions : BrickDimensions) 
  (h1 : brickDimensions.length = 3)
  (h2 : brickDimensions.width = 12)
  (h3 : brickDimensions.height = 20)
  (h4 : totalBricks = 100) :
  calculateTowerHeights brickDimensions totalBricks = 187 := by
    sorry

end NUMINAMATH_CALUDE_tower_heights_count_l3056_305604


namespace NUMINAMATH_CALUDE_original_class_size_l3056_305667

theorem original_class_size (x : ℕ) : 
  (x * 40 + 12 * 32) / (x + 12) = 36 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l3056_305667


namespace NUMINAMATH_CALUDE_total_spent_on_games_l3056_305619

def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

theorem total_spent_on_games :
  batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_games_l3056_305619


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3056_305656

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_prod : a 1 * a 5 = 4)
  (h_a4 : a 4 = 1) :
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3056_305656


namespace NUMINAMATH_CALUDE_max_true_statements_l3056_305668

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 2),
    (x^2 > 2),
    (-2 < x ∧ x < 0),
    (0 < x ∧ x < 2),
    (0 < x - x^2 ∧ x - x^2 < 2)
  ]
  (∀ (s : Finset (Fin 5)), s.card > 3 → ¬(∀ i ∈ s, statements[i]))
  ∧
  (∃ (s : Finset (Fin 5)), s.card = 3 ∧ (∀ i ∈ s, statements[i])) :=
by sorry

#check max_true_statements

end NUMINAMATH_CALUDE_max_true_statements_l3056_305668


namespace NUMINAMATH_CALUDE_students_behind_yoongi_l3056_305635

/-- Given a line of students, prove the number standing behind a specific student. -/
theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) :
  total_students = 20 →
  jungkook_position = 3 →
  yoongi_position = jungkook_position - 1 →
  total_students - yoongi_position = 18 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_l3056_305635


namespace NUMINAMATH_CALUDE_delta_triple_72_l3056_305626

/-- Definition of Δ function -/
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

/-- Theorem stating that Δ(Δ(Δ72)) = 7.728 -/
theorem delta_triple_72 : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end NUMINAMATH_CALUDE_delta_triple_72_l3056_305626


namespace NUMINAMATH_CALUDE_vector_problem_l3056_305631

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Two vectors are in opposite directions if their dot product is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

theorem vector_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (9, x)
  collinear a b → opposite_directions a b → x = -3 := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l3056_305631


namespace NUMINAMATH_CALUDE_intersecting_spheres_equal_volumes_l3056_305689

theorem intersecting_spheres_equal_volumes (r : ℝ) (d : ℝ) : 
  r = 1 → 
  0 < d ∧ d < 2 * r →
  (4 * π * r^3 / 3 - π * (r - d / 2)^2 * (2 * r + d / 2) / 3) * 2 = 4 * π * r^3 / 3 →
  d = 4 * Real.cos (4 * π / 9) :=
sorry

end NUMINAMATH_CALUDE_intersecting_spheres_equal_volumes_l3056_305689


namespace NUMINAMATH_CALUDE_power_function_properties_l3056_305697

-- Define the power function
def f (m : ℕ) (x : ℝ) : ℝ := x^(3*m - 5)

-- Define the theorem
theorem power_function_properties (m : ℕ) :
  (∀ x y, 0 < x ∧ x < y → f m y < f m x) ∧  -- f is decreasing on (0, +∞)
  (∀ x, f m (-x) = f m x) →                 -- f(-x) = f(x)
  m = 1 := by sorry

end NUMINAMATH_CALUDE_power_function_properties_l3056_305697


namespace NUMINAMATH_CALUDE_square_plus_double_is_perfect_square_l3056_305646

theorem square_plus_double_is_perfect_square (a : ℕ) : 
  ∃ (k : ℕ), a^2 + 2*a = k^2 ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_double_is_perfect_square_l3056_305646


namespace NUMINAMATH_CALUDE_line_up_five_people_two_youngest_not_first_l3056_305642

/-- The number of ways to arrange 5 people in a line with restrictions -/
def lineUpWays (n : ℕ) (y : ℕ) (f : ℕ) : ℕ :=
  (n - y) * (n - 1) * (n - 2) * (n - 3) * (n - 4)

/-- Theorem: There are 72 ways for 5 people to line up when 2 youngest can't be first -/
theorem line_up_five_people_two_youngest_not_first :
  lineUpWays 5 2 1 = 72 := by sorry

end NUMINAMATH_CALUDE_line_up_five_people_two_youngest_not_first_l3056_305642


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3056_305623

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3056_305623


namespace NUMINAMATH_CALUDE_counterexample_exists_l3056_305674

theorem counterexample_exists : ∃ n : ℝ, n < 1 ∧ n^2 - 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3056_305674


namespace NUMINAMATH_CALUDE_distinct_weights_theorem_l3056_305692

/-- The number of distinct weights that can be measured with four weights on a two-pan balance scale. -/
def distinct_weights : ℕ := 40

/-- The number of weights available. -/
def num_weights : ℕ := 4

/-- The number of possible placements for each weight (left pan, right pan, or not used). -/
def placement_options : ℕ := 3

/-- Represents the two-pan balance scale. -/
structure BalanceScale :=
  (left_pan : Finset ℕ)
  (right_pan : Finset ℕ)

/-- Calculates the total number of possible configurations. -/
def total_configurations : ℕ := placement_options ^ num_weights

/-- Theorem stating the number of distinct weights that can be measured. -/
theorem distinct_weights_theorem :
  distinct_weights = (total_configurations - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_weights_theorem_l3056_305692


namespace NUMINAMATH_CALUDE_factor_polynomial_l3056_305632

theorem factor_polynomial (x : ℝ) : 60 * x^5 - 135 * x^9 = 15 * x^5 * (4 - 9 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3056_305632


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l3056_305652

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_root_existence (a b c : ℝ) (ha : a ≠ 0) :
  quadratic_function a b c (-3) = -11 →
  quadratic_function a b c (-2) = -5 →
  quadratic_function a b c (-1) = -1 →
  quadratic_function a b c 0 = 1 →
  quadratic_function a b c 1 = 1 →
  ∃ x₁ : ℝ, quadratic_function a b c x₁ = 0 ∧ -1 < x₁ ∧ x₁ < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l3056_305652


namespace NUMINAMATH_CALUDE_quantity_cost_relation_l3056_305622

theorem quantity_cost_relation (Q : ℝ) (h1 : Q * 20 = 1) (h2 : 3.5 * Q * 28 = 1) :
  20 / 8 = 2.5 := by sorry

end NUMINAMATH_CALUDE_quantity_cost_relation_l3056_305622


namespace NUMINAMATH_CALUDE_factorial_inequality_l3056_305620

/-- A function satisfying the given property -/
def special_function (f : ℕ → ℕ) : Prop :=
  ∀ w x y z : ℕ, f (f (f z)) * f (w * x * f (y * f z)) = z^2 * f (x * f y) * f w

/-- The main theorem -/
theorem factorial_inequality (f : ℕ → ℕ) (h : special_function f) : 
  ∀ n : ℕ, f (n.factorial) ≥ n.factorial :=
sorry

end NUMINAMATH_CALUDE_factorial_inequality_l3056_305620


namespace NUMINAMATH_CALUDE_mushroom_trip_theorem_l3056_305638

def mushroom_trip_earnings (day1_earnings day2_price day3_price day4_price day5_price day6_price day7_price : ℝ)
  (day2_mushrooms : ℕ) (day3_increase day4_increase day5_mushrooms day6_decrease day7_mushrooms : ℝ)
  (expenses : ℝ) : Prop :=
  let day2_earnings := day2_mushrooms * day2_price
  let day3_mushrooms := day2_mushrooms + day3_increase
  let day3_earnings := day3_mushrooms * day3_price
  let day4_mushrooms := day3_mushrooms * (1 + day4_increase)
  let day4_earnings := day4_mushrooms * day4_price
  let day5_earnings := day5_mushrooms * day5_price
  let day6_mushrooms := day5_mushrooms * (1 - day6_decrease)
  let day6_earnings := day6_mushrooms * day6_price
  let day7_earnings := day7_mushrooms * day7_price
  let total_earnings := day1_earnings + day2_earnings + day3_earnings + day4_earnings + day5_earnings + day6_earnings + day7_earnings
  total_earnings - expenses = 703.40

theorem mushroom_trip_theorem : 
  mushroom_trip_earnings 120 2.50 1.75 1.30 2.00 2.50 1.80 20 18 0.40 72 0.25 80 25 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_trip_theorem_l3056_305638


namespace NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l3056_305616

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 2| + (y - 1)^2 = 0 → x = -2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l3056_305616


namespace NUMINAMATH_CALUDE_seconds_in_five_and_half_minutes_l3056_305676

-- Define the number of seconds in a minute
def seconds_per_minute : ℕ := 60

-- Define the number of minutes we're converting
def minutes : ℚ := 5 + 1/2

-- Theorem statement
theorem seconds_in_five_and_half_minutes : 
  (minutes * seconds_per_minute : ℚ) = 330 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_five_and_half_minutes_l3056_305676


namespace NUMINAMATH_CALUDE_sydney_more_suitable_l3056_305684

/-- Represents a city with its time difference from Beijing -/
structure City where
  name : String
  timeDiff : Int

/-- Determines if a given hour is suitable for communication -/
def isSuitableHour (hour : Int) : Bool :=
  8 ≤ hour ∧ hour ≤ 22

/-- Calculates the local time given Beijing time and time difference -/
def localTime (beijingTime hour : Int) : Int :=
  (beijingTime + hour + 24) % 24

/-- Theorem: Sydney is more suitable for communication when it's 18:00 in Beijing -/
theorem sydney_more_suitable (sydney : City) (la : City) :
  sydney.name = "Sydney" →
  sydney.timeDiff = 2 →
  la.name = "Los Angeles" →
  la.timeDiff = -15 →
  isSuitableHour (localTime 18 sydney.timeDiff) ∧
  ¬isSuitableHour (localTime 18 la.timeDiff) :=
by
  sorry

#check sydney_more_suitable

end NUMINAMATH_CALUDE_sydney_more_suitable_l3056_305684


namespace NUMINAMATH_CALUDE_quarters_sale_amount_l3056_305693

/-- The amount received for selling quarters at a percentage of their face value -/
def amount_received (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) : ℚ :=
  (num_quarters : ℚ) * face_value * ((percentage : ℚ) / 100)

/-- Theorem stating that selling 8 quarters with face value $0.25 at 500% yields $10 -/
theorem quarters_sale_amount : 
  amount_received 8 (1/4) 500 = 10 := by sorry

end NUMINAMATH_CALUDE_quarters_sale_amount_l3056_305693


namespace NUMINAMATH_CALUDE_david_investment_time_l3056_305690

/-- Simple interest calculation -/
def simpleInterest (principal time rate : ℝ) : ℝ :=
  principal * (1 + time * rate)

theorem david_investment_time :
  ∀ (rate : ℝ),
  rate > 0 →
  simpleInterest 710 3 rate = 815 →
  simpleInterest 710 4 rate = 850 :=
by
  sorry

end NUMINAMATH_CALUDE_david_investment_time_l3056_305690


namespace NUMINAMATH_CALUDE_temperature_difference_l3056_305677

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 8) (h2 : lowest = -1) :
  highest - lowest = 9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3056_305677


namespace NUMINAMATH_CALUDE_min_distance_four_points_l3056_305640

/-- Given four points A, B, C, and D on a line, where the distances between consecutive
    points are AB = 10, BC = 4, and CD = 3, the minimum possible distance between A and D is 3. -/
theorem min_distance_four_points (A B C D : ℝ) : 
  |B - A| = 10 → |C - B| = 4 → |D - C| = 3 → 
  (∃ (A' B' C' D' : ℝ), |B' - A'| = 10 ∧ |C' - B'| = 4 ∧ |D' - C'| = 3 ∧ 
    ∀ (X Y Z W : ℝ), |Y - X| = 10 → |Z - Y| = 4 → |W - Z| = 3 → |W - X| ≥ |D' - A'|) →
  (∃ (A₀ B₀ C₀ D₀ : ℝ), |B₀ - A₀| = 10 ∧ |C₀ - B₀| = 4 ∧ |D₀ - C₀| = 3 ∧ |D₀ - A₀| = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_four_points_l3056_305640


namespace NUMINAMATH_CALUDE_red_cube_possible_l3056_305679

/-- Represents a small cube with colored faces -/
structure SmallCube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents the arrangement of small cubes into a large cube -/
structure LargeCube where
  small_cubes : List SmallCube
  visible_red_faces : Nat
  visible_blue_faces : Nat

/-- Given conditions of the problem -/
def problem_conditions : Prop :=
  ∃ (cubes : List SmallCube) (large_cube : LargeCube),
    -- There are 8 identical cubes
    cubes.length = 8 ∧
    -- Each cube has 6 faces
    ∀ c ∈ cubes, c.blue_faces + c.red_faces = 6 ∧
    -- One-third of all faces are blue, the rest are red
    (cubes.map (λ c => c.blue_faces)).sum = 16 ∧
    (cubes.map (λ c => c.red_faces)).sum = 32 ∧
    -- When assembled into a larger cube, one-third of the visible faces are red
    large_cube.small_cubes = cubes ∧
    large_cube.visible_red_faces = 8 ∧
    large_cube.visible_blue_faces = 16

/-- The theorem to be proved -/
theorem red_cube_possible (h : problem_conditions) :
  ∃ (arrangement : List SmallCube),
    arrangement.length = 8 ∧
    (∀ c ∈ arrangement, c.red_faces ≥ 3) :=
  sorry

end NUMINAMATH_CALUDE_red_cube_possible_l3056_305679


namespace NUMINAMATH_CALUDE_solve_annas_candy_problem_l3056_305608

def annas_candy_problem (initial_money : ℚ) 
                         (gum_price : ℚ) 
                         (gum_quantity : ℕ) 
                         (chocolate_price : ℚ) 
                         (chocolate_quantity : ℕ) 
                         (candy_cane_price : ℚ) 
                         (money_left : ℚ) : Prop :=
  let total_spent := gum_price * gum_quantity + chocolate_price * chocolate_quantity
  let money_for_candy_canes := initial_money - total_spent - money_left
  let candy_canes_bought := money_for_candy_canes / candy_cane_price
  candy_canes_bought = 2

theorem solve_annas_candy_problem : 
  annas_candy_problem 10 1 3 1 5 (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_annas_candy_problem_l3056_305608


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3056_305654

theorem absolute_value_equation_solution :
  {x : ℝ | |2007*x - 2007| = 2007} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3056_305654


namespace NUMINAMATH_CALUDE_wooden_stick_sawing_theorem_l3056_305601

/-- Represents the sawing of a wooden stick into segments -/
structure WoodenStickSawing where
  num_segments : ℕ
  total_time : ℕ
  
/-- Calculates the average time per cut for a wooden stick sawing -/
def average_time_per_cut (sawing : WoodenStickSawing) : ℚ :=
  sawing.total_time / (sawing.num_segments - 1)

/-- Theorem stating that for a wooden stick sawed into 5 segments in 20 minutes,
    the average time per cut is 5 minutes -/
theorem wooden_stick_sawing_theorem (sawing : WoodenStickSawing) 
    (h1 : sawing.num_segments = 5) 
    (h2 : sawing.total_time = 20) : 
    average_time_per_cut sawing = 5 := by
  sorry

end NUMINAMATH_CALUDE_wooden_stick_sawing_theorem_l3056_305601


namespace NUMINAMATH_CALUDE_simple_interest_rate_proof_l3056_305637

/-- The rate at which a sum becomes 4 times of itself in 15 years at simple interest -/
def simple_interest_rate : ℝ := 20

/-- The time period in years -/
def time_period : ℝ := 15

/-- The factor by which the sum increases -/
def growth_factor : ℝ := 4

theorem simple_interest_rate_proof : 
  (1 + simple_interest_rate * time_period / 100) = growth_factor := by
  sorry

#check simple_interest_rate_proof

end NUMINAMATH_CALUDE_simple_interest_rate_proof_l3056_305637


namespace NUMINAMATH_CALUDE_dimes_borrowed_l3056_305636

/-- Represents the number of dimes Sam had initially -/
def initial_dimes : ℕ := 8

/-- Represents the number of dimes Sam has now -/
def remaining_dimes : ℕ := 4

/-- Represents the number of dimes Sam's sister borrowed -/
def borrowed_dimes : ℕ := initial_dimes - remaining_dimes

theorem dimes_borrowed :
  borrowed_dimes = initial_dimes - remaining_dimes :=
by sorry

end NUMINAMATH_CALUDE_dimes_borrowed_l3056_305636


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l3056_305673

theorem root_shift_polynomial (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 5*s₁ - 7 = 0) →
  (s₂^3 - 4*s₂^2 + 5*s₂ - 7 = 0) →
  (s₃^3 - 4*s₃^2 + 5*s₃ - 7 = 0) →
  ((s₁ + 3)^3 - 13*(s₁ + 3)^2 + 56*(s₁ + 3) - 85 = 0) ∧
  ((s₂ + 3)^3 - 13*(s₂ + 3)^2 + 56*(s₂ + 3) - 85 = 0) ∧
  ((s₃ + 3)^3 - 13*(s₃ + 3)^2 + 56*(s₃ + 3) - 85 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l3056_305673


namespace NUMINAMATH_CALUDE_daily_water_intake_l3056_305641

-- Define the given conditions
def daily_soda_cans : ℕ := 5
def ounces_per_can : ℕ := 12
def weekly_total_fluid : ℕ := 868

-- Define the daily soda intake in ounces
def daily_soda_ounces : ℕ := daily_soda_cans * ounces_per_can

-- Define the weekly soda intake in ounces
def weekly_soda_ounces : ℕ := daily_soda_ounces * 7

-- Define the weekly water intake in ounces
def weekly_water_ounces : ℕ := weekly_total_fluid - weekly_soda_ounces

-- Theorem to prove
theorem daily_water_intake : weekly_water_ounces / 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_daily_water_intake_l3056_305641


namespace NUMINAMATH_CALUDE_g_satisfies_equation_l3056_305624

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -4 * x^5 + 7 * x^3 - 5 * x^2 - x + 6

-- State the theorem
theorem g_satisfies_equation : ∀ x : ℝ, 4 * x^5 - 3 * x^3 + x + g x = 7 * x^3 - 5 * x^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_g_satisfies_equation_l3056_305624


namespace NUMINAMATH_CALUDE_toy_value_proof_l3056_305671

theorem toy_value_proof (total_toys : ℕ) (total_worth : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_worth = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    (total_toys - 1) * other_toy_value + special_toy_value = total_worth ∧
    other_toy_value = 5 :=
by sorry

end NUMINAMATH_CALUDE_toy_value_proof_l3056_305671


namespace NUMINAMATH_CALUDE_equation_equivalence_l3056_305695

theorem equation_equivalence (a : ℝ) : (a - 1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3056_305695


namespace NUMINAMATH_CALUDE_sum_of_combinations_l3056_305683

theorem sum_of_combinations : (Nat.choose 4 4) + (Nat.choose 5 4) + (Nat.choose 6 4) + 
  (Nat.choose 7 4) + (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 462 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l3056_305683


namespace NUMINAMATH_CALUDE_m_equals_three_l3056_305609

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

/-- Definition of the complex number z in terms of m -/
def z (m : ℝ) : ℂ := m^2 * (1 + Complex.I) - m * (3 + 6 * Complex.I)

/-- Theorem: If z(m) is pure imaginary, then m = 3 -/
theorem m_equals_three (h : isPureImaginary (z m)) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_equals_three_l3056_305609


namespace NUMINAMATH_CALUDE_min_value_theorem_l3056_305699

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  2 * x + 2 / (x - 1) ≥ 6 ∧ ∃ y > 1, 2 * y + 2 / (y - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3056_305699


namespace NUMINAMATH_CALUDE_salary_sum_proof_l3056_305688

/-- Given 5 people with an average salary of 8600 and one person's salary of 5000,
    prove that the sum of the other 4 people's salaries is 38000 -/
theorem salary_sum_proof (average_salary : ℕ) (num_people : ℕ) (one_salary : ℕ) 
  (h1 : average_salary = 8600)
  (h2 : num_people = 5)
  (h3 : one_salary = 5000) :
  average_salary * num_people - one_salary = 38000 := by
  sorry

#check salary_sum_proof

end NUMINAMATH_CALUDE_salary_sum_proof_l3056_305688


namespace NUMINAMATH_CALUDE_juice_vitamin_c_content_l3056_305657

/-- Vitamin C content in milligrams for different juice combinations -/
theorem juice_vitamin_c_content 
  (apple orange grapefruit : ℝ) 
  (h1 : apple + orange + grapefruit = 275) 
  (h2 : 2 * apple + 3 * orange + 4 * grapefruit = 683) : 
  orange + 2 * grapefruit = 133 := by
sorry

end NUMINAMATH_CALUDE_juice_vitamin_c_content_l3056_305657


namespace NUMINAMATH_CALUDE_tens_digit_sum_factorials_l3056_305669

def factorial (n : ℕ) : ℕ := sorry

def tensDigit (n : ℕ) : ℕ := sorry

def sumFactorials (n : ℕ) : ℕ := sorry

theorem tens_digit_sum_factorials :
  tensDigit (sumFactorials 100) = 0 := by sorry

end NUMINAMATH_CALUDE_tens_digit_sum_factorials_l3056_305669


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3056_305680

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^4 + 14 * X^3 - 56 * X^2 - 72 * X + 88 = 
  (X^2 + 9 * X - 4) * q + (533 * X - 204) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3056_305680


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3056_305672

theorem thirty_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 80 - 0.3 * 80 → x = 44.8 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3056_305672


namespace NUMINAMATH_CALUDE_intersection_k_value_l3056_305687

/-- Given two lines that intersect at x = 5, prove that k = 10 -/
theorem intersection_k_value (k : ℝ) : 
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l3056_305687


namespace NUMINAMATH_CALUDE_quadratic_real_roots_max_integer_a_for_integer_roots_l3056_305664

/-- The quadratic equation x^2 - 2ax + 64 = 0 -/
def quadratic_equation (a x : ℝ) : Prop :=
  x^2 - 2*a*x + 64 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ :=
  4*a^2 - 256

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, quadratic_equation a x) ↔ (a ≥ 8 ∨ a ≤ -8) :=
sorry

theorem max_integer_a_for_integer_roots :
  (∃ a : ℕ+, ∃ x y : ℤ, 
    quadratic_equation a x ∧ 
    quadratic_equation a y ∧ 
    (∀ b : ℕ+, b > a → ¬∃ z w : ℤ, quadratic_equation b z ∧ quadratic_equation b w)) →
  (∃ a : ℕ+, a = 17) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_max_integer_a_for_integer_roots_l3056_305664


namespace NUMINAMATH_CALUDE_bicycle_price_problem_l3056_305627

theorem bicycle_price_problem (cp_a : ℝ) (sp_b sp_c : ℝ) : 
  sp_b = 1.5 * cp_a →
  sp_c = 1.25 * sp_b →
  sp_c = 225 →
  cp_a = 120 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_problem_l3056_305627


namespace NUMINAMATH_CALUDE_natasha_exercise_time_l3056_305651

/-- Proves that Natasha exercised for 30 minutes daily given the conditions of the problem -/
theorem natasha_exercise_time :
  ∀ (d : ℕ),
  let natasha_daily_time : ℕ := 30
  let natasha_total_time : ℕ := d * natasha_daily_time
  let esteban_daily_time : ℕ := 10
  let esteban_days : ℕ := 9
  let esteban_total_time : ℕ := esteban_daily_time * esteban_days
  let total_exercise_time : ℕ := 5 * 60
  natasha_total_time + esteban_total_time = total_exercise_time →
  natasha_daily_time = 30 :=
by
  sorry

#check natasha_exercise_time

end NUMINAMATH_CALUDE_natasha_exercise_time_l3056_305651


namespace NUMINAMATH_CALUDE_charlie_snowball_count_l3056_305618

theorem charlie_snowball_count (lucy_snowballs : ℕ) (charlie_extra : ℕ) 
  (h1 : lucy_snowballs = 19)
  (h2 : charlie_extra = 31) : 
  lucy_snowballs + charlie_extra = 50 := by
  sorry

end NUMINAMATH_CALUDE_charlie_snowball_count_l3056_305618


namespace NUMINAMATH_CALUDE_matrix_product_50_l3056_305659

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (fun acc k => acc * !![1, 2*(k+1); 0, 1])
    !![1, 0; 0, 1]

theorem matrix_product_50 :
  matrix_product 50 = !![1, 2550; 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_50_l3056_305659


namespace NUMINAMATH_CALUDE_fraction_simplification_l3056_305600

theorem fraction_simplification (x : ℝ) : (2*x + 3) / 4 + (4 - 2*x) / 3 = (-2*x + 25) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3056_305600


namespace NUMINAMATH_CALUDE_vertical_shift_equivalence_l3056_305675

/-- A function that represents a vertical shift of another function -/
def verticalShift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x ↦ f x + c

/-- Theorem stating that a vertical shift of a function is equivalent to adding a constant to its output -/
theorem vertical_shift_equivalence (f : ℝ → ℝ) (c : ℝ) :
  ∀ x : ℝ, verticalShift f c x = f x + c := by sorry

end NUMINAMATH_CALUDE_vertical_shift_equivalence_l3056_305675


namespace NUMINAMATH_CALUDE_parabola_vertex_l3056_305698

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 + 1 is (3, 1) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3056_305698


namespace NUMINAMATH_CALUDE_hannahs_speed_l3056_305660

/-- 
Given two drivers, Glen and Hannah, driving towards each other and then away,
prove that Hannah's speed is 15 km/h under the following conditions:
- Glen drives at a constant speed of 37 km/h
- They are 130 km apart at 6 am and 11 am
- They pass each other at some point between 6 am and 11 am
-/
theorem hannahs_speed 
  (glen_speed : ℝ) 
  (initial_distance final_distance : ℝ)
  (time_interval : ℝ) :
  glen_speed = 37 →
  initial_distance = 130 →
  final_distance = 130 →
  time_interval = 5 →
  ∃ (hannah_speed : ℝ), hannah_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_speed_l3056_305660


namespace NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l3056_305662

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_900_prime_factors :
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  sum_of_divisors 900 = p * q * r ∧
  ∀ (s : ℕ), Nat.Prime s → s ∣ sum_of_divisors 900 → (s = p ∨ s = q ∨ s = r) :=
sorry

end NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l3056_305662


namespace NUMINAMATH_CALUDE_extreme_values_sum_reciprocals_l3056_305694

theorem extreme_values_sum_reciprocals (x y : ℝ) :
  (4 * x^2 - 5 * x * y + 4 * y^2 = 5) →
  let S := x^2 + y^2
  (∃ S_max : ℝ, ∀ x y : ℝ, (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → x^2 + y^2 ≤ S_max) ∧
  (∃ S_min : ℝ, ∀ x y : ℝ, (4 * x^2 - 5 * x * y + 4 * y^2 = 5) → S_min ≤ x^2 + y^2) ∧
  (1 / (10/3) + 1 / (10/13) = 8/5) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_sum_reciprocals_l3056_305694


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l3056_305634

theorem coefficient_x5_in_expansion : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * (2 ^ (7 - k)) * if k == 5 then 1 else 0) = 84 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l3056_305634


namespace NUMINAMATH_CALUDE_angle_in_specific_pyramid_l3056_305655

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  AB : ℝ
  CD : ℝ
  distance : ℝ
  volume : ℝ

/-- The angle between two lines in a triangular pyramid -/
def angle_between_lines (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the angle between AB and CD in the specific triangular pyramid -/
theorem angle_in_specific_pyramid :
  let p : TriangularPyramid := {
    AB := 8,
    CD := 12,
    distance := 6,
    volume := 48
  }
  angle_between_lines p = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_in_specific_pyramid_l3056_305655


namespace NUMINAMATH_CALUDE_possible_k_values_l3056_305639

def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (k : ℝ) : Set ℝ := {x | k*x + 1 = 0}

theorem possible_k_values :
  ∀ k : ℝ, (N k ⊆ M) ↔ (k = 0 ∨ k = -1/2 ∨ k = 1/3) := by sorry

end NUMINAMATH_CALUDE_possible_k_values_l3056_305639


namespace NUMINAMATH_CALUDE_square_side_lengths_l3056_305606

theorem square_side_lengths (a b : ℝ) (h1 : a > b) (h2 : a - b = 2) (h3 : a^2 - b^2 = 40) : 
  a = 11 ∧ b = 9 := by
sorry

end NUMINAMATH_CALUDE_square_side_lengths_l3056_305606


namespace NUMINAMATH_CALUDE_power_of_power_l3056_305645

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3056_305645


namespace NUMINAMATH_CALUDE_purchase_problem_l3056_305696

theorem purchase_problem (a b c : ℕ) : 
  a + b + c = 50 →
  60 * a + 500 * b + 400 * c = 10000 →
  a = 30 :=
by sorry

end NUMINAMATH_CALUDE_purchase_problem_l3056_305696


namespace NUMINAMATH_CALUDE_blank_value_l3056_305605

theorem blank_value : (6 : ℝ) / Real.sqrt 18 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_blank_value_l3056_305605


namespace NUMINAMATH_CALUDE_compound_interest_principal_l3056_305615

/-- Given a future value, time, annual interest rate, and compounding frequency,
    calculate the principal amount using the compound interest formula. -/
theorem compound_interest_principal
  (A : ℝ) -- Future value
  (t : ℝ) -- Time in years
  (r : ℝ) -- Annual interest rate (as a decimal)
  (n : ℝ) -- Number of times interest is compounded per year
  (h1 : A = 1000000)
  (h2 : t = 5)
  (h3 : r = 0.08)
  (h4 : n = 4)
  : ∃ P : ℝ, A = P * (1 + r/n)^(n*t) :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l3056_305615


namespace NUMINAMATH_CALUDE_fill_tank_times_l3056_305610

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

/-- Represents the dimensions of the tank -/
def tank_length : ℝ := 30
def tank_width : ℝ := 20
def tank_height : ℝ := 5

/-- Represents the dimensions of the bowl -/
def bowl_length : ℝ := 6
def bowl_width : ℝ := 4
def bowl_height : ℝ := 1

/-- Theorem stating the number of times needed to fill the tank -/
theorem fill_tank_times : 
  (cuboid_volume tank_length tank_width tank_height) / 
  (cuboid_volume bowl_length bowl_width bowl_height) = 125 := by
  sorry

end NUMINAMATH_CALUDE_fill_tank_times_l3056_305610


namespace NUMINAMATH_CALUDE_annes_speed_l3056_305633

theorem annes_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 6) 
  (h2 : time = 3) 
  (h3 : speed = distance / time) : 
  speed = 2 := by
sorry

end NUMINAMATH_CALUDE_annes_speed_l3056_305633


namespace NUMINAMATH_CALUDE_inequality_proofs_l3056_305661

theorem inequality_proofs (a b : ℝ) :
  (a ≥ b ∧ b > 0) →
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b ∧
  (a > 0 ∧ b > 0 ∧ a + b = 10) →
  Real.sqrt (1 + 3 * a) + Real.sqrt (1 + 3 * b) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l3056_305661


namespace NUMINAMATH_CALUDE_fraction_equality_l3056_305648

theorem fraction_equality : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3056_305648


namespace NUMINAMATH_CALUDE_angle_is_90_degrees_l3056_305686

def vector1 : ℝ × ℝ := (4, -3)
def vector2 : ℝ × ℝ := (6, 8)

def angle_between_vectors (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem angle_is_90_degrees :
  angle_between_vectors vector1 vector2 = 90 := by sorry

end NUMINAMATH_CALUDE_angle_is_90_degrees_l3056_305686


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3056_305607

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term is 123/40 -/
theorem arithmetic_sequence_fifth_term
  (x y : ℚ)
  (h1 : (x + y) - (x - y) = (x - y) - (x * y))
  (h2 : (x - y) - (x * y) = (x * y) - (x / y))
  (h3 : y ≠ 0)
  : (x / y) + ((x / y) - (x * y)) = 123/40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3056_305607


namespace NUMINAMATH_CALUDE_prom_attendance_l3056_305612

/-- The number of students who attended the prom on their own -/
def solo_students : ℕ := 3

/-- The number of couples who came to the prom -/
def couples : ℕ := 60

/-- The total number of students who attended the prom -/
def total_students : ℕ := solo_students + 2 * couples

/-- Theorem: The total number of students who attended the prom is 123 -/
theorem prom_attendance : total_students = 123 := by
  sorry

end NUMINAMATH_CALUDE_prom_attendance_l3056_305612


namespace NUMINAMATH_CALUDE_ratio_y_to_x_l3056_305603

theorem ratio_y_to_x (x y z : ℝ) : 
  (0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)) →
  (∃ k : ℤ, z = k * y) →
  (z = 7 * y) →
  (y = 5 * x / 7) →
  y / x = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_y_to_x_l3056_305603


namespace NUMINAMATH_CALUDE_deepak_age_l3056_305611

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) 
    (h1 : rahul_ratio = 4)
    (h2 : deepak_ratio = 3)
    (h3 : rahul_future_age = 26)
    (h4 : rahul_ratio * (rahul_future_age - 10) = deepak_ratio * deepak_present_age) :
  deepak_present_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l3056_305611


namespace NUMINAMATH_CALUDE_range_of_sum_l3056_305630

theorem range_of_sum (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) :
  0 ≤ a + b ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l3056_305630


namespace NUMINAMATH_CALUDE_game_lasts_12_rounds_l3056_305665

/-- Represents the state of the game at any point -/
structure GameState where
  tokens_A : ℕ
  tokens_B : ℕ
  tokens_C : ℕ

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., any player has 0 tokens) -/
def game_ended (state : GameState) : Bool :=
  sorry

/-- Plays the game until it ends, returning the number of rounds played -/
def play_game (initial_state : GameState) : ℕ :=
  sorry

/-- The main theorem stating that the game lasts exactly 12 rounds -/
theorem game_lasts_12_rounds :
  let initial_state : GameState := { tokens_A := 14, tokens_B := 13, tokens_C := 12 }
  play_game initial_state = 12 := by
  sorry

end NUMINAMATH_CALUDE_game_lasts_12_rounds_l3056_305665


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3056_305663

-- Problem 1
theorem problem_1 : 
  (3 * Real.sqrt 18 + (1/6) * Real.sqrt 72 - 4 * Real.sqrt (1/8)) / (4 * Real.sqrt 2) = 9/4 := by
  sorry

-- Problem 2
theorem problem_2 : 
  let x : ℝ := Real.sqrt 2 + 1
  ((x + 2) / (x * (x - 1)) - 1 / (x - 1)) * (x / (x - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3056_305663


namespace NUMINAMATH_CALUDE_ellipse_equation_l3056_305643

theorem ellipse_equation (major_axis : ℝ) (eccentricity : ℝ) :
  major_axis = 8 →
  eccentricity = 3/4 →
  (∃ x y : ℝ, x^2/16 + y^2/7 = 1) ∨ (∃ x y : ℝ, x^2/7 + y^2/16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3056_305643


namespace NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_cube_lt_one_l3056_305650

theorem abs_lt_one_sufficient_not_necessary_for_cube_lt_one :
  (∃ x : ℝ, (|x| < 1 → x^3 < 1) ∧ ¬(x^3 < 1 → |x| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_abs_lt_one_sufficient_not_necessary_for_cube_lt_one_l3056_305650


namespace NUMINAMATH_CALUDE_smallest_valid_integers_difference_l3056_305617

def is_valid (n : ℕ) : Prop :=
  n > 2 ∧ ∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 2

theorem smallest_valid_integers_difference :
  ∃ n m : ℕ, is_valid n ∧ is_valid m ∧
  (∀ x : ℕ, is_valid x → n ≤ x) ∧
  (∀ x : ℕ, is_valid x ∧ x ≠ n → m ≤ x) ∧
  m - n = 13860 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_integers_difference_l3056_305617


namespace NUMINAMATH_CALUDE_book_weight_l3056_305653

theorem book_weight (total_weight : ℝ) (num_books : ℕ) (h1 : total_weight = 42) (h2 : num_books = 14) :
  total_weight / num_books = 3 := by
sorry

end NUMINAMATH_CALUDE_book_weight_l3056_305653


namespace NUMINAMATH_CALUDE_unique_number_in_range_l3056_305681

theorem unique_number_in_range : ∃! x : ℝ, 3 < x ∧ x < 8 ∧ 6 < x ∧ x < 10 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_in_range_l3056_305681
