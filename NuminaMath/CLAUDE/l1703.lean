import Mathlib

namespace NUMINAMATH_CALUDE_mia_fruit_probability_l1703_170378

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 4

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals

/-- The probability of eating at least two different kinds of fruit -/
def prob_different_fruits : ℚ := 1 - (num_fruit_types * prob_same_fruit)

theorem mia_fruit_probability :
  prob_different_fruits = 63 / 64 :=
sorry

end NUMINAMATH_CALUDE_mia_fruit_probability_l1703_170378


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l1703_170314

/-- A line with slope k passing through a fixed point (x₀, y₀) -/
def line_equation (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

/-- The fixed point theorem for a family of lines -/
theorem fixed_point_theorem :
  ∃! p : ℝ × ℝ, ∀ k : ℝ, line_equation k p.1 p.2 (-3) 4 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l1703_170314


namespace NUMINAMATH_CALUDE_double_burgers_count_l1703_170308

/-- Represents the number of single burgers -/
def S : ℕ := sorry

/-- Represents the number of double burgers -/
def D : ℕ := sorry

/-- The total number of burgers -/
def total_burgers : ℕ := 50

/-- The cost of a single burger in cents -/
def single_cost : ℕ := 100

/-- The cost of a double burger in cents -/
def double_cost : ℕ := 150

/-- The total cost of all burgers in cents -/
def total_cost : ℕ := 6650

theorem double_burgers_count :
  S + D = total_burgers ∧
  S * single_cost + D * double_cost = total_cost →
  D = 33 := by sorry

end NUMINAMATH_CALUDE_double_burgers_count_l1703_170308


namespace NUMINAMATH_CALUDE_library_experience_problem_l1703_170332

/-- Represents the years of experience of a library employee -/
structure LibraryExperience where
  current : ℕ
  fiveYearsAgo : ℕ

/-- Represents the age and experience of a library employee -/
structure Employee where
  name : String
  age : ℕ
  experience : LibraryExperience

/-- The problem statement -/
theorem library_experience_problem 
  (bill : Employee)
  (joan : Employee)
  (h1 : bill.age = 40)
  (h2 : joan.age = 50)
  (h3 : joan.experience.fiveYearsAgo = 3 * bill.experience.fiveYearsAgo)
  (h4 : joan.experience.current = 2 * bill.experience.current)
  (h5 : bill.experience.current = bill.experience.fiveYearsAgo + 5)
  (h6 : ∃ (total_experience : ℕ), total_experience = bill.experience.current + 5) :
  bill.experience.current = 10 := by
  sorry

end NUMINAMATH_CALUDE_library_experience_problem_l1703_170332


namespace NUMINAMATH_CALUDE_night_ride_ratio_l1703_170397

def ferris_wheel_total : ℕ := 13
def roller_coaster_total : ℕ := 9
def ferris_wheel_day : ℕ := 7
def roller_coaster_day : ℕ := 4

theorem night_ride_ratio :
  (ferris_wheel_total - ferris_wheel_day) * 5 = (roller_coaster_total - roller_coaster_day) * 6 := by
  sorry

end NUMINAMATH_CALUDE_night_ride_ratio_l1703_170397


namespace NUMINAMATH_CALUDE_product_remainder_l1703_170361

theorem product_remainder (a b m : ℕ) (h : a = 98) (h' : b = 102) (h'' : m = 8) :
  (a * b) % m = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1703_170361


namespace NUMINAMATH_CALUDE_bear_ate_54_pies_l1703_170323

/-- Represents the eating scenario of Masha and the Bear -/
structure EatingScenario where
  totalPies : ℕ
  bearRaspberrySpeed : ℕ
  bearPieSpeed : ℕ
  bearRaspberryRatio : ℕ

/-- Calculates the number of pies eaten by the Bear -/
def bearPies (scenario : EatingScenario) : ℕ :=
  sorry

/-- Theorem stating that the Bear ate 54 pies -/
theorem bear_ate_54_pies (scenario : EatingScenario) 
  (h1 : scenario.totalPies = 60)
  (h2 : scenario.bearRaspberrySpeed = 6)
  (h3 : scenario.bearPieSpeed = 3)
  (h4 : scenario.bearRaspberryRatio = 2) :
  bearPies scenario = 54 := by
  sorry

end NUMINAMATH_CALUDE_bear_ate_54_pies_l1703_170323


namespace NUMINAMATH_CALUDE_area_of_two_sectors_l1703_170311

/-- The area of a figure formed by two sectors of a circle -/
theorem area_of_two_sectors (r : ℝ) (angle1 angle2 : ℝ) (h1 : r = 10) (h2 : angle1 = 45) (h3 : angle2 = 90) :
  (angle1 / 360) * π * r^2 + (angle2 / 360) * π * r^2 = 37.5 * π := by
  sorry

end NUMINAMATH_CALUDE_area_of_two_sectors_l1703_170311


namespace NUMINAMATH_CALUDE_acute_angle_vector_range_l1703_170382

/-- The range of k for acute angle between vectors (2, 1) and (1, k) -/
theorem acute_angle_vector_range :
  ∀ k : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (1, k)
  -- Acute angle condition
  (0 < (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  -- Non-parallel condition
  (a.1 / a.2 ≠ b.1 / b.2) →
  -- Range of k
  (k > -2 ∧ k ≠ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_vector_range_l1703_170382


namespace NUMINAMATH_CALUDE_study_group_selection_probability_l1703_170348

/-- Represents the probability of selecting a member with specific characteristics from a study group -/
def study_group_probability (women_percent : ℝ) (men_percent : ℝ) 
  (women_lawyer_percent : ℝ) (women_doctor_percent : ℝ) (women_engineer_percent : ℝ)
  (men_lawyer_percent : ℝ) (men_doctor_percent : ℝ) (men_engineer_percent : ℝ) : ℝ :=
  let woman_lawyer_prob := women_percent * women_lawyer_percent
  let man_doctor_prob := men_percent * men_doctor_percent
  woman_lawyer_prob + man_doctor_prob

/-- The probability of selecting a woman lawyer or a man doctor from the study group is 0.33 -/
theorem study_group_selection_probability : 
  study_group_probability 0.65 0.35 0.40 0.30 0.30 0.50 0.20 0.30 = 0.33 := by
  sorry

#eval study_group_probability 0.65 0.35 0.40 0.30 0.30 0.50 0.20 0.30

end NUMINAMATH_CALUDE_study_group_selection_probability_l1703_170348


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1703_170313

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_distribution_unique_parameters (ξ : BinomialRV) 
  (h_exp : expectation ξ = 12) 
  (h_var : variance ξ = 2.4) : 
  ξ.n = 15 ∧ ξ.p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l1703_170313


namespace NUMINAMATH_CALUDE_haley_magazines_l1703_170358

theorem haley_magazines 
  (num_boxes : ℕ) 
  (magazines_per_box : ℕ) 
  (h1 : num_boxes = 7)
  (h2 : magazines_per_box = 9) :
  num_boxes * magazines_per_box = 63 := by
  sorry

end NUMINAMATH_CALUDE_haley_magazines_l1703_170358


namespace NUMINAMATH_CALUDE_handshakes_in_gathering_l1703_170381

/-- The number of handshakes in a gathering of couples with specific rules -/
theorem handshakes_in_gathering (n : ℕ) (h : n = 6) : 
  (2 * n) * (2 * n - 3) / 2 = 54 := by sorry

end NUMINAMATH_CALUDE_handshakes_in_gathering_l1703_170381


namespace NUMINAMATH_CALUDE_lab_workstations_l1703_170398

theorem lab_workstations (total_students : ℕ) (two_student_stations : ℕ) (three_student_stations : ℕ) :
  total_students = 38 →
  two_student_stations = 10 →
  two_student_stations * 2 + three_student_stations * 3 = total_students →
  two_student_stations + three_student_stations = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_lab_workstations_l1703_170398


namespace NUMINAMATH_CALUDE_factorization_equality_l1703_170338

theorem factorization_equality (x y : ℝ) : 4 * x^2 - 2 * x * y = 2 * x * (2 * x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1703_170338


namespace NUMINAMATH_CALUDE_pascal_row15_element4_l1703_170306

/-- Pascal's triangle element -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- The fourth element in Row 15 of Pascal's triangle -/
def row15_element4 : ℕ := pascal 15 3

/-- Theorem: The fourth element in Row 15 of Pascal's triangle is 455 -/
theorem pascal_row15_element4 : row15_element4 = 455 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row15_element4_l1703_170306


namespace NUMINAMATH_CALUDE_basketball_win_rate_l1703_170385

theorem basketball_win_rate (games_won : ℕ) (first_games : ℕ) (total_games : ℕ) (remaining_games : ℕ) (win_rate : ℚ) : 
  games_won = 25 ∧ 
  first_games = 35 ∧ 
  total_games = 60 ∧ 
  remaining_games = 25 ∧ 
  win_rate = 4/5 →
  (games_won + remaining_games : ℚ) / total_games = win_rate ↔ 
  remaining_games = 23 := by
sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l1703_170385


namespace NUMINAMATH_CALUDE_sum_squared_l1703_170360

theorem sum_squared (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 + b^2 = 25) : (a + b)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_l1703_170360


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1703_170388

-- Define the filling rates of pipes p and q
def fill_rate_p : ℚ := 1 / 10
def fill_rate_q : ℚ := 1 / 15

-- Define the time both pipes are open together
def initial_time : ℚ := 4

-- Define the total capacity of the cistern
def total_capacity : ℚ := 1

-- Theorem statement
theorem cistern_filling_time :
  let filled_initially := (fill_rate_p + fill_rate_q) * initial_time
  let remaining_to_fill := total_capacity - filled_initially
  let remaining_time := remaining_to_fill / fill_rate_q
  remaining_time = 5 := by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1703_170388


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1703_170331

theorem fraction_equation_solution : ∃ x : ℚ, (1 / 2 - 1 / 3 : ℚ) = 1 / x ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1703_170331


namespace NUMINAMATH_CALUDE_cryptarithm_solution_exists_l1703_170324

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different_digits (Φ E B P A J : ℕ) : Prop :=
  is_valid_digit Φ ∧ is_valid_digit E ∧ is_valid_digit B ∧ 
  is_valid_digit P ∧ is_valid_digit A ∧ is_valid_digit J ∧
  Φ ≠ E ∧ Φ ≠ B ∧ Φ ≠ P ∧ Φ ≠ A ∧ Φ ≠ J ∧
  E ≠ B ∧ E ≠ P ∧ E ≠ A ∧ E ≠ J ∧
  B ≠ P ∧ B ≠ A ∧ B ≠ J ∧
  P ≠ A ∧ P ≠ J ∧
  A ≠ J

theorem cryptarithm_solution_exists :
  ∃ (Φ E B P A J : ℕ), 
    are_different_digits Φ E B P A J ∧
    (Φ : ℚ) / E + (B * 10 + P : ℚ) / A / J = 1 :=
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_exists_l1703_170324


namespace NUMINAMATH_CALUDE_croissant_distribution_l1703_170341

theorem croissant_distribution (total : Nat) (neighbors : Nat) (h1 : total = 59) (h2 : neighbors = 8) :
  total - (neighbors * (total / neighbors)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_croissant_distribution_l1703_170341


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l1703_170346

/-- Given points A, B, C, and D (midpoint of AB), prove that the sum of 
    the slope and y-intercept of the line passing through C and D is 27/5 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 2) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := D.2
  m + b = 27 / 5 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l1703_170346


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1703_170365

theorem quadratic_roots_range (m : ℝ) : 
  (∀ x, x^2 + (m-2)*x + (5-m) = 0 → x > 2) →
  m ∈ Set.Ioc (-5) (-4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1703_170365


namespace NUMINAMATH_CALUDE_fraction_equality_l1703_170370

theorem fraction_equality : (24 + 12) / ((5 - 3) * 2) = 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1703_170370


namespace NUMINAMATH_CALUDE_school_home_time_ratio_l1703_170359

/-- Represents the road segments in Xiaoming's journey --/
inductive RoadSegment
| Flat
| Uphill
| Downhill

/-- Represents the direction of Xiaoming's journey --/
inductive Direction
| ToSchool
| ToHome

/-- Calculates the time taken for a segment of the journey --/
def segmentTime (segment : RoadSegment) (direction : Direction) : ℚ :=
  match segment, direction with
  | RoadSegment.Flat, _ => 1 / 3
  | RoadSegment.Uphill, Direction.ToSchool => 1
  | RoadSegment.Uphill, Direction.ToHome => 1
  | RoadSegment.Downhill, Direction.ToSchool => 1 / 4
  | RoadSegment.Downhill, Direction.ToHome => 1 / 2

/-- Calculates the total time for a journey in a given direction --/
def journeyTime (direction : Direction) : ℚ :=
  segmentTime RoadSegment.Flat direction +
  2 * segmentTime RoadSegment.Uphill direction +
  segmentTime RoadSegment.Downhill direction

/-- Main theorem: The ratio of time to school vs time to home is 19:16 --/
theorem school_home_time_ratio :
  (journeyTime Direction.ToSchool) / (journeyTime Direction.ToHome) = 19 / 16 := by
  sorry


end NUMINAMATH_CALUDE_school_home_time_ratio_l1703_170359


namespace NUMINAMATH_CALUDE_train_length_proof_l1703_170315

-- Define the speed of the train in km/hr
def train_speed_kmh : ℝ := 108

-- Define the time it takes for the train to pass the tree in seconds
def passing_time : ℝ := 8

-- Theorem to prove the length of the train
theorem train_length_proof : 
  train_speed_kmh * 1000 / 3600 * passing_time = 240 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l1703_170315


namespace NUMINAMATH_CALUDE_selection_ways_l1703_170305

def group_size : ℕ := 8
def roles_to_fill : ℕ := 3

theorem selection_ways : (group_size.factorial) / ((group_size - roles_to_fill).factorial) = 336 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l1703_170305


namespace NUMINAMATH_CALUDE_rectangle_breadth_ratio_l1703_170327

/-- Given a rectangle where the length is halved and the area is reduced by 50%,
    prove that the ratio of new breadth to original breadth is 0.5 -/
theorem rectangle_breadth_ratio
  (L B : ℝ)  -- Original length and breadth
  (L' B' : ℝ) -- New length and breadth
  (h1 : L' = L / 2)  -- New length is half of original
  (h2 : L' * B' = (L * B) / 2)  -- New area is half of original
  : B' / B = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_breadth_ratio_l1703_170327


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1703_170330

theorem distinct_prime_factors_count (n : ℕ) : n = 95 * 97 * 99 * 101 → Finset.card (Nat.factors n).toFinset = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1703_170330


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_l1703_170387

theorem negation_of_universal_positive_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_l1703_170387


namespace NUMINAMATH_CALUDE_lcm_1320_924_l1703_170393

theorem lcm_1320_924 : Nat.lcm 1320 924 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1320_924_l1703_170393


namespace NUMINAMATH_CALUDE_chair_and_vase_cost_indeterminate_l1703_170321

/-- Represents the cost of items at a garage sale. -/
structure GarageSale where
  total : ℝ
  table : ℝ
  chairs : ℕ
  mirror : ℝ
  lamp : ℝ
  vases : ℕ
  chair_cost : ℝ
  vase_cost : ℝ

/-- Conditions of Nadine's garage sale purchase -/
def nadines_purchase : GarageSale where
  total := 105
  table := 34
  chairs := 2
  mirror := 15
  lamp := 6
  vases := 3
  chair_cost := 0  -- placeholder, actual value unknown
  vase_cost := 0   -- placeholder, actual value unknown

/-- Theorem stating that the sum of one chair and one vase cost cannot be uniquely determined -/
theorem chair_and_vase_cost_indeterminate (g : GarageSale) (h : g = nadines_purchase) :
  ¬ ∃! x : ℝ, x = g.chair_cost + g.vase_cost ∧
    g.total = g.table + g.mirror + g.lamp + g.chairs * g.chair_cost + g.vases * g.vase_cost :=
sorry

end NUMINAMATH_CALUDE_chair_and_vase_cost_indeterminate_l1703_170321


namespace NUMINAMATH_CALUDE_painted_faces_difference_l1703_170376

/-- Represents a 3D cube structure --/
structure CubeStructure where
  length : Nat
  width : Nat
  height : Nat

/-- Counts cubes with exactly n painted faces in the structure --/
def countPaintedFaces (cs : CubeStructure) (n : Nat) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem painted_faces_difference (cs : CubeStructure) :
  cs.length = 7 → cs.width = 7 → cs.height = 3 →
  countPaintedFaces cs 3 - countPaintedFaces cs 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_painted_faces_difference_l1703_170376


namespace NUMINAMATH_CALUDE_math_club_teams_l1703_170377

theorem math_club_teams (girls boys : ℕ) (h1 : girls = 4) (h2 : boys = 6) :
  (girls.choose 2) * (boys.choose 2) = 90 := by
sorry

end NUMINAMATH_CALUDE_math_club_teams_l1703_170377


namespace NUMINAMATH_CALUDE_lock_code_difference_l1703_170325

def is_valid_code (a b c : Nat) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (b * b) = (a * c * c)

def code_value (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

theorem lock_code_difference : 
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : Nat),
    is_valid_code a₁ b₁ c₁ ∧
    is_valid_code a₂ b₂ c₂ ∧
    (∀ a b c, is_valid_code a b c → 
      code_value a b c ≤ code_value a₁ b₁ c₁ ∧
      code_value a b c ≥ code_value a₂ b₂ c₂) ∧
    code_value a₁ b₁ c₁ - code_value a₂ b₂ c₂ = 541 :=
by sorry

end NUMINAMATH_CALUDE_lock_code_difference_l1703_170325


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_correct_l1703_170326

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

def smallest_dual_palindrome : ℕ := 33

theorem smallest_dual_palindrome_correct :
  (smallest_dual_palindrome > 10) ∧
  (is_palindrome smallest_dual_palindrome 3) ∧
  (is_palindrome smallest_dual_palindrome 5) ∧
  (∀ m : ℕ, m > 10 ∧ m < smallest_dual_palindrome →
    ¬(is_palindrome m 3 ∧ is_palindrome m 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_correct_l1703_170326


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_100011_l1703_170373

theorem smallest_six_digit_divisible_by_100011 :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 → n % 100011 = 0 → n ≥ 100011 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_100011_l1703_170373


namespace NUMINAMATH_CALUDE_sum_of_squares_rational_l1703_170394

theorem sum_of_squares_rational (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ q : ℚ, a + b = q) → 
  (∃ r : ℚ, a^3 + b^3 = r) → 
  (∃ s : ℚ, a^2 + b^2 = s) ∧ 
  ¬(∀ t u : ℚ, a = t ∧ b = u) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_rational_l1703_170394


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1703_170337

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the value of c given the conditions -/
theorem quadratic_inequality_solution (a b m : ℝ) :
  (∀ x, f a b x ≥ 0) →
  (∃ c, ∀ x, f a b x < c ↔ m < x ∧ x < m + 6) →
  ∃ c, (∀ x, f a b x < c ↔ m < x ∧ x < m + 6) ∧ c = 9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1703_170337


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1703_170364

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The given complex number -/
def z : ℂ := (1 - 2*i) * i

/-- Theorem stating that the magnitude of z is √5 -/
theorem magnitude_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1703_170364


namespace NUMINAMATH_CALUDE_area_distance_relation_l1703_170300

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  a : ℝ  -- Length of the rectangle
  b : ℝ  -- Width of the rectangle
  t : ℝ  -- Area of the original rectangle
  t₁ : ℝ  -- Area of the first smaller rectangle
  t₂ : ℝ  -- Area of the second smaller rectangle
  t₃ : ℝ  -- Area of the third smaller rectangle
  t₄ : ℝ  -- Area of the fourth smaller rectangle
  z : ℝ  -- Distance from the center of the original rectangle to line e
  z₁ : ℝ  -- Distance from the center of the first smaller rectangle to line e
  z₂ : ℝ  -- Distance from the center of the second smaller rectangle to line e
  z₃ : ℝ  -- Distance from the center of the third smaller rectangle to line e
  z₄ : ℝ  -- Distance from the center of the fourth smaller rectangle to line e
  h_positive : a > 0 ∧ b > 0  -- Ensure positive dimensions
  h_area : t = a * b  -- Area of the original rectangle
  h_sum_areas : t = t₁ + t₂ + t₃ + t₄  -- Sum of areas of smaller rectangles

/-- The theorem stating the relationship between areas and distances -/
theorem area_distance_relation (r : DividedRectangle) :
    r.t₁ * r.z₁ + r.t₂ * r.z₂ + r.t₃ * r.z₃ + r.t₄ * r.z₄ = r.t * r.z := by
  sorry

end NUMINAMATH_CALUDE_area_distance_relation_l1703_170300


namespace NUMINAMATH_CALUDE_candy_distribution_l1703_170347

theorem candy_distribution (hugh tommy melany lily : ℝ) 
  (h_hugh : hugh = 8.5)
  (h_tommy : tommy = 6.75)
  (h_melany : melany = 7.25)
  (h_lily : lily = 5.5) :
  let total := hugh + tommy + melany + lily
  let num_people := 4
  (total / num_people) = 7 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1703_170347


namespace NUMINAMATH_CALUDE_overall_profit_percentage_l1703_170334

/-- Calculate the overall profit percentage for four items --/
theorem overall_profit_percentage
  (sp_a : ℝ) (cp_percent_a : ℝ)
  (sp_b : ℝ) (cp_percent_b : ℝ)
  (sp_c : ℝ) (cp_percent_c : ℝ)
  (sp_d : ℝ) (cp_percent_d : ℝ)
  (h_sp_a : sp_a = 120)
  (h_cp_percent_a : cp_percent_a = 30)
  (h_sp_b : sp_b = 200)
  (h_cp_percent_b : cp_percent_b = 20)
  (h_sp_c : sp_c = 75)
  (h_cp_percent_c : cp_percent_c = 40)
  (h_sp_d : sp_d = 180)
  (h_cp_percent_d : cp_percent_d = 25) :
  let cp_a := sp_a * (cp_percent_a / 100)
  let cp_b := sp_b * (cp_percent_b / 100)
  let cp_c := sp_c * (cp_percent_c / 100)
  let cp_d := sp_d * (cp_percent_d / 100)
  let total_cp := cp_a + cp_b + cp_c + cp_d
  let total_sp := sp_a + sp_b + sp_c + sp_d
  let total_profit := total_sp - total_cp
  let profit_percentage := (total_profit / total_cp) * 100
  abs (profit_percentage - 280.79) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_overall_profit_percentage_l1703_170334


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l1703_170352

theorem quadratic_roots_nature (x : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -4 * Real.sqrt 5
  let c : ℝ := 20
  let discriminant := b^2 - 4*a*c
  discriminant = 0 ∧ ∃ (root : ℝ), x^2 - 4*x*(Real.sqrt 5) + 20 = 0 → x = root :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l1703_170352


namespace NUMINAMATH_CALUDE_last_ten_seconds_distance_l1703_170303

/-- The distance function of a plane's taxiing after landing -/
def distance (t : ℝ) : ℝ := 60 * t - 1.5 * t^2

/-- The time at which the plane stops -/
def stop_time : ℝ := 20

/-- Theorem: The plane travels 150 meters in the last 10 seconds before stopping -/
theorem last_ten_seconds_distance : 
  distance stop_time - distance (stop_time - 10) = 150 := by
  sorry

end NUMINAMATH_CALUDE_last_ten_seconds_distance_l1703_170303


namespace NUMINAMATH_CALUDE_contrapositive_example_l1703_170343

theorem contrapositive_example (a b : ℝ) :
  (¬(a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l1703_170343


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_inequality_l1703_170355

theorem cubic_polynomial_root_inequality (A B C : ℝ) (α β γ : ℂ) 
  (h : ∀ x : ℂ, x^3 + A*x^2 + B*x + C = 0 ↔ x = α ∨ x = β ∨ x = γ) :
  (1 + |A| + |B| + |C|) / (Complex.abs α + Complex.abs β + Complex.abs γ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_inequality_l1703_170355


namespace NUMINAMATH_CALUDE_milton_books_l1703_170322

theorem milton_books (total : ℕ) (zoology : ℕ) (botany : ℕ) : 
  total = 960 → 
  botany = 7 * zoology → 
  total = zoology + botany → 
  zoology = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_milton_books_l1703_170322


namespace NUMINAMATH_CALUDE_equation_solution_l1703_170395

theorem equation_solution :
  ∀ y : ℝ, (45 : ℝ) / 75 = Real.sqrt (y / 25) → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1703_170395


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l1703_170392

-- Define the polynomials
def p1 (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 5 * x - 2
def p2 (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 4

-- Define the product of the polynomials
def product (x : ℝ) : ℝ := p1 x * p2 x

-- Theorem statement
theorem coefficient_of_x_squared :
  ∃ (a b c d : ℝ), product = fun x => a * x^3 + (-5) * x^2 + b * x + c + d * x^4 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l1703_170392


namespace NUMINAMATH_CALUDE_xy_squared_equals_one_l1703_170301

theorem xy_squared_equals_one 
  (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 6) : 
  x^2 * y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_equals_one_l1703_170301


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1703_170357

theorem unique_solution_condition (c k : ℝ) (h_c : c ≠ 0) : 
  (∃! b : ℝ, b > 0 ∧ 
    (∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ∧ 
    b^4 + (2 - 4*c) * b^2 + k = 0) ↔ 
  c = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1703_170357


namespace NUMINAMATH_CALUDE_equation_solutions_l1703_170302

/-- The integer part of a real number -/
noncomputable def intPart (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part of a real number -/
noncomputable def fracPart (x : ℝ) : ℝ :=
  x - intPart x

/-- The solutions to the equation [x] · {x} = 1991x -/
theorem equation_solutions :
  ∀ x : ℝ, intPart x * fracPart x = 1991 * x ↔ x = 0 ∨ x = -1 / 1992 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1703_170302


namespace NUMINAMATH_CALUDE_percentage_increase_l1703_170391

theorem percentage_increase (initial final : ℝ) (h1 : initial = 60) (h2 : final = 90) :
  (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1703_170391


namespace NUMINAMATH_CALUDE_lottery_winnings_l1703_170386

/-- Calculates the total winnings for lottery tickets -/
theorem lottery_winnings
  (num_tickets : ℕ)
  (winning_numbers_per_ticket : ℕ)
  (value_per_winning_number : ℕ)
  (h1 : num_tickets = 3)
  (h2 : winning_numbers_per_ticket = 5)
  (h3 : value_per_winning_number = 20) :
  num_tickets * winning_numbers_per_ticket * value_per_winning_number = 300 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winnings_l1703_170386


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1703_170335

theorem gcd_lcm_sum : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1703_170335


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l1703_170349

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    a > 1 → b > 1 → c > 1 →
    (c ∣ (a * b + 1)) →
    (a ∣ (b * c + 1)) →
    (b ∣ (c * a + 1)) →
    ((a = 2 ∧ b = 3 ∧ c = 7) ∨
     (a = 2 ∧ b = 7 ∧ c = 3) ∨
     (a = 3 ∧ b = 2 ∧ c = 7) ∨
     (a = 3 ∧ b = 7 ∧ c = 2) ∨
     (a = 7 ∧ b = 2 ∧ c = 3) ∨
     (a = 7 ∧ b = 3 ∧ c = 2)) :=
by sorry


end NUMINAMATH_CALUDE_abc_divisibility_problem_l1703_170349


namespace NUMINAMATH_CALUDE_intersection_A_B_zero_range_of_m_l1703_170304

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem 1: Intersection of A and B when m = 0
theorem intersection_A_B_zero : A ∩ B 0 = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of m when q is necessary but not sufficient for p
theorem range_of_m (h : ∀ x, p x → q x m) : 
  m ≤ -2 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_zero_range_of_m_l1703_170304


namespace NUMINAMATH_CALUDE_triangle_cos_2C_l1703_170380

theorem triangle_cos_2C (a b : ℝ) (S_ABC : ℝ) (C : ℝ) :
  a = 8 →
  b = 5 →
  S_ABC = 12 →
  S_ABC = 1/2 * a * b * Real.sin C →
  Real.cos (2 * C) = 7/25 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_cos_2C_l1703_170380


namespace NUMINAMATH_CALUDE_aubree_beaver_count_l1703_170333

/-- The number of beavers Aubree initially saw -/
def initial_beavers : ℕ := 20

/-- The number of chipmunks Aubree initially saw -/
def initial_chipmunks : ℕ := 40

/-- The total number of animals Aubree saw -/
def total_animals : ℕ := 130

theorem aubree_beaver_count :
  initial_beavers = 20 ∧
  initial_chipmunks = 40 ∧
  total_animals = 130 ∧
  initial_beavers + initial_chipmunks + 2 * initial_beavers + (initial_chipmunks - 10) = total_animals :=
by sorry

end NUMINAMATH_CALUDE_aubree_beaver_count_l1703_170333


namespace NUMINAMATH_CALUDE_derivative_at_one_l1703_170372

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1703_170372


namespace NUMINAMATH_CALUDE_max_min_difference_d_l1703_170362

theorem max_min_difference_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 5)
  (sum_sq_eq : a^2 + b^2 + c^2 + d^2 = 18) : 
  ∃ (d_max d_min : ℝ),
    (∀ d', a + b + c + d' = 5 ∧ a^2 + b^2 + c^2 + d'^2 = 18 → d' ≤ d_max) ∧
    (∀ d', a + b + c + d' = 5 ∧ a^2 + b^2 + c^2 + d'^2 = 18 → d_min ≤ d') ∧
    d_max - d_min = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_d_l1703_170362


namespace NUMINAMATH_CALUDE_mushroom_count_l1703_170317

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem: Given the conditions, the number of mushrooms is 950 -/
theorem mushroom_count : ∃ (N : ℕ), 
  100 ≤ N ∧ N < 1000 ∧  -- N is a three-digit number
  sumOfDigits N = 14 ∧  -- sum of digits is 14
  N % 50 = 0 ∧  -- N is divisible by 50
  N % 100 = 50 ∧  -- N ends in 50
  N = 950 := by
sorry

end NUMINAMATH_CALUDE_mushroom_count_l1703_170317


namespace NUMINAMATH_CALUDE_triangle_theorem_l1703_170353

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle)
  (h1 : 3 * (t.b^2 + t.c^2) = 3 * t.a^2 + 2 * t.b * t.c) :
  (∀ (h2 : Real.sin t.B = Real.sqrt 2 * Real.cos t.C),
    Real.tan t.C = Real.sqrt 2) ∧
  (∀ (h3 : t.a = 2)
     (h4 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 2 / 2)
     (h5 : t.b > t.c),
    t.b = 3 * Real.sqrt 2 / 2 ∧ t.c = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1703_170353


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1703_170383

/-- A triangle with sides a, b, c exists -/
def triangle_exists (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Triangle PQR with sides p, q, r -/
structure Triangle (p q r : ℝ) : Type :=
  (exists_triangle : triangle_exists p q r)

/-- For any positive integer n, a triangle with sides p^n, q^n, r^n exists -/
def power_triangle_exists (p q r : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → triangle_exists (p^n) (q^n) (r^n)

/-- Main theorem: If a triangle PQR with sides p, q, r exists, and for any positive integer n,
    a triangle with sides p^n, q^n, r^n also exists, then at least two sides of triangle PQR are equal -/
theorem isosceles_triangle (p q r : ℝ) (tr : Triangle p q r) 
    (h : power_triangle_exists p q r) : 
    p = q ∨ q = r ∨ r = p :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1703_170383


namespace NUMINAMATH_CALUDE_fermat_number_properties_l1703_170367

/-- Fermat number F_n -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Main theorem -/
theorem fermat_number_properties (n : ℕ) (p : ℕ) (h_n : n ≥ 2) (h_p : Nat.Prime p) (h_factor : p ∣ F n) :
  (∃ x : ℤ, x^2 ≡ 2 [ZMOD p]) ∧ p ≡ 1 [ZMOD 2^(n+2)] := by sorry

end NUMINAMATH_CALUDE_fermat_number_properties_l1703_170367


namespace NUMINAMATH_CALUDE_oddProbabilityConvergesTo1Third_l1703_170316

/-- Represents the state of the calculator --/
structure CalculatorState where
  display : ℕ
  lastOperation : Option (ℕ → ℕ → ℕ)

/-- Represents a button press on the calculator --/
inductive ButtonPress
  | Digit (d : Fin 10)
  | Add
  | Multiply

/-- The probability of the display showing an odd number after n button presses --/
def oddProbability (n : ℕ) : ℝ := sorry

/-- The limiting probability of the display showing an odd number as n approaches infinity --/
def limitingOddProbability : ℝ := sorry

/-- The main theorem stating that the limiting probability converges to 1/3 --/
theorem oddProbabilityConvergesTo1Third :
  limitingOddProbability = 1/3 := by sorry

end NUMINAMATH_CALUDE_oddProbabilityConvergesTo1Third_l1703_170316


namespace NUMINAMATH_CALUDE_mean_equality_problem_l1703_170369

theorem mean_equality_problem (z : ℝ) : 
  (7 + 10 + 15 + 21) / 4 = (18 + z) / 2 → z = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l1703_170369


namespace NUMINAMATH_CALUDE_ivy_collectors_edition_fraction_l1703_170351

theorem ivy_collectors_edition_fraction (dina_dolls : ℕ) (ivy_collectors : ℕ) : 
  dina_dolls = 60 →
  ivy_collectors = 20 →
  2 * (dina_dolls / 2) = dina_dolls →
  (ivy_collectors : ℚ) / (dina_dolls / 2 : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ivy_collectors_edition_fraction_l1703_170351


namespace NUMINAMATH_CALUDE_condition_relationship_l1703_170310

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a - b > 0 → a^2 - b^2 > 0) ∧
  (∃ a b, a^2 - b^2 > 0 ∧ a - b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1703_170310


namespace NUMINAMATH_CALUDE_stratified_sample_composition_l1703_170309

/-- Represents the number of male students in the class -/
def male_students : ℕ := 40

/-- Represents the number of female students in the class -/
def female_students : ℕ := 30

/-- Represents the total number of students in the class -/
def total_students : ℕ := male_students + female_students

/-- Represents the size of the stratified sample -/
def sample_size : ℕ := 7

/-- Calculates the number of male students in the stratified sample -/
def male_sample : ℕ := (male_students * sample_size + total_students - 1) / total_students

/-- Calculates the number of female students in the stratified sample -/
def female_sample : ℕ := sample_size - male_sample

/-- Theorem stating that the stratified sample consists of 4 male and 3 female students -/
theorem stratified_sample_composition :
  male_sample = 4 ∧ female_sample = 3 :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_composition_l1703_170309


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l1703_170363

-- Define propositions p and q
def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_not_q :
  (∀ x, ¬(q x) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ q x) :=
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l1703_170363


namespace NUMINAMATH_CALUDE_abs_neg_x_eq_five_implies_x_plus_minus_five_l1703_170399

theorem abs_neg_x_eq_five_implies_x_plus_minus_five (x : ℝ) : 
  |(-x)| = 5 → x = -5 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_neg_x_eq_five_implies_x_plus_minus_five_l1703_170399


namespace NUMINAMATH_CALUDE_investment_interest_l1703_170345

theorem investment_interest (total_investment : ℝ) (high_rate_investment : ℝ) (high_rate : ℝ) (low_rate : ℝ) :
  total_investment = 22000 →
  high_rate_investment = 7000 →
  high_rate = 0.18 →
  low_rate = 0.14 →
  let low_rate_investment := total_investment - high_rate_investment
  let high_rate_interest := high_rate_investment * high_rate
  let low_rate_interest := low_rate_investment * low_rate
  let total_interest := high_rate_interest + low_rate_interest
  total_interest = 3360 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_l1703_170345


namespace NUMINAMATH_CALUDE_four_solutions_l1703_170379

/-- S(n) denotes the sum of the digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- The number of positive integers n such that n + S(n) + S(S(n)) = 2010 -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 4 solutions -/
theorem four_solutions : count_solutions = 4 := by sorry

end NUMINAMATH_CALUDE_four_solutions_l1703_170379


namespace NUMINAMATH_CALUDE_division_problem_l1703_170342

theorem division_problem : ∃ (d r : ℕ), d > 0 ∧ 1270 = 74 * d + r ∧ r < d := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_division_problem_l1703_170342


namespace NUMINAMATH_CALUDE_greater_number_proof_l1703_170366

theorem greater_number_proof (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 8) (h3 : x > y) : x = 19 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_proof_l1703_170366


namespace NUMINAMATH_CALUDE_train_length_l1703_170312

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 135 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1703_170312


namespace NUMINAMATH_CALUDE_soccer_balls_count_l1703_170354

theorem soccer_balls_count (soccer : ℕ) (baseball : ℕ) (volleyball : ℕ) : 
  baseball = 5 * soccer →
  volleyball = 3 * soccer →
  baseball + volleyball = 160 →
  soccer = 20 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l1703_170354


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l1703_170384

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5, 6}

theorem intersection_complement_equals : A ∩ (U \ B) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l1703_170384


namespace NUMINAMATH_CALUDE_labor_market_effects_l1703_170374

-- Define the labor market for doctors
structure LaborMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibriumWage : ℝ  -- Equilibrium wage

-- Define the commercial healthcare market
structure HealthcareMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibriumPrice : ℝ  -- Equilibrium price

-- Define the government policy
def governmentPolicy (minYears : ℕ) : Prop :=
  ∃ (requirement : ℕ), requirement ≥ minYears

-- Theorem statement
theorem labor_market_effects
  (initialMarket : LaborMarket)
  (initialHealthcare : HealthcareMarket)
  (policy : governmentPolicy 1)
  (newMarket : LaborMarket)
  (newHealthcare : HealthcareMarket) :
  (newMarket.equilibriumWage > initialMarket.equilibriumWage) ∧
  (newHealthcare.equilibriumPrice < initialHealthcare.equilibriumPrice) :=
sorry

end NUMINAMATH_CALUDE_labor_market_effects_l1703_170374


namespace NUMINAMATH_CALUDE_black_ball_prob_compare_l1703_170396

-- Define the number of balls in each box
def box_a_red : ℕ := 40
def box_a_black : ℕ := 10
def box_b_red : ℕ := 60
def box_b_black : ℕ := 40
def box_b_white : ℕ := 50

-- Define the total number of balls in each box
def total_a : ℕ := box_a_red + box_a_black
def total_b : ℕ := box_b_red + box_b_black + box_b_white

-- Define the probabilities of drawing a black ball from each box
def prob_a : ℚ := box_a_black / total_a
def prob_b : ℚ := box_b_black / total_b

-- Theorem statement
theorem black_ball_prob_compare : prob_b > prob_a := by
  sorry

end NUMINAMATH_CALUDE_black_ball_prob_compare_l1703_170396


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l1703_170344

theorem quadratic_equation_transformation (x : ℝ) :
  x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l1703_170344


namespace NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l1703_170320

noncomputable def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

theorem min_value_and_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f a b c x ≥ 5) 
  (hex : ∃ x, f a b c x = 5) :
  (a + b + c = 5) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 1/a' + 1/b' + 1/c' ≥ 9/5) ∧
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 1/b' + 1/c' = 9/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l1703_170320


namespace NUMINAMATH_CALUDE_sams_initial_dimes_l1703_170390

theorem sams_initial_dimes (initial_dimes final_dimes dimes_from_dad : ℕ) 
  (h1 : final_dimes = initial_dimes + dimes_from_dad)
  (h2 : final_dimes = 16)
  (h3 : dimes_from_dad = 7) : 
  initial_dimes = 9 := by
  sorry

end NUMINAMATH_CALUDE_sams_initial_dimes_l1703_170390


namespace NUMINAMATH_CALUDE_fifth_root_unity_product_l1703_170340

theorem fifth_root_unity_product (r : ℂ) (h1 : r^5 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_product_l1703_170340


namespace NUMINAMATH_CALUDE_product_of_numbers_l1703_170368

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 8) (h2 : x^2 + y^2 = 160) : x * y = 48 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1703_170368


namespace NUMINAMATH_CALUDE_sum_of_integers_l1703_170318

theorem sum_of_integers (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 16) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1703_170318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1703_170319

/-- Given an arithmetic sequence {aₙ}, prove that S₁₃ = 13 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence condition
  (∀ n : ℕ, S n = (n / 2) * (a 1 + a n)) →               -- sum formula
  a 3 + a 5 + 2 * a 10 = 4 →                             -- given condition
  S 13 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1703_170319


namespace NUMINAMATH_CALUDE_sand_collection_total_weight_l1703_170307

theorem sand_collection_total_weight (eden_buckets mary_buckets iris_buckets : ℕ) 
  (sand_weight_per_bucket : ℕ) :
  eden_buckets = 4 →
  mary_buckets = eden_buckets + 3 →
  iris_buckets = mary_buckets - 1 →
  sand_weight_per_bucket = 2 →
  (eden_buckets + mary_buckets + iris_buckets) * sand_weight_per_bucket = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_sand_collection_total_weight_l1703_170307


namespace NUMINAMATH_CALUDE_equation_solution_l1703_170356

theorem equation_solution : ∃! x : ℝ, (2 / (x - 3) = 3 / (x - 6)) ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1703_170356


namespace NUMINAMATH_CALUDE_seed_cost_calculation_l1703_170389

def seed_cost_2lb : ℝ := 44.68
def seed_amount : ℝ := 6

theorem seed_cost_calculation : 
  seed_amount * (seed_cost_2lb / 2) = 134.04 := by
  sorry

end NUMINAMATH_CALUDE_seed_cost_calculation_l1703_170389


namespace NUMINAMATH_CALUDE_a_minus_b_values_l1703_170339

theorem a_minus_b_values (a b : ℤ) (ha : |a| = 7) (hb : |b| = 5) (hab : a < b) :
  a - b = -12 ∨ a - b = -2 :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l1703_170339


namespace NUMINAMATH_CALUDE_count_valid_numbers_l1703_170371

/-- The set of digits that can be used to form the numbers -/
def digits : Finset Nat := {0, 1, 2, 3}

/-- A predicate that checks if a number is a four-digit even number -/
def is_valid_number (n : Nat) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ Even n

/-- A function that returns the digits of a number as a list -/
def number_to_digits (n : Nat) : List Nat :=
  sorry

/-- A predicate that checks if a number uses only the allowed digits without repetition -/
def uses_valid_digits (n : Nat) : Prop :=
  let d := number_to_digits n
  d.toFinset ⊆ digits ∧ d.length = 4 ∧ d.Nodup

/-- The set of all valid numbers according to the problem conditions -/
def valid_numbers : Finset Nat :=
  sorry

theorem count_valid_numbers : valid_numbers.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l1703_170371


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_10_l1703_170329

/-- The slope of the tangent line to y = x^2 + 3x at (2, 10) is 7 -/
theorem tangent_slope_at_2_10 : 
  let f (x : ℝ) := x^2 + 3*x
  let A : ℝ × ℝ := (2, 10)
  let slope := (deriv f) A.1
  slope = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_10_l1703_170329


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1703_170375

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, ((a - 8) * x > a - 8) ↔ (x < 1)) → (a < 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1703_170375


namespace NUMINAMATH_CALUDE_remainder_171_pow_2147_mod_52_l1703_170350

theorem remainder_171_pow_2147_mod_52 : ∃ k : ℕ, 171^2147 = 52 * k + 7 := by sorry

end NUMINAMATH_CALUDE_remainder_171_pow_2147_mod_52_l1703_170350


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l1703_170336

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically --/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c - d }

/-- Translates a parabola horizontally --/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * d + p.b, c := p.a * d^2 - p.b * d + p.c }

theorem parabola_translation_theorem :
  let original := Parabola.mk 3 0 0
  let down_3 := translate_vertical original 3
  let right_2 := translate_horizontal down_3 2
  right_2 = Parabola.mk 3 (-12) 9 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l1703_170336


namespace NUMINAMATH_CALUDE_dog_food_calculation_l1703_170328

theorem dog_food_calculation (num_dogs : ℕ) (food_per_dog : ℕ) (vacation_days : ℕ) :
  num_dogs = 4 →
  food_per_dog = 250 →
  vacation_days = 14 →
  (num_dogs * food_per_dog * vacation_days : ℕ) / 1000 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_calculation_l1703_170328
