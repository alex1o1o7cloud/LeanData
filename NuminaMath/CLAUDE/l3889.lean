import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l3889_388923

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- State the theorem
theorem f_properties :
  (∀ x y, x < y ∧ ((x < 0 ∧ y ≤ 0) ∨ (x ≥ 2 ∧ y > 2)) → f x < f y) ∧
  (∃ δ₁ > 0, ∀ x, 0 < |x| ∧ |x| < δ₁ → f x < f 0) ∧
  (∃ δ₂ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ₂ → f x > f 2) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l3889_388923


namespace NUMINAMATH_CALUDE_three_positions_from_eight_people_l3889_388976

theorem three_positions_from_eight_people :
  (8 : ℕ).descFactorial 3 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_eight_people_l3889_388976


namespace NUMINAMATH_CALUDE_both_a_and_b_must_join_at_least_one_of_a_or_b_must_join_l3889_388987

-- Define the total number of doctors
def total_doctors : ℕ := 20

-- Define the number of doctors to be chosen
def team_size : ℕ := 5

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem for part (1)
theorem both_a_and_b_must_join : 
  combination (total_doctors - 2) (team_size - 2) = 816 := by sorry

-- Theorem for part (2)
theorem at_least_one_of_a_or_b_must_join : 
  2 * combination (total_doctors - 2) (team_size - 1) + 
  combination (total_doctors - 2) (team_size - 2) = 5661 := by sorry

end NUMINAMATH_CALUDE_both_a_and_b_must_join_at_least_one_of_a_or_b_must_join_l3889_388987


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3889_388948

theorem digit_equation_solution : ∃! (X : ℕ), X < 10 ∧ (510 : ℚ) / X = 40 + 3 * X :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3889_388948


namespace NUMINAMATH_CALUDE_roses_cut_proof_l3889_388933

/-- Given a vase with an initial number of roses and a final number of roses,
    calculate the number of roses that were added. -/
def roses_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 2 initial roses and 23 final roses,
    the number of roses added is 21. -/
theorem roses_cut_proof :
  roses_added 2 23 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_proof_l3889_388933


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l3889_388966

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : std_dev > 0

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 15 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 12 -/
theorem two_std_dev_below_mean (d : NormalDistribution) 
    (h1 : d.mean = 15) (h2 : d.std_dev = 1.5) : 
    value_n_std_dev_below d 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l3889_388966


namespace NUMINAMATH_CALUDE_tire_circumference_l3889_388980

/-- Given a tire rotating at 400 revolutions per minute on a car traveling at 144 km/h, 
    the circumference of the tire is 6 meters. -/
theorem tire_circumference (revolutions_per_minute : ℝ) (speed_km_per_hour : ℝ) 
  (h1 : revolutions_per_minute = 400) 
  (h2 : speed_km_per_hour = 144) : 
  let speed_m_per_minute : ℝ := speed_km_per_hour * 1000 / 60
  let circumference : ℝ := speed_m_per_minute / revolutions_per_minute
  circumference = 6 := by
sorry

end NUMINAMATH_CALUDE_tire_circumference_l3889_388980


namespace NUMINAMATH_CALUDE_number_equal_to_square_plus_opposite_l3889_388925

theorem number_equal_to_square_plus_opposite :
  ∀ x : ℝ, x = x^2 + (-x) → x = 0 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_number_equal_to_square_plus_opposite_l3889_388925


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3889_388943

theorem rectangular_prism_diagonal (length width height : ℝ) :
  length = 24 ∧ width = 16 ∧ height = 12 →
  Real.sqrt (length^2 + width^2 + height^2) = 4 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3889_388943


namespace NUMINAMATH_CALUDE_unique_solution_for_power_sum_l3889_388918

theorem unique_solution_for_power_sum : 
  ∃! (x y z : ℕ), x < y ∧ y < z ∧ 3^x + 3^y + 3^z = 179415 ∧ x = 4 ∧ y = 7 ∧ z = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_power_sum_l3889_388918


namespace NUMINAMATH_CALUDE_max_value_of_f_l3889_388988

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3889_388988


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3889_388927

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-4 : ℝ) 4 ∧
  f x = 16 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-4 : ℝ) 4 → f y ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3889_388927


namespace NUMINAMATH_CALUDE_smallest_three_digit_mod_congruence_l3889_388949

theorem smallest_three_digit_mod_congruence :
  ∃ n : ℕ, 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    45 * n % 315 = 90 ∧ 
    ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 45 * m % 315 = 90 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_mod_congruence_l3889_388949


namespace NUMINAMATH_CALUDE_square_area_ratio_l3889_388959

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 45) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3889_388959


namespace NUMINAMATH_CALUDE_intersection_M_N_l3889_388991

def M : Set ℝ := {x : ℝ | x^2 - 3*x = 0}
def N : Set ℝ := {-1, 1, 3}

theorem intersection_M_N : M ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3889_388991


namespace NUMINAMATH_CALUDE_quadratic_min_bound_l3889_388922

theorem quadratic_min_bound (p q α β : ℝ) (n : ℤ) (h : ℝ → ℝ) :
  (∀ x, h x = x^2 + p*x + q) →
  h α = 0 →
  h β = 0 →
  α ≠ β →
  (n : ℝ) < α →
  α < β →
  β < (n + 1 : ℝ) →
  min (h n) (h (n + 1)) < (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_min_bound_l3889_388922


namespace NUMINAMATH_CALUDE_wine_drinkers_l3889_388975

theorem wine_drinkers (soda : Nat) (both : Nat) (total : Nat) (h1 : soda = 22) (h2 : both = 17) (h3 : total = 31) :
  ∃ (wine : Nat), wine + soda - both = total ∧ wine = 26 := by
  sorry

end NUMINAMATH_CALUDE_wine_drinkers_l3889_388975


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3889_388911

theorem simplify_fraction_product : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3889_388911


namespace NUMINAMATH_CALUDE_alex_movie_count_l3889_388999

theorem alex_movie_count (total_different_movies : ℕ) 
  (movies_watched_together : ℕ) 
  (dalton_movies : ℕ) 
  (hunter_movies : ℕ) 
  (h1 : total_different_movies = 30)
  (h2 : movies_watched_together = 2)
  (h3 : dalton_movies = 7)
  (h4 : hunter_movies = 12) :
  total_different_movies - movies_watched_together - dalton_movies - hunter_movies = 9 := by
  sorry

end NUMINAMATH_CALUDE_alex_movie_count_l3889_388999


namespace NUMINAMATH_CALUDE_cube_volume_l3889_388944

/-- Given a box with dimensions 8 cm x 15 cm x 5 cm that can be built using a minimum of 60 cubes,
    the volume of each cube is 10 cm³. -/
theorem cube_volume (length width height min_cubes : ℕ) : 
  length = 8 → width = 15 → height = 5 → min_cubes = 60 →
  (length * width * height : ℚ) / min_cubes = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l3889_388944


namespace NUMINAMATH_CALUDE_remainder_17_45_mod_5_l3889_388916

theorem remainder_17_45_mod_5 : 17^45 % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_17_45_mod_5_l3889_388916


namespace NUMINAMATH_CALUDE_light_reflection_l3889_388903

/-- Given a light ray emitted from point P (6, 4) intersecting the x-axis at point Q (2, 0)
    and reflecting off the x-axis, prove that the equations of the lines on which the
    incident and reflected rays lie are x - y - 2 = 0 and x + y - 2 = 0, respectively. -/
theorem light_reflection (P Q : ℝ × ℝ) : 
  P = (6, 4) → Q = (2, 0) → 
  ∃ (incident_ray reflected_ray : ℝ → ℝ → Prop),
    (∀ x y, incident_ray x y ↔ x - y - 2 = 0) ∧
    (∀ x y, reflected_ray x y ↔ x + y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_l3889_388903


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l3889_388934

theorem expansion_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (m * x - 1)^5 = a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 33 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l3889_388934


namespace NUMINAMATH_CALUDE_prime_value_of_cubic_polynomial_l3889_388965

theorem prime_value_of_cubic_polynomial (n : ℕ) (a : ℚ) (b : ℕ) :
  b = n^3 - 4*a*n^2 - 12*n + 144 →
  Nat.Prime b →
  b = 11 := by
  sorry

end NUMINAMATH_CALUDE_prime_value_of_cubic_polynomial_l3889_388965


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l3889_388901

/-- An arithmetic sequence {a_n} satisfying given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 18)
  (h_sum2 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 := by
    sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l3889_388901


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3889_388996

theorem junk_mail_distribution (total_mail : ℕ) (houses : ℕ) (mail_per_house : ℕ) : 
  total_mail = 14 → houses = 7 → mail_per_house = total_mail / houses → mail_per_house = 2 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3889_388996


namespace NUMINAMATH_CALUDE_faulty_token_identifiable_l3889_388977

/-- Represents the possible outcomes of a weighing --/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a token with a nominal value and an actual weight --/
structure Token where
  nominal_value : ℕ
  actual_weight : ℕ

/-- Represents a set of four tokens --/
def TokenSet := (Token × Token × Token × Token)

/-- Represents a weighing action on the balance scale --/
def Weighing := (List Token) → (List Token) → WeighingResult

/-- Represents a strategy for determining the faulty token --/
def Strategy := TokenSet → Weighing → Weighing → Option Token

/-- States that exactly one token in the set has an incorrect weight --/
def ExactlyOneFaulty (ts : TokenSet) : Prop := sorry

/-- States that a strategy correctly identifies the faulty token --/
def StrategyCorrect (s : Strategy) : Prop := sorry

theorem faulty_token_identifiable :
  ∃ (s : Strategy), StrategyCorrect s :=
sorry

end NUMINAMATH_CALUDE_faulty_token_identifiable_l3889_388977


namespace NUMINAMATH_CALUDE_log_equation_solution_l3889_388971

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (5 * x) / Real.log 3 → x = (5 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3889_388971


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_divisible_by_360_l3889_388953

theorem smallest_k_for_sum_of_squares_divisible_by_360 :
  ∀ k : ℕ, k > 0 → (k * (k + 1) * (2 * k + 1)) % 2160 = 0 → k ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_divisible_by_360_l3889_388953


namespace NUMINAMATH_CALUDE_birds_joining_fence_l3889_388912

theorem birds_joining_fence (initial_storks initial_birds joining_birds : ℕ) : 
  initial_storks = 6 →
  initial_birds = 2 →
  initial_storks = initial_birds + joining_birds + 1 →
  joining_birds = 3 := by
sorry

end NUMINAMATH_CALUDE_birds_joining_fence_l3889_388912


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l3889_388986

theorem reciprocal_equation_solution (x : ℝ) : 
  3 - 1 / (4 * (1 - x)) = 2 * (1 / (4 * (1 - x))) → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l3889_388986


namespace NUMINAMATH_CALUDE_complex_power_simplification_l3889_388909

theorem complex_power_simplification :
  ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 2000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l3889_388909


namespace NUMINAMATH_CALUDE_group_a_trees_l3889_388950

theorem group_a_trees (group_a_plots : ℕ) (group_b_plots : ℕ) : 
  (4 * group_a_plots = 5 * group_b_plots) →  -- Both groups planted the same total number of trees
  (group_b_plots = group_a_plots - 3) →      -- Group B worked on 3 fewer plots than Group A
  (4 * group_a_plots = 60) :=                -- Group A planted 60 trees in total
by
  sorry

#check group_a_trees

end NUMINAMATH_CALUDE_group_a_trees_l3889_388950


namespace NUMINAMATH_CALUDE_max_ratio_theorem_l3889_388978

theorem max_ratio_theorem :
  ∃ (A B : ℝ), 
    (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y → x ≤ A ∧ y ≤ B) ∧
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y ∧ x = A) ∧
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y ∧ y = B) ∧
    A/B = 729/1024 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_theorem_l3889_388978


namespace NUMINAMATH_CALUDE_expression_evaluation_l3889_388915

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + 2*x + 2) / x * (y^2 + 2*y + 2) / y + (x^2 - 3*x + 2) / y * (y^2 - 3*y + 2) / x =
  2*x*y - x/y - y/x + 13 + 10/x + 4/y + 8/(x*y) := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3889_388915


namespace NUMINAMATH_CALUDE_initial_pigeons_l3889_388984

theorem initial_pigeons (initial final joined : ℕ) : 
  initial > 0 → 
  joined = 1 → 
  final = initial + joined → 
  final = 2 → 
  initial = 1 := by
sorry

end NUMINAMATH_CALUDE_initial_pigeons_l3889_388984


namespace NUMINAMATH_CALUDE_circle_theorem_part1_circle_theorem_part2_l3889_388928

-- Define the points A and B
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - 2 * y - 3 = 0

-- Define the circle equation for part 1
def circle_eq1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 10

-- Define the circle equation for part 2
def circle_eq2 (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 5

-- Part 1: Circle passing through A and B with center on the line
theorem circle_theorem_part1 :
  ∃ (center : ℝ × ℝ), 
    (line_eq center.1 center.2) ∧
    (∀ (x y : ℝ), circle_eq1 x y ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (A.1 - center.1)^2 + (A.2 - center.2)^2) ∧
      ((x - center.1)^2 + (y - center.2)^2 = (B.1 - center.1)^2 + (B.2 - center.2)^2)) :=
sorry

-- Part 2: Circle passing through A and B with minimum area
theorem circle_theorem_part2 :
  ∃ (center : ℝ × ℝ),
    (∀ (other_center : ℝ × ℝ),
      (A.1 - center.1)^2 + (A.2 - center.2)^2 ≤ (A.1 - other_center.1)^2 + (A.2 - other_center.2)^2) ∧
    (∀ (x y : ℝ), circle_eq2 x y ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (A.1 - center.1)^2 + (A.2 - center.2)^2) ∧
      ((x - center.1)^2 + (y - center.2)^2 = (B.1 - center.1)^2 + (B.2 - center.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_part1_circle_theorem_part2_l3889_388928


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l3889_388998

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 55)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * marks_per_correct - (total_sums - correct_sums) * marks_per_incorrect = total_marks ∧ 
    correct_sums = 23 := by
sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l3889_388998


namespace NUMINAMATH_CALUDE_whitney_whale_books_l3889_388952

/-- The number of whale books Whitney bought -/
def whale_books : ℕ := sorry

/-- The number of fish books Whitney bought -/
def fish_books : ℕ := 7

/-- The number of magazines Whitney bought -/
def magazines : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 11

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 1

/-- The total amount Whitney spent in dollars -/
def total_spent : ℕ := 179

/-- Theorem stating that Whitney bought 9 whale books -/
theorem whitney_whale_books : 
  whale_books * book_cost + fish_books * book_cost + magazines * magazine_cost = total_spent ∧
  whale_books = 9 := by sorry

end NUMINAMATH_CALUDE_whitney_whale_books_l3889_388952


namespace NUMINAMATH_CALUDE_volleyball_game_employees_l3889_388990

/-- Calculates the number of employees participating in a volleyball game given the number of managers, teams, and people per team. -/
def employees_participating (managers : ℕ) (teams : ℕ) (people_per_team : ℕ) : ℕ :=
  teams * people_per_team - managers

/-- Theorem stating that with 23 managers, 6 teams, and 5 people per team, there are 7 employees participating. -/
theorem volleyball_game_employees :
  employees_participating 23 6 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_game_employees_l3889_388990


namespace NUMINAMATH_CALUDE_black_white_difference_l3889_388992

/-- Represents a chessboard square color -/
inductive Color
| Black
| White

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)
  (startColor : Color)

/-- Counts the number of squares of a given color on the chessboard -/
def countSquares (board : Chessboard) (color : Color) : Nat :=
  sorry

theorem black_white_difference (board : Chessboard) :
  board.rows = 7 ∧ board.cols = 9 ∧ board.startColor = Color.Black →
  countSquares board Color.Black = countSquares board Color.White + 1 := by
  sorry

end NUMINAMATH_CALUDE_black_white_difference_l3889_388992


namespace NUMINAMATH_CALUDE_central_square_side_length_l3889_388930

/-- Given a rectangular hallway and total flooring area, calculates the side length of a central square area --/
theorem central_square_side_length 
  (hallway_length : ℝ) 
  (hallway_width : ℝ) 
  (total_area : ℝ) 
  (h1 : hallway_length = 6)
  (h2 : hallway_width = 4)
  (h3 : total_area = 124) :
  let hallway_area := hallway_length * hallway_width
  let central_area := total_area - hallway_area
  let side_length := Real.sqrt central_area
  side_length = 10 := by sorry

end NUMINAMATH_CALUDE_central_square_side_length_l3889_388930


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3889_388995

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 10 * b = 5 * c) 
  (h2 : 3 * c = 120) : 
  b = 20 := by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3889_388995


namespace NUMINAMATH_CALUDE_main_theorem_l3889_388905

/-- A nondecreasing function satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧
  (∀ x y : ℝ, f (f x) + f y = f (x + f y) + 1)

/-- The set of all solutions to the functional equation. -/
def SolutionSet : Set (ℝ → ℝ) :=
  {f | FunctionalEquation f ∧
    (∀ x, f x = 1) ∨
    (∀ x, f x = x + 1) ∨
    (∃ n : ℕ+, ∃ α : ℝ, 0 ≤ α ∧ α < 1 ∧ 
      (∀ x, f x = (1 / n) * ⌊n * x + α⌋ + 1)) ∨
    (∃ n : ℕ+, ∃ α : ℝ, 0 ≤ α ∧ α < 1 ∧ 
      (∀ x, f x = (1 / n) * ⌈n * x - α⌉ + 1))}

/-- The main theorem stating that the SolutionSet contains all solutions to the functional equation. -/
theorem main_theorem : ∀ f : ℝ → ℝ, FunctionalEquation f → f ∈ SolutionSet := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l3889_388905


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l3889_388958

/-- Given a store with coloring books, prove the number of shelves used -/
theorem coloring_book_shelves 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : initial_stock = 27)
  (h2 : books_sold = 6)
  (h3 : books_per_shelf = 7)
  : (initial_stock - books_sold) / books_per_shelf = 3 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l3889_388958


namespace NUMINAMATH_CALUDE_cats_not_liking_catnip_or_tuna_l3889_388937

/-- Given a pet shop with cats, prove the number of cats that don't like catnip or tuna -/
theorem cats_not_liking_catnip_or_tuna
  (total_cats : ℕ)
  (cats_like_catnip : ℕ)
  (cats_like_tuna : ℕ)
  (cats_like_both : ℕ)
  (h1 : total_cats = 80)
  (h2 : cats_like_catnip = 15)
  (h3 : cats_like_tuna = 60)
  (h4 : cats_like_both = 10) :
  total_cats - (cats_like_catnip + cats_like_tuna - cats_like_both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_cats_not_liking_catnip_or_tuna_l3889_388937


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3889_388967

theorem chinese_remainder_theorem_example :
  ∃! x : ℕ, x < 504 ∧ 
    x % 7 = 1 ∧
    x % 8 = 1 ∧
    x % 9 = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3889_388967


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l3889_388993

open Real

theorem function_inequality_implies_parameter_bound 
  (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = 1/2 * x^2 - 2*x)
  (hg : ∀ x, g x = a * log x)
  (hh : ∀ x, h x = f x - g x)
  (h_pos : ∀ x, x > 0)
  (h_ineq : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (h x₁ - h x₂) / (x₁ - x₂) > 2)
  : a ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l3889_388993


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3889_388945

/-- Given a triangle with angles 40°, 3x, and x + 10°, prove that x = 32.5° --/
theorem triangle_angle_problem (x : ℝ) : 
  (40 : ℝ) + 3 * x + (x + 10) = 180 → x = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3889_388945


namespace NUMINAMATH_CALUDE_max_radius_difference_l3889_388994

/-- The ellipse Γ in a 2D coordinate system -/
def Γ : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The first quadrant -/
def firstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

/-- Point P on the ellipse Γ in the first quadrant -/
def P : (ℝ × ℝ) :=
  sorry

/-- Left focus F₁ of the ellipse Γ -/
def F₁ : (ℝ × ℝ) :=
  sorry

/-- Right focus F₂ of the ellipse Γ -/
def F₂ : (ℝ × ℝ) :=
  sorry

/-- Point Q₁ where extended PF₁ intersects Γ -/
def Q₁ : (ℝ × ℝ) :=
  sorry

/-- Point Q₂ where extended PF₂ intersects Γ -/
def Q₂ : (ℝ × ℝ) :=
  sorry

/-- Radius r₁ of the inscribed circle in triangle PF₁Q₂ -/
def r₁ : ℝ :=
  sorry

/-- Radius r₂ of the inscribed circle in triangle PF₂Q₁ -/
def r₂ : ℝ :=
  sorry

/-- Theorem stating that the maximum value of r₁ - r₂ is 1/3 -/
theorem max_radius_difference :
  P ∈ Γ ∩ firstQuadrant →
  ∃ (max : ℝ), max = (1 : ℝ) / 3 ∧ ∀ (p : ℝ × ℝ), p ∈ Γ ∩ firstQuadrant → r₁ - r₂ ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_radius_difference_l3889_388994


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3889_388962

/-- A triangle with perimeter 60, two equal sides, and a difference of 21 between two sides has side lengths 27, 27, and 6. -/
theorem triangle_side_lengths :
  ∀ a b : ℝ,
  a > 0 ∧ b > 0 ∧
  2 * a + b = 60 ∧
  a - b = 21 →
  a = 27 ∧ b = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3889_388962


namespace NUMINAMATH_CALUDE_dog_water_consumption_l3889_388955

/-- Calculates the water needed for a dog during a hike given the total water capacity,
    human water consumption rate, and duration of the hike. -/
theorem dog_water_consumption
  (total_water : ℝ)
  (human_rate : ℝ)
  (duration : ℝ)
  (h1 : total_water = 4.8 * 1000) -- 4.8 L converted to ml
  (h2 : human_rate = 800)
  (h3 : duration = 4) :
  (total_water - human_rate * duration) / duration = 400 :=
by sorry

end NUMINAMATH_CALUDE_dog_water_consumption_l3889_388955


namespace NUMINAMATH_CALUDE_valid_numbers_l3889_388935

def is_valid_number (n : ℕ) : Prop :=
  30 ∣ n ∧ (Finset.card (Nat.divisors n) = 30)

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {11250, 4050, 7500, 1620, 1200, 720} := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l3889_388935


namespace NUMINAMATH_CALUDE_routes_on_3x2_grid_l3889_388902

/-- The number of routes on a grid from top-left to bottom-right -/
def num_routes (width : ℕ) (height : ℕ) : ℕ :=
  Nat.choose (width + height) height

/-- The theorem stating that the number of routes on a 3x2 grid is 10 -/
theorem routes_on_3x2_grid : num_routes 3 2 = 10 := by sorry

end NUMINAMATH_CALUDE_routes_on_3x2_grid_l3889_388902


namespace NUMINAMATH_CALUDE_total_ways_is_eight_l3889_388956

/-- The number of ways an individual can sign up -/
def sign_up_ways : ℕ := 2

/-- The number of individuals signing up -/
def num_individuals : ℕ := 3

/-- The total number of different ways all individuals can sign up -/
def total_ways : ℕ := sign_up_ways ^ num_individuals

/-- Theorem: The total number of different ways all individuals can sign up is 8 -/
theorem total_ways_is_eight : total_ways = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_eight_l3889_388956


namespace NUMINAMATH_CALUDE_color_theorem_l3889_388900

theorem color_theorem :
  ∃ (f : ℕ → ℕ),
    (∀ x, x ∈ Finset.range 2013 → f x ∈ Finset.range 7) ∧
    (∀ y, y ∈ Finset.range 7 → ∃ x ∈ Finset.range 2013, f x = y) ∧
    (∀ a b c, a ∈ Finset.range 2013 → b ∈ Finset.range 2013 → c ∈ Finset.range 2013 →
      a ≠ b → b ≠ c → a ≠ c → f a = f b → f b = f c →
        ¬(2014 ∣ (a * b * c)) ∧
        f ((a * b * c) % 2014) = f a) :=
by sorry

end NUMINAMATH_CALUDE_color_theorem_l3889_388900


namespace NUMINAMATH_CALUDE_shovel_time_closest_to_17_l3889_388940

/-- Represents the snow shoveling problem --/
structure SnowShoveling where
  /-- Initial shoveling rate in cubic yards per hour --/
  initial_rate : ℕ
  /-- Decrease in shoveling rate per hour --/
  rate_decrease : ℕ
  /-- Break duration in hours --/
  break_duration : ℚ
  /-- Hours of shoveling before a break --/
  hours_before_break : ℕ
  /-- Driveway width in yards --/
  driveway_width : ℕ
  /-- Driveway length in yards --/
  driveway_length : ℕ
  /-- Snow depth in yards --/
  snow_depth : ℕ

/-- Calculates the time taken to shovel the driveway clean, including breaks --/
def time_to_shovel (problem : SnowShoveling) : ℚ :=
  sorry

/-- Theorem stating that the time taken to shovel the driveway is closest to 17 hours --/
theorem shovel_time_closest_to_17 (problem : SnowShoveling) 
  (h1 : problem.initial_rate = 25)
  (h2 : problem.rate_decrease = 1)
  (h3 : problem.break_duration = 1/2)
  (h4 : problem.hours_before_break = 2)
  (h5 : problem.driveway_width = 5)
  (h6 : problem.driveway_length = 12)
  (h7 : problem.snow_depth = 4) :
  ∃ (t : ℚ), time_to_shovel problem = t ∧ abs (t - 17) < abs (t - 14) ∧ 
             abs (t - 17) < abs (t - 15) ∧ abs (t - 17) < abs (t - 16) ∧ 
             abs (t - 17) < abs (t - 18) :=
  sorry

end NUMINAMATH_CALUDE_shovel_time_closest_to_17_l3889_388940


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l3889_388981

/-- The number of noodles Daniel gave away -/
def noodles_given : ℕ := 12

/-- The number of noodles Daniel has now -/
def noodles_left : ℕ := 54

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℕ := noodles_given + noodles_left

theorem daniel_initial_noodles :
  initial_noodles = 66 := by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l3889_388981


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3889_388932

theorem angle_measure_proof :
  ∀ (A B : ℝ),
  A + B = 180 →
  A = 5 * B →
  A = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3889_388932


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3889_388908

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 10)
  (h_a12 : a 12 = 31) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3889_388908


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l3889_388985

theorem polynomial_sum_simplification (x : ℝ) : 
  (2*x^4 + 3*x^3 - 5*x^2 + 9*x - 8) + (-x^5 + x^4 - 2*x^3 + 4*x^2 - 6*x + 14) = 
  -x^5 + 3*x^4 + x^3 - x^2 + 3*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l3889_388985


namespace NUMINAMATH_CALUDE_area_to_paint_l3889_388936

-- Define the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_width : ℝ := 5
def door_height : ℝ := 2
def door_width : ℝ := 7

-- Define the theorem
theorem area_to_paint :
  wall_height * wall_length - (window_height * window_width + door_height * door_width) = 121 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_l3889_388936


namespace NUMINAMATH_CALUDE_conference_handshakes_eq_360_l3889_388969

/-- Represents the number of handshakes in a conference with specific groupings -/
def conference_handshakes (total : ℕ) (group_a : ℕ) (group_b1 : ℕ) (group_b2 : ℕ) : ℕ :=
  let handshakes_a_b1 := group_b1 * (group_a - group_a / 2)
  let handshakes_a_b2 := group_b2 * group_a
  let handshakes_b2 := group_b2 * (group_b2 - 1) / 2
  handshakes_a_b1 + handshakes_a_b2 + handshakes_b2

/-- The theorem stating that the number of handshakes in the given conference scenario is 360 -/
theorem conference_handshakes_eq_360 :
  conference_handshakes 40 25 5 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_eq_360_l3889_388969


namespace NUMINAMATH_CALUDE_complex_sum_real_imag_parts_l3889_388941

theorem complex_sum_real_imag_parts (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : 
  z.re + z.im = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_real_imag_parts_l3889_388941


namespace NUMINAMATH_CALUDE_trig_identity_l3889_388920

theorem trig_identity (θ a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sin θ)^4 / a + (Real.cos θ)^4 / b = 1 / (2 * (a + b)) →
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a + b)^2 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l3889_388920


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l3889_388904

/-- The time (in hours) taken by pipe B to empty the full cistern -/
def time_B : ℝ := 25

/-- The time (in hours) taken to fill the cistern when both pipes are opened -/
def time_both : ℝ := 99.99999999999999

/-- The time (in hours) taken by pipe A to fill the cistern -/
def time_A : ℝ := 20

/-- Theorem stating that the time taken by pipe A to fill the cistern is 20 hours -/
theorem pipe_A_fill_time :
  (1 / time_A - 1 / time_B) * time_both = 1 :=
sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l3889_388904


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l3889_388914

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16245775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l3889_388914


namespace NUMINAMATH_CALUDE_distance_before_collision_l3889_388929

/-- Theorem: Distance between boats 3 minutes before collision -/
theorem distance_before_collision
  (river_current : ℝ)
  (boat1_speed : ℝ)
  (boat2_speed : ℝ)
  (initial_distance : ℝ)
  (h1 : river_current = 2)
  (h2 : boat1_speed = 5)
  (h3 : boat2_speed = 25)
  (h4 : initial_distance = 20) :
  let relative_speed := (boat1_speed - river_current) + (boat2_speed - river_current)
  let time_before_collision : ℝ := 3 / 60
  let distance_covered := relative_speed * time_before_collision
  initial_distance - distance_covered = 1.3 := by
  sorry

#check distance_before_collision

end NUMINAMATH_CALUDE_distance_before_collision_l3889_388929


namespace NUMINAMATH_CALUDE_zenith_school_reading_fraction_l3889_388957

/-- Represents the student body at Zenith Middle School -/
structure StudentBody where
  total : ℕ
  enjoy_reading : ℕ
  dislike_reading : ℕ
  enjoy_and_express : ℕ
  enjoy_but_pretend_dislike : ℕ
  dislike_and_express : ℕ
  dislike_but_pretend_enjoy : ℕ

/-- The conditions of the problem -/
def zenith_school (s : StudentBody) : Prop :=
  s.total > 0 ∧
  s.enjoy_reading = (70 * s.total) / 100 ∧
  s.dislike_reading = s.total - s.enjoy_reading ∧
  s.enjoy_and_express = (70 * s.enjoy_reading) / 100 ∧
  s.enjoy_but_pretend_dislike = s.enjoy_reading - s.enjoy_and_express ∧
  s.dislike_and_express = (75 * s.dislike_reading) / 100 ∧
  s.dislike_but_pretend_enjoy = s.dislike_reading - s.dislike_and_express

/-- The theorem to be proved -/
theorem zenith_school_reading_fraction (s : StudentBody) :
  zenith_school s →
  (s.enjoy_but_pretend_dislike : ℚ) / (s.enjoy_but_pretend_dislike + s.dislike_and_express) = 21 / 43 := by
  sorry


end NUMINAMATH_CALUDE_zenith_school_reading_fraction_l3889_388957


namespace NUMINAMATH_CALUDE_logical_equivalence_l3889_388947

theorem logical_equivalence (P Q R : Prop) :
  (¬P ∧ ¬Q → ¬R) ↔ (R → P ∨ Q) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalence_l3889_388947


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3889_388931

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a - b) / c = (Real.sin B + Real.sin C) / (Real.sin B + Real.sin A) ∧
  a = Real.sqrt 7 ∧
  b = 2 * c

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  A = 2 * Real.pi / 3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3889_388931


namespace NUMINAMATH_CALUDE_power_two_divides_odd_power_minus_one_l3889_388926

theorem power_two_divides_odd_power_minus_one (k : ℕ) (h : Odd k) :
  ∀ n : ℕ, n ≥ 1 → (2^(n+2) : ℕ) ∣ k^(2^n) - 1 :=
by sorry

end NUMINAMATH_CALUDE_power_two_divides_odd_power_minus_one_l3889_388926


namespace NUMINAMATH_CALUDE_graveyard_skeletons_l3889_388939

/-- Represents the number of skeletons in the graveyard -/
def S : ℕ := sorry

/-- The number of bones in an adult woman's skeleton -/
def womanBones : ℕ := 20

/-- The number of bones in an adult man's skeleton -/
def manBones : ℕ := womanBones + 5

/-- The number of bones in a child's skeleton -/
def childBones : ℕ := womanBones / 2

/-- The total number of bones in the graveyard -/
def totalBones : ℕ := 375

theorem graveyard_skeletons :
  (S / 2 * womanBones + S / 4 * manBones + S / 4 * childBones = totalBones) →
  S = 20 := by sorry

end NUMINAMATH_CALUDE_graveyard_skeletons_l3889_388939


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3889_388907

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n^2 - 19*n + 95 = k^2) ↔ (n = 5 ∨ n = 14) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3889_388907


namespace NUMINAMATH_CALUDE_stamps_theorem_l3889_388921

/-- Given denominations 3, n, and n+1, this function checks if k cents can be formed -/
def can_form (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), k = 3 * a + n * b + (n + 1) * c

/-- The main theorem -/
theorem stamps_theorem :
  ∃! (n : ℕ), 
    n > 0 ∧ 
    (∀ (k : ℕ), k ≤ 115 → ¬(can_form n k)) ∧
    (∀ (k : ℕ), k > 115 → can_form n k) ∧
    n = 59 := by
  sorry

end NUMINAMATH_CALUDE_stamps_theorem_l3889_388921


namespace NUMINAMATH_CALUDE_tv_conditional_probability_l3889_388982

theorem tv_conditional_probability 
  (p_10000 : ℝ) 
  (p_15000 : ℝ) 
  (h1 : p_10000 = 0.80) 
  (h2 : p_15000 = 0.60) : 
  p_15000 / p_10000 = 0.75 := by
sorry

end NUMINAMATH_CALUDE_tv_conditional_probability_l3889_388982


namespace NUMINAMATH_CALUDE_deaf_to_blind_ratio_l3889_388924

theorem deaf_to_blind_ratio (total : ℕ) (deaf : ℕ) (h1 : total = 240) (h2 : deaf = 180) :
  (deaf : ℚ) / (total - deaf) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_deaf_to_blind_ratio_l3889_388924


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l3889_388997

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions --/
theorem maxwell_walking_speed :
  ∀ (maxwell_speed : ℝ),
    maxwell_speed > 0 →
    (4 * maxwell_speed + 18 = 34) →
    maxwell_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l3889_388997


namespace NUMINAMATH_CALUDE_no_zeros_of_g_l3889_388968

open Set
open Function
open Topology

theorem no_zeros_of_g (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ≠ 0, deriv f x + f x / x > 0) : 
  ∀ x ≠ 0, f x + 1 / x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeros_of_g_l3889_388968


namespace NUMINAMATH_CALUDE_five_in_range_of_quadratic_l3889_388906

theorem five_in_range_of_quadratic (b : ℝ) : ∃ x : ℝ, x^2 + b*x + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_in_range_of_quadratic_l3889_388906


namespace NUMINAMATH_CALUDE_count_valid_pairs_l3889_388983

def harmonic_mean (x y : ℕ+) : ℚ := 2 * (x * y) / (x + y)

def valid_pair (x y : ℕ+) : Prop :=
  x < y ∧ harmonic_mean x y = 1024

theorem count_valid_pairs : 
  ∃ (S : Finset (ℕ+ × ℕ+)), (∀ p ∈ S, valid_pair p.1 p.2) ∧ S.card = 9 ∧ 
  (∀ x y : ℕ+, valid_pair x y → (x, y) ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l3889_388983


namespace NUMINAMATH_CALUDE_odd_prime_product_probability_l3889_388963

/-- A standard die with six faces numbered from 1 to 6. -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of odd prime numbers on a standard die. -/
def OddPrimeOnDie : Finset ℕ := {3, 5}

/-- The number of times the die is rolled. -/
def NumRolls : ℕ := 8

/-- The probability of rolling an odd prime on a single roll of a standard die. -/
def SingleRollProbability : ℚ := (OddPrimeOnDie.card : ℚ) / (StandardDie.card : ℚ)

theorem odd_prime_product_probability :
  (SingleRollProbability ^ NumRolls : ℚ) = 1 / 6561 :=
sorry

end NUMINAMATH_CALUDE_odd_prime_product_probability_l3889_388963


namespace NUMINAMATH_CALUDE_yanni_paintings_l3889_388964

def painting_count : ℕ := 5

def square_feet_per_painting : List ℕ := [25, 25, 25, 80, 45]

theorem yanni_paintings :
  (painting_count = 5) ∧
  (square_feet_per_painting.length = painting_count) ∧
  (square_feet_per_painting.sum = 200) := by
  sorry

end NUMINAMATH_CALUDE_yanni_paintings_l3889_388964


namespace NUMINAMATH_CALUDE_brads_cookies_brads_cookies_solution_l3889_388913

theorem brads_cookies (total_cookies : ℕ) (greg_ate : ℕ) (leftover : ℕ) : ℕ :=
  let total_halves := total_cookies * 2
  let after_greg := total_halves - greg_ate
  after_greg - leftover

theorem brads_cookies_solution :
  brads_cookies 14 4 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_cookies_brads_cookies_solution_l3889_388913


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3889_388972

/-- Calculate compound interest for a fixed deposit -/
theorem compound_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℕ) 
  (h1 : principal = 50000) 
  (h2 : rate = 0.04) 
  (h3 : time = 3) : 
  (principal * (1 + rate)^time - principal) = (5 * (1 + 0.04)^3 - 5) * 10000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l3889_388972


namespace NUMINAMATH_CALUDE_initial_solution_strength_l3889_388989

/-- Proves that the initial solution strength is 60% given the problem conditions --/
theorem initial_solution_strength 
  (initial_volume : ℝ)
  (drained_volume : ℝ)
  (replacement_strength : ℝ)
  (final_strength : ℝ)
  (h1 : initial_volume = 50)
  (h2 : drained_volume = 35)
  (h3 : replacement_strength = 40)
  (h4 : final_strength = 46)
  (h5 : initial_volume - drained_volume + drained_volume = initial_volume)
  (h6 : (initial_volume - drained_volume) * (initial_strength / 100) + 
        drained_volume * (replacement_strength / 100) = 
        initial_volume * (final_strength / 100)) :
  initial_strength = 60 := by
  sorry

#check initial_solution_strength

end NUMINAMATH_CALUDE_initial_solution_strength_l3889_388989


namespace NUMINAMATH_CALUDE_largest_number_proof_l3889_388917

def is_valid_expression (expr : ℕ → ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    (a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 8 ∧ e = 8 ∧ f = 8) ∧
    (∀ n, expr n = n ∨ expr n = a ∨ expr n = b ∨ expr n = c ∨ expr n = d ∨ expr n = e ∨ expr n = f ∨
      ∃ (x y : ℕ), (expr n = expr x + expr y ∨ expr n = expr x - expr y ∨ 
                    expr n = expr x * expr y ∨ expr n = expr x / expr y ∨ 
                    expr n = expr x ^ expr y))

def largest_expression : ℕ → ℕ :=
  fun n => 3^(3^(3^(8^(8^8))))

theorem largest_number_proof :
  (is_valid_expression largest_expression) ∧
  (∀ expr, is_valid_expression expr → ∀ n, expr n ≤ largest_expression n) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3889_388917


namespace NUMINAMATH_CALUDE_complex_number_location_l3889_388919

theorem complex_number_location (z : ℂ) (h : z / (4 + 2*I) = I) :
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3889_388919


namespace NUMINAMATH_CALUDE_problem_solution_l3889_388942

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (a+2)*x + 4

-- Define the function g
def g (m x : ℝ) : ℝ := m*x + 5 - 2*m

theorem problem_solution :
  -- Part 1
  (∀ a : ℝ, 
    (a < 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | a ≤ x ∧ x ≤ 2}) ∧
    (a = 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | x = 2}) ∧
    (a > 2 → {x : ℝ | f a x ≤ 4 - 2*a} = {x : ℝ | 2 ≤ x ∧ x ≤ a})) ∧
  -- Part 2
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 1 4 → f a x + a + 1 ≥ 0) → a ≤ 4) ∧
  -- Part 3
  (∀ m : ℝ, 
    (∀ x₁ : ℝ, x₁ ∈ Set.Icc 1 4 → 
      ∃ x₂ : ℝ, x₂ ∈ Set.Icc 1 4 ∧ f 2 x₁ = g m x₂) →
    m ≤ -5/2 ∨ m ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3889_388942


namespace NUMINAMATH_CALUDE_wuzhen_conference_impact_l3889_388910

/-- Represents the cultural impact of the World Internet Conference in Wuzhen -/
structure CulturalImpact where
  promote_chinese_culture : Bool
  innovate_world_culture : Bool
  enhance_chinese_influence : Bool

/-- The World Internet Conference venue -/
def Wuzhen : String := "Wuzhen, China"

/-- Characteristics of Wuzhen -/
structure WuzhenCharacteristics where
  tradition_modernity_blend : Bool
  chinese_foreign_embrace : Bool

/-- The cultural impact of the World Internet Conference -/
def conference_impact (venue : String) (characteristics : WuzhenCharacteristics) : CulturalImpact :=
  { promote_chinese_culture := true,
    innovate_world_culture := true,
    enhance_chinese_influence := true }

/-- Theorem stating the cultural impact of the World Internet Conference in Wuzhen -/
theorem wuzhen_conference_impact :
  let venue := Wuzhen
  let characteristics := { tradition_modernity_blend := true, chinese_foreign_embrace := true }
  let impact := conference_impact venue characteristics
  impact.promote_chinese_culture ∧ impact.innovate_world_culture ∧ impact.enhance_chinese_influence :=
by
  sorry

end NUMINAMATH_CALUDE_wuzhen_conference_impact_l3889_388910


namespace NUMINAMATH_CALUDE_age_ratio_l3889_388951

def cody_age : ℕ := 14
def grandmother_age : ℕ := 84

theorem age_ratio : (grandmother_age : ℚ) / (cody_age : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l3889_388951


namespace NUMINAMATH_CALUDE_product_mod_fifteen_l3889_388961

theorem product_mod_fifteen : 59 * 67 * 78 ≡ 9 [ZMOD 15] := by sorry

end NUMINAMATH_CALUDE_product_mod_fifteen_l3889_388961


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_l3889_388938

theorem factor_w4_minus_81 (w : ℂ) : w^4 - 81 = (w-3)*(w+3)*(w-3*I)*(w+3*I) := by
  sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_l3889_388938


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_m_l3889_388970

/-- Given a hyperbola with equation y² + x²/m = 1 and asymptote y = ±(√3/3)x, prove that m = -3 -/
theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y : ℝ, y^2 + x^2/m = 1 → (y = (Real.sqrt 3)/3 * x ∨ y = -(Real.sqrt 3)/3 * x)) → 
  m = -3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_m_l3889_388970


namespace NUMINAMATH_CALUDE_smallest_num_rectangles_l3889_388946

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Whether two natural numbers are in the ratio 5:4 -/
def in_ratio_5_4 (a b : ℕ) : Prop := 5 * b = 4 * a

/-- The number of small rectangles needed to cover a larger rectangle -/
def num_small_rectangles (small large : Rectangle) : ℕ :=
  large.area / small.area

theorem smallest_num_rectangles :
  let small_rectangle : Rectangle := ⟨2, 3⟩
  ∃ (large_rectangle : Rectangle),
    in_ratio_5_4 large_rectangle.width large_rectangle.height ∧
    num_small_rectangles small_rectangle large_rectangle = 30 ∧
    ∀ (other_rectangle : Rectangle),
      in_ratio_5_4 other_rectangle.width other_rectangle.height →
      num_small_rectangles small_rectangle other_rectangle ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_num_rectangles_l3889_388946


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3889_388960

theorem consecutive_integers_sum (n : ℕ) (x : ℤ) : 
  (n > 0) → 
  (x + n - 1 = 9) → 
  (n * (2 * x + n - 1) / 2 = 24) → 
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3889_388960


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3889_388973

theorem arithmetic_sequence_common_difference_range (a : ℕ → ℝ) (d : ℝ) :
  (a 1 = -3) →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∀ n : ℕ, n ≥ 5 → a n > 0) →
  d ∈ Set.Ioo (3/4) 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l3889_388973


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3889_388974

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧ 
  a 1 = 8 ∧
  a 4 = a 3 * a 5

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 7 = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3889_388974


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3889_388954

/-- The length of the major axis of the ellipse x^2/49 + y^2/81 = 1 is 18 -/
theorem ellipse_major_axis_length : 
  let a := Real.sqrt (max 49 81)
  2 * a = 18 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3889_388954


namespace NUMINAMATH_CALUDE_product_difference_sum_l3889_388979

theorem product_difference_sum (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A * B = 72 →
  C * D = 72 →
  A - B = C + D →
  A = 18 := by
sorry

end NUMINAMATH_CALUDE_product_difference_sum_l3889_388979
