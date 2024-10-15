import Mathlib

namespace NUMINAMATH_CALUDE_football_tournament_l3998_399854

theorem football_tournament (n : ℕ) (k : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (n * (n - 1)) / 2 + k * n = 77 →  -- Total matches equation
  2 * n = 14  -- Prove that the initial number of teams is 14
  := by sorry

end NUMINAMATH_CALUDE_football_tournament_l3998_399854


namespace NUMINAMATH_CALUDE_planar_graph_iff_euler_l3998_399880

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  v : ℕ  -- number of vertices
  g : ℕ  -- number of edges
  s : ℕ  -- number of faces

/-- Euler's formula for planar graphs states that v - g + s = 2 -/
def satisfiesEulersFormula (graph : PlanarGraph) : Prop :=
  graph.v - graph.g + graph.s = 2

/-- A planar graph can be constructed if and only if it satisfies Euler's formula -/
theorem planar_graph_iff_euler (graph : PlanarGraph) :
  ∃ (G : PlanarGraph), G.v = graph.v ∧ G.g = graph.g ∧ G.s = graph.s ↔ satisfiesEulersFormula graph :=
sorry

end NUMINAMATH_CALUDE_planar_graph_iff_euler_l3998_399880


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_n_binomial_coefficient_1000_1000_l3998_399867

theorem binomial_coefficient_n_n (n : ℕ) : Nat.choose n n = 1 := by
  sorry

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_n_binomial_coefficient_1000_1000_l3998_399867


namespace NUMINAMATH_CALUDE_problem_solution_l3998_399824

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3998_399824


namespace NUMINAMATH_CALUDE_small_circle_radius_l3998_399815

theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 →  -- radius of the large circle is 10 meters
  4 * (2 * r) = 2 * R →  -- four diameters of small circles equal the diameter of the large circle
  r = 2.5 :=  -- radius of each small circle is 2.5 meters
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l3998_399815


namespace NUMINAMATH_CALUDE_power_of_a_l3998_399890

theorem power_of_a (a b : ℝ) : b = Real.sqrt (3 - a) + Real.sqrt (a - 3) + 2 → a^b = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_a_l3998_399890


namespace NUMINAMATH_CALUDE_congruence_problem_l3998_399835

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -527 ≡ n [ZMOD 31] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3998_399835


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3998_399899

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 6 = 2 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3998_399899


namespace NUMINAMATH_CALUDE_log_half_inequality_condition_l3998_399842

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

theorem log_half_inequality_condition (x : ℝ) (hx : x ∈ Set.Ioo 0 (1/2)) :
  (∀ a : ℝ, a < 0 → log_half x > x + a) ∧
  ∃ a : ℝ, a ≥ 0 ∧ log_half x > x + a :=
by
  sorry

#check log_half_inequality_condition

end NUMINAMATH_CALUDE_log_half_inequality_condition_l3998_399842


namespace NUMINAMATH_CALUDE_jellybean_average_proof_l3998_399834

/-- Proves that the initial average number of jellybeans per bag was 117,
    given the conditions of the problem. -/
theorem jellybean_average_proof 
  (initial_bags : ℕ) 
  (new_bag_jellybeans : ℕ) 
  (average_increase : ℕ) 
  (h1 : initial_bags = 34)
  (h2 : new_bag_jellybeans = 362)
  (h3 : average_increase = 7) :
  ∃ (initial_average : ℕ),
    (initial_average * initial_bags + new_bag_jellybeans) / (initial_bags + 1) = 
    initial_average + average_increase ∧ 
    initial_average = 117 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_average_proof_l3998_399834


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l3998_399831

theorem square_difference_fourth_power : (7^2 - 6^2)^4 = 28561 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l3998_399831


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3998_399876

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the condition for OA and OB to be perpendicular
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Main theorem
theorem ellipse_intersection_theorem :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  C x₁ y₁ ∧ C x₂ y₂ ∧
  Line k x₁ y₁ ∧ Line k x₂ y₂ ∧
  Perpendicular x₁ y₁ x₂ y₂ →
  (k = 1/2 ∨ k = -1/2) ∧
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (4*Real.sqrt 65/17)^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3998_399876


namespace NUMINAMATH_CALUDE_compute_expression_l3998_399862

theorem compute_expression : 
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2/3) = 41.65 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l3998_399862


namespace NUMINAMATH_CALUDE_consecutive_decreasing_difference_l3998_399844

/-- Represents a three-digit number with consecutive decreasing digits -/
structure ConsecutiveDecreasingNumber where
  x : ℕ
  h1 : x ≥ 1
  h2 : x ≤ 7

/-- Calculates the value of a three-digit number given its digits -/
def number_value (n : ConsecutiveDecreasingNumber) : ℕ :=
  100 * (n.x + 2) + 10 * (n.x + 1) + n.x

/-- Calculates the value of the reversed three-digit number given its digits -/
def reversed_value (n : ConsecutiveDecreasingNumber) : ℕ :=
  100 * n.x + 10 * (n.x + 1) + (n.x + 2)

/-- Theorem stating that the difference between a three-digit number with consecutive 
    decreasing digits and its reverse is always 198 -/
theorem consecutive_decreasing_difference 
  (n : ConsecutiveDecreasingNumber) : 
  number_value n - reversed_value n = 198 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_decreasing_difference_l3998_399844


namespace NUMINAMATH_CALUDE_marble_probability_l3998_399851

theorem marble_probability (red green white blue : ℕ) 
  (h_red : red = 5)
  (h_green : green = 4)
  (h_white : white = 12)
  (h_blue : blue = 2) :
  let total := red + green + white + blue
  (red / total) * (blue / (total - 1)) = 5 / 253 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3998_399851


namespace NUMINAMATH_CALUDE_unwashed_shirts_l3998_399860

theorem unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) : 
  short_sleeve + long_sleeve - washed = 1 := by
  sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l3998_399860


namespace NUMINAMATH_CALUDE_range_of_a_l3998_399869

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3998_399869


namespace NUMINAMATH_CALUDE_sum_of_2008th_powers_l3998_399841

theorem sum_of_2008th_powers (a b c : ℝ) 
  (sum_eq_3 : a + b + c = 3) 
  (sum_squares_eq_3 : a^2 + b^2 + c^2 = 3) : 
  a^2008 + b^2008 + c^2008 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_2008th_powers_l3998_399841


namespace NUMINAMATH_CALUDE_derivative_of_f_l3998_399866

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt 2) * Real.log (Real.sqrt 2 * Real.tan x + Real.sqrt (1 + 2 * Real.tan x ^ 2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 1 / (Real.cos x ^ 2 * Real.sqrt (1 + 2 * Real.tan x ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3998_399866


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3998_399859

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ,
  x₁^2 - 4*x₁ + 2 = 0 →
  x₂^2 - 4*x₂ + 2 = 0 →
  x₁ + x₂ - x₁*x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3998_399859


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_non_negative_condition_two_roots_condition_l3998_399807

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2*a + 6

/-- Theorem for part I -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ≥ 4, ∀ y ≥ 4, x < y → f a x < f a y) ↔ a ≥ -3 :=
sorry

/-- Theorem for part II -/
theorem non_negative_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

/-- Theorem for part III -/
theorem two_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ -5/4 < a ∧ a < -1 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_non_negative_condition_two_roots_condition_l3998_399807


namespace NUMINAMATH_CALUDE_jeffs_remaining_laps_l3998_399897

/-- Given Jeff's swimming requirements and progress, calculate the remaining laps before his break. -/
theorem jeffs_remaining_laps (total_laps : ℕ) (saturday_laps : ℕ) (sunday_morning_laps : ℕ) 
  (h1 : total_laps = 98)
  (h2 : saturday_laps = 27)
  (h3 : sunday_morning_laps = 15) :
  total_laps - saturday_laps - sunday_morning_laps = 56 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_remaining_laps_l3998_399897


namespace NUMINAMATH_CALUDE_weight_difference_l3998_399895

def john_weight : ℕ := 81
def roy_weight : ℕ := 4

theorem weight_difference : john_weight - roy_weight = 77 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3998_399895


namespace NUMINAMATH_CALUDE_salary_increase_l3998_399857

-- Define the salary function
def salary (x : ℝ) : ℝ := 60 + 90 * x

-- State the theorem
theorem salary_increase (x : ℝ) :
  salary (x + 1) - salary x = 90 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3998_399857


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3998_399811

/-- Given two 2D vectors, find the value of k that makes one vector parallel to another. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (1, 2)) :
  ∃ k : ℝ, k = 1/4 ∧ 
  ∃ c : ℝ, c • (2 • a + b) = (1/2 • a + k • b) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3998_399811


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3998_399870

theorem complex_number_quadrant (z : ℂ) (h : z = 1 - 2*I) : 
  (z.re > 0 ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3998_399870


namespace NUMINAMATH_CALUDE_select_gloves_count_l3998_399837

/-- The number of ways to select 4 gloves from 5 pairs of gloves with exactly one pair of the same color -/
def select_gloves (n : ℕ) : ℕ :=
  let total_pairs := 5
  let select_size := 4
  let pair_combinations := Nat.choose total_pairs 1
  let remaining_gloves := 2 * (total_pairs - 1)
  let other_combinations := Nat.choose remaining_gloves 2
  let same_color_pair := Nat.choose (total_pairs - 1) 1
  pair_combinations * (other_combinations - same_color_pair)

/-- Theorem stating that the number of ways to select 4 gloves from 5 pairs of gloves 
    with exactly one pair of the same color is 120 -/
theorem select_gloves_count : select_gloves 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_gloves_count_l3998_399837


namespace NUMINAMATH_CALUDE_tan_five_pi_four_equals_one_l3998_399865

theorem tan_five_pi_four_equals_one : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_four_equals_one_l3998_399865


namespace NUMINAMATH_CALUDE_count_prime_in_sequence_count_1973_in_sequence_l3998_399894

def generate_sequence (steps : Nat) : List Nat :=
  sorry

def count_occurrences (n : Nat) (list : List Nat) : Nat :=
  sorry

def is_prime (n : Nat) : Prop :=
  sorry

theorem count_prime_in_sequence (p : Nat) (h : is_prime p) :
  count_occurrences p (generate_sequence 1973) = p - 1 :=
sorry

theorem count_1973_in_sequence :
  count_occurrences 1973 (generate_sequence 1973) = 1972 :=
sorry

end NUMINAMATH_CALUDE_count_prime_in_sequence_count_1973_in_sequence_l3998_399894


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l3998_399874

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l3998_399874


namespace NUMINAMATH_CALUDE_solve_for_y_l3998_399889

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 10) (h2 : x = 8) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3998_399889


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_l3998_399881

theorem range_of_x_plus_y (x y : Real) 
  (h1 : 0 ≤ y) (h2 : y ≤ x) (h3 : x ≤ π/2)
  (h4 : 4 * (Real.cos y)^2 + 4 * Real.cos x * Real.sin y - 4 * (Real.cos x)^2 ≤ 1) :
  (x + y ∈ Set.Icc 0 (π/6)) ∨ (x + y ∈ Set.Icc (5*π/6) π) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_l3998_399881


namespace NUMINAMATH_CALUDE_P_below_line_l3998_399875

/-- A line in 2D space represented by the equation 2x - y + 3 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The point P with coordinates (1, -1) -/
def P : Point := ⟨1, -1⟩

/-- A point is below the line if 2x - y + 3 > 0 -/
def is_below (p : Point) : Prop := 2 * p.x - p.y + 3 > 0

theorem P_below_line : is_below P := by
  sorry

end NUMINAMATH_CALUDE_P_below_line_l3998_399875


namespace NUMINAMATH_CALUDE_five_by_seven_domino_five_by_seven_minus_corner_domino_five_by_seven_minus_second_row_domino_six_by_six_tetromino_l3998_399828

-- Define the types of tiles
inductive Tile
| Domino    -- 2x1 tile
| Tetromino -- 4x1 tile

-- Define a board
structure Board :=
(rows : ℕ)
(cols : ℕ)
(removed_cells : List (ℕ × ℕ)) -- List of removed cells' coordinates

-- Define a function to check if a board can be tiled
def can_be_tiled (b : Board) (t : Tile) : Prop :=
  match t with
  | Tile.Domino    => sorry
  | Tile.Tetromino => sorry

-- Theorem 1: A 5x7 board cannot be tiled with dominoes
theorem five_by_seven_domino :
  ¬ can_be_tiled { rows := 5, cols := 7, removed_cells := [] } Tile.Domino :=
sorry

-- Theorem 2: A 5x7 board with bottom left corner removed can be tiled with dominoes
theorem five_by_seven_minus_corner_domino :
  can_be_tiled { rows := 5, cols := 7, removed_cells := [(1, 1)] } Tile.Domino :=
sorry

-- Theorem 3: A 5x7 board with leftmost cell on second row removed cannot be tiled with dominoes
theorem five_by_seven_minus_second_row_domino :
  ¬ can_be_tiled { rows := 5, cols := 7, removed_cells := [(2, 1)] } Tile.Domino :=
sorry

-- Theorem 4: A 6x6 board can be tiled with tetrominoes
theorem six_by_six_tetromino :
  can_be_tiled { rows := 6, cols := 6, removed_cells := [] } Tile.Tetromino :=
sorry

end NUMINAMATH_CALUDE_five_by_seven_domino_five_by_seven_minus_corner_domino_five_by_seven_minus_second_row_domino_six_by_six_tetromino_l3998_399828


namespace NUMINAMATH_CALUDE_added_number_proof_l3998_399887

theorem added_number_proof (n : ℕ) (original_avg new_avg : ℚ) (added_num : ℚ) : 
  n = 15 →
  original_avg = 17 →
  new_avg = 20 →
  (n : ℚ) * original_avg + added_num = (n + 1 : ℚ) * new_avg →
  added_num = 65 := by
  sorry

end NUMINAMATH_CALUDE_added_number_proof_l3998_399887


namespace NUMINAMATH_CALUDE_mashas_balls_l3998_399853

theorem mashas_balls (r w n p : ℕ) : 
  r + n * w = 101 →
  p * r + w = 103 →
  (r + w = 51 ∨ r + w = 68) :=
by sorry

end NUMINAMATH_CALUDE_mashas_balls_l3998_399853


namespace NUMINAMATH_CALUDE_pauls_lost_crayons_l3998_399846

/-- Given that Paul initially had 110 crayons, gave 90 crayons to his friends,
    and lost 322 more crayons than those he gave to his friends,
    prove that Paul lost 412 crayons. -/
theorem pauls_lost_crayons
  (initial_crayons : ℕ)
  (crayons_given : ℕ)
  (extra_lost_crayons : ℕ)
  (h1 : initial_crayons = 110)
  (h2 : crayons_given = 90)
  (h3 : extra_lost_crayons = 322)
  : crayons_given + extra_lost_crayons = 412 := by
  sorry

end NUMINAMATH_CALUDE_pauls_lost_crayons_l3998_399846


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3998_399850

theorem roots_of_quadratic_equation :
  let equation := fun (x : ℂ) => x^2 + 4
  ∃ (r₁ r₂ : ℂ), r₁ = -2*I ∧ r₂ = 2*I ∧ equation r₁ = 0 ∧ equation r₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3998_399850


namespace NUMINAMATH_CALUDE_trimmed_square_area_l3998_399836

/-- The area of a rectangle formed by trimming a square --/
theorem trimmed_square_area (original_side : ℝ) (trim1 : ℝ) (trim2 : ℝ) 
  (h1 : original_side = 18)
  (h2 : trim1 = 4)
  (h3 : trim2 = 3) :
  (original_side - trim1) * (original_side - trim2) = 210 := by
sorry

end NUMINAMATH_CALUDE_trimmed_square_area_l3998_399836


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3998_399814

theorem inequality_solution_set (x : ℝ) : 
  (((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4) ↔ 
  (x > -1/4 ∧ x < 0) ∨ (x ≥ 3/2 ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3998_399814


namespace NUMINAMATH_CALUDE_prob_less_than_one_third_l3998_399810

/-- The probability that a number randomly selected from (0, 1/2) is less than 1/3 is 2/3. -/
theorem prob_less_than_one_third : 
  ∀ (P : Set ℝ → ℝ) (Ω : Set ℝ),
    (∀ a b, a < b → P (Set.Ioo a b) = b - a) →  -- P is a uniform probability measure
    Ω = Set.Ioo 0 (1/2) →                       -- Ω is the interval (0, 1/2)
    P {x ∈ Ω | x < 1/3} / P Ω = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_one_third_l3998_399810


namespace NUMINAMATH_CALUDE_class_vision_most_suitable_l3998_399830

/-- Represents a survey option -/
inductive SurveyOption
  | SleepTimeNationwide
  | RiverWaterQuality
  | PocketMoneyCity
  | ClassVision

/-- Checks if a survey option is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.ClassVision => true
  | _ => false

/-- Theorem stating that investigating the vision of all classmates in a class
    is the most suitable for a comprehensive survey -/
theorem class_vision_most_suitable :
  isSuitableForComprehensiveSurvey SurveyOption.ClassVision ∧
  (∀ (option : SurveyOption),
    isSuitableForComprehensiveSurvey option →
    option = SurveyOption.ClassVision) :=
by
  sorry

#check class_vision_most_suitable

end NUMINAMATH_CALUDE_class_vision_most_suitable_l3998_399830


namespace NUMINAMATH_CALUDE_tens_digit_of_nine_to_1010_l3998_399858

theorem tens_digit_of_nine_to_1010 :
  (9 : ℕ) ^ 1010 % 100 = 1 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_nine_to_1010_l3998_399858


namespace NUMINAMATH_CALUDE_horner_V₃_eq_9_l3998_399856

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 - 3x^4 + 7x^3 - 9x^2 + 4x - 10 -/
def f : List ℚ := [2, -3, 7, -9, 4, -10]

/-- V₃ in Horner's method for f(x) at x = 2 -/
def V₃ : ℚ := horner [2, -3, 7] 2

theorem horner_V₃_eq_9 : V₃ = 9 := by
  sorry

end NUMINAMATH_CALUDE_horner_V₃_eq_9_l3998_399856


namespace NUMINAMATH_CALUDE_business_investment_problem_l3998_399803

/-- Proves that A's investment is 16000, given the conditions of the business problem -/
theorem business_investment_problem (b_investment c_investment : ℕ) 
  (b_profit : ℕ) (profit_difference : ℕ) :
  b_investment = 10000 →
  c_investment = 12000 →
  b_profit = 1400 →
  profit_difference = 560 →
  ∃ (a_investment : ℕ), 
    a_investment * b_profit = b_investment * (a_investment * b_profit / b_investment - c_investment * b_profit / b_investment + profit_difference) ∧ 
    a_investment = 16000 := by
  sorry

end NUMINAMATH_CALUDE_business_investment_problem_l3998_399803


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3998_399892

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  is_geometric : (a 3)^2 = a 1 * a 7

/-- The main theorem -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 3) / (seq.a 2 + seq.a 4) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3998_399892


namespace NUMINAMATH_CALUDE_max_sum_of_first_three_l3998_399883

theorem max_sum_of_first_three (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℕ) 
  (h_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ < x₆ ∧ x₆ < x₇)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ = 159) :
  (∀ y₁ y₂ y₃ : ℕ, y₁ < y₂ ∧ y₂ < y₃ ∧ 
    (∃ y₄ y₅ y₆ y₇ : ℕ, y₃ < y₄ ∧ y₄ < y₅ ∧ y₅ < y₆ ∧ y₆ < y₇ ∧
      y₁ + y₂ + y₃ + y₄ + y₅ + y₆ + y₇ = 159) →
    y₁ + y₂ + y₃ ≤ 61) ∧
  (x₁ + x₂ + x₃ = 61) := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_first_three_l3998_399883


namespace NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l3998_399888

open Real

/-- Triangle ABC with given properties --/
structure Triangle where
  A : ℝ  -- Angle A in radians
  b : ℝ  -- Side length b
  area : ℝ  -- Area of the triangle

/-- The area of the circumcircle of a triangle --/
def circumcircle_area (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the area of the circumcircle for the given triangle --/
theorem circumcircle_area_of_special_triangle :
  let t : Triangle := {
    A := π/4,  -- 45° in radians
    b := 2 * sqrt 2,
    area := 1
  }
  circumcircle_area t = 5*π/2 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_special_triangle_l3998_399888


namespace NUMINAMATH_CALUDE_equation_solution_l3998_399838

theorem equation_solution :
  ∀ x : ℝ, (Real.sqrt (9 * x - 2) + 18 / Real.sqrt (9 * x - 2) = 11) ↔ (x = 83 / 9 ∨ x = 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3998_399838


namespace NUMINAMATH_CALUDE_line_through_points_l3998_399826

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points :
  ∀ (a b : ℝ),
  (line_equation a b 6 = 7) →
  (line_equation a b 10 = 23) →
  a + b = -13 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3998_399826


namespace NUMINAMATH_CALUDE_triangle_area_scaling_l3998_399819

theorem triangle_area_scaling (original_area new_area : ℝ) : 
  new_area = 54 → 
  new_area = 9 * original_area → 
  original_area = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_scaling_l3998_399819


namespace NUMINAMATH_CALUDE_congruent_triangles_equal_perimeters_l3998_399809

/-- Two triangles are congruent if they have the same shape and size -/
def CongruentTriangles (T1 T2 : Set (ℝ × ℝ)) : Prop := sorry

/-- The perimeter of a triangle is the sum of the lengths of its sides -/
def Perimeter (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- If two triangles are congruent, then their perimeters are equal -/
theorem congruent_triangles_equal_perimeters (T1 T2 : Set (ℝ × ℝ)) :
  CongruentTriangles T1 T2 → Perimeter T1 = Perimeter T2 := by sorry

end NUMINAMATH_CALUDE_congruent_triangles_equal_perimeters_l3998_399809


namespace NUMINAMATH_CALUDE_power_of_32_l3998_399829

theorem power_of_32 (n : ℕ) : 
  2^200 * 2^203 + 2^163 * 2^241 + 2^126 * 2^277 = 32^n → n = 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_32_l3998_399829


namespace NUMINAMATH_CALUDE_cubic_root_simplification_l3998_399820

theorem cubic_root_simplification (s : ℝ) : s = 1 / (2 - Real.rpow 3 (1/3)) → s = 2 + Real.rpow 3 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_simplification_l3998_399820


namespace NUMINAMATH_CALUDE_x_values_l3998_399840

theorem x_values (A : Set ℝ) (x : ℝ) (h1 : A = {0, 1, x^2 - 5*x}) (h2 : -4 ∈ A) :
  x = 1 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_x_values_l3998_399840


namespace NUMINAMATH_CALUDE_cookie_markup_is_twenty_percent_l3998_399817

/-- The percentage markup on cookies sold by Joe -/
def percentage_markup (num_cookies : ℕ) (total_earned : ℚ) (cost_per_cookie : ℚ) : ℚ :=
  ((total_earned / num_cookies.cast) / cost_per_cookie - 1) * 100

/-- Theorem stating that the percentage markup is 20% given the problem conditions -/
theorem cookie_markup_is_twenty_percent :
  let num_cookies : ℕ := 50
  let total_earned : ℚ := 60
  let cost_per_cookie : ℚ := 1
  percentage_markup num_cookies total_earned cost_per_cookie = 20 := by
sorry

end NUMINAMATH_CALUDE_cookie_markup_is_twenty_percent_l3998_399817


namespace NUMINAMATH_CALUDE_apple_preference_percentage_l3998_399845

theorem apple_preference_percentage (total_responses : ℕ) (apple_responses : ℕ) 
  (h1 : total_responses = 300) (h2 : apple_responses = 70) :
  (apple_responses : ℚ) / (total_responses : ℚ) * 100 = 23 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l3998_399845


namespace NUMINAMATH_CALUDE_single_elimination_games_l3998_399871

/-- The number of games required to determine a champion in a single-elimination tournament -/
def gamesRequired (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with n players, 
    the number of games required to determine a champion is n - 1 -/
theorem single_elimination_games (n : ℕ) (h : n > 0) : 
  gamesRequired n = n - 1 := by sorry

end NUMINAMATH_CALUDE_single_elimination_games_l3998_399871


namespace NUMINAMATH_CALUDE_min_value_sum_ratios_l3998_399816

theorem min_value_sum_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (b / a) ≥ 4 ∧
  ((a / b) + (b / c) + (c / a) + (b / a) = 4 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_ratios_l3998_399816


namespace NUMINAMATH_CALUDE_only_one_divisible_l3998_399852

theorem only_one_divisible (n : ℕ+) : (3^(n : ℕ) + 1) % (n : ℕ)^2 = 0 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divisible_l3998_399852


namespace NUMINAMATH_CALUDE_log_inequality_l3998_399832

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality (a x₁ x₂ : ℝ) (ha : a > 0 ∧ a ≠ 1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) :
  (a > 1 → (f a x₁ + f a x₂) / 2 ≤ f a ((x₁ + x₂) / 2)) ∧
  (0 < a ∧ a < 1 → (f a x₁ + f a x₂) / 2 ≥ f a ((x₁ + x₂) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l3998_399832


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_equality_condition_l3998_399843

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 → a + 2 * b ≤ x + 2 * y :=
by sorry

theorem min_value_is_2_sqrt_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  a + 2 * b = 2 * Real.sqrt 2 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_equality_condition_l3998_399843


namespace NUMINAMATH_CALUDE_number_problem_l3998_399873

theorem number_problem : ∃ x : ℝ, x / 100 = 31.76 + 0.28 ∧ x = 3204 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3998_399873


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3998_399885

theorem min_sum_of_squares (x y : ℝ) (h : (x + 8) * (y - 8) = 0) :
  ∃ (min : ℝ), min = 128 ∧ ∀ (a b : ℝ), (a + 8) * (b - 8) = 0 → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3998_399885


namespace NUMINAMATH_CALUDE_kitchen_width_l3998_399878

/-- Calculates the width of a rectangular kitchen given its dimensions and painting information. -/
theorem kitchen_width (length height : ℝ) (total_painted_area : ℝ) : 
  length = 12 ∧ 
  height = 10 ∧ 
  total_painted_area = 1680 → 
  (total_painted_area / 3) = 2 * (length * height + height * (total_painted_area / (3 * height) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_kitchen_width_l3998_399878


namespace NUMINAMATH_CALUDE_sequence_a_closed_form_l3998_399868

def sequence_a : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * sequence_a (n + 2) - 4 * (n + 3) * sequence_a (n + 1) + (4 * (n + 3) - 8) * sequence_a n

theorem sequence_a_closed_form (n : ℕ) : sequence_a n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_closed_form_l3998_399868


namespace NUMINAMATH_CALUDE_horner_rule_v3_value_l3998_399872

def horner_v3 (a b c d e x : ℝ) : ℝ := (((x + a) * x + b) * x + c)

theorem horner_rule_v3_value :
  let f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1
  let x : ℝ := 2
  horner_v3 2 1 (-3) (-1) 0 x = 15 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_v3_value_l3998_399872


namespace NUMINAMATH_CALUDE_different_color_prob_l3998_399822

def bag_prob (p_red_red p_white_white : ℚ) : Prop :=
  p_red_red = 2/15 ∧ p_white_white = 1/3

theorem different_color_prob (p_red_red p_white_white : ℚ) 
  (h : bag_prob p_red_red p_white_white) : 
  1 - (p_red_red + p_white_white) = 8/15 :=
sorry

end NUMINAMATH_CALUDE_different_color_prob_l3998_399822


namespace NUMINAMATH_CALUDE_line_equation_proof_l3998_399821

/-- The parabola y^2 = (5/2)x -/
def parabola (x y : ℝ) : Prop := y^2 = (5/2) * x

/-- Point O is the origin (0,0) -/
def O : ℝ × ℝ := (0, 0)

/-- Point through which the line passes -/
def P : ℝ × ℝ := (2, 1)

/-- Predicate to check if a point is on the line -/
def on_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

/-- Two points are perpendicular with respect to the origin -/
def perpendicular_to_origin (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

theorem line_equation_proof :
  ∃ (A B : ℝ × ℝ),
    A ≠ O ∧ B ≠ O ∧
    A ≠ B ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    perpendicular_to_origin A B ∧
    on_line A.1 A.2 ∧
    on_line B.1 B.2 ∧
    on_line P.1 P.2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3998_399821


namespace NUMINAMATH_CALUDE_min_value_and_max_t_l3998_399898

/-- Given a > 0, b > 0, and f(x) = |x + a| + |2x - b| with a minimum value of 1 -/
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

theorem min_value_and_max_t (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) (hmin_exists : ∃ x, f a b x = 1) :
  (2*a + b = 2) ∧ 
  (∀ t, (∀ a b, a > 0 → b > 0 → a + 2*b ≥ t*a*b) → t ≤ 9/2) ∧
  (∃ t, t = 9/2 ∧ ∀ a b, a > 0 → b > 0 → a + 2*b ≥ t*a*b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_t_l3998_399898


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3998_399891

theorem sqrt_product_equality : Real.sqrt 50 * Real.sqrt 18 * Real.sqrt 8 = 60 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3998_399891


namespace NUMINAMATH_CALUDE_positive_reals_inequalities_l3998_399801

theorem positive_reals_inequalities (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 1) : 
  x + y - 4*x*y ≥ 0 ∧ 1/x + 4/(1+y) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequalities_l3998_399801


namespace NUMINAMATH_CALUDE_absolute_value_v_l3998_399848

theorem absolute_value_v (u v : ℂ) : 
  u * v = 20 - 15 * I → Complex.abs u = 5 → Complex.abs v = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_v_l3998_399848


namespace NUMINAMATH_CALUDE_book_reading_time_l3998_399877

/-- The number of weeks required to read a book -/
def weeks_to_read (total_pages : ℕ) (pages_per_week : ℕ) : ℕ :=
  (total_pages + pages_per_week - 1) / pages_per_week

theorem book_reading_time : 
  let total_pages : ℕ := 2100
  let pages_per_day1 : ℕ := 100
  let pages_per_day2 : ℕ := 150
  let days_type1 : ℕ := 3
  let days_type2 : ℕ := 2
  let pages_per_week : ℕ := pages_per_day1 * days_type1 + pages_per_day2 * days_type2
  weeks_to_read total_pages pages_per_week = 4 := by
sorry

end NUMINAMATH_CALUDE_book_reading_time_l3998_399877


namespace NUMINAMATH_CALUDE_recliner_sales_increase_l3998_399800

theorem recliner_sales_increase 
  (price_reduction : ℝ) 
  (gross_increase : ℝ) 
  (sales_increase : ℝ) : 
  price_reduction = 0.2 → 
  gross_increase = 0.4400000000000003 → 
  sales_increase = (1 + gross_increase) / (1 - price_reduction) - 1 →
  sales_increase = 0.8 := by
sorry

end NUMINAMATH_CALUDE_recliner_sales_increase_l3998_399800


namespace NUMINAMATH_CALUDE_root_approximation_l3998_399839

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem root_approximation (root : ℕ+) :
  (f root = 0) →
  (f 1 = -2) →
  (f 1.5 = 0.625) →
  (f 1.25 = -0.984) →
  (f 1.375 = -0.260) →
  (f 1.4375 = 0.162) →
  (f 1.40625 = -0.054) →
  ∃ x : ℝ, x ∈ (Set.Ioo 1.375 1.4375) ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_approximation_l3998_399839


namespace NUMINAMATH_CALUDE_monster_family_eyes_l3998_399863

/-- A monster family with a specific number of eyes for each member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  kid_eyes : ℕ

/-- Calculate the total number of eyes in a monster family -/
def total_eyes (family : MonsterFamily) : ℕ :=
  family.mom_eyes + family.dad_eyes + family.num_kids * family.kid_eyes

/-- Theorem: The total number of eyes in the given monster family is 16 -/
theorem monster_family_eyes :
  ∃ (family : MonsterFamily),
    family.mom_eyes = 1 ∧
    family.dad_eyes = 3 ∧
    family.num_kids = 3 ∧
    family.kid_eyes = 4 ∧
    total_eyes family = 16 := by
  sorry

end NUMINAMATH_CALUDE_monster_family_eyes_l3998_399863


namespace NUMINAMATH_CALUDE_jen_shooting_game_times_l3998_399896

theorem jen_shooting_game_times (shooting_cost carousel_cost russel_rides total_tickets : ℕ) 
  (h1 : shooting_cost = 5)
  (h2 : carousel_cost = 3)
  (h3 : russel_rides = 3)
  (h4 : total_tickets = 19) :
  ∃ (jen_times : ℕ), jen_times * shooting_cost + russel_rides * carousel_cost = total_tickets ∧ jen_times = 2 := by
  sorry

end NUMINAMATH_CALUDE_jen_shooting_game_times_l3998_399896


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l3998_399823

theorem watch_loss_percentage (CP : ℝ) (SP : ℝ) : 
  CP = 1357.142857142857 →
  SP + 190 = CP * (1 + 4 / 100) →
  (CP - SP) / CP * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_l3998_399823


namespace NUMINAMATH_CALUDE_quadratic_sum_l3998_399847

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), quadratic a b c y ≥ quadratic a b c x) ∧  -- minimum exists
  (quadratic a b c 3 = 0) ∧  -- passes through (3,0)
  (quadratic a b c 7 = 0) ∧  -- passes through (7,0)
  (∀ (x : ℝ), quadratic a b c x ≥ 36) ∧  -- minimum value is 36
  (∃ (x : ℝ), quadratic a b c x = 36)  -- minimum value is achieved
  →
  a + b + c = -108 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3998_399847


namespace NUMINAMATH_CALUDE_book_purchase_problem_l3998_399833

theorem book_purchase_problem :
  ∀ (total_A total_B only_A only_B both : ℕ),
    total_A = 2 * total_B →
    both = 500 →
    both = 2 * only_B →
    total_A = only_A + both →
    total_B = only_B + both →
    only_A = 1000 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l3998_399833


namespace NUMINAMATH_CALUDE_six_foldable_configurations_l3998_399864

/-- Represents a square in the puzzle -/
inductive Square
| A | B | C | D | E | F | G | H

/-- Represents the T-shaped figure -/
structure TShape :=
  (squares : Finset Square)
  (h_count : squares.card = 4)

/-- Represents a configuration of the puzzle -/
structure Configuration :=
  (base : TShape)
  (added : Square)

/-- Predicate to check if a configuration can be folded into a topless cubical box -/
def is_foldable (c : Configuration) : Prop :=
  sorry  -- Definition of foldability

/-- The main theorem statement -/
theorem six_foldable_configurations :
  ∃ (valid_configs : Finset Configuration),
    valid_configs.card = 6 ∧
    (∀ c ∈ valid_configs, is_foldable c) ∧
    (∀ c : Configuration, is_foldable c → c ∈ valid_configs) :=
  sorry

end NUMINAMATH_CALUDE_six_foldable_configurations_l3998_399864


namespace NUMINAMATH_CALUDE_problem_statement_l3998_399861

theorem problem_statement (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a*x - 2) ≥ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3998_399861


namespace NUMINAMATH_CALUDE_red_note_rows_l3998_399804

theorem red_note_rows (red_notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ) (total_notes : ℕ) :
  red_notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  total_notes = 100 →
  ∃ (rows : ℕ), rows * red_notes_per_row + rows * red_notes_per_row * blue_notes_per_red + additional_blue_notes = total_notes ∧ rows = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_red_note_rows_l3998_399804


namespace NUMINAMATH_CALUDE_f_composition_of_three_l3998_399855

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_three : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l3998_399855


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3998_399808

/-- Given two parallel vectors a and b in R³, where a = (2, -1, 2) and b = (-4, 2, x),
    prove that x = -4. -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ × ℝ) (x : ℝ) :
  a = (2, -1, 2) →
  b = (-4, 2, x) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  x = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3998_399808


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3998_399879

/-- A quadratic function satisfying the given conditions -/
def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x > 2 * x + m) ↔ m < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3998_399879


namespace NUMINAMATH_CALUDE_fifth_power_minus_fifth_power_equals_sixteen_product_l3998_399805

theorem fifth_power_minus_fifth_power_equals_sixteen_product (m n : ℤ) :
  m^5 - n^5 = 16*m*n ↔ (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
sorry

end NUMINAMATH_CALUDE_fifth_power_minus_fifth_power_equals_sixteen_product_l3998_399805


namespace NUMINAMATH_CALUDE_typists_productivity_l3998_399886

/-- Given that 20 typists can type 48 letters in 20 minutes, 
    prove that 30 typists can type 216 letters in 1 hour at the same rate. -/
theorem typists_productivity (typists_base : ℕ) (letters_base : ℕ) (minutes_base : ℕ)
  (typists_new : ℕ) (minutes_new : ℕ) :
  typists_base = 20 →
  letters_base = 48 →
  minutes_base = 20 →
  typists_new = 30 →
  minutes_new = 60 →
  (typists_new * letters_base * minutes_new) / (typists_base * minutes_base) = 216 :=
by sorry

end NUMINAMATH_CALUDE_typists_productivity_l3998_399886


namespace NUMINAMATH_CALUDE_books_sold_and_remaining_l3998_399825

/-- Given that a person sells 45 books and has 6 books remaining, prove that they initially had 51 books. -/
theorem books_sold_and_remaining (books_sold : ℕ) (books_remaining : ℕ) : 
  books_sold = 45 → books_remaining = 6 → books_sold + books_remaining = 51 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_and_remaining_l3998_399825


namespace NUMINAMATH_CALUDE_interval_change_l3998_399813

/-- Represents the interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℚ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem interval_change (total_time : ℕ) :
  total_time = 2 * 21 →
  interval 2 total_time = 21 →
  interval 3 total_time = 14 := by
  sorry

#eval interval 3 42  -- Should output 14

end NUMINAMATH_CALUDE_interval_change_l3998_399813


namespace NUMINAMATH_CALUDE_friends_who_bought_is_five_l3998_399893

/-- The number of pencils in one color box -/
def pencils_per_box : ℕ := 7

/-- The total number of pencils -/
def total_pencils : ℕ := 42

/-- The number of color boxes Chloe has -/
def chloe_boxes : ℕ := 1

/-- Calculate the number of friends who bought the color box -/
def friends_who_bought : ℕ :=
  (total_pencils - chloe_boxes * pencils_per_box) / pencils_per_box

theorem friends_who_bought_is_five : friends_who_bought = 5 := by
  sorry

end NUMINAMATH_CALUDE_friends_who_bought_is_five_l3998_399893


namespace NUMINAMATH_CALUDE_log_equation_solution_l3998_399818

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3998_399818


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l3998_399812

theorem gold_coin_distribution (x y : ℤ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25*(x - y)) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l3998_399812


namespace NUMINAMATH_CALUDE_exists_cutting_method_for_person_to_fit_l3998_399884

/-- Represents a sheet of paper -/
structure Sheet :=
  (length : ℝ)
  (width : ℝ)
  (thickness : ℝ)

/-- Represents a person -/
structure Person :=
  (height : ℝ)
  (width : ℝ)

/-- Represents a cutting method -/
structure CuttingMethod :=
  (cuts : List (ℝ × ℝ))  -- List of cut coordinates

/-- Represents the result of applying a cutting method to a sheet -/
def apply_cutting_method (s : Sheet) (cm : CuttingMethod) : ℝ := sorry

/-- Determines if a person can fit through an opening -/
def can_fit_through (p : Person) (opening_size : ℝ) : Prop := sorry

/-- Main theorem: There exists a cutting method that creates an opening large enough for a person -/
theorem exists_cutting_method_for_person_to_fit (s : Sheet) (p : Person) : 
  ∃ (cm : CuttingMethod), can_fit_through p (apply_cutting_method s cm) :=
sorry

end NUMINAMATH_CALUDE_exists_cutting_method_for_person_to_fit_l3998_399884


namespace NUMINAMATH_CALUDE_exists_two_equal_types_l3998_399806

/-- Represents the types of sweets -/
inductive SweetType
  | Blackberry
  | Coconut
  | Chocolate

/-- Represents the number of sweets for each type -/
structure Sweets where
  blackberry : Nat
  coconut : Nat
  chocolate : Nat

/-- The initial number of sweets -/
def initialSweets : Sweets :=
  { blackberry := 7, coconut := 6, chocolate := 3 }

/-- The number of sweets Sofia eats -/
def eatenSweets : Nat := 2

/-- Checks if two types of sweets have the same number -/
def hasTwoEqualTypes (s : Sweets) : Prop :=
  (s.blackberry = s.coconut) ∨ (s.blackberry = s.chocolate) ∨ (s.coconut = s.chocolate)

/-- Theorem: It's possible for grandmother to receive the same number of sweets for two varieties -/
theorem exists_two_equal_types :
  ∃ (finalSweets : Sweets),
    finalSweets.blackberry + finalSweets.coconut + finalSweets.chocolate =
      initialSweets.blackberry + initialSweets.coconut + initialSweets.chocolate - eatenSweets ∧
    finalSweets.blackberry ≤ initialSweets.blackberry ∧
    finalSweets.coconut ≤ initialSweets.coconut ∧
    finalSweets.chocolate ≤ initialSweets.chocolate ∧
    hasTwoEqualTypes finalSweets :=
  sorry

end NUMINAMATH_CALUDE_exists_two_equal_types_l3998_399806


namespace NUMINAMATH_CALUDE_square_side_length_l3998_399882

/-- Given a square with diagonal length 4, prove that its side length is 2√2 -/
theorem square_side_length (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s^2 + s^2 = d^2 ∧ s = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3998_399882


namespace NUMINAMATH_CALUDE_highway_scenario_solution_l3998_399849

/-- Represents the scenario of a person walking along a highway with buses passing by -/
structure HighwayScenario where
  personSpeed : ℝ
  busSpeed : ℝ
  busDepartureInterval : ℝ
  oncomingBusInterval : ℝ
  overtakingBusInterval : ℝ
  busDistance : ℝ

/-- Checks if the given scenario satisfies all conditions -/
def isValidScenario (s : HighwayScenario) : Prop :=
  s.personSpeed > 0 ∧
  s.busSpeed > s.personSpeed ∧
  s.oncomingBusInterval * (s.busSpeed + s.personSpeed) = s.busDistance ∧
  s.overtakingBusInterval * (s.busSpeed - s.personSpeed) = s.busDistance ∧
  s.busDepartureInterval = s.busDistance / s.busSpeed

/-- The main theorem stating the unique solution to the highway scenario -/
theorem highway_scenario_solution :
  ∃! s : HighwayScenario, isValidScenario s ∧
    s.oncomingBusInterval = 4 ∧
    s.overtakingBusInterval = 6 ∧
    s.busDistance = 1200 ∧
    s.personSpeed = 50 ∧
    s.busSpeed = 250 ∧
    s.busDepartureInterval = 4.8 := by
  sorry


end NUMINAMATH_CALUDE_highway_scenario_solution_l3998_399849


namespace NUMINAMATH_CALUDE_trajectory_is_straight_line_l3998_399802

/-- The set of points P(x,y) satisfying the given equation forms a straight line -/
theorem trajectory_is_straight_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (x y : ℝ), Real.sqrt ((x - 1)^2 + (y - 1)^2) = |x + y - 2| / Real.sqrt 2 →
  a * x + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_straight_line_l3998_399802


namespace NUMINAMATH_CALUDE_chocolate_distribution_chocolate_problem_l3998_399827

theorem chocolate_distribution (initial_bars : ℕ) (sisters : ℕ) (father_ate : ℕ) (father_left : ℕ) : ℕ :=
  let total_people := sisters + 1
  let bars_per_person := initial_bars / total_people
  let bars_given_to_father := (bars_per_person / 2) * total_people
  let bars_father_had := bars_given_to_father - father_ate
  bars_father_had - father_left

theorem chocolate_problem : 
  chocolate_distribution 20 4 2 5 = 3 := by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_chocolate_problem_l3998_399827
