import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_205_between_14_and_15_l3559_355971

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_205_between_14_and_15_l3559_355971


namespace NUMINAMATH_CALUDE_unique_solution_k_l3559_355919

theorem unique_solution_k (k : ℝ) : 
  (∃! x : ℝ, (1 / (3 * x) = (k - x) / 8)) ↔ k = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_k_l3559_355919


namespace NUMINAMATH_CALUDE_max_value_squared_sum_max_value_squared_sum_achieved_l3559_355941

theorem max_value_squared_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
by sorry

theorem max_value_squared_sum_achieved (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ a^2 + b^2 + c^4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_squared_sum_max_value_squared_sum_achieved_l3559_355941


namespace NUMINAMATH_CALUDE_max_value_implies_m_l3559_355953

/-- The function f(x) = -x³ + 3x² + 9x + m has a maximum value of 20 on the interval [-2, 2] -/
def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

/-- The maximum value of f(x) on the interval [-2, 2] is 20 -/
def has_max_20 (m : ℝ) : Prop :=
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 2 ∧
  f x m = 20 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-2) 2 → f y m ≤ 20

theorem max_value_implies_m (m : ℝ) :
  has_max_20 m → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l3559_355953


namespace NUMINAMATH_CALUDE_sequence_a_l3559_355926

theorem sequence_a (a : ℕ → ℕ) (h : ∀ n, a (n + 1) = a n + n) :
  a 0 = 19 → a 1 = 20 ∧ a 2 = 22 := by sorry

end NUMINAMATH_CALUDE_sequence_a_l3559_355926


namespace NUMINAMATH_CALUDE_lilys_lottery_prize_l3559_355913

/-- The amount of money the lottery winner will receive -/
def lottery_prize (num_tickets : ℕ) (initial_price : ℕ) (price_increment : ℕ) (profit : ℕ) : ℕ :=
  let total_sales := (num_tickets * (2 * initial_price + (num_tickets - 1) * price_increment)) / 2
  total_sales - profit

/-- Theorem stating the lottery prize for Lily's specific scenario -/
theorem lilys_lottery_prize :
  lottery_prize 5 1 1 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_lilys_lottery_prize_l3559_355913


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3559_355951

theorem no_infinite_prime_sequence : 
  ¬ ∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧ (∀ n, p (n + 1) = 2 * p n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l3559_355951


namespace NUMINAMATH_CALUDE_generating_function_value_l3559_355989

/-- The generating function of two linear functions -/
def generating_function (m n x : ℝ) : ℝ := m * (x + 1) + n * (2 * x)

/-- Theorem: The generating function equals 2 when x = 1 and m + n = 1 -/
theorem generating_function_value : 
  ∀ m n : ℝ, m + n = 1 → generating_function m n 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_generating_function_value_l3559_355989


namespace NUMINAMATH_CALUDE_work_completion_time_l3559_355965

theorem work_completion_time (a_time b_time : ℕ) (remaining_fraction : ℚ) : 
  a_time = 15 → b_time = 20 → remaining_fraction = 8/15 → 
  (1 - remaining_fraction) / ((1 / a_time) + (1 / b_time)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3559_355965


namespace NUMINAMATH_CALUDE_k_less_than_one_necessary_not_sufficient_l3559_355995

/-- For x in the open interval (0, π/2), "k < 1" is a necessary but not sufficient condition for k*sin(x)*cos(x) < x. -/
theorem k_less_than_one_necessary_not_sufficient :
  ∀ k : ℝ, (∃ x : ℝ, 0 < x ∧ x < π/2 ∧ k * Real.sin x * Real.cos x < x) →
  (k < 1 ∧ ∃ k' ≥ 1, ∃ x : ℝ, 0 < x ∧ x < π/2 ∧ k' * Real.sin x * Real.cos x < x) :=
by sorry

end NUMINAMATH_CALUDE_k_less_than_one_necessary_not_sufficient_l3559_355995


namespace NUMINAMATH_CALUDE_problem_grid_square_count_l3559_355915

/-- Represents a grid with vertical and horizontal lines -/
structure Grid :=
  (vertical_lines : ℕ)
  (horizontal_lines : ℕ)
  (vertical_spacing : List ℕ)
  (horizontal_spacing : List ℕ)

/-- Counts the number of squares in a grid -/
def count_squares (g : Grid) : ℕ :=
  sorry

/-- The specific grid described in the problem -/
def problem_grid : Grid :=
  { vertical_lines := 5,
    horizontal_lines := 6,
    vertical_spacing := [1, 2, 1, 1],
    horizontal_spacing := [2, 1, 1, 1] }

/-- Theorem stating that the number of squares in the problem grid is 23 -/
theorem problem_grid_square_count :
  count_squares problem_grid = 23 :=
by sorry

end NUMINAMATH_CALUDE_problem_grid_square_count_l3559_355915


namespace NUMINAMATH_CALUDE_max_true_statements_four_true_statements_possible_l3559_355910

theorem max_true_statements (a b : ℝ) : 
  ¬(1/a < 1/b ∧ a^3 < b^3 ∧ a < b ∧ a < 0 ∧ b < 0) :=
by sorry

theorem four_true_statements_possible (a b : ℝ) : 
  ∃ (a b : ℝ), a^3 < b^3 ∧ a < b ∧ a < 0 ∧ b < 0 :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_four_true_statements_possible_l3559_355910


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3559_355982

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ r : ℝ, r ≠ 0 ∧ a 5 = r * a 1 ∧ a 17 = r * a 5) →
  (∃ r : ℝ, r ≠ 0 ∧ a 5 = r * a 1 ∧ a 17 = r * a 5 ∧ r = 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3559_355982


namespace NUMINAMATH_CALUDE_tree_height_proof_l3559_355954

/-- The initial height of a tree that grows 0.5 feet per year for 6 years and is 1/6 taller
    at the end of the 6th year compared to the 4th year. -/
def initial_tree_height : ℝ :=
  let growth_rate : ℝ := 0.5
  let years : ℕ := 6
  let h : ℝ := 4  -- Initial height to be proved
  h

theorem tree_height_proof (h : ℝ) (growth_rate : ℝ) (years : ℕ) 
    (h_growth : growth_rate = 0.5)
    (h_years : years = 6)
    (h_ratio : h + years * growth_rate = (h + 4 * growth_rate) * (1 + 1/6)) :
  h = initial_tree_height :=
sorry

#check tree_height_proof

end NUMINAMATH_CALUDE_tree_height_proof_l3559_355954


namespace NUMINAMATH_CALUDE_workers_wage_increase_l3559_355938

theorem workers_wage_increase (original_wage new_wage : ℝ) : 
  (original_wage * 1.5 = new_wage) → 
  (new_wage = 51) → 
  (original_wage = 34) := by
sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l3559_355938


namespace NUMINAMATH_CALUDE_vector_calculation_l3559_355933

/-- Given two plane vectors a and b, prove that (1/2)a - (3/2)b equals (-1, 2) -/
theorem vector_calculation (a b : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l3559_355933


namespace NUMINAMATH_CALUDE_range_of_a_for_C_subset_B_l3559_355972

def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem range_of_a_for_C_subset_B :
  {a : ℝ | C a ⊆ B} = {a : ℝ | 2 ≤ a ∧ a ≤ 8} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_C_subset_B_l3559_355972


namespace NUMINAMATH_CALUDE_one_fourth_of_7_2_l3559_355992

theorem one_fourth_of_7_2 : 
  (7.2 : ℚ) / 4 = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_7_2_l3559_355992


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3559_355907

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 :=
by sorry

theorem min_value_achieved (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (a₀ b₀ c₀ : ℝ), 2 ≤ a₀ ∧ a₀ ≤ b₀ ∧ b₀ ≤ c₀ ∧ c₀ ≤ 5 ∧
    (a₀ - 2)^2 + (b₀/a₀ - 1)^2 + (c₀/b₀ - 1)^2 + (5/c₀ - 1)^2 = 4 * (5^(1/4) - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3559_355907


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l3559_355957

theorem unique_number_with_gcd : ∃! n : ℕ, 90 < n ∧ n < 100 ∧ Nat.gcd 35 n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l3559_355957


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l3559_355934

theorem cylinder_radius_proof (r : ℝ) : 
  (r > 0) →                            -- r is positive (radius)
  (2 > 0) →                            -- original height is positive
  (π * (r + 6)^2 * 2 = π * r^2 * 8) →  -- volumes are equal when increased
  r = 6 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l3559_355934


namespace NUMINAMATH_CALUDE_lukas_average_points_l3559_355905

/-- Lukas's average points per game in basketball -/
def average_points (total_points : ℕ) (num_games : ℕ) : ℚ :=
  (total_points : ℚ) / (num_games : ℚ)

/-- Theorem: Lukas averages 12 points per game -/
theorem lukas_average_points :
  average_points 60 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_lukas_average_points_l3559_355905


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3559_355947

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3559_355947


namespace NUMINAMATH_CALUDE_probability_acute_triangle_in_pentagon_l3559_355932

-- Define a regular pentagon
def RegularPentagon : Type := Unit

-- Define a function to select 3 distinct vertices from 5
def selectThreeVertices (p : RegularPentagon) : ℕ := 10

-- Define a function to count acute triangles in a regular pentagon
def countAcuteTriangles (p : RegularPentagon) : ℕ := 5

-- Define the probability of forming an acute triangle
def probabilityAcuteTriangle (p : RegularPentagon) : ℚ :=
  (countAcuteTriangles p : ℚ) / (selectThreeVertices p : ℚ)

-- Theorem statement
theorem probability_acute_triangle_in_pentagon (p : RegularPentagon) :
  probabilityAcuteTriangle p = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_acute_triangle_in_pentagon_l3559_355932


namespace NUMINAMATH_CALUDE_isabel_homework_l3559_355962

/-- Given the total number of problems, completed problems, and problems per page,
    calculate the number of remaining pages. -/
def remaining_pages (total : ℕ) (completed : ℕ) (per_page : ℕ) : ℕ :=
  (total - completed) / per_page

/-- Theorem stating that given Isabel's homework conditions, 
    the number of remaining pages is 5. -/
theorem isabel_homework : 
  remaining_pages 72 32 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_l3559_355962


namespace NUMINAMATH_CALUDE_add_4500_seconds_to_10_45_00_l3559_355983

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time 10:45:00 -/
def startTime : Time :=
  { hours := 10, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 4500

/-- The expected end time 12:00:00 -/
def endTime : Time :=
  { hours := 12, minutes := 0, seconds := 0 }

theorem add_4500_seconds_to_10_45_00 :
  addSeconds startTime secondsToAdd = endTime := by
  sorry

end NUMINAMATH_CALUDE_add_4500_seconds_to_10_45_00_l3559_355983


namespace NUMINAMATH_CALUDE_point_in_region_implies_a_negative_l3559_355974

theorem point_in_region_implies_a_negative (a : ℝ) :
  (2 * a + 3 < 3) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_implies_a_negative_l3559_355974


namespace NUMINAMATH_CALUDE_binomial_expectation_from_variance_l3559_355909

/-- 
Given a binomial distribution with 4 trials and probability p of success on each trial,
if the variance of the distribution is 1, then the expected value is 2.
-/
theorem binomial_expectation_from_variance 
  (p : ℝ) 
  (h_prob : 0 ≤ p ∧ p ≤ 1) 
  (h_var : 4 * p * (1 - p) = 1) : 
  4 * p = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expectation_from_variance_l3559_355909


namespace NUMINAMATH_CALUDE_store_a_cheaper_for_15_boxes_store_a_cheaper_for_x_boxes_l3559_355977

/-- Represents the cost of purchasing table tennis equipment from a store. -/
structure StoreCost where
  ballCost : ℝ  -- Cost per box of balls
  racketCost : ℝ  -- Cost per racket
  numRackets : ℕ  -- Number of rackets needed
  discount : ℝ  -- Discount factor (1 for no discount, 0.9 for 10% discount)
  freeBoxes : ℕ  -- Number of free boxes of balls

/-- Calculates the total cost for a given number of ball boxes. -/
def totalCost (s : StoreCost) (x : ℕ) : ℝ :=
  s.discount * (s.ballCost * (x - s.freeBoxes) + s.racketCost * s.numRackets)

/-- Store A's cost structure -/
def storeA : StoreCost :=
  { ballCost := 5
  , racketCost := 30
  , numRackets := 5
  , discount := 1
  , freeBoxes := 5 }

/-- Store B's cost structure -/
def storeB : StoreCost :=
  { ballCost := 5
  , racketCost := 30
  , numRackets := 5
  , discount := 0.9
  , freeBoxes := 0 }

/-- Theorem stating that Store A is cheaper than or equal to Store B for 15 boxes of balls -/
theorem store_a_cheaper_for_15_boxes :
  totalCost storeA 15 ≤ totalCost storeB 15 :=
by
  sorry

/-- Theorem stating that Store A is cheaper than or equal to Store B for any number of boxes ≥ 5 -/
theorem store_a_cheaper_for_x_boxes (x : ℕ) (h : x ≥ 5) :
  totalCost storeA x ≤ totalCost storeB x :=
by
  sorry

end NUMINAMATH_CALUDE_store_a_cheaper_for_15_boxes_store_a_cheaper_for_x_boxes_l3559_355977


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3559_355928

def A : Set ℤ := {x | x^2 + x - 6 ≤ 0}
def B : Set ℤ := {x | x ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3559_355928


namespace NUMINAMATH_CALUDE_bug_prob_after_8_meters_l3559_355978

/-- Represents the probability of the bug being at vertex A after n meters -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - Q n) / 3

/-- The vertices of the tetrahedron -/
inductive Vertex
| A | B | C | D

/-- The probability of the bug being at vertex A after 8 meters -/
def prob_at_A_after_8 : ℚ := Q 8

theorem bug_prob_after_8_meters :
  prob_at_A_after_8 = 547 / 2187 :=
sorry

end NUMINAMATH_CALUDE_bug_prob_after_8_meters_l3559_355978


namespace NUMINAMATH_CALUDE_decimal_representation_of_225_999_l3559_355923

theorem decimal_representation_of_225_999 :
  ∃ (d : ℕ → ℕ), 
    (∀ n, d n < 10) ∧ 
    (∀ n, d (n + 3) = d n) ∧
    (d 0 = 2 ∧ d 1 = 2 ∧ d 2 = 5) ∧
    (d 80 = 5) ∧
    (225 : ℚ) / 999 = ∑' n, (d n : ℚ) / 10 ^ (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_225_999_l3559_355923


namespace NUMINAMATH_CALUDE_distance_AB_is_300_l3559_355955

/-- The distance between points A and B in meters -/
def distance_AB : ℝ := 300

/-- The speed ratio of Person A to Person B -/
def speed_ratio : ℝ := 2

/-- The distance Person B is from point B when Person A reaches B -/
def distance_B_to_B : ℝ := 100

/-- The distance from B where Person A and B meet when A returns -/
def meeting_distance : ℝ := 60

/-- Theorem stating the distance between A and B is 300 meters -/
theorem distance_AB_is_300 :
  distance_AB = 300 ∧
  speed_ratio = 2 ∧
  distance_B_to_B = 100 ∧
  meeting_distance = 60 →
  distance_AB = 300 := by
  sorry

#check distance_AB_is_300

end NUMINAMATH_CALUDE_distance_AB_is_300_l3559_355955


namespace NUMINAMATH_CALUDE_labourer_income_l3559_355924

/-- Prove that the monthly income of a labourer is 75 --/
theorem labourer_income :
  ∀ (avg_expenditure_6m : ℝ) (debt : ℝ) (expenditure_4m : ℝ) (savings : ℝ),
    avg_expenditure_6m = 80 →
    debt > 0 →
    expenditure_4m = 60 →
    savings = 30 →
    ∃ (income : ℝ),
      income * 6 - debt + income * 4 = avg_expenditure_6m * 6 + expenditure_4m * 4 + debt + savings ∧
      income = 75 := by
  sorry

end NUMINAMATH_CALUDE_labourer_income_l3559_355924


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3559_355956

theorem polygon_interior_angles (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 1260 → (n - 2) * 180 = sum_angles → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3559_355956


namespace NUMINAMATH_CALUDE_dice_sum_probability_l3559_355912

-- Define the number of dice
def num_dice : ℕ := 8

-- Define the target sum
def target_sum : ℕ := 11

-- Define the function to calculate the number of ways to achieve the target sum
def num_ways_to_achieve_sum (n d s : ℕ) : ℕ :=
  Nat.choose (s - n + d - 1) (d - 1)

-- Theorem statement
theorem dice_sum_probability :
  num_ways_to_achieve_sum num_dice num_dice target_sum = 120 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l3559_355912


namespace NUMINAMATH_CALUDE_sum_of_squares_and_squared_sum_l3559_355952

theorem sum_of_squares_and_squared_sum : (5 + 9 - 3)^2 + (5^2 + 9^2 + 3^2) = 236 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_squared_sum_l3559_355952


namespace NUMINAMATH_CALUDE_option_c_is_linear_system_l3559_355961

-- Define what a linear equation in two variables is
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

-- Define a system of two equations
def is_system_of_two_equations (f g : ℝ → ℝ → ℝ) : Prop :=
  true  -- This is always true as we're given two equations

-- Define the specific equations from Option C
def eq1 (x y : ℝ) : ℝ := x + y - 5
def eq2 (x y : ℝ) : ℝ := 3 * x - 4 * y - 12

-- Theorem stating that eq1 and eq2 form a system of two linear equations
theorem option_c_is_linear_system :
  is_linear_equation eq1 ∧ is_linear_equation eq2 ∧ is_system_of_two_equations eq1 eq2 :=
sorry

end NUMINAMATH_CALUDE_option_c_is_linear_system_l3559_355961


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_and_f_properties_l3559_355999

/-- Triangle ABC with base AB and height 1 -/
structure Triangle :=
  (base : ℝ)
  (height : ℝ)
  (height_eq_one : height = 1)

/-- Rectangle PQRS with width PQ and height 1 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (height_eq_one : height = 1)

/-- The function f(x) representing the height of rectangle PQNM -/
def f (x : ℝ) : ℝ := 2 * x - x^2

theorem triangle_rectangle_ratio_and_f_properties 
  (triangle : Triangle) (rectangle : Rectangle) :
  (triangle.base / rectangle.width = 2) ∧
  (triangle.base * triangle.height / 2 = rectangle.width * rectangle.height) ∧
  (f (1/2) = 3/4) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 
    f x = 2 * x - x^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_and_f_properties_l3559_355999


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l3559_355901

theorem smallest_five_digit_divisible_by_first_five_primes :
  (∀ n : ℕ, n ≥ 10000 ∧ n < 11550 → ¬(2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n)) ∧
  (11550 ≥ 10000) ∧
  (2 ∣ 11550) ∧ (3 ∣ 11550) ∧ (5 ∣ 11550) ∧ (7 ∣ 11550) ∧ (11 ∣ 11550) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l3559_355901


namespace NUMINAMATH_CALUDE_sum_even_positive_lt_100_l3559_355984

/-- The sum of all even, positive integers less than 100 is 2450 -/
theorem sum_even_positive_lt_100 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ 0 < n) (Finset.range 100)).sum id = 2450 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_positive_lt_100_l3559_355984


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l3559_355920

/-- Given two congruent squares with side length 20 that overlap to form a 20 by 30 rectangle,
    the percentage of the area of the rectangle that is shaded is 100/3%. -/
theorem shaded_area_percentage (square_side : ℝ) (rect_width rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 20 →
  rect_length = 30 →
  (((2 * square_side - rect_length) * square_side) / (rect_width * rect_length)) * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l3559_355920


namespace NUMINAMATH_CALUDE_jam_distribution_l3559_355921

/-- The jam distribution problem -/
theorem jam_distribution (total_jam : ℝ) (ponchik_hypothetical_days : ℝ) (syrupchik_hypothetical_days : ℝ)
  (h_total : total_jam = 100)
  (h_ponchik : ponchik_hypothetical_days = 45)
  (h_syrupchik : syrupchik_hypothetical_days = 20) :
  ∃ (ponchik_jam syrupchik_jam ponchik_rate syrupchik_rate : ℝ),
    ponchik_jam + syrupchik_jam = total_jam ∧
    ponchik_jam = 40 ∧
    syrupchik_jam = 60 ∧
    ponchik_rate = 4/3 ∧
    syrupchik_rate = 2 ∧
    ponchik_jam / ponchik_rate = syrupchik_jam / syrupchik_rate ∧
    syrupchik_jam / ponchik_hypothetical_days = ponchik_rate ∧
    ponchik_jam / syrupchik_hypothetical_days = syrupchik_rate :=
by sorry

end NUMINAMATH_CALUDE_jam_distribution_l3559_355921


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3559_355963

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3 * z) / (x + 2 * y) + (5 * x) / (2 * y + 3 * z) + (2 * y) / (3 * x + z) ≥ (3 : ℝ) / 4 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (3 * z) / (x + 2 * y) + (5 * x) / (2 * y + 3 * z) + (2 * y) / (3 * x + z) < (3 : ℝ) / 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3559_355963


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3559_355940

theorem complex_fraction_simplification :
  (5 : ℂ) / (Complex.I - 2) = -2 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3559_355940


namespace NUMINAMATH_CALUDE_statement_holds_only_in_specific_cases_l3559_355980

-- Define the basic types
inductive GeometricObject
| Line
| Plane

-- Define the relationships
def perpendicular (a b : GeometricObject) : Prop := sorry
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the statement
def statement (x y z : GeometricObject) : Prop :=
  perpendicular x z → perpendicular y z → parallel x y

-- Theorem to prove
theorem statement_holds_only_in_specific_cases 
  (x y z : GeometricObject) : 
  statement x y z ↔ 
    ((x = GeometricObject.Line ∧ y = GeometricObject.Line ∧ z = GeometricObject.Plane) ∨
     (x = GeometricObject.Plane ∧ y = GeometricObject.Plane ∧ z = GeometricObject.Line)) :=
by sorry

end NUMINAMATH_CALUDE_statement_holds_only_in_specific_cases_l3559_355980


namespace NUMINAMATH_CALUDE_product_inspection_problem_l3559_355903

def total_products : ℕ := 100
def defective_products : ℕ := 3
def drawn_products : ℕ := 4
def defective_in_sample : ℕ := 2

theorem product_inspection_problem :
  (Nat.choose defective_products defective_in_sample) *
  (Nat.choose (total_products - defective_products) (drawn_products - defective_in_sample)) = 13968 := by
  sorry

end NUMINAMATH_CALUDE_product_inspection_problem_l3559_355903


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l3559_355986

theorem actual_distance_traveled (speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  speed = 10 →
  faster_speed = 20 →
  extra_distance = 40 →
  (∃ (time : ℝ), speed * time = faster_speed * time - extra_distance) →
  speed * (extra_distance / (faster_speed - speed)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l3559_355986


namespace NUMINAMATH_CALUDE_number_pair_theorem_l3559_355943

theorem number_pair_theorem (S P : ℝ) (x y : ℝ) 
  (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_number_pair_theorem_l3559_355943


namespace NUMINAMATH_CALUDE_system_solution_l3559_355948

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 6 * y = -2) ∧ 
    (5 * x + 3 * y = 13/2) ∧ 
    (x = 7/22) ∧ 
    (y = 6/11) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3559_355948


namespace NUMINAMATH_CALUDE_isosceles_non_equilateral_distinct_lines_l3559_355911

/-- A triangle in a 2D Euclidean space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is isosceles --/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Predicate to check if a triangle is equilateral --/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Function to count distinct lines representing altitudes, medians, and interior angle bisectors --/
def countDistinctLines (t : Triangle) : ℕ := sorry

/-- Theorem stating that an isosceles non-equilateral triangle has 5 distinct lines --/
theorem isosceles_non_equilateral_distinct_lines (t : Triangle) :
  isIsosceles t ∧ ¬isEquilateral t → countDistinctLines t = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_non_equilateral_distinct_lines_l3559_355911


namespace NUMINAMATH_CALUDE_count_students_in_line_l3559_355927

/-- The number of students in a line formation -/
def students_in_line (between : ℕ) : ℕ :=
  between + 2

/-- Theorem: Given 14 people between Yoojung and Eunji, there are 16 students in line -/
theorem count_students_in_line :
  students_in_line 14 = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_students_in_line_l3559_355927


namespace NUMINAMATH_CALUDE_intersection_of_intervals_l3559_355967

open Set

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 0 < x}

theorem intersection_of_intervals : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_l3559_355967


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l3559_355993

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBits (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBits (m / 2)
  toBits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, true, true]        -- 110₂
  let product := [false, true, true, true, true, false, true]  -- 1011110₂
  binaryToNat a * binaryToNat b = binaryToNat product :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l3559_355993


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3559_355987

theorem other_root_of_quadratic (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3559_355987


namespace NUMINAMATH_CALUDE_polygon_sides_l3559_355902

theorem polygon_sides (n : ℕ) (n_pos : n > 0) :
  (((n - 2) * 180) / n = 108) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3559_355902


namespace NUMINAMATH_CALUDE_f_composition_range_l3559_355906

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2^x

theorem f_composition_range : 
  {a : ℝ | f (f a) = 2^(f a)} = {a : ℝ | a ≥ 2/3} := by sorry

end NUMINAMATH_CALUDE_f_composition_range_l3559_355906


namespace NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l3559_355925

theorem y_over_z_equals_negative_five (x y z : ℝ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21) :
  y / z = -5 := by
sorry

end NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l3559_355925


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l3559_355936

theorem min_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧
  (∀ (c d : ℝ), c > 0 → d > 0 → c + d = 1 →
    Real.sqrt (c^2 + 1) + Real.sqrt (d^2 + 4) ≥ Real.sqrt (x^2 + 1) + Real.sqrt (y^2 + 4)) ∧
  Real.sqrt (x^2 + 1) + Real.sqrt (y^2 + 4) = Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l3559_355936


namespace NUMINAMATH_CALUDE_all_points_on_line_l3559_355973

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def isOnLine (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem all_points_on_line :
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨-2, -4⟩
  let points : List Point := [⟨1, 2⟩, ⟨0, 0⟩, ⟨2, 4⟩, ⟨5, 10⟩, ⟨-1, -2⟩]
  ∀ p ∈ points, isOnLine p p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_all_points_on_line_l3559_355973


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3559_355942

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a + 2 > 0) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3559_355942


namespace NUMINAMATH_CALUDE_one_pair_probability_l3559_355979

/-- The number of colors of socks --/
def num_colors : ℕ := 5

/-- The number of socks per color --/
def socks_per_color : ℕ := 2

/-- The total number of socks --/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn --/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks of the same color --/
theorem one_pair_probability : 
  (Nat.choose num_colors 4 * 4 * (2^3)) / Nat.choose total_socks socks_drawn = 40 / 63 :=
sorry

end NUMINAMATH_CALUDE_one_pair_probability_l3559_355979


namespace NUMINAMATH_CALUDE_max_candies_bob_l3559_355991

theorem max_candies_bob (total : ℕ) (h1 : total = 30) : ∃ (bob : ℕ), bob ≤ 10 ∧ bob + 2 * bob = total := by
  sorry

end NUMINAMATH_CALUDE_max_candies_bob_l3559_355991


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3559_355930

theorem roots_of_quadratic_equation :
  ∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l3559_355930


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3559_355985

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The number of bananas that are worth as much as 12 oranges -/
def bananas_worth_12_oranges : ℚ := 12

/-- The fraction of 16 bananas that are worth as much as 12 oranges -/
def fraction_of_16_bananas : ℚ := 3/4

/-- The number of bananas we're considering in the question -/
def question_bananas : ℚ := 9

/-- The fraction of question_bananas we're considering -/
def fraction_of_question_bananas : ℚ := 2/3

theorem banana_orange_equivalence :
  fraction_of_16_bananas * 16 = bananas_worth_12_oranges →
  fraction_of_question_bananas * question_bananas * banana_value = 6 := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3559_355985


namespace NUMINAMATH_CALUDE_investment_sum_l3559_355959

/-- Proves that if a sum P invested at 18% p.a. for two years yields Rs. 504 more interest
    than if invested at 12% p.a. for the same period, then P = 4200. -/
theorem investment_sum (P : ℚ) : 
  (P * 18 * 2 / 100) - (P * 12 * 2 / 100) = 504 → P = 4200 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l3559_355959


namespace NUMINAMATH_CALUDE_total_dogs_l3559_355994

/-- Represents the properties of dogs in a kennel -/
structure Kennel where
  longFurred : Nat
  brown : Nat
  longFurredBrown : Nat
  neitherLongFurredNorBrown : Nat

/-- Theorem stating the total number of dogs in the kennel -/
theorem total_dogs (k : Kennel) 
  (h1 : k.longFurred = 26)
  (h2 : k.brown = 22)
  (h3 : k.longFurredBrown = 11)
  (h4 : k.neitherLongFurredNorBrown = 8) :
  k.longFurred + k.brown - k.longFurredBrown + k.neitherLongFurredNorBrown = 45 := by
  sorry

#check total_dogs

end NUMINAMATH_CALUDE_total_dogs_l3559_355994


namespace NUMINAMATH_CALUDE_assignment_statement_properties_l3559_355918

-- Define what an assignment statement is
def AssignmentStatement : Type := Unit

-- Define the properties of assignment statements
def can_provide_initial_values (a : AssignmentStatement) : Prop := sorry
def assigns_expression_value (a : AssignmentStatement) : Prop := sorry
def can_assign_multiple_times (a : AssignmentStatement) : Prop := sorry

-- Theorem stating the properties of assignment statements
theorem assignment_statement_properties (a : AssignmentStatement) :
  can_provide_initial_values a ∧
  assigns_expression_value a ∧
  can_assign_multiple_times a := by sorry

end NUMINAMATH_CALUDE_assignment_statement_properties_l3559_355918


namespace NUMINAMATH_CALUDE_triangle_trig_max_value_l3559_355970

theorem triangle_trig_max_value (A B C : ℝ) (h_sum : A + B + C = Real.pi) :
  (∀ A' B' C' : ℝ, A' + B' + C' = Real.pi →
    (Real.sin A * Real.cos B + Real.sin B * Real.cos C + Real.sin C * Real.cos A)^2 ≤
    (Real.sin A' * Real.cos B' + Real.sin B' * Real.cos C' + Real.sin C' * Real.cos A')^2) →
  (Real.sin A * Real.cos B + Real.sin B * Real.cos C + Real.sin C * Real.cos A)^2 = 27 / 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_trig_max_value_l3559_355970


namespace NUMINAMATH_CALUDE_minimum_chocolates_l3559_355908

theorem minimum_chocolates (n : ℕ) : n ≥ 118 →
  (n % 6 = 4 ∧ n % 8 = 6 ∧ n % 10 = 8) →
  ∃ (m : ℕ), m < n → ¬(m % 6 = 4 ∧ m % 8 = 6 ∧ m % 10 = 8) :=
by sorry

end NUMINAMATH_CALUDE_minimum_chocolates_l3559_355908


namespace NUMINAMATH_CALUDE_square_diagonal_point_theorem_l3559_355937

/-- A square with side length 10 -/
structure Square :=
  (E F G H : ℝ × ℝ)
  (is_square : 
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = 100 ∧
    (F.1 - G.1)^2 + (F.2 - G.2)^2 = 100 ∧
    (G.1 - H.1)^2 + (G.2 - H.2)^2 = 100 ∧
    (H.1 - E.1)^2 + (H.2 - E.2)^2 = 100)

/-- Point Q on diagonal EH -/
def Q (s : Square) : ℝ × ℝ := sorry

/-- R1 is the circumcenter of triangle EFQ -/
def R1 (s : Square) : ℝ × ℝ := sorry

/-- R2 is the circumcenter of triangle GHQ -/
def R2 (s : Square) : ℝ × ℝ := sorry

/-- The angle between R1, Q, and R2 is 150° -/
def angle_R1QR2 (s : Square) : ℝ := sorry

theorem square_diagonal_point_theorem (s : Square) 
  (h1 : (Q s).1 > s.E.1 ∧ (Q s).1 < s.H.1)  -- EQ > HQ
  (h2 : angle_R1QR2 s = 150 * π / 180) :
  let EQ := Real.sqrt ((Q s).1 - s.E.1)^2 + ((Q s).2 - s.E.2)^2
  EQ = Real.sqrt 100 + Real.sqrt 150 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_point_theorem_l3559_355937


namespace NUMINAMATH_CALUDE_candy_distribution_l3559_355958

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) : 
  total_candy = 344 →
  num_students = 43 →
  total_candy = num_students * pieces_per_student →
  pieces_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3559_355958


namespace NUMINAMATH_CALUDE_fair_distribution_theorem_l3559_355944

/-- Represents the outcome of a chess game -/
inductive GameOutcome
  | A_Win
  | B_Win

/-- Represents the state of the chess competition -/
structure ChessCompetition where
  total_games : Nat
  games_played : Nat
  a_wins : Nat
  prize_money : Nat
  deriving Repr

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (comp : ChessCompetition) : Rat :=
  sorry

/-- Calculates the fair distribution of prize money -/
def fair_distribution (comp : ChessCompetition) : Nat × Nat :=
  sorry

/-- Theorem stating the fair distribution of prize money -/
theorem fair_distribution_theorem (comp : ChessCompetition) 
  (h1 : comp.total_games = 7)
  (h2 : comp.games_played = 5)
  (h3 : comp.a_wins = 3)
  (h4 : comp.prize_money = 10000) :
  fair_distribution comp = (7500, 2500) :=
sorry

end NUMINAMATH_CALUDE_fair_distribution_theorem_l3559_355944


namespace NUMINAMATH_CALUDE_sticker_sharing_l3559_355964

theorem sticker_sharing (total_stickers : ℕ) (andrew_final : ℕ) : 
  total_stickers = 1500 →
  andrew_final = 900 →
  (2 : ℚ) / 3 = (andrew_final - total_stickers / 5) / (3 * total_stickers / 5) :=
by sorry

end NUMINAMATH_CALUDE_sticker_sharing_l3559_355964


namespace NUMINAMATH_CALUDE_max_value_of_f_l3559_355976

/-- Given a function f(x) = (x^2 - 4)(x - a) where a is a real number and f'(1) = 0,
    the maximum value of f(x) on the interval [-2, 2] is 50/27. -/
theorem max_value_of_f (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = (x^2 - 4) * (x - a)) 
    (h2 : deriv f 1 = 0) : 
    ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y ≤ f x ∧ f x = 50/27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3559_355976


namespace NUMINAMATH_CALUDE_complement_of_supplement_35_l3559_355916

/-- The supplement of an angle in degrees -/
def supplement (x : ℝ) : ℝ := 180 - x

/-- The complement of an angle in degrees -/
def complement (x : ℝ) : ℝ := 90 - x

/-- Theorem: The degree measure of the complement of the supplement of a 35-degree angle is -55 degrees -/
theorem complement_of_supplement_35 : complement (supplement 35) = -55 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_supplement_35_l3559_355916


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3559_355996

theorem sqrt_simplification :
  Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3559_355996


namespace NUMINAMATH_CALUDE_lucy_sold_29_packs_l3559_355960

/-- The number of packs of cookies sold by Robyn -/
def robyn_packs : ℕ := 47

/-- The total number of packs of cookies sold by Robyn and Lucy -/
def total_packs : ℕ := 76

/-- The number of packs of cookies sold by Lucy -/
def lucy_packs : ℕ := total_packs - robyn_packs

theorem lucy_sold_29_packs : lucy_packs = 29 := by
  sorry

end NUMINAMATH_CALUDE_lucy_sold_29_packs_l3559_355960


namespace NUMINAMATH_CALUDE_sugar_for_frosting_l3559_355969

theorem sugar_for_frosting (total_sugar cake_sugar frosting_sugar : ℚ) : 
  total_sugar = 0.8 →
  cake_sugar = 0.2 →
  total_sugar = cake_sugar + frosting_sugar →
  frosting_sugar = 0.6 := by
sorry

end NUMINAMATH_CALUDE_sugar_for_frosting_l3559_355969


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3559_355946

theorem quadratic_equation_solution (p q : ℝ) : 
  p = 15 * q^2 - 5 → p = 40 → q = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3559_355946


namespace NUMINAMATH_CALUDE_min_m_for_perfect_fourth_power_min_m_value_exact_min_m_l3559_355998

theorem min_m_for_perfect_fourth_power (m n : ℕ+) (h : 24 * m = n ^ 4) : 
  ∀ k : ℕ+, 24 * k = (some_nat : ℕ) ^ 4 → m ≤ k := by
  sorry

theorem min_m_value (m n : ℕ+) (h : 24 * m = n ^ 4) : m ≥ 54 := by
  sorry

theorem exact_min_m (m n : ℕ+) (h : 24 * m = n ^ 4) : 
  (∃ k : ℕ+, 24 * 54 = k ^ 4) ∧ m ≥ 54 := by
  sorry

end NUMINAMATH_CALUDE_min_m_for_perfect_fourth_power_min_m_value_exact_min_m_l3559_355998


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3559_355922

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem solution_set_of_inequality
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_point_A : f 0 = 3)
  (h_point_B : f 3 = -1) :
  {x : ℝ | |f (x + 1) - 1| < 2} = Set.Ioo (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3559_355922


namespace NUMINAMATH_CALUDE_train_crossing_time_l3559_355914

/-- Given a train and a platform with specific dimensions, calculate the time taken for the train to cross the platform. -/
theorem train_crossing_time (train_length : ℝ) (signal_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : train_length = 450)
  (h2 : signal_crossing_time = 18)
  (h3 : platform_length = 525) :
  (train_length + platform_length) / (train_length / signal_crossing_time) = 39 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3559_355914


namespace NUMINAMATH_CALUDE_mars_network_connected_min_tunnels_for_connectivity_l3559_355968

/-- A graph representing the Mars settlement network -/
structure MarsNetwork where
  settlements : Nat
  tunnels : Nat

/-- The property that a MarsNetwork is connected -/
def is_connected (network : MarsNetwork) : Prop :=
  network.tunnels ≥ network.settlements - 1

/-- The Mars settlement network with 2004 settlements -/
def mars_network : MarsNetwork :=
  { settlements := 2004, tunnels := 2003 }

/-- Theorem stating that the Mars network with 2003 tunnels is connected -/
theorem mars_network_connected :
  is_connected mars_network :=
sorry

/-- Theorem stating that 2003 is the minimum number of tunnels required for connectivity -/
theorem min_tunnels_for_connectivity (network : MarsNetwork) :
  network.settlements = 2004 →
  is_connected network →
  network.tunnels ≥ 2003 :=
sorry

end NUMINAMATH_CALUDE_mars_network_connected_min_tunnels_for_connectivity_l3559_355968


namespace NUMINAMATH_CALUDE_smallest_positive_b_squared_l3559_355988

/-- Definition of circle u₁ -/
def u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0

/-- Definition of circle u₂ -/
def u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 175 = 0

/-- A circle is externally tangent to u₂ and internally tangent to u₁ -/
def is_tangent_circle (x y t : ℝ) : Prop :=
  t + 7 = Real.sqrt ((x - 4)^2 + (y - 10)^2) ∧
  11 - t = Real.sqrt ((x + 4)^2 + (y - 10)^2)

/-- The center of the tangent circle lies on the line y = bx -/
def center_on_line (x y b : ℝ) : Prop := y = b * x

/-- Main theorem: The smallest positive b satisfying the conditions has b² = 5/16 -/
theorem smallest_positive_b_squared (b : ℝ) :
  (∃ x y t, u₁ x y ∧ u₂ x y ∧ is_tangent_circle x y t ∧ center_on_line x y b) →
  (∀ b' : ℝ, 0 < b' → b' < b →
    ¬∃ x y t, u₁ x y ∧ u₂ x y ∧ is_tangent_circle x y t ∧ center_on_line x y b') →
  b^2 = 5/16 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_squared_l3559_355988


namespace NUMINAMATH_CALUDE_product_with_miscopied_digit_l3559_355939

theorem product_with_miscopied_digit (x y : ℕ) 
  (h1 : x * y = 4500)
  (h2 : x * (y - 2) = 4380) :
  x = 60 ∧ y = 75 := by
sorry

end NUMINAMATH_CALUDE_product_with_miscopied_digit_l3559_355939


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3559_355949

theorem sqrt_sum_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  2 * Real.sqrt (a + b + c + d) ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3559_355949


namespace NUMINAMATH_CALUDE_greenhouse_optimization_l3559_355966

/-- Given a rectangle with area 800 m², prove that the maximum area of the inner rectangle
    formed by subtracting a 1 m border on three sides and a 3 m border on one side
    is 648 m², achieved when the original rectangle has dimensions 40 m × 20 m. -/
theorem greenhouse_optimization (a b : ℝ) :
  a > 0 ∧ b > 0 ∧ a * b = 800 →
  (a - 2) * (b - 4) ≤ 648 ∧
  (a - 2) * (b - 4) = 648 ↔ a = 40 ∧ b = 20 :=
by sorry

end NUMINAMATH_CALUDE_greenhouse_optimization_l3559_355966


namespace NUMINAMATH_CALUDE_expected_collectors_is_120_l3559_355975

-- Define the number of customers
def num_customers : ℕ := 3000

-- Define the probability of a customer collecting a prize
def prob_collect : ℝ := 0.04

-- Define the expected number of prize collectors
def expected_collectors : ℝ := num_customers * prob_collect

-- Theorem statement
theorem expected_collectors_is_120 : expected_collectors = 120 := by
  sorry

end NUMINAMATH_CALUDE_expected_collectors_is_120_l3559_355975


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l3559_355935

def B_inverse : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3, 4;
    -2, -3]

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = B_inverse) : 
  (B^3)⁻¹ = B_inverse := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l3559_355935


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3559_355990

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 110 → percentage = 50 → result = initial * (1 + percentage / 100) → result = 165 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3559_355990


namespace NUMINAMATH_CALUDE_equation_root_l3559_355904

theorem equation_root (m : ℝ) : 
  (∃ x : ℝ, x^2 + 5*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + 5*y + m = 0 ∧ y = -4) :=
sorry

end NUMINAMATH_CALUDE_equation_root_l3559_355904


namespace NUMINAMATH_CALUDE_perpendicular_planes_l3559_355945

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b : Line) 
  (ξ ζ : Plane) 
  (diff_lines : a ≠ b) 
  (diff_planes : ξ ≠ ζ) 
  (h1 : perp_line_line a b) 
  (h2 : perp_line_plane a ξ) 
  (h3 : perp_line_plane b ζ) : 
  perp_plane_plane ξ ζ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l3559_355945


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l3559_355900

/-- The maximum number of students among whom 781 pens and 710 pencils can be distributed equally -/
theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 781) (h_pencils : pencils = 710) :
  (∃ (students pen_per_student pencil_per_student : ℕ), 
    students * pen_per_student = pens ∧ 
    students * pencil_per_student = pencils ∧ 
    ∀ s : ℕ, s * pen_per_student = pens → s * pencil_per_student = pencils → s ≤ students) →
  Nat.gcd pens pencils = 71 :=
by sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l3559_355900


namespace NUMINAMATH_CALUDE_B_subset_A_l3559_355997

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem B_subset_A (B : Set ℝ) (h : A ∩ B = B) : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l3559_355997


namespace NUMINAMATH_CALUDE_no_real_roots_l3559_355917

theorem no_real_roots : ¬∃ x : ℝ, |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3559_355917


namespace NUMINAMATH_CALUDE_jose_land_division_l3559_355950

/-- 
Given that Jose divides his land equally among himself and his four siblings,
and he ends up with 4,000 square meters, prove that the total amount of land
he initially bought was 20,000 square meters.
-/
theorem jose_land_division (jose_share : ℝ) (num_siblings : ℕ) :
  jose_share = 4000 →
  num_siblings = 4 →
  (jose_share * (num_siblings + 1) : ℝ) = 20000 := by
  sorry

end NUMINAMATH_CALUDE_jose_land_division_l3559_355950


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3559_355931

theorem necessary_but_not_sufficient 
  (a b : ℝ) 
  (ha : a > 0) :
  (a > abs b → a + b > 0) ∧ 
  ¬(∀ a b : ℝ, a > 0 → a + b > 0 → a > abs b) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3559_355931


namespace NUMINAMATH_CALUDE_apple_distribution_l3559_355929

/-- Represents the number of apples each person has -/
structure Apples where
  greg : ℕ
  sarah : ℕ
  susan : ℕ
  mark : ℕ

/-- The ratio of Susan's apples to Greg's apples -/
def apple_ratio (a : Apples) : ℚ :=
  a.susan / a.greg

theorem apple_distribution (a : Apples) :
  a.greg = a.sarah ∧
  a.greg + a.sarah = 18 ∧
  a.mark = a.susan - 5 ∧
  a.greg + a.sarah + a.susan + a.mark = 49 →
  apple_ratio a = 2 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l3559_355929


namespace NUMINAMATH_CALUDE_expression_factorization_l3559_355981

theorem expression_factorization (x : ℝ) : 
  (12 * x^6 + 30 * x^4 - 6) - (2 * x^6 - 4 * x^4 - 6) = 2 * x^4 * (5 * x^2 + 17) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3559_355981
