import Mathlib

namespace m_range_theorem_l4040_404063

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem m_range_theorem (f : ℝ → ℝ) (m : ℝ) 
  (h_odd : is_odd_function f)
  (h_prop : ∀ x, f (3/4 * x) = f (3/4 * x))
  (h_lower_bound : ∀ x, f x > -2)
  (h_f_1 : f 1 = -3/m) :
  (0 < m ∧ m < 3) ∨ m < -1 := by
sorry

end m_range_theorem_l4040_404063


namespace largest_mersenne_prime_factor_of_1000_l4040_404081

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2^p - 1 ∧ Prime n

theorem largest_mersenne_prime_factor_of_1000 :
  ∃ n : ℕ, is_mersenne_prime n ∧ n < 500 ∧ n ∣ 1000 ∧
  ∀ m : ℕ, is_mersenne_prime m ∧ m < 500 ∧ m ∣ 1000 → m ≤ n :=
by
  use 3
  sorry

end largest_mersenne_prime_factor_of_1000_l4040_404081


namespace probability_of_humanities_course_l4040_404049

/-- Represents a course --/
inductive Course
| Mathematics
| Chinese
| Politics
| Geography
| English
| History
| PhysicalEducation

/-- Represents the time of day --/
inductive TimeOfDay
| Morning
| Afternoon

/-- Defines whether a course is in humanities and social sciences --/
def isHumanities (c : Course) : Bool :=
  match c with
  | Course.Politics | Course.History | Course.Geography => true
  | _ => false

/-- Defines the courses available in each time slot --/
def availableCourses (t : TimeOfDay) : List Course :=
  match t with
  | TimeOfDay.Morning => [Course.Mathematics, Course.Chinese, Course.Politics, Course.Geography]
  | TimeOfDay.Afternoon => [Course.English, Course.History, Course.PhysicalEducation]

theorem probability_of_humanities_course :
  let totalChoices := (availableCourses TimeOfDay.Morning).length * (availableCourses TimeOfDay.Afternoon).length
  let humanitiesChoices := totalChoices - ((availableCourses TimeOfDay.Morning).filter (fun c => !isHumanities c)).length *
                                          ((availableCourses TimeOfDay.Afternoon).filter (fun c => !isHumanities c)).length
  (humanitiesChoices : ℚ) / totalChoices = 2 / 3 := by
  sorry

end probability_of_humanities_course_l4040_404049


namespace pyramid_height_proof_l4040_404042

/-- The height of a square-based pyramid with base edge length 10 units, 
    which has the same volume as a cube with edge length 5 units. -/
def pyramid_height : ℝ := 3.75

theorem pyramid_height_proof :
  let cube_edge : ℝ := 5
  let pyramid_base : ℝ := 10
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume (h : ℝ) : ℝ := (1/3) * pyramid_base ^ 2 * h
  pyramid_volume pyramid_height = cube_volume :=
by sorry

end pyramid_height_proof_l4040_404042


namespace arithmetic_sequence_first_term_l4040_404024

/-- An arithmetic sequence is increasing if its common difference is positive -/
def IsIncreasingArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + d) ∧ d > 0

/-- The sum of the first three terms of an arithmetic sequence -/
def SumFirstThree (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3

/-- The product of the first three terms of an arithmetic sequence -/
def ProductFirstThree (a : ℕ → ℝ) : ℝ :=
  a 1 * a 2 * a 3

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ) (d : ℝ)
  (h_increasing : IsIncreasingArithmeticSequence a d)
  (h_sum : SumFirstThree a = 12)
  (h_product : ProductFirstThree a = 48) :
  a 1 = 2 := by
  sorry

end arithmetic_sequence_first_term_l4040_404024


namespace imaginary_part_of_z_l4040_404046

theorem imaginary_part_of_z (z : ℂ) : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I →
  z.im = (Real.sqrt 2 + 1) / 2 := by
sorry

end imaginary_part_of_z_l4040_404046


namespace matt_work_time_l4040_404009

/-- The number of minutes Matt worked on Monday -/
def monday_minutes : ℕ := 450

/-- The number of minutes Matt worked on Tuesday -/
def tuesday_minutes : ℕ := monday_minutes / 2

/-- The additional minutes Matt worked on the certain day compared to Tuesday -/
def additional_minutes : ℕ := 75

/-- The number of minutes Matt worked on the certain day -/
def certain_day_minutes : ℕ := tuesday_minutes + additional_minutes

theorem matt_work_time : certain_day_minutes = 300 := by
  sorry

end matt_work_time_l4040_404009


namespace domain_intersection_l4040_404057

def A : Set ℝ := {x : ℝ | x > -1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem domain_intersection : A ∩ B = {0, 1, 2} := by sorry

end domain_intersection_l4040_404057


namespace fraction_value_when_a_equals_4b_l4040_404001

theorem fraction_value_when_a_equals_4b (a b : ℝ) (h : a = 4 * b) :
  (a^2 + b^2) / (a * b) = 17 / 4 := by
  sorry

end fraction_value_when_a_equals_4b_l4040_404001


namespace larger_number_problem_l4040_404021

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 52) (h2 : x = 3 * y) (h3 : x > 0) (h4 : y > 0) : x = 39 := by
  sorry

end larger_number_problem_l4040_404021


namespace dog_catches_rabbit_problem_l4040_404090

/-- The number of leaps required for a dog to catch a rabbit -/
def dog_catches_rabbit (initial_distance : ℕ) (dog_leap : ℕ) (rabbit_jump : ℕ) : ℕ :=
  initial_distance / (dog_leap - rabbit_jump)

theorem dog_catches_rabbit_problem :
  dog_catches_rabbit 150 9 7 = 75 := by
  sorry

end dog_catches_rabbit_problem_l4040_404090


namespace cube_fraction_equals_150_l4040_404019

theorem cube_fraction_equals_150 :
  (68^3 - 65^3) * (32^3 + 18^3) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2)) = 150 := by
  sorry

end cube_fraction_equals_150_l4040_404019


namespace cesar_watched_fraction_l4040_404065

theorem cesar_watched_fraction (total_seasons : ℕ) (episodes_per_season : ℕ) (remaining_episodes : ℕ) :
  total_seasons = 12 →
  episodes_per_season = 20 →
  remaining_episodes = 160 →
  (total_seasons * episodes_per_season - remaining_episodes) / (total_seasons * episodes_per_season) = 1 / 3 := by
  sorry

end cesar_watched_fraction_l4040_404065


namespace geometric_sequence_sum_l4040_404058

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  is_geometric_sequence a →
  (a 5 + a 6 + a 7 + a 8 = 15/8) →
  (a 6 * a 7 = -9/8) →
  (1 / a 5 + 1 / a 6 + 1 / a 7 + 1 / a 8 = -5/3) := by
  sorry

end geometric_sequence_sum_l4040_404058


namespace min_cooking_time_is_15_l4040_404002

/-- Represents the time required for each step in the noodle cooking process -/
structure CookingTimes where
  washPot : ℕ
  washVegetables : ℕ
  prepareIngredients : ℕ
  boilWater : ℕ
  cookNoodles : ℕ

/-- Calculates the minimum time to cook noodles given the cooking times -/
def minCookingTime (times : CookingTimes) : ℕ :=
  let simultaneousTime := max times.washVegetables times.prepareIngredients
  times.washPot + simultaneousTime + times.cookNoodles

/-- Theorem stating that the minimum cooking time is 15 minutes -/
theorem min_cooking_time_is_15 (times : CookingTimes) 
  (h1 : times.washPot = 2)
  (h2 : times.washVegetables = 6)
  (h3 : times.prepareIngredients = 2)
  (h4 : times.boilWater = 10)
  (h5 : times.cookNoodles = 3) :
  minCookingTime times = 15 := by
  sorry

#eval minCookingTime ⟨2, 6, 2, 10, 3⟩

end min_cooking_time_is_15_l4040_404002


namespace circle_area_from_polar_equation_l4040_404070

/-- The area of the circle described by the polar equation r = 4 cos θ - 3 sin θ is equal to 25π/4 -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ ↦ 4 * Real.cos θ - 3 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ, r θ * Real.sin θ) ∈ Metric.sphere center radius) ∧
    Real.pi * radius^2 = 25 * Real.pi / 4 :=
by sorry

end circle_area_from_polar_equation_l4040_404070


namespace intersection_A_complement_B_l4040_404083

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {y | -1 < y ∧ y < 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ Bᶜ = {x : ℝ | x ≥ 2} := by sorry

end intersection_A_complement_B_l4040_404083


namespace min_value_theorem_equality_condition_l4040_404059

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 4 / x - 1 ≥ 3 :=
by sorry

theorem equality_condition : ∃ x : ℝ, x > 0 ∧ x + 4 / x - 1 = 3 :=
by sorry

end min_value_theorem_equality_condition_l4040_404059


namespace hexagon_diagonal_small_triangle_l4040_404038

/-- A convex hexagon in the plane -/
structure ConvexHexagon where
  -- We don't need to define the specific properties of a convex hexagon for this statement
  area : ℝ
  area_pos : area > 0

/-- A diagonal of a hexagon -/
structure Diagonal (h : ConvexHexagon) where
  -- We don't need to define the specific properties of a diagonal for this statement

/-- The area of the triangle cut off by a diagonal -/
noncomputable def triangle_area (h : ConvexHexagon) (d : Diagonal h) : ℝ :=
  sorry -- Definition not provided, as it's not part of the original conditions

theorem hexagon_diagonal_small_triangle (h : ConvexHexagon) :
  ∃ (d : Diagonal h), triangle_area h d ≤ h.area / 6 := by
  sorry

end hexagon_diagonal_small_triangle_l4040_404038


namespace necessary_not_sufficient_exists_x0_negation_is_false_l4040_404071

-- Define the necessary condition
def necessary_condition (a b : ℝ) : Prop := a + b > 4

-- Define the stronger condition
def stronger_condition (a b : ℝ) : Prop := a > 2 ∧ b > 2

-- Statement 1: Necessary but not sufficient condition
theorem necessary_not_sufficient :
  (∀ a b : ℝ, stronger_condition a b → necessary_condition a b) ∧
  (∃ a b : ℝ, necessary_condition a b ∧ ¬stronger_condition a b) := by sorry

-- Statement 2: Existence of x₀
theorem exists_x0 : ∃ x₀ : ℝ, x₀^2 - x₀ > 0 := by sorry

-- Statement 3: Negation is false
theorem negation_is_false : ¬(∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end necessary_not_sufficient_exists_x0_negation_is_false_l4040_404071


namespace geometric_to_arithmetic_sequence_l4040_404082

theorem geometric_to_arithmetic_sequence :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ (q : ℝ), q ≠ 0 ∧ b = a * q ∧ c = b * q) ∧
  a + b + c = 19 ∧
  b - a = (c - 1) - b :=
by sorry

end geometric_to_arithmetic_sequence_l4040_404082


namespace ones_digit_of_13_power_l4040_404085

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Define the exponent
def exponent : ℕ := 13 * (7^7)

-- Theorem statement
theorem ones_digit_of_13_power : ones_digit (13^exponent) = 7 := by
  sorry

end ones_digit_of_13_power_l4040_404085


namespace part_one_part_two_l4040_404092

/-- Definition of the sequence sum -/
def S (n : ℕ) (a : ℝ) : ℝ := a * 2^n - 1

/-- Definition of the sequence terms -/
def a (n : ℕ) (a : ℝ) : ℝ := S n a - S (n-1) a

/-- Part 1: Prove the values of a_1 and a_4 when a = 3 -/
theorem part_one :
  a 1 3 = 5 ∧ a 4 3 = 24 :=
sorry

/-- Definition of geometric sequence -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

/-- Part 2: Prove the value of a when {a_n} is a geometric sequence -/
theorem part_two :
  ∃ f : ℕ → ℝ, is_geometric_sequence f ∧ (∀ n : ℕ, S n 1 = f n - f 0) :=
sorry

end part_one_part_two_l4040_404092


namespace parabola_standard_equation_l4040_404018

/-- A parabola with its focus on the line 3x - 4y - 12 = 0 -/
structure Parabola where
  focus : ℝ × ℝ
  focus_on_line : 3 * focus.1 - 4 * focus.2 - 12 = 0

/-- The standard equation of a parabola -/
inductive StandardEquation
  | VerticalAxis (p : ℝ) : StandardEquation  -- y² = 4px
  | HorizontalAxis (p : ℝ) : StandardEquation  -- x² = 4py

theorem parabola_standard_equation (p : Parabola) :
  (∃ (eq : StandardEquation), eq = StandardEquation.VerticalAxis 4 ∨ eq = StandardEquation.HorizontalAxis (-3)) :=
sorry

end parabola_standard_equation_l4040_404018


namespace simplify_and_rationalize_l4040_404075

theorem simplify_and_rationalize (x : ℝ) (h : x = Real.sqrt 5) :
  1 / (2 + 2 / (x + 3)) = (7 + x) / 22 := by
  sorry

end simplify_and_rationalize_l4040_404075


namespace intersection_point_is_unique_l4040_404079

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (-15/8, 13/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 4 * y = -8 * x - 2

theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point) :=
by sorry

end intersection_point_is_unique_l4040_404079


namespace imaginary_part_of_z_is_two_l4040_404097

-- Define the complex number (2+i)i
def z : ℂ := (2 + Complex.I) * Complex.I

-- Theorem statement
theorem imaginary_part_of_z_is_two : Complex.im z = 2 := by
  sorry

end imaginary_part_of_z_is_two_l4040_404097


namespace millions_to_scientific_l4040_404033

-- Define the number in millions
def number_in_millions : ℝ := 3.111

-- Define the number in standard form
def number_standard : ℝ := 3111000

-- Define the number in scientific notation
def number_scientific : ℝ := 3.111 * (10 ^ 6)

-- Theorem to prove
theorem millions_to_scientific : number_standard = number_scientific := by
  sorry

end millions_to_scientific_l4040_404033


namespace chicken_crossing_ratio_l4040_404051

theorem chicken_crossing_ratio (initial_feathers final_feathers cars_dodged : ℕ) 
  (h1 : initial_feathers = 5263)
  (h2 : final_feathers = 5217)
  (h3 : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 := by
sorry

end chicken_crossing_ratio_l4040_404051


namespace f_min_at_three_l4040_404052

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : ∀ x : ℝ, f x ≥ f 3 := by sorry

end f_min_at_three_l4040_404052


namespace error_percentage_bounds_l4040_404064

theorem error_percentage_bounds (y : ℝ) (h : y > 0) :
  let error_percentage := (20 / (y + 8)) * 100
  100 < error_percentage ∧ error_percentage < 120 := by
sorry

end error_percentage_bounds_l4040_404064


namespace system_solution_l4040_404000

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^3 + y₁^3) * (x₁^2 + y₁^2) = 64 ∧
    x₁ + y₁ = 2 ∧
    x₁ = 1 + Real.sqrt (5/3) ∧
    y₁ = 1 - Real.sqrt (5/3) ∧
    (x₂^3 + y₂^3) * (x₂^2 + y₂^2) = 64 ∧
    x₂ + y₂ = 2 ∧
    x₂ = 1 - Real.sqrt (5/3) ∧
    y₂ = 1 + Real.sqrt (5/3) := by
  sorry

end system_solution_l4040_404000


namespace irina_square_area_l4040_404037

/-- Given a square with side length 12 cm, if another square has a perimeter 8 cm larger,
    then the area of the second square is 196 cm². -/
theorem irina_square_area (original_side : ℝ) (irina_side : ℝ) : 
  original_side = 12 →
  4 * irina_side = 4 * original_side + 8 →
  irina_side * irina_side = 196 :=
by
  sorry

#check irina_square_area

end irina_square_area_l4040_404037


namespace min_cubes_for_box_l4040_404084

theorem min_cubes_for_box (box_length box_width box_height cube_volume : ℕ) 
  (h1 : box_length = 10)
  (h2 : box_width = 13)
  (h3 : box_height = 5)
  (h4 : cube_volume = 5) :
  (box_length * box_width * box_height) / cube_volume = 130 := by
  sorry

end min_cubes_for_box_l4040_404084


namespace susan_money_problem_l4040_404040

theorem susan_money_problem (S : ℚ) :
  S - (S / 6 + S / 8 + S * (30 / 100) + 100) = 480 →
  S = 1420 := by
sorry

end susan_money_problem_l4040_404040


namespace tangent_line_equation_l4040_404034

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := 3*x - 2*x^3

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3 - 6*x^2

/-- The x-coordinate of the point of tangency -/
def a : ℝ := -1

/-- Theorem: The equation of the tangent line to y = 3x - 2x^3 at x = -1 is 3x + y + 4 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (y - f a) = f' a * (x - a) ↔ 3*x + y + 4 = 0 := by sorry

end tangent_line_equation_l4040_404034


namespace pq_divides_3p_minus_1_q_minus_1_l4040_404023

theorem pq_divides_3p_minus_1_q_minus_1 (p q : ℕ+) :
  (p * q : ℕ) ∣ (3 * (p - 1) * (q - 1) : ℕ) ↔
  ((p = 6 ∧ q = 5) ∨ (p = 5 ∧ q = 6) ∨
   (p = 9 ∧ q = 4) ∨ (p = 4 ∧ q = 9) ∨
   (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) :=
by sorry

end pq_divides_3p_minus_1_q_minus_1_l4040_404023


namespace thirty_percent_less_than_80_l4040_404010

theorem thirty_percent_less_than_80 : ∃ x : ℝ, (80 - 0.3 * 80) = x + 0.25 * x ∧ x = 45 := by
  sorry

end thirty_percent_less_than_80_l4040_404010


namespace chelsea_needs_52_bullseyes_l4040_404050

/-- Represents the archery contest scenario -/
structure ArcheryContest where
  total_shots : Nat
  chelsea_lead : Nat
  chelsea_min_score : Nat
  opponent_min_score : Nat
  bullseye_score : Nat

/-- Calculates the minimum number of bullseyes needed for Chelsea to guarantee a win -/
def min_bullseyes_needed (contest : ArcheryContest) : Nat :=
  let remaining_shots := contest.total_shots / 2
  let max_opponent_gain := remaining_shots * contest.bullseye_score
  let chelsea_gain_per_bullseye := contest.bullseye_score - contest.chelsea_min_score
  ((max_opponent_gain - contest.chelsea_lead) / chelsea_gain_per_bullseye) + 1

/-- Theorem stating that Chelsea needs at least 52 bullseyes to guarantee a win -/
theorem chelsea_needs_52_bullseyes (contest : ArcheryContest) 
  (h1 : contest.total_shots = 120)
  (h2 : contest.chelsea_lead = 60)
  (h3 : contest.chelsea_min_score = 3)
  (h4 : contest.opponent_min_score = 1)
  (h5 : contest.bullseye_score = 10) :
  min_bullseyes_needed contest ≥ 52 := by
  sorry

#eval min_bullseyes_needed { total_shots := 120, chelsea_lead := 60, chelsea_min_score := 3, opponent_min_score := 1, bullseye_score := 10 }

end chelsea_needs_52_bullseyes_l4040_404050


namespace waiter_customers_l4040_404089

theorem waiter_customers (initial new_customers customers_left : ℕ) :
  initial ≥ customers_left →
  (initial - customers_left + new_customers : ℕ) = initial - customers_left + new_customers :=
by sorry

end waiter_customers_l4040_404089


namespace always_negative_l4040_404076

-- Define the chessboard as a function from positions to integers
def Chessboard := Fin 8 → Fin 8 → Int

-- Initial configuration of the chessboard
def initial_board : Chessboard :=
  fun row col => if row = 1 ∧ col = 1 then -1 else 1

-- Define a single operation (flipping signs in a row or column)
def flip_row_or_col (board : Chessboard) (is_row : Bool) (index : Fin 8) : Chessboard :=
  fun row col => 
    if (is_row ∧ row = index) ∨ (¬is_row ∧ col = index) then
      -board row col
    else
      board row col

-- Define a sequence of operations
def apply_operations (board : Chessboard) (ops : List (Bool × Fin 8)) : Chessboard :=
  ops.foldl (fun b (is_row, index) => flip_row_or_col b is_row index) board

-- Theorem statement
theorem always_negative (ops : List (Bool × Fin 8)) :
  ∃ row col, (apply_operations initial_board ops) row col < 0 := by
  sorry

end always_negative_l4040_404076


namespace broadcast_methods_count_l4040_404047

/-- The number of different advertisements -/
def total_ads : ℕ := 5

/-- The number of commercial advertisements -/
def commercial_ads : ℕ := 3

/-- The number of Olympic promotional advertisements -/
def olympic_ads : ℕ := 2

/-- A function that calculates the number of ways to arrange the advertisements -/
def arrangement_count : ℕ :=
  Nat.factorial commercial_ads * Nat.choose 4 2

/-- Theorem stating that the number of different broadcasting methods is 36 -/
theorem broadcast_methods_count :
  arrangement_count = 36 :=
by sorry

end broadcast_methods_count_l4040_404047


namespace unique_base_solution_l4040_404015

/-- Converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

/-- The main theorem statement -/
theorem unique_base_solution :
  ∃! b : ℕ, b > 7 ∧ (toBase10 276 b) * 2 + (toBase10 145 b) = (toBase10 697 b) :=
by sorry

end unique_base_solution_l4040_404015


namespace line_through_intersection_and_origin_l4040_404066

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2*x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Define the intersection point of l1 and l2
def intersection : ℝ × ℝ := ((-2 : ℝ), (3 : ℝ))

-- Define the origin
def origin : ℝ × ℝ := ((0 : ℝ), (0 : ℝ))

-- Theorem statement
theorem line_through_intersection_and_origin :
  ∀ (x y : ℝ), l1 (intersection.1) (intersection.2) ∧ 
               l2 (intersection.1) (intersection.2) ∧ 
               (3*x + 2*y = 0 ↔ ∃ t : ℝ, x = t * (intersection.1 - origin.1) ∧ 
                                        y = t * (intersection.2 - origin.2)) :=
sorry

end line_through_intersection_and_origin_l4040_404066


namespace polynomial_factorization_l4040_404039

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end polynomial_factorization_l4040_404039


namespace num_lines_formula_l4040_404069

/-- The number of lines through n points in a plane, where n ≥ 3 and no three points are collinear -/
def num_lines (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of lines through n points in a plane, where n ≥ 3 and no three points are collinear, is n(n-1)/2 -/
theorem num_lines_formula (n : ℕ) (h : n ≥ 3) :
  num_lines n = n * (n - 1) / 2 := by sorry

end num_lines_formula_l4040_404069


namespace sum_of_coefficients_l4040_404087

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a + a₂ + a₄ = 8 := by
sorry

end sum_of_coefficients_l4040_404087


namespace books_per_shelf_l4040_404044

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 504) (h2 : num_shelves = 9) :
  total_books / num_shelves = 56 := by
sorry

end books_per_shelf_l4040_404044


namespace richards_score_l4040_404074

/-- Richard and Bruno's miniature golf scores -/
def miniature_golf (richard_score bruno_score : ℕ) : Prop :=
  bruno_score = richard_score - 14 ∧ bruno_score = 48

theorem richards_score : ∃ (richard_score : ℕ), miniature_golf richard_score 48 ∧ richard_score = 62 := by
  sorry

end richards_score_l4040_404074


namespace simplify_expression_l4040_404088

theorem simplify_expression (p : ℝ) : ((6*p+2)-3*p*5)^2 + (5-2/4)*(8*p-12) = 81*p^2 - 50 := by
  sorry

end simplify_expression_l4040_404088


namespace expression_evaluation_l4040_404026

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 1) :
  (a^2 * b - 4 * a * b^2 - 1) - 3 * (a * b^2 - 2 * a^2 * b + 1) = 10 := by
  sorry

end expression_evaluation_l4040_404026


namespace curve_through_center_l4040_404045

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a curve
structure Curve where
  path : ℝ → ℝ × ℝ

-- Define the property of dividing the square into equal areas
def divides_equally (γ : Curve) (s : Square) : Prop :=
  ∃ (area1 area2 : ℝ), area1 = area2 ∧ area1 + area2 = s.side * s.side

-- Define the property of a line segment passing through a point
def passes_through (a b c : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ c = (1 - t) • a + t • b

-- The main theorem
theorem curve_through_center (s : Square) (γ : Curve) 
  (h : divides_equally γ s) :
  ∃ (a b : ℝ × ℝ), (∃ (t1 t2 : ℝ), γ.path t1 = a ∧ γ.path t2 = b) ∧ 
    passes_through a b s.center :=
sorry

end curve_through_center_l4040_404045


namespace simplify_expression_l4040_404072

theorem simplify_expression : 
  (((Real.sqrt 2 - 1) ^ (-(Real.sqrt 3) + Real.sqrt 5)) / 
   ((Real.sqrt 2 + 1) ^ (Real.sqrt 5 - Real.sqrt 3))) = 1 := by
  sorry

end simplify_expression_l4040_404072


namespace circle_tangent_and_intersections_l4040_404035

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*Real.sqrt 3*y + 3 = 0

-- Define point A
def A : ℝ × ℝ := (-1, 0)

-- Define line l₁
def l₁ (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define line l₂
def l₂ (x : ℝ) : Prop := x = 1

-- Define the condition for points R, M, and N
def RMN_condition (R M N : ℝ × ℝ) : Prop :=
  C R.1 R.2 → l₂ M.1 → l₂ N.1 → 
  (R.1 - N.1)^2 + (R.2 - N.2)^2 = 3 * ((R.1 - M.1)^2 + (R.2 - M.2)^2)

-- Main theorem
theorem circle_tangent_and_intersections :
  -- Length of tangent line from A to C is √6
  (∃ T : ℝ × ℝ, C T.1 T.2 ∧ (T.1 - A.1)^2 + (T.2 - A.2)^2 = 6) ∧
  -- Slope k of l₁ satisfies k = √3/3 or k = 11√3/15
  (∃ k : ℝ, (k = Real.sqrt 3 / 3 ∨ k = 11 * Real.sqrt 3 / 15) ∧
    ∃ P Q : ℝ × ℝ, C P.1 P.2 ∧ C Q.1 Q.2 ∧ l₁ k P.1 P.2 ∧ l₁ k Q.1 Q.2 ∧
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2) ∧
  -- Coordinates of M and N
  ((∃ M N : ℝ × ℝ, M = (1, 4 * Real.sqrt 3 / 3) ∧ N = (1, 2 * Real.sqrt 3) ∧
    ∀ R : ℝ × ℝ, RMN_condition R M N) ∨
   (∃ M N : ℝ × ℝ, M = (1, 2 * Real.sqrt 3 / 3) ∧ N = (1, 0) ∧
    ∀ R : ℝ × ℝ, RMN_condition R M N)) :=
by sorry

end circle_tangent_and_intersections_l4040_404035


namespace floor_sum_equals_140_l4040_404027

theorem floor_sum_equals_140 
  (p q r s : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) (pos_s : 0 < s)
  (sum_squares : p^2 + q^2 = 2512 ∧ r^2 + s^2 = 2512)
  (products : p * r = 1225 ∧ q * s = 1225) : 
  ⌊p + q + r + s⌋ = 140 := by
sorry

end floor_sum_equals_140_l4040_404027


namespace system_integer_solutions_l4040_404012

theorem system_integer_solutions (a b c d : ℤ) :
  (∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) →
  (a * d - b * c = 1 ∨ a * d - b * c = -1) :=
sorry

end system_integer_solutions_l4040_404012


namespace triangle_area_l4040_404048

/-- The area of a triangle with vertices at (-3, 7), (-7, 3), and (0, 0) in a coordinate plane is 50 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let v1 : Prod Real Real := (-3, 7)
  let v2 : Prod Real Real := (-7, 3)
  let v3 : Prod Real Real := (0, 0)

  -- Calculate the area of the triangle
  let area : Real := sorry

  -- Prove that the calculated area is equal to 50
  have h : area = 50 := by sorry

  -- Return the area
  exact 50


end triangle_area_l4040_404048


namespace pond_amphibians_l4040_404031

/-- Calculates the total number of amphibians observed in a pond -/
def total_amphibians (green_frogs : ℕ) (observed_tree_frogs : ℕ) (bullfrogs : ℕ) 
  (exotic_tree_frogs : ℕ) (salamanders : ℕ) (first_tadpole_group : ℕ) (baby_frogs : ℕ) 
  (newts : ℕ) (toads : ℕ) (caecilians : ℕ) : ℕ :=
  let total_tree_frogs := observed_tree_frogs * 3
  let second_tadpole_group := first_tadpole_group - (first_tadpole_group / 5)
  green_frogs + total_tree_frogs + bullfrogs + exotic_tree_frogs + salamanders + 
  first_tadpole_group + second_tadpole_group + baby_frogs + newts + toads + caecilians

/-- Theorem stating the total number of amphibians observed in the pond -/
theorem pond_amphibians : 
  total_amphibians 6 5 2 8 3 50 10 1 2 1 = 138 := by
  sorry

end pond_amphibians_l4040_404031


namespace screen_height_is_100_l4040_404095

/-- The height of a computer screen given the side length of a square paper and the difference between the screen height and the paper's perimeter. -/
def screen_height (square_side : ℝ) (perimeter_difference : ℝ) : ℝ :=
  4 * square_side + perimeter_difference

/-- Theorem stating that the height of the computer screen is 100 cm. -/
theorem screen_height_is_100 :
  screen_height 20 20 = 100 := by
  sorry

end screen_height_is_100_l4040_404095


namespace resistor_value_l4040_404030

/-- Given two identical resistors R₀ connected in series, with a voltmeter reading U across one resistor
    and an ammeter reading I when replacing the voltmeter, prove that R₀ = 9 Ω. -/
theorem resistor_value (R₀ : ℝ) (U I : ℝ) : 
  U = 9 → I = 2 → R₀ = 9 := by
  sorry

end resistor_value_l4040_404030


namespace sin_alpha_value_l4040_404016

theorem sin_alpha_value (α : Real) : 
  (∃ (x y : Real), x = 2 * Real.sin (30 * π / 180) ∧ 
                   y = 2 * Real.cos (30 * π / 180) ∧ 
                   x = 2 * Real.sin α ∧ 
                   y = 2 * Real.cos α) → 
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end sin_alpha_value_l4040_404016


namespace replacement_concentration_l4040_404008

/-- Proves that the concentration of the replacing solution is 25% given the initial and final conditions --/
theorem replacement_concentration (initial_concentration : ℝ) (final_concentration : ℝ) (replaced_fraction : ℝ) :
  initial_concentration = 0.40 →
  final_concentration = 0.35 →
  replaced_fraction = 1/3 →
  (1 - replaced_fraction) * initial_concentration + replaced_fraction * final_concentration = final_concentration →
  replaced_fraction * (final_concentration - initial_concentration) / replaced_fraction = 0.25 := by
  sorry

end replacement_concentration_l4040_404008


namespace min_value_trig_expression_l4040_404032

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 12)^2 = 36 :=
sorry

end min_value_trig_expression_l4040_404032


namespace max_sum_of_factors_l4040_404096

theorem max_sum_of_factors (x y : ℕ+) : 
  x.val * y.val = 48 → 
  4 ∣ x.val → 
  ∀ (a b : ℕ+), a.val * b.val = 48 → 4 ∣ a.val → a + b ≤ x + y → 
  x + y = 49 := by sorry

end max_sum_of_factors_l4040_404096


namespace roots_equation_s_value_l4040_404055

theorem roots_equation_s_value (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 6 = 0) → 
  (d^2 - n*d + 6 = 0) → 
  ((c + 2/d)^2 - r*(c + 2/d) + s = 0) → 
  ((d + 2/c)^2 - r*(d + 2/c) + s = 0) → 
  (s = 32/3) := by
sorry

end roots_equation_s_value_l4040_404055


namespace tank_capacity_proof_l4040_404017

theorem tank_capacity_proof (T : ℚ) 
  (h1 : (5/8 : ℚ) * T + 15 = (4/5 : ℚ) * T) : 
  T = 86 := by
sorry

end tank_capacity_proof_l4040_404017


namespace negation_equivalence_l4040_404098

theorem negation_equivalence :
  (¬ ∀ a b : ℝ, a = b → a^2 = a*b) ↔ (∀ a b : ℝ, a ≠ b → a^2 ≠ a*b) := by
  sorry

end negation_equivalence_l4040_404098


namespace product_base8_units_digit_l4040_404093

theorem product_base8_units_digit (a b : ℕ) (ha : a = 256) (hb : b = 72) :
  (a * b) % 8 = 0 := by
  sorry

end product_base8_units_digit_l4040_404093


namespace parallel_vectors_k_eq_two_l4040_404028

/-- Two vectors in ℝ² are parallel if and only if their components are proportional -/
axiom vector_parallel_iff_proportional {a b : ℝ × ℝ} :
  (∃ (t : ℝ), a = (t * b.1, t * b.2)) ↔ ∃ (s : ℝ), a.1 * b.2 = s * a.2 * b.1

/-- Given vectors a = (k, 2) and b = (1, 1), if a is parallel to b, then k = 2 -/
theorem parallel_vectors_k_eq_two (k : ℝ) :
  let a : ℝ × ℝ := (k, 2)
  let b : ℝ × ℝ := (1, 1)
  (∃ (t : ℝ), a = (t * b.1, t * b.2)) → k = 2 := by
sorry

end parallel_vectors_k_eq_two_l4040_404028


namespace arithmetic_sequence_log_implies_square_product_square_product_not_sufficient_for_arithmetic_sequence_log_l4040_404025

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem arithmetic_sequence_log_implies_square_product
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z)) :
  y^2 = x * z :=
sorry

theorem square_product_not_sufficient_for_arithmetic_sequence_log :
  ∃ x y z : ℝ, y^2 = x * z ∧ ¬is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z) :=
sorry

end arithmetic_sequence_log_implies_square_product_square_product_not_sufficient_for_arithmetic_sequence_log_l4040_404025


namespace min_value_on_line_l4040_404080

theorem min_value_on_line (x y : ℝ) (h : x + 2*y + 1 = 0) :
  2^x + 4^y ≥ Real.sqrt 2 := by
sorry

end min_value_on_line_l4040_404080


namespace expression_simplification_l4040_404056

theorem expression_simplification (m : ℝ) (hm : m = 2) : 
  (((m + 1) / (m - 1) + 1) / ((m + m^2) / (m^2 - 2*m + 1))) - ((2 - 2*m) / (m^2 - 1)) = 4/3 := by
  sorry

end expression_simplification_l4040_404056


namespace fraction_simplification_l4040_404067

theorem fraction_simplification (x : ℝ) : 
  (2*x - 3)/4 + (3*x + 5)/5 - (x - 1)/2 = (12*x + 15)/20 := by
  sorry

end fraction_simplification_l4040_404067


namespace fraction_multiplication_result_l4040_404003

theorem fraction_multiplication_result : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5040 = 1512 := by
  sorry

end fraction_multiplication_result_l4040_404003


namespace second_investment_amount_l4040_404094

/-- Proves that the amount of the second investment is $1500 given the conditions of the problem -/
theorem second_investment_amount
  (total_return : ℝ → ℝ → ℝ)
  (first_investment : ℝ)
  (first_return_rate : ℝ)
  (second_return_rate : ℝ)
  (total_return_rate : ℝ)
  (h1 : first_investment = 500)
  (h2 : first_return_rate = 0.07)
  (h3 : second_return_rate = 0.09)
  (h4 : total_return_rate = 0.085)
  (h5 : ∀ x, total_return first_investment x = total_return_rate * (first_investment + x))
  (h6 : ∀ x, total_return first_investment x = first_return_rate * first_investment + second_return_rate * x) :
  ∃ x, x = 1500 ∧ total_return first_investment x = total_return_rate * (first_investment + x) :=
by sorry

end second_investment_amount_l4040_404094


namespace no_linear_term_implies_m_value_l4040_404077

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x^2 - x + m) * (x - 8) = a * x^3 + b * x^2 + c) → m = -8 :=
sorry

end no_linear_term_implies_m_value_l4040_404077


namespace cannot_retile_after_replacement_l4040_404011

-- Define a type for tiles
inductive Tile
| OneByFour : Tile
| TwoByTwo : Tile

-- Define a type for a tiling of a rectangle
structure Tiling :=
  (width : ℕ)
  (height : ℕ)
  (tiles : List Tile)

-- Define a function to check if a tiling is valid
def isValidTiling (t : Tiling) : Prop :=
  -- Add conditions for a valid tiling
  sorry

-- Define a function to replace one 2×2 tile with a 1×4 tile
def replaceTile (t : Tiling) : Tiling :=
  -- Implement the replacement logic
  sorry

-- Theorem statement
theorem cannot_retile_after_replacement (t : Tiling) :
  isValidTiling t → ¬(isValidTiling (replaceTile t)) :=
by
  sorry

end cannot_retile_after_replacement_l4040_404011


namespace midpoint_sum_after_doubling_x_l4040_404006

/-- Given a segment with endpoints (10, 3) and (-4, 7), prove that the sum of the doubled x-coordinate
and the y-coordinate of the midpoint is 11. -/
theorem midpoint_sum_after_doubling_x : 
  let p1 : ℝ × ℝ := (10, 3)
  let p2 : ℝ × ℝ := (-4, 7)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let doubled_x : ℝ := 2 * midpoint.1
  doubled_x + midpoint.2 = 11 := by sorry

end midpoint_sum_after_doubling_x_l4040_404006


namespace root_of_equation_l4040_404078

theorem root_of_equation (a b c d : ℝ) (h1 : a + d = 2015) (h2 : b + c = 2015) (h3 : a ≠ c) :
  ∀ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) → x = 0 := by
  sorry

end root_of_equation_l4040_404078


namespace files_per_folder_l4040_404020

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 43 →
  deleted_files = 31 →
  num_folders = 2 →
  num_folders > 0 →
  ∃ (files_per_folder : ℕ),
    files_per_folder * num_folders = initial_files - deleted_files ∧
    files_per_folder = 6 :=
by sorry

end files_per_folder_l4040_404020


namespace horner_method_for_f_l4040_404099

/-- Horner's method representation of a polynomial -/
def horner_rep (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

/-- The original polynomial function -/
def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem horner_method_for_f :
  f 5 = horner_rep [1, 2, 1, -1, 3, -5] 5 ∧ 
  horner_rep [1, 2, 1, -1, 3, -5] 5 = 4485 :=
sorry

end horner_method_for_f_l4040_404099


namespace quadratic_function_positive_l4040_404062

/-- The quadratic function y = ax² - 2ax + 3 -/
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

/-- The set of x values we're interested in -/
def X : Set ℝ := {x | 0 < x ∧ x < 3}

/-- The set of a values that satisfy the condition -/
def A : Set ℝ := {a | -1 ≤ a ∧ a < 0} ∪ {a | 0 < a ∧ a < 3}

theorem quadratic_function_positive (a : ℝ) :
  (∀ x ∈ X, f a x > 0) ↔ a ∈ A :=
sorry

end quadratic_function_positive_l4040_404062


namespace tshirt_sale_problem_l4040_404061

theorem tshirt_sale_problem (sale_duration : ℕ) (black_price white_price : ℚ) 
  (revenue_per_minute : ℚ) (h1 : sale_duration = 25) 
  (h2 : black_price = 30) (h3 : white_price = 25) (h4 : revenue_per_minute = 220) :
  ∃ (total_shirts : ℕ), 
    (total_shirts : ℚ) / 2 * black_price + (total_shirts : ℚ) / 2 * white_price = 
      sale_duration * revenue_per_minute ∧ total_shirts = 200 := by
  sorry

end tshirt_sale_problem_l4040_404061


namespace partitioned_rectangle_is_square_l4040_404013

-- Define the structure for a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the structure for the partitioned rectangle
structure PartitionedRectangle where
  main : Rectangle
  p1 : Rectangle
  p2 : Rectangle
  p3 : Rectangle
  p4 : Rectangle
  p5 : Rectangle

-- Define the property of being a square
def isSquare (r : Rectangle) : Prop :=
  r.width = r.height

-- Define the property of having equal areas
def equalAreas (r1 r2 r3 r4 : Rectangle) : Prop :=
  r1.width * r1.height = r2.width * r2.height ∧
  r2.width * r2.height = r3.width * r3.height ∧
  r3.width * r3.height = r4.width * r4.height

-- Theorem statement
theorem partitioned_rectangle_is_square 
  (pr : PartitionedRectangle) 
  (h1 : isSquare pr.p5)
  (h2 : equalAreas pr.p1 pr.p2 pr.p3 pr.p4) :
  isSquare pr.main :=
sorry

end partitioned_rectangle_is_square_l4040_404013


namespace trent_travel_distance_l4040_404036

/-- The total distance Trent traveled -/
def total_distance (house_to_bus bus_to_library : ℕ) : ℕ :=
  2 * (house_to_bus + bus_to_library)

/-- Theorem stating that Trent's total travel distance is 22 blocks -/
theorem trent_travel_distance :
  ∃ (house_to_bus bus_to_library : ℕ),
    house_to_bus = 4 ∧
    bus_to_library = 7 ∧
    total_distance house_to_bus bus_to_library = 22 :=
by
  sorry

end trent_travel_distance_l4040_404036


namespace consecutive_page_numbers_sum_l4040_404068

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end consecutive_page_numbers_sum_l4040_404068


namespace quadratic_inequality_solution_l4040_404054

theorem quadratic_inequality_solution (p : ℝ) : 
  (∀ x, x^2 + p*x - 6 < 0 ↔ -3 < x ∧ x < 2) → p = 1 := by
  sorry

end quadratic_inequality_solution_l4040_404054


namespace calculation_proof_l4040_404086

theorem calculation_proof : (-7)^3 / 7^2 - 2^5 + 4^3 - 8 = 81 := by
  sorry

end calculation_proof_l4040_404086


namespace property_set_characterization_l4040_404014

/-- A number is a prime power if it's of the form p^k where p is prime and k ≥ 1 -/
def IsPrimePower (n : Nat) : Prop :=
  ∃ (p k : Nat), Prime p ∧ k ≥ 1 ∧ n = p^k

/-- A perfect square n satisfies the property if for all its divisors a ≥ 15, a + 15 is a prime power -/
def SatisfiesProperty (n : Nat) : Prop :=
  ∃ m : Nat, n = m^2 ∧ ∀ a : Nat, a ≥ 15 → a ∣ n → IsPrimePower (a + 15)

/-- The set of all perfect squares satisfying the property -/
def PropertySet : Set Nat :=
  {n : Nat | SatisfiesProperty n}

/-- The theorem stating that the set of perfect squares satisfying the property
    is exactly {1, 4, 9, 16, 49, 64, 196} -/
theorem property_set_characterization :
  PropertySet = {1, 4, 9, 16, 49, 64, 196} := by
  sorry


end property_set_characterization_l4040_404014


namespace binary_to_quaternary_conversion_l4040_404073

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List (Fin 4) := sorry

/-- The binary representation of the number 10110010 -/
def binary_number : List Bool := [true, false, true, true, false, false, true, false]

/-- The quaternary representation of the number 2302 -/
def quaternary_number : List (Fin 4) := [2, 3, 0, 2]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_number) = quaternary_number := by sorry

end binary_to_quaternary_conversion_l4040_404073


namespace union_equals_A_implies_m_zero_or_three_l4040_404029

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equals_A_implies_m_zero_or_three (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end union_equals_A_implies_m_zero_or_three_l4040_404029


namespace village_population_distribution_l4040_404004

theorem village_population_distribution (pop_20k_to_50k : ℝ) (pop_under_20k : ℝ) (pop_50k_and_above : ℝ) :
  pop_20k_to_50k = 45 →
  pop_under_20k = 30 →
  pop_50k_and_above = 25 →
  pop_20k_to_50k + pop_under_20k = 75 :=
by sorry

end village_population_distribution_l4040_404004


namespace total_timeout_time_is_185_l4040_404007

/-- Calculates the total time spent in time-out given the number of running time-outs and the duration of each time-out. -/
def total_timeout_time (running_timeouts : ℕ) (timeout_duration : ℕ) : ℕ :=
  let food_throwing_timeouts := 5 * running_timeouts - 1
  let swearing_timeouts := food_throwing_timeouts / 3
  let total_timeouts := running_timeouts + food_throwing_timeouts + swearing_timeouts
  total_timeouts * timeout_duration

/-- Proves that the total time spent in time-out is 185 minutes given the specified conditions. -/
theorem total_timeout_time_is_185 : 
  total_timeout_time 5 5 = 185 := by
  sorry

#eval total_timeout_time 5 5

end total_timeout_time_is_185_l4040_404007


namespace no_negative_roots_l4040_404041

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 3*x^3 - 2*x^2 - 4*x + 1 ≠ 0 :=
by
  sorry

end no_negative_roots_l4040_404041


namespace variation_problem_l4040_404022

/-- Given that R varies directly as S and inversely as T^2, prove that when R = 50 and T = 5, S = 5000/3 -/
theorem variation_problem (c : ℝ) (R S T : ℝ → ℝ) (t : ℝ) :
  (∀ t, R t = c * S t / (T t)^2) →  -- Relationship between R, S, and T
  R 0 = 3 →                        -- Initial condition for R
  S 0 = 16 →                       -- Initial condition for S
  T 0 = 2 →                        -- Initial condition for T
  R t = 50 →                       -- New value for R
  T t = 5 →                        -- New value for T
  S t = 5000 / 3 := by             -- Prove that S equals 5000/3
sorry


end variation_problem_l4040_404022


namespace pencil_exchange_coloring_l4040_404043

-- Define a permutation as a bijective function from ℕ to ℕ
def Permutation (n : ℕ) := {f : ℕ → ℕ // Function.Bijective f ∧ ∀ i, i ≥ n → f i = i}

-- Define a coloring as a function from ℕ to a three-element type
def Coloring (n : ℕ) := ℕ → Fin 3

-- The main theorem
theorem pencil_exchange_coloring (n : ℕ) (p : Permutation n) :
  ∃ c : Coloring n, ∀ i < n, c i ≠ c (p.val i) :=
sorry

end pencil_exchange_coloring_l4040_404043


namespace mass_percentage_K_is_23_81_l4040_404005

/-- The mass percentage of K in a compound -/
def mass_percentage_K : ℝ := 23.81

/-- Theorem stating that the mass percentage of K in the compound is 23.81% -/
theorem mass_percentage_K_is_23_81 :
  mass_percentage_K = 23.81 := by sorry

end mass_percentage_K_is_23_81_l4040_404005


namespace sum_of_segments_eq_165_l4040_404091

/-- The sum of lengths of all possible line segments formed by dividing a line segment of length 9 into 9 equal parts -/
def sum_of_segments : ℕ :=
  let n : ℕ := 9  -- number of divisions
  (n * (n + 1) * (n + 2)) / 6

/-- Theorem: The sum of lengths of all possible line segments formed by dividing a line segment of length 9 into 9 equal parts is equal to 165 -/
theorem sum_of_segments_eq_165 : sum_of_segments = 165 := by
  sorry

end sum_of_segments_eq_165_l4040_404091


namespace right_triangle_leg_square_l4040_404053

/-- In a right triangle with hypotenuse c and legs a and b, where c = a + 2, 
    the square of b is equal to 4a + 4. -/
theorem right_triangle_leg_square (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Right triangle condition
  (h_diff : c = a + 2)         -- Hypotenuse and leg difference condition
  : b^2 = 4*a + 4 := by
sorry

end right_triangle_leg_square_l4040_404053


namespace petya_has_five_five_ruble_coins_l4040_404060

/-- Represents the coin denominations --/
inductive Denomination
  | One
  | Two
  | Five
  | Ten

/-- Represents Petya's coin collection --/
structure CoinCollection where
  total : Nat
  not_two : Nat
  not_ten : Nat
  not_one : Nat

/-- Calculates the number of five-ruble coins in the collection --/
def count_five_ruble_coins (c : CoinCollection) : Nat :=
  c.total - ((c.total - c.not_two) + (c.total - c.not_ten) + (c.total - c.not_one))

/-- Theorem stating that Petya has 5 five-ruble coins --/
theorem petya_has_five_five_ruble_coins :
  let petya_coins : CoinCollection := {
    total := 25,
    not_two := 19,
    not_ten := 20,
    not_one := 16
  }
  count_five_ruble_coins petya_coins = 5 := by
  sorry

#eval count_five_ruble_coins {
  total := 25,
  not_two := 19,
  not_ten := 20,
  not_one := 16
}

end petya_has_five_five_ruble_coins_l4040_404060
