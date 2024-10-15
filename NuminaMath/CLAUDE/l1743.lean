import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_lengths_and_circumradius_l1743_174365

/-- Given a triangle ABC with side lengths a, b, and c satisfying the equation,
    prove that the side lengths are 3, 4, 5 and the circumradius is 2.5 -/
theorem triangle_side_lengths_and_circumradius 
  (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) : 
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (2.5 : ℝ) = (1/2 : ℝ) * c := by
  sorry

#check triangle_side_lengths_and_circumradius

end NUMINAMATH_CALUDE_triangle_side_lengths_and_circumradius_l1743_174365


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1743_174364

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1743_174364


namespace NUMINAMATH_CALUDE_simplify_expression_l1743_174321

variable (y : ℝ)

theorem simplify_expression :
  3 * y - 5 * y^2 + 7 - (6 - 3 * y + 5 * y^2 - 2 * y^3) = 2 * y^3 - 10 * y^2 + 6 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1743_174321


namespace NUMINAMATH_CALUDE_election_majority_l1743_174383

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6500 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 1300 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l1743_174383


namespace NUMINAMATH_CALUDE_water_displaced_by_cube_l1743_174394

/-- The volume of water displaced by a partially submerged cube in a cylinder -/
theorem water_displaced_by_cube (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h_cube_side : cube_side = 10)
  (h_cylinder_radius : cylinder_radius = 5)
  (h_cylinder_height : cylinder_height = 12) :
  ∃ (v : ℝ), v = 75 * Real.sqrt 3 ∧ v^2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_by_cube_l1743_174394


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1743_174371

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1743_174371


namespace NUMINAMATH_CALUDE_parallel_vector_sum_diff_l1743_174305

/-- Given two vectors a and b in ℝ², where a = (1, -1) and b = (t, 1),
    if a + b is parallel to a - b, then t = -1. -/
theorem parallel_vector_sum_diff (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![t, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) → t = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_sum_diff_l1743_174305


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt3_squared_l1743_174320

theorem opposite_of_negative_sqrt3_squared : -((-Real.sqrt 3)^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt3_squared_l1743_174320


namespace NUMINAMATH_CALUDE_no_divisible_by_four_exists_l1743_174341

theorem no_divisible_by_four_exists : 
  ¬ ∃ (B : ℕ), B < 10 ∧ (8000000 + 100000 * B + 4000 + 635 + 1) % 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_four_exists_l1743_174341


namespace NUMINAMATH_CALUDE_largest_coin_distribution_l1743_174397

theorem largest_coin_distribution (n : ℕ) : n ≤ 108 ∧ n < 120 ∧ ∃ (k : ℕ), n = 15 * k + 3 →
  ∀ m : ℕ, m < 120 ∧ ∃ (k : ℕ), m = 15 * k + 3 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_l1743_174397


namespace NUMINAMATH_CALUDE_share_distribution_l1743_174391

theorem share_distribution (total : ℚ) (a b c d : ℚ) 
  (h1 : total = 1000)
  (h2 : a = b + 100)
  (h3 : a = c - 100)
  (h4 : d = b - 50)
  (h5 : d = a + 150)
  (h6 : a + b + c + d = total) :
  a = 212.5 ∧ b = 112.5 ∧ c = 312.5 ∧ d = 362.5 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l1743_174391


namespace NUMINAMATH_CALUDE_bag_of_balls_l1743_174354

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 17)
  (h4 : red = 3)
  (h5 : purple = 1)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 95 / 100) :
  white + green + yellow + red + purple = 80 := by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l1743_174354


namespace NUMINAMATH_CALUDE_percentage_calculation_l1743_174345

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * (30 / 100) * (50 / 100) * 5200 = 117 → P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1743_174345


namespace NUMINAMATH_CALUDE_workshop_workers_l1743_174359

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers (avg_salary : ℕ) (tech_count : ℕ) (tech_salary : ℕ) (non_tech_salary : ℕ) :
  avg_salary = 8000 →
  tech_count = 7 →
  tech_salary = 12000 →
  non_tech_salary = 6000 →
  ∃ (total_workers : ℕ), 
    (tech_count * tech_salary + (total_workers - tech_count) * non_tech_salary) / total_workers = avg_salary ∧
    total_workers = 21 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l1743_174359


namespace NUMINAMATH_CALUDE_haley_concert_spending_l1743_174337

/-- Calculates the total cost of concert tickets based on a pricing structure -/
def calculate_total_cost (initial_price : ℕ) (discounted_price : ℕ) (initial_quantity : ℕ) (discounted_quantity : ℕ) : ℕ :=
  initial_price * initial_quantity + discounted_price * discounted_quantity

/-- Proves that Haley's total spending on concert tickets is $27 -/
theorem haley_concert_spending :
  let initial_price : ℕ := 4
  let discounted_price : ℕ := 3
  let initial_quantity : ℕ := 3
  let discounted_quantity : ℕ := 5
  calculate_total_cost initial_price discounted_price initial_quantity discounted_quantity = 27 :=
by
  sorry

#eval calculate_total_cost 4 3 3 5

end NUMINAMATH_CALUDE_haley_concert_spending_l1743_174337


namespace NUMINAMATH_CALUDE_tshirt_original_price_l1743_174399

/-- Proves that the original price of a t-shirt is $20 given the conditions of the problem -/
theorem tshirt_original_price 
  (num_friends : ℕ) 
  (discount_percent : ℚ) 
  (total_spent : ℚ) : 
  num_friends = 4 → 
  discount_percent = 1/2 → 
  total_spent = 40 → 
  (total_spent / num_friends) / (1 - discount_percent) = 20 := by
sorry

end NUMINAMATH_CALUDE_tshirt_original_price_l1743_174399


namespace NUMINAMATH_CALUDE_vasya_drove_two_fifths_l1743_174395

/-- Represents the fraction of total distance driven by each person -/
structure DriverDistances where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- The conditions of the driving problem -/
def drivingConditions (d : DriverDistances) : Prop :=
  d.anton = d.vasya / 2 ∧
  d.sasha = d.anton + d.dima ∧
  d.dima = 1 / 10 ∧
  d.anton + d.vasya + d.sasha + d.dima = 1

theorem vasya_drove_two_fifths :
  ∀ d : DriverDistances, drivingConditions d → d.vasya = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_drove_two_fifths_l1743_174395


namespace NUMINAMATH_CALUDE_negation_square_positive_negation_root_equation_negation_sum_positive_negation_prime_odd_l1743_174300

-- 1. The square of every natural number is positive.
theorem negation_square_positive : 
  (∀ n : ℕ, n^2 > 0) ↔ ¬(∃ n : ℕ, ¬(n^2 > 0)) :=
by sorry

-- 2. Every real number x is a root of the equation 5x-12=0.
theorem negation_root_equation : 
  (∀ x : ℝ, 5*x - 12 = 0) ↔ ¬(∃ x : ℝ, 5*x - 12 ≠ 0) :=
by sorry

-- 3. For every real number x, there exists a real number y such that x+y>0.
theorem negation_sum_positive : 
  (∀ x : ℝ, ∃ y : ℝ, x + y > 0) ↔ ¬(∃ x : ℝ, ∀ y : ℝ, x + y ≤ 0) :=
by sorry

-- 4. Some prime numbers are odd.
theorem negation_prime_odd : 
  (∃ p : ℕ, Prime p ∧ Odd p) ↔ ¬(∀ p : ℕ, Prime p → ¬Odd p) :=
by sorry

end NUMINAMATH_CALUDE_negation_square_positive_negation_root_equation_negation_sum_positive_negation_prime_odd_l1743_174300


namespace NUMINAMATH_CALUDE_carla_wins_one_l1743_174306

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Carla : Player

/-- Represents the result of a game for a player -/
inductive GameResult : Type
| Win : GameResult
| Loss : GameResult

/-- The number of games each player plays against each other player -/
def gamesPerPair : Nat := 2

/-- The total number of games in the tournament -/
def totalGames : Nat := 12

/-- The number of wins for a given player -/
def wins (p : Player) : Nat :=
  match p with
  | Player.Alice => 5
  | Player.Bob => 6
  | Player.Carla => 1  -- This is what we want to prove

/-- The number of losses for a given player -/
def losses (p : Player) : Nat :=
  match p with
  | Player.Alice => 3
  | Player.Bob => 2
  | Player.Carla => 5

theorem carla_wins_one :
  (∀ p : Player, wins p + losses p = totalGames / 2) ∧
  (wins Player.Alice + wins Player.Bob + wins Player.Carla = totalGames) :=
by sorry

end NUMINAMATH_CALUDE_carla_wins_one_l1743_174306


namespace NUMINAMATH_CALUDE_domain_of_g_l1743_174315

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 1 5

-- Define the new function g(x) = f(2x - 3)
def g (x : ℝ) : ℝ := f (2 * x - 3)

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1743_174315


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l1743_174348

theorem systematic_sampling_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (h1 : total_students = 52) 
  (h2 : selected_students = 10) :
  (1 - 2 / total_students) * (selected_students / (total_students - 2)) = 5 / 26 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l1743_174348


namespace NUMINAMATH_CALUDE_parabola_directrix_l1743_174310

/-- Represents a parabola with equation y = -4x^2 + 4 -/
structure Parabola where
  /-- The y-coordinate of the focus -/
  f : ℝ
  /-- The y-coordinate of the directrix -/
  d : ℝ

/-- Theorem: The directrix of the parabola y = -4x^2 + 4 is y = 65/16 -/
theorem parabola_directrix (p : Parabola) : p.d = 65/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1743_174310


namespace NUMINAMATH_CALUDE_dice_probability_l1743_174398

def number_of_dice : ℕ := 8
def probability_even : ℚ := 1/2
def probability_odd : ℚ := 1/2

theorem dice_probability :
  (number_of_dice.choose (number_of_dice / 2)) * 
  (probability_even ^ (number_of_dice / 2)) * 
  (probability_odd ^ (number_of_dice / 2)) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1743_174398


namespace NUMINAMATH_CALUDE_watsonville_marching_band_max_members_l1743_174309

theorem watsonville_marching_band_max_members
  (m : ℕ)
  (band_size : ℕ)
  (h1 : band_size = 30 * m)
  (h2 : band_size % 31 = 7)
  (h3 : band_size < 1500) :
  band_size ≤ 720 ∧ ∃ (k : ℕ), 30 * k = 720 ∧ 720 % 31 = 7 :=
sorry

end NUMINAMATH_CALUDE_watsonville_marching_band_max_members_l1743_174309


namespace NUMINAMATH_CALUDE_train_length_l1743_174330

/-- Given a train that passes a pole in 11 seconds and a 120 m long platform in 22 seconds, 
    its length is 120 meters. -/
theorem train_length (pole_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) 
    (h1 : pole_time = 11)
    (h2 : platform_time = 22)
    (h3 : platform_length = 120) : ℝ :=
  by sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1743_174330


namespace NUMINAMATH_CALUDE_hexagon_CF_length_l1743_174311

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  square_side : ℝ
  other_side : ℝ
  h_square : square_side = 20
  h_other : other_side = 23
  h_square_ABDE : A.x = 0 ∧ A.y = 0 ∧ B.x = square_side ∧ B.y = 0 ∧
                  D.x = square_side ∧ D.y = square_side ∧ E.x = 0 ∧ E.y = square_side
  h_parallel : C.x = B.x ∧ F.x = A.x
  h_BC : (C.x - B.x)^2 + (C.y - B.y)^2 = other_side^2
  h_CD : (D.x - C.x)^2 + (D.y - C.y)^2 = other_side^2
  h_EF : (F.x - E.x)^2 + (F.y - E.y)^2 = other_side^2
  h_FA : (A.x - F.x)^2 + (A.y - F.y)^2 = other_side^2

/-- The theorem to be proved -/
theorem hexagon_CF_length (h : Hexagon) :
  ∃ n : ℕ, n = 28 ∧ n = ⌊Real.sqrt ((h.C.x - h.F.x)^2 + (h.C.y - h.F.y)^2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_hexagon_CF_length_l1743_174311


namespace NUMINAMATH_CALUDE_duck_travel_east_l1743_174342

def days_to_south : ℕ := 40
def days_to_north : ℕ := 2 * days_to_south
def total_days : ℕ := 180

def days_to_east : ℕ := total_days - days_to_south - days_to_north

theorem duck_travel_east : days_to_east = 60 := by
  sorry

end NUMINAMATH_CALUDE_duck_travel_east_l1743_174342


namespace NUMINAMATH_CALUDE_thirty_divisor_numbers_l1743_174374

def is_valid_number (n : ℕ) : Prop :=
  (n % 30 = 0) ∧ (Nat.divisors n).card = 30

def valid_numbers : Finset ℕ := {720, 1200, 1620, 4050, 7500, 11250}

theorem thirty_divisor_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n ∈ valid_numbers := by
  sorry

end NUMINAMATH_CALUDE_thirty_divisor_numbers_l1743_174374


namespace NUMINAMATH_CALUDE_tyler_double_flips_l1743_174323

/-- Represents the number of flips in a single move for each gymnast -/
def triple_flip : ℕ := 3
def double_flip : ℕ := 2

/-- Represents the number of triple-flips Jen performed -/
def jen_triple_flips : ℕ := 16

/-- Calculates the total number of flips Jen performed -/
def jen_total_flips : ℕ := jen_triple_flips * triple_flip

/-- Calculates the total number of flips Tyler performed -/
def tyler_total_flips : ℕ := jen_total_flips / 2

/-- Theorem: Given the conditions, Tyler performed 12 double-flips -/
theorem tyler_double_flips : tyler_total_flips / double_flip = 12 := by
  sorry

end NUMINAMATH_CALUDE_tyler_double_flips_l1743_174323


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l1743_174339

theorem quadratic_equation_integer_roots (m : ℕ) (a : ℝ) :
  (1 ≤ m) →
  (m ≤ 50) →
  (∃ x₁ x₂ : ℕ, 
    x₁ ≠ x₂ ∧
    (x₁ - 2)^2 + (a - m)^2 = 2 * m * x₁ + a^2 - 2 * a * m ∧
    (x₂ - 2)^2 + (a - m)^2 = 2 * m * x₂ + a^2 - 2 * a * m) →
  ∃ k : ℕ, m = k^2 ∧ k^2 ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l1743_174339


namespace NUMINAMATH_CALUDE_square_puzzle_l1743_174313

theorem square_puzzle (n : ℕ) 
  (h1 : n^2 + 20 = (n + 1)^2 - 9) : n = 14 ∧ n^2 + 20 = 216 := by
  sorry

#check square_puzzle

end NUMINAMATH_CALUDE_square_puzzle_l1743_174313


namespace NUMINAMATH_CALUDE_number_divided_by_three_l1743_174360

theorem number_divided_by_three : ∃ n : ℝ, n / 3 = 10 ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l1743_174360


namespace NUMINAMATH_CALUDE_half_of_number_l1743_174334

theorem half_of_number (N : ℚ) : 
  (4/15 * 5/7 * N) - (4/9 * 2/5 * N) = 24 → N/2 = 945 := by
sorry

end NUMINAMATH_CALUDE_half_of_number_l1743_174334


namespace NUMINAMATH_CALUDE_gcd_of_135_and_81_l1743_174347

theorem gcd_of_135_and_81 : Nat.gcd 135 81 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_135_and_81_l1743_174347


namespace NUMINAMATH_CALUDE_log_equality_implies_p_q_equal_three_l1743_174333

theorem log_equality_implies_p_q_equal_three (p q : ℝ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) 
  (h_log : Real.log p + Real.log q = Real.log (2*p + q)) : 
  p = 3 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_p_q_equal_three_l1743_174333


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1743_174393

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6*I
  let z₂ : ℂ := 4 - 6*I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1743_174393


namespace NUMINAMATH_CALUDE_circle_circumference_irrational_l1743_174326

/-- The circumference of a circle with rational radius is irrational -/
theorem circle_circumference_irrational (r : ℚ) : 
  Irrational (2 * Real.pi * (r : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_circle_circumference_irrational_l1743_174326


namespace NUMINAMATH_CALUDE_castle_doors_problem_l1743_174351

theorem castle_doors_problem (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_castle_doors_problem_l1743_174351


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1743_174307

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℝ) :
  -- The axis of symmetry is at x = 3.5
  (∀ x : ℝ, QuadraticFunction a b c (3.5 - x) = QuadraticFunction a b c (3.5 + x)) →
  -- The function passes through the point (2, -1)
  QuadraticFunction a b c 2 = -1 →
  -- p(5) is an integer
  ∃ n : ℤ, QuadraticFunction a b c 5 = n →
  -- Then p(5) = -1
  QuadraticFunction a b c 5 = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1743_174307


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1743_174385

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 5)^2 + (y - 3)^2 + (z - 1)^2 = 0 → x + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1743_174385


namespace NUMINAMATH_CALUDE_jake_has_seven_peaches_l1743_174380

-- Define the number of peaches and apples for Steven and Jake
def steven_peaches : ℕ := 19
def steven_apples : ℕ := 14
def jake_peaches : ℕ := steven_peaches - 12
def jake_apples : ℕ := steven_apples + 79

-- Theorem to prove
theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_peaches_l1743_174380


namespace NUMINAMATH_CALUDE_stone_to_crystal_ratio_is_two_to_one_l1743_174349

/-- A bracelet making scenario with Nancy and Rose -/
structure BraceletScenario where
  beads_per_bracelet : ℕ
  nancy_metal_beads : ℕ
  nancy_pearl_beads : ℕ
  rose_crystal_beads : ℕ
  total_bracelets : ℕ

/-- Calculate the ratio of Rose's stone beads to crystal beads -/
def stone_to_crystal_ratio (scenario : BraceletScenario) : ℚ :=
  let total_beads := scenario.total_bracelets * scenario.beads_per_bracelet
  let nancy_total_beads := scenario.nancy_metal_beads + scenario.nancy_pearl_beads
  let rose_total_beads := total_beads - nancy_total_beads
  let rose_stone_beads := rose_total_beads - scenario.rose_crystal_beads
  (rose_stone_beads : ℚ) / scenario.rose_crystal_beads

/-- The given bracelet scenario -/
def given_scenario : BraceletScenario :=
  { beads_per_bracelet := 8
  , nancy_metal_beads := 40
  , nancy_pearl_beads := 60  -- 40 + 20
  , rose_crystal_beads := 20
  , total_bracelets := 20 }

theorem stone_to_crystal_ratio_is_two_to_one :
  stone_to_crystal_ratio given_scenario = 2 := by
  sorry

end NUMINAMATH_CALUDE_stone_to_crystal_ratio_is_two_to_one_l1743_174349


namespace NUMINAMATH_CALUDE_string_length_problem_l1743_174379

/-- The length of strings problem -/
theorem string_length_problem (red white blue : ℝ) : 
  red = 8 → 
  white = 5 * red → 
  blue = 8 * white → 
  blue = 320 := by
  sorry

end NUMINAMATH_CALUDE_string_length_problem_l1743_174379


namespace NUMINAMATH_CALUDE_calculate_expression_l1743_174302

theorem calculate_expression : 
  2 * Real.sin (π / 4) + |(-Real.sqrt 2)| - (π - 2023)^0 - Real.sqrt 2 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1743_174302


namespace NUMINAMATH_CALUDE_age_puzzle_l1743_174317

theorem age_puzzle (A N : ℕ) (h1 : A = 30) (h2 : (A + 5) * N - (A - 5) * N = A) : N = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l1743_174317


namespace NUMINAMATH_CALUDE_midSectionAreaProperty_l1743_174384

-- Define a right triangular pyramid
structure RightTriangularPyramid where
  -- We don't need to define all properties, just the essential ones for our theorem
  obliqueFace : Set (ℝ × ℝ)  -- Representing the base as a set of points in 2D
  midSection : Set (ℝ × ℝ)   -- Representing a mid-section as a set of points in 2D

-- Define the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem midSectionAreaProperty (p : RightTriangularPyramid) :
  area p.midSection = (1/4) * area p.obliqueFace := by sorry

end NUMINAMATH_CALUDE_midSectionAreaProperty_l1743_174384


namespace NUMINAMATH_CALUDE_solution_sets_l1743_174361

-- Define the set A as (-∞, 1)
def A : Set ℝ := Set.Iio 1

-- Define the solution set B
def B (a : ℝ) : Set ℝ :=
  if a < -1 then Set.Icc a (-1)
  else if a = -1 then {-1}
  else if -1 < a ∧ a < 0 then Set.Icc (-1) a
  else ∅

-- Theorem statement
theorem solution_sets (a : ℝ) (h1 : A = {x | a * x + (-2 * a) > 0}) :
  B a = {x | (a * x - (-2 * a)) * (x - a) ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_l1743_174361


namespace NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_ten_l1743_174332

theorem exists_n_pow_half_n_eq_ten : ∃ n : ℝ, n ^ (n / 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_pow_half_n_eq_ten_l1743_174332


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1743_174322

/-- Given two parallel lines 3x - 2y - 1 = 0 and 6x + ay + c = 0 with a distance of 2√13/13 between them, prove that (c + 2)/a = 1 -/
theorem parallel_lines_distance (a c : ℝ) : 
  (∀ x y : ℝ, 3 * x - 2 * y - 1 = 0 ↔ 6 * x + a * y + c = 0) →  -- lines are equivalent
  (∃ k : ℝ, k ≠ 0 ∧ 3 = k * 6 ∧ -2 = k * a) →  -- lines are parallel
  (|c/2 + 1| / Real.sqrt 13 = 2 * Real.sqrt 13 / 13) →  -- distance between lines
  (c + 2) / a = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1743_174322


namespace NUMINAMATH_CALUDE_hide_and_seek_players_l1743_174304

-- Define variables for each person
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis) ∧ ¬Andrew ∧ ¬Gena :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_players_l1743_174304


namespace NUMINAMATH_CALUDE_roots_depend_on_k_l1743_174338

theorem roots_depend_on_k : 
  ∀ (k : ℝ), 
  ∃ (δ : ℝ), 
  δ = 1 + 4*k ∧ 
  (δ > 0 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁ - 1)*(x₁ - 2) = k ∧ (x₂ - 1)*(x₂ - 2) = k) ∧
  (δ = 0 → ∃ (x : ℝ), (x - 1)*(x - 2) = k) ∧
  (δ < 0 → ¬∃ (x : ℝ), (x - 1)*(x - 2) = k) :=
by sorry


end NUMINAMATH_CALUDE_roots_depend_on_k_l1743_174338


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l1743_174388

def hop_distance (n : ℕ) (remaining : ℚ) : ℚ :=
  if n % 2 = 1 then remaining / 2 else remaining / 4

def total_distance (hops : ℕ) : ℚ :=
  let rec aux (n : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if n = 0 then acc
    else
      let dist := hop_distance n remaining
      aux (n - 1) (remaining - dist) (acc + dist)
  aux hops 2 0

theorem kevin_kangaroo_hops :
  total_distance 6 = 485 / 256 := by
  sorry

#eval total_distance 6

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l1743_174388


namespace NUMINAMATH_CALUDE_complex_number_range_l1743_174366

theorem complex_number_range (a : ℝ) (z : ℂ) : 
  z = 2 + (a + 1) * I → Complex.abs z < 2 * Real.sqrt 2 → -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_range_l1743_174366


namespace NUMINAMATH_CALUDE_inverse_mod_two_million_l1743_174377

/-- The multiplicative inverse of (222222 * 142857) modulo 2,000,000 is 126. -/
theorem inverse_mod_two_million : ∃ N : ℕ, 
  N < 1000000 ∧ (N * (222222 * 142857)) % 2000000 = 1 :=
by
  use 126
  sorry

end NUMINAMATH_CALUDE_inverse_mod_two_million_l1743_174377


namespace NUMINAMATH_CALUDE_max_regions_lines_theorem_max_regions_circles_theorem_l1743_174362

/-- The maximum number of regions in a plane divided by n lines -/
def max_regions_lines (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- The maximum number of regions in a plane divided by n circles -/
def max_regions_circles (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem: The maximum number of regions in a plane divided by n lines is (n^2 + n + 2) / 2 -/
theorem max_regions_lines_theorem (n : ℕ) :
  max_regions_lines n = (n^2 + n + 2) / 2 := by sorry

/-- Theorem: The maximum number of regions in a plane divided by n circles is n^2 - n + 2 -/
theorem max_regions_circles_theorem (n : ℕ) :
  max_regions_circles n = n^2 - n + 2 := by sorry

end NUMINAMATH_CALUDE_max_regions_lines_theorem_max_regions_circles_theorem_l1743_174362


namespace NUMINAMATH_CALUDE_deck_size_proof_l1743_174325

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  r + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_proof_l1743_174325


namespace NUMINAMATH_CALUDE_decreasing_interval_of_symmetric_quadratic_l1743_174389

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem decreasing_interval_of_symmetric_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x ∈ Set.range (f a b)) →
  (∀ x, f a b x = f a b (-x)) →
  a ≠ 0 →
  ∃ (l r : ℝ), l = -2/3 ∧ r = 0 ∧
    ∀ x y, l ≤ x ∧ x < y ∧ y ≤ r → f a b y < f a b x :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_symmetric_quadratic_l1743_174389


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l1743_174392

theorem trig_identity_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.sin (x - y) - Real.cos (x + y) * Real.cos (x - y) = -Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l1743_174392


namespace NUMINAMATH_CALUDE_sum_of_81_and_15_l1743_174327

theorem sum_of_81_and_15 : 81 + 15 = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_81_and_15_l1743_174327


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l1743_174363

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the probability of stacking crates to a specific height -/
def stackProbability (dimensions : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of stacking 15 crates to 50ft -/
theorem crate_stacking_probability :
  let dimensions : CrateDimensions := ⟨2, 3, 5⟩
  stackProbability dimensions 15 50 = 1162161 / 14348907 := by
  sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l1743_174363


namespace NUMINAMATH_CALUDE_percentage_subtraction_l1743_174372

theorem percentage_subtraction (a : ℝ) (p : ℝ) (h : a - p * a = 0.94 * a) : p = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l1743_174372


namespace NUMINAMATH_CALUDE_sin_cos_difference_l1743_174308

theorem sin_cos_difference (x : ℝ) : 
  Real.sin (65 * π / 180 - x) * Real.cos (x - 20 * π / 180) - 
  Real.cos (65 * π / 180 - x) * Real.sin (20 * π / 180 - x) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l1743_174308


namespace NUMINAMATH_CALUDE_social_gathering_attendance_l1743_174344

theorem social_gathering_attendance
  (num_men : ℕ)
  (dances_per_man : ℕ)
  (dances_per_woman : ℕ)
  (h_num_men : num_men = 15)
  (h_dances_per_man : dances_per_man = 4)
  (h_dances_per_woman : dances_per_woman = 3) :
  (num_men * dances_per_man) / dances_per_woman = 20 := by
sorry

end NUMINAMATH_CALUDE_social_gathering_attendance_l1743_174344


namespace NUMINAMATH_CALUDE_min_sum_squares_l1743_174355

theorem min_sum_squares (a b : ℝ) (h : (9 : ℝ) / a^2 + 4 / b^2 = 1) :
  ∃ (min : ℝ), min = 25 ∧ ∀ (x y : ℝ), (9 : ℝ) / x^2 + 4 / y^2 = 1 → x^2 + y^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1743_174355


namespace NUMINAMATH_CALUDE_divisor_sum_and_totient_inequality_divisor_sum_and_totient_equality_l1743_174346

def σ (n : ℕ) : ℕ := sorry

def φ (n : ℕ) : ℕ := sorry

theorem divisor_sum_and_totient_inequality (n : ℕ) :
  n ≠ 0 → (1 : ℝ) / σ n + (1 : ℝ) / φ n ≥ 2 / n :=
sorry

theorem divisor_sum_and_totient_equality (n : ℕ) :
  n ≠ 0 → ((1 : ℝ) / σ n + (1 : ℝ) / φ n = 2 / n ↔ n = 1) :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_and_totient_inequality_divisor_sum_and_totient_equality_l1743_174346


namespace NUMINAMATH_CALUDE_g_values_l1743_174336

-- Define the function g
def g (x : ℝ) : ℝ := -2 * x^2 - 3 * x + 1

-- State the theorem
theorem g_values : g (-1) = 2 ∧ g (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_values_l1743_174336


namespace NUMINAMATH_CALUDE_no_real_solutions_l1743_174314

/-- Given a function f(x) = x^2 + 2x + a, where f(bx) = 9x^2 - 6x + 2,
    prove that the equation f(ax + b) = 0 has no real solutions. -/
theorem no_real_solutions (a b : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = x^2 + 2*x + a) ∧
   (∀ x, f (b*x) = 9*x^2 - 6*x + 2)) →
  (∀ x, (x^2 + 2*x + a) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1743_174314


namespace NUMINAMATH_CALUDE_power_product_equality_l1743_174386

theorem power_product_equality (x : ℝ) : (-2 * x^2) * (-4 * x^3) = 8 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1743_174386


namespace NUMINAMATH_CALUDE_recipe_total_cups_l1743_174382

-- Define the ratio of ingredients
def butter_ratio : ℚ := 2
def flour_ratio : ℚ := 5
def sugar_ratio : ℚ := 3

-- Define the amount of sugar used
def sugar_cups : ℚ := 9

-- Theorem statement
theorem recipe_total_cups : 
  let total_ratio := butter_ratio + flour_ratio + sugar_ratio
  let scale_factor := sugar_cups / sugar_ratio
  let total_cups := scale_factor * total_ratio
  total_cups = 30 := by sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l1743_174382


namespace NUMINAMATH_CALUDE_calculation_proof_l1743_174373

theorem calculation_proof : 2014 * (1 / 19 - 1 / 53) = 68 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1743_174373


namespace NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l1743_174318

/-- Sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | (n + 3) => 2 * (a (n + 2) + a (n + 1)) - a n

/-- Theorem: All terms in the sequence a_n are perfect squares -/
theorem all_terms_are_perfect_squares :
  ∃ x : ℕ → ℤ, ∀ n : ℕ, a n = (x n)^2 := by
  sorry

end NUMINAMATH_CALUDE_all_terms_are_perfect_squares_l1743_174318


namespace NUMINAMATH_CALUDE_expected_value_theorem_l1743_174329

def N : ℕ := 123456789

/-- The expected value of N' when two distinct digits of N are randomly swapped -/
def expected_value_N_prime : ℚ := 555555555

/-- Theorem stating that the expected value of N' is 555555555 -/
theorem expected_value_theorem : expected_value_N_prime = 555555555 := by sorry

end NUMINAMATH_CALUDE_expected_value_theorem_l1743_174329


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l1743_174312

theorem sqrt_equation_solutions :
  {x : ℝ | x ≥ 2 ∧ Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 2} =
  {8.25, 22.25} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l1743_174312


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equal_triangle_area_l1743_174357

/-- Given a triangle with sides 9, 12, and 15 units, and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter_equal_triangle_area (a b c w : ℝ) : 
  a = 9 → b = 12 → c = 15 → w = 6 → 
  (1/2) * a * b = w * ((1/2) * a * b / w) → 
  2 * (w + ((1/2) * a * b / w)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equal_triangle_area_l1743_174357


namespace NUMINAMATH_CALUDE_line_passes_through_intersections_l1743_174328

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

/-- Line equation -/
def line (x y : ℝ) : Prop := x - 2*y = 0

/-- Theorem stating that the line passes through the intersection points of the circles -/
theorem line_passes_through_intersections :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_intersections_l1743_174328


namespace NUMINAMATH_CALUDE_sum_of_repeated_addition_and_multiplication_l1743_174378

theorem sum_of_repeated_addition_and_multiplication (m n : ℕ+) :
  (m.val * 2) + (3 ^ n.val) = 2 * m.val + 3 ^ n.val := by sorry

end NUMINAMATH_CALUDE_sum_of_repeated_addition_and_multiplication_l1743_174378


namespace NUMINAMATH_CALUDE_next_joint_performance_l1743_174356

theorem next_joint_performance (ella_interval : Nat) (felix_interval : Nat) 
  (grace_interval : Nat) (hugo_interval : Nat) 
  (h1 : ella_interval = 5)
  (h2 : felix_interval = 6)
  (h3 : grace_interval = 9)
  (h4 : hugo_interval = 10) :
  Nat.lcm (Nat.lcm (Nat.lcm ella_interval felix_interval) grace_interval) hugo_interval = 90 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_performance_l1743_174356


namespace NUMINAMATH_CALUDE_no_unique_solution_l1743_174343

theorem no_unique_solution (a : ℝ) : ¬ ∃! p : ℝ × ℝ, 
  p.1^2 + p.2^2 = 2 ∧ |p.2| - p.1 = a :=
by
  sorry

end NUMINAMATH_CALUDE_no_unique_solution_l1743_174343


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1743_174331

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  s : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, s n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: If s_30 = s_60 for an arithmetic sequence, then s_90 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.s 30 = seq.s 60) : seq.s 90 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1743_174331


namespace NUMINAMATH_CALUDE_yw_approx_6_32_l1743_174375

/-- Triangle XYZ with W on XY -/
structure TriangleXYZW where
  /-- Point X -/
  X : ℝ × ℝ
  /-- Point Y -/
  Y : ℝ × ℝ
  /-- Point Z -/
  Z : ℝ × ℝ
  /-- Point W on XY -/
  W : ℝ × ℝ
  /-- XZ = YZ = 10 -/
  xz_eq_yz : dist X Z = dist Y Z ∧ dist X Z = 10
  /-- W is on XY -/
  w_on_xy : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ W = (1 - t) • X + t • Y
  /-- XW = 5 -/
  xw_eq_5 : dist X W = 5
  /-- ZW = 6 -/
  zw_eq_6 : dist Z W = 6

/-- The length of YW is approximately 6.32 -/
theorem yw_approx_6_32 (t : TriangleXYZW) : 
  abs (dist t.Y t.W - 6.32) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_yw_approx_6_32_l1743_174375


namespace NUMINAMATH_CALUDE_xiaoming_average_is_92_l1743_174340

/-- Calculates the weighted average of Xiao Ming's math scores -/
def xiaoming_weighted_average : ℚ :=
  let regular_score : ℚ := 89
  let midterm_score : ℚ := 91
  let final_score : ℚ := 95
  let regular_weight : ℚ := 3
  let midterm_weight : ℚ := 3
  let final_weight : ℚ := 4
  (regular_score * regular_weight + midterm_score * midterm_weight + final_score * final_weight) /
  (regular_weight + midterm_weight + final_weight)

/-- Theorem stating that Xiao Ming's weighted average math score is 92 -/
theorem xiaoming_average_is_92 : xiaoming_weighted_average = 92 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_average_is_92_l1743_174340


namespace NUMINAMATH_CALUDE_min_groups_needed_l1743_174370

def total_students : ℕ := 24
def max_group_size : ℕ := 10

theorem min_groups_needed : 
  (∃ (group_size : ℕ), 
    group_size > 0 ∧ 
    group_size ≤ max_group_size ∧ 
    total_students % group_size = 0 ∧
    total_students / group_size = 3) ∧
  (∀ (n : ℕ), 
    n > 0 → 
    n < 3 → 
    (∀ (group_size : ℕ), 
      group_size > 0 → 
      group_size ≤ max_group_size → 
      total_students % group_size = 0 → 
      total_students / group_size ≠ n)) :=
by sorry

end NUMINAMATH_CALUDE_min_groups_needed_l1743_174370


namespace NUMINAMATH_CALUDE_no_fixed_points_implies_a_range_l1743_174324

/-- A quadratic function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The property of having no fixed points -/
def has_no_fixed_points (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≠ x

/-- The main theorem -/
theorem no_fixed_points_implies_a_range (a : ℝ) :
  has_no_fixed_points a → -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_implies_a_range_l1743_174324


namespace NUMINAMATH_CALUDE_ounces_per_pound_l1743_174301

-- Define constants
def pounds_per_ton : ℝ := 2500
def gunny_bag_capacity_tons : ℝ := 13
def num_packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

-- Define the theorem
theorem ounces_per_pound : ∃ (x : ℝ), 
  (gunny_bag_capacity_tons * pounds_per_ton = num_packets * (packet_weight_pounds + packet_weight_ounces / x)) → 
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_pound_l1743_174301


namespace NUMINAMATH_CALUDE_dvds_in_book_l1743_174396

/-- Given a DVD book with a total capacity and some empty spaces,
    calculate the number of DVDs already in the book. -/
theorem dvds_in_book (total_capacity : ℕ) (empty_spaces : ℕ)
    (h1 : total_capacity = 126)
    (h2 : empty_spaces = 45) :
    total_capacity - empty_spaces = 81 := by
  sorry

end NUMINAMATH_CALUDE_dvds_in_book_l1743_174396


namespace NUMINAMATH_CALUDE_talking_segment_duration_l1743_174390

/-- Represents the duration of a radio show in minutes -/
def show_duration : ℕ := 3 * 60

/-- Represents the number of talking segments in the show -/
def num_talking_segments : ℕ := 3

/-- Represents the number of ad breaks in the show -/
def num_ad_breaks : ℕ := 5

/-- Represents the duration of each ad break in minutes -/
def ad_break_duration : ℕ := 5

/-- Represents the total duration of songs played in the show in minutes -/
def song_duration : ℕ := 125

/-- Theorem stating that each talking segment lasts 10 minutes -/
theorem talking_segment_duration :
  (show_duration - num_ad_breaks * ad_break_duration - song_duration) / num_talking_segments = 10 := by
  sorry

end NUMINAMATH_CALUDE_talking_segment_duration_l1743_174390


namespace NUMINAMATH_CALUDE_A_intersect_B_l1743_174353

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ m : ℕ, x = 2 * m}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1743_174353


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1743_174316

theorem geometric_sequence_sum_ratio 
  (a q : ℝ) 
  (h_q : q ≠ 1) : 
  let S : ℕ → ℝ := λ n => a * (1 - q^n) / (1 - q)
  (S 6 / S 3 = 1 / 2) → (S 9 / S 3 = 3 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1743_174316


namespace NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l1743_174387

theorem sum_of_roots_is_negative_one (m n : ℝ) : 
  m ≠ 0 → n ≠ 0 → (∀ x, x^2 + m*x + n = 0 ↔ (x = m ∨ x = n)) → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_negative_one_l1743_174387


namespace NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l1743_174368

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the theorem
theorem least_prime_angle_in_right_triangle :
  ∀ a b : ℕ,
    a + b = 90 →  -- Sum of acute angles in a right triangle is 90°
    a > b →        -- Given condition: a > b
    isPrime a →    -- a is prime
    isPrime b →    -- b is prime
    b ≥ 7 :=       -- The least possible value of b is 7
by
  sorry  -- Proof is omitted as per instructions


end NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l1743_174368


namespace NUMINAMATH_CALUDE_inequality_solution_l1743_174358

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (20 - x) + Real.sqrt (20 * x - x^3) ≥ 20) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1743_174358


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1743_174367

/-- Given an arithmetic sequence with first term 2 and sum of second and fourth terms 10,
    the third term is 5. -/
theorem arithmetic_sequence_third_term (a d : ℚ) : 
  a = 2 ∧ (a + d) + (a + 3*d) = 10 → a + 2*d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1743_174367


namespace NUMINAMATH_CALUDE_min_white_surface_area_l1743_174350

/-- Represents a cube with side length 4, composed of unit cubes -/
structure LargeCube :=
  (side_length : Nat)
  (total_cubes : Nat)
  (red_cubes : Nat)
  (white_cubes : Nat)

/-- The fraction of the surface area that is white when minimized -/
def min_white_fraction (c : LargeCube) : Rat :=
  5 / 96

/-- Theorem stating the minimum fraction of white surface area -/
theorem min_white_surface_area (c : LargeCube) 
  (h1 : c.side_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.red_cubes = 58)
  (h4 : c.white_cubes = 6) :
  min_white_fraction c = 5 / 96 := by
  sorry

end NUMINAMATH_CALUDE_min_white_surface_area_l1743_174350


namespace NUMINAMATH_CALUDE_increasing_function_condition_l1743_174369

open Real

/-- The function f(x) = (ln x) / x - kx is increasing on (0, +∞) iff k ≤ -1/(2e³) -/
theorem increasing_function_condition (k : ℝ) :
  (∀ x > 0, StrictMono (λ x => (log x) / x - k * x)) ↔ k ≤ -1 / (2 * (exp 3)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l1743_174369


namespace NUMINAMATH_CALUDE_boys_circle_distance_l1743_174352

theorem boys_circle_distance (n : ℕ) (r : ℝ) (h1 : n = 8) (h2 : r = 50) : 
  n * (2 * (2 * r)) = 800 := by
  sorry

end NUMINAMATH_CALUDE_boys_circle_distance_l1743_174352


namespace NUMINAMATH_CALUDE_intersection_range_l1743_174319

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (9 - p.1^2) ∧ p.2 ≠ 0}
def N (b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + b}

-- State the theorem
theorem intersection_range (b : ℝ) : 
  (M ∩ N b).Nonempty → b ∈ Set.Ioo (-3) (3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l1743_174319


namespace NUMINAMATH_CALUDE_phone_price_is_3000_l1743_174335

/-- Represents the payment plan for a phone purchase -/
structure PaymentPlan where
  initialPayment : ℕ
  monthlyPayment : ℕ
  duration : ℕ

/-- Calculates the total cost of a payment plan -/
def totalCost (plan : PaymentPlan) : ℕ :=
  plan.initialPayment + plan.monthlyPayment * (plan.duration - 1)

/-- Represents the two-part payment plan -/
structure TwoPartPlan where
  firstHalfPayment : ℕ
  secondHalfPayment : ℕ
  duration : ℕ

/-- Calculates the total cost of a two-part payment plan -/
def twoPartTotalCost (plan : TwoPartPlan) : ℕ :=
  (plan.firstHalfPayment * (plan.duration / 2)) + (plan.secondHalfPayment * (plan.duration / 2))

/-- The theorem stating that the phone price is 3000 yuan given the described payment plans -/
theorem phone_price_is_3000 (plan1 : PaymentPlan) (plan2 : TwoPartPlan) 
    (h1 : plan1.initialPayment = 800)
    (h2 : plan1.monthlyPayment = 200)
    (h3 : plan2.firstHalfPayment = 350)
    (h4 : plan2.secondHalfPayment = 150)
    (h5 : plan1.duration = plan2.duration)
    (h6 : totalCost plan1 = twoPartTotalCost plan2) :
    totalCost plan1 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_phone_price_is_3000_l1743_174335


namespace NUMINAMATH_CALUDE_scrabble_multiplier_is_three_l1743_174303

/-- Represents a three-letter word in Scrabble --/
structure ScrabbleWord where
  first_letter_value : ℕ
  middle_letter_value : ℕ
  last_letter_value : ℕ

/-- Calculates the multiplier for a given Scrabble word and final score --/
def calculate_multiplier (word : ScrabbleWord) (final_score : ℕ) : ℚ :=
  final_score / (word.first_letter_value + word.middle_letter_value + word.last_letter_value)

theorem scrabble_multiplier_is_three :
  let word : ScrabbleWord := {
    first_letter_value := 1,
    middle_letter_value := 8,
    last_letter_value := 1
  }
  let final_score : ℕ := 30
  calculate_multiplier word final_score = 3 := by
    sorry

end NUMINAMATH_CALUDE_scrabble_multiplier_is_three_l1743_174303


namespace NUMINAMATH_CALUDE_simplify_radical_product_l1743_174381

theorem simplify_radical_product : 
  (3 * 5) ^ (1/3) * (5^2 * 3^4) ^ (1/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l1743_174381


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_sum_l1743_174376

theorem binomial_expansion_coefficients_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 + 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ - 2*a₂ + 3*a₃ - 4*a₄ = 48 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_sum_l1743_174376
