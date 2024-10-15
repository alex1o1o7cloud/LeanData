import Mathlib

namespace NUMINAMATH_CALUDE_tan_150_and_pythagorean_identity_l3103_310381

theorem tan_150_and_pythagorean_identity :
  (Real.tan (150 * π / 180) = -Real.sqrt 3 / 3) ∧
  (Real.sin (150 * π / 180))^2 + (Real.cos (150 * π / 180))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_and_pythagorean_identity_l3103_310381


namespace NUMINAMATH_CALUDE_sequence_limit_l3103_310386

def a (n : ℕ) : ℚ := (7 * n + 4) / (2 * n + 1)

theorem sequence_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 7/2| < ε := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_l3103_310386


namespace NUMINAMATH_CALUDE_equation_solution_range_l3103_310333

theorem equation_solution_range (x m : ℝ) : 
  (2 * x + 4 = m - x) → (x < 0) → (m < 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3103_310333


namespace NUMINAMATH_CALUDE_expression_evaluation_l3103_310352

theorem expression_evaluation (a b : ℤ) (ha : a = 4) (hb : b = -2) :
  -a - b^4 + a*b = -28 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3103_310352


namespace NUMINAMATH_CALUDE_bella_steps_l3103_310377

-- Define the constants
def total_distance_miles : ℝ := 3
def speed_ratio : ℝ := 4
def feet_per_step : ℝ := 3
def feet_per_mile : ℝ := 5280

-- Define the theorem
theorem bella_steps : 
  ∀ (bella_speed : ℝ),
  bella_speed > 0 →
  (bella_speed * (total_distance_miles * feet_per_mile / (bella_speed * (1 + speed_ratio)))) / feet_per_step = 1056 :=
by
  sorry


end NUMINAMATH_CALUDE_bella_steps_l3103_310377


namespace NUMINAMATH_CALUDE_train_length_l3103_310327

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) →
  platform_length = 240 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 280 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3103_310327


namespace NUMINAMATH_CALUDE_molecular_weight_AlOH3_is_correct_l3103_310356

/-- The molecular weight of Al(OH)3 -/
def molecular_weight_AlOH3 : ℝ := 78

/-- The number of moles given in the problem -/
def given_moles : ℝ := 7

/-- The total molecular weight for the given number of moles -/
def total_weight : ℝ := 546

/-- Theorem stating that the molecular weight of Al(OH)3 is correct -/
theorem molecular_weight_AlOH3_is_correct :
  molecular_weight_AlOH3 = total_weight / given_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_AlOH3_is_correct_l3103_310356


namespace NUMINAMATH_CALUDE_pet_shelter_problem_l3103_310341

theorem pet_shelter_problem (total : ℕ) (apples chicken cheese : ℕ)
  (apples_chicken apples_cheese chicken_cheese : ℕ) (all_three : ℕ)
  (h_total : total = 100)
  (h_apples : apples = 20)
  (h_chicken : chicken = 70)
  (h_cheese : cheese = 10)
  (h_apples_chicken : apples_chicken = 7)
  (h_apples_cheese : apples_cheese = 3)
  (h_chicken_cheese : chicken_cheese = 5)
  (h_all_three : all_three = 2) :
  total - (apples + chicken + cheese
          - apples_chicken - apples_cheese - chicken_cheese
          + all_three) = 13 := by
  sorry

end NUMINAMATH_CALUDE_pet_shelter_problem_l3103_310341


namespace NUMINAMATH_CALUDE_f_symmetry_l3103_310340

/-- Given a function f(x) = x^3 + 2x, prove that f(a) + f(-a) = 0 for any real number a -/
theorem f_symmetry (a : ℝ) : (fun x : ℝ ↦ x^3 + 2*x) a + (fun x : ℝ ↦ x^3 + 2*x) (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3103_310340


namespace NUMINAMATH_CALUDE_sum_of_squares_first_10_base6_l3103_310346

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- Computes the sum of squares of the first n base-6 numbers --/
def sumOfSquaresBase6 (n : ℕ) : ℕ := sorry

theorem sum_of_squares_first_10_base6 :
  base10ToBase6 (sumOfSquaresBase6 10) = 231 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_10_base6_l3103_310346


namespace NUMINAMATH_CALUDE_rational_square_difference_l3103_310345

theorem rational_square_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_difference_l3103_310345


namespace NUMINAMATH_CALUDE_gis_not_just_computer_system_l3103_310331

/-- Represents a Geographic Information System (GIS) -/
structure GIS where
  provides_decision_info : Bool
  used_in_urban_management : Bool
  has_data_functions : Bool
  is_computer_system : Bool

/-- The properties of a valid GIS based on the given conditions -/
def is_valid_gis (g : GIS) : Prop :=
  g.provides_decision_info ∧
  g.used_in_urban_management ∧
  g.has_data_functions ∧
  ¬g.is_computer_system

/-- The statement to be proven false -/
def incorrect_statement (g : GIS) : Prop :=
  g.is_computer_system

theorem gis_not_just_computer_system :
  ∃ (g : GIS), is_valid_gis g ∧ ¬incorrect_statement g :=
sorry

end NUMINAMATH_CALUDE_gis_not_just_computer_system_l3103_310331


namespace NUMINAMATH_CALUDE_new_room_size_l3103_310316

/-- Given a bedroom and bathroom size, calculate the size of a new room that is twice as large as both combined -/
theorem new_room_size (bedroom : ℝ) (bathroom : ℝ) (new_room : ℝ) : 
  bedroom = 309 → bathroom = 150 → new_room = 2 * (bedroom + bathroom) → new_room = 918 := by
  sorry

end NUMINAMATH_CALUDE_new_room_size_l3103_310316


namespace NUMINAMATH_CALUDE_existence_of_functions_composition_inequality_l3103_310348

-- Part 1
theorem existence_of_functions :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f (g x) = g (f x)) ∧ 
    (∀ x, f (f x) = g (g x)) ∧ 
    (∀ x, f x ≠ g x) := by sorry

-- Part 2
theorem composition_inequality 
  (f₁ g₁ : ℝ → ℝ) 
  (h₁ : ∀ x, f₁ (g₁ x) = g₁ (f₁ x)) 
  (h₂ : ∀ x, f₁ x ≠ g₁ x) : 
  ∀ x, f₁ (f₁ x) ≠ g₁ (g₁ x) := by sorry

end NUMINAMATH_CALUDE_existence_of_functions_composition_inequality_l3103_310348


namespace NUMINAMATH_CALUDE_set_B_equals_l3103_310373

-- Define set A
def A : Set Int := {-1, 0, 1, 2}

-- Define the function f(x) = x^2 - 2x
def f (x : Int) : Int := x^2 - 2*x

-- Define set B
def B : Set Int := {y | ∃ x ∈ A, f x = y}

-- Theorem statement
theorem set_B_equals : B = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_set_B_equals_l3103_310373


namespace NUMINAMATH_CALUDE_least_valid_number_l3103_310390

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is the least positive integer divisible by 17 with digit sum 17 -/
def is_least_valid (m : ℕ) : Prop :=
  m > 0 ∧ m % 17 = 0 ∧ digit_sum m = 17 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → ¬(k % 17 = 0 ∧ digit_sum k = 17)

theorem least_valid_number : is_least_valid 476 := by sorry

end NUMINAMATH_CALUDE_least_valid_number_l3103_310390


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3103_310365

theorem trigonometric_identity : 
  Real.sin (4/3 * π) * Real.cos (11/6 * π) * Real.tan (3/4 * π) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3103_310365


namespace NUMINAMATH_CALUDE_inequality_proof_l3103_310371

theorem inequality_proof (a b c : ℝ) : 
  a = 0.1 * Real.exp 0.1 → 
  b = 1 / 9 → 
  c = -Real.log 0.9 → 
  c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3103_310371


namespace NUMINAMATH_CALUDE_ellipse_sum_a_k_l3103_310305

def ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

def focus1 : ℝ × ℝ := (2, 2)
def focus2 : ℝ × ℝ := (2, 6)
def point_on_ellipse : ℝ × ℝ := (-3, 4)

theorem ellipse_sum_a_k (h k a b : ℝ) :
  a > 0 → b > 0 →
  ellipse h k a b (point_on_ellipse.1) (point_on_ellipse.2) →
  (∀ x y, ellipse h k a b x y →
    Real.sqrt ((x - focus1.1)^2 + (y - focus1.2)^2) +
    Real.sqrt ((x - focus2.1)^2 + (y - focus2.2)^2) =
    Real.sqrt ((point_on_ellipse.1 - focus1.1)^2 + (point_on_ellipse.2 - focus1.2)^2) +
    Real.sqrt ((point_on_ellipse.1 - focus2.1)^2 + (point_on_ellipse.2 - focus2.2)^2)) →
  a + k = (Real.sqrt 29 + 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_a_k_l3103_310305


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3103_310332

theorem isosceles_triangle_perimeter : ∀ a b : ℝ,
  (a^2 - 6*a + 8 = 0) →
  (b^2 - 6*b + 8 = 0) →
  (a ≠ b) →
  (∃ c : ℝ, c = max a b ∧ c = min a b + (max a b - min a b) ∧ a + b + c = 10) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3103_310332


namespace NUMINAMATH_CALUDE_fraction_power_five_l3103_310320

theorem fraction_power_five : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_five_l3103_310320


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3103_310324

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 16) :
  a 2 + a 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3103_310324


namespace NUMINAMATH_CALUDE_rihlelo_symmetry_l3103_310317

/-- Represents a design pattern -/
structure Design where
  /-- The type of object the design is for -/
  objectType : String
  /-- The country of origin for the design -/
  origin : String
  /-- The number of lines of symmetry in the design -/
  symmetryLines : ℕ

/-- The rihlèlò design from Mozambique -/
def rihlelo : Design where
  objectType := "winnowing tray"
  origin := "Mozambique"
  symmetryLines := 4

/-- Theorem stating that the rihlèlò design has 4 lines of symmetry -/
theorem rihlelo_symmetry : rihlelo.symmetryLines = 4 := by
  sorry

end NUMINAMATH_CALUDE_rihlelo_symmetry_l3103_310317


namespace NUMINAMATH_CALUDE_beef_weight_calculation_l3103_310358

theorem beef_weight_calculation (weight_after : ℝ) (percent_lost : ℝ) 
  (h1 : weight_after = 640)
  (h2 : percent_lost = 20) : 
  weight_after / (1 - percent_lost / 100) = 800 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_calculation_l3103_310358


namespace NUMINAMATH_CALUDE_milk_container_problem_l3103_310313

-- Define the capacity of container A
def A : ℝ := 1232

-- Define the quantity of milk in container B after initial pouring
def B : ℝ := 0.375 * A

-- Define the quantity of milk in container C after initial pouring
def C : ℝ := 0.625 * A

-- Define the amount transferred from C to B
def transfer : ℝ := 154

-- Theorem statement
theorem milk_container_problem :
  -- All milk from A was poured into B and C
  (B + C = A) ∧
  -- B had 62.5% less milk than A's capacity
  (B = 0.375 * A) ∧
  -- After transfer, B and C have equal quantities
  (B + transfer = C - transfer) →
  -- The initial quantity of milk in A was 1232 liters
  A = 1232 := by
  sorry


end NUMINAMATH_CALUDE_milk_container_problem_l3103_310313


namespace NUMINAMATH_CALUDE_bd_squared_equals_nine_l3103_310323

theorem bd_squared_equals_nine 
  (h1 : a - b - c + d = 12) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_bd_squared_equals_nine_l3103_310323


namespace NUMINAMATH_CALUDE_cyclists_problem_l3103_310335

/-- Two cyclists problem -/
theorem cyclists_problem (x : ℝ) 
  (h1 : x > 0) -- Distance between A and B is positive
  (h2 : 70 + x + 90 = 3 * 70) -- Equation derived from the problem conditions
  : x = 120 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_problem_l3103_310335


namespace NUMINAMATH_CALUDE_shopping_money_l3103_310395

theorem shopping_money (initial_amount remaining_amount : ℝ) : 
  remaining_amount = initial_amount * (1 - 0.3) ∧ remaining_amount = 840 →
  initial_amount = 1200 := by
sorry

end NUMINAMATH_CALUDE_shopping_money_l3103_310395


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l3103_310392

theorem regular_polygon_with_150_degree_angles (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : (n : ℝ) * 150 = 180 * (n - 2)) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l3103_310392


namespace NUMINAMATH_CALUDE_pipe_b_rate_is_30_l3103_310314

/-- Represents the capacity of the tank in liters -/
def tank_capacity : ℕ := 900

/-- Represents the rate at which pipe A fills the tank in liters per minute -/
def pipe_a_rate : ℕ := 40

/-- Represents the rate at which pipe C drains the tank in liters per minute -/
def pipe_c_rate : ℕ := 20

/-- Represents the time taken to fill the tank in minutes -/
def fill_time : ℕ := 54

/-- Represents the duration of each pipe's operation in a cycle in minutes -/
def cycle_duration : ℕ := 3

/-- Theorem: Given the tank capacity, fill rates of pipes A and C, fill time, and cycle duration,
    the fill rate of pipe B is 30 liters per minute -/
theorem pipe_b_rate_is_30 :
  ∃ (pipe_b_rate : ℕ),
    pipe_b_rate = 30 ∧
    (fill_time / cycle_duration) * (pipe_a_rate + pipe_b_rate - pipe_c_rate) = tank_capacity :=
  sorry

end NUMINAMATH_CALUDE_pipe_b_rate_is_30_l3103_310314


namespace NUMINAMATH_CALUDE_max_value_of_circle_l3103_310351

theorem max_value_of_circle (x y : ℝ) :
  x^2 + y^2 + 4*x - 2*y - 4 = 0 →
  x^2 + y^2 ≤ 14 + 6 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_circle_l3103_310351


namespace NUMINAMATH_CALUDE_starting_team_combinations_count_l3103_310303

def team_size : ℕ := 18
def starting_team_size : ℕ := 8
def other_players_size : ℕ := starting_team_size - 2  -- 6 players excluding goalie and captain

def number_of_starting_team_combinations : ℕ :=
  team_size *  -- ways to choose goalie
  (team_size - 1) *  -- ways to choose captain (excluding goalie)
  (Nat.choose (team_size - 2) other_players_size)  -- ways to choose remaining 6 players

theorem starting_team_combinations_count :
  number_of_starting_team_combinations = 2455344 :=
by sorry

end NUMINAMATH_CALUDE_starting_team_combinations_count_l3103_310303


namespace NUMINAMATH_CALUDE_smallest_result_l3103_310337

def S : Finset Nat := {2, 4, 6, 8, 10, 12}

def process (a b c : Nat) : Nat := (a + b) * c

def valid_triple (a b c : Nat) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : Nat), valid_triple a b c ∧
    (∀ (x y z : Nat), valid_triple x y z →
      min (process a b c) (min (process a c b) (process b c a)) ≤
      min (process x y z) (min (process x z y) (process y z x))) ∧
    min (process a b c) (min (process a c b) (process b c a)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l3103_310337


namespace NUMINAMATH_CALUDE_car_speed_calculation_l3103_310388

/-- Calculates the speed of a car given distance and time -/
theorem car_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 360 ∧ time = 4.5 → speed = distance / time → speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l3103_310388


namespace NUMINAMATH_CALUDE_game_show_probability_l3103_310387

def num_questions : ℕ := 4
def num_options : ℕ := 4
def min_correct : ℕ := 3

def prob_correct_one : ℚ := 1 / num_options

def prob_all_correct : ℚ := prob_correct_one ^ num_questions

def prob_exactly_three_correct : ℚ := num_questions * (prob_correct_one ^ 3) * (1 - prob_correct_one)

theorem game_show_probability :
  prob_all_correct + prob_exactly_three_correct = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_game_show_probability_l3103_310387


namespace NUMINAMATH_CALUDE_problem_solution_l3103_310338

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 + Real.log x₀ = 0

-- Theorem to prove
theorem problem_solution : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3103_310338


namespace NUMINAMATH_CALUDE_somus_age_l3103_310302

theorem somus_age (s f : ℕ) (h1 : s = f / 3) (h2 : s - 9 = (f - 9) / 5) : s = 18 := by
  sorry

end NUMINAMATH_CALUDE_somus_age_l3103_310302


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3103_310334

theorem quadratic_function_properties (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 1 3 → a = 1 ∧ b = 1) ∧
  ((a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 1) → ∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) ∧
  (∀ x, |x| ≥ 2 → f x ≥ 0) ∧
  (∀ x ∈ Set.Ioc 2 3, f x ≤ 1) ∧
  (f 3 = 1) →
  (32 : ℝ) ≤ a^2 + b^2 ∧ a^2 + b^2 ≤ 74 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3103_310334


namespace NUMINAMATH_CALUDE_swimming_practice_months_l3103_310359

theorem swimming_practice_months (total_required : ℕ) (completed : ℕ) (monthly_practice : ℕ) : 
  total_required = 1500 →
  completed = 180 →
  monthly_practice = 220 →
  (total_required - completed) / monthly_practice = 6 := by
sorry

end NUMINAMATH_CALUDE_swimming_practice_months_l3103_310359


namespace NUMINAMATH_CALUDE_gcd_2021_2048_l3103_310306

theorem gcd_2021_2048 : Nat.gcd 2021 2048 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2021_2048_l3103_310306


namespace NUMINAMATH_CALUDE_no_solutions_exist_l3103_310353

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l3103_310353


namespace NUMINAMATH_CALUDE_amanda_candy_bars_l3103_310354

/-- Amanda's candy bar problem -/
theorem amanda_candy_bars :
  let initial_bars : ℕ := 7
  let first_gift : ℕ := 3
  let new_bars : ℕ := 30
  let second_gift : ℕ := 4 * first_gift
  let kept_bars : ℕ := (initial_bars - first_gift) + (new_bars - second_gift)
  kept_bars = 22 := by sorry

end NUMINAMATH_CALUDE_amanda_candy_bars_l3103_310354


namespace NUMINAMATH_CALUDE_average_weight_problem_l3103_310378

/-- Given the average weights of three people and two of them, along with the weight of one person,
    prove that the average weight of the other two is as stated. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 → 
  (a + b) / 2 = 40 → 
  b = 33 → 
  (b + c) / 2 = 44 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l3103_310378


namespace NUMINAMATH_CALUDE_locus_of_rectangle_vertex_l3103_310370

/-- Given a circle centered at the origin with radius r and a point M(a,b) inside the circle,
    prove that the locus of point T for all rectangles MKTP where K and P lie on the circle
    is a circle centered at the origin with radius √(2r² - (a² + b²)). -/
theorem locus_of_rectangle_vertex (r a b : ℝ) (hr : r > 0) (hab : a^2 + b^2 < r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = 2 * r^2 - (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_rectangle_vertex_l3103_310370


namespace NUMINAMATH_CALUDE_prob_one_success_in_three_trials_l3103_310343

/-- The probability of exactly one success in three independent trials with success probability 3/4 -/
theorem prob_one_success_in_three_trials : 
  let p : ℚ := 3/4  -- Probability of success in each trial
  let n : ℕ := 3    -- Number of trials
  let k : ℕ := 1    -- Number of successes we're interested in
  Nat.choose n k * p^k * (1-p)^(n-k) = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_success_in_three_trials_l3103_310343


namespace NUMINAMATH_CALUDE_max_k_value_l3103_310399

theorem max_k_value (k : ℝ) : 
  (k > 0 ∧ ∀ x > 0, k * Real.log (k * x) - Real.exp x ≤ 0) →
  k ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3103_310399


namespace NUMINAMATH_CALUDE_magnitude_2a_minus_b_l3103_310344

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-1, 1)

theorem magnitude_2a_minus_b :
  Real.sqrt ((2 * vector_a.1 - vector_b.1)^2 + (2 * vector_a.2 - vector_b.2)^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_2a_minus_b_l3103_310344


namespace NUMINAMATH_CALUDE_lily_cups_count_l3103_310376

/-- Represents Gina's cup painting rates and order details -/
structure PaintingOrder where
  rose_rate : ℕ  -- Roses painted per hour
  lily_rate : ℕ  -- Lilies painted per hour
  rose_order : ℕ  -- Number of rose cups ordered
  total_pay : ℕ  -- Total payment for the order in dollars
  hourly_rate : ℕ  -- Gina's hourly rate in dollars

/-- Calculates the number of lily cups in the order -/
def lily_cups (order : PaintingOrder) : ℕ :=
  let total_hours := order.total_pay / order.hourly_rate
  let rose_hours := order.rose_order / order.rose_rate
  let lily_hours := total_hours - rose_hours
  lily_hours * order.lily_rate

/-- Theorem stating that for the given order, the number of lily cups is 14 -/
theorem lily_cups_count (order : PaintingOrder) 
  (h1 : order.rose_rate = 6)
  (h2 : order.lily_rate = 7)
  (h3 : order.rose_order = 6)
  (h4 : order.total_pay = 90)
  (h5 : order.hourly_rate = 30) :
  lily_cups order = 14 := by
  sorry

#eval lily_cups { rose_rate := 6, lily_rate := 7, rose_order := 6, total_pay := 90, hourly_rate := 30 }

end NUMINAMATH_CALUDE_lily_cups_count_l3103_310376


namespace NUMINAMATH_CALUDE_sams_letters_l3103_310363

theorem sams_letters (letters_tuesday : ℕ) (average_per_day : ℕ) (total_days : ℕ) :
  letters_tuesday = 7 →
  average_per_day = 5 →
  total_days = 2 →
  (average_per_day * total_days - letters_tuesday : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sams_letters_l3103_310363


namespace NUMINAMATH_CALUDE_academic_year_school_days_l3103_310389

/-- The number of school days in the academic year -/
def school_days : ℕ := sorry

/-- The number of days Aliyah packs lunch -/
def aliyah_lunch_days : ℕ := sorry

/-- The number of days Becky packs lunch -/
def becky_lunch_days : ℕ := 45

theorem academic_year_school_days :
  (aliyah_lunch_days = 2 * becky_lunch_days) →
  (school_days = 2 * aliyah_lunch_days) →
  (school_days = 180) :=
by sorry

end NUMINAMATH_CALUDE_academic_year_school_days_l3103_310389


namespace NUMINAMATH_CALUDE_average_temperature_of_three_cities_l3103_310308

/-- The average temperature of three cities given specific temperature relationships --/
theorem average_temperature_of_three_cities 
  (temp_new_york : ℝ)
  (temp_diff_miami_new_york : ℝ)
  (temp_diff_san_diego_miami : ℝ)
  (h1 : temp_new_york = 80)
  (h2 : temp_diff_miami_new_york = 10)
  (h3 : temp_diff_san_diego_miami = 25) :
  (temp_new_york + (temp_new_york + temp_diff_miami_new_york) + 
   (temp_new_york + temp_diff_miami_new_york + temp_diff_san_diego_miami)) / 3 = 95 := by
  sorry

#check average_temperature_of_three_cities

end NUMINAMATH_CALUDE_average_temperature_of_three_cities_l3103_310308


namespace NUMINAMATH_CALUDE_set_operation_result_l3103_310379

def A : Set ℕ := {0, 1, 2, 4, 5, 7, 8}
def B : Set ℕ := {1, 3, 6, 7, 9}
def C : Set ℕ := {3, 4, 7, 8}

theorem set_operation_result : (A ∩ B) ∪ C = {1, 3, 4, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l3103_310379


namespace NUMINAMATH_CALUDE_student_D_most_stable_l3103_310336

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define a function to get the variance for each student
def variance : Student → ℝ
| Student.A => 2.1
| Student.B => 3.5
| Student.C => 9.0
| Student.D => 0.7

-- Define a predicate for most stable performance
def most_stable (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

-- Theorem: Student D has the most stable performance
theorem student_D_most_stable : most_stable Student.D := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_student_D_most_stable_l3103_310336


namespace NUMINAMATH_CALUDE_equation_solution_l3103_310382

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (8 * x)^12 = (16 * x)^6 ↔ x = 1/4 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3103_310382


namespace NUMINAMATH_CALUDE_exists_coprime_sequence_l3103_310347

theorem exists_coprime_sequence : ∃ (a : ℕ → ℕ), 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ i j p q r, i ≠ j ∧ i ≠ p ∧ i ≠ q ∧ i ≠ r ∧ j ≠ p ∧ j ≠ q ∧ j ≠ r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r → 
    Nat.gcd (a i + a j) (a p + a q + a r) = 1) :=
by sorry

end NUMINAMATH_CALUDE_exists_coprime_sequence_l3103_310347


namespace NUMINAMATH_CALUDE_mighty_l_league_teams_l3103_310393

/-- The number of teams in the league -/
def n : ℕ := 8

/-- The total number of games played -/
def total_games : ℕ := 28

/-- Formula for the number of games in a round-robin tournament -/
def games (x : ℕ) : ℕ := x * (x - 1) / 2

theorem mighty_l_league_teams :
  (n ≥ 2) ∧ (games n = total_games) := by sorry

end NUMINAMATH_CALUDE_mighty_l_league_teams_l3103_310393


namespace NUMINAMATH_CALUDE_line_perpendicular_implies_plane_perpendicular_l3103_310375

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (in_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_implies_plane_perpendicular
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : in_plane m α)
  (m_perp_β : perpendicular_line_plane m β) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_implies_plane_perpendicular_l3103_310375


namespace NUMINAMATH_CALUDE_min_value_theorem_l3103_310301

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 2 / a₀ + 3 / b₀ = 5 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3103_310301


namespace NUMINAMATH_CALUDE_prize_distribution_l3103_310396

theorem prize_distribution (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  n^k = 8000 := by
  sorry

end NUMINAMATH_CALUDE_prize_distribution_l3103_310396


namespace NUMINAMATH_CALUDE_thomas_total_bill_l3103_310367

-- Define the shipping rates
def flat_rate : ℝ := 5.00
def clothes_rate : ℝ := 0.20
def accessories_rate : ℝ := 0.10
def price_threshold : ℝ := 50.00

-- Define the prices of items
def shirt_price : ℝ := 12.00
def socks_price : ℝ := 5.00
def shorts_price : ℝ := 15.00
def swim_trunks_price : ℝ := 14.00
def hat_price : ℝ := 6.00
def sunglasses_price : ℝ := 30.00

-- Define the quantities of items
def shirt_quantity : ℕ := 3
def shorts_quantity : ℕ := 2

-- Calculate the total cost of clothes and accessories
def clothes_cost : ℝ := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity + swim_trunks_price
def accessories_cost : ℝ := hat_price + sunglasses_price

-- Calculate the shipping costs
def clothes_shipping : ℝ := clothes_rate * clothes_cost
def accessories_shipping : ℝ := accessories_rate * accessories_cost

-- Calculate the total bill
def total_bill : ℝ := clothes_cost + accessories_cost + clothes_shipping + accessories_shipping

-- Theorem to prove
theorem thomas_total_bill : total_bill = 141.60 := by sorry

end NUMINAMATH_CALUDE_thomas_total_bill_l3103_310367


namespace NUMINAMATH_CALUDE_price_difference_enhanced_basic_computer_l3103_310364

/-- Prove the price difference between enhanced and basic computers --/
theorem price_difference_enhanced_basic_computer :
  ∀ (basic_price enhanced_price printer_price : ℕ),
  basic_price = 1500 →
  basic_price + printer_price = 2500 →
  printer_price = (enhanced_price + printer_price) / 3 →
  enhanced_price - basic_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_enhanced_basic_computer_l3103_310364


namespace NUMINAMATH_CALUDE_computer_distribution_l3103_310304

def distribute_computers (n : ℕ) (k : ℕ) (min : ℕ) : ℕ :=
  -- The number of ways to distribute n identical items among k recipients,
  -- with each recipient receiving at least min items
  sorry

theorem computer_distribution :
  distribute_computers 9 3 2 = 10 := by sorry

end NUMINAMATH_CALUDE_computer_distribution_l3103_310304


namespace NUMINAMATH_CALUDE_attendees_count_l3103_310368

/-- The number of people attending the family reunion --/
def attendees : ℕ := sorry

/-- The number of cans in each box of soda --/
def cans_per_box : ℕ := 10

/-- The cost of each box of soda in dollars --/
def cost_per_box : ℕ := 2

/-- The number of cans each person consumes --/
def cans_per_person : ℕ := 2

/-- The number of family members paying for the soda --/
def paying_family_members : ℕ := 6

/-- The amount each family member pays in dollars --/
def payment_per_member : ℕ := 4

/-- Theorem stating that the number of attendees is 60 --/
theorem attendees_count : attendees = 60 := by sorry

end NUMINAMATH_CALUDE_attendees_count_l3103_310368


namespace NUMINAMATH_CALUDE_triangle_area_is_four_l3103_310385

/-- The area of the triangle formed by the intersection of lines y = x + 2, y = -x + 8, and y = 3 -/
def triangleArea : ℝ := 4

/-- The first line equation: y = x + 2 -/
def line1 (x y : ℝ) : Prop := y = x + 2

/-- The second line equation: y = -x + 8 -/
def line2 (x y : ℝ) : Prop := y = -x + 8

/-- The third line equation: y = 3 -/
def line3 (x y : ℝ) : Prop := y = 3

theorem triangle_area_is_four :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line1 x₁ y₁ ∧ line3 x₁ y₁ ∧
    line2 x₂ y₂ ∧ line3 x₂ y₂ ∧
    line1 x₃ y₃ ∧ line2 x₃ y₃ ∧
    triangleArea = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_four_l3103_310385


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l3103_310342

/-- Calculates the surface area of a cuboid given its length, breadth, and height. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + length * height + breadth * height)

/-- Theorem stating that the surface area of a cuboid with length 12, breadth 14, and height 7 is 700. -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 12 14 7 = 700 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l3103_310342


namespace NUMINAMATH_CALUDE_zero_exponent_l3103_310325

theorem zero_exponent (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l3103_310325


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3103_310398

theorem quadratic_roots_product (b c : ℝ) : 
  (1 : ℝ) ∈ {x : ℝ | x^2 + b*x + c = 0} ∧ 
  (-2 : ℝ) ∈ {x : ℝ | x^2 + b*x + c = 0} → 
  b * c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3103_310398


namespace NUMINAMATH_CALUDE_kekai_mms_packs_l3103_310312

/-- The number of sundaes made on Monday -/
def monday_sundaes : ℕ := 40

/-- The number of m&ms per sundae on Monday -/
def monday_mms_per_sundae : ℕ := 6

/-- The number of sundaes made on Tuesday -/
def tuesday_sundaes : ℕ := 20

/-- The number of m&ms per sundae on Tuesday -/
def tuesday_mms_per_sundae : ℕ := 10

/-- The number of m&ms in each pack -/
def mms_per_pack : ℕ := 40

/-- The total number of m&m packs used -/
def total_packs_used : ℕ := 11

theorem kekai_mms_packs :
  (monday_sundaes * monday_mms_per_sundae + tuesday_sundaes * tuesday_mms_per_sundae) / mms_per_pack = total_packs_used :=
by sorry

end NUMINAMATH_CALUDE_kekai_mms_packs_l3103_310312


namespace NUMINAMATH_CALUDE_mika_stickers_decoration_l3103_310300

/-- The number of stickers Mika used to decorate the greeting card -/
def stickers_used_for_decoration (initial : ℕ) (bought : ℕ) (received : ℕ) (given_away : ℕ) (left : ℕ) : ℕ :=
  initial + bought + received - given_away - left

/-- Theorem stating that Mika used 58 stickers to decorate the greeting card -/
theorem mika_stickers_decoration :
  stickers_used_for_decoration 20 26 20 6 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_decoration_l3103_310300


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l3103_310361

theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (3 + Complex.I)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l3103_310361


namespace NUMINAMATH_CALUDE_min_value_fourth_power_l3103_310372

theorem min_value_fourth_power (x : ℝ) : 
  x ∈ Set.Icc 0 1 → (x^4 + (1-x)^4 : ℝ) ≥ 1/8 ∧ ∃ y ∈ Set.Icc 0 1, y^4 + (1-y)^4 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fourth_power_l3103_310372


namespace NUMINAMATH_CALUDE_sqrt_7200_minus_61_cube_l3103_310321

theorem sqrt_7200_minus_61_cube (a b : ℕ+) :
  (Real.sqrt 7200 - 61 : ℝ) = (Real.sqrt a.val - b.val)^3 →
  a.val + b.val = 21 := by
sorry

end NUMINAMATH_CALUDE_sqrt_7200_minus_61_cube_l3103_310321


namespace NUMINAMATH_CALUDE_five_in_range_of_f_l3103_310360

/-- The function f(x) = x^2 + bx - 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 3

/-- Theorem stating that 5 is always in the range of f(x) for all real b -/
theorem five_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_in_range_of_f_l3103_310360


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l3103_310369

theorem lcm_gcd_relation (m n : ℕ) (h1 : m > n) (h2 : m > 0) (h3 : n > 0) 
  (h4 : Nat.lcm m n = 30 * Nat.gcd m n) 
  (h5 : (m - n) ∣ Nat.lcm m n) : 
  (m + n) / Nat.gcd m n = 11 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l3103_310369


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l3103_310315

theorem square_minus_product_equals_one (x : ℝ) : (x + 2)^2 - (x + 1) * (x + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l3103_310315


namespace NUMINAMATH_CALUDE_rotated_line_x_intercept_l3103_310380

/-- The x-coordinate of the x-intercept of a rotated line -/
theorem rotated_line_x_intercept 
  (m : Real → Real → Prop) -- Original line
  (θ : Real) -- Rotation angle
  (p : Real × Real) -- Point of rotation
  (n : Real → Real → Prop) -- Rotated line
  (h1 : ∀ x y, m x y ↔ 4 * x - 3 * y + 20 = 0) -- Equation of line m
  (h2 : θ = π / 3) -- 60° in radians
  (h3 : p = (10, 10)) -- Point of rotation
  (h4 : ∀ x y, n x y ↔ 
    y - p.2 = ((24 + 25 * Real.sqrt 3) / (-39)) * (x - p.1)) -- Equation of line n
  (C : Real) -- Constant C
  (h5 : C = 10 - (240 + 250 * Real.sqrt 3) / (-39)) -- Definition of C
  : ∃ x_intercept : Real, 
    x_intercept = -39 * C / (24 + 25 * Real.sqrt 3) ∧ 
    n x_intercept 0 := by sorry

end NUMINAMATH_CALUDE_rotated_line_x_intercept_l3103_310380


namespace NUMINAMATH_CALUDE_parabola_C_passes_through_origin_l3103_310311

-- Define the parabolas
def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2*x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

-- Define what it means for a parabola to pass through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- Theorem stating that parabola C passes through the origin while others do not
theorem parabola_C_passes_through_origin :
  passes_through_origin parabola_C ∧
  ¬passes_through_origin parabola_A ∧
  ¬passes_through_origin parabola_B ∧
  ¬passes_through_origin parabola_D :=
by sorry

end NUMINAMATH_CALUDE_parabola_C_passes_through_origin_l3103_310311


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3103_310397

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 → 1/x + 1/y ≥ 3 + 2*Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ 1/x + 1/y = 3 + 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3103_310397


namespace NUMINAMATH_CALUDE_third_year_interest_l3103_310357

/-- Calculates the compound interest for a given principal, rate, and time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Represents the loan scenario with given parameters -/
structure LoanScenario where
  initialLoan : ℝ
  rate1 : ℝ
  rate2 : ℝ
  rate3 : ℝ

/-- Theorem stating the interest paid in the third year of the loan -/
theorem third_year_interest (loan : LoanScenario) 
  (h1 : loan.initialLoan = 9000)
  (h2 : loan.rate1 = 0.09)
  (h3 : loan.rate2 = 0.105)
  (h4 : loan.rate3 = 0.085) :
  let firstYearTotal := loan.initialLoan * (1 + loan.rate1)
  let secondYearTotal := firstYearTotal * (1 + loan.rate2)
  compoundInterest secondYearTotal loan.rate3 1 = 922.18 := by
  sorry

end NUMINAMATH_CALUDE_third_year_interest_l3103_310357


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3103_310374

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℤ, x > 2*m ∧ x ≥ m - 3 ∧ (∀ y : ℤ, y > 2*m ∧ y ≥ m - 3 → y ≥ x) ∧ x = 1) 
  ↔ 0 ≤ m ∧ m < 1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3103_310374


namespace NUMINAMATH_CALUDE_units_digit_p_plus_4_l3103_310310

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Definition of a positive even integer with a positive units digit -/
def isPositiveEvenWithPositiveUnitsDigit (p : ℕ) : Prop :=
  p > 0 ∧ p % 2 = 0 ∧ unitsDigit p > 0

/-- The main theorem -/
theorem units_digit_p_plus_4 (p : ℕ) 
  (h1 : isPositiveEvenWithPositiveUnitsDigit p)
  (h2 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 4) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_p_plus_4_l3103_310310


namespace NUMINAMATH_CALUDE_luna_has_seventeen_badges_l3103_310383

/-- The number of spelling badges Luna has, given the total number of badges and the number of badges Hermione and Celestia have. -/
def luna_badges (total : ℕ) (hermione : ℕ) (celestia : ℕ) : ℕ :=
  total - (hermione + celestia)

/-- Theorem stating that Luna has 17 spelling badges given the conditions in the problem. -/
theorem luna_has_seventeen_badges :
  luna_badges 83 14 52 = 17 := by
  sorry

end NUMINAMATH_CALUDE_luna_has_seventeen_badges_l3103_310383


namespace NUMINAMATH_CALUDE_right_triangle_cos_c_l3103_310355

theorem right_triangle_cos_c (A B C : ℝ) (h1 : A + B + C = π) (h2 : A = π/2) (h3 : Real.sin B = 3/5) : 
  Real.cos C = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_c_l3103_310355


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l3103_310326

theorem quadratic_equation_general_form :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 3 ∧
  ∀ x, (x - 1)^2 = 3*x - 2 ↔ a*x^2 + b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l3103_310326


namespace NUMINAMATH_CALUDE_twelfth_odd_multiple_of_5_l3103_310318

/-- The nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Predicate for a number being odd and a multiple of 5 -/
def isOddMultipleOf5 (x : ℕ) : Prop :=
  x % 2 = 1 ∧ x % 5 = 0

theorem twelfth_odd_multiple_of_5 :
  nthOddMultipleOf5 12 = 115 ∧
  isOddMultipleOf5 (nthOddMultipleOf5 12) :=
sorry

end NUMINAMATH_CALUDE_twelfth_odd_multiple_of_5_l3103_310318


namespace NUMINAMATH_CALUDE_data_statistics_l3103_310349

def data : List ℝ := [6, 8, 8, 9, 8, 9, 8, 8, 7, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_statistics :
  mode data = 8 ∧
  median data = 8 ∧
  mean data = 8 ∧
  variance data ≠ 8 := by sorry

end NUMINAMATH_CALUDE_data_statistics_l3103_310349


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3103_310350

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ (∀ x : ℝ, x^2 + 2*x - 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3103_310350


namespace NUMINAMATH_CALUDE_total_tagged_numbers_l3103_310339

def card_sum (w x y z : ℕ) : ℕ := w + x + y + z

theorem total_tagged_numbers : ∃ (w x y z : ℕ),
  w = 200 ∧
  x = w / 2 ∧
  y = x + w ∧
  z = 400 ∧
  card_sum w x y z = 1000 := by
sorry

end NUMINAMATH_CALUDE_total_tagged_numbers_l3103_310339


namespace NUMINAMATH_CALUDE_purely_imaginary_sufficient_not_necessary_l3103_310366

theorem purely_imaginary_sufficient_not_necessary (m : ℝ) :
  (∃ (z : ℂ), z = (m^2 - 1 : ℂ) + (m - 1 : ℂ) * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0) →
  (m = 1 ∨ m = -1) ∧
  ¬(∀ m : ℝ, (m = 1 ∨ m = -1) → 
    (∃ (z : ℂ), z = (m^2 - 1 : ℂ) + (m - 1 : ℂ) * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_sufficient_not_necessary_l3103_310366


namespace NUMINAMATH_CALUDE_lunch_ratio_is_one_half_l3103_310322

/-- The number of school days in the academic year -/
def total_school_days : ℕ := 180

/-- The number of days Becky packs her lunch -/
def becky_lunch_days : ℕ := 45

/-- The number of days Aliyah packs her lunch -/
def aliyah_lunch_days : ℕ := 2 * becky_lunch_days

/-- The ratio of Aliyah's lunch-packing days to total school days -/
def lunch_ratio : ℚ := aliyah_lunch_days / total_school_days

theorem lunch_ratio_is_one_half : lunch_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lunch_ratio_is_one_half_l3103_310322


namespace NUMINAMATH_CALUDE_john_car_profit_l3103_310309

/-- Calculates the profit from fixing and racing a car given the following parameters:
    * original_cost: The original cost to fix the car
    * discount_percentage: The discount percentage on the repair cost
    * prize_money: The total prize money won
    * kept_percentage: The percentage of prize money kept by the racer
-/
def calculate_car_profit (original_cost discount_percentage prize_money kept_percentage : ℚ) : ℚ :=
  let discounted_cost := original_cost * (1 - discount_percentage / 100)
  let kept_prize := prize_money * (kept_percentage / 100)
  kept_prize - discounted_cost

/-- Theorem stating that given the specific conditions of John's car repair and race,
    his profit is $47,000 -/
theorem john_car_profit :
  calculate_car_profit 20000 20 70000 90 = 47000 := by
  sorry

end NUMINAMATH_CALUDE_john_car_profit_l3103_310309


namespace NUMINAMATH_CALUDE_power_equality_implies_y_equals_two_l3103_310391

theorem power_equality_implies_y_equals_two : 
  ∀ y : ℝ, (3 : ℝ)^6 = 27^y → y = 2 := by
sorry

end NUMINAMATH_CALUDE_power_equality_implies_y_equals_two_l3103_310391


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l3103_310330

/-- Given a train of length 2400 meters that takes 60 seconds to pass a point,
    calculate the time required for the same train to pass a platform of length 800 meters. -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_pass_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 2400)
  (h2 : time_to_pass_point = 60)
  (h3 : platform_length = 800) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 80 := by
  sorry

#check train_platform_passing_time

end NUMINAMATH_CALUDE_train_platform_passing_time_l3103_310330


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l3103_310384

theorem least_possible_smallest_integer
  (a b c d : ℤ) -- Four different integers
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) -- Integers are different
  (h_avg : (a + b + c + d) / 4 = 68) -- Average is 68
  (h_max : d = 90) -- Largest integer is 90
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) -- Order of integers
  : a ≥ 5 := -- Least possible value of smallest integer is 5
by sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l3103_310384


namespace NUMINAMATH_CALUDE_only_C_is_pythagorean_triple_l3103_310307

-- Define the sets of numbers
def set_A : Vector ℕ 3 := ⟨[7, 8, 9], by rfl⟩
def set_B : Vector ℕ 3 := ⟨[5, 6, 7], by rfl⟩
def set_C : Vector ℕ 3 := ⟨[5, 12, 13], by rfl⟩
def set_D : Vector ℕ 3 := ⟨[21, 25, 28], by rfl⟩

-- Define a function to check if a set of three numbers is a Pythagorean triple
def is_pythagorean_triple (v : Vector ℕ 3) : Prop :=
  v[0] * v[0] + v[1] * v[1] = v[2] * v[2]

-- Theorem statement
theorem only_C_is_pythagorean_triple :
  ¬(is_pythagorean_triple set_A) ∧
  ¬(is_pythagorean_triple set_B) ∧
  (is_pythagorean_triple set_C) ∧
  ¬(is_pythagorean_triple set_D) :=
by sorry

end NUMINAMATH_CALUDE_only_C_is_pythagorean_triple_l3103_310307


namespace NUMINAMATH_CALUDE_linear_function_unique_solution_l3103_310394

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_unique_solution (f : ℝ → ℝ) 
  (h_linear : LinearFunction f) (h1 : f 2 = 1) (h2 : f (-1) = -5) :
  ∀ x, f x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_unique_solution_l3103_310394


namespace NUMINAMATH_CALUDE_largest_number_with_sum_18_l3103_310319

def is_valid_number (n : ℕ) : Prop :=
  (n.digits 10).sum = 18 ∧ (n.digits 10).Nodup

theorem largest_number_with_sum_18 :
  ∀ n : ℕ, is_valid_number n → n ≤ 843210 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_18_l3103_310319


namespace NUMINAMATH_CALUDE_john_mission_duration_l3103_310362

theorem john_mission_duration :
  let initial_duration : ℝ := 5
  let first_mission_duration : ℝ := initial_duration * (1 + 0.6)
  let second_mission_duration : ℝ := first_mission_duration * 0.5
  let third_mission_duration : ℝ := min (2 * second_mission_duration) (first_mission_duration * 0.8)
  let fourth_mission_duration : ℝ := 3 + (third_mission_duration * 0.5)
  first_mission_duration + second_mission_duration + third_mission_duration + fourth_mission_duration = 24.6 :=
by sorry

end NUMINAMATH_CALUDE_john_mission_duration_l3103_310362


namespace NUMINAMATH_CALUDE_sum_of_nine_terms_l3103_310329

/-- An arithmetic sequence with sum Sₙ of first n terms, where a₄ = 9 and a₆ = 11 -/
structure ArithSeq where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)
  a4_eq_9 : a 4 = 9
  a6_eq_11 : a 6 = 11

/-- The sum of the first 9 terms of the arithmetic sequence is 90 -/
theorem sum_of_nine_terms (seq : ArithSeq) : seq.S 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nine_terms_l3103_310329


namespace NUMINAMATH_CALUDE_equation_transformation_l3103_310328

theorem equation_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 4 * b) :
  a / 4 = b / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3103_310328
