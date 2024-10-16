import Mathlib

namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l489_48938

/-- Represents a parabola opening to the right with equation y² = 2px -/
structure Parabola where
  p : ℝ
  opens_right : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_point_to_directrix_distance
  (C : Parabola) (A : Point)
  (h1 : A.x = 1)
  (h2 : A.y = Real.sqrt 5)
  (h3 : A.y ^ 2 = 2 * C.p * A.x) :
  A.x + C.p / 2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l489_48938


namespace NUMINAMATH_CALUDE_mascot_problem_solution_l489_48986

/-- Represents the sales data for a week -/
structure WeekSales where
  bing : ℕ
  shuey : ℕ
  revenue : ℕ

/-- Solves for mascot prices and maximum purchase given sales data and budget -/
def solve_mascot_problem (week1 week2 : WeekSales) (total_budget total_mascots : ℕ) :
  (ℕ × ℕ × ℕ) :=
sorry

/-- Theorem stating the correctness of the solution -/
theorem mascot_problem_solution :
  let week1 : WeekSales := ⟨3, 5, 1800⟩
  let week2 : WeekSales := ⟨4, 10, 3100⟩
  let (bing_price, shuey_price, max_bing) := solve_mascot_problem week1 week2 6700 30
  bing_price = 250 ∧ shuey_price = 210 ∧ max_bing = 10 :=
sorry

end NUMINAMATH_CALUDE_mascot_problem_solution_l489_48986


namespace NUMINAMATH_CALUDE_volume_ratio_in_cycle_l489_48957

/-- Represents the state of an ideal gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents a cycle of an ideal gas -/
structure GasCycle where
  state1 : GasState
  state2 : GasState
  state3 : GasState

/-- Conditions for the gas cycle -/
def cycleConditions (cycle : GasCycle) : Prop :=
  -- 1-2 is isobaric and volume increases by 4 times
  cycle.state1.pressure = cycle.state2.pressure ∧
  cycle.state2.volume = 4 * cycle.state1.volume ∧
  -- 2-3 is isothermal
  cycle.state2.temperature = cycle.state3.temperature ∧
  cycle.state3.pressure > cycle.state2.pressure ∧
  -- 3-1 follows T = γV²
  ∃ γ : ℝ, cycle.state3.temperature = γ * cycle.state1.volume^2

theorem volume_ratio_in_cycle (cycle : GasCycle) 
  (h : cycleConditions cycle) : 
  cycle.state3.volume = 2 * cycle.state1.volume :=
sorry

end NUMINAMATH_CALUDE_volume_ratio_in_cycle_l489_48957


namespace NUMINAMATH_CALUDE_ferry_speed_proof_l489_48922

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 8

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 4

/-- The time taken by ferry P in hours -/
def time_P : ℝ := 2

/-- The time taken by ferry Q in hours -/
def time_Q : ℝ := time_P + 2

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := 3 * distance_P

theorem ferry_speed_proof :
  speed_P = 8 ∧
  speed_Q = speed_P + 4 ∧
  time_Q = time_P + 2 ∧
  distance_Q = 3 * distance_P ∧
  distance_Q = speed_Q * time_Q ∧
  distance_P = speed_P * time_P :=
by sorry

end NUMINAMATH_CALUDE_ferry_speed_proof_l489_48922


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l489_48912

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 235) 
  (h2 : train_speed_kmh = 64) 
  (h3 : crossing_time = 45) : 
  ∃ (bridge_length : ℝ), 
    (bridge_length ≥ 565) ∧ 
    (bridge_length < 566) :=
by
  sorry


end NUMINAMATH_CALUDE_bridge_length_calculation_l489_48912


namespace NUMINAMATH_CALUDE_new_average_weight_l489_48944

def initial_average_weight : ℝ := 48
def initial_members : ℕ := 23
def new_person1_weight : ℝ := 78
def new_person2_weight : ℝ := 93

theorem new_average_weight :
  let total_initial_weight := initial_average_weight * initial_members
  let total_new_weight := new_person1_weight + new_person2_weight
  let total_weight := total_initial_weight + total_new_weight
  let new_members := initial_members + 2
  total_weight / new_members = 51 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l489_48944


namespace NUMINAMATH_CALUDE_probability_three_black_cards_l489_48931

/-- The probability of drawing three black cards consecutively from a standard deck --/
theorem probability_three_black_cards (total_cards : ℕ) (black_cards : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : black_cards = 26) : 
  (black_cards * (black_cards - 1) * (black_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 4 / 17 := by
sorry

end NUMINAMATH_CALUDE_probability_three_black_cards_l489_48931


namespace NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l489_48930

theorem smallest_m_satisfying_conditions : ∃ m : ℕ,
  (100 ≤ m ∧ m ≤ 999) ∧
  (m + 6) % 9 = 0 ∧
  (m - 9) % 6 = 0 ∧
  (∀ n : ℕ, (100 ≤ n ∧ n < m ∧ (n + 6) % 9 = 0 ∧ (n - 9) % 6 = 0) → False) ∧
  m = 111 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_satisfying_conditions_l489_48930


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l489_48906

theorem absolute_value_of_z (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + 1/z = r) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l489_48906


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l489_48900

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 + (3 - x)^3
  ∃ (x₁ x₂ : ℝ), (x₁ = 1.5 + Real.sqrt 5 / 2) ∧ 
                 (x₂ = 1.5 - Real.sqrt 5 / 2) ∧ 
                 (f x₁ = 18) ∧ 
                 (f x₂ = 18) ∧ 
                 (∀ x : ℝ, f x = 18 → (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l489_48900


namespace NUMINAMATH_CALUDE_count_divisible_by_11_eq_36_l489_48939

/-- The number obtained by concatenating integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- The count of k values in [1, 200] for which a_k is divisible by 11 -/
def count_divisible_by_11 : ℕ := sorry

/-- Theorem stating that the count of k values in [1, 200] for which a_k is divisible by 11 is 36 -/
theorem count_divisible_by_11_eq_36 : count_divisible_by_11 = 36 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_eq_36_l489_48939


namespace NUMINAMATH_CALUDE_divisibility_by_nine_highest_power_of_three_in_M_l489_48997

/-- The integer formed by concatenating 2-digit integers from 15 to 95 -/
def M : ℕ := sorry

/-- The sum of digits of M -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9 -/
theorem divisibility_by_nine (n : ℕ) : n % 9 = 0 ↔ sum_of_digits n % 9 = 0 := sorry

/-- The highest power of 3 that divides M is 3^2 -/
theorem highest_power_of_three_in_M : 
  ∃ (k : ℕ), M % (3^3) ≠ 0 ∧ M % (3^2) = 0 := sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_highest_power_of_three_in_M_l489_48997


namespace NUMINAMATH_CALUDE_abs_rational_nonnegative_l489_48943

theorem abs_rational_nonnegative (a : ℚ) : 0 ≤ |a| := by
  sorry

end NUMINAMATH_CALUDE_abs_rational_nonnegative_l489_48943


namespace NUMINAMATH_CALUDE_inverse_quadratic_sum_l489_48910

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The inverse function f^(-1)(x) = cx^2 + bx + a -/
def f_inv (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

/-- Theorem: If f and f_inv are inverse functions, then a + c = -1 -/
theorem inverse_quadratic_sum (a b c : ℝ) :
  (∀ x, f a b c (f_inv a b c x) = x) → a + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_quadratic_sum_l489_48910


namespace NUMINAMATH_CALUDE_purple_or_orange_probability_l489_48978

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  green : ℕ
  purple : ℕ
  orange : ℕ
  sum_faces : green + purple + orange = sides

/-- The probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The main theorem -/
theorem purple_or_orange_probability (d : ColoredDie)
    (h : d.sides = 10 ∧ d.green = 5 ∧ d.purple = 3 ∧ d.orange = 2) :
    probability (d.purple + d.orange) d.sides = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_purple_or_orange_probability_l489_48978


namespace NUMINAMATH_CALUDE_fraction_problem_l489_48968

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * (1/3) * F * N = 15)
  (h2 : 0.40 * N = 180) : 
  F = 2/5 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l489_48968


namespace NUMINAMATH_CALUDE_base9_726_to_base3_l489_48967

/-- Converts a digit from base 9 to two digits in base 3 -/
def base9ToBase3Digit (d : Nat) : Nat × Nat :=
  sorry

/-- Converts a number from base 9 to base 3 -/
def base9ToBase3 (n : Nat) : Nat :=
  sorry

theorem base9_726_to_base3 :
  base9ToBase3 726 = 210220 :=
sorry

end NUMINAMATH_CALUDE_base9_726_to_base3_l489_48967


namespace NUMINAMATH_CALUDE_power_product_equals_75600_l489_48961

theorem power_product_equals_75600 : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_75600_l489_48961


namespace NUMINAMATH_CALUDE_square_diagonal_l489_48940

theorem square_diagonal (A : ℝ) (s : ℝ) (d : ℝ) (h1 : A = 9/16) (h2 : A = s^2) (h3 : d = s * Real.sqrt 2) :
  d = 3/4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l489_48940


namespace NUMINAMATH_CALUDE_small_boxes_count_l489_48908

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 300) 
  (h2 : chocolates_per_box = 20) : 
  total_chocolates / chocolates_per_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_count_l489_48908


namespace NUMINAMATH_CALUDE_farmer_pumpkin_seeds_l489_48907

/-- Represents the farmer's vegetable planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  pumpkin_per_row : ℕ
  plant_beds : ℕ
  rows_per_bed : ℕ

/-- Calculates the number of pumpkin seeds in the given planting scenario -/
def calculate_pumpkin_seeds (f : FarmerPlanting) : ℕ :=
  let total_rows := f.plant_beds * f.rows_per_bed
  let bean_rows := f.bean_seedlings / f.bean_per_row
  let radish_rows := f.radishes / f.radishes_per_row
  let pumpkin_rows := total_rows - bean_rows - radish_rows
  pumpkin_rows * f.pumpkin_per_row

/-- Theorem stating that the farmer had 84 pumpkin seeds -/
theorem farmer_pumpkin_seeds :
  let f : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    radishes := 48,
    radishes_per_row := 6,
    pumpkin_per_row := 7,
    plant_beds := 14,
    rows_per_bed := 2
  }
  calculate_pumpkin_seeds f = 84 := by
  sorry

end NUMINAMATH_CALUDE_farmer_pumpkin_seeds_l489_48907


namespace NUMINAMATH_CALUDE_percent_problem_l489_48946

theorem percent_problem (x : ℝ) : 2 = (4 / 100) * x → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l489_48946


namespace NUMINAMATH_CALUDE_rider_pedestrian_problem_l489_48964

/-- A problem about a rider and a pedestrian traveling between two points. -/
theorem rider_pedestrian_problem
  (total_time : ℝ) -- Total time for the rider's journey
  (time_difference : ℝ) -- Time difference between rider and pedestrian arriving at B
  (meeting_distance : ℝ) -- Distance from B where rider meets pedestrian on return
  (h_total_time : total_time = 100 / 60) -- Total time is 1 hour 40 minutes (100 minutes)
  (h_time_difference : time_difference = 50 / 60) -- Rider arrives 50 minutes earlier
  (h_meeting_distance : meeting_distance = 2) -- They meet 2 km from B
  : ∃ (distance speed_rider speed_pedestrian : ℝ),
    distance = 6 ∧ 
    speed_rider = 7.2 ∧ 
    speed_pedestrian = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_rider_pedestrian_problem_l489_48964


namespace NUMINAMATH_CALUDE_exam_score_l489_48974

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 60 →
  correct_answers = 38 →
  marks_per_correct = 4 →
  marks_lost_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 130 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_l489_48974


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l489_48994

def total_students : ℕ := 300
def cat_owners : ℕ := 45

theorem percentage_of_cat_owners : 
  (cat_owners : ℚ) / (total_students : ℚ) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l489_48994


namespace NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l489_48984

theorem sixth_root_of_24414062515625 : 
  (24414062515625 : ℝ) ^ (1/6 : ℝ) = 51 := by sorry

end NUMINAMATH_CALUDE_sixth_root_of_24414062515625_l489_48984


namespace NUMINAMATH_CALUDE_largest_ball_radius_is_four_l489_48952

/-- Represents a torus in 3D space -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  circle_center : ℝ × ℝ × ℝ
  circle_radius : ℝ

/-- Represents a spherical ball in 3D space -/
structure SphericalBall where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The largest spherical ball that can be placed on top of a torus -/
def largest_ball_on_torus (t : Torus) : SphericalBall :=
  { center := (0, 0, 4),
    radius := 4 }

/-- Theorem stating that the largest ball on the torus has radius 4 -/
theorem largest_ball_radius_is_four (t : Torus) 
  (h1 : t.inner_radius = 3)
  (h2 : t.outer_radius = 5)
  (h3 : t.circle_center = (4, 0, 1))
  (h4 : t.circle_radius = 1) :
  (largest_ball_on_torus t).radius = 4 := by
  sorry

#check largest_ball_radius_is_four

end NUMINAMATH_CALUDE_largest_ball_radius_is_four_l489_48952


namespace NUMINAMATH_CALUDE_derivative_x_over_one_minus_cos_l489_48918

/-- The derivative of x / (1 - cos x) is (1 - cos x - x * sin x) / (1 - cos x)^2 -/
theorem derivative_x_over_one_minus_cos (x : ℝ) :
  deriv (fun x => x / (1 - Real.cos x)) x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
sorry

end NUMINAMATH_CALUDE_derivative_x_over_one_minus_cos_l489_48918


namespace NUMINAMATH_CALUDE_london_to_edinburgh_distance_l489_48969

theorem london_to_edinburgh_distance :
  ∀ D : ℝ,
  (∃ x : ℝ, x = 200 ∧ x + 3.5 = D / 2) →
  D = 393 :=
by
  sorry

end NUMINAMATH_CALUDE_london_to_edinburgh_distance_l489_48969


namespace NUMINAMATH_CALUDE_differential_equation_solution_l489_48905

/-- The differential equation dy/dx + xy = x^2 has the general solution y(x) = x^3/4 + C/x -/
theorem differential_equation_solution (x : ℝ) (C : ℝ) :
  let y : ℝ → ℝ := λ x => x^3 / 4 + C / x
  let dy_dx : ℝ → ℝ := λ x => 3 * x^2 / 4 - C / x^2
  ∀ x ≠ 0, dy_dx x + x * y x = x^2 := by
sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l489_48905


namespace NUMINAMATH_CALUDE_households_with_car_l489_48985

theorem households_with_car (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 14)
  (h4 : bike_only = 35) :
  ∃ (car : ℕ), car = 44 ∧ 
    car + bike_only + both + neither = total ∧
    car + bike_only + neither = total - both :=
by
  sorry

#check households_with_car

end NUMINAMATH_CALUDE_households_with_car_l489_48985


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l489_48990

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of our geometric sequence -/
def a : ℚ := 1/4

/-- The common ratio of our geometric sequence -/
def r : ℚ := 1/4

/-- The number of terms we're summing -/
def n : ℕ := 6

theorem geometric_sequence_sum : 
  geometricSum a r n = 4095/12288 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l489_48990


namespace NUMINAMATH_CALUDE_stock_worth_calculation_l489_48980

/-- Proves that the total worth of stock is 10000 given the problem conditions -/
theorem stock_worth_calculation (X : ℝ) : 
  (0.04 * X - 0.02 * X = 200) → X = 10000 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_calculation_l489_48980


namespace NUMINAMATH_CALUDE_willies_stickers_l489_48977

/-- The final sticker count for Willie -/
def final_sticker_count (initial_count : ℝ) (received_count : ℝ) : ℝ :=
  initial_count + received_count

/-- Theorem stating that Willie's final sticker count is the sum of his initial count and received stickers -/
theorem willies_stickers (initial_count received_count : ℝ) :
  final_sticker_count initial_count received_count = initial_count + received_count :=
by sorry

end NUMINAMATH_CALUDE_willies_stickers_l489_48977


namespace NUMINAMATH_CALUDE_number_of_fours_is_even_l489_48976

theorem number_of_fours_is_even (x y z : ℕ) : 
  x + y + z = 80 →
  3 * x + 4 * y + 5 * z = 276 →
  Even y :=
by sorry

end NUMINAMATH_CALUDE_number_of_fours_is_even_l489_48976


namespace NUMINAMATH_CALUDE_child_playing_time_l489_48975

/-- Calculates the playing time for each child in a game where 6 children take turns playing for 120 minutes, with only two children playing at a time. -/
theorem child_playing_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) :
  total_time = 120 ∧ num_children = 6 ∧ players_per_game = 2 →
  (total_time * players_per_game) / num_children = 40 := by
  sorry

end NUMINAMATH_CALUDE_child_playing_time_l489_48975


namespace NUMINAMATH_CALUDE_functional_equation_solution_l489_48948

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x - f y) = f x * f y) : 
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l489_48948


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l489_48923

theorem tan_alpha_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β + π/6) = 1/2)
  (h2 : Real.tan (β - π/6) = -1/3) :
  Real.tan (α + π/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l489_48923


namespace NUMINAMATH_CALUDE_complex_equation_proof_l489_48950

theorem complex_equation_proof (x : ℂ) (h : x - 1/x = 3*I) : x^12 - 1/x^12 = 103682 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l489_48950


namespace NUMINAMATH_CALUDE_dinosaur_weight_theorem_l489_48929

def regular_dinosaur_weight : ℕ := 800
def number_of_regular_dinosaurs : ℕ := 5
def barney_weight_difference : ℕ := 1500

def total_weight : ℕ :=
  (regular_dinosaur_weight * number_of_regular_dinosaurs) +
  (regular_dinosaur_weight * number_of_regular_dinosaurs + barney_weight_difference)

theorem dinosaur_weight_theorem :
  total_weight = 9500 :=
by sorry

end NUMINAMATH_CALUDE_dinosaur_weight_theorem_l489_48929


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l489_48979

theorem simplify_trig_expression : 
  (Real.sqrt 3 / Real.cos (10 * π / 180)) - (1 / Real.sin (170 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l489_48979


namespace NUMINAMATH_CALUDE_evaluate_expression_l489_48965

theorem evaluate_expression : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l489_48965


namespace NUMINAMATH_CALUDE_lcm_of_21_and_12_l489_48996

theorem lcm_of_21_and_12 (h : Nat.gcd 21 12 = 6) : Nat.lcm 21 12 = 42 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_21_and_12_l489_48996


namespace NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l489_48913

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 17 * n ≡ 5678 [ZMOD 11] → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 17 * 4 ≡ 5678 [ZMOD 11] :=
by sorry

theorem four_is_smallest : ∀ m : ℕ, m > 0 ∧ m < 4 → ¬(17 * m ≡ 5678 [ZMOD 11]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 17 * n ≡ 5678 [ZMOD 11] ∧ ∀ m : ℕ, m > 0 ∧ 17 * m ≡ 5678 [ZMOD 11] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l489_48913


namespace NUMINAMATH_CALUDE_sequence_problem_l489_48903

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a)
  (h_arith : is_arithmetic_sequence b)
  (h_a : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_b : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l489_48903


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l489_48921

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 ∨ Real.sqrt (Real.sqrt 81) = -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l489_48921


namespace NUMINAMATH_CALUDE_evaluate_expression_l489_48915

theorem evaluate_expression : (2^3)^4 * 3^2 = 36864 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l489_48915


namespace NUMINAMATH_CALUDE_min_max_sum_l489_48945

theorem min_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 4020) : 
  804 ≤ max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))) := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l489_48945


namespace NUMINAMATH_CALUDE_number_greater_than_one_eighth_l489_48932

theorem number_greater_than_one_eighth : ∃ x : ℝ, x = 1/8 + 0.0020000000000000018 ∧ x = 0.1270000000000000018 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_one_eighth_l489_48932


namespace NUMINAMATH_CALUDE_product_of_x_and_z_l489_48935

theorem product_of_x_and_z (x y z : ℕ+) 
  (hx : x = 4 * y) 
  (hz : z = 2 * x) 
  (hsum : x + y + z = 3 * y^2) : 
  (x : ℚ) * (z : ℚ) = 5408 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_x_and_z_l489_48935


namespace NUMINAMATH_CALUDE_complex_square_l489_48933

theorem complex_square (a b : ℝ) (h : (a : ℂ) + Complex.I = 2 - b * Complex.I) :
  (a + b * Complex.I)^2 = 3 - 4 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_l489_48933


namespace NUMINAMATH_CALUDE_quadratic_one_solution_positive_m_value_l489_48949

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) → m = 36 ∨ m = -36 :=
by sorry

theorem positive_m_value (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ m > 0 → m = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_positive_m_value_l489_48949


namespace NUMINAMATH_CALUDE_point_not_on_graph_l489_48987

/-- The function f(x) = x^2 / (x + 1) -/
def f (x : ℚ) : ℚ := x^2 / (x + 1)

/-- The point (-1/2, 1/6) -/
def point : ℚ × ℚ := (-1/2, 1/6)

/-- Theorem: The point (-1/2, 1/6) is not on the graph of f(x) = x^2 / (x + 1) -/
theorem point_not_on_graph : f point.1 ≠ point.2 := by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l489_48987


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l489_48928

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 5 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 25 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l489_48928


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l489_48902

theorem factorial_equation_solution : ∃ (n : ℕ), n > 0 ∧ (n + 1).factorial + (n + 3).factorial = n.factorial * 1190 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l489_48902


namespace NUMINAMATH_CALUDE_tournament_sequences_l489_48919

/-- Represents a team in the tournament -/
structure Team :=
  (players : Finset ℕ)
  (size : players.card = 7)

/-- Represents a tournament between two teams -/
structure Tournament :=
  (teamA : Team)
  (teamB : Team)

/-- Represents a sequence of matches in the tournament -/
def MatchSequence (t : Tournament) := Finset ℕ

/-- The number of possible match sequences in a tournament -/
def numSequences (t : Tournament) : ℕ := Nat.choose 14 7

/-- Theorem: The number of possible match sequences in a tournament
    between two teams of 7 players each is equal to C(14,7) -/
theorem tournament_sequences (t : Tournament) :
  numSequences t = 3432 :=
by sorry

end NUMINAMATH_CALUDE_tournament_sequences_l489_48919


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l489_48947

theorem smallest_integer_solution (x : ℤ) : 
  (x^4 - 40*x^2 + 324 = 0) → x ≥ -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l489_48947


namespace NUMINAMATH_CALUDE_bread_cost_l489_48998

/-- The cost of a loaf of bread, given the costs of ham and cake, and that the combined cost of ham and bread equals the cost of cake. -/
theorem bread_cost (ham_cost cake_cost : ℕ) (h1 : ham_cost = 150) (h2 : cake_cost = 200)
  (h3 : ∃ (bread_cost : ℕ), bread_cost + ham_cost = cake_cost) : 
  ∃ (bread_cost : ℕ), bread_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l489_48998


namespace NUMINAMATH_CALUDE_solve_equation_l489_48936

theorem solve_equation (y : ℚ) (h : 1/3 - 1/4 = 1/y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l489_48936


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_reciprocal_l489_48917

theorem cubic_root_sum_squares_reciprocal (α β γ : ℂ) : 
  α^3 - 6*α^2 + 11*α - 6 = 0 →
  β^3 - 6*β^2 + 11*β - 6 = 0 →
  γ^3 - 6*γ^2 + 11*γ - 6 = 0 →
  α ≠ β → β ≠ γ → γ ≠ α →
  1/α^2 + 1/β^2 + 1/γ^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_reciprocal_l489_48917


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_composite_sum_l489_48941

theorem diophantine_equation_solutions :
  {(m, n) : ℕ × ℕ | 5 * m + 8 * n = 120} = {(24, 0), (16, 5), (8, 10), (0, 15)} := by sorry

theorem composite_sum :
  ∀ (a b c : ℕ+), c > 1 → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c →
  (∃ d : ℕ, 1 < d ∧ d < a + c ∧ (a + c) % d = 0) ∨
  (∃ d : ℕ, 1 < d ∧ d < b + c ∧ (b + c) % d = 0) := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_composite_sum_l489_48941


namespace NUMINAMATH_CALUDE_park_road_perimeter_l489_48955

/-- Given a square park with an inner road, calculates the perimeter of the outer edge of the road. -/
def outer_road_perimeter (park_side_length : ℝ) : ℝ :=
  4 * park_side_length

/-- Calculates the area occupied by the road in the park. -/
def road_area (park_side_length : ℝ) : ℝ :=
  park_side_length^2 - (park_side_length - 6)^2

theorem park_road_perimeter (park_side_length : ℝ) :
  road_area park_side_length = 1764 →
  outer_road_perimeter park_side_length = 600 := by
  sorry


end NUMINAMATH_CALUDE_park_road_perimeter_l489_48955


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l489_48954

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : 
  a^2 / (a + 1) - 1 / (a + 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l489_48954


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l489_48973

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l489_48973


namespace NUMINAMATH_CALUDE_count_representations_l489_48914

/-- The number of ways to represent 5040 in the given form -/
def M : ℕ :=
  (Finset.range 100).sum (fun b₃ =>
    (Finset.range 100).sum (fun b₂ =>
      (Finset.range 100).sum (fun b₁ =>
        (Finset.range 100).sum (fun b₀ =>
          if b₃ * 10^3 + b₂ * 10^2 + b₁ * 10 + b₀ = 5040 then 1 else 0))))

theorem count_representations : M = 504 := by
  sorry

end NUMINAMATH_CALUDE_count_representations_l489_48914


namespace NUMINAMATH_CALUDE_average_of_25_results_l489_48966

theorem average_of_25_results (first_12_avg : ℝ) (last_12_avg : ℝ) (result_13 : ℝ) 
  (h1 : first_12_avg = 14)
  (h2 : last_12_avg = 17)
  (h3 : result_13 = 228) :
  (12 * first_12_avg + result_13 + 12 * last_12_avg) / 25 = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_of_25_results_l489_48966


namespace NUMINAMATH_CALUDE_students_not_enrolled_l489_48960

theorem students_not_enrolled (total : ℕ) (english : ℕ) (history : ℕ) (both : ℕ) : 
  total = 60 → english = 42 → history = 30 → both = 18 →
  total - (english + history - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l489_48960


namespace NUMINAMATH_CALUDE_march_greatest_drop_l489_48991

-- Define the months
inductive Month
| January
| February
| March
| April
| May
| June

-- Define the price change for each month
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => -0.50
  | Month.February => 2.00
  | Month.March => -2.50
  | Month.April => 3.00
  | Month.May => -0.50
  | Month.June => -2.00

-- Define a function to check if a month has a price drop
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

-- Define a function to compare price drops between two months
def greater_price_drop (m1 m2 : Month) : Prop :=
  has_price_drop m1 ∧ has_price_drop m2 ∧ price_change m1 < price_change m2

-- Theorem statement
theorem march_greatest_drop :
  ∀ m : Month, m ≠ Month.March → ¬(greater_price_drop m Month.March) :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l489_48991


namespace NUMINAMATH_CALUDE_ball_drawing_game_l489_48956

def total_balls : ℕ := 10
def red_balls : ℕ := 2
def black_balls : ℕ := 4
def white_balls : ℕ := 4
def win_reward : ℚ := 10
def loss_fine : ℚ := 2
def num_draws : ℕ := 10

def prob_win : ℚ := 1 / 15

theorem ball_drawing_game :
  -- Probability of winning in a single draw
  prob_win = (Nat.choose black_balls 3 + Nat.choose white_balls 3) / Nat.choose total_balls 3 ∧
  -- Probability of more than one win in 10 draws
  1 - (1 - prob_win) ^ num_draws - num_draws * prob_win * (1 - prob_win) ^ (num_draws - 1) = 1 / 6 ∧
  -- Expected total amount won (or lost) by 10 people
  (prob_win * win_reward - (1 - prob_win) * loss_fine) * num_draws = -12
  := by sorry

end NUMINAMATH_CALUDE_ball_drawing_game_l489_48956


namespace NUMINAMATH_CALUDE_yoongi_has_smallest_points_l489_48989

def jungkook_points : ℕ := 9
def yoongi_points : ℕ := 4
def yuna_points : ℕ := 5

theorem yoongi_has_smallest_points : 
  yoongi_points ≤ jungkook_points ∧ yoongi_points ≤ yuna_points :=
by sorry

end NUMINAMATH_CALUDE_yoongi_has_smallest_points_l489_48989


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l489_48926

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n

theorem minimum_value_of_expression (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (∀ k : ℕ, a k > 0) →
  a m * a n = 4 * (a 2)^2 →
  (2 : ℝ) / m + 1 / (2 * n) ≥ 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l489_48926


namespace NUMINAMATH_CALUDE_flag_run_time_l489_48901

/-- The time taken to run between equally spaced flags -/
def run_time (start_flag end_flag : ℕ) (time : ℚ) : Prop :=
  start_flag < end_flag ∧ time > 0 ∧
  ∀ (i j : ℕ), start_flag ≤ i ∧ i < j ∧ j ≤ end_flag →
    (time * (j - i : ℚ)) / (end_flag - start_flag : ℚ) =
    time * ((j - start_flag : ℚ) / (end_flag - start_flag : ℚ) - (i - start_flag : ℚ) / (end_flag - start_flag : ℚ))

theorem flag_run_time :
  run_time 1 8 8 → run_time 1 12 (88/7 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_flag_run_time_l489_48901


namespace NUMINAMATH_CALUDE_daps_equivalent_to_48_dips_l489_48981

/-- Represents the number of units of a currency -/
structure Currency where
  amount : ℚ
  name : String

/-- Defines the exchange rate between two currencies -/
def exchange_rate (a b : Currency) : ℚ := a.amount / b.amount

/-- Given conditions of the problem -/
axiom daps_to_dops : exchange_rate (Currency.mk 5 "daps") (Currency.mk 4 "dops") = 1
axiom dops_to_dips : exchange_rate (Currency.mk 3 "dops") (Currency.mk 8 "dips") = 1

/-- The theorem to be proved -/
theorem daps_equivalent_to_48_dips :
  exchange_rate (Currency.mk 22.5 "daps") (Currency.mk 48 "dips") = 1 := by
  sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_48_dips_l489_48981


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l489_48963

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given A(a,1) and B(5,b) are symmetric with respect to the origin, prove a - b = -4 -/
theorem symmetric_points_difference (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a - b = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l489_48963


namespace NUMINAMATH_CALUDE_quadratic_root_product_l489_48927

theorem quadratic_root_product (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I) ^ 2 + p * (1 - Complex.I) + q = 0 →
  p * q = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l489_48927


namespace NUMINAMATH_CALUDE_cookie_bags_count_l489_48988

theorem cookie_bags_count (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) :
  total_cookies / cookies_per_bag = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_count_l489_48988


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l489_48925

theorem product_of_three_numbers (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = 30)
  (h_first : x = 3 * (y + z))
  (h_second : y = 8 * z) : 
  x * y * z = 125 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l489_48925


namespace NUMINAMATH_CALUDE_sequence_b_decreasing_l489_48959

/-- Given a sequence {a_n} that satisfies the following conditions:
    1) a_1 = 2
    2) 2 * a_n * a_{n+1} = a_n^2 + 1
    Define b_n = (a_n - 1) / (a_n + 1)
    Then the sequence {b_n} is decreasing. -/
theorem sequence_b_decreasing (a : ℕ → ℝ) (b : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, 2 * a n * a (n + 1) = a n ^ 2 + 1) ∧
  (∀ n : ℕ, b n = (a n - 1) / (a n + 1)) →
  ∀ n : ℕ, b (n + 1) < b n :=
by sorry

end NUMINAMATH_CALUDE_sequence_b_decreasing_l489_48959


namespace NUMINAMATH_CALUDE_problem_statement_l489_48953

theorem problem_statement (t : ℝ) : 
  let x := 3 - 2*t
  let y := 5*t + 3
  x = 1 → y = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l489_48953


namespace NUMINAMATH_CALUDE_extra_invitations_needed_carol_extra_invitations_l489_48904

theorem extra_invitations_needed 
  (packs_bought : ℕ) 
  (invitations_per_pack : ℕ) 
  (friends_to_invite : ℕ) : ℕ :=
  friends_to_invite - (packs_bought * invitations_per_pack)

theorem carol_extra_invitations : 
  extra_invitations_needed 2 3 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_invitations_needed_carol_extra_invitations_l489_48904


namespace NUMINAMATH_CALUDE_outfit_count_is_18900_l489_48909

def red_shirts : ℕ := 6
def green_shirts : ℕ := 7
def blue_shirts : ℕ := 8
def pants : ℕ := 9
def green_hats : ℕ := 10
def red_hats : ℕ := 10
def blue_hats : ℕ := 10
def ties_per_color : ℕ := 5

def valid_outfit (shirt_color hat_color : String) : Bool :=
  shirt_color ≠ hat_color

def count_outfits_for_hat_color (hat_color : String) : ℕ :=
  match hat_color with
  | "green" => (red_shirts + blue_shirts) * pants * green_hats * ties_per_color
  | "red" => (green_shirts + blue_shirts) * pants * red_hats * ties_per_color
  | "blue" => (red_shirts + green_shirts) * pants * blue_hats * ties_per_color
  | _ => 0

def total_outfits : ℕ :=
  count_outfits_for_hat_color "green" +
  count_outfits_for_hat_color "red" +
  count_outfits_for_hat_color "blue"

theorem outfit_count_is_18900 : total_outfits = 18900 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_is_18900_l489_48909


namespace NUMINAMATH_CALUDE_empty_quadratic_inequality_implies_a_range_l489_48983

theorem empty_quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4) 4 := by
  sorry

end NUMINAMATH_CALUDE_empty_quadratic_inequality_implies_a_range_l489_48983


namespace NUMINAMATH_CALUDE_nine_point_four_minutes_in_seconds_l489_48995

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * 60

/-- Theorem stating that 9.4 minutes is equal to 564 seconds -/
theorem nine_point_four_minutes_in_seconds : 
  minutes_to_seconds 9.4 = 564 := by
  sorry

end NUMINAMATH_CALUDE_nine_point_four_minutes_in_seconds_l489_48995


namespace NUMINAMATH_CALUDE_edmund_earnings_is_64_l489_48972

/-- Calculates Edmund's earnings for extra chores over two weeks -/
def edmund_earnings (normal_chores_per_week : ℕ) (chores_per_day : ℕ) (days : ℕ) (pay_per_extra_chore : ℕ) : ℕ :=
  let total_chores := chores_per_day * days
  let normal_chores := normal_chores_per_week * 2
  let extra_chores := total_chores - normal_chores
  extra_chores * pay_per_extra_chore

/-- Proves that Edmund's earnings for extra chores over two weeks is $64 -/
theorem edmund_earnings_is_64 :
  edmund_earnings 12 4 14 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_edmund_earnings_is_64_l489_48972


namespace NUMINAMATH_CALUDE_max_four_digit_divisible_by_36_11_l489_48971

def digit_reverse (n : Nat) : Nat :=
  -- Implementation of digit reversal (not provided)
  sorry

theorem max_four_digit_divisible_by_36_11 :
  ∃ (m : Nat),
    1000 ≤ m ∧ m ≤ 9999 ∧
    m % 36 = 0 ∧
    (digit_reverse m) % 36 = 0 ∧
    m % 11 = 0 ∧
    ∀ (k : Nat), 1000 ≤ k ∧ k ≤ 9999 ∧
      k % 36 = 0 ∧ (digit_reverse k) % 36 = 0 ∧ k % 11 = 0 →
      k ≤ m ∧
    m = 9504 :=
by
  sorry

end NUMINAMATH_CALUDE_max_four_digit_divisible_by_36_11_l489_48971


namespace NUMINAMATH_CALUDE_prime_factorization_l489_48951

theorem prime_factorization (n : ℕ) (h : n ≥ 2) :
  ∃ (primes : List ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ (n = primes.prod) := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_l489_48951


namespace NUMINAMATH_CALUDE_group_average_weight_increase_l489_48924

theorem group_average_weight_increase (initial_weight : ℝ) : 
  let group_size : ℕ := 10
  let old_member_weight : ℝ := 58
  let new_member_weight : ℝ := 83
  let weight_difference : ℝ := new_member_weight - old_member_weight
  let average_increase : ℝ := weight_difference / group_size
  average_increase = 2.5 := by sorry

end NUMINAMATH_CALUDE_group_average_weight_increase_l489_48924


namespace NUMINAMATH_CALUDE_square_sum_of_solution_l489_48993

theorem square_sum_of_solution (x y : ℝ) : 
  x * y = 8 → 
  x^2 * y + x * y^2 + x + y = 80 → 
  x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_solution_l489_48993


namespace NUMINAMATH_CALUDE_square_difference_l489_48999

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) : (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l489_48999


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l489_48916

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem first_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_prod : a 2 * a 3 * a 4 = 27) 
  (h_seventh : a 7 = 27) : 
  a 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l489_48916


namespace NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l489_48982

theorem infinitely_many_primes_mod_3_eq_2 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_mod_3_eq_2_l489_48982


namespace NUMINAMATH_CALUDE_equation_transformation_l489_48992

theorem equation_transformation (x : ℝ) : 3*(x+1) - 5*(1-x) = 3*x + 3 - 5 + 5*x := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l489_48992


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l489_48937

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 : ℚ) / 3 = 27 / 5 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l489_48937


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l489_48911

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → (∀ a b : ℕ, a^2 - b^2 = 187 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 205 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l489_48911


namespace NUMINAMATH_CALUDE_exponential_inequality_range_l489_48934

theorem exponential_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (2 : ℝ)^(x^2 - 4*x) > (2 : ℝ)^(2*a*x + a)) ↔ -4 < a ∧ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_range_l489_48934


namespace NUMINAMATH_CALUDE_consecutive_integers_equation_l489_48970

theorem consecutive_integers_equation (x y z : ℤ) : 
  (y = x - 1) →
  (z = x - 2) →
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 11) →
  z = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_equation_l489_48970


namespace NUMINAMATH_CALUDE_right_triangle_area_in_square_yards_l489_48958

/-- The area of a right triangle with legs of 60 feet and 80 feet in square yards -/
theorem right_triangle_area_in_square_yards : 
  let leg1 : ℝ := 60
  let leg2 : ℝ := 80
  let triangle_area_sqft : ℝ := (1/2) * leg1 * leg2
  let sqft_per_sqyd : ℝ := 9
  triangle_area_sqft / sqft_per_sqyd = 800/3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_in_square_yards_l489_48958


namespace NUMINAMATH_CALUDE_quintic_integer_root_count_l489_48962

/-- Represents a polynomial of degree 5 with integer coefficients -/
structure QuinticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The set of possible numbers of integer roots for a quintic polynomial with integer coefficients -/
def PossibleRootCounts : Set ℕ := {0, 1, 2, 3, 5}

/-- Counts the number of integer roots of a quintic polynomial, including multiplicity -/
def countIntegerRoots (p : QuinticPolynomial) : ℕ := sorry

/-- Theorem stating that the number of integer roots of a quintic polynomial with integer coefficients
    can only be 0, 1, 2, 3, or 5 -/
theorem quintic_integer_root_count (p : QuinticPolynomial) :
  countIntegerRoots p ∈ PossibleRootCounts := by sorry

end NUMINAMATH_CALUDE_quintic_integer_root_count_l489_48962


namespace NUMINAMATH_CALUDE_sum_remainder_l489_48942

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_l489_48942


namespace NUMINAMATH_CALUDE_units_digit_of_5_pow_150_plus_7_l489_48920

theorem units_digit_of_5_pow_150_plus_7 : 
  (5^150 + 7) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_5_pow_150_plus_7_l489_48920
