import Mathlib

namespace NUMINAMATH_CALUDE_bridge_brick_ratio_l3521_352103

theorem bridge_brick_ratio (total_bricks : ℕ) (type_a_bricks : ℕ) (other_bricks : ℕ) : 
  total_bricks = 150 →
  type_a_bricks = 40 →
  other_bricks = 90 →
  ∃ (type_b_bricks : ℕ), 
    type_a_bricks + type_b_bricks + other_bricks = total_bricks ∧
    type_b_bricks * 2 = type_a_bricks :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_brick_ratio_l3521_352103


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l3521_352183

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 2
def num_lamps_on : ℕ := 3

def total_arrangements : ℕ := (Nat.choose (num_red_lamps + num_blue_lamps) num_blue_lamps) * 
                               (Nat.choose (num_red_lamps + num_blue_lamps) num_lamps_on)

def constrained_arrangements : ℕ := (Nat.choose (num_red_lamps + num_blue_lamps - 1) (num_blue_lamps - 1)) * 
                                    (Nat.choose (num_red_lamps + num_blue_lamps - 2) (num_lamps_on - 1))

theorem lava_lamp_probability : 
  (constrained_arrangements : ℚ) / total_arrangements = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l3521_352183


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3521_352148

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  (a 1) * (a 19) = 16 →
  (a 1) + (a 19) = 10 →
  (a 8) * (a 10) * (a 12) = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3521_352148


namespace NUMINAMATH_CALUDE_parallelogram_xy_product_l3521_352130

-- Define the parallelogram EFGH
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ

-- Define the theorem
theorem parallelogram_xy_product 
  (EFGH : Parallelogram) 
  (h1 : EFGH.EF = 58) 
  (h2 : ∃ y, EFGH.FG = 4 * y^3) 
  (h3 : ∃ x, EFGH.GH = 3 * x + 5) 
  (h4 : EFGH.HE = 24) :
  ∃ x y, x * y = (53 * Real.rpow 6 (1/3)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_xy_product_l3521_352130


namespace NUMINAMATH_CALUDE_initial_bees_calculation_l3521_352107

/-- Calculates the initial number of bees given the daily hatch rate, daily loss rate,
    number of days, and final number of bees. -/
def initialBees (hatchRate dailyLoss : ℕ) (days : ℕ) (finalBees : ℕ) : ℕ :=
  finalBees - (hatchRate - dailyLoss) * days

theorem initial_bees_calculation 
  (hatchRate dailyLoss days finalBees : ℕ) 
  (hatchRate_pos : hatchRate > dailyLoss) :
  initialBees hatchRate dailyLoss days finalBees = 
    finalBees - (hatchRate - dailyLoss) * days := by
  sorry

#eval initialBees 3000 900 7 27201

end NUMINAMATH_CALUDE_initial_bees_calculation_l3521_352107


namespace NUMINAMATH_CALUDE_number_selection_theorem_l3521_352193

def number_pairs : List (ℕ × ℕ) := [
  (1, 36), (2, 35), (3, 34), (4, 33),
  (5, 32), (6, 31), (7, 30), (8, 29),
  (9, 28), (10, 27), (11, 26), (12, 25)
]

def number_pairs_reduced : List (ℕ × ℕ) := [
  (1, 36), (2, 35), (3, 34), (4, 33),
  (5, 32), (6, 31), (7, 30), (8, 29),
  (9, 28), (10, 27)
]

def is_valid_selection (pairs : List (ℕ × ℕ)) (selection : List Bool) : Prop :=
  selection.length = pairs.length ∧
  (selection.zip pairs).foldl (λ sum (b, (x, y)) => sum + if b then x else y) 0 =
  (selection.zip pairs).foldl (λ sum (b, (x, y)) => sum + if b then y else x) 0

theorem number_selection_theorem :
  (∃ selection, is_valid_selection number_pairs selection) ∧
  (¬ ∃ selection, is_valid_selection number_pairs_reduced selection) := by sorry

end NUMINAMATH_CALUDE_number_selection_theorem_l3521_352193


namespace NUMINAMATH_CALUDE_extremum_implies_a_eq_one_f_less_than_c_squared_implies_c_range_l3521_352189

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 - 9*x + 5

-- Theorem 1: If f has an extremum at x = 1, then a = 1
theorem extremum_implies_a_eq_one (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 1 :=
sorry

-- Theorem 2: If f(x) < c² for all x in [-4, 4], then c is in (-∞, -9) ∪ (9, +∞)
theorem f_less_than_c_squared_implies_c_range :
  (∀ x ∈ Set.Icc (-4) 4, f 1 x < c^2) →
  c ∈ Set.Iio (-9) ∪ Set.Ioi 9 :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_eq_one_f_less_than_c_squared_implies_c_range_l3521_352189


namespace NUMINAMATH_CALUDE_remainder_of_A_mod_9_l3521_352123

-- Define the arithmetic sequence
def arithmetic_sequence : List Nat :=
  List.range 502 |> List.map (fun k => 4 * k + 2)

-- Define the large number A as a string
def A : String :=
  arithmetic_sequence.foldl (fun acc x => acc ++ toString x) ""

-- Theorem statement
theorem remainder_of_A_mod_9 :
  (A.foldl (fun acc c => (10 * acc + c.toNat - '0'.toNat) % 9) 0) = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_A_mod_9_l3521_352123


namespace NUMINAMATH_CALUDE_whitewashing_cost_l3521_352177

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the window dimensions
def window_height : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3

-- Define the cost per square foot
def cost_per_sqft : ℝ := 8

-- Theorem statement
theorem whitewashing_cost :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := window_height * window_width * num_windows
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sqft = 7248 := by
sorry


end NUMINAMATH_CALUDE_whitewashing_cost_l3521_352177


namespace NUMINAMATH_CALUDE_magic_king_episodes_l3521_352173

/-- Calculates the total number of episodes for a TV show with the given parameters -/
def total_episodes (total_seasons : ℕ) (episodes_first_half : ℕ) (episodes_second_half : ℕ) : ℕ :=
  let half_seasons := total_seasons / 2
  half_seasons * episodes_first_half + half_seasons * episodes_second_half

/-- Proves that the TV show Magic King has 225 episodes in total -/
theorem magic_king_episodes : 
  total_episodes 10 20 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l3521_352173


namespace NUMINAMATH_CALUDE_cube_increase_correct_l3521_352118

/-- Represents the percentage increase in a cube's dimensions and properties -/
structure CubeIncrease where
  edge : ℝ
  surface_area : ℝ
  volume : ℝ

/-- The percentage increases when a cube's edge is increased by 60% -/
def cube_increase : CubeIncrease :=
  { edge := 60
  , surface_area := 156
  , volume := 309.6 }

theorem cube_increase_correct :
  let original_edge := 1
  let new_edge := original_edge * (1 + cube_increase.edge / 100)
  let original_surface_area := 6 * original_edge^2
  let new_surface_area := 6 * new_edge^2
  let original_volume := original_edge^3
  let new_volume := new_edge^3
  (new_surface_area / original_surface_area - 1) * 100 = cube_increase.surface_area ∧
  (new_volume / original_volume - 1) * 100 = cube_increase.volume :=
by sorry

end NUMINAMATH_CALUDE_cube_increase_correct_l3521_352118


namespace NUMINAMATH_CALUDE_cocktail_cost_per_litre_l3521_352152

/-- Calculate the cost per litre of a superfruit juice cocktail --/
theorem cocktail_cost_per_litre 
  (mixed_fruit_cost : ℝ) 
  (acai_cost : ℝ) 
  (mixed_fruit_volume : ℝ) 
  (acai_volume : ℝ) 
  (h1 : mixed_fruit_cost = 262.85)
  (h2 : acai_cost = 3104.35)
  (h3 : mixed_fruit_volume = 32)
  (h4 : acai_volume = 21.333333333333332) : 
  ∃ (cost_per_litre : ℝ), abs (cost_per_litre - 1399.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cocktail_cost_per_litre_l3521_352152


namespace NUMINAMATH_CALUDE_digit_sum_reduction_count_l3521_352170

def digitSumReduction (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

def countDigits (d : ℕ) : ℕ := 
  (999999999 / 9 : ℕ) + (if d = 1 then 1 else 0)

theorem digit_sum_reduction_count :
  countDigits 1 = countDigits 2 + 1 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_reduction_count_l3521_352170


namespace NUMINAMATH_CALUDE_least_sum_of_primes_l3521_352116

theorem least_sum_of_primes (p q : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  p > 1 → 
  q > 1 → 
  15 * (p^2 + 1) = 29 * (q^2 + 1) → 
  ∃ (p' q' : ℕ), Nat.Prime p' ∧ Nat.Prime q' ∧ p' > 1 ∧ q' > 1 ∧
    15 * (p'^2 + 1) = 29 * (q'^2 + 1) ∧
    p' + q' = 14 ∧
    ∀ (p'' q'' : ℕ), Nat.Prime p'' → Nat.Prime q'' → p'' > 1 → q'' > 1 →
      15 * (p''^2 + 1) = 29 * (q''^2 + 1) → p'' + q'' ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_primes_l3521_352116


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3521_352133

theorem profit_percentage_calculation (original_cost selling_price : ℝ) :
  original_cost = 3000 →
  selling_price = 3450 →
  (selling_price - original_cost) / original_cost * 100 = 15 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3521_352133


namespace NUMINAMATH_CALUDE_two_identical_objects_five_recipients_l3521_352188

theorem two_identical_objects_five_recipients : ∀ n : ℕ, n = 5 →
  (Nat.choose n 2) + (Nat.choose n 1) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_two_identical_objects_five_recipients_l3521_352188


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3521_352100

theorem trigonometric_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (100 * π / 180)) / 
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + 
   Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3521_352100


namespace NUMINAMATH_CALUDE_ten_n_value_l3521_352137

theorem ten_n_value (n : ℝ) (h : 2 * n = 14) : 10 * n = 70 := by sorry

end NUMINAMATH_CALUDE_ten_n_value_l3521_352137


namespace NUMINAMATH_CALUDE_nap_time_is_three_hours_l3521_352136

-- Define flight duration in minutes
def flight_duration : ℕ := 11 * 60 + 20

-- Define durations of activities in minutes
def reading_time : ℕ := 2 * 60
def movie_time : ℕ := 4 * 60
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 1 * 60 + 10

-- Define total activity time
def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Define nap time in hours
def nap_time_hours : ℕ := (flight_duration - total_activity_time) / 60

-- Theorem statement
theorem nap_time_is_three_hours : nap_time_hours = 3 := by
  sorry

end NUMINAMATH_CALUDE_nap_time_is_three_hours_l3521_352136


namespace NUMINAMATH_CALUDE_money_sharing_l3521_352181

theorem money_sharing (amanda_share : ℕ) (total : ℕ) : 
  amanda_share = 30 →
  3 * total = 16 * amanda_share →
  total = 160 := by sorry

end NUMINAMATH_CALUDE_money_sharing_l3521_352181


namespace NUMINAMATH_CALUDE_product_of_numbers_l3521_352195

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 72) (h2 : x - y = 12) (h3 : x / y = 3/2) :
  x * y = 1244.16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3521_352195


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3521_352160

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (n % 11 = 0) ∧
  (∀ d : ℕ, d ≥ 1 ∧ d ≤ 9 → (∃! p : ℕ, p ≥ 0 ∧ p < 9 ∧ (n / 10^p) % 10 = d))

theorem smallest_valid_number :
  is_valid_number 123475869 ∧
  ∀ m : ℕ, is_valid_number m → m ≥ 123475869 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3521_352160


namespace NUMINAMATH_CALUDE_constant_solution_implies_product_l3521_352178

/-- 
Given constants a and b, if the equation (2kx+a)/3 = 2 + (x-bk)/6 
always has a solution of x = 1 for any k, then ab = -26
-/
theorem constant_solution_implies_product (a b : ℚ) : 
  (∀ k : ℚ, ∃ x : ℚ, x = 1 ∧ (2*k*x + a) / 3 = 2 + (x - b*k) / 6) → 
  a * b = -26 := by
sorry

end NUMINAMATH_CALUDE_constant_solution_implies_product_l3521_352178


namespace NUMINAMATH_CALUDE_inequality_condition_l3521_352175

theorem inequality_condition : 
  (∀ x : ℝ, -3 < x ∧ x < 0 → (x + 3) * (x - 2) < 0) ∧ 
  (∃ x : ℝ, (x + 3) * (x - 2) < 0 ∧ ¬(-3 < x ∧ x < 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l3521_352175


namespace NUMINAMATH_CALUDE_problem_solution_l3521_352179

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem problem_solution :
  (B (-1/3) ⊆ A) ∧
  (∀ a : ℝ, A ∪ B a = A ↔ a = 0 ∨ a = -1/3 ∨ a = -1/5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3521_352179


namespace NUMINAMATH_CALUDE_otimes_composition_l3521_352138

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y

-- State the theorem
theorem otimes_composition (h : ℝ) : otimes h (otimes h h) = 2 * h^2 + h := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l3521_352138


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3521_352182

theorem negative_fraction_comparison :
  -3/4 > -4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3521_352182


namespace NUMINAMATH_CALUDE_log_8_4096_sum_bounds_l3521_352185

theorem log_8_4096_sum_bounds : ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) ≤ Real.log 4096 / Real.log 8 ∧ Real.log 4096 / Real.log 8 < (b : ℝ) ∧ a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_8_4096_sum_bounds_l3521_352185


namespace NUMINAMATH_CALUDE_root_sum_squares_l3521_352174

/-- The polynomial p(x) = 4x^3 - 2x^2 - 15x + 9 -/
def p (x : ℝ) : ℝ := 4 * x^3 - 2 * x^2 - 15 * x + 9

/-- The polynomial q(x) = 12x^3 + 6x^2 - 7x + 1 -/
def q (x : ℝ) : ℝ := 12 * x^3 + 6 * x^2 - 7 * x + 1

/-- A is the largest root of p(x) -/
def A : ℝ := sorry

/-- B is the largest root of q(x) -/
def B : ℝ := sorry

/-- p(x) has exactly three distinct real roots -/
axiom p_has_three_roots : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ (w : ℝ), p w = 0 ↔ w = x ∨ w = y ∨ w = z)

/-- q(x) has exactly three distinct real roots -/
axiom q_has_three_roots : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ (w : ℝ), q w = 0 ↔ w = x ∨ w = y ∨ w = z)

/-- A is a root of p(x) -/
axiom A_is_root_of_p : p A = 0

/-- B is a root of q(x) -/
axiom B_is_root_of_q : q B = 0

/-- A is the largest root of p(x) -/
axiom A_is_largest_root_of_p : ∀ (x : ℝ), p x = 0 → x ≤ A

/-- B is the largest root of q(x) -/
axiom B_is_largest_root_of_q : ∀ (x : ℝ), q x = 0 → x ≤ B

theorem root_sum_squares : A^2 + 3 * B^2 = 4 := by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l3521_352174


namespace NUMINAMATH_CALUDE_expression_evaluation_l3521_352197

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) :
  x^2 - 2*(3*y^2 - x*y) + (y^2 - 2*x*y) = -19 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3521_352197


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3521_352124

theorem min_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  Real.cos (Real.arccos (1/2)) = d / (2 * e) →
  Real.cos (Real.arccos (3/5)) = e / (2 * f) →
  Real.cos (Real.arccos (-1/8)) = f / (2 * d) →
  d + e + f ≥ 33 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3521_352124


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l3521_352161

def digit_set : Finset Nat := {0, 2, 3, 4, 5, 7, 8, 9}

theorem digit_sum_puzzle :
  ∃ (a b c d e f : Nat),
    a ∈ digit_set ∧ b ∈ digit_set ∧ c ∈ digit_set ∧
    d ∈ digit_set ∧ e ∈ digit_set ∧ f ∈ digit_set ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a + b + c = 24 ∧
    b + d + e + f = 14 ∧
    a + b + c + d + e + f = 31 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l3521_352161


namespace NUMINAMATH_CALUDE_cylinder_minimal_material_l3521_352151

/-- For a cylindrical beverage can with a fixed volume, the material used is minimized when the base radius is half the height -/
theorem cylinder_minimal_material (V : ℝ) (h R : ℝ) (h_pos : h > 0) (R_pos : R > 0) :
  V = π * R^2 * h → (∀ R' h', V = π * R'^2 * h' → 2 * π * R^2 + 2 * π * R * h ≤ 2 * π * R'^2 + 2 * π * R' * h') ↔ R = h / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minimal_material_l3521_352151


namespace NUMINAMATH_CALUDE_rope_and_well_depth_l3521_352115

/-- Given a rope of length L and a well of depth H, prove that if L/2 + 9 = H and L/3 + 2 = H, then L = 42 and H = 30. -/
theorem rope_and_well_depth (L H : ℝ) 
  (h1 : L/2 + 9 = H) 
  (h2 : L/3 + 2 = H) : 
  L = 42 ∧ H = 30 := by
sorry

end NUMINAMATH_CALUDE_rope_and_well_depth_l3521_352115


namespace NUMINAMATH_CALUDE_product_of_numbers_l3521_352113

theorem product_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 16) 
  (sum_squares_eq : x^2 + y^2 = 200) : 
  x * y = 28 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3521_352113


namespace NUMINAMATH_CALUDE_jacks_walking_speed_l3521_352167

/-- The problem of determining Jack's walking speed -/
theorem jacks_walking_speed
  (initial_distance : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (lindy_distance : ℝ)
  (h1 : initial_distance = 270)
  (h2 : christina_speed = 5)
  (h3 : lindy_speed = 8)
  (h4 : lindy_distance = 240) :
  ∃ (jack_speed : ℝ),
    jack_speed = 4 ∧
    jack_speed * (lindy_distance / lindy_speed) +
    christina_speed * (lindy_distance / lindy_speed) =
    initial_distance :=
by sorry

end NUMINAMATH_CALUDE_jacks_walking_speed_l3521_352167


namespace NUMINAMATH_CALUDE_extended_annuity_duration_l3521_352132

/-- Calculates the number of years an annuity will last given initial conditions and a delay --/
def calculate_extended_annuity_years (initial_rate : ℝ) (initial_years : ℕ) (annual_payment : ℝ) (delay_years : ℕ) : ℕ :=
  sorry

/-- Theorem stating that under given conditions, the annuity will last for 34 years --/
theorem extended_annuity_duration :
  let initial_rate : ℝ := 0.045
  let initial_years : ℕ := 26
  let annual_payment : ℝ := 5000
  let delay_years : ℕ := 3
  calculate_extended_annuity_years initial_rate initial_years annual_payment delay_years = 34 :=
by sorry

end NUMINAMATH_CALUDE_extended_annuity_duration_l3521_352132


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l3521_352192

theorem min_a_for_quadratic_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 ∧ x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), x > 0 ∧ x ≤ 1/2 → x^2 + b*x + 1 ≥ 0) → b ≥ -5/2) :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l3521_352192


namespace NUMINAMATH_CALUDE_wrong_to_right_ratio_l3521_352139

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) (h1 : total = 24) (h2 : correct = 8) :
  (total - correct) / correct = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_right_ratio_l3521_352139


namespace NUMINAMATH_CALUDE_jane_albert_same_committee_l3521_352154

/-- The number of second-year MBAs -/
def total_mbas : ℕ := 9

/-- The number of committees to be formed -/
def num_committees : ℕ := 3

/-- The number of members in each committee -/
def committee_size : ℕ := 4

/-- The probability that Jane and Albert are on the same committee -/
def probability_same_committee : ℚ := 1 / 6

theorem jane_albert_same_committee :
  let total_ways := (total_mbas.choose committee_size) * ((total_mbas - committee_size).choose committee_size)
  let ways_together := ((total_mbas - 2).choose (committee_size - 2)) * ((total_mbas - committee_size).choose committee_size)
  (ways_together : ℚ) / total_ways = probability_same_committee :=
sorry

end NUMINAMATH_CALUDE_jane_albert_same_committee_l3521_352154


namespace NUMINAMATH_CALUDE_same_solution_implies_b_value_l3521_352172

theorem same_solution_implies_b_value (x b : ℚ) : 
  (3 * x + 5 = 1) → 
  (b * x + 6 = 0) → 
  b = 9/2 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_b_value_l3521_352172


namespace NUMINAMATH_CALUDE_inscribed_triangle_inequality_l3521_352159

/-- A triangle with semiperimeter, inradius, and circumradius -/
structure Triangle where
  semiperimeter : ℝ
  inradius : ℝ
  circumradius : ℝ
  semiperimeter_pos : 0 < semiperimeter
  inradius_pos : 0 < inradius
  circumradius_pos : 0 < circumradius

/-- An inscribed triangle with semiperimeter -/
structure InscribedTriangle (T : Triangle) where
  semiperimeter : ℝ
  semiperimeter_pos : 0 < semiperimeter
  semiperimeter_le : semiperimeter ≤ T.semiperimeter

/-- The theorem stating the inequality for inscribed triangles -/
theorem inscribed_triangle_inequality (T : Triangle) (IT : InscribedTriangle T) :
  T.inradius / T.circumradius ≤ IT.semiperimeter / T.semiperimeter ∧ 
  IT.semiperimeter / T.semiperimeter ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_inequality_l3521_352159


namespace NUMINAMATH_CALUDE_least_multiple_of_first_four_primes_two_ten_divisible_by_first_four_primes_least_multiple_is_two_ten_l3521_352186

theorem least_multiple_of_first_four_primes : 
  ∀ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 210 :=
by sorry

theorem two_ten_divisible_by_first_four_primes : 
  2 ∣ 210 ∧ 3 ∣ 210 ∧ 5 ∣ 210 ∧ 7 ∣ 210 :=
by sorry

theorem least_multiple_is_two_ten : 
  ∃! n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_of_first_four_primes_two_ten_divisible_by_first_four_primes_least_multiple_is_two_ten_l3521_352186


namespace NUMINAMATH_CALUDE_colonization_combinations_l3521_352150

/-- Represents the number of Earth-like planets -/
def earth_like_planets : Nat := 7

/-- Represents the number of Mars-like planets -/
def mars_like_planets : Nat := 8

/-- Represents the colonization units required for an Earth-like planet -/
def earth_like_units : Nat := 3

/-- Represents the colonization units required for a Mars-like planet -/
def mars_like_units : Nat := 1

/-- Represents the total available colonization units -/
def total_units : Nat := 21

/-- Calculates the number of different combinations of planets that can be occupied -/
def count_combinations : Nat := sorry

theorem colonization_combinations : count_combinations = 981 := by sorry

end NUMINAMATH_CALUDE_colonization_combinations_l3521_352150


namespace NUMINAMATH_CALUDE_root_of_quadratic_l3521_352176

theorem root_of_quadratic (x v : ℝ) : 
  x = (-15 - Real.sqrt 409) / 12 →
  v = -23 / 3 →
  6 * x^2 + 15 * x + v = 0 := by sorry

end NUMINAMATH_CALUDE_root_of_quadratic_l3521_352176


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3521_352126

theorem quadratic_roots_sum_of_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (2 * x₁^2 + k * x₁ - 2 * k + 1 = 0) ∧ 
    (2 * x₂^2 + k * x₂ - 2 * k + 1 = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (x₁^2 + x₂^2 = 29/4)) → 
  (k = 3 ∨ k = -11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3521_352126


namespace NUMINAMATH_CALUDE_fraction_simplification_l3521_352169

theorem fraction_simplification :
  (3 / 7 - 2 / 9) / (5 / 12 + 1 / 4) = 13 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3521_352169


namespace NUMINAMATH_CALUDE_square_minus_product_l3521_352163

theorem square_minus_product : (422 + 404)^2 - (4 * 422 * 404) = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_l3521_352163


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3521_352153

theorem fractional_equation_solution :
  ∃ x : ℝ, x ≠ 3 ∧ (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3521_352153


namespace NUMINAMATH_CALUDE_sum_c_d_equals_five_l3521_352112

theorem sum_c_d_equals_five (a b c d : ℝ) 
  (h1 : a + b = 4)
  (h2 : b + c = 7)
  (h3 : a + d = 2) :
  c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_d_equals_five_l3521_352112


namespace NUMINAMATH_CALUDE_landscape_breadth_l3521_352157

theorem landscape_breadth (length width : ℝ) (playground_area : ℝ) : 
  width = 6 * length →
  playground_area = 4200 →
  length * width = 7 * playground_area →
  width = 420 := by
sorry

end NUMINAMATH_CALUDE_landscape_breadth_l3521_352157


namespace NUMINAMATH_CALUDE_no_common_solution_l3521_352109

theorem no_common_solution : ¬∃ y : ℝ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3521_352109


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3521_352110

theorem complex_modulus_problem (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3521_352110


namespace NUMINAMATH_CALUDE_family_admission_price_l3521_352125

/-- The total price for a family's admission to an amusement park --/
def total_price (adult_price child_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

/-- Theorem: The total price for a family of 2 adults and 2 children,
    with adult tickets costing $22 and child tickets costing $7, is $58 --/
theorem family_admission_price :
  total_price 22 7 2 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_family_admission_price_l3521_352125


namespace NUMINAMATH_CALUDE_cost_of_pencils_l3521_352164

/-- The cost of a single pencil in cents -/
def pencil_cost : ℕ := 3

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of pencils to calculate the cost for -/
def num_pencils : ℕ := 500

/-- Theorem: The cost of 500 pencils in dollars is 15.00 -/
theorem cost_of_pencils : 
  (num_pencils * pencil_cost) / cents_per_dollar = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_pencils_l3521_352164


namespace NUMINAMATH_CALUDE_parabola_with_directrix_x_eq_1_l3521_352144

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- The standard equation of a parabola represents the set of points (x, y) that satisfy the parabola's definition. -/
def standard_equation (p : Parabola) : (ℝ × ℝ) → Prop :=
  sorry

theorem parabola_with_directrix_x_eq_1 (p : Parabola) (h : p.directrix = 1) :
  standard_equation p = fun (x, y) ↦ y^2 = -4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_x_eq_1_l3521_352144


namespace NUMINAMATH_CALUDE_euler_line_equation_l3521_352127

/-- The Euler line of a triangle ABC with vertices A(2,0), B(0,4), and AC = BC -/
def euler_line (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2 * p.2 + 3 = 0}

/-- Triangle ABC with given properties -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A_coord : A = (2, 0)
  B_coord : B = (0, 4)
  isosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem euler_line_equation (t : TriangleABC) :
  euler_line t.A t.B t.C = {p : ℝ × ℝ | p.1 - 2 * p.2 + 3 = 0} :=
by sorry

end NUMINAMATH_CALUDE_euler_line_equation_l3521_352127


namespace NUMINAMATH_CALUDE_correct_ratio_achieved_l3521_352166

/-- Represents the ratio of diesel to water in the final mixture -/
def diesel_water_ratio : ℚ := 3 / 5

/-- The initial amount of diesel in quarts -/
def initial_diesel : ℚ := 4

/-- The initial amount of petrol in quarts -/
def initial_petrol : ℚ := 4

/-- The amount of water to be added in quarts -/
def water_to_add : ℚ := 20 / 3

/-- Theorem stating that adding the calculated amount of water results in the desired ratio -/
theorem correct_ratio_achieved :
  diesel_water_ratio = initial_diesel / water_to_add := by
  sorry

#check correct_ratio_achieved

end NUMINAMATH_CALUDE_correct_ratio_achieved_l3521_352166


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l3521_352117

/-- Calculates the market value of a machine after a given number of years,
    given its initial value and yearly depreciation rate. -/
def marketValue (initialValue : ℝ) (depreciationRate : ℝ) (years : ℕ) : ℝ :=
  initialValue - (depreciationRate * initialValue * years)

/-- Theorem stating that a machine with an initial value of $8,000 and a yearly
    depreciation of 30% of its purchase price will have a market value of $3,200
    after 2 years. -/
theorem machine_value_after_two_years :
  marketValue 8000 0.3 2 = 3200 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_after_two_years_l3521_352117


namespace NUMINAMATH_CALUDE_sum_a_d_equals_five_l3521_352146

theorem sum_a_d_equals_five 
  (a b c d : ℤ) 
  (eq1 : a + b = 11) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_five_l3521_352146


namespace NUMINAMATH_CALUDE_vector_representation_l3521_352198

def a : Fin 2 → ℝ := ![3, -1]
def e1B : Fin 2 → ℝ := ![-1, 2]
def e2B : Fin 2 → ℝ := ![3, 2]
def e1A : Fin 2 → ℝ := ![0, 0]
def e2A : Fin 2 → ℝ := ![3, 2]
def e1C : Fin 2 → ℝ := ![3, 5]
def e2C : Fin 2 → ℝ := ![6, 10]
def e1D : Fin 2 → ℝ := ![-3, 5]
def e2D : Fin 2 → ℝ := ![3, -5]

theorem vector_representation :
  (∃ α β : ℝ, a = α • e1B + β • e2B) ∧
  (∀ α β : ℝ, a ≠ α • e1A + β • e2A) ∧
  (∀ α β : ℝ, a ≠ α • e1C + β • e2C) ∧
  (∀ α β : ℝ, a ≠ α • e1D + β • e2D) :=
by sorry

end NUMINAMATH_CALUDE_vector_representation_l3521_352198


namespace NUMINAMATH_CALUDE_no_xy_term_implies_k_eq_two_l3521_352122

/-- Given a polynomial x^2 + kxy + 4x - 2xy + y^2 - 1, if it does not contain the term xy, then k = 2 -/
theorem no_xy_term_implies_k_eq_two (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x*y + 4*x - 2*x*y + y^2 - 1 = x^2 + 4*x + y^2 - 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_implies_k_eq_two_l3521_352122


namespace NUMINAMATH_CALUDE_brad_reads_more_than_greg_l3521_352134

/-- Greg's daily reading pages -/
def greg_pages : ℕ := 18

/-- Brad's daily reading pages -/
def brad_pages : ℕ := 26

/-- The difference in pages read between Brad and Greg -/
def page_difference : ℕ := brad_pages - greg_pages

theorem brad_reads_more_than_greg : page_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_brad_reads_more_than_greg_l3521_352134


namespace NUMINAMATH_CALUDE_five_items_four_boxes_l3521_352184

/-- The number of ways to distribute n distinct items into k identical boxes --/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 46 ways to distribute 5 distinct items into 4 identical boxes --/
theorem five_items_four_boxes : distribute 5 4 = 46 := by sorry

end NUMINAMATH_CALUDE_five_items_four_boxes_l3521_352184


namespace NUMINAMATH_CALUDE_geometry_relations_l3521_352120

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel_line : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (m l : Line) (α β : Plane)
  (h1 : perp_line_plane m α)
  (h2 : subset_line_plane l β) :
  (parallel_plane α β → perp_line m l) ∧
  (parallel_line m l → perp_plane α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l3521_352120


namespace NUMINAMATH_CALUDE_wendy_trip_miles_l3521_352190

theorem wendy_trip_miles : 
  let day1_miles : ℕ := 125
  let day2_miles : ℕ := 223
  let day3_miles : ℕ := 145
  day1_miles + day2_miles + day3_miles = 493 := by sorry

end NUMINAMATH_CALUDE_wendy_trip_miles_l3521_352190


namespace NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l3521_352168

theorem arctan_sum_of_cubic_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 - 10*x₁ + 11 = 0 →
  x₂^3 - 10*x₂ + 11 = 0 →
  x₃^3 - 10*x₃ + 11 = 0 →
  -5 < x₁ ∧ x₁ < 5 →
  -5 < x₂ ∧ x₂ < 5 →
  -5 < x₃ ∧ x₃ < 5 →
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = π/4 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_of_cubic_roots_l3521_352168


namespace NUMINAMATH_CALUDE_sequence_problem_l3521_352102

/-- Given a sequence {aₙ} where a₂ = 3, a₄ = 15, and {aₙ₊₁} is a geometric sequence, prove that a₆ = 63. -/
theorem sequence_problem (a : ℕ → ℝ) 
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) + 1 = (a n + 1) * q) :
  a 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3521_352102


namespace NUMINAMATH_CALUDE_certain_number_is_thirty_l3521_352108

theorem certain_number_is_thirty : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → ∃ (b : ℕ), k * n = b^2 → k ≥ 30) ∧
  (∃ (b : ℕ), 30 * n = b^2) ∧
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_thirty_l3521_352108


namespace NUMINAMATH_CALUDE_strawberry_preference_percentage_l3521_352145

def total_responses : ℕ := 80 + 70 + 90 + 60 + 50
def strawberry_responses : ℕ := 90

def strawberry_percentage : ℚ :=
  (strawberry_responses : ℚ) / (total_responses : ℚ) * 100

theorem strawberry_preference_percentage :
  (strawberry_percentage : ℚ) = 25.71 := by sorry

end NUMINAMATH_CALUDE_strawberry_preference_percentage_l3521_352145


namespace NUMINAMATH_CALUDE_red_peaches_count_l3521_352105

theorem red_peaches_count (total_peaches : ℕ) (num_baskets : ℕ) (green_peaches : ℕ) :
  total_peaches = 10 →
  num_baskets = 1 →
  green_peaches = 6 →
  total_peaches - green_peaches = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l3521_352105


namespace NUMINAMATH_CALUDE_function_value_symmetry_l3521_352147

/-- Given a function f(x) = ax^5 - bx^3 + cx where a, b, c are real numbers,
    if f(-3) = 7, then f(3) = -7 -/
theorem function_value_symmetry (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 - b * x^3 + c * x)
    (h2 : f (-3) = 7) : 
  f 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_symmetry_l3521_352147


namespace NUMINAMATH_CALUDE_retail_prices_correct_l3521_352111

def calculate_retail_price (wholesale_price : ℚ) : ℚ :=
  let tax_rate : ℚ := 5 / 100
  let shipping_fee : ℚ := 10
  let profit_margin_rate : ℚ := 20 / 100
  let total_cost : ℚ := wholesale_price + (wholesale_price * tax_rate) + shipping_fee
  let profit_margin : ℚ := wholesale_price * profit_margin_rate
  total_cost + profit_margin

theorem retail_prices_correct :
  let machine1_wholesale : ℚ := 99
  let machine2_wholesale : ℚ := 150
  let machine3_wholesale : ℚ := 210
  (calculate_retail_price machine1_wholesale = 133.75) ∧
  (calculate_retail_price machine2_wholesale = 197.50) ∧
  (calculate_retail_price machine3_wholesale = 272.50) := by
  sorry

end NUMINAMATH_CALUDE_retail_prices_correct_l3521_352111


namespace NUMINAMATH_CALUDE_books_bought_from_first_shop_l3521_352140

theorem books_bought_from_first_shop
  (total_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_second_shop : ℕ)
  (average_price : ℕ)
  (h1 : total_first_shop = 600)
  (h2 : books_second_shop = 20)
  (h3 : total_second_shop = 240)
  (h4 : average_price = 14)
  : ∃ (books_first_shop : ℕ),
    (total_first_shop + total_second_shop) / (books_first_shop + books_second_shop) = average_price ∧
    books_first_shop = 40 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_from_first_shop_l3521_352140


namespace NUMINAMATH_CALUDE_toy_poodle_height_l3521_352187

/-- Proves that the height of a toy poodle is 14 inches given the heights of standard and miniature poodles -/
theorem toy_poodle_height 
  (standard_height : ℕ) 
  (standard_miniature_diff : ℕ) 
  (miniature_toy_diff : ℕ) 
  (h1 : standard_height = 28)
  (h2 : standard_miniature_diff = 8)
  (h3 : miniature_toy_diff = 6) : 
  standard_height - standard_miniature_diff - miniature_toy_diff = 14 := by
  sorry

#check toy_poodle_height

end NUMINAMATH_CALUDE_toy_poodle_height_l3521_352187


namespace NUMINAMATH_CALUDE_expression_evaluation_l3521_352194

theorem expression_evaluation :
  let f (x : ℚ) := (3 * x + 2) / (2 * x - 1)
  let g (x : ℚ) := (3 * f x + 2) / (2 * f x - 1)
  g (1/3) = 113/31 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3521_352194


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3521_352129

theorem unknown_number_proof (x : ℝ) : x + 5 * 12 / (180 / 3) = 41 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3521_352129


namespace NUMINAMATH_CALUDE_cube_labeling_theorem_l3521_352171

/-- A labeling of a cube's edges -/
def CubeLabeling := Fin 12 → Fin 13

/-- The sum of labels at a vertex given a labeling -/
def vertexSum (l : CubeLabeling) (v : Fin 8) : ℕ := sorry

/-- Predicate for a valid labeling using numbers 1 to 12 -/
def validLabeling12 (l : CubeLabeling) : Prop :=
  (∀ i : Fin 12, l i < 13) ∧ (∀ i j : Fin 12, i ≠ j → l i ≠ l j)

/-- Predicate for a valid labeling using numbers 1 to 13 with one unused -/
def validLabeling13 (l : CubeLabeling) : Prop :=
  (∀ i : Fin 12, l i > 0) ∧ (∀ i j : Fin 12, i ≠ j → l i ≠ l j)

theorem cube_labeling_theorem :
  (∀ l : CubeLabeling, validLabeling12 l →
    ∃ v1 v2 : Fin 8, v1 ≠ v2 ∧ vertexSum l v1 ≠ vertexSum l v2) ∧
  (∃ l : CubeLabeling, validLabeling13 l ∧
    ∀ v1 v2 : Fin 8, vertexSum l v1 = vertexSum l v2) :=
by sorry

end NUMINAMATH_CALUDE_cube_labeling_theorem_l3521_352171


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l3521_352162

theorem quadratic_always_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_range_l3521_352162


namespace NUMINAMATH_CALUDE_count_solution_pairs_l3521_352104

/-- The number of pairs of positive integers (x, y) satisfying 2x + 3y = 2007 -/
def solution_count : ℕ := 334

/-- The predicate that checks if a pair of natural numbers satisfies the equation -/
def satisfies_equation (x y : ℕ) : Prop :=
  2 * x + 3 * y = 2007

theorem count_solution_pairs :
  (∃! n : ℕ, n = solution_count ∧
    ∃ s : Finset (ℕ × ℕ),
      s.card = n ∧
      (∀ p : ℕ × ℕ, p ∈ s ↔ (satisfies_equation p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0))) :=
sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l3521_352104


namespace NUMINAMATH_CALUDE_train_length_calculation_l3521_352149

theorem train_length_calculation (passing_time man_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  passing_time = 8 →
  man_time = 8 →
  platform_length = 273 →
  platform_time = 20 →
  ∃ (train_length : ℝ), train_length = 182 ∧
    train_length / man_time = (train_length + platform_length) / platform_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3521_352149


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l3521_352106

/-- Systematic sampling selection function -/
def systematicSample (initialSelection : ℕ) (interval : ℕ) (groupNumber : ℕ) : ℕ :=
  initialSelection + interval * (groupNumber - 1)

/-- Theorem for the systematic sampling problem -/
theorem systematic_sampling_problem (totalStudents : ℕ) (sampleSize : ℕ) (interval : ℕ) 
    (initialSelection : ℕ) (targetGroupStart : ℕ) (targetGroupEnd : ℕ) :
    totalStudents = 800 →
    sampleSize = 50 →
    interval = 16 →
    initialSelection = 7 →
    targetGroupStart = 65 →
    targetGroupEnd = 80 →
    systematicSample initialSelection interval 
      ((targetGroupStart - 1) / interval + 1) = 71 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_problem_l3521_352106


namespace NUMINAMATH_CALUDE_martha_cards_l3521_352135

/-- The number of cards Martha ends up with after receiving more cards -/
def total_cards (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Martha ends up with 79 cards -/
theorem martha_cards : total_cards 3 76 = 79 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l3521_352135


namespace NUMINAMATH_CALUDE_eggs_per_group_l3521_352131

/-- Given 15 eggs split into 3 equal groups, prove that each group contains 5 eggs. -/
theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (h1 : total_eggs = 15) (h2 : num_groups = 3) :
  total_eggs / num_groups = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_group_l3521_352131


namespace NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l3521_352155

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem product_one_when_equal_absolute_log 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hf : f a = f b) : 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l3521_352155


namespace NUMINAMATH_CALUDE_overlapping_semicircles_area_l3521_352128

/-- Given a pattern of overlapping semicircles, this theorem calculates the shaded area. -/
theorem overlapping_semicircles_area (diameter : ℝ) (overlap : ℝ) (total_length : ℝ) : 
  diameter = 3 ∧ overlap = 0.5 ∧ total_length = 12 →
  (∃ (shaded_area : ℝ), shaded_area = 5.625 * Real.pi) := by
  sorry

#check overlapping_semicircles_area

end NUMINAMATH_CALUDE_overlapping_semicircles_area_l3521_352128


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3521_352196

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 → n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3521_352196


namespace NUMINAMATH_CALUDE_gcd_of_160_200_360_l3521_352114

theorem gcd_of_160_200_360 : Nat.gcd 160 (Nat.gcd 200 360) = 40 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_160_200_360_l3521_352114


namespace NUMINAMATH_CALUDE_congested_sections_probability_l3521_352165

/-- The probability of selecting exactly 4 congested sections out of 10 randomly selected sections,
    given that there are 7 congested sections out of 16 total sections. -/
theorem congested_sections_probability :
  let total_sections : ℕ := 16
  let congested_sections : ℕ := 7
  let selected_sections : ℕ := 10
  let target_congested : ℕ := 4
  
  (Nat.choose congested_sections target_congested *
   Nat.choose (total_sections - congested_sections) (selected_sections - target_congested)) /
  Nat.choose total_sections selected_sections =
  (Nat.choose congested_sections target_congested *
   Nat.choose (total_sections - congested_sections) (selected_sections - target_congested)) /
  Nat.choose total_sections selected_sections :=
by
  sorry

end NUMINAMATH_CALUDE_congested_sections_probability_l3521_352165


namespace NUMINAMATH_CALUDE_division_rebus_proof_l3521_352101

theorem division_rebus_proof :
  -- Given conditions
  let dividend : ℕ := 1089708
  let divisor : ℕ := 12
  let quotient : ℕ := 90809

  -- Divisor is a two-digit number
  (10 ≤ divisor) ∧ (divisor < 100) →
  
  -- When divisor is multiplied by 8, it results in a two-digit number
  (10 ≤ divisor * 8) ∧ (divisor * 8 < 100) →
  
  -- When divisor is multiplied by the first (or last) digit of quotient, it results in a three-digit number
  (100 ≤ divisor * (quotient / 10000)) ∧ (divisor * (quotient / 10000) < 1000) →
  
  -- Quotient has 5 digits
  (10000 ≤ quotient) ∧ (quotient < 100000) →
  
  -- Second and fourth digits of quotient are 0
  (quotient % 10000 / 1000 = 0) ∧ (quotient % 100 / 10 = 0) →
  
  -- The division problem has a unique solution
  ∃! (d q : ℕ), d * q = dividend ∧ q = quotient →
  
  -- Prove that the division is correct
  dividend / divisor = quotient ∧ dividend % divisor = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_division_rebus_proof_l3521_352101


namespace NUMINAMATH_CALUDE_root_sum_equation_l3521_352180

theorem root_sum_equation (a b : ℝ) : 
  (Complex.I + 1) ^ 2 * a + (Complex.I + 1) * b + 2 = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_equation_l3521_352180


namespace NUMINAMATH_CALUDE_exists_multiple_factorizations_l3521_352119

-- Define the set V
def V (p : Nat) : Set Nat :=
  {n : Nat | ∃ k : Nat, (n = k * p + 1 ∨ n = k * p - 1) ∧ k > 0}

-- Define indecomposability in V
def isIndecomposable (p : Nat) (n : Nat) : Prop :=
  n ∈ V p ∧ ∀ k l : Nat, k ∈ V p → l ∈ V p → n ≠ k * l

-- Theorem statement
theorem exists_multiple_factorizations (p : Nat) (h : p > 5) :
  ∃ N : Nat, N ∈ V p ∧
    ∃ (factors1 factors2 : List Nat),
      factors1 ≠ factors2 ∧
      (∀ f ∈ factors1, isIndecomposable p f) ∧
      (∀ f ∈ factors2, isIndecomposable p f) ∧
      N = factors1.prod ∧
      N = factors2.prod :=
by sorry

end NUMINAMATH_CALUDE_exists_multiple_factorizations_l3521_352119


namespace NUMINAMATH_CALUDE_remainder_of_g_x12_div_g_x_l3521_352158

/-- The polynomial g(x) = x^5 + x^4 + x^3 + x^2 + x + 1 -/
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The theorem stating that the remainder of g(x^12) divided by g(x) is 6 -/
theorem remainder_of_g_x12_div_g_x :
  ∃ (q : ℝ → ℝ), g (x^12) = g x * q x + 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_g_x12_div_g_x_l3521_352158


namespace NUMINAMATH_CALUDE_total_highlighters_l3521_352191

theorem total_highlighters (yellow : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : yellow = 7)
  (h2 : pink = yellow + 7)
  (h3 : blue = pink + 5) :
  yellow + pink + blue = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l3521_352191


namespace NUMINAMATH_CALUDE_mark_election_votes_l3521_352143

theorem mark_election_votes (first_area_voters : ℕ) (first_area_percentage : ℚ) :
  first_area_voters = 100000 →
  first_area_percentage = 70 / 100 →
  (first_area_voters * first_area_percentage).floor +
  2 * (first_area_voters * first_area_percentage).floor = 210000 :=
by sorry

end NUMINAMATH_CALUDE_mark_election_votes_l3521_352143


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l3521_352156

/-- Two circles are internally tangent if the distance between their centers
    plus the radius of the smaller circle equals the radius of the larger circle. -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d + min r₁ r₂ = max r₁ r₂

/-- The theorem states that two circles with radii 3 and 7, whose centers are 4 units apart,
    are internally tangent. -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 3
  let r₂ : ℝ := 7
  let d : ℝ := 4
  internally_tangent r₁ r₂ d :=
by
  sorry


end NUMINAMATH_CALUDE_circles_internally_tangent_l3521_352156


namespace NUMINAMATH_CALUDE_tomato_types_salad_bar_problem_l3521_352141

theorem tomato_types (lettuce_types : Nat) (olive_types : Nat) (soup_types : Nat) 
  (total_options : Nat) : Nat :=
  let tomato_types := total_options / (lettuce_types * olive_types * soup_types)
  tomato_types

theorem salad_bar_problem :
  let lettuce_types := 2
  let olive_types := 4
  let soup_types := 2
  let total_options := 48
  tomato_types lettuce_types olive_types soup_types total_options = 3 := by
  sorry

end NUMINAMATH_CALUDE_tomato_types_salad_bar_problem_l3521_352141


namespace NUMINAMATH_CALUDE_sequence_inequality_l3521_352142

theorem sequence_inequality (a : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, a n > 0) 
  (h2 : k > 0) (h3 : ∀ n, a (n + 1) ≤ (a n)^k * (1 - a n)) :
  ∀ n ≥ 2, (1 / a n) ≥ ((k + 1 : ℝ)^(k + 1) / k^k) + (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3521_352142


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l3521_352121

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_with_current = 18)
  (h2 : current_speed = 3.4) :
  speed_with_current - 2 * current_speed = 11.2 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l3521_352121


namespace NUMINAMATH_CALUDE_circle_equation_l3521_352199

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line_l1 (x y : ℝ) : Prop := x - 3 * y = 0
def line_l2 (x y : ℝ) : Prop := x - y = 0

-- Define the conditions
def tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

def center_on_l1 (c : Circle) : Prop :=
  line_l1 c.center.1 c.center.2

def intersects_l2_with_chord (c : Circle) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l2 x₁ y₁ ∧ line_l2 x₂ y₂ ∧
    (x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
    (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8

-- Define the theorem
theorem circle_equation (c : Circle) :
  tangent_to_y_axis c →
  center_on_l1 c →
  intersects_l2_with_chord c →
  ((∀ x y, (x - 6 * Real.sqrt 2)^2 + (y - 2 * Real.sqrt 2)^2 = 72) ∨
   (∀ x y, (x + 6 * Real.sqrt 2)^2 + (y + 2 * Real.sqrt 2)^2 = 72)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3521_352199
