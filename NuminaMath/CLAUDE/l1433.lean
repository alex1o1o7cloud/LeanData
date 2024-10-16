import Mathlib

namespace NUMINAMATH_CALUDE_beetles_consumed_1080_l1433_143321

/-- Represents the daily consumption and population changes in a tropical forest ecosystem --/
structure ForestEcosystem where
  beetles_per_bird : ℕ
  birds_per_snake : ℕ
  snakes_per_jaguar : ℕ
  jaguars_per_crocodile : ℕ
  bird_increase : ℕ
  snake_increase : ℕ
  jaguar_increase : ℕ
  initial_jaguars : ℕ
  initial_crocodiles : ℕ

/-- Calculates the number of beetles consumed in one day in the forest ecosystem --/
def beetles_consumed (eco : ForestEcosystem) : ℕ :=
  eco.initial_jaguars * eco.snakes_per_jaguar * eco.birds_per_snake * eco.beetles_per_bird

/-- Theorem stating that the number of beetles consumed in one day is 1080 --/
theorem beetles_consumed_1080 (eco : ForestEcosystem) 
  (h1 : eco.beetles_per_bird = 12)
  (h2 : eco.birds_per_snake = 3)
  (h3 : eco.snakes_per_jaguar = 5)
  (h4 : eco.jaguars_per_crocodile = 2)
  (h5 : eco.bird_increase = 4)
  (h6 : eco.snake_increase = 2)
  (h7 : eco.jaguar_increase = 1)
  (h8 : eco.initial_jaguars = 6)
  (h9 : eco.initial_crocodiles = 30) :
  beetles_consumed eco = 1080 := by
  sorry


end NUMINAMATH_CALUDE_beetles_consumed_1080_l1433_143321


namespace NUMINAMATH_CALUDE_equation_solutions_l1433_143399

theorem equation_solutions : 
  {x : ℝ | (x - 2)^2 + (x - 2) = 0} = {2, 1} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1433_143399


namespace NUMINAMATH_CALUDE_spiders_can_catch_fly_l1433_143311

-- Define the cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)

-- Define the creatures
inductive Creature
| Spider
| Fly

-- Define the position of a creature on the cube
structure Position where
  creature : Creature
  vertex : Fin 8

-- Define the speed of creatures
def speed (c : Creature) : ℕ :=
  match c with
  | Creature.Spider => 1
  | Creature.Fly => 3

-- Define the initial state
def initial_state (cube : Cube) : Finset Position :=
  sorry

-- Define the catching condition
def can_catch (cube : Cube) (positions : Finset Position) : Prop :=
  sorry

-- The main theorem
theorem spiders_can_catch_fly (cube : Cube) :
  ∃ (final_positions : Finset Position),
    can_catch cube final_positions :=
  sorry

end NUMINAMATH_CALUDE_spiders_can_catch_fly_l1433_143311


namespace NUMINAMATH_CALUDE_complex_fraction_third_quadrant_l1433_143309

/-- Given a complex fraction equal to 2-i, prove the resulting point is in the third quadrant -/
theorem complex_fraction_third_quadrant (a b : ℝ) : 
  (a + Complex.I) / (b - Complex.I) = 2 - Complex.I → 
  a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_third_quadrant_l1433_143309


namespace NUMINAMATH_CALUDE_percentage_relation_l1433_143393

theorem percentage_relation (p t j : ℝ) (e : ℝ) : 
  j = 0.75 * p → 
  j = 0.8 * t → 
  t = p * (1 - e / 100) → 
  e = 6.25 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l1433_143393


namespace NUMINAMATH_CALUDE_num_bounces_correct_l1433_143306

/-- The initial height of the ball in meters -/
def initial_height : ℝ := 500

/-- The ratio of the bounce height to the previous height -/
def bounce_ratio : ℝ := 0.6

/-- The height threshold for counting bounces, in meters -/
def bounce_threshold : ℝ := 5

/-- The height at which the ball stops bouncing, in meters -/
def stop_threshold : ℝ := 0.1

/-- The height of the ball after k bounces -/
def height_after_bounces (k : ℕ) : ℝ := initial_height * bounce_ratio ^ k

/-- The number of bounces after which the ball first reaches a maximum height less than the bounce threshold -/
def num_bounces : ℕ := sorry

theorem num_bounces_correct :
  (∀ k < num_bounces, height_after_bounces k ≥ bounce_threshold) ∧
  height_after_bounces num_bounces < bounce_threshold ∧
  (∀ n : ℕ, height_after_bounces n ≥ stop_threshold → n ≤ num_bounces) ∧
  num_bounces = 10 := by sorry

end NUMINAMATH_CALUDE_num_bounces_correct_l1433_143306


namespace NUMINAMATH_CALUDE_bobby_chocolate_pieces_l1433_143314

/-- The number of chocolate pieces Bobby ate -/
def chocolate_pieces : ℕ := 58

/-- The number of candy pieces Bobby ate initially -/
def initial_candy : ℕ := 38

/-- The number of additional candy pieces Bobby ate -/
def additional_candy : ℕ := 36

/-- The difference between candy and chocolate pieces -/
def candy_chocolate_difference : ℕ := 58

theorem bobby_chocolate_pieces :
  chocolate_pieces = 
    (initial_candy + additional_candy + candy_chocolate_difference) - (initial_candy + additional_candy) :=
by
  sorry

end NUMINAMATH_CALUDE_bobby_chocolate_pieces_l1433_143314


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l1433_143315

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1052 ∧ m = 23) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l1433_143315


namespace NUMINAMATH_CALUDE_extra_minutes_per_A_is_correct_l1433_143391

/-- The number of extra minutes earned for each A grade -/
def extra_minutes_per_A : ℕ := 2

/-- The normal recess time in minutes -/
def normal_recess : ℕ := 20

/-- The number of A grades -/
def num_A : ℕ := 10

/-- The number of B grades -/
def num_B : ℕ := 12

/-- The number of C grades -/
def num_C : ℕ := 14

/-- The number of D grades -/
def num_D : ℕ := 5

/-- The total recess time in minutes -/
def total_recess : ℕ := 47

theorem extra_minutes_per_A_is_correct :
  extra_minutes_per_A * num_A + num_B - num_D = total_recess - normal_recess :=
by sorry

end NUMINAMATH_CALUDE_extra_minutes_per_A_is_correct_l1433_143391


namespace NUMINAMATH_CALUDE_box_percentage_difference_l1433_143325

theorem box_percentage_difference
  (stan_boxes : ℕ)
  (john_boxes : ℕ)
  (jules_boxes : ℕ)
  (joseph_boxes : ℕ)
  (h1 : stan_boxes = 100)
  (h2 : john_boxes = 30)
  (h3 : john_boxes = jules_boxes + jules_boxes / 5)
  (h4 : jules_boxes = joseph_boxes + 5) :
  (stan_boxes - joseph_boxes) / stan_boxes = 4/5 :=
sorry

end NUMINAMATH_CALUDE_box_percentage_difference_l1433_143325


namespace NUMINAMATH_CALUDE_solve_equation_l1433_143394

theorem solve_equation : ∃ x : ℚ, (3 * x + 5) / 7 = 13 ∧ x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1433_143394


namespace NUMINAMATH_CALUDE_tiles_in_row_l1433_143320

theorem tiles_in_row (area : ℝ) (length : ℝ) (tile_size : ℝ) : 
  area = 320 → length = 16 → tile_size = 1 → 
  (area / length) / tile_size = 20 := by sorry

end NUMINAMATH_CALUDE_tiles_in_row_l1433_143320


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_specific_digits_l1433_143379

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def has_odd_units_and_thousands (n : ℕ) : Prop :=
  n % 2 = 1 ∧ (n / 1000) % 2 = 1

def has_even_tens_and_hundreds (n : ℕ) : Prop :=
  ((n / 10) % 10) % 2 = 0 ∧ ((n / 100) % 10) % 2 = 0

theorem smallest_four_digit_divisible_by_9_with_specific_digits : 
  ∀ n : ℕ, is_four_digit n → 
  is_divisible_by_9 n → 
  has_odd_units_and_thousands n → 
  has_even_tens_and_hundreds n → 
  3609 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_specific_digits_l1433_143379


namespace NUMINAMATH_CALUDE_parabola_focus_l1433_143349

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := y = 2 * x^2 + 4 * x + 5

-- Define the focus of a parabola
def is_focus (x y : ℝ) (f : ℝ × ℝ) : Prop :=
  f.1 = x ∧ f.2 = y

-- Theorem statement
theorem parabola_focus :
  ∃ (f : ℝ × ℝ), is_focus (-1) (25/8) f ∧
  ∀ (x y : ℝ), parabola_equation x y →
  is_focus x y f :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1433_143349


namespace NUMINAMATH_CALUDE_laylas_score_l1433_143383

theorem laylas_score (total : ℕ) (difference : ℕ) (laylas_score : ℕ) : 
  total = 112 → difference = 28 → laylas_score = 70 →
  ∃ (nahimas_score : ℕ), 
    nahimas_score + laylas_score = total ∧ 
    laylas_score = nahimas_score + difference :=
by
  sorry

end NUMINAMATH_CALUDE_laylas_score_l1433_143383


namespace NUMINAMATH_CALUDE_irrational_pair_sum_six_l1433_143369

theorem irrational_pair_sum_six : ∃ (x y : ℝ), Irrational x ∧ Irrational y ∧ x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_irrational_pair_sum_six_l1433_143369


namespace NUMINAMATH_CALUDE_sum_of_divisor_and_quotient_l1433_143304

/-- Given a valid vertical division, prove that the sum of the divisor and quotient is 723. -/
theorem sum_of_divisor_and_quotient : 
  ∀ (D Q : ℕ), 
  (D = 581) →  -- Divisor condition
  (Q = 142) →  -- Quotient condition
  (D + Q = 723) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisor_and_quotient_l1433_143304


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l1433_143328

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- Checks if the configuration is valid according to the problem conditions -/
def isValidConfiguration (config : ConeConfiguration) : Prop :=
  config.cone1 = config.cone2 ∧
  config.cone1.baseRadius = 3 ∧
  config.cone1.height = 8 ∧
  config.intersectionDistance = 3

/-- Theorem stating the maximum possible squared radius of the sphere -/
theorem max_sphere_radius_squared 
  (config : ConeConfiguration) 
  (h : isValidConfiguration config) : 
  (∀ c : ConeConfiguration, isValidConfiguration c → c.sphereRadius ^ 2 ≤ config.sphereRadius ^ 2) →
  config.sphereRadius ^ 2 = 225 / 73 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l1433_143328


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1433_143389

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) + Real.sqrt 3 / Real.sin (70 * π / 180) = 4 * Real.tan (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1433_143389


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_8_l1433_143323

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Calculates the units' digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

/-- Calculates the sum of factorials from 1! to n! -/
def sumFactorials (n : ℕ) : ℕ :=
  List.range n |>.map (λ i => factorial (i + 1)) |>.sum

/-- The main theorem: The units' digit of the sum of factorials from 1! to 8! is 3 -/
theorem units_digit_sum_factorials_8 :
  unitsDigit (sumFactorials 8) = 3 := by
  sorry


end NUMINAMATH_CALUDE_units_digit_sum_factorials_8_l1433_143323


namespace NUMINAMATH_CALUDE_sin_690_degrees_l1433_143319

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l1433_143319


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1433_143367

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * x + 1 > 4 * x - 6) → x ≤ 6 ∧ (3 * 6 + 1 > 4 * 6 - 6) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1433_143367


namespace NUMINAMATH_CALUDE_triangular_array_sum_recurrence_l1433_143370

def triangular_array_sum (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+1 => 2 * triangular_array_sum n + 2 * n

theorem triangular_array_sum_recurrence (n : ℕ) (h : n ≥ 2) :
  triangular_array_sum n = 2 * triangular_array_sum (n-1) + 2 * (n-1) :=
by sorry

#eval triangular_array_sum 20

end NUMINAMATH_CALUDE_triangular_array_sum_recurrence_l1433_143370


namespace NUMINAMATH_CALUDE_max_candies_bob_l1433_143374

theorem max_candies_bob (total : ℕ) (h1 : total = 30) : ∃ (bob : ℕ), bob ≤ 10 ∧ bob + 2 * bob = total := by
  sorry

end NUMINAMATH_CALUDE_max_candies_bob_l1433_143374


namespace NUMINAMATH_CALUDE_power_of_81_l1433_143350

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l1433_143350


namespace NUMINAMATH_CALUDE_cafe_working_days_l1433_143352

def days_in_period : ℕ := 13
def days_open_per_week : ℕ := 6
def first_day_is_monday : Prop := True

theorem cafe_working_days :
  let total_days := days_in_period
  let mondays := (total_days + 6) / 7
  total_days - mondays = 11 :=
sorry

end NUMINAMATH_CALUDE_cafe_working_days_l1433_143352


namespace NUMINAMATH_CALUDE_permutations_theorem_l1433_143312

def alphabet_size : ℕ := 26

def excluded_words : List String := ["dog", "god", "gum", "depth", "thing"]

def permutations_without_substrings (n : ℕ) (words : List String) : ℕ :=
  n.factorial - 3 * (n - 2).factorial + 3 * (n - 6).factorial + 2 * (n - 7).factorial - (n - 9).factorial

theorem permutations_theorem :
  permutations_without_substrings alphabet_size excluded_words =
  alphabet_size.factorial - 3 * (alphabet_size - 2).factorial + 3 * (alphabet_size - 6).factorial +
  2 * (alphabet_size - 7).factorial - (alphabet_size - 9).factorial :=
by sorry

end NUMINAMATH_CALUDE_permutations_theorem_l1433_143312


namespace NUMINAMATH_CALUDE_pentagon_area_l1433_143396

/-- The area of a pentagon formed by an equilateral triangle sharing a side with a square -/
theorem pentagon_area (s : ℝ) (h_perimeter : 5 * s = 20) : 
  s^2 + (s^2 * Real.sqrt 3) / 4 = 16 + 4 * Real.sqrt 3 := by
  sorry

#check pentagon_area

end NUMINAMATH_CALUDE_pentagon_area_l1433_143396


namespace NUMINAMATH_CALUDE_multiply_25_26_8_multiply_divide_340_40_17_sum_products_15_l1433_143346

-- Part 1
theorem multiply_25_26_8 : 25 * 26 * 8 = 5200 := by sorry

-- Part 2
theorem multiply_divide_340_40_17 : 340 * 40 / 17 = 800 := by sorry

-- Part 3
theorem sum_products_15 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := by sorry

end NUMINAMATH_CALUDE_multiply_25_26_8_multiply_divide_340_40_17_sum_products_15_l1433_143346


namespace NUMINAMATH_CALUDE_sufficient_condition_for_x_squared_minus_a_nonnegative_l1433_143392

theorem sufficient_condition_for_x_squared_minus_a_nonnegative 
  (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≥ 0) ↔ 
  (a ≤ -1 ∧ ∃ b : ℝ, b > -1 ∧ ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_x_squared_minus_a_nonnegative_l1433_143392


namespace NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l1433_143305

theorem sufficient_condition_for_f_less_than_one
  (a : ℝ) (ha : a > 1)
  (f : ℝ → ℝ) (hf : ∀ x, f x = a^(x^2 + 2*x)) :
  ∀ x, -1 < x ∧ x < 0 → f x < 1 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l1433_143305


namespace NUMINAMATH_CALUDE_line_mb_product_l1433_143377

theorem line_mb_product (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + b) →  -- Line equation
  b = -3 →                      -- y-intercept
  5 = m * 3 + b →               -- Line passes through (3, 5)
  m * b = -8 := by sorry

end NUMINAMATH_CALUDE_line_mb_product_l1433_143377


namespace NUMINAMATH_CALUDE_truncated_cube_edge_count_l1433_143397

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  -- The number of original cube edges
  original_edges : ℕ := 12
  -- The number of corners (vertices) in the original cube
  corners : ℕ := 8
  -- The number of edges in each pentagonal face created by truncation
  pentagonal_edges : ℕ := 5
  -- Condition that cutting planes do not intersect within the cube
  non_intersecting_cuts : Prop

/-- The number of edges in a truncated cube -/
def edge_count (tc : TruncatedCube) : ℕ :=
  tc.original_edges + (tc.corners * tc.pentagonal_edges) / 2

/-- Theorem stating that a truncated cube has 32 edges -/
theorem truncated_cube_edge_count (tc : TruncatedCube) :
  edge_count tc = 32 := by
  sorry

#check truncated_cube_edge_count

end NUMINAMATH_CALUDE_truncated_cube_edge_count_l1433_143397


namespace NUMINAMATH_CALUDE_total_flight_distance_l1433_143340

/-- The total distance to fly from Germany to Russia and then return to Spain,
    given the distances between Spain-Russia and Spain-Germany. -/
theorem total_flight_distance (spain_russia spain_germany : ℕ) 
  (h1 : spain_russia = 7019)
  (h2 : spain_germany = 1615) :
  spain_russia + (spain_russia - spain_germany) = 12423 :=
by sorry

end NUMINAMATH_CALUDE_total_flight_distance_l1433_143340


namespace NUMINAMATH_CALUDE_function_value_l1433_143330

/-- Given a function f(x) = x^α that passes through (2, √2/2), prove f(4) = 1/2 -/
theorem function_value (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) :
  f 2 = Real.sqrt 2 / 2 → f 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l1433_143330


namespace NUMINAMATH_CALUDE_skateboard_distance_l1433_143300

/-- The sum of an arithmetic sequence with first term 8, common difference 10, and 40 terms -/
theorem skateboard_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) : 
  a₁ = 8 → d = 10 → n = 40 → 
  (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = 8120 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l1433_143300


namespace NUMINAMATH_CALUDE_division_simplification_l1433_143333

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a^3 / (2 * a^2) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1433_143333


namespace NUMINAMATH_CALUDE_original_number_proof_l1433_143342

theorem original_number_proof (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Both parts are positive
  a ≤ b ∧          -- a is the smaller part
  a = 35 ∧         -- The smallest part is 35
  a / 7 = b / 9 →  -- The seventh part of the first equals the ninth part of the second
  a + b = 80       -- The original number is 80
  := by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1433_143342


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l1433_143336

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_intersection_A_B :
  (Set.univ \ (A ∩ B)) = {x : ℝ | x < 1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l1433_143336


namespace NUMINAMATH_CALUDE_smallest_b_for_inequality_l1433_143365

theorem smallest_b_for_inequality (b : ℕ) : (∀ k : ℕ, 27^k > 3^24 → k ≥ b) ↔ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_inequality_l1433_143365


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l1433_143341

/-- The fixed point of a family of parabolas -/
theorem fixed_point_of_parabola_family :
  ∀ t : ℝ, (4 : ℝ) * 3^2 + 2 * t * 3 - 3 * t = 36 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l1433_143341


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1433_143303

theorem smallest_fraction_between (p q : ℕ+) : 
  (4 : ℚ) / 11 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (3 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (4 : ℚ) / 11 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (3 : ℚ) / 8 → q ≤ q') →
  q - p = 12 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1433_143303


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1433_143347

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1433_143347


namespace NUMINAMATH_CALUDE_flag_distribution_l1433_143351

theorem flag_distribution (total_flags : ℕ) (blue_percent red_percent : ℚ) :
  total_flags % 2 = 0 ∧
  blue_percent = 60 / 100 ∧
  red_percent = 45 / 100 ∧
  blue_percent + red_percent > 1 →
  blue_percent + red_percent - 1 = 5 / 100 :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_l1433_143351


namespace NUMINAMATH_CALUDE_league_games_count_l1433_143308

theorem league_games_count (n : ℕ) (h : n = 14) : 
  (n * (n - 1)) / 2 = 91 := by
  sorry

#check league_games_count

end NUMINAMATH_CALUDE_league_games_count_l1433_143308


namespace NUMINAMATH_CALUDE_lawyer_upfront_payment_l1433_143388

theorem lawyer_upfront_payment
  (hourly_rate : ℕ)
  (court_time : ℕ)
  (prep_time_multiplier : ℕ)
  (total_payment : ℕ)
  (h1 : hourly_rate = 100)
  (h2 : court_time = 50)
  (h3 : prep_time_multiplier = 2)
  (h4 : total_payment = 8000) :
  let prep_time := prep_time_multiplier * court_time
  let total_hours := court_time + prep_time
  let total_fee := hourly_rate * total_hours
  let johns_share := total_payment / 2
  let upfront_payment := johns_share
  upfront_payment = 4000 := by
sorry

end NUMINAMATH_CALUDE_lawyer_upfront_payment_l1433_143388


namespace NUMINAMATH_CALUDE_order_of_values_l1433_143359

theorem order_of_values : 
  let a := Real.sin (80 * π / 180)
  let b := (1/2)⁻¹
  let c := Real.log 3 / Real.log (1/2)
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_values_l1433_143359


namespace NUMINAMATH_CALUDE_element_selection_theorem_l1433_143337

variable {α : Type*} [DecidableEq α]

def SubsetProperty (S : Finset α) (n k : ℕ) (S_i : ℕ → Finset α) : Prop :=
  (S.card = n) ∧ 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ k * n → S_i i ⊆ S ∧ (S_i i).card = 2) ∧
  (∀ e ∈ S, (Finset.filter (fun i => e ∈ S_i i) (Finset.range (k * n))).card = 2 * k)

theorem element_selection_theorem (S : Finset α) (n k : ℕ) (S_i : ℕ → Finset α) 
  (h : SubsetProperty S n k S_i) :
  ∃ f : ℕ → α, 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ k * n → f i ∈ S_i i) ∧ 
    (∀ e ∈ S, (Finset.filter (fun i => f i = e) (Finset.range (k * n))).card = k) :=
sorry

end NUMINAMATH_CALUDE_element_selection_theorem_l1433_143337


namespace NUMINAMATH_CALUDE_least_sum_exponents_3125_l1433_143301

theorem least_sum_exponents_3125 : 
  let n := 3125
  let is_valid_representation (rep : List ℕ) := 
    (rep.map (λ i => 2^i)).sum = n ∧ rep.Nodup
  ∃ (rep : List ℕ), is_valid_representation rep ∧
    ∀ (other_rep : List ℕ), is_valid_representation other_rep → 
      rep.sum ≤ other_rep.sum :=
by sorry

end NUMINAMATH_CALUDE_least_sum_exponents_3125_l1433_143301


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1433_143310

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1433_143310


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1433_143343

def numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

theorem arithmetic_mean_problem (x : ℕ) :
  (numbers.sum + x) / (numbers.length + 1) = 12 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1433_143343


namespace NUMINAMATH_CALUDE_even_divisors_of_factorial_8_l1433_143372

/-- The factorial of 8 -/
def factorial_8 : ℕ := 40320

/-- The prime factorization of 8! -/
axiom factorial_8_factorization : factorial_8 = 2^7 * 3^2 * 5 * 7

/-- A function that counts the number of even divisors of a natural number -/
def count_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating that 8! has 84 even divisors -/
theorem even_divisors_of_factorial_8 :
  count_even_divisors factorial_8 = 84 := by sorry

end NUMINAMATH_CALUDE_even_divisors_of_factorial_8_l1433_143372


namespace NUMINAMATH_CALUDE_estimate_negative_sqrt_17_l1433_143335

theorem estimate_negative_sqrt_17 : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end NUMINAMATH_CALUDE_estimate_negative_sqrt_17_l1433_143335


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l1433_143361

theorem sum_of_reciprocals_of_quadratic_roots :
  let a := 1
  let b := -17
  let c := 8
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (1 / r₁ + 1 / r₂) = 17 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l1433_143361


namespace NUMINAMATH_CALUDE_fence_length_l1433_143344

/-- The total length of a fence for a land shaped like a rectangle combined with a semicircle,
    given the dimensions and an opening. -/
theorem fence_length
  (rect_length : ℝ)
  (rect_width : ℝ)
  (semicircle_radius : ℝ)
  (opening_length : ℝ)
  (h1 : rect_length = 20)
  (h2 : rect_width = 14)
  (h3 : semicircle_radius = 7)
  (h4 : opening_length = 3)
  : rect_length * 2 + rect_width + π * semicircle_radius + rect_width - opening_length = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_fence_length_l1433_143344


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1433_143355

theorem cubic_equation_roots (P : ℤ) : 
  (∃ x y z : ℤ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^3 - 10*x^2 + P*x - 30 = 0 ∧
    y^3 - 10*y^2 + P*y - 30 = 0 ∧
    z^3 - 10*z^2 + P*z - 30 = 0 ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  P = 31 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1433_143355


namespace NUMINAMATH_CALUDE_half_level_associated_point_of_A_l1433_143327

/-- Given a point P(x,y) in the Cartesian plane, its a-level associated point Q has coordinates (ax+y, x+ay) where a is a constant. -/
def associated_point (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (a * x + y, x + a * y)

/-- The coordinates of point A -/
def A : ℝ × ℝ := (2, 6)

/-- The theorem states that the 1/2-level associated point of A(2,6) is B(7,5) -/
theorem half_level_associated_point_of_A :
  associated_point (1/2) A = (7, 5) := by
sorry


end NUMINAMATH_CALUDE_half_level_associated_point_of_A_l1433_143327


namespace NUMINAMATH_CALUDE_chess_draw_probability_l1433_143362

theorem chess_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.9) :
  prob_A_not_lose - prob_A_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l1433_143362


namespace NUMINAMATH_CALUDE_pencil_count_l1433_143307

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := 115

/-- The number of pencils added to the drawer -/
def added_pencils : ℕ := 100

/-- The total number of pencils after addition -/
def total_pencils : ℕ := 215

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1433_143307


namespace NUMINAMATH_CALUDE_parabola_r_value_l1433_143326

/-- A parabola with equation x = py^2 + qy + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (para : Parabola) (y : ℝ) : ℝ :=
  para.p * y^2 + para.q * y + para.r

theorem parabola_r_value (para : Parabola) :
  para.x_coord 1 = 4 →  -- vertex at (4,1)
  para.x_coord 0 = 2 →  -- passes through (2,0)
  para.r = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_r_value_l1433_143326


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1433_143371

/-- The y-intercept of the line 2x - 3y = 6 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1433_143371


namespace NUMINAMATH_CALUDE_floor_inequality_and_factorial_divisibility_l1433_143357

theorem floor_inequality_and_factorial_divisibility 
  (x y : ℝ) (m n : ℕ+) 
  (hx : x ≥ 0) (hy : y ≥ 0) : 
  (⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋) ∧ 
  (∃ k : ℕ, k * (m.val.factorial * n.val.factorial * (3 * m.val + n.val).factorial * (3 * n.val + m.val).factorial) = 
   (5 * m.val).factorial * (5 * n.val).factorial) :=
sorry

end NUMINAMATH_CALUDE_floor_inequality_and_factorial_divisibility_l1433_143357


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l1433_143356

theorem complex_magnitude_equality (t : ℝ) (h : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 5 → t = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l1433_143356


namespace NUMINAMATH_CALUDE_range_of_a_l1433_143395

-- Define the conditions
def sufficient_condition (x : ℝ) : Prop := -2 < x ∧ x < 4

def necessary_condition (x a : ℝ) : Prop := (x + 2) * (x - a) < 0

-- Define the theorem
theorem range_of_a : 
  (∀ x a : ℝ, sufficient_condition x → necessary_condition x a) ∧ 
  (∃ x a : ℝ, ¬sufficient_condition x ∧ necessary_condition x a) → 
  ∀ a : ℝ, (a ∈ Set.Ioi 4) ↔ (∃ x : ℝ, necessary_condition x a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1433_143395


namespace NUMINAMATH_CALUDE_expression_equivalence_l1433_143345

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) + ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) = 2 * x^2 * y^2 + 2 / (x^2 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l1433_143345


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1433_143384

theorem polygon_interior_angles (n : ℕ) : (n - 2) * 180 = 360 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1433_143384


namespace NUMINAMATH_CALUDE_multiplicative_inverse_300_mod_2399_l1433_143339

theorem multiplicative_inverse_300_mod_2399 :
  (39 : ℤ)^2 + 80^2 = 89^2 →
  (300 * 1832) % 2399 = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_300_mod_2399_l1433_143339


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_expansion_l1433_143334

theorem binomial_coefficient_third_term_expansion (x : ℤ) :
  Nat.choose 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_expansion_l1433_143334


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l1433_143376

theorem no_infinite_prime_sequence : 
  ¬ ∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧ (∀ n, p (n + 1) = 2 * p n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l1433_143376


namespace NUMINAMATH_CALUDE_fourth_derivative_of_f_l1433_143318

open Real

noncomputable def f (x : ℝ) : ℝ := exp (1 - 2*x) * sin (2 + 3*x)

theorem fourth_derivative_of_f (x : ℝ) :
  (deriv^[4] f) x = -119 * exp (1 - 2*x) * sin (2 + 3*x) + 120 * exp (1 - 2*x) * cos (2 + 3*x) :=
by sorry

end NUMINAMATH_CALUDE_fourth_derivative_of_f_l1433_143318


namespace NUMINAMATH_CALUDE_polynomial_B_value_l1433_143358

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 9*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 81

theorem polynomial_B_value (A B C D : ℤ) :
  (∀ r : ℤ, polynomial r A B C D = 0 → r > 0) →
  B = -46 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l1433_143358


namespace NUMINAMATH_CALUDE_first_discount_percentage_l1433_143398

theorem first_discount_percentage (original_price final_price second_discount : ℝ) 
  (h1 : original_price = 33.78)
  (h2 : final_price = 19)
  (h3 : second_discount = 0.25)
  : ∃ (first_discount : ℝ), 
    first_discount = 0.25 ∧ 
    final_price = original_price * (1 - first_discount) * (1 - second_discount) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l1433_143398


namespace NUMINAMATH_CALUDE_power_function_problem_l1433_143348

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- Define the problem statement
theorem power_function_problem (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 3 = Real.sqrt 3) : 
  f 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_problem_l1433_143348


namespace NUMINAMATH_CALUDE_stream_speed_equation_l1433_143380

/-- The speed of the stream for a boat trip -/
theorem stream_speed_equation (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 9)
  (h2 : distance = 210)
  (h3 : total_time = 84) :
  ∃ x : ℝ, x^2 = 39 ∧ 
    (distance / (boat_speed + x) + distance / (boat_speed - x) = total_time) := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_equation_l1433_143380


namespace NUMINAMATH_CALUDE_problem_1_l1433_143360

theorem problem_1 : Real.sqrt 18 - 4 * Real.sqrt (1/2) + Real.sqrt 24 / Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1433_143360


namespace NUMINAMATH_CALUDE_expression_value_at_three_l1433_143382

theorem expression_value_at_three : 
  let x : ℝ := 3
  x + x * (x^3 - x) = 75 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l1433_143382


namespace NUMINAMATH_CALUDE_inequality_implies_range_l1433_143313

theorem inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, a * Real.sin x - Real.cos x ^ 2 ≤ 3) →
  -3 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l1433_143313


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_five_sixths_l1433_143386

theorem sum_of_fractions_equals_five_sixths :
  let sum : ℚ := (1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6))
  sum = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_five_sixths_l1433_143386


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1433_143385

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1433_143385


namespace NUMINAMATH_CALUDE_loom_weaving_rate_l1433_143378

/-- The rate at which an industrial loom weaves cloth, given the time and length of cloth woven. -/
theorem loom_weaving_rate (time : ℝ) (length : ℝ) (h : time = 195.3125 ∧ length = 25) :
  length / time = 0.128 := by
  sorry

end NUMINAMATH_CALUDE_loom_weaving_rate_l1433_143378


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1433_143322

theorem max_value_of_expression (x : ℝ) (h : x > 0) :
  1 - x - 16 / x ≤ -7 ∧ ∃ y > 0, 1 - y - 16 / y = -7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1433_143322


namespace NUMINAMATH_CALUDE_concert_ticket_purchase_daria_concert_money_l1433_143368

/-- Calculates the additional money needed to purchase concert tickets --/
theorem concert_ticket_purchase (num_tickets : ℕ) (original_price : ℚ) 
  (discount_percent : ℚ) (gift_card : ℚ) (current_money : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_percent / 100)
  let total_cost := num_tickets * discounted_price
  let after_gift_card := total_cost - gift_card
  after_gift_card - current_money

/-- Proves that Daria needs to earn $85 more for the concert tickets --/
theorem daria_concert_money : 
  concert_ticket_purchase 4 90 10 50 189 = 85 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_purchase_daria_concert_money_l1433_143368


namespace NUMINAMATH_CALUDE_inequality_range_l1433_143317

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 1 - 2 * x^2) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1433_143317


namespace NUMINAMATH_CALUDE_remainder_1021_pow_1022_mod_1023_l1433_143366

theorem remainder_1021_pow_1022_mod_1023 :
  (1021 : ℤ) ^ 1022 ≡ 16 [ZMOD 1023] := by
  sorry

end NUMINAMATH_CALUDE_remainder_1021_pow_1022_mod_1023_l1433_143366


namespace NUMINAMATH_CALUDE_jose_land_division_l1433_143375

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

end NUMINAMATH_CALUDE_jose_land_division_l1433_143375


namespace NUMINAMATH_CALUDE_pennsylvania_quarters_l1433_143332

theorem pennsylvania_quarters (total : ℕ) (state_fraction : ℚ) (penn_fraction : ℚ) : 
  total = 35 → 
  state_fraction = 2 / 5 → 
  penn_fraction = 1 / 2 → 
  (total : ℚ) * state_fraction * penn_fraction = 7 := by
sorry

end NUMINAMATH_CALUDE_pennsylvania_quarters_l1433_143332


namespace NUMINAMATH_CALUDE_prob_product_144_three_dice_l1433_143364

/-- A function representing the roll of a standard die -/
def dieRoll : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of a specific outcome when rolling three dice -/
def probSpecificOutcome : ℚ := (1 / 6) * (1 / 6) * (1 / 6)

/-- The number of ways to get a product of 144 with three dice -/
def waysToGet144 : ℕ := 3

theorem prob_product_144_three_dice :
  (waysToGet144 : ℚ) * probSpecificOutcome = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_prob_product_144_three_dice_l1433_143364


namespace NUMINAMATH_CALUDE_expression_value_l1433_143353

theorem expression_value (x : ℝ) (hx : x^2 - x - 1 = 0) :
  (2 / (x + 1) - 1 / x) / ((x^2 - x) / (x^2 + 2*x + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1433_143353


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1433_143338

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (g 1 = 2) ∧ 
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y)

/-- The main theorem stating that the function g satisfying the functional equation
    is equal to 2(4^x - 3^x) for all real x -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1433_143338


namespace NUMINAMATH_CALUDE_jane_shorter_than_sarah_l1433_143302

-- Define the lengths of the sticks and the covered portion
def pat_stick_length : ℕ := 30
def pat_covered_length : ℕ := 7
def jane_stick_length : ℕ := 22

-- Define Sarah's stick length based on Pat's uncovered portion
def sarah_stick_length : ℕ := 2 * (pat_stick_length - pat_covered_length)

-- State the theorem
theorem jane_shorter_than_sarah : sarah_stick_length - jane_stick_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_jane_shorter_than_sarah_l1433_143302


namespace NUMINAMATH_CALUDE_trees_in_yard_l1433_143354

/-- Calculates the number of trees in a yard given the yard length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

theorem trees_in_yard (yard_length : ℕ) (tree_spacing : ℕ) 
  (h1 : yard_length = 250)
  (h2 : tree_spacing = 5) :
  num_trees yard_length tree_spacing = 51 := by
  sorry

#eval num_trees 250 5

end NUMINAMATH_CALUDE_trees_in_yard_l1433_143354


namespace NUMINAMATH_CALUDE_interest_difference_is_520_l1433_143329

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem stating that the difference between the principal and 
    the simple interest is $520 under the given conditions -/
theorem interest_difference_is_520 :
  let principal : ℝ := 1000
  let rate : ℝ := 0.06
  let time : ℝ := 8
  principal - simple_interest principal rate time = 520 := by
sorry


end NUMINAMATH_CALUDE_interest_difference_is_520_l1433_143329


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1433_143331

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 36 = 0) → m = 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1433_143331


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1433_143390

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x*y*z) ≥ 216 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x*y*z) = 216 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1433_143390


namespace NUMINAMATH_CALUDE_symmetric_complex_sum_l1433_143316

theorem symmetric_complex_sum (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  let w : ℂ := Complex.I * (Complex.I - 2)
  (z.re = w.re ∧ z.im = -w.im) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_sum_l1433_143316


namespace NUMINAMATH_CALUDE_yard_length_is_360_l1433_143363

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The yard is 360 meters long -/
theorem yard_length_is_360 :
  yard_length 31 12 = 360 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_is_360_l1433_143363


namespace NUMINAMATH_CALUDE_choose_two_from_seven_eq_twentyone_l1433_143324

/-- The number of ways to choose 2 people from 7 -/
def choose_two_from_seven : ℕ := Nat.choose 7 2

/-- Theorem stating that choosing 2 from 7 results in 21 possibilities -/
theorem choose_two_from_seven_eq_twentyone : choose_two_from_seven = 21 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_seven_eq_twentyone_l1433_143324


namespace NUMINAMATH_CALUDE_min_value_theorem_l1433_143387

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 1) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z > 0 → z + y = 1 →
    2 / (z + 3 * y) + 1 / (z - y) ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1433_143387


namespace NUMINAMATH_CALUDE_school_boys_count_l1433_143381

/-- Represents the number of boys in the school -/
def num_boys : ℕ := 410

/-- Represents the initial number of girls in the school -/
def initial_girls : ℕ := 632

/-- Represents the number of additional girls that joined the school -/
def additional_girls : ℕ := 465

/-- Represents the difference between girls and boys after the addition -/
def girl_boy_difference : ℕ := 687

/-- Proves that the number of boys in the school is 410 -/
theorem school_boys_count :
  initial_girls + additional_girls = num_boys + girl_boy_difference := by
  sorry

#check school_boys_count

end NUMINAMATH_CALUDE_school_boys_count_l1433_143381


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1433_143373

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 110 → percentage = 50 → result = initial * (1 + percentage / 100) → result = 165 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1433_143373
