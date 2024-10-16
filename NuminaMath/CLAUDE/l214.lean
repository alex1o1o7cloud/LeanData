import Mathlib

namespace NUMINAMATH_CALUDE_vector_sum_diff_magnitude_bounds_l214_21455

theorem vector_sum_diff_magnitude_bounds (a b : ℝ × ℝ) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) : 
  (∃ x : ℝ × ℝ, ‖x‖ = 1 ∧ ∃ y : ℝ × ℝ, ‖y‖ = 2 ∧ ‖x + y‖ + ‖x - y‖ = 4) ∧
  (∃ x : ℝ × ℝ, ‖x‖ = 1 ∧ ∃ y : ℝ × ℝ, ‖y‖ = 2 ∧ ‖x + y‖ + ‖x - y‖ = 2 * Real.sqrt 5) ∧
  (∀ x y : ℝ × ℝ, ‖x‖ = 1 → ‖y‖ = 2 → 4 ≤ ‖x + y‖ + ‖x - y‖ ∧ ‖x + y‖ + ‖x - y‖ ≤ 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_diff_magnitude_bounds_l214_21455


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_1985_l214_21482

def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sumOfDistinctPowersOfTwo (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.map (λ k => 2^k)).sum = n ∧ powers.Nodup

def exponentSum (powers : List ℕ) : ℕ := powers.sum

theorem least_exponent_sum_for_1985 :
  ∃ (powers : List ℕ),
    sumOfDistinctPowersOfTwo 1985 powers ∧
    ∀ (other_powers : List ℕ),
      sumOfDistinctPowersOfTwo 1985 other_powers →
      exponentSum powers ≤ exponentSum other_powers ∧
      exponentSum powers = 40 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_1985_l214_21482


namespace NUMINAMATH_CALUDE_grace_age_l214_21432

theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ) :
  mother_age = 80 →
  grandmother_age = 2 * mother_age →
  grace_age = (3 * grandmother_age) / 8 →
  grace_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_grace_age_l214_21432


namespace NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l214_21459

/-- Prove that the wolf does not catch the hare given the initial conditions -/
theorem wolf_does_not_catch_hare (initial_distance : ℝ) (distance_to_refuge : ℝ) 
  (wolf_speed : ℝ) (hare_speed : ℝ) 
  (h1 : initial_distance = 30) 
  (h2 : distance_to_refuge = 250) 
  (h3 : wolf_speed = 600) 
  (h4 : hare_speed = 550) : 
  (distance_to_refuge / hare_speed) < ((initial_distance + distance_to_refuge) / wolf_speed) :=
by
  sorry

#check wolf_does_not_catch_hare

end NUMINAMATH_CALUDE_wolf_does_not_catch_hare_l214_21459


namespace NUMINAMATH_CALUDE_abby_damon_weight_l214_21411

theorem abby_damon_weight 
  (a b c d : ℝ)  -- Weights of Abby, Bart, Cindy, and Damon
  (h1 : a + b = 280)  -- Abby and Bart's combined weight
  (h2 : b + c = 255)  -- Bart and Cindy's combined weight
  (h3 : c + d = 290)  -- Cindy and Damon's combined weight
  : a + d = 315 := by
  sorry

end NUMINAMATH_CALUDE_abby_damon_weight_l214_21411


namespace NUMINAMATH_CALUDE_range_of_a_l214_21436

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + 4 < 0 ↔ a - 1 < x ∧ x < a + 1) → 
  (2 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l214_21436


namespace NUMINAMATH_CALUDE_temperature_theorem_l214_21474

def temperature_problem (temp_ny temp_miami temp_sd temp_phoenix : ℝ) : Prop :=
  temp_ny = 80 ∧
  temp_miami = temp_ny + 10 ∧
  temp_sd = temp_miami + 25 ∧
  temp_phoenix = temp_sd * 1.15 ∧
  (temp_ny + temp_miami + temp_sd + temp_phoenix) / 4 = 104.3125

theorem temperature_theorem :
  ∃ temp_ny temp_miami temp_sd temp_phoenix : ℝ,
    temperature_problem temp_ny temp_miami temp_sd temp_phoenix := by
  sorry

end NUMINAMATH_CALUDE_temperature_theorem_l214_21474


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_l214_21483

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2 + 2*x, 5],
    ![4 - 2*x, 9]]

theorem matrix_not_invertible_iff (x : ℚ) :
  ¬(Matrix.det (matrix x) ≠ 0) ↔ x = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_l214_21483


namespace NUMINAMATH_CALUDE_max_cube_volume_in_tetrahedron_l214_21447

/-- Regular tetrahedron with edge length 2 -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq : edge_length = 2

/-- Cube placed inside the tetrahedron -/
structure InsideCube where
  side_length : ℝ
  bottom_face_parallel : Prop
  top_vertices_touch : Prop

/-- The maximum volume of the cube inside the tetrahedron -/
def max_cube_volume (t : RegularTetrahedron) (c : InsideCube) : ℝ :=
  c.side_length ^ 3

/-- Theorem stating the maximum volume of the cube -/
theorem max_cube_volume_in_tetrahedron (t : RegularTetrahedron) (c : InsideCube) :
  max_cube_volume t c = 8 * Real.sqrt 3 / 243 :=
sorry

end NUMINAMATH_CALUDE_max_cube_volume_in_tetrahedron_l214_21447


namespace NUMINAMATH_CALUDE_set_A_definition_l214_21441

def U : Set ℝ := {x | x > 1}

theorem set_A_definition (A : Set ℝ) (h1 : A ⊆ U) (h2 : (U \ A) = {x | x > 9}) : 
  A = {x | 1 < x ∧ x ≤ 9} := by
  sorry

end NUMINAMATH_CALUDE_set_A_definition_l214_21441


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_81_l214_21416

theorem fraction_of_powers_equals_81 : (75000 ^ 4) / (25000 ^ 4) = 81 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_81_l214_21416


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l214_21410

theorem largest_power_dividing_factorial_squared (p : ℕ) (hp : Prime p) :
  (∃ k : ℕ, (p ^ k : ℕ) ∣ (p^2).factorial ∧ 
   ∀ m : ℕ, (p ^ m : ℕ) ∣ (p^2).factorial → m ≤ k) ↔ 
  (∃ k : ℕ, k = p + 1 ∧ (p ^ k : ℕ) ∣ (p^2).factorial ∧ 
   ∀ m : ℕ, (p ^ m : ℕ) ∣ (p^2).factorial → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_squared_l214_21410


namespace NUMINAMATH_CALUDE_pens_distribution_eq_six_l214_21420

/-- The number of ways to distribute n identical items among k distinct groups,
    where each group gets at least m items. -/
def distribute_with_minimum (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The number of ways to distribute 8 pens among 3 friends,
    where each friend gets at least 2 pens. -/
def pens_distribution : ℕ :=
  distribute_with_minimum 8 3 2

theorem pens_distribution_eq_six :
  pens_distribution = 6 := by
  sorry

end NUMINAMATH_CALUDE_pens_distribution_eq_six_l214_21420


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_72_l214_21494

theorem sqrt_sum_equals_sqrt_72 (k : ℕ+) :
  Real.sqrt 2 + Real.sqrt 8 + Real.sqrt 18 = Real.sqrt k → k = 72 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_72_l214_21494


namespace NUMINAMATH_CALUDE_paint_mixture_weight_l214_21478

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue_percent : ℝ
  red_percent : ℝ
  yellow_percent : ℝ
  weight : ℝ

/-- The problem setup for the paint mixture calculation -/
def paint_problem : Prop :=
  ∃ (sky_blue green brown : PaintMixture),
    -- Sky blue paint composition
    sky_blue.blue_percent = 0.1 ∧
    sky_blue.red_percent = 0.9 ∧
    sky_blue.yellow_percent = 0 ∧
    -- Green paint composition
    green.blue_percent = 0.7 ∧
    green.red_percent = 0 ∧
    green.yellow_percent = 0.3 ∧
    -- Brown paint composition
    brown.blue_percent = 0.4 ∧
    -- Red pigment weight in brown paint
    brown.red_percent * brown.weight = 4.5 ∧
    -- Total weight of brown paint
    brown.weight = sky_blue.weight + green.weight ∧
    -- Blue pigment balance
    sky_blue.blue_percent * sky_blue.weight + green.blue_percent * green.weight = 
      brown.blue_percent * brown.weight ∧
    -- Total weight of brown paint is 10 grams
    brown.weight = 10

/-- The main theorem stating that the paint problem implies a 10-gram brown paint -/
theorem paint_mixture_weight : paint_problem → ∃ (brown : PaintMixture), brown.weight = 10 := by
  sorry


end NUMINAMATH_CALUDE_paint_mixture_weight_l214_21478


namespace NUMINAMATH_CALUDE_unique_solution_for_floor_equation_l214_21491

theorem unique_solution_for_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_floor_equation_l214_21491


namespace NUMINAMATH_CALUDE_mary_clothing_expenditure_l214_21486

/-- The amount Mary spent on a shirt -/
def shirt_cost : ℚ := 13.04

/-- The amount Mary spent on a jacket -/
def jacket_cost : ℚ := 12.27

/-- The total amount Mary spent on clothing -/
def total_cost : ℚ := shirt_cost + jacket_cost

theorem mary_clothing_expenditure :
  total_cost = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_mary_clothing_expenditure_l214_21486


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l214_21461

/-- The probability of getting exactly k heads in n tosses of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The main theorem -/
theorem unfair_coin_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  binomial_probability 7 4 p = 210 / 1024 → p = 4 / 7 := by
  sorry

#check unfair_coin_probability

end NUMINAMATH_CALUDE_unfair_coin_probability_l214_21461


namespace NUMINAMATH_CALUDE_pancake_cooking_theorem_l214_21406

/-- Represents the minimum time needed to cook a given number of pancakes -/
def min_cooking_time (num_pancakes : ℕ) : ℕ :=
  sorry

/-- The pancake cooking theorem -/
theorem pancake_cooking_theorem :
  let pan_capacity : ℕ := 2
  let cooking_time_per_pancake : ℕ := 2
  let num_pancakes : ℕ := 3
  min_cooking_time num_pancakes = 3 :=
sorry

end NUMINAMATH_CALUDE_pancake_cooking_theorem_l214_21406


namespace NUMINAMATH_CALUDE_train_speed_proof_l214_21443

/-- Proves that a train with given crossing times has a specific speed -/
theorem train_speed_proof (platform_length : ℝ) (platform_cross_time : ℝ) (man_cross_time : ℝ) :
  platform_length = 280 →
  platform_cross_time = 32 →
  man_cross_time = 18 →
  ∃ (train_speed : ℝ), train_speed = 72 ∧ 
    (train_speed * man_cross_time = train_speed * platform_cross_time - platform_length) :=
by
  sorry

#check train_speed_proof

end NUMINAMATH_CALUDE_train_speed_proof_l214_21443


namespace NUMINAMATH_CALUDE_x_equals_negative_x_is_valid_l214_21456

/-- An assignment statement is valid if it assigns a value to a variable -/
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (val : String), stmt = var ++ " = " ++ val

/-- The statement "x = -x" -/
def statement : String := "x = -x"

/-- Theorem: The statement "x = -x" is a valid assignment statement -/
theorem x_equals_negative_x_is_valid : is_valid_assignment statement := by
  sorry

end NUMINAMATH_CALUDE_x_equals_negative_x_is_valid_l214_21456


namespace NUMINAMATH_CALUDE_modular_inverse_34_mod_35_l214_21463

theorem modular_inverse_34_mod_35 : ∃ x : ℕ, x ≤ 34 ∧ (34 * x) % 35 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_34_mod_35_l214_21463


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l214_21475

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ (n : ℕ), 
  n > 0 → 
  (360 : ℝ) / n = 18 → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_18_deg_has_20_sides_l214_21475


namespace NUMINAMATH_CALUDE_binary_equals_21_l214_21430

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of the number in question -/
def binary_number : List Bool := [true, false, true, false, true]

/-- Theorem stating that the given binary number equals 21 in decimal -/
theorem binary_equals_21 : binary_to_decimal binary_number = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_equals_21_l214_21430


namespace NUMINAMATH_CALUDE_perfect_squares_between_powers_of_three_l214_21496

theorem perfect_squares_between_powers_of_three : 
  (Finset.range (Nat.succ (Nat.sqrt (3^10 + 3))) 
    |>.filter (λ n => n^2 ≥ 3^5 + 3 ∧ n^2 ≤ 3^10 + 3)).card = 228 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_powers_of_three_l214_21496


namespace NUMINAMATH_CALUDE_monkey_climb_time_l214_21453

/-- Represents the climbing process of a monkey on a tree. -/
structure MonkeyClimb where
  treeHeight : ℕ  -- Height of the tree in feet
  hopDistance : ℕ  -- Distance the monkey hops up each hour
  slipDistance : ℕ  -- Distance the monkey slips back each hour

/-- Calculates the time taken for the monkey to reach the top of the tree. -/
def timeToReachTop (climb : MonkeyClimb) : ℕ :=
  let netClimbPerHour := climb.hopDistance - climb.slipDistance
  let timeToReachNearTop := (climb.treeHeight - 1) / netClimbPerHour
  timeToReachNearTop + 1

/-- Theorem stating that for the given conditions, the monkey takes 19 hours to reach the top. -/
theorem monkey_climb_time :
  let climb : MonkeyClimb := { treeHeight := 19, hopDistance := 3, slipDistance := 2 }
  timeToReachTop climb = 19 := by
  sorry


end NUMINAMATH_CALUDE_monkey_climb_time_l214_21453


namespace NUMINAMATH_CALUDE_complex_equation_solution_l214_21465

theorem complex_equation_solution (z : ℂ) :
  (-3 + 4 * Complex.I) * z = 25 * Complex.I → z = 4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l214_21465


namespace NUMINAMATH_CALUDE_balloon_ratio_l214_21498

theorem balloon_ratio (mary_balloons nancy_balloons : ℕ) 
  (h1 : mary_balloons = 28) (h2 : nancy_balloons = 7) :
  mary_balloons / nancy_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_balloon_ratio_l214_21498


namespace NUMINAMATH_CALUDE_work_completion_time_l214_21412

/-- The time it takes for worker C to complete the work alone -/
def time_C : ℕ := 36

/-- The time it takes for workers A, B, and C to complete the work together -/
def time_ABC : ℕ := 4

/-- The time it takes for worker A to complete the work alone -/
def time_A : ℕ := 6

/-- The time it takes for worker B to complete the work alone -/
def time_B : ℕ := 18

theorem work_completion_time :
  (1 : ℚ) / time_ABC = (1 : ℚ) / time_A + (1 : ℚ) / time_B + (1 : ℚ) / time_C :=
by sorry


end NUMINAMATH_CALUDE_work_completion_time_l214_21412


namespace NUMINAMATH_CALUDE_sum_of_squares_l214_21457

variables {x y z w a b c d : ℝ}

theorem sum_of_squares (h1 : x * y = a) (h2 : x * z = b) (h3 : y * z = c) (h4 : x * w = d)
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : d ≠ 0) :
  x^2 + y^2 + z^2 + w^2 = (a * b + b * d + d * a)^2 / (a * b * d) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l214_21457


namespace NUMINAMATH_CALUDE_inequality_solution_set_l214_21468

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 5*a)*(x + a) > 0} = {x : ℝ | x < 5*a ∨ x > -a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l214_21468


namespace NUMINAMATH_CALUDE_square_root_of_1024_l214_21431

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l214_21431


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l214_21437

theorem smallest_four_digit_mod_8_5 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 5) → m ≥ n) ∧
  (n = 1005) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l214_21437


namespace NUMINAMATH_CALUDE_specific_test_result_l214_21439

/-- Represents a test with a given number of questions, points for correct and incorrect answers --/
structure Test where
  total_questions : ℕ
  points_correct : ℤ
  points_incorrect : ℤ

/-- Represents the result of a test --/
structure TestResult where
  test : Test
  correct_answers : ℕ
  final_score : ℤ

/-- Theorem stating that for a specific test configuration, 
    if the final score is 0, then the number of correct answers is 10 --/
theorem specific_test_result (t : Test) (r : TestResult) : 
  t.total_questions = 26 ∧ 
  t.points_correct = 8 ∧ 
  t.points_incorrect = -5 ∧ 
  r.test = t ∧
  r.correct_answers + (t.total_questions - r.correct_answers) = t.total_questions ∧
  r.final_score = r.correct_answers * t.points_correct + (t.total_questions - r.correct_answers) * t.points_incorrect ∧
  r.final_score = 0 →
  r.correct_answers = 10 := by
sorry

end NUMINAMATH_CALUDE_specific_test_result_l214_21439


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l214_21488

/-- Given a circle divided into 16 equal arcs, this theorem proves that 
    the sum of an inscribed angle over 3 arcs and an inscribed angle over 5 arcs is 90°. -/
theorem inscribed_angles_sum (circle : Real) (arcs : ℕ) (x y : Real) :
  arcs = 16 →
  x = 3 * (360 / (2 * arcs)) →
  y = 5 * (360 / (2 * arcs)) →
  x + y = 90 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l214_21488


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l214_21404

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (4, -3) and (-8, 7) is equal to -4. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -3
  let x2 : ℝ := -8
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l214_21404


namespace NUMINAMATH_CALUDE_rings_per_game_l214_21469

theorem rings_per_game (total_rings : ℕ) (num_games : ℕ) (rings_per_game : ℕ) 
  (h1 : total_rings = 48) 
  (h2 : num_games = 8) 
  (h3 : total_rings = num_games * rings_per_game) : 
  rings_per_game = 6 := by
  sorry

end NUMINAMATH_CALUDE_rings_per_game_l214_21469


namespace NUMINAMATH_CALUDE_age_difference_proof_l214_21487

theorem age_difference_proof (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 22 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l214_21487


namespace NUMINAMATH_CALUDE_fraction_simplification_l214_21402

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) :
  (2 * a) / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l214_21402


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l214_21415

theorem polygon_interior_exterior_angles_equality (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 360) → 
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angles_equality_l214_21415


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l214_21428

theorem perfect_cube_units_digits :
  ∀ d : Fin 10, ∃ n : ℤ, (n^3) % 10 = d.val :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l214_21428


namespace NUMINAMATH_CALUDE_abs_g_one_equals_31_l214_21471

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that the absolute value of g at specific points is 24 -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  |g (-1)| = 24 ∧ |g 0| = 24 ∧ |g 2| = 24 ∧ |g 4| = 24 ∧ |g 5| = 24 ∧ |g 8| = 24

/-- The main theorem -/
theorem abs_g_one_equals_31 (g : ThirdDegreePolynomial) 
  (h : SatisfiesCondition g) : |g 1| = 31 := by
  sorry

end NUMINAMATH_CALUDE_abs_g_one_equals_31_l214_21471


namespace NUMINAMATH_CALUDE_planks_per_tree_value_l214_21433

/-- The number of planks John can make from each tree -/
def planks_per_tree : ℕ := sorry

/-- The number of trees John chops down -/
def num_trees : ℕ := 30

/-- The number of planks needed to make one table -/
def planks_per_table : ℕ := 15

/-- The selling price of one table in dollars -/
def table_price : ℕ := 300

/-- The total labor cost in dollars -/
def labor_cost : ℕ := 3000

/-- The total profit in dollars -/
def total_profit : ℕ := 12000

/-- Theorem stating the number of planks John can make from each tree -/
theorem planks_per_tree_value : planks_per_tree = 25 := by sorry

end NUMINAMATH_CALUDE_planks_per_tree_value_l214_21433


namespace NUMINAMATH_CALUDE_james_music_beats_l214_21409

/-- Calculate the number of beats heard in a week given the beats per minute,
    hours of listening per day, and days in a week. -/
def beats_per_week (beats_per_minute : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  beats_per_minute * 60 * hours_per_day * days_per_week

/-- Theorem stating that listening to 200 beats per minute music for 2 hours a day
    for 7 days results in hearing 168,000 beats in a week. -/
theorem james_music_beats :
  beats_per_week 200 2 7 = 168000 := by
  sorry

end NUMINAMATH_CALUDE_james_music_beats_l214_21409


namespace NUMINAMATH_CALUDE_min_knights_in_village_l214_21413

theorem min_knights_in_village (total_people : Nat) (total_statements : Nat) (liar_statements : Nat) :
  total_people = 7 →
  total_statements = total_people * (total_people - 1) →
  total_statements = 42 →
  liar_statements = 24 →
  ∃ (knights : Nat), knights ≥ 3 ∧ 
    knights + (total_people - knights) = total_people ∧
    2 * knights * (total_people - knights) = liar_statements :=
by sorry

end NUMINAMATH_CALUDE_min_knights_in_village_l214_21413


namespace NUMINAMATH_CALUDE_sin_inequalities_l214_21451

theorem sin_inequalities (x : ℝ) (h : x > 0) :
  (Real.sin x ≤ x) ∧
  (Real.sin x ≥ x - x^3 / 6) ∧
  (Real.sin x ≤ x - x^3 / 6 + x^5 / 120) ∧
  (Real.sin x ≥ x - x^3 / 6 + x^5 / 120 - x^7 / 5040) := by
  sorry

end NUMINAMATH_CALUDE_sin_inequalities_l214_21451


namespace NUMINAMATH_CALUDE_student_handshake_problem_l214_21418

theorem student_handshake_problem (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  let total_handshakes := (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) / 2
  total_handshakes = 1020 → m * n = 280 := by
  sorry

end NUMINAMATH_CALUDE_student_handshake_problem_l214_21418


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l214_21450

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 9 * x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 33 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l214_21450


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l214_21440

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l214_21440


namespace NUMINAMATH_CALUDE_georgia_buttons_l214_21460

/-- Georgia's button problem -/
theorem georgia_buttons (yellow black green given_away remaining : ℕ) :
  yellow + black + green = given_away + remaining →
  remaining = 5 →
  yellow = 4 →
  black = 2 →
  green = 3 →
  given_away = 4 :=
by sorry

end NUMINAMATH_CALUDE_georgia_buttons_l214_21460


namespace NUMINAMATH_CALUDE_x_value_l214_21434

theorem x_value :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 90 →
    x = 137 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l214_21434


namespace NUMINAMATH_CALUDE_jessica_non_work_days_l214_21499

/-- Calculates the number of non-work days given the problem conditions -/
theorem jessica_non_work_days 
  (total_days : ℕ) 
  (full_day_earnings : ℚ) 
  (non_work_deduction : ℚ) 
  (half_days : ℕ) 
  (total_earnings : ℚ) 
  (h1 : total_days = 30)
  (h2 : full_day_earnings = 80)
  (h3 : non_work_deduction = 40)
  (h4 : half_days = 5)
  (h5 : total_earnings = 1600) :
  ∃ (non_work_days : ℕ), 
    non_work_days = 5 ∧ 
    (total_days : ℚ) = (non_work_days : ℚ) + (half_days : ℚ) + 
      ((total_earnings + non_work_deduction * (non_work_days : ℚ) - 
        (half_days : ℚ) * full_day_earnings / 2) / full_day_earnings) :=
by sorry

end NUMINAMATH_CALUDE_jessica_non_work_days_l214_21499


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l214_21426

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + 2*b = 0) → 
  (∃ x : ℝ, x^2 + 2*b*x + a = 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x : ℝ, x^2 + a'*x + 2*b' = 0) → 
    (∃ x : ℝ, x^2 + 2*b'*x + a' = 0) → 
    a' + b' ≥ a + b) → 
  a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l214_21426


namespace NUMINAMATH_CALUDE_notebook_cost_l214_21462

/-- Given the following conditions:
  * Total spent on school supplies is $32
  * A backpack costs $15
  * A pack of pens costs $1
  * A pack of pencils costs $1
  * 5 multi-subject notebooks were bought
Prove that each notebook costs $3 -/
theorem notebook_cost (total_spent : ℚ) (backpack_cost : ℚ) (pen_cost : ℚ) (pencil_cost : ℚ) (notebook_count : ℕ) :
  total_spent = 32 →
  backpack_cost = 15 →
  pen_cost = 1 →
  pencil_cost = 1 →
  notebook_count = 5 →
  (total_spent - backpack_cost - pen_cost - pencil_cost) / notebook_count = 3 := by
  sorry

#check notebook_cost

end NUMINAMATH_CALUDE_notebook_cost_l214_21462


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_six_l214_21442

/-- The probability of rolling an even number on a fair 12-sided die -/
def prob_even : ℚ := 1 / 2

/-- The number of ways to choose 3 dice from 6 -/
def choose_3_from_6 : ℕ := 20

/-- The probability of a specific scenario where exactly 3 dice show even -/
def prob_specific_scenario : ℚ := (1 / 2) ^ 6

/-- The probability of exactly three out of six fair 12-sided dice showing an even number -/
theorem prob_three_even_out_of_six : 
  choose_3_from_6 * prob_specific_scenario = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_six_l214_21442


namespace NUMINAMATH_CALUDE_rhombus_side_length_l214_21490

theorem rhombus_side_length 
  (d1 d2 : ℝ) 
  (h1 : d1 * d2 = 22) 
  (h2 : d1 + d2 = 10) 
  (h3 : (1/2) * d1 * d2 = 11) : 
  ∃ (side : ℝ), side = Real.sqrt 14 ∧ side^2 = (1/4) * (d1^2 + d2^2) := by
sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l214_21490


namespace NUMINAMATH_CALUDE_special_triangle_third_side_l214_21493

/-- A triangle with two sides of lengths 2 and 3, and the third side length satisfying a quadratic equation. -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a = 2
  h2 : b = 3
  h3 : c^2 - 10*c + 21 = 0
  h4 : a + b > c ∧ b + c > a ∧ c + a > b  -- Triangle inequality

/-- The third side of the SpecialTriangle has length 3. -/
theorem special_triangle_third_side (t : SpecialTriangle) : t.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_third_side_l214_21493


namespace NUMINAMATH_CALUDE_total_clothes_washed_l214_21438

/-- Represents the number of clothes a person has -/
structure ClothesCount where
  whiteShirts : ℕ
  coloredShirts : ℕ
  shorts : ℕ
  pants : ℕ

/-- Calculates the total number of clothes for a person -/
def totalClothes (c : ClothesCount) : ℕ :=
  c.whiteShirts + c.coloredShirts + c.shorts + c.pants

/-- Cally's clothes count -/
def cally : ClothesCount :=
  { whiteShirts := 10
    coloredShirts := 5
    shorts := 7
    pants := 6 }

/-- Danny's clothes count -/
def danny : ClothesCount :=
  { whiteShirts := 6
    coloredShirts := 8
    shorts := 10
    pants := 6 }

/-- Theorem stating that the total number of clothes washed by Cally and Danny is 58 -/
theorem total_clothes_washed : totalClothes cally + totalClothes danny = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_clothes_washed_l214_21438


namespace NUMINAMATH_CALUDE_tensor_identity_implies_unit_vector_l214_21497

def Vector2D := ℝ × ℝ

def tensor_product (m n : Vector2D) : Vector2D :=
  let (a, b) := m
  let (c, d) := n
  (a * c + b * d, a * d + b * c)

theorem tensor_identity_implies_unit_vector (p : Vector2D) :
  (∀ m : Vector2D, tensor_product m p = m) → p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_tensor_identity_implies_unit_vector_l214_21497


namespace NUMINAMATH_CALUDE_wednesday_dressing_time_l214_21421

/-- Represents the dressing times for each day of the school week -/
structure DressingTimes where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average dressing time for the week -/
def weekAverage (times : DressingTimes) : ℚ :=
  (times.monday + times.tuesday + times.wednesday + times.thursday + times.friday) / 5

/-- Theorem: Given the dressing times for Monday, Tuesday, Thursday, and Friday,
    and the old average dressing time, the dressing time for Wednesday must be 3 minutes
    to maintain the same average over the entire week. -/
theorem wednesday_dressing_time
  (times : DressingTimes)
  (h_monday : times.monday = 2)
  (h_tuesday : times.tuesday = 4)
  (h_thursday : times.thursday = 4)
  (h_friday : times.friday = 2)
  (h_old_avg : weekAverage times = 3) :
  times.wednesday = 3 := by
  sorry

#check wednesday_dressing_time

end NUMINAMATH_CALUDE_wednesday_dressing_time_l214_21421


namespace NUMINAMATH_CALUDE_pet_shop_inventory_l214_21464

/-- Represents the pet shop inventory problem --/
theorem pet_shop_inventory (num_kittens : ℕ) (puppy_cost kitten_cost total_value : ℕ) :
  num_kittens = 4 →
  puppy_cost = 20 →
  kitten_cost = 15 →
  total_value = 100 →
  ∃ (num_puppies : ℕ), num_puppies = 2 ∧ num_puppies * puppy_cost + num_kittens * kitten_cost = total_value :=
by
  sorry

end NUMINAMATH_CALUDE_pet_shop_inventory_l214_21464


namespace NUMINAMATH_CALUDE_sum_first_60_eq_1830_l214_21414

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 60 natural numbers is 1830 -/
theorem sum_first_60_eq_1830 : sum_first_n 60 = 1830 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_60_eq_1830_l214_21414


namespace NUMINAMATH_CALUDE_max_candies_drawn_exists_ten_candies_drawn_l214_21446

/-- Represents the number of candies of each color --/
structure CandyCount where
  yellow : ℕ
  red : ℕ
  blue : ℕ

/-- Represents the state of candies before and after drawing --/
structure CandyState where
  initial : CandyCount
  drawn : ℕ
  final : CandyCount

/-- Checks if the candy state satisfies all conditions --/
def satisfiesConditions (state : CandyState) : Prop :=
  state.initial.yellow * 3 = state.initial.red * 5 ∧
  state.final.yellow = 2 ∧
  state.final.red = 2 ∧
  state.final.blue ≥ 5 ∧
  state.drawn = state.initial.yellow + state.initial.red + state.initial.blue -
                (state.final.yellow + state.final.red + state.final.blue)

/-- Theorem stating that the maximum number of candies Petya can draw is 10 --/
theorem max_candies_drawn (state : CandyState) :
  satisfiesConditions state → state.drawn ≤ 10 :=
by
  sorry

/-- Theorem stating that it's possible to draw exactly 10 candies while satisfying all conditions --/
theorem exists_ten_candies_drawn :
  ∃ state : CandyState, satisfiesConditions state ∧ state.drawn = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_max_candies_drawn_exists_ten_candies_drawn_l214_21446


namespace NUMINAMATH_CALUDE_not_right_triangle_l214_21408

theorem not_right_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 2 * B) (h3 : A = 3 * C) :
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l214_21408


namespace NUMINAMATH_CALUDE_dog_treat_expenditure_l214_21401

/-- Calculate John's total expenditure on dog treats for a month --/
theorem dog_treat_expenditure :
  let treats_first_half : ℕ := 3 * 15
  let treats_second_half : ℕ := 4 * 15
  let total_treats : ℕ := treats_first_half + treats_second_half
  let original_price : ℚ := 0.1
  let discount_threshold : ℕ := 50
  let discount_rate : ℚ := 0.1
  let discounted_price : ℚ := original_price * (1 - discount_rate)
  total_treats > discount_threshold →
  (total_treats : ℚ) * discounted_price = 9.45 :=
by sorry

end NUMINAMATH_CALUDE_dog_treat_expenditure_l214_21401


namespace NUMINAMATH_CALUDE_factorization_example_l214_21481

theorem factorization_example (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_example_l214_21481


namespace NUMINAMATH_CALUDE_our_triangle_can_be_right_or_obtuse_l214_21400

/-- A triangle with given perimeter and inradius -/
structure Triangle where
  perimeter : ℝ
  inradius : ℝ

/-- Definition of our specific triangle -/
def our_triangle : Triangle := { perimeter := 12, inradius := 1 }

/-- A function to determine if a triangle can be right-angled or obtuse-angled -/
def can_be_right_or_obtuse (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = t.perimeter ∧
    (a * b * c) / (a + b + c) = 2 * t.inradius * t.perimeter ∧
    (a^2 + b^2 ≥ c^2 ∨ b^2 + c^2 ≥ a^2 ∨ c^2 + a^2 ≥ b^2)

/-- Theorem stating that our triangle can be right-angled or obtuse-angled -/
theorem our_triangle_can_be_right_or_obtuse :
  can_be_right_or_obtuse our_triangle := by
  sorry

end NUMINAMATH_CALUDE_our_triangle_can_be_right_or_obtuse_l214_21400


namespace NUMINAMATH_CALUDE_solution_of_quadratic_equations_l214_21419

theorem solution_of_quadratic_equations :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 = 3 * (2 * x + 1)
  let eq2 : ℝ → Prop := λ x ↦ 3 * x * (x + 2) = 4 * x + 8
  let sol1 : Set ℝ := {(3 + Real.sqrt 15) / 2, (3 - Real.sqrt 15) / 2}
  let sol2 : Set ℝ := {-2, 4/3}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_quadratic_equations_l214_21419


namespace NUMINAMATH_CALUDE_reflect_P_x_axis_l214_21417

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -2)

/-- Theorem: Reflecting P(3,-2) across the x-axis results in (3,2) -/
theorem reflect_P_x_axis : reflect_x P = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_x_axis_l214_21417


namespace NUMINAMATH_CALUDE_points_collinear_l214_21429

/-- Given vectors a and b in a vector space, and points A, B, C, D such that
    AB = a + 2b, BC = -5a + 6b, and CD = 7a - 2b, prove that A, B, and D are collinear. -/
theorem points_collinear 
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (a b : V) (A B C D : V) 
  (hAB : B - A = a + 2 • b)
  (hBC : C - B = -5 • a + 6 • b)
  (hCD : D - C = 7 • a - 2 • b) :
  ∃ (t : ℝ), D - A = t • (B - A) :=
sorry

end NUMINAMATH_CALUDE_points_collinear_l214_21429


namespace NUMINAMATH_CALUDE_cos_seven_pi_fourth_l214_21458

theorem cos_seven_pi_fourth : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourth_l214_21458


namespace NUMINAMATH_CALUDE_equation_solution_l214_21473

theorem equation_solution :
  ∃ x : ℚ, x = -62/29 ∧ (Real.sqrt (7*x + 1) / Real.sqrt (4*(x + 2) - 1) = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l214_21473


namespace NUMINAMATH_CALUDE_composite_numbers_l214_21449

theorem composite_numbers (n : ℕ) (h : n = 3^2001) : 
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 2^n + 1 = a * b) ∧ 
  (∃ (c d : ℕ), c > 1 ∧ d > 1 ∧ 2^n - 1 = c * d) := by
sorry


end NUMINAMATH_CALUDE_composite_numbers_l214_21449


namespace NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l214_21477

theorem no_two_digit_number_satisfies_conditions : ¬∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  Even n ∧            -- even
  n % 13 = 0 ∧        -- multiple of 13
  ∃ a b : ℕ,          -- digits a and b
    n = 10 * a + b ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    ∃ k : ℕ, a * b = k * k  -- product of digits is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l214_21477


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l214_21405

theorem trigonometric_simplification (α : ℝ) :
  (1 + Real.cos α + Real.cos (2 * α) + Real.cos (3 * α)) /
  (Real.cos α + 2 * (Real.cos α)^2 - 1) = 2 * Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l214_21405


namespace NUMINAMATH_CALUDE_no_divisible_by_six_l214_21476

theorem no_divisible_by_six : ∀ z : ℕ, z < 10 → ¬(35000 + z * 100 + 45) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_six_l214_21476


namespace NUMINAMATH_CALUDE_chicken_count_after_purchase_l214_21403

theorem chicken_count_after_purchase (initial_count purchase_count : ℕ) 
  (h1 : initial_count = 26) 
  (h2 : purchase_count = 28) : 
  initial_count + purchase_count = 54 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_after_purchase_l214_21403


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l214_21480

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 4 * x + 15 = 0 ↔ x = Complex.mk a b ∨ x = Complex.mk a (-b)) → 
  a + b^2 = 162 / 50 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l214_21480


namespace NUMINAMATH_CALUDE_opposite_of_one_third_l214_21485

theorem opposite_of_one_third :
  -(1/3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_third_l214_21485


namespace NUMINAMATH_CALUDE_prime_arithmetic_mean_median_l214_21495

theorem prime_arithmetic_mean_median (a b c : ℕ) : 
  a = 2 → 
  Nat.Prime a → 
  Nat.Prime b → 
  Nat.Prime c → 
  a < b → 
  b < c → 
  b ≠ a + 1 → 
  (a + b + c) / 3 = 6 * b → 
  c / b = 83 / 5 := by
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_mean_median_l214_21495


namespace NUMINAMATH_CALUDE_percentage_problem_l214_21492

/-- The percentage P that satisfies the equation (1/10 * 8000) - (P/100 * 8000) = 796 -/
theorem percentage_problem (P : ℝ) : (1/10 * 8000) - (P/100 * 8000) = 796 ↔ P = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l214_21492


namespace NUMINAMATH_CALUDE_max_value_expression_l214_21472

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6 ≤ a ∧ a ≤ 6)
  (hb : -6 ≤ b ∧ b ≤ 6)
  (hc : -6 ≤ c ∧ c ≤ 6)
  (hd : -6 ≤ d ∧ d ≤ 6) :
  (∀ x y z w, -6 ≤ x ∧ x ≤ 6 → -6 ≤ y ∧ y ≤ 6 → -6 ≤ z ∧ z ≤ 6 → -6 ≤ w ∧ w ≤ 6 →
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) →
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a = 156 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l214_21472


namespace NUMINAMATH_CALUDE_dividend_problem_l214_21422

theorem dividend_problem (total : ℚ) (a b c : ℚ) 
  (h1 : total = 527)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  a = 62 := by
sorry

end NUMINAMATH_CALUDE_dividend_problem_l214_21422


namespace NUMINAMATH_CALUDE_function_domain_range_l214_21448

theorem function_domain_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 + m * x + 1)) ↔ 0 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_range_l214_21448


namespace NUMINAMATH_CALUDE_rosas_phone_calls_l214_21466

/-- Rosa's phone book calling problem -/
theorem rosas_phone_calls (pages_last_week pages_this_week : ℝ) 
  (h1 : pages_last_week = 10.2)
  (h2 : pages_this_week = 8.6) : 
  pages_last_week + pages_this_week = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_rosas_phone_calls_l214_21466


namespace NUMINAMATH_CALUDE_binary_linear_equation_ab_eq_one_l214_21489

/-- A binary linear equation is an equation of the form ax + by = c, where a, b, and c are constants and x and y are variables. -/
def IsBinaryLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

theorem binary_linear_equation_ab_eq_one (a b : ℝ) :
  IsBinaryLinearEquation (fun x y => x^(2*a) + y^(b-1)) →
  a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_binary_linear_equation_ab_eq_one_l214_21489


namespace NUMINAMATH_CALUDE_cube_less_than_self_l214_21435

theorem cube_less_than_self (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : a^3 < a := by
  sorry

end NUMINAMATH_CALUDE_cube_less_than_self_l214_21435


namespace NUMINAMATH_CALUDE_expression_evaluation_l214_21423

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  a / (a^2 - 2*a + 1) / (1 + 1/(a - 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l214_21423


namespace NUMINAMATH_CALUDE_px_length_l214_21425

-- Define the quadrilateral CDXW
structure Quadrilateral :=
  (C D W X P : ℝ × ℝ)
  (cd_parallel_wx : (D.1 - C.1) * (X.2 - W.2) = (D.2 - C.2) * (X.1 - W.1))
  (p_on_cx : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (C.1 + t * (X.1 - C.1), C.2 + t * (X.2 - C.2)))
  (p_on_dw : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (D.1 + s * (W.1 - D.1), D.2 + s * (W.2 - D.2)))
  (cx_length : Real.sqrt ((X.1 - C.1)^2 + (X.2 - C.2)^2) = 30)
  (dp_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 15)
  (pw_length : Real.sqrt ((W.1 - P.1)^2 + (W.2 - P.2)^2) = 45)

-- Theorem statement
theorem px_length (q : Quadrilateral) : 
  Real.sqrt ((q.X.1 - q.P.1)^2 + (q.X.2 - q.P.2)^2) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_px_length_l214_21425


namespace NUMINAMATH_CALUDE_inequality_equivalence_l214_21445

theorem inequality_equivalence (x : ℝ) : 
  -2 < (x^2 - 12*x + 20) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 12*x + 20) / (x^2 - 4*x + 8) < 2 ↔ 
  x > 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l214_21445


namespace NUMINAMATH_CALUDE_direct_proportion_implies_m_zero_l214_21484

/-- A function f is a direct proportion function if there exists a constant k such that f x = k * x for all x -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function y = -2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x + m

theorem direct_proportion_implies_m_zero (m : ℝ) :
  is_direct_proportion (f m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_implies_m_zero_l214_21484


namespace NUMINAMATH_CALUDE_south_opposite_of_north_l214_21452

/-- Represents the direction of movement --/
inductive Direction
  | North
  | South

/-- Represents a distance with direction --/
structure DirectedDistance where
  distance : ℝ
  direction : Direction

/-- Denotes a distance in kilometers with a sign --/
def denote (d : DirectedDistance) : ℝ :=
  match d.direction with
  | Direction.North => d.distance
  | Direction.South => -d.distance

theorem south_opposite_of_north 
  (h : denote { distance := 3, direction := Direction.North } = 3) :
  denote { distance := 5, direction := Direction.South } = -5 := by
  sorry


end NUMINAMATH_CALUDE_south_opposite_of_north_l214_21452


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l214_21454

theorem inequality_system_solution_set : 
  ∀ x : ℝ, (abs x < 1 ∧ x * (x + 2) > 0) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l214_21454


namespace NUMINAMATH_CALUDE_chord_length_is_sqrt_34_l214_21470

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + m = 0
def l (x y : ℝ) : Prop := x + y = 0

-- Define external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ x y, C₁ x y ∧ C₂ x y m ∧ (x - 0)^2 + (y - 0)^2 = (2 + Real.sqrt (25 - m))^2

-- Theorem statement
theorem chord_length_is_sqrt_34 (m : ℝ) :
  externally_tangent m →
  ∃ x₁ y₁ x₂ y₂,
    C₂ x₁ y₁ m ∧ C₂ x₂ y₂ m ∧
    l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 34 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_is_sqrt_34_l214_21470


namespace NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_ab_plus_one_neq_a_plus_b_iff_min_value_of_expression_l214_21479

-- Statement 1
theorem sqrt_plus_inverse_geq_two (x : ℝ) (hx : x > 0) :
  Real.sqrt x + 1 / Real.sqrt x ≥ 2 := by sorry

-- Statement 2
theorem ab_plus_one_neq_a_plus_b_iff (a b : ℝ) :
  a * b + 1 ≠ a + b ↔ a ≠ 1 ∧ b ≠ 1 := by sorry

-- Statement 3
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → a / b + 1 / (a * b) ≥ m) ∧
  a / b + 1 / (a * b) = 2 * Real.sqrt 2 + 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_plus_inverse_geq_two_ab_plus_one_neq_a_plus_b_iff_min_value_of_expression_l214_21479


namespace NUMINAMATH_CALUDE_negation_of_all_exponential_monotonic_l214_21444

-- Define the set of exponential functions
def ExponentialFunction : Type := ℝ → ℝ

-- Define the property of being monotonic
def Monotonic (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem negation_of_all_exponential_monotonic :
  (¬ ∀ f : ExponentialFunction, Monotonic f) ↔ (∃ f : ExponentialFunction, ¬ Monotonic f) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_exponential_monotonic_l214_21444


namespace NUMINAMATH_CALUDE_set_equality_through_double_complement_l214_21467

universe u

theorem set_equality_through_double_complement 
  {U : Type u} [Nonempty U] (M N P : Set U) 
  (h1 : M = (Nᶜ : Set U)) 
  (h2 : N = (Pᶜ : Set U)) : 
  M = P := by
  sorry

end NUMINAMATH_CALUDE_set_equality_through_double_complement_l214_21467


namespace NUMINAMATH_CALUDE_consecutive_digit_sum_divisibility_l214_21424

theorem consecutive_digit_sum_divisibility (a : ℕ) (h : a ≤ 5) :
  ∃ (k : ℤ), (10000 * a + 1000 * (a + 1) + 100 * (a + 2) + 10 * (a + 3) + (a + 4)) +
             (10000 * (a + 4) + 1000 * (a + 3) + 100 * (a + 2) + 10 * (a + 1) + a) = 11211 * k :=
sorry

end NUMINAMATH_CALUDE_consecutive_digit_sum_divisibility_l214_21424


namespace NUMINAMATH_CALUDE_three_propositions_are_true_l214_21407

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a point
def Point : Type := sorry

-- Define the relation of two lines being skew
def are_skew (a b : Line) : Prop := sorry

-- Define the relation of a line intersecting another line at a point
def intersects_at (l1 l2 : Line) (p : Point) : Prop := sorry

-- Define the relation of two lines being parallel
def are_parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the relation of two lines determining a plane
def determine_plane (l1 l2 : Line) (p : Plane) : Prop := sorry

theorem three_propositions_are_true :
  -- Proposition 1
  (∀ (a b c d : Line) (E F G H : Point),
    are_skew a b ∧
    intersects_at c a E ∧ intersects_at c b F ∧
    intersects_at d a G ∧ intersects_at d b H ∧
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
    are_skew c d) ∧
  -- Proposition 2
  (∀ (a b l : Line),
    are_skew a b →
    are_parallel l a →
    ¬(are_parallel l b)) ∧
  -- Proposition 3
  (∀ (a b l : Line),
    are_skew a b →
    (∃ (P Q : Point), intersects_at l a P ∧ intersects_at l b Q) →
    ∃ (p1 p2 : Plane), determine_plane a l p1 ∧ determine_plane b l p2) :=
by sorry

end NUMINAMATH_CALUDE_three_propositions_are_true_l214_21407


namespace NUMINAMATH_CALUDE_negation_of_proposition_l214_21427

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l214_21427
