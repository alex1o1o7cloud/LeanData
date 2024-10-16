import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l305_30552

/-- Given a geometric sequence with first term 120, second term b, and third term 60/24,
    prove that b = 10√3 when b is positive. -/
theorem geometric_sequence_b_value (b : ℝ) (h1 : b > 0) 
    (h2 : ∃ (r : ℝ), 120 * r = b ∧ b * r = 60 / 24) : b = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l305_30552


namespace NUMINAMATH_CALUDE_problem_solution_l305_30544

theorem problem_solution (a x y : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x = 2*a)
  (h4 : 4*y^3 + Real.sin y * Real.cos y = -a) :
  3 * Real.sin ((π + x)/2 + y) = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l305_30544


namespace NUMINAMATH_CALUDE_solution_correctness_l305_30543

variables {α : Type*} [Field α]
variables (x y z x' y' z' x'' y'' z'' u' v' w' : α)

def system_solution (u v w : α) : Prop :=
  x * u + y * v + z * w = u' ∧
  x' * u + y' * v + z' * w = v' ∧
  x'' * u + y'' * v + z'' * w = w'

theorem solution_correctness :
  ∃ (u v w : α),
    system_solution x y z x' y' z' x'' y'' z'' u' v' w' u v w ∧
    u = u' * x + v' * x' + w' * x'' ∧
    v = u' * y + v' * y' + w' * y'' ∧
    w = u' * z + v' * z' + w' * z'' :=
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l305_30543


namespace NUMINAMATH_CALUDE_average_fishes_is_45_2_l305_30548

-- Define the number of lakes
def num_lakes : ℕ := 5

-- Define the number of fishes caught in each lake
def lake_marion : ℕ := 38
def lake_norman : ℕ := 52
def lake_wateree : ℕ := 27
def lake_wylie : ℕ := 45
def lake_keowee : ℕ := 64

-- Define the total number of fishes caught
def total_fishes : ℕ := lake_marion + lake_norman + lake_wateree + lake_wylie + lake_keowee

-- Define the average number of fishes caught per lake
def average_fishes : ℚ := total_fishes / num_lakes

-- Theorem statement
theorem average_fishes_is_45_2 : average_fishes = 45.2 := by
  sorry

end NUMINAMATH_CALUDE_average_fishes_is_45_2_l305_30548


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l305_30532

/-- Given two points P and Q that are symmetric about a line l, prove that the equation of l is x - y + 1 = 0 --/
theorem symmetric_points_line_equation (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let Q : ℝ × ℝ := (b - 1, a + 1)
  let l : Set (ℝ × ℝ) := {(x, y) | x - y + 1 = 0}
  (∀ (M : ℝ × ℝ), M ∈ l ↔ (dist M P)^2 = (dist M Q)^2) →
  l = {(x, y) | x - y + 1 = 0} :=
by sorry


end NUMINAMATH_CALUDE_symmetric_points_line_equation_l305_30532


namespace NUMINAMATH_CALUDE_percentage_increase_l305_30551

theorem percentage_increase (x : ℝ) (h1 : x = 14.4) (h2 : x > 12) :
  (x - 12) / 12 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l305_30551


namespace NUMINAMATH_CALUDE_square_area_increase_l305_30508

theorem square_area_increase (a : ℝ) (ha : a > 0) : 
  let side_b := 2 * a
  let side_c := side_b * 1.8
  let area_a := a ^ 2
  let area_b := side_b ^ 2
  let area_c := side_c ^ 2
  (area_c - (area_a + area_b)) / (area_a + area_b) = 1.592 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l305_30508


namespace NUMINAMATH_CALUDE_henry_lawn_mowing_l305_30587

/-- The number of lawns Henry was supposed to mow -/
def total_lawns : ℕ := 12

/-- The amount Henry earns per lawn -/
def earnings_per_lawn : ℕ := 5

/-- The number of lawns Henry forgot to mow -/
def forgotten_lawns : ℕ := 7

/-- The amount Henry actually earned -/
def actual_earnings : ℕ := 25

theorem henry_lawn_mowing :
  total_lawns = (actual_earnings / earnings_per_lawn) + forgotten_lawns :=
by sorry

end NUMINAMATH_CALUDE_henry_lawn_mowing_l305_30587


namespace NUMINAMATH_CALUDE_max_value_of_expression_upper_bound_achievable_l305_30504

theorem max_value_of_expression (x : ℝ) :
  x^4 / (x^8 + 2*x^6 - 3*x^4 + 5*x^3 + 8*x^2 + 5*x + 25) ≤ 1/15 :=
by sorry

theorem upper_bound_achievable :
  ∃ x : ℝ, x^4 / (x^8 + 2*x^6 - 3*x^4 + 5*x^3 + 8*x^2 + 5*x + 25) = 1/15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_upper_bound_achievable_l305_30504


namespace NUMINAMATH_CALUDE_zeros_of_derivative_form_arithmetic_progression_l305_30598

/-- A fourth-degree polynomial whose zeros form an arithmetic progression -/
def ArithmeticZerosPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (a r : ℝ) (α : ℝ), α ≠ 0 ∧ r > 0 ∧
    ∀ x, f x = α * (x - a) * (x - (a + r)) * (x - (a + 2*r)) * (x - (a + 3*r))

/-- The zeros of a polynomial form an arithmetic progression -/
def ZerosFormArithmeticProgression (f : ℝ → ℝ) : Prop :=
  ∃ (a d : ℝ), ∀ x, f x = 0 → ∃ n : ℕ, x = a + n * d

/-- The main theorem -/
theorem zeros_of_derivative_form_arithmetic_progression
  (f : ℝ → ℝ) (hf : ArithmeticZerosPolynomial f) :
  ZerosFormArithmeticProgression f' :=
sorry

end NUMINAMATH_CALUDE_zeros_of_derivative_form_arithmetic_progression_l305_30598


namespace NUMINAMATH_CALUDE_ralph_weekly_tv_hours_l305_30570

/-- Represents Ralph's TV watching habits for a week -/
structure TVWatchingHabits where
  weekday_hours : ℕ
  weekend_hours : ℕ
  weekdays : ℕ
  weekend_days : ℕ

/-- Calculates the total hours of TV watched in a week -/
def total_weekly_hours (habits : TVWatchingHabits) : ℕ :=
  habits.weekday_hours * habits.weekdays + habits.weekend_hours * habits.weekend_days

/-- Theorem stating that Ralph watches 32 hours of TV in a week -/
theorem ralph_weekly_tv_hours :
  let habits : TVWatchingHabits := {
    weekday_hours := 4,
    weekend_hours := 6,
    weekdays := 5,
    weekend_days := 2
  }
  total_weekly_hours habits = 32 := by
  sorry

end NUMINAMATH_CALUDE_ralph_weekly_tv_hours_l305_30570


namespace NUMINAMATH_CALUDE_equation_solutions_l305_30590

theorem equation_solutions : 
  ∀ x y z : ℕ+, 
    (x.val * y.val + y.val * z.val + z.val * x.val - x.val * y.val * z.val = 2) ↔ 
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ 
     (x = 2 ∧ y = 3 ∧ z = 4) ∨ (x = 2 ∧ y = 4 ∧ z = 3) ∨
     (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 3 ∧ y = 4 ∧ z = 2) ∨
     (x = 4 ∧ y = 2 ∧ z = 3) ∨ (x = 4 ∧ y = 3 ∧ z = 2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l305_30590


namespace NUMINAMATH_CALUDE_carpet_length_l305_30596

theorem carpet_length (floor_area : ℝ) (carpet_coverage : ℝ) (carpet_width : ℝ) :
  floor_area = 120 →
  carpet_coverage = 0.3 →
  carpet_width = 4 →
  (carpet_coverage * floor_area) / carpet_width = 9 := by
  sorry

end NUMINAMATH_CALUDE_carpet_length_l305_30596


namespace NUMINAMATH_CALUDE_cleaner_solution_calculation_l305_30518

/-- Represents the amount of cleaner solution needed for each type of stain -/
structure StainCleaner where
  dog : Nat
  cat : Nat
  bird : Nat
  rabbit : Nat
  fish : Nat

/-- Represents the number of stains for each type -/
structure StainCount where
  dog : Nat
  cat : Nat
  bird : Nat
  rabbit : Nat
  fish : Nat

def cleaner : StainCleaner :=
  { dog := 6, cat := 4, bird := 3, rabbit := 1, fish := 2 }

def weeklyStains : StainCount :=
  { dog := 10, cat := 8, bird := 5, rabbit := 1, fish := 3 }

def bottleSize : Nat := 64

/-- Calculates the total amount of cleaner solution needed -/
def totalSolutionNeeded (c : StainCleaner) (s : StainCount) : Nat :=
  c.dog * s.dog + c.cat * s.cat + c.bird * s.bird + c.rabbit * s.rabbit + c.fish * s.fish

/-- Calculates the additional amount of cleaner solution needed -/
def additionalSolutionNeeded (total : Nat) (bottleSize : Nat) : Nat :=
  if total > bottleSize then total - bottleSize else 0

theorem cleaner_solution_calculation :
  totalSolutionNeeded cleaner weeklyStains = 114 ∧
  additionalSolutionNeeded (totalSolutionNeeded cleaner weeklyStains) bottleSize = 50 := by
  sorry

end NUMINAMATH_CALUDE_cleaner_solution_calculation_l305_30518


namespace NUMINAMATH_CALUDE_rectangle_area_l305_30554

/-- Given a rectangle with perimeter 100 cm and diagonal x cm, its area is 1250 - (x^2 / 2) square cm -/
theorem rectangle_area (x : ℝ) :
  let perimeter : ℝ := 100
  let diagonal : ℝ := x
  let area : ℝ := 1250 - (x^2 / 2)
  (∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    2 * (length + width) = perimeter ∧
    length^2 + width^2 = diagonal^2 ∧
    length * width = area) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l305_30554


namespace NUMINAMATH_CALUDE_laurens_mail_problem_l305_30569

/-- Lauren's mail sending problem -/
theorem laurens_mail_problem (monday tuesday wednesday thursday : ℕ) :
  monday = 65 ∧
  tuesday > monday ∧
  wednesday = tuesday - 5 ∧
  thursday = wednesday + 15 ∧
  monday + tuesday + wednesday + thursday = 295 →
  tuesday - monday = 10 := by
sorry

end NUMINAMATH_CALUDE_laurens_mail_problem_l305_30569


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l305_30502

/-- Given a football team with the following properties:
  * There are 120 players in total
  * 62 players are throwers
  * Of the non-throwers, three-fifths are left-handed
  * All throwers are right-handed
  Prove that the total number of right-handed players is 86 -/
theorem football_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (non_throwers : ℕ) 
  (left_handed_non_throwers : ℕ) 
  (right_handed_non_throwers : ℕ) : 
  total_players = 120 →
  throwers = 62 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = (3 * non_throwers) / 5 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  throwers + right_handed_non_throwers = 86 := by
  sorry

#check football_team_right_handed_players

end NUMINAMATH_CALUDE_football_team_right_handed_players_l305_30502


namespace NUMINAMATH_CALUDE_cherry_tomato_jars_l305_30514

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 550) (h2 : tomatoes_per_jar = 14) :
  ∃ (jars : ℕ), jars = ((total_tomatoes + tomatoes_per_jar - 1) / tomatoes_per_jar) ∧ jars = 40 :=
by sorry

end NUMINAMATH_CALUDE_cherry_tomato_jars_l305_30514


namespace NUMINAMATH_CALUDE_rationalize_denominator_l305_30594

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l305_30594


namespace NUMINAMATH_CALUDE_parallelepiped_volume_example_l305_30512

/-- The volume of a parallelepiped with given dimensions -/
def parallelepipedVolume (base height depth : ℝ) : ℝ :=
  base * depth * height

/-- Theorem: The volume of a parallelepiped with base 28 cm, height 32 cm, and depth 15 cm is 13440 cubic centimeters -/
theorem parallelepiped_volume_example : parallelepipedVolume 28 32 15 = 13440 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_example_l305_30512


namespace NUMINAMATH_CALUDE_problem_statement_l305_30579

theorem problem_statement : 
  |1 - Real.sqrt 3| + 2 * Real.cos (30 * π / 180) - Real.sqrt 12 - 2023 = -2024 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l305_30579


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l305_30595

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -1; 2, 4]
  A * B = !![17, 1; 16, -12] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l305_30595


namespace NUMINAMATH_CALUDE_bears_in_stock_is_four_l305_30555

/-- The number of bears in the new shipment -/
def new_shipment : ℕ := 10

/-- The number of bears on each shelf -/
def bears_per_shelf : ℕ := 7

/-- The number of shelves used -/
def shelves_used : ℕ := 2

/-- The number of bears in stock before the shipment -/
def bears_in_stock : ℕ := shelves_used * bears_per_shelf - new_shipment

theorem bears_in_stock_is_four : bears_in_stock = 4 := by
  sorry

end NUMINAMATH_CALUDE_bears_in_stock_is_four_l305_30555


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l305_30550

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (n : ℕ) : 
  a₁ = 165 → aₙ = 45 → d = -5 → a₁ + (n - 1) * d = aₙ → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l305_30550


namespace NUMINAMATH_CALUDE_fruit_basket_count_l305_30560

/-- The number of different fruit baskets that can be created -/
def num_fruit_baskets (num_apples : ℕ) (num_oranges : ℕ) : ℕ :=
  num_apples * num_oranges

/-- Theorem stating that the number of different fruit baskets with 7 apples and 12 oranges is 84 -/
theorem fruit_basket_count :
  num_fruit_baskets 7 12 = 84 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l305_30560


namespace NUMINAMATH_CALUDE_ones_digit_73_pow_351_l305_30576

theorem ones_digit_73_pow_351 : (73^351) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_73_pow_351_l305_30576


namespace NUMINAMATH_CALUDE_number_problem_l305_30563

theorem number_problem (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l305_30563


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l305_30503

theorem cos_alpha_plus_5pi_12 (α : ℝ) (h : Real.sin (α - π/12) = 1/3) :
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l305_30503


namespace NUMINAMATH_CALUDE_inequality_proof_l305_30540

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) :
  a < 2*b - b^2/a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l305_30540


namespace NUMINAMATH_CALUDE_sandbox_length_l305_30515

/-- The length of a rectangular sandbox given its width and area -/
theorem sandbox_length (width : ℝ) (area : ℝ) (h1 : width = 146) (h2 : area = 45552) :
  area / width = 312 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_length_l305_30515


namespace NUMINAMATH_CALUDE_minimum_students_l305_30589

theorem minimum_students (b g : ℕ) : 
  (3 * b = 8 * g) →  -- From the equation (3/4)b = 2(2/3)g simplified
  (b ≥ 1) →          -- At least one boy
  (g ≥ 1) →          -- At least one girl
  (∀ b' g', (3 * b' = 8 * g') → b' + g' ≥ b + g) →  -- Minimum condition
  b + g = 25 := by
sorry

end NUMINAMATH_CALUDE_minimum_students_l305_30589


namespace NUMINAMATH_CALUDE_center_is_three_l305_30541

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements on the main diagonal (top-left to bottom-right) -/
def mainDiagonalSum (g : Grid) : ℕ :=
  (g 0 0).val + (g 1 1).val + (g 2 2).val

/-- The sum of elements on the other diagonal (top-right to bottom-left) -/
def otherDiagonalSum (g : Grid) : ℕ :=
  (g 0 2).val + (g 1 1).val + (g 2 0).val

/-- All numbers in the grid are distinct and from 1 to 9 -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ (g i j).val ∧ (g i j).val ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

theorem center_is_three (g : Grid) 
  (h1 : isValidGrid g)
  (h2 : mainDiagonalSum g = 6)
  (h3 : otherDiagonalSum g = 20) :
  (g 1 1).val = 3 := by
  sorry

end NUMINAMATH_CALUDE_center_is_three_l305_30541


namespace NUMINAMATH_CALUDE_magical_points_on_quadratic_unique_magical_point_condition_l305_30571

/-- Definition of a magical point -/
def is_magical_point (x y : ℝ) : Prop := y = 2 * x

/-- The quadratic function y = x^2 - x - 4 -/
def quadratic_function (x : ℝ) : ℝ := x^2 - x - 4

/-- The generalized quadratic function y = tx^2 + (t-2)x - 4 -/
def generalized_quadratic_function (t x : ℝ) : ℝ := t * x^2 + (t - 2) * x - 4

theorem magical_points_on_quadratic :
  ∀ x y : ℝ, is_magical_point x y ∧ y = quadratic_function x ↔ (x = -1 ∧ y = -2) ∨ (x = 4 ∧ y = 8) :=
sorry

theorem unique_magical_point_condition :
  ∀ t : ℝ, t ≠ 0 →
  (∃! x y : ℝ, is_magical_point x y ∧ y = generalized_quadratic_function t x) ↔ t = -4 :=
sorry

end NUMINAMATH_CALUDE_magical_points_on_quadratic_unique_magical_point_condition_l305_30571


namespace NUMINAMATH_CALUDE_sqrt_two_division_l305_30553

theorem sqrt_two_division : 3 * Real.sqrt 2 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_division_l305_30553


namespace NUMINAMATH_CALUDE_hedge_trimming_charge_equals_685_l305_30513

/-- Calculates the total charge for trimming a hedge with various shapes -/
def hedge_trimming_charge (basic_trim_price : ℚ) (sphere_price : ℚ) (pyramid_price : ℚ) 
  (cube_price : ℚ) (combined_shape_extra : ℚ) (total_boxwoods : ℕ) (sphere_count : ℕ) 
  (pyramid_count : ℕ) (cube_count : ℕ) (sphere_pyramid_count : ℕ) (sphere_cube_count : ℕ) : ℚ :=
  let basic_trim_total := basic_trim_price * total_boxwoods
  let sphere_total := sphere_price * sphere_count
  let pyramid_total := pyramid_price * pyramid_count
  let cube_total := cube_price * cube_count
  let sphere_pyramid_total := (sphere_price + pyramid_price + combined_shape_extra) * sphere_pyramid_count
  let sphere_cube_total := (sphere_price + cube_price + combined_shape_extra) * sphere_cube_count
  basic_trim_total + sphere_total + pyramid_total + cube_total + sphere_pyramid_total + sphere_cube_total

/-- The total charge for trimming the hedge is $685.00 -/
theorem hedge_trimming_charge_equals_685 : 
  hedge_trimming_charge 5 15 20 25 10 40 2 5 3 4 2 = 685 := by
  sorry

end NUMINAMATH_CALUDE_hedge_trimming_charge_equals_685_l305_30513


namespace NUMINAMATH_CALUDE_problem_statement_l305_30526

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The equation given in the problem -/
def equation (z a : ℂ) : Prop := (2 + i) * z = 1 + a * i^3

/-- A complex number is in Quadrant IV if its real part is positive and imaginary part is negative -/
def inQuadrantIV (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem problem_statement (z a : ℂ) :
  isPurelyImaginary z → equation z a → inQuadrantIV (a + z) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l305_30526


namespace NUMINAMATH_CALUDE_fraction_evaluation_l305_30573

theorem fraction_evaluation :
  let x : ℚ := 2/3
  let y : ℚ := 8/10
  (6*x + 10*y) / (60*x*y) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l305_30573


namespace NUMINAMATH_CALUDE_average_speed_theorem_l305_30584

/-- Proves that the average speed of a trip is 40 mph given specific conditions -/
theorem average_speed_theorem (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

#check average_speed_theorem

end NUMINAMATH_CALUDE_average_speed_theorem_l305_30584


namespace NUMINAMATH_CALUDE_bee_return_theorem_l305_30523

/-- Represents a position on the hexagonal grid -/
structure HexPosition where
  x : ℤ
  y : ℤ

/-- Represents a move on the hexagonal grid -/
structure HexMove where
  direction : Fin 6
  length : ℕ

/-- Applies a move to a position -/
def applyMove (pos : HexPosition) (move : HexMove) : HexPosition :=
  sorry

/-- Applies a sequence of moves to a position -/
def applyMoves (pos : HexPosition) (moves : List HexMove) : HexPosition :=
  sorry

/-- Generates a sequence of moves for a given N -/
def generateMoves (N : ℕ) : List HexMove :=
  sorry

theorem bee_return_theorem (N : ℕ) (h : N ≥ 3) :
  ∃ (startPos : HexPosition),
    applyMoves startPos (generateMoves N) = startPos :=
  sorry

end NUMINAMATH_CALUDE_bee_return_theorem_l305_30523


namespace NUMINAMATH_CALUDE_unique_solution_x_power_x_power_x_eq_2_l305_30577

theorem unique_solution_x_power_x_power_x_eq_2 :
  ∃! (x : ℝ), x > 0 ∧ x^(x^x) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_x_power_x_power_x_eq_2_l305_30577


namespace NUMINAMATH_CALUDE_power_equality_implies_exponent_l305_30538

theorem power_equality_implies_exponent (a : ℝ) (m : ℕ) (h : (a^2)^m = a^6) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_exponent_l305_30538


namespace NUMINAMATH_CALUDE_functional_equation_solution_l305_30525

/-- A polynomial of degree 2015 -/
def Polynomial2015 := Polynomial ℝ

/-- An odd polynomial of degree 2015 -/
def OddPolynomial2015 := {Q : Polynomial2015 // ∀ x, Q.eval (-x) = -Q.eval x}

/-- The functional equation P(x) + P(1-x) = 1 -/
def SatisfiesFunctionalEquation (P : Polynomial2015) : Prop :=
  ∀ x, P.eval x + P.eval (1 - x) = 1

theorem functional_equation_solution :
  ∀ P : Polynomial2015, SatisfiesFunctionalEquation P →
  ∃ Q : OddPolynomial2015, ∀ x, P.eval x = Q.val.eval (1/2 - x) + 1/2 :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l305_30525


namespace NUMINAMATH_CALUDE_circle_configuration_diameter_l305_30597

/-- Given a configuration of circles as described, prove the diameter length --/
theorem circle_configuration_diameter : 
  ∀ (r s : ℝ) (shaded_area circle_c_area : ℝ),
  r > 0 → s > 0 →
  shaded_area = 39 * Real.pi →
  circle_c_area = 9 * Real.pi →
  shaded_area = (Real.pi / 2) * ((r + s)^2 - r^2 - s^2) - circle_c_area →
  2 * (r + s) = 32 := by
  sorry

end NUMINAMATH_CALUDE_circle_configuration_diameter_l305_30597


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l305_30599

theorem isabel_piggy_bank (initial_amount : ℝ) : 
  (initial_amount / 2) / 2 = 51 → initial_amount = 204 := by
  sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l305_30599


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l305_30549

/-- Given a quadratic equation x^2 + ax + a - 1 = 0 where -2 is a root, 
    prove that a = 3 and the other root is -1, and that for any real a, 
    the equation always has real roots. -/
theorem quadratic_equation_properties (a : ℝ) : 
  ((-2 : ℝ)^2 + a*(-2) + a - 1 = 0) → 
  (a = 3 ∧ ∃ x : ℝ, x ≠ -2 ∧ x^2 + a*x + a - 1 = 0 ∧ x = -1) ∧
  (∀ b : ℝ, ∃ x : ℝ, x^2 + b*x + b - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l305_30549


namespace NUMINAMATH_CALUDE_hilt_remaining_money_l305_30546

def remaining_money (initial_amount cost : ℕ) : ℕ :=
  initial_amount - cost

theorem hilt_remaining_money :
  remaining_money 15 11 = 4 := by sorry

end NUMINAMATH_CALUDE_hilt_remaining_money_l305_30546


namespace NUMINAMATH_CALUDE_exists_n_plus_Sn_1980_consecutive_n_plus_Sn_l305_30574

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Part a: Existence of n such that n + S(n) = 1980
theorem exists_n_plus_Sn_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Part b: At least one of two consecutive naturals is of the form n + S(n)
theorem consecutive_n_plus_Sn (k : ℕ) : 
  (∃ n : ℕ, k = n + S n) ∨ (∃ n : ℕ, k + 1 = n + S n) := by sorry

end NUMINAMATH_CALUDE_exists_n_plus_Sn_1980_consecutive_n_plus_Sn_l305_30574


namespace NUMINAMATH_CALUDE_prob_998th_toss_heads_l305_30531

/-- A fair coin is a coin where the probability of getting heads is 1/2. -/
def fair_coin (coin : Type) : Prop :=
  ∃ (p : coin → ℝ), (∀ c, p c = 1/2) ∧ (∀ c, 0 ≤ p c ∧ p c ≤ 1)

/-- An independent event is an event whose probability is not affected by other events. -/
def independent_event (event : Type) (p : event → ℝ) : Prop :=
  ∀ (e₁ e₂ : event), p e₁ = p e₂

/-- The probability of getting heads on the 998th toss of a fair coin in a sequence of 1000 tosses. -/
theorem prob_998th_toss_heads (coin : Type) (toss : ℕ → coin) :
  fair_coin coin →
  independent_event coin (λ c => 1/2) →
  (λ c => 1/2) (toss 998) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_998th_toss_heads_l305_30531


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l305_30505

theorem ellipse_hyperbola_product (A B : ℝ) 
  (h1 : B^2 - A^2 = 25)
  (h2 : A^2 + B^2 = 64) :
  |A * B| = Real.sqrt 867.75 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l305_30505


namespace NUMINAMATH_CALUDE_intersection_complement_A_and_B_l305_30580

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- Define the complement of A in ℝ
def complementA : Set ℝ := {x : ℝ | x ≤ 3}

-- State the theorem
theorem intersection_complement_A_and_B :
  (complementA ∩ B) = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_A_and_B_l305_30580


namespace NUMINAMATH_CALUDE_election_winner_votes_l305_30567

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 62 / 100 →
  vote_difference = 324 →
  (winner_percentage * total_votes).num = total_votes * winner_percentage.num →
  (winner_percentage * total_votes).num - ((1 - winner_percentage) * total_votes).num = vote_difference →
  (winner_percentage * total_votes).num = 837 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l305_30567


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_two_l305_30592

/-- 
Given a system of equations:
  ax + y - 1 = 0
  4x + ay - 2 = 0
If there are infinitely many solutions, then a = 2.
-/
theorem infinite_solutions_imply_a_equals_two (a : ℝ) :
  (∀ x y : ℝ, a * x + y - 1 = 0 ∧ 4 * x + a * y - 2 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    a * x₁ + y₁ - 1 = 0 ∧ 4 * x₁ + a * y₁ - 2 = 0 ∧
    a * x₂ + y₂ - 1 = 0 ∧ 4 * x₂ + a * y₂ - 2 = 0) →
  a = 2 :=
by sorry


end NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_two_l305_30592


namespace NUMINAMATH_CALUDE_corner_rectangles_area_sum_l305_30557

/-- Given a 2019x2019 square divided into 9 rectangles, with the central rectangle
    having dimensions 1511x1115, the sum of the areas of the four corner rectangles
    is 1832128. -/
theorem corner_rectangles_area_sum (square_side : ℕ) (central_length central_width : ℕ) :
  square_side = 2019 →
  central_length = 1511 →
  central_width = 1115 →
  4 * ((square_side - central_length) * (square_side - central_width)) = 1832128 :=
by sorry

end NUMINAMATH_CALUDE_corner_rectangles_area_sum_l305_30557


namespace NUMINAMATH_CALUDE_one_true_related_proposition_l305_30510

theorem one_true_related_proposition :
  let P : ℝ → ℝ → Prop := fun a b => a^2 + b^2 = 0
  let Q : ℝ → ℝ → Prop := fun a b => a^2 - b^2 = 0
  let original := ∀ a b : ℝ, P a b → Q a b
  let contrapositive := ∀ a b : ℝ, ¬(Q a b) → ¬(P a b)
  let inverse := ∀ a b : ℝ, ¬(P a b) → ¬(Q a b)
  let converse := ∀ a b : ℝ, Q a b → P a b
  (original ∧ contrapositive ∧ ¬inverse ∧ ¬converse) ∨
  (original ∧ ¬contrapositive ∧ inverse ∧ ¬converse) ∨
  (original ∧ ¬contrapositive ∧ ¬inverse ∧ converse) ∨
  (¬original ∧ contrapositive ∧ ¬inverse ∧ ¬converse) ∨
  (¬original ∧ ¬contrapositive ∧ inverse ∧ ¬converse) ∨
  (¬original ∧ ¬contrapositive ∧ ¬inverse ∧ converse) :=
by
  sorry


end NUMINAMATH_CALUDE_one_true_related_proposition_l305_30510


namespace NUMINAMATH_CALUDE_exist_four_digit_square_sum_l305_30542

/-- A four-digit number that is equal to the square of the sum of its first two digits and last two digits. -/
def IsFourDigitSquareSum (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 100 ∧
    n = 100 * a + b ∧ n = (a + b)^2

/-- There exist at least three distinct four-digit numbers that are equal to the square of the sum of their first two digits and last two digits. -/
theorem exist_four_digit_square_sum : 
  ∃ (n₁ n₂ n₃ : ℕ), n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ 
    IsFourDigitSquareSum n₁ ∧ IsFourDigitSquareSum n₂ ∧ IsFourDigitSquareSum n₃ := by
  sorry

end NUMINAMATH_CALUDE_exist_four_digit_square_sum_l305_30542


namespace NUMINAMATH_CALUDE_other_number_proof_l305_30537

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 6300)
  (h2 : Nat.gcd a b = 15)
  (h3 : a = 210) : 
  b = 450 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l305_30537


namespace NUMINAMATH_CALUDE_locus_of_point_on_moving_segment_l305_30562

/-- 
Given two perpendicular lines with moving points M and N, where the distance MN 
remains constant, and P is an arbitrary point on segment MN, prove that the locus 
of points P(x,y) forms an ellipse described by the equation x²/a² + y²/b² = 1, 
where a and b are constants related to the distance MN.
-/
theorem locus_of_point_on_moving_segment (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (M N P : ℝ × ℝ) (dist_MN : ℝ),
    (∀ t : ℝ, ∃ (Mt Nt : ℝ × ℝ), 
      (Mt.1 = 0 ∧ Nt.2 = 0) ∧  -- M and N move on perpendicular lines
      (Mt.2 - Nt.2)^2 + (Mt.1 - Nt.1)^2 = dist_MN^2 ∧  -- constant distance MN
      (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 
        P = (s * Mt.1 + (1 - s) * Nt.1, s * Mt.2 + (1 - s) * Nt.2))) →  -- P on segment MN
    P.1^2 / a^2 + P.2^2 / b^2 = 1  -- locus is an ellipse
  := by sorry

end NUMINAMATH_CALUDE_locus_of_point_on_moving_segment_l305_30562


namespace NUMINAMATH_CALUDE_chord_intersection_triangles_l305_30585

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of chords is the number of ways to choose 2 points from n points -/
def num_chords : ℕ := n.choose 2

/-- The number of intersection points is the number of ways to choose 4 points from n points -/
def num_intersections : ℕ := n.choose 4

/-- The number of triangles is the number of ways to choose 3 intersection points -/
def num_triangles : ℕ := num_intersections.choose 3

/-- Theorem stating the number of triangles formed by chord intersections -/
theorem chord_intersection_triangles :
  num_triangles = 1524180 :=
sorry

end NUMINAMATH_CALUDE_chord_intersection_triangles_l305_30585


namespace NUMINAMATH_CALUDE_survey_total_students_l305_30558

theorem survey_total_students :
  let mac_preference : ℕ := 60
  let both_preference : ℕ := mac_preference / 3
  let no_preference : ℕ := 90
  let windows_preference : ℕ := 40
  mac_preference + both_preference + no_preference + windows_preference = 210 := by
sorry

end NUMINAMATH_CALUDE_survey_total_students_l305_30558


namespace NUMINAMATH_CALUDE_triangle_area_l305_30511

/-- Given a triangle with perimeter 42 cm and inradius 5.0 cm, its area is 105 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 42 → inradius = 5 → area = perimeter / 2 * inradius → area = 105 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l305_30511


namespace NUMINAMATH_CALUDE_stick_markings_l305_30522

/-- The number of unique markings on a one-foot stick marked in both 1/4 and 1/5 portions -/
def num_markings : ℕ := 9

/-- The set of markings for 1/4 portions -/
def quarter_markings : Set ℚ :=
  {0, 1/4, 1/2, 3/4, 1}

/-- The set of markings for 1/5 portions -/
def fifth_markings : Set ℚ :=
  {0, 1/5, 2/5, 3/5, 4/5, 1}

/-- The theorem stating that the number of unique markings is 9 -/
theorem stick_markings :
  (quarter_markings ∪ fifth_markings).ncard = num_markings :=
sorry

end NUMINAMATH_CALUDE_stick_markings_l305_30522


namespace NUMINAMATH_CALUDE_c_share_is_36_l305_30578

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the total ox-months for a usage -/
def oxMonths (u : Usage) : ℕ := u.oxen * u.months

/-- Represents the rental situation -/
structure RentalSituation where
  usageA : Usage
  usageB : Usage
  usageC : Usage
  totalRent : ℚ

/-- The specific rental situation from the problem -/
def problemSituation : RentalSituation := {
  usageA := { oxen := 10, months := 7 }
  usageB := { oxen := 12, months := 5 }
  usageC := { oxen := 15, months := 3 }
  totalRent := 140
}

/-- Calculates C's share of the rent -/
def cShare (s : RentalSituation) : ℚ :=
  let totalUsage := oxMonths s.usageA + oxMonths s.usageB + oxMonths s.usageC
  let costPerOxMonth := s.totalRent / totalUsage
  (oxMonths s.usageC : ℚ) * costPerOxMonth

/-- Theorem stating that C's share in the problem situation is 36 -/
theorem c_share_is_36 : cShare problemSituation = 36 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_36_l305_30578


namespace NUMINAMATH_CALUDE_lisa_quiz_goal_l305_30534

theorem lisa_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 3/4 →
  completed_quizzes = 40 →
  current_as = 26 →
  ∃ (max_non_as : ℕ), 
    max_non_as = 1 ∧
    (current_as + (total_quizzes - completed_quizzes - max_non_as) : ℚ) / total_quizzes ≥ goal_percentage ∧
    ∀ (n : ℕ), n > max_non_as →
      (current_as + (total_quizzes - completed_quizzes - n) : ℚ) / total_quizzes < goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_lisa_quiz_goal_l305_30534


namespace NUMINAMATH_CALUDE_kitchen_tile_comparison_l305_30535

theorem kitchen_tile_comparison : 
  let area_figure1 : ℝ := π / 3 - Real.sqrt 3 / 4
  let area_figure2 : ℝ := Real.sqrt 3 / 2 - π / 6
  area_figure1 > area_figure2 := by
sorry

end NUMINAMATH_CALUDE_kitchen_tile_comparison_l305_30535


namespace NUMINAMATH_CALUDE_congruence_solution_l305_30572

theorem congruence_solution (x : ℤ) : 
  (10 * x + 3) % 18 = 7 % 18 → 
  ∃ (a m : ℕ), 
    0 < m ∧ 
    0 < a ∧ 
    a < m ∧
    x % m = a % m ∧
    a = 4 ∧ 
    m = 9 ∧
    a + m = 13 := by
  sorry

#check congruence_solution

end NUMINAMATH_CALUDE_congruence_solution_l305_30572


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l305_30581

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (choose total_republicans subcommittee_republicans) * (choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l305_30581


namespace NUMINAMATH_CALUDE_combined_value_l305_30568

def i : ℕ := 2  -- The only prime even integer from 2 to √500

def k : ℕ := 44  -- Sum of even integers from 8 to √200 (8 + 10 + 12 + 14)

def j : ℕ := 23  -- Sum of prime odd integers from 5 to √133 (5 + 7 + 11)

theorem combined_value : 2 * i - k + 3 * j = 29 := by
  sorry

end NUMINAMATH_CALUDE_combined_value_l305_30568


namespace NUMINAMATH_CALUDE_carol_invitation_packs_l305_30545

def invitations_per_pack : ℕ := 3
def friends_to_invite : ℕ := 9
def extra_invitations : ℕ := 3

theorem carol_invitation_packs :
  (friends_to_invite + extra_invitations) / invitations_per_pack = 4 := by
  sorry

end NUMINAMATH_CALUDE_carol_invitation_packs_l305_30545


namespace NUMINAMATH_CALUDE_parrots_per_cage_l305_30588

theorem parrots_per_cage (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 4 →
  parakeets_per_cage = 2 →
  total_birds = 40 →
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l305_30588


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l305_30519

theorem absolute_value_inequality (x : ℝ) : |x - 1| < 2 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l305_30519


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_point_one_l305_30529

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 0.1 is -0.1 -/
theorem opposite_of_point_one : opposite 0.1 = -0.1 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_point_one_l305_30529


namespace NUMINAMATH_CALUDE_contract_copies_per_person_l305_30524

theorem contract_copies_per_person 
  (contract_pages : ℕ) 
  (total_pages : ℕ) 
  (num_people : ℕ) 
  (h1 : contract_pages = 20) 
  (h2 : total_pages = 360) 
  (h3 : num_people = 9) :
  (total_pages / contract_pages) / num_people = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_contract_copies_per_person_l305_30524


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l305_30500

theorem pet_store_siamese_cats :
  let initial_house_cats : ℝ := 5.0
  let added_cats : ℝ := 10.0
  let total_cats_after : ℕ := 28
  let initial_siamese_cats : ℝ := initial_house_cats + added_cats + total_cats_after - (initial_house_cats + added_cats)
  initial_siamese_cats = 13 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l305_30500


namespace NUMINAMATH_CALUDE_distribution_schemes_l305_30507

/-- The number of ways to distribute students among projects -/
def distribute_students (n_students : ℕ) (n_projects : ℕ) : ℕ :=
  -- Number of ways to choose 2 students from n_students
  (n_students.choose 2) * 
  -- Number of ways to permute n_projects
  (n_projects.factorial)

/-- Theorem stating the number of distribution schemes -/
theorem distribution_schemes :
  distribute_students 5 4 = 240 :=
sorry

end NUMINAMATH_CALUDE_distribution_schemes_l305_30507


namespace NUMINAMATH_CALUDE_five_student_committees_l305_30521

theorem five_student_committees (n k : ℕ) (h1 : n = 8) (h2 : k = 5) : 
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_l305_30521


namespace NUMINAMATH_CALUDE_glycerin_percentage_after_dilution_l305_30583

def initial_glycerin_percentage : ℝ := 0.9
def initial_volume : ℝ := 4
def added_water : ℝ := 0.8

theorem glycerin_percentage_after_dilution :
  let initial_glycerin := initial_glycerin_percentage * initial_volume
  let final_volume := initial_volume + added_water
  let final_glycerin_percentage := initial_glycerin / final_volume
  final_glycerin_percentage = 0.75 := by sorry

end NUMINAMATH_CALUDE_glycerin_percentage_after_dilution_l305_30583


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l305_30527

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- For any real number x, the point (x^2 + 1, -4) is in the fourth quadrant -/
theorem point_in_fourth_quadrant (x : ℝ) :
  in_fourth_quadrant (x^2 + 1, -4) := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l305_30527


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l305_30559

/-- The x-coordinate of the point on the x-axis equidistant from A(-3, 0) and B(3, 5) is 25/12 -/
theorem equidistant_point_x_coordinate :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (3, 5)
  ∃ x : ℝ, x = 25 / 12 ∧
    (x - A.1) ^ 2 + A.2 ^ 2 = (x - B.1) ^ 2 + (0 - B.2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l305_30559


namespace NUMINAMATH_CALUDE_profit_maximization_and_threshold_l305_30593

def price (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x < 40 then x + 45
  else if 40 ≤ x ∧ x ≤ 70 then 85
  else 0

def dailySales (x : ℕ) : ℕ := 150 - 2 * x

def costPrice : ℕ := 30

def dailyProfit (x : ℕ) : ℤ :=
  if 1 ≤ x ∧ x < 40 then -2 * x^2 + 120 * x + 2250
  else if 40 ≤ x ∧ x ≤ 70 then -110 * x + 8250
  else 0

theorem profit_maximization_and_threshold (x : ℕ) :
  (∀ x, 1 ≤ x ∧ x ≤ 70 → dailyProfit x ≤ dailyProfit 30) ∧
  dailyProfit 30 = 4050 ∧
  (Finset.filter (fun x => dailyProfit x ≥ 3250) (Finset.range 70)).card = 36 := by
  sorry


end NUMINAMATH_CALUDE_profit_maximization_and_threshold_l305_30593


namespace NUMINAMATH_CALUDE_area_of_union_equals_20_5_l305_30556

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point about the line y = x -/
def reflect (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- Calculates the area of a triangle given its three vertices using the shoelace formula -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- The main theorem stating the area of the union of the original and reflected triangles -/
theorem area_of_union_equals_20_5 :
  let A : Point := { x := 3, y := 4 }
  let B : Point := { x := 5, y := -2 }
  let C : Point := { x := 7, y := 3 }
  let A' := reflect A
  let B' := reflect B
  let C' := reflect C
  triangleArea A B C + triangleArea A' B' C' = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_union_equals_20_5_l305_30556


namespace NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l305_30501

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 3 → ¬(∃ k : ℕ, n! = (k + 1) * (k + 2) * (k + 3) * (k + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l305_30501


namespace NUMINAMATH_CALUDE_sum_of_a_values_l305_30566

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := 4 * x^2 + a * x + 8 * x + 9

-- Define the condition for the equation to have only one solution
def has_one_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, quadratic_equation a x = 0

-- Define the set of 'a' values that satisfy the condition
def a_values : Set ℝ := {a | has_one_solution a}

-- State the theorem
theorem sum_of_a_values :
  ∃ a₁ a₂ : ℝ, a₁ ∈ a_values ∧ a₂ ∈ a_values ∧ a₁ ≠ a₂ ∧ a₁ + a₂ = -16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l305_30566


namespace NUMINAMATH_CALUDE_range_of_m_l305_30565

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - (m+3)*x + m^2 = 0}

theorem range_of_m : 
  ∀ m : ℝ, (A ∪ (Set.univ \ B m) = Set.univ) ↔ (m < -1 ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l305_30565


namespace NUMINAMATH_CALUDE_school_population_theorem_l305_30517

theorem school_population_theorem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 150 →
  boys + girls = total →
  girls = (boys * total) / 100 →
  boys = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_population_theorem_l305_30517


namespace NUMINAMATH_CALUDE_pair_count_theorem_l305_30547

def count_pairs (n : ℕ) : ℕ :=
  (n - 50) * (n - 51) / 2 + 1275

theorem pair_count_theorem :
  count_pairs 100 = 2500 :=
sorry

end NUMINAMATH_CALUDE_pair_count_theorem_l305_30547


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l305_30509

theorem binomial_expansion_sum (m : ℝ) : 
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, 
    (∀ x : ℝ, (1 + m * x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6) ∧
    (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64)) → 
  m = 1 ∨ m = -3 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l305_30509


namespace NUMINAMATH_CALUDE_odd_rolls_probability_l305_30575

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has a success probability of p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of rolls of the die -/
def num_rolls : ℕ := 7

/-- The number of odd outcomes we're interested in -/
def num_odd : ℕ := 5

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

theorem odd_rolls_probability :
  binomial_probability num_rolls num_odd prob_odd = 21/128 := by
  sorry

end NUMINAMATH_CALUDE_odd_rolls_probability_l305_30575


namespace NUMINAMATH_CALUDE_simplify_expression_l305_30561

theorem simplify_expression (m n : ℝ) : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l305_30561


namespace NUMINAMATH_CALUDE_milford_lake_algae_count_l305_30533

theorem milford_lake_algae_count (current : ℕ) (increase : ℕ) (original : ℕ)
  (h1 : current = 3263)
  (h2 : increase = 2454)
  (h3 : current = original + increase) :
  original = 809 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_count_l305_30533


namespace NUMINAMATH_CALUDE_infinite_sum_equals_three_fortieths_l305_30506

/-- The sum of the infinite series n / (n^4 + 16) from n = 1 to infinity equals 3/40 -/
theorem infinite_sum_equals_three_fortieths :
  (∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 + 16)) = 3/40 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_three_fortieths_l305_30506


namespace NUMINAMATH_CALUDE_absolute_value_four_l305_30530

theorem absolute_value_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_four_l305_30530


namespace NUMINAMATH_CALUDE_population_growth_proof_l305_30528

/-- The annual population growth rate -/
def annual_growth_rate : ℝ := 0.10

/-- The population after 2 years -/
def final_population : ℝ := 18150

/-- The present population of the town -/
def present_population : ℝ := 15000

/-- Theorem stating that the present population results in the final population after 2 years of growth -/
theorem population_growth_proof :
  present_population * (1 + annual_growth_rate)^2 = final_population :=
by sorry

end NUMINAMATH_CALUDE_population_growth_proof_l305_30528


namespace NUMINAMATH_CALUDE_cost_per_page_is_five_l305_30520

/-- The cost per page in cents when buying notebooks -/
def cost_per_page (num_notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (num_notebooks * pages_per_notebook)

/-- Theorem: The cost per page is 5 cents when buying 2 notebooks with 50 pages each for $5 -/
theorem cost_per_page_is_five :
  cost_per_page 2 50 5 = 5 := by
  sorry

#eval cost_per_page 2 50 5

end NUMINAMATH_CALUDE_cost_per_page_is_five_l305_30520


namespace NUMINAMATH_CALUDE_parabola_properties_l305_30582

/-- Represents a parabola of the form y = ax^2 -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Compares the steepness of two parabolas at a given x -/
def steeper_at (p1 p2 : Parabola) (x : ℝ) : Prop :=
  p1.a * x^2 > p2.a * x^2

/-- A parabola p1 is considered steeper than p2 if it's steeper for all non-zero x -/
def steeper (p1 p2 : Parabola) : Prop :=
  ∀ x ≠ 0, steeper_at p1 p2 x

/-- A parabola p approaches the x-axis as its 'a' approaches 0 -/
def approaches_x_axis (p : Parabola → Prop) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ q : Parabola, q.a < δ → p q → ∀ x, |q.a * x^2| < ε

theorem parabola_properties :
  ∀ p : Parabola,
    (0 < p.a ∧ p.a < 1 → steeper {a := 1, h := by norm_num} p) ∧
    (p.a > 1 → steeper p {a := 1, h := by norm_num}) ∧
    (approaches_x_axis (λ q ↦ q.a < p.a)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l305_30582


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l305_30539

theorem simple_interest_calculation (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 2323 → rate = 8 → time = 5 →
  (principal * rate * time) / 100 = 1861.84 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l305_30539


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l305_30564

theorem fraction_sum_simplification : (3 : ℚ) / 462 + 28 / 42 = 311 / 462 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l305_30564


namespace NUMINAMATH_CALUDE_z_real_z_pure_imaginary_z_second_quadrant_l305_30536

/-- Definition of the complex number z in terms of real number m -/
def z (m : ℝ) : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 + 3*m + 2 : ℝ) * Complex.I

/-- z is a real number if and only if m = -1 or m = -2 -/
theorem z_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

/-- z is a pure imaginary number if and only if m = 3 -/
theorem z_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 3 := by sorry

/-- z is in the second quadrant of the complex plane if and only if -1 < m < 3 -/
theorem z_second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -1 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_z_real_z_pure_imaginary_z_second_quadrant_l305_30536


namespace NUMINAMATH_CALUDE_zero_exponent_equals_one_l305_30591

theorem zero_exponent_equals_one (r : ℚ) (h : r ≠ 0) : r ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_equals_one_l305_30591


namespace NUMINAMATH_CALUDE_two_red_balls_in_bag_l305_30586

/-- Represents the contents of a bag of balls -/
structure BagOfBalls where
  redBalls : ℕ
  yellowBalls : ℕ

/-- The probability of selecting a yellow ball given another yellow ball was selected -/
def probYellowGivenYellow (bag : BagOfBalls) : ℚ :=
  (bag.yellowBalls - 1) / (bag.redBalls + bag.yellowBalls - 1)

theorem two_red_balls_in_bag :
  ∀ (bag : BagOfBalls),
    bag.yellowBalls = 3 →
    probYellowGivenYellow bag = 1/2 →
    bag.redBalls = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_red_balls_in_bag_l305_30586


namespace NUMINAMATH_CALUDE_lawn_length_is_four_l305_30516

-- Define the lawn's properties
def lawn_area : ℝ := 20
def lawn_width : ℝ := 5

-- Theorem statement
theorem lawn_length_is_four :
  ∃ (length : ℝ), length * lawn_width = lawn_area ∧ length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_four_l305_30516
