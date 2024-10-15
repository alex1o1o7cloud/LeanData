import Mathlib

namespace NUMINAMATH_CALUDE_max_height_triangle_DEF_l2944_294477

/-- Triangle DEF with side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ

/-- The maximum possible height of a table constructed from a triangle -/
def max_table_height (t : Triangle) : ℝ := sorry

/-- The given triangle DEF -/
def triangle_DEF : Triangle :=
  { DE := 25,
    EF := 28,
    FD := 33 }

theorem max_height_triangle_DEF :
  max_table_height triangle_DEF = 60 * Real.sqrt 129 / 61 := by sorry

end NUMINAMATH_CALUDE_max_height_triangle_DEF_l2944_294477


namespace NUMINAMATH_CALUDE_expected_net_profit_l2944_294498

/-- The expected value of net profit from selling one electronic product -/
theorem expected_net_profit (purchase_price : ℝ) (pass_rate : ℝ) (profit_qualified : ℝ) (loss_defective : ℝ)
  (h1 : purchase_price = 10)
  (h2 : pass_rate = 0.95)
  (h3 : profit_qualified = 2)
  (h4 : loss_defective = 10) :
  profit_qualified * pass_rate + (-loss_defective) * (1 - pass_rate) = 1.4 := by
sorry

end NUMINAMATH_CALUDE_expected_net_profit_l2944_294498


namespace NUMINAMATH_CALUDE_stellas_dolls_count_l2944_294433

theorem stellas_dolls_count : 
  ∀ (num_dolls : ℕ),
  (num_dolls : ℝ) * 5 + 2 * 15 + 5 * 4 - 40 = 25 →
  num_dolls = 3 := by
sorry

end NUMINAMATH_CALUDE_stellas_dolls_count_l2944_294433


namespace NUMINAMATH_CALUDE_basket_balls_count_l2944_294480

theorem basket_balls_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob : ℚ) : 
  red = 8 →
  prob = 2/5 →
  total = red + yellow →
  prob = red / total →
  yellow = 12 := by
sorry

end NUMINAMATH_CALUDE_basket_balls_count_l2944_294480


namespace NUMINAMATH_CALUDE_ricky_age_solution_l2944_294403

def ricky_age_problem (rickys_age : ℕ) (fathers_age : ℕ) : Prop :=
  fathers_age = 45 ∧
  rickys_age + 5 = (1 / 5 : ℚ) * (fathers_age + 5 : ℚ) + 5

theorem ricky_age_solution :
  ∃ (rickys_age : ℕ), ricky_age_problem rickys_age 45 ∧ rickys_age = 10 :=
sorry

end NUMINAMATH_CALUDE_ricky_age_solution_l2944_294403


namespace NUMINAMATH_CALUDE_paige_homework_pages_l2944_294434

/-- Given the total number of homework problems, the number of finished problems,
    and the number of problems per page, calculate the number of remaining pages. -/
def remaining_pages (total_problems : ℕ) (finished_problems : ℕ) (problems_per_page : ℕ) : ℕ :=
  (total_problems - finished_problems) / problems_per_page

/-- Theorem stating that for Paige's homework scenario, the number of remaining pages is 7. -/
theorem paige_homework_pages : remaining_pages 110 47 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_paige_homework_pages_l2944_294434


namespace NUMINAMATH_CALUDE_function_value_at_2009_l2944_294426

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (2 * x * y + 3) + 3 * f (x + y) - 3 * f x = -6 * x

/-- The main theorem stating that for a function satisfying the functional equation, f(2009) = 4021 -/
theorem function_value_at_2009 (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2009 = 4021 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2009_l2944_294426


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2944_294478

/-- The equation of a circle with center (0, b) that is tangent to the line y = 2x + 1 at point (1, 3) -/
theorem circle_equation_tangent_to_line (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + 1 → (x - 1)^2 + (y - 3)^2 ≠ 0) →
  (1 : ℝ)^2 + (3 - b)^2 = (0 - 1)^2 + ((2 * 0 + 1) - b)^2 →
  (∀ x y : ℝ, (x : ℝ)^2 + (y - 7/2)^2 = 5/4 ↔ (x - 0)^2 + (y - b)^2 = (1 - 0)^2 + (3 - b)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2944_294478


namespace NUMINAMATH_CALUDE_min_force_to_submerge_cube_l2944_294435

/-- Minimum force required to submerge a cube -/
theorem min_force_to_submerge_cube 
  (cube_volume : Real) 
  (cube_density : Real) 
  (water_density : Real) 
  (gravity : Real) :
  cube_volume = 1e-5 →  -- 10 cm³ = 1e-5 m³
  cube_density = 700 →
  water_density = 1000 →
  gravity = 10 →
  (water_density - cube_density) * cube_volume * gravity = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_min_force_to_submerge_cube_l2944_294435


namespace NUMINAMATH_CALUDE_converse_square_sum_nonzero_l2944_294479

theorem converse_square_sum_nonzero (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_converse_square_sum_nonzero_l2944_294479


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l2944_294446

open Set

theorem intersection_complement_problem (U M N : Set ℕ) : 
  U = {0, 1, 2, 3, 4, 5} →
  M = {0, 3, 5} →
  N = {1, 4, 5} →
  M ∩ (U \ N) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l2944_294446


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l2944_294413

/-- A geometric sequence with positive integer terms -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = (a n : ℚ) * r

theorem geometric_sequence_second_term
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 0 = 5)
  (h_fourth : a 3 = 480) :
  a 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l2944_294413


namespace NUMINAMATH_CALUDE_joan_balloons_l2944_294489

/-- Joan and Melanie's blue balloons problem -/
theorem joan_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (total_balloons : ℕ)
    (h1 : melanie_balloons = 41)
    (h2 : total_balloons = 81)
    (h3 : joan_balloons + melanie_balloons = total_balloons) :
  joan_balloons = 40 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l2944_294489


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2944_294425

theorem min_value_sum_of_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 10) :
  2/p + 3/q + 5/r + 7/s + 11/t + 13/u ≥ 23.875 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ), 
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 10 ∧
    2/p' + 3/q' + 5/r' + 7/s' + 11/t' + 13/u' = 23.875 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2944_294425


namespace NUMINAMATH_CALUDE_gcd_equality_implies_equal_l2944_294460

theorem gcd_equality_implies_equal (a b c : ℕ+) :
  a + Nat.gcd a b = b + Nat.gcd b c ∧
  b + Nat.gcd b c = c + Nat.gcd c a →
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_gcd_equality_implies_equal_l2944_294460


namespace NUMINAMATH_CALUDE_expression_evaluation_l2944_294487

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2944_294487


namespace NUMINAMATH_CALUDE_perpendicular_when_a_neg_one_passes_through_zero_one_l2944_294471

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

-- Define perpendicularity of two lines given their slopes
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem for statement A
theorem perpendicular_when_a_neg_one :
  perpendicular (-((-1)^2 + (-1) + 1)) 1 :=
sorry

-- Theorem for statement C
theorem passes_through_zero_one (a : ℝ) :
  line_l a 0 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_when_a_neg_one_passes_through_zero_one_l2944_294471


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2944_294431

theorem tangent_line_to_circle (m : ℝ) :
  (∀ x y : ℝ, 3 * x - 4 * y - 6 = 0 →
    (x^2 + y^2 - 2*y + m = 0 →
      ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 2*y₀ + m = 0 ∧
        3 * x₀ - 4 * y₀ - 6 = 0 ∧
        ∀ (x' y' : ℝ), x'^2 + y'^2 - 2*y' + m = 0 →
          (x' - x₀)^2 + (y' - y₀)^2 > 0)) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2944_294431


namespace NUMINAMATH_CALUDE_largest_negative_integer_solution_l2944_294474

theorem largest_negative_integer_solution :
  ∃ (x : ℝ), x = -1 ∧ 
  x < 0 ∧
  |x - 1| > 1 ∧
  (x - 2) / x > 0 ∧
  (x - 2) / x > |x - 1| ∧
  ∀ (y : ℤ), y < 0 → 
    (y < x ∨ ¬(|y - 1| > 1) ∨ ¬((y - 2) / y > 0) ∨ ¬((y - 2) / y > |y - 1|)) :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_solution_l2944_294474


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l2944_294492

theorem sin_cos_equation_solution :
  ∃ x : ℝ, x = π / 14 ∧ Real.sin (3 * x) * Real.sin (4 * x) = Real.cos (3 * x) * Real.cos (4 * x) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l2944_294492


namespace NUMINAMATH_CALUDE_f_nonnegative_condition_f_two_zeros_condition_l2944_294441

/-- The function f(x) defined as |x^2 - 1| + x^2 + kx -/
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

theorem f_nonnegative_condition (k : ℝ) :
  (∀ x > 0, f k x ≥ 0) ↔ k ≥ -1 := by sorry

theorem f_two_zeros_condition (k : ℝ) (x₁ x₂ : ℝ) :
  (0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (-7/2 < k ∧ k < -1 ∧ 2 < 1/x₁ + 1/x₂ ∧ 1/x₁ + 1/x₂ < 4) := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_condition_f_two_zeros_condition_l2944_294441


namespace NUMINAMATH_CALUDE_proportion_problem_l2944_294405

theorem proportion_problem (hours_per_day : ℝ) (h : hours_per_day = 24) :
  ∃ x : ℝ, (24 : ℝ) / (6 / hours_per_day) = x / 8 ∧ x = 768 :=
by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l2944_294405


namespace NUMINAMATH_CALUDE_bertha_age_difference_l2944_294456

structure Grandparents where
  arthur : ℕ
  bertha : ℕ
  christoph : ℕ
  dolores : ℕ

def is_valid_grandparents (g : Grandparents) : Prop :=
  (max g.arthur (max g.bertha (max g.christoph g.dolores))) - 
  (min g.arthur (min g.bertha (min g.christoph g.dolores))) = 4 ∧
  g.arthur = g.bertha + 2 ∧
  g.christoph = g.dolores + 2 ∧
  g.bertha < g.dolores

theorem bertha_age_difference (g : Grandparents) (h : is_valid_grandparents g) :
  g.bertha + 2 = (g.arthur + g.bertha + g.christoph + g.dolores) / 4 := by
  sorry

#check bertha_age_difference

end NUMINAMATH_CALUDE_bertha_age_difference_l2944_294456


namespace NUMINAMATH_CALUDE_g_at_negative_two_l2944_294452

def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 5 * x^2 - x + 8

theorem g_at_negative_two : g (-2) = -186 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l2944_294452


namespace NUMINAMATH_CALUDE_bert_equals_kameron_in_40_days_l2944_294461

/-- The number of days required for Bert to have the same number of kangaroos as Kameron -/
def days_to_equal_kangaroos (kameron_kangaroos : ℕ) (bert_kangaroos : ℕ) (bert_buying_rate : ℕ) : ℕ :=
  (kameron_kangaroos - bert_kangaroos) / bert_buying_rate

/-- Proof that it takes 40 days for Bert to have the same number of kangaroos as Kameron -/
theorem bert_equals_kameron_in_40_days :
  days_to_equal_kangaroos 100 20 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bert_equals_kameron_in_40_days_l2944_294461


namespace NUMINAMATH_CALUDE_binomial_15_4_l2944_294475

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l2944_294475


namespace NUMINAMATH_CALUDE_value_of_a_l2944_294430

theorem value_of_a (a b c : ℤ) (h1 : a + b = 10) (h2 : b + c = 8) (h3 : c = 4) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2944_294430


namespace NUMINAMATH_CALUDE_stock_percentage_change_l2944_294468

theorem stock_percentage_change 
  (initial_value : ℝ) 
  (day1_decrease_rate : ℝ) 
  (day2_increase_rate : ℝ) 
  (h1 : day1_decrease_rate = 0.3) 
  (h2 : day2_increase_rate = 0.4) : 
  (initial_value - (initial_value * (1 - day1_decrease_rate) * (1 + day2_increase_rate))) / initial_value = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_change_l2944_294468


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2944_294443

/-- The repeating decimal 4.363636... -/
def repeating_decimal : ℚ := 4 + 36 / 99

/-- The fraction 144/33 -/
def fraction : ℚ := 144 / 33

/-- Theorem stating that the repeating decimal 4.363636... is equal to the fraction 144/33 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2944_294443


namespace NUMINAMATH_CALUDE_psychological_survey_selection_l2944_294467

theorem psychological_survey_selection (boys girls selected : ℕ) : 
  boys = 4 → girls = 2 → selected = 4 →
  (Nat.choose (boys + girls) selected) - (Nat.choose boys selected) = 14 :=
by sorry

end NUMINAMATH_CALUDE_psychological_survey_selection_l2944_294467


namespace NUMINAMATH_CALUDE_equation_solution_l2944_294409

theorem equation_solution : ∃ k : ℤ, 2^4 - 6 = 3^3 + k ∧ k = -17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2944_294409


namespace NUMINAMATH_CALUDE_no_integer_solution_implies_k_range_l2944_294490

theorem no_integer_solution_implies_k_range (k : ℝ) : 
  (∀ x : ℤ, ¬((k * x - k^2 - 4) * (x - 4) < 0)) → 
  1 ≤ k ∧ k ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_no_integer_solution_implies_k_range_l2944_294490


namespace NUMINAMATH_CALUDE_correct_dimes_calculation_l2944_294497

/-- Represents the number of dimes each sibling has -/
structure Dimes where
  barry : ℕ
  dan : ℕ
  emily : ℕ
  frank : ℕ

/-- Calculates the correct number of dimes for each sibling based on the given conditions -/
def calculate_dimes : Dimes :=
  let barry_dimes := 1000 / 10  -- $10.00 worth of dimes
  let dan_initial := barry_dimes / 2
  let dan_final := dan_initial + 2
  let emily_dimes := 2 * dan_initial
  let frank_dimes := emily_dimes - 7
  { barry := barry_dimes
  , dan := dan_final
  , emily := emily_dimes
  , frank := frank_dimes }

/-- Theorem stating that the calculated dimes match the expected values -/
theorem correct_dimes_calculation : 
  let dimes := calculate_dimes
  dimes.barry = 100 ∧ 
  dimes.dan = 52 ∧ 
  dimes.emily = 100 ∧ 
  dimes.frank = 93 := by
  sorry

end NUMINAMATH_CALUDE_correct_dimes_calculation_l2944_294497


namespace NUMINAMATH_CALUDE_power_mul_l2944_294412

theorem power_mul (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

end NUMINAMATH_CALUDE_power_mul_l2944_294412


namespace NUMINAMATH_CALUDE_largest_angle_in_circle_l2944_294410

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Define a function to check if a point is inside a circle
def isInside (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if a point is on the circumference of a circle
def isOnCircumference (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if three points form a diameter of a circle
def formsDiameter (A B C : Point) (circle : Circle) : Prop := sorry

theorem largest_angle_in_circle (circle : Circle) (A B : Point) 
  (hA : isInside A circle) (hB : isInside B circle) :
  ∃ C, isOnCircumference C circle ∧ 
    (∀ D, isOnCircumference D circle → angle A B C ≥ angle A B D) ∧
    formsDiameter A B C circle := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_circle_l2944_294410


namespace NUMINAMATH_CALUDE_cards_distribution_l2944_294465

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) 
  (h1 : total_cards = 60) (h2 : total_people = 9) : 
  (total_people - (total_cards % total_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2944_294465


namespace NUMINAMATH_CALUDE_segment_length_sum_l2944_294417

theorem segment_length_sum (a : ℝ) : 
  let point1 := (3 * a, 2 * a - 5)
  let point2 := (5, -2)
  let distance := Real.sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2)
  distance = 3 * Real.sqrt 5 →
  ∃ (a1 a2 : ℝ), a1 ≠ a2 ∧ 
    (∀ x : ℝ, Real.sqrt ((3*x - 5)^2 + (2*x - 3)^2) = 3 * Real.sqrt 5 ↔ x = a1 ∨ x = a2) ∧
    a1 + a2 = 3.231 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_sum_l2944_294417


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2944_294464

-- Define a type for lines in space
structure Line3D where
  -- You might want to add more specific properties here
  -- but for this problem, we just need to distinguish between lines

-- Define parallelism for lines in space
def parallel (l1 l2 : Line3D) : Prop :=
  -- The actual definition of parallelism would go here
  sorry

-- The theorem statement
theorem parallel_transitivity (l m n : Line3D) : 
  parallel l m → parallel l n → parallel m n := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2944_294464


namespace NUMINAMATH_CALUDE_cars_already_parked_equals_62_l2944_294432

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  totalCapacity : ℕ
  levels : ℕ
  additionalCapacity : ℕ

/-- The number of cars already parked on one level -/
def carsAlreadyParked (p : ParkingLot) : ℕ :=
  p.totalCapacity / p.levels - p.additionalCapacity

/-- Theorem stating the number of cars already parked on one level -/
theorem cars_already_parked_equals_62 (p : ParkingLot) 
    (h1 : p.totalCapacity = 425)
    (h2 : p.levels = 5)
    (h3 : p.additionalCapacity = 62) :
    carsAlreadyParked p = 62 := by
  sorry

#eval carsAlreadyParked { totalCapacity := 425, levels := 5, additionalCapacity := 62 }

end NUMINAMATH_CALUDE_cars_already_parked_equals_62_l2944_294432


namespace NUMINAMATH_CALUDE_triangle_formation_count_l2944_294472

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of checkpoints on the first track -/
def checkpoints_track1 : ℕ := 6

/-- The number of checkpoints on the second track -/
def checkpoints_track2 : ℕ := 10

/-- The number of ways to form triangles by selecting one point from the first track
    and two points from the second track -/
def triangle_formations : ℕ := checkpoints_track1 * choose checkpoints_track2 2

theorem triangle_formation_count :
  triangle_formations = 270 := by sorry

end NUMINAMATH_CALUDE_triangle_formation_count_l2944_294472


namespace NUMINAMATH_CALUDE_min_value_system_l2944_294493

theorem min_value_system (x y k : ℝ) :
  (3 * x + y ≥ 0) →
  (4 * x + 3 * y ≥ k) →
  (∀ x' y', (3 * x' + y' ≥ 0) → (4 * x' + 3 * y' ≥ k) → (2 * x' + 4 * y' ≥ 2 * x + 4 * y)) →
  (2 * x + 4 * y = -6) →
  (k ≤ 0 ∧ ∀ m : ℤ, m > 0 → ¬(k ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_system_l2944_294493


namespace NUMINAMATH_CALUDE_expression_equality_l2944_294424

theorem expression_equality (a b : ℝ) : -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2944_294424


namespace NUMINAMATH_CALUDE_boxes_with_neither_l2944_294404

theorem boxes_with_neither (total : ℕ) (crayons : ℕ) (markers : ℕ) (both : ℕ) : 
  total = 15 → crayons = 9 → markers = 6 → both = 4 →
  total - (crayons + markers - both) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l2944_294404


namespace NUMINAMATH_CALUDE_specific_truck_toll_l2944_294451

/-- Calculates the toll for a truck crossing a bridge -/
def calculate_toll (x : ℕ) (w : ℝ) (peak_hours : Bool) : ℝ :=
  let y : ℝ := if peak_hours then 2 else 0
  3.50 + 0.50 * (x - 2 : ℝ) + 0.10 * w + y

/-- Theorem: The toll for a specific truck is $8.50 -/
theorem specific_truck_toll :
  calculate_toll 5 15 true = 8.50 := by
  sorry

end NUMINAMATH_CALUDE_specific_truck_toll_l2944_294451


namespace NUMINAMATH_CALUDE_carl_lemonade_sales_l2944_294408

/-- 
Given:
- Stanley sells 4 cups of lemonade per hour
- Carl sells some cups of lemonade per hour
- Carl sold 9 more cups than Stanley in 3 hours

Prove that Carl sold 7 cups of lemonade per hour
-/
theorem carl_lemonade_sales (stanley_rate : ℕ) (carl_rate : ℕ) (hours : ℕ) (difference : ℕ) :
  stanley_rate = 4 →
  hours = 3 →
  difference = 9 →
  carl_rate * hours = stanley_rate * hours + difference →
  carl_rate = 7 :=
by
  sorry

#check carl_lemonade_sales

end NUMINAMATH_CALUDE_carl_lemonade_sales_l2944_294408


namespace NUMINAMATH_CALUDE_negation_of_implication_l2944_294483

theorem negation_of_implication (x : ℝ) : 
  (¬(x^2 = 1 → x = 1)) ↔ (x^2 ≠ 1 → x ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2944_294483


namespace NUMINAMATH_CALUDE_calculation_proof_l2944_294406

theorem calculation_proof :
  (2 * (Real.sqrt 3 - Real.sqrt 5) + 3 * (Real.sqrt 3 + Real.sqrt 5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  (-1^2 - |1 - Real.sqrt 3| + Real.rpow 8 (1/3) - (-3) * Real.sqrt 9 = 11 - Real.sqrt 3) := by
sorry


end NUMINAMATH_CALUDE_calculation_proof_l2944_294406


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2944_294416

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = { x | -1 < x ∧ x < 2 } := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2944_294416


namespace NUMINAMATH_CALUDE_max_min_A_values_l2944_294402

open Complex Real

theorem max_min_A_values (z : ℂ) (h : abs (z - I) ≤ 1) :
  let A := (z.re : ℝ) * ((abs (z - I))^2 - 1)
  ∃ (max_A min_A : ℝ), 
    (∀ z', abs (z' - I) ≤ 1 → (z'.re : ℝ) * ((abs (z' - I))^2 - 1) ≤ max_A) ∧
    (∀ z', abs (z' - I) ≤ 1 → (z'.re : ℝ) * ((abs (z' - I))^2 - 1) ≥ min_A) ∧
    max_A = 2 * Real.sqrt 3 / 9 ∧
    min_A = -2 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_min_A_values_l2944_294402


namespace NUMINAMATH_CALUDE_division_remainder_l2944_294420

theorem division_remainder (dividend quotient divisor remainder : ℕ) : 
  dividend = 507 → 
  quotient = 61 → 
  divisor = 8 → 
  dividend = divisor * quotient + remainder → 
  remainder = 19 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2944_294420


namespace NUMINAMATH_CALUDE_f_max_value_l2944_294488

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt (2 * x^3 + 7 * x^2 + 6 * x)) / (x^2 + 4 * x + 3)

theorem f_max_value :
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f x ≤ 1/2) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f x = 1/2) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l2944_294488


namespace NUMINAMATH_CALUDE_parabola_trajectory_l2944_294427

/-- The trajectory of point M given the conditions of the parabola and vector relationship -/
theorem parabola_trajectory (x y t : ℝ) : 
  let F : ℝ × ℝ := (1, 0)  -- Focus of the parabola
  let P : ℝ → ℝ × ℝ := λ t => (t^2/4, t)  -- Point on the parabola
  let M : ℝ × ℝ := (x, y)  -- Point M
  (∀ t, (P t).2^2 = 4 * (P t).1) →  -- P is on the parabola y^2 = 4x
  ((P t).1 - F.1, (P t).2 - F.2) = (2*(x - F.1), 2*(y - F.2)) →  -- FP = 2FM
  y^2 = 2*x - 1  -- Trajectory equation
:= by sorry

end NUMINAMATH_CALUDE_parabola_trajectory_l2944_294427


namespace NUMINAMATH_CALUDE_triangle_piece_count_l2944_294499

/-- Calculate the sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculate the sum of the first n natural numbers -/
def triangle_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The number of rows in the rod triangle -/
def rod_rows : ℕ := 10

/-- The number of rows in the connector triangle -/
def connector_rows : ℕ := rod_rows + 1

/-- The first term of the rod arithmetic sequence -/
def first_rod_count : ℕ := 3

/-- The common difference of the rod arithmetic sequence -/
def rod_increment : ℕ := 3

theorem triangle_piece_count : 
  arithmetic_sum first_rod_count rod_increment rod_rows + 
  triangle_number connector_rows = 231 := by
  sorry

end NUMINAMATH_CALUDE_triangle_piece_count_l2944_294499


namespace NUMINAMATH_CALUDE_percentage_difference_l2944_294481

theorem percentage_difference : (60 / 100 * 50) - (50 / 100 * 30) = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2944_294481


namespace NUMINAMATH_CALUDE_train_crossing_time_l2944_294423

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 150 → 
  train_speed_kmh = 180 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2944_294423


namespace NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_vector_ratio_l2944_294459

-- Define the point P
def P : ℝ × ℝ := (-3, 1)

-- Define the line l passing through P and intersecting x-axis at A and y-axis at B
def line_l (A B : ℝ × ℝ) : Prop :=
  A.2 = 0 ∧ B.1 = 0 ∧ ∃ t : ℝ, P = t • A + (1 - t) • B

-- Define the midpoint condition
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the vector ratio condition
def vector_ratio (P A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (2 * (P.1 - B.1), 2 * (P.2 - B.2))

-- Theorem for case I
theorem line_equation_midpoint (A B : ℝ × ℝ) :
  line_l A B → is_midpoint P A B →
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, (x - 3*y + 6 = 0) ↔ k * (x - A.1) = k * (y - A.2) :=
sorry

-- Theorem for case II
theorem line_equation_vector_ratio (A B : ℝ × ℝ) :
  line_l A B → vector_ratio P A B →
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, (x - 6*y + 9 = 0) ↔ k * (x - A.1) = k * (y - A.2) :=
sorry

end NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_vector_ratio_l2944_294459


namespace NUMINAMATH_CALUDE_eight_solutions_of_g_fourth_composition_l2944_294429

/-- The function g(x) = x^2 - 3x -/
def g (x : ℝ) : ℝ := x^2 - 3*x

/-- The theorem stating that there are exactly 8 distinct real numbers d such that g(g(g(g(d)))) = 2 -/
theorem eight_solutions_of_g_fourth_composition :
  ∃! (s : Finset ℝ), (∀ d ∈ s, g (g (g (g d))) = 2) ∧ s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_solutions_of_g_fourth_composition_l2944_294429


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2944_294400

theorem necessary_but_not_sufficient :
  (∀ a b c d : ℝ, (a > b ∧ c > d) → (a + c > b + d)) ∧
  (∃ a b c d : ℝ, (a + c > b + d) ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2944_294400


namespace NUMINAMATH_CALUDE_prob_C_is_five_thirtysix_l2944_294418

/-- A spinner with 5 regions A, B, C, D, and E -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)
  (probE : ℚ)

/-- The properties of the spinner as given in the problem -/
def spinner_properties (s : Spinner) : Prop :=
  s.probA = 5/12 ∧
  s.probB = 1/6 ∧
  s.probC = s.probD ∧
  s.probE = s.probD ∧
  s.probA + s.probB + s.probC + s.probD + s.probE = 1

/-- The theorem stating that the probability of region C is 5/36 -/
theorem prob_C_is_five_thirtysix (s : Spinner) 
  (h : spinner_properties s) : s.probC = 5/36 := by
  sorry

end NUMINAMATH_CALUDE_prob_C_is_five_thirtysix_l2944_294418


namespace NUMINAMATH_CALUDE_fourth_term_is_eight_l2944_294449

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum function
  first_term : a 1 = -1
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating that a_4 = 8 given the conditions -/
theorem fourth_term_is_eight (seq : ArithmeticSequence) (sum_4 : seq.S 4 = 14) :
  seq.a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eight_l2944_294449


namespace NUMINAMATH_CALUDE_car_average_speed_l2944_294439

/-- Given a car's speed for two hours, calculate its average speed. -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 85) (h2 : speed2 = 45) :
  (speed1 + speed2) / 2 = 65 := by
  sorry

#check car_average_speed

end NUMINAMATH_CALUDE_car_average_speed_l2944_294439


namespace NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l2944_294496

theorem linear_diophantine_equation_solutions
  (a b c x₀ y₀ : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (h_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c →
    ∃ k : ℤ, x = x₀ + k * b ∧ y = y₀ - k * a :=
sorry

end NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l2944_294496


namespace NUMINAMATH_CALUDE_quadratic_form_completion_l2944_294473

theorem quadratic_form_completion (z : ℝ) : ∃ (b : ℝ) (c : ℤ), z^2 - 6*z + 20 = (z + b)^2 + c ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_completion_l2944_294473


namespace NUMINAMATH_CALUDE_triangle_similarity_l2944_294476

theorem triangle_similarity (DC CB : ℝ) (AD AB ED : ℝ) (FC : ℝ) : 
  DC = 9 → 
  CB = 6 → 
  AB = (1/3) * AD → 
  ED = (2/3) * AD → 
  FC = 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_similarity_l2944_294476


namespace NUMINAMATH_CALUDE_only_101_prime_l2944_294470

/-- A number in the form 101010...101 with 2n+1 digits -/
def A (n : ℕ) : ℕ := (10^(2*n+2) - 1) / 99

/-- Predicate to check if a number is in the form 101010...101 -/
def is_alternating_101 (x : ℕ) : Prop :=
  ∃ n : ℕ, x = A n

/-- Main theorem: 101 is the only prime number with alternating 1s and 0s -/
theorem only_101_prime :
  ∀ p : ℕ, Prime p ∧ is_alternating_101 p ↔ p = 101 :=
sorry

end NUMINAMATH_CALUDE_only_101_prime_l2944_294470


namespace NUMINAMATH_CALUDE_graph_quadrants_l2944_294445

def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

theorem graph_quadrants (k : ℝ) (h : k < 0) :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (x₁ > 0 ∧ linear_function k x₁ > 0) ∧  -- Quadrant I
    (x₂ < 0 ∧ linear_function k x₂ > 0) ∧  -- Quadrant II
    (x₃ > 0 ∧ linear_function k x₃ < 0)    -- Quadrant IV
  := by sorry

end NUMINAMATH_CALUDE_graph_quadrants_l2944_294445


namespace NUMINAMATH_CALUDE_wheel_rotation_l2944_294444

/-- Given three wheels A, B, and C with radii 35 cm, 20 cm, and 8 cm respectively,
    where wheel A rotates through an angle of 72°, and all wheels rotate without slipping,
    prove that wheel C rotates through an angle of 315°. -/
theorem wheel_rotation (r_A r_B r_C : ℝ) (θ_A θ_C : ℝ) : 
  r_A = 35 →
  r_B = 20 →
  r_C = 8 →
  θ_A = 72 →
  r_A * θ_A = r_C * θ_C →
  θ_C = 315 := by
  sorry

#check wheel_rotation

end NUMINAMATH_CALUDE_wheel_rotation_l2944_294444


namespace NUMINAMATH_CALUDE_expression_evaluation_l2944_294448

theorem expression_evaluation : 
  |(-1/2 : ℝ)| + ((-27 : ℝ) ^ (1/3 : ℝ)) - (1/4 : ℝ).sqrt + (12 : ℝ).sqrt * (3 : ℝ).sqrt = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2944_294448


namespace NUMINAMATH_CALUDE_factorial_ratio_l2944_294401

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 6) = 210 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2944_294401


namespace NUMINAMATH_CALUDE_max_material_a_units_l2944_294484

/-- Represents the quantities of materials A, B, and C --/
structure Materials where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given quantities satisfy the initial cost condition --/
def satisfiesInitialCost (m : Materials) : Prop :=
  3 * m.a + 5 * m.b + 7 * m.c = 62

/-- Checks if the given quantities satisfy the final cost condition --/
def satisfiesFinalCost (m : Materials) : Prop :=
  2 * m.a + 4 * m.b + 6 * m.c = 50

/-- Theorem stating the maximum number of units of material A --/
theorem max_material_a_units :
  ∃ (m : Materials),
    satisfiesInitialCost m ∧
    satisfiesFinalCost m ∧
    m.a = 5 ∧
    ∀ (m' : Materials),
      satisfiesInitialCost m' ∧
      satisfiesFinalCost m' →
      m'.a ≤ m.a :=
by sorry


end NUMINAMATH_CALUDE_max_material_a_units_l2944_294484


namespace NUMINAMATH_CALUDE_james_payment_is_correct_l2944_294453

def james_total_payment (steak_price dessert_price drink_price : ℚ)
  (steak_discount : ℚ) (friend_steak_price friend_dessert_price friend_drink_price : ℚ)
  (friend_steak_discount : ℚ) (meal_tax_rate drink_tax_rate : ℚ)
  (james_tip_rate : ℚ) : ℚ :=
  let james_meal := steak_price * (1 - steak_discount)
  let friend_meal := friend_steak_price * (1 - friend_steak_discount)
  let james_total := james_meal + dessert_price + drink_price
  let friend_total := friend_meal + friend_dessert_price + friend_drink_price
  let james_tax := james_meal * meal_tax_rate + dessert_price * meal_tax_rate + drink_price * drink_tax_rate
  let friend_tax := friend_meal * meal_tax_rate + friend_dessert_price * meal_tax_rate + friend_drink_price * drink_tax_rate
  let total_bill := james_total + friend_total + james_tax + friend_tax
  let james_share := total_bill / 2
  let james_tip := james_share * james_tip_rate
  james_share + james_tip

theorem james_payment_is_correct :
  james_total_payment 16 5 3 0.1 14 4 2 0.05 0.08 0.05 0.2 = 265/10 := by sorry

end NUMINAMATH_CALUDE_james_payment_is_correct_l2944_294453


namespace NUMINAMATH_CALUDE_expression_evaluation_l2944_294414

theorem expression_evaluation : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) - (6^2 - 6) = -32 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2944_294414


namespace NUMINAMATH_CALUDE_min_value_2a5_plus_a4_l2944_294495

/-- A geometric sequence with positive terms satisfying a specific condition -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  condition : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

/-- The minimum value of 2a_5 + a_4 for the given geometric sequence -/
theorem min_value_2a5_plus_a4 (seq : GeometricSequence) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ x : ℝ, (2 * seq.a 5 + seq.a 4) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_2a5_plus_a4_l2944_294495


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2944_294437

/-- A dodecahedron is a 3-dimensional figure with 12 pentagonal faces and 20 vertices,
    where 3 faces meet at each vertex. -/
structure Dodecahedron where
  vertices : Nat
  faces : Nat
  faces_per_vertex : Nat
  vertices_eq : vertices = 20
  faces_eq : faces = 12
  faces_per_vertex_eq : faces_per_vertex = 3

/-- An interior diagonal of a dodecahedron is a segment connecting two vertices
    which do not lie on a common face. -/
def interior_diagonal (d : Dodecahedron) : Nat :=
  sorry

/-- The number of interior diagonals in a dodecahedron is 160. -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) :
  interior_diagonal d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l2944_294437


namespace NUMINAMATH_CALUDE_tree_growth_condition_l2944_294457

/-- Represents the annual growth of a tree over 6 years -/
structure TreeGrowth where
  initial_height : ℝ
  annual_increase : ℝ

/-- Calculates the height of the tree after a given number of years -/
def height_after_years (t : TreeGrowth) (years : ℕ) : ℝ :=
  t.initial_height + t.annual_increase * years

/-- Theorem stating the condition for the tree's growth -/
theorem tree_growth_condition (t : TreeGrowth) : 
  t.initial_height = 4 ∧ 
  height_after_years t 6 = height_after_years t 4 + (1/7) * height_after_years t 4 →
  t.annual_increase = 2/5 :=
sorry

end NUMINAMATH_CALUDE_tree_growth_condition_l2944_294457


namespace NUMINAMATH_CALUDE_fixed_point_satisfies_equation_fixed_point_is_unique_l2944_294428

/-- The line equation passing through a fixed point for all real values of a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- The fixed point coordinates -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the fixed point satisfies the line equation for all real a -/
theorem fixed_point_satisfies_equation :
  ∀ (a : ℝ), line_equation a (fixed_point.1) (fixed_point.2) :=
by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_is_unique :
  ∀ (x y : ℝ), (∀ (a : ℝ), line_equation a x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_satisfies_equation_fixed_point_is_unique_l2944_294428


namespace NUMINAMATH_CALUDE_vector_problem_l2944_294442

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 6)
def c (x : ℝ) : ℝ × ℝ := 2 • a + b x

theorem vector_problem (x : ℝ) :
  (∃ y, b y ≠ r • a ∧ r ≠ 0) →  -- non-collinearity condition
  ‖a - b x‖ = 2 * Real.sqrt 5 →
  c x = (1, 10) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2944_294442


namespace NUMINAMATH_CALUDE_cube_decomposition_l2944_294494

/-- The smallest odd number in the decomposition of m³ -/
def smallest_odd (m : ℕ+) : ℕ := 2 * (m - 1) + 3

/-- The number of odd terms in the decomposition of m³ -/
def num_terms (m : ℕ+) : ℕ := (m + 2) * (m - 1) / 2

theorem cube_decomposition (m : ℕ+) :
  smallest_odd m = 91 → m = 10 := by sorry

end NUMINAMATH_CALUDE_cube_decomposition_l2944_294494


namespace NUMINAMATH_CALUDE_digit_452_of_7_19_is_6_l2944_294450

/-- The decimal representation of 7/19 is repeating -/
def decimal_rep_7_19_repeating : Prop := 
  ∃ (s : List Nat), s.length > 0 ∧ (7 : ℚ) / 19 = (s.map (λ n => (n : ℚ) / 10^s.length)).sum

/-- The 452nd digit after the decimal point in the decimal representation of 7/19 -/
def digit_452_of_7_19 : Nat := sorry

theorem digit_452_of_7_19_is_6 (h : decimal_rep_7_19_repeating) : 
  digit_452_of_7_19 = 6 := by sorry

end NUMINAMATH_CALUDE_digit_452_of_7_19_is_6_l2944_294450


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2944_294419

/-- Given two congruent squares with side length 12 that overlap to form a 12 by 20 rectangle,
    prove that 20% of the rectangle's area is shaded. -/
theorem shaded_area_percentage (side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  side_length = 12 →
  rectangle_width = 12 →
  rectangle_length = 20 →
  (side_length * side_length - rectangle_width * rectangle_length) / (rectangle_width * rectangle_length) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2944_294419


namespace NUMINAMATH_CALUDE_smallest_divisor_square_plus_divisor_square_l2944_294421

theorem smallest_divisor_square_plus_divisor_square (n : ℕ) : n ≥ 2 → (
  (∃ k d : ℕ, 
    k > 1 ∧ 
    k ∣ n ∧ 
    (∀ m : ℕ, m > 1 ∧ m ∣ n → k ≤ m) ∧ 
    d ∣ n ∧ 
    n = k^2 + d^2
  ) ↔ (n = 8 ∨ n = 20)
) := by sorry

end NUMINAMATH_CALUDE_smallest_divisor_square_plus_divisor_square_l2944_294421


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2944_294482

theorem floor_equation_solution :
  {x : ℚ | ⌊(8*x + 19)/7⌋ = (16*(x + 1))/11} =
  {1 + 1/16, 1 + 3/4, 2 + 7/16, 3 + 1/8, 3 + 13/16} := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2944_294482


namespace NUMINAMATH_CALUDE_abs_z_eq_one_l2944_294407

theorem abs_z_eq_one (z : ℂ) (h : (1 - Complex.I) / z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_eq_one_l2944_294407


namespace NUMINAMATH_CALUDE_min_l_pieces_in_8x8_l2944_294469

/-- Represents an 8x8 square board --/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents a three-cell L-shaped piece --/
structure LPiece where
  x : Fin 8
  y : Fin 8
  orientation : Fin 4

/-- Checks if an L-piece can be placed on the board --/
def canPlace (board : Board) (piece : LPiece) : Bool :=
  sorry

/-- Places an L-piece on the board --/
def placePiece (board : Board) (piece : LPiece) : Board :=
  sorry

/-- Checks if any more L-pieces can be placed on the board --/
def canPlaceMore (board : Board) : Bool :=
  sorry

/-- The main theorem --/
theorem min_l_pieces_in_8x8 :
  ∃ (pieces : List LPiece),
    pieces.length = 11 ∧
    (∃ (board : Board),
      (∀ p ∈ pieces, canPlace board p) ∧
      (∀ p ∈ pieces, board = placePiece board p) ∧
      ¬canPlaceMore board) ∧
    (∀ (pieces' : List LPiece),
      pieces'.length < 11 →
      ∀ (board : Board),
        (∀ p ∈ pieces', canPlace board p) →
        (∀ p ∈ pieces', board = placePiece board p) →
        canPlaceMore board) :=
  sorry

end NUMINAMATH_CALUDE_min_l_pieces_in_8x8_l2944_294469


namespace NUMINAMATH_CALUDE_percentage_of_adult_men_l2944_294455

theorem percentage_of_adult_men (total : ℕ) (children : ℕ) 
  (h1 : total = 2000) 
  (h2 : children = 200) 
  (h3 : ∃ (men women : ℕ), men + women + children = total ∧ women = 2 * men) :
  ∃ (men : ℕ), men * 100 / total = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_adult_men_l2944_294455


namespace NUMINAMATH_CALUDE_card_game_proof_l2944_294436

theorem card_game_proof (total_credits : ℕ) (red_cards : ℕ) (red_credit_value : ℕ) (blue_credit_value : ℕ)
  (h1 : total_credits = 84)
  (h2 : red_cards = 8)
  (h3 : red_credit_value = 3)
  (h4 : blue_credit_value = 5) :
  ∃ (blue_cards : ℕ), red_cards + blue_cards = 20 ∧ 
    red_cards * red_credit_value + blue_cards * blue_credit_value = total_credits :=
by
  sorry

end NUMINAMATH_CALUDE_card_game_proof_l2944_294436


namespace NUMINAMATH_CALUDE_x_value_l2944_294438

theorem x_value (x : ℚ) (h : 1/3 - 1/4 = 4/x) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2944_294438


namespace NUMINAMATH_CALUDE_extreme_value_implies_sum_l2944_294466

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem: If f(x) has an extreme value of 10 at x = 1, then a + b = -7 -/
theorem extreme_value_implies_sum (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_sum_l2944_294466


namespace NUMINAMATH_CALUDE_f_sum_property_l2944_294458

def f (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 4*x^4 + x^3 - 2*x^2 - 2*x + 8

theorem f_sum_property : f 5 = 20 → f 5 + f (-5) = 68343 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_property_l2944_294458


namespace NUMINAMATH_CALUDE_total_saltwater_animals_l2944_294463

theorem total_saltwater_animals (num_aquariums : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : num_aquariums = 26)
  (h2 : animals_per_aquarium = 2) :
  num_aquariums * animals_per_aquarium = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_saltwater_animals_l2944_294463


namespace NUMINAMATH_CALUDE_octagon_coloring_count_l2944_294415

/-- The number of disks in the octagonal pattern -/
def num_disks : ℕ := 8

/-- The number of blue disks -/
def num_blue : ℕ := 3

/-- The number of red disks -/
def num_red : ℕ := 3

/-- The number of green disks -/
def num_green : ℕ := 2

/-- The symmetry group of a regular octagon -/
def octagon_symmetry_group_order : ℕ := 16

/-- The number of distinct colorings considering symmetries -/
def distinct_colorings : ℕ := 43

/-- Theorem stating the number of distinct colorings -/
theorem octagon_coloring_count :
  let total_colorings := (Nat.choose num_disks num_blue) * (Nat.choose (num_disks - num_blue) num_red)
  (total_colorings / octagon_symmetry_group_order : ℚ).num = distinct_colorings := by
  sorry

end NUMINAMATH_CALUDE_octagon_coloring_count_l2944_294415


namespace NUMINAMATH_CALUDE_roses_picked_l2944_294462

theorem roses_picked (initial : ℕ) (sold : ℕ) (final : ℕ) : initial = 37 → sold = 16 → final = 40 → final - (initial - sold) = 19 := by
  sorry

end NUMINAMATH_CALUDE_roses_picked_l2944_294462


namespace NUMINAMATH_CALUDE_length_to_breadth_ratio_l2944_294485

/-- Represents a rectangular plot -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  area : ℝ
  length_multiple_of_breadth : ∃ (k : ℝ), length = k * breadth
  area_eq : area = length * breadth

/-- Theorem: The ratio of length to breadth is 3:1 for a rectangular plot with area 2028 and breadth 26 -/
theorem length_to_breadth_ratio (plot : RectangularPlot) 
  (h_area : plot.area = 2028)
  (h_breadth : plot.breadth = 26) :
  plot.length / plot.breadth = 3 := by
sorry

end NUMINAMATH_CALUDE_length_to_breadth_ratio_l2944_294485


namespace NUMINAMATH_CALUDE_line_relationship_sum_l2944_294486

/-- Represents a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.A * l2.B = l1.B * l2.A

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.A * l2.A + l1.B * l2.B = 0

theorem line_relationship_sum (m n : ℝ) : 
  let l1 : Line := ⟨2, 2, -1⟩
  let l2 : Line := ⟨4, n, 3⟩
  let l3 : Line := ⟨m, 6, 1⟩
  parallel l1 l2 → perpendicular l1 l3 → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_relationship_sum_l2944_294486


namespace NUMINAMATH_CALUDE_illumination_ways_l2944_294422

theorem illumination_ways (n : ℕ) (h : n = 6) : 2^n - 1 = 63 := by
  sorry

end NUMINAMATH_CALUDE_illumination_ways_l2944_294422


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l2944_294454

/-- Represents a 3x3 matrix of integers -/
def Matrix3x3 := Fin 3 → Fin 3 → ℤ

/-- Checks if two integers are coprime -/
def are_coprime (a b : ℤ) : Prop := Nat.gcd a.natAbs b.natAbs = 1

/-- Checks if the matrix satisfies the adjacency condition -/
def satisfies_adjacency_condition (m : Matrix3x3) : Prop :=
  ∀ i j i' j', (i = i' ∧ j.succ = j') ∨ (i = i' ∧ j = j'.succ) ∨
                (i.succ = i' ∧ j = j') ∨ (i = i'.succ ∧ j = j') ∨
                (i.succ = i' ∧ j.succ = j') ∨ (i.succ = i' ∧ j = j'.succ) ∨
                (i = i'.succ ∧ j.succ = j') ∨ (i = i'.succ ∧ j = j'.succ) →
                are_coprime (m i j) (m i' j')

/-- Checks if the matrix contains nine consecutive integers -/
def contains_consecutive_integers (m : Matrix3x3) : Prop :=
  ∃ start : ℤ, ∀ i j, ∃ k : ℕ, k < 9 ∧ m i j = start + k

/-- The main theorem stating the existence of a valid arrangement -/
theorem valid_arrangement_exists : ∃ m : Matrix3x3, 
  satisfies_adjacency_condition m ∧ contains_consecutive_integers m := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l2944_294454


namespace NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l2944_294440

def sum_even_integers (a b : ℕ) : ℕ :=
  let first := if a % 2 = 0 then a else a + 1
  let last := if b % 2 = 0 then b else b - 1
  let n := (last - first) / 2 + 1
  n * (first + last) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  let first := if a % 2 = 0 then a else a + 1
  let last := if b % 2 = 0 then b else b - 1
  (last - first) / 2 + 1

theorem gcd_sum_and_count_even_integers :
  Nat.gcd (sum_even_integers 13 63) (count_even_integers 13 63) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l2944_294440


namespace NUMINAMATH_CALUDE_tan_sum_identity_l2944_294411

theorem tan_sum_identity (x y z : Real) 
  (hx : x = 20 * π / 180)
  (hy : y = 30 * π / 180)
  (hz : z = 40 * π / 180)
  (h1 : Real.tan (60 * π / 180) = Real.sqrt 3)
  (h2 : Real.tan (30 * π / 180) = 1 / Real.sqrt 3) :
  Real.tan x * Real.tan y + Real.tan y * Real.tan z + Real.tan z * Real.tan x = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l2944_294411


namespace NUMINAMATH_CALUDE_inequality_proof_l2944_294491

theorem inequality_proof (x y : ℝ) : x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2944_294491


namespace NUMINAMATH_CALUDE_intersection_points_form_circle_l2944_294447

-- Define the system of equations
def equation1 (s x y : ℝ) : Prop := 3 * s * x - 5 * y - 7 * s = 0
def equation2 (s x y : ℝ) : Prop := 2 * x - 5 * s * y + 4 = 0

-- Define the set of points satisfying both equations
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ s : ℝ, equation1 s p.1 p.2 ∧ equation2 s p.1 p.2}

-- Theorem stating that the intersection points form a circle
theorem intersection_points_form_circle :
  ∃ c : ℝ × ℝ, ∃ r : ℝ, ∀ p ∈ intersection_points,
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_form_circle_l2944_294447
