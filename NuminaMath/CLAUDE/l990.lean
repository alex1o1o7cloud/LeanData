import Mathlib

namespace NUMINAMATH_CALUDE_painted_cube_probability_l990_99050

/-- Represents a cube with painted faces --/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ

/-- Calculates the number of unit cubes with exactly three painted faces --/
def num_three_painted_faces (cube : PaintedCube) : ℕ :=
  if cube.painted_faces = 2 then 4 else 0

/-- Calculates the number of unit cubes with no painted faces --/
def num_no_painted_faces (cube : PaintedCube) : ℕ :=
  (cube.size - 2) ^ 3

/-- Calculates the total number of unit cubes --/
def total_unit_cubes (cube : PaintedCube) : ℕ :=
  cube.size ^ 3

/-- Calculates the number of ways to choose 2 cubes from the total --/
def choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: The probability of selecting one unit cube with three painted faces
    and one with no painted faces is 9/646 for a 5x5x5 cube with two adjacent
    painted faces --/
theorem painted_cube_probability (cube : PaintedCube)
    (h1 : cube.size = 5)
    (h2 : cube.painted_faces = 2) :
    (num_three_painted_faces cube * num_no_painted_faces cube : ℚ) /
    choose_two (total_unit_cubes cube) = 9 / 646 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l990_99050


namespace NUMINAMATH_CALUDE_min_total_cards_problem_l990_99014

def min_total_cards (carlos_cards : ℕ) (matias_diff : ℕ) (ella_multiplier : ℕ) (divisor : ℕ) : ℕ :=
  let matias_cards := carlos_cards - matias_diff
  let jorge_cards := matias_cards
  let ella_cards := ella_multiplier * (jorge_cards + matias_cards)
  let total_cards := carlos_cards + matias_cards + jorge_cards + ella_cards
  ((total_cards + divisor - 1) / divisor) * divisor

theorem min_total_cards_problem :
  min_total_cards 20 6 2 15 = 105 := by sorry

end NUMINAMATH_CALUDE_min_total_cards_problem_l990_99014


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l990_99035

def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l990_99035


namespace NUMINAMATH_CALUDE_triangle_theorem_l990_99019

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle satisfying the given condition,
    angle C is π/4 and the maximum area when c = 2 is 1 + √2. -/
theorem triangle_theorem (t : Triangle) 
    (h : t.a * Real.cos t.B + t.b * Real.cos t.A - Real.sqrt 2 * t.c * Real.cos t.C = 0) :
    t.C = π / 4 ∧ 
    (t.c = 2 → ∃ (S : ℝ), S = (1 + Real.sqrt 2) ∧ ∀ (S' : ℝ), S' ≤ S) := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_triangle_theorem_l990_99019


namespace NUMINAMATH_CALUDE_simplify_expression_l990_99095

theorem simplify_expression : (27 ^ (1/6) - Real.sqrt (6 + 3/4)) ^ 2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l990_99095


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_integers_sum_2025_l990_99064

theorem smallest_of_five_consecutive_integers_sum_2025 (n : ℤ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 2025) → n = 403 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_integers_sum_2025_l990_99064


namespace NUMINAMATH_CALUDE_max_value_on_circle_l990_99028

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 10 →
  4*x + 3*y ≤ 32 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l990_99028


namespace NUMINAMATH_CALUDE_pictures_per_album_l990_99040

/-- Given a total of 20 pictures sorted equally into 5 albums, prove that each album contains 4 pictures. -/
theorem pictures_per_album :
  let total_pictures : ℕ := 7 + 13
  let num_albums : ℕ := 5
  let pictures_per_album : ℕ := total_pictures / num_albums
  pictures_per_album = 4 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l990_99040


namespace NUMINAMATH_CALUDE_largest_reciprocal_l990_99045

theorem largest_reciprocal (a b c d e : ℝ) 
  (ha : a = 1/4) 
  (hb : b = 3/7) 
  (hc : c = 0.25) 
  (hd : d = 7) 
  (he : e = 5000) : 
  (1/a > 1/b) ∧ (1/a > 1/c) ∧ (1/a > 1/d) ∧ (1/a > 1/e) :=
by sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l990_99045


namespace NUMINAMATH_CALUDE_workshop_workers_l990_99022

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- Represents the number of technicians -/
def technicians : ℕ := 5

/-- Represents the average salary of all workers -/
def avg_salary_all : ℚ := 700

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℚ := 800

/-- Represents the average salary of the rest of the workers -/
def avg_salary_rest : ℚ := 650

theorem workshop_workers :
  (avg_salary_all * total_workers : ℚ) = 
  (avg_salary_technicians * technicians : ℚ) + 
  (avg_salary_rest * (total_workers - technicians) : ℚ) := by
  sorry

#check workshop_workers

end NUMINAMATH_CALUDE_workshop_workers_l990_99022


namespace NUMINAMATH_CALUDE_left_handed_fraction_l990_99054

/-- Represents the number of participants from each world -/
structure Participants where
  red : ℚ
  blue : ℚ
  green : ℚ

/-- Calculates the total number of participants -/
def total_participants (p : Participants) : ℚ :=
  p.red + p.blue + p.green

/-- Calculates the number of left-handed participants -/
def left_handed_participants (p : Participants) : ℚ :=
  p.red / 3 + 2 * p.blue / 3

/-- The main theorem stating the fraction of left-handed participants -/
theorem left_handed_fraction (p : Participants) 
  (h1 : p.red = 3 * p.blue / 2)  -- ratio of red to blue is 3:2
  (h2 : p.blue = 5 * p.green / 4)  -- ratio of blue to green is 5:4
  : left_handed_participants p / total_participants p = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fraction_l990_99054


namespace NUMINAMATH_CALUDE_extreme_points_condition_l990_99008

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^3 + 2*a*x^2 + x + 1

-- Define the derivative of f(x)
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 4*a*x + 1

-- Theorem statement
theorem extreme_points_condition (a x₁ x₂ : ℝ) : 
  (f_derivative a x₁ = 0) →  -- x₁ is an extreme point
  (f_derivative a x₂ = 0) →  -- x₂ is an extreme point
  (x₂ - x₁ = 2) →            -- Given condition
  (a^2 = 3) :=               -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_extreme_points_condition_l990_99008


namespace NUMINAMATH_CALUDE_project_completion_time_l990_99038

theorem project_completion_time 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (work_days_A : ℝ) 
  (remaining_days_B : ℝ) 
  (h1 : days_A = 10) 
  (h2 : days_B = 15) 
  (h3 : work_days_A = 3) : 
  work_days_A / days_A + remaining_days_B / days_B = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l990_99038


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l990_99092

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_monotonicity (a b : ℝ) :
  (f' a b (-1) = 0 ∧ f' a b 2 = 0) →
  (a = -3/2 ∧ b = -6) ∧
  (∀ x, x ∈ Set.Ioo (-1) 2 → (f' (-3/2) (-6) x < 0)) ∧
  (∀ x, (x < -1 ∨ x > 2) → (f' (-3/2) (-6) x > 0)) ∧
  (∀ m, (∀ x, x ∈ Set.Icc (-2) 3 → f (-3/2) (-6) x < m) ↔ m > 7/2) :=
by sorry


end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l990_99092


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l990_99016

theorem complex_magnitude_equation (n : ℝ) : 
  (n > 0 ∧ Complex.abs (5 + n * Complex.I) = Real.sqrt 34) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l990_99016


namespace NUMINAMATH_CALUDE_betty_height_l990_99084

theorem betty_height (dog_height : ℕ) (carter_height : ℕ) (betty_height_inches : ℕ) :
  dog_height = 24 →
  carter_height = 2 * dog_height →
  betty_height_inches = carter_height - 12 →
  betty_height_inches / 12 = 3 :=
by sorry

end NUMINAMATH_CALUDE_betty_height_l990_99084


namespace NUMINAMATH_CALUDE_expression_equality_l990_99041

theorem expression_equality : (2^2 / 3) + (-3^2 + 5) + (-3)^2 * (2/3)^2 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l990_99041


namespace NUMINAMATH_CALUDE_partner_a_profit_share_l990_99009

/-- Calculates the share of profit for partner A in a business partnership --/
theorem partner_a_profit_share
  (initial_a initial_b : ℕ)
  (withdrawal_a addition_b : ℕ)
  (total_months : ℕ)
  (change_month : ℕ)
  (total_profit : ℕ)
  (h1 : initial_a = 2000)
  (h2 : initial_b = 4000)
  (h3 : withdrawal_a = 1000)
  (h4 : addition_b = 1000)
  (h5 : total_months = 12)
  (h6 : change_month = 8)
  (h7 : total_profit = 630) :
  let investment_months_a := initial_a * change_month + (initial_a - withdrawal_a) * (total_months - change_month)
  let investment_months_b := initial_b * change_month + (initial_b + addition_b) * (total_months - change_month)
  let total_investment_months := investment_months_a + investment_months_b
  let a_share := (investment_months_a * total_profit) / total_investment_months
  a_share = 175 := by sorry

end NUMINAMATH_CALUDE_partner_a_profit_share_l990_99009


namespace NUMINAMATH_CALUDE_inscribed_circles_area_limit_l990_99036

/-- Represents the sum of areas of the first n inscribed circles -/
def S (n : ℕ) (a : ℝ) : ℝ := sorry

/-- The limit of S_n as n approaches infinity -/
def S_limit (a : ℝ) : ℝ := sorry

theorem inscribed_circles_area_limit (a b : ℝ) (h : 0 < a ∧ a ≤ b) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n a - S_limit a| < ε ∧ S_limit a = (π * a^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_limit_l990_99036


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l990_99074

/-- Represents a triangular lattice structure made of toothpicks -/
structure TriangularLattice :=
  (toothpicks : ℕ)
  (triangles : ℕ)
  (horizontal_toothpicks : ℕ)

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (lattice : TriangularLattice) : ℕ :=
  lattice.horizontal_toothpicks

theorem min_toothpicks_removal (lattice : TriangularLattice) 
  (h1 : lattice.toothpicks = 40)
  (h2 : lattice.triangles > 40)
  (h3 : lattice.horizontal_toothpicks = 15) :
  min_toothpicks_to_remove lattice = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l990_99074


namespace NUMINAMATH_CALUDE_sarahs_weeds_total_l990_99094

theorem sarahs_weeds_total (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (friday : ℕ) : 
  tuesday = 25 →
  wednesday = 3 * tuesday →
  thursday = wednesday / 5 →
  friday = thursday - 10 →
  tuesday + wednesday + thursday + friday = 120 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_weeds_total_l990_99094


namespace NUMINAMATH_CALUDE_julia_played_with_17_kids_on_monday_l990_99000

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := sorry

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := 2

/-- The total number of kids Julia played with -/
def total_kids : ℕ := 34

/-- Theorem stating that Julia played with 17 kids on Monday -/
theorem julia_played_with_17_kids_on_monday :
  monday_kids = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_17_kids_on_monday_l990_99000


namespace NUMINAMATH_CALUDE_binomial_probability_one_third_l990_99024

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial variable -/
def expectation (X : BinomialVariable) : ℝ := X.n * X.p

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_probability_one_third 
  (X : BinomialVariable) 
  (h_expectation : expectation X = 30)
  (h_variance : variance X = 20) : 
  X.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_one_third_l990_99024


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l990_99004

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, (∃ A B : ℤ, ∀ x, 5*x^2 + n*x + 50 = (5*x + A)*(x + B)) → n ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_l990_99004


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l990_99060

theorem inequality_solution_existence (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l990_99060


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l990_99026

theorem smallest_five_digit_divisible_by_53 : ∀ n : ℕ, 
  10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 → n ≥ 10017 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l990_99026


namespace NUMINAMATH_CALUDE_leah_daily_earnings_l990_99081

/-- Represents Leah's earnings over a period of time -/
structure Earnings where
  total : ℕ  -- Total earnings in dollars
  weeks : ℕ  -- Number of weeks worked
  daily : ℕ  -- Daily earnings in dollars

/-- Calculates the number of days in a given number of weeks -/
def daysInWeeks (weeks : ℕ) : ℕ :=
  7 * weeks

/-- Theorem: Leah's daily earnings are 60 dollars -/
theorem leah_daily_earnings (e : Earnings) (h1 : e.total = 1680) (h2 : e.weeks = 4) :
  e.daily = 60 := by
  sorry

end NUMINAMATH_CALUDE_leah_daily_earnings_l990_99081


namespace NUMINAMATH_CALUDE_inequality_system_solution_l990_99025

theorem inequality_system_solution :
  {x : ℝ | 3*x - 1 ≥ x + 1 ∧ x + 4 > 4*x - 2} = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l990_99025


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_five_l990_99059

theorem square_of_negative_sqrt_five : (-Real.sqrt 5)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_five_l990_99059


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l990_99052

/-- The number of sides of a polygon given the sum of its interior angles -/
theorem polygon_sides_from_angle_sum (angle_sum : ℝ) : angle_sum = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = angle_sum := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l990_99052


namespace NUMINAMATH_CALUDE_tan_three_pi_halves_minus_alpha_l990_99096

theorem tan_three_pi_halves_minus_alpha (α : Real) 
  (h : Real.cos (Real.pi - α) = -3/5) : 
  Real.tan (3/2 * Real.pi - α) = 3/4 ∨ Real.tan (3/2 * Real.pi - α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_pi_halves_minus_alpha_l990_99096


namespace NUMINAMATH_CALUDE_xyz_inequality_l990_99047

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x*y + y*z + z*x = 1) : x*y*z*(x + y + z) ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l990_99047


namespace NUMINAMATH_CALUDE_avery_building_time_l990_99015

theorem avery_building_time (tom_time : ℝ) (joint_work_time : ℝ) (tom_remaining_time : ℝ) 
  (h1 : tom_time = 2)
  (h2 : joint_work_time = 1)
  (h3 : tom_remaining_time = 20.000000000000007 / 60) :
  ∃ (avery_time : ℝ), 
    1 / avery_time + 1 / tom_time + (tom_remaining_time / tom_time) = 1 ∧ 
    avery_time = 3 := by
sorry

end NUMINAMATH_CALUDE_avery_building_time_l990_99015


namespace NUMINAMATH_CALUDE_farmer_seeds_total_l990_99062

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := 20

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := 2

/-- The total number of seeds planted -/
def total_seeds : ℕ := seeds_wednesday + seeds_thursday

theorem farmer_seeds_total :
  total_seeds = 22 :=
by sorry

end NUMINAMATH_CALUDE_farmer_seeds_total_l990_99062


namespace NUMINAMATH_CALUDE_caravan_keepers_l990_99098

/-- The number of keepers in a caravan with hens, goats, and camels. -/
def num_keepers : ℕ := by sorry

theorem caravan_keepers :
  let hens : ℕ := 50
  let goats : ℕ := 45
  let camels : ℕ := 8
  let hen_feet : ℕ := 2
  let goat_feet : ℕ := 4
  let camel_feet : ℕ := 4
  let keeper_head : ℕ := 1
  let keeper_feet : ℕ := 2
  let total_animal_feet : ℕ := hens * hen_feet + goats * goat_feet + camels * camel_feet
  let total_animal_heads : ℕ := hens + goats + camels
  let extra_feet : ℕ := 224
  num_keepers * keeper_feet + total_animal_feet = num_keepers * keeper_head + total_animal_heads + extra_feet →
  num_keepers = 15 := by sorry

end NUMINAMATH_CALUDE_caravan_keepers_l990_99098


namespace NUMINAMATH_CALUDE_second_order_implies_first_order_l990_99018

/-- A function f: ℝ → ℝ is increasing on an interval D if for any x, y ∈ D, x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x y, x ∈ D → y ∈ D → x < y → f x < f y

/-- x₀ is a second-order fixed point of f if f(f(x₀)) = x₀ -/
def SecondOrderFixedPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (f x₀) = x₀

/-- x₀ is a first-order fixed point of f if f(x₀) = x₀ -/
def FirstOrderFixedPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = x₀

theorem second_order_implies_first_order
    (f : ℝ → ℝ) (D : Set ℝ) (x₀ : ℝ)
    (h_inc : IncreasingOn f D)
    (h_x₀ : x₀ ∈ D)
    (h_second : SecondOrderFixedPoint f x₀) :
    FirstOrderFixedPoint f x₀ := by
  sorry

end NUMINAMATH_CALUDE_second_order_implies_first_order_l990_99018


namespace NUMINAMATH_CALUDE_water_cup_fills_l990_99030

theorem water_cup_fills (container_volume : ℚ) (cup_volume : ℚ) : 
  container_volume = 13/3 → cup_volume = 1/6 → 
  (container_volume / cup_volume : ℚ) = 26 := by
  sorry

end NUMINAMATH_CALUDE_water_cup_fills_l990_99030


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l990_99072

theorem no_solution_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) → m = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l990_99072


namespace NUMINAMATH_CALUDE_revenue_calculation_l990_99090

/-- The revenue from a single sold-out performance for Steve's circus production -/
def revenue_per_performance : ℕ := sorry

/-- The overhead cost for Steve's circus production -/
def overhead_cost : ℕ := 81000

/-- The production cost per performance for Steve's circus production -/
def production_cost_per_performance : ℕ := 7000

/-- The number of sold-out performances needed to break even -/
def performances_to_break_even : ℕ := 9

/-- Theorem stating that the revenue from a single sold-out performance is $16,000 -/
theorem revenue_calculation :
  revenue_per_performance = 16000 :=
by
  sorry

#check revenue_calculation

end NUMINAMATH_CALUDE_revenue_calculation_l990_99090


namespace NUMINAMATH_CALUDE_max_annual_average_profit_l990_99021

/-- The annual average profit function -/
def f (n : ℕ+) : ℚ :=
  (110 * n - (n^2 + n) - 90) / n

/-- Theorem stating that f(n) reaches its maximum when n = 5 -/
theorem max_annual_average_profit :
  ∀ k : ℕ+, f 5 ≥ f k :=
sorry

end NUMINAMATH_CALUDE_max_annual_average_profit_l990_99021


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l990_99017

theorem boys_to_girls_ratio (T : ℚ) (G : ℚ) (h : (2/3) * G = (1/4) * T) : 
  (T - G) / G = 5/3 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l990_99017


namespace NUMINAMATH_CALUDE_omega_range_l990_99077

theorem omega_range (ω : ℝ) (a b : ℝ) (h_pos : ω > 0) 
  (h_ab : π ≤ a ∧ a < b ∧ b ≤ 2*π) 
  (h_sin : Real.sin (ω*a) + Real.sin (ω*b) = 2) : 
  (9/4 ≤ ω ∧ ω ≤ 5/2) ∨ (13/4 ≤ ω) :=
sorry

end NUMINAMATH_CALUDE_omega_range_l990_99077


namespace NUMINAMATH_CALUDE_complement_of_M_l990_99068

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M :
  (U \ M) = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l990_99068


namespace NUMINAMATH_CALUDE_total_ndfl_is_11050_l990_99043

/-- Calculates the total NDFL (personal income tax) on income from securities --/
def calculate_ndfl (dividend_income : ℝ) (ofz_coupon_income : ℝ) (corporate_coupon_income : ℝ) 
  (shares_sold : ℕ) (sale_price_per_share : ℝ) (purchase_price_per_share : ℝ) 
  (dividend_tax_rate : ℝ) (corporate_coupon_tax_rate : ℝ) (capital_gains_tax_rate : ℝ) : ℝ :=
  let capital_gains := shares_sold * (sale_price_per_share - purchase_price_per_share)
  let dividend_tax := dividend_income * dividend_tax_rate
  let corporate_coupon_tax := corporate_coupon_income * corporate_coupon_tax_rate
  let capital_gains_tax := capital_gains * capital_gains_tax_rate
  dividend_tax + corporate_coupon_tax + capital_gains_tax

/-- Theorem stating that the total NDFL on income from securities is 11,050 rubles --/
theorem total_ndfl_is_11050 :
  calculate_ndfl 50000 40000 30000 100 200 150 0.13 0.13 0.13 = 11050 := by
  sorry

end NUMINAMATH_CALUDE_total_ndfl_is_11050_l990_99043


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l990_99093

theorem floor_abs_negative_real : ⌊|(-54.7 : ℝ)|⌋ = 54 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l990_99093


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l990_99066

theorem max_value_of_linear_combination (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 6 * y ≤ 12) : 
  x + 2 * y ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), 4 * x₀ + 3 * y₀ ≤ 10 ∧ 3 * x₀ + 6 * y₀ ≤ 12 ∧ x₀ + 2 * y₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l990_99066


namespace NUMINAMATH_CALUDE_nested_g_equals_cos_fifteen_fourths_l990_99034

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x / 2)

theorem nested_g_equals_cos_fifteen_fourths :
  0 < (1 : ℝ) / 2 ∧
  (∀ x : ℝ, 0 < x → 0 < g x) →
  g (g (g (g (g ((1 : ℝ) / 2) + 1) + 1) + 1) + 1) = Real.cos (15 / 4 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_nested_g_equals_cos_fifteen_fourths_l990_99034


namespace NUMINAMATH_CALUDE_smallest_integer_with_eight_factors_l990_99097

theorem smallest_integer_with_eight_factors : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (factors : Finset ℕ), factors.card = 8 ∧ 
    (∀ m ∈ factors, m > 0 ∧ n % m = 0) ∧
    (∀ m : ℕ, m > 0 → n % m = 0 → m ∈ factors)) ∧
  (∀ k : ℕ, k > 0 → k < n →
    ¬(∃ (factors : Finset ℕ), factors.card = 8 ∧ 
      (∀ m ∈ factors, m > 0 ∧ k % m = 0) ∧
      (∀ m : ℕ, m > 0 → k % m = 0 → m ∈ factors))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_eight_factors_l990_99097


namespace NUMINAMATH_CALUDE_power_of_power_l990_99073

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l990_99073


namespace NUMINAMATH_CALUDE_expression_simplification_l990_99049

theorem expression_simplification :
  Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt 3 - 1| + Real.sqrt 3 = -13 / 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l990_99049


namespace NUMINAMATH_CALUDE_angle_ADB_is_270_degrees_l990_99029

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop := sorry

def angleAIs45 (t : Triangle) : Prop := sorry

def angleBIs45 (t : Triangle) : Prop := sorry

-- Define the angle bisectors and their intersection
def angleBisectorA (t : Triangle) : Line := sorry

def angleBisectorB (t : Triangle) : Line := sorry

def D (t : Triangle) : Point := sorry

-- Define the measure of angle ADB
def measureAngleADB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_ADB_is_270_degrees (t : Triangle) :
  isRightTriangle t → angleAIs45 t → angleBIs45 t →
  measureAngleADB t = 270 :=
sorry

end NUMINAMATH_CALUDE_angle_ADB_is_270_degrees_l990_99029


namespace NUMINAMATH_CALUDE_two_inequalities_true_l990_99053

theorem two_inequalities_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : y^2 < b^2) : 
  ∃! n : ℕ, n = 2 ∧ 
    (n = (if x^2 + y^2 < a^2 + b^2 then 1 else 0) +
         (if x^2 - y^2 < a^2 - b^2 then 1 else 0) +
         (if x^2 * y^2 < a^2 * b^2 then 1 else 0) +
         (if x^2 / y^2 < a^2 / b^2 then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_two_inequalities_true_l990_99053


namespace NUMINAMATH_CALUDE_child_b_share_l990_99082

theorem child_b_share (total_money : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_money = 900 → 
  ratio_a = 2 → 
  ratio_b = 3 → 
  ratio_c = 4 → 
  (ratio_b * total_money) / (ratio_a + ratio_b + ratio_c) = 300 := by
  sorry

end NUMINAMATH_CALUDE_child_b_share_l990_99082


namespace NUMINAMATH_CALUDE_martha_clothes_count_l990_99031

/-- Calculates the total number of clothes Martha takes home from a shopping trip -/
def total_clothes (jackets_bought : ℕ) (tshirts_bought : ℕ) : ℕ :=
  let free_jackets := jackets_bought / 2
  let free_tshirts := tshirts_bought / 3
  jackets_bought + free_jackets + tshirts_bought + free_tshirts

/-- Proves that Martha takes home 18 clothes given the conditions of the problem -/
theorem martha_clothes_count :
  total_clothes 4 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_clothes_count_l990_99031


namespace NUMINAMATH_CALUDE_water_balloon_problem_l990_99076

/-- The number of water balloons that popped on the ground --/
def popped_balloons (max_rate max_time zach_rate zach_time total_filled : ℕ) : ℕ :=
  max_rate * max_time + zach_rate * zach_time - total_filled

theorem water_balloon_problem :
  popped_balloons 2 30 3 40 170 = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_problem_l990_99076


namespace NUMINAMATH_CALUDE_conference_hall_tables_l990_99013

/-- Represents the number of tables in the conference hall -/
def num_tables : ℕ := 16

/-- Represents the number of stools per table -/
def stools_per_table : ℕ := 8

/-- Represents the number of chairs per table -/
def chairs_per_table : ℕ := 4

/-- Represents the number of legs per stool -/
def legs_per_stool : ℕ := 3

/-- Represents the number of legs per chair -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs per table -/
def legs_per_table : ℕ := 4

/-- Represents the total number of legs for all furniture -/
def total_legs : ℕ := 704

theorem conference_hall_tables :
  num_tables * (stools_per_table * legs_per_stool + 
                chairs_per_table * legs_per_chair + 
                legs_per_table) = total_legs :=
by sorry

end NUMINAMATH_CALUDE_conference_hall_tables_l990_99013


namespace NUMINAMATH_CALUDE_expression_equals_negative_two_over_tan_l990_99011

theorem expression_equals_negative_two_over_tan (α : Real) 
  (h : α ∈ Set.Ioo π (3 * π / 2)) : 
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) - 
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  -2 / Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_two_over_tan_l990_99011


namespace NUMINAMATH_CALUDE_sum_difference_squares_l990_99046

theorem sum_difference_squares (x y : ℝ) 
  (h1 : x > y) 
  (h2 : x + y = 10) 
  (h3 : x - y = 19) : 
  (x + y)^2 - (x - y)^2 = -261 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_squares_l990_99046


namespace NUMINAMATH_CALUDE_inequality_proof_l990_99005

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l990_99005


namespace NUMINAMATH_CALUDE_fraction_equality_l990_99051

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (4*a + 2*b) / (2*a - 4*b) = 3) : 
  (2*a + 4*b) / (4*a - 2*b) = 9/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l990_99051


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l990_99088

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 2| + |x - 4| + |x - 6| - |2*x - 6|

-- Define the domain of x
def domain (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max min : ℝ), 
    (∀ x, domain x → g x ≤ max) ∧
    (∃ x, domain x ∧ g x = max) ∧
    (∀ x, domain x → min ≤ g x) ∧
    (∃ x, domain x ∧ g x = min) ∧
    max + min = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l990_99088


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_characterization_l990_99089

/-- A quadrilateral is cyclic if and only if the sum of products of opposite angles equals π². -/
theorem cyclic_quadrilateral_characterization (α β γ δ : Real) 
  (h_angles : α + β + γ + δ = 2 * Real.pi) : 
  (α + γ = Real.pi ∧ β + δ = Real.pi) ↔ α * β + α * δ + γ * β + γ * δ = Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_characterization_l990_99089


namespace NUMINAMATH_CALUDE_expected_hits_greater_than_half_l990_99048

/-- The expected number of hit targets is always greater than or equal to half the number of boys/targets. -/
theorem expected_hits_greater_than_half (n : ℕ) (hn : n > 0) :
  n * (1 - (1 - 1 / n)^n) ≥ n / 2 := by
  sorry

#check expected_hits_greater_than_half

end NUMINAMATH_CALUDE_expected_hits_greater_than_half_l990_99048


namespace NUMINAMATH_CALUDE_total_apples_in_pile_l990_99012

def initial_apples : ℕ := 8
def added_apples : ℕ := 5
def package_size : ℕ := 11

theorem total_apples_in_pile :
  initial_apples + added_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_in_pile_l990_99012


namespace NUMINAMATH_CALUDE_problem_solution_l990_99099

theorem problem_solution (a : ℝ) (h : a^2 + a = 0) : a^2011 + a^2010 + 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l990_99099


namespace NUMINAMATH_CALUDE_bracelet_sale_earnings_l990_99069

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  total_bracelets : ℕ
  single_price : ℕ
  pair_price : ℕ
  single_sales : ℕ

/-- Calculates the total earnings from selling bracelets -/
def total_earnings (sale : BraceletSale) : ℕ :=
  let remaining_bracelets := sale.total_bracelets - (sale.single_sales / sale.single_price)
  let pair_sales := remaining_bracelets / 2
  (sale.single_sales / sale.single_price) * sale.single_price + pair_sales * sale.pair_price

/-- Theorem stating that the total earnings from the given scenario is $132 -/
theorem bracelet_sale_earnings :
  let sale : BraceletSale := {
    total_bracelets := 30,
    single_price := 5,
    pair_price := 8,
    single_sales := 60
  }
  total_earnings sale = 132 := by sorry

end NUMINAMATH_CALUDE_bracelet_sale_earnings_l990_99069


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l990_99006

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l990_99006


namespace NUMINAMATH_CALUDE_initial_distance_of_specific_program_l990_99071

/-- Represents a running program with weekly increments -/
structure RunningProgram where
  initial_distance : ℕ  -- Initial daily running distance
  weeks : ℕ             -- Number of weeks in the program
  increment : ℕ         -- Weekly increment in daily distance

/-- Calculates the final daily running distance after the program -/
def final_distance (program : RunningProgram) : ℕ :=
  program.initial_distance + (program.weeks - 1) * program.increment

/-- Theorem stating the initial distance given the conditions -/
theorem initial_distance_of_specific_program :
  ∃ (program : RunningProgram),
    program.weeks = 5 ∧
    program.increment = 1 ∧
    final_distance program = 7 ∧
    program.initial_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_of_specific_program_l990_99071


namespace NUMINAMATH_CALUDE_max_value_interval_condition_l990_99087

/-- The function f(x) = (1/3)x^3 - x has a maximum value on the interval (2m, 1-m) if and only if m ∈ [-1, -1/2). -/
theorem max_value_interval_condition (m : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Ioo (2*m) (1-m) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo (2*m) (1-m) → 
      (1/3 * x^3 - x) ≥ (1/3 * y^3 - y))) ↔ 
  m ∈ Set.Icc (-1) (-1/2) := by
sorry

end NUMINAMATH_CALUDE_max_value_interval_condition_l990_99087


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_star_l990_99027

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - a*b - b^2

-- State the theorem
theorem sin_cos_pi_12_star : 
  star (Real.sin (π/12)) (Real.cos (π/12)) = -(1 + 2*Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_star_l990_99027


namespace NUMINAMATH_CALUDE_polynomial_equality_l990_99003

theorem polynomial_equality (s t : ℝ) : -1/4 * s * t + 0.25 * s * t = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l990_99003


namespace NUMINAMATH_CALUDE_mirror_area_l990_99055

/-- The area of a rectangular mirror inside a frame with given external dimensions and frame width -/
theorem mirror_area (frame_height frame_width frame_side_width : ℝ) :
  frame_height = 100 ∧ 
  frame_width = 140 ∧ 
  frame_side_width = 15 →
  (frame_height - 2 * frame_side_width) * (frame_width - 2 * frame_side_width) = 7700 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l990_99055


namespace NUMINAMATH_CALUDE_suv_max_distance_l990_99070

/-- Represents the fuel efficiency of an SUV in different driving conditions -/
structure SUVFuelEfficiency where
  highway : Float
  city : Float

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def maxDistance (efficiency : SUVFuelEfficiency) (fuel : Float) : Float :=
  efficiency.highway * fuel

/-- Theorem stating the maximum distance an SUV can travel with given efficiency and fuel -/
theorem suv_max_distance (efficiency : SUVFuelEfficiency) (fuel : Float) :
  efficiency.highway = 12.2 →
  efficiency.city = 7.6 →
  fuel = 24 →
  maxDistance efficiency fuel = 292.8 := by
  sorry

end NUMINAMATH_CALUDE_suv_max_distance_l990_99070


namespace NUMINAMATH_CALUDE_room_width_calculation_l990_99023

/-- Given a rectangular room with length 5.5 m, and a total paving cost of 12375 at a rate of 600 per sq. meter, the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) (width : ℝ) : 
  length = 5.5 → 
  total_cost = 12375 → 
  cost_per_sqm = 600 → 
  width = total_cost / (length * cost_per_sqm) → 
  width = 3.75 := by
sorry

#eval (12375 : Float) / (5.5 * 600) -- Evaluates to 3.75

end NUMINAMATH_CALUDE_room_width_calculation_l990_99023


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_inequality_l990_99044

theorem sum_of_reciprocals_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 1) + 1 / (5 * b^2 - 4 * b + 1) + 1 / (5 * c^2 - 4 * c + 1)) ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_inequality_l990_99044


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l990_99086

theorem number_exceeding_percentage (x : ℝ) : x = 0.2 * x + 40 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l990_99086


namespace NUMINAMATH_CALUDE_claps_per_second_is_seventeen_l990_99058

/-- The number of claps achieved in one minute -/
def claps_per_minute : ℕ := 1020

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of claps per second -/
def claps_per_second : ℚ := claps_per_minute / seconds_per_minute

theorem claps_per_second_is_seventeen : 
  claps_per_second = 17 := by sorry

end NUMINAMATH_CALUDE_claps_per_second_is_seventeen_l990_99058


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l990_99061

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/4
  let a₂ : ℚ := 28/9
  let a₃ : ℚ := 112/27
  let r : ℚ := a₂ / a₁
  r = 16/9 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l990_99061


namespace NUMINAMATH_CALUDE_lcm_20_45_75_l990_99080

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_20_45_75_l990_99080


namespace NUMINAMATH_CALUDE_straight_insertion_sort_four_steps_l990_99037

def initial_sequence : List Int := [7, 1, 3, 12, 8, 4, 9, 10]

def straight_insertion_sort (list : List Int) : List Int :=
  sorry

def first_four_steps (list : List Int) : List Int :=
  (straight_insertion_sort list).take 4

theorem straight_insertion_sort_four_steps :
  first_four_steps initial_sequence = [1, 3, 4, 7, 8, 12, 9, 10] :=
sorry

end NUMINAMATH_CALUDE_straight_insertion_sort_four_steps_l990_99037


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_max_a_value_exists_x_for_a_eq_3_l990_99056

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the maximum value of a
theorem max_a_value (a : ℝ) :
  (∃ x : ℝ, f x ≤ -a^2 + a + 7) → a ≤ 3 := by sorry

-- Theorem that 3 is indeed the maximum value
theorem exists_x_for_a_eq_3 :
  ∃ x : ℝ, f x ≤ -3^2 + 3 + 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_max_a_value_exists_x_for_a_eq_3_l990_99056


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l990_99067

/-- The perimeter of a semi-circle with radius 6.83 cm is approximately 35.12 cm. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 6.83
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 35.12) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l990_99067


namespace NUMINAMATH_CALUDE_clothes_fraction_is_one_eighth_l990_99042

/-- The fraction of Gina's initial money used to buy clothes -/
def fraction_for_clothes (initial_amount : ℚ) 
  (fraction_to_mom : ℚ) (fraction_to_charity : ℚ) (amount_kept : ℚ) : ℚ :=
  let amount_to_mom := initial_amount * fraction_to_mom
  let amount_to_charity := initial_amount * fraction_to_charity
  let amount_for_clothes := initial_amount - amount_to_mom - amount_to_charity - amount_kept
  amount_for_clothes / initial_amount

theorem clothes_fraction_is_one_eighth :
  fraction_for_clothes 400 (1/4) (1/5) 170 = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_clothes_fraction_is_one_eighth_l990_99042


namespace NUMINAMATH_CALUDE_wrong_to_correct_ratio_l990_99010

theorem wrong_to_correct_ratio (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36)
  (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_correct_ratio_l990_99010


namespace NUMINAMATH_CALUDE_quadratic_solution_base_n_l990_99063

/-- Given an integer n > 8, if n is a solution of x^2 - ax + b = 0 where a in base-n is 21,
    then b in base-n is 101. -/
theorem quadratic_solution_base_n (n : ℕ) (a b : ℕ) (h1 : n > 8) 
  (h2 : n^2 - a*n + b = 0) (h3 : a = 2*n + 1) : 
  b = n^2 + n := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_base_n_l990_99063


namespace NUMINAMATH_CALUDE_animal_farm_count_l990_99079

theorem animal_farm_count (total_legs : ℕ) (chicken_count : ℕ) : 
  total_legs = 26 →
  chicken_count = 5 →
  ∃ (buffalo_count : ℕ),
    2 * chicken_count + 4 * buffalo_count = total_legs ∧
    chicken_count + buffalo_count = 9 :=
by sorry

end NUMINAMATH_CALUDE_animal_farm_count_l990_99079


namespace NUMINAMATH_CALUDE_day_150_of_year_n_minus_2_is_thursday_l990_99039

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Represents a day in a year -/
structure DayInYear where
  day : ℕ
  year : Year

def is_leap_year (y : Year) : Prop :=
  sorry

def day_of_week (d : DayInYear) : DayOfWeek :=
  sorry

theorem day_150_of_year_n_minus_2_is_thursday
  (N : Year)
  (h1 : day_of_week ⟨256, N⟩ = DayOfWeek.Wednesday)
  (h2 : is_leap_year ⟨N.value + 1⟩)
  (h3 : day_of_week ⟨164, ⟨N.value + 1⟩⟩ = DayOfWeek.Wednesday) :
  day_of_week ⟨150, ⟨N.value - 2⟩⟩ = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_day_150_of_year_n_minus_2_is_thursday_l990_99039


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_one_nonzero_l990_99002

theorem square_sum_nonzero_iff_one_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_one_nonzero_l990_99002


namespace NUMINAMATH_CALUDE_exchange_rate_proof_l990_99001

-- Define the given quantities
def jack_pounds : ℝ := 42
def jack_euros : ℝ := 11
def jack_yen : ℝ := 3000
def pounds_per_euro : ℝ := 2
def total_yen : ℝ := 9400

-- Define the exchange rate we want to prove
def yen_per_pound : ℝ := 100

-- Theorem statement
theorem exchange_rate_proof :
  (jack_pounds + jack_euros * pounds_per_euro) * yen_per_pound + jack_yen = total_yen :=
by sorry

end NUMINAMATH_CALUDE_exchange_rate_proof_l990_99001


namespace NUMINAMATH_CALUDE_chord_of_ellipse_l990_99020

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 2*y^2 - 4 = 0

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The midpoint of a line segment -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem chord_of_ellipse :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  ellipse_equation x₁ y₁ ∧ 
  ellipse_equation x₂ y₂ ∧
  (∀ x y : ℝ, line_equation x y ↔ is_midpoint 1 1 x₁ y₁ x₂ y₂) →
  line_equation x₁ y₁ ∧ line_equation x₂ y₂ := by sorry

end NUMINAMATH_CALUDE_chord_of_ellipse_l990_99020


namespace NUMINAMATH_CALUDE_f_property_l990_99091

noncomputable def f (x : ℝ) : ℝ := (4^x) / (4^x + 2)

theorem f_property (x : ℝ) :
  f x + f (1 - x) = 1 ∧
  (2 * (f x)^2 < f (1 - x) ↔ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_f_property_l990_99091


namespace NUMINAMATH_CALUDE_range_of_special_set_l990_99057

def three_number_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c

theorem range_of_special_set (a b c : ℝ) 
  (h_set : three_number_set a b c)
  (h_mean : (a + b + c) / 3 = 6)
  (h_median : b = 6)
  (h_min : a = 2) :
  c - a = 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_special_set_l990_99057


namespace NUMINAMATH_CALUDE_seeds_sown_count_l990_99083

/-- The number of seeds that germinated -/
def seeds_germinated : ℕ := 970

/-- The frequency of normal seed germination -/
def germination_rate : ℚ := 97/100

/-- The total number of seeds sown -/
def total_seeds : ℕ := 1000

/-- Theorem stating that the total number of seeds sown is 1000 -/
theorem seeds_sown_count : 
  (seeds_germinated : ℚ) / germination_rate = total_seeds := by sorry

end NUMINAMATH_CALUDE_seeds_sown_count_l990_99083


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l990_99065

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - x - 5
  let x₁ : ℝ := (1 + Real.sqrt 21) / 2
  let x₂ : ℝ := (1 - Real.sqrt 21) / 2
  f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l990_99065


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l990_99007

theorem complex_modulus_problem (a : ℝ) (z : ℂ) : 
  z = (1 - a * Complex.I) / (1 + Complex.I) →
  z.re = -2 →
  Complex.abs z = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l990_99007


namespace NUMINAMATH_CALUDE_equality_comparison_l990_99078

theorem equality_comparison : 
  (-2^2 ≠ (-2)^2) ∧ 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  ((-3 * 2)^2 ≠ -3^2 * 2^2) := by
  sorry

end NUMINAMATH_CALUDE_equality_comparison_l990_99078


namespace NUMINAMATH_CALUDE_blue_then_red_probability_l990_99085

/-- The probability of drawing a blue marble first and a red marble second -/
theorem blue_then_red_probability (red white blue : ℕ) 
  (h_red : red = 4)
  (h_white : white = 6)
  (h_blue : blue = 2) : 
  (blue : ℚ) / (red + white + blue) * red / (red + white + blue - 1) = 2 / 33 := by
sorry

end NUMINAMATH_CALUDE_blue_then_red_probability_l990_99085


namespace NUMINAMATH_CALUDE_hexagon_ratio_l990_99032

/-- A hexagon with specific properties -/
structure Hexagon where
  area : ℝ
  below_rs_area : ℝ
  triangle_base : ℝ
  xr : ℝ
  rs : ℝ

/-- The theorem statement -/
theorem hexagon_ratio (h : Hexagon) (h_area : h.area = 13)
  (h_bisect : h.below_rs_area = h.area / 2)
  (h_below : h.below_rs_area = 2 + (h.triangle_base * (h.below_rs_area - 2) / h.triangle_base) / 2)
  (h_base : h.triangle_base = 4)
  (h_sum : h.xr + h.rs = h.triangle_base) :
  h.xr / h.rs = 1 := by sorry

end NUMINAMATH_CALUDE_hexagon_ratio_l990_99032


namespace NUMINAMATH_CALUDE_model_x_completion_time_l990_99033

/-- The time (in minutes) it takes for a Model Y computer to complete the task -/
def model_y_time : ℝ := 30

/-- The number of Model X computers used -/
def num_model_x : ℕ := 20

/-- The time (in minutes) it takes to complete the task when using equal numbers of both models -/
def combined_time : ℝ := 1

/-- The time (in minutes) it takes for a Model X computer to complete the task -/
def model_x_time : ℝ := 60

theorem model_x_completion_time :
  (num_model_x : ℝ) * (1 / model_x_time + 1 / model_y_time) = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_model_x_completion_time_l990_99033


namespace NUMINAMATH_CALUDE_complex_number_equation_l990_99075

/-- Given a complex number z = 1 + √2i, prove that z^2 - 2z = -3 -/
theorem complex_number_equation : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 2
  z^2 - 2*z = -3 := by sorry

end NUMINAMATH_CALUDE_complex_number_equation_l990_99075
