import Mathlib

namespace NUMINAMATH_CALUDE_max_wickets_bowler_l2160_216029

/-- Represents the maximum number of wickets a bowler can take in an over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the number of overs bowled by the bowler in an innings -/
def overs_bowled : ℕ := 6

/-- Represents the total number of players in a cricket team -/
def players_in_team : ℕ := 11

/-- Represents the maximum number of wickets that can be taken in an innings -/
def max_wickets_in_innings : ℕ := players_in_team - 1

/-- Theorem stating the maximum number of wickets a bowler can take in an innings -/
theorem max_wickets_bowler (wickets_per_over : ℕ) (overs : ℕ) (team_size : ℕ) :
  wickets_per_over = max_wickets_per_over →
  overs = overs_bowled →
  team_size = players_in_team →
  min (wickets_per_over * overs) (team_size - 1) = max_wickets_in_innings := by
  sorry

end NUMINAMATH_CALUDE_max_wickets_bowler_l2160_216029


namespace NUMINAMATH_CALUDE_solutions_are_correct_l2160_216079

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x^2 - 5 * x - 2 = 0
def equation2 (x : ℝ) : Prop := x^2 - 1 = 2 * (x + 1)
def equation3 (x : ℝ) : Prop := 4 * x^2 + 4 * x + 1 = 3 * (3 - x)^2
def equation4 (x : ℝ) : Prop := (2 * x + 8) * (x - 2) = x^2 + 2 * x - 17

-- Theorem stating the solutions are correct
theorem solutions_are_correct :
  (equation1 (-1/3) ∧ equation1 2) ∧
  (equation2 (-1) ∧ equation2 3) ∧
  (equation3 (-11 + 7 * Real.sqrt 3) ∧ equation3 (-11 - 7 * Real.sqrt 3)) ∧
  (equation4 (-1)) := by
  sorry

#check solutions_are_correct

end NUMINAMATH_CALUDE_solutions_are_correct_l2160_216079


namespace NUMINAMATH_CALUDE_tan_value_for_given_sin_cos_sum_l2160_216068

theorem tan_value_for_given_sin_cos_sum (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = Real.sqrt 5 / 5)
  (h2 : θ ∈ Set.Icc 0 Real.pi) : 
  Real.tan θ = -2 := by
sorry

end NUMINAMATH_CALUDE_tan_value_for_given_sin_cos_sum_l2160_216068


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l2160_216027

theorem percentage_error_calculation (x y : ℝ) (hx : x ≠ 0) (hy : y + 15 ≠ 0) :
  let error_x := ((x * 10 - x / 10) / (x * 10)) * 100
  let error_y := (30 / (y + 15)) * 100
  let total_error := ((10 * x - x / 10 + 30) / (10 * x + y + 15)) * 100
  (error_x = 99) ∧
  (error_y = (30 / (y + 15)) * 100) ∧
  (total_error = ((10 * x - x / 10 + 30) / (10 * x + y + 15)) * 100) := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l2160_216027


namespace NUMINAMATH_CALUDE_partnership_profit_l2160_216039

/-- 
Given a business partnership between Mary and Mike:
- Mary invests $700
- Mike invests $300
- 1/3 of profit is divided equally for efforts
- 2/3 of profit is divided in ratio of investments (7:3)
- Mary received $800 more than Mike

This theorem proves that the total profit P satisfies the equation:
[(P/6) + (7/10) * (2P/3)] - [(P/6) + (3/10) * (2P/3)] = 800
-/
theorem partnership_profit (P : ℝ) : 
  (P / 6 + 7 / 10 * (2 * P / 3)) - (P / 6 + 3 / 10 * (2 * P / 3)) = 800 → 
  P = 3000 := by
sorry


end NUMINAMATH_CALUDE_partnership_profit_l2160_216039


namespace NUMINAMATH_CALUDE_f_neg_two_value_l2160_216087

def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + x^2

theorem f_neg_two_value (f : ℝ → ℝ) :
  (∀ x, F f x = -F f (-x)) →  -- F is an odd function
  f 2 = 1 →                   -- f(2) = 1
  f (-2) = -9 :=               -- Conclusion: f(-2) = -9
by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_value_l2160_216087


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_3_l2160_216007

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - a

/-- f is monotonically decreasing on (-1, 1) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo (-1) 1, f_deriv a x ≤ 0

theorem monotone_decreasing_implies_a_geq_3 :
  ∀ a : ℝ, is_monotone_decreasing a → a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_3_l2160_216007


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2160_216023

theorem consecutive_integers_average (n : ℤ) : 
  (n * (n + 6) = 391) → 
  (((n + n + 1 + n + 2 + n + 3 + n + 4 + n + 5 + n + 6) : ℚ) / 7 = 20) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2160_216023


namespace NUMINAMATH_CALUDE_marbles_left_l2160_216013

theorem marbles_left (initial_red : ℕ) (initial_blue : ℕ) (red_taken : ℕ) (blue_taken : ℕ) : 
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  blue_taken = 4 * red_taken →
  initial_red - red_taken + initial_blue - blue_taken = 35 := by
sorry

end NUMINAMATH_CALUDE_marbles_left_l2160_216013


namespace NUMINAMATH_CALUDE_candy_cost_in_dollars_l2160_216046

/-- The cost of a single piece of candy in cents -/
def candy_cost : ℕ := 2

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of candy pieces we're calculating the cost for -/
def candy_pieces : ℕ := 500

theorem candy_cost_in_dollars : 
  (candy_pieces * candy_cost) / cents_per_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_in_dollars_l2160_216046


namespace NUMINAMATH_CALUDE_monkey_pole_height_l2160_216017

/-- Calculates the height of a pole given the ascent pattern and time taken by a monkey to reach the top -/
def poleHeight (ascent : ℕ) (descent : ℕ) (totalTime : ℕ) : ℕ :=
  let fullCycles := (totalTime - 1) / 2
  let remainingDistance := ascent
  fullCycles * (ascent - descent) + remainingDistance

/-- The height of the pole given the monkey's climbing pattern and time -/
theorem monkey_pole_height : poleHeight 2 1 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_monkey_pole_height_l2160_216017


namespace NUMINAMATH_CALUDE_marble_game_solution_l2160_216081

/-- Represents a player in the game -/
inductive Player
| A
| B
| C

/-- Represents the game state -/
structure GameState where
  p : ℕ
  q : ℕ
  r : ℕ
  rounds : ℕ
  final_marbles : Player → ℕ
  b_last_round : ℕ

/-- The theorem statement -/
theorem marble_game_solution (g : GameState) 
  (h1 : g.p < g.q ∧ g.q < g.r)
  (h2 : g.rounds ≥ 2)
  (h3 : g.final_marbles Player.A = 20)
  (h4 : g.final_marbles Player.B = 10)
  (h5 : g.final_marbles Player.C = 9)
  (h6 : g.b_last_round = g.r) :
  ∃ (first_round : Player → ℕ), first_round Player.B = 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_game_solution_l2160_216081


namespace NUMINAMATH_CALUDE_michaels_initial_money_l2160_216050

def total_cost : ℕ := 61
def additional_needed : ℕ := 11

theorem michaels_initial_money :
  total_cost - additional_needed = 50 := by
  sorry

end NUMINAMATH_CALUDE_michaels_initial_money_l2160_216050


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_is_four_min_value_achieved_min_value_fraction_is_four_l2160_216077

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x * y * z) ≤ (a + b) / (a * b * c) :=
by sorry

theorem min_value_is_four (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  (x + y) / (x * y * z) ≥ 4 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ (a + b) / (a * b * c) = 4 :=
by sorry

theorem min_value_fraction_is_four :
  ∃ m : ℝ, m = 4 ∧ 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 2 → (x + y) / (x * y * z) ≥ m) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ (a + b) / (a * b * c) = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_is_four_min_value_achieved_min_value_fraction_is_four_l2160_216077


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_one_l2160_216015

theorem simplify_and_evaluate (m : ℝ) (h1 : m ≠ -3) (h2 : m ≠ 3) (h3 : m ≠ 0) :
  (m / (m + 3) - 2 * m / (m - 3)) / (m / (m^2 - 9)) = -m - 9 :=
by sorry

-- Evaluation at m = 1
theorem evaluate_at_one :
  (1 / (1 + 3) - 2 * 1 / (1 - 3)) / (1 / (1^2 - 9)) = -10 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_one_l2160_216015


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2160_216043

/-- In a triangle ABC, given side lengths and an angle, prove that angle B has two possible values. -/
theorem triangle_angle_B (a b : ℝ) (A B : ℝ) : 
  a = (5 * Real.sqrt 3) / 3 → 
  b = 5 → 
  A = π / 6 → 
  (B = π / 3 ∨ B = 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2160_216043


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_l2160_216049

/-- The total amount Jason spent on clothing -/
def total_spent : ℚ := 19.02

/-- The amount Jason spent on shorts -/
def shorts_cost : ℚ := 14.28

/-- The amount Jason spent on a jacket -/
def jacket_cost : ℚ := 4.74

/-- Theorem stating that the total amount spent equals the sum of shorts and jacket costs -/
theorem total_spent_equals_sum : total_spent = shorts_cost + jacket_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_l2160_216049


namespace NUMINAMATH_CALUDE_jean_trips_l2160_216022

theorem jean_trips (total : ℕ) (extra : ℕ) (h1 : total = 40) (h2 : extra = 6) :
  ∃ (bill : ℕ) (jean : ℕ), bill + jean = total ∧ jean = bill + extra ∧ jean = 23 :=
by sorry

end NUMINAMATH_CALUDE_jean_trips_l2160_216022


namespace NUMINAMATH_CALUDE_max_abs_ab_l2160_216019

/-- The quadratic function f(x) -/
def f (a b c x : ℝ) : ℝ := a * (3 * a + 2 * c) * x^2 - 2 * b * (2 * a + c) * x + b^2 + (c + a)^2

/-- Theorem stating the maximum value of |ab| given the conditions -/
theorem max_abs_ab (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≤ 1) : 
  |a * b| ≤ 3 * Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_ab_l2160_216019


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2160_216020

theorem cube_volume_ratio (edge_ratio : ℝ) (small_volume : ℝ) :
  edge_ratio = 4.999999999999999 →
  small_volume = 1 →
  (edge_ratio ^ 3) * small_volume = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2160_216020


namespace NUMINAMATH_CALUDE_square_plot_area_l2160_216055

/-- Proves that a square plot with given fencing costs has an area of 36 square feet -/
theorem square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) :
  cost_per_foot = 58 →
  total_cost = 1392 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    4 * side_length * cost_per_foot = total_cost ∧
    side_length ^ 2 = 36 := by
  sorry

#check square_plot_area

end NUMINAMATH_CALUDE_square_plot_area_l2160_216055


namespace NUMINAMATH_CALUDE_ratio_equality_l2160_216031

theorem ratio_equality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2160_216031


namespace NUMINAMATH_CALUDE_curve_symmetry_l2160_216003

/-- The curve represented by the equation xy(x+y)=1 is symmetric about the line y=x -/
theorem curve_symmetry (x y : ℝ) : x * y * (x + y) = 1 ↔ y * x * (y + x) = 1 := by sorry

end NUMINAMATH_CALUDE_curve_symmetry_l2160_216003


namespace NUMINAMATH_CALUDE_product_of_roots_rational_l2160_216038

-- Define the polynomial
def polynomial (a b c d e : ℤ) (z : ℂ) : ℂ :=
  a * z^4 + b * z^3 + c * z^2 + d * z + e

-- Define the theorem
theorem product_of_roots_rational
  (a b c d e : ℤ)
  (r₁ r₂ r₃ r₄ : ℂ)
  (h₁ : a ≠ 0)
  (h₂ : polynomial a b c d e r₁ = 0)
  (h₃ : polynomial a b c d e r₂ = 0)
  (h₄ : polynomial a b c d e r₃ = 0)
  (h₅ : polynomial a b c d e r₄ = 0)
  (h₆ : ∃ q : ℚ, r₁ + r₂ = q)
  (h₇ : r₃ + r₄ ≠ r₁ + r₂)
  : ∃ q : ℚ, r₁ * r₂ = q :=
sorry

end NUMINAMATH_CALUDE_product_of_roots_rational_l2160_216038


namespace NUMINAMATH_CALUDE_pascal_triangle_probability_l2160_216063

/-- Pascal's Triangle up to the nth row -/
def pascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Count the number of elements equal to k in the first n rows of Pascal's Triangle -/
def countElements (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ :=
  sorry

theorem pascal_triangle_probability :
  let n := 20
  let ones := countElements n 1
  let twos := countElements n 2
  let total := totalElements n
  (ones + twos : ℚ) / total = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_probability_l2160_216063


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2160_216082

/-- A complex number z is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_fraction_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((a + 3 * Complex.I) / (1 + 2 * Complex.I)) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2160_216082


namespace NUMINAMATH_CALUDE_inverse_modulo_31_l2160_216033

theorem inverse_modulo_31 (h : (17⁻¹ : ZMod 31) = 13) : (21⁻¹ : ZMod 31) = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_modulo_31_l2160_216033


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l2160_216095

theorem sum_of_specific_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l2160_216095


namespace NUMINAMATH_CALUDE_cubic_equation_transformation_and_solutions_l2160_216018

theorem cubic_equation_transformation_and_solutions 
  (p q : ℝ) 
  (hp : p < 0) 
  (hpq : 4 * p^3 + 27 * q^2 ≤ 0) : 
  ∃ (k r : ℝ), 
    k = 2 * Real.sqrt (-p/3) ∧ 
    r = q / (2 * Real.sqrt (-p^3/27)) ∧
    (∀ x, x^3 + p*x + q = 0 ↔ (∃ t, x = k*t ∧ 4*t^3 - 3*t - r = 0)) ∧
    (∃ φ, r = Real.cos φ ∧
      (∀ t, 4*t^3 - 3*t - r = 0 ↔ 
        t = Real.cos (φ/3) ∨ 
        t = Real.cos ((φ + 2*Real.pi)/3) ∨ 
        t = Real.cos ((φ + 4*Real.pi)/3))) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_transformation_and_solutions_l2160_216018


namespace NUMINAMATH_CALUDE_solve_beef_problem_l2160_216012

def beef_problem (pounds_per_pack : ℝ) (price_per_pound : ℝ) (total_paid : ℝ) : Prop :=
  let price_per_pack := pounds_per_pack * price_per_pound
  let num_packs := total_paid / price_per_pack
  num_packs = 5

theorem solve_beef_problem :
  beef_problem 4 5.50 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_beef_problem_l2160_216012


namespace NUMINAMATH_CALUDE_sector_max_area_angle_l2160_216073

/-- Given a sector with circumference 20, the radian measure of its central angle is 2 when the area of the sector is maximized. -/
theorem sector_max_area_angle (r : ℝ) (θ : ℝ) : 
  0 < r ∧ r < 10 →  -- Ensure r is in the valid range
  r * θ + 2 * r = 20 →  -- Circumference of sector is 20
  (∀ r' θ' : ℝ, 0 < r' ∧ r' < 10 → r' * θ' + 2 * r' = 20 → 
    r * θ / 2 ≥ r' * θ' / 2) →  -- Area is maximized
  θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_angle_l2160_216073


namespace NUMINAMATH_CALUDE_inequality_implies_range_l2160_216088

theorem inequality_implies_range (a : ℝ) : 
  (∀ x : ℝ, |2*x + 1| + |x - 2| ≥ a^2 - a + 1/2) → 
  -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l2160_216088


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l2160_216061

/-- The equation of a line passing through the intersection of two lines and parallel to a third line -/
theorem intersection_parallel_line (a b c d e f g h i j : ℝ) :
  (∃ x y, a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →  -- Intersection exists
  (g * x + h * y + i = 0) →                                -- Third line
  (∃ k l m : ℝ, k ≠ 0 ∧
    (∀ x y, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
      k * (g * x + h * y) + l * x + m * y + j = 0)) →      -- Parallel condition
  (∃ p q r : ℝ, p ≠ 0 ∧
    (∀ x y, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
      p * x + q * y + r = 0) ∧                             -- Line through intersection
    ∃ s, p = s * g ∧ q = s * h) →                          -- Parallel to third line
  ∃ t, 2 * x + y = t                                       -- Resulting equation
  := by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l2160_216061


namespace NUMINAMATH_CALUDE_power_function_properties_l2160_216052

noncomputable def f (x : ℝ) : ℝ := x^(2/3)

theorem power_function_properties :
  (∃ a : ℝ, ∀ x : ℝ, f x = x^a) ∧ 
  f 8 = 4 ∧
  f 0 = 0 ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_power_function_properties_l2160_216052


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l2160_216001

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n ≤ 9999 }

/-- Extracts the leftmost digit of a four-digit number -/
def leftmostDigit (n : FourDigitNumber) : ℕ := n.val / 1000

/-- Extracts the three-digit number obtained by removing the leftmost digit -/
def rightThreeDigits (n : FourDigitNumber) : ℕ := n.val % 1000

/-- Checks if a four-digit number satisfies the given property -/
def satisfiesProperty (n : FourDigitNumber) : Prop :=
  7 * (rightThreeDigits n) = n.val

theorem count_numbers_with_property :
  ∃ (S : Finset FourDigitNumber),
    (∀ n ∈ S, satisfiesProperty n) ∧
    (∀ n : FourDigitNumber, satisfiesProperty n → n ∈ S) ∧
    Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l2160_216001


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2160_216016

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (b > a ∧ a > 0 → (1 : ℝ) / a^2 > (1 : ℝ) / b^2) ∧
  ∃ a b : ℝ, (1 : ℝ) / a^2 > (1 : ℝ) / b^2 ∧ ¬(b > a ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2160_216016


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2160_216032

theorem geometric_sequence_sum (n : ℕ) (a : ℝ) (r : ℝ) (S : ℝ) : 
  a = 1 → r = (1/2 : ℝ) → S = (31/16 : ℝ) → S = a * (1 - r^n) / (1 - r) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2160_216032


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2160_216048

theorem mean_equality_implies_x_value : 
  ∃ x : ℝ, (6 + 9 + 18) / 3 = (x + 15) / 2 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2160_216048


namespace NUMINAMATH_CALUDE_no_prime_satisfies_equation_l2160_216090

theorem no_prime_satisfies_equation : 
  ¬ ∃ (p : ℕ), Nat.Prime p ∧ 
  (2 * p^3 + 0 * p^2 + 3 * p + 4) + 
  (4 * p^2 + 0 * p + 5) + 
  (2 * p^2 + 1 * p + 7) + 
  (1 * p^2 + 5 * p + 0) + 
  4 = 
  (3 * p^2 + 0 * p + 2) + 
  (5 * p^2 + 2 * p + 0) + 
  (4 * p^2 + 3 * p + 1) :=
sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_equation_l2160_216090


namespace NUMINAMATH_CALUDE_company_workforce_l2160_216065

theorem company_workforce (initial_workforce : ℕ) : 
  (initial_workforce * 60 = initial_workforce * 100 * 3 / 5) →
  ((initial_workforce * 60 : ℕ) = ((initial_workforce + 28) * 55 : ℕ)) →
  (initial_workforce + 28 = 336) := by
  sorry

end NUMINAMATH_CALUDE_company_workforce_l2160_216065


namespace NUMINAMATH_CALUDE_grandfather_grandmother_age_difference_is_two_l2160_216044

/-- The age difference between Milena's grandfather and grandmother -/
def grandfather_grandmother_age_difference (milena_age : ℕ) (grandmother_age_factor : ℕ) (milena_grandfather_age_difference : ℕ) : ℕ :=
  (milena_age + milena_grandfather_age_difference) - (milena_age * grandmother_age_factor)

theorem grandfather_grandmother_age_difference_is_two :
  grandfather_grandmother_age_difference 7 9 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_grandmother_age_difference_is_two_l2160_216044


namespace NUMINAMATH_CALUDE_circle_symmetry_and_properties_l2160_216078

-- Define the circle C1 and line l
def C1 (m : ℝ) (x y : ℝ) : Prop := (x + 1)^2 + (y - 3*m - 3)^2 = 4*m^2
def l (m : ℝ) (x y : ℝ) : Prop := y = x + m + 2

-- Define the circle C2
def C2 (m : ℝ) (x y : ℝ) : Prop := (x - 2*m - 1)^2 + (y - m - 1)^2 = 4*m^2

-- Define the line on which centers of C2 lie
def centerLine (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the common tangent line
def commonTangent (x y : ℝ) : Prop := y = -3/4 * x + 7/4

theorem circle_symmetry_and_properties 
  (m : ℝ) (h : m ≠ 0) :
  (∀ x y, C2 m x y ↔ 
    ∃ x' y', C1 m x' y' ∧ l m ((x + x') / 2) ((y + y') / 2)) ∧ 
  (∀ m x y, C2 m x y → centerLine x y) ∧
  (∀ m x y, C2 m x y → ∃ x₀ y₀, commonTangent x₀ y₀ ∧ 
    (x₀ - x)^2 + (y₀ - y)^2 = ((x - (2*m + 1))^2 + (y - (m + 1))^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_and_properties_l2160_216078


namespace NUMINAMATH_CALUDE_subset_divisibility_property_l2160_216084

theorem subset_divisibility_property (A : Finset ℕ) (hA : A.card = 3) :
  ∃ (B : Finset ℕ) (x y : ℕ), B ⊆ A ∧ B.card = 2 ∧ x ∈ B ∧ y ∈ B ∧
    ∀ (m n : ℕ), Odd m → Odd n →
      (10 : ℤ) ∣ (((x ^ m : ℕ) * (y ^ n : ℕ)) - ((x ^ n : ℕ) * (y ^ m : ℕ))) :=
by sorry


end NUMINAMATH_CALUDE_subset_divisibility_property_l2160_216084


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2160_216036

/-- Represents the amount of lingonberries picked on each day -/
structure BerryPicking where
  monday : ℕ
  tuesday : ℕ
  thursday : ℕ

/-- Represents the job parameters -/
structure JobParameters where
  totalEarningsGoal : ℕ
  payRate : ℕ

theorem tuesday_to_monday_ratio (job : JobParameters) (pick : BerryPicking) :
  job.totalEarningsGoal = 100 ∧
  job.payRate = 2 ∧
  pick.monday = 8 ∧
  pick.thursday = 18 →
  pick.tuesday / pick.monday = 3 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l2160_216036


namespace NUMINAMATH_CALUDE_f_increasing_iff_l2160_216051

/-- Definition of the piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < -1 then (-a + 4) * x - 3 * a
  else x^2 + a * x - 8

/-- Theorem stating the condition for f to be increasing --/
theorem f_increasing_iff (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ 3 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_iff_l2160_216051


namespace NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l2160_216025

theorem no_geometric_progression_of_2n_plus_1 :
  ¬ ∃ (k m n : ℕ), k ≠ m ∧ m ≠ n ∧ k ≠ n ∧
    (2^m + 1)^2 = (2^k + 1) * (2^n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l2160_216025


namespace NUMINAMATH_CALUDE_k_range_theorem_l2160_216054

-- Define the function f(x) = (e^x / x) + x^2 - 2x
noncomputable def f (x : ℝ) : ℝ := (Real.exp x / x) + x^2 - 2*x

theorem k_range_theorem (k : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x / Real.exp x < 1 / (k + 2*x - x^2)) → 
  k ∈ Set.Icc 0 (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l2160_216054


namespace NUMINAMATH_CALUDE_vector_BC_l2160_216067

/-- Given points A(0,1), B(3,2), and vector AC(-4,-3), prove that vector BC is (-7,-4) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (0, 1) → 
  B = (3, 2) → 
  C - A = (-4, -3) → 
  C - B = (-7, -4) := by
sorry

end NUMINAMATH_CALUDE_vector_BC_l2160_216067


namespace NUMINAMATH_CALUDE_triangle_value_l2160_216085

theorem triangle_value (p : ℤ) (h1 : ∃ triangle : ℤ, triangle + p = 47) 
  (h2 : 3 * (47) - p = 133) : 
  ∃ triangle : ℤ, triangle = 39 :=
sorry

end NUMINAMATH_CALUDE_triangle_value_l2160_216085


namespace NUMINAMATH_CALUDE_eleven_row_triangle_pieces_l2160_216005

/-- Calculates the total number of pieces in a triangle with given number of rows -/
def totalPieces (rows : ℕ) : ℕ :=
  let rodSum := (rows * (rows + 1) * 3) / 2
  let connectorSum := (rows + 1) * (rows + 2) / 2
  rodSum + connectorSum

/-- Theorem stating that an eleven-row triangle has 276 pieces -/
theorem eleven_row_triangle_pieces :
  totalPieces 11 = 276 := by sorry

end NUMINAMATH_CALUDE_eleven_row_triangle_pieces_l2160_216005


namespace NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_50_l2160_216071

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to find the nth prime number greater than a given value
def nthPrimeGreaterThan (n : ℕ) (start : ℕ) : ℕ :=
  sorry

theorem least_product_of_three_primes_greater_than_50 :
  ∃ p q r : ℕ,
    isPrime p ∧ isPrime q ∧ isPrime r ∧
    p > 50 ∧ q > 50 ∧ r > 50 ∧
    p < q ∧ q < r ∧
    p * q * r = 191557 ∧
    ∀ a b c : ℕ,
      isPrime a ∧ isPrime b ∧ isPrime c ∧
      a > 50 ∧ b > 50 ∧ c > 50 ∧
      a < b ∧ b < c →
      a * b * c ≥ 191557 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_three_primes_greater_than_50_l2160_216071


namespace NUMINAMATH_CALUDE_power_of_prime_iff_only_prime_factor_l2160_216098

theorem power_of_prime_iff_only_prime_factor (p n : ℕ) : 
  Prime p → (∃ k : ℕ, n = p ^ k) ↔ (∀ q : ℕ, Prime q → q ∣ n → q = p) :=
sorry

end NUMINAMATH_CALUDE_power_of_prime_iff_only_prime_factor_l2160_216098


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2160_216030

theorem fraction_evaluation : 
  (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2160_216030


namespace NUMINAMATH_CALUDE_magnitude_z_l2160_216083

theorem magnitude_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (25 * Real.sqrt 34) / 34 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_l2160_216083


namespace NUMINAMATH_CALUDE_line_circle_relationship_l2160_216045

theorem line_circle_relationship (m : ℝ) :
  ∃ (x y : ℝ), (m * x + y - m - 1 = 0 ∧ x^2 + y^2 = 2) ∨
  ∃ (x y : ℝ), (m * x + y - m - 1 = 0 ∧ x^2 + y^2 = 2 ∧
    ∀ (x' y' : ℝ), m * x' + y' - m - 1 = 0 → x'^2 + y'^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l2160_216045


namespace NUMINAMATH_CALUDE_trig_simplification_trig_value_given_tan_l2160_216093

/-- Proves that the given trigonometric expression simplifies to -1 --/
theorem trig_simplification : 
  (Real.sqrt (1 - 2 * Real.sin (10 * π / 180) * Real.cos (10 * π / 180))) / 
  (Real.sin (170 * π / 180) - Real.sqrt (1 - Real.sin (170 * π / 180) ^ 2)) = -1 := by
  sorry

/-- Proves that given tan θ = 2, the expression 2 + sin θ * cos θ - cos² θ equals 11/5 --/
theorem trig_value_given_tan (θ : Real) (h : Real.tan θ = 2) : 
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_trig_value_given_tan_l2160_216093


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l2160_216096

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 9 * (a + b) →
  (10 * a + b) + (10 * b + a) = 11 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l2160_216096


namespace NUMINAMATH_CALUDE_double_root_values_l2160_216070

theorem double_root_values (b₃ b₂ b₁ : ℤ) (r : ℤ) :
  (∀ x : ℝ, x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 72 = (x - r : ℝ)^2 * ((x - r)^2 + c * (x - r) + d))
  → (r = -6 ∨ r = -3 ∨ r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 6) :=
by sorry

end NUMINAMATH_CALUDE_double_root_values_l2160_216070


namespace NUMINAMATH_CALUDE_elberta_money_l2160_216097

theorem elberta_money (granny_smith : ℕ) (elberta anjou : ℕ) : 
  granny_smith = 72 →
  elberta = anjou + 5 →
  anjou = granny_smith / 4 →
  elberta = 23 := by
sorry

end NUMINAMATH_CALUDE_elberta_money_l2160_216097


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l2160_216002

theorem integer_solutions_quadratic_equation :
  ∀ x y : ℤ, x^2 + 2*x*y + 3*y^2 - 2*x + y + 1 = 0 ↔ 
  (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨ (x = 3 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l2160_216002


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2160_216072

theorem arithmetic_expression_evaluation :
  5 * 7 + 9 * 4 - 30 / 3 + 2^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2160_216072


namespace NUMINAMATH_CALUDE_max_amount_received_back_l2160_216059

/-- Represents the result of a gambling session -/
structure GamblingResult where
  initial_amount : ℕ
  chip_30_value : ℕ
  chip_100_value : ℕ
  total_chips_lost : ℕ
  chip_30_lost : ℕ
  chip_100_lost : ℕ

/-- Calculates the amount received back after a gambling session -/
def amount_received_back (result : GamblingResult) : ℕ :=
  result.initial_amount - (result.chip_30_lost * result.chip_30_value + result.chip_100_lost * result.chip_100_value)

/-- Theorem stating the maximum amount received back under given conditions -/
theorem max_amount_received_back :
  ∀ (result : GamblingResult),
    result.initial_amount = 3000 ∧
    result.chip_30_value = 30 ∧
    result.chip_100_value = 100 ∧
    result.total_chips_lost = 16 ∧
    (result.chip_30_lost = result.chip_100_lost + 2 ∨ result.chip_30_lost = result.chip_100_lost - 2) →
    amount_received_back result ≤ 1890 :=
by sorry

end NUMINAMATH_CALUDE_max_amount_received_back_l2160_216059


namespace NUMINAMATH_CALUDE_exponent_comparison_l2160_216047

theorem exponent_comparison : 65^1000 - 8^2001 > 0 := by
  sorry

end NUMINAMATH_CALUDE_exponent_comparison_l2160_216047


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l2160_216024

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 693 :=
by sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l2160_216024


namespace NUMINAMATH_CALUDE_probability_one_defective_is_half_l2160_216075

/-- Represents the total number of items -/
def total_items : Nat := 4

/-- Represents the number of genuine items -/
def genuine_items : Nat := 3

/-- Represents the number of defective items -/
def defective_items : Nat := 1

/-- Represents the number of items to be selected -/
def items_to_select : Nat := 2

/-- Calculates the number of ways to select k items from n items -/
def combinations (n k : Nat) : Nat := sorry

/-- Calculates the probability of selecting exactly one defective item -/
def probability_one_defective : Rat :=
  (combinations defective_items 1 * combinations genuine_items (items_to_select - 1)) /
  (combinations total_items items_to_select)

/-- Theorem stating that the probability of selecting exactly one defective item is 1/2 -/
theorem probability_one_defective_is_half :
  probability_one_defective = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_one_defective_is_half_l2160_216075


namespace NUMINAMATH_CALUDE_point_to_line_distance_l2160_216053

theorem point_to_line_distance (M : ℝ) : 
  (|(3 : ℝ) + Real.sqrt 3 * M - 4| / Real.sqrt (1 + 3) = 1) ↔ 
  (M = Real.sqrt 3 ∨ M = -(Real.sqrt 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_point_to_line_distance_l2160_216053


namespace NUMINAMATH_CALUDE_coordinate_system_change_l2160_216076

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The relative position of two points -/
def relativePosition (p q : Point) : Point :=
  ⟨p.x - q.x, p.y - q.y⟩

theorem coordinate_system_change (A B : Point) :
  relativePosition A B = ⟨2, 5⟩ → relativePosition B A = ⟨-2, -5⟩ := by
  sorry

end NUMINAMATH_CALUDE_coordinate_system_change_l2160_216076


namespace NUMINAMATH_CALUDE_lcm_gcd_difference_even_nonnegative_l2160_216040

theorem lcm_gcd_difference_even_nonnegative (a b : ℕ+) :
  let n := Nat.lcm a b + Nat.gcd a b - a - b
  n % 2 = 0 ∧ n ≥ 0 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_difference_even_nonnegative_l2160_216040


namespace NUMINAMATH_CALUDE_unique_solution_l2160_216034

-- Define the set S
inductive S
  | A
  | A1
  | A2
  | A3
  | A4

-- Define the operation ⊕
def oplus : S → S → S
  | S.A, S.A => S.A
  | S.A, S.A1 => S.A1
  | S.A, S.A2 => S.A2
  | S.A, S.A3 => S.A3
  | S.A, S.A4 => S.A4
  | S.A1, S.A => S.A1
  | S.A1, S.A1 => S.A2
  | S.A1, S.A2 => S.A3
  | S.A1, S.A3 => S.A4
  | S.A1, S.A4 => S.A
  | S.A2, S.A => S.A2
  | S.A2, S.A1 => S.A3
  | S.A2, S.A2 => S.A4
  | S.A2, S.A3 => S.A
  | S.A2, S.A4 => S.A1
  | S.A3, S.A => S.A3
  | S.A3, S.A1 => S.A4
  | S.A3, S.A2 => S.A
  | S.A3, S.A3 => S.A1
  | S.A3, S.A4 => S.A2
  | S.A4, S.A => S.A4
  | S.A4, S.A1 => S.A
  | S.A4, S.A2 => S.A1
  | S.A4, S.A3 => S.A2
  | S.A4, S.A4 => S.A3

theorem unique_solution :
  ∃! x : S, (oplus (oplus x x) S.A2) = S.A1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2160_216034


namespace NUMINAMATH_CALUDE_locker_count_l2160_216094

/-- Calculates the cost of labeling lockers given the number of lockers -/
def labelingCost (n : ℕ) : ℚ :=
  let cost1 := (min n 9 : ℚ) * 2 / 100
  let cost2 := (min (max (n - 9) 0) 90 : ℚ) * 4 / 100
  let cost3 := (min (max (n - 99) 0) 900 : ℚ) * 6 / 100
  let cost4 := (max (n - 999) 0 : ℚ) * 8 / 100
  cost1 + cost2 + cost3 + cost4

theorem locker_count : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    labelingCost n = 13794 / 100 ∧ 
    n = 2001 :=
sorry

end NUMINAMATH_CALUDE_locker_count_l2160_216094


namespace NUMINAMATH_CALUDE_least_repeating_block_seven_thirteenths_l2160_216014

/-- The least number of digits in a repeating block of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7/13 is a repeating decimal -/
axiom seven_thirteenths_repeats : ∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n + (↑k : ℚ) / (10^repeating_block_length - 1)

theorem least_repeating_block_seven_thirteenths :
  ∀ m : ℕ, m < repeating_block_length → ¬∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n + (↑k : ℚ) / (10^m - 1) :=
sorry

end NUMINAMATH_CALUDE_least_repeating_block_seven_thirteenths_l2160_216014


namespace NUMINAMATH_CALUDE_possible_distances_l2160_216006

/-- Represents the position of a house on a street. -/
structure House where
  position : ℝ

/-- Represents a street with four houses. -/
structure Street where
  andrey : House
  boris : House
  vova : House
  gleb : House

/-- The distance between two houses. -/
def distance (h1 h2 : House) : ℝ :=
  |h1.position - h2.position|

/-- A street is valid if it satisfies the given conditions. -/
def validStreet (s : Street) : Prop :=
  distance s.andrey s.boris = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrey s.gleb = 3 * distance s.boris s.vova

/-- The theorem stating the possible distances between Andrey's and Gleb's houses. -/
theorem possible_distances (s : Street) (h : validStreet s) :
  distance s.andrey s.gleb = 900 ∨ distance s.andrey s.gleb = 1800 :=
sorry

end NUMINAMATH_CALUDE_possible_distances_l2160_216006


namespace NUMINAMATH_CALUDE_product_increase_by_2016_l2160_216069

theorem product_increase_by_2016 : ∃ (a b c : ℕ), 
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 :=
sorry

end NUMINAMATH_CALUDE_product_increase_by_2016_l2160_216069


namespace NUMINAMATH_CALUDE_seating_theorem_l2160_216080

/-- The number of seating arrangements for 3 people in 6 seats with exactly two adjacent empty seats -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  2 * Nat.factorial (people + 1)

theorem seating_theorem :
  seating_arrangements 6 3 2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_seating_theorem_l2160_216080


namespace NUMINAMATH_CALUDE_min_value_theorem_l2160_216057

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m + 1| - 2

-- State the theorem
theorem min_value_theorem (m : ℝ) (a b : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →  -- f is even
  (a > 0 ∧ b > 0) →              -- a and b are positive
  f m a + f m (2 * b) = m →      -- condition given in the problem
  (∀ x y : ℝ, x > 0 → y > 0 → 1 / x + 2 / y ≥ 1 / a + 2 / b) →  -- minimum condition
  1 / a + 2 / b = 9 / 5 :=        -- conclusion
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2160_216057


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l2160_216092

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l2160_216092


namespace NUMINAMATH_CALUDE_prize_distribution_methods_l2160_216021

-- Define the number of prizes
def num_prizes : ℕ := 6

-- Define the number of people
def num_people : ℕ := 5

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define a function to calculate permutations
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem prize_distribution_methods :
  (combination num_prizes 2) * (permutation num_people num_people) =
  (number_of_distribution_methods : ℕ) :=
sorry

end NUMINAMATH_CALUDE_prize_distribution_methods_l2160_216021


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2160_216099

theorem distance_from_origin_to_point : Real.sqrt (12^2 + (-16)^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2160_216099


namespace NUMINAMATH_CALUDE_rounding_inequality_l2160_216074

/-- The number of digits in a natural number -/
def num_digits (k : ℕ) : ℕ := sorry

/-- Rounds a natural number to the nearest power of 10 -/
def round_to_power_of_10 (k : ℕ) (power : ℕ) : ℕ := sorry

/-- Applies n-1 roundings to the nearest power of 10 -/
def apply_n_minus_1_roundings (k : ℕ) : ℕ := sorry

theorem rounding_inequality (k : ℕ) (h1 : k = 10 * 106) :
  let n := num_digits k
  let k_bar := apply_n_minus_1_roundings k
  k_bar < (18 : ℚ) / 13 * k := by sorry

end NUMINAMATH_CALUDE_rounding_inequality_l2160_216074


namespace NUMINAMATH_CALUDE_midpoint_coordinate_ratio_range_l2160_216066

/-- Given two parallel lines and a point between them, prove the ratio of its coordinates is within a specific range. -/
theorem midpoint_coordinate_ratio_range 
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (x₀ : ℝ) (y₀ : ℝ)
  (hP : P.1 + 2 * P.2 - 1 = 0)
  (hQ : Q.1 + 2 * Q.2 + 3 = 0)
  (hM : (x₀, y₀) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))
  (h_ineq : y₀ > x₀ + 2)
  : -1/2 < y₀ / x₀ ∧ y₀ / x₀ < -1/5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_ratio_range_l2160_216066


namespace NUMINAMATH_CALUDE_power_function_domain_and_odd_l2160_216008

def A : Set ℝ := {-1, 1, 1/2, 3}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem power_function_domain_and_odd (a : ℝ) :
  a ∈ A →
  (Set.univ = {x : ℝ | ∃ y, y = x^a} ∧ is_odd_function (λ x => x^a)) ↔
  (a = 1 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_power_function_domain_and_odd_l2160_216008


namespace NUMINAMATH_CALUDE_smallest_d_value_l2160_216056

/-- The smallest positive value of d that satisfies the equation (4√3)² + (d-2)² = (4d)² -/
theorem smallest_d_value : ∃ d : ℝ, d > 0 ∧ 
  (4 * Real.sqrt 3)^2 + (d - 2)^2 = (4 * d)^2 ∧ 
  ∀ d' : ℝ, d' > 0 → (4 * Real.sqrt 3)^2 + (d' - 2)^2 = (4 * d')^2 → d ≤ d' := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_value_l2160_216056


namespace NUMINAMATH_CALUDE_perimeter_is_18_l2160_216000

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the arrangement of rectangles -/
structure Arrangement where
  base : Rectangle
  middle : Rectangle
  top : Rectangle

/-- Calculates the perimeter of the arrangement -/
def perimeter (arr : Arrangement) : ℕ :=
  2 * (arr.base.width + arr.base.height + arr.middle.height + arr.top.height)

/-- The theorem stating that the perimeter of the specific arrangement is 18 -/
theorem perimeter_is_18 : 
  let r := Rectangle.mk 2 1
  let arr := Arrangement.mk (Rectangle.mk 4 2) (Rectangle.mk 4 2) r
  perimeter arr = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_18_l2160_216000


namespace NUMINAMATH_CALUDE_intersection_condition_l2160_216010

/-- The set A on a 2D plane -/
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- The set B on a 2D plane, parameterized by r -/
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

/-- Theorem stating the conditions for r when A and B intersect at exactly one point -/
theorem intersection_condition (r : ℝ) (h1 : r > 0) 
  (h2 : ∃! p, p ∈ A ∩ B r) : r = 3 ∨ r = 7 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l2160_216010


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2160_216060

theorem complex_fraction_calculation : 
  (1 / (7 / 9) - (3 / 5) / 7) * (11 / (6 + 3 / 5)) / (4 / 13) - 2.4 = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2160_216060


namespace NUMINAMATH_CALUDE_sum_first_6_primes_l2160_216011

-- Define a function that returns the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function that sums the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ :=
  (List.range n).map (fun i => nthPrime (i + 1)) |>.sum

-- Theorem stating that the sum of the first 6 prime numbers is 41
theorem sum_first_6_primes : sumFirstNPrimes 6 = 41 := by sorry

end NUMINAMATH_CALUDE_sum_first_6_primes_l2160_216011


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_l2160_216026

theorem unique_number_with_remainders : ∃! n : ℕ,
  35 < n ∧ n < 70 ∧
  n % 6 = 3 ∧
  n % 7 = 1 ∧
  n % 8 = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_l2160_216026


namespace NUMINAMATH_CALUDE_sum_of_roots_l2160_216041

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y z : ℝ, x^2 - x - 34 = (x - y) * (x - z) ∧ y + z = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2160_216041


namespace NUMINAMATH_CALUDE_kati_age_l2160_216062

/-- Represents a person's age and birthday information -/
structure Person where
  age : ℕ
  birthdays : ℕ

/-- Represents the family members -/
structure Family where
  kati : Person
  brother : Person
  grandfather : Person

/-- The conditions of the problem -/
def problem_conditions (f : Family) : Prop :=
  f.kati.age = f.grandfather.birthdays ∧
  f.kati.age + f.brother.age + f.grandfather.age = 111 ∧
  f.kati.age > f.brother.age ∧
  f.kati.age - f.brother.age < 4 ∧
  f.grandfather.age = 4 * f.grandfather.birthdays + (f.grandfather.age % 4)

/-- The theorem to prove -/
theorem kati_age (f : Family) : 
  problem_conditions f → f.kati.age = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_kati_age_l2160_216062


namespace NUMINAMATH_CALUDE_range_of_sum_l2160_216028

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 3 then |Real.log x / Real.log 3|
  else if x ≥ 3 then 1/3 * x^2 - 10/3 * x + 8
  else 0  -- Define for all reals, though we only care about positive x

theorem range_of_sum (a b c d : ℝ) :
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧
  f a = f b ∧ f b = f c ∧ f c = f d →
  ∃ (x : ℝ), x ∈ Set.Icc (10 + 2 * Real.sqrt 2) (41/3) ∧
             x = 2*a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_l2160_216028


namespace NUMINAMATH_CALUDE_card_purchase_cost_l2160_216091

/-- Calculates the total cost of cards purchased, including discounts and tax --/
def totalCost (typeA_price typeB_price typeC_price typeD_price : ℚ)
               (typeA_count typeB_count typeC_count typeD_count : ℕ)
               (discount_AB discount_CD : ℚ)
               (min_count_AB min_count_CD : ℕ)
               (tax_rate : ℚ) : ℚ :=
  let subtotal := typeA_price * typeA_count + typeB_price * typeB_count +
                  typeC_price * typeC_count + typeD_price * typeD_count
  let discount_amount := 
    (if typeA_count ≥ min_count_AB ∧ typeB_count ≥ min_count_AB then
      discount_AB * (typeA_price * typeA_count + typeB_price * typeB_count)
    else 0) +
    (if typeC_count ≥ min_count_CD ∧ typeD_count ≥ min_count_CD then
      discount_CD * (typeC_price * typeC_count + typeD_price * typeD_count)
    else 0)
  let discounted_total := subtotal - discount_amount
  let tax := tax_rate * discounted_total
  discounted_total + tax

/-- The total cost of cards is $60.82 given the specified conditions --/
theorem card_purchase_cost : 
  totalCost 1.25 1.50 2.25 2.50  -- Card prices
            6 4 10 12            -- Number of cards purchased
            0.1 0.15             -- Discount rates
            5 8                  -- Minimum count for discounts
            0.06                 -- Tax rate
  = 60.82 := by
  sorry

end NUMINAMATH_CALUDE_card_purchase_cost_l2160_216091


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2160_216089

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2160_216089


namespace NUMINAMATH_CALUDE_f_fixed_points_l2160_216009

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem f_fixed_points : 
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 3 ∨ x = -1 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_fixed_points_l2160_216009


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l2160_216035

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def convex_quadrilaterals : ℕ := n.choose k

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quadrilaterals / total_selections

theorem convex_quadrilateral_probability :
  probability = 2 / 585 := by sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l2160_216035


namespace NUMINAMATH_CALUDE_average_marks_proof_l2160_216086

/-- Given the marks for three subjects, proves that the average is 65 -/
theorem average_marks_proof (physics chemistry maths : ℝ) : 
  physics = 125 → 
  (physics + maths) / 2 = 90 → 
  (physics + chemistry) / 2 = 70 → 
  (physics + chemistry + maths) / 3 = 65 := by
sorry


end NUMINAMATH_CALUDE_average_marks_proof_l2160_216086


namespace NUMINAMATH_CALUDE_number_division_problem_l2160_216037

theorem number_division_problem (n : ℕ) : 
  n % 23 = 19 → n / 23 = 17 → (10 * n) / 23 + (10 * n) % 23 = 184 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2160_216037


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l2160_216042

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_9_15_under_100 :
  ∃ (k : ℕ), k < 100 ∧ is_common_multiple 9 15 k ∧
  ∀ (m : ℕ), m < 100 → is_common_multiple 9 15 m → m ≤ k :=
by
  use 90
  sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l2160_216042


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l2160_216058

theorem distribute_and_simplify (a : ℝ) :
  (-12 * a) * (2 * a^2 - 2/3 * a + 5/6) = -24 * a^3 + 8 * a^2 - 10 * a := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l2160_216058


namespace NUMINAMATH_CALUDE_target_not_reachable_l2160_216004

/-- Represents a point in 3D space -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- The set of initial vertices of the cube -/
def initialVertices : Set Point3D :=
  { ⟨0,0,0⟩, ⟨0,0,1⟩, ⟨0,1,0⟩, ⟨1,0,0⟩, ⟨0,1,1⟩, ⟨1,0,1⟩, ⟨1,1,0⟩ }

/-- The target vertex we want to reach -/
def targetVertex : Point3D := ⟨1,1,1⟩

/-- Performs a symmetry operation on a point relative to another point -/
def symmetryOperation (p : Point3D) (center : Point3D) : Point3D :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y, 2 * center.z - p.z⟩

/-- Defines the set of points reachable through symmetry operations -/
def reachablePoints : Set Point3D :=
  sorry  -- Definition of reachable points would go here

/-- Theorem: The target vertex is not reachable from the initial vertices -/
theorem target_not_reachable : targetVertex ∉ reachablePoints := by
  sorry


end NUMINAMATH_CALUDE_target_not_reachable_l2160_216004


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2160_216064

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, (x - 2*a)^2 + (y - a - 3)^2 = 4 ∧ x^2 + y^2 = 1) ↔ -6/5 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2160_216064
