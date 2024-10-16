import Mathlib

namespace NUMINAMATH_CALUDE_tan_alpha_value_l740_74023

theorem tan_alpha_value (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l740_74023


namespace NUMINAMATH_CALUDE_alligator_walking_time_l740_74074

/-- The combined walking time of alligators given Paul's initial journey time and additional return time -/
theorem alligator_walking_time (initial_time return_additional_time : ℕ) :
  initial_time = 4 ∧ return_additional_time = 2 →
  initial_time + (initial_time + return_additional_time) = 10 := by
  sorry

#check alligator_walking_time

end NUMINAMATH_CALUDE_alligator_walking_time_l740_74074


namespace NUMINAMATH_CALUDE_max_salary_theorem_l740_74094

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  maxTotalSalary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def maxSinglePlayerSalary (team : BaseballTeam) : ℕ :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player in the given conditions -/
theorem max_salary_theorem (team : BaseballTeam) 
  (h1 : team.players = 18)
  (h2 : team.minSalary = 20000)
  (h3 : team.maxTotalSalary = 800000) :
  maxSinglePlayerSalary team = 460000 := by
  sorry

#eval maxSinglePlayerSalary ⟨18, 20000, 800000⟩

end NUMINAMATH_CALUDE_max_salary_theorem_l740_74094


namespace NUMINAMATH_CALUDE_birds_in_tree_l740_74075

/-- The total number of birds in a tree after two groups join -/
def total_birds (initial : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  initial + group1 + group2

/-- Theorem stating that the total number of birds is 76 -/
theorem birds_in_tree : total_birds 24 37 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l740_74075


namespace NUMINAMATH_CALUDE_proposition_equivalence_l740_74082

theorem proposition_equivalence (A : Set α) (x y : α) :
  (x ∈ A → y ∉ A) ↔ (y ∈ A → x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l740_74082


namespace NUMINAMATH_CALUDE_tan_family_ticket_cost_l740_74000

-- Define the group composition
def num_children : ℕ := 2
def num_adults : ℕ := 2
def num_seniors : ℕ := 2

-- Define the ticket prices and discounts
def adult_price : ℚ := 10
def senior_discount : ℚ := 0.3
def child_discount : ℚ := 0.2
def group_discount : ℚ := 0.1

-- Define the group size threshold for additional discount
def group_discount_threshold : ℕ := 5

-- Calculate the total group size
def total_group_size : ℕ := num_children + num_adults + num_seniors

-- Define the theorem
theorem tan_family_ticket_cost :
  let senior_price := adult_price * (1 - senior_discount)
  let child_price := adult_price * (1 - child_discount)
  let total_before_group_discount := 
    num_seniors * senior_price + num_adults * adult_price + num_children * child_price
  let final_total := 
    if total_group_size > group_discount_threshold
    then total_before_group_discount * (1 - group_discount)
    else total_before_group_discount
  final_total = 45 := by
  sorry

end NUMINAMATH_CALUDE_tan_family_ticket_cost_l740_74000


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l740_74031

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThroughPoint (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line), passesThroughPoint l ⟨-3, -2⟩ ∧ hasEqualIntercepts l ∧
  ((l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = 5)) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l740_74031


namespace NUMINAMATH_CALUDE_callie_summer_frogs_count_l740_74090

def alster_frogs : ℚ := 2

def quinn_frogs (alster_frogs : ℚ) : ℚ := 2 * alster_frogs

def bret_frogs (quinn_frogs : ℚ) : ℚ := 3 * quinn_frogs

def callie_summer_frogs (bret_frogs : ℚ) : ℚ := (5/8) * bret_frogs

theorem callie_summer_frogs_count :
  callie_summer_frogs (bret_frogs (quinn_frogs alster_frogs)) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_callie_summer_frogs_count_l740_74090


namespace NUMINAMATH_CALUDE_parabola_point_ordinate_l740_74044

/-- The y-coordinate of a point on the parabola y = 4x^2 that is at a distance of 1 from the focus -/
theorem parabola_point_ordinate : ∀ (x y : ℝ),
  y = 4 * x^2 →  -- Point is on the parabola
  (x - 0)^2 + (y - 1/16)^2 = 1 →  -- Distance from focus is 1
  y = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordinate_l740_74044


namespace NUMINAMATH_CALUDE_octagon_area_error_l740_74079

theorem octagon_area_error (L : ℝ) (h : L > 0) : 
  let measured_length := 1.1 * L
  let true_area := 2 * (1 + Real.sqrt 2) * L^2 / 4
  let estimated_area := 2 * (1 + Real.sqrt 2) * measured_length^2 / 4
  (estimated_area - true_area) / true_area * 100 = 21 := by sorry

end NUMINAMATH_CALUDE_octagon_area_error_l740_74079


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l740_74059

/-- Given a quadratic function f(x) = ax^2 - c, 
    if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20 -/
theorem quadratic_function_bounds (a c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - c
  (-4 : ℝ) ≤ f 1 ∧ f 1 ≤ -1 ∧ -1 ≤ f 2 ∧ f 2 ≤ 5 → 
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l740_74059


namespace NUMINAMATH_CALUDE_meeting_distance_l740_74035

theorem meeting_distance (initial_speed : ℝ) (speed_increase : ℝ) (initial_distance : ℝ) 
  (late_time : ℝ) (early_time : ℝ) :
  initial_speed = 45 ∧ 
  speed_increase = 20 ∧ 
  initial_distance = 45 ∧ 
  late_time = 0.75 ∧ 
  early_time = 0.25 → 
  ∃ (total_distance : ℝ),
    total_distance = initial_speed * (total_distance / initial_speed + late_time) ∧
    total_distance - initial_distance = (initial_speed + speed_increase) * 
      (total_distance / initial_speed - 1 - early_time) ∧
    total_distance = 191.25 := by
  sorry

end NUMINAMATH_CALUDE_meeting_distance_l740_74035


namespace NUMINAMATH_CALUDE_transmission_time_is_128_seconds_l740_74039

/-- The number of blocks to be sent -/
def num_blocks : ℕ := 80

/-- The number of chunks in each block -/
def chunks_per_block : ℕ := 256

/-- The transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- The time it takes to send all blocks in seconds -/
def transmission_time : ℕ := num_blocks * chunks_per_block / transmission_rate

theorem transmission_time_is_128_seconds : transmission_time = 128 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_128_seconds_l740_74039


namespace NUMINAMATH_CALUDE_max_profit_selling_price_daily_profit_unachievable_monthly_profit_prices_l740_74004

/-- Represents the profit function for desk lamp sales -/
def profit_function (x : ℝ) : ℝ :=
  (x - 30) * (600 - 10 * (x - 40))

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem max_profit_selling_price :
  ∃ (max_price max_profit : ℝ),
    max_price = 65 ∧
    max_profit = 12250 ∧
    ∀ (x : ℝ), profit_function x ≤ max_profit :=
by
  sorry

/-- Theorem stating that 15,000 yuan daily profit is not achievable -/
theorem daily_profit_unachievable :
  ∀ (x : ℝ), profit_function x < 15000 :=
by
  sorry

/-- Theorem stating the selling prices for 10,000 yuan monthly profit -/
theorem monthly_profit_prices :
  ∃ (price1 price2 : ℝ),
    price1 = 80 ∧
    price2 = 50 ∧
    profit_function price1 = 10000 ∧
    profit_function price2 = 10000 ∧
    ∀ (x : ℝ), profit_function x = 10000 → (x = price1 ∨ x = price2) :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_selling_price_daily_profit_unachievable_monthly_profit_prices_l740_74004


namespace NUMINAMATH_CALUDE_equation_solution_l740_74069

theorem equation_solution : 
  ∃ x : ℝ, (2 * x + 16 ≥ 0) ∧ 
  ((Real.sqrt (2 * x + 16) - 8 / Real.sqrt (2 * x + 16)) = 4) ∧ 
  (x = 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l740_74069


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l740_74021

/-- Given that x and y are positive real numbers, 3x² and y vary inversely,
    y = 18 when x = 3, and y = 2400, prove that x = 9√6 / 85. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h_inverse : ∃ k, k > 0 ∧ ∀ x' y', x' > 0 → y' > 0 → 3 * x'^2 * y' = k)
    (h_initial : 3 * 3^2 * 18 = 3 * x^2 * 2400) :
    x = 9 * Real.sqrt 6 / 85 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l740_74021


namespace NUMINAMATH_CALUDE_acute_triangle_sine_sum_l740_74063

theorem acute_triangle_sine_sum (α β γ : Real) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : α + β + γ = Real.pi)
  (h_acute_triangle : α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_sum_l740_74063


namespace NUMINAMATH_CALUDE_intersection_triangle_area_l740_74019

-- Define the line L: x - 2y - 5 = 0
def L (x y : ℝ) : Prop := x - 2*y - 5 = 0

-- Define the circle C: x^2 + y^2 = 50
def C (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the intersection points
def A : ℝ × ℝ := (-5, -5)
def B : ℝ × ℝ := (7, 1)

-- Theorem statement
theorem intersection_triangle_area :
  L A.1 A.2 ∧ L B.1 B.2 ∧ C A.1 A.2 ∧ C B.1 B.2 →
  abs ((A.1 * B.2 - B.1 * A.2) / 2) = 15 :=
by sorry

end NUMINAMATH_CALUDE_intersection_triangle_area_l740_74019


namespace NUMINAMATH_CALUDE_intersection_M_N_l740_74022

def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | Real.log x / Real.log (1/2) > -1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l740_74022


namespace NUMINAMATH_CALUDE_nancys_weight_l740_74041

theorem nancys_weight (water_intake : ℝ) (water_percentage : ℝ) :
  water_intake = 54 →
  water_percentage = 0.60 →
  water_intake = water_percentage * 90 :=
by
  sorry

end NUMINAMATH_CALUDE_nancys_weight_l740_74041


namespace NUMINAMATH_CALUDE_pool_earnings_theorem_l740_74087

def calculate_weekly_earnings (kid_fee : ℚ) (adult_fee : ℚ) (weekend_surcharge : ℚ) 
  (weekday_kids : ℕ) (weekday_adults : ℕ) (weekend_kids : ℕ) (weekend_adults : ℕ) 
  (weekdays : ℕ) (weekend_days : ℕ) : ℚ :=
  let weekday_earnings := (kid_fee * weekday_kids + adult_fee * weekday_adults) * weekdays
  let weekend_kid_fee := kid_fee * (1 + weekend_surcharge)
  let weekend_adult_fee := adult_fee * (1 + weekend_surcharge)
  let weekend_earnings := (weekend_kid_fee * weekend_kids + weekend_adult_fee * weekend_adults) * weekend_days
  weekday_earnings + weekend_earnings

theorem pool_earnings_theorem : 
  calculate_weekly_earnings 3 6 (1/2) 8 10 12 15 5 2 = 798 := by
  sorry

end NUMINAMATH_CALUDE_pool_earnings_theorem_l740_74087


namespace NUMINAMATH_CALUDE_algorithm_steps_are_determinate_l740_74060

/-- Represents a step in an algorithm -/
structure AlgorithmStep where
  precise : Bool
  effective : Bool
  determinate : Bool

/-- Represents an algorithm -/
structure Algorithm where
  steps : List AlgorithmStep
  solvesProblem : Bool
  finite : Bool

/-- Theorem: Given an algorithm with finite, precise, and effective steps that solve a problem, 
    prove that all steps in the algorithm are determinate -/
theorem algorithm_steps_are_determinate (a : Algorithm) 
  (h1 : a.solvesProblem)
  (h2 : a.finite)
  (h3 : ∀ s ∈ a.steps, s.precise)
  (h4 : ∀ s ∈ a.steps, s.effective) :
  ∀ s ∈ a.steps, s.determinate := by
  sorry


end NUMINAMATH_CALUDE_algorithm_steps_are_determinate_l740_74060


namespace NUMINAMATH_CALUDE_stating_meal_distribution_count_l740_74058

/-- Represents the number of people having dinner -/
def n : ℕ := 12

/-- Represents the number of meal types -/
def meal_types : ℕ := 4

/-- Represents the number of people who ordered each meal type -/
def people_per_meal : ℕ := 3

/-- Represents the number of people who should receive their ordered meal type -/
def correct_meals : ℕ := 2

/-- 
Theorem stating that the number of ways to distribute meals 
such that exactly two people receive their ordered meal type is 88047666
-/
theorem meal_distribution_count : 
  (Nat.choose n correct_meals) * (Nat.factorial (n - correct_meals)) = 88047666 := by
  sorry

end NUMINAMATH_CALUDE_stating_meal_distribution_count_l740_74058


namespace NUMINAMATH_CALUDE_board_numbers_product_l740_74084

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {2, 6, 10, 10, 12, 14, 16, 18, 20, 24} → 
  a * b * c * d * e = -3003 := by
sorry

end NUMINAMATH_CALUDE_board_numbers_product_l740_74084


namespace NUMINAMATH_CALUDE_remainder_of_1531_base12_div_8_l740_74020

/-- Represents a base-12 number as a list of digits (least significant first) -/
def Base12 := List Nat

/-- Converts a base-12 number to base-10 -/
def toBase10 (n : Base12) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 number 1531 -/
def num : Base12 := [1, 3, 5, 1]

theorem remainder_of_1531_base12_div_8 :
  toBase10 num % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1531_base12_div_8_l740_74020


namespace NUMINAMATH_CALUDE_f_iterative_application_l740_74032

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 3*x + 2 else x + 10

theorem f_iterative_application : f (f (f (f (f 2)))) = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_iterative_application_l740_74032


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l740_74051

/-- Given positive real numbers a, b, c satisfying the condition,
    the minimum value of the expression is 50 -/
theorem min_value_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  ∃ m : ℝ, m = 50 ∧ ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    x/y + y/z + z/x + y/x + z/y + x/z = 10 →
    (x/y + y/z + z/x)^2 + (y/x + z/y + x/z)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l740_74051


namespace NUMINAMATH_CALUDE_largest_angle_is_112_5_l740_74089

/-- Represents a quadrilateral formed by folding two sides of a square along its diagonal -/
structure FoldedSquare where
  /-- The side length of the original square -/
  side : ℝ
  /-- Assumption that the side length is positive -/
  side_pos : side > 0

/-- The largest angle in the folded square -/
def largest_angle (fs : FoldedSquare) : ℝ := 112.5

/-- Theorem stating that the largest angle in the folded square is 112.5° -/
theorem largest_angle_is_112_5 (fs : FoldedSquare) :
  largest_angle fs = 112.5 := by sorry

end NUMINAMATH_CALUDE_largest_angle_is_112_5_l740_74089


namespace NUMINAMATH_CALUDE_reflect_F_coordinates_l740_74061

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The original point F -/
def F : ℝ × ℝ := (-1, -1)

theorem reflect_F_coordinates :
  (reflect_y_eq_x (reflect_x F)) = (1, -1) := by
sorry

end NUMINAMATH_CALUDE_reflect_F_coordinates_l740_74061


namespace NUMINAMATH_CALUDE_even_function_monotone_interval_l740_74064

theorem even_function_monotone_interval
  (ω φ : ℝ) (x₁ x₂ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (h_even : ∀ x, 2 * Real.sin (ω * x + φ) = 2 * Real.sin (ω * (-x) + φ))
  (h_intersect : 2 * Real.sin (ω * x₁ + φ) = 2 ∧ 2 * Real.sin (ω * x₂ + φ) = 2)
  (h_min_distance : ∀ x y, 2 * Real.sin (ω * x + φ) = 2 → 2 * Real.sin (ω * y + φ) = 2 → |x - y| ≥ π)
  (h_exists_min : ∃ x y, 2 * Real.sin (ω * x + φ) = 2 ∧ 2 * Real.sin (ω * y + φ) = 2 ∧ |x - y| = π) :
  ∃ a b, a = -π/2 ∧ b = -π/4 ∧
    ∀ x y, a < x ∧ x < y ∧ y < b →
      2 * Real.sin (ω * x + φ) < 2 * Real.sin (ω * y + φ) :=
by sorry

end NUMINAMATH_CALUDE_even_function_monotone_interval_l740_74064


namespace NUMINAMATH_CALUDE_inequality_preservation_l740_74091

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l740_74091


namespace NUMINAMATH_CALUDE_krishans_money_l740_74088

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan's amount is 3774. -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 637 →
  krishan = 3774 := by
sorry

end NUMINAMATH_CALUDE_krishans_money_l740_74088


namespace NUMINAMATH_CALUDE_sum_of_parts_of_complex_number_l740_74016

theorem sum_of_parts_of_complex_number : ∃ (z : ℂ), 
  z = (Complex.I * 2 - 3) * (Complex.I - 2) / Complex.I ∧ 
  z.re + z.im = -11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_parts_of_complex_number_l740_74016


namespace NUMINAMATH_CALUDE_paul_homework_hours_l740_74042

/-- Calculates the total hours of homework on weeknights for Paul --/
def weeknight_homework (total_weeknights : ℕ) (practice_nights : ℕ) (average_hours : ℕ) : ℕ :=
  (total_weeknights - practice_nights) * average_hours

/-- Proves that Paul has 9 hours of homework on weeknights --/
theorem paul_homework_hours :
  let total_weeknights := 5
  let practice_nights := 2
  let average_hours := 3
  weeknight_homework total_weeknights practice_nights average_hours = 9 := by
  sorry

#eval weeknight_homework 5 2 3

end NUMINAMATH_CALUDE_paul_homework_hours_l740_74042


namespace NUMINAMATH_CALUDE_closest_fraction_l740_74037

def medals_won : ℚ := 17 / 100

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧
    ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
    f = 1/6 :=
  sorry

end NUMINAMATH_CALUDE_closest_fraction_l740_74037


namespace NUMINAMATH_CALUDE_cube_equation_solution_l740_74048

theorem cube_equation_solution :
  ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l740_74048


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l740_74012

theorem fraction_equals_zero (x : ℝ) : 
  (|x| - 2) / (x - 2) = 0 ∧ x - 2 ≠ 0 ↔ x = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l740_74012


namespace NUMINAMATH_CALUDE_original_ratio_proof_l740_74054

theorem original_ratio_proof (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 220 →
  new_boarders = 44 →
  (initial_boarders + new_boarders) * 2 = (initial_boarders + new_boarders + (initial_boarders + new_boarders) * 2) →
  (5 : ℚ) / 12 = initial_boarders / ((initial_boarders + new_boarders) * 2 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_original_ratio_proof_l740_74054


namespace NUMINAMATH_CALUDE_max_fraction_sum_l740_74005

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- The fraction (A+B)/(C+D) is an integer -/
def is_integer_fraction (a b c d : Digit) : Prop :=
  ∃ k : ℕ, k * (c.val + d.val) = a.val + b.val

/-- The fraction (A+B)/(C+D) is maximized -/
def is_maximized (a b c d : Digit) : Prop :=
  ∀ w x y z : Digit, distinct w x y z →
    is_integer_fraction w x y z →
    (a.val + b.val : ℚ) / (c.val + d.val) ≥ (w.val + x.val : ℚ) / (y.val + z.val)

theorem max_fraction_sum (a b c d : Digit) :
  distinct a b c d →
  is_integer_fraction a b c d →
  is_maximized a b c d →
  a.val + b.val = 17 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l740_74005


namespace NUMINAMATH_CALUDE_unique_prime_pair_sum_53_l740_74018

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating that there is exactly one pair of primes summing to 53 -/
theorem unique_prime_pair_sum_53 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_sum_53_l740_74018


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l740_74011

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2003 + a 2005 + a 2007 + a 2009 + a 2011 + a 2013 = 120) :
  2 * a 2018 - a 2028 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l740_74011


namespace NUMINAMATH_CALUDE_basketball_shooting_averages_l740_74007

/-- Represents the average number of successful shots -/
structure ShootingAverage where
  male : ℝ
  female : ℝ

/-- Represents the number of students -/
structure StudentCount where
  male : ℝ
  female : ℝ

/-- The theorem stating the average number of successful shots for male and female students -/
theorem basketball_shooting_averages 
  (avg : ShootingAverage) 
  (count : StudentCount) 
  (h1 : avg.male = 1.25 * avg.female) 
  (h2 : count.female = 1.25 * count.male) 
  (h3 : (avg.male * count.male + avg.female * count.female) / (count.male + count.female) = 4) :
  avg.male = 4.5 ∧ avg.female = 3.6 := by
  sorry

#check basketball_shooting_averages

end NUMINAMATH_CALUDE_basketball_shooting_averages_l740_74007


namespace NUMINAMATH_CALUDE_prove_total_workers_l740_74002

def total_workers : ℕ := 9
def other_workers : ℕ := 7
def chosen_workers : ℕ := 2

theorem prove_total_workers :
  (total_workers = other_workers + 2) →
  (Nat.choose total_workers chosen_workers = 36) →
  (1 / (Nat.choose total_workers chosen_workers : ℚ) = 1 / 36) →
  total_workers = 9 := by
sorry

end NUMINAMATH_CALUDE_prove_total_workers_l740_74002


namespace NUMINAMATH_CALUDE_isabel_homework_problem_l740_74030

theorem isabel_homework_problem (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problem_l740_74030


namespace NUMINAMATH_CALUDE_cousins_assignment_count_l740_74078

/-- The number of ways to assign n indistinguishable objects to k indistinguishable containers -/
def assign_indistinguishable (n k : ℕ) : ℕ :=
  sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to assign -/
def num_cousins : ℕ := 5

/-- The number of ways to assign the cousins to the rooms is 51 -/
theorem cousins_assignment_count : assign_indistinguishable num_cousins num_rooms = 51 := by
  sorry

end NUMINAMATH_CALUDE_cousins_assignment_count_l740_74078


namespace NUMINAMATH_CALUDE_converse_proposition_l740_74025

theorem converse_proposition (a : ℝ) : a > 2 → a^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_converse_proposition_l740_74025


namespace NUMINAMATH_CALUDE_machine_quality_comparison_l740_74095

/-- Data for machine production quality --/
structure MachineData where
  first_class : ℕ
  second_class : ℕ

/-- Calculate the frequency of first-class products --/
def frequency (data : MachineData) : ℚ :=
  data.first_class / (data.first_class + data.second_class)

/-- Calculate K² statistic --/
def k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the frequencies and significance of difference --/
theorem machine_quality_comparison 
  (machine_a machine_b : MachineData)
  (h_a : machine_a = ⟨150, 50⟩)
  (h_b : machine_b = ⟨120, 80⟩) :
  (frequency machine_a = 3/4) ∧ 
  (frequency machine_b = 3/5) ∧ 
  (k_squared machine_a.first_class machine_a.second_class 
              machine_b.first_class machine_b.second_class > 6635/1000) := by
  sorry

#eval frequency ⟨150, 50⟩
#eval frequency ⟨120, 80⟩
#eval k_squared 150 50 120 80

end NUMINAMATH_CALUDE_machine_quality_comparison_l740_74095


namespace NUMINAMATH_CALUDE_smallest_AC_solution_exists_l740_74086

-- Define the triangle and its properties
def Triangle (AC CD : ℕ) : Prop :=
  ∃ (AB BD : ℕ),
    AB = AC ∧  -- AB = AC
    BD * BD = 68 ∧  -- BD² = 68
    AC = (CD * CD + 68) / (2 * CD) ∧  -- Derived from the Pythagorean theorem
    CD < 10 ∧  -- CD is less than 10
    Nat.Prime CD  -- CD is prime

-- State the theorem
theorem smallest_AC :
  ∀ AC CD, Triangle AC CD → AC ≥ 18 :=
by sorry

-- State the existence of a solution
theorem solution_exists :
  ∃ AC CD, Triangle AC CD ∧ AC = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AC_solution_exists_l740_74086


namespace NUMINAMATH_CALUDE_fraction_inequality_l740_74034

theorem fraction_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (a + 1)) + (b^2 / (b + 1)) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l740_74034


namespace NUMINAMATH_CALUDE_solution_value_l740_74092

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^3 + c^2

/-- Theorem stating that 5/19 is the value of a that satisfies the equation -/
theorem solution_value : ∃ (a : ℝ), F a 3 2 = F a 2 3 ∧ a = 5/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l740_74092


namespace NUMINAMATH_CALUDE_rectangle_squares_sum_l740_74085

theorem rectangle_squares_sum (a b : ℝ) : 
  a + b = 3 → a * b = 1 → a^2 + b^2 = 7 := by sorry

end NUMINAMATH_CALUDE_rectangle_squares_sum_l740_74085


namespace NUMINAMATH_CALUDE_complement_equivalence_l740_74043

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event "at least one item is defective"
def at_least_one_defective : Set Ω :=
  {ω | ω.1 = true ∨ ω.2 = true}

-- Define the event "neither of the items is defective"
def neither_defective : Set Ω :=
  {ω | ω.1 = false ∧ ω.2 = false}

-- Theorem: The complement of "at least one item is defective" 
-- is equivalent to "neither of the items is defective"
theorem complement_equivalence :
  at_least_one_defective.compl = neither_defective :=
sorry

end NUMINAMATH_CALUDE_complement_equivalence_l740_74043


namespace NUMINAMATH_CALUDE_representatives_count_l740_74052

/-- The number of ways to select representatives from boys and girls -/
def select_representatives (num_boys num_girls num_representatives : ℕ) : ℕ :=
  Nat.choose num_boys 2 * Nat.choose num_girls 1 +
  Nat.choose num_boys 1 * Nat.choose num_girls 2

/-- Theorem stating that selecting 3 representatives from 5 boys and 3 girls,
    with both genders represented, can be done in 45 ways -/
theorem representatives_count :
  select_representatives 5 3 3 = 45 := by
  sorry

#eval select_representatives 5 3 3

end NUMINAMATH_CALUDE_representatives_count_l740_74052


namespace NUMINAMATH_CALUDE_u_n_satisfies_property_u_n_is_smallest_u_n_equals_2n_minus_1_l740_74045

/-- Given a positive integer n, u_n is the smallest positive integer such that
    for every positive integer d, the number of numbers divisible by d
    in any u_n consecutive positive odd numbers is no less than
    the number of numbers divisible by d in the set of odd numbers 1, 3, 5, ..., 2n-1 -/
def u_n (n : ℕ+) : ℕ :=
  2 * n.val - 1

/-- For any positive integer n, u_n satisfies the required property -/
theorem u_n_satisfies_property (n : ℕ+) :
  ∀ (d : ℕ+) (a : ℕ),
    (∀ k : Fin (2 * n.val - 1), ∃ m : ℕ, 2 * (a + k.val) - 1 = d * (2 * m + 1)) →
    (∃ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * (2 * m + 1)) :=
  sorry

/-- u_n is the smallest positive integer satisfying the required property -/
theorem u_n_is_smallest (n : ℕ+) :
  ∀ m : ℕ+, m.val < u_n n →
    ∃ (d : ℕ+) (a : ℕ),
      (∀ k : Fin m, ∃ l : ℕ, 2 * (a + k.val) - 1 = d * (2 * l + 1)) ∧
      ¬(∃ k : Fin n, ∃ l : ℕ, 2 * k.val + 1 = d * (2 * l + 1)) :=
  sorry

/-- The main theorem stating that u_n is equal to 2n - 1 -/
theorem u_n_equals_2n_minus_1 (n : ℕ+) :
  u_n n = 2 * n.val - 1 :=
  sorry

end NUMINAMATH_CALUDE_u_n_satisfies_property_u_n_is_smallest_u_n_equals_2n_minus_1_l740_74045


namespace NUMINAMATH_CALUDE_cafe_meal_combinations_l740_74001

theorem cafe_meal_combinations (n : ℕ) (h : n = 12) : 
  (n * (n - 1) : ℕ) = 132 := by
  sorry

end NUMINAMATH_CALUDE_cafe_meal_combinations_l740_74001


namespace NUMINAMATH_CALUDE_boa_constrictor_length_alberts_boa_length_l740_74017

/-- The length of Albert's boa constrictor given the length of his garden snake and their relative sizes. -/
theorem boa_constrictor_length (garden_snake_length : ℕ) (relative_size : ℕ) : ℕ :=
  garden_snake_length * relative_size

/-- Proof that Albert's boa constrictor is 70 inches long. -/
theorem alberts_boa_length : boa_constrictor_length 10 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_boa_constrictor_length_alberts_boa_length_l740_74017


namespace NUMINAMATH_CALUDE_f_composition_of_one_l740_74096

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then 3 * x / 2 else 2 * x + 1

theorem f_composition_of_one : f (f (f (f 1))) = 31 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_one_l740_74096


namespace NUMINAMATH_CALUDE_chloe_recycled_28_pounds_l740_74036

/-- Represents the recycling scenario with Chloe and her friends -/
structure RecyclingScenario where
  pounds_per_point : ℕ
  friends_recycled : ℕ
  total_points : ℕ

/-- Calculates the amount of paper Chloe recycled given the recycling scenario -/
def chloe_recycled (scenario : RecyclingScenario) : ℕ :=
  scenario.pounds_per_point * scenario.total_points - scenario.friends_recycled

/-- Theorem stating that Chloe recycled 28 pounds given the specific scenario -/
theorem chloe_recycled_28_pounds : 
  let scenario : RecyclingScenario := {
    pounds_per_point := 6,
    friends_recycled := 2,
    total_points := 5
  }
  chloe_recycled scenario = 28 := by
  sorry

end NUMINAMATH_CALUDE_chloe_recycled_28_pounds_l740_74036


namespace NUMINAMATH_CALUDE_ibrahim_purchase_l740_74083

/-- The amount of money Ibrahim lacks to purchase an MP3 player and a CD -/
def money_lacking (mp3_cost cd_cost savings father_contribution : ℕ) : ℕ :=
  (mp3_cost + cd_cost) - (savings + father_contribution)

/-- Theorem: Ibrahim lacks 64 euros -/
theorem ibrahim_purchase :
  money_lacking 120 19 55 20 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ibrahim_purchase_l740_74083


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l740_74076

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l740_74076


namespace NUMINAMATH_CALUDE_egg_packing_problem_l740_74014

/-- The number of baskets containing eggs -/
def num_baskets : ℕ := 21

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 48

/-- The number of eggs each box can hold -/
def eggs_per_box : ℕ := 28

/-- The number of boxes needed to pack all the eggs -/
def boxes_needed : ℕ := (num_baskets * eggs_per_basket) / eggs_per_box

theorem egg_packing_problem : boxes_needed = 36 := by
  sorry

end NUMINAMATH_CALUDE_egg_packing_problem_l740_74014


namespace NUMINAMATH_CALUDE_range_of_a_l740_74099

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) a, f x ∈ Set.Icc (-4 : ℝ) 5) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) a, f x = -4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) a, f x = 5) →
  a ∈ Set.Icc 2 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l740_74099


namespace NUMINAMATH_CALUDE_bob_over_budget_l740_74065

def budget : ℕ := 100
def necklaceA : ℕ := 34
def necklaceB : ℕ := 42
def necklaceC : ℕ := 50
def book1 : ℕ := necklaceA + 20
def book2 : ℕ := necklaceC - 10

theorem bob_over_budget : 
  necklaceA + necklaceB + necklaceC + book1 + book2 - budget = 120 := by
  sorry

end NUMINAMATH_CALUDE_bob_over_budget_l740_74065


namespace NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l740_74038

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the infinite series ∑(n=0 to ∞) F_n / 5^n -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- Theorem stating that the sum of the infinite series equals 5/19 -/
theorem fibSum_eq_five_nineteenths : fibSum = 5 / 19 := by sorry

end NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l740_74038


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l740_74055

theorem units_digit_of_sum_of_powers (a b : ℕ) (ha : a = 15) (hb : b = 220) :
  ∃ k : ℤ, (a + Real.sqrt b)^19 + (a - Real.sqrt b)^19 = 10 * k + 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l740_74055


namespace NUMINAMATH_CALUDE_magnitude_relationship_l740_74028

theorem magnitude_relationship : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l740_74028


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l740_74073

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (α : ℝ) (h_d : d = 12) (h_α : α = π/4) :
  let r := d / (2 * Real.tan (α/2))
  (4/3) * π * r^3 = 72 * Real.sqrt 2 * π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l740_74073


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l740_74053

def arithmeticSequence (a₁ a₂ a₃ : ℕ) : ℕ → ℕ :=
  fun n => a₁ + (n - 1) * (a₂ - a₁)

theorem fifteenth_term_of_sequence (h : arithmeticSequence 3 14 25 3 = 25) :
  arithmeticSequence 3 14 25 15 = 157 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l740_74053


namespace NUMINAMATH_CALUDE_garden_ratio_l740_74093

/-- Proves that a rectangular garden with area 432 square meters and width 12 meters has a length to width ratio of 3:1 -/
theorem garden_ratio :
  ∀ (length width : ℝ),
    width = 12 →
    length * width = 432 →
    length / width = 3 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l740_74093


namespace NUMINAMATH_CALUDE_equation_solution_l740_74066

theorem equation_solution : 
  ∃ x : ℝ, (3 : ℝ)^(x - 2) = 9^(x + 2) ∧ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l740_74066


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l740_74047

/-- 
Given a rectangular plot where the length is thrice the breadth and the breadth is 17 meters,
prove that the area of the plot is 867 square meters.
-/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 17 →
  length = 3 * breadth →
  area = length * breadth →
  area = 867 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l740_74047


namespace NUMINAMATH_CALUDE_infinite_product_value_l740_74068

/-- The nth term of the sequence in the infinite product -/
def a (n : ℕ) : ℝ := (2^n)^(1 / 3^n)

/-- The sum of the exponents in the infinite product -/
noncomputable def S : ℝ := ∑' n, n / 3^n

/-- The infinite product -/
noncomputable def infiniteProduct : ℝ := 2^S

theorem infinite_product_value :
  infiniteProduct = 2^(3/4) := by sorry

end NUMINAMATH_CALUDE_infinite_product_value_l740_74068


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l740_74040

/-- Calculates the marks in Chemistry given the marks in other subjects and the average --/
def calculate_chemistry_marks (english : ℕ) (mathematics : ℕ) (physics : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + physics + biology)

/-- Proves that given the specific marks and average, the Chemistry marks are 67 --/
theorem chemistry_marks_proof (english : ℕ) (mathematics : ℕ) (physics : ℕ) (biology : ℕ) (average : ℕ)
  (h1 : english = 61)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 72) :
  calculate_chemistry_marks english mathematics physics biology average = 67 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l740_74040


namespace NUMINAMATH_CALUDE_power_function_increasing_l740_74080

/-- A power function f(x) = (m^2 - m - 1)x^m is increasing on (0, +∞) if and only if m = 2 -/
theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, Monotone (fun x => (m^2 - m - 1) * x^m)) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_l740_74080


namespace NUMINAMATH_CALUDE_negative_product_probability_l740_74071

def S : Finset Int := {-3, -1, 2, 6, 5, -4}

theorem negative_product_probability :
  let total_pairs := Finset.card (S.powerset.filter (fun s => s.card = 2))
  let negative_product_pairs := (S.filter (· < 0)).card * (S.filter (· > 0)).card
  (negative_product_pairs : ℚ) / total_pairs = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_negative_product_probability_l740_74071


namespace NUMINAMATH_CALUDE_remainder_doubling_l740_74029

theorem remainder_doubling (N : ℤ) : 
  N % 367 = 241 → (2 * N) % 367 = 115 := by
sorry

end NUMINAMATH_CALUDE_remainder_doubling_l740_74029


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l740_74072

theorem point_on_hyperbola : 
  let f : ℝ → ℝ := λ x ↦ 6 / x
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l740_74072


namespace NUMINAMATH_CALUDE_coin_distribution_l740_74024

theorem coin_distribution (a b c d e : ℚ) : 
  a + b + c + d + e = 5 →  -- total is 5 coins
  ∃ (x y : ℚ), (a = x - 2*y ∧ b = x - y ∧ c = x ∧ d = x + y ∧ e = x + 2*y) →  -- arithmetic sequence
  a + b = c + d + e →  -- sum of first two equals sum of last three
  b = 4/3 :=  -- second person receives 4/3 coins
by sorry

end NUMINAMATH_CALUDE_coin_distribution_l740_74024


namespace NUMINAMATH_CALUDE_exists_unique_affine_transformation_basis_exists_unique_affine_transformation_triangle_exists_unique_affine_transformation_parallelogram_l740_74067

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define points and vectors
variable (O O' : V) (e1 e2 e1' e2' : V)
variable (A B C A1 B1 C1 : V)

-- Define affine transformation
def AffineTransformation (f : V → V) :=
  ∃ (T : V →L[ℝ] V) (b : V), ∀ x, f x = T x + b

-- Statement for part (a)
theorem exists_unique_affine_transformation_basis :
  ∃! f : V → V, AffineTransformation f ∧
  f O = O' ∧ f (O + e1) = O' + e1' ∧ f (O + e2) = O' + e2' :=
sorry

-- Statement for part (b)
theorem exists_unique_affine_transformation_triangle :
  ∃! f : V → V, AffineTransformation f ∧
  f A = A1 ∧ f B = B1 ∧ f C = C1 :=
sorry

-- Define parallelogram
def IsParallelogram (P Q R S : V) :=
  P - Q = S - R ∧ P - S = Q - R

-- Statement for part (c)
theorem exists_unique_affine_transformation_parallelogram
  (P Q R S P' Q' R' S' : V)
  (h1 : IsParallelogram P Q R S)
  (h2 : IsParallelogram P' Q' R' S') :
  ∃! f : V → V, AffineTransformation f ∧
  f P = P' ∧ f Q = Q' ∧ f R = R' ∧ f S = S' :=
sorry

end NUMINAMATH_CALUDE_exists_unique_affine_transformation_basis_exists_unique_affine_transformation_triangle_exists_unique_affine_transformation_parallelogram_l740_74067


namespace NUMINAMATH_CALUDE_sheridan_cats_l740_74049

/-- The number of cats Mrs. Sheridan has after giving some away -/
def remaining_cats (initial : ℝ) (given_away : ℝ) : ℝ :=
  initial - given_away

/-- Proof that Mrs. Sheridan has 3.0 cats after giving away 14.0 cats -/
theorem sheridan_cats : remaining_cats 17.0 14.0 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l740_74049


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l740_74010

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l740_74010


namespace NUMINAMATH_CALUDE_triangle_angles_from_radii_relations_l740_74008

/-- Given a triangle with excircle radii r_a, r_b, r_c, and circumcircle radius R,
    if r_a + r_b = 3R and r_b + r_c = 2R, then the angles of the triangle are 90°, 60°, and 30°. -/
theorem triangle_angles_from_radii_relations (r_a r_b r_c R : ℝ) 
    (h1 : r_a + r_b = 3 * R) (h2 : r_b + r_c = 2 * R) :
    ∃ (α β γ : ℝ),
      α = π / 2 ∧ β = π / 6 ∧ γ = π / 3 ∧
      α + β + γ = π ∧
      0 < α ∧ 0 < β ∧ 0 < γ :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_from_radii_relations_l740_74008


namespace NUMINAMATH_CALUDE_equation_solution_l740_74070

theorem equation_solution : ∃ x : ℝ, 
  3.5 * ((3.6 * x * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 0.4799999999999999) < 1e-15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l740_74070


namespace NUMINAMATH_CALUDE_min_troupe_size_l740_74003

theorem min_troupe_size : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 10 ∣ n ∧ 12 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 8 ∣ m ∧ 10 ∣ m ∧ 12 ∣ m) → n ≤ m :=
by
  use 120
  sorry

end NUMINAMATH_CALUDE_min_troupe_size_l740_74003


namespace NUMINAMATH_CALUDE_g_inverse_exists_g_inverse_composition_g_inverse_triple_composition_l740_74046

def g : Fin 5 → Fin 5
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2

theorem g_inverse_exists : Function.Bijective g := sorry

theorem g_inverse_composition (x : Fin 5) : Function.invFun g (g x) = x := sorry

theorem g_inverse_triple_composition :
  Function.invFun g (Function.invFun g (Function.invFun g 3)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_exists_g_inverse_composition_g_inverse_triple_composition_l740_74046


namespace NUMINAMATH_CALUDE_equal_roots_iff_discriminant_zero_equal_roots_h_l740_74033

/-- For a quadratic equation ax² + bx + c = 0, the discriminant is b² - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has equal roots if and only if its discriminant is zero -/
theorem equal_roots_iff_discriminant_zero (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x, a*x^2 + b*x + c = 0 ∧ (∀ y, a*y^2 + b*y + c = 0 → y = x) ↔ discriminant a b c = 0 :=
sorry

/-- The value of h for which the equation 3x² - 4x + h/3 = 0 has equal roots -/
theorem equal_roots_h : ∃! h : ℝ, discriminant 3 (-4) (h/3) = 0 ∧ h = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_iff_discriminant_zero_equal_roots_h_l740_74033


namespace NUMINAMATH_CALUDE_unique_solution_rational_equation_l740_74027

theorem unique_solution_rational_equation :
  ∃! x : ℚ, x ≠ 4 ∧ x ≠ 1 ∧
  (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_rational_equation_l740_74027


namespace NUMINAMATH_CALUDE_not_perfect_squares_l740_74097

theorem not_perfect_squares : 
  (∃ x : ℕ, 7^2040 = x^2) ∧
  (¬∃ x : ℕ, 8^2041 = x^2) ∧
  (∃ x : ℕ, 9^2042 = x^2) ∧
  (¬∃ x : ℕ, 10^2043 = x^2) ∧
  (∃ x : ℕ, 11^2044 = x^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_squares_l740_74097


namespace NUMINAMATH_CALUDE_condition1_condition2_max_type_A_dictionaries_l740_74050

/-- The price of dictionary A -/
def price_A : ℝ := 70

/-- The price of dictionary B -/
def price_B : ℝ := 50

/-- The total number of dictionaries to be purchased -/
def total_dictionaries : ℕ := 300

/-- The maximum total cost -/
def max_cost : ℝ := 16000

/-- Verification of the first condition -/
theorem condition1 : price_A + 2 * price_B = 170 := by sorry

/-- Verification of the second condition -/
theorem condition2 : 2 * price_A + 3 * price_B = 290 := by sorry

/-- The main theorem proving the maximum number of type A dictionaries -/
theorem max_type_A_dictionaries : 
  ∀ m : ℕ, m ≤ total_dictionaries ∧ 
    m * price_A + (total_dictionaries - m) * price_B ≤ max_cost → 
    m ≤ 50 := by sorry

end NUMINAMATH_CALUDE_condition1_condition2_max_type_A_dictionaries_l740_74050


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l740_74015

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The problem statement -/
theorem perpendicular_line_through_point :
  ∃ (l : Line),
    perpendicular l (Line.mk 1 (-2) (-1)) ∧
    point_on_line 1 1 l ∧
    l = Line.mk 2 1 (-3) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l740_74015


namespace NUMINAMATH_CALUDE_practice_time_is_three_l740_74006

/-- Calculates the practice time per minute of singing given the performance duration,
    tantrum time per minute of singing, and total time. -/
def practice_time_per_minute (performance_duration : ℕ) (tantrum_time_per_minute : ℕ) (total_time : ℕ) : ℕ :=
  ((total_time - performance_duration) / performance_duration) - tantrum_time_per_minute

/-- Proves that given a 6-minute performance, 5 minutes of tantrums per minute of singing,
    and a total time of 54 minutes, the practice time per minute of singing is 3 minutes. -/
theorem practice_time_is_three :
  practice_time_per_minute 6 5 54 = 3 := by
  sorry

#eval practice_time_per_minute 6 5 54

end NUMINAMATH_CALUDE_practice_time_is_three_l740_74006


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_5020030_l740_74077

def numbers : List Nat := [1000, 1001, 1002, 1003, 1004]

theorem sum_of_squares_equals_5020030 :
  (numbers.map (λ x => x * x)).sum = 5020030 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_5020030_l740_74077


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_neg_i_l740_74081

theorem imaginary_sum_equals_neg_i :
  let i : ℂ := Complex.I
  (1 / i) + (1 / i^3) + (1 / i^5) + (1 / i^7) + (1 / i^9) = -i :=
by sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_neg_i_l740_74081


namespace NUMINAMATH_CALUDE_expression_simplification_l740_74056

theorem expression_simplification (x y z : ℝ) : 
  (x - (2*y + z)) - ((x + 2*y) - 3*z) = -4*y + 2*z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l740_74056


namespace NUMINAMATH_CALUDE_constant_value_l740_74009

theorem constant_value (f : ℝ → ℝ) (c : ℝ) 
  (h1 : ∀ x : ℝ, f x + c * f (8 - x) = x) 
  (h2 : f 2 = 2) : 
  c = 3 := by sorry

end NUMINAMATH_CALUDE_constant_value_l740_74009


namespace NUMINAMATH_CALUDE_evaluate_expression_l740_74013

theorem evaluate_expression : 3 - 6 * (7 - 2^3)^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l740_74013


namespace NUMINAMATH_CALUDE_debby_flour_problem_l740_74062

theorem debby_flour_problem (x : ℝ) : x + 4 = 16 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_debby_flour_problem_l740_74062


namespace NUMINAMATH_CALUDE_difference_of_squares_l740_74098

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l740_74098


namespace NUMINAMATH_CALUDE_g_composed_four_times_is_even_l740_74026

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

theorem g_composed_four_times_is_even 
  (g : ℝ → ℝ) 
  (h : is_even_function g) : 
  is_even_function (fun x ↦ g (g (g (g x)))) :=
by
  sorry

end NUMINAMATH_CALUDE_g_composed_four_times_is_even_l740_74026


namespace NUMINAMATH_CALUDE_distinct_values_of_c_l740_74057

theorem distinct_values_of_c (c : ℂ) (p q r : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ z : ℂ, (z - p) * (z - q) * (z - r) + 1 = (z - c*p) * (z - c*q) * (z - c*r) + 1) →
  ∃ S : Finset ℂ, S.card = 4 ∧ c ∈ S ∧ ∀ x : ℂ, x ∈ S → 
    ∀ z : ℂ, (z - p) * (z - q) * (z - r) + 1 = (z - x*p) * (z - x*q) * (z - x*r) + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_values_of_c_l740_74057
