import Mathlib

namespace NUMINAMATH_CALUDE_marbles_difference_l3011_301181

/-- Given information about Josh's marble collection -/
structure MarbleCollection where
  initial : ℕ
  found : ℕ
  lost : ℕ

/-- Theorem stating the difference between lost and found marbles -/
theorem marbles_difference (josh : MarbleCollection)
  (h1 : josh.initial = 15)
  (h2 : josh.found = 9)
  (h3 : josh.lost = 23) :
  josh.lost - josh.found = 14 := by
  sorry

end NUMINAMATH_CALUDE_marbles_difference_l3011_301181


namespace NUMINAMATH_CALUDE_largest_nine_digit_divisible_by_127_l3011_301191

theorem largest_nine_digit_divisible_by_127 :
  ∀ n : ℕ, n ≤ 999999999 ∧ n % 127 = 0 → n ≤ 999999945 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_nine_digit_divisible_by_127_l3011_301191


namespace NUMINAMATH_CALUDE_woods_area_calculation_l3011_301132

/-- The area of rectangular woods -/
def woods_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of woods with width 8 miles and length 3 miles is 24 square miles -/
theorem woods_area_calculation :
  woods_area 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_woods_area_calculation_l3011_301132


namespace NUMINAMATH_CALUDE_base3_sum_correct_l3011_301157

/-- Represents a number in base 3 --/
def Base3 : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def toDecimal (n : Base3) : Nat :=
  n.reverse.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The theorem stating that the sum of the given base 3 numbers is correct --/
theorem base3_sum_correct : 
  let a : Base3 := [2]
  let b : Base3 := [2, 0, 1]
  let c : Base3 := [2, 0, 1, 1]
  let d : Base3 := [1, 2, 0, 1, 1]
  let sum : Base3 := [1, 2, 2, 1]
  toDecimal a + toDecimal b + toDecimal c + toDecimal d = toDecimal sum := by
  sorry

end NUMINAMATH_CALUDE_base3_sum_correct_l3011_301157


namespace NUMINAMATH_CALUDE_system_solution_l3011_301195

theorem system_solution (x y m : ℝ) 
  (eq1 : 2*x + y = 1) 
  (eq2 : x + 2*y = 2) 
  (eq3 : x + y = 2*m - 1) : 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l3011_301195


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l3011_301159

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.64 * A
  let r' := Real.sqrt (A' / π)
  (r' - r) / r = -0.2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l3011_301159


namespace NUMINAMATH_CALUDE_equation_holds_except_two_values_l3011_301162

theorem equation_holds_except_two_values (a : ℝ) (ha : a ≠ 0) :
  ∀ y : ℝ, y ≠ a → y ≠ -a →
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_except_two_values_l3011_301162


namespace NUMINAMATH_CALUDE_max_sum_constrained_integers_l3011_301150

theorem max_sum_constrained_integers (a b c d e f g : ℕ) 
  (eq1 : a + b + c = 2)
  (eq2 : b + c + d = 2)
  (eq3 : c + d + e = 2)
  (eq4 : d + e + f = 2)
  (eq5 : e + f + g = 2) :
  a + b + c + d + e + f + g ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constrained_integers_l3011_301150


namespace NUMINAMATH_CALUDE_proposition_d_is_false_l3011_301138

/-- Proposition D is false: There exist four mutually different non-zero vectors on a plane 
    such that the sum vector of any two vectors is perpendicular to the sum vector of 
    the remaining two vectors. -/
theorem proposition_d_is_false :
  ∃ (a b c d : ℝ × ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a.1 + b.1) * (c.1 + d.1) + (a.2 + b.2) * (c.2 + d.2) = 0 ∧
    (a.1 + c.1) * (b.1 + d.1) + (a.2 + c.2) * (b.2 + d.2) = 0 ∧
    (a.1 + d.1) * (b.1 + c.1) + (a.2 + d.2) * (b.2 + c.2) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_proposition_d_is_false_l3011_301138


namespace NUMINAMATH_CALUDE_dan_makes_fifteen_tshirts_l3011_301100

/-- The number of t-shirts Dan makes in two hours -/
def tshirts_made (minutes_per_hour : ℕ) (rate_hour1 : ℕ) (rate_hour2 : ℕ) : ℕ :=
  (minutes_per_hour / rate_hour1) + (minutes_per_hour / rate_hour2)

/-- Proof that Dan makes 15 t-shirts in two hours -/
theorem dan_makes_fifteen_tshirts :
  tshirts_made 60 12 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dan_makes_fifteen_tshirts_l3011_301100


namespace NUMINAMATH_CALUDE_line_equation_l3011_301103

/-- Given a line with slope -2 and y-intercept 4, its equation is 2x+y-4=0 -/
theorem line_equation (x y : ℝ) : 
  (∃ (m b : ℝ), m = -2 ∧ b = 4 ∧ y = m * x + b) → 2 * x + y - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3011_301103


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3011_301180

theorem quadratic_inequality_solution_range (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 6*x + c < 0) ↔ (c > 0 ∧ c < 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3011_301180


namespace NUMINAMATH_CALUDE_tangent_line_circle_product_range_l3011_301171

theorem tangent_line_circle_product_range (a b : ℝ) :
  a > 0 →
  b > 0 →
  (∃ x y : ℝ, x + y = 1 ∧ (x - a)^2 + (y - b)^2 = 2) →
  (∀ x y : ℝ, x + y = 1 → (x - a)^2 + (y - b)^2 ≥ 2) →
  0 < a * b ∧ a * b ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_circle_product_range_l3011_301171


namespace NUMINAMATH_CALUDE_fraction_of_25_comparison_l3011_301130

theorem fraction_of_25_comparison : ∃ x : ℚ, 
  (x * 25 = 80 / 100 * 60 - 28) ∧ 
  (x = 4 / 5) := by
sorry

end NUMINAMATH_CALUDE_fraction_of_25_comparison_l3011_301130


namespace NUMINAMATH_CALUDE_cricket_average_l3011_301170

theorem cricket_average (innings : Nat) (next_runs : Nat) (increase : Nat) (current_average : Nat) : 
  innings = 10 →
  next_runs = 84 →
  increase = 4 →
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 40 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l3011_301170


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l3011_301128

/-- The cost of fencing an irregularly shaped field -/
theorem fencing_cost_theorem (triangle_side1 triangle_side2 triangle_side3 circle_radius : ℝ)
  (triangle_cost_per_meter circle_cost_per_meter : ℝ)
  (h1 : triangle_side1 = 100)
  (h2 : triangle_side2 = 150)
  (h3 : triangle_side3 = 50)
  (h4 : circle_radius = 30)
  (h5 : triangle_cost_per_meter = 5)
  (h6 : circle_cost_per_meter = 7) :
  ∃ (total_cost : ℝ), 
    abs (total_cost - ((triangle_side1 + triangle_side2 + triangle_side3) * triangle_cost_per_meter +
    2 * Real.pi * circle_radius * circle_cost_per_meter)) < 1 ∧
    total_cost = 2819 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l3011_301128


namespace NUMINAMATH_CALUDE_tims_lunch_cost_l3011_301124

/-- The total amount Tim spent on lunch, including taxes, surcharge, and tips -/
def total_lunch_cost (meal_cost : ℝ) (tip_rate state_tax_rate city_tax_rate surcharge_rate : ℝ) : ℝ :=
  let tip := meal_cost * tip_rate
  let state_tax := meal_cost * state_tax_rate
  let city_tax := meal_cost * city_tax_rate
  let subtotal := meal_cost + state_tax + city_tax
  let surcharge := subtotal * surcharge_rate
  meal_cost + tip + state_tax + city_tax + surcharge

/-- Theorem stating that Tim's total lunch cost is $78.43 -/
theorem tims_lunch_cost :
  total_lunch_cost 60.50 0.20 0.05 0.03 0.015 = 78.43 := by
  sorry


end NUMINAMATH_CALUDE_tims_lunch_cost_l3011_301124


namespace NUMINAMATH_CALUDE_cube_root_of_110592_l3011_301184

theorem cube_root_of_110592 :
  ∃! (x : ℕ), x^3 = 110592 ∧ x > 0 :=
by
  use 48
  constructor
  · simp
  · intro y hy
    sorry

#eval 48^3  -- This will output 110592

end NUMINAMATH_CALUDE_cube_root_of_110592_l3011_301184


namespace NUMINAMATH_CALUDE_temperature_increase_proof_l3011_301163

/-- Represents the temperature increase per century -/
def temperature_increase_per_century : ℝ := 4

/-- Represents the total number of years -/
def total_years : ℕ := 1600

/-- Represents the total temperature change over the given years -/
def total_temperature_change : ℝ := 64

/-- Represents the number of years in a century -/
def years_per_century : ℕ := 100

theorem temperature_increase_proof :
  temperature_increase_per_century * (total_years / years_per_century) = total_temperature_change := by
  sorry

end NUMINAMATH_CALUDE_temperature_increase_proof_l3011_301163


namespace NUMINAMATH_CALUDE_P_roots_count_l3011_301109

/-- Recursive definition of the polynomial sequence Pₙ(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | 1, x => x
  | (n+2), x => x * P (n+1) x - P n x

/-- The number of distinct real roots of Pₙ(x) -/
def num_roots (n : ℕ) : ℕ := n

theorem P_roots_count (n : ℕ) : 
  (∃ (s : Finset ℝ), s.card = num_roots n ∧ 
   (∀ x ∈ s, P n x = 0) ∧
   (∀ x : ℝ, P n x = 0 → x ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_P_roots_count_l3011_301109


namespace NUMINAMATH_CALUDE_smallest_solution_is_negative_one_l3011_301125

-- Define the equation
def equation (x : ℝ) : Prop :=
  3 * x / (x - 3) + (3 * x^2 - 36) / (x + 3) = 15

-- Theorem statement
theorem smallest_solution_is_negative_one :
  (∃ x : ℝ, equation x) ∧ 
  (∀ y : ℝ, equation y → y ≥ -1) ∧
  equation (-1) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_is_negative_one_l3011_301125


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l3011_301131

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos (x + y) ≥ Real.cos x + Real.cos y - 1) ↔ 
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l3011_301131


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l3011_301169

theorem sum_and_ratio_to_difference (x y : ℝ) 
  (sum_eq : x + y = 399)
  (ratio_eq : x / y = 0.9) : 
  y - x = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l3011_301169


namespace NUMINAMATH_CALUDE_procedure_arrangement_count_l3011_301139

/-- The number of ways to arrange 6 procedures with specific constraints -/
def arrangement_count : ℕ := 96

/-- The number of procedures -/
def total_procedures : ℕ := 6

/-- The number of ways to place procedure A (first or last) -/
def a_placements : ℕ := 2

/-- The number of ways to arrange B and C within their unit -/
def bc_arrangements : ℕ := 2

/-- The number of elements to arrange (BC unit + 3 other procedures) -/
def elements_to_arrange : ℕ := 4

theorem procedure_arrangement_count :
  arrangement_count = 
    a_placements * elements_to_arrange.factorial * bc_arrangements :=
by sorry

end NUMINAMATH_CALUDE_procedure_arrangement_count_l3011_301139


namespace NUMINAMATH_CALUDE_task_selection_ways_l3011_301144

/-- The number of ways to select individuals for tasks with specific requirements -/
def select_for_tasks (total_people : ℕ) (task_a_people : ℕ) (task_b_people : ℕ) (task_c_people : ℕ) : ℕ :=
  Nat.choose total_people task_a_people *
  (Nat.choose (total_people - task_a_people) (task_b_people + task_c_people) * Nat.factorial (task_b_people + task_c_people))

/-- Theorem stating the number of ways to select 4 individuals from 10 for the given tasks -/
theorem task_selection_ways :
  select_for_tasks 10 2 1 1 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_task_selection_ways_l3011_301144


namespace NUMINAMATH_CALUDE_max_days_for_88_alligators_l3011_301160

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the eating rate of the snake in alligators per week -/
def eating_rate : ℕ := 1

/-- Represents the total number of alligators eaten -/
def total_alligators : ℕ := 88

/-- Calculates the maximum number of days to eat a given number of alligators -/
def max_days_to_eat (alligators : ℕ) (rate : ℕ) (days_in_week : ℕ) : ℕ :=
  alligators * days_in_week / rate

/-- Theorem stating that the maximum number of days to eat 88 alligators is 616 -/
theorem max_days_for_88_alligators :
  max_days_to_eat total_alligators eating_rate days_per_week = 616 := by
  sorry

end NUMINAMATH_CALUDE_max_days_for_88_alligators_l3011_301160


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_x_l3011_301172

theorem integral_sqrt_plus_x :
  ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - x^2) + x) = π / 4 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_x_l3011_301172


namespace NUMINAMATH_CALUDE_art_show_pricing_l3011_301121

/-- The price of a large painting that satisfies the given conditions -/
def large_painting_price : ℕ → ℕ → ℕ → ℕ → ℕ := λ small_price large_count small_count total_earnings =>
  (total_earnings - small_price * small_count) / large_count

theorem art_show_pricing (small_price large_count small_count total_earnings : ℕ) 
  (h1 : small_price = 80)
  (h2 : large_count = 5)
  (h3 : small_count = 8)
  (h4 : total_earnings = 1140) :
  large_painting_price small_price large_count small_count total_earnings = 100 := by
sorry

#eval large_painting_price 80 5 8 1140

end NUMINAMATH_CALUDE_art_show_pricing_l3011_301121


namespace NUMINAMATH_CALUDE_simplify_expression_l3011_301115

theorem simplify_expression (z : ℝ) : (4 - 5 * z^2) - (2 + 7 * z^2 - z) = 2 - 12 * z^2 + z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3011_301115


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l3011_301142

-- Define the monomial structure
structure Monomial where
  coefficient : ℚ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define our specific monomial
def our_monomial : Monomial := {
  coefficient := -2/3,
  x_exponent := 1,
  y_exponent := 2
}

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  our_monomial.coefficient = -2/3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  our_monomial.x_exponent + our_monomial.y_exponent = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l3011_301142


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_five_ninths_l3011_301118

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculate the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  let total_surface_area := 6 * c.edge_length^2
  let black_faces := 3 * c.black_cube_count
  let white_faces := total_surface_area - black_faces
  white_faces / total_surface_area

/-- The specific cube described in the problem -/
def problem_cube : CompositeCube :=
  { edge_length := 3
  , small_cube_count := 27
  , white_cube_count := 19
  , black_cube_count := 8 }

theorem white_surface_fraction_is_five_ninths :
  white_surface_fraction problem_cube = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_five_ninths_l3011_301118


namespace NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l3011_301107

theorem short_bingo_first_column_possibilities : Fintype.card { p : Fin 8 → Fin 4 | Function.Injective p } = 1680 := by
  sorry

end NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l3011_301107


namespace NUMINAMATH_CALUDE_dave_apps_left_l3011_301199

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 5

/-- The difference between the number of apps and files Dave has left -/
def app_file_difference : ℕ := 7

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := files_left + app_file_difference

theorem dave_apps_left : apps_left = 12 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_left_l3011_301199


namespace NUMINAMATH_CALUDE_no_integer_solution_l3011_301148

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), 
    (x^6 + x^3 + x^3*y + y = 147^137) ∧ 
    (x^3 + x^3*y + y^2 + y + z^9 = 157^117) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3011_301148


namespace NUMINAMATH_CALUDE_simplify_expressions_l3011_301176

theorem simplify_expressions :
  (∃ x y : ℝ, x^2 = 8 ∧ y^2 = 3 ∧ 
    x + 2*y - (3*y - Real.sqrt 2) = 3*Real.sqrt 2 - y) ∧
  (∃ a b : ℝ, a^2 = 2 ∧ b^2 = 3 ∧ 
    (a - b)^2 + 2*Real.sqrt (1/3) * 3*a = 5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3011_301176


namespace NUMINAMATH_CALUDE_driver_net_rate_of_pay_l3011_301186

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_per_mile : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_price = 2.50)
  : (pay_per_mile * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 := by
  sorry

#check driver_net_rate_of_pay

end NUMINAMATH_CALUDE_driver_net_rate_of_pay_l3011_301186


namespace NUMINAMATH_CALUDE_original_number_proof_l3011_301108

def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

theorem original_number_proof :
  let original := 1453789
  let swapped := 8453719
  (∃ i j, swap_digits original i j = swapped) ∧
  (swapped > 3 * original) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l3011_301108


namespace NUMINAMATH_CALUDE_math_class_size_l3011_301158

/-- Represents a class of students who took a math test. -/
structure MathClass where
  total_students : ℕ
  both_solvers : ℕ
  harder_solvers : ℕ
  easier_solvers : ℕ

/-- Conditions for the math class problem. -/
def valid_math_class (c : MathClass) : Prop :=
  -- Each student solved at least one problem
  c.total_students = c.both_solvers + c.harder_solvers + c.easier_solvers
  -- Number of students who solved only one problem is one less than twice the number who solved both
  ∧ c.harder_solvers + c.easier_solvers = 2 * c.both_solvers - 1
  -- Total homework solutions from (both + harder) equals total from easier
  ∧ c.both_solvers + 4 * c.harder_solvers = c.easier_solvers

/-- The theorem stating that the class has 32 students. -/
theorem math_class_size :
  ∃ (c : MathClass), valid_math_class c ∧ c.total_students = 32 :=
sorry

end NUMINAMATH_CALUDE_math_class_size_l3011_301158


namespace NUMINAMATH_CALUDE_salt_concentration_dilution_l3011_301174

/-- Proves that adding 70 kg of fresh water to 30 kg of sea water with 5% salt concentration
    results in a solution with 1.5% salt concentration. -/
theorem salt_concentration_dilution
  (initial_mass : ℝ)
  (initial_concentration : ℝ)
  (target_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_mass = 30)
  (h2 : initial_concentration = 0.05)
  (h3 : target_concentration = 0.015)
  (h4 : added_water = 70) :
  let final_mass := initial_mass + added_water
  let salt_mass := initial_mass * initial_concentration
  (salt_mass / final_mass) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_concentration_dilution_l3011_301174


namespace NUMINAMATH_CALUDE_stickers_per_page_l3011_301167

theorem stickers_per_page (total_stickers : ℕ) (total_pages : ℕ) (h1 : total_stickers = 220) (h2 : total_pages = 22) :
  total_stickers / total_pages = 10 := by
  sorry

end NUMINAMATH_CALUDE_stickers_per_page_l3011_301167


namespace NUMINAMATH_CALUDE_intersection_of_P_and_M_l3011_301123

-- Define the sets P and M
def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | x^2 ≤ 9}

-- State the theorem
theorem intersection_of_P_and_M : P ∩ M = {x | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_M_l3011_301123


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3011_301153

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members the host team contributes to the committee -/
def host_contribution : ℕ := 3

/-- The number of members each non-host team contributes to the committee -/
def non_host_contribution : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 11

/-- Theorem stating the number of possible tournament committees -/
theorem tournament_committee_count : 
  (num_teams * (Nat.choose team_size host_contribution) * 
   (Nat.choose team_size non_host_contribution)^(num_teams - 1)) = 172043520 := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3011_301153


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l3011_301193

def marble_problem (initial_marbles : ℕ) (lost_through_hole : ℕ) (final_marbles : ℕ) : Prop :=
  let dog_eaten : ℕ := lost_through_hole / 2
  let before_giving_away : ℕ := initial_marbles - lost_through_hole - dog_eaten
  let given_away : ℕ := before_giving_away - final_marbles
  (given_away : ℚ) / lost_through_hole = 2

theorem marble_ratio_proof :
  marble_problem 24 4 10 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l3011_301193


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3011_301183

/-- The quadratic equation x^2 + 2x + m + 1 = 0 has two distinct real roots if and only if m < 0 -/
theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + m + 1 = 0 ∧ y^2 + 2*y + m + 1 = 0) ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3011_301183


namespace NUMINAMATH_CALUDE_faucet_filling_time_l3011_301194

/-- Given that four faucets can fill a 150-gallon tub in 8 minutes,
    prove that eight faucets will fill a 50-gallon tub in 4/3 minutes. -/
theorem faucet_filling_time 
  (volume_large : ℝ) 
  (volume_small : ℝ)
  (time_large : ℝ)
  (faucets_large : ℕ)
  (faucets_small : ℕ)
  (h1 : volume_large = 150)
  (h2 : volume_small = 50)
  (h3 : time_large = 8)
  (h4 : faucets_large = 4)
  (h5 : faucets_small = 8) :
  (volume_small * time_large * faucets_large) / (volume_large * faucets_small) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_faucet_filling_time_l3011_301194


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l3011_301179

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The average point of a dataset -/
structure AveragePoint where
  x : ℝ
  y : ℝ

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect_at_average_point 
  (l₁ l₂ : RegressionLine) 
  (avg : AveragePoint) : 
  (avg.x * l₁.slope + l₁.intercept = avg.y) ∧ 
  (avg.x * l₂.slope + l₂.intercept = avg.y) := by
  sorry

#check regression_lines_intersect_at_average_point

end NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l3011_301179


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l3011_301187

theorem different_tens_digit_probability : 
  let total_integers : ℕ := 70
  let chosen_integers : ℕ := 7
  let tens_digits : ℕ := 7
  let integers_per_tens : ℕ := 10

  let favorable_outcomes : ℕ := integers_per_tens ^ chosen_integers
  let total_outcomes : ℕ := Nat.choose total_integers chosen_integers

  (favorable_outcomes : ℚ) / total_outcomes = 20000 / 83342961 := by
  sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l3011_301187


namespace NUMINAMATH_CALUDE_frog_hops_l3011_301145

theorem frog_hops (frog1 frog2 frog3 : ℕ) : 
  frog1 = 4 * frog2 →
  frog2 = 2 * frog3 →
  frog2 = 18 →
  frog1 + frog2 + frog3 = 99 := by
sorry

end NUMINAMATH_CALUDE_frog_hops_l3011_301145


namespace NUMINAMATH_CALUDE_problem_solution_l3011_301168

theorem problem_solution (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 849 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3011_301168


namespace NUMINAMATH_CALUDE_pie_slices_theorem_l3011_301190

/-- Given the total number of pie slices sold and the number sold yesterday,
    calculate the number of slices served today. -/
def slices_served_today (total : ℕ) (yesterday : ℕ) : ℕ :=
  total - yesterday

theorem pie_slices_theorem :
  slices_served_today 7 5 = 2 :=
by sorry

end NUMINAMATH_CALUDE_pie_slices_theorem_l3011_301190


namespace NUMINAMATH_CALUDE_two_digit_S_equals_50_l3011_301101

/-- R(n) is the sum of remainders when n is divided by 2, 3, 4, 5, and 6 -/
def R (n : ℕ) : ℕ :=
  n % 2 + n % 3 + n % 4 + n % 5 + n % 6

/-- S(n) is defined as R(n) + R(n+2) -/
def S (n : ℕ) : ℕ :=
  R n + R (n + 2)

/-- A two-digit number is between 10 and 99, inclusive -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- There are exactly 2 two-digit integers n such that S(n) = 50 -/
theorem two_digit_S_equals_50 :
  ∃! (count : ℕ), count = (Finset.filter (fun n => S n = 50) (Finset.range 90)).card ∧ count = 2 :=
sorry

end NUMINAMATH_CALUDE_two_digit_S_equals_50_l3011_301101


namespace NUMINAMATH_CALUDE_counterexample_exists_l3011_301117

theorem counterexample_exists : ∃ n : ℕ, 
  ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 1)) ∧ ¬(Nat.Prime (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3011_301117


namespace NUMINAMATH_CALUDE_equation_solution_l3011_301136

theorem equation_solution : 
  ∃ x : ℚ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3011_301136


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3011_301111

/-- The eccentricity of a hyperbola with equation x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 2
  let e : ℝ := c / a
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3011_301111


namespace NUMINAMATH_CALUDE_opposite_event_of_hit_at_least_once_l3011_301149

-- Define the sample space for two shots
inductive ShotOutcome
  | Hit
  | Miss

-- Define the event of hitting the target at least once
def hitAtLeastOnce (outcome1 outcome2 : ShotOutcome) : Prop :=
  outcome1 = ShotOutcome.Hit ∨ outcome2 = ShotOutcome.Hit

-- Define the event of both shots missing
def bothShotsMiss (outcome1 outcome2 : ShotOutcome) : Prop :=
  outcome1 = ShotOutcome.Miss ∧ outcome2 = ShotOutcome.Miss

-- Theorem statement
theorem opposite_event_of_hit_at_least_once 
  (outcome1 outcome2 : ShotOutcome) : 
  ¬(hitAtLeastOnce outcome1 outcome2) ↔ bothShotsMiss outcome1 outcome2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_event_of_hit_at_least_once_l3011_301149


namespace NUMINAMATH_CALUDE_number_of_reactions_l3011_301133

def visible_readings : List ℝ := [2, 2.1, 2, 2.2]

theorem number_of_reactions (x : ℝ) (h1 : (visible_readings.sum + x) / (visible_readings.length + 1) = 2) :
  visible_readings.length + 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_number_of_reactions_l3011_301133


namespace NUMINAMATH_CALUDE_f_two_thirds_eq_three_halves_l3011_301156

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (y : ℝ) : ℝ := y / ((2 - y) / 3)

theorem f_two_thirds_eq_three_halves :
  f (2/3) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_f_two_thirds_eq_three_halves_l3011_301156


namespace NUMINAMATH_CALUDE_car_travel_and_budget_l3011_301196

/-- Represents a car with its fuel-to-distance ratio and fuel usage -/
structure Car where
  fuel_ratio : Rat
  distance_ratio : Rat
  fuel_used : ℚ
  fuel_cost : ℚ

/-- Calculates the distance traveled by a car -/
def distance_traveled (c : Car) : ℚ :=
  c.distance_ratio * c.fuel_used / c.fuel_ratio

/-- Calculates the fuel cost for a car -/
def fuel_cost (c : Car) : ℚ :=
  c.fuel_cost * c.fuel_used

theorem car_travel_and_budget (car_a car_b : Car) (budget : ℚ) :
  car_a.fuel_ratio = 4/7 ∧
  car_a.distance_ratio = 7/4 ∧
  car_a.fuel_used = 44 ∧
  car_a.fuel_cost = 7/2 ∧
  car_b.fuel_ratio = 3/5 ∧
  car_b.distance_ratio = 5/3 ∧
  car_b.fuel_used = 27 ∧
  car_b.fuel_cost = 13/4 ∧
  budget = 200 →
  distance_traveled car_a + distance_traveled car_b = 122 ∧
  fuel_cost car_a + fuel_cost car_b = 967/4 ∧
  fuel_cost car_a + fuel_cost car_b - budget = 167/4 :=
by sorry

end NUMINAMATH_CALUDE_car_travel_and_budget_l3011_301196


namespace NUMINAMATH_CALUDE_g_evaluation_l3011_301198

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem g_evaluation : 3 * g 2 + 2 * g (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_g_evaluation_l3011_301198


namespace NUMINAMATH_CALUDE_chooseBoxes_eq_sixteen_l3011_301119

/-- The number of ways to choose 3 out of 6 boxes with at least one of A or B chosen -/
def chooseBoxes : ℕ := sorry

/-- There are 6 boxes in total -/
def totalBoxes : ℕ := 6

/-- The number of boxes to be chosen -/
def boxesToChoose : ℕ := 3

/-- The theorem stating that the number of ways to choose 3 out of 6 boxes 
    with at least one of A or B chosen is 16 -/
theorem chooseBoxes_eq_sixteen : chooseBoxes = 16 := by sorry

end NUMINAMATH_CALUDE_chooseBoxes_eq_sixteen_l3011_301119


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3011_301154

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![4, 3; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3011_301154


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3011_301165

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: If a_6 = S_3 = 12 in an arithmetic sequence, then a_8 = 16 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 6 = 12) (h2 : seq.S 3 = 12) : seq.a 8 = 16 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3011_301165


namespace NUMINAMATH_CALUDE_inequality_proof_l3011_301126

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3011_301126


namespace NUMINAMATH_CALUDE_fernandez_family_has_nine_children_l3011_301182

/-- Represents the Fernandez family structure and ages -/
structure FernandezFamily where
  num_children : ℕ
  mother_age : ℕ
  children_ages : ℕ → ℕ
  average_family_age : ℕ
  father_age : ℕ
  grandmother_age : ℕ
  average_mother_children_age : ℕ

/-- The Fernandez family satisfies the given conditions -/
def is_valid_fernandez_family (f : FernandezFamily) : Prop :=
  f.average_family_age = 25 ∧
  f.father_age = 50 ∧
  f.grandmother_age = 70 ∧
  f.average_mother_children_age = 18

/-- The theorem stating that the Fernandez family has 9 children -/
theorem fernandez_family_has_nine_children (f : FernandezFamily) 
  (h : is_valid_fernandez_family f) : f.num_children = 9 := by
  sorry

#check fernandez_family_has_nine_children

end NUMINAMATH_CALUDE_fernandez_family_has_nine_children_l3011_301182


namespace NUMINAMATH_CALUDE_paint_O_circles_l3011_301178

theorem paint_O_circles (num_circles : Nat) (num_colors : Nat) : 
  num_circles = 4 → num_colors = 3 → num_colors ^ num_circles = 81 := by
  sorry

#check paint_O_circles

end NUMINAMATH_CALUDE_paint_O_circles_l3011_301178


namespace NUMINAMATH_CALUDE_max_trig_sum_l3011_301127

theorem max_trig_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
  Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
  Real.cos θ₅ * Real.sin θ₁ ≤ 5/2 := by
sorry

end NUMINAMATH_CALUDE_max_trig_sum_l3011_301127


namespace NUMINAMATH_CALUDE_hotel_room_encoding_l3011_301146

theorem hotel_room_encoding (x : ℕ) : 
  1 ≤ x ∧ x ≤ 30 ∧ x % 5 = 1 ∧ x % 7 = 6 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_encoding_l3011_301146


namespace NUMINAMATH_CALUDE_concert_attendance_l3011_301185

theorem concert_attendance (adults : ℕ) (children : ℕ) : 
  children = 3 * adults →
  7 * adults + 3 * children = 6000 →
  adults + children = 1500 := by
sorry

end NUMINAMATH_CALUDE_concert_attendance_l3011_301185


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3011_301152

theorem simplify_sqrt_expression (x : ℝ) (h : x < 0) :
  x * Real.sqrt (-1/x) = -Real.sqrt (-x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3011_301152


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3011_301137

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 3 ≥ -6 ∧ ∃ y : ℝ, y^2 + 6*y + 3 = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3011_301137


namespace NUMINAMATH_CALUDE_circle_center_sum_l3011_301166

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 9, 
    the sum of the x and y coordinates of its center is 7 -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 9 → ∃ h k : ℝ, (h, k) = (3, 4) ∧ h + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3011_301166


namespace NUMINAMATH_CALUDE_child_worker_wage_l3011_301116

def num_male : ℕ := 20
def num_female : ℕ := 15
def num_child : ℕ := 5
def wage_male : ℕ := 35
def wage_female : ℕ := 20
def average_wage : ℕ := 26

theorem child_worker_wage :
  ∃ (wage_child : ℕ),
    (num_male * wage_male + num_female * wage_female + num_child * wage_child) / 
    (num_male + num_female + num_child) = average_wage ∧
    wage_child = 8 := by
  sorry

end NUMINAMATH_CALUDE_child_worker_wage_l3011_301116


namespace NUMINAMATH_CALUDE_product_not_always_greater_than_factors_l3011_301141

theorem product_not_always_greater_than_factors : ∃ (a b : ℝ), a * b ≤ a ∨ a * b ≤ b := by
  sorry

end NUMINAMATH_CALUDE_product_not_always_greater_than_factors_l3011_301141


namespace NUMINAMATH_CALUDE_august_day_occurrences_l3011_301105

/-- Represents days of the week -/
inductive Weekday
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Returns the next day of the week -/
def nextDay (d : Weekday) : Weekday :=
  match d with
  | Weekday.sunday => Weekday.monday
  | Weekday.monday => Weekday.tuesday
  | Weekday.tuesday => Weekday.wednesday
  | Weekday.wednesday => Weekday.thursday
  | Weekday.thursday => Weekday.friday
  | Weekday.friday => Weekday.saturday
  | Weekday.saturday => Weekday.sunday

/-- Counts occurrences of a specific day in a month -/
def countDayOccurrences (startDay : Weekday) (days : Nat) (targetDay : Weekday) : Nat :=
  sorry

theorem august_day_occurrences
  (july_start : Weekday)
  (july_days : Nat)
  (july_sundays : Nat)
  (august_days : Nat)
  (h1 : july_start = Weekday.saturday)
  (h2 : july_days = 31)
  (h3 : july_sundays = 5)
  (h4 : august_days = 31) :
  let august_start := (List.range july_days).foldl (fun d _ => nextDay d) july_start
  (countDayOccurrences august_start august_days Weekday.tuesday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.wednesday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.thursday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.friday = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_august_day_occurrences_l3011_301105


namespace NUMINAMATH_CALUDE_equation_describes_two_lines_l3011_301120

theorem equation_describes_two_lines :
  ∀ x y : ℝ, (x - y)^2 = x^2 - y^2 ↔ x * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_describes_two_lines_l3011_301120


namespace NUMINAMATH_CALUDE_apples_picked_total_l3011_301104

/-- The total number of apples picked by Mike, Nancy, Keith, Olivia, and Thomas -/
def total_apples (mike nancy keith olivia thomas : Real) : Real :=
  mike + nancy + keith + olivia + thomas

/-- Theorem stating that the total number of apples picked is 37.8 -/
theorem apples_picked_total :
  total_apples 7.5 3.2 6.1 12.4 8.6 = 37.8 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_total_l3011_301104


namespace NUMINAMATH_CALUDE_fraction_equality_l3011_301192

theorem fraction_equality (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3011_301192


namespace NUMINAMATH_CALUDE_price_after_discount_l3011_301175

/-- 
Theorem: If an article's price after a 50% decrease is 1200 (in some currency unit), 
then its original price was 2400 (in the same currency unit).
-/
theorem price_after_discount (price_after : ℝ) (discount_percent : ℝ) (original_price : ℝ) : 
  price_after = 1200 ∧ discount_percent = 50 → original_price = 2400 :=
by sorry

end NUMINAMATH_CALUDE_price_after_discount_l3011_301175


namespace NUMINAMATH_CALUDE_two_roots_condition_l3011_301151

theorem two_roots_condition (a : ℝ) :
  (∃! (x y : ℝ), x ≠ y ∧ x^2 + 2*x + 2*|x + 1| = a ∧ y^2 + 2*y + 2*|y + 1| = a) ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_two_roots_condition_l3011_301151


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3011_301129

theorem perfect_square_condition (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3*n = k^2 ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3011_301129


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l3011_301114

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- Constructs the next triangle in the sequence if it exists -/
def nextTriangle (T : Triangle) : Option Triangle := sorry

/-- The sequence of triangles starting from T₁ -/
def triangleSequence : ℕ → Option Triangle
  | 0 => some ⟨20, 21, 29, sorry, sorry⟩
  | n + 1 => match triangleSequence n with
    | none => none
    | some T => nextTriangle T

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Finds the last valid triangle in the sequence -/
def lastTriangle : Option Triangle := sorry

theorem last_triangle_perimeter :
  ∀ T, lastTriangle = some T → perimeter T = 35 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l3011_301114


namespace NUMINAMATH_CALUDE_intersected_half_of_non_intersected_for_three_l3011_301143

/-- The number of unit cubes intersected by space diagonals in a cube of edge length n -/
def intersected_cubes (n : ℕ) : ℕ :=
  if n % 2 = 0 then 4 * n else 4 * n - 3

/-- The total number of unit cubes in a cube of edge length n -/
def total_cubes (n : ℕ) : ℕ := n^3

/-- The number of unit cubes not intersected by space diagonals in a cube of edge length n -/
def non_intersected_cubes (n : ℕ) : ℕ := total_cubes n - intersected_cubes n

/-- Theorem stating that for a cube with edge length 3, the number of intersected cubes
    is exactly half the number of non-intersected cubes -/
theorem intersected_half_of_non_intersected_for_three :
  2 * intersected_cubes 3 = non_intersected_cubes 3 := by
  sorry

end NUMINAMATH_CALUDE_intersected_half_of_non_intersected_for_three_l3011_301143


namespace NUMINAMATH_CALUDE_cube_monotonicity_l3011_301177

theorem cube_monotonicity (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end NUMINAMATH_CALUDE_cube_monotonicity_l3011_301177


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l3011_301197

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z + 6*I) = 22) :
  ∃ (w : ℂ), Complex.abs (z - 15) + Complex.abs (z + 6*I) = 22 ∧ 
             Complex.abs w ≤ Complex.abs z ∧
             Complex.abs w = 45/11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l3011_301197


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3011_301110

/-- Given two polynomials in p, prove that their difference simplifies to the given result. -/
theorem polynomial_simplification (p : ℝ) :
  (2 * p^4 - 3 * p^3 + 7 * p - 4) - (-6 * p^3 - 5 * p^2 + 4 * p + 3) =
  2 * p^4 + 3 * p^3 + 5 * p^2 + 3 * p - 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3011_301110


namespace NUMINAMATH_CALUDE_one_in_set_l3011_301102

theorem one_in_set : 1 ∈ {x : ℝ | (x - 1) * (x + 2) = 0} := by sorry

end NUMINAMATH_CALUDE_one_in_set_l3011_301102


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3011_301188

theorem other_root_of_quadratic (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3011_301188


namespace NUMINAMATH_CALUDE_inequality_chain_l3011_301135

/-- Given a > 0, b > 0, a ≠ b, prove that f((a+b)/2) < f(√(ab)) < f(2ab/(a+b)) where f(x) = (1/3)^x -/
theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  let f : ℝ → ℝ := fun x ↦ (1/3)^x
  f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f (2 * a * b / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l3011_301135


namespace NUMINAMATH_CALUDE_max_value_and_constraint_l3011_301112

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - 2*|x + 1|

-- Define the maximum value of f
def m : ℝ := 4

-- Theorem statement
theorem max_value_and_constraint (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_constraint : a^2 + 2*b^2 + c^2 = 2*m) :
  (∀ x, f x ≤ m) ∧ (∃ x, f x = m) ∧ (ab + bc ≤ 2) ∧ (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a'^2 + 2*b'^2 + c'^2 = 2*m ∧ a'*b' + b'*c' = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_constraint_l3011_301112


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l3011_301147

theorem decimal_to_percentage (x : ℝ) : x = 5.02 → (x * 100 : ℝ) = 502 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l3011_301147


namespace NUMINAMATH_CALUDE_unique_digit_A_l3011_301140

def base5ToDecimal (a : ℕ) : ℕ := 25 + 6 * a

def base6ToDecimal (a : ℕ) : ℕ := 36 + 7 * a

def isPerfectSquare (n : ℕ) : Prop := ∃ x : ℕ, x * x = n

def isPerfectCube (n : ℕ) : Prop := ∃ y : ℕ, y * y * y = n

theorem unique_digit_A : 
  ∃! a : ℕ, a ≤ 4 ∧ 
    isPerfectSquare (base5ToDecimal a) ∧ 
    isPerfectCube (base6ToDecimal a) :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_A_l3011_301140


namespace NUMINAMATH_CALUDE_min_value_abc_l3011_301173

theorem min_value_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + 2*a*b + 2*a*c + 4*b*c = 16) :
  ∃ m : ℝ, ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  x^2 + 2*x*y + 2*x*z + 4*y*z = 16 → m ≤ x + y + z :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l3011_301173


namespace NUMINAMATH_CALUDE_sallys_peaches_l3011_301155

/-- The total number of peaches Sally has after picking more from the orchard -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem stating that given Sally's initial 13 peaches and her additional 55 picked peaches, 
    the total number of peaches is 68 -/
theorem sallys_peaches : total_peaches 13 55 = 68 := by
  sorry

end NUMINAMATH_CALUDE_sallys_peaches_l3011_301155


namespace NUMINAMATH_CALUDE_range_of_a_l3011_301106

open Set Real

def A (a : ℝ) : Set ℝ := {x | 3 + a ≤ x ∧ x ≤ 4 + 3*a}
def B : Set ℝ := {x | (x + 4) / (5 - x) ≥ 0}

theorem range_of_a :
  ∀ a : ℝ, (A a).Nonempty ∧ (∀ x : ℝ, x ∈ A a → x ∈ B) →
  a ∈ Icc (-1/2) (1/3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3011_301106


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3011_301122

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 7 →
  a * b = t - 6 * Complex.I →
  t = 9 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3011_301122


namespace NUMINAMATH_CALUDE_lcm_180_504_l3011_301164

theorem lcm_180_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_180_504_l3011_301164


namespace NUMINAMATH_CALUDE_system_solution_l3011_301134

theorem system_solution (x y : ℝ) (eq1 : x + 5*y = 5) (eq2 : 3*x - y = 3) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3011_301134


namespace NUMINAMATH_CALUDE_megatek_rd_percentage_l3011_301161

theorem megatek_rd_percentage :
  ∀ (manufacturing_angle hr_angle sales_angle rd_angle : ℝ),
  manufacturing_angle = 54 →
  hr_angle = 2 * manufacturing_angle →
  sales_angle = (1/2) * hr_angle →
  rd_angle = 360 - (manufacturing_angle + hr_angle + sales_angle) →
  (rd_angle / 360) * 100 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_megatek_rd_percentage_l3011_301161


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3011_301189

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 2/7
  let a₂ : ℚ := 10/49
  let a₃ : ℚ := 50/343
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = (2/7) * (5/7)^(n-1)) →
  r = 5/7 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3011_301189


namespace NUMINAMATH_CALUDE_petyas_addition_mistake_l3011_301113

theorem petyas_addition_mistake :
  ∃ (x y : ℕ) (c : Fin 10),
    x + y = 12345 ∧
    (10 * x + c.val) + y = 44444 ∧
    x = 3566 ∧
    y = 8779 := by
  sorry

end NUMINAMATH_CALUDE_petyas_addition_mistake_l3011_301113
