import Mathlib

namespace NUMINAMATH_CALUDE_buttons_per_shirt_l1591_159162

/-- Given Jack's shirt-making scenario, prove the number of buttons per shirt. -/
theorem buttons_per_shirt (num_kids : ℕ) (shirts_per_kid : ℕ) (total_buttons : ℕ) : 
  num_kids = 3 →
  shirts_per_kid = 3 →
  total_buttons = 63 →
  ∃ (buttons_per_shirt : ℕ), 
    buttons_per_shirt * (num_kids * shirts_per_kid) = total_buttons ∧
    buttons_per_shirt = 7 :=
by sorry

end NUMINAMATH_CALUDE_buttons_per_shirt_l1591_159162


namespace NUMINAMATH_CALUDE_inequality_solution_l1591_159130

theorem inequality_solution (x : ℕ+) : 
  (x.val - 3) / 3 < 7 - (5 / 3) * x.val ↔ x.val = 1 ∨ x.val = 2 ∨ x.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1591_159130


namespace NUMINAMATH_CALUDE_line_l_properties_l1591_159194

/-- The line l is defined by the equation (a^2 + a + 1)x - y + 1 = 0, where a is a real number -/
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

/-- The perpendicular line is defined by the equation x + y = 0 -/
def perp_line (x y : ℝ) : Prop :=
  x + y = 0

theorem line_l_properties :
  (∀ a : ℝ, line_l a 0 1) ∧ 
  (∀ x y : ℝ, line_l 0 x y → perp_line x y) :=
by sorry

end NUMINAMATH_CALUDE_line_l_properties_l1591_159194


namespace NUMINAMATH_CALUDE_fraction_transformation_l1591_159132

theorem fraction_transformation (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →
  (2 + 3 : ℚ) / (d + 3) = 1 / 3 →
  d = 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1591_159132


namespace NUMINAMATH_CALUDE_amanda_notebooks_problem_l1591_159134

theorem amanda_notebooks_problem (initial_notebooks ordered_notebooks loss_percentage : ℕ) 
  (h1 : initial_notebooks = 65)
  (h2 : ordered_notebooks = 23)
  (h3 : loss_percentage = 15) : 
  initial_notebooks + ordered_notebooks - (((initial_notebooks + ordered_notebooks) * loss_percentage) / 100) = 75 := by
  sorry

end NUMINAMATH_CALUDE_amanda_notebooks_problem_l1591_159134


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1591_159122

theorem matrix_equation_solution :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ),
  M^3 - 5 • M^2 + 6 • M = !![16, 8; 24, 12] →
  M = !![4, 2; 6, 3] := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1591_159122


namespace NUMINAMATH_CALUDE_teams_of_four_from_seven_l1591_159108

theorem teams_of_four_from_seven (n : ℕ) (k : ℕ) : n = 7 → k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_teams_of_four_from_seven_l1591_159108


namespace NUMINAMATH_CALUDE_profit_at_twenty_reduction_max_profit_at_fifteen_reduction_l1591_159119

-- Define the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Theorem for part 1
theorem profit_at_twenty_reduction (x : ℝ) :
  x = 20 → profit_function x = 1200 := by sorry

-- Theorem for part 2
theorem max_profit_at_fifteen_reduction :
  ∃ (x : ℝ), x = 15 ∧ 
  profit_function x = 1250 ∧ 
  ∀ (y : ℝ), profit_function y ≤ profit_function x := by sorry

end NUMINAMATH_CALUDE_profit_at_twenty_reduction_max_profit_at_fifteen_reduction_l1591_159119


namespace NUMINAMATH_CALUDE_special_triangle_not_necessarily_right_l1591_159152

/-- A triangle with sides a, b, and c where a² = 5, b² = 12, and c² = 13 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a^2 = 5
  hb : b^2 = 12
  hc : c^2 = 13

/-- A right triangle is a triangle where one of its angles is 90 degrees -/
def IsRightTriangle (t : SpecialTriangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2

/-- Theorem stating that it cannot be determined if a SpecialTriangle is a right triangle -/
theorem special_triangle_not_necessarily_right (t : SpecialTriangle) :
  ¬ (IsRightTriangle t) := by sorry

end NUMINAMATH_CALUDE_special_triangle_not_necessarily_right_l1591_159152


namespace NUMINAMATH_CALUDE_faye_coloring_books_l1591_159171

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := sorry

theorem faye_coloring_books : 
  let initial_books : ℕ := 34
  let books_bought : ℕ := 48
  let final_books : ℕ := 79
  initial_books - books_given_away + books_bought = final_books ∧ 
  books_given_away = 3 := by sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l1591_159171


namespace NUMINAMATH_CALUDE_ant_walk_theorem_l1591_159197

/-- The length of a cube's side in centimeters -/
def cube_side_length : ℝ := 18

/-- The number of cube edges the ant walks along -/
def number_of_edges : ℕ := 5

/-- The distance the ant walks on the cube's surface -/
def ant_walk_distance : ℝ := cube_side_length * number_of_edges

theorem ant_walk_theorem : ant_walk_distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_ant_walk_theorem_l1591_159197


namespace NUMINAMATH_CALUDE_total_sum_lent_l1591_159153

/-- Represents the sum of money lent in two parts -/
structure LoanParts where
  first : ℝ
  second : ℝ

/-- Calculates the interest for a given principal, rate, and time -/
def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent (loan : LoanParts) :
  loan.second = 1648 →
  interest loan.first 0.03 8 = interest loan.second 0.05 3 →
  loan.first + loan.second = 2678 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_lent_l1591_159153


namespace NUMINAMATH_CALUDE_largest_interesting_number_l1591_159157

/-- A real number is interesting if removing one digit from its decimal representation results in 2x -/
def IsInteresting (x : ℝ) : Prop :=
  ∃ (y : ℕ) (z : ℝ), 0 < x ∧ x < 1 ∧ x = y / 10 + z ∧ 2 * x = z

/-- The largest interesting number is 0.375 -/
theorem largest_interesting_number :
  IsInteresting (3 / 8) ∧ ∀ x : ℝ, IsInteresting x → x ≤ 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_largest_interesting_number_l1591_159157


namespace NUMINAMATH_CALUDE_transportation_charges_calculation_l1591_159170

theorem transportation_charges_calculation (purchase_price repair_cost selling_price : ℕ) 
  (h1 : purchase_price = 14000)
  (h2 : repair_cost = 5000)
  (h3 : selling_price = 30000)
  (h4 : selling_price = (purchase_price + repair_cost + transportation_charges) * 3 / 2) :
  transportation_charges = 1000 :=
by
  sorry

#check transportation_charges_calculation

end NUMINAMATH_CALUDE_transportation_charges_calculation_l1591_159170


namespace NUMINAMATH_CALUDE_optimal_solution_is_valid_and_unique_l1591_159124

/-- Represents the solution for the tourist attraction problem -/
structure TouristAttractionSolution where
  small_car_cost : ℕ
  large_car_cost : ℕ
  small_car_trips : ℕ
  large_car_trips : ℕ

/-- Checks if a solution is valid for the tourist attraction problem -/
def is_valid_solution (s : TouristAttractionSolution) : Prop :=
  -- Total number of employees is 70
  4 * s.small_car_trips + 11 * s.large_car_trips = 70 ∧
  -- Small car cost is 5 more than large car cost
  s.small_car_cost = s.large_car_cost + 5 ∧
  -- Revenue difference between large and small car when fully loaded
  11 * s.large_car_cost - 4 * s.small_car_cost = 50 ∧
  -- Total cost does not exceed 5000
  70 * 60 + 4 * s.small_car_trips * s.small_car_cost + 
  11 * s.large_car_trips * s.large_car_cost ≤ 5000

/-- The optimal solution for the tourist attraction problem -/
def optimal_solution : TouristAttractionSolution :=
  { small_car_cost := 15
  , large_car_cost := 10
  , small_car_trips := 1
  , large_car_trips := 6 }

/-- Theorem stating that the optimal solution is valid and unique -/
theorem optimal_solution_is_valid_and_unique :
  is_valid_solution optimal_solution ∧
  ∀ s : TouristAttractionSolution, 
    is_valid_solution s → s = optimal_solution :=
sorry


end NUMINAMATH_CALUDE_optimal_solution_is_valid_and_unique_l1591_159124


namespace NUMINAMATH_CALUDE_chocolates_distribution_l1591_159149

/-- Given a large box containing small boxes and chocolate bars, 
    calculate the number of chocolate bars in each small box. -/
def chocolates_per_small_box (total_chocolates : ℕ) (num_small_boxes : ℕ) : ℕ :=
  total_chocolates / num_small_boxes

/-- Theorem: In a large box with 15 small boxes and 300 chocolate bars,
    each small box contains 20 chocolate bars. -/
theorem chocolates_distribution :
  chocolates_per_small_box 300 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_distribution_l1591_159149


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l1591_159175

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l1591_159175


namespace NUMINAMATH_CALUDE_raisin_mixture_problem_l1591_159143

theorem raisin_mixture_problem (raisin_cost nut_cost : ℝ) (raisin_weight : ℝ) :
  nut_cost = 3 * raisin_cost →
  raisin_weight * raisin_cost = 0.29411764705882354 * (raisin_weight * raisin_cost + 4 * nut_cost) →
  raisin_weight = 5 := by
sorry

end NUMINAMATH_CALUDE_raisin_mixture_problem_l1591_159143


namespace NUMINAMATH_CALUDE_f_pi_third_eq_half_l1591_159150

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi / 2 + α)) / 
  (Real.cos (-Real.pi / 2 + α) * Real.tan (Real.pi + α))

theorem f_pi_third_eq_half : f (Real.pi / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_f_pi_third_eq_half_l1591_159150


namespace NUMINAMATH_CALUDE_total_tax_percentage_l1591_159126

/-- Calculate the total tax percentage given spending percentages, discounts, and tax rates --/
theorem total_tax_percentage
  (total_amount : ℝ)
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (electronics_percent : ℝ)
  (other_percent : ℝ)
  (clothing_discount : ℝ)
  (electronics_discount : ℝ)
  (clothing_tax : ℝ)
  (food_tax : ℝ)
  (electronics_tax : ℝ)
  (other_tax : ℝ)
  (h1 : clothing_percent = 0.4)
  (h2 : food_percent = 0.15)
  (h3 : electronics_percent = 0.25)
  (h4 : other_percent = 0.2)
  (h5 : clothing_discount = 0.1)
  (h6 : electronics_discount = 0.05)
  (h7 : clothing_tax = 0.04)
  (h8 : food_tax = 0)
  (h9 : electronics_tax = 0.06)
  (h10 : other_tax = 0.08)
  (h11 : total_amount > 0) :
  let clothing_amount := clothing_percent * total_amount
  let food_amount := food_percent * total_amount
  let electronics_amount := electronics_percent * total_amount
  let other_amount := other_percent * total_amount
  let discounted_clothing := clothing_amount * (1 - clothing_discount)
  let discounted_electronics := electronics_amount * (1 - electronics_discount)
  let total_tax := clothing_tax * discounted_clothing +
                   food_tax * food_amount +
                   electronics_tax * discounted_electronics +
                   other_tax * other_amount
  ∃ ε > 0, |total_tax / total_amount - 0.04465| < ε :=
by sorry

end NUMINAMATH_CALUDE_total_tax_percentage_l1591_159126


namespace NUMINAMATH_CALUDE_lcm_three_integers_l1591_159138

theorem lcm_three_integers (A₁ A₂ A₃ : ℤ) :
  let D := Int.gcd (A₁ * A₂) (Int.gcd (A₂ * A₃) (A₃ * A₁))
  Int.lcm A₁ (Int.lcm A₂ A₃) = (A₁ * A₂ * A₃) / D :=
by sorry

end NUMINAMATH_CALUDE_lcm_three_integers_l1591_159138


namespace NUMINAMATH_CALUDE_number_percentage_equality_l1591_159145

theorem number_percentage_equality (x : ℝ) :
  (40 / 100) * x = (30 / 100) * 50 → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l1591_159145


namespace NUMINAMATH_CALUDE_work_hours_calculation_l1591_159160

/-- Calculates the required weekly work hours given summer work details and additional earnings needed --/
def required_weekly_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (additional_earnings_needed : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := additional_earnings_needed / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that under given conditions, the required weekly work hours is 16 --/
theorem work_hours_calculation (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (additional_earnings_needed : ℕ) :
  summer_weeks = 10 →
  summer_hours_per_week = 40 →
  summer_earnings = 4000 →
  school_year_weeks = 50 →
  additional_earnings_needed = 8000 →
  required_weekly_hours summer_weeks summer_hours_per_week summer_earnings school_year_weeks additional_earnings_needed = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_work_hours_calculation_l1591_159160


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_squared_plus_one_l1591_159178

theorem point_in_fourth_quadrant_implies_a_squared_plus_one (a : ℤ) : 
  (3 * a - 9 > 0) → (2 * a - 10 < 0) → a^2 + 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_a_squared_plus_one_l1591_159178


namespace NUMINAMATH_CALUDE_janes_breakfast_problem_l1591_159186

/-- Represents the number of breakfast items bought -/
structure BreakfastItems where
  muffins : ℕ
  bagels : ℕ
  croissants : ℕ

/-- Calculates the total cost in cents -/
def totalCost (items : BreakfastItems) : ℕ :=
  50 * items.muffins + 75 * items.bagels + 65 * items.croissants

theorem janes_breakfast_problem :
  ∃ (items : BreakfastItems),
    items.muffins + items.bagels + items.croissants = 6 ∧
    items.bagels = 2 ∧
    (totalCost items) % 100 = 0 ∧
    items.muffins = 4 :=
  sorry

end NUMINAMATH_CALUDE_janes_breakfast_problem_l1591_159186


namespace NUMINAMATH_CALUDE_proportional_function_ratio_l1591_159117

/-- Proves that for a proportional function y = kx passing through the points (1, 3) and (a, b) where b ≠ 0, a/b = 1/3 -/
theorem proportional_function_ratio (k a b : ℝ) (h1 : b ≠ 0) (h2 : 3 = k * 1) (h3 : b = k * a) : a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_ratio_l1591_159117


namespace NUMINAMATH_CALUDE_order_of_powers_l1591_159179

theorem order_of_powers : 
  let a : ℕ := 2^55
  let b : ℕ := 3^44
  let c : ℕ := 5^33
  let d : ℕ := 6^22
  a < d ∧ d < b ∧ b < c :=
by sorry

end NUMINAMATH_CALUDE_order_of_powers_l1591_159179


namespace NUMINAMATH_CALUDE_carrie_weeks_to_buy_iphone_l1591_159161

def iphone_cost : ℕ := 1200
def trade_in_value : ℕ := 180
def weekly_earnings : ℕ := 50

def weeks_needed : ℕ :=
  (iphone_cost - trade_in_value + weekly_earnings - 1) / weekly_earnings

theorem carrie_weeks_to_buy_iphone :
  weeks_needed = 21 :=
sorry

end NUMINAMATH_CALUDE_carrie_weeks_to_buy_iphone_l1591_159161


namespace NUMINAMATH_CALUDE_range_of_sum_of_reciprocals_l1591_159141

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 4*y + 1/x + 1/y = 10) : 
  1 ≤ 1/x + 1/y ∧ 1/x + 1/y ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_of_reciprocals_l1591_159141


namespace NUMINAMATH_CALUDE_square_field_area_proof_l1591_159128

def square_field_area (wire_cost_per_meter : ℚ) (total_cost : ℚ) (gate_width : ℚ) (num_gates : ℕ) : ℚ :=
  let side_length := ((total_cost / wire_cost_per_meter + 2 * gate_width * num_gates) / 4 : ℚ)
  side_length * side_length

theorem square_field_area_proof (wire_cost_per_meter : ℚ) (total_cost : ℚ) (gate_width : ℚ) (num_gates : ℕ) :
  wire_cost_per_meter = 3/2 ∧ total_cost = 999 ∧ gate_width = 1 ∧ num_gates = 2 →
  square_field_area wire_cost_per_meter total_cost gate_width num_gates = 27889 := by
  sorry

#eval square_field_area (3/2) 999 1 2

end NUMINAMATH_CALUDE_square_field_area_proof_l1591_159128


namespace NUMINAMATH_CALUDE_no_intersection_l1591_159156

theorem no_intersection : ¬∃ x : ℝ, |3*x + 6| = -|4*x - 3| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l1591_159156


namespace NUMINAMATH_CALUDE_square_diff_cubed_l1591_159190

theorem square_diff_cubed : (7^2 - 5^2)^3 = 13824 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_cubed_l1591_159190


namespace NUMINAMATH_CALUDE_complex_in_third_quadrant_l1591_159123

def complex_number (x : ℝ) : ℂ := Complex.mk (x^2 - 6*x + 5) (x - 2)

def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

theorem complex_in_third_quadrant (x : ℝ) :
  in_third_quadrant (complex_number x) ↔ 1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_complex_in_third_quadrant_l1591_159123


namespace NUMINAMATH_CALUDE_square_fraction_count_l1591_159135

theorem square_fraction_count : 
  ∃! (s : Finset Int), 
    (∀ n ∈ s, ∃ k : Int, (n : ℚ) / (25 - n) = k^2) ∧ 
    (∀ n ∉ s, ¬∃ k : Int, (n : ℚ) / (25 - n) = k^2) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l1591_159135


namespace NUMINAMATH_CALUDE_soda_cost_per_ounce_l1591_159118

/-- The cost of soda per ounce, given initial money, remaining money, and amount bought. -/
def cost_per_ounce (initial_money remaining_money amount_bought : ℚ) : ℚ :=
  (initial_money - remaining_money) / amount_bought

/-- Theorem stating that the cost per ounce is $0.25 under given conditions. -/
theorem soda_cost_per_ounce :
  cost_per_ounce 2 0.5 6 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_per_ounce_l1591_159118


namespace NUMINAMATH_CALUDE_max_people_satisfying_conditions_l1591_159159

/-- Represents a group of people and their relationships -/
structure PeopleGroup where
  n : ℕ
  knows : Fin n → Fin n → Prop
  knows_sym : ∀ i j, knows i j ↔ knows j i

/-- Any 3 people have at least 2 who know each other -/
def condition_a (g : PeopleGroup) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    g.knows i j ∨ g.knows j k ∨ g.knows i k

/-- Any 4 people have at least 2 who don't know each other -/
def condition_b (g : PeopleGroup) : Prop :=
  ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l →
    ¬g.knows i j ∨ ¬g.knows i k ∨ ¬g.knows i l ∨
    ¬g.knows j k ∨ ¬g.knows j l ∨ ¬g.knows k l

/-- The maximum number of people satisfying both conditions is 8 -/
theorem max_people_satisfying_conditions :
  (∃ g : PeopleGroup, g.n = 8 ∧ condition_a g ∧ condition_b g) ∧
  (∀ g : PeopleGroup, condition_a g ∧ condition_b g → g.n ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_people_satisfying_conditions_l1591_159159


namespace NUMINAMATH_CALUDE_triangle_side_range_l1591_159183

/-- Given a triangle ABC where c = √2 and a cos C = c sin A, 
    the length of side BC is in the range (√2, 2) -/
theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  ∃ (BC : ℝ), BC > Real.sqrt 2 ∧ BC < 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1591_159183


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1591_159176

theorem complex_equation_solution (z : ℂ) : 
  z + (1 + 2*I) = 10 - 3*I → z = 9 - 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1591_159176


namespace NUMINAMATH_CALUDE_differential_equation_solution_l1591_159144

open Real

theorem differential_equation_solution 
  (y : ℝ → ℝ) 
  (C₁ C₂ : ℝ) 
  (h : ∀ x, y x = (C₁ + C₂ * x) * exp (3 * x) + exp x - 8 * x^2 * exp (3 * x)) :
  ∀ x, (deriv^[2] y) x - 6 * (deriv y) x + 9 * y x = 4 * exp x - 16 * exp (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l1591_159144


namespace NUMINAMATH_CALUDE_race_overtake_equation_l1591_159181

/-- The time it takes for John to overtake Steve in a race --/
def overtake_time (initial_distance : ℝ) (john_initial_speed : ℝ) (john_acceleration : ℝ) (steve_speed : ℝ) (final_distance : ℝ) : ℝ → Prop :=
  λ t => 0.5 * john_acceleration * t^2 + john_initial_speed * t - steve_speed * t - initial_distance - final_distance = 0

theorem race_overtake_equation :
  let initial_distance : ℝ := 15
  let john_initial_speed : ℝ := 3.5
  let john_acceleration : ℝ := 0.25
  let steve_speed : ℝ := 3.8
  let final_distance : ℝ := 2
  ∃ t : ℝ, overtake_time initial_distance john_initial_speed john_acceleration steve_speed final_distance t :=
by sorry

end NUMINAMATH_CALUDE_race_overtake_equation_l1591_159181


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1591_159133

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (non_intersecting : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane
  (m n : Line) (α β : Plane)
  (different_lines : m ≠ n)
  (non_intersecting_planes : non_intersecting α β)
  (m_parallel_n : parallel m n)
  (n_perp_β : perpendicular n β) :
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1591_159133


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1591_159106

theorem sqrt_meaningful_range (a : ℝ) : (∃ (x : ℝ), x^2 = 2 + a) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1591_159106


namespace NUMINAMATH_CALUDE_triangle_property_l1591_159187

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.a / t.b * Real.cos t.C + t.c / (2 * t.b) = 1) 
  (h2 : t.A + t.B + t.C = π) 
  (h3 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0) 
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) :
  (t.A = π / 3) ∧ 
  (t.a = 1 → ∃ l : Real, l > 2 ∧ l ≤ 3 ∧ l = t.a + t.b + t.c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l1591_159187


namespace NUMINAMATH_CALUDE_complex_fraction_value_l1591_159180

theorem complex_fraction_value : 
  let i : ℂ := Complex.I
  (1 + Real.sqrt 3 * i)^2 / (Real.sqrt 3 * i - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l1591_159180


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1591_159110

theorem constant_term_binomial_expansion (n : ℕ) (A B : ℕ) : 
  A = (4 : ℝ) ^ n →
  B = 2 ^ n →
  A + B = 72 →
  ∃ (r : ℕ), r = 1 ∧ 3 * (Nat.choose n r) = 9 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1591_159110


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1591_159121

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x * f y / f (x * y)

/-- Theorem stating the possible forms of f -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x : ℝ, f x = 0) ∨ ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1591_159121


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1591_159185

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^12 - x^6 + 1 = (x^2 - 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1591_159185


namespace NUMINAMATH_CALUDE_fourth_to_third_l1591_159102

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem stating that if P(a,b) is in the fourth quadrant, 
    then Q(-a,b-1) is in the third quadrant -/
theorem fourth_to_third (a b : ℝ) :
  in_fourth_quadrant ⟨a, b⟩ → in_third_quadrant ⟨-a, b-1⟩ := by
  sorry


end NUMINAMATH_CALUDE_fourth_to_third_l1591_159102


namespace NUMINAMATH_CALUDE_parallelogram_height_l1591_159104

/-- Given a parallelogram with area 384 cm² and base 24 cm, its height is 16 cm -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 384 ∧ base = 24 ∧ area = base * height → height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1591_159104


namespace NUMINAMATH_CALUDE_range_of_a_l1591_159198

def A (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 = 0}
def B : Set ℝ := {1, 2}

theorem range_of_a (a : ℝ) : 
  (A a ∪ B = B) ↔ a ∈ Set.Icc (-2 : ℝ) 2 ∧ a ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1591_159198


namespace NUMINAMATH_CALUDE_equation_solutions_l1591_159155

theorem equation_solutions :
  (∃ x : ℝ, 2 * (2 - x) - 5 * (2 - x) = 9 ∧ x = 5) ∧
  (∃ x : ℝ, x / 3 - (3 * x - 1) / 6 = 1 ∧ x = -5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1591_159155


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1591_159115

/-- The circumference of the base of a right circular cone formed by gluing together
    the edges of a 180° sector cut from a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  (2 * π * r) / 2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1591_159115


namespace NUMINAMATH_CALUDE_salary_change_percentage_l1591_159174

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.3)
  let final_salary := decreased_salary * (1 + 0.3)
  (initial_salary - final_salary) / initial_salary * 100 = 9 := by
sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l1591_159174


namespace NUMINAMATH_CALUDE_trapezoid_division_l1591_159169

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  base_ratio : ℝ
  smaller_base : ℝ
  larger_base : ℝ
  height : ℝ
  area_eq : area = (smaller_base + larger_base) * height / 2
  base_ratio_eq : larger_base = base_ratio * smaller_base

/-- Represents the two smaller trapezoids formed by the median line -/
structure SmallerTrapezoids where
  top_area : ℝ
  bottom_area : ℝ

/-- The main theorem stating the areas of smaller trapezoids -/
theorem trapezoid_division (t : Trapezoid) 
  (h1 : t.area = 80)
  (h2 : t.base_ratio = 3) :
  ∃ (st : SmallerTrapezoids), st.top_area = 30 ∧ st.bottom_area = 50 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_division_l1591_159169


namespace NUMINAMATH_CALUDE_picnic_group_size_l1591_159158

theorem picnic_group_size (initial_avg : ℝ) (new_persons : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_persons = 12 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ n : ℕ, n * initial_avg + new_persons * new_avg = (n + new_persons) * final_avg ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_picnic_group_size_l1591_159158


namespace NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l1591_159165

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if A(a, 2) and B(-3, b) are symmetric with respect to the origin, then a + b = 1 -/
theorem sum_of_coordinates_symmetric_points (a b : ℝ) 
  (h : symmetric_wrt_origin a 2 (-3) b) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_symmetric_points_l1591_159165


namespace NUMINAMATH_CALUDE_merchant_profit_l1591_159167

theorem merchant_profit (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 50 →
  discount_percentage = 20 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by sorry

end NUMINAMATH_CALUDE_merchant_profit_l1591_159167


namespace NUMINAMATH_CALUDE_units_digit_of_4659_to_157_l1591_159189

theorem units_digit_of_4659_to_157 :
  (4659^157) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_4659_to_157_l1591_159189


namespace NUMINAMATH_CALUDE_local_minimum_at_negative_one_l1591_159163

open Real

/-- The function f(x) = xe^x has a local minimum at x = -1 -/
theorem local_minimum_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = x * exp x) :
  IsLocalMin f (-1) :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_negative_one_l1591_159163


namespace NUMINAMATH_CALUDE_tan_function_property_l1591_159107

/-- Given a function y = a * tan(b * x) where a and b are positive constants,
    if the function passes through (π/4, 3) and has a period of 3π/2,
    then a * b = 2 * √3 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * Real.tan (b * (π / 4)) = 3) →
  (π / b = 3 * π / 2) →
  a * b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l1591_159107


namespace NUMINAMATH_CALUDE_cos_negative_75_degrees_l1591_159151

theorem cos_negative_75_degrees :
  Real.cos (-(75 * π / 180)) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_75_degrees_l1591_159151


namespace NUMINAMATH_CALUDE_race_lead_calculation_l1591_159113

theorem race_lead_calculation (total_length max_remaining : ℕ) 
  (initial_together first_lead second_lead : ℕ) : 
  total_length = 5000 →
  max_remaining = 3890 →
  initial_together = 200 →
  first_lead = 300 →
  second_lead = 170 →
  (total_length - max_remaining - initial_together) - (first_lead - second_lead) = 780 :=
by sorry

end NUMINAMATH_CALUDE_race_lead_calculation_l1591_159113


namespace NUMINAMATH_CALUDE_proposition_relationship_l1591_159177

theorem proposition_relationship (a b : ℝ) : 
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧ 
  (∃ a b : ℝ, a > 0 ∧ ¬(a > b ∧ a⁻¹ > b⁻¹)) := by
sorry

end NUMINAMATH_CALUDE_proposition_relationship_l1591_159177


namespace NUMINAMATH_CALUDE_eggs_remaining_l1591_159148

def dozen : ℕ := 12

def initial_eggs (num_dozens : ℕ) : ℕ := num_dozens * dozen

def remaining_after_half (total : ℕ) : ℕ := total / 2

def final_eggs (after_half : ℕ) (broken : ℕ) : ℕ := after_half - broken

theorem eggs_remaining (num_dozens : ℕ) (broken : ℕ) 
  (h1 : num_dozens = 6) 
  (h2 : broken = 15) : 
  final_eggs (remaining_after_half (initial_eggs num_dozens)) broken = 21 := by
  sorry

#check eggs_remaining

end NUMINAMATH_CALUDE_eggs_remaining_l1591_159148


namespace NUMINAMATH_CALUDE_student_mistake_difference_l1591_159193

theorem student_mistake_difference (number : ℕ) (h : number = 192) : 
  (5 / 6 : ℚ) * number - (5 / 16 : ℚ) * number = 100 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l1591_159193


namespace NUMINAMATH_CALUDE_max_value_of_b_plus_c_l1591_159112

/-- A cubic function f(x) = x³ + bx² + cx + d -/
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The derivative of f(x) -/
def f_deriv (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

/-- f(x) is decreasing on the interval [-2, 2] -/
def is_decreasing_on_interval (b c d : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 2, f_deriv b c x ≤ 0

theorem max_value_of_b_plus_c (b c d : ℝ) 
  (h : is_decreasing_on_interval b c d) : 
  b + c ≤ -12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_b_plus_c_l1591_159112


namespace NUMINAMATH_CALUDE_john_relatives_money_l1591_159196

theorem john_relatives_money (grandpa : ℕ) : 
  grandpa = 30 → 
  (grandpa + 3 * grandpa + 2 * grandpa + (3 * grandpa) / 2 : ℕ) = 225 := by
  sorry

end NUMINAMATH_CALUDE_john_relatives_money_l1591_159196


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1591_159191

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 221) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1591_159191


namespace NUMINAMATH_CALUDE_second_exam_score_l1591_159199

theorem second_exam_score (total_marks : ℕ) (num_exams : ℕ) (first_exam_percent : ℚ) 
  (third_exam_marks : ℕ) (overall_average_percent : ℚ) :
  total_marks = 500 →
  num_exams = 3 →
  first_exam_percent = 45 / 100 →
  third_exam_marks = 100 →
  overall_average_percent = 40 / 100 →
  (first_exam_percent * total_marks + (55 / 100) * total_marks + third_exam_marks) / 
    (num_exams * total_marks) = overall_average_percent :=
by sorry

end NUMINAMATH_CALUDE_second_exam_score_l1591_159199


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_shaded_area_l1591_159111

theorem isosceles_right_triangle_shaded_area (leg_length : ℝ) (total_partitions : ℕ) (shaded_partitions : ℕ) : 
  leg_length = 12 →
  total_partitions = 36 →
  shaded_partitions = 15 →
  (shaded_partitions : ℝ) * (leg_length^2 / (2 * total_partitions : ℝ)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_shaded_area_l1591_159111


namespace NUMINAMATH_CALUDE_intersection_sum_l1591_159140

/-- Given two lines y = 2x + c and y = 4x + d intersecting at (3, 11), prove that c + d = 4 -/
theorem intersection_sum (c d : ℝ) 
  (h1 : 11 = 2 * 3 + c) 
  (h2 : 11 = 4 * 3 + d) : 
  c + d = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_sum_l1591_159140


namespace NUMINAMATH_CALUDE_point_not_in_region_l1591_159131

def plane_region (x y : ℝ) : Prop := 3*x + 2*y > 3

theorem point_not_in_region :
  ¬(plane_region 0 0) ∧
  (plane_region 1 1) ∧
  (plane_region 0 2) ∧
  (plane_region 2 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l1591_159131


namespace NUMINAMATH_CALUDE_ball_count_after_50_moves_l1591_159142

/-- Represents the state of the boxes --/
structure BoxState :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Performs one iteration of the ball-moving process --/
def moveOnce (state : BoxState) : BoxState :=
  sorry

/-- Performs n iterations of the ball-moving process --/
def moveNTimes (n : ℕ) (state : BoxState) : BoxState :=
  sorry

/-- The initial state of the boxes --/
def initialState : BoxState :=
  { A := 8, B := 6, C := 3, D := 1 }

theorem ball_count_after_50_moves :
  (moveNTimes 50 initialState).A = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_after_50_moves_l1591_159142


namespace NUMINAMATH_CALUDE_john_profit_l1591_159182

/-- Calculate the selling price given the cost price and profit percentage -/
def selling_price (cost : ℚ) (profit_percent : ℚ) : ℚ :=
  cost * (1 + profit_percent / 100)

/-- Calculate the overall profit given the cost and selling prices of two items -/
def overall_profit (cost1 cost2 sell1 sell2 : ℚ) : ℚ :=
  (sell1 + sell2) - (cost1 + cost2)

theorem john_profit :
  let grinder_cost : ℚ := 15000
  let mobile_cost : ℚ := 10000
  let grinder_loss_percent : ℚ := 4
  let mobile_profit_percent : ℚ := 10
  let grinder_sell := selling_price grinder_cost (-grinder_loss_percent)
  let mobile_sell := selling_price mobile_cost mobile_profit_percent
  overall_profit grinder_cost mobile_cost grinder_sell mobile_sell = 400 := by
  sorry

end NUMINAMATH_CALUDE_john_profit_l1591_159182


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1591_159172

theorem consecutive_integers_sum (a b c d : ℝ) : 
  (b = a + 1 ∧ c = b + 1 ∧ d = c + 1) →  -- consecutive integers condition
  (a + d = 180) →                        -- sum of first and fourth is 180
  b = 90.5 :=                            -- second integer is 90.5
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1591_159172


namespace NUMINAMATH_CALUDE_middleton_marching_band_max_members_l1591_159109

theorem middleton_marching_band_max_members :
  ∀ n : ℕ,
  (30 * n % 21 = 9) →
  (30 * n < 1500) →
  (∀ m : ℕ, (30 * m % 21 = 9) → (30 * m < 1500) → (30 * m ≤ 30 * n)) →
  30 * n = 1470 :=
by sorry

end NUMINAMATH_CALUDE_middleton_marching_band_max_members_l1591_159109


namespace NUMINAMATH_CALUDE_fraction_comparison_l1591_159184

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1591_159184


namespace NUMINAMATH_CALUDE_lorenzo_stamps_l1591_159101

def stamps_needed (current : ℕ) (row_size : ℕ) : ℕ :=
  (row_size - (current % row_size)) % row_size

theorem lorenzo_stamps : stamps_needed 37 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lorenzo_stamps_l1591_159101


namespace NUMINAMATH_CALUDE_special_dog_food_ounces_per_pound_l1591_159139

/-- Represents the number of ounces in a pound of special dog food -/
def ounces_per_pound : ℕ := 16

/-- Represents the number of days in a year -/
def days_in_year : ℕ := 365

/-- Represents the number of days the puppy eats 2 ounces per day -/
def initial_feeding_days : ℕ := 60

/-- Represents the number of ounces the puppy eats per day during the initial feeding period -/
def initial_feeding_ounces : ℕ := 2

/-- Represents the number of ounces the puppy eats per day after the initial feeding period -/
def later_feeding_ounces : ℕ := 4

/-- Represents the number of pounds in each bag of special dog food -/
def pounds_per_bag : ℕ := 5

/-- Represents the number of bags the family needs to buy -/
def bags_needed : ℕ := 17

theorem special_dog_food_ounces_per_pound :
  ounces_per_pound = 16 :=
by sorry

end NUMINAMATH_CALUDE_special_dog_food_ounces_per_pound_l1591_159139


namespace NUMINAMATH_CALUDE_compound_weight_proof_l1591_159125

/-- Atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Number of Aluminum atoms in the compound -/
def num_Al : ℕ := 1

/-- Number of Bromine atoms in the compound -/
def num_Br : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 267

/-- Theorem stating that the molecular weight of the compound is approximately 267 g/mol -/
theorem compound_weight_proof :
  ∃ ε > 0, abs (molecular_weight - (num_Al * atomic_weight_Al + num_Br * atomic_weight_Br)) < ε :=
sorry

end NUMINAMATH_CALUDE_compound_weight_proof_l1591_159125


namespace NUMINAMATH_CALUDE_f_positive_range_f_always_negative_l1591_159154

def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |2*x - a|

theorem f_positive_range (x : ℝ) : 
  f 3 x > 0 ↔ 1 < x ∧ x < 5/3 := by sorry

theorem f_always_negative (a : ℝ) : 
  (∀ x < 2, f a x < 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_f_positive_range_f_always_negative_l1591_159154


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1591_159173

/-- Given a square with side length 40 cm that is cut into 5 identical rectangles,
    the length of the shorter side of each rectangle that maximizes its area is 8 cm. -/
theorem rectangle_max_area (square_side : ℝ) (num_rectangles : ℕ) 
  (h1 : square_side = 40)
  (h2 : num_rectangles = 5) :
  let rectangle_area := square_side^2 / num_rectangles
  let shorter_side := square_side / num_rectangles
  shorter_side = 8 ∧ 
  ∀ (w : ℝ), w > 0 → w * (square_side^2 / (num_rectangles * w)) ≤ rectangle_area :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1591_159173


namespace NUMINAMATH_CALUDE_circle_on_grid_regions_l1591_159146

/-- Represents a grid with uniform spacing -/
structure Grid :=
  (spacing : ℝ)

/-- Represents a circle on the grid -/
structure CircleOnGrid :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents a region formed by circle arcs and grid line segments -/
structure Region

/-- Calculates the number of regions formed by a circle on a grid -/
def count_regions (g : Grid) (c : CircleOnGrid) : ℕ :=
  sorry

/-- Calculates the areas of regions formed by a circle on a grid -/
def region_areas (g : Grid) (c : CircleOnGrid) : List ℝ :=
  sorry

/-- Main theorem: Number and areas of regions formed by a circle on a grid -/
theorem circle_on_grid_regions 
  (g : Grid) 
  (c : CircleOnGrid) 
  (h1 : g.spacing = 1) 
  (h2 : c.radius = 5) 
  (h3 : c.center = (0, 0)) :
  (count_regions g c = 56) ∧ 
  (region_areas g c ≈ [0.966, 0.761, 0.317, 0.547]) :=
by sorry

#check circle_on_grid_regions

end NUMINAMATH_CALUDE_circle_on_grid_regions_l1591_159146


namespace NUMINAMATH_CALUDE_y_intercept_after_translation_intersection_point_l1591_159168

/-- The y-intercept of a line after vertical translation -/
theorem y_intercept_after_translation (m b h : ℝ) :
  let original_line := fun x => m * x + b
  let translated_line := fun x => m * x + (b + h)
  (translated_line 0) = b + h :=
by
  sorry

/-- Proof that the translated line y = 2x - 1 + 3 intersects y-axis at (0, 2) -/
theorem intersection_point :
  let original_line := fun x => 2 * x - 1
  let translated_line := fun x => 2 * x - 1 + 3
  (translated_line 0) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_y_intercept_after_translation_intersection_point_l1591_159168


namespace NUMINAMATH_CALUDE_inscribed_square_area_largest_inscribed_square_area_l1591_159127

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

/-- The side length of the inscribed square -/
noncomputable def s : ℝ := -1 + Real.sqrt 3

/-- The area of the inscribed square -/
noncomputable def area : ℝ := (2*s)^2

theorem inscribed_square_area :
  ∀ (a : ℝ), a > 0 →
  (∀ (x : ℝ), x ∈ Set.Icc (3 - a/2) (3 + a/2) → f x ≥ 0) →
  (f (3 - a/2) = 0 ∨ f (3 + a/2) = 0) →
  a ≤ 2*s :=
by sorry

theorem largest_inscribed_square_area :
  area = 16 - 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_largest_inscribed_square_area_l1591_159127


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l1591_159114

/-- The trajectory of a point equidistant from a fixed point and a line is a parabola -/
theorem trajectory_is_parabola (M : ℝ × ℝ) :
  (∀ (x y : ℝ), M = (x, y) →
    dist M (0, -3) = |y - 3|) →
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 = -12*y :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l1591_159114


namespace NUMINAMATH_CALUDE_school_garbage_plan_l1591_159105

/-- Represents a purchasing plan for warm reminder signs and garbage bins -/
structure PurchasePlan where
  signs : ℕ
  bins : ℕ

/-- Calculates the total cost of a purchasing plan given the prices -/
def totalCost (plan : PurchasePlan) (signPrice binPrice : ℕ) : ℕ :=
  plan.signs * signPrice + plan.bins * binPrice

theorem school_garbage_plan :
  ∃ (signPrice binPrice : ℕ) (bestPlan : PurchasePlan),
    -- Conditions
    (2 * signPrice + 3 * binPrice = 550) ∧
    (binPrice = 3 * signPrice) ∧
    (bestPlan.signs + bestPlan.bins = 100) ∧
    (bestPlan.bins ≥ 48) ∧
    (totalCost bestPlan signPrice binPrice ≤ 10000) ∧
    -- Conclusions
    (signPrice = 50) ∧
    (binPrice = 150) ∧
    (bestPlan.signs = 52) ∧
    (bestPlan.bins = 48) ∧
    (totalCost bestPlan signPrice binPrice = 9800) ∧
    (∀ (plan : PurchasePlan),
      (plan.signs + plan.bins = 100) →
      (plan.bins ≥ 48) →
      (totalCost plan signPrice binPrice ≤ 10000) →
      (totalCost plan signPrice binPrice ≥ totalCost bestPlan signPrice binPrice)) :=
by
  sorry

end NUMINAMATH_CALUDE_school_garbage_plan_l1591_159105


namespace NUMINAMATH_CALUDE_shorts_cost_l1591_159147

def football_cost : ℝ := 3.75
def shoes_cost : ℝ := 11.85
def zachary_money : ℝ := 10
def additional_money_needed : ℝ := 8

theorem shorts_cost : 
  ∃ (shorts_price : ℝ), 
    football_cost + shoes_cost + shorts_price = zachary_money + additional_money_needed ∧ 
    shorts_price = 2.40 := by
sorry

end NUMINAMATH_CALUDE_shorts_cost_l1591_159147


namespace NUMINAMATH_CALUDE_no_valid_division_l1591_159166

/-- The total weight of all stones -/
def total_weight : ℕ := (77 * 78) / 2

/-- The weight of the heaviest group for a given k -/
def heaviest_group_weight (k : ℕ) : ℕ := 
  (total_weight + k - 1) / k

/-- The number of stones in the heaviest group for a given k -/
def stones_in_heaviest_group (k : ℕ) : ℕ := 
  (heaviest_group_weight k + 76) / 77

/-- The total number of stones in all groups for a given k -/
def total_stones (k : ℕ) : ℕ := 
  k * (stones_in_heaviest_group k + (k - 1) / 2)

/-- The set of possible values for k -/
def possible_k : Finset ℕ := {9, 10, 11, 12}

theorem no_valid_division : 
  ∀ k ∈ possible_k, total_stones k > 77 := by sorry

end NUMINAMATH_CALUDE_no_valid_division_l1591_159166


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1591_159116

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the set M
def M : Set Nat := {1, 3, 5}

-- Theorem stating that the complement of M in U is {2, 4, 6}
theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1591_159116


namespace NUMINAMATH_CALUDE_two_digit_numbers_product_5681_sum_154_l1591_159129

theorem two_digit_numbers_product_5681_sum_154 : 
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5681 ∧ a + b = 154 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_product_5681_sum_154_l1591_159129


namespace NUMINAMATH_CALUDE_house_transactions_result_l1591_159120

/-- Represents the state of cash and house ownership for both Mr. A and Mr. B -/
structure State where
  a_cash : Int
  b_cash : Int
  a_has_house : Bool

/-- Represents a transaction between Mr. A and Mr. B -/
inductive Transaction
  | sell_to_b (price : Int)
  | buy_from_b (price : Int)

def initial_state : State := {
  a_cash := 12000,
  b_cash := 13000,
  a_has_house := true
}

def apply_transaction (s : State) (t : Transaction) : State :=
  match t with
  | Transaction.sell_to_b price =>
      { a_cash := s.a_cash + price,
        b_cash := s.b_cash - price,
        a_has_house := false }
  | Transaction.buy_from_b price =>
      { a_cash := s.a_cash - price,
        b_cash := s.b_cash + price,
        a_has_house := true }

def transactions : List Transaction := [
  Transaction.sell_to_b 14000,
  Transaction.buy_from_b 11000,
  Transaction.sell_to_b 15000
]

def final_state : State :=
  transactions.foldl apply_transaction initial_state

theorem house_transactions_result :
  final_state.a_cash = 30000 ∧ final_state.b_cash = -5000 := by
  sorry

end NUMINAMATH_CALUDE_house_transactions_result_l1591_159120


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1591_159192

theorem complex_equation_solution (a b c : ℂ) 
  (eq : 3*a + 4*b + 5*c = 0) 
  (norm_a : Complex.abs a = 1) 
  (norm_b : Complex.abs b = 1) 
  (norm_c : Complex.abs c = 1) : 
  a * (b + c) = -3/5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1591_159192


namespace NUMINAMATH_CALUDE_circle_inequality_m_range_l1591_159136

theorem circle_inequality_m_range :
  ∀ m : ℝ,
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 1 → x + y + m ≥ 0) ↔
  m > -1 :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_m_range_l1591_159136


namespace NUMINAMATH_CALUDE_fifth_graders_count_l1591_159164

def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def num_buses : ℕ := 5
def seats_per_bus : ℕ := 72

def total_seats : ℕ := num_buses * seats_per_bus
def total_chaperones : ℕ := (teachers_per_grade + parents_per_grade) * 3
def sixth_and_seventh : ℕ := sixth_graders + seventh_graders
def seats_taken : ℕ := sixth_and_seventh + total_chaperones

theorem fifth_graders_count : 
  total_seats - seats_taken = 109 := by sorry

end NUMINAMATH_CALUDE_fifth_graders_count_l1591_159164


namespace NUMINAMATH_CALUDE_double_age_in_four_years_l1591_159100

/-- The number of years until Fouad's age is double Ahmed's age -/
def years_until_double_age (ahmed_age : ℕ) (fouad_age : ℕ) : ℕ :=
  fouad_age - ahmed_age

theorem double_age_in_four_years (ahmed_age : ℕ) (fouad_age : ℕ) 
  (h1 : ahmed_age = 11) (h2 : fouad_age = 26) : 
  years_until_double_age ahmed_age fouad_age = 4 := by
  sorry

#check double_age_in_four_years

end NUMINAMATH_CALUDE_double_age_in_four_years_l1591_159100


namespace NUMINAMATH_CALUDE_evaluate_expression_l1591_159103

theorem evaluate_expression (c d : ℝ) (h : c^2 ≠ d^2) :
  (c^4 - d^4) / (2 * (c^2 - d^2)) = (c^2 + d^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1591_159103


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1591_159188

theorem modulus_of_complex_number (z : ℂ) : z = 2 / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1591_159188


namespace NUMINAMATH_CALUDE_roots_of_f_minus_x_and_f_of_f_minus_x_l1591_159137

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem roots_of_f_minus_x_and_f_of_f_minus_x :
  (∀ x : ℝ, f x - x = 0 ↔ x = 1 ∨ x = 2) ∧
  (∀ x : ℝ, f (f x) - x = 0 ↔ x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_f_minus_x_and_f_of_f_minus_x_l1591_159137


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1591_159195

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x > 0 → x > -2) ∧ 
  (∃ x : ℝ, x > -2 ∧ x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1591_159195
