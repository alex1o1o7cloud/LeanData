import Mathlib

namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2341_234173

theorem complex_expression_evaluation :
  (1 : ℝ) * (0.25 ^ (1/2 : ℝ)) - 
  (-2 * ((3/7 : ℝ) ^ (0 : ℝ))) ^ 2 * 
  ((-2 : ℝ) ^ 3) ^ (4/3 : ℝ) + 
  ((2 : ℝ) ^ (1/2 : ℝ) - 1) ^ (-1 : ℝ) - 
  (2 : ℝ) ^ (1/2 : ℝ) = -125/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2341_234173


namespace NUMINAMATH_CALUDE_smallest_base_for_78_l2341_234171

theorem smallest_base_for_78 :
  ∃ (b : ℕ), b > 0 ∧ b^2 ≤ 78 ∧ 78 < b^3 ∧ ∀ (x : ℕ), x > 0 ∧ x^2 ≤ 78 ∧ 78 < x^3 → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_78_l2341_234171


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l2341_234104

theorem grape_rate_calculation (grape_quantity : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : 
  grape_quantity = 8 →
  mango_quantity = 9 →
  mango_rate = 45 →
  total_paid = 965 →
  ∃ (grape_rate : ℕ), grape_rate * grape_quantity + mango_rate * mango_quantity = total_paid ∧ grape_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l2341_234104


namespace NUMINAMATH_CALUDE_georges_number_l2341_234141

theorem georges_number : ∃ (n : ℕ), n > 0 ∧ n ≤ 900 ∧ n % 4 ≠ 0 ∧
  (∃ (m : ℕ), m > 0 ∧ n = 3 * (3 * (3 * (3 * (3 * (3 * m - 1) - 1) - 1) - 1) - 1) - 1) ∧
  (∀ (k : ℕ), k > 0 ∧ k < n → k % 4 ≠ 0 →
    ¬∃ (l : ℕ), l > 0 ∧ k = 3 * (3 * (3 * (3 * (3 * (3 * l - 1) - 1) - 1) - 1) - 1) - 1) ∧
  n = 242 :=
by sorry

end NUMINAMATH_CALUDE_georges_number_l2341_234141


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2341_234106

theorem sqrt_equation_solution (a b : ℕ+) (h1 : a < b) :
  Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b →
  a = 2 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2341_234106


namespace NUMINAMATH_CALUDE_supermarket_max_profit_l2341_234131

/-- Represents the daily profit function for a supermarket selling daily necessities -/
def daily_profit (x : ℝ) : ℝ :=
  (200 - 10 * (x - 50)) * (x - 40)

/-- The maximum daily profit achievable by the supermarket -/
def max_daily_profit : ℝ := 2250

theorem supermarket_max_profit :
  ∃ (x : ℝ), daily_profit x = max_daily_profit ∧
  ∀ (y : ℝ), daily_profit y ≤ max_daily_profit :=
by sorry

end NUMINAMATH_CALUDE_supermarket_max_profit_l2341_234131


namespace NUMINAMATH_CALUDE_total_books_correct_l2341_234140

/-- Calculates the total number of books after buying more. -/
def totalBooks (initialBooks newBooks : ℕ) : ℕ :=
  initialBooks + newBooks

/-- Proves that the total number of books is the sum of initial and new books. -/
theorem total_books_correct (initialBooks newBooks : ℕ) :
  totalBooks initialBooks newBooks = initialBooks + newBooks := by
  sorry

/-- Verifies the specific case in the problem. -/
example : totalBooks 9 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_books_correct_l2341_234140


namespace NUMINAMATH_CALUDE_five_trip_ticket_cost_l2341_234108

/-- Represents the cost of tickets in gold coins -/
structure TicketCost where
  one : ℕ
  five : ℕ
  twenty : ℕ

/-- Conditions for the ticket costs -/
def valid_ticket_cost (t : TicketCost) : Prop :=
  5 * t.one > t.five ∧ 
  4 * t.five > t.twenty ∧
  t.twenty + 3 * t.five = 33 ∧
  20 + 3 * 5 = 35

theorem five_trip_ticket_cost (t : TicketCost) (h : valid_ticket_cost t) : t.five = 5 :=
sorry

end NUMINAMATH_CALUDE_five_trip_ticket_cost_l2341_234108


namespace NUMINAMATH_CALUDE_intersection_S_T_equals_T_l2341_234111

-- Define set S
def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}

-- Define set T
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

-- Theorem statement
theorem intersection_S_T_equals_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_equals_T_l2341_234111


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l2341_234148

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- Theorem: In a rhombus with one diagonal of 20 cm and an area of 250 cm², the other diagonal is 25 cm -/
theorem rhombus_other_diagonal
  (r : Rhombus)
  (h1 : r.diagonal1 = 20)
  (h2 : r.area = 250)
  (h3 : r.area = r.diagonal1 * r.diagonal2 / 2) :
  r.diagonal2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l2341_234148


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l2341_234100

noncomputable def y (k : ℝ) (x : ℝ) : ℝ :=
  if k ≠ 0 then
    1/2 * ((1/(1 - k*x))^(1/k) + (1 - k*x)^(1/k))
  else
    Real.cosh x

noncomputable def z (k : ℝ) (x : ℝ) : ℝ :=
  if k ≠ 0 then
    1/2 * ((1/(1 - k*x))^(1/k) - (1 - k*x)^(1/k))
  else
    Real.sinh x

theorem solution_satisfies_conditions (k : ℝ) :
  (∀ x, (deriv (y k)) x = (z k x) * ((y k x) + (z k x))^k) ∧
  (∀ x, (deriv (z k)) x = (y k x) * ((y k x) + (z k x))^k) ∧
  y k 0 = 1 ∧
  z k 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_conditions_l2341_234100


namespace NUMINAMATH_CALUDE_range_of_a_l2341_234152

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → 
  a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2341_234152


namespace NUMINAMATH_CALUDE_dogs_per_box_l2341_234196

theorem dogs_per_box (total_boxes : ℕ) (total_dogs : ℕ) (dogs_per_box : ℕ) :
  total_boxes = 7 →
  total_dogs = 28 →
  total_dogs = total_boxes * dogs_per_box →
  dogs_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_dogs_per_box_l2341_234196


namespace NUMINAMATH_CALUDE_triangle_gp_ratio_lt_two_l2341_234191

/-- Given a triangle with side lengths forming a geometric progression,
    prove that the common ratio of the progression is less than 2. -/
theorem triangle_gp_ratio_lt_two (b q : ℝ) (hb : b > 0) (hq : q > 0) :
  (b + b*q > b*q^2) ∧ (b + b*q^2 > b*q) ∧ (b*q + b*q^2 > b) →
  q < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_gp_ratio_lt_two_l2341_234191


namespace NUMINAMATH_CALUDE_candice_spending_l2341_234122

def total_money : ℕ := 100
def mildred_spent : ℕ := 25
def money_left : ℕ := 40

theorem candice_spending : 
  total_money - mildred_spent - money_left = 35 := by
sorry

end NUMINAMATH_CALUDE_candice_spending_l2341_234122


namespace NUMINAMATH_CALUDE_nancy_bills_l2341_234199

/-- The number of 5-dollar bills Nancy has -/
def num_bills : ℕ := sorry

/-- The value of each bill in dollars -/
def bill_value : ℕ := 5

/-- The total amount of money Nancy has in dollars -/
def total_money : ℕ := 45

/-- Theorem stating that Nancy has 9 five-dollar bills -/
theorem nancy_bills : num_bills = 45 / 5 := by sorry

end NUMINAMATH_CALUDE_nancy_bills_l2341_234199


namespace NUMINAMATH_CALUDE_room_tiles_count_l2341_234120

/-- Calculates the total number of tiles needed for a room with given dimensions and tile specifications. -/
def total_tiles (room_length room_width border_width border_tile_size inner_tile_size : ℕ) : ℕ :=
  let border_tiles := 2 * (2 * (room_length - 2 * border_width) + 2 * (room_width - 2 * border_width)) - 8 * border_width
  let inner_area := (room_length - 2 * border_width) * (room_width - 2 * border_width)
  let inner_tiles := inner_area / (inner_tile_size * inner_tile_size)
  border_tiles + inner_tiles

/-- Theorem stating that for a 15-foot by 20-foot room with a double border of 1-foot tiles
    and the rest filled with 2-foot tiles, the total number of tiles used is 144. -/
theorem room_tiles_count :
  total_tiles 20 15 2 1 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_count_l2341_234120


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2341_234162

theorem quadratic_equation_solution (p q : ℤ) (h1 : p + q = 2010) 
  (h2 : ∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) :
  p = -2278 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2341_234162


namespace NUMINAMATH_CALUDE_distinct_arrangements_eq_factorial_l2341_234176

/-- The number of ways to arrange n distinct objects in n positions -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of boxes -/
def num_boxes : ℕ := 5

/-- The number of digits to place -/
def num_digits : ℕ := 4

/-- Theorem: The number of ways to arrange 4 distinct digits and 1 blank in 5 boxes
    is equal to 5! -/
theorem distinct_arrangements_eq_factorial :
  factorial num_boxes = 120 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_eq_factorial_l2341_234176


namespace NUMINAMATH_CALUDE_min_students_for_three_discussing_same_l2341_234134

/-- Represents a discussion between two students about a problem -/
structure Discussion where
  student1 : ℕ
  student2 : ℕ
  problem : Fin 3

/-- Represents a valid discussion configuration for n students -/
def ValidConfiguration (n : ℕ) (discussions : List Discussion) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    ∃! d : Discussion, d ∈ discussions ∧
      ((d.student1 = i.val ∧ d.student2 = j.val) ∨
       (d.student1 = j.val ∧ d.student2 = i.val))

/-- Checks if there are at least 3 students discussing the same problem -/
def HasThreeDiscussingSame (n : ℕ) (discussions : List Discussion) : Prop :=
  ∃ p : Fin 3, ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (∃ d1 d2 d3 : Discussion,
      d1 ∈ discussions ∧ d2 ∈ discussions ∧ d3 ∈ discussions ∧
      d1.problem = p ∧ d2.problem = p ∧ d3.problem = p ∧
      ((d1.student1 = i.val ∧ d1.student2 = j.val) ∨ (d1.student1 = j.val ∧ d1.student2 = i.val)) ∧
      ((d2.student1 = j.val ∧ d2.student2 = k.val) ∨ (d2.student1 = k.val ∧ d2.student2 = j.val)) ∧
      ((d3.student1 = i.val ∧ d3.student2 = k.val) ∨ (d3.student1 = k.val ∧ d3.student2 = i.val)))

theorem min_students_for_three_discussing_same :
  (∃ n : ℕ, ∀ discussions : List Discussion,
    ValidConfiguration n discussions → HasThreeDiscussingSame n discussions) ∧
  (∀ m : ℕ, m < 17 →
    ∃ discussions : List Discussion,
      ValidConfiguration m discussions ∧ ¬HasThreeDiscussingSame m discussions) :=
by sorry

end NUMINAMATH_CALUDE_min_students_for_three_discussing_same_l2341_234134


namespace NUMINAMATH_CALUDE_drink_cost_is_2_50_l2341_234116

/-- The cost of a meal and drink with tip, given the following conditions:
  * The meal costs $10
  * The tip is 20% of the total cost (meal + drink)
  * The total amount paid is $15 -/
def total_cost (drink_cost : ℝ) : ℝ :=
  10 + drink_cost + 0.2 * (10 + drink_cost)

/-- Proves that the cost of the drink is $2.50 given the conditions -/
theorem drink_cost_is_2_50 :
  ∃ (drink_cost : ℝ), total_cost drink_cost = 15 ∧ drink_cost = 2.5 := by
sorry

end NUMINAMATH_CALUDE_drink_cost_is_2_50_l2341_234116


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l2341_234175

/-- Theorem: For a parabola y² = ax (a > 0) with a point P(3/2, y₀) on it,
    if the distance from P to the focus is 2, then a = 2. -/
theorem parabola_focus_distance (a : ℝ) (y₀ : ℝ) :
  a > 0 →
  y₀^2 = a * (3/2) →
  2 = (|3/2 - a/4| + |y₀|) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l2341_234175


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l2341_234174

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 2 → ℝ := ![3, -8]
  let v₂ : Fin 2 → ℝ := ![4, 6]
  let scalar : ℝ := -5
  let result : Fin 2 → ℝ := ![23, 22]
  v₁ - scalar • v₂ = result := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l2341_234174


namespace NUMINAMATH_CALUDE_circle_area_equals_square_perimeter_l2341_234125

theorem circle_area_equals_square_perimeter (side_length : ℝ) (radius : ℝ) : 
  side_length = 25 → 4 * side_length = Real.pi * radius^2 → Real.pi * radius^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equals_square_perimeter_l2341_234125


namespace NUMINAMATH_CALUDE_range_of_a_l2341_234145

open Real

noncomputable def f (a x : ℝ) : ℝ := x - (a + 1) * log x

noncomputable def g (a x : ℝ) : ℝ := a / x - 3

noncomputable def h (a x : ℝ) : ℝ := f a x - g a x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ g a x) →
  a ∈ Set.Iic (exp 1 * (exp 1 + 2) / (exp 1 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2341_234145


namespace NUMINAMATH_CALUDE_toy_cost_calculation_l2341_234170

theorem toy_cost_calculation (initial_amount : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_amount = 83 →
  game_cost = 47 →
  num_toys = 9 →
  (initial_amount - game_cost) % num_toys = 0 →
  (initial_amount - game_cost) / num_toys = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_cost_calculation_l2341_234170


namespace NUMINAMATH_CALUDE_domain_of_g_l2341_234198

-- Define the function f with domain [-2, 4]
def f : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 4 }

-- Define the function g as g(x) = f(x) + f(-x)
def g (x : ℝ) : Prop := x ∈ f ∧ (-x) ∈ f

-- Theorem stating that the domain of g is [-2, 2]
theorem domain_of_g : { x : ℝ | g x } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end NUMINAMATH_CALUDE_domain_of_g_l2341_234198


namespace NUMINAMATH_CALUDE_total_cows_l2341_234186

theorem total_cows (cows_per_herd : ℕ) (num_herds : ℕ) 
  (h1 : cows_per_herd = 40) 
  (h2 : num_herds = 8) : 
  cows_per_herd * num_herds = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_cows_l2341_234186


namespace NUMINAMATH_CALUDE_magnitude_2a_equals_6_l2341_234101

def a : Fin 3 → ℝ := ![-1, 2, 2]

theorem magnitude_2a_equals_6 : ‖(2 : ℝ) • a‖ = 6 := by sorry

end NUMINAMATH_CALUDE_magnitude_2a_equals_6_l2341_234101


namespace NUMINAMATH_CALUDE_unique_solution_equation_l2341_234129

theorem unique_solution_equation (b : ℝ) : 
  (b + ⌈b⌉ = 21.6) ∧ (b - ⌊b⌋ = 0.6) → b = 10.6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l2341_234129


namespace NUMINAMATH_CALUDE_binomial_distribution_p_value_l2341_234102

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (b : BinomialDistribution) : ℝ := b.n * b.p

/-- The variance of a binomial distribution -/
def variance (b : BinomialDistribution) : ℝ := b.n * b.p * (1 - b.p)

theorem binomial_distribution_p_value 
  (b : BinomialDistribution) 
  (h_exp : expectation b = 300)
  (h_var : variance b = 200) : 
  b.p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_binomial_distribution_p_value_l2341_234102


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2341_234119

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    prove that the eccentricity of the corresponding hyperbola is sqrt(5)/2 -/
theorem ellipse_hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 3/4) : 
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2341_234119


namespace NUMINAMATH_CALUDE_inequality_proof_l2341_234172

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_inequality : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2341_234172


namespace NUMINAMATH_CALUDE_trig_simplification_l2341_234130

open Real

theorem trig_simplification (α : ℝ) :
  (tan (π/4 - α) / (1 - tan (π/4 - α)^2)) * ((sin α * cos α) / (cos α^2 - sin α^2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2341_234130


namespace NUMINAMATH_CALUDE_horner_method_result_l2341_234194

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Theorem statement
theorem horner_method_result : f (-2) = 325.4 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_result_l2341_234194


namespace NUMINAMATH_CALUDE_sum_of_digits_2000_l2341_234180

/-- The number of digits in a positive integer n -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the number of digits in 2^2000 and 5^2000 is 2001 -/
theorem sum_of_digits_2000 : num_digits (2^2000) + num_digits (5^2000) = 2001 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_2000_l2341_234180


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_l2341_234181

theorem absolute_value_of_negative (a : ℝ) (h : a < 0) : |a| = -a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_l2341_234181


namespace NUMINAMATH_CALUDE_system_solution_l2341_234160

theorem system_solution (x y k : ℝ) : 
  (x + 2*y = 7 + k) → 
  (5*x - y = k) → 
  (y = -x) → 
  (k = -6) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2341_234160


namespace NUMINAMATH_CALUDE_stationary_tank_radius_l2341_234154

theorem stationary_tank_radius 
  (h : Real) 
  (r : Real) 
  (h_truck : Real) 
  (r_truck : Real) 
  (h_drop : Real) :
  h = 25 → 
  r_truck = 4 → 
  h_truck = 10 → 
  h_drop = 0.016 → 
  π * r^2 * h_drop = π * r_truck^2 * h_truck → 
  r = 100 := by
sorry

end NUMINAMATH_CALUDE_stationary_tank_radius_l2341_234154


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l2341_234149

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs (z - 3 + 3 * Complex.I) = 3) :
  ∃ (min : ℝ), min = 59 ∧ ∀ (w : ℂ), Complex.abs (w - 3 + 3 * Complex.I) = 3 →
    Complex.abs (w - 2 - Complex.I) ^ 2 + Complex.abs (w - 6 + 2 * Complex.I) ^ 2 ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l2341_234149


namespace NUMINAMATH_CALUDE_first_day_pen_sales_l2341_234143

/-- Proves that given the conditions of pen sales over 13 days, 
    the number of pens sold on the first day is 96. -/
theorem first_day_pen_sales : ∀ (first_day_sales : ℕ),
  (first_day_sales + 12 * 44 = 13 * 48) →
  first_day_sales = 96 := by
  sorry

#check first_day_pen_sales

end NUMINAMATH_CALUDE_first_day_pen_sales_l2341_234143


namespace NUMINAMATH_CALUDE_base_6_sum_theorem_l2341_234123

/-- Represents a base-6 number with three digits --/
def Base6Number (a b c : Nat) : Nat :=
  a * 36 + b * 6 + c

/-- Checks if a number is a valid base-6 digit (1-5) --/
def IsValidBase6Digit (n : Nat) : Prop :=
  0 < n ∧ n < 6

/-- The main theorem --/
theorem base_6_sum_theorem (A B C : Nat) 
  (h1 : IsValidBase6Digit A)
  (h2 : IsValidBase6Digit B)
  (h3 : IsValidBase6Digit C)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : Base6Number A B C + Base6Number B C 0 = Base6Number A C A) :
  A + B + C = Base6Number 1 5 0 := by
  sorry

#check base_6_sum_theorem

end NUMINAMATH_CALUDE_base_6_sum_theorem_l2341_234123


namespace NUMINAMATH_CALUDE_count_special_integers_l2341_234163

def f (n : ℕ) : ℚ := (n^2 + n) / 2

def is_product_of_two_primes (q : ℚ) : Prop :=
  ∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ q = p1 * p2

theorem count_special_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, f n ≤ 1000 ∧ is_product_of_two_primes (f n)) ∧
                   (∀ n : ℕ, f n ≤ 1000 ∧ is_product_of_two_primes (f n) → n ∈ S) ∧
                   S.card = 5) :=
sorry

end NUMINAMATH_CALUDE_count_special_integers_l2341_234163


namespace NUMINAMATH_CALUDE_fraction_problem_l2341_234128

theorem fraction_problem : ∃ f : ℚ, f * 1 = (144 : ℚ) / 216 ∧ f = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2341_234128


namespace NUMINAMATH_CALUDE_awards_distribution_l2341_234121

/-- The number of ways to distribute n distinct awards to k students,
    where each student receives at least one award. -/
def distribute_awards (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n distinct items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem awards_distribution :
  distribute_awards 5 4 = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_awards_distribution_l2341_234121


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2341_234192

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I : ℂ)⁻¹ * (a + Complex.I) = 1 + b * Complex.I → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2341_234192


namespace NUMINAMATH_CALUDE_max_trucks_orchard_l2341_234112

def apples : ℕ := 170
def tangerines : ℕ := 268
def mangoes : ℕ := 120

def apples_leftover : ℕ := 8
def tangerines_short : ℕ := 2
def mangoes_leftover : ℕ := 12

theorem max_trucks_orchard : 
  let apples_distributed := apples - apples_leftover
  let tangerines_distributed := tangerines + tangerines_short
  let mangoes_distributed := mangoes - mangoes_leftover
  ∃ (n : ℕ), n > 0 ∧ 
    apples_distributed % n = 0 ∧ 
    tangerines_distributed % n = 0 ∧ 
    mangoes_distributed % n = 0 ∧
    ∀ (m : ℕ), m > n → 
      (apples_distributed % m = 0 ∧ 
       tangerines_distributed % m = 0 ∧ 
       mangoes_distributed % m = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_max_trucks_orchard_l2341_234112


namespace NUMINAMATH_CALUDE_eighteenth_term_of_equal_sum_sequence_l2341_234178

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. -/
def EqualSumSequence (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = sum

theorem eighteenth_term_of_equal_sum_sequence
  (a : ℕ → ℝ)
  (sum : ℝ)
  (h_equal_sum : EqualSumSequence a sum)
  (h_first_term : a 1 = 2)
  (h_sum : sum = 5) :
  a 18 = 3 := by
sorry

end NUMINAMATH_CALUDE_eighteenth_term_of_equal_sum_sequence_l2341_234178


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2341_234105

theorem cubic_equation_solutions :
  ∀ (x y z n : ℕ), x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 3) ∨
   (x = 1 ∧ y = 2 ∧ z = 3 ∧ n = 1) ∨
   (x = 1 ∧ y = 3 ∧ z = 2 ∧ n = 1) ∨
   (x = 2 ∧ y = 1 ∧ z = 3 ∧ n = 1) ∨
   (x = 2 ∧ y = 3 ∧ z = 1 ∧ n = 1) ∨
   (x = 3 ∧ y = 1 ∧ z = 2 ∧ n = 1) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2341_234105


namespace NUMINAMATH_CALUDE_place_letters_in_mailboxes_l2341_234150

theorem place_letters_in_mailboxes :
  let n_letters : ℕ := 3
  let n_mailboxes : ℕ := 5
  (n_letters > 0) → (n_mailboxes > 0) →
  (number_of_ways : ℕ := n_mailboxes ^ n_letters) →
  number_of_ways = 125 := by
  sorry

end NUMINAMATH_CALUDE_place_letters_in_mailboxes_l2341_234150


namespace NUMINAMATH_CALUDE_infinite_zeros_or_nines_in_difference_l2341_234190

/-- Represents an infinite decimal fraction -/
def InfiniteDecimalFraction := ℕ → Fin 10

/-- Given a set of 11 infinite decimal fractions, there exist two fractions
    whose difference has either infinite zeros or infinite nines -/
theorem infinite_zeros_or_nines_in_difference 
  (fractions : Fin 11 → InfiniteDecimalFraction) :
  ∃ i j : Fin 11, i ≠ j ∧ 
    (∀ k : ℕ, (fractions i k - fractions j k) % 10 = 0 ∨
              (fractions i k - fractions j k) % 10 = 9) :=
sorry

end NUMINAMATH_CALUDE_infinite_zeros_or_nines_in_difference_l2341_234190


namespace NUMINAMATH_CALUDE_john_vacation_expenses_l2341_234107

def octal_to_decimal (n : Nat) : Nat :=
  let digits := n.digits 8
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (8 ^ i)) 0

theorem john_vacation_expenses :
  octal_to_decimal 5555 - 1500 = 1425 := by
  sorry

end NUMINAMATH_CALUDE_john_vacation_expenses_l2341_234107


namespace NUMINAMATH_CALUDE_fraction_product_equals_seven_fifty_fourths_l2341_234197

theorem fraction_product_equals_seven_fifty_fourths : 
  (7 : ℚ) / 4 * 8 / 12 * 14 / 6 * 18 / 30 * 16 / 24 * 35 / 49 * 27 / 54 * 40 / 20 = 7 / 54 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_seven_fifty_fourths_l2341_234197


namespace NUMINAMATH_CALUDE_cookies_per_person_l2341_234135

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℕ) 
  (h1 : total_cookies = 420) (h2 : num_people = 14) :
  total_cookies / num_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l2341_234135


namespace NUMINAMATH_CALUDE_divisibility_property_l2341_234168

theorem divisibility_property (n p q : ℕ) : 
  n > 0 → 
  Prime p → 
  q ∣ ((n + 1)^p - n^p) → 
  p ∣ (q - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2341_234168


namespace NUMINAMATH_CALUDE_probability_one_and_three_faces_l2341_234156

/-- Represents a cube with side length 5, assembled from unit cubes -/
def LargeCube := Fin 5 → Fin 5 → Fin 5 → Bool

/-- The number of unit cubes in the large cube -/
def totalUnitCubes : ℕ := 125

/-- The number of unit cubes with exactly one painted face -/
def oneRedFaceCubes : ℕ := 26

/-- The number of unit cubes with exactly three painted faces -/
def threeRedFaceCubes : ℕ := 4

/-- The probability of selecting one cube with one red face and one with three red faces -/
def probabilityOneAndThree : ℚ := 52 / 3875

theorem probability_one_and_three_faces (cube : LargeCube) :
  probabilityOneAndThree = (oneRedFaceCubes * threeRedFaceCubes : ℚ) / (totalUnitCubes.choose 2) :=
sorry

end NUMINAMATH_CALUDE_probability_one_and_three_faces_l2341_234156


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l2341_234124

/-- Given a mixture of milk and water, proves that the initial ratio was 4:1 --/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) :
  total_volume = 45 →
  added_water = 3 →
  final_ratio = 3 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    initial_milk / (initial_water + added_water) = final_ratio ∧
    initial_milk / initial_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l2341_234124


namespace NUMINAMATH_CALUDE_sum_mod_seven_l2341_234183

theorem sum_mod_seven : (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l2341_234183


namespace NUMINAMATH_CALUDE_factors_of_1320_l2341_234185

theorem factors_of_1320 : Finset.card (Nat.divisors 1320) = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l2341_234185


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_point_l2341_234165

theorem exponential_function_passes_through_point
  (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_point_l2341_234165


namespace NUMINAMATH_CALUDE_pepper_remaining_l2341_234169

theorem pepper_remaining (initial_amount used_amount : ℝ) 
  (h1 : initial_amount = 0.25)
  (h2 : used_amount = 0.16) : 
  initial_amount - used_amount = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_pepper_remaining_l2341_234169


namespace NUMINAMATH_CALUDE_cube_root_fourteen_problem_l2341_234167

theorem cube_root_fourteen_problem (x y z : ℝ) 
  (eq1 : (x + y) / (1 + z) = (1 - z + z^2) / (x^2 - x*y + y^2))
  (eq2 : (x - y) / (3 - z) = (9 + 3*z + z^2) / (x^2 + x*y + y^2)) :
  x = (14 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fourteen_problem_l2341_234167


namespace NUMINAMATH_CALUDE_red_cars_count_l2341_234164

/-- Represents the car rental problem --/
structure CarRental where
  num_white_cars : ℕ
  white_car_cost : ℕ
  red_car_cost : ℕ
  rental_duration : ℕ
  total_earnings : ℕ

/-- Calculates the number of red cars given the rental information --/
def calculate_red_cars (rental : CarRental) : ℕ :=
  (rental.total_earnings - rental.num_white_cars * rental.white_car_cost * rental.rental_duration) /
  (rental.red_car_cost * rental.rental_duration)

/-- Theorem stating that the number of red cars is 3 --/
theorem red_cars_count (rental : CarRental)
  (h1 : rental.num_white_cars = 2)
  (h2 : rental.white_car_cost = 2)
  (h3 : rental.red_car_cost = 3)
  (h4 : rental.rental_duration = 180)
  (h5 : rental.total_earnings = 2340) :
  calculate_red_cars rental = 3 := by
  sorry

#eval calculate_red_cars { num_white_cars := 2, white_car_cost := 2, red_car_cost := 3, rental_duration := 180, total_earnings := 2340 }

end NUMINAMATH_CALUDE_red_cars_count_l2341_234164


namespace NUMINAMATH_CALUDE_typists_letters_problem_l2341_234177

theorem typists_letters_problem (typists_initial : ℕ) (letters_initial : ℕ) (time_initial : ℕ) 
  (typists_final : ℕ) (time_final : ℕ) :
  typists_initial = 20 →
  letters_initial = 44 →
  time_initial = 20 →
  typists_final = 30 →
  time_final = 60 →
  (typists_final : ℚ) * (letters_initial : ℚ) * (time_final : ℚ) / 
    ((typists_initial : ℚ) * (time_initial : ℚ)) = 198 := by
  sorry

end NUMINAMATH_CALUDE_typists_letters_problem_l2341_234177


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2341_234195

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of the quadratic equation 5x^2 - 2x - 7 is 144 -/
theorem quadratic_discriminant :
  discriminant 5 (-2) (-7) = 144 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2341_234195


namespace NUMINAMATH_CALUDE_power_mod_1111_l2341_234161

theorem power_mod_1111 : ∃ n : ℕ, 0 ≤ n ∧ n < 1111 ∧ 2^1110 ≡ n [ZMOD 1111] := by
  use 1024
  sorry

end NUMINAMATH_CALUDE_power_mod_1111_l2341_234161


namespace NUMINAMATH_CALUDE_four_students_same_group_probability_l2341_234144

/-- The number of students in the school -/
def total_students : ℕ := 720

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- The probability of four specific students being assigned to the same lunch group -/
def prob_four_students_same_group : ℚ := prob_assigned_to_group ^ 3

theorem four_students_same_group_probability :
  prob_four_students_same_group = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_four_students_same_group_probability_l2341_234144


namespace NUMINAMATH_CALUDE_marys_height_marys_final_height_l2341_234133

theorem marys_height (initial_height : ℝ) (sallys_new_height : ℝ) : ℝ :=
  let sallys_growth_factor : ℝ := 1.2
  let sallys_growth : ℝ := sallys_new_height - initial_height
  let marys_growth : ℝ := sallys_growth / 2
  initial_height + marys_growth

theorem marys_final_height : 
  ∀ (initial_height : ℝ),
    initial_height > 0 →
    marys_height initial_height 180 = 165 :=
by
  sorry

end NUMINAMATH_CALUDE_marys_height_marys_final_height_l2341_234133


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2341_234188

theorem quadratic_root_property (a : ℝ) : 
  2 * a^2 + 3 * a - 2022 = 0 → 2 - 6 * a - 4 * a^2 = -4042 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2341_234188


namespace NUMINAMATH_CALUDE_expression_value_l2341_234179

theorem expression_value : 
  (2015^3 - 3 * 2015^2 * 2016 + 3 * 2015 * 2016^2 - 2016^3 + 1) / (2015 * 2016) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2341_234179


namespace NUMINAMATH_CALUDE_function_inequality_l2341_234117

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * deriv f x > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2341_234117


namespace NUMINAMATH_CALUDE_set_intersection_complement_l2341_234138

/-- Given sets A and B, if the intersection of the complement of A and B equals B,
    then m is less than or equal to -11 or greater than or equal to 3. -/
theorem set_intersection_complement (m : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x < 3}
  let B : Set ℝ := {x | m < x ∧ x < m + 9}
  (Aᶜ ∩ B = B) → (m ≤ -11 ∨ m ≥ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_set_intersection_complement_l2341_234138


namespace NUMINAMATH_CALUDE_cos_sum_when_sin_product_one_l2341_234132

theorem cos_sum_when_sin_product_one (α β : Real) 
  (h : Real.sin α * Real.sin β = 1) : 
  Real.cos (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_when_sin_product_one_l2341_234132


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2341_234110

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) → 
  p = -173/15 ∧ q = -466/15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2341_234110


namespace NUMINAMATH_CALUDE_mutually_inscribed_pentagons_exist_l2341_234109

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a pentagon in a 2D plane -/
structure Pentagon where
  vertices : Fin 5 → Point

/-- Checks if a point lies on a line segment or its extension -/
def pointOnLineSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Checks if two pentagons are mutually inscribed -/
def areMutuallyInscribed (p1 p2 : Pentagon) : Prop :=
  ∀ (i : Fin 5), 
    (pointOnLineSegment (p1.vertices i) (p2.vertices i) (p2.vertices ((i + 1) % 5))) ∧
    (pointOnLineSegment (p2.vertices i) (p1.vertices i) (p1.vertices ((i + 1) % 5)))

/-- Theorem: For any given pentagon, there exists another pentagon mutually inscribed with it -/
theorem mutually_inscribed_pentagons_exist (p : Pentagon) : 
  ∃ (q : Pentagon), areMutuallyInscribed p q := by sorry

end NUMINAMATH_CALUDE_mutually_inscribed_pentagons_exist_l2341_234109


namespace NUMINAMATH_CALUDE_coffee_pastry_budget_l2341_234146

theorem coffee_pastry_budget (B : ℝ) (c p : ℝ) 
  (hc : c = (1/4) * (B - p)) 
  (hp : p = (1/10) * (B - c)) : 
  c + p = (4/13) * B := by
sorry

end NUMINAMATH_CALUDE_coffee_pastry_budget_l2341_234146


namespace NUMINAMATH_CALUDE_sum_of_odd_divisors_180_l2341_234189

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_odd_divisors_180 : sum_of_odd_divisors 180 = 78 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_divisors_180_l2341_234189


namespace NUMINAMATH_CALUDE_willam_land_percentage_l2341_234113

/-- Given that farm tax is levied on 40% of cultivated land, prove that Mr. Willam's
    taxable land is 12.5% of the village's total taxable land. -/
theorem willam_land_percentage (total_tax : ℝ) (willam_tax : ℝ)
    (h1 : total_tax = 3840)
    (h2 : willam_tax = 480) :
    willam_tax / total_tax * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_willam_land_percentage_l2341_234113


namespace NUMINAMATH_CALUDE_square_odd_implies_odd_l2341_234137

theorem square_odd_implies_odd (n : ℤ) : Odd (n^2) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_square_odd_implies_odd_l2341_234137


namespace NUMINAMATH_CALUDE_sum_of_abc_l2341_234142

theorem sum_of_abc (a b c : ℕ+) 
  (h1 : (a + b + c)^3 - a^3 - b^3 - c^3 = 150)
  (h2 : a < b) (h3 : b < c) : 
  a + b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l2341_234142


namespace NUMINAMATH_CALUDE_escalator_speed_l2341_234155

theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 180 →
  person_speed = 3 →
  time_taken = 10 →
  ∃ (escalator_speed : ℝ),
    escalator_speed = 15 ∧
    (person_speed + escalator_speed) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_l2341_234155


namespace NUMINAMATH_CALUDE_invitation_ways_l2341_234187

def number_of_teachers : ℕ := 10
def teachers_to_invite : ℕ := 6

def ways_to_invite (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem invitation_ways : 
  ways_to_invite number_of_teachers teachers_to_invite - 
  ways_to_invite (number_of_teachers - 2) (teachers_to_invite - 2) = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_invitation_ways_l2341_234187


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2341_234159

theorem simplify_trig_expression (α : Real) (h : 270 * π / 180 < α ∧ α < 360 * π / 180) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2 * α))) = -Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2341_234159


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2341_234103

theorem inequality_not_always_true
  (x y z : ℝ) (k : ℤ)
  (hx : x > 0)
  (hy : y > 0)
  (hxy : x > y)
  (hz : z ≠ 0)
  (hk : k ≠ 0) :
  ¬ ∀ (x y z : ℝ) (k : ℤ), x / z^k > y / z^k :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2341_234103


namespace NUMINAMATH_CALUDE_sum_digits_base_seven_999_l2341_234139

/-- Represents a number in base 7 as a list of digits (least significant digit first) -/
def BaseSevenRepresentation := List Nat

/-- Converts a natural number to its base 7 representation -/
def toBaseSeven (n : Nat) : BaseSevenRepresentation :=
  sorry

/-- Computes the sum of digits in a base 7 representation -/
def sumDigitsBaseSeven (rep : BaseSevenRepresentation) : Nat :=
  sorry

theorem sum_digits_base_seven_999 :
  sumDigitsBaseSeven (toBaseSeven 999) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base_seven_999_l2341_234139


namespace NUMINAMATH_CALUDE_division_and_subtraction_l2341_234182

theorem division_and_subtraction : (12 / (1/12)) - 5 = 139 := by
  sorry

end NUMINAMATH_CALUDE_division_and_subtraction_l2341_234182


namespace NUMINAMATH_CALUDE_tommys_pencils_l2341_234157

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total_items : ℕ
  num_pencils : ℕ
  num_pens : ℕ
  num_erasers : ℕ

/-- Theorem stating the number of pencils in Tommy's pencil case -/
theorem tommys_pencils (pc : PencilCase) 
  (h1 : pc.total_items = 13)
  (h2 : pc.num_pens = 2 * pc.num_pencils)
  (h3 : pc.num_erasers = 1)
  (h4 : pc.total_items = pc.num_pencils + pc.num_pens + pc.num_erasers) :
  pc.num_pencils = 4 := by
  sorry

end NUMINAMATH_CALUDE_tommys_pencils_l2341_234157


namespace NUMINAMATH_CALUDE_cos_four_pi_thirds_l2341_234114

theorem cos_four_pi_thirds : Real.cos (4 * Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_pi_thirds_l2341_234114


namespace NUMINAMATH_CALUDE_product_remainder_main_theorem_l2341_234115

theorem product_remainder (a b : Nat) : (a * b) % 9 = ((a % 9) * (b % 9)) % 9 := by sorry

theorem main_theorem : (98 * 102) % 9 = 3 := by
  -- The proof would go here, but we're only providing the statement
  sorry

end NUMINAMATH_CALUDE_product_remainder_main_theorem_l2341_234115


namespace NUMINAMATH_CALUDE_max_d_is_zero_l2341_234153

/-- Represents a 6-digit number of the form 6d6,33e -/
def SixDigitNumber (d e : Nat) : Nat :=
  606330 + d * 1000 + e

theorem max_d_is_zero :
  ∀ d e : Nat,
    d < 10 →
    e < 10 →
    SixDigitNumber d e % 33 = 0 →
    d ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_d_is_zero_l2341_234153


namespace NUMINAMATH_CALUDE_inequality_proof_l2341_234118

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / (a * b) + (b^2 + c^2) / (b * c) + (c^2 + a^2) / (c * a) ≥ 6 ∧
  (a + b) / 2 * (b + c) / 2 * (c + a) / 2 ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2341_234118


namespace NUMINAMATH_CALUDE_gabby_fruit_count_l2341_234193

/-- The number of fruits Gabby picked in total -/
def total_fruits (watermelons peaches plums : ℕ) : ℕ := watermelons + peaches + plums

/-- The number of watermelons Gabby got -/
def watermelons : ℕ := 1

/-- The number of peaches Gabby got -/
def peaches : ℕ := watermelons + 12

/-- The number of plums Gabby got -/
def plums : ℕ := peaches * 3

theorem gabby_fruit_count :
  total_fruits watermelons peaches plums = 53 := by
  sorry

end NUMINAMATH_CALUDE_gabby_fruit_count_l2341_234193


namespace NUMINAMATH_CALUDE_min_value_theorem_l2341_234166

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (a + 1) + 4 / (b + 1) ≥ 9 / 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2341_234166


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l2341_234158

/-- The determinant of a 2x2 matrix [[x, 4], [-3, y]] is xy + 12 -/
theorem det_2x2_matrix (x y : ℝ) : 
  Matrix.det !![x, 4; -3, y] = x * y + 12 := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l2341_234158


namespace NUMINAMATH_CALUDE_box_volume_increase_l2341_234136

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + h * l) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2341_234136


namespace NUMINAMATH_CALUDE_triangle_properties_l2341_234127

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def triangle (a b c : ℝ) := true

theorem triangle_properties (a b c : ℝ) (h : triangle a b c) 
  (h1 : a^2 + 11*b^2 = 2 * Real.sqrt 3 * a * b)
  (h2 : Real.sin c = 2 * Real.sqrt 3 * Real.sin b)
  (h3 : Real.cos b * a * c = Real.tan b) :
  Real.cos b = 1/2 ∧ (1/2 * a * c * Real.sin b = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2341_234127


namespace NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2341_234151

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- Common elements between the arithmetic and geometric progressions -/
def common_elements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmetic_progression n = geometric_progression k

/-- The sum of the first 10 common elements -/
def sum_of_common_elements : ℕ := 13981000

theorem sum_of_first_10_common_elements :
  sum_of_common_elements = 13981000 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2341_234151


namespace NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l2341_234147

theorem no_perfect_square_n_n_plus_one : ¬∃ (n : ℕ), n > 0 ∧ ∃ (k : ℕ), n * (n + 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_n_n_plus_one_l2341_234147


namespace NUMINAMATH_CALUDE_jeff_purchases_total_l2341_234184

def round_to_nearest_dollar (x : ℚ) : ℤ :=
  if x - ↑(⌊x⌋) < 1/2 then ⌊x⌋ else ⌈x⌉

theorem jeff_purchases_total :
  let purchase1 : ℚ := 245/100
  let purchase2 : ℚ := 375/100
  let purchase3 : ℚ := 856/100
  let discount : ℚ := 50/100
  round_to_nearest_dollar purchase1 +
  round_to_nearest_dollar purchase2 +
  round_to_nearest_dollar (purchase3 - discount) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jeff_purchases_total_l2341_234184


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2341_234126

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ  -- length of equal sides
  base : ℕ  -- length of the base
  is_isosceles : leg > base / 2

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

/-- Theorem: The minimum possible common perimeter of two noncongruent
    integer-sided isosceles triangles with the same area and a base ratio of 5:4 is 840 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 840 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 840) := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2341_234126
