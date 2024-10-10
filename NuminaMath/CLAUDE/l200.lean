import Mathlib

namespace figure_sides_l200_20098

/-- A figure with a perimeter of 49 cm and a side length of 7 cm has 7 sides. -/
theorem figure_sides (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 49) (h2 : side_length = 7) :
  perimeter / side_length = 7 := by
  sorry

end figure_sides_l200_20098


namespace inverse_of_exponential_function_l200_20041

noncomputable def f (x : ℝ) : ℝ := 3^x

theorem inverse_of_exponential_function (x : ℝ) (h : x > 0) : 
  f⁻¹ x = Real.log x / Real.log 3 := by
sorry

end inverse_of_exponential_function_l200_20041


namespace sum_of_a_and_b_l200_20061

theorem sum_of_a_and_b (a b : ℝ) : (a - 2)^2 + |b + 4| = 0 → a + b = -2 := by
  sorry

end sum_of_a_and_b_l200_20061


namespace boat_current_rate_l200_20047

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 9.6 km downstream in 24 minutes, the rate of the current is 4 km/hr. -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 9.6 →
  downstream_time = 24 / 60 →
  ∃ (current_rate : ℝ),
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end boat_current_rate_l200_20047


namespace isosceles_triangle_perimeter_l200_20083

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7cm, one side is 3cm
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 17 :=  -- Perimeter is 17cm
by
  sorry


end isosceles_triangle_perimeter_l200_20083


namespace square_rectangle_area_ratio_l200_20025

/-- Represents a square with side length s -/
structure Square (s : ℝ) where
  area : ℝ := s^2

/-- Represents a rectangle with width w and height h -/
structure Rectangle (w h : ℝ) where
  area : ℝ := w * h

/-- The theorem statement -/
theorem square_rectangle_area_ratio 
  (s : ℝ) 
  (w h : ℝ) 
  (square : Square s) 
  (rect : Rectangle w h) 
  (h1 : rect.area = 0.25 * square.area) 
  (h2 : w = 8 * h) : 
  square.area / rect.area = 4 := by
  sorry

end square_rectangle_area_ratio_l200_20025


namespace sum_of_coordinates_D_l200_20032

/-- Given a line segment CD with midpoint M(6,6) and endpoint C(2,10),
    prove that the sum of coordinates of the other endpoint D is 12. -/
theorem sum_of_coordinates_D (C D M : ℝ × ℝ) : 
  C = (2, 10) →
  M = (6, 6) →
  M.1 = (C.1 + D.1) / 2 →
  M.2 = (C.2 + D.2) / 2 →
  D.1 + D.2 = 12 := by
  sorry

end sum_of_coordinates_D_l200_20032


namespace food_drive_mark_cans_l200_20034

/-- Represents the number of cans brought by each person -/
structure Cans where
  rachel : ℕ
  jaydon : ℕ
  mark : ℕ
  sophie : ℕ

/-- Conditions for the food drive -/
def FoodDriveConditions (c : Cans) : Prop :=
  c.mark = 4 * c.jaydon ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  4 * c.sophie = 3 * c.jaydon ∧
  c.rachel ≥ 5 ∧ c.jaydon ≥ 5 ∧ c.mark ≥ 5 ∧ c.sophie ≥ 5 ∧
  Odd (c.rachel + c.jaydon + c.mark + c.sophie) ∧
  c.rachel + c.jaydon + c.mark + c.sophie ≥ 250

theorem food_drive_mark_cans (c : Cans) (h : FoodDriveConditions c) : c.mark = 148 := by
  sorry

end food_drive_mark_cans_l200_20034


namespace max_xy_value_l200_20084

theorem max_xy_value (x y : ℕ+) (h1 : 7 * x + 2 * y = 140) (h2 : x ≤ 15) : 
  x * y ≤ 350 ∧ ∃ (x₀ y₀ : ℕ+), 7 * x₀ + 2 * y₀ = 140 ∧ x₀ ≤ 15 ∧ x₀ * y₀ = 350 :=
by sorry

end max_xy_value_l200_20084


namespace parametric_line_position_vector_l200_20009

/-- A line in a plane parameterized by t -/
structure ParametricLine where
  a : ℝ × ℝ  -- Point on the line
  d : ℝ × ℝ  -- Direction vector

/-- The position vector on a parametric line at a given t -/
def position_vector (line : ParametricLine) (t : ℝ) : ℝ × ℝ :=
  (line.a.1 + t * line.d.1, line.a.2 + t * line.d.2)

theorem parametric_line_position_vector :
  ∀ (line : ParametricLine),
    position_vector line 5 = (4, -1) →
    position_vector line (-1) = (-2, 13) →
    position_vector line 8 = (7, -8/3) := by
  sorry

end parametric_line_position_vector_l200_20009


namespace jimmys_father_emails_l200_20071

/-- The number of emails Jimmy's father received per day before subscribing to the news channel -/
def initial_emails_per_day : ℕ := 20

/-- The number of additional emails per day after subscribing to the news channel -/
def additional_emails : ℕ := 5

/-- The total number of days in April -/
def total_days : ℕ := 30

/-- The day Jimmy's father subscribed to the news channel -/
def subscription_day : ℕ := 15

/-- The total number of emails Jimmy's father received in April -/
def total_emails : ℕ := 675

theorem jimmys_father_emails :
  initial_emails_per_day * subscription_day +
  (initial_emails_per_day + additional_emails) * (total_days - subscription_day) =
  total_emails :=
by
  sorry

#check jimmys_father_emails

end jimmys_father_emails_l200_20071


namespace factorial_difference_l200_20051

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: 6! - 4! = 696 -/
theorem factorial_difference : factorial 6 - factorial 4 = 696 := by
  sorry

end factorial_difference_l200_20051


namespace sin_45_degrees_l200_20033

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_45_degrees_l200_20033


namespace simplify_expression_l200_20040

theorem simplify_expression (t : ℝ) (h : t ≠ 0) : (t^5 * t^7) / t^3 = t^9 := by
  sorry

end simplify_expression_l200_20040


namespace pipe_fill_time_l200_20018

/-- Given pipes P, Q, and R that can fill a tank, this theorem proves the time it takes for pipe P to fill the tank. -/
theorem pipe_fill_time (fill_rate_Q : ℝ) (fill_rate_R : ℝ) (fill_rate_all : ℝ) 
  (hQ : fill_rate_Q = 1 / 9)
  (hR : fill_rate_R = 1 / 18)
  (hAll : fill_rate_all = 1 / 2)
  (h_sum : ∃ (fill_rate_P : ℝ), fill_rate_P + fill_rate_Q + fill_rate_R = fill_rate_all) :
  ∃ (fill_time_P : ℝ), fill_time_P = 3 := by
  sorry

end pipe_fill_time_l200_20018


namespace geometric_sequence_sixth_term_l200_20050

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_second : a 2 = 1)
  (h_relation : a 8 = a 6 + 2 * a 4) :
  a 6 = 4 := by
sorry

end geometric_sequence_sixth_term_l200_20050


namespace hyperbola_midpoint_existence_l200_20016

theorem hyperbola_midpoint_existence : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 - y₁^2/9 = 1) ∧
  (x₂^2 - y₂^2/9 = 1) ∧
  ((x₁ + x₂)/2 = -1) ∧
  ((y₁ + y₂)/2 = -4) := by
  sorry

end hyperbola_midpoint_existence_l200_20016


namespace interior_angles_sum_increase_l200_20043

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_sum_increase {n : ℕ} (h : sum_interior_angles n = 1800) :
  sum_interior_angles (n + 2) = 2160 := by
  sorry

end interior_angles_sum_increase_l200_20043


namespace range_of_m_l200_20056

/-- Proposition P: The equation x^2 + mx + 1 = 0 has two distinct negative roots -/
def P (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition Q: The equation 4x^2 + 4(m - 2)x + 1 = 0 has no real roots -/
def Q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0

/-- The range of real values for m satisfying the given conditions -/
def M : Set ℝ :=
  {m : ℝ | m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3}

theorem range_of_m :
  ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ∈ M :=
sorry

end range_of_m_l200_20056


namespace mark_chicken_nuggets_cost_l200_20046

-- Define the number of chicken nuggets Mark orders
def total_nuggets : ℕ := 100

-- Define the number of nuggets in a box
def nuggets_per_box : ℕ := 20

-- Define the cost of one box
def cost_per_box : ℕ := 4

-- Theorem to prove
theorem mark_chicken_nuggets_cost :
  (total_nuggets / nuggets_per_box) * cost_per_box = 20 := by
  sorry

end mark_chicken_nuggets_cost_l200_20046


namespace expression_simplification_l200_20059

theorem expression_simplification :
  (5^2010)^2 - (5^2008)^2 / (5^2009)^2 - (5^2007)^2 = 25 := by
sorry

end expression_simplification_l200_20059


namespace not_prime_n_squared_plus_75_l200_20057

theorem not_prime_n_squared_plus_75 (n : ℕ) (h : Nat.Prime n) : ¬ Nat.Prime (n^2 + 75) := by
  sorry

end not_prime_n_squared_plus_75_l200_20057


namespace motorcycle_sales_decrease_l200_20048

/-- Represents the pricing and sales of motorcycles before and after a price increase --/
structure MotorcycleSales where
  original_price : ℝ
  new_price : ℝ
  original_quantity : ℕ
  new_quantity : ℕ
  original_revenue : ℝ
  new_revenue : ℝ

/-- The theorem stating the decrease in motorcycle sales after the price increase --/
theorem motorcycle_sales_decrease (sales : MotorcycleSales) : 
  sales.new_price = sales.original_price + 1000 →
  sales.new_revenue = sales.original_revenue + 26000 →
  sales.new_revenue = 594000 →
  sales.new_quantity = 63 →
  sales.original_quantity - sales.new_quantity = 4 := by
  sorry

#check motorcycle_sales_decrease

end motorcycle_sales_decrease_l200_20048


namespace maruti_car_sales_decrease_l200_20090

theorem maruti_car_sales_decrease (initial_price initial_sales : ℝ) 
  (price_increase : ℝ) (revenue_increase : ℝ) (sales_decrease : ℝ) :
  price_increase = 0.3 →
  revenue_increase = 0.04 →
  (initial_price * (1 + price_increase)) * (initial_sales * (1 - sales_decrease)) = 
    initial_price * initial_sales * (1 + revenue_increase) →
  sales_decrease = 0.2 := by
sorry

end maruti_car_sales_decrease_l200_20090


namespace smallest_side_of_triangle_l200_20053

/-- Given a triangle with specific properties, prove its smallest side length -/
theorem smallest_side_of_triangle (S : ℝ) (p : ℝ) (d : ℝ) :
  S = 6 * Real.sqrt 6 →
  p = 18 →
  d = (2 * Real.sqrt 42) / 3 →
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = p ∧
    S = Real.sqrt (p/2 * (p/2 - a) * (p/2 - b) * (p/2 - c)) ∧
    d^2 = ((p/2 - b) * (p/2 - c) / (p/2))^2 + (S / p)^2 ∧
    min a (min b c) = 5 :=
by sorry

end smallest_side_of_triangle_l200_20053


namespace car_rental_problem_l200_20082

/-- Represents a car rental company -/
structure Company where
  totalCars : Nat
  baseRent : ℝ
  rentIncrease : ℝ
  maintenanceFee : ℝ → ℝ

/-- Calculates the profit for a company given the number of cars rented -/
def profit (c : Company) (rented : ℝ) : ℝ :=
  (c.baseRent + (c.totalCars - rented) * c.rentIncrease) * rented - c.maintenanceFee rented

/-- Company A as described in the problem -/
def companyA : Company :=
  { totalCars := 50
  , baseRent := 3000
  , rentIncrease := 50
  , maintenanceFee := λ x => 200 * x }

/-- Company B as described in the problem -/
def companyB : Company :=
  { totalCars := 50
  , baseRent := 3500
  , rentIncrease := 0
  , maintenanceFee := λ _ => 1850 }

theorem car_rental_problem :
  (profit companyA 10 = 48000) ∧
  (∃ x : ℝ, x = 37 ∧ profit companyA x = profit companyB x) ∧
  (∃ max : ℝ, max = 33150 ∧ 
    ∀ x : ℝ, 0 ≤ x ∧ x ≤ 50 → |profit companyA x - profit companyB x| ≤ max) ∧
  (∀ a : ℝ, 50 < a ∧ a < 150 ↔
    (let f := λ x => profit companyA x - a * x - profit companyB x
     ∃ max : ℝ, max = f 17 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 50 → f x ≤ max ∧ f x > 0)) := by
  sorry

end car_rental_problem_l200_20082


namespace max_min_product_l200_20023

theorem max_min_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 9 → 
  a * b + b * c + c * a = 27 → 
  ∀ m : ℝ, m = min (a * b) (min (b * c) (c * a)) → 
  m ≤ 6.75 ∧ ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 9 ∧ 
    a' * b' + b' * c' + c' * a' = 27 ∧ 
    min (a' * b') (min (b' * c') (c' * a')) = 6.75 := by
  sorry

end max_min_product_l200_20023


namespace probability_of_different_colors_l200_20060

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 3

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := 4

/-- The number of white marbles in the jar -/
def white_marbles : ℕ := 5

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := 3

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + green_marbles + white_marbles + blue_marbles

/-- The number of ways to choose 2 marbles from the total number of marbles -/
def total_ways : ℕ := total_marbles.choose 2

/-- The number of ways to choose 2 marbles of different colors -/
def different_color_ways : ℕ :=
  red_marbles * green_marbles +
  red_marbles * white_marbles +
  red_marbles * blue_marbles +
  green_marbles * white_marbles +
  green_marbles * blue_marbles +
  white_marbles * blue_marbles

/-- The probability of drawing two marbles of different colors -/
def probability : ℚ := different_color_ways / total_ways

theorem probability_of_different_colors :
  probability = 83 / 105 :=
sorry

end probability_of_different_colors_l200_20060


namespace min_sum_squares_l200_20019

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2 * a + 3 * b + 5 * c = 100) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 2 * x + 3 * y + 5 * z = 100 → 
  a^2 + b^2 + c^2 ≤ x^2 + y^2 + z^2 ∧ 
  a^2 + b^2 + c^2 = 5000 / 19 := by
  sorry

end min_sum_squares_l200_20019


namespace problem_solution_l200_20094

theorem problem_solution (x y : ℝ) (h1 : 15 * x = x + 280) (h2 : y = x^2 + 5*x - 12) :
  x = 20 ∧ y = 488 := by
  sorry

end problem_solution_l200_20094


namespace regular_quad_pyramid_angle_relation_l200_20039

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Dihedral angle between a lateral face and the base -/
  α : ℝ
  /-- Dihedral angle between two adjacent lateral faces -/
  β : ℝ

/-- Theorem: In a regular quadrilateral pyramid, 2cosβ + cos2α = -1 -/
theorem regular_quad_pyramid_angle_relation (p : RegularQuadPyramid) :
  2 * Real.cos p.β + Real.cos (2 * p.α) = -1 := by
  sorry

end regular_quad_pyramid_angle_relation_l200_20039


namespace largest_divisor_of_Q_l200_20058

/-- Q is the product of two consecutive even numbers and their immediate preceding odd integer -/
def Q (n : ℕ) : ℕ := (2*n - 1) * (2*n) * (2*n + 2)

/-- 8 is the largest integer that divides Q for all n -/
theorem largest_divisor_of_Q :
  ∀ k : ℕ, (∀ n : ℕ, k ∣ Q n) → k ≤ 8 :=
sorry

end largest_divisor_of_Q_l200_20058


namespace probability_less_than_4_is_7_9_l200_20076

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies x + y < 4 --/
def probabilityLessThan4 (s : Square) : ℝ :=
  sorry

/-- Our specific square with vertices (0,0), (0,3), (3,3), and (3,0) --/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

theorem probability_less_than_4_is_7_9 :
  probabilityLessThan4 specificSquare = 7/9 := by
  sorry

end probability_less_than_4_is_7_9_l200_20076


namespace pennsylvania_quarter_percentage_l200_20011

theorem pennsylvania_quarter_percentage 
  (total_quarters : ℕ) 
  (state_quarter_ratio : ℚ) 
  (pennsylvania_quarters : ℕ) 
  (h1 : total_quarters = 35)
  (h2 : state_quarter_ratio = 2 / 5)
  (h3 : pennsylvania_quarters = 7) :
  (pennsylvania_quarters : ℚ) / ((state_quarter_ratio * total_quarters) : ℚ) = 1 / 2 := by
sorry

end pennsylvania_quarter_percentage_l200_20011


namespace mary_garden_potatoes_l200_20054

/-- The number of potatoes left in Mary's garden after planting and rabbit eating -/
def potatoes_left (initial : ℕ) (added : ℕ) (eaten : ℕ) : ℕ :=
  let rows := initial
  let per_row := 1 + added
  max (rows * per_row - rows * eaten) 0

theorem mary_garden_potatoes :
  potatoes_left 8 2 3 = 0 := by
  sorry

end mary_garden_potatoes_l200_20054


namespace solution_set_of_inequality_l200_20010

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x + 1) * (1 - 2 * x) > 0 ↔ -1/3 < x ∧ x < 1/2 := by
  sorry

end solution_set_of_inequality_l200_20010


namespace average_difference_l200_20095

theorem average_difference (x : ℚ) : 
  (10 + 80 + x) / 3 = (20 + 40 + 60) / 3 - 5 ↔ x = 15 := by
  sorry

end average_difference_l200_20095


namespace valid_arrays_l200_20068

def is_valid_array (p q r : ℕ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  p ≥ q ∧ q ≥ r ∧
  ((Prime p ∧ Prime q) ∨ (Prime p ∧ Prime r) ∨ (Prime q ∧ Prime r)) ∧
  ∃ k : ℕ, k > 0 ∧ (p + q + r)^2 = k * (p * q * r)

theorem valid_arrays :
  ∀ p q r : ℕ, is_valid_array p q r ↔
    (p = 3 ∧ q = 3 ∧ r = 3) ∨
    (p = 2 ∧ q = 2 ∧ r = 4) ∨
    (p = 3 ∧ q = 3 ∧ r = 12) ∨
    (p = 3 ∧ q = 2 ∧ r = 1) ∨
    (p = 3 ∧ q = 2 ∧ r = 25) :=
by sorry

#check valid_arrays

end valid_arrays_l200_20068


namespace card_digits_problem_l200_20070

theorem card_digits_problem (a b c : ℕ) : 
  0 < a → a < b → b < c → c < 10 →
  (999 * c + 90 * b - 990 * a) + 
  (100 * c + 9 * b - 99 * a) + 
  (10 * c + b - 10 * a) + 
  (c - a) = 9090 →
  a = 1 ∧ b = 2 ∧ c = 9 := by
sorry

end card_digits_problem_l200_20070


namespace axis_of_symmetry_l200_20035

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x + 3)^2

-- Theorem statement
theorem axis_of_symmetry (x : ℝ) :
  (∀ h : ℝ, f (x + h) = f (x - h)) ↔ x = -3 :=
by sorry

end axis_of_symmetry_l200_20035


namespace complex_equation_result_l200_20097

theorem complex_equation_result (m n : ℝ) (i : ℂ) (h : i * i = -1) :
  m + i = (1 + 2 * i) * n * i → n - m = 3 := by
  sorry

end complex_equation_result_l200_20097


namespace f_properties_l200_20091

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (∀ x, f (2*x + 1) + f x < 0 ↔ x < -1/3) :=
by sorry

end f_properties_l200_20091


namespace distance_between_points_l200_20005

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-3, 4)
  let p2 : ℝ × ℝ := (4, -5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 130 := by
  sorry

end distance_between_points_l200_20005


namespace somu_age_problem_l200_20072

theorem somu_age_problem (somu_age father_age : ℕ) : 
  (somu_age = father_age / 3) →
  (somu_age - 10 = (father_age - 10) / 5) →
  somu_age = 20 := by
sorry

end somu_age_problem_l200_20072


namespace min_distance_squared_l200_20003

/-- Given real numbers a, b, c, and d satisfying certain conditions,
    the minimum value of (a-c)^2 + (b-d)^2 is 1. -/
theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) : 
  ∃ (min_val : ℝ), min_val = 1 ∧ 
  ∀ (a' b' c' d' : ℝ), 
    Real.log (b' + 1) + a' - 3 * b' = 0 → 
    2 * d' - c' + Real.sqrt 5 = 0 → 
    (a' - c')^2 + (b' - d')^2 ≥ min_val :=
by sorry

end min_distance_squared_l200_20003


namespace max_value_m_l200_20067

/-- Represents a number in base 8 as XYZ₈ -/
def base8_repr (X Y Z : ℕ) : ℕ := 64 * X + 8 * Y + Z

/-- Represents a number in base 12 as ZYX₁₂ -/
def base12_repr (X Y Z : ℕ) : ℕ := 144 * Z + 12 * Y + X

/-- Theorem stating the maximum value of m given the conditions -/
theorem max_value_m (m : ℕ) (X Y Z : ℕ) 
  (h1 : m > 0)
  (h2 : m = base8_repr X Y Z)
  (h3 : m = base12_repr X Y Z)
  (h4 : X < 8 ∧ Y < 8 ∧ Z < 8)  -- X, Y, Z are single digits in base 8
  (h5 : Z < 12 ∧ Y < 12 ∧ X < 12)  -- Z, Y, X are single digits in base 12
  : m ≤ 475 :=
sorry

end max_value_m_l200_20067


namespace last_student_age_l200_20099

theorem last_student_age 
  (total_students : ℕ) 
  (avg_age_all : ℝ) 
  (group1_size : ℕ) 
  (avg_age_group1 : ℝ) 
  (group2_size : ℕ) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 13)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (last_student_age : ℝ), 
    last_student_age = total_students * avg_age_all - 
      (group1_size * avg_age_group1 + group2_size * avg_age_group2) ∧
    last_student_age = 16 := by
  sorry

end last_student_age_l200_20099


namespace unique_fixed_point_of_f_and_f_inv_l200_20088

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem unique_fixed_point_of_f_and_f_inv :
  ∃! x : ℝ, f x = f_inv x :=
sorry

end unique_fixed_point_of_f_and_f_inv_l200_20088


namespace chinese_space_station_altitude_l200_20086

theorem chinese_space_station_altitude :
  ∃ (n : ℝ), n = 389000 ∧ n = 3.89 * (10 ^ 5) := by
  sorry

end chinese_space_station_altitude_l200_20086


namespace milk_students_l200_20062

theorem milk_students (total : ℕ) (soda_percent : ℚ) (milk_percent : ℚ) (soda_count : ℕ) :
  soda_percent = 70 / 100 →
  milk_percent = 20 / 100 →
  soda_count = 84 →
  total = soda_count / soda_percent →
  ↑(total * milk_percent) = 24 := by
  sorry

end milk_students_l200_20062


namespace octal_26_is_decimal_22_l200_20017

def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones_digit := octal % 10
  let eights_digit := octal / 10
  eights_digit * 8 + ones_digit

theorem octal_26_is_decimal_22 : octal_to_decimal 26 = 22 := by
  sorry

end octal_26_is_decimal_22_l200_20017


namespace largest_negative_congruent_to_two_mod_twentynine_l200_20065

theorem largest_negative_congruent_to_two_mod_twentynine : 
  ∃ (n : ℤ), 
    n = -1011 ∧ 
    n ≡ 2 [ZMOD 29] ∧ 
    n < 0 ∧ 
    -9999 ≤ n ∧ 
    n ≥ -999 ∧ 
    ∀ (m : ℤ), 
      m ≡ 2 [ZMOD 29] → 
      m < 0 → 
      -9999 ≤ m → 
      m ≥ -999 → 
      m ≤ n :=
by sorry

end largest_negative_congruent_to_two_mod_twentynine_l200_20065


namespace triangle_OAB_area_and_point_C_l200_20045

-- Define points in 2D space
def O : Fin 2 → ℝ := ![0, 0]
def A : Fin 2 → ℝ := ![2, 4]
def B : Fin 2 → ℝ := ![6, -2]

-- Define the area of a triangle given three points
def triangleArea (p1 p2 p3 : Fin 2 → ℝ) : ℝ := sorry

-- Define a function to check if two line segments are parallel
def isParallel (p1 p2 p3 p4 : Fin 2 → ℝ) : Prop := sorry

-- Define a function to calculate the length of a line segment
def segmentLength (p1 p2 : Fin 2 → ℝ) : ℝ := sorry

theorem triangle_OAB_area_and_point_C :
  (triangleArea O A B = 14) ∧
  (∃ (C : Fin 2 → ℝ), (C = ![4, -6] ∨ C = ![8, 2]) ∧
                      isParallel O A B C ∧
                      segmentLength O A = segmentLength B C) := by
  sorry

end triangle_OAB_area_and_point_C_l200_20045


namespace daniel_goats_count_l200_20021

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of dogs Daniel has -/
def num_dogs : ℕ := 5

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The total number of legs of all animals -/
def total_legs : ℕ := 72

/-- The number of legs each animal has -/
def legs_per_animal : ℕ := 4

theorem daniel_goats_count :
  num_goats * legs_per_animal + 
  num_horses * legs_per_animal + 
  num_dogs * legs_per_animal + 
  num_cats * legs_per_animal + 
  num_turtles * legs_per_animal = total_legs :=
by sorry

end daniel_goats_count_l200_20021


namespace cube_side_length_l200_20038

/-- The side length of a cube given paint cost and coverage -/
theorem cube_side_length 
  (paint_cost : ℝ)  -- Cost of paint per kg
  (paint_coverage : ℝ)  -- Area covered by 1 kg of paint in sq. ft
  (total_cost : ℝ)  -- Total cost to paint the cube
  (h1 : paint_cost = 40)  -- Paint costs Rs. 40 per kg
  (h2 : paint_coverage = 20)  -- 1 kg of paint covers 20 sq. ft
  (h3 : total_cost = 10800)  -- Total cost is Rs. 10800
  : ∃ (side_length : ℝ), side_length = 30 ∧ 
    total_cost = 6 * side_length^2 * paint_cost / paint_coverage :=
by sorry

end cube_side_length_l200_20038


namespace inequality_system_solution_set_l200_20022

theorem inequality_system_solution_set (x : ℝ) : 
  (2/3 * (2*x + 5) > 2 ∧ x - 2 < 0) ↔ (-1 < x ∧ x < 2) := by sorry

end inequality_system_solution_set_l200_20022


namespace blocks_between_39_and_40_l200_20008

/-- Represents the number of blocks in the original tower -/
def original_tower_size : ℕ := 90

/-- Represents the number of blocks taken at a time to build the new tower -/
def blocks_per_group : ℕ := 3

/-- Calculates the group number for a given block number in the original tower -/
def group_number (block : ℕ) : ℕ :=
  (original_tower_size - block) / blocks_per_group + 1

/-- Calculates the position of a block within its group in the new tower -/
def position_in_group (block : ℕ) : ℕ :=
  (original_tower_size - block) % blocks_per_group + 1

/-- Theorem stating that there are 4 blocks between blocks 39 and 40 in the new tower -/
theorem blocks_between_39_and_40 :
  ∃ (a b c d : ℕ),
    group_number 39 = group_number a ∧
    group_number 39 = group_number b ∧
    group_number 40 = group_number c ∧
    group_number 40 = group_number d ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧
    position_in_group 39 < position_in_group a ∧
    position_in_group a < position_in_group b ∧
    position_in_group b < position_in_group c ∧
    position_in_group c < position_in_group d ∧
    position_in_group d < position_in_group 40 :=
by
  sorry

end blocks_between_39_and_40_l200_20008


namespace sum_of_x_and_y_l200_20096

def arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j ≤ n → a j - a i = (j - i : ℝ) * (a 1 - a 0)

theorem sum_of_x_and_y (a : ℕ → ℝ) (n : ℕ) :
  arithmetic_sequence a n ∧ a 0 = 3 ∧ a n = 33 →
  a (n - 1) + a (n - 2) = 48 :=
by sorry

end sum_of_x_and_y_l200_20096


namespace triangle_inequality_l200_20078

/-- Given a triangle with side lengths a, b, c and area S, 
    prove that a^2 + b^2 + c^2 ≥ 4√3 S -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end triangle_inequality_l200_20078


namespace age_difference_proof_l200_20015

/-- Proves that the age difference between a man and his son is 34 years. -/
theorem age_difference_proof (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 32 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 34 := by
  sorry

#check age_difference_proof

end age_difference_proof_l200_20015


namespace A_alone_days_l200_20073

-- Define work rates for A, B, and C
def work_rate_A : ℝ := sorry
def work_rate_B : ℝ := sorry
def work_rate_C : ℝ := sorry

-- Define conditions
axiom cond1 : work_rate_A + work_rate_B = 1 / 3
axiom cond2 : work_rate_B + work_rate_C = 1 / 6
axiom cond3 : work_rate_A + work_rate_C = 5 / 18
axiom cond4 : work_rate_A + work_rate_B + work_rate_C = 1 / 2

-- Theorem to prove
theorem A_alone_days : 1 / work_rate_A = 36 / 7 := by sorry

end A_alone_days_l200_20073


namespace range_of_a_l200_20027

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - 2*x + 3 ≤ a^2 - 2*a - 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem statement
theorem range_of_a : 
  (∀ a : ℝ, solution_set a = ∅) ↔ (∀ a : ℝ, -1 < a ∧ a < 3) :=
sorry

end range_of_a_l200_20027


namespace ice_melting_problem_l200_20081

theorem ice_melting_problem (initial_volume : ℝ) : 
  initial_volume = 3.2 →
  (1/4) * ((1/4) * initial_volume) = 0.2 := by
  sorry

end ice_melting_problem_l200_20081


namespace rate_of_interest_l200_20006

/-- Simple interest formula -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- Given conditions -/
def principal : ℝ := 400
def interest : ℝ := 160
def time : ℝ := 2

/-- Theorem: The rate of interest is 0.2 given the conditions -/
theorem rate_of_interest :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.2 := by
  sorry

end rate_of_interest_l200_20006


namespace certain_number_proof_l200_20030

theorem certain_number_proof : ∃ x : ℝ, 0.80 * x = (4/5 * 20) + 16 ∧ x = 40 := by
  sorry

end certain_number_proof_l200_20030


namespace definite_integral_arctg_x_l200_20085

theorem definite_integral_arctg_x : 
  ∫ x in (0 : ℝ)..1, (4 * Real.arctan x - x) / (1 + x^2) = (π^2 - 4 * Real.log 2) / 8 := by
  sorry

end definite_integral_arctg_x_l200_20085


namespace log_expression_simplification_l200_20026

theorem log_expression_simplification :
  Real.log 16 / Real.log 4 / (Real.log (1/16) / Real.log 4) + Real.log 32 / Real.log 4 = 1.5 := by
  sorry

end log_expression_simplification_l200_20026


namespace scientific_notation_of_nine_billion_l200_20077

theorem scientific_notation_of_nine_billion :
  9000000000 = 9 * (10 ^ 9) := by sorry

end scientific_notation_of_nine_billion_l200_20077


namespace joey_study_time_l200_20014

/-- Calculates the total study time for Joey's SAT exam preparation --/
def total_study_time (weekday_hours_per_night : ℕ) (weekday_nights : ℕ) 
  (weekend_hours_per_day : ℕ) (weekend_days : ℕ) (weeks_until_exam : ℕ) : ℕ :=
  ((weekday_hours_per_night * weekday_nights + weekend_hours_per_day * weekend_days) * weeks_until_exam)

/-- Proves that Joey will spend 96 hours studying for his SAT exam --/
theorem joey_study_time : 
  total_study_time 2 5 3 2 6 = 96 := by
  sorry

end joey_study_time_l200_20014


namespace min_socks_for_different_pairs_l200_20044

/-- Represents a sock with a size and a color -/
structure Sock :=
  (size : Nat)
  (color : Nat)

/-- Represents the total number of socks -/
def totalSocks : Nat := 8

/-- Represents the number of different sizes -/
def numSizes : Nat := 2

/-- Represents the number of different colors -/
def numColors : Nat := 2

/-- Theorem stating the minimum number of socks needed to guarantee two pairs of different sizes and colors -/
theorem min_socks_for_different_pairs :
  ∀ (socks : Finset Sock),
    (Finset.card socks = totalSocks) →
    (∀ s ∈ socks, s.size < numSizes ∧ s.color < numColors) →
    (∃ (n : Nat),
      ∀ (subset : Finset Sock),
        (Finset.card subset = n) →
        (subset ⊆ socks) →
        (∃ (s1 s2 s3 s4 : Sock),
          s1 ∈ subset ∧ s2 ∈ subset ∧ s3 ∈ subset ∧ s4 ∈ subset ∧
          s1.size ≠ s2.size ∧ s1.color ≠ s2.color ∧
          s3.size ≠ s4.size ∧ s3.color ≠ s4.color)) →
    n = 7 := by
  sorry

end min_socks_for_different_pairs_l200_20044


namespace complex_simplification_l200_20075

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  6 * (4 - 2*i) + 2*i * (7 - 3*i) = 30 + 2*i := by
  sorry

end complex_simplification_l200_20075


namespace xy_value_from_inequality_l200_20002

theorem xy_value_from_inequality (x y : ℝ) :
  2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2) →
  x * y = -9/4 := by
sorry

end xy_value_from_inequality_l200_20002


namespace root_difference_l200_20020

-- Define the equation
def equation (r : ℝ) : Prop :=
  (r^2 - 5*r - 20) / (r - 2) = 2*r + 7

-- Define the roots of the equation
def roots : Set ℝ :=
  {r : ℝ | equation r}

-- Theorem statement
theorem root_difference : ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 4 :=
sorry

end root_difference_l200_20020


namespace max_value_theorem_l200_20007

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 = 1 ∧ 2 * a * b * Real.sqrt 3 + 2 * b * c = 2 :=
sorry

end max_value_theorem_l200_20007


namespace union_A_B_when_a_4_intersection_A_B_equals_A_iff_l200_20000

open Set
open Real

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0}

-- Theorem 1: When a = 4, A ∪ B = {x | x ≥ 3 ∨ x ≤ 1}
theorem union_A_B_when_a_4 : 
  A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} := by sorry

-- Theorem 2: A ∩ B = A if and only if a ≥ 5 or a ≤ 0
theorem intersection_A_B_equals_A_iff (a : ℝ) : 
  A a ∩ B = A a ↔ a ≥ 5 ∨ a ≤ 0 := by sorry

end union_A_B_when_a_4_intersection_A_B_equals_A_iff_l200_20000


namespace same_terminal_side_angle_l200_20031

theorem same_terminal_side_angle :
  ∃ α : ℝ, 0 ≤ α ∧ α < 360 ∧ ∃ k : ℤ, α = k * 360 - 30 ∧ α = 330 := by
  sorry

end same_terminal_side_angle_l200_20031


namespace geometric_sequence_common_ratio_l200_20066

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a3 : a 3 = 7) 
  (h_S3 : a 0 + a 1 + a 2 = 21) : 
  (a 1 / a 0 = 1) ∨ (a 1 / a 0 = -1/2) := by
sorry

end geometric_sequence_common_ratio_l200_20066


namespace silver_to_gold_ratio_is_two_to_one_l200_20055

-- Define the number of gold, silver, and black balloons
def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def total_balloons : ℕ := 573

-- Define the number of silver balloons
def silver_balloons : ℕ := total_balloons - gold_balloons - black_balloons

-- Define the ratio of silver to gold balloons
def silver_to_gold_ratio : ℚ := silver_balloons / gold_balloons

-- Theorem statement
theorem silver_to_gold_ratio_is_two_to_one :
  silver_to_gold_ratio = 2 / 1 := by
  sorry


end silver_to_gold_ratio_is_two_to_one_l200_20055


namespace sum_of_integers_minus15_to_5_l200_20092

-- Define the range of integers
def lower_bound : Int := -15
def upper_bound : Int := 5

-- Define the sum of integers function
def sum_of_integers (a b : Int) : Int :=
  let n := b - a + 1
  let avg := (a + b) / 2
  n * avg

-- Theorem statement
theorem sum_of_integers_minus15_to_5 :
  sum_of_integers lower_bound upper_bound = -105 := by
  sorry

end sum_of_integers_minus15_to_5_l200_20092


namespace function_identification_l200_20069

theorem function_identification (f : ℝ → ℝ) (b : ℝ) 
  (h1 : ∀ x, f (3 * x) = 3 * x^2 + b) 
  (h2 : f 1 = 0) : 
  ∀ x, f x = (1/3) * x^2 - (1/3) := by
sorry

end function_identification_l200_20069


namespace x_intercept_of_specific_line_l200_20013

/-- The x-intercept of a line is a point on the x-axis where the line intersects it. -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  (c / a, 0)

/-- The line equation is of the form ax + by = c, where a, b, and c are rational numbers. -/
structure LineEquation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Theorem: The x-intercept of the line 4x - 3y = 24 is the point (6, 0). -/
theorem x_intercept_of_specific_line :
  let line : LineEquation := { a := 4, b := -3, c := 24 }
  x_intercept line.a line.b line.c = (6, 0) := by
  sorry

end x_intercept_of_specific_line_l200_20013


namespace max_profit_allocation_l200_20089

/-- Profit function for product A -/
def profit_A (x : ℝ) : ℝ := -x^2 + 4*x

/-- Profit function for product B -/
def profit_B (x : ℝ) : ℝ := 2*x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (3 - x)

/-- Theorem stating the maximum profit and optimal investment allocation -/
theorem max_profit_allocation :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧
  (∀ y ∈ Set.Icc 0 3, total_profit x ≥ total_profit y) ∧
  x = 1 ∧ total_profit x = 7 := by
  sorry

end max_profit_allocation_l200_20089


namespace ball_motion_problem_l200_20012

/-- Ball motion problem -/
theorem ball_motion_problem 
  (dist_A_to_wall : ℝ) 
  (dist_wall_to_B : ℝ) 
  (dist_AB : ℝ) 
  (initial_velocity : ℝ) 
  (acceleration : ℝ) 
  (h1 : dist_A_to_wall = 5)
  (h2 : dist_wall_to_B = 2)
  (h3 : dist_AB = 9)
  (h4 : initial_velocity = 5)
  (h5 : acceleration = -0.4) :
  ∃ (return_speed : ℝ) (required_initial_speed : ℝ),
    return_speed = 3 ∧ required_initial_speed = 4 := by
  sorry


end ball_motion_problem_l200_20012


namespace passes_through_origin_symmetric_about_y_axis_symmetric_expression_inequality_condition_l200_20049

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(m-1)*x - 2*m + m^2

-- Theorem 1: The graph passes through the origin when m = 0 or m = 2
theorem passes_through_origin (m : ℝ) : 
  f m 0 = 0 ↔ m = 0 ∨ m = 2 := by sorry

-- Theorem 2: The graph is symmetric about the y-axis when m = 1
theorem symmetric_about_y_axis (m : ℝ) :
  (∀ x, f m x = f m (-x)) ↔ m = 1 := by sorry

-- Theorem 3: Expression when symmetric about y-axis
theorem symmetric_expression (x : ℝ) :
  f 1 x = x^2 - 1 := by sorry

-- Theorem 4: Condition for f(x) ≥ 3 in the interval [1, 3]
theorem inequality_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x ≥ 3) ↔ m ≤ 0 ∨ m ≥ 6 := by sorry

end passes_through_origin_symmetric_about_y_axis_symmetric_expression_inequality_condition_l200_20049


namespace corner_sum_is_164_l200_20004

/-- Represents a 9x9 grid filled with numbers 1 to 81 in row-major order -/
def Grid := Fin 9 → Fin 9 → Nat

/-- The value at position (i, j) in the grid -/
def gridValue (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The grid filled with numbers 1 to 81 -/
def numberGrid : Grid :=
  λ i j => gridValue i j

/-- The sum of the numbers in the four corners of the grid -/
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 8 + g 8 0 + g 8 8

theorem corner_sum_is_164 :
  cornerSum numberGrid = 164 := by sorry

end corner_sum_is_164_l200_20004


namespace existence_of_s_l200_20029

theorem existence_of_s (a : ℕ → ℕ) (k r : ℕ) (h1 : ∀ n m : ℕ, n ≤ m → a n ≤ a m) 
  (h2 : k > 0) (h3 : r > 0) (h4 : r = a r * (k + 1)) :
  ∃ s : ℕ, s > 0 ∧ s = a s * k := by
  sorry

end existence_of_s_l200_20029


namespace quadratic_no_roots_l200_20063

theorem quadratic_no_roots (b c : ℝ) 
  (h : ∀ x : ℝ, x^2 + b*x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b*x + c = 0 := by
  sorry

end quadratic_no_roots_l200_20063


namespace ice_water_masses_l200_20052

/-- Given a cylindrical vessel with ice and water, calculate the initial masses. -/
theorem ice_water_masses (S : ℝ) (ρw ρi : ℝ) (Δh hf : ℝ) 
  (hS : S = 15) 
  (hρw : ρw = 1) 
  (hρi : ρi = 0.92) 
  (hΔh : Δh = 5) 
  (hhf : hf = 115) :
  ∃ (mi mw : ℝ), 
    mi = 862.5 ∧ 
    mw = 1050 ∧ 
    mi / ρi - mi / ρw = S * Δh ∧ 
    mw + mi = ρw * S * hf := by
  sorry

#check ice_water_masses

end ice_water_masses_l200_20052


namespace negation_of_existence_negation_of_quadratic_inequality_l200_20001

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l200_20001


namespace units_digit_of_product_with_sum_factorials_l200_20093

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product_with_sum_factorials : 
  units_digit (7 * sum_factorials 2023) = 1 := by sorry

end units_digit_of_product_with_sum_factorials_l200_20093


namespace divisibility_by_eight_l200_20024

theorem divisibility_by_eight (a : ℤ) (h : Even a) :
  (∃ k : ℤ, a * (a^2 + 20) = 8 * k) ∧
  (∃ l : ℤ, a * (a^2 - 20) = 8 * l) ∧
  (∃ m : ℤ, a * (a^2 - 4) = 8 * m) := by
  sorry

end divisibility_by_eight_l200_20024


namespace three_digit_special_property_l200_20079

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ 0 ∧
    b < 10 ∧
    c < 10 ∧
    3 * a * (10 * b + c) = n

theorem three_digit_special_property :
  {n : ℕ | is_valid_number n} = {150, 240, 735} :=
sorry

end three_digit_special_property_l200_20079


namespace max_value_sin_cos_function_l200_20064

theorem max_value_sin_cos_function :
  ∃ (M : ℝ), M = 1/2 - Real.sqrt 3/4 ∧
  ∀ (x : ℝ), Real.sin (3*Real.pi/2 + x) * Real.cos (Real.pi/6 - x) ≤ M :=
by sorry

end max_value_sin_cos_function_l200_20064


namespace sum_of_numeric_values_l200_20037

/-- The numeric value assigned to a letter based on its position in the alphabet. -/
def letterValue (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 1
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -1
  | 0 => 0
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

/-- The positions of the letters in "numeric" in the alphabet. -/
def numericPositions : List ℕ := [14, 21, 13, 5, 18, 9, 3]

/-- The theorem stating that the sum of the numeric values of the letters in "numeric" is -1. -/
theorem sum_of_numeric_values :
  (numericPositions.map letterValue).sum = -1 := by
  sorry

#eval (numericPositions.map letterValue).sum

end sum_of_numeric_values_l200_20037


namespace mary_flour_calculation_l200_20087

/-- Given a cake recipe that requires 12 cups of flour, and knowing that Mary still needs 2 more cups,
    prove that Mary has already put in 10 cups of flour. -/
theorem mary_flour_calculation (recipe_flour : ℕ) (flour_needed : ℕ) (flour_put_in : ℕ) : 
  recipe_flour = 12 → flour_needed = 2 → flour_put_in = recipe_flour - flour_needed := by
  sorry

#check mary_flour_calculation

end mary_flour_calculation_l200_20087


namespace correct_selling_prices_l200_20036

-- Define the types of items
inductive Item
| Pencil
| Eraser
| Sharpener

-- Define the cost price function in A-coins
def costPriceA (item : Item) : ℝ :=
  match item with
  | Item.Pencil => 15
  | Item.Eraser => 25
  | Item.Sharpener => 35

-- Define the exchange rate
def exchangeRate : ℝ := 2

-- Define the profit percentage function
def profitPercentage (item : Item) : ℝ :=
  match item with
  | Item.Pencil => 0.20
  | Item.Eraser => 0.25
  | Item.Sharpener => 0.30

-- Define the selling price function in B-coins
def sellingPriceB (item : Item) : ℝ :=
  let costB := costPriceA item * exchangeRate
  costB + (costB * profitPercentage item)

-- Theorem to prove the selling prices are correct
theorem correct_selling_prices :
  sellingPriceB Item.Pencil = 36 ∧
  sellingPriceB Item.Eraser = 62.5 ∧
  sellingPriceB Item.Sharpener = 91 := by
  sorry

end correct_selling_prices_l200_20036


namespace roots_product_equation_l200_20074

theorem roots_product_equation (p q : ℝ) (α β γ δ : ℂ) 
  (h1 : α^2 + p*α + 4 = 0) 
  (h2 : β^2 + p*β + 4 = 0)
  (h3 : γ^2 + q*γ + 4 = 0)
  (h4 : δ^2 + q*δ + 4 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -4 * (p^2 - q^2) := by
  sorry

end roots_product_equation_l200_20074


namespace cricket_players_count_l200_20080

/-- The number of cricket players in a games hour -/
def cricket_players (total_players hockey_players football_players softball_players : ℕ) : ℕ :=
  total_players - (hockey_players + football_players + softball_players)

/-- Theorem: There are 12 cricket players present in the ground -/
theorem cricket_players_count :
  cricket_players 50 17 11 10 = 12 := by
  sorry

end cricket_players_count_l200_20080


namespace function_divisibility_property_l200_20042

theorem function_divisibility_property (f : ℕ → ℕ) :
  (∀ x y : ℕ, (f x + f y) ∣ (x^2 - y^2)) →
  ∀ n : ℕ, f n = n :=
by sorry

end function_divisibility_property_l200_20042


namespace virus_radius_scientific_notation_l200_20028

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The radius of the virus in meters -/
def virus_radius : ℝ := 0.00000000495

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_radius_scientific_notation :
  to_scientific_notation virus_radius = ScientificNotation.mk 4.95 (-9) (by norm_num) :=
sorry

end virus_radius_scientific_notation_l200_20028
