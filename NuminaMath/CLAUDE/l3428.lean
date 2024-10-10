import Mathlib

namespace scientific_notation_equivalence_l3428_342889

theorem scientific_notation_equivalence : 
  ∃ (a : ℝ) (n : ℤ), 0.0000907 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 9.07 ∧ n = -5 := by
  sorry

end scientific_notation_equivalence_l3428_342889


namespace ellipse_axis_endpoint_distance_l3428_342870

/-- Given an ellipse defined by 16(x-2)^2 + 4y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 16 * (x - 2)^2 + 4 * y^2 = 64 ↔ 
      (x - 2)^2 / 4 + y^2 / 16 = 1) →
    (C.1 - 2)^2 / 4 + C.2^2 / 16 = 1 →
    (D.1 - 2)^2 / 4 + D.2^2 / 16 = 1 →
    C.2 = 4 ∨ C.2 = -4 →
    D.1 = 4 ∨ D.1 = 0 →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end ellipse_axis_endpoint_distance_l3428_342870


namespace sum_opposite_angles_inscribed_quadrilateral_l3428_342879

/-- A quadrilateral WXYZ inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The measure of the angle subtended by arc WZ at the circumference -/
  angle_WZ : ℝ
  /-- The measure of the angle subtended by arc XY at the circumference -/
  angle_XY : ℝ

/-- Theorem: Sum of opposite angles in an inscribed quadrilateral -/
theorem sum_opposite_angles_inscribed_quadrilateral 
  (quad : InscribedQuadrilateral) 
  (h1 : quad.angle_WZ = 40)
  (h2 : quad.angle_XY = 20) :
  ∃ (angle_WXY angle_WZY : ℝ), angle_WXY + angle_WZY = 120 :=
sorry

end sum_opposite_angles_inscribed_quadrilateral_l3428_342879


namespace remainder_six_divisor_count_l3428_342828

theorem remainder_six_divisor_count : 
  ∃! (n : ℕ), n > 6 ∧ 67 % n = 6 :=
sorry

end remainder_six_divisor_count_l3428_342828


namespace fourth_month_sale_l3428_342891

/-- Proves that the sale in the fourth month is 5399, given the sales for other months and the required average. -/
theorem fourth_month_sale
  (sale1 sale2 sale3 sale5 sale6 : ℕ)
  (average : ℕ)
  (h1 : sale1 = 5124)
  (h2 : sale2 = 5366)
  (h3 : sale3 = 5808)
  (h5 : sale5 = 6124)
  (h6 : sale6 = 4579)
  (h_avg : average = 5400)
  (h_total : sale1 + sale2 + sale3 + sale5 + sale6 + (6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)) = 6 * average) :
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6) = 5399 :=
by sorry


end fourth_month_sale_l3428_342891


namespace cos_squared_minus_sin_squared_15_deg_l3428_342826

theorem cos_squared_minus_sin_squared_15_deg (π : Real) :
  let deg15 : Real := π / 12
  (Real.cos deg15)^2 - (Real.sin deg15)^2 = Real.sqrt 3 / 2 := by sorry

end cos_squared_minus_sin_squared_15_deg_l3428_342826


namespace simultaneous_equations_solution_l3428_342838

theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ (∃ x y : ℝ, y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) :=
by sorry

end simultaneous_equations_solution_l3428_342838


namespace factory_working_days_l3428_342857

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 4560

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1140

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days :
  working_days = 4 :=
sorry

end factory_working_days_l3428_342857


namespace fruit_sales_theorem_l3428_342860

/-- Represents the pricing and sales model of a fruit in Huimin Fresh Supermarket -/
structure FruitSalesModel where
  cost_price : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℝ
  price_reduction_rate : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction -/
def daily_profit (model : FruitSalesModel) (price_reduction : ℝ) : ℝ :=
  (model.initial_selling_price - price_reduction - model.cost_price) *
  (model.initial_daily_sales + model.sales_increase_rate * price_reduction)

/-- The main theorem about the fruit sales model -/
theorem fruit_sales_theorem (model : FruitSalesModel) 
  (h_cost : model.cost_price = 20)
  (h_initial_price : model.initial_selling_price = 40)
  (h_initial_sales : model.initial_daily_sales = 20)
  (h_price_reduction : model.price_reduction_rate = 1)
  (h_sales_increase : model.sales_increase_rate = 2) :
  (∃ (x : ℝ), x = 10 ∧ daily_profit model x = daily_profit model 0) ∧
  (¬ ∃ (y : ℝ), daily_profit model y = 460) := by
  sorry


end fruit_sales_theorem_l3428_342860


namespace alex_chocolates_l3428_342866

theorem alex_chocolates : 
  ∀ n : ℕ, n ≥ 150 ∧ n % 19 = 17 → n ≥ 150 := by
  sorry

end alex_chocolates_l3428_342866


namespace kim_shirts_left_l3428_342895

def initial_shirts : ℚ := 4.5 * 12
def bought_shirts : ℕ := 7
def lost_shirts : ℕ := 2
def fraction_given : ℚ := 2 / 5

theorem kim_shirts_left : 
  let total_before_giving := initial_shirts + bought_shirts - lost_shirts
  let given_to_sister := ⌊fraction_given * total_before_giving⌋
  total_before_giving - given_to_sister = 36 := by
  sorry

end kim_shirts_left_l3428_342895


namespace solution_set_f_geq_3_range_of_a_l3428_342809

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ a ∈ Set.Icc (-1 : ℝ) 9 := by sorry

end solution_set_f_geq_3_range_of_a_l3428_342809


namespace mango_per_tree_l3428_342852

theorem mango_per_tree (papaya_trees : ℕ) (mango_trees : ℕ) (papaya_per_tree : ℕ) (total_fruits : ℕ)
  (h1 : papaya_trees = 2)
  (h2 : mango_trees = 3)
  (h3 : papaya_per_tree = 10)
  (h4 : total_fruits = 80) :
  (total_fruits - papaya_trees * papaya_per_tree) / mango_trees = 20 := by
  sorry

end mango_per_tree_l3428_342852


namespace martian_age_conversion_l3428_342830

/-- Converts a single digit from base 9 to base 10 -/
def base9ToBase10Digit (d : Nat) : Nat := d

/-- Converts a 3-digit number from base 9 to base 10 -/
def base9ToBase10 (d₂ d₁ d₀ : Nat) : Nat :=
  base9ToBase10Digit d₂ * 9^2 + base9ToBase10Digit d₁ * 9^1 + base9ToBase10Digit d₀ * 9^0

/-- The age of the Martian robot's manufacturing facility in base 9 -/
def martianAge : Nat := 376

theorem martian_age_conversion :
  base9ToBase10 3 7 6 = 312 := by
  sorry

end martian_age_conversion_l3428_342830


namespace traffic_light_probability_l3428_342897

theorem traffic_light_probability (p_A p_B p_C : ℚ) 
  (h_A : p_A = 1/3) (h_B : p_B = 1/2) (h_C : p_C = 2/3) : 
  (1 - p_A) * p_B * p_C + p_A * (1 - p_B) * p_C + p_A * p_B * (1 - p_C) = 7/18 := by
  sorry

end traffic_light_probability_l3428_342897


namespace quadratic_function_property_l3428_342856

theorem quadratic_function_property (a m : ℝ) (h_a : a > 0) : 
  let f := fun (x : ℝ) ↦ x^2 + x + a
  f m < 0 → f (m + 1) > 0 := by
sorry

end quadratic_function_property_l3428_342856


namespace playground_area_l3428_342835

/-- Proves that a rectangular playground with given conditions has an area of 29343.75 square feet -/
theorem playground_area : 
  ∀ (width length : ℝ),
  length = 3 * width + 40 →
  2 * (width + length) = 820 →
  width * length = 29343.75 := by
sorry

end playground_area_l3428_342835


namespace yellow_red_difference_after_border_l3428_342843

/-- Represents a hexagonal figure with red and yellow tiles -/
structure HexagonalFigure where
  red_tiles : ℕ
  yellow_tiles : ℕ

/-- Calculates the number of tiles needed for a border around a hexagonal figure -/
def border_tiles : ℕ := 18

/-- Adds a border of yellow tiles to a hexagonal figure -/
def add_border (figure : HexagonalFigure) : HexagonalFigure :=
  { red_tiles := figure.red_tiles,
    yellow_tiles := figure.yellow_tiles + border_tiles }

/-- The initial hexagonal figure -/
def initial_figure : HexagonalFigure :=
  { red_tiles := 15, yellow_tiles := 9 }

/-- Theorem: The difference between yellow and red tiles after adding a border is 12 -/
theorem yellow_red_difference_after_border :
  let new_figure := add_border initial_figure
  new_figure.yellow_tiles - new_figure.red_tiles = 12 := by
  sorry

end yellow_red_difference_after_border_l3428_342843


namespace mixture_composition_l3428_342814

/-- Represents a solution with a given percentage of carbonated water -/
structure Solution :=
  (carbonated_water_percent : ℝ)
  (h_percent : 0 ≤ carbonated_water_percent ∧ carbonated_water_percent ≤ 1)

/-- Represents a mixture of two solutions -/
structure Mixture (P Q : Solution) :=
  (p_volume : ℝ)
  (q_volume : ℝ)
  (h_positive : 0 < p_volume ∧ 0 < q_volume)
  (carbonated_water_percent : ℝ)
  (h_mixture_percent : 0 ≤ carbonated_water_percent ∧ carbonated_water_percent ≤ 1)
  (h_balance : p_volume * P.carbonated_water_percent + q_volume * Q.carbonated_water_percent = 
               (p_volume + q_volume) * carbonated_water_percent)

/-- The main theorem to prove -/
theorem mixture_composition 
  (P : Solution) 
  (Q : Solution) 
  (mix : Mixture P Q) 
  (h_P : P.carbonated_water_percent = 0.8) 
  (h_Q : Q.carbonated_water_percent = 0.55) 
  (h_mix : mix.carbonated_water_percent = 0.6) : 
  mix.p_volume / (mix.p_volume + mix.q_volume) = 0.2 := by
  sorry

end mixture_composition_l3428_342814


namespace probability_divisible_by_3_l3428_342839

/-- The set of digits to choose from -/
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- A four-digit number formed from the given set of digits -/
structure FourDigitNumber where
  d₁ : ℕ
  d₂ : ℕ
  d₃ : ℕ
  d₄ : ℕ
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  h₅ : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄
  h₆ : d₁ ≠ 0

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : ℕ :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- A number is divisible by 3 if the sum of its digits is divisible by 3 -/
def FourDigitNumber.divisibleBy3 (n : FourDigitNumber) : Prop :=
  (n.d₁ + n.d₂ + n.d₃ + n.d₄) % 3 = 0

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

/-- The set of all four-digit numbers divisible by 3 -/
def divisibleBy3Numbers : Finset FourDigitNumber :=
  sorry

/-- The main theorem -/
theorem probability_divisible_by_3 :
  (Finset.card divisibleBy3Numbers : ℚ) / (Finset.card allFourDigitNumbers) = 8 / 15 :=
sorry

end probability_divisible_by_3_l3428_342839


namespace lcm_1364_884_minus_100_l3428_342861

def lcm_minus_100 (a b : Nat) : Nat :=
  Nat.lcm a b - 100

theorem lcm_1364_884_minus_100 :
  lcm_minus_100 1364 884 = 1509692 := by
  sorry

end lcm_1364_884_minus_100_l3428_342861


namespace granger_grocery_bill_l3428_342802

def spam_price : ℕ := 3
def peanut_butter_price : ℕ := 5
def bread_price : ℕ := 2

def spam_quantity : ℕ := 12
def peanut_butter_quantity : ℕ := 3
def bread_quantity : ℕ := 4

def total_cost : ℕ := spam_price * spam_quantity + 
                      peanut_butter_price * peanut_butter_quantity + 
                      bread_price * bread_quantity

theorem granger_grocery_bill : total_cost = 59 := by
  sorry

end granger_grocery_bill_l3428_342802


namespace red_ball_certain_event_l3428_342824

/-- Represents a bag of balls -/
structure Bag where
  balls : Set Color

/-- Represents the color of a ball -/
inductive Color where
  | Red

/-- Represents an event -/
structure Event where
  occurs : Prop

/-- Defines a certain event -/
def CertainEvent (e : Event) : Prop :=
  e.occurs = True

/-- Defines the event of drawing a ball from a bag -/
def DrawBall (b : Bag) (c : Color) : Event where
  occurs := c ∈ b.balls

/-- Theorem: Drawing a red ball from a bag containing only red balls is a certain event -/
theorem red_ball_certain_event (b : Bag) (h : b.balls = {Color.Red}) :
  CertainEvent (DrawBall b Color.Red) := by
  sorry

end red_ball_certain_event_l3428_342824


namespace intersection_at_diametrically_opposite_points_l3428_342871

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure ThreeCircles where
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  tangent_point : ℝ × ℝ

def are_touching (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  dist c1.center p = c1.radius ∧
  dist c2.center p = c2.radius ∧
  dist c1.center c2.center = c1.radius + c2.radius

def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  dist c.center p = c.radius

def are_diametrically_opposite (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  dist p1 p2 = 2 * c.radius

theorem intersection_at_diametrically_opposite_points
  (tc : ThreeCircles)
  (h1 : tc.circle1.radius = tc.circle2.radius)
  (h2 : tc.circle2.radius = tc.circle3.radius)
  (h3 : are_touching tc.circle1 tc.circle2 tc.tangent_point)
  (h4 : passes_through tc.circle3 tc.tangent_point) :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ≠ tc.tangent_point ∧
    p2 ≠ tc.tangent_point ∧
    passes_through tc.circle1 p1 ∧
    passes_through tc.circle2 p2 ∧
    passes_through tc.circle3 p1 ∧
    passes_through tc.circle3 p2 ∧
    are_diametrically_opposite tc.circle3 p1 p2 :=
  sorry

end intersection_at_diametrically_opposite_points_l3428_342871


namespace unique_function_solution_l3428_342876

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x + f (1 - x) = x^2) ↔ (∀ x : ℝ, f x = (1/3) * (x^2 + 2*x - 1)) :=
sorry

end unique_function_solution_l3428_342876


namespace percentage_of_sheet_used_for_typing_l3428_342859

/-- Calculates the percentage of a rectangular sheet used for typing, given its dimensions and margins. -/
theorem percentage_of_sheet_used_for_typing 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (side_margin : ℝ) 
  (top_bottom_margin : ℝ) 
  (h1 : sheet_length = 30)
  (h2 : sheet_width = 20)
  (h3 : side_margin = 2)
  (h4 : top_bottom_margin = 3)
  : (((sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin)) / (sheet_width * sheet_length)) * 100 = 64 := by
  sorry

#check percentage_of_sheet_used_for_typing

end percentage_of_sheet_used_for_typing_l3428_342859


namespace sea_turtle_collection_age_difference_l3428_342868

/-- Converts a number from base 8 to base 10 -/
def octalToDecimal (n : ℕ) : ℕ :=
  (n % 10) + 8 * ((n / 10) % 10) + 64 * (n / 100)

theorem sea_turtle_collection_age_difference : 
  octalToDecimal 724 - octalToDecimal 560 = 100 := by
  sorry

end sea_turtle_collection_age_difference_l3428_342868


namespace equal_distribution_of_chicken_wings_l3428_342874

def chicken_wings_per_person (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) : ℕ :=
  (initial_wings + additional_wings) / num_friends

theorem equal_distribution_of_chicken_wings :
  chicken_wings_per_person 5 20 25 = 9 := by
  sorry

end equal_distribution_of_chicken_wings_l3428_342874


namespace repeating_decimal_equals_fraction_product_is_222_l3428_342819

/-- The repeating decimal 0.018018018... as a real number -/
def repeating_decimal : ℚ := 18 / 999

/-- The fraction 2/111 -/
def target_fraction : ℚ := 2 / 111

/-- Theorem stating that the repeating decimal 0.018018018... is equal to 2/111 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

/-- The product of the numerator and denominator of the fraction -/
def numerator_denominator_product : ℕ := 2 * 111

/-- Theorem stating that the product of the numerator and denominator is 222 -/
theorem product_is_222 : numerator_denominator_product = 222 := by
  sorry

end repeating_decimal_equals_fraction_product_is_222_l3428_342819


namespace collinear_vectors_x_value_l3428_342827

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a b → x = (1 : ℝ) / 2 := by
  sorry

end collinear_vectors_x_value_l3428_342827


namespace triangle_side_expression_l3428_342886

theorem triangle_side_expression (m : ℝ) : 
  (2 : ℝ) > 0 ∧ 5 > 0 ∧ m > 0 ∧ 
  2 + 5 > m ∧ 2 + m > 5 ∧ 5 + m > 2 →
  Real.sqrt ((m - 3)^2) + Real.sqrt ((m - 7)^2) = 4 := by
sorry

end triangle_side_expression_l3428_342886


namespace greatest_two_digit_with_digit_product_12_l3428_342805

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by
  sorry

end greatest_two_digit_with_digit_product_12_l3428_342805


namespace binomial_25_5_l3428_342875

theorem binomial_25_5 (h1 : (23 : ℕ).choose 3 = 1771)
                      (h2 : (23 : ℕ).choose 4 = 8855)
                      (h3 : (23 : ℕ).choose 5 = 33649) :
  (25 : ℕ).choose 5 = 53130 := by
  sorry

end binomial_25_5_l3428_342875


namespace find_T_l3428_342898

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/5 : ℚ) * T = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ T = 75 := by
  sorry

end find_T_l3428_342898


namespace parabola_properties_l3428_342869

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate to check if a point (x, y) is on the parabola -/
def Parabola.contains_point (p : Parabola) (x y : ℝ) : Prop :=
  p.y_at x = y

theorem parabola_properties (p : Parabola) 
  (h1 : p.contains_point (-1) 3)
  (h2 : p.contains_point 0 0)
  (h3 : p.contains_point 1 (-1))
  (h4 : p.contains_point 2 0)
  (h5 : p.contains_point 3 3) :
  (∃ x_sym : ℝ, x_sym = 1 ∧ ∀ x : ℝ, p.y_at (x_sym - x) = p.y_at (x_sym + x)) ∧ 
  (p.a > 0) ∧
  (∀ x y : ℝ, x < 0 ∧ y < 0 → ¬p.contains_point x y) :=
sorry

end parabola_properties_l3428_342869


namespace no_solution_exists_l3428_342812

theorem no_solution_exists : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (3 / a + 4 / b = 12 / (a + b)) := by
  sorry

end no_solution_exists_l3428_342812


namespace max_x_2009_l3428_342867

def sequence_property (x : ℕ → ℝ) :=
  ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0

theorem max_x_2009 (x : ℕ → ℝ) 
  (h : sequence_property x)
  (h0 : x 0 = 1)
  (h20 : x 20 = 9)
  (h200 : x 200 = 6) :
  x 2009 ≤ 6 := by
  sorry

end max_x_2009_l3428_342867


namespace geometric_sequence_sum_l3428_342850

/-- A geometric sequence with specific conditions -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 3 = 5 ∧
  a 2 + a 4 = 10

/-- The sum of the 6th and 8th terms equals 160 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 8 = 160 := by
  sorry

end geometric_sequence_sum_l3428_342850


namespace quadratic_factorization_count_l3428_342818

theorem quadratic_factorization_count :
  ∃! (S : Finset Int), 
    (∀ k ∈ S, ∃ a b c d : Int, 2 * X^2 - k * X + 6 = (a * X + b) * (c * X + d)) ∧
    (∀ k : Int, (∃ a b c d : Int, 2 * X^2 - k * X + 6 = (a * X + b) * (c * X + d)) → k ∈ S) ∧
    Finset.card S = 6 :=
by sorry


end quadratic_factorization_count_l3428_342818


namespace gcd_204_85_l3428_342880

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l3428_342880


namespace smallest_lcm_with_gcd_7_l3428_342888

theorem smallest_lcm_with_gcd_7 :
  ∃ (m n : ℕ), 
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 7 ∧
    Nat.lcm m n = 144001 ∧
    ∀ (a b : ℕ), 
      1000 ≤ a ∧ a < 10000 ∧
      1000 ≤ b ∧ b < 10000 ∧
      Nat.gcd a b = 7 →
      Nat.lcm a b ≥ 144001 :=
by sorry

end smallest_lcm_with_gcd_7_l3428_342888


namespace mryak_bryak_price_difference_l3428_342831

/-- The price of one "mryak" in rubles -/
def mryak_price : ℝ := sorry

/-- The price of one "bryak" in rubles -/
def bryak_price : ℝ := sorry

/-- Three "mryak" are 10 rubles more expensive than five "bryak" -/
axiom price_relation1 : 3 * mryak_price = 5 * bryak_price + 10

/-- Six "mryak" are 31 rubles more expensive than eight "bryak" -/
axiom price_relation2 : 6 * mryak_price = 8 * bryak_price + 31

/-- The price difference between seven "mryak" and nine "bryak" is 38 rubles -/
theorem mryak_bryak_price_difference : 7 * mryak_price - 9 * bryak_price = 38 := by
  sorry

end mryak_bryak_price_difference_l3428_342831


namespace perpendicular_line_equation_l3428_342834

/-- Given two lines in the plane, this theorem states that if one line passes through 
    a specific point and is perpendicular to the other line, then it has a specific equation. -/
theorem perpendicular_line_equation 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Prop) 
  (h₁ : l₁ = fun x y ↦ 2 * x - 3 * y + 4 = 0) 
  (h₂ : l₂ = fun x y ↦ 3 * x + 2 * y - 1 = 0) : 
  (∀ x y, l₂ x y ↔ (x = -1 ∧ y = 2 ∨ 
    ∃ m : Real, m * (2 : Real) / 3 = -1 ∧ 
    y - 2 = m * (x + 1))) := by 
  sorry

end perpendicular_line_equation_l3428_342834


namespace smallest_positive_difference_l3428_342890

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) :
  ∃ (k : ℤ), a - b = 17 + 23 * k ∧ ∀ (m : ℤ), m > 0 → m = a - b → m ≥ 17 :=
sorry

end smallest_positive_difference_l3428_342890


namespace evaluate_expression_l3428_342844

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/2) (hy : y = 1/3) (hz : z = 2) :
  (x^3 * y^4 * z)^2 = 1/104976 := by
  sorry

end evaluate_expression_l3428_342844


namespace box_balls_problem_l3428_342864

theorem box_balls_problem (B X : ℕ) (h1 : B = 57) (h2 : B - 44 = X - B) : X = 70 := by
  sorry

end box_balls_problem_l3428_342864


namespace sticker_redistribution_l3428_342803

theorem sticker_redistribution (noah emma liam : ℕ) 
  (h1 : emma = 3 * noah) 
  (h2 : liam = 4 * emma) : 
  (7 : ℚ) / 36 = (liam - (liam + emma + noah) / 3) / liam := by
  sorry

end sticker_redistribution_l3428_342803


namespace exp_monotone_in_interval_l3428_342800

theorem exp_monotone_in_interval (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b := by
  sorry

end exp_monotone_in_interval_l3428_342800


namespace extreme_point_property_l3428_342820

def f (a b x : ℝ) : ℝ := x^3 - a*x - b

theorem extreme_point_property (a b x₀ x₁ : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ x₀ ∧ |x - x₀| < ε → f a b x ≠ f a b x₀) →
  x₁ ≠ x₀ →
  f a b x₁ = f a b x₀ →
  x₁ + 2*x₀ = 0 := by
sorry

end extreme_point_property_l3428_342820


namespace cube_difference_negative_l3428_342806

theorem cube_difference_negative {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : a^3 - b^3 < 0 := by
  sorry

end cube_difference_negative_l3428_342806


namespace part_one_part_two_part_three_l3428_342853

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Define the universal set U
def U (m : ℝ) : Set ℝ := A ∪ B m

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : m = 3) : 
  A ∩ (U m \ B m) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for part (2)
theorem part_two : 
  {m : ℝ | A ∩ B m = ∅} = {m : ℝ | m ≤ -2} := by sorry

-- Theorem for part (3)
theorem part_three : 
  {m : ℝ | A ∩ B m = A} = {m : ℝ | m ≥ 4} := by sorry

end part_one_part_two_part_three_l3428_342853


namespace buying_goods_equations_l3428_342884

/-- Represents the problem of buying goods collectively --/
def BuyingGoods (x y : ℤ) : Prop :=
  (∃ (leftover : ℤ), 8 * x - y = leftover ∧ leftover = 3) ∧
  (∃ (shortage : ℤ), y - 7 * x = shortage ∧ shortage = 4)

/-- The correct system of equations for the buying goods problem --/
theorem buying_goods_equations (x y : ℤ) :
  BuyingGoods x y ↔ (8 * x - 3 = y ∧ 7 * x + 4 = y) :=
sorry

end buying_goods_equations_l3428_342884


namespace imaginary_part_of_complex_fraction_l3428_342855

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (3 - 4*i) / (1 + i)
  Complex.im z = -7/2 := by sorry

end imaginary_part_of_complex_fraction_l3428_342855


namespace max_plus_min_equals_13_l3428_342841

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem max_plus_min_equals_13 :
  ∃ (a b : ℝ), (∀ x ∈ domain, f x ≤ a) ∧
               (∀ x ∈ domain, b ≤ f x) ∧
               (∃ x₁ ∈ domain, f x₁ = a) ∧
               (∃ x₂ ∈ domain, f x₂ = b) ∧
               a + b = 13 :=
by sorry

end max_plus_min_equals_13_l3428_342841


namespace dog_turns_four_in_two_years_l3428_342887

/-- The number of years until a dog turns 4, given the owner's current age and age when the dog was born. -/
def years_until_dog_turns_four (owner_current_age : ℕ) (owner_age_when_dog_born : ℕ) : ℕ :=
  4 - (owner_current_age - owner_age_when_dog_born)

/-- Theorem: Given that the dog was born when the owner was 15 and the owner is now 17,
    the dog will turn 4 in 2 years. -/
theorem dog_turns_four_in_two_years :
  years_until_dog_turns_four 17 15 = 2 := by
  sorry

end dog_turns_four_in_two_years_l3428_342887


namespace smallest_x_value_l3428_342813

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (4 : ℚ) / 5 = y / (205 + x)) : 
  5 ≤ x ∧ ∃ (y' : ℕ+), (4 : ℚ) / 5 = y' / (205 + 5) :=
sorry

end smallest_x_value_l3428_342813


namespace retailer_profit_percentage_retailer_profit_is_65_percent_l3428_342896

/-- Calculates the profit percentage for a retailer selling pens -/
theorem retailer_profit_percentage 
  (num_pens : ℕ) 
  (cost_price : ℝ) 
  (market_price : ℝ) 
  (discount_rate : ℝ) : ℝ :=
  let selling_price := num_pens * (market_price * (1 - discount_rate))
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- Proves that the retailer's profit percentage is 65% under given conditions -/
theorem retailer_profit_is_65_percent : 
  retailer_profit_percentage 60 36 1 0.01 = 65 := by
  sorry

end retailer_profit_percentage_retailer_profit_is_65_percent_l3428_342896


namespace factorial_sum_remainder_l3428_342825

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_remainder (n : ℕ) (h : n = 10) : sum_factorials n % 12 = 9 := by
  sorry

end factorial_sum_remainder_l3428_342825


namespace complex_to_exponential_form_l3428_342851

theorem complex_to_exponential_form (z : ℂ) : z = 2 - 2 * Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = 5 * Real.pi / 3 := by
  sorry

end complex_to_exponential_form_l3428_342851


namespace square_symbol_function_l3428_342854

/-- Represents the possible functions of symbols in a program flowchart -/
inductive FlowchartSymbolFunction
  | Output
  | Assignment
  | Decision
  | EndOfAlgorithm
  | Calculation

/-- Represents a symbol in a program flowchart -/
structure FlowchartSymbol where
  shape : String
  function : FlowchartSymbolFunction

/-- The square symbol in a program flowchart -/
def squareSymbol : FlowchartSymbol :=
  { shape := "□", function := FlowchartSymbolFunction.Assignment }

/-- Theorem stating the function of the square symbol in a program flowchart -/
theorem square_symbol_function :
  (squareSymbol.function = FlowchartSymbolFunction.Assignment) ∨
  (squareSymbol.function = FlowchartSymbolFunction.Calculation) :=
by sorry

end square_symbol_function_l3428_342854


namespace ellipse_and_line_problem_l3428_342840

-- Define the ellipse C
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

-- Define the line l
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem statement
theorem ellipse_and_line_problem 
  (C : Ellipse)
  (l₁ : Line)
  (l₂ : Line)
  (h₁ : l₁.slope = Real.sqrt 3)
  (h₂ : l₁.intercept = -2 * Real.sqrt 3)
  (h₃ : C.a^2 - C.b^2 = 4)
  (h₄ : (C.a^2 - C.b^2) / C.a^2 = 6 / 9)
  (h₅ : l₂.intercept = -3) :
  (C.a^2 = 6 ∧ C.b^2 = 2) ∧ 
  ((l₂.slope = Real.sqrt 3 ∧ l₂.intercept = -3) ∨ 
   (l₂.slope = -Real.sqrt 3 ∧ l₂.intercept = -3)) := by
  sorry


end ellipse_and_line_problem_l3428_342840


namespace max_value_theorem_l3428_342842

theorem max_value_theorem (x y z : ℝ) (h : x + 2 * y + z = 4) :
  ∃ (max : ℝ), max = 4 ∧ ∀ (a b c : ℝ), a + 2 * b + c = 4 → a * b + a * c + b * c ≤ max :=
sorry

end max_value_theorem_l3428_342842


namespace red_probability_both_jars_l3428_342873

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents the state of both jars -/
structure JarState :=
  (jarA : Jar)
  (jarB : Jar)

/-- Initial state of the jars -/
def initialState : JarState :=
  { jarA := { red := 6, blue := 10 },
    jarB := { red := 2, blue := 3 } }

/-- Function to transfer buttons between jars -/
def transfer (s : JarState) (n : ℕ) : JarState :=
  { jarA := { red := s.jarA.red - n, blue := s.jarA.blue - n },
    jarB := { red := s.jarB.red + n, blue := s.jarB.blue + n } }

/-- Final state after transfer -/
def finalState : JarState :=
  transfer initialState 3

/-- Probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / (j.red + j.blue)

/-- Theorem: The probability of selecting red buttons from both jars is 3/22 -/
theorem red_probability_both_jars :
  redProbability finalState.jarA * redProbability finalState.jarB = 3 / 22 := by
  sorry

end red_probability_both_jars_l3428_342873


namespace cone_from_sector_l3428_342848

/-- Given a 270° sector of a circle with radius 8, prove that the cone formed by aligning
    the straight sides of the sector has a base radius of 6 and a slant height of 8. -/
theorem cone_from_sector (r : ℝ) (angle : ℝ) (h1 : r = 8) (h2 : angle = 270) :
  let sector_arc_length := (angle / 360) * (2 * Real.pi * r)
  let cone_base_circumference := sector_arc_length
  let cone_base_radius := cone_base_circumference / (2 * Real.pi)
  let cone_slant_height := r
  (cone_base_radius = 6) ∧ (cone_slant_height = 8) :=
by sorry

end cone_from_sector_l3428_342848


namespace sum_of_ages_is_fifty_l3428_342877

/-- The sum of ages of 5 children born at intervals of 3 years -/
def sum_of_ages (youngest_age : ℕ) (interval : ℕ) (num_children : ℕ) : ℕ :=
  let ages := List.range num_children
  ages.map (fun i => youngest_age + i * interval) |> List.sum

/-- Theorem stating the sum of ages for the given conditions -/
theorem sum_of_ages_is_fifty :
  sum_of_ages 4 3 5 = 50 := by
  sorry

#eval sum_of_ages 4 3 5

end sum_of_ages_is_fifty_l3428_342877


namespace hyperbola_eccentricity_l3428_342836

/-- The eccentricity of a hyperbola with equation y²/4 - x² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  ∃ (e : ℝ), e = (Real.sqrt 5) / 2 ∧ 
  ∀ (x y : ℝ), y^2 / 4 - x^2 = 1 → 
  e = Real.sqrt ((y^2 / 4) + x^2) / (y / 2) :=
by sorry

end hyperbola_eccentricity_l3428_342836


namespace complex_subtraction_simplification_l3428_342865

theorem complex_subtraction_simplification :
  (-3 - 2*I) - (1 + 4*I) = -4 - 6*I :=
by sorry

end complex_subtraction_simplification_l3428_342865


namespace permutations_of_five_l3428_342849

theorem permutations_of_five (d : ℕ) : d = Nat.factorial 5 → d = 120 := by
  sorry

end permutations_of_five_l3428_342849


namespace orange_harvest_l3428_342804

/-- The number of sacks of oranges kept after a given number of harvest days -/
def sacksKept (harvestedPerDay discardedPerDay harvestDays : ℕ) : ℕ :=
  (harvestedPerDay - discardedPerDay) * harvestDays

/-- Theorem: The number of sacks of oranges kept after 51 days of harvest is 153 -/
theorem orange_harvest :
  sacksKept 74 71 51 = 153 := by
  sorry

end orange_harvest_l3428_342804


namespace compound_molecular_weight_l3428_342807

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_count : ℕ) (hydrogen_count : ℕ) (oxygen_count : ℕ) (nitrogen_count : ℕ) (sulfur_count : ℕ) : ℝ :=
  let carbon_weight : ℝ := 12.01
  let hydrogen_weight : ℝ := 1.008
  let oxygen_weight : ℝ := 16.00
  let nitrogen_weight : ℝ := 14.01
  let sulfur_weight : ℝ := 32.07
  carbon_count * carbon_weight + hydrogen_count * hydrogen_weight + oxygen_count * oxygen_weight + 
  nitrogen_count * nitrogen_weight + sulfur_count * sulfur_weight

/-- Theorem stating that the molecular weight of the given compound is approximately 323.46 g/mol -/
theorem compound_molecular_weight : 
  ∃ ε > 0, |molecular_weight 10 15 4 2 3 - 323.46| < ε :=
sorry

end compound_molecular_weight_l3428_342807


namespace three_x_squared_y_squared_l3428_342885

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
sorry

end three_x_squared_y_squared_l3428_342885


namespace x_value_l3428_342894

theorem x_value : ∀ x : ℕ, x = 225 + 2 * 15 * 9 + 81 → x = 576 := by
  sorry

end x_value_l3428_342894


namespace joe_new_average_l3428_342846

/-- Calculates the new average score after dropping the lowest score -/
def new_average (num_tests : ℕ) (original_average : ℚ) (lowest_score : ℚ) : ℚ :=
  (num_tests * original_average - lowest_score) / (num_tests - 1)

/-- Theorem: Given Joe's test scores, his new average after dropping the lowest score is 95 -/
theorem joe_new_average :
  let num_tests : ℕ := 4
  let original_average : ℚ := 90
  let lowest_score : ℚ := 75
  new_average num_tests original_average lowest_score = 95 := by
sorry

end joe_new_average_l3428_342846


namespace investment_problem_l3428_342815

/-- Proves that given the conditions of the investment problem, the invested sum is 15000 --/
theorem investment_problem (P : ℝ) 
  (h1 : P * (15 / 100) * 2 - P * (12 / 100) * 2 = 900) : 
  P = 15000 := by
  sorry

end investment_problem_l3428_342815


namespace red_light_runners_estimate_l3428_342823

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : ℕ
  yes_answers : ℕ
  id_range : Set ℕ

/-- Calculates the estimated number of people who have run a red light -/
def estimate_red_light_runners (data : SurveyData) : ℕ :=
  2 * (data.yes_answers - data.total_students / 4)

/-- Theorem stating the estimated number of red light runners -/
theorem red_light_runners_estimate (data : SurveyData) 
  (h1 : data.total_students = 800)
  (h2 : data.yes_answers = 240)
  (h3 : data.id_range = {n : ℕ | 1 ≤ n ∧ n ≤ 800}) :
  estimate_red_light_runners data = 80 := by
  sorry

end red_light_runners_estimate_l3428_342823


namespace walking_speed_problem_l3428_342899

/-- The walking speeds of two people meeting on a path --/
theorem walking_speed_problem (total_distance : ℝ) (time_diff : ℝ) (meeting_time : ℝ) (speed_diff : ℝ) :
  total_distance = 1200 →
  time_diff = 6 →
  meeting_time = 12 →
  speed_diff = 20 →
  ∃ (v : ℝ),
    v > 0 ∧
    (meeting_time + time_diff) * v + meeting_time * (v + speed_diff) = total_distance ∧
    v = 32 := by
  sorry

end walking_speed_problem_l3428_342899


namespace peter_pizza_fraction_l3428_342882

theorem peter_pizza_fraction (total_slices : ℕ) (peter_alone : ℕ) (shared_paul : ℚ) (shared_patty : ℚ) :
  total_slices = 16 →
  peter_alone = 3 →
  shared_paul = 1 / 2 →
  shared_patty = 1 / 2 →
  (peter_alone : ℚ) / total_slices + shared_paul / total_slices + shared_patty / total_slices = 1 / 4 := by
  sorry

end peter_pizza_fraction_l3428_342882


namespace distance_to_pole_for_given_point_l3428_342811

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The distance from a point to the pole in polar coordinates -/
def distanceToPole (p : PolarPoint) : ℝ := p.r

theorem distance_to_pole_for_given_point :
  let A : PolarPoint := { r := 3, θ := -4 }
  distanceToPole A = 3 := by sorry

end distance_to_pole_for_given_point_l3428_342811


namespace best_fitting_highest_r_squared_l3428_342883

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared : 0 ≤ r_squared ∧ r_squared ≤ 1

/-- Determines if a model is the best-fitting among a list of models -/
def is_best_fitting (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fitting_highest_r_squared 
  (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.98)
  (h2 : model2.r_squared = 0.80)
  (h3 : model3.r_squared = 0.50)
  (h4 : model4.r_squared = 0.25) :
  is_best_fitting model1 [model1, model2, model3, model4] :=
by sorry

end best_fitting_highest_r_squared_l3428_342883


namespace sum_of_five_digit_binary_numbers_l3428_342829

/-- The set of all positive integers with five digits in base 2 -/
def T : Set Nat :=
  {n | 16 ≤ n ∧ n ≤ 31}

/-- The sum of all elements in T -/
def sum_T : Nat :=
  (Finset.range 16).sum (fun i => i + 16)

theorem sum_of_five_digit_binary_numbers :
  sum_T = 248 :=
sorry

end sum_of_five_digit_binary_numbers_l3428_342829


namespace least_cars_serviced_per_day_l3428_342862

/-- The number of cars that can be serviced in a workday by two mechanics -/
def cars_serviced_per_day (hours_per_day : ℕ) (rate1 : ℕ) (rate2 : ℕ) : ℕ :=
  (rate1 + rate2) * hours_per_day

/-- Theorem stating the least number of cars that can be serviced by Paul and Jack in a workday -/
theorem least_cars_serviced_per_day :
  cars_serviced_per_day 8 2 3 = 40 := by
  sorry

end least_cars_serviced_per_day_l3428_342862


namespace monthly_income_calculation_l3428_342808

/-- Given a person's monthly income and their spending on transport, 
    prove that their income is $2000 if they have $1900 left after transport expenses. -/
theorem monthly_income_calculation (I : ℝ) : 
  I - 0.05 * I = 1900 → I = 2000 := by
  sorry

end monthly_income_calculation_l3428_342808


namespace staffing_problem_l3428_342837

def number_of_staffing_ways (total_candidates : ℕ) (qualified_for_first : ℕ) (positions : ℕ) : ℕ :=
  qualified_for_first * (List.range (positions - 1)).foldl (fun acc i => acc * (total_candidates - i - 1)) 1

theorem staffing_problem (total_candidates : ℕ) (qualified_for_first : ℕ) (positions : ℕ) 
  (h1 : total_candidates = 15)
  (h2 : qualified_for_first = 8)
  (h3 : positions = 5)
  (h4 : qualified_for_first ≤ total_candidates) :
  number_of_staffing_ways total_candidates qualified_for_first positions = 17472 := by
  sorry

end staffing_problem_l3428_342837


namespace ellipse_equation_and_chord_length_l3428_342816

noncomputable section

-- Define the ellipse C₁
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the left endpoint of the ellipse
def left_endpoint : ℝ × ℝ := (-Real.sqrt 6, 0)

-- Define the line l₂
def line_l2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 2)

theorem ellipse_equation_and_chord_length 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse a b (Prod.fst parabola_focus) (Prod.snd parabola_focus))
  (h4 : ellipse a b (Prod.fst left_endpoint) (Prod.snd left_endpoint)) :
  (∀ x y, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    ellipse a b A.1 A.2 ∧ 
    ellipse a b B.1 B.2 ∧
    line_l2 A.1 A.2 ∧
    line_l2 B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 6 / 5) :=
sorry

end ellipse_equation_and_chord_length_l3428_342816


namespace sum_of_a_and_b_l3428_342817

/-- The function g(x) -/
def g (a b x : ℝ) : ℝ := (a * x - 2) * (x + b)

/-- The theorem stating that if g(x) > 0 has solution set (-1, 2), then a + b = -4 -/
theorem sum_of_a_and_b (a b : ℝ) :
  (∀ x, g a b x > 0 ↔ -1 < x ∧ x < 2) →
  a + b = -4 := by
  sorry

end sum_of_a_and_b_l3428_342817


namespace parallelogram_area_l3428_342832

/-- The area of a parallelogram with base 3.6 and height 2.5 times the base is 32.4 -/
theorem parallelogram_area : 
  let base : ℝ := 3.6
  let height : ℝ := 2.5 * base
  let area : ℝ := base * height
  area = 32.4 := by sorry

end parallelogram_area_l3428_342832


namespace leg_ratio_is_sqrt_seven_l3428_342893

/-- Configuration of squares and triangles -/
structure SquareTriangleConfig where
  /-- Side length of the inner square -/
  s : ℝ
  /-- Length of the shorter leg of each triangle -/
  a : ℝ
  /-- Length of the longer leg of each triangle -/
  b : ℝ
  /-- Side length of the outer square -/
  t : ℝ
  /-- The triangles are right triangles -/
  triangle_right : a^2 + b^2 = t^2
  /-- The area of the outer square is twice the area of the inner square -/
  area_relation : t^2 = 2 * s^2
  /-- The shorter legs of two triangles form one side of the inner square -/
  inner_side : 2 * a = s

/-- The ratio of the longer leg to the shorter leg is √7 -/
theorem leg_ratio_is_sqrt_seven (config : SquareTriangleConfig) :
  config.b / config.a = Real.sqrt 7 := by sorry

end leg_ratio_is_sqrt_seven_l3428_342893


namespace counterexample_exists_l3428_342858

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) ∧ n = 27 := by
  sorry

end counterexample_exists_l3428_342858


namespace acid_mixture_water_volume_l3428_342822

/-- Represents the composition of a mixture --/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- Calculates the total volume of a mixture --/
def totalVolume (m : Mixture) : ℝ := m.acid + m.water

/-- Represents the problem setup --/
structure AcidMixtureProblem where
  initialMixture : Mixture
  pureAcidVolume : ℝ
  finalWaterPercentage : ℝ

/-- Calculates the final mixture composition --/
def finalMixture (problem : AcidMixtureProblem) (addedVolume : ℝ) : Mixture :=
  { acid := problem.pureAcidVolume + addedVolume * problem.initialMixture.acid,
    water := addedVolume * problem.initialMixture.water }

/-- The main theorem to prove --/
theorem acid_mixture_water_volume
  (problem : AcidMixtureProblem)
  (h1 : problem.initialMixture.acid = 0.1)
  (h2 : problem.initialMixture.water = 0.9)
  (h3 : problem.pureAcidVolume = 5)
  (h4 : problem.finalWaterPercentage = 0.4) :
  ∃ (addedVolume : ℝ),
    let finalMix := finalMixture problem addedVolume
    finalMix.water / totalVolume finalMix = problem.finalWaterPercentage ∧
    finalMix.water = 3.6 := by
  sorry


end acid_mixture_water_volume_l3428_342822


namespace bed_weight_problem_l3428_342872

theorem bed_weight_problem (single_bed_weight : ℝ) (double_bed_weight : ℝ) : 
  (5 * single_bed_weight = 50) →
  (double_bed_weight = single_bed_weight + 10) →
  (2 * single_bed_weight + 4 * double_bed_weight = 100) :=
by
  sorry

end bed_weight_problem_l3428_342872


namespace sugar_spill_ratio_l3428_342847

/-- Proves that the ratio of sugar that fell to the ground to the sugar in the torn bag before it fell is 1:2 -/
theorem sugar_spill_ratio (initial_sugar : ℕ) (num_bags : ℕ) (remaining_sugar : ℕ) : 
  initial_sugar = 24 →
  num_bags = 4 →
  remaining_sugar = 21 →
  (initial_sugar - remaining_sugar) * 2 = initial_sugar / num_bags :=
by
  sorry

#check sugar_spill_ratio

end sugar_spill_ratio_l3428_342847


namespace triangle_sin_A_values_l3428_342810

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  AC : Real
  BC : Real

-- Define the theorem
theorem triangle_sin_A_values
  (abc : Triangle)
  (non_obtuse : abc.A ≤ 90 ∧ abc.B ≤ 90 ∧ abc.C ≤ 90)
  (ab_gt_ac : abc.AB > abc.AC)
  (angle_b_45 : abc.B = 45)
  (O : Real) -- Circumcenter
  (I : Real) -- Incenter
  (oi_relation : Real.sqrt 2 * (O - I) = abc.AB - abc.AC) :
  Real.sin abc.A = Real.sqrt 2 / 2 ∨ Real.sin abc.A = Real.sqrt (Real.sqrt 2 - 1 / 2) :=
by sorry

end triangle_sin_A_values_l3428_342810


namespace four_points_planes_l3428_342833

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in space -/
def FourPoints : Type := Fin 4 → Point3D

/-- Three points are not collinear if they don't lie on the same line -/
def NotCollinear (p q r : Point3D) : Prop :=
  ∀ t : ℝ, (q.x - p.x, q.y - p.y, q.z - p.z) ≠ t • (r.x - p.x, r.y - p.y, r.z - p.z)

/-- The number of planes determined by any three points from a set of four points -/
def NumberOfPlanes (points : FourPoints) : ℕ :=
  sorry

/-- Theorem: Given four points in space where any three are not collinear,
    the number of planes determined by any three of these points is either 1 or 4 -/
theorem four_points_planes (points : FourPoints)
    (h : ∀ i j k : Fin 4, i ≠ j → j ≠ k → i ≠ k → NotCollinear (points i) (points j) (points k)) :
    NumberOfPlanes points = 1 ∨ NumberOfPlanes points = 4 := by
  sorry

end four_points_planes_l3428_342833


namespace cost_of_seven_cds_cost_of_seven_cds_is_112_l3428_342801

/-- The cost of seven CDs given that two identical CDs cost $32 -/
theorem cost_of_seven_cds : ℝ :=
  let cost_of_two : ℝ := 32
  let cost_of_one : ℝ := cost_of_two / 2
  7 * cost_of_one

/-- Proof that the cost of seven CDs is $112 -/
theorem cost_of_seven_cds_is_112 : cost_of_seven_cds = 112 := by
  sorry

end cost_of_seven_cds_cost_of_seven_cds_is_112_l3428_342801


namespace fewer_cards_l3428_342845

/-- The number of soccer cards Chris has -/
def chris_cards : ℕ := 18

/-- The number of soccer cards Charlie has -/
def charlie_cards : ℕ := 32

/-- The difference in the number of cards between Charlie and Chris -/
def card_difference : ℕ := charlie_cards - chris_cards

theorem fewer_cards : card_difference = 14 := by
  sorry

end fewer_cards_l3428_342845


namespace power_of_three_plus_five_mod_eight_l3428_342878

theorem power_of_three_plus_five_mod_eight : (3^101 + 5) % 8 = 0 := by
  sorry

end power_of_three_plus_five_mod_eight_l3428_342878


namespace lines_coplanar_iff_k_l3428_342821

def line1 (t k : ℝ) : ℝ × ℝ × ℝ := (3 + 2*t, 2 + 3*t, 2 - k*t)
def line2 (u k : ℝ) : ℝ × ℝ × ℝ := (1 + k*u, 5 - u, 6 + 2*u)

def are_coplanar (k : ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ (t u : ℝ),
    a * (line1 t k).1 + b * (line1 t k).2.1 + c * (line1 t k).2.2 + d = 0 ∧
    a * (line2 u k).1 + b * (line2 u k).2.1 + c * (line2 u k).2.2 + d = 0

theorem lines_coplanar_iff_k (k : ℝ) :
  are_coplanar k ↔ (k = -5 - 3 * Real.sqrt 3 ∨ k = -5 + 3 * Real.sqrt 3) :=
by sorry

end lines_coplanar_iff_k_l3428_342821


namespace find_number_l3428_342892

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem find_number (xy : ℕ) (h1 : is_two_digit xy) 
  (h2 : (xy / 10) + (xy % 10) = 8)
  (h3 : reverse_digits xy - xy = 18) : xy = 35 := by
  sorry

end find_number_l3428_342892


namespace inequality_proof_l3428_342863

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end inequality_proof_l3428_342863


namespace train_platform_length_l3428_342881

/-- The length of the platform given the conditions of the train problem -/
theorem train_platform_length 
  (train_speed : ℝ) 
  (opposite_train_speed : ℝ) 
  (crossing_time : ℝ) 
  (platform_passing_time : ℝ) 
  (h1 : train_speed = 48) 
  (h2 : opposite_train_speed = 42) 
  (h3 : crossing_time = 12) 
  (h4 : platform_passing_time = 45) : 
  ∃ (platform_length : ℝ), 
    (abs (platform_length - 600) < 1) ∧ 
    (platform_length = train_speed * (5/18) * platform_passing_time) :=
sorry


end train_platform_length_l3428_342881
