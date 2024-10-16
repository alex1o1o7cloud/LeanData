import Mathlib

namespace NUMINAMATH_CALUDE_clock_cost_price_l1246_124680

theorem clock_cost_price (total_clocks : ℕ) (clocks_10_percent : ℕ) (clocks_20_percent : ℕ)
  (price_difference : ℝ) :
  total_clocks = 90 →
  clocks_10_percent = 40 →
  clocks_20_percent = 50 →
  price_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price = 80 ∧
    (clocks_10_percent : ℝ) * cost_price * 1.1 +
    (clocks_20_percent : ℝ) * cost_price * 1.2 -
    (total_clocks : ℝ) * cost_price * 1.15 = price_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l1246_124680


namespace NUMINAMATH_CALUDE_tomato_price_is_fifty_cents_l1246_124646

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers : ℕ
  lettucePerCustomer : ℕ
  tomatoesPerCustomer : ℕ
  lettucePricePerHead : ℚ
  totalSales : ℚ

/-- Calculates the price of each tomato based on the sales data --/
def tomatoPrice (sales : VillageFoodsSales) : ℚ :=
  let lettuceSales := sales.customers * sales.lettucePerCustomer * sales.lettucePricePerHead
  let tomatoSales := sales.totalSales - lettuceSales
  let totalTomatoes := sales.customers * sales.tomatoesPerCustomer
  tomatoSales / totalTomatoes

/-- Theorem stating that the tomato price is $0.50 given the specific sales data --/
theorem tomato_price_is_fifty_cents 
  (sales : VillageFoodsSales)
  (h1 : sales.customers = 500)
  (h2 : sales.lettucePerCustomer = 2)
  (h3 : sales.tomatoesPerCustomer = 4)
  (h4 : sales.lettucePricePerHead = 1)
  (h5 : sales.totalSales = 2000) :
  tomatoPrice sales = 1/2 := by
  sorry

#eval tomatoPrice {
  customers := 500,
  lettucePerCustomer := 2,
  tomatoesPerCustomer := 4,
  lettucePricePerHead := 1,
  totalSales := 2000
}

end NUMINAMATH_CALUDE_tomato_price_is_fifty_cents_l1246_124646


namespace NUMINAMATH_CALUDE_absolute_value_expression_l1246_124636

theorem absolute_value_expression (x : ℤ) (h : x = 1999) :
  |4*x^2 - 5*x + 1| - 4*|x^2 + 2*x + 2| + 3*x + 7 = -19990 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l1246_124636


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1246_124694

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_ratio : ℝ) 
  (base_altitude_angle : ℝ) :
  area = 162 →
  altitude_base_ratio = 2 →
  base_altitude_angle = 60 * π / 180 →
  ∃ (base : ℝ), base = 9 ∧ area = base * (altitude_base_ratio * base) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1246_124694


namespace NUMINAMATH_CALUDE_circle_trajectory_l1246_124609

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 77 = 0

-- Define the property of being externally tangent
def externally_tangent (P C : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), P x y ∧ C x y ∧ ∀ (x' y' : ℝ), P x' y' → C x' y' → (x = x' ∧ y = y')

-- Define the property of being internally tangent
def internally_tangent (P C : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), P x y ∧ C x y ∧ ∀ (x' y' : ℝ), P x' y' → C x' y' → (x = x' ∧ y = y')

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2 / 25 + y^2 / 21 = 1

-- State the theorem
theorem circle_trajectory :
  ∀ (P : ℝ → ℝ → Prop),
  (externally_tangent P C₁ ∧ internally_tangent P C₂) →
  (∀ (x y : ℝ), P x y → trajectory x y) :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_l1246_124609


namespace NUMINAMATH_CALUDE_noodle_problem_l1246_124669

theorem noodle_problem (x : ℚ) : 
  (2 / 3 : ℚ) * x = 54 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_noodle_problem_l1246_124669


namespace NUMINAMATH_CALUDE_sales_equation_solution_l1246_124672

/-- Given the sales equation and conditions, prove the value of p. -/
theorem sales_equation_solution (f w p : ℂ) (h1 : f * p - w = 15000) 
  (h2 : f = 10) (h3 : w = 10 + 250 * Complex.I) : p = 1501 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_sales_equation_solution_l1246_124672


namespace NUMINAMATH_CALUDE_expr_is_monomial_l1246_124644

-- Define what a monomial is
def is_monomial (expr : ℚ → ℚ) : Prop :=
  ∃ (a : ℚ) (n : ℕ), ∀ x, expr x = a * x^n

-- Define the expression y/2023
def expr (y : ℚ) : ℚ := y / 2023

-- Theorem statement
theorem expr_is_monomial : is_monomial expr :=
sorry

end NUMINAMATH_CALUDE_expr_is_monomial_l1246_124644


namespace NUMINAMATH_CALUDE_no_integer_pairs_l1246_124660

theorem no_integer_pairs : ¬∃ (x y : ℤ), 0 < x ∧ x < y ∧ Real.sqrt 2500 = Real.sqrt x + 2 * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_l1246_124660


namespace NUMINAMATH_CALUDE_hex_tile_difference_specific_hex_tile_difference_l1246_124621

/-- Represents a hexagonal tile arrangement with blue and green tiles -/
structure HexTileArrangement where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of hexagonal tiles around an existing arrangement -/
def add_border (arrangement : HexTileArrangement) (border_color : String) : HexTileArrangement :=
  match border_color with
  | "green" => { blue_tiles := arrangement.blue_tiles, 
                 green_tiles := arrangement.green_tiles + (arrangement.blue_tiles + arrangement.green_tiles) / 2 }
  | "blue" => { blue_tiles := arrangement.blue_tiles + (arrangement.blue_tiles + arrangement.green_tiles) / 2 + 3, 
                green_tiles := arrangement.green_tiles }
  | _ => arrangement

/-- The main theorem stating the difference in tile counts after adding two borders -/
theorem hex_tile_difference (initial : HexTileArrangement) :
  let with_green_border := add_border initial "green"
  let final := add_border with_green_border "blue"
  final.blue_tiles - final.green_tiles = 16 :=
by
  sorry

/-- The specific instance of the hexagonal tile arrangement -/
def initial_arrangement : HexTileArrangement := { blue_tiles := 20, green_tiles := 10 }

/-- Applying the theorem to the specific instance -/
theorem specific_hex_tile_difference :
  let with_green_border := add_border initial_arrangement "green"
  let final := add_border with_green_border "blue"
  final.blue_tiles - final.green_tiles = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_hex_tile_difference_specific_hex_tile_difference_l1246_124621


namespace NUMINAMATH_CALUDE_negative_m_exponent_division_l1246_124613

theorem negative_m_exponent_division (m : ℝ) :
  ((-m)^7) / ((-m)^2) = -m^5 := by sorry

end NUMINAMATH_CALUDE_negative_m_exponent_division_l1246_124613


namespace NUMINAMATH_CALUDE_ellipse_parameter_inequality_l1246_124682

/-- An ellipse with equation ax^2 + by^2 = 1 and foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b > 0
  foci_on_x_axis : a ≠ b

theorem ellipse_parameter_inequality (e : Ellipse) : 0 < e.a ∧ e.a < e.b := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parameter_inequality_l1246_124682


namespace NUMINAMATH_CALUDE_intersection_condition_union_condition_l1246_124643

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + a^2 - 12 = 0}

-- Part 1
theorem intersection_condition (a : ℝ) : A ∩ B a = A → a = -2 := by sorry

-- Part 2
theorem union_condition (a : ℝ) : A ∪ B a = A → a ≥ 4 ∨ a < -4 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_union_condition_l1246_124643


namespace NUMINAMATH_CALUDE_return_speed_calculation_l1246_124671

/-- Proves that given a round trip with specified conditions, the return speed is 160 km/h -/
theorem return_speed_calculation (total_time : ℝ) (outbound_time_minutes : ℝ) (outbound_speed : ℝ) :
  total_time = 5 →
  outbound_time_minutes = 192 →
  outbound_speed = 90 →
  let outbound_time_hours : ℝ := outbound_time_minutes / 60
  let distance : ℝ := outbound_speed * outbound_time_hours
  let return_time : ℝ := total_time - outbound_time_hours
  let return_speed : ℝ := distance / return_time
  return_speed = 160 := by
  sorry

#check return_speed_calculation

end NUMINAMATH_CALUDE_return_speed_calculation_l1246_124671


namespace NUMINAMATH_CALUDE_spade_calculation_l1246_124633

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : spade 3 (spade 5 (spade 7 10)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l1246_124633


namespace NUMINAMATH_CALUDE_expression_evaluation_l1246_124645

theorem expression_evaluation : 
  (2015^3 - 2 * 2015^2 * 2016 + 3 * 2015 * 2016^2 - 2016^3 + 1) / (2015 * 2016) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1246_124645


namespace NUMINAMATH_CALUDE_picks_theorem_irregular_polygon_area_l1246_124630

/-- Pick's Theorem for a polygon on a lattice -/
theorem picks_theorem (B I : ℕ) (A : ℚ) : A = I + B / 2 - 1 →
  B = 10 → I = 12 → A = 16 := by
  sorry

/-- The area of the irregular polygon -/
theorem irregular_polygon_area : ∃ A : ℚ, A = 16 := by
  sorry

end NUMINAMATH_CALUDE_picks_theorem_irregular_polygon_area_l1246_124630


namespace NUMINAMATH_CALUDE_prime_squares_and_fourth_powers_l1246_124607

theorem prime_squares_and_fourth_powers (p : ℕ) : 
  Prime p ↔ 
  (p = 2 ∨ p = 3) ∧ 
  (∃ (a b c k : ℤ), a^2 + b^2 + c^2 = p ∧ a^4 + b^4 + c^4 = k * p) :=
sorry

end NUMINAMATH_CALUDE_prime_squares_and_fourth_powers_l1246_124607


namespace NUMINAMATH_CALUDE_matrix_N_computation_l1246_124695

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec (![3, -2]) = ![4, 0])
  (h2 : N.mulVec (![(-4), 6]) = ![(-2), -2]) :
  N.mulVec (![7, 2]) = ![16, -4] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_computation_l1246_124695


namespace NUMINAMATH_CALUDE_decimal_calculation_l1246_124649

theorem decimal_calculation : (3.15 * 2.5) - 1.75 = 6.125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_calculation_l1246_124649


namespace NUMINAMATH_CALUDE_range_of_a_l1246_124647

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1246_124647


namespace NUMINAMATH_CALUDE_school_play_ticket_sales_l1246_124693

/-- Calculates the total sales from school play tickets -/
def total_ticket_sales (student_price adult_price : ℕ) (student_tickets adult_tickets : ℕ) : ℕ :=
  student_price * student_tickets + adult_price * adult_tickets

/-- Theorem: The total sales from the school play tickets is $216 -/
theorem school_play_ticket_sales :
  total_ticket_sales 6 8 20 12 = 216 := by
  sorry

end NUMINAMATH_CALUDE_school_play_ticket_sales_l1246_124693


namespace NUMINAMATH_CALUDE_dividend_calculation_l1246_124686

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 17 →
  quotient = 10 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  dividend = 172 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1246_124686


namespace NUMINAMATH_CALUDE_income_increase_percentage_l1246_124622

theorem income_increase_percentage 
  (initial_income : ℝ)
  (initial_expenditure_ratio : ℝ)
  (expenditure_increase_ratio : ℝ)
  (savings_increase_ratio : ℝ)
  (income_increase_ratio : ℝ)
  (h1 : initial_expenditure_ratio = 0.75)
  (h2 : expenditure_increase_ratio = 0.1)
  (h3 : savings_increase_ratio = 0.5)
  (h4 : initial_income > 0) :
  let initial_expenditure := initial_income * initial_expenditure_ratio
  let initial_savings := initial_income - initial_expenditure
  let new_income := initial_income * (1 + income_increase_ratio)
  let new_expenditure := initial_expenditure * (1 + expenditure_increase_ratio)
  let new_savings := new_income - new_expenditure
  (new_savings = initial_savings * (1 + savings_increase_ratio)) →
  (income_increase_ratio = 0.2) :=
by sorry

end NUMINAMATH_CALUDE_income_increase_percentage_l1246_124622


namespace NUMINAMATH_CALUDE_polygon_sides_l1246_124651

theorem polygon_sides (sum_angles : ℕ) (h1 : sum_angles = 1980) : ∃ n : ℕ, n = 13 ∧ sum_angles = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1246_124651


namespace NUMINAMATH_CALUDE_largest_non_representable_amount_l1246_124675

/-- Represents the denominations of coins in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ i => 5^(n - i) * 7^i)

/-- Determines if a number is representable using the given coin denominations -/
def is_representable (s n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), s = List.sum (List.zipWith (·*·) coeffs (coin_denominations n))

/-- The main theorem stating the largest non-representable amount -/
theorem largest_non_representable_amount (n : ℕ) :
  ∀ s : ℕ, s > 2 * 7^(n+1) - 3 * 5^(n+1) → is_representable s n ∧
  ¬is_representable (2 * 7^(n+1) - 3 * 5^(n+1)) n :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_amount_l1246_124675


namespace NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l1246_124657

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_quadratic (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a,
    b := 2 * f.a * shift + f.b,
    c := f.a * shift^2 + f.b * shift + f.c }

/-- The original quadratic function y = 3x^2 - 2x + 6 -/
def original_function : QuadraticFunction :=
  { a := 3, b := -2, c := 6 }

/-- The amount of left shift -/
def left_shift : ℝ := 5

theorem sum_of_coefficients_after_shift :
  let shifted := shift_quadratic original_function left_shift
  shifted.a + shifted.b + shifted.c = 102 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l1246_124657


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l1246_124608

theorem wire_cut_ratio (a b : ℝ) : 
  a > 0 → b > 0 → (∃ (r : ℝ), a = 2 * Real.pi * r) → (∃ (s : ℝ), b = 4 * s) → a = b → a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l1246_124608


namespace NUMINAMATH_CALUDE_max_d_value_l1246_124688

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (n : ℕ), d n = 401 ∧ ∀ (m : ℕ), d m ≤ 401 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1246_124688


namespace NUMINAMATH_CALUDE_infinite_rational_square_sum_169_l1246_124699

theorem infinite_rational_square_sum_169 : 
  ∀ n : ℕ, ∃ x y : ℚ, x^2 + y^2 = 169 ∧ 
  (∀ m : ℕ, m < n → ∃ x' y' : ℚ, x'^2 + y'^2 = 169 ∧ (x' ≠ x ∨ y' ≠ y)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_rational_square_sum_169_l1246_124699


namespace NUMINAMATH_CALUDE_acme_profit_l1246_124634

/-- Calculates the profit for a horseshoe manufacturing company. -/
def calculate_profit (initial_outlay : ℝ) (cost_per_set : ℝ) (selling_price : ℝ) (num_sets : ℕ) : ℝ :=
  let revenue := selling_price * num_sets
  let total_cost := initial_outlay + cost_per_set * num_sets
  revenue - total_cost

/-- Proves that the profit for Acme's horseshoe manufacturing is $15,337.50 -/
theorem acme_profit :
  calculate_profit 12450 20.75 50 950 = 15337.50 := by
  sorry

end NUMINAMATH_CALUDE_acme_profit_l1246_124634


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l1246_124697

theorem mistaken_multiplication (correct_multiplier : ℕ) (actual_number : ℕ) (difference : ℕ) :
  correct_multiplier = 43 →
  actual_number = 135 →
  actual_number * correct_multiplier - actual_number * (correct_multiplier - (difference / actual_number)) = difference →
  correct_multiplier - (difference / actual_number) = 34 :=
by sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l1246_124697


namespace NUMINAMATH_CALUDE_angle_calculation_l1246_124623

theorem angle_calculation (α : ℝ) (h : α = 30) : 
  2 * (90 - α) - (90 - (180 - α)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l1246_124623


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1246_124662

theorem hot_dogs_remainder : 25197643 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1246_124662


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l1246_124650

/-- Pentagon PQRST with given side lengths and angles -/
structure Pentagon where
  PQ : ℝ
  QR : ℝ
  ST : ℝ
  perimeter : ℝ
  angle_QRS : ℝ
  angle_RST : ℝ
  angle_STP : ℝ

/-- The area of a pentagon with given properties -/
def pentagon_area (p : Pentagon) : ℝ :=
  sorry

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  ∀ (p : Pentagon),
    p.PQ = 13 ∧
    p.QR = 18 ∧
    p.ST = 30 ∧
    p.perimeter = 82 ∧
    p.angle_QRS = 90 ∧
    p.angle_RST = 90 ∧
    p.angle_STP = 90 →
    pentagon_area p = 270 :=
  sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l1246_124650


namespace NUMINAMATH_CALUDE_no_three_digit_odd_sum_30_l1246_124631

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_three_digit_odd_sum_30 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 30 ∧ n % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_odd_sum_30_l1246_124631


namespace NUMINAMATH_CALUDE_triangle_property_l1246_124676

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) ∧
  ((c - 2*a) * Real.cos B + b * Real.cos C = 0) →
  (B = π/3) ∧
  (a + b + c = 6 ∧ b = 2 → 
    1/2 * a * c * Real.sin B = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1246_124676


namespace NUMINAMATH_CALUDE_walking_time_calculation_l1246_124668

/-- Given a man who walks and runs at different speeds, this theorem proves
    the time taken to walk a distance that he can run in 1.5 hours. -/
theorem walking_time_calculation (walk_speed run_speed : ℝ) (run_time : ℝ) 
    (h1 : walk_speed = 8)
    (h2 : run_speed = 16)
    (h3 : run_time = 1.5) : 
  (run_speed * run_time) / walk_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l1246_124668


namespace NUMINAMATH_CALUDE_jelly_beans_initial_amount_l1246_124698

theorem jelly_beans_initial_amount :
  ∀ (initial_amount eaten_amount : ℕ) 
    (num_piles pile_weight : ℕ),
  eaten_amount = 6 →
  num_piles = 3 →
  pile_weight = 10 →
  initial_amount = eaten_amount + num_piles * pile_weight →
  initial_amount = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_beans_initial_amount_l1246_124698


namespace NUMINAMATH_CALUDE_vals_money_value_is_38_80_l1246_124617

/-- Calculates the total value of Val's money in USD -/
def valsMoneyValue (initialNickels : ℕ) (dimesToNickelsRatio : ℕ) (quartersToDimesRatio : ℕ) 
  (newNickelsMultiplier : ℕ) (canadianNickelRatio : ℚ) (exchangeRate : ℚ) : ℚ :=
  let initialDimes := initialNickels * dimesToNickelsRatio
  let initialQuarters := initialDimes * quartersToDimesRatio
  let newNickels := initialNickels * newNickelsMultiplier
  let canadianNickels := (newNickels : ℚ) * canadianNickelRatio
  let usNickels := (newNickels : ℚ) - canadianNickels
  let initialValue := (initialNickels : ℚ) * (5 / 100) + (initialDimes : ℚ) * (10 / 100) + (initialQuarters : ℚ) * (25 / 100)
  let newUsNickelsValue := usNickels * (5 / 100)
  let canadianNickelsValue := canadianNickels * (5 / 100) * exchangeRate
  initialValue + newUsNickelsValue + canadianNickelsValue

/-- Theorem stating that Val's money value is $38.80 given the problem conditions -/
theorem vals_money_value_is_38_80 :
  valsMoneyValue 20 3 2 2 (1/2) (4/5) = 388/10 := by
  sorry

end NUMINAMATH_CALUDE_vals_money_value_is_38_80_l1246_124617


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1246_124637

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 4 * x * y) : 1 / x + 1 / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1246_124637


namespace NUMINAMATH_CALUDE_book_distribution_l1246_124616

/-- The number of ways to distribute n distinct books among k people, 
    with each person receiving m books -/
def distribute_books (n k m : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n distinct items -/
def choose (n r : ℕ) : ℕ := sorry

theorem book_distribution :
  distribute_books 6 3 2 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l1246_124616


namespace NUMINAMATH_CALUDE_fraction_simplification_l1246_124641

theorem fraction_simplification :
  (-45 : ℚ) / 25 / (15 : ℚ) / 40 = -24 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1246_124641


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1246_124655

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_of_A_and_B :
  A ∪ B = {1, 2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1246_124655


namespace NUMINAMATH_CALUDE_leila_spending_l1246_124691

/-- The amount Leila spent at the supermarket -/
def supermarket_cost : ℝ := 100

/-- The cost of fixing Leila's automobile -/
def automobile_cost : ℝ := 350

/-- The total amount Leila spent -/
def total_cost : ℝ := supermarket_cost + automobile_cost

theorem leila_spending :
  (automobile_cost = 3 * supermarket_cost + 50) →
  total_cost = 450 := by
  sorry

end NUMINAMATH_CALUDE_leila_spending_l1246_124691


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_negative_two_l1246_124654

theorem at_least_one_not_greater_than_negative_two 
  (a b : ℝ) (ha : a < 0) (hb : b < 0) : 
  (a + 1/b ≤ -2) ∨ (b + 1/a ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_negative_two_l1246_124654


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l1246_124600

theorem sum_remainder_zero (m : ℤ) : (11 - m + (m + 5)) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l1246_124600


namespace NUMINAMATH_CALUDE_triangle_similarity_l1246_124632

-- Define the points as complex numbers
variable (z₁ z₂ z₃ t₁ t₂ t₃ z₁' z₂' z₃' : ℂ)

-- Define the similarity relation
def similar (a b c d e f : ℂ) : Prop :=
  (e - d) / (f - d) = (b - a) / (c - a)

-- State the theorem
theorem triangle_similarity :
  similar z₁ z₂ z₃ t₁ t₂ t₃ →  -- DBC similar to ABC
  similar z₂ z₃ z₁ t₂ t₃ t₁ →  -- ECA similar to ABC
  similar z₃ z₁ z₂ t₃ t₁ t₂ →  -- FAB similar to ABC
  similar t₂ t₃ t₁ z₁' t₃ t₂ →  -- A'FE similar to DBC
  similar t₃ t₁ t₂ z₂' t₁ t₃ →  -- B'DF similar to ECA
  similar t₁ t₂ t₃ z₃' t₂ t₁ →  -- C'ED similar to FAB
  similar z₁ z₂ z₃ z₁' z₂' z₃'  -- A'B'C' similar to ABC
:= by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l1246_124632


namespace NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l1246_124612

/-- The number of unique two-scoop sundae combinations given a total number of flavors and vanilla as a required flavor. -/
def sundae_combinations (total_flavors : ℕ) (vanilla_required : Bool) : ℕ :=
  if vanilla_required then total_flavors - 1 else 0

/-- Theorem: Given 8 ice cream flavors with vanilla required, the number of unique two-scoop sundae combinations is 7. -/
theorem ice_cream_sundae_combinations :
  sundae_combinations 8 true = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l1246_124612


namespace NUMINAMATH_CALUDE_parallel_lines_k_equals_negative_one_l1246_124696

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.b ≠ 0

theorem parallel_lines_k_equals_negative_one :
  ∀ k : ℝ,
  let l1 : Line := ⟨k, -1, 1⟩
  let l2 : Line := ⟨1, -k, 1⟩
  parallel l1 l2 → k = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_equals_negative_one_l1246_124696


namespace NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l1246_124652

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 7 distinguishable balls into 2 distinguishable boxes is 128 -/
theorem distribute_seven_balls_two_boxes : 
  distribute_balls 7 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l1246_124652


namespace NUMINAMATH_CALUDE_abs_diff_properties_l1246_124627

-- Define the binary operation ⊕
def abs_diff (x y : ℝ) : ℝ := |x - y|

-- Main theorem
theorem abs_diff_properties :
  -- 1. ⊕ is commutative
  (∀ x y : ℝ, abs_diff x y = abs_diff y x) ∧
  -- 2. Addition distributes over ⊕
  (∀ a b c : ℝ, a + abs_diff b c = abs_diff (a + b) (a + c)) ∧
  -- 3. ⊕ is not associative
  (∃ x y z : ℝ, abs_diff x (abs_diff y z) ≠ abs_diff (abs_diff x y) z) ∧
  -- 4. ⊕ does not have an identity element
  (∀ e : ℝ, ∃ x : ℝ, abs_diff x e ≠ x) ∧
  -- 5. ⊕ does not distribute over addition
  (∃ x y z : ℝ, abs_diff x (y + z) ≠ abs_diff x y + abs_diff x z) :=
by sorry

end NUMINAMATH_CALUDE_abs_diff_properties_l1246_124627


namespace NUMINAMATH_CALUDE_least_square_tiles_l1246_124638

/-- Given a rectangular room with length 624 cm and width 432 cm, 
    the least number of square tiles of equal size required to cover the entire floor is 117. -/
theorem least_square_tiles (length width : ℕ) (h1 : length = 624) (h2 : width = 432) : 
  (length / (Nat.gcd length width)) * (width / (Nat.gcd length width)) = 117 := by
  sorry

end NUMINAMATH_CALUDE_least_square_tiles_l1246_124638


namespace NUMINAMATH_CALUDE_ratio_equality_counterexample_l1246_124689

theorem ratio_equality_counterexample (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : c ≠ 0) (h3 : a / b = c / d) : 
  ¬ ((a + d) / (b + c) = a / b) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_counterexample_l1246_124689


namespace NUMINAMATH_CALUDE_new_numbers_mean_l1246_124601

/-- Given 7 numbers with mean 36 and 3 new numbers making a total of 10 with mean 48,
    prove that the mean of the 3 new numbers is 76. -/
theorem new_numbers_mean (original_count : Nat) (new_count : Nat) 
  (original_mean : ℝ) (new_mean : ℝ) : 
  original_count = 7 →
  new_count = 3 →
  original_mean = 36 →
  new_mean = 48 →
  (original_count * original_mean + new_count * 
    ((original_count + new_count) * new_mean - original_count * original_mean) / new_count) / 
    new_count = 76 := by
  sorry

end NUMINAMATH_CALUDE_new_numbers_mean_l1246_124601


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1246_124604

theorem smaller_number_in_ratio (a b d x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / b → x * y = d → 
  x = Real.sqrt ((a * d) / b) ∧ x < y := by sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1246_124604


namespace NUMINAMATH_CALUDE_count_distinct_digit_numbers_l1246_124610

/-- The number of four-digit numbers with distinct digits, including numbers beginning with zero -/
def distinctDigitNumbers : ℕ :=
  10 * 9 * 8 * 7

/-- Theorem stating that the number of four-digit numbers with distinct digits,
    including numbers beginning with zero, is equal to 5040 -/
theorem count_distinct_digit_numbers :
  distinctDigitNumbers = 5040 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_digit_numbers_l1246_124610


namespace NUMINAMATH_CALUDE_min_value_theorem_l1246_124629

theorem min_value_theorem (a b : ℝ) 
  (h : ∀ x : ℝ, Real.log (x + 1) - (a + 2) * x ≤ b - 2) : 
  ∃ m : ℝ, m = 1 - Real.exp 1 ∧ ∀ y : ℝ, y = (b - 3) / (a + 2) → y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1246_124629


namespace NUMINAMATH_CALUDE_problem_solution_l1246_124658

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y ^ 2) 
  (h3 : x / 6 = 3 * y) : 
  x = 108 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1246_124658


namespace NUMINAMATH_CALUDE_custom_mul_five_three_l1246_124685

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mul_five_three : custom_mul 5 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_five_three_l1246_124685


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1246_124667

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is y = -√5/2 * x, then its eccentricity is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 5 / 2) : 
  Real.sqrt (a^2 + b^2) / a = 3/2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1246_124667


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1246_124683

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(a,1) is symmetric to point A'(5,b) with respect to the origin, prove that a + b = -6 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1246_124683


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1246_124659

theorem complex_expression_equality : ∀ (i : ℂ), i^2 = -1 →
  (2 + i) / (1 - i) - (1 - i) = -1/2 + 5/2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1246_124659


namespace NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l1246_124628

theorem cubic_greater_than_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l1246_124628


namespace NUMINAMATH_CALUDE_three_glass_bottles_weight_l1246_124674

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℝ := sorry

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℝ := sorry

/-- The total weight of 4 glass bottles and 5 plastic bottles is 1050 grams -/
axiom total_weight : 4 * glass_bottle_weight + 5 * plastic_bottle_weight = 1050

/-- A glass bottle is 150 grams heavier than a plastic bottle -/
axiom weight_difference : glass_bottle_weight = plastic_bottle_weight + 150

/-- The weight of 3 glass bottles is 600 grams -/
theorem three_glass_bottles_weight : 3 * glass_bottle_weight = 600 := by sorry

end NUMINAMATH_CALUDE_three_glass_bottles_weight_l1246_124674


namespace NUMINAMATH_CALUDE_frisbee_sales_theorem_l1246_124614

/-- Represents the total number of frisbees sold -/
def total_frisbees : ℕ := 60

/-- Represents the number of $3 frisbees sold -/
def frisbees_3 : ℕ := 36

/-- Represents the number of $4 frisbees sold -/
def frisbees_4 : ℕ := 24

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 204

/-- Theorem stating that the total number of frisbees sold is 60 -/
theorem frisbee_sales_theorem :
  (frisbees_3 * 3 + frisbees_4 * 4 = total_receipts) ∧
  (frisbees_4 ≥ 24) ∧
  (total_frisbees = frisbees_3 + frisbees_4) :=
by sorry

end NUMINAMATH_CALUDE_frisbee_sales_theorem_l1246_124614


namespace NUMINAMATH_CALUDE_total_pupils_across_schools_l1246_124635

theorem total_pupils_across_schools (
  girls_A boys_A girls_B boys_B girls_C boys_C : ℕ
) (h1 : girls_A = 542) (h2 : boys_A = 387)
  (h3 : girls_B = 713) (h4 : boys_B = 489)
  (h5 : girls_C = 628) (h6 : boys_C = 361) :
  girls_A + boys_A + girls_B + boys_B + girls_C + boys_C = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_across_schools_l1246_124635


namespace NUMINAMATH_CALUDE_thirtieth_sum_l1246_124639

/-- Represents the sum of elements in the nth set of a sequence where each set starts one more than
    the last element of the preceding set and has one more element than the one before it. -/
def T (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  n * (first + last) / 2

/-- The 30th sum in the sequence equals 13515. -/
theorem thirtieth_sum : T 30 = 13515 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_sum_l1246_124639


namespace NUMINAMATH_CALUDE_candy_division_l1246_124625

theorem candy_division (total_candies : ℕ) (num_groups : ℕ) (candies_per_group : ℕ) 
  (h1 : total_candies = 30)
  (h2 : num_groups = 10)
  (h3 : candies_per_group = total_candies / num_groups) :
  candies_per_group = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l1246_124625


namespace NUMINAMATH_CALUDE_triangle_side_length_l1246_124679

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  b + c = 2 * Real.sqrt 3 →
  A = π / 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1246_124679


namespace NUMINAMATH_CALUDE_convex_prism_right_iff_not_four_l1246_124611

/-- A convex n-sided prism with congruent lateral faces -/
structure ConvexPrism (n : ℕ) where
  /-- The prism is convex -/
  convex : Bool
  /-- The prism has n sides -/
  sides : Fin n
  /-- All lateral faces are congruent -/
  congruentLateralFaces : Bool

/-- A prism is right if all its lateral edges are perpendicular to its bases -/
def isRight (p : ConvexPrism n) : Prop := sorry

/-- Main theorem: A convex n-sided prism with congruent lateral faces is necessarily right if and only if n ≠ 4 -/
theorem convex_prism_right_iff_not_four (n : ℕ) (p : ConvexPrism n) :
  p.convex ∧ p.congruentLateralFaces → (isRight p ↔ n ≠ 4) := by sorry

end NUMINAMATH_CALUDE_convex_prism_right_iff_not_four_l1246_124611


namespace NUMINAMATH_CALUDE_mod_63_calculation_l1246_124603

theorem mod_63_calculation : ∃ (a b : ℤ), 
  (7 * a) % 63 = 1 ∧ 
  (13 * b) % 63 = 1 ∧ 
  (3 * a + 9 * b) % 63 = 48 := by
  sorry

end NUMINAMATH_CALUDE_mod_63_calculation_l1246_124603


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1246_124653

theorem solution_set_inequality (x : ℝ) : -x^2 + 2*x > 0 ↔ 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1246_124653


namespace NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l1246_124618

theorem divisibility_of_sum_and_powers (a b c : ℤ) : 
  (6 ∣ (a + b + c)) → (6 ∣ (a^5 + b^3 + c)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_and_powers_l1246_124618


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1246_124665

/-- Represents a theater with two rows of seats. -/
structure Theater :=
  (front_seats : Nat)
  (back_seats : Nat)

/-- Calculates the number of valid seating arrangements for two people in a theater. -/
def validArrangements (t : Theater) (middle_seats : Nat) : Nat :=
  sorry

/-- The theorem stating that the number of valid seating arrangements is 114. -/
theorem seating_arrangements_count (t : Theater) :
  t.front_seats = 9 ∧ t.back_seats = 8 →
  validArrangements t 3 = 114 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1246_124665


namespace NUMINAMATH_CALUDE_tangent_point_exists_min_sum_of_squares_l1246_124615

noncomputable section

-- Define the parabola C: x^2 = 2y
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the focus F(0, 1/2)
def focus : ℝ × ℝ := (0, 1/2)

-- Define the origin O(0, 0)
def origin : ℝ × ℝ := (0, 0)

-- Define a point M on the parabola in the first quadrant
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ M.1 > 0 ∧ M.2 > 0

-- Define the circle through M, F, and O with center Q
def circle_MFO (M Q : ℝ × ℝ) : Prop :=
  (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = (focus.1 - Q.1)^2 + (focus.2 - Q.2)^2 ∧
  (origin.1 - Q.1)^2 + (origin.2 - Q.2)^2 = (focus.1 - Q.1)^2 + (focus.2 - Q.2)^2

-- Distance from Q to the directrix is 3/4
def Q_to_directrix (Q : ℝ × ℝ) : Prop := Q.2 + 1/2 = 3/4

-- Theorem 1: Existence of point M where MQ is tangent to C
theorem tangent_point_exists :
  ∃ M : ℝ × ℝ, point_on_parabola M ∧
  ∃ Q : ℝ × ℝ, circle_MFO M Q ∧ Q_to_directrix Q ∧
  (M.1 = Real.sqrt 2 ∧ M.2 = 1) :=
sorry

-- Define the line l: y = kx + 1/4
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1/4

-- Theorem 2: Minimum value of |AB|^2 + |DE|^2
theorem min_sum_of_squares (k : ℝ) (h : 1/2 ≤ k ∧ k ≤ 2) :
  ∃ A B D E : ℝ × ℝ,
  point_on_parabola A ∧ point_on_parabola B ∧
  line k A.1 A.2 ∧ line k B.1 B.2 ∧
  (∃ Q : ℝ × ℝ, circle_MFO (Real.sqrt 2, 1) Q ∧ Q_to_directrix Q ∧
    line k D.1 D.2 ∧ line k E.1 E.2 ∧
    circle_MFO D Q ∧ circle_MFO E Q) ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (D.1 - E.1)^2 + (D.2 - E.2)^2 ≥ 13/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_exists_min_sum_of_squares_l1246_124615


namespace NUMINAMATH_CALUDE_sealant_cost_per_square_foot_l1246_124648

/-- Calculates the cost per square foot of sealant for a deck -/
theorem sealant_cost_per_square_foot
  (length : ℝ)
  (width : ℝ)
  (construction_cost_per_sqft : ℝ)
  (total_paid : ℝ)
  (h1 : length = 30)
  (h2 : width = 40)
  (h3 : construction_cost_per_sqft = 3)
  (h4 : total_paid = 4800) :
  (total_paid - construction_cost_per_sqft * length * width) / (length * width) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sealant_cost_per_square_foot_l1246_124648


namespace NUMINAMATH_CALUDE_floor_of_4_7_l1246_124681

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l1246_124681


namespace NUMINAMATH_CALUDE_white_ball_probability_l1246_124687

/-- Represents the number of balls initially in the bag -/
def initial_balls : ℕ := 6

/-- Represents the total number of balls after adding the white ball -/
def total_balls : ℕ := initial_balls + 1

/-- Represents the number of white balls added -/
def white_balls : ℕ := 1

/-- The probability of extracting the white ball -/
def prob_white : ℚ := white_balls / total_balls

theorem white_ball_probability :
  prob_white = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_white_ball_probability_l1246_124687


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l1246_124677

/-- The weight of a marble statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) (cut_week1 : ℝ) (cut_week2 : ℝ) (cut_week3 : ℝ) : ℝ :=
  initial_weight * (1 - cut_week1) * (1 - cut_week2) * (1 - cut_week3)

/-- Theorem stating the final weight of the statue --/
theorem statue_weight_calculation :
  let initial_weight : ℝ := 180
  let cut_week1 : ℝ := 0.28
  let cut_week2 : ℝ := 0.18
  let cut_week3 : ℝ := 0.20
  final_statue_weight initial_weight cut_week1 cut_week2 cut_week3 = 85.0176 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_calculation_l1246_124677


namespace NUMINAMATH_CALUDE_product_of_roots_l1246_124642

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 50*x + 35 = 0) → 
  (∃ a b c : ℝ, x^3 - 15*x^2 + 50*x + 35 = (x - a) * (x - b) * (x - c) ∧ a * b * c = -35) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l1246_124642


namespace NUMINAMATH_CALUDE_square_configuration_l1246_124606

theorem square_configuration (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let x : ℝ := (a - Real.sqrt 2) / b
  2 * Real.sqrt 2 * x + x = 1 →
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_square_configuration_l1246_124606


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1246_124663

theorem cubic_equation_solution (x y : ℝ) (h1 : x^(3*y) = 27) (h2 : x = 3) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1246_124663


namespace NUMINAMATH_CALUDE_cartesian_angle_properties_l1246_124620

/-- An angle in the Cartesian coordinate system -/
structure CartesianAngle where
  /-- The x-coordinate of the point on the terminal side -/
  x : ℝ
  /-- The y-coordinate of the point on the terminal side -/
  y : ℝ

/-- Theorem about properties of a specific angle in the Cartesian coordinate system -/
theorem cartesian_angle_properties (α : CartesianAngle) 
  (h1 : α.x = -1) 
  (h2 : α.y = 2) : 
  (Real.sin α.y * Real.tan α.y = -4 * Real.sqrt 5 / 5) ∧ 
  ((Real.sin (α.y + Real.pi / 2) * Real.cos (7 * Real.pi / 2 - α.y) * Real.tan (2 * Real.pi - α.y)) / 
   (Real.sin (2 * Real.pi - α.y) * Real.tan (-α.y)) = -Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_cartesian_angle_properties_l1246_124620


namespace NUMINAMATH_CALUDE_complex_fraction_value_l1246_124690

theorem complex_fraction_value : 
  let i : ℂ := Complex.I
  (1 + Real.sqrt 3 * i)^2 / (Real.sqrt 3 * i - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l1246_124690


namespace NUMINAMATH_CALUDE_distributive_property_implies_fraction_additivity_l1246_124692

theorem distributive_property_implies_fraction_additivity 
  {a b c : ℝ} (h1 : c ≠ 0) (h2 : (a + b) * c = a * c + b * c) :
  (a + b) / c = a / c + b / c :=
sorry

end NUMINAMATH_CALUDE_distributive_property_implies_fraction_additivity_l1246_124692


namespace NUMINAMATH_CALUDE_equal_diagonal_polygon_is_quadrilateral_or_pentagon_l1246_124656

/-- A convex polygon with n sides and all diagonals equal -/
structure EqualDiagonalPolygon where
  n : ℕ
  sides : n ≥ 4
  convex : Bool
  all_diagonals_equal : Bool

/-- The set of quadrilaterals -/
def Quadrilaterals : Set EqualDiagonalPolygon :=
  {p : EqualDiagonalPolygon | p.n = 4}

/-- The set of pentagons -/
def Pentagons : Set EqualDiagonalPolygon :=
  {p : EqualDiagonalPolygon | p.n = 5}

theorem equal_diagonal_polygon_is_quadrilateral_or_pentagon 
  (F : EqualDiagonalPolygon) (h_convex : F.convex = true) 
  (h_diag : F.all_diagonals_equal = true) :
  F ∈ Quadrilaterals ∪ Pentagons :=
sorry

end NUMINAMATH_CALUDE_equal_diagonal_polygon_is_quadrilateral_or_pentagon_l1246_124656


namespace NUMINAMATH_CALUDE_waffle_cooking_time_l1246_124666

/-- The time it takes Carla to cook a batch of waffles -/
def waffle_time : ℕ := sorry

/-- The time it takes Carla to cook a chicken-fried steak -/
def steak_time : ℕ := 6

/-- The total time it takes Carla to cook 3 steaks and a batch of waffles -/
def total_time : ℕ := 28

/-- Theorem stating that the time to cook a batch of waffles is 10 minutes -/
theorem waffle_cooking_time : waffle_time = 10 := by sorry

end NUMINAMATH_CALUDE_waffle_cooking_time_l1246_124666


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l1246_124626

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l1246_124626


namespace NUMINAMATH_CALUDE_max_point_of_product_l1246_124661

/-- Linear function f(x) -/
def f (x : ℝ) : ℝ := 2 * x + 2

/-- Linear function g(x) -/
def g (x : ℝ) : ℝ := -x - 3

/-- Product function h(x) = f(x) * g(x) -/
def h (x : ℝ) : ℝ := f x * g x

theorem max_point_of_product (x : ℝ) :
  f (-1) = 0 ∧ f 0 = 2 ∧ g 3 = 0 ∧ g 0 = -3 →
  ∃ (max_x : ℝ), max_x = -2 ∧ ∀ y, h y ≤ h max_x :=
sorry

end NUMINAMATH_CALUDE_max_point_of_product_l1246_124661


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1246_124664

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 + (m + 1) * x + (m + 2) ≥ 0) ↔ m ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1246_124664


namespace NUMINAMATH_CALUDE_new_person_weight_l1246_124624

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight new_average : ℝ) :
  n = 10 ∧ 
  replaced_weight = 45 ∧
  new_average = initial_weight + 3 →
  (n * new_average - (n * initial_weight - replaced_weight)) = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1246_124624


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l1246_124670

-- Define the number of each type of animal
def num_parrots : ℕ := 5
def num_dogs : ℕ := 3
def num_cats : ℕ := 4

-- Define the total number of animals
def total_animals : ℕ := num_parrots + num_dogs + num_cats

-- Define the function to calculate the number of arrangements
def num_arrangements : ℕ :=
  2 * (Nat.factorial num_parrots) * (Nat.factorial num_dogs) * (Nat.factorial num_cats)

-- Theorem statement
theorem animal_arrangement_count :
  num_arrangements = 34560 := by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l1246_124670


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1246_124678

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (1, 2) (x, 4) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1246_124678


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_sum_l1246_124673

theorem smallest_prime_factor_of_sum (n : ℕ) (m : ℕ) : 
  2 ∣ (2005^2007 + 2007^20015) ∧ 
  ∀ p : ℕ, p < 2 → p.Prime → ¬(p ∣ (2005^2007 + 2007^20015)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_sum_l1246_124673


namespace NUMINAMATH_CALUDE_cost_per_square_meter_is_two_l1246_124602

-- Define the lawn dimensions
def lawn_length : ℝ := 80
def lawn_width : ℝ := 60

-- Define the road width
def road_width : ℝ := 10

-- Define the total cost of traveling both roads
def total_cost : ℝ := 2600

-- Theorem to prove
theorem cost_per_square_meter_is_two :
  let road_area := (lawn_length * road_width + lawn_width * road_width) - road_width * road_width
  total_cost / road_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_square_meter_is_two_l1246_124602


namespace NUMINAMATH_CALUDE_bread_tear_ratio_l1246_124619

/-- Represents the number of bread slices -/
def num_slices : ℕ := 2

/-- Represents the total number of pieces after tearing -/
def total_pieces : ℕ := 8

/-- Represents the number of pieces each slice is torn into -/
def pieces_per_slice : ℕ := total_pieces / num_slices

/-- Proves that the ratio of pieces after the first tear to pieces after the second tear is 1:1 -/
theorem bread_tear_ratio :
  pieces_per_slice = pieces_per_slice → (pieces_per_slice : ℚ) / pieces_per_slice = 1 := by
  sorry

end NUMINAMATH_CALUDE_bread_tear_ratio_l1246_124619


namespace NUMINAMATH_CALUDE_carter_reads_30_pages_l1246_124640

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Theorem: Carter can read 30 pages in 1 hour -/
theorem carter_reads_30_pages : carter_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_reads_30_pages_l1246_124640


namespace NUMINAMATH_CALUDE_smallest_union_size_l1246_124605

theorem smallest_union_size (A B : Finset ℕ) : 
  Finset.card A = 30 → 
  Finset.card B = 20 → 
  Finset.card (A ∩ B) ≥ 10 → 
  Finset.card (A ∪ B) ≥ 40 ∧ 
  ∃ (C D : Finset ℕ), Finset.card C = 30 ∧ 
                      Finset.card D = 20 ∧ 
                      Finset.card (C ∩ D) ≥ 10 ∧ 
                      Finset.card (C ∪ D) = 40 :=
by sorry

end NUMINAMATH_CALUDE_smallest_union_size_l1246_124605


namespace NUMINAMATH_CALUDE_smallest_non_prime_without_small_factors_l1246_124684

theorem smallest_non_prime_without_small_factors :
  ∃ n : ℕ,
    n > 1 ∧
    ¬ (Nat.Prime n) ∧
    (∀ p : ℕ, Nat.Prime p → p < 10 → ¬ (p ∣ n)) ∧
    (∀ m : ℕ, m > 1 → ¬ (Nat.Prime m) → (∀ q : ℕ, Nat.Prime q → q < 10 → ¬ (q ∣ m)) → m ≥ n) ∧
    120 < n ∧
    n ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_without_small_factors_l1246_124684
