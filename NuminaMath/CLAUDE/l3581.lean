import Mathlib

namespace NUMINAMATH_CALUDE_abs_sum_minimum_l3581_358190

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 6| + |x + 7| ≥ 10 ∧ ∃ y : ℝ, |y + 3| + |y + 6| + |y + 7| = 10 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l3581_358190


namespace NUMINAMATH_CALUDE_problem_l3581_358178

def f (m : ℕ) (x : ℝ) : ℝ := |x - m| + |x|

theorem problem (m : ℕ) (h1 : m > 0) (h2 : ∃ x : ℝ, f m x < 2) :
  m = 1 ∧
  ∀ α β : ℝ, α > 1 → β > 1 → f m α + f m β = 6 → 4/α + 1/β ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_problem_l3581_358178


namespace NUMINAMATH_CALUDE_largest_number_problem_l3581_358112

theorem largest_number_problem (a b c : ℝ) : 
  a < b ∧ b < c →
  a + b + c = 77 →
  c - b = 9 →
  b - a = 5 →
  c = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l3581_358112


namespace NUMINAMATH_CALUDE_determine_xy_condition_l3581_358192

/-- Given two integers m and n, this theorem states the conditions under which 
    it's always possible to determine xy given x^m + y^m and x^n + y^n. -/
theorem determine_xy_condition (m n : ℤ) :
  (∀ x y : ℝ, ∃! (xy : ℝ), ∀ x' y' : ℝ, 
    x'^m + y'^m = x^m + y^m ∧ x'^n + y'^n = x^n + y^n → x' * y' = xy) ↔
  (∃ k t : ℤ, m = 2*k + 1 ∧ n = 2*t*(2*k + 1) ∧ t > 0) :=
sorry

end NUMINAMATH_CALUDE_determine_xy_condition_l3581_358192


namespace NUMINAMATH_CALUDE_line_symmetry_l3581_358132

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Returns true if two lines are symmetric about the y-axis -/
def symmetricAboutYAxis (l1 l2 : Line) : Prop :=
  l1.slope = -l2.slope ∧ l1.intercept = l2.intercept

theorem line_symmetry (l1 l2 : Line) :
  l1.slope = 2 ∧ l1.intercept = 3 →
  symmetricAboutYAxis l1 l2 →
  l2.slope = -2 ∧ l2.intercept = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3581_358132


namespace NUMINAMATH_CALUDE_total_pepper_weight_l3581_358157

-- Define the weights of green and red peppers
def green_peppers : ℝ := 0.33
def red_peppers : ℝ := 0.33

-- Theorem stating the total weight of peppers
theorem total_pepper_weight : green_peppers + red_peppers = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_total_pepper_weight_l3581_358157


namespace NUMINAMATH_CALUDE_remove_six_maximizes_probability_l3581_358135

def original_list : List ℤ := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def is_valid_pair (x y : ℤ) : Prop :=
  x ∈ original_list ∧ y ∈ original_list ∧ x ≠ y ∧ x + y = 12

def count_valid_pairs (removed : ℤ) : ℕ :=
  (original_list.filter (λ x => x ≠ removed)).length.choose 2

theorem remove_six_maximizes_probability :
  ∀ n ∈ original_list, count_valid_pairs 6 ≥ count_valid_pairs n :=
by sorry

end NUMINAMATH_CALUDE_remove_six_maximizes_probability_l3581_358135


namespace NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l3581_358196

-- Define the annual inflation rate
def annual_inflation_rate : ℝ := 0.015

-- Define the nominal annual interest rate
def nominal_interest_rate : ℝ := 0.07

-- Theorem for two-year inflation rate
theorem two_year_inflation_rate : 
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 3.0225 := by sorry

-- Theorem for real yield of bank deposit
theorem real_yield_bank_deposit : 
  ((1 + nominal_interest_rate)^2 / (1 + ((1 + annual_inflation_rate)^2 - 1)) - 1) * 100 = 11.13 := by sorry

end NUMINAMATH_CALUDE_two_year_inflation_rate_real_yield_bank_deposit_l3581_358196


namespace NUMINAMATH_CALUDE_wilfred_wednesday_carrots_l3581_358161

/-- The number of carrots Wilfred ate on Tuesday -/
def tuesday_carrots : ℕ := 4

/-- The number of carrots Wilfred ate on Thursday -/
def thursday_carrots : ℕ := 5

/-- The total number of carrots Wilfred wants to eat from Tuesday to Thursday -/
def total_carrots : ℕ := 15

/-- The number of carrots Wilfred ate on Wednesday -/
def wednesday_carrots : ℕ := total_carrots - tuesday_carrots - thursday_carrots

theorem wilfred_wednesday_carrots : wednesday_carrots = 6 := by
  sorry

end NUMINAMATH_CALUDE_wilfred_wednesday_carrots_l3581_358161


namespace NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l3581_358150

/-- Represents a table with crosses --/
structure CrossTable (m n : ℕ) :=
  (crosses : Fin m → Fin n → Bool)

/-- Checks if a row has an odd number of crosses --/
def rowHasOddCrosses (t : CrossTable m n) (i : Fin m) : Prop :=
  (Finset.filter (λ j => t.crosses i j) (Finset.univ : Finset (Fin n))).card % 2 = 1

/-- Checks if a column has an odd number of crosses --/
def colHasOddCrosses (t : CrossTable m n) (j : Fin n) : Prop :=
  (Finset.filter (λ i => t.crosses i j) (Finset.univ : Finset (Fin m))).card % 2 = 1

/-- Checks if all rows and columns have odd number of crosses --/
def allOddCrosses (t : CrossTable m n) : Prop :=
  (∀ i, rowHasOddCrosses t i) ∧ (∀ j, colHasOddCrosses t j)

/-- Counts the total number of crosses in the table --/
def totalCrosses (t : CrossTable m n) : ℕ :=
  (Finset.filter (λ (i, j) => t.crosses i j) (Finset.univ : Finset (Fin m × Fin n))).card

/-- Theorem: The maximum number of crosses in a 4x10 table with odd crosses in each row and column is 30 --/
theorem max_crosses_4x10 :
  (∃ t : CrossTable 4 10, allOddCrosses t ∧ totalCrosses t = 30) ∧
  (∀ t : CrossTable 4 10, allOddCrosses t → totalCrosses t ≤ 30) := by sorry

/-- Theorem: It's impossible to place crosses in a 5x10 table with odd crosses in each row and column --/
theorem impossible_5x10 :
  ¬ ∃ t : CrossTable 5 10, allOddCrosses t := by sorry

end NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l3581_358150


namespace NUMINAMATH_CALUDE_range_of_a_l3581_358186

-- Define the sets p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≥ 0
def q (a x : ℝ) : Prop := 2*a - 1 ≤ x ∧ x ≤ a + 3

-- Define the property that ¬p is a necessary but not sufficient condition for q
def not_p_necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, q a x → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ ¬(q a x))

-- State the theorem
theorem range_of_a (a : ℝ) :
  not_p_necessary_not_sufficient a ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3581_358186


namespace NUMINAMATH_CALUDE_park_area_is_3750_l3581_358176

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The perimeter of the park in meters -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- The area of the park in square meters -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- The cost of fencing per meter in dollars -/
def fencingCostPerMeter : ℝ := 0.80

/-- The total cost of fencing the park in dollars -/
def totalFencingCost (park : RectangularPark) : ℝ :=
  perimeter park * fencingCostPerMeter

theorem park_area_is_3750 (park : RectangularPark) 
    (h : totalFencingCost park = 200) : area park = 3750 := by
  sorry

end NUMINAMATH_CALUDE_park_area_is_3750_l3581_358176


namespace NUMINAMATH_CALUDE_eggs_needed_is_84_l3581_358138

/-- Represents the number of eggs in an omelette type -/
inductive OmeletteType
| ThreeEgg
| FourEgg

/-- Represents an hour's worth of orders -/
structure HourlyOrder where
  customerCount : Nat
  omeletteType : OmeletteType

/-- Calculates the total number of eggs needed for all omelettes -/
def totalEggsNeeded (orders : List HourlyOrder) : Nat :=
  orders.foldl (fun acc order =>
    acc + order.customerCount * match order.omeletteType with
      | OmeletteType.ThreeEgg => 3
      | OmeletteType.FourEgg => 4
  ) 0

theorem eggs_needed_is_84 (orders : List HourlyOrder) 
  (h1 : orders = [
    ⟨5, OmeletteType.ThreeEgg⟩, 
    ⟨7, OmeletteType.FourEgg⟩,
    ⟨3, OmeletteType.ThreeEgg⟩,
    ⟨8, OmeletteType.FourEgg⟩
  ]) : 
  totalEggsNeeded orders = 84 := by
  sorry

#eval totalEggsNeeded [
  ⟨5, OmeletteType.ThreeEgg⟩, 
  ⟨7, OmeletteType.FourEgg⟩,
  ⟨3, OmeletteType.ThreeEgg⟩,
  ⟨8, OmeletteType.FourEgg⟩
]

end NUMINAMATH_CALUDE_eggs_needed_is_84_l3581_358138


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3581_358133

theorem largest_divisor_of_n (n : ℕ+) (h : 50 ∣ n^2) : 5 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3581_358133


namespace NUMINAMATH_CALUDE_maglev_train_speed_l3581_358118

/-- Proves that the average speed of a maglev train is 225 km/h given specific conditions --/
theorem maglev_train_speed :
  ∀ (subway_speed : ℝ),
    subway_speed > 0 →
    let maglev_speed := 6.25 * subway_speed
    let distance := 30
    let subway_time := distance / subway_speed
    let maglev_time := distance / maglev_speed
    subway_time - maglev_time = 0.7 →
    maglev_speed = 225 := by
  sorry

#check maglev_train_speed

end NUMINAMATH_CALUDE_maglev_train_speed_l3581_358118


namespace NUMINAMATH_CALUDE_farm_sheep_ratio_l3581_358100

/-- Proves that the ratio of sheep sold to total sheep is 2:3 given the farm conditions --/
theorem farm_sheep_ratio :
  ∀ (goats sheep sold_sheep : ℕ) (sale_amount : ℚ),
    goats + sheep = 360 →
    goats * 7 = sheep * 5 →
    sale_amount = 7200 →
    sale_amount = (goats / 2) * 40 + sold_sheep * 30 →
    sold_sheep / sheep = 2 / 3 :=
by sorry


end NUMINAMATH_CALUDE_farm_sheep_ratio_l3581_358100


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3581_358105

theorem complex_expression_simplification :
  Real.sqrt 2 * (Real.sqrt 6 - Real.sqrt 12) + (Real.sqrt 3 + 1)^2 + 12 / Real.sqrt 6 = 4 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3581_358105


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_l3581_358185

theorem inscribed_circle_distance (a b : ℝ) (h1 : a = 6) (h2 : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let m := s - b
  2 * Real.sqrt ((a^2 + m^2 - 2 * a * m * (a / c)) / 5) = 2 * Real.sqrt (29 / 5) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_l3581_358185


namespace NUMINAMATH_CALUDE_power_of_product_l3581_358187

theorem power_of_product (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3581_358187


namespace NUMINAMATH_CALUDE_dividend_calculation_l3581_358148

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 8) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 141 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3581_358148


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l3581_358155

/-- Linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (2, f 2)

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (-1, f (-1))

/-- y₁ coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ coordinate of point B -/
def y₂ : ℝ := B.2

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l3581_358155


namespace NUMINAMATH_CALUDE_hayley_close_friends_l3581_358189

/-- The number of stickers Hayley has -/
def total_stickers : ℕ := 72

/-- The number of stickers each friend receives -/
def stickers_per_friend : ℕ := 8

/-- Hayley's number of close friends -/
def num_friends : ℕ := total_stickers / stickers_per_friend

theorem hayley_close_friends : num_friends = 9 := by
  sorry

end NUMINAMATH_CALUDE_hayley_close_friends_l3581_358189


namespace NUMINAMATH_CALUDE_square_area_ratio_l3581_358116

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (3 * y)^2 / (9 * y)^2 = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3581_358116


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3581_358140

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x - 35 = (x - 7)*(x + 5)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3581_358140


namespace NUMINAMATH_CALUDE_largest_number_l3581_358104

theorem largest_number (a b c d e : ℝ) :
  a = (7 * 8)^(1/4)^(1/2) →
  b = (8 * 7^(1/3))^(1/4) →
  c = (7 * 8^(1/4))^(1/2) →
  d = (7 * 8^(1/4))^(1/3) →
  e = (8 * 7^(1/3))^(1/4) →
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3581_358104


namespace NUMINAMATH_CALUDE_inverse_composition_problem_l3581_358117

def f : Fin 6 → Fin 6
| 1 => 4
| 2 => 5
| 3 => 3
| 4 => 2
| 5 => 1
| 6 => 6

theorem inverse_composition_problem (h : Function.Bijective f) :
  (Function.invFun f) ((Function.invFun f) ((Function.invFun f) 2)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_problem_l3581_358117


namespace NUMINAMATH_CALUDE_race_head_start_l3581_358109

/-- Given two runners A and B, where A's speed is 20/19 times B's speed,
    the head start fraction that A should give B for a dead heat is 1/20 of the race length. -/
theorem race_head_start (speedA speedB : ℝ) (length headStart : ℝ) :
  speedA = (20 / 19) * speedB →
  (length / speedA = (length - headStart) / speedB) →
  headStart = (1 / 20) * length :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l3581_358109


namespace NUMINAMATH_CALUDE_oranges_per_box_l3581_358164

def total_oranges : ℕ := 45
def num_boxes : ℕ := 9

theorem oranges_per_box : total_oranges / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l3581_358164


namespace NUMINAMATH_CALUDE_total_dogs_count_l3581_358169

/-- The number of boxes of stuffed toy dogs -/
def num_boxes : ℕ := 15

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 8

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem total_dogs_count : total_dogs = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_count_l3581_358169


namespace NUMINAMATH_CALUDE_bird_count_difference_l3581_358123

theorem bird_count_difference (monday_count tuesday_count wednesday_count : ℕ) : 
  monday_count = 70 →
  tuesday_count = monday_count / 2 →
  monday_count + tuesday_count + wednesday_count = 148 →
  wednesday_count - tuesday_count = 8 := by
sorry

end NUMINAMATH_CALUDE_bird_count_difference_l3581_358123


namespace NUMINAMATH_CALUDE_unique_base_solution_l3581_358174

/-- Given a natural number b ≥ 2, convert a number in base b to its decimal representation -/
def toDecimal (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- Given a natural number b ≥ 2, check if the equation 161_b + 134_b = 315_b holds -/
def checkEquation (b : ℕ) : Prop :=
  toDecimal 161 b + toDecimal 134 b = toDecimal 315 b

/-- The main theorem stating that 8 is the unique solution to the equation -/
theorem unique_base_solution :
  ∃! b : ℕ, b ≥ 2 ∧ checkEquation b ∧ b = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l3581_358174


namespace NUMINAMATH_CALUDE_points_collinear_l3581_358128

-- Define the function for log base 8
noncomputable def log8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- Define the function for log base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the line passing through the origin
def line_through_origin (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the theorem
theorem points_collinear (k a b : ℝ) 
  (ha : line_through_origin k a = log8 a)
  (hb : line_through_origin k b = log8 b)
  (hc : ∃ c, c = (a, log2 a))
  (hd : ∃ d, d = (b, log2 b)) :
  ∃ m, line_through_origin m a = log2 a ∧ 
       line_through_origin m b = log2 b ∧
       line_through_origin m 0 = 0 := by
  sorry


end NUMINAMATH_CALUDE_points_collinear_l3581_358128


namespace NUMINAMATH_CALUDE_compute_expression_l3581_358188

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3581_358188


namespace NUMINAMATH_CALUDE_advertising_sales_prediction_l3581_358143

-- Define the relationship between advertising expenditure and sales revenue
def advertising_sales_relation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 4

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ :=
  6.5 * x + 17.5

-- Theorem statement
theorem advertising_sales_prediction :
  ∀ x y : ℝ, advertising_sales_relation x y →
  (linear_regression 10 = 82.5) ∧
  (∀ x : ℝ, y = linear_regression x) :=
by sorry

end NUMINAMATH_CALUDE_advertising_sales_prediction_l3581_358143


namespace NUMINAMATH_CALUDE_sequence_constraint_l3581_358122

/-- An arithmetic sequence of four real numbers -/
structure ArithmeticSequence (x a₁ a₂ y : ℝ) : Prop where
  diff₁ : a₁ - x = a₂ - a₁
  diff₂ : a₂ - a₁ = y - a₂

/-- A geometric sequence of four real numbers -/
structure GeometricSequence (x b₁ b₂ y : ℝ) : Prop where
  ratio₁ : x ≠ 0
  ratio₂ : b₁ / x = b₂ / b₁
  ratio₃ : b₂ / b₁ = y / b₂

theorem sequence_constraint (x a₁ a₂ y b₁ b₂ : ℝ) 
  (h₁ : ArithmeticSequence x a₁ a₂ y) (h₂ : GeometricSequence x b₁ b₂ y) : 
  x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sequence_constraint_l3581_358122


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3581_358136

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/17) 
  (h2 : x - y = 1/51) : 
  x^2 - y^2 = 9/867 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3581_358136


namespace NUMINAMATH_CALUDE_right_triangle_area_l3581_358124

theorem right_triangle_area (a b : ℝ) (h_a : a = 25) (h_b : b = 20) :
  (1 / 2) * a * b = 250 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3581_358124


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3581_358108

theorem algebraic_simplification (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3581_358108


namespace NUMINAMATH_CALUDE_library_book_loan_l3581_358165

theorem library_book_loan (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 65 / 100)
  (h3 : final_books = 54)
  : ∃ (loaned_books : ℕ), loaned_books = 60 ∧ 
    (initial_books - final_books : ℚ) = (1 - return_rate) * loaned_books := by
  sorry

end NUMINAMATH_CALUDE_library_book_loan_l3581_358165


namespace NUMINAMATH_CALUDE_inequality_proof_l3581_358145

theorem inequality_proof (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c + b + a = 0) : b * c > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3581_358145


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l3581_358139

theorem mango_rate_calculation (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  apple_quantity = 8 →
  apple_rate = 70 →
  mango_quantity = 9 →
  total_paid = 1055 →
  (total_paid - apple_quantity * apple_rate) / mango_quantity = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l3581_358139


namespace NUMINAMATH_CALUDE_sum_of_intercepts_is_zero_l3581_358137

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = y^3 - 3*y^2 + 3*y

/-- Theorem stating that the sum of x and y intercepts is 0 -/
theorem sum_of_intercepts_is_zero (a b c d : ℝ) : 
  (parabola a 0) ∧ 
  (parabola 0 b) ∧ 
  (parabola 0 c) ∧ 
  (parabola 0 d) → 
  a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_is_zero_l3581_358137


namespace NUMINAMATH_CALUDE_max_daily_sales_l3581_358158

def P (t : ℕ) : ℝ :=
  if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else if 1 ≤ t ∧ t ≤ 24 then t + 20
  else 0

def Q (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 30 then -t + 40
  else 0

def y (t : ℕ) : ℝ := P t * Q t

theorem max_daily_sales :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → y t ≤ 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 1125 → t = 25) :=
by sorry

end NUMINAMATH_CALUDE_max_daily_sales_l3581_358158


namespace NUMINAMATH_CALUDE_third_difference_zero_implies_quadratic_l3581_358106

/-- A function from integers to real numbers -/
def IntFunction := ℤ → ℝ

/-- The third difference of a function -/
def thirdDifference (f : IntFunction) : IntFunction :=
  fun n => f (n + 3) - 3 * f (n + 2) + 3 * f (n + 1) - f n

/-- A function is quadratic if it can be expressed as a*n^2 + b*n + c for some real a, b, c -/
def isQuadratic (f : IntFunction) : Prop :=
  ∃ a b c : ℝ, ∀ n : ℤ, f n = a * n^2 + b * n + c

theorem third_difference_zero_implies_quadratic (f : IntFunction) 
  (h : ∀ n : ℤ, thirdDifference f n = 0) : 
  isQuadratic f := by
  sorry

end NUMINAMATH_CALUDE_third_difference_zero_implies_quadratic_l3581_358106


namespace NUMINAMATH_CALUDE_petya_vasya_journey_l3581_358180

/-- Represents the problem of Petya and Vasya's journey to the football match. -/
theorem petya_vasya_journey
  (distance : ℝ)
  (walking_speed : ℝ)
  (bicycle_speed_multiplier : ℝ)
  (late_time : ℝ)
  (h1 : distance = 4)
  (h2 : walking_speed = 4)
  (h3 : bicycle_speed_multiplier = 3)
  (h4 : late_time = 10)
  (h5 : distance / walking_speed * 60 - late_time = 50) :
  let bicycle_speed := walking_speed * bicycle_speed_multiplier
  let half_distance := distance / 2
  let walking_time := half_distance / walking_speed * 60
  let cycling_time := half_distance / bicycle_speed * 60
  let total_time := walking_time + cycling_time
  50 - total_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_journey_l3581_358180


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3581_358149

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (x + 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3581_358149


namespace NUMINAMATH_CALUDE_spelling_homework_time_l3581_358146

theorem spelling_homework_time (total_time math_time reading_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : reading_time = 27) :
  total_time - math_time - reading_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_spelling_homework_time_l3581_358146


namespace NUMINAMATH_CALUDE_total_fruits_eq_137_l3581_358195

/-- The number of fruits picked by George, Amelia, and Olivia --/
def total_fruits (george_oranges amelia_apples olivia_time olivia_rate_time olivia_rate_oranges olivia_rate_apples : ℕ) : ℕ :=
  let george_apples := amelia_apples + 5
  let amelia_oranges := george_oranges - 18
  let olivia_sets := olivia_time / olivia_rate_time
  let olivia_oranges := olivia_sets * olivia_rate_oranges
  let olivia_apples := olivia_sets * olivia_rate_apples
  (george_oranges + george_apples) + (amelia_oranges + amelia_apples) + (olivia_oranges + olivia_apples)

/-- Theorem stating the total number of fruits picked --/
theorem total_fruits_eq_137 :
  total_fruits 45 15 30 5 3 2 = 137 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_eq_137_l3581_358195


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l3581_358121

/-- 
Given a cubic polynomial y = ax³ + bx² + cx + d, 
if (1, y₁) and (-1, y₂) lie on its graph and y₁ - y₂ = -8, 
then a = -4.
-/
theorem cubic_polynomial_coefficient (a b c d y₁ y₂ : ℝ) : 
  y₁ = a + b + c + d → 
  y₂ = -a + b - c + d → 
  y₁ - y₂ = -8 → 
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l3581_358121


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3581_358198

theorem discount_percentage_proof (total_cost : ℝ) (num_shirts : ℕ) (discounted_price : ℝ) :
  total_cost = 60 ∧ num_shirts = 3 ∧ discounted_price = 12 →
  (1 - discounted_price / (total_cost / num_shirts)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3581_358198


namespace NUMINAMATH_CALUDE_division_line_exists_l3581_358162

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given 2000 distinct points in a plane, there exists a line that divides them into two equal sets -/
theorem division_line_exists (points : Finset Point) (h : points.card = 2000) :
  ∃ (a : ℝ), (points.filter (λ p => p.x < a)).card = 1000 ∧ (points.filter (λ p => p.x > a)).card = 1000 := by
  sorry

end NUMINAMATH_CALUDE_division_line_exists_l3581_358162


namespace NUMINAMATH_CALUDE_existence_of_abc_l3581_358160

theorem existence_of_abc (n k : ℕ) (hn : n > 20) (hk : k > 1) (hdiv : k^2 ∣ n) :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a * b + b * c + c * a :=
sorry

end NUMINAMATH_CALUDE_existence_of_abc_l3581_358160


namespace NUMINAMATH_CALUDE_derivative_special_function_l3581_358184

open Real

/-- The derivative of (1 + 8 cosh² x * ln(cosh x)) / (2 cosh² x) -/
theorem derivative_special_function (x : ℝ) :
  deriv (λ x => (1 + 8 * (cosh x)^2 * log (cosh x)) / (2 * (cosh x)^2)) x
  = (sinh x * (4 * (cosh x)^2 - 1)) / (cosh x)^3 :=
by sorry

end NUMINAMATH_CALUDE_derivative_special_function_l3581_358184


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_two_l3581_358166

def h (x : ℝ) : ℝ := 5 * x - 10

theorem h_zero_iff_b_eq_two :
  ∀ b : ℝ, h b = 0 ↔ b = 2 := by
sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_two_l3581_358166


namespace NUMINAMATH_CALUDE_M_perfect_square_divisors_l3581_358127

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The product M as defined in the problem -/
def M : ℕ := (factorial 1) * (factorial 2) * (factorial 3) * (factorial 4) * 
              (factorial 5) * (factorial 6) * (factorial 7) * (factorial 8) * (factorial 9)

/-- Count of perfect square divisors of a natural number -/
def count_perfect_square_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating that M has 672 perfect square divisors -/
theorem M_perfect_square_divisors : count_perfect_square_divisors M = 672 := by sorry

end NUMINAMATH_CALUDE_M_perfect_square_divisors_l3581_358127


namespace NUMINAMATH_CALUDE_cages_needed_proof_l3581_358114

def initial_gerbils : ℕ := 150
def sold_gerbils : ℕ := 98

theorem cages_needed_proof :
  initial_gerbils - sold_gerbils = 52 :=
by sorry

end NUMINAMATH_CALUDE_cages_needed_proof_l3581_358114


namespace NUMINAMATH_CALUDE_average_first_ten_even_numbers_l3581_358126

theorem average_first_ten_even_numbers :
  let first_ten_even : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  (first_ten_even.sum / first_ten_even.length : ℚ) = 11 := by
sorry

end NUMINAMATH_CALUDE_average_first_ten_even_numbers_l3581_358126


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l3581_358125

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  n = 15625 ∧                 -- the specific number
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) :=                 -- least such number
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l3581_358125


namespace NUMINAMATH_CALUDE_cube_sum_odd_numbers_l3581_358110

theorem cube_sum_odd_numbers (m : ℕ) : 
  (∃ k : ℕ, k ≥ m^2 - m + 1 ∧ k ≤ m^2 + m - 1 ∧ k = 2015) → m = 45 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_odd_numbers_l3581_358110


namespace NUMINAMATH_CALUDE_horner_evaluation_approx_l3581_358103

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List Float) (x : Float) : Float :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial function f(x) = 1 + x + 0.5x^2 + 0.16667x^3 + 0.04167x^4 + 0.00833x^5 -/
def f (x : Float) : Float :=
  horner [1, 1, 0.5, 0.16667, 0.04167, 0.00833] x

theorem horner_evaluation_approx :
  (f (-0.2) - 0.81873).abs < 1e-5 := by
  sorry

end NUMINAMATH_CALUDE_horner_evaluation_approx_l3581_358103


namespace NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l3581_358101

/-- A rectangle with a tangent circle passing through one vertex -/
structure TangentCircleRectangle where
  /-- Length of the rectangle -/
  l : ℝ
  /-- Width of the rectangle -/
  w : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- The circle is tangent to two adjacent sides of the rectangle -/
  tangent : l = 2 * r
  /-- The circle passes through the opposite corner -/
  passes_through : w = r

/-- The area of a rectangle with a tangent circle passing through one vertex is 2r² -/
theorem tangent_circle_rectangle_area (rect : TangentCircleRectangle) : 
  rect.l * rect.w = 2 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_rectangle_area_l3581_358101


namespace NUMINAMATH_CALUDE_polygon_with_150_degree_interior_angles_is_12_gon_polygon_with_14_diagonals_has_900_degree_sum_l3581_358163

-- Define a polygon
structure Polygon where
  sides : ℕ
  interiorAngle : ℝ
  diagonals : ℕ

-- Theorem 1
theorem polygon_with_150_degree_interior_angles_is_12_gon (P : Polygon) 
  (h : P.interiorAngle = 150) : P.sides = 12 := by
  sorry

-- Theorem 2
theorem polygon_with_14_diagonals_has_900_degree_sum (P : Polygon) 
  (h : P.diagonals = 14) : (P.sides - 2) * 180 = 900 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_150_degree_interior_angles_is_12_gon_polygon_with_14_diagonals_has_900_degree_sum_l3581_358163


namespace NUMINAMATH_CALUDE_smallest_block_with_360_hidden_l3581_358171

/-- Given a rectangular block made of unit cubes, this function calculates
    the number of hidden cubes when three surfaces are visible. -/
def hidden_cubes (l m n : ℕ) : ℕ := (l - 1) * (m - 1) * (n - 1)

/-- The total number of cubes in the rectangular block. -/
def total_cubes (l m n : ℕ) : ℕ := l * m * n

/-- Theorem stating that the smallest possible number of cubes in a rectangular block
    with 360 hidden cubes when three surfaces are visible is 560. -/
theorem smallest_block_with_360_hidden : 
  (∃ l m n : ℕ, 
    l > 1 ∧ m > 1 ∧ n > 1 ∧ 
    hidden_cubes l m n = 360 ∧
    (∀ l' m' n' : ℕ, 
      l' > 1 → m' > 1 → n' > 1 → 
      hidden_cubes l' m' n' = 360 → 
      total_cubes l m n ≤ total_cubes l' m' n')) ∧
  (∀ l m n : ℕ,
    l > 1 → m > 1 → n > 1 →
    hidden_cubes l m n = 360 →
    total_cubes l m n ≥ 560) := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_with_360_hidden_l3581_358171


namespace NUMINAMATH_CALUDE_angle_inequality_l3581_358154

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ ≥ 0) →
  Real.pi / 12 ≤ θ ∧ θ ≤ 5 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l3581_358154


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3581_358134

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  isArithmeticSequence a → a 6 + a 9 + a 12 = 48 → a 8 + a 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3581_358134


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3581_358120

theorem square_floor_tiles (diagonal_tiles : ℕ) (total_tiles : ℕ) : 
  diagonal_tiles = 37 → total_tiles = 361 → 
  (∃ (side_length : ℕ), 
    2 * side_length - 1 = diagonal_tiles ∧ 
    side_length * side_length = total_tiles) := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3581_358120


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3581_358152

theorem arithmetic_calculation : 6^2 - 4*5 + 2^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3581_358152


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3581_358172

theorem unique_solution_quadratic (n : ℝ) : 
  (n > 0 ∧ ∃! x : ℝ, 16 * x^2 + n * x + 4 = 0) ↔ n = 16 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3581_358172


namespace NUMINAMATH_CALUDE_odd_power_sum_divisible_l3581_358142

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ k : ℕ, k > 0 →
    (∃ m : ℤ, x^(2*k-1) + y^(2*k-1) = m * (x + y)) →
    (∃ n : ℤ, x^(2*k+1) + y^(2*k+1) = n * (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisible_l3581_358142


namespace NUMINAMATH_CALUDE_total_money_of_three_people_l3581_358113

/-- Given three people A, B, and C with some money between them, prove that their total amount is 400. -/
theorem total_money_of_three_people (a b c : ℕ) : 
  a + c = 300 →
  b + c = 150 →
  c = 50 →
  a + b + c = 400 := by
sorry

end NUMINAMATH_CALUDE_total_money_of_three_people_l3581_358113


namespace NUMINAMATH_CALUDE_junk_mail_calculation_l3581_358183

theorem junk_mail_calculation (blocks : ℕ) (houses_per_block : ℕ) (mail_per_house : ℕ)
  (h1 : blocks = 16)
  (h2 : houses_per_block = 17)
  (h3 : mail_per_house = 4) :
  blocks * houses_per_block * mail_per_house = 1088 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_calculation_l3581_358183


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_mod_11_and_7_l3581_358179

theorem greatest_three_digit_number_mod_11_and_7 :
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    n % 11 = 10 ∧ 
    n % 7 = 4 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 11 = 10 ∧ m % 7 = 4 → m ≤ n) ∧
    n = 956 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_mod_11_and_7_l3581_358179


namespace NUMINAMATH_CALUDE_base7_digit_sum_theorem_l3581_358115

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a * base7ToBase10 b)

/-- Adds two base-7 numbers --/
def addBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a + base7ToBase10 b)

/-- Sums the digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem base7_digit_sum_theorem : 
  sumDigitsBase7 (addBase7 (multiplyBase7 36 52) 20) = 23 := by sorry

end NUMINAMATH_CALUDE_base7_digit_sum_theorem_l3581_358115


namespace NUMINAMATH_CALUDE_product_repeating_decimal_and_fraction_l3581_358156

theorem product_repeating_decimal_and_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^n - x.floor) * 10 ≥ 6 ∧ (x * 10^n - x.floor) * 10 < 7) →
  x * (7/3) = 14/9 := by
sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_and_fraction_l3581_358156


namespace NUMINAMATH_CALUDE_area_perimeter_product_l3581_358193

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℕ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

/-- Represents a square on the grid -/
structure Square where
  E : Point
  F : Point
  G : Point
  H : Point

/-- The specific square EFGH from the problem -/
def EFGH : Square :=
  { E := { x := 1, y := 5 },
    F := { x := 5, y := 6 },
    G := { x := 6, y := 2 },
    H := { x := 2, y := 1 } }

/-- Theorem stating the product of area and perimeter of EFGH -/
theorem area_perimeter_product (s : Square) (h : s = EFGH) :
  (↑(squaredDistance s.E s.F) : ℝ) * (4 * Real.sqrt (↑(squaredDistance s.E s.F))) = 68 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_area_perimeter_product_l3581_358193


namespace NUMINAMATH_CALUDE_composition_equality_l3581_358199

variables (m n p q : ℝ)

def f (x : ℝ) : ℝ := m * x^2 + n * x

def g (x : ℝ) : ℝ := p * x + q

theorem composition_equality :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n :=
sorry

end NUMINAMATH_CALUDE_composition_equality_l3581_358199


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3581_358153

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 5 * n + 4 * n^3

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3581_358153


namespace NUMINAMATH_CALUDE_probability_of_three_given_sum_fourteen_l3581_358111

-- Define a type for die outcomes
def DieOutcome := Fin 6

-- Define a type for a set of four tosses
def FourTosses := Fin 4 → DieOutcome

-- Function to calculate the sum of four tosses
def sumTosses (tosses : FourTosses) : Nat :=
  (tosses 0).val + (tosses 1).val + (tosses 2).val + (tosses 3).val + 4

-- Function to check if a set of tosses contains at least one 3
def hasThree (tosses : FourTosses) : Prop :=
  ∃ i, (tosses i).val = 2

-- Theorem statement
theorem probability_of_three_given_sum_fourteen (tosses : FourTosses) :
  sumTosses tosses = 14 → hasThree tosses := by
  sorry

#check probability_of_three_given_sum_fourteen

end NUMINAMATH_CALUDE_probability_of_three_given_sum_fourteen_l3581_358111


namespace NUMINAMATH_CALUDE_product_nine_sum_zero_l3581_358182

theorem product_nine_sum_zero (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 9 →
  a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_nine_sum_zero_l3581_358182


namespace NUMINAMATH_CALUDE_total_calories_burned_l3581_358147

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 60

/-- The number of stairs in the first half of the staircase -/
def stairs_first_half : ℕ := 20

/-- The number of stairs in the second half of the staircase -/
def stairs_second_half : ℕ := 25

/-- The number of calories burned per stair in the first half -/
def calories_per_stair_first_half : ℕ := 3

/-- The number of calories burned per stair in the second half -/
def calories_per_stair_second_half : ℕ := 4

/-- The total number of stairs in the staircase -/
def total_stairs : ℕ := stairs_first_half + stairs_second_half

/-- Theorem stating the total calories burned by each player -/
theorem total_calories_burned :
  num_runs * (stairs_first_half * calories_per_stair_first_half +
              stairs_second_half * calories_per_stair_second_half) = 9600 :=
by sorry

end NUMINAMATH_CALUDE_total_calories_burned_l3581_358147


namespace NUMINAMATH_CALUDE_school_sections_theorem_l3581_358197

/-- Calculates the total number of sections needed in a school with given constraints -/
def totalSections (numBoys numGirls : ℕ) (maxBoysPerSection maxGirlsPerSection : ℕ) (numSubjects : ℕ) : ℕ :=
  let boySections := (numBoys + maxBoysPerSection - 1) / maxBoysPerSection * numSubjects
  let girlSections := (numGirls + maxGirlsPerSection - 1) / maxGirlsPerSection * numSubjects
  boySections + girlSections

/-- Theorem stating that the total number of sections is 87 under the given constraints -/
theorem school_sections_theorem :
  totalSections 408 192 24 16 3 = 87 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_theorem_l3581_358197


namespace NUMINAMATH_CALUDE_quadratic_root_l3581_358191

theorem quadratic_root (a b c : ℝ) : 
  (∃ d : ℝ, d ≥ 0 ∧ b = a + d ∧ c = a + 2*d) →  -- arithmetic sequence
  a ≤ b → b ≤ c → c ≤ 10 → c > 0 →  -- given inequalities
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →  -- exactly one root
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l3581_358191


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l3581_358130

theorem partnership_investment_ratio 
  (x : ℝ) 
  (m : ℝ) 
  (total_gain : ℝ) 
  (a_share : ℝ) 
  (h1 : total_gain = 27000) 
  (h2 : a_share = 9000) 
  (h3 : a_share = (1/3) * total_gain) 
  (h4 : (12*x) / (12*x + 12*x + 4*m*x) = 1/3) : 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l3581_358130


namespace NUMINAMATH_CALUDE_paris_cafe_contribution_l3581_358175

/-- Represents the currency exchange problem in the Paris cafe --/
theorem paris_cafe_contribution
  (pastry_cost : ℝ)
  (emily_dollars : ℝ)
  (exchange_rate : ℝ)
  (h_pastry_cost : pastry_cost = 8)
  (h_emily_dollars : emily_dollars = 10)
  (h_exchange_rate : exchange_rate = 1.20)
  : ∃ (berengere_contribution : ℝ),
    berengere_contribution = 0 ∧
    emily_dollars / exchange_rate + berengere_contribution ≥ pastry_cost :=
by sorry

end NUMINAMATH_CALUDE_paris_cafe_contribution_l3581_358175


namespace NUMINAMATH_CALUDE_network_sum_is_132_l3581_358159

/-- Represents a network of interconnected circles with integers --/
structure Network where
  size : Nat
  sum_of_ends : Nat
  given_numbers : Fin 2 → Nat

/-- The total sum of all integers in a completed network --/
def total_sum (n : Network) : Nat :=
  n.size * (n.given_numbers 0 + n.given_numbers 1) / 2

/-- Theorem stating the total sum of the specific network described in the problem --/
theorem network_sum_is_132 (n : Network) 
  (h_size : n.size = 24)
  (h_given : n.given_numbers = ![4, 7]) :
  total_sum n = 132 := by
  sorry

#eval total_sum { size := 24, sum_of_ends := 11, given_numbers := ![4, 7] }

end NUMINAMATH_CALUDE_network_sum_is_132_l3581_358159


namespace NUMINAMATH_CALUDE_triangle_abc_property_l3581_358144

theorem triangle_abc_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  b * Real.sin B - a * Real.sin A = c →
  -- Additional conditions
  c = Real.sqrt 3 →
  C = π / 3 →
  -- Conclusions
  B - A = π / 2 ∧
  (1 / 2 : Real) * a * c * Real.sin B = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_property_l3581_358144


namespace NUMINAMATH_CALUDE_watermelon_seeds_count_l3581_358129

/-- The number of seeds in each watermelon -/
def seeds_per_watermelon : ℕ := 345

/-- The number of watermelons -/
def number_of_watermelons : ℕ := 27

/-- The total number of seeds in all watermelons -/
def total_seeds : ℕ := seeds_per_watermelon * number_of_watermelons

theorem watermelon_seeds_count : total_seeds = 9315 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_count_l3581_358129


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3581_358173

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 + a - 1 = 0) → 
  (b^3 - 2*b^2 + b - 1 = 0) → 
  (c^3 - 2*c^2 + c - 1 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = -5) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3581_358173


namespace NUMINAMATH_CALUDE_prob_S7_eq_3_l3581_358102

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of a single draw -/
def drawOutcome (c : BallColor) : Int :=
  match c with
  | BallColor.Red => -1
  | BallColor.White => 1

/-- The probability of drawing a red ball -/
def probRed : ℚ := 2/3

/-- The probability of drawing a white ball -/
def probWhite : ℚ := 1/3

/-- The number of draws -/
def n : ℕ := 7

/-- The sum we're interested in -/
def targetSum : Int := 3

/-- The probability of getting the target sum after n draws -/
def probTargetSum (n : ℕ) (targetSum : Int) : ℚ :=
  sorry

theorem prob_S7_eq_3 :
  probTargetSum n targetSum = 28 / 3^6 :=
sorry

end NUMINAMATH_CALUDE_prob_S7_eq_3_l3581_358102


namespace NUMINAMATH_CALUDE_parabola_directrix_l3581_358170

/-- Given a parabola with equation y = (x^2 - 4x + 3) / 8, its directrix is y = -9/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = (x^2 - 4*x + 3) / 8) → 
  (∃ (d : ℝ), d = -9/8 ∧ 
    ∀ (p : ℝ × ℝ), 
      p.1 = x ∧ p.2 = y → 
      ∃ (f : ℝ × ℝ), 
        (f.1 - p.1)^2 + (f.2 - p.2)^2 = (p.2 - d)^2 ∧
        ∀ (q : ℝ × ℝ), q.2 = d → 
          (f.1 - p.1)^2 + (f.2 - p.2)^2 ≤ (q.1 - p.1)^2 + (q.2 - p.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3581_358170


namespace NUMINAMATH_CALUDE_no_real_solutions_for_composition_l3581_358141

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: If f(x) = ax^2 + bx + c is a quadratic function and f(x) = x has no real solutions,
    then f(f(x)) = x also has no real solutions -/
theorem no_real_solutions_for_composition
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c x ≠ x) :
  ∀ x : ℝ, quadratic_function a b c (quadratic_function a b c x) ≠ x :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_composition_l3581_358141


namespace NUMINAMATH_CALUDE_exchange_rate_problem_l3581_358151

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Proves that under the given exchange rate and spending conditions, 
    the sum of digits of the initial U.S. dollars is 8 -/
theorem exchange_rate_problem (d : ℕ) : 
  (8 * d) / 5 - 75 = d → sum_of_digits d = 8 := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_problem_l3581_358151


namespace NUMINAMATH_CALUDE_karen_rolls_count_l3581_358107

/-- The number of egg rolls Omar rolled -/
def omar_rolls : ℕ := 219

/-- The total number of egg rolls Omar and Karen rolled -/
def total_rolls : ℕ := 448

/-- The number of egg rolls Karen rolled -/
def karen_rolls : ℕ := total_rolls - omar_rolls

theorem karen_rolls_count : karen_rolls = 229 := by
  sorry

end NUMINAMATH_CALUDE_karen_rolls_count_l3581_358107


namespace NUMINAMATH_CALUDE_cars_on_remaining_days_l3581_358167

/-- Represents the number of cars passing through a toll booth on different days of the week -/
structure TollBoothTraffic where
  total : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  remaining_days : ℕ

/-- Theorem stating the number of cars passing through the toll booth on each remaining day -/
theorem cars_on_remaining_days (t : TollBoothTraffic) : 
  t.total = 450 ∧ 
  t.monday = 50 ∧ 
  t.tuesday = 50 ∧ 
  t.wednesday = 2 * t.monday ∧ 
  t.thursday = 2 * t.monday ∧ 
  t.remaining_days * 3 = t.total - (t.monday + t.tuesday + t.wednesday + t.thursday) → 
  t.remaining_days = 50 := by
  sorry


end NUMINAMATH_CALUDE_cars_on_remaining_days_l3581_358167


namespace NUMINAMATH_CALUDE_locus_is_equidistant_l3581_358181

/-- The locus of points equidistant from the x-axis and point F(0, 2) -/
def locus_equation (x y : ℝ) : Prop :=
  y = x^2 / 4 + 1

/-- A point is equidistant from the x-axis and F(0, 2) -/
def is_equidistant (x y : ℝ) : Prop :=
  abs y = Real.sqrt (x^2 + (y - 2)^2)

/-- Theorem: The locus equation represents points equidistant from x-axis and F(0, 2) -/
theorem locus_is_equidistant :
  ∀ x y : ℝ, locus_equation x y ↔ is_equidistant x y :=
by sorry

end NUMINAMATH_CALUDE_locus_is_equidistant_l3581_358181


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_l3581_358131

theorem shaded_area_rectangle (total_width total_height : ℝ)
  (small_rect_width small_rect_height : ℝ)
  (triangle1_base triangle1_height : ℝ)
  (triangle2_base triangle2_height : ℝ) :
  total_width = 8 ∧ total_height = 5 ∧
  small_rect_width = 4 ∧ small_rect_height = 2 ∧
  triangle1_base = 5 ∧ triangle1_height = 2 ∧
  triangle2_base = 3 ∧ triangle2_height = 2 →
  total_width * total_height -
  (2 * small_rect_width * small_rect_height +
   2 * (1/2 * triangle1_base * triangle1_height) +
   2 * (1/2 * triangle2_base * triangle2_height)) = 6.5 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_l3581_358131


namespace NUMINAMATH_CALUDE_shoes_lost_l3581_358168

theorem shoes_lost (initial_pairs : ℕ) (max_pairs_left : ℕ) (h1 : initial_pairs = 25) (h2 : max_pairs_left = 20) :
  initial_pairs * 2 - max_pairs_left * 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_shoes_lost_l3581_358168


namespace NUMINAMATH_CALUDE_ben_savings_days_l3581_358119

/-- Calculates the number of days elapsed given Ben's savings scenario --/
def days_elapsed (daily_start : ℕ) (daily_spend : ℕ) (final_amount : ℕ) : ℕ :=
  let daily_save := daily_start - daily_spend
  let d : ℕ := (final_amount - 10) / (2 * daily_save)
  d

/-- Theorem stating that the number of days elapsed is 7 --/
theorem ben_savings_days : days_elapsed 50 15 500 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ben_savings_days_l3581_358119


namespace NUMINAMATH_CALUDE_train_journey_time_l3581_358177

/-- Represents the train journey from A to B -/
structure TrainJourney where
  d : ℝ  -- Total distance
  v : ℝ  -- Initial speed
  t : ℝ  -- Total scheduled time

/-- The conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  j.d > 0 ∧ j.v > 0 ∧
  (j.d / (2 * j.v)) + 15 + (j.d / (8 * j.v)) = j.t

/-- The theorem stating that the total journey time is 40 minutes -/
theorem train_journey_time (j : TrainJourney) 
  (h : journey_conditions j) : j.t = 40 := by
  sorry

#check train_journey_time

end NUMINAMATH_CALUDE_train_journey_time_l3581_358177


namespace NUMINAMATH_CALUDE_karen_walnuts_l3581_358194

/-- The amount of nuts in cups added to the trail mix -/
def total_nuts : ℝ := 0.5

/-- The amount of almonds in cups added to the trail mix -/
def almonds : ℝ := 0.25

/-- The amount of walnuts in cups added to the trail mix -/
def walnuts : ℝ := total_nuts - almonds

theorem karen_walnuts : walnuts = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_karen_walnuts_l3581_358194
