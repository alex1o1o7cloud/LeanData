import Mathlib

namespace NUMINAMATH_CALUDE_train_length_calculation_l3424_342414

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time taken for the train to pass the jogger. -/
def train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed - jogger_speed) * passing_time - initial_distance

/-- Theorem stating that under the given conditions, the length of the train is 120 meters. -/
theorem train_length_calculation :
  let jogger_speed : ℝ := 9 * (1000 / 3600)  -- 9 kmph in m/s
  let train_speed : ℝ := 45 * (1000 / 3600)  -- 45 kmph in m/s
  let initial_distance : ℝ := 240  -- meters
  let passing_time : ℝ := 36  -- seconds
  train_length jogger_speed train_speed initial_distance passing_time = 120 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l3424_342414


namespace NUMINAMATH_CALUDE_mary_age_proof_l3424_342407

/-- Mary's current age -/
def mary_age : ℕ := 2

/-- Jay's current age -/
def jay_age : ℕ := mary_age + 7

theorem mary_age_proof :
  (∃ (j m : ℕ),
    j - 5 = (m - 5) + 7 ∧
    j + 5 = 2 * (m + 5) ∧
    m = mary_age) :=
by sorry

end NUMINAMATH_CALUDE_mary_age_proof_l3424_342407


namespace NUMINAMATH_CALUDE_remainder_9_pow_2048_mod_50_l3424_342475

theorem remainder_9_pow_2048_mod_50 : 9^2048 % 50 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9_pow_2048_mod_50_l3424_342475


namespace NUMINAMATH_CALUDE_square_roots_theorem_l3424_342456

theorem square_roots_theorem (a : ℝ) (n : ℝ) : 
  (2 * a + 3)^2 = n ∧ (a - 18)^2 = n → n = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l3424_342456


namespace NUMINAMATH_CALUDE_average_of_numbers_l3424_342431

def numbers : List ℝ := [2, 3, 4, 7, 9]

theorem average_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3424_342431


namespace NUMINAMATH_CALUDE_paint_six_boards_time_l3424_342409

/-- The minimum time required to paint both sides of wooden boards. -/
def paint_time (num_boards : ℕ) (paint_time_per_side : ℕ) (drying_time : ℕ) : ℕ :=
  2 * num_boards * paint_time_per_side

theorem paint_six_boards_time :
  paint_time 6 1 5 = 12 :=
by sorry

end NUMINAMATH_CALUDE_paint_six_boards_time_l3424_342409


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l3424_342420

theorem trig_expression_equals_four : 
  (1 / Real.sin (10 * π / 180)) - (Real.sqrt 3 / Real.cos (10 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l3424_342420


namespace NUMINAMATH_CALUDE_solve_for_A_l3424_342489

theorem solve_for_A : ∀ A : ℤ, A + 10 = 15 → A = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l3424_342489


namespace NUMINAMATH_CALUDE_function_property_implies_zero_l3424_342444

open Set
open Function

theorem function_property_implies_zero (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Ioo a b, f x + f (-x) = 0) : f (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_zero_l3424_342444


namespace NUMINAMATH_CALUDE_noodle_shop_solution_l3424_342419

/-- Represents the prices and sales of noodles in a shop -/
structure NoodleShop where
  dine_in_price : ℚ
  fresh_price : ℚ
  april_dine_in_sales : ℕ
  april_fresh_sales : ℕ
  may_fresh_price_decrease : ℚ
  may_fresh_sales_increase : ℚ
  may_total_sales_increase : ℚ

/-- Theorem stating the solution to the noodle shop problem -/
theorem noodle_shop_solution (shop : NoodleShop) : 
  (3 * shop.dine_in_price + 2 * shop.fresh_price = 31) →
  (4 * shop.dine_in_price + shop.fresh_price = 33) →
  (shop.april_dine_in_sales = 2500) →
  (shop.april_fresh_sales = 1500) →
  (shop.may_fresh_price_decrease = 3/4 * shop.may_total_sales_increase) →
  (shop.may_fresh_sales_increase = 5/2 * shop.may_total_sales_increase) →
  (shop.dine_in_price = 7) ∧ 
  (shop.fresh_price = 5) ∧ 
  (shop.may_total_sales_increase = 40/9) := by
  sorry


end NUMINAMATH_CALUDE_noodle_shop_solution_l3424_342419


namespace NUMINAMATH_CALUDE_line_slope_m_values_l3424_342474

theorem line_slope_m_values (m : ℝ) : 
  (∃ a b c : ℝ, (m^2 + m - 4) * a + (m + 4) * b + (2 * m + 1) = c ∧ 
   (m^2 + m - 4) = -(m + 4) ∧ (m^2 + m - 4) ≠ 0) → 
  m = 0 ∨ m = -2 := by
sorry

end NUMINAMATH_CALUDE_line_slope_m_values_l3424_342474


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3424_342400

theorem tangent_line_to_ln_curve (b : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1/2 * x + b = Real.log x) ∧ 
  (∀ y : ℝ, y > 0 → 1/2 * y + b ≥ Real.log y)) → 
  b = Real.log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l3424_342400


namespace NUMINAMATH_CALUDE_max_value_of_f_l3424_342413

-- Define the function f
def f (x : ℝ) : ℝ := x * (6 - 2*x)^2

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 3 ∧ 
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f c) ∧
  f c = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3424_342413


namespace NUMINAMATH_CALUDE_inverse_direct_variation_l3424_342425

/-- Given positive real numbers x, y, and z satisfying the following conditions:
    1. x² and y vary inversely
    2. y and z vary directly
    3. y = 8 when x = 4
    4. z = 32 when x = 4
    Prove that z = 512 when x = 1 -/
theorem inverse_direct_variation (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x^2 * y = k)
  (h_direct : ∃ c : ℝ, ∀ y z, y / z = c)
  (h_y : y = 8 → x = 4)
  (h_z : z = 32 → x = 4) :
  x = 1 → z = 512 := by
  sorry

end NUMINAMATH_CALUDE_inverse_direct_variation_l3424_342425


namespace NUMINAMATH_CALUDE_can_collection_difference_l3424_342449

/-- Theorem: Difference in can collection between two days -/
theorem can_collection_difference
  (sarah_yesterday : ℝ)
  (lara_yesterday : ℝ)
  (alex_yesterday : ℝ)
  (sarah_today : ℝ)
  (lara_today : ℝ)
  (alex_today : ℝ)
  (h1 : sarah_yesterday = 50.5)
  (h2 : lara_yesterday = sarah_yesterday + 30.3)
  (h3 : alex_yesterday = 90.2)
  (h4 : sarah_today = 40.7)
  (h5 : lara_today = 70.5)
  (h6 : alex_today = 55.3) :
  (sarah_yesterday + lara_yesterday + alex_yesterday) -
  (sarah_today + lara_today + alex_today) = 55 := by
  sorry

end NUMINAMATH_CALUDE_can_collection_difference_l3424_342449


namespace NUMINAMATH_CALUDE_sqrt_of_square_root_three_plus_one_squared_l3424_342410

theorem sqrt_of_square_root_three_plus_one_squared :
  Real.sqrt ((Real.sqrt 3 + 1) ^ 2) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_root_three_plus_one_squared_l3424_342410


namespace NUMINAMATH_CALUDE_min_value_problem_l3424_342494

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) : 
  (1 / x + 1 / (3 * y)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3424_342494


namespace NUMINAMATH_CALUDE_dealership_sedan_sales_l3424_342477

/-- Represents the ratio of sports cars to sedans -/
structure CarRatio :=
  (sports : ℕ)
  (sedans : ℕ)

/-- Calculates the expected sedan sales given a car ratio and anticipated sports car sales -/
def expectedSedanSales (ratio : CarRatio) (anticipatedSportsCars : ℕ) : ℕ :=
  (anticipatedSportsCars * ratio.sedans) / ratio.sports

theorem dealership_sedan_sales :
  let ratio : CarRatio := ⟨3, 5⟩
  let anticipatedSportsCars : ℕ := 36
  expectedSedanSales ratio anticipatedSportsCars = 60 := by
  sorry

end NUMINAMATH_CALUDE_dealership_sedan_sales_l3424_342477


namespace NUMINAMATH_CALUDE_five_circles_theorem_l3424_342498

/-- A circle in a plane -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this problem

/-- A point in a plane -/
structure Point where
  -- We don't need to define the internal structure of a point for this problem

/-- Predicate to check if a point is on a circle -/
def PointOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is common to a list of circles -/
def CommonPoint (p : Point) (circles : List Circle) : Prop :=
  ∀ c ∈ circles, PointOnCircle p c

theorem five_circles_theorem (circles : List Circle) :
  circles.length = 5 →
  (∀ (subset : List Circle), subset.length = 4 ∧ subset ⊆ circles →
    ∃ (p : Point), CommonPoint p subset) →
  ∃ (p : Point), CommonPoint p circles := by
  sorry

end NUMINAMATH_CALUDE_five_circles_theorem_l3424_342498


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_range_l3424_342482

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio_range 
  (a : ℕ → ℝ) (q : ℝ) (h1 : is_geometric_sequence a) 
  (h2 : a 1 * (a 2 + a 3) = 6 * a 1 - 9) :
  (-1 - Real.sqrt 5) / 2 ≤ q ∧ q ≤ (-1 + Real.sqrt 5) / 2 ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_range_l3424_342482


namespace NUMINAMATH_CALUDE_mike_remaining_cards_l3424_342427

/-- Calculates the number of baseball cards Mike has after Sam's purchase -/
def remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem stating that Mike has 74 baseball cards after Sam's purchase -/
theorem mike_remaining_cards :
  remaining_cards 87 13 = 74 := by
  sorry

end NUMINAMATH_CALUDE_mike_remaining_cards_l3424_342427


namespace NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l3424_342490

theorem only_negative_three_less_than_negative_two :
  let numbers : List ℚ := [-3, -1/2, 0, 2]
  ∀ x ∈ numbers, x < -2 ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_only_negative_three_less_than_negative_two_l3424_342490


namespace NUMINAMATH_CALUDE_mary_total_spent_approx_l3424_342484

/-- Calculates the total amount Mary spent at the mall --/
def total_spent (shirt_price : ℝ) (shirt_tax : ℝ) 
                (jacket_price : ℝ) (jacket_discount : ℝ) (jacket_tax : ℝ) 
                (currency_rate : ℝ)
                (scarf_price : ℝ) (hat_price : ℝ) (accessories_tax : ℝ) : ℝ :=
  let shirt_total := shirt_price * (1 + shirt_tax)
  let jacket_discounted := jacket_price * (1 - jacket_discount)
  let jacket_total := jacket_discounted * (1 + jacket_tax) * currency_rate
  let accessories_total := (scarf_price + hat_price) * (1 + accessories_tax)
  shirt_total + jacket_total + accessories_total

/-- The theorem stating that Mary's total spent is approximately $49.13 --/
theorem mary_total_spent_approx :
  ∃ ε > 0, abs (total_spent 13.04 0.07 15.34 0.20 0.085 1.28 7.90 9.13 0.065 - 49.13) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_mary_total_spent_approx_l3424_342484


namespace NUMINAMATH_CALUDE_aaron_position_2015_l3424_342446

/-- Represents a point on a 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Defines Aaron's walking pattern -/
def walk (n : Nat) : Point :=
  sorry

/-- The theorem to be proved -/
theorem aaron_position_2015 : walk 2015 = Point.mk 22 13 := by
  sorry

end NUMINAMATH_CALUDE_aaron_position_2015_l3424_342446


namespace NUMINAMATH_CALUDE_joy_tape_problem_l3424_342405

/-- The initial amount of tape given field dimensions and leftover tape -/
def initial_tape (width length leftover : ℕ) : ℕ :=
  2 * (width + length) + leftover

/-- Theorem: Given a field 20 feet wide and 60 feet long, with 90 feet of tape left over after wrapping once, the initial amount of tape is 250 feet -/
theorem joy_tape_problem :
  initial_tape 20 60 90 = 250 := by
  sorry

end NUMINAMATH_CALUDE_joy_tape_problem_l3424_342405


namespace NUMINAMATH_CALUDE_committee_seating_arrangements_l3424_342452

/-- The number of distinct arrangements of chairs and stools -/
def distinct_arrangements (n_women : ℕ) (n_men : ℕ) : ℕ :=
  Nat.choose (n_women + n_men - 1) (n_men - 1)

/-- Theorem stating the number of distinct arrangements for the given problem -/
theorem committee_seating_arrangements :
  distinct_arrangements 12 3 = 91 := by
  sorry

end NUMINAMATH_CALUDE_committee_seating_arrangements_l3424_342452


namespace NUMINAMATH_CALUDE_total_money_found_l3424_342478

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 10

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 3

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 4

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 200

theorem total_money_found :
  (num_quarters : ℚ) * quarter_value +
  (num_dimes : ℚ) * dime_value +
  (num_nickels : ℚ) * nickel_value +
  (num_pennies : ℚ) * penny_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_money_found_l3424_342478


namespace NUMINAMATH_CALUDE_only_f₁_is_quadratic_l3424_342436

-- Define the four functions
def f₁ (x : ℝ) : ℝ := -3 * x^2
def f₂ (x : ℝ) : ℝ := 2 * x
def f₃ (x : ℝ) : ℝ := x + 1
def f₄ (x : ℝ) : ℝ := x^3

-- Define what it means for a function to be quadratic
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- State the theorem
theorem only_f₁_is_quadratic :
  is_quadratic f₁ ∧ ¬is_quadratic f₂ ∧ ¬is_quadratic f₃ ∧ ¬is_quadratic f₄ :=
sorry

end NUMINAMATH_CALUDE_only_f₁_is_quadratic_l3424_342436


namespace NUMINAMATH_CALUDE_table_and_chair_price_l3424_342408

/-- The price of a chair in dollars -/
def chair_price : ℝ := by sorry

/-- The price of a table in dollars -/
def table_price : ℝ := 52.5

/-- The relation between chair and table prices -/
axiom price_relation : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

theorem table_and_chair_price : table_price + chair_price = 60 := by sorry

end NUMINAMATH_CALUDE_table_and_chair_price_l3424_342408


namespace NUMINAMATH_CALUDE_solution_properties_l3424_342476

def is_valid_solution (x y z : ℕ+) : Prop :=
  x.val + y.val + z.val = 2013

def count_solutions : ℕ := sorry

def count_solutions_x_eq_y : ℕ := sorry

def max_product_solution : ℕ+ × ℕ+ × ℕ+ := sorry

theorem solution_properties :
  (count_solutions = 2023066) ∧
  (count_solutions_x_eq_y = 1006) ∧
  (max_product_solution = (⟨671, sorry⟩, ⟨671, sorry⟩, ⟨671, sorry⟩)) :=
by sorry

end NUMINAMATH_CALUDE_solution_properties_l3424_342476


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3424_342423

theorem binomial_coefficient_divisibility (p k : ℕ) : 
  Prime p → 1 ≤ k → k ≤ p - 1 → p ∣ Nat.choose p k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3424_342423


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3424_342479

/-- A regular hexagon inscribed in a semicircle -/
structure InscribedHexagon where
  /-- The diameter of the semicircle -/
  diameter : ℝ
  /-- One side of the hexagon lies along the diameter -/
  side_on_diameter : Bool
  /-- Two opposite vertices of the hexagon are on the semicircle -/
  vertices_on_semicircle : Bool

/-- The area of an inscribed hexagon -/
def area (h : InscribedHexagon) : ℝ := sorry

/-- Theorem: The area of a regular hexagon inscribed in a semicircle of diameter 1 is 3√3/26 -/
theorem inscribed_hexagon_area :
  ∀ (h : InscribedHexagon), h.diameter = 1 → h.side_on_diameter = true → h.vertices_on_semicircle = true →
  area h = 3 * Real.sqrt 3 / 26 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3424_342479


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3424_342469

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 2 + 35 / 99) ∧ (x = 233 / 99) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3424_342469


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3424_342471

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 2 * (x^3 - 2*x^2 + x) + 4 * (x^4 + 3*x^3 - x^2 + x) - 3 * (x - 5*x^3 + 2*x^5)
  ∃ (a b c d e : ℝ), expression = a*x^5 + b*x^4 + 29*x^3 + c*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3424_342471


namespace NUMINAMATH_CALUDE_sin_2010_degrees_l3424_342439

theorem sin_2010_degrees : Real.sin (2010 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2010_degrees_l3424_342439


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3424_342459

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) * i = b + i) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3424_342459


namespace NUMINAMATH_CALUDE_scooter_gain_percent_correct_l3424_342430

def scooter_gain_percent (purchase_price repair1 repair2 repair3 taxes maintenance selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair1 + repair2 + repair3 + taxes + maintenance
  let gain := selling_price - total_cost
  (gain / total_cost) * 100

theorem scooter_gain_percent_correct 
  (purchase_price repair1 repair2 repair3 taxes maintenance selling_price : ℚ) :
  scooter_gain_percent purchase_price repair1 repair2 repair3 taxes maintenance selling_price =
  ((selling_price - (purchase_price + repair1 + repair2 + repair3 + taxes + maintenance)) / 
   (purchase_price + repair1 + repair2 + repair3 + taxes + maintenance)) * 100 :=
by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_correct_l3424_342430


namespace NUMINAMATH_CALUDE_existence_of_triangle_l3424_342464

theorem existence_of_triangle (l : Fin 7 → ℝ) 
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    l i + l j > l k ∧ 
    l j + l k > l i ∧ 
    l k + l i > l j :=
sorry

end NUMINAMATH_CALUDE_existence_of_triangle_l3424_342464


namespace NUMINAMATH_CALUDE_estimate_grade_a_in_population_l3424_342426

def sample_data : List ℕ := [11, 10, 6, 15, 9, 16, 13, 12, 0, 8,
                             2, 8, 10, 17, 6, 13, 7, 5, 7, 3,
                             12, 10, 7, 11, 3, 6, 8, 14, 15, 12]

def is_grade_a (m : ℕ) : Bool := m ≥ 10

def count_grade_a (data : List ℕ) : ℕ :=
  data.filter is_grade_a |>.length

def sample_size : ℕ := 30

def total_population : ℕ := 1000

theorem estimate_grade_a_in_population :
  (count_grade_a sample_data : ℚ) / sample_size * total_population = 500 := by
  sorry

end NUMINAMATH_CALUDE_estimate_grade_a_in_population_l3424_342426


namespace NUMINAMATH_CALUDE_number_times_five_equals_hundred_l3424_342429

theorem number_times_five_equals_hundred (x : ℝ) : 5 * x = 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_times_five_equals_hundred_l3424_342429


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l3424_342483

theorem roots_of_quadratic (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l3424_342483


namespace NUMINAMATH_CALUDE_part_one_part_two_combined_theorem_l3424_342485

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a m : ℝ) :
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) →
  a = 2 ∧ m = 3 := by sorry

-- Part II
theorem part_two (t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) := by sorry

-- Combined theorem
theorem combined_theorem (a m t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) →
  (a = 2 ∧ m = 3) ∧
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_combined_theorem_l3424_342485


namespace NUMINAMATH_CALUDE_raisins_cranberries_fraction_l3424_342406

/-- Represents the quantities (in pounds) of each ingredient in the mixture -/
structure Quantities where
  raisins : ℕ
  almonds : ℕ
  cashews : ℕ
  walnuts : ℕ
  dried_apricots : ℕ
  dried_cranberries : ℕ

/-- Represents the prices (in dollars per pound) of each ingredient -/
structure Prices where
  raisins : ℕ
  almonds : ℕ
  cashews : ℕ
  walnuts : ℕ
  dried_apricots : ℕ
  dried_cranberries : ℕ

/-- Calculates the total cost of the mixture -/
def total_cost (q : Quantities) (p : Prices) : ℕ :=
  q.raisins * p.raisins + q.almonds * p.almonds + q.cashews * p.cashews +
  q.walnuts * p.walnuts + q.dried_apricots * p.dried_apricots + q.dried_cranberries * p.dried_cranberries

/-- Calculates the cost of raisins and dried cranberries combined -/
def raisins_cranberries_cost (q : Quantities) (p : Prices) : ℕ :=
  q.raisins * p.raisins + q.dried_cranberries * p.dried_cranberries

/-- Theorem stating that the fraction of the total cost that is the cost of raisins and dried cranberries is 19/107 -/
theorem raisins_cranberries_fraction (q : Quantities) (p : Prices)
  (h_quantities : q = { raisins := 5, almonds := 4, cashews := 3, walnuts := 2, dried_apricots := 4, dried_cranberries := 3 })
  (h_prices : p = { raisins := 2, almonds := 6, cashews := 8, walnuts := 10, dried_apricots := 5, dried_cranberries := 3 }) :
  (raisins_cranberries_cost q p : ℚ) / (total_cost q p) = 19 / 107 := by
  sorry

end NUMINAMATH_CALUDE_raisins_cranberries_fraction_l3424_342406


namespace NUMINAMATH_CALUDE_tip_percentage_lower_limit_l3424_342491

theorem tip_percentage_lower_limit 
  (meal_cost : ℝ) 
  (total_paid : ℝ) 
  (tip_percentage : ℝ → Prop) : 
  meal_cost = 35.50 →
  total_paid = 40.825 →
  (∀ x, tip_percentage x → x ≥ 15 ∧ x < 25) →
  total_paid = meal_cost + (meal_cost * (15 / 100)) :=
by sorry

end NUMINAMATH_CALUDE_tip_percentage_lower_limit_l3424_342491


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l3424_342462

theorem reciprocal_of_negative_one_fifth :
  ((-1 : ℚ) / 5)⁻¹ = -5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l3424_342462


namespace NUMINAMATH_CALUDE_plane_perpendicular_condition_l3424_342470

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perp : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_condition 
  (a : Line) (α β : Plane) :
  perpendicular a β ∧ parallel a α → perp α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_condition_l3424_342470


namespace NUMINAMATH_CALUDE_inequality_proof_l3424_342481

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_condition : x * y + y * z + z * x = x + y + z) : 
  1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) ≤ 1 ∧ 
  (1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3424_342481


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l3424_342480

theorem power_of_three_mod_five : 3^2040 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l3424_342480


namespace NUMINAMATH_CALUDE_alex_amount_l3424_342402

def total : ℚ := 972.45
def sam : ℚ := 325.67
def erica : ℚ := 214.29

theorem alex_amount : total - (sam + erica) = 432.49 := by
  sorry

end NUMINAMATH_CALUDE_alex_amount_l3424_342402


namespace NUMINAMATH_CALUDE_digit_equation_solution_l3424_342496

/-- Represents a base-ten digit -/
def Digit := Fin 10

/-- Checks if three digits are all different -/
def all_different (d1 d2 d3 : Digit) : Prop :=
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- Converts a pair of digits to a two-digit number -/
def to_two_digit (tens ones : Digit) : Nat :=
  10 * tens.val + ones.val

/-- Converts a digit to a three-digit number with all digits the same -/
def to_three_digit (d : Digit) : Nat :=
  111 * d.val

theorem digit_equation_solution :
  ∃ (V E A : Digit),
    all_different V E A ∧
    (to_two_digit V E) * (to_two_digit A E) = to_three_digit A ∧
    E.val + A.val + A.val + V.val = 26 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l3424_342496


namespace NUMINAMATH_CALUDE_recursive_sum_value_l3424_342458

def recursive_sum (n : ℕ) : ℚ :=
  if n = 0 then 3
  else (n + 3 : ℚ) + (1 / 3) * recursive_sum (n - 1)

theorem recursive_sum_value : 
  recursive_sum 3000 = 4504 - (1 / 4) * (1 - 1 / 3^2999) :=
by sorry

end NUMINAMATH_CALUDE_recursive_sum_value_l3424_342458


namespace NUMINAMATH_CALUDE_base_conversion_and_addition_l3424_342404

/-- Converts a number from base 8 to base 10 -/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10To7 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 7 -/
def addBase7 (a b : ℕ) : ℕ := sorry

theorem base_conversion_and_addition :
  addBase7 (base10To7 (base8To10 123)) 25 = 264 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_addition_l3424_342404


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3424_342438

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def original_number : ℕ := 12910000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation :=
  { coefficient := 1.291
    exponent := 7
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3424_342438


namespace NUMINAMATH_CALUDE_max_value_expression_l3424_342473

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26*a*b*c) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3424_342473


namespace NUMINAMATH_CALUDE_workers_count_l3424_342440

/-- Given a group of workers who collectively contribute 300,000 and would contribute 350,000 if each gave 50 more, prove that there are 1000 workers. -/
theorem workers_count (total : ℕ) (extra_total : ℕ) (extra_per_worker : ℕ) : 
  total = 300000 →
  extra_total = 350000 →
  extra_per_worker = 50 →
  ∃ (num_workers : ℕ), num_workers * (total / num_workers + extra_per_worker) = extra_total ∧ 
                        num_workers = 1000 := by
  sorry

end NUMINAMATH_CALUDE_workers_count_l3424_342440


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3424_342447

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 2*m - 7 = 0) → (n^2 + 2*n - 7 = 0) → m^2 + 3*m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3424_342447


namespace NUMINAMATH_CALUDE_integer_pairs_equation_difficulty_l3424_342488

theorem integer_pairs_equation_difficulty : ¬ ∃ (count : ℕ), 
  (∀ m n : ℤ, m^2 + n^2 = m*n + 3 → count > 0) ∧ 
  (∀ k : ℕ, k ≠ count → ¬(∀ m n : ℤ, m^2 + n^2 = m*n + 3 → k > 0)) :=
sorry

end NUMINAMATH_CALUDE_integer_pairs_equation_difficulty_l3424_342488


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_tan_value_l3424_342466

theorem pure_imaginary_implies_tan_value (θ : ℝ) :
  (Complex.I * (Complex.cos θ - 4/5) = Complex.sin θ - 3/5) →
  Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_tan_value_l3424_342466


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l3424_342428

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = 8x² --/
def original_parabola : Parabola := { a := 8, b := 0, c := 0 }

/-- The translation of 3 units left and 5 units down --/
def translation : Translation := { dx := -3, dy := -5 }

/-- Applies a translation to a parabola --/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx
    c := p.a * t.dx^2 + p.b * t.dx + p.c + t.dy }

theorem parabola_translation_theorem :
  apply_translation original_parabola translation = { a := 8, b := 48, c := -5 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l3424_342428


namespace NUMINAMATH_CALUDE_square_difference_pattern_l3424_342451

theorem square_difference_pattern (n : ℕ) (h : n ≥ 1) :
  (n + 2)^2 - n^2 = 4 * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l3424_342451


namespace NUMINAMATH_CALUDE_five_percent_of_255_l3424_342421

theorem five_percent_of_255 : 
  let percent_5 : ℝ := 0.05
  255 * percent_5 = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_five_percent_of_255_l3424_342421


namespace NUMINAMATH_CALUDE_ratio_sum_equality_l3424_342443

theorem ratio_sum_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 16)
  (sum_xyz : x^2 + y^2 + z^2 = 49)
  (sum_prod : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equality_l3424_342443


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l3424_342455

/-- An arithmetic progression is represented by its first term and common difference. -/
structure ArithmeticProgression where
  a : ℤ  -- First term
  d : ℤ  -- Common difference

/-- A term in an arithmetic progression. -/
def ArithmeticProgression.term (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  ap.a + n * ap.d

/-- Predicate to check if a number is a perfect square. -/
def is_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k * k

/-- An arithmetic progression contains a square. -/
def contains_square (ap : ArithmeticProgression) : Prop :=
  ∃ n : ℕ, is_square (ap.term n)

/-- An arithmetic progression contains infinitely many squares. -/
def contains_infinite_squares (ap : ArithmeticProgression) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ is_square (ap.term n)

/-- 
If an infinite arithmetic progression contains a square number, 
then it contains infinitely many square numbers.
-/
theorem arithmetic_progression_squares 
  (ap : ArithmeticProgression) 
  (h : contains_square ap) : 
  contains_infinite_squares ap :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l3424_342455


namespace NUMINAMATH_CALUDE_intersection_equals_A_intersection_is_empty_l3424_342435

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for the first question
theorem intersection_equals_A (m : ℝ) :
  A ∩ B m = A ↔ m ≤ -2 :=
sorry

-- Theorem for the second question
theorem intersection_is_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ 0 ≤ m :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_A_intersection_is_empty_l3424_342435


namespace NUMINAMATH_CALUDE_symmetry_of_abs_f_shifted_l3424_342461

-- Define a function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the property of |f(x)| being an even function
def abs_f_is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |f x| = |f (-x)|

-- State the theorem
theorem symmetry_of_abs_f_shifted (h : abs_f_is_even f) :
  ∀ y : ℝ, |f ((1 - y) - 1)| = |f ((1 + y) - 1)| :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_abs_f_shifted_l3424_342461


namespace NUMINAMATH_CALUDE_eighth_grade_percentage_combined_schools_combined_schools_eighth_grade_percentage_l3424_342465

theorem eighth_grade_percentage_combined_schools : ℝ → Prop :=
  fun p =>
    let pinecrest_total : ℕ := 160
    let mapleridge_total : ℕ := 250
    let pinecrest_eighth_percent : ℝ := 18
    let mapleridge_eighth_percent : ℝ := 22
    let pinecrest_eighth : ℝ := (pinecrest_eighth_percent / 100) * pinecrest_total
    let mapleridge_eighth : ℝ := (mapleridge_eighth_percent / 100) * mapleridge_total
    let total_eighth : ℝ := pinecrest_eighth + mapleridge_eighth
    let total_students : ℝ := pinecrest_total + mapleridge_total
    p = (total_eighth / total_students) * 100 ∧ p = 20

/-- The percentage of 8th grade students in both schools combined is 20%. -/
theorem combined_schools_eighth_grade_percentage :
  ∃ p, eighth_grade_percentage_combined_schools p :=
sorry

end NUMINAMATH_CALUDE_eighth_grade_percentage_combined_schools_combined_schools_eighth_grade_percentage_l3424_342465


namespace NUMINAMATH_CALUDE_workday_meetings_percentage_l3424_342411

def workday_hours : ℕ := 10
def minutes_per_hour : ℕ := 60
def first_meeting_duration : ℕ := 60
def second_meeting_duration : ℕ := 2 * first_meeting_duration
def third_meeting_duration : ℕ := first_meeting_duration / 2

def total_workday_minutes : ℕ := workday_hours * minutes_per_hour
def total_meeting_minutes : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

theorem workday_meetings_percentage :
  (total_meeting_minutes : ℚ) / (total_workday_minutes : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_workday_meetings_percentage_l3424_342411


namespace NUMINAMATH_CALUDE_square_of_complex_l3424_342432

theorem square_of_complex : (3 - Complex.I) ^ 2 = 8 - 6 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l3424_342432


namespace NUMINAMATH_CALUDE_ivan_pension_sufficient_for_ticket_l3424_342457

theorem ivan_pension_sufficient_for_ticket : 
  (149^6 - 199^3) / (149^4 + 199^2 + 199 * 149^2) > 22000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_pension_sufficient_for_ticket_l3424_342457


namespace NUMINAMATH_CALUDE_quadratic_sum_l3424_342433

/-- A quadratic function g(x) = dx^2 + ex + f passing through (1, 3) and (2, 0) with vertex at (3, -3) -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := λ x => d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (QuadraticFunction d e f 1 = 3) →
  (QuadraticFunction d e f 2 = 0) →
  (∀ x, QuadraticFunction d e f x ≥ QuadraticFunction d e f 3) →
  (QuadraticFunction d e f 3 = -3) →
  d + e + 2 * f = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3424_342433


namespace NUMINAMATH_CALUDE_constant_prime_sequence_l3424_342472

-- Define the sequence of prime numbers
def isPrimeSequence (p : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 1 → Nat.Prime (p n)

-- Define the recurrence relation
def satisfiesRecurrence (p : ℕ → ℕ) (k : ℤ) : Prop :=
  ∀ n, n ≥ 1 → p (n + 2) = p (n + 1) + p n + k

-- Theorem statement
theorem constant_prime_sequence
  (p : ℕ → ℕ) (k : ℤ)
  (h_prime : isPrimeSequence p)
  (h_recurrence : satisfiesRecurrence p k) :
  ∃ c, ∀ n, n ≥ 1 → p n = c ∧ Nat.Prime c :=
by sorry

end NUMINAMATH_CALUDE_constant_prime_sequence_l3424_342472


namespace NUMINAMATH_CALUDE_circle_symmetry_l3424_342487

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-2)^2 + (y+3)^2 = 2

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧
    (x - x₀ = x₀ - 2) ∧ (y - y₀ = y₀ + 3) ∧
    symmetry_line ((x + x₀) / 2) ((y + y₀) / 2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3424_342487


namespace NUMINAMATH_CALUDE_hardwood_flooring_area_l3424_342493

/-- Represents the dimensions of a rectangular area -/
structure RectangularArea where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular area -/
def area (r : RectangularArea) : ℝ := r.length * r.width

/-- Represents Nancy's bathroom -/
structure Bathroom where
  centralArea : RectangularArea
  hallway : RectangularArea

/-- The actual bathroom dimensions -/
def nancysBathroom : Bathroom :=
  { centralArea := { length := 10, width := 10 }
  , hallway := { length := 6, width := 4 } }

/-- Theorem: The total area of hardwood flooring in Nancy's bathroom is 124 square feet -/
theorem hardwood_flooring_area :
  area nancysBathroom.centralArea + area nancysBathroom.hallway = 124 := by
  sorry

end NUMINAMATH_CALUDE_hardwood_flooring_area_l3424_342493


namespace NUMINAMATH_CALUDE_expand_polynomial_l3424_342499

theorem expand_polynomial (x : ℝ) : (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3424_342499


namespace NUMINAMATH_CALUDE_parabola_vertex_l3424_342450

/-- The equation of a parabola is y^2 + 8y + 2x + 1 = 0. 
    This theorem proves that the vertex of the parabola is (7.5, -4). -/
theorem parabola_vertex (x y : ℝ) : 
  (y^2 + 8*y + 2*x + 1 = 0) → (x = 7.5 ∧ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3424_342450


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3424_342442

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3424_342442


namespace NUMINAMATH_CALUDE_calculation_difference_is_zero_l3424_342424

def salesTaxRate : ℝ := 0.08
def originalPrice : ℝ := 120.00
def mainDiscount : ℝ := 0.25
def additionalDiscount : ℝ := 0.10
def numberOfSweaters : ℕ := 4

def amyCalculation : ℝ :=
  numberOfSweaters * (originalPrice * (1 + salesTaxRate) * (1 - mainDiscount) * (1 - additionalDiscount))

def bobCalculation : ℝ :=
  numberOfSweaters * (originalPrice * (1 - mainDiscount) * (1 - additionalDiscount) * (1 + salesTaxRate))

theorem calculation_difference_is_zero :
  amyCalculation = bobCalculation :=
by sorry

end NUMINAMATH_CALUDE_calculation_difference_is_zero_l3424_342424


namespace NUMINAMATH_CALUDE_max_value_condition_l3424_342495

theorem max_value_condition (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 4, |x^2 - 4*x + 9 - 2*m| + 2*m ≤ 9) ∧ 
  (∃ x ∈ Set.Icc 0 4, |x^2 - 4*x + 9 - 2*m| + 2*m = 9) ↔ 
  m ≤ 7/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_condition_l3424_342495


namespace NUMINAMATH_CALUDE_mike_found_four_more_seashells_l3424_342448

/-- The number of seashells Mike initially found -/
def initial_seashells : ℝ := 6.0

/-- The total number of seashells Mike ended up with -/
def total_seashells : ℝ := 10

/-- The number of additional seashells Mike found -/
def additional_seashells : ℝ := total_seashells - initial_seashells

theorem mike_found_four_more_seashells : additional_seashells = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_mike_found_four_more_seashells_l3424_342448


namespace NUMINAMATH_CALUDE_chord_length_polar_curves_l3424_342416

/-- The length of the chord formed by the intersection of two curves in polar coordinates -/
theorem chord_length_polar_curves : 
  ∃ (ρ₁ ρ₂ : ℝ → ℝ) (θ₁ θ₂ : ℝ),
    (∀ θ, ρ₁ θ * Real.sin θ = 1) →
    (∀ θ, ρ₂ θ = 4 * Real.sin θ) →
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      x₁^2 + y₁^2 = (ρ₁ θ₁)^2 ∧
      x₂^2 + y₂^2 = (ρ₁ θ₂)^2 ∧
      x₁ = ρ₁ θ₁ * Real.cos θ₁ ∧
      y₁ = ρ₁ θ₁ * Real.sin θ₁ ∧
      x₂ = ρ₁ θ₂ * Real.cos θ₂ ∧
      y₂ = ρ₁ θ₂ * Real.sin θ₂ ∧
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_polar_curves_l3424_342416


namespace NUMINAMATH_CALUDE_insurance_payment_count_l3424_342454

/-- Calculates the number of insurance payments per year -/
def insurance_payments_per_year (quarterly_payment : ℕ) (annual_total : ℕ) : ℕ :=
  annual_total / quarterly_payment

/-- Proves that the number of insurance payments per year is 4 -/
theorem insurance_payment_count :
  insurance_payments_per_year 378 1512 = 4 := by
  sorry

end NUMINAMATH_CALUDE_insurance_payment_count_l3424_342454


namespace NUMINAMATH_CALUDE_vector_equality_l3424_342401

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points in the vector space
variable (A B C M O : V)

-- Define vectors as differences between points
def vec (P Q : V) : V := Q - P

-- State the theorem
theorem vector_equality :
  (vec A B + vec M B) + (vec B O + vec B C) + vec O M = vec A C :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_l3424_342401


namespace NUMINAMATH_CALUDE_simplify_expressions_l3424_342422

theorem simplify_expressions :
  ((-4 : ℝ)^2023 * (-0.25)^2024 = -0.25) ∧
  (23 * (-4/11 : ℝ) + (-5/11) * 23 - 23 * (2/11) = -23) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3424_342422


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l3424_342441

theorem other_solution_quadratic (x : ℚ) :
  (72 * (3/8)^2 + 37 = -95 * (3/8) + 12) →
  (72 * x^2 + 37 = -95 * x + 12) →
  (x ≠ 3/8) →
  x = 5/8 := by
sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l3424_342441


namespace NUMINAMATH_CALUDE_baseball_cost_calculation_l3424_342468

/-- The amount spent on marbles in dollars -/
def marbles_cost : ℚ := 9.05

/-- The amount spent on the football in dollars -/
def football_cost : ℚ := 4.95

/-- The total amount spent on toys in dollars -/
def total_cost : ℚ := 20.52

/-- The amount spent on the baseball in dollars -/
def baseball_cost : ℚ := total_cost - (marbles_cost + football_cost)

theorem baseball_cost_calculation :
  baseball_cost = 6.52 := by sorry

end NUMINAMATH_CALUDE_baseball_cost_calculation_l3424_342468


namespace NUMINAMATH_CALUDE_bus_driver_hours_l3424_342497

-- Define constants
def regular_rate : ℝ := 15
def regular_hours : ℝ := 40
def overtime_rate_factor : ℝ := 1.75
def total_compensation : ℝ := 976

-- Define functions
def overtime_rate : ℝ := regular_rate * overtime_rate_factor

def total_hours (overtime_hours : ℝ) : ℝ :=
  regular_hours + overtime_hours

def compensation (overtime_hours : ℝ) : ℝ :=
  regular_rate * regular_hours + overtime_rate * overtime_hours

-- Theorem to prove
theorem bus_driver_hours :
  ∃ (overtime_hours : ℝ),
    compensation overtime_hours = total_compensation ∧
    total_hours overtime_hours = 54 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_hours_l3424_342497


namespace NUMINAMATH_CALUDE_omar_coffee_cup_size_l3424_342403

/-- Represents the size of Omar's coffee cup in ounces -/
def coffee_cup_size : ℝ := 6

theorem omar_coffee_cup_size :
  ∀ (remaining_after_work : ℝ) (remaining_after_office : ℝ),
  remaining_after_work = coffee_cup_size - (1/4 * coffee_cup_size + 1/2 * coffee_cup_size) →
  remaining_after_office = remaining_after_work - 1 →
  remaining_after_office = 2 →
  coffee_cup_size = 6 := by
sorry

end NUMINAMATH_CALUDE_omar_coffee_cup_size_l3424_342403


namespace NUMINAMATH_CALUDE_no_x_term_l3424_342467

theorem no_x_term (m : ℝ) : (∀ x : ℝ, (x + m) * (x + 3) = x^2 + 3*m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_x_term_l3424_342467


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3424_342463

theorem candy_bar_cost (marvin_sales : ℕ) (tina_sales : ℕ) (price : ℚ) : 
  marvin_sales = 35 →
  tina_sales = 3 * marvin_sales →
  tina_sales * price = marvin_sales * price + 140 →
  price = 2 := by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3424_342463


namespace NUMINAMATH_CALUDE_prove_newly_added_groups_l3424_342460

/-- Represents the number of groups of students recently added to the class -/
def newly_added_groups : ℕ := 2

theorem prove_newly_added_groups :
  let tables : ℕ := 6
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_students : ℕ := 3 * bathroom_students
  let students_per_group : ℕ := 4
  let foreign_students : ℕ := 3 * 3  -- 3 each from 3 countries
  let total_students : ℕ := 47
  newly_added_groups = 
    (total_students - (tables * students_per_table + bathroom_students + canteen_students + foreign_students)) / students_per_group :=
by
  sorry

#check prove_newly_added_groups

end NUMINAMATH_CALUDE_prove_newly_added_groups_l3424_342460


namespace NUMINAMATH_CALUDE_shipping_weight_calculation_l3424_342486

/-- The maximum weight a shipping box can hold in pounds, given the initial number of plates,
    weight of each plate, and number of plates removed. -/
def max_shipping_weight (initial_plates : ℕ) (plate_weight : ℕ) (removed_plates : ℕ) : ℚ :=
  ((initial_plates - removed_plates) * plate_weight : ℚ) / 16

theorem shipping_weight_calculation :
  max_shipping_weight 38 10 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shipping_weight_calculation_l3424_342486


namespace NUMINAMATH_CALUDE_age_of_fifteenth_student_l3424_342415

theorem age_of_fifteenth_student
  (total_students : ℕ)
  (average_age : ℝ)
  (group1_count : ℕ)
  (group1_average : ℝ)
  (group2_count : ℕ)
  (group2_average : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_count = 4)
  (h4 : group1_average = 14)
  (h5 : group2_count = 10)
  (h6 : group2_average = 16)
  (h7 : group1_count + group2_count + 1 = total_students) :
  (total_students : ℝ) * average_age - 
  ((group1_count : ℝ) * group1_average + (group2_count : ℝ) * group2_average) = 9 :=
by sorry

end NUMINAMATH_CALUDE_age_of_fifteenth_student_l3424_342415


namespace NUMINAMATH_CALUDE_sum_first_ten_enhanced_nice_l3424_342434

def is_prime (n : ℕ) : Prop := sorry

def proper_divisors (n : ℕ) : Set ℕ := sorry

def product_of_set (s : Set ℕ) : ℕ := sorry

def prime_factors (n : ℕ) : List ℕ := sorry

def is_enhanced_nice (n : ℕ) : Prop :=
  (n > 1) ∧
  ((product_of_set (proper_divisors n) = n) ∨
   (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p * q) ∨
   (∃ p : ℕ, is_prime p ∧ n = p^3))

def first_ten_enhanced_nice_under_100 : List ℕ :=
  [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

theorem sum_first_ten_enhanced_nice :
  (List.sum first_ten_enhanced_nice_under_100 = 182) ∧
  (∀ n ∈ first_ten_enhanced_nice_under_100, is_enhanced_nice n) ∧
  (∀ n < 100, is_enhanced_nice n → n ∈ first_ten_enhanced_nice_under_100) :=
sorry

end NUMINAMATH_CALUDE_sum_first_ten_enhanced_nice_l3424_342434


namespace NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l3424_342437

theorem cos_15_cos_45_minus_cos_75_sin_45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_cos_45_minus_cos_75_sin_45_l3424_342437


namespace NUMINAMATH_CALUDE_initial_money_calculation_l3424_342418

theorem initial_money_calculation (X : ℝ) : 
  X * (1 - (0.30 + 0.25 + 0.15)) = 3500 → X = 11666.67 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l3424_342418


namespace NUMINAMATH_CALUDE_greenfield_basketball_association_l3424_342445

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost for one player's home game equipment in dollars -/
def home_cost : ℕ := 2 * sock_cost + tshirt_cost

/-- The cost for one player's away game equipment in dollars -/
def away_cost : ℕ := sock_cost + tshirt_cost

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := home_cost + away_cost

/-- The total cost for equipping all players in dollars -/
def total_cost : ℕ := 3100

/-- The number of players in the Association -/
def num_players : ℕ := 72

theorem greenfield_basketball_association :
  total_cost = num_players * player_cost := by
  sorry

end NUMINAMATH_CALUDE_greenfield_basketball_association_l3424_342445


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_l3424_342492

/-- Given a circle with two chords AB and AC, where AB = a, AC = b, and the length of arc AC is twice
    the length of arc AB, prove that the radius R of the circle is equal to a^2 / sqrt(4a^2 - b^2). -/
theorem circle_radius_from_chords (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R = a^2 / Real.sqrt (4 * a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_chords_l3424_342492


namespace NUMINAMATH_CALUDE_prob_not_edge_10x10_l3424_342417

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  perimeter_squares : ℕ

/-- Calculates the probability of a randomly chosen square not touching the outer edge -/
def prob_not_edge (board : Checkerboard) : ℚ :=
  (board.total_squares - board.perimeter_squares : ℚ) / board.total_squares

/-- Theorem: The probability of a randomly chosen square not touching the outer edge
    on a 10x10 checkerboard is 16/25 -/
theorem prob_not_edge_10x10 :
  ∃ (board : Checkerboard),
    board.size = 10 ∧
    board.total_squares = 100 ∧
    board.perimeter_squares = 36 ∧
    prob_not_edge board = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_edge_10x10_l3424_342417


namespace NUMINAMATH_CALUDE_meeting_attendance_l3424_342412

/-- The number of people attending a meeting where each person receives two copies of a contract --/
def number_of_people (pages_per_contract : ℕ) (copies_per_person : ℕ) (total_pages_copied : ℕ) : ℕ :=
  total_pages_copied / (pages_per_contract * copies_per_person)

/-- Theorem stating that the number of people in the meeting is 9 --/
theorem meeting_attendance : number_of_people 20 2 360 = 9 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendance_l3424_342412


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3424_342453

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3424_342453
