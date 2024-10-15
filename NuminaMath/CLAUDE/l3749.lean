import Mathlib

namespace NUMINAMATH_CALUDE_min_values_l3749_374900

def min_value_exponential (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → 2^x + 4^y ≥ 2*Real.sqrt 2

def min_value_reciprocal (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → 1/x + 2/y ≥ 9

def min_value_squared (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → x^2 + 4*y^2 ≥ 1/2

theorem min_values (x y : ℝ) :
  min_value_exponential x y ∧
  min_value_reciprocal x y ∧
  min_value_squared x y :=
sorry

end NUMINAMATH_CALUDE_min_values_l3749_374900


namespace NUMINAMATH_CALUDE_system_solution_l3749_374941

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0) ∧
  (3 * x^2 * y^2 + y^4 = 84) →
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3749_374941


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3749_374994

/-- A right triangle with specific median lengths has a hypotenuse of 4√14 -/
theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a^2 + b^2 = (a + b)^2 / 4) 
  (h_median1 : b^2 + (a/2)^2 = 34) (h_median2 : a^2 + (b/2)^2 = 36) : 
  (a + b) = 4 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3749_374994


namespace NUMINAMATH_CALUDE_preceding_binary_l3749_374968

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def M : List Bool := [true, false, true, false, true, false]

theorem preceding_binary (M : List Bool) : 
  M = [true, false, true, false, true, false] → 
  decimal_to_binary (binary_to_decimal M - 1) = [true, false, true, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_preceding_binary_l3749_374968


namespace NUMINAMATH_CALUDE_model1_best_fit_l3749_374945

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.976
def R2_model2 : ℝ := 0.776
def R2_model3 : ℝ := 0.076
def R2_model4 : ℝ := 0.351

-- Define a function to determine if a model has the best fitting effect
def has_best_fit (model_R2 : ℝ) : Prop :=
  model_R2 > R2_model2 ∧ model_R2 > R2_model3 ∧ model_R2 > R2_model4

-- Theorem stating that Model 1 has the best fitting effect
theorem model1_best_fit : has_best_fit R2_model1 := by
  sorry

end NUMINAMATH_CALUDE_model1_best_fit_l3749_374945


namespace NUMINAMATH_CALUDE_circle_and_lines_properties_l3749_374913

-- Define the circle C
def circle_C (a : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + y^2 = 4}

-- Define the tangent line
def tangent_line := {(x, y) : ℝ × ℝ | 3*x - 4*y + 4 = 0}

-- Define the intersecting line l
def line_l (k : ℝ) := {(x, y) : ℝ × ℝ | y = k*x - 3}

-- Main theorem
theorem circle_and_lines_properties :
  ∃ (a : ℝ), a > 0 ∧
  (∀ (p : ℝ × ℝ), p ∈ circle_C a → p ∉ tangent_line) ∧
  (∃ (q : ℝ × ℝ), q ∈ circle_C a ∧ q ∈ tangent_line) →
  (a = 2) ∧
  (∀ (k x₁ y₁ x₂ y₂ : ℝ),
    (x₁, y₁) ∈ circle_C 2 ∧ (x₁, y₁) ∈ line_l k ∧
    (x₂, y₂) ∈ circle_C 2 ∧ (x₂, y₂) ∈ line_l k ∧
    (x₁, y₁) ≠ (x₂, y₂) →
    (k = 3 → x₁ * x₂ + y₁ * y₂ = -9/5) ∧
    (x₁ * x₂ + y₁ * y₂ = 8 → k = (-3 + Real.sqrt 29) / 4)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_lines_properties_l3749_374913


namespace NUMINAMATH_CALUDE_expression_evaluation_l3749_374911

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 - 2*x*y = 24 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3749_374911


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3749_374923

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1400 → 
  loss_percentage = 5 → 
  selling_price = cost_price * (1 - loss_percentage / 100) → 
  selling_price = 1330 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3749_374923


namespace NUMINAMATH_CALUDE_seventh_term_is_seven_l3749_374957

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- The sixth term is 6
  sixth_term : a + 5*d = 6

/-- The seventh term of the arithmetic sequence is 7 -/
theorem seventh_term_is_seven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_seven_l3749_374957


namespace NUMINAMATH_CALUDE_cats_after_sale_l3749_374901

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_after_sale 
  (siamese : ℕ) -- Initial number of Siamese cats
  (house : ℕ) -- Initial number of house cats
  (sold : ℕ) -- Number of cats sold during the sale
  (h1 : siamese = 12)
  (h2 : house = 20)
  (h3 : sold = 20) :
  siamese + house - sold = 12 := by
  sorry

end NUMINAMATH_CALUDE_cats_after_sale_l3749_374901


namespace NUMINAMATH_CALUDE_unit_prices_min_type_A_boxes_l3749_374930

-- Define the types of gift boxes
inductive GiftBox
| A
| B

-- Define the unit prices as variables
variable (price_A price_B : ℕ)

-- Define the conditions of the problem
axiom first_purchase : 10 * price_A + 15 * price_B = 2800
axiom second_purchase : 6 * price_A + 5 * price_B = 1200

-- Define the total number of boxes and maximum cost
def total_boxes : ℕ := 40
def max_cost : ℕ := 4500

-- Theorem for the unit prices
theorem unit_prices : price_A = 100 ∧ price_B = 120 := by sorry

-- Function to calculate the total cost
def total_cost (num_A : ℕ) : ℕ :=
  num_A * price_A + (total_boxes - num_A) * price_B

-- Theorem for the minimum number of type A boxes
theorem min_type_A_boxes : 
  ∀ num_A : ℕ, num_A ≥ 15 → total_cost num_A ≤ max_cost := by sorry

end NUMINAMATH_CALUDE_unit_prices_min_type_A_boxes_l3749_374930


namespace NUMINAMATH_CALUDE_cone_base_radius_l3749_374966

theorem cone_base_radius (r : ℝ) (θ : ℝ) (base_radius : ℝ) : 
  r = 9 → θ = 240 * π / 180 → base_radius = r * θ / (2 * π) → base_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3749_374966


namespace NUMINAMATH_CALUDE_problem_statement_l3749_374961

theorem problem_statement (x y : ℝ) :
  |x - 8*y| + (4*y - 1)^2 = 0 → (x + 2*y)^3 = 125/8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3749_374961


namespace NUMINAMATH_CALUDE_lori_marble_sharing_l3749_374951

def total_marbles : ℕ := 30
def marbles_per_friend : ℕ := 6

theorem lori_marble_sharing :
  total_marbles / marbles_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_lori_marble_sharing_l3749_374951


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3749_374986

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ (a : ℕ) + (b : ℕ) = 64 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → (c : ℕ) + (d : ℕ) ≥ 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3749_374986


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3749_374980

/-- An equilateral triangle with a point inside -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The perpendicular distances from the point to the sides
  dist_to_AB : ℝ
  dist_to_BC : ℝ
  dist_to_CA : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  side_length_pos : 0 < side_length
  dist_pos : 0 < dist_to_AB ∧ 0 < dist_to_BC ∧ 0 < dist_to_CA
  point_inside : dist_to_AB + dist_to_BC + dist_to_CA < side_length * Real.sqrt 3

/-- The theorem statement -/
theorem equilateral_triangle_side_length 
  (triangle : EquilateralTriangleWithPoint) 
  (h1 : triangle.dist_to_AB = 2)
  (h2 : triangle.dist_to_BC = 3)
  (h3 : triangle.dist_to_CA = 4) : 
  triangle.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3749_374980


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l3749_374964

/-- The number of green lava lamps -/
def green_lamps : ℕ := 4

/-- The number of purple lava lamps -/
def purple_lamps : ℕ := 4

/-- The total number of lamps -/
def total_lamps : ℕ := green_lamps + purple_lamps

/-- The number of lamps in each row -/
def lamps_per_row : ℕ := 4

/-- The number of rows -/
def num_rows : ℕ := 2

/-- The number of lamps turned on -/
def lamps_on : ℕ := 4

/-- The probability of the specific arrangement -/
def specific_arrangement_probability : ℚ := 1 / 7

theorem lava_lamp_probability :
  (green_lamps = 4) →
  (purple_lamps = 4) →
  (total_lamps = green_lamps + purple_lamps) →
  (lamps_per_row = 4) →
  (num_rows = 2) →
  (lamps_on = 4) →
  (specific_arrangement_probability = 1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l3749_374964


namespace NUMINAMATH_CALUDE_y_exceeds_x_by_100_percent_l3749_374985

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : 
  (y - x) / x = 1 := by sorry

end NUMINAMATH_CALUDE_y_exceeds_x_by_100_percent_l3749_374985


namespace NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l3749_374992

/-- The surface area of a sphere circumscribed about a rectangular solid -/
theorem surface_area_circumscribed_sphere
  (length width height : ℝ)
  (h_length : length = 2)
  (h_width : width = 1)
  (h_height : height = 2) :
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 9 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_surface_area_circumscribed_sphere_l3749_374992


namespace NUMINAMATH_CALUDE_apple_price_difference_l3749_374993

/-- The price difference between Shimla apples and Fuji apples -/
def price_difference (shimla_price fuji_price : ℝ) : ℝ :=
  shimla_price - fuji_price

/-- The condition that the sum of Shimla and Red Delicious prices is 250 more than Red Delicious and Fuji -/
def price_condition (shimla_price red_delicious_price fuji_price : ℝ) : Prop :=
  shimla_price + red_delicious_price = red_delicious_price + fuji_price + 250

theorem apple_price_difference 
  (shimla_price red_delicious_price fuji_price : ℝ) 
  (h : price_condition shimla_price red_delicious_price fuji_price) : 
  price_difference shimla_price fuji_price = 250 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_difference_l3749_374993


namespace NUMINAMATH_CALUDE_quadratic_roots_exist_sum_minus_product_equals_two_l3749_374909

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 3*x + 1 = 0

-- Define the roots
theorem quadratic_roots_exist : ∃ (x₁ x₂ : ℝ), quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂ :=
sorry

-- Theorem to prove
theorem sum_minus_product_equals_two :
  ∃ (x₁ x₂ : ℝ), quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂ ∧ x₁ + x₂ - x₁*x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_exist_sum_minus_product_equals_two_l3749_374909


namespace NUMINAMATH_CALUDE_king_of_red_suit_probability_l3749_374921

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)

/-- Represents the number of Kings of red suits in a standard deck -/
structure RedKings :=
  (count : Nat)

/-- The probability of selecting a specific card from a deck -/
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

theorem king_of_red_suit_probability (d : Deck) (rk : RedKings) :
  d.cards = 52 → rk.count = 2 → probability rk.count d.cards = 1 / 26 := by
  sorry

end NUMINAMATH_CALUDE_king_of_red_suit_probability_l3749_374921


namespace NUMINAMATH_CALUDE_correct_calculation_l3749_374943

theorem correct_calculation : (-0.5)^2010 * 2^2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3749_374943


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l3749_374958

theorem initial_alcohol_percentage
  (initial_volume : Real)
  (added_alcohol : Real)
  (final_percentage : Real)
  (h1 : initial_volume = 6)
  (h2 : added_alcohol = 1.2)
  (h3 : final_percentage = 50)
  (h4 : (initial_percentage / 100) * initial_volume + added_alcohol = 
        (final_percentage / 100) * (initial_volume + added_alcohol)) :
  initial_percentage = 40 := by
  sorry

#check initial_alcohol_percentage

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l3749_374958


namespace NUMINAMATH_CALUDE_tomatoes_left_after_yesterday_l3749_374972

/-- The number of tomatoes left after yesterday's picking -/
def tomatoes_left (initial : ℕ) (picked_yesterday : ℕ) : ℕ :=
  initial - picked_yesterday

/-- Theorem: Given 160 initial tomatoes and 56 tomatoes picked yesterday,
    the number of tomatoes left after yesterday's picking is 104. -/
theorem tomatoes_left_after_yesterday :
  tomatoes_left 160 56 = 104 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_after_yesterday_l3749_374972


namespace NUMINAMATH_CALUDE_least_N_for_P_condition_l3749_374936

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2 / 5 : ℚ) * N⌉) / (N + 1 : ℚ)

theorem least_N_for_P_condition :
  ∀ N : ℕ, N > 0 ∧ N % 10 = 0 →
    (P N 2 < 8 / 10 ↔ N ≥ 10) ∧
    (∀ M : ℕ, M > 0 ∧ M % 10 = 0 ∧ M < 10 → P M 2 ≥ 8 / 10) :=
by sorry

end NUMINAMATH_CALUDE_least_N_for_P_condition_l3749_374936


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3749_374914

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a * b + a * c + b * c = -4) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = -5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3749_374914


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3749_374915

/-- The diagonal of a rectangular prism with dimensions 15, 25, and 15 is 5√43 -/
theorem rectangular_prism_diagonal : 
  ∀ (a b c d : ℝ), 
    a = 15 → 
    b = 25 → 
    c = 15 → 
    d ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 → 
    d = 5 * Real.sqrt 43 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3749_374915


namespace NUMINAMATH_CALUDE_decimal_division_l3749_374956

theorem decimal_division (x y : ℚ) (hx : x = 45/100) (hy : y = 5/1000) : x / y = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l3749_374956


namespace NUMINAMATH_CALUDE_car_clock_accuracy_l3749_374969

def actual_time (start_time : ℕ) (elapsed_time : ℕ) (gain_rate : ℚ) : ℚ :=
  start_time + elapsed_time / gain_rate

theorem car_clock_accuracy (start_time : ℕ) (elapsed_time : ℕ) : 
  start_time = 8 * 60 →  -- 8:00 AM in minutes
  elapsed_time = 14 * 60 →  -- 14 hours from 8:00 AM to 10:00 PM in minutes
  actual_time start_time elapsed_time (37/36) = 21 * 60 + 47  -- 9:47 PM in minutes
  := by sorry

end NUMINAMATH_CALUDE_car_clock_accuracy_l3749_374969


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3749_374918

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is perpendicular to c, 
    then the k-coordinate of c is -3. -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) : 
  a = (Real.sqrt 3, 1) → 
  b = (0, -1) → 
  c.1 = k → 
  c.2 = Real.sqrt 3 → 
  (a.1 - 2 * b.1, a.2 - 2 * b.2) • c = 0 → 
  k = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3749_374918


namespace NUMINAMATH_CALUDE_burger_cost_is_12_l3749_374947

/-- The cost of each burger Owen bought in June -/
def burger_cost (burgers_per_day : ℕ) (total_spent : ℕ) (days_in_june : ℕ) : ℚ :=
  total_spent / (burgers_per_day * days_in_june)

/-- Theorem stating that each burger costs 12 dollars -/
theorem burger_cost_is_12 :
  burger_cost 2 720 30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_burger_cost_is_12_l3749_374947


namespace NUMINAMATH_CALUDE_ghee_mixture_problem_l3749_374926

theorem ghee_mixture_problem (x : ℝ) : 
  (0.6 * x = x - 0.4 * x) →  -- 60% is pure ghee, 40% is vanaspati
  (0.4 * x = 0.2 * (x + 10)) →  -- After adding 10 kg, vanaspati becomes 20%
  (x = 10) :=  -- The original quantity was 10 kg
by sorry

end NUMINAMATH_CALUDE_ghee_mixture_problem_l3749_374926


namespace NUMINAMATH_CALUDE_units_digit_17_1987_l3749_374983

theorem units_digit_17_1987 : (17^1987) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_1987_l3749_374983


namespace NUMINAMATH_CALUDE_bears_per_shelf_l3749_374908

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 5)
  (h2 : new_shipment = 7)
  (h3 : num_shelves = 2)
  : (initial_stock + new_shipment) / num_shelves = 6 := by
  sorry

end NUMINAMATH_CALUDE_bears_per_shelf_l3749_374908


namespace NUMINAMATH_CALUDE_two_number_difference_l3749_374995

theorem two_number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : 
  |y - x| = 80 / 7 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l3749_374995


namespace NUMINAMATH_CALUDE_village_cats_l3749_374974

theorem village_cats (total_cats : ℕ) 
  (striped_ratio : ℚ) (spotted_ratio : ℚ) 
  (fluffy_striped_ratio : ℚ) (fluffy_spotted_ratio : ℚ)
  (h_total : total_cats = 180)
  (h_striped : striped_ratio = 1/2)
  (h_spotted : spotted_ratio = 1/3)
  (h_fluffy_striped : fluffy_striped_ratio = 1/8)
  (h_fluffy_spotted : fluffy_spotted_ratio = 3/7) :
  ⌊striped_ratio * total_cats * fluffy_striped_ratio⌋ + 
  ⌊spotted_ratio * total_cats * fluffy_spotted_ratio⌋ = 36 := by
sorry

end NUMINAMATH_CALUDE_village_cats_l3749_374974


namespace NUMINAMATH_CALUDE_sin_graph_shift_l3749_374904

theorem sin_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x - π / 10)) = 3 * Real.sin (2 * x - π / 5) := by sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l3749_374904


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l3749_374937

/-- The fixed point through which all lines of the form kx-y+1=3k pass -/
def fixed_point : ℝ × ℝ := (3, 1)

/-- The equation of the line parameterized by k -/
def line_equation (k x y : ℝ) : Prop := k*x - y + 1 = 3*k

/-- Theorem stating that the fixed_point is the unique point through which all lines pass -/
theorem fixed_point_theorem :
  ∀ (k : ℝ), line_equation k (fixed_point.1) (fixed_point.2) ∧
  ∀ (x y : ℝ), (∀ (k : ℝ), line_equation k x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l3749_374937


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l3749_374996

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h1 : prob_red = 0.42)
  (h2 : prob_white = 0.28)
  (h3 : prob_red + prob_white + prob_black = 1) :
  prob_black = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l3749_374996


namespace NUMINAMATH_CALUDE_average_age_combined_l3749_374920

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 40 →
  n_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents) = 28.8 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_l3749_374920


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l3749_374948

/-- The number of sacks of oranges harvested per day -/
def sacks_per_day : ℕ := 38

/-- The number of days of harvest -/
def harvest_days : ℕ := 49

/-- The total number of sacks of oranges harvested after the harvest period -/
def total_sacks : ℕ := sacks_per_day * harvest_days

theorem orange_harvest_theorem : total_sacks = 1862 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l3749_374948


namespace NUMINAMATH_CALUDE_parallel_perpendicular_line_coefficient_l3749_374989

/-- Given two lines in the plane, if there exists a third line parallel to one and perpendicular to the other, prove that the coefficient k in the equations must be zero. -/
theorem parallel_perpendicular_line_coefficient (k : ℝ) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), (3 * x - k * y + c = 0) ∧ 
    ((3 * x - k * y + c = 0) ↔ (3 * x - k * y + 6 = 0)) ∧
    ((3 * k + (-k) * 1 = 0) ↔ (k * x + y + 1 = 0))) → 
  k = 0 := by
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_line_coefficient_l3749_374989


namespace NUMINAMATH_CALUDE_equation_solution_l3749_374963

theorem equation_solution (a b : ℝ) (x₁ x₂ x₃ : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∀ x : ℝ, Real.sqrt (|x|) + Real.sqrt (|x + a|) = b ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₁ < x₂ →
  x₂ < x₃ →
  x₃ = b →
  a + b = 144 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3749_374963


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3749_374991

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/x + 1/y ≤ 1/a + 1/b) ∧ (1/x + 1/y = 4 → x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3749_374991


namespace NUMINAMATH_CALUDE_calculation_proof_l3749_374975

theorem calculation_proof : 65 + 5 * 12 / (180 / 3) = 66 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3749_374975


namespace NUMINAMATH_CALUDE_fraction_equality_proof_l3749_374910

theorem fraction_equality_proof (x : ℝ) : 
  x ≠ 4 ∧ x ≠ 8 → ((x - 3) / (x - 4) = (x - 5) / (x - 8) ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_proof_l3749_374910


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l3749_374905

theorem no_solution_to_inequalities : 
  ¬∃ (x y : ℝ), (4*x^2 + 4*x*y + 19*y^2 ≤ 2) ∧ (x - y ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l3749_374905


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l3749_374987

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1233 :=
by sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l3749_374987


namespace NUMINAMATH_CALUDE_class_size_proof_l3749_374981

/-- Given a student's rank from top and bottom in a class, 
    calculate the total number of students -/
def total_students (rank_from_top rank_from_bottom : ℕ) : ℕ :=
  rank_from_top + rank_from_bottom - 1

/-- Theorem stating that a class with a student ranking 24th from top 
    and 34th from bottom has 57 students in total -/
theorem class_size_proof :
  total_students 24 34 = 57 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l3749_374981


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3749_374938

theorem polynomial_factorization (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^2 - x - 1) * q x = (-987 * x^18 + 2584 * x^17 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3749_374938


namespace NUMINAMATH_CALUDE_expansion_coefficient_equality_l3749_374903

theorem expansion_coefficient_equality (n : ℕ+) : 
  (8 * (Nat.choose n 3)) = (8 * 2 * (Nat.choose n 1)) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_equality_l3749_374903


namespace NUMINAMATH_CALUDE_sweatshirt_cost_is_15_l3749_374946

def hannah_shopping (sweatshirt_cost : ℝ) : Prop :=
  let num_sweatshirts : ℕ := 3
  let num_tshirts : ℕ := 2
  let tshirt_cost : ℝ := 10
  let total_spent : ℝ := 65
  (num_sweatshirts * sweatshirt_cost) + (num_tshirts * tshirt_cost) = total_spent

theorem sweatshirt_cost_is_15 : 
  ∃ (sweatshirt_cost : ℝ), hannah_shopping sweatshirt_cost ∧ sweatshirt_cost = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_sweatshirt_cost_is_15_l3749_374946


namespace NUMINAMATH_CALUDE_walnut_trees_increase_l3749_374990

/-- Calculates the total number of walnut trees after planting given the initial number and percentage increase -/
def total_trees_after_planting (initial_trees : ℕ) (percent_increase : ℕ) : ℕ :=
  initial_trees + (initial_trees * percent_increase) / 100

/-- Theorem stating that with 22 initial trees and 150% increase, the total after planting is 55 -/
theorem walnut_trees_increase :
  total_trees_after_planting 22 150 = 55 := by
  sorry

#eval total_trees_after_planting 22 150

end NUMINAMATH_CALUDE_walnut_trees_increase_l3749_374990


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l3749_374978

theorem quadratic_equation_from_means (η ζ : ℝ) 
  (h_arithmetic_mean : (η + ζ) / 2 = 7)
  (h_geometric_mean : Real.sqrt (η * ζ) = 8) :
  ∀ x : ℝ, x^2 - 14*x + 64 = 0 ↔ (x = η ∨ x = ζ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l3749_374978


namespace NUMINAMATH_CALUDE_total_participants_grandmasters_top_positions_l3749_374944

/-- A round-robin chess tournament with grandmasters and masters -/
structure ChessTournament where
  num_grandmasters : ℕ
  num_masters : ℕ
  total_points_grandmasters : ℕ
  total_points_masters : ℕ

/-- The conditions of the tournament -/
def tournament_conditions (t : ChessTournament) : Prop :=
  t.num_masters = 3 * t.num_grandmasters ∧
  t.total_points_masters = (12 * t.total_points_grandmasters) / 10 ∧
  t.total_points_grandmasters + t.total_points_masters = (t.num_grandmasters + t.num_masters) * (t.num_grandmasters + t.num_masters - 1)

/-- The theorem stating the total number of participants -/
theorem total_participants (t : ChessTournament) (h : tournament_conditions t) : 
  t.num_grandmasters + t.num_masters = 12 := by
  sorry

/-- The theorem stating that grandmasters took the top positions -/
theorem grandmasters_top_positions (t : ChessTournament) (h : tournament_conditions t) : 
  t.num_grandmasters ≤ 3 ∧ t.num_grandmasters > 0 := by
  sorry

end NUMINAMATH_CALUDE_total_participants_grandmasters_top_positions_l3749_374944


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3749_374932

theorem trig_expression_simplification :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 = 
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3749_374932


namespace NUMINAMATH_CALUDE_seconds_in_minutes_l3749_374940

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we are converting to seconds -/
def minutes : ℚ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_minutes : (minutes * seconds_per_minute : ℚ) = 750 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_minutes_l3749_374940


namespace NUMINAMATH_CALUDE_twenty_four_game_l3749_374929

theorem twenty_four_game (a b c d : ℤ) (e f g h : ℕ) : 
  (a = 3 ∧ b = 4 ∧ c = -6 ∧ d = 10) →
  (e = 3 ∧ f = 2 ∧ g = 6 ∧ h = 7) →
  ∃ (expr1 expr2 : ℤ → ℤ → ℤ → ℤ → ℤ),
    expr1 a b c d = 24 ∧
    expr2 e f g h = 24 :=
by sorry

end NUMINAMATH_CALUDE_twenty_four_game_l3749_374929


namespace NUMINAMATH_CALUDE_unique_two_digit_sum_diff_product_l3749_374916

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_sum_diff_product (n : ℕ) : Prop :=
  ∃ x y : ℕ,
    n = 10 * x + y ∧
    1 ≤ x ∧ x ≤ 9 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    n = (x + y) * (y - x)

theorem unique_two_digit_sum_diff_product :
  ∃! n : ℕ, is_two_digit n ∧ digits_sum_diff_product n ∧ n = 48 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_sum_diff_product_l3749_374916


namespace NUMINAMATH_CALUDE_vegetable_garden_theorem_l3749_374967

def vegetable_garden_total (potatoes cucumbers tomatoes peppers carrots : ℕ) : Prop :=
  potatoes = 1200 ∧
  cucumbers = potatoes - 160 ∧
  tomatoes = 4 * cucumbers ∧
  peppers * peppers = cucumbers * tomatoes ∧
  carrots = (cucumbers + tomatoes) + (cucumbers + tomatoes) / 5 ∧
  potatoes + cucumbers + tomatoes + peppers + carrots = 14720

theorem vegetable_garden_theorem :
  ∃ (potatoes cucumbers tomatoes peppers carrots : ℕ),
    vegetable_garden_total potatoes cucumbers tomatoes peppers carrots :=
by
  sorry

end NUMINAMATH_CALUDE_vegetable_garden_theorem_l3749_374967


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3749_374931

theorem complex_equation_solution :
  ∀ z : ℂ, z + Complex.abs z = 2 + Complex.I → z = (3/4 : ℝ) + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3749_374931


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_bounds_l3749_374977

/-- Represents the dimensions of a rectangular solid --/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a rectangular solid --/
def surfaceArea (d : Dimensions) : ℕ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Checks if the given dimensions use exactly 12 unit cubes --/
def usestwelveCubes (d : Dimensions) : Prop :=
  d.length * d.width * d.height = 12

theorem rectangular_solid_surface_area_bounds :
  ∃ (min max : ℕ),
    (∀ d : Dimensions, usestwelveCubes d → min ≤ surfaceArea d) ∧
    (∃ d : Dimensions, usestwelveCubes d ∧ surfaceArea d = min) ∧
    (∀ d : Dimensions, usestwelveCubes d → surfaceArea d ≤ max) ∧
    (∃ d : Dimensions, usestwelveCubes d ∧ surfaceArea d = max) ∧
    min = 32 ∧ max = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_bounds_l3749_374977


namespace NUMINAMATH_CALUDE_square_of_85_l3749_374939

theorem square_of_85 : (85 : ℕ)^2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l3749_374939


namespace NUMINAMATH_CALUDE_light_nanosecond_distance_l3749_374933

/-- The speed of light in meters per second -/
def speed_of_light : ℝ := 3e8

/-- One billionth of a second in seconds -/
def one_billionth : ℝ := 1e-9

/-- The distance traveled by light in one billionth of a second in meters -/
def light_nanosecond : ℝ := speed_of_light * one_billionth

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

theorem light_nanosecond_distance :
  light_nanosecond * meters_to_cm = 30 := by sorry

end NUMINAMATH_CALUDE_light_nanosecond_distance_l3749_374933


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3749_374973

/-- The function f(x) = |x - a| is increasing on [-3, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Ici (-3) → y ∈ Set.Ici (-3) → x ≤ y → |x - a| ≤ |y - a|

/-- "a = -3" is a sufficient condition -/
theorem sufficient_condition (a : ℝ) (h : a = -3) : is_increasing_on_interval a :=
sorry

/-- "a = -3" is not a necessary condition -/
theorem not_necessary_condition : ∃ a : ℝ, a ≠ -3 ∧ is_increasing_on_interval a :=
sorry

/-- "a = -3" is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = -3 → is_increasing_on_interval a) ∧
  (∃ a : ℝ, a ≠ -3 ∧ is_increasing_on_interval a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3749_374973


namespace NUMINAMATH_CALUDE_f_inverse_a_eq_28_l3749_374960

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x^(1/3)
  else if x ≥ 1 then 4*(x-1)
  else 0  -- undefined for x ≤ 0

theorem f_inverse_a_eq_28 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 28 := by
  sorry

end NUMINAMATH_CALUDE_f_inverse_a_eq_28_l3749_374960


namespace NUMINAMATH_CALUDE_inequality_proof_l3749_374970

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3749_374970


namespace NUMINAMATH_CALUDE_marianne_age_always_12_more_than_bella_l3749_374998

/-- Represents the age difference between Marianne and Bella -/
def age_difference : ℕ := 12

/-- Marianne's age when Bella is 8 years old -/
def marianne_age_when_bella_8 : ℕ := 20

/-- Bella's age when Marianne is 30 years old -/
def bella_age_when_marianne_30 : ℕ := 18

/-- Marianne's age as a function of Bella's age -/
def marianne_age (bella_age : ℕ) : ℕ := bella_age + age_difference

theorem marianne_age_always_12_more_than_bella :
  ∀ (bella_age : ℕ),
    marianne_age bella_age = bella_age + age_difference :=
by
  sorry

#check marianne_age_always_12_more_than_bella

end NUMINAMATH_CALUDE_marianne_age_always_12_more_than_bella_l3749_374998


namespace NUMINAMATH_CALUDE_exponential_function_properties_l3749_374952

theorem exponential_function_properties (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  let f : ℝ → ℝ := fun x ↦ 2^x
  (f (x₁ + x₂) = f x₁ * f x₂) ∧
  (f (-x₁) = 1 / f x₁) ∧
  ((f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_properties_l3749_374952


namespace NUMINAMATH_CALUDE_triangle_range_theorem_l3749_374962

theorem triangle_range_theorem (a b x : ℝ) (B : ℝ) (has_two_solutions : Prop) :
  a = x →
  b = 2 →
  B = π / 3 →
  has_two_solutions →
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_range_theorem_l3749_374962


namespace NUMINAMATH_CALUDE_squirrel_count_l3749_374935

theorem squirrel_count :
  ∀ (first_count second_count : ℕ),
  second_count = first_count + (first_count / 3) →
  first_count + second_count = 28 →
  first_count = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_count_l3749_374935


namespace NUMINAMATH_CALUDE_largest_in_set_l3749_374925

theorem largest_in_set (a : ℝ) (h : a = -3) :
  let S : Set ℝ := {-2*a, 3*a, 18/a, a^3, 2}
  ∀ x ∈ S, -2*a ≥ x :=
by sorry

end NUMINAMATH_CALUDE_largest_in_set_l3749_374925


namespace NUMINAMATH_CALUDE_english_test_average_l3749_374934

theorem english_test_average (avg_two_months : ℝ) (third_month_score : ℝ) :
  avg_two_months = 86 →
  third_month_score = 98 →
  (2 * avg_two_months + third_month_score) / 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_english_test_average_l3749_374934


namespace NUMINAMATH_CALUDE_football_team_progress_l3749_374965

def team_progress (loss : Int) (gain : Int) : Int :=
  gain - loss

theorem football_team_progress :
  team_progress 5 8 = 3 := by sorry

end NUMINAMATH_CALUDE_football_team_progress_l3749_374965


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3749_374912

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (x^3 - 5*x^2 + 13*x - 7 = (x - a) * (x - b) * (x - c)) → 
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 490 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3749_374912


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3749_374942

-- Define the quadratic function
def f (m x : ℝ) : ℝ := (m + 1) * x^2 + (m^2 - 2*m - 3) * x - m + 3

-- State the theorem
theorem quadratic_inequality_range (m : ℝ) :
  (∀ x, f m x > 0) ↔ (m ∈ Set.Icc (-1) 1 ∪ Set.Ioo 1 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3749_374942


namespace NUMINAMATH_CALUDE_max_intersection_points_l3749_374979

-- Define the geometric objects
def Circle : Type := Unit
def Ellipse : Type := Unit
def Line : Type := Unit

-- Define the intersection function
def intersection_points (c : Circle) (e : Ellipse) (l : Line) : ℕ := sorry

-- Theorem statement
theorem max_intersection_points :
  ∃ (c : Circle) (e : Ellipse) (l : Line),
    ∀ (c' : Circle) (e' : Ellipse) (l' : Line),
      intersection_points c e l ≥ intersection_points c' e' l' ∧
      intersection_points c e l = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l3749_374979


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3749_374902

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x - 2*y)^5 = a*(x + 2*y)^5 + a₁*(x + 2*y)^4*y + a₂*(x + 2*y)^3*y^2 + 
                            a₃*(x + 2*y)^2*y^3 + a₄*(x + 2*y)*y^4 + a₅*y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3749_374902


namespace NUMINAMATH_CALUDE_monthly_income_proof_l3749_374922

/-- Given the average monthly incomes of three people, prove the income of one person. -/
theorem monthly_income_proof (P Q R : ℝ) 
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : (P + R) / 2 = 6200) :
  P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_proof_l3749_374922


namespace NUMINAMATH_CALUDE_remainder_b_107_mod_64_l3749_374949

theorem remainder_b_107_mod_64 : (5^107 + 9^107) % 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_b_107_mod_64_l3749_374949


namespace NUMINAMATH_CALUDE_fishing_problem_l3749_374999

theorem fishing_problem (jason ryan jeffery : ℕ) : 
  ryan = 3 * jason →
  jeffery = 2 * ryan →
  jeffery = 60 →
  jason + ryan + jeffery = 100 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l3749_374999


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l3749_374976

theorem sqrt_a_sqrt_a_sqrt_a (a : ℝ) (ha : a ≥ 0) : 
  Real.sqrt (a * Real.sqrt a * Real.sqrt a) = a := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_sqrt_a_l3749_374976


namespace NUMINAMATH_CALUDE_graces_coins_worth_l3749_374988

/-- The total worth of Grace's coins in pennies -/
def total_worth (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes + 25 * quarters

/-- Theorem stating that Grace's coins are worth 550 pennies -/
theorem graces_coins_worth : total_worth 25 15 20 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_graces_coins_worth_l3749_374988


namespace NUMINAMATH_CALUDE_g_odd_g_strictly_increasing_l3749_374950

/-- The function g(x) = lg(x + √(x^2 + 1)) -/
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

/-- g is an odd function -/
theorem g_odd : ∀ x, g (-x) = -g x := by sorry

/-- g is strictly increasing on ℝ -/
theorem g_strictly_increasing : StrictMono g := by sorry

end NUMINAMATH_CALUDE_g_odd_g_strictly_increasing_l3749_374950


namespace NUMINAMATH_CALUDE_linear_function_composition_l3749_374959

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 9 * x + 8) →
  (∀ x, f x = 3 * x + 2) ∨ (∀ x, f x = -3 * x - 4) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3749_374959


namespace NUMINAMATH_CALUDE_sequence_sum_values_l3749_374971

def is_valid_sequence (a b : ℕ → ℕ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∀ n, b (n + 1) > b n) ∧
  (a 10 = b 10) ∧ 
  (a 10 < 2017) ∧
  (∀ n, a (n + 2) = a (n + 1) + a n) ∧
  (∀ n, b (n + 1) = 2 * b n)

theorem sequence_sum_values (a b : ℕ → ℕ) :
  is_valid_sequence a b → (a 1 + b 1 = 13 ∨ a 1 + b 1 = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_values_l3749_374971


namespace NUMINAMATH_CALUDE_crayon_division_l3749_374955

theorem crayon_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  crayons_per_person = total_crayons / num_people →
  crayons_per_person = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_crayon_division_l3749_374955


namespace NUMINAMATH_CALUDE_remaining_pictures_l3749_374928

/-- The number of pictures Haley took at the zoo -/
def zoo_pictures : ℕ := 50

/-- The number of pictures Haley took at the museum -/
def museum_pictures : ℕ := 8

/-- The number of pictures Haley deleted -/
def deleted_pictures : ℕ := 38

/-- Theorem: The number of pictures Haley still has from her vacation is 20 -/
theorem remaining_pictures : 
  zoo_pictures + museum_pictures - deleted_pictures = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pictures_l3749_374928


namespace NUMINAMATH_CALUDE_particular_number_problem_l3749_374907

theorem particular_number_problem (x : ℤ) : 
  (x - 29 + 64 = 76) → x = 41 := by
sorry

end NUMINAMATH_CALUDE_particular_number_problem_l3749_374907


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l3749_374953

theorem sqrt_two_times_sqrt_eight_equals_four :
  Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_eight_equals_four_l3749_374953


namespace NUMINAMATH_CALUDE_infinitely_many_n_for_f_congruence_l3749_374906

/-- The function f(p, n) represents the largest integer k such that p^k divides n! -/
def f (p n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem infinitely_many_n_for_f_congruence 
  (p : ℕ) 
  (m c : ℕ+) 
  (h_prime : Nat.Prime p) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, f p n ≡ c.val [MOD m.val] := by sorry

end NUMINAMATH_CALUDE_infinitely_many_n_for_f_congruence_l3749_374906


namespace NUMINAMATH_CALUDE_cat_speed_l3749_374954

/-- Proves that given a rabbit running at 25 miles per hour, a cat with a 15-minute head start,
    and the rabbit taking 1 hour to catch up, the cat's speed is 20 miles per hour. -/
theorem cat_speed (rabbit_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  rabbit_speed = 25 →
  head_start = 0.25 →
  catch_up_time = 1 →
  ∃ cat_speed : ℝ,
    cat_speed * (head_start + catch_up_time) = rabbit_speed * catch_up_time ∧
    cat_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_cat_speed_l3749_374954


namespace NUMINAMATH_CALUDE_unique_intersection_implies_a_value_l3749_374919

/-- Given a line y = 2a and a function y = |x-a| - 1 in the Cartesian coordinate system,
    if they have only one intersection point, then a = -1/2 --/
theorem unique_intersection_implies_a_value (a : ℝ) :
  (∃! p : ℝ × ℝ, p.2 = 2*a ∧ p.2 = |p.1 - a| - 1) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_a_value_l3749_374919


namespace NUMINAMATH_CALUDE_saved_amount_l3749_374917

theorem saved_amount (x : ℕ) : (3 * x - 42)^2 = 2241 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_saved_amount_l3749_374917


namespace NUMINAMATH_CALUDE_amanda_hourly_rate_l3749_374984

/-- Amanda's work scenario --/
structure AmandaWork where
  hours_per_day : ℕ
  pay_percentage : ℚ
  reduced_pay : ℚ

/-- Calculate Amanda's hourly rate --/
def hourly_rate (w : AmandaWork) : ℚ :=
  (w.reduced_pay / w.pay_percentage) / w.hours_per_day

/-- Theorem: Amanda's hourly rate is $50 --/
theorem amanda_hourly_rate (w : AmandaWork) 
  (h1 : w.hours_per_day = 10)
  (h2 : w.pay_percentage = 4/5)
  (h3 : w.reduced_pay = 400) :
  hourly_rate w = 50 := by
  sorry

#eval hourly_rate { hours_per_day := 10, pay_percentage := 4/5, reduced_pay := 400 }

end NUMINAMATH_CALUDE_amanda_hourly_rate_l3749_374984


namespace NUMINAMATH_CALUDE_percentage_difference_l3749_374924

theorem percentage_difference : (0.6 * 50) - (0.42 * 30) = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3749_374924


namespace NUMINAMATH_CALUDE_nells_baseball_cards_l3749_374997

/-- Nell's card collection problem -/
theorem nells_baseball_cards
  (initial_ace : ℕ)
  (final_ace final_baseball : ℕ)
  (ace_difference baseball_difference : ℕ)
  (h1 : initial_ace = 18)
  (h2 : final_ace = 55)
  (h3 : final_baseball = 178)
  (h4 : baseball_difference = 123)
  (h5 : final_baseball = final_ace + baseball_difference)
  : final_baseball + baseball_difference = 301 := by
  sorry

#check nells_baseball_cards

end NUMINAMATH_CALUDE_nells_baseball_cards_l3749_374997


namespace NUMINAMATH_CALUDE_second_day_sales_l3749_374927

def ice_cream_sales (x : ℕ) : List ℕ := [100, x, 109, 96, 103, 96, 105]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem second_day_sales :
  ∃ (x : ℕ), mean (ice_cream_sales x) = 100.1 ∧ x = 92 := by sorry

end NUMINAMATH_CALUDE_second_day_sales_l3749_374927


namespace NUMINAMATH_CALUDE_thursday_to_tuesday_ratio_l3749_374982

/-- Represents the number of crates sold on each day --/
structure DailySales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Theorem stating the ratio of Thursday to Tuesday sales --/
theorem thursday_to_tuesday_ratio
  (sales : DailySales)
  (h1 : sales.monday = 5)
  (h2 : sales.tuesday = 2 * sales.monday)
  (h3 : sales.wednesday = sales.tuesday - 2)
  (h4 : sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28) :
  sales.thursday / sales.tuesday = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_thursday_to_tuesday_ratio_l3749_374982
