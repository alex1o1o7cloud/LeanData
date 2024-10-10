import Mathlib

namespace exists_valid_configuration_l3054_305493

/-- A point in a plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points form an isosceles triangle -/
def isIsosceles (p1 p2 p3 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d31 := (p3.x - p1.x)^2 + (p3.y - p1.y)^2
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- A configuration of five points in a plane -/
def Configuration := Fin 5 → Point

/-- Check if a configuration satisfies the isosceles condition for all triplets -/
def validConfiguration (config : Configuration) : Prop :=
  ∀ i j k, i < j → j < k → isIsosceles (config i) (config j) (config k)

/-- There exists a configuration of five points satisfying the isosceles condition -/
theorem exists_valid_configuration : ∃ (config : Configuration), validConfiguration config := by
  sorry

end exists_valid_configuration_l3054_305493


namespace mean_of_four_integers_l3054_305463

theorem mean_of_four_integers (x : ℤ) : 
  (78 + 83 + 82 + x) / 4 = 80 → x = 77 ∧ x = 80 - 3 := by
  sorry

end mean_of_four_integers_l3054_305463


namespace constant_function_derivative_l3054_305497

theorem constant_function_derivative (f : ℝ → ℝ) (h : ∀ x, f x = 7) :
  ∀ x, deriv f x = 0 := by sorry

end constant_function_derivative_l3054_305497


namespace x_value_proof_l3054_305440

theorem x_value_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 3)
  (h2 : y^2 / z = 4)
  (h3 : z^2 / x = 5) :
  x = (6480 : ℝ)^(1/7) := by
sorry

end x_value_proof_l3054_305440


namespace crazy_silly_school_books_l3054_305471

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 21

/-- The number of books that have been read -/
def books_read : ℕ := 13

/-- The number of books yet to be read -/
def books_unread : ℕ := 8

/-- Theorem: The total number of books is equal to the sum of read and unread books -/
theorem crazy_silly_school_books : total_books = books_read + books_unread := by
  sorry

end crazy_silly_school_books_l3054_305471


namespace possible_values_of_a_l3054_305436

def A (a : ℝ) : Set ℝ := {2, a^2 - a + 2, 1 - a}

theorem possible_values_of_a (a : ℝ) : 4 ∈ A a → a = -3 ∨ a = 2 := by
  sorry

end possible_values_of_a_l3054_305436


namespace ab_range_l3054_305423

def f (x : ℝ) : ℝ := |2 - x^2|

theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  0 < a * b ∧ a * b < 2 := by
  sorry

end ab_range_l3054_305423


namespace hash_nested_20_l3054_305404

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem hash_nested_20 : hash (hash (hash (hash 20))) = 5 := by sorry

end hash_nested_20_l3054_305404


namespace root_in_interval_l3054_305467

noncomputable def f (x : ℝ) := 2^x - 3*x

theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo 3 4 ∧ f r = 0 := by
  sorry

end root_in_interval_l3054_305467


namespace irrational_sum_product_theorem_l3054_305442

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), (q : ℝ) = x)

-- State the theorem
theorem irrational_sum_product_theorem (a : ℝ) (h : IsIrrational a) :
  ∃ (b b' : ℝ), IsIrrational b ∧ IsIrrational b' ∧
    (∃ (q1 q2 : ℚ), (a + b : ℝ) = q1 ∧ (a * b' : ℝ) = q2) ∧
    IsIrrational (a * b) ∧ IsIrrational (a + b') := by
  sorry


end irrational_sum_product_theorem_l3054_305442


namespace sector_area_l3054_305485

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 5) (h2 : θ = 2) :
  (1 / 2) * r^2 * θ = 25 := by
  sorry

end sector_area_l3054_305485


namespace high_jump_probabilities_l3054_305413

/-- The probability of clearing the height in a single jump -/
def p : ℝ := 0.8

/-- The probability of clearing the height on two consecutive jumps -/
def prob_two_consecutive : ℝ := p * p

/-- The probability of clearing the height for the first time on the third attempt -/
def prob_third_attempt : ℝ := (1 - p) * (1 - p) * p

/-- The minimum number of attempts required to clear the height with a 99% probability -/
def min_attempts : ℕ := 3

/-- Theorem stating the probabilities and minimum attempts -/
theorem high_jump_probabilities :
  prob_two_consecutive = 0.64 ∧
  prob_third_attempt = 0.032 ∧
  min_attempts = 3 ∧
  (1 - (1 - p) ^ min_attempts ≥ 0.99) :=
by sorry

end high_jump_probabilities_l3054_305413


namespace triangle_count_is_sixteen_l3054_305429

/-- Represents a rectangle with diagonals and internal rectangle --/
structure ConfiguredRectangle where
  vertices : Fin 4 → Point
  diagonals : List (Point × Point)
  midpoints : Fin 4 → Point
  internal_rectangle : List (Point × Point)

/-- Counts the number of triangles in the configured rectangle --/
def count_triangles (rect : ConfiguredRectangle) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles is 16 --/
theorem triangle_count_is_sixteen (rect : ConfiguredRectangle) : 
  count_triangles rect = 16 :=
sorry

end triangle_count_is_sixteen_l3054_305429


namespace baseball_hits_percentage_l3054_305417

/-- 
Given a baseball player's hit statistics for a season:
- Total hits: 50
- Home runs: 2
- Triples: 3
- Doubles: 8

This theorem proves that the percentage of hits that were singles is 74%.
-/
theorem baseball_hits_percentage (total_hits home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 8) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 74 := by
  sorry

#eval (50 - (2 + 3 + 8)) / 50 * 100  -- Should output 74

end baseball_hits_percentage_l3054_305417


namespace absolute_value_fraction_sum_l3054_305409

theorem absolute_value_fraction_sum (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  |x| / x + |x * y| / (x * y) = 0 := by
  sorry

end absolute_value_fraction_sum_l3054_305409


namespace odd_function_extrema_l3054_305475

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- State the theorem
theorem odd_function_extrema :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (-x) = -(f a b c x)) →  -- f is odd
  (f a b c 1 = 2) →                     -- maximum value of 2 at x = 1
  (∀ x, f a b c x ≤ f a b c 1) →        -- global maximum at x = 1
  (∃ (f_max f_min : ℝ),
    (∀ x ∈ Set.Icc (-4) 3, f (-1) 0 3 x ≤ f_max) ∧
    (∀ x ∈ Set.Icc (-4) 3, f_min ≤ f (-1) 0 3 x) ∧
    f_max = 52 ∧ f_min = -18) :=
by sorry


end odd_function_extrema_l3054_305475


namespace line_contains_point_l3054_305414

/-- A line in the xy-plane represented by the equation 2 - kx = 5y -/
def line (k : ℝ) (x y : ℝ) : Prop := 2 - k * x = 5 * y

/-- The point (2, -1) -/
def point : ℝ × ℝ := (2, -1)

/-- Theorem: The line contains the point (2, -1) if and only if k = 7/2 -/
theorem line_contains_point :
  ∀ k : ℝ, line k point.1 point.2 ↔ k = 7/2 := by sorry

end line_contains_point_l3054_305414


namespace exponential_inequality_l3054_305486

theorem exponential_inequality (x : ℝ) : 
  Real.exp (2 * x - 1) < 1 ↔ x < (1 : ℝ) / 2 := by
  sorry

end exponential_inequality_l3054_305486


namespace gas_purchase_cost_l3054_305406

/-- Calculates the total cost of gas purchases given a price rollback and two separate purchases. -/
theorem gas_purchase_cost 
  (rollback : ℝ) 
  (initial_price : ℝ) 
  (liters_today : ℝ) 
  (liters_friday : ℝ) 
  (h1 : rollback = 0.4) 
  (h2 : initial_price = 1.4) 
  (h3 : liters_today = 10) 
  (h4 : liters_friday = 25) : 
  initial_price * liters_today + (initial_price - rollback) * liters_friday = 39 := by
sorry

end gas_purchase_cost_l3054_305406


namespace razorback_shop_profit_l3054_305431

/-- Calculates the total profit from selling various items in the Razorback shop -/
def total_profit (jersey_profit t_shirt_profit hoodie_profit hat_profit : ℕ)
                 (jerseys_sold t_shirts_sold hoodies_sold hats_sold : ℕ) : ℕ :=
  jersey_profit * jerseys_sold +
  t_shirt_profit * t_shirts_sold +
  hoodie_profit * hoodies_sold +
  hat_profit * hats_sold

/-- The total profit from the Razorback shop during the Arkansas and Texas Tech game -/
theorem razorback_shop_profit :
  total_profit 76 204 132 48 2 158 75 120 = 48044 := by
  sorry

end razorback_shop_profit_l3054_305431


namespace max_value_wxyz_l3054_305448

theorem max_value_wxyz (w x y z : ℝ) 
  (nonneg_w : w ≥ 0) (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_100 : w + x + y + z = 100) : 
  w * x + x * y + y * z ≤ 2500 := by
sorry

end max_value_wxyz_l3054_305448


namespace increasing_function_condition_l3054_305458

theorem increasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → ((a - 1) * x + 2) < ((a - 1) * y + 2)) →
  a > 1 := by sorry

end increasing_function_condition_l3054_305458


namespace adjacent_sum_6_l3054_305420

/-- Represents a 3x3 table filled with numbers from 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table is valid according to the given conditions --/
def is_valid_table (t : Table) : Prop :=
  (∀ i j, t i j ≠ 0) ∧  -- All cells are filled
  (∀ x, ∃! i j, t i j = x) ∧  -- Each number appears exactly once
  t 0 0 = 1 ∧ t 2 0 = 2 ∧ t 0 2 = 3 ∧ t 2 2 = 4 ∧  -- Given positions
  (∃ i j, t i j = 5 ∧ 
    (t (i-1) j + t (i+1) j + t i (j-1) + t i (j+1) : ℕ) = 9)  -- Sum around 5 is 9

/-- Sum of adjacent numbers to a given position --/
def adjacent_sum (t : Table) (i j : Fin 3) : ℕ :=
  (t (i-1) j + t (i+1) j + t i (j-1) + t i (j+1) : ℕ)

/-- The main theorem --/
theorem adjacent_sum_6 (t : Table) (h : is_valid_table t) :
  ∃ i j, t i j = 6 ∧ adjacent_sum t i j = 29 :=
sorry

end adjacent_sum_6_l3054_305420


namespace expression_equals_one_l3054_305446

theorem expression_equals_one (a : ℝ) (h : a = Real.sqrt 2) : 
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4)) = 1 := by
  sorry

end expression_equals_one_l3054_305446


namespace annual_bill_calculation_correct_l3054_305447

/-- Calculates the total annual bill for Noah's calls to his Grammy -/
def annual_bill_calculation : ℝ :=
  let weekday_duration : ℝ := 25
  let weekend_duration : ℝ := 45
  let holiday_duration : ℝ := 60
  
  let total_weekdays : ℝ := 260
  let total_weekends : ℝ := 104
  let total_holidays : ℝ := 11
  
  let intl_weekdays : ℝ := 130
  let intl_weekends : ℝ := 52
  let intl_holidays : ℝ := 6
  
  let local_weekday_rate : ℝ := 0.05
  let local_weekend_rate : ℝ := 0.06
  let local_holiday_rate : ℝ := 0.07
  
  let intl_weekday_rate : ℝ := 0.09
  let intl_weekend_rate : ℝ := 0.11
  let intl_holiday_rate : ℝ := 0.12
  
  let tax_rate : ℝ := 0.10
  let monthly_service_fee : ℝ := 2.99
  let intl_holiday_discount : ℝ := 0.05
  
  let local_weekday_cost := (total_weekdays - intl_weekdays) * weekday_duration * local_weekday_rate
  let local_weekend_cost := (total_weekends - intl_weekends) * weekend_duration * local_weekend_rate
  let local_holiday_cost := (total_holidays - intl_holidays) * holiday_duration * local_holiday_rate
  
  let intl_weekday_cost := intl_weekdays * weekday_duration * intl_weekday_rate
  let intl_weekend_cost := intl_weekends * weekend_duration * intl_weekend_rate
  let intl_holiday_cost := intl_holidays * holiday_duration * intl_holiday_rate * (1 - intl_holiday_discount)
  
  let total_call_cost := local_weekday_cost + local_weekend_cost + local_holiday_cost + 
                         intl_weekday_cost + intl_weekend_cost + intl_holiday_cost
  
  let total_tax := total_call_cost * tax_rate
  let total_service_fee := monthly_service_fee * 12
  
  total_call_cost + total_tax + total_service_fee

theorem annual_bill_calculation_correct : 
  annual_bill_calculation = 1042.20 := by sorry

end annual_bill_calculation_correct_l3054_305447


namespace plant_original_price_l3054_305462

/-- Given a 10% discount on a plant and a final price of $9, prove that the original price was $10. -/
theorem plant_original_price (discount_percentage : ℚ) (discounted_price : ℚ) : 
  discount_percentage = 10 →
  discounted_price = 9 →
  (1 - discount_percentage / 100) * 10 = discounted_price := by
  sorry

end plant_original_price_l3054_305462


namespace triangle_perimeter_from_average_side_length_l3054_305464

/-- Given a triangle with three sides where the average length of the sides is 12,
    prove that the perimeter of the triangle is 36. -/
theorem triangle_perimeter_from_average_side_length :
  ∀ (a b c : ℝ), (a + b + c) / 3 = 12 → a + b + c = 36 := by
  sorry

end triangle_perimeter_from_average_side_length_l3054_305464


namespace estimate_fish_population_l3054_305444

/-- Estimate the total number of fish in a pond using mark-recapture method -/
theorem estimate_fish_population (initially_caught marked_in_second_catch second_catch : ℕ) :
  initially_caught = 30 →
  marked_in_second_catch = 2 →
  second_catch = 50 →
  (initially_caught * second_catch) / marked_in_second_catch = 750 :=
by sorry

end estimate_fish_population_l3054_305444


namespace cyclist_stump_problem_l3054_305498

/-- Represents the problem of cyclists on a road with stumps -/
theorem cyclist_stump_problem 
  (road_length : ℝ)
  (speed_1 speed_2 : ℝ)
  (rest_time : ℕ)
  (num_stumps : ℕ) :
  road_length = 37 →
  speed_1 = 15 →
  speed_2 = 20 →
  rest_time > 0 →
  num_stumps > 1 →
  (road_length / speed_1 + num_stumps * rest_time / 60) =
  (road_length / speed_2 + num_stumps * (2 * rest_time) / 60) →
  num_stumps = 37 :=
by sorry

end cyclist_stump_problem_l3054_305498


namespace circle_x_plus_y_bounds_l3054_305449

-- Define the circle in polar form
def polar_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi/4) + 6 = 0

-- Define a point on the circle in Cartesian coordinates
def point_on_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 4

-- Theorem statement
theorem circle_x_plus_y_bounds :
  ∀ x y : ℝ, point_on_circle x y →
  (∃ θ : ℝ, polar_circle (Real.sqrt (x^2 + y^2)) θ) →
  2 ≤ x + y ∧ x + y ≤ 6 :=
by sorry

end circle_x_plus_y_bounds_l3054_305449


namespace fourth_vertex_exists_l3054_305472

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

-- Define properties of the quadrilateral
def is_cyclic (q : Quadrilateral) : Prop := sorry

def is_tangential (q : Quadrilateral) : Prop := sorry

def is_convex (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem fourth_vertex_exists 
  (A B C : Point) 
  (h_convex : is_convex ⟨A, B, C, C⟩) 
  (h_cyclic : ∀ D, is_cyclic ⟨A, B, C, D⟩) 
  (h_tangential : ∀ D, is_tangential ⟨A, B, C, D⟩) : 
  ∃ D, is_cyclic ⟨A, B, C, D⟩ ∧ is_tangential ⟨A, B, C, D⟩ ∧ is_convex ⟨A, B, C, D⟩ :=
sorry

end fourth_vertex_exists_l3054_305472


namespace range_of_g_minus_x_l3054_305490

def g (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem range_of_g_minus_x :
  Set.range (fun x => g x - x) ∩ Set.Icc (-2 : ℝ) 2 = Set.Icc 0 16 := by
  sorry

end range_of_g_minus_x_l3054_305490


namespace account_balance_after_transfer_l3054_305412

/-- Given an initial account balance and an amount transferred out, 
    calculate the final account balance. -/
def final_balance (initial : ℕ) (transferred : ℕ) : ℕ :=
  initial - transferred

theorem account_balance_after_transfer :
  final_balance 27004 69 = 26935 := by
  sorry

end account_balance_after_transfer_l3054_305412


namespace opposite_of_negative_2022_l3054_305481

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2022 : opposite (-2022) = 2022 := by
  sorry

end opposite_of_negative_2022_l3054_305481


namespace quadratic_inequality_solution_set_l3054_305435

theorem quadratic_inequality_solution_set (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ m ≥ 1/3 := by
  sorry

end quadratic_inequality_solution_set_l3054_305435


namespace pt_length_in_special_quadrilateral_l3054_305461

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (quad : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def intersection_point (p1 p2 p3 p4 : Point) : Point := sorry

theorem pt_length_in_special_quadrilateral 
  (PQRS : Quadrilateral) 
  (T : Point)
  (h_convex : is_convex PQRS)
  (h_PQ : distance PQRS.P PQRS.Q = 10)
  (h_RS : distance PQRS.R PQRS.S = 15)
  (h_PR : distance PQRS.P PQRS.R = 18)
  (h_T : T = intersection_point PQRS.P PQRS.R PQRS.Q PQRS.S)
  (h_equal_areas : triangle_area PQRS.P T PQRS.S = triangle_area PQRS.Q T PQRS.R) :
  distance PQRS.P T = 7.2 := by
  sorry

end pt_length_in_special_quadrilateral_l3054_305461


namespace A₁_Aₒ₂_independent_l3054_305491

/-- A bag containing black and white balls -/
structure Bag where
  black : ℕ
  white : ℕ

/-- An event in the probability space of drawing balls from the bag -/
structure Event (bag : Bag) where
  prob : ℝ
  nonneg : 0 ≤ prob
  le_one : prob ≤ 1

/-- Drawing a ball from the bag with replacement -/
def draw (bag : Bag) : Event bag := sorry

/-- The event of drawing a black ball -/
def black_ball (bag : Bag) : Event bag := sorry

/-- The event of drawing a white ball -/
def white_ball (bag : Bag) : Event bag := sorry

/-- The probability of an event -/
def P (bag : Bag) (e : Event bag) : ℝ := e.prob

/-- Two events are independent if the probability of their intersection
    is equal to the product of their individual probabilities -/
def independent (bag : Bag) (e1 e2 : Event bag) : Prop :=
  P bag (draw bag) = P bag e1 * P bag e2

/-- A₁: The event of drawing a black ball on the first draw -/
def A₁ (bag : Bag) : Event bag := black_ball bag

/-- A₂: The event of drawing a black ball on the second draw -/
def A₂ (bag : Bag) : Event bag := black_ball bag

/-- Aₒ₂: The complement of A₂ (drawing a white ball on the second draw) -/
def Aₒ₂ (bag : Bag) : Event bag := white_ball bag

/-- Theorem: A₁ and Aₒ₂ are independent events when drawing with replacement -/
theorem A₁_Aₒ₂_independent (bag : Bag) : independent bag (A₁ bag) (Aₒ₂ bag) := by
  sorry

end A₁_Aₒ₂_independent_l3054_305491


namespace solution_part_i_solution_part_ii_l3054_305483

/-- The function f(x) defined as |x - a| + |x - 1| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

/-- Theorem for part (I) of the problem -/
theorem solution_part_i :
  let a : ℝ := 2
  {x : ℝ | f a x < 4} = {x : ℝ | -1/2 < x ∧ x < 7/2} := by sorry

/-- Theorem for part (II) of the problem -/
theorem solution_part_ii :
  {a : ℝ | ∀ x, f a x ≥ 2} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end solution_part_i_solution_part_ii_l3054_305483


namespace mod_twelve_six_eight_l3054_305433

theorem mod_twelve_six_eight (m : ℕ) : 12^6 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 0 := by
  sorry

end mod_twelve_six_eight_l3054_305433


namespace joan_remaining_oranges_l3054_305459

theorem joan_remaining_oranges (joan_initial : ℕ) (sara_sold : ℕ) (joan_remaining : ℕ) : 
  joan_initial = 37 → sara_sold = 10 → joan_remaining = joan_initial - sara_sold → joan_remaining = 27 := by
  sorry

end joan_remaining_oranges_l3054_305459


namespace largest_sum_is_994_l3054_305428

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the sum of XXX + YY + Z -/
def sum (X Y Z : Digit) : ℕ := 111 * X.val + 11 * Y.val + Z.val

theorem largest_sum_is_994 :
  ∃ (X Y Z : Digit), X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
    sum X Y Z ≤ 999 ∧
    (∀ (A B C : Digit), A ≠ B ∧ A ≠ C ∧ B ≠ C → sum A B C ≤ sum X Y Z) ∧
    sum X Y Z = 994 ∧
    X = Y ∧ Y ≠ Z :=
by sorry

end largest_sum_is_994_l3054_305428


namespace teal_survey_result_l3054_305476

/-- Represents the survey results about teal color perception -/
structure TealSurvey where
  total : ℕ
  more_blue : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe teal is "more green" -/
def more_green (survey : TealSurvey) : ℕ :=
  survey.total - (survey.more_blue - survey.both) - survey.both - survey.neither

/-- Theorem stating the result of the teal color survey -/
theorem teal_survey_result : 
  let survey : TealSurvey := {
    total := 150,
    more_blue := 90,
    both := 40,
    neither := 20
  }
  more_green survey = 80 := by sorry

end teal_survey_result_l3054_305476


namespace p_sufficient_not_necessary_for_not_q_l3054_305494

/-- Given conditions p and q, prove that p is a sufficient but not necessary condition for ¬q -/
theorem p_sufficient_not_necessary_for_not_q :
  ∀ x : ℝ,
  (0 < x ∧ x ≤ 1) →  -- condition p
  ((1 / x < 1) → False) →  -- ¬q
  ∃ y : ℝ, ((1 / y < 1) → False) ∧ ¬(0 < y ∧ y ≤ 1) :=
by sorry


end p_sufficient_not_necessary_for_not_q_l3054_305494


namespace maximize_x_cubed_y_fourth_l3054_305416

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2 * y = 35) :
  x^3 * y^4 ≤ 21^3 * 7^4 ∧ 
  (x^3 * y^4 = 21^3 * 7^4 ↔ x = 21 ∧ y = 7) :=
sorry

end maximize_x_cubed_y_fourth_l3054_305416


namespace repeating_decimal_fraction_sum_l3054_305422

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n : ℚ) / d = 0.714714714 ∧ 
  (∀ (n' d' : ℕ), (n' : ℚ) / d' = 0.714714714 → n ≤ n' ∧ d ≤ d') ∧
  n + d = 571 :=
sorry

end repeating_decimal_fraction_sum_l3054_305422


namespace mona_monday_distance_l3054_305441

/-- Represents the distance biked on a given day -/
structure DailyBike where
  distance : ℝ
  time : ℝ
  speed : ℝ

/-- Represents Mona's weekly biking schedule -/
structure WeeklyBike where
  monday : DailyBike
  wednesday : DailyBike
  saturday : DailyBike
  total_distance : ℝ

theorem mona_monday_distance (w : WeeklyBike) :
  w.total_distance = 30 ∧
  w.wednesday.distance = 12 ∧
  w.wednesday.time = 2 ∧
  w.saturday.distance = 2 * w.monday.distance ∧
  w.monday.speed = 15 ∧
  w.monday.time = 1.5 ∧
  w.saturday.speed = 0.8 * w.monday.speed →
  w.monday.distance = 6 := by
  sorry

end mona_monday_distance_l3054_305441


namespace total_gum_packages_l3054_305477

theorem total_gum_packages : ∀ (robin_pieces_per_package : ℕ) 
                               (robin_extra_pieces : ℕ) 
                               (robin_total_pieces : ℕ)
                               (alex_pieces_per_package : ℕ) 
                               (alex_extra_pieces : ℕ) 
                               (alex_total_pieces : ℕ),
  robin_pieces_per_package = 7 →
  robin_extra_pieces = 6 →
  robin_total_pieces = 41 →
  alex_pieces_per_package = 5 →
  alex_extra_pieces = 3 →
  alex_total_pieces = 23 →
  (robin_total_pieces - robin_extra_pieces) / robin_pieces_per_package +
  (alex_total_pieces - alex_extra_pieces) / alex_pieces_per_package = 9 :=
by
  sorry

end total_gum_packages_l3054_305477


namespace school_play_ticket_ratio_l3054_305474

theorem school_play_ticket_ratio :
  ∀ (total_tickets student_tickets adult_tickets : ℕ),
    total_tickets = 366 →
    adult_tickets = 122 →
    total_tickets = student_tickets + adult_tickets →
    ∃ (k : ℕ), student_tickets = k * adult_tickets →
    (student_tickets : ℚ) / (adult_tickets : ℚ) = 2 / 1 :=
by sorry

end school_play_ticket_ratio_l3054_305474


namespace expansion_equality_l3054_305438

theorem expansion_equality (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4) = x^4 - 16 := by
  sorry

end expansion_equality_l3054_305438


namespace B_power_6_l3054_305427

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 4, 5]

theorem B_power_6 : 
  B^6 = 1715 • B - 16184 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end B_power_6_l3054_305427


namespace percentage_B_of_D_l3054_305484

theorem percentage_B_of_D (A B C D : ℝ) 
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) :
  B = 1.1115 * D := by
sorry

end percentage_B_of_D_l3054_305484


namespace felix_axe_sharpening_cost_l3054_305403

/-- Calculates the total cost of axe sharpening given the number of trees chopped,
    trees per sharpening, and cost per sharpening. -/
def axeSharpeningCost (treesChopped : ℕ) (treesPerSharpening : ℕ) (costPerSharpening : ℕ) : ℕ :=
  ((treesChopped - 1) / treesPerSharpening + 1) * costPerSharpening

/-- Proves that given the conditions, the total cost of axe sharpening is $35. -/
theorem felix_axe_sharpening_cost :
  ∀ (treesChopped : ℕ),
    treesChopped ≥ 91 →
    axeSharpeningCost treesChopped 13 5 = 35 := by
  sorry

end felix_axe_sharpening_cost_l3054_305403


namespace sum_of_ages_sum_of_ages_proof_l3054_305405

theorem sum_of_ages : ℕ → ℕ → Prop :=
  fun john_age father_age =>
    (john_age = 15) →
    (father_age = 2 * john_age + 32) →
    (john_age + father_age = 77)

-- Proof
theorem sum_of_ages_proof : sum_of_ages 15 62 := by
  sorry

end sum_of_ages_sum_of_ages_proof_l3054_305405


namespace geometric_mean_of_45_and_80_l3054_305495

theorem geometric_mean_of_45_and_80 : 
  ∃ x : ℝ, (x ^ 2 = 45 * 80) ∧ (x = 60 ∨ x = -60) := by
  sorry

end geometric_mean_of_45_and_80_l3054_305495


namespace map_scale_l3054_305418

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (real_km : ℝ) (h : map_cm / 15 = real_km / 90) :
  (20 * real_km) / map_cm = 120 :=
sorry

end map_scale_l3054_305418


namespace expand_and_simplify_l3054_305488

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by sorry

end expand_and_simplify_l3054_305488


namespace cousins_distribution_l3054_305482

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to accommodate -/
def num_cousins : ℕ := 5

/-- The number of ways to distribute the cousins is 51 -/
theorem cousins_distribution :
  distribute num_cousins num_rooms = 51 := by sorry

end cousins_distribution_l3054_305482


namespace floor_sqrt_50_squared_l3054_305410

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l3054_305410


namespace common_tangent_implies_a_value_l3054_305425

/-- Two curves with a common tangent line at their common point imply a specific value for a parameter -/
theorem common_tangent_implies_a_value (e : ℝ) (a s t : ℝ) : 
  (t = (1 / (2 * Real.exp 1)) * s^2) →  -- Point P(s,t) is on the first curve
  (t = a * Real.log s) →                -- Point P(s,t) is on the second curve
  ((s / Real.exp 1) = (a / s)) →        -- Slopes are equal at point P(s,t)
  (a = 1) := by
sorry

end common_tangent_implies_a_value_l3054_305425


namespace circles_externally_tangent_l3054_305419

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

/-- The equation of the first circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The equation of the second circle: x^2 + y^2 - 10x + 16 = 0 -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 16 = 0

theorem circles_externally_tangent :
  externally_tangent (0, 0) (5, 0) 2 3 :=
by sorry

#check circles_externally_tangent

end circles_externally_tangent_l3054_305419


namespace variance_transformation_l3054_305443

/-- Given a sample of 10 data points, this function represents their variance. -/
def sample_variance (x : Fin 10 → ℝ) : ℝ := sorry

/-- Given a sample of 10 data points, this function represents the variance of the transformed data. -/
def transformed_variance (x : Fin 10 → ℝ) : ℝ := 
  sample_variance (fun i => 2 * x i - 1)

/-- Theorem stating the relationship between the original variance and the transformed variance. -/
theorem variance_transformation (x : Fin 10 → ℝ) 
  (h : sample_variance x = 8) : transformed_variance x = 32 := by
  sorry

end variance_transformation_l3054_305443


namespace evaluate_expression_l3054_305455

theorem evaluate_expression (x z : ℝ) (hx : x = 2) (hz : z = 1) :
  z * (z - 4 * x) = -7 := by sorry

end evaluate_expression_l3054_305455


namespace max_y_coordinate_ellipse_l3054_305492

theorem max_y_coordinate_ellipse :
  let f (x y : ℝ) := x^2 / 49 + (y + 3)^2 / 25
  ∀ x y : ℝ, f x y = 1 → y ≤ 2 ∧ ∃ x₀ : ℝ, f x₀ 2 = 1 := by
  sorry

end max_y_coordinate_ellipse_l3054_305492


namespace zeros_before_first_nonzero_digit_l3054_305489

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d = 64000) :
  (∃ k : ℕ, (7 : ℚ) / d = k / 10^(n + 1) ∧ k % 10 ≠ 0 ∧ k < 10^n) → n = 4 :=
sorry

end zeros_before_first_nonzero_digit_l3054_305489


namespace tangent_line_equation_l3054_305469

-- Define the function f(x) = -x^3 + 3x^2
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_equation (a : ℝ) :
  ∃ b : ℝ, f' a = 3 ∧ f a = 3*a + b :=
sorry

end tangent_line_equation_l3054_305469


namespace high_octane_half_cost_l3054_305473

/-- Represents the composition and cost of a fuel mixture -/
structure FuelMixture where
  high_octane : ℚ
  regular_octane : ℚ
  cost_ratio : ℚ
  total : ℚ

/-- Calculates the fraction of total cost due to high octane fuel -/
def high_octane_cost_fraction (fuel : FuelMixture) : ℚ :=
  (fuel.high_octane * fuel.cost_ratio) / ((fuel.high_octane * fuel.cost_ratio) + fuel.regular_octane)

/-- Theorem: For a fuel mixture with 15 parts high octane and 45 parts regular octane,
    where high octane costs 3 times as much as regular octane,
    the fraction of the total cost due to high octane is 1/2 -/
theorem high_octane_half_cost (fuel : FuelMixture)
  (h1 : fuel.high_octane = 15)
  (h2 : fuel.regular_octane = 45)
  (h3 : fuel.cost_ratio = 3)
  (h4 : fuel.total = fuel.high_octane + fuel.regular_octane) :
  high_octane_cost_fraction fuel = 1/2 := by
  sorry

end high_octane_half_cost_l3054_305473


namespace f_intersects_x_axis_min_distance_between_roots_range_of_a_l3054_305415

/-- The quadratic function f(x) = x^2 - 2ax - 2(a + 1) -/
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 2*(a + 1)

theorem f_intersects_x_axis (a : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 := by
  sorry

theorem min_distance_between_roots (a : ℝ) :
  ∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ |x₁ - x₂| ≥ 2 ∧ (∀ y₁ y₂ : ℝ, f a y₁ = 0 → f a y₂ = 0 → |y₁ - y₂| ≥ 2) := by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > -1 → f a x + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := by
  sorry

end f_intersects_x_axis_min_distance_between_roots_range_of_a_l3054_305415


namespace flour_needed_l3054_305479

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℕ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_extra : ℕ := 2

/-- The total amount of flour needed by Katie and Sheila -/
def total_flour : ℕ := katie_flour + (katie_flour + sheila_extra)

theorem flour_needed : total_flour = 8 := by
  sorry

end flour_needed_l3054_305479


namespace simplify_expression_l3054_305460

theorem simplify_expression : 2 - (2 / (2 + 2 * Real.sqrt 2)) + (2 / (2 - 2 * Real.sqrt 2)) = 2 := by
  sorry

end simplify_expression_l3054_305460


namespace cube_not_always_positive_l3054_305452

theorem cube_not_always_positive : ¬ (∀ x : ℝ, x^3 > 0) := by
  sorry

end cube_not_always_positive_l3054_305452


namespace projection_region_area_l3054_305465

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- The region inside the trapezoid with the given projection property -/
def ProjectionRegion (t : IsoscelesTrapezoid) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem -/
theorem projection_region_area (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 1) (h2 : t.base2 = 2) (h3 : t.height = 1) : 
  area (ProjectionRegion t) = 5/8 := by
  sorry

end projection_region_area_l3054_305465


namespace total_age_in_three_years_l3054_305453

def age_problem (sam sue kendra : ℕ) : Prop :=
  sam = 2 * sue ∧ 
  kendra = 3 * sam ∧ 
  kendra = 18

theorem total_age_in_three_years 
  (sam sue kendra : ℕ) 
  (h : age_problem sam sue kendra) : 
  (sue + 3) + (sam + 3) + (kendra + 3) = 36 := by
  sorry

end total_age_in_three_years_l3054_305453


namespace probability_divisible_by_2_3_5_or_7_l3054_305496

theorem probability_divisible_by_2_3_5_or_7 : 
  let S : Finset ℕ := Finset.range 120
  let A : Finset ℕ := S.filter (fun n => n % 2 = 0)
  let B : Finset ℕ := S.filter (fun n => n % 3 = 0)
  let C : Finset ℕ := S.filter (fun n => n % 5 = 0)
  let D : Finset ℕ := S.filter (fun n => n % 7 = 0)
  (A ∪ B ∪ C ∪ D).card / S.card = 13 / 15 := by
sorry


end probability_divisible_by_2_3_5_or_7_l3054_305496


namespace min_perimeter_l3054_305401

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  equalSide : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.equalSide + t.base

/-- Represents the pair of isosceles triangles in the problem -/
structure TrianglePair where
  t1 : IsoscelesTriangle
  t2 : IsoscelesTriangle

/-- The conditions given in the problem -/
def satisfiesConditions (pair : TrianglePair) : Prop :=
  let t1 := pair.t1
  let t2 := pair.t2
  -- Same perimeter
  perimeter t1 = perimeter t2 ∧
  -- Ratio of bases is 10:9
  10 * t2.base = 9 * t1.base ∧
  -- Base relations
  t1.base = 2 * t1.equalSide - 12 ∧
  t2.base = 3 * t2.equalSide - 30 ∧
  -- Non-congruent
  t1 ≠ t2

theorem min_perimeter (pair : TrianglePair) :
  satisfiesConditions pair → perimeter pair.t1 ≥ 228 :=
sorry

end min_perimeter_l3054_305401


namespace olympiad_survey_l3054_305499

theorem olympiad_survey (P : ℝ) (a b c d : ℝ) 
  (h1 : (a + b + d) / P = 0.9)
  (h2 : (a + c + d) / P = 0.6)
  (h3 : (b + c + d) / P = 0.9)
  (h4 : a + b + c + d = P)
  (h5 : P > 0) :
  d / P = 0.4 := by
  sorry

end olympiad_survey_l3054_305499


namespace locus_of_point_in_cube_l3054_305407

/-- The locus of a point M in a unit cube, where the sum of squares of distances 
    from M to the faces of the cube is constant, is a sphere centered at (1/2, 1/2, 1/2). -/
theorem locus_of_point_in_cube (x y z : ℝ) (k : ℝ) : 
  (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) →
  x^2 + (1 - x)^2 + y^2 + (1 - y)^2 + z^2 + (1 - z)^2 = k →
  ∃ r : ℝ, (x - 1/2)^2 + (y - 1/2)^2 + (z - 1/2)^2 = r^2 :=
by sorry


end locus_of_point_in_cube_l3054_305407


namespace power_of_product_l3054_305437

theorem power_of_product (a b : ℝ) : (3 * a * b)^2 = 9 * a^2 * b^2 := by
  sorry

end power_of_product_l3054_305437


namespace tangent_line_to_circle_l3054_305478

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), x + y = r ∧ x^2 + y^2 = r) → 
  (∀ (x y : ℝ), x + y = r → x^2 + y^2 ≥ r) → 
  r = 2 := by
sorry

end tangent_line_to_circle_l3054_305478


namespace polynomial_equality_l3054_305426

theorem polynomial_equality (x : ℝ) (h : ℝ → ℝ) :
  (8 * x^4 - 4 * x^2 + 2 + h x = 2 * x^3 - 6 * x + 4) →
  (h x = -8 * x^4 + 2 * x^3 + 4 * x^2 - 6 * x + 2) :=
by sorry

end polynomial_equality_l3054_305426


namespace oliver_seashell_collection_l3054_305424

/-- Represents the number of seashells Oliver collected on a given day -/
structure DailyCollection where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total number of seashells Oliver has after Thursday -/
def totalAfterThursday (c : DailyCollection) : ℕ :=
  c.monday + c.tuesday / 2 + c.wednesday + 5

/-- Theorem stating that Oliver collected 71 seashells on Monday, Tuesday, and Wednesday -/
theorem oliver_seashell_collection (c : DailyCollection) :
  totalAfterThursday c = 76 →
  c.monday + c.tuesday + c.wednesday = 71 := by
  sorry

end oliver_seashell_collection_l3054_305424


namespace smallest_difference_is_one_l3054_305402

/-- Triangle with integer side lengths and specific ordering -/
structure OrderedTriangle where
  de : ℕ
  ef : ℕ
  fd : ℕ
  de_lt_ef : de < ef
  ef_le_fd : ef ≤ fd

/-- The perimeter of the triangle is 2050 -/
def hasPerimeter2050 (t : OrderedTriangle) : Prop :=
  t.de + t.ef + t.fd = 2050

/-- The triangle inequality holds -/
def satisfiesTriangleInequality (t : OrderedTriangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

theorem smallest_difference_is_one :
  ∃ (t : OrderedTriangle), 
    hasPerimeter2050 t ∧ 
    satisfiesTriangleInequality t ∧
    (∀ (u : OrderedTriangle), 
      hasPerimeter2050 u → satisfiesTriangleInequality u → 
      u.ef - u.de ≥ t.ef - t.de) ∧
    t.ef - t.de = 1 :=
  sorry

end smallest_difference_is_one_l3054_305402


namespace markers_count_l3054_305456

/-- Given a ratio of pens : pencils : markers as 2 : 2 : 5, and 10 pens, the number of markers is 25. -/
theorem markers_count (pens pencils markers : ℕ) : 
  pens = 10 → 
  pens * 5 = markers * 2 → 
  markers = 25 := by
  sorry

end markers_count_l3054_305456


namespace intersection_M_N_l3054_305457

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by sorry

end intersection_M_N_l3054_305457


namespace opposite_signs_and_larger_absolute_value_l3054_305411

theorem opposite_signs_and_larger_absolute_value (a b : ℚ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0) ∧ 
  (max (abs a) (abs b) = abs (max a b)) :=
sorry

end opposite_signs_and_larger_absolute_value_l3054_305411


namespace dirichlet_approximation_l3054_305400

theorem dirichlet_approximation (x : ℝ) (h_irr : Irrational x) (h_pos : 0 < x) :
  ∀ N : ℕ, ∃ p q : ℤ, N < q ∧ 0 < q ∧ |x - (p : ℝ) / (q : ℝ)| < 1 / (q : ℝ)^2 := by
  sorry

end dirichlet_approximation_l3054_305400


namespace johns_age_l3054_305434

theorem johns_age (john_age dad_age : ℕ) 
  (h1 : john_age = dad_age - 24)
  (h2 : john_age + dad_age = 68) : 
  john_age = 22 := by
  sorry

end johns_age_l3054_305434


namespace remainder_proof_l3054_305454

theorem remainder_proof (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 10 = 2 →
  (7 * c) % 10 = 3 →
  (8 * b) % 10 = (4 + b) % 10 →
  (2 * a + b + 3 * c) % 10 = 1 :=
by sorry

end remainder_proof_l3054_305454


namespace cos_negative_seventy_nine_sixths_pi_l3054_305487

theorem cos_negative_seventy_nine_sixths_pi :
  Real.cos (-79/6 * Real.pi) = -Real.sqrt 3 / 2 := by
  sorry

end cos_negative_seventy_nine_sixths_pi_l3054_305487


namespace abs_inequality_solution_set_l3054_305480

theorem abs_inequality_solution_set :
  {x : ℝ | |2*x + 1| < 3} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end abs_inequality_solution_set_l3054_305480


namespace max_y_coordinate_ellipse_l3054_305432

theorem max_y_coordinate_ellipse :
  ∀ x y : ℝ, x^2/25 + (y-3)^2/25 = 0 → y ≤ 3 :=
by sorry

end max_y_coordinate_ellipse_l3054_305432


namespace work_problem_solution_l3054_305421

def work_problem (a_rate b_rate : ℝ) (combined_days : ℝ) : Prop :=
  a_rate = 2 * b_rate →
  combined_days = 6 →
  b_rate * (a_rate + b_rate)⁻¹ * combined_days = 18

theorem work_problem_solution :
  ∀ (a_rate b_rate combined_days : ℝ),
    work_problem a_rate b_rate combined_days :=
by
  sorry

end work_problem_solution_l3054_305421


namespace quadratic_single_solution_l3054_305466

theorem quadratic_single_solution (p : ℝ) : 
  (∃! y : ℝ, 2 * y^2 - 8 * y = p) → p = -8 := by
  sorry

end quadratic_single_solution_l3054_305466


namespace contemporary_probability_correct_l3054_305445

/-- The duration in years of the period considered -/
def period : ℕ := 800

/-- The lifespan of each mathematician in years -/
def lifespan : ℕ := 150

/-- The probability that two mathematicians born within a given period
    are contemporaries, given their lifespans and assuming uniform distribution
    of birth years -/
def contemporaryProbability (p : ℕ) (l : ℕ) : ℚ :=
  let totalArea := p * p
  let nonOverlapArea := 2 * (p - l) * l / 2
  let overlapArea := totalArea - nonOverlapArea
  overlapArea / totalArea

theorem contemporary_probability_correct :
  contemporaryProbability period lifespan = 27125 / 32000 := by
  sorry

end contemporary_probability_correct_l3054_305445


namespace original_pencils_count_l3054_305430

/-- The number of pencils Mike placed in the drawer -/
def pencils_added : ℕ := 30

/-- The total number of pencils now in the drawer -/
def total_pencils : ℕ := 71

/-- The original number of pencils in the drawer -/
def original_pencils : ℕ := total_pencils - pencils_added

theorem original_pencils_count : original_pencils = 41 := by
  sorry

end original_pencils_count_l3054_305430


namespace sue_fill_time_l3054_305468

def jim_rate : ℚ := 1 / 30
def tony_rate : ℚ := 1 / 90
def combined_rate : ℚ := 1 / 15

def sue_time : ℚ := 45

theorem sue_fill_time (sue_rate : ℚ) 
  (h1 : sue_rate = 1 / sue_time)
  (h2 : jim_rate + sue_rate + tony_rate = combined_rate) : 
  sue_time = 45 := by sorry

end sue_fill_time_l3054_305468


namespace max_digit_diff_l3054_305450

/-- Two-digit number representation -/
def two_digit_number (tens units : Nat) : Nat :=
  10 * tens + units

/-- The difference between two two-digit numbers -/
def digit_diff (a b : Nat) : Int :=
  (two_digit_number a b : Int) - (two_digit_number b a)

theorem max_digit_diff :
  ∀ a b : Nat,
    a ≠ b →
    a ≠ 0 →
    b ≠ 0 →
    a ≤ 9 →
    b ≤ 9 →
    digit_diff a b ≤ 72 ∧
    ∃ a b : Nat, a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ digit_diff a b = 72 :=
by sorry

end max_digit_diff_l3054_305450


namespace last_digit_sum_l3054_305408

theorem last_digit_sum (n : ℕ) : (3^1991 + 1991^3) % 10 = 8 := by
  sorry

end last_digit_sum_l3054_305408


namespace f_nonnegative_when_a_is_one_f_two_extreme_points_condition_l3054_305451

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.cos x - x

-- Theorem 1: When a = 1, f(x) ≥ 0 for all x
theorem f_nonnegative_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 0 := by sorry

-- Theorem 2: f(x) has two extreme points in (0, π) iff 0 < a < e^(-π)
theorem f_two_extreme_points_condition (a : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < π ∧
   (∀ z : ℝ, 0 < z ∧ z < π → f a z ≤ f a x ∨ f a z ≤ f a y) ∧
   (∀ w : ℝ, x < w ∧ w < y → f a w > f a x ∧ f a w > f a y)) ↔
  (0 < a ∧ a < Real.exp (-π)) := by sorry

end f_nonnegative_when_a_is_one_f_two_extreme_points_condition_l3054_305451


namespace candy_distribution_l3054_305439

theorem candy_distribution (total_candy : ℕ) (family_members : ℕ) 
  (h1 : total_candy = 45) (h2 : family_members = 5) : 
  total_candy % family_members = 0 := by
  sorry

end candy_distribution_l3054_305439


namespace product_of_sum_and_difference_l3054_305470

theorem product_of_sum_and_difference : 
  let a : ℝ := 4.93
  let b : ℝ := 3.78
  (a + b) * (a - b) = 10.0165 := by
  sorry

end product_of_sum_and_difference_l3054_305470
