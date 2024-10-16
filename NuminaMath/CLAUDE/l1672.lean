import Mathlib

namespace NUMINAMATH_CALUDE_infinite_chain_resistance_l1672_167215

/-- The resistance of a single resistor in the chain -/
def R₀ : ℝ := 50

/-- The resistance of an infinitely long chain of identical resistors -/
noncomputable def R_X : ℝ := R₀ * (1 + Real.sqrt 5) / 2

/-- Theorem stating that R_X satisfies the equation for the infinite chain resistance -/
theorem infinite_chain_resistance : R_X = R₀ + (R₀ * R_X) / (R₀ + R_X) := by
  sorry

end NUMINAMATH_CALUDE_infinite_chain_resistance_l1672_167215


namespace NUMINAMATH_CALUDE_original_salary_is_twenty_thousand_l1672_167216

/-- Calculates the original salary of employees given the conditions of Emily's salary change --/
def calculate_original_salary (emily_original_salary : ℕ) (emily_new_salary : ℕ) (num_employees : ℕ) (new_employee_salary : ℕ) : ℕ :=
  let salary_difference := emily_original_salary - emily_new_salary
  let salary_increase_per_employee := salary_difference / num_employees
  new_employee_salary - salary_increase_per_employee

/-- Theorem stating that given the problem conditions, the original salary of each employee was $20,000 --/
theorem original_salary_is_twenty_thousand :
  calculate_original_salary 1000000 850000 10 35000 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_is_twenty_thousand_l1672_167216


namespace NUMINAMATH_CALUDE_milk_cost_l1672_167239

/-- Proves that the cost of a gallon of milk is $3 given the total groceries cost and the costs of other items. -/
theorem milk_cost (total : ℝ) (cereal_price cereal_qty : ℝ) (banana_price banana_qty : ℝ) 
  (apple_price apple_qty : ℝ) (cookie_qty : ℝ) :
  total = 25 ∧ 
  cereal_price = 3.5 ∧ cereal_qty = 2 ∧
  banana_price = 0.25 ∧ banana_qty = 4 ∧
  apple_price = 0.5 ∧ apple_qty = 4 ∧
  cookie_qty = 2 →
  ∃ (milk_price : ℝ),
    milk_price = 3 ∧
    total = cereal_price * cereal_qty + banana_price * banana_qty + 
            apple_price * apple_qty + milk_price + 2 * milk_price * cookie_qty :=
by sorry

end NUMINAMATH_CALUDE_milk_cost_l1672_167239


namespace NUMINAMATH_CALUDE_final_s_is_negative_one_l1672_167264

/-- Represents the state of the algorithm at each iteration -/
structure AlgorithmState where
  s : Int
  iterations : Nat

/-- The algorithm's step function -/
def step (state : AlgorithmState) : AlgorithmState :=
  if state.iterations % 2 = 0 then
    { s := state.s + 1, iterations := state.iterations + 1 }
  else
    { s := state.s - 1, iterations := state.iterations + 1 }

/-- The initial state of the algorithm -/
def initialState : AlgorithmState := { s := 0, iterations := 0 }

/-- Applies the step function n times -/
def applyNTimes (n : Nat) (state : AlgorithmState) : AlgorithmState :=
  match n with
  | 0 => state
  | n + 1 => step (applyNTimes n state)

/-- The final state after 5 iterations -/
def finalState : AlgorithmState := applyNTimes 5 initialState

/-- The theorem stating that the final value of s is -1 -/
theorem final_s_is_negative_one : finalState.s = -1 := by
  sorry


end NUMINAMATH_CALUDE_final_s_is_negative_one_l1672_167264


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l1672_167212

/-- The probability of drawing 3 blue jellybeans in a row without replacement -/
def probability : ℚ := 10526315789473684 / 100000000000000000

/-- The number of blue jellybeans in the bag -/
def blue_jellybeans : ℕ := 10

/-- Calculates the probability of drawing 3 blue jellybeans in a row without replacement -/
def calculate_probability (red : ℕ) : ℚ :=
  (blue_jellybeans : ℚ) / (blue_jellybeans + red) *
  ((blue_jellybeans - 1) : ℚ) / (blue_jellybeans + red - 1) *
  ((blue_jellybeans - 2) : ℚ) / (blue_jellybeans + red - 2)

/-- Theorem stating that the number of red jellybeans is 10 -/
theorem red_jellybeans_count : ∃ (red : ℕ), calculate_probability red = probability ∧ red = 10 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l1672_167212


namespace NUMINAMATH_CALUDE_integral_inequality_l1672_167271

theorem integral_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a ≤ b) :
  (2 / Real.sqrt 3) * Real.arctan ((2 * (b^2 - a^2)) / ((a^2 + 2) * (b^2 + 2))) ≤
  (∫ (x : ℝ) in a..b, ((x^2 + 1) * (x^2 + x + 1)) / ((x^3 + x^2 + 1) * (x^3 + x + 1))) ∧
  (∫ (x : ℝ) in a..b, ((x^2 + 1) * (x^2 + x + 1)) / ((x^3 + x^2 + 1) * (x^3 + x + 1))) ≤
  (4 / Real.sqrt 3) * Real.arctan (((b - a) * Real.sqrt 3) / (a + b + 2 * (1 + a * b))) :=
by sorry

end NUMINAMATH_CALUDE_integral_inequality_l1672_167271


namespace NUMINAMATH_CALUDE_max_volume_at_eight_l1672_167226

/-- The volume of the box as a function of the side length of the removed square -/
def boxVolume (x : ℝ) : ℝ := (48 - 2*x)^2 * x

/-- The derivative of the box volume with respect to x -/
def boxVolumeDerivative (x : ℝ) : ℝ := (48 - 2*x) * (48 - 6*x)

theorem max_volume_at_eight :
  ∃ (x : ℝ), 0 < x ∧ x < 24 ∧
  (∀ (y : ℝ), 0 < y ∧ y < 24 → boxVolume y ≤ boxVolume x) ∧
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_max_volume_at_eight_l1672_167226


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1672_167231

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a ≥ 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1672_167231


namespace NUMINAMATH_CALUDE_some_number_value_l1672_167204

theorem some_number_value (x : ℝ) : 
  7^8 - 6/x + 9^3 + 3 + 12 = 95 → x = 1 / 960908.333 :=
by sorry

end NUMINAMATH_CALUDE_some_number_value_l1672_167204


namespace NUMINAMATH_CALUDE_inequality_proof_l1672_167285

theorem inequality_proof (x y z a n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_xyz : x * y * z = 1)
  (ha : a ≥ 1) (hn : n ≥ 1) : 
  x^n / ((a+y)*(a+z)) + y^n / ((a+z)*(a+x)) + z^n / ((a+x)*(a+y)) ≥ 3 / (1+a)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1672_167285


namespace NUMINAMATH_CALUDE_log_equation_implies_m_value_l1672_167274

theorem log_equation_implies_m_value 
  (m n : ℝ) (c : ℝ) 
  (h : Real.log (m^2) = c - 2 * Real.log n) :
  m = Real.sqrt (Real.exp c / n) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_implies_m_value_l1672_167274


namespace NUMINAMATH_CALUDE_jasons_books_l1672_167259

theorem jasons_books (books_per_shelf : ℕ) (num_shelves : ℕ) (h1 : books_per_shelf = 45) (h2 : num_shelves = 7) :
  books_per_shelf * num_shelves = 315 := by
  sorry

end NUMINAMATH_CALUDE_jasons_books_l1672_167259


namespace NUMINAMATH_CALUDE_even_function_extension_l1672_167250

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_nonpos : ∀ x ≤ 0, f x = x^3 - x^2) :
  ∀ x > 0, f x = -x^3 - x^2 :=
by sorry

end NUMINAMATH_CALUDE_even_function_extension_l1672_167250


namespace NUMINAMATH_CALUDE_find_k_l1672_167288

theorem find_k (k : ℝ) (h : 24 / k = 4) : k = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l1672_167288


namespace NUMINAMATH_CALUDE_correct_pies_left_l1672_167254

/-- Calculates the number of pies left after baking and dropping some -/
def pies_left (oven_capacity : ℕ) (num_batches : ℕ) (dropped_pies : ℕ) : ℕ :=
  oven_capacity * num_batches - dropped_pies

theorem correct_pies_left :
  let oven_capacity : ℕ := 5
  let num_batches : ℕ := 7
  let dropped_pies : ℕ := 8
  pies_left oven_capacity num_batches dropped_pies = 27 := by
  sorry

end NUMINAMATH_CALUDE_correct_pies_left_l1672_167254


namespace NUMINAMATH_CALUDE_min_exercise_hours_l1672_167205

/-- Represents the exercise data for a month -/
structure ExerciseData where
  days_20min : Nat
  days_40min : Nat
  days_2hours : Nat
  min_exercise_time : Nat
  max_exercise_time : Nat

/-- Calculates the minimum number of hours exercised in a month -/
def min_hours_exercised (data : ExerciseData) : Rat :=
  let hours_2hours := data.days_2hours * 2
  let hours_40min := (data.days_40min - data.days_2hours) * 2 / 3
  let hours_20min := (data.days_20min - data.days_40min) * 1 / 3
  hours_2hours + hours_40min + hours_20min

/-- Theorem stating the minimum number of hours exercised -/
theorem min_exercise_hours (data : ExerciseData) 
  (h1 : data.days_20min = 26)
  (h2 : data.days_40min = 24)
  (h3 : data.days_2hours = 4)
  (h4 : data.min_exercise_time = 20)
  (h5 : data.max_exercise_time = 120) :
  min_hours_exercised data = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_exercise_hours_l1672_167205


namespace NUMINAMATH_CALUDE_divisibility_of_repeating_digits_l1672_167208

theorem divisibility_of_repeating_digits : ∃ (k m : ℕ), k > 0 ∧ (1989 * (10^(4*k) - 1) / 9) * 10^m % 1988 = 0 ∧
                                          ∃ (n : ℕ), n > 0 ∧ (1988 * (10^(4*n) - 1) / 9) % 1989 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_repeating_digits_l1672_167208


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1672_167210

theorem constant_term_binomial_expansion :
  ∃ (n : ℕ), n = 11 ∧ 
  (∀ (r : ℕ), (15 : ℝ) - (3 / 2 : ℝ) * r = 0 → r = 10) ∧
  (∀ (k : ℕ), k ≠ n - 1 → 
    ∃ (c : ℝ), c ≠ 0 ∧ 
    (Nat.choose 15 k * (6 : ℝ)^(15 - k) * (-1 : ℝ)^k) * (0 : ℝ)^(15 - (3 * k) / 2) = c) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1672_167210


namespace NUMINAMATH_CALUDE_minimum_crossing_time_l1672_167246

/-- Represents an individual with their crossing time -/
structure Individual where
  name : String
  time : Nat

/-- Represents a crossing of the bridge -/
inductive Crossing
  | Single : Individual → Crossing
  | Pair : Individual → Individual → Crossing

/-- Calculates the time taken for a single crossing -/
def crossingTime (c : Crossing) : Nat :=
  match c with
  | Crossing.Single i => i.time
  | Crossing.Pair i j => max i.time j.time

/-- The problem statement -/
theorem minimum_crossing_time
  (a b c d : Individual)
  (ha : a.time = 2)
  (hb : b.time = 3)
  (hc : c.time = 8)
  (hd : d.time = 10)
  (crossings : List Crossing)
  (hcross : crossings = [Crossing.Pair a b, Crossing.Single a, Crossing.Pair c d, Crossing.Single b, Crossing.Pair a b]) :
  (crossings.map crossingTime).sum = 21 ∧
  ∀ (otherCrossings : List Crossing),
    (otherCrossings.map crossingTime).sum ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_minimum_crossing_time_l1672_167246


namespace NUMINAMATH_CALUDE_total_treats_value_l1672_167224

-- Define constants for given values
def hotel_per_night : ℝ := 4000
def hotel_nights : ℝ := 2
def hotel_discount : ℝ := 0.05
def car_value : ℝ := 30000
def car_tax : ℝ := 0.10
def house_multiplier : ℝ := 4
def house_tax : ℝ := 0.02
def yacht_multiplier : ℝ := 2
def yacht_discount : ℝ := 0.07
def gold_multiplier : ℝ := 1.5
def gold_tax : ℝ := 0.03

-- Define functions for calculating values
def hotel_cost (per_night : ℝ) (nights : ℝ) (discount : ℝ) : ℝ :=
  per_night * nights * (1 - discount)

def car_cost (value : ℝ) (tax : ℝ) : ℝ :=
  value * (1 + tax)

def house_cost (car_value : ℝ) (multiplier : ℝ) (tax : ℝ) : ℝ :=
  car_value * multiplier * (1 + tax)

def yacht_cost (hotel_value : ℝ) (car_value : ℝ) (multiplier : ℝ) (discount : ℝ) : ℝ :=
  (hotel_value + car_value) * multiplier * (1 - discount)

def gold_cost (yacht_value : ℝ) (multiplier : ℝ) (tax : ℝ) : ℝ :=
  yacht_value * multiplier * (1 + tax)

-- Theorem statement
theorem total_treats_value :
  hotel_cost hotel_per_night hotel_nights hotel_discount +
  car_cost car_value car_tax +
  house_cost car_value house_multiplier house_tax +
  yacht_cost (hotel_per_night * hotel_nights) car_value yacht_multiplier yacht_discount +
  gold_cost ((hotel_per_night * hotel_nights + car_value) * yacht_multiplier) gold_multiplier gold_tax
  = 339100 := by
  sorry

end NUMINAMATH_CALUDE_total_treats_value_l1672_167224


namespace NUMINAMATH_CALUDE_franks_daily_work_hours_l1672_167202

/-- Given that Frank worked a total of 8.0 hours over 4.0 days, with equal time worked each day,
    prove that he worked 2.0 hours per day. -/
theorem franks_daily_work_hours (total_hours : ℝ) (total_days : ℝ) (hours_per_day : ℝ)
    (h1 : total_hours = 8.0)
    (h2 : total_days = 4.0)
    (h3 : hours_per_day * total_days = total_hours) :
    hours_per_day = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_franks_daily_work_hours_l1672_167202


namespace NUMINAMATH_CALUDE_min_value_product_l1672_167248

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 9) :
  x^3 * y^3 * z^2 ≥ 1/27 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_l1672_167248


namespace NUMINAMATH_CALUDE_coordinates_sum_of_point_B_l1672_167223

/-- Given points A and B, where A is at (0, 0) and B is on the line y = 4,
    if the slope of segment AB is 2/3, then the sum of B's coordinates is 10. -/
theorem coordinates_sum_of_point_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 4)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  slope = 2/3 → x + 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_point_B_l1672_167223


namespace NUMINAMATH_CALUDE_distance_between_points_l1672_167268

/-- The distance between two points A and B given train travel conditions -/
theorem distance_between_points (v_pas v_freight : ℝ) (d : ℝ) : 
  (d / v_freight - d / v_pas = 3.2) →
  (v_pas * (d / v_freight) = d + 288) →
  (d / (v_freight + 10) - d / (v_pas + 10) = 2.4) →
  d = 360 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l1672_167268


namespace NUMINAMATH_CALUDE_triangle_side_difference_l1672_167297

theorem triangle_side_difference (a b : ℕ) (ha : a = 8) (hb : b = 13) : 
  (∃ (x_max x_min : ℕ), 
    (∀ x : ℕ, (x + a > b ∧ x + b > a ∧ a + b > x) → x_min ≤ x ∧ x ≤ x_max) ∧
    (x_max + a > b ∧ x_max + b > a ∧ a + b > x_max) ∧
    (x_min + a > b ∧ x_min + b > a ∧ a + b > x_min) ∧
    (∀ y : ℕ, y > x_max ∨ y < x_min → ¬(y + a > b ∧ y + b > a ∧ a + b > y)) ∧
    x_max - x_min = 14) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l1672_167297


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1672_167275

theorem absolute_value_inequality (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x + 3| < a) → a > 7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1672_167275


namespace NUMINAMATH_CALUDE_marble_probability_l1672_167279

/-- Represents a box of marbles -/
structure Box where
  gold : Nat
  black : Nat

/-- The probability of selecting a gold marble from a box -/
def prob_gold (b : Box) : Rat :=
  b.gold / (b.gold + b.black)

/-- The probability of selecting a black marble from a box -/
def prob_black (b : Box) : Rat :=
  b.black / (b.gold + b.black)

/-- The initial state of the boxes -/
def initial_boxes : List Box :=
  [⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 3⟩]

/-- The probability of the final outcome after the marble movements -/
def final_probability : Rat :=
  let box1 := initial_boxes[0]
  let box2 := initial_boxes[1]
  let box3 := initial_boxes[2]

  let prob_gold_to_box2 := prob_gold box1 * prob_gold (⟨box2.gold + 1, box2.black⟩) +
                           prob_black box1 * prob_gold box2
  
  let prob_black_to_box3 := 1 - prob_gold_to_box2

  prob_gold_to_box2 * prob_gold (⟨box3.gold + 1, box3.black⟩) +
  prob_black_to_box3 * prob_gold box3

theorem marble_probability :
  final_probability = 11 / 40 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l1672_167279


namespace NUMINAMATH_CALUDE_last_problem_number_l1672_167263

theorem last_problem_number (start : ℕ) (solved : ℕ) (last : ℕ) : 
  start = 78 → solved = 48 → last = start + solved - 1 → last = 125 := by
  sorry

end NUMINAMATH_CALUDE_last_problem_number_l1672_167263


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1672_167258

theorem polar_to_rectangular_conversion (r : ℝ) (θ : ℝ) :
  r = 6 ∧ θ = π / 3 →
  (r * Real.cos θ = 3 ∧ r * Real.sin θ = 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1672_167258


namespace NUMINAMATH_CALUDE_rectangle_x_value_l1672_167200

/-- Given a rectangle with vertices (x, 1), (1, 1), (1, -2), and (x, -2) and area 12, prove that x = -3 -/
theorem rectangle_x_value (x : ℝ) : 
  let vertices := [(x, 1), (1, 1), (1, -2), (x, -2)]
  let width := 1 - (-2)
  let area := 12
  let length := area / width
  x = 1 - length := by
  sorry

#check rectangle_x_value

end NUMINAMATH_CALUDE_rectangle_x_value_l1672_167200


namespace NUMINAMATH_CALUDE_square_side_length_l1672_167276

theorem square_side_length (A : ℝ) (h : A = 144) : 
  ∃ s : ℝ, s > 0 ∧ s * s = A ∧ s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1672_167276


namespace NUMINAMATH_CALUDE_birthday_check_value_l1672_167243

theorem birthday_check_value (initial_balance : ℝ) (check_value : ℝ) : 
  initial_balance = 150 →
  check_value = (1/4) * (initial_balance + check_value) →
  check_value = 50 := by
sorry

end NUMINAMATH_CALUDE_birthday_check_value_l1672_167243


namespace NUMINAMATH_CALUDE_simplify_expression_l1672_167255

theorem simplify_expression (a : ℝ) (h : a < (1/4 : ℝ)) :
  4 * (4*a - 1)^2 = Real.sqrt (1 - 4*a) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1672_167255


namespace NUMINAMATH_CALUDE_only_first_option_exact_l1672_167269

/-- Represents a measurement with a value and whether it's approximate or exact -/
structure Measurement where
  value : ℝ
  isApproximate : Bool

/-- The four options given in the problem -/
def options : List Measurement := [
  ⟨1752, false⟩,  -- A: Dictionary pages
  ⟨150, true⟩,    -- B: Water in teacup
  ⟨13.5, true⟩,   -- C: Running time
  ⟨6.2, true⟩     -- D: World population
]

/-- Theorem stating that only the first option is not an approximate number -/
theorem only_first_option_exact : 
  ∃! i : Fin 4, (options.get i).isApproximate = false := by
  sorry

end NUMINAMATH_CALUDE_only_first_option_exact_l1672_167269


namespace NUMINAMATH_CALUDE_trees_chopped_per_day_l1672_167203

/-- Represents the number of blocks of wood Ragnar gets per tree -/
def blocks_per_tree : ℕ := 3

/-- Represents the total number of blocks of wood Ragnar gets in 5 days -/
def total_blocks : ℕ := 30

/-- Represents the number of days Ragnar works -/
def days : ℕ := 5

/-- Theorem stating the number of trees Ragnar chops each day -/
theorem trees_chopped_per_day : 
  (total_blocks / days) / blocks_per_tree = 2 := by sorry

end NUMINAMATH_CALUDE_trees_chopped_per_day_l1672_167203


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l1672_167284

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The main theorem: if (a+i)/(1-i) is pure imaginary, then a = 1 -/
theorem complex_pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a + Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l1672_167284


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l1672_167261

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > -2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l1672_167261


namespace NUMINAMATH_CALUDE_rectangular_box_sum_l1672_167227

theorem rectangular_box_sum (A B C : ℝ) 
  (h1 : A * B = 30)
  (h2 : A * C = 50)
  (h3 : B * C = 90) :
  A + B + C = 58 * Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_sum_l1672_167227


namespace NUMINAMATH_CALUDE_line_slope_problem_l1672_167299

/-- Given m > 0 and points (m,1) and (2,√m) on a line with slope 2m, prove m = 4 -/
theorem line_slope_problem (m : ℝ) (h1 : m > 0) : 
  (2 * m = (Real.sqrt m - 1) / (2 - m)) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1672_167299


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l1672_167265

theorem units_digit_of_fraction : (30 * 31 * 32 * 33 * 34 * 35) / 7200 ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l1672_167265


namespace NUMINAMATH_CALUDE_sin_6phi_value_l1672_167262

theorem sin_6phi_value (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (6 * φ) = -396 * Real.sqrt 2 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_sin_6phi_value_l1672_167262


namespace NUMINAMATH_CALUDE_sally_bought_twenty_cards_l1672_167294

/-- The number of cards Sally bought -/
def cards_bought (initial : ℕ) (received : ℕ) (total : ℕ) : ℕ :=
  total - (initial + received)

/-- Theorem: Sally bought 20 cards -/
theorem sally_bought_twenty_cards :
  cards_bought 27 41 88 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_twenty_cards_l1672_167294


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1672_167240

theorem simplify_polynomial (x : ℝ) : 
  4 * x^3 + 5 * x + 6 * x^2 + 10 - (3 - 6 * x^2 - 4 * x^3 + 2 * x) = 
  8 * x^3 + 12 * x^2 + 3 * x + 7 := by sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1672_167240


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1672_167241

noncomputable def f (a b x : ℝ) := x^3 + 3*(a-1)*x^2 - 12*a*x + b

theorem cubic_function_properties (a b : ℝ) :
  let f := f a b
  ∃ (x₁ x₂ M N : ℝ),
    (∀ x, x ≠ x₁ → x ≠ x₂ → f x ≤ f x₁ ∨ f x ≥ f x₂) →
    (∃ m c, ∀ x, m*x - f x - c = 0 → x = 0 ∧ m = 24 ∧ c = 10) →
    (x₁ = 2 ∧ x₂ = 4 ∧ M = f x₁ ∧ N = f x₂ ∧ M = 10 ∧ N = 6) ∧
    (f 1 > f 2 → x₂ - x₁ = 4 → b = 10 →
      (∀ x, x ≤ -2 → f x ≤ f (-2)) ∧
      (∀ x, -2 ≤ x ∧ x ≤ 2 → f 2 ≤ f x) ∧
      (∀ x, 2 ≤ x → f x ≥ f 2) ∧
      M = 26 ∧ N = -6) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1672_167241


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1672_167273

/-- A linear function passing through the first, third, and fourth quadrants -/
def passes_through_134_quadrants (k : ℝ) : Prop :=
  (k - 3 > 0) ∧ (-k + 2 < 0)

/-- Theorem stating that if a linear function y=(k-3)x-k+2 passes through
    the first, third, and fourth quadrants, then k > 3 -/
theorem linear_function_quadrants (k : ℝ) :
  passes_through_134_quadrants k → k > 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1672_167273


namespace NUMINAMATH_CALUDE_triangle_ABC_area_l1672_167225

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (4, -1)

-- Function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_ABC_area :
  triangleArea A B C = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_area_l1672_167225


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1672_167281

theorem quadratic_inequality (x : ℝ) : x^2 - x - 12 < 0 ↔ -3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1672_167281


namespace NUMINAMATH_CALUDE_line_through_point_l1672_167229

/-- Given a line equation 2bx + (b+2)y = b + 6 that passes through the point (-3, 4), prove that b = 2/3 -/
theorem line_through_point (b : ℝ) : 
  (2 * b * (-3) + (b + 2) * 4 = b + 6) → b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1672_167229


namespace NUMINAMATH_CALUDE_corner_sum_is_200_l1672_167286

/-- Represents a 9x9 grid filled with numbers from 10 to 90 --/
def Grid := Fin 9 → Fin 9 → ℕ

/-- The grid is filled sequentially from 10 to 90 --/
def sequential_fill (g : Grid) : Prop :=
  ∀ i j, g i j = i.val * 9 + j.val + 10

/-- The sum of the numbers in the four corners of the grid --/
def corner_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 8 + g 8 0 + g 8 8

/-- Theorem stating that the sum of the numbers in the four corners is 200 --/
theorem corner_sum_is_200 (g : Grid) (h : sequential_fill g) : corner_sum g = 200 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_200_l1672_167286


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1672_167217

theorem trigonometric_equation_solution :
  ∃ (α β : ℝ),
    α ∈ Set.Ioo (-π/2) (π/2) ∧
    β ∈ Set.Ioo 0 π ∧
    Real.sin (3*π - α) = Real.sqrt 2 * Real.cos (π/2 - β) ∧
    Real.sqrt 3 * Real.cos (-α) = -Real.sqrt 2 * Real.cos (π + β) ∧
    α = π/4 ∧
    β = π/6 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1672_167217


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l1672_167283

/-- A sequence is geometric if the ratio of consecutive terms is constant -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_constant (a : ℕ → ℝ) (c : ℝ) :
  IsGeometric a → IsGeometric (fun n ↦ a n + c) → c = 0 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_constant_l1672_167283


namespace NUMINAMATH_CALUDE_gwen_spent_l1672_167256

theorem gwen_spent (received : ℕ) (left : ℕ) (spent : ℕ) : 
  received = 7 → left = 5 → spent = received - left → spent = 2 := by
  sorry

end NUMINAMATH_CALUDE_gwen_spent_l1672_167256


namespace NUMINAMATH_CALUDE_line_intersects_equidistant_points_in_first_and_second_quadrants_l1672_167298

/-- The line equation 4x + 6y = 24 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 6 * y = 24

/-- A point (x, y) is equidistant from coordinate axes if |x| = |y| -/
def equidistant (x y : ℝ) : Prop := abs x = abs y

/-- A point (x, y) is in the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in the second quadrant -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The line 4x + 6y = 24 intersects with y = x and y = -x only in the first and second quadrants -/
theorem line_intersects_equidistant_points_in_first_and_second_quadrants :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ ∧ equidistant x₁ y₁ ∧ first_quadrant x₁ y₁ ∧
    line_equation x₂ y₂ ∧ equidistant x₂ y₂ ∧ second_quadrant x₂ y₂ ∧
    (∀ (x y : ℝ), line_equation x y ∧ equidistant x y →
      first_quadrant x y ∨ second_quadrant x y) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_equidistant_points_in_first_and_second_quadrants_l1672_167298


namespace NUMINAMATH_CALUDE_complement_of_union_l1672_167201

open Set

universe u

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def A : Finset ℕ := {1,2,3}
def B : Finset ℕ := {3,4,5,6}

theorem complement_of_union :
  (U \ (A ∪ B) : Finset ℕ) = {7,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1672_167201


namespace NUMINAMATH_CALUDE_percentage_chain_l1672_167245

theorem percentage_chain (n : ℝ) : 
  (0.20 * 0.15 * 0.40 * 0.30 * 0.50 * n = 180) → n = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_chain_l1672_167245


namespace NUMINAMATH_CALUDE_max_squares_after_triangles_l1672_167277

/-- Represents the number of matchsticks used to form triangles efficiently -/
def triangleMatchsticks : ℕ := 13

/-- Represents the total number of matchsticks available -/
def totalMatchsticks : ℕ := 24

/-- Represents the number of matchsticks required to form a square -/
def matchsticksPerSquare : ℕ := 4

/-- Represents the number of triangles to be formed -/
def numTriangles : ℕ := 6

/-- Theorem stating the maximum number of squares that can be formed -/
theorem max_squares_after_triangles :
  (totalMatchsticks - triangleMatchsticks) / matchsticksPerSquare = 4 :=
sorry

end NUMINAMATH_CALUDE_max_squares_after_triangles_l1672_167277


namespace NUMINAMATH_CALUDE_sqrt_52_rational_l1672_167260

theorem sqrt_52_rational : 
  (((52 : ℝ).sqrt + 5) ^ (1/3 : ℝ)) - (((52 : ℝ).sqrt - 5) ^ (1/3 : ℝ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_52_rational_l1672_167260


namespace NUMINAMATH_CALUDE_monkeys_for_three_bananas_l1672_167253

/-- The number of monkeys needed to eat a given number of bananas in 8 minutes -/
def monkeys_needed (bananas : ℕ) : ℕ :=
  bananas

theorem monkeys_for_three_bananas :
  monkeys_needed 3 = 3 :=
by
  sorry

/-- Given condition: 8 monkeys take 8 minutes to eat 8 bananas -/
axiom eight_monkeys_eight_bananas : monkeys_needed 8 = 8

end NUMINAMATH_CALUDE_monkeys_for_three_bananas_l1672_167253


namespace NUMINAMATH_CALUDE_ratio_MBQ_ABQ_l1672_167218

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- State the conditions
axiom trisect_ABC : angle A B P = angle P B Q ∧ angle P B Q = angle Q B C
axiom trisect_PBQ : angle P B M = angle M B Q

-- State the theorem
theorem ratio_MBQ_ABQ :
  (angle M B Q) / (angle A B Q) = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_ratio_MBQ_ABQ_l1672_167218


namespace NUMINAMATH_CALUDE_cos_equality_l1672_167222

theorem cos_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → n = 138 → Real.cos (n * π / 180) = Real.cos (942 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_l1672_167222


namespace NUMINAMATH_CALUDE_ratio_problem_l1672_167214

theorem ratio_problem (x y : ℚ) (h : (8*x - 5*y) / (10*x - 3*y) = 4/7) : x/y = 23/16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1672_167214


namespace NUMINAMATH_CALUDE_triangle_inequality_l1672_167219

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D, E, F on the sides of the triangle
structure TriangleWithPoints extends Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)

-- Define the condition DC + CE = EA + AF = FB + BD
def satisfiesCondition (t : TriangleWithPoints) : Prop :=
  let distAB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let distBC := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let distCA := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let distDC := Real.sqrt ((t.D.1 - t.C.1)^2 + (t.D.2 - t.C.2)^2)
  let distCE := Real.sqrt ((t.C.1 - t.E.1)^2 + (t.C.2 - t.E.2)^2)
  let distEA := Real.sqrt ((t.E.1 - t.A.1)^2 + (t.E.2 - t.A.2)^2)
  let distAF := Real.sqrt ((t.A.1 - t.F.1)^2 + (t.A.2 - t.F.2)^2)
  let distFB := Real.sqrt ((t.F.1 - t.B.1)^2 + (t.F.2 - t.B.2)^2)
  let distBD := Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2)
  distDC + distCE = distEA + distAF ∧ distEA + distAF = distFB + distBD

-- State the theorem
theorem triangle_inequality (t : TriangleWithPoints) (h : satisfiesCondition t) :
  let distDE := Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2)
  let distEF := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let distFD := Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2)
  let distAB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let distBC := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let distCA := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  distDE + distEF + distFD ≥ (1/2) * (distAB + distBC + distCA) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1672_167219


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1672_167292

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 24 ∧ x - y = 8 → x * y = 128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1672_167292


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l1672_167221

theorem mixed_number_calculation : 
  36 * ((5 + 1/6) - (6 + 1/7)) / ((3 + 1/6) + (2 + 1/7)) = -(6 + 156/223) :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l1672_167221


namespace NUMINAMATH_CALUDE_initial_ratio_proof_l1672_167230

/-- Proves that given a 30-liter mixture of liquids p and q, if adding 12 liters of liquid q
    results in a 3:4 ratio of p to q, then the initial ratio of p to q was 3:2. -/
theorem initial_ratio_proof (p q : ℝ) 
  (h1 : p + q = 30)  -- Initial mixture is 30 liters
  (h2 : p / (q + 12) = 3 / 4)  -- After adding 12 liters of q, the ratio becomes 3:4
  : p / q = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_proof_l1672_167230


namespace NUMINAMATH_CALUDE_yard_area_l1672_167236

/-- Calculates the area of a rectangular yard given the length of one side and the total length of the other three sides. -/
theorem yard_area (side_length : ℝ) (other_sides : ℝ) : 
  side_length = 40 → other_sides = 50 → side_length * ((other_sides - side_length) / 2) = 200 :=
by
  sorry

#check yard_area

end NUMINAMATH_CALUDE_yard_area_l1672_167236


namespace NUMINAMATH_CALUDE_smartphone_price_proof_l1672_167228

def laptop_price : ℕ := 600
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def total_paid : ℕ := 3000
def change_received : ℕ := 200

theorem smartphone_price_proof :
  ∃ (smartphone_price : ℕ),
    smartphone_price * num_smartphones + laptop_price * num_laptops = total_paid - change_received ∧
    smartphone_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_proof_l1672_167228


namespace NUMINAMATH_CALUDE_carrie_tomatoes_l1672_167282

/-- The number of tomatoes Carrie harvested -/
def tomatoes : ℕ := sorry

/-- The number of carrots Carrie harvested -/
def carrots : ℕ := 350

/-- The price of a tomato in dollars -/
def tomato_price : ℚ := 1

/-- The price of a carrot in dollars -/
def carrot_price : ℚ := 3/2

/-- The total revenue from selling all tomatoes and carrots in dollars -/
def total_revenue : ℚ := 725

theorem carrie_tomatoes : 
  tomatoes = 200 :=
sorry

end NUMINAMATH_CALUDE_carrie_tomatoes_l1672_167282


namespace NUMINAMATH_CALUDE_ttakji_count_l1672_167287

theorem ttakji_count (n : ℕ) (h : n^2 + 36 = (n + 1)^2 + 3) : n^2 + 36 = 292 := by
  sorry

end NUMINAMATH_CALUDE_ttakji_count_l1672_167287


namespace NUMINAMATH_CALUDE_problem_statement_l1672_167252

theorem problem_statement (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → 
  (40/100 : ℝ) * N = 204 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1672_167252


namespace NUMINAMATH_CALUDE_min_dot_product_hyperbola_l1672_167251

/-- The minimum dot product of two vectors from the origin to points on the right branch of x² - y² = 1 is 1 -/
theorem min_dot_product_hyperbola (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁ > 0 → x₂ > 0 → x₁^2 - y₁^2 = 1 → x₂^2 - y₂^2 = 1 → x₁*x₂ + y₁*y₂ ≥ 1 := by
  sorry

#check min_dot_product_hyperbola

end NUMINAMATH_CALUDE_min_dot_product_hyperbola_l1672_167251


namespace NUMINAMATH_CALUDE_twin_pairs_probability_l1672_167249

/-- Represents the gender composition of a pair of twins -/
inductive TwinPair
  | BothBoys
  | BothGirls
  | Mixed

/-- The probability of each outcome for a pair of twins -/
def pairProbability : TwinPair → ℚ
  | TwinPair.BothBoys => 1/3
  | TwinPair.BothGirls => 1/3
  | TwinPair.Mixed => 1/3

/-- The probability of two pairs of twins having a specific composition -/
def twoTwinPairsProbability (pair1 pair2 : TwinPair) : ℚ :=
  pairProbability pair1 * pairProbability pair2

theorem twin_pairs_probability :
  (twoTwinPairsProbability TwinPair.BothBoys TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.BothGirls TwinPair.BothGirls) =
  (twoTwinPairsProbability TwinPair.Mixed TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.Mixed TwinPair.BothGirls) ∧
  (twoTwinPairsProbability TwinPair.BothBoys TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.BothGirls TwinPair.BothGirls) = 2/9 :=
by sorry

#check twin_pairs_probability

end NUMINAMATH_CALUDE_twin_pairs_probability_l1672_167249


namespace NUMINAMATH_CALUDE_second_smallest_odd_is_three_l1672_167296

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

def second_smallest_odd : ℕ → Prop
| 3 => ∃ (x : ℕ), (is_odd x ∧ in_range x ∧ x < 3) ∧
                  ∀ (y : ℕ), (is_odd y ∧ in_range y ∧ y ≠ x ∧ y ≠ 3) → y > 3
| _ => False

theorem second_smallest_odd_is_three : second_smallest_odd 3 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_odd_is_three_l1672_167296


namespace NUMINAMATH_CALUDE_fraction_simplification_l1672_167235

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 - 32*x^4 + 256) / (x^4 - 8) = 65 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1672_167235


namespace NUMINAMATH_CALUDE_perfect_square_property_l1672_167247

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem perfect_square_property : 
  (is_perfect_square (factorial 101 * 102 * 102)) ∧ 
  (¬ is_perfect_square (factorial 102 * 103 * 103)) ∧
  (¬ is_perfect_square (factorial 103 * 104 * 104)) ∧
  (¬ is_perfect_square (factorial 104 * 105 * 105)) ∧
  (¬ is_perfect_square (factorial 105 * 106 * 106)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_property_l1672_167247


namespace NUMINAMATH_CALUDE_kiki_scarf_problem_l1672_167211

/-- Kiki's scarf and hat buying problem -/
theorem kiki_scarf_problem (total_money : ℝ) (scarf_price : ℝ) :
  total_money = 90 →
  scarf_price = 2 →
  ∃ (num_scarves num_hats : ℕ) (hat_price : ℝ),
    num_hats = 2 * num_scarves ∧
    hat_price * num_hats = 0.6 * total_money ∧
    scarf_price * num_scarves = 0.4 * total_money ∧
    num_scarves = 18 := by
  sorry


end NUMINAMATH_CALUDE_kiki_scarf_problem_l1672_167211


namespace NUMINAMATH_CALUDE_sample_customers_l1672_167278

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (samples_left : ℕ) : 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  (samples_per_box * boxes_opened - samples_left) = 235 :=
by
  sorry

end NUMINAMATH_CALUDE_sample_customers_l1672_167278


namespace NUMINAMATH_CALUDE_equal_pair_proof_l1672_167234

theorem equal_pair_proof : 
  ((-3 : ℤ)^2 = Int.sqrt 81) ∧ 
  (|(-3 : ℤ)| ≠ -3) ∧ 
  (-|(-4 : ℤ)| ≠ (-2 : ℤ)^2) ∧ 
  (Int.sqrt ((-4 : ℤ)^2) ≠ -4) :=
by sorry

end NUMINAMATH_CALUDE_equal_pair_proof_l1672_167234


namespace NUMINAMATH_CALUDE_alex_grocery_charge_percentage_l1672_167244

/-- The problem of determining Alex's grocery delivery charge percentage --/
theorem alex_grocery_charge_percentage :
  ∀ (car_cost savings_initial trip_charge trips_made grocery_total charge_percentage : ℚ),
  car_cost = 14600 →
  savings_initial = 14500 →
  trip_charge = (3/2) →
  trips_made = 40 →
  grocery_total = 800 →
  car_cost - savings_initial = trip_charge * trips_made + charge_percentage * grocery_total →
  charge_percentage = (1/20) := by
  sorry

end NUMINAMATH_CALUDE_alex_grocery_charge_percentage_l1672_167244


namespace NUMINAMATH_CALUDE_last_draw_same_color_prob_l1672_167242

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 2

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of people drawing marbles -/
def num_people : ℕ := 3

/-- Represents the number of marbles each person draws -/
def marbles_per_draw : ℕ := 2

/-- Calculates the probability of the last person drawing two marbles of the same color -/
def prob_last_draw_same_color : ℚ :=
  (num_colors * (Nat.choose (total_marbles - 2 * marbles_per_draw) marbles_per_draw)) /
  (Nat.choose total_marbles marbles_per_draw * 
   Nat.choose (total_marbles - marbles_per_draw) marbles_per_draw)

theorem last_draw_same_color_prob :
  prob_last_draw_same_color = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_last_draw_same_color_prob_l1672_167242


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l1672_167207

/-- The volume of a triangular prism with given dimensions -/
theorem triangular_prism_volume 
  (thickness : ℝ) 
  (side1 side2 side3 : ℝ) 
  (h_thickness : thickness = 2)
  (h_side1 : side1 = 7)
  (h_side2 : side2 = 24)
  (h_side3 : side3 = 25)
  (h_right_triangle : side1^2 + side2^2 = side3^2) :
  thickness * (1/2 * side1 * side2) = 168 := by
sorry

end NUMINAMATH_CALUDE_triangular_prism_volume_l1672_167207


namespace NUMINAMATH_CALUDE_not_always_parallel_if_perpendicular_to_same_plane_l1672_167257

-- Define a type for planes
axiom Plane : Type

-- Define a relation for perpendicularity between planes
axiom perpendicular : Plane → Plane → Prop

-- Define a relation for parallelism between planes
axiom parallel : Plane → Plane → Prop

-- State the theorem
theorem not_always_parallel_if_perpendicular_to_same_plane :
  ¬ (∀ (P Q R : Plane), perpendicular P R → perpendicular Q R → parallel P Q) :=
sorry

end NUMINAMATH_CALUDE_not_always_parallel_if_perpendicular_to_same_plane_l1672_167257


namespace NUMINAMATH_CALUDE_orange_pyramid_sum_l1672_167290

/-- Calculates the number of oranges in a single layer of the pyramid -/
def oranges_in_layer (n : ℕ) : ℕ := n * n / 2

/-- Calculates the total number of oranges in the pyramid stack -/
def total_oranges (base_size : ℕ) : ℕ :=
  (List.range base_size).map oranges_in_layer |>.sum

/-- The theorem stating that a pyramid with base size 6 contains 44 oranges -/
theorem orange_pyramid_sum : total_oranges 6 = 44 := by
  sorry

#eval total_oranges 6

end NUMINAMATH_CALUDE_orange_pyramid_sum_l1672_167290


namespace NUMINAMATH_CALUDE_john_weekly_earnings_l1672_167266

/-- Calculates John's weekly earnings from streaming -/
def weekly_earnings (days_off : ℕ) (hours_per_stream : ℕ) (hourly_rate : ℕ) : ℕ :=
  let days_streaming := 7 - days_off
  let total_hours := days_streaming * hours_per_stream
  total_hours * hourly_rate

/-- Theorem: John's weekly earnings from streaming are $160 -/
theorem john_weekly_earnings :
  weekly_earnings 3 4 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_earnings_l1672_167266


namespace NUMINAMATH_CALUDE_inverse_f_at_46_l1672_167233

def f (x : ℝ) : ℝ := 5 * x^3 + 6

theorem inverse_f_at_46 : f⁻¹ 46 = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_46_l1672_167233


namespace NUMINAMATH_CALUDE_max_prime_factors_b_l1672_167295

theorem max_prime_factors_b (a b : ℕ+) 
  (h_gcd : (Nat.gcd a b).factors.length = 5)
  (h_lcm : (Nat.lcm a b).factors.length = 20)
  (h_fewer : (b.val.factors.length : ℕ) < a.val.factors.length) :
  b.val.factors.length ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_prime_factors_b_l1672_167295


namespace NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l1672_167293

theorem min_abs_z_on_line_segment (z : ℂ) (h : Complex.abs (z - 2*Complex.I) + Complex.abs (z - 5) = 7) :
  ∃ (min_abs_z : ℝ), min_abs_z = Real.sqrt (100 / 29) ∧
  ∀ (w : ℂ), Complex.abs (w - 2*Complex.I) + Complex.abs (w - 5) = 7 →
  Complex.abs w ≥ min_abs_z :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_on_line_segment_l1672_167293


namespace NUMINAMATH_CALUDE_johns_donation_l1672_167209

theorem johns_donation (n : ℕ) (new_avg : ℚ) (increase_percent : ℚ) :
  n = 1 →
  new_avg = 75 →
  increase_percent = 50 / 100 →
  let old_avg := new_avg / (1 + increase_percent)
  let total_before := old_avg * n
  let total_after := new_avg * (n + 1)
  total_after - total_before = 100 := by
sorry

end NUMINAMATH_CALUDE_johns_donation_l1672_167209


namespace NUMINAMATH_CALUDE_intersection_perpendicular_points_l1672_167237

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def l (x y m : ℝ) : Prop := y = 2 * x + m

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem intersection_perpendicular_points (m : ℝ) : 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    l x₁ y₁ m ∧ l x₂ y₂ m ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    perpendicular x₁ y₁ x₂ y₂ ↔ 
    m = 2 ∨ m = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_points_l1672_167237


namespace NUMINAMATH_CALUDE_solve_equation_l1672_167289

theorem solve_equation (x : ℝ) : (2 * x + 7) / 6 = 13 → x = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1672_167289


namespace NUMINAMATH_CALUDE_sqrt_900_squared_times_6_l1672_167206

theorem sqrt_900_squared_times_6 : (Real.sqrt 900)^2 * 6 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_900_squared_times_6_l1672_167206


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1672_167291

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1672_167291


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1672_167272

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 2*x - 5 = 0) ↔ ((x + 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1672_167272


namespace NUMINAMATH_CALUDE_value_in_scientific_notation_l1672_167280

/-- Represents 1 billion -/
def billion : ℝ := 10^9

/-- The value we want to express in scientific notation -/
def value : ℝ := 45 * billion

/-- The scientific notation representation of the value -/
def scientific_notation : ℝ := 4.5 * 10^9

theorem value_in_scientific_notation : value = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_value_in_scientific_notation_l1672_167280


namespace NUMINAMATH_CALUDE_triangle_medians_inequality_l1672_167213

/-- Given a triangle with sides a, b, c, medians ta, tb, tc, and circumcircle diameter D,
    the sum of the ratios of the squared sides to their opposite medians
    is less than or equal to 6 times the diameter of the circumcircle. -/
theorem triangle_medians_inequality (a b c ta tb tc D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ta : 0 < ta) (h_pos_tb : 0 < tb) (h_pos_tc : 0 < tc)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_medians : ta^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧ 
               tb^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧ 
               tc^2 = (2*a^2 + 2*b^2 - c^2) / 4)
  (h_circumcircle : D = (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (h_s : s = (a + b + c) / 2) :
  (a^2 + b^2) / tc + (b^2 + c^2) / ta + (c^2 + a^2) / tb ≤ 6 * D :=
sorry

end NUMINAMATH_CALUDE_triangle_medians_inequality_l1672_167213


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1672_167232

theorem modular_arithmetic_problem :
  (3 * (7⁻¹ : ZMod 120) + 9 * (13⁻¹ : ZMod 120) + 4 * (17⁻¹ : ZMod 120)) = (86 : ZMod 120) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1672_167232


namespace NUMINAMATH_CALUDE_train_speed_l1672_167267

/-- The speed of a train crossing a tunnel -/
theorem train_speed (train_length tunnel_length : ℝ) (crossing_time : ℝ) :
  train_length = 800 →
  tunnel_length = 500 →
  crossing_time = 1 / 60 →
  (train_length + tunnel_length) / crossing_time = 78000 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1672_167267


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1672_167238

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 24) (hxz : x * z = 48) (hyz : y * z = 72) :
  x + y + z = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1672_167238


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l1672_167220

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factor_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, p < 20 → is_prime p → ¬(n % p = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, is_nonprime n ∧ 
           has_no_prime_factor_less_than_20 n ∧
           (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_prime_factor_less_than_20 m)) ∧
           n = 529 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l1672_167220


namespace NUMINAMATH_CALUDE_one_fifth_of_five_times_seven_l1672_167270

theorem one_fifth_of_five_times_seven :
  (1 / 5 : ℚ) * (5 * 7) = 7 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_five_times_seven_l1672_167270
