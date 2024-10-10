import Mathlib

namespace triangle_problem_l983_98361

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (Real.sqrt 3 * Real.sin C - 2 * Real.cos A) * Real.sin B = (2 * Real.sin A - Real.sin C) * Real.cos B →
  a^2 + c^2 = 4 + Real.sqrt 3 →
  (1/2) * a * c * Real.sin B = (3 + Real.sqrt 3) / 4 →
  B = π / 3 ∧ a + b + c = (Real.sqrt 6 + 2 * Real.sqrt 3 + 3 * Real.sqrt 2) / 2 := by
  sorry

end triangle_problem_l983_98361


namespace base_subtraction_equality_l983_98305

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement --/
theorem base_subtraction_equality : 
  let base_9_num := to_base_10 [5, 2, 3] 9
  let base_6_num := to_base_10 [5, 4, 2] 6
  base_9_num - base_6_num = 165 := by
  sorry

end base_subtraction_equality_l983_98305


namespace hydroton_rainfall_l983_98390

/-- The total rainfall in Hydroton from 2019 to 2021 -/
def total_rainfall (r2019 r2020 r2021 : ℝ) : ℝ :=
  12 * (r2019 + r2020 + r2021)

/-- Theorem: The total rainfall in Hydroton from 2019 to 2021 is 1884 mm -/
theorem hydroton_rainfall : 
  let r2019 : ℝ := 50
  let r2020 : ℝ := r2019 + 5
  let r2021 : ℝ := r2020 - 3
  total_rainfall r2019 r2020 r2021 = 1884 :=
by
  sorry


end hydroton_rainfall_l983_98390


namespace three_planes_division_l983_98347

/-- A type representing the possible configurations of three non-coincident planes in space -/
inductive PlaneConfiguration
  | AllParallel
  | TwoParallelOneIntersecting
  | IntersectAlongLine
  | IntersectPairwiseParallelLines
  | IntersectAtPoint

/-- The number of parts that space is divided into by three non-coincident planes -/
def numParts (config : PlaneConfiguration) : ℕ :=
  match config with
  | .AllParallel => 4
  | .TwoParallelOneIntersecting => 6
  | .IntersectAlongLine => 6
  | .IntersectPairwiseParallelLines => 7
  | .IntersectAtPoint => 8

/-- Theorem stating that the number of parts is always 4, 6, 7, or 8 -/
theorem three_planes_division (config : PlaneConfiguration) :
  ∃ n : ℕ, (n = 4 ∨ n = 6 ∨ n = 7 ∨ n = 8) ∧ numParts config = n :=
sorry

end three_planes_division_l983_98347


namespace reciprocal_squares_sum_l983_98313

theorem reciprocal_squares_sum (a b : ℕ) (h : a * b = 3) :
  (1 : ℚ) / a^2 + 1 / b^2 = 10 / 9 := by
  sorry

end reciprocal_squares_sum_l983_98313


namespace range_of_a_l983_98328

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

theorem range_of_a (a : ℝ) : (A ∪ B a = A) → a ≥ 1 := by
  sorry

end range_of_a_l983_98328


namespace dice_remainder_prob_l983_98375

/-- The probability of getting a specific remainder when the sum of two dice is divided by 4 -/
def remainder_probability (r : Fin 4) : ℚ := sorry

/-- The sum of all probabilities should be 1 -/
axiom prob_sum_one : remainder_probability 0 + remainder_probability 1 + remainder_probability 2 + remainder_probability 3 = 1

/-- The probabilities are non-negative -/
axiom prob_non_negative (r : Fin 4) : remainder_probability r ≥ 0

theorem dice_remainder_prob :
  2 * remainder_probability 3 - 3 * remainder_probability 2 + remainder_probability 1 - remainder_probability 0 = -2/9 := by
  sorry

end dice_remainder_prob_l983_98375


namespace hyperbola_asymptotes_l983_98395

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0,
    and the length of its real axis is 1,
    prove that the equation of its asymptotes is y = ±2x -/
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) (h2 : 2 * a = 1) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = 2 * x ∨ f x = -2 * x) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x y, x^2/a^2 - y^2 = 1 → x > δ → |y - f x| < ε) :=
sorry

end hyperbola_asymptotes_l983_98395


namespace cubic_root_sum_l983_98314

theorem cubic_root_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
   x^3 - 8*x^2 + a*x - b = 0 ∧
   y^3 - 8*y^2 + a*y - b = 0 ∧
   z^3 - 8*z^2 + a*z - b = 0) →
  a + b = 27 ∨ a + b = 31 := by
sorry

end cubic_root_sum_l983_98314


namespace four_mat_weaves_four_days_l983_98374

-- Define the rate of weaving (mats per mat-weave per day)
def weaving_rate (mats : ℕ) (mat_weaves : ℕ) (days : ℕ) : ℚ :=
  (mats : ℚ) / ((mat_weaves : ℚ) * (days : ℚ))

theorem four_mat_weaves_four_days (mats : ℕ) :
  -- Condition: 8 mat-weaves weave 16 mats in 8 days
  weaving_rate 16 8 8 = weaving_rate mats 4 4 →
  -- Conclusion: 4 mat-weaves weave 4 mats in 4 days
  mats = 4 := by
  sorry

end four_mat_weaves_four_days_l983_98374


namespace basic_computer_price_l983_98366

theorem basic_computer_price
  (total_price : ℝ)
  (enhanced_price_difference : ℝ)
  (printer_ratio : ℝ)
  (h1 : total_price = 2500)
  (h2 : enhanced_price_difference = 500)
  (h3 : printer_ratio = 1/3)
  : ∃ (basic_price printer_price : ℝ),
    basic_price + printer_price = total_price ∧
    printer_price = printer_ratio * (basic_price + enhanced_price_difference + printer_price) ∧
    basic_price = 1500 :=
by
  sorry

end basic_computer_price_l983_98366


namespace f_derivative_and_value_l983_98362

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 4

theorem f_derivative_and_value :
  (∀ x, deriv f x = -Real.sin (4 * x)) ∧
  (deriv f (π / 6) = -Real.sqrt 3 / 2) := by
  sorry

end f_derivative_and_value_l983_98362


namespace money_left_over_l983_98388

/-- Calculates the money left over after buying a bike given work parameters and bike cost -/
theorem money_left_over 
  (hourly_rate : ℝ) 
  (weekly_hours : ℝ) 
  (weeks_worked : ℝ) 
  (bike_cost : ℝ) 
  (h1 : hourly_rate = 8)
  (h2 : weekly_hours = 35)
  (h3 : weeks_worked = 4)
  (h4 : bike_cost = 400) :
  hourly_rate * weekly_hours * weeks_worked - bike_cost = 720 := by
  sorry


end money_left_over_l983_98388


namespace summer_grain_scientific_notation_l983_98368

def summer_grain_production : ℝ := 11534000000

/-- Converts a number to scientific notation with a specified number of significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem summer_grain_scientific_notation :
  to_scientific_notation summer_grain_production 4 = (1.153, 8) :=
sorry

end summer_grain_scientific_notation_l983_98368


namespace rice_purchase_amount_l983_98336

/-- The price of rice in cents per pound -/
def rice_price : ℚ := 75

/-- The price of beans in cents per pound -/
def bean_price : ℚ := 35

/-- The total weight of rice and beans in pounds -/
def total_weight : ℚ := 30

/-- The total cost in cents -/
def total_cost : ℚ := 1650

/-- The amount of rice purchased in pounds -/
def rice_amount : ℚ := 15

theorem rice_purchase_amount :
  ∃ (bean_amount : ℚ),
    rice_amount + bean_amount = total_weight ∧
    rice_price * rice_amount + bean_price * bean_amount = total_cost :=
sorry

end rice_purchase_amount_l983_98336


namespace staircase_steps_l983_98363

/-- Represents the number of toothpicks used in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

/-- The number of toothpicks used in a 3-step staircase -/
def three_step_toothpicks : ℕ := 27

/-- The target number of toothpicks -/
def target_toothpicks : ℕ := 270

theorem staircase_steps :
  ∃ (n : ℕ), toothpicks n = target_toothpicks ∧ n = 12 :=
sorry

end staircase_steps_l983_98363


namespace quadratic_inequality_range_l983_98360

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, x^2 + (a - 4)*x + 4 > 0) ↔ a > 0 := by
  sorry

end quadratic_inequality_range_l983_98360


namespace monthly_subscription_more_cost_effective_l983_98300

/-- Represents the cost of internet access plans -/
def internet_cost (pay_per_minute_rate : ℚ) (monthly_fee : ℚ) (communication_fee : ℚ) (hours : ℚ) : ℚ × ℚ :=
  let minutes : ℚ := hours * 60
  let pay_per_minute_cost : ℚ := (pay_per_minute_rate + communication_fee) * minutes
  let monthly_subscription_cost : ℚ := monthly_fee + communication_fee * minutes
  (pay_per_minute_cost, monthly_subscription_cost)

theorem monthly_subscription_more_cost_effective :
  let (pay_per_minute_cost, monthly_subscription_cost) :=
    internet_cost (5 / 100) 50 (2 / 100) 20
  monthly_subscription_cost < pay_per_minute_cost :=
by sorry

end monthly_subscription_more_cost_effective_l983_98300


namespace factorization_ax_squared_minus_a_l983_98310

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end factorization_ax_squared_minus_a_l983_98310


namespace not_three_equal_root_equation_three_equal_root_with_negative_one_root_three_equal_root_on_line_l983_98309

/-- A quadratic equation is a three equal root equation if one root is 1/3 of the other --/
def is_three_equal_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ = (1/3) * x₂

/-- The first part of the problem --/
theorem not_three_equal_root_equation : ¬ is_three_equal_root_equation 1 (-8) 11 := by
  sorry

/-- The second part of the problem --/
theorem three_equal_root_with_negative_one_root (b c : ℤ) :
  is_three_equal_root_equation 1 b c ∧ (∃ x : ℝ, x^2 + b*x + c = 0 ∧ x = -1) → b = 4 ∧ c = 3 := by
  sorry

/-- The third part of the problem --/
theorem three_equal_root_on_line (m n : ℝ) :
  n = 2*m + 1 ∧ is_three_equal_root_equation m n 2 → m = 3/2 ∨ m = 1/6 := by
  sorry

end not_three_equal_root_equation_three_equal_root_with_negative_one_root_three_equal_root_on_line_l983_98309


namespace sum_of_cubes_equation_l983_98352

theorem sum_of_cubes_equation (x y : ℝ) (h : x^3 + 21*x*y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 := by sorry

end sum_of_cubes_equation_l983_98352


namespace cube_division_l983_98373

theorem cube_division (original_size : ℝ) (num_divisions : ℕ) (num_painted : ℕ) :
  original_size = 3 →
  num_divisions ^ 3 = 27 →
  num_painted = 26 →
  ∃ (smaller_size : ℝ),
    smaller_size = 1 ∧
    num_divisions * smaller_size = original_size :=
by sorry

end cube_division_l983_98373


namespace profit_per_meter_l983_98349

/-- The profit per meter of cloth given the selling price, quantity sold, and cost price per meter -/
theorem profit_per_meter
  (selling_price : ℕ)
  (quantity : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : selling_price = 4950)
  (h2 : quantity = 75)
  (h3 : cost_price_per_meter = 51) :
  (selling_price - quantity * cost_price_per_meter) / quantity = 15 :=
by sorry

end profit_per_meter_l983_98349


namespace valid_routes_l983_98399

/-- Represents the lengths of route segments between consecutive cities --/
structure RouteLengths where
  ab : ℕ
  bc : ℕ
  cd : ℕ
  de : ℕ
  ef : ℕ

/-- Checks if the given route lengths satisfy all conditions --/
def isValidRoute (r : RouteLengths) : Prop :=
  r.ab > r.bc ∧ r.bc > r.cd ∧ r.cd > r.de ∧ r.de > r.ef ∧
  r.ab = 2 * r.ef ∧
  r.ab + r.bc + r.cd + r.de + r.ef = 53

/-- The theorem stating that only three specific combinations of route lengths are valid --/
theorem valid_routes :
  ∀ r : RouteLengths, isValidRoute r →
    (r = ⟨14, 12, 11, 9, 7⟩ ∨ r = ⟨14, 13, 11, 8, 7⟩ ∨ r = ⟨14, 13, 10, 9, 7⟩) :=
by sorry

end valid_routes_l983_98399


namespace heath_planting_time_l983_98311

/-- The number of hours Heath spent planting carrots -/
def planting_time (rows : ℕ) (plants_per_row : ℕ) (plants_per_hour : ℕ) : ℕ :=
  (rows * plants_per_row) / plants_per_hour

/-- Theorem stating that Heath spent 20 hours planting carrots -/
theorem heath_planting_time :
  planting_time 400 300 6000 = 20 := by
  sorry

end heath_planting_time_l983_98311


namespace largest_sum_is_three_fourths_l983_98378

theorem largest_sum_is_three_fourths : 
  let sums : List ℚ := [1/4 + 1/2, 1/4 + 1/3, 1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/11]
  (∀ x ∈ sums, x ≤ 1/4 + 1/2) ∧ (1/4 + 1/2 = 3/4) := by
  sorry

end largest_sum_is_three_fourths_l983_98378


namespace geometric_sequence_ratio_l983_98339

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (q > 0) →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (a 3 * a 7 = 4 * (a 4)^2) →
  q = 2 := by
sorry

end geometric_sequence_ratio_l983_98339


namespace diamond_six_three_l983_98367

/-- Diamond operation defined as a ◇ b = 4a + 2b -/
def diamond (a b : ℝ) : ℝ := 4 * a + 2 * b

/-- Theorem stating that 6 ◇ 3 = 30 -/
theorem diamond_six_three : diamond 6 3 = 30 := by
  sorry

end diamond_six_three_l983_98367


namespace probability_arithmetic_progression_l983_98340

def dice_sides := 4

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  (b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ b = c + 1)

def favorable_outcomes : ℕ := 12

def total_outcomes : ℕ := dice_sides ^ 3

theorem probability_arithmetic_progression :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 16 := by
  sorry

end probability_arithmetic_progression_l983_98340


namespace inequality_solution_set_l983_98391

def solution_set : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}

theorem inequality_solution_set :
  ∀ x : ℝ, (x + 2 ≠ 0) → ((x - 3) / (x + 2) ≤ 0 ↔ x ∈ solution_set) :=
by sorry

end inequality_solution_set_l983_98391


namespace angle_measure_possibilities_l983_98381

theorem angle_measure_possibilities :
  ∃! X : ℕ+, 
    ∃ Y : ℕ+, 
      (X : ℝ) + Y = 180 ∧ 
      (X : ℝ) = 3 * Y := by
  sorry

end angle_measure_possibilities_l983_98381


namespace books_left_l983_98398

theorem books_left (initial_books sold_books : ℝ) 
  (h1 : initial_books = 51.5)
  (h2 : sold_books = 45.75) : 
  initial_books - sold_books = 5.75 := by
  sorry

end books_left_l983_98398


namespace quadratic_equation_solution_l983_98372

theorem quadratic_equation_solution :
  let f (x : ℂ) := 2 * (5 * x^2 + 4 * x + 3) - 6
  let g (x : ℂ) := -3 * (2 - 4 * x)
  ∀ x : ℂ, f x = g x ↔ x = (1 + Complex.I * Real.sqrt 14) / 5 ∨ x = (1 - Complex.I * Real.sqrt 14) / 5 := by
sorry

end quadratic_equation_solution_l983_98372


namespace quadratic_properties_l983_98384

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 4*x + 3

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem quadratic_properties :
  (∀ x ∈ domain, f (-x + 4) = f x) ∧  -- Axis of symmetry at x = 2
  (f 2 = 7) ∧  -- Vertex at (2, 7)
  (∀ x ∈ domain, f x ≤ 7) ∧  -- Maximum value
  (∀ x ∈ domain, f x ≥ 6) ∧  -- Minimum value
  (∃ x ∈ domain, f x = 7) ∧  -- Maximum is attained
  (∃ x ∈ domain, f x = 6) :=  -- Minimum is attained
by sorry

end quadratic_properties_l983_98384


namespace train_speed_proof_l983_98317

/-- Proves that the new train speed is 256 km/h given the problem conditions -/
theorem train_speed_proof (distance : ℝ) (speed_multiplier : ℝ) (time_reduction : ℝ) 
  (h1 : distance = 1280)
  (h2 : speed_multiplier = 3.2)
  (h3 : time_reduction = 11)
  (h4 : ∀ x : ℝ, distance / x - distance / (speed_multiplier * x) = time_reduction) :
  speed_multiplier * (distance / (distance / speed_multiplier + time_reduction)) = 256 :=
by sorry

end train_speed_proof_l983_98317


namespace speed_ratio_is_seven_to_eight_l983_98383

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := 400

-- Define the time intervals
def time1 : ℝ := 3
def time2 : ℝ := 12

-- Theorem statement
theorem speed_ratio_is_seven_to_eight :
  -- Condition 1: After 3 minutes, A and B are equidistant from O
  (v_A * time1 = |initial_B_position - v_B * time1|) →
  -- Condition 2: After 12 minutes, A and B are again equidistant from O
  (v_A * time2 = |initial_B_position - v_B * time2|) →
  -- Conclusion: The ratio of A's speed to B's speed is 7:8
  (v_A / v_B = 7 / 8) := by
sorry

end speed_ratio_is_seven_to_eight_l983_98383


namespace instantaneous_rate_of_change_at_zero_l983_98393

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_zero :
  deriv f 0 = 2 * Real.exp 0 := by
  sorry

end instantaneous_rate_of_change_at_zero_l983_98393


namespace davids_age_l983_98316

/-- Given that Yuan is 14 years old and twice David's age, prove that David is 7 years old. -/
theorem davids_age (yuan_age : ℕ) (david_age : ℕ) 
  (h1 : yuan_age = 14) 
  (h2 : yuan_age = 2 * david_age) : 
  david_age = 7 := by
  sorry

end davids_age_l983_98316


namespace taylor_painting_time_l983_98364

/-- The time it takes for Taylor to paint the room alone -/
def taylor_time : ℝ := 12

/-- The time it takes for Jennifer to paint the room alone -/
def jennifer_time : ℝ := 10

/-- The time it takes for Taylor and Jennifer to paint the room together -/
def combined_time : ℝ := 5.45454545455

theorem taylor_painting_time : 
  (1 / taylor_time + 1 / jennifer_time = 1 / combined_time) → taylor_time = 12 := by
  sorry

end taylor_painting_time_l983_98364


namespace polynomial_remainder_l983_98354

theorem polynomial_remainder (x : ℤ) : (x^2008 + 2008*x + 2008) % (x + 1) = 1 := by
  sorry

end polynomial_remainder_l983_98354


namespace two_and_three_digit_sum_l983_98379

theorem two_and_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 
  100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 4 * x * y ∧ 
  x + y = 266 := by
sorry

end two_and_three_digit_sum_l983_98379


namespace multiplication_increase_l983_98323

theorem multiplication_increase (x : ℝ) : 18 * x = 18 + 198 → x = 12 := by
  sorry

end multiplication_increase_l983_98323


namespace concentric_circles_area_l983_98312

theorem concentric_circles_area (r : Real) : 
  r > 0 → 
  (π * (3*r)^2 - π * (2*r)^2) + (π * (2*r)^2 - π * r^2) = 72 * π := by
  sorry

end concentric_circles_area_l983_98312


namespace square_sum_given_sum_square_and_product_l983_98318

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 := by
  sorry

end square_sum_given_sum_square_and_product_l983_98318


namespace triangle_interior_center_points_l983_98327

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle ABC in the Cartesian plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Count of interior center points in a triangle -/
def interiorCenterPoints (t : Triangle) : ℕ :=
  sorry

/-- The main theorem -/
theorem triangle_interior_center_points :
  let t : Triangle := {
    A := { x := 0, y := 0 },
    B := { x := 200, y := 100 },
    C := { x := 30, y := 330 }
  }
  interiorCenterPoints t = 31480 := by sorry

end triangle_interior_center_points_l983_98327


namespace triangle_abc_properties_l983_98338

theorem triangle_abc_properties (A B C : Real) (h : Real) :
  A + B + C = Real.pi →
  A + B = 3 * C →
  2 * Real.sin (A - C) = Real.sin B →
  h * 5 / 2 = Real.sin C * Real.sin A * Real.sin B * 25 →
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧ h = 6 := by
  sorry

end triangle_abc_properties_l983_98338


namespace range_of_a_l983_98382

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) →
  (a < -3 ∨ a > 3) :=
by sorry

end range_of_a_l983_98382


namespace triangle_angle_relation_minimum_l983_98301

theorem triangle_angle_relation_minimum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hSum : A + B + C = π) (hTriangle : 3 * (Real.cos (2 * A) - Real.cos (2 * C)) = 1 - Real.cos (2 * B)) :
  ∃ (x : ℝ), ∀ (y : ℝ), 
    (Real.sin C / (Real.sin A * Real.sin B) + Real.cos C / Real.sin C) ≥ y → 
    y ≥ 2 * Real.sqrt 7 / 3 :=
sorry

end triangle_angle_relation_minimum_l983_98301


namespace problem_statement_l983_98385

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (x + y + z) * (1/x + 1/y + 1/z) = 91/10) :
  ⌊(x^3 + y^3 + z^3) * (1/x^3 + 1/y^3 + 1/z^3)⌋ = 9 := by
sorry

end problem_statement_l983_98385


namespace equation_solution_l983_98324

theorem equation_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y + 4 = 2 * (Real.sqrt (2*x+1) + Real.sqrt (2*y+1))) : 
  x = 1 + Real.sqrt 2 ∧ y = 1 + Real.sqrt 2 := by
  sorry

end equation_solution_l983_98324


namespace mikes_video_game_earnings_l983_98302

theorem mikes_video_game_earnings :
  let total_games : ℕ := 20
  let non_working_games : ℕ := 11
  let price_per_game : ℚ := 8
  let sales_tax_rate : ℚ := 12 / 100
  
  let working_games : ℕ := total_games - non_working_games
  let total_revenue : ℚ := working_games * price_per_game
  
  total_revenue = 72 :=
by sorry

end mikes_video_game_earnings_l983_98302


namespace max_value_trig_expression_l983_98342

theorem max_value_trig_expression (a b : ℝ) :
  (∀ θ : ℝ, a * Real.cos (2 * θ) + b * Real.sin (2 * θ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (2 * θ) + b * Real.sin (2 * θ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end max_value_trig_expression_l983_98342


namespace problem_solution_l983_98343

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 5) : 
  x^2*y + x*y^2 + x^2 + y^2 = 18 := by sorry

end problem_solution_l983_98343


namespace gcd_factorial_eight_ten_l983_98331

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l983_98331


namespace equality_condition_l983_98344

theorem equality_condition (p q r : ℝ) : p + q * r = (p + q) * (p + r) ↔ p + q + r = 0 := by
  sorry

end equality_condition_l983_98344


namespace lake_crossing_wait_time_l983_98356

theorem lake_crossing_wait_time 
  (lake_width : ℝ) 
  (janet_initial_speed : ℝ) 
  (janet_speed_decrease : ℝ) 
  (sister_initial_speed : ℝ) 
  (sister_speed_increase : ℝ) 
  (h1 : lake_width = 60) 
  (h2 : janet_initial_speed = 30) 
  (h3 : janet_speed_decrease = 0.15) 
  (h4 : sister_initial_speed = 12) 
  (h5 : sister_speed_increase = 0.20) :
  ∃ (wait_time : ℝ), 
    abs (wait_time - 2.156862745) < 0.000001 ∧ 
    wait_time = 
      ((lake_width / sister_initial_speed) + 
       ((lake_width - sister_initial_speed) / (sister_initial_speed * (1 + sister_speed_increase)))) - 
      ((lake_width / (2 * janet_initial_speed)) + 
       (lake_width / (2 * janet_initial_speed * (1 - janet_speed_decrease)))) := by
  sorry

end lake_crossing_wait_time_l983_98356


namespace point_on_x_axis_l983_98319

theorem point_on_x_axis (m : ℝ) : (3, m) ∈ {p : ℝ × ℝ | p.2 = 0} → m = 0 := by
  sorry

end point_on_x_axis_l983_98319


namespace f_of_4_equals_23_l983_98386

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * (2 * x + 2) + 3

-- State the theorem
theorem f_of_4_equals_23 : f 4 = 23 := by
  sorry

end f_of_4_equals_23_l983_98386


namespace degree_to_radian_conversion_negative_300_degrees_to_radians_l983_98371

theorem degree_to_radian_conversion (angle_in_degrees : ℝ) : 
  angle_in_degrees * (π / 180) = angle_in_degrees * π / 180 := by sorry

theorem negative_300_degrees_to_radians : 
  -300 * (π / 180) = -5 * π / 3 := by sorry

end degree_to_radian_conversion_negative_300_degrees_to_radians_l983_98371


namespace missing_part_equation_l983_98307

theorem missing_part_equation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  ∃ x : ℝ, x * (2/3 * a * b) = 2 * a^2 * b^3 + (1/3) * a^3 * b^2 ∧ 
           x = 3 * a * b^2 + (1/2) * a^2 * b :=
sorry

end missing_part_equation_l983_98307


namespace complement_of_union_equals_five_l983_98337

def U : Finset ℕ := {1, 3, 5, 9}
def A : Finset ℕ := {1, 3, 9}
def B : Finset ℕ := {1, 9}

theorem complement_of_union_equals_five : (U \ (A ∪ B)) = {5} := by
  sorry

end complement_of_union_equals_five_l983_98337


namespace grace_total_pennies_l983_98353

/-- The value of a dime in pennies -/
def dime_value : ℕ := 10

/-- The value of a nickel in pennies -/
def nickel_value : ℕ := 5

/-- The number of dimes Grace has -/
def grace_dimes : ℕ := 10

/-- The number of nickels Grace has -/
def grace_nickels : ℕ := 10

/-- Theorem: Grace will have 150 pennies after exchanging her dimes and nickels -/
theorem grace_total_pennies : 
  grace_dimes * dime_value + grace_nickels * nickel_value = 150 := by
  sorry

end grace_total_pennies_l983_98353


namespace max_value_part1_m_value_part2_l983_98346

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x^2 + 4 * m * x - 1

-- Part 1
theorem max_value_part1 :
  ∀ θ : ℝ, 0 < θ ∧ θ < π/2 →
  (f 2 (Real.sin θ)) / (Real.sin θ) ≤ -2 * Real.sqrt 2 + 8 :=
sorry

-- Part 2
theorem m_value_part2 :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f m x ≤ 7) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f m x = 7) →
  m = -2.5 ∨ m = 2.5 :=
sorry

end max_value_part1_m_value_part2_l983_98346


namespace not_p_or_q_l983_98348

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sin x > 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

-- Theorem to prove
theorem not_p_or_q : ¬(p ∨ q) := by
  sorry

end not_p_or_q_l983_98348


namespace right_triangle_consecutive_even_legs_l983_98308

theorem right_triangle_consecutive_even_legs (a b c : ℕ) : 
  -- a and b are the legs, c is the hypotenuse
  (a * a + b * b = c * c) →  -- Pythagorean theorem
  (∃ k : ℕ, a = 2 * k ∧ b = 2 * k + 2) →  -- consecutive even numbers
  (c = 34) →  -- hypotenuse is 34
  (a + b = 46) :=  -- sum of legs is 46
by sorry

end right_triangle_consecutive_even_legs_l983_98308


namespace adult_ticket_price_l983_98334

-- Define the variables and constants
def adult_price : ℝ := sorry
def child_price : ℝ := 3.50
def total_tickets : ℕ := 21
def total_revenue : ℝ := 83.50
def adult_tickets : ℕ := 5

-- Theorem statement
theorem adult_ticket_price :
  adult_price = 5.50 :=
by
  -- Proof goes here
  sorry

end adult_ticket_price_l983_98334


namespace jerry_tom_distance_difference_l983_98315

/-- The difference in distance run by Jerry and Tom around a square block -/
def distance_difference (block_side : ℝ) (street_width : ℝ) : ℝ :=
  4 * (block_side + 2 * street_width) - 4 * block_side

/-- Theorem stating the difference in distance run by Jerry and Tom -/
theorem jerry_tom_distance_difference :
  distance_difference 500 30 = 240 := by
  sorry

end jerry_tom_distance_difference_l983_98315


namespace factor_divisor_statements_l983_98320

theorem factor_divisor_statements :
  (∃ n : ℕ, 24 = 4 * n) ∧
  (∃ n : ℕ, 200 = 10 * n) ∧
  (¬ ∃ n : ℕ, 133 = 19 * n ∨ ∃ n : ℕ, 57 = 19 * n) ∧
  (∃ n : ℕ, 90 = 30 * n ∨ ∃ n : ℕ, 65 = 30 * n) ∧
  (¬ ∃ n : ℕ, 49 = 7 * n ∨ ∃ n : ℕ, 98 = 7 * n) :=
by sorry

end factor_divisor_statements_l983_98320


namespace salary_decrease_l983_98392

theorem salary_decrease (initial_salary : ℝ) (cut1 cut2 cut3 : ℝ) 
  (h1 : cut1 = 0.08) (h2 : cut2 = 0.14) (h3 : cut3 = 0.18) :
  1 - (1 - cut1) * (1 - cut2) * (1 - cut3) = 1 - (0.92 * 0.86 * 0.82) := by
  sorry

end salary_decrease_l983_98392


namespace insane_vampire_statement_l983_98333

/-- Represents a being in Transylvania -/
inductive TransylvanianBeing
| Human
| Vampire

/-- Represents the mental state of a being -/
inductive MentalState
| Sane
| Insane

/-- Represents a Transylvanian entity with a mental state -/
structure Transylvanian :=
  (being : TransylvanianBeing)
  (state : MentalState)

/-- Predicate for whether a Transylvanian makes the statement "I am not a sane person" -/
def makesSanityStatement (t : Transylvanian) : Prop :=
  t.state = MentalState.Insane

/-- Theorem: A Transylvanian who states "I am not a sane person" must be an insane vampire -/
theorem insane_vampire_statement 
  (t : Transylvanian) 
  (h : makesSanityStatement t) : 
  t.being = TransylvanianBeing.Vampire ∧ t.state = MentalState.Insane :=
by sorry


end insane_vampire_statement_l983_98333


namespace polynomial_factorization_l983_98303

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + a*b + a*c + b^2 + b*c + c^2) := by
  sorry

end polynomial_factorization_l983_98303


namespace g_sum_zero_l983_98365

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end g_sum_zero_l983_98365


namespace quadratic_integer_roots_l983_98387

theorem quadratic_integer_roots (p : ℕ) (b : ℕ) (hp : Prime p) (hb : b > 0) :
  (∃ x y : ℤ, x^2 - b*x + b*p = 0 ∧ y^2 - b*y + b*p = 0) ↔ b = (p + 1)^2 ∨ b = 4*p :=
sorry

end quadratic_integer_roots_l983_98387


namespace percent_fifteen_percent_l983_98325

-- Define the operations
def percent (y : Int) : Int := 8 - y
def prepercent (y : Int) : Int := y - 8

-- Theorem statement
theorem percent_fifteen_percent : prepercent (percent 15) = -15 := by
  sorry

end percent_fifteen_percent_l983_98325


namespace line_ellipse_intersection_condition_l983_98322

/-- The range of m for which a line y = kx + 1 and an ellipse x²/5 + y²/m = 1 always intersect -/
theorem line_ellipse_intersection_condition (k : ℝ) :
  ∃ (m : ℝ), (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1 → 
    (m ≥ 1 ∧ m ≠ 5)) :=
sorry

end line_ellipse_intersection_condition_l983_98322


namespace union_of_A_and_B_is_reals_l983_98358

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- State the theorem
theorem union_of_A_and_B_is_reals : A ∪ B = Set.univ := by sorry

end union_of_A_and_B_is_reals_l983_98358


namespace intersection_complement_equality_l983_98357

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement_equality : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end intersection_complement_equality_l983_98357


namespace fraction_simplification_l983_98380

theorem fraction_simplification (x y : ℝ) (h : x ≠ 3*y ∧ x ≠ -3*y) : 
  (2*x)/(x^2 - 9*y^2) - 1/(x - 3*y) = 1/(x + 3*y) := by
  sorry

end fraction_simplification_l983_98380


namespace fraction_is_positive_integer_l983_98351

theorem fraction_is_positive_integer (q : ℕ+) :
  (∃ k : ℕ+, (5 * q + 40 : ℚ) / (3 * q - 8 : ℚ) = k) ↔ 3 ≤ q ∧ q ≤ 28 := by
  sorry

end fraction_is_positive_integer_l983_98351


namespace max_value_of_expression_l983_98335

theorem max_value_of_expression (x y z w : ℝ) (h : x + y + z + w = 1) :
  ∃ (M : ℝ), M = x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z ∧
  M ≤ (3/2 : ℝ) ∧
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ + y₀ + z₀ + w₀ = 1 ∧
    (3/2 : ℝ) = x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀ :=
sorry

end max_value_of_expression_l983_98335


namespace radius_of_special_isosceles_triangle_l983_98376

/-- Represents an isosceles triangle with a circumscribed circle. -/
structure IsoscelesTriangleWithCircle where
  /-- The length of the base of the isosceles triangle -/
  base : ℝ
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The two equal sides of the triangle are each twice the length of the base -/
  equal_sides_twice_base : base > 0
  /-- The perimeter in inches equals the area of the circumscribed circle in square inches -/
  perimeter_equals_circle_area : 5 * base = π * radius^2

/-- 
The radius of the circumscribed circle of an isosceles triangle is 2√5/π inches,
given that the perimeter in inches equals the area of the circumscribed circle in square inches,
and the two equal sides of the triangle are each twice the length of the base.
-/
theorem radius_of_special_isosceles_triangle (t : IsoscelesTriangleWithCircle) : 
  t.radius = 2 * Real.sqrt 5 / π :=
by sorry

end radius_of_special_isosceles_triangle_l983_98376


namespace larger_solution_quadratic_l983_98370

theorem larger_solution_quadratic : ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 9*x - 22 = 0 ∧ 
  y^2 - 9*y - 22 = 0 ∧ 
  (∀ z : ℝ, z^2 - 9*z - 22 = 0 → z = x ∨ z = y) ∧
  max x y = 11 := by
sorry

end larger_solution_quadratic_l983_98370


namespace no_adjacent_knights_probability_l983_98326

/-- The number of knights seated in a circle -/
def total_knights : ℕ := 20

/-- The number of knights selected for the quest -/
def selected_knights : ℕ := 4

/-- The probability that no two of the selected knights are sitting next to each other -/
def probability : ℚ := 60 / 7

/-- Theorem stating that the probability of no two selected knights sitting next to each other is 60/7 -/
theorem no_adjacent_knights_probability :
  probability = 60 / 7 := by sorry

end no_adjacent_knights_probability_l983_98326


namespace pet_store_cages_l983_98397

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 13 → sold_puppies = 7 → puppies_per_cage = 2 →
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l983_98397


namespace average_song_length_l983_98377

-- Define the given conditions
def hours_per_month : ℝ := 20
def cost_per_song : ℝ := 0.5
def yearly_cost : ℝ := 2400
def months_per_year : ℕ := 12
def minutes_per_hour : ℕ := 60

-- Define the theorem
theorem average_song_length :
  let songs_per_year : ℝ := yearly_cost / cost_per_song
  let songs_per_month : ℝ := songs_per_year / months_per_year
  let total_minutes_per_month : ℝ := hours_per_month * minutes_per_hour
  total_minutes_per_month / songs_per_month = 3 := by
  sorry


end average_song_length_l983_98377


namespace decimal_difference_l983_98394

/-- The value of the repeating decimal 0.737373... -/
def repeating_decimal : ℚ := 73 / 99

/-- The value of the terminating decimal 0.73 -/
def terminating_decimal : ℚ := 73 / 100

/-- The difference between the repeating decimal 0.737373... and the terminating decimal 0.73 -/
def difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference : difference = 73 / 9900 := by
  sorry

end decimal_difference_l983_98394


namespace fish_tank_problem_l983_98321

theorem fish_tank_problem (initial_fish caught_fish : ℕ) : 
  caught_fish = initial_fish - 4 →
  initial_fish + caught_fish = 20 →
  caught_fish = 8 := by
  sorry

end fish_tank_problem_l983_98321


namespace pine_saplings_sample_count_l983_98304

/-- Calculates the number of pine saplings in a stratified sample -/
def pine_saplings_in_sample (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℕ :=
  (pine_saplings * sample_size) / total_saplings

/-- Theorem: The number of pine saplings in the stratified sample is 20 -/
theorem pine_saplings_sample_count :
  pine_saplings_in_sample 30000 4000 150 = 20 := by
  sorry

#eval pine_saplings_in_sample 30000 4000 150

end pine_saplings_sample_count_l983_98304


namespace pants_price_satisfies_conditions_l983_98341

/-- The original price of pants that satisfies the given conditions -/
def original_pants_price : ℝ := 110

/-- The number of pairs of pants purchased -/
def num_pants : ℕ := 4

/-- The number of pairs of socks purchased -/
def num_socks : ℕ := 2

/-- The original price of socks -/
def original_socks_price : ℝ := 60

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.3

/-- The total cost after discount -/
def total_cost_after_discount : ℝ := 392

/-- Theorem stating that the original pants price satisfies the given conditions -/
theorem pants_price_satisfies_conditions :
  (num_pants : ℝ) * original_pants_price * (1 - discount_rate) +
  (num_socks : ℝ) * original_socks_price * (1 - discount_rate) =
  total_cost_after_discount := by sorry

end pants_price_satisfies_conditions_l983_98341


namespace correct_statements_count_l983_98332

-- Define a structure for a statistical statement
structure StatStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : StatStatement :=
  ⟨1, "Subtracting the same number from each datum in a data set does not change the mean or the variance", false⟩

def statement2 : StatStatement :=
  ⟨2, "In a survey of audience feedback in a theater, randomly selecting one row from 50 rows (equal number of people in each row) for the survey is an example of stratified sampling", false⟩

def statement3 : StatStatement :=
  ⟨3, "It is known that random variable X follows a normal distribution N(3,1), and P(2≤X≤4) = 0.6826, then P(X>4) is equal to 0.1587", true⟩

def statement4 : StatStatement :=
  ⟨4, "A unit has 750 employees, of which there are 350 young workers, 250 middle-aged workers, and 150 elderly workers. To understand the health status of the workers in the unit, stratified sampling is used to draw a sample. If there are 7 young workers in the sample, then the sample size is 15", true⟩

-- Define the list of all statements
def allStatements : List StatStatement := [statement1, statement2, statement3, statement4]

-- Theorem to prove
theorem correct_statements_count :
  (allStatements.filter (λ s => s.isCorrect)).length = 2 := by
  sorry

end correct_statements_count_l983_98332


namespace candidate_X_votes_l983_98306

/-- Represents the number of votes for each candidate -/
structure Votes where
  X : ℕ
  Y : ℕ
  Z : ℕ
  W : ℕ

/-- Represents the conditions of the mayoral election -/
def ElectionConditions (v : Votes) : Prop :=
  v.X = v.Y + v.Y / 2 ∧
  v.Y = v.Z - (2 * v.Z) / 5 ∧
  v.W = (3 * v.X) / 4 ∧
  v.Z = 25000

theorem candidate_X_votes (v : Votes) (h : ElectionConditions v) : v.X = 22500 := by
  sorry

end candidate_X_votes_l983_98306


namespace remaining_budget_for_public_spaces_l983_98330

/-- Proof of remaining budget for public spaces -/
theorem remaining_budget_for_public_spaces 
  (total_budget : ℝ) 
  (education_budget : ℝ) 
  (h1 : total_budget = 32000000)
  (h2 : education_budget = 12000000) :
  total_budget - (total_budget / 2 + education_budget) = 4000000 := by
  sorry

end remaining_budget_for_public_spaces_l983_98330


namespace simplify_fraction_product_l983_98369

theorem simplify_fraction_product : (144 : ℚ) / 1296 * 72 = 8 := by
  sorry

end simplify_fraction_product_l983_98369


namespace no_intersection_for_given_scenarios_l983_98396

/-- Determines if two circles intersect based on their radii and the distance between their centers -/
def circlesIntersect (r1 r2 d : ℝ) : Prop :=
  |r1 - r2| ≤ d ∧ d ≤ r1 + r2

theorem no_intersection_for_given_scenarios :
  let r1 : ℝ := 3
  let r2 : ℝ := 5
  let d1 : ℝ := 9
  let d2 : ℝ := 1
  ¬(circlesIntersect r1 r2 d1) ∧ ¬(circlesIntersect r1 r2 d2) :=
by
  sorry

#check no_intersection_for_given_scenarios

end no_intersection_for_given_scenarios_l983_98396


namespace quadratic_solution_property_l983_98350

theorem quadratic_solution_property : 
  ∀ p q : ℝ, 
  (2 * p^2 + 8 * p - 42 = 0) → 
  (2 * q^2 + 8 * q - 42 = 0) → 
  p ≠ q → 
  (p - q + 2)^2 = 144 := by
sorry

end quadratic_solution_property_l983_98350


namespace black_tiles_imply_total_tiles_l983_98389

/-- Represents a square floor tiled with congruent square tiles -/
structure TiledFloor where
  side_length : ℕ

/-- Counts the number of black tiles on the diagonals of a square floor -/
def diagonal_black_tiles (floor : TiledFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Counts the number of black tiles in a quarter of the floor -/
def quarter_black_tiles (floor : TiledFloor) : ℕ :=
  (floor.side_length ^ 2) / 4

/-- Calculates the total number of tiles on the floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem stating that if there are 225 black tiles in total, then the total number of tiles is 1024 -/
theorem black_tiles_imply_total_tiles (floor : TiledFloor) :
  diagonal_black_tiles floor + quarter_black_tiles floor = 225 →
  total_tiles floor = 1024 := by
  sorry

end black_tiles_imply_total_tiles_l983_98389


namespace decimal_division_l983_98355

theorem decimal_division : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by
  sorry

end decimal_division_l983_98355


namespace root_implies_range_l983_98345

-- Define the function f(x) = ax^2 - 2ax + a - 9
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + a - 9

-- Define the property that f has at least one root in (-2, 0)
def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -2 < x ∧ x < 0 ∧ f a x = 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -9 ∨ (1 < a ∧ a < 9) ∨ 9 < a

-- State the theorem
theorem root_implies_range :
  ∀ a : ℝ, has_root_in_interval a → a_range a :=
sorry

end root_implies_range_l983_98345


namespace factorial_square_root_square_l983_98329

-- Define factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- State the theorem
theorem factorial_square_root_square : 
  (Real.sqrt (factorial 5 * factorial 4 : ℝ))^2 = 2880 := by
  sorry

end factorial_square_root_square_l983_98329


namespace problem_solution_l983_98359

theorem problem_solution (A B : ℝ) 
  (h1 : 100 * A = 35^2 - 15^2) 
  (h2 : (A - 1)^6 = 27^B) : 
  A = 10 ∧ B = 4 := by
sorry

end problem_solution_l983_98359
