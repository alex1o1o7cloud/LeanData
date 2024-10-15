import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2282_228280

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, x^2 + (a - 4)*x + 4 > 0) ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2282_228280


namespace NUMINAMATH_CALUDE_opposite_of_abs_neg_five_l2282_228279

theorem opposite_of_abs_neg_five : -(|-5|) = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_abs_neg_five_l2282_228279


namespace NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l2282_228257

theorem largest_sum_is_three_fourths : 
  let sums : List ℚ := [1/4 + 1/2, 1/4 + 1/3, 1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/11]
  (∀ x ∈ sums, x ≤ 1/4 + 1/2) ∧ (1/4 + 1/2 = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l2282_228257


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l2282_228221

/-- The number of wheels on a car -/
def car_wheels : ℕ := 4

/-- The number of wheels on a bike -/
def bike_wheels : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 14

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 5

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * car_wheels + num_bikes * bike_wheels

theorem parking_lot_wheels : total_wheels = 66 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l2282_228221


namespace NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l2282_228299

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- State the theorem
theorem union_of_A_and_B_is_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l2282_228299


namespace NUMINAMATH_CALUDE_staircase_steps_l2282_228244

/-- Represents the number of toothpicks used in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

/-- The number of toothpicks used in a 3-step staircase -/
def three_step_toothpicks : ℕ := 27

/-- The target number of toothpicks -/
def target_toothpicks : ℕ := 270

theorem staircase_steps :
  ∃ (n : ℕ), toothpicks n = target_toothpicks ∧ n = 12 :=
sorry

end NUMINAMATH_CALUDE_staircase_steps_l2282_228244


namespace NUMINAMATH_CALUDE_books_left_l2282_228283

theorem books_left (initial_books sold_books : ℝ) 
  (h1 : initial_books = 51.5)
  (h2 : sold_books = 45.75) : 
  initial_books - sold_books = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_books_left_l2282_228283


namespace NUMINAMATH_CALUDE_decimal_difference_l2282_228253

/-- The value of the repeating decimal 0.737373... -/
def repeating_decimal : ℚ := 73 / 99

/-- The value of the terminating decimal 0.73 -/
def terminating_decimal : ℚ := 73 / 100

/-- The difference between the repeating decimal 0.737373... and the terminating decimal 0.73 -/
def difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference : difference = 73 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l2282_228253


namespace NUMINAMATH_CALUDE_grace_total_pennies_l2282_228272

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

end NUMINAMATH_CALUDE_grace_total_pennies_l2282_228272


namespace NUMINAMATH_CALUDE_two_and_three_digit_sum_l2282_228247

theorem two_and_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 
  100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 4 * x * y ∧ 
  x + y = 266 := by
sorry

end NUMINAMATH_CALUDE_two_and_three_digit_sum_l2282_228247


namespace NUMINAMATH_CALUDE_distinct_configurations_correct_l2282_228290

/-- Represents the number of distinct configurations of n coins arranged in a circle
    that cannot be transformed into one another by flipping adjacent pairs of coins
    with the same orientation. -/
def distinctConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else 2

theorem distinct_configurations_correct (n : ℕ) :
  distinctConfigurations n = if n % 2 = 0 then n + 1 else 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_configurations_correct_l2282_228290


namespace NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_max_value_l2282_228251

/-- A quadratic function f(x) = ax^2 + bx + c where the solution set of f(x) > -2x is {x | 1 < x < 3} -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) 
  (h_solution_set : ∀ x, 1 < x ∧ x < 3 ↔ QuadraticFunction a b c x > -2 * x) :
  (∃ x₀ > 0, QuadraticFunction a b c x₀ = 2 * a ∧ 
   ∀ x, QuadraticFunction a b c x = 2 * a → x = x₀) →
  QuadraticFunction a b c = fun x ↦ -x^2 + 2*x - 3 :=
sorry

theorem quadratic_function_max_value (a b c : ℝ) 
  (h_solution_set : ∀ x, 1 < x ∧ x < 3 ↔ QuadraticFunction a b c x > -2 * x) :
  (∃ x₀, ∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c x₀ ∧ 
   QuadraticFunction a b c x₀ > 0) →
  (-2 - Real.sqrt 3 < a ∧ a < 0) ∨ (-2 + Real.sqrt 3 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_max_value_l2282_228251


namespace NUMINAMATH_CALUDE_basic_computer_price_l2282_228262

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

end NUMINAMATH_CALUDE_basic_computer_price_l2282_228262


namespace NUMINAMATH_CALUDE_y_value_proof_l2282_228211

theorem y_value_proof (x y z a b c : ℝ) 
  (ha : x * y / (x + y) = a)
  (hb : x * z / (x + z) = b)
  (hc : y * z / (y + z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  y = 2 * a * b * c / (b * c + a * c - a * b) :=
sorry

end NUMINAMATH_CALUDE_y_value_proof_l2282_228211


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2282_228289

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2282_228289


namespace NUMINAMATH_CALUDE_inequality_proof_l2282_228255

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2282_228255


namespace NUMINAMATH_CALUDE_insane_vampire_statement_l2282_228296

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


end NUMINAMATH_CALUDE_insane_vampire_statement_l2282_228296


namespace NUMINAMATH_CALUDE_zoo_meat_supply_duration_l2282_228228

/-- The number of full days a meat supply lasts for a group of animals -/
def meat_supply_duration (lion_consumption tiger_consumption leopard_consumption hyena_consumption total_meat : ℕ) : ℕ :=
  (total_meat / (lion_consumption + tiger_consumption + leopard_consumption + hyena_consumption))

/-- Theorem: Given the specified daily meat consumption for four animals and a total meat supply of 500 kg, the meat supply will last for 7 full days -/
theorem zoo_meat_supply_duration :
  meat_supply_duration 25 20 15 10 500 = 7 := by
  sorry

end NUMINAMATH_CALUDE_zoo_meat_supply_duration_l2282_228228


namespace NUMINAMATH_CALUDE_equivalent_rotation_l2282_228215

/-- Given a full rotation of 450 degrees, if a point is rotated 650 degrees clockwise
    to reach a destination, then the equivalent counterclockwise rotation to reach
    the same destination is 250 degrees. -/
theorem equivalent_rotation (full_rotation : ℕ) (clockwise_rotation : ℕ) (counterclockwise_rotation : ℕ) : 
  full_rotation = 450 → 
  clockwise_rotation = 650 → 
  counterclockwise_rotation < full_rotation →
  (clockwise_rotation % full_rotation + counterclockwise_rotation) % full_rotation = 0 →
  counterclockwise_rotation = 250 := by
  sorry

#check equivalent_rotation

end NUMINAMATH_CALUDE_equivalent_rotation_l2282_228215


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2282_228223

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  (9 / x + 4 / y + 25 / z) ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2282_228223


namespace NUMINAMATH_CALUDE_lcm_18_30_l2282_228261

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l2282_228261


namespace NUMINAMATH_CALUDE_ada_original_seat_l2282_228234

-- Define the number of seats
def num_seats : ℕ := 6

-- Define the movements of friends
def bea_move : ℤ := 3
def ceci_move : ℤ := 1
def dee_move : ℤ := -2
def edie_move : ℤ := -1

-- Define Ada's final position
def ada_final_seat : ℕ := 2

-- Theorem statement
theorem ada_original_seat :
  let net_displacement := bea_move + ceci_move + dee_move + edie_move
  net_displacement = 1 →
  ∃ (ada_original : ℕ), 
    ada_original > 0 ∧ 
    ada_original ≤ num_seats ∧
    ada_original - ada_final_seat = 1 := by
  sorry

end NUMINAMATH_CALUDE_ada_original_seat_l2282_228234


namespace NUMINAMATH_CALUDE_blue_chip_percentage_l2282_228214

theorem blue_chip_percentage
  (total : ℕ)
  (blue : ℕ)
  (white : ℕ)
  (green : ℕ)
  (h1 : blue = 3)
  (h2 : white = total / 2)
  (h3 : green = 12)
  (h4 : total = blue + white + green) :
  (blue : ℚ) / total * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_blue_chip_percentage_l2282_228214


namespace NUMINAMATH_CALUDE_trapezoid_midline_length_l2282_228209

/-- Given a trapezoid with parallel sides of length a and b, 
    the length of the line segment joining the midpoints of these parallel sides is (a + b) / 2 -/
theorem trapezoid_midline_length (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  let midline_length := (a + b) / 2
  midline_length = (a + b) / 2 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_midline_length_l2282_228209


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2282_228276

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (q > 0) →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (a 3 * a 7 = 4 * (a 4)^2) →
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2282_228276


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2282_228266

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 3 + a * (Complex.I + 2) + b = 0 → a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2282_228266


namespace NUMINAMATH_CALUDE_factorial_square_root_square_l2282_228254

-- Define factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- State the theorem
theorem factorial_square_root_square : 
  (Real.sqrt (factorial 5 * factorial 4 : ℝ))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_square_l2282_228254


namespace NUMINAMATH_CALUDE_zhe_same_meaning_and_usage_l2282_228227

/-- Represents a function word in classical Chinese --/
structure FunctionWord where
  word : String
  meaning : String
  usage : String

/-- Represents a sentence in classical Chinese --/
structure Sentence where
  text : String
  functionWords : List FunctionWord

/-- The function word "者" as it appears in the first sentence --/
def zhe1 : FunctionWord := {
  word := "者",
  meaning := "the person",
  usage := "nominalizer"
}

/-- The function word "者" as it appears in the second sentence --/
def zhe2 : FunctionWord := {
  word := "者",
  meaning := "the person",
  usage := "nominalizer"
}

/-- The first sentence containing "者" --/
def sentence1 : Sentence := {
  text := "智者能勿丧",
  functionWords := [zhe1]
}

/-- The second sentence containing "者" --/
def sentence2 : Sentence := {
  text := "所知贫穷者，将从我乎？",
  functionWords := [zhe2]
}

/-- Theorem stating that the function word "者" has the same meaning and usage in both sentences --/
theorem zhe_same_meaning_and_usage : 
  zhe1.meaning = zhe2.meaning ∧ zhe1.usage = zhe2.usage :=
sorry

end NUMINAMATH_CALUDE_zhe_same_meaning_and_usage_l2282_228227


namespace NUMINAMATH_CALUDE_diamond_six_three_l2282_228263

/-- Diamond operation defined as a ◇ b = 4a + 2b -/
def diamond (a b : ℝ) : ℝ := 4 * a + 2 * b

/-- Theorem stating that 6 ◇ 3 = 30 -/
theorem diamond_six_three : diamond 6 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_diamond_six_three_l2282_228263


namespace NUMINAMATH_CALUDE_candidate_X_votes_l2282_228231

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

end NUMINAMATH_CALUDE_candidate_X_votes_l2282_228231


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2282_228275

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l2282_228275


namespace NUMINAMATH_CALUDE_f_derivative_and_value_l2282_228282

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.cos x ^ 4

theorem f_derivative_and_value :
  (∀ x, deriv f x = -Real.sin (4 * x)) ∧
  (deriv f (π / 6) = -Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_and_value_l2282_228282


namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l2282_228286

/-- Given two lines that intersect at (2, 3), prove the equation of the line
    passing through the points formed by their coefficients. -/
theorem line_through_coefficient_points
  (A₁ B₁ A₂ B₂ : ℝ) 
  (h₁ : A₁ * 2 + B₁ * 3 = 1)
  (h₂ : A₂ * 2 + B₂ * 3 = 1) :
  ∀ x y : ℝ, (x = A₁ ∧ y = B₁) ∨ (x = A₂ ∧ y = B₂) → 2*x + 3*y = 1 :=
sorry

end NUMINAMATH_CALUDE_line_through_coefficient_points_l2282_228286


namespace NUMINAMATH_CALUDE_work_completion_time_l2282_228267

/-- 
Given that:
- A does 20% less work than B
- A completes the work in 15/2 hours
Prove that B will complete the work in 6 hours
-/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) 
  (h1 : work_rate_A = 0.8 * work_rate_B) 
  (h2 : work_rate_A * (15/2) = 1) : 
  work_rate_B * 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2282_228267


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2282_228264

theorem polynomial_remainder (x : ℤ) : (x^2008 + 2008*x + 2008) % (x + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2282_228264


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2282_228201

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -1, 5]
  Matrix.det A = 44 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l2282_228201


namespace NUMINAMATH_CALUDE_third_month_sale_l2282_228260

def sales_problem (m1 m2 m4 m5 m6 average : ℕ) : Prop :=
  ∃ m3 : ℕ,
    m3 = 6 * average - (m1 + m2 + m4 + m5 + m6) ∧
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = average

theorem third_month_sale :
  sales_problem 5420 5660 6350 6500 8270 6400 →
  ∃ m3 : ℕ, m3 = 6200
:= by sorry

end NUMINAMATH_CALUDE_third_month_sale_l2282_228260


namespace NUMINAMATH_CALUDE_multiplication_increase_l2282_228297

theorem multiplication_increase (x : ℝ) : 18 * x = 18 + 198 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_increase_l2282_228297


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_six_l2282_228229

theorem sqrt_expression_equals_six :
  (Real.sqrt 3 - 1)^2 + Real.sqrt 12 + (1/2)⁻¹ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_six_l2282_228229


namespace NUMINAMATH_CALUDE_function_product_l2282_228268

theorem function_product (f : ℕ → ℝ) 
  (h₁ : ∀ n : ℕ, n > 0 → f (n + 3) = (f n - 1) / (f n + 1))
  (h₂ : f 1 ≠ 0)
  (h₃ : f 1 ≠ 1 ∧ f 1 ≠ -1) :
  f 8 * f 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_product_l2282_228268


namespace NUMINAMATH_CALUDE_angle_supplement_complement_difference_l2282_228202

theorem angle_supplement_complement_difference (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_complement_difference_l2282_228202


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2282_228242

def solution_set : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}

theorem inequality_solution_set :
  ∀ x : ℝ, (x + 2 ≠ 0) → ((x - 3) / (x + 2) ≤ 0 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2282_228242


namespace NUMINAMATH_CALUDE_lee_apple_harvest_l2282_228287

/-- The number of baskets Mr. Lee used to pack apples -/
def num_baskets : ℕ := 19

/-- The number of apples in each basket -/
def apples_per_basket : ℕ := 25

/-- The total number of apples harvested by Mr. Lee -/
def total_apples : ℕ := num_baskets * apples_per_basket

theorem lee_apple_harvest : total_apples = 475 := by
  sorry

end NUMINAMATH_CALUDE_lee_apple_harvest_l2282_228287


namespace NUMINAMATH_CALUDE_log_expression_equals_two_l2282_228217

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  2 * (log10 2) + log10 25 = 2 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_two_l2282_228217


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2282_228241

theorem simplify_fraction_product : (144 : ℚ) / 1296 * 72 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2282_228241


namespace NUMINAMATH_CALUDE_lake_crossing_wait_time_l2282_228238

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

end NUMINAMATH_CALUDE_lake_crossing_wait_time_l2282_228238


namespace NUMINAMATH_CALUDE_mathville_running_difference_l2282_228207

/-- The side length of a square block in Mathville -/
def block_side_length : ℝ := 500

/-- The width of streets in Mathville -/
def street_width : ℝ := 30

/-- The length of Matt's path around the block -/
def matt_path_length : ℝ := 4 * block_side_length

/-- The length of Mike's path around the block -/
def mike_path_length : ℝ := 4 * (block_side_length + 2 * street_width)

/-- The difference between Mike's and Matt's path lengths -/
def path_length_difference : ℝ := mike_path_length - matt_path_length

theorem mathville_running_difference : path_length_difference = 240 := by
  sorry

end NUMINAMATH_CALUDE_mathville_running_difference_l2282_228207


namespace NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_exterior_angle_regular_pentagon_proof_l2282_228236

/-- The size of an exterior angle of a regular pentagon is 72 degrees. -/
theorem exterior_angle_regular_pentagon : ℝ :=
  72

/-- The number of sides in a pentagon. -/
def pentagon_sides : ℕ := 5

/-- The sum of exterior angles of any polygon in degrees. -/
def sum_exterior_angles : ℝ := 360

/-- Theorem: The size of an exterior angle of a regular pentagon is 72 degrees. -/
theorem exterior_angle_regular_pentagon_proof :
  exterior_angle_regular_pentagon = sum_exterior_angles / pentagon_sides :=
by sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_pentagon_exterior_angle_regular_pentagon_proof_l2282_228236


namespace NUMINAMATH_CALUDE_max_value_part1_m_value_part2_l2282_228288

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

end NUMINAMATH_CALUDE_max_value_part1_m_value_part2_l2282_228288


namespace NUMINAMATH_CALUDE_more_pups_than_adults_l2282_228278

def num_huskies : ℕ := 5
def num_pitbulls : ℕ := 2
def num_golden_retrievers : ℕ := 4

def pups_per_husky : ℕ := 3
def pups_per_pitbull : ℕ := 3
def pups_per_golden_retriever : ℕ := pups_per_husky + 2

def total_adult_dogs : ℕ := num_huskies + num_pitbulls + num_golden_retrievers

def total_pups : ℕ := 
  num_huskies * pups_per_husky + 
  num_pitbulls * pups_per_pitbull + 
  num_golden_retrievers * pups_per_golden_retriever

theorem more_pups_than_adults : total_pups - total_adult_dogs = 30 := by
  sorry

end NUMINAMATH_CALUDE_more_pups_than_adults_l2282_228278


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2282_228235

/-- Given three mutually externally tangent circles with radii a, b, and c,
    the radius r of the inscribed circle satisfies the equation:
    1/r = 1/a + 1/b + 1/c + 2 * sqrt(1/(a*b) + 1/(a*c) + 1/(b*c)) -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  a = 5 → b = 10 → c = 20 → r = 20 / (3.5 + 2 * Real.sqrt 14) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2282_228235


namespace NUMINAMATH_CALUDE_stool_height_is_53_l2282_228222

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height (ceiling_height floor_dip alice_height hat_height reach_above_head light_bulb_distance : ℝ) : ℝ :=
  ceiling_height * 100 - light_bulb_distance - (alice_height * 100 + hat_height + reach_above_head - floor_dip)

/-- Theorem stating that the stool height is 53 cm given the problem conditions -/
theorem stool_height_is_53 :
  stool_height 2.8 3 1.6 5 50 15 = 53 := by
  sorry

end NUMINAMATH_CALUDE_stool_height_is_53_l2282_228222


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l2282_228271

theorem sum_of_cubes_equation (x y : ℝ) (h : x^3 + 21*x*y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l2282_228271


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2282_228273

theorem max_value_of_expression (x y z w : ℝ) (h : x + y + z + w = 1) :
  ∃ (M : ℝ), M = x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z ∧
  M ≤ (3/2 : ℝ) ∧
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ + y₀ + z₀ + w₀ = 1 ∧
    (3/2 : ℝ) = x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀ :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2282_228273


namespace NUMINAMATH_CALUDE_pages_copied_l2282_228208

/-- Given the cost of 7 cents for 5 pages, prove that $35 allows copying 2500 pages. -/
theorem pages_copied (cost_per_5_pages : ℚ) (total_dollars : ℚ) : 
  cost_per_5_pages = 7 / 100 → 
  total_dollars = 35 → 
  (total_dollars * 100 * 5) / cost_per_5_pages = 2500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_l2282_228208


namespace NUMINAMATH_CALUDE_population_average_age_l2282_228249

theorem population_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 10 / 9)
  (h_women_age : avg_age_women = 36)
  (h_men_age : avg_age_men = 33) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 34 + 13 / 19 :=
by sorry

end NUMINAMATH_CALUDE_population_average_age_l2282_228249


namespace NUMINAMATH_CALUDE_equation_solution_l2282_228298

theorem equation_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y + 4 = 2 * (Real.sqrt (2*x+1) + Real.sqrt (2*y+1))) : 
  x = 1 + Real.sqrt 2 ∧ y = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2282_228298


namespace NUMINAMATH_CALUDE_salary_decrease_l2282_228243

theorem salary_decrease (initial_salary : ℝ) (cut1 cut2 cut3 : ℝ) 
  (h1 : cut1 = 0.08) (h2 : cut2 = 0.14) (h3 : cut3 = 0.18) :
  1 - (1 - cut1) * (1 - cut2) * (1 - cut3) = 1 - (0.92 * 0.86 * 0.82) := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_l2282_228243


namespace NUMINAMATH_CALUDE_jerry_tom_distance_difference_l2282_228204

/-- The difference in distance run by Jerry and Tom around a square block -/
def distance_difference (block_side : ℝ) (street_width : ℝ) : ℝ :=
  4 * (block_side + 2 * street_width) - 4 * block_side

/-- Theorem stating the difference in distance run by Jerry and Tom -/
theorem jerry_tom_distance_difference :
  distance_difference 500 30 = 240 := by
  sorry

end NUMINAMATH_CALUDE_jerry_tom_distance_difference_l2282_228204


namespace NUMINAMATH_CALUDE_distribute_tickets_count_l2282_228216

/-- The number of ways to distribute 4 consecutive numbered tickets among 3 people -/
def distribute_tickets : ℕ :=
  -- Number of ways to split 4 tickets into 3 portions
  let split_ways := Nat.choose 3 2
  -- Number of ways to distribute 3 portions to 3 people
  let distribute_ways := Nat.factorial 3
  -- Total number of distribution methods
  split_ways * distribute_ways

/-- Theorem stating that the number of distribution methods is 18 -/
theorem distribute_tickets_count : distribute_tickets = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_tickets_count_l2282_228216


namespace NUMINAMATH_CALUDE_greatest_number_jo_thinking_l2282_228218

theorem greatest_number_jo_thinking : ∃ n : ℕ,
  n < 100 ∧
  (∃ k : ℕ, n = 5 * k - 2) ∧
  (∃ m : ℕ, n = 9 * m - 4) ∧
  (∀ x : ℕ, x < 100 ∧ (∃ k : ℕ, x = 5 * k - 2) ∧ (∃ m : ℕ, x = 9 * m - 4) → x ≤ n) ∧
  n = 68 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_jo_thinking_l2282_228218


namespace NUMINAMATH_CALUDE_triangle_problem_l2282_228281

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (Real.sqrt 3 * Real.sin C - 2 * Real.cos A) * Real.sin B = (2 * Real.sin A - Real.sin C) * Real.cos B →
  a^2 + c^2 = 4 + Real.sqrt 3 →
  (1/2) * a * c * Real.sin B = (3 + Real.sqrt 3) / 4 →
  B = π / 3 ∧ a + b + c = (Real.sqrt 6 + 2 * Real.sqrt 3 + 3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2282_228281


namespace NUMINAMATH_CALUDE_voronovich_inequality_l2282_228213

theorem voronovich_inequality (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6*a*b*c ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_voronovich_inequality_l2282_228213


namespace NUMINAMATH_CALUDE_negative_eight_interpretations_l2282_228291

theorem negative_eight_interpretations :
  (-(- 8) = -(-8)) ∧
  (-(- 8) = (-1) * (-8)) ∧
  (-(- 8) = |(-8)|) ∧
  (-(- 8) = 8) := by
  sorry

end NUMINAMATH_CALUDE_negative_eight_interpretations_l2282_228291


namespace NUMINAMATH_CALUDE_cos_315_degrees_l2282_228239

theorem cos_315_degrees : Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l2282_228239


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2282_228206

def P : Set ℝ := {x | x ≤ 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2282_228206


namespace NUMINAMATH_CALUDE_no_intersection_for_given_scenarios_l2282_228293

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

end NUMINAMATH_CALUDE_no_intersection_for_given_scenarios_l2282_228293


namespace NUMINAMATH_CALUDE_average_song_length_l2282_228256

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


end NUMINAMATH_CALUDE_average_song_length_l2282_228256


namespace NUMINAMATH_CALUDE_missing_part_equation_l2282_228232

theorem missing_part_equation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  ∃ x : ℝ, x * (2/3 * a * b) = 2 * a^2 * b^3 + (1/3) * a^3 * b^2 ∧ 
           x = 3 * a * b^2 + (1/2) * a^2 * b :=
sorry

end NUMINAMATH_CALUDE_missing_part_equation_l2282_228232


namespace NUMINAMATH_CALUDE_angle_measure_possibilities_l2282_228245

theorem angle_measure_possibilities :
  ∃! X : ℕ+, 
    ∃ Y : ℕ+, 
      (X : ℝ) + Y = 180 ∧ 
      (X : ℝ) = 3 * Y := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_possibilities_l2282_228245


namespace NUMINAMATH_CALUDE_range_of_a_l2282_228246

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) →
  (a < -3 ∨ a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2282_228246


namespace NUMINAMATH_CALUDE_decimal_division_l2282_228265

theorem decimal_division : (0.45 : ℚ) / (0.005 : ℚ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l2282_228265


namespace NUMINAMATH_CALUDE_hot_drink_sales_at_2_degrees_l2282_228210

/-- Represents the linear regression equation for hot drink sales -/
def hot_drink_sales (x : ℝ) : ℝ := -2.35 * x + 147.77

/-- Theorem stating that when the temperature is 2℃, approximately 143 hot drinks are sold -/
theorem hot_drink_sales_at_2_degrees :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |hot_drink_sales 2 - 143| < ε :=
sorry

end NUMINAMATH_CALUDE_hot_drink_sales_at_2_degrees_l2282_228210


namespace NUMINAMATH_CALUDE_linear_function_properties_l2282_228226

/-- A linear function y = kx + b where k < 0 and b > 0 -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  h₁ : k < 0
  h₂ : b > 0

/-- Properties of the linear function -/
theorem linear_function_properties (f : LinearFunction) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f.k * x₁ + f.b > f.k * x₂ + f.b) ∧ 
  (f.k * (-1) + f.b ≠ -2) ∧
  (f.k * 0 + f.b = f.b) ∧
  (∀ x : ℝ, x > -f.b / f.k → f.k * x + f.b < 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2282_228226


namespace NUMINAMATH_CALUDE_correct_statements_count_l2282_228295

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

end NUMINAMATH_CALUDE_correct_statements_count_l2282_228295


namespace NUMINAMATH_CALUDE_class_composition_l2282_228224

/-- Represents a child's response about the number of classmates -/
structure Response :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a response is valid according to the problem conditions -/
def is_valid_response (actual_boys actual_girls : ℕ) (r : Response) : Prop :=
  (r.boys = actual_boys ∧ (r.girls = actual_girls + 2 ∨ r.girls = actual_girls - 2)) ∨
  (r.girls = actual_girls ∧ (r.boys = actual_boys + 2 ∨ r.boys = actual_boys - 2))

/-- The main theorem stating the correct number of boys and girls in the class -/
theorem class_composition :
  ∃ (actual_boys actual_girls : ℕ),
    actual_boys = 15 ∧
    actual_girls = 12 ∧
    is_valid_response actual_boys actual_girls ⟨13, 11⟩ ∧
    is_valid_response actual_boys actual_girls ⟨17, 11⟩ ∧
    is_valid_response actual_boys actual_girls ⟨14, 14⟩ :=
  sorry

end NUMINAMATH_CALUDE_class_composition_l2282_228224


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2282_228294

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement_equality : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2282_228294


namespace NUMINAMATH_CALUDE_base_subtraction_equality_l2282_228230

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement --/
theorem base_subtraction_equality : 
  let base_9_num := to_base_10 [5, 2, 3] 9
  let base_6_num := to_base_10 [5, 4, 2] 6
  base_9_num - base_6_num = 165 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_equality_l2282_228230


namespace NUMINAMATH_CALUDE_three_std_dev_below_mean_l2282_228205

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- Calculates the value that is n standard deviations below the mean --/
def valueBelow (nd : NormalDistribution) (n : ℝ) : ℝ :=
  nd.mean - n * nd.stdDev

/-- Theorem: For a normal distribution with standard deviation 2 and mean 51,
    the value 3 standard deviations below the mean is 45 --/
theorem three_std_dev_below_mean (nd : NormalDistribution) 
    (h1 : nd.stdDev = 2) 
    (h2 : nd.mean = 51) : 
    valueBelow nd 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_std_dev_below_mean_l2282_228205


namespace NUMINAMATH_CALUDE_sum_lent_is_300_l2282_228284

/-- Proves that the sum lent is 300, given the conditions of the problem -/
theorem sum_lent_is_300 
  (interest_rate : ℝ) 
  (loan_duration : ℕ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.04)
  (h2 : loan_duration = 8)
  (h3 : interest_difference = 204) :
  ∃ (principal : ℝ), 
    principal * interest_rate * loan_duration = principal - interest_difference ∧ 
    principal = 300 := by
sorry


end NUMINAMATH_CALUDE_sum_lent_is_300_l2282_228284


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2282_228292

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0,
    and the length of its real axis is 1,
    prove that the equation of its asymptotes is y = ±2x -/
theorem hyperbola_asymptotes (a : ℝ) (h1 : a > 0) (h2 : 2 * a = 1) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = 2 * x ∨ f x = -2 * x) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x y, x^2/a^2 - y^2 = 1 → x > δ → |y - f x| < ε) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2282_228292


namespace NUMINAMATH_CALUDE_pet_store_cages_l2282_228258

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 13 → sold_puppies = 7 → puppies_per_cage = 2 →
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2282_228258


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2282_228203

theorem least_positive_integer_with_remainder_one (n : ℕ) : 
  (n > 1) →
  (n % 3 = 1) →
  (n % 4 = 1) →
  (n % 5 = 1) →
  (n % 6 = 1) →
  (n % 7 = 1) →
  (n % 10 = 1) →
  (n % 11 = 1) →
  (∀ m : ℕ, m > 1 → 
    (m % 3 = 1) →
    (m % 4 = 1) →
    (m % 5 = 1) →
    (m % 6 = 1) →
    (m % 7 = 1) →
    (m % 10 = 1) →
    (m % 11 = 1) →
    (n ≤ m)) →
  n = 4621 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l2282_228203


namespace NUMINAMATH_CALUDE_second_number_proof_l2282_228252

theorem second_number_proof : ∃ x : ℕ, 
  (1657 % 1 = 10) ∧ 
  (x % 1 = 7) ∧ 
  (∀ y : ℕ, y > x → ¬(y % 1 = 7)) ∧ 
  (x = 1655) := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2282_228252


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2282_228240

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_1 + a_4 = 10 and a_2 + a_5 = 20, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = q * a n) 
  (h_sum1 : a 1 + a 4 = 10) 
  (h_sum2 : a 2 + a 5 = 20) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2282_228240


namespace NUMINAMATH_CALUDE_probability_arithmetic_progression_l2282_228277

def dice_sides := 4

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  (b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ b = c + 1)

def favorable_outcomes : ℕ := 12

def total_outcomes : ℕ := dice_sides ^ 3

theorem probability_arithmetic_progression :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_arithmetic_progression_l2282_228277


namespace NUMINAMATH_CALUDE_teacher_age_proof_l2282_228270

def teacher_age (num_students : ℕ) (student_avg_age : ℕ) (new_avg_age : ℕ) (total_people : ℕ) : ℕ :=
  (new_avg_age * total_people) - (student_avg_age * num_students)

theorem teacher_age_proof :
  teacher_age 23 22 23 24 = 46 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_proof_l2282_228270


namespace NUMINAMATH_CALUDE_z_value_theorem_l2282_228233

theorem z_value_theorem (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1 / w = 15) (h2 : w^2 + 1 / z = 3) : z = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_z_value_theorem_l2282_228233


namespace NUMINAMATH_CALUDE_current_speed_l2282_228219

/-- The speed of the current in a river, given the rowing speed in still water and the time taken to cover a certain distance downstream. -/
theorem current_speed (still_water_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  still_water_speed = 22 →
  downstream_distance = 80 →
  downstream_time = 11.519078473722104 →
  ∃ current_speed : ℝ, 
    (current_speed * 1000 / 3600 + still_water_speed * 1000 / 3600) * downstream_time = downstream_distance ∧ 
    abs (current_speed - 2.9988) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l2282_228219


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2282_228200

theorem point_on_x_axis (m : ℝ) : (3, m) ∈ {p : ℝ × ℝ | p.2 = 0} → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2282_228200


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l2282_228225

/-- The range of m for which a line y = kx + 1 and an ellipse x²/5 + y²/m = 1 always intersect -/
theorem line_ellipse_intersection_condition (k : ℝ) :
  ∃ (m : ℝ), (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1 → 
    (m ≥ 1 ∧ m ≠ 5)) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_condition_l2282_228225


namespace NUMINAMATH_CALUDE_fraction_simplification_l2282_228248

theorem fraction_simplification (x y : ℝ) (h : x ≠ 3*y ∧ x ≠ -3*y) : 
  (2*x)/(x^2 - 9*y^2) - 1/(x - 3*y) = 1/(x + 3*y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2282_228248


namespace NUMINAMATH_CALUDE_two_trains_meeting_time_l2282_228269

/-- Two trains problem -/
theorem two_trains_meeting_time 
  (distance : ℝ) 
  (fast_speed slow_speed : ℝ) 
  (head_start : ℝ) 
  (h_distance : distance = 270) 
  (h_fast_speed : fast_speed = 120) 
  (h_slow_speed : slow_speed = 75) 
  (h_head_start : head_start = 1) :
  ∃ x : ℝ, slow_speed * head_start + (fast_speed + slow_speed) * x = distance :=
by sorry

end NUMINAMATH_CALUDE_two_trains_meeting_time_l2282_228269


namespace NUMINAMATH_CALUDE_carpenters_completion_time_l2282_228237

def carpenter1_rate : ℚ := 1 / 5
def carpenter2_rate : ℚ := 1 / 5
def combined_rate : ℚ := carpenter1_rate + carpenter2_rate
def job_completion : ℚ := 1

theorem carpenters_completion_time :
  ∃ (time : ℚ), time * combined_rate = job_completion ∧ time = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carpenters_completion_time_l2282_228237


namespace NUMINAMATH_CALUDE_rectangle_enumeration_l2282_228285

/-- Represents a rectangle in the Cartesian plane with sides parallel to the axes. -/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- Defines when one rectangle is below another. -/
def is_below (r1 r2 : Rectangle) : Prop :=
  r1.y_max < r2.y_min

/-- Defines when one rectangle is to the right of another. -/
def is_right_of (r1 r2 : Rectangle) : Prop :=
  r1.x_min > r2.x_max

/-- Defines when two rectangles are disjoint. -/
def are_disjoint (r1 r2 : Rectangle) : Prop :=
  r1.x_max ≤ r2.x_min ∨ r2.x_max ≤ r1.x_min ∨
  r1.y_max ≤ r2.y_min ∨ r2.y_max ≤ r1.y_min

/-- The main theorem stating that any finite set of pairwise disjoint rectangles
    can be enumerated such that each rectangle is to the right of or below all
    subsequent rectangles in the enumeration. -/
theorem rectangle_enumeration (n : ℕ) (rectangles : Fin n → Rectangle)
    (h_disjoint : ∀ i j : Fin n, i ≠ j → are_disjoint (rectangles i) (rectangles j)) :
    ∃ σ : Equiv.Perm (Fin n),
      ∀ i j : Fin n, i < j →
        is_right_of (rectangles (σ i)) (rectangles (σ j)) ∨
        is_below (rectangles (σ i)) (rectangles (σ j)) :=
  sorry

end NUMINAMATH_CALUDE_rectangle_enumeration_l2282_228285


namespace NUMINAMATH_CALUDE_max_distance_for_given_car_l2282_228259

/-- Represents a car with front and rear tires that can be switched --/
structure Car where
  frontTireLife : ℕ
  rearTireLife : ℕ

/-- Calculates the maximum distance a car can travel by switching tires once --/
def maxDistanceWithSwitch (car : Car) : ℕ :=
  let switchPoint := min car.frontTireLife car.rearTireLife / 2
  switchPoint + (car.frontTireLife - switchPoint) + (car.rearTireLife - switchPoint)

/-- Theorem stating the maximum distance for the given car specifications --/
theorem max_distance_for_given_car :
  let car := { frontTireLife := 24000, rearTireLife := 36000 : Car }
  maxDistanceWithSwitch car = 48000 := by
  sorry

#eval maxDistanceWithSwitch { frontTireLife := 24000, rearTireLife := 36000 }

end NUMINAMATH_CALUDE_max_distance_for_given_car_l2282_228259


namespace NUMINAMATH_CALUDE_remaining_budget_for_public_spaces_l2282_228274

/-- Proof of remaining budget for public spaces -/
theorem remaining_budget_for_public_spaces 
  (total_budget : ℝ) 
  (education_budget : ℝ) 
  (h1 : total_budget = 32000000)
  (h2 : education_budget = 12000000) :
  total_budget - (total_budget / 2 + education_budget) = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_budget_for_public_spaces_l2282_228274


namespace NUMINAMATH_CALUDE_f_def_f_5_eq_0_l2282_228220

def f (x : ℝ) : ℝ := sorry

theorem f_def (x : ℝ) : f (2 * x + 1) = x^2 - 2*x := sorry

theorem f_5_eq_0 : f 5 = 0 := by sorry

end NUMINAMATH_CALUDE_f_def_f_5_eq_0_l2282_228220


namespace NUMINAMATH_CALUDE_black_tiles_imply_total_tiles_l2282_228250

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

end NUMINAMATH_CALUDE_black_tiles_imply_total_tiles_l2282_228250


namespace NUMINAMATH_CALUDE_fraction_difference_l2282_228212

theorem fraction_difference : (7 : ℚ) / 12 - (3 : ℚ) / 8 = (5 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l2282_228212
