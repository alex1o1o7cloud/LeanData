import Mathlib

namespace july_capsule_intake_l4128_412844

/-- Represents the number of capsules taken in a month -/
def capsulesTaken (totalDays : ℕ) (forgottenDays : ℕ) : ℕ :=
  totalDays - forgottenDays

/-- Theorem: Given a 31-day month where a person forgets to take capsules on 3 days,
    the total number of capsules taken is 28 -/
theorem july_capsule_intake :
  capsulesTaken 31 3 = 28 := by
  sorry

end july_capsule_intake_l4128_412844


namespace sticker_trade_result_l4128_412886

/-- Calculates the final number of stickers after a given number of trades -/
def final_sticker_count (initial_count : ℕ) (num_trades : ℕ) : ℕ :=
  initial_count + num_trades * 4

/-- Theorem stating that after 50 trades, starting with 1 sticker, 
    the final count is 201 stickers -/
theorem sticker_trade_result : final_sticker_count 1 50 = 201 := by
  sorry

end sticker_trade_result_l4128_412886


namespace nalani_puppy_price_l4128_412826

/-- The price per puppy in Nalani's sale --/
def price_per_puppy (num_dogs : ℕ) (puppies_per_dog : ℕ) (fraction_sold : ℚ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (fraction_sold * (num_dogs * puppies_per_dog))

/-- Theorem stating the price per puppy in Nalani's specific case --/
theorem nalani_puppy_price :
  price_per_puppy 2 10 (3/4) 3000 = 200 := by
  sorry

#eval price_per_puppy 2 10 (3/4) 3000

end nalani_puppy_price_l4128_412826


namespace monotone_increasing_condition_l4128_412828

/-- The function f(x) = kx - ln x is monotonically increasing on (1/2, +∞) if and only if k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (fun x => k * x - Real.log x)) ↔ k ≥ 2 := by
  sorry

end monotone_increasing_condition_l4128_412828


namespace skew_lines_sufficient_not_necessary_for_non_intersection_l4128_412875

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define lines in the space
def Line (V : Type*) [NormedAddCommGroup V] := V → Set V

-- Define the property of being skew
def are_skew (l₁ l₂ : Line V) : Prop := sorry

-- Define the property of not intersecting
def do_not_intersect (l₁ l₂ : Line V) : Prop := sorry

-- Theorem statement
theorem skew_lines_sufficient_not_necessary_for_non_intersection :
  (∀ l₁ l₂ : Line V, are_skew l₁ l₂ → do_not_intersect l₁ l₂) ∧
  (∃ l₁ l₂ : Line V, do_not_intersect l₁ l₂ ∧ ¬are_skew l₁ l₂) :=
sorry

end skew_lines_sufficient_not_necessary_for_non_intersection_l4128_412875


namespace particular_number_problem_l4128_412845

theorem particular_number_problem (x : ℚ) (h : (x + 10) / 5 = 4) : 3 * x - 18 = 12 := by
  sorry

end particular_number_problem_l4128_412845


namespace repeating_decimal_sum_l4128_412835

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = (14 / 37 : ℚ) := by
  sorry

end repeating_decimal_sum_l4128_412835


namespace pump_calculations_l4128_412871

/-- Ultraflow pump rate in gallons per hour -/
def ultraflow_rate : ℚ := 560

/-- MiniFlow pump rate in gallons per hour -/
def miniflow_rate : ℚ := 220

/-- Convert minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ := minutes / 60

/-- Calculate gallons pumped given rate and time -/
def gallons_pumped (rate : ℚ) (time : ℚ) : ℚ := rate * time

theorem pump_calculations :
  (gallons_pumped ultraflow_rate (minutes_to_hours 75) = 700) ∧
  (gallons_pumped ultraflow_rate (minutes_to_hours 50) + gallons_pumped miniflow_rate (minutes_to_hours 50) = 883) := by
  sorry

end pump_calculations_l4128_412871


namespace arithmetic_calculations_l4128_412860

theorem arithmetic_calculations :
  (7 + (-14) - (-9) - 12 = -10) ∧
  (25 / (-5) * (1 / 5) / (3 / 4) = -4 / 3) := by
sorry

end arithmetic_calculations_l4128_412860


namespace hiking_equipment_cost_l4128_412825

def hoodie_cost : ℝ := 80
def flashlight_cost : ℝ := 0.2 * hoodie_cost
def boots_original_cost : ℝ := 110
def boots_discount : ℝ := 0.1
def water_filter_cost : ℝ := 65
def water_filter_discount : ℝ := 0.25
def camping_mat_cost : ℝ := 45
def camping_mat_discount : ℝ := 0.15
def backpack_cost : ℝ := 105

def clothing_tax_rate : ℝ := 0.05
def electronics_tax_rate : ℝ := 0.1
def other_equipment_tax_rate : ℝ := 0.08

def total_cost : ℝ :=
  (hoodie_cost * (1 + clothing_tax_rate)) +
  (flashlight_cost * (1 + electronics_tax_rate)) +
  (boots_original_cost * (1 - boots_discount) * (1 + clothing_tax_rate)) +
  (water_filter_cost * (1 - water_filter_discount) * (1 + other_equipment_tax_rate)) +
  (camping_mat_cost * (1 - camping_mat_discount) * (1 + other_equipment_tax_rate)) +
  (backpack_cost * (1 + other_equipment_tax_rate))

theorem hiking_equipment_cost : total_cost = 413.91 := by
  sorry

end hiking_equipment_cost_l4128_412825


namespace word_sum_proof_l4128_412890

theorem word_sum_proof :
  ∀ A B C : ℕ,
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 →
    A ≠ B ∧ B ≠ C ∧ A ≠ C →
    A < 10 ∧ B < 10 ∧ C < 10 →
    100 * A + 10 * B + C +
    100 * B + 10 * C + A +
    100 * C + 10 * A + B =
    1000 * A + 100 * B + 10 * B + C →
    A = 1 ∧ B = 9 ∧ C = 8 := by
sorry

end word_sum_proof_l4128_412890


namespace intersection_with_complement_l4128_412801

def U : Finset ℕ := {0,1,2,3,4,5,6}
def A : Finset ℕ := {0,1,3,5}
def B : Finset ℕ := {1,2,4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0,3,5} := by sorry

end intersection_with_complement_l4128_412801


namespace quadratic_sum_equality_l4128_412896

/-- A quadratic function satisfying specific conditions -/
def P : ℝ → ℝ := fun x ↦ 6 * x^2 - 3 * x + 7

/-- The theorem statement -/
theorem quadratic_sum_equality (a b c : ℤ) :
  P 0 = 7 ∧ P 1 = 10 ∧ P 2 = 25 ∧
  (∀ x : ℝ, 0 < x → x < 1 →
    (∑' n, P n * x^n) = (a * x^2 + b * x + c) / (1 - x)^3) →
  (a, b, c) = (16, -11, 7) := by sorry

end quadratic_sum_equality_l4128_412896


namespace perpendicular_line_equation_l4128_412843

/-- Given a line L1 with equation x + y - 5 = 0 and a point P (2, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x - y - 3 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | x + y - 5 = 0} →
  P = (2, -1) →
  ∃ L2 : Set (ℝ × ℝ),
    (P ∈ L2) ∧
    (∀ (A B : ℝ × ℝ), A ∈ L1 → B ∈ L1 → A ≠ B →
      ∀ (C D : ℝ × ℝ), C ∈ L2 → D ∈ L2 → C ≠ D →
        ((A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0)) ∧
    L2 = {(x, y) | x - y - 3 = 0} :=
by sorry

end perpendicular_line_equation_l4128_412843


namespace time_addition_theorem_l4128_412880

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Adds a duration to a given time, wrapping around a 12-hour clock -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : ℕ) : Time :=
  sorry

/-- Converts 24-hour time to 12-hour time -/
def to12HourFormat (time : Time) : Time :=
  sorry

/-- Calculates the sum of hours, minutes, and seconds digits -/
def sumTimeDigits (time : Time) : ℕ :=
  sorry

theorem time_addition_theorem :
  let startTime := Time.mk 15 15 20
  let finalTime := addTime startTime 198 47 36
  let finalTime12Hour := to12HourFormat finalTime
  finalTime12Hour = Time.mk 10 2 56 ∧ sumTimeDigits finalTime12Hour = 68 := by
  sorry

end time_addition_theorem_l4128_412880


namespace max_product_sum_300_l4128_412878

theorem max_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 :=
by sorry

end max_product_sum_300_l4128_412878


namespace trig_expression_equality_l4128_412807

theorem trig_expression_equality : 
  2 * Real.sin (30 * π / 180) - Real.tan (45 * π / 180) - Real.sqrt ((1 - Real.tan (60 * π / 180))^2) = Real.sqrt 3 - 1 := by
sorry

end trig_expression_equality_l4128_412807


namespace sandwich_fraction_l4128_412882

theorem sandwich_fraction (total : ℚ) (ticket : ℚ) (book : ℚ) (leftover : ℚ) 
  (h1 : total = 180)
  (h2 : ticket = 1/6)
  (h3 : book = 1/2)
  (h4 : leftover = 24) :
  ∃ (sandwich : ℚ), 
    sandwich * total + ticket * total + book * total = total - leftover ∧ 
    sandwich = 1/5 := by
  sorry

end sandwich_fraction_l4128_412882


namespace geometric_arithmetic_interleaving_l4128_412815

theorem geometric_arithmetic_interleaving (n : ℕ) (h : n > 3) :
  ∃ (x y : ℕ → ℕ),
    (∀ i, i < n → x i > 0) ∧
    (∀ i, i < n → y i > 0) ∧
    (∃ r : ℚ, r > 1 ∧ ∀ i, i < n - 1 → x (i + 1) = (x i : ℚ) * r) ∧
    (∃ d : ℚ, d > 0 ∧ ∀ i, i < n - 1 → y (i + 1) = y i + d) ∧
    (∀ i, i < n - 1 → x i < y i ∧ y i < x (i + 1)) ∧
    x (n - 1) < y (n - 1) :=
by sorry

end geometric_arithmetic_interleaving_l4128_412815


namespace problem_solution_l4128_412809

theorem problem_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3) → d = 13/4 := by
  sorry

end problem_solution_l4128_412809


namespace root_equation_problem_l4128_412811

theorem root_equation_problem (b c x₁ x₂ : ℝ) (y : ℝ) : 
  x₁ ≠ x₂ →
  (x₁^2 + 5*b*x₁ + c = 0) →
  (x₂^2 + 5*b*x₂ + c = 0) →
  (y^2 + 2*x₁*y + 2*x₂ = 0) →
  (y^2 + 2*x₂*y + 2*x₁ = 0) →
  b = 1/10 := by
sorry

end root_equation_problem_l4128_412811


namespace remaining_amount_correct_l4128_412881

/-- Calculates the remaining amount in Will's original currency after shopping --/
def remaining_amount (initial_amount conversion_fee exchange_rate sweater_price tshirt_price
                      shoes_price hat_price socks_price shoe_refund_rate discount_rate
                      sales_tax_rate : ℚ) : ℚ :=
  let amount_after_fee := initial_amount - conversion_fee
  let local_currency_amount := amount_after_fee * exchange_rate
  let total_cost := sweater_price + tshirt_price + shoes_price + hat_price + socks_price
  let refund := shoes_price * shoe_refund_rate
  let cost_after_refund := total_cost - refund
  let discountable_items := sweater_price + tshirt_price + hat_price + socks_price
  let discount := discountable_items * discount_rate
  let cost_after_discount := cost_after_refund - discount
  let sales_tax := cost_after_discount * sales_tax_rate
  let final_cost := cost_after_discount + sales_tax
  let remaining_local := local_currency_amount - final_cost
  remaining_local / exchange_rate

/-- Theorem stating that the remaining amount is correct --/
theorem remaining_amount_correct :
  remaining_amount 74 2 (3/2) (27/2) (33/2) 45 (15/2) 6 (17/20) (1/10) (1/20) = (3987/100) := by
  sorry

end remaining_amount_correct_l4128_412881


namespace system_solution_l4128_412833

theorem system_solution (x y : ℚ) : 
  (x + y = x^2 + 2*x*y + y^2 ∧ x - y = x^2 - 2*x*y + y^2) ↔ 
  ((x = 1/2 ∧ y = -1/2) ∨ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 1/2 ∧ y = 1/2)) :=
by sorry

end system_solution_l4128_412833


namespace tangent_point_coordinates_l4128_412808

/-- Given a curve y = x^2 + a ln(x) where a > 0, if the minimum value of the slope
    of the tangent line at any point on the curve is 4, then the coordinates of the
    point of tangency at this minimum slope are (1, 1). -/
theorem tangent_point_coordinates (a : ℝ) (h1 : a > 0) :
  (∀ x > 0, 2 * x + a / x ≥ 4) ∧ (∃ x > 0, 2 * x + a / x = 4) →
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ y = x^2 + a * Real.log x ∧ 2 * x + a / x = 4 := by
  sorry

end tangent_point_coordinates_l4128_412808


namespace y_increases_as_x_decreases_l4128_412889

theorem y_increases_as_x_decreases (α : Real) (h_acute : 0 < α ∧ α < π / 2) :
  let f : Real → Real := λ x ↦ (Real.sin α - 1) * x - 6
  ∀ x₁ x₂ : Real, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end y_increases_as_x_decreases_l4128_412889


namespace rhombus_perimeter_l4128_412865

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 8 * Real.sqrt 41 := by
  sorry

end rhombus_perimeter_l4128_412865


namespace basketball_court_fits_l4128_412827

theorem basketball_court_fits (total_area : ℝ) (court_area : ℝ) (length_width_ratio : ℝ) (space_width : ℝ) :
  total_area = 1100 ∧ 
  court_area = 540 ∧ 
  length_width_ratio = 5/3 ∧
  space_width = 1 →
  ∃ (width : ℝ), 
    width > 0 ∧
    length_width_ratio * width * width = court_area ∧
    (length_width_ratio * width + 2 * space_width) * (width + 2 * space_width) ≤ total_area :=
by sorry

#check basketball_court_fits

end basketball_court_fits_l4128_412827


namespace solution_set_when_a_is_2_solution_set_for_any_a_l4128_412821

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for the second part of the problem
theorem solution_set_for_any_a (a : ℝ) :
  {x : ℝ | f a x > 0} = 
    if a > -1 then
      {x : ℝ | x < -1 ∨ x > a}
    else if a = -1 then
      {x : ℝ | x < -1 ∨ x > -1}
    else
      {x : ℝ | x < a ∨ x > -1} := by sorry

end solution_set_when_a_is_2_solution_set_for_any_a_l4128_412821


namespace peter_wants_17_dogs_l4128_412824

/-- The number of dogs Peter wants to have -/
def PetersDogs (samGS : ℕ) (samFB : ℕ) (peterGSFactor : ℕ) (peterFBFactor : ℕ) : ℕ :=
  peterGSFactor * samGS + peterFBFactor * samFB

/-- Theorem stating the number of dogs Peter wants to have -/
theorem peter_wants_17_dogs :
  PetersDogs 3 4 3 2 = 17 := by
  sorry

end peter_wants_17_dogs_l4128_412824


namespace different_terminal_sides_not_equal_l4128_412855

-- Define an angle
def Angle : Type := ℝ

-- Define the initial side of an angle
def initial_side (a : Angle) : ℝ × ℝ := sorry

-- Define the terminal side of an angle
def terminal_side (a : Angle) : ℝ × ℝ := sorry

-- Define equality of angles
def angle_eq (a b : Angle) : Prop := 
  initial_side a = initial_side b ∧ terminal_side a = terminal_side b

-- Theorem statement
theorem different_terminal_sides_not_equal (a b : Angle) :
  initial_side a = initial_side b → 
  terminal_side a ≠ terminal_side b → 
  ¬(angle_eq a b) := by sorry

end different_terminal_sides_not_equal_l4128_412855


namespace snow_probability_l4128_412898

theorem snow_probability (p1 p2 p3 : ℚ) 
  (h1 : p1 = 1/2) 
  (h2 : p2 = 3/4) 
  (h3 : p3 = 2/3) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 23/24 := by
  sorry

end snow_probability_l4128_412898


namespace variance_of_specific_random_variable_l4128_412819

/-- A random variable that takes values 0, 1, and 2 -/
structure RandomVariable where
  prob0 : ℝ
  prob1 : ℝ
  prob2 : ℝ
  sum_to_one : prob0 + prob1 + prob2 = 1
  nonnegative : prob0 ≥ 0 ∧ prob1 ≥ 0 ∧ prob2 ≥ 0

/-- The expectation of a random variable -/
def expectation (ξ : RandomVariable) : ℝ :=
  0 * ξ.prob0 + 1 * ξ.prob1 + 2 * ξ.prob2

/-- The variance of a random variable -/
def variance (ξ : RandomVariable) : ℝ :=
  (0 - expectation ξ)^2 * ξ.prob0 +
  (1 - expectation ξ)^2 * ξ.prob1 +
  (2 - expectation ξ)^2 * ξ.prob2

/-- Theorem: If P(ξ=0) = 1/5 and E(ξ) = 1, then D(ξ) = 2/5 -/
theorem variance_of_specific_random_variable :
  ∀ (ξ : RandomVariable),
    ξ.prob0 = 1/5 →
    expectation ξ = 1 →
    variance ξ = 2/5 := by
  sorry

end variance_of_specific_random_variable_l4128_412819


namespace sqrt_fraction_equality_l4128_412892

theorem sqrt_fraction_equality : 
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 36)) = (17 * Real.sqrt 61) / 61 := by
  sorry

end sqrt_fraction_equality_l4128_412892


namespace parabola_line_intersection_sum_l4128_412820

/-- Parabola P with equation y = x^2 -/
def P : ℝ → ℝ := fun x ↦ x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := fun x ↦ m * (x - Q.1) + Q.2

/-- The line does not intersect the parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

/-- Theorem stating that r + s = 40 -/
theorem parabola_line_intersection_sum :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 :=
sorry

end parabola_line_intersection_sum_l4128_412820


namespace sum_of_coordinates_is_14_l4128_412818

/-- Given two points A and B in a 2D plane, where:
  - A is at (0, 0)
  - B is on the line y = 6
  - The slope of segment AB is 3/4
  This theorem proves that the sum of the x- and y-coordinates of point B is 14. -/
theorem sum_of_coordinates_is_14 (B : ℝ × ℝ) : 
  B.2 = 6 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 4 →
  B.1 + B.2 = 14 := by
sorry

end sum_of_coordinates_is_14_l4128_412818


namespace min_value_of_y_l4128_412814

-- Define a function that calculates the sum of squares of 11 consecutive integers
def sumOfSquares (x : ℤ) : ℤ := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2

-- Theorem statement
theorem min_value_of_y (y : ℤ) : (∃ x : ℤ, y^2 = sumOfSquares x) → y ≥ -11 ∧ (∃ x : ℤ, (-11)^2 = sumOfSquares x) := by
  sorry

end min_value_of_y_l4128_412814


namespace exists_x_y_for_a_l4128_412874

def a : ℕ → ℤ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * a (n + 1) - a n

def b : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | (n + 2) => 2 * b (n + 1) + b n

theorem exists_x_y_for_a : ∃ (x y : ℕ → ℕ), ∀ n, 
  (y n)^2 + 7 = (x n - y n) * a n :=
sorry

end exists_x_y_for_a_l4128_412874


namespace trig_identity_l4128_412877

theorem trig_identity (x y : ℝ) : 
  Real.sin (x - y) * Real.sin x + Real.cos (x - y) * Real.cos x = Real.cos y := by
  sorry

end trig_identity_l4128_412877


namespace largest_integer_with_remainder_l4128_412834

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n :=
by sorry

end largest_integer_with_remainder_l4128_412834


namespace sum_of_roots_cubic_sum_of_roots_specific_cubic_l4128_412859

theorem sum_of_roots_cubic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (x + y + z = -b / a) := by sorry

theorem sum_of_roots_specific_cubic :
  let f : ℝ → ℝ := λ x => 3 * x^3 + 7 * x^2 - 12 * x - 4
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (x + y + z = -7 / 3) := by sorry

end sum_of_roots_cubic_sum_of_roots_specific_cubic_l4128_412859


namespace equation_represents_intersecting_lines_l4128_412897

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 - y^2 = 0

-- Theorem statement
theorem equation_represents_intersecting_lines :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f x = x ∧ g x = -x) ∧
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end equation_represents_intersecting_lines_l4128_412897


namespace equation_has_three_real_solutions_l4128_412850

-- Define the equation
def equation (x : ℝ) : Prop :=
  (18 * x - 2) ^ (1/3) + (14 * x - 4) ^ (1/3) = 5 * (2 * x + 4) ^ (1/3)

-- State the theorem
theorem equation_has_three_real_solutions :
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    equation x₁ ∧ equation x₂ ∧ equation x₃ :=
sorry

end equation_has_three_real_solutions_l4128_412850


namespace side_length_S2_correct_l4128_412883

/-- The side length of square S2 in a specific rectangular arrangement -/
def side_length_S2 : ℕ := 650

/-- The width of the overall rectangle -/
def total_width : ℕ := 3400

/-- The height of the overall rectangle -/
def total_height : ℕ := 2100

/-- Theorem stating that the side length of S2 is correct given the constraints -/
theorem side_length_S2_correct :
  ∃ (r : ℕ),
    (2 * r + side_length_S2 = total_height) ∧
    (2 * r + 3 * side_length_S2 = total_width) := by
  sorry

end side_length_S2_correct_l4128_412883


namespace remainder_problem_l4128_412816

theorem remainder_problem (N : ℕ) : 
  (∃ R : ℕ, N = 44 * 432 + R ∧ R < 44) → 
  (∃ Q : ℕ, N = 31 * Q + 5) → 
  (∃ R : ℕ, N = 44 * 432 + R ∧ R = 2) := by
sorry

end remainder_problem_l4128_412816


namespace sets_theorem_l4128_412868

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem sets_theorem (a : ℝ) :
  (A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 10}) ∧
  ((Set.compl A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10}) ∧
  ((A ∩ C a).Nonempty ↔ a > -3) :=
by sorry

end sets_theorem_l4128_412868


namespace arithmetic_sequence_properties_l4128_412837

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 4 + seq.a 14 = 2 → S seq 17 = 17) ∧
  (seq.a 11 = 10 → S seq 21 = 210) ∧
  (S seq 11 = 55 → seq.a 6 = 5) ∧
  (S seq 8 = 100 ∧ S seq 16 = 392 → S seq 24 = 876) := by
  sorry

end arithmetic_sequence_properties_l4128_412837


namespace find_m_l4128_412857

def A (m : ℕ) : Set ℕ := {1, 2, m}
def B : Set ℕ := {4, 7, 13}

def f (x : ℕ) : ℕ := 3 * x + 1

theorem find_m : ∃ m : ℕ, 
  (∀ x ∈ A m, f x ∈ B) ∧ 
  m = 4 := by sorry

end find_m_l4128_412857


namespace fraction_equality_l4128_412800

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 := by
sorry

end fraction_equality_l4128_412800


namespace min_value_theorem_l4128_412891

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_constraint : a + b + c = 5) :
  (9 / a) + (16 / b) + (25 / c^2) ≥ 50 := by
  sorry

end min_value_theorem_l4128_412891


namespace janinas_daily_rent_l4128_412830

/-- Janina's pancake stand financial model -/
def pancake_stand_model (daily_supply_cost : ℝ) (pancake_price : ℝ) (breakeven_pancakes : ℕ) : ℝ :=
  pancake_price * (breakeven_pancakes : ℝ) - daily_supply_cost

/-- Theorem: Janina's daily rent is $30 -/
theorem janinas_daily_rent :
  pancake_stand_model 12 2 21 = 30 := by
  sorry

end janinas_daily_rent_l4128_412830


namespace three_numbers_sum_square_counterexample_l4128_412846

theorem three_numbers_sum_square_counterexample :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b^2 + c^2 = b + a^2 + c^2) ∧
    (b + a^2 + c^2 = c + a^2 + b^2) ∧
    (a ≠ b ∨ b ≠ c ∨ a ≠ c) :=
by sorry

end three_numbers_sum_square_counterexample_l4128_412846


namespace money_distribution_l4128_412887

theorem money_distribution (a b c : ℤ) : 
  a + b + c = 900 → a + c = 400 → b + c = 750 → c = 250 := by
  sorry

end money_distribution_l4128_412887


namespace duodecimal_reversal_difference_divisibility_l4128_412866

/-- Represents a duodecimal digit (0 to 11) -/
def DuodecimalDigit := {n : ℕ // n ≤ 11}

/-- Converts a two-digit duodecimal number to its decimal representation -/
def toDecimal (a b : DuodecimalDigit) : ℤ :=
  12 * a.val + b.val

theorem duodecimal_reversal_difference_divisibility
  (a b : DuodecimalDigit)
  (h : a ≠ b) :
  ∃ k : ℤ, toDecimal a b - toDecimal b a = 11 * k := by
  sorry

end duodecimal_reversal_difference_divisibility_l4128_412866


namespace fractional_equation_solution_l4128_412864

theorem fractional_equation_solution :
  ∃ x : ℚ, (x + 1) / (4 * (x - 1)) = 2 / (3 * x - 3) - 1 ↔ x = 17 / 15 :=
by sorry

end fractional_equation_solution_l4128_412864


namespace carpet_cost_calculation_l4128_412861

/-- Calculate the cost of a carpet given its dimensions and the price per square meter -/
def calculate_carpet_cost (length width price_per_sqm : ℝ) : ℝ :=
  length * width * price_per_sqm

/-- The problem statement -/
theorem carpet_cost_calculation :
  let first_carpet_breadth : ℝ := 6
  let first_carpet_length : ℝ := 1.44 * first_carpet_breadth
  let second_carpet_length : ℝ := first_carpet_length * 1.427
  let second_carpet_breadth : ℝ := first_carpet_breadth * 1.275
  let price_per_sqm : ℝ := 46.35
  
  abs (calculate_carpet_cost second_carpet_length second_carpet_breadth price_per_sqm - 4371.78) < 0.01 := by
  sorry


end carpet_cost_calculation_l4128_412861


namespace product_of_first_three_terms_l4128_412858

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem product_of_first_three_terms 
  (a₁ : ℝ) -- first term
  (d : ℝ) -- common difference
  (h1 : arithmetic_sequence a₁ d 7 = 20) -- seventh term is 20
  (h2 : d = 2) -- common difference is 2
  : a₁ * (a₁ + d) * (a₁ + 2 * d) = 960 := by
  sorry

end product_of_first_three_terms_l4128_412858


namespace intersection_of_A_and_B_l4128_412888

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 < 4}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l4128_412888


namespace equation_solution_l4128_412842

theorem equation_solution (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 9) 
  (eq2 : x + 3 * y = 8) : 
  3 * x^2 + 7 * x * y + 3 * y^2 = 145 := by
sorry

end equation_solution_l4128_412842


namespace geometric_sequence_fifth_term_l4128_412823

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_roots : a 3 * a 7 = 4 ∧ a 3 + a 7 = 5) :
  a 5 = 2 :=
sorry

end geometric_sequence_fifth_term_l4128_412823


namespace ray_nickels_left_l4128_412870

def nickel_value : ℕ := 5
def initial_cents : ℕ := 95
def cents_to_peter : ℕ := 25

theorem ray_nickels_left : 
  let initial_nickels := initial_cents / nickel_value
  let nickels_to_peter := cents_to_peter / nickel_value
  let cents_to_randi := 2 * cents_to_peter
  let nickels_to_randi := cents_to_randi / nickel_value
  initial_nickels - nickels_to_peter - nickels_to_randi = 4 := by
  sorry

end ray_nickels_left_l4128_412870


namespace race_catchup_time_l4128_412840

/-- Proves that Nicky runs for 48 seconds before Cristina catches up to him in a 500-meter race --/
theorem race_catchup_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 500)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  let catchup_time := head_start + (head_start * nicky_speed) / (cristina_speed - nicky_speed)
  catchup_time = 48 := by
  sorry

end race_catchup_time_l4128_412840


namespace sum_and_product_positive_iff_both_positive_l4128_412854

theorem sum_and_product_positive_iff_both_positive (a b : ℝ) :
  (a + b > 0 ∧ a * b > 0) ↔ (a > 0 ∧ b > 0) := by
  sorry

end sum_and_product_positive_iff_both_positive_l4128_412854


namespace slope_product_is_negative_one_l4128_412853

/-- Parabola C: y^2 = 2px (p > 0) passing through (2, 2) -/
def parabola_C (p : ℝ) : Set (ℝ × ℝ) :=
  {point | point.2^2 = 2 * p * point.1 ∧ p > 0}

/-- Point Q on the parabola -/
def point_Q : ℝ × ℝ := (2, 2)

/-- Point M through which the intersecting line passes -/
def point_M : ℝ × ℝ := (2, 0)

/-- Origin O -/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem: The product of slopes of OA and OB is -1 -/
theorem slope_product_is_negative_one
  (p : ℝ)
  (h_p : p > 0)
  (h_Q : point_Q ∈ parabola_C p)
  (A B : ℝ × ℝ)
  (h_A : A ∈ parabola_C p)
  (h_B : B ∈ parabola_C p)
  (h_line : ∃ (m : ℝ), A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2)
  (k1 : ℝ) (h_k1 : k1 = (A.2 - origin.2) / (A.1 - origin.1))
  (k2 : ℝ) (h_k2 : k2 = (B.2 - origin.2) / (B.1 - origin.1)) :
  k1 * k2 = -1 := by sorry

end slope_product_is_negative_one_l4128_412853


namespace missing_digit_is_seven_l4128_412848

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def insert_digit (d : ℕ) : ℕ := 351000 + d * 100 + 92

theorem missing_digit_is_seven :
  ∃! d : ℕ, d < 10 ∧ is_divisible_by_9 (insert_digit d) :=
by
  sorry

#check missing_digit_is_seven

end missing_digit_is_seven_l4128_412848


namespace fraction_domain_l4128_412852

theorem fraction_domain (x : ℝ) : 
  (∃ y : ℝ, y = 5 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end fraction_domain_l4128_412852


namespace jason_seashells_theorem_l4128_412863

/-- Calculates the number of seashells Jason gave to Tim -/
def seashells_given_to_tim (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Proves that the number of seashells Jason gave to Tim is correct -/
theorem jason_seashells_theorem (initial_seashells current_seashells : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : current_seashells = 36)
  (h3 : initial_seashells ≥ current_seashells) :
  seashells_given_to_tim initial_seashells current_seashells = 13 := by
  sorry

#eval seashells_given_to_tim 49 36

end jason_seashells_theorem_l4128_412863


namespace product_remainder_l4128_412869

theorem product_remainder (a b m : ℕ) (ha : a = 103) (hb : b = 107) (hm : m = 13) :
  (a * b) % m = 10 := by
  sorry

end product_remainder_l4128_412869


namespace max_guaranteed_score_is_four_l4128_412894

/-- Represents a player in the game -/
inductive Player : Type
| B : Player
| R : Player

/-- Represents a color of a square -/
inductive Color : Type
| White : Color
| Blue : Color
| Red : Color

/-- Represents a square on the infinite grid -/
structure Square :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the game state -/
structure GameState :=
  (grid : Square → Color)
  (currentPlayer : Player)

/-- Represents a simple polygon on the grid -/
structure SimplePolygon :=
  (squares : Set Square)

/-- The score of player B is the area of the largest simple polygon of blue squares -/
def score (state : GameState) : ℕ :=
  sorry

/-- A strategy for player B -/
def Strategy : Type :=
  GameState → Square

/-- The maximum guaranteed score for player B -/
def maxGuaranteedScore : ℕ :=
  sorry

/-- The main theorem stating that the maximum guaranteed score for B is 4 -/
theorem max_guaranteed_score_is_four :
  maxGuaranteedScore = 4 :=
sorry

end max_guaranteed_score_is_four_l4128_412894


namespace root_of_polynomial_l4128_412867

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 16*x^2 + 4

-- State the theorem
theorem root_of_polynomial :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 16*x^2 + 4) ∧
  -- The polynomial has degree 4
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- √3 + √5 is a root of the polynomial
  p (Real.sqrt 3 + Real.sqrt 5) = 0 :=
by
  sorry

end root_of_polynomial_l4128_412867


namespace inequality_solution_set_l4128_412879

theorem inequality_solution_set :
  ∀ x : ℝ, (6 - x - 2 * x^2 < 0) ↔ (x > 3/2 ∨ x < -2) :=
by sorry

end inequality_solution_set_l4128_412879


namespace inverse_prop_problem_l4128_412893

/-- Two numbers are inversely proportional if their product is constant -/
def inverse_proportional (a b : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_prop_problem (a b : ℝ → ℝ) 
  (h1 : inverse_proportional a b)
  (h2 : ∃ x : ℝ, a x + b x = 60 ∧ a x = 3 * b x) :
  b (-12) = -56.25 := by
  sorry


end inverse_prop_problem_l4128_412893


namespace election_votes_l4128_412851

theorem election_votes (total_members : ℕ) (winner_percentage : ℚ) (winner_total_percentage : ℚ) :
  total_members = 1600 →
  winner_percentage = 60 / 100 →
  winner_total_percentage = 19.6875 / 100 →
  (↑total_members * winner_total_percentage : ℚ) / winner_percentage = 525 := by
  sorry

end election_votes_l4128_412851


namespace theater_line_up_ways_l4128_412805

theorem theater_line_up_ways : 
  let number_of_windows : ℕ := 2
  let number_of_people : ℕ := 6
  number_of_windows ^ number_of_people * Nat.factorial number_of_people = 46080 :=
by sorry

end theater_line_up_ways_l4128_412805


namespace radar_arrangements_l4128_412832

def word_length : ℕ := 5
def r_count : ℕ := 2
def a_count : ℕ := 2

theorem radar_arrangements : 
  (word_length.factorial) / (r_count.factorial * a_count.factorial) = 30 := by
  sorry

end radar_arrangements_l4128_412832


namespace bus_interval_theorem_l4128_412872

/-- Given a circular bus route with two buses operating at the same speed with an interval of 21 minutes,
    the interval between three buses operating on the same route at the same speed is 14 minutes. -/
theorem bus_interval_theorem (interval_two_buses : ℕ) (interval_three_buses : ℕ) : 
  interval_two_buses = 21 → interval_three_buses = (2 * interval_two_buses) / 3 := by
  sorry

end bus_interval_theorem_l4128_412872


namespace parallel_lines_count_l4128_412804

/-- Given two sets of intersecting parallel lines in a plane, where one set has 8 lines
    and the intersection forms 588 parallelograms, prove that the other set has 85 lines. -/
theorem parallel_lines_count (n : ℕ) 
  (h1 : n > 0)
  (h2 : (n - 1) * 7 = 588) : 
  n = 85 := by
sorry

end parallel_lines_count_l4128_412804


namespace dime_exchange_theorem_l4128_412810

/-- Represents the number of dimes each person has at each stage -/
structure DimeState :=
  (a : ℤ) (b : ℤ) (c : ℤ)

/-- Represents the transactions between A, B, and C -/
def exchange (state : DimeState) : DimeState :=
  let state1 := DimeState.mk (state.a - state.b - state.c) (2 * state.b) (2 * state.c)
  let state2 := DimeState.mk (2 * state1.a) (state1.b - state1.a - state1.c) (2 * state1.c)
  DimeState.mk (2 * state2.a) (2 * state2.b) (state2.c - state2.a - state2.b)

theorem dime_exchange_theorem (initial : DimeState) :
  exchange initial = DimeState.mk 36 36 36 → initial.a = 36 :=
by sorry

end dime_exchange_theorem_l4128_412810


namespace a_greater_than_b_l4128_412802

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 12345 = (111 + a) * (111 - b)) : a > b := by
  sorry

end a_greater_than_b_l4128_412802


namespace twenty_four_point_game_l4128_412873

theorem twenty_four_point_game (Q : ℕ) (h : Q = 12) : 
  (Q * 9) - (Q * 7) = 24 := by
  sorry

end twenty_four_point_game_l4128_412873


namespace individual_test_scores_l4128_412876

/-- Represents a student's test score -/
structure TestScore where
  value : ℝ

/-- Represents the population of students -/
def Population : Type := Fin 2100

/-- Represents the sample of students -/
def Sample : Type := Fin 100

/-- A function that assigns a test score to each student in the population -/
def scoreAssignment : Population → TestScore := sorry

/-- A function that selects the sample from the population -/
def sampleSelection : Sample → Population := sorry

theorem individual_test_scores 
  (p : Population) 
  (s : Sample) : 
  scoreAssignment p ≠ scoreAssignment (sampleSelection s) → p ≠ sampleSelection s := by
  sorry

end individual_test_scores_l4128_412876


namespace range_of_a_l4128_412847

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := a * x^2 - (3 - a) * x + 1

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x

/-- The theorem stating the range of a -/
theorem range_of_a : 
  {a : ℝ | ∀ x, max (f a x) (g x) > 0} = Set.Icc 0 9 := by sorry

end range_of_a_l4128_412847


namespace square_area_increase_l4128_412829

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := 1.15 * s
  let new_area := new_side^2
  (new_area - original_area) / original_area * 100 = 32.25 := by
  sorry

end square_area_increase_l4128_412829


namespace inequality_proof_l4128_412885

theorem inequality_proof (x a : ℝ) (hx : x > 0 ∧ x ≠ 1) (ha : a < 1) :
  (1 - x^a) / (1 - x) < (1 + x)^(a - 1) := by
  sorry

end inequality_proof_l4128_412885


namespace sticker_problem_solution_l4128_412856

def sticker_problem (initial_stickers : ℕ) (front_page_stickers : ℕ) (stickers_per_page : ℕ) (remaining_stickers : ℕ) : ℕ :=
  (initial_stickers - remaining_stickers - front_page_stickers) / stickers_per_page

theorem sticker_problem_solution :
  sticker_problem 89 3 7 44 = 6 := by
  sorry

end sticker_problem_solution_l4128_412856


namespace f_is_odd_l4128_412836

def f (p : ℝ) (x : ℝ) : ℝ := x * |x| + p * x

theorem f_is_odd (p : ℝ) : 
  ∀ x : ℝ, f p (-x) = -(f p x) := by
sorry

end f_is_odd_l4128_412836


namespace sum_f_negative_l4128_412822

def f (x : ℝ) : ℝ := x + x^3

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end sum_f_negative_l4128_412822


namespace lcm_48_180_l4128_412839

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l4128_412839


namespace toothpick_grid_30_15_l4128_412831

/-- Represents a rectangular grid made of toothpicks -/
structure ToothpickGrid where
  height : ℕ  -- Number of toothpicks in height
  width : ℕ   -- Number of toothpicks in width

/-- Calculates the total number of toothpicks in a grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  (grid.height + 1) * grid.width + (grid.width + 1) * grid.height

/-- Theorem: A 30x15 toothpick grid uses 945 toothpicks -/
theorem toothpick_grid_30_15 :
  totalToothpicks { height := 30, width := 15 } = 945 := by
  sorry


end toothpick_grid_30_15_l4128_412831


namespace cosine_sine_identity_l4128_412813

theorem cosine_sine_identity (θ : Real) (h : Real.tan θ = 1/3) :
  Real.cos θ ^ 2 + (1/2) * Real.sin (2 * θ) = 6/5 := by
  sorry

end cosine_sine_identity_l4128_412813


namespace g_of_3_equals_3_over_17_l4128_412817

-- Define the function g
def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

-- State the theorem
theorem g_of_3_equals_3_over_17 : g 3 = 3 / 17 := by
  sorry

end g_of_3_equals_3_over_17_l4128_412817


namespace number_of_factors_l4128_412841

theorem number_of_factors (n : ℕ+) : 
  (Finset.range n).card = n :=
by sorry

#check number_of_factors

end number_of_factors_l4128_412841


namespace min_games_for_prediction_l4128_412849

/-- Represents the chess tournament setup -/
structure ChessTournament where
  white_rook_students : ℕ
  black_elephant_students : ℕ
  total_games : ℕ
  games_per_white_student : ℕ

/-- Defines the specific chess tournament in the problem -/
def problem_tournament : ChessTournament :=
  { white_rook_students := 15,
    black_elephant_students := 20,
    total_games := 300,
    games_per_white_student := 20 }

/-- Theorem stating the minimum number of games for Sasha's prediction -/
theorem min_games_for_prediction (t : ChessTournament) 
  (h1 : t.white_rook_students * t.black_elephant_students = t.total_games)
  (h2 : t.games_per_white_student = t.black_elephant_students) :
  t.total_games - (t.white_rook_students - 1) * t.games_per_white_student = 280 :=
sorry

end min_games_for_prediction_l4128_412849


namespace integral_sqrt_one_minus_x_squared_plus_x_l4128_412812

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end integral_sqrt_one_minus_x_squared_plus_x_l4128_412812


namespace remaining_lawn_after_one_hour_l4128_412803

/-- Given that Mary can mow the entire lawn in 3 hours, 
    this function calculates the fraction of the lawn mowed in a given time. -/
def fraction_mowed (hours : ℚ) : ℚ := hours / 3

/-- This theorem states that if Mary works for 1 hour, 
    then 2/3 of the lawn remains to be mowed. -/
theorem remaining_lawn_after_one_hour : 
  1 - (fraction_mowed 1) = 2/3 := by sorry

end remaining_lawn_after_one_hour_l4128_412803


namespace number_equation_solution_l4128_412895

theorem number_equation_solution : 
  ∃ x : ℝ, (3 * x - 5 = 40) ∧ (x = 15) := by sorry

end number_equation_solution_l4128_412895


namespace same_color_probability_l4128_412884

/-- The probability of drawing two balls of the same color from a bag with 6 green and 7 white balls -/
theorem same_color_probability (total_balls : ℕ) (green_balls : ℕ) (white_balls : ℕ)
  (h1 : total_balls = green_balls + white_balls)
  (h2 : green_balls = 6)
  (h3 : white_balls = 7) :
  (green_balls * (green_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 6 / 13 := by
sorry

end same_color_probability_l4128_412884


namespace one_billion_scientific_notation_l4128_412862

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- One billion -/
def oneBillion : ℕ := 1000000000

/-- Theorem: The scientific notation of one billion is 1 × 10^9 -/
theorem one_billion_scientific_notation :
  ∃ (sn : ScientificNotation), sn.a = 1 ∧ sn.n = 9 ∧ (sn.a * (10 : ℝ) ^ sn.n = oneBillion) :=
sorry

end one_billion_scientific_notation_l4128_412862


namespace divisor_greater_than_remainder_l4128_412806

theorem divisor_greater_than_remainder (a b q r : ℕ) : 
  a = b * q + r → r = 8 → b > 8 := by sorry

end divisor_greater_than_remainder_l4128_412806


namespace rectangle_length_fraction_l4128_412899

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ) 
  (h1 : square_area = 1600)
  (h2 : rectangle_area = 160)
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end rectangle_length_fraction_l4128_412899


namespace fifteenth_student_age_l4128_412838

theorem fifteenth_student_age (total_students : ℕ) (avg_age : ℕ) 
  (group1_size : ℕ) (group1_avg : ℕ) (group2_size : ℕ) (group2_avg : ℕ) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 3 →
  group1_avg = 14 →
  group2_size = 11 →
  group2_avg = 16 →
  total_students * avg_age - (group1_size * group1_avg + group2_size * group2_avg) = 7 := by
  sorry

#check fifteenth_student_age

end fifteenth_student_age_l4128_412838
