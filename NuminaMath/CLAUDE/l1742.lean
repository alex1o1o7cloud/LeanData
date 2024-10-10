import Mathlib

namespace complex_modulus_sum_l1742_174262

theorem complex_modulus_sum : Complex.abs (3 - 5*Complex.I) + Complex.abs (3 + 7*Complex.I) = Real.sqrt 34 + Real.sqrt 58 := by
  sorry

end complex_modulus_sum_l1742_174262


namespace eggs_in_fridge_l1742_174281

/-- Given a chef with eggs and cake-making information, calculate the number of eggs left in the fridge. -/
theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (cakes_made : ℕ) : 
  total_eggs = 60 → eggs_per_cake = 5 → cakes_made = 10 → 
  total_eggs - (eggs_per_cake * cakes_made) = 10 := by
  sorry

#check eggs_in_fridge

end eggs_in_fridge_l1742_174281


namespace unique_solution_l1742_174200

theorem unique_solution : ∃! x : ℝ, 70 + 5 * 12 / (180 / x) = 71 :=
by
  -- The proof goes here
  sorry

end unique_solution_l1742_174200


namespace grasshopper_cannot_return_after_25_jumps_l1742_174290

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of the grasshopper after n jumps -/
def grasshopper_position (n : ℕ) : ℕ := sum_first_n n

theorem grasshopper_cannot_return_after_25_jumps :
  ∃ k : ℕ, grasshopper_position 25 = 2 * k + 1 :=
sorry

end grasshopper_cannot_return_after_25_jumps_l1742_174290


namespace division_remainder_proof_l1742_174294

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 125)
  (h2 : divisor = 15)
  (h3 : quotient = 8)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 5 := by
sorry

end division_remainder_proof_l1742_174294


namespace rectangle_perimeter_13km_l1742_174221

/-- The perimeter of a rectangle with both sides equal to 13 km is 52 km. -/
theorem rectangle_perimeter_13km (l w : ℝ) : 
  l = 13 → w = 13 → 2 * (l + w) = 52 := by
  sorry

end rectangle_perimeter_13km_l1742_174221


namespace quadratic_factorization_l1742_174288

theorem quadratic_factorization :
  ∀ x : ℝ, 12 * x^2 + 16 * x - 20 = 4 * (x - 1) * (3 * x + 5) := by
  sorry

end quadratic_factorization_l1742_174288


namespace inequality_cubic_quadratic_l1742_174260

theorem inequality_cubic_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 > a^2 * b + a * b^2 := by sorry

end inequality_cubic_quadratic_l1742_174260


namespace money_distribution_l1742_174217

theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 → 
  a + c = 200 → 
  b + c = 320 → 
  c = 20 := by
sorry

end money_distribution_l1742_174217


namespace binary_101_is_5_l1742_174219

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₍₂₎ -/
def binary_101 : List Bool := [true, false, true]

/-- Theorem stating that the decimal representation of 101₍₂₎ is 5 -/
theorem binary_101_is_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end binary_101_is_5_l1742_174219


namespace boots_sold_on_monday_l1742_174223

/-- Represents the sales data for a shoe store on a given day -/
structure DailySales where
  shoes : ℕ
  boots : ℕ
  total : ℚ

/-- Represents the pricing structure of the shoe store -/
structure Pricing where
  shoe_price : ℚ
  boot_price : ℚ

def monday_sales : DailySales :=
  { shoes := 22, boots := 24, total := 460 }

def tuesday_sales : DailySales :=
  { shoes := 8, boots := 32, total := 560 }

def store_pricing : Pricing :=
  { shoe_price := 2, boot_price := 17 }

theorem boots_sold_on_monday :
  ∃ (x : ℕ), 
    x = monday_sales.boots ∧
    store_pricing.boot_price = store_pricing.shoe_price + 15 ∧
    monday_sales.shoes * store_pricing.shoe_price + x * store_pricing.boot_price = monday_sales.total ∧
    tuesday_sales.shoes * store_pricing.shoe_price + tuesday_sales.boots * store_pricing.boot_price = tuesday_sales.total :=
by sorry

end boots_sold_on_monday_l1742_174223


namespace line_intercepts_sum_l1742_174266

/-- Given a line with equation y - 3 = -3(x - 6), 
    prove that the sum of its x-intercept and y-intercept is 28 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 6)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 3 = -3 * (x_int - 6)) ∧ 
    (0 - 3 = -3 * (x_int - 6)) ∧
    (y_int - 3 = -3 * (0 - 6)) ∧
    (x_int + y_int = 28)) :=
by sorry

end line_intercepts_sum_l1742_174266


namespace min_box_value_l1742_174293

theorem min_box_value (a b Box : ℤ) : 
  (a ≠ b ∧ a ≠ Box ∧ b ≠ Box) →
  (∀ x, (a * x + b) * (b * x + a) = 31 * x^2 + Box * x + 31) →
  962 ≤ Box :=
by sorry

end min_box_value_l1742_174293


namespace adjustment_ways_l1742_174242

def front_row : ℕ := 4
def back_row : ℕ := 8
def students_to_move : ℕ := 2

def ways_to_select : ℕ := Nat.choose back_row students_to_move
def ways_to_insert : ℕ := Nat.factorial (front_row + students_to_move) / Nat.factorial front_row

theorem adjustment_ways : 
  ways_to_select * ways_to_insert = 840 := by sorry

end adjustment_ways_l1742_174242


namespace clothing_price_comparison_l1742_174238

theorem clothing_price_comparison (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 120 →
  increase_rate = 0.2 →
  discount_rate = 0.2 →
  original_price * (1 + increase_rate) * (1 - discount_rate) < original_price :=
by sorry

end clothing_price_comparison_l1742_174238


namespace right_triangle_area_l1742_174256

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 12 →
  angle = 30 * π / 180 →
  let a := h / 2
  let b := a * Real.sqrt 3
  (1 / 2) * a * b = 18 * Real.sqrt 3 := by
sorry

end right_triangle_area_l1742_174256


namespace sum_remainder_mod_9_l1742_174282

theorem sum_remainder_mod_9 : (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 := by
  sorry

end sum_remainder_mod_9_l1742_174282


namespace os_overhead_cost_value_l1742_174254

/-- The cost per millisecond of computer time -/
def cost_per_ms : ℚ := 23 / 1000

/-- The cost for mounting a data tape -/
def tape_cost : ℚ := 5.35

/-- The total cost for 1 run with 1.5 seconds of computer time -/
def total_cost : ℚ := 40.92

/-- The duration of the computer run in milliseconds -/
def run_duration_ms : ℕ := 1500

/-- The operating-system overhead cost -/
def os_overhead_cost : ℚ := total_cost - (cost_per_ms * run_duration_ms) - tape_cost

theorem os_overhead_cost_value : os_overhead_cost = 1.07 := by
  sorry

end os_overhead_cost_value_l1742_174254


namespace ellipse_properties_l1742_174253

/-- Ellipse C in the Cartesian coordinate system -/
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Line intersecting ellipse C -/
def L (k x : ℝ) : ℝ := k * (x - 4)

/-- Point A: left vertex of ellipse C -/
def A : ℝ × ℝ := (-2, 0)

/-- Point M: first intersection of line L and ellipse C -/
noncomputable def M (k : ℝ) : ℝ × ℝ := 
  let x₁ := (16 * k^2 + 4 * k * Real.sqrt (1 - 12 * k^2)) / (1 + 4 * k^2)
  (x₁, L k x₁)

/-- Point N: second intersection of line L and ellipse C -/
noncomputable def N (k : ℝ) : ℝ × ℝ := 
  let x₂ := (16 * k^2 - 4 * k * Real.sqrt (1 - 12 * k^2)) / (1 + 4 * k^2)
  (x₂, L k x₂)

/-- Point P: intersection of x = 1 and line BM -/
noncomputable def P (k : ℝ) : ℝ × ℝ := 
  let x₁ := (M k).1
  (1, k * (x₁ - 4) / (x₁ - 2))

/-- Area of triangle OMN -/
noncomputable def area_OMN (k : ℝ) : ℝ := 
  8 * Real.sqrt (1/k^2 - 12) / (1/k^2 + 4)

theorem ellipse_properties (k : ℝ) (hk : k ≠ 0) :
  (∃ (t : ℝ), t • (A.1 - (P k).1, A.2 - (P k).2) = ((N k).1 - A.1, (N k).2 - A.2)) ∧
  (∀ (k : ℝ), k ≠ 0 → area_OMN k ≤ 1) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ area_OMN k = 1) := by sorry

end ellipse_properties_l1742_174253


namespace smallest_number_proof_l1742_174202

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof (x : ℕ) (hx : x > 0) : 
  (∀ n : ℕ, n < 5064 → ¬(is_divisible_by (n - 24) x ∧ 
                         is_divisible_by (n - 24) 10 ∧ 
                         is_divisible_by (n - 24) 15 ∧ 
                         is_divisible_by (n - 24) 20 ∧ 
                         (n - 24) / x = 84 ∧ 
                         (n - 24) / 10 = 84 ∧ 
                         (n - 24) / 15 = 84 ∧ 
                         (n - 24) / 20 = 84)) ∧
  (is_divisible_by (5064 - 24) x ∧ 
   is_divisible_by (5064 - 24) 10 ∧ 
   is_divisible_by (5064 - 24) 15 ∧ 
   is_divisible_by (5064 - 24) 20 ∧ 
   (5064 - 24) / x = 84 ∧ 
   (5064 - 24) / 10 = 84 ∧ 
   (5064 - 24) / 15 = 84 ∧ 
   (5064 - 24) / 20 = 84) :=
by sorry

end smallest_number_proof_l1742_174202


namespace dans_balloons_l1742_174224

theorem dans_balloons (dans_balloons : ℕ) (tims_balloons : ℕ) : 
  tims_balloons = 203 → tims_balloons = 7 * dans_balloons → dans_balloons = 29 := by
  sorry

end dans_balloons_l1742_174224


namespace inverse_sum_mod_25_l1742_174283

theorem inverse_sum_mod_25 :
  ∃ (a b c : ℤ), (7 * a) % 25 = 1 ∧ 
                 (7 * b) % 25 = a % 25 ∧ 
                 (7 * c) % 25 = b % 25 ∧ 
                 (a + b + c) % 25 = 9 := by
  sorry

end inverse_sum_mod_25_l1742_174283


namespace quadratic_real_roots_condition_l1742_174204

/-- For a quadratic equation x^2 - 2x + m = 0 to have real roots, m must be less than or equal to 1 -/
theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end quadratic_real_roots_condition_l1742_174204


namespace new_average_after_grace_marks_l1742_174259

theorem new_average_after_grace_marks 
  (num_students : ℕ) 
  (original_average : ℚ) 
  (grace_marks : ℚ) :
  num_students = 35 →
  original_average = 37 →
  grace_marks = 3 →
  (num_students : ℚ) * original_average + num_students * grace_marks = num_students * 40 :=
by sorry

end new_average_after_grace_marks_l1742_174259


namespace inequality_solutions_l1742_174277

-- Define the inequalities
def ineq1a (x : ℝ) := 2*x + 8 > 5*x + 2
def ineq1b (x : ℝ) := 2*x + 8 + 4/(x-1) > 5*x + 2 + 4/(x-1)

def ineq2a (x : ℝ) := 2*x + 8 < 5*x + 2
def ineq2b (x : ℝ) := 2*x + 8 + 4/(x-1) < 5*x + 2 + 4/(x-1)

def ineq3a (x : ℝ) := 3/(x-1) > (x+2)/(x-2)
def ineq3b (x : ℝ) := 3/(x-1) + (3*x-4)/(x-1) > (x+2)/(x-2) + (3*x-4)/(x-1)

-- Define the theorem
theorem inequality_solutions :
  (∃ x : ℝ, ineq1a x ≠ ineq1b x) ∧
  (∀ x : ℝ, ineq2a x ↔ ineq2b x) ∧
  (∀ x : ℝ, x ≠ 1 → x ≠ 2 → (ineq3a x ↔ ineq3b x)) :=
sorry

end inequality_solutions_l1742_174277


namespace intersection_complement_equality_l1742_174249

def U : Set ℝ := Set.univ
def A : Set ℝ := {-3, -2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {-3, -2, -1, 0} := by sorry

end intersection_complement_equality_l1742_174249


namespace travel_time_difference_l1742_174274

/-- Proves the equation for the travel time difference between two groups -/
theorem travel_time_difference 
  (x : ℝ) -- walking speed in km/h
  (h1 : x > 0) -- walking speed is positive
  (distance : ℝ) -- distance traveled
  (h2 : distance = 4) -- distance is 4 km
  (time_diff : ℝ) -- time difference in hours
  (h3 : time_diff = 1/3) -- time difference is 1/3 hours
  : 
  distance / x - distance / (2 * x) = time_diff :=
by sorry

end travel_time_difference_l1742_174274


namespace circle_radius_from_area_l1742_174228

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) :
  A = Real.pi * r^2 → r = 8 := by
  sorry

end circle_radius_from_area_l1742_174228


namespace three_numbers_sum_l1742_174243

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 8 →  -- Median is 8
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 60 := by
sorry

end three_numbers_sum_l1742_174243


namespace corn_preference_result_l1742_174284

/-- The percentage of children preferring corn in Carolyn's daycare -/
def corn_preference_percentage (total_children : ℕ) (corn_preference : ℕ) : ℚ :=
  (corn_preference : ℚ) / (total_children : ℚ) * 100

/-- Theorem stating that the percentage of children preferring corn is 17.5% -/
theorem corn_preference_result : 
  corn_preference_percentage 40 7 = 17.5 := by
  sorry

end corn_preference_result_l1742_174284


namespace system_solution_condition_l1742_174234

/-- The system of equations has a solution for any a if and only if 0 ≤ b ≤ 2. -/
theorem system_solution_condition (b : ℝ) :
  (∀ a : ℝ, ∃ x y : ℝ, x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := by
  sorry

end system_solution_condition_l1742_174234


namespace robins_total_distance_l1742_174212

/-- The total distance Robin walks given his journey to the city center -/
theorem robins_total_distance (distance_to_center : ℕ) (initial_distance : ℕ) : 
  distance_to_center = 500 → initial_distance = 200 → 
  initial_distance + initial_distance + distance_to_center = 900 := by
sorry

end robins_total_distance_l1742_174212


namespace third_roll_wraps_four_gifts_l1742_174275

/-- Represents the number of gifts wrapped with the third roll of paper. -/
def gifts_wrapped_third_roll (total_rolls : ℕ) (total_gifts : ℕ) (gifts_first_roll : ℕ) (gifts_second_roll : ℕ) : ℕ :=
  total_gifts - (gifts_first_roll + gifts_second_roll)

/-- Proves that given 3 rolls of wrapping paper and 12 gifts, if 1 roll wraps 3 gifts
    and 1 roll wraps 5 gifts, then the number of gifts wrapped with the third roll is 4. -/
theorem third_roll_wraps_four_gifts :
  gifts_wrapped_third_roll 3 12 3 5 = 4 := by
  sorry

end third_roll_wraps_four_gifts_l1742_174275


namespace soap_box_dimension_proof_l1742_174273

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.length * d.width * d.height

theorem soap_box_dimension_proof 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h_carton : carton = ⟨25, 48, 60⟩)
  (h_soap : soap = ⟨8, soap.width, 5⟩)
  (h_max_boxes : (300 : ℝ) * boxVolume soap ≤ boxVolume carton) :
  soap.width ≤ 6 := by
sorry

end soap_box_dimension_proof_l1742_174273


namespace library_books_count_l1742_174214

theorem library_books_count :
  ∀ (n : ℕ),
    500 < n ∧ n < 650 ∧
    ∃ (r : ℕ), n = 12 * r + 7 ∧
    ∃ (l : ℕ), n = 25 * l - 5 →
    n = 595 :=
by sorry

end library_books_count_l1742_174214


namespace system_three_solutions_l1742_174291

def system (a : ℝ) (x y : ℝ) : Prop :=
  y = |x - Real.sqrt a| + Real.sqrt a - 2 ∧
  (|x| - 4)^2 + (|y| - 3)^2 = 25

def has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    system a x₁ y₁ ∧ system a x₂ y₂ ∧ system a x₃ y₃ ∧
    (∀ x y, system a x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃))

theorem system_three_solutions :
  ∀ a : ℝ, has_exactly_three_solutions a ↔ a = 1 ∨ a = 16 ∨ a = ((5 * Real.sqrt 2 + 1) / 2)^2 :=
sorry

end system_three_solutions_l1742_174291


namespace smallest_three_digit_with_digit_product_8_l1742_174285

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem smallest_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
sorry

end smallest_three_digit_with_digit_product_8_l1742_174285


namespace argument_not_pi_over_four_l1742_174261

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z-|z+1|| = |z+|z-1||
def condition (z : ℂ) : Prop :=
  Complex.abs (z - Complex.abs (z + 1)) = Complex.abs (z + Complex.abs (z - 1))

-- Theorem statement
theorem argument_not_pi_over_four (h : condition z) :
  Complex.arg z ≠ Real.pi / 4 :=
sorry

end argument_not_pi_over_four_l1742_174261


namespace trigonometric_identity_l1742_174239

open Real

theorem trigonometric_identity :
  sin (12 * π / 180) * cos (36 * π / 180) * sin (48 * π / 180) * cos (72 * π / 180) * tan (18 * π / 180) =
  1/2 * (sin (12 * π / 180)^2 + sin (12 * π / 180) * cos (6 * π / 180)) * sin (18 * π / 180)^2 / cos (18 * π / 180) := by
  sorry

end trigonometric_identity_l1742_174239


namespace min_value_theorem_l1742_174280

theorem min_value_theorem (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 2*m + n = 1) :
  1/m + 2/n ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), 0 < m₀ ∧ 0 < n₀ ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
by sorry

end min_value_theorem_l1742_174280


namespace trajectory_and_tangent_line_l1742_174216

-- Define points A, B, and P
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)
def P : ℝ → ℝ → ℝ × ℝ := λ x y => (x, y)

-- Define the condition |PA| = 2|PB|
def condition (x y : ℝ) : Prop :=
  (x + 3)^2 + y^2 = 4 * ((x - 3)^2 + y^2)

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 16

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x + y + c = 0

-- Define the tangent condition
def is_tangent (c : ℝ) : Prop :=
  ∃ x y : ℝ, trajectory_C x y ∧ line_l c x y ∧
  ∀ x' y' : ℝ, trajectory_C x' y' → line_l c x' y' → (x', y') = (x, y)

-- State the theorem
theorem trajectory_and_tangent_line :
  (∀ x y : ℝ, condition x y ↔ trajectory_C x y) ∧
  (∃ c : ℝ, is_tangent c ∧ c = -5 + 4 * Real.sqrt 2 ∨ c = -5 - 4 * Real.sqrt 2) :=
sorry

end trajectory_and_tangent_line_l1742_174216


namespace correct_rounding_l1742_174206

def round_to_nearest_hundred (x : ℤ) : ℤ :=
  (x + 50) / 100 * 100

theorem correct_rounding : round_to_nearest_hundred ((58 + 44) * 3) = 300 := by
  sorry

end correct_rounding_l1742_174206


namespace perpendicular_line_equation_l1742_174233

/-- Given a line L1 with equation mx - m^2y = 1 passing through point P(2, 1),
    prove that the perpendicular line L2 at P has equation x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  (∀ x y, m * x - m^2 * y = 1 → x = 2 ∧ y = 1) →
  (∀ x y, x + y - 3 = 0 ↔ 
    (m * x - m^2 * y = 1 → 
      (x - 2) * (x - 2) + (y - 1) * (y - 1) = 
      (2 - 2) * (2 - 2) + (1 - 1) * (1 - 1))) :=
by sorry

end perpendicular_line_equation_l1742_174233


namespace union_A_B_when_a_is_one_A_subset_complement_B_iff_l1742_174272

-- Define set A
def A : Set ℝ := {x | (x - 1) / (x - 5) < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

-- Part 1
theorem union_A_B_when_a_is_one : 
  A ∪ B 1 = {x : ℝ | 0 < x ∧ x < 5} := by sorry

-- Part 2
theorem A_subset_complement_B_iff : 
  ∀ a : ℝ, A ⊆ (Set.univ \ B a) ↔ a ≤ 0 ∨ a ≥ 6 := by sorry

end union_A_B_when_a_is_one_A_subset_complement_B_iff_l1742_174272


namespace page_number_added_twice_l1742_174279

theorem page_number_added_twice (n : ℕ) (x : ℕ) 
  (h1 : x ≤ n) 
  (h2 : n * (n + 1) / 2 + x = 3050) : 
  x = 47 := by
  sorry

end page_number_added_twice_l1742_174279


namespace safari_lions_count_l1742_174255

theorem safari_lions_count (L S G : ℕ) : 
  S = L / 2 →
  G = S - 10 →
  2 * L + 3 * S + (G + 20) = 410 →
  L = 72 := by
sorry

end safari_lions_count_l1742_174255


namespace fish_population_estimate_l1742_174278

/-- The number of fish tagged on April 1 -/
def tagged_april : ℕ := 120

/-- The number of fish captured on August 1 -/
def captured_august : ℕ := 150

/-- The number of tagged fish found in the August 1 sample -/
def tagged_in_august : ℕ := 5

/-- The proportion of fish that left the pond between April 1 and August 1 -/
def left_pond : ℚ := 3/10

/-- The proportion of fish in the August sample that were not in the pond in April -/
def new_fish_proportion : ℚ := 1/2

/-- The estimated number of fish in the pond on April 1 -/
def fish_population : ℕ := 1800

/-- Theorem stating that given the conditions, the fish population on April 1 was 1800 -/
theorem fish_population_estimate :
  tagged_april = 120 →
  captured_august = 150 →
  tagged_in_august = 5 →
  left_pond = 3/10 →
  new_fish_proportion = 1/2 →
  fish_population = 1800 := by
  sorry

end fish_population_estimate_l1742_174278


namespace cone_base_diameter_l1742_174207

/-- Represents a cone with given properties -/
structure Cone where
  surfaceArea : ℝ
  lateralSurfaceIsSemicircle : Prop

/-- Theorem stating that a cone with surface area 3π and lateral surface unfolding 
    into a semicircle has a base diameter of √6 -/
theorem cone_base_diameter (c : Cone) 
    (h1 : c.surfaceArea = 3 * Real.pi)
    (h2 : c.lateralSurfaceIsSemicircle) : 
    ∃ (d : ℝ), d = Real.sqrt 6 ∧ d = 2 * (Real.sqrt ((3 : ℝ) / 2)) := by
  sorry

end cone_base_diameter_l1742_174207


namespace unique_root_of_unity_polynomial_l1742_174222

def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

def is_cube_root_of_unity (z : ℂ) : Prop :=
  ∃ k : ℕ, z^(3*k) = 1

theorem unique_root_of_unity_polynomial (c d : ℤ) :
  ∃! z : ℂ, is_root_of_unity z ∧ is_cube_root_of_unity z ∧ z^3 + c*z + d = 0 :=
sorry

end unique_root_of_unity_polynomial_l1742_174222


namespace solve_oliver_money_problem_l1742_174244

def oliver_money_problem (initial_amount savings puzzle_cost gift final_amount : ℕ) 
  (frisbee_cost : ℕ) : Prop :=
  initial_amount + savings + gift - puzzle_cost - frisbee_cost = final_amount

theorem solve_oliver_money_problem :
  ∃ (frisbee_cost : ℕ), oliver_money_problem 9 5 3 8 15 frisbee_cost ∧ frisbee_cost = 4 := by
  sorry

end solve_oliver_money_problem_l1742_174244


namespace angle_sum_equals_pi_over_four_l1742_174247

theorem angle_sum_equals_pi_over_four (α β : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : Real.tan α = 1 / 7)
  (h6 : Real.tan β = 3 / 4) : 
  α + β = π / 4 := by
  sorry

end angle_sum_equals_pi_over_four_l1742_174247


namespace smallest_divisible_number_l1742_174252

theorem smallest_divisible_number : ∃ N : ℕ,
  (∀ k : ℕ, 2 ≤ k → k ≤ 10 → (N + k) % k = 0) ∧
  (∀ M : ℕ, M < N → ∃ j : ℕ, 2 ≤ j ∧ j ≤ 10 ∧ (M + j) % j ≠ 0) ∧
  N = 2520 :=
by sorry

end smallest_divisible_number_l1742_174252


namespace corrected_mean_l1742_174211

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 20 ∧ original_mean = 36 ∧ incorrect_value = 40 ∧ correct_value = 25 →
  (n * original_mean - (incorrect_value - correct_value)) / n = 35.25 := by
  sorry

end corrected_mean_l1742_174211


namespace function_composition_l1742_174215

/-- Given a function f such that f(3x) = 5 / (3 + x) for all x > 0,
    prove that 3f(x) = 45 / (9 + x) --/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 5 / (3 + x)) :
  ∀ x > 0, 3 * f x = 45 / (9 + x) := by
  sorry

end function_composition_l1742_174215


namespace train_distance_problem_l1742_174226

theorem train_distance_problem (speed1 speed2 extra_distance : ℝ) 
  (h1 : speed1 = 50)
  (h2 : speed2 = 60)
  (h3 : extra_distance = 100)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0) :
  ∃ (distance1 distance2 : ℝ),
    distance1 > 0 ∧
    distance2 > 0 ∧
    distance2 = distance1 + extra_distance ∧
    distance1 / speed1 = distance2 / speed2 ∧
    distance1 + distance2 = 1100 :=
by sorry

end train_distance_problem_l1742_174226


namespace xy_squared_sum_l1742_174209

theorem xy_squared_sum (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end xy_squared_sum_l1742_174209


namespace systematic_sample_count_in_range_l1742_174267

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (start + i * (total / sampleSize)) % total + 1)

/-- Count numbers in a given range -/
def countInRange (list : List ℕ) (low : ℕ) (high : ℕ) : ℕ :=
  list.filter (fun n => low ≤ n && n ≤ high) |>.length

theorem systematic_sample_count_in_range :
  let total := 840
  let sampleSize := 42
  let start := 13
  let sample := systematicSample total sampleSize start
  countInRange sample 490 700 = 11 := by
  sorry

end systematic_sample_count_in_range_l1742_174267


namespace sin_cos_sum_14_46_l1742_174265

theorem sin_cos_sum_14_46 :
  Real.sin (14 * π / 180) * Real.cos (46 * π / 180) +
  Real.sin (46 * π / 180) * Real.cos (14 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_14_46_l1742_174265


namespace max_gcd_of_sum_1155_l1742_174245

theorem max_gcd_of_sum_1155 :
  ∃ (a b : ℕ+), a + b = 1155 ∧
  ∀ (c d : ℕ+), c + d = 1155 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 105 :=
by sorry

end max_gcd_of_sum_1155_l1742_174245


namespace count_special_numbers_is_279_l1742_174297

/-- A function that counts the number of positive integers less than 100,000 
    with at most two different digits, where one of the digits must be 1. -/
def count_special_numbers : ℕ :=
  let max_number := 100000
  let required_digit := 1
  -- Implementation details are omitted
  279

/-- Theorem stating that the count of special numbers is 279. -/
theorem count_special_numbers_is_279 : count_special_numbers = 279 := by
  sorry

end count_special_numbers_is_279_l1742_174297


namespace inequality_proof_l1742_174289

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) := by
sorry

end inequality_proof_l1742_174289


namespace kenny_time_ratio_l1742_174203

/-- Proves that the ratio of Kenny's running time to basketball playing time is 2:1 -/
theorem kenny_time_ratio : 
  ∀ (basketball_time trumpet_time running_time : ℕ),
    basketball_time = 10 →
    trumpet_time = 40 →
    trumpet_time = 2 * running_time →
    running_time / basketball_time = 2 / 1 :=
by
  sorry

end kenny_time_ratio_l1742_174203


namespace amy_total_crumbs_l1742_174231

/-- Theorem: Amy's total crumbs given Arthur's total crumbs -/
theorem amy_total_crumbs (c : ℝ) : ℝ := by
  -- Define Arthur's trips and crumbs per trip
  let arthur_trips : ℝ := c / (c / c)
  let arthur_crumbs_per_trip : ℝ := c / arthur_trips

  -- Define Amy's trips and crumbs per trip
  let amy_trips : ℝ := 2 * arthur_trips
  let amy_crumbs_per_trip : ℝ := 1.5 * arthur_crumbs_per_trip

  -- Calculate Amy's total crumbs
  let amy_total : ℝ := amy_trips * amy_crumbs_per_trip

  -- Prove that Amy's total crumbs equals 3c
  sorry

end amy_total_crumbs_l1742_174231


namespace expression_not_prime_l1742_174268

theorem expression_not_prime :
  ∀ x : ℕ, 0 < x → x < 100 →
  ∃ k : ℕ, 3^x + 5^x + 7^x + 11^x + 13^x + 17^x + 19^x = 3 * k :=
by sorry

end expression_not_prime_l1742_174268


namespace new_person_weight_l1742_174246

/-- Given a group of 8 persons, if replacing one person weighing 65 kg with a new person
    increases the average weight by 3.5 kg, then the new person weighs 93 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end new_person_weight_l1742_174246


namespace total_marks_calculation_l1742_174208

theorem total_marks_calculation (num_candidates : ℕ) (average_mark : ℚ) :
  num_candidates = 120 →
  average_mark = 35 →
  (num_candidates : ℚ) * average_mark = 4200 := by
  sorry

end total_marks_calculation_l1742_174208


namespace no_max_cos_squared_sum_l1742_174220

theorem no_max_cos_squared_sum (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  ∃ d > 0, B - A = d ∧ C - B = d →  -- Arithmetic sequence with positive difference
  ¬ ∃ M : ℝ, ∀ A' B' C' : ℝ,
    (0 < A' ∧ 0 < B' ∧ 0 < C' ∧
     A' + B' + C' = π ∧
     ∃ d > 0, B' - A' = d ∧ C' - B' = d) →
    Real.cos A' ^ 2 + Real.cos C' ^ 2 ≤ M :=
by sorry

end no_max_cos_squared_sum_l1742_174220


namespace max_a_value_l1742_174257

theorem max_a_value (a : ℤ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (2*x + 3 > 3*x - 1 ∧ 6*x - a ≥ 2*x + 2) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (∀ x : ℤ, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → ¬(2*x + 3 > 3*x - 1 ∧ 6*x - a ≥ 2*x + 2))) →
  (∃ y : ℝ, y ≥ 0 ∧ (y + a)/(y - 1) + 2*a/(1 - y) = 2) →
  (∀ a' : ℤ, 
    (∃ (x₁' x₂' x₃' : ℤ), 
      (∀ x : ℤ, (2*x + 3 > 3*x - 1 ∧ 6*x - a' ≥ 2*x + 2) ↔ (x = x₁' ∨ x = x₂' ∨ x = x₃')) ∧
      (∀ x : ℤ, x ≠ x₁' ∧ x ≠ x₂' ∧ x ≠ x₃' → ¬(2*x + 3 > 3*x - 1 ∧ 6*x - a' ≥ 2*x + 2))) →
    (∃ y : ℝ, y ≥ 0 ∧ (y + a')/(y - 1) + 2*a'/(1 - y) = 2) →
    a' ≤ a) →
  a = 2 := by sorry

end max_a_value_l1742_174257


namespace prime_arithmetic_sequence_ones_digit_l1742_174276

theorem prime_arithmetic_sequence_ones_digit (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧
  p > 10 ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 →
  p % 10 = 1 := by
sorry

end prime_arithmetic_sequence_ones_digit_l1742_174276


namespace flower_combinations_l1742_174227

/-- The number of valid combinations of roses and carnations -/
def valid_combinations : ℕ := sorry

/-- Predicate for valid combination of roses and carnations -/
def is_valid_combination (r c : ℕ) : Prop :=
  3 * r + 2 * c = 70 ∧ r + c ≥ 20

theorem flower_combinations :
  valid_combinations = 12 ∨
  valid_combinations = 13 ∨
  valid_combinations = 15 ∨
  valid_combinations = 17 ∨
  valid_combinations = 18 :=
sorry

end flower_combinations_l1742_174227


namespace perfect_square_sum_l1742_174271

theorem perfect_square_sum (n : ℕ+) : 
  (∃ m : ℕ, 4^7 + 4^n.val + 4^1998 = m^2) → (n.val = 1003 ∨ n.val = 3988) :=
by sorry

end perfect_square_sum_l1742_174271


namespace four_digit_numbers_count_l1742_174210

/-- The number of different four-digit numbers that can be formed using two 1s, one 2, and one 0 -/
def four_digit_numbers : ℕ :=
  let zero_placements := 3  -- 0 can be placed in hundreds, tens, or ones place
  let two_placements := 3   -- 2 can be placed in any of the remaining 3 positions
  zero_placements * two_placements

/-- Proof that the number of different four-digit numbers formed is 9 -/
theorem four_digit_numbers_count : four_digit_numbers = 9 := by
  sorry

end four_digit_numbers_count_l1742_174210


namespace positive_integer_sum_with_square_twelve_l1742_174235

theorem positive_integer_sum_with_square_twelve (M : ℕ+) :
  (M : ℝ)^2 + M = 12 → M = 3 := by
  sorry

end positive_integer_sum_with_square_twelve_l1742_174235


namespace worker_B_completion_time_l1742_174201

-- Define the time it takes for Worker A to complete the job
def worker_A_time : ℝ := 5

-- Define the time it takes for both workers together to complete the job
def combined_time : ℝ := 3.333333333333333

-- Define the time it takes for Worker B to complete the job
def worker_B_time : ℝ := 10

-- Theorem statement
theorem worker_B_completion_time :
  (1 / worker_A_time + 1 / worker_B_time = 1 / combined_time) →
  worker_B_time = 10 :=
by sorry

end worker_B_completion_time_l1742_174201


namespace triangle_area_part1_triangle_side_part2_l1742_174213

-- Part 1
theorem triangle_area_part1 (A B C : ℝ) (a b c : ℝ) :
  A = π/6 → C = π/4 → a = 2 →
  (1/2) * a * b * Real.sin C = 1 + Real.sqrt 3 :=
sorry

-- Part 2
theorem triangle_side_part2 (A B C : ℝ) (a b c : ℝ) :
  (1/2) * a * b * Real.sin C = Real.sqrt 3 → b = 2 → C = π/3 →
  a = 2 :=
sorry

end triangle_area_part1_triangle_side_part2_l1742_174213


namespace sum_of_reciprocals_equals_243_l1742_174229

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 24*x^2 + 143*x - 210

-- Define the roots of the polynomial
variables (p q r : ℝ)

-- State that p, q, r are the roots of f
axiom roots_of_f : f p = 0 ∧ f q = 0 ∧ f r = 0

-- Define A, B, C as real numbers
variables (A B C : ℝ)

-- State the partial fraction decomposition
axiom partial_fraction_decomposition :
  ∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 24*s^2 + 143*s - 210) = A / (s - p) + B / (s - q) + C / (s - r)

-- The theorem to prove
theorem sum_of_reciprocals_equals_243 :
  1 / A + 1 / B + 1 / C = 243 :=
sorry

end sum_of_reciprocals_equals_243_l1742_174229


namespace total_pictures_uploaded_l1742_174205

/-- Proves that the total number of pictures uploaded is 25 -/
theorem total_pictures_uploaded (first_album : ℕ) (num_other_albums : ℕ) (pics_per_other_album : ℕ) 
  (h1 : first_album = 10)
  (h2 : num_other_albums = 5)
  (h3 : pics_per_other_album = 3) :
  first_album + num_other_albums * pics_per_other_album = 25 := by
  sorry

#check total_pictures_uploaded

end total_pictures_uploaded_l1742_174205


namespace remainder_problem_l1742_174225

theorem remainder_problem (divisor remainder_1657 : ℕ) 
  (h1 : divisor = 127)
  (h2 : remainder_1657 = 6)
  (h3 : ∃ k : ℕ, 1657 = k * divisor + remainder_1657)
  (h4 : ∃ m r : ℕ, 2037 = m * divisor + r ∧ r < divisor)
  (h5 : ∀ d : ℕ, d > divisor → ¬(∃ k1 k2 r1 r2 : ℕ, 1657 = k1 * d + r1 ∧ 2037 = k2 * d + r2 ∧ r1 < d ∧ r2 < d)) :
  ∃ m : ℕ, 2037 = m * divisor + 5 :=
sorry

end remainder_problem_l1742_174225


namespace smallest_integer_solution_system_of_inequalities_solution_l1742_174296

-- Part 1
theorem smallest_integer_solution (x : ℤ) :
  (5 * x + 15 > x - 1) ∧ (∀ y : ℤ, y < x → ¬(5 * y + 15 > y - 1)) ↔ x = -3 :=
sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) ↔ -4 < x ∧ x ≤ 1 :=
sorry

end smallest_integer_solution_system_of_inequalities_solution_l1742_174296


namespace permutations_not_adjacent_l1742_174264

/-- The number of permutations of three 'a's, four 'b's, and two 'c's -/
def total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 4 * Nat.factorial 2)

/-- Permutations where all 'a's are adjacent -/
def perm_a_adjacent : ℕ := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)

/-- Permutations where all 'b's are adjacent -/
def perm_b_adjacent : ℕ := Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Permutations where all 'c's are adjacent -/
def perm_c_adjacent : ℕ := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)

/-- Permutations where both 'a's and 'b's are adjacent -/
def perm_ab_adjacent : ℕ := Nat.factorial 4 / Nat.factorial 2

/-- Permutations where both 'a's and 'c's are adjacent -/
def perm_ac_adjacent : ℕ := Nat.factorial 6 / Nat.factorial 4

/-- Permutations where both 'b's and 'c's are adjacent -/
def perm_bc_adjacent : ℕ := Nat.factorial 5 / Nat.factorial 3

/-- Permutations where 'a's, 'b's, and 'c's are all adjacent -/
def perm_abc_adjacent : ℕ := Nat.factorial 3

theorem permutations_not_adjacent : 
  total_permutations - (perm_a_adjacent + perm_b_adjacent + perm_c_adjacent - 
  perm_ab_adjacent - perm_ac_adjacent - perm_bc_adjacent + perm_abc_adjacent) = 871 := by
  sorry

end permutations_not_adjacent_l1742_174264


namespace ad_ratio_proof_l1742_174232

theorem ad_ratio_proof (page1_ads page2_ads page3_ads page4_ads total_ads : ℕ) : 
  page1_ads = 12 →
  page3_ads = page2_ads + 24 →
  page4_ads = (3 * page2_ads) / 4 →
  total_ads = page1_ads + page2_ads + page3_ads + page4_ads →
  (2 * total_ads) / 3 = 68 →
  page2_ads / page1_ads = 2 := by
sorry

end ad_ratio_proof_l1742_174232


namespace eventual_stability_l1742_174236

/-- Represents the state of the circular arrangement at a given time step -/
def CircularState := Vector Bool 101

/-- Defines the update rule for a single element based on its neighbors -/
def updateElement (left right current : Bool) : Bool :=
  if left ≠ current ∧ right ≠ current then !current else current

/-- Applies the update rule to the entire circular arrangement -/
def updateState (state : CircularState) : CircularState :=
  Vector.ofFn (fun i =>
    updateElement
      (state.get ((i - 1 + 101) % 101))
      (state.get ((i + 1) % 101))
      (state.get i))

/-- Predicate to check if a state is stable (doesn't change under update) -/
def isStable (state : CircularState) : Prop :=
  updateState state = state

/-- The main theorem: there exists a stable state reachable from any initial state -/
theorem eventual_stability :
  ∀ (initialState : CircularState),
  ∃ (n : ℕ) (stableState : CircularState),
  (∀ k, k ≥ n → (updateState^[k] initialState) = stableState) ∧
  isStable stableState :=
sorry


end eventual_stability_l1742_174236


namespace dishonest_dealer_profit_l1742_174263

/-- Represents the weight used by the dealer in grams -/
def dealer_weight : ℝ := 500

/-- Represents the standard weight of 1 kg in grams -/
def standard_weight : ℝ := 1000

/-- The dealer's profit percentage -/
def profit_percentage : ℝ := 50

theorem dishonest_dealer_profit :
  dealer_weight / standard_weight = 1 - (100 / (100 + profit_percentage)) :=
sorry

end dishonest_dealer_profit_l1742_174263


namespace sofia_shopping_cost_l1742_174287

def shirt_cost : ℕ := 7
def shoes_cost : ℕ := shirt_cost + 3
def total_shirts_shoes : ℕ := 2 * shirt_cost + shoes_cost
def bag_cost : ℕ := total_shirts_shoes / 2
def total_cost : ℕ := 2 * shirt_cost + shoes_cost + bag_cost

theorem sofia_shopping_cost : total_cost = 36 := by
  sorry

end sofia_shopping_cost_l1742_174287


namespace combination_three_choose_two_l1742_174299

theorem combination_three_choose_two : Finset.card (Finset.powerset {0, 1, 2} |>.filter (fun s => Finset.card s = 2)) = 3 := by
  sorry

end combination_three_choose_two_l1742_174299


namespace part_one_part_two_l1742_174258

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part I
theorem part_one : 
  let a : ℝ := -4
  {x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ 0 ∨ x ≥ 6} := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 3|) → -1 ≤ a ∧ a ≤ 0 := by sorry

end part_one_part_two_l1742_174258


namespace correct_operation_l1742_174237

theorem correct_operation (m : ℝ) : (2 * m^3)^2 / (2 * m)^2 = m^4 := by
  sorry

end correct_operation_l1742_174237


namespace ford_younger_than_christopher_l1742_174240

/-- Proves that Ford is 2 years younger than Christopher given the conditions of the problem -/
theorem ford_younger_than_christopher :
  ∀ (george christopher ford : ℕ),
    george = christopher + 8 →
    george + christopher + ford = 60 →
    christopher = 18 →
    ∃ (y : ℕ), ford = christopher - y ∧ y = 2 :=
by sorry

end ford_younger_than_christopher_l1742_174240


namespace distance_at_speed1_l1742_174251

def total_distance : ℝ := 250
def speed1 : ℝ := 40
def speed2 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_at_speed1 (x : ℝ) 
  (h1 : x / speed1 + (total_distance - x) / speed2 = total_time) :
  x = 124 := by
  sorry

end distance_at_speed1_l1742_174251


namespace shift_selection_count_l1742_174241

def workers : Nat := 3
def positions : Nat := 2

theorem shift_selection_count : (workers * (workers - 1) = 6) := by
  sorry

end shift_selection_count_l1742_174241


namespace parallel_segment_sum_l1742_174230

/-- Given two points A(a,-2) and B(1,b) in a plane rectangular coordinate system,
    if AB is parallel to the x-axis and AB = 3, then a + b = 2 or a + b = -4 -/
theorem parallel_segment_sum (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (a, -2) ∧ B = (1, b) ∧ 
   (A.2 = B.2) ∧  -- AB is parallel to x-axis
   ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 3^2))  -- AB = 3
  → (a + b = 2 ∨ a + b = -4) := by
sorry

end parallel_segment_sum_l1742_174230


namespace triangle_side_length_l1742_174218

theorem triangle_side_length (PQ PR PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 3.5) :
  ∃ QR : ℝ, QR = 9 ∧ PM^2 = (1/2) * (PQ^2 + PR^2 + QR^2) - (1/4) * QR^2 :=
sorry

end triangle_side_length_l1742_174218


namespace quadratic_roots_properties_l1742_174292

theorem quadratic_roots_properties (r1 r2 : ℝ) : 
  r1 ≠ r2 → 
  r1^2 - 5*r1 + 6 = 0 → 
  r2^2 - 5*r2 + 6 = 0 → 
  (|r1 + r2| ≤ 6) ∧ 
  (|r1 * r2| ≤ 3 ∨ |r1 * r2| ≥ 8) ∧ 
  (r1 ≥ 0 ∨ r2 ≥ 0) := by
  sorry

end quadratic_roots_properties_l1742_174292


namespace total_air_removed_l1742_174250

def air_removal_fractions : List Rat := [1/3, 1/4, 1/5, 1/6, 1/7]

def remaining_air (fractions : List Rat) : Rat :=
  fractions.foldl (fun acc f => acc * (1 - f)) 1

theorem total_air_removed (fractions : List Rat) :
  fractions = air_removal_fractions →
  1 - remaining_air fractions = 5/7 := by
  sorry

end total_air_removed_l1742_174250


namespace median_sum_squares_l1742_174269

theorem median_sum_squares (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let m₁ := (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)
  let m₂ := (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)
  let m₃ := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  m₁^2 + m₂^2 + m₃^2 = 442.5 := by
sorry

end median_sum_squares_l1742_174269


namespace right_triangle_hypotenuse_l1742_174270

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 := by
  sorry

end right_triangle_hypotenuse_l1742_174270


namespace fourth_grade_students_l1742_174295

theorem fourth_grade_students (initial_students : ℝ) (left_students : ℝ) (transferred_students : ℝ) :
  initial_students = 42.0 →
  left_students = 4.0 →
  transferred_students = 10.0 →
  initial_students - left_students - transferred_students = 28.0 := by
  sorry

end fourth_grade_students_l1742_174295


namespace cubic_roots_sum_l1742_174298

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -6 := by
sorry

end cubic_roots_sum_l1742_174298


namespace number_of_unique_lines_l1742_174248

/-- The set of possible coefficients for A and B -/
def S : Finset ℕ := {0, 1, 2, 3, 5}

/-- A line is represented by a pair of distinct coefficients (A, B) -/
def Line : Type := { p : ℕ × ℕ // p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 }

/-- The set of all possible lines -/
def AllLines : Finset Line := sorry

theorem number_of_unique_lines : Finset.card AllLines = 14 := by
  sorry

end number_of_unique_lines_l1742_174248


namespace milk_dilution_l1742_174286

theorem milk_dilution (initial_volume : ℝ) (initial_milk_percentage : ℝ) (water_added : ℝ) :
  initial_volume = 60 →
  initial_milk_percentage = 0.84 →
  water_added = 18.75 →
  let initial_milk_volume := initial_volume * initial_milk_percentage
  let final_volume := initial_volume + water_added
  let final_milk_percentage := initial_milk_volume / final_volume
  final_milk_percentage = 0.64 := by
  sorry

end milk_dilution_l1742_174286
