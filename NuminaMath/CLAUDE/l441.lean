import Mathlib

namespace special_triangle_perimeter_l441_44122

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- One side of the triangle -/
  a : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Radius of the circumscribed circle -/
  R : ℝ
  /-- The side length is positive -/
  a_pos : 0 < a
  /-- The inscribed radius is positive -/
  r_pos : 0 < r
  /-- The circumscribed radius is positive -/
  R_pos : 0 < R

/-- Theorem: The perimeter of the special triangle is 24 -/
theorem special_triangle_perimeter (t : SpecialTriangle)
    (h1 : t.a = 6)
    (h2 : t.r = 2)
    (h3 : t.R = 5) :
    ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ t.a + b + c = 24 := by
  sorry

end special_triangle_perimeter_l441_44122


namespace queen_high_school_teachers_l441_44187

/-- The number of teachers at Queen High School -/
def num_teachers (total_students : ℕ) (classes_per_student : ℕ) (students_per_class : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher)

/-- Theorem: There are 72 teachers at Queen High School -/
theorem queen_high_school_teachers :
  num_teachers 1500 6 25 5 = 72 := by
  sorry

end queen_high_school_teachers_l441_44187


namespace yevgeniy_age_unique_l441_44184

def birth_year (y : ℕ) := 1900 + y

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Define the condition from the problem
def condition (y : ℕ) : Prop :=
  y ≥ 0 ∧ y < 100 ∧ (2011 - birth_year y = sum_of_digits (birth_year y))

-- The theorem to prove
theorem yevgeniy_age_unique :
  ∃! y : ℕ, condition y ∧ (2014 - birth_year y = 23) :=
sorry

end yevgeniy_age_unique_l441_44184


namespace parabola_decreases_left_of_vertex_given_parabola_decreases_left_of_vertex_l441_44143

/-- Represents a parabola of the form y = (x - h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem parabola_decreases_left_of_vertex (p : Parabola) :
  ∀ x₁ x₂, x₁ < x₂ → x₂ < p.h → p.y_coord x₁ > p.y_coord x₂ := by
  sorry

/-- The specific parabola y = (x - 2)^2 + 1 -/
def given_parabola : Parabola :=
  { h := 2, k := 1 }

theorem given_parabola_decreases_left_of_vertex :
  ∀ x₁ x₂, x₁ < x₂ → x₂ < 2 → given_parabola.y_coord x₁ > given_parabola.y_coord x₂ := by
  sorry

end parabola_decreases_left_of_vertex_given_parabola_decreases_left_of_vertex_l441_44143


namespace fourth_rectangle_area_l441_44157

theorem fourth_rectangle_area (total_area : ℝ) (area1 area2 area3 : ℝ) :
  total_area = 168 ∧ 
  area1 = 33 ∧ 
  area2 = 45 ∧ 
  area3 = 20 →
  total_area - (area1 + area2 + area3) = 70 :=
by sorry

end fourth_rectangle_area_l441_44157


namespace even_increasing_negative_ordering_l441_44145

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is increasing on (-∞, 0) if f(x) < f(y) whenever x < y < 0 -/
def IncreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → y < 0 → f x < f y

theorem even_increasing_negative_ordering (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_incr : IncreasingOnNegative f) : 
    f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end even_increasing_negative_ordering_l441_44145


namespace rahim_book_purchase_l441_44180

/-- The amount Rahim paid for books from the first shop -/
def amount_first_shop (books_first_shop : ℕ) (books_second_shop : ℕ) (price_second_shop : ℚ) (average_price : ℚ) : ℚ :=
  (average_price * (books_first_shop + books_second_shop : ℚ)) - price_second_shop

/-- Theorem stating the amount Rahim paid for books from the first shop -/
theorem rahim_book_purchase :
  amount_first_shop 65 50 920 (18088695652173913 / 1000000000000000) = 1160 := by
  sorry

end rahim_book_purchase_l441_44180


namespace parking_cost_excess_hours_l441_44178

theorem parking_cost_excess_hours (base_cost : ℝ) (avg_cost : ℝ) (excess_cost : ℝ) : 
  base_cost = 10 →
  avg_cost = 2.4722222222222223 →
  (base_cost + 7 * excess_cost) / 9 = avg_cost →
  excess_cost = 1.75 := by
sorry

end parking_cost_excess_hours_l441_44178


namespace parabola_one_y_intercept_l441_44155

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define what a y-intercept is
def is_y_intercept (y : ℝ) : Prop := f 0 = y

-- Theorem: The parabola has exactly one y-intercept
theorem parabola_one_y_intercept :
  ∃! y : ℝ, is_y_intercept y :=
sorry

end parabola_one_y_intercept_l441_44155


namespace nine_digit_multiplier_problem_l441_44162

theorem nine_digit_multiplier_problem : 
  ∃! (N : ℕ), 
    (100000000 ≤ N ∧ N ≤ 999999999) ∧ 
    (N * 123456789) % 1000000000 = 987654321 := by
  sorry

end nine_digit_multiplier_problem_l441_44162


namespace students_taking_neither_l441_44109

theorem students_taking_neither (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 75) 
  (h2 : chem = 40) 
  (h3 : bio = 35) 
  (h4 : both = 25) : 
  total - (chem + bio - both) = 25 :=
by sorry

end students_taking_neither_l441_44109


namespace optimal_price_and_profit_l441_44107

/-- Represents the monthly sales quantity as a function of price -/
def sales_quantity (x : ℝ) : ℝ := -10000 * x + 80000

/-- Represents the monthly profit as a function of price -/
def monthly_profit (x : ℝ) : ℝ := (x - 4) * (sales_quantity x)

theorem optimal_price_and_profit :
  let price_1 : ℝ := 5
  let quantity_1 : ℝ := 30000
  let price_2 : ℝ := 6
  let quantity_2 : ℝ := 20000
  let unit_cost : ℝ := 4
  
  -- The sales quantity function is correct
  (∀ x, sales_quantity x = -10000 * x + 80000) ∧
  
  -- The function satisfies the given points
  (sales_quantity price_1 = quantity_1) ∧
  (sales_quantity price_2 = quantity_2) ∧
  
  -- The optimal price is 6
  (∀ x, monthly_profit x ≤ monthly_profit 6) ∧
  
  -- The maximum monthly profit is 40000
  (monthly_profit 6 = 40000) := by
    sorry

end optimal_price_and_profit_l441_44107


namespace ford_vehicle_count_l441_44182

/-- Represents the number of vehicles of each brand on Louie's store parking lot -/
structure VehicleCounts where
  D : ℕ  -- Dodge
  H : ℕ  -- Hyundai
  K : ℕ  -- Kia
  Ho : ℕ -- Honda
  F : ℕ  -- Ford

/-- Conditions for the vehicle counts -/
def satisfiesConditions (v : VehicleCounts) : Prop :=
  v.D + v.H + v.K + v.Ho + v.F = 1000 ∧
  (35 : ℕ) * (v.D + v.H + v.K + v.Ho + v.F) = 100 * v.D ∧
  (10 : ℕ) * (v.D + v.H + v.K + v.Ho + v.F) = 100 * v.H ∧
  v.K = 2 * v.Ho + 50 ∧
  v.F = v.D - 200

theorem ford_vehicle_count (v : VehicleCounts) 
  (h : satisfiesConditions v) : v.F = 150 := by
  sorry

end ford_vehicle_count_l441_44182


namespace round_trip_average_speed_l441_44149

/-- Calculates the average speed of a round trip given the speed of the outbound journey and the fact that the return journey takes twice as long. -/
theorem round_trip_average_speed (outbound_speed : ℝ) :
  outbound_speed = 51 →
  (2 * outbound_speed) / 3 = 34 := by
  sorry

#check round_trip_average_speed

end round_trip_average_speed_l441_44149


namespace no_unique_solution_l441_44160

/-- 
Theorem: The system of equations 4(3x + 4y) = 48 and kx + 12y = 30 
does not have a unique solution if and only if k = -9.
-/
theorem no_unique_solution (k : ℝ) : 
  (∀ x y : ℝ, 4*(3*x + 4*y) = 48 ∧ k*x + 12*y = 30) → 
  (¬∃! (x y : ℝ), 4*(3*x + 4*y) = 48 ∧ k*x + 12*y = 30) ↔ 
  k = -9 :=
sorry


end no_unique_solution_l441_44160


namespace factor_iff_t_eq_neg_six_or_one_l441_44127

/-- The polynomial in question -/
def f (x : ℝ) : ℝ := 4 * x^2 + 20 * x - 24

/-- Theorem stating that x - t is a factor of f(x) if and only if t is -6 or 1 -/
theorem factor_iff_t_eq_neg_six_or_one :
  ∀ t : ℝ, (∃ g : ℝ → ℝ, ∀ x, f x = (x - t) * g x) ↔ (t = -6 ∨ t = 1) := by
  sorry

end factor_iff_t_eq_neg_six_or_one_l441_44127


namespace rationalize_denominator_l441_44175

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end rationalize_denominator_l441_44175


namespace shirts_sold_l441_44177

/-- Proves that the number of shirts sold is 4 given the conditions of the problem -/
theorem shirts_sold (total_money : ℕ) (num_dresses : ℕ) (price_dress : ℕ) (price_shirt : ℕ) :
  total_money = 69 →
  num_dresses = 7 →
  price_dress = 7 →
  price_shirt = 5 →
  (total_money - num_dresses * price_dress) / price_shirt = 4 := by
  sorry

end shirts_sold_l441_44177


namespace arctan_sum_greater_than_pi_half_l441_44189

theorem arctan_sum_greater_than_pi_half (a b : ℝ) : 
  a = 2/3 → (a + 1) * (b + 1) = 3 → Real.arctan a + Real.arctan b > π/2 := by
  sorry

end arctan_sum_greater_than_pi_half_l441_44189


namespace pencil_distribution_remainder_l441_44159

theorem pencil_distribution_remainder : 25197629 % 4 = 1 := by
  sorry

end pencil_distribution_remainder_l441_44159


namespace prob_at_least_one_correct_l441_44108

/-- The probability of subscribing to at least one of two newspapers -/
def prob_at_least_one (p1 p2 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2)

theorem prob_at_least_one_correct (p1 p2 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) : 
  prob_at_least_one p1 p2 = 1 - (1 - p1) * (1 - p2) := by
  sorry

end prob_at_least_one_correct_l441_44108


namespace tangent_parallel_points_l441_44179

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, (y = f x ∧ f' x = 4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by sorry

end tangent_parallel_points_l441_44179


namespace triangle_area_l441_44121

theorem triangle_area (a b : ℝ) (θ : Real) (h1 : a = 30) (h2 : b = 24) (h3 : θ = π/3) :
  (1/2) * a * b * Real.sin θ = 180 * Real.sqrt 3 := by
  sorry

end triangle_area_l441_44121


namespace pecan_pies_count_l441_44188

/-- The number of pecan pies baked by Mrs. Hilt -/
def pecan_pies : ℝ := 16

/-- The number of apple pies baked by Mrs. Hilt -/
def apple_pies : ℝ := 14

/-- The factor by which the total number of pies needs to be increased -/
def increase_factor : ℝ := 5

/-- The total number of pies needed -/
def total_pies_needed : ℝ := 150

/-- Theorem stating that the number of pecan pies is correct given the conditions -/
theorem pecan_pies_count : 
  increase_factor * (pecan_pies + apple_pies) = total_pies_needed := by
  sorry

end pecan_pies_count_l441_44188


namespace theater_eye_colors_l441_44193

theorem theater_eye_colors (total : ℕ) (blue : ℕ) (brown : ℕ) (black : ℕ) (green : ℕ)
  (h_total : total = 100)
  (h_blue : blue = 19)
  (h_brown : brown = total / 2)
  (h_black : black = total / 4)
  (h_green : green = total - (blue + brown + black)) :
  green = 6 := by
sorry

end theater_eye_colors_l441_44193


namespace min_value_product_squares_l441_44135

theorem min_value_product_squares (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 3) :
  x^2 * y^2 * z^2 ≥ 1/64 ∧ ∃ (a : ℝ), a > 0 ∧ a^2 * a^2 * a^2 = 1/64 ∧ 1/a + 1/a + 1/a = 3 :=
by sorry

end min_value_product_squares_l441_44135


namespace complex_multiplication_division_l441_44102

theorem complex_multiplication_division (z₁ z₂ : ℂ) :
  z₁ = 1 + Complex.I →
  z₂ = 2 - Complex.I →
  (z₁ * z₂) / Complex.I = 1 - 3 * Complex.I :=
by sorry

end complex_multiplication_division_l441_44102


namespace zero_not_in_range_of_g_l441_44124

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- arbitrary value for x = -3, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 :=
sorry

end zero_not_in_range_of_g_l441_44124


namespace sqrt_equation_solution_l441_44106

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 * y + 6) = 5 → y = (19 : ℝ) / 2 := by
  sorry

end sqrt_equation_solution_l441_44106


namespace original_raspberry_count_l441_44197

/-- The number of lemon candies Liam originally had -/
def original_lemon : ℕ := sorry

/-- The number of raspberry candies Liam originally had -/
def original_raspberry : ℕ := sorry

/-- The condition that Liam originally had three times as many raspberry candies as lemon candies -/
axiom original_ratio : original_raspberry = 3 * original_lemon

/-- The condition that after giving away 15 raspberry candies and 5 lemon candies, 
    he has five times as many raspberry candies as lemon candies -/
axiom new_ratio : original_raspberry - 15 = 5 * (original_lemon - 5)

/-- The theorem stating that the original number of raspberry candies is 15 -/
theorem original_raspberry_count : original_raspberry = 15 := by sorry

end original_raspberry_count_l441_44197


namespace solve_system_l441_44192

theorem solve_system (x y : ℚ) (eq1 : 2 * x - 3 * y = 15) (eq2 : x + 2 * y = 8) : x = 54 / 7 := by
  sorry

end solve_system_l441_44192


namespace dog_park_problem_l441_44144

theorem dog_park_problem (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_eared_dogs : ℕ) :
  spotted_dogs = 15 →
  2 * spotted_dogs = total_dogs →
  5 * pointy_eared_dogs = total_dogs →
  pointy_eared_dogs = 6 := by
  sorry

end dog_park_problem_l441_44144


namespace stock_investment_l441_44140

theorem stock_investment (annual_income : ℝ) (stock_percentage : ℝ) (stock_price : ℝ) :
  annual_income = 2000 ∧ 
  stock_percentage = 40 ∧ 
  stock_price = 136 →
  ∃ amount_invested : ℝ, amount_invested = 6800 ∧
    annual_income = (amount_invested / stock_price) * (stock_percentage / 100) * 100 :=
by sorry

end stock_investment_l441_44140


namespace base7_321_equals_base10_162_l441_44174

def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base7_321_equals_base10_162 :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end base7_321_equals_base10_162_l441_44174


namespace inequality_equivalence_l441_44181

theorem inequality_equivalence (x : ℝ) : 
  (3 / (5 - 3 * x) > 1) ↔ (2 / 3 < x ∧ x < 5 / 3) := by
  sorry

end inequality_equivalence_l441_44181


namespace b_investment_value_l441_44191

/-- Calculates the investment of partner B in a partnership business --/
def calculate_b_investment (a_investment b_investment c_investment total_profit a_profit : ℚ) : Prop :=
  let total_investment := a_investment + b_investment + c_investment
  (a_investment / total_investment = a_profit / total_profit) ∧
  b_investment = 13650

/-- Theorem stating B's investment given the problem conditions --/
theorem b_investment_value :
  calculate_b_investment 6300 13650 10500 12500 3750 := by
  sorry

end b_investment_value_l441_44191


namespace solve_for_a_l441_44170

theorem solve_for_a : ∃ a : ℝ, 
  (2 : ℝ) - a * (1 : ℝ) = -1 ∧ a = 3 := by
  sorry

end solve_for_a_l441_44170


namespace repeating_decimal_eq_fraction_l441_44165

/-- The repeating decimal 0.6̄3 as a real number -/
def repeating_decimal : ℚ := 19/30

/-- Theorem stating that the repeating decimal 0.6̄3 is equal to 19/30 -/
theorem repeating_decimal_eq_fraction : repeating_decimal = 19/30 := by sorry

end repeating_decimal_eq_fraction_l441_44165


namespace composite_number_l441_44153

theorem composite_number (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 6 * 2^(2^(4*n)) + 1 = a * b := by
  sorry

end composite_number_l441_44153


namespace parabola_equation_l441_44123

/-- Given a parabola y^2 = 2px (p > 0) and a line with slope 1 passing through its focus,
    intersecting the parabola at points A and B, if |AB| = 8, then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) (h_p : p > 0) : 
  (∀ x y, y^2 = 2*p*x → (∃ t, y = x - p/2 + t)) →  -- Line passes through focus (p/2, 0) with slope 1
  (A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1) →            -- A and B are on the parabola
  (A.2 = A.1 - p/2 ∧ B.2 = B.1 - p/2) →            -- A and B are on the line
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 →             -- |AB|^2 = 8^2 = 64
  (∀ x y, y^2 = 4*x ↔ y^2 = 2*p*x) :=               -- The parabola equation is y^2 = 4x
by sorry

end parabola_equation_l441_44123


namespace round_trip_percentage_l441_44198

/-- The percentage of passengers with round-trip tickets, given the conditions -/
theorem round_trip_percentage (total_passengers : ℝ) 
  (h1 : 0 < total_passengers)
  (h2 : (0.2 : ℝ) * total_passengers = 
        (0.8 : ℝ) * (round_trip_passengers : ℝ)) : 
  round_trip_passengers / total_passengers = 0.25 := by
  sorry

#check round_trip_percentage

end round_trip_percentage_l441_44198


namespace solve_candy_problem_l441_44172

def candy_problem (debby_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) : Prop :=
  let total_candy := debby_candy + sister_candy
  let eaten_candy := total_candy - remaining_candy
  eaten_candy = 35

theorem solve_candy_problem :
  candy_problem 32 42 39 := by
  sorry

end solve_candy_problem_l441_44172


namespace complex_modulus_problem_l441_44154

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l441_44154


namespace joans_books_l441_44117

theorem joans_books (tom_books : ℕ) (total_books : ℕ) (h1 : tom_books = 38) (h2 : total_books = 48) :
  total_books - tom_books = 10 := by
sorry

end joans_books_l441_44117


namespace boat_speed_in_still_water_l441_44120

/-- The speed of a boat in still water, given its downstream speed and the current speed -/
theorem boat_speed_in_still_water
  (downstream_speed : ℝ) -- Speed of the boat downstream
  (current_speed : ℝ)    -- Speed of the current
  (h1 : downstream_speed = 36) -- Given downstream speed
  (h2 : current_speed = 6)     -- Given current speed
  : downstream_speed - current_speed = 30 := by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l441_44120


namespace linear_function_proof_l441_44150

/-- A linear function passing through (-2, 0) with the form y = ax + 1 -/
def linear_function (x : ℝ) : ℝ → ℝ := λ a ↦ a * x + 1

theorem linear_function_proof :
  ∃ a : ℝ, (∀ x : ℝ, linear_function x a = (1/2) * x + 1) ∧ linear_function (-2) a = 0 :=
by
  sorry

end linear_function_proof_l441_44150


namespace distinguishable_triangles_count_l441_44119

/-- The number of available colors for the triangles -/
def num_colors : ℕ := 7

/-- A large triangle is made up of this many smaller triangles -/
def triangles_per_large : ℕ := 4

/-- The number of corner triangles in a large triangle -/
def num_corners : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  let corner_same := num_colors -- All corners same color
  let corner_two_same := num_colors * (num_colors - 1) -- Two corners same, one different
  let corner_all_diff := choose num_colors num_corners -- All corners different
  let total_corner_combinations := corner_same + corner_two_same + corner_all_diff
  total_corner_combinations * num_colors -- Multiply by center triangle color choices

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 588 :=
sorry

end distinguishable_triangles_count_l441_44119


namespace identical_balls_distribution_seven_balls_four_boxes_l441_44141

theorem identical_balls_distribution (n m : ℕ) (hn : n ≥ m) :
  (Nat.choose (n + m - 1) (m - 1) : ℕ) = (Nat.choose (n - 1) (m - 1) : ℕ) := by
  sorry

theorem seven_balls_four_boxes :
  (Nat.choose 6 3 : ℕ) = 20 := by
  sorry

end identical_balls_distribution_seven_balls_four_boxes_l441_44141


namespace rachel_bought_three_tables_l441_44156

/-- Represents the number of minutes spent on each piece of furniture -/
def time_per_furniture : ℕ := 4

/-- Represents the total number of chairs bought -/
def num_chairs : ℕ := 7

/-- Represents the total time spent assembling all furniture -/
def total_time : ℕ := 40

/-- Calculates the number of tables bought -/
def num_tables : ℕ :=
  (total_time - time_per_furniture * num_chairs) / time_per_furniture

theorem rachel_bought_three_tables :
  num_tables = 3 :=
sorry

end rachel_bought_three_tables_l441_44156


namespace intersecting_rectangles_area_l441_44176

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.length

/-- Represents the overlap between two rectangles -/
structure Overlap where
  width : ℕ
  length : ℕ

/-- Calculates the area of overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.length

theorem intersecting_rectangles_area (r1 r2 r3 : Rectangle) 
  (o12 o13 o23 o123 : Overlap) : 
  r1.width = 4 → r1.length = 12 →
  r2.width = 5 → r2.length = 10 →
  r3.width = 3 → r3.length = 6 →
  o12.width = 4 → o12.length = 5 →
  o13.width = 3 → o13.length = 4 →
  o23.width = 3 → o23.length = 3 →
  o123.width = 3 → o123.length = 3 →
  area r1 + area r2 + area r3 - (overlapArea o12 + overlapArea o13 + overlapArea o23) + overlapArea o123 = 84 := by
  sorry

end intersecting_rectangles_area_l441_44176


namespace decimal_addition_l441_44161

theorem decimal_addition : 1 + 0.01 + 0.0001 = 1.0101 := by
  sorry

end decimal_addition_l441_44161


namespace tile_arrangements_l441_44167

def num_red_tiles : ℕ := 1
def num_blue_tiles : ℕ := 2
def num_green_tiles : ℕ := 2
def num_yellow_tiles : ℕ := 4

def total_tiles : ℕ := num_red_tiles + num_blue_tiles + num_green_tiles + num_yellow_tiles

theorem tile_arrangements :
  (total_tiles.factorial) / (num_red_tiles.factorial * num_blue_tiles.factorial * num_green_tiles.factorial * num_yellow_tiles.factorial) = 3780 :=
by sorry

end tile_arrangements_l441_44167


namespace geometric_progression_sum_not_end_20_l441_44163

/-- Given a, b, c form a geometric progression, prove that a^3 + b^3 + c^3 - 3abc cannot end with 20 -/
theorem geometric_progression_sum_not_end_20 
  (a b c : ℤ) 
  (h_geom : ∃ (q : ℚ), b = a * q ∧ c = b * q) : 
  ¬ (∃ (k : ℤ), a^3 + b^3 + c^3 - 3*a*b*c = 100*k + 20) := by
sorry

end geometric_progression_sum_not_end_20_l441_44163


namespace quadratic_equation_nonnegative_integer_solutions_l441_44100

theorem quadratic_equation_nonnegative_integer_solutions :
  ∃! (x : ℕ), x^2 + x - 6 = 0 :=
by sorry

end quadratic_equation_nonnegative_integer_solutions_l441_44100


namespace purely_imaginary_complex_number_l441_44118

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m - 1) * Complex.I + (m^2 - 1) : ℂ).re = 0 ∧ ((m - 1) * Complex.I + (m^2 - 1) : ℂ).im ≠ 0) → 
  m = -1 := by
  sorry

end purely_imaginary_complex_number_l441_44118


namespace quadratic_coefficient_l441_44168

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 1)^2 - 2) →
  QuadraticFunction a b c 3 = 7 →
  a = 3 := by sorry

end quadratic_coefficient_l441_44168


namespace chicken_count_l441_44152

theorem chicken_count (C : ℚ) 
  (roosters : ℚ → ℚ) (hens : ℚ → ℚ) (laying_hens : ℚ → ℚ) (non_laying : ℚ) :
  roosters C = (1 / 4) * C →
  hens C = (3 / 4) * C →
  laying_hens C = (3 / 4) * hens C →
  roosters C + (hens C - laying_hens C) = 35 →
  C = 80 := by
sorry

end chicken_count_l441_44152


namespace impossible_sequence_is_invalid_l441_44146

/-- Represents a sequence of letters --/
def Sequence := List Nat

/-- Checks if a sequence is valid according to the letter printing process --/
def is_valid_sequence (s : Sequence) : Prop :=
  ∀ i j, i < j → (s.indexOf i < s.indexOf j → ∀ k, i < k ∧ k < j → s.indexOf k < s.indexOf j)

/-- The impossible sequence --/
def impossible_sequence : Sequence := [4, 5, 2, 3, 1]

/-- Theorem stating that the impossible sequence is indeed impossible --/
theorem impossible_sequence_is_invalid : 
  ¬ is_valid_sequence impossible_sequence := by sorry

end impossible_sequence_is_invalid_l441_44146


namespace total_cost_of_kept_shirts_l441_44190

def all_shirts : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def returned_shirts : List ℕ := [20, 25, 30, 22, 23, 29]

theorem total_cost_of_kept_shirts :
  (all_shirts.sum - returned_shirts.sum) = 85 := by
  sorry

end total_cost_of_kept_shirts_l441_44190


namespace max_value_of_quadratic_l441_44138

open Real

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 1) : 
  ∃ (max_val : ℝ), max_val = 1/4 ∧ ∀ y, 0 < y ∧ y < 1 → y * (1 - y) ≤ max_val :=
by sorry

end max_value_of_quadratic_l441_44138


namespace cube_opposite_face_l441_44196

-- Define the faces of the cube
inductive Face : Type
| X | Y | Z | U | V | W

-- Define the adjacency relation
def adjacent : Face → Face → Prop := sorry

-- Define the opposite relation
def opposite : Face → Face → Prop := sorry

-- State the theorem
theorem cube_opposite_face :
  (∀ f : Face, f ≠ Face.X ∧ f ≠ Face.Y → adjacent Face.X f) →
  opposite Face.X Face.Y := by sorry

end cube_opposite_face_l441_44196


namespace alpha_range_l441_44115

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ π) 
  (h3 : ∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) :
  α ∈ Set.Icc 0 (π / 6) ∪ Set.Icc (5 * π / 6) π := by
  sorry

end alpha_range_l441_44115


namespace fourth_root_of_sum_of_powers_of_two_l441_44125

theorem fourth_root_of_sum_of_powers_of_two :
  (2^3 + 2^4 + 2^5 + 2^6 : ℝ)^(1/4) = 2^(3/4) * 15^(1/4) := by
  sorry

end fourth_root_of_sum_of_powers_of_two_l441_44125


namespace hyperbola_foci_coordinates_l441_44103

/-- Given a hyperbola passing through a specific point, prove the coordinates of its foci -/
theorem hyperbola_foci_coordinates :
  ∀ (a : ℝ),
  (((2 * Real.sqrt 2) ^ 2) / a ^ 2) - 1 ^ 2 = 1 →
  ∃ (c : ℝ),
  c ^ 2 = 5 ∧
  (∀ (x y : ℝ), x ^ 2 / a ^ 2 - y ^ 2 = 1 → 
    ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
by sorry

end hyperbola_foci_coordinates_l441_44103


namespace sasha_took_right_triangle_l441_44110

-- Define the triangle types
inductive TriangleType
  | Acute
  | Right
  | Obtuse

-- Define a function to check if two triangles can form the third
def canFormThird (t1 t2 t3 : TriangleType) : Prop :=
  (t1 ≠ t2) ∧ (t2 ≠ t3) ∧ (t1 ≠ t3) ∧
  ((t1 = TriangleType.Acute ∧ t2 = TriangleType.Obtuse) ∨
   (t1 = TriangleType.Obtuse ∧ t2 = TriangleType.Acute)) ∧
  t3 = TriangleType.Right

-- Theorem statement
theorem sasha_took_right_triangle (t1 t2 t3 : TriangleType) :
  (t1 ≠ t2) ∧ (t2 ≠ t3) ∧ (t1 ≠ t3) →
  canFormThird t1 t2 t3 →
  t3 = TriangleType.Right :=
by sorry

end sasha_took_right_triangle_l441_44110


namespace mohamed_age_ratio_l441_44105

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Represents the current year -/
def currentYear : ℕ := 2023

theorem mohamed_age_ratio (kody : Age) (mohamed : Age) :
  kody.value = 32 →
  (currentYear - 4 : ℕ) - kody.value + 4 = 2 * ((currentYear - 4 : ℕ) - mohamed.value + 4) →
  ∃ k : ℕ, mohamed.value = 30 * k →
  mohamed.value / 30 = 2 := by
  sorry

end mohamed_age_ratio_l441_44105


namespace q_satisfies_conditions_l441_44186

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (6/7) * x^2 - (2/7) * x + 2

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-2) = 6 ∧ q 0 = 2 ∧ q 3 = 8 := by
  sorry

#eval q (-2)
#eval q 0
#eval q 3

end q_satisfies_conditions_l441_44186


namespace debate_team_boys_l441_44111

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) (boys : ℕ) : 
  girls = 4 → 
  groups = 8 → 
  group_size = 4 → 
  total = groups * group_size → 
  boys = total - girls → 
  boys = 28 := by
sorry

end debate_team_boys_l441_44111


namespace triangle_centroid_property_l441_44199

variable (A B C G : ℝ × ℝ)

def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def distance_squared (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem triangle_centroid_property (h_centroid : is_centroid G A B C)
  (h_condition : distance_squared G A + 2 * distance_squared G B + 3 * distance_squared G C = 123) :
  distance_squared A B + distance_squared A C + distance_squared B C = 246 := by
  sorry

end triangle_centroid_property_l441_44199


namespace sqrt_108_simplification_l441_44104

theorem sqrt_108_simplification : Real.sqrt 108 = 6 * Real.sqrt 3 := by
  sorry

end sqrt_108_simplification_l441_44104


namespace tile_border_ratio_l441_44133

theorem tile_border_ratio :
  ∀ (n s d : ℝ),
  n > 0 →
  s > 0 →
  d > 0 →
  n = 24 →
  (24 * s)^2 / (24 * s + 25 * d)^2 = 64 / 100 →
  d / s = 6 / 25 := by
sorry

end tile_border_ratio_l441_44133


namespace complex_solutions_count_l441_44134

/-- The equation (z^3 - 1) / (z^2 + z - 6) = 0 has exactly 3 complex solutions. -/
theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 1) / (z^2 + z - 6) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 1) / (z^2 + z - 6) = 0 → z ∈ S) ∧
  Finset.card S = 3 := by
  sorry

end complex_solutions_count_l441_44134


namespace max_product_sum_2000_l441_44139

theorem max_product_sum_2000 :
  ∃ (a b : ℤ), a + b = 2000 ∧
  ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b ∧
  a * b = 1000000 := by
  sorry

end max_product_sum_2000_l441_44139


namespace trailing_zeros_of_power_sum_l441_44147

theorem trailing_zeros_of_power_sum : ∃ n : ℕ, n > 0 ∧ 
  (4^(5^6) + 6^(5^4) : ℕ) % (10^n) = 0 ∧ 
  (4^(5^6) + 6^(5^4) : ℕ) % (10^(n+1)) ≠ 0 ∧ 
  n = 5 := by sorry

end trailing_zeros_of_power_sum_l441_44147


namespace multiply_mixed_number_l441_44173

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end multiply_mixed_number_l441_44173


namespace constant_value_proof_l441_44158

theorem constant_value_proof (t : ℝ) (constant : ℝ) : 
  let x := 1 - 3 * t
  let y := constant * t - 3
  (t = 0.8 → x = y) → constant = 2 := by
sorry

end constant_value_proof_l441_44158


namespace sheila_fewer_acorns_l441_44142

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
def total_acorns : ℕ := 80

/-- The number of acorns Shawna has -/
def shawna_acorns : ℕ := 7

/-- The ratio of Sheila's acorns to Shawna's acorns -/
def sheila_ratio : ℕ := 5

/-- The number of acorns Sheila has -/
def sheila_acorns : ℕ := sheila_ratio * shawna_acorns

/-- The number of acorns Danny has -/
def danny_acorns : ℕ := total_acorns - sheila_acorns - shawna_acorns

/-- The difference in acorns between Danny and Sheila -/
def acorn_difference : ℕ := danny_acorns - sheila_acorns

theorem sheila_fewer_acorns : acorn_difference = 3 := by
  sorry

end sheila_fewer_acorns_l441_44142


namespace lawnmower_value_drop_l441_44131

theorem lawnmower_value_drop (initial_price : ℝ) (first_drop_percent : ℝ) (final_value : ℝ) :
  initial_price = 100 →
  first_drop_percent = 25 →
  final_value = 60 →
  let value_after_six_months := initial_price * (1 - first_drop_percent / 100)
  let drop_over_next_year := value_after_six_months - final_value
  let drop_percent_next_year := (drop_over_next_year / value_after_six_months) * 100
  drop_percent_next_year = 20 := by
  sorry

end lawnmower_value_drop_l441_44131


namespace remainder_of_2857916_div_4_l441_44136

theorem remainder_of_2857916_div_4 : 2857916 % 4 = 0 := by
  sorry

end remainder_of_2857916_div_4_l441_44136


namespace third_sample_is_51_l441_44166

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalItems : Nat
  numGroups : Nat
  firstSample : Nat

/-- Calculates the sample for a given group in a systematic sampling -/
def getSample (s : SystematicSampling) (group : Nat) : Nat :=
  s.firstSample + (group - 1) * (s.totalItems / s.numGroups)

/-- Theorem: In a systematic sampling of 400 items into 20 groups, 
    if the first sample is 11, then the third sample will be 51 -/
theorem third_sample_is_51 (s : SystematicSampling) 
  (h1 : s.totalItems = 400) 
  (h2 : s.numGroups = 20) 
  (h3 : s.firstSample = 11) : 
  getSample s 3 = 51 := by
  sorry

/-- Example setup for the given problem -/
def exampleSampling : SystematicSampling := {
  totalItems := 400
  numGroups := 20
  firstSample := 11
}

#eval getSample exampleSampling 3

end third_sample_is_51_l441_44166


namespace square_area_on_parabola_prove_square_area_l441_44130

theorem square_area_on_parabola : ℝ → Prop :=
  fun area =>
    ∃ (x₁ x₂ : ℝ),
      -- The endpoints lie on the parabola
      x₁^2 + 4*x₁ + 3 = 6 ∧
      x₂^2 + 4*x₂ + 3 = 6 ∧
      -- The side length is the distance between x-coordinates
      (x₂ - x₁)^2 = area ∧
      -- The area is 28
      area = 28

theorem prove_square_area : square_area_on_parabola 28 := by
  sorry

end square_area_on_parabola_prove_square_area_l441_44130


namespace range_of_a_l441_44113

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 :=
by sorry

end range_of_a_l441_44113


namespace small_paintings_sold_l441_44169

/-- Given the prices of paintings and sales information, prove the number of small paintings sold. -/
theorem small_paintings_sold
  (large_price : ℕ)
  (small_price : ℕ)
  (large_sold : ℕ)
  (total_earnings : ℕ)
  (h1 : large_price = 100)
  (h2 : small_price = 80)
  (h3 : large_sold = 5)
  (h4 : total_earnings = 1140) :
  (total_earnings - large_price * large_sold) / small_price = 8 := by
  sorry

end small_paintings_sold_l441_44169


namespace correct_operation_l441_44129

theorem correct_operation (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end correct_operation_l441_44129


namespace problem_solution_l441_44126

def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * x

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≥ 3 ↔ x ≥ 0) ∧
  (∀ x : ℝ, (f a x ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6)) :=
by sorry

end problem_solution_l441_44126


namespace no_good_points_iff_a_in_range_l441_44137

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 1

def has_no_good_points (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≠ x

theorem no_good_points_iff_a_in_range :
  ∀ a : ℝ, has_no_good_points a ↔ -1/2 < a ∧ a < 3/2 := by
  sorry

end no_good_points_iff_a_in_range_l441_44137


namespace arithmetic_sequence_sum_l441_44112

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 5 + a 6 + a 7 = 1) →
  (a 3 + a 9 = 2/3) := by
  sorry

end arithmetic_sequence_sum_l441_44112


namespace min_value_trig_expression_l441_44194

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2 ≥ 100 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 10)^2 = 100 :=
by sorry

end min_value_trig_expression_l441_44194


namespace unknown_blanket_rate_l441_44101

/-- Proves that given the specified blanket purchases and average price, 
    the unknown rate for two blankets must be 275. -/
theorem unknown_blanket_rate 
  (price1 : ℕ) (count1 : ℕ) 
  (price2 : ℕ) (count2 : ℕ) 
  (count3 : ℕ) 
  (avg_price : ℕ) 
  (h1 : price1 = 100) 
  (h2 : count1 = 3) 
  (h3 : price2 = 150) 
  (h4 : count2 = 5) 
  (h5 : count3 = 2) 
  (h6 : avg_price = 160) 
  (h7 : (price1 * count1 + price2 * count2 + count3 * unknown_rate) / (count1 + count2 + count3) = avg_price) : 
  unknown_rate = 275 := by
  sorry

#check unknown_blanket_rate

end unknown_blanket_rate_l441_44101


namespace exam_mean_score_l441_44185

/-- Given an exam where a score of 86 is 7 standard deviations below the mean,
    and a score of 90 is 3 standard deviations above the mean,
    prove that the mean score is 88.8 -/
theorem exam_mean_score (μ σ : ℝ) 
    (h1 : 86 = μ - 7 * σ) 
    (h2 : 90 = μ + 3 * σ) : 
  μ = 88.8 := by
sorry

end exam_mean_score_l441_44185


namespace jacket_final_price_l441_44183

/-- Calculates the final price of an item after two discounts and a tax --/
def finalPrice (originalPrice firstDiscount secondDiscount taxRate : ℝ) : ℝ :=
  let priceAfterFirstDiscount := originalPrice * (1 - firstDiscount)
  let priceAfterSecondDiscount := priceAfterFirstDiscount * (1 - secondDiscount)
  priceAfterSecondDiscount * (1 + taxRate)

/-- Theorem stating that the final price of the jacket is approximately $77.11 --/
theorem jacket_final_price :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (finalPrice 120 0.3 0.15 0.08 - 77.11) < ε :=
sorry

end jacket_final_price_l441_44183


namespace dress_price_difference_l441_44195

theorem dress_price_difference (original_price : ℝ) : 
  (0.85 * original_price = 85) →
  (original_price - (85 + 0.25 * 85) = -6.25) := by
sorry

end dress_price_difference_l441_44195


namespace power_division_equality_l441_44114

theorem power_division_equality : (3 : ℕ)^15 / (27 : ℕ)^3 = 729 := by sorry

end power_division_equality_l441_44114


namespace complement_A_inter_B_when_m_3_A_inter_B_empty_iff_l441_44116

/-- The set A defined as {x | -1 ≤ x < 4} -/
def A : Set ℝ := {x | -1 ≤ x ∧ x < 4}

/-- The set B defined as {x | m ≤ x ≤ m+2} for a real number m -/
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

/-- Part 1: The complement of A ∩ B when m = 3 -/
theorem complement_A_inter_B_when_m_3 :
  (A ∩ B 3)ᶜ = {x | x < 3 ∨ x ≥ 4} := by sorry

/-- Part 2: Characterization of m when A ∩ B is empty -/
theorem A_inter_B_empty_iff (m : ℝ) :
  A ∩ B m = ∅ ↔ m < -3 ∨ m ≥ 4 := by sorry

end complement_A_inter_B_when_m_3_A_inter_B_empty_iff_l441_44116


namespace min_n_for_sum_greater_than_1020_l441_44164

def sequence_term (n : ℕ) : ℕ := 2^n - 1

def sequence_sum (n : ℕ) : ℕ := 2^(n+1) - 2 - n

theorem min_n_for_sum_greater_than_1020 :
  (∀ k < 10, sequence_sum k ≤ 1020) ∧ (sequence_sum 10 > 1020) := by sorry

end min_n_for_sum_greater_than_1020_l441_44164


namespace complex_fraction_simplification_l441_44128

theorem complex_fraction_simplification : 
  (((12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500)) / 
   ((6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500))) = -182 := by
  sorry

end complex_fraction_simplification_l441_44128


namespace tim_prank_combinations_l441_44151

theorem tim_prank_combinations :
  let day1_choices : ℕ := 1
  let day2_choices : ℕ := 2
  let day3_choices : ℕ := 6
  let day4_choices : ℕ := 5
  let day5_choices : ℕ := 1
  day1_choices * day2_choices * day3_choices * day4_choices * day5_choices = 60 :=
by sorry

end tim_prank_combinations_l441_44151


namespace xy_positive_sufficient_not_necessary_l441_44132

theorem xy_positive_sufficient_not_necessary (x y : ℝ) :
  (x * y > 0 → |x + y| = |x| + |y|) ∧
  ¬(∀ x y : ℝ, |x + y| = |x| + |y| → x * y > 0) :=
by sorry

end xy_positive_sufficient_not_necessary_l441_44132


namespace triangle_area_l441_44148

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side a
  c : ℝ  -- Side c

-- Define the conditions of the problem
def problem_triangle : Triangle where
  A := sorry
  B := sorry
  C := sorry
  a := 2
  c := 5

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

-- Define the angle sum property
def angle_sum (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h1 : is_arithmetic_sequence t) 
  (h2 : angle_sum t) 
  (h3 : t.a = 2) 
  (h4 : t.c = 5) : 
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = (5 * Real.sqrt 3) / 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end triangle_area_l441_44148


namespace inequality_solution_set_l441_44171

theorem inequality_solution_set :
  {x : ℝ | 1 + x > 6 - 4 * x} = {x : ℝ | x > 1} := by sorry

end inequality_solution_set_l441_44171
